# 今日论文推荐 - 2025-10-24

共 71 篇论文

---

## 1. A Scalable, Causal, and Energy Efficient Framework for Neural Decoding with Spiking Neural Networks

**论文链接:** [http://arxiv.org/abs/2510.20683v1](http://arxiv.org/abs/2510.20683v1)

**作者:** Georgios Mentzelopoulos, Ioannis Asmanis, Konrad P. Kording, Eva L. Dyer, Kostas Daniilidis, Flavia Vitale

**发布时间:** 2025-10-23

### GPT解析

### 总结

Spikachu是一种基于脉冲神经网络的脑机接口解码框架，具有可扩展性、因果性和高能效性，解决了现有方法在实时应用和能源效率方面的限制。

### 背景

脑机接口对神经运动障碍人群具有重要意义，但现有解码方法要么简单但缺乏泛化能力，要么复杂但难以实时应用，且都依赖于高能耗的人工神经网络，难以集成到资源有限的现实系统中。

### 目的

开发一种基于脉冲神经网络的、可扩展、因果且节能的神经解码框架，以克服现有方法的局限性。

### 方法

Spikachu直接处理分箱脉冲，将其投影到共享潜在空间，利用适应输入时序的脉冲模块提取特征，然后整合这些潜在表示并解码生成行为预测。研究在6只非人灵长类动物的113个记录会话（总计43小时）上评估该方法。

### 主要发现

与因果基线相比，Spikachu在使用少2.26到418.81倍能源的情况下表现更好；将训练扩展到多个会话和受试者可提高性能，并实现向未见过的会话、受试者和任务的少样本迁移。

### 结论

Spikachu引入了一种基于SNN的可扩展、在线兼容的神经解码框架，其性能与最先进模型相当，同时能耗低几个数量级。

### 翻译

脑机接口(BCIs)有望为神经运动障碍个体实现言语和假肢控制等关键功能。其成功的关键是神经解码器，即将神经活动映射到预期行为的模型。当前基于学习的解码方法分为两类：简单但缺乏泛化能力的因果模型，或复杂但难以实时应用的非因果模型。两者都面临一个共同挑战，它们依赖于能耗高的人工神经网络骨干，这使得集成到现实世界的资源有限系统中变得困难。脉冲神经网络(SNNs)提供了一种有前景的替代方案。由于它们以因果方式运行，这些模型适合实时使用，并且它们的低能耗需求使其成为电池受限环境的理想选择。为此，我们引入了Spikachu：一种基于SNN的可扩展、因果且节能的神经解码框架。我们的方法通过将分箱脉冲直接投影到共享潜在空间来处理它们，在该空间中，适应输入时序的脉冲模块提取相关特征；然后将这些潜在表示整合和解码以生成行为预测。我们在6只非人灵长类动物的113个记录会话上评估了我们的方法，总计43小时的记录。与因果基线相比，我们的方法在使用少2.26到418.81倍能源的情况下表现更好。此外，我们证明将训练扩展到多个会话和受试者可以提高性能，并实现向未见过的会话、受试者和任务的少样本迁移。总体而言，Spikachu引入了一种基于SNN的可扩展、在线兼容的神经解码框架，其性能与最先进模型相当，同时能耗低几个数量级。


### 论文摘要

Brain-computer interfaces (BCIs) promise to enable vital functions, such as speech and prosthetic control, for individuals with neuromotor impairments. Central to their success are neural decoders, models that map neural activity to intended behavior. Current learning-based decoding approaches fall into two classes: simple, causal models that lack generalization, or complex, non-causal models that generalize and scale offline but struggle in real-time settings. Both face a common challenge, their reliance on power-hungry artificial neural network backbones, which makes integration into real-world, resource-limited systems difficult. Spiking neural networks (SNNs) offer a promising alternative. Because they operate causally these models are suitable for real-time use, and their low energy demands make them ideal for battery-constrained environments. To this end, we introduce Spikachu: a scalable, causal, and energy-efficient neural decoding framework based on SNNs. Our approach processes binned spikes directly by projecting them into a shared latent space, where spiking modules, adapted to the timing of the input, extract relevant features; these latent representations are then integrated and decoded to generate behavioral predictions. We evaluate our approach on 113 recording sessions from 6 non-human primates, totaling 43 hours of recordings. Our method outperforms causal baselines when trained on single sessions using between 2.26 and 418.81 times less energy. Furthermore, we demonstrate that scaling up training to multiple sessions and subjects improves performance and enables few-shot transfer to unseen sessions, subjects, and tasks. Overall, Spikachu introduces a scalable, online-compatible neural decoding framework based on SNNs, whose performance is competitive relative to state-of-the-art models while consuming orders of magnitude less energy.

---

## 2. Monocular Visual 8D Pose Estimation for Articulated Bicycles and Cyclists

**论文链接:** [http://arxiv.org/abs/2510.20158v1](http://arxiv.org/abs/2510.20158v1)

**作者:** Eduardo R. Corral-Soto, Yang Liu, Yuan Ren, Bai Dongfeng, Liu Bingbing

**发布时间:** 2025-10-23

### GPT解析

### 总结

本文提出了一种针对关节式自行车和骑行者的类别级8D姿态估计方法，能够从单张RGB图像估计自行车的3D平移、旋转以及转向手柄和踏板相对于车身的旋转，从而提供更细粒度的自行车姿态状态和行驶方向估计。

### 背景

在自动驾驶中，骑行者属于安全关键类弱势道路使用者(VRU)，准确估计其姿态对过马路意图分类、行为预测和碰撞避免至关重要。与刚性物体不同，关节式自行车由通过关节连接的可移动刚性部件组成，6D姿态方法在自行车转向/踏板角度变化时变得不足。

### 目的

开发一种能够估计自行车完整关节状态的方法，包括3D位置、旋转以及转向手柄和踏板的相对旋转，以提供更准确的骑行者行为预测和碰撞避免能力。

### 方法

提出联合估计关节式自行车8D姿态和3D关键点的模型，使用合成和真实图像数据的混合进行训练，以在真实图像上实现良好的泛化能力。

### 主要发现

所提出的8D姿态估计方法能够提供更细粒度的自行车姿态状态和行驶方向估计，与使用刚性规范对象模板的最先进6D姿态估计器相比具有竞争力。

### 结论

该方法在处理关节式自行车姿态变化方面表现出色，能够更好地理解骑行者的实际行驶意图，为自动驾驶系统提供更可靠的骑行者行为预测能力。

### 翻译

在自动驾驶中，骑行者属于安全关键类弱势道路使用者(VRU)，准确估计他们的姿态对于骑行者过马路意图分类、行为预测和碰撞避免至关重要。与刚性物体不同，关节式自行车由通过关节连接的可移动刚性部件组成，并受运动学结构约束。6D姿态方法可以估计刚性自行车的3D旋转和平移，但当自行车的转向/踏板角度变化时，6D方法变得不足。这是因为：1)自行车关节姿态的变化会导致其3D边界框也发生变化，2)3D框的方向不一定与决定实际预期行驶方向的转向方向对齐。在这项工作中，我们介绍了一种针对关节式自行车和骑行者的类别级8D姿态估计方法，可以从单张RGB图像进行估计。除了能够从单张图像估计自行车的3D平移和旋转外，我们的方法还估计其转向手柄和踏板相对于自行车车身的旋转。这两个新参数能够估计更细粒度的自行车姿态状态和行驶方向。我们提出的模型联合估计关节式自行车的8D姿态和3D关键点，并使用合成和真实图像数据的混合进行训练，以在真实图像上泛化。我们包含了一个评估部分，评估了估计的8D姿态参数的准确性，与使用刚性规范对象模板进行匹配的最先进类别级6D姿态估计器相比，我们的方法取得了具有竞争力的分数。


### 论文摘要

In Autonomous Driving, cyclists belong to the safety-critical class of Vulnerable Road Users (VRU), and accurate estimation of their pose is critical for cyclist crossing intention classification, behavior prediction, and collision avoidance. Unlike rigid objects, articulated bicycles are composed of movable rigid parts linked by joints and constrained by a kinematic structure. 6D pose methods can estimate the 3D rotation and translation of rigid bicycles, but 6D becomes insufficient when the steering/pedals angles of the bicycle vary. That is because: 1) varying the articulated pose of the bicycle causes its 3D bounding box to vary as well, and 2) the 3D box orientation is not necessarily aligned to the orientation of the steering which determines the actual intended travel direction. In this work, we introduce a method for category-level 8D pose estimation for articulated bicycles and cyclists from a single RGB image. Besides being able to estimate the 3D translation and rotation of a bicycle from a single image, our method also estimates the rotations of its steering handles and pedals with respect to the bicycle body frame. These two new parameters enable the estimation of a more fine-grained bicycle pose state and travel direction. Our proposed model jointly estimates the 8D pose and the 3D Keypoints of articulated bicycles, and trains with a mix of synthetic and real image data to generalize on real images. We include an evaluation section where we evaluate the accuracy of our estimated 8D pose parameters, and our method shows promising results by achieving competitive scores when compared against state-of-the-art category-level 6D pose estimators that use rigid canonical object templates for matching.

---

## 3. OmniMotion-X: Versatile Multimodal Whole-Body Motion Generation

**论文链接:** [http://arxiv.org/abs/2510.19789v1](http://arxiv.org/abs/2510.19789v1)

**作者:** Guowei Xu, Yuxuan Bian, Ailing Zeng, Mingyi Shi, Shaoli Huang, Wen Li, Lixin Duan, Qiang Xu

**发布时间:** 2025-10-22

### GPT解析

### 总结

本文介绍了OmniMotion-X，一个用于全身人体运动生成的多模态框架，采用自回归扩散变换器以统一的序列到序列方式工作。该框架支持多种多模态任务，包括文本到运动、音乐到舞蹈、语音到手势以及全局时空控制场景，并能灵活组合这些任务。

### 背景

人体运动生成领域需要能够处理多种模态输入并生成连贯、可控运动的系统。现有方法在处理多模态任务组合和保持生成内容一致性方面存在挑战。

### 目的

开发一个统一的多模态框架，能够处理多种运动生成任务，提高生成内容的一致性、风格和时序动态，并实现长时间的真实、连贯、可控运动生成。

### 方法

使用自回归扩散变换器作为核心架构，引入参考运动作为条件信号增强一致性，采用渐进式弱到强混合条件训练策略处理多模态冲突，并构建了OmniMoCap-X数据集，整合28个公开MoCap来源，使用GPT-4o自动生成结构化字幕。

### 主要发现

OmniMotion-X在多个多模态任务上显著超越现有方法，实现了最先进的性能，能够生成交互式的真实、连贯、可控的长时间运动。

### 结论

OmniMotion-X通过统一的多模态框架和创新的条件训练策略，解决了人体运动生成中的多模态整合问题，为动画制作和人机交互提供了有力工具。

### 翻译

本文介绍了OmniMotion-X，一个用于全身人体运动生成的多模态框架，利用自回归扩散变换器以统一的序列到序列方式工作。OmniMotion-X有效支持多种多模态任务，包括文本到运动、音乐到舞蹈、语音到手势和全局时空控制场景（如运动预测、中间帧生成、补全和关节/轨迹引导合成），以及这些任务的灵活组合。具体而言，我们提出使用参考运动作为新的条件信号，显著提高了生成内容、风格和时序动态的一致性，这对真实动画至关重要。为处理多模态冲突，我们引入了渐进式弱到强混合条件训练策略。为支持高质量多模态训练，我们构建了OmniMoCap-X，这是迄今为止最大的统一多模态运动数据集，整合了10个不同任务中的28个公开MoCap来源，标准化为30fps的SMPL-X格式。为确保详细且一致的标注，我们将序列渲染为视频并使用GPT-4o自动生成结构化和层次化字幕，捕捉低级行动和高层语义。大量实验评估证实，OmniMotion-X显著超越现有方法，在多个多模态任务上展示了最先进的性能，并能实现真实、连贯、可控的长时间运动的交互式生成。


### 论文摘要

This paper introduces OmniMotion-X, a versatile multimodal framework for whole-body human motion generation, leveraging an autoregressive diffusion transformer in a unified sequence-to-sequence manner. OmniMotion-X efficiently supports diverse multimodal tasks, including text-to-motion, music-to-dance, speech-to-gesture, and global spatial-temporal control scenarios (e.g., motion prediction, in-betweening, completion, and joint/trajectory-guided synthesis), as well as flexible combinations of these tasks. Specifically, we propose the use of reference motion as a novel conditioning signal, substantially enhancing the consistency of generated content, style, and temporal dynamics crucial for realistic animations. To handle multimodal conflicts, we introduce a progressive weak-to-strong mixed-condition training strategy. To enable high-quality multimodal training, we construct OmniMoCap-X, the largest unified multimodal motion dataset to date, integrating 28 publicly available MoCap sources across 10 distinct tasks, standardized to the SMPL-X format at 30 fps. To ensure detailed and consistent annotations, we render sequences into videos and use GPT-4o to automatically generate structured and hierarchical captions, capturing both low-level actions and high-level semantics. Extensive experimental evaluations confirm that OmniMotion-X significantly surpasses existing methods, demonstrating state-of-the-art performance across multiple multimodal tasks and enabling the interactive generation of realistic, coherent, and controllable long-duration motions.

---

## 4. ProTerrain: Probabilistic Physics-Informed Rough Terrain World Modeling

**论文链接:** [http://arxiv.org/abs/2510.19364v1](http://arxiv.org/abs/2510.19364v1)

**作者:** Golnaz Raja, Ruslan Agishev, Miloš Prágr, Joni Pajarinen, Karel Zimmermann, Arun Kumar Singh, Reza Ghabcheloo

**发布时间:** 2025-10-22

**备注:** This paper is submitted to IEEE International Conference on Robotics  and Automation (ICRA) 2026

### GPT解析

### 总结

该研究提出了一种有效的概率框架，用于处理非结构化越野环境中机器人运动预测的不确定性，通过建模地形参数的空间相关偶然不确定性并传播到轨迹预测中，显著提高了预测的准确性。

### 背景

不确定性感知的机器人运动预测对于非结构化越野环境中的下游可通行性估计和自主导航至关重要，因为在这种环境中地形是异构的且感知不确定性很高。

### 目的

引入一个有效的概率框架，明确地对地形参数的空间相关偶然不确定性建模为概率世界模型，并通过可微分物理引擎传播这种不确定性，实现概率轨迹预测。

### 方法

利用结构化卷积算子，提供高分辨率多变量预测，同时保持可管理的计算成本。

### 主要发现

在公开数据集上的实验评估显示，与偶然不确定性估计基线相比，该方法的不确定性估计和轨迹预测准确性显著提高。

### 结论

通过明确建模和传播空间相关的偶然不确定性，该方法能够提供更可靠的机器人轨迹预测，适用于复杂的越野环境。

### 翻译

不确定性感知的机器人运动预测对于非结构化越野环境中的下游可通行性估计和安全自主导航至关重要，在这种环境中地形是异构的且感知不确定性很高。大多数现有方法假设确定性或空间独立的地面不确定性，忽略了3D空间数据的固有局部相关性，并且通常产生不可靠的预测。在这项工作中，我们引入了一个有效的概率框架，明确地将地形参数的空间相关偶然不确定性建模为概率世界模型，并通过可微分物理引擎传播这种不确定性以实现概率轨迹预测。通过利用结构化卷积算子，我们的方法在可管理的计算成本下提供了高分辨率的多变量预测。在公开可用数据集上的实验评估表明，与偶然不确定性估计基线相比，不确定性估计和轨迹预测准确性显著提高。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决机器人越野导航中不确定性感知的运动预测问题。具体来说，现有方法大多假设地形是确定性的或空间独立的不确定性，忽略了3D空间数据的固有局部相关性，导致预测不可靠。这个问题在现实中非常重要，因为越野环境地形异构、感知不确定性高，不确定性感知的运动预测对可通行性评估和安全的自主导航至关重要，而传统方法忽略了关键的物理地形参数、空间相关性和环境不确定性。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先分析了现有方法的局限性，包括确定性方法、空间独立的不确定性估计方法和物理信息方法。他们借鉴了MonoForce的物理信息世界建模框架，但扩展了它以包含空间相关的不确定性。方法设计考虑了计算效率，通过结构化卷积算子避免显式构造高维协方差矩阵，结合深度学习与可微分物理引擎，实现了从感知到轨迹预测的端到端不确定性传播。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是提出一个端到端的概率框架，显式建模地形参数上的空间相关偶然不确定性，并通过可微分物理引擎传播这种不确定性用于概率轨迹预测，利用结构化卷积算子以可管理的计算成本提供高分辨率多变量预测。整体流程：1)输入车载摄像头图像；2)使用Lift-Splat-Shoot将特征投影到鸟瞰图网格；3)通过卷积层预测地形参数的平均图和方差图；4)将方差图与固定高斯核卷积形成结构化多元高斯分布；5)从概率世界模型中采样；6)通过可微分物理引擎进行轨迹预测。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)具有空间相关偶然不确定性的地形端到端概率世界建模；2)基于卷积的结构化协方差估计的可扩展损失公式；3)通过可微分物理引擎进行不确定性感知的轨迹预测；4)在真实世界数据集上的广泛验证。相比之前工作，不同之处在于：与确定性方法相比显式建模了不确定性；与空间独立方法相比考虑了空间相关性；与现有物理信息方法相比在模型中明确包含空间相关不确定性；与高斯过程等方法相比通过结构化卷积提高了计算效率。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'ProTerrain通过引入基于结构化卷积的高效概率框架，解决了越野导航中地形参数空间相关不确定性建模的挑战，实现了从感知到轨迹预测的端到端不确定性传播，显著提高了机器人在复杂环境中的导航安全性和可靠性。'}


### 论文摘要

Uncertainty-aware robot motion prediction is crucial for downstream traversability estimation and safe autonomous navigation in unstructured, off-road environments, where terrain is heterogeneous and perceptual uncertainty is high. Most existing methods assume deterministic or spatially independent terrain uncertainties, ignoring the inherent local correlations of 3D spatial data and often producing unreliable predictions. In this work, we introduce an efficient probabilistic framework that explicitly models spatially correlated aleatoric uncertainty over terrain parameters as a probabilistic world model and propagates this uncertainty through a differentiable physics engine for probabilistic trajectory forecasting. By leveraging structured convolutional operators, our approach provides high-resolution multivariate predictions at manageable computational cost. Experimental evaluation on a publicly available dataset shows significantly improved uncertainty estimation and trajectory prediction accuracy over aleatoric uncertainty estimation baselines.

---

## 5. HumanCM: One Step Human Motion Prediction

**论文链接:** [http://arxiv.org/abs/2510.16709v2](http://arxiv.org/abs/2510.16709v2)

**作者:** Liu Haojie, Gao Suixiang

**发布时间:** 2025-10-19

**备注:** 6 pages, 3 figures, 2 tables

### GPT解析

### 总结

HumanCM是一种基于一致性模型的人体运动预测框架，能够实现高效的单步生成，无需依赖多步去噪过程。

### 背景

现有基于扩散模型的人体运动预测方法通常需要多步去噪过程，计算效率较低。

### 目的

开发一种高效的单步人体运动预测框架，在减少计算步骤的同时保持或提高预测准确性。

### 方法

HumanCM学习嘈杂和清洁运动状态之间的自一致映射，采用基于Transformer的时空架构，使用时间嵌入来建模长程依赖并保持运动一致性。

### 主要发现

在Human3.6M和HumanEva-I数据集上的实验表明，HumanCM能够达到与最先进的扩散模型相当或更好的准确性，同时将推理步骤减少最多两个数量级。

### 结论

HumanCM是一种高效的人体运动预测方法，能够在单步生成中实现高准确性。

### 翻译

我们提出了HumanCM，一种基于一致性模型构建的单步人体运动预测框架。HumanCM不依赖于扩散模型中的多步去噪，而是通过学习嘈杂和清洁运动状态之间的自一致映射，执行高效的单步生成。该框架采用基于Transformer的时空架构，使用时间嵌入来建模长程依赖并保持运动一致性。在Human3.6M和HumanEva-I上的实验表明，HumanCM能够达到与最先进的扩散模型相当或更好的准确性，同时将推理步骤减少最多两个数量级。


### 论文摘要

We present HumanCM, a one-step human motion prediction framework built upon consistency models. Instead of relying on multi-step denoising as in diffusion-based methods, HumanCM performs efficient single-step generation by learning a self-consistent mapping between noisy and clean motion states. The framework adopts a Transformer-based spatiotemporal architecture with temporal embeddings to model long-range dependencies and preserve motion coherence. Experiments on Human3.6M and HumanEva-I demonstrate that HumanCM achieves comparable or superior accuracy to state-of-the-art diffusion models while reducing inference steps by up to two orders of magnitude.

---

## 6. Bayesian Inference of Primordial Magnetic Field Parameters from CMB with Spherical Graph Neural Networks

**论文链接:** [http://arxiv.org/abs/2510.20795v1](http://arxiv.org/abs/2510.20795v1)

**作者:** Juan Alejandro Pinto Castro, Héctor J. Hortúa, Jorge Enrique García-Farieta, Roger Anderson Hurtado

**发布时间:** 2025-10-23

**备注:** 16 pages, 6 figures, 4 tables

### GPT解析

### 总结

本文提出了一种基于贝叶斯图深度学习的框架，用于从宇宙微波背景图中估算原始磁场宇宙学参数，并结合贝叶斯神经网络实现不确定性量化。

### 背景

深度学习已成为现代宇宙学中的变革性方法，为从复杂数据集中提取物理信息提供了强大工具。

### 目的

实现一种新颖的贝叶斯图深度学习框架，用于从模拟的宇宙微波背景(CMB)图中直接估算原始磁场(PMF)宇宙学中的关键宇宙学参数。

### 方法

使用DeepSphere球形卷积神经网络架构，通过HEALPix像素化尊重CMB数据的球形几何特性，并集成贝叶斯神经网络(BNNs)来捕获偶然性和认知性不确定性，实现稳健的不确定性量化。

### 主要发现

所提出的方法表现出卓越的性能，在磁场参数估计中实现了超过0.89的R²分数，并通过方差缩放和GPNormal后训练技术获得了校准良好的不确定性估计。

### 结论

集成的DeepSphere-BNNs框架不仅提供了来自带有PMF贡献的CMB图的准确参数估计，还提供了可靠的不确定性量化，为精确宇宙学时代的稳健宇宙学推理提供了必要工具。

### 翻译

深度学习已成为现代宇宙学中的变革性方法，为从复杂数据集中提取有意义的物理信息提供了强大工具。本文实现了一种新颖的贝叶斯图深度学习框架，用于从模拟的宇宙微波背景(CMB)图中直接估算原始磁场(PMF)宇宙学中的关键宇宙学参数。我们的方法利用了DeepSphere，这是一种专门设计用于通过HEALPix像素化尊重CMB数据球形几何形状的球形卷积神经网络架构。为了超越确定性点估计并实现稳健的不确定性量化，我们将贝叶斯神经网络(BNNs)集成到框架中，捕获反映模型对其预测置信度的偶然性和认知性不确定性。所提出的方法表现出卓越的性能，在磁场参数估计中实现了超过0.89的R²分数。我们通过方差缩放和GPNormal等后训练技术获得了校准良好的不确定性估计。这种集成的DeepSphere-BNNs框架不仅提供了来自带有PMF贡献的CMB图的准确参数估计，还提供了可靠的不确定性量化，为精确宇宙学时代的稳健宇宙学推理提供了必要工具。


### 论文摘要

Deep learning has emerged as a transformative methodology in modern cosmology, providing powerful tools to extract meaningful physical information from complex astronomical datasets. This paper implements a novel Bayesian graph deep learning framework for estimating key cosmological parameters in a primordial magnetic field (PMF) cosmology directly from simulated Cosmic Microwave Background (CMB) maps. Our methodology utilizes DeepSphere, a spherical convolutional neural network architecture specifically designed to respect the spherical geometry of CMB data through HEALPix pixelization. To advance beyond deterministic point estimates and enable robust uncertainty quantification, we integrate Bayesian Neural Networks (BNNs) into the framework, capturing aleatoric and epistemic uncertainties that reflect the model confidence in its predictions. The proposed approach demonstrates exceptional performance, achieving $R^{2}$ scores exceeding 0.89 for the magnetic parameter estimation. We further obtain well-calibrated uncertainty estimates through post-hoc training techniques including Variance Scaling and GPNormal. This integrated DeepSphere-BNNs framework not only delivers accurate parameter estimation from CMB maps with PMF contributions but also provides reliable uncertainty quantification, providing the necessary tools for robust cosmological inference in the era of precision cosmology.

---

## 7. Learning to Triage Taint Flows Reported by Dynamic Program Analysis in Node.js Packages

**论文链接:** [http://arxiv.org/abs/2510.20739v1](http://arxiv.org/abs/2510.20739v1)

**作者:** Ronghao Ni, Aidan Z. H. Yang, Min-Chien Hsu, Nuno Sabino, Limin Jia, Ruben Martins, Darion Cassel, Kevin Cheang

**发布时间:** 2025-10-23

### GPT解析

### 总结

该论文研究了如何使用机器学习技术优先处理程序分析工具报告的漏洞，通过在Node.js包上的实验表明，机器学习模型能有效减少人工审查工作量，同时保持高准确率。

### 背景

程序分析工具会产生大量候选漏洞报告，需要昂贵的人工审查，这给安全分析师带来了如何优先处理最可能是真实漏洞的报告的实际挑战。

### 目的

研究机器学习是否可以应用于优先处理程序分析工具报告的漏洞，以减轻安全分析师的工作负担。

### 方法

收集了1,883个Node.js包的基准测试数据集，每个包包含一个报告的ACE或ACI漏洞；评估了多种机器学习方法，包括经典模型、图神经网络、大型语言模型以及混合模型；所有模型都基于动态程序分析工具的输出数据进行训练。

### 主要发现

顶级大型语言模型达到F1分数0.915，最佳图神经网络和经典机器学习模型达到F1分数0.904；在低于7%的假阴性率下，领先模型可消除66.9%的良性包无需人工审查，每个包处理时间约60毫秒；当模型调整为0.8精度水平时，可检测99.2%的可利用污染流，仅遗漏0.8%。

### 结论

该机器学习方法在现实世界漏洞分类中显示出强大的潜力，能显著减少人工审查工作量，同时保持高检测率。

### 翻译

程序分析工具经常产生大量候选漏洞报告，需要昂贵的人工审查，这带来了一个实际挑战：安全分析师如何优先处理最可能是真实漏洞的报告？本文研究了机器学习是否可以应用于优先处理程序分析工具报告的漏洞。我们专注于Node.js包，收集了1,883个Node.js包的基准测试，每个包包含一个报告的ACE或ACI漏洞。我们评估了多种机器学习方法，包括经典模型、图神经网络、大型语言模型以及结合GNN和LLMs的混合模型，这些模型都基于动态程序分析工具的输出数据进行训练。顶级LLM达到F1分数0.915，而最佳GNN和经典ML模型达到F1分数0.904。在低于7%的假阴性率下，领先模型消除了66.9%需要人工审查的良性包，每个包处理时间约60毫秒。如果将最佳模型调整为在0.8的精度水平下运行（即在所有警告中允许20%的假阳性），我们的方法可以检测99.2%的可利用污染流，仅遗漏0.8%，这表明在现实世界漏洞分类方面具有强大的潜力。


### 论文摘要

Program analysis tools often produce large volumes of candidate vulnerability reports that require costly manual review, creating a practical challenge: how can security analysts prioritize the reports most likely to be true vulnerabilities?   This paper investigates whether machine learning can be applied to prioritizing vulnerabilities reported by program analysis tools. We focus on Node.js packages and collect a benchmark of 1,883 Node.js packages, each containing one reported ACE or ACI vulnerability. We evaluate a variety of machine learning approaches, including classical models, graph neural networks (GNNs), large language models (LLMs), and hybrid models that combine GNN and LLMs, trained on data based on a dynamic program analysis tool's output. The top LLM achieves $F_{1} {=} 0.915$, while the best GNN and classical ML models reaching $F_{1} {=} 0.904$. At a less than 7% false-negative rate, the leading model eliminates 66.9% of benign packages from manual review, taking around 60 ms per package. If the best model is tuned to operate at a precision level of 0.8 (i.e., allowing 20% false positives amongst all warnings), our approach can detect 99.2% of exploitable taint flows while missing only 0.8%, demonstrating strong potential for real-world vulnerability triage.

---

## 8. Unsupervised Anomaly Prediction with N-BEATS and Graph Neural Network in Multi-variate Semiconductor Process Time Series

**论文链接:** [http://arxiv.org/abs/2510.20718v1](http://arxiv.org/abs/2510.20718v1)

**作者:** Daniel Sorensen, Bappaditya Dey, Minjin Hwang, Sandip Halder

**发布时间:** 2025-10-23

**备注:** 17 pages, 27 figures

### GPT解析

### 总结

该论文研究了半导体制造中的异常预测问题，提出了两种新方法：基于N-BEATS的单变量预测模型和基于图神经网络(GNN)的多变量关系模型。研究显示GNN在性能和效率上均优于N-BEATS模型，为制造环境中的在线异常预测提供了有前景的解决方案。

### 背景

半导体制造是极其复杂且精度要求高的过程，涉及数千个相互依赖的参数。多变量时间序列分析在此类环境中至关重要，但异常预测面临传感器数据高维度、真实故障稀有导致的类别不平衡，以及变量间复杂相互依赖关系等挑战。

### 目的

推动半导体制造领域从异常检测向异常预测发展，为实现实时工艺校正和主动故障预防提供技术支持。

### 方法

提出包含两个阶段的异常预测框架：首先在假设无异常的数据集上训练预测模型，然后对未见时间序列数据进行预测。两种方法差异在于：第一种使用N-BEATS模型进行单变量时间序列预测，假设变量间独立；第二种使用图神经网络捕捉变量间关系，取消独立性假设。

### 主要发现

两种模型在20个时间点的预测范围内表现出色，并在50个时间点范围内保持稳定的异常预测能力。GNN持续优于N-BEATS模型，同时需要更少的可训练参数和更低的计算成本。

### 结论

图神经网络(GNN)作为在线异常预测解决方案具有显著优势，适合在制造环境中部署，为半导体制造业的实时监控和故障预防提供了有效工具。

### 翻译

半导体制造是一个极其复杂且精度要求高的过程，其特点是在各种工具和工艺步骤中收集数千个相互依赖的参数。多变量时间序列分析已成为这类环境中实时监控和故障检测的关键领域。然而，半导体制造中的异常预测面临几个关键挑战，包括传感器数据的高维度和由于真实故障的稀有性导致的严重类别不平衡。此外，变量之间复杂的相互依赖关系使异常预测和根本原因分析都变得复杂。本文提出了两种新颖的方法，推动该领域从异常检测向异常预测发展，这是实现实时工艺校正和主动故障预防的关键步骤。提出的异常预测框架包含两个主要阶段：在假设不含异常的数据集上训练预测模型，然后对未见的时间序列数据进行预测。将预测结果与训练信号的预测进行比较，超出预定阈值的偏差被标记为异常。这两种方法在采用的预测模型上有所不同。第一种利用N-BEATS模型进行单变量时间序列预测，假设变量间相互独立。第二种利用图神经网络捕捉变量间关系，取消了这一假设。两种模型在长达20个时间点的预测范围内都表现出强大的预测性能，并在长达50个时间点的范围内保持稳定的异常预测能力。图神经网络始终优于N-BEATS模型，同时需要显著更少的可训练参数和更低的计算成本。这些结果将图神经网络定位为一种有前景的在线异常预测解决方案，适合在制造环境中部署。


### 论文摘要

Semiconductor manufacturing is an extremely complex and precision-driven process, characterized by thousands of interdependent parameters collected across diverse tools and process steps. Multi-variate time-series analysis has emerged as a critical field for real-time monitoring and fault detection in such environments. However, anomaly prediction in semiconductor fabrication presents several critical challenges, including high dimensionality of sensor data and severe class imbalance due to the rarity of true faults. Furthermore, the complex interdependencies between variables complicate both anomaly prediction and root-cause-analysis. This paper proposes two novel approaches to advance the field from anomaly detection to anomaly prediction, an essential step toward enabling real-time process correction and proactive fault prevention. The proposed anomaly prediction framework contains two main stages: (a) training a forecasting model on a dataset assumed to contain no anomalies, and (b) performing forecast on unseen time series data. The forecast is compared with the forecast of the trained signal. Deviations beyond a predefined threshold are flagged as anomalies. The two approaches differ in the forecasting model employed. The first assumes independence between variables by utilizing the N-BEATS model for univariate time series forecasting. The second lifts this assumption by utilizing a Graph Neural Network (GNN) to capture inter-variable relationships. Both models demonstrate strong forecasting performance up to a horizon of 20 time points and maintain stable anomaly prediction up to 50 time points. The GNN consistently outperforms the N-BEATS model while requiring significantly fewer trainable parameters and lower computational cost. These results position the GNN as promising solution for online anomaly forecasting to be deployed in manufacturing environments.

---

## 9. GRACE: GRaph-based Addiction Care prEdiction

**论文链接:** [http://arxiv.org/abs/2510.20671v1](http://arxiv.org/abs/2510.20671v1)

**作者:** Subham Kumar, Prakrithi Shivaprakash, Koustav Rudra, Lekhansh Shukla, Animesh Mukherjee

**发布时间:** 2025-10-23

### GPT解析

### 总结

本研究提出了一种名为GRACE的图神经网络框架，用于自动确定成瘾患者的适当护理场所，解决了传统方法中存在的类别不平衡问题，并在真实数据中取得显著改进。

### 背景

确定成瘾患者的适当护理场所是影响治疗效果和资源利用的关键临床决策。由于专科治疗资源不足，且现有决策方法在成瘾数据集中存在严重类别不平衡问题，需要开发自动化框架。

### 目的

开发一个自动化框架来确定成瘾患者的适当护理场所，并解决成瘾数据集中的类别不平衡问题。

### 方法

提出新的图神经网络框架GRACE，将护理场所预测形式化为结构化学习问题；进行广泛特征工程；提出获取无偏元图的新方法训练图神经网络以克服类别不平衡问题。

### 主要发现

在真实世界数据上的实验结果表明，与竞争基线相比，少数类别的F1分数提高了11-35%。

### 结论

GRACE框架在解决成瘾患者护理场所预测问题上有效，特别是在处理类别不平衡问题时表现优异。

### 翻译

确定成瘾患者的适当护理场所是影响患者治疗效果和资源有效利用的最关键临床决策之一。由于缺乏足够的专科治疗资源，如住院床位或人员，开发自动化框架的需求尚未得到满足。当前的决策方法在成瘾数据集中存在严重的类别不平衡问题。为解决这一限制，我们提出了一个新的图神经网络框架，将护理场所预测形式化为一个结构化学习问题。此外，我们进行了广泛的特征工程，并提出了一种获得无偏元图的新方法来训练图神经网络，以克服类别不平衡问题。真实世界数据的实验结果表明，与竞争基线相比，少数类别的F1分数提高了11-35%。代码和注释嵌入可在https://anonymous.4open.science/r/GRACE-F8E1/获取。


### 论文摘要

Determining the appropriate locus of care for addiction patients is one of the most critical clinical decisions that affects patient treatment outcomes and effective use of resources. With a lack of sufficient specialized treatment resources, such as inpatient beds or staff, there is an unmet need to develop an automated framework for the same. Current decision-making approaches suffer from severe class imbalances in addiction datasets. To address this limitation, we propose a novel graph neural network (GRACE) framework that formalizes locus of care prediction as a structured learning problem. Further, we perform extensive feature engineering and propose a new approach of obtaining an unbiased meta-graph to train a GNN to overcome the class imbalance problem. Experimental results in real-world data show an improvement of 11-35% in terms of the F1 score of the minority class over competitive baselines. The codes and note embeddings are available at https://anonymous.4open.science/r/GRACE-F8E1/.

---

## 10. Transferable Graph Learning for Transmission Congestion Management via Busbar Splitting

**论文链接:** [http://arxiv.org/abs/2510.20591v1](http://arxiv.org/abs/2510.20591v1)

**作者:** Ali Rajaei, Peter Palensky, Jochen L. Cremer

**发布时间:** 2025-10-23

### GPT解析

### 总结

本研究提出了一种基于图神经网络(GNN)的电网拓扑优化(NTO)加速方法，用于解决输电网拥堵管理问题。该方法通过母线分裂技术，能够实现大规模系统的近实时优化，并具有良好的泛化能力和跨系统可转移性。

### 背景

电网拓扑优化(NTO)通过母线分裂可以缓解输电网拥堵并减少重新调度成本。然而，对于大规模系统，使用现有求解器解决这种混合整数非线性问题在近实时情况下目前是不可行的。虽然机器学习方法是一种有前景的替代方案，但它们对未见的拓扑、变化的运行条件和不同系统的泛化能力有限，限制了其实际应用。

### 目的

本研究旨在开发一种能够实现大规模系统近实时电网拓扑优化的方法，同时解决现有方法在泛化能力和跨系统可转移性方面的局限性。

### 方法

本文考虑线性化交流潮流(AC PF)，将NTO公式化为拥堵管理问题，并提出了一种图神经网络(GNN)加速方法。研究人员开发了一种异构边缘感知消息传递神经网络，以预测有效的母线分裂行为作为候选NTO解决方案。该方法能够捕获局部流模式，实现对未见的拓扑变化的泛化，并提高跨系统的可转移性。

### 主要发现

案例研究显示，所提出的GNN方法在速度上提高了4个数量级，能够在GOC 2000母线系统上一分钟内提供交流可行解，优化间隙为2.3%。这表明该方法在近实时优化方面取得了显著进展。

### 结论

这些结果表明，所提出的GNN方法在具有拓扑和跨系统泛化能力的大规模系统近实时NTO方面取得了重要进展，为解决电网拥堵管理问题提供了新的有效途径。

### 翻译

电网拓扑优化(NTO)通过母线分裂可以缓解输电网拥堵并减少重新调度成本。然而，使用现有求解器解决大规模系统的这种混合整数非线性问题在近实时情况下目前是不可行的。机器学习方法已成为一种有前景的替代方案，但它们对未见的拓扑、变化的运行条件和不同系统的泛化能力有限，这限制了它们的实际应用。本文考虑线性化交流潮流(AC PF)，将NTO公式化为拥堵管理问题，并提出了一种图神经网络(GNN)加速方法。我们开发了一种异构边缘感知消息传递神经网络，以预测有效的母线分裂行为作为候选NTO解决方案。所提出的GNN捕获局部流模式，实现对未见的拓扑变化的泛化，并提高了跨系统的可转移性。案例研究显示速度提高了4个数量级，在GOC 2000母线系统上在一分钟内提供交流可行解，优化间隙为2.3%。这些结果表明，在具有拓扑和跨系统泛化能力的大规模系统近实时NTO方面取得了重要进展。


### 论文摘要

Network topology optimization (NTO) via busbar splitting can mitigate transmission grid congestion and reduce redispatch costs. However, solving this mixed-integer non-linear problem for large-scale systems in near-real-time is currently intractable with existing solvers. Machine learning (ML) approaches have emerged as a promising alternative, but they have limited generalization to unseen topologies, varying operating conditions, and different systems, which limits their practical applicability. This paper formulates NTO for congestion management problem considering linearized AC PF, and proposes a graph neural network (GNN)-accelerated approach. We develop a heterogeneous edge-aware message passing NN to predict effective busbar splitting actions as candidate NTO solutions. The proposed GNN captures local flow patterns, achieves generalization to unseen topology changes, and improves transferability across systems. Case studies show up to 4 orders-of-magnitude speed-up, delivering AC-feasible solutions within one minute and a 2.3% optimality gap on the GOC 2000-bus system. These results demonstrate a significant step toward near-real-time NTO for large-scale systems with topology and cross-system generalization.

---

## 11. Structural Invariance Matters: Rethinking Graph Rewiring through Graph Metrics

**论文链接:** [http://arxiv.org/abs/2510.20556v1](http://arxiv.org/abs/2510.20556v1)

**作者:** Alexandre Benoit, Catherine Aitken, Yu He

**发布时间:** 2025-10-23

**备注:** 21 pages, 5 figures, conference

### GPT解析

### 总结

图重连是减轻图神经网络和图变换器中过度压缩的关键技术，通过修改图拓扑改善信息流动，但会改变图结构，可能扭曲重要拓扑信号。本研究首次系统分析了重连对图结构指标的影响及其与下游任务性能的关系。

### 背景

图重连已成为减轻图神经网络（GNNs）和图变换器（Graph Transformers）中过度压缩的关键技术，通过修改图拓扑来改善信息流动。然而，重连会改变图的结构，有扭曲重要拓扑相关信号的风险。

### 目的

提供关于重连如何影响各种图结构指标的系统性分析，以及这些变化如何与下游任务性能相关联，明确需要保留哪些结构属性以确保性能提升和结构保真度。

### 方法

研究七种不同的重连策略，并将局部和全局图属性的变化与节点分类准确性进行关联分析。

### 主要发现

成功的重连方法倾向于保留局部结构，同时在全局连接方面保持灵活性。这一模式在研究中呈现出一致性。

### 结论

这些发现为设计有效的重连策略提供了新的见解，弥合了图理论与实际GNN优化之间的差距。

### 翻译

图重连已成为减轻图神经网络（GNNs）和图变换器（Graph Transformers）中过度压缩的关键技术，通过修改图拓扑来改善信息流动。虽然有效，但重连本质上改变了图的结构，带来了扭曲重要拓扑相关信号的风险。然而，尽管重连的使用日益增多，人们尚不清楚需要保留哪些结构属性以确保性能提升和结构保真度。在本工作中，我们首次提供了关于重连如何影响各种图结构指标的系统性分析，以及这些变化如何与下游任务性能相关联。我们研究了七种不同的重连策略，并将局部和全局图属性的变化与节点分类准确性进行关联。我们的结果揭示了一个一致的规律：成功的重连方法倾向于保留局部结构，同时在全局连接方面保持灵活性。这些发现为设计有效的重连策略提供了新的见解，弥合了图理论与实际GNN优化之间的差距。


### 论文摘要

Graph rewiring has emerged as a key technique to alleviate over-squashing in Graph Neural Networks (GNNs) and Graph Transformers by modifying the graph topology to improve information flow. While effective, rewiring inherently alters the graph's structure, raising the risk of distorting important topology-dependent signals. Yet, despite the growing use of rewiring, little is known about which structural properties must be preserved to ensure both performance gains and structural fidelity. In this work, we provide the first systematic analysis of how rewiring affects a range of graph structural metrics, and how these changes relate to downstream task performance. We study seven diverse rewiring strategies and correlate changes in local and global graph properties with node classification accuracy. Our results reveal a consistent pattern: successful rewiring methods tend to preserve local structure while allowing for flexibility in global connectivity. These findings offer new insights into the design of effective rewiring strategies, bridging the gap between graph theory and practical GNN optimization.

---

## 12. Intransitive Player Dominance and Market Inefficiency in Tennis Forecasting: A Graph Neural Network Approach

**论文链接:** [http://arxiv.org/abs/2510.20454v1](http://arxiv.org/abs/2510.20454v1)

**作者:** Lawrence Clegg, John Cartlidge

**发布时间:** 2025-10-23

**备注:** 39 pages, 8 figures

### GPT解析

### 总结

该研究提出了一种基于图神经网络的网球比赛预测方法，专门处理非传递性玩家优势现象，并发现博彩市场在此类比赛中存在效率低下的问题。

### 背景

非传递性玩家优势（即A击败B，B击败C，但C击败A）在网球比赛中很常见，但很少有预测方法能处理这种复杂关系。

### 目的

开发一种能够建模和利用非传递性关系的预测方法，以识别和利用博彩市场中的效率低下点。

### 方法

使用图神经网络方法，通过时序有向图建模玩家关系，其中玩家作为节点，历史比赛结果作为有向边。

### 主要发现

博彩公司Pinnacle Sports在处理高非传递性复杂度的比赛时表现不佳；基于图的方法能有效捕捉这些场景中的关系动态；在1903次投注中，使用Kelly下注策略获得了3.26%的显著正回报率。

### 结论

博彩市场在处理非传递性比赛时存在效率低下的问题，而所提出的图神经网络方法能够成功利用这种市场效率低下问题。

### 翻译

非传递性玩家优势，即玩家A击败B，B击败C，但C击败A，在竞争性网球比赛中很常见。然而，很少有已知的方法尝试将其纳入预测方法中。我们通过图神经网络方法解决了这个问题，该方法通过时序有向图明确建模这些非传递性关系，玩家作为节点，他们的历史比赛结果作为有向边。我们发现博彩公司Pinnacle Sports在处理高非传递性复杂度的比赛时表现不佳，并认为我们的基于图的方法能够捕捉这些场景中的关系动态。当有选择地使用我们的模型对高非传递性比赛进行下注时（准确率65.7%，0.215 Brier分数），在1903次投注中，使用Kelly下注策略获得了3.26%的显著正回报率，表明市场在处理非传递性比赛时存在效率低下的问题，而我们的方法成功地利用了这一点。


### 论文摘要

Intransitive player dominance, where player A beats B, B beats C, but C beats A, is common in competitive tennis. Yet, there are few known attempts to incorporate it within forecasting methods. We address this problem with a graph neural network approach that explicitly models these intransitive relationships through temporal directed graphs, with players as nodes and their historical match outcomes as directed edges. We find the bookmaker Pinnacle Sports poorly handles matches with high intransitive complexity and posit that our graph-based approach is uniquely positioned to capture relational dynamics in these scenarios. When selectively betting on higher intransitivity matchups with our model (65.7% accuracy, 0.215 Brier Score), we achieve significant positive returns of 3.26% ROI with Kelly staking over 1903 bets, suggesting a market inefficiency in handling intransitive matchups that our approach successfully exploits.

---

## 13. Quantifying Distributional Invariance in Causal Subgraph for IRM-Free Graph Generalization

**论文链接:** [http://arxiv.org/abs/2510.20295v1](http://arxiv.org/abs/2510.20295v1)

**作者:** Yang Qiu, Yixiong Zou, Jun Wang, Wei Liu, Xiangyu Fu, Ruixuan Li

**发布时间:** 2025-10-23

### GPT解析

### 总结

本文提出了一种不需要不变风险最小化框架的方法，用于捕获因果子图，解决分布偏移下图神经网络的分布外泛化挑战。通过识别因果子图在不同环境中具有较小分布变化的特性，建立了不变分布标准，并基于此提出了范数引导的不变分布目标方法，实验证明该方法优于现有最先进方法。

### 背景

分布偏移下图神经网络的分布外泛化仍然是一个关键挑战。现有方法通常采用不变风险最小化框架，需要昂贵的环境注释或启发式生成的合成分割。

### 目的

开发一种不需要不变风险最小化框架的方法，用于捕获因果子图，以克服现有方法的限制。

### 方法

基于不变分布标准，系统地揭示分布偏移与表示范数之间的定量关系，用于识别因果子图；提出一种范数引导的不变分布目标方法，用于因果子图发现和预测。

### 主要发现

因果子图在不同环境中表现出比非因果成分小得多的分布变化，这被形式化为不变分布标准，并从理论上得到了证明。

### 结论

在两个广泛使用的基准测试上，所提出的方法在图泛化任务中始终优于现有的最先进方法，证明了该方法的有效性。

### 翻译

在分布偏移下的分布外泛化对于图神经网络仍然是一个关键挑战。现有方法通常采用不变风险最小化框架，需要昂贵的环境注释或启发式生成的合成分割。为避免这些限制，本文旨在开发一种不需要不变风险最小化的方法来捕获因果子图。我们首先确定因果子图在不同环境中表现出比非因果成分小得多的分布变化，这被形式化为不变分布标准，并在本文中从理论上进行了证明。基于这一标准，我们系统地揭示了分布偏移与表示范数之间的定量关系，用于识别因果子图，并深入研究了其潜在机制。最后，我们通过引入一个基于范数的不变分布目标，提出了一种不需要不变风险最小化的方法，用于因果子图发现和预测。在两个广泛使用的基准测试上的大量实验表明，我们的方法在图泛化方面始终优于最先进的方法。


### 论文摘要

Out-of-distribution generalization under distributional shifts remains a critical challenge for graph neural networks. Existing methods generally adopt the Invariant Risk Minimization (IRM) framework, requiring costly environment annotations or heuristically generated synthetic splits. To circumvent these limitations, in this work, we aim to develop an IRM-free method for capturing causal subgraphs. We first identify that causal subgraphs exhibit substantially smaller distributional variations than non-causal components across diverse environments, which we formalize as the Invariant Distribution Criterion and theoretically prove in this paper. Building on this criterion, we systematically uncover the quantitative relationship between distributional shift and representation norm for identifying the causal subgraph, and investigate its underlying mechanisms in depth. Finally, we propose an IRM-free method by introducing a norm-guided invariant distribution objective for causal subgraph discovery and prediction. Extensive experiments on two widely used benchmarks demonstrate that our method consistently outperforms state-of-the-art methods in graph generalization.

---

## 14. Layer-to-Layer Knowledge Mixing in Graph Neural Network for Chemical Property Prediction

**论文链接:** [http://arxiv.org/abs/2510.20236v1](http://arxiv.org/abs/2510.20236v1)

**作者:** Teng Jiek See, Daokun Zhang, Mario Boley, David K. Chalmers

**发布时间:** 2025-10-23

### GPT解析

### 总结

本文提出了一种名为层到层知识混合(LKM)的自知识蒸馏方法，能够在不显著增加计算复杂性的情况下提高图神经网络(GNNs)预测分子性质的准确性。

### 背景

图神经网络(GNNs)是目前预测分子性质最有效的方法，但仍需要更准确的模型。增加模型复杂度可以提高准确性，但也会增加训练和推理过程中的计算成本和内存需求。

### 目的

开发一种提高GNN准确性的方法，同时不显著增加计算复杂性和内存需求。

### 方法

提出层到层知识混合(LKM)方法，通过最小化GNN层现有隐藏嵌入之间的平均绝对距离，有效聚合多跳和多尺度信息，改进局部和全局分子特征的表示。

### 主要发现

LKM方法在不显著增加训练和推理复杂性的情况下，提高了最先进GNN的准确性。使用三种不同的GNN架构(DimeNet++、MXMNet和PAMNet)和三个数据集(QM9、MD17和Chignolin)进行评估，LKM将量子化学和生物物理性质预测的平均绝对误差分别降低了最高9.8%(QM9)、45.3%(MD17能量)和22.9%(Chignolin)。

### 结论

LKM有潜力显著提高GNN预测化学性质的准确性，而不会显著增加训练和推理成本。

### 翻译

图神经网络(GNNs)是目前预测分子性质最有效的方法，但仍需要更准确的模型。通过增加模型复杂度可以提高GNN的准确性，但这也会增加训练和推理过程中的计算成本和内存需求。在本研究中，我们开发了层到层知识混合(LKM)，一种新颖的自知识蒸馏方法，它在提高最先进GNN准确性的同时，在训练和推理过程中只增加了微不足道的计算复杂度。通过最小化GNN层现有隐藏嵌入之间的平均绝对距离，LKM有效地聚合了多跳和多尺度信息，实现了对局部和全局分子特征的改进表示。我们使用三种不同的GNN架构(DimeNet++、MXMNet和PAMNet)和量子化学性质数据集(QM9、MD17和Chignolin)评估了LKM。我们发现LKM方法将量子化学和生物物理性质预测的平均绝对误差分别降低了高达9.8%(QM9)、45.3%(MD17能量)和22.9%(Chignolin)。这项工作证明了LKM在不显著增加训练和推理成本的情况下，显著提高GNN化学性质预测准确性的潜力。


### 论文摘要

Graph Neural Networks (GNNs) are the currently most effective methods for predicting molecular properties but there remains a need for more accurate models. GNN accuracy can be improved by increasing the model complexity but this also increases the computational cost and memory requirement during training and inference. In this study, we develop Layer-to-Layer Knowledge Mixing (LKM), a novel self-knowledge distillation method that increases the accuracy of state-of-the-art GNNs while adding negligible computational complexity during training and inference. By minimizing the mean absolute distance between pre-existing hidden embeddings of GNN layers, LKM efficiently aggregates multi-hop and multi-scale information, enabling improved representation of both local and global molecular features. We evaluated LKM using three diverse GNN architectures (DimeNet++, MXMNet, and PAMNet) using datasets of quantum chemical properties (QM9, MD17 and Chignolin). We found that the LKM method effectively reduces the mean absolute error of quantum chemical and biophysical property predictions by up to 9.8% (QM9), 45.3% (MD17 Energy), and 22.9% (Chignolin). This work demonstrates the potential of LKM to significantly improve the accuracy of GNNs for chemical property prediction without any substantial increase in training and inference cost.

---

## 15. Extending machine learning model for implicit solvation to free energy calculations

**论文链接:** [http://arxiv.org/abs/2510.20103v1](http://arxiv.org/abs/2510.20103v1)

**作者:** Rishabh Dey, Michael Brocidiacono, Kushal Koirala, Alexander Tropsha, Konstantin I. Popov

**发布时间:** 2025-10-23

### GPT解析

### 总结

本研究提出了一种基于图神经网络的隐式溶剂模型LSNN，通过结合力匹配和化学变量导数匹配，实现了与显式溶剂模型相当的自由能预测准确性，同时提高了计算效率，为药物发现应用奠定了基础。

### 背景

隐式溶剂方法在分子模拟中计算效率高，但与显式溶剂模型相比准确性不足，限制了其在精确热力学计算中的应用。基于机器学习的方法目前主要依靠力匹配，可能导致能量预测相差任意常数，不适合绝对自由能比较。

### 目的

开发更精确的隐式溶剂势能，解决当前基于机器学习的方法仅依靠力匹配的缺点，确保不同化学物种的溶剂化自由能可以进行有意义的比较。

### 方法

引入基于图神经网络(GNN)的隐式溶剂模型Lambda Solvation Neural Network (LSNN)，除了力匹配外，还训练网络匹配化学变量的导数。在约30万个小分子的数据集上进行训练。

### 主要发现

LSNN实现了与显式溶剂化学模拟相当的自由能预测准确性，同时提供了计算速度提升，为药物发现中的未来应用建立了基础框架。

### 结论

LSNN克服了传统隐式溶剂方法的局限性，通过结合力匹配和化学变量导数匹配，确保了不同化学物种的溶剂化自由能可以进行有意义的比较。

### 翻译

隐式溶剂方法为分子模拟中的溶剂化效应建模提供了计算效率高的框架。然而，与显式溶剂模型相比，其准确性往往不足，限制了其在精确热力学计算中的应用。机器学习(ML)的最新进展提供了一种克服这些局限性的机会，通过利用神经网络为各种应用开发更精确的隐式溶剂势能。当前基于ML方法的一个主要缺点是其仅依靠力匹配，这可能导致能量预测相差任意常数，因此不适合绝对自由能比较。在此，我们介绍了一种新颖的方法，即基于图神经网络(GNN)的隐式溶剂模型，称为Lambda Solvation Neural Network (LSNN)。除了力匹配外，该网络还被训练以匹配化学变量的导数，确保不同化学物种的溶剂化自由能可以进行有意义的比较。在约30万个小分子的数据集上训练后，LSNN实现了与显式溶剂化学模拟相当的自由能预测准确性，同时提供了计算加速，并为药物发现中的未来应用建立了基础框架。


### 论文摘要

The implicit solvent approach offers a computationally efficient framework to model solvation effects in molecular simulations. However, its accuracy often falls short compared to explicit solvent models, limiting its use in precise thermodynamic calculations. Recent advancements in machine learning (ML) present an opportunity to overcome these limitations by leveraging neural networks to develop more precise implicit solvent potentials for diverse applications. A major drawback of current ML-based methods is their reliance on force-matching alone, which can lead to energy predictions that differ by an arbitrary constant and are therefore unsuitable for absolute free energy comparisons. Here, we introduce a novel methodology with a graph neural network (GNN)-based implicit solvent model, dubbed Lambda Solvation Neural Network (LSNN). In addition to force-matching, this network was trained to match the derivatives of alchemical variables, ensuring that solvation free energies can be meaningfully compared across chemical species.. Trained on a dataset of approximately 300,000 small molecules, LSNN achieves free energy predictions with accuracy comparable to explicit-solvent alchemical simulations, while offering a computational speedup and establishing a foundational framework for future applications in drug discovery.

---

## 16. RELATE: A Schema-Agnostic Perceiver Encoder for Multimodal Relational Graphs

**论文链接:** [http://arxiv.org/abs/2510.19954v1](http://arxiv.org/abs/2510.19954v1)

**作者:** Joseph Meyer, Divyansha Lachi, Reza Mohammadi, Roshan Reddy Upendra, Eva L. Dyer, Mark Li, Tom Palczewski

**发布时间:** 2025-10-22

**备注:** 6 pages

### GPT解析

### 总结

这篇论文提出了RELATE，一种与模式无关的特征编码器，用于处理关系型多表数据，能与任何通用图神经网络一起使用，在保持性能的同时大幅减少参数数量。

### 背景

关系型多表数据在电子商务、医疗健康和科学研究等领域很常见，可表示为具有多模态节点属性的异质时间图。现有图神经网络依赖特定模式的特征编码器，需为每种节点类型和特征列设计单独模块，限制了可扩展性和参数共享。

### 目的

开发一种与模式无关、即插即用的特征编码器，能与任何通用图神经网络配合使用，解决现有方法的可扩展性和参数共享问题。

### 方法

RELATE采用针对分类、数值、文本和时间属性的共享模态特定编码器，后接Perceiver风格的交叉注意力模块，将特征聚合成固定大小、排列不变的节点表示。

### 主要发现

在RelBench基准测试中，RELATE实现了与特定模式编码器相当的性能（差距在3%以内），同时将参数数量减少了高达5倍。

### 结论

RELATE设计支持变化的数据模式，并支持通用图神经网络的多数据集预训练，为关系图数据的基础模型铺平了道路。

### 翻译

关系型多表数据在电子商务、医疗健康和科学研究等领域很常见，可以自然地表示为具有多模态节点属性的异质时间图。现有的图神经网络依赖于特定模式的特征编码器，需要为每种节点类型和特征列设计单独的模块，这阻碍了可扩展性和参数共享。我们引入了RELATE（关系型实体潜在聚合编码器），这是一种与模式无关的、即插即用的特征编码器，可以与任何通用图神经网络一起使用。RELATE采用针对分类、数值、文本和时间属性的共享模态特定编码器，然后是一个Perceiver风格的交叉注意力模块，将特征聚合成固定大小、排列不变的节点表示。我们在RelBench基准测试的ReLGNN和HGT上评估了RELATE，结果显示它实现了与特定模式编码器相当的性能（差距在3%以内），同时将参数数量减少了高达5倍。这种设计支持变化的数据模式，并支持通用图神经网络的多数据集预训练，为关系图数据的基础模型铺平了道路。


### 论文摘要

Relational multi-table data is common in domains such as e-commerce, healthcare, and scientific research, and can be naturally represented as heterogeneous temporal graphs with multi-modal node attributes. Existing graph neural networks (GNNs) rely on schema-specific feature encoders, requiring separate modules for each node type and feature column, which hinders scalability and parameter sharing. We introduce RELATE (Relational Encoder for Latent Aggregation of Typed Entities), a schema-agnostic, plug-and-play feature encoder that can be used with any general purpose GNN. RELATE employs shared modality-specific encoders for categorical, numerical, textual, and temporal attributes, followed by a Perceiver-style cross-attention module that aggregates features into a fixed-size, permutation-invariant node representation. We evaluate RELATE on ReLGNN and HGT in the RelBench benchmark, where it achieves performance within 3% of schema-specific encoders while reducing parameter counts by up to 5x. This design supports varying schemas and enables multi-dataset pretraining for general-purpose GNNs, paving the way toward foundation models for relational graph data.

---

## 17. FnRGNN: Distribution-aware Fairness in Graph Neural Network

**论文链接:** [http://arxiv.org/abs/2510.19257v1](http://arxiv.org/abs/2510.19257v1)

**作者:** Soyoung Park, Sungsu Lim

**发布时间:** 2025-10-22

**DOI:** 10.1145/3746252.3760796

### GPT解析

### 总结

本文提出了FnRGNN，一种用于图神经网络节点回归的公平性感知处理框架，通过在结构、表示和预测三个层面进行干预，有效减少了组间差异而不牺牲性能。

### 背景

图神经网络在结构化数据学习中表现出色，但在回归任务中的公平性研究仍然不足。

### 目的

解决图神经网络在节点级回归任务中的公平性问题，特别是处理现有方法无法解决的连续特性挑战。

### 方法

FnRGNN框架采用三级干预策略：结构层面的边重加权、表示层面的MMD对齐、以及预测层面的Sinkhorn分布匹配归一化。

### 主要发现

在四个真实世界数据集上的实验表明，FnRGNN能够有效减少组间差异，同时保持模型性能。

### 结论

多层级干预策略确保了FnRGNN在复杂图拓扑结构下的鲁棒公平性，为图神经网络回归任务中的公平性问题提供了有效解决方案。

### 翻译

图神经网络(GNNs)在结构化数据学习中表现出色，但在回归任务中的公平性研究仍然不足。现有方法主要针对分类任务和表示层面的去偏见，无法完全处理节点级回归的连续特性。我们提出了FnRGNN，一个基于GNN的节点回归的公平性感知处理框架，在三个层面进行干预：(i)结构层面的边重加权，(ii)表示层面的MMD对齐，(iii)预测层面的Sinkhorn分布匹配归一化。这种多层级策略确保了在复杂图拓扑结构下的鲁棒公平性。在四个真实世界数据集上的实验表明，FnRGNN减少了组间差异而不牺牲性能。代码可在https://github.com/sybeam27/FnRGNN获取。


### 论文摘要

Graph Neural Networks (GNNs) excel at learning from structured data, yet fairness in regression tasks remains underexplored. Existing approaches mainly target classification and representation-level debiasing, which cannot fully address the continuous nature of node-level regression. We propose FnRGNN, a fairness-aware in-processing framework for GNN-based node regression that applies interventions at three levels: (i) structure-level edge reweighting, (ii) representation-level alignment via MMD, and (iii) prediction-level normalization through Sinkhorn-based distribution matching. This multi-level strategy ensures robust fairness under complex graph topologies. Experiments on four real-world datasets demonstrate that FnRGNN reduces group disparities without sacrificing performance. Code is available at https://github.com/sybeam27/FnRGNN.

---

## 18. Enhancing Graph Neural Networks: A Mutual Learning Approach

**论文链接:** [http://arxiv.org/abs/2510.19223v1](http://arxiv.org/abs/2510.19223v1)

**作者:** Paul Agbaje, Akajyoti Mitra, Afia Anjum, Pranali Khose, Ebelechukwu Nwafor, Habeeb Olufowobi

**发布时间:** 2025-10-22

### GPT解析

### 总结

本研究提出了一种图神经网络之间的协作学习框架，在没有预训练教师模型的情况下，让简单浅层的GNN架构相互教学，从而提升模型在推理时的性能，特别是在处理多个任务时。

### 背景

知识蒸馏技术是将复杂教师模型专业知识转移到轻量级学生模型的有效工具，特别适合在资源受限设备上部署高性能模型。该方法已成功应用于图神经网络，利用其表达能力生成捕获结构和特征相关信息的节点嵌入。

### 目的

探索GNNs之间协作学习的潜力，使相对简单和浅层的GNN架构能够协同学习高效模型，在推理时表现更好，特别是在处理多个任务时。

### 方法

提出一个协作学习框架，其中学生GNN集合在整个训练过程中相互教学；引入自适应logit加权单元促进模型间的高效知识交换；采用熵增强技术改进相互学习；这些组件动态赋能模型在训练过程中调整学习策略，优化下游任务性能。

### 主要发现

简单浅层的GNN架构能够协同学习高效模型；这些模型在推理时表现更好，特别是在处理多个任务时；提出的自适应logit加权单元和熵增强技术有效促进了模型间的知识交换和相互学习。

### 结论

通过协作学习框架，学生GNN能够在没有预训练教师模型的情况下相互教学；提出的方法在节点分类和图分类任务上表现出色；为资源受限环境中的高效模型部署提供了新思路。

### 翻译

知识蒸馏技术已成为一种强大的工具，用于将复杂教师模型的专业知识转移到轻量级学生模型中，特别适合在资源受限设备上部署高性能模型。这种方法已成功应用于图神经网络，利用其表达能力生成捕获结构和特征相关信息的节点嵌入。在本研究中，我们通过探索GNNs之间协作学习的潜力，偏离了传统的KD方法。在没有预训练教师模型的情况下，我们证明相对简单和浅层的GNN架构能够协同学习高效模型，在推理时表现更好，特别是在处理多个任务时。我们提出了一个协作学习框架，其中学生GNN集合在整个训练过程中相互教学。我们引入了自适应logit加权单元以促进模型间的高效知识交换，以及熵增强技术以改进相互学习。这些组件动态赋能模型在训练过程中调整学习策略，优化下游任务性能。在三个节点分类和图分类数据集上进行的广泛实验证明了我们方法的有效性。


### 论文摘要

Knowledge distillation (KD) techniques have emerged as a powerful tool for transferring expertise from complex teacher models to lightweight student models, particularly beneficial for deploying high-performance models in resource-constrained devices. This approach has been successfully applied to graph neural networks (GNNs), harnessing their expressive capabilities to generate node embeddings that capture structural and feature-related information. In this study, we depart from the conventional KD approach by exploring the potential of collaborative learning among GNNs. In the absence of a pre-trained teacher model, we show that relatively simple and shallow GNN architectures can synergetically learn efficient models capable of performing better during inference, particularly in tackling multiple tasks. We propose a collaborative learning framework where ensembles of student GNNs mutually teach each other throughout the training process. We introduce an adaptive logit weighting unit to facilitate efficient knowledge exchange among models and an entropy enhancement technique to improve mutual learning. These components dynamically empower the models to adapt their learning strategies during training, optimizing their performance for downstream tasks. Extensive experiments conducted on three datasets each for node and graph classification demonstrate the effectiveness of our approach.

---

## 19. An Active Diffusion Neural Network for Graphs

**论文链接:** [http://arxiv.org/abs/2510.19202v1](http://arxiv.org/abs/2510.19202v1)

**作者:** Mengying Jiang

**发布时间:** 2025-10-22

### GPT解析

### 总结

本文提出了一种名为ADGNN的新型图神经网络，通过整合多个外部信息源实现主动扩散，解决了传统扩散GNN的过平滑问题，使节点能保留独特特征同时获取全局图结构信息。

### 背景

热扩散类比促进了图信息流理解和图神经网络发展，但大多数扩散GNN模拟被动热扩散，存在过平滑问题，限制了捕获全局图信息的能力。这类似于宇宙热寂理论，即封闭系统中能量分布随时间变得均匀，导致节点表示收敛到相同特征向量。

### 目的

解决传统扩散GNN中的过平滑问题，使节点能保留独特特征同时有效获取图全局结构信息。

### 方法

提出ADGNN（主动扩散图神经网络），通过整合多个外部信息源实现主动扩散，动态影响扩散过程克服过平滑问题。通过直接计算主动扩散迭代公式的闭式解，实现真正的无限扩散。

### 主要发现

ADGNN在多种图任务上与最先进GNN模型相比，显著提高了准确性和效率，有效捕获全局图信息并保持节点独特性。

### 结论

ADGNN通过主动扩散机制解决了传统扩散GNN的过平滑问题，使节点既能保持独特特征又能获取全局图结构信息，在多种图任务上表现出色。

### 翻译

热扩散的类比增强了对图中信息流的理解，并启发了图神经网络(GNNs)的发展。然而，大多数基于扩散的GNN模拟被动热扩散，仍然存在过平滑问题，限制了它们捕获全局图信息的能力。受宇宙热寂理论的启发，该理论认为在封闭系统中能量分布随时间变得均匀，我们认识到，在没有外部输入的情况下，节点表示会随着扩散过程收敛到相同的特征向量。为解决这个问题，我们提出了主动扩散图神经网络(ADGNN)。ADGNN通过整合多个外部信息源实现主动扩散，这些信息源动态影响扩散过程，有效克服了过平滑问题。此外，我们的方法通过直接计算主动扩散迭代公式的闭式解，实现了真正的无限扩散。这使得节点能够保留其独特特征，同时有效地获取对图全局结构的全面理解。我们在各种图任务上将ADGNN与几个最先进的GNN模型进行了评估。结果表明，ADGNN显著提高了准确性和效率，突显了其在捕获全局图信息和保持节点独特性方面的有效性。


### 论文摘要

The analogy to heat diffusion has enhanced our understanding of information flow in graphs and inspired the development of Graph Neural Networks (GNNs). However, most diffusion-based GNNs emulate passive heat diffusion, which still suffers from over-smoothing and limits their ability to capture global graph information. Inspired by the heat death of the universe, which posits that energy distribution becomes uniform over time in a closed system, we recognize that, without external input, node representations in a graph converge to identical feature vectors as diffusion progresses. To address this issue, we propose the Active Diffusion-based Graph Neural Network (ADGNN). ADGNN achieves active diffusion by integrating multiple external information sources that dynamically influence the diffusion process, effectively overcoming the over-smoothing problem. Furthermore, our approach realizes true infinite diffusion by directly calculating the closed-form solution of the active diffusion iterative formula. This allows nodes to preserve their unique characteristics while efficiently gaining comprehensive insights into the graph's global structure. We evaluate ADGNN against several state-of-the-art GNN models across various graph tasks. The results demonstrate that ADGNN significantly improves both accuracy and efficiency, highlighting its effectiveness in capturing global graph information and maintaining node distinctiveness.

---

## 20. Learning noisy tissue dynamics across time scales

**论文链接:** [http://arxiv.org/abs/2510.19090v1](http://arxiv.org/abs/2510.19090v1)

**作者:** Ming Han, John Devany, Michel Fruchart, Margaret L. Gardel, Vincenzo Vitelli

**发布时间:** 2025-10-21

**备注:** 15 pages, 6 figures

### GPT解析

### 总结

研究团队开发了一种仿生机器学习框架，能够直接从实验电影中推断噪声多细胞动力学，成功应用于上皮组织、果蝇翅膀发育和ERK波介导的细胞信号过程。

### 背景

组织动力学在从伤口愈合到形态发生的生物过程中起着关键作用，但这些噪声多细胞动力学难以预测。

### 目的

引入一个能够直接从实验电影中推断噪声多细胞动力学的仿生机器学习框架。

### 方法

该生成模型结合了图神经网络、归一化流和WaveNet算法，将组织表示为神经随机微分方程，其中细胞是 evolving graph 的边。

### 主要发现

该机器学习架构反映了底层生物组织的架构，大大减少了训练所需的数据量；该模型能捕捉随机细胞运动并预测细胞在分裂周期中状态的演变；可以准确生成发育系统和细胞信号过程的实验动力学。

### 结论

该方法为在生物工程和临床环境中作为数字孪生使用铺平了道路。

### 翻译

组织动力学在从伤口愈合到形态发生的生物过程中起着关键作用。然而，这些噪声多细胞动力学难以预测。在此，我们引入了一种仿生机器学习框架，能够直接从实验电影中推断噪声多细胞动力学。该生成模型结合了图神经网络、归一化流和WaveNet算法，将组织表示为神经随机微分方程，其中细胞是 evolving graph 的边。与卷积或全连接神经网络相比，该机器学习架构反映了底层生物组织的架构，大大减少了训练所需的数据量。以上皮组织实验为例，我们表明该模型不仅能捕捉随机细胞运动，还能预测细胞在分裂周期中状态的演变。最后，我们证明了该方法可以准确生成发育系统（如果蝇翅膀）和由随机ERK波介导的细胞信号过程的实验动力学，为在生物工程和临床环境中作为数字孪生使用铺平了道路。


### 论文摘要

Tissue dynamics play a crucial role in biological processes ranging from wound healing to morphogenesis. However, these noisy multicellular dynamics are notoriously hard to predict. Here, we introduce a biomimetic machine learning framework capable of inferring noisy multicellular dynamics directly from experimental movies. This generative model combines graph neural networks, normalizing flows and WaveNet algorithms to represent tissues as neural stochastic differential equations where cells are edges of an evolving graph. This machine learning architecture reflects the architecture of the underlying biological tissues, substantially reducing the amount of data needed to train it compared to convolutional or fully-connected neural networks. Taking epithelial tissue experiments as a case study, we show that our model not only captures stochastic cell motion but also predicts the evolution of cell states in their division cycle. Finally, we demonstrate that our method can accurately generate the experimental dynamics of developmental systems, such as the fly wing, and cell signaling processes mediated by stochastic ERK waves, paving the way for its use as a digital twin in bioengineering and clinical contexts.

---

## 21. Committors without Descriptors

**论文链接:** [http://arxiv.org/abs/2510.18018v2](http://arxiv.org/abs/2510.18018v2)

**作者:** Peilin Kang, Jintu Zhang, Enrico Trizio, TingJun Hou, Michele Parrinello

**发布时间:** 2025-10-20

### GPT解析

### 总结

该研究提出了一种基于committor的增强采样方法，用于解决原子模拟中的稀有事件研究挑战。通过结合图神经网络技术，实现了对系统亚稳态之间频繁转换的促进，并对过程过渡态集合进行了广泛采样。

### 背景

稀有事件研究是原子模拟中的主要挑战之一，已经提出了几种增强采样方法来解决这一问题。最近有研究建议使用committor（提供对稀有事件的精确形式化描述）来解决这一挑战。

### 目的

进一步自动化基于committor的方法，通过将其与图神经网络的强大表达能力相结合，使方法能够直接处理原子坐标而非描述符，并展示基于图的方法在描述溶剂分子在离子对解离或配体结合等系统中作用方面的优势。

### 方法

提出了一种基于committor的方法，该方法促进系统亚稳态之间的频繁转换，并允许对过程过渡态集合进行广泛采样。该方法的特点是自洽和半自动，利用变分准则迭代优化基于神经网络的committor参数化，使用一组物理描述符作为输入。进一步通过将先前方法与图神经网络的强大表达能力相结合，直接处理原子坐标而非描述符。

### 主要发现

基于committor的方法能够促进系统亚稳态之间的频繁转换；该方法允许对过程过渡态集合进行广泛采样；结合图神经网络使方法更加自动化；基于图的方法在描述溶剂分子在离子对解离或配体结合等系统中作用方面具有优势。

### 结论

通过将基于committor的方法与图神经网络相结合，研究成功实现了方法的进一步自动化，并展示了基于图的方法在描述溶剂分子在特定系统中的角色方面的优势，为原子模拟中稀有事件的研究提供了新思路。

### 翻译

稀有事件研究是原子模拟中的主要挑战之一，已经提出了几种解决这一问题的增强采样方法。最近有研究建议使用committor（提供对稀有事件的精确形式化描述）来解决这一挑战。我们最近跟进这一建议，提出了一种基于committor的方法，该方法促进系统亚稳态之间的频繁转换，并允许对过程过渡态集合进行广泛采样。我们方法的优势之一是自洽和半自动，利用变分准则迭代优化基于神经网络的committor参数化，使用一组物理描述符作为输入。在这里，我们通过将先前方法与图神经网络的强大表达能力相结合，进一步自动化了这一过程，图神经网络可以直接处理原子坐标而非描述符。除了在基准系统上的应用外，我们强调了基于图的方法在描述离子对解离或配体结合等系统中溶剂分子作用方面的优势。


### 论文摘要

The study of rare events is one of the major challenges in atomistic simulations, and several enhanced sampling methods towards its solution have been proposed. Recently, it has been suggested that the use of the committor, which provides a precise formal description of rare events, could be of use in this context. We have recently followed up on this suggestion and proposed a committor-based method that promotes frequent transitions between the metastable states of the system and allows extensive sampling of the process transition state ensemble. One of the strengths of our approach is being self-consistent and semi-automatic, exploiting a variational criterion to iteratively optimize a neural-network-based parametrization of the committor, which uses a set of physical descriptors as input. Here, we further automate this procedure by combining our previous method with the expressive power of graph neural networks, which can directly process atomic coordinates rather than descriptors. Besides applications on benchmark systems, we highlight the advantages of a graph-based approach in describing the role of solvent molecules in systems, such as ion pair dissociation or ligand binding.

---

## 22. AutoScape: Geometry-Consistent Long-Horizon Scene Generation

**论文链接:** [http://arxiv.org/abs/2510.20726v1](http://arxiv.org/abs/2510.20726v1)

**作者:** Jiacheng Chen, Ziyu Jiang, Mingfu Liang, Bingbing Zhuang, Jong-Chyi Su, Sparsh Garg, Ying Wu, Manmohan Chandraker

**发布时间:** 2025-10-23

**备注:** ICCV 2025. Project page: https://auto-scape.github.io

### GPT解析

### 总结

AutoScape是一个长时程驾驶场景生成框架，通过创新的RGB-D扩散模型生成几何一致的关键帧，并使用视频扩散模型生成连贯的驾驶视频。

### 背景

在自动驾驶和场景生成领域，需要生成长时间、几何一致的驾驶场景，这是一个具有挑战性的任务。

### 目的

开发一个能够生成长时间（超过20秒）、真实且几何一致的驾驶视频的框架，解决现有方法在长时程场景生成中的局限性。

### 方法

AutoScape框架包含一个RGB-D扩散模型，该模型在共享潜在空间中联合处理图像和深度，基于先前生成的关键帧场景几何条件，并使用一致的引导来引导采样过程。给定高质量RGB-D关键帧后，视频扩散模型在关键帧之间进行插值。

### 主要发现

AutoScape能够生成超过20秒的真实且几何一致的驾驶视频，相比之前的最先进方法，在长时程FID和FVD评分上分别提高了48.6%和43.0%。

### 结论

AutoScape通过创新的RGB-D扩散模型和视频插值方法，显著提高了长时程驾驶场景生成的质量和一致性，为自动驾驶模拟和训练提供了更真实的数据来源。

### 翻译

本文提出了AutoScape，一个长时程驾驶场景生成框架。其核心是一个创新的RGB-D扩散模型，迭代生成稀疏的、几何一致的关键帧，作为场景外观和几何的可靠锚点。为了保持长距离几何一致性，模型1)在共享潜在空间中联合处理图像和深度，2)显式地基于先前生成的关键帧的场景几何（即渲染的点云）进行条件化，3)使用一致的引导来引导采样过程。给定高质量的RGB-D关键帧后，视频扩散模型在它们之间进行插值，生成密集且连贯的视频帧。AutoScape生成超过20秒的真实且几何一致的驾驶视频，相比之前的最先进方法，长时程FID和FVD评分分别提高了48.6%和43.0%。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决如何生成长时间（20秒以上）且保持3D几何一致性的驾驶场景视频问题。这个问题在自动驾驶领域至关重要，因为自动驾驶系统需要大量高质量、长时间一致的场景数据进行安全可靠的测试和验证，而当前方法在长时间生成时难以保持几何一致性，限制了实际应用价值。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者将长时间场景生成问题分解为稀疏关键帧生成和密集帧插值两个子问题，采用层次化方法解决。核心洞察是几何一致性退化是长时间生成的关键瓶颈，因此需要显式建模几何信息。方法借鉴了扩散模型在图像生成中的成功应用、RGB-D联合建模、ControlNet的条件控制机制以及视频扩散模型，但创新性地设计了RGB-D扩散模型、几何条件机制和Warp Consistent Guidance来提高几何一致性。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是通过层次化方法解决长时间场景生成：先生成稀疏但几何一致的关键帧作为场景锚点，然后在关键帧间插值生成密集视频帧。整体流程分为两个阶段：1)关键帧生成阶段：从输入图像反投影为3D点云，投影到下一视角生成渲染点和掩码，使用RGB-D扩散模型生成新关键帧，迭代添加到点云集合，并用Warp Consistent Guidance提高一致性；2)插值阶段：使用视频扩散模型在关键帧间生成密集视频帧，条件于从关键帧渲染的3D点云确保几何一致性。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)层次化框架，分解为关键帧生成和插值；2)RGB-D扩散模型，联合建模颜色和深度；3)显式几何条件，将渲染点云作为条件输入；4)Warp Consistent Guidance，减少误差累积；5)两阶段训练策略。相比WonderJourney和Vista等之前工作，AutoScape专门设计的RGB-D扩散模型具有更好的几何感知能力，通过关键帧锚点确保长期一致性，并显式利用几何信息而非仅依赖时间一致性模块。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'AutoScape提出了一种创新的层次化框架，通过RGB-D扩散模型生成几何一致的关键帧，并结合视频插值实现了长达20秒的高质量、3D一致的驾驶场景视频生成，显著提升了长时间场景生成的质量和一致性。'}


### 论文摘要

This paper proposes AutoScape, a long-horizon driving scene generation framework. At its core is a novel RGB-D diffusion model that iteratively generates sparse, geometrically consistent keyframes, serving as reliable anchors for the scene's appearance and geometry. To maintain long-range geometric consistency, the model 1) jointly handles image and depth in a shared latent space, 2) explicitly conditions on the existing scene geometry (i.e., rendered point clouds) from previously generated keyframes, and 3) steers the sampling process with a warp-consistent guidance. Given high-quality RGB-D keyframes, a video diffusion model then interpolates between them to produce dense and coherent video frames. AutoScape generates realistic and geometrically consistent driving videos of over 20 seconds, improving the long-horizon FID and FVD scores over the prior state-of-the-art by 48.6\% and 43.0\%, respectively.

---

## 23. ALICE-LRI: A General Method for Lossless Range Image Generation for Spinning LiDAR Sensors without Calibration Metadata

**论文链接:** [http://arxiv.org/abs/2510.20708v1](http://arxiv.org/abs/2510.20708v1)

**作者:** Samuel Soutullo, Miguel Yermo, David L. Vilariño, Óscar G. Lorenzo, José C. Cabaleiro, Francisco F. Rivera

**发布时间:** 2025-10-23

### GPT解析

### 总结

本研究提出了一种名为ALICE-LRI的新型方法，实现了从旋转LiDAR点云生成无损距离图像，无需制造商元数据或校准文件。该方法通过自动推断LiDAR传感器的内几何参数，实现了零点损失的点云投影和重建。

### 背景

3D LiDAR传感器在自主导航、环境监测和遥感应用中至关重要。为了高效处理这些传感器产生的大量点云数据，LiDAR数据通常被投影到2D距离图像中，这些图像根据点的角度位置和距离来组织点。然而，传统的投影方法存在基本的几何不一致性，导致不可逆的信息丢失，影响高保真应用。

### 目的

开发一种通用的、与传感器无关的方法，能够从旋转LiDAR点云生成无损距离图像，无需制造商元数据或校准文件，并保持几何精度在传感器精度范围内。

### 方法

ALICE-LRI是一种自动LiDAR内标定估计方法，能够自动逆向工程任何旋转LiDAR传感器的内几何。该方法通过推断关键参数，包括激光束配置、角度分布和每束校准校正，实现无损投影和完整的点云重建，零点损失。

### 主要发现

在完整的KITTI和DurLAR数据集上的全面评估表明，ALICE-LRI实现了完美的点保留，所有点云中都没有点丢失。几何精度保持在传感器精度范围内，建立了具有实时性能的几何无损性。压缩案例研究验证了显著的下游效益，展示了实际应用中的显著质量改进。

### 结论

从近似到无损LiDAR投影的范式转变，为需要完整几何保存的高精度遥感应用开辟了新的可能性。ALICE-LRI方法代表了LiDAR数据处理领域的重要进展，能够在不损失信息的情况下实现高效处理。

### 翻译

3D LiDAR传感器对于自主导航、环境监测和遥感应用中的精密制图至关重要。为了高效处理这些传感器产生的大量点云数据，LiDAR数据通常被投影到2D距离图像中，这些图像根据点的角度位置和距离来组织点。虽然这些距离图像表示能够实现高效处理，但传统的投影方法存在基本的几何不一致性，导致不可逆的信息丢失，影响高保真应用。我们提出了ALICE-LRI（无损距离图像的自动LiDAR内标定估计），这是第一个通用的、与传感器无关的方法，能够从旋转LiDAR点云生成无损距离图像，无需制造商元数据或校准文件。我们的算法通过推断关键参数（包括激光束配置、角度分布和每束校准校正）来自动逆向工程任何旋转LiDAR传感器的内几何，实现无损投影和零点损失的完整点云重建。在完整的KITTI和DurLAR数据集上的全面评估表明，ALICE-LRI实现了完美的点保留，所有点云中都没有点丢失。几何精度保持在传感器精度范围内，建立了具有实时性能的几何无损性。我们还介绍了压缩案例研究，验证了显著的下游效益，展示了实际应用中的显著质量改进。从近似到无损LiDAR投影的范式转变，为需要完整几何保存的高精度遥感应用开辟了新的可能性。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文解决的是如何在不依赖制造商提供的校准元数据的情况下，从旋转式激光雷达(LiDAR)点云生成无损的2D距离图像。这个问题很重要，因为LiDAR传感器在自动驾驶、环境监测和遥感等领域至关重要，而传统投影方法存在几何不一致性导致信息损失，会影响高精度应用的质量。许多实际场景中我们只有已校准的点云数据，而没有原始传感器数据或校准文件，限制了现有方法的应用。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先分析了理想传感器模型和实际传感器模型之间的差异，认识到需要推断出厂校准的几何参数。他们设计了ALICE-LRI方法，包含参数估计和无损投影两个主要阶段。参数估计又分为垂直和水平参数估计，使用Hough变换识别候选扫描线，加权最小二乘法进行拟合，以及冲突解决机制确保一致性。作者借鉴了Hough变换用于特征提取的技术和加权最小二乘法处理异方差噪声，但解决了现有校准研究的逆问题——从已校准点云推断参数而非从原始数据估计参数。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是自动反向推断LiDAR传感器的内在几何结构，无需制造商元数据，直接从点云数据推断关键参数，生成与传感器几何完全一致的距离图像，实现点云完全重建。整体流程分为：1)垂直参数估计：使用Hough变换识别候选扫描线参数，选择一致点，加权最小二乘拟合，解决冲突；2)水平参数估计：对每束进行分辨率穷举搜索，计算水平和方位角偏移，对点数不足的扫描线使用启发式方法；3)距离图像生成：使用估计参数将点云投影到2D图像，确保双射性和完全重建。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)首个通用方法实现从已校准点云生成无损距离图像，无需元数据；2)自动推断传感器内在几何参数；3)实现完全零点损失和几何无损；4)具有实时性能；5)提供开源实现。相比之前工作，传统方法依赖制造商提供的查找表或数据包信息，现有校准研究处理正向问题而非逆问题，大多数方法接受轻微几何失真，而ALICE-LRI实现了完全无损且通用。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'ALICE-LRI首次实现了无需传感器元数据即可从旋转式LiDAR点云生成完全无损的距离图像，通过自动推断传感器内在几何参数解决了传统投影方法中的信息损失问题，为高精度遥感应用提供了新的可能性。'}


### 论文摘要

3D LiDAR sensors are essential for autonomous navigation, environmental monitoring, and precision mapping in remote sensing applications. To efficiently process the massive point clouds generated by these sensors, LiDAR data is often projected into 2D range images that organize points by their angular positions and distances. While these range image representations enable efficient processing, conventional projection methods suffer from fundamental geometric inconsistencies that cause irreversible information loss, compromising high-fidelity applications. We present ALICE-LRI (Automatic LiDAR Intrinsic Calibration Estimation for Lossless Range Images), the first general, sensor-agnostic method that achieves lossless range image generation from spinning LiDAR point clouds without requiring manufacturer metadata or calibration files. Our algorithm automatically reverse-engineers the intrinsic geometry of any spinning LiDAR sensor by inferring critical parameters including laser beam configuration, angular distributions, and per-beam calibration corrections, enabling lossless projection and complete point cloud reconstruction with zero point loss. Comprehensive evaluation across the complete KITTI and DurLAR datasets demonstrates that ALICE-LRI achieves perfect point preservation, with zero points lost across all point clouds. Geometric accuracy is maintained well within sensor precision limits, establishing geometric losslessness with real-time performance. We also present a compression case study that validates substantial downstream benefits, demonstrating significant quality improvements in practical applications. This paradigm shift from approximate to lossless LiDAR projections opens new possibilities for high-precision remote sensing applications requiring complete geometric preservation.

---

## 24. PointMapPolicy: Structured Point Cloud Processing for Multi-Modal Imitation Learning

**论文链接:** [http://arxiv.org/abs/2510.20406v1](http://arxiv.org/abs/2510.20406v1)

**作者:** Xiaogang Jia, Qian Wang, Anrui Wang, Han A. Wang, Balázs Gyenes, Emiliyan Gospodinov, Xinkai Jiang, Ge Li, Hongyi Zhou, Weiran Liao, Xi Huang, Maximilian Beck, Moritz Reuss, Rudolf Lioutikov, Gerhard Neumann

**发布时间:** 2025-10-23

### GPT解析

### 总结

这篇论文提出了PointMapPolicy，一种新颖的机器人操作方法，通过将点云组织成结构化网格并结合RGB数据，实现了高效的多模态感知，在多种操作任务中达到了最先进的性能。

### 背景

机器人操作系统受益于互补的传感模态，其中每种模态提供独特的环境信息。点云能捕获详细的几何结构，而RGB图像提供丰富的语义上下文。然而，当前点云方法难以捕获细粒度细节，特别是对于复杂任务；而RGB方法缺乏几何意识，限制了其精度和泛化能力。

### 目的

开发一种新型方法，结合点云和RGB图像的优势，解决现有方法在几何细节和语义理解方面的局限性，提高机器人操作系统的性能和泛化能力。

### 方法

作者提出了PointMapPolicy，一种将扩散策略基于未下采样的结构化点网格的新方法。这种方法创建的数据类型更容易从观测中提取形状和空间关系，并且可以在参考帧之间转换。由于点网格的结构规整，可以直接使用成熟的计算机视觉技术处理3D数据。模型使用xLSTM作为骨干网络，高效融合点图与RGB数据以增强多模态感知。

### 主要发现

在RoboCasa和CALVIN基准测试以及真实机器人评估上的大量实验表明，该方法在各种操作任务中实现了最先进的性能。

### 结论

PointMapPolicy有效结合了点云和RGB数据的优势，通过结构化点网格和多模态融合，显著提升了机器人操作系统的性能，特别是在需要精细几何理解和语义上下文的复杂任务中。

### 翻译

机器人操作系统受益于互补的传感模态，每种模态提供独特的环境信息。点云捕获详细的几何结构，而RGB图像提供丰富的语义上下文。当前点云方法难以捕获细粒度细节，特别是对于复杂任务，而RGB方法缺乏几何意识，这限制了它们的精度和泛化能力。我们引入了PointMapPolicy，一种新颖的方法，将扩散策略基于未下采样的结构化点网格。产生的数据类型更容易从观测中提取形状和空间关系，并且可以在参考帧之间转换。但由于它们在规则网格中的结构，我们能够直接使用成熟的计算机视觉技术处理3D数据。使用xLSTM作为骨干网络，我们的模型高效地将点图与RGB数据融合以增强多模态感知。在RoboCasa和CALVIN基准测试以及真实机器人评估的大量实验中，我们证明我们的方法在各种操作任务中实现了最先进的性能。概述和演示可在我们的项目页面查看：https://point-map.github.io/Point-Map/

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决机器人系统中多模态感知的挑战，即如何同时利用点云提供的几何信息和RGB图像提供的语义信息。这个问题很重要，因为机器人执行复杂任务时需要同时理解场景的几何结构和语义内容，而现有的方法要么缺乏几何细节（仅用RGB图像），要么难以处理精细细节（仅用点云），限制了机器人在复杂环境中的精确操作能力和泛化能力。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者受到计算机视觉社区在立体重建方面的最新进展启发，提出将点云转换为结构化的点图表示。他们借鉴了立体重建技术中的点图方法，将其应用于机器人模仿学习领域。设计过程中，他们考虑了点云和RGB图像的优缺点，创建了可以与标准视觉架构兼容的点图表示，并探索了多种融合策略来结合几何和语义信息。同时，他们借鉴了xLSTM架构作为骨干网络，平衡了时间建模能力和计算效率。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是将无结构的点云转换为结构化的点图表示，使其能够与标准视觉架构兼容，同时融合点图的几何信息和RGB图像的语义信息。整体实现流程包括：1)将深度图转换为结构化的点图表示，编码为2D网格中的XYZ坐标；2)使用预训练的视觉编码器处理RGB图像和点图；3)采用晚期融合策略(如拼接)来结合多模态信息；4)使用xLSTM作为骨干网络处理多模态输入；5)基于EDM框架应用扩散策略生成动作序列；6)通过少量去噪步骤(4步)生成最终动作。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)提出点图(point maps)这一新的观察模态，首次在基于扩散的模仿学习中使用；2)将点云结构化为规则的2D网格，可直接应用标准视觉架构，无需KNN和FPS等昂贵操作；3)设计高效的多模态融合策略，平衡几何精度和语义理解；4)使用xLSTM作为骨干网络，相比Transformer具有更高的计算效率。相比之前的工作，PointMapPolicy不需要复杂的点云处理步骤，能同时利用几何和语义信息，且计算效率更高，在多个基准测试中取得了最先进性能。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'PointMapPolicy通过结构化点云处理和多模态融合，在机器人模仿学习中实现了几何精度与语义理解的平衡，并在多个基准测试中取得了最先进性能。'}


### 论文摘要

Robotic manipulation systems benefit from complementary sensing modalities, where each provides unique environmental information. Point clouds capture detailed geometric structure, while RGB images provide rich semantic context. Current point cloud methods struggle to capture fine-grained detail, especially for complex tasks, which RGB methods lack geometric awareness, which hinders their precision and generalization. We introduce PointMapPolicy, a novel approach that conditions diffusion policies on structured grids of points without downsampling. The resulting data type makes it easier to extract shape and spatial relationships from observations, and can be transformed between reference frames. Yet due to their structure in a regular grid, we enable the use of established computer vision techniques directly to 3D data. Using xLSTM as a backbone, our model efficiently fuses the point maps with RGB data for enhanced multi-modal perception. Through extensive experiments on the RoboCasa and CALVIN benchmarks and real robot evaluations, we demonstrate that our method achieves state-of-the-art performance across diverse manipulation tasks. The overview and demos are available on our project page: https://point-map.github.io/Point-Map/

---

## 25. NeuralTouch: Neural Descriptors for Precise Sim-to-Real Tactile Robot Control

**论文链接:** [http://arxiv.org/abs/2510.20390v1](http://arxiv.org/abs/2510.20390v1)

**作者:** Yijiong Lin, Bowen Deng, Chenghua Lu, Max Yang, Efi Psomopoulou, Nathan F. Lepora

**发布时间:** 2025-10-23

### GPT解析

### 总结

本文介绍了NeuralTouch，一个集成神经描述场（NDF）和触觉传感的多模态框架，通过轻柔的物理交互实现准确、可推广的机器人抓取。

### 背景

抓取精度对精确物体操作至关重要，通常需要机器人手与物体仔细对齐。NDF是一种基于视觉的生成抓取姿势的方法，但单独使用时可能因相机校准不完美、点云不完整和物体变化而产生不准确姿势。触觉传感虽能实现更精确接触，但现有方法通常仅限于简单、预定义的接触几何形状。

### 目的

开发一个集成NDF和触觉传感的框架，通过轻柔的物理交互实现准确、可推广的抓取，解决视觉方法在精确抓取方面的局限性。

### 方法

利用NDF隐式表示目标接触几何形状，训练深度强化学习策略使用触觉反馈来优化抓取。该策略以神经描述符为条件，无需明确指定接触类型。通过模拟中的消融研究和零样本迁移到现实世界任务（如销钉插入孔和瓶盖开启）进行验证，无需额外微调。

### 主要发现

NeuralTouch显著提高了抓取精度和鲁棒性，优于基线方法，为精确、接触丰富的机器人操作提供了一个通用框架。

### 结论

通过结合视觉（NDF）和触觉传感，NeuralTouch实现了准确、可推广的抓取，为需要精确接触的机器人操作任务提供了有效解决方案。

### 翻译

抓取精度是精确物体操作的关键前提，通常需要机器人手与物体之间的仔细对齐。神经描述场（NDF）提供了一种有前景的基于视觉的方法，可生成跨物体类别的抓取姿势。然而，仅靠NDF可能因不完美的相机校准、不完整的点云和物体变化而产生不准确姿势。同时，触觉传感能实现更精确接触，但现有方法通常学习限于简单、预定义接触几何形状的策略。在这项工作中，我们介绍了NeuralTouch，一个集成NDF和触觉传感的多模态框架，通过轻柔的物理交互实现准确、可推广的抓取。我们的方法利用NDF隐式表示目标接触几何形状，基于此训练深度强化学习策略，使用触觉反馈来优化抓取。该策略以神经描述符为条件，不需要明确指定接触类型。我们通过模拟中的消融研究和零样本迁移到现实世界操作任务（如销钉插入孔和瓶盖开启）来验证NeuralTouch，无需额外微调。结果表明，NeuralTouch显著提高了抓取精度和鲁棒性，优于基线方法，为精确、接触丰富的机器人操作提供了一个通用框架。


### 论文摘要

Grasping accuracy is a critical prerequisite for precise object manipulation, often requiring careful alignment between the robot hand and object. Neural Descriptor Fields (NDF) offer a promising vision-based method to generate grasping poses that generalize across object categories. However, NDF alone can produce inaccurate poses due to imperfect camera calibration, incomplete point clouds, and object variability. Meanwhile, tactile sensing enables more precise contact, but existing approaches typically learn policies limited to simple, predefined contact geometries. In this work, we introduce NeuralTouch, a multimodal framework that integrates NDF and tactile sensing to enable accurate, generalizable grasping through gentle physical interaction. Our approach leverages NDF to implicitly represent the target contact geometry, from which a deep reinforcement learning (RL) policy is trained to refine the grasp using tactile feedback. This policy is conditioned on the neural descriptors and does not require explicit specification of contact types. We validate NeuralTouch through ablation studies in simulation and zero-shot transfer to real-world manipulation tasks--such as peg-out-in-hole and bottle lid opening--without additional fine-tuning. Results show that NeuralTouch significantly improves grasping accuracy and robustness over baseline methods, offering a general framework for precise, contact-rich robotic manipulation.

---

## 26. AnyPcc: Compressing Any Point Cloud with a Single Universal Model

**论文链接:** [http://arxiv.org/abs/2510.20331v1](http://arxiv.org/abs/2510.20331v1)

**作者:** Kangli Wang, Qianxi Yi, Yuqi Ye, Shihao Li, Wei Gao

**发布时间:** 2025-10-23

**备注:** 11 pages, 5 figures

### GPT解析

### 总结

本研究提出了一种名为AnyPcc的通用点云压缩框架，解决了深度学习点云几何压缩中的泛化性问题。该框架通过通用上下文模型和实例自适应微调策略，有效处理了分布外数据，并在15个数据集上实现了最先进的压缩性能。

### 背景

基于深度学习的点云几何压缩面临泛化性挑战，主要原因是缺乏鲁棒的上下文模型和对分布外数据的低效处理。

### 目的

开发一个通用点云压缩框架AnyPcc，以解决点云几何压缩中的泛化性问题，特别是上下文建模和分布外数据处理方面的挑战。

### 方法

AnyPcc包含两个主要组件：1) 通用上下文模型，利用空间和通道分组的先验信息捕获鲁棒的上下文依赖关系；2) 实例自适应微调策略，通过协同显式和隐式压缩范式处理分布外数据，为每个实例微调一小部分网络权重并整合到位流中。

### 主要发现

在15个不同数据集上的广泛实验表明，AnyPcc在点云压缩方面设立了新的最先进水平，证明了其有效性和优越性。

### 结论

AnyPcc成功解决了点云几何压缩中的泛化性问题，通过创新的通用上下文模型和实例自适应微调策略，实现了更好的压缩性能，为点云压缩领域提供了新的解决方案。

### 翻译

泛化性是基于深度学习的点云几何压缩的一个关键挑战。我们认为这源于两个关键限制：缺乏鲁棒的上下文模型和对分布外数据的低效处理。为解决这两个问题，我们引入了AnyPcc，一个通用的点云压缩框架。AnyPcc首先采用通用上下文模型，利用空间和通道分组的先验信息来捕获鲁棒的上下文依赖关系。其次，我们新颖的实例自适应微调策略通过协同显式和隐式压缩范式来处理分布外数据。它为每个实例微调一小部分网络权重，并将它们整合到位流中，其中权重的边际位成本远小于几何压缩带来的节省。在15个不同数据集组成的基准测试上的大量实验证实，AnyPcc在点云压缩方面设立了新的最先进水平。我们的代码和数据集将被发布以鼓励可重复研究。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决点云压缩中的泛化挑战。现有方法通常针对特定点云密度设计，无法在真实世界中各种密度的点云上保持稳定性能，并且在遇到分布外数据时压缩效率急剧下降。这个问题很重要，因为随着自动驾驶和虚拟现实等应用中3D内容的普及，点云已成为主要数据表示形式，对高效压缩算法有迫切需求，而真实世界的点云数据具有广泛的多样性，许多关键类型（如医学扫描、3D高斯溅射）通常没有专用的训练数据。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先识别出现有方法的局限性：空间上下文模型在密集数据上表现好但在稀疏场景中性能差，而通道级模型虽能处理稀疏输入但牺牲了空间信息。为了解决这一权衡，他们设计了通用上下文模型(UCM)协同整合空间和通道先验。对于分布外数据问题，他们借鉴了参数高效微调技术，结合了隐式神经表示的优点和预训练模型的效率。作者借鉴了图像压缩中的上下文建模技术，但将其应用于几何占用码，还参考了参数高效微调方法，但这些技术在点云压缩领域尚未被探索。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是结合通用强大的预训练模型与快速的实例特定适应。整体流程包括：1)UCM采用从粗到细的层次结构，通过空间-通道上下文分解处理点云数据；2)使用3D棋盘模式将体素分为两组(G1和G2)，并在每组内将8位占用码分解为两个4位子符号；3)通过协同特征聚合增强上下文；4)IAFT策略只微调网络的一小部分参数(最终预测头)，实现快速实例适应；5)最终压缩位流包含微调后的权重和几何组件两部分；6)解码时先重建模型参数，再对几何数据进行解码。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)AnyPcc框架是首个使用单一统一模型实现高压缩率和跨多样化点云类型鲁棒性能的方法；2)UCM通过创新的棋盘分组策略协同整合空间和通道先验，建立跨越所有点云密度的鲁棒上下文建模；3)IAFT策略通过只微调预训练模型的一小部分参数，实现快速实例特定适应；4)实现了统一的无损和有损压缩。相比之前的工作，AnyPcc解决了类别特定方法的局限性，避免了Unicorn等泛化尝试的非统一架构问题，克服了隐式压缩的高计算成本，并解决了现有上下文模型在密集和稀疏数据之间的权衡问题。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'AnyPcc通过引入通用上下文模型和实例自适应微调策略，首次实现了使用单一统一模型在各种点云类型上达到最先进的压缩性能，解决了点云压缩中长期存在的泛化挑战。'}


### 论文摘要

Generalization remains a critical challenge for deep learning-based point cloud geometry compression. We argue this stems from two key limitations: the lack of robust context models and the inefficient handling of out-of-distribution (OOD) data. To address both, we introduce AnyPcc, a universal point cloud compression framework. AnyPcc first employs a Universal Context Model that leverages priors from both spatial and channel-wise grouping to capture robust contextual dependencies. Second, our novel Instance-Adaptive Fine-Tuning (IAFT) strategy tackles OOD data by synergizing explicit and implicit compression paradigms. It fine-tunes a small subset of network weights for each instance and incorporates them into the bitstream, where the marginal bit cost of the weights is dwarfed by the resulting savings in geometry compression. Extensive experiments on a benchmark of 15 diverse datasets confirm that AnyPcc sets a new state-of-the-art in point cloud compression. Our code and datasets will be released to encourage reproducible research.

---

## 27. VO-DP: Semantic-Geometric Adaptive Diffusion Policy for Vision-Only Robotic Manipulation

**论文链接:** [http://arxiv.org/abs/2510.15530v3](http://arxiv.org/abs/2510.15530v3)

**作者:** Zehao Ni, Yonghao He, Lingfeng Qian, Jilei Mao, Fa Fu, Wei Sui, Hu Su, Junran Peng, Zhipeng Wang, Bin He

**发布时间:** 2025-10-17

### GPT解析

### 总结

本研究提出了一种纯视觉单视图扩散策略学习方法(VO-DP)，通过融合预训练视觉基础模型的语义和几何特征，在机器人操作任务中取得了优异性能，特别是在真实世界任务中明显优于现有方法。

### 背景

在模仿学习领域，基于视觉运动的扩散策略学习是机器人操作的主要方向。现有方法多依赖点云作为输入，通过点云特征学习构建场景表示，但缺乏对纯视觉解决方案的深入探索，而纯视觉方案具有显著潜力。

### 目的

提出一种纯视觉且单视图的扩散策略学习方法(VO-DP)，利用预训练的视觉基础模型实现语义和几何特征的有效融合，以提高机器人操作的性能。

### 方法

利用VGGT的中间特征，结合来自DINOv2的语义特征和来自交替注意力块的几何特征，通过交叉注意力融合特征，并使用CNN进行空间压缩，形成策略头的输入。

### 主要发现

VO-DP在仿真任务中达到64.6%的平均成功率，与基于点云的方法DP3(64.0%)相当，远高于纯视觉基线DP(34.8%)；在真实世界任务中达到87.9%的成功率，显著优于DP3(67.5%)和DP(11.2%)。鲁棒性评估表明VO-DP在各种变化条件下保持高度稳定。

### 结论

提出的VO-DP方法在机器人操作任务中表现出色，特别是在真实世界任务中。研究团队还开源了一个支持多机器和多GPU并行训练的机器人操作训练库，兼容多种视觉运动策略。

### 翻译

在模仿学习的背景下，基于视觉运动的扩散策略学习是机器人操作的主要方向之一。大多数方法依赖点云作为观察输入，通过点云特征学习构建场景表示，从而实现显著的准确性。然而，现有文献缺乏对具有显著潜力的纯视觉解决方案的深入探索。在本文中，我们提出了一种纯视觉且单视图的扩散策略学习方法(VO-DP)，利用预训练的视觉基础模型实现语义和几何特征的有效融合。我们利用VGGT的中间特征，结合来自DINOv2的语义特征和来自交替注意力块的几何特征。特征通过交叉注意力融合，并使用CNN进行空间压缩，形成策略头的输入。大量实验证明，VO-DP不仅显著优于纯视觉基线DP，而且与基于点云的方法DP3相比表现出明显的性能趋势：在仿真任务中，VO-DP的平均成功率为64.6%，与DP3的64.0%相当，远高于DP的34.8%；而在真实世界任务中，它达到87.9%，明显优于DP3的67.5%和DP的11.2%。进一步的鲁棒性评估证实，VO-DP在颜色、大小、背景和光照等变化条件下保持高度稳定。最后，我们开源了一个机器人操作训练库。该库基于Accelerate构建，支持多机器和多GPU并行训练以及混合精度训练。它与DP、DP3和VO-DP等视觉运动策略兼容，并支持RoboTwin模拟器。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决的是如何仅使用RGB图像（vision-only）作为输入，实现与基于点云的方法相媲美的机器人操作性能。这个问题重要是因为当前主流机器人操作方法依赖昂贵的深度传感器（如深度相机或LiDAR），导致硬件成本高、系统复杂（需要多传感器校准），且在复杂场景中表现受限。相比之下，仅使用RGB图像的方法成本低得多、实用性高，但现有研究显示其性能通常不如基于点云的方法。提升vision-only方法的性能具有显著的实际应用价值，可以大幅降低机器人系统的部署成本和复杂度。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先分析了当前机器人操作领域的研究现状，指出vision-only方法成本低但性能不足，而基于点云的方法性能好但成本高。他们评估了现有方法，发现vision-only方法的瓶颈主要在于表示学习模块不够发达。作者认识到视觉基础模型（如VGGT、DINOv2）的潜力，这些模型可以直接从RGB图像中提取几何信息。因此，他们提出将语义特征和几何特征进行有效融合的思路，并在不依赖3D传感器的情况下获得丰富的场景理解。该方法借鉴了多个现有工作：使用预训练的VGGT模型作为视觉编码器，利用DINOv2进行语义特征提取，采用Alternating Attention网络进行几何特征提取，并借鉴了扩散模型（Diffusion Policy）的思想进行动作生成。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是通过预训练的视觉基础模型，从单视角RGB图像中同时提取语义和几何特征，并将这些特征自适应融合，从而在不依赖3D传感器的情况下实现高性能的机器人操作。整体实现流程包括四个主要步骤：1) 视觉编码：使用DINOv2提取语义特征，通过VGGT模型的Alternating Attention网络提取几何特征；2) 特征融合：使用残差交叉注意力机制将语义和几何特征融合，并通过前馈网络进一步处理；3) 场景表示压缩：使用轻量级ResNet对融合特征进行下采样和空间压缩，然后将压缩后的空间特征与本体感受信息连接，形成紧凑的场景表示；4) 动作生成：使用基于去噪扩散概率模型（DDPM）的策略头，以压缩后的场景表示为条件，通过迭代去噪过程预测动作轨迹。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1) 首次证明了仅使用RGB图像的vision-only方法可以达到与基于点云方法相媲美的性能水平；2) 设计了基于交叉注意力的特征融合模块，能够根据任务需求自适应地融合语义和几何特征；3) 通过空间特征压缩模块，从高维特征中提取关键信息，实现高效的单视角场景表示；4) 开源了DRRM训练库，支持多机多GPU并行训练和混合精度训练。相比之前的工作，VO-DP不再依赖点云或RGB-D等3D输入，仅使用RGB图像；利用预训练的视觉基础模型提取更丰富的特征；设计了专门的特征融合机制；在保持高性能的同时，显著降低了硬件成本和系统复杂度。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'VO-DP通过创新性地融合预训练视觉模型的语义和几何特征，首次实现了仅使用低成本RGB图像输入的机器人操作方法达到与基于昂贵3D传感器方法相媲美的性能，同时显著提升了在真实世界环境中的鲁棒性。'}


### 论文摘要

In the context of imitation learning, visuomotor-based diffusion policy learning is one of the main directions in robotic manipulation. Most of these approaches rely on point clouds as observation inputs and construct scene representations through point clouds feature learning, which enables them to achieve remarkable accuracy. However, the existing literature lacks an in-depth exploration of vision-only solutions that have significant potential. In this paper, we propose a Vision-Only and single-view Diffusion Policy learning method (VO-DP) that leverages pretrained visual foundation models to achieve effective fusion of semantic and geometric features. We utilize intermediate features from VGGT incorporating semantic features from DINOv2 and geometric features from Alternating Attention blocks. Features are fused via cross-attention and spatially compressed with a CNN to form the input to the policy head. Extensive experiments demonstrate that VO-DP not only outperforms the vision-only baseline DP significantly but also exhibits distinct performance trends against the point cloud-based method DP3: in simulation tasks, VO-DP achieves an average success rate of 64.6% on par with DP3 64.0% and far higher than DP 34.8%, while in real-world tasks, it reaches 87.9%, outperforming both DP3 67.5% and DP 11.2% by a notable margin. Further robustness evaluations confirm that VO-DP remains highly stable under varying conditions including color, size, background, and lighting. Lastly, we open-source a training library for robotic manipulation. Built on Accelerate, this library supports multi-machine and multi-GPU parallel training, as well as mixed precision training. It is compatible with visuomotor policies such as DP, DP3 and VO-DP, and also supports the RoboTwin simulator.

---

## 28. On Optimal Hyperparameters for Differentially Private Deep Transfer Learning

**论文链接:** [http://arxiv.org/abs/2510.20616v1](http://arxiv.org/abs/2510.20616v1)

**作者:** Aki Rehn, Linzh Zhao, Mikko A. Heikkilä, Antti Honkela

**发布时间:** 2025-10-23

**备注:** 25 pages, 30 figures

### GPT解析

### 总结

该研究探讨了差分隐私迁移学习中的关键超参数选择问题，特别关注裁剪边界C和批量大小B，揭示了理论理解与实际结果之间的不匹配，并提出改进方法。

### 背景

差分隐私迁移学习（即在私有数据上微调预训练模型）是当前在隐私约束下训练大型模型的最先进方法。然而，在关键超参数选择方面存在理论与实践的脱节。

### 目的

研究差分隐私迁移学习中两个关键超参数（裁剪边界C和批量大小B）的最优选择方法，解决当前理论理解与实证结果之间的不匹配问题。

### 方法

分析裁剪边界C和批量大小B对模型性能的影响，考察梯度分布的变化，在固定计算预算（固定周期）下评估现有启发式方法，分析累积DP噪声对批量大小选择的影响，研究跨任务使用单一(C,B)设置的效果，并分析裁剪作为梯度重加权形式和累积DP噪声的作用。

### 主要发现

1. 当前关于如何选择最优C的理论理解（更强的隐私需要更小的C）与实证结果（更强的隐私下更大的C表现更好）之间存在明显不匹配，这是由梯度分布变化引起的。2. 在计算预算有限的情况下，现有的调整批量大小B的启发式方法不适用，而累积DP噪声能更好地解释较小或较大批量的表现差异。3. 跨任务使用单一(C,B)设置会导致次优性能，特别是在从宽松隐私转向严格隐私以及从充足计算转向有限计算的情况下。

### 结论

差分隐私迁移学习中的超参数选择需要考虑梯度分布变化和累积DP噪声的影响，不应简单地跨任务应用相同的超参数设置，而应根据隐私需求和计算资源进行针对性调整。

### 翻译

差分隐私（DP）迁移学习，即在私有数据上微调预训练模型，是当前在隐私约束下训练大型模型的最先进方法。我们关注此设置中的两个关键超参数：裁剪边界C和批量大小B。我们展示了当前关于如何选择最优C的理论理解（更强的隐私需要更小的C）与实证结果（更强的隐私下更大的C表现更好）之间的明显不匹配，这是由梯度分布变化引起的。假设计算预算有限（固定周期），我们证明了现有的调整B的启发式方法不适用，而累积DP噪声能更好地解释较小或较大批量的表现差异。我们还强调了跨任务使用单一(C,B)设置的常见做法可能导致次优性能。我们发现，当在宽松与严格隐私之间转换以及在充足与有限计算之间转换时，性能下降特别明显，我们通过分析裁剪作为梯度重加权形式和检查累积DP噪声来解释这一现象。


### 论文摘要

Differentially private (DP) transfer learning, i.e., fine-tuning a pretrained model on private data, is the current state-of-the-art approach for training large models under privacy constraints. We focus on two key hyperparameters in this setting: the clipping bound $C$ and batch size $B$. We show a clear mismatch between the current theoretical understanding of how to choose an optimal $C$ (stronger privacy requires smaller $C$) and empirical outcomes (larger $C$ performs better under strong privacy), caused by changes in the gradient distributions. Assuming a limited compute budget (fixed epochs), we demonstrate that the existing heuristics for tuning $B$ do not work, while cumulative DP noise better explains whether smaller or larger batches perform better. We also highlight how the common practice of using a single $(C,B)$ setting across tasks can lead to suboptimal performance. We find that performance drops especially when moving between loose and tight privacy and between plentiful and limited compute, which we explain by analyzing clipping as a form of gradient re-weighting and examining cumulative DP noise.

---

## 29. Reliable and Reproducible Demographic Inference for Fairness in Face Analysis

**论文链接:** [http://arxiv.org/abs/2510.20482v1](http://arxiv.org/abs/2510.20482v1)

**作者:** Alexandre Fournier-Montgieux, Hervé Le Borgne, Adrian Popescu, Bertrand Luvison

**发布时间:** 2025-10-23

### GPT解析

### 总结

该研究提出了一个模块化迁移学习方法的人口统计属性推断流水线，以提高面部分析系统公平性评估的可靠性。该方法通过整合预训练的人脸识别编码器与非线性分类头，在性别和种族推断任务上超越了基线方法，特别是在更具挑战性的种族属性上表现优异。研究还引入了通过身份一致性定义的鲁棒性指标，适用于任何人口统计分割方案。

### 背景

面部分析系统中的公平性评估通常依赖于自动人口统计属性推断，而人口统计属性推断又依赖于预定义的人口统计分割。公平性审计的有效性取决于DAI过程的可靠性，但这一问题在以往研究中未得到充分重视。

### 目的

提高DAI的可靠性，从而获得更少偏差和更低方差的面部分析系统公平性估计；提出一个完全可复现的DAI流水线；为公平审计中的人口统计推断提供可靠基础。

### 方法

用模块化迁移学习方法替代传统的端到端训练；整合预训练的人脸识别编码器与非线性分类头；从准确性、公平性和新引入的鲁棒性（通过身份一致性定义）三个维度评估流水线；在多个数据集和训练设置上对性别和种族推断进行基准测试。

### 主要发现

提出的模块化迁移学习方法在性别和种族推断上优于强大的基线方法；在更具挑战性的种族属性上表现尤其出色；新引入的鲁棒性指标适用于任何人口统计分割方案。

### 结论

该工作为公平审计中的人口统计推断提供了可靠的基础；通过公开训练数据集元数据、完整代码库、预训练模型和评估工具包，促进了研究的透明度和可复现性。

### 翻译

面部分析系统中的公平性评估通常依赖于自动人口统计属性推断，而人口统计属性推断本身又依赖于预定义的人口统计分割。然而，公平性审计的有效性取决于DAI过程的可靠性。我们首先提供了这种依赖关系的理论动机，表明提高DAI可靠性可以减少偏差并降低面部分析系统公平性估计的方差。为此，我们提出了一个完全可复现的DAI流水线，用模块化迁移学习方法替代传统的端到端训练。我们的设计整合了预训练的人脸识别编码器与非线性分类头。我们从三个维度评估了这个流水线：准确性、公平性，以及通过身份一致性定义的新引入的鲁棒性概念。所提出的鲁棒性指标适用于任何人口统计分割方案。我们在多个数据集和训练设置上对性别和种族推断进行了基准测试。我们的结果表明，所提出的方法优于强大的基线方法，特别是在更具挑战性的种族属性上。为了促进透明度和可复现性，我们将公开训练数据集元数据、完整代码库、预训练模型和评估工具包。这项工作为公平审计中的人口统计推断提供了可靠的基础。


### 论文摘要

Fairness evaluation in face analysis systems (FAS) typically depends on automatic demographic attribute inference (DAI), which itself relies on predefined demographic segmentation. However, the validity of fairness auditing hinges on the reliability of the DAI process. We begin by providing a theoretical motivation for this dependency, showing that improved DAI reliability leads to less biased and lower-variance estimates of FAS fairness. To address this, we propose a fully reproducible DAI pipeline that replaces conventional end-to-end training with a modular transfer learning approach. Our design integrates pretrained face recognition encoders with non-linear classification heads. We audit this pipeline across three dimensions: accuracy, fairness, and a newly introduced notion of robustness, defined via intra-identity consistency. The proposed robustness metric is applicable to any demographic segmentation scheme. We benchmark the pipeline on gender and ethnicity inference across multiple datasets and training setups. Our results show that the proposed method outperforms strong baselines, particularly on ethnicity, which is the more challenging attribute. To promote transparency and reproducibility, we will publicly release the training dataset metadata, full codebase, pretrained models, and evaluation toolkit. This work contributes a reliable foundation for demographic inference in fairness auditing.

---

## 30. Machine learning identification of fractional-order vortex beam diffraction process

**论文链接:** [http://arxiv.org/abs/2510.20245v1](http://arxiv.org/abs/2510.20245v1)

**作者:** Yan Guo, Heng Lyu, Chunling Ding, Chenzhi Yuan, Ruibo Jin

**发布时间:** 2025-10-23

**DOI:** 10.7498/aps.74.20241458

**备注:** 14 pages, 5 figures

### GPT解析

### 总结

本文提出了一种基于ResNet的深度学习方法，用于在衍射条件下准确识别分数阶涡旋光束的传播距离和拓扑荷，考虑了大气湍流的影响，实现了高精度的识别。

### 背景

分数阶涡旋光束具有分数阶轨道角动量(FOAM)模式，理论上可以无限增加传输容量，在测量、光通信和微粒操纵等领域有重要应用前景。然而，当分数阶涡旋光束在自由空间传播时，其螺旋相位的连续性使其在实际应用中容易受到衍射的影响，从而影响OAM模式识别的准确性，严重限制了基于FOAM的光通信的使用。

### 目的

实现衍射条件下分数阶涡旋光束的机器学习识别，解决目前尚未报道的紧急问题。

### 方法

基于ResNet，提出了一种深度学习方法。利用实验测量和数值模拟的强度分布，创建了大气湍流环境中涡旋光束衍射强度模式的数据集。采用基于迁移学习的改进101层ResNet结构，实现不同传播距离下FOAM模型的准确高效识别。

### 主要发现

该方法可以在湍流条件下准确识别传播距离为100厘米、间距为5厘米、模式间距为0.1的FOAM模式，准确率达到99.69%。该方法考虑了空间传输过程中大气湍流的影响，使得识别方案即使在特殊环境中也能实现高精度。它具有区分超细FOAM模式和传播距离的能力，这是传统方法无法实现的。

### 结论

所提出的方法解决了分数阶涡旋光束在衍射条件下识别的难题，特别是在考虑大气湍流影响的情况下，实现了高精度的FOAM模式识别，为实际应用提供了新的可能性。

### 翻译

分数阶涡旋光束具有分数阶轨道角动量(FOAM)模式，理论上具有无限增加传输容量的潜力。因此，它们在测量、光通信和微粒操纵领域具有重要的应用前景。然而，当分数阶涡旋光束在自由空间传播时，螺旋相位的连续性使它们在实际应用中容易受到衍射的影响，从而影响OAM模式识别的准确性，严重限制了基于FOAM的光通信的使用。实现衍射条件下分数阶涡旋光束的机器学习识别目前是一个紧迫且尚未报道的问题。在本工作中，基于ResNet，提出了一种深度学习(DL)方法，用于准确识别分数阶涡旋光束衍射过程中的传播距离和拓扑荷。利用实验测量和数值模拟的强度分布，创建了大气湍流环境中涡旋光束衍射强度模式的数据集。采用基于迁移学习的改进101层ResNet结构，实现不同传播距离下FOAM模型的准确高效识别。实验结果表明，所提出的方法可以在湍流条件下准确识别传播距离为100厘米、间距为5厘米、模式间距为0.1的FOAM模式，准确率为99.69%。该方法考虑了空间传输过程中大气湍流的影响，使得识别方案即使在特殊环境中也能实现高精度。它具有区分超细FOAM模式和传播距离的能力，这是传统方法无法实现的。


### 论文摘要

Fractional-order vortex beams possess fractional orbital angular momentum (FOAM) modes, which theoretically have the potential to increase transmission capacity infinitely. Therefore, they have significant application prospects in the fields of measurement, optical communication and micro-particle manipulation. However, when fractional-order vortex beams propagate in free space, the discontinuity of the helical phase makes them susceptible to diffraction in practical applications, thereby affecting the accuracy of OAM mode recognition and severely limiting the use of FOAM-based optical communication. Achieving machine learning recognition of fractional-order vortex beams under diffraction conditions is currently an urgent and unreported issue. Based on ResNet, a deep learning (DL) method of accurately recognizing the propagation distance and topological charge of fractional-order vortex beam diffraction process is proposed in this work. Utilizing both experimentally measured and numerically simulated intensity distributions, a dataset of vortex beam diffraction intensity patterns in atmospheric turbulence environments is created. An improved 101-layer ResNet structure based on transfer learning is employed to achieve accurate and efficient recognition of the FOAM model at different propagation distances. Experimental results show that the proposed method can accurately recognize FOAM modes with a propagation distance of 100 cm, a spacing of 5 cm, and a mode spacing of 0.1 under turbulent conditions, with an accuracy of 99.69%. This method considers the effect of atmospheric turbulence during spatial transmission, allowing the recognition scheme to achieve high accuracy even in special environments. It has the ability to distinguish ultra-fine FOAM modes and propagation distances, which cannot be achieved by traditional methods.

---

## 31. Improving Transfer Learning for Sequence Labeling Tasks by Adapting Pre-trained Neural Language Models

**论文链接:** [http://arxiv.org/abs/2510.20033v1](http://arxiv.org/abs/2510.20033v1)

**作者:** David Dukić

**发布时间:** 2025-10-22

### GPT解析

### 总结

这篇博士论文提出了三种改进序列标注任务迁移学习的方法，通过调整预训练神经语言模型来提高性能

### 背景

序列标注任务的迁移学习需要更有效的预训练语言模型适应方法

### 目的

改进序列标注任务的迁移学习方法，使预训练神经语言模型能够更好地适应特定任务

### 方法

提出了三种改进方法：1) 引入额外信号的多任务模型；2) 修改自回归大语言模型架构以实现层间双向信息流；3) 利用监督上下文微调结合响应导向适应策略的序列标注框架

### 主要发现

预训练神经语言模型在有针对性的迁移学习范式下，在序列标注任务上能够达到最佳性能

### 结论

通过有针对性的迁移学习范式调整预训练神经语言模型，可以显著提高其在序列标注任务上的性能

### 翻译

这篇博士论文通过调整预训练的神经语言模型，改进了序列标注任务的迁移学习。所提出的迁移学习改进包括引入一个额外信号的多任务模型、基于自回归大语言模型架构修改的方法，以及利用监督上下文微调结合响应导向适应策略的自回归大语言模型序列标注框架。第一个改进是在事件触发检测任务的领域迁移背景下提出的，通过将独立于领域的文本处理系统获得的额外信号整合到多任务模型中来改进领域迁移。第二个改进涉及修改模型架构，为此提出了一个方法，使自回归大语言模型的层之间能够实现双向信息流。第三个改进利用自回归大语言模型作为文本生成器，通过生成式监督上下文微调框架实现。所提出的模型、方法和框架表明，当通过有针对性的迁移学习范式进行调整时，预训练的神经语言模型在序列标注任务上能够达到最佳性能。


### 论文摘要

This doctoral thesis improves the transfer learning for sequence labeling tasks by adapting pre-trained neural language models. The proposed improvements in transfer learning involve introducing a multi-task model that incorporates an additional signal, a method based on architectural modifications in autoregressive large language models, and a sequence labeling framework for autoregressive large language models utilizing supervised in-context fine-tuning combined with response-oriented adaptation strategies. The first improvement is given in the context of domain transfer for the event trigger detection task. The domain transfer of the event trigger detection task can be improved by incorporating an additional signal obtained from a domain-independent text processing system into a multi-task model. The second improvement involves modifying the model's architecture. For that purpose, a method is proposed to enable bidirectional information flow across layers of autoregressive large language models. The third improvement utilizes autoregressive large language models as text generators through a generative supervised in-context fine-tuning framework. The proposed model, method, and framework demonstrate that pre-trained neural language models achieve their best performance on sequence labeling tasks when adapted through targeted transfer learning paradigms.

---

## 32. Novel Class Discovery for Point Cloud Segmentation via Joint Learning of Causal Representation and Reasoning

**论文链接:** [http://arxiv.org/abs/2510.13307v2](http://arxiv.org/abs/2510.13307v2)

**作者:** Yang Li, Aming Wu, Zihao Zhang, Yahong Han

**发布时间:** 2025-10-15

**备注:** Accepted by NeurIPS 2025

### GPT解析

### 总结

本文提出了一种基于因果表示和推理的联合学习方法，用于解决点云分割中的新类别发现问题

### 背景

在点云分割任务中，需要学习一个模型，仅使用已标记的基础类别监督信息，来分割未标记的新类别点云

### 目的

建立点表示与基础类别标签之间的精确相关性，以及基础类别与新类别点之间的表示相关性

### 方法

引入结构因果模型(SCM)重新形式化3D-NCD问题，分析基础类别表示中的隐藏混杂因素，设计消除混杂因素的因果表示原型，并使用图结构建模基础类别与新类别之间的因果关系

### 主要发现

粗略或统计相关性学习可能导致新类别推理的混淆，而施加因果关系作为强相关约束可以准确揭示对应类别的本质点云表示

### 结论

在3D和2D NCD语义分割任务上的大量实验和可视化结果证明了该方法的优势

### 翻译

在本文中，我们专注于点云分割的新类别发现(3D-NCD)，旨在学习一个模型，仅使用已标记的基础3D类别的监督信息，来分割未标记的新3D类别。此任务的关键在于建立点表示与其基础类别标签之间的精确相关性，以及基础类别与新类别点之间的表示相关性。粗略或统计相关性学习可能导致新类别推理的混淆。如果我们将因果关系作为强相关约束强加于学习过程，应该能够准确揭示对应于类别的本质点云表示。为此，我们引入结构因果模型(SCM)重新形式化3D-NCD问题，并提出一种新方法，即因果表示和推理的联合学习。具体而言，我们首先通过SCM分析基础类别表示中的隐藏混杂因素以及基础类别与新类别之间的因果关系。我们设计了一个消除混杂因素的因果表示原型，以捕获基础类别的因果表示。然后使用图结构建模基础类别的因果表示原型与新类别原型之间的因果关系，实现从基础类别到新类别的因果推理。在3D和2D NCD语义分割上的大量实验和可视化结果证明了我们方法的优势

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文解决点云分割中的新类别发现问题，即如何仅使用已标记的基础类别数据来分割未标记的新类别。这个问题在自动驾驶、机器人感知等实际应用中非常重要，因为这些场景中环境是动态开放的，可能会突然出现新的物体类别，而传统方法无法处理这些未预先定义的类别，限制了系统在真实世界中的实用性。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': "作者首先分析了传统方法的局限性：它们本质上是统计模型，倾向于学习'捷径特征'而非本质特征，且难以处理新类别。因此，作者引入结构因果模型(SCM)重新形式化问题，识别出混杂因素对基础类别学习的干扰，以及基础到新类别的因果路径。方法设计借鉴了点云分割领域常用的MinkowskiUNet架构、因果学习理论中的独立因果机制原则、对抗训练思想以及图神经网络技术，将它们创新性地结合来解决3D-NCD问题。", '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是通过因果表示学习识别和去除点云中的非因果特征(混杂因素)，学习到能准确对应类别的本质表示，并利用基础类别的因果表示通过图结构建模基础到新类别的因果关系，实现因果推理。整体流程包括：1)因果表示原型学习，通过对抗训练去除混杂因素，提取基础类别的因果表示并生成原型；2)构建因果推理图，使用自注意力机制调整边权重，应用因果剪枝和推理方向一致性约束；3)基于图卷积网络生成高质量伪标签，实现新类别的分割。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点有三：1)首次将因果学习引入3D-NCD领域；2)提出因果表示原型学习方法，通过对抗机制消除混杂因素；3)设计基于图的因果推理框架，显式建模类别间因果关系。相比之前工作，不同之处在于：传统方法依赖统计相似性且易受捷径特征干扰，而本文方法通过因果学习提取本质特征；传统方法直接测量类别相似性，而本文使用图结构建模复杂的高阶依赖关系；实验表明本文方法在多个数据集上显著优于现有方法，特别是在新类别分割任务上。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '本文提出了一种结合因果表示学习和推理的新方法，通过识别和去除点云中的非因果特征并建模基础到新类别的因果关系，显著提升了点云分割中新类别的发现和分割能力。'}


### 论文摘要

In this paper, we focus on Novel Class Discovery for Point Cloud Segmentation (3D-NCD), aiming to learn a model that can segment unlabeled (novel) 3D classes using only the supervision from labeled (base) 3D classes. The key to this task is to setup the exact correlations between the point representations and their base class labels, as well as the representation correlations between the points from base and novel classes. A coarse or statistical correlation learning may lead to the confusion in novel class inference. lf we impose a causal relationship as a strong correlated constraint upon the learning process, the essential point cloud representations that accurately correspond to the classes should be uncovered. To this end, we introduce a structural causal model (SCM) to re-formalize the 3D-NCD problem and propose a new method, i.e., Joint Learning of Causal Representation and Reasoning. Specifically, we first analyze hidden confounders in the base class representations and the causal relationships between the base and novel classes through SCM. We devise a causal representation prototype that eliminates confounders to capture the causal representations of base classes. A graph structure is then used to model the causal relationships between the base classes' causal representation prototypes and the novel class prototypes, enabling causal reasoning from base to novel classes. Extensive experiments and visualization results on 3D and 2D NCD semantic segmentation demonstrate the superiorities of our method.

---

## 33. Amplifying Prominent Representations in Multimodal Learning via Variational Dirichlet Process

**论文链接:** [http://arxiv.org/abs/2510.20736v1](http://arxiv.org/abs/2510.20736v1)

**作者:** Tsai Hor Chan, Feng Wu, Yihang Chen, Guosheng Yin, Lequan Yu

**发布时间:** 2025-10-23

**备注:** Accepted by NeruIPS 2025

### GPT解析

### 总结

提出了一种基于狄利克雷过程(DP)的多模态学习框架，通过DP的'富者愈富'特性自动实现显著的模态内表示学习和跨模态对齐之间的最佳平衡，有效解决了多模态融合中保持特征表达能力和学习跨模态交互的挑战。

### 背景

多模态融合在医疗保健和金融等现实世界场景中变得越来越重要，关键挑战是如何在保持每个模态特征表达能力的同时学习跨模态交互。

### 目的

开发一种新的多模态学习方法，避免过度强调模态边缘分布对齐带来的问题，同时保持每个模态内的有意义表示。

### 方法

提出DP驱动的多模态学习框架，假设每个模态遵循多元高斯分布的混合，并采用狄利克雷过程计算所有组件的混合权重，利用其'富者愈富'特性动态分配特征贡献并选择最突出的特征。

### 主要发现

在多个多模态数据集上的实验表明，该模型优于其他竞争方法；消融分析验证了DP在模态分布对齐中的有效性及其对关键超参数变化的鲁棒性。

### 结论

DP驱动的多模态学习框架能够自动实现显著的模态内表示学习和跨模态对齐之间的最佳平衡，有效解决了多模态融合中的关键挑战。

### 翻译

开发有效的多模态融合方法在医疗保健和金融等现实世界场景中变得越来越重要。关键挑战是如何在保持每个模态特征表达能力的同时学习跨模态交互。先前的方法主要关注跨模态对齐，但过度强调模态边缘分布的对齐可能会施加过度的正则化，并阻碍每个模态内的有意义表示。狄利克雷过程(DP)混合模型是一种强大的贝叶斯非参数方法，可以通过其'富者愈富'特性放大最突出的特征，为它们分配不断增加的权重。受DP这一独特特性的启发，我们提出了一种新的DP驱动的多模态学习框架，自动实现显著的模态内表示学习和跨模态对齐之间的最佳平衡。具体而言，我们假设每个模态遵循多元高斯分布的混合，并进一步采用DP计算所有组件的混合权重。这种范式允许DP动态分配特征的贡献并选择最突出的特征，利用其'富者愈富'特性，从而促进多模态特征融合。在多个多模态数据集上的广泛实验证明了我们的模型优于其他竞争模型。消融分析进一步验证了DP在模态分布对齐中的有效性及其对关键超参数变化的鲁棒性。代码已在https://github.com/HKU-MedAI/DPMM.git匿名提供。


### 论文摘要

Developing effective multimodal fusion approaches has become increasingly essential in many real-world scenarios, such as health care and finance. The key challenge is how to preserve the feature expressiveness in each modality while learning cross-modal interactions. Previous approaches primarily focus on the cross-modal alignment, while over-emphasis on the alignment of marginal distributions of modalities may impose excess regularization and obstruct meaningful representations within each modality. The Dirichlet process (DP) mixture model is a powerful Bayesian non-parametric method that can amplify the most prominent features by its richer-gets-richer property, which allocates increasing weights to them. Inspired by this unique characteristic of DP, we propose a new DP-driven multimodal learning framework that automatically achieves an optimal balance between prominent intra-modal representation learning and cross-modal alignment. Specifically, we assume that each modality follows a mixture of multivariate Gaussian distributions and further adopt DP to calculate the mixture weights for all the components. This paradigm allows DP to dynamically allocate the contributions of features and select the most prominent ones, leveraging its richer-gets-richer property, thus facilitating multimodal feature fusion. Extensive experiments on several multimodal datasets demonstrate the superior performance of our model over other competitors. Ablation analysis further validates the effectiveness of DP in aligning modality distributions and its robustness to changes in key hyperparameters. Code is anonymously available at https://github.com/HKU-MedAI/DPMM.git

---

## 34. From Masks to Worlds: A Hitchhiker's Guide to World Models

**论文链接:** [http://arxiv.org/abs/2510.20668v1](http://arxiv.org/abs/2510.20668v1)

**作者:** Jinbin Bai, Yu Lei, Hecong Wu, Yuchen Zhu, Shufan Li, Yi Xin, Xiangtai Li, Molei Tao, Aditya Grover, Ming-Hsuan Yang

**发布时间:** 2025-10-23

**备注:** Github: https://github.com/M-E-AGI-Lab/Awesome-World-Models

### GPT解析

### 总结

本文不是典型的世界模型综述，而是面向世界构建者的指南，聚焦于从早期掩码模型到记忆增强系统的世界模型发展路径

### 背景

现有关于世界模型的文献分散且缺乏系统性，许多论文仅提及'世界模型'概念但未深入探讨

### 目的

提供一条清晰的世界模型发展路径，专注于核心组件而非列举所有相关研究

### 方法

遵循从跨模态表示学习的掩码模型，到统一架构，再到交互式生成模型，最后到记忆增强系统的发展脉络

### 主要发现

世界模型的核心在于三个关键组件：生成核心、交互循环和记忆系统

### 结论

通过关注这三个核心组件构建的系统是实现真正世界模型的最有前途路径

### 翻译

这不是一篇典型的世界模型综述；这是一份面向那些想要构建世界的人的指南。我们的目标不是罗列所有曾经提及'世界模型'的论文。相反，我们遵循一条清晰的道路：从早期跨模态统一表示学习的掩码模型，到共享单一范式的统一架构，再到闭合动作-感知循环的交互式生成模型，最后到随时间保持一致世界的记忆增强系统。我们绕过了松散相关的分支，专注于核心：生成核心、交互循环和记忆系统。我们表明这是实现真正世界模型的最有前途的路径。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': "论文主要解决如何构建真正的世界模型的问题。尽管有数百篇相关论文，但对于如何实际构建一个真正的世界模型还没有明确共识。这个问题很重要，因为真正的世界模型可以模拟整个环境，用于强化学习、智能体规划、大型语言模型模拟社会等多个领域。它能从预测引擎转变为'活的世界'，具有持久性、代理性和涌现性，对理解复杂系统、进行科学实验和创造交互体验具有重要意义。", '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': "作者通过分析世界模型的历史发展，提出了一条'狭窄的道路'，将世界模型发展分为五个阶段：基于掩码的模型、统一模型、交互式生成模型、记忆与一致性系统，以及真正的世界模型。论文大量借鉴了现有工作，每个阶段都列举了代表性模型和方法，如BERT、MAE等（第一阶段），EMU3、Chameleon等（第二阶段），Genie系列等（第三阶段），RETRO、MemGPT等（第四阶段）。作者通过分析这些工作的演进，提出了构建真正世界模型的路径。", '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是：真正的世界模型不是单一实体，而是由三个核心子系统合成的系统：生成核心（产生世界状态）、交互循环（实时关闭行动-感知循环）和持久记忆系统（随时间维持一致的世界）。整体实现流程遵循五个阶段：首先建立基于掩码的模型，为跨模态表示学习提供通用范式；然后统一架构，使单一架构能处理和生成多种模态；接着引入交互循环，将静态生成器转变为实时模拟器；然后添加记忆系统，使模拟能随时间持续；最后将这些组件合成为自主整体，实现真正的世界模型。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1）提出了清晰的世界模型五阶段发展路线图；2）定义了真正世界模型的三个核心子系统；3）指出了构建真正世界模型的三个基本挑战：一致性问题、压缩问题和对齐问题；4）提出了真正世界模型的三个定义属性：持久性、代理性和涌现性。与之前工作相比，不同之处在于：它不是简单罗列论文，而是提供清晰发展路径；不仅关注技术细节，还关注哲学意义和潜在影响；将世界模型视为综合系统而非孤立组件；提出了构建真正世界模型的具体挑战和未来方向。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': "这篇论文提供了一个从基于掩码的模型到真正世界模型的五阶段发展路线图，定义了真正世界模型的三个核心子系统和三个关键属性，并指出了构建真正世界模型的三个基本挑战，为构建能够持久、交互和涌现的'活的世界'提供了清晰的指南。"}


### 论文摘要

This is not a typical survey of world models; it is a guide for those who want to build worlds. We do not aim to catalog every paper that has ever mentioned a ``world model". Instead, we follow one clear road: from early masked models that unified representation learning across modalities, to unified architectures that share a single paradigm, then to interactive generative models that close the action-perception loop, and finally to memory-augmented systems that sustain consistent worlds over time. We bypass loosely related branches to focus on the core: the generative heart, the interactive loop, and the memory system. We show that this is the most promising path towards true world models.

---

## 35. Connecting Jensen-Shannon and Kullback-Leibler Divergences: A New Bound for Representation Learning

**论文链接:** [http://arxiv.org/abs/2510.20644v1](http://arxiv.org/abs/2510.20644v1)

**作者:** Reuben Dorent, Polina Golland, William Wells III

**发布时间:** 2025-10-23

**备注:** Accepted at NeurIPS 2025. Code available at  https://github.com/ReubenDo/JSDlowerbound/

### GPT解析

### 总结

该研究探讨了互信息(MI)与Jensen-Shannon散度(JSD)之间的关系，通过推导一个新的紧密且可处理的Kullback-Leibler散度(KLD)下界作为JSD的函数，建立了两者之间的理论联系。研究证明最大化基于JSD的信息会增加对互信息的保证下界，并通过实验验证了该方法的有效性和实用性，为在基于MI的表示学习中使用判别学习提供了理论依据和实证支持。

### 背景

互信息(MI)是表示学习中广泛使用的基本统计依赖性度量。然而，直接通过其定义为Kullback-Leibler散度(KLD)来优化MI通常是不可行的。因此，最近的方法转而最大化替代的依赖性度量，特别是Jensen-Shannon散度(JSD)作为判别损失。但这些替代目标与MI之间的联系尚未被充分理解。

### 目的

本研究旨在填补替代目标(特别是基于JSD的目标)与互信息之间理论理解的空白，通过建立它们之间的严格数学关系，为在表示学习中使用判别学习提供理论依据。

### 方法

研究通过推导一个新的、紧密且可处理的KLD下界作为JSD的函数来建立MI与JSD之间的理论联系。通过将这一边界专门应用于联合分布和边缘分布，证明了最大化基于JSD的信息会增加对互信息的保证下界。此外，研究重新审视了基于JSD目标的实际实现，并观察到最小化二元分类器的交叉熵损失可以恢复JSD的已知变分下界。

### 主要发现

1. 推导出了一个新的、紧密且可处理的KLD下界作为JSD的函数；2. 证明了最大化基于JSD的信息会增加对互信息的保证下界；3. 最小化区分联合分布与边缘分布对的二元分类器的交叉熵损失可以恢复JSD的已知变分下界；4. 实验表明该下界应用于MI估计时是紧密的；5. 与最先进的神经变分下界估计器相比，该下界估计器提供了稳定的低方差估计；6. 在信息瓶颈框架中展示了该方法的有效性。

### 结论

研究的结果为在基于互信息的表示学习中使用判别学习提供了新的理论依据和强有力的实证证据。所提出的下界估计器能够提供对互信息的稳定、低方差估计，并且在信息瓶颈框架中具有实际应用价值。

### 翻译

互信息(MI)是表示学习中广泛使用的基本统计依赖性度量。虽然直接通过其定义为Kullback-Leibler散度(KLD)来优化MI通常是不可行的，但最近的方法转而最大化替代的依赖性度量，特别是通过判别损失来最大化联合分布与边缘分布乘积之间的Jensen-Shannon散度(JSD)。然而，这些替代目标与MI之间的联系尚未被充分理解。在本工作中，我们通过推导一个新的、紧密且可处理的KLD下界作为JSD的函数来填补这一空白。通过将这一边界专门应用于联合分布和边缘分布，我们证明了最大化基于JSD的信息会增加对互信息的保证下界。此外，我们重新审视了基于JSD目标的实际实现，并观察到最小化训练以区分联合分布与边缘分布对的二元分类器的交叉熵损失可以恢复JSD的已知变分下界。广泛的实验表明该下界应用于MI估计时是紧密的。我们将该下界与一系列既定参考场景中最先进的神经变分下界估计器进行了比较。我们的下界估计器一致地提供了对互信息紧密下界的稳定、低方差估计。我们还展示了其在信息瓶颈框架中的实际实用性。综上所述，我们的结果为在基于MI的表示学习中使用判别学习提供了新的理论依据和强有力的实证证据。


### 论文摘要

Mutual Information (MI) is a fundamental measure of statistical dependence widely used in representation learning. While direct optimization of MI via its definition as a Kullback-Leibler divergence (KLD) is often intractable, many recent methods have instead maximized alternative dependence measures, most notably, the Jensen-Shannon divergence (JSD) between joint and product of marginal distributions via discriminative losses. However, the connection between these surrogate objectives and MI remains poorly understood. In this work, we bridge this gap by deriving a new, tight, and tractable lower bound on KLD as a function of JSD in the general case. By specializing this bound to joint and marginal distributions, we demonstrate that maximizing the JSD-based information increases a guaranteed lower bound on mutual information. Furthermore, we revisit the practical implementation of JSD-based objectives and observe that minimizing the cross-entropy loss of a binary classifier trained to distinguish joint from marginal pairs recovers a known variational lower bound on the JSD. Extensive experiments demonstrate that our lower bound is tight when applied to MI estimation. We compared our lower bound to state-of-the-art neural estimators of variational lower bound across a range of established reference scenarios. Our lower bound estimator consistently provides a stable, low-variance estimate of a tight lower bound on MI. We also demonstrate its practical usefulness in the context of the Information Bottleneck framework. Taken together, our results provide new theoretical justifications and strong empirical evidence for using discriminative learning in MI-based representation learning.

---

## 36. Diffusion Autoencoders with Perceivers for Long, Irregular and Multimodal Astronomical Sequences

**论文链接:** [http://arxiv.org/abs/2510.20595v1](http://arxiv.org/abs/2510.20595v1)

**作者:** Yunyi Shen, Alexander Gagliano

**发布时间:** 2025-10-23

### GPT解析

### 总结

这篇论文介绍了一种名为Diffusion Autoencoder with Perceivers (daep)的新架构，用于处理科学领域中不规则、多模态序列数据。该架构通过将异构测量值标记化，使用Perceiver编码器压缩，并使用Perceiver-IO扩散解码器重建，实现了在不同数据设置中的可扩展学习。研究还建立了maep作为基线模型，实验表明daep在重建误差、判别性潜在空间保存和精细结构保留方面均优于VAE和maep基线模型。

### 背景

自监督学习已成为表征学习的中心策略，但大多数用于编码数据的架构仅在规则采样的输入（如图像、音频和视频）上得到验证。然而，在许多科学领域中，数据是以长序列、不规则和多模态的形式出现的。

### 目的

为了从这些不规则、多模态序列数据中提取语义信息，作者提出了daep架构，并建立了一个强基线模型maep，以评估daep的性能。

### 方法

daep架构通过以下步骤工作：将异构测量值标记化，使用Perceiver编码器进行压缩，使用Perceiver-IO扩散解码器进行重建。为了评估daep，作者将掩码自编码器调整为Perceiver编码器/解码器设计，建立了maep基线模型。

### 主要发现

在多样化的光谱和光度天文数据集上，daep实现了比VAE和maep基线模型更低的重建误差，产生了更具判别性的潜在空间，并更好地保留了精细结构。

### 结论

这些结果表明daep是处理科学领域中不规则、异构序列数据的有效框架。

### 翻译

自监督学习已成为表征学习的中心策略，但大多数用于编码数据的架构仅在规则采样的输入（如图像、音频和视频）上得到验证。在许多科学领域中，数据实际上是以长序列、不规则和多模态的形式出现的。为了从这些数据中提取语义信息，我们引入了带有Perceiver的扩散自编码器（daep）。daep将异构测量值标记化，使用Perceiver编码器压缩它们，并使用Perceiver-IO扩散解码器重建它们，从而能够在各种数据设置中实现可扩展学习。为了对daep架构进行基准测试，我们将掩码自编码器调整为Perceiver编码器/解码器设计，并在与daep相同的架构家族中建立了强基线（maep）。在多样化的光谱和光度天文数据集上，daep实现了比VAE和maep基线模型更低的重建误差，产生更具判别性的潜在空间，并更好地保留了精细结构。这些结果确立了daep作为科学领域中数据以不规则、异构序列形式出现的有效框架。


### 论文摘要

Self-supervised learning has become a central strategy for representation learning, but the majority of architectures used for encoding data have only been validated on regularly-sampled inputs such as images, audios. and videos. In many scientific domains, data instead arrive as long, irregular, and multimodal sequences. To extract semantic information from these data, we introduce the Diffusion Autoencoder with Perceivers (daep). daep tokenizes heterogeneous measurements, compresses them with a Perceiver encoder, and reconstructs them with a Perceiver-IO diffusion decoder, enabling scalable learning in diverse data settings. To benchmark the daep architecture, we adapt the masked autoencoder to a Perceiver encoder/decoder design, and establish a strong baseline (maep) in the same architectural family as daep. Across diverse spectroscopic and photometric astronomical datasets, daep achieves lower reconstruction errors, produces more discriminative latent spaces, and better preserves fine-scale structure than both VAE and maep baselines. These results establish daep as an effective framework for scientific domains where data arrives as irregular, heterogeneous sequences.

---

## 37. Mitigating Cross-modal Representation Bias for Multicultural Image-to-Recipe Retrieval

**论文链接:** [http://arxiv.org/abs/2510.20393v1](http://arxiv.org/abs/2510.20393v1)

**作者:** Qing Wang, Chong-Wah Ngo, Yu Cao, Ee-Peng Lim

**发布时间:** 2025-10-23

**备注:** ACM Multimedia 2025

### GPT解析

### 总结

本文提出了一种新的因果方法，用于解决图像到食谱检索中的表示偏差问题，通过预测并注入图像中可能被忽视的烹饪元素，提高了检索性能。

### 背景

现有图像到食谱检索方法假设食物图像能完全捕捉食谱细节，但实际上图像只反映烹饪结果而非过程。这导致跨模态表示学习忽略视觉上不明显但对检索关键的细节，使表示偏向主要视觉元素，难以区分相似食谱。当训练数据包含不同菜系时，这种偏差更严重。

### 目的

提出一种因果方法，预测图像中可能被忽视的烹饪元素，并将这些元素明确注入跨模态表示学习中，以减轻偏差。

### 方法

采用因果方法预测图像中可能被忽视的烹饪元素，并将这些元素注入跨模态表示学习过程。在标准单语Recipe1M数据集和新策划的多语言多文化菜系数据集上进行实验验证。

### 主要发现

提出的因果表示学习能够揭示细微的成分和烹饪动作，在单语和多语言多文化数据集上都取得了令人印象深刻的检索性能。

### 结论

通过因果方法预测并注入图像中可能被忽视的烹饪元素，可以有效减轻跨模态表示学习中的偏差，特别是在处理不同菜系的图像和食谱时效果显著。

### 翻译

现有的图像到食谱检索方法隐含假设食物图像可以完全捕捉其食谱中文本记录的细节。然而，食物图像只反映了烹饪菜肴的视觉结果，而不是底层的烹饪过程。因此，学习跨模态表示来弥合图像和食谱之间的模态差距时，往往会忽略那些在视觉上不明显但对食谱检索至关重要的细微、特定于食谱的细节。具体来说，这些表示偏向于捕捉主要的视觉元素，导致难以对在使用成分和烹饪方法上有细微差异的相似食谱进行排序。当训练数据混合来自不同菜系的图像和食谱时，表示学习中的偏差预计会更严重。本文提出了一种新的因果方法，预测图像中可能被忽视的烹饪元素，同时明确地将这些元素注入跨模态表示学习中以减轻偏差。实验在标准的单语Recipe1M数据集和一个新策划的多语言多文化菜系数据集上进行。结果表明，提出的因果表示学习能够揭示细微的成分和烹饪动作，并在单语和多语言多文化数据集上都取得了令人印象深刻的检索性能。


### 论文摘要

Existing approaches for image-to-recipe retrieval have the implicit assumption that a food image can fully capture the details textually documented in its recipe. However, a food image only reflects the visual outcome of a cooked dish and not the underlying cooking process. Consequently, learning cross-modal representations to bridge the modality gap between images and recipes tends to ignore subtle, recipe-specific details that are not visually apparent but are crucial for recipe retrieval. Specifically, the representations are biased to capture the dominant visual elements, resulting in difficulty in ranking similar recipes with subtle differences in use of ingredients and cooking methods. The bias in representation learning is expected to be more severe when the training data is mixed of images and recipes sourced from different cuisines. This paper proposes a novel causal approach that predicts the culinary elements potentially overlooked in images, while explicitly injecting these elements into cross-modal representation learning to mitigate biases. Experiments are conducted on the standard monolingual Recipe1M dataset and a newly curated multilingual multicultural cuisine dataset. The results indicate that the proposed causal representation learning is capable of uncovering subtle ingredients and cooking actions and achieves impressive retrieval performance on both monolingual and multilingual multicultural datasets.

---

## 38. GUSL-Dehaze: A Green U-Shaped Learning Approach to Image Dehazing

**论文链接:** [http://arxiv.org/abs/2510.20266v1](http://arxiv.org/abs/2510.20266v1)

**作者:** Mahtab Movaheddrad, Laurence Palmer, C. -C. Jay Kuo

**发布时间:** 2025-10-23

### GPT解析

### 总结

GUSL-Dehaze是一种绿色U型学习方法的图像去雾技术，结合了基于物理的模型与绿色学习框架，避免了深度学习的高计算成本和大参数量问题。

### 背景

图像去雾是恢复清晰图像的任务，传统方法依赖统计先验和物理模型，而最先进的方法主要基于深度学习，但这些方法计算成本高且参数量大，不适合资源受限设备。

### 目的

开发一种轻量级、透明的图像去雾方法，避免深度学习的高计算成本，同时保持与最先进方法相当的性能。

### 方法

GUSL-Dehaze采用改进的暗通道先验进行初始去雾，然后通过U型架构实现绿色学习流程，使用无监督表示学习进行特征提取，并结合相关特征测试和最小二乘归一化变换等特征工程技术，最后通过透明的监督学习策略获得去雾图像。

### 主要发现

GUSL-Dehaze显著减少了参数数量，同时确保了数学可解释性，并取得了与最先进深度学习模型相当的性能。

### 结论

GUSL-Dehaze为图像去雾提供了一种轻量级、透明的替代方案，避免了深度学习的计算负担，同时保持了高性能和可解释性。

### 翻译

图像去雾是一项恢复任务，旨在从单幅有雾输入中恢复清晰图像。传统方法依赖于统计先验和基于物理的大气散射模型来重建无雾图像。虽然最近最先进的方法主要基于深度学习架构，但这些模型通常涉及高计算成本和大参数量，使其不适合资源受限设备。在本工作中，我们提出了GUSL-Dehaze，一种绿色U型学习方法的图像去雾技术。我们的方法将基于物理的模型与绿色学习框架相结合，提供了比传统深度学习技术更轻量、更透明的替代方案。与基于神经网络的解决方案不同，GUSL-Dehaze完全避免了深度学习。相反，我们首先使用改进的暗通道先验进行初始去雾步骤，然后通过U型架构实现绿色学习流程。该架构采用无监督表示学习进行有效特征提取，并结合相关特征测试和最小二乘归一化变换等特征工程技术来保持紧凑的模型大小。最后，通过透明的监督学习策略获得去雾图像。GUSL-Dehaze显著减少了参数数量，同时确保了数学可解释性，并取得了与最先进深度学习模型相当的性能。


### 论文摘要

Image dehazing is a restoration task that aims to recover a clear image from a single hazy input. Traditional approaches rely on statistical priors and the physics-based atmospheric scattering model to reconstruct the haze-free image. While recent state-of-the-art methods are predominantly based on deep learning architectures, these models often involve high computational costs and large parameter sizes, making them unsuitable for resource-constrained devices. In this work, we propose GUSL-Dehaze, a Green U-Shaped Learning approach to image dehazing. Our method integrates a physics-based model with a green learning (GL) framework, offering a lightweight, transparent alternative to conventional deep learning techniques. Unlike neural network-based solutions, GUSL-Dehaze completely avoids deep learning. Instead, we begin with an initial dehazing step using a modified Dark Channel Prior (DCP), which is followed by a green learning pipeline implemented through a U-shaped architecture. This architecture employs unsupervised representation learning for effective feature extraction, together with feature-engineering techniques such as the Relevant Feature Test (RFT) and the Least-Squares Normal Transform (LNT) to maintain a compact model size. Finally, the dehazed image is obtained via a transparent supervised learning strategy. GUSL-Dehaze significantly reduces parameter count while ensuring mathematical interpretability and achieving performance on par with state-of-the-art deep learning models.

---

## 39. Towards Objective Obstetric Ultrasound Assessment: Contrastive Representation Learning for Fetal Movement Detection

**论文链接:** [http://arxiv.org/abs/2510.20214v1](http://arxiv.org/abs/2510.20214v1)

**作者:** Talha Ilyas, Duong Nhu, Allison Thomas, Arie Levin, Lim Wei Yap, Shu Gong, David Vera Anaya, Yiwen Jiang, Deval Mehta, Ritesh Warty, Vinayak Smith, Maya Reddy, Euan Wallace, Wenlong Cheng, Zongyuan Ge, Faezeh Marzbanrad

**发布时间:** 2025-10-23

**备注:** This is the preprint version of the manuscript submitted to IEEE  Journal of Biomedical and Health Informatics (JBHI) for review

### GPT解析

### 总结

本文提出了一种名为CURL的新型自监督学习框架，用于从胎儿超声视频中准确检测胎儿运动，解决了传统方法的主观性和准确性有限的问题。

### 背景

准确的胎儿运动检测对于评估产前健康至关重要，异常的运动模式可能表明存在潜在的并发症，如胎盘功能障碍或胎儿窘迫。传统方法包括母亲感知和胎心宫缩监护图，但这些方法存在主观性和准确性有限的问题。

### 目的

为了解决传统胎儿运动检测方法的挑战，研究人员提出了一种新型自监督学习框架CURL，用于从延长的胎儿超声视频记录中检测胎儿运动。

### 方法

CURL方法利用双重对比损失，结合空间和时间对比学习，来学习鲁棒的运动表示。此外，研究还引入了一种特定任务的采样策略，确保在自监督训练过程中有效分离运动和非运动段，同时通过概率微调方法实现对任意长度的超声记录的灵活推断。

### 主要发现

在包含92名受试者（每人进行30分钟超声检查）的内部数据集上评估，CURL达到了78.01%的敏感性和81.60%的AUROC，证明了其在胎儿运动分析方面的可靠性和客观性。

### 结论

这些结果突显了自监督对比学习在胎儿运动分析中的潜力，为改进产前监测和临床决策铺平了道路。

### 翻译

准确的胎儿运动检测对于评估产前健康至关重要，因为异常的运动模式可能表明存在潜在的并发症，如胎盘功能障碍或胎儿窘迫。传统方法，包括母亲感知和胎心宫缩监护图，存在主观性和准确性有限的问题。为了解决这些挑战，我们提出了对比超声视频表示学习，这是一种新颖的自监督学习框架，用于从延长的胎儿超声视频记录中检测胎儿运动。我们的方法利用双重对比损失，结合空间和时间对比学习，来学习鲁棒的运动表示。此外，我们引入了一种特定任务的采样策略，确保在自监督训练过程中有效分离运动和非运动段，同时通过概率微调方法实现对任意长度的超声记录的灵活推断。在包含92名受试者（每人进行30分钟超声检查）的内部数据集上评估，CURL达到了78.01%的敏感性和81.60%的AUROC，证明了其在胎儿运动分析方面的可靠性和客观性。这些结果突显了自监督对比学习在胎儿运动分析中的潜力，为改进产前监测和临床决策铺平了道路。


### 论文摘要

Accurate fetal movement (FM) detection is essential for assessing prenatal health, as abnormal movement patterns can indicate underlying complications such as placental dysfunction or fetal distress. Traditional methods, including maternal perception and cardiotocography (CTG), suffer from subjectivity and limited accuracy. To address these challenges, we propose Contrastive Ultrasound Video Representation Learning (CURL), a novel self-supervised learning framework for FM detection from extended fetal ultrasound video recordings. Our approach leverages a dual-contrastive loss, incorporating both spatial and temporal contrastive learning, to learn robust motion representations. Additionally, we introduce a task-specific sampling strategy, ensuring the effective separation of movement and non-movement segments during self-supervised training, while enabling flexible inference on arbitrarily long ultrasound recordings through a probabilistic fine-tuning approach. Evaluated on an in-house dataset of 92 subjects, each with 30-minute ultrasound sessions, CURL achieves a sensitivity of 78.01% and an AUROC of 81.60%, demonstrating its potential for reliable and objective FM analysis. These results highlight the potential of self-supervised contrastive learning for fetal movement analysis, paving the way for improved prenatal monitoring and clinical decision-making.

---

## 40. A Structured Review and Quantitative Profiling of Public Brain MRI Datasets for Foundation Model Development

**论文链接:** [http://arxiv.org/abs/2510.20196v1](http://arxiv.org/abs/2510.20196v1)

**作者:** Minh Sao Khue Luu, Margaret V. Benedichuk, Ekaterina I. Roppert, Roman M. Kenzhin, Bair N. Tuchinov

**发布时间:** 2025-10-23

### GPT解析

### 总结

本研究分析了54个公开可用的脑部MRI数据集，评估了数据规模、多样性和一致性对基础模型开发的影响，发现数据集间存在显著不平衡和异质性，预处理无法完全消除数据集间的偏差，强调了设计脑部MRI基础模型时需要考虑预处理感知和领域自适应策略的必要性。

### 背景

脑部MRI基础模型的发展依赖于可用数据的规模、多样性和一致性，然而对这些因素的系统评估仍然很少见。

### 目的

分析54个公开可用的脑部MRI数据集（包含超过538,031个样本），为脑部MRI基础模型开发提供结构化、多层次的概述。

### 方法

在数据集层面分析模态组成、疾病覆盖范围和数据集规模；在图像层面量化15个代表性数据集中的体素间距、方向和强度分布；评估预处理步骤（强度归一化、偏置场校正、颅骨剥离、空间配准和插值）对体素统计和几何形状的影响；使用3D DenseNet121进行特征空间案例研究，评估标准化预处理后的协变量偏移。

### 主要发现

数据集层面存在大型健康队列与较小临床人群之间的严重不平衡；图像层面存在显著的异质性，可能影响表示学习；预处理步骤虽提高了数据集内部一致性，但数据集间的残余差异仍然存在；标准化预处理后仍可测量到残余协变量偏移，确认仅靠调和无法消除数据集间的偏差。

### 结论

这些分析提供了公共脑部MRI资源中变异性的统一表征，强调在设计可推广的脑部MRI基础模型时需要考虑预处理感知和领域自适应策略。

### 翻译

脑部MRI基础模型的发展在很大程度上取决于可用数据的规模、多样性和一致性，然而对这些因素的系统评估仍然很少。在本研究中，我们分析了54个公开可用的脑部MRI数据集，包含超过538,031个样本，为基础模型开发提供了结构化、多层次的概述。在数据集层面，我们表征了模态组成、疾病覆盖范围和数据集规模，揭示了大型健康队列与较小临床人群之间的严重不平衡。在图像层面，我们量化了15个代表性数据集中的体素间距、方向和强度分布，证明了可能影响表示学习的显著异质性。然后，我们对预处理变异性进行了定量评估，检查了强度归一化、偏置场校正、颅骨剥离、空间配准和插值如何改变体素统计和几何形状。虽然这些步骤提高了数据集内部的一致性，但数据集之间的残余差异仍然存在。最后，使用3D DenseNet121进行的特征空间案例研究显示，在标准化预处理后仍可测量到残余协变量偏移，确认仅靠调和无法消除数据集间的偏差。总之，这些分析提供了公共脑部MRI资源中变异性的统一表征，并强调了在设计可推广的脑部MRI基础模型时需要考虑预处理感知和领域自适应策略的必要性。


### 论文摘要

The development of foundation models for brain MRI depends critically on the scale, diversity, and consistency of available data, yet systematic assessments of these factors remain scarce. In this study, we analyze 54 publicly accessible brain MRI datasets encompassing over 538,031 to provide a structured, multi-level overview tailored to foundation model development. At the dataset level, we characterize modality composition, disease coverage, and dataset scale, revealing strong imbalances between large healthy cohorts and smaller clinical populations. At the image level, we quantify voxel spacing, orientation, and intensity distributions across 15 representative datasets, demonstrating substantial heterogeneity that can influence representation learning. We then perform a quantitative evaluation of preprocessing variability, examining how intensity normalization, bias field correction, skull stripping, spatial registration, and interpolation alter voxel statistics and geometry. While these steps improve within-dataset consistency, residual differences persist between datasets. Finally, feature-space case study using a 3D DenseNet121 shows measurable residual covariate shift after standardized preprocessing, confirming that harmonization alone cannot eliminate inter-dataset bias. Together, these analyses provide a unified characterization of variability in public brain MRI resources and emphasize the need for preprocessing-aware and domain-adaptive strategies in the design of generalizable brain MRI foundation models.

---

## 41. IB-GAN: Disentangled Representation Learning with Information Bottleneck Generative Adversarial Networks

**论文链接:** [http://arxiv.org/abs/2510.20165v1](http://arxiv.org/abs/2510.20165v1)

**作者:** Insu Jeon, Wonkwang Lee, Myeongjang Pyeon, Gunhee Kim

**发布时间:** 2025-10-23

**DOI:** 10.1609/aaai.v35i9.16967

**备注:** Published in the Proceedings of the Thirty Fifth AAAI Conference on  Artificial Intelligence (AAAI 2021), paper number 7926

### GPT解析

### 总结

研究提出了一种基于GAN的无监督解纠缠表征学习模型IB-GAN，利用信息瓶颈框架优化GAN，通过生成器的中间层约束输入与生成输出之间的互信息，实现了对潜在空间的解纠缠和可解释性利用。

### 背景

解纠缠表征学习是机器学习领域的重要研究方向，现有的方法如InfoGAN和β-VAEs存在一定局限性，需要改进模型架构以获得更好的解纠缠能力和样本质量。

### 目的

提出一种新的基于GAN的无监督解纠缠表征学习模型IB-GAN，利用信息瓶颈框架优化GAN，实现更好的解纠缠能力和样本质量。

### 方法

IB-GAN架构与InfoGAN部分相似但有关键差异：利用生成器的中间层约束输入与生成输出之间的互信息；中间随机层可作为可学习的潜在分布，与生成器端到端联合训练；使生成器能够以解纠缠和可解释的方式利用潜在空间。

### 主要发现

在dSprites和Color-dSprites数据集上的实验表明，IB-GAN实现了与最先进的β-VAEs相当的解纠缠分数，并优于InfoGAN；在CelebA和3D Chairs数据集上，IB-GAN在FID分数方面通常比β-VAEs和Info-GAN生成的样本具有更好的视觉质量和多样性。

### 结论

IB-GAN通过利用信息瓶颈框架优化GAN，有效提升了模型的解纠缠能力和样本生成质量，是一种有效的无监督解纠缠表征学习方法。

### 翻译

我们提出了一种新的基于GAN的无监督解纠缠表征学习模型。这一新模型是在尝试将信息瓶颈框架应用于GAN优化的过程中发现的，因此命名为IB-GAN。IB-GAN的架构与InfoGAN部分相似，但有一个关键区别：利用生成器的中间层来约束输入与生成输出之间的互信息。中间随机层可以作为可学习的潜在分布，与生成器以端到端的方式联合训练。因此，IB-GAN的生成器能够以解纠缠和可解释的方式利用潜在空间。在dSprites和Color-dSprites数据集上的实验表明，IB-GAN实现了与最先进的β-VAEs相当的解纠缠分数，并优于InfoGAN。此外，在CelebA和3D Chairs数据集上，IB-GAN生成的样本在FID分数方面通常比β-VAEs和Info-GAN具有更好的视觉质量和多样性。


### 论文摘要

We propose a new GAN-based unsupervised model for disentangled representation learning. The new model is discovered in an attempt to utilize the Information Bottleneck (IB) framework to the optimization of GAN, thereby named IB-GAN. The architecture of IB-GAN is partially similar to that of InfoGAN but has a critical difference; an intermediate layer of the generator is leveraged to constrain the mutual information between the input and the generated output. The intermediate stochastic layer can serve as a learnable latent distribution that is trained with the generator jointly in an end-to-end fashion. As a result, the generator of IB-GAN can harness the latent space in a disentangled and interpretable manner. With the experiments on dSprites and Color-dSprites dataset, we demonstrate that IB-GAN achieves competitive disentanglement scores to those of state-of-the-art \b{eta}-VAEs and outperforms InfoGAN. Moreover, the visual quality and the diversity of samples generated by IB-GAN are often better than those by \b{eta}-VAEs and Info-GAN in terms of FID score on CelebA and 3D Chairs dataset.

---

## 42. TOMCAT: Test-time Comprehensive Knowledge Accumulation for Compositional Zero-Shot Learning

**论文链接:** [http://arxiv.org/abs/2510.20162v1](http://arxiv.org/abs/2510.20162v1)

**作者:** Xudong Yan, Songhe Feng

**发布时间:** 2025-10-23

**备注:** Accepted to NeurIPS 2025

### GPT解析

### 总结

这篇论文提出了一种新的组合零样本学习方法，通过无监督数据积累多模态知识并更新原型，解决了测试时分布偏移的问题，在多个基准数据集上取得了最先进的性能。

### 背景

组合零样本学习旨在基于已学习知识识别新的属性-对象组合。现有方法在测试时由于标签空间分布偏移导致性能下降，这种偏移源于包含了从未见过的属性和对象重新组合的样本。

### 目的

克服测试时标签空间分布偏移带来的挑战，提出一种方法来更新多模态原型，使模型能够灵活适应测试时的分布偏移。

### 方法

提出一种新方法，通过无监督数据积累文本和视觉模态的综合知识；设计自适应更新权重控制原型调整程度；引入动态优先队列存储高置信度图像，从历史图像获取视觉知识；通过多模态协同表示学习对齐文本和视觉原型。

### 主要发现

在四个基准数据集上，无论是在封闭世界还是开放世界设置下，该方法都达到了最先进的性能。

### 结论

该方法通过更新多模态原型和自适应权重，有效解决了组合零样本学习中的分布偏移问题，代码将在https://github.com/xud-yan/TOMCAT上提供。

### 翻译

组合零样本学习旨在基于已学习知识识别新的属性-对象组合。现有方法在测试时由于标签空间分布偏移导致性能下降，这源于包含了从未见过的属性和对象重新组合的样本。为克服这一挑战，我们提出了一种新方法，通过无监督数据在文本和视觉模态中积累综合知识，以在测试时更新多模态原型。基于此，我们进一步设计了自适应更新权重来控制原型调整程度，使模型能够在测试过程中灵活适应分布偏移。此外，我们引入了动态优先队列，存储高置信度图像，以便从历史图像获取视觉知识进行推理。考虑到多模态知识的语义一致性，我们通过多模态协同表示学习对齐文本和视觉原型。大量实验表明，我们的方法在四个基准数据集上，无论是在封闭世界还是开放世界设置下，都达到了最先进的性能。代码将在https://github.com/xud-yan/TOMCAT上提供。


### 论文摘要

Compositional Zero-Shot Learning (CZSL) aims to recognize novel attribute-object compositions based on the knowledge learned from seen ones. Existing methods suffer from performance degradation caused by the distribution shift of label space at test time, which stems from the inclusion of unseen compositions recombined from attributes and objects. To overcome the challenge, we propose a novel approach that accumulates comprehensive knowledge in both textual and visual modalities from unsupervised data to update multimodal prototypes at test time. Building on this, we further design an adaptive update weight to control the degree of prototype adjustment, enabling the model to flexibly adapt to distribution shift during testing. Moreover, a dynamic priority queue is introduced that stores high-confidence images to acquire visual knowledge from historical images for inference. Considering the semantic consistency of multimodal knowledge, we align textual and visual prototypes by multimodal collaborative representation learning. Extensive experiments indicate that our approach achieves state-of-the-art performance on four benchmark datasets under both closed-world and open-world settings. Code will be available at https://github.com/xud-yan/TOMCAT .

---

## 43. Improving Predictive Confidence in Medical Imaging via Online Label Smoothing

**论文链接:** [http://arxiv.org/abs/2510.20011v1](http://arxiv.org/abs/2510.20011v1)

**作者:** Kushan Choudhury, Shubhrodeep Roy, Ankur Chanda, Shubhajit Biswas, Somenath Kuiry

**发布时间:** 2025-10-22

**备注:** Accepted and presented in International Conference on Advancing  Science and Technologies in Health Science

### GPT解析

### 总结

本研究探索了在线标签平滑(OLS)在医学图像分类中的应用，结果显示OLS能提高分类准确率并改善模型校准性。

### 背景

深度学习模型，特别是卷积神经网络，在医学图像分类中取得了显著成果，但这些模型经常产生过度自信的预测，影响在关键医疗环境中的可靠性。

### 目的

研究使用在线标签平滑(OLS)，一种基于模型自身预测模式动态调整软标签的方法，以提高医学图像分类模型的性能和可靠性。

### 方法

在大型RadImageNet数据集上使用三种架构评估OLS：ResNet-50、MobileNetV2和VGG-19，并与标准训练方法进行比较。

### 主要发现

OLS相比标准训练方法持续提高了Top-1和Top-5分类准确率，并产生更紧凑和良好分离的特征嵌入，表明表示学习得到改善。

### 结论

OLS不仅增强了预测性能，还提高了校准性，使其成为医学成像领域开发可信AI系统的实用有效解决方案。

### 翻译

深度学习模型，特别是卷积神经网络，在医学图像分类中已取得了令人印象深刻的结果。然而，这些模型通常会产生过度自信的预测，这可能削弱它们在关键医疗环境中的可靠性。虽然传统的标签平滑提供了一种减少这种过度自信的简单方法，但它未能考虑类别之间的关系，将所有非目标类别同等对待。在本研究中，我们探索了在线标签平滑(OLS)的使用，这是一种动态方法，基于模型自身的预测模式在训练过程中调整软标签。我们在大型RadImageNet数据集上使用三种广泛使用的架构评估了OLS：ResNet-50、MobileNetV2和VGG-19。我们的结果表明，与标准训练方法（包括硬标签、传统标签平滑和无教师知识蒸馏）相比，OLS持续提高了Top-1和Top-5分类准确率。除了准确率的提升外，OLS还导致更紧凑和良好分离的特征嵌入，表明表示学习得到改善。这些发现表明，OLS不仅增强了预测性能，还提高了校准性，使其成为医学成像领域开发可信AI系统的实用有效解决方案。


### 论文摘要

Deep learning models, especially convolutional neural networks, have achieved impressive results in medical image classification. However, these models often produce overconfident predictions, which can undermine their reliability in critical healthcare settings. While traditional label smoothing offers a simple way to reduce such overconfidence, it fails to consider relationships between classes by treating all non-target classes equally. In this study, we explore the use of Online Label Smoothing (OLS), a dynamic approach that adjusts soft labels throughout training based on the model's own prediction patterns. We evaluate OLS on the large-scale RadImageNet dataset using three widely used architectures: ResNet-50, MobileNetV2, and VGG-19. Our results show that OLS consistently improves both Top-1 and Top-5 classification accuracy compared to standard training methods, including hard labels, conventional label smoothing, and teacher-free knowledge distillation. In addition to accuracy gains, OLS leads to more compact and well-separated feature embeddings, indicating improved representation learning. These findings suggest that OLS not only strengthens predictive performance but also enhances calibration, making it a practical and effective solution for developing trustworthy AI systems in the medical imaging domain.

---

## 44. Transformed Multi-view 3D Shape Features with Contrastive Learning

**论文链接:** [http://arxiv.org/abs/2510.19955v1](http://arxiv.org/abs/2510.19955v1)

**作者:** Márcus Vinícius Lobo Costa, Sherlon Almeida da Silva, Bárbara Caroline Benato, Leo Sampaio Ferraz Ribeiro, Moacir Antonelli Ponti

**发布时间:** 2025-10-22

### GPT解析

### 总结

该论文研究了3D形状特征表示学习的挑战，通过将Vision Transformers架构与对比学习目标相结合，在3D形状理解方面取得了良好效果

### 背景

计算机视觉方法在从2D图像识别3D物体方面存在困难，通常需要大量标记数据，且依赖的卷积神经网络可能会忽略关键的形状关系

### 目的

解决3D形状特征表示学习中的挑战，探索减少对大量标记数据依赖的方法，克服CNNs在捕获关键形状关系方面的局限性

### 方法

使用Vision Transformers (ViTs)架构与现代对比目标相结合，进行多视图3D分析，结合ViTs理解整体形状的能力和对比学习的有效性

### 主要发现

监督对比损失在ModelNet10上达到了约90.6%的准确率；ViTs能够捕获全局形状语义，而对比优化能够完善局部判别特征

### 结论

通过结合ViTs与对比目标，成功实现了3D表示学习，这种方法基于大量实验评估，证明了其有效性

### 翻译

这篇论文通过研究最先进的骨干网络与对比监督和自监督学习目标的组合，解决了3D形状特征表示学习中的挑战。计算机视觉方法在从2D图像识别3D物体方面存在困难，通常需要大量标记数据，并依赖于卷积神经网络(CNNs)，而这些网络可能会忽略关键的形状关系。我们的研究表明，当Vision Transformers (ViTs)架构与现代对比目标配对时，在我们的下游任务中多视图3D分析取得了有希望的结果，统一了对比学习和3D形状理解的流程。例如，监督对比损失在ModelNet10上达到了约90.6%的准确率。ViTs和对比学习的应用，利用了ViTs理解整体形状的能力和对比学习的有效性，克服了对大量标记数据的需求以及CNNs在捕获关键形状关系方面的局限性。成功的原因在于通过ViTs捕获全局形状语义，并通过对比优化完善局部判别特征。重要的是，我们的方法是经验性的，因为它基于大量的实验评估来验证将ViTs与对比目标相结合用于3D表示学习的有效性。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决3D形状特征表示学习中的挑战，特别是计算机视觉方法从2D图像识别3D物体的困难。这个问题很重要，因为3D形状理解对机器人、虚拟现实等应用至关重要，而当前方法需要大量标记数据且依赖CNN，这些网络可能忽略关键的形状关系，限制了模型在真实世界应用中的表现。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先认识到CNN在3D形状理解中的局限性，注意到ViT架构在视觉识别任务中表现优异，并观察到对比学习在利用未标记数据方面的潜力。他们借鉴了MVCNN(首次使用CNN进行3D形状理解)、ViT架构以及多种对比学习方法(如InfoNCE、SimCLR、SupCon)，创新性地将这些技术结合应用于3D多视图形状理解，这是之前研究较少探索的组合。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是利用Vision Transformer的注意力机制捕获全局形状语义，通过对比学习优化细化局部判别特征，统一对比学习和3D形状理解流程。整体流程分为两个阶段：第一阶段是多视图渲染和对比学习，从3D网格生成12个视图图像，使用多种对比损失函数训练ViT和CNN骨干；第二阶段是下游任务评估，包括分类(线性评估、k-NN分类、t-SNE可视化)和检索任务(基于余弦相似度排序，使用mAP评估)。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：首次将ViT与对比学习结合用于3D多视图形状理解；系统评估四种ViT骨干和五种对比损失函数；统一对比学习和3D形状理解流程；在多个下游任务进行全面评估。相比之前工作，本文主要使用ViT而非CNN骨干；将对比学习从2D扩展到3D多视图数据；提供更全面的评估；在ModelNet10上达到90.6%分类准确率和95.5%的mAP，超越之前方法。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '这篇论文通过将Vision Transformer与先进的对比学习目标相结合，显著提升了3D多视图形状理解的性能，同时减少了对大量标记数据的依赖。'}


### 论文摘要

This paper addresses the challenges in representation learning of 3D shape features by investigating state-of-the-art backbones paired with both contrastive supervised and self-supervised learning objectives. Computer vision methods struggle with recognizing 3D objects from 2D images, often requiring extensive labeled data and relying on Convolutional Neural Networks (CNNs) that may overlook crucial shape relationships. Our work demonstrates that Vision Transformers (ViTs) based architectures, when paired with modern contrastive objectives, achieve promising results in multi-view 3D analysis on our downstream tasks, unifying contrastive and 3D shape understanding pipelines. For example, supervised contrastive losses reached about 90.6% accuracy on ModelNet10. The use of ViTs and contrastive learning, leveraging ViTs' ability to understand overall shapes and contrastive learning's effectiveness, overcomes the need for extensive labeled data and the limitations of CNNs in capturing crucial shape relationships. The success stems from capturing global shape semantics via ViTs and refining local discriminative features through contrastive optimization. Importantly, our approach is empirical, as it is grounded on extensive experimental evaluation to validate the effectiveness of combining ViTs with contrastive objectives for 3D representation learning.

---

## 45. Unsupervised Domain Adaptation via Similarity-based Prototypes for Cross-Modality Segmentation

**论文链接:** [http://arxiv.org/abs/2510.20596v1](http://arxiv.org/abs/2510.20596v1)

**作者:** Ziyu Ye, Chen Ju, Chaofan Ma, Xiaoyun Zhang

**发布时间:** 2025-10-23

**备注:** MICCAI 2021

### GPT解析

### 总结

本文提出了一种基于相似性原型的跨模态分割新框架，通过在嵌入空间中学习类别原型并引入相似性约束，以及使用字典存储不同图像中提取的原型，解决了深度学习模型在未见数据上性能下降的问题，实验证明该方法优于其他最先进方法。

### 背景

深度学习模型在各种视觉挑战中取得了巨大成功，但训练好的模型在应用于未见过的数据时性能会急剧下降，模型对域偏移敏感。

### 目的

减少域差距，避免对未见域的昂贵标注，提高模型在跨模态分割任务中的性能。

### 方法

提出一种基于相似性原型的跨模态分割框架，在嵌入空间中学习类别的代表性原型，引入相似性约束使原型对每个语义类别具有代表性且不同类别间可分离，使用字典存储从不同图像中提取的原型防止类别缺失问题，实现原型的对比学习以提高性能。

### 主要发现

通过原型学习和对比学习的方法可以有效解决域偏移问题，提高跨模态分割性能。

### 结论

本文提出的基于相似性原型的跨模态分割框架比其他最先进方法取得了更好的结果。

### 翻译

深度学习模型在各种视觉挑战中取得了巨大成功，但训练好的模型在应用于未见过的数据时性能会急剧下降。由于模型对域偏移敏感，无监督域适应尝试减少域差距并避免对未见域的昂贵标注。本文提出了一种基于相似性原型的跨模态分割新框架。具体来说，我们在嵌入空间中学习类别的原型，然后引入相似性约束，使这些原型对每个语义类别具有代表性，同时不同类别之间可分离。此外，我们使用字典存储从不同图像中提取的原型，这可以防止类别缺失问题，并实现原型的对比学习，进一步提高性能。大量实验表明，我们的方法比其他最先进方法取得了更好的结果。


### 论文摘要

Deep learning models have achieved great success on various vision challenges, but a well-trained model would face drastic performance degradation when applied to unseen data. Since the model is sensitive to domain shift, unsupervised domain adaptation attempts to reduce the domain gap and avoid costly annotation of unseen domains. This paper proposes a novel framework for cross-modality segmentation via similarity-based prototypes. In specific, we learn class-wise prototypes within an embedding space, then introduce a similarity constraint to make these prototypes representative for each semantic class while separable from different classes. Moreover, we use dictionaries to store prototypes extracted from different images, which prevents the class-missing problem and enables the contrastive learning of prototypes, and further improves performance. Extensive experiments show that our method achieves better results than other state-of-the-art methods.

---

## 46. SheafAlign: A Sheaf-theoretic Framework for Decentralized Multimodal Alignment

**论文链接:** [http://arxiv.org/abs/2510.20540v1](http://arxiv.org/abs/2510.20540v1)

**作者:** Abdulmomen Ghalkha, Zhuojun Tian, Chaouki Ben Issaid, Mehdi Bennis

**发布时间:** 2025-10-23

**备注:** 5 pages, 3 figures, 1 table

### GPT解析

### 总结

SheafAlign是一种基于层理论的多模态对齐框架，适用于分布式场景，不要求所有模态相互冗余，能有效保留共享和独特信息，并在多个方面表现优于现有方法。

### 背景

传统多模态对齐方法假设所有模态之间存在相互冗余，这一假设在现实世界的分布式场景中并不成立。

### 目的

提出一种名为SheafAlign的框架，用于去中心化的多模态对齐，用多个比较空间替代单一空间对齐。

### 方法

使用层理论框架，通过层结构建模成对模态关系，并利用基于去中心化对比学习的目标进行训练。

### 主要发现

Sheaf克服了先前方法的局限性，不需要所有模态之间存在相互冗余；在多模态传感数据集上表现出优越的零样本泛化能力、优秀的跨模态对齐能力、对缺失模态具有鲁棒性，与最先进的基线相比，通信成本降低50%。

### 结论

SheafAlign是一种有效的去中心化多模态对齐方法，能够在分布式场景中更好地处理模态间关系。

### 翻译

传统多模态对齐方法假设所有模态之间存在相互冗余，这一假设在现实世界的分布式场景中并不成立。我们提出了SheafAlign，一种基于层理论的去中心化多模态对齐框架，它用多个比较空间替代了单一空间对齐。这种方法通过层结构建模成对模态关系，并利用基于去中心化对比学习的目标进行训练。SheafAlign通过不要求所有模态之间存在相互冗余，克服了先前方法的局限性，同时保留了共享信息和独特信息。在多模态传感数据集上的实验显示，它在零样本泛化、跨模态对齐和对缺失模态的鲁棒性方面具有优越性，并且与最先进的基线相比，通信成本降低了50%。


### 论文摘要

Conventional multimodal alignment methods assume mutual redundancy across all modalities, an assumption that fails in real-world distributed scenarios. We propose SheafAlign, a sheaf-theoretic framework for decentralized multimodal alignment that replaces single-space alignment with multiple comparison spaces. This approach models pairwise modality relations through sheaf structures and leverages decentralized contrastive learning-based objectives for training. SheafAlign overcomes the limitations of prior methods by not requiring mutual redundancy among all modalities, preserving both shared and unique information. Experiments on multimodal sensing datasets show superior zero-shot generalization, cross-modal alignment, and robustness to missing modalities, with 50\% lower communication cost than state-of-the-art baselines.

---

## 47. ViTacGen: Robotic Pushing with Vision-to-Touch Generation

**论文链接:** [http://arxiv.org/abs/2510.14117v2](http://arxiv.org/abs/2510.14117v2)

**作者:** Zhiyuan Wu, Yijiong Lin, Yongqiang Zhao, Xuyang Zhang, Zhuo Chen, Nathan Lepora, Shan Luo

**发布时间:** 2025-10-15

### GPT解析

### 总结

ViTacGen是一种新颖的机器人操作框架，通过视觉到触觉生成在强化学习中消除对高分辨率真实触觉传感器的依赖，实现仅视觉机器人系统上的有效零样本部署，在模拟和真实世界实验中展现出高达86%的成功率。

### 背景

机器人推动是一种需要触觉反馈来捕捉末端执行器与物体间细微接触力和动力学的基本操作任务。真实触觉传感器面临高成本、脆弱性、校准和传感器差异等挑战，而仅依赖视觉的策略性能有限。

### 目的

开发一种能够从视觉推断触觉状态的机器人操作框架，减少对昂贵且脆弱的真实触觉传感器的依赖，实现仅视觉机器人系统上的有效零样本部署。

### 方法

ViTacGen包含一个编码器-解码器视觉到触觉生成网络，直接从视觉图像序列生成接触深度图像（标准化触觉表示），以及一个使用对比学习融合视觉-触觉数据的强化学习策略。

### 主要发现

在模拟和真实世界实验中验证了ViTacGen的有效性，其性能优于传统方法，成功率达到86%。

### 结论

ViTacGen成功实现了从视觉到触觉的生成，使仅视觉的机器人系统能够在没有真实触觉传感器的情况下执行有效的机器人推动任务。

### 翻译

机器人推动是一种基本操作任务，需要触觉反馈来捕捉末端执行器与物体之间的细微接触力和动力学特性。然而，真实触觉传感器通常面临高成本和脆弱性等硬件限制，以及涉及校准和不同传感器间差异的部署挑战，而仅视觉的策略难以获得令人满意的性能。受人类从视觉推断触觉状态能力的启发，我们提出了ViTacGen，一种专为视觉机器人推动设计的新颖机器人操作框架，在强化学习中使用视觉到触觉生成，消除对高分辨率真实触觉传感器的依赖，实现仅视觉机器人系统上的有效零样本部署。具体来说，ViTacGen包含一个编码器-解码器视觉到触觉生成网络，直接从视觉图像序列生成接触深度图像（标准化的触觉表示），随后是一个基于视觉和生成触觉观察使用对比学习融合视觉-触觉数据的强化学习策略。我们在模拟和真实世界实验中都验证了我们方法的有效性，展示了其优越的性能，成功率高达86%。


### 论文摘要

Robotic pushing is a fundamental manipulation task that requires tactile feedback to capture subtle contact forces and dynamics between the end-effector and the object. However, real tactile sensors often face hardware limitations such as high costs and fragility, and deployment challenges involving calibration and variations between different sensors, while vision-only policies struggle with satisfactory performance. Inspired by humans' ability to infer tactile states from vision, we propose ViTacGen, a novel robot manipulation framework designed for visual robotic pushing with vision-to-touch generation in reinforcement learning to eliminate the reliance on high-resolution real tactile sensors, enabling effective zero-shot deployment on visual-only robotic systems. Specifically, ViTacGen consists of an encoder-decoder vision-to-touch generation network that generates contact depth images, a standardized tactile representation, directly from visual image sequence, followed by a reinforcement learning policy that fuses visual-tactile data with contrastive learning based on visual and generated tactile observations. We validate the effectiveness of our approach in both simulation and real world experiments, demonstrating its superior performance and achieving a success rate of up to 86\%.

---

## 48. Real Deep Research for AI, Robotics and Beyond

**论文链接:** [http://arxiv.org/abs/2510.20809v1](http://arxiv.org/abs/2510.20809v1)

**作者:** Xueyan Zou, Jianglong Ye, Hao Zhang, Xiaoyu Xiang, Mingyu Ding, Zhaojing Yang, Yong Jae Lee, Zhuowen Tu, Sifei Liu, Xiaolong Wang

**发布时间:** 2025-10-23

**备注:** website: https://realdeepresearch.github.io

### GPT解析

### 总结

本文提出了一种名为Real Deep Research (RDR)的全面框架，用于系统分析AI和机器人研究领域，帮助研究人员识别新兴趋势、发现跨领域机会并为新研究提供起点。

### 背景

AI和机器人研究快速增长，每年发表超过10,000篇论文，使得研究人员难以跟上最新发展。快速发展的趋势、跨学科工作的兴起以及需要探索专业领域之外的知识都构成了这一挑战。

### 目的

提出一个通用流程，能够系统分析任何研究领域，识别新兴趋势，发现跨领域机会，并为新研究提供具体起点。

### 方法

开发了Real Deep Research (RDR)全面框架，应用于AI和机器人领域，特别关注基础模型和机器人进展，同时简要扩展到其他科学领域。主论文详细介绍了RDR流程的构建，附录提供了各分析主题的广泛结果。

### 主要发现

摘要中未明确提及具体的研究发现。

### 结论

希望这项工作能为AI领域及更广泛领域的研究人员提供启示，帮助他们应对信息过载的挑战。

### 翻译

随着AI和机器人研究的快速增长，现在每年产生超过10,000篇论文，研究人员越来越难以跟上最新发展。快速发展的趋势、跨学科工作的兴起以及探索专业领域之外知识的需求都构成了这一挑战。为解决这些问题，我们提出了一种能够系统分析任何研究领域的通用流程：识别新兴趋势，发现跨领域机会，并为新研究提供具体起点。在本工作中，我们介绍了Real Deep Research (RDR)，这是一个应用于AI和机器人领域的全面框架，特别关注基础模型和机器人进展。我们还简要扩展了对其他科学领域的分析。主论文详细介绍了RDR流程的构建，附录提供了每个分析主题的广泛结果。我们希望这项工作能为AI领域及其他领域的研究人员提供启示。


### 论文摘要

With the rapid growth of research in AI and robotics now producing over 10,000 papers annually it has become increasingly difficult for researchers to stay up to date. Fast evolving trends, the rise of interdisciplinary work, and the need to explore domains beyond one's expertise all contribute to this challenge. To address these issues, we propose a generalizable pipeline capable of systematically analyzing any research area: identifying emerging trends, uncovering cross domain opportunities, and offering concrete starting points for new inquiry. In this work, we present Real Deep Research (RDR) a comprehensive framework applied to the domains of AI and robotics, with a particular focus on foundation models and robotics advancements. We also briefly extend our analysis to other areas of science. The main paper details the construction of the RDR pipeline, while the appendix provides extensive results across each analyzed topic. We hope this work sheds light for researchers working in the field of AI and beyond.

---

## 49. EmbodiedBrain: Expanding Performance Boundaries of Task Planning for Embodied Intelligence

**论文链接:** [http://arxiv.org/abs/2510.20578v1](http://arxiv.org/abs/2510.20578v1)

**作者:** Ding Zou, Feifan Wang, Mengyu Ge, Siyuan Fan, Zongbing Zhang, Wei Chen, Lingfeng Wang, Zhongyou Hu, Wenrui Yan, Zhengwei Gao, Hao Wang, Weizhao Jin, Yu Zhang, Hainan Zhao, Mingliang Zhang, Xianxian Xi, Yaru Zhang, Wenyuan Li, Zhengguang Gao, Yurui Zhu

**发布时间:** 2025-10-23

### GPT解析

### 总结

本文提出了EmbodiedBrain，一种新型的视觉语言基础模型，解决了当前具身AI模型在模型设计、实时性能评估和离线指标方面的局限性，实现了在所有评估指标上的最先进性能。

### 背景

实现通用人工智能(AGI)需要能够在物理环境中进行稳健的空间感知、有效任务规划和自适应执行的具身AI智能体。然而，当前用于具身任务的大型语言模型和多模态大型语言模型存在关键局限性。

### 目的

解决当前具身AI模型的局限性，提出一种新的视觉语言基础模型，提高具身智能体在物理环境中的感知、规划和执行能力。

### 方法

开发了EmbodiedBrain模型(7B和32B参数规模)，采用与智能体对齐的数据结构，结合大规模监督微调(SFT)和步骤增强组相对策略优化(Step-GRPO)训练方法，引入包含生成奖励模型(GRM)的全面奖励系统，并建立三部分评估体系(通用、规划和端到端模拟基准测试)。

### 主要发现

EmbodiedBrain在所有评估指标上实现了卓越性能，为具身基础模型建立了新的最先进水平，有效解决了模型设计与智能体需求之间的差距、实时延迟与性能的权衡问题，以及不真实的离线评估指标问题。

### 结论

EmbodiedBrain为下一代通用具身智能体的发展铺平了道路，所有数据、模型权重和评估方法均已开源，可供研究社区使用。

### 翻译

实现通用人工智能(AGI)需要能够在物理环境中进行稳健的空间感知、有效任务规划和自适应执行的具身AI智能体。然而，当前用于具身任务的大型语言模型(LLMs)和多模态大型语言模型(MLLMs)存在关键局限性，包括模型设计与智能体需求之间的显著差距、实时延迟与性能之间的不可避免权衡，以及使用不真实的离线评估指标。为解决这些挑战，我们提出了EmbodiedBrain，一种有7B和32B两种参数规模的新型视觉语言基础模型。我们的框架具有与智能体对齐的数据结构，采用结合大规模监督微调(SFT)和步骤增强组相对策略优化(Step-GRPO)的强大训练方法，通过将前序步骤整合为引导前体来提高长距离任务成功率。此外，我们引入了包含在基础设施层面加速的生成奖励模型(GRM)的全面奖励系统，以提高训练效率。为进行彻底验证，我们建立了包含通用、规划和端到端模拟基准测试的三部分评估体系，并提出了一个具有挑战性的新模拟环境并开源。实验结果表明，EmbodiedBrain在所有指标上都实现了卓越性能，为具身基础模型建立了新的最先进水平。为铺平下一代通用具身智能体的发展道路，我们开源了所有数据、模型权重和评估方法，可在https://zterobot.github.io/EmbodiedBrain.github.io获取。


### 论文摘要

The realization of Artificial General Intelligence (AGI) necessitates Embodied AI agents capable of robust spatial perception, effective task planning, and adaptive execution in physical environments. However, current large language models (LLMs) and multimodal LLMs (MLLMs) for embodied tasks suffer from key limitations, including a significant gap between model design and agent requirements, an unavoidable trade-off between real-time latency and performance, and the use of unauthentic, offline evaluation metrics. To address these challenges, we propose EmbodiedBrain, a novel vision-language foundation model available in both 7B and 32B parameter sizes. Our framework features an agent-aligned data structure and employs a powerful training methodology that integrates large-scale Supervised Fine-Tuning (SFT) with Step-Augumented Group Relative Policy Optimization (Step-GRPO), which boosts long-horizon task success by integrating preceding steps as Guided Precursors. Furthermore, we incorporate a comprehensive reward system, including a Generative Reward Model (GRM) accelerated at the infrastructure level, to improve training efficiency. For enable thorough validation, we establish a three-part evaluation system encompassing General, Planning, and End-to-End Simulation Benchmarks, highlighted by the proposal and open-sourcing of a novel, challenging simulation environment. Experimental results demonstrate that EmbodiedBrain achieves superior performance across all metrics, establishing a new state-of-the-art for embodied foundation models. Towards paving the way for the next generation of generalist embodied agents, we open-source all of our data, model weight, and evaluating methods, which are available at https://zterobot.github.io/EmbodiedBrain.github.io.

---

## 50. A Unified Framework for Zero-Shot Reinforcement Learning

**论文链接:** [http://arxiv.org/abs/2510.20542v1](http://arxiv.org/abs/2510.20542v1)

**作者:** Jacopo Di Ventura, Jan Felix Kleuker, Aske Plaat, Thomas Moerland

**发布时间:** 2025-10-23

### GPT解析

### 总结

该论文提出了零样本强化学习的第一个统一框架，引入了一致的符号和分类法，将现有方法组织为直接表示和组合表示两大类，为该领域提供了原则性基础和研究方向。

### 背景

零样本强化学习允许代理在不监督的情况下开发通用能力，无需额外训练或规划即可解决下游任务。与传统RL优化固定奖励不同，零样本RL需要代理编码足够丰富的表示以立即适应任何目标，类似于视觉和语言基础模型。尽管该领域兴趣增长，但缺乏共同的分析视角。

### 目的

提出零样本强化学习的第一个统一框架，引入一致的符号和分类法，组织现有方法并允许直接比较不同方法，为该领域提供原则性基础。

### 方法

框架将算法分为两类：直接表示（学习从奖励到策略的端到端映射）和组合表示（利用值函数的子结构分解表示）。在此框架内，突出方法的共同原则和关键差异，为后续特征方法推导扩展界限，提供其在零样本环境中的新视角。

### 主要发现

通过共同视角整合了现有工作，揭示了不同方法间的共享原则和关键差异，为后续特征方法提供了新的理论视角，表明组合表示可能更适合零样本场景。

### 结论

该框架为零样本强化学习的未来研究提供了原则性基础，并指明了开发更通用代理的明确路径，有助于推动通用人工智能代理的发展。

### 翻译

零样本强化学习(RL)已成为一种在不监督情况下开发通用代理的设置，能够在测试时无需额外训练或规划的情况下解决下游任务。与传统优化固定奖励的RL不同，零样本RL需要代理编码足够丰富的表示以支持立即适应任何目标，这与视觉和语言基础模型相类似。尽管兴趣日益增长，该领域仍缺乏共同的分析视角。我们提出了零样本RL的第一个统一框架，我们的引入了一致的符号和分类法，组织了现有方法并允许直接比较它们。我们框架的核心是将算法分为两个家族：直接表示，学习从奖励到策略的端到端映射；以及组合表示，利用值函数的子结构分解表示。在此框架内，我们突出了跨方法的共同原则和关键差异，并为后续特征方法推导了扩展界限，提供了它们在零样本环境中性能的新视角。通过在共同视角下整合现有工作，我们的框架为未来零样本RL研究提供了原则性基础，并概述了开发更通用代理的明确路径。


### 论文摘要

Zero-shot reinforcement learning (RL) has emerged as a setting for developing general agents in an unsupervised manner, capable of solving downstream tasks without additional training or planning at test-time. Unlike conventional RL, which optimizes policies for a fixed reward, zero-shot RL requires agents to encode representations rich enough to support immediate adaptation to any objective, drawing parallels to vision and language foundation models. Despite growing interest, the field lacks a common analytical lens.   We present the first unified framework for zero-shot RL. Our formulation introduces a consistent notation and taxonomy that organizes existing approaches and allows direct comparison between them. Central to our framework is the classification of algorithms into two families: direct representations, which learn end-to-end mappings from rewards to policies, and compositional representations, which decompose the representation leveraging the substructure of the value function. Within this framework, we highlight shared principles and key differences across methods, and we derive an extended bound for successor-feature methods, offering a new perspective on their performance in the zero-shot regime. By consolidating existing work under a common lens, our framework provides a principled foundation for future research in zero-shot RL and outlines a clear path toward developing more general agents.

---

## 51. Dino-Diffusion Modular Designs Bridge the Cross-Domain Gap in Autonomous Parking

**论文链接:** [http://arxiv.org/abs/2510.20335v1](http://arxiv.org/abs/2510.20335v1)

**作者:** Zixuan Wu, Hengyuan Zhang, Ting-Hsuan Chen, Yuliang Guo, David Paz, Xinyu Huang, Liu Ren

**发布时间:** 2025-10-23

**备注:** Code is at  https://github.com/ChampagneAndfragrance/Dino_Diffusion_Parking_Official

### GPT解析

### 总结

该研究提出了Dino-Diffusion Parking (DDP)，一种结合视觉基础模型与基于扩散规划的自动停车流水线，解决了在不同环境条件下停车的鲁棒性问题。

### 背景

停车是驾驶安全的关键支柱，尽管端到端方法在领域内取得了良好结果，但在天气和光照变化等条件下的鲁棒性仍是主要挑战。

### 目的

开发一种领域无关的自动停车流水线，实现分布变化下的通用感知和鲁棒运动规划。

### 方法

提出Dino-Diffusion Parking (DDP)，将视觉基础模型与基于扩散的规划相结合，在CARLA常规设置中训练，然后以零样本方式转移到更具挑战性的环境。

### 主要发现

模型在所有测试的分布外场景中停车成功率 consistently保持在90%以上，消融研究证实网络架构和算法设计显著提高了跨域性能，在3D高斯飞溅环境中的测试显示了有希望的模拟到现实迁移能力。

### 结论

所提出的DDP方法在不同环境条件下都能实现高成功率的自动停车，并且具有从模拟到现实的迁移能力。

### 翻译

停车是驾驶安全的关键支柱。尽管最近的端到端方法在领域内取得了有希望的结果，但在领域变化（如天气和光照变化）下的鲁棒性仍然是一个关键挑战。我们提出Dino-Diffusion Parking (DDP)，一个领域无关的自动停车流水线，它将视觉基础模型与基于扩散的规划相结合，以实现分布变化下的通用感知和鲁棒运动规划。我们在CARLA的常规设置中训练我们的流水线，并以零样本方式将其转移到更具挑战性的设置中。我们的模型在所有测试的分布外场景中停车成功率始终保持在90%以上，消融研究证实，网络架构和算法设计都显著提高了跨域性能，优于现有基线。此外，在从真实停车场重建的3D高斯飞溅环境中的测试显示了有希望的模拟到现实迁移。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决自动驾驶停车系统在不同环境条件下的跨域适应问题。当系统从训练环境（如晴天）转移到部署环境（如雨天、雾天或不同光照条件）时，性能会显著下降。这个问题很重要，因为停车占美国车辆事故的20%，且91%与倒车操作相关，准确的感知、规划和控制对安全至关重要。此外，传统解决方案需要大量收集不同条件下的数据，成本高昂，而本文方法无需额外数据就能适应各种环境。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': "作者将自动驾驶停车问题分解为感知、规划和控制三个模块，而非使用单一端到端模型。他们借鉴了多个现有工作：使用DINOv2视觉基础模型实现鲁棒感知，参考'Lift, Splat, Shoot'和'BEVFormer'进行BEV转换，借鉴机器人领域的扩散模型进行运动规划，并采用经典的Stanley控制器进行轨迹跟踪。作者特别设计了'后视目标重标记'的数据增强策略，通过人工扰动目标位置增强数据多样性，提高目标识别的鲁棒性。", '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是使用模块化设计解耦感知和规划，利用视觉基础模型实现跨环境鲁棒感知，通过扩散模型减少运动规划中的累积误差，并结合经典控制器实现精确跟踪。整体流程：1)使用DINOv2处理摄像头图像生成鲁棒特征；2)将特征转换为鸟瞰图(BEV)表示；3)通过交叉注意力和FiLM结构将目标位置信息融合到BEV特征；4)扩散模型基于融合特征和目标位置预测轨迹；5)Stanley控制器根据预测轨迹生成控制命令执行停车。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点：1)首次研究跨域自动驾驶停车问题，实现零样本迁移；2)模块化设计解耦感知和规划，避免过拟合；3)目标重标记数据增强技术提高目标识别鲁棒性；4)在SE(2)空间进行扩散运动规划，包含位置和方向信息；5)在3D高斯飞溅环境中验证模拟到真实世界的迁移能力。相比之前工作，本文方法在跨域场景中表现更好，不需要额外收集不同条件的数据，结合了模块化设计和扩散模型优势，并针对停车任务进行了专门优化。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '这篇论文提出了一种基于视觉基础模型和扩散模型的模块化自动驾驶停车框架，实现了在无需额外数据的情况下跨环境零样本迁移的能力，显著提升了自动驾驶系统在不同天气、光照条件下的停车鲁棒性。'}


### 论文摘要

Parking is a critical pillar of driving safety. While recent end-to-end (E2E) approaches have achieved promising in-domain results, robustness under domain shifts (e.g., weather and lighting changes) remains a key challenge. Rather than relying on additional data, in this paper, we propose Dino-Diffusion Parking (DDP), a domain-agnostic autonomous parking pipeline that integrates visual foundation models with diffusion-based planning to enable generalized perception and robust motion planning under distribution shifts. We train our pipeline in CARLA at regular setting and transfer it to more adversarial settings in a zero-shot fashion. Our model consistently achieves a parking success rate above 90% across all tested out-of-distribution (OOD) scenarios, with ablation studies confirming that both the network architecture and algorithmic design significantly enhance cross-domain performance over existing baselines. Furthermore, testing in a 3D Gaussian splatting (3DGS) environment reconstructed from a real-world parking lot demonstrates promising sim-to-real transfer.

---

## 52. Breakdance Video classification in the age of Generative AI

**论文链接:** [http://arxiv.org/abs/2510.20287v1](http://arxiv.org/abs/2510.20287v1)

**作者:** Sauptik Dhar, Naveen Ramakrishnan, Michelle Munson

**发布时间:** 2025-10-23

**备注:** 11 pages

### GPT解析

### 总结

本研究分析了现代视频基础模型在霹雳舞这一小众但流行的舞蹈体育中的应用性，发现视频编码器模型在预测任务上表现优于最先进的视频语言模型。

### 背景

大型视觉语言模型已在多个体育用例中得到广泛应用，但大多数研究仅针对足球、板球、篮球等流行体育项目，专注于视觉问答和精彩片段生成等生成任务。

### 目的

分析现代视频基础模型（包括编码器和解码器）在霹雳舞这一非常小众但极受欢迎的舞蹈体育中的应用性。

### 方法

评估视频编码器模型和视频语言模型在霹雳舞视频分类任务上的表现，并提供编码器模型选择和微调解码器模型分析的见解。

### 主要发现

视频编码器模型在预测任务上继续优于最先进的视频语言模型，研究提供了如何选择编码器模型的见解，并对微调后的解码器模型在霹雳舞视频分类中的工作机制进行了详细分析。

### 结论

视频编码器模型在特定体育应用（如霹雳舞）中可能比视频语言模型更有效。

### 翻译

大型视觉语言模型最近在多个体育用例中得到了广泛应用。这些工作大多针对足球、板球、篮球等流行体育项目的一个有限子集，专注于视觉问答、精彩片段生成等生成任务。这项工作分析了现代视频基础模型（包括编码器和解码器）在霹雳舞这种非常小众但极受欢迎的舞蹈体育中的应用性。我们的结果表明，视频编码器模型在预测任务上继续优于最先进的视频语言模型。我们提供了如何选择编码器模型的见解，并对微调后的解码器模型在霹雳舞视频分类中的工作机制进行了详细分析。


### 论文摘要

Large Vision Language models have seen huge application in several sports use-cases recently. Most of these works have been targeted towards a limited subset of popular sports like soccer, cricket, basketball etc; focusing on generative tasks like visual question answering, highlight generation. This work analyzes the applicability of the modern video foundation models (both encoder and decoder) for a very niche but hugely popular dance sports - breakdance. Our results show that Video Encoder models continue to outperform state-of-the-art Video Language Models for prediction tasks. We provide insights on how to choose the encoder model and provide a thorough analysis into the workings of a finetuned decoder model for breakdance video classification.

---

## 53. Optimistic Task Inference for Behavior Foundation Models

**论文链接:** [http://arxiv.org/abs/2510.20264v1](http://arxiv.org/abs/2510.20264v1)

**作者:** Thomas Rupf, Marco Bagatella, Marin Vlastelica, Andreas Krause

**发布时间:** 2025-10-23

### GPT解析

### 总结

OpTI-BFM是一种改进的行为基础模型方法，通过直接对奖励函数不确定性建模和乐观决策，实现了在测试时仅通过环境交互来高效推断和优化奖励函数，显著减少了数据需求。

### 背景

行为基础模型(BFMs)能够在测试时直接检索针对任何指定奖励函数的高性能策略，实现零样本强化学习。尽管这种方法在计算上高效，但在数据方面效率较低，因为它通常需要在非平凡的推断数据集上计算奖励，假设可以访问奖励的功能形式或需要大量标注工作。

### 目的

解决BFMs在数据效率方面的问题，使模型能够通过在测试时仅与环境交互来进行任务推断，避免对奖励函数功能形式的依赖或大量标注工作。

### 方法

提出OpTI-BFM，一种乐观决策标准，直接对奖励函数的不确定性进行建模，并指导BFMs进行任务推断的数据收集。通过与线性bandit的上置信度算法的直接连接，为训练良好的BFMs提供了遗憾界限。

### 主要发现

在既定的零样本基准上评估OpTI-BFM后，观察到它使基于后继特征的BFMs能够在少量回合中识别和优化未见过的奖励函数，且计算开销最小。

### 结论

OpTI-BFM解决了传统BFMs在数据效率方面的限制，使其能够在测试时仅通过环境交互来推断和优化任务，显著减少了数据需求。

### 翻译

行为基础模型(BFMs)能够检索针对任何在测试时直接指定的奖励函数的高性能策略，通常被称为零样本强化学习(RL)。虽然这在计算方面是一个非常高效的过程，但在数据方面可能效率较低：作为标准假设，BFMs需要在非平凡的推断数据集上计算奖励，假设可以访问奖励的功能形式或需要大量的标注工作。为了减轻这些限制，我们解决了在测试时仅通过与环境交互来进行任务推断的问题。我们提出了OpTI-BFM，一种乐观决策标准，直接对奖励函数的不确定性进行建模，并指导BFMs进行任务推断的数据收集。形式上，我们通过与线性bandit的上置信度算法的直接连接，为训练良好的BFMs提供了遗憾界限。经验上，我们在既定的零样本基准上评估了OpTI-BFM，并观察到它使基于后继特征的BFMs能够在少量回合中识别和优化未见过的奖励函数，且计算开销最小。代码可在https://github.com/ThomasRupf/opti-bfm获取。


### 论文摘要

Behavior Foundation Models (BFMs) are capable of retrieving high-performing policy for any reward function specified directly at test-time, commonly referred to as zero-shot reinforcement learning (RL). While this is a very efficient process in terms of compute, it can be less so in terms of data: as a standard assumption, BFMs require computing rewards over a non-negligible inference dataset, assuming either access to a functional form of rewards, or significant labeling efforts. To alleviate these limitations, we tackle the problem of task inference purely through interaction with the environment at test-time. We propose OpTI-BFM, an optimistic decision criterion that directly models uncertainty over reward functions and guides BFMs in data collection for task inference. Formally, we provide a regret bound for well-trained BFMs through a direct connection to upper-confidence algorithms for linear bandits. Empirically, we evaluate OpTI-BFM on established zero-shot benchmarks, and observe that it enables successor-features-based BFMs to identify and optimize an unseen reward function in a handful of episodes with minimal compute overhead. Code is available at https://github.com/ThomasRupf/opti-bfm.

---

## 54. There is No "apple" in Timeseries: Rethinking TSFM through the Lens of Invariance

**论文链接:** [http://arxiv.org/abs/2510.20119v1](http://arxiv.org/abs/2510.20119v1)

**作者:** Arian Prabowo, Flora D. Salim

**发布时间:** 2025-10-23

### GPT解析

### 总结

时间序列基础模型(TSFMs)与轻量级监督基线模型和经典模型性能相当，差距源于简单导入NLP或CV流程。时间序列数据不像图像和文本那样直接捕捉人类概念，因此'在线抓取一切'的范式不适用。进步需要从机会性聚合转向原则性设计，构建系统跨越保持时间语义不变性空间的数据集，并基于第一原理构建时间序列不变性本体论，以确保表示完整性，使TSFMs实现泛化、推理和真正涌现行为所需的对齐结构。

### 背景

时间序列基础模型(TSFMs)数量不断增加，但轻量级监督基线模型甚至经典模型通常与它们表现相当。这种差距源于简单导入NLP或CV的流程。

### 目的

提出需要从机会性聚合转向原则性设计：构建数据集，系统性地跨越保持时间语义的不变性空间。

### 方法

建议基于第一原理构建时间序列不变性的本体论，通过不变性覆盖确保表示的完整性。

### 主要发现

在语言和视觉领域，大规模网络语料库密集捕捉人类概念，但时间序列数据没有直接对应的概念，因此'在线抓取一切'的范式对时间序列不适用。

### 结论

只有通过不变性覆盖确保表示的完整性，TSFMs才能实现泛化、推理和真正涌现行为所需的对齐结构。

### 翻译

时间序列基础模型(TSFMs)不断增加，然而轻量级监督基线和甚至经典模型常常与它们匹敌。我们认为这种差距源于简单导入NLP或CV流程。在语言和视觉中，大规模网络语料库密集捕捉人类概念，即有无数的苹果图像和文本。相比之下，时间序列数据设计用来补充图像和文本模态。没有包含'苹果'概念的时间序列数据集。因此，'在线抓取一切'的范式对时间序列不适用。我们认为进步需要从机会性转向原则性设计：构建数据集，系统性地跨越保持时间语义的不变性空间。为此，我们建议时间序列不变性的本体论应基于第一原理构建。只有通过不变性覆盖确保表示的完整性，TSFMs才能实现泛化、推理和真正涌现行为所需的对齐结构。


### 论文摘要

Timeseries foundation models (TSFMs) have multiplied, yet lightweight supervised baselines and even classical models often match them. We argue this gap stems from the naive importation of NLP or CV pipelines. In language and vision, large web-scale corpora densely capture human concepts i.e. there are countless images and text of apples. In contrast, timeseries data is built to complement the image and text modalities. There are no timeseries dataset that contains the concept apple. As a result, the scrape-everything-online paradigm fails for TS. We posit that progress demands a shift from opportunistic aggregation to principled design: constructing datasets that systematically span the space of invariance that preserve temporal semantics. To this end, we suggest that the ontology of timeseries invariances should be built based on first principles. Only by ensuring representational completeness through invariance coverage can TSFMs achieve the aligned structure necessary for generalisation, reasoning, and truly emergent behaviour.

---

## 55. BIOCAP: Exploiting Synthetic Captions Beyond Labels in Biological Foundation Models

**论文链接:** [http://arxiv.org/abs/2510.20095v1](http://arxiv.org/abs/2510.20095v1)

**作者:** Ziheng Zhang, Xinyue Ma, Arpita Chowdhury, Elizabeth G. Campolongo, Matthew J. Thompson, Net Zhang, Samuel Stevens, Hilmar Lapp, Tanya Berger-Wolf, Yu Su, Wei-Lun Chao, Jianyang Gu

**发布时间:** 2025-10-23

**备注:** Project page: https://imageomics.github.io/biocap/

### GPT解析

### 总结

本研究探讨了描述性字幕作为生物多模态基础模型的额外监督来源，通过使用多模态大语言模型生成合成字幕，训练出BIOCAP模型，在物种分类和文本图像检索方面表现优异。

### 背景

图像和字幕可以被视为物种潜在形态空间中的互补样本，各自捕捉特定的生物学特征。在训练中加入字幕可以促进与共享潜在结构的对齐，强调可能有诊断价值的特征，同时抑制虚假相关性。

### 目的

解决生物有机体生物学中大规模获取忠实、实例特定字幕的挑战，以充分利用自然语言监督在生物多模态基础模型中的应用。

### 方法

使用多模态大语言模型(MLLMs)生成合成字幕，这些字幕由维基百科衍生的视觉信息和特定于分类群的格式示例指导，以减少幻觉并产生准确的描述性字幕。然后使用这些字幕训练BIOCAP(即带有字幕的BIOCLIP)模型。

### 主要发现

BIOCAP模型能够捕捉丰富的语义，并在物种分类和文本图像检索方面取得强大的性能。

### 结论

描述性字幕在连接生物图像与多模态基础模型方面具有超越标签的价值。

### 翻译

本研究探讨了描述性字幕作为生物多模态基础模型的额外监督来源。图像和字幕可以被视为物种潜在形态空间中的互补样本，每种都捕捉了特定的生物学特征。在训练中加入字幕可以鼓励与这种共享潜在结构的对齐，强调可能有诊断价值的特征，同时抑制虚假相关性。然而，主要挑战在于大规模获取忠实、实例特定的字幕。这一要求限制了自然语言监督在生物有机体生物学中的应用，与其他许多科学领域相比。我们通过使用多模态大语言模型(MLLMs)生成合成字幕来弥补这一差距，这些字幕由维基百科衍生的视觉信息和特定于分类群的格式示例指导。这些特定领域的上下文有助于减少幻觉，并产生准确、基于实例的描述性字幕。使用这些字幕，我们训练了BIOCAP(即带有字幕的BIOCLIP)，这是一个能够捕捉丰富语义并在物种分类和文本图像检索方面取得强大性能的生物基础模型。这些结果证明了描述性字幕在连接生物图像与多模态基础模型方面超越标签的价值。


### 论文摘要

This work investigates descriptive captions as an additional source of supervision for biological multimodal foundation models. Images and captions can be viewed as complementary samples from the latent morphospace of a species, each capturing certain biological traits. Incorporating captions during training encourages alignment with this shared latent structure, emphasizing potentially diagnostic characters while suppressing spurious correlations. The main challenge, however, lies in obtaining faithful, instance-specific captions at scale. This requirement has limited the utilization of natural language supervision in organismal biology compared with many other scientific domains. We complement this gap by generating synthetic captions with multimodal large language models (MLLMs), guided by Wikipedia-derived visual information and taxon-tailored format examples. These domain-specific contexts help reduce hallucination and yield accurate, instance-based descriptive captions. Using these captions, we train BIOCAP (i.e., BIOCLIP with Captions), a biological foundation model that captures rich semantics and achieves strong performance in species classification and text-image retrieval. These results demonstrate the value of descriptive captions beyond labels in bridging biological images with multimodal foundation models.

---

## 56. Surfer 2: The Next Generation of Cross-Platform Computer Use Agents

**论文链接:** [http://arxiv.org/abs/2510.19949v1](http://arxiv.org/abs/2510.19949v1)

**作者:** Mathieu Andreux, Märt Bakler, Yanael Barbier, Hamza Ben Chekroun, Emilien Biré, Antoine Bonnet, Riaz Bordie, Nathan Bout, Matthias Brunel, Aleix Cambray, Pierre-Louis Cedoz, Antoine Chassang, Gautier Cloix, Ethan Connelly, Alexandra Constantinou, Ramzi De Coster, Hubert de la Jonquiere, Aurélien Delfosse, Maxime Delpit, Alexis Deprez, Augustin Derupti, Mathieu Diaz, Shannon D'Souza, Julie Dujardin, Abai Edmund, Michael Eickenberg, Armand Fatalot, Wissem Felissi, Isaac Herring, Xavier Koegler, Erwan Le Jumeau de Kergaradec, Aurélien Lac, Maxime Langevin, Corentin Lauverjat, Antonio Loison, Avshalom Manevich, Axel Moyal, Axel Nguyen Kerbel, Marinela Parovic, Julien Revelle, Guillaume Richard, Mats Richter, Ronan Riochet, María Santos, Romain Savidan, Laurent Sifre, Maxime Theillard, Marc Thibault, Ivan Valentini, Tony Wu, Laura Yie, Kai Yuan, Jevgenij Zubovskij

**发布时间:** 2025-10-22

**备注:** 21 pages, 9 figures, 2 tables

### GPT解析

### 总结

Surfer 2是一个统一的架构，仅从视觉观察操作，在Web、桌面和移动三种环境中实现了最先进的性能，无需任务特定微调即可超越人类表现。

### 背景

构建能够在网络、桌面和移动环境中通用的代理仍然是一个开放的挑战，因为之前的系统依赖于特定环境的接口，限制了跨平台部署。

### 目的

介绍Surfer 2，一个统一的架构，仅从视觉观察操作，在所有三种环境中实现最先进的性能。

### 方法

Surfer 2集成了分层上下文管理、解耦的规划和执行，以及自适应恢复的自验证， enabling可靠操作在长任务范围内。

### 主要发现

在WebVoyager上达到97.1%的准确率，在WebArena上达到69.6%的准确率，在OSWorld上达到60.1%的准确率，在AndroidWorld上达到87.1%的准确率，超越了所有之前的系统，无需任务特定的微调，多次尝试后，Surfer 2在所有基准测试中超过了人类性能。

### 结论

这些结果表明，系统编排增强了基础模型的能力，仅通过视觉交互实现了通用计算机控制，同时呼吁新一代视觉语言模型以实现帕累托最优的成本效益。

### 翻译

构建能够在网络、桌面和移动环境中通用的代理仍然是一个开放的挑战，因为之前的系统依赖于特定环境的接口，这限制了跨平台部署。我们介绍了Surfer 2，一个统一的架构，仅从视觉观察操作，在所有三种环境中实现最先进的性能。Surfer 2集成了分层上下文管理、解耦的规划和执行，以及自适应恢复的自验证， enabling可靠操作在长任务范围内。我们的系统在WebVoyager上达到97.1%的准确率，在WebArena上达到69.6%，在OSWorld上达到60.1%，在AndroidWorld上达到87.1%，超越了所有之前的系统，无需任务特定的微调。多次尝试后，Surfer 2在所有基准测试中超过了人类性能。这些结果表明，系统编排增强了基础模型的能力，仅通过视觉交互实现了通用计算机控制，同时呼吁新一代视觉语言模型以实现帕累托最优的成本效益。


### 论文摘要

Building agents that generalize across web, desktop, and mobile environments remains an open challenge, as prior systems rely on environment-specific interfaces that limit cross-platform deployment. We introduce Surfer 2, a unified architecture operating purely from visual observations that achieves state-of-the-art performance across all three environments. Surfer 2 integrates hierarchical context management, decoupled planning and execution, and self-verification with adaptive recovery, enabling reliable operation over long task horizons. Our system achieves 97.1% accuracy on WebVoyager, 69.6% on WebArena, 60.1% on OSWorld, and 87.1% on AndroidWorld, outperforming all prior systems without task-specific fine-tuning. With multiple attempts, Surfer 2 exceeds human performance on all benchmarks. These results demonstrate that systematic orchestration amplifies foundation model capabilities and enables general-purpose computer control through visual interaction alone, while calling for a next-generation vision language model to achieve Pareto-optimal cost-efficiency.

---

## 57. Seed3D 1.0: From Images to High-Fidelity Simulation-Ready 3D Assets

**论文链接:** [http://arxiv.org/abs/2510.19944v1](http://arxiv.org/abs/2510.19944v1)

**作者:** Jiashi Feng, Xiu Li, Jing Lin, Jiahang Liu, Gaohong Liu, Weiqiang Lou, Su Ma, Guang Shi, Qinlong Wang, Jun Wang, Zhongcong Xu, Xuanyu Yi, Zihao Yu, Jianfeng Zhang, Yifan Zhu, Rui Chen, Jinxin Chi, Zixian Du, Li Han, Lixin Huang, Kaihua Jiang, Yuhan Li, Guan Luo, Shuguang Wang, Qianyi Wu, Fan Yang, Junyang Zhang, Xuanmeng Zhang

**发布时间:** 2025-10-22

**备注:** Seed3D 1.0 Technical Report; Official Page on  https://seed.bytedance.com/seed3d

### GPT解析

### 总结

Seed3D 1.0是一个基础模型，可以从单张图像生成可用于仿真的3D资产，解决了具身AI代理训练环境中的可扩展性问题，同时保持物理准确性。

### 背景

开发具身AI代理需要平衡内容多样性和物理准确性的可扩展训练环境。现有世界模拟器存在局限：基于视频的方法内容多样但缺乏实时物理反馈，基于物理的引擎物理准确但受限于昂贵的手动资产创建。

### 目的

开发一个能够从单张图像生成仿真就绪3D资产的基础模型，解决可扩展性挑战，同时保持物理严谨性。

### 方法

提出Seed3D 1.0基础模型，生成具有准确几何、良好对齐纹理和真实物理材质的3D资产，可直接集成到物理引擎中，支持机器人操作和仿真训练，并能扩展到完整场景生成。

### 主要发现

Seed3D 1.0生成的3D资产具有准确的几何结构、对齐良好的纹理和真实的物理材质，可直接用于物理引擎，支持机器人操作和仿真训练，并能扩展到完整场景生成。

### 结论

Seed3D 1.0通过实现可扩展的仿真就绪内容创建，为推进基于物理的世界模拟器提供了基础，现已可在指定网址获取。

### 翻译

开发具身AI代理需要可扩展的训练环境，这些环境需要在内容多样性和物理准确性之间取得平衡。世界模拟器提供了这样的环境，但面临不同的限制：基于视频的方法可以生成多样化的内容，但缺乏实时物理反馈以支持交互式学习；而基于物理的引擎能提供准确的动力学，但由于昂贵的手动资产创建而面临可扩展性限制。我们提出了Seed3D 1.0，这是一个基础模型，可以从单张图像生成仿真就绪的3D资产，解决了可扩展性挑战，同时保持物理严谨性。与现有的3D生成模型不同，我们的系统生成具有准确几何、对齐良好的纹理和真实物理材质的资产。这些资产可以直接集成到物理引擎中，只需最少的配置，支持在机器人操作和仿真训练中部署。除了单个对象外，系统还能通过将对象组装成连贯的环境来扩展到完整场景生成。通过实现可扩展的仿真就绪内容创建，Seed3D 1.0为推进基于物理的世界模拟器提供了基础。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决如何从单张图像生成高质量的、可直接用于物理仿真的3D资产的问题。这个问题重要是因为开发具身AI代理需要可扩展的训练环境，平衡内容多样性和物理准确性，而现有世界模拟器面临根本性权衡：基于视频的方法缺乏实时物理反馈，基于物理的引擎则受限于手动资产创建的可扩展性，制约了训练环境的多样性和规模。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者通过分析现有世界模拟器的局限性，认识到需要结合生成多样性和物理严谨性，设计了Seed3D 1.0基础模型。系统借鉴了现有工作：几何生成部分采用VAE和基于修正流的扩散Transformer架构；使用DINOv2和RADIO作为图像编码器；纹理生成部分借鉴多模态扩散Transformer；数据预处理参考3DShape2VecSet设计；训练基础设施采用FlashAttention和混合分片数据并行等技术。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是通过结合生成多样性和物理严谨性，解决3D资产创建的可扩展性问题，生成可直接用于物理仿真的高质量3D资产。整体流程包括：1)几何生成：使用Seed3D-VAE学习紧凑潜在表示，Seed3D-DiT合成3D形状；2)纹理生成：Seed3D-MV生成多视图图像，Seed3D-PBR分解为PBR材质图，Seed3D-UV补全UV纹理；3)数据处理：自动化预处理管道、格式标准化和质量过滤；4)训练和推理：采用渐进式策略训练模型，通过多阶段处理生成最终3D资产。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)高质量资产生成：生成具有精确几何、高分辨率纹理和真实物理材质的3D资产；2)物理引擎兼容性：资产可直接集成到物理引擎中；3)可扩展场景合成：从室内到城市环境实现连贯场景；4)技术创新：开发了Seed3D-VAE、Seed3D-DiT、Seed3D-MV、Seed3D-PBR和Seed3D-UV五个核心组件。相比之前工作，解决了几何伪影和纹理错位问题，通过UV纹理补全解决自遮挡，采用混合架构平衡跨模态学习和模态特定处理，使用长度感知时间步长维持生成质量。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'Seed3D 1.0通过从单张图像生成高质量的、物理兼容的3D资产，解决了具身AI训练环境中内容多样性和物理准确性之间的权衡问题，为物理驱动的世界模拟器提供了可扩展的基础。'}


### 论文摘要

Developing embodied AI agents requires scalable training environments that balance content diversity with physics accuracy. World simulators provide such environments but face distinct limitations: video-based methods generate diverse content but lack real-time physics feedback for interactive learning, while physics-based engines provide accurate dynamics but face scalability limitations from costly manual asset creation. We present Seed3D 1.0, a foundation model that generates simulation-ready 3D assets from single images, addressing the scalability challenge while maintaining physics rigor. Unlike existing 3D generation models, our system produces assets with accurate geometry, well-aligned textures, and realistic physically-based materials. These assets can be directly integrated into physics engines with minimal configuration, enabling deployment in robotic manipulation and simulation training. Beyond individual objects, the system scales to complete scene generation through assembling objects into coherent environments. By enabling scalable simulation-ready content creation, Seed3D 1.0 provides a foundation for advancing physics-based world simulators. Seed3D 1.0 is now available on https://console.volcengine.com/ark/region:ark+cn-beijing/experience/vision?modelId=doubao-seed3d-1-0-250928&tab=Gen3D

---

## 58. FairGRPO: Fair Reinforcement Learning for Equitable Clinical Reasoning

**论文链接:** [http://arxiv.org/abs/2510.19893v1](http://arxiv.org/abs/2510.19893v1)

**作者:** Shiqi Dai, Wei Dai, Jiaee Cheong, Paul Pu Liang

**发布时间:** 2025-10-22

**备注:** Accepted as Oral on NeurIPS 2025 GenAI4Health Workshop

### GPT解析

### 总结

研究提出了一种名为FairGRPO的分层强化学习方法，用于提高医学人工智能系统在不同人口统计群体中的诊断公平性，通过自适应重要性加权和无监督聚类处理缺失标签问题，实验表明该方法显著提高了预测平等性和F1分数。

### 背景

医学人工智能系统在诊断方面取得显著成就，但在不同人口统计群体中表现出明显的性能差异，对代表性不足人群造成实际伤害。多模态推理基础模型推动了临床诊断，但通过强化学习进行的推理训练继承了并放大了训练数据集中的偏见。

### 目的

提出一种促进跨异质临床人群公平学习的方法，解决临床领域常见的缺乏人口统计标签的问题。

### 方法

引入Fairness-aware Group Relative Policy Optimization (FairGRPO)，一种分层强化学习方法，采用基于代表性、任务难度和数据源的自适应优势重要性加权。采用无监督聚类来处理缺失的人口统计标签，当标签不可用时自动发现潜在的人口统计群体。

### 主要发现

在7个涵盖5种临床模态的临床诊断数据集上，FairGRPO与所有普通和偏见缓解的强化学习基线相比，将预测平等性提高了27.2%，同时F1分数提高了12.49%。训练动态分析显示，FairGRPO在整个优化过程中逐步改善公平性，而基线强化学习方法在训练过程中表现出公平性恶化。基于FairGRPO，发布了FairMedGemma-4B，一个公平感知的临床VLLM，在实现最先进性能的同时显著减少了不同人口统计群体之间的差异。

### 结论

FairGRPO是一种有效的医学人工智能系统公平性提升方法，能够在不牺牲性能的情况下提高跨人群的诊断公平性，解决了临床数据中缺乏人口统计标签的常见问题。

### 翻译

医学人工智能系统已取得显著的诊断能力，然而它们在不同人口统计群体中持续表现出性能差异，对代表性不足的人群造成实际伤害。虽然最近的多模态推理基础模型通过整合分析多样化的医疗数据推动了临床诊断的发展，但通过强化学习进行的推理训练继承了主导多数人群的训练数据集中存在的偏见，并往往放大这些偏见。我们引入了公平感知的群体相对策略优化（FairGRPO），这是一种分层强化学习方法，促进跨异质临床人群的公平学习。FairGRPO基于代表性、任务难度和数据源采用自适应的优势重要性加权。为解决临床领域中常见的人口统计标签缺失问题，我们进一步采用无监督聚类，当标签不可用时自动发现潜在的人口统计群体。在跨越X光、CT扫描、皮肤镜检查、乳腺X光检查和超声波5种临床模态的7个临床诊断数据集上进行综合实验，我们证明FairGRPO与所有普通和偏见缓解的强化学习基线相比，将预测平等性提高了27.2%，同时F1分数提高了12.49%。此外，训练动态分析显示，FairGRPO在整个优化过程中逐步改善公平性，而基线强化学习方法在训练过程中表现出公平性恶化。基于FairGRPO，我们发布了FairMedGemma-4B，一个公平感知的临床VLLM，在实现最先进性能的同时，显著减少了不同人口统计群体之间的差异。


### 论文摘要

Medical artificial intelligence systems have achieved remarkable diagnostic capabilities, yet they consistently exhibit performance disparities across demographic groups, causing real-world harm to underrepresented populations. While recent multimodal reasoning foundation models have advanced clinical diagnosis through integrated analysis of diverse medical data, reasoning trainings via reinforcement learning inherit and often amplify biases present in training datasets dominated by majority populations. We introduce Fairness-aware Group Relative Policy Optimization (FairGRPO), a hierarchical reinforcement learning approach that promotes equitable learning across heterogeneous clinical populations. FairGRPO employs adaptive importance weighting of advantages based on representation, task difficulty, and data source. To address the common issue of missing demographic labels in the clinical domain, we further employ unsupervised clustering, which automatically discovers latent demographic groups when labels are unavailable. Through comprehensive experiments across 7 clinical diagnostic datasets spanning 5 clinical modalities across X-ray, CT scan, dermoscropy, mammography and ultrasound, we demonstrate that FairGRPO reduces predictive parity by 27.2% against all vanilla and bias mitigated RL baselines, while improving F1 score by 12.49%. Furthermore, training dynamics analysis reveals that FairGRPO progressively improves fairness throughout optimization, while baseline RL methods exhibit deteriorating fairness as training progresses. Based on FairGRPO, we release FairMedGemma-4B, a fairness-aware clinical VLLM that achieves state-of-the-art performance while demonstrating significantly reduced disparities across demographic groups.

---

## 59. SEMPO: Lightweight Foundation Models for Time Series Forecasting

**论文链接:** [http://arxiv.org/abs/2510.19710v1](http://arxiv.org/abs/2510.19710v1)

**作者:** Hui He, Kun Yi, Yuanchi Ma, Qi Zhang, Zhendong Niu, Guansong Pang

**发布时间:** 2025-10-22

**备注:** Accepted by NeurIPS 2025

### GPT解析

### 总结

SEMPO是一种新型轻量级时间序列预测基础模型，通过两个创新模块在减少预训练数据规模和模型大小的同时实现强大的预测性能。

### 背景

现有的时间序列基础模型虽然性能出色，但网络架构庞大，需要大规模数据集进行预训练，难以在资源受限环境中部署。

### 目的

开发一种多功能且经济实惠的时间序列基础模型，解决现有模型在多功能性和可负担性之间的矛盾。

### 方法

SEMPO包含两个关键模块：(1)能量感知的频谱分解模块，同时建模高能量和低能量但信息丰富的频率信号；(2)基于提示混合的Transformer，通过小型数据集特定提示学习异构时间模式，实现参数高效模型适应。

### 主要发现

SEMPO在两个大规模基准测试(包含16个数据集)上的实验表明，与最先进方法相比，它在零样本和少样本预测场景中表现出优越性能，同时显著减少了预训练数据规模和模型大小。

### 结论

SEMPO成功实现了时间序列预测领域多功能性和可负担性的平衡，为资源受限环境中的时间序列预测提供了有效解决方案。

### 翻译

最近大型预训练模型的兴起见证了在时间序列预测领域开发基础模型的显著成功。尽管在各种下游预测任务中表现出令人印象深刻的性能，但现有时间序列基础模型拥有庞大的网络架构，需要在大规模数据集上进行大量预训练，这严重阻碍了它们在资源受限环境中的部署。为了应对多功能性和可负担性之间日益加剧的矛盾，我们提出了SEMPO，一种新型轻量级基础模型，它只需要在相对小规模的数据上进行预训练，却表现出强大的通用时间序列预测能力。具体而言，SEMPO包含两个关键模块：1)能量感知的频谱分解模块，通过不仅建模高能量频率信号，还建模当前方法中被忽略的低能量但信息丰富的频率信号，显著提高了预训练数据的利用率；以及2)基于提示混合的Transformer，通过小型数据集特定的提示学习异构时间模式，并将时间序列标记自适应路由到基于提示的专家，实现跨不同数据集和领域的参数高效模型适应。配备这些模块后，SEMPO显著减少了预训练数据规模和模型大小，同时实现了强大的泛化能力。在覆盖16个数据集的两个大规模基准测试上进行的广泛实验表明，与最先进的方法相比，SEMPO在零样本和少样本预测场景中表现出优越性能。代码和数据可在https://github.com/mala-lab/SEMPO获取。


### 论文摘要

The recent boom of large pre-trained models witnesses remarkable success in developing foundation models (FMs) for time series forecasting. Despite impressive performance across diverse downstream forecasting tasks, existing time series FMs possess massive network architectures and require substantial pre-training on large-scale datasets, which significantly hinders their deployment in resource-constrained environments. In response to this growing tension between versatility and affordability, we propose SEMPO, a novel lightweight foundation model that requires pretraining on relatively small-scale data, yet exhibits strong general time series forecasting. Concretely, SEMPO comprises two key modules: 1) energy-aware SpEctral decomposition module, that substantially improves the utilization of pre-training data by modeling not only the high-energy frequency signals but also the low-energy yet informative frequency signals that are ignored in current methods; and 2) Mixture-of-PrOmpts enabled Transformer, that learns heterogeneous temporal patterns through small dataset-specific prompts and adaptively routes time series tokens to prompt-based experts for parameter-efficient model adaptation across different datasets and domains. Equipped with these modules, SEMPO significantly reduces both pre-training data scale and model size, while achieving strong generalization. Extensive experiments on two large-scale benchmarks covering 16 datasets demonstrate the superior performance of SEMPO in both zero-shot and few-shot forecasting scenarios compared with state-of-the-art methods. Code and data are available at https://github.com/mala-lab/SEMPO.

---

## 60. Detecting Latin in Historical Books with Large Language Models: A Multimodal Benchmark

**论文链接:** [http://arxiv.org/abs/2510.19585v1](http://arxiv.org/abs/2510.19585v1)

**作者:** Yu Wu, Ke Shu, Jonas Fischer, Lidia Pivovarova, David Rosson, Eetu Mäkelä, Mikko Tolonen

**发布时间:** 2025-10-22

**备注:** Under review. Both the dataset and code will be published

### GPT解析

### 总结

本文提出了一项从混合语言历史文档中提取拉丁语片段的新任务，并评估了大型基础模型在此任务上的性能。

### 背景

历史文档通常包含多种语言和不同的布局，从中提取特定语言片段具有挑战性。

### 目的

评估大型基础模型在从混合语言历史文档中提取拉丁语片段任务上的能力和局限性。

### 方法

使用包含724个标注页面的多模态数据集，对大型基础模型进行了基准测试和性能评估。

### 主要发现

当代模型能够可靠地检测和提取拉丁语片段。

### 结论

该研究首次全面分析了大型基础模型在从混合语言历史文档中提取拉丁语片段任务上的能力和局限性。

### 翻译

本文提出了一项从混合语言历史文档中提取拉丁语片段的新任务，这些文档具有不同的布局。我们使用一个包含724个标注页面的多模态数据集，对大型基础模型进行了基准测试和性能评估。结果表明，使用当代模型进行可靠的拉丁语检测是可行的。我们的研究首次对这些模型在此任务上的能力和局限性进行了全面分析。


### 论文摘要

This paper presents a novel task of extracting Latin fragments from mixed-language historical documents with varied layouts. We benchmark and evaluate the performance of large foundation models against a multimodal dataset of 724 annotated pages. The results demonstrate that reliable Latin detection with contemporary models is achievable. Our study provides the first comprehensive analysis of these models' capabilities and limits for this task.

---

## 61. GigaBrain-0: A World Model-Powered Vision-Language-Action Model

**论文链接:** [http://arxiv.org/abs/2510.19430v1](http://arxiv.org/abs/2510.19430v1)

**作者:** GigaBrain Team, Angen Ye, Boyuan Wang, Chaojun Ni, Guan Huang, Guosheng Zhao, Haoyun Li, Jie Li, Jiagang Zhu, Lv Feng, Peng Li, Qiuping Deng, Runqi Ouyang, Wenkang Qin, Xinze Chen, Xiaofeng Wang, Yang Wang, Yifan Li, Yilong Li, Yiran Ding, Yuan Xu, Yun Ye, Yukun Zhou, Zhehao Dong, Zhenan Wang, Zhichao Liu, Zheng Zhu

**发布时间:** 2025-10-22

**备注:** https://gigabrain0.github.io/

### GPT解析

### 总结

这篇论文介绍了GigaBrain-0，一个利用世界模型生成数据的新型视觉-语言-动作(VLA)基础模型，减少了对真实机器人数据的依赖，提高了跨任务泛化能力和策略鲁棒性，在灵巧操作、长视野和移动操作任务中实现了显著性能提升。

### 背景

为通用机器人训练视觉-语言-动作(VLA)模型通常需要大规模的真实世界机器人数据，这些数据的收集既昂贵又耗时。物理数据收集的低效严重限制了当前VLA系统的可扩展性和泛化能力。

### 目的

解决物理数据收集低效的问题，减少对真实机器人数据的依赖，同时提高VLA系统的跨任务泛化能力和策略鲁棒性。

### 方法

1. 引入GigaBrain-0，由世界模型生成数据(如视频生成、真实到真实转移、人类转移、视角转移、仿真到真实转移数据)赋能的新型VLA基础模型；2. 利用世界模型大规模生成多样化数据；3. 通过RGBD输入建模和具身思维链(CoT)监督提高策略鲁棒性；4. 开发了GigaBrain-0-Small，一个优化的轻量级变体，可在NVIDIA Jetson AGX Orin等设备上高效运行。

### 主要发现

1. GigaBrain-0在灵巧操作、长视野和移动操作任务中实现了显著的性能提升；2. 广泛的实验证明GigaBrain-0在外观(如纹理、颜色)、物体放置和摄像机视点变化方面具有优越的泛化能力。

### 结论

通过利用世界模型生成数据，GigaBrain-0显著减少了对真实机器人数据的依赖，同时提高了跨任务泛化能力和策略鲁棒性，为通用机器人提供了一个更高效、更可扩展的VLA解决方案。

### 翻译

为通用机器人训练视觉-语言-动作(VLA)模型通常需要大规模的真实世界机器人数据，这些数据的收集既昂贵又耗时。物理数据收集的低效严重限制了当前VLA系统的可扩展性和泛化能力。为应对这一挑战，我们引入了GigaBrain-0，一个由世界模型生成数据(如视频生成、真实到真实转移、人类转移、视角转移、仿真到真实转移数据)赋能的新型VLA基础模型。通过利用世界模型大规模生成多样化数据，GigaBrain-0显著减少了对真实机器人数据的依赖，同时提高了跨任务泛化能力。我们的方法通过RGBD输入建模和具身思维链(CoT)监督进一步提高了策略鲁棒性，使模型能够在任务执行过程中推理空间几何、物体状态和长视野依赖关系。这导致在灵巧操作、长视野和移动操作任务中的实际性能显著提升。广泛的实验证明，GigaBrain-0在外观(如纹理、颜色)、物体放置和摄像机视点变化方面实现了优越的泛化能力。此外，我们提出了GigaBrain-0-Small，一个优化的轻量级变体，设计用于在NVIDIA Jetson AGX Orin等设备上高效运行。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决训练视觉-语言-行动（VLA）模型需要大规模真实世界机器人数据的问题，而收集这些数据既昂贵又耗时。这个问题很重要，因为它严重限制了当前VLA系统的可扩展性和泛化能力，阻碍了通用机器人在多样化环境中的应用。缺乏足够多样性的训练数据导致模型在现实世界中表现不佳，限制了机器人技术的实际部署。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先认识到真实世界机器人数据收集的局限性，然后提出利用世界模型生成多样化训练数据的解决方案。他们设计了混合transformer架构，结合预训练视觉语言模型和动作扩散变换器，并引入RGB-D输入和具身思维链机制。作者借鉴了多项现有工作，包括π0等VLA模型架构、世界模型作为数据生成器、视觉语言模型如PaliGemma2、扩散模型用于视频生成，以及思维链推理和知识隔离技术。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是利用世界模型生成多样化、逼真的合成训练数据，减少对真实世界机器人数据的依赖，并通过RGB-D输入和具身思维链增强模型的感知和推理能力。整体流程包括：1）收集真实世界数据并利用GigaWorld生成多种合成数据（Real2Real转移、视图转移等）；2）采用混合transformer架构处理RGB-D输入和语言指令；3）训练模型生成具身思维链作为中间表示；4）基于思维链输出连续动作序列；5）提供轻量级版本适配边缘设备。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1）整合多种世界模型生成的数据源，增强数据多样性；2）RGB-D输入建模提升3D空间理解；3）具身思维链监督机制改善推理能力；4）混合架构与知识隔离技术提高训练效率；5）高效数据生成与质量评估。相比之前工作，GigaBrain-0利用了更多样化的数据源（包括视图转移和Real2Real转移），具有更强的3D空间理解能力，能显式生成中间推理步骤，训练效率更高，在变化条件下泛化能力更强。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'GigaBrain-0通过创新性地整合世界模型生成的多样化训练数据和具身思维链推理机制，显著提升了视觉-语言-行动模型在真实世界任务中的泛化能力和执行效率，同时大幅减少了对昂贵真实世界机器人数据的依赖。'}


### 论文摘要

Training Vision-Language-Action (VLA) models for generalist robots typically requires large-scale real-world robot data, which is expensive and time-consuming to collect. The inefficiency of physical data collection severely limits the scalability, and generalization capacity of current VLA systems. To address this challenge, we introduce GigaBrain-0, a novel VLA foundation model empowered by world model-generated data (e.g., video generation, real2real transfer, human transfer, view transfer, sim2real transfer data). By leveraging world models to generate diverse data at scale, GigaBrain-0 significantly reduces reliance on real robot data while improving cross-task generalization. Our approach further improves policy robustness through RGBD input modeling and embodied Chain-of-Thought (CoT) supervision, enabling the model to reason about spatial geometry, object states, and long-horizon dependencies during task execution. This leads to substantial gains in real-world performance on dexterous, long-horizon, and mobile manipulation tasks. Extensive experiments demonstrate that GigaBrain-0 achieves superior generalization across variations in appearances (e.g., textures, colors), object placements, and camera viewpoints. Additionally, we present GigaBrain-0-Small, an optimized lightweight variant designed to run efficiently on devices such as the NVIDIA Jetson AGX Orin.

---

## 62. Using Temperature Sampling to Effectively Train Robot Learning Policies on Imbalanced Datasets

**论文链接:** [http://arxiv.org/abs/2510.19373v1](http://arxiv.org/abs/2510.19373v1)

**作者:** Basavasagar Patil, Sydney Belt, Jayjun Lee, Nima Fazeli, Bernadette Bucher

**发布时间:** 2025-10-22

### GPT解析

### 总结

本文提出了一种简单的策略训练采样方法，用于缓解机器人任务数据集中物理动作序列的不平衡问题。

### 背景

随着越来越多的机器人动作和感官观测数据集被收集用于训练大型神经网络，发现许多基于不同描述的任务实际上涉及非常相似的身体动作序列，导致数据集在物理机器人动作方面存在严重不平衡。

### 目的

提出一种简单的采样策略来缓解机器人任务数据集中的动作不平衡问题，提高模型在多任务场景下的泛化能力。

### 方法

提出一种简单的策略训练采样方法，只需几行代码即可集成到现有代码库中，并在预训练小型模型和微调大型基础模型上进行了评估。

### 主要发现

与之前最先进的方法相比，该方法在低资源任务上取得了显著改进，同时没有降低高资源任务上的性能，使得模型容量能够更有效地用于多任务策略。

### 结论

在Franka Panda机械臂上的多样化任务中进一步验证了该方法的有效性，证明了其在实际应用中的可行性。

### 翻译

越来越多的机器人动作和感官观测数据集被收集起来，用于训练日益庞大的神经网络。这些数据集是基于任务收集的，尽管这些任务在描述上可能不同，但许多任务涉及非常相似的身体动作序列（例如，'拿起苹果'与'拿起橙子'）。因此，许多机器人任务数据集在所代表的物理机器人动作方面存在严重不平衡。在这项工作中，我们提出了一种简单的策略训练采样方法来缓解这种不平衡。我们的方法只需要几行代码就可以集成到现有代码库中，并提高了泛化能力。我们在预训练小型模型和微调大型基础模型上都评估了我们的方法。结果表明，与之前最先进的方法相比，在低资源任务上取得了显著改进，同时没有降低高资源任务上的性能。这使得模型容量能够更有效地用于多任务策略。我们还进一步在Franka Panda机械臂上的多样化任务设置中验证了我们的方法。


### 论文摘要

Increasingly large datasets of robot actions and sensory observations are being collected to train ever-larger neural networks. These datasets are collected based on tasks and while these tasks may be distinct in their descriptions, many involve very similar physical action sequences (e.g., 'pick up an apple' versus 'pick up an orange'). As a result, many datasets of robotic tasks are substantially imbalanced in terms of the physical robotic actions they represent. In this work, we propose a simple sampling strategy for policy training that mitigates this imbalance. Our method requires only a few lines of code to integrate into existing codebases and improves generalization. We evaluate our method in both pre-training small models and fine-tuning large foundational models. Our results show substantial improvements on low-resource tasks compared to prior state-of-the-art methods, without degrading performance on high-resource tasks. This enables more effective use of model capacity for multi-task policies. We also further validate our approach in a real-world setup on a Franka Panda robot arm across a diverse set of tasks.

---

## 63. AMAuT: A Flexible and Efficient Multiview Audio Transformer Framework Trained from Scratch

**论文链接:** [http://arxiv.org/abs/2510.19368v1](http://arxiv.org/abs/2510.19368v1)

**作者:** Weichuang Shao, Iman Yi Liao, Tomas Henrique Bode Maul, Tissa Chandesa

**发布时间:** 2025-10-22

### GPT解析

### 总结

本文提出了AMAuT框架，一个无需预训练权重且支持任意采样率和音频长度的音频模型，在多个基准测试中达到高准确度同时大幅减少计算资源消耗。

### 背景

最近的SSAST、EAT、HuBERT、Qwen-Audio和AudioFlamingo等基础模型在标准音频基准测试中表现优异，但受限于固定的输入速率和持续时间，影响了它们的重用性。

### 目的

开发一个无需依赖预训练权重、支持任意采样率和音频长度的音频分类框架，提高模型的灵活性和效率。

### 方法

AMAuT集成了四个关键组件：增强驱动的多视图学习提高鲁棒性；conv1+conv7+conv1一维CNN瓶颈实现稳定的时间编码；双CLS+TAL令牌进行双向上下文表示；测试时自适应/增强(TTA²)提高推理可靠性。

### 主要发现

在AudioMNIST、SpeechCommands V1&V2、VocalSound和CochlScene五个公共基准测试上，AMAuT准确度高达99.8%，同时消耗的GPU小时数不到可比预训练模型的3%。

### 结论

AMAuT为大型预训练模型提供了一个高效且灵活的替代方案，使最先进的音频分类在计算受限环境中变得可行。

### 翻译

最近的SSAST、EAT、HuBERT、Qwen-Audio和AudioFlamingo等基础模型在标准音频基准测试中取得了顶尖结果，但受限于固定的输入速率和持续时间，阻碍了它们的重用性。本文引入了增强驱动多视图音频变换器(AMAuT)，这是一个从头开始训练的框架，消除对预训练权重的依赖，同时支持任意采样率和音频长度。AMAuT集成了四个关键组件：(1)增强驱动的多视图学习，提高鲁棒性；(2)conv1+conv7+conv1一维CNN瓶颈，用于稳定的时间编码；(3)双CLS+TAL令牌，用于双向上下文表示；(4)测试时自适应/增强(TTA²)，提高推理可靠性。在AudioMNIST、SpeechCommands V1&V2、VocalSound和CochlScene五个公共基准测试上的实验表明，AMAuT准确度高达99.8%，同时消耗的GPU小时数不到可比预训练模型的3%。因此，AMAuT为大型预训练模型提供了一个高效且灵活的替代方案，使最先进的音频分类在计算受限环境中变得可行。


### 论文摘要

Recent foundational models, SSAST, EAT, HuBERT, Qwen-Audio, and Audio Flamingo, achieve top-tier results across standard audio benchmarks but are limited by fixed input rates and durations, hindering their reusability. This paper introduces the Augmentation-driven Multiview Audio Transformer (AMAuT), a training-from-scratch framework that eliminates the dependency on pre-trained weights while supporting arbitrary sample rates and audio lengths. AMAuT integrates four key components: (1) augmentation-driven multiview learning for robustness, (2) a conv1 + conv7 + conv1 one-dimensional CNN bottleneck for stable temporal encoding, (3) dual CLS + TAL tokens for bidirectional context representation, and (4) test-time adaptation/augmentation (TTA^2) to improve inference reliability. Experiments on five public benchmarks, AudioMNIST, SpeechCommands V1 & V2, VocalSound, and CochlScene, show that AMAuT achieves accuracies up to 99.8% while consuming less than 3% of the GPU hours required by comparable pre-trained models. Thus, AMAuT presents a highly efficient and flexible alternative to large pre-trained models, making state-of-the-art audio classification accessible in computationally constrained settings.

---

## 64. Foundation Model Forecasts: Form and Function

**论文链接:** [http://arxiv.org/abs/2510.19345v1](http://arxiv.org/abs/2510.19345v1)

**作者:** Alvaro Perez-Diaz, James C. Loach, Danielle E. Toutoungi, Lee Middleton

**发布时间:** 2025-10-22

**备注:** 28 pages, 3 figures

### GPT解析

### 总结

时间序列基础模型(TSFMs)虽然预测准确性高，但预测形式（点预测、分位数预测、参数化预测或轨迹集合）决定了其实际应用价值。研究发现大多数TSFMs只能提供点或参数化预测，而实际操作任务常需要保留时间依赖性的轨迹集合。研究确定了预测类型间的转换条件，证明边际分布无法确定路径相关事件概率，并将六个基本预测任务映射到最小充分预测类型，表明预测类型而非准确性才是区分模型实用价值的关键。

### 背景

时间序列基础模型(TSFMs)在预测准确性方面表现出色，但准确性并不完全决定其实际价值。

### 目的

研究不同预测形式对实际操作任务的支持能力，确定预测类型间的转换条件，并提供任务对齐的评估框架。

### 方法

分析现有TSFMs的预测类型，研究预测类型之间的转换条件，证明边际分布与联合分布的关系，并将基本预测任务映射到最小充分预测类型。

### 主要发现

1. 三分之二的TSFMs只产生点或参数化预测，而许多操作任务需要保留时间依赖性的轨迹集合；2. 轨迹集合可通过边际化转换为简单形式，但反向转换需要额外方法；3. 边际分布无法确定路径相关事件概率，无限多联合分布可具有相同边际分布但给出不同操作答案；4. 六个基本预测任务可映射到最小充分预测类型。

### 结论

在实际应用中，预测类型而非准确性是区分模型实用价值的关键因素。选择适当的预测形式对于支持特定操作任务至关重要。

### 翻译

时间序列基础模型(TSFMs)实现了强大的预测准确性，然而准确性本身并不决定实际价值。预测的形式——点预测、分位数预测、参数化预测或轨迹集合——从根本上限制了它能够支持的操作任务。我们调查了最近的TSFMs，发现三分之二只产生点预测或参数化预测，而许多操作任务需要保留时间依赖性的轨迹集合。我们确定了预测类型何时可以转换、何时不可以转换：轨迹集合可以通过边际化转换为更简单的形式而无需额外假设，但反向转换则需要通过copulas或conformal方法施加时间依赖性。我们证明了边际分布无法确定路径相关事件概率——无限多的联合分布具有相同的边际分布，但对操作问题给出不同的答案。我们将六个基本预测任务映射到最小充分预测类型，并提供了任务对齐的评估框架。我们的分析阐明了当预测类型而非准确性区分实用价值时的情况。


### 论文摘要

Time-series foundation models (TSFMs) achieve strong forecast accuracy, yet accuracy alone does not determine practical value. The form of a forecast -- point, quantile, parametric, or trajectory ensemble -- fundamentally constrains which operational tasks it can support. We survey recent TSFMs and find that two-thirds produce only point or parametric forecasts, while many operational tasks require trajectory ensembles that preserve temporal dependence. We establish when forecast types can be converted and when they cannot: trajectory ensembles convert to simpler forms via marginalization without additional assumptions, but the reverse requires imposing temporal dependence through copulas or conformal methods. We prove that marginals cannot determine path-dependent event probabilities -- infinitely many joint distributions share identical marginals but yield different answers to operational questions. We map six fundamental forecasting tasks to minimal sufficient forecast types and provide a task-aligned evaluation framework. Our analysis clarifies when forecast type, not accuracy, differentiates practical utility.

---

## 65. Slot Filling as a Reasoning Task for SpeechLLMs

**论文链接:** [http://arxiv.org/abs/2510.19326v1](http://arxiv.org/abs/2510.19326v1)

**作者:** Kadri Hacioglu, Manjunath K E, Andreas Stolcke

**发布时间:** 2025-10-22

### GPT解析

### 总结

本文提出将推理能力整合到语音大语言模型中用于端到端槽填充任务，通过链式思维框架分解任务并创建推理数据集，实验表明混合语音LLM结合直接和推理模式表现最佳。

### 背景

受到最近推理大语言模型发展的启发，研究者尝试将推理能力引入语音大语言模型。

### 目的

通过链式思维框架将槽填充任务分解为多个推理步骤，创建推理数据集，并应用监督微调策略到语音大语言模型中。

### 方法

区分常规和推理语音大语言模型，实验不同类型和大小的LLM作为文本基础模型，通过引入推理步骤展示性能改进。

### 主要发现

引入推理步骤可提升性能；主要为数学、逻辑和编码领域开发的推理文本LLM作为基础模型时表现不佳；混合语音LLM结合直接和推理操作模式比单一模式微调的模型性能更好。

### 结论

混合语音LLM（结合直接和推理模式）在性能上优于仅使用一种模式的模型，是更优的选择。

### 翻译

我们提出将推理整合到语音大语言模型中用于端到端槽填充任务。受推理大语言模型最近发展的启发，我们使用链式思维框架将槽填充任务分解为多个推理步骤，创建推理数据集，并应用监督微调策略到语音大语言模型中。我们区分常规和推理语音大语言模型，并实验不同类型和大小的LLM作为它们的文本基础模型。我们通过引入推理（中间）步骤展示了性能改进。然而，我们表明主要为数学、逻辑和编码领域开发的推理文本LLM作为推理语音LLM的基础模型时可能表现不佳。我们进一步表明，构建在混合文本基础LLM上并微调以保留直接和推理操作模式的混合语音LLM，比仅使用一种操作模式微调的模型有更好的性能。


### 论文摘要

We propose integration of reasoning into speech large language models (speechLLMs) for the end-to-end slot-filling task. Inspired by the recent development of reasoning LLMs, we use a chain-of-thought framework to decompose the slot-filling task into multiple reasoning steps, create a reasoning dataset and apply the supervised fine-tuning strategy to a speechLLM. We distinguish between regular and reasoning speechLLMs and experiment with different types and sizes of LLMs as their text foundation models. We demonstrate performance improvements by introducing reasoning (intermediate) steps. However, we show that a reasoning textual LLM developed mainly for math, logic and coding domains might be inferior as a foundation model for a reasoning speechLLM. We further show that hybrid speechLLMs, built on a hybrid text foundation LLM and fine-tuned to preserve both direct and reasoning modes of operation, have better performance than those fine-tuned employing only one mode of operation.

---

## 66. Balancing Rewards in Text Summarization: Multi-Objective Reinforcement Learning via HyperVolume Optimization

**论文链接:** [http://arxiv.org/abs/2510.19325v1](http://arxiv.org/abs/2510.19325v1)

**作者:** Junjie Song, Yiwen Liu, Dapeng Li, Yin Sun, Shukun Fu, Siqi Chen, Yuji Cao

**发布时间:** 2025-10-22

### GPT解析

### 总结

本文提出了一种名为超体积优化(HVO)的新策略，用于解决大型语言模型在文本摘要任务中的多目标优化问题，通过动态调整奖励过程中的分数，使模型逐步逼近帕累托前沿，生成在多个维度上平衡的摘要。

### 背景

文本摘要需要同时优化一致性、连贯性、相关性和流畅性等多个目标，这带来了很大挑战。虽然大型语言模型通过强化学习已展现出卓越性能，但很少有研究关注基于LLMs通过RL优化摘要的多目标问题。

### 目的

开发一种新的优化策略，用于解决基于大型语言模型的文本摘要任务中的多目标优化问题，生成在多个维度上更加平衡的摘要。

### 方法

提出超体积优化(HVO)方法，在强化学习的奖励过程中使用超体积方法动态调整组之间的分数，引导模型优化逐步逼近帕累托前沿，从而在多个目标上生成平衡的摘要。

### 主要发现

在多个代表性摘要数据集上的实验表明，HVO在总体得分上优于组相对策略优化(GRPO)，且在不同维度上表现更平衡。通过HVO增强的7B基础模型在摘要任务中表现与GPT-4相当，同时保持更短的生成长度。

### 结论

HVO是一种有效的多目标优化方法，能够生成在多个维度上平衡的摘要，代码已在GitHub公开。

### 翻译

文本摘要是一项关键任务，需要同时优化一致性、连贯性、相关性和流畅性等多个目标，这带来了相当大的挑战。尽管大型语言模型已经展示了卓越的性能，并通过强化学习得到了增强，但很少有研究关注基于LLMs通过RL优化摘要的多目标问题。在本文中，我们引入了超体积优化(HVO)，一种新颖的优化策略，通过使用超体积方法在强化学习的奖励过程中动态调整组之间的分数。这种方法引导模型的优化逐步逼近帕累托前沿，从而在多个目标上生成平衡的摘要。在几个代表性摘要数据集上的实验结果表明，我们的方法在总体得分上优于组相对策略优化(GRPO)，并在不同维度上表现出更平衡的性能。此外，通过HVO增强的7B基础模型在摘要任务中表现与GPT-4相当，同时保持更短的生成长度。我们的代码已在https://github.com/ai4business-LiAuto/HVO.git公开。


### 论文摘要

Text summarization is a crucial task that requires the simultaneous optimization of multiple objectives, including consistency, coherence, relevance, and fluency, which presents considerable challenges. Although large language models (LLMs) have demonstrated remarkable performance, enhanced by reinforcement learning (RL), few studies have focused on optimizing the multi-objective problem of summarization through RL based on LLMs. In this paper, we introduce hypervolume optimization (HVO), a novel optimization strategy that dynamically adjusts the scores between groups during the reward process in RL by using the hypervolume method. This method guides the model's optimization to progressively approximate the pareto front, thereby generating balanced summaries across multiple objectives. Experimental results on several representative summarization datasets demonstrate that our method outperforms group relative policy optimization (GRPO) in overall scores and shows more balanced performance across different dimensions. Moreover, a 7B foundation model enhanced by HVO performs comparably to GPT-4 in the summarization task, while maintaining a shorter generation length. Our code is publicly available at https://github.com/ai4business-LiAuto/HVO.git

---

## 67. Advances in 4D Representation: Geometry, Motion, and Interaction

**论文链接:** [http://arxiv.org/abs/2510.19255v1](http://arxiv.org/abs/2510.19255v1)

**作者:** Mingrui Zhao, Sauradip Nag, Kai Wang, Aditya Vora, Guangda Ji, Peter Chun, Ali Mahdavi-Amiri, Hao Zhang

**发布时间:** 2025-10-22

**备注:** 21 pages. Project Page: https://mingrui-zhao.github.io/4DRep-GMI/

### GPT解析

### 总结

这是一篇关于4D生成和重建的调查论文，从4D表示的独特视角出发，帮助读者了解如何选择和定制适合自己任务的4D表示方法。

### 背景

4D生成和重建是计算机图形学中一个快速发展的子领域，其发展受到神经场、几何和深度学习以及3D生成人工智能(GenAI)最近进展的推动。

### 目的

帮助读者了解如何选择和定制适合自己任务的4D表示方法，以建模随时间演变的3D几何并展示运动和交互。

### 方法

采用选择性方法，重点关注代表性工作，以突出不同计算、应用和数据场景下每种表示的理想特性和随之而来的挑战。

### 主要发现

将4D表示基于几何、运动和交互三个关键支柱进行分类；涵盖当前流行的表示方法如NeRFs和3DGS，以及相对未被充分探索的表示；讨论大型语言模型和视频基础模型在4D应用中的作用及其局限性；分析当前可用的4D数据集及推动领域发展所需的更多数据集。

### 结论

选择和定制适当的4D表示对于完成特定任务至关重要。

### 翻译

我们提出了关于4D生成和重建的调查，这是计算机图形学中一个快速发展的子领域，其发展受到神经场、几何和深度学习以及3D生成人工智能(GenAI)最近进展的推动。虽然我们的调查不是首创，但我们从4D表示的独特视角构建了该领域的覆盖范围，用于建模随时间演变的3D几何并展示运动和交互。具体而言，我们没有提供大量工作的详尽列举，而是采用更选择性的方法，重点关注代表性工作，以突出不同计算、应用和数据场景下每种表示的理想特性和随之而来的挑战。我们旨在传达给读者的主要信息是如何为他们的任务选择和定制适当的4D表示。在组织上，我们基于三个关键支柱来区分4D表示：几何、运动和交互。我们的讨论不仅将涵盖当今最受欢迎的表示，如神经辐射场(NeRFs)和3D高斯溅射(3DGS)，还将引起对4D背景下相对未被充分探索的表示的关注，如结构化模型和长程运动。在整个调查中，我们将回顾大型语言模型(LLMs)和视频基础模型(VFMs)在多种4D应用中的作用，同时引导讨论它们当前的局限性以及如何解决这些局限性。我们还专门介绍了当前可用的4D数据集，以及推动该子领域发展所缺乏的数据集。项目页面：https://mingrui-zhao.github.io/4DRep-GMI/

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决4D表示（随时间变化的3D几何形状）的系统分类和分析问题。这个问题在现实中非常重要，因为随着计算机图形学应用扩展到电影视觉效果、虚拟现实、自主机器人、医学成像和电子商务等领域，能够捕获、表示和操作4D内容已成为连接图形学、视觉和机器学习的基本挑战。4D表示技术能够帮助我们理解和建模动态世界，为各种应用提供基础支撑。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者采用以表示为中心的独特视角，而非像之前综述那样按应用或方法分类。他们借鉴了Cao等人、Fan等人、Miao等人的工作，但认为这些综述未能充分涵盖所有相关表示方法，特别是结构化表示、运动和交互方面。作者通过三个关键支柱（几何、运动和交互）构建分析框架，并在几何部分进一步区分结构化和非结构化表示，从而提供了一个更全面、更有条理的分析视角。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '作为一篇综述论文，其核心思想是从表示角度系统分类和分析4D表示方法，帮助研究人员理解不同表示的特性、优势和局限性。整体流程分为六个部分：1）引言介绍背景和问题；2）几何建模分析非结构化表示（网格、点云、NeRF、3D高斯飞溅）和结构化表示（模板、部件、图）；3）运动建模分析不同运动类型与表示的交互；4）交互建模讨论多实体交互表示；5）数据集、评估指标和基准测试；6）整体分析和未来方向。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1）采用表示中心视角而非应用或方法分类；2）提出几何、运动、交互三支柱框架；3）明确区分结构化与非结构化表示；4）全面分析不同运动类型与表示选择的相互作用；5）专门讨论交互表示问题；6）探讨大型语言模型和视频基础模型在4D应用中的作用。相比之前工作，这篇论文提供了更全面的表示分类，更深入分析表示方法的优缺点和适用场景，并提供了如何为特定任务选择表示的实用指导。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '这篇论文通过从几何、运动和交互三个关键支柱对4D表示方法进行系统分类和分析，为研究人员提供了如何为特定4D任务选择和定制适当表示的全面指导框架。'}


### 论文摘要

We present a survey on 4D generation and reconstruction, a fast-evolving subfield of computer graphics whose developments have been propelled by recent advances in neural fields, geometric and motion deep learning, as well 3D generative artificial intelligence (GenAI). While our survey is not the first of its kind, we build our coverage of the domain from a unique and distinctive perspective of 4D representations\/}, to model 3D geometry evolving over time while exhibiting motion and interaction. Specifically, instead of offering an exhaustive enumeration of many works, we take a more selective approach by focusing on representative works to highlight both the desirable properties and ensuing challenges of each representation under different computation, application, and data scenarios. The main take-away message we aim to convey to the readers is on how to select and then customize the appropriate 4D representations for their tasks. Organizationally, we separate the 4D representations based on three key pillars: geometry, motion, and interaction. Our discourse will not only encompass the most popular representations of today, such as neural radiance fields (NeRFs) and 3D Gaussian Splatting (3DGS), but also bring attention to relatively under-explored representations in the 4D context, such as structured models and long-range motions. Throughout our survey, we will reprise the role of large language models (LLMs) and video foundational models (VFMs) in a variety of 4D applications, while steering our discussion towards their current limitations and how they can be addressed. We also provide a dedicated coverage on what 4D datasets are currently available, as well as what is lacking, in driving the subfield forward. Project page:https://mingrui-zhao.github.io/4DRep-GMI/

---

## 68. TinyUSFM: Towards Compact and Efficient Ultrasound Foundation Models

**论文链接:** [http://arxiv.org/abs/2510.19239v1](http://arxiv.org/abs/2510.19239v1)

**作者:** Chen Ma, Jing Jiao, Shuyu Liang, Junhu Fu, Qin Wang, Zeju Li, Yuanyuan Wang, Yi Guo

**发布时间:** 2025-10-22

**备注:** Submit to JBHI, 14 pages, 6 figures

### GPT解析

### 总结

TinyUSLM是一种通过知识蒸馏技术开发的轻量级超声基础模型，能够在保持优异性能的同时显著减少计算资源需求，使其适用于资源有限的临床环境。

### 背景

医学成像的基础模型在多样化的解剖结构和临床应用中表现出优越的泛化能力，但其出色的性能依赖于大量计算资源，限制了在资源有限的临床环境中的部署。

### 目的

开发一种轻量级超声基础模型，能够在保持大规模超声基础模型(USFM)的优异器官多样性和任务适应性的同时，实现显著的计算效率。

### 方法

提出特征梯度驱动的核心集选择策略筛选高质量训练数据；开发域分离的掩码图像建模辅助一致性驱动的动态蒸馏保留空间和频域特性；建立包含8个分类和10个分割数据集的UniUS-Bench超声基准。

### 主要发现

TinyUSLM仅使用20万张图像进行蒸馏，就能以仅6.36%的参数和6.40%的GFLOPs达到与USFM相当的性能；在分类和分割任务上分别比普通模型高出9.45%和7.72%；超越了所有最先进的轻量级模型；实现了84.91%的平均分类准确率和85.78%的平均分割Dice分数。

### 结论

TinyUSLM成功实现了轻量级超声基础模型的开发，在保持优异性能的同时显著降低了计算资源需求，使其适用于资源有限的临床环境，为医学成像领域提供了实用的解决方案。

### 翻译

医学成像基础模型在多样化的解剖结构和临床应用中表现出优越的泛化能力。它们的出色性能依赖于大量计算资源，限制了在资源有限的临床环境中的部署。本文提出了TinyUSLM，这是第一个轻量级超声基础模型，通过使用精心筛选的小型数据集进行知识蒸馏，保持了我们的大规模超声基础模型(USFM)的卓越器官多样性和任务适应性，在不牺牲性能的情况下提供了显著的计算效率。考虑到轻量级模型的有限容量和表示能力，我们提出了一个特征梯度驱动的核心集选择策略，用于筛选高质量的紧凑训练数据，避免因低质量冗余图像导致的训练退化。为了在知识转移过程中保留基本的空间和频域特性，我们开发了域分离的掩码图像建模辅助一致性驱动的动态蒸馏。这个新颖的框架通过利用教师模型在不同域掩码上的一致性，自适应地从大型基础模型转移知识，专门针对超声解释进行定制。为了评估，我们建立了UniUS-Bench，这是最大的公开可用超声基准，包含跨15个器官的8个分类和10个分割数据集。仅使用20万张图像进行蒸馏，TinyUSLM就能以仅6.36%的参数和6.40%的GFLOPs达到USLM的性能。TinyUSLM在分类和分割任务上分别比普通模型高出9.45%和7.72%，超越了所有最先进的轻量级模型，并在各种医疗设备和中心实现了84.91%的平均分类准确率和85.78%的平均分割Dice分数。


### 论文摘要

Foundation models for medical imaging demonstrate superior generalization capabilities across diverse anatomical structures and clinical applications. Their outstanding performance relies on substantial computational resources, limiting deployment in resource-constrained clinical environments. This paper presents TinyUSFM, the first lightweight ultrasound foundation model that maintains superior organ versatility and task adaptability of our large-scale Ultrasound Foundation Model (USFM) through knowledge distillation with strategically curated small datasets, delivering significant computational efficiency without sacrificing performance. Considering the limited capacity and representation ability of lightweight models, we propose a feature-gradient driven coreset selection strategy to curate high-quality compact training data, avoiding training degradation from low-quality redundant images. To preserve the essential spatial and frequency domain characteristics during knowledge transfer, we develop domain-separated masked image modeling assisted consistency-driven dynamic distillation. This novel framework adaptively transfers knowledge from large foundation models by leveraging teacher model consistency across different domain masks, specifically tailored for ultrasound interpretation. For evaluation, we establish the UniUS-Bench, the largest publicly available ultrasound benchmark comprising 8 classification and 10 segmentation datasets across 15 organs. Using only 200K images in distillation, TinyUSFM matches USFM's performance with just 6.36% of parameters and 6.40% of GFLOPs. TinyUSFM significantly outperforms the vanilla model by 9.45% in classification and 7.72% in segmentation, surpassing all state-of-the-art lightweight models, and achieving 84.91% average classification accuracy and 85.78% average segmentation Dice score across diverse medical devices and centers.

---

## 69. Understanding the Implicit Biases of Design Choices for Time Series Foundation Models

**论文链接:** [http://arxiv.org/abs/2510.19236v1](http://arxiv.org/abs/2510.19236v1)

**作者:** Annan Yu, Danielle C. Maddix, Boran Han, Xiyuan Zhang, Abdul Fatir Ansari, Oleksandr Shchur, Christos Faloutsos, Andrew Gordon Wilson, Michael W. Mahoney, Yuyang Wang

**发布时间:** 2025-10-22

### GPT解析

### 总结

该研究通过理论和实证方法分析了时间序列基础模型(TSFMs)中设计选择的影响，揭示了不同设计如何导致模型中的隐式偏置，以及这些偏置如何影响模型行为，研究结果对于理解和改进未来TSFMs的发展具有重要意义。

### 背景

时间序列基础模型(TSFMs)是一类强大的通用工具，用于时间序列预测和相关时间任务，但这些模型的行为受到其设计中微妙归纳偏置的强烈影响。

### 目的

理解训练过程中的各种'旋钮'如何影响模型质量，而非开发一个声称比现有TSFMs更好的新模型；探讨设计选择如何导致模型基本属性中的隐式偏置。

### 方法

使用理论和受控经验评估相结合的方法；识别几种设计选择(如patch大小、嵌入选择、训练目标等)；研究这些设计选择如何影响模型的基本属性。

### 主要发现

不同的设计选择会导致模型基本属性中的隐式偏置；这些偏置可能是直观的或非常违反直觉的，取决于模型和数据的特性；在异常值处理的案例研究中，展示了多种偏置如何以复杂方式相互作用；讨论了研究结果对学习'苦涩教训'和构建TSFMs的启示。

### 结论

理解设计选择对模型行为的影响对于构建有效的TSFMs至关重要；模型中的隐式偏置可以是有益的，但也可能导致意想不到的行为。

### 翻译

时间序列基础模型(TSFMs)是一类潜在的强大通用工具，用于时间序列预测和相关时间任务，但它们的行为受到其设计中微妙归纳偏置的强烈影响。我们不是开发一个新模型并声称它比现有的TSFMs更好，例如通过在现有成熟的基准测试中获胜，我们的目标是理解训练过程中的各种'旋钮'如何影响模型质量。结合理论和受控经验评估，我们确定了几个设计选择(补丁大小、嵌入选择、训练目标等)，并展示了它们如何导致模型基本属性中的隐式偏置(时间行为、几何结构、模型回归到均值的激进程度等)；我们展示了这些偏置如何可能是直观的或非常违反直觉的，这取决于模型和数据的特性。我们还在异常值处理的案例研究中说明了多种偏置如何以复杂方式相互作用；我们讨论了我们的结果对学习苦涩教训和构建TSFMs的启示。


### 论文摘要

Time series foundation models (TSFMs) are a class of potentially powerful, general-purpose tools for time series forecasting and related temporal tasks, but their behavior is strongly shaped by subtle inductive biases in their design. Rather than developing a new model and claiming that it is better than existing TSFMs, e.g., by winning on existing well-established benchmarks, our objective is to understand how the various ``knobs'' of the training process affect model quality. Using a mix of theory and controlled empirical evaluation, we identify several design choices (patch size, embedding choice, training objective, etc.) and show how they lead to implicit biases in fundamental model properties (temporal behavior, geometric structure, how aggressively or not the model regresses to the mean, etc.); and we show how these biases can be intuitive or very counterintuitive, depending on properties of the model and data. We also illustrate in a case study on outlier handling how multiple biases can interact in complex ways; and we discuss implications of our results for learning the bitter lesson and building TSFMs.

---

## 70. PoSh: Using Scene Graphs To Guide LLMs-as-a-Judge For Detailed Image Descriptions

**论文链接:** [http://arxiv.org/abs/2510.19060v1](http://arxiv.org/abs/2510.19060v1)

**作者:** Amith Ananthram, Elias Stengel-Eskin, Lorena A. Bradford, Julia Demarest, Adam Purvis, Keith Krut, Robert Stein, Rina Elster Pantalony, Mohit Bansal, Kathleen McKeown

**发布时间:** 2025-10-21

**备注:** 24 pages, 9 figures. Metric/benchmark available at  https://github.com/amith-ananthram/posh

### GPT解析

### 总结

本文提出了一种名为PoSh的新型评估指标，用于评估视觉语言模型生成的详细图像描述，并引入了DOCENT数据集作为基准测试。PoSh使用场景图作为结构化评分指南，能够更好地模拟人类评分行为，并且在多个方面优于现有评估方法。

### 背景

视觉语言模型(VLMs)已发展到能够生成详细的图像描述，但评估这些描述仍然面临挑战。现有标准评估指标(如CIDEr、SPICE)是为短文本设计的，主要针对现在已不常见的错误类型(如对象识别错误)进行调整，无法有效评估长文本中属性和关系的连接性。

### 目的

开发一种能够评估详细图像描述的新指标，特别关注长文本中属性和关系的连接性，并能将错误定位到特定文本跨度。同时创建一个具有挑战性的新数据集来验证该指标的有效性。

### 方法

提出PoSh评估指标，利用场景图作为结构化评分指南指导大语言模型作为评判者，产生基于细粒度错误的聚合分数。同时创建DOCENT数据集，包含艺术品、专家参考描述和模型生成描述，配有艺术史学生的质量评估。通过PoSh评估开放和封闭模型在描述绘画、素描和雕像方面的性能。

### 主要发现

PoSh在DOCENT上与人类判断的相关性比最佳开源替代方案更强，对图像类型具有鲁棒性，且作为奖励函数优于标准监督微调。研究发现基础模型难以实现对具有丰富场景动态的图像的完整、无错误覆盖，确立了评估VLM进展的新任务标准。

### 结论

PoSh和DOCENT为评估详细图像描述提供了新工具，有望促进辅助文本生成等重要领域的进步，为视觉语言模型的发展提供更准确的评估方法。

### 翻译

虽然视觉语言模型已经发展到能够进行详细的图像描述，但评估仍然是一个挑战。标准指标是为短文本设计的，并且调整为识别现在不常见的错误，如对象识别错误。相比之下，长文本需要对属性和关系连接的敏感性，以及将错误定位到特定文本跨度的评分。在这项工作中，我们介绍了PoSh，一种用于详细图像描述的指标，它使用场景图作为结构化评分指南来指导大语言模型作为评判者，产生基于细粒度错误的聚合分数。PoSh是可复制的、可解释的，并且比现有指标更好地模拟人类评分者的行为。为了验证PoSh，我们引入了一个具有挑战性的新数据集DOCENT。这个新的基准包含艺术品，配以专家撰写的参考文本和模型生成的描述，并附有艺术史学生对它们质量的细致和粗略判断。因此，DOCENT能够在一个具有挑战性的新领域评估详细的图像描述指标和详细的图像描述本身。我们表明，PoSh在DOCENT上与人类判断的相关性比最佳开源替代方案更强，对图像类型具有鲁棒性，并且是一个有效的奖励函数，优于标准的监督微调。然后，使用PoSh，我们描述了开放和封闭模型在描述绘画、素描和雕像方面的性能，发现基础模型难以实现对具有丰富场景动态的图像的完整、无错误的覆盖，从而确立了一个具有挑战性的新任务来衡量VLM的进展。通过PoSh和DOCENT，我们希望能够在辅助文本生成等重要领域取得进展。


### 论文摘要

While vision-language models (VLMs) have advanced into detailed image description, evaluation remains a challenge. Standard metrics (e.g. CIDEr, SPICE) were designed for short texts and tuned to recognize errors that are now uncommon, such as object misidentification. In contrast, long texts require sensitivity to attribute and relation attachments and scores that localize errors to particular text spans. In this work, we introduce PoSh, a metric for detailed image description that uses scene graphs as structured rubrics to guide LLMs-as-a-Judge, producing aggregate scores grounded in fine-grained errors (e.g. mistakes in compositional understanding). PoSh is replicable, interpretable and a better proxy for human raters than existing metrics (including GPT4o-as-a-Judge). To validate PoSh, we introduce a challenging new dataset, DOCENT. This novel benchmark contains artwork, paired with expert-written references, and model-generated descriptions, augmented with granular and coarse judgments of their quality from art history students. Thus, DOCENT enables evaluating both detailed image description metrics and detailed image description itself in a challenging new domain. We show that PoSh achieves stronger correlations (+0.05 Spearman $\rho$) with the human judgments in DOCENT than the best open-weight alternatives, is robust to image type (using CapArena, an existing dataset of web imagery) and is a capable reward function, outperforming standard supervised fine-tuning. Then, using PoSh, we characterize the performance of open and closed models in describing the paintings, sketches and statues in DOCENT and find that foundation models struggle to achieve full, error-free coverage of images with rich scene dynamics, establishing a demanding new task to gauge VLM progress. Through both PoSh and DOCENT, we hope to enable advances in important areas such as assistive text generation.

---

## 71. QKCV Attention: Enhancing Time Series Forecasting with Static Categorical Embeddings for Both Lightweight and Pre-trained Foundation Models

**论文链接:** [http://arxiv.org/abs/2510.20222v1](http://arxiv.org/abs/2510.20222v1)

**作者:** Hao Wang, Baojun Ma

**发布时间:** 2025-10-21

**备注:** 10 pages, 5 figures

### GPT解析

### 总结

本文提出了一种名为QKCV的注意力机制，通过融入静态类别嵌入来增强传统QKV框架，提高时间序列预测准确性。

### 背景

在现实世界的时间序列预测任务中，类别信息在捕捉固有数据模式方面起着关键作用。

### 目的

开发一种能够有效利用类别信息提高预测准确性的注意力机制。

### 方法

引入QKCV（Query-Key-Category-Value）注意力机制，作为传统QKV框架的扩展，融入静态类别嵌入C来强调特定类别的信息。

### 主要发现

QKCV作为即插即用模块能增强多种基于注意力的模型在现实数据集上的预测准确性；在微调单变量时间序列基础模型时，只需更新静态嵌入C，保留预训练权重，减少计算开销并提高微调性能。

### 结论

QKCV注意力机制能有效利用类别信息提高时间序列预测的准确性，具有良好的适应性和计算效率。

### 翻译

在现实世界的时间序列预测任务中，类别信息在捕捉固有数据模式方面起着关键作用。本文引入了QKCV（查询-键-类别-值）注意力，这是传统QKV框架的扩展，融入了静态类别嵌入C来强调特定类别的信息。作为一个通用的即插即用模块，QKCV增强了基于注意力的模型（如普通Transformer、Informer、PatchTST、TFT）在各种现实世界数据集上的预测准确性。此外，QKCV在通过仅更新静态嵌入C同时保留预训练权重来微调单变量时间序列基础模型时表现出显著的适应性，从而减少了计算开销并实现了更好的微调性能。


### 论文摘要

In real-world time series forecasting tasks, category information plays a pivotal role in capturing inherent data patterns. This paper introduces QKCV (Query-Key-Category-Value) attention, an extension of the traditional QKV framework that incorporates a static categorical embedding C to emphasize category-specific information. As a versatile plug-in module, QKCV enhances the forecasting accuracy of attention-based models (e.g., Vanilla Transformer, Informer, PatchTST, TFT) across diverse real-world datasets. Furthermore, QKCV demonstrates remarkable adaptability in fine-tuning univariate time series foundation model by solely updating the static embedding C while preserving pretrained weights, thereby reducing computational overhead and achieving superior fine-tuning performance.

---

