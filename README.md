# 今日论文推荐 - 2025-11-27

共 157 篇论文

---

## 1. TraceGen: World Modeling in 3D Trace Space Enables Learning from Cross-Embodiment Videos

**论文链接:** [http://arxiv.org/abs/2511.21690v1](http://arxiv.org/abs/2511.21690v1)

**作者:** Seungjae Lee, Yoonkyo Jung, Inkook Chun, Yao-Chih Lee, Zikui Cai, Hongjia Huang, Aayush Talreja, Tan Dat Dao, Yongyuan Liang, Jia-Bin Huang, Furong Huang

**发布时间:** 2025-11-26

### GPT解析

### 总结

本研究提出了一种通过统一的3D轨迹空间表示，使机器人能够从少量演示中学习新任务，并实现跨不同实体形态、环境和任务的方法。

### 背景

在新的机器人平台和场景中仅通过少量演示学习新任务具有挑战性。虽然人类和其他机器人的视频资源丰富，但不同实体形态、相机视角和环境差异限制了这些资源的直接利用。

### 目的

开发一种能够从跨实体形态、跨环境和跨任务视频中学习的方法，解决机器人学习中的小样本问题，使机器人能够快速适应新任务和新环境。

### 方法

作者提出了TraceGen，一个在3D轨迹空间而非像素空间中预测未来运动的世界模型，抽象了外观信息同时保留了操作所需的几何结构。同时开发了TraceForge数据管道，将异构的人类和机器人视频转换为一致的3D轨迹，构建了一个包含123K视频和180万观察-轨迹-语言三元组的训练语料库。

### 主要发现

1. 预训练的TraceGen模型仅使用五个目标机器人视频，在四项任务中达到80%的成功率；2. 推理速度比最先进的基于视频的世界模型快50-600倍；3. 仅使用五个手持手机拍摄的非标定人类演示视频，在真实机器人上仍达到67.5%的成功率；4. 模型能够跨实体形态进行适应，无需依赖目标检测器或重像素空间生成。

### 结论

通过引入统一的3D轨迹空间表示和相应的世界模型，研究成功解决了机器人学习中的小样本问题，实现了高效的跨实体形态、跨环境和跨任务学习，为机器人技能的快速迁移提供了有效途径。

### 翻译

从仅少量演示中在新平台和新场景学习机器人任务仍然具有挑战性。虽然人类和其他实体形态的机器人视频资源丰富，但实体形态、相机和环境的差异阻碍了它们的直接使用。我们通过引入统一的符号表示——场景级轨迹的紧凑3D'轨迹空间'来解决小样本问题，该表示使能够从跨实体形态、跨环境和跨任务视频中学习。我们提出了TraceGen，一个在轨迹空间而非像素空间中预测未来运动的世界模型，它抽象了外观同时保留了操作所需的几何结构。为了大规模训练TraceGen，我们开发了TraceForge，一个将异构人类和机器人视频转换为一致3D轨迹的数据管道，产生了包含123K视频和180万观察-轨迹-语言三元组的语料库。在这个语料库上预训练产生了可迁移的3D运动先验，能够高效适应：仅使用五个目标机器人视频，TraceGen在四项任务中达到80%的成功率，比最先进的基于视频的世界模型快50-600倍的推理速度。在更具挑战性的情况下，当只有五个手持手机拍摄的非标定人类演示视频可用时，它在真实机器人上仍达到67.5%的成功率，突显了TraceGen跨实体形态适应的能力，无需依赖目标检测器或重像素空间生成。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决机器人如何从少量演示中学习新任务的问题，特别是在新平台和新场景中学习。这个问题很重要，因为收集特定任务的机器人演示数据既缓慢又昂贵，而人类视频数据丰富却因形态、相机和场景差异难以直接重用。现有方法要么在像素空间计算成本高，要么在语言空间缺乏空间精度，限制了机器人学习效率。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先分析了现有方法的局限性，认识到不同形态的物体和机器人末端执行器运动具有共享的3D结构。基于这一洞察，他们设计了轨迹空间表示，丢弃外观信息保留几何结构。该方法借鉴了现有3D估计技术(TAPIP3D、CoTracker3)、CogVideoX的架构、Prismatic VLM的多编码器融合策略以及Stochastic Interpolant框架，但将其创新性地应用于3D轨迹空间预测。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是使用3D轨迹空间作为统一表示，捕获场景中物体和机器人的运动，同时对相机和环境变化具有不变性。实现流程包括：1)数据准备(TraceForge)：从视频中提取任务片段，生成指令，跟踪3D点，转换坐标系，速度重定向；2)模型训练(TraceGen)：多编码器提取视觉和文本特征，融合后通过基于流的解码器预测3D轨迹；3)应用：给定单张RGB-D图像和指令，预测轨迹并转换为机器人命令。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)TraceGen世界模型在3D轨迹空间而非像素空间操作；2)TraceForge数据管道将异构视频转换为一致3D轨迹；3)在123K视频和180万三元组上训练单一形态不可知策略；4)高效少样本适应(5个视频达80%成功率)。相比之前工作，它避免了像素空间的高计算成本和语言空间的空间精度不足，扩展到野外视频而非仅限实验室，预测完整场景级轨迹无需辅助检测。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'TraceGen通过在3D轨迹空间中进行世界建模，实现了从跨形态视频中高效学习，仅需少量演示即可使机器人在新环境中执行新任务，同时比现有方法快50倍以上。'}


### 论文摘要

Learning new robot tasks on new platforms and in new scenes from only a handful of demonstrations remains challenging. While videos of other embodiments - humans and different robots - are abundant, differences in embodiment, camera, and environment hinder their direct use. We address the small-data problem by introducing a unifying, symbolic representation - a compact 3D "trace-space" of scene-level trajectories - that enables learning from cross-embodiment, cross-environment, and cross-task videos. We present TraceGen, a world model that predicts future motion in trace-space rather than pixel space, abstracting away appearance while retaining the geometric structure needed for manipulation. To train TraceGen at scale, we develop TraceForge, a data pipeline that transforms heterogeneous human and robot videos into consistent 3D traces, yielding a corpus of 123K videos and 1.8M observation-trace-language triplets. Pretraining on this corpus produces a transferable 3D motion prior that adapts efficiently: with just five target robot videos, TraceGen attains 80% success across four tasks while offering 50-600x faster inference than state-of-the-art video-based world models. In the more challenging case where only five uncalibrated human demonstration videos captured on a handheld phone are available, it still reaches 67.5% success on a real robot, highlighting TraceGen's ability to adapt across embodiments without relying on object detectors or heavy pixel-space generation.

---

## 2. Continual Error Correction on Low-Resource Devices

**论文链接:** [http://arxiv.org/abs/2511.21652v1](http://arxiv.org/abs/2511.21652v1)

**作者:** Kirill Paramonov, Mete Ozay, Aristeidis Mystakidis, Nikolaos Tsalikidis, Dimitrios Sotos, Anastasios Drosou, Dimitrios Tzovaras, Hyunjun Kim, Kiseok Chang, Sangdok Mo, Namwoong Kim, Woojong Yoo, Jijoong Moon, Umberto Michieli

**发布时间:** 2025-11-26

**备注:** ACM MMSys 2025

### GPT解析

### 总结

该研究提出了一种结合服务器端和设备端处理的新型AI错误纠正系统，使用少样本学习和原型适应，能够在资源受限设备上高效纠正AI预测错误。

### 背景

AI模型在日常设备中的广泛应用引发了一个关键挑战：预测错误会降低用户体验。现有解决方案主要集中在错误检测上，很少提供高效的纠错机制，特别是对于资源受限的设备。

### 目的

提出一个新颖的系统，使用少样本学习使用户能够纠正AI的错误分类，同时需要最少的计算资源和存储空间。

### 方法

结合服务器端基础模型训练和设备端基于原型的分类，通过原型更新而非模型重新训练实现高效错误纠正。系统包含两个关键组件：服务器端管道利用知识蒸馏将基础模型的鲁棒特征表示转移到设备兼容架构；设备端机制通过原型适应实现超高效的错误纠正。

### 主要发现

在图像分类和目标检测任务上证明了系统有效性，在Food-101和Flowers-102数据集的一个样本场景中实现了超过50%的错误纠正率，同时保持最小遗忘（小于0.02%）和可忽略的计算开销。通过Android演示应用验证了系统在实际场景中的实用性。

### 结论

该系统为资源受限设备上的AI错误纠错提供了一个实用且高效的解决方案。

### 翻译

AI模型在日常设备中的激增凸显了一个关键挑战：预测错误会降低用户体验。虽然现有解决方案专注于错误检测，但它们很少提供高效的纠错机制，特别是对于资源受限的设备。我们提出了一种新颖的系统，使用户能够通过少样本学习纠正AI的错误分类，只需要最少的计算资源和存储空间。我们的方法结合了服务器端基础模型训练和设备端基于原型的分类，通过原型更新而非模型重新训练来实现高效的错误纠正。该系统包含两个关键组件：(1) 服务器端管道利用知识蒸馏将基础模型的鲁棒特征表示转移到设备兼容架构；(2) 设备端机制通过原型适应实现超高效的错误纠正。我们在图像分类和目标检测任务上证明了系统的有效性，在Food-101和Flowers-102数据集的一个样本场景中实现了超过50%的错误纠正率，同时保持最小遗忘（小于0.02%）和可忽略的计算开销。我们的实现通过Android演示应用得到验证，证明了系统在实际场景中的实用性。


### 论文摘要

The proliferation of AI models in everyday devices has highlighted a critical challenge: prediction errors that degrade user experience. While existing solutions focus on error detection, they rarely provide efficient correction mechanisms, especially for resource-constrained devices. We present a novel system enabling users to correct AI misclassifications through few-shot learning, requiring minimal computational resources and storage. Our approach combines server-side foundation model training with on-device prototype-based classification, enabling efficient error correction through prototype updates rather than model retraining. The system consists of two key components: (1) a server-side pipeline that leverages knowledge distillation to transfer robust feature representations from foundation models to device-compatible architectures, and (2) a device-side mechanism that enables ultra-efficient error correction through prototype adaptation. We demonstrate our system's effectiveness on both image classification and object detection tasks, achieving over 50% error correction in one-shot scenarios on Food-101 and Flowers-102 datasets while maintaining minimal forgetting (less than 0.02%) and negligible computational overhead. Our implementation, validated through an Android demonstration app, proves the system's practicality in real-world scenarios.

---

## 3. An AI-Enabled Hybrid Cyber-Physical Framework for Adaptive Control in Smart Grids

**论文链接:** [http://arxiv.org/abs/2511.21590v1](http://arxiv.org/abs/2511.21590v1)

**作者:** Muhammad Siddique, Sohaib Zafar

**发布时间:** 2025-11-26

**备注:** 16 pages, 11 figures, IEEEaccess journal

### GPT解析

### 总结

本文提出了一种部署在云平台上的基于机器学习的智能电网数字取证框架，该框架结合了传感器级数据采集、认证通信、可扩展云存储和自动取证分析功能，能够有效检测和缓解智能电网中的安全威胁。

### 背景

智能电网是传统电力基础设施与先进通信网络和智能控制的融合，创造了前所未有的高效灵活的物理信息环境，但这种集成也带来了可能破坏电网稳定性和可靠性的漏洞。

### 目的

开发一种基于机器学习的智能电网系统数字取证框架，用于学习、识别、检测和缓解智能电网中的安全事件，提高电网的安全性和可靠性。

### 方法

提出一个一体化的数字取证框架，结合传感器级数据采集、认证通信、可扩展云存储和自动取证分析，使用随机森林、支持向量机、梯度提升树和深度神经网络等算法进行实时异常检测、事件重建和入侵分析。

### 主要发现

通过在实时智能电表数据流上的模拟和实验研究表明，所提出的框架具有很高的准确性、可扩展性和弹性，能够有效抵御数据篡改、虚假数据注入和协调控制回路操纵等网络攻击。

### 结论

云服务是大数据驱动取证工作流程的最佳骨干，使能源公用事业公司能够实现快速态势感知和智能事件响应，从而提高智能电网的安全性和可靠性。

### 翻译

智能电网是传统电力基础设施与先进通信网络和智能控制的融合，创造了前所未有的高效灵活的物理信息环境。这种集成会导致可能破坏电网稳定性和可靠性的漏洞。数字取证是学习、识别、检测和缓解此类安全事件的基本概念。本文提出了一种部署在云平台上的基于机器学习的智能电网系统一体化数字取证框架。该框架结合了传感器级数据采集、认证通信、可扩展云存储和自动取证分析。该模型使用监督和无监督学习算法，如随机森林、支持向量机、梯度提升树和深度神经网络架构，用于实时异常检测、事件重建和入侵分析。在实时智能电表数据流上进行多次模拟和实验研究后，结果表明所提出的框架具有很高的准确性、可扩展性和抵御网络攻击的能力，包括数据篡改、虚假数据注入和协调控制回路操纵。结果表明，云服务是大数据驱动取证工作流程的最佳骨干，使能源公用事业公司能够实现快速态势感知和智能事件响应。


### 论文摘要

Smart grids are a fusion of classical power infrastructure and advanced communication networks and smart control, to create a cyber-physical environment that is more efficient and flexible than ever before. This integration causes vulnerabilities that can undermine grid stability as well as reliability. Digital forensics is a fundamental concept of learning and identifying, detecting, and mitigating such security incidents. This paper presents an all-in-one machine learning-based digital forensic framework of smart grid systems deployed on the Cloud. The framework combines the data acquisition at the sensor-level, authenticated communication, scalable cloud storage and automated forensic analytics. The model uses supervised and unsupervised learning algorithms - such as Random Forest, Support Vector Machine, Gradient Boosted Trees and deep neural architectures for anomaly detection, event reconstruction and intrusion analysis in real time. After several simulation and experimental studies on real-time smart-meter data streams, the proposed framework is shown to be very accurate, scalable and resilient to cyber-attacks including data tampering, false-data injection and coordinated control-loop manipulation. The results indicate that cloud services are the best backbone for big-data-driven forensic workflows, which allows energy utilities to achieve a fast situational awareness and intelligent incident response.

---

## 4. Multimodal Robust Prompt Distillation for 3D Point Cloud Models

**论文链接:** [http://arxiv.org/abs/2511.21574v1](http://arxiv.org/abs/2511.21574v1)

**作者:** Xiang Gu, Liming Lu, Xu Zheng, Anan Du, Yongbin Zhou, Shuchao Pang

**发布时间:** 2025-11-26

### GPT解析

### 总结

本文提出了一种名为多模态鲁棒提示蒸馏(MRPD)的学生-教师框架，用于防御3D点云模型面临的对抗攻击，解决了现有防御方法计算开销大和泛化能力差的问题。

### 背景

对抗攻击对基于学习的3D点云模型构成重大威胁，严重削弱了它们在安全敏感应用中的可靠性。现有的防御方法通常存在高计算开销和对不同攻击类型泛化能力差的问题。

### 目的

为了解决现有防御方法的局限性，作者提出了一种新颖而高效的学生-教师框架，用于提炼鲁棒的3D点云模型。

### 方法

MRPD通过将学生点云模型的特征与三个不同教师的鲁棒嵌入对齐来学习轻量级提示，这三个教师分别是处理深度投影的视觉模型、高性能3D模型和文本编码器。蒸馏过程由置信门控机制引导，动态平衡各模态贡献。由于蒸馏完全在训练阶段进行，推理时没有额外计算成本。

### 主要发现

大量实验表明，MRPD在对抗广泛的白盒和黑盒攻击方面显著优于最先进的防御方法，甚至在干净数据上也能实现更好的性能。

### 结论

通过有效利用多模态知识，作者的工作为构建鲁棒的3D视觉系统提供了一种新的、实用的范式。

### 翻译

对抗攻击对基于学习的3D点云模型构成重大威胁，严重削弱了它们在安全敏感应用中的可靠性。现有的防御方法通常存在(1)高计算开销和(2)对不同攻击类型的泛化能力差。为了弥补这些差距，我们提出了一种新颖而高效的学生-教师框架，即多模态鲁棒提示蒸馏(MRPD)，用于提炼鲁棒的3D点云模型。它通过将学生点云模型的特征与来自三个不同教师的鲁棒嵌入对齐来学习轻量级提示：处理深度投影的视觉模型、高性能3D模型和文本编码器。为确保可靠的知识转移，这种蒸馏由置信门控机制引导，该机制动态平衡所有输入模态的贡献。值得注意的是，由于蒸馏完全在训练阶段进行，因此在推理时没有额外的计算成本。大量实验表明，MRPD在对抗广泛的白盒和黑盒攻击方面显著优于最先进的防御方法，甚至在干净数据上也能实现更好的性能。我们的工作通过有效利用多模态知识，为构建鲁棒的3D视觉系统提供了一种新的、实用的范式。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决3D点云模型容易受到对抗攻击的问题，导致模型在安全敏感应用中可靠性降低。这个问题在现实中非常重要，因为3D点云模型被广泛应用于自动驾驶、机器人等关键领域，对抗攻击可能导致严重后果，如自动驾驶系统中恶意修改的点云可能误识别行人，造成致命事故。同时，现有防御方法要么计算开销高，要么泛化能力差，难以满足实时应用需求。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先分析了3D点云对抗防御的两个主要挑战：高计算开销和低泛化能力。观察到大规模视觉语言模型(VLM)具有强大的鲁棒性和知识迁移能力，但现有方法如PointCLIP和ULIP主要关注任务准确性而非鲁棒性。因此，作者设计了多模态教师-学生框架，借鉴了知识蒸馏的思想，但创新性地引入了三个互补的教师模型(视觉、文本和3D模型)和置信门控机制。这种方法融合了多模态学习的概念，但专门针对对抗鲁棒性进行了优化，避免了传统防御方法的高计算开销问题。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是将鲁棒性知识蒸馏到轻量级的可学习提示中，而不是修改模型架构或使用昂贵的对抗训练。整体流程分为两个阶段：1) 训练阶段：利用三个教师模型(视觉、文本和3D)提供鲁棒监督，通过置信门控对比损失过滤不可靠信号，并使用动态加权机制平衡不同模态的贡献，将知识转移到学生模型的提示中；2) 推理阶段：仅保留优化后的轻量级提示，提供对抗防御能力，无需额外计算开销。这种方法实现了零推理开销的鲁棒防御，同时保持了模型架构不变。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1) 提出多模态鲁棒提示蒸馏框架，将图像、文本和3D教师的知识融合到轻量级提示中；2) 引入置信门控机制，动态过滤不可靠的教师信号；3) 设计动态加权策略，自动平衡不同模态的重要性；4) 实现零推理开销的鲁棒防御。相比之前的工作，MRPD突破了传统防御方法在计算开销和泛化能力之间的权衡，超越了现有视觉语言模型只关注任务准确性的局限，提供了更全面、高效的对抗防御方案，特别是在真实世界场景中表现优异。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'MRDP通过多模态知识蒸馏将鲁棒性嵌入到轻量级提示中，实现了零计算开销的3D点云模型对抗防御，显著提高了模型在各种攻击下的泛化能力和实用性。'}


### 论文摘要

Adversarial attacks pose a significant threat to learning-based 3D point cloud models, critically undermining their reliability in security-sensitive applications. Existing defense methods often suffer from (1) high computational overhead and (2) poor generalization ability across diverse attack types. To bridge these gaps, we propose a novel yet efficient teacher-student framework, namely Multimodal Robust Prompt Distillation (MRPD) for distilling robust 3D point cloud model. It learns lightweight prompts by aligning student point cloud model's features with robust embeddings from three distinct teachers: a vision model processing depth projections, a high-performance 3D model, and a text encoder. To ensure a reliable knowledge transfer, this distillation is guided by a confidence-gated mechanism which dynamically balances the contribution of all input modalities. Notably, since the distillation is all during the training stage, there is no additional computational cost at inference. Extensive experiments demonstrate that MRPD substantially outperforms state-of-the-art defense methods against a wide range of white-box and black-box attacks, while even achieving better performance on clean data. Our work presents a new, practical paradigm for building robust 3D vision systems by efficiently harnessing multimodal knowledge.

---

## 5. $\mathcal{E}_0$: Enhancing Generalization and Fine-Grained Control in VLA Models via Continuized Discrete Diffusion

**论文链接:** [http://arxiv.org/abs/2511.21542v1](http://arxiv.org/abs/2511.21542v1)

**作者:** Zhihao Zhan, Jiaying Zhou, Likui Zhang, Qinhan Lv, Hao Liu, Jusheng Zhang, Weizheng Li, Ziliang Chen, Tianshui Chen, Keze Wang, Liang Lin, Guangrun Wang

**发布时间:** 2025-11-26

### GPT解析

### 总结

本文介绍了E0，一种连续化的离散扩散框架，用于改进机器人操作中的Vision-Language-Action模型，解决了现有模型在泛化能力和动作精确性方面的问题。

### 背景

Vision-Language-Action模型为机器人操作提供了统一框架，但现有模型难以在多样化任务、场景和摄像机视点间泛化，且常产生粗糙或不稳定的动作。

### 目的

开发一种能够实现更强泛化能力和更精确动作控制的VLA模型，解决现有模型在跨任务、场景和视点泛化方面的不足。

### 方法

E0是一种连续化的离散扩散框架，将动作生表述为对量化动作token的迭代去噪过程。相比连续扩散策略，其优势包括：离散动作token与预训练VLM/VLA骨干网络符号结构自然对齐；离散扩散与真实世界机器人控制的量化本质匹配；支持更大粒度动作词汇表；避免基于掩码的损坏引起的分布不匹配。同时引入球形视点扰动增强方法提高鲁棒性。

### 主要发现

在LIBERO、VLABench和ManiSkill等14个多样化环境中的实验表明，E0实现了最先进性能，平均比强基线高10.7%。Franka机械臂的真实世界评估证实了E0的精确性、鲁棒性和可迁移性。

### 结论

研究确立了离散扩散作为可泛化VLA策略学习的一个有前途的方向，能够提供精确、鲁棒和可迁移的机器人操作能力。

### 翻译

Vision-Language-Action模型通过整合视觉感知、语言理解和控制生成，为机器人操作提供了统一框架。然而，现有VLA模型仍然难以在多样化的任务、场景和摄像机视点之间进行泛化，并且常常产生粗糙或不稳定的动作。我们引入了E0，一个连续化的离散扩散框架，将动作生成表述为对量化动作token的迭代去噪过程。与连续扩散策略相比，E0有两个关键优势：首先，离散动作token与预训练的VLM/VLA骨干网络的符号结构自然对齐，能够实现更强的语义条件化；其次，离散扩散与真实世界机器人控制的量化本质相匹配——硬件约束本质上使连续信号离散化——因此能够从建模正确离散动作分布的贝叶斯最优去噪器中受益，从而实现更强的泛化能力。与离散自回归和基于掩码的离散扩散模型相比，E0支持更大粒度的动作词汇表，避免了基于掩码的损坏引起的分布不匹配，从而产生更精确的细粒度动作控制。我们还引入了一种球形视点扰动增强方法，在不增加额外数据的情况下提高对摄像机变化的鲁棒性。在LIBERO、VLABench和ManiSkill上的实验表明，E0在14个多样化的环境中实现了最先进的性能，平均比强基线高出10.7%。在Franka机械臂上的真实世界评估证实，E0能够提供精确、鲁棒和可迁移的操作能力，确立了离散扩散作为可泛化VLA策略学习的一个有前途的方向。


### 论文摘要

Vision-Language-Action (VLA) models offer a unified framework for robotic manipulation by integrating visual perception, language understanding, and control generation. Yet existing VLA models still struggle to generalize across diverse tasks, scenes, and camera viewpoints, and often produce coarse or unstable actions. We introduce E0, a continuized discrete diffusion framework that formulates action generation as iterative denoising over quantized action tokens. Compared with continuous diffusion policies, E0 offers two key advantages: (1) discrete action tokens align naturally with the symbolic structure of pretrained VLM/VLA backbones, enabling stronger semantic conditioning; and 2. discrete diffusion matches the true quantized nature of real-world robot control-whose hardware constraints (e.g., encoder resolution, control frequency, actuation latency) inherently discretize continuous signals-and therefore benefits from a Bayes-optimal denoiser that models the correct discrete action distribution, leading to stronger generalization. Compared with discrete autoregressive and mask-based discrete diffusion models, E0 supports a significantly larger and finer-grained action vocabulary and avoids the distributional mismatch introduced by masking-based corruptions-yielding more accurate fine-grained action control. We further introduce a spherical viewpoint perturbation augmentation method to improve robustness to camera shifts without additional data. Experiments on LIBERO, VLABench, and ManiSkill show that E0 achieves state-of-the-art performance across 14 diverse environments, outperforming strong baselines by 10.7% on average. Real-world evaluation on a Franka arm confirms that E0 delivers precise, robust, and transferable manipulation, establishing discrete diffusion as a promising direction for generalizable VLA policy learning.

---

## 6. CanKD: Cross-Attention-based Non-local operation for Feature-based Knowledge Distillation

**论文链接:** [http://arxiv.org/abs/2511.21503v1](http://arxiv.org/abs/2511.21503v1)

**作者:** Shizhe Sun, Wataru Ohyama

**发布时间:** 2025-11-26

**备注:** WACV 2026 Accepted

### GPT解析

### 总结

提出了一种基于交叉注意力的非局部知识蒸馏方法（CanKD），这是一种新颖的基于特征的知识蒸馏框架，利用交叉注意力机制增强知识转移过程。

### 背景

传统的基于自注意力的知识蒸馏方法独立地对齐教师和学生特征图，存在局限性。

### 目的

开发一种能够更全面地捕获像素级关系，提高特征表示学习的知识蒸馏方法。

### 方法

CanKD允许学生特征图中的每个像素动态考虑教师特征图中的所有像素，实现非局部知识转移。该方法仅引入额外的损失函数，就能比现有的注意力引导蒸馏方法实现更好的性能。

### 主要发现

在目标检测和图像分割任务上的大量实验表明，CanKD优于最先进特征和混合蒸馏方法。

### 结论

实验结果突显了CanKD作为计算机视觉任务中注意力引导蒸馏新范式的潜力。

### 翻译

我们提出了基于交叉注意力的非局部知识蒸馏（CanKD），这是一种新颖的基于特征的知识蒸馏框架，利用交叉注意力机制来增强知识转移过程。与传统的基于自注意力的蒸馏方法独立对齐教师和学生特征图不同，CanKD使学生特征图中的每个像素能够动态考虑教师特征图中的所有像素。这种非局部知识转移更全面地捕获了像素间关系，提高了特征表示学习能力。与现有的注意力引导蒸馏方法相比，我们的方法仅引入一个额外的损失函数就能实现更优的性能。在目标检测和图像分割任务上的大量实验表明，CanKD优于最先进的特征和混合蒸馏方法。这些实验结果突显了CanKD作为计算机视觉任务中注意力引导蒸馏新范式的潜力。代码可在https://github.com/tori-hotaru/CanKD获取。


### 论文摘要

We propose Cross-Attention-based Non-local Knowledge Distillation (CanKD), a novel feature-based knowledge distillation framework that leverages cross-attention mechanisms to enhance the knowledge transfer process. Unlike traditional self-attention-based distillation methods that align teacher and student feature maps independently, CanKD enables each pixel in the student feature map to dynamically consider all pixels in the teacher feature map. This non-local knowledge transfer more thoroughly captures pixel-wise relationships, improving feature representation learning. Our method introduces only an additional loss function to achieve superior performance compared with existing attention-guided distillation methods. Extensive experiments on object detection and image segmentation tasks demonstrate that CanKD outperforms state-of-the-art feature and hybrid distillation methods. These experimental results highlight CanKD's potential as a new paradigm for attention-guided distillation in computer vision tasks. Code is available at https://github.com/tori-hotaru/CanKD

---

## 7. A Systematic Study of Model Merging Techniques in Large Language Models

**论文链接:** [http://arxiv.org/abs/2511.21437v1](http://arxiv.org/abs/2511.21437v1)

**作者:** Oğuz Kağan Hitit, Leander Girrbach, Zeynep Akata

**发布时间:** 2025-11-26

### GPT解析

### 总结

这篇论文系统评估了六种模型合并方法在大型语言模型上的表现，发现只有最简单的方法Task Arithmetic能有效提升LLM性能，而其他先进方法通常会导致性能下降，指出了需要开发专门针对LLMs的合并技术。

### 背景

模型合并是一种将多个微调检查点组合成单一模型的方法，无需额外训练，对模型重用和性能提升具有吸引力。然而，目前尚不清楚这种方法在小模型和分类器上的优势是否适用于大型语言模型。

### 目的

对六种最先进的模型合并方法进行大规模、系统性评估，研究这些方法在四个开源权重LLM、每个基础模型的十二个微调检查点以及十六个标准LLM基准测试上的表现。

### 方法

通过标准化基准测试评估六种最先进的合并方法（包括最近出现的子空间方法），测量合并模型优于基础模型的可能性以及相对于最佳单个检查点的相对增益。

### 主要发现

最古老和最简单的方法Task Arithmetic是唯一能够可靠地提高LLM性能的方法；其他基于干扰感知和子空间的合并方法通常会导致显著的性能下降；当前的合并技术不能直接转移到现代LLMs上。

### 结论

需要设计专门针对LLMs的合并算法以及合并感知的微调方法，论文接受后将发布相关代码。

### 翻译

模型合并将多个微调检查点组合成单一模型而无需额外训练，提供了一种重用模型和高效提升性能的有吸引力的方法。然而，目前尚不清楚在小模型和分类器上报告的优势是否适用于大型语言模型。我们针对四个开源权重LLM、每个基础模型的十二个微调检查点以及十六个标准LLM基准测试，对六种最先进的合并方法（包括最近的子空间方法）进行了大规模、系统性评估。通过标准化基准测试评估，我们测量了合并模型优于基础模型的可能性以及相对于最佳单个检查点的相对增益。我们的结果表明，最古老和最简单的方法Task Arithmetic是唯一能够可靠地在LLMs上提高性能的方法。其他干扰感知和子空间合并方法通常会导致显著的性能下降。我们的研究结果表明，当前的合并技术不能直接转移到现代LLMs上。这促使我们设计专门针对LLMs的合并算法和合并感知的微调方法。论文接受后将发布代码。


### 论文摘要

Model merging combines multiple fine-tuned checkpoints into a single model without additional training, offering an attractive approach to reusing models and efficiently improving performance. However, it remains unclear whether the advantages reported for smaller models and classifiers generalize to LLMs. We present a large-scale, systematic evaluation of six state-of-the-art merging methods, including recent subspace methods, across four open-weight LLMs, twelve fine-tuned checkpoints per base model, and sixteen standard LLM benchmarks. Evaluating through standardized benchmarks, we measure both the probability that a merged model outperforms the base model and relative gains over the best individual checkpoint. Our results show that the oldest and simplest method, Task Arithmetic, is the only approach that reliably yields performance gains on LLMs. Other interference-aware and subspace merging methods typically result in significant performance drops. Our findings indicate that current merging techniques do not directly transfer to modern LLMs. This motivates the design of LLM-specific merging algorithms and merging-aware fine-tuning methods. Code will be released upon acceptance of this paper.

---

## 8. Training Introspective Behavior: Fine-Tuning Induces Reliable Internal State Detection in a 7B Model

**论文链接:** [http://arxiv.org/abs/2511.21399v1](http://arxiv.org/abs/2511.21399v1)

**作者:** Joshua Fonseca Rivera

**发布时间:** 2025-11-26

**备注:** 16 pages, 8 figures

### GPT解析

### 总结

Lindsey (2025)研究了语言模型中的内省意识，发现模型有时能检测注入的激活模式但不可靠。本研究通过微调训练，使模型从几乎完全失败转变为可靠检测注入的'思想'，满足准确性、基础性和内在性三个标准，表明内省能力可以直接训练而非仅等待自然出现。

### 背景

Lindsey (2025)通过四个实验发现语言模型有时能够检测和识别注入的激活模式，但不可靠（最佳模型成功率约20%）。

### 目的

研究语言模型的内省意识能力是否可以直接训练出来，而不是等待其自然出现。

### 方法

通过对短暂的单令牌注入进行微调，将一个70亿参数模型从几乎完全失败（准确率0.4%，假阳性率6.7%）转变为可靠检测（在保留概念上达到85%的准确率，α=40，0%假阳性）。

### 主要发现

模型能够检测在单个令牌位置注入的转瞬即逝的'思想'并保留信息；训练后的模型满足Lindsey的三个标准：准确性、基础性和内在性；模型学习的是可转移技能而非记忆特定向量；这解决了Lindsey提出的开放性问题。

### 结论

至少内省行为的一个组成部分可以直接诱导，为内置AI透明度提供了一条途径。

### 翻译

林赛(2025)通过四个实验研究了语言模型中的内省意识，发现模型有时能够检测和识别注入的激活模式——但不可靠(最佳模型成功率约20%)。我们关注这些实验中的第一个——注入'思想'的自我报告，并询问这种能力是否可以直接训练出来，而不是等待其自然出现。通过对短暂的单令牌注入进行微调，我们将一个70亿参数模型从几乎完全失败(准确率0.4%，假阳性率6.7%)转变为可靠检测(在保留概念上达到85%的准确率，α=40，0%假阳性)。我们的模型检测在单个令牌位置注入的转瞬即逝的'思想'，保留该信息，并在后续生成步骤中报告语义内容。在该任务上，我们的训练模型满足Lindsey的三个标准：准确性(正确识别)、基础性(0/60假阳性)和内在性(检测先于言语表达)。对未见概念向量的泛化能力(7.5pp差距)表明模型学习了一种可转移的技能，而不是记忆特定的向量，尽管这并未建立Lindsey意义上的元认知表征。这些结果解决了Lindsey提出的一个开放性问题：即'为内省进行训练是否能帮助消除跨模型差异'。我们表明至少内省行为的一个组成部分可以直接诱导，为内置AI透明度提供了一条途径。


### 论文摘要

Lindsey (2025) investigates introspective awareness in language models through four experiments, finding that models can sometimes detect and identify injected activation patterns -- but unreliably (~20% success in the best model). We focus on the first of these experiments -- self-report of injected "thoughts" -- and ask whether this capability can be directly trained rather than waiting for emergence. Through fine-tuning on transient single-token injections, we transform a 7B parameter model from near-complete failure (0.4% accuracy, 6.7% false positive rate) to reliable detection (85% accuracy on held-out concepts at α=40, 0% false positives). Our model detects fleeting "thoughts" injected at a single token position, retains that information, and reports the semantic content across subsequent generation steps. On this task, our trained model satisfies three of Lindsey's criteria: accuracy (correct identification), grounding (0/60 false positives), and internality (detection precedes verbalization). Generalization to unseen concept vectors (7.5pp gap) demonstrates the model learns a transferable skill rather than memorizing specific vectors, though this does not establish metacognitive representation in Lindsey's sense. These results address an open question raised by Lindsey: whether "training for introspection would help eliminate cross-model differences." We show that at least one component of introspective behavior can be directly induced, offering a pathway to built-in AI transparency.

---

## 9. RIA: A Ranking-Infused Approach for Optimized listwise CTR Prediction

**论文链接:** [http://arxiv.org/abs/2511.21394v1](http://arxiv.org/abs/2511.21394v1)

**作者:** Guoxiao Zhang, Tan Qu, Ao Li, DongLin Ni, Qianlong Xie, Xingxing Wang

**发布时间:** 2025-11-26

### GPT解析

### 总结

本文提出RIA框架，通过四个关键组件实现排序和重排序的无缝集成，在保持低延迟的同时提升了推荐系统的性能。

### 背景

现有重排序方法通常将排序和重排序解耦，导致在严格延迟约束下，弱列表级评估模型面临组合稀疏性和有限表示能力的问题。

### 目的

开发一个统一、端到端的框架，无缝集成点级和列表级评估，以提升推荐质量。

### 方法

提出RIA(Ranking-Infused Architecture)框架，包含四个关键组件：(1)用户和候选双Transformer(UCDT)用于细粒度用户-物品-上下文建模；(2)上下文感知用户历史和目标(CUHT)模块用于位置敏感偏好学习；(3)列表级多HSTU(LMH)模块捕获层次化物品依赖关系；(4)嵌入缓存(EC)模块在推理中桥接效率和效果。通过在排序和重排序间共享表示，实现知识转移并保持低延迟。

### 主要发现

RIA在公共和工业数据集上超越最先进模型，显著提升AUC和LogLoss指标；在美团广告系统部署后，CTR提升1.69%，CPM增加4.54%。

### 结论

RIA是一个有效的统一框架，能够解决现有排序和重排序解耦方法的问题，在保持低延迟的同时提升推荐系统性能。

### 翻译

重排序通过建模物品交互来提升推荐质量。然而，现有方法通常将排序和重排序解耦，导致在严格延迟约束下，弱列表级评估模型遭受组合稀疏性和有限表示能力的问题。在本文中，我们提出RIA(Ranking-Infused Architecture)，这是一个统一、端到端的框架，无缝集成了点级和列表级评估。RIA引入了四个关键组件：(1)用户和候选双Transformer(UCDT)用于细粒度的用户-物品-上下文建模；(2)上下文感知用户历史和目标(CUHT)模块用于位置敏感的偏好学习；(3)列表级多HSTU(LMH)模块捕获层次化的物品依赖关系；(4)嵌入缓存(EC)模块在推理过程中桥接效率和效果。通过在排序和重排序之间共享表示，RIA能够在保持低延迟的同时实现丰富的上下文知识转移。大量实验表明，RIA在公共和工业数据集上都优于最先进的模型，在AUC和LogLoss方面取得显著提升。部署在美团广告系统中，RIA在在线A/B测试中实现了点击率(CTR)提升1.69%和千次展示成本(CPM)增加4.54%。


### 论文摘要

Reranking improves recommendation quality by modeling item interactions. However, existing methods often decouple ranking and reranking, leading to weak listwise evaluation models that suffer from combinatorial sparsity and limited representational power under strict latency constraints. In this paper, we propose RIA (Ranking-Infused Architecture), a unified, end-to-end framework that seamlessly integrates pointwise and listwise evaluation. RIA introduces four key components: (1) the User and Candidate DualTransformer (UCDT) for fine-grained user-item-context modeling; (2) the Context-aware User History and Target (CUHT) module for position-sensitive preference learning; (3) the Listwise Multi-HSTU (LMH) module to capture hierarchical item dependencies; and (4) the Embedding Cache (EC) module to bridge efficiency and effectiveness during inference. By sharing representations across ranking and reranking, RIA enables rich contextual knowledge transfer while maintaining low latency. Extensive experiments show that RIA outperforms state-of-the-art models on both public and industrial datasets, achieving significant gains in AUC and LogLoss. Deployed in Meituan advertising system, RIA yields a +1.69% improvement in Click-Through Rate (CTR) and a +4.54% increase in Cost Per Mille (CPM) in online A/B tests.

---

## 10. Discovery and recovery of crystalline materials with property-conditioned transformers

**论文链接:** [http://arxiv.org/abs/2511.21299v1](http://arxiv.org/abs/2511.21299v1)

**作者:** Cyprien Bone, Matthew Walker, Kuangdai Leng, Luis M. Antunes, Ricardo Grau-Crespo, Amil Aligayev, Javier Dominguez, Keith T. Butler

**发布时间:** 2025-11-26

### GPT解析

### 总结

CrystaLLM-π是一种条件自回归框架，通过将连续属性表示直接集成到transformer的注意力机制中，解决了传统条件生成方法中的离散标记化限制和灾难性遗忘问题。

### 背景

生成式模型在加速新型功能材料设计和发现方面显示出巨大潜力，条件生成增强了逆向设计能力，但基于transformer的方法受到离散标记方案和微调过程中灾难性遗忘的制约。

### 目的

开发一种条件生成框架，能够直接处理连续属性表示，避免离散标记化的限制，并保留预训练知识。

### 方法

提出Property-Key-Value (PKV) Prefix attention和PKV Residual attention两种架构，绕过序列级标记化，保留在晶体学信息文件上无监督预训练的基础知识。

### 主要发现

该框架在结构恢复任务中处理高维异质X射线衍射图案，实现与专业模型相竞争的准确性；在材料发现任务中生成经DFT验证的新型稳定光伏材料，并隐式学习针对最佳带隙区域的能力。

### 结论

CrystaLLM-π为逆向材料设计提供了一个统一、灵活且计算高效的框架。

### 翻译

生成式模型最近在加速新型功能材料的设计和发现方面显示出巨大潜力。条件生成通过允许逆向设计增强了这种能力，在生成过程中可以请求特定的期望属性。然而，基于transformer方法的条件生成受到离散标记方案的限制，以及在微调过程中灾难性遗忘的风险。本研究介绍了CrystaLLM-π（属性注入），一种条件自回归框架，将连续属性表示直接集成到transformer的注意力机制中。提出了Property-Key-Value (PKV) Prefix attention和PKV Residual attention两种架构。这些方法绕过了低效的序列级标记化，并保留了从晶体学信息文件（CIFs）作为文本输入的无监督预训练的基础知识。我们通过系统性的鲁棒性研究验证了这些机制的有效性，并评估了框架在两个不同任务上的多功能性。首先，对于结构恢复，模型处理高维、异质的X射线衍射图案，实现了与专业模型相竞争的结构准确性，展示了在实验结构恢复和多晶型物区分中的应用。其次，对于材料发现，模型在专用光伏数据集上进行微调，生成经密度泛函理论（DFT）验证的新型稳定候选材料。它隐式地学习针对高光伏效率的最佳带隙区域，展示了映射复杂结构-属性关系的能力。CrystaLLM-π为逆向材料设计提供了一个统一、灵活且计算高效的框架。


### 论文摘要

Generative models have recently shown great promise for accelerating the design and discovery of new functional materials. Conditional generation enhances this capacity by allowing inverse design, where specific desired properties can be requested during the generation process. However, conditioning of transformer-based approaches, in particular, is constrained by discrete tokenisation schemes and the risk of catastrophic forgetting during fine-tuning. This work introduces CrystaLLM-π (property injection), a conditional autoregressive framework that integrates continuous property representations directly into the transformer's attention mechanism. Two architectures, Property-Key-Value (PKV) Prefix attention and PKV Residual attention, are presented. These methods bypass inefficient sequence-level tokenisation and preserve foundational knowledge from unsupervised pre-training on Crystallographic Information Files (CIFs) as textual input. We establish the efficacy of these mechanisms through systematic robustness studies and evaluate the framework's versatility across two distinct tasks. First, for structure recovery, the model processes high-dimensional, heterogeneous X-ray diffraction patterns, achieving structural accuracy competitive with specialised models and demonstrating applications to experimental structure recovery and polymorph differentiation. Second, for materials discovery, the model is fine-tuned on a specialised photovoltaic dataset to generate novel, stable candidates validated by Density Functional Theory (DFT). It implicitly learns to target optimal band gap regions for high photovoltaic efficiency, demonstrating a capability to map complex structure-property relationships. CrystaLLM-π provides a unified, flexible, and computationally efficient framework for inverse materials design.

---

## 11. Sampling-Based Optimization with Parallelized Physics Simulator for Bimanual Manipulation

**论文链接:** [http://arxiv.org/abs/2511.21264v1](http://arxiv.org/abs/2511.21264v1)

**作者:** Iryna Hurova, Alinjar Dan, Karl Kruusamäe, Arun Kumar Singh

**发布时间:** 2025-11-26

**备注:** 9 pages, 5 figures

### GPT解析

### 总结

本文提出了一种基于采样的优化框架，使用GPU加速的物理模拟器作为世界模型，解决存在静态障碍物时的复杂双臂操作任务。

### 背景

近年来，双臂操作已成为机器人学领域的热点，端到端学习是解决双臂任务的主要策略，但这类方法难以推广到新场景，特别是在杂乱环境中。

### 目的

提出一种替代端到端学习的范式，解决基于学习的方法在复杂环境中的泛化问题，实现双臂机器人在障碍物环境中的高效操作。

### 方法

开发了一种定制的模型预测路径积分控制(MPPI)算法，使用精心设计的特定任务成本函数进行指导，并利用GPU加速的MuJoCo物理引擎评估机器人-物体交互。

### 主要发现

该方法能解决PerAct²基准测试中显著更难版本的任务，如要求球通过障碍物课程进行点对点传输；在普通GPU上实现实时性能；利用MuJoCo的独特功能促进成功的模拟到现实转移。

### 结论

通过样本复杂性和鲁棒性的统计分析，量化了所提出方法的性能，证明了其在复杂双臂操作任务中的有效性。

### 翻译

近年来，双臂操作已成为机器人学领域的强烈兴趣点，端到端学习已成为解决双臂任务的主要策略。然而，这类基于学习方法的一个关键局限是难以推广到新场景，特别是在杂乱环境中。本文提出了一种替代范式：一种基于采样的优化框架，使用GPU加速的物理模拟器作为其世界模型。我们证明这种方法可以在存在静态障碍物的情况下解决复杂双臂操作任务。我们的贡献是一种定制的模型预测路径积分控制(MPPI)算法，由精心设计的特定任务成本函数指导，使用GPU加速的MuJoCo高效评估机器人-物体交互。我们将此方法应用于解决PerAct²基准测试中显著更具挑战性的任务版本，例如要求球通过障碍物课程进行点对点传输。此外，我们证实该方法在普通GPU上实现实时性能，并利用MuJoCo的独特功能促进成功的模拟到现实转移。本文最后对样本复杂性和鲁棒性进行了统计分析，量化了我们方法的性能。项目网站可在以下网址访问：https://sites.google.com/view/bimanualakslabunitartu 。


### 论文摘要

In recent years, dual-arm manipulation has become an area of strong interest in robotics, with end-to-end learning emerging as the predominant strategy for solving bimanual tasks. A critical limitation of such learning-based approaches, however, is their difficulty in generalizing to novel scenarios, especially within cluttered environments. This paper presents an alternative paradigm: a sampling-based optimization framework that utilizes a GPU-accelerated physics simulator as its world model. We demonstrate that this approach can solve complex bimanual manipulation tasks in the presence of static obstacles. Our contribution is a customized Model Predictive Path Integral Control (MPPI) algorithm, \textbf{guided by carefully designed task-specific cost functions,} that uses GPU-accelerated MuJoCo for efficiently evaluating robot-object interaction. We apply this method to solve significantly more challenging versions of tasks from the PerAct$^{2}$ benchmark, such as requiring the point-to-point transfer of a ball through an obstacle course. Furthermore, we establish that our method achieves real-time performance on commodity GPUs and facilitates successful sim-to-real transfer by leveraging unique features within MuJoCo. The paper concludes with a statistical analysis of the sample complexity and robustness, quantifying the performance of our approach. The project website is available at: https://sites.google.com/view/bimanualakslabunitartu .

---

## 12. Portfolio Optimization via Transfer Learning

**论文链接:** [http://arxiv.org/abs/2511.21221v1](http://arxiv.org/abs/2511.21221v1)

**作者:** Kexin Wang, Xiaomeng Zhang, Xinyu Zhang

**发布时间:** 2025-11-26

### GPT解析

### 总结

该研究提出了一种基于迁移学习的投资组合策略，利用跨市场信息提高目标市场的投资表现，通过前向验证增强策略有效性，能够选择性整合有效信息并丢弃误导信息，从而实现最大夏普比率。

### 背景

资产市场通常表现出共享的信息特征，这为跨市场信息的利用提供了基础。

### 目的

开发一种投资组合策略，利用跨市场信息提高目标市场的投资表现，并通过前向验证来增强策略的有效性。

### 方法

基于迁移学习的投资组合策略，通过识别和利用信息数据集，选择性整合有效信息并丢弃误导信息，以实现最大夏普比率。

### 主要发现

该策略能够渐近地实现最大夏普比率，在数值研究和案例研究中表现出良好的性能。

### 结论

基于迁移学习的投资组合策略能够有效利用跨市场信息提高投资表现，在双重上市股票和跨行业股票投资组合中表现出色。

### 翻译

认识到资产市场通常表现出共享的信息特征，我们开发了一种基于迁移学习的投资组合策略，通过利用跨市场信息并通过前向验证来提高目标市场的投资表现。我们的策略渐近地识别并利用信息数据集，选择性地整合有效信息同时丢弃误导信息。这使得我们的策略能够渐近地实现最大夏普比率。数值研究和两个投资组合的案例研究证明了其良好的性能：一个由A股和H股双重上市股票组成的投资组合，另一个由美国各行业股票组成的投资组合。


### 论文摘要

Recognizing that asset markets generally exhibit shared informational characteristics, we develop a portfolio strategy based on transfer learning that leverages cross-market information to enhance the investment performance in the market of interest by forward validation. Our strategy asymptotically identifies and utilizes the informative datasets, selectively incorporating valid information while discarding the misleading information. This enables our strategy to achieve the maximum Sharpe ratio asymptotically. The promising performance is demonstrated by numerical studies and case studies of two portfolios: one consisting of stocks dual-listed in A-shares and H-shares, and another comprising equities from various industries of the United States.

---

## 13. BotaCLIP: Contrastive Learning for Botany-Aware Representation of Earth Observation Data

**论文链接:** [http://arxiv.org/abs/2511.21194v1](http://arxiv.org/abs/2511.21194v1)

**作者:** Selene Cerna, Sara Si-Moussi, Wilfried Thuiller, Hadrien Hendrikx, Vincent Miele

**发布时间:** 2025-11-26

### GPT解析

### 总结

本文提出了BotaCLIP，一种轻量级多模态对比框架，用于将预训练的基础模型适应到特定领域（植物学领域），通过对比学习内化生态结构，缓解灾难性遗忘，并在三个生态任务中展示了优于现有方法的性能。

### 背景

基础模型已展现出在图像、文本和音频等多种模态中学习丰富、可迁移表示的能力，在现代机器学习管道中，这些表示通常替代原始数据作为下游任务的主要输入。

### 目的

解决如何在不从头训练或不产生显著计算成本的情况下，将预训练的基础模型适应以注入领域特定知识的挑战。

### 方法

引入BotaCLIP，一个轻量级多模态对比框架，通过将高分辨率航空影像与植物学记录进行对比学习来适应预训练的地球观测基础模型（DOFA），使用正则化策略缓解灾难性遗忘。

### 主要发现

在三个生态任务（植物存在预测、蝴蝶出现建模和土壤营养群丰度估计）中，BotaCLIP表示相比DOFA和监督基线显示出一致的改进。

### 结论

领域感知的基础模型适应可以将专家知识注入数据稀缺环境，实现节俭的表示学习，为实际应用如生物多样性建模提供了有效方法。

### 翻译

基础模型已展现出在图像、文本和音频等多种模态中学习丰富、可迁移表示的非凡能力。在现代机器学习管道中，这些表示通常替代原始数据作为下游任务的主要输入。在本文中，我们解决了如何在不从头训练或不产生显著计算成本的情况下，将预训练的基础模型适应以注入领域特定知识的挑战。为此，我们引入了BotaCLIP，一个轻量级多模态对比框架，通过将高分辨率航空影像与植物学记录进行对比学习，来适应预训练的地球观测基础模型（DOFA）。与通用嵌入不同，BotaCLIP通过对比学习和正则化策略内化生态结构，缓解灾难性遗忘。一旦训练完成，所得嵌入将作为下游预测器的可迁移表示。受生物多样性建模实际应用的启发，我们在三个生态任务中评估了BotaCLIP表示：植物存在预测、蝴蝶出现建模和土壤营养群丰度估计。结果显示，与DOFA和监督基线相比，BotaCLIP取得了持续改进。更广泛地说，这项工作展示了如何通过领域感知的基础模型适应，将专家知识注入数据稀缺环境，实现节俭的表示学习。


### 论文摘要

Foundation models have demonstrated a remarkable ability to learn rich, transferable representations across diverse modalities such as images, text, and audio. In modern machine learning pipelines, these representations often replace raw data as the primary input for downstream tasks. In this paper, we address the challenge of adapting a pre-trained foundation model to inject domain-specific knowledge, without retraining from scratch or incurring significant computational costs. To this end, we introduce BotaCLIP, a lightweight multimodal contrastive framework that adapts a pre-trained Earth Observation foundation model (DOFA) by aligning high-resolution aerial imagery with botanical relevés. Unlike generic embeddings, BotaCLIP internalizes ecological structure through contrastive learning with a regularization strategy that mitigates catastrophic forgetting. Once trained, the resulting embeddings serve as transferable representations for downstream predictors. Motivated by real-world applications in biodiversity modeling, we evaluated BotaCLIP representations in three ecological tasks: plant presence prediction, butterfly occurrence modeling, and soil trophic group abundance estimation. The results showed consistent improvements over those derived from DOFA and supervised baselines. More broadly, this work illustrates how domain-aware adaptation of foundation models can inject expert knowledge into data-scarce settings, enabling frugal representation learning.

---

## 14. When Robots Obey the Patch: Universal Transferable Patch Attacks on Vision-Language-Action Models

**论文链接:** [http://arxiv.org/abs/2511.21192v1](http://arxiv.org/abs/2511.21192v1)

**作者:** Hui Lu, Yi Yu, Yiming Yang, Chenyu Yi, Qixin Zhang, Bingquan Shen, Alex C. Kot, Xudong Jiang

**发布时间:** 2025-11-26

### GPT解析

### 总结

本研究提出了一种名为UPA-RFAS的统一框架，用于生成可迁移的对抗性补丁，能够在未知架构、微调变体和模拟到现实转移的情况下攻击VLA驱动的机器人。该框架通过特征空间优化、两阶段极小极大过程和VLA特定损失函数的组合，实现了跨模型、任务和视角的一致迁移攻击效果。

### 背景

Vision-Language-Action (VLA)模型容易受到对抗性攻击，但现有的通用和可迁移攻击研究不足。大多数现有补丁只针对单一模型过拟合，在黑盒设置中失败，这表明需要一种能够在不同模型和环境条件下工作的通用攻击方法。

### 目的

解决VLA驱动机器人在未知架构、微调变体和模拟到现实转移情况下的通用、可迁移对抗性补丁攻击问题，揭示VLA系统的安全漏洞并为未来防御建立基线。

### 方法

提出UPA-RFAS（基于鲁棒特征、注意力和语义的通用补丁攻击）统一框架，包含三个关键组件：(1) 特征空间目标与l1偏差先验和排斥性InfoNCE损失，诱导可迁移表示偏移；(2) 增强鲁棒性的两阶段极小极大过程，内循环学习样本级扰动，外循环优化通用补丁；(3) 两个VLA特定损失：补丁注意力主导性劫持文本到视觉注意力，补丁语义失准诱导图像文本不匹配。

### 主要发现

在多样化的VLA模型、操作套件和物理执行中，UPA-RFAS能够跨模型、任务和视角一致迁移，揭示了实用的基于补丁的攻击面，为理解VLA系统安全性和开发防御措施提供了重要基础。

### 结论

UPA-RFAS成功解决了VLA模型中通用和可迁移对抗性补丁的挑战，为研究VLA系统的安全性和脆弱性提供了新视角，并为未来防御措施建立了强基线。

### 翻译

视觉-语言-行动（VLA）模型容易受到对抗性攻击，然而通用和可迁移攻击仍然研究不足，因为大多数现有补丁针对单一模型过拟合，在黑盒设置中失败。为解决这一差距，我们进行了针对未知架构、微调变体和模拟到现实转移下VLA驱动机器人的通用、可迁移对抗性补丁的系统研究。我们引入了UPA-RFAS（基于鲁棒特征、注意力和语义的通用补丁攻击），这是一个统一框架，在共享特征空间中学习单个物理补丁，同时促进跨模型迁移。UPA-RFAS结合了：(i) 特征空间目标与l1偏差先验和排斥性InfoNCE损失，以诱导可迁移的表示偏移；(ii) 增强鲁棒性的两阶段极小极大过程，其中内循环学习不可见的样本级扰动，外循环针对此强化邻域优化通用补丁；(iii) 两个VLA特定损失：补丁注意力主导性以劫持文本到视觉注意力，以及补丁语义失准以诱导无标签的图像文本不匹配。跨越多样化VLA模型、操作套件和物理执行的实验表明，UPA-RFAS能够跨模型、任务和视角一致迁移，揭示了实用的基于补丁的攻击面，并为未来防御建立了强基线。


### 论文摘要

Vision-Language-Action (VLA) models are vulnerable to adversarial attacks, yet universal and transferable attacks remain underexplored, as most existing patches overfit to a single model and fail in black-box settings. To address this gap, we present a systematic study of universal, transferable adversarial patches against VLA-driven robots under unknown architectures, finetuned variants, and sim-to-real shifts. We introduce UPA-RFAS (Universal Patch Attack via Robust Feature, Attention, and Semantics), a unified framework that learns a single physical patch in a shared feature space while promoting cross-model transfer. UPA-RFAS combines (i) a feature-space objective with an $\ell_1$ deviation prior and repulsive InfoNCE loss to induce transferable representation shifts, (ii) a robustness-augmented two-phase min-max procedure where an inner loop learns invisible sample-wise perturbations and an outer loop optimizes the universal patch against this hardened neighborhood, and (iii) two VLA-specific losses: Patch Attention Dominance to hijack text$\to$vision attention and Patch Semantic Misalignment to induce image-text mismatch without labels. Experiments across diverse VLA models, manipulation suites, and physical executions show that UPA-RFAS consistently transfers across models, tasks, and viewpoints, exposing a practical patch-based attack surface and establishing a strong baseline for future defenses.

---

## 15. 论文ID: 2511.21188v1

**错误:** 无法读取论文信息 - [Errno 2] No such file or directory: 'storage/result/2511.21188v1.json'

---

## 16. MorphingDB: A Task-Centric AI-Native DBMS for Model Management and Inference

**论文链接:** [http://arxiv.org/abs/2511.21160v1](http://arxiv.org/abs/2511.21160v1)

**作者:** Wu Sai, Xia Ruichen, Yang Dingyu, Wang Rui, Lai Huihang, Guan Jiarui, Bai Jiameng, Zhang Dongxiang, Tang Xiu, Xie Zhongle, Lu Peng, Chen Gang

**发布时间:** 2025-11-26

### GPT解析

### 总结

MorphingDB是一个任务为中心的AI原生数据库管理系统，在PostgreSQL中实现了模型存储、选择和推理的自动化，通过专门的存储架构、两阶段迁移学习框架和优化技术，在多个任务上表现优于现有系统，实现了准确性与效率的良好平衡。

### 背景

数据库环境中对深度神经网络推理的需求不断增长，催生了AI原生数据库管理系统。现有解决方案要么是以模型为中心的设计，需要开发者手动选择、配置和维护模型，导致开发开销高；要么是采用以任务为中心的AutoML方法，计算成本高，与数据库管理系统集成度差。

### 目的

提出MorphingDB，一个以任务为中心的AI原生数据库管理系统，用于在PostgreSQL中自动化模型存储、选择和推理。

### 方法

1) 引入专门的模式和多维张量数据类型，支持基于BLOB的一体化和解耦模型存储；2) 设计两阶段迁移学习框架进行模型选择：通过离线嵌入历史任务构建可迁移性子空间，并通过特征感知映射进行在线投影；3) 提出预嵌入和向量共享消除冗余计算，基于DAG的批处理流水线和成本感知调度最小化推理时间。

### 主要发现

在九个公共数据集上（包括时间序列、NLP和图像任务），MorphingDB优于AI原生DBMS和AutoML平台；在模型选择中，在准确性、资源消耗和时间成本之间取得了稳健的平衡；在吞吐量和资源效率方面有显著提升。

### 结论

MorphingDB作为PostgreSQL扩展实现，使用LibTorch，展示了在数据库环境中进行深度神经网络推理的有效方法，实现了准确性与效率的良好平衡。

### 翻译

数据库环境中对深度神经网络推理日益增长的需求推动了AI原生数据库管理系统的出现。然而，现有解决方案要么依赖于需要开发者手动选择、配置和维护模型的以模型为中心的设计，导致开发开销高；要么采用计算成本高且与数据库管理系统集成度差的以任务为中心的AutoML方法。我们提出了MorphingDB，一个以任务为中心的AI原生数据库管理系统，在PostgreSQL中自动化模型存储、选择和推理。为了实现深度学习模型的灵活、I/O高效存储，我们首先引入了专门的模式和多维张量数据类型，以支持基于BLOB的一体化和解耦模型存储。然后我们设计了一个两阶段的迁移学习框架进行模型选择，通过离线嵌入历史任务构建可迁移性子空间，并通过特征感知映射的在线投影实现实时任务。为进一步优化推理吞吐量，我们提出了带有向量共享的预嵌入以消除冗余计算，以及基于DAG的批处理流水线和成本感知调度以最小化推理时间。作为PostgreSQL扩展使用LibTorch实现，MorphingDB在九个公共数据集上（包括时间序列、NLP和图像任务）优于AI原生数据库管理系统（EvaDB、Madlib、GaussML）和AutoML平台（AutoGluon、AutoKeras、AutoSklearn）。我们的评估表明，在模型选择中，在准确性、资源消耗和时间成本之间取得了稳健的平衡，并且在吞吐量和资源效率方面有显著提升。


### 论文摘要

The increasing demand for deep neural inference within database environments has driven the emergence of AI-native DBMSs. However, existing solutions either rely on model-centric designs requiring developers to manually select, configure, and maintain models, resulting in high development overhead, or adopt task-centric AutoML approaches with high computational costs and poor DBMS integration. We present MorphingDB, a task-centric AI-native DBMS that automates model storage, selection, and inference within PostgreSQL. To enable flexible, I/O-efficient storage of deep learning models, we first introduce specialized schemas and multi-dimensional tensor data types to support BLOB-based all-in-one and decoupled model storage. Then we design a transfer learning framework for model selection in two phases, which builds a transferability subspace via offline embedding of historical tasks and employs online projection through feature-aware mapping for real-time tasks. To further optimize inference throughput, we propose pre-embedding with vectoring sharing to eliminate redundant computations and DAG-based batch pipelines with cost-aware scheduling to minimize the inference time. Implemented as a PostgreSQL extension with LibTorch, MorphingDB outperforms AI-native DBMSs (EvaDB, Madlib, GaussML) and AutoML platforms (AutoGluon, AutoKeras, AutoSklearn) across nine public datasets, encompassing series, NLP, and image tasks. Our evaluation demonstrates a robust balance among accuracy, resource consumption, and time cost in model selection and significant gains in throughput and resource efficiency.

---

## 17. Which Layer Causes Distribution Deviation? Entropy-Guided Adaptive Pruning for Diffusion and Flow Models

**论文链接:** [http://arxiv.org/abs/2511.21122v1](http://arxiv.org/abs/2511.21122v1)

**作者:** Changlin Li, Jiawei Zhang, Zeyi Shi, Zongxin Yang, Zhihui Li, Xiaojun Chang

**发布时间:** 2025-11-26

**备注:** Project page: https://github.com/changlin31/EntPruner

### GPT解析

### 总结

本文提出EntPruner框架，一种基于熵引导的自动渐进式剪枝方法，用于减少扩散模型和流模型的参数冗余，提高推理速度同时保持生成质量。

### 背景

大规模视觉生成模型在视觉生成任务中表现出色，但将其迁移到下游任务时往往存在显著的参数冗余问题。

### 目的

开发一种有效的剪枝框架，减少扩散模型和流模型的参数量，提高推理效率，同时保持生成质量。

### 方法

1) 引入基于熵的块级重要性评估策略，专门针对生成模型设计；2) 使用条件熵偏差(CED)作为指导指标，优先剪枝不重要的块；3) 提出零样本自适应剪枝框架，动态确定剪枝时机和程度；4) 采用渐进式剪枝策略，避免单次剪枝导致的模式崩溃问题。

### 主要发现

在DiT和SiT模型上的实验表明，EntPruner可实现高达2.22倍的推理加速，同时在ImageNet和三个下游数据集上保持具有竞争力的生成质量。

### 结论

EntPruner框架能有效减少大规模视觉生成模型的参数冗余，提高推理速度，同时保持生成质量，解决了预训练模型迁移到下游任务时的参数冗余问题。

### 翻译

大规模视觉生成模型，包括扩散模型和流模型，在视觉生成任务中表现出色。然而，将这些预训练模型迁移到下游任务通常会导致显著的参数冗余。在本文中，我们提出了EntPruner，一种用于扩散模型和流模型的基于熵引导的自动渐进式剪枝框架。首先，我们引入了基于熵的剪枝，这是一种专门为生成模型设计的块级重要性评估策略。与判别模型不同，生成模型需要保持输出分布的多样性和条件保真度。由于每个模块在不同下游任务中的重要性可能差异很大，EntPruner使用数据依赖的条件熵偏差(CED)作为指导指标，优先剪枝不太重要的块。CED量化了移除一个块后分布与学习到的条件数据分布的偏离程度。其次，我们提出了一个零样本自适应剪枝框架，用于在训练过程中自动确定何时以及剪枝多少。这种动态策略避免了单次剪枝的缺陷，减轻了模式崩溃，并保留了模型性能。在DiT和SiT模型上的大量实验证明了EntPruner的有效性，在ImageNet和三个下游数据集上实现了高达2.22倍的推理加速，同时保持了具有竞争力的生成质量。


### 论文摘要

Large-scale vision generative models, including diffusion and flow models, have demonstrated remarkable performance in visual generation tasks. However, transferring these pre-trained models to downstream tasks often results in significant parameter redundancy. In this paper, we propose EntPruner, an entropy-guided automatic progressive pruning framework for diffusion and flow models. First, we introduce entropy-guided pruning, a block-level importance assessment strategy specifically designed for generative models. Unlike discriminative models, generative models require preserving the diversity and condition-fidelity of the output distribution. As the importance of each module can vary significantly across downstream tasks, EntPruner prioritizes pruning of less important blocks using data-dependent Conditional Entropy Deviation (CED) as a guiding metric. CED quantifies how much the distribution diverges from the learned conditional data distribution after removing a block. Second, we propose a zero-shot adaptive pruning framework to automatically determine when and how much to prune during training. This dynamic strategy avoids the pitfalls of one-shot pruning, mitigating mode collapse, and preserving model performance. Extensive experiments on DiT and SiT models demonstrate the effectiveness of EntPruner, achieving up to 2.22$\times$ inference speedup while maintaining competitive generation quality on ImageNet and three downstream datasets.

---

## 18. Scaling Foundation Models for Radar Scene Understanding

**论文链接:** [http://arxiv.org/abs/2511.21105v1](http://arxiv.org/abs/2511.21105v1)

**作者:** Pushkal Mishra, Kshitiz Bansal, Dinesh Bharadia

**发布时间:** 2025-11-26

### GPT解析

### 总结

本文介绍了RadarFM，一个雷达基础模型，通过结构化的空间语言监督学习统一的场景级表示，解决了现有雷达方法分散且特定于任务的问题。

### 背景

雷达传感器能在恶劣天气、光照条件和长距离范围内提供可靠感知，基础模型在视觉和语言理解方面取得了进展，但这些进展与雷达传感的结合尚未被充分探索。现有雷达方法分散且特定于任务，阻碍了任务间的迁移。

### 目的

引入RadarFM雷达基础模型，通过结构化的空间语言监督学习统一的场景级表示。

### 方法

提出结构化标注框架在原生雷达坐标中编码车辆分布；设计感知哈希的对比学习目标量化连续场景相似性；利用CARLA模拟器生成大规模、良好标注的雷达数据集；提出感知定位的评估指标评估空间准确性。

### 主要发现

结构化标注框架在原生雷达坐标中编码车辆分布；感知哈希的对比学习目标实现细粒度空间推理。

### 结论

RadarFM模型实现了雷达感知与基础模型的结合，解决了现有雷达方法分散且特定于任务的问题。

### 翻译

雷达传感器能在恶劣天气、光照条件和长距离范围内提供可靠的感知能力。基础模型的最新进展已转变了视觉和语言理解，但它们与雷达传感的结合在很大程度上仍未被探索。现有的雷达方法分散且特定于任务；每个下游任务采用不同的架构和训练目标，阻碍了任务间的迁移。在这项工作中，我们介绍了RadarFM：一个雷达基础模型，通过结构化的空间语言监督学习统一的场景级表示。我们做出两个关键贡献：(1)一个结构化标注框架，在原生雷达坐标中编码车辆分布，以及(2)一种感知哈希的对比学习目标，量化连续场景相似性而非二元匹配，实现细粒度空间推理。利用CARLA模拟器，我们在多样化的驾驶场景中生成大规模、良好标注的雷达数据集。我们还提出了感知定位的评估指标，评估超越传统检测度量的空间准确性。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决雷达感知方法碎片化且任务特定的问题，导致无法学习跨任务可迁移的表示。这个问题在现实中很重要，因为自动驾驶系统需要能在各种环境条件下可靠工作的感知能力，而雷达传感器在恶劣天气和长距离条件下具有独特优势。当前方法无法统一处理不同任务，限制了自动驾驶系统在复杂环境中的可靠性和适应性。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者受视觉语言模型进步的启发，特别是CLIP等基础模型展示了自然语言作为监督信号的力量。他们认识到语言可以作为一种通用标签空间，提供高级语义来统一多个感知目标。借鉴了CLIP的对比学习框架，但针对雷达场景的特殊性进行了改进。同时利用了CARLA模拟器生成大规模雷达数据，并参考了ViT-B/16视觉编码器和GPT-2类Transformer编码器等现有架构。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是通过结构化空间语言监督学习统一的场景级表示，使用原生雷达坐标编码车辆分布，并量化连续场景相似性而非二元匹配。整体流程包括：1)使用CARLA模拟器生成多样化雷达数据；2)将场景划分为距离箱和角度扇区，生成结构化标题；3)将空间信息编码为哈希向量；4)使用视觉编码器处理雷达热图，文本编码器处理标题；5)应用哈希感知对比学习对齐相似场景；6)通过生成式标题验证学习到的表示质量。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)结构化空间标题框架，以原生雷达坐标编码车辆分布；2)哈希感知对比学习目标，量化连续场景相似性；3)定位感知评估指标，直接评估空间推理准确性；4)创建了首个大规模雷达-语言数据集。相比之前工作，RadarFM避免了碎片化方法，使用语言监督替代分类监督，采用连续相似性度量而非二元匹配，并提出了专门的评估指标来评估空间推理能力。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'RadarFM通过结构化语言监督和哈希感知对比学习，实现了雷达场景的统一、细粒度空间理解，并贡献了首个大规模雷达-语言数据集和定位感知评估指标。'}


### 论文摘要

Radar sensors provide reliable perception across adverse weather, lighting, and long-range conditions. Recent advances in foundation models have transformed visual and language understanding, yet their integration with radar sensing remains largely underexplored. Existing radar approaches are fragmented and task-specific; each downstream task employs distinct architectures and training objectives, preventing transfer across tasks. In this work, we introduce RadarFM: a radar foundation model that learns unified scene-level representations through structured spatial language supervision. We make two key contributions: (1) a structured caption framework that encodes vehicle distributions in native radar coordinates, and (2) a hash-aware contrastive learning objective that quantifies continuous scene similarity rather than binary matching, enabling fine-grained spatial reasoning. Leveraging the CARLA simulator, we generate large-scale, well-annotated radar datasets across diverse driving scenarios. We also propose localization-aware metrics that assess spatial accuracy beyond traditional detection measures.

---

## 19. 论文ID: 2511.21101v1

**错误:** 无法读取论文信息 - [Errno 2] No such file or directory: 'storage/result/2511.21101v1.json'

---

## 20. Data-Driven Assessment of Concrete Slab Integrity via Impact-Echo Signals and Neural Networks

**论文链接:** [http://arxiv.org/abs/2511.21080v1](http://arxiv.org/abs/2511.21080v1)

**作者:** Yeswanth Ravichandran, Duoduo Liao, Charan Teja Kurakula

**发布时间:** 2025-11-26

**备注:** Accepted by IEEE Big Data 2025

### GPT解析

### 总结

该研究提出了一种基于机器学习的冲击回声(IE)框架，用于自动化检测混凝土桥梁桥面中的缺陷，包括分层、空隙和蜂窝等。

### 背景

混凝土桥梁桥面中的缺陷严重影响其耐久性，但难以通过目视检查或人工敲击可靠地检测。

### 目的

开发一种能够同时实现缺陷定位和常见混凝土缺陷多类分类自动化的检测框架。

### 方法

使用快速傅里叶变换将原始IE信号转换为峰值频率特征并插值到空间图中；采用无监督k-means聚类识别缺陷区域；使用实验室植入缺陷得出的真实掩码验证空间准确性；构建空间有序峰值频率序列并输入堆叠长短期记忆网络进行分类。

### 主要发现

模型对四种缺陷类型的分类整体准确率达到73%；现场验证显示实验室训练的模型能够适应真实的耦合、噪声和环境变化条件。

### 结论

该框架提高了无损评估的客观性、可扩展性和可重复性，支持网络规模的智能、数据驱动的桥梁健康监测。

### 翻译

混凝土桥梁桥面中的内部缺陷如分层、空隙和蜂窝严重影响其耐久性，但难以通过目视检查或人工敲击可靠地检测。本文提出了一种基于机器学习的冲击回声(IE)框架，可自动化实现缺陷定位和常见混凝土缺陷的多类分类。将联邦公路管理局(FHWA)实验室板和服役桥梁桥面的原始IE信号通过快速傅里叶变换(FFT)转换为主导峰值频率特征，并插值到空间图中用于缺陷区域可视化。无监督k-means聚类突出了低频率、易缺陷区域，而源于实验室植入缺陷的真实掩码(GTM)用于验证空间准确性并生成高置信度训练标签。从这些验证区域构建空间有序峰值频率序列，并输入堆叠长短期记忆(LSTM)网络，对浅层分层、深层分层、空隙和蜂窝四种缺陷类型进行分类，整体准确率为73%。在桥梁桥面上的现场验证表明，在实验室数据上训练的模型能够在真实的耦合、噪声和环境变化条件下泛化。所提出的框架提高了无损评估(NDE)的客观性、可扩展性和可重复性，支持网络规模的智能、数据驱动的桥梁健康监测。


### 论文摘要

Subsurface defects such as delamination, voids, and honeycombing critically affect the durability of concrete bridge decks but are difficult to detect reliably using visual inspection or manual sounding. This paper presents a machine learning based Impact Echo (IE) framework that automates both defect localization and multi-class classification of common concrete defects. Raw IE signals from Federal Highway Administration (FHWA) laboratory slabs and in-service bridge decks are transformed via Fast Fourier Transform (FFT) into dominant peak-frequency features and interpolated into spatial maps for defect zone visualization. Unsupervised k-means clustering highlights low-frequency, defect-prone regions, while Ground Truth Masks (GTMs) derived from seeded lab defects are used to validate spatial accuracy and generate high-confidence training labels. From these validated regions, spatially ordered peak-frequency sequences are constructed and fed into a stacked Long Short-Term Memory (LSTM) network that classifies four defect types shallow delamination, deep delamination, voids, and honeycombing with 73% overall accuracy. Field validation on the bridge deck demonstrates that models trained on laboratory data generalize under realistic coupling, noise, and environmental variability. The proposed framework enhances the objectivity, scalability, and repeatability of Non-Destructive Evaluation (NDE), supporting intelligent, data-driven bridge health monitoring at a network scale.

---

## 21. MetaRank: Task-Aware Metric Selection for Model Transferability Estimation

**论文链接:** [http://arxiv.org/abs/2511.21007v1](http://arxiv.org/abs/2511.21007v1)

**作者:** Yuhang Liu, Wenjie Zhao, Yunhui Guo

**发布时间:** 2025-11-26

**备注:** 10 figures

### GPT解析

### 总结

MetaRank是一个用于自动、任务感知的MTE指标选择的元学习框架，解决了迁移学习中预训练模型选择的高计算成本问题。

### 背景

在迁移学习中，选择合适的预训练源模型是一个关键但计算成本高昂的任务。模型迁移性估计(MTE)方法通过提供高效的代理指标来解决这个问题，无需完整微调即可对模型进行排序。然而，实践中MTE指标的选择通常是随意的，仅基于指标的历史平均表现。

### 目的

解决MTE指标效果高度依赖于任务的问题，没有单一指标能在所有目标数据集上表现最优。引入MetaRank框架，实现自动、任务感知的MTE指标选择。

### 方法

将指标选择表述为学习排序问题。使用预训练语言模型编码数据集和MTE指标的文本描述，嵌入共享语义空间。在多样化元任务上离线训练元预测器，学习数据集特征与指标机制的关系。使用列表级目标优化，优先正确排序表现最佳的指标。在线阶段基于新目标数据集的文本描述高效排序候选MTE指标。

### 主要发现

MTE指标的效果高度依赖于目标任务，没有单一指标能在所有目标数据集上表现最优。MetaRank能够有效为新的目标数据集选择最合适的MTE指标。

### 结论

在11个预训练模型和11个目标数据集上的广泛实验证明了MetaRank方法的有效性。

### 翻译

选择合适的预训练源模型是迁移学习中的一个关键但计算成本高昂的任务。模型迁移性估计(MTE)方法通过提供高效的代理指标来解决这一问题，无需完整微调即可对模型进行排序。在实践中，选择使用哪种MTE指标通常是随意的，或者仅基于指标的历史平均表现。然而，我们观察到MTE指标的效果高度依赖于任务，没有单一指标能在所有目标数据集上表现最优。为了解决这一差距，我们引入了MetaRank，一个用于自动、任务感知的MTE指标选择的元学习框架。我们将指标选择表述为一个学习排序问题。与依赖传统元特征不同，MetaRank使用预训练语言模型编码数据集和MTE指标的文本描述，将它们嵌入到共享语义空间中。然后在多样化的元任务上离线训练元预测器，学习数据集特征与指标机制之间的复杂关系，使用列表级目标进行优化，优先正确排序表现最佳的指标。在随后的在线阶段，MetaRank基于新目标数据集的文本描述高效地排序候选MTE指标，使从业者能够事先选择最合适的指标。在11个预训练模型和11个目标数据集上的广泛实验证明了我们方法的强大有效性。


### 论文摘要

Selecting an appropriate pre-trained source model is a critical, yet computationally expensive, task in transfer learning. Model Transferability Estimation (MTE) methods address this by providing efficient proxy metrics to rank models without full fine-tuning. In practice, the choice of which MTE metric to use is often ad hoc or guided simply by a metric's average historical performance. However, we observe that the effectiveness of MTE metrics is highly task-dependent and no single metric is universally optimal across all target datasets. To address this gap, we introduce MetaRank, a meta-learning framework for automatic, task-aware MTE metric selection. We formulate metric selection as a learning-to-rank problem. Rather than relying on conventional meta-features, MetaRank encodes textual descriptions of both datasets and MTE metrics using a pretrained language model, embedding them into a shared semantic space. A meta-predictor is then trained offline on diverse meta-tasks to learn the intricate relationship between dataset characteristics and metric mechanisms, optimized with a listwise objective that prioritizes correctly ranking the top-performing metrics. During the subsequent online phase, MetaRank efficiently ranks the candidate MTE metrics for a new, unseen target dataset based on its textual description, enabling practitioners to select the most appropriate metric a priori. Extensive experiments across 11 pretrained models and 11 target datasets demonstrate the strong effectiveness of our approach.

---

## 22. 论文ID: 2511.20991v1

**错误:** 无法读取论文信息 - [Errno 2] No such file or directory: 'storage/result/2511.20991v1.json'

---

## 23. AI4X Roadmap: Artificial Intelligence for the advancement of scientific pursuit and its future directions

**论文链接:** [http://arxiv.org/abs/2511.20976v1](http://arxiv.org/abs/2511.20976v1)

**作者:** Stephen G. Dale, Nikita Kazeev, Alastair J. A. Price, Victor Posligua, Stephan Roche, O. Anatole von Lilienfeld, Konstantin S. Novoselov, Xavier Bresson, Gianmarco Mengaldo, Xudong Chen, Terence J. O'Kane, Emily R. Lines, Matthew J. Allen, Amandine E. Debus, Clayton Miller, Jiayu Zhou, Hiroko H. Dodge, David Rousseau, Andrey Ustyuzhanin, Ziyun Yan, Mario Lanza, Fabio Sciarrino, Ryo Yoshida, Zhidong Leong, Teck Leong Tan, Qianxiao Li, Adil Kabylda, Igor Poltavsky, Alexandre Tkatchenko, Sherif Abdulkader Tawfik, Prathami Divakar Kamath, Theo Jaffrelot Inizan, Kristin A. Persson, Bryant Y. Li, Vir Karan, Chenru Duan, Haojun Jia, Qiyuan Zhao, Hiroyuki Hayashi, Atsuto Seko, Isao Tanaka, Omar M. Yaghi, Tim Gould, Bun Chan, Stefan Vuckovic, Tianbo Li, Min Lin, Zehcen Tang, Yang Li, Yong Xu, Amrita Joshi, Xiaonan Wang, Leonard W. T. Ng, Sergei V. Kalinin, Mahshid Ahmadi, Jiyizhe Zhang, Shuyuan Zhang, Alexei Lapkin, Ming Xiao, Zhe Wu, Kedar Hippalgaonkar, Limsoon Wong, Lorenzo Bastonero, Nicola Marzari, Dorye Luis Esteras Cordoba, Andrei Tomut, Alba Quinones Andrade, Jose-Hugo Garcia

**发布时间:** 2025-11-26

### GPT解析

### 总结

人工智能和机器学习正在重塑科学发现的方式，通过扩展研究人员能够探索、预测和设计的能力，而非替代现有方法。本文提供了AI赋能科学的前瞻性视角，涵盖生物学、化学、气候科学、数学、材料科学、物理、自动驾驶实验室和非常规计算等多个领域。

### 背景

人工智能和机器学习正在改变科学研究的范式，不是取代传统方法，而是增强科研能力，使研究人员能够探索更多可能性。

### 目的

提供AI赋能科学的前瞻性路线图，分析跨领域共同主题，确定当前瓶颈，并指出未来发展方向。

### 方法

通过分析多个科学领域中AI的应用，总结共同主题和趋势，包括数据需求、模型开发、工作流程集成和生成系统设计等方面。

### 主要发现

各领域共同需要多样化和可信赖的数据、可转移的电子结构和原子间模型、连接模拟与实验的端到端AI工作流程，以及以可合成性为基础的生成系统。大型基础模型、主动学习和自动驾驶实验室能够弥合预测与验证之间的差距，同时保持可重复性和物理解释性。

### 结论

AI赋能科学已取得显著进展，但仍面临数据、方法和基础设施方面的瓶颈。未来需要构建更强大、更透明的AI系统，以加速复杂现实世界环境中的科学发现。

### 翻译

人工智能和机器学习正在重塑我们进行科学发现的方式，不是通过取代已确立的方法，而是通过扩展研究人员能够探索、预测和设计的能力。在本路线图中，我们提供了跨越生物学、化学、气候科学、数学、材料科学、物理、自动驾驶实验室和非常规计算等领域的AI赋能科学的前瞻性观点。几个共同主题浮现：对多样化和可信赖数据的需求，可转移的电子结构和原子间模型，集成到连接模拟与实验的端到端科学工作流程中的AI系统，以及基于可合成性而非纯理想化阶段的生成系统。跨领域，我们强调了大型基础模型、主动学习和自动驾驶实验室如何在保持可重复性和物理解释性的同时，弥合预测与验证之间的循环。综合这些观点，概述了AI赋能科学的现状，确定了数据、方法和基础设施中的瓶颈，并规划了构建更强大、更透明且能够加速复杂现实世界环境中发现的AI系统的具体方向。


### 论文摘要

Artificial intelligence and machine learning are reshaping how we approach scientific discovery, not by replacing established methods but by extending what researchers can probe, predict, and design. In this roadmap we provide a forward-looking view of AI-enabled science across biology, chemistry, climate science, mathematics, materials science, physics, self-driving laboratories and unconventional computing. Several shared themes emerge: the need for diverse and trustworthy data, transferable electronic-structure and interatomic models, AI systems integrated into end-to-end scientific workflows that connect simulations to experiments and generative systems grounded in synthesisability rather than purely idealised phases. Across domains, we highlight how large foundation models, active learning and self-driving laboratories can close loops between prediction and validation while maintaining reproducibility and physical interpretability. Taken together, these perspectives outline where AI-enabled science stands today, identify bottlenecks in data, methods and infrastructure, and chart concrete directions for building AI systems that are not only more powerful but also more transparent and capable of accelerating discovery in complex real-world environments.

---

## 24. 论文ID: 2511.20936v1

**错误:** 无法读取论文信息 - [Errno 2] No such file or directory: 'storage/result/2511.20936v1.json'

---

## 25. Operationalizing Quantized Disentanglement

**论文链接:** [http://arxiv.org/abs/2511.20927v1](http://arxiv.org/abs/2511.20927v1)

**作者:** Vitoria Barin-Pacela, Kartik Ahuja, Simon Lacoste-Julien, Pascal Vincent

**发布时间:** 2025-11-25

### GPT解析

### 总结

该研究开发了一种名为Cliff的方法，通过鼓励轴对齐不连续性来实现无监督解缠结，在所有解缠结基准测试中都优于基线方法。

### 背景

最近的理论工作确立了量化因子在任意微分同胚下的无监督可识别性。该理论假设量化阈值对应于潜在因子概率密度中的轴对齐不连续点，但将这一高级原则转化为有效的实际标准仍然具有挑战性，特别是在非线性映射下。

### 目的

开发一种有效的标准来实现无监督解缠结，通过鼓励轴对齐不连续性来克服现有方法的局限性。

### 方法

作者开发了一种名为Cliff的方法，通过鼓励轴对齐不连续性来实现无监督解缠结。不连续性表现为因子估计密度的急剧变化，形成所谓的'悬崖'。根据理论中独立不连续性的定义，该方法鼓励沿一个因子的悬崖位置独立于其他因子的值。

### 主要发现

研究结果表明，Cliff方法在所有解缠结基准测试中都优于基线方法，证明了其在无监督解缠结中的有效性。

### 结论

通过鼓励轴对齐不连续性，Cliff方法成功地实现了无监督解缠结，并且在各种基准测试中表现出色，为量化因子的无监督学习提供了有效的解决方案。

### 翻译

最近的理论工作确立了量化因子在任意微分同胚下的无监督可识别性。该理论假设量化阈值对应于潜在因子概率密度中的轴对齐不连续点。通过约束学习映射具有轴对齐不连续点的密度，我们可以恢复因子的量化。然而，将这一高级原则转化为有效的实际标准仍然具有挑战性，特别是在非线性映射下。在这里，我们通过鼓励轴对齐不连续性开发了一种无监督解缠结的标准。不连续性表现为因子估计密度的急剧变化，形成我们所谓的'悬崖'。遵循理论中独立不连续性的定义，我们鼓励沿一个因子的悬崖位置独立于其他因子的值。我们表明，我们的方法Cliff在所有解缠结基准测试中都优于基线方法，证明了其在无监督解缠结中的有效性。


### 论文摘要

Recent theoretical work established the unsupervised identifiability of quantized factors under any diffeomorphism. The theory assumes that quantization thresholds correspond to axis-aligned discontinuities in the probability density of the latent factors. By constraining a learned map to have a density with axis-aligned discontinuities, we can recover the quantization of the factors. However, translating this high-level principle into an effective practical criterion remains challenging, especially under nonlinear maps. Here, we develop a criterion for unsupervised disentanglement by encouraging axis-aligned discontinuities. Discontinuities manifest as sharp changes in the estimated density of factors and form what we call cliffs. Following the definition of independent discontinuities from the theory, we encourage the location of the cliffs along a factor to be independent of the values of the other factors. We show that our method, Cliff, outperforms the baselines on all disentanglement benchmarks, demonstrating its effectiveness in unsupervised disentanglement.

---

## 26. Representation Integrity in Temporal Graph Learning Methods

**论文链接:** [http://arxiv.org/abs/2511.20873v1](http://arxiv.org/abs/2511.20873v1)

**作者:** Elahe Kooshafar

**发布时间:** 2025-11-25

**备注:** 70 pages

### GPT解析

### 总结

本研究提出了一种评估动态图表示完整性的新框架，通过一系列指标衡量嵌入变化与图变化的接近程度，为动态图学习模型提供了与任务无关且可解释的评估工具。

### 背景

现实世界系统如航空路线和加密货币转账等自然地建模为动态图，其拓扑结构随时间变化。传统基准仅通过特定任务评分评估模型，很少考虑嵌入是否真实反映网络演化。

### 目的

将表示完整性要求形式化，并开发指标来衡量嵌入变化与图变化的一致性，为动态图学习模型提供更全面的评估方法。

### 方法

使用三种合成场景（逐渐合并、突然移动和周期性重连线）筛选42个候选指标，推荐通过所有理论和实证测试的指标，并用其对常见动态图学习模型进行对比研究。

### 主要发现

验证的指标始终将理论上稳定的UASE和IPP模型排名最高；研究揭示了神经方法在特定场景下的优势；该指标与一步链接预测AUC有很强的正等级相关性。

### 结论

提出的完整性框架为动态图表示质量提供了与任务无关且可解释的评估工具，为模型选择和未来架构设计提供了更明确的指导。

### 翻译

从航空路线到加密货币转账等现实世界系统自然地建模为拓扑结构随时间变化的动态图。传统基准通过少数特定任务评分来判断动态图学习器的性能，但很少考虑嵌入本身是否仍然是 evolving network 的真实、可解释的反映。我们将这一要求形式化为表示完整性，并衍生出一组指标来衡量嵌入变化与图变化的接近程度。使用三种合成场景（逐渐合并、突然移动和周期性重连线）筛选42个候选指标。基于此，我们推荐了一个通过所有理论和实证测试的指标。特别是，这个验证的指标始终将理论上稳定的UASE和IPP模型排名最高。然后，我们使用该指标对常见动态图学习模型的表示完整性进行对比研究。这项研究揭示了神经方法在特定场景下的优势，并显示与一步链接预测AUC有很强的正等级相关性。因此，提出的完整性框架为动态图表示质量提供了与任务无关且可解释的评估工具，为模型选择和未来架构设计提供了更明确的指导。


### 论文摘要

Real-world systems ranging from airline routes to cryptocurrency transfers are naturally modelled as dynamic graphs whose topology changes over time. Conventional benchmarks judge dynamic-graph learners by a handful of task-specific scores, yet seldom ask whether the embeddings themselves remain a truthful, interpretable reflection of the evolving network. We formalize this requirement as representation integrity and derive a family of indexes that measure how closely embedding changes follow graph changes. Three synthetic scenarios, Gradual Merge, Abrupt Move, and Periodic Re-wiring, are used to screen forty-two candidate indexes. Based on which we recommend one index that passes all of our theoretical and empirical tests. In particular, this validated metric consistently ranks the provably stable UASE and IPP models highest. We then use this index to do a comparative study on representation integrity of common dynamic graph learning models. This study exposes the scenario-specific strengths of neural methods, and shows a strong positive rank correlation with one-step link-prediction AUC. The proposed integrity framework, therefore, offers a task-agnostic and interpretable evaluation tool for dynamic-graph representation quality, providing more explicit guidance for model selection and future architecture design.

---

## 27. Winning with Less for Low Resource Languages: Advantage of Cross-Lingual English_Persian Argument Mining Model over LLM Augmentation

**论文链接:** [http://arxiv.org/abs/2511.20872v1](http://arxiv.org/abs/2511.20872v1)

**作者:** Ali Jahan, Masood Ghayoomi, Annette Hautli-Janisz

**发布时间:** 2025-11-25

**备注:** Preprint. Under review

### GPT解析

### 总结

该研究探讨了跨语言方法在低资源语言论证挖掘中的应用，通过构建三种训练场景并评估其在英语和波斯语上的表现。

### 背景

论证挖掘是自然语言处理的子领域，旨在识别和提取文本中的论证组件（如前提和结论）及其关系，揭示文本的逻辑结构，可用于知识提取等任务。

### 目的

利用跨语言方法为低资源语言进行论证挖掘，构建三种训练场景，并评估其在高资源语言（英语）和低资源语言（波斯语）上的效果。

### 方法

在英语Microtext语料库及其并行波斯语翻译上评估模型；构建三种学习场景：(1)零样本迁移（仅用英语数据训练）；(2)英语-only训练增强（通过大型语言模型生成合成示例）；(3)跨语言模型（结合原始英语数据和手动翻译的波斯语句）。

### 主要发现

零样本迁移模型在英语和波斯语测试集上分别获得50.2%和50.7%的F1分数；基于LLM的增强模型将性能提高到英语上的59.2%和波斯语上的69.3%；跨语言模型在波斯语测试集上达到74.8%的F1分数，优于其他方法。

### 结论

轻量级跨语言组合可以明显胜过资源密集型的增强流程，为论证挖掘任务提供了一条实用的路径，以克服低资源语言的数据资源短缺。

### 翻译

论证挖掘是自然语言处理的一个子领域，用于识别和提取文本中的论证组件，如前提和结论，以及它们之间的关系。它揭示了文本的逻辑结构，可用于知识提取等任务。本文旨在利用跨语言方法为低资源语言进行论证挖掘，通过构建三种训练场景。我们在英语（作为高资源语言）和波斯语（作为低资源语言）上检查模型。为此，我们基于英语Microtext语料库及其并行波斯语翻译评估模型。学习场景如下：(i)零样本迁移，模型仅用英语数据训练；(ii)仅英语训练，通过大型语言模型(LLM)生成的合成示例增强；(iii)跨语言模型，结合原始英语数据和手动翻译的波斯语句。零样本迁移模型在英语测试集上获得50.2%的F1分数，在波斯语测试集上获得50.7%。基于LLM的增强模型将性能提高到英语上的59.2%和波斯语上的69.3%。跨语言模型在两种语言上训练但仅在波斯语测试集上评估，通过实现74.8%的F1分数超越了基于LLM的变体。结果表明，轻量级跨语言组合可以明显胜过资源密集型的增强流程，它为论证挖掘任务提供了一条实用的路径，以克服低资源语言的数据资源短缺。


### 论文摘要

Argument mining is a subfield of natural language processing to identify and extract the argument components, like premises and conclusions, within a text and to recognize the relations between them. It reveals the logical structure of texts to be used in tasks like knowledge extraction. This paper aims at utilizing a cross-lingual approach to argument mining for low-resource languages, by constructing three training scenarios. We examine the models on English, as a high-resource language, and Persian, as a low-resource language. To this end, we evaluate the models based on the English Microtext corpus \citep{PeldszusStede2015}, and its parallel Persian translation. The learning scenarios are as follow: (i) zero-shot transfer, where the model is trained solely with the English data, (ii) English-only training enhanced by synthetic examples generated by Large Language Models (LLMs), and (iii) a cross-lingual model that combines the original English data with manually translated Persian sentences. The zero-shot transfer model attains F1 scores of 50.2\% on the English test set and 50.7\% on the Persian test set. LLM-based augmentation model improves the performance up to 59.2\% on English and 69.3\% on Persian. The cross-lingual model, trained on both languages but evaluated solely on the Persian test set, surpasses the LLM-based variant, by achieving a F1 of 74.8\%. Results indicate that a lightweight cross-lingual blend can outperform considerably the more resource-intensive augmentation pipelines, and it offers a practical pathway for the argument mining task to overcome data resource shortage on low-resource languages.

---

## 28. Bridging Atomistic and Mesoscale Lithium Transport via Machine-Learned Force Fields and Markov State Models

**论文链接:** [http://arxiv.org/abs/2511.20863v1](http://arxiv.org/abs/2511.20863v1)

**作者:** Muhammad Nawaz Qaisrani, Christoph Kirsch, Aaron Flötotto, Jonas Hänseroth, Jules Oumard, Daniel Sebastiani, Christian Dreßler

**发布时间:** 2025-11-25

**备注:** 36 pages, 9 figures

### GPT解析

### 总结

该研究提出了一种多尺度工作流程，整合AIMD、机器学习力场和马尔可夫状态模型，建立了锂离子在固态电池阳极中扩散的原子尺度跳跃机制与介观尺度输运之间的定量联系。

### 背景

锂离子在固态电池阳极中的扩散是通过热激活在亚稳态位点之间跳跃实现的，这些位点通常被大的能量势垒分隔，因此在从头算分子动力学时间尺度上，这类事件很少发生。

### 目的

建立一个从原子尺度跳跃机制到介观尺度输运的定量一致性联系，克服AIMD时间尺度上的限制。

### 方法

采用自底而上的多尺度工作流程，整合AIMD、机器学习力场(MLFFs)和马尔可夫状态模型(MSMs)，通过微调MLFFs在AIMD参考数据上，实现大规模分子动力学模拟，并构建MSMs分析锂扩散过程。

### 主要发现

MLFFs保留了接近DFT的准确性，同时能扩展到数十纳秒的模拟；扩展的轨迹消除了AIMD中的有限尺寸偏差，产生的扩散系数与实验结果一致；从长MLFF轨迹中获得的统计收敛锂跳跃网络构建的MSMs能重现均方位移并恢复稀有扩散过程；MSM转移矩阵提供了特征弛豫时间和主要输运途径的机制见解。

### 结论

虽然该方法在无缺陷的结晶Li_xSi_y相上展示，但AIMD→MLFF→MSM框架是通用的，为描述非晶材料、缺陷介导扩散和下一代固态阳极中的锂输运提供了可转移的方法。

### 翻译

锂离子在固态电池阳极中的扩散通过热激活在亚稳态位点之间的跳跃发生，这些位点通常被大的能量势垒分隔，使得在从头算分子动力学时间尺度上此类事件罕见。本文提出了一种自底而上的多尺度工作流程，整合了从头算分子动力学、机器学习力场和马尔可夫状态模型，建立了原子尺度跳跃机制与介观尺度输运之间的定量一致性联系。在从头算分子动力学参考数据上微调的机器学习力场保留了接近密度泛函理论的准确性，同时能够扩展到数十纳秒的大规模分子动力学模拟。这些扩展的轨迹消除了从头算分子动力学中存在的强有限尺寸偏差，产生的扩散系数与实验结果高度一致。此外，从这些长的机器学习力场轨迹中，我们获得了统计收敛的锂跳跃网络并构建了马尔可夫状态模型，即使在构建过程中使用的时间滞后跨越两个数量级以上，这些模型仍保持马尔可夫性质。得到的马尔可夫状态模型忠实地重现了均方位移，并恢复了在从头算分子动力学时间尺度上不发生的稀有扩散过程。除了传播锂分布外，马尔可夫状态模型转移矩阵还提供了机制见解：它们的特征值和特征向量编码了特征弛豫时间和主要输运途径。虽然该方法在无缺陷的结晶Li_xSi_y相上得到验证，但从头算分子动力学→机器学习力场→马尔可夫状态模型框架是通用的，为描述非晶材料、缺陷介导扩散和下一代固态阳极中的锂输运提供了可转移的方法。


### 论文摘要

Lithium diffusion in solid-state battery anodes occurs through thermally activated hops between metastable sites often separated by large energy barriers, making such events rare on ab initio molecular dynamics (AIMD) timescales. Here, we present a bottom-up multiscale workflow that integrates AIMD, machine-learned force fields (MLFFs), and Markov state models (MSMs) to establish a quantitatively consistent link between atomistic hopping mechanisms and mesoscale transport. MLFFs fine-tuned on AIMD reference data retain near-DFT accuracy while enabling large-scale molecular dynamics simulations extending to tens of nanoseconds. These extended trajectories remove the strong finite-size bias present in AIMD and yield diffusion coefficients in excellent agreement with experiment. Furthermore, from these long MLFF trajectories, we obtain statistically converged lithium jump networks and construct MSMs that remain Markovian across more than two orders of magnitude in the lag times used for their construction. The resulting MSMs faithfully reproduce mean-square displacements and recover rare diffusion processes that do not occur on AIMD timescales. In addition to propagating lithium distributions, the MSM transition matrices provide mechanistic insight: their eigenvalues and eigenvectors encode characteristic relaxation timescales and dominant transport pathways. Although demonstrated for defect-free crystalline Li$_x$Si$_y$ phases, the AIMD$\rightarrow$MLFF$\rightarrow$MSM framework is general and provides a transferable approach for describing lithium transport in amorphous materials, defect-mediated diffusion, and next-generation solid-state anodes.

---

## 29. $Δ$-NeRF: Incremental Refinement of Neural Radiance Fields through Residual Control and Knowledge Transfer

**论文链接:** [http://arxiv.org/abs/2511.20804v1](http://arxiv.org/abs/2511.20804v1)

**作者:** Kriti Ghosh, Devjyoti Chakraborty, Lakshmish Ramaswamy, Suchendra M. Bhandarkar, In Kee Kim, Nancy O'Hare, Deepak Mishra

**发布时间:** 2025-11-25

### GPT解析

### 总结

Δ-NeRF是一种创新的模块化残差框架，用于神经辐射场的增量优化，解决了传统NeRF在顺序数据到达场景中的灾难性遗忘问题，实现了高效且高质量的3D重建。

### 背景

神经辐射场(NeRFs)在3D重建和新视角合成方面表现出色，但大多数现有框架在引入新视角时需要完全重新训练，这在数据顺序到达的领域应用受限，特别是在卫星地形分析中，区域会随时间被重复观测。

### 目的

开发一种能够增量优化NeRF的框架，使其能够在不访问过去数据的情况下进行更新，避免灾难性遗忘问题，同时提高训练效率和模型性能。

### 方法

Δ-NeRF框架包含：(1)残差控制器，向冻结的基础NeRF注入每层校正，无需访问过去数据；(2)不确定性感知门控机制，通过自适应组合基础和优化预测防止过度校正；(3)视角选择策略，减少高达47%的训练数据同时保持性能；(4)知识蒸馏技术，将模型压缩为原始大小20%的紧凑网络。

### 主要发现

在卫星图像实验中，Δ-NeRF实现了与联合训练相当的性能，同时减少了30-42%的训练时间；比简单微调方法在PSNR指标上提高高达43.5%；在某些指标上超越了完全联合训练的效果。

### 结论

Δ-NeRF成功解决了NeRF增量优化中的灾难性遗忘问题，使NeRF能够在顺序数据到达的场景中有效应用，特别是在卫星地形分析等领域，显著提高了训练效率和模型性能。

### 翻译

神经辐射场(NeRFs)已在3D重建和新视角合成方面展现出卓越能力。然而，大多数现有NeRF框架在引入新视角时需要完全重新训练，限制了它们在数据顺序到达领域的适用性。这一限制在卫星地形分析等需要随时间重复观测区域的领域中尤为突出。NeRF的增量优化研究仍然不足，而简单方法在无法访问过去数据时会遭受灾难性遗忘。我们提出了Δ-NeRF，一种独特的模块化残差框架，用于增量式NeRF优化。Δ-NeRF引入了几种创新技术，包括：(1)残差控制器，向冻结的基础NeRF注入每层校正，使无需访问过去数据即可进行优化；(2)不确定性感知门控机制，通过自适应组合基础和优化预测来防止过度校正；(3)视角选择策略，在保持性能的同时减少高达47%的训练数据。此外，我们采用知识蒸馏将增强模型压缩为紧凑的学生网络（原始大小的20%）。卫星图像实验表明，Δ-NeRF实现了与联合训练相当的性能，同时减少了30-42%的训练时间。Δ-NeRF持续优于现有基线，在PSNR上比简单微调提高高达43.5%，并在某些指标上超越了联合训练。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决NeRF模型在新增视图时需要完全重新训练的问题。这个问题在现实中很重要，特别是在卫星地形分析等应用中，数据是随时间顺序收集的，完全重新训练计算成本高、效率低下，限制了NeRF在数据持续到达场景中的应用。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者分析了NeRF在增量学习中的不足和灾难性遗忘问题，认识到卫星图像应用中有限视角、光照变化等额外挑战。他们设计了模块化残差框架保持基础模型冻结，借鉴了ControlNet的残差调制思想、知识蒸馏技术以及Sat-NeRF中的不确定性建模，但创新性地将这些技术整合为一种无需访问历史数据的增量学习方案。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是使用残差控制器修改冻结的基础NeRF模型，而非重新训练整个模型。整体流程包括：1)初始训练基础模型并冻结；2)增量训练阶段，训练残差控制器学习新视图校正；3)使用不确定性感知门控机制结合基础和校正输出；4)通过视图选择减少训练数据量；5)应用知识蒸馏压缩模型；6)推理时使用门控融合输出最终结果。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)零初始化残差网络控制器实现增量校正；2)不确定性感知门控机制防止过度校正；3)深度感知视图选择减少47%训练数据；4)知识蒸馏将模型压缩至20%大小。相比之前工作，Δ-NeRF无需访问历史数据，专注于卫星图像增量学习，结合门控机制和视图选择，在某些指标上甚至超过联合训练性能，计算效率提升35-45%。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'Δ-NeRF提出了一种创新的残差控制框架，使NeRF模型能够在不访问历史数据的情况下增量适应新视图，同时保持与完全重新训练相当的性能并显著降低计算成本，特别适用于卫星图像等数据顺序到达的场景。'}


### 论文摘要

Neural Radiance Fields (NeRFs) have demonstrated remarkable capabilities in 3D reconstruction and novel view synthesis. However, most existing NeRF frameworks require complete retraining when new views are introduced incrementally, limiting their applicability in domains where data arrives sequentially. This limitation is particularly problematic in satellite-based terrain analysis, where regions are repeatedly observed over time. Incremental refinement of NeRFs remains underexplored, and naive approaches suffer from catastrophic forgetting when past data is unavailable. We propose $Δ$-NeRF, a unique modular residual framework for incremental NeRF refinement. $Δ$-NeRF introduces several novel techniques including: (1) a residual controller that injects per-layer corrections into a frozen base NeRF, enabling refinement without access to past data; (2) an uncertainty-aware gating mechanism that prevents overcorrection by adaptively combining base and refined predictions; and (3) a view selection strategy that reduces training data by up to 47\% while maintaining performance. Additionally, we employ knowledge distillation to compress the enhanced model into a compact student network (20\% of original size). Experiments on satellite imagery demonstrate that $Δ$-NeRF achieves performance comparable to joint training while reducing training time by 30-42\%. $Δ$-NeRF consistently outperforms existing baselines, achieving an improvement of up to 43.5\% in PSNR over naive fine-tuning and surpassing joint training on some metrics.

---

## 30. Knowledge Distillation for Continual Learning of Biomedical Neural Fields

**论文链接:** [http://arxiv.org/abs/2511.21409v1](http://arxiv.org/abs/2511.21409v1)

**作者:** Wouter Visser, Jelmer M. Wolterink

**发布时间:** 2025-11-26

**备注:** 5 pages, 6 figures. Submitted to IEEE International Symposium on Biomedical Imaging (ISBI) 2026

### GPT解析

### 总结

本研究探讨了神经场在医学影像应用中面临的灾难性遗忘问题，并提出了使用知识蒸馏来缓解这一问题的策略。

### 背景

神经场越来越多地被用作医学影像中轻量级、连续且可微分的信号表示方法，但与离散信号表示不同，神经场不易扩展，且由于本质上是神经网络，当遇到新数据时会出现灾难性遗忘现象。

### 目的

研究不同神经场方法遭受灾难性遗忘的程度，并提出减轻此问题的策略。

### 方法

在数据增量可用的场景下，使用心脏动态MRI数据进行实验，应用知识蒸馏来减轻当时空域扩大或表示信号维度增加时的灾难性遗忘。

### 主要发现

灾难性遗忘的程度在很大程度上取决于所使用的神经场模型，知识蒸馏可以使神经场中的持续学习成为可能。

### 结论

知识蒸馏能够有效减轻神经场中的灾难性遗忘问题，使神经场能够进行持续学习。

### 翻译

神经场越来越多地被用作(生物)医学影像中轻量级、连续且可微分的信号表示。然而，与体素网格等离散信号表示不同，神经场不易扩展。由于神经场本质上是神经网络，当模型遇到新数据时，先前表示在神经场中的信号会因灾难性遗忘而退化。本研究考察了不同神经场方法遭受灾难性遗忘的程度，并提出了减轻此问题的策略。我们考虑数据增量可用的场景，只有最近的数据可用于神经场拟合。在心脏动态MRI数据的一系列实验中，我们展示了当时空域扩大或表示信号的维度增加时，知识蒸馏如何减轻灾难性遗忘。我们发现，灾难性遗忘的程度在很大程度上取决于所使用的神经场模型，而蒸馏可以使神经场中的持续学习成为可能。


### 论文摘要

Neural fields are increasingly used as a light-weight, continuous, and differentiable signal representation in (bio)medical imaging. However, unlike discrete signal representations such as voxel grids, neural fields cannot be easily extended. As neural fields are, in essence, neural networks, prior signals represented in a neural field will degrade when the model is presented with new data due to catastrophic forgetting. This work examines the extent to which different neural field approaches suffer from catastrophic forgetting and proposes a strategy to mitigate this issue. We consider the scenario in which data becomes available incrementally, with only the most recent data available for neural field fitting. In a series of experiments on cardiac cine MRI data, we demonstrate how knowledge distillation mitigates catastrophic forgetting when the spatiotemporal domain is enlarged or the dimensionality of the represented signal is increased. We find that the amount of catastrophic forgetting depends, to a large extent, on the neural fields model used, and that distillation could enable continual learning in neural fields.

---

## 31. STARFlow-V: End-to-End Video Generative Modeling with Normalizing Flows

**论文链接:** [http://arxiv.org/abs/2511.20462v2](http://arxiv.org/abs/2511.20462v2)

**作者:** Jiatao Gu, Ying Shen, Tianrong Chen, Laurent Dinh, Yuyang Wang, Miguel Angel Bautista, David Berthelot, Josh Susskind, Shuangfei Zhai

**发布时间:** 2025-11-25

**备注:** 21 pages, 9 figures. Code and samples are available at https://github.com/apple/ml-starflow

### GPT解析

### 总结

STARFlow-V是一种基于归一化流的视频生成器，具有端到端学习、鲁棒因果预测和原生似然估计等优势。它在时空潜在空间中采用全局-局部架构，通过flow-score matching提供轻量级因果去噪器，并使用视频感知的雅可比迭代方案提高采样效率。该模型支持多种生成任务，并在视觉保真度和时间一致性方面表现出色。

### 背景

归一化流是用于连续数据的基于似然的端到端生成模型，在图像生成领域已取得进展。然而，在视频生成领域，由于时空复杂性和计算成本较高，最先进的系统几乎完全依赖基于扩散的模型。

### 目的

重新审视视频生成领域的设计空间，提出一种基于归一化流的视频生成器，以解决现有方法的局限性。

### 方法

基于STARFlow构建，在时空潜在空间中操作，采用全局-局部架构将因果依赖限制在全局潜在空间同时保留丰富的帧内局部交互。提出flow-score matching配备轻量级因果去噪器，采用视频感知的雅可比迭代方案提高采样效率。利用可逆结构支持多种生成任务。

### 主要发现

STARFlow-V在视觉保真度和时间一致性方面取得了强有力效果，相比基于扩散的基线模型具有实用的采样吞吐量。首次证明归一化流能够进行高质量的自回归视频生成。

### 结论

归一化流是构建世界模型的有希望的研究方向，为视频生成领域提供了新的可能性。

### 翻译

归一化流是用于连续数据的端到端基于似然的生成模型，最近在图像生成方面取得了进展并重新受到关注。然而，在视频生成领域，由于时空复杂性和计算成本显著更高，最先进的系统几乎完全依赖基于扩散的模型。在这项工作中，我们通过提出STARFlow-V重新审视了这一设计空间，这是一种基于归一化流的视频生成器，具有端到端学习、鲁棒因果预测和原生似然估计等显著优势。基于最近提出的STARFlow，STARFlow-V在时空潜在空间中操作，采用全局-局部架构，将因果依赖限制在全局潜在空间，同时保留丰富的帧内局部交互。这减轻了随时间累积的错误，这是标准自回归扩散模型生成的常见缺陷。此外，我们提出了flow-score matching，为模型配备轻量级因果去噪器，以自回归方式提高视频生成一致性。为了提高采样效率，STARFlow-V采用视频感知的雅可比迭代方案，将内部更新重新表述为可并行化的迭代而不破坏因果性。得益于可逆结构，同一模型可原生支持文本到视频、图像到视频以及视频到视频生成任务。经验表明，STARFlow-V相对于基于扩散的基线模型，实现了强视觉保真度和时间一致性，同时具有实用的采样吞吐量。据我们所知，这些结果首次证明了归一化流能够进行高质量的自回归视频生成，确立了它们作为构建世界模型的有希望的研究方向。代码和生成的样本可在https://github.com/apple/ml-starflow获取。


### 论文摘要

Normalizing flows (NFs) are end-to-end likelihood-based generative models for continuous data, and have recently regained attention with encouraging progress on image generation. Yet in the video generation domain, where spatiotemporal complexity and computational cost are substantially higher, state-of-the-art systems almost exclusively rely on diffusion-based models. In this work, we revisit this design space by presenting STARFlow-V, a normalizing flow-based video generator with substantial benefits such as end-to-end learning, robust causal prediction, and native likelihood estimation. Building upon the recently proposed STARFlow, STARFlow-V operates in the spatiotemporal latent space with a global-local architecture which restricts causal dependencies to a global latent space while preserving rich local within-frame interactions. This eases error accumulation over time, a common pitfall of standard autoregressive diffusion model generation. Additionally, we propose flow-score matching, which equips the model with a light-weight causal denoiser to improve the video generation consistency in an autoregressive fashion. To improve the sampling efficiency, STARFlow-V employs a video-aware Jacobi iteration scheme that recasts inner updates as parallelizable iterations without breaking causality. Thanks to the invertible structure, the same model can natively support text-to-video, image-to-video as well as video-to-video generation tasks. Empirically, STARFlow-V achieves strong visual fidelity and temporal consistency with practical sampling throughput relative to diffusion-based baselines. These results present the first evidence, to our knowledge, that NFs are capable of high-quality autoregressive video generation, establishing them as a promising research direction for building world models. Code and generated samples are available at https://github.com/apple/ml-starflow.

---

## 32. Diagonal Scaling: A Multi-Dimensional Resource Model and Optimization Framework for Distributed Databases

**论文链接:** [http://arxiv.org/abs/2511.21612v1](http://arxiv.org/abs/2511.21612v1)

**作者:** Shahir Abdullah, Syed Rohit Zaman

**发布时间:** 2025-11-26

### GPT解析

### 总结

这篇论文提出了一种名为'Scaling Plane'的二维扩展模型，结合水平扩展和垂直扩展，并开发了DIAGONALSCALE算法来计算最优扩展路径。实验表明，这种对角线扩展方法相比传统方法能显著降低延迟、成本和重新平衡开销。

### 背景

现代云数据库将扩展视为二元决策：要么通过添加节点进行水平扩展，要么通过增加单节点资源进行垂直扩展。这种一维视图具有局限性，因为数据库性能、成本和协调开销来自于水平弹性和单节点资源的联合交互，导致系统反应过度或不足，或在次优状态间振荡。

### 目的

解决当前云数据库扩展模型的一维局限性，提出一个更全面的二维扩展模型，综合考虑水平扩展和垂直扩展的联合效应，实现更优的性能、成本和协调开销平衡。

### 方法

提出'Scaling Plane'模型，将分布式数据库配置表示为点(H, V)，H表示节点数量，V表示资源向量。定义延迟、吞吐量、协调开销和成本的平滑近似，提供性能权衡的统一视图。开发了DIAGONALSCALE算法，评估水平、垂直和对角线移动，选择最小化多目标函数且满足SLA约束的配置。

### 主要发现

最优扩展轨迹通常沿对角线路径，同时利用集群并行性和单节点改进。与仅水平或仅垂直扩展相比，对角线扩展可将p95延迟降低高达40%，将每查询成本降低高达37%，并将重新平衡减少2至5倍。

### 结论

云数据库系统需要多维扩展模型，这种综合考虑水平和垂直扩展的方法为下一代自动扩展提供了基础，能够实现更优的性能和成本效益。

### 翻译

现代云数据库将扩展呈现为二元决策：通过添加节点进行水平扩展或通过增加单节点资源进行垂直扩展。这种一维观点具有局限性，因为数据库性能、成本和协调开销来自于水平弹性和单节点CPU、内存、网络带宽以及存储IOPS的联合交互。因此，系统经常对负载峰值反应过度，对内存压力反应不足，或在次优状态之间振荡。我们引入了扩展平面，一个二维模型，其中每个分布式数据库配置表示为一个点(H, V)，H表示节点数量，V表示资源向量。在这个平面上，我们定义了延迟、吞吐量、协调开销和货币成本的平滑近似，提供了性能权衡的统一视图。我们通过分析和经验证明，最优扩展轨迹通常沿对角线路径：同时利用集群并行性和单节点改进的水平与垂直调整序列。为了计算此类操作，我们提出了DIAGONALSCALE，一种离散局部搜索算法，它在扩展平面中评估水平、垂直和对角线移动，并选择最小化多目标函数且满足SLA约束的配置。使用合成表面、微基准测试以及在分布式SQL和键值系统上的实验，我们证明对角线扩展可将p95延迟降低高达40%，将每查询成本降低高达37%，并将重新平衡减少2至5倍，相比仅水平和仅垂直自动扩展。我们的结果突显了多维扩展模型的需求，并为云数据库系统的下一代自动扩展提供了基础。


### 论文摘要

Modern cloud databases present scaling as a binary decision: scale-out by adding nodes or scale-up by increasing per-node resources. This one-dimensional view is limiting because database performance, cost, and coordination overhead emerge from the joint interaction of horizontal elasticity and per-node CPU, memory, network bandwidth, and storage IOPS. As a result, systems often overreact to load spikes, underreact to memory pressure, or oscillate between suboptimal states. We introduce the Scaling Plane, a two-dimensional model in which each distributed database configuration is represented as a point (H, V), with H denoting node count and V a vector of resources. Over this plane, we define smooth approximations of latency, throughput, coordination overhead, and monetary cost, providing a unified view of performance trade-offs. We show analytically and empirically that optimal scaling trajectories frequently lie along diagonal paths: sequences of joint horizontal and vertical adjustments that simultaneously exploit cluster parallelism and per-node improvements. To compute such actions, we propose DIAGONALSCALE, a discrete local-search algorithm that evaluates horizontal, vertical, and diagonal moves in the Scaling Plane and selects the configuration minimizing a multi-objective function subject to SLA constraints. Using synthetic surfaces, microbenchmarks, and experiments on distributed SQL and KV systems, we demonstrate that diagonal scaling reduces p95 latency by up to 40 percent, lowers cost-per-query by up to 37 percent, and reduces rebalancing by 2 to 5 times compared to horizontal-only and vertical-only autoscaling. Our results highlight the need for multi-dimensional scaling models and provide a foundation for next-generation autoscaling in cloud database systems.

---

## 33. E-M3RF: An Equivariant Multimodal 3D Re-assembly Framework

**论文链接:** [http://arxiv.org/abs/2511.21422v1](http://arxiv.org/abs/2511.21422v1)

**作者:** Adeela Islam, Stefano Fiorini, Manuel Lecha, Theodore Tsesmelis, Stuart James, Pietro Morerio, Alessio Del Bue

**发布时间:** 2025-11-26

### GPT解析

### 总结

E-M3RF是一个等变多模态3D重组框架，结合几何和颜色特征，通过SE(3)流匹配预测碎片重组所需的变换，在多个数据集上表现出色。

### 背景

3D重组是基础几何问题，近年深度学习方法逐渐取代传统优化方法，但现有方法主要依赖几何特征，在几何信息不足或模糊时表现不佳，且缺乏防止重叠的物理约束。

### 目的

解决现有3D重组方法在几何信息不足或模糊情况下的局限性，并提出施加物理约束的解决方案。

### 方法

E-M3RF框架输入包含点位置和颜色的点云数据，使用SE(3)流匹配预测重组变换。每个碎片由几何特征（通过旋转等变编码器编码）和颜色特征（使用transformer编码）表示，两者结合形成多模态表示。

### 主要发现

在RePAIR数据集上，E-M3RF相比竞争方法，旋转误差减少23.1%，平移误差减少13.2%，Chamfer距离减少18.4%，证明了其有效性。

### 结论

E-M3RF通过结合几何和颜色特征，并施加物理约束，有效解决了传统3D重组方法在处理小型、腐蚀或对称碎片时的局限性。

### 翻译

3D重组是一个基础的几何问题，近年来，深度学习方法日益取代传统优化方法。虽然学习方法已显示出 promising 的结果，但大多数仍主要依赖几何特征将碎片重组为整体。因此，当几何信息不足或模糊时，方法表现不佳，例如对于小型、腐蚀或对称的碎片。此外，现有解决方案没有施加明确防止重叠的物理约束。为解决这些局限性，我们引入了E-M3RF，一个等变多模态3D重组框架，它将包含断裂碎片点位置和颜色的点云作为输入，并使用SE(3)流匹配预测重组所需的变换。每个碎片由几何和颜色特征表示：i) 3D点位置通过旋转等变编码器编码为旋转一致的几何特征；ii) 每个3D点的颜色使用transformer编码。然后将两个特征集结合形成多模态表示。我们在四个数据集上进行了实验：两个合成数据集Breaking Bad和Fantastic Breaks，以及两个真实文化遗产数据集RePAIR和Presious，结果表明在RePAIR数据集上，E-M3RF将旋转误差减少23.1%，平移误差减少13.2%，Chamfer距离减少18.4%，优于竞争方法。


### 论文摘要

3D reassembly is a fundamental geometric problem, and in recent years it has increasingly been challenged by deep learning methods rather than classical optimization. While learning approaches have shown promising results, most still rely primarily on geometric features to assemble a whole from its parts. As a result, methods struggle when geometry alone is insufficient or ambiguous, for example, for small, eroded, or symmetric fragments. Additionally, solutions do not impose physical constraints that explicitly prevent overlapping assemblies. To address these limitations, we introduce E-M3RF, an equivariant multimodal 3D reassembly framework that takes as input the point clouds, containing both point positions and colors of fractured fragments, and predicts the transformations required to reassemble them using SE(3) flow matching. Each fragment is represented by both geometric and color features: i) 3D point positions are encoded as rotationconsistent geometric features using a rotation-equivariant encoder, ii) the colors at each 3D point are encoded with a transformer. The two feature sets are then combined to form a multimodal representation. We experimented on four datasets: two synthetic datasets, Breaking Bad and Fantastic Breaks, and two real-world cultural heritage datasets, RePAIR and Presious, demonstrating that E-M3RF on the RePAIR dataset reduces rotation error by 23.1% and translation error by 13.2%, while Chamfer Distance decreases by 18.4% compared to competing methods.

---

## 34. 论文ID: 2511.21374v1

**错误:** 无法读取论文信息 - [Errno 2] No such file or directory: 'storage/result/2511.21374v1.json'

---

## 35. PFF-Net: Patch Feature Fitting for Point Cloud Normal Estimation

**论文链接:** [http://arxiv.org/abs/2511.21365v1](http://arxiv.org/abs/2511.21365v1)

**作者:** Qing Li, Huifang Feng, Kanle Shi, Yue Gao, Yi Fang, Yu-Shen Liu, Zhizhong Han

**发布时间:** 2025-11-26

**备注:** Accepted by TVCG

### GPT解析

### 总结

本文提出了一种基于多尺度特征融合的点云法线估计方法，通过补丁特征拟合(PFF)模型解决不同数据或几何体中邻域大小选择困难的问题。

### 背景

估计点的法线需要构建局部补丁提供上下文信息，但确定合适的邻域大小具有挑战性。现有方法虽然采用各种参数密集策略提取特征，但在准确高效预测各种点云法线方面仍有局限。

### 目的

开发一种新的特征提取方法，实现鲁棒的点云法线估计，解决不同几何体和数据类型的补丁大小选择问题。

### 方法

采用多尺度特征融合策略，通过补丁特征拟合(PFF)模型近似最佳几何描述。包含特征聚合模块(渐进聚合不同尺度补丁特征到中心点)和特征补偿模块(确保大尺度特征的再利用和不同补丁尺寸的相关信息揭示)。

### 主要发现

多尺度特征聚合的近似策略使模型能够实现不同局部补丁的尺度自适应，并提供最佳特征描述，在保持高性能的同时减少了网络参数和运行时间。

### 结论

该方法在合成和真实数据集上实现了最先进的性能，具有更少的网络参数和更快的运行速度，有效解决了点云法线估计中的邻域选择难题。

### 翻译

估计点的法线需要构建局部补丁来提供中心-周围上下文，但在处理不同数据或几何体时确定适当的邻域大小很困难。现有方法通常采用各种参数密集的策略从输入补丁中提取完整的特征描述，但在准确高效地预测各种点云的法线方面仍有困难。在这项工作中，我们提出了用于鲁棒点云法线估计的新特征提取思路。我们使用来自不同邻域大小的多尺度特征融合来解决选择各种数据或几何体合理补丁大小的问题。我们基于多尺度特征构建补丁特征拟合(PFF)来近似法线估计的最佳几何描述，并通过多尺度特征聚合和跨尺度特征补偿实现近似过程。特征聚合模块渐进地将不同尺度的补丁特征聚合到补丁中心，并通过移除远离中心的点来缩小补丁大小。它不仅使网络能够精确捕获大范围的结构特征，还能描述高度详细的几何形状。特征补偿模块确保了大尺度早期特征的再利用，并揭示了不同补丁尺寸中的相关信息。我们基于多尺度特征聚合的近似策略使模型能够实现不同局部补丁的尺度自适应并提供最佳特征描述。大量实验证明，我们的方法在合成和真实数据集上都实现了最先进的性能，且网络参数更少，运行时间更短。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决点云法线估计中的邻域大小选择问题。当处理不同数据或几何形状时，难以确定构建局部块时应该使用多大的邻域范围。邻域太小无法提供足够的邻近点信息，邻域太大则会引入冗余信息或弱化细节特征。这个问题很重要，因为点云法线估计是3D计算机视觉的基础任务，直接影响表面重建、图形渲染和点云去噪等众多下游应用的质量。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先通过实验分析了不同邻域大小对法线估计的影响，发现对于噪声数据需要较大邻域，对于平滑平面适合较小邻域，而对于高曲率结构则有效点集中在查询点附近。他们借鉴了泰勒展开理论，认为可以通过多尺度特征融合来解决邻域选择问题。作者参考了表面拟合方法、多尺度网络和多分支网络的思想，但指出这些方法存在特征学习能力有限或参数过多的问题。基于这些观察，他们设计了多尺度特征聚合和跨尺度特征补偿的架构。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是基于多尺度特征的块特征拟合（PFF），通过融合不同邻域大小的特征来逼近法线估计的最佳几何描述，使模型能适应不同几何形状。整体流程包括：1）点特征提取，为每个点学习点特征和局部邻域特征；2）多尺度特征聚合，使用两种不同层构建F1和F2两个块，F1逐渐减少块大小过滤冗余信息，F2进一步细化特征；3）跨尺度补偿，使用注意力机制权衡不同尺度特征；4）法线预测，通过加权最大池化预测查询点法线。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1）提出基于多尺度特征的块特征拟合策略；2）设计多尺度特征聚合模块，逐渐将不同尺度特征聚合到块中心并减小块大小；3）设计跨尺度特征补偿模块，确保大尺度特征可重用性；4）实现不同局部块的尺度自适应。相比之前工作，不同之处在于：不是直接拟合3D表面而是在特征空间中进行更高阶逼近；不是平等处理所有输入点而是更关注接近中心的点；通过多尺度融合和补偿实现更有效的几何描述；相比多尺度网络有更强学习能力，相比多分支网络参数更少效率更高。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '这篇论文提出了一种基于多尺度特征融合的点云法线估计方法，通过块特征拟合策略实现了对不同几何形状的自适应处理，在保持高精度的同时显著提高了效率和泛化能力。'}


### 论文摘要

Estimating the normal of a point requires constructing a local patch to provide center-surrounding context, but determining the appropriate neighborhood size is difficult when dealing with different data or geometries. Existing methods commonly employ various parameter-heavy strategies to extract a full feature description from the input patch. However, they still have difficulties in accurately and efficiently predicting normals for various point clouds. In this work, we present a new idea of feature extraction for robust normal estimation of point clouds. We use the fusion of multi-scale features from different neighborhood sizes to address the issue of selecting reasonable patch sizes for various data or geometries. We seek to model a patch feature fitting (PFF) based on multi-scale features to approximate the optimal geometric description for normal estimation and implement the approximation process via multi-scale feature aggregation and cross-scale feature compensation. The feature aggregation module progressively aggregates the patch features of different scales to the center of the patch and shrinks the patch size by removing points far from the center. It not only enables the network to precisely capture the structure characteristic in a wide range, but also describes highly detailed geometries. The feature compensation module ensures the reusability of features from earlier layers of large scales and reveals associated information in different patch sizes. Our approximation strategy based on aggregating the features of multiple scales enables the model to achieve scale adaptation of varying local patches and deliver the optimal feature description. Extensive experiments demonstrate that our method achieves state-of-the-art performance on both synthetic and real-world datasets with fewer network parameters and running time.

---

## 36. LaGen: Towards Autoregressive LiDAR Scene Generation

**论文链接:** [http://arxiv.org/abs/2511.21256v1](http://arxiv.org/abs/2511.21256v1)

**作者:** Sizhuo Zhou, Xiaosong Jia, Fanrui Zhang, Junjie Li, Juyong Zhang, Yukang Feng, Jianwen Sun, Songbur Wong, Junqi You, Junchi Yan

**发布时间:** 2025-11-26

### GPT解析

### 总结

本文提出LaGen框架，实现了长距离激光雷达场景的逐帧自回归生成，解决了现有方法无法支持长距离交互式生成的问题。

### 背景

生成式世界模型在自动驾驶领域成为热门话题，但现有研究多集中在图像模态，而激光雷达数据的生成式模型研究较少。

### 目的

开发一种能够支持长距离交互式生成的激光雷达数据生成模型，解决现有生成方法和预测方法的局限性。

### 方法

提出LaGen框架，接受单帧激光雷达输入作为起点，利用边界框信息作为条件生成高保真4D场景点云；引入场景解耦估计模块增强对象级内容的交互生成能力；引入噪声调制模块减轻长距离生成过程中的误差累积。

### 主要发现

实验结果表明LaGen优于最先进的激光雷达生成和预测模型，特别是在后续帧的生成上表现更佳。

### 结论

LaGen是首个能够实现长距离激光雷达场景逐帧自回归生成的框架，有效解决了现有方法在长距离交互式生成方面的局限性。

### 翻译

自动驾驶的生成式世界模型已成为一个热门话题。与广泛研究的图像模态不同，本文探索了激光雷达数据的生成式世界模型。现有的激光雷达数据生成方法仅支持单帧生成，而现有的预测方法需要多帧历史输入，且只能一次性确定性预测多帧，缺乏交互性。这两种范式都不支持长距离的交互式生成。为此，我们引入了LaGen，据我们所知，这是第一个能够实现长距离激光雷达场景逐帧自回归生成的框架。LaGen能够以单帧激光雷达输入为起点，有效利用边界框信息作为条件生成高保真4D场景点云。此外，我们引入了场景解耦估计模块来增强模型对对象级内容的交互生成能力，以及一个噪声调制模块来减轻长距离生成过程中的误差累积。我们基于nuScenes构建了一个用于评估长距离激光雷达场景生成的协议。实验结果全面证明LaGen优于最先进的激光雷达生成和预测模型，特别是在后续帧上表现更佳。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决的是现有LiDAR数据生成方法无法生成连续场景，以及现有预测方法需要多帧历史输入且缺乏交互性的问题。在自动驾驶领域，这个问题非常重要，因为自动驾驶需要时间上连续的LiDAR场景片段，而闭环模拟中通常只有初始帧可用。此外，当前方法只能沿固定轨迹预测，无法根据不同决策生成多样化轨迹，也不能准确预测长时间范围内的LiDAR数据。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先认识到LiDAR数据在自动驾驶中的重要性以及现有方法的局限性，然后决定探索基于LiDAR的生成世界模型。他们借鉴了现有工作中的球形投影技术，将三维LiDAR数据转换为二维范围图像以提高计算效率。同时，他们基于潜在扩散模型构建生成器，并创新性地设计了场景解耦估计模块和噪声调制模块，分别用于增强交互生成能力和减轻长距离生成中的误差累积。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是创建一个能够逐帧自回归生成长距离LiDAR场景的框架，利用单帧输入作为起点，结合边界框信息生成高质量4D场景点云。整体流程包括：首先将三维LiDAR数据通过球形投影转换为二维范围图像；然后基于潜在扩散模型构建生成器，通过多个控制条件引导扩散过程；接着使用场景解耦估计模块提供实时对象级估计；再利用噪声调制模块减轻误差累积；最后通过自回归方式逐帧生成LiDAR场景，每帧生成时只依赖前一帧的信息。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1) 提出了LaGen框架，首次实现逐帧自回归生成长距离LiDAR场景；2) 设计了场景解耦估计模块，增强交互生成能力；3) 设计了噪声调制模块，减轻长距离生成中的误差累积；4) 构建了基于nuScenes的评估协议。相比之前的工作，LaGen不需要多帧历史输入，只需单帧即可生成；能够根据不同决策生成多样化轨迹；支持交互式生成，可在每个时间步融入决策；能够生成长时间范围内的高质量LiDAR数据。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'LaGen首次实现了基于单帧输入的自回归长距离LiDAR场景生成，通过创新的场景解耦估计和噪声调制模块，解决了自动驾驶中长距离场景生成和交互式模拟的关键挑战。'}


### 论文摘要

Generative world models for autonomous driving (AD) have become a trending topic. Unlike the widely studied image modality, in this work we explore generative world models for LiDAR data. Existing generation methods for LiDAR data only support single frame generation, while existing prediction approaches require multiple frames of historical input and can only deterministically predict multiple frames at once, lacking interactivity. Both paradigms fail to support long-horizon interactive generation. To this end, we introduce LaGen, which to the best of our knowledge is the first framework capable of frame-by-frame autoregressive generation of long-horizon LiDAR scenes. LaGen is able to take a single-frame LiDAR input as a starting point and effectively utilize bounding box information as conditions to generate high-fidelity 4D scene point clouds. In addition, we introduce a scene decoupling estimation module to enhance the model's interactive generation capability for object-level content, as well as a noise modulation module to mitigate error accumulation during long-horizon generation. We construct a protocol based on nuScenes for evaluating long-horizon LiDAR scene generation. Experimental results comprehensively demonstrate LaGen outperforms state-of-the-art LiDAR generation and prediction models, especially on the later frames.

---

## 37. Scenes as Tokens: Multi-Scale Normal Distributions Transform Tokenizer for General 3D Vision-Language Understanding

**论文链接:** [http://arxiv.org/abs/2511.21191v1](http://arxiv.org/abs/2511.21191v1)

**作者:** Yutao Tang, Cheng Zhao, Gaurav Mittal, Rohith Kukkala, Rama Chellappa, Cheng Peng, Mei Chen

**发布时间:** 2025-11-26

### GPT解析

### 总结

NDTokenizer3D是一种通用的3D视觉语言模型，通过新颖的三阶段场景标记化流程和多尺度NDT解码器，实现了广泛的3D场景理解任务，并支持人类交互。

### 背景

3D视觉语言模型在3D场景理解和推理方面具有巨大潜力，但将3D场景有效标记化为整体场景标记，并在各种3D理解任务中利用这些标记仍面临重大挑战。

### 目的

提出NDTokenizer3D，一种通用3D视觉语言模型，能够执行多种3D场景理解任务，同时自然支持人类交互，弥合语言级推理与3D空间理解之间的差距。

### 方法

核心是基于多尺度正态分布变换表示构建的三阶段场景标记化流程，配有多尺度NDT解码器。首先从原始高分辨率点云构建多尺度NDT表示，保留全局上下文和细粒度几何细节；然后MSDec逐步融合跨尺度NDT特征，生成整体场景标记；最后将MSDec重新用作人类交互提示和分割掩码解码的通用接口。

### 主要发现

通过紧凑和统一的设计，NDTokenizer3D提供了细粒度、通用的3D视觉语言模型，在3D指代表分割、3D视觉问答和3D密集描述等任务中取得了显著改进。

### 结论

NDTokenizer3D是一个有效的统一架构，能够处理多种3D场景理解任务，并与人类交互自然结合。

### 翻译

最近的3D视觉语言模型进展突显了3D场景理解和推理的巨大潜力。然而，有效地将3D场景标记化为整体场景标记，并在各种3D理解任务中利用这些标记仍然极具挑战性。我们提出了NDTokenizer3D，一种通用3D视觉语言模型，能够执行广泛的3D场景理解任务，同时自然支持人类交互，从而弥合语言级推理与3D空间理解之间的差距。我们方法的核心是一种新颖的三阶段场景标记化流程，基于多尺度正态分布变换表示，配有多尺度NDT解码器。具体而言，NDTokenizer3D首先从原始高分辨率点云构建多尺度NDT表示，保留全局上下文和细粒度几何细节。接下来，MSDec逐步融合跨尺度NDT特征，生成大语言模型端点可消费的整体场景标记。除了标记化，MSDec还被重新用作人类交互提示（点、框、掩码）和分割掩码解码的通用接口，在单一架构中统一了多种3D场景理解任务。凭借这种紧凑和统一的设计，NDTokenizer3D提供了细粒度、通用的3D视觉语言模型，在3D指代表分割、3D视觉问答和3D密集描述等任务中取得了显著改进。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决如何有效地将3D场景tokenize（标记化）为信息丰富的表示，以便在统一框架内支持多种3D视觉-语言理解任务。这个问题在现实和研究中非常重要，因为3D视觉-语言理解是自动驾驶、具身AI和3D AR/VR等应用的关键技术，这些应用需要强大的推理能力、细粒度的3D场景理解和无缝的人-智能体交互。目前的方法要么是为特定任务设计的单任务模型，要么需要任务特定的模块或微调，限制了它们的灵活性和可扩展性。此外，现有的3D tokenization方法通常通过下采样处理原始点云，牺牲了细粒度的3D几何细节，且缺乏有效的机制来捕获抽象的全局结构。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先认识到3D场景tokenization需要同时解决两个关键问题：如何保留局部几何细节和如何捕获全局上下文信息。他们借鉴了Normal Distributions Transform (NDT)表示法，这是一种最初用于同步定位与地图构建(SLAM)的技术。NDT将3D点云划分为均匀网格，并将每个单元的局部表面建模为高斯分布，这种基于网格的公式自然支持多分辨率表示。作者在此基础上创新地设计了一个三阶段的场景tokenization流程，开发了多尺度NDT解码器(MSDec)用于跨尺度融合特征，并将MSDec重新设计为多用途接口支持用户交互提示和分割掩码解码。训练策略也分为两个阶段：首先预训练3D编码器和MSDec，然后进行指令微调，结合了3D实例分割任务和2D视觉-语言监督。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是使用多尺度Normal Distributions Transform (NDT)表示法来同时保留3D场景中的局部几何细节和全局上下文信息，并通过三阶段的场景tokenization流程将复杂的3D场景转换为信息丰富的场景tokens，同时将多尺度NDT解码器(MSDec)设计为多用途接口，支持用户交互提示和分割掩码解码，统一多种3D理解任务。整体实现流程包括：第一阶段构建多尺度NDT表示，将高分辨率3D点云划分为网格并计算每个单元的高斯统计量；第二阶段使用3D编码器提取多尺度特征，粗尺度特征强调全局上下文，细尺度特征编码几何细节；第三阶段通过MSDec逐步融合不同尺度特征生成整体场景表示；最后将场景tokens与文本指令输入大语言模型进行任务特定输出，同时MSDec处理用户交互提示和分割解码。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1) 三阶段场景tokenization流程，使用多尺度NDT表示法同时保留局部几何细节和全局上下文；2) 多尺度NDT解码器(MSDec)，不仅用于tokenization，还作为统一接口支持用户交互提示和分割解码；3) 统一的交互式框架，支持用户输入的视觉提示和3D分割掩码生成。相比之前的工作，不同之处在于：传统方法通常使用下采样导致几何细节丢失，而NDT通过高斯统计量保留原始点信息；大多数现有方法要么将3D场景视为单尺度实体，要么使用物体实例表示，而NDT捕获多尺度关系；传统方法通常是单任务模型或需要任务特定模块，而NDTokenizer3D通过统一框架支持多种任务；大多数3D VLM缺乏交互能力，而NDTokenizer3D提供交互式提示功能。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'NDTokenizer3D通过创新的多尺度Normal Distributions Transform tokenization方法和统一的交互式框架，实现了在单一架构中高效处理多种3D场景理解任务的能力，同时保留了局部几何细节和全局上下文信息，显著提升了3D视觉-语言模型的性能和通用性。'}


### 论文摘要

Recent advances in 3D vision-language models (VLMs) highlight a strong potential for 3D scene understanding and reasoning. However, effectively tokenizing 3D scenes into holistic scene tokens, and leveraging these tokens across diverse 3D understanding tasks, remain highly challenging. We present NDTokenizer3D, a generalist 3D VLM that performs a wide range of 3D scene understanding tasks while naturally supporting human interactions, thereby bridging language-level reasoning with 3D spatial understanding. The core of our approach is a novel three-stage scene tokenization pipeline built upon a Multi-Scale Normal Distributions Transform (NDT) representation, paired with a Multi-Scale NDT Decoder (MSDec). Specifically, NDTokenizer3D first constructs a multi-scale NDT representation from raw high-resolution point clouds, preserving both global context and fine-grained geometric details. Next, the MSDec progressively fuses cross-scale NDT features, producing holistic scene tokens consumable by LLM endpoints. Beyond tokenization, MSDec is repurposed as a general interface for human-interactive prompting (points, boxes, masks) and segmentation-mask decoding, unifying diverse 3D scene understanding tasks within a single architecture. With this compact and unified design, NDTokenizer3D offers a fine-grained, general-purpose 3D VLM, achieving remarkable improvements in 3D Referring Segmentation, 3D Visual Question Answering, and 3D Dense Captioning.

---

## 38. $δ$-core subsampling, strong collapses and TDA

**论文链接:** [http://arxiv.org/abs/2511.20954v1](http://arxiv.org/abs/2511.20954v1)

**作者:** Elias Gabriel Minian

**发布时间:** 2025-11-26

**备注:** 14 pages, 8 figures, 5 tables

### GPT解析

### 总结

提出了一种基于单纯复形强坍缩的拓扑数据分析子采样方法，能够在保持全局和局部拓扑特征的同时显著降低持续同态计算的计算复杂度。

### 背景

拓扑数据分析中需要处理大量点云数据，计算复杂度高。

### 目的

开发一种子采样方法，既能保持拓扑特征又能降低计算复杂度。

### 方法

基于单纯复形强坍缩的子采样方法，给定点云和尺度参数δ进行构建。

### 主要发现

该方法能有效保持全局和局部拓扑特征，显著降低持续同态计算的计算复杂度，并在持续同态近似方面优于其他子采样技术。

### 结论

该方法在合成和真实数据集上均表现出色，是一种有效的拓扑数据分析子采样方法。

### 翻译

我们介绍了一种基于单纯复形强坍缩的拓扑数据分析子采样方法。给定一个点云和一个尺度参数δ，我们构建了一种子采样方法，既能保持全局和局部拓扑特征，又能显著降低持续同态计算的计算复杂度。我们通过在合成和真实数据集上的实验展示了该方法的有效性，与其他子采样技术相比，显示出改进的持续同态近似。


### 论文摘要

We introduce a subsampling method for topological data analysis based on strong collapses of simplicial complexes. Given a point cloud and a scale parameter $δ$, we construct a subsampling that preserves both global and local topological features while significantly reducing computational complexity of persistent homology calculations. We illustrate the effectiveness of our approach through experiments on synthetic and real datasets, showing improved persistence approximations compared to other subsampling techniques.

---

## 39. Accelerating Sparse Convolutions in Voxel-Based Point Cloud Networks

**论文链接:** [http://arxiv.org/abs/2511.20834v1](http://arxiv.org/abs/2511.20834v1)

**作者:** Dionysios Adamopoulos, Anastasia Poulopoulou, Georgios Goumas, Christina Giannoula

**发布时间:** 2025-11-25

### GPT解析

### 总结

Spira是一种创新的体素属性感知稀疏卷积引擎，通过利用体素坐标的整数值、有限空间范围和几何连续性等特性，显著提高了GPU上稀疏卷积的性能。

### 背景

稀疏卷积(SpC)被广泛应用于自动驾驶和AR/VR领域的3D点云网络，它通过构建核映射来存储输入体素坐标、输出坐标和权重偏移之间的映射关系，并计算输出坐标的特征向量。然而，先前的SpC引擎没有充分利用体素坐标的关键特性，导致在核映射构建过程中存在较高的预处理和后处理开销。

### 目的

设计一个能够充分利用体素坐标特性的稀疏卷积引擎，以减少预处理和后处理开销，提高GPU上稀疏卷积的执行效率。

### 方法

Spira提出了四个关键创新：(i) 高性能的单次搜索算法，无需预处理即可构建核映射且具有高内存局部性；(ii) 有效的打包原生处理方案，以低成本访问打包的体素坐标；(iii) 灵活的双数据流执行机制，通过适应层特性高效计算输出特征向量；(iv) 网络级并行化策略，在网络启动时同时为所有SpC层构建核映射。

### 主要发现

体素坐标具有三个关键特性：整数值、限制在有限空间范围内、几何连续性（同一物体表面的相邻体素很可能存在于小的空间偏移处）。先前的SpC引擎没有充分利用这些特性，导致性能低下。Spira通过利用这些特性，显著提高了性能。

### 结论

Spira显著优于先前的SpC引擎，在端到端推理方面平均快1.71倍，最高可达2.31倍；在逐层执行方面平均快2.13倍，最高可达3.32倍，适用于各种层配置。

### 翻译

稀疏卷积(SpC)广泛应用于自动驾驶和AR/VR领域的3D点云网络。SpC构建一个核映射，存储输入体素坐标、输出坐标和权重偏移之间的映射关系，然后使用这个映射计算输出坐标的特征向量。我们的研究识别出体素坐标的三个关键特性：它们是整数值的，限制在有限的空间范围内，并且在几何上是连续的——同一物体表面的相邻体素很可能存在于小的空间偏移处。先前的SpC引擎没有充分利用这些特性，在核映射构建过程中存在较高的预处理和后处理开销。为解决这一问题，我们设计了Spira，这是第一个面向GPU的体素属性感知的SpC引擎。Spira提出了：(i) 高性能的单次搜索算法，无需预处理即可构建核映射且具有高内存局部性；(ii) 有效的打包原生处理方案，以低成本访问打包的体素坐标；(iii) 灵活的双数据流执行机制，通过适应层特性高效计算输出特征向量；(iv) 网络级并行化策略，在网络启动时同时为所有SpC层构建核映射。我们的评估显示，Spira在端到端推理方面比先前的SpC引擎平均快1.71倍，最高可达2.31倍；在逐层执行方面平均快2.13倍，最高可达3.32倍，适用于各种层配置。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决稀疏卷积在体素化点云网络中的处理效率问题。这个问题很重要，因为稀疏卷积是3D点云网络的核心计算组件，广泛应用于自动驾驶、AR/VR、机器人和无人机等领域。点云数据极其稀疏（通常只占包围体积的不到0.01%），现有稀疏卷积引擎在预处理和后处理阶段存在显著开销，限制了这些实时应用的性能和响应速度。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先识别了体素坐标数据的三个关键特性：整数性质、有界性和几何连续性。然后分析了现有稀疏卷积引擎的局限性，包括预处理和后处理开销大、未充分利用坐标特性、对两种数据流支持不充分等问题。基于这些观察，设计了Spira系统，包含四个关键创新：一次性搜索算法、打包原生索引、自适应混合数据流和网络范围并行化。作者借鉴了现有稀疏卷积引擎的基本架构和数据流研究，但创新性地利用了体素坐标的结构特性来优化性能。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是充分利用体素坐标数据的三个关键特性来优化稀疏卷积的各个阶段：利用整数性设计高效搜索算法消除预处理，利用有界性将坐标打包减少内存访问，利用几何连续性选择最佳数据流。整体流程包括：1)将坐标打包并排序；2)网络范围并行执行所有层的下采样和映射；3)根据权重偏移密度自适应选择输出静态、权重静态或混合数据流进行特征计算；4)管理核图内存，逐步释放不再需要的部分。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)一次性Z-增量搜索算法，完全消除预处理；2)打包原生体素索引，减少内存占用；3)自适应混合数据流，根据权重偏移密度选择最佳处理方式；4)网络范围并行化，提高GPU利用率。相比之前工作，Spira消除了预处理开销，支持灵活的混合数据流而非仅限一种，通过坐标打包减少内存访问，并实现了层间并行处理而非顺序执行。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'Spira通过创新地利用体素坐标的结构特性，设计了一次性搜索、打包原生索引、自适应混合数据流和网络范围并行化技术，将稀疏卷积速度提升了平均1.71倍，最高达3.32倍，显著推动了点云网络在实时应用中的性能边界。'}


### 论文摘要

Sparse Convolution (SpC) powers 3D point cloud networks widely used in autonomous driving and AR/VR. SpC builds a kernel map that stores mappings between input voxel coordinates, output coordinates, and weight offsets, then uses this map to compute feature vectors for output coordinates. Our work identifies three key properties of voxel coordinates: they are integer-valued, bounded within a limited spatial range, and geometrically continuous-neighboring voxels on the same object surface are highly likely to exist at small spatial offsets from each other. Prior SpC engines do not fully exploit these properties and suffer from high pre-processing and post-processing overheads during kernel map construction. To address this, we design Spira, the first voxel-property-aware SpC engine for GPUs. Spira proposes: (i) a high-performance one-shot search algorithm that builds the kernel map with no preprocessing and high memory locality, (ii) an effective packed-native processing scheme that accesses packed voxel coordinates at low cost, (iii) a flexible dual-dataflow execution mechanism that efficiently computes output feature vectors by adapting to layer characteristics, and (iv) a network-wide parallelization strategy that builds kernel maps for all SpC layers concurrently at network start. Our evaluation shows that Spira significantly outperforms prior SpC engines by 1.71x on average and up to 2.31x for end-to-end inference, and by 2.13x on average and up to 3.32x for layer-wise execution across diverse layer configurations.

---

## 40. VGGTFace: Topologically Consistent Facial Geometry Reconstruction in the Wild

**论文链接:** [http://arxiv.org/abs/2511.20366v2](http://arxiv.org/abs/2511.20366v2)

**作者:** Xin Ming, Yuxuan Han, Tianyu Huang, Feng Xu

**发布时间:** 2025-11-25

### GPT解析

### 总结

本文提出了VGGTFace，一种基于3D基础模型VGGT的自动方法，用于从日常用户拍摄的野外多视角图像中重建拓扑一致的面部几何形状。

### 背景

重建拓扑一致的面部几何形状对于数字化身创建流程至关重要。现有方法要么需要繁琐的手动工作，要么难以推广到野外数据，或者受限于3D Morphable Models的有限表达能力。

### 目的

解决现有方法的局限性，开发一种自动化的方法，能够从日常用户拍摄的野外多视角图像中重建拓扑一致的面部几何形状。

### 方法

创新性地应用3D基础模型VGGT进行面部几何重建，通过Pixel3DMM注入拓扑信息将点图转换为具有拓扑的点云，并提出一种拓扑感知的捆绑调整策略，构建拉普拉斯能量作为捆绑调整目标来融合点云。

### 主要发现

该方法在单个NVIDIA RTX 4090上，16个视角只需10秒即可实现高质量重建。在基准测试上达到了最先进的结果，并能很好地推广到野外数据。

### 结论

VGGTFace通过结合VGGT的强大泛化能力和表达能力，以及注入的拓扑信息，解决了现有方法在面部几何重建中的局限性，实现了高效、高质量且具有拓扑一致性的面部重建。

### 翻译

重建拓扑一致的面部几何形状对于数字化身创建流程至关重要。现有方法要么需要繁琐的手动工作，要么难以推广到野外数据，或者受限于3D Morphable Models的有限表达能力。为解决这些局限性，我们提出了VGGTFace，一种自动化的方法，创新性地将3D基础模型即VGGT应用于从日常用户拍摄的野外多视角图像中进行拓扑一致的面部几何重建。我们的关键见解是，通过利用VGGT，我们的方法自然地从其大规模训练和点图表示中继承了强大的泛化能力和表达能力。然而，如何从VGGT重建拓扑一致的网格尚不清楚，因为其预测中缺少拓扑信息。为此，我们通过像素对齐的UV值使用Pixel3DMM增强VGGT，以注入拓扑信息。通过这种方式，我们将VGGT的像素对齐点图转换为具有拓扑的点云。针对这种具有已知拓扑的点云，我们提出了一种新颖的拓扑感知的捆绑调整策略来融合它们，其中我们为捆绑调整目标构建了拉普拉斯能量。我们的方法在单个NVIDIA RTX 4090上，16个视角只需10秒即可实现高质量重建。实验证明了在基准测试上的最先进结果，以及对野外数据的令人印象深刻的泛化能力。代码可在https://github.com/grignarder/vggtface获取。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决如何从日常用户在野外拍摄的多视角图像中自动重建拓扑一致的面部几何结构问题。这个问题在现实中很重要，因为拓扑一致的面部几何对数字头像创建流程至关重要，它 enables dense mesh correspondence 和可转移的动画、绑定和纹理编辑。传统方法需要大量手动工作，难以扩展到日常用户，而自动化的高质量面部重建可以让普通人轻松将自己扫描到数字世界。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先认识到3D基础模型VGGT在几何重建领域的潜力，但发现其预测缺少拓扑信息。他们创新性地结合Pixel3DMM来添加UV坐标信息，从而注入拓扑信息。面对如何融合多视角预测的挑战，他们提出了拓扑感知的束调整策略，利用拉普拉斯能量作为正则化项。该方法借鉴了VGGT的几何预测能力、Pixel3DMM的UV预测能力，以及束调整和拉普拉斯能量等现有技术，但将它们创新性地结合用于面部重建任务。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是利用VGGT的强大泛化能力和表达能力从野外多视角图像中重建面部几何，并通过Pixel3DMM添加拓扑信息，最后使用拓扑感知的束调整策略确保重建结果拓扑一致。整体流程为：1)输入多视角图像；2)用VGGT预测点图和相机参数；3)用Pixel3DMM预测UV坐标；4)根据模板网格UV建立对应关系；5)融合多视角预测生成初始点云；6)应用拓扑感知束调整优化相机参数和点云位置；7)连接优化后的点云生成拓扑一致的面部网格。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)提出VGGTFace方法实现从野外多视角图像自动重建拓扑一致面部几何；2)提出拓扑感知的束调整技术从VGGT原始点云中提取高质量网格；3)在多个基准测试和野外数据上取得最先进结果。相比之前工作，不同之处在于：利用VGGT这一3D基础模型而非专门训练的小模型获得更好泛化能力；使用点图表示而非3DMM具有更强表达能力；创新结合VGGT和Pixel3DMM的优势；提出的拓扑感知束调整能处理低质量跟踪点并确保每个点都有监督；速度更快，仅需10秒即可从16个视角重建高质量网格。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'VGGTFace通过创新性地应用3D基础模型VGGT和提出拓扑感知的束调整策略，实现了从野外多视角图像快速、高质量重建拓扑一致的面部几何，为日常用户提供了将自己扫描到数字世界的实用工具。'}


### 论文摘要

Reconstructing topologically consistent facial geometry is crucial for the digital avatar creation pipelines. Existing methods either require tedious manual efforts, lack generalization to in-the-wild data, or are constrained by the limited expressiveness of 3D Morphable Models. To address these limitations, we propose VGGTFace, an automatic approach that innovatively applies the 3D foundation model, i.e. VGGT, for topologically consistent facial geometry reconstruction from in-the-wild multi-view images captured by everyday users. Our key insight is that, by leveraging VGGT, our method naturally inherits strong generalization ability and expressive power from its large-scale training and point map representation. However, it is unclear how to reconstruct a topologically consistent mesh from VGGT, as the topology information is missing in its prediction. To this end, we augment VGGT with Pixel3DMM for injecting topology information via pixel-aligned UV values. In this manner, we convert the pixel-aligned point map of VGGT to a point cloud with topology. Tailored to this point cloud with known topology, we propose a novel Topology-Aware Bundle Adjustment strategy to fuse them, where we construct a Laplacian energy for the Bundle Adjustment objective. Our method achieves high-quality reconstruction in 10 seconds for 16 views on a single NVIDIA RTX 4090. Experiments demonstrate state-of-the-art results on benchmarks and impressive generalization to in-the-wild data. Code is available at https://github.com/grignarder/vggtface.

---

## 41. Foundry: Distilling 3D Foundation Models for the Edge

**论文链接:** [http://arxiv.org/abs/2511.20721v1](http://arxiv.org/abs/2511.20721v1)

**作者:** Guillaume Letellier, Siddharth Srivastava, Frédéric Jurie, Gaurav Sharma

**发布时间:** 2025-11-25

### GPT解析

### 总结

本文提出了一种名为基础模型蒸馏（FMD）的新范式，用于压缩大型自监督学习模型，同时保持其通用表征能力。作者实现了Foundry，这是FMD首次针对3D点云的应用，通过训练学生模型学习压缩的SuperTokens来重建教师模型的token级表征，从而在保持高性能的同时显著减少计算需求。

### 背景

基础模型通过在大规模数据集上进行自监督学习预训练，已成为强大的通用特征提取器。然而，这些模型体积庞大且计算成本高，难以在边缘设备上部署。现有压缩技术如标准知识蒸馏会牺牲基础模型的关键通用性。

### 目的

引入FMD压缩范式，将大型SSL模型压缩为紧凑、高效且忠实的代理模型，同时保留其通用表征能力。实现Foundry，这是FMD首次针对3D点云的实现。

### 方法

Foundry方法训练学生模型学习一组压缩的SuperTokens，这些SuperTokens能够重建教师的token级表征，捕捉教师潜在空间的紧凑基。使用更少的token和FLOPs，同时保持对多样化下游任务的强可转移性。

### 主要发现

单个蒸馏模型在分类、部分分割和少样本场景等多种下游任务上接近完整基础模型的性能，同时使用显著更少的token和FLOPs，使其在资源受限硬件上的部署更加实用。

### 结论

FMD是一种有效的基础模型压缩方法，能够在保持通用表征能力的同时，显著减少模型大小和计算需求。Foundry展示了这种方法在边缘设备部署基础模型的潜力。

### 翻译

基础模型通过在大规模数据集上进行自监督学习（SSL）预训练，已成为强大的通用特征提取器。然而，它们的巨大规模和计算成本使得在机器人和AR/VR头显等边缘设备上部署变得不切实际。现有的压缩技术如标准知识蒸馏能够创建高效的'专家'模型，但牺牲了使基础模型如此宝贵的关键的、与下游任务无关的通用性。在本文中，我们引入了基础模型蒸馏（FMD），这是一种新的压缩范式，用于将大型SSL模型压缩为紧凑、高效且忠实的代理模型，同时保留其通用表征能力。我们提出了Foundry，这是FMD针对3D点云的首次实现。我们的方法Foundry训练一个学生模型，学习一组压缩的SuperTokens，这些SuperTokens能够重建教师的token级表征，捕捉其潜在空间的紧凑基。单个蒸馏模型在分类、部分分割和少样本场景等多样化下游任务上保持强可转移性，接近完整基础模型的性能，同时使用显著更少的token和FLOPs，使得这些模型在资源受限硬件上的部署更加实用。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决大型3D基础模型（如PointTransformer）因其巨大规模和计算成本而无法在边缘设备（如机器人、AR/VR头显等）上部署的问题。这个问题很重要，因为基础模型在3D视觉领域（如机器人、自动驾驶、AR/VR）非常有用，但计算障碍阻止了它们到达最需要它们的设备上，限制了边缘智能的发展。即使现代GPU也可能无法处理中等大小的点云（30万个点），更不用说实际应用中常见的百万点场景。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': "作者首先分析了现有压缩技术的局限性，指出它们只能创建特定任务的'专家'模型而牺牲了基础模型的通用性。因此，他们提出了'基础模型蒸馏'（FMD）新范式，设计了'SuperTokens'概念，训练轻量级学生模型将教师模型的密集标记压缩成少量SuperTokens，再重建原始表示。作者借鉴了知识蒸馏的基本思想，以及3DLST中的SuperToken概念，但应用目的不同；还参考了ToMe和PiToMe等Token合并技术，但Foundry将其用于离线蒸馏而非在线加速。", '这个方法的核心思想是什么？整体实现流程是怎样的？': "核心思想是通过'基础模型蒸馏'（FMD）将大型基础模型的知识转移到紧凑学生模型中，同时保持其通用特征提取能力。使用'SuperTokens'作为教师潜在空间的紧凑基础，让学生学习从这些压缩的SuperTokens重建教师的完整表示。实现流程分三步：1）教师前向传播：输入点云经教师模型产生目标表示Y；2）学生前向传播：通过DSO模块将输入压缩为SuperTokens，再经学生编码器和CAU模块重建表示Ŷ；3）蒸馏目标：最小化重建输出Ŷ与教师真实输出Y之间的Smooth L1损失。", '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1）提出FMD新范式，压缩大型模型同时保持通用性；2）引入可学习的SuperTokens作为潜在空间紧凑基础；3）使用压缩-重建目标而非传统特征模仿；4）首次将FMD应用于3D点云Transformers；5）通过门控机制实现预算感知推理。相比之前工作，传统KD创建任务特定专家而非通用模型；CLIP-KD等使用直接特征模仿而非压缩重建；ToMe等关注在线加速而非创建新独立模型；静态方法（如K-Means）无法端到端学习SuperTokens基础。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'Foundry通过引入基础模型蒸馏新范式和可学习的SuperTokens机制，首次将大型3D基础模型压缩为高效且保持通用性的代理，使其能够在资源受限的边缘设备上部署，同时接近原始模型的性能。'}


### 论文摘要

Foundation models pre-trained with self-supervised learning (SSL) on large-scale datasets have become powerful general-purpose feature extractors. However, their immense size and computational cost make them prohibitive for deployment on edge devices such as robots and AR/VR headsets. Existing compression techniques like standard knowledge distillation create efficient 'specialist' models but sacrifice the crucial, downstream-agnostic generality that makes foundation models so valuable.  In this paper, we introduce Foundation Model Distillation (FMD), a new paradigm for compressing large SSL models into compact, efficient, and faithful proxies that retain their general-purpose representational power. We present Foundry, the first implementation of FMD for 3D point clouds. Our approach, Foundry, trains a student to learn a compressed set of SuperTokens that reconstruct the teacher's token-level representations, capturing a compact basis of its latent space. A single distilled model maintains strong transferability across diverse downstream tasks-classification, part segmentation, and few-shot scenarios-approaching full foundation-model performance while using significantly fewer tokens and FLOPs, making such models more practical for deployment on resourceconstrained hardware.

---

## 42. Multi-Crit: Benchmarking Multimodal Judges on Pluralistic Criteria-Following

**论文链接:** [http://arxiv.org/abs/2511.21662v1](http://arxiv.org/abs/2511.21662v1)

**作者:** Tianyi Xiong, Yi Ge, Ming Li, Zuolong Zhang, Pranav Kulkarni, Kaishen Wang, Qi He, Zeying Zhu, Chenxi Liu, Ruibo Chen, Tong Zheng, Yanshuo Chen, Xiyao Wang, Renrui Zhang, Wenhu Chen, Heng Huang

**发布时间:** 2025-11-26

### GPT解析

### 总结

本研究开发了Multi-Crit基准，用于评估多模态评判者遵循多元标准的能力，揭示了当前大型多模态模型在多标准评估方面的局限性。

### 背景

大型多模态模型(LMMs)越来越多地被用作多模态评估系统中的评判者，因其具有强大的指令遵循能力和与人类偏好的一致性，但它们遵循多样化、细粒度评估标准的能力尚未得到充分探索。

### 目的

开发Multi-Crit基准，评估多模态评判者遵循多元标准并产生可靠标准级别判断的能力。

### 方法

构建涵盖开放式生成和可验证推理任务的基准，通过严格的数据整理流程收集具有多标准人工注释的有挑战性响应对，并引入三个新颖指标评估多元遵循能力、标准切换灵活性和识别标准级别偏好冲突的能力，对25个LMMs进行全面分析。

### 主要发现

1)专有模型在保持对多元标准的一致遵循方面存在困难，特别是在开放式评估中；2)开源模型在灵活遵循多样化标准方面进一步落后；3)使用整体判断信号进行批评微调增强了视觉基础，但无法泛化到多元标准级别判断。

### 结论

Multi-Crit作为开创性研究，为构建可靠和可操控的多模态AI评估奠定了基础，并通过进一步分析探索了当前多模态评判者的局限性。

### 翻译

大型多模态模型(LMMs)由于其强大的指令遵循能力和与人类偏好的一致性，越来越多地被采用作为多模态评估系统中的评判者。然而，它们遵循多样化、细粒度评估标准的能力仍未得到充分探索。我们开发了Multi-Crit，一个用于评估多模态评判者遵循多元标准并产生可靠标准级别判断能力的基准。Multi-Crit涵盖了开放式生成和可验证推理任务，通过严格的数据整理流程构建，收集了具有多标准人工注释的有挑战性的响应对。它进一步引入了三个新颖指标，用于系统评估多元遵循能力、标准切换灵活性和识别标准级别偏好冲突的能力。对25个LMMs的全面分析揭示：1)专有模型仍然难以保持对多元标准的一致遵循，特别是在开放式评估中；2)开源模型在灵活遵循多样化标准方面进一步落后；3)使用整体判断信号进行批评微调增强了视觉基础，但无法泛化到多元标准级别判断。对推理微调、测试时扩展以及开源和专有模型之间边界一致性的进一步分析探索了当前多模态评判者的局限性。作为一项开创性研究，Multi-Crit为构建可靠和可操控的多模态AI评估奠定了基础。


### 论文摘要

Large multimodal models (LMMs) are increasingly adopted as judges in multimodal evaluation systems due to their strong instruction following and consistency with human preferences. However, their ability to follow diverse, fine-grained evaluation criteria remains underexplored. We develop Multi-Crit, a benchmark for evaluating multimodal judges on their capacity to follow pluralistic criteria and produce reliable criterion-level judgments. Covering both open-ended generation and verifiable reasoning tasks, Multi-Crit is built through a rigorous data curation pipeline that gathers challenging response pairs with multi-criterion human annotations. It further introduces three novel metrics for systematically assessing pluralistic adherence, criterion-switching flexibility, and the ability to recognize criterion-level preference conflicts. Comprehensive analysis of 25 LMMs reveals that 1) proprietary models still struggle to maintain consistent adherence to pluralistic criteria--especially in open-ended evaluation; 2) open-source models lag further behind in flexibly following diverse criteria; and 3) critic fine-tuning with holistic judgment signals enhances visual grounding but fails to generalize to pluralistic criterion-level judgment. Additional analyses on reasoning fine-tuning, test-time scaling, and boundary consistency between open-source and proprietary models further probe the limits of current multimodal judges. As a pioneering study, Multi-Crit lays the foundation for building reliable and steerable multimodal AI evaluation.

---

## 43. Optimal Bit Detection in Thermal Noise Communication Systems Under Rician Fading

**论文链接:** [http://arxiv.org/abs/2511.21649v1](http://arxiv.org/abs/2511.21649v1)

**作者:** Mohamed El Jbari, Fernando D. A. García, Hugerles S. Silva, Felipe A. P. de Figueiredo, Rausley A. A. de Souza

**发布时间:** 2025-11-26

### GPT解析

### 总结

本文提出了一种在Rician衰落条件下热噪声通信系统中的精确解析框架，用于最优比特检测，显著提高了比特错误性能。

### 背景

现有热噪声通信分析通常依赖高斯近似并忽略衰落效应，限制了准确性；热噪声通信通过调制热噪声方差而非使用主动载波，为物联网设备提供超低功耗无线连接。

### 目的

开发一个精确的解析框架，用于Rician衰落条件下TNC系统中的最优比特检测，消除现有方法的近似误差。

### 方法

使用卡方统计，通过Gauss-Laguerre求导推导出最优的最大似然检测阈值和比特错误概率表达式，并通过蒙特卡洛模拟验证结果。

### 主要发现

提出的模型消除了近似误差，能准确描述有限样本大小的性能；与次优的高斯检测相比，比特错误概率显著改善；量化了样本大小、电阻比和Rician K因子等关键参数的影响。

### 结论

所提出的框架为未来B5G/6G和大规模物联网系统中设计能效高的热噪声通信接收器提供了坚实基础。

### 翻译

热噪声通信(TNC)通过调制热噪声方差而非使用主动载波，为物联网(IoT)设备实现超低功耗无线连接。现有分析通常依赖高斯近似并忽略衰落效应，限制了其准确性。本文提出了Rician衰落条件下TNC系统中最优比特检测的精确解析框架。使用卡方统计，我们通过Gauss-Laguerre求导推导出最优的最大似然检测阈值和比特错误概率(BEP)的表达式。所提出的模型消除了近似误差，并能准确描述有限样本大小的性能。蒙特卡洛模拟证实了分析结果，并与次优的高斯检测相比，BEP有显著改善。此外，量化了关键参数(样本大小、电阻比和Rician K因子)的影响。所提出的框架为未来B5G/6G和大规模物联网系统中设计能效高的TNC接收器提供了坚实基础。


### 论文摘要

Thermal noise communication (TNC) enables ultra-low-power wireless links for Internet of Things (IoT) devices by modulating the variance of thermal noise, rather than using active carriers. Existing analyses often rely on Gaussian approximations and overlook fading effects, which limits their accuracy. This paper presents an accurate analytical framework for optimal bit detection in TNC systems under Rician fading. Using chi-squared statistics, we derive the optimal maximum-likelihood detection threshold and an expression for the bit error probability (BEP) via Gauss-Laguerre quadrature. The proposed model eliminates approximation errors and accurately characterizes performance for finite sample sizes. Monte Carlo simulations confirm the analytical results and demonstrate significant improvements in BEP compared with suboptimal Gaussian-based detection. Furthermore, the influence of key parameters, sample size, resistance ratio, and Rician K-factor, is quantified. The proposed framework provides a solid foundation for designing energy-efficient TNC receivers in future B5G/6G and large-scale IoT systems.

---

## 44. Stochastic Optimal Control of Interacting Particle Systems in Hilbert Spaces and Applications

**论文链接:** [http://arxiv.org/abs/2511.21646v1](http://arxiv.org/abs/2511.21646v1)

**作者:** Filippo de Feo, Fausto Gozzi, Andrzej Święch, Lukas Wessels

**发布时间:** 2025-11-26

### GPT解析

### 总结

研究希尔伯特空间中随机发展方程控制的相互作用粒子系统的最优控制问题，建立了粒子数量趋于无穷时的极限理论，证明了值函数的收敛性和最优控制的对应关系。

### 背景

相互作用粒子的最优控制是希尔伯特空间中随机发展方程的一个开放研究领域，这类系统可由随机偏微分方程、路径依赖随机微分方程（如随机延迟微分方程或随机Volterra积分方程）或部分观测随机系统建模。

### 目的

为粒子数量趋于无穷时的极限理论建立基础，证明有限粒子系统值函数的收敛性，并确定最优控制之间的对应关系。

### 方法

证明有限粒子系统的值函数收敛到平均场Hamilton-Jacobi-Bellman方程的L-粘性解，研究提升极限最优控制问题，分析正则性和最优控制的对应关系。

### 主要发现

证明了值函数u_n收敛到函数V，V是相应方程的L-粘性解；确定了V与提升问题值函数U的关系；在适当假设下证明了U的C^{1,1}正则性；证明了最优控制的对应关系。

### 结论

建立了粒子系统极限理论的基础，证明了值函数收敛性和最优控制对应关系，并将理论应用于经济学中的相关问题。

### 翻译

研究希尔伯特空间中由随机发展方程控制的相互作用粒子的最优控制是一个开放研究领域。这类系统自然出现在每个粒子由随机偏微分方程、路径依赖随机微分方程（如随机延迟微分方程或随机Volterra积分方程）或部分观测随机系统建模的公式中。本文的目的是为粒子数量趋于无穷时的极限理论建立基础。我们证明了有限粒子系统的值函数u_n收敛到一个函数V，V是对应的平均场Hamilton-Jacobi-Bellman方程在概率测度空间中的唯一L-粘性解，并且我们确定了其提升与所谓'提升'极限最优控制问题的值函数U的关系。在适当的附加假设下，我们证明了U的C^{1,1}正则性，证明了V恰好投影到值函数u_n，并且粒子系统的最优（最优反馈）控制对应于在相应初始条件下开始的提升控制问题的最优（最优反馈）控制。据我们所知，这是希尔伯特空间中随机发展方程相互作用粒子系统的随机最优控制问题的首类结果。我们将发展的理论应用于经济学中的问题，其中粒子由随机延迟微分方程和随机偏微分方程建模。


### 论文摘要

Optimal control of interacting particles governed by stochastic evolution equations in Hilbert spaces is an open area of research. Such systems naturally arise in formulations where each particle is modeled by stochastic partial differential equations, path-dependent stochastic differential equations (such as stochastic delay differential equations or stochastic Volterra integral equations), or partially observed stochastic systems. The purpose of this manuscript is to build the foundations for a limiting theory as the number of particles tends to infinity. We prove the convergence of the value functions $u_n$ of finite particle systems to a function $\mathcal{V}$, {which} is the unique {$L$}-viscosity solution of the corresponding mean-field Hamilton-Jacobi-Bellman equation {in the space of probability measures}, and we identify its lift with the value function $U$ of the so-called ``lifted'' limit optimal control problem. Under suitable additional assumptions, we show $C^{1,1}$-regularity of $U$, we prove that $\mathcal{V}$ projects precisely onto the value functions $u_n$, and that optimal (resp. optimal feedback) controls of the particle system correspond to optimal (resp. optimal feedback) controls of the lifted control problem started at the corresponding initial condition. To the best of our knowledge, these are the first results of this kind for stochastic optimal control problems for interacting particle systems of stochastic evolution equations in Hilbert spaces. We apply the developed theory to problems arising in economics where the particles are modeled by stochastic delay differential equations and stochastic partial differential equations.

---

## 45. Qwen3-VL Technical Report

**论文链接:** [http://arxiv.org/abs/2511.21631v1](http://arxiv.org/abs/2511.21631v1)

**作者:** Shuai Bai, Yuxuan Cai, Ruizhe Chen, Keqin Chen, Xionghui Chen, Zesen Cheng, Lianghao Deng, Wei Ding, Chang Gao, Chunjiang Ge, Wenbin Ge, Zhifang Guo, Qidong Huang, Jie Huang, Fei Huang, Binyuan Hui, Shutong Jiang, Zhaohai Li, Mingsheng Li, Mei Li, Kaixin Li, Zicheng Lin, Junyang Lin, Xuejing Liu, Jiawei Liu, Chenglong Liu, Yang Liu, Dayiheng Liu, Shixuan Liu, Dunjie Lu, Ruilin Luo, Chenxu Lv, Rui Men, Lingchen Meng, Xuancheng Ren, Xingzhang Ren, Sibo Song, Yuchong Sun, Jun Tang, Jianhong Tu, Jianqiang Wan, Peng Wang, Pengfei Wang, Qiuyue Wang, Yuxuan Wang, Tianbao Xie, Yiheng Xu, Haiyang Xu, Jin Xu, Zhibo Yang, Mingkun Yang, Jianxin Yang, An Yang, Bowen Yu, Fei Zhang, Hang Zhang, Xi Zhang, Bo Zheng, Humen Zhong, Jingren Zhou, Fan Zhou, Jing Zhou, Yuanzhi Zhu, Ke Zhu

**发布时间:** 2025-11-26

**备注:** 42 pages

### GPT解析

### 总结

Qwen3-VL是Qwen系列迄今为止最强大的视觉语言模型，在多模态基准测试中表现出色，原生支持高达256K token的交错上下文，无缝集成文本、图像和视频。

### 背景

介绍Qwen3-VL作为Qwen系列最新、最强大的视觉语言模型，旨在解决多模态理解和长上下文处理的需求。

### 目的

开发一个在多模态任务中表现出色，同时具备强大文本理解能力和长上下文处理能力的视觉语言模型。

### 方法

采用包括密集型和专家混合型的多种架构变体，引入增强的交错-MRoPE、DeepStack集成和基于文本的时间对齐等架构升级。

### 主要发现

Qwen3-VL在纯文本理解、长上下文理解和多模态推理方面表现出色，特别是在MMMU和视觉数学基准测试上实现了领先性能。

### 结论

Qwen3-VL在可比的token预算和延迟约束下，在密集型和专家混合架构中均实现了更优的性能，有望成为图像推理、智能体决策和多模态代码智能的基础引擎。

### 翻译

我们介绍了Qwen3-VL，这是Qwen系列迄今为止最强大的视觉语言模型，在广泛的多模态基准测试中取得了卓越的性能。它原生支持高达256K token的交错上下文，无缝集成文本、图像和视频。该模型系列包括密集型(2B/4B/8B/32B)和专家混合型(30B-A3B/235B-A22B)变体，以适应不同的延迟-质量权衡。Qwen3-VL提供三大核心优势：(i)显著更强的纯文本理解能力，在多个案例中超越可比的纯文本骨干模型；(ii)强大的长上下文理解能力，具有原生的256K token窗口，适用于文本和交错多模态输入，能够在长文档和视频中实现忠实保留、检索和交叉引用；(iii)先进的多模态推理能力，涵盖单图像、多图像和视频任务，在包括MMMU和视觉数学基准测试(如MathVista和MathVision)的全面评估中展示了领先性能。在架构上，我们引入了三个关键升级：(i)增强的交错-MRoPE，用于图像和视频的更强时空建模；(ii)DeepStack集成，有效利用多级ViT特征来加强视觉语言对齐；(iii)基于文本的时间对齐，从T-RoPE演变为显式文本时间戳对齐，以实现更精确的时间定位。在可比的token预算和延迟约束下，Qwen3-VL在密集型和专家混合架构中均实现了更优的性能。我们期望Qwen3-VL能在实际工作流程中作为图像推理、智能体决策和多模态代码智能的基础引擎。


### 论文摘要

We introduce Qwen3-VL, the most capable vision-language model in the Qwen series to date, achieving superior performance across a broad range of multimodal benchmarks. It natively supports interleaved contexts of up to 256K tokens, seamlessly integrating text, images, and video. The model family includes both dense (2B/4B/8B/32B) and mixture-of-experts (30B-A3B/235B-A22B) variants to accommodate diverse latency-quality trade-offs. Qwen3-VL delivers three core pillars: (i) markedly stronger pure-text understanding, surpassing comparable text-only backbones in several cases; (ii) robust long-context comprehension with a native 256K-token window for both text and interleaved multimodal inputs, enabling faithful retention, retrieval, and cross-referencing across long documents and videos; and (iii) advanced multimodal reasoning across single-image, multi-image, and video tasks, demonstrating leading performance on comprehensive evaluations such as MMMU and visual-math benchmarks (e.g., MathVista and MathVision). Architecturally, we introduce three key upgrades: (i) an enhanced interleaved-MRoPE for stronger spatial-temporal modeling across images and video; (ii) DeepStack integration, which effectively leverages multi-level ViT features to tighten vision-language alignment; and (iii) text-based time alignment for video, evolving from T-RoPE to explicit textual timestamp alignment for more precise temporal grounding. Under comparable token budgets and latency constraints, Qwen3-VL achieves superior performance in both dense and Mixture-of-Experts (MoE) architectures. We envision Qwen3-VL serving as a foundational engine for image-grounded reasoning, agentic decision-making, and multimodal code intelligence in real-world workflows.

---

## 46. A Category of the Political \Large{Part I - Homónoia}

**论文链接:** [http://arxiv.org/abs/2511.21623v1](http://arxiv.org/abs/2511.21623v1)

**作者:** Joseph Abdou

**发布时间:** 2025-11-26

**备注:** 43 pages

### GPT解析

### 总结

本研究构建了政治组织及其转型的数学模型，引入了政治配置和政治基础两个类别，并通过基础对变量（包括基础和根基）以及相关函子建立了它们之间的关系。

### 背景

需要为政治组织及其转型提供数学模型的理论研究

### 目的

提供政治组织及其转型的数学模型

### 方法

构建政治配置和政治基础两个类别，引入基础对变量（包括由有限成员组成的基础和反映成员利益/价值观/愿望的根基状态），定义p-formation和p-site对象，以及相应的态射和函子（Knit、Nerve和Canon）

### 主要发现

Canon函子在适当意义上是Knit和Nerve函子的逆

### 结论

通过建立政治配置和政治基础之间的函子关系，为政治组织及其转型提供了数学基础

### 翻译

本研究旨在提供政治组织及其转型的数学模型。为此，我们构建了两个类别，分别称为政治配置和政治基础。我们的构建依赖于称为基础对的一对变量。其中一个变量称为基础，由有限数量的成员（代理人）组成，而另一个变量称为根基，由一组状态组成，反映基础成员的所有相关利益/价值观/愿望。配置中的一个对象称为p-formation，扩展了单纯复形的概念，而一个态射，表示基础的重组，扩展了单纯映射的概念。基础中的一个对象称为p-site，描述政治概况，即根基状态如何在代理人之间交织。政治站点之间的态射由一对映射组成，即基础映射和根基映射，满足适当条件。两个函子将基础和配置联系起来：Knit函子将每个p-site分配给一个p-formation，Nerve函子将每个p-site分配给一个单纯复形。相反方向的一个函子称为Canon，它将任何p-formation分配给其规范p-site，结果在适当意义上是Knit和Nerve的逆。


### 论文摘要

This research aims at providing a mathematical model of the organization of the polity and its transformation. For that purpose we construct two categories named respectively Political Configuration and Political Foundation. Our construction depends on a couple of variables called the foundational pair. One variable, called the Base, consists of a finite number of members (agents), while the other, called the Ground, consists of a set of states that reflect all relevant interests/values/aspirations of the base members. An object of the Configuration, called p-formation, extends the notion of simplicial complex, and a morphism, which expresses the recomposition of the base, extends the notion of simplicial map. An object of the Foundation, called p-site, describes the profile of the polity, that is, how the states of the ground are intertwined between the agents. A morphism between political sites consists of a pair of maps, namely a Base map and a Ground map, satisfying appropriate conditions. Two functors relate the Foundation and the Configuration: the Knit which attributes to each p-site a p-formation and the Nerve which attributes to each p-site a simplicial complex. In the opposite direction a functor, called Canon, which attributes to any p-formation its canonical p-site, turns out to be in an appropriate sense the inverse of the Knit and the Nerve.

---

## 47. ReSAM: Refine, Requery, and Reinforce: Self-Prompting Point-Supervised Segmentation for Remote Sensing Images

**论文链接:** [http://arxiv.org/abs/2511.21606v1](http://arxiv.org/abs/2511.21606v1)

**作者:** M. Naseer Subhani

**发布时间:** 2025-11-26

### GPT解析

### 总结

提出了一种自提示的点监督框架，通过Refine-Requery-Reinforce循环将Segment Anything Model (SAM)适应到遥感影像上，无需全掩码监督即可逐步提高分割质量和领域鲁棒性。

### 背景

交互式分割模型如SAM在自然图像上表现出色，但在遥感影像上表现不佳，这是由于严重的领域偏移和密集标注的稀缺性。

### 目的

开发一种仅使用稀疏点标注就能将SAM适应到遥感影像的方法，解决领域偏移和标注稀缺问题。

### 方法

采用Refine-Requery-Reinforce循环：从初始点生成粗略伪掩码(Refine)，通过自构建的框提示改进(Requery)，跨迭代对齐嵌入以减少确认偏差(Reinforce)。

### 主要发现

在三个RSI基准数据集(WHU, HRSID, 和NWPU VHR-10)上评估，结果显示该方法持续优于预训练SAM和最近的点监督分割方法。

### 结论

自提示和语义对齐为基础分割模型在遥感应用中的可扩展点级适应提供了有效途径。

### 翻译

交互式分割模型如Segment Anything Model (SAM)在自然图像上表现出 remarkable 的泛化能力，但由于严重的领域偏移和密集标注的稀缺性，在遥感影像(RSI)上表现不佳。为此，我们提出了一种自提示、点监督的框架，仅使用稀疏点标注即可将SAM适应到RSI上。我们的方法采用Refine-Requery-Reinforce循环，其中从初始点生成粗略伪掩码(Refine)，通过自构建的框提示进行改进(Requery)，并在迭代过程中对齐嵌入以减少确认偏差(Reinforce)。在不依赖全掩码监督的情况下，我们的方法通过自引导提示适应逐步提高SAM的分割质量和领域鲁棒性。我们在三个RSI基准数据集(WHU、HRSID和NWPU VHR-10)上评估了我们的方法，结果表明我们的方法持续优于预训练SAM和最近的点监督分割方法。我们的结果表明，自提示和语义对齐为基础分割模型在遥感应用中的可扩展点级适应提供了高效途径。


### 论文摘要

Interactive segmentation models such as the Segment Anything Model (SAM) have demonstrated remarkable generalization on natural images, but perform suboptimally on remote sensing imagery (RSI) due to severe domain shift and the scarcity of dense annotations. To address this, we propose a self-prompting, point-supervised framework that adapts SAM to RSIs using only sparse point annotations. Our method employs a Refine-Requery-Reinforce loop, where coarse pseudo-masks are generated from initial points (Refine), improved with self-constructed box prompts (Requery), and embeddings are aligned across iterations to reduce confirmation bias (Reinforce). Without relying on full-mask supervision, our approach progressively enhances SAM's segmentation quality and domain robustness through self-guided prompt adaptation . We evaluate our proposed method on three RSI benchmark datasets, including WHU, HRSID, and NWPU VHR-10, showing that our method consistently surpasses pretrained SAM and recent point-supervised segmentation methods. Our results demonstrate that self-prompting and semantic alignment provide an efficient path towards scalable, point-level adaptation of foundation segmentation models for remote sensing applications.

---

## 48. On the Degrees of Freedom of some Lasso procedures

**论文链接:** [http://arxiv.org/abs/2511.21595v1](http://arxiv.org/abs/2511.21595v1)

**作者:** Mauro Bernardi, Antonio Canale, Marco Stefanucci

**发布时间:** 2025-11-26

### GPT解析

### 总结

本文首次为自适应Lasso和自适应群组Lasso等惩罚回归方法提供了有效自由度的无偏估计器，并推导了它们的主要理论性质，适用于正交和非正交设计。这些结果基于Stein的无偏风险估计框架，包含受正则化参数、系数符号和最小二乘估计影响的膨胀项，使更准确的模型选择和无偏预测误差估计成为可能。

### 背景

有效自由度量化了用于生成预测的实际信息量，在模型评估和选择中起关键作用。虽然Lasso惩罚有闭式估计器，但自适应Lasso和自适应群组Lasso等广泛使用的惩罚方法的自适应扩展一直缺乏类似的理论表征。

### 目的

为自适应惩罚回归方法提供有效自由度的无偏估计器，推导其理论性质，并实现更准确的模型选择标准和预测误差估计。

### 方法

在Stein的无偏风险估计框架下，为自适应Lasso和自适应群组Lasso推导有效自由度的无偏估计器，考虑了正交和非正交设计情况，推导出包含膨胀项的表达式。

### 主要发现

得到了有效自由度的无偏估计器及其主要理论性质，发现表达式包含受正则化参数、系数符号和最小二乘估计影响的膨胀项，这些进展使更准确的模型选择和无偏预测误差估计成为可能。

### 结论

这些贡献为理解自适应回归中的模型复杂性提供了严格的理论基础，弥合了理论与实践之间的关键差距，使模型评估和选择更加准确。

### 翻译

惩罚回归模型的有效自由度量化了用于生成预测的实际信息量，在模型评估和选择中起着关键作用。虽然Lasso惩罚有闭式估计器，但广泛使用的惩罚方法的自适应扩展，包括自适应Lasso和自适应群组Lasso，一直缺乏类似的理论表征。本文首次为这些方法提供了有效自由度的无偏估计器，以及它们的主要理论性质，适用于正交和非正交设计，这些结果是在Stein的无偏风险估计框架下推导出来的。得到的表达式包含受正则化参数、系数符号和最小二乘估计影响的膨胀项。这些进展使得更准确的模型选择标准和无偏预测误差估计成为可能，通过合成和真实数据进行了说明。这些贡献为理解自适应回归中的模型复杂性提供了严格的理论基础，弥合了理论与实践之间的关键差距。


### 论文摘要

The effective degrees of freedom of penalized regression models quantify the actual amount of information used to generate predictions, playing a pivotal role in model evaluation and selection. Although a closed-form estimator is available for the Lasso penalty, adaptive extensions of widely used penalized approaches, including the Adaptive Lasso and Adaptive Group Lasso, have remained without analogous theoretical characterization. This paper presents the first unbiased estimator of the effective degrees of freedom for these methods, along with their main theoretical properties, for both orthogonal and non-orthogonal designs, derived within Stein's unbiased risk estimation framework. The resulting expressions feature inflation terms influenced by the regularization parameter, coefficient signs, and least-squares estimates. These advances enable more accurate model selection criteria and unbiased prediction error estimates, illustrated through synthetic and real data. These contributions offer a rigorous theoretical foundation for understanding model complexity in adaptive regression, bridging a critical gap between theory and practice.

---

## 49. Learning When to Stop: Adaptive Latent Reasoning via Reinforcement Learning

**论文链接:** [http://arxiv.org/abs/2511.21581v1](http://arxiv.org/abs/2511.21581v1)

**作者:** Alex Ning, Yen-Ling Kuo, Gabe Gomes

**发布时间:** 2025-11-26

**备注:** 13 pages, 6 figures

### GPT解析

### 总结

本文提出了一种新的Transformer语言模型推理方法——潜在推理(Latent reasoning)，相比传统的思维链推理能有效压缩推理长度，同时保持准确性。

### 背景

潜在推理是Transformer语言模型的新发展，它通过直接传递信息丰富的潜在状态而非依赖人类语言标记作为推理媒介，突破了传统推理方法的限制。

### 目的

开发自适应长度的潜在推理模型，并通过优化推理长度来减少计算使用量，提高潜在推理模型的压缩能力。

### 方法

研究人员开发了自适应长度的潜在推理模型，并引入了一种后SFT强化学习方法，通过最小化推理长度同时保持准确性来优化推理过程。

### 主要发现

在Llama 3.2 1B模型和GSM8K-Aug数据集上的实验表明，潜在推理方法可实现52%的总推理长度下降，同时不损失准确性。

### 结论

潜在推理方法能有效减少计算资源使用，提高推理效率，为Transformer语言模型提供了一种更高效的推理途径。

### 翻译

潜在推理代表了Transformer语言模型的新发展，相比思维链推理在压缩推理长度方面显示出潜力。通过将信息丰富的先前最终潜在状态直接传递到下一个序列，潜在推理消除了以人类语言标记作为推理媒介的限制。我们开发了自适应长度的潜在推理模型，并引入了一种后SFT强化学习方法，通过最小化推理长度同时保持准确性来优化潜在推理长度。这进一步减少了计算使用量，并提高了潜在推理模型的压缩能力。在Llama 3.2 1B模型和GSM8K-Aug数据集上的实验显示，总推理长度下降了52%，且没有准确性的损失。未来工作计划包括扩展到更多模型和数据集、分析训练系数之间的关系、尝试架构变化，并继续进行潜在推理SFT的知识蒸馏工作。我们在GitHub上提供了代码和预训练权重。


### 论文摘要

Latent reasoning represents a new development in Transformer language models that has shown potential in compressing reasoning lengths compared to chain-of-thought reasoning. By directly passing the information-rich previous final latent state into the next sequence, latent reasoning removes the restriction to human language tokens as the medium for reasoning. We develop adaptive-length latent reasoning models and introduce a post-SFT reinforcement-learning methodology to optimize latent reasoning length by minimizing reasoning length while maintaining accuracy. This, in turn, further reduces compute usage and raises the bar on the compressive capabilities of latent reasoning models. Experiments on the Llama 3.2 1B model and the GSM8K-Aug dataset show a $52\%$ drop in total reasoning length with no penalty to accuracy. In future work, we plan to extend to additional models and datasets, analyze relationships between training coefficients, experiment with architecture variations, and continue our knowledge distillation for latent reasoning SFT efforts. We make our code and pretrained weights available at https://github.com/apning/adaptive-latent-reasoning.

---

## 50. From Prediction to Foresight: The Role of AI in Designing Responsible Futures

**论文链接:** [http://arxiv.org/abs/2511.21570v1](http://arxiv.org/abs/2511.21570v1)

**作者:** Maria Perez-Ortiz

**发布时间:** 2025-11-26

**DOI:** 10.69828/4d4kja

**备注:** Accessible at https://projecteuclid.org/journals/journal-of-artificial-intelligence-for-sustainable-development/volume-1/issue-1/From-Prediction-to-Foresight--The-Role-of-AI-in/10.69828/4d4kja.full

### GPT解析

### 总结

本文提出了'负责任的计算前瞻性思考'概念，探讨了以人为中心的人工智能和计算模型在推进负责任前瞻性思考中的作用，建立了这一新领域的基础原则，并展示了相关AI工具。

### 背景

在技术快速发展和全球挑战复杂化的时代，负责任的前瞻性思考已成为政策制定者应对未来不确定性和塑造未来的必要框架。

### 目的

提出并定义'负责任的计算前瞻性思考'概念，探讨AI在其中的作用，建立基础原则，展示相关工具，并倡导AI与前瞻性实践的整合。

### 方法

研究AI与模拟和情景分析相结合的方式，分析如何增强政策制定者应对不确定性、评估风险和制定策略的能力。

### 主要发现

AI能帮助政策制定者应对不确定性并制定面向可持续未来的策略；负责任的前瞻性思考需要理解社会、环境、经济和政治系统间的相互依赖关系；AI应作为支持工具而非替代人类判断。

### 结论

AI在负责任、以人为中心的前瞻性思考中扮演补充性角色，应深思熟虑地整合到前瞻性实践中，以赋能政策制定者和社区应对21世纪的重大挑战。

### 翻译

在一个以技术快速发展和复杂全球挑战为特征的时代，负责任的前瞻性思考已成为政策制定者旨在应对未来不确定性并塑造未来的必要框架。负责任的前瞻性思考包含对新兴机会和风险的道德预期，重点是促进主动、可持续和负责任的未来设计。本文提出了'负责任的计算前瞻性思考'这一术语，探讨了以人为中心的人工智能和计算模型在推进负责任前瞻性思考中的作用，为这一新领域建立了一套基础原则，并展示了目前正在塑造它的AI驱动的前瞻性工具套件。AI，特别是与模拟和情景分析相结合，增强了政策制定者应对不确定性、评估风险和制定面向可持续、有韧性未来的策略的能力。然而，负责任的前瞻性思考超越了单纯的技术预测；它需要对社会、环境、经济和政治系统内部的相互依赖关系有细致的理解，并致力于支持人类智能的道德、长期决策。我们认为，AI将在负责任的、以人为中心的前瞻性思考中扮演支持工具的角色，补充而非替代政策制定者的判断，从而能够主动塑造有韧性和道德上健全的未来。本文倡导将AI深思熟虑地整合到前瞻性实践中，以赋能政策制定者和社区，使他们能够应对21世纪的重大挑战。


### 论文摘要

In an era marked by rapid technological advancements and complex global challenges, responsible foresight has emerged as an essential framework for policymakers aiming to navigate future uncertainties and shape the future. Responsible foresight entails the ethical anticipation of emerging opportunities and risks, with a focus on fostering proactive, sustainable, and accountable future design. This paper coins the term "responsible computational foresight", examining the role of human-centric artificial intelligence and computational modeling in advancing responsible foresight, establishing a set of foundational principles for this new field and presenting a suite of AI-driven foresight tools currently shaping it. AI, particularly in conjunction with simulations and scenario analysis, enhances policymakers' ability to address uncertainty, evaluate risks, and devise strategies geared toward sustainable, resilient futures. However, responsible foresight extends beyond mere technical forecasting; it demands a nuanced understanding of the interdependencies within social, environmental, economic and political systems, alongside a commitment to ethical, long-term decision-making that supports human intelligence. We argue that AI will play a role as a supportive tool in responsible, human-centered foresight, complementing rather than substituting policymaker judgment to enable the proactive shaping of resilient and ethically sound futures. This paper advocates for the thoughtful integration of AI into foresight practices to empower policymakers and communities as they confront the grand challenges of the 21st century.

---

## 51. VacuumVLA: Boosting VLA Capabilities via a Unified Suction and Gripping Tool for Complex Robotic Manipulation

**论文链接:** [http://arxiv.org/abs/2511.21557v1](http://arxiv.org/abs/2511.21557v1)

**作者:** Hui Zhou, Siyuan Huang, Minxing Li, Hao Zhang, Lue Fan, Shaoshuai Shi

**发布时间:** 2025-11-26

**备注:** 8 pages

### GPT解析

### 总结

本文提出了一种低成本、集成的硬件设计，将机械双指夹钳与真空吸盘单元结合，实现单一末端执行器内的双模式操作，扩展了机器人的任务能力。

### 背景

视觉语言动作模型通过利用大规模预训练的视觉和语言表示，显著推进了通用机器人操作。然而，现有VLA系统大多采用平行双指夹钳作为默认末端执行器，这种夹钳存在固有局限性，如接触面积不足或缺乏附着力，无法处理擦拭玻璃表面或无把手抽屉等现实世界任务。

### 目的

克服传统双指夹钳的局限性，设计一种能够处理更广泛任务的末端执行器，实现灵活切换或协同使用两种抓取模式。

### 方法

提出一种低成本、集成的硬件设计，将机械双指夹钳与真空吸盘单元结合，实现单一末端执行器内的双模式操作。在DexVLA和Pi0两个最先进的VLA框架中验证设计的效率和实用性。

### 主要发现

实验结果表明，使用这种混合末端执行器，机器人可以成功执行多个传统双指夹钳无法完成的复杂任务，显著扩展了可行任务的范围。

### 结论

所提出的混合末端执行器通过灵活切换或协同使用机械夹钳和真空吸盘两种模式，有效克服了传统双指夹钳的局限性，使机器人能够处理更广泛的现实世界任务。所有硬件设计和控制系统都将公开发布。

### 翻译

视觉语言动作模型通过利用大规模预训练的视觉和语言表示，显著推进了通用机器人操作。在现有方法中，大多数当前VLA系统采用平行双指夹钳作为其默认末端执行器。然而，这种夹钳在处理某些现实世界任务时存在固有局限性，如擦拭玻璃表面或打开无把手抽屉，原因是接触面积不足或缺乏附着力。为克服这些挑战，我们提出了一种低成本、集成的硬件设计，将机械双指夹钳与真空吸盘单元结合，实现了单一末端执行器内的双模式操作。我们的系统支持两种模式的灵活切换或协同使用，扩展了可行任务的范围。我们在两个最先进的VLA框架（DexVLA和Pi0）中验证了设计的效率和实用性。实验结果表明，使用所提出的混合末端执行器，机器人可以成功执行多个传统双指夹钳无法完成的复杂任务。所有硬件设计和控制系统都将公开发布。


### 论文摘要

Vision Language Action models have significantly advanced general purpose robotic manipulation by harnessing large scale pretrained vision and language representations. Among existing approaches, a majority of current VLA systems employ parallel two finger grippers as their default end effectors. However, such grippers face inherent limitations in handling certain real world tasks such as wiping glass surfaces or opening drawers without handles due to insufficient contact area or lack of adhesion. To overcome these challenges, we present a low cost, integrated hardware design that combines a mechanical two finger gripper with a vacuum suction unit, enabling dual mode manipulation within a single end effector. Our system supports flexible switching or synergistic use of both modalities, expanding the range of feasible tasks. We validate the efficiency and practicality of our design within two state of the art VLA frameworks: DexVLA and Pi0. Experimental results demonstrate that with the proposed hybrid end effector, robots can successfully perform multiple complex tasks that are infeasible for conventional two finger grippers alone. All hardware designs and controlling systems will be released.

---

## 52. EoS-FM: Can an Ensemble of Specialist Models act as a Generalist Feature Extractor?

**论文链接:** [http://arxiv.org/abs/2511.21523v1](http://arxiv.org/abs/2511.21523v1)

**作者:** Pierre Adorni, Minh-Tan Pham, Stéphane May, Sébastien Lefèvre

**发布时间:** 2025-11-26

### GPT解析

### 总结

该论文提出了一种新颖的专家集合框架用于构建遥感基础模型，旨在解决当前基础模型训练中资源消耗过大和可持续性问题。

### 背景

基础模型在自然语言处理和计算机视觉领域已取得显著进展，地球观测社区也开始类似努力。当前策略主要关注扩大模型规模和数据集规模，需要巨大计算和数据资源，限制了只有少数大型机构能够访问，且与环境可持续AI原则相悖。

### 目的

提出一种高效替代方案，构建专家集合框架用于遥感基础模型(RSFM)，解决资源消耗和可持续性问题。

### 方法

将训练过程分解为轻量级、任务特定的ConvNeXtV2专家模型，这些专家可被冻结和重用。这种模块化方法在效率、可解释性和可扩展性方面具有优势，并自然支持联邦训练、剪枝和持续专家集成，特别适合协作和资源受限环境。

### 主要发现

专家集合框架为构建可扩展和高效的遥感基础模型提供了新方向，解决了当前策略中的资源限制和可持续性问题。

### 结论

该框架为构建可扩展和高效的遥感基础模型设定了新方向，为资源受限和协作环境下的遥感模型开发提供了可行解决方案。

### 翻译

最近基础模型的进展在自然语言处理和计算机视觉等领域显示出巨大潜力，地球观测社区也正在出现类似努力。这些模型旨在有限监督下跨任务泛化，减少为每个任务单独训练模型的需求。然而，当前策略主要关注扩大模型规模和数据集规模，需要巨大的计算和数据资源，限制了只有少数大型机构能够访问。此外，这种不断扩大的模型范式与可持续和环境负责的AI原则形成鲜明对比，因为它导致巨大的碳足迹和资源效率低下。在这项工作中，我们提出了一种新颖且高效的替代方案：用于构建遥感基础模型(RSFMs)的专家集合框架。我们的方法将训练过程分解为轻量级的、任务特定的ConvNeXtV2专家模型，这些模型可以被冻结和重用。这种模块化方法在效率、可解释性和可扩展性方面具有显著优势。此外，它自然支持联邦训练、剪枝和持续专家集成，使其特别适合协作和资源受限的环境。我们的框架为构建可扩展和高效的RSFMs设定了新方向。


### 论文摘要

Recent advances in foundation models have shown great promise in domains such as natural language processing and computer vision, and similar efforts are now emerging in the Earth Observation community. These models aim to generalize across tasks with limited supervision, reducing the need for training separate models for each task. However, current strategies, which largely focus on scaling model size and dataset volume, require prohibitive computational and data resources, limiting accessibility to only a few large institutions. Moreover, this paradigm of ever-larger models stands in stark contrast with the principles of sustainable and environmentally responsible AI, as it leads to immense carbon footprints and resource inefficiency. In this work, we present a novel and efficient alternative: an Ensemble-of-Specialists framework for building Remote Sensing Foundation Models (RSFMs). Our method decomposes the training process into lightweight, task-specific ConvNeXtV2 specialists that can be frozen and reused. This modular approach offers strong advantages in efficiency, interpretability, and extensibility. Moreover, it naturally supports federated training, pruning, and continuous specialist integration, making it particularly well-suited for collaborative and resource-constrained settings. Our framework sets a new direction for building scalable and efficient RSFMs.

---

## 53. Causal Inference: A Tale of Three Frameworks

**论文链接:** [http://arxiv.org/abs/2511.21516v1](http://arxiv.org/abs/2511.21516v1)

**作者:** Linbo Wang, Thomas Richardson, James Robins

**发布时间:** 2025-11-26

### GPT解析

### 总结

本文比较介绍了三种主要的因果推断框架：潜在结果框架、结构方程模型和有向无环图，阐明了它们之间的联系、各自的优势和局限性，以及它们如何在实践中结合使用。

### 背景

因果推断是许多科学领域的中心目标。过去几十年中，出现了三个主要框架来形式化因果问题并指导分析。

### 目的

对三种因果推断框架进行比较性介绍，阐明它们之间的联系，突出各自的优势和局限性，并说明它们如何在实践中结合使用。

### 方法

比较性介绍三种因果推断框架。

### 主要发现

尽管这些框架在语言、假设和哲学取向上有所不同，但它们往往能产生兼容或互补的见解。

### 结论

讨论面向具有统计学或因果推断背景的研究生和研究人员，他们正在寻求在各种实质领域应用因果方法的概念基础。

### 翻译

因果推断是许多科学领域的中心目标。过去几十年中，三个主要框架已出现以形式化因果问题并指导其分析：潜在结果框架、结构方程模型和有向无环图。尽管这些框架在语言、假设和哲学取向上有所不同，但它们往往能产生兼容或互补的见解。本文对这三个框架进行了比较性介绍，阐明了它们之间的联系，突出了各自的优势和局限性，并说明了它们如何在实践中结合使用。讨论面向具有统计学或因果推断背景的研究生和研究人员，他们正在寻求在各种实质领域应用因果方法的概念基础。


### 论文摘要

Causal inference is a central goal across many scientific disciplines. Over the past several decades, three major frameworks have emerged to formalize causal questions and guide their analysis: the potential outcomes framework, structural equation models, and directed acyclic graphs. Although these frameworks differ in language, assumptions, and philosophical orientation, they often lead to compatible or complementary insights. This paper provides a comparative introduction to the three frameworks, clarifying their connections, highlighting their distinct strengths and limitations, and illustrating how they can be used together in practice. The discussion is aimed at researchers and graduate students with some background in statistics or causal inference who are seeking a conceptual foundation for applying causal methods across a range of substantive domains.

---

## 54. SpatialBench: Benchmarking Multimodal Large Language Models for Spatial Cognition

**论文链接:** [http://arxiv.org/abs/2511.21471v1](http://arxiv.org/abs/2511.21471v1)

**作者:** Peiran Xu, Sudong Wang, Yao Zhu, Jianing Li, Yunjian Zhang

**发布时间:** 2025-11-26

### GPT解析

### 总结

本研究提出了一种层次空间认知框架和相应基准测试SpatialBench，用于评估多模态大语言模型(MLLMs)的空间认知能力。实验发现模型在感知基础方面表现良好，但在符号推理、因果推理和规划方面存在局限，且与人类的目标导向抽象能力不同。

### 背景

空间认知对现实世界多模态智能至关重要，使模型能够有效与物理环境交互。尽管多模态大语言模型(MLLMs)取得了显著进展，但现有基准测试往往过度简化空间认知，将其简化为单一维度指标，无法捕捉空间能力的层次结构和相互依赖性。

### 目的

解决现有基准测试过度简化空间认知的问题，建立第一个用于测量多模态大语言模型中层次空间认知的系统框架，为未来的空间智能系统奠定基础。

### 方法

提出一个层次空间认知框架，将空间智能分解为从基本观察到高级规划的五个渐进复杂层次；构建SpatialBench，一个大规模、细粒度的基准，涵盖与这些认知水平对齐的15个任务；引入面向高级能力指标，提供跨异构任务的统一评估；在大量MLLMs上进行实验，并进行额外的人类测试对比。

### 主要发现

1. 模型在认知水平上表现出明显的性能分层；2. 模型在感知基础方面表现强劲，但在符号推理、因果推理和规划方面仍然有限；3. 人类执行选择性、目标导向的抽象，而MLLMs则倾向于过度关注表面细节，缺乏连贯的空间意图。

### 结论

该研究建立了第一个用于测量多模态大语言模型中层次空间认知的系统框架，揭示了当前模型在高级空间认知能力上的局限性，为未来空间智能系统的发展提供了基础和方向。

### 翻译

空间认知是现实世界多模态智能的基础，使模型能够有效与物理环境交互。虽然多模态大语言模型已经取得了显著进展，但现有基准测试往往过度简化空间认知，将其简化为单一维度指标，无法捕捉空间能力的层次结构和相互依赖性。为解决这一差距，我们提出了一个层次空间认知框架，将空间智能分解为从基本观察到高级规划的五个渐进复杂层次。基于此分类法，我们构建了SpatialBench，这是一个大规模、细粒度的基准，涵盖与这些认知水平对齐的15个任务。为了提供跨异构任务的统一评估，我们进一步引入了面向高级能力指标，可靠评估模型的整体空间推理能力。在大量多模态大语言模型上的广泛实验揭示了认知水平上的明显性能分层：模型在感知基础方面表现强劲，但在符号推理、因果推理和规划方面仍然有限。额外的人类测试表明，人类执行选择性、目标导向的抽象，而多模态大语言模型则倾向于过度关注表面细节，缺乏连贯的空间意图。我们的工作建立了第一个用于测量多模态大语言模型中层次空间认知的系统框架，为未来的空间智能系统奠定了基础。


### 论文摘要

Spatial cognition is fundamental to real-world multimodal intelligence, allowing models to effectively interact with the physical environment. While multimodal large language models (MLLMs) have made significant strides, existing benchmarks often oversimplify spatial cognition, reducing it to a single-dimensional metric, which fails to capture the hierarchical structure and interdependence of spatial abilities. To address this gap, we propose a hierarchical spatial cognition framework that decomposes spatial intelligence into five progressively complex levels from basic observation to high-level planning. Building upon this taxonomy, we construct SpatialBench, a large-scale, fine-grained benchmark covering 15 tasks aligned with these cognitive levels. To provide a unified evaluation across heterogeneous tasks, we further introduce a high-level capability-oriented metric that reliably assesses a model's overall spatial reasoning ability. Extensive experiments over massive MLLMs reveal distinct performance stratification across cognitive levels: models exhibit strong perceptual grounding yet remain limited in symbolic reasoning, causal inference, and planning. Additional human tests demonstrate that humans perform selective, goal-directed abstraction, while MLLMs tend to over-attend to surface details without coherent spatial intent. Our work establishes the first systematic framework for measuring hierarchical spatial cognition in MLLMs, laying the foundation for future spatially intelligent systems.

---

## 55. Semantic-Enhanced Feature Matching with Learnable Geometric Verification for Cross-Modal Neuron Registration

**论文链接:** [http://arxiv.org/abs/2511.21452v1](http://arxiv.org/abs/2511.21452v1)

**作者:** Wenwei Li, Lingyi Cai, Hui Gong, Qingming Luo, Anan Li

**发布时间:** 2025-11-26

### GPT解析

### 总结

该研究提出了一种新的深度学习框架，用于解决体内双光子显微镜图像和体外荧光显微光学断层扫描图像的精确配准问题，克服了跨模态外观差异、标注数据稀缺和严重组织变形等挑战。

### 背景

在神经科学的结构-功能分析中，准确配准单个神经元的体内双光子图像和体外荧光显微光学断层扫描图像至关重要。然而，这一任务面临显著挑战，包括跨模态外观差异大、标注数据稀缺以及严重的组织变形问题。

### 目的

开发一种新颖的深度学习框架，能够克服跨模态配准的挑战，实现高精度的神经元图像配准，以支持大规模相关研究。

### 方法

1. 引入一种语义增强的混合特征描述符，融合局部特征的几何精度和视觉基础模型DINOV3的上下文鲁棒性，以弥合模态差距。2. 用可学习的几何一致性置信度模块替代传统的RANSAC，该模块是一个新型分类器，经过训练用于识别和拒绝物理上不合理的对应关系。3. 采用数据高效的两阶段训练策略，包括在合成变形数据上预训练和在有限真实数据上微调，以解决数据稀缺问题。

### 主要发现

该框架为具有挑战性的生物医学成像场景中的高精度配准提供了稳健且准确的解决方案，能够实现大规模的相关性研究。

### 结论

所提出的深度学习框架成功解决了跨模态配准中的关键挑战，为神经科学中的结构-功能分析提供了可靠工具。

### 翻译

准确配准体内双光子图像和体外荧光显微光学断层扫描图像的单个神经元图像对神经科学中的结构-功能分析至关重要。由于显著的跨模态外观差异、标注数据稀缺和严重的组织变形，这一任务极具挑战性。我们提出了一种新颖的深度学习框架来解决这些问题。我们的方法引入了一种语义增强的混合特征描述符，它融合了局部特征的几何精度和视觉基础模型DINOV3的上下文鲁棒性，以弥合模态差距。为了处理复杂变形，我们用可学习的几何一致性置信度模块替代了传统的RANSAC，这是一个经过训练用于识别和拒绝物理上不合理对应关系的新型分类器。一种数据高效的两阶段训练策略，包括在合成变形数据上预训练和在有限真实数据上微调，克服了数据稀缺问题。我们的框架为具有挑战性的生物医学成像场景中的高精度配准提供了稳健且准确的解决方案，实现了大规模的相关性研究。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决如何准确对齐活体双光子显微镜图像和离体荧光显微光学断层扫描(fMOST)图像的神经元图像。这个问题在神经科学研究中至关重要，因为它能帮助研究人员将神经元在活体内的功能活动与其在离体组织中的精确解剖结构关联起来，从而理解大脑的基本结构-功能关系，但这种对齐面临跨模态外观差异大、标注数据稀缺和组织变形严重的挑战。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者分析了现有方法的局限性：传统方法依赖人工干预缺乏可扩展性；基于图的CellGPR计算复杂度高；主流深度学习方法需要大规模训练数据；计算机视觉中的特征匹配方法在神经光学成像上表现不佳。作者借鉴了SuperPoint提取局部几何特征、DINOV3提取高级语义特征、LightGlue进行初始匹配，并受可微分RANSAC启发设计了几何一致性验证模块。作者还设计了两阶段训练策略：先在合成变形数据上预训练，再在有限真实数据上微调，以解决数据稀缺问题。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是融合局部几何特征与高级语义信息创建混合特征描述符，用可学习的几何一致性模块替代传统RANSAC，并采用数据高效的两阶段训练策略。整体流程为：1)从两种图像中分别提取局部几何特征和语义特征；2)通过MLP融合这些特征创建混合描述符；3)使用LightGlue生成初始匹配；4)通过几何一致性模块过滤初始匹配，得到最终的高置信度对应点对；5)先在合成数据上预训练，再在真实数据上微调模型。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)语义增强的混合特征描述符，结合几何精确性和语义鲁棒性；2)可学习的几何一致性置信度模块，直接评估生物组织变形的物理合理性；3)数据高效的两阶段训练策略，克服数据稀缺问题。相比之前的工作，该方法实现了自动化对齐，计算效率比CellGPR快200倍且能找到更密集的匹配点(1325个vs 58个)，比主流深度学习方法更适合数据稀缺场景，比通用特征匹配方法更专门针对神经光学成像数据优化。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '这篇论文提出了一种融合语义增强特征和可学习几何验证的深度学习框架，实现了跨模态神经元图像的高精度、高密度、高效对齐，为大规模神经结构-功能关联研究提供了强有力的工具。'}


### 论文摘要

Accurately registering in-vivo two-photon and ex-vivo fluorescence micro-optical sectioning tomography images of individual neurons is critical for structure-function analysis in neuroscience. This task is profoundly challenging due to a significant cross-modality appearance gap, the scarcity of annotated data and severe tissue deformations. We propose a novel deep learning framework to address these issues. Our method introduces a semantic-enhanced hybrid feature descriptor, which fuses the geometric precision of local features with the contextual robustness of a vision foundation model DINOV3 to bridge the modality gap. To handle complex deformations, we replace traditional RANSAC with a learnable Geometric Consistency Confidence Module, a novel classifier trained to identify and reject physically implausible correspondences. A data-efficient two-stage training strategy, involving pre-training on synthetically deformed data and fine-tuning on limited real data, overcomes the data scarcity problem. Our framework provides a robust and accurate solution for high-precision registration in challenging biomedical imaging scenarios, enabling large-scale correlative studies.

---

## 56. Quantum electrodynamic description of the neutral hydrogen molecule ionization

**论文链接:** [http://arxiv.org/abs/2511.21430v1](http://arxiv.org/abs/2511.21430v1)

**作者:** Hui-hui Miao

**发布时间:** 2025-11-26

**备注:** 11 pages, 5 figures; Supplementary Information: 2 videos

### GPT解析

### 总结

本研究探讨了氢分子的电离动力学，结合量子电动力学和林布拉德主方程，提供了光-物质相互作用的从头描述，并考虑了耗散过程和外部粒子流入。

### 背景

氢分子电离动力学是量子化学中的一个基本基准系统。

### 目的

提供光与物质相互作用的从头描述，同时考虑耗散过程和外部粒子流入的影响。

### 方法

采用结合量子电动力学和林布拉德主方程的综合框架，系统探索系统在三种不同环境下的演化：封闭系统、耗散开放系统和流入驱动的开放量子系统。

### 主要发现

1)在所有配置中，普遍存在向中性氢分子形成的趋势；2)光子、电子和声子的耗散强度是关键控制参数，其中光子耗散显著加速系统稳定；3)粒子流入导致能量重分布，特别使原子态布居增加；4)电离途径对初始量子态极其敏感，由光子的组成和数量决定；5)在嵌入阳极的模型中，轨道杂化将电离概率的最大值限制在3/4。

### 结论

该研究为量子控制化学提供了统一的理论基础，对腔量子电动力学和量子信息处理的未来实验有直接意义。

### 翻译

氢分子的电离动力学作为量子化学中的一个基本基准，在结合量子电动力学和林布拉德主方程的综合框架内进行了研究。这种方法能够在考虑耗散过程和外部粒子流入的同时，提供光-物质相互作用的从头描述。我们系统地探索了系统在三种不同环境下的演化：封闭系统、耗散开放系统和流入驱动的开放量子系统。结果表明，在所有配置中，系统普遍趋向于形成中性氢分子。光子、电子和声子的耗散强度被确定为关键控制参数，其中光子耗散显著加速系统稳定。此外，粒子流入导致能量复杂重分布，特别是使原子态布居增加。电离途径对初始量子态极其敏感，由光子的组成和数量决定，这些因素控制着可及的自旋选择性激发通道。在嵌入阳极的模型中，轨道杂化将电离概率的最大值从根本上限制在3/4。该研究为量子控制化学提供了统一的理论基础，对腔量子电动力学和量子信息处理的未来实验具有直接意义。


### 论文摘要

The ionization dynamics of a hydrogen molecule, serving as a fundamental benchmark in quantum chemistry, is investigated within a comprehensive framework combining quantum electrodynamics and the Lindblad master equation. This approach enables a first-principles description of light--matter interactions while accounting for dissipative processes and external particle influx. We systematically explore the system's evolution across three distinct regimes: closed, dissipative open, and influx-driven open quantum systems. Our results reveal a universal tendency towards the formation of the neutral hydrogen molecule ($|\rm{H}_2\rangle$) across all configurations. The dissipation strengths for photons ($γ_Ω$), electrons ($γ_e$), and phonons ($γ_ω$) are identified as critical control parameters, with $γ_Ω$ significantly accelerating system stabilization. Furthermore, the introduction of particle influx ($μ_k$) leads to a complex redistribution of energy, notably populating the atomic state ($|\rm{H},\rm{H}\rangle$). The ionization pathway is exquisitely sensitive to the initial quantum state, dictated by the composition and number of photons, which governs the accessible spin-selective excitation channels. This is conclusively demonstrated in a model with an embedded anode, where the maximum ionization probability is fundamentally constrained to $\frac{3}{4}$ by orbital hybridization. This study provides a unified theoretical foundation for quantum-controlled chemistry, with direct implications for future experiments in cavity QED and quantum information processing.

---

## 57. SAM Guided Semantic and Motion Changed Region Mining for Remote Sensing Change Captioning

**论文链接:** [http://arxiv.org/abs/2511.21420v1](http://arxiv.org/abs/2511.21420v1)

**作者:** Futian Wang, Mengqi Wang, Xiao Wang, Haowen Wang, Jin Tang

**发布时间:** 2025-11-26

### GPT解析

### 总结

本研究提出了一种基于SAM基础模型的遥感变化描述方法，通过提取区域级表示和融合多种信息源，解决了现有方法区域感知弱和时间对齐有限的问题，在多个基准数据集上取得了最先进的性能。

### 背景

遥感变化描述是一个新兴且热门的研究任务，旨在用自然语言描述不同时间拍摄的遥感图像之间的变化内容。现有方法通常使用CNNs/Transformers提取视觉表示或通过辅助任务增强结果，但存在区域感知弱和时间对齐有限的问题。

### 目的

解决现有遥感变化描述方法中区域感知弱和时间对齐有限的问题，提高变化描述的准确性。

### 方法

使用SAM基础模型提取区域级表示，将感兴趣区域知识注入描述框架。具体包括：使用CNN/Transformer模型提取全局视觉特征，利用SAM基础模型划分语义级和运动级变化区域，利用专门构建的知识图提供感兴趣对象的信息，通过跨注意力融合这些异构信息源，最后使用Transformer解码器生成最终的自然语言描述。

### 主要发现

通过大量实验结果证明，该方法在多个广泛使用的基准数据集上取得了最先进的性能。

### 结论

该方法通过引入SAM基础模型和知识图，有效解决了现有方法在区域感知和时间对齐方面的问题，显著提高了遥感变化描述的性能。

### 翻译

遥感变化描述是一项新兴且热门的研究任务，旨在用自然语言描述不同时间拍摄的遥感图像之间感兴趣内容的变化。现有方法通常采用CNNs/Transformers从给定图像中提取视觉表示或结合辅助任务来增强最终结果，但存在区域感知弱和时间对齐有限的问题。为解决这些问题，本文探索使用SAM基础模型提取区域级表示，并将感兴趣区域知识注入描述框架。具体而言，我们使用CNN/Transformer模型提取全局视觉特征，利用SAM基础模型划分语义级和运动级变化区域，并利用专门构建的知识图提供感兴趣对象的信息。这些异构信息源通过跨注意力融合，然后使用Transformer解码器生成所观察变化的最终自然语言描述。大量实验结果表明，我们的方法在多个广泛使用的基准数据集上取得了最先进的性能。本文的源代码将在https://github.com/Event-AHU/SAM_ChangeCaptioning上发布。


### 论文摘要

Remote sensing change captioning is an emerging and popular research task that aims to describe, in natural language, the content of interest that has changed between two remote sensing images captured at different times. Existing methods typically employ CNNs/Transformers to extract visual representations from the given images or incorporate auxiliary tasks to enhance the final results, with weak region awareness and limited temporal alignment. To address these issues, this paper explores the use of the SAM (Segment Anything Model) foundation model to extract region-level representations and inject region-of-interest knowledge into the captioning framework. Specifically, we employ a CNN/Transformer model to extract global-level vision features, leverage the SAM foundation model to delineate semantic- and motion-level change regions, and utilize a specially constructed knowledge graph to provide information about objects of interest. These heterogeneous sources of information are then fused via cross-attention, and a Transformer decoder is used to generate the final natural language description of the observed changes. Extensive experimental results demonstrate that our method achieves state-of-the-art performance across multiple widely used benchmark datasets. The source code of this paper will be released on https://github.com/Event-AHU/SAM_ChangeCaptioning

---

## 58. Monet: Reasoning in Latent Visual Space Beyond Images and Language

**论文链接:** [http://arxiv.org/abs/2511.21395v1](http://arxiv.org/abs/2511.21395v1)

**作者:** Qixun Wang, Yang Shi, Yifei Wang, Yuanxing Zhang, Pengfei Wan, Kun Gai, Xianghua Ying, Yisen Wang

**发布时间:** 2025-11-26

### GPT解析

### 总结

本文提出了Monet训练框架，使多模态大语言模型能够在潜在视觉空间内直接进行推理，通过生成连续嵌入作为中间视觉思维，解决了现有方法在抽象视觉思维方面的局限性。

### 背景

基于图像的思考已成为推进视觉推理的有效范式，超越了纯文本思维链，但现有方法缺乏类似人类的抽象视觉思维能力，其灵活性受到外部工具的根本限制。

### 目的

开发一个训练框架，使多模态大语言模型能够在潜在视觉空间内直接进行推理，通过生成连续嵌入作为中间视觉思维。

### 方法

确定了训练MLLMs进行潜在视觉推理的两个核心挑战（高计算成本和监督不足），采用三阶段基于蒸馏的监督微调管道解决；提出VLPO强化学习方法，将潜在嵌入纳入策略梯度更新；构建Monet-SFT-125K高质量文本图像交错CoT数据集。

### 主要发现

Monet-7B模型在真实世界感知和推理基准测试中表现出一致性能提升，在抽象视觉推理任务上展现出强大的分布外泛化能力；GRPO主要增强基于文本的推理而非潜在推理。

### 结论

Monet框架成功解决了MLLMs在抽象视觉推理方面的局限性，通过创新的训练方法和高质量数据集提高了模型性能，为未来视觉潜在推理发展提供了见解。

### 翻译

基于图像的思考已成为推进视觉推理的有效范式，通过在中间推理步骤中注入视觉证据，超越了纯文本的思维链。然而，现有方法缺乏类似人类的抽象视觉思维能力，因为它们的灵活性受到外部工具的根本限制。在这项工作中，我们介绍了Monet，一个训练框架，使多模态大语言模型能够在潜在视觉空间内直接进行推理，通过生成连续的嵌入作为中间视觉思维。我们确定了训练MLLMs进行潜在视觉推理的两个核心挑战：潜在视觉对齐的高计算成本和对潜在嵌入的监督不足，并通过三阶段基于蒸馏的监督微调管道解决了这些问题。我们还揭示了将GRPO应用于潜在推理的局限性：它主要增强基于文本的推理而非潜在推理。为此，我们提出了VLPO，一种将潜在嵌入明确纳入策略梯度更新的强化学习方法。为了支持SFT，我们构建了Monet-SFT-125K，一个包含12.5万个真实世界、图表、OCR和几何CoT的高质量文本图像交错CoT数据集。我们的模型Monet-7B在真实世界感知和推理基准测试中表现出一致的性能提升，并在具有挑战性的抽象视觉推理任务上展现出强大的分布外泛化能力。我们还经验性地分析了每个训练组件的作用，讨论了早期不成功的尝试，为未来视觉潜在推理的发展提供了见解。


### 论文摘要

"Thinking with images" has emerged as an effective paradigm for advancing visual reasoning, extending beyond text-only chains of thought by injecting visual evidence into intermediate reasoning steps. However, existing methods fall short of human-like abstract visual thinking, as their flexibility is fundamentally limited by external tools. In this work, we introduce Monet, a training framework that enables multimodal large language models (MLLMs) to reason directly within the latent visual space by generating continuous embeddings that function as intermediate visual thoughts. We identify two core challenges in training MLLMs for latent visual reasoning: high computational cost in latent-vision alignment and insufficient supervision over latent embeddings, and address them with a three-stage distillation-based supervised fine-tuning (SFT) pipeline. We further reveal a limitation of applying GRPO to latent reasoning: it primarily enhances text-based reasoning rather than latent reasoning. To overcome this, we propose VLPO (Visual-latent Policy Optimization), a reinforcement learning method that explicitly incorporates latent embeddings into policy gradient updates. To support SFT, we construct Monet-SFT-125K, a high-quality text-image interleaved CoT dataset containing 125K real-world, chart, OCR, and geometry CoTs. Our model, Monet-7B, shows consistent gains across real-world perception and reasoning benchmarks and exhibits strong out-of-distribution generalization on challenging abstract visual reasoning tasks. We also empirically analyze the role of each training component and discuss our early unsuccessful attempts, providing insights for future developments in visual latent reasoning. Our model, data, and code are available at https://github.com/NOVAglow646/Monet.

---

## 59. Thinking With Bounding Boxes: Enhancing Spatio-Temporal Video Grounding via Reinforcement Fine-Tuning

**论文链接:** [http://arxiv.org/abs/2511.21375v1](http://arxiv.org/abs/2511.21375v1)

**作者:** Xin Gu, Haoji Zhang, Qihang Fan, Jingxuan Niu, Zhipeng Zhang, Libo Zhang, Guang Chen, Fan Chen, Longyin Wen, Sijie Zhu

**发布时间:** 2025-11-26

### GPT解析

### 总结

STVG-o1是一种新型框架，使现成的多模态大语言模型能够实现最先进的时空视频定位性能，无需架构修改。通过边界框思维链机制和多维强化奖励函数，STVG-o1在多个数据集上取得了突破性成果。

### 背景

时空视频定位(STVG)要求根据自然语言描述在未修剪视频中同时定位目标对象的时间和空间位置。尽管多模态大语言模型(MLLMs)在语言理解方面表现强大，但在STVG任务上表现不佳，原因是训练目标不匹配以及标准视觉编码器中细粒度的区域-词对齐能力较弱。

### 目的

提出STVG-o1框架，使现成的MLLMs能够实现最先进的STVG性能，无需任何架构修改。

### 方法

引入边界框思维链机制，在最终预测之前明确推理时空位置；设计一个多维强化奖励函数，包括格式、一致性、时间、空间和思考奖励，通过强化微调提供几何感知的监督。

### 主要发现

在HCSTVG-v1/v2和VidSTG上评估，STVG-o1在HCSTVG上设置了新的最先进结果，在HCSTVG-v1上比最佳特定任务方法高出7.3%的m_tIoU；在VidSTG上与专业模型相匹配；以较大优势超越所有现有的基于MLLM的方法；展示了跨数据集的强大开放词汇泛化能力。

### 结论

确立了MLLMs作为精确时空定位的可行且强大的主干网络。

### 翻译

时空视频定位(STVG)需要根据自然语言描述在未修剪的视频中同时时间和空间上定位目标对象。尽管多模态大语言模型(MLLMs)在语言理解方面表现强大，但由于训练目标不匹配和标准视觉编码器中细粒度的区域-词对齐能力较弱，它们在STVG上表现不佳。为此，我们提出了STVG-o1，这是第一个使现成的MLLMs能够实现最先进STVG性能的框架，无需任何架构修改。我们的方法引入了边界框思维链机制，在最终预测之前明确推理时空位置。我们进一步设计了一个包含格式、一致性、时间、空间和思考奖励的多维强化奖励函数，通过强化微调提供几何感知的监督。在HCSTVG-v1/v2和VidSTG上评估，STVG-o1在HCSTVG上设置了新的最先进结果，在HCSTVG-v1上比最佳特定任务方法高出7.3%的m_tIoU，在VidSTG上与专业模型相匹配，并以较大优势超越所有现有的基于MLLM的方法。它还展示了跨数据集的强大开放词汇泛化能力，确立了MLLMs作为精确时空定位的可行且强大的主干网络。我们的代码和模型将被发布。


### 论文摘要

Spatio-temporal video grounding (STVG) requires localizing a target object in untrimmed videos both temporally and spatially from natural language descriptions. Despite their strong language understanding, multimodal large language models (MLLMs) underperform on STVG due to misaligned training objectives and weak fine-grained region-word alignment in standard visual encoders. To address this, we propose STVG-o1, the first framework that enables off-the-shelf MLLMs to achieve state-of-the-art STVG performance without any architectural modifications. Our method introduces a bounding-box chain-of-thought mechanism that explicitly reasons about spatio-temporal locations in an intermediate step before producing the final prediction. We further design a multi-dimensional reinforcement reward function consisting of format, consistency, temporal, spatial, and think rewards, which provides geometry-aware supervision through reinforcement fine-tuning. Evaluated on HCSTVG-v1/v2 and VidSTG, STVG-o1 sets new state-of-the-art results on HCSTVG, outperforming the best task-specific method by 7.3\% m\_tIoU on HCSTVG-v1, matching specialized models on VidSTG, and surpassing all existing MLLM-based approaches by large margins. It also demonstrates strong open-vocabulary generalization across datasets, establishing MLLMs as viable and powerful backbones for precise spatio-temporal grounding. Our code and models will be released.

---

## 60. SurgMLLMBench: A Multimodal Large Language Model Benchmark Dataset for Surgical Scene Understanding

**论文链接:** [http://arxiv.org/abs/2511.21339v1](http://arxiv.org/abs/2511.21339v1)

**作者:** Tae-Min Choi, Tae Kyeong Jeong, Garam Kim, Jaemin Lee, Yeongyoon Koh, In Cheul Choi, Jae-Ho Chung, Jong Woong Park, Juyoun Park

**发布时间:** 2025-11-26

**备注:** 10 pages, 5 figures

### GPT解析

### 总结

本文提出了SurgMLLMBench，一个专门用于开发和评估交互式多模态大语言模型对手术场景理解的统一多模态基准，整合了像素级器械分割掩码和结构化VQA注释，涵盖多种手术领域，支持超越传统VQA任务的全面评估。

### 背景

多模态大语言模型在医疗和手术应用方面显示出潜力，但现有手术数据集主要采用视觉问答格式，具有异构分类法，缺乏像素级分割支持，限制了评估的一致性和适用性。

### 目的

提出一个统一的多模态基准SurgMLLMBench，专门用于开发和评估交互式多模态大语言模型对手术场景的理解，包括新收集的MAVIS数据集。

### 方法

整合像素级器械分割掩码和结构化VQA注释，涵盖腹腔镜、机器人辅助和显微手术领域，采用统一分类法，支持超越传统VQA任务的全面评估和更丰富的视觉对话交互。

### 主要发现

在SurgMLLMBench上训练的单个模型在不同领域上实现了一致的性能，并且能够有效推广到未见过的数据集。

### 结论

SurgMLLMBench将公开发布，作为推进多模态手术AI研究的强大资源，支持可重复评估和交互式手术推理模型的发展。

### 翻译

多模态大语言模型的最新进展突显了它们在医疗和手术应用方面的潜力。然而，现有的手术数据集主要采用视觉问答格式，具有异构分类法，并且缺乏像素级分割支持，限制了评估的一致性和适用性。我们提出了SurgMLLMBench，一个统一的多模态基准，专门用于开发和评估交互式多模态大语言模型对手术场景的理解，包括新收集的显微手术人工血管吻合数据集。它在统一分类法下整合了像素级器械分割掩码和结构化VQA注释，涵盖腹腔镜、机器人辅助和显微手术领域， enabling超越传统VQA任务的全面评估和更丰富的视觉对话交互。大量的基线实验表明，在SurgMLLMBench上训练的单个模型在不同领域上实现了一致的性能，并能有效推广到未见过的数据集。SurgMLLMBench将公开发布，作为推进多模态手术AI研究的强大资源，支持可重复评估和交互式手术推理模型的发展。


### 论文摘要

Recent advances in multimodal large language models (LLMs) have highlighted their potential for medical and surgical applications. However, existing surgical datasets predominantly adopt a Visual Question Answering (VQA) format with heterogeneous taxonomies and lack support for pixel-level segmentation, limiting consistent evaluation and applicability. We present SurgMLLMBench, a unified multimodal benchmark explicitly designed for developing and evaluating interactive multimodal LLMs for surgical scene understanding, including the newly collected Micro-surgical Artificial Vascular anastomosIS (MAVIS) dataset. It integrates pixel-level instrument segmentation masks and structured VQA annotations across laparoscopic, robot-assisted, and micro-surgical domains under a unified taxonomy, enabling comprehensive evaluation beyond traditional VQA tasks and richer visual-conversational interactions. Extensive baseline experiments show that a single model trained on SurgMLLMBench achieves consistent performance across domains and generalizes effectively to unseen datasets. SurgMLLMBench will be publicly released as a robust resource to advance multimodal surgical AI research, supporting reproducible evaluation and development of interactive surgical reasoning models.

---

## 61. TALES: A Taxonomy and Analysis of Cultural Representations in LLM-generated Stories

**论文链接:** [http://arxiv.org/abs/2511.21322v1](http://arxiv.org/abs/2511.21322v1)

**作者:** Kirti Bhagat, Shaily Bhatt, Athul Velagapudi, Aditya Vashistha, Shachi Dave, Danish Pruthi

**发布时间:** 2025-11-26

### GPT解析

### 总结

本研究提出TALES，评估大型语言模型生成的故事中针对不同印度文化身份的文化表征错误，发现88%的故事包含文化不准确之处，且错误在低资源语言和城市周边地区故事中更为普遍。

### 背景

全球数百万用户使用AI聊天机器人满足创意需求，了解聊天机器人如何代表不同文化引起广泛兴趣，但评估开放式任务中的文化表征仍具挑战性且研究不足。

### 目的

评估大型语言模型生成的故事中针对不同印度文化身份的文化表征错误。

### 方法

开发TALES-Tax分类法，通过焦点小组和个人调查收集见解；使用该分类法评估6个模型；进行大规模标注研究，涵盖来自印度71个地区和14种语言的108名标注者的2,925个标注；将标注转换为TALES-QA题库。

### 主要发现

88%的生成故事包含一个或多个文化不准确之处；错误在资源较少的语言和基于印度城市周边地区的故事中更为普遍；模型通常拥有必要的文化知识，尽管生成的故事充满文化表征错误。

### 结论

尽管模型拥有文化知识，但在生成故事时仍存在大量文化表征错误，这些错误在低资源语言和城市周边地区的故事中更为普遍，TALES-QA可作为评估模型文化知识的工具。

### 翻译

全球数百万用户转向AI聊天机器人满足其创意需求，引发了对这类聊天机器人如何代表多元文化的广泛兴趣。同时，评估开放式任务中的文化表征仍然具有挑战性且研究不足。在这项工作中，我们提出了TALES，一项针对不同印度文化身份的大型语言模型生成故事中文化表征错误的评估。首先，我们通过焦点小组(N=9)和个人调查(N=15)收集有印度生活经验的参与者的见解，开发了TALES-Tax，一种文化表征错误的分类法。使用TALES-Tax，我们通过一项大规模标注研究评估了6个模型，该研究涵盖了来自印度71个地区和14种语言的108名具有文化生活经验的标注者的2,925个标注。令人担忧的是，我们发现88%的生成故事包含一个或多个文化不准确之处，并且此类错误在资源较少的语言和基于印度城市周边地区的故事中更为普遍。最后，我们将标注转换为TALES-QA，一个用于评估基础模型文化知识的独立题库。通过这一评估，我们惊讶地发现，尽管生成的故事充满文化表征错误，但模型通常拥有必要的文化知识。


### 论文摘要

Millions of users across the globe turn to AI chatbots for their creative needs, inviting widespread interest in understanding how such chatbots represent diverse cultures. At the same time, evaluating cultural representations in open-ended tasks remains challenging and underexplored. In this work, we present TALES, an evaluation of cultural misrepresentations in LLM-generated stories for diverse Indian cultural identities. First, we develop TALES-Tax, a taxonomy of cultural misrepresentations by collating insights from participants with lived experiences in India through focus groups (N=9) and individual surveys (N=15). Using TALES-Tax, we evaluate 6 models through a large-scale annotation study spanning 2,925 annotations from 108 annotators with lived cultural experience from across 71 regions in India and 14 languages. Concerningly, we find that 88\% of the generated stories contain one or more cultural inaccuracies, and such errors are more prevalent in mid- and low-resourced languages and stories based in peri-urban regions in India. Lastly, we transform the annotations into TALES-QA, a standalone question bank to evaluate the cultural knowledge of foundational models. Through this evaluation, we surprisingly discover that models often possess the requisite cultural knowledge despite generating stories rife with cultural misrepresentations.

---

## 62. A Dynamics-Informed Gaussian Process Framework for 2D Stochastic Navier-Stokes via Quasi-Gaussianity

**论文链接:** [http://arxiv.org/abs/2511.21281v1](http://arxiv.org/abs/2511.21281v1)

**作者:** Boumediene Hamzi, Houman Owhadi

**发布时间:** 2025-11-26

### GPT解析

### 总结

该研究基于二维随机Navier-Stokes方程的拟高斯性证明，提出了一种具有严格理论基础的高斯过程先验框架，用于湍流建模和数据同化。

### 背景

高斯过程框架在流体动力学中应用日益增多，但其先验分布通常基于便利性选择，缺乏系统长期动力学的严格理论支持。Coe、Hairer和Tolomeo最近证明了二维随机Navier-Stokes方程的拟高斯性，建立了系统不变测度与线性Ornstein-Uhlenbeck过程高斯测度的等价关系。

### 目的

弥合高斯过程在流体动力学应用中的理论与实际差距，提供一种具有严格长期动力学理论基础的高斯过程先验框架。

### 方法

基于二维随机Navier-Stokes方程的拟高斯性理论，从线性OU模型的平稳协方差构建高斯过程先验，该协方差由强迫谱和耗散明确界定。

### 主要发现

通过从线性OU模型的平稳协方差构建高斯过程先验，可以为湍流提供具有严格长期动力学原理的先验分布。

### 结论

该研究建立了SPDE理论与实际数据同化之间的桥梁，为湍流建模提供了具有理论基础的高斯过程框架。

### 翻译

Coe、Hairer和Tolomeo最近对二维随机Navier-Stokes(SNS)方程的拟高斯性证明确立了该系统的唯一不变测度与其相应线性Ornstein-Uhlenbeck(OU)过程的高斯测度等价（相互绝对连续）。尽管高斯过程(GP)框架越来越多地用于流体动力学，但它们的先验分布通常基于便利性选择，而不是由系统的长期动力学严格证明。在这项工作中，我们通过引入一个直接基于这一理论基础的二维SNS概率框架来弥合这一差距。我们直接从线性OU模型的平稳协方差构建GP先验，该协方差由强迫谱和耗散明确界定。这为湍流提供了具有严格长期动力学原理的GP先验，弥合了SPDE理论和实际数据同化之间的差距。


### 论文摘要

The recent proof of quasi-Gaussianity for the 2D stochastic Navier--Stokes (SNS) equations by Coe, Hairer, and Tolomeo establishes that the system's unique invariant measure is equivalent (mutually absolutely continuous) to the Gaussian measure of its corresponding linear Ornstein--Uhlenbeck (OU) process. While Gaussian process (GP) frameworks are increasingly used for fluid dynamics, their priors are often chosen for convenience rather than being rigorously justified by the system's long-term dynamics.   In this work, we bridge this gap by introducing a probabilistic framework for 2D SNS built directly upon this theoretical foundation. We construct our GP prior precisely from the stationary covariance of the linear OU model, which is explicitly defined by the forcing spectrum and dissipation. This provides a principled, GP prior with rigorous long-time dynamical justification for turbulent flows, bridging SPDE theory and practical data assimilation.

---

## 63. Multi-Reward GRPO for Stable and Prosodic Single-Codebook TTS LLMs at Scale

**论文链接:** [http://arxiv.org/abs/2511.21270v1](http://arxiv.org/abs/2511.21270v1)

**作者:** Yicheng Zhong, Peiji Yang, Zhisheng Wang

**发布时间:** 2025-11-26

**备注:** 4 pages, 2 figures

### GPT解析

### 总结

该研究提出了一种多奖励组相对策略优化(GRPO)框架，用于解决单码本TTS LLMs中的韵律不稳定、说话人漂移和自然度下降问题，通过直接优化标记生成策略，显著提升了语音合成的质量。

### 背景

大型语言模型(LLMs)的最新进展改变了文本转语音(TTS)合成，启发了将语音表示为离散编解码器标记序列的自回归框架。单码本TTS LLMs已成为紧凑且可流式传输的架构，但存在韵律不稳定、说话人漂移和自然度下降的问题。

### 目的

解决单码本TTS LLMs中韵律不稳定、说话人漂移和自然度下降的问题，提升语音合成质量。

### 方法

提出多奖励GRPO框架，直接优化单码本TTS LLMs的标记生成策略，整合三个基于规则的奖励：长度惩罚(持续时间一致性)、熵正则化奖励(解码稳定性)和LLM注释的韵律对齐奖励(节奏监督)。外部推理LLM通过上下文学习预测停顿结构，提供符合人类偏好的监督信号。附加流匹配(FM)解码器评估通用性。

### 主要发现

GRPO优化增强了内在AR策略，在跨数据大小和模型规模的扩展性分析中，该方法一致提高了单码本TTS LLMs的韵律稳定性、说话人相似度和整体语音自然度。

### 结论

所提出的多奖励GRPO框架有效解决了单码本TTS LLMs中的关键问题，在各种数据大小和模型规模上均表现出良好的扩展性，显著提升了语音合成质量。

### 翻译

大型语言模型(LLMs)的最新进展已经改变了文本转语音(TTS)合成，启发了将语音表示为离散编解码器标记序列的自回归框架。其中，单码本TTS LLMs已成为紧凑且可流式传输的架构，能够联合建模语义和声学集成。然而，尽管这些模型效率高，但通常表现出不稳定的韵律、说话人漂移和降低的自然度。为解决这些问题，我们提出了一种多奖励组相对策略优化(GRPO)框架，直接优化单码本TTS LLMs的标记生成策略。除了标准的可懂度和说话人相似度目标外，我们的设计还集成了三个基于规则的奖励：用于持续时间一致性的长度惩罚，用于解码稳定性的熵正则化奖励，以及明确监督节奏的LLM注释的韵律对齐奖励。在这个韵律奖励中，外部推理LLM通过上下文学习预测多种可能的停顿结构，为GRPO训练提供符合人类偏好的监督信号。为评估通用性，我们在GRPO优化的AR骨干网络上进一步附加了一个流匹配(FM)解码器，观察到一致的额外收益，表明我们的强化优化增强了内在的AR策略。我们还进行了跨数据大小和模型规模的扩展性分析，揭示所提出的方法一致地提高了单码本TTS LLMs中的韵律稳定性、说话人相似度和整体语音自然度。


### 论文摘要

Recent advances in Large Language Models (LLMs) have transformed text-to-speech (TTS) synthesis, inspiring autoregressive frameworks that represent speech as sequences of discrete codec tokens. Among them, single-codebook TTS LLMs have emerged as compact and streamable architectures that jointly model semantic and acoustic integration. However, despite their efficiency, these models often exhibit unstable prosody, speaker drift, and degraded naturalness. To address these issues, we propose a multi-reward Group Relative Policy Optimization (GRPO) framework that directly optimizes the token generation policy of single-codebook TTS LLMs. Beyond standard intelligibility and speaker similarity objectives, our design integrates three rule-based rewards: a length penalty for duration consistency, an entropy regularization reward for decoding stability, and an LLM-annotated prosody alignment reward that explicitly supervises rhythm. In this prosody reward, an external reasoning LLM predicts multiple plausible pause structures via in-context learning, providing a human-preference-aligned supervisory signal for GRPO training. To assess universality, we further attach a flow-matching (FM) decoder on top of the GRPO-optimized AR backbone and observe consistent additional gains, indicating that our reinforcement optimization enhances the intrinsic AR policy. We further conduct a scalability analysis across data sizes and model scales, revealing that the proposed method consistently enhances prosodic stability, speaker similarity, and overall speech naturalness in single-codebook TTS LLMs.

---

## 64. Coupled Structural and Electronic Requirements in Alpha-FASnI3 Imposed by the Sn(II) Lone Pair

**论文链接:** [http://arxiv.org/abs/2511.21254v1](http://arxiv.org/abs/2511.21254v1)

**作者:** Mridhula Venkatanarayanan, Vladislav Slama, Madhubanti Mukherjee, Andrea Vezzosi, Ursula Rothlisberger, Virginia Carnevali

**发布时间:** 2025-11-26

**备注:** 30 pages (Supplementary Information (SI) included), 2 figures, 6 tables (SI), 4 tables (main)

### GPT解析

### 总结

本研究针对alpha-FASnI3（甲脒锡碘化物）的光伏应用进行了第一性原理建模，确定了可靠描述该材料所需的计算方法和结构模型。

### 背景

alpha-FASnI3是一种有前景的无铅光伏材料，在室温下具有近立方结构，但其稳定性受氧化驱动降解限制。文献中关于该材料的光伏相建模存在结构模型和理论水平不一致的问题。

### 目的

识别对alpha-FASnI3进行物理合理描述所需的结构和电子要求，理解其由Sn(II)孤对电子引起的伪扬-特勒不稳定性调控的行为机制。

### 方法

采用0 K弛豫、跨代码杂泛函基准测试和有限温度从头算分子动力学方法进行研究。

### 主要发现

1) 4x4x4超胞与随机取向的FA+阳离子是最小模型，可消除宏观偶极矩并保持立方对称性；2) 准确的能带边缘和能隙需要PBE0级别的杂泛函处理Sn相对论效应，结合非局域色散增强Sn-I共价性；3) 有限温度模拟显示Sn偏心保持局部性、<111>取向，对热波动具有鲁棒性；4) 复现实验300 K能隙需要6x6x6超胞。

### 结论

这些结果定义了可靠建模alpha-FASnI3的基本要素，为研究锡卤化物钙钛矿中孤对电子驱动的物理提供了严格基础。

### 翻译

甲脒锡碘化物（alpha-FASnI3）是无铅光伏应用的有力候选材料，在室温下采用近立方结构，但其稳定性受氧化驱动降解的限制。文献中关于光伏alpha相的第一性原理建模因结构模型和理论水平不一致而更加复杂。本研究确定了alpha-FASnI3物理合理描述所需的结构和电子要求，其行为由Sn(II)孤对电子引起的伪扬-特勒不稳定性调控。通过0 K弛豫、跨代码杂泛函基准测试和有限温度从头算分子动力学，我们证明4x4x4超胞与随机取向的FA+阳离子是最小模型，可消除宏观偶极矩，保持立方对称性，恢复局部八面体倾斜，并捕获特征PJT驱动的Sn偏心。准确的能带边缘和可靠的能隙需要PBE0级别的杂泛函处理Sn相对论效应，结合非局域色散(rVV10)来增强Sn-I共价性。有限温度模拟显示Sn偏心保持局部性、<111>取向，并对热波动具有鲁棒性，且复现实验300 K能隙需要6x6x6超胞。这些结果定义了可靠建模alpha-FASnI3的基本要素，为研究锡卤化物钙钛矿中孤对电子驱动的物理提供了严格基础。


### 论文摘要

Alpha-Formamidinium-tin-iodide (alpha-FASnI3) is a leading candidate for lead-free photovoltaic applications, adopting a nearly cubic structure at room temperature, but its stability remains limited by oxidation-driven degradation. Reliable first-principles modelling of the photovoltaic alpha-phase is further complicated by inconsistent structural models and levels of theory in the literature. Here, we identify the structural and electronic requirements needed for a physically sound description of alpha-FASnI3, whose behaviour is governed by a pseudo-Jahn-Teller (PJT) instability arising from the stereochemically active Sn(II) lone pair.   Using 0 K relaxations, cross-code hybrid-functional benchmarks, and finite-temperature ab initio molecular dynamics, we show that a 4x4x4 supercell with randomly oriented FA+ cations is the smallest model that removes macroscopic dipoles, preserves cubic symmetry, recovers local octahedral tilts, and captures the characteristic PJT-driven Sn off-centering. Accurate band edges and a reliable band gap require a PBE0-level hybrid functional with spin-orbit coupling to treat Sn relativistic effects, together with nonlocal dispersion (rVV10) to capture the enhanced Sn-I covalency. Finite-temperature simulations reveal that Sn off-centering remains local, <111>-oriented, and robust against thermal fluctuations, and that reproducing the experimental 300 K band gap requires a 6x6x6 supercell. These results define the essential ingredients for reliable modelling of alpha-FASnI3 and provide a rigorous foundation for studying lone-pair-driven physics in tin halide perovskites.

---

## 65. AVFakeBench: A Comprehensive Audio-Video Forgery Detection Benchmark for AV-LMMs

**论文链接:** [http://arxiv.org/abs/2511.21251v1](http://arxiv.org/abs/2511.21251v1)

**作者:** Shuhan Xia, Peipei Li, Xuannan Liu, Dongsen Zhang, Xinyu Guo, Zekun Li

**发布时间:** 2025-11-26

### GPT解析

### 总结

本研究引入了AVFakeBench，首个全面的音视频伪造检测基准，涵盖人类和一般主体的丰富伪造语义，包含12K个音视频问题，覆盖7种伪造类型和4个级别的注释。研究提出了多阶段混合伪造框架和多任务评估框架，评估了11个AV-LMMs和2种检测方法，展示了AV-LMMs作为新兴伪造检测器的潜力，同时揭示了其在细粒度感知和推理方面的弱点。

### 背景

音视频伪造威胁正迅速发展，超越以人为中心的深度伪造，扩展到复杂自然场景中的多样化操作。然而，现有基准测试仍局限于基于DeepFake的伪造和单一粒度注释，无法捕捉真实世界伪造场景的多样性和复杂性。

### 目的

引入首个全面的音视频伪造检测基准AVFakeBench，涵盖人类主体和一般主体之间的丰富伪造语义，解决现有基准测试的局限性，更好地反映真实世界中伪造场景的多样性和复杂性。

### 方法

AVFakeBench包含12,000个精心策划的音视频问题，涵盖7种伪造类型和4个级别的注释。提出多阶段混合伪造框架，整合专有模型进行任务规划和专家生成模型进行精确操作，确保高质量和多样化的伪造。建立多任务评估框架，包括二元判断、伪造类型分类、伪造细节选择和解释性推理。

### 主要发现

在AVFakeBench上评估了11个音视频大语言模型(AV-LMMs)和2种主流检测方法，展示了AV-LMMs作为新兴伪造检测器的潜力，同时揭示了其在细粒度感知和推理方面的显著弱点。

### 结论

AVFakeBench为音视频伪造检测提供了一个更全面的基准测试平台。AV-LMMs有潜力成为伪造检测工具，但在细粒度任务方面仍需改进。

### 翻译

音视频(AV)伪造的威胁正在迅速发展，超越以人为为中心的深度伪造，扩展到更复杂的自然场景中的多样化操作。然而，现有的基准测试仍然局限于基于DeepFake的伪造和单一粒度的注释，无法捕捉真实世界伪造场景的多样性和复杂性。为解决这一问题，我们引入了AVFakeBench，这是首个全面的音视频伪造检测基准，涵盖人类主体和一般主体之间的丰富伪造语义。AVFakeBench包含12,000个精心策划的音视频问题，涵盖7种伪造类型和4个级别的注释。为确保高质量和多样化的伪造，我们提出了多阶段混合伪造框架，整合专有模型进行任务规划，以及专家生成模型进行精确操作。该基准建立了多任务评估框架，包括二元判断、伪造类型分类、伪造细节选择和解释性推理。我们在AVFakeBench上评估了11个音视频大语言模型(AV-LMMs)和2种主流检测方法，展示了AV-LMMs作为新兴伪造检测器的潜力，同时揭示了其在细粒度感知和推理方面的显著弱点。


### 论文摘要

The threat of Audio-Video (AV) forgery is rapidly evolving beyond human-centric deepfakes to include more diverse manipulations across complex natural scenes. However, existing benchmarks are still confined to DeepFake-based forgeries and single-granularity annotations, thus failing to capture the diversity and complexity of real-world forgery scenarios. To address this, we introduce AVFakeBench, the first comprehensive audio-video forgery detection benchmark that spans rich forgery semantics across both human subject and general subject. AVFakeBench comprises 12K carefully curated audio-video questions, covering seven forgery types and four levels of annotations. To ensure high-quality and diverse forgeries, we propose a multi-stage hybrid forgery framework that integrates proprietary models for task planning with expert generative models for precise manipulation. The benchmark establishes a multi-task evaluation framework covering binary judgment, forgery types classification, forgery detail selection, and explanatory reasoning. We evaluate 11 Audio-Video Large Language Models (AV-LMMs) and 2 prevalent detection methods on AVFakeBench, demonstrating the potential of AV-LMMs as emerging forgery detectors while revealing their notable weaknesses in fine-grained perception and reasoning.

---

## 66. The effect of tip-speed ratio and free-stream turbulence on the coupled wind turbine blade/wake dynamics

**论文链接:** [http://arxiv.org/abs/2511.21206v1](http://arxiv.org/abs/2511.21206v1)

**作者:** Francisco J. G. de Oliveira, Martin Bourhis, Zahra Sharif Khodaei, Oliver R. H. Buxton

**发布时间:** 2025-11-26

**备注:** 34 pages, preprint submitted to WES

### GPT解析

### 总结

本研究实现了同时测量模型风力涡轮机叶片空间分布应变和尾流动力学的新技术，为气动弹性模型验证和疲劳感知控制策略提供了数据基础。

### 背景

风力涡轮机在风场中运行时，会经历由尾流诱导的速度亏损、增强湍流和变化运行条件引起的复杂空气动力学载荷。理解叶片对不同运行条件的结构响应与尾流中流动结构的关系，对预测疲劳损伤和优化涡轮性能至关重要。

### 目的

研究叶片结构响应与涡轮机尾流中产生的流动结构之间的关系，以预测疲劳损伤和优化涡轮机性能，并实现同时测量叶片应变和尾流动力学的方法。

### 方法

实施一种新技术，同时测量叶片应变和尾流动力学；使用直径1米的三叶片转子，配备分布式瑞利背散射光纤传感器；使用同步热线风速仪捕获多达4倍转子直径下游的尾流演化；在受控的自由流湍流和叶尖速比条件下进行21种不同参数条件的实验。

### 主要发现

空气动力学引起的应变波动在叶尖速比约为3.5时达到峰值，接近设计叶尖速比；在设计条件下，叶片叶尖处空气动力学驱动的应变波动贡献可达总波动应变的75%；频谱分析显示尾流结构与叶片响应之间存在频率选择性耦合，主要由与转子旋转频率动态相关的流动结构主导。

### 结论

新的实验方法和结果为未来气动弹性模型的验证和疲劳感知控制策略建立了数据驱动的基础。

### 翻译

风力涡轮机在风场中运行时，会经历由尾流诱导的速度亏损、增强湍流和变化运行条件引起的复杂空气动力学载荷。理解叶片对不同运行条件的结构响应与涡轮机尾流中产生的流动结构之间的关系，对于预测疲劳损伤和优化涡轮机性能至关重要。在本工作中，我们实现了一种新技术，能够在受控的自由流湍流和叶尖速比条件下，同时测量模型风力涡轮机的空间分布叶片应变和尾流动力学。直径1米的三叶片转子配备了分布式瑞利背散射光纤传感器，同时同步热线风速仪捕获了多达4倍转子直径下游的尾流演化。实验覆盖了广泛的自由流湍流和叶尖速比参数空间，总共进行了21种情况。结果表明，空气动力学引起的应变波动在叶尖速比约为3.5时达到峰值，接近设计叶尖速比，在设计条件下，叶片叶尖处空气动力学驱动的应变波动贡献可达总波动应变的75%。频谱分析显示尾流结构与叶片响应之间存在频率选择性耦合，主要由与转子旋转频率动态相关的流动结构主导。新的实验方法和结果为未来气动弹性模型的验证和疲劳感知控制策略建立了数据驱动的基础。


### 论文摘要

Wind turbines operating within wind farms experience complex aerodynamic loading arising from the interplay between wake-induced velocity deficits, enhanced turbulence, and varying operational conditions. Understanding the relationship between the blade's structural response to the different operating regimes and flow structures generated in the turbine's wake is critical for predicting fatigue damage and optimizing turbine performance. In this work, we implement a novel technique, allowing us to simultaneously measure spatially distributed blade strain and wake dynamics for a model wind turbine under controlled free-stream turbulence (FST) and tip-speed ratio ($λ$) conditions. A $1$ $\mathrm{m}$ diameter three-bladed rotor was instrumented with distributed Rayleigh backscattering fibre-optic sensors, while synchronised hot-wire anemometry captured wake evolution up to $4$ rotor diameters downstream. Experiments were conducted covering a wide $\{\mathrm{FST}, λ\}$ parameter space -- $21$ cases in total. Results reveal that aerodynamic-induced strain fluctuations peak at $λ\approx 3.5$, close to the design tip -speed ratio ($λ_d = 4$), with the blade's tip experiencing a contribution from the aerodynamically-driven strain fluctuations of up to $75\%$ of the total fluctuating strain at design conditions. Spectral analysis shows frequency-selective coupling between wake flow structures and the blade response, dominated by flow structures dynamically related to the rotor's rotating frequency (\textit{eg.} tip vortex structure). The novel experimental methodology and results establish a data-driven foundation for future aeroelastic models' validation, and fatigue-informed control strategies.

---

## 67. Transformer Driven Visual Servoing and Dual Arm Impedance Control for Fabric Texture Matching

**论文链接:** [http://arxiv.org/abs/2511.21203v1](http://arxiv.org/abs/2511.21203v1)

**作者:** Fuyuki Tokuda, Akira Seino, Akinari Kobayashi, Kai Tang, Kazuhiro Kosuge

**发布时间:** 2025-11-26

**备注:** 8 pages, 11 figures. Accepted to IEEE Robotics and Automation Letters (RA-L)

### GPT解析

### 总结

该论文提出了一种使用双臂机械臂和灰度相机对齐和放置织物件的方法，结合了基于Transformer的视觉伺服控制和双臂阻抗控制，通过DEAM模块提高姿态差异预测准确性，并在现实场景中实现了零样本部署。

### 背景

织物对齐和放置是机器人操作中的挑战性问题，需要精确匹配表面纹理。

### 目的

开发一种方法，将一块织物精确地放置在另一块织物上，使它们的表面纹理准确匹配。

### 方法

提出了一种新颖的控制方案，结合了基于Transformer的视觉伺服控制和双臂阻抗控制；开发了一个基于Transformer的网络，包含预训练骨干网络和新的差异提取注意力模块(DEAM)；使用渲染软件生成的合成图像进行训练，实现了在现实场景中的零样本部署。

### 主要发现

DEAM模块显著提高了姿态差异预测的准确性；系统能够同时控制织物件的姿态并将其放置在底层织物上，同时施加张力以保持织物平整；系统可以在不针对特定织物纹理进行预先训练的情况下，在现实场景中准确对齐不同纹理的织物件。

### 结论

所提出的系统在现实实验中能够准确对齐不同纹理的织物件。

### 翻译

在本文中，我们提出了一种使用双臂机械臂和灰度相机将一块织物放置在另一块织物上方并使其对齐的方法，以使它们的表面纹理精确匹配。我们提出了一种新颖的控制方案，将基于Transformer的视觉伺服控制与双臂阻抗控制相结合。这种方法使系统能够同时控制织物件的姿态并将其放置在底层织物上，同时施加张力以保持织物平整。我们的基于Transformer的网络集成了预训练骨干网络和一种新引入的差异提取注意力模块(DEAM)，显著提高了姿态差异预测的准确性。网络完全使用渲染软件生成的合成图像进行训练，能够在现实场景中实现零样本部署，无需针对特定织物纹理进行预先训练。现实实验证明，所提出的系统能够准确对齐不同纹理的织物件。


### 论文摘要

In this paper, we propose a method to align and place a fabric piece on top of another using a dual-arm manipulator and a grayscale camera, so that their surface textures are accurately matched. We propose a novel control scheme that combines Transformer-driven visual servoing with dualarm impedance control. This approach enables the system to simultaneously control the pose of the fabric piece and place it onto the underlying one while applying tension to keep the fabric piece flat. Our transformer-based network incorporates pretrained backbones and a newly introduced Difference Extraction Attention Module (DEAM), which significantly enhances pose difference prediction accuracy. Trained entirely on synthetic images generated using rendering software, the network enables zero-shot deployment in real-world scenarios without requiring prior training on specific fabric textures. Real-world experiments demonstrate that the proposed system accurately aligns fabric pieces with different textures.

---

## 68. Beyond URLs: Metadata Diversity and Position for Efficient LLM Pretraining

**论文链接:** [http://arxiv.org/abs/2511.21613v1](http://arxiv.org/abs/2511.21613v1)

**作者:** Dongyang Fan, Diba Hashemi, Sai Praneeth Karimireddy, Martin Jaggi

**发布时间:** 2025-11-26

### GPT解析

### 总结

本研究探讨了在大型语言模型预训练中整合多种元数据类型以提高训练效率和有效性的方法，发现了除URLs外的其他有效元数据形式，并提出了元数据追加和可学习元标记等创新方法。

### 背景

在大型语言模型预训练中整合元数据是一种很有前景的训练加速方法，但之前的研究只关注了URLs这一种有用信号，未探索其他元数据类型可能带来的更大好处。

### 目的

研究更广泛的元数据类型，探索它们是否能够为大型语言模型预训练带来更大的加速和效果提升。

### 方法

研究多种元数据类型的效果；引入元数据追加方法作为提高训练效率的手段；使用可学习的元标记(masked loss训练)来恢复部分加速；通过探测分析潜在表征以理解元数据如何塑造学习过程。

### 主要发现

其他类型的元数据(如文档质量的细粒度指标)也能加速预训练；有效元数据的共同特征是它们以更细的粒度编码信息；元数据追加方法(将预测合适的元数据作为辅助任务)可以加速预训练；可学习的元标记可以通过masked loss训练，诱导质量感知的潜在结构，从而恢复部分加速。

### 结论

这些结果为整合元数据以提高大型语言模型预训练的效率和有效性提供了实用的指导原则。

### 翻译

在大型语言模型预训练中整合元数据最近已成为一种有前景的加速训练方法。然而，先前的工作只强调了一种有用的信号——URL，留下了其他形式的元数据是否能带来更大好处的问题。在这项研究中，我们研究了更广泛的元数据类型，发现其他类型的元数据，如文档质量的细粒度指标，在添加时也能加速预训练。我们确定了有效元数据之间的共同特征：它们以更细的粒度编码信息。我们进一步引入了元数据追加作为提高训练效率的手段，其中预测合适的元数据作为辅助任务可以帮助加速预训练。此外，使用掩码损失训练的可学习元标记可以通过诱导质量感知的潜在结构来恢复部分加速。通过探测，我们分析了潜在表征以理解元数据如何塑造学习。总的来说，这些结果为整合元数据以提高大型语言模型预训练的效率和有效性提供了实用的指导原则。


### 论文摘要

Incorporating metadata in Large Language Models (LLMs) pretraining has recently emerged as a promising approach to accelerate training. However prior work highlighted only one useful signal-URLs, leaving open the question of whether other forms of metadata could yield greater benefits. In this study, we investigate a wider range of metadata types and find other types of metadata, such as fine-grained indicators of document quality that can also accelerate pretraining when prepended. We identify a common feature among effective metadata: they encode information at a finer granularity. We further introduce metadata appending as a means of improving training efficiency, where predicting an appropriate metadata as auxiliary task can help speed up pretraining. In addition, learnable meta-tokens trained with masked loss can recover part of the speedup by inducing quality-aware latent structure. Using probing, we analyze latent representations to understand how metadata shapes learning. Together, these results yield practical guidelines for integrating metadata to improve both the efficiency and effectiveness of LLM pretraining.

---

## 69. Lost in Time? A Meta-Learning Framework for Time-Shift-Tolerant Physiological Signal Transformation

**论文链接:** [http://arxiv.org/abs/2511.21500v1](http://arxiv.org/abs/2511.21500v1)

**作者:** Qian Hong, Cheng Bian, Xiao Zhou, Xiaoyu Li, Yelei Li, Zijing Zeng

**发布时间:** 2025-11-26

**备注:** The 40th Annual AAAI Conference on Artificial Intelligence (AAAI 26)

### GPT解析

### 总结

本研究提出ShiftSyncNet，一种基于元学习的双层优化框架，用于解决多模态生理信号转换中的时间错位问题，提高信号转换准确性。

### 背景

将非侵入式信号（如光电容积描记法PPG和心冲击图BCG）转换为临床上有意义的信号（如动脉血压ABP）对持续、低成本的医疗监测至关重要。然而，多模态信号转换中的时间错位会降低转换准确性，特别是在捕获ABP峰值等关键特征时。传统同步方法依赖强相似性假设或手动调整，而现有带噪声标签学习方法在时间偏移监督下效果不佳。

### 目的

解决多模态信号转换中的时间错位问题，提高信号转换的准确性，特别是在捕获关键特征如ABP峰值方面。

### 方法

提出ShiftSyncNet框架，由转换网络（TransNet）和时间偏移校正网络（SyncNet）组成。SyncNet学习训练对之间的时间偏移，并应用傅里叶相移来对齐监督信号，自动减轻时间错位导致的性能下降。

### 主要发现

在一个真实工业数据集和两个公共数据集上的实验表明，ShiftSyncNet分别比强基线方法提高了9.4%、6.0%和12.8%。结果突显了它在纠正时间偏移、提高标签质量和增强转换准确性方面的有效性。

### 结论

ShiftSyncNet为解决多模态生理转换中的时间不一致性问题提供了统一的方向，能够在各种错位场景下有效提高转换性能。

### 翻译

将非侵入式信号（如光电容积描记法和心冲击图）转换为临床上有意义的信号（如动脉血压）对于持续、低成本的医疗监测至关重要。然而，多模态信号转换中的时间错位会降低转换准确性，特别是在捕获动脉血压峰值等关键特征时。传统的同步方法通常依赖强相似性假设或手动调整，而现有的带噪声标签学习方法在时间偏移监督下效果不佳。为了解决这一挑战，我们提出了ShiftSyncNet，一种基于元学习的双层优化框架，能够自动减轻时间错位导致的性能下降。它由转换网络和时间偏移校正网络组成，其中校正网络学习训练对之间的时间偏移，并应用傅里叶相移来对齐监督信号。


### 论文摘要

Translating non-invasive signals such as photoplethysmography (PPG) and ballistocardiography (BCG) into clinically meaningful signals like arterial blood pressure (ABP) is vital for continuous, low-cost healthcare monitoring. However, temporal misalignment in multimodal signal transformation impairs transformation accuracy, especially in capturing critical features like ABP peaks. Conventional synchronization methods often rely on strong similarity assumptions or manual tuning, while existing Learning with Noisy Labels (LNL) approaches are ineffective under time-shifted supervision, either discarding excessive data or failing to correct label shifts. To address this challenge, we propose ShiftSyncNet, a meta-learning-based bi-level optimization framework that automatically mitigates performance degradation due to time misalignment. It comprises a transformation network (TransNet) and a time-shift correction network (SyncNet), where SyncNet learns time offsets between training pairs and applies Fourier phase shifts to align supervision signals. Experiments on one real-world industrial dataset and two public datasets show that ShiftSyncNet outperforms strong baselines by 9.4%, 6.0%, and 12.8%, respectively. The results highlight its effectiveness in correcting time shifts, improving label quality, and enhancing transformation accuracy across diverse misalignment scenarios, pointing toward a unified direction for addressing temporal inconsistencies in multimodal physiological transformation.

---

## 70. Evaluation of Large Language Models for Numeric Anomaly Detection in Power Systems

**论文链接:** [http://arxiv.org/abs/2511.21371v1](http://arxiv.org/abs/2511.21371v1)

**作者:** Yichen Liu, Hongyu Wu, Bo Liu

**发布时间:** 2025-11-26

### GPT解析

### 总结

本文评估了大型语言模型在电力系统数值异常检测中的应用，使用GPT-OSS-20B模型在IEEE 14-bus系统上测试了多种方法，并提出了基于三准则的规则感知设计。

### 背景

大型语言模型在电力系统中受到关注，异常检测对电网韧性至关重要，但LLMs在大规模数值数据上的异常检测性能尚未被充分探索。

### 目的

对电力系统中用于数值异常检测的大型语言模型进行全面评估。

### 方法

使用GPT-OSS-20B作为代表性模型，在IEEE 14-bus系统上评估，应用标准化提示框架包括零样本、少样本、上下文学习、低秩适应、微调和混合方法，采用基于三准则的规则感知设计，报告检测性能和推理质量。

### 主要发现

摘要中未明确提及具体发现结果。

### 结论

该研究为探索基于LLM的异常检测的局限性和能力及其与传统检测器在网络物理电网应用中的集成奠定了基础。

### 翻译

大型语言模型因其通用能力在电力系统中获得越来越多的关注。同时，异常检测对电网韧性仍然至关重要，需要基于多元遥测数据做出准确且可解释的决策。然而，大型语言模型在大规模数值数据上进行异常检测的性能在很大程度上尚未被探索。本文对电力系统中用于数值异常检测的大型语言模型进行了全面评估。我们使用GPT-OSS-20B作为代表性模型，并在IEEE 14-bus系统上对其进行评估。应用了标准化的提示框架，包括零样本、少样本、上下文学习、低秩适应、微调和混合大型语言模型-传统方法。我们采用基于三准则的规则感知设计，并报告了检测性能和推理质量。这项研究为进一步探索基于大型语言模型的异常检测的局限性和能力及其在网络物理电网应用中与传统检测器的集成奠定了基础。


### 论文摘要

Large language models (LLMs) have gained increasing attention in power grids for their general-purpose capabilities. Meanwhile, anomaly detection (AD) remains critical for grid resilience, requiring accurate and interpretable decisions based on multivariate telemetry. Yet the performance of LLMs on large-scale numeric data for AD remains largely unexplored. This paper presents a comprehensive evaluation of LLMs for numeric AD in power systems. We use GPT-OSS-20B as a representative model and evaluate it on the IEEE 14-bus system. A standardized prompt framework is applied across zero-shot, few-shot, in-context learning, low rank adaptation (LoRA), fine-tuning, and a hybrid LLM-traditional approach. We adopt a rule-aware design based on the three-sigma criterion, and report detection performance and rationale quality. This study lays the groundwork for further investigation into the limitations and capabilities of LLM-based AD and its integration with classical detectors in cyber-physical power grid applications.

---

## 71. Semantic Anchors in In-Context Learning: Why Small LLMs Cannot Flip Their Labels

**论文链接:** [http://arxiv.org/abs/2511.21038v1](http://arxiv.org/abs/2511.21038v1)

**作者:** Anantha Padmanaban Krishna Kumar

**发布时间:** 2025-11-26

**备注:** 13 pages total (7 pages main text, 3 pages references, 3 pages appendix), 2 figures, 14 tables. Code available at https://github.com/AnanthaPadmanaban-KrishnaKumar/semantic-anchors-icl

### GPT解析

### 总结

该研究探讨了上下文学习(ICL)是否能覆盖预训练标签语义，或仅是完善现有语义结构。通过对比自然演示和反转演示下的模型行为，研究发现ICL主要调整输入如何投射到预训练期间学习的稳定语义方向上，而非灵活重新映射标签含义。

### 背景

上下文学习(ICL)在大型语言模型中的应用日益广泛，但其是否能覆盖预训练标签语义或仅是完善已有语义结构仍存在疑问。

### 目的

探究ICL是能够覆盖预训练标签语义，还是仅仅完善已有的语义结构，以及ICL行为的基本限制。

### 方法

将大型语言模型视为提示诱导的分类器，对比自然演示(正确标签)和反转演示(翻转标签含义)下的行为；将ICL行为分解为三个对齐指标(真实对齐、先验对齐和提示对齐)；引入语义覆盖率指标；在8个分类任务和8个开源LLMs(1-120亿参数)上进行实验。

### 主要发现

支持语义锚点观点的一致证据；使用自然演示时，ICL提高准确性同时保持强先验对齐；大多数正确预测与零样本行为一致；使用反转演示时，模型无法学习连贯的反语义分类器；语义覆盖率在少样本1-120亿设置中保持为零。

### 结论

ICL不能灵活地重新映射标签含义，而是调整输入如何投射到预训练期间学习的稳定语义方向上；这澄清了少样本提示的基本限制，表明在这些规模上覆盖标签语义需要超越ICL的干预。

### 翻译

上下文学习(ICL能否覆盖预训练标签语义，或者它仅仅是完善现有的语义主干？我们通过将大型语言模型视为提示诱导的分类器，并对比其在自然演示(带有正确标签)和反转演示(系统性地翻转标签含义)下的行为来解决这个问题。我们将ICL行为分解为三个对齐指标(真实对齐、先验对齐和提示对齐)，并引入了语义覆盖率，定义为在翻转语义下的正确率。在八个分类任务和八个开源LLMs(1-120亿参数)中，我们发现了一致支持语义锚点观点的证据。使用自然演示时，ICL提高准确性同时保持强先验对齐；大多数正确预测与零样本行为一致，即使先验较弱。使用反转演示时，模型无法学习连贯的反语义分类器：提示对齐仅通过牺牲准确性来提高，语义覆盖率在我们的少样本1-120亿设置中保持为零。ICL不能灵活地重新映射标签含义，而是主要调整输入如何投射到预训练期间学习的稳定语义方向上，这澄清了少样本提示的基本限制，并表明在这些规模上覆盖标签语义需要超越ICL的干预。所有代码可在https://github.com/AnanthaPadmanaban-KrishnaKumar/semantic-anchors-icl获取。


### 论文摘要

Can in-context learning (ICL) override pre-trained label semantics, or does it merely refine an existing semantic backbone? We address this question by treating LLMs as prompt-induced classifiers and contrasting their behavior under \emph{natural} demonstrations (with correct labels) and \emph{inverted} demonstrations (systematically flipping label meanings). We decompose ICL behavior into three alignment metrics (truth, prior, and prompt alignment) and introduce a semantic override rate, defined as correctness under flipped semantics. Across eight classification tasks and eight open-source LLMs (1--12B parameters), we find consistent evidence for a semantic anchor view. With natural demonstrations, ICL improves accuracy while maintaining strong prior alignment; most correct predictions coincide with zero-shot behavior, even when the prior is weak. With inverted demonstrations, models cannot learn coherent anti-semantic classifiers: prompt alignment increases only by sacrificing accuracy, and semantic override rates remain exactly zero in our few-shot 1--12B setting. Rather than flexibly remapping label meanings, ICL primarily adjusts how inputs project onto stable semantic directions learned during pre-training, clarifying fundamental limits of few-shot prompting and suggesting that overriding label semantics at these scales requires interventions beyond ICL. All code is available at: https://github.com/AnanthaPadmanaban-KrishnaKumar/semantic-anchors-icl.

---

## 72. Even with AI, Bijection Discovery is Still Hard: The Opportunities and Challenges of OpenEvolve for Novel Bijection Construction

**论文链接:** [http://arxiv.org/abs/2511.20987v1](http://arxiv.org/abs/2511.20987v1)

**作者:** Davis Brown, Jesse He, Helen Jenne, Henry Kvinge, Max Vargas

**发布时间:** 2025-11-26

**备注:** 16 pages, 3 figures. This is an extended abstract submitted to FPSAC 2026

### GPT解析

### 总结

本文探讨了使用进化程序合成系统（如OpenEvolve）来发现组合双射的方法，评估了这些系统在解决数学研究问题上的能力和局限性。

### 背景

进化程序合成系统如AlphaEvolve、OpenEvolve和ShinkaEvolve利用大型语言模型团队生成可读代码形式的候选解决方案，并通过'进化'过程改进这些解决方案。现有数学应用主要集中在建立边界问题上，而程序合成方法适用于解决方案以显式构造形式出现的问题。

### 目的

探索使用OpenEvolve进行组合双射发现的可能性，并评估这些系统在解决数学研究问题上的能力和局限性。

### 方法

将OpenEvolve系统应用于三个涉及Dyck路径的双射构造问题，其中两个是已知问题，一个是开放性问题。系统通过生成候选代码解决方案并对其进行进化改进来工作。

### 主要发现

虽然像OpenEvolve这样的系统显示出作为组合学家有价值工具的潜力，但发现新颖的、研究级的双射对当前前沿系统来说仍然是一项具有挑战性的任务。

### 结论

进化程序合成系统在数学发现方面有潜力，但在解决复杂研究级数学问题上仍存在局限性，需要人类数学家的参与和指导。

### 翻译

进化程序合成系统如AlphaEvolve、OpenEvolve和ShinkaEvolve为AI辅助数学发现提供了新方法。这些系统利用大型语言模型团队生成可读代码形式的候选解决方案，然后通过'进化'过程改进这些候选解决方案，使其超越LLM单次生成的能力。虽然现有的数学应用主要集中在建立边界问题上（如球体填充问题），但程序合成方法适用于任何解决方案以显式构造形式出现的问题。鉴于此，在本文中，我们探索使用OpenEvolve进行组合双射发现。我们描述了将OpenEvolve应用于涉及Dyck路径的三个双射构造问题的结果，其中两个是已知的，一个是开放性问题。我们发现，虽然像OpenEvolve这样的系统显示出作为组合学家有价值工具的潜力，但发现新颖的、研究级双射对当前前沿系统来说仍然是一项具有挑战性的任务，这强化了人类数学家在研究循环中的必要性。我们为有兴趣探索使用这些系统的领域内人士描述了一些经验教训。


### 论文摘要

Evolutionary program synthesis systems such as AlphaEvolve, OpenEvolve, and ShinkaEvolve offer a new approach to AI-assisted mathematical discovery. These systems utilize teams of large language models (LLMs) to generate candidate solutions to a problem as human readable code. These candidate solutions are then 'evolved' with the goal of improving them beyond what an LLM can produce in a single shot. While existing mathematical applications have mostly focused on problems of establishing bounds (e.g., sphere packing), the program synthesis approach is well suited to any problem where the solution takes the form of an explicit construction. With this in mind, in this paper we explore the use of OpenEvolve for combinatorial bijection discovery. We describe the results of applying OpenEvolve to three bijection construction problems involving Dyck paths, two of which are known and one of which is open. We find that while systems like OpenEvolve show promise as a valuable tool for combinatorialists, the problem of finding novel, research-level bijections remains a challenging task for current frontier systems, reinforcing the need for human mathematicians in the loop. We describe some lessons learned for others in the field interested in exploring the use of these systems.

---

## 73. NOIR 2.0: Neural Signal Operated Intelligent Robots for Everyday Activities

**论文链接:** [http://arxiv.org/abs/2511.20848v1](http://arxiv.org/abs/2511.20848v1)

**作者:** Tasha Kim, Yingke Wang, Hanvit Cho, Alex Hodges

**发布时间:** 2025-11-25

**备注:** Conference on Robot Learning (CoRL 2024), CoRoboLearn

### GPT解析

### 总结

NOIR系统是一种多功能脑-机器人接口，允许人类使用脑信号控制机器人完成日常任务。

### 背景

NOIR系统利用脑电图(EEG)将人类对特定对象和期望意图的脑信号直接转化为机器人可执行的命令。

### 目的

介绍NOIR 2.0，这是NOIR的增强版本，旨在改进脑信号解码和机器人学习算法。

### 方法

NOIR 2.0包含更快更准确的脑解码算法，采用少样本机器人学习算法来适应用户并预测其意图，利用基础模型进行更高效的样本学习和适应。

### 主要发现

新算法将任务完成时间减少了46%，将整体人类时间减少了65%（从15次演示减少到单次演示）。

### 结论

NOIR 2.0通过改进算法显著提高了脑-机器人接口的效率和用户适应性。

### 翻译

神经信号控制智能机器人(NOIR)系统是一种多功能的脑-机器人接口，允许人类使用他们的脑信号控制机器人完成日常任务。该接口利用脑电图(EEG)将人类关于特定对象和期望动作的意图直接转化为机器人可执行的命令。我们介绍了NOIR 2.0，这是NOIR的增强版本。NOIR 2.0包括更快更准确的脑解码算法，将任务完成时间减少了46%。NOIR 2.0使用少样本机器人学习算法来适应个体用户并预测他们的意图。新的学习算法利用基础模型进行更高效的样本学习和适应（15次演示与单次演示相比），显著减少了65%的整体人类时间。


### 论文摘要

Neural Signal Operated Intelligent Robots (NOIR) system is a versatile brain-robot interface that allows humans to control robots for daily tasks using their brain signals. This interface utilizes electroencephalography (EEG) to translate human intentions regarding specific objects and desired actions directly into commands that robots can execute. We present NOIR 2.0, an enhanced version of NOIR. NOIR 2.0 includes faster and more accurate brain decoding algorithms, which reduce task completion time by 46%. NOIR 2.0 uses few-shot robot learning algorithms to adapt to individual users and predict their intentions. The new learning algorithms leverage foundation models for more sample-efficient learning and adaptation (15 demos vs. a single demo), significantly reducing overall human time by 65%.

---

## 74. Vision-Language Enhanced Foundation Model for Semi-supervised Medical Image Segmentation

**论文链接:** [http://arxiv.org/abs/2511.19759v2](http://arxiv.org/abs/2511.19759v2)

**作者:** Jiaqi Guo, Mingzhen Li, Hanyu Su, Santiago López, Lexiaozi Fan, Daniel Kim, Aggelos Katsaggelos

**发布时间:** 2025-11-24

### GPT解析

### 总结

该研究提出了一种名为VESSA的视觉语言增强半监督分割助手，将视觉语言模型的基础视觉-语义理解整合到半监督学习框架中，用于医学图像分割，显著提高了在有限标注条件下的分割准确性。

### 背景

半监督学习(SSL)已成为医学图像分割的有效范式，减少了对大量专家标注的依赖。同时，视觉语言模型(VLMs)在多个视觉领域展示了强大的泛化和少样本能力。

### 目的

将基于VLM的分割整合到半监督医学图像分割中，通过引入视觉语言增强的半监督分割助手(VESSA)，将基础级的视觉-语义理解整合到SSL框架中，提高分割准确性，特别是在标注数据有限的情况下。

### 方法

该方法包含两个阶段：第一阶段，VLM增强的分割基础模型VESSA使用包含黄金标准样本的模板库作为参考引导的分割助手进行训练；给定输入-模板对，VESSA执行视觉特征匹配，从样本分割中提取代表性的语义和空间线索，为受SAM2启发的掩码解码器生成结构化提示。第二阶段，VESSA被集成到最先进的SSL框架中，与学生模型实现动态交互：随着学生预测变得更加精细，它们被反馈给VESSA作为提示，使VESSA能够生成更高质量的伪标签和更强的指导。

### 主要发现

在多个分割数据集和领域进行的广泛实验表明，VESSA增强的SSL显著提高了分割准确性，在极有限的标注条件下优于最先进的基线方法。

### 结论

VESSA增强的SSL显著提升了医学图像分割的准确性，特别是在标注数据极其有限的条件下，为医学图像分割领域提供了一种新的有效方法。

### 翻译

半监督学习(SSL)已成为医学图像分割的有效范式，减少了对大量专家标注的依赖。同时，视觉语言模型(VLMs)在多个视觉领域展示了强大的泛化和少样本能力。在本工作中，我们将基于VLM的分割整合到半监督医学图像分割中，通过引入一个视觉语言增强的半监督分割助手(VESSA)，将基础级的视觉-语义理解整合到SSL框架中。我们的方法包含两个阶段。第一阶段，VLM增强的分割基础模型VESSA使用包含黄金标准样本的模板库作为参考引导的分割助手进行训练，模拟从有限标注数据中学习。给定输入-模板对，VESSA执行视觉特征匹配，从样本分割中提取代表性的语义和空间线索，为受SAM2启发的掩码解码器生成结构化提示，以生成分割掩码。第二阶段，VESSA被集成到最先进的SSL框架中，与学生模型实现动态交互：随着学生预测变得更加精细，它们被反馈给VESSA作为提示，使VESSA能够生成更高质量的伪标签和更强的指导。在多个分割数据集和领域进行的广泛实验表明，VESSA增强的SSL显著提高了分割准确性，在极有限的标注条件下优于最先进的基线方法。


### 论文摘要

Semi-supervised learning (SSL) has emerged as an effective paradigm for medical image segmentation, reducing the reliance on extensive expert annotations. Meanwhile, vision-language models (VLMs) have demonstrated strong generalization and few-shot capabilities across diverse visual domains. In this work, we integrate VLM-based segmentation into semi-supervised medical image segmentation by introducing a Vision-Language Enhanced Semi-supervised Segmentation Assistant (VESSA) that incorporates foundation-level visual-semantic understanding into SSL frameworks. Our approach consists of two stages. In Stage 1, the VLM-enhanced segmentation foundation model VESSA is trained as a reference-guided segmentation assistant using a template bank containing gold-standard exemplars, simulating learning from limited labeled data. Given an input-template pair, VESSA performs visual feature matching to extract representative semantic and spatial cues from exemplar segmentations, generating structured prompts for a SAM2-inspired mask decoder to produce segmentation masks. In Stage 2, VESSA is integrated into a state-of-the-art SSL framework, enabling dynamic interaction with the student model: as student predictions become more refined, they are fed back to VESSA as prompts, allowing it to generate higher-quality pseudo-labels and stronger guidance. Extensive experiments across multiple segmentation datasets and domains show that VESSA-augmented SSL significantly enhances segmentation accuracy, outperforming state-of-the-art baselines under extremely limited annotation conditions.

---

## 75. G$^2$VLM: Geometry Grounded Vision Language Model with Unified 3D Reconstruction and Spatial Reasoning

**论文链接:** [http://arxiv.org/abs/2511.21688v1](http://arxiv.org/abs/2511.21688v1)

**作者:** Wenbo Hu, Jingli Lin, Yilin Long, Yunlong Ran, Lihan Jiang, Yifan Wang, Chenming Zhu, Runsen Xu, Tai Wang, Jiangmiao Pang

**发布时间:** 2025-11-26

**备注:** code are released at https://github.com/InternRobotics/G2VLM

### GPT解析

### 总结

G²VLM是一种基于几何的视觉-语言模型，通过结合3D视觉几何特征与视觉-语言模型，解决了现有模型在空间智能方面的不足，在空间3D重建和空间理解任务上取得了优异表现。

### 背景

视觉-语言模型在空间智能方面缺乏鲁棒性，在空间理解和推理任务上表现不佳。

### 目的

开发一种能够同时处理空间3D重建和空间理解的模型，弥补现有模型在空间智能方面的缺陷。

### 方法

提出G²VLM，利用学习的3D视觉几何特征直接预测3D属性，通过上下文学习和交错推理增强空间推理任务，在丰富的多视图图像和视频数据上训练，同时利用3D视觉先验。

### 主要发现

G²VLM在空间3D重建和空间理解两项任务上都表现出色，与最先进的前馈3D重建模型相当，在空间理解和推理任务上取得了更好或具有竞争力的结果。

### 结论

通过将语义强大的视觉-语言模型与低级3D视觉任务统一起来，G²VLM为社区提供了强大的基线，并解锁了更多未来应用的可能性，如3D场景编辑。

### 翻译

视觉-语言模型在空间智能方面仍缺乏鲁棒性，在空间理解和推理任务上表现不佳。我们将这一差距归因于缺乏能够从2D图像重建3D空间的视觉几何学习过程。我们提出了G²VLM，一种基于几何的视觉-语言模型，它连接了空间智能的两个基本方面：空间3D重建和空间理解。G²VLM原生地利用学习的3D视觉几何特征直接预测3D属性，并通过上下文学习和交错推理增强空间推理任务。我们的统一设计在空间理解方面具有高度可扩展性：它在丰富的多视图图像和视频数据上训练，同时利用通常只能从难以收集的注释中获得的3D视觉先验的好处。实验结果表明，G²VLM在两项任务上都表现出色，与最先进的前馈3D重建模型相当，并在空间理解和推理任务上取得了更好或具有竞争力的结果。通过将语义强大的视觉-语言模型与低级3D视觉任务统一起来，我们希望G²VLM能成为社区的强大基线，并解锁更多未来应用，如3D场景编辑。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决视觉语言模型(VLMs)在空间智能方面的不足，特别是在空间理解和推理任务上的表现不佳。这个问题很重要，因为真正的空间智能对于机器人技术、具身AI以及需要理解3D环境的AI应用至关重要。现有模型缺乏从2D图像重建3D空间的视觉几何学习过程，无法将2D感知提升为连贯的3D世界表示，限制了它们在复杂空间任务中的表现。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': "作者受到人类认知中'双流假说'的启发，该假说认为人类视觉系统存在'腹侧流'(物体识别)和'背侧流'(空间位置处理)两个通路。基于这一理论，他们设计了G²VLM，采用Mixture-of-Transformer-Experts架构，包含几何感知专家和语义感知专家两个组件。作者借鉴了VGGT、π3等前馈3D重建模型的设计理念，但将它们整合到统一的VLM框架中；参考了MapAnything等视觉几何模型，但采用了适合LLM框架的全局注意力机制；还采用了两阶段训练策略，先训练几何专家，再联合训练两个专家。", '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是将3D视觉几何学习与视觉语言模型统一在一个框架内，使模型能够从2D图像中直接预测3D属性，并通过上下文学习和交错推理增强空间推理。整体实现流程包括：1)模型架构采用双专家设计，几何感知专家使用DINOv2编码器预测3D属性，语义感知专家基于预训练VLM进行多模态理解；2)训练分为两阶段，第一阶段冻结语义专家，训练几何专家在3D标注数据上学习几何特征；第二阶段解冻语义专家，与几何专家在空间理解数据上联合训练；3)推理时，几何专家预测3D属性，语义专家利用这些特征进行空间理解和推理，通过交错推理解决复杂空间问题。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)首次将空间3D重建和高层次空间理解统一在一个视觉语言模型中；2)采用双专家架构(几何感知专家和语义感知专家)并通过共享自注意力机制实现交互；3)设计了两阶段训练策略，使模型能从纯2D图像学习3D几何；4)针对几何专家设计了适合LLM框架的全局注意力机制。相比之前工作，G²VLM不同于标准VLM(缺乏3D几何学习)、传统3D重建模型(忽视高层次理解)和其他空间VLM(将图像视为2D数据)，它直接整合几何专家，提供更自然的对齐，统一了视觉几何预测与空间推理任务。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'G²VLM通过统一3D重建与空间理解于一个视觉语言模型框架中，首次实现了从2D图像直接预测3D几何并增强空间推理能力，显著提升了模型在空间智能任务上的表现。'}


### 论文摘要

Vision-Language Models (VLMs) still lack robustness in spatial intelligence, demonstrating poor performance on spatial understanding and reasoning tasks. We attribute this gap to the absence of a visual geometry learning process capable of reconstructing 3D space from 2D images. We present G$^2$VLM, a geometry grounded vision-language model that bridges two fundamental aspects of spatial intelligence: spatial 3D reconstruction and spatial understanding. G$^2$VLM natively leverages learned 3D visual geometry features to directly predict 3D attributes and enhance spatial reasoning tasks via in-context learning and interleaved reasoning. Our unified design is highly scalable for spatial understanding: it trains on abundant multi-view image and video data, while simultaneously leveraging the benefits of 3D visual priors that are typically only derived from hard-to-collect annotations. Experimental results demonstrate G$^2$VLM is proficient in both tasks, achieving comparable results to state-of-the-art feed-forward 3D reconstruction models and achieving better or competitive results across spatial understanding and reasoning tasks. By unifying a semantically strong VLM with low-level 3D vision tasks, we hope G$^2$VLM can serve as a strong baseline for the community and unlock more future applications, such as 3D scene editing.

---

## 76. CaFlow: Enhancing Long-Term Action Quality Assessment with Causal Counterfactual Flow

**论文链接:** [http://arxiv.org/abs/2511.21653v1](http://arxiv.org/abs/2511.21653v1)

**作者:** Ruisheng Han, Kanglei Zhou, Shuang Chen, Amir Atapour-Abarghouei, Hubert P. H. Shum

**发布时间:** 2025-11-26

### GPT解析

### 总结

本文提出了一种名为CaFlow的统一框架，用于解决长期动作质量评估(AQA)中的挑战，该框架结合了反事实去混杂与双向时间条件流技术。

### 背景

动作质量评估(AQA)是从动作视频中预测细粒度执行分数的技术，广泛应用于体育、康复和技能评估领域。长期AQA(如花样滑冰或艺术体操)特别具有挑战性，因为它需要建模扩展的时间动态，同时保持对上下文混淆因素的鲁棒性。

### 目的

开发一个统一的框架来解决长期AQA中的挑战，避免现有方法对昂贵标注的依赖和单向时间建模的局限性。

### 方法

提出CaFlow框架，包含两个核心模块：1) Causal Counterfactual Regularization (CCR)模块，以自监督方式解纠缠因果和混杂特征，并通过反事实干预强制因果鲁棒性；2) BiT-Flow模块，使用循环一致性约束建模前向和后向动态，产生更平滑、更连贯的表示。

### 主要发现

在多个长期AQA基准上的广泛实验表明，CaFlow达到了最先进的性能，有效解决了现有方法的局限性。

### 结论

CaFlow是一个有效的框架，能够解决长期AQA中的挑战，通过整合反事实去混杂与双向时间条件流技术，提高了动作质量评估的准确性和鲁棒性。

### 翻译

动作质量评估(AQA)从动作视频中预测细粒度执行分数，广泛应用于体育、康复和技能评估。长期AQA，如花样滑冰或艺术体操，尤其具有挑战性，因为它需要建模扩展的时间动态，同时保持对上下文混淆因素的鲁棒性。现有方法要么依赖昂贵的标注，要么依赖单向时间建模，使它们容易受到虚假相关的影响，并且长期表示不稳定。为此，我们提出了CaFlow，一个整合了反事实去混杂与双向时间条件流的统一框架。因果反事实正则化(CCR)模块以自监督方式解纠缠因果和混杂特征，并通过反事实干预强制因果鲁棒性，而BiT-Flow模块使用循环一致性约束建模前向和后向动态，以产生更平滑、更连贯的表示。在多个长期AQA基准上的广泛实验表明，CaFlow达到了最先进的性能。代码可在https://github.com/Harrison21/CaFlow获取。


### 论文摘要

Action Quality Assessment (AQA) predicts fine-grained execution scores from action videos and is widely applied in sports, rehabilitation, and skill evaluation. Long-term AQA, as in figure skating or rhythmic gymnastics, is especially challenging since it requires modeling extended temporal dynamics while remaining robust to contextual confounders. Existing approaches either depend on costly annotations or rely on unidirectional temporal modeling, making them vulnerable to spurious correlations and unstable long-term representations. To this end, we propose CaFlow, a unified framework that integrates counterfactual de-confounding with bidirectional time-conditioned flow. The Causal Counterfactual Regularization (CCR) module disentangles causal and confounding features in a self-supervised manner and enforces causal robustness through counterfactual interventions, while the BiT-Flow module models forward and backward dynamics with a cycle-consistency constraint to produce smoother and more coherent representations. Extensive experiments on multiple long-term AQA benchmarks demonstrate that CaFlow achieves state-of-the-art performance. Code is available at https://github.com/Harrison21/CaFlow

---

## 77. MoGAN: Improving Motion Quality in Video Diffusion via Few-Step Motion Adversarial Post-Training

**论文链接:** [http://arxiv.org/abs/2511.21592v1](http://arxiv.org/abs/2511.21592v1)

**作者:** Haotian Xue, Qi Chen, Zhonghao Wang, Xun Huang, Eli Shechtman, Jinrong Xie, Yongxin Chen

**发布时间:** 2025-11-26

### GPT解析

### 总结

本文提出MoGAN，一个以运动为中心的后训练框架，用于改善视频扩散模型的运动连贯性和真实性，无需奖励模型或人类偏好数据。

### 背景

视频扩散模型在帧级保真度方面表现良好，但在运动连贯性、动态和真实性方面存在不足，常产生抖动、重影或不合理的动态效果。标准去噪MSE目标函数对时间一致性缺乏直接监督，导致模型在低损失情况下仍生成较差的运动效果。

### 目的

开发一个能够显著提高视频生成质量中运动真实性的方法，同时保持视觉保真度和生成效率。

### 方法

MoGAN基于三步蒸馏视频扩散模型构建，通过训练一个基于DiT的光流判别器来区分真实与生成的运动，并结合分布匹配正则化器以保持视觉保真度。

### 主要发现

在Wan2.1-T2V-1.3B模型上的实验表明，MoGAN在多个基准测试中显著提高了运动质量：在VBench上比50步教师模型提高7.3%，比3步DMD模型提高13.3%；在VideoJAM-Bench上比教师模型提高7.4%，比DMD提高8.8%，同时保持相当或更好的美学和图像质量。人类研究确认MoGAN在运动质量方面更受青睐。

### 结论

MoGAN能够在不牺牲视觉保真度或效率的情况下，提供更真实的运动效果，为快速、高质量的视频生成提供了实用路径。

### 翻译

视频扩散模型在帧级保真度方面表现出色，但在运动连贯性、动态和真实性方面仍然存在困难，常常产生抖动、重影或不合理的动态效果。一个主要限制是标准的去噪MSE目标函数对时间一致性没有直接监督，允许模型在低损失的情况下仍然生成较差的运动。我们提出了MoGAN，一个以运动为中心的后训练框架，可以在没有奖励模型或人类偏好数据的情况下提高运动的真实性。基于一个三步蒸馏的视频扩散模型，我们训练了一个基于DiT的光流判别器来区分真实和生成的运动，并结合了分布匹配正则化器以保持视觉保真度。在Wan2.1-T2V-1.3B上的实验表明，MoGAN在多个基准测试中显著提高了运动质量。在VBench上，MoGAN比50步教师模型提高运动分数7.3%，比3步DMD模型提高13.3%。在VideoJAM-Bench上，MoGAN比教师模型提高运动分数7.4%，比DMD提高8.8%，同时保持相当甚至更好的美学和图像质量分数。一项人类研究进一步证实，MoGAN在运动质量方面更受欢迎（与教师模型相比52%对38%；与DMD相比56%对29%）。总体而言，MoGAN在不牺牲视觉保真度或效率的情况下，提供了显著更真实的运动，为快速、高质量的视频生成提供了实用途径。项目网页是：https://xavihart.github.io/mogan。


### 论文摘要

Video diffusion models achieve strong frame-level fidelity but still struggle with motion coherence, dynamics and realism, often producing jitter, ghosting, or implausible dynamics. A key limitation is that the standard denoising MSE objective provides no direct supervision on temporal consistency, allowing models to achieve low loss while still generating poor motion. We propose MoGAN, a motion-centric post-training framework that improves motion realism without reward models or human preference data. Built atop a 3-step distilled video diffusion model, we train a DiT-based optical-flow discriminator to differentiate real from generated motion, combined with a distribution-matching regularizer to preserve visual fidelity. With experiments on Wan2.1-T2V-1.3B, MoGAN substantially improves motion quality across benchmarks. On VBench, MoGAN boosts motion score by +7.3% over the 50-step teacher and +13.3% over the 3-step DMD model. On VideoJAM-Bench, MoGAN improves motion score by +7.4% over the teacher and +8.8% over DMD, while maintaining comparable or even better aesthetic and image-quality scores. A human study further confirms that MoGAN is preferred for motion quality (52% vs. 38% for the teacher; 56% vs. 29% for DMD). Overall, MoGAN delivers significantly more realistic motion without sacrificing visual fidelity or efficiency, offering a practical path toward fast, high-quality video generation. Project webpage is: https://xavihart.github.io/mogan.

---

## 78. Harmony: Harmonizing Audio and Video Generation through Cross-Task Synergy

**论文链接:** [http://arxiv.org/abs/2511.21579v1](http://arxiv.org/abs/2511.21579v1)

**作者:** Teng Hu, Zhentao Yu, Guozhen Zhang, Zihan Su, Zhengguang Zhou, Youliang Zhang, Yuan Zhou, Qinglin Lu, Ran Yi

**发布时间:** 2025-11-26

### GPT解析

### 总结

该研究提出了Harmony框架，解决了生成式AI中同步音视频内容合成的关键挑战，通过跨任务协同训练、全局-局部解耦交互模块和同步增强的CFG方法，显著提升了音视频对齐效果。

### 背景

生成式AI中同步音视频内容的合成是一个关键挑战，开源模型在稳健的音视频对齐方面面临困难。

### 目的

克服联合扩散过程中的三个基本挑战：对应漂移、低效的全局注意力机制和传统无分类器引导的模态内偏差，实现高质量的音视频同步生成。

### 方法

引入Harmony框架，包括：(1)跨任务协同训练范式，利用音频驱动视频和视频驱动音频任务的监督信号减轻漂移；(2)全局-局部解耦交互模块，实现高效精确的时序-风格对齐；(3)同步增强的CFG，在推理过程中隔离和放大对齐信号。

### 主要发现

音视频对齐问题源于三个基本挑战：对应漂移阻碍稳定学习、低效的全局注意力机制无法捕捉细粒度时序线索、传统CFG增强条件性但不改善跨模态同步。

### 结论

Harmony框架建立了新的最先进水平，在生成保真度和细粒度音视频同步方面显著优于现有方法。

### 翻译

同步音视频内容的合成是生成式AI中的一个关键挑战，开源模型在稳健的音视频对齐方面面临挑战。我们的分析表明，这个问题源于联合扩散过程的三个基本挑战：(1)对应漂移，其中同时演化的噪声潜在表示阻碍了对齐的稳定学习；(2)低效的全局注意力机制，无法捕捉细粒度的时序线索；(3)传统无分类器引导(CFG)的模态内偏差，它增强了条件性但没有改善跨模态同步。为了克服这些挑战，我们引入了Harmony，一个强制执行音视频同步的新框架。我们首先提出了跨任务协同训练范式，通过利用音频驱动的视频和视频驱动的音频生成任务中的强监督信号来减轻漂移。然后，我们设计了一个全局-局部解耦交互模块，用于高效精确的时序-风格对齐。最后，我们提出了一个新颖的同步增强CFG(SyncCFG)，在推理过程中明确隔离和放大对齐信号。大量实验表明，Harmony建立了新的最先进水平，在生成保真度和关键地实现细粒度音视频同步方面显著优于现有方法。


### 论文摘要

The synthesis of synchronized audio-visual content is a key challenge in generative AI, with open-source models facing challenges in robust audio-video alignment. Our analysis reveals that this issue is rooted in three fundamental challenges of the joint diffusion process: (1) Correspondence Drift, where concurrently evolving noisy latents impede stable learning of alignment; (2) inefficient global attention mechanisms that fail to capture fine-grained temporal cues; and (3) the intra-modal bias of conventional Classifier-Free Guidance (CFG), which enhances conditionality but not cross-modal synchronization. To overcome these challenges, we introduce Harmony, a novel framework that mechanistically enforces audio-visual synchronization. We first propose a Cross-Task Synergy training paradigm to mitigate drift by leveraging strong supervisory signals from audio-driven video and video-driven audio generation tasks. Then, we design a Global-Local Decoupled Interaction Module for efficient and precise temporal-style alignment. Finally, we present a novel Synchronization-Enhanced CFG (SyncCFG) that explicitly isolates and amplifies the alignment signal during inference. Extensive experiments demonstrate that Harmony establishes a new state-of-the-art, significantly outperforming existing methods in both generation fidelity and, critically, in achieving fine-grained audio-visual synchronization.

---

## 79. HarmonicAttack: An Adaptive Cross-Domain Audio Watermark Removal

**论文链接:** [http://arxiv.org/abs/2511.21577v1](http://arxiv.org/abs/2511.21577v1)

**作者:** Kexin Li, Xiao Hu, Ilya Grishchenko, David Lie

**发布时间:** 2025-11-26

### GPT解析

### 总结

该研究提出了一种名为HarmonicAttack的高效音频水印移除方法，用于评估AI生成音频水印的鲁棒性，解决了现有水印移除方案的局限性。

### 背景

高质量AI生成音频的普及带来了安全挑战，包括虚假信息传播和语音克隆欺诈。水印技术是区分AI生成音频与真实音频的关键防御手段，但研究水印移除技术对评估水印鲁棒性至关重要。

### 目的

开发一种高效的水印移除方法，仅需要基本的水印生成能力，无需了解水印的具体细节，以客观评估现有音频水印方案的鲁棒性。

### 方法

提出HarmonicAttack方法，采用双路径卷积自编码器架构，同时在时域和频域操作，结合GAN风格训练，将水印从原始音频中分离。该方法只需要目标水印方案的基本水印生成能力即可进行训练。

### 主要发现

HarmonicAttack在与最先进的水印方案AudioSeal、WavMark和Silentcipher的对比测试中，表现出比先前水印移除方法更强的水印移除能力，且具有接近实时的性能。尽管需要训练，但该方法能够很好地泛化到分布外的样本，性能下降有限。

### 结论

HarmonicAttack为评估音频水印鲁棒性提供了有效工具，有助于开发更安全的水印方案，应对AI生成音频带来的安全挑战。

### 翻译

高质量AI生成音频的可用性带来了诸如虚假信息传播和语音克隆欺诈等安全挑战。对抗AI生成音频滥用的一个关键防御手段是通过对其进行水印处理，使其能够与真实音频轻易区分。由于那些试图滥用AI生成音频的人可能会寻求移除音频水印，因此研究有效的水印移除技术对于能够客观评估水印对移除的鲁棒性至关重要。先前的水印移除方案要么假设具有不切实际的水印知识，要么计算成本高昂，可能对当前水印方案产生虚假的安全感。我们引入了HarmonicAttack，一种高效的音频水印移除方法，只需要能够从目标方案生成水印的基本能力，无需其他条件。通过这种方法，我们能够训练一个通用水印移除模型，能够从任何带水印的音频样本中移除目标方案生成的水印。HarmonicAttack采用在时域和频域操作的双路径卷积自编码器，结合GAN风格训练，将水印与原始音频分离。在与最先进的水印方案AudioSeal、WavMark和Silentcipher的评估中，HarmonicAttack表现出比先前水印移除方法更强的水印移除能力，且具有接近实时的性能。此外，尽管HarmonicAttack需要训练，但我们发现它能够很好地泛化到分布外的样本，性能下降有限。


### 论文摘要

The availability of high-quality, AI-generated audio raises security challenges such as misinformation campaigns and voice-cloning fraud. A key defense against the misuse of AI-generated audio is by watermarking it, so that it can be easily distinguished from genuine audio. As those seeking to misuse AI-generated audio may thus seek to remove audio watermarks, studying effective watermark removal techniques is critical to being able to objectively evaluate the robustness of audio watermarks against removal. Previous watermark removal schemes either assume impractical knowledge of the watermarks they are designed to remove or are computationally expensive, potentially generating a false sense of confidence in current watermark schemes.   We introduce HarmonicAttack, an efficient audio watermark removal method that only requires the basic ability to generate the watermarks from the targeted scheme and nothing else. With this, we are able to train a general watermark removal model that is able to remove the watermarks generated by the targeted scheme from any watermarked audio sample. HarmonicAttack employs a dual-path convolutional autoencoder that operates in both temporal and frequency domains, along with GAN-style training, to separate the watermark from the original audio. When evaluated against state-of-the-art watermark schemes AudioSeal, WavMark, and Silentcipher, HarmonicAttack demonstrates greater watermark removal ability than previous watermark removal methods with near real-time performance. Moreover, while HarmonicAttack requires training, we find that it is able to transfer to out-of-distribution samples with minimal degradation in performance.

---

## 80. Machine Learning Approaches to Clinical Risk Prediction: Multi-Scale Temporal Alignment in Electronic Health Records

**论文链接:** [http://arxiv.org/abs/2511.21561v1](http://arxiv.org/abs/2511.21561v1)

**作者:** Wei-Chen Chang, Lu Dai, Ting Xu

**发布时间:** 2025-11-26

**备注:** 5 pages, 3 figures

### GPT解析

### 总结

本研究提出了一种基于多尺度时间对齐网络(MSTAN)的风险预测方法，用于解决电子健康记录(EHR)中时间不规则性、采样间隔差异和多尺度动态依赖的挑战。

### 背景

电子健康记录(EHR)存在时间不规则性、采样间隔差异和多尺度动态依赖等挑战，影响风险预测的准确性。

### 目的

开发一种能够有效处理EHR数据中时间特性的风险预测方法，提高疾病风险预测和健康状态评估的准确性。

### 方法

引入可学习的时间对齐机制和多尺度卷积特征提取结构，将多源临床特征映射到统一高维语义空间，使用时间嵌入和对齐模块处理不规则采样数据，通过多尺度特征提取捕获不同时间粒度的关键模式，最后使用基于注意力的聚合机制整合全局时间依赖性。

### 主要发现

在公开EHR数据集上的实验表明，该模型在准确率、召回率、精确度和F1分数上均优于主流基线方法，证明了多尺度时间对齐在复杂医疗时间序列分析中的有效性和稳健性。

### 结论

该方法为高维异步医疗序列的智能表示提供了新解决方案，为EHR驱动的临床风险预测提供了重要技术支持。

### 翻译

本研究提出了一种基于多尺度时间对齐网络(MSTAN)的风险预测方法，以解决电子健康记录(EHR)中时间不规则性、采样间隔差异和多尺度动态依赖的挑战。该方法通过引入可学习的时间对齐机制和多尺度卷积特征提取结构，专注于时间特征建模，联合建模EHR序列中的长期趋势和短期波动。在输入层面，模型将多源临床特征映射到统一的高维语义空间，并采用时间嵌入和对齐模块动态加权不规则采样数据，减少时间分布差异对模型性能的影响。随后，多尺度特征提取模块通过多层卷积和层次融合捕获不同时间粒度的关键模式，实现对患者状态的细粒度表示。最后，基于注意力的聚合机制整合全局时间依赖性，为疾病风险预测和健康状态评估生成个体级风险表示。在公开可用的EHR数据集上进行的实验表明，所提出的模型在准确率、召回率、精确度和F1分数上优于主流基线，证明了多尺度时间对齐在复杂医疗时间序列分析中的有效性和稳健性。该研究为高维异步医疗序列的智能表示提供了新解决方案，并为EHR驱动的临床风险预测提供了重要的技术支持。


### 论文摘要

This study proposes a risk prediction method based on a Multi-Scale Temporal Alignment Network (MSTAN) to address the challenges of temporal irregularity, sampling interval differences, and multi-scale dynamic dependencies in Electronic Health Records (EHR). The method focuses on temporal feature modeling by introducing a learnable temporal alignment mechanism and a multi-scale convolutional feature extraction structure to jointly model long-term trends and short-term fluctuations in EHR sequences. At the input level, the model maps multi-source clinical features into a unified high-dimensional semantic space and employs temporal embedding and alignment modules to dynamically weight irregularly sampled data, reducing the impact of temporal distribution differences on model performance. The multi-scale feature extraction module then captures key patterns across different temporal granularities through multi-layer convolution and hierarchical fusion, achieving a fine-grained representation of patient states. Finally, an attention-based aggregation mechanism integrates global temporal dependencies to generate individual-level risk representations for disease risk prediction and health status assessment. Experiments conducted on publicly available EHR datasets show that the proposed model outperforms mainstream baselines in accuracy, recall, precision, and F1-Score, demonstrating the effectiveness and robustness of multi-scale temporal alignment in complex medical time-series analysis. This study provides a new solution for intelligent representation of high-dimensional asynchronous medical sequences and offers important technical support for EHR-driven clinical risk prediction.

---

## 81. MMA: A Momentum Mamba Architecture for Human Activity Recognition with Inertial Sensors

**论文链接:** [http://arxiv.org/abs/2511.21550v1](http://arxiv.org/abs/2511.21550v1)

**作者:** Thai-Khanh Nguyen, Uyen Vo, Tan M. Nguyen, Thieu N. Vo, Trung-Hieu Le, Cuong Pham

**发布时间:** 2025-11-26

**备注:** 14 pages, 5 pages

### GPT解析

### 总结

本文介绍了一种名为'Momentum Mamba'的新型结构化状态空间模型，通过动量增强和二阶动力学改进人类活动识别的性能。

### 背景

人类活动识别对于普适计算、移动健康和环境智能至关重要。传统深度模型(CNN、RNN、transformers)存在梯度消失/爆炸、高计算成本和难以捕捉长程依赖等问题。现有SSM模型如Mamba仅限于一阶动力学，缺乏稳定的长时记忆机制。

### 目的

开发一种改进的SSM模型，解决现有模型局限性，提高HAR的准确性、鲁棒性和收敛速度，同时保持良好的计算效率。

### 方法

提出'Momentum Mamba'，一种动量增强的SSM，结合二阶动力学改进信息流稳定性、鲁棒性和长序列建模能力。同时提出Complex Momentum Mamba扩展，用于频率选择性记忆缩放。

### 主要发现

在多个HAR基准测试中，Momentum Mamba相比原始Mamba和Transformer基线在准确性、鲁棒性和收敛速度方面都有一致提升，且仅以适度训练成本增加获得良好的准确性-效率平衡。

### 结论

动量增强的SSM为HAR提供了可扩展的范式，并为更广泛的序列建模应用成为有前途的主干框架。

### 翻译

从惯性传感器进行人类活动识别对于普适计算、移动健康和环境智能至关重要。传统深度模型推动了HAR发展，但仍受限于梯度消失/爆炸、高计算成本和难以捕捉长程依赖等问题。结构化状态空间模型如Mamba通过线复杂度和有效时序建模解决了这些挑战，但仅限于一阶动力学，没有稳定长期记忆机制。我们引入了Momentum Mamba，这是一种动量增强的SSM，结合二阶动力学改进时间步间信息流稳定性、鲁棒性和长序列建模能力。两种扩展进一步扩展了其能力：Complex Momentum Mamba用于频率选择性记忆缩放。在多个HAR基准测试中，Momentum Mamba相比原始Mamba和Transformer基线在准确性、鲁棒性和收敛速度方面都有一致提升。仅以适度训练成本增加，动量增强的SSM提供了良好的准确性-效率平衡，使其成为HAR的可扩展范式，并为更广泛的序列建模应用成为有前途的主干框架。


### 论文摘要

Human activity recognition (HAR) from inertial sensors is essential for ubiquitous computing, mobile health, and ambient intelligence. Conventional deep models such as Convolutional Neural Networks (CNNs), Recurrent Neural Networks (RNNs), and transformers have advanced HAR but remain limited by vanishing or exloding gradients, high computational cost, and difficulty in capturing long-range dependencies. Structured state-space models (SSMs) like Mamba address these challenges with linear complexity and effective temporal modeling, yet they are restricted to first-order dynamics without stable longterm memory mechanisms. We introduce Momentum Mamba, a momentum-augmented SSM that incorporates second-order dynamics to improve stability of information flow across time steps, robustness, and long-sequence modeling. Two extensions further expand its capacity: Complex Momentum Mamba for frequency-selective memory scaling. Experiments on multiple HAR benchmarks demonstrate consistent gains over vanilla Mamba and Transformer baselines in accuracy, robustness, and convergence speed. With only moderate increases in training cost, momentum-augmented SSMs offer a favorable accuracy-efficiency balance, establishing them as a scalable paradigm for HAR and a promising principal framework for broader sequence modeling applications.

---

## 82. Video Generation Models Are Good Latent Reward Models

**论文链接:** [http://arxiv.org/abs/2511.21541v1](http://arxiv.org/abs/2511.21541v1)

**作者:** Xiaoyue Mi, Wenqing Yu, Jiesong Lian, Shibo Jie, Ruizhe Zhong, Zijun Liu, Guozhen Zhang, Zixiang Zhou, Zhiyong Xu, Yuan Zhou, Qinglin Lu, Fan Tang

**发布时间:** 2025-11-26

### GPT解析

### 总结

本研究提出了一种名为PRFL的新框架，用于在潜在空间中进行视频生成的偏好优化，解决了传统像素空间ReFL方法的内存和效率问题，同时提高了与人类偏好的一致性。

### 背景

奖励反馈学习(ReFL)在图像生成中已被证明有效，但在视频生成领域面临重大挑战。现有视频奖励模型依赖为像素空间输入设计的视觉语言模型，限制了优化只能在VAE解码后的去噪步骤中进行。

### 目的

解决现有视频生成中奖励反馈学习的内存开销大、训练时间长，以及缺乏早期阶段监督的问题，提高与人类偏好的一致性。

### 方法

提出Process Reward Feedback Learning (PRFL)框架，完全在潜在空间中进行偏好优化，利用预训练视频生成模型处理嘈杂潜在表示的能力，实现整个去噪链的高效梯度反向传播，无需VAE解码。

### 主要发现

PRFL显著提高了视频生成与人类偏好的一致性，同时相比传统RGB ReFL方法实现了内存消耗和训练时间的显著减少。

### 结论

通过在潜在空间而非像素空间进行奖励反馈学习，PRFL解决了视频生成中的关键效率问题，同时保持了高质量的结果，为视频生成模型的人对齐提供了更有效的方法。

### 翻译

奖励反馈学习(ReFL)已被证明能够有效使图像生成与人类偏好保持一致。然而，其在视频生成中的扩展面临重大挑战。现有的视频奖励模型依赖于为像素空间输入设计的视觉语言模型，将ReFL优化限制在计算昂贵的VAE解码后的接近完全去噪步骤中。这种像素空间方法导致大量内存开销和增加的训练时间，且其后期优化缺乏早期阶段的监督，仅改进视觉质量而非基本运动动态和结构连贯性。在本工作中，我们表明预训练的视频生成模型自然适合于嘈杂潜在空间中的奖励建模，因为它们被明确设计用于处理任意时间步的嘈杂潜在表示，并通过顺序建模能力固有地保留时间信息。因此，我们提出了过程奖励反馈学习(PRFL)，一个完全在潜在空间中进行偏好优化的框架，能够在没有VAE解码的情况下实现整个去噪链的高效梯度反向传播。大量实验表明，PRFL显著提高了与人类偏好的一致性，同时相比RGB ReFL实现了内存消耗和训练时间的显著减少。


### 论文摘要

Reward feedback learning (ReFL) has proven effective for aligning image generation with human preferences. However, its extension to video generation faces significant challenges. Existing video reward models rely on vision-language models designed for pixel-space inputs, confining ReFL optimization to near-complete denoising steps after computationally expensive VAE decoding. This pixel-space approach incurs substantial memory overhead and increased training time, and its late-stage optimization lacks early-stage supervision, refining only visual quality rather than fundamental motion dynamics and structural coherence. In this work, we show that pre-trained video generation models are naturally suited for reward modeling in the noisy latent space, as they are explicitly designed to process noisy latent representations at arbitrary timesteps and inherently preserve temporal information through their sequential modeling capabilities. Accordingly, we propose Process Reward Feedback Learning~(PRFL), a framework that conducts preference optimization entirely in latent space, enabling efficient gradient backpropagation throughout the full denoising chain without VAE decoding. Extensive experiments demonstrate that PRFL significantly improves alignment with human preferences, while achieving substantial reductions in memory consumption and training time compared to RGB ReFL.

---

## 83. Mechanistic Interpretability for Transformer-based Time Series Classification

**论文链接:** [http://arxiv.org/abs/2511.21514v1](http://arxiv.org/abs/2511.21514v1)

**作者:** Matīss Kalnāre, Sofoklis Kitharidis, Thomas Bäck, Niki van Stein

**发布时间:** 2025-11-26

### GPT解析

### 总结

本文将自然语言处理中的机械可解释性技术适应到时间序列分类的Transformer架构中，探究模型内部机制和因果结构。

### 背景

Transformer模型已成为时间序列分类等机器学习任务的先进工具，但其复杂性使得理解内部决策过程具有挑战性。现有可解释性方法主要关注输入-输出归因，内部机制仍不透明。

### 目的

弥合Transformer模型内部机制不透明的差距，将机械可解释性技术从NLP适应到时间序列分类的Transformer架构中。

### 方法

采用激活修补、注意力和稀疏自编码器三种技术，系统探究单个注意力头和时间步的内部因果作用，构建因果图说明信息传播过程。

### 主要发现

揭示了模型中的因果结构，确定了驱动正确分类的关键注意力头和时间位置，展示了稀疏自编码器在发现可解释潜在特征方面的潜力。

### 结论

研究结果为Transformer可解释性提供了方法论贡献，并对Transformer在时间序列分类任务中的功能机制提供了新见解。

### 翻译

基于Transformer的模型已成为各种机器学习任务的先进工具，包括时间序列分类，但其复杂性使得理解其内部决策过程具有挑战性。现有的可解释性方法通常关注输入-输出归因，而内部机制在很大程度上仍然不透明。本文通过调整各种机械可解释性技术，弥合这一差距，将这些技术从自然语言处理适应到专门为时间序列分类设计的Transformer架构中。我们系统地探究了单个注意力头和时间步的内部因果作用，揭示了这些模型中的因果结构。通过在基准时间序列数据集上的实验，我们构建了因果图来说明信息如何在内部传播，突出了驱动正确分类的关键注意力头和时间位置。此外，我们展示了稀疏自编码器在发现可解释潜在特征方面的潜力。我们的发现为Transformer可解释性提供了方法论贡献，并对Transformer在时间序列分类任务中性能的潜在功能机制提供了新的见解。


### 论文摘要

Transformer-based models have become state-of-the-art tools in various machine learning tasks, including time series classification, yet their complexity makes understanding their internal decision-making challenging. Existing explainability methods often focus on input-output attributions, leaving the internal mechanisms largely opaque. This paper addresses this gap by adapting various Mechanistic Interpretability techniques; activation patching, attention saliency, and sparse autoencoders, from NLP to transformer architectures designed explicitly for time series classification. We systematically probe the internal causal roles of individual attention heads and timesteps, revealing causal structures within these models. Through experimentation on a benchmark time series dataset, we construct causal graphs illustrating how information propagates internally, highlighting key attention heads and temporal positions driving correct classifications. Additionally, we demonstrate the potential of sparse autoencoders for uncovering interpretable latent features. Our findings provide both methodological contributions to transformer interpretability and novel insights into the functional mechanics underlying transformer performance in time series classification tasks.

---

## 84. Making sense of quantum teleportation: An intervention study on students' conceptions using a diagrammatic approach

**论文链接:** [http://arxiv.org/abs/2511.21443v1](http://arxiv.org/abs/2511.21443v1)

**作者:** Sebastian Kilde-Westberg, Andreas Johansson, Anna Pearson, Jonas Enger

**发布时间:** 2025-11-26

### GPT解析

### 总结

本研究探讨了使用基于ZX演算的简化图示形式教授量子隐形传态对高中生和职前教师理解的影响，确定了四种不同层次的理解类别，并指出图示方法虽提供可接受的切入点，但不能自动解决基本概念挑战。

### 背景

传统中学量子物理教育采用历史方法，很少超越20世纪初的想法，使学生无法理解现代量子技术，而这些技术在日常生活和现代工业的许多方面都很重要。

### 目的

解决传统量子物理教育的局限性，探究高中生和职前教师在基于ZX演算的简化图示形式教授下理解量子隐形传态的方式。

### 方法

通过现象学分析视频记录的小组工作、书面练习回答和小组访谈，对21名参与者进行研究，确定描述体验量子隐形传态的不同方式的类别。

### 主要发现

概念进展取决于四个因素：对量子过程中时间性的理解、量子隐形传态中纠缠的作用、量子测量的主动性质以及对图表中数学操作的解释。

### 结论

简化的图示形式为中学水平的量子物理教学提供了可接受的切入点，但不能自动解决基本的概念挑战，需要仔细考虑开发教学和学习序列。研究结果为教育工作者提供了设计改进教学的更深入理解，并强调需要进一步探索师生对量子现象的理解。

### 翻译

中学阶段的量子物理教育传统上遵循历史方法，很少超越20世纪初的想法，使学生无法理解现代量子技术，而这些技术在日常生活和现代工业的许多方面都至关重要。为了解决这一差距，我们研究了高中生和职前教师在基于ZX演算的简化图示形式教授量子隐形传态时如何理解这一概念。通过对视频记录的小组工作、书面练习回答和小组访谈进行现象学分析，共21名参与者，我们确定了一个包含四种不同层次描述类别的结果空间，这些类别描述了体验量子隐形传态的不同方式。这些类别表明，概念进展取决于一个人如何理解量子过程中的时间性、量子隐形传态中纠缠的作用、量子测量的主动性质以及对图表中数学操作的解释。我们的研究结果表明，虽然简化的图示形式为中学水平的量子物理教学提供了可接受的切入点，但它并不能自动解决基本的概念挑战，需要仔细考虑开发教学和学习序列。最后，这些结果为教育工作者提供了更深入的理解，用于设计和改进教学，同时还强调了需要进一步探索学生和教师对量子现象的理解方式。


### 论文摘要

Quantum physics education at the upper-secondary level traditionally follows a historical approach, rarely extending beyond early 20th-century ideas, leaving students unprepared for comprehending modern quantum technologies central to everyday life and many facets of modern industry. To address this gap, we investigated how upper-secondary students and pre-service teachers understand quantum teleportation when taught with a simplified diagrammatic formalism based on the ZX-calculus, which represents quantum processes as diagrams of wires and boxes. Through phenomenographic analysis of video-recorded group work sessions, written responses to exercises, and a group interview, with a total of n=21 participants, we identified an outcome space consisting of four qualitatively different, hierarchically ordered categories of description encapsulating the different ways of experiencing quantum teleportation. The categories revealed that a conceptual progression depends on how one understands the temporality in quantum processes, the role of entanglement in quantum teleportation, the active nature of quantum measurements, and interpretations of mathematical operations in the diagrams. Our findings demonstrate that while a simplified diagrammatic formalism for teaching quantum physics provides an accessible entry point at the upper-secondary level, it does not automatically resolve fundamental conceptual challenges, and requires careful consideration in terms of developing teaching and learning sequences. Finally, these results provide educators with a deeper understanding of conceptual affordances and challenges for designing and improving instruction, whilst also highlighting the need for further exploring how students and teachers alike understand quantum phenomena.

---

## 85. Subjective Depth and Timescale Transformers: Learning Where and When to Compute

**论文链接:** [http://arxiv.org/abs/2511.21408v1](http://arxiv.org/abs/2511.21408v1)

**作者:** Frederico Wieser, Martin Benfeghoul, Haitham Bou Ammar, Jun Wang, Zafeirios Fountas

**发布时间:** 2025-11-26

### GPT解析

### 总结

论文提出Subjective Depth Transformers (SDT)和Subjective Timescale Transformers (STT)两种架构，利用贝叶斯惊喜信号动态路由计算，提高Transformer效率，减少计算量。

### 背景

标准Transformer架构中计算资源的刚性、均匀分配限制了效率和可扩展性，特别是在处理大规模模型和长序列时。

### 目的

引入SDT和STT两种架构，利用贝叶斯惊喜信号动态路由计算，学习在仅解码器Transformer中何时何地进行计算。

### 方法

SDT通过交替的决策层和动态层增强仅解码器堆栈；STT将条件计算扩展到时间域，使用转换网络预测残差更新，形成时间'变化假设'，动态执行或跳过Transformer块。

### 主要发现

两种架构在训练过程中表现出从新颖性到预测驱动门控的转变，与基于惊喜的原则一致；在降低计算能力的同时，提供了对条件计算中计算-准确性权衡的初步见解。

### 结论

提出的架构建立了效率的灵活框架，在每个计算跳过层中将自注意力计算减少75%，KV缓存需求减少50%，为更高效的模型铺平了道路。

### 翻译

标准Transformer架构中计算的刚性、均匀分配可能限制其效率和可扩展性，特别是对于大规模模型和长序列。针对这一问题，我们引入了主观深度Transformer(SDT)和主观时间尺度Transformer(STT)两种不同的架构，它们利用贝叶斯惊喜信号来动态路由计算，学习在仅解码器Transformer中何时何地进行计算。SDT通过交替的决策层和动态层增强仅解码器堆栈：决策层计算完整块的后验和轻量级先验，而动态层基于贝叶斯惊喜(预期变化和意外变化)使用固定容量的Top-K路由，保持静态计算图。STT将这种条件计算扩展到时间域：转换网络预测残差更新，形成时间'变化假设'，指导路由器动态执行或跳过每个token的Transformer块，管理KV缓存贡献。两种架构都表现出训练过程中从新颖性到预测驱动门控的转变，表明与基于惊喜的原则一致。虽然以降低的计算能力运行，但它们为条件计算中的计算-准确性权衡提供了初步见解。所提出的架构建立了效率的灵活框架，在每个计算跳过层中将自注意力计算减少75%，将KV缓存需求减少50%，为更高效的模型铺平了道路。


### 论文摘要

The rigid, uniform allocation of computation in standard Transformer (TF) architectures can limit their efficiency and scalability, particularly for large-scale models and long sequences. Addressing this, we introduce Subjective Depth Transformers (SDT) and Subjective Timescale Transformers (STT), two distinct architectures that leverage Bayesian surprise signals to dynamically route computation, learning where and when to compute within decoder-only TFs. SDT augments a decoder-only stack with alternating Decision and Dynamic layers: a Decision layer computes a full block 'posterior' and a lightweight 'prior,' while a Dynamic layer employs fixed-capacity Top-K routing based on Bayesian surprise (Expected and Unexpected Change), maintaining a static compute graph. STT extends this conditional computation to the temporal domain: a transition network predicts residual updates, forming a temporal 'change hypothesis' that informs a router to dynamically execute or bypass TF blocks for each token, managing KV-cache contributions. Both architectures exhibit the predicted shift from novelty to prediction driven gating over training, suggesting alignment with surprise based principles. While operating at reduced capacity, they offer preliminary insights into the compute-accuracy trade-offs of conditional computation. The proposed architectures establish a flexible framework for efficiency, reducing self-attention computation by 75% and KV-cache requirements by 50% within each compute skipping layer, setting a pathway for more efficient models.

---

## 86. HTTM: Head-wise Temporal Token Merging for Faster VGGT

**论文链接:** [http://arxiv.org/abs/2511.21317v1](http://arxiv.org/abs/2511.21317v1)

**作者:** Weitian Wang, Lukas Meiner, Rai Shubham, Cecilia De La Parra, Akash Kumar

**发布时间:** 2025-11-26

### GPT解析

### 总结

本文提出了一种名为head-wise temporal merging (HTTM)的训练免费3D token合并方法，用于加速Visual Geometry Grounded Transformer (VGGT)在大场景3D重建中的推理速度。

### 背景

VGGT是3D场景重建领域的重大突破，首次能够直接一次性推断所有关键3D属性（相机姿态、深度和密集几何）。然而，其联合推断机制需要全局注意力层进行全对全注意力计算，导致大场景重建时出现显著的延迟瓶颈。

### 目的

解决VGGT在大场景重建中的性能瓶颈问题，提出一种能够加速VGGT推理同时保持性能的方法。

### 方法

提出head-wise temporal merging (HTTM)，一种训练免费的3D token合并方法。与现有技术不同，HTTM在多头粒度上合并token，保留了特征token的独特性，并利用头级别的空间局部性和时间对应关系实现更高的合并比和更低的合并成本。

### 主要发现

HTTM能够在GPU推理中实现高达7倍的加速，同时性能下降可忽略不计。

### 结论

HTTM是一种有效的加速VGGT的方法，能够在保持性能的同时显著提高大场景3D重建的效率。

### 翻译

视觉几何基础变换器（VGGT）在3D场景重建中标志着重大飞跃，它是第一个直接一次性推断所有关键3D属性（相机姿态、深度和密集几何）的模型。然而，这种联合推断机制需要全局注意力层对所有视图的token执行全对全注意力计算。对于具有长序列输入的大场景重建，这会导致显著的延迟瓶颈。在本文中，我们提出了head-wise temporal merging (HTTM)，一种用于加速VGGT的免费训练3D token合并方法。现有的合并技术在不同注意力头之间均匀合并token，导致层输出中token相同，这阻碍了模型的表示能力。HTTM通过在多头粒度上合并token来解决这一问题，保留了头连接后特征token的独特性。此外，这使得HTTM能够利用在头级别观察到的空间局部性和时间对应关系，与现有方法相比实现更高的合并比和更低的合并成本。因此，HTTM在基于GPU的推理中实现了高达7倍的加速，同时性能下降可忽略不计。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决VGGT模型在3D场景重建中的计算效率问题。VGGT模型的全局注意力层需要对所有视图的tokens执行全对全注意力计算，对于大型场景重建来说计算成本非常高，导致显著的延迟瓶颈。这个问题很重要，因为VGGT是首个能直接在一个过程中联合推断所有关键3D属性(相机姿态、深度和密集几何)的模型，提高其效率可以使3D重建在实际应用中更加可行，如机器人、自动驾驶和增强现实等领域。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先分析了VGGT的特性和瓶颈，发现其全局注意力层是计算瓶颈，且注意力分布模式与LLM不同。通过研究VGGT中的相似性模式，包括空间和时间token相似性、RoPE(旋转位置编码)的影响以及输入相似性的作用，作者识别出现有方法的局限性。论文借鉴了token merging方法(如ToMe、ToMeSD)和长序列模型中的稀疏注意力方法，但发现这些方法不能直接应用于VGGT，因此需要针对VGGT的特殊性进行改进，特别是其head级别的相似性模式。', '这个方法的核心思想是什么？整体实现流程是怎样的？': 'HTTM的核心思想是在多头注意力的每个头级别上独立进行token合并，同时利用时空相关性来优化合并过程。具体包括：1) Head-wise token merging：每个注意力头独立合并tokens；2) Block-wise token merging：使用块状策略减少匹配成本；3) Temporal reordering：将token重新组织为时空块；4) Head-wise adaptive outlier filtering：过滤异常值。整体流程是：首先对每个头的Q、K、V tokens独立计算相似度矩阵并合并，然后通过时间重新排序形成时空块，在减少长度的tokens上执行注意力计算，最后通过解合并恢复完整token序列并应用异常值过滤。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1) Head-wise token merging：每个注意力头独立合并tokens，保留头特定信息；2) Block-wise token merging with temporal reordering：通过时空块显著降低合并成本；3) Head-wise adaptive outlier filtering：在全局预算下过滤异常值。相比之前的工作，HTTM不同于现有token合并方法(如ToMe)的统一合并策略，而是利用VGGT特有的时空相似性模式；与FastVGGT相比，HTTM考虑了VGGT在head级别的特殊相似性模式，使用时间重新排序提高合并质量；与稀疏注意力方法相比，不依赖稀疏注意力模式，更适合VGGT的窄注意力分布。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'HTTM提出了一种基于head-wise temporal token merging的训练免费方法，通过在VGGT的每个注意力头独立合并时空相关的tokens并采用块状合并策略，实现了高达7倍的加速同时保持了高质量的3D场景重建性能。'}


### 论文摘要

The Visual Geometry Grounded Transformer (VGGT) marks a significant leap forward in 3D scene reconstruction, as it is the first model that directly infers all key 3D attributes (camera poses, depths, and dense geometry) jointly in one pass. However, this joint inference mechanism requires global attention layers that perform all-to-all attention computation on tokens from all views. For reconstruction of large scenes with long-sequence inputs, this causes a significant latency bottleneck. In this paper, we propose head-wise temporal merging (HTTM), a training-free 3D token merging method for accelerating VGGT. Existing merging techniques merge tokens uniformly across different attention heads, resulting in identical tokens in the layers' output, which hinders the model's representational ability. HTTM tackles this problem by merging tokens in multi-head granularity, which preserves the uniqueness of feature tokens after head concatenation. Additionally, this enables HTTM to leverage the spatial locality and temporal correspondence observed at the head level to achieve higher merging ratios with lower merging costs compared to existing methods. Thus, HTTM achieves up to 7x acceleration with negligible performance drops in a GPU-based inference.

---

## 87. Acoustic neural networks: Identifying design principles and exploring physical feasibility

**论文链接:** [http://arxiv.org/abs/2511.21313v1](http://arxiv.org/abs/2511.21313v1)

**作者:** Ivan Kalthoff, Marcel Rey, Raphael Wittkowski

**发布时间:** 2025-11-26

**备注:** 13 pages, 4 figures, 8 tables

### GPT解析

### 总结

本文介绍了一种设计和模拟声学神经网络的框架，通过声波传播实现计算，在AudioMNIST数据集上达到95%的准确率，同时保持与无源声学组件的兼容性。

### 背景

基于波导的物理系统为超越传统电子学的节能模拟计算提供了有前途的途径，声学神经网络在电子效率低下或受限的环境中实现低功耗计算是一种有前景的方法，但其系统设计在很大程度上仍未被探索。

### 目的

引入一个设计和模拟声学神经网络的框架，通过声波传播执行计算，建立可实现的声学计算系统的系统设计。

### 方法

使用数字孪生方法，在物理约束下训练传统神经网络架构，包括非负信号和权重、无偏置项、与基于强度的非负声学信号兼容的非线性；提出SincHSRNN，一种结合可学习声学带通滤波器和分层时间处理的混合模型。

### 主要发现

约束的循环和分层架构可以执行准确的语音分类；SincHSRNN在AudioMNIST数据集上达到95%的准确率，同时保持与无源声学组件的兼容性；学习到的参数对应可测量的材料和几何特性，如衰减和传输。

### 结论

建立了可实现声学神经网络的通用设计原则，为低功耗、基于波的神经计算指明了一条途径。

### 翻译

基于波导的物理系统为超越传统电子学的节能模拟计算提供了有前途的途径。在这一背景下，声学神经网络为实现低功耗计算提供了一种有前景的方法，特别是在电子效率低下或受限的环境中，然而其系统设计在很大程度上仍未被探索。在此，我们介绍了一个设计和模拟声学神经网络的框架，这些网络通过声波传播执行计算。采用数字孪生方法，我们在物理约束下训练传统神经网络架构，包括非负信号和权重、无偏置项，以及与基于强度的非负声学信号兼容的非线性。我们的工作为声学神经网络提供了一个通用框架，将可学习的网络组件直接与可测量的声学特性联系起来，实现了可实现声学计算系统的系统设计。我们证明，约束的循环和分层架构可以执行准确的语音分类，我们提出了SincHSRNN，这是一种结合了可学习声学带通滤波器和分层时间处理的混合模型。SincHSRNN在AudioMNIST数据集上达到95%的准确率，同时保持与无源声学组件的兼容性。除了计算性能外，学习到的参数对应于可测量的材料和几何特性，如衰减和传输。我们的结果建立了可实现声学神经网络的通用设计原则，并指出了低功耗、基于波的神经计算的途径。


### 论文摘要

Wave-guide-based physical systems provide a promising route toward energy-efficient analog computing beyond traditional electronics. Within this landscape, acoustic neural networks represent a promising approach for achieving low-power computation in environments where electronics are inefficient or limited, yet their systematic design has remained largely unexplored. Here we introduce a framework for designing and simulating acoustic neural networks, which perform computation through the propagation of sound waves. Using a digital-twin approach, we train conventional neural network architectures under physically motivated constraints including non-negative signals and weights, the absence of bias terms, and nonlinearities compatible with intensity-based, non-negative acoustic signals. Our work provides a general framework for acoustic neural networks that connects learnable network components directly to physically measurable acoustic properties, enabling the systematic design of realizable acoustic computing systems. We demonstrate that constrained recurrent and hierarchical architectures can perform accurate speech classification, and we propose the SincHSRNN, a hybrid model that combines learnable acoustic bandpass filters with hierarchical temporal processing. The SincHSRNN achieves up to 95% accuracy on the AudioMNIST dataset while remaining compatible with passive acoustic components. Beyond computational performance, the learned parameters correspond to measurable material and geometric properties such as attenuation and transmission. Our results establish general design principles for physically realizable acoustic neural networks and outline a pathway toward low-power, wave-based neural computing.

---

## 88. 3-Tracer: A Tri-level Temporal-Aware Framework for Audio Forgery Detection and Localization

**论文链接:** [http://arxiv.org/abs/2511.21237v1](http://arxiv.org/abs/2511.21237v1)

**作者:** Shuhan Xia, Xuannan Liu, Xing Cui, Peipei Li

**发布时间:** 2025-11-26

### GPT解析

### 总结

部分音频伪造是一种新型的音频操纵形式，攻击者选择性地修改部分语义上重要的帧同时保持整体感知真实性，使得检测变得困难。现有方法缺乏层次结构来捕获不同时间级别上的异常。

### 背景

部分音频伪造已成为一种新型的音频操纵形式。攻击者会选择性地修改部分语义上重要的帧，同时保持整体感知真实性，这使得这种伪造特别难以检测。

### 目的

解决现有方法独立检测单个帧是否被伪造且缺乏层次结构的局限性，提出一个能够在帧、段和音频级别联合分析音频以全面检测伪造痕迹的框架。

### 方法

T3-Tracer包含两个互补的核心模块：帧-音频特征聚合模块(FA-FAM)和段级多尺度差异感知模块(SMDAM)。FA-FAM结合帧级和音频级的时间信息检测帧内伪造线索和全局语义不一致性。SMDAM采用双分支架构，在多尺度时间窗口上联合建模帧特征和帧间差异，识别伪造边界上的异常。

### 主要发现

在三个具有挑战性的数据集上进行的大量实验表明，该方法达到了最先进的性能。

### 结论

T3-Tracer是第一个在帧、段和音频级别联合分析音频以全面检测伪造痕迹的框架，通过其两个互补的核心模块，能够有效检测部分音频伪造。

### 翻译

最近，部分音频伪造已成为一种新型的音频操纵形式。攻击者选择性地修改部分语义上重要的帧，同时保持整体感知真实性，使得这种伪造特别难以检测。现有方法专注于独立检测单个帧是否被伪造，缺乏能够捕获不同时间级别上的瞬时和持续异常的层次结构。为解决这些局限性，我们确定了与部分音频伪造检测相关的三个关键级别，并提出了T3-Tracer，这是第一个在帧、段和音频级别联合分析音频以全面检测伪造痕迹的框架。T3-Tracer包含两个互补的核心模块：帧-音频特征聚合模块(FA-FAM)和段级多尺度差异感知模块(SMDAM)。FA-FAM旨在检测每个音频帧的真实性，它结合帧级和音频级的时间信息来检测帧内伪造线索和全局语义不一致性。为了进一步改进和纠正帧检测，我们引入了SMDAM来检测段级别的伪造边界，它采用双分支架构，在多尺度时间窗口上联合建模帧特征和帧间差异，有效识别出现在伪造边界上的突然异常。在三个具有挑战性的数据集上进行的大量实验表明，我们的方法达到了最先进的性能。


### 论文摘要

Recently, partial audio forgery has emerged as a new form of audio manipulation. Attackers selectively modify partial but semantically critical frames while preserving the overall perceptual authenticity, making such forgeries particularly difficult to detect. Existing methods focus on independently detecting whether a single frame is forged, lacking the hierarchical structure to capture both transient and sustained anomalies across different temporal levels. To address these limitations, We identify three key levels relevant to partial audio forgery detection and present T3-Tracer, the first framework that jointly analyzes audio at the frame, segment, and audio levels to comprehensively detect forgery traces. T3-Tracer consists of two complementary core modules: the Frame-Audio Feature Aggregation Module (FA-FAM) and the Segment-level Multi-Scale Discrepancy-Aware Module (SMDAM). FA-FAM is designed to detect the authenticity of each audio frame. It combines both frame-level and audio-level temporal information to detect intra-frame forgery cues and global semantic inconsistencies. To further refine and correct frame detection, we introduce SMDAM to detect forgery boundaries at the segment level. It adopts a dual-branch architecture that jointly models frame features and inter-frame differences across multi-scale temporal windows, effectively identifying abrupt anomalies that appeared on the forged boundaries. Extensive experiments conducted on three challenging datasets demonstrate that our approach achieves state-of-the-art performance.

---

## 89. Towards an Effective Action-Region Tracking Framework for Fine-grained Video Action Recognition

**论文链接:** [http://arxiv.org/abs/2511.21202v1](http://arxiv.org/abs/2511.21202v1)

**作者:** Baoli Sun, Yihan Wang, Xinzhu Ma, Zhihui Wang, Kun Lu, Zhiyong Wang

**发布时间:** 2025-11-26

### GPT解析

### 总结

本文提出了一种名为动作区域跟踪(ART)的新框架，利用查询-响应机制来发现和跟踪独特局部细节的动态，有效区分相似动作。

### 背景

细粒度动作识别(FGAR)旨在识别动作类别间的细微差异，但当前方法往往只捕捉粗粒度运动模式，难以识别随时间演变的局部区域中的细微细节。

### 目的

开发一种能够发现和跟踪独特局部细节动态的新方法，从而有效区分相似动作。

### 方法

提出ART框架，包括区域特定语义激活模块(使用判别性和文本约束语义作为查询)、将区域响应组织成动作轨迹、使用视觉语言模型中的文本描述编码语义表示、多层次轨迹对比约束优化、任务特定微调机制优化文本语义。

### 主要发现

在广泛使用的动作识别基准上进行的综合实验表明，该方法优于先前最先进的基线。

### 结论

ART框架通过关注局部细节动态，有效解决了细粒度动作识别中的挑战，为区分相似动作提供了新思路。

### 翻译

本文提出了一种名为动作区域跟踪(ART)的新框架，利用查询-响应机制来发现和跟踪独特局部细节的动态，有效区分相似动作。细粒度动作识别(FGAR)旨在识别动作类别间的细微差异，但当前方法往往只捕捉粗粒度运动模式，难以识别随时间演变的局部区域中的细微细节。ART框架包括区域特定语义激活模块、将区域响应组织成动作轨迹、使用视觉语言模型中的文本描述编码语义表示、多层次轨迹对比约束优化和任务特定微调机制。在广泛使用的动作识别基准上进行的综合实验表明，该方法优于先前最先进的基线。


### 论文摘要

Fine-grained action recognition (FGAR) aims to identify subtle and distinctive differences among fine-grained action categories. However, current recognition methods often capture coarse-grained motion patterns but struggle to identify subtle details in local regions evolving over time. In this work, we introduce the Action-Region Tracking (ART) framework, a novel solution leveraging a query-response mechanism to discover and track the dynamics of distinctive local details, enabling effective distinction of similar actions. Specifically, we propose a region-specific semantic activation module that employs discriminative and text-constrained semantics as queries to capture the most action-related region responses in each video frame, facilitating interaction among spatial and temporal dimensions with corresponding video features. The captured region responses are organized into action tracklets, which characterize region-based action dynamics by linking related responses across video frames in a coherent sequence. The text-constrained queries encode nuanced semantic representations derived from textual descriptions of action labels extracted by language branches within Visual Language Models (VLMs). To optimize the action tracklets, we design a multi-level tracklet contrastive constraint among region responses at spatial and temporal levels, enabling effective discrimination within each frame and correlation between adjacent frames. Additionally, a task-specific fine-tuning mechanism refines textual semantics such that semantic representations encoded by VLMs are preserved while optimized for task preferences. Comprehensive experiments on widely used action recognition benchmarks demonstrate the superiority to previous state-of-the-art baselines.

---

## 90. Broad-band temporal and spectral study of TeV blazar TXS 0518+211

**论文链接:** [http://arxiv.org/abs/2511.21182v1](http://arxiv.org/abs/2511.21182v1)

**作者:** Avik Kumar Das, Pankaj Kushwaha, Veeresh Singh, Goldy Ahuja, Deekshya R. Sarkar

**发布时间:** 2025-11-26

**备注:** 20 pages, 8 figures, 4 tables. Submitted to ApJ

### GPT解析

### 总结

研究了对TeV BL Lac天体TXS 0518+211的长期宽带时间和光谱特性，分析了近16年同时期的光学、紫外、X射线和伽马射线光曲线。

### 背景

TXS 0518+211是一个TeV BL Lac天体，研究其长期行为有助于理解活动星系核喷流活动特性。

### 目的

通过分析多波段同时观测数据，识别不同时期并研究TXS 0518+211的时间变化和光谱变化，以了解该天体的发射特性。

### 方法

分析了约16年（MJD 54682-60670）的Swift-XRT/UVOT光学、紫外、X射线数据和Fermi-LAT伽马射线数据，识别出11个时期，进行了变异性分析和通量-通量相关性研究。

### 主要发现

X射线波段在所有时期都表现出相对较高的变异性；不同波段间存在弱到中等相关性（Spearman相关系数0.29-0.58）；Epoch-I期间检测到X射线通量增加的孤立耀斑，但光学和紫外波段无对应变化；Epoch-K期间X射线通量显著减少，但其他波段无相应变化；揭示了通量状态变化和喷流主导发射过程的复杂性质。

### 结论

在所有研究时期，双区轻子模型能更好地描述这个TeV BL Lac天体的宽波段发射特性。

### 翻译

我们对一个TeV BL Lac天体TXS 0518+211进行了长期的宽带时间和光谱研究，通过分析近16年（MJD 54682-60670）Swift-XRT/UVOT和Fermi-LAT提供的同期光学、紫外、X射线和伽马射线光曲线。基于可用同期多波段数据和通量水平作为活动星系核-喷流活动的表征，我们确定了11个时期（命名为Epoch-A至Epoch-K），并研究了这些时期的时间变化和光谱变化，以了解该天体的发射特性。变异性分析显示，在所有时期，X射线光曲线相比光学、紫外和伽马射线光曲线表现出相对较高的变异性。不同波段间的通量-通量图通常显示弱到中等相关性，Spearman相关系数范围从0.29到0.58。值得注意的是，在Epoch-I期间，我们检测到一个可能的孤立耀斑，X射线通量水平增加（约为总平均通量的2.4倍），但在光学、紫外波段没有看到相应的对应变化。相反，在Epoch-K期间，我们检测到X射线通量显著减少，但在光学、紫外和伽马射线波段没有相应的减少。总体而言，我们的研究揭示了通量状态的几种变化和喷流主导发射过程的复杂性质。在所有时期，双区轻子模型能更好地描述这个TeV BL Lac天体的宽波段发射。


### 论文摘要

We present a long-term broad-band temporal and spectral study of a TeV BL Lac source TXS 0518+211 by analyzing nearly 16 years (MJD 54682 -- 60670) of simultaneous optical, UV and X-ray light curves from Swift-XRT/UVOT and gamma-ray light curves from Fermi-LAT. Based on the availability of simultaneous multi-wavelength data and considering flux level as the depiction of AGN-jet activity we identified 11 epochs (named as Epoch-A to Epoch-K) and investigated temporal as well as spectral variability during these epochs to understand the emission properties in this source. The fractional variability analysis reveals that, in all epochs, X-ray light curve exhibits relatively high degree of variability in compared to the optical, UV and gamma-ray light curves. The flux-flux plots among different bands, in general, show weak to moderate correlation with Spearman correlation coefficient ranging from 0.29 to 0.58. Notably, during Epoch-I, we detect a possible orphan flare exhibiting increase in the X-ray flux level ($\sim$ 2.4 times of the total average flux) but with no corresponding counterpart seen in the optical, UV bands. In contrast, during Epoch-K, we detect a significant decrease in the X-ray flux but no corresponding decrease in optical, UV and gamma-ray bands. Overall, our study reveals several changes in the flux states and complex nature of jet dominated emission processes. In all the epochs, two-zone leptonic model provides a better description of the broad-band emission in this TeV BL Lac source.

---

## 91. TEAR: Temporal-aware Automated Red-teaming for Text-to-Video Models

**论文链接:** [http://arxiv.org/abs/2511.21145v1](http://arxiv.org/abs/2511.21145v1)

**作者:** Jiaming He, Guanyu Hou, Hongwei Li, Zhicong Huang, Kangjie Chen, Yi Yu, Wenbo Jiang, Guowen Xu, Tianwei Zhang

**发布时间:** 2025-11-26

### GPT解析

### 总结

TEAR是一个专门针对Text-to-Video模型时间动态特性的安全评估框架，通过两阶段优化方法提高测试生成器效果，实验证明其在多种T2V系统上攻击成功率超过80%，显著优于之前最好的57%结果。

### 背景

Text-to-Video模型能够合成高质量、时间连贯的动态视频内容，但其多样化生成也带来了关键安全挑战。现有安全评估方法主要关注静态图像和文本生成，不足以捕捉视频生成中的复杂时间动态。

### 目的

提出一个名为TEAR(TEmporal-aware Automated Red-teaming)的框架，专门用于发现与T2V模型动态时间序列相关的安全风险。

### 方法

TEAR采用时间感知测试生成器，通过两阶段方法优化：初始生成器训练和时间感知在线偏好学习，用于制作文本上无害的提示，利用时间动态来引发违反政策的视频输出。同时采用一个改进模型，周期性地提高提示的隐蔽性和对抗有效性。

### 主要发现

在开源和商业T2V系统上的广泛实验评估证明了TEAR的有效性，攻击成功率超过80%，比之前最好的结果57%有显著提升。

### 结论

TEAR框架能够有效发现T2V模型中与时间动态相关的安全风险，通过专门针对视频时间特性的测试，提高了安全评估的准确性和有效性。

### 翻译

文本到视频(T2V)模型能够合成高质量、时间连贯的动态视频内容，但其多样化生成也 inherently 带来了关键的安全挑战。现有的安全评估方法主要关注静态图像和文本生成，不足以捕捉视频生成中的复杂时间动态。为解决这一问题，我们提出了一个时间感知自动化红队框架，名为TEAR，这是一个专门设计用于发现与T2V模型动态时间序列相关的安全风险的自动化框架。TEAR采用时间感知测试生成器，通过两阶段方法优化：初始生成器训练和时间感知在线偏好学习，用于制作文本上无害的提示，利用时间动态来引发违反政策的视频输出。同时采用一个改进模型，周期性地提高提示的隐蔽性和对抗有效性。广泛的实验评估证明了TEAR在开源和商业T2V系统上的有效性，攻击成功率超过80%，比之前最好的57%结果有显著提升。


### 论文摘要

Text-to-Video (T2V) models are capable of synthesizing high-quality, temporally coherent dynamic video content, but the diverse generation also inherently introduces critical safety challenges. Existing safety evaluation methods,which focus on static image and text generation, are insufficient to capture the complex temporal dynamics in video generation. To address this, we propose a TEmporal-aware Automated Red-teaming framework, named TEAR, an automated framework designed to uncover safety risks specifically linked to the dynamic temporal sequencing of T2V models. TEAR employs a temporal-aware test generator optimized via a two-stage approach: initial generator training and temporal-aware online preference learning, to craft textually innocuous prompts that exploit temporal dynamics to elicit policy-violating video output. And a refine model is adopted to improve the prompt stealthiness and adversarial effectiveness cyclically. Extensive experimental evaluation demonstrates the effectiveness of TEAR across open-source and commercial T2V systems with over 80% attack success rate, a significant boost from prior best result of 57%.

---

## 92. Referring Video Object Segmentation with Cross-Modality Proxy Queries

**论文链接:** [http://arxiv.org/abs/2511.21139v1](http://arxiv.org/abs/2511.21139v1)

**作者:** Baoli Sun, Xinzhu Ma, Ning Wang, Zhihui Wang, Zhiyong Wang

**发布时间:** 2025-11-26

### GPT解析

### 总结

该论文提出了ProxyFormer，一种新型的Referring Video Object Segmentation (RVOS)架构，通过引入代理查询解决现有方法在目标跟踪和跨模态对齐方面的局限性，显著提高了目标分割的准确性和连贯性。

### 背景

Referring video object segmentation (RVOS)是一个新兴的跨模态任务，旨在生成给定文本表达所指目标对象的像素级地图。主要概念是在语义空间内学习视觉元素和语言表达的准确对齐。

### 目的

解决现有RVOS方法中的两个主要局限性：(1)条件查询缺乏帧间依赖性和变化建模；(2)延迟整合文本约束导致视频特征可能集中在非指代对象上。

### 方法

提出ProxyFormer架构，引入一组代理查询来整合视觉和文本语义，促进它们之间的语义流动。通过在视频特征编码器的多个阶段中渐进式更新和传播代理查询，确保视频特征集中在感兴趣的对象上，同时建立帧间依赖性。为降低计算成本，将跨模态交互解耦为时间和空间维度，并设计了联合语义一致性(JSC)训练策略。

### 主要发现

在四个广泛使用的RVOS基准测试上进行的全面实验证明，ProxyFormer优于现有的最先进方法，能够更准确和连贯地进行目标跟踪和分割。

### 结论

ProxyFormer通过引入代理查询机制和渐进式更新策略，有效解决了现有RVOS方法在帧间依赖性和文本约束整合方面的局限性，为视频目标分割任务提供了更有效的解决方案。

### 翻译

引用视频对象分割(RVOS)是一个新兴的跨模态任务，旨在生成给定文本表达所指目标对象的像素级地图。主要概念涉及在语义空间内学习视觉元素和语言表达的准确对齐。最近的方法通过条件查询解决跨模态对齐，使用基于transformer结构的查询-响应机制跟踪目标对象。然而，它们表现出两个局限性：(1)这些条件查询缺乏帧间依赖性和变化建模，在帧间变化显著时难以进行准确的目标跟踪；(2)它们延迟整合文本约束，可能导致视频特征集中在非指代对象上。因此，我们提出了一种名为ProxyFormer的新型RVOS架构，引入一组代理查询来整合视觉和文本语义，促进它们之间的语义流动。通过在视频特征编码器的多个阶段中渐进式更新和传播代理查询，ProxyFormer确保视频特征集中在感兴趣的对象上。这种动态演化还能建立帧间依赖性，提高目标跟踪的准确性和连贯性。为降低高计算成本，我们将跨模态交互解耦为时间和空间维度。此外，我们还设计了联合语义一致性(JSC)训练策略，使代理查询与组合的视频-文本对之间保持语义共识。在四个广泛使用的RVOS基准测试上进行的全面实验证明了我们的ProxyFormer优于最先进的方法。


### 论文摘要

Referring video object segmentation (RVOS) is an emerging cross-modality task that aims to generate pixel-level maps of the target objects referred by given textual expressions. The main concept involves learning an accurate alignment of visual elements and language expressions within a semantic space. Recent approaches address cross-modality alignment through conditional queries, tracking the target object using a query-response based mechanism built upon transformer structure. However, they exhibit two limitations: (1) these conditional queries lack inter-frame dependency and variation modeling, making accurate target tracking challenging amid significant frame-to-frame variations; and (2) they integrate textual constraints belatedly, which may cause the video features potentially focus on the non-referred objects. Therefore, we propose a novel RVOS architecture called ProxyFormer, which introduces a set of proxy queries to integrate visual and text semantics and facilitate the flow of semantics between them. By progressively updating and propagating proxy queries across multiple stages of video feature encoder, ProxyFormer ensures that the video features are focused on the object of interest. This dynamic evolution also enables the establishment of inter-frame dependencies, enhancing the accuracy and coherence of object tracking. To mitigate high computational costs, we decouple cross-modality interactions into temporal and spatial dimensions. Additionally, we design a Joint Semantic Consistency (JSC) training strategy to align semantic consensus between the proxy queries and the combined video-text pairs. Comprehensive experiments on four widely used RVOS benchmarks demonstrate the superiority of our ProxyFormer to the state-of-the-art methods.

---

## 93. SocialNav: Training Human-Inspired Foundation Model for Socially-Aware Embodied Navigation

**论文链接:** [http://arxiv.org/abs/2511.21135v1](http://arxiv.org/abs/2511.21135v1)

**作者:** Ziyi Chen, Yingnan Guo, Zedong Chu, Minghua Luo, Yanfen Shen, Mingchao Sun, Junjun Hu, Shichao Xie, Kuan Yang, Pei Shi, Zhining Gu, Lu Liu, Honglin Han, Xiaolong Wu, Mu Xu, Yu Zhang

**发布时间:** 2025-11-26

### GPT解析

### 总结

SocialNav是一个具有层次化'大脑-行动'架构的基础模型，用于社会感知导航，通过多阶段训练流程注入和优化导航智能，在SocNav数据集上训练，实现了显著的性能提升。

### 背景

符合社会规范的具身导航仍然是一个开放的研究挑战。

### 目的

开发一个能够理解高级社会规范并生成低级、符合社会规范轨迹的导航模型。

### 方法

构建SocNav数据集（700万样本），包括认知激活数据集和专家轨迹金字塔；提出多阶段训练流程，先通过模仿学习注入导航技能和社会规范理解，再使用SAFE-GRPO框架进行强化学习优化。

### 主要发现

SocialNav与最先进方法相比实现了+38%的成功率和+46%的社会合规率，在导航性能和社会合规性方面都有显著提升。

### 结论

SocialNav是一个有效的基础模型，能够实现社会感知导航，显著提高导航性能和社会合规性。

### 翻译

符合社会规范的具身导航仍然是一个开放的研究挑战。我们的SocialNav是一个具有层次化'大脑-行动'架构的基础模型，用于社会感知导航，能够理解高级社会规范并生成低级、符合社会规范的轨迹。为实现这种双重能力，我们构建了SocNav数据集，这是一个包含700万个样本的大规模集合，包括(1)提供社会推理信号（如思维链解释和社会可通行性预测）的认知激活数据集，以及(2)来自互联网视频、模拟环境和真实机器人的多样化导航演示的专家轨迹金字塔。我们提出了多阶段训练流程，逐步注入和优化导航智能：我们首先通过模仿学习将通用导航技能和社会规范理解注入模型，然后通过专门设计的Socially-Aware Flow Exploration GRPO (SAFE-GRPO)来完善这些技能，这是第一个明确奖励符合社会行为的具身导航的基于流的强化学习框架。与最先进的方法相比，SocialNav实现了+38%的成功率和+46%的社会合规率，展示了在导航性能和社会合规性方面的显著提升。我们的项目页面：https://amap-eai.github.io/SocialNav/


### 论文摘要

Embodied navigation that adheres to social norms remains an open research challenge. Our \textbf{SocialNav} is a foundational model for socially-aware navigation with a hierarchical "brain-action" architecture, capable of understanding high-level social norms and generating low-level, socially compliant trajectories. To enable such dual capabilities, we construct the SocNav Dataset, a large-scale collection of 7 million samples, comprising (1) a Cognitive Activation Dataset providing social reasoning signals such as chain-of-thought explanations and social traversability prediction, and (2) an Expert Trajectories Pyramid aggregating diverse navigation demonstrations from internet videos, simulated environments, and real-world robots. A multi-stage training pipeline is proposed to gradually inject and refine navigation intelligence: we first inject general navigation skills and social norms understanding into the model via imitation learning, and then refine such skills through a deliberately designed Socially-Aware Flow Exploration GRPO (SAFE-GRPO), the first flow-based reinforcement learning framework for embodied navigation that explicitly rewards socially compliant behaviors. SocialNav achieves +38% success rate and +46% social compliance rate compared to the state-of-the-art method, demonstrating strong gains in both navigation performance and social compliance. Our project page: https://amap-eai.github.io/SocialNav/

---

## 94. CtrlVDiff: Controllable Video Generation via Unified Multimodal Video Diffusion

**论文链接:** [http://arxiv.org/abs/2511.21129v1](http://arxiv.org/abs/2511.21129v1)

**作者:** Dianbing Xi, Jiepeng Wang, Yuanzhi Liang, Xi Qiu, Jialun Liu, Hao Pan, Yuchi Huo, Rui Wang, Haibin Huang, Chi Zhang, Xuelong Li

**发布时间:** 2025-11-26

**备注:** 27 pages, 18 figures, 9 tables. Project page: https://tele-ai.github.io/CtrlVDiff/

### GPT解析

### 总结

这篇论文提出了一种名为CtrlVDiff的统一扩散模型，结合多种模态（深度、法线、分割、边缘和内在特征）来解决视频理解和可控视频生成的双重挑战，通过混合模态控制策略实现精确的视频编辑和生成。

### 背景

仅依赖几何线索（如深度、边缘）的视频理解和生成方法存在局限性，它们虽然指定了布局但不足以约束外观、材质和照明，限制了物理上有意义的编辑（如重新照明或材质交换），并常常导致时间漂移。

### 目的

构建一个能够接受任意模态子集、对缺失输入保持鲁棒性、注入控制信号而不牺牲时间一致性的统一模型，实现精确、可控的视频理解和生成。

### 方法

提出CtrlVDiff，一种使用混合模态控制策略(HMCS)训练的统一扩散模型，能够路由和融合来自深度、法线、分割、边缘和基于图形的内在特征（反照率、粗糙度、金属度）的特征，并从任何选定的子集重新渲染具有强时间一致性的视频。为此，构建了MMVideo，一个跨模态和对齐的混合真实和合成数据集。

### 主要发现

通过添加基于图形的模态（内在特征和语义）提供互补约束，可以同时解决理解问题和生成过程中的精确控制问题。CtrlVDiff在理解和生成基准测试中提供了优越的可控性和保真度，能够实现分层编辑（重新照明、材质调整、对象插入）。

### 结论

CtrlVDiff在视频理解和可控视频生成任务上超越了最先进的基线方法，并且在某些模态不可用时仍能保持鲁棒性，为视频编辑和生成提供了新的统一框架。

### 翻译

我们在一个统一的扩散框架内解决视频理解和可控视频生成的双重挑战。我们的关键见解有两方面：仅依赖几何线索（如深度、边缘）是不够的：它们指定了布局但不足以约束外观、材质和照明，限制了物理上有意义的编辑，如重新照明或材质交换，并常常导致时间漂移。用额外的基于图形的模态（内在特征和语义）丰富模型，可以提供互补约束，既能消除理解的歧义，也能在生成过程中实现精确、可预测的控制。然而，构建一个使用许多异构线索的单一模型会引入两个核心困难。在架构上，模型必须接受任何模态子集，对缺失输入保持鲁棒，并注入控制信号而不牺牲时间一致性。在数据方面，训练需要大规模、时间对齐的监督，将真实视频与每像素的多模态注释相关联。随后我们提出了CtrlVDiff，一个使用混合模态控制策略(HMCS)训练的统一扩散模型，该策略路由和融合来自深度、法线、分割、边缘和基于图形的内在特征（反照率、粗糙度、金属度）的特征，并从任何选定的子集重新渲染具有强时间一致性的视频。为此，我们构建了MMVideo，一个跨模态和对齐的混合真实和合成数据集。在理解和生成基准测试中，CtrlVDiff提供了优越的可控性和保真度，能够实现分层编辑（重新照明、材质调整、对象插入），并超越了最先进的基线方法，同时在某些模态不可用时保持鲁棒性。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决视频理解和可控视频生成的双重挑战。现有方法主要依赖几何线索（如深度、边缘）来指定场景布局，但不足以约束外观、材料和光照，导致物理上有意义的编辑（如重新照明或材质替换）困难并经常产生时间不一致问题。这个问题在研究中很重要，因为可控视频生成弥合了高级意图（通过文本、草图等表达）和动态视觉实现之间的差距，能够实现精确的场景操纵，在影视制作、虚拟现实和内容创作等领域有广泛应用价值。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先分析了现有方法的局限性，认识到仅依赖几何线索不足以控制视频的外观和材质。他们借鉴了OmniVDiff的工作，该工作展示了在统一扩散框架内处理多种视频模态的能力，并采用了CogVideoX作为基础模型。作者设计了混合模态控制策略（HMCS）来灵活处理不同模态组合，同时构建了MMVideo数据集来解决多模态训练中的数据稀缺问题。整体设计采用三阶段训练流程，逐步增强模型能力，最终实现了在单一模型中同时处理视频理解和生成。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是通过统一的扩散框架同时处理视频理解和生成，整合几何、外观、语义和结构四类场景属性来实现精确控制。整体流程包括：1) 使用共享3D-VAE编码器将输入视频的多种模态编码为潜在表示；2) 应用混合模态控制策略（HMCS）动态确定哪些模态作为条件、哪些作为生成目标；3) 通过扩散变换器处理这些模态特征；4) 使用模态特定投影层将输出投影到各自空间；5) 采用三阶段训练方法逐步增强模型能力。这种方法实现了从文本提示到视频生成的可控过程，并支持多种编辑应用如重新照明、材质编辑和对象插入。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1) 首个在单一模型中联合支持视频生成（前向渲染）和视频理解（逆向渲染）的统一框架；2) 提出混合模态控制策略（HMCS），支持灵活的模态组合并保持训练稳定性；3) 构建了大规模MMVideo数据集，结合真实和合成数据解决多模态训练的数据稀缺问题；4) 设计三阶段训练范式逐步增强模型能力。相比之前工作，CtrlVDiff不再依赖外部专家估计器获取控制信号，避免了域偏移和错误传播；同时整合了更多模态类型（包括外观相关模态如反照率、粗糙度等），提供了更全面的控制能力，能够实现更精确的物理上有意义的视频编辑。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'CtrlVDiff通过统一的扩散框架和混合模态控制策略，首次实现了在单一模型中同时支持高精度的视频理解和多种可控视频生成，为视频内容创作提供了前所未有的细粒度控制能力。'}


### 论文摘要

We tackle the dual challenges of video understanding and controllable video generation within a unified diffusion framework. Our key insights are two-fold: geometry-only cues (e.g., depth, edges) are insufficient: they specify layout but under-constrain appearance, materials, and illumination, limiting physically meaningful edits such as relighting or material swaps and often causing temporal drift. Enriching the model with additional graphics-based modalities (intrinsics and semantics) provides complementary constraints that both disambiguate understanding and enable precise, predictable control during generation.   However, building a single model that uses many heterogeneous cues introduces two core difficulties. Architecturally, the model must accept any subset of modalities, remain robust to missing inputs, and inject control signals without sacrificing temporal consistency. Data-wise, training demands large-scale, temporally aligned supervision that ties real videos to per-pixel multimodal annotations.   We then propose CtrlVDiff, a unified diffusion model trained with a Hybrid Modality Control Strategy (HMCS) that routes and fuses features from depth, normals, segmentation, edges, and graphics-based intrinsics (albedo, roughness, metallic), and re-renders videos from any chosen subset with strong temporal coherence. To enable this, we build MMVideo, a hybrid real-and-synthetic dataset aligned across modalities and captions. Across understanding and generation benchmarks, CtrlVDiff delivers superior controllability and fidelity, enabling layer-wise edits (relighting, material adjustment, object insertion) and surpassing state-of-the-art baselines while remaining robust when some modalities are unavailable.

---

## 95. FaithFusion: Harmonizing Reconstruction and Generation via Pixel-wise Information Gain

**论文链接:** [http://arxiv.org/abs/2511.21113v1](http://arxiv.org/abs/2511.21113v1)

**作者:** YuAn Wang, Xiaofan Li, Chi Huang, Wenhao Zhang, Hao Li, Bosheng Wang, Xun Sun, Jun Wang

**发布时间:** 2025-11-26

**备注:** 16 pages, 10 figures

### GPT解析

### 总结

FaithFusion是一种3DGS-扩散融合框架，通过期望信息增益(EIG)作为统一的时空合成策略，解决了在大视角变化下保持几何保真度同时合成视觉合理外观的问题。

### 背景

在可控驾驶场景重建和3D场景生成中，在大视角变化下保持几何保真度同时合成视觉上合理的表现是至关重要的。

### 目的

解决基于几何的3DGS和外观驱动的扩散模型有效融合面临的固有挑战，避免缺乏逐像素、3D一致的编辑标准导致的过度恢复和几何漂移问题。

### 方法

FaithFusion框架利用期望信息增益(EIG)作为统一的时空合成策略：EIG引导扩散作为空间先验来优化高不确定性区域，同时其像素级权重将编辑提炼回3DGS，形成无需额外先验条件和结构修改的即插即用系统。

### 主要发现

在Waymo数据集上的实验表明，FaithFusion在NTA-IoU、NTL-IoU和FID方面达到了最先进的性能，即使在6米车道偏移时也能保持107.47的FID。

### 结论

FaithFusion框架成功解决了3D场景生成中几何保真度和视觉合理表现之间的平衡问题，为可控驾驶场景重建提供了有效解决方案。

### 翻译

在可控驾驶场景重建和3D场景生成中，在大视角变化下保持几何保真度同时合成视觉上合理的表现是至关重要的。然而，基于几何的3DGS和外观驱动的扩散模型的有效融合面临固有挑战，因为缺乏逐像素、3D一致的编辑标准通常会导致过度恢复和几何漂移。为解决这些问题，我们引入了FaithFusion，一个由逐像素期望信息增益驱动的3DGS-扩散融合框架。EIG作为统一的时空合成策略：它引导扩散作为空间先验来优化高不确定性区域，同时其像素级权重将编辑提炼回3DGS。得到的即插即用系统无需额外的先验条件和结构修改。在Waymo数据集上的广泛实验表明，我们的方法在NTA-IoU、NTL-IoU和FID方面达到了SOTA性能，即使在6米车道偏移时也能保持107.47的FID。我们的代码可在https://github.com/wangyuanbiubiubiu/FaithFusion获取。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决在可控驾驶场景重建和3D场景生成中，如何在大视角变化下同时保持几何保真度和合成视觉上合理的appearance的问题。这个问题在自动驾驶和封闭环仿真中至关重要，因为现有方法在稀疏观测、严重遮挡或远离训练轨迹的视点下常产生几何不一致和伪影，而缺乏像素级的、3D一致的编辑标准会导致过度修复和几何漂移，影响驾驶场景的准确性和可靠性。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': "作者首先识别出现有方法依赖视图级别启发式决策('在哪里、何时以及编辑多少')的局限性，导致生成控制不足。核心洞察是将编辑决策重新表述为信息论度量——编辑如何降低后验不确定性。方法设计上，作者引入像素级期望信息增益(EIG)作为统一策略，设计了EIGent双分支架构，并采用渐进式知识集成。作者借鉴了FisherRF的理论(将其从视图级扩展到像素级)，以及Laplace近似来估计EIG，同时基于现有的3DGS-扩散融合框架如ReconDreamer和DIFIX3D+进行改进。", '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是使用像素级期望信息增益(EIG)作为统一的策略，协调重建和生成，将编辑决策从启发式方法转变为信息论原则。整体流程包括：1)像素级EIG计算，使用Laplace近似估计每个像素的不确定性；2)EIGent引导的双分支可控生成，构建数据集并设计双分支架构；3)渐进式EIG感知扩散到3DGS知识集成，使用EIG作为权重矩阵调制图像损失，结合稀疏深度监督，通过渐进式更新将生成内容整合回3DGS。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)像素级EIG引导，实现细粒度控制；2)EIGent双分支架构，分离背景保存与前景生成；3)统一的EIG编辑策略，同时指导生成和重建；4)渐进式知识集成，形成完整反馈循环。相比之前工作，FaithFusion不依赖额外先验条件或结构修改，使用像素级而非视图级控制，将启发式决策转变为信息论原则，并让EIG同时作为空间权重函数和像素级损失权重，发挥双重作用。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'FaithFusion通过引入像素级期望信息增益作为统一编辑策略，实现了3D高斯溅射与扩散模型的和谐融合，在保持几何保真度的同时显著提升了大视角变化下的时空一致性和感知质量，为可控3D场景建模提供了简洁、可解释且可泛化的框架。'}


### 论文摘要

In controllable driving-scene reconstruction and 3D scene generation, maintaining geometric fidelity while synthesizing visually plausible appearance under large viewpoint shifts is crucial. However, effective fusion of geometry-based 3DGS and appearance-driven diffusion models faces inherent challenges, as the absence of pixel-wise, 3D-consistent editing criteria often leads to over-restoration and geometric drift. To address these issues, we introduce \textbf{FaithFusion}, a 3DGS-diffusion fusion framework driven by pixel-wise Expected Information Gain (EIG). EIG acts as a unified policy for coherent spatio-temporal synthesis: it guides diffusion as a spatial prior to refine high-uncertainty regions, while its pixel-level weighting distills the edits back into 3DGS. The resulting plug-and-play system is free from extra prior conditions and structural modifications.Extensive experiments on the Waymo dataset demonstrate that our approach attains SOTA performance across NTA-IoU, NTL-IoU, and FID, maintaining an FID of 107.47 even at 6 meters lane shift. Our code is available at https://github.com/wangyuanbiubiubiu/FaithFusion.

---

## 96. CLRecogEye : Curriculum Learning towards exploiting convolution features for Dynamic Iris Recognition

**论文链接:** [http://arxiv.org/abs/2511.21097v1](http://arxiv.org/abs/2511.21097v1)

**作者:** Geetanjali Sharma, Gaurav Jaswal, Aditya Nigam, Raghavendra Ramachandra

**发布时间:** 2025-11-26

**备注:** 12 Pages, 3 figures, ISVC conference 2025

### GPT解析

### 总结

本文提出了一种新颖的虹膜认证匹配管道，通过学习虹膜特征的空间-空间-时间表示，有效解决了旋转、尺度变化、镜面反射和散焦模糊等挑战，实现了鲁棒且可推广的虹膜认证解决方案。

### 背景

虹膜认证算法已取得令人印象深刻的识别性能，在边境控制、公民身份识别、刑事调查和商业系统等领域有很好的应用前景。

### 目的

解决现有虹膜认证算法在面对旋转、尺度变化、镜面反射和散焦模糊等挑战时的鲁棒性问题，以及有效利用虹膜模式的空间-空间-时间结构。

### 方法

将每个虹膜图像沿一个维度分割，生成子图像序列作为3D-CNN输入，以捕获空间和空间-空间-时间线索；通过课程方式训练模型，将时间依赖性直接嵌入特征空间；使用三元组和ArcFace损失进行端到端训练，实现高度判别的嵌入。

### 主要发现

通过学习虹膜特征的空间-空间-时间表示，能够有效处理旋转、尺度变化、镜面反射和散焦模糊等挑战，提高虹膜认证的鲁棒性和泛化能力。

### 结论

所提出的方法提供了一个鲁棒且可推广的虹膜认证解决方案，能够有效处理各种挑战情况。

### 翻译

虹膜认证算法已取得令人印象深刻的识别性能，使其在边境控制、公民身份识别、刑事调查和商业系统等实际应用中极具前景。然而，其鲁棒性仍面临旋转、尺度变化、镜面反射和散焦模糊等挑战。此外，大多数现有方法依赖于简单的点对点比较，通常使用余弦或L2距离，没有有效利用虹膜模式的空间-空间-时间结构。为了解决这些局限性，我们提出了一种新颖且通用的匹配管道，能够学习虹膜特征丰富的空间-空间-时间表示。我们的方法首先将每个虹膜图像沿一个维度分割，生成一系列子图像作为3D-CNN的输入，使网络能够捕获空间和空间-空间-时间线索。为了进一步增强对空间-空间-时间特征动态的建模，我们以课程方式训练模型。这种设计允许网络将时间依赖性直接嵌入到特征空间中，改进深度度量域的判别能力。该框架以课程方式使用三元组和ArcFace损失进行端到端训练，尽管面临旋转、尺度变化、反射和模糊等挑战，但仍能强制实现高度判别的嵌入。这种设计产生了一个鲁棒且可推广的虹膜认证解决方案。


### 论文摘要

Iris authentication algorithms have achieved impressive recognition performance, making them highly promising for real-world applications such as border control, citizen identification, and both criminal investigations and commercial systems. However, their robustness is still challenged by variations in rotation, scale, specular reflections, and defocus blur. In addition, most existing approaches rely on straightforward point-to-point comparisons, typically using cosine or L2 distance, without effectively leveraging the spatio-spatial-temporal structure of iris patterns. To address these limitations, we propose a novel and generalized matching pipeline that learns rich spatio-spatial-temporal representations of iris features. Our approach first splits each iris image along one dimension, generating a sequence of sub-images that serve as input to a 3D-CNN, enabling the network to capture both spatial and spatio-spatial-temporal cues. To further enhance the modeling of spatio-spatial-temporal feature dynamics, we train the model in curriculum manner. This design allows the network to embed temporal dependencies directly into the feature space, improving discriminability in the deep metric domain. The framework is trained end-to-end with triplet and ArcFace loss in a curriculum manner, enforcing highly discriminative embeddings despite challenges like rotation, scale, reflections, and blur. This design yields a robust and generalizable solution for iris authentication.Github code: https://github.com/GeetanjaliGTZ/CLRecogEye

---

## 97. A Network Dynamical Systems Approach to SDGs

**论文链接:** [http://arxiv.org/abs/2511.21091v1](http://arxiv.org/abs/2511.21091v1)

**作者:** Wuyang Zhang, Lejun Xu

**发布时间:** 2025-11-26

### GPT解析

### 总结

本研究通过建模联合国可持续发展目标(SDGs)作为网络化动态系统，识别出优先投资教育(SDG 4)能产生最大系统效益的杠杆点。

### 背景

联合国可持续发展目标是一个复杂、相互依赖的框架，一个领域的进展可能促进或抑制其他领域的进展，政策制定者面临如何确定资源分配最优点的挑战。

### 目的

帮助国际发展政策制定者识别'杠杆点'——即有限资源分配能产生最大系统效益的具体可持续发展目标。

### 方法

将SDGs建模为网络化动态系统，使用Our World in Data的2018年数据构建16个SDG指标的加权交互网络，采用主成分分析和多元线性回归推导耦合权重，使用扩展的Lotka-Volterra模型模拟发展指标的时间演变，并采用Runge-Kutta 4方法确保数值稳定性。

### 主要发现

SDG 4(优质教育)被确定为关键驱动因素，优先考虑教育可以在发展网络中产生最大的积极溢出效应；研究还进行了敏感性分析并探索了投资与稳定性之间的幂律关系。

### 结论

优先投资教育可以带来最大的系统效益，应作为政策制定的关键杠杆点。

### 翻译

联合国的可持续发展目标(SDGs)代表了一个复杂、相互依赖的框架，其中一个领域的进展可以协同促进或竞争性地抑制其他领域的进展。对于国际发展领域的政策制定者来说，一个关键挑战是确定'杠杆点'——即有限资源分配能产生最大系统效益的具体目标。本研究通过将SDGs建模为网络化动态系统(NDS)来应对这一挑战。使用Our World in Data(2018)的实证数据，我们构建了一个包含16个SDG指标的加权交互网络。我们采用主成分分析(PCA)和多元线性回归来 empirically 推导耦合权重。与以往静态分析不同，我们使用扩展的Lotka-Volterra模型模拟发展指标的时间演变。为确保数值稳定性和复杂性，我们将模拟方法从标准的欧拉积分升级到Runge-Kutta 4(RK4)方法。我们对墨西哥进行的案例研究模拟显示，SDG 4(优质教育)起着关键驱动作用，这表明优先发展教育可以在发展网络中产生最大的积极溢出效应。此外，我们进行了敏感性分析并探索了投资与稳定性之间的幂律关系。


### 论文摘要

The United Nations' Sustainable Development Goals (SDGs) represent a complex, interdependent framework where progress in one area can synergistically promote or competitively inhibit progress in others. For policymakers in international development, a critical challenge is identifying "leverage points" - specific goals where limited resource allocation yields the maximum system-wide benefit. This study addresses this challenge by modeling the SDGs as a Networked Dynamical System (NDS). Using empirical data from Our World in Data (2018), we construct a weighted interaction network of 16 SDG indicators. We employ Principal Component Analysis (PCA) and multiple linear regression to derive coupling weights empirically. Unlike previous static analyses, we simulate the temporal evolution of development indicators using an extended Lotka-Volterra model. To ensure numerical stability and sophistication, we upgrade the simulation method from standard Euler integration to the Runge-Kutta 4 (RK4) method. Our simulation, applied to a case study of Mexico, reveals that SDG 4 (Quality Education) acts as a critical driver, suggesting that prioritizing education yields the most significant positive spillover effects across the development network. Furthermore, we perform sensitivity analysis and explore the power-law relationship between investment and stability.

---

## 98. Long-Term Alzheimers Disease Prediction: A Novel Image Generation Method Using Temporal Parameter Estimation with Normal Inverse Gamma Distribution on Uneven Time Series

**论文链接:** [http://arxiv.org/abs/2511.21057v1](http://arxiv.org/abs/2511.21057v1)

**作者:** Xin Hong, Xinze Sun, Yinhao Li, Yen-Wei Chen

**发布时间:** 2025-11-26

**备注:** 13pages, 6 figures

### GPT解析

### 总结

本研究提出了一种T-NIG模型，用于解决阿尔茨海默病预测中长期图像生成中不规则时间间隔的问题，通过估计正态逆伽马分布中的时间参数，有效维持了疾病相关特征并实现了准确的疾病预测。

### 背景

图像生成可为医生在阿尔茨海默病预测中提供影像诊断基础，但近期研究表明，处理序列数据中的不规则时间间隔时，长期AD预测难以维持疾病相关特征。

### 目的

提出一种模型估计正态逆伽马分布中的时间参数(T-NIG)，辅助长期图像生成，解决不规则时间间隔下的图像生成问题。

### 方法

T-NIG模型使用两个不同时间点的脑部图像创建中间图像和预测未来图像；通过识别坐标邻域特征设计模型；将时间参数纳入正态逆伽马分布理解特征变化；利用不确定性估计减少认知不确定性和偶然不确定性。

### 主要发现

T-NIG在数据集内的短期和长期预测任务中都展示了最先进的性能；能够预测疾病进展同时维持疾病相关特征，即使面对不规则的时间数据分布。

### 结论

T-NIG模型有效解决了长期AD预测中不规则时间间隔的问题，能够在预测疾病进展的同时保持疾病相关特征。

### 翻译

图像生成可以为医生在阿尔茨海默病(AD)的预测中提供影像诊断基础。近期研究表明，在处理序列数据中的不规则时间间隔时，基于图像生成的长期AD预测往往难以维持疾病相关特征。考虑到分布中的时间相关方面可以在图像分布不均时反映疾病相关特征的变化，本研究提出了一种估计正态逆伽马分布中时间参数的T-NIG模型，辅助长期图像生成。T-NIG模型使用两个不同时间点的脑部图像创建中间脑部图像、预测未来图像和预测疾病。T-NIG通过识别坐标邻域中的特征进行设计，将时间参数纳入正态逆伽马分布，以理解具有不同时间间隔的脑成像序列中特征的变化。此外，T-NIG利用不确定性估计来减少模型中的认知不确定性和偶然不确定性，这些不确定性源于时间数据不足。特别是，T-NIG模型在数据集内的短期和长期预测任务中都展示了最先进的性能。实验结果表明，T-NIG能够预测疾病进展同时维持疾病相关特征，即使面对不规则的时间数据分布。


### 论文摘要

Image generation can provide physicians with an imaging diagnosis basis in the prediction of Alzheimer's Disease (AD). Recent research has shown that long-term AD predictions by image generation often face difficulties maintaining disease-related characteristics when dealing with irregular time intervals in sequential data. Considering that the time-related aspects of the distribution can reflect changes in disease-related characteristics when images are distributed unevenly, this research proposes a model to estimate the temporal parameter within the Normal Inverse Gamma Distribution (T-NIG) to assist in generating images over the long term. The T-NIG model employs brain images from two different time points to create intermediate brain images, forecast future images, and predict the disease. T-NIG is designed by identifying features using coordinate neighborhoods. It incorporates a time parameter into the normal inverse gamma distribution to understand how features change in brain imaging sequences that have varying time intervals. Additionally, T-NIG utilizes uncertainty estimation to reduce both epistemic and aleatoric uncertainties in the model, which arise from insufficient temporal data. In particular, the T-NIG model demonstrates state-of-the-art performance in both short-term and long-term prediction tasks within the dataset. Experimental results indicate that T-NIG is proficient in forecasting disease progression while maintaining disease-related characteristics, even when faced with an irregular temporal data distribution.

---

## 99. CNN-LSTM Hybrid Architecture for Over-the-Air Automatic Modulation Classification Using SDR

**论文链接:** [http://arxiv.org/abs/2511.21040v1](http://arxiv.org/abs/2511.21040v1)

**作者:** Dinanath Padhya, Krishna Acharya, Bipul Kumar Dahal, Dinesh Baniya Kshatri

**发布时间:** 2025-11-26

**备注:** 8 Pages, 10 figures, 2 Tables, Accepted in Journal (Journal of Innovations in Engineering Education), Issue is not Published Yet

### GPT解析

### 总结

这篇论文提出了一种基于混合卷积神经网络(CNN)和长短期记忆网络(LSTM)架构的自动调制分类系统，结合软件定义无线电(SDR)平台实现。该系统在识别复杂时变通信信号方面表现出色，在0到30分贝信噪比条件下测试，达到了超过93%的准确率、精确率、召回率和F1分数。

### 背景

自动调制分类是未来无线通信系统的核心技术，能够在没有先验知识的情况下识别调制方案。这项技术对于认知无线电、频谱监测和智能通信网络等应用至关重要。

### 目的

开发一个高效的自动调制分类系统，能够准确识别各种调制方案，特别是在复杂和噪声环境下的通信信号。

### 方法

提出了一种混合CNN-LSTM架构，集成软件定义无线电(SDR)平台，利用CNN进行空间特征提取，利用LSTM捕获时间依赖性。使用混合数据集进行训练，结合RadioML2018数据集和自定义生成的数据集，在0到30dB信噪比条件下测试。使用准确率、精确率、召回率、F1分数和ROC曲线下面积评估系统性能。

### 主要发现

系统能够成功识别空中(OTA)信号，包括来自自制FM发射器的信号以及其他调制方案。优化后的模型达到93.48%的准确率、93.53%的精确率、93.48%的召回率和93.45%的F1分数。AUC-ROC分析确认了模型即使在噪声条件下也具有很强的判别能力。

### 结论

混合CNN-LSTM架构对自动调制分类是有效的，该系统在自适应频谱管理和高级认知无线电系统中有潜在应用价值。

### 翻译

自动调制分类是未来无线通信系统的核心技术，使能够在没有先验知识的情况下识别调制方案。这种能力对于认知无线电、频谱监测和智能通信网络中的应用至关重要。我们提出了一种基于混合卷积神经网络(CNN)和长短期记忆网络(LSTM)架构的AMC系统，集成了软件定义无线电(SDR)平台。所提出的架构利用CNN进行空间特征提取，利用LSTM捕获时间依赖性，能够有效处理复杂、时变的通信信号。该系统的实际能力通过识别来自自制FM发射器的空中(OTA)信号以及其他调制方案得到了证明。该系统在结合了RadioML2018数据集和自定义生成数据集的混合数据集上进行训练，样本信噪比(SNR)从0到30分贝。系统性能使用准确率、精确率、召回率、F1分数和接收机工作特性曲线下面积(AUC-ROC)进行了评估。优化后的模型达到了93.48%的准确率、93.53%的精确率、93.48%的召回率和93.45%的F1分数。AUC-ROC分析确认了模型即使在噪声条件下也具有很强的判别能力。本文的实验结果验证了混合CNN-LSTM架构用于AMC的有效性，表明其在自适应频谱管理和高级认知无线电系统中的潜在应用。


### 论文摘要

Automatic Modulation Classification (AMC) is a core technology for future wireless communication systems, enabling the identification of modulation schemes without prior knowledge. This capability is essential for applications in cognitive radio, spectrum monitoring, and intelligent communication networks. We propose an AMC system based on a hybrid Convolutional Neural Network (CNN) and Long Short-Term Memory (LSTM) architecture, integrated with a Software Defined Radio (SDR) platform. The proposed architecture leverages CNNs for spatial feature extraction and LSTMs for capturing temporal dependencies, enabling efficient handling of complex, time-varying communication signals. The system's practical ability was demonstrated by identifying over-the-air (OTA) signals from a custom-built FM transmitter alongside other modulation schemes. The system was trained on a hybrid dataset combining the RadioML2018 dataset with a custom-generated dataset, featuring samples at Signal-to-Noise Ratios (SNRs) from 0 to 30dB. System performance was evaluated using accuracy, precision, recall, F1 score, and the Area Under the Receiver Operating Characteristic Curve (AUC-ROC). The optimized model achieved 93.48% accuracy, 93.53% precision, 93.48% recall, and an F1 score of 93.45%. The AUC-ROC analysis confirmed the model's discriminative power, even in noisy conditions. This paper's experimental results validate the effectiveness of the hybrid CNN-LSTM architecture for AMC, suggesting its potential application in adaptive spectrum management and advanced cognitive radio systems.

---

## 100. A Probabilistic Framework for Temporal Distribution Generalization in Industry-Scale Recommender Systems

**论文链接:** [http://arxiv.org/abs/2511.21032v1](http://arxiv.org/abs/2511.21032v1)

**作者:** Yuxuan Zhu, Cong Fu, Yabo Ni, Anxiang Zeng, Yuan Fang

**发布时间:** 2025-11-26

### GPT解析

### 总结

ELBO_TDS是一种解决时间分布偏移问题的概率框架，通过数据增强和基于因果结构的自监督变分目标，提高了推荐系统的长期准确性和时间泛化能力。

### 背景

时间分布偏移(TDS)会侵蚀推荐系统的长期准确性，工业实践依赖的周期性增量训练难以同时捕捉稳定和瞬时模式。

### 目的

解决现有方法如不变学习和自监督学习在处理时间分布偏移时存在的不稳定时间泛化、表示崩溃或数据利用效率低等问题。

### 方法

提出ELBO_TDS概率框架，首先通过统计分析确定关键偏移因素并设计数据增强策略扩展训练支持；其次使用因果图建模时间推荐场景，推导基于因果结构的自监督变分目标ELBO_TDS。

### 主要发现

ELBO_TDS实现了优越的时间泛化能力，每个用户的GMV提升了2.33%，已在Shopee产品搜索中成功部署。

### 结论

ELBO_TDS有效解决了时间分布偏移问题，提高了推荐系统的长期准确性，并已成功应用于工业实践。

### 翻译

时间分布偏移(TDS)侵蚀了推荐系统的长期准确性，然而工业实践仍然依赖于周期性增量训练，这种方法难以同时捕捉稳定和瞬时模式。现有的方法如不变学习和自监督学习提供部分解决方案，但常常遭受不稳定的时间泛化、表示崩溃或低效的数据利用等问题。为了解决这些限制，我们提出了ELBO_TDS，一个可以无缝集成到工业规模增量学习管道中的概率框架。首先，我们通过真实生产数据的统计分析确定关键偏移因素，并设计一个简单有效的数据增强策略，对这些时间变化因素进行重新采样以扩展训练支持。其次，为了利用扩展分布的好处同时防止表示崩溃，我们使用因果图对时间推荐场景进行建模，并推导出一个基于因果结构的自监督变分目标ELBO_TDS。通过理论和经验分析支持的广泛实验表明，我们的方法实现了优越的时间泛化能力，每个用户的GMV提升了2.33%，并已在Shopee产品搜索中成功部署。代码可在https://github.com/FuCongResearchSquad/ELBO4TDS获取。


### 论文摘要

Temporal distribution shift (TDS) erodes the long-term accuracy of recommender systems, yet industrial practice still relies on periodic incremental training, which struggles to capture both stable and transient patterns. Existing approaches such as invariant learning and self-supervised learning offer partial solutions but often suffer from unstable temporal generalization, representation collapse, or inefficient data utilization. To address these limitations, we propose ELBO$_\text{TDS}$, a probabilistic framework that integrates seamlessly into industry-scale incremental learning pipelines. First, we identify key shifting factors through statistical analysis of real-world production data and design a simple yet effective data augmentation strategy that resamples these time-varying factors to extend the training support. Second, to harness the benefits of this extended distribution while preventing representation collapse, we model the temporal recommendation scenario using a causal graph and derive a self-supervised variational objective, ELBO$_\text{TDS}$, grounded in the causal structure. Extensive experiments supported by both theoretical and empirical analysis demonstrate that our method achieves superior temporal generalization, yielding a 2.33\% uplift in GMV per user and has been successfully deployed in Shopee Product Search. Code is available at https://github.com/FuCongResearchSquad/ELBO4TDS.

---

## 101. Probabilistic Wildfire Spread Prediction Using an Autoregressive Conditional Generative Adversarial Network

**论文链接:** [http://arxiv.org/abs/2511.21019v1](http://arxiv.org/abs/2511.21019v1)

**作者:** Taehoon Kang, Taeyong Kim

**发布时间:** 2025-11-26

**备注:** 22 pages, 15 figures, Submitted to Journal of Environmental Management

### GPT解析

### 总结

本研究提出了一种自回归条件生成对抗网络（CGAN）用于概率性野火蔓延预测，解决了现有方法在计算效率和预测准确性方面的局限性，提高了预测的准确性和物理可解释性。

### 背景

气候变化增加了野火的频率和严重性，快速准确的火势蔓延预测对有效缓解和响应至关重要。基于物理的模拟器如FARSITE虽能提供高保真预测但计算密集，限制了实时决策应用；而现有深度学习模型往往产生过于平滑的预测，无法捕捉野火传播的复杂非线性动态。

### 目的

提出一种自回归条件生成对抗网络（CGAN）用于概率性野火蔓延预测，通过将预测任务表述为自回归问题，学习序列状态转换，确保长期预测稳定性。

### 方法

使用自回归条件生成对抗网络（CGAN）进行野火蔓延预测，将预测任务作为自回归问题处理，利用对抗学习来捕捉野火蔓延的强非线性和不确定性。

### 主要发现

提出的基于CGAN的模型在总体预测精度和火周边界 delineation 方面优于传统深度学习模型；对抗学习使模型能够捕捉野火蔓延的强非线性和不确定性，而非仅拟合像素平均值；自回归框架促进了野火演变的系统时间预测。

### 结论

提出的基于CGAN的自回归框架提高了野火蔓延预测的准确性和物理可解释性，为时间敏感的响应和疏散规划提供了有希望的基础。

### 翻译

气候变化加剧了野火的频率和严重性，使得快速准确地预测火势蔓延对于有效的缓解和响应至关重要。基于物理的模拟器如FARSITE提供高保真预测，但计算密集，限制了它们在实时决策中的应用，而现有的深度学习模型往往产生过于平滑的预测，无法捕捉野火传播的复杂非线性动态。本研究提出了一种自回归条件生成对抗网络用于概率性野火蔓延预测。通过将预测任务表述为自回归问题，模型学习顺序状态转换，确保长期预测稳定性。实验结果表明，所提出的基于CGAN的模型在总体预测精度和火周边界 delineation 方面均优于传统深度学习模型。这些结果表明对抗学习使模型能够捕捉野火蔓延的强非线性和不确定性，而不是简单地拟合像素平均值。此外，自回归框架促进了野火演变的系统时间预测。所提出的基于CGAN的自回归框架提高了野火蔓延预测的准确性和物理可解释性，为时间敏感的响应和疏散规划提供了有希望的基础。


### 论文摘要

Climate change has intensified the frequency and severity of wildfires, making rapid and accurate prediction of fire spread essential for effective mitigation and response. Physics-based simulators such as FARSITE offer high-fidelity predictions but are computationally intensive, limiting their applicability in real-time decision-making, while existing deep learning models often yield overly smooth predictions that fail to capture the complex, nonlinear dynamics of wildfire propagation. This study proposes an autoregressive conditional generative adversarial network (CGAN) for probabilistic wildfire spread prediction. By formulating the prediction task as an autoregressive problem, the model learns sequential state transitions, ensuring long-term prediction stability. Experimental results demonstrate that the proposed CGAN-based model outperforms conventional deep learning models in both overall predictive accuracy and boundary delineation of fire perimeters. These results demonstrate that adversarial learning allows the model to capture the strong nonlinearity and uncertainty of wildfire spread, instead of simply fitting the pixel average. Furthermore, the autoregressive framework facilitates systematic temporal forecasting of wildfire evolution. The proposed CGAN-based autoregressive framework enhances both the accuracy and physical interpretability of wildfire spread prediction, offering a promising foundation for time-sensitive response and evacuation planning.

---

## 102. Seeing without Pixels: Perception from Camera Trajectories

**论文链接:** [http://arxiv.org/abs/2511.21681v1](http://arxiv.org/abs/2511.21681v1)

**作者:** Zihui Xue, Kristen Grauman, Dima Damen, Andrew Zisserman, Tengda Han

**发布时间:** 2025-11-26

**备注:** Project website: https://sites.google.com/view/seeing-without-pixels

### GPT解析

### 总结

本文首次系统性研究了仅通过相机轨迹（相机在空间中移动的路径）而非像素来感知视频内容的可能性，提出了CamFormer这一专用编码器，并通过多种下游任务验证了其有效性。

### 背景

相机轨迹作为视频的一种轻量级信号，其是否能独立于像素信息揭示视频内容是一个尚未被系统研究的问题。

### 目的

探索仅通过相机轨迹信息感知视频内容的可能性，并开发一种能够将相机姿态轨迹与自然语言对齐的编码器。

### 方法

提出了一种对比学习框架来训练CamFormer，该编码器将相机姿态轨迹投影到联合嵌入空间，使其与自然语言对齐，并在各种下游任务上进行验证。

### 主要发现

相机轨迹是一个极具信息量的信号，能够揭示视频内容；'如何移动'可以指示'你在做什么'（以自我为中心）或'你在观察什么'（以外为中心）；CamFormer嵌入在不同相机姿态估计方法下具有鲁棒性。

### 结论

相机轨迹可以作为一种轻量级、鲁棒且通用的模态，用于感知视频内容，无需依赖像素信息。

### 翻译

人们能否在不看到像素的情况下，仅通过相机轨迹——相机在空间中移动的路径——来感知视频内容？本文首次系统性研究了这个看似不可能的问题。为此，我们提出了一种对比学习框架来训练CamFormer，这是一个专用编码器，将相机姿态轨迹投影到联合嵌入空间，使其与自然语言对齐。我们发现，与其表面的简单性相反，相机轨迹是一个极具信息量的信号，可以揭示视频内容。换句话说，'如何移动'确实可以揭示'你在做什么'（以自我为中心）或'你在观察什么'（以外为中心）。我们在各种下游任务上展示了学习到的CamFormer嵌入的通用性，从跨模态对齐到分类和时间分析。重要的是，我们的表示在不同相机姿态估计方法中都具有鲁棒性，包括高保真多传感器和标准RGB-only估计器。我们的研究确立了相机轨迹作为感知视频内容的一种轻量级、鲁棒且通用的模态。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要研究是否可以通过相机轨迹（相机在空间中的移动路径）来感知视频内容，而不需要查看视频的像素。这个问题很重要，因为相机轨迹是一种轻量级、隐私保护的信号，可以替代或补充视觉信息，在保护隐私的同时提供有价值的语义内容，还能降低计算成本，为视频理解提供新视角。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者从人类感知是主动的这一原理出发，观察到相机运动可能包含丰富的语义信息。他们注意到高质量相机轨迹数据的获取曾是障碍，但最近技术进步使这一数据变得可用。他们借鉴了多模态对比学习（如CLIP）的方法，使用Transformer架构处理序列数据，并参考了现有的相机姿态估计工作。设计上，他们提出了CamFormer模型和上下文化轨迹编码方法来解决相机轨迹信息密度低的问题。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是相机轨迹本身包含丰富的语义信息，可以独立或作为视觉的补充来理解视频内容。整体流程包括：1)收集大规模配对的轨迹-文本数据；2)设计CamFormer模型，使用轻量级Transformer编码相机轨迹；3)使用对比学习将轨迹映射到与文本对齐的语义空间；4)采用上下文化轨迹编码解决信息密度低的问题；5)在多种下游任务上评估模型性能。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)首次将相机轨迹作为新模态用于视频内容理解；2)提出CamFormer专用模型；3)设计上下文化轨迹编码方法；4)在自我中心和外部中心两个领域进行系统评估；5)展示模型对不同姿态估计方法的鲁棒性。相比之前工作，本文将相机轨迹视为语义信号而非几何工具，扩展了应用场景，创新了方法，并利用了最新技术解决数据获取挑战。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '这篇论文首次系统性地证明了相机轨迹本身包含丰富的语义信息，可以独立或作为视觉的补充来理解视频内容，并提出了一种名为CamFormer的轻量级模型，通过对比学习将相机轨迹映射到与文本对齐的语义空间，为视频理解提供了一种新的、隐私友好的、计算高效的模态。'}


### 论文摘要

Can one perceive a video's content without seeing its pixels, just from the camera trajectory-the path it carves through space? This paper is the first to systematically investigate this seemingly implausible question. Towards this end, we propose a contrastive learning framework to train CamFormer, a dedicated encoder that projects camera pose trajectories into a joint embedding space, aligning them with natural language. We find that, contrary to its apparent simplicity, the camera trajectory is a remarkably informative signal to uncover video content. In other words, "how you move" can indeed reveal "what you are doing" (egocentric) or "observing" (exocentric). We demonstrate the versatility of our learned CamFormer embeddings on a diverse suite of downstream tasks, ranging from cross-modal alignment to classification and temporal analysis. Importantly, our representations are robust across diverse camera pose estimation methods, including both high-fidelity multi-sensored and standard RGB-only estimators. Our findings establish camera trajectory as a lightweight, robust, and versatile modality for perceiving video content.

---

## 103. Voice, Bias, and Coreference: An Interpretability Study of Gender in Speech Translation

**论文链接:** [http://arxiv.org/abs/2511.21517v1](http://arxiv.org/abs/2511.21517v1)

**作者:** Lina Conti, Dennis Fucci, Marco Gaido, Matteo Negri, Guillaume Wisniewski, Luisa Bentivogli

**发布时间:** 2025-11-26

**备注:** Submitted to LREC 2026

### GPT解析

### 总结

该研究探讨了语音翻译模型中基于声学线索的性别分配机制，揭示了模型如何利用第一人称代词和频谱信息而非仅依赖音高来确定说话者性别。

### 背景

与文本不同，语音通过声学线索(如音高)传达说话者的性别信息，这导致在将具有概念性别的语言翻译到语法性别分配语言时，存在模态特定偏见和性别误判风险。

### 目的

研究ST模型如何跨三种语言对(英语-西班牙语/法语/意大利语)将性别分配给指代说话者的术语，并检查训练数据模式、内部语言模型偏见和声学信息的相互作用。

### 方法

通过分析ST模型的性别分配机制，使用对比特征归因技术(spectrograms)来揭示模型如何从声学信息中提取性别线索。

### 主要发现

模型并非简单复制训练数据中的术语特定性别关联，而是学习男性普遍性模式；尽管内部语言模型表现出强烈男性偏见，但模型可根据声学输入覆盖这些偏好；具有更高性别准确性的模型使用第一人称代词将性别术语链接回说话者，从整个频谱而非仅音高中提取性别信息。

### 结论

语音翻译模型能够利用声学信息纠正内部语言模型中的性别偏见，并采用第一人称代词作为锚点，从全频谱中提取性别信息。

### 翻译

与文本不同，语音通过音高等声学线索传达关于说话者的信息，如性别。这引发了模态特定偏见问题。例如，在语音翻译中，当将具有概念性别的语言(如英语)翻译到性别模糊术语被分配语法性别的语言时，说话者的声音特征可能在性别分配中发挥作用。这可能导致通过默认男性或基于声音的假设而对说话者进行性别误判。然而，ST模型如何做出这些决策仍知之甚少。我们研究ST模型如何跨三种语言对(英语-西班牙语/法语/意大利语)将性别分配给指代说话者的术语，检查训练数据模式、内部语言模型偏见和声学信息如何相互作用。我们发现模型并非简单复制训练数据中的术语特定性别关联，而是学习更广泛的男性普遍性模式。虽然内部语言模型表现出强烈的男性偏见，但模型可以根据声学输入覆盖这些偏好。通过对频谱图使用对比特征归因，我们揭示出具有更高性别准确性的模型依赖于一种先前未知的机制：使用第一人称代词将性别术语链接回说话者，访问分布在频谱范围内的性别信息，而不是集中在音高中。


### 论文摘要

Unlike text, speech conveys information about the speaker, such as gender, through acoustic cues like pitch. This gives rise to modality-specific bias concerns. For example, in speech translation (ST), when translating from languages with notional gender, such as English, into languages where gender-ambiguous terms referring to the speaker are assigned grammatical gender, the speaker's vocal characteristics may play a role in gender assignment. This risks misgendering speakers, whether through masculine defaults or vocal-based assumptions. Yet, how ST models make these decisions remains poorly understood. We investigate the mechanisms ST models use to assign gender to speaker-referring terms across three language pairs (en-es/fr/it), examining how training data patterns, internal language model (ILM) biases, and acoustic information interact. We find that models do not simply replicate term-specific gender associations from training data, but learn broader patterns of masculine prevalence. While the ILM exhibits strong masculine bias, models can override these preferences based on acoustic input. Using contrastive feature attribution on spectrograms, we reveal that the model with higher gender accuracy relies on a previously unknown mechanism: using first-person pronouns to link gendered terms back to the speaker, accessing gender information distributed across the frequency spectrum rather than concentrated in pitch.

---

## 104. The More, the Merrier: Contrastive Fusion for Higher-Order Multimodal Alignment

**论文链接:** [http://arxiv.org/abs/2511.21331v1](http://arxiv.org/abs/2511.21331v1)

**作者:** Stefanos Koutoupis, Michaela Areti Zervou, Konstantinos Kontras, Maarten De Vos, Panagiotis Tsakalides, Grigorios Tsagatakis

**发布时间:** 2025-11-26

### GPT解析

### 总结

ConFu是一种新的多模态表示学习方法，能够在统一空间中同时嵌入单个模态和它们的融合组合，有效捕获高阶依赖关系，同时保持成对对应关系，在多种任务和设置中表现出色。

### 背景

多模态机器学习中，跨模态的联合表示学习仍然是一个核心挑战。当前主流方法主要在成对设置中操作，一次对齐两个模态，而一些最近的方法虽然试图捕获多个模态之间的高阶交互，但经常忽视或不充分保留成对关系，限制了它们在单模态任务上的有效性。

### 目的

引入一个能够同时嵌入单个模态及其融合组合的框架，在统一的表示空间中对齐模态及其融合对应物，捕获传统成对对比目标无法恢复的高阶依赖关系，同时保持强大的成对对应关系。

### 方法

提出了对比融合（Contrastive Fusion，ConFu）框架，将传统的成对对比目标与额外的融合模态对比项相结合，鼓励将模态对与第三个模态进行联合嵌入，这种公式使ConFu能够捕获XOR类关系等高阶依赖。

### 主要发现

ConFu在合成和真实世界多模态基准测试中表现出竞争性性能，能够利用跨模态互补性，捕获高阶依赖关系，随多模态复杂度的增加而扩展，并在单一对比框架内支持统一的一对一和二对一检索。

### 结论

ConFu框架成功解决了多模态学习中同时处理成对关系和高阶交互的挑战，通过统一的对比学习框架，实现了多种检索任务的支持，在各种设置下都表现出色。

### 翻译

多模态机器学习中跨模态的联合表示学习仍然是一个核心挑战。当前主流方法主要在成对设置中操作，一次对齐两个模态。虽然最近的一些方法试图捕获多个模态之间的高阶交互，但它们经常忽视或不充分保留成对关系，限制了它们在单模态任务上的有效性。在这项工作中，我们引入了对比融合（ConFu）框架，它将单个模态及其融合组合共同嵌入到统一的表示空间中，在该空间中模态及其融合对应物被对齐。ConFu通过额外的融合模态对比项扩展了传统的成对对比目标，鼓励将模态对与第三个模态进行联合嵌入。这种公式使ConFu能够捕获XOR类关系等高阶依赖关系，这些关系无法仅通过成对对齐来恢复，同时仍保持强大的成对对应关系。我们在合成和真实世界的多模态基准测试上评估了ConFu，评估其利用跨模态互补性、捕获高阶依赖性以及随多模态复杂度增加而扩展的能力。在这些设置中，ConFu在检索和分类任务上展示了竞争性性能，同时在单一对比框架内支持统一的一对一和二对一检索。


### 论文摘要

Learning joint representations across multiple modalities remains a central challenge in multimodal machine learning. Prevailing approaches predominantly operate in pairwise settings, aligning two modalities at a time. While some recent methods aim to capture higher-order interactions among multiple modalities, they often overlook or insufficiently preserve pairwise relationships, limiting their effectiveness on single-modality tasks. In this work, we introduce Contrastive Fusion (ConFu), a framework that jointly embeds both individual modalities and their fused combinations into a unified representation space, where modalities and their fused counterparts are aligned. ConFu extends traditional pairwise contrastive objectives with an additional fused-modality contrastive term, encouraging the joint embedding of modality pairs with a third modality. This formulation enables ConFu to capture higher-order dependencies, such as XOR-like relationships, that cannot be recovered through pairwise alignment alone, while still maintaining strong pairwise correspondence. We evaluate ConFu on synthetic and real-world multimodal benchmarks, assessing its ability to exploit cross-modal complementarity, capture higher-order dependencies, and scale with increasing multimodal complexity. Across these settings, ConFu demonstrates competitive performance on retrieval and classification tasks, while supporting unified one-to-one and two-to-one retrieval within a single contrastive framework.

---

## 105. SONAR: Spectral-Contrastive Audio Residuals for Generalizable Deepfake Detection

**论文链接:** [http://arxiv.org/abs/2511.21325v1](http://arxiv.org/abs/2511.21325v1)

**作者:** Ido Nitzan HIdekel, Gal lifshitz, Khen Cohen, Dan Raviv

**发布时间:** 2025-11-26

### GPT解析

### 总结

本文提出了一种名为SONAR的频率引导框架，用于解决Deepfake音频检测器在分布外输入上的泛化问题。该框架通过处理频谱偏差，明确分解音频信号为互补表示，并利用频率交叉对比学习提高检测性能。

### 背景

Deepfake音频检测器在处理分布外输入时仍存在泛化困难，主要原因是频谱偏差——神经网络倾向于先学习低频结构再学习高频细节，导致生成器留下高频伪影而检测器未能充分利用这些伪影。

### 目的

解决现有Deepfake音频检测器的泛化问题，通过处理频谱偏差提高对高频伪影的利用能力，提升检测性能。

### 方法

提出Spectral-cONtrastive Audio Residuals (SONAR)框架，使用XLSR编码器捕获低频内容，通过可学习的SRM和高通滤波器提取高频残差，利用频率交叉注意力处理频率依赖，并采用频率感知的Jensen-Shannon对比损失优化决策边界。

### 主要发现

在ASVspoof 2021和in-the-wild基准测试中，SONAR达到最先进性能，收敛速度比基线快四倍。该框架将潜在空间分成自然-HF和失真-HF两个不相交的流形，有效锐化了决策边界。

### 结论

SONAR完全在表示级别运行，与架构无关，未来可无缝集成到任何依赖高频线索的模型或模态中。

### 翻译

Deepfake音频检测器仍然难以泛化到分布外的输入。一个核心原因是频谱偏差，即神经网络倾向于先学习低频结构再学习高频细节，这既导致DF生成器留下高频伪影，又使常见检测器未能充分利用这些伪影。为解决这一差距，我们提出了Spectral-cONtrastive Audio Residuals (SONAR)，一个频率引导的框架，明确地将音频信号分解为互补表示。XLSR编码器捕获主导的低频内容，而相同的克隆路径前加上可学习的SRM、值约束的高通滤波器，则提炼微弱的高频残差。频率交叉注意力重新连接两个视图，处理长程和短程频率依赖，频率感知的Jensen-Shannon对比损失将真实内容-噪声对拉近，同时将假嵌入推开，加速优化并锐化决策边界。在ASVspoof 2021和in-the-wild基准测试中评估，SONAR取得了最先进的性能，比强大的基线收敛速度快四倍。通过将微弱的高频残差提升为一级学习信号，SONAR揭示了一个完全数据驱动的、频率引导的对比框架，将潜在空间分成两个不相交的流形：自然-HF用于真实音频，失真-HF用于合成音频，从而锐化决策边界。由于该方案完全在表示级别运行，它与架构无关，并且在未来的工作中，可以无缝集成到任何模型或模态中，其中细微的高频线索是决定性的。


### 论文摘要

Deepfake (DF) audio detectors still struggle to generalize to out of distribution inputs. A central reason is spectral bias, the tendency of neural networks to learn low-frequency structure before high-frequency (HF) details, which both causes DF generators to leave HF artifacts and leaves those same artifacts under-exploited by common detectors. To address this gap, we propose Spectral-cONtrastive Audio Residuals (SONAR), a frequency-guided framework that explicitly disentangles an audio signal into complementary representations. An XLSR encoder captures the dominant low-frequency content, while the same cloned path, preceded by learnable SRM, value-constrained high-pass filters, distills faint HF residuals. Frequency cross-attention reunites the two views for long- and short-range frequency dependencies, and a frequency-aware Jensen-Shannon contrastive loss pulls real content-noise pairs together while pushing fake embeddings apart, accelerating optimization and sharpening decision boundaries. Evaluated on the ASVspoof 2021 and in-the-wild benchmarks, SONAR attains state-of-the-art performance and converges four times faster than strong baselines. By elevating faint high-frequency residuals to first-class learning signals, SONAR unveils a fully data-driven, frequency-guided contrastive framework that splits the latent space into two disjoint manifolds: natural-HF for genuine audio and distorted-HF for synthetic audio, thereby sharpening decision boundaries. Because the scheme operates purely at the representation level, it is architecture-agnostic and, in future work, can be seamlessly integrated into any model or modality where subtle high-frequency cues are decisive.

---

## 106. Unlocking Zero-shot Potential of Semi-dense Image Matching via Gaussian Splatting

**论文链接:** [http://arxiv.org/abs/2511.21265v1](http://arxiv.org/abs/2511.21265v1)

**作者:** Juncheng Chen, Chao Xu, Yanjun Cao

**发布时间:** 2025-11-26

### GPT解析

### 总结

MatchGS是首个系统纠正和利用3D高斯溅射(3DGS)进行鲁棒零样本图像匹配的框架，通过几何忠实的数据生成和2D-3D表示对齐策略，显著提升了图像匹配性能。

### 背景

基于学习的图像匹配依赖于大规模、多样化且几何精确的训练数据。3DGS虽能实现照片级真实感的新视角合成，但其几何不准确性和有偏见的深度渲染阻碍了鲁棒的对应关系标记。

### 目的

开发MatchGS框架，系统纠正和利用3DGS，实现鲁棒的零样本图像匹配。

### 方法

双重方法：(1)几何忠实的数据生成流程，优化3DGS几何生成高精度对应关系标签；(2)2D-3D表示对齐策略，将3DGS的显式3D知识注入2D匹配器，引导学习视角不变的3D表示。

### 主要发现

生成的真实对应关系将极线误差减少最多40倍，支持极端视角变化下的监督，通过高斯属性提供自监督信号，使最先进匹配器在公共基准测试上实现最高17.7%的零样本性能提升。

### 结论

适当几何修正后，3DGS可作为可扩展、高保真和结构丰富的数据源，为新一代鲁棒零样本图像匹配器铺平道路。

### 翻译

基于学习的图像匹配严重依赖于大规模、多样化且几何精确的训练数据。3D高斯溅射(3DGS)能够实现照片级真实感的新视角合成，因此对数据生成具有吸引力。然而，其几何不准确性和有偏见的深度渲染目前阻碍了鲁棒的对应关系标记。为此，我们引入MatchGS，这是首个专为鲁棒零样本图像匹配而设计、系统纠正和利用3DGS的框架。我们的方法是双重的：(1)几何忠实的数据生成流程，优化3DGS几何以生成高精度的对应关系标签，能够在不妥协渲染保真度的前提下合成大量多样化的视角；(2)2D-3D表示对齐策略，将3DGS的显式3D知识注入2D匹配器，引导2D半密集匹配器学习视角不变的3D表示。我们生成的真实对应关系与现有数据集相比将极线误差减少了最多40倍，能够在极端视角变化下实现监督，并通过高斯属性提供自监督信号。因此，仅使用我们数据训练的最先进匹配器在公共基准测试上实现了显著的零样本性能提升，最高达17.7%。我们的工作表明，适当的几何修正后，3DGS可作为可扩展、高保真和结构丰富的数据源，为新一代鲁棒零样本图像匹配器铺平道路。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决如何利用3D高斯溅射(3DGS)技术提升零样本图像匹配性能的问题。现有基于学习的图像匹配方法严重依赖大规模、多样化的训练数据，而现有数据集在场景和视角多样性方面存在局限。虽然3DGS能生成多样化视角，但其几何不准确性和有偏见的深度阻碍了可靠对应关系的标注。这个问题很重要，因为图像匹配是现代3D视觉的基础，支持从SfM、SLAM到4D重建等多种应用，而训练数据的限制制约了模型在极端视角变化下的鲁棒性。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者从两个核心问题出发设计方法：1)能否仅依赖3DGS构建无需额外预训练的数据管道？2)如何利用高斯基元属性指导2D匹配器学习视角不变的3D表示？作者借鉴了多项现有工作：3DGS本身用于场景重建；改进深度渲染的方法如平面高斯模型；表示对齐方法如CLIP中的对比学习；以及LoFTR和ELoFTR等图像匹配方法作为基线。通过整合这些技术并针对图像匹配任务进行改进，作者设计了MatchGS框架。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是通过几何忠实的数据生成管道和2D-3D表示对齐策略，将3DGS的显式3D知识注入2D匹配器。数据生成管道包括：改进3DGS几何精度、平面高斯模型深度渲染、扰动生成增强视角和预渲染质量检查。表示对齐策略包括：粗粒度补丁到体素的特征对齐(使用InfoNCE损失)和细粒度高斯属性对齐。整体流程是：先使用改进的3DGS重建场景，生成精确深度图和多样化视角图像对，然后训练匹配模型同时使用对应关系监督和表示对齐作为自监督信号，最终实现具有3D感知能力的零样本图像匹配器。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)高保真数据生成管道，通过平面高斯模型和深度正则化解决3DGS几何不准确问题；2)2D-3D表示对齐策略，使匹配器具有视角不变的3D感知能力；3)有效的零样本泛化，在ScanNet上提升17.7%，在ZEB上提升16.2%。相比传统数据集，提供更大视角多样性和更好几何一致性；相比其他合成方法，基于完整3D场景重建而非简单2D提升；相比其他3D利用方法，直接构建统一2D-3D表示空间并专注于半密集匹配任务。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'MatchGS通过改进3D高斯溅射的几何精度并开发2D-3D表示对齐策略，实现了高质量的零样本半密集图像匹配，显著提升了模型在极端视角变化下的泛化能力。'}


### 论文摘要

Learning-based image matching critically depends on large-scale, diverse, and geometrically accurate training data. 3D Gaussian Splatting (3DGS) enables photorealistic novel-view synthesis and thus is attractive for data generation. However, its geometric inaccuracies and biased depth rendering currently prevent robust correspondence labeling. To address this, we introduce MatchGS, the first framework designed to systematically correct and leverage 3DGS for robust, zero-shot image matching. Our approach is twofold: (1) a geometrically-faithful data generation pipeline that refines 3DGS geometry to produce highly precise correspondence labels, enabling the synthesis of a vast and diverse range of viewpoints without compromising rendering fidelity; and (2) a 2D-3D representation alignment strategy that infuses 3DGS' explicit 3D knowledge into the 2D matcher, guiding 2D semi-dense matchers to learn viewpoint-invariant 3D representations. Our generated ground-truth correspondences reduce the epipolar error by up to 40 times compared to existing datasets, enable supervision under extreme viewpoint changes, and provide self-supervisory signals through Gaussian attributes. Consequently, state-of-the-art matchers trained solely on our data achieve significant zero-shot performance gains on public benchmarks, with improvements of up to 17.7%. Our work demonstrates that with proper geometric refinement, 3DGS can serve as a scalable, high-fidelity, and structurally-rich data source, paving the way for a new generation of robust zero-shot image matchers.

---

## 107. FIELDS: Face reconstruction with accurate Inference of Expression using Learning with Direct Supervision

**论文链接:** [http://arxiv.org/abs/2511.21245v1](http://arxiv.org/abs/2511.21245v1)

**作者:** Chen Ling, Henglin Shi, Hedvig Kjellström

**发布时间:** 2025-11-26

### GPT解析

### 总结

面部表情在人类交流中传递大部分情感信息，但现有3D面部重建方法常因依赖2D监督和缺乏3D真实数据而错过细微情感细节。FIELDS方法通过结合直接3D表情参数监督和辅助情感识别分支解决了这些问题，能从单张图像生成情感丰富、高度真实的面部模型，显著改善了自然场景中的面部表情识别性能。

### 背景

面部表情在人类交流中传递了大部分情感信息，但现有的3D面部重建方法由于依赖2D监督和缺乏3D真实数据，常常错过细微的情感细节。

### 目的

提出FIELDS方法来解决现有3D面部重建方法的限制，能够捕捉细微的情感细节。

### 方法

FIELDS通过扩展自监督的2D图像一致性线索，结合直接的3D表情参数监督和辅助情感识别分支。编码器由自发的4D面部扫描中的真实表情参数指导，而感知强度的情感损失鼓励3D表情参数捕捉真实的情感内容而不夸张。

### 主要发现

这种双重监督策略弥合了2D/3D域差距，减轻了表情强度偏差，产生高保真的3D重建，保留了细微的情感线索。

### 结论

从单张图像中，FIELDS产生情感丰富的面部模型，具有高度真实的表情，显著改善了自然场景中的面部表情识别性能，同时不失自然性。

### 翻译

面部表情在人类交流中传递了大部分情感信息，然而现有的3D面部重建方法通常由于依赖2D监督和缺乏3D真实数据而错过细微的情感细节。我们提出了FIELDS（使用直接监督学习进行准确表情推断的面部重建）方法，通过扩展自监督的2D图像一致性线索，结合直接的3D表情参数监督和辅助情感识别分支来解决这些限制。我们的编码器由自发的4D面部扫描中的真实表情参数指导，而感知强度的情感损失鼓励3D表情参数捕捉真实的情感内容而不夸张。这种双重监督策略弥合了2D/3D域差距，减轻了表情强度偏差，产生高保真的3D重建，保留了细微的情感线索。从单张图像中，FIELDS产生情感丰富的面部模型，具有高度真实的表情，显著改善了自然场景中的面部表情识别性能，同时不失自然性。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决3D人脸重建中准确捕捉面部表情的问题，特别是细微情感细节。现有方法依赖2D监督且缺乏3D真实标签，导致难以捕捉细微情感表达，且过度强调表情强度使结果不自然。这个问题在现实和研究中的重要性在于：面部表情传达人际交流中大部分情感信息；精确的表情识别对社交机器人、驾驶员疲劳检测、临床疼痛评估等应用至关重要；需要能准确捕捉几何感知表情表示的方法来捕捉潜在情感信号。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先分析了现有方法的局限性：自监督3D人脸重建缺乏统一的3D表情真实标签，导致2D一致性损失使解码器补偿编码器误差而非改进几何预测，且2D情感一致性损失过度强调表情强度。作者设计双重监督策略扩展自监督范式：直接3D参数指导和辅助情感识别模块。借鉴了FLAME模型、TEASER的Token引导合成器、EMOCA的情感一致性损失、SMIRK的神经渲染和交替优化，以及BP4D数据集获取真实3D表情参数。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是通过结合2D/3D混合监督策略，直接使用3D表情参数指导和强度感知的情感损失，生成既保留几何准确性又富含真实情感内容的3D人脸重建。整体流程包括：1)架构设计：FLAME模型参数化面部几何，编码器分为几何编码器和标记编码器，神经合成器生成图像，情感识别头预测情感标签；2)优化策略：3D表情监督使用BP4D真实参数，辅助情感监督使用强度感知损失，2D一致性监督确保图像级一致，参数正则化保持稳定性；3)训练过程：交替优化编码器和合成器，防止组件间误差补偿。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)混合2D/3D监督：扩展2D自监督，直接添加FLAME表情参数监督；2)BP4D数据集对齐：将FLAME拟合到BP4D自发表情扫描获取真实表情参数；3)辅助情感监督：添加并行情感识别头，使用强度感知损失约束夸张。相比之前工作的不同：相比EMOCA不依赖外部情感一致性损失而直接使用3D真实标签；相比SMIRK添加情感识别分支防止过度强调表情强度；相比TEASER直接监督3D表情参数而非仅依赖2D一致性；解决了2D/3D域差距问题，减轻表情强度偏差。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'FIELDS通过结合直接3D表情参数监督和强度感知情感损失，实现了在保持高保真人脸重建的同时显著提升表情识别性能，解决了传统方法中2D监督缺乏和表情过度夸张的问题。'}


### 论文摘要

Facial expressions convey the bulk of emotional information in human communication, yet existing 3D face reconstruction methods often miss subtle affective details due to reliance on 2D supervision and lack of 3D ground truth. We propose FIELDS (Face reconstruction with accurate Inference of Expression using Learning with Direct Supervision) to address these limitations by extending self-supervised 2D image consistency cues with direct 3D expression parameter supervision and an auxiliary emotion recognition branch. Our encoder is guided by authentic expression parameters from spontaneous 4D facial scans, while an intensity-aware emotion loss encourages the 3D expression parameters to capture genuine emotion content without exaggeration. This dual-supervision strategy bridges the 2D/3D domain gap and mitigates expression-intensity bias, yielding high-fidelity 3D reconstructions that preserve subtle emotional cues. From a single image, FIELDS produces emotion-rich face models with highly realistic expressions, significantly improving in-the-wild facial expression recognition performance without sacrificing naturalness.

---

## 108. Privacy in Federated Learning with Spiking Neural Networks

**论文链接:** [http://arxiv.org/abs/2511.21181v1](http://arxiv.org/abs/2511.21181v1)

**作者:** Dogukan Aksu, Jesus Martinez del Rincon, Ihsen Alouani

**发布时间:** 2025-11-26

### GPT解析

### 总结

本文研究了脉冲神经网络(SNNs)在梯度泄漏方面的隐私保护特性，发现SNNs的梯度能够产生嘈杂、时间不一致的重构，无法恢复有意义的时空结构，表明脉冲计算具有固有的隐私保护潜力。

### 背景

脉冲神经网络已成为嵌入式和边缘AI的重要候选技术，其固有的低功耗特性使其在能源预算受限场景中比传统人工神经网络更高效。联邦学习已成为此类环境中的主流训练范式，但梯度反转攻击代表了一个关键隐私威胁，虽然这种脆弱性在传统ANNs中已被广泛研究，但对SNNs的影响仍 largely unexplored。

### 目的

对SNNs中梯度泄漏进行全面实证研究，调查不同数据域中SNNs的隐私保护特性，评估脉冲神经网络在梯度反转攻击中的表现。

### 方法

适应不同的梯度泄漏攻击到脉冲域，对比SNNs和传统ANNs在梯度泄露方面的表现，进行系统性实验评估梯度反转攻击对脉冲架构的影响。

### 主要发现

SNNs本质上是非可微的，通常使用替代梯度进行训练，这些替代梯度与原始输入的相关性较低；SNNs梯度产生嘈杂、时间不一致的重构，无法恢复有意义的时空结构；事件驱动动力学和替代梯度训练的组合显著降低了梯度的信息量。

### 结论

脉冲计算具有固有的隐私保护潜力，该工作首次为脉冲架构提供了梯度反转攻击的系统基准，表明SNNs在隐私保护方面优于传统ANNs。

### 翻译

脉冲神经网络(SNNs)已成为嵌入式和边缘AI的重要候选技术。其固有的低功耗特性使其在能源预算受限的场景中比传统人工神经网络(ANNs)更高效。同时，联邦学习(FL)已成为此类环境中的主流训练范式，使设备端学习成为可能，同时限制原始数据的暴露。然而，梯度反转攻击是FL中的一个关键隐私威胁，敏感训练数据可以直接从共享梯度中重建。虽然这种脆弱性在传统ANNs中已被广泛研究，但对SNNs的影响仍 largely unexplored。在本工作中，我们首次对SNNs在不同数据域中的梯度泄漏进行了全面的实证研究。SNNs本质上是非可微的，通常使用替代梯度进行训练，我们假设这些替代梯度与原始输入的相关性较低，因此在隐私方面的信息量较少。为此，我们将不同的梯度泄漏攻击适应到脉冲域。我们的实验揭示了与传统ANNs的鲜明对比：虽然ANN梯度可靠地暴露了显著的输入内容，但SNN梯度产生了嘈杂、时间不一致的重构，无法恢复有意义的时空结构。这些结果表明，事件驱动动力学和替代梯度训练的组合显著降低了梯度的信息量。据我们所知，这项工作首次为脉冲架构提供了梯度反转攻击的系统基准，突显了神经形态计算的固有隐私保护潜力。


### 论文摘要

Spiking neural networks (SNNs) have emerged as prominent candidates for embedded and edge AI. Their inherent low power consumption makes them far more efficient than conventional ANNs in scenarios where energy budgets are tightly constrained. In parallel, federated learning (FL) has become the prevailing training paradigm in such settings, enabling on-device learning while limiting the exposure of raw data. However, gradient inversion attacks represent a critical privacy threat in FL, where sensitive training data can be reconstructed directly from shared gradients. While this vulnerability has been widely investigated in conventional ANNs, its implications for SNNs remain largely unexplored. In this work, we present the first comprehensive empirical study of gradient leakage in SNNs across diverse data domains. SNNs are inherently non-differentiable and are typically trained using surrogate gradients, which we hypothesized would be less correlated with the original input and thus less informative from a privacy perspective. To investigate this, we adapt different gradient leakage attacks to the spike domain. Our experiments reveal a striking contrast with conventional ANNs: whereas ANN gradients reliably expose salient input content, SNN gradients yield noisy, temporally inconsistent reconstructions that fail to recover meaningful spatial or temporal structure. These results indicate that the combination of event-driven dynamics and surrogate-gradient training substantially reduces gradient informativeness. To the best of our knowledge, this work provides the first systematic benchmark of gradient inversion attacks for spiking architectures, highlighting the inherent privacy-preserving potential of neuromorphic computation.

---

## 109. AV-Edit: Multimodal Generative Sound Effect Editing via Audio-Visual Semantic Joint Control

**论文链接:** [http://arxiv.org/abs/2511.21146v1](http://arxiv.org/abs/2511.21146v1)

**作者:** Xinyue Guo, Xiaoran Yang, Lipan Zhang, Jianxuan Yang, Zhao Wang, Jian Luan

**发布时间:** 2025-11-26

### GPT解析

### 总结

本文提出了AV-Edit，一个生成式音效编辑框架，能够通过联合利用视觉、音频和文本语义对视频中的音轨进行细粒度编辑，解决了现有方法仅依赖低级信号处理或粗略文本提示导致的灵活性和音频质量有限的问题。

### 背景

当前音效编辑方法仅依赖于低级信号处理或粗略的文本提示，导致灵活性和音频质量有限。

### 目的

提出AV-Edit框架，通过联合利用视觉、音频和文本语义，实现对视频现有音轨的细粒度编辑。

### 方法

提出对比音频视觉掩码自编码器(CAV-MAE-Edit)进行多模态预训练，学习对齐的跨模态表示；训练编辑式多模态扩散变换器(MM-DiT)，能够移除视觉不相关声音并生成与视频内容一致的缺失音频元素；构建专门的视频音效编辑数据集作为评估基准。

### 主要发现

实验表明AV-Edit能基于视觉内容生成高质量音频并进行精确修改，在音效编辑领域取得最先进性能，在音频生成领域表现强大竞争力。

### 结论

AV-Edit是一个有效的音效编辑框架，能够通过结合视觉、音频和文本语义实现高质量的音效编辑。

### 翻译

音效编辑-通过添加、移除或替换元素来修改音频-仍然受到现有方法的限制，这些方法仅依赖于低级信号处理或粗略的文本提示，通常导致灵活性和音频质量有限。为解决这一问题，我们提出了AV-Edit，一个生成式音效编辑框架，通过联合利用视觉、音频和文本语义，能够对视频中的现有音轨进行细粒度编辑。具体来说，所提出的方法采用一种专门设计的对比音频视觉掩码自编码器进行多模态预训练，学习对齐的跨模态表示。然后使用这些表示训练一个编辑式多模态扩散变换器，它能够通过基于相关性的特征门控训练策略移除视觉上不相关的声音，并生成与视频内容一致的缺失音频元素。此外，我们构建了一个专门的视频音效编辑数据集作为评估基准。实验证明，所提出的AV-Edit能够基于视觉内容生成高质量音频并进行精确修改，在音效编辑领域取得了最先进的性能，并在音频生成领域表现出强大的竞争力。


### 论文摘要

Sound effect editing-modifying audio by adding, removing, or replacing elements-remains constrained by existing approaches that rely solely on low-level signal processing or coarse text prompts, often resulting in limited flexibility and suboptimal audio quality. To address this, we propose AV-Edit, a generative sound effect editing framework that enables fine-grained editing of existing audio tracks in videos by jointly leveraging visual, audio, and text semantics. Specifically, the proposed method employs a specially designed contrastive audio-visual masking autoencoder (CAV-MAE-Edit) for multimodal pre-training, learning aligned cross-modal representations. These representations are then used to train an editorial Multimodal Diffusion Transformer (MM-DiT) capable of removing visually irrelevant sounds and generating missing audio elements consistent with video content through a correlation-based feature gating training strategy. Furthermore, we construct a dedicated video-based sound editing dataset as an evaluation benchmark. Experiments demonstrate that the proposed AV-Edit generates high-quality audio with precise modifications based on visual content, achieving state-of-the-art performance in the field of sound effect editing and exhibiting strong competitiveness in the domain of audio generation.

---

## 110. Dynamic Stratified Contrastive Learning with Upstream Augmentation for MILP Branching

**论文链接:** [http://arxiv.org/abs/2511.21107v1](http://arxiv.org/abs/2511.21107v1)

**作者:** Tongkai Lu, Shuai Ma, Chongyang Tao

**发布时间:** 2025-11-26

**备注:** 18 pages

### GPT解析

### 总结

该研究提出了一种名为DSC-MILP Branching的动态分层对比训练框架，用于解决混合整数线性规划(MILP)中的分支问题，有效提高了分支准确性和求解效率。

### 背景

混合整数线性规划(MILP)是一类NP难问题，分支定界(B&B)是解决MILP的主导方法，其中分支策略至关重要。基于神经学习的框架最近被开发用于增强分支策略，但仍面临深度间语义变化、上游节点稀缺和强分支样本收集成本高等挑战。

### 目的

解决现有神经学习方法在MILP分支问题中的挑战，提高分支策略的效率和MILP求解的准确性，特别是针对上游节点。

### 方法

提出DSC-MILP Branching框架，基于特征分布对分支定界节点进行分组，训练GCNN判别模型逐步分离不同组节点，学习细粒度节点表示；引入上游增强MILP推导程序生成等效和扰动实例，解决数据稀缺和不平衡问题。

### 主要发现

DSC-MILP能有效建模节点间细微语义差异，显著提高分支准确性和求解效率，尤其对上游节点效果明显。

### 结论

在标准MILP基准上的实验表明，该方法增强了分支准确性，减少了求解时间，并能有效泛化到未见过的实例。

### 翻译

混合整数线性规划(MILP)是一类基础NP难问题，受到学术界和工业界的广泛关注。分支定界(B&B)方法是解决MILP的主导方法，分支在B&B方法中起重要作用。最近已开发基于神经学习的框架来增强分支策略和提高MILP求解效率。然而，这些方法仍然面临不同深度间的语义变化、上游节点稀缺以及强分支样本收集成本高等问题。为解决这些问题，我们提出了DSC-MILP Branching，一种用于MILP分支的动态分层对比训练框架。它基于特征分布对分支定界节点进行分组，并训练基于GCNN的判别模型逐步分离不同组中的节点，在整个树中学习更细粒度的节点表示。为解决上游节点的数据稀缺和不平衡问题，我们引入了上游增强的MILP推导程序，生成理论上等效和扰动实例。DSC-MILP有效建模了节点间的细微语义差异，显著提高了分支准确性和求解效率，特别是对上游节点。在标准MILP基准上的广泛实验表明，我们的方法增强了分支准确性，减少了求解时间，并能有效泛化到未见过的实例。


### 论文摘要

Mixed Integer Linear Programming (MILP) is a fundamental class of NP-hard problems that has garnered significant attention from both academia and industry. The Branch-and-Bound (B\&B) method is the dominant approach for solving MILPs and the branching plays an important role in B\&B methods. Neural-based learning frameworks have recently been developed to enhance branching policies and the efficiency of solving MILPs. However, these methods still struggle with semantic variation across depths, the scarcity of upstream nodes, and the costly collection of strong branching samples. To address these issues, we propose \ours, a Dynamic \underline{\textbf{S}}tratified \underline{\textbf{C}}ontrastive Training Framework for \underline{\textbf{MILP}} Branching. It groups branch-and-bound nodes based on their feature distributions and trains a GCNN-based discriminative model to progressively separate nodes across groups, learning finer-grained node representations throughout the tree. To address data scarcity and imbalance at upstream nodes, we introduce an upstream-augmented MILP derivation procedure that generates both theoretically equivalent and perturbed instances. \ours~effectively models subtle semantic differences between nodes, significantly enhancing branching accuracy and solving efficiency, particularly for upstream nodes. Extensive experiments on standard MILP benchmarks demonstrate that our method enhances branching accuracy, reduces solving time, and generalizes effectively to unseen instances.

---

## 111. OVOD-Agent: A Markov-Bandit Framework for Proactive Visual Reasoning and Self-Evolving Detection

**论文链接:** [http://arxiv.org/abs/2511.21064v1](http://arxiv.org/abs/2511.21064v1)

**作者:** Chujie Wang, Jianyu Lu, Zhiyuan Luo, Xi Chen, Chu He

**发布时间:** 2025-11-26

### GPT解析

### 总结

本文提出了一种名为OVOD-Agent的新型开放词汇目标检测框架，通过将被动类别匹配转变为主动视觉推理和自我演化检测，显著提升了检测器在稀有类别上的性能。

### 背景

现有开放词汇目标检测方法虽然在大型视觉语言数据集上预训练，但其推理仍局限于固定类别名称，造成多模态训练与单模态推理之间的差距，且文本表示空间仍有探索空间。

### 目的

开发一种能够主动进行视觉推理并自我演化的检测框架，扩展文本优化过程为具有明确解释性动作的可视化思维链，解决现有方法的局限性。

### 方法

OVOD-Agent受思维链范式启发，将视觉上下文转换建模为八个状态空间上的弱马尔可夫决策过程，包含Bandit模块生成探索信号，并将马尔可夫转移矩阵与Bandit轨迹集成形成闭环学习系统。

### 主要发现

在COCO和LVIS数据集上的实验表明，OVOD-Agent在各种OVOD骨干网络上提供了一致的性能提升，特别是在稀有类别上表现突出。

### 结论

所提出的OVOD-Agent框架有效解决了开放词汇目标检测中的关键挑战，通过主动视觉推理和自我演化机制显著提升了检测性能，特别是在处理稀有类别时。

### 翻译

开放词汇目标检测旨在利用语义信息使检测器能够泛化到不同类别。尽管现有方法在大型视觉语言数据集上进行了预训练，但其推理仍局限于固定类别名称，这造成了多模态训练和单模态推理之间的差距。先前的研究表明，改进文本表示可以显著提高OVOD性能，表明文本空间仍有探索空间。为此，我们提出了OVOD-Agent，它将被动类别匹配转变为主动视觉推理和自我演化的检测。受思维链范式的启发，OVOD-Agent将文本优化过程扩展为具有明确解释性动作的可解释Visual-CoT。OVOD的轻量级特性使得基于大型语言模型的管理不合适；相反，我们将视觉上下文转换建模为八个状态空间上的弱马尔可夫决策过程，自然地表示代理的状态、记忆和交互动态。一个Bandit模块在有限监督下生成探索信号，帮助代理关注不确定区域并调整其检测策略。我们进一步将马尔可夫转移矩阵与Bandit轨迹集成，用于自监督奖励模型优化，形成从Bandit探索到RM学习的闭环。在COCO和LVIS上的实验表明，OVOD-Agent在各种OVOD骨干网络上提供了一致的改进，特别是在稀有类别上，证实了所提出框架的有效性。


### 论文摘要

Open-Vocabulary Object Detection (OVOD) aims to enable detectors to generalize across categories by leveraging semantic information. Although existing methods are pretrained on large vision-language datasets, their inference is still limited to fixed category names, creating a gap between multimodal training and unimodal inference. Previous work has shown that improving textual representation can significantly enhance OVOD performance, indicating that the textual space is still underexplored. To this end, we propose OVOD-Agent, which transforms passive category matching into proactive visual reasoning and self-evolving detection. Inspired by the Chain-of-Thought (CoT) paradigm, OVOD-Agent extends the textual optimization process into an interpretable Visual-CoT with explicit actions. OVOD's lightweight nature makes LLM-based management unsuitable; instead, we model visual context transitions as a Weakly Markovian Decision Process (w-MDP) over eight state spaces, which naturally represents the agent's state, memory, and interaction dynamics. A Bandit module generates exploration signals under limited supervision, helping the agent focus on uncertain regions and adapt its detection policy. We further integrate Markov transition matrices with Bandit trajectories for self-supervised Reward Model (RM) optimization, forming a closed loop from Bandit exploration to RM learning. Experiments on COCO and LVIS show that OVOD-Agent provides consistent improvements across OVOD backbones, particularly on rare categories, confirming the effectiveness of the proposed framework.

---

## 112. Efficient Diffusion Planning with Temporal Diffusion

**论文链接:** [http://arxiv.org/abs/2511.21054v1](http://arxiv.org/abs/2511.21054v1)

**作者:** Jiaming Guo, Rui Zhang, Zerun Li, Yunkai Gao, Shaohui Peng, Siming Lan, Xing Hu, Zidong Du, Xishan Zhang, Ling Li

**发布时间:** 2025-11-26

**备注:** Accepted by the AAAI26 Conference Main Track

### GPT解析

### 总结

本文提出了一种时间扩散规划器(TDP)，通过将去噪步骤分布在时间维度上提高决策效率，减少计算开销，同时保持或提高性能。

### 背景

扩散规划是一种从离线数据中学习高性能策略的很有前景的方法，但传统方法在每个时间步都生成新计划，导致计算开销大、决策频率低，且频繁计划切换可能影响性能。

### 目的

减少扩散规划的计算开销，提高决策频率，同时保持或提高性能，避免规划与现实之间的差异对性能的影响。

### 方法

提出时间扩散规划器(TDP)，首先生成随时间推移逐渐模糊的初始计划，然后在每个后续时间步用少量去噪步骤更新之前的计划，而非生成全新计划。同时引入自动重规划机制防止计划与现实出现显著偏差。

### 主要发现

在D4RL上的实验表明，TDP相比在每个时间步生成新计划的前期工作，将决策频率提高了11-24.8倍，同时实现了更高或相当的性能。

### 结论

TDP通过智能分配去噪步骤在时间维度上，有效提高了决策效率，减少了计算开销，同时保持了高性能，为离线数据学习高性能策略提供了更有效的方法。

### 翻译

扩散规划是一种很有前景的从离线数据中学习高性能策略的方法。为了避免规划与现实之间的差异对性能的影响，之前的工作在每个时间步都会生成新计划。然而，这带来了巨大的计算开销，导致决策频率降低，且频繁的计划切换也可能影响性能。相比之下，人类可能会创建详细的短期计划和更概括、有时模糊的长期计划，并随时间调整它们。受此启发，我们提出了时间扩散规划器(TDP)，通过将去噪步骤分布在时间维度上提高决策效率。TDP首先生成一个初始计划，随时间推移逐渐变得模糊。在每个后续时间步，TDP不会生成全新的计划，而是用少量去噪步骤更新之前的计划。这减少了平均去噪步骤数量，提高了决策效率。此外，我们引入了自动重规划机制，防止计划与现实之间出现显著偏差。在D4RL上的实验表明，与在每个时间步生成新计划的前期工作相比，TDP将决策频率提高了11-24.8倍，同时实现了更高或相当的性能。


### 论文摘要

Diffusion planning is a promising method for learning high-performance policies from offline data. To avoid the impact of discrepancies between planning and reality on performance, previous works generate new plans at each time step. However, this incurs significant computational overhead and leads to lower decision frequencies, and frequent plan switching may also affect performance. In contrast, humans might create detailed short-term plans and more general, sometimes vague, long-term plans, and adjust them over time. Inspired by this, we propose the Temporal Diffusion Planner (TDP) which improves decision efficiency by distributing the denoising steps across the time dimension. TDP begins by generating an initial plan that becomes progressively more vague over time. At each subsequent time step, rather than generating an entirely new plan, TDP updates the previous one with a small number of denoising steps. This reduces the average number of denoising steps, improving decision efficiency. Additionally, we introduce an automated replanning mechanism to prevent significant deviations between the plan and reality. Experiments on D4RL show that, compared to previous works that generate new plans every time step, TDP improves the decision-making frequency by 11-24.8 times while achieving higher or comparable performance.

---

## 113. AerialMind: Towards Referring Multi-Object Tracking in UAV Scenarios

**论文链接:** [http://arxiv.org/abs/2511.21053v1](http://arxiv.org/abs/2511.21053v1)

**作者:** Chenglizhao Chen, Shaofeng Liang, Runwei Guan, Xiaolou Sun, Haocheng Zhao, Haiyun Jiang, Tao Huang, Henghui Ding, Qing-Long Han

**发布时间:** 2025-11-26

**备注:** AAAI 2026

### GPT解析

### 总结

AerialMind是首个无人机场景的大规模参考多目标跟踪基准测试，通过半自动化标记框架COALA和创新方法HETrack，解决了无人机场景下自然语言交互目标跟踪的问题。

### 背景

参考多目标跟踪(RMOT)研究目前主要局限于地面场景，限制了捕捉大范围场景上下文的能力。无人机凭借其广阔视角和机动性可实现大范围监控，且作为具身智能平台，对具有自然语言交互能力的智能空中系统有迫切需求。

### 目的

引入AerialMind，填补无人机场景下RMOT研究的空白，构建首个大规模无人机场景RMOT基准测试。

### 方法

开发半自动化协作代理标记助手(COALA)框架降低标注成本，提出HawkEyeTrack (HETrack)方法协同增强视觉语言表示学习并改进无人机场景感知。

### 主要发现

全面的实验验证了该数据集具有挑战性，所提出的方法是有效的。

### 结论

AerialMind填补了无人机场景下RMOT研究的空白，为智能空中系统提供了自然语言交互能力的基础。

### 翻译

参考多目标跟踪(RMOT)旨在通过自然语言指令实现精确的目标检测和跟踪，这是智能机器人的基本能力。然而，当前的RMOT研究主要局限于地面场景，限制了其捕捉大范围场景上下文和进行全面跟踪与路径规划的能力。相比之下，无人机凭借其广阔的空中视角和卓越的机动性能够实现大范围监控。此外，无人机已成为具身智能的关键平台，这催生了对具有自然语言交互能力的智能空中系统的需求。为此，我们引入了AerialMind，这是首个无人机场景的大规模RMOT基准测试，旨在填补这一研究空白。为促进其构建，我们开发了一种创新的半自动化协作代理标记助手(COALA)框架，显著降低了人工成本同时保持了标注质量。此外，我们提出了HawkEyeTrack (HETrack)这一新方法，协同增强视觉语言表示学习并改进对无人机场景的感知。全面的实验验证了我们的数据集具有挑战性，且我们的方法有效。


### 论文摘要

Referring Multi-Object Tracking (RMOT) aims to achieve precise object detection and tracking through natural language instructions, representing a fundamental capability for intelligent robotic systems. However, current RMOT research remains mostly confined to ground-level scenarios, which constrains their ability to capture broad-scale scene contexts and perform comprehensive tracking and path planning. In contrast, Unmanned Aerial Vehicles (UAVs) leverage their expansive aerial perspectives and superior maneuverability to enable wide-area surveillance. Moreover, UAVs have emerged as critical platforms for Embodied Intelligence, which has given rise to an unprecedented demand for intelligent aerial systems capable of natural language interaction. To this end, we introduce AerialMind, the first large-scale RMOT benchmark in UAV scenarios, which aims to bridge this research gap. To facilitate its construction, we develop an innovative semi-automated collaborative agent-based labeling assistant (COALA) framework that significantly reduces labor costs while maintaining annotation quality. Furthermore, we propose HawkEyeTrack (HETrack), a novel method that collaboratively enhances vision-language representation learning and improves the perception of UAV scenarios. Comprehensive experiments validated the challenging nature of our dataset and the effectiveness of our method.

---

## 114. FedAPA: Federated Learning with Adaptive Prototype Aggregation Toward Heterogeneous Wi-Fi CSI-based Crowd Counting

**论文链接:** [http://arxiv.org/abs/2511.21048v1](http://arxiv.org/abs/2511.21048v1)

**作者:** Jingtao Guo, Yuyi Mao, Ivan Wang-Hei Ho

**发布时间:** 2025-11-26

**备注:** 17 pages, 11 figures, this article was submitted to IEEE for possible publication

### GPT解析

### 总结

FedAPA是一种基于自适应原型聚合的协作Wi-Fi CSI感知算法，通过相似度权重分配实现个性化全局原型，结合分类学习和表示对比学习对齐本地与全局知识，在多环境人群计数中显著优于基线方法。

### 背景

Wi-Fi CSI感知提供非侵入式活动识别方法，但大规模部署受限于需要大量特定站点训练数据；联邦学习虽可避免原始数据共享，但面临异构感知数据和设备资源挑战。

### 目的

解决联邦学习在异构感知数据和设备资源方面的挑战，提出一种协作的基于Wi-Fi CSI的感知算法。

### 方法

提出FedAPA算法，使用自适应原型聚合策略基于相似度为对等原型分配权重，实现自适应客户端贡献和个性化全局原型；本地训练采用混合目标，结合分类学习和表示对比学习对齐本地和全局知识，并提供收敛分析。

### 主要发现

在六环境和最多20人的真实分布式Wi-Fi人群计数场景中评估，FedAPA在准确性、F1分数、MAE和通信开销方面优于多个基线，实现至少9.65%的准确性提升、9%的F1分数提升、0.29的MAE降低和95.94%的通信开销减少。

### 结论

FedAPA能有效解决异构数据和资源限制问题，是一种高效的协作Wi-Fi CSI感知算法。

### 翻译

基于Wi-Fi信道状态信息(CSI)的感知提供了一种非侵入式、无需设备的活动识别和人群计数方法，但大规模部署受到需要大量特定站点训练数据的阻碍。联邦学习(FL)提供了一种避免原始数据共享的方式，但面临着异构感知数据和设备资源的挑战。本文提出了FedAPA，一种协作的基于Wi-Fi CSI的感知算法，使用自适应原型聚合(APA)策略为对等原型分配基于相似度的权重，实现自适应客户端贡献，并为每个客户端生成个性化的全局原型而非固定权重聚合。在本地训练中，我们采用结合分类学习和表示对比学习的混合目标，对齐本地和全局知识。我们提供了FedAPA的收敛分析，并在六个环境和最多20人的真实分布式Wi-Fi人群计数场景中对其进行了评估。结果表明，在准确性、F1分数、平均绝对误差(MAE)和通信开销方面，我们的方法优于多个基线，FedAPA实现了至少9.65%的准确性提升、9%的F1分数提升、0.29的MAE降低和95.94%的通信开销减少。


### 论文摘要

Wi-Fi channel state information (CSI)-based sensing provides a non-invasive, device-free approach for tasks such as human activity recognition and crowd counting, but large-scale deployment is hindered by the need for extensive site-specific training data. Federated learning (FL) offers a way to avoid raw data sharing but is challenged by heterogeneous sensing data and device resources. This paper proposes FedAPA, a collaborative Wi-Fi CSI-based sensing algorithm that uses adaptive prototype aggregation (APA) strategy to assign similarity-based weights to peer prototypes, enabling adaptive client contributions and yielding a personalized global prototype for each client instead of a fixed-weight aggregation. During local training, we adopt a hybrid objective that combines classification learning with representation contrastive learning to align local and global knowledge. We provide a convergence analysis of FedAPA and evaluate it in a real-world distributed Wi-Fi crowd counting scenario with six environments and up to 20 people. The results show that our method outperform multiple baselines in terms of accuracy, F1 score, mean absolute error (MAE), and communication overhead, with FedAPA achieving at least a 9.65% increase in accuracy, a 9% gain in F1 score, a 0.29 reduction in MAE, and a 95.94% reduction in communication overhead.

---

## 115. Different Origins of Nucleated and Non-nucleated Dwarf Elliptical Galaxies: Identified by the Deep-learning

**论文链接:** [http://arxiv.org/abs/2511.21010v1](http://arxiv.org/abs/2511.21010v1)

**作者:** Sanjaya Paudel, Cristiano G. Sabiu, Suk-Jin Yoon, Daya Nidhi Chhatkuli, Woong-Bae G. Zee, Jaewon Yoo, Binod Adhikari

**发布时间:** 2025-11-26

**DOI:** 10.3847/1538-4357/ae2097

**备注:** Accepted for publication in ApJ

### GPT解析

### 总结

该研究探讨了室女座星系团中矮椭圆星系的分布特征和形成机制，发现核化与非核化矮椭圆星系具有不同的空间分布和起源

### 背景

矮椭圆星系是星系团中的主导群体，作为研究环境对星系演化影响的理想探针。许多矮椭圆星系拥有中心核，这些是宇宙中最密集的恒星系统之一。然而，核化与非核化矮椭圆星系的大尺度分布和起源问题尚未解决

### 目的

研究核化和非核化矮椭圆星系的大尺度分布特征，并探讨它们的形成机制

### 方法

使用最先进的机器学习框架，系统扫描室女座星系团区域，构建了包含2,123个矮椭圆星系的最大同质样本，并对它们进行了可靠的核分类

### 主要发现

核化矮椭圆星系在空间上更聚集，并与大质量星系有更强的关联，表明它们可能是在星系团内与大质量星系一起形成的；非核化矮椭圆星系在星系团内分布更广泛，与星系团的整体势阱更一致，表明它们起源于星系团外，后来被吸积并在星系团内重新分布

### 结论

研究结果为矮椭圆星系及其中心核的形成和演化提供了新的见解，揭示了环境因素对星系演化的重要影响

### 翻译

矮椭圆星系(dEs)是星系团中的主导群体，是研究环境对星系演化影响的理想探针。已知相当一部分dEs拥有中心核，这些是宇宙中最密集的恒星系统之一。核化和非核化dEs的大尺度分布和基本起源仍未解决。使用最先进的机器学习框架，我们系统扫描室女座星系团区域，构建了包含2,123个dEs的最大同质样本，并进行了可靠的核分类。我们发现核化dEs在空间上更聚集，并与大质量星系表现出比非核化dEs更强的关联。这表明大多数核化dEs很可能与大质量星系一起在星系团内形成（即原地形成）。相比之下，非核化dEs在星系团内分布更广泛，与室女座的整体势阱更一致（由星系团的热气体追踪）。这表明大多数非核化dEs起源于星系团外（即外部形成），后来被吸积并在星系团内重新分布。我们的发现为dEs及其中心核的形成和演化提供了新的见解


### 论文摘要

Dwarf elliptical galaxies (dEs) are the dominant population in galaxy clusters and serve as ideal probes for studying the environmental impact on galactic evolution. A substantial fraction of dEs are known to harbor central nuclei, which are among the densest stellar systems in the Universe. The large-scale distribution and the underlying origin of nucleated and non-nucleated dEs remain unresolved. Using a state-of-the-art machine learning framework, we systematically scan the Virgo cluster region ($15\arcdeg \times 20\arcdeg$ centered at $R.A. = 187.2\arcdeg$ and $Dec. = 9.6\arcdeg$) and construct the largest homogeneous sample of dEs (of total 2,123) with robust nucleus classifications. We find that nucleated dEs are more spatially clustered and exhibit a stronger association with massive galaxies than their non-nucleated counterparts. This suggests that most nucleated dEs likely formed alongside massive galaxies within the cluster (i.e, the in-situ formation). In contrast, non-nucleated dEs are more widely distributed across the cluster and align more closely with Virgo's global potential well, as traced by the cluster's hot gas. This indicates that most non-nucleated dEs originated outside the cluster (i.e, the ex-situ formation) and were later accreted and redistributed within it. Our findings shed new light on how dEs and their central nuclei form and evolve.

---

## 116. FANoise: Singular Value-Adaptive Noise Modulation for Robust Multimodal Representation Learning

**论文链接:** [http://arxiv.org/abs/2511.20997v1](http://arxiv.org/abs/2511.20997v1)

**作者:** Jiaoyang Li, Jun Fang, Tianhao Gao, Xiaohui Zhang, Zhiyuan Liu, Chao Liu, Pengzhang Liu, Qixia Jiang

**发布时间:** 2025-11-26

**备注:** 13 pages, 5 figures, accept to AAAI2026

### GPT解析

### 总结

本文提出了一种名为FANoise的特征自适应噪声注入策略，用于改进多模态表示学习。该方法通过利用对比学习的动态特性，有效减轻了噪声的负面影响同时保留其益处，在各种基础视觉语言模型上提高了多模态任务的性能。

### 背景

表示学习是现代机器学习的基础，应用于文本检索和多模态理解等领域。然而，学习鲁棒且可泛化的表示仍然具有挑战性。现有方法主要使用启发式或静态噪声，忽略了训练过程中特征分布的动态特性。

### 目的

系统研究噪声在表示学习中的作用，从梯度和特征分布两个角度进行分析，并开发一种能够有效利用噪声优势的方法。

### 方法

提出FANoise，一种新颖的特征自适应噪声注入策略，专注于多模态表示学习。该方法利用对比学习的动态特性，有效减轻噪声的负面影响同时保留其益处。

### 主要发现

在各种基础视觉语言模型上，FANoise一致地提高了多模态任务的性能。该方法通过动态调整噪声注入策略，解决了传统静态噪声方法的局限性。

### 结论

FANoise是一个有理论基础的方法，能够有效改进表示学习性能。综合实验证明了其在多模态任务上的有效性，为噪声注入在表示学习中的应用提供了新思路。

### 翻译

表示学习是现代机器学习的基础，支持文本检索和多模态理解等应用。然而，学习鲁棒且可泛化的表示仍然具有挑战性。虽然先前的研究表明，主动噪声注入（一种数据增强形式）可以提高编码性能，但大多数现有方法依赖于启发式或静态噪声，忽略了训练过程中特征分布的动态特性。在这项工作中，我们使用InfoNCE损失作为代表性例子，从基于梯度和特征分布两个角度系统研究了噪声在表示学习中的作用。专注于多模态表示学习，我们提出了FANoise，一种新颖的特征自适应噪声注入策略。通过利用对比学习的动态特性，FANoise有效减轻了噪声的负面影响，同时保留了其益处。在这个有理论基础的框架下，全面的实验证明，FANoise在各种基础视觉语言模型上，一致地提高了多模态任务的总体性能。


### 论文摘要

Representation learning is fundamental to modern machine learning, powering applications such as text retrieval and multimodal understanding. However, learning robust and generalizable representations remains challenging. While prior work has demonstrated that active noise injection, a form of data augmentation, can enhance encoding performance, most existing methods rely on heuristic or static noise, overlooking the dynamic nature of feature distributions during training. In this work, we systematically study the role of noise in representation learning from both gradient-based and feature distribution perspectives, using InfoNCE loss as a representative example. Focusing on multimodal representation learning, we propose FANoise, a novel feature-adaptive noise injection strategy. By leveraging the dynamics of contrastive learning, FANoise effectively mitigates the negative impacts of noise while preserving its benefits. Under this theoretically grounded framework, comprehensive experiments demonstrate that FANoise consistently improves overall performance on multimodal tasks across various base VLM models.

---

## 117. Privacy-Preserving Federated Vision Transformer Learning Leveraging Lightweight Homomorphic Encryption in Medical AI

**论文链接:** [http://arxiv.org/abs/2511.20983v1](http://arxiv.org/abs/2511.20983v1)

**作者:** Al Amin, Kamrul Hasan, Liang Hong, Sharif Ullah

**发布时间:** 2025-11-26

**备注:** 7 pages, 4 figures

### GPT解析

### 总结

本文提出了一种结合Vision Transformers和同态加密的隐私保护联邦学习框架，用于安全的跨机构组织病理学分类，在保护隐私的同时保持了高准确率并显著减少通信开销。

### 背景

医疗机构间的协作机器学习可通过多样化数据集提高诊断准确性，但隐私法规如HIPAA禁止直接共享患者数据。传统联邦学习中的模型梯度容易受到重构攻击，可能暴露敏感医疗信息。

### 目的

开发一种隐私保护的联邦学习框架，结合Vision Transformers和同态加密，实现安全的跨机构组织病理学分类，同时保护患者数据隐私。

### 方法

利用ViT的CLS令牌作为768维特征表示进行安全聚合，在传输前使用CKKS同态加密对这些令牌进行加密，实现30倍的通信减少，并在三客户端联邦设置中评估肺癌组织病理学分类。

### 主要发现

传统FL中的梯度极易受到模型反转攻击，可实现近乎完美的图像重建；提出的CLS保护的HE方法可防止此类攻击，支持密文直接推理，每次聚合仅需传输326 KB加密数据；未加密领域准确率96.12%，加密领域90.02%。

### 结论

该隐私保护联邦学习框架在保护医疗数据隐私的同时，保持了较高的分类准确率，并且显著减少了通信开销，为医疗机构的协作AI提供了安全有效的解决方案。

### 翻译

跨医疗机构协作机器学习有望通过利用多样化数据集提高诊断准确性，但HIPAA等隐私法规禁止直接共享患者数据。虽然联邦学习(FL)允许在不交换原始数据的情况下进行去中心化训练，但最近研究表明，传统FL中的模型梯度仍然容易受到重构攻击，可能暴露敏感的医疗信息。本文提出了一种结合Vision Transformers(ViT)和同态加密(HE)的隐私保护联邦学习框架，用于安全的跨机构组织病理学分类。该方法利用ViT CLS令牌作为紧凑的768维特征表示进行安全聚合，在传输到服务器之前使用CKKS同态加密对这些令牌进行加密。我们证明，与梯度加密相比，加密CLS令牌实现了30倍的通信减少，同时保持强隐私保证。在用于肺癌组织病理学分类的三客户端联邦设置评估中，我们显示梯度极易受到模型反转攻击(PSNR: 52.26 dB, SSIM: 0.999, NMI: 0.741)，能够实现近乎完美的图像重建。相比之下，提出的CLS保护的HE方法可防止此类攻击，同时支持直接在密文上进行加密推理，每次聚合轮次仅需传输326 KB的加密数据。该框架在未加密领域实现了96.12%的全局分类准确率，在加密领域实现了90.02%的准确率。


### 论文摘要

Collaborative machine learning across healthcare institutions promises improved diagnostic accuracy by leveraging diverse datasets, yet privacy regulations such as HIPAA prohibit direct patient data sharing. While federated learning (FL) enables decentralized training without raw data exchange, recent studies show that model gradients in conventional FL remain vulnerable to reconstruction attacks, potentially exposing sensitive medical information. This paper presents a privacy-preserving federated learning framework combining Vision Transformers (ViT) with homomorphic encryption (HE) for secure multi-institutional histopathology classification. The approach leverages the ViT CLS token as a compact 768-dimensional feature representation for secure aggregation, encrypting these tokens using CKKS homomorphic encryption before transmission to the server. We demonstrate that encrypting CLS tokens achieves a 30-fold communication reduction compared to gradient encryption while maintaining strong privacy guarantees. Through evaluation on a three-client federated setup for lung cancer histopathology classification, we show that gradients are highly susceptible to model inversion attacks (PSNR: 52.26 dB, SSIM: 0.999, NMI: 0.741), enabling near-perfect image reconstruction. In contrast, the proposed CLS-protected HE approach prevents such attacks while enabling encrypted inference directly on ciphertexts, requiring only 326 KB of encrypted data transmission per aggregation round. The framework achieves 96.12 percent global classification accuracy in the unencrypted domain and 90.02 percent in the encrypted domain.

---

## 118. A deep learning model to reduce agent dose for contrast-enhanced MRI of the cerebellopontine angle cistern

**论文链接:** [http://arxiv.org/abs/2511.20926v1](http://arxiv.org/abs/2511.20926v1)

**作者:** Yunjie Chen, Rianne A. Weber, Olaf M. Neve, Stephan R. Romeijn, Erik F. Hensen, Jelmer M. Wolterink, Qian Tao, Marius Staring, Berit M. Verbist

**发布时间:** 2025-11-25

### GPT解析

### 总结

该研究评估了一种深度学习模型，用于减少小脑脑桥角池对比增强T1加权MRI中的对比剂剂量，同时保持图像质量和诊断性能。

### 背景

传统对比增强MRI需要较高剂量的对比剂，可能带来潜在风险，因此需要开发技术来降低剂量同时保持图像质量。

### 目的

评估深度学习模型在减少小脑脑桥角池对比增强T1加权MRI对比剂剂量方面的效果。

### 方法

多中心回顾性研究，使用前庭神经瘤患者的T1和T1ce图像模拟低剂量T1ce，训练深度学习模型恢复标准剂量图像，评估图像质量和分割性能，并由放射科医生进行主观评估。

### 主要发现

随着输入剂量增加，图像质量指标改善；在10%输入剂量下，分割性能显著提高；10%和30%输入剂量的深度学习恢复图像质量优秀，后者信息量更大。

### 结论

深度学习模型可提高CPA池低剂量MRI的图像质量，使在标准剂量的10%-30%剂量下仍能进行病变检测和诊断特征描述。

### 翻译

目的：评估一种深度学习模型，用于减少小脑脑桥角池的对比增强T1加权MRI中的对比剂剂量。材料与方法：在这项多中心回顾性研究中，使用前庭神经瘤患者的T1和T1ce图像来模拟不同对比剂剂量减少的低剂量T1ce。训练深度学习模型从低剂量模拟中恢复标准剂量的T1ce。评估深度学习恢复的T1ce的图像质量和分割性能。要求头颈部放射科医生从多个方面评估深度学习恢复的图像，包括图像质量和诊断特征。结果：评估了72名VS患者（平均年龄58.51±14.73岁，39名男性）的203项MRI研究。随着输入剂量的增加，恢复的T1ce的结构相似性指数测量值从0.639±0.113增加到0.993±0.009，峰值信噪比从21.6±3.73 dB增加到41.4±4.84 dB。在10%输入剂量下，使用深度学习恢复的T1ce进行分割，Dice系数从0.673提高到0.734，95% Hausdorff距离从2.38 mm减少到2.07 mm，平均表面距离从1.00 mm减少到0.59 mm。来自10%和30%输入剂量的深度学习恢复的T1ce都显示出优秀的图像质量，后者被认为信息量更大。结论：深度学习模型提高了CPA池低剂量MRI的图像质量，这使在标准剂量的10%-30%剂量下可以进行病变检测和诊断特征描述。


### 论文摘要

Objectives: To evaluate a deep learning (DL) model for reducing the agent dose of contrast-enhanced T1-weighted MRI (T1ce) of the cerebellopontine angle (CPA) cistern. Materials and methods: In this multi-center retrospective study, T1 and T1ce of vestibular schwannoma (VS) patients were used to simulate low-dose T1ce with varying reductions of contrast agent dose. DL models were trained to restore standard-dose T1ce from the low-dose simulation. The image quality and segmentation performance of the DL-restored T1ce were evaluated. A head and neck radiologist was asked to rate DL-restored images in multiple aspects, including image quality and diagnostic characterization. Results: 203 MRI studies from 72 VS patients (mean age, 58.51 \pm 14.73, 39 men) were evaluated. As the input dose increased, the structural similarity index measure of the restored T1ce increased from 0.639 \pm 0.113 to 0.993 \pm 0.009, and the peak signal-to-noise ratio increased from 21.6 \pm 3.73 dB to 41.4 \pm 4.84 dB. At 10% input dose, using DL-restored T1ce for segmentation improved the Dice from 0.673 to 0.734, the 95% Hausdorff distance from 2.38 mm to 2.07 mm, and the average surface distance from 1.00 mm to 0.59 mm. Both DL-restored T1ce from 10% and 30% input doses showed excellent images, with the latter being considered more informative. Conclusion: The DL model improved the image quality of low-dose MRI of the CPA cistern, which makes lesion detection and diagnostic characterization possible with 10% - 30% of the standard dose.

---

## 119. Pre-train to Gain: Robust Learning Without Clean Labels

**论文链接:** [http://arxiv.org/abs/2511.20844v1](http://arxiv.org/abs/2511.20844v1)

**作者:** David Szczecina, Nicholas Pellegrino, Paul Fieguth

**发布时间:** 2025-11-25

**备注:** 5 pages, 3 figures

### GPT解析

### 总结

这篇论文提出了一种不需要干净标签子集的自监督预训练方法，用于训练对噪声标签鲁棒的深度网络，在各种噪声条件下都表现出色。

### 背景

使用噪声标签训练深度网络会导致泛化能力差和准确率下降，这是由于对标签噪声的过拟合造成的。现有的处理噪声标签的方法通常依赖于可用的一小部分干净数据。

### 目的

提出一种不需要干净标签子集的噪声鲁棒模型训练方法。

### 方法

使用自监督学习(SSL)预先训练特征提取器骨干网络，不使用标签，然后在 noisy 数据集上进行标准的监督训练。评估了 SimCLR 和 Barlow Twins 作为 SSL 方法，在 CIFAR-10 和 CIFAR-100 数据集上，使用合成和真实世界的噪声。

### 主要发现

在所有噪声率下，自监督预训练都能持续提高分类准确率，并增强下游标签错误检测能力(F1 和平衡准确率)。随着噪声率的增加，性能差距扩大，显示出更好的鲁棒性。在低噪声水平下，该方法与 ImageNet 预训练模型相当的性能，而在高噪声条件下显著优于它们。

### 结论

通过自监督预训练可以训练出更鲁棒的模型，不需要干净的标签子集。

### 翻译

使用噪声标签训练深度网络会导致泛化能力差和准确率下降，这是由于对标签噪声的过拟合造成的。现有的处理噪声标签的方法通常依赖于可用的一小部分干净数据。通过使用自监督学习(SSL)在不使用标签的情况下预训练特征提取器骨干网络，然后在 noisy 数据集上进行标准的监督训练，我们可以在不需要干净标签子集的情况下训练出更鲁棒的噪声模型。我们在 CIFAR-10 和 CIFAR-100 上评估了 SimCLR 和 Barlow Twins 作为 SSL 方法，使用合成和真实世界的噪声。在所有噪声率下，自监督预训练都能持续提高分类准确率，并增强下游标签错误检测能力(F1 和平衡准确率)。随着噪声率的增加，性能差距扩大，显示出更好的鲁棒性。值得注意的是，在低噪声水平下，我们的方法与 ImageNet 预训练模型相当的性能，而在高噪声条件下显著优于它们。


### 论文摘要

Training deep networks with noisy labels leads to poor generalization and degraded accuracy due to overfitting to label noise. Existing approaches for learning with noisy labels often rely on the availability of a clean subset of data. By pre-training a feature extractor backbone without labels using self-supervised learning (SSL), followed by standard supervised training on the noisy dataset, we can train a more noise robust model without requiring a subset with clean labels. We evaluate the use of SimCLR and Barlow~Twins as SSL methods on CIFAR-10 and CIFAR-100 under synthetic and real world noise. Across all noise rates, self-supervised pre-training consistently improves classification accuracy and enhances downstream label-error detection (F1 and Balanced Accuracy). The performance gap widens as the noise rate increases, demonstrating improved robustness. Notably, our approach achieves comparable results to ImageNet pre-trained models at low noise levels, while substantially outperforming them under high noise conditions.

---

## 120. MoRE: Batch-Robust Multi-Omics Representations from Frozen Pre-trained Transformers

**论文链接:** [http://arxiv.org/abs/2511.20382v2](http://arxiv.org/abs/2511.20382v2)

**作者:** Audrey Pei-Hsuan Chen

**发布时间:** 2025-11-25

### GPT解析

### 总结

本文提出了MoRE（Multi-Omics Representation Embedding）框架，通过重用冻结的预训练transformers将异质检测方法对齐到共享潜在空间，实现多组学数据的表示学习。

### 背景

多组学数据表示学习面临极端维度性、模态异质性和队列特异性批次效应的挑战，而预训练transformer在多组学整合中的应用探索不足。

### 目的

开发一种能够处理多组学数据异质性、减少批次效应并实现跨平台泛化的表示学习方法。

### 方法

MoRE采用参数高效微调策略，将轻量级模态特定适配器和任务自适应融合层附加到冻结的transformer主干上，优化掩码建模目标与监督对比和批次不变对齐损失的结合。

### 主要发现

MoRE在批次鲁棒性和生物保守性方面与scGPT、scVI和Harmony with Scrublet等基线方法具有竞争力，同时显著减少了可训练参数，在集成保真度、稀有群体检测和模态转移方面表现良好。

### 结论

MoRE为构建通用组学基础模型提供了实用步骤，能够在未见过的细胞类型和平台上实现结构保留的嵌入表示。

### 翻译

多组学数据的表示学习具有挑战性，因为其极端维度性、模态异质性和队列特异性批次效应。虽然预训练的transformer主干已在生物序列建模中展现出广泛的泛化能力，但在多组学整合中的应用仍探索不足。我们提出了MoRE（多组学表示嵌入），这是一个重用冻结预训练transformers的框架，将异质检测方法对齐到共享的潜在空间。与纯生成方法不同，MoRE采用参数高效微调策略，优先考虑跨样本和跨模态对齐，而非简单的序列重建。具体而言，MoRE将轻量级的模态特定适配器和任务自适应融合层附加到冻结主干上，它优化了掩码建模目标，同时结合监督对比和批次不变对齐损失，产生结构保留的嵌入，可泛化到未见过的细胞类型和平台。我们与scGPT、scVI和Harmony with Scrublet等既定基线对MoRE进行了基准测试，评估了集成保真度、稀有群体检测和模态转移。我们的结果表明，MoRE实现了有竞争力的批次鲁棒性和生物保守性，同时显著减少了与完全微调模型相比的可训练参数。这项工作将MoRE定位为迈向通用组学基础模型的实用步骤。


### 论文摘要

Representation learning on multi-omics data is challenging due to extreme dimensionality, modality heterogeneity, and cohort-specific batch effects. While pre-trained transformer backbones have shown broad generalization capabilities in biological sequence modeling, their application to multi-omics integration remains underexplored. We present MoRE (Multi-Omics Representation Embedding), a framework that repurposes frozen pre-trained transformers to align heterogeneous assays into a shared latent space. Unlike purely generative approaches, MoRE employs a parameter-efficient fine-tuning (PEFT) strategy, prioritizing cross-sample and cross-modality alignment over simple sequence reconstruction. Specifically, MoRE attaches lightweight, modality-specific adapters and a task-adaptive fusion layer to the frozen backbone. It optimizes a masked modeling objective jointly with supervised contrastive and batch-invariant alignment losses, yielding structure-preserving embeddings that generalize across unseen cell types and platforms. We benchmark MoRE against established baselines, including scGPT, scVI, and Harmony with Scrublet, evaluating integration fidelity, rare population detection, and modality transfer. Our results demonstrate that MoRE achieves competitive batch robustness and biological conservation while significantly reducing trainable parameters compared to fully fine-tuned models. This work positions MoRE as a practical step toward general-purpose omics foundation models.

---

## 121. Odin: Oriented Dual-module Integration for Text-rich Network Representation Learning

**论文链接:** [http://arxiv.org/abs/2511.21416v1](http://arxiv.org/abs/2511.21416v1)

**作者:** Kaifeng Hong, Yinglong Zhang, Xiaoying Hong, Xuewen Xia, Xing Xu

**发布时间:** 2025-11-26

**备注:** 32 pages, 2 figures

### GPT解析

### 总结

Odin是一种新型架构，通过定向双模块机制将图结构注入到Transformer中，解决了现有方法在处理文本属性图时的局限性，既避免了GNNs的过平滑问题，又考虑了图拓扑结构。

### 背景

文本属性图需要模型有效地结合强大的文本理解和结构化推理。现有方法要么依赖图神经网络（GNNs），受限于过平滑和跳跃依赖扩散；要么使用Transformer，忽略了图拓扑结构，将节点视为独立序列。

### 目的

提出一种名为Odin（Oriented Dual-module INtegration）的新架构，通过定向双模块机制在选定深度将图结构注入到Transformer中。

### 方法

Odin不依赖多跳扩散，而是在特定的Transformer层中集成多跳结构。聚合操作在全局[CLS]表示上进行，从根本上避免了过平滑，并将结构抽象与邻域大小或图拓扑解耦。还提出了Light Odin，这是一个轻量级变体，保留了相同的层对齐结构抽象，以实现更快的训练和推理。

### 主要发现

Odin能够产生低、中、高水平的结构抽象，与模型的语义层次结构对齐。Odin的表达能力严格包含了纯Transformer和GNNs的表达能力。在多个文本丰富的图基准测试上，Odin实现了最先进的准确性，而Light Odin在显著降低计算成本的同时提供了有竞争力的性能。

### 结论

Odin和Light Odin共同形成了一个统一的、无跳跃的框架，用于结构-文本的集成。

### 翻译

文本属性图需要模型有效地结合强大的文本理解和结构化推理。现有方法要么依赖图神经网络，受限于过平滑和跳跃依赖扩散，要么采用Transformer，忽略了图拓扑结构，将节点视为独立序列。我们提出了Odin（定向双模块集成），一种新架构，通过定向双模块机制在选定深度将图结构注入到Transformer中。与消息传递的GNN不同，Odin不依赖多跳扩散；相反，多跳结构在特定的Transformer层中被集成，产生与模型语义层次对齐的低、中、高水平的结构抽象。因为聚合操作在全局[CLS]表示上进行，Odin从根本上避免了过平滑，并将结构抽象与邻域大小或图拓扑解耦。我们进一步证明了Odin的表达能力严格包含了纯Transformer和GNNs的表达能力。为了在大规模或低资源设置中使设计更高效，我们引入了Light Odin，这是一个轻量级变体，保留了相同的层对齐结构抽象，以实现更快的训练和推理。在多个文本丰富的图基准上的实验表明，Odin实现了最先进的准确性，而Light Odin在显著降低计算成本的同时提供了有竞争力的性能。Odin和Light Odin共同形成了一个统一的、无跳跃的框架，用于结构-文本的原则性集成。该模型的源代码已在https://github.com/hongkaifeng/Odin发布。


### 论文摘要

Text-attributed graphs require models to effectively combine strong textual understanding with structurally informed reasoning. Existing approaches either rely on GNNs--limited by over-smoothing and hop-dependent diffusion--or employ Transformers that overlook graph topology and treat nodes as isolated sequences. We propose Odin (Oriented Dual-module INtegration), a new architecture that injects graph structure into Transformers at selected depths through an oriented dual-module mechanism.Unlike message-passing GNNs, Odin does not rely on multi-hop diffusion; instead, multi-hop structures are integrated at specific Transformer layers, yielding low-, mid-, and high-level structural abstraction aligned with the model's semantic hierarchy. Because aggregation operates on the global [CLS] representation, Odin fundamentally avoids over-smoothing and decouples structural abstraction from neighborhood size or graph topology. We further establish that Odin's expressive power strictly contains that of both pure Transformers and GNNs.To make the design efficient in large-scale or low-resource settings, we introduce Light Odin, a lightweight variant that preserves the same layer-aligned structural abstraction for faster training and inference. Experiments on multiple text-rich graph benchmarks show that Odin achieves state-of-the-art accuracy, while Light Odin delivers competitive performance with significantly reduced computational cost. Together, Odin and Light Odin form a unified, hop-free framework for principled structure-text integration. The source code of this model has been released at https://github.com/hongkaifeng/Odin.

---

## 122. A Research and Development Portfolio of GNN Centric Malware Detection, Explainability, and Dataset Curation

**论文链接:** [http://arxiv.org/abs/2511.20801v1](http://arxiv.org/abs/2511.20801v1)

**作者:** Hossein Shokouhinejad, Griffin Higgins, Roozbeh Razavi-Far, Ali A. Ghorbani

**发布时间:** 2025-11-25

**DOI:** 10.1109/ICDMW69685.2025.00126

**备注:** Accepted in 2025 IEEE International Conference on Data Mining Workshops (ICDMW)

### GPT解析

### 总结

该论文汇集了六项相关研究，共同解决图神经网络在恶意软件检测中面临的可扩展性、可解释性和可靠数据可用性等挑战。

### 背景

图神经网络(GNNs)已成为通过图结构表示捕获程序执行来检测恶意软件的有效工具，但仍面临重要挑战。

### 目的

解决图神经网络在恶意软件检测中的可扩展性、可解释性和可靠数据可用性问题。

### 方法

基于图的恶意检测和可解释性调查、新的图约简方法、集成约简-学习方法、解释一致性调查、基于子图匹配的双重解释技术、注意力引导的堆叠GNNs集成模型，以及发布控制流图精选数据集。

### 主要发现

这些贡献形成了一致的研究路线，增强了基于GNN的恶意软件检测能力。

### 结论

通过提高效率、增加透明度和提供坚实的实验基础，这些研究共同加强了基于GNN的恶意软件检测。

### 翻译

图神经网络(GNNs)已成为通过图结构表示捕获程序执行来检测恶意软件的有效工具。然而，关于可扩展性、可解释性和可靠数据可用性等重要挑战仍然存在。本文汇集了六项相关研究，共同解决这些问题。该组合首先对基于图的恶意软件检测和可解释性进行了调查，然后推进到新的图约简方法、集成约简-学习方法以及对解释一致性的研究。它还引入了基于子图匹配的双重解释技术，并开发了基于注意力的堆叠GNN集成模型以提高可解释性。同时，发布了控制流图的精选数据集以支持可重复性和未来研究。这些贡献共同形成了一致的研究路线，通过提高效率、增加透明度和提供坚实的实验基础，加强了基于GNN的恶意软件检测。


### 论文摘要

Graph Neural Networks (GNNs) have become an effective tool for malware detection by capturing program execution through graph-structured representations. However, important challenges remain regarding scalability, interpretability, and the availability of reliable datasets. This paper brings together six related studies that collectively address these issues. The portfolio begins with a survey of graph-based malware detection and explainability, then advances to new graph reduction methods, integrated reduction-learning approaches, and investigations into the consistency of explanations. It also introduces dual explanation techniques based on subgraph matching and develops ensemble-based models with attention-guided stacked GNNs to improve interpretability. In parallel, curated datasets of control flow graphs are released to support reproducibility and enable future research. Together, these contributions form a coherent line of research that strengthens GNN-based malware detection by enhancing efficiency, increasing transparency, and providing solid experimental foundations.

---

## 123. Pretraining Transformer-Based Models on Diffusion-Generated Synthetic Graphs for Alzheimer's Disease Prediction

**论文链接:** [http://arxiv.org/abs/2511.20704v1](http://arxiv.org/abs/2511.20704v1)

**作者:** Abolfazl Moslemi, Hossein Peyvandi

**发布时间:** 2025-11-24

**备注:** 14 pages. Preprint

### GPT解析

### 总结

本研究提出了一种基于Transformer的阿尔茨海默病诊断框架，结合扩散合成数据生成、图表示学习和迁移学习，有效解决了标记数据有限、多站点异质性和类别不平衡等挑战，在NACC数据集上表现出优异的诊断性能。

### 背景

早期准确检测阿尔茨海默病对及时干预和改善预后至关重要。然而，开发可靠的机器学习诊断模型面临三大挑战：标记数据有限、多站点数据异质性以及类别不平衡问题。

### 目的

提出一种结合基于扩散的合成数据生成、图表示学习和迁移学习的Transformer诊断框架，以提高阿尔茨海默病诊断的准确性和鲁棒性。

### 方法

使用条件类去噪扩散概率模型在真实NACC数据集上训练，生成模仿多模态临床和神经影像特征分布的合成队列；通过模态特定的图Transformer编码器在此合成数据上预学习鲁棒表示，然后冻结编码器，在原始NACC数据上训练分类器；使用多种指标评估真实与合成数据的分布对齐，并结合校准和敏感性分析进行评估。

### 主要发现

该框架在NACC数据集的主题级交叉验证中，显著优于包括早期/晚期融合深度神经网络和多模态图模型MaGNet在内的标准基线，实现了更高的AUC、准确率、敏感性和特异性。

### 结论

基于扩散的合成数据预训练与图Transformer相结合，可有效提高低样本、不平衡临床预测场景下的模型泛化能力，为阿尔茨海默病的早期诊断提供了新思路。

### 翻译

早期和准确检测阿尔茨海默病对于实现及时干预和改善结果至关重要。然而，由于标记数据有限、多站点异质性和类别不平衡，开发可靠的机器学习模型用于阿尔茨海默病诊断具有挑战性。我们提出了一种基于Transformer的诊断框架，结合了基于扩散的合成数据生成、图表示学习和迁移学习。在真实的NACC数据集上训练一个条件类去噪扩散概率模型，生成一个大型合成队列，该队列镜像多模态临床和神经影像特征分布，同时平衡诊断类别。模态特定的图Transformer编码器首先在此合成数据上预训练，学习鲁棒、类别判别性表示，然后在原始NACC数据的嵌入上训练神经分类器时冻结这些编码器。我们使用最大均值差异、Fréchet距离和能量距离等指标量化真实和合成队列之间的分布对齐，并通过校准和固定特异性敏感性分析补充判别指标。经验上，我们的框架优于标准基线，包括早期和晚期融合深度神经网络以及多模态基于图的模型MaGNet，在NACC上的主题级交叉验证中产生更高的AUC、准确率、敏感性和特异性。这些结果表明，基于扩散的合成预训练与图Transformer相结合可以提高低样本、不平衡临床预测设置中的泛化能力。


### 论文摘要

Early and accurate detection of Alzheimer's disease (AD) is crucial for enabling timely intervention and improving outcomes. However, developing reliable machine learning (ML) models for AD diagnosis is challenging due to limited labeled data, multi-site heterogeneity, and class imbalance. We propose a Transformer-based diagnostic framework that combines diffusion-based synthetic data generation with graph representation learning and transfer learning. A class-conditional denoising diffusion probabilistic model (DDPM) is trained on the real-world NACC dataset to generate a large synthetic cohort that mirrors multimodal clinical and neuroimaging feature distributions while balancing diagnostic classes. Modality-specific Graph Transformer encoders are first pretrained on this synthetic data to learn robust, class-discriminative representations and are then frozen while a neural classifier is trained on embeddings from the original NACC data. We quantify distributional alignment between real and synthetic cohorts using metrics such as Maximum Mean Discrepancy (MMD), Frechet distance, and energy distance, and complement discrimination metrics with calibration and fixed-specificity sensitivity analyses. Empirically, our framework outperforms standard baselines, including early and late fusion deep neural networks and the multimodal graph-based model MaGNet, yielding higher AUC, accuracy, sensitivity, and specificity under subject-wise cross-validation on NACC. These results show that diffusion-based synthetic pretraining with Graph Transformers can improve generalization in low-sample, imbalanced clinical prediction settings.

---

## 124. Canvas-to-Image: Compositional Image Generation with Multimodal Controls

**论文链接:** [http://arxiv.org/abs/2511.21691v1](http://arxiv.org/abs/2511.21691v1)

**作者:** Yusuf Dalva, Guocheng Gordon Qian, Maya Goldenberg, Tsai-Shien Chen, Kfir Aberman, Sergey Tulyakov, Pinar Yanardag, Kuan-Chieh Jackson Wang

**发布时间:** 2025-11-26

**备注:** 24 pages; webpage: https://snap-research.github.io/canvas-to-image/

### GPT解析

### 总结

Canvas-to-Image是一个统一框架，整合了文本提示、主题参考、空间排列、姿势约束和布局注释等异构控制信号，通过将它们编码为单一画布图像，使扩散模型能够进行集成的视觉-空间推理，从而生成更忠实于用户意图的图像。

### 背景

现代扩散模型在生成高质量和多样化的图像方面表现出色，但在高保真度的组合和多模态控制方面仍然存在问题，特别是当用户需要同时指定多种控制信号时。

### 目的

引入Canvas-to-Image框架，将多种异构控制整合到单一画布界面中，解决扩散模型在处理多模态控制时的局限性，使用户能够生成更符合其意图的图像。

### 方法

将多样化控制信号编码成单个复合画布图像，模型可直接解释这些图像进行视觉-空间推理；整理多任务数据集，提出多任务画布训练策略，优化扩散模型在统一学习范式中联合理解和整合异构控制。

### 主要发现

联合训练使模型能够跨多个控制模态进行推理而非依赖特定任务启发式方法；在多人物组合、姿势控制组合、布局约束生成和多控制生成等挑战性基准测试中，Canvas-to-Image在身份保持和控制遵循方面显著优于现有方法。

### 结论

Canvas-to-Image有效解决了扩散模型在处理多种控制信号时的局限性，提供了一个统一接口整合不同类型控制，生成更符合用户意图的高质量图像。

### 翻译

虽然现代扩散模型在生成高质量和多样化的图像方面表现出色，但它们仍然在高保真度的组合和多模态控制方面存在困难，特别是当用户同时指定文本提示、主题参考、空间排列、姿势约束和布局注释时。我们引入了Canvas-to-Image，这是一个统一框架，将这些异构控制整合到单一画布界面中，使用户能够生成忠实反映其意图的图像。我们的关键思想是将多样化的控制信号编码成单个复合画布图像，模型可以直接解释这些图像进行集成的视觉-空间推理。我们进一步整理了一套多任务数据集，并提出了多任务画布训练策略，优化扩散模型在统一学习范式中联合理解和整合异构控制到文本到图像生成中。这种联合训练使Canvas-to-Image能够跨多个控制模态进行推理，而不是依赖于特定任务的启发式方法，并且在推理过程中能够很好地泛化到多控制场景。大量实验表明，Canvas-to-Image在具有挑战性的基准测试中显著优于最先进的方法，包括身份保持和控制遵循，特别是在多人物组合、姿势控制的组合、布局约束的生成和多控制生成方面。


### 论文摘要

While modern diffusion models excel at generating high-quality and diverse images, they still struggle with high-fidelity compositional and multimodal control, particularly when users simultaneously specify text prompts, subject references, spatial arrangements, pose constraints, and layout annotations. We introduce Canvas-to-Image, a unified framework that consolidates these heterogeneous controls into a single canvas interface, enabling users to generate images that faithfully reflect their intent. Our key idea is to encode diverse control signals into a single composite canvas image that the model can directly interpret for integrated visual-spatial reasoning. We further curate a suite of multi-task datasets and propose a Multi-Task Canvas Training strategy that optimizes the diffusion model to jointly understand and integrate heterogeneous controls into text-to-image generation within a unified learning paradigm. This joint training enables Canvas-to-Image to reason across multiple control modalities rather than relying on task-specific heuristics, and it generalizes well to multi-control scenarios during inference. Extensive experiments show that Canvas-to-Image significantly outperforms state-of-the-art methods in identity preservation and control adherence across challenging benchmarks, including multi-person composition, pose-controlled composition, layout-constrained generation, and multi-control generation.

---

## 125. Efficient bayesian spatially varying coefficients modeling for censored data using the vecchia approximation

**论文链接:** [http://arxiv.org/abs/2511.21553v1](http://arxiv.org/abs/2511.21553v1)

**作者:** Yacine Mohamed Idir, Thomas Romary

**发布时间:** 2025-11-26

### GPT解析

### 总结

本研究提出了一种基于Vecchia近似的高效贝叶斯高斯过程建模框架，解决了传统贝叶斯高斯过程模型在大数据集或多协变量情况下计算成本过高的问题，并在法国图卢兹的土壤污染数据集上验证了其有效性。

### 背景

空间变化系数(SVC)模型允许边际效应在空间上呈现非平稳性，比标准地统计模型更灵活且易于解释。常用建模方法包括地理加权回归(GWR)和贝叶斯高斯过程(Bayes-GP)，但Bayes-GP模型在处理大数据或多协变量时计算成本过高，因为需要重复求逆密集协方差矩阵。

### 目的

提出一种高效的贝叶斯高斯过程建模框架，利用Vecchia近似降低计算复杂度，同时保持模型准确性，以解决大型空间数据集的计算挑战。

### 方法

开发基于Vecchia近似的贝叶斯高斯过程建模框架，并将其应用于法国图卢兹的土壤污染数据集，该数据集具有高度删失(三分之二观测值被删失)和空间聚类特点。

### 主要发现

基于Vecchia的贝叶斯高斯过程模型能够有效捕捉空间变化效应，即使在删失数据的约束下，也能提供关于空间异质性的有意义的见解。

### 结论

基于Vecchia近似的贝叶斯高斯过程模型为处理大型空间数据集提供了一种有效方法，能在保持模型准确性的同时显著降低计算成本，特别适用于具有高度删失和空间聚类特性的数据。

### 翻译

空间变化系数(SVC)模型允许边际效应在空间上呈现非平稳性，因此与具有外部漂移的标准地统计模型相比，具有更高的灵活性。同时，SVC模型具有易于解释的优势。它们为理解因变量和自变量之间的关系如何随空间变化提供了灵活的框架。建模此类数据的最常见方法是地理加权回归(GWR)和贝叶斯高斯过程(Bayes-GP)。假设系数遵循高斯过程的贝叶斯SVC模型为处理空间非平稳性提供了严谨的方法。然而，当处理大型数据集和/或使用大量协变量时，Bayes-GP模型的计算成本可能高得令人望而却步，因为在每个马尔可夫链蒙特卡洛(MCMC)迭代中需要重复求逆密集协方差矩阵。在本研究中，我们提出了一种高效的基于贝叶斯高斯过程的建模框架，利用Vecchia近似来降低计算复杂度，同时保持准确性。所提出的方法应用于法国图卢兹的一个具有挑战性的土壤污染数据集，该数据集具有高度删失（三分之二的观测值被删失）和空间聚类的特点。我们的结果表明，基于Vecchia的贝叶斯高斯过程模型能够捕捉空间变化效应，即使在删失数据的约束下，也能提供关于空间异质性的有意义的见解。


### 论文摘要

Spatially varying coefficients (SVC) models allow for marginal effects to be non-stationary over space and thus offer a higher degree of flexibility with respect to standard geostatistical models with external drift. At the same time, SVC models have the advantage that they are easily interpretable. They offer a flexible framework for understanding how the relationships between dependent and independent variables vary across space. The most common methods for modelling such data are the Geographically Weighted Regression (GWR) and Bayesian Gaussian Process (Bayes-GP). The Bayesian SVC model, which assumes that the coefficients follow Gaussian processes, provides a rigorous approach to account for spatial non-stationarity. However, the computational cost of Bayes-GP models can be prohibitively high when dealing with large datasets or/and when using a large number of covariates, due to the repeated inversion of dense covariance matrices required at each Markov chain Monte Carlo (MCMC) iteration. In this study, we propose an efficient Bayes-GP modeling framework leveraging the Vecchia approximation to reduce computational complexity while maintaining accuracy. The proposed method is applied to a challenging soil pollution data set in Toulouse, France, characterized by a high degree of censorship (two-thirds censored observations) and spatial clustering. Our results demonstrate the ability of the Vecchia-based Bayes-GP model to capture spatially varying effects and provide meaningful insights into spatial heterogeneity, even under the constraints of censored data.

---

## 126. Modeling and Optimizing Performance Bottlenecks for Neuromorphic Accelerators

**论文链接:** [http://arxiv.org/abs/2511.21549v1](http://arxiv.org/abs/2511.21549v1)

**作者:** Jason Yik, Walter Gallego Gomez, Andrew Cheng, Benedetto Leto, Alessandro Pierro, Noah Pacik-Nelson, Korneel Van den Berghe, Vittorio Fra, Andreea Danielescu, Gianvito Urgese, Vijay Janapa Reddi

**发布时间:** 2025-11-26

### GPT解析

### 总结

该论文对神经形态加速器进行了首次全面的性能边界和瓶颈分析，揭示了传统度量的不足，并提出了一种结合稀疏感知训练与floorline引导分区的优化方法，在等精度条件下实现了显著的性能提升。

### 背景

神经形态加速器利用事件驱动、空间扩展的架构，通过内存和计算的协同定位自然利用非结构化稀疏性，为机器学习推理提供了有前途的平台。然而，其独特的架构特性与常规加速器有根本不同的性能动态，现有基于网络级稀疏性和操作计数的工作负载优化方法的有效性尚不明确。

### 目的

对神经形态加速器进行首次全面的性能边界和瓶颈分析，揭示传统度量的不足，了解对工作负载性能重要的方面，并提出有效的优化方法。

### 方法

采用理论分析建模和对三种真实神经形态加速器(Brainchip AKD1000、Synsense Speck和Intel Loihi 2)进行广泛的实证表征，建立三种不同的加速器瓶颈状态(内存限制型、计算限制型和流量限制型)，并合成见解到floorline性能模型中，最后提出一种将稀疏感知训练与floorline引导分区相结合的优化方法。

### 主要发现

存在三种不同的加速器瓶颈状态：内存限制型、计算限制型和流量限制型；确定了可能表现出这些瓶颈状态的工作负载配置特征；floorline性能模型可以识别性能边界并指导如何优化给定工作负载。

### 结论

所提出的优化方法在等精度条件下实现了显著的性能改进，与之前手动调整的配置相比，最高可达3.86倍的运行时间改进和3.38倍的能耗降低。

### 翻译

神经形态加速器通过利用事件驱动、空间扩展的架构，通过内存和计算的协同定位自然利用非结构化稀疏性，为机器学习推理提供了有前途的平台。然而，其独特的架构特性与常规加速器有根本不同的性能动态。现有针对神经形态加速器的工作负载优化方法依赖于网络级稀疏性和操作计数，但这些指标实际上是否能提高部署性能尚不清楚。本文首次对神经形态加速器进行了全面的性能边界和瓶颈分析，揭示了传统度量的不足，并阐明了哪些方面对工作负载性能很重要。我们提出了理论分析建模和对三种真实神经形态加速器(Brainchip AKD1000、Synsense Speck和Intel Loihi 2)的广泛实证表征。从中，我们确立了三种不同的加速器瓶颈状态：内存限制型、计算限制型和流量限制型，并确定了哪些工作负载配置特征可能表现出这些瓶颈状态。我们将所有见解综合到floorline性能模型中，这是一种视觉模型，可以识别性能边界，并根据工作负载在模型中的位置指导如何优化给定工作负载。最后，我们提出了一种将稀疏感知训练与floorline引导分区相结合的优化方法。我们的方法在等精度条件下实现了显著的性能改进：与之前手动调整的配置相比，最高可达3.86倍的运行时间改进和3.38倍的能耗降低。


### 论文摘要

Neuromorphic accelerators offer promising platforms for machine learning (ML) inference by leveraging event-driven, spatially-expanded architectures that naturally exploit unstructured sparsity through co-located memory and compute. However, their unique architectural characteristics create performance dynamics that differ fundamentally from conventional accelerators. Existing workload optimization approaches for neuromorphic accelerators rely on aggregate network-wide sparsity and operation counting, but the extent to which these metrics actually improve deployed performance remains unknown. This paper presents the first comprehensive performance bound and bottleneck analysis of neuromorphic accelerators, revealing the shortcomings of the conventional metrics and offering an understanding of what facets matter for workload performance. We present both theoretical analytical modeling and extensive empirical characterization of three real neuromorphic accelerators: Brainchip AKD1000, Synsense Speck, and Intel Loihi 2. From these, we establish three distinct accelerator bottleneck states, memory-bound, compute-bound, and traffic-bound, and identify which workload configuration features are likely to exhibit these bottleneck states. We synthesize all of our insights into the floorline performance model, a visual model that identifies performance bounds and informs how to optimize a given workload, based on its position on the model. Finally, we present an optimization methodology that combines sparsity-aware training with floorline-informed partitioning. Our methodology achieves substantial performance improvements at iso-accuracy: up to 3.86x runtime improvement and 3.38x energy reduction compared to prior manually-tuned configurations.

---

## 127. Context-Specific Causal Graph Discovery with Unobserved Contexts: Non-Stationarity, Regimes and Spatio-Temporal Patterns

**论文链接:** [http://arxiv.org/abs/2511.21537v1](http://arxiv.org/abs/2511.21537v1)

**作者:** Martin Rabel, Jakob Runge

**发布时间:** 2025-11-26

### GPT解析

### 总结

该研究提出了一种框架，用于分析空间网格时间序列数据中的因果图变化，解决了传统因果发现方法在处理非平稳数据时面临的挑战。

### 背景

现实世界数据（如气候应用数据）通常包含空间网格时间序列数据，虽然底层系统在不同时空点可能表现相似，但存在的变异本身包含重要信息，且可能影响假设平稳性算法的稳定性和可靠性。

### 目的

研究因果图中变化所编码的信息，同时考虑算法稳定性，开发能够处理非平稳数据的因果发现方法。

### 方法

分析因果图变化分析的核心挑战，开发指导原则，并提供一个修改基于约束的因果发现方法的框架，特别是在独立性测试层面进行修改。

### 主要发现

开发了一个极其模块化、易于扩展和广泛适用的框架，能够利用现有的基于约束的因果发现方法（如PC、PC-stable、FCI、PCMCI、PCMCI+、LPCMCI），只需很少或无需修改。

### 结论

该框架通过模块化设计允许系统地理解和改进一系列子问题，可利用变化点检测、聚类等相关领域的方法进行扩展，简化了对基本限制、超参数和结果统计解释的理解。开源实现即将推出。

### 翻译

现实世界数据，例如气候应用中的数据，通常由空间网格时间序列数据或具有类似结构的数据组成。虽然底层系统通常被认为在不同时空点表现相似，但存在的变异具有双重相关性：它们本身通常包含重要信息，并且可能对假设平稳性或空间平移不变性的算法的稳定性和结果的可靠性产生负面影响。我们研究了因果图中变化所编码的信息，同时考虑稳定性。对该任务的分析确定了两个核心挑战。我们开发了克服这些挑战的指导原则，并提供了一个通过修改基于约束的因果发现方法（在独立性测试层面）来实现这些原则的框架。这导致了一个极其模块化、易于扩展和广泛适用的框架。它可以利用现有的基于约束的因果发现方法（在IID算法PC、PC-stable、FCI和时间序列算法PCMCI、PCMCI+、LPCMCI上进行了演示），只需很少或无需修改。内置的模块化允许系统地理解和改进一系列子问题。通过设计，它可以利用来自变化点检测、聚类、独立性测试和其他相关问题的见解进行扩展。划分为更易处理的子问题也简化了对基本限制、控制权衡的超参数和结果的统计解释的理解。开源实现即将推出。


### 论文摘要

Real-world data, for example in climate applications, often consists of spatially gridded time series data or data with comparable structure. While the underlying system is often believed to behave similar at different points in space and time, those variations that do exist are twofold relevant: They often encode important information in and of themselves. And they may negatively affect the stability / convergence and reliability\Slash{}validity of results of algorithms assuming stationarity or space-translation invariance. We study the information encoded in changes of the causal graph, with stability in mind. An analysis of this general task identifies two core challenges. We develop guiding principles to overcome these challenges, and provide a framework realizing these principles by modifying constraint-based causal discovery approaches on the level of independence testing. This leads to an extremely modular, easily extensible and widely applicable framework. It can leverage existing constraint-based causal discovery methods (demonstrated on IID-algorithms PC, PC-stable, FCI and time series algorithms PCMCI, PCMCI+, LPCMCI) with little to no modification. The built-in modularity allows to systematically understand and improve upon an entire array of subproblems. By design, it can be extended by leveraging insights from change-point-detection, clustering, independence-testing and other well-studied related problems. The division into more accessible sub-problems also simplifies the understanding of fundamental limitations, hyperparameters controlling trade-offs and the statistical interpretation of results. An open-source implementation will be available soon.

---

## 128. DeepRFTv2: Kernel-level Learning for Image Deblurring

**论文链接:** [http://arxiv.org/abs/2511.21132v1](http://arxiv.org/abs/2511.21132v1)

**作者:** Xintian Mao, Haofei Song, Yin-Nian Liu, Qingli Li, Yan Wang

**发布时间:** 2025-11-26

### GPT解析

### 总结

本文提出了一种名为Fourier Kernel Estimator (FKE)的新方法，通过在核级别学习模糊过程来提高图像去模糊性能，当前深度网络仍在像素级学习阶段，无法真正理解模糊的本质。

### 背景

模糊是由清晰图像与模糊核的卷积自然引起的，而当前深度网络仍处于像素级学习阶段，要么进行端到端的像素级恢复，要么进行分阶段的伪核级恢复，无法使去模糊模型理解模糊的本质。

### 目的

提出一种方法让网络在核级别学习模糊过程，提高图像去模糊性能，降低复杂度，无需额外监督。

### 方法

提出Fourier Kernel Estimator (FKE)，考虑傅里叶空间中的激活操作，将空间域的卷积问题转换为傅里叶空间的乘法问题；与去模糊模型联合优化；将卷积对象从'图像'改为网络提取的'特征'；设计了具有可逆策略的解耦多尺度架构，包含多个层次子网络，实现更好的多尺度编码和解码。

### 主要发现

该方法在运动去模糊方面取得了最先进的结果；在处理其他与核相关的问题方面显示出潜力；核估计器能够学习物理上有意义的核。

### 结论

通过核级别学习模糊过程，显著提高了图像去模糊性能，该方法在复杂度低且无需额外监督的情况下有效。

### 翻译

众所周知，如果一个网络旨在学习如何去模糊，它应该理解模糊过程。模糊是由清晰图像与模糊核的卷积自然引起的。因此，允许网络在核级别学习模糊过程可以显著提高图像去模糊性能。然而，当前的深度网络仍处于像素级学习阶段，要么执行端到端的像素级恢复，要么执行分阶段的伪核级恢复，无法使去模糊模型理解模糊的本质。为此，我们提出了Fourier Kernel Estimator (FKE)，它考虑了傅里叶空间中的激活操作，并将空间域的卷积问题转换为傅里叶空间的乘法问题。我们的FKE与去模糊模型联合优化，使网络能够以低复杂度学习核级别的模糊过程，且无需任何额外监督。此外，我们将核的卷积对象从'图像'改为网络提取的'特征'，其丰富的语义和结构信息更适合模糊过程的学习。通过与估计的核进行特征卷积，我们的模型能够在核级别学习模糊的本质。为了进一步提高特征提取效率，我们设计了一种具有可逆策略的解耦多尺度架构，包含多个层次子网络，允许在低训练内存下实现更好的多尺度编码和解码。大量实验表明，我们的方法在运动去模糊方面取得了最先进的结果，并显示出处理其他与核相关问题的潜力。分析还表明，我们的核估计器能够学习物理上有意义的核。代码将在https://github.com/DeepMed-Lab-ECNU/Single-Image-Deblur上提供。


### 论文摘要

It is well-known that if a network aims to learn how to deblur, it should understand the blur process. Blurring is naturally caused by the convolution of the sharp image with the blur kernel. Thus, allowing the network to learn the blur process in the kernel-level can significantly improve the image deblurring performance. But, current deep networks are still at the pixel-level learning stage, either performing end-to-end pixel-level restoration or stage-wise pseudo kernel-level restoration, failing to enable the deblur model to understand the essence of the blur. To this end, we propose Fourier Kernel Estimator (FKE), which considers the activation operation in Fourier space and converts the convolution problem in the spatial domain to a multiplication problem in Fourier space. Our FKE, jointly optimized with the deblur model, enables the network to learn the kernel-level blur process with low complexity and without any additional supervision. Furthermore, we change the convolution object of the kernel from ``image" to network extracted ``feature", whose rich semantic and structural information is more suitable to blur process learning. With the convolution of the feature and the estimated kernel, our model can learn the essence of blur in kernel-level. To further improve the efficiency of feature extraction, we design a decoupled multi-scale architecture with multiple hierarchical sub-unets with a reversible strategy, which allows better multi-scale encoding and decoding in low training memory. Extensive experiments indicate that our method achieves state-of-the-art motion deblurring results and show potential for handling other kernel-related problems. Analysis also shows our kernel estimator is able to learn physically meaningful kernels. The code will be available at https://github.com/DeepMed-Lab-ECNU/Single-Image-Deblur.

---

## 129. Local Geometric and Transport Properties of Networks that are Generated from Hyperuniform Point Patterns

**论文链接:** [http://arxiv.org/abs/2511.21082v1](http://arxiv.org/abs/2511.21082v1)

**作者:** James V. Raj, Xiaohan Sun, Charles Emmett Maher, Katherine A. Newhall, Mason A. Porter

**发布时间:** 2025-11-26

**备注:** 13 pages; 8 figures, lots of ORCs; abstract on arXiv page shortened slightly due to maximum length requirement

### GPT解析

### 总结

本研究利用超均匀点模式生成了一类无序空间嵌入网络(HuPPI网络)，并与泊松点模式诱导网络(PoPPI网络)比较，发现超均匀性在传输效率和鲁棒性方面赋予优势，如更小的总有效电阻、更快的随机游走混合时间和更少的极端曲率边。

### 背景

超均匀性是一种长程有序性，其特点是与标准无序系统相比，长程密度波动被抑制，已成为理解各种自然和工程现象的有力概念。

### 目的

利用超均匀点模式生成一类无序的、空间嵌入的网络，这些网络既不同于完全有序的晶格，也不同于均匀随机几何图。

### 方法

创建超均匀点模式诱导网络(HuPPI网络)并与泊松点模式诱导网络(PoPPI网络)比较，通过计算局部几何和传输特性进行分析。

### 主要发现

HuPPI网络比PoPPI网络具有更小的总有效电阻、稍快的随机游走混合时间和更少的极端曲率边；同时具有更负的平均Ollivier-Ricci曲率和更小的全局电阻；网络生成方法对这些特性有强烈影响，常常掩盖底层点模式产生的差异。

### 结论

这些结果总体上展示了超均匀性在网络设计中的潜在优势，鼓励进一步对HuPPI网络进行理论和实验探索。

### 翻译

超均匀性是一种长程有序性，其特点是与标准无序系统相比，长程密度波动被抑制，已成为理解各种自然和工程现象的有力概念。在本文中，我们利用超均匀点模式生成一类无序的、空间嵌入的网络，这些网络既不同于完全有序的晶格，也不同于均匀随机几何图。我们称这些网络为超均匀点模式诱导网络，并将它们与对应的泊松点模式诱导网络进行比较。通过计算HuPPI网络的局部几何和传输特性，我们展示了超均匀性如何在传输效率和鲁棒性方面赋予优势。具体来说，我们表明HuPPI网络比PoPPI网络具有更小的总有效电阻、稍快的随机游走混合时间和更少的极端曲率边。反直觉的是，我们还发现HuPPI网络同时具有更负的平均Ollivier-Ricci曲率和更小的全局电阻，这表明具有适度负曲率的边不一定会对传输造成严重瓶颈。我们还证明了网络生成方法对这些特性有强烈影响，特别是它常常掩盖了底层点模式产生的差异。这些结果总体上展示了超均匀性在网络设计中的潜在优势，并鼓励进一步对HuPPI网络进行理论和实验探索。


### 论文摘要

Hyperuniformity, which is a type of long-range order that is characterized by the suppression of long-range density fluctuations in comparison to the fluctuations in standard disordered systems, has emerged as a powerful concept to aid in the understanding of diverse natural and engineered phenomena. In the present paper, we harness hyperuniform point patterns to generate a class of disordered, spatially embedded networks that are distinct from both perfectly ordered lattices and uniformly random geometric graphs. We refer to these networks as \emph{hyperuniform-point-pattern-induced (HuPPI) networks}, and we compare them to their counterpart \emph{Poisson-point-pattern-induced (PoPPI) networks}. By computing the local geometric and transport properties of HuPPI networks, we demonstrate how hyperuniformity imparts advantages in both transport efficiency and robustness. Specifically, we show that HuPPI networks have systematically smaller total effective resistances, slightly faster random-walk mixing times, and fewer extreme-curvature edges than PoPPI networks. Counterintuitively, we also find that HuPPI networks simultaneously have more negative mean Ollivier--Ricci curvatures and smaller global resistances than PoPPI networks, indicating that edges with moderately negative curvatures need not create severe bottlenecks to transport. We also demonstrate that the network-generation method strongly influences these properties and in particular that it often overshadows differences that arise from underlying point patterns. These results collectively demonstrate potential advantages of hyperuniformity in network design and motivate further theoretical and experimental exploration of HuPPI networks.

---

## 130. Non-uniform Thermal Conductivity in Nanoscale Multiple Hotspot Systems

**论文链接:** [http://arxiv.org/abs/2511.21070v1](http://arxiv.org/abs/2511.21070v1)

**作者:** Yu He, Zhihao Zhou, Lina Yang, Nuo Yang

**发布时间:** 2025-11-26

**备注:** 11 pages, 4 figures

### GPT解析

### 总结

研究纳米尺度多个热点系统中的热传输特性，发现紧密排列的热点可增强散热能力，挑战了热传输空间均匀的传统观点，为高功率密度集成电路热管理提供新见解。

### 背景

理解纳米尺度热点热传输对电子设备至关重要。与普遍认知不同，最近的实验表明，紧密排列的纳米尺度多个热点可以增强散热能力。

### 目的

研究纳米尺度多个热点系统中的热传输特性，探索热点间距对热传输的影响和机制。

### 方法

通过求解声子玻尔兹曼传输方程研究纳米尺度多个热点系统中的热传输，提出使用局部热导率来描述热传输能力的非均匀空间分布。

### 主要发现

1) 局部热导率的最大值比均匀加热情况高出高达27%，这是由于从热点发出的未散射声子的空间变化分数所致；2) 减少热点间距可以增强热通量，最多可达40%。

### 结论

这项工作挑战了热传输能力在整个系统中空间均匀的传统观点，为高功率密度集成电路的热管理提供了基础见解。

### 翻译

理解纳米尺度热点热传输对电子设备至关重要。与普遍认知不同，最近的实验表明，紧密排列的纳米尺度多个热点可以增强散热能力。在此，通过求解声子玻尔兹曼传输方程研究了纳米尺度多个热点系统中的热传输。提出了局部热导率来描述纳米尺度多个热点系统中热传输能力的非均匀空间分布。其最大值比均匀加热情况高出高达27%，这是由于从热点发出的未散射声子的空间变化分数所致。此外，研究了热点间距对热传输的影响和机制，表明减少热点间距可以增强热通量，最多可达40%。这项工作挑战了热传输能力在整个系统中空间均匀的传统观点，为高功率密度集成电路的热管理提供了基础见解。


### 论文摘要

Understanding nanoscale hotspot thermal transport is crucial in electronic devices. Contrary to common perception, recent experiments show that closely spaced nanoscale multiple hotspots can enhance heat dissipation. Here, the thermal transport in nanoscale multiple hotspot systems is investigated by solving the phonon Boltzmann transport equation. The local thermal conductivity is proposed to describe the non-uniform spatial distribution of heat transport capability in nanoscale multiple hotspot systems. The maximum value exceeds the uniform heating case by up to 27%, which is attributed to the spatially varying fraction of unscattered phonons emitted from hotspots. Moreover, the effects and mechanisms of hotspot spacing on thermal transport are investigated, showing that reducing the hotspot spacing can enhance the heat flux by up to 40%. This work challenges the conventional view that thermal transport capability is spatially uniform throughout the system and provides fundamental insights for thermal management in high-power-density integrated circuits.

---

## 131. Open Vocabulary Compositional Explanations for Neuron Alignment

**论文链接:** [http://arxiv.org/abs/2511.20931v1](http://arxiv.org/abs/2511.20931v1)

**作者:** Biagio La Rosa, Leilani H. Gilpin

**发布时间:** 2025-11-25

**备注:** 47 pages, 11 figures

### GPT解析

### 总结

本文提出了一种适用于视觉领域的框架，使用户能够探测任意概念和数据集的神经元，突破了传统组合解释方法依赖人工标注数据集的限制。

### 背景

神经元是深度神经网络的基本构建块，其互连使AI能够取得前所未有的成果。组合解释利用概念间的逻辑关系表达神经元激活与人类知识之间的空间对齐，但这些解释通常依赖人工标注的数据集，限制了其在特定领域和预定义概念中的应用。

### 目的

解决现有组合解释方法对人工标注数据集的依赖，开发一个适用于视觉领域的框架，支持用户对任意概念和数据集进行神经元探测。

### 方法

该框架利用开放词汇语义分割生成的掩码来计算开放词汇组合解释，包含三个步骤：指定任意概念、使用开放词汇模型生成语义分割掩码、从这些掩码推导组合解释。

### 主要发现

论文将新框架与现有方法在定量指标和人类可解释性方面进行了比较，分析了从人工标注数据转向模型标注数据时解释的差异，展示了框架在解释相对于任务和感兴趣属性的灵活性方面的优势。

### 结论

所提出的框架突破了传统方法在特定领域和预定义概念上的限制，为神经元解释提供了更灵活、更广泛的应用可能性。

### 翻译

神经元是深度神经网络的基本构建块，它们的互连使AI能够取得前所未有的成果。受理解神经元如何编码信息的启发，组合解释利用概念间的逻辑关系来表达神经元激活与人类知识之间的空间对齐。然而，这些解释依赖于人工标注的数据集，限制了它们在特定领域和预定义概念中的应用。本文通过引入一个适用于视觉领域的框架解决了这一限制，该框架允许用户探测任意概念和数据集的神经元。具体而言，该框架利用开放词汇语义分割生成的掩码来计算开放词汇组合解释。所提出的框架包括三个步骤：指定任意概念、使用开放词汇模型生成语义分割掩码、从这些掩码推导组合解释。论文将所提出的框架与之前计算组合解释的方法在定量指标和人类可解释性方面进行了比较，分析了从人工标注数据转向模型标注数据时解释的差异，并展示了框架在解释相对于任务和感兴趣属性的灵活性方面提供的额外能力。


### 论文摘要

Neurons are the fundamental building blocks of deep neural networks, and their interconnections allow AI to achieve unprecedented results. Motivated by the goal of understanding how neurons encode information, compositional explanations leverage logical relationships between concepts to express the spatial alignment between neuron activations and human knowledge. However, these explanations rely on human-annotated datasets, restricting their applicability to specific domains and predefined concepts. This paper addresses this limitation by introducing a framework for the vision domain that allows users to probe neurons for arbitrary concepts and datasets. Specifically, the framework leverages masks generated by open vocabulary semantic segmentation to compute open vocabulary compositional explanations. The proposed framework consists of three steps: specifying arbitrary concepts, generating semantic segmentation masks using open vocabulary models, and deriving compositional explanations from these masks. The paper compares the proposed framework with previous methods for computing compositional explanations both in terms of quantitative metrics and human interpretability, analyzes the differences in explanations when shifting from human-annotated data to model-annotated data, and showcases the additional capabilities provided by the framework in terms of flexibility of the explanations with respect to the tasks and properties of interest.

---

## 132. Dynamic Modeling of Load Demand in Electrified Highways Based on the EV Composition

**论文链接:** [http://arxiv.org/abs/2511.20874v1](http://arxiv.org/abs/2511.20874v1)

**作者:** Ashutossh Gupta, Vassilis Kekatos, Dionysios Aliprantis, Steve Pekarek

**发布时间:** 2025-11-25

**备注:** 5 pages, 3 figures, 1 table

### GPT解析

### 总结

该研究探讨了配备动态无线充电技术的电气化道路如何影响电动汽车的电力需求特性，并建立了相关模型来分析其动态行为和谐波特性。

### 背景

电气化道路配备动态无线充电技术可延长电动汽车行驶里程并减少车载电池需求，但由于发射器线圈的空间排列，电动汽车接收器线圈吸收的功率具有振荡特性。

### 目的

理解动态无线充电总负载的动态行为，为电力系统动态研究提供支持，并分析不同因素对负载特性的影响。

### 方法

在恒定车速下，在时间和频域中对单个电动汽车的负载进行建模；提出并分析了由电气化道路段服务的动态无线充电总负载的随机模型；使用交通模拟器的真实流量验证分析结果。

### 主要发现

非线性控制方案比线性方案具有更温和的频率谐波；EV负载的谐波幅度随接收器线圈长度减小；电气化道路上电动汽车的组成影响其频谱特性；为具有更长接收器线圈的更多电动汽车提供服务不一定意味着谐波更温和。

### 结论

该研究建立的模型和分析发现为电网运营商和电气化道路设计师提供了有价值的见解，有助于优化电力系统设计和电气化道路规划。

### 翻译

配备动态无线充电技术的电气化道路可以实现更长的行驶里程并减少电动汽车的车载电池需求。由于嵌入电气化道路路面的发射器线圈的空间排列，电动汽车接收器线圈吸收的功率本质上是振荡的。因此，理解动态无线充电总负载的动态行为对电力系统动态研究很重要。为此，我们在恒定车速下对单个电动汽车的负载在时间和频域中进行了建模。我们确定，现有动态无线充电电动汽车中实施的非线性控制方案与线性方案相比表现出更温和的频率谐波。根据该模型，电动汽车负载的谐波幅度随接收器线圈长度减小。我们进一步提出并分析了由电气化道路段服务的动态无线充电总负载的随机模型。我们的模型解释了电气化道路上电动汽车的组成如何影响其频谱。有趣的是，我们表明为具有更长接收器线圈(卡车)的更多电动汽车提供服务不一定意味着谐波更温和。我们的分析发现通过交通模拟器的真实流量得到证实，并为电网运营商和电气化道路设计师提供了有价值的见解。


### 论文摘要

Electrified roadways (ERs) equipped with the dynamic wireless power transfer (DWPT) technology can achieve longer driving range and reduce on-board battery requirements for electric vehicles (EVs). Due to the spatial arrangement of transmitter (Tx) coils embedded into the ER pavement, the power drawn by the EV's receiver (Rx) coil is oscillatory in nature. Therefore, understanding the dynamic behavior of the total DWPT load is important for power system dynamic studies. To this end, we model the load of individual EVs in the time and frequency domains for constant EV speed. We establish that a nonlinear control scheme implemented in existing DWPT-enabled EVs exhibits milder frequency harmonics compared to its linear alternative. According to this model, the harmonics of an EV load decrease in amplitude with the Rx coil length. We further propose and analyze stochastic models for the total DWPT load served by an ER segment. Our models explain how the EV composition on the ER affects its frequency spectrum. Interestingly, we show that serving more EVs with longer Rx coils (trucks) does not necessarily entail milder harmonics. Our analytical findings are corroborated using realistic flows from a traffic simulator and offer valuable insights to grid operators and ER designers.

---

## 133. A new Fractal Mean-Field analysis in phase transition

**论文链接:** [http://arxiv.org/abs/2511.20846v1](http://arxiv.org/abs/2511.20846v1)

**作者:** Ismael S. S. Carrasco, Henrique A. de Lima, Fernando A. Oliveira

**发布时间:** 2025-11-25

### GPT解析

### 总结

该研究重新审视了二阶相变系统中相关性的理论基础，特别关注扩展到非整数空间维度的伊辛模型，建立了相关性与分形几何之间的联系，并验证了临界标度理论的普适性。

### 背景

理解相变不仅需要识别序参量，还需要表征它们在不同尺度上的相关性。相关性函数通过量化不同空间或时间点上的波动如何相关，揭示了复杂系统的结构组织。

### 目的

重新审视二阶相变系统中相关性的理论基础，特别关注扩展到非整数空间维度的伊辛模型，建立相关性与分形几何之间的联系。

### 方法

从Fisher引入的经典框架出发，讨论标准欧几里得处理如何引入临界指数来捕捉临界温度下相关性的空间衰减。假设在临界点，平衡动力学被有效地限制在自旋簇的分形边缘内，推导连接两个分形维度的几何关系。

### 主要发现

在临界点，控制子空间相关性的分形维度与Fisher指数直接相关，后者量化了临界点附近相关性函数的奇异行为。这种相关性分形维度与序参量相关的分形维度不同，但存在明确的几何关系连接它们。这种处理方法扩展到非整数空间维度仍然有效，并产生正确的Fisher指数值。

### 结论

Rushbrooke标度关系在将空间维度视为连续参数时仍然成立，这强化了临界标度的普适性，并强调了分形几何在表征临界点相关性中的重要作用。

### 翻译

理解相变不仅需要识别序参量，还需要表征它们在不同尺度上的相关性。通过量化不同空间或时间点上的波动如何相关，相关性函数揭示了复杂系统的结构组织。在此，我们重新审视了二阶相变系统中相关性的理论基础，特别关注扩展到非整数空间维度的伊辛模型。从Fisher引入的经典框架出发，我们讨论了标准欧几里得处理（限于整数维度）如何引入临界指数来捕捉临界温度时相关性的空间衰减。我们假设，在临界点，平衡动力学被有效地限制在自旋簇的分形边缘内。在该框架内，控制该子空间相关性的分形维度与Fisher指数直接相关，后者量化了临界点附近相关性函数的奇异行为。重要的是，这种相关性分形维度与序参量相关的分形维度不同。我们进一步推导了连接两个分形维度的明确几何关系，从而将空间自相似性与临界点观察到的标度行为联系起来。这种处理方法自然地扩展到非整数空间维度，在上临界维度以下仍然有效，并为连续空间维度产生正确的Fisher指数值。我们的分析还证实，Rushbrooke标度关系在将空间维度视为连续参数时仍然成立，强化了临界标度的普适性，并强调了分形几何在表征临界点相关性中的作用。


### 论文摘要

Understanding phase transitions requires not only identifying order parameters but also characterizing how their correlations behave across scales. By quantifying how fluctuations at distinct spatial or temporal points are related, correlation functions reveal the structural organization of complex systems. Here, we revisit the theoretical foundations of these correlations in systems undergoing second-order phase transitions, with emphasis on the Ising model extended to non-integer spatial dimensions. Starting from the classical framework introduced by Fisher, we discuss how the standard Euclidean treatment, restricted to integer dimensions, necessitates the introduction of the critical exponent $η$ to capture the spatial decay of correlations at $T=T_c$. We suppose that, at criticality, the equilibrium dynamics become effectively confined to the fractal edge of spin clusters. Within this framework, the fractal dimension that governs the correlations in that subspace is directly related to Fisher exponent, which quantifies the singular behavior of the correlation function near criticality. Importantly, this correlation fractal dimension is distinct from the fractal dimension associated with the order parameter. We further derive an explicit geometrical relation connecting the two fractal dimensions, thereby linking spatial self-similarity to the observed scaling behavior at criticality. This treatment naturally extends to non-integer spatial dimensions, which remain valid below the upper critical dimension and produce the correct value of Fisher exponent $η$ for a continuous space dimension. Our analysis also confirms that the Rushbrooke scaling relation, continues to hold when the spatial dimension is treated as a continuous parameter, reinforcing the universality of critical scaling and underscoring the role of fractal geometry in characterizing correlations at criticality.

---

## 134. SPHINX: A Synthetic Environment for Visual Perception and Reasoning

**论文链接:** [http://arxiv.org/abs/2511.20814v1](http://arxiv.org/abs/2511.20814v1)

**作者:** Md Tanvirul Alam, Saksham Aggarwal, Justin Yang Chae, Nidhi Rastogi

**发布时间:** 2025-11-25

### GPT解析

### 总结

介绍Sphinx，一个用于视觉感知和推理的合成环境，包含程序化生成的谜题和可验证的解决方案，用于评估和提升视觉语言模型的表现。

### 背景

现有的大型视觉语言模型在视觉推理任务上的表现有待提高，缺乏专门的评估基准。

### 目的

创建一个全面的视觉推理环境，用于精确评估模型性能并推动多模态推理能力的进步。

### 方法

开发Sphinx合成环境，使用图案、瓦片、图表等元素生成25种类型的视觉推理任务，并应用可验证奖励的强化学习方法进行模型训练。

### 主要发现

即使是先进的GPT-5模型在Sphinx基准测试中也仅达到51.1%的准确率，远低于人类表现；使用可验证奖励的强化学习能显著提高模型准确率。

### 结论

Sphinx为视觉推理提供了全面的评估框架，可验证奖励的强化学习方法是提升模型视觉推理能力的有效途径。

### 翻译

我们提出了Sphinx，一个针对核心认知原语的可视化感知与推理的合成环境。Sphinx使用图案、瓦片、图表、图标和几何原语程序化生成谜题，每个都配有可验证的基准真值解决方案，既能实现精确评估，也能进行大规模数据集构建。该基准测试涵盖25种任务类型，包括对称性检测、几何变换、空间推理、图表解释和序列预测。评估近期的大型视觉语言模型显示，即使是先进的GPT-5也只达到51.1%的准确率，远低于人类表现。最后，我们展示了使用可验证奖励的强化学习能显著提高模型在这些任务上的准确率，并在外部视觉推理基准测试中取得进步，凸显了其在推进多模态推理方面的潜力。


### 论文摘要

We present Sphinx, a synthetic environment for visual perception and reasoning that targets core cognitive primitives. Sphinx procedurally generates puzzles using motifs, tiles, charts, icons, and geometric primitives, each paired with verifiable ground-truth solutions, enabling both precise evaluation and large-scale dataset construction. The benchmark covers 25 task types spanning symmetry detection, geometric transformations, spatial reasoning, chart interpretation, and sequence prediction. Evaluating recent large vision-language models (LVLMs) shows that even state-of-the-art GPT-5 attains only 51.1% accuracy, well below human performance. Finally, we demonstrate that reinforcement learning with verifiable rewards (RLVR) substantially improves model accuracy on these tasks and yields gains on external visual reasoning benchmarks, highlighting its promise for advancing multimodal reasoning.

---

## 135. Intriguing Properties of Dynamic Sampling Networks

**论文链接:** [http://arxiv.org/abs/2511.20800v1](http://arxiv.org/abs/2511.20800v1)

**作者:** Dario Morle, Reid Zaffino

**发布时间:** 2025-11-25

### GPT解析

### 总结

本文提出了一种名为'warping'的新型算子，统一了各种动态采样方法，提供了动态采样的最小实现，可用于重建现有架构。通过理论分析和实证研究，作者发现了模型训练前向和后向传播间的独特不对称性，并证明动态采样机制与传统卷积算子正交。此外，作者还提出了新的损失景观可视化方法。

### 背景

深度学习架构中的动态采样机制已在许多计算机视觉模型中显示出效用，但这些结构的理论分析尚未统一。

### 目的

连接各种动态采样方法，通过开发和分析一种称为'warping'的新型算子来泛化现有方法，并提供一个易于分析的动态采样最小实现。

### 方法

开发并分析名为'warping'的新型算子；将输入建模为独立同分布变量和齐次随机场；进行理论分析和实证研究；研究离散化效应的统计分析；引入利用梯度更新信息的新的损失景观可视化方法。

### 主要发现

动态采样机制代表了一类与传统卷积定义的平移不变算子正交的完全不同的算子；模型训练的前向和后向传播之间存在独特的不对称性；找到了确保动态采样网络稳定训练的必要条件。

### 结论

通过理论分析和实证研究相结合，作者发现了确保动态采样网络稳定训练的必要条件，并研究了离散化效应的统计分析，同时提出了新的损失景观可视化方法以更好地理解学习行为。

### 翻译

深度学习架构中的动态采样机制已在许多计算机视觉模型中显示出效用，但这些结构的理论分析尚未统一。在本文中，我们通过开发和分析一种称为'warping'的新型算子来连接各种动态采样方法，该算子泛化了现有方法。Warping提供了动态采样的最小实现，便于分析，并可用于重建现有架构，包括可变形卷积、活动卷积单元和空间变换网络。使用我们的形式化方法，我们通过将输入建模为独立同分布变量和齐次随机场，对该算子进行了统计分析。扩展这一分析，我们发现了模型训练的前向和后向传播之间的独特不对称性。我们证明这些机制代表了一类与传统卷积定义的平移不变算子正交的完全不同的算子。通过结合理论分析和实证研究，我们发现了确保动态采样网络稳定训练的必要条件。此外，还研究了离散化效应的统计分析。最后，我们引入了一种新的损失景观可视化方法，它直接利用梯度更新信息，以更好地理解学习行为。


### 论文摘要

Dynamic sampling mechanisms in deep learning architectures have demonstrated utility across many computer vision models, though the theoretical analysis of these structures has not yet been unified. In this paper we connect the various dynamic sampling methods by developing and analyzing a novel operator which generalizes existing methods, which we term "warping". Warping provides a minimal implementation of dynamic sampling which is amenable to analysis, and can be used to reconstruct existing architectures including deformable convolutions, active convolutional units, and spatial transformer networks. Using our formalism, we provide statistical analysis of the operator by modeling the inputs as both IID variables and homogeneous random fields. Extending this analysis, we discover a unique asymmetry between the forward and backward pass of the model training. We demonstrate that these mechanisms represent an entirely different class of orthogonal operators to the traditional translationally invariant operators defined by convolutions. With a combination of theoretical analysis and empirical investigation, we find the conditions necessary to ensure stable training of dynamic sampling networks. In addition, statistical analysis of discretization effects are studied. Finally, we introduce a novel loss landscape visualization which utilizes gradient update information directly, to better understand learning behavior.

---

## 136. One Patch is All You Need: Joint Surface Material Reconstruction and Classification from Minimal Visual Cues

**论文链接:** [http://arxiv.org/abs/2511.20784v1](http://arxiv.org/abs/2511.20784v1)

**作者:** Sindhuja Penchala, Gavin Money, Gabriel Marques, Samuel Wood, Jessica Kirschman, Travis Atkison, Shahram Rahimi, Noorbakhsh Amiri Golilarz

**发布时间:** 2025-11-25

**备注:** 9 pages,3 figures, 5 tables

### GPT解析

### 总结

SMARC是一种统一的表面材料重建和分类模型，仅需10%的图像连续块即可完成完整RGB表面重建和材料分类，在稀疏视觉条件下表现优异。

### 背景

理解材料表面对机器人、模拟和材料感知应用至关重要，但现有方法大多依赖于密集或完整场景观察，在受限或部分视角环境中效果有限。

### 目的

开发一种从最少的视觉输入中进行表面材料重建和分类的模型，解决从稀疏视觉线索理解材料表面的挑战。

### 方法

结合部分卷积U-Net和分类头的SMARC架构，能够在极端观察稀疏性下实现空间修复和语义理解；在Touch and Go数据集上与卷积自编码器、视觉Transformer、掩码自编码器、Swin Transformer和DETR等模型进行了比较。

### 主要发现

SMARC在PSNR上达到17.55 dB的最新结果，材料分类准确率达85.10%；部分卷积在缺失数据下的空间推理方面具有显著优势。

### 结论

SMARC建立了最小视觉表面理解的坚实基础，为受限环境中的材料感知提供了有效解决方案。

### 翻译

从稀疏视觉线索理解材料表面对机器人、模拟和材料感知应用至关重要。然而，大多数现有方法依赖于密集或完整场景的观察，限制了它们在受限或部分视角环境中的有效性。为应对这一挑战，我们引入了SMARC，一个用于从最少视觉输入中进行表面材料重建和分类的统一模型。仅需提供图像的10%连续块，SMARC就能识别和重建完整的RGB表面，同时分类材料类别。我们的架构将部分卷积U-Net与分类头相结合，能够在极端观察稀疏性下实现空间修复和语义理解。我们使用真实世界表面纹理的Touch and Go数据集[16]，将SMARC与包括卷积自编码器[17]、视觉Transformer(ViT)[13]、掩码自编码器(MAE)[5]、Swin Transformer[9]和DETR[2]在内的五种模型进行了比较。SMARC以17.55 dB的PSNR和85.10%的材料分类准确率取得了最新成果。我们的研究突显了部分卷积在缺失数据下空间推理的优势，并为最小视觉表面理解奠定了坚实基础。


### 论文摘要

Understanding material surfaces from sparse visual cues is critical for applications in robotics, simulation, and material perception. However, most existing methods rely on dense or full-scene observations, limiting their effectiveness in constrained or partial view environment. To address this challenge, we introduce SMARC, a unified model for Surface MAterial Reconstruction and Classification from minimal visual input. By giving only a single 10% contiguous patch of the image, SMARC recognizes and reconstructs the full RGB surface while simultaneously classifying the material category. Our architecture combines a Partial Convolutional U-Net with a classification head, enabling both spatial inpainting and semantic understanding under extreme observation sparsity. We compared SMARC against five models including convolutional autoencoders [17], Vision Transformer (ViT) [13], Masked Autoencoder (MAE) [5], Swin Transformer [9], and DETR [2] using Touch and Go dataset [16] of real-world surface textures. SMARC achieves state-of-the-art results with a PSNR of 17.55 dB and a material classification accuracy of 85.10%. Our findings highlight the advantages of partial convolution in spatial reasoning under missing data and establish a strong foundation for minimal-vision surface understanding.

---

## 137. Agentic Learner with Grow-and-Refine Multimodal Semantic Memory

**论文链接:** [http://arxiv.org/abs/2511.21678v1](http://arxiv.org/abs/2511.21678v1)

**作者:** Weihao Bo, Shan Zhang, Yanpeng Sun, Jingjing Wu, Qunyi Xie, Xiao Tan, Kunbin Chen, Wei He, Xiaofan Li, Na Zhao, Jingdong Wang, Zechao Li

**发布时间:** 2025-11-26

### GPT解析

### 总结

这篇论文介绍了一种名为ViLoMem的双流记忆框架，旨在解决多模态大语言模型(MLLMs)在解决独立问题时重复犯错的问题。该框架通过分别编码视觉干扰模式和逻辑推理错误，使模型能够从成功和失败的经验中学习，并在多模态问题解决环境中提高性能。

### 背景

MLLMs在处理独立查询时表现出强大的推理能力，但它们是独立解决每个问题，往往会重复同样的错误。现有的记忆增强型代理主要存储过去的轨迹以便重用，但基于轨迹的记忆存在简洁性偏差，逐渐丢失必要的领域知识。更重要的是，即使在真正的多模态问题解决环境中，现有方法也只记录了过去行为的单一模态痕迹，无法保存视觉注意力和逻辑推理如何共同促成解决方案的过程。

### 目的

开发一种能够保存多模态语义记忆的框架，使MLLMs能够从过去的经验中学习，避免重复犯错，并在多模态问题解决环境中提高性能。

### 方法

作者引入了ViLoMem，这是一种双流记忆框架，构建紧凑的基于模式的记忆。它分别编码视觉干扰模式和逻辑推理错误，使MLLMs能够从成功和失败的经验中学习。系统遵循'增长和精炼'原则，逐步积累和更新多模态语义知识，同时保持稳定、可推广的策略，避免灾难性遗忘。

### 主要发现

在六个多模态基准测试中，ViLoMem一致提高了pass@1准确率，并显著减少了重复的视觉和逻辑错误。消融研究证实了具有明确干扰-幻觉分离的双流记忆的必要性，展示了错误感知的多模态记忆对于终身和跨领域代理学习的价值。

### 结论

ViLoMem框架通过双流记忆机制有效解决了MLLMs在多模态问题解决中重复犯错的问题，使模型能够从经验中学习并保持长期记忆。这种方法与人类认知模式更为接近，通过分别处理视觉和抽象知识，实现了更有效的多模态学习和推理。

### 翻译

MLLMs在独立查询上表现出强大的推理能力，但它们是全新操作——独立解决每个问题并经常重复同样的错误。现有的记忆增强型代理主要存储过去的轨迹以便重用。然而，基于轨迹的记忆存在简洁性偏差，逐渐丢失必要的领域知识。更重要的是，即使在真正的多模态问题解决环境中，它也只记录了过去行为的单一模态痕迹，无法保存视觉注意力和逻辑推理如何共同促成解决方案的过程。这与人类认知根本不匹配：语义记忆是多模态和集成的，通过协调但不同的表征流保存视觉和抽象知识。因此，我们引入了ViLoMem，一种双流记忆框架，构建紧凑的基于模式的记忆。它分别编码视觉干扰模式和逻辑推理错误，使MLLMs能够从成功和失败的经验中学习。遵循增长和精炼原则，系统逐步积累和更新多模态语义知识——保持稳定、可推广的策略，同时避免灾难性遗忘。在六个多模态基准测试中，ViLoMem一致提高了pass@1准确率，并显著减少了重复的视觉和逻辑错误。消融研究证实了具有明确干扰-幻觉分离的双流记忆的必要性，展示了错误感知的多模态记忆对于终身和跨领域代理学习的价值。我们的项目页面将在https://weihao-bo.github.io/ViLoMeo-page上提供。


### 论文摘要

MLLMs exhibit strong reasoning on isolated queries, yet they operate de novo -- solving each problem independently and often repeating the same mistakes. Existing memory-augmented agents mainly store past trajectories for reuse. However, trajectory-based memory suffers from brevity bias, gradually losing essential domain knowledge. More critically, even in truly multimodal problem-solving settings, it records only a single-modality trace of past behavior, failing to preserve how visual attention and logical reasoning jointly contributed to the solution. This is fundamentally misaligned with human cognition: semantic memory is both multimodal and integrated, preserving visual and abstract knowledge through coordinated but distinct representational streams. We thus introduce ViLoMem, a dual-stream memory framework that constructs compact, schema-based memory. It separately encodes visual distraction patterns and logical reasoning errors, enabling MLLMs to learn from their successful and failed experiences. Following a grow-and-refine principle, the system incrementally accumulates and updates multimodal semantic knowledge -- preserving stable, generalizable strategies while avoiding catastrophic forgetting. Across six multimodal benchmarks, ViLoMem consistently improves pass@1 accuracy and substantially reduces repeated visual and logical errors. Ablations confirm the necessity of dual-stream memory with explicit distraction--hallucination separation, demonstrating the value of error-aware multimodal memory for lifelong and cross-domain agentic learning. Our project page will be available at https://weihao-bo.github.io/ViLoMeo-page.

---

## 138. Harmonic-Percussive Disentangled Neural Audio Codec for Bandwidth Extension

**论文链接:** [http://arxiv.org/abs/2511.21580v1](http://arxiv.org/abs/2511.21580v1)

**作者:** Benoît Giniès, Xiaoyu Bie, Olivier Fercoq, Gaël Richard

**发布时间:** 2025-11-26

### GPT解析

### 总结

这项工作将带宽扩展重新定义为音频令牌预测问题，通过创新的编解码器设计和transformer模型的结合，实现了高质量的音频重建，研究结果表明整体设计方法对音频处理任务有积极影响。

### 背景

带宽扩展是音频处理中的一个长期存在的问题，其任务是从音频信号的低通分量重建高频分量。传统方法随着信号处理的趋势而发展，但最近神经架构的进步显著提高了各种音频任务的性能。

### 目的

将带宽扩展框架化为音频令牌预测问题，通过联合设计编解码器结构和transformer建模来提高重建质量。

### 方法

使用基于transformer的语言模型，在由解耦神经音频编解码器产生的离散表示上进行训练，解耦过程由输入信号的谐波-打击分解指导，引入了一种新的编解码器设计，明确考虑了下游令牌预测任务，实现编解码器结构与transformer建模之间的更有效耦合。

### 主要发现

联合设计能够高质量地重建原始信号，客观指标和主观评估都证明了这一点。

### 结论

强调了将编解码器解耦和表示学习与生成建模阶段对齐的重要性，展示了全局、感知表示设计在推进带宽扩展方面的潜力。

### 翻译

带宽扩展，即从音频信号的低通分量重建其高频分量的任务，是音频处理中的一个长期存在的问题。虽然传统方法随着信号处理的更广泛趋势而发展，但最近神经架构的进步已经显著提高了各种音频任务的性能。在这项工作中，我们将这些进展扩展，将带宽扩展框架化为音频令牌预测问题。具体来说，我们在解耦神经音频编解码器产生的离散表示上训练一个基于transformer的语言模型，其中解耦由输入信号的谐波-打击分解指导，突出了对带宽扩展特别相关的频谱结构。我们的方法引入了一种新的编解码器设计，明确考虑了下游令牌预测任务，实现了编解码器结构与transformer建模之间的更有效耦合。这种联合设计通过客观指标和主观评估测量，能够高质量地重建原始信号。这些结果强调了将编解码器解耦和表示学习与生成建模阶段对齐的重要性，并展示了全局、感知表示设计在推进带宽扩展方面的潜力。


### 论文摘要

Bandwidth extension, the task of reconstructing the high-frequency components of an audio signal from its low-pass counterpart, is a long-standing problem in audio processing. While traditional approaches have evolved alongside the broader trends in signal processing, recent advances in neural architectures have significantly improved performance across a wide range of audio tasks, In this work, we extend these advances by framing bandwidth extension as an audio token prediction problem. Specifically, we train a transformer-based language model on the discrete representations produced by a disentangled neural audio codec, where the disentanglement is guided by a Harmonic-Percussive decomposition of the input signals, highlighting spectral structures particularly relevant for bandwidth extension. Our approach introduces a novel codec design that explicitly accounts for the downstream token prediction task, enabling a more effective coupling between codec structure and transformer modeling. This joint design yields high-quality reconstructions of the original signal, as measured by both objective metrics and subjective evaluations. These results highlight the importance of aligning codec disentanglement and representation learning with the generative modeling stage, and demonstrate the potential of global, representation-aware design for advancing bandwidth extension.

---

## 139. Guiding Generative Models for Protein Design: Prompting, Steering and Aligning

**论文链接:** [http://arxiv.org/abs/2511.21476v1](http://arxiv.org/abs/2511.21476v1)

**作者:** Filippo Stocco, Michele Garibbo, Noelia Ferruz

**发布时间:** 2025-11-26

### GPT解析

### 总结

生成式AI模型从数据中学习概率分布并产生新样本，蛋白质因其丰富数据和多样表示形式特别适合生成式方法。生成模型在蛋白质设计领域取得显著成功，但倾向于探索训练分布中最可能的区域，忽略可能包含有价值属性的低概率区域。近期方法开始解决这一限制，使生成模型能够产生具有期望属性的蛋白质。

### 背景

生成式AI模型可以从数据中学习概率分布并生成新样本。蛋白质因其丰富的数据和多样的表示形式（序列、结构和功能）而特别适合生成式方法。

### 目的

这篇综述旨在分类和概述条件生成模型在蛋白质设计中的最新进展，特别是那些指导生成模型产生具有用户指定属性的蛋白质的方法。

### 方法

作者区分了两种主要方法：1) 修改模型参数的方法，如强化学习或监督微调；2) 保持模型固定的方法，包括条件生成、检索增强策略、贝叶斯引导和定制采样方法。

### 主要发现

生成模型在蛋白质设计领域取得了显著成功，但倾向于探索训练分布中最可能的区域，而忽略可能包含有价值属性的低概率区域。近期方法已经开始解决这一限制，使生成模型能够产生具有期望属性的蛋白质。

### 结论

这些新的条件生成方法正在开始实现生成模型向具有期望的、通常是以前难以获得的属性的蛋白质的引导。

### 翻译

生成式人工智能(AI)模型从数据中学习概率分布，并产生捕捉其训练集显著属性的新颖样本。鉴于蛋白质的丰富数据和表示形式的多样性（从序列到结构和功能），它们特别适合这类方法。这种多样性推动了蛋白质设计生成模型的快速发展，使得以前所未有的成功生成功能蛋白质和酶成为可能。然而，由于这些模型反映了其训练分布，它们倾向于从最可能的模式中采样，而通常编码有价值属性的低概率区域则探索不足。为了应对这一挑战，近期工作专注于指导生成模型产生具有用户指定属性的蛋白质，即使这些属性在原始训练分布中很少见或不存在。在本综述中，我们对蛋白质设计中有条件生成模型的最新进展进行了分类和概述。我们区分了修改模型参数的方法（如强化学习或监督微调）和保持模型固定的方法（包括条件生成、检索增强策略、贝叶斯引导和定制采样方法）。这些发展共同开始使生成模型能够转向具有期望的、通常是以前难以获得的属性的蛋白质。


### 论文摘要

Generative artificial intelligence (AI) models learn probability distributions from data and produce novel samples that capture the salient properties of their training sets. Proteins are particularly attractive for such approaches given their abundant data and the versatility of their representations, ranging from sequences to structures and functions. This versatility has motivated the rapid development of generative models for protein design, enabling the generation of functional proteins and enzymes with unprecedented success. However, because these models mirror their training distribution, they tend to sample from its most probable modes, while low-probability regions, often encoding valuable properties, remain underexplored. To address this challenge, recent work has focused on guiding generative models to produce proteins with user-specified properties, even when such properties are rare or absent from the original training distribution. In this review, we survey and categorize recent advances in conditioning generative models for protein design. We distinguish approaches that modify model parameters, such as reinforcement learning or supervised fine-tuning, from those that keep the model fixed, including conditional generation, retrieval-augmented strategies, Bayesian guidance, and tailored sampling methods. Together, these developments are beginning to enable the steering of generative models toward proteins with desired, and often previously inaccessible, properties.

---

## 140. Ensemble Performance Through the Lens of Linear Independence of Classifier Votes in Data Streams

**论文链接:** [http://arxiv.org/abs/2511.21465v1](http://arxiv.org/abs/2511.21465v1)

**作者:** Enes Bektas, Fazli Can

**发布时间:** 2025-11-26

**备注:** 14 pages, 3 figures, 5 tables

### GPT解析

### 总结

该研究通过线性独立性的视角探讨了集成大小与性能的关系，提出了理论框架和估计方法，并通过实验验证了其有效性。

### 背景

集成学习通过组合多个基础分类器来提高分类性能。增加分类器数量通常会提高准确性，但过大的集成会导致计算效率低下和收益递减。

### 目的

研究集成大小与性能之间的关系，特别关注数据流中分类器投票的线性独立性。

### 方法

提出由线性独立分类器组成的集成能最大化表示能力，特别是在几何模型下；将线性独立性的重要性推广到加权多数投票问题；通过建模分类器输出实现线性独立的概率，推导出解释集成大小与准确性之间权衡的理论框架；在真实和合成数据集上使用OzaBagging和GOOWE两种集成方法进行实验验证。

### 主要发现

理论估计能有效识别稳健集成(如OzaBagging)的性能饱和点；对于复杂的加权方案(如GOOWE)，该框架揭示了高理论多样性可能引发算法不稳定性。

### 结论

实现公开可用，以支持可复现性和未来研究。

### 翻译

集成学习通过组合多个基础分类器来提高分类性能。虽然增加分类器数量通常会提高准确性，但过大的集成会导致计算效率低下和收益递减。本文从数据流中分类器投票的线性独立性角度研究了集成大小与性能之间的关系。我们提出由线性独立分类器组成的集成能最大化表示能力，特别是在几何模型下。然后我们将线性独立性的重要性推广到加权多数投票问题。通过建模分类器输出实现线性独立的概率，我们推导出一个解释集成大小与准确性之间权衡的理论框架。我们的分析得出了实现用户指定的线性独立性概率所需集成大小的理论估计。我们使用OzaBagging和GOOWE两种集成方法在真实和合成数据集上的实验验证了我们的理论。我们的结果确认，这一理论估计能有效识别稳健集成(如OzaBagging)的性能饱和点。相反，对于复杂的加权方案(如GOOWE)，我们的框架揭示了高理论多样性可能引发算法不稳定性。我们的实现公开可用，以支持可复现性和未来研究。


### 论文摘要

Ensemble learning improves classification performance by combining multiple base classifiers. While increasing the number of classifiers generally enhances accuracy, excessively large ensembles can lead to computational inefficiency and diminishing returns. This paper investigates the relationship between ensemble size and performance through the lens of linear independence among classifier votes in data streams. We propose that ensembles composed of linearly independent classifiers maximize representational capacity, particularly under a geometric model. We then generalize the importance of linear independence to the weighted majority voting problem. By modeling the probability of achieving linear independence among classifier outputs, we derive a theoretical framework that explains the trade-off between ensemble size and accuracy. Our analysis leads to a theoretical estimate of the ensemble size required to achieve a user-specified probability of linear independence. We validate our theory through experiments on both real-world and synthetic datasets using two ensemble methods, OzaBagging and GOOWE. Our results confirm that this theoretical estimate effectively identifies the point of performance saturation for robust ensembles like OzaBagging. Conversely, for complex weighting schemes like GOOWE, our framework reveals that high theoretical diversity can trigger algorithmic instability. Our implementation is publicly available to support reproducibility and future research.

---

## 141. EvRainDrop: HyperGraph-guided Completion for Effective Frame and Event Stream Aggregation

**论文链接:** [http://arxiv.org/abs/2511.21439v1](http://arxiv.org/abs/2511.21439v1)

**作者:** Futian Wang, Fan Zhang, Xiao Wang, Mengqi Wang, Dexing Huang, Jin Tang

**发布时间:** 2025-11-26

### GPT解析

### 总结

本文提出了一种超图引导的时空事件流补全机制，解决了事件相机数据的空间稀疏性问题，通过超图连接不同时空位置的事件令牌，利用上下文信息进行补全，并支持多模态信息融合。

### 背景

事件相机产生异步事件流，在空间上稀疏但在时间上密集。主流的事件表征学习算法通常使用事件帧、体素或张量作为输入，但这些方法难以解决由空间稀疏性引起的欠采样问题。

### 目的

提出一种新的超图引导的时空事件流补全机制，解决事件相机数据的空间稀疏性问题，并实现多模态信息融合。

### 方法

提出超图引导的时空事件流补全机制，通过超图连接不同时间和空间位置的事件令牌，利用上下文信息消息传递来补全稀疏事件；灵活将RGB令牌作为超图节点纳入补全框架，实现多模态超图信息补全；通过自注意力聚合不同时间步的超图节点信息，实现多模态特征的有效学习和融合。

### 主要发现

在单标签和多标签事件分类任务上的大量实验充分验证了所提出框架的有效性。

### 结论

提出的超图引导的时空事件流补全机制有效解决了事件相机数据的空间稀疏性问题，能够整合RGB信息实现多模态信息补全，并通过自注意力机制有效聚合多模态特征。

### 翻译

事件相机产生异步事件流，在空间上稀疏但在时间上密集。主流的事件表征学习算法通常使用事件帧、体素或张量作为输入。尽管这些方法取得了显著进展，但难以解决由空间稀疏性引起的欠采样问题。在本文中，我们提出了一种新的超图引导的时空事件流补全机制，通过超图连接不同时间和空间位置的事件令牌，并利用上下文信息消息传递来补全这些稀疏事件。所提出的方法可以灵活地将RGB令牌作为超图中的节点纳入此补全框架，实现基于超图的多模态信息补全。随后，我们通过自注意力聚合不同时间步的超图节点信息，实现多模态特征的有效学习和融合。在单标签和多标签事件分类任务上的大量实验充分验证了我们所提出框架的有效性。本文的源代码将在https://github.com/Event-AHU/EvRainDrop上发布。


### 论文摘要

Event cameras produce asynchronous event streams that are spatially sparse yet temporally dense. Mainstream event representation learning algorithms typically use event frames, voxels, or tensors as input. Although these approaches have achieved notable progress, they struggle to address the undersampling problem caused by spatial sparsity. In this paper, we propose a novel hypergraph-guided spatio-temporal event stream completion mechanism, which connects event tokens across different times and spatial locations via hypergraphs and leverages contextual information message passing to complete these sparse events. The proposed method can flexibly incorporate RGB tokens as nodes in the hypergraph within this completion framework, enabling multi-modal hypergraph-based information completion. Subsequently, we aggregate hypergraph node information across different time steps through self-attention, enabling effective learning and fusion of multi-modal features. Extensive experiments on both single- and multi-label event classification tasks fully validated the effectiveness of our proposed framework. The source code of this paper will be released on https://github.com/Event-AHU/EvRainDrop.

---

## 142. $\texttt{CRLS}$: Convolutional Regularized Least Squares Framework for Reduced Order Modeling of Transonic Flows

**论文链接:** [http://arxiv.org/abs/2511.21425v1](http://arxiv.org/abs/2511.21425v1)

**作者:** Muhammad Bilal, Ashwin Renganathan

**发布时间:** 2025-11-26

**备注:** 24 pages, 13 figures

### GPT解析

### 总结

本文开发了一种卷积正则化最小二乘(CRLS)框架，用于跨音速流(包含激波)的降阶建模。该方法通过平滑处理、POD基提取和高效解卷积步骤，解决了传统POD方法在处理参数依赖性间断时的局限性，显著提高了激波建模的准确性。

### 背景

传统的基于POD的降阶模型虽然具有最优性和低在线成本的优势，但在处理包含参数依赖性间断的快照时表现不佳，会导致激波模糊、阶梯效应或非物理振荡。

### 目的

开发一种新的降阶建模框架，能够准确处理跨音速流中的激波结构，提高激波位置和强度的预测精度。

### 方法

首先通过高斯卷积和反射填充将全阶快照映射到更平滑的表示；使用贝叶斯优化自动选择卷积超参数；从平滑数据中提取POD基；通过径向基函数插值学习POD系数的参数依赖性；最后引入一个高效的解卷积步骤，公式化为正则化最小二乘问题，围绕参数空间中的最近邻参考快照进行重建。

### 主要发现

与标准POD和平滑POD基线相比，CRLS显著提高了激波位置和强度的准确性，降低了表面压力和场级误差，并将捕获固定比例快照能量所需的POD模式数量减少了42%。

### 结论

CRLS为高速气动设计提供了一种准确、数据高效且 largely 自动化的激波感知降阶模型途径。

### 翻译

我们开发了一种用于跨音速流(含激波)降阶建模的卷积正则化最小二乘(CRLS)框架。传统的基于POD的降阶模型因其最优性和低在线成本而具有吸引力；然而，当快照包含参数依赖性间断时，它们表现不佳，导致激波模糊、阶梯效应或非物理振荡。在CRLS中，我们首先通过沿流场坐标方向应用一维高斯卷积和反射填充，将每个全阶快照映射到更平滑的表示。卷积超参数(核宽度和支持)通过在保留的快照集上使用贝叶斯优化自动选择。然后从平滑数据中提取POD基，并通过径向基函数插值学习POD系数的参数依赖性。为了恢复尖锐的激波结构，我们引入了一个高效的解卷积步骤，公式化为正则化最小二乘问题，其中正则化围绕参数空间中最近邻参考快照进行重建。所得的CRLS代理模型在RAE2822翼型上的无粘跨音速流上进行评估，该模型由使用SU2求解的稳态可压缩欧拉方程建模，针对马赫数和攻角的拉丁超立方样本。与标准POD和平滑POD基线相比，CRLS显著提高了激波位置和强度的准确性，降低了表面压力和场级误差，并将捕获固定比例快照能量所需的POD模式数量减少了42%。这些结果表明，CRLS为高速气动设计提供了一种准确、数据高效且 largely 自动化的激波感知降阶模型途径。


### 论文摘要

We develop a convolutional regularized least squares ($\texttt{CRLS}$) framework for reduced-order modeling of transonic flows with shocks. Conventional proper orthogonal decomposition (POD) based reduced models are attractive because of their optimality and low online cost; however, but they perform poorly when snapshots contain parameter-dependent discontinuities, leading to smeared shocks, stair-stepping, or non-physical oscillations. In $\texttt{CRLS}$, we first map each full-order snapshot to a smoother representation by applying a one-dimensional Gaussian convolution with reflect padding along the flow field coordinates. The convolution hyperparameters (kernel width and support) are selected automatically by Bayesian optimization on a held-out set of snapshots. POD bases are then extracted from the smoothed data, and the parametric dependence of the POD coefficients is learned via radial basis function interpolation. To recover sharp shock structures, we introduce an efficient deconvolution step formulated as a regularized least squares problem, where the regularization centers the reconstruction around a nearest-neighbor reference snapshot in parameter space. The resulting $\texttt{CRLS}$ surrogate is evaluated on inviscid transonic flow over the RAE2822 airfoil, modeled by the steady compressible Euler equations solved with SU2 over a Latin hypercube sample of Mach number and angle of attack. Compared with standard POD and smoothed-POD baselines, $\texttt{CRLS}$ yields markedly improved shock location and strength, lower surface-pressure and field-level errors, and a $42$\% reduction in the number of POD modes required to capture a fixed fraction of snapshot energy. These results demonstrate that $\texttt{CRLS}$ provides an accurate, data-efficient, and largely automated route to shock-aware reduced order models for high-speed aerodynamic design.

---

## 143. Learning Cell-Aware Hierarchical Multi-Modal Representations for Robust Molecular Modeling

**论文链接:** [http://arxiv.org/abs/2511.21120v1](http://arxiv.org/abs/2511.21120v1)

**作者:** Mengran Li, Zelin Zang, Wenbin Xing, Junzhou Chen, Ronghui Zhang, Jiebo Luo, Stan Z. Li

**发布时间:** 2025-11-26

**备注:** Accepted to AAAI 2026 (Oral)

### GPT解析

### 总结

本研究提出了CHMR框架，通过联合建模分子和细胞响应间的局部-全局依赖关系，并利用树结构向量量化模块捕获潜在生物层次结构，显著提升了分子属性预测的准确性。

### 背景

理解化学扰动在生物系统中的传播机制对分子属性预测至关重要。现有方法主要关注化学结构，忽视了细胞反应（如形态和基因表达）的关键作用，且当前细胞感知方法存在外部生物数据模态不完整和对分子-细胞-基因组层次依赖关系建模不足两大局限。

### 目的

开发一个能够克服现有细胞感知方法局限性，同时建模分子和细胞响应间关系并捕获生物层次结构的框架，以提高分子属性预测的准确性和可靠性。

### 方法

提出CHMR（Cell-aware Hierarchical Multi-modal Representations）框架，通过树结构向量量化模块联合建模分子和细胞响应间的局部-全局依赖关系，捕获潜在生物层次结构。

### 主要发现

在9个公共基准测试的728个任务上，CHMR超越最先进基线方法，分类任务平均提高3.6%，回归任务平均提高17.2%，证明了层次感知、多模态学习的优势。

### 结论

层次感知、多模态学习对于构建可靠且具有生物学基础的分子表示具有显著优势，CHMR为整合生物医学建模提供了可推广的框架。

### 翻译

理解化学扰动如何在生物系统中传播对于稳健的分子属性预测至关重要。虽然大多数现有方法仅关注化学结构本身，但最近的进展强调了细胞反应（如形态和基因表达）在塑造药物效应中的关键作用。然而，当前细胞感知方法面临两个关键限制：(1)外部生物数据中的模态不完整性，以及(2)对分子、细胞和基因组层次间依赖关系的建模不足。我们提出了CHMR（Cell-aware Hierarchical Multi-modal Representations），一个稳健的框架，该框架联合建模分子和细胞响应之间的局部-全局依赖关系，并通过新颖的树结构向量量化模块捕获潜在生物层次结构。在9个涵盖728个任务的公共基准测试上评估，CHMR优于最先进的基线方法，在分类任务上平均提高3.6%，在回归任务上平均提高17.2%。这些结果证明了层次感知、多模态学习对于可靠且具有生物学基础的分子表示的优势，为整合生物医学建模提供了一个可推广的框架。代码位于https://github.com/limengran98/CHMR。


### 论文摘要

Understanding how chemical perturbations propagate through biological systems is essential for robust molecular property prediction. While most existing methods focus on chemical structures alone, recent advances highlight the crucial role of cellular responses such as morphology and gene expression in shaping drug effects. However, current cell-aware approaches face two key limitations: (1) modality incompleteness in external biological data, and (2) insufficient modeling of hierarchical dependencies across molecular, cellular, and genomic levels. We propose CHMR (Cell-aware Hierarchical Multi-modal Representations), a robust framework that jointly models local-global dependencies between molecules and cellular responses and captures latent biological hierarchies via a novel tree-structured vector quantization module. Evaluated on nine public benchmarks spanning 728 tasks, CHMR outperforms state-of-the-art baselines, yielding average improvements of 3.6% on classification and 17.2% on regression tasks. These results demonstrate the advantage of hierarchy-aware, multimodal learning for reliable and biologically grounded molecular representations, offering a generalizable framework for integrative biomedical modeling. The code is in https://github.com/limengran98/CHMR.

---

## 144. Generative Early Stage Ranking

**论文链接:** [http://arxiv.org/abs/2511.21095v1](http://arxiv.org/abs/2511.21095v1)

**作者:** Juhee Hong, Meng Liu, Shengzhi Wang, Xiaoheng Mao, Huihui Cheng, Leon Gao, Christopher Leung, Jin Zhou, Chandra Mouli Sekar, Zhao Zhu, Ruochen Liu, Tuan Trieu, Dawei Sun, Jeet Kanjani, Rui Li, Jing Qian, Xuan Cao, Minjie Fan, Mingze Gao

**发布时间:** 2025-11-26

### GPT解析

### 总结

本文提出生成式早期阶段排序(GESR)范式，通过引入注意力混合(MoA)模块解决传统ESR系统在捕捉细粒度用户-项目亲和力方面的局限性，并结合多种优化技术确保效率和低延迟。

### 背景

大规模推荐系统通常采用多级级联排序系统范式来平衡效果和效率。早期阶段排序(ESR)系统采用'用户-项目解耦'方法，独立学习的用户和项目表示仅在最后一层组合，虽然效率高但效果有限。

### 目的

解决传统ESR系统难以捕捉细粒度用户-项目亲和力和交叉信号的问题，提高推荐系统的效果。

### 方法

提出GESR范式，引入注意力混合(MoA)模块，包含硬匹配注意力(HMA)模块、目标感知自注意力模块和交叉注意力模块；通过多逻辑门参数化(MLPG)模块优化注意力编码；同时采用一系列优化技术确保效率和低延迟。

### 主要发现

GESR范式在顶级指标、参与度和消费任务方面显示出显著改进，通过离线和在线实验得到验证。

### 结论

GESR范式成功解决了传统ESR系统的局限性，实现了在如此规模下首次完整的目标感知注意力序列建模部署。

### 翻译

大规模推荐系统通常采用多级级联排序系统范式来平衡效果和效率。早期阶段排序(ESR)系统采用'用户-项目解耦'方法，其中独立学习的用户和项目表示仅在最后一层组合。虽然这种方法效率高，但在效果上有限，因为它难以捕捉细粒度的用户-项目亲和力和交叉信号。为了解决这些问题，我们提出了生成式早期阶段排序(GESR)范式，引入了注意力混合(MoA)模块，利用多种注意力机制来缩小效果差距：硬匹配注意力(HMA)模块通过计算用户和项目特征之间的原始匹配计数来编码明确的交叉信号；目标感知自注意力模块基于项目生成目标感知的用户表示，实现更个性化的学习；交叉注意力模块促进用户-项目特征之间更早、更丰富的交互。MoA的专门注意力编码在最后一层通过多逻辑门参数化(MLPG)模块进一步优化，该模块通过门控整合新学习的嵌入，并生成与主要逻辑融合的次要逻辑。为了解决效率和延迟挑战，我们引入了一系列优化技术，从充分利用最新硬件的自定义内核到由缓存机制支持的高效服务解决方案。提出的GESR范式在顶级指标、参与度和消费任务方面显示出显著改进，这已通过离线和在线实验得到验证。据我们所知，这是首次在如此规模的ESR阶段成功部署完整的目标感知注意力序列建模。


### 论文摘要

Large-scale recommendations commonly adopt a multi-stage cascading ranking system paradigm to balance effectiveness and efficiency. Early Stage Ranking (ESR) systems utilize the "user-item decoupling" approach, where independently learned user and item representations are only combined at the final layer. While efficient, this design is limited in effectiveness, as it struggles to capture fine-grained user-item affinities and cross-signals. To address these, we propose the Generative Early Stage Ranking (GESR) paradigm, introducing the Mixture of Attention (MoA) module which leverages diverse attention mechanisms to bridge the effectiveness gap: the Hard Matching Attention (HMA) module encodes explicit cross-signals by computing raw match counts between user and item features; the Target-Aware Self Attention module generates target-aware user representations conditioned on the item, enabling more personalized learning; and the Cross Attention modules facilitate early and more enriched interactions between user-item features. MoA's specialized attention encodings are further refined in the final layer through a Multi-Logit Parameterized Gating (MLPG) module, which integrates the newly learned embeddings via gating and produces secondary logits that are fused with the primary logit. To address the efficiency and latency challenges, we have introduced a comprehensive suite of optimization techniques. These span from custom kernels that maximize the capabilities of the latest hardware to efficient serving solutions powered by caching mechanisms. The proposed GESR paradigm has shown substantial improvements in topline metrics, engagement, and consumption tasks, as validated by both offline and online experiments. To the best of our knowledge, this marks the first successful deployment of full target-aware attention sequence modeling within an ESR stage at such a scale.

---

## 145. Detecting absence: A dedicated prediction-error signal emerging in the auditory thalamus

**论文链接:** [http://arxiv.org/abs/2511.21605v1](http://arxiv.org/abs/2511.21605v1)

**作者:** Alejandro Tabas, Heike Sönnichsen, Sandeep Kaur, Marco Meixner, Katharina von Kriegstein

**发布时间:** 2025-11-26

### GPT解析

### 总结

本研究探讨了大脑如何检测感官刺激的存在与缺席，提出了一种名为'缺席预测误差'的神经机制，并通过fMRI实验在人类听觉系统中验证了这一假设。

### 背景

生物体不能仅依靠感官信号进行感知，因为这些信号是嘈杂和模糊的。大脑使用先验知识或信念将感官信号转化为稳定的感知。当前理论认为感知信念是刺激特征上的概率分布，通过特征预测误差（预期和观察到的特征值之间的不匹配）来更新。

### 目的

探究大脑如何更新关于刺激存在或缺席的信念，并提出并验证一种专门的神经机制来检测感官刺激的缺席。

### 方法

使用人类听觉系统作为感官处理的模型，开发了一种实验范式来测试'缺席预测误差'的假设。通过fMRI技术观察大脑对刺激存在和缺席的反应。

### 主要发现

fMRI结果显示，缺席预测误差在听觉丘脑和皮层中被编码，表明缺席在皮层下感觉通路中被明确表示。而特征预测误差在听觉中脑中已被编码，但缺席预测误差没有，这表明与缺席相关的误差信号由不同的神经回路支持。

### 结论

这些结果确定了检测感官缺席的神经机制。在精神病等情况下，这种机制可能会受到干扰，因为关于缺席和存在的预测受损。

### 翻译

大脑如何知道外界有什么和没有什么？生物体不能仅依靠感官信号进行感知，因为这些信号是嘈杂和模糊的。为了将感官信号转化为稳定的感知，大脑使用其先验知识或信念。当前理论将感知信念描述为刺激特征上的概率分布，由均值和方差总结。信念通过特征预测误差（预期和观察到的特征值之间的不匹配）来更新。这一框架解释了大脑如何编码刺激特征的意外变化（例如，更高或更低的音调，更强或更弱的运动）。然而，大脑如何更新关于刺激存在或缺席的信念尚不清楚。我们提出，缺席检测依赖于一种专门的预测误差形式，专门用于减少对刺激发生的信念。我们称这种信号为缺席预测误差。使用人类听觉系统作为感官处理的模型，我们开发了一种设计用于测试这一假设的范式。fMRI结果表明，缺席预测误差在听觉丘脑和皮层中被编码，表明缺席在皮层下感觉通路中被明确表示。此外，虽然特征预测误差已经在听觉中脑中被编码，但缺席预测误差没有被编码，这表明与缺席相关的误差信号由不同的回路支持。这些结果确定了检测感官缺席的神经机制。在精神病等情况下，关于缺席和存在的预测可能会受到损害，从而破坏这种机制。


### 论文摘要

How does the brain know what is out there and what is not? Living organisms cannot rely solely on sensory signals for perception because they are noisy and ambiguous. To transform sensory signals into stable percepts, the brain uses its prior knowledge or beliefs. Current theories describe perceptual beliefs as probability distributions over the features of the stimuli, summarised by their mean and variance. Beliefs are updated by feature prediction errors: the mismatch between expected and observed feature values. This framework explains how the brain encodes unexpected changes in stimulus features (e.g., higher or lower pitch, stronger or weaker motion). How the brain updates beliefs about a stimulus' presence or absence is, however, unclear.   We propose that the detection of absence relies on a distinct form of prediction error dedicated to reducing the beliefs on stimulus occurrence. We call this signal absence prediction error. Using the human auditory system as a model for sensory processing, we developed a paradigm designed to test this hypothesis. fMRI results showed that absence prediction error is encoded in the auditory thalamus and cortex, indicating that absence is explicitly represented in subcortical sensory pathways. Moreover, while feature prediction error is already encoded in the auditory midbrain, absence prediction error was not, implying that absence-related error signals are supported by a different circuit.   These results identify a neural mechanism for the detection of sensory absence. Such mechanisms may be disrupted in conditions such as psychosis, where predictions about absence and presence are impaired.

---

## 146. Self-Transparency Failures in Expert-Persona LLMs: A Large-Scale Behavioral Audit

**论文链接:** [http://arxiv.org/abs/2511.21569v1](http://arxiv.org/abs/2511.21569v1)

**作者:** Alex Diep

**发布时间:** 2025-11-26

### GPT解析

### 总结

本研究探讨了语言模型在不同专业身份下的自我透明度问题，发现模型在不同领域表现出显著的不一致性，透明度更多反映训练因素而非模型规模。

### 背景

当语言模型无法在专业环境中可靠披露其AI身份时，用户无法信任其能力边界。在高风险领域，虚假专业知识可能导致用户受到伤害。

### 目的

检验在高风险领域中分配专业身份的模型的自我透明度表现，了解模型披露其AI身份的可靠性。

### 方法

采用共同花园设计，对16个不同参数规模(40亿至6710亿)的开源模型进行了19,200次试验审计。

### 主要发现

模型表现出明显的领域特定不一致性：金融顾问身份引发30.8%的初始披露，而神经外科医生身份仅引发3.5%。披露率范围从2.8%到73.6%，其中140亿参数模型达到61.4%，而700亿参数模型仅4.1%。模型身份比参数数量更能预测行为，推理优化在某些模型中主动抑制了自我透明度。

### 结论

透明度反映的是训练因素而非模型规模。组织不能假设安全属性会自动转移到部署环境中，需要专门的行为设计和经验验证。

### 翻译

如果语言模型无法在专业环境中可靠披露其AI身份，用户就无法信任其能力边界。本研究检查了在错误专业知识可能危害用户的高风险领域中分配专业身份的模型的自我透明度。采用共同花园设计，对16个开源模型(40亿至6710亿参数)进行了19,200次试验审计。模型表现出明显的领域特定不一致性：金融顾问身份引发30.8%的初始披露，而神经外科医生身份仅引发3.5%。这为'反向盖尔曼记忆缺失'效应创造了前提条件，即在某些领域的透明度会导致用户过度信任到披露失败的情境。披露率从2.8%到73.6%不等，其中140亿参数模型达到61.4%，而700亿参数模型仅产生4.1%。模型身份比参数数量更能预测行为。推理优化在某些模型中主动抑制了自我透明度，推理变体比基础版本披露率低48.4%。使用Rogan-Gladen校正的贝叶斯验证确认了对测量误差的稳健性。这些发现表明透明度反映的是训练因素而非规模。组织不能假设安全属性会转移到部署环境中，需要专门的行为设计和经验验证。


### 论文摘要

If a language model cannot reliably disclose its AI identity in expert contexts, users cannot trust its competence boundaries. This study examines self-transparency in models assigned professional personas within high-stakes domains where false expertise risks user harm. Using a common-garden design, sixteen open-weight models (4B--671B parameters) were audited across 19,200 trials. Models exhibited sharp domain-specific inconsistency: a Financial Advisor persona elicited 30.8% disclosure initially, while a Neurosurgeon persona elicited only 3.5%. This creates preconditions for a "Reverse Gell-Mann Amnesia" effect, where transparency in some domains leads users to overgeneralize trust to contexts where disclosure fails. Disclosure ranged from 2.8% to 73.6%, with a 14B model reaching 61.4% while a 70B produced just 4.1%. Model identity predicted behavior better than parameter count ($ΔR_{adj}^{2} = 0.359$ vs 0.018). Reasoning optimization actively suppressed self-transparency in some models, with reasoning variants showing up to 48.4% lower disclosure than base counterparts. Bayesian validation with Rogan--Gladen correction confirmed robustness to measurement error ($κ= 0.908$). These findings demonstrate transparency reflects training factors rather than scale. Organizations cannot assume safety properties transfer to deployment contexts, requiring deliberate behavior design and empirical verification.

---

## 147. Modeling the Effect of Data Redundancy on Speedup in MLFMA Near-Field Computation

**论文链接:** [http://arxiv.org/abs/2511.21535v1](http://arxiv.org/abs/2511.21535v1)

**作者:** Morteza Sadeghi

**发布时间:** 2025-11-26

### GPT解析

### 总结

这项研究针对MLFMA算法中近场(P2P)算子在GPU上的性能瓶颈问题，通过引入数据冗余来改善空间局部性，减少内存访问分散。研究提出了基于局部性指标的分析模型，并在两个MLFMA应用中验证了该方法。结果显示内核速度up可提高7倍，但由于数据重组开销增加，端到端应用速度提升限制在1.04倍。该技术可轻松集成到现有实现中，且在局部性收益超过数据移动成本时能有效提升GPU性能。

### 背景

在多级快速多极算法(MLFMA)中，近场(P2P)算子在GPU上表现为性能瓶颈，主要原因是内存局部性差，这限制了GPU在该算法中的应用效率。

### 目的

改善MLFMA中P2P算子在GPU上的性能，通过提高内存局部性来减少内存访问分散，从而解决性能瓶颈问题。

### 方法

1. 引入数据冗余来改善空间局部性，减少内存访问分散；2. 提出一个基于局部性指标的分析模型，该模型结合数据量和访问分散度来预测加速趋势；3. 在两个基于MLFMA的应用中验证该方法：具有规则结构的电磁求解器(DBIM-MLFMA)和具有不规则粒子分布的恒星动力学代码(PhotoNs-2.0)。

### 主要发现

1. 由于改进的缓存行为，内核速度up可提高7倍；2. 数据量的增加提高了数据重组的开销，限制了端到端应用速度up为1.04倍；3. 该模型无法精确预测绝对加速比，但能可靠地捕捉不同问题规模和密度下的性能趋势；4. 该技术可以以最小的代码更改注入到现有实现中。

### 结论

数据冗余可以增强P2P算子在GPU上的性能，前提是局部性收益超过数据移动成本。这项工作证明了通过有意识地引入数据冗余来提高GPU性能的有效性，特别是在内存访问模式分散的情况下。

### 翻译

多级快速多极算法(MLFMA)中的近场(P2P)算子在GPU上由于内存局部性差而成为性能瓶颈。这项工作通过引入数据冗余来减少内存访问分散，从而改善空间局部性。为了验证结果，我们提出了一个基于局部性指标的分析模型，该指标结合数据量和访问分散度来预测加速趋势，无需硬件特定的性能分析。该方法在两个基于MLFMA的应用中得到了验证：具有规则结构的电磁求解器(DBIM-MLFMA)和具有不规则粒子分布的恒星动力学代码(PhotoNs-2.0)。结果显示，由于改进的缓存行为，内核速度up可达7倍。然而，数据量的增加提高了数据重组的开销，将端到端应用速度up限制在1.04倍。虽然该模型无法精确预测绝对加速比，但它能可靠地捕捉不同问题规模和密度下的性能趋势。该技术可以以最小的代码更改注入到现有实现中。这项工作表明，只要局部性收益超过数据移动成本，数据冗余就可以提高P2P算子在GPU上的性能。


### 论文摘要

The near-field (P2P) operator in the Multilevel Fast Multipole Algorithm (MLFMA) is a performance bottleneck on GPUs due to poor memory locality. This work introduces data redundancy to improve spatial locality by reducing memory access dispersion. For validation of results, we propose an analytical model based on a Locality metric that combines data volume and access dispersion to predict speedup trends without hardware-specific profiling. The approach is validated on two MLFMA-based applications: an electromagnetic solver (DBIM-MLFMA) with regular structure, and a stellar dynamics code (PhotoNs-2.0) with irregular particle distribution. Results show up to 7X kernel speedup due to improved cache behavior. However, increased data volume raises overheads in data restructuring, limiting end-to-end application speedup to 1.04X. While the model cannot precisely predict absolute speedups, it reliably captures performance trends across different problem sizes and densities. The technique is injectable into existing implementations with minimal code changes. This work demonstrates that data redundancy can enhance GPU performance for P2P operator, provided locality gains outweigh data movement costs.

---

## 148. Quantum theory of electrically levitated nanoparticle-ion systems: Motional dynamics and sympathetic cooling

**论文链接:** [http://arxiv.org/abs/2511.21495v1](http://arxiv.org/abs/2511.21495v1)

**作者:** Saurabh Gupta, Dmitry S. Bykov, Tracy E. Northup, Carlos Gonzalez-Ballestero

**发布时间:** 2025-11-26

**备注:** 18 pages, 6 figures

### GPT解析

### 总结

研究纳米颗粒与离子团在双频线性保罗陷阱中共陷时质心运动的量子耦合动力学理论

### 背景

在双频线性保罗陷阱中共陷纳米颗粒和离子团的量子系统研究

### 目的

建立离子-纳米颗粒系统的量子耦合动力学理论，探索离子辅助的非高斯运动态制备

### 方法

推导纳米颗粒和离子运动频率及经典轨迹的解析表达式，建立量子主方程，量化协同冷却效果，并将分析扩展到多离子系统

### 主要发现

1) 即使在无运动反馈和存在微运动情况下也能实现亚开尔文温度冷却；2) 冷却速率随离子数量N线性增加；3) 当前实验平台可实现纳米颗粒运动冷却至毫开尔文十分之几的温度

### 结论

建立了探索离子辅助悬浮纳米颗粒非高斯运动态制备所需的理论工具箱

### 翻译

我们发展了描述在双频线性保罗陷阱中共陷的纳米颗粒和离子团质心运动量子耦合动力学的理论。我们首先推导了纳米颗粒和离子运动频率及经典轨迹的解析表达式。然后推导了离子-纳米颗粒系统的量子主方程，并量化了通过库仑耦合到连续多普勒冷却离子的纳米颗粒运动的协同冷却效果。我们预测即使在不存在运动反馈和存在微运动的情况下，在现有实验中也能实现亚开尔文温度的运动冷却。我们将分析扩展到N个离子组成的离子团，预测冷却速率随N线性增加，并在当前实验平台中实现纳米颗粒运动冷却至毫开尔文十分之几的温度。我们的工作建立了探索离子辅助悬浮纳米颗粒非高斯运动态制备所需的理论工具箱。


### 论文摘要

We develop the theory describing the quantum coupled dynamics of the center-of-mass motion of a nanoparticle and an ensemble of ions co-trapped in a dual-frequency linear Paul trap. We first derive analytical expressions for the motional frequencies and classical trajectories of both nanoparticle and ions. We then derive a quantum master equation for the ion-nanoparticle system and quantify the sympathetic cooling of the nanoparticle motion enabled by its Coulomb coupling to a continuously Doppler-cooled ion. We predict that motional cooling down to sub-kelvin temperatures is achievable in state-of-the-art experiments even in the absence of motional feedback and in the presence of micromotion. We then extend our analysis to an ensemble of $N$ ions, predicting a linear increase of the cooling rate as a function of $N$ and motional cooling of the nanoparticle down to tenths of millikelvin in current experimental platforms. Our work establishes the theoretical toolbox needed to explore the ion-assisted preparation of non-Gaussian motional states of levitated nanoparticles.

---

## 149. Scaling limits of critical FK-decorated random planar maps with $q=4$

**论文链接:** [http://arxiv.org/abs/2511.21480v1](http://arxiv.org/abs/2511.21480v1)

**作者:** William Da Silva, Xingjian Hu, Ellen Powell, Mo Dick Wong

**发布时间:** 2025-11-26

**备注:** 74 pages, 9 figures, comments welcome!

### GPT解析

### 总结

该研究建立了FK(4)加权的平面地图在临界情况下的第一个标度极限，解决了Sheffield开创性工作以来长期悬而未决的问题。研究证明在临界情况下，相关的汉堡计数和差异满足特定收敛关系，并与临界树交配相匹配，建立了平面图向CLE₄和临界刘维量子引力收敛的第一个严格证明。

### 背景

Sheffield在其开创性工作(arXiv:1108.2241)中通过著名的汉堡-奶酪汉堡双射证明了q<4时的标度极限，这开启了刘维量子引力的皮亚诺球面(树交配)方法。然而，在临界情况q=4下的标度极限问题一直未能解决。

### 目的

建立FK(4)加权的平面地图在临界情况下的标度极限，确定相关的汉堡计数和差异的极限行为，并将极限过程与临界树交配匹配，建立平面图向CLE₄和临界刘维量子引力收敛的严格证明。

### 方法

研究基于一种新方法，通过揭示模型与三角剖分上的(双色)完全填充环-O(2)模型的对应关系，展示了模型的完全可解性质，并得到了与共形场理论预测一致的临界几何指数。

### 主要发现

1. 在临界情况下，(S_{⌊nt⌋}/√n, (log(n)/(2π√n))D_{⌊nt⌋})依分布收敛到两个独立的双边布朗运动。2. 这是第一个关于正确差异标度因子的猜想。3. 将极限过程与临界树交配匹配，建立了第一个严格的平面图向CLE₄和临界刘维量子引力在皮亚诺球面意义上的收敛。4. 临界几何指数与共形场理论的预测相匹配。

### 结论

该研究解决了FK(q)加权的平面地图在临界情况q=4下的标度极限问题，建立了与刘维量子引力的严格联系，并通过新方法揭示了模型的完全可解性质，为后续研究奠定了基础。

### 翻译

我们建立了FK(q)加权的平面地图在临界情况q=4下的第一个标度极限，解决了自Sheffield的开创性工作arXiv:1108.2241以来一直悬而未决的问题。在那项工作中，Sheffield通过著名的汉堡-奶酪汉堡双射证明了q<4时的标度极限，这开启了刘维量子引力的皮亚诺球面(树交配)方法。我们证明，在临界情况下，相关的汉堡计数S和差异D满足(S_{⌊nt⌋}/√n, (log(n)/(2π√n))D_{⌊nt⌋})依分布收敛到(B¹_t, B²_t)，其中B¹和B²是独立的两个方向的布朗运动。据我们所知，之前从未有人提出关于正确差异标度因子的猜想。将极限过程与临界树交配匹配，我们建立了第一个严格的平面图向CLE₄和临界刘维量子引力在皮亚诺球面意义上的收敛。我们的证明基于一种新方法，通过揭示模型与三角剖分上的(双色)完全填充环-O(2)模型的对应关系，展示了模型的完全可解性质，并得到了与共形场理论预测一致的临界几何指数。


### 论文摘要

We establish the first scaling limit for FK($q$)-weighted planar maps in the critical case $q=4$, resolving a problem that has remained open since Sheffield's seminal work arXiv:1108.2241. In that work, Sheffield proved a scaling limit for $q<4$ via the celebrated hamburger-cheeseburger bijection, which initiated the peanosphere (mating-of-trees) approach to Liouville quantum gravity. We prove that, at criticality, the associated burger count $\mathcal{S}$ and discrepancy $\mathcal{D}$ satisfy \[ \left(\frac{\mathcal{S}_{\lfloor nt \rfloor}}{\sqrt{n}}, \frac{\log(n)}{{2π}\sqrt{n}} \mathcal{D}_{\lfloor nt \rfloor}\right)_{t\in\mathbb{R}} \stackrel{\text{d}}{\longrightarrow} (B^1_t, B^2_{t})_{t\in\mathbb{R}}, \] where $B^1$ and $B^2$ are independent two-sided Brownian motions. To the best of our knowledge, no conjecture for the correct discrepancy scaling factor had previously been formulated. Matching the limiting process with the critical mating of trees arXiv:2109.00275, we establish the first rigorous planar map convergence towards CLE$_4$ and critical ($γ=2$) Liouville quantum gravity, in the peanosphere sense. Our proof is based on a novel approach that reveals the exactly solvable nature of the model through a correspondence with the (bicoloured) fully packed loop-$O(2)$ model on triangulations, and yields critical geometric exponents matching the predictions of conformal field theory.

---

## 150. Enabling the bulk photovoltaic effect in centrosymmetric materials through an external electric field

**论文链接:** [http://arxiv.org/abs/2511.21458v1](http://arxiv.org/abs/2511.21458v1)

**作者:** Guilherme J. Inacio, Juan José Esteve-Paredes, Maurício F. C. Martins Quintela, Wendel S. Paz, Juan José Palacios

**发布时间:** 2025-11-26

**备注:** 11 pages, 6 figures

### GPT解析

### 总结

研究开发了一种通过电场调谐二维半导体非线性光响应的实用方法，通过将静态面外电场纳入电子基态作为栅极偏压，实现了对轨道杂化和色散关系的调控。

### 背景

二维半导体的非线性光响应调谐研究，特别是如何通过外部电场控制其光电特性。

### 目的

开发一种实际方法，通过电场调谐二维半导体的非线性光响应，超越传统微扰处理的限制。

### 方法

通过位置矩阵元将静态面外电场引入Wannier插值哈密顿量，在独立粒子近似下评估中心对称和非中心对称层状系统的二阶位移电导率。

### 主要发现

1) 在中心对称双层MoS₂中观察到有限位移电流；2) 极性结构中的内在响应具有可调性；3) 位移电导率在小场下线性上升，在高强度时饱和；4) 场修饰电导率与三阶光学响应相关，揭示了场诱导非线性的统一图像。

### 结论

Wannier哈密顿量的场修饰是模拟和预测层状材料非线性光电流的有效实用方法。

### 翻译

我们开发了一种实际方法，通过在光学激发前将静态面外电场明确地纳入电子基态作为栅极偏压，来调谐二维半导体的非线性光响应。该方法通过位置矩阵元将场引入Wannier插值哈密顿量，使栅极偏压能够超越微扰处理来修改轨道杂化和色散关系。在独立粒子近似下，我们评估了中心对称和非中心对称层状系统的二阶（位移）电导率。应用于MoS₂时，该方法捕捉到中心对称双层中有限位移电流的出现以及极性结构中内在响应的可调性。位移电导率在小场下线性上升，在高强度时饱和，这反映了位移矢量的增长与共振跃迁远离高对称谷时减弱的能带间竞争之间的平衡。场修饰电导率的泰勒展开将这种行为与三阶光学响应联系起来，揭示了场诱导非线性的统一图像。这些结果建立了Wannier哈密顿量的场修饰作为模拟和预测层状材料非线性光电流的实际途径。


### 论文摘要

We develop a practical approach to electrically tuning the nonlinear photoresponse of two-dimensional semiconductors by explicitly incorporating a static out-of-plane electric field into the electronic ground state prior to optical excitation, as a gate bias. The method is implemented by dressing a Wannier-interpolated Hamiltonian with the field through its position matrix elements, which allows the gate bias to modify orbital hybridization and band dispersion beyond perturbative treatments. Within the independent-particle approximation, the resulting second-order (shift) conductivity is evaluated for both centrosymmetric and non-centrosymmetric layered systems. Applied to MoS$_2$, the approach captures the emergence of a finite shift current in centrosymmetric bilayers and the tunability of intrinsic responses in polar structures. The shift conductivity rises linearly at small fields and saturates at higher intensities, reflecting the competition between the growing shift vector and the weakening interband coupling as resonant transitions move away from high-symmetry valleys. A Taylor expansion of the field-dressed conductivity connects this behavior to the third-order optical response, revealing a unified picture of field-induced nonlinearities. These results establish field dressing of Wannier Hamiltonians as a practical route to model and predict nonlinear photocurrents in layered materials.

---

## 151. Revealing Fast Ionic Conduction in Solid Electrolytes through Machine Learning Accelerated Raman Calculations

**论文链接:** [http://arxiv.org/abs/2511.21404v1](http://arxiv.org/abs/2511.21404v1)

**作者:** Manuel Grumet, Takeru Miyagawa, Olivier Pittet, Paolo Pegolo, Karin S. Thalmann, Waldemar Kaiser, David A. Egger

**发布时间:** 2025-11-26

### GPT解析

### 总结

本研究开发了一种基于机器学习的计算方法，利用低频扩散拉曼散射作为快速离子传导的光谱特征，用于识别有前途的固体电解质。该方法克服了计算强无序材料拉曼光谱的计算障碍，实现了接近从头算的精度，并成功应用于钠离子导体。

### 背景

快速离子传导是全固态电池固体电解质的定义性特征。先前研究表明，与快速离子传输相关的类液态阳离子运动会破坏晶体对称性，从而消除拉曼选择规则。

### 目的

开发一种机器学习加速的计算管道，利用低频扩散拉曼散射作为光谱特征来识别具有快速离子传导特性的固体电解质。

### 方法

利用机器学习技术克服计算强无序材料在有限温度下拉曼光谱计算的巨大计算障碍，开发计算管道来识别具有快速离子传导特性的固体电解质，并在钠离子导体上进行验证。

### 主要发现

成功实现了接近从头算的精度，展示了该方法对钠离子导体的预测能力，揭示了类液态离子传导的清晰拉曼特征。

### 结论

机器学习能够连接原子模拟和实验可观测性，实现数据高效地发现快速离子导体。

### 翻译

快速离子传导是全固态电池固体电解质的定义性特征。先前研究表明，与快速离子传输相关的类液态阳离子运动会破坏晶体对称性，从而消除拉曼选择规则。在此，我们利用由此产生的低频扩散拉曼散射作为快速离子传导的光谱特征，并开发了一个机器学习加速的计算管道来基于这一特征识别有前途的固体电解质。通过克服计算强无序材料在有限温度下拉曼光谱计算的巨大计算障碍，我们实现了接近从头算的精度，并展示了该方法对钠离子导体的预测能力，揭示了类液态离子传导的清晰拉曼特征。这项工作强调了机器学习如何连接原子模拟和实验可观测性，实现数据高效地发现快速离子导体。


### 论文摘要

Fast ionic conduction is a defining property of solid electrolytes for all-solid-state batteries. Previous studies have suggested that liquid-like cation motion associated with fast ionic transport can disrupt crystalline symmetry, thereby lifting Raman selection rules. Here, we exploit the resulting low-frequency, diffusive Raman scattering as a spectral signature of fast ionic conduction and develop a machine learning-accelerated computational pipeline to identify promising solid electrolytes based on this feature. By overcoming the steep computational barriers to calculating Raman spectra of strongly disordered materials at finite temperatures, we achieve near-ab initio accuracy and demonstrate the predictive power of our approach for sodium-ion conductors, revealing clear Raman signatures of liquid-like ion conduction. This work highlights how machine learning can bridge atomistic simulations and experimental observables, enabling data-efficient discovery of fast-ion conductors.

---

## 152. Conditional Generative Modeling of Stochastic LTI Systems: A Behavioral Approach

**论文链接:** [http://arxiv.org/abs/2511.21219v1](http://arxiv.org/abs/2511.21219v1)

**作者:** Jiayun Li, Yilin Mo

**发布时间:** 2025-11-26

### GPT解析

### 总结

本文提出了一种用于线性时不变(LTI)随机系统的数据驱动模型，称为行为条件生成模型(CGM)。该模型通过从给定过去输入-输出和未来输入的条件概率分布中采样来运行，完全基于当前轨迹和预先收集的数据，无需显式识别系统参数。研究证明了随着轨迹库大小增加，CGM生成样本分布的收敛性，并表明CGM的渐近分布与卡尔曼滤波器获得的真实后验分布之间的差距随过去样本长度呈指数减小。最后，将此生成模型集成到LTI随机系统的预测控制器中，数值结果验证了理论界限并证明了控制器的有效性。

### 背景

线性时不变(LTI)随机系统的建模与控制

### 目的

提出一种基于行为方式的条件生成模型(CGM)，用于LTI随机系统的建模与控制

### 方法

通过从给定过去输入-输出和未来输入的条件概率分布中采样来构建模型；模型以完全行为方式运行，仅依赖当前轨迹和预先收集的输入-输出数据；无需显式识别系统参数；证明CGM的收敛性；将生成模型集成到预测控制器中

### 主要发现

随着轨迹库大小增加，CGM生成的样本分布收敛；CGM的渐近分布与卡尔曼滤波器获得的真实后验分布之间的差距随过去样本长度呈指数减小

### 结论

数值结果验证了推导的界限，证明了配备所提出的行为CGM的控制器是有效的

### 翻译

本文提出了一种用于线性时不变(LTI)随机系统的数据驱动模型，通过从给定过去输入-输出和未来输入的条件概率分布中采样来构建。该模型完全以行为方式运行，仅依赖当前轨迹和预先收集的输入-输出数据，无需显式识别系统参数。我们将此模型称为行为条件生成模型(CGM)。我们证明了随着轨迹库大小增加，CGM生成的样本分布的收敛性，并对收敛速率进行了明确的表征。此外，我们证明了所提出的CGM的渐近分布与利用所有系统参数和所有历史数据获得的卡尔曼滤波器的真实后验分布之间的差距，随着过去样本长度的增加呈指数减小。最后，我们将此生成模型集成到LTI随机系统的预测控制器中。数值结果验证了推导的界限，并证明了配备所提出的行为CGM的控制器的有效性。


### 论文摘要

This paper presents a data-driven model for Linear Time-Invariant (LTI) stochastic systems by sampling from the conditional probability distribution of future outputs given past input-outputs and future inputs. It operates in a fully behavioral manner, relying solely on the current trajectory and pre-collected input-output data, without requiring explicit identification of system parameters. We refer to this model as a behavioral Conditional Generative Model (CGM). We prove the convergence of the distribution of samples generated by the CGM as the size of the trajectory library increases, with an explicit characterization of the convergence rate. Furthermore, we demonstrate that the gap between the asymptotic distribution of the proposed CGM and the true posterior distribution obtained by Kalman filter, which leverages the knowledge of all system parameters and all historical data, decreases exponentially with respect to the length of past samples. Finally, we integrate this generative model into predictive controllers for stochastic LTI systems. Numerical results verify the derived bounds and demonstrate the effectiveness of the controller equipped with the proposed behavioral CGM.

---

## 153. Vortex-Enhanced Zitterbewegung in Relativistic Electron Wave Packets

**论文链接:** [http://arxiv.org/abs/2511.21142v1](http://arxiv.org/abs/2511.21142v1)

**作者:** Zhongze Guo, Bei Xu, Qiang Gu

**发布时间:** 2025-11-26

**备注:** 5 pages, no figures

### GPT解析

### 总结

本文通过构建相对论性涡旋电子波包，成功放大了狄拉克方程预测的Zitterbewegung (ZBW)颤抖运动的振幅，使其可被观测，并统一了高斯和贝塞尔-高斯模型。

### 背景

ZBW是狄拉克方程预测的一种颤抖运动，但由于其尺度低于康普顿波长，在自由电子中长期无法观测到。

### 目的

构建一种能够放大ZBW振幅并保持相干性的电子波包，以实现对ZBW现象的观测。

### 方法

将相对论性涡旋电子波包精心构造为正能和负能狄拉克态的相干叠加，并推导其空间-时间动力学。

### 主要发现

引入轨道角动量为放大ZBW振幅提供了一种机制，使其远超传统高斯波包的振幅，同时保持相干性。

### 结论

所得到的相对论性涡旋态将高斯和贝塞尔-高斯模型统一在一个单一框架内，为在结构化电子波包中观测相对论量子动力学开辟了新的可能性。

### 翻译

Zitterbewegung (ZBW)是狄拉克方程预测的一种颤抖运动，由于它的尺度低于康普顿波长，在自由电子中长期无法观测。我们精心构建了一个相对论性涡旋电子波包，作为正能和负能狄拉克态的相干叠加，并推导了它们的空间-时间动力学。我们的分析表明，引入轨道角动量为放大ZBW振幅提供了一种机制，使其远超传统高斯波包，同时保持相干性。所得到的相对论性涡旋态将高斯和贝塞尔-高斯模型统一在一个单一框架内，为在结构化电子波包中观测相对论量子动力学开辟了新的可能性。


### 论文摘要

Zitterbewegung (ZBW), the trembling motion predicted by the Dirac equation, has long remained unobservable in free electrons due to its sub-Compton scale. We elaborately construct a relativistic vortex electron wave packet as a coherent superposition of both positive- and negative-energy Dirac states and derive their space-time dynamics. Our analysis demonstrates that introducing orbital angular momentum provides a mechanism for amplifying the ZBW amplitude far beyond that of conventional Gaussian packets, while maintaining coherence. The resulting relativistic vortex states unify Gaussian and Bessel-Gaussian models within a single framework and opens new possibilities for observing relativistic quantum dynamics in structured electron wave packets.

---

## 154. Cosmological tensions in Proca-Nuevo theory

**论文链接:** [http://arxiv.org/abs/2511.21071v1](http://arxiv.org/abs/2511.21071v1)

**作者:** Hsu-Wen Chiang, Claudia de Rham, Sebastian Garcia-Saenz, Xue Zhou

**发布时间:** 2025-11-26

**备注:** 26+11 pages

### GPT解析

### 总结

本研究探讨了(扩展的)Proca-Nuevo理论的宇宙学预测，这是一种具有稳定均匀且各向同性解的矢量-张量理论，表现为有效的暗能量流体，行为范围从冻结的类quintessence到解冻的类phantom。

### 背景

标准ΛCDM模型面临宇宙学张力问题，需要寻找替代理论来解释观测数据。

### 目的

检验Proca-Nuevo理论作为解决宇宙学张力问题的可行框架，并与标准ΛCDM模型进行比较。

### 方法

考虑一般的单参数背景模型类，确定'特殊'模型，分析观测约束，同时考虑扰动，使用最新的数据目录集包括CMB、BAO和低红移数据。

### 主要发现

单参数Proca-Nuevo模型比ΛCDM模型更优(1.5σ至2.4σ)；哈勃张力得到缓解(从5.8σ降至2.3σ或1.5σ)；矢量场显著增强有效牛顿常数，需要轻微调整以匹配观测物质功率谱。

### 结论

Proca-Nuevo理论为解决宇宙学张力问题提供了有希望的框架，其结果也与更广泛的设置相关，包括且不限于矢量-张量模型。

### 翻译

我们研究(扩展的)Proca-Nuevo理论的宇宙学预测。这种矢量-张量理论具有稳定的均匀且各向同性解，表现为有效的暗能量流体，行为范围从冻结的类quintessence到解冻的类phantom，作为检验影响标准ΛCDM模型的宇宙学张力的动机框架。虽然我们考虑的模型足够通用，可以涵盖一大类场理论，但与标量暗能量模型(类quintessence模型、动力学模型和非最小耦合模型)相比，它因包含矢量自由度而区别，这些自由度可能继承自更广义的引力理论。我们在几个方面改进了先前的工作：考虑一般的单参数背景模型类；确定所谓的'特殊'模型，并在考虑扰动的情况下分析观测约束，使用最新的数据目录集，包括最近发布的数据。我们发现，当拟合CMB和BAO数据时，单参数Proca-Nuevo模型比ΛCDM更优1.5σ，当进一步添加低红移数据时更优2.4σ。哈勃张力得到缓解，在有(和没有)BAO数据的CMB与局部测量之间，从5.8σ降至2.3σ(和1.5σ)。另一方面，我们发现矢量场通常会显著增强有效牛顿常数，因此要匹配观测到的物质功率谱需要轻微调整以抑制扰动的影响。由于在背景水平上，Proca-Nuevo与其他理论类是简并的，我们的结果也与更广泛的设置相关，包括且不限于矢量-张量模型。


### 论文摘要

We study the cosmological predictions of (extended) Proca-Nuevo theory. This vector-tensor theory enjoys stable homogeneous and isotropic solutions characterized by an effective dark energy fluid, with behavior that ranges from freezing quintessential to thawing phantom-like, serving as a motivated framework to scrutinize the cosmological tensions that affect the standard $Λ$CDM model. While the model we consider is sufficiently generic to encompass a large class of field theories, it distinguishes itself from scalar dark energy models (quintessential ones, kinetic ones and non-minimally coupled ones) by the presence of what would be classed as a vector degree of freedom which can be for instance inherited from more generic theories of gravity. We improve on previous work in several directions: we consider a general one-parameter class of background models; identify a so-called 'special' model and analyze observational constraints taking also into account perturbations and making use of wide up-to-date catalogs of datasets including recently released ones. We find that the one-parameter Proca-Nuevo model is preferred over $Λ$CDM at $1.5σ$ when fitting CMB and BAO data, and at $2.4σ$ when further adding low-redshift data. The Hubble tension is alleviated, dropping from $5.8σ$ to $2.3σ$ (resp. $1.5σ$) between CMB with (and resp. without) BAO data and local measurements. On the other hand, we find that the vector field generically introduces a significant enhancement of the effective Newton constant, so that matching the observed matter power spectrum requires a mild amount of tuning to suppress the impact of perturbations. Since, at the background level, Proca-Nuevo is degenerate with other classes of theories, our results are also relevant to a wider range of set-ups including and beyond vector-tensor models.

---

## 155. Zipf Distributions from Two-Stage Symbolic Processes: Stability Under Stochastic Lexical Filtering

**论文链接:** [http://arxiv.org/abs/2511.21060v1](http://arxiv.org/abs/2511.21060v1)

**作者:** Vladimir Berman

**发布时间:** 2025-11-26

**备注:** 16 pages

### GPT解析

### 总结

本研究通过几何机制而非语言学元素解释了齐普夫定律，表明齐普夫类行为源于几何约束而非交际效率。

### 背景

齐普夫定律在语言中的起源尚无定论，这一问题在多个学科领域都有广泛讨论和争议。

### 目的

解释齐普夫类行为，不使用语言学元素，而是通过几何机制来解释这一现象。

### 方法

使用完全组合词模型(FCWM)从有限字母表形成单词，生成词长的几何分布；通过相互作用的指数力产生幂律排名-频率曲线，该曲线由字母表大小和空白符号概率决定；进行模拟以支持预测。

### 主要发现

模拟结果与英语、俄语和混合类型数据相匹配；符号模型表明齐普夫类定律源于几何约束，而非交际效率。

### 结论

齐普夫类行为可以通过几何机制来解释，不需要依赖语言学元素，这类定律可能源于几何约束而非语言效率因素。

### 翻译

语言中的齐普夫定律缺乏确切的起源，在多个领域都有争议。本研究使用几何机制而非语言学元素解释了齐普夫类行为。完全组合词模型(FCWM)从有限字母表形成单词，生成词长的几何分布。相互作用的指数力产生幂律排名-频率曲线，该曲线由字母表大小和空白符号概率决定。模拟结果支持预测，与英语、俄语和混合类型数据相匹配。符号模型表明齐普夫类定律源于几何约束，而非交际效率。


### 论文摘要

Zipf's law in language lacks a definitive origin, debated across fields. This study explains Zipf-like behavior using geometric mechanisms without linguistic elements. The Full Combinatorial Word Model (FCWM) forms words from a finite alphabet, generating a geometric distribution of word lengths. Interacting exponential forces yield a power-law rank-frequency curve, determined by alphabet size and blank symbol probability. Simulations support predictions, matching English, Russian, and mixed-genre data. The symbolic model suggests Zipf-type laws arise from geometric constraints, not communicative efficiency.

---

## 156. A novel third-order accurate and stable scheme for micromagnetic simulations

**论文链接:** [http://arxiv.org/abs/2511.21047v1](http://arxiv.org/abs/2511.21047v1)

**作者:** Changjian Xie

**发布时间:** 2025-11-26

**备注:** This scheme brings a stable results and a more complicated linear system of equations

### GPT解析

### 总结

本文提出了一种针对Landau-Lifshitz-Gilbert方程的高效三阶时间精确且稳定的数值方案，解决了传统方法在精度和效率方面的局限性，通过纳米条带模拟验证了其优越性能。

### 背景

高保真数值模拟是研究微磁学中磁化动力学的基石，但传统方法在精度和效率方面存在局限性。

### 目的

开发一种新的三阶时间精确且稳定的数值方案，用于Landau-Lifshitz-Gilbert方程，以提高模拟的准确性和计算效率。

### 方法

提出了一种新的三阶时间精确且稳定的数值方案，并通过纳米条带模拟进行验证，与现有三阶半隐式方法进行比较。

### 主要发现

该方法能实现严格的三阶时间精度，提供卓越的计算效率；对于Gilbert阻尼系数α从0.1到低于10的范围内保持强稳定性；预测的磁微观结构与基准方法结果一致；相比现有方法，求解线性系统计算时间更少，对阻尼参数更敏感但收敛可靠，计算能级显著更低。

### 结论

该方案可靠且适用于定量物理分析，在磁化动力学模拟中具有显著优势。

### 翻译

高保真数值模拟是研究微磁学中磁化动力学的基石。本文介绍了一种针对Landau-Lifshitz-Gilbert方程的新型三阶时间精确且稳定的数值方案，旨在解决传统方法在精度和效率方面常遇到的局限性。通过纳米条带模拟验证，确认了所提出方法的两个主要优势：它达到了严格的三阶时间精度，超越了当前许多技术，并且提供了卓越的计算效率，能够快速收敛而不牺牲数值精度。对于Gilbert阻尼系数α从0.1到低于10的范围内，该方案保持强稳定性并有效避免非物理磁化状态。该方法预测的磁微观结构与已建立的基准方法极好一致，证实了其在定量物理分析中的可靠性。所提出方案与现有三阶半隐式方法之间的显著区别包括：(1) 求解与现有方案相关的线性系统需要多得多的计算时间，突显了对高效求解器的需求；(2) 尽管所提出方法对阻尼参数表现出更高的敏感性，但它能可靠地收敛到稳定的物理状态，并且在模拟磁畴壁运动方面有效，产生与先前已验证研究一致的结果；(3) 所提出方法计算的能级显著低于通过现有三阶方案获得的能级。


### 论文摘要

High-fidelity numerical simulation serves as a cornerstone for exploring magnetization dynamics in micromagnetics. This work introduces a novel third-order temporally accurate and stable numerical scheme for the Landau-Lifshitz-Gilbert (LLG) equation, aiming to address the limitations in accuracy and efficiency often encountered with conventional approaches. Validation via nanostrip simulations confirms two principal advantages of the proposed method: it attains strict third-order temporal accuracy, surpassing many current techniques, and it offers superior computational efficiency, enabling rapid convergence without sacrificing numerical precision. For Gilbert damping coefficients $α$ ranging from $0.1$ to values below $10$, the scheme preserves strong stability and effectively avoids non-physical magnetization states. The magnetic microstructures predicted by this method are in excellent agreement with those from established benchmark methods, affirming its reliability for quantitative physical analysis. Salient distinctions between the proposed scheme and an existing third-order semi-implicit method include:   (1) Solving the linear system associated with the existing scheme demands substantially greater computational time, underscoring the need for highly efficient solvers;   (2) Although the proposed method shows increased sensitivity to damping parameters, it reliably converges to stable physical states and is effective in simulating magnetic domain wall motion, producing outcomes consistent with prior validated studies;   (3) The energy levels computed by the proposed method are significantly lower than those obtained via the existing third-order scheme.

---

## 157. Beyond Realism: Learning the Art of Expressive Composition with StickerNet

**论文链接:** [http://arxiv.org/abs/2511.20957v1](http://arxiv.org/abs/2511.20957v1)

**作者:** Haoming Lu, David Kocharian, Humphrey Shi

**发布时间:** 2025-11-26

### GPT解析

### 总结

本文提出了一种新的图像合成任务——表达性合成，以及一个名为StickerNet的两阶段框架来处理这种任务。与传统的图像合成不同，这种方法更注重艺术性、趣味性和社交参与度，而非视觉真实性和语义合理性。

### 背景

图像合成是图像编辑工作流程中的常用操作，传统研究主要关注视觉真实性和语义合理性。然而在现代内容创作环境中，许多合成并不旨在保持真实感，而是用户为了获得社区认可而创作更具艺术性、趣味性或社交性的内容。

### 目的

定义一个新的图像合成任务——表达性合成，反映用户在实际创意平台上的编辑方式，并开发一个能够处理这种新任务的框架。

### 方法

作者提出了StickerNet，一个两阶段框架，首先确定合成类型，然后预测放置参数（如不透明度、掩码、位置和缩放）。他们直接从匿名在线视觉创作和编辑平台上收集的180万个编辑动作构建数据集，这些数据反映了用户-社区验证的放置决策。

### 主要发现

用户研究和定量评估显示，StickerNet优于常见基线方法，并 closely 匹配人类放置行为，表明尽管任务本身存在固有的模糊性，但从真实世界编辑模式中学习是有效的。

### 结论

这项工作引入了视觉理解的新方向，强调表达性和用户意图而非真实感。

### 翻译

作为图像编辑工作流程中广泛使用的操作，图像合成传统上一直侧重于实现视觉真实性和语义合理性。然而，在现代内容创作景观的实际编辑场景中，许多合成并不旨在保持真实感。相反，在线平台用户为获得社区认可，往往旨在创作更具艺术性、趣味性或社交参与度的内容。受此观察启发，我们定义了表达性合成任务，这是图像合成的新公式，接受风格多样性和更宽松的放置逻辑，反映了用户在实际创意平台上编辑图像的方式。为解决这一探索不足的问题，我们提出了StickerNet，一个两阶段框架，首先确定合成类型，然后相应地预测不透明度、掩码、位置和缩放等放置参数。与之前通过在真实图像上模拟对象放置来构建数据集的工作不同，我们直接从匿名在线视觉创作和编辑平台上收集的180万个编辑动作构建数据集，每个动作都反映了用户-社区验证的放置决策。这种基于真实编辑行为的确保了任务定义和训练监督之间的强一致性。用户研究和定量评估显示，StickerNet优于常见基线方法，并 closely 匹配人类放置行为，证明了尽管任务存在固有模糊性，但从真实世界编辑模式中学习的有效性。这项工作引入了视觉理解的新方向，强调表达性和用户意图而非真实感。


### 论文摘要

As a widely used operation in image editing workflows, image composition has traditionally been studied with a focus on achieving visual realism and semantic plausibility. However, in practical editing scenarios of the modern content creation landscape, many compositions are not intended to preserve realism. Instead, users of online platforms motivated by gaining community recognition often aim to create content that is more artistic, playful, or socially engaging. Taking inspiration from this observation, we define the expressive composition task, a new formulation of image composition that embraces stylistic diversity and looser placement logic, reflecting how users edit images on real-world creative platforms. To address this underexplored problem, we present StickerNet, a two-stage framework that first determines the composition type, then predicts placement parameters such as opacity, mask, location, and scale accordingly. Unlike prior work that constructs datasets by simulating object placements on real images, we directly build our dataset from 1.8 million editing actions collected on an anonymous online visual creation and editing platform, each reflecting user-community validated placement decisions. This grounding in authentic editing behavior ensures strong alignment between task definition and training supervision. User studies and quantitative evaluations show that StickerNet outperforms common baselines and closely matches human placement behavior, demonstrating the effectiveness of learning from real-world editing patterns despite the inherent ambiguity of the task. This work introduces a new direction in visual understanding that emphasizes expressiveness and user intent over realism.

---

