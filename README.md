# 今日论文推荐 - 2025-10-23

共 25 篇论文

---

## 1. MobiAct: Efficient MAV Action Recognition Using MobileNetV4 with Contrastive Learning and Knowledge Distillation

**论文链接:** [http://arxiv.org/abs/2510.19273v1](http://arxiv.org/abs/2510.19273v1)

**作者:** Zhang Nengbo, Ho Hann Woei

**发布时间:** 2025-10-22

### GPT解析

### 总结

本文提出了一个轻量级的MAV动作识别框架MobiAct，实现了高精度与低计算成本的平衡，在保持92.12%平均识别准确率的同时，仅消耗136.16 pJ能量并以每秒8.84个动作的速度处理，解码速度比领先方法快2倍。

### 背景

微型飞行器(MAV)的精确高效运动识别对于自主空中群体的实时感知和协调至关重要。然而，现有方法大多依赖于大型、计算密集型模型，不适合资源有限的MAV平台，导致识别精度和推理速度之间的权衡。

### 目的

提出一种轻量级的MAV动作识别框架MobiAct，旨在以低计算成本实现高精度的MAV动作识别。

### 方法

采用MobileNetV4作为骨干网络；引入分阶段正交知识蒸馏(SOKD)策略将教师网络(ResNet18)的MAV运动特征有效转移到学生网络；集成无参数注意力机制提高识别精度而不增加模型复杂度；开发混合损失训练策略结合多个损失目标确保训练过程的稳定和鲁棒优化。

### 主要发现

MobiAct实现了低能耗、低计算的MAV动作识别；在所有三个自收集数据集上，平均识别准确率达到92.12%；仅消耗136.16 pJ能量，处理速度为每秒8.84个动作；动作解码速度比领先方法快2倍，同时保持高度相当的识别精度。

### 结论

MobiAct在MAV动作识别方面展现出卓越的效率，成功解决了识别精度与计算资源消耗之间的权衡问题。

### 翻译

微型飞行器(MAV)运动的精确高效识别对于自主空中群体的实时感知和协调至关重要。然而，大多数现有方法依赖于大型、计算密集型模型，不适合资源有限的MAV平台，这导致了识别精度和推理速度之间的权衡。为解决这些挑战，本文提出了一个轻量级的MAV动作识别框架MobiAct，旨在以低计算成本实现高精度。具体而言，MobiAct采用MobileNetV4作为骨干网络，并引入分阶段正交知识蒸馏(SOKD)策略，将MAV运动特征从教师网络(ResNet18)有效转移到学生网络，从而提高知识转移效率。此外，架构中集成了无参数注意力机制，在不增加模型复杂度的情况下提高识别精度。此外，还开发了混合损失训练策略，结合多个损失目标，确保训练过程中的稳定和鲁棒优化。实验结果表明，所提出的MobiAct实现了低能耗、低计算的MAV动作识别，同时在比较的方法中保持最快的动作解码速度。在所有三个自收集数据集上，MobiAct平均识别准确率达到92.12%，而仅消耗136.16 pJ的能量，并以每秒8.84个动作的速度进行识别。值得注意的是，MobiAct的动作解码速度比领先方法快2倍，同时具有高度相当的识别精度，突显了其在MAV动作识别方面的卓越效率。


### 论文摘要

Accurate and efficient recognition of Micro Air Vehicle (MAV) motion is essential for enabling real-time perception and coordination in autonomous aerial swarm. However, most existing approaches rely on large, computationally intensive models that are unsuitable for resource-limited MAV platforms, which results in a trade-off between recognition accuracy and inference speed. To address these challenges, this paper proposes a lightweight MAV action recognition framework, MobiAct, designed to achieve high accuracy with low computational cost. Specifically, MobiAct adopts MobileNetV4 as the backbone network and introduces a Stage-wise Orthogonal Knowledge Distillation (SOKD) strategy to effectively transfer MAV motion features from a teacher network (ResNet18) to a student network, thereby enhancing knowledge transfer efficiency. Furthermore, a parameter-free attention mechanism is integrated into the architecture to improve recognition accuracy without increasing model complexity. In addition, a hybrid loss training strategy is developed to combine multiple loss objectives, which ensures stable and robust optimization during training. Experimental results demonstrate that the proposed MobiAct achieves low-energy and low-computation MAV action recognition, while maintaining the fastest action decoding speed among compared methods. Across all three self-collected datasets, MobiAct achieves an average recognition accuracy of 92.12%, while consuming only 136.16 pJ of energy and processing recognition at a rate of 8.84 actions per second. Notably, MobiAct decodes actions up to 2 times faster than the leading method, with highly comparable recognition accuracy, highlighting its superior efficiency in MAV action recognition.

---

## 2. X-Ego: Acquiring Team-Level Tactical Situational Awareness via Cross-Egocentric Contrastive Video Representation Learning

**论文链接:** [http://arxiv.org/abs/2510.19150v1](http://arxiv.org/abs/2510.19150v1)

**作者:** Yunzhe Wang, Soham Hans, Volkan Ustun

**发布时间:** 2025-10-22

**备注:** 8 pages, 5 figures

### GPT解析

### 总结

该研究引入了X-Ego-CS基准数据集和交叉自我中心对比学习(CECL)方法，用于研究电竞游戏中的多智能体决策和团队战术学习。

### 背景

人类团队战术源于个人视角及其预测、解释和适应队友意图的能力。现有视频理解研究虽改善了体育中团队互动建模，但大多依赖第三方广播视角，忽视了多智能体学习的同步、自我中心特性。

### 目的

引入X-Ego-CS基准数据集，促进复杂3D环境中多智能体决策的研究，并提供交叉自我中心视角来捕捉团队互动。

### 方法

X-Ego-CS数据集包含45场专业级《反恐精英2》比赛的124小时游戏录像，提供所有玩家的同步第一人称视角和状态-行动轨迹。提出CECL方法，对齐队友的自我中心视觉流，培养团队战术情境意识。

### 主要发现

CECL能有效增强智能体从单一第一人称视图推断队友和对手位置的能力，使用最先进的视频编码器实现了有效性能。

### 结论

X-Ego-CS和CECL为电竞中的交叉自我中心多智能体基准测试奠定基础，将游戏理解定位为多智能体建模和战术学习的测试平台，对虚拟和现实领域中的时空推理和人类-AI团队协作具有启示意义。

### 翻译

人类团队战术源于每个球员的个人视角及其预测、解释和适应队友意图的能力。尽管视频理解方面的进展已改善了体育中团队互动的建模，但大多数现有工作依赖第三方广播视角，并忽视了多智能体学习的同步、自我中心特性。我们引入X-Ego-CS基准数据集，包含来自45场专业级流行电竞游戏《反恐精英2》的124小时游戏录像，旨在促进复杂3D环境中多智能体决策的研究。X-Ego-CS提供交叉自我中心视频流，同步捕捉所有玩家的第一人称视角以及状态-行动轨迹。基于此资源，我们提出交叉自我中心对比学习(CECL)，对齐队友的自我中心视觉流，从个人视角培养团队层面的战术情境意识。我们在队友-对手位置预测任务上评估CECL，证明了其有效性，能够增强智能体使用最先进的视频编码器从单一第一人称视图推断队友和对手位置的能力。X-Ego-CS和CECL共同为电竞中的交叉自我中心多智能体基准测试奠定基础。更广泛地说，我们的工作将游戏理解定位为多智能体建模和战术学习的测试平台，对虚拟和现实领域中的时空推理以及人类-AI团队协作具有启示意义。代码和数据集可在https://github.com/HATS-ICT/x-ego获取。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决如何让AI系统从团队成员的第一人称视角中获得团队层面的战术态势感知能力。这个问题在现实中很重要，因为真实世界的团队协作（如体育竞技、军事行动、应急响应等）需要参与者能够根据队友和对手的意图来协调行动，而现有方法大多依赖第三人称视角，无法捕捉个体感知和协调的第一人称特性，限制了智能体在部分可观测环境中的表现。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先分析了现有团队行为建模方法的局限性，特别是在处理部分可观测环境中的团队协调时。他们借鉴了体育理解中的第三人称分析方法、游戏理解中的人-AI协作研究以及对比学习在计算机视觉和多智能体学习中的应用。作者选择使用第一人称射击游戏（反恐精英）作为研究平台，因为它提供了丰富的游戏状态和决策复杂性。基于此，他们创建了X-Ego-CS数据集并设计了跨自我中心对比学习（CECL）方法，通过对比学习对齐队友的第一人称视觉表征。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是通过对比学习对齐队友的第一人称视觉表征，使模型能够从有限的第一人称视角中推断团队层面的战术态势。整体流程包括：1) 收集和处理专业级反恐精英比赛数据，提取第一人称视频流和状态-动作轨迹；2) 使用时空视频编码器处理每个玩家的视角；3) 应用对比学习目标函数，使同一时间点的队友视角产生相似表征；4) 设计下游任务（队友和对手位置预测）来评估模型性能；5) 结合对比损失和分类损失进行训练，使模型能够从单个视角推断团队态势。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1) X-Ego-CS数据集：首个包含同步第一人称视频流和结构化状态-动作轨迹的专业电子竞技数据集；2) 跨自我中心对比学习（CECL）方法：通过对比学习对齐队友视角，实现团队态势感知；3) 队友-对手位置预测任务：为评估团队理解能力提供标准化基准。相比之前工作，本文方法使用同步第一人称视角而非第三人称视角，提供完整的第一人称视频流和精确轨迹数据，并通过对比学习模拟人类心智理论能力，在复杂3D环境中验证而非简化环境。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '本文贡献了一个包含同步第一人称视频流的专业电子竞技数据集和一个通过对比学习对齐队友视角的方法，使AI系统能够从有限的第一人称视角中获取团队层面的战术态势感知能力，为多智能体系统中的团队协作研究建立了新的基准。'}


### 论文摘要

Human team tactics emerge from each player's individual perspective and their ability to anticipate, interpret, and adapt to teammates' intentions. While advances in video understanding have improved the modeling of team interactions in sports, most existing work relies on third-person broadcast views and overlooks the synchronous, egocentric nature of multi-agent learning. We introduce X-Ego-CS, a benchmark dataset consisting of 124 hours of gameplay footage from 45 professional-level matches of the popular e-sports game Counter-Strike 2, designed to facilitate research on multi-agent decision-making in complex 3D environments. X-Ego-CS provides cross-egocentric video streams that synchronously capture all players' first-person perspectives along with state-action trajectories. Building on this resource, we propose Cross-Ego Contrastive Learning (CECL), which aligns teammates' egocentric visual streams to foster team-level tactical situational awareness from an individual's perspective. We evaluate CECL on a teammate-opponent location prediction task, demonstrating its effectiveness in enhancing an agent's ability to infer both teammate and opponent positions from a single first-person view using state-of-the-art video encoders. Together, X-Ego-CS and CECL establish a foundation for cross-egocentric multi-agent benchmarking in esports. More broadly, our work positions gameplay understanding as a testbed for multi-agent modeling and tactical learning, with implications for spatiotemporal reasoning and human-AI teaming in both virtual and real-world domains. Code and dataset are available at https://github.com/HATS-ICT/x-ego.

---

## 3. UniHPR: Unified Human Pose Representation via Singular Value Contrastive Learning

**论文链接:** [http://arxiv.org/abs/2510.19078v1](http://arxiv.org/abs/2510.19078v1)

**作者:** Zhongyu Jiang, Wenhao Chai, Lei Li, Zhuoran Zhou, Cheng-Yen Yang, Jenq-Neng Hwang

**发布时间:** 2025-10-21

### GPT解析

### 总结

本文提出了一种名为UniHPR的统一人体姿态表示学习管道，通过创新的基于奇异值的对比学习损失函数，实现了图像、2D和3D人体姿态表示的有效对齐，并在人体姿态估计和检索任务中取得了优异性能。

### 背景

近年来，开发有效的对齐管道以从不同模态生成统一表示受到越来越多的关注。人体姿态表示作为以人为中心应用的关键组成部分，在人体姿态估计、动作识别、人机交互、目标跟踪等下游任务中至关重要。然而，目前很少有研究使用对比范式清晰研究多种人体姿态表示之间的相关性。

### 目的

提出UniHPR，一个统一的人体姿态表示学习管道，用于对齐来自图像、2D和3D人体姿态的人体姿态嵌入。

### 方法

提出了一种新颖的基于奇异值的对比学习损失函数，用于同时对齐超过两种数据表示，更好地对齐不同模态并进一步提高性能。选择2D和3D人体姿态估计作为评估任务，以验证对齐表示的有效性。

### 主要发现

使用简单的3D人体姿态解码器，UniHPR在Human3.6M数据集上实现了49.9mm的MPJPE性能指标，在3DPW数据集上实现了跨域评估的51.6mm PA-MPJPE性能指标。此外，在Human3.6M数据集上，使用统一的人体姿态表示实现了2D和3D姿态检索，检索误差为9.24mm的MPJPE。

### 结论

UniHPR能够有效对齐不同模态的人体姿态表示，并在多种下游任务中展现出优异的性能，为多模态人体姿态表示的学习提供了新的解决方案。

### 翻译

近年来，人们越来越关注开发有效的对齐管道，从不同模态生成统一表示，用于多模态融合和生成。作为以人为中心应用的重要组成部分，人体姿态表示在许多下游任务中至关重要，如人体姿态估计、动作识别、人机交互、目标跟踪等。人体姿态表示或嵌入可以从图像、2D关键点、3D骨架、网格模型等多种模态中提取。然而，使用对比范式清晰研究所有这些表示之间相关性的实例有限。在本文中，我们提出UniHPR，一个统一的人体姿态表示学习管道，用于对齐来自图像、2D和3D人体姿态的人体姿态嵌入。为了同时对齐超过两种数据表示，我们提出了一种新颖的基于奇异值的对比学习损失函数，更好地对齐不同模态并进一步提高性能。为了评估对齐表示的有效性，我们选择2D和3D人体姿态估计(HPE)作为评估任务。在我们的评估中，使用简单的3D人体姿态解码器，UniHPR在Human3.6M数据集上实现了49.9mm的MPJPE性能指标，在3DPW数据集上实现了跨域评估的51.6mm的PA-MPJPE性能指标。同时，我们能够在Human3.6M数据集上使用统一的人体姿态表示实现2D和3D姿态检索，检索误差为9.24mm的MPJPE。


### 论文摘要

In recent years, there has been a growing interest in developing effective alignment pipelines to generate unified representations from different modalities for multi-modal fusion and generation. As an important component of Human-Centric applications, Human Pose representations are critical in many downstream tasks, such as Human Pose Estimation, Action Recognition, Human-Computer Interaction, Object tracking, etc. Human Pose representations or embeddings can be extracted from images, 2D keypoints, 3D skeletons, mesh models, and lots of other modalities. Yet, there are limited instances where the correlation among all of those representations has been clearly researched using a contrastive paradigm. In this paper, we propose UniHPR, a unified Human Pose Representation learning pipeline, which aligns Human Pose embeddings from images, 2D and 3D human poses. To align more than two data representations at the same time, we propose a novel singular value-based contrastive learning loss, which better aligns different modalities and further boosts performance. To evaluate the effectiveness of the aligned representation, we choose 2D and 3D Human Pose Estimation (HPE) as our evaluation tasks. In our evaluation, with a simple 3D human pose decoder, UniHPR achieves remarkable performance metrics: MPJPE 49.9mm on the Human3.6M dataset and PA-MPJPE 51.6mm on the 3DPW dataset with cross-domain evaluation. Meanwhile, we are able to achieve 2D and 3D pose retrieval with our unified human pose representations in Human3.6M dataset, where the retrieval error is 9.24mm in MPJPE.

---

## 4. ProCLIP: Progressive Vision-Language Alignment via LLM-based Embedder

**论文链接:** [http://arxiv.org/abs/2510.18795v2](http://arxiv.org/abs/2510.18795v2)

**作者:** Xiaoxing Hu, Kaicheng Yang, Ziyang Gong, Qi Ming, Zonghao Guo, Xiang An, Ziyong Feng, Junchi Yan, Xue Yang

**发布时间:** 2025-10-21

**备注:** 17 pages, 5 fiugres

### GPT解析

### 总结

本文提出ProCLIP框架，解决CLIP文本编码器在处理长文本和多语言输入方面的局限性，通过课程学习实现CLIP图像编码器与LLM嵌入器的有效对齐。

### 背景

原始CLIP文本编码器受限于77个token的最大输入长度，不支持多语言输入，这些限制显著阻碍了其在更广泛任务中的应用。虽然近期研究尝试用基于LLM的嵌入器替代CLIP文本编码器，但由于LLM和CLIP的表示空间独立预训练且缺乏先验对齐，直接使用对比学习会破坏CLIP图像编码器中固有的视觉-语言对齐。

### 目的

开发一种方法来有效对齐CLIP图像编码器与基于LLM的嵌入器，同时保留CLIP的预训练知识，从而增强模型在处理长文本、多语言理解和细粒度语义理解方面的能力。

### 方法

ProCLIP采用课程学习的渐进式视觉-语言对齐框架：首先从CLIP文本编码器中蒸馏知识到LLM嵌入器建立初始对齐；然后通过图像-文本对比微调进一步对齐，并使用自蒸馏正则化避免过拟合；在表示继承和对比微调过程中采用实例语义对齐损失和嵌入结构对齐损失以实现更有效的对齐。

### 主要发现

直接对齐LLM和CLIP的表示空间会破坏CLIP图像编码器中固有的视觉-语言对齐，导致预训练知识利用不足；而ProCLIP框架能够有效对齐两者并保留CLIP的预训练知识。

### 结论

ProCLIP通过课程学习的渐进式对齐方法，解决了LLM嵌入器和CLIP图像编码器之间的对齐问题，同时保留了CLIP的预训练知识，显著提升了模型在长文本处理、多语言理解和细粒度语义理解方面的能力。

### 翻译

原始的CLIP文本编码器受限于最大77个token的输入长度，这妨碍了它有效处理长文本和进行细粒度语义理解的能力。此外，CLIP文本编码器不支持多语言输入。所有这些限制显著限制了它在更广泛任务中的应用性。最近的研究尝试用基于LLM的嵌入器替换CLIP文本编码器，以增强其处理长文本、多语言理解和细粒度语义理解的能力。然而，由于LLM的表示空间和CLIP的视觉-语言空间是独立预训练且没有对齐先验，直接使用对比学习对齐会破坏CLIP图像编码器中固有的视觉-语言对齐，导致预训练知识利用不足。为解决这一挑战，我们提出ProCLIP，一种基于课程学习的渐进式视觉-语言对齐框架，以有效对齐CLIP图像编码器和基于LLM的嵌入器。具体而言，ProCLIP首先从CLIP的文本编码器中蒸馏知识到基于LLM的嵌入器，利用CLIP丰富的预训练知识，同时建立LLM嵌入器和CLIP图像编码器之间的初始对齐。随后，ProCLIP通过图像-文本对比微调进一步对齐CLIP图像编码器和基于LLM的嵌入器，采用自蒸馏正则化来避免过拟合。为了实现更有效的对齐，在表示继承和对比微调过程中采用了实例语义对齐损失和嵌入结构对齐损失。代码可在https://github.com/VisionXLab/ProCLIP获取。


### 论文摘要

The original CLIP text encoder is limited by a maximum input length of 77 tokens, which hampers its ability to effectively process long texts and perform fine-grained semantic understanding. In addition, the CLIP text encoder lacks support for multilingual inputs. All these limitations significantly restrict its applicability across a broader range of tasks. Recent studies have attempted to replace the CLIP text encoder with an LLM-based embedder to enhance its ability in processing long texts, multilingual understanding, and fine-grained semantic comprehension. However, because the representation spaces of LLMs and the vision-language space of CLIP are pretrained independently without alignment priors, direct alignment using contrastive learning can disrupt the intrinsic vision-language alignment in the CLIP image encoder, leading to an underutilization of the knowledge acquired during pre-training. To address this challenge, we propose ProCLIP, a curriculum learning-based progressive vision-language alignment framework to effectively align the CLIP image encoder with an LLM-based embedder. Specifically, ProCLIP first distills knowledge from CLIP's text encoder into the LLM-based embedder to leverage CLIP's rich pretrained knowledge while establishing initial alignment between the LLM embedder and CLIP image encoder. Subsequently, ProCLIP further aligns the CLIP image encoder with the LLM-based embedder through image-text contrastive tuning, employing self-distillation regularization to avoid overfitting. To achieve a more effective alignment, instance semantic alignment loss and embedding structure alignment loss are employed during representation inheritance and contrastive tuning. The Code is available at https://github.com/VisionXLab/ProCLIP.

---

## 5. Decomposed Attention Fusion in MLLMs for Training-Free Video Reasoning Segmentation

**论文链接:** [http://arxiv.org/abs/2510.19592v1](http://arxiv.org/abs/2510.19592v1)

**作者:** Su Ho Han, Jeongseok Hyun, Pilhyeon Lee, Minho Shim, Dongyoon Wee, Seon Joo Kim

**发布时间:** 2025-10-22

**备注:** Project page: https://www.jshyun.me/projects/decaf

### GPT解析

### 总结

本文提出了一种名为DecAF的分解注意力融合方法，用于在无需重新训练多模态大语言模型(MLLMs)的情况下实现视频理解与定位，通过改进注意力图实现了与需要训练的方法相当的性能。

### 背景

多模态大语言模型(MLLMs)能够通过关注与文本查询相关的视觉标记来展示强大的视频理解能力，但直接将其应用于定位任务存在挑战。

### 目的

开发一种无需训练的方法，将MLLMs的视频理解能力直接适应于视频推理分割任务。

### 方法

将视频推理分割视为视频问答任务并通过展开机制提取注意力图；提出DecAF方法，通过对比对象-背景融合和互补视频帧融合两种机制改进原始注意力图；引入注意力引导的SAM2提示获取精细掩码。

### 主要发现

DecAF能够抑制不相关的激活并增强对象聚焦的线索，使注意力图可以直接转换为粗略分割掩码；无需训练的方法实现了与需要训练的方法相当的性能。

### 结论

DecAF优于现有的无需训练的方法，并在指代和推理VOS基准测试上达到了与基于训练方法相当的性能；与现有的将MLLMs与SAM联合训练的方法不同，DecAF完全无需重新训练。

### 翻译

多模态大语言模型(MLLMs)通过关注与文本查询相关的视觉标记展示了强大的视频理解能力。为了直接以无需训练的方式将其适应于定位任务，我们将视频推理分割视为视频问答任务，并通过展开机制提取注意力图。然而，原始注意力图嘈杂且与对象区域对齐不良。我们提出了分解注意力融合(DecAF)，通过两种机制改进这些图：(1)对比对象-背景融合和(2)互补视频帧融合。此方法抑制了不相关的激活并增强了对象聚焦的线索，使注意力图可以直接转换为粗略分割掩码。此外，我们引入了注意力引导的SAM2提示来获取精细掩码。与现有的将MLLMs与SAM联合训练的方法不同，我们的方法完全无需重新训练。DecAF优于无需训练的方法，并在指代和推理VOS基准测试上实现了与基于训练方法相当的性能。代码将在https://github.com/HYUNJS/DecAF上提供。


### 论文摘要

Multimodal large language models (MLLMs) demonstrate strong video understanding by attending to visual tokens relevant to textual queries. To directly adapt this for localization in a training-free manner, we cast video reasoning segmentation as a video QA task and extract attention maps via rollout mechanism. However, raw attention maps are noisy and poorly aligned with object regions. We propose Decomposed Attention Fusion (DecAF), which refines these maps through two mechanisms: (1) contrastive object-background fusion and (2) complementary video-frame fusion. This method suppresses irrelevant activations and enhances object-focused cues, enabling direct conversion of attention maps into coarse segmentation masks. In addition, we introduce attention-guided SAM2 prompting for obtaining fine-grained masks. Unlike existing methods that jointly train MLLMs with SAM, our method operates entirely without retraining. DecAF outperforms training-free methods and achieves performance comparable to training-based methods on both referring and reasoning VOS benchmarks. The code will be available at https://github.com/HYUNJS/DecAF.

---

## 6. A Matter of Time: Revealing the Structure of Time in Vision-Language Models

**论文链接:** [http://arxiv.org/abs/2510.19559v1](http://arxiv.org/abs/2510.19559v1)

**作者:** Nidham Tekaya, Manuela Waldner, Matthias Zeppelzauer

**发布时间:** 2025-10-22

**DOI:** 10.1145/3746027.3758163

### GPT解析

### 总结

本文研究了大规模视觉语言模型的时间感知能力，提出了TIME10k基准数据集，并发现时间信息在VLM嵌入空间中沿低维非线性流形结构化，基于此提出了时间线表示方法，该方法在计算效率高的同时实现了优异的时间推理性能。

### 背景

大规模视觉语言模型如CLIP因其可泛化和表达性的多模态表示而受到欢迎。这些模型通过利用具有多样化文本元数据的大规模训练数据，获得了开放词汇能力，能够解决超出其训练范围的任务。

### 目的

研究视觉语言模型的时间感知能力，评估它们将视觉内容定位在时间中的能力。

### 方法

引入TIME10k基准数据集（包含超过10,000张图像的时间基准数据），通过一种新方法评估37个VLMs的时间感知能力，并基于发现提出从嵌入空间推导显式'时间线'表示的方法。

### 主要发现

时间信息在VLM嵌入空间中沿着低维、非线性的流形结构化，基于此可以推导出显式的'时间线'表示。

### 结论

提出的时间线表示方法能够模拟时间及其时间进展，促进时间推理任务，在计算效率高的同时，实现了与基于提示的基线相当或更优的准确性。

### 翻译

大规模视觉语言模型如CLIP因其可泛化和表达性的多模态表示而受到欢迎。通过利用具有多样化文本元数据的大规模训练数据，VLMs获得了开放词汇能力，能够解决超出其训练范围的任务。本文研究了VLMs的时间感知能力，评估它们将视觉内容定位在时间中的能力。我们引入了TIME10k，一个包含超过10,000张图像的时间基准数据集，并通过一种新方法评估了37个VLMs的时间感知能力。我们的研究揭示，时间信息在VLM嵌入空间中沿着低维、非线性的流形结构化。基于这一见解，我们提出了从嵌入空间推导显式'时间线'表示的方法。这些表示模拟时间及其时间进展，从而促进时间推理任务。我们的时间线方法在计算效率高的同时，实现了与基于提示的基线相当或更优的准确性。所有代码和数据都在https://tekayanidham.github.io/timeline-page/上提供。


### 论文摘要

Large-scale vision-language models (VLMs) such as CLIP have gained popularity for their generalizable and expressive multimodal representations. By leveraging large-scale training data with diverse textual metadata, VLMs acquire open-vocabulary capabilities, solving tasks beyond their training scope. This paper investigates the temporal awareness of VLMs, assessing their ability to position visual content in time. We introduce TIME10k, a benchmark dataset of over 10,000 images with temporal ground truth, and evaluate the time-awareness of 37 VLMs by a novel methodology. Our investigation reveals that temporal information is structured along a low-dimensional, non-linear manifold in the VLM embedding space. Based on this insight, we propose methods to derive an explicit ``timeline'' representation from the embedding space. These representations model time and its chronological progression and thereby facilitate temporal reasoning tasks. Our timeline approaches achieve competitive to superior accuracy compared to a prompt-based baseline while being computationally efficient. All code and data are available at https://tekayanidham.github.io/timeline-page/.

---

## 7. PRGCN: A Graph Memory Network for Cross-Sequence Pattern Reuse in 3D Human Pose Estimation

**论文链接:** [http://arxiv.org/abs/2510.19475v1](http://arxiv.org/abs/2510.19475v1)

**作者:** Zhuoyang Xie, Yibo Zhao, Hui Huang, Riwei Wang, Zan Gao

**发布时间:** 2025-10-22

**备注:** 29 pages, 6 figures, 6 tables

### GPT解析

### 总结

本文提出了一种名为PRGCN的新型框架，通过跨序列模式检索和适应来解决单目3D人体姿态估计中的深度模糊性问题。该方法利用图记忆库存储姿态原型，并通过注意力机制动态检索，结合双流混合架构实现了最先进的性能和跨域泛化能力。

### 背景

单目3D人体姿态估计是一个不适定的逆问题，因为从2D到3D的提升中存在固有的深度模糊性。现有基于视频的方法虽然利用时间上下文增强空间推理，但独立处理每个序列，未能充分利用跨序列中人类运动的强结构规律性和重复运动模式。

### 目的

解决单目3D人体姿态估计中的深度模糊性问题，突破现有方法仅独立处理每个序列的局限，通过跨序列模式重用机制提升姿态估计的性能和泛化能力。

### 方法

提出PRGCN框架，将姿态估计形式化为模式检索和适应问题；引入图记忆库学习和存储姿态原型；通过注意力机制动态检索提供结构化先验；通过内存驱动的图卷积将先验与解剖约束融合；设计双流混合架构，结合Mamba的局部时间建模和自注意力的全局关系能力。

### 主要发现

在Human3.6M和MPI-INF-3DHP基准测试上，PRGCN实现了37.1mm和13.4mm的MPJPE，建立了新的最先进水平，同时表现出增强的跨域泛化能力。

### 结论

跨序列模式重用机制对推进人体姿态估计领域至关重要，将研究范式从每序列优化转向累积知识学习。

### 翻译

单目3D人体姿态估计由于2D到3D提升中的固有深度模糊性，仍然是一个根本性的不适定逆问题。虽然当代基于视频的方法利用时间上下文来增强空间推理，但它们在关键范式限制下运行：独立处理每个序列，因此未能充分利用跨序列中普遍存在的强结构规律性和重复运动模式。这项工作引入了模式重用图卷积网络，一个将姿态估计形式化为模式检索和适应问题的新型框架。其核心是，PRGCN具有一个图记忆库，学习和存储一组紧凑的姿态原型，编码为关系图，这些原型通过注意力机制动态检索以提供结构化先验。这些先验通过内存驱动的图卷积与硬编码的解剖约束自适应融合，确保几何合理性。为了用鲁棒的空间-时间特征支持这一检索过程，我们设计了一个双流混合架构，协同结合了基于Mamba的状态空间模型的线性复杂度局部时间建模与自注意力的全局关系能力。在Human3.6M和MPI-INF-3DHP基准测试上的广泛评估表明，PRGCN建立了新的最先进水平，分别实现了37.1mm和13.4mm的MPJPE，同时表现出增强的跨域泛化能力。我们的研究认为，长期以来被忽视的跨序列模式重用机制对推进该领域至关重要，将范式从每序列优化转向累积知识学习。


### 论文摘要

Monocular 3D human pose estimation remains a fundamentally ill-posed inverse problem due to the inherent depth ambiguity in 2D-to-3D lifting. While contemporary video-based methods leverage temporal context to enhance spatial reasoning, they operate under a critical paradigm limitation: processing each sequence in isolation, thereby failing to exploit the strong structural regularities and repetitive motion patterns that pervade human movement across sequences. This work introduces the Pattern Reuse Graph Convolutional Network (PRGCN), a novel framework that formalizes pose estimation as a problem of pattern retrieval and adaptation. At its core, PRGCN features a graph memory bank that learns and stores a compact set of pose prototypes, encoded as relational graphs, which are dynamically retrieved via an attention mechanism to provide structured priors. These priors are adaptively fused with hard-coded anatomical constraints through a memory-driven graph convolution, ensuring geometrical plausibility. To underpin this retrieval process with robust spatiotemporal features, we design a dual-stream hybrid architecture that synergistically combines the linear-complexity, local temporal modeling of Mamba-based state-space models with the global relational capacity of self-attention. Extensive evaluations on Human3.6M and MPI-INF-3DHP benchmarks demonstrate that PRGCN establishes a new state-of-the-art, achieving an MPJPE of 37.1mm and 13.4mm, respectively, while exhibiting enhanced cross-domain generalization capability. Our work posits that the long-overlooked mechanism of cross-sequence pattern reuse is pivotal to advancing the field, shifting the paradigm from per-sequence optimization towards cumulative knowledge learning.

---

## 8. $Δ$t-Mamba3D: A Time-Aware Spatio-Temporal State-Space Model for Breast Cancer Risk Prediction

**论文链接:** [http://arxiv.org/abs/2510.19003v1](http://arxiv.org/abs/2510.19003v1)

**作者:** Zhengbo Zhou, Dooman Arefan, Margarita Zuley, Shandong Wu

**发布时间:** 2025-10-21

### GPT解析

### 总结

这项研究提出了Time-Aware Δt-Mamba3D，一种新型的状态空间架构，专门用于纵向医学图像分析。该模型能够有效编码不规则访问间隔和丰富的时空上下文，同时保持计算效率，在乳腺癌风险预测任务上表现出色。

### 背景

纵向放射图像分析面临一个基本数据挑战：如何有效建模在非规则时间间隔采集的高分辨率图像序列。这种数据结构包含重要的空间和时间线索，但当前方法无法充分利用。现有方法通常要么将空间信息压缩为向量，要么使用计算效率低下且与非均匀时间步不兼容的时空模型。

### 目的

开发一种能够有效处理不规则时间间隔采集的图像序列的模型，同时充分利用空间和时间信息，并保持计算效率，应用于纵向医学图像分析，特别是乳腺癌风险预测。

### 方法

研究者提出了Time-Aware Δt-Mamba3D，一种专为纵向医学成像设计的新的状态空间架构。该模型的核心创新是一个连续时间选择性扫描机制，明确地将检查之间的真实时间差异整合到状态转换中。此外，还采用了多尺度3D邻域融合模块，稳健地捕获时空关系。

### 主要发现

在乳腺癌风险预测基准测试中，该模型表现出色，验证c-index提高了2-5个百分点，相比现有的循环、变压器和状态空间模型的变体，实现了更高的1-5年AUC分数。由于具有线性复杂度，该模型能够高效处理长期复杂的患者筛查历史。

### 结论

Time-Aware Δt-Mamba3D为纵向图像分析形成了一个新框架，能够有效处理不规则时间间隔采集的图像序列，充分利用时空信息，同时保持计算效率，在医学图像分析任务中表现出色。

### 翻译

纵向连续放射图像分析受到一个基本数据挑战的阻碍：如何有效建模在非规则时间间隔采集的高分辨率图像序列。这种数据结构包含了当前方法无法充分利用的必不可少的空间和时间线索。模型通常要么将空间信息压缩为向量，要么使用计算效率低下且与非均匀时间步不兼容的时空模型。我们通过Time-Aware Δt-Mamba3D解决了这一挑战，这是一种专为纵向医学成像设计的新型状态空间架构。我们的模型同时编码不规则访问间隔和丰富的时空上下文，同时保持计算效率。其核心创新是一个连续时间选择性扫描机制，明确地将检查之间的真实时间差异整合到其状态转换中。这辅以一个多尺度3D邻域融合模块，稳健地捕获时空关系。在使用连续筛查乳腺X光检查的乳腺癌风险预测综合基准中，我们的模型表现出卓越性能，相比现有的循环、变压器和状态空间模型的变体，将验证c-index提高了2-5个百分点，并实现了更高的1-5年AUC分数。由于其线性复杂度，该模型能够高效处理长期复杂的患者筛查历史，为纵向图像分析形成了一个新框架。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决如何有效建模在不同时间间隔捕获的高分辨率图像序列的问题，特别是在乳腺癌筛查中不规则时间间隔的纵向放射学图像分析。这个问题很重要，因为乳腺癌是全球女性最常见的癌症之一，早期风险预测可以提高筛查效率；现有方法未能充分利用不规则时间间隔这一重要预测因素；医生评估风险时会考虑多次检查的比较，而大多数深度学习系统仍只处理单次检查；开发能够处理不规则时间间隔的高效模型对临床应用至关重要。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先认识到乳腺癌筛查是纵向的，患者会定期返回检查，乳房随年龄变化，病变可能逐渐显现。他们指出大多数深度学习系统忽略了时序背景，现有方法要么将空间信息压缩为向量，要么使用计算效率低下的模型，或者无法处理非均匀时间步长。作者分析了各种处理不规则时间序列的方法，发现它们都有局限性，而状态空间模型如Mamba虽能捕获长期依赖关系，但尚未显式编码不规则时间间隔。作者借鉴了Mamba的状态空间架构、3D卷积网络处理空间信息的思想、时间感知模型处理时间间隔的方法，以及视频视觉 transformers处理时空信息的思路，但都进行了改进以适应不规则时间间隔的医学成像数据。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是：1) 时间感知选择性扫描：将真实时间差Δt直接注入选择性扫描中，使模型能根据实际时间间隔调整状态更新；2) 多尺度3D邻域融合：使用深度3D卷积捕获空间和时间依赖关系，同时保持计算效率；3) 线性复杂度：能高效处理长期复杂的患者筛查历史。整体流程：1) 输入处理：每个患者的纵向成像序列，使用Swin-V2处理每个图像并融合特征；2) Δt-Mamba3D块处理：将特征展平为标记序列，运行Mamba选择性扫描，状态更新由真实Δt调制，然后重塑回3D格式并应用3D邻域融合；3) 患者嵌入和风险模块：跨空间和时间聚合特征获得患者嵌入，使用加性危害模型估计未来乳腺癌风险；4) 处理可变长度序列：左填充序列到固定长度，使用掩码处理填充标记。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点：1) 时间感知选择性扫描机制：将真实时间差Δt注入选择性扫描，实现连续时间记忆衰减或累积；2) 多尺度深度3D融合模块：使用深度3D卷积捕获空间和时间依赖关系；3) 线性复杂度设计：能高效处理长期复杂历史。相比之前工作的不同：1) 与时间感知模型(如GRU-D)相比，使用连续时间状态空间模型，能更好地建模观测间的演变风险；2) 与连续时间模型(如Neural ODEs)相比，专为高维医学成像数据设计，处理长时间间隔；3) 与视频视觉transformers相比，明确编码不规则时间间隔，计算效率更高；4) 与视觉状态空间模型相比，引入真实时间间隔信息，结合3D邻域融合；5) 与标准Mamba相比，显式编码真实时间间隔，添加3D邻域融合。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'Time-Aware Δt-Mamba3D通过将真实时间间隔和多尺度3D空间-时间信息整合到高效的状态空间模型中，显著提高了乳腺癌风险预测的准确性，同时保持了线性计算复杂度，为纵向医学图像分析提供了新框架。'}


### 论文摘要

Longitudinal analysis of sequential radiological images is hampered by a fundamental data challenge: how to effectively model a sequence of high-resolution images captured at irregular time intervals. This data structure contains indispensable spatial and temporal cues that current methods fail to fully exploit. Models often compromise by either collapsing spatial information into vectors or applying spatio-temporal models that are computationally inefficient and incompatible with non-uniform time steps. We address this challenge with Time-Aware $\Delta$t-Mamba3D, a novel state-space architecture adapted for longitudinal medical imaging. Our model simultaneously encodes irregular inter-visit intervals and rich spatio-temporal context while remaining computationally efficient. Its core innovation is a continuous-time selective scanning mechanism that explicitly integrates the true time difference between exams into its state transitions. This is complemented by a multi-scale 3D neighborhood fusion module that robustly captures spatio-temporal relationships. In a comprehensive breast cancer risk prediction benchmark using sequential screening mammogram exams, our model shows superior performance, improving the validation c-index by 2-5 percentage points and achieving higher 1-5 year AUC scores compared to established variants of recurrent, transformer, and state-space models. Thanks to its linear complexity, the model can efficiently process long and complex patient screening histories of mammograms, forming a new framework for longitudinal image analysis.

---

## 9. SFGFusion: Surface Fitting Guided 3D Object Detection with 4D Radar and Camera Fusion

**论文链接:** [http://arxiv.org/abs/2510.19215v1](http://arxiv.org/abs/2510.19215v1)

**作者:** Xiaozhi Li, Huijun Di, Jian Li, Feng Liu, Wei Liang

**发布时间:** 2025-10-22

**备注:** Submitted to Pattern Recognition

### GPT解析

### 总结

本文提出了一种名为SFGFusion的新型相机-4D成像雷达检测网络，通过表面拟合引导来解决3D物体检测中的多模态融合问题。

### 背景

3D物体检测对自动驾驶至关重要，4D成像雷达作为一种新兴传感器具有低成本、长距离检测和精确速度测量的优势，但其稀疏点云和低分辨率限制了物体的几何表示和跨模态融合。

### 目的

开发一种能够有效融合相机和4D成像雷达数据的方法，提高3D物体检测的准确性和可靠性。

### 方法

SFGFusion通过估计物体的二次曲面参数增强空间表示和跨模态交互，预测细粒度密集深度用于图像特征转换和伪点云生成，采用基于支柱的方法处理雷达点云，并在BEV空间中进行特征融合和检测。

### 主要发现

SFGFusion有效融合了相机和4D雷达特征，在TJ4DRadSet和view-of-delft(VoD)物体检测基准上取得了优越性能。

### 结论

基于表面拟合的SFGFusion网络能够有效解决4D成像雷达的稀疏性问题，提升多模态融合效果，提高3D物体检测性能。

### 翻译

3D物体检测对自动驾驶至关重要。作为一种新兴传感器，4D成像雷达具有低成本、长距离检测和精确速度测量的优势，使其非常适合物体检测。然而，其稀疏点云和低分辨率限制了物体的几何表示并阻碍了多模态融合。在本研究中，我们引入了SFGFusion，一种基于表面拟合引导的新型相机-4D成像雷达检测网络。通过从图像和雷达数据估计物体的二次曲面参数，显式表面拟合模型增强了空间表示和跨模态交互，实现了对细粒度密集深度更可靠的预测。预测的深度有两个用途：1)在图像分支中引导图像特征从透视视图(PV)转换为统一的鸟瞰图(BEV)用于多模态融合，提高空间映射准确性；2)在表面伪点分支中生成密集伪点云，减轻雷达点稀疏性。原始雷达点云也在单独的雷达分支中编码。这两个点云分支采用基于支柱的方法，然后将特征转换为BEV空间。最后，使用标准的2D主干和检测头从BEV特征预测物体标签和边界框。实验结果表明，SFGFusion有效融合了相机和4D雷达特征，在TJ4DRadSet和view-of-delft(VoD)物体检测基准上取得了优越性能。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决4D成像雷达点云稀疏性和低分辨率导致的物体几何表示不足，以及由此带来的多模态融合困难问题。这个问题在现实中非常重要，因为3D物体检测是自动驾驶的核心技术，而4D成像雷达具有成本低、远距离检测和精确测速的优势，能有效弥补相机在深度信息上的不足。解决这一问题可以提升自动驾驶系统的环境感知能力，增强在复杂场景下的检测精度和可靠性。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者通过分析现有多模态3D检测框架的局限性，特别是图像特征从2D到3D转换过程中因雷达点云稀疏导致的几何约束不足问题，提出了表面拟合模型作为解决方案。作者借鉴了基于图像的3D检测中的特征投影方法、基于点云的3D检测中的柱状处理方法(PointPillars)，以及多模态融合中的BEV特征融合技术，但针对4D成像雷达的特点进行了专门优化和创新设计。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是利用表面拟合模型估计物体表面深度，通过结合图像语义和雷达几何信息来增强深度预测精度，然后用这些深度信息指导图像特征转换和生成密集伪点云。整体流程包括：1)表面拟合模型融合图像和雷达信息预测物体深度；2)图像分支在深度指导下将特征从透视视图转换为鸟瞰视图；3)雷达分支处理原始4D雷达点云；4)表面伪点分支利用预测深度生成密集伪点云；5)多分支特征融合后通过检测头输出3D物体检测结果。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)提出表面拟合模型增强跨模态交互和深度估计；2)利用拟合深度指导图像特征视图变换提高空间映射精度；3)生成密集伪点云缓解雷达点云稀疏问题；4)设计针对4D雷达特点的多维特征提取方法。相比之前工作，本文专门针对4D成像雷达而非LiDAR进行优化，解决了雷达点云稀疏不规则带来的挑战，同时结合了图像语义和雷达几何信息的优势，实现了更准确的特征对齐和物体表示。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'SFGFusion通过表面拟合模型有效融合相机和4D成像雷达数据，解决了雷达点云稀疏性导致的几何表示不足问题，显著提升了3D物体检测的准确性和鲁棒性。'}


### 论文摘要

3D object detection is essential for autonomous driving. As an emerging sensor, 4D imaging radar offers advantages as low cost, long-range detection, and accurate velocity measurement, making it highly suitable for object detection. However, its sparse point clouds and low resolution limit object geometric representation and hinder multi-modal fusion. In this study, we introduce SFGFusion, a novel camera-4D imaging radar detection network guided by surface fitting. By estimating quadratic surface parameters of objects from image and radar data, the explicit surface fitting model enhances spatial representation and cross-modal interaction, enabling more reliable prediction of fine-grained dense depth. The predicted depth serves two purposes: 1) in an image branch to guide the transformation of image features from perspective view (PV) to a unified bird's-eye view (BEV) for multi-modal fusion, improving spatial mapping accuracy; and 2) in a surface pseudo-point branch to generate dense pseudo-point cloud, mitigating the radar point sparsity. The original radar point cloud is also encoded in a separate radar branch. These two point cloud branches adopt a pillar-based method and subsequently transform the features into the BEV space. Finally, a standard 2D backbone and detection head are used to predict object labels and bounding boxes from BEV features. Experimental results show that SFGFusion effectively fuses camera and 4D radar features, achieving superior performance on the TJ4DRadSet and view-of-delft (VoD) object detection benchmarks.

---

## 10. AgentSense: LLMs Empower Generalizable and Explainable Web-Based Participatory Urban Sensing

**论文链接:** [http://arxiv.org/abs/2510.19661v1](http://arxiv.org/abs/2510.19661v1)

**作者:** Xusen Guo, Mingxing Peng, Xixuan Hao, Xingchen Zou, Qiongyan Wang, Sijie Ruan, Yuxuan Liang

**发布时间:** 2025-10-22

**备注:** 13 pages, 10 pages

### GPT解析

### 总结

AgentSense是一种混合的、无需训练的框架，通过多智能体进化系统将大型语言模型集成到参与式城市感知中，解决了现有系统在多样化城市场景中泛化能力差和决策解释性不足的问题。

### 背景

基于网络的参与式城市感知已成为现代城市管理的重要方法，通过利用移动个体作为分布式传感器收集城市数据。

### 目的

开发一种能够适应动态城市条件和异构工作者偏好的城市感知系统，同时提供自然语言解释以提高透明度和信任度。

### 方法

AgentSense首先使用经典规划器生成基线解决方案，然后通过多智能体进化系统迭代优化这些解决方案，使感知任务分配适应动态变化，并生成自然语言解释。

### 主要发现

在两个大规模移动数据集和七种动态干扰上的实验表明，AgentSense在适应性和可解释性方面明显优于传统方法，且比单智能体LLM基线在性能和鲁棒性方面表现更好。

### 结论

AgentSense代表了在网络上部署自适应和可解释的城市感知系统的重要进展，为现代城市管理提供了更有效的工具。

### 翻译

基于网络的参与式城市感知已通过利用移动个体作为分布式传感器成为现代城市管理的重要方法。然而，现有的城市感知系统难以在多样化的城市场景中泛化，并且在决策过程中解释性差。在这项工作中，我们介绍了AgentSense，一个混合的、无需训练的框架，通过多智能体进化系统将大型语言模型集成到参与式城市感知中。AgentSense最初使用经典规划器生成基线解决方案，然后迭代优化它们，使感知任务分配适应动态城市条件和异构工作者偏好，同时产生自然语言解释以提高透明度和信任度。在两个大规模移动数据集和七种动态干扰上的大量实验表明，AgentSense在适应性和可解释性方面比传统方法具有明显优势。此外，与单智能体LLM基线相比，我们的方法在性能和鲁棒性方面表现更好，并提供更合理和透明的解释。这些结果表明AgentSense是在网络上部署自适应和可解释的城市感知系统的重要进展。


### 论文摘要

Web-based participatory urban sensing has emerged as a vital approach for modern urban management by leveraging mobile individuals as distributed sensors. However, existing urban sensing systems struggle with limited generalization across diverse urban scenarios and poor interpretability in decision-making. In this work, we introduce AgentSense, a hybrid, training-free framework that integrates large language models (LLMs) into participatory urban sensing through a multi-agent evolution system. AgentSense initially employs classical planner to generate baseline solutions and then iteratively refines them to adapt sensing task assignments to dynamic urban conditions and heterogeneous worker preferences, while producing natural language explanations that enhance transparency and trust. Extensive experiments across two large-scale mobility datasets and seven types of dynamic disturbances demonstrate that AgentSense offers distinct advantages in adaptivity and explainability over traditional methods. Furthermore, compared to single-agent LLM baselines, our approach outperforms in both performance and robustness, while delivering more reasonable and transparent explanations. These results position AgentSense as a significant advancement towards deploying adaptive and explainable urban sensing systems on the web.

---

## 11. Conditions for Catastrophic Forgetting in Multilingual Translation

**论文链接:** [http://arxiv.org/abs/2510.19546v1](http://arxiv.org/abs/2510.19546v1)

**作者:** Danni Liu, Jan Niehues

**发布时间:** 2025-10-22

**备注:** Multilingual Representation Learning (MRL) Workshop 2025

### GPT解析

### 总结

该研究通过系统实证研究，探讨了多语言模型微调中的灾难性遗忘现象，发现模型与数据规模比例是遗忘的主要决定因素，指令遵循能力比架构更重要，参数高效微调无明显优势，而跨语言对齐可有效减轻遗忘并促进知识迁移。

### 背景

在多语言基础模型上针对特定语言进行微调通常会导致灾难性遗忘，降低在微调中未见语言的性能。虽然这种现象有广泛记录，但文献中关于何时发生遗忘的结果是零散的。

### 目的

解决关于灾难性遗忘发生条件的模糊性，进行系统的实证研究，使用机器翻译作为测试平台，以识别多语言微调中触发灾难性遗忘的条件。

### 方法

使用机器翻译作为测试平台，进行受控实验，跨越不同的模型架构、数据规模和微调方法。

### 主要发现

模型和数据规模之间的相对规模是遗忘的主要决定因素；模型的指令遵循能力对于保留多语言知识比其架构更重要；与假设相反，参数高效微调在减轻遗忘方面没有明显优于完全微调；跨语言对齐可以减轻遗忘，同时促进对未见目标语言的积极迁移。

### 结论

跨语言对齐是一种有效的策略，可以减轻灾难性遗忘，并促进知识迁移到未见语言。

### 翻译

在多语言基础模型上针对特定语言进行微调通常会导致灾难性遗忘，降低在微调中未见语言的性能。虽然这种现象有广泛记录，但文献中关于何时发生遗忘的结果是零散的。为解决这一模糊性，我们使用机器翻译作为测试平台进行系统实证研究，以识别多语言微调中触发灾难性遗忘的条件。通过跨越不同模型架构、数据规模和微调方法的受控实验，我们揭示了模型与数据规模之间的相对比例是遗忘的主要决定因素。此外，我们证明模型的指令遵循能力对于保留多语言知识比其架构更为关键。与假设相反，参数高效微调在减轻遗忘方面并未显示出比完全微调的明显优势。最后，我们表明跨语言对齐可以减轻遗忘，同时促进对未见目标语言的积极迁移。


### 论文摘要

Fine-tuning multilingual foundation models on specific languages often induces catastrophic forgetting, degrading performance on languages unseen in fine-tuning. While this phenomenon is widely-documented, the literature presents fragmented results about when forgetting occurs. To address this ambiguity, we conduct a systematic empirical study using machine translation as a testbed to identify the conditions that trigger catastrophic forgetting in multilingual fine-tuning. Through controlled experiments across different model architectures, data scales, and fine-tuning approaches, we reveal that the relative scale between model and data size is a primary determinant of forgetting. Moreover, we demonstrate that a model's instruction-following ability is more critical for retaining multilingual knowledge than its architecture. Contrary to assumptions, parameter-efficient fine-tuning offers no clear advantage over full fine-tuning in mitigating forgetting. Lastly, we show that cross-lingual alignment can mitigate forgetting while also facilitating positive transfer to unseen target languages.

---

## 12. Which Evaluation for Which Model? A Taxonomy for Speech Model Assessment

**论文链接:** [http://arxiv.org/abs/2510.19509v1](http://arxiv.org/abs/2510.19509v1)

**作者:** Maureen de Seyssel, Eeshan Gunesh Dhekane

**发布时间:** 2025-10-22

**备注:** 57 pages (26 main, 25 appendix, 6 references)

### GPT解析

### 总结

该研究提出了一种统一的分类法，用于解决语音基础模型评估的匹配问题，通过三个正交轴对现有评估方法进行系统分类，为模型与合适评估方法的匹配提供了原则性框架，并揭示了未来基准设计的优先事项。

### 背景

语音基础模型最近在广泛任务中取得了显著能力，但其评估在不同任务和模型类型之间仍然分散，不同模型在语音处理的不同方面表现出色，因此需要不同的评估协议。

### 目的

提出一个统一的分类法，解决'哪种评估适合哪种模型'的问题，为选择、解释和扩展语音模型评估提供概念基础和实践指南。

### 方法

定义三个正交轴：测量的评估方面、尝试任务所需的模型能力、执行任务所需的任务或协议要求，沿着这些轴对现有评估和基准进行分类，涵盖表示学习、语音生成和交互式对话等领域。

### 主要发现

通过将每个评估映射到模型展示的能力和方法论需求，该分类法揭示了系统性的差距，如韵律、交互或推理覆盖有限，突显了未来基准设计的优先事项。

### 结论

该统一的分类法为模型与合适评估方法的匹配提供了原则性框架，为语音模型评估领域提供了概念基础和实践指导。

### 翻译

语音基础模型最近在广泛任务中取得了显著能力。然而，它们的评估在不同任务和模型类型之间仍然分散。不同的模型在语音处理的不同方面表现出色，因此需要不同的评估协议。本文提出了一种统一的分类法，解决'哪种评估适合哪种模型'的问题。该分类法定义了三个正交轴：测量的评估方面、尝试任务所需的模型能力、执行任务所需的任务或协议要求。我们沿着这些轴对广泛的现有评估和基准进行分类，涵盖表示学习、语音生成和交互式对话等领域。通过将每个评估映射到模型展示的能力（如语音生成、实时处理）及其方法论需求（如微调数据、人工判断），该分类法为模型与合适评估方法的匹配提供了原则性框架。它还揭示了系统性的差距，如韵律、交互或推理覆盖有限，突显了未来基准设计的优先事项。总体而言，这项工作为选择、解释和扩展语音模型评估提供了概念基础和实践指南。


### 论文摘要

Speech foundation models have recently achieved remarkable capabilities across a wide range of tasks. However, their evaluation remains disjointed across tasks and model types. Different models excel at distinct aspects of speech processing and thus require different evaluation protocols. This paper proposes a unified taxonomy that addresses the question: Which evaluation is appropriate for which model? The taxonomy defines three orthogonal axes: the \textbf{evaluation aspect} being measured, the model capabilities required to attempt the task, and the task or protocol requirements needed to perform it. We classify a broad set of existing evaluations and benchmarks along these axes, spanning areas such as representation learning, speech generation, and interactive dialogue. By mapping each evaluation to the capabilities a model exposes (e.g., speech generation, real-time processing) and to its methodological demands (e.g., fine-tuning data, human judgment), the taxonomy provides a principled framework for aligning models with suitable evaluation methods. It also reveals systematic gaps, such as limited coverage of prosody, interaction, or reasoning, that highlight priorities for future benchmark design. Overall, this work offers a conceptual foundation and practical guide for selecting, interpreting, and extending evaluations of speech models.

---

## 13. Universal Quantitative Abstraction: Categorical Duality and Logical Completeness for Probabilistic Systems

**论文链接:** [http://arxiv.org/abs/2510.19444v1](http://arxiv.org/abs/2510.19444v1)

**作者:** Nivar Anwer

**发布时间:** 2025-10-22

### GPT解析

### 总结

本文提出了一种概率系统的定量抽象统一理论，结合了范畴论、最优传输和定量模态逻辑。核心是一个具有普遍性质的规范ε-商，在所有ε-抽象中最为信息丰富且满足值损失上限。该理论建立了抽象与实现函子之间的伴随关系，揭示了度量结构与逻辑语义的范畴对偶。研究还引入了定量模态μ演算，证明了其在逻辑可表示系统中的表达完整性，并分析了接口细化下的组合性。通过在有限马尔可夫决策过程上的验证，证实了理论的收缩性、值损失界限等性质，为状态聚合和表示学习提供了数学精确的保证。

### 背景

概率系统的抽象和近似是计算机科学和人工智能中的重要问题，特别是在处理复杂系统时。现有的抽象方法往往缺乏统一的数学框架，难以保证近似的质量和性质。范畴论提供了描述系统结构和关系的强大工具，最优传输提供了度量概率空间之间距离的方法，而定量模态逻辑则允许对系统的行为进行精确描述。这些理论领域的结合为概率系统的定量抽象提供了新的可能性。

### 目的

本研究旨在构建一个统一的概率系统定量抽象理论，该理论能够：1) 提供一个信息丰富且满足值损失上限的抽象方法；2) 建立抽象与实现之间的数学关系；3) 刻画行为伪度量的性质；4) 开发表达完整的定量模态μ演算；5) 分析系统组合性；6) 为状态聚合和表示学习提供数学保证。

### 方法

本研究采用了以下方法：1) 构建具有普遍性质的规范ε-商作为核心抽象机制；2) 应用范畴论建立抽象与实现函子之间的伴随关系；3) 使用贝尔曼风格算子刻画行为伪度量并证明其不动点性质；4) 在余代数框架中证明收缩性和利普希茨性质；5) 引入定量模态μ演算并证明其表达完整性；6) 分析接口细化下的组合性质；7) 在有限马尔可夫决策过程上进行实验验证。

### 主要发现

研究的主要发现包括：1) 规范ε-商在所有ε-抽象中是最能保留信息且满足值损失上限的；2) 抽象与实现函子之间存在伴随关系，揭示了度量结构与逻辑语义的范畴对偶；3) 行为伪度量是贝尔曼风格算子的唯一不动点，具有收缩性和利普希茨性质；4) 定量模态μ演算在逻辑可表示系统中具有表达完整性，行为距离与最大逻辑偏差一致；5) 在接口细化下，抽象具有组合性质，能够清晰描述系统边界处的交互；6) 通过实验验证，理论具有收缩性、值损失界限、扰动稳定性、对抗区分性和可扩展性。

### 结论

本研究提出的概率系统定量抽象统一理论为复杂系统的抽象和近似提供了坚实的数学基础。该理论通过结合范畴论、最优传输和定量模态逻辑，建立了抽象与实现之间的严格关系，并确保了抽象的质量和性质。定量模态μ演算的表达完整性以及行为距离与逻辑偏差的一致性，为系统分析和验证提供了有力工具。实验验证证明了理论的鲁棒性和计算可行性。该框架不仅为状态聚合和表示学习提供了有原则的目标，还在随机域中为值函数近似提供了数学上精确的保证，对概率系统的建模、分析和应用具有重要意义。

### 翻译

本文提出了一种概率系统的定量抽象统一理论，该理论将范畴论、最优传输和定量模态逻辑联系起来。其核心是一个具有普遍性质的规范ε-商：在所有ε-抽象中，它是最能保留信息且满足规定值损失上限的。这种构造通过特殊伴随函子定理在抽象函子和实现函子之间诱导了一个伴随关系，揭示了度量结构和逻辑语义之间的范畴对偶。行为伪度量被刻画为贝尔曼风格算子的唯一不动点，并在余代数框架中证明了其收缩性和利普希茨性质。引入了定量模态μ演算并证明其在逻辑可表示系统中具有表达完整性，使得行为距离与最大逻辑偏差一致。分析了在接口细化下的组合性，阐明了抽象如何在系统边界处交互。在有限马尔可夫决策过程上的精确验证套件证实了收缩性、值损失界限、扰动下的稳定性、对抗区分性和可扩展性，展示了鲁棒性和计算可行性。由此产生的框架为状态聚合和表示学习提供了有原则的目标，并在随机域中为值函数近似提供了数学上精确的保证。


### 论文摘要

A unified theory of quantitative abstraction is presented for probabilistic systems that links category theory, optimal transport, and quantitative modal logic. At its core is a canonical $ \varepsilon $-quotient endowed with a universal property: among all $ \varepsilon $-abstractions, it is the most informative one that respects a prescribed bound on value loss. This construction induces an adjunction between abstraction and realization functors $ (Q_{\varepsilon} \dashv R_{\varepsilon}) $, established via the Special Adjoint Functor Theorem, revealing a categorical duality between metric structure and logical semantics. A behavioral pseudometric is characterized as the unique fixed point of a Bellman-style operator, with contraction and Lipschitz properties proved in a coalgebraic setting. A quantitative modal $ \mu $-calculus is introduced and shown to be expressively complete for logically representable systems, so that behavioral distance coincides with maximal logical deviation. Compositionality under interface refinement is analyzed, clarifying how abstractions interact across system boundaries. An exact validation suite on finite Markov decision processes corroborates the contraction property, value-loss bounds, stability under perturbation, adversarial distinguishability, and scalability, demonstrating both robustness and computational feasibility. The resulting framework provides principled targets for state aggregation and representation learning, with mathematically precise guarantees for value-function approximation in stochastic domains.

---

## 14. Learning Noise-Resilient and Transferable Graph-Text Alignment via Dynamic Quality Assessment

**论文链接:** [http://arxiv.org/abs/2510.19384v1](http://arxiv.org/abs/2510.19384v1)

**作者:** Yuhang Liu, Minglai Shao, Zengyi Wo, Yunlong Chu, Bing Hao, Shengzhong Liu, Ruijie Wang, Jianxin Li

**发布时间:** 2025-10-22

### GPT解析

### 总结

本文提出了ADAligner，一个动态、质量感知的图文本对齐框架，解决了现有CLIP风格图文本对齐器在处理多对多关系和适应不同数据质量方面的局限性。ADAligner能够根据监督质量在表达性的多对多和保守的一对一目标之间动态调整，在多个任务上表现出色，并具有更强的鲁棒性和更快的预训练速度。

### 背景

在文本属性图上预训练图基础模型对于搜索、推荐和知识发现等网络规模应用至关重要。然而，现有的CLIP风格图文本对齐器面临两个关键限制：假设节点和文本之间存在严格的一对一对应关系，忽略了现实世界图中的固有多对多关系；以及依赖于静态对齐目标，无法适应不同的数据质量，在有噪声监督下变得脆弱。

### 目的

解决现有图文本对齐器的局限性，提出一个动态、质量感知的图文本对齐框架，能够根据监督质量在表达性的多对多和保守的一对一目标之间动态调整。

### 方法

提出了ADAligner框架，实时估计批次级别的对齐可靠性，并相应地调整优化过程：当监督干净时，促进软的、子图级别的多对多对齐；在噪声下，通过动态过滤低置信度配对来强调可靠的一对一对齐。理论上证明了这种动态机制形成一个稳定的负反馈过程，确保收敛性和鲁棒性。

### 主要发现

在九个不同的TAG数据集上的实验表明，ADAligner在零样本/少样本节点分类、链接预测和跨模态检索任务上一致优于先前的图文本对齐器。在有噪声监督下保持强大的鲁棒性，与多模态基线相比，预训练速度加快约2到3倍。

### 结论

ADAligner为现实网络环境中的图文本表示学习建立了一个可扩展和可靠的基础。

### 翻译

在文本属性图上预训练图基础模型对于搜索、推荐和知识发现等网络规模应用至关重要。然而，现有的CLIP风格图文本对齐器面临两个关键限制：它们假设节点和文本之间存在严格的一对一对应关系，忽略了现实世界图中的固有多对多关系；并且它们依赖于静态对齐目标，无法适应不同的数据质量，在有噪声监督下变得脆弱。总之，这些限制暴露了一个核心困境：拥抱表达性的多对多对齐会放大噪声，而恢复到严格的一对一策略则会牺牲语义多样性，无法处理本质上不匹配的配对。为了应对这些挑战，我们提出了ADAligner，一个动态、质量感知的图文本对齐框架，根据监督质量在表达性的多对多和保守的一对一目标之间动态调整。ADAligner实时估计批次级别的对齐可靠性，并相应地调整其优化过程，在监督干净时促进软的、子图级别的多对多对齐，同时在噪声下通过动态过滤低置信度配对来强调可靠的一对一对齐。理论上，我们证明这种动态机制形成一个稳定的负反馈过程，确保收敛性和鲁棒性。在九个不同的TAG数据集上的综合实验表明，ADAligner在零样本/少样本节点分类、链接预测和跨模态检索任务上一致地优于先前的图文本对齐器。它在有噪声监督下保持强大的鲁棒性，与多模态基线相比，预训练速度加快约2到3倍，为现实网络环境中的图文本表示学习建立了一个可扩展和可靠的基础。


### 论文摘要

Pre-training Graph Foundation Models (GFMs) on text-attributed graphs (TAGs) is central to web-scale applications such as search, recommendation, and knowledge discovery. However, existing CLIP-style graph-text aligners face two key limitations: they assume strict one-to-one correspondences between nodes and texts, overlooking the inherent many-to-many relations in real-world graphs; and they rely on static alignment objectives that cannot adapt to varying data quality, making them brittle under noisy supervision. Together, these limitations expose a core dilemma: embracing expressive many-to-many alignment amplifies noise, while reverting to strict one-to-one strategies sacrifices semantic diversity and fails to handle inherently mismatched pairs. To address these challenges, we propose ADAligner, a dynamic, quality-aware graph-text alignment framework that dynamically adjusts between expressive many-to-many and conservative one-to-one objectives according to supervision quality. ADAligner estimates batch-level alignment reliability in real time and adapts its optimization accordingly, promoting soft, subgraph-level many-to-many alignment when supervision is clean, while emphasizing reliable one-to-one alignment by dynamically filtering low-confidence pairs under noise. Theoretically, we prove that this dynamic mechanism forms a stable negative feedback process, ensuring convergence and robustness. Comprehensive experiments on nine diverse TAG datasets demonstrate that ADAligner consistently outperforms prior graph-text aligners on zero-/few-shot node classification, link prediction and cross-modal retrieval tasks. It maintains strong robustness under noisy supervision and accelerates pre-training by approximately 2 to 3 times compared to multimodal baselines, establishing a scalable and reliable foundation for graph-text representation learning in real-world web environments.

---

## 15. From Newborn to Impact: Bias-Aware Citation Prediction

**论文链接:** [http://arxiv.org/abs/2510.19246v1](http://arxiv.org/abs/2510.19246v1)

**作者:** Mingfei Lu, Mengjia Wu, Jiawei Xu, Weikai Li, Feng Liu, Ying Ding, Yizhou Sun, Jie Lu, Yi Zhang

**发布时间:** 2025-10-22

### GPT解析

### 总结

本研究提出了一种偏差感知引用预测框架，通过多智能体特征提取和鲁棒图表示学习，解决了新生论文引用预测中的两个关键研究空白，实验证明其有效性。

### 背景

引用动态是获取研究影响的关键，支撑研究评估、学术推荐和知识扩散研究。引用预测对新生论文尤为重要，因为在没有引用信号和高度长尾分布的情况下，必须进行早期评估。

### 目的

解决两个关键研究空白：一是对科学影响的隐含因素建模不足，导致依赖粗略代理指标；二是缺乏偏差感知学习，无法在低引用论文上提供稳定预测。

### 方法

提出偏差感知引用预测框架，结合多智能体特征提取和鲁棒图表示学习。多智能体图共学习模块从元数据和外部资源中提取细粒度可解释信号，并与异构网络嵌入融合；同时采用鲁棒机制，包括两阶段前向过程、GroupDRO优化和正则化头。

### 主要发现

在两个真实世界数据集上的综合实验证明了所提模型的有效性。模型实现了约百分之十三的错误指标降低和百分之五点五的排名指标显著改善。

### 结论

提出的偏差感知引用预测框架能够有效解决现有研究空白，提高引用预测的准确性和稳定性。

### 翻译

作为获取研究影响的关键，引用动态支撑着研究评估、学术推荐和知识扩散研究。引用预测对新生论文尤为重要，因为在没有引用信号和高度长尾分布的情况下，必须进行早期评估。我们确定了两个关键研究空白：一是对科学影响的隐含因素建模不足，导致依赖粗略代理指标；二是缺乏偏差感知学习，无法在低引用论文上提供稳定预测。我们通过提出偏差感知引用预测框架来解决这些空白，该框架结合了多智能体特征提取和鲁棒图表示学习。首先，多智能体图共学习模块从元数据和外部资源中推导出细粒度、可解释的信号，如可重复性、协作网络和文本质量，并将它们与异构网络嵌入融合，即使在缺乏早期引用信号的情况下也能提供丰富的监督。其次，我们加入了一套鲁棒机制：一个将显性因素通过中间曝光估计路由的两阶段前向过程，用于优化跨环境最坏情况组风险的GroupDRO，以及在单调性和平滑性约束下对可控因素执行假设分析的正则化头。在两个真实世界数据集上的综合实验证明了我们提出的模型的有效性。具体而言，我们的模型在错误指标上实现了约百分之十三的降低，在排名指标上比基线方法有显著的百分之五点五的改善。


### 论文摘要

As a key to accessing research impact, citation dynamics underpins research evaluation, scholarly recommendation, and the study of knowledge diffusion. Citation prediction is particularly critical for newborn papers, where early assessment must be performed without citation signals and under highly long-tailed distributions. We identify two key research gaps: (i) insufficient modeling of implicit factors of scientific impact, leading to reliance on coarse proxies; and (ii) a lack of bias-aware learning that can deliver stable predictions on lowly cited papers. We address these gaps by proposing a Bias-Aware Citation Prediction Framework, which combines multi-agent feature extraction with robust graph representation learning. First, a multi-agent x graph co-learning module derives fine-grained, interpretable signals, such as reproducibility, collaboration network, and text quality, from metadata and external resources, and fuses them with heterogeneous-network embeddings to provide rich supervision even in the absence of early citation signals. Second, we incorporate a set of robust mechanisms: a two-stage forward process that routes explicit factors through an intermediate exposure estimate, GroupDRO to optimize worst-case group risk across environments, and a regularization head that performs what-if analyses on controllable factors under monotonicity and smoothness constraints. Comprehensive experiments on two real-world datasets demonstrate the effectiveness of our proposed model. Specifically, our model achieves around a 13% reduction in error metrics (MALE and RMSLE) and a notable 5.5% improvement in the ranking metric (NDCG) over the baseline methods.

---

## 16. No Intelligence Without Statistics: The Invisible Backbone of Artificial Intelligence

**论文链接:** [http://arxiv.org/abs/2510.19212v1](http://arxiv.org/abs/2510.19212v1)

**作者:** Ernest Fokoué

**发布时间:** 2025-10-22

**备注:** 37 pages, 6 figures

### GPT解析

### 总结

人工智能的理论和方法基础实际上是统计学，而非仅仅来自计算机科学。统计学为机器学习和现代AI提供了不可或缺的基础。

### 背景

人工智能的快速发展通常被描述为来自计算机科学和工程的革命，但这种描述掩盖了一个基本事实：AI的理论和方法核心一直是统计学。

### 目的

系统性地论证统计学为机器学习和现代AI提供了不可或缺的基础，并呼吁教育、研究和实践重新拥抱这一统计学基础。

### 方法

将AI分解为九个基础支柱（推断、密度估计、序列学习、泛化、表示学习、可解释性、因果性、优化和统一），展示每个支柱都建立在百年统计原理之上。

### 主要发现

AI的九个基础支柱都建立在统计原理之上；从假设检验和估计的推断框架到聚类和生成式AI的密度估计根源；从启发循环网络的时间序列分析到提供真正理解的因果模型；统计学提供了理论框架、不确定性量化等'大脑'功能，而计算机科学提供了可扩展算法和硬件等'肌肉'功能。

### 结论

承认统计学的基础对于开发更强大、可解释和值得信赖的智能系统是必要的步骤。没有统计学习就没有机器学习；没有统计思维就没有人工智能。

### 翻译

人工智能的迅速崛起通常被描述为一场源于计算机科学和工程学的革命。然而，这种叙事掩盖了一个基本事实：AI的理论和方法核心，并且一直是，统计学的。本文系统性地论证统计学领域为机器学习和现代AI提供了不可或缺的基础。我们将AI分解为九个基础支柱——推断、密度估计、序列学习、泛化、表示学习、可解释性、因果性、优化和统一——证明每一个都建立在百年统计原理之上。从支撑模型评估的假设检验和估计推断框架，到聚类和生成式AI的密度估计根源；从启发循环网络的时间序列分析到提供真正理解的因果模型，我们追溯了一条不间断的统计谱系。在庆祝推动现代AI的计算引擎的同时，我们认为统计学提供了'大脑'——理论框架、不确定性量化和推断目标——而计算机科学提供了'肌肉'——可扩展算法和硬件。认识到这一统计基础不仅仅是一个学术练习，而是开发更强大、可解释和值得信赖的智能系统的必要步骤。我们呼吁教育、研究和实践重新拥抱这一统计基础。忽视这些根基可能会构建一个脆弱的未来；拥抱它们才是通向真正智能机器的道路。没有统计学习就没有机器学习；没有统计思维就没有人工智能。


### 论文摘要

The rapid ascent of artificial intelligence (AI) is often portrayed as a revolution born from computer science and engineering. This narrative, however, obscures a fundamental truth: the theoretical and methodological core of AI is, and has always been, statistical. This paper systematically argues that the field of statistics provides the indispensable foundation for machine learning and modern AI. We deconstruct AI into nine foundational pillars-Inference, Density Estimation, Sequential Learning, Generalization, Representation Learning, Interpretability, Causality, Optimization, and Unification-demonstrating that each is built upon century-old statistical principles. From the inferential frameworks of hypothesis testing and estimation that underpin model evaluation, to the density estimation roots of clustering and generative AI; from the time-series analysis inspiring recurrent networks to the causal models that promise true understanding, we trace an unbroken statistical lineage. While celebrating the computational engines that power modern AI, we contend that statistics provides the brain-the theoretical frameworks, uncertainty quantification, and inferential goals-while computer science provides the brawn-the scalable algorithms and hardware. Recognizing this statistical backbone is not merely an academic exercise, but a necessary step for developing more robust, interpretable, and trustworthy intelligent systems. We issue a call to action for education, research, and practice to re-embrace this statistical foundation. Ignoring these roots risks building a fragile future; embracing them is the path to truly intelligent machines. There is no machine learning without statistical learning; no artificial intelligence without statistical thought.

---

## 17. An Encode-then-Decompose Approach to Unsupervised Time Series Anomaly Detection on Contaminated Training Data--Extended Version

**论文链接:** [http://arxiv.org/abs/2510.18998v1](http://arxiv.org/abs/2510.18998v1)

**作者:** Buang Zhang, Tung Kieu, Xiangfei Qiu, Chenjuan Guo, Jilin Hu, Aoying Zhou, Christian S. Jensen, Bin Yang

**发布时间:** 2025-10-21

**备注:** 15 pages. An extended version of "An Encode-then-Decompose Approach  to Unsupervised Time Series Anomaly Detection on Contaminated Training Data"  accepted at ICDE 2026

### GPT解析

### 总结

该论文提出了一种新的编码-分解范式和基于互信息的度量方法，用于时间序列异常检测，提高了对污染时间序列的鲁棒性，并在多个基准测试上取得了优异性能。

### 背景

时间序列异常检测在现代大规模系统中至关重要，应用于多个领域分析和监控系统运行。无监督方法因不需要异常标签而受到广泛关注，避免了高成本并具有更广泛的应用。

### 目的

解决自动编码器学习到的表示对训练时间序列中的异常敏感导致准确性降低的问题，提高方法在污染数据上的鲁棒性。

### 方法

提出编码-分解范式，将编码表示分解为稳定表示和辅助表示；同时提出基于互信息的新度量方法替代重构误差来识别异常。

### 主要发现

在八个常用的多变量和单变量时间序列基准测试上展示了具有竞争力或最先进的性能，对不同污染比例的时间序列表现出鲁棒性。

### 结论

新方法通过分解编码表示和使用互信息度量，有效提高了时间序列异常检测的准确性和鲁棒性，特别是在训练数据存在异常污染的情况下。

### 翻译

时间序列异常检测在现代大规模系统中很重要，并应用于各种领域以分析和监控不同系统的运行。无监督方法引起了广泛关注，因为它们在训练期间不需要异常标签，从而避免了潜在的高成本并具有更广泛的应用。其中，自动编码器受到了广泛关注。它们使用来自压缩表示的重构误差来定义异常分数。然而，自动编码器学习到的表示对训练时间序列中的异常敏感，导致准确性降低。我们提出了一种新颖的编码-分解范式，将编码表示分解为稳定表示和辅助表示，从而在使用污染时间序列进行训练时增强鲁棒性。此外，我们提出了一种基于互信息的新指标来替代重构误差以识别异常。我们的提案在八个常用的多变量和单变量时间序列基准测试上展示了具有竞争力或最先进的性能，并对具有不同污染比例的时间序列表现出鲁棒性。


### 论文摘要

Time series anomaly detection is important in modern large-scale systems and is applied in a variety of domains to analyze and monitor the operation of diverse systems. Unsupervised approaches have received widespread interest, as they do not require anomaly labels during training, thus avoiding potentially high costs and having wider applications. Among these, autoencoders have received extensive attention. They use reconstruction errors from compressed representations to define anomaly scores. However, representations learned by autoencoders are sensitive to anomalies in training time series, causing reduced accuracy. We propose a novel encode-then-decompose paradigm, where we decompose the encoded representation into stable and auxiliary representations, thereby enhancing the robustness when training with contaminated time series. In addition, we propose a novel mutual information based metric to replace the reconstruction errors for identifying anomalies. Our proposal demonstrates competitive or state-of-the-art performance on eight commonly used multi- and univariate time series benchmarks and exhibits robustness to time series with different contamination ratios.

---

## 18. SBAN: A Framework \& Multi-Dimensional Dataset for Large Language Model Pre-Training and Software Code Mining

**论文链接:** [http://arxiv.org/abs/2510.18936v1](http://arxiv.org/abs/2510.18936v1)

**作者:** Hamed Jelodar, Mohammad Meymani, Samita Bai, Roozbeh Razavi-Far, Ali A. Ghorbani

**发布时间:** 2025-10-21

### GPT解析

### 总结

这篇论文介绍了一个名为SBAN的大规模多维度数据集，用于推进大型语言模型在软件代码分析方面的预训练和评估。

### 背景

软件代码分析领域需要大规模、多模态的数据集来支持大型语言模型的训练和评估，特别是在安全分析和软件理解方面。

### 目的

创建一个包含源代码、二进制代码、汇编指令和自然语言描述的多维度数据集，以支持跨表示学习、软件语义理解和自动化恶意软件检测等研究。

### 方法

构建了一个包含超过300万个样本的数据集，其中包括290万个良性样本和672,000个恶意软件样本，每个样本都通过四个互补层表示：二进制代码、汇编指令、自然语言描述和源代码。

### 主要发现

这种独特的多模态结构支持跨表示学习研究，并且可以应用于安全分析、代码翻译、代码解释和其他涉及异构数据的软件挖掘任务。

### 结论

SBAN数据集通过桥接低级机器表示和高级人类语义，为构建能够推理代码的智能系统提供了坚实的基础，为挖掘软件行为、改进安全分析和增强大型语言模型在软件代码挖掘方面的能力开辟了新的机会。

### 翻译

这篇论文介绍了SBAN（源代码、二进制代码、汇编指令和自然语言描述），这是一个大规模、多维度数据集，旨在推进大型语言模型在软件代码分析方面的预训练和评估。SBAN包含超过300万个样本，其中包括290万个良性样本和672,000个恶意软件样本，每个样本都通过四个互补层表示：二进制代码、汇编指令、自然语言描述和源代码。这种独特的多模态结构支持跨表示学习研究、软件语义理解和自动化恶意软件检测。除了安全应用外，SBAN还支持更广泛的任务，如代码翻译、代码解释和其他涉及异构数据的软件挖掘任务。它特别适合深度模型的可扩展训练，包括变压器和其他大型语言模型架构。通过桥接低级机器表示和高级人类语义，SBAN为构建能够推理代码的智能系统提供了坚实的基础。我们相信，这个数据集为挖掘软件行为、改进安全分析和增强大型语言模型在软件代码挖掘的预训练和微调任务方面的能力开辟了新的机会。


### 论文摘要

This paper introduces SBAN (Source code, Binary, Assembly, and Natural Language Description), a large-scale, multi-dimensional dataset designed to advance the pre-training and evaluation of large language models (LLMs) for software code analysis. SBAN comprises more than 3 million samples, including 2.9 million benign and 672,000 malware respectively, each represented across four complementary layers: binary code, assembly instructions, natural language descriptions, and source code. This unique multimodal structure enables research on cross-representation learning, semantic understanding of software, and automated malware detection. Beyond security applications, SBAN supports broader tasks such as code translation, code explanation, and other software mining tasks involving heterogeneous data. It is particularly suited for scalable training of deep models, including transformers and other LLM architectures. By bridging low-level machine representations and high-level human semantics, SBAN provides a robust foundation for building intelligent systems that reason about code. We believe that this dataset opens new opportunities for mining software behavior, improving security analytics, and enhancing LLM capabilities in pre-training and fine-tuning tasks for software code mining.

---

## 19. A flexible framework for structural plasticity in GPU-accelerated sparse spiking neural networks

**论文链接:** [http://arxiv.org/abs/2510.19764v1](http://arxiv.org/abs/2510.19764v1)

**作者:** James C. Knight, Johanna Senk, Thomas Nowotny

**发布时间:** 2025-10-22

**备注:** 22 pages, 9 figures, 2 tables

### GPT解析

### 总结

本文提出了一种新的灵活框架，用于实现GPU加速的结构可塑性规则，展示了如何训练高效稀疏的脉冲神经网络分类器并学习拓扑图，稀疏模型可比密集模型训练速度快10倍。

### 背景

大多数人工神经网络和生物大脑学习研究集中在突触可塑性上，而生物大脑中的结构可塑性（创建和移除连接）对有效学习、损伤恢复和资源优化同样重要。尽管受此启发，机器学习中常使用剪枝移除弱连接，但现有框架针对密集连接优化，无法降低大型模型的训练成本。

### 目的

开发一种支持结构可塑性规则的GPU加速框架，用于训练高效稀疏的SNN分类器，并在无监督学习背景下实现拓扑图形成，探索稀疏性的计算优势。

### 方法

基于GeNN模拟器，使用e-prop监督学习规则和DEEP R训练稀疏SNN分类器，然后在无监督学习场景中应用该框架学习拓扑图。

### 主要发现

稀疏分类器比基准密集模型训练时间减少高达10倍，同时通过DEEP R重连线保持与原始模型相当的性能；在比实时更快的模拟中成功展示了拓扑图形成，提供了连接演变的见解，并测量了模拟速度与网络规模的关系。

### 结论

该框架使研究人员能够探索网络结构和神经通信中的稀疏性维持，以及在各种神经形态应用中稀疏性的计算优势。

### 翻译

关于人工神经网络训练和生物大脑学习建模的大多数研究都集中在突触可塑性上，其中学习等同于改变现有连接的强度。然而，在生物大脑中，结构可塑性——创建新连接和移除其他连接——同样重要，不仅对有效学习至关重要，还有助于从损伤中恢复和优化资源使用。受结构可塑性启发，剪枝常用于机器学习以从训练好的模型中移除弱连接，从而降低推理的计算需求。然而，通常用于基于反向传播训练ANN和SNN的机器学习框架针对密集连接进行了优化，这意味着剪枝无法帮助降低不断增长的模型的训练成本。GeNN模拟器已经支持稀疏SNN的高效GPU加速模拟，用于计算神经科学和机器学习。在这里，我们提出了一个新的灵活框架，用于实现GPU加速的结构可塑性规则，并首先使用e-prop监督学习规则和DEEP R训练高效稀疏的SNN分类器，然后在无监督学习背景下学习拓扑图。与基准密集模型相比，我们的稀疏分类器将训练时间减少了高达10倍，而DEEP R重连线使它们能够与原始模型一样好地执行。我们在比实时更快的模拟中展示了拓扑图的形成，提供了连接演变的见解，并测量了模拟速度与网络规模的关系。所提出的框架将使进一步研究能够在网络结构和神经通信中实现和保持稀疏性，以及探索稀疏性在各种神经形态应用中的计算优势。


### 论文摘要

The majority of research in both training Artificial Neural Networks (ANNs) and modeling learning in biological brains focuses on synaptic plasticity, where learning equates to changing the strength of existing connections. However, in biological brains, structural plasticity - where new connections are created and others removed - is also vital, not only for effective learning but also for recovery from damage and optimal resource usage. Inspired by structural plasticity, pruning is often used in machine learning to remove weak connections from trained models to reduce the computational requirements of inference. However, the machine learning frameworks typically used for backpropagation-based training of both ANNs and Spiking Neural Networks (SNNs) are optimized for dense connectivity, meaning that pruning does not help reduce the training costs of ever-larger models. The GeNN simulator already supports efficient GPU-accelerated simulation of sparse SNNs for computational neuroscience and machine learning. Here, we present a new flexible framework for implementing GPU-accelerated structural plasticity rules and demonstrate this first using the e-prop supervised learning rule and DEEP R to train efficient, sparse SNN classifiers and then, in an unsupervised learning context, to learn topographic maps. Compared to baseline dense models, our sparse classifiers reduce training time by up to 10x while the DEEP R rewiring enables them to perform as well as the original models. We demonstrate topographic map formation in faster-than-realtime simulations, provide insights into the connectivity evolution, and measure simulation speed versus network size. The proposed framework will enable further research into achieving and maintaining sparsity in network structure and neural communication, as well as exploring the computational benefits of sparsity in a range of neuromorphic applications.

---

## 20. Study of Training Dynamics for Memory-Constrained Fine-Tuning

**论文链接:** [http://arxiv.org/abs/2510.19675v1](http://arxiv.org/abs/2510.19675v1)

**作者:** Aël Quélennec, Nour Hezbri, Pavlo Mozharovskyi, Van-Tam Nguyen, Enzo Tartaglione

**发布时间:** 2025-10-22

### GPT解析

### 总结

论文提出了TraDy，一种内存高效的深度神经网络迁移学习方案，通过动态通道选择和层重要性预判实现严格内存约束下的高性能训练。

### 背景

随着深度神经网络模型规模不断增大，而部署环境对资源有严格限制，内存高效训练变得越来越重要。

### 目的

开发一种能够在严格内存约束下实现高效训练的深度神经网络训练方法。

### 方法

TraDy利用两个关键洞察：更新的层重要性依赖于架构且可预先确定；动态随机通道选择相比静态方法能提供更好的梯度近似。引入动态通道选择方法，在预选层内周期性地随机重新采样通道。

### 主要发现

TraDy在各种下游任务和架构上取得最先进性能；实现高达99%的激活稀疏性；实现95%的权重导数稀疏性；权重导数计算的计算量减少97%。

### 结论

TraDy是一种有效的内存高效训练方法，能够在保持性能的同时显著减少内存使用和计算需求。

### 翻译

随着模型规模扩大而部署环境施加严格的资源限制，深度神经网络的内存高效训练变得越来越重要。我们提出了TraDy，一种利用两个关键洞察的新型迁移学习方案：更新的层重要性依赖于架构且可预先确定，而动态随机通道选择相比静态方法能提供更好的梯度近似。我们引入了一种动态通道选择方法，在预选层内周期性地随机重新采样通道。大量实验表明，TraDy在各种下游任务和架构上取得了最先进性能，同时保持严格的内存约束，实现了高达99%的激活稀疏性、95%的权重导数稀疏性，以及97%的权重导数计算FLOPs减少。


### 论文摘要

Memory-efficient training of deep neural networks has become increasingly important as models grow larger while deployment environments impose strict resource constraints. We propose TraDy, a novel transfer learning scheme leveraging two key insights: layer importance for updates is architecture-dependent and determinable a priori, while dynamic stochastic channel selection provides superior gradient approximation compared to static approaches. We introduce a dynamic channel selection approach that stochastically resamples channels between epochs within preselected layers. Extensive experiments demonstrate TraDy achieves state-of-the-art performance across various downstream tasks and architectures while maintaining strict memory constraints, achieving up to 99% activation sparsity, 95% weight derivative sparsity, and 97% reduction in FLOPs for weight derivative computation.

---

## 21. Transfer Learning Beyond the Standard Model

**论文链接:** [http://arxiv.org/abs/2510.19168v1](http://arxiv.org/abs/2510.19168v1)

**作者:** Veena Krishnaraj, Adrian E. Bayer, Christian Kragh Jespersen, Peter Melchior

**发布时间:** 2025-10-22

**备注:** 4+8 pages, 7 figures. Accepted at NeurIPS 2025 Workshop: Machine  Learning and the Physical Sciences

### GPT解析

### 总结

该研究展示了如何通过迁移学习减少宇宙学模拟成本，使用标准ΛCDM模型预训练，然后在各种超越ΛCDM场景上微调，发现预训练可以显著减少所需模拟数量，但也存在负迁移风险，特别是当参数间存在强物理简并时。

### 背景

机器学习能够实现强大的宇宙学推断，但通常需要大量高保真模拟来覆盖多种宇宙学模型，这带来了高昂的计算成本。

### 目的

探索通过迁移学习重用不同宇宙学模型间的知识，以减少模拟成本并提高推断效率。

### 方法

在标准ΛCDM宇宙学模型上进行预训练，然后在各种超越ΛCDM场景(包括大质量中微子、修正引力、原始非高斯性)上进行微调，并测试包含瓶颈结构的不同迁移架构。

### 主要发现

预训练可以在使用显著更少的超越ΛCDM模拟的情况下实现推断；当ΛCDM和超越ΛCDM参数之间存在强物理简并时，可能会发生负迁移；包含瓶颈结构的迁移架构提供了最佳性能。

### 结论

预训练可以加速宇宙学推断过程，但也可能阻碍对新物理的学习，基础模型方法在物理学应用中既带来机会也存在潜在陷阱。

### 翻译

机器学习能够实现强大的宇宙学推断，但通常需要覆盖多种宇宙学模型的大量高保真模拟。迁移学习提供了一种通过跨模型重用知识来减少模拟成本的方法。我们展示了在标准宇宙学模型ΛCDM上进行预训练，并在各种超越ΛCDM场景(包括大质量中微子、修正引力和原始非高斯性)上进行微调，可以使用显著更少的超越ΛCDM模拟实现推断。然而，我们也表明当ΛCDM和超越ΛCDM参数之间存在强物理简并时，可能会发生负迁移。我们考虑了各种迁移架构，发现包含瓶颈结构的架构提供最佳性能。我们的研究结果阐明了基础模型方法在物理学中的机会和陷阱：预训练可以加速推断，但也可能阻碍对新物理的学习。


### 论文摘要

Machine learning enables powerful cosmological inference but typically requires many high-fidelity simulations covering many cosmological models. Transfer learning offers a way to reduce the simulation cost by reusing knowledge across models. We show that pre-training on the standard model of cosmology, $\Lambda$CDM, and fine-tuning on various beyond-$\Lambda$CDM scenarios -- including massive neutrinos, modified gravity, and primordial non-Gaussianities -- can enable inference with significantly fewer beyond-$\Lambda$CDM simulations. However, we also show that negative transfer can occur when strong physical degeneracies exist between $\Lambda$CDM and beyond-$\Lambda$CDM parameters. We consider various transfer architectures, finding that including bottleneck structures provides the best performance. Our findings illustrate the opportunities and pitfalls of foundation-model approaches in physics: pre-training can accelerate inference, but may also hinder learning new physics.

---

## 22. Rethinking Hebbian Principle: Low-Dimensional Structural Projection for Unsupervised Learning

**论文链接:** [http://arxiv.org/abs/2510.14810v2](http://arxiv.org/abs/2510.14810v2)

**作者:** Shikuang Deng, Jiayuan Zhang, Yuhang Wu, Ting Chen, Shi Gu

**发布时间:** 2025-10-16

### GPT解析

### 总结

本研究提出了一种名为结构投影Hebbian表示（SPHeRe）的新型无监督学习方法，通过整合正交性和结构信息保留，解决了传统Hebbian学习在机器学习中的局限性。实验表明，该方法在图像分类、持续学习、迁移学习和图像重建任务中均表现出色，证明了Hebbian无监督学习在现代深度学习框架中的竞争力和潜力。

### 背景

Hebbian学习是一种生物学原理，描述了神经元如何通过重复刺激调整其连接。然而，在机器学习中应用时，由于连接更新不受约束且缺乏反馈介导，它存在严重问题，限制了其在复杂网络架构和任务中的有效扩展。

### 目的

解决Hebbian学习在机器学习中的局限性，提出一种能够有效扩展到复杂网络架构和任务的无监督学习方法。

### 方法

引入结构投影Hebbian表示（SPHeRe），一种新的无监督学习方法，通过局部的辅助非线性块整合正交性和结构信息保留。结构信息保留的损失通过辅助的轻量级投影反向传播到输入，充当反馈介导，正交性约束则考虑了更新幅度的有界性。

### 主要发现

SPHeRe在CIFAR-10、CIFAR-100和Tiny-ImageNet等标准图像分类基准测试上，在无监督突触可塑性方法中达到了最先进的性能。该方法在持续学习和迁移学习场景中表现出强大的有效性，图像重建任务显示了所提取特征的鲁棒性和泛化能力。

### 结论

Hebbian无监督学习规则在现代深度学习框架中具有竞争力和潜力，展示了不依赖于严格反向传播的高效和生物启发式学习算法的可能性。

### 翻译

Hebbian学习是一种生物学原理，直观地描述了神经元如何通过重复刺激来调整其连接。然而，当应用于机器学习时，由于连接更新不受约束且缺乏反馈介导，它存在严重问题。这些缺点限制了其有效扩展到复杂的网络架构和任务。为此，我们在这里引入了结构投影Hebbian表示（SPHeRe），一种新的无监督学习方法，它通过一个局部的辅助非线性块整合了正交性和结构信息保留。结构信息保留的损失通过一个辅助的轻量级投影反向传播到输入，该投影在概念上充当反馈介导，而正交性约束则考虑了更新幅度的有界性。大量的实验结果表明，SPHeRe在CIFAR-10、CIFAR-100和Tiny-ImageNet等标准图像分类基准测试中，在无监督突触可塑性方法中达到了最先进的性能。此外，该方法在持续学习和迁移学习场景中表现出强大的有效性，图像重建任务显示了所提取特征的鲁棒性和泛化能力。这项工作证明了Hebbian无监督学习规则在现代深度学习框架中的竞争力和潜力，展示了不依赖于严格反向传播的高效和生物启发式学习算法的可能性。我们的代码可在https://github.com/brain-intelligence-lab/SPHeRe获取。


### 论文摘要

Hebbian learning is a biological principle that intuitively describes how neurons adapt their connections through repeated stimuli. However, when applied to machine learning, it suffers serious issues due to the unconstrained updates of the connections and the lack of accounting for feedback mediation. Such shortcomings limit its effective scaling to complex network architectures and tasks. To this end, here we introduce the Structural Projection Hebbian Representation (SPHeRe), a novel unsupervised learning method that integrates orthogonality and structural information preservation through a local auxiliary nonlinear block. The loss for structural information preservation backpropagates to the input through an auxiliary lightweight projection that conceptually serves as feedback mediation while the orthogonality constraints account for the boundedness of updating magnitude. Extensive experimental results show that SPHeRe achieves SOTA performance among unsupervised synaptic plasticity approaches on standard image classification benchmarks, including CIFAR-10, CIFAR-100, and Tiny-ImageNet. Furthermore, the method exhibits strong effectiveness in continual learning and transfer learning scenarios, and image reconstruction tasks show the robustness and generalizability of the extracted features. This work demonstrates the competitiveness and potential of Hebbian unsupervised learning rules within modern deep learning frameworks, demonstrating the possibility of efficient and biologically inspired learning algorithms without the strong dependence on strict backpropagation. Our code is available at https://github.com/brain-intelligence-lab/SPHeRe.

---

## 23. $\nabla$-SDF: Learning Euclidean Signed Distance Functions Online with Gradient-Augmented Octree Interpolation and Neural Residual

**论文链接:** [http://arxiv.org/abs/2510.18999v1](http://arxiv.org/abs/2510.18999v1)

**作者:** Zhirui Dai, Qihao Qian, Tianxing Fan, Nikolay Atanasov

**发布时间:** 2025-10-21

### GPT解析

### 总结

本研究提出了一种名为∇-SDF的混合方法，用于从点云数据估计符号距离函数(SDF)，结合了显式先验和隐式神经残差，实现了高效率、高准确性和可微性的SDF重建。

### 背景

从点云数据估计符号距离函数(SDF)对机器人自主能力(如定位、建图、运动规划和控制)有很多好处。现有方法中，支持在线和大规模SDF重建的体积方法会影响SDF估计的连续性和可微性；而神经网络方法虽然能提供高保真度和可微的SDF重建，但效率较低，在大环境中可能面临灾难性遗忘和内存限制，且通常仅限于截断的SDF。

### 目的

开发一种能够实现非截断(欧几里得)SDF重建的方法，同时具有体积方法的计算和内存效率以及神经网络方法的可微性和准确性。

### 方法

提出∇-SDF，一种混合方法，结合了从梯度增强八叉树插值获得的显式先验和隐式神经残差。

### 主要发现

大量实验证明，∇-SDF在准确性和效率方面优于现有最先进的技术，为机器人技术和计算机视觉中的下游任务提供了可扩展的解决方案。

### 结论

∇-SDF为机器人自主能力中的SDF估计提供了一个高效、准确且可微的解决方案，克服了现有方法的局限性，为下游任务提供了可扩展的解决方案。

### 翻译

从点云数据估计符号距离函数(SDF)已被证明有益于许多机器人自主能力，包括定位、建图、运动规划和控制。支持在线和大规模SDF重建的方法往往依赖于离散的体积数据结构，这会影响SDF估计的连续性和可微性。最近，使用隐式特征的神经网络方法展示了高保真度和可微的SDF重建，但它们往往效率较低，在大环境中可能会经历灾难性遗忘和内存限制，并且通常仅限于截断的SDF。本文提出了∇-SDF，一种混合方法，结合了从梯度增强八叉树插值获得的显式先验和隐式神经残差。我们的方法实现了非截断(欧几里得)的SDF重建，计算和内存效率与体积方法相当，可微性和准确性可与神经网络方法相媲美。大量实验证明，∇-SDF在准确性和效率方面优于最先进的技术，为机器人技术和计算机视觉中的下游任务提供了可扩展的解决方案。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决从点云数据在线学习欧几里得符号距离函数（SDF）的问题。这个问题在机器人自主和计算机视觉领域非常重要，因为准确且可微分的几何环境表示对机器人定位、建图、运动规划和控制等关键功能至关重要。快速更新环境模型和获取梯度信息能让机器人更安全、精确地导航和交互环境，而小内存占用对表示在大场景中的可扩展性很重要。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者分析了现有SDF重建方法的优缺点：体积方法实时性好但不可微；神经网络方法可微分但效率低且易遗忘；高斯过程方法连续但计算复杂。作者借鉴了H2-Mapping的八叉树先验和神经网络残差思想，以及HIO-SDF的全局SDF表示方法。作者设计了一种混合方法，结合显式八叉树先验和隐式神经残差，使用半稀疏八叉树结构和梯度增强插值提高精度，并设计了三种损失函数加速训练，从而克服了现有方法的局限性。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是结合显式八叉树先验和隐式神经残差的混合模型，实现高效、可微分且全局准确的SDF重建。整体流程包括：1)使用半稀疏八叉树存储SDF值和梯度，通过梯度增强插值获得SDF先验；2)使用多分辨率哈希网格编码器和MLP解码器预测SDF残差修正；3)选择关键帧保持训练数据代表性；4)生成表面点、扰动点和自由空间点三种训练样本；5)使用重建损失、Eikonal损失和投影损失训练模型；6)最终SDF预测为八叉树先验与神经网络残差之和。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)梯度增强的八叉树插值方法，在顶点存储SDF值和梯度，提高先验精度；2)半稀疏八叉树结构，平衡内存和精度；3)混合显式-隐式模型，实现全空间而非仅近表面的SDF学习；4)三种精心设计的损失函数加速收敛。相比H2-Mapping，∇-SDF实现非截断SDF重建；相比HIO-SDF，直接优化八叉参数学习更准确先验；相比体积方法，提供可微SDF；相比纯神经网络方法，解决了大环境中的遗忘问题。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '∇-SDF通过梯度增强的八叉树插值与神经残差相结合，实现了高效、可微分且全局准确的在线符号距离函数重建，结合了体积方法和神经网络方法的优点，同时克服了它们的局限性。'}


### 论文摘要

Estimation of signed distance functions (SDFs) from point cloud data has been shown to benefit many robot autonomy capabilities, including localization, mapping, motion planning, and control. Methods that support online and large-scale SDF reconstruction tend to rely on discrete volumetric data structures, which affect the continuity and differentiability of the SDF estimates. Recently, using implicit features, neural network methods have demonstrated high-fidelity and differentiable SDF reconstruction but they tend to be less efficient, can experience catastrophic forgetting and memory limitations in large environments, and are often restricted to truncated SDFs. This work proposes $\nabla$-SDF, a hybrid method that combines an explicit prior obtained from gradient-augmented octree interpolation with an implicit neural residual. Our method achieves non-truncated (Euclidean) SDF reconstruction with computational and memory efficiency comparable to volumetric methods and differentiability and accuracy comparable to neural network methods. Extensive experiments demonstrate that \methodname{} outperforms the state of the art in terms of accuracy and efficiency, providing a scalable solution for downstream tasks in robotics and computer vision.

---

## 24. PAGE-4D: Disentangled Pose and Geometry Estimation for 4D Perception

**论文链接:** [http://arxiv.org/abs/2510.17568v2](http://arxiv.org/abs/2510.17568v2)

**作者:** Kaichen Zhou, Yuhan Wang, Grace Chen, Xinhai Chang, Gaspard Beaudouin, Fangneng Zhan, Paul Pu Liang, Mengyu Wang

**发布时间:** 2025-10-20

### GPT解析

### 总结

PAGE-4D是一个扩展到动态场景的前馈模型，解决了现有3D前馈模型在处理动态元素时的局限性，通过动态感知聚合器实现了无需后处理的相机姿态估计、深度预测和点云重建。

### 背景

最新的3D前馈模型（如VGGT）在推断静态场景的3D属性方面表现出色，但这些模型通常在静态数据集上训练，因此在涉及移动人类或可变形物体等复杂动态元素的真实场景中表现不佳。

### 目的

引入PAGE-4D模型，将VGGT扩展到动态场景，实现相机姿态估计、深度预测和点云重建，无需后处理。

### 方法

提出一种动态感知聚合器，通过预测动态感知掩码来解耦静态和动态信息，对于姿态估计抑制运动线索，对于几何重建则增强这些线索，从而解决多任务4D重建中任务间的固有冲突。

### 主要发现

PAGE-4D在动态场景中始终优于原始VGGT，在相机姿态估计、单目和视频深度估计以及密集点图重建方面取得了优越的结果。

### 结论

PAGE-4D成功解决了多任务4D重建中任务之间的固有冲突，通过动态感知聚合器有效分离了静态和动态信息，在动态场景中表现优异。

### 翻译

最近的3D前馈模型，如视觉几何基础变换器（VGGT），在推断静态场景的3D属性方面表现出强大能力。然而，由于它们通常在静态数据集上训练，这些模型在涉及复杂动态元素的真实场景中往往表现不佳，例如移动的人或像伞这样的可变形物体。为解决这一局限性，我们引入了PAGE-4D，一种将VGGT扩展到动态场景的前馈模型，能够实现相机姿态估计、深度预测和点云重建，且无需后处理。多任务4D重建的一个核心挑战是任务之间的内在冲突：准确的相机姿态估计需要抑制动态区域，而几何重建则需要建模这些区域。为解决这一矛盾，我们提出了一种动态感知聚合器，通过预测动态感知掩码来解耦静态和动态信息——在姿态估计中抑制运动线索，而在几何重建中增强它们。大量实验表明，PAGE-4D在动态场景中始终优于原始VGGT，在相机姿态估计、单目和视频深度估计以及密集点图重建方面取得了优越结果。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决如何在动态场景（包含移动人或可变形物体如伞的场景）中进行准确的3D重建问题。这个问题在现实中非常重要，因为我们的世界本质上是动态的，包含大量移动的物体和人。能够在动态场景中进行准确的3D重建对于机器人导航、增强现实、自动驾驶、视频编辑等多个应用领域至关重要。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先观察到现有的静态3D重建模型在动态场景中表现不佳，尤其是在相机姿态估计和几何重建之间存在冲突：姿态估计需要抑制动态区域，而几何重建则需要建模这些区域。他们借鉴了VGGT作为基础模型，但针对动态场景进行了改进。通过分析VGGT在动态条件下的行为，发现它会忽略动态内容。基于这些观察，作者设计了一个动态感知聚合器，通过预测掩码来分离静态和动态信息，并采用针对性的微调策略，只更新对动态最敏感的中间层。', '这个方法的核心思想是什么？整体实现流程是怎样的？': 'PAGE-4D的核心思想是解耦动态区域在不同任务中的作用：在相机姿态估计时抑制动态区域，而在几何重建时利用这些区域的动态信息。整体实现流程包括：1) 使用预训练编码器提取图像特征；2) 通过动态感知聚合器整合空间和时间线索，包括帧间注意、帧内注意和动态感知全局注意；3) 使用轻量级解码器进行深度和点图重建；4) 专门的相机姿态估计解码器。特别的是，PAGE-4D预测一个动态掩码，通过交叉注意机制应用：过滤相机姿态令牌的动态内容，同时为几何令牌强调这些内容。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': 'PAGE-4D的关键创新点包括：1) 动态感知聚合器，通过预测掩码分离静态和动态信息；2) 针对性的微调策略，只更新对动态最敏感的中间层；3) 任务特定的动态处理，在不同任务中不同方式处理动态区域；4) 统一高效的框架，能在单一前向传递中同时完成多个任务。相比之前的工作，PAGE-4D不需要后处理，运行速度快，在动态场景中表现更好，能产生更密集和准确的点云重建。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'PAGE-4D通过解耦姿态和几何估计中的动态信息处理，首次实现了在单一前向模型中对动态场景的高效准确4D感知，显著超越了之前静态模型在动态环境中的表现。'}


### 论文摘要

Recent 3D feed-forward models, such as the Visual Geometry Grounded Transformer (VGGT), have shown strong capability in inferring 3D attributes of static scenes. However, since they are typically trained on static datasets, these models often struggle in real-world scenarios involving complex dynamic elements, such as moving humans or deformable objects like umbrellas. To address this limitation, we introduce PAGE-4D, a feedforward model that extends VGGT to dynamic scenes, enabling camera pose estimation, depth prediction, and point cloud reconstruction -- all without post-processing. A central challenge in multi-task 4D reconstruction is the inherent conflict between tasks: accurate camera pose estimation requires suppressing dynamic regions, while geometry reconstruction requires modeling them. To resolve this tension, we propose a dynamics-aware aggregator that disentangles static and dynamic information by predicting a dynamics-aware mask -- suppressing motion cues for pose estimation while amplifying them for geometry reconstruction. Extensive experiments show that PAGE-4D consistently outperforms the original VGGT in dynamic scenarios, achieving superior results in camera pose estimation, monocular and video depth estimation, and dense point map reconstruction.

---

## 25. VO-DP: Semantic-Geometric Adaptive Diffusion Policy for Vision-Only Robotic Manipulation

**论文链接:** [http://arxiv.org/abs/2510.15530v2](http://arxiv.org/abs/2510.15530v2)

**作者:** Zehao Ni, Yonghao He, Lingfeng Qian, Jilei Mao, Fa Fu, Wei Sui, Hu Su, Junran Peng, Zhipeng Wang, Bin He

**发布时间:** 2025-10-17

### GPT解析

### 总结

本研究提出了一种名为VO-DP的纯视觉单视角扩散策略学习方法，利用预训练视觉基础模型融合语义和几何特征，在模拟和真实世界任务中均表现出色，并开源了机器人操作训练库。

### 背景

在模仿学习中，基于视觉运动的扩散策略学习是机器人操作的主要方向。现有方法大多依赖点云作为输入并通过点云特征学习构建场景表示，但对纯视觉解决方案的探索不足。

### 目的

探索一种纯视觉且单视角的扩散策略学习方法，以克服对点云输入的依赖，并发挥视觉基础模型在机器人操作中的潜力。

### 方法

提出VO-DP方法，利用VGGT的中间特征结合DINOv2的语义特征和交替注意力块的几何特征，通过交叉注意力融合特征，并用CNN空间压缩后输入策略头。同时开源基于Accelerate的机器人操作训练库，支持多GPU并行训练和混合精度训练。

### 主要发现

模拟任务中VO-DP成功率达64.6%，与点云方法DP3(64.0%)相当，远高于基线DP(34.8%)；真实世界任务中达到87.9%，显著优于DP3(67.5%)和DP(11.2%)。VO-DP在颜色、尺寸、背景和光照变化条件下保持高度稳定。

### 结论

VO-DP证明了纯视觉解决方案在机器人操作中的巨大潜力，特别是在真实世界任务中表现出色。开源的训练库为机器人操作研究提供了有价值的资源。

### 翻译

在模仿学习背景下，基于视觉运动的扩散策略学习是机器人操作的主要方向之一。大多数方法依赖点云作为观察输入，通过点云特征学习构建场景表示，实现显著准确性。然而，现有文献缺乏对具有巨大潜力的纯视觉解决方案的深入探索。本文提出纯视觉和单视角扩散策略学习方法(VO-DP)，利用预训练视觉基础模型实现语义和几何特征有效融合。使用VGGT中间特征，结合DINOv2语义特征和交替注意力块几何特征。特征通过交叉注意力融合，用CNN空间压缩后输入策略头。大量实验表明，VO-DP不仅显著优于纯视觉基线DP，且与点云方法DP3表现不同：模拟任务中VO-DP平均成功率达64.6%，与DP3的64.0%相当，远高于DP的34.8%；真实世界任务中达87.9%，显著优于DP3的67.5%和DP的11.2%。进一步鲁棒性评估证实VO-DP在颜色、尺寸、背景和光照变化条件下保持高度稳定。最后开源机器人操作训练库，基于Accelerate构建，支持多机器多GPU并行训练和混合精度训练，兼容DP、DP3和VO-DP等视觉运动策略，支持RoboTwin模拟器。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决机器人操作领域中纯视觉（仅RGB图像）模仿学习方法性能不足的问题。这个问题很重要，因为现有基于点云或RGB-D图像的方法虽然精度高，但依赖昂贵的深度传感器，而纯视觉方法成本低、实用性强，但性能通常不如基于点云的方法。探索纯视觉方法的潜力可以显著降低机器人系统的硬件成本和复杂度，避免多传感器校准问题，并更接近生物感知-行动系统，具有广泛的应用前景。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先分析了现有方法的优缺点，指出纯视觉方法的性能瓶颈主要在于表示学习模块不完善。他们借鉴了多项现有工作：利用预训练的VGGT模型提取几何信息，使用DINOv2提取语义特征，采用DP中的扩散策略框架，以及Transformer中的cross-attention机制进行特征融合。在此基础上，他们创新设计了语义-几何自适应融合模块和空间特征压缩模块，实现了从单目RGB图像中同时提取和融合语义与几何信息，为下游策略学习提供高质量输入。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是利用预训练的视觉基础模型，从单目RGB图像中同时提取语义和几何特征，并通过自适应融合机制将这些特征有效结合，为下游策略学习提供高质量的输入。整体流程包括：1) 输入处理：接收单视图RGB图像序列；2) 特征提取：使用DINOv2提取语义特征，用VGGT的Alternating Attention网络提取几何特征；3) 特征融合：通过残差交叉注意力机制自适应融合语义和几何特征；4) 场景表示压缩：使用轻量级ResNet压缩融合后的特征，并与机器人关节状态连接形成场景表示；5) 动作生成：基于DDPM的策略头根据场景表示生成动作轨迹。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1) 首次实现纯视觉方法达到点云级别的性能；2) 创新设计语义-几何自适应融合机制；3) 高效的单视图表示学习方法；4) 开源DRRM训练框架。相比之前的工作：1) 相比传统纯视觉方法（如DP），性能显著提升；2) 相比基于点云的方法（如DP3），不需要昂贵深度传感器，在真实世界任务中表现更好；3) 相比其他纯视觉方法，更注重语义和几何特征的融合，在复杂场景中表现更好，对环境变化具有更强鲁棒性。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'VO-DP通过创新性地融合预训练视觉模型的语义和几何特征，首次实现了仅使用RGB图像的机器人操作方法达到与基于点云方法相当的精度，同时大幅降低了硬件成本和系统复杂度。'}


### 论文摘要

In the context of imitation learning, visuomotor-based diffusion policy learning is one of the main directions in robotic manipulation. Most of these approaches rely on point clouds as observation inputs and construct scene representations through point clouds feature learning, which enables them to achieve remarkable accuracy. However, the existing literature lacks an in-depth exploration of vision-only solutions that have significant potential. In this paper, we propose a Vision-Only and single-view Diffusion Policy learning method (VO-DP) that leverages pretrained visual foundation models to achieve effective fusion of semantic and geometric features. We utilize intermediate features from VGGT incorporating semantic features from DINOv2 and geometric features from Alternating Attention blocks. Features are fused via cross-attention and spatially compressed with a CNN to form the input to the policy head. Extensive experiments demonstrate that VO-DP not only outperforms the vision-only baseline DP significantly but also exhibits distinct performance trends against the point cloud-based method DP3: in simulation tasks, VO-DP achieves an average success rate of 64.6% on par with DP3 64.0% and far higher than DP 34.8%, while in real-world tasks, it reaches 87.9%, outperforming both DP3 67.5% and DP 11.2% by a notable margin. Further robustness evaluations confirm that VO-DP remains highly stable under varying conditions including color, size, background, and lighting. Lastly, we open-source a training library for robotic manipulation. Built on Accelerate, this library supports multi-machine and multi-GPU parallel training, as well as mixed precision training. It is compatible with visuomotor policies such as DP, DP3 and VO-DP, and also supports the RoboTwin simulator.

---

