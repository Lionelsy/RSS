# 今日论文推荐 - 2025-10-06

共 50 篇论文

---

## 1. Long-Term Human Motion Prediction Using Spatio-Temporal Maps of Dynamics

**论文链接:** [http://arxiv.org/abs/2510.03031v1](http://arxiv.org/abs/2510.03031v1)

**作者:** Yufei Zhu, Andrey Rudenko, Tomasz P. Kucner, Achim J. Lilienthal, Martin Magnusson

**发布时间:** 2025-10-03

**备注:** IEEE Robotics and Automation Letters

### GPT解析

### 总结

本文提出了一种基于动力学地图（MoDs）的长期人类运动预测（LHMP）框架，能够在长达60秒的时间范围内准确预测人类运动，显著提高了机器人在与人类共享环境中的安全性和实用性。

### 背景

长期人类运动预测对自主机器人和车辆在与人类共享的环境中安全高效运行至关重要，准确的预测对运动规划、跟踪、人机交互和安全监控等应用具有重要意义。

### 目的

实现长达60秒的长期人类运动预测，提高机器人在实际应用中的实用性。

### 方法

利用动力学地图（MoDs）将空间或时空运动模式编码为环境特征，提出MoD信息化的LHMP框架支持多种MoD类型并包含排序方法输出最可能预测轨迹，引入时间条件MoD捕捉不同时间段的运动模式变化，使用三种MoD类型实例化MoD-LHMP并在两个真实世界数据集上评估。

### 主要发现

MoD信息化的方法优于基于学习的方法，平均位移误差提高了高达50%，时间条件变体整体上达到了最高精度。

### 结论

基于动力学地图的长期人类运动预测方法表现出色，其中时间条件MoD变体效果最佳。

### 翻译

长期人类运动预测对于自主机器人和车辆在与人类共享的环境中的安全高效运行非常重要。准确的预测对于包括运动规划、跟踪、人机交互和安全监控在内的应用很重要。在本文中，我们利用动力学地图（MoDs），将空间或时空运动模式编码为环境特征，实现长达60秒的长期人类运动预测。我们提出了一个MoD信息化的LHMP框架，支持各种类型的MoDs，并包含一种排序方法来输出最可能的预测轨迹，提高了机器人的实际实用性。此外，引入了时间条件MoD来捕捉在不同时间段变化的运动模式。我们评估了使用三种类型MoD实例化的MoD-LHMP。在两个真实世界数据集上的实验表明，MoD信息化的方法优于基于学习的方法，平均位移误差提高了高达50%，时间条件变体整体上达到了最高精度。项目代码可在https://github.com/test-bai-cpu/LHMP-with-MoDs.git获取。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决长期人类运动预测问题，即预测人类在未来长达60秒的运动轨迹。这个问题对于自主机器人和车辆在人类共享环境中的安全高效运行至关重要，应用于运动规划、跟踪、人机交互和安全监控等多个领域。长期预测特别重要，因为它需要考虑环境对人类运动的复杂影响，而不仅仅是当前状态。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先认识到人类运动受个人意图和环境因素共同影响，长期预测需要更全面的建模。他们借鉴了现有的CLiFF-LHMP方法，扩展了其使用范围，使其能适用于各种类型的动力学地图(MoDs)。作者还参考了CLiFF-map、STeF-map等现有地图表示方法，并创新性地引入了时间条件概念，捕捉不同时段的运动模式变化。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是利用动力学地图(MoDs)编码环境中的运动模式，并利用这些模式来引导长期人类运动预测。整体流程包括：1)从历史轨迹估计当前速度；2)在每个预测步骤从当前位置的MoD中采样速度；3)用采样速度偏置恒定速度模型(CVM)的预测；4)通过高斯核函数控制MoD的影响程度；5)迭代预测直到达到预测时间范围；6)使用排名方法输出最可能的预测轨迹。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)提出通用的MoD-LHMP框架，支持各种类型的MoDs；2)引入排名方法，输出最可能的预测轨迹；3)提出时间条件CLiFF-map，捕捉不同时段的运动模式；4)在两个真实世界数据集上全面评估。相比之前的工作，该方法能更好地处理长期预测，考虑时间变化因素，且比基于学习的方法在长期预测上表现更优。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '这篇论文提出了一种基于动力学地图的长期人类运动预测框架，通过引入时间条件地图和排名机制，显著提高了长期预测的准确性，为机器人在人类共享环境中的安全导航提供了更可靠的预测能力。'}


### 论文摘要

Long-term human motion prediction (LHMP) is important for the safe and efficient operation of autonomous robots and vehicles in environments shared with humans. Accurate predictions are important for applications including motion planning, tracking, human-robot interaction, and safety monitoring. In this paper, we exploit Maps of Dynamics (MoDs), which encode spatial or spatio-temporal motion patterns as environment features, to achieve LHMP for horizons of up to 60 seconds. We propose an MoD-informed LHMP framework that supports various types of MoDs and includes a ranking method to output the most likely predicted trajectory, improving practical utility in robotics. Further, a time-conditioned MoD is introduced to capture motion patterns that vary across different times of day. We evaluate MoD-LHMP instantiated with three types of MoDs. Experiments on two real-world datasets show that MoD-informed method outperforms learning-based ones, with up to 50\% improvement in average displacement error, and the time-conditioned variant achieves the highest accuracy overall. Project code is available at https://github.com/test-bai-cpu/LHMP-with-MoDs.git

---

## 2. When Researchers Say Mental Model/Theory of Mind of AI, What Are They Really Talking About?

**论文链接:** [http://arxiv.org/abs/2510.02660v1](http://arxiv.org/abs/2510.02660v1)

**作者:** Xiaoyun Yin, Elmira Zahmat Doost, Shiwen Zhou, Garima Arya Yadav, Jamie C. Gorman

**发布时间:** 2025-10-03

### GPT解析

### 总结

研究人员声称AI系统拥有心智理论或心智模型时，实际上是在讨论行为预测和偏差修正，而非真正的心理状态。当前讨论混淆了复杂的模式匹配与真实认知，忽略了模拟与经验之间的区别。大型语言模型在心智理论实验室任务中的表现仅基于行为模仿，而非真实理解。现有测试范式存在缺陷，不应简单将人类认知测试应用于AI系统。建议转向相互的心智理论框架，关注人类与AI的互动动态。

### 背景

当前学术界对AI系统是否拥有心智理论的讨论存在混淆，研究者经常将AI系统的行为表现误认为真正的认知能力。

### 目的

区分AI系统的行为模拟与真实认知，改进AI心智理论的测试方法，提出更合理的评估框架。

### 方法

分析当前AI心智理论研究的局限性，批判现有测试范式，提出相互的心智理论框架。

### 主要发现

大型语言模型在心智理论实验室任务中的表现仅基于行为模仿；现有测试范式在评估AI认知能力时存在根本性缺陷；孤立测试AI无法准确评估其真正的认知能力。

### 结论

需要区分AI系统的行为预测与真正的心理状态；应将研究重点转向人类与AI之间的互动动态；采用相互的心智理论框架，同时考虑人类认知和AI算法的贡献。

### 翻译

当研究人员声称AI系统拥有心智理论或心智模型时，他们实际上是在讨论行为预测和偏差修正，而不是真正的心理状态。这篇立场论文认为，当前的讨论将复杂的模式匹配与真实的认知混为一谈，忽略了模拟与经验之间的关键区别。尽管最近的研究显示大型语言模型在心智理论实验室任务中达到了人类水平的性能，但这些结果仅基于行为模仿。更重要的是，整个测试范式可能存在缺陷，它将人类个体认知测试应用于AI系统，而不是在人类与AI互动的瞬间直接评估人类认知。我建议将重点转向相互的心智理论框架，承认人类认知和AI算法的同时贡献，强调互动动态，而不是孤立地测试AI。


### 论文摘要

When researchers claim AI systems possess ToM or mental models, they are fundamentally discussing behavioral predictions and bias corrections rather than genuine mental states. This position paper argues that the current discourse conflates sophisticated pattern matching with authentic cognition, missing a crucial distinction between simulation and experience. While recent studies show LLMs achieving human-level performance on ToM laboratory tasks, these results are based only on behavioral mimicry. More importantly, the entire testing paradigm may be flawed in applying individual human cognitive tests to AI systems, but assessing human cognition directly in the moment of human-AI interaction. I suggest shifting focus toward mutual ToM frameworks that acknowledge the simultaneous contributions of human cognition and AI algorithms, emphasizing the interaction dynamics, instead of testing AI in isolation.

---

## 3. SIMSplat: Predictive Driving Scene Editing with Language-aligned 4D Gaussian Splatting

**论文链接:** [http://arxiv.org/abs/2510.02469v1](http://arxiv.org/abs/2510.02469v1)

**作者:** Sung-Yeon Park, Adam Lee, Juanwu Lu, Can Cui, Luyang Jiang, Rohit Gupta, Kyungtae Han, Ahmadreza Moradipari, Ziran Wang

**发布时间:** 2025-10-02

### GPT解析

### 总结

本文提出了SIMSplat，一种基于语言对齐高斯飞溅的预测性驾驶场景编辑器，能够通过自然语言提示实现直观的场景操控和精确的对象级编辑。

### 背景

基于传感器数据的驾驶场景操控正在成为传统虚拟驾驶模拟器的一个有前景的替代方案，但现有框架由于编辑能力有限，难以高效生成真实场景。

### 目的

开发一种能够支持直观操控和精确编辑的驾驶场景编辑框架，解决现有方法在场景生成和编辑方面的局限性。

### 方法

SIMSplat是一种语言控制的编辑器，通过将语言与高斯重建的场景对齐，支持直接查询道路对象，实现精确和灵活的编辑。该方法提供详细的对象级编辑，包括添加新对象和修改车辆与行人轨迹，并通过多智能体运动预测进行预测路径优化，生成真实的场景交互。

### 主要发现

在Waymo数据集上的实验表明，SIMSplat具有广泛的编辑能力和在各种场景中的适应性，能够高效生成真实感强的驾驶场景。

### 结论

SIMSplat为驾驶场景操控提供了一个有效的解决方案，通过自然语言控制实现精确的对象级编辑，并生成真实的场景交互，为自动驾驶研究提供了有价值的工具。

### 翻译

基于传感器数据的驾驶场景操控正在成为传统虚拟驾驶模拟器的一个有前景的替代方案。然而，现有框架由于编辑能力有限，难以高效生成真实场景。为解决这些挑战，我们提出了SIMSplat，一种具有语言对齐高斯飞溅的预测性驾驶场景编辑器。作为语言控制的编辑器，SIMSplat能够使用自然语言提示进行直观操控。通过将语言与高斯重建的场景对齐，它进一步支持直接查询道路对象，实现精确和灵活的编辑。我们的方法提供详细的对象级编辑，包括添加新对象和修改车辆与行人的轨迹，同时通过多智能体运动预测进行预测路径优化，生成场景中所有智能体之间的真实交互。在Waymo数据集上的实验证明了SIMSplat广泛的编辑能力和在各种场景中的适应性。项目页面：https://sungyeonparkk.github.io/simsplat/

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决驾驶场景编辑的效率和真实性问题。现有方法在创建多样化驾驶场景时能力有限，难以实现细粒度的对象级编辑，且通常只验证自车和目标对象的可行性，忽略了周围所有代理的响应。这个问题很重要，因为驾驶模拟器是自动驾驶算法开发的关键测试平台，能够高效编辑和生成真实驾驶场景对于测试自动驾驶系统在各种情况下的表现至关重要，特别是对于创建难以在现实中模拟的安全关键场景。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者通过分析现有方法在对象级编辑、场景可行性验证和行人编辑方面的局限性，设计了一个统一的语言控制驾驶模拟器。他们借鉴了4D高斯溅射技术用于场景重建，参考了LangSplat等语言-场景对齐方法但进行了改进以适应动态驾驶场景，使用了场景图表示方法和SAM-2、CLIP等工具。关键创新在于提出了运动感知语言对齐，使系统能理解驾驶场景中的行为描述，并设计了多代理路径精确保证编辑后场景的全局一致性。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是通过将语言特征与4D高斯场景对齐，实现自然语言驱动的驾驶场景编辑，并使用多代理路径精确保证场景真实性。整体流程包括：1)使用场景图4D高斯溅射重建场景；2)通过外观和时间对齐将语言特征嵌入高斯；3)LLM代理解析用户提示并执行编辑操作；4)多代理路径精修确保所有对象自然响应变化；5)扩散修复润色修改区域，确保输出无缝真实。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)运动感知语言对齐，能理解驾驶场景中的行为描述；2)基于LLM的对象级编辑器，支持细粒度的车辆和行人修改；3)多代理路径精修，确保编辑后场景的全局一致性。相比ChatSim，SIMSplat支持行人编辑和修改现有对象；相比OmniRe，支持通过详细提示控制对象；相比LangSplat等，专注于动态驾驶场景且查询性能显著提升；相比SceneCrafter，支持动态代理的显式运动编辑且无需用户提供3D边界框。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'SIMSplat通过结合运动感知语言对齐、基于LLM的对象级编辑和多代理路径精修，实现了自然语言驱动的真实驾驶场景编辑，支持细粒度的车辆和行人修改，并确保编辑后场景的全局一致性和真实性。'}


### 论文摘要

Driving scene manipulation with sensor data is emerging as a promising alternative to traditional virtual driving simulators. However, existing frameworks struggle to generate realistic scenarios efficiently due to limited editing capabilities. To address these challenges, we present SIMSplat, a predictive driving scene editor with language-aligned Gaussian splatting. As a language-controlled editor, SIMSplat enables intuitive manipulation using natural language prompts. By aligning language with Gaussian-reconstructed scenes, it further supports direct querying of road objects, allowing precise and flexible editing. Our method provides detailed object-level editing, including adding new objects and modifying the trajectories of both vehicles and pedestrians, while also incorporating predictive path refinement through multi-agent motion prediction to generate realistic interactions among all agents in the scene. Experiments on the Waymo dataset demonstrate SIMSplat's extensive editing capabilities and adaptability across a wide range of scenarios. Project page: https://sungyeonparkk.github.io/simsplat/

---

## 4. MM-Nav: Multi-View VLA Model for Robust Visual Navigation via Multi-Expert Learning

**论文链接:** [http://arxiv.org/abs/2510.03142v1](http://arxiv.org/abs/2510.03142v1)

**作者:** Tianyu Xu, Jiawei Chen, Jiazhao Zhang, Wenyao Zhang, Zekun Qi, Minghan Li, Zhizheng Zhang, He Wang

**发布时间:** 2025-10-03

**备注:** Project page: https://pku-epic.github.io/MM-Nav-Web/

### GPT解析

### 总结

该研究提出了一种基于视觉-语言-动作（VLA）模型的视觉导航方法，通过教师-学生学习方式从合成专家数据中学习多样化的导航能力，并在合成和真实环境中验证了其有效性和泛化能力。

### 背景

视觉导航策略被认为是有前途的方向，因为它通过使用以自我为中心的视觉观察来模拟人类导航行为。然而，视觉观察的光学信息难以像LiDAR点云或深度图那样被明确建模，这需要智能模型和大规模数据。

### 目的

利用视觉-语言-动作（VLA）模型的智能，通过教师-学生方式从合成专家数据中学习多样化的导航能力。

### 方法

实现VLA模型MM-Nav作为基于预训练大型语言模型和视觉基础模型的多视图VLA（具有360度观察）；从三个具有不同导航能力的挑战性定制环境中收集专家数据（到达、挤压和避免）；使用从RL专家在线收集的数据迭代训练VLA模型，并根据个体能力的性能动态平衡训练比例。

### 主要发现

在合成环境中的实验表明模型实现了强大的泛化能力；学生VLA模型优于RL教师，展示了整合多种能力的协同效应。

### 结论

大量的真实世界实验进一步证实了该方法的有效性。

### 翻译

视觉导航策略被认为是一个有前途的方向，因为它通过使用以自我为中心的视觉观察来模仿人类的导航行为。然而，视觉观察的光学信息难以像LiDAR点云或深度图那样被明确建模，这需要智能模型和大规模数据。为此，我们提出利用视觉-语言-动作（VLA）模型的智能，通过教师-学生方式从合成专家数据中学习多样化的导航能力。具体来说，我们实现了VLA模型MM-Nav，这是一个基于预训练大型语言模型和视觉基础模型的多视图VLA（具有360度观察）。对于大规模导航数据，我们从三个使用特权深度信息训练的强化学习（RL）专家中收集专家数据，这些专家数据来自三个针对不同导航能力（到达、挤压和避免）定制的具有挑战性的环境。我们使用从RL专家在线收集的数据迭代训练VLA模型，其中训练比例基于个体能力的性能进行动态平衡。通过在合成环境中的大量实验，我们证明我们的模型实现了强大的泛化能力。此外，我们发现我们的学生VLA模型优于RL教师，这展示了整合多种能力的协同效应。大量的真实世界实验进一步证实了我们方法的有效性。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决机器人视觉导航中的挑战，特别是在复杂环境中如何让机器人仅通过视觉信息进行有效导航。这个问题很重要，因为视觉导航能让机器人像人类一样通过观察环境来移动，但视觉信息难以像激光雷达那样直接建模，需要智能模型和大量数据。解决这一问题将使机器人在更复杂、更具挑战性的环境中工作，扩大机器人的应用范围。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': "作者注意到视觉导航面临一个矛盾：真实世界导航数据缺乏极端挑战场景，而合成数据虽有挑战性但存在模拟到现实的差距。为了解决这个矛盾，作者借鉴了视觉-语言-动作模型、强化学习专家和教师-学生训练策略等现有工作，设计了MM-Nav方法。他们使用四个摄像头实现360度环境感知，并训练三个专门的强化学习专家分别学习'到达'、'挤压'和'躲避'三种能力，然后将这些专家的知识整合到一个统一的VLA模型中。", '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是通过多视图感知和多专家学习，让机器人从不同专业专家那里学习并整合多种导航能力。整体流程分为三步：首先，在三个定制环境中训练专门的强化学习专家；其次，收集这些专家的成功轨迹数据，用于初始训练VLA模型；最后，采用教师-学生迭代训练方法，在模拟环境中部署VLA模型并在线收集专家数据，通过能力平衡的数据聚合策略动态调整不同能力数据的训练比例，持续优化模型性能。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1）多视图VLA模型，使用四个摄像头实现360度环境感知；2）多专家学习策略，从三个专业强化学习专家学习不同导航能力；3）能力平衡的训练方法，根据模型在不同能力上的表现差距动态调整训练数据比例；4）教师-学生迭代训练框架。相比之前的工作，MM-Nav不仅提供更全面的环境感知，还直接输出连续速度命令而非离散动作，并通过整合多种能力实现了比单一专家更强的导航性能。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'MM-Nav通过多视图视觉-语言-动作模型和多专家学习策略，让机器人从多个专业强化学习专家中学习并整合不同导航能力，在复杂环境中展现出比单一专家更强的导航性能。'}


### 论文摘要

Visual navigation policy is widely regarded as a promising direction, as it mimics humans by using egocentric visual observations for navigation. However, optical information of visual observations is difficult to be explicitly modeled like LiDAR point clouds or depth maps, which subsequently requires intelligent models and large-scale data. To this end, we propose to leverage the intelligence of the Vision-Language-Action (VLA) model to learn diverse navigation capabilities from synthetic expert data in a teacher-student manner. Specifically, we implement the VLA model, MM-Nav, as a multi-view VLA (with 360 observations) based on pretrained large language models and visual foundation models. For large-scale navigation data, we collect expert data from three reinforcement learning (RL) experts trained with privileged depth information in three challenging tailor-made environments for different navigation capabilities: reaching, squeezing, and avoiding. We iteratively train our VLA model using data collected online from RL experts, where the training ratio is dynamically balanced based on performance on individual capabilities. Through extensive experiments in synthetic environments, we demonstrate that our model achieves strong generalization capability. Moreover, we find that our student VLA model outperforms the RL teachers, demonstrating the synergistic effect of integrating multiple capabilities. Extensive real-world experiments further confirm the effectiveness of our method.

---

## 5. GeoComplete: Geometry-Aware Diffusion for Reference-Driven Image Completion

**论文链接:** [http://arxiv.org/abs/2510.03110v1](http://arxiv.org/abs/2510.03110v1)

**作者:** Beibei Lin, Tingting Chen, Robby T. Tan

**发布时间:** 2025-10-03

**备注:** Accepted by NeurIPS 2025. Project page:  https://bb12346.github.io/GeoComplete/

### GPT解析

### 总结

GeoComplete是一种新型框架，通过引入显式3D结构指导来强制补全区域中的几何一致性，解决了目标视图与参考图像差异大时的图像补全挑战。

### 背景

基于参考的图像补全技术在目标视图与参考图像差异显著时特别具有挑战性。现有生成方法仅依赖扩散先验，缺乏几何线索（如相机姿态或深度），常产生错位或不可信的内容。

### 目的

提出一个名为GeoComplete的新框架，在补全区域中强制执行几何一致性，区别于先前的仅基于图像的方法。

### 方法

GeoComplete引入两个关键想法：将扩散过程条件化到投影点云上注入几何信息，以及应用目标感知掩码引导模型关注相关参考线索。框架采用双分支扩散架构，一个分支合成缺失区域，另一个提取几何特征，通过跨分支联合自注意力确保连贯准确的补全。此外，通过投影目标视图到参考图像中检测遮挡区域并掩码处理，指导模型关注有用线索。

### 主要发现

通过整合几何感知的双分支扩散架构和目标感知掩码策略，GeoComplete为几何条件图像补全提供了统一且强大的解决方案。实验表明，GeoComplete比最先进方法实现了17.1 PSNR的改进，显著提高了几何准确性同时保持高质量视觉效果。

### 结论

GeoComplete是一种统一且强大的解决方案，用于几何条件图像补全，通过结合几何信息和目标感知掩码策略，在补全质量和几何准确性方面优于现有方法。

### 翻译

基于参考的图像补全使用额外图像恢复目标视图中的缺失区域，当目标视图与参考图像差异显著时尤其具有挑战性。现有生成方法仅依赖扩散先验，没有几何线索如相机姿态或深度，常常产生错位或不可信的内容。我们提出了GeoComplete，一个新框架，通过引入显式3D结构指导来强制补全区域中的几何一致性，区别于先前的仅基于图像的方法。GeoComplete引入两个关键想法：将扩散过程条件化到投影点云上以注入几何信息，以及应用目标感知掩码引导模型关注相关参考线索。该框架采用双分支扩散架构，一个分支从掩码目标合成缺失区域，另一个分支从投影点云中提取几何特征。跨分支的联合自注意力确保连贯和准确的补全。为解决参考中可见但目标中缺失的区域，我们将目标视图投影到每个参考中以检测遮挡区域，然后在训练期间对这些区域进行掩码处理。这种目标感知掩码指导模型关注有用的线索，提高在困难场景中的性能。通过整合几何感知的双分支扩散架构和目标感知掩码策略，GeoComplete为几何条件图像补全提供了统一且强大的解决方案。实验表明，GeoComplete比最先进方法实现了17.1 PSNR的改进，显著提高了几何准确性同时保持高质量视觉效果。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决参考驱动的图像补全中的几何一致性问题。当目标图像与参考图像视角差异较大时，现有生成方法往往会产生错位或不合理的内容。这个问题在现实中非常重要，因为它直接影响图像修复的质量和可信度，对于需要精确空间关系的应用（如自动驾驶、增强现实、图像编辑等）尤为关键。几何一致性是保持图像真实感和合理性的基础，缺乏这种一致性的补全结果会显得不自然甚至荒谬。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先分析了现有方法的局限性：传统几何方法采用多阶段流程（姿态估计、深度重建等），容易在复杂场景中出现错误累积；而生成方法如RealFill在目标视图与参考差异较大时表现不佳，因为没有几何线索。作者借鉴了扩散模型的强大生成能力、VGGT用于统一预测相机参数和深度图、LangSAM用于分割动态对象。在此基础上，作者设计了将显式3D结构指导整合到扩散模型中的方法，通过双分支架构和目标感知掩码策略来解决几何一致性问题。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是将显式3D几何信息注入扩散模型，强制补全区域保持几何一致性，并通过目标感知掩码引导模型关注有意义的参考线索。整体实现流程包括：1)点云生成：使用LangSAM过滤动态区域，用VGGT估计相机参数和深度图，构建并投影3D点云；2)目标感知掩码：将目标图像投影到参考视图，识别信息丰富区域，应用条件参考掩码和条件点云掩码；3)双分支扩散：目标分支处理掩码图像生成缺失内容，云分支处理投影点云提供几何指导，通过联合自注意力融合两个分支信息，并使用注意力掩码确保掩码标记直接接收几何线索。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)双分支扩散架构：结合目标分支和云分支，通过联合自注意力融合视觉和几何信息；2)目标感知掩码策略：选择性掩码信息丰富区域，引导模型关注有意义的参考线索；3)几何信息注入：将显式3D几何信息整合到扩散模型中。相比之前的工作，GeoComplete避免了传统几何方法的多阶段错误累积，超越了纯图像方法的几何限制，不假设所有视图共享相同几何（区别于NeRF等方法），并通过LangSAM提高了动态场景的处理能力。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'GeoComplete通过将显式3D几何信息整合到双分支扩散模型中，并引入目标感知掩码策略，显著提升了参考驱动图像补全中的几何一致性和视觉质量，实现了比现有方法高17.1%的PSNR性能提升。'}


### 论文摘要

Reference-driven image completion, which restores missing regions in a target view using additional images, is particularly challenging when the target view differs significantly from the references. Existing generative methods rely solely on diffusion priors and, without geometric cues such as camera pose or depth, often produce misaligned or implausible content. We propose GeoComplete, a novel framework that incorporates explicit 3D structural guidance to enforce geometric consistency in the completed regions, setting it apart from prior image-only approaches. GeoComplete introduces two key ideas: conditioning the diffusion process on projected point clouds to infuse geometric information, and applying target-aware masking to guide the model toward relevant reference cues. The framework features a dual-branch diffusion architecture. One branch synthesizes the missing regions from the masked target, while the other extracts geometric features from the projected point cloud. Joint self-attention across branches ensures coherent and accurate completion. To address regions visible in references but absent in the target, we project the target view into each reference to detect occluded areas, which are then masked during training. This target-aware masking directs the model to focus on useful cues, enhancing performance in difficult scenarios. By integrating a geometry-aware dual-branch diffusion architecture with a target-aware masking strategy, GeoComplete offers a unified and robust solution for geometry-conditioned image completion. Experiments show that GeoComplete achieves a 17.1 PSNR improvement over state-of-the-art methods, significantly boosting geometric accuracy while maintaining high visual quality.

---

## 6. Point Cloud-Based Control Barrier Functions for Model Predictive Control in Safety-Critical Navigation of Autonomous Mobile Robots

**论文链接:** [http://arxiv.org/abs/2510.02885v1](http://arxiv.org/abs/2510.02885v1)

**作者:** Faduo Liang, Yunfeng Yang, Shi-Lu Dai

**发布时间:** 2025-10-03

**备注:** 8 pages, 8 figures, accepted to IROS2025

### GPT解析

### 总结

该研究提出了一种新型运动规划算法，通过结合实时动态障碍物跟踪、控制屏障函数和非线性模型预测控制，实现了自主移动机器人在复杂环境中的安全导航。

### 背景

自主移动机器人在动态环境中进行安全导航是一个挑战性问题，需要有效处理静态和动态障碍物。

### 目的

开发一种能够处理静态和动态障碍物的运动规划算法，确保机器人在复杂环境中的安全导航。

### 方法

1. 集成实时动态障碍物跟踪和映射系统，将点云分类为动态和静态组件；2. 使用卡尔曼滤波器估计和预测动态点云的运动状态；3. 外推动态点云的未来状态并与静态点云合并构建前向时间域地图；4. 结合控制屏障函数与非线性模型预测控制进行障碍物避障；5. 基于预测状态与地图碰撞检测确定的风险点制定CBF约束。

### 主要发现

1. 该算法在模拟和真实场景的复杂环境中表现出有效性；2. 与两种基线方法相比，该算法在障碍物避障的安全性和鲁棒性方面表现更优。

### 结论

所提出的算法通过有效整合动态障碍物预测和控制策略，显著提高了机器人在复杂环境中的安全导航能力。

### 翻译

在这项工作中，我们提出了一种新颖的运动规划算法，以促进自主移动机器人的安全关键导航。所提出的算法集成了一个实时动态障碍物跟踪和映射系统，将点云分为动态和静态组件。对于动态点云，采用卡尔曼滤波器来估计和预测它们的运动状态。基于这些预测，我们外推动态点云的未来状态，随后与静态点云合并以构建前向时间域地图。通过将控制屏障函数与非线性模型预测控制相结合，所提出的算法使机器人能够有效避开静态和动态障碍物。CBF约束是基于通过预测未来状态与FTD地图之间的碰撞检测确定的风险点制定的。来自模拟和真实场景的实验结果证明了所提出的算法在复杂环境中的有效性。在模拟实验中，将所提出的算法与两种基线方法进行了比较，显示出在障碍物避障方面的安全性和鲁棒性方面的优越性能。源代码已发布供机器人社区参考。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决自主移动机器人在同时包含静态和动态障碍物的复杂环境中进行安全导航的问题。这个问题在现实中非常重要，因为自主机器人需要在人类共享空间中安全工作，而现有方法通常将障碍物简化为椭球体，导致过度保守的避障行为，特别是在处理不规则形状障碍物时，限制了机器人的运动能力和效率。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者通过分析现有方法的局限性（如过度保守的障碍物建模、难以同时处理静态和动态障碍物）来设计新方法。他们借鉴了动态障碍物检测技术（如点云聚类、YOLO检测器）、控制屏障函数(CBF)理论和非线性模型预测控制(NMPC)框架，但进行了创新性整合。具体来说，他们改进了障碍物检测方法（YOLO-Fusion检测器），开发了基于点云的直接障碍物表示方法，并设计了前向时域(FTD)地图来整合当前和预测的未来障碍物状态。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是将控制屏障函数(CBF)与非线模型预测控制(NMPC)相结合，直接使用点云数据表示障碍物，并构建包含预测信息的环境地图。整体流程包括：1)使用RGB-D相机获取环境数据并生成点云；2)通过YOLO-Fusion检测器识别动态障碍物；3)使用卡尔曼滤波预测动态障碍物运动状态；4)构建前向时域(FTD)地图，整合静态和动态点云；5)识别静态和动态风险点；6)基于风险点生成控制屏障函数；7)将CBF约束集成到NMPC框架中生成安全轨迹；8)实时执行控制指令。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)基于点云的直接障碍物表示，而非简化为几何形状；2)前向时域(FTD)地图整合当前和预测的未来障碍物状态；3)启发式高风险碰撞点识别方法，包括前向-逆向碰撞检测器和动态风险点识别；4)YOLO-Fusion检测器提高动态障碍物检测精度。相比之前的工作，这种方法避免了过度保守的避障行为，能够处理不规则形状障碍物（如有空心区域的黑板），同时能有效预测和避让动态障碍物，在复杂环境中表现出更高的安全性和鲁棒性。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '这篇论文提出了一种基于点云的控制屏障函数与非线模型预测控制相结合的新方法，通过构建前向时域地图和识别高风险碰撞点，实现了自主移动机器人在复杂动态环境中安全、高效地同时避让静态和动态障碍物。'}


### 论文摘要

In this work, we propose a novel motion planning algorithm to facilitate safety-critical navigation for autonomous mobile robots. The proposed algorithm integrates a real-time dynamic obstacle tracking and mapping system that categorizes point clouds into dynamic and static components. For dynamic point clouds, the Kalman filter is employed to estimate and predict their motion states. Based on these predictions, we extrapolate the future states of dynamic point clouds, which are subsequently merged with static point clouds to construct the forward-time-domain (FTD) map. By combining control barrier functions (CBFs) with nonlinear model predictive control, the proposed algorithm enables the robot to effectively avoid both static and dynamic obstacles. The CBF constraints are formulated based on risk points identified through collision detection between the predicted future states and the FTD map. Experimental results from both simulated and real-world scenarios demonstrate the efficacy of the proposed algorithm in complex environments. In simulation experiments, the proposed algorithm is compared with two baseline approaches, showing superior performance in terms of safety and robustness in obstacle avoidance. The source code is released for the reference of the robotics community.

---

## 7. Visualizing Spatial Point Clouds: A Task-Oriented Taxonomy

**论文链接:** [http://arxiv.org/abs/2510.02651v1](http://arxiv.org/abs/2510.02651v1)

**作者:** Mahsa Partovi, Federico Iuricich

**发布时间:** 2025-10-03

**备注:** 12 pages, 3 figures, 1 table

### GPT解析

### 总结

本文分析了3D点云可视化的设计空间，提出了一种分类法来映射四十年来可视化设计选择与现代应用挑战的关系，为开发更有效、可解释和以用户为中心的可视化技术提供了框架。

### 背景

3D点云数据可视化在自主导航、环境监测和灾难响应等领域至关重要，这些领域的任务如物体识别、结构分析和时空探索依赖于清晰有效的视觉表示。尽管AI驱动处理技术有所进步，可视化仍然是解释复杂数据集的关键工具。然而，由于数据的稀疏性、密度变化和规模问题，设计有效的点云可视化面临重大挑战。

### 目的

分析空间点云可视化的设计空间，并解决可视化技术与分析目标之间缺乏系统映射的问题。

### 方法

引入一种分类法，对四十年来可视化设计选择进行分类，并将这些选择与现代应用中的基本挑战联系起来。基于数据类型、用户目标和可视化技术构建可视化策略框架。

### 主要发现

在可视化技术与分析目标之间缺乏系统映射，需要一种将可视化设计选择与挑战联系起来的分类法。

### 结论

该框架为推进更有效、可解释和以用户为中心的可视化技术提供了基础。

### 翻译

3D点云数据的可视化在自主导航、环境监测和灾难响应等领域至关重要，在这些领域中，物体识别、结构分析和时空探索等任务依赖于清晰有效的视觉表示。尽管AI驱动处理技术有所进步，可视化仍然是解释复杂数据集的关键工具。然而，由于数据的稀疏性、密度变化和规模问题，设计有效的点云可视化面临重大挑战。在这项工作中，我们分析了空间点云可视化的设计空间，突显了可视化技术与分析目标之间系统映射的差距。我们引入了一种分类法，对四十年来可视化设计选择进行分类，并将它们与现代应用中的基本挑战联系起来。通过基于数据类型、用户目标和可视化技术构建可视化策略，我们的框架为推进更有效、可解释和以用户为中心的可视化技术提供了基础。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决空间点云可视化的系统化分类和设计空间映射问题。这个问题很重要，因为点云数据在自动驾驶、环境监测和灾害响应等领域至关重要，这些领域的任务如物体识别和结构分析依赖于清晰有效的视觉表示。尽管AI处理技术有所进步，可视化仍是解释复杂数据集的关键工具，而点云数据的稀疏性、密度变化和规模使得设计有效可视化面临重大挑战，缺乏系统化框架阻碍了更有效可视化技术的发展。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者采用系统化文献综述方法，从主要可视化期刊收集论文并使用关键词搜索，初始获得1200多篇论文后筛选出相关研究。他们采用What-Why-How框架组织知识，这一框架在可视化领域已建立。作者确实借鉴了现有工作，参考了点云处理计算方法的调查、特定可视化技术的调查，并融合了抽象点数据表示和基于深度学习的点云处理的分类法。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是通过任务导向分类法系统化组织和理解空间点云可视化设计空间，强调可视化应紧密联系用户分析目标而非仅关注技术实现。整体流程分三部分：数据抽象(What)定义点云关键特征；任务抽象(Why)将用户任务分为部分-整体关系、空间关系和时间关系；设计选择(How)按渲染基元分为基于点、基于网格和基于几何三类技术，每类包含具体实现方法。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：创建首个系统化框架将四十年点云可视化技术映射到用户任务；采用以用户为中心的What-Why-How框架；进行全面的文献综述并连接可视化技术与机器学习方法；提出新的任务分类体系。相比之前工作，本文专注于空间点云而非非空间点可视化，强调低级视觉任务，采用系统化文献综述而非技术性综述，以用户任务而非技术为中心，同时考虑传统和基于深度学习的方法。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '这篇论文通过创建一个系统化的任务导向分类法，填补了点云可视化技术与用户分析目标之间的映射差距，为未来更有效、可解释和以用户为中心的点云可视化技术的发展奠定了基础。'}


### 论文摘要

The visualization of 3D point cloud data is essential in fields such as autonomous navigation, environmental monitoring, and disaster response, where tasks like object recognition, structural analysis, and spatiotemporal exploration rely on clear and effective visual representation. Despite advancements in AI-driven processing, visualization remains a critical tool for interpreting complex spatial datasets. However, designing effective point cloud visualizations presents significant challenges due to the sparsity, density variations, and scale of the data. In this work, we analyze the design space of spatial point cloud visualization, highlighting a gap in systematically mapping visualization techniques to analytical objectives. We introduce a taxonomy that categorizes four decades of visualization design choices, linking them to fundamental challenges in modern applications. By structuring visualization strategies based on data types, user objectives, and visualization techniques, our framework provides a foundation for advancing more effective, interpretable, and user-centered visualization techniques.

---

## 8. ERUPT: An Open Toolkit for Interfacing with Robot Motion Planners in Extended Reality

**论文链接:** [http://arxiv.org/abs/2510.02464v1](http://arxiv.org/abs/2510.02464v1)

**作者:** Isaac Ngui, Courtney McBeth, André Santos, Grace He, Katherine J. Mimnaugh, James D. Motes, Luciano Soares, Marco Morales, Nancy M. Amato

**发布时间:** 2025-10-02

### GPT解析

### 总结

本研究提出了Extended Reality Universal Planning Toolkit (ERUPT)，一个扩展现实(XR)交互式运动规划系统，允许用户在三维环境中创建、重新配置环境并规划机器人路径。

### 背景

传统机器人路径规划通常在二维屏幕上进行，缺乏空间直观性和自然交互能力。扩展现实技术为机器人运动规划提供了新的可能性。

### 目的

开发一个基于扩展现实的交互式机器人运动规划系统，提供更直观的空间理解和更自然的交互方式，以提高规划效率和准确性。

### 方法

设计并实现了一个名为ERUPT的扩展现实系统，该系统允许用户在沉浸式三维环境中自然地与虚拟对象交互，集成了MoveIt操作规划框架，并提供多种交互模态。

### 主要发现

通过扩展现实环境，用户能够获得更好的空间理解，使用自然交互方式调整环境对象，可视化机器人运动，并在无碰撞风险的环境中进行规划。

### 结论

ERUPT系统通过结合扩展现实技术和机器人运动规划，提供了一种更直观、更自然的机器人路径规划方法，可以在虚拟环境中验证规划结果后部署到实际机器人。

### 翻译

我们提出了扩展现实通用规划工具包(ERUPT)，一个用于交互式运动规划的扩展现实(XR)系统。我们的系统允许用户在规划机器人路径时创建和动态重新配置环境。在沉浸式三维XR环境中，用户获得更好的空间理解。XR还解锁了更广泛的自然交互能力，允许用户像在现实世界中一样抓取和调整环境中的物体，而不是使用鼠标和键盘将场景投影到二维计算机屏幕上。我们的系统与MoveIt（操作规划框架）集成，允许用户发送运动规划请求并在虚拟或增强现实中可视化 resulting机器人路径。我们提供广泛的交互模态，允许用户修改环境中的对象并与虚拟机器人交互。我们的系统允许操作员可视化机器人运动，确保其在整个环境中移动时具有期望的行为，在虚拟空间内没有碰撞风险，然后将规划好的路径部署到现实世界中的物理机器人。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决机器人运动规划中的交互式环境配置和路径规划问题。传统方法在二维屏幕上可视化环境，限制了用户的空间理解，难以准确感知机器人在三维空间中的行为。这个问题很重要，因为机器人正越来越多地应用于生活和工业环境，这些场景常需要人类重新配置环境，机器人需适应新环境。二维可视化增加了规划难度和错误风险，而直接在真实环境中测试路径可能存在碰撞风险。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先分析了传统二维可视化的局限性，然后考察了扩展现实技术在机器人领域的应用潜力。他们注意到游戏引擎如Unity提供了创建XR应用的便利性，并且有ROS接口可实现与机器人操作系统的通信。作者确实借鉴了现有工作，包括使用ROS2作为基础框架，集成MoveIt作为运动规划平台，利用Unity游戏引擎和OpenXR作为XR接口，以及使用Unity ROS-TCP Connector实现通信。但他们将这些技术以新方式组合，创造了更直观的交互体验。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是通过扩展现实技术提供沉浸式三维环境，让用户能自然地与虚拟机器人和环境物体交互，从而更直观地配置环境和规划路径。实现流程是：1)系统由XR界面和ROS节点组成；2)用户可在XR中导入机器人、添加/修改物体、与机器人交互、设置路径起点终点；3)XR中的变化通过ROS2同步到MoveIt规划场景；4)用户发起规划请求，请求被转发到MoveIt；5)MoveIt计算路径后返回并在XR中可视化；6)用户预览路径满意后可执行到物理机器人上。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)开源XR系统用于交互式运动规划；2)沉浸式可视化运动规划反馈；3)与ROS2集成，支持虚拟到物理的路径执行；4)提供系统演示。相比之前工作，ERUPT完全基于ROS2构建，而大多数现有系统不支持；允许用户与环境中物体交互，而不仅仅是规划路径；支持创建和修改虚拟物体评估不同环境配置，无需停止系统；提供更自然的三维交互方式；实现了XR环境与MoveIt规划场景的实时同步。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'ERUPT通过将扩展现实技术与机器人操作系统和运动规划框架相结合，创造了一个直观的三维交互环境，使用户能自然地配置环境、规划路径并预览运动，然后将结果部署到物理机器人上，大大提高了机器人路径规划的效率和安全性。'}


### 论文摘要

We propose the Extended Reality Universal Planning Toolkit (ERUPT), an extended reality (XR) system for interactive motion planning. Our system allows users to create and dynamically reconfigure environments while they plan robot paths. In immersive three-dimensional XR environments, users gain a greater spatial understanding. XR also unlocks a broader range of natural interaction capabilities, allowing users to grab and adjust objects in the environment similarly to the real world, rather than using a mouse and keyboard with the scene projected onto a two-dimensional computer screen. Our system integrates with MoveIt, a manipulation planning framework, allowing users to send motion planning requests and visualize the resulting robot paths in virtual or augmented reality. We provide a broad range of interaction modalities, allowing users to modify objects in the environment and interact with a virtual robot. Our system allows operators to visualize robot motions, ensuring desired behavior as it moves throughout the environment, without risk of collisions within a virtual space, and to then deploy planned paths on physical robots in the real world.

---

## 9. MIXER: Mixed Hyperspherical Random Embedding Neural Network for Texture Recognition

**论文链接:** [http://arxiv.org/abs/2510.03228v1](http://arxiv.org/abs/2510.03228v1)

**作者:** Ricardo T. Fares, Lucas C. Ribas

**发布时间:** 2025-10-03

### GPT解析

### 总结

本文提出了一种名为Mixer的新型随机神经网络，用于纹理表示学习。该方法结合了传统技术和基于学习的方法的优势，通过超球面随机嵌入和双分支学习模块捕获通道内和通道间关系，并通过新制定的优化问题构建丰富的纹理表示。

### 背景

随机神经网络在纹理识别任务中一直取得了显著成果，有效地结合了传统技术和基于学习的方法的优势。然而，现有方法主要集中在改进跨信息预测上，而没有对整体随机网络架构引入重大改进。

### 目的

提出一种名为Mixer的新型随机神经网络，用于纹理表示学习，以解决现有方法的局限性。

### 方法

该方法的核心是利用超球面随机嵌入，结合双分支学习模块来捕获通道内和通道间的关系，并通过新制定的优化问题进一步增强，以构建丰富的纹理表示。

### 主要发现

实验结果表明，所提出的方法在几个纯纹理基准测试中取得了有趣的结果，每个基准测试都具有不同的特征和挑战。

### 结论

Mixer方法在纹理表示学习方面表现出色，源代码将在发表后提供。

### 翻译

随机神经网络用于表示学习在纹理识别任务中一直取得了显著成果，有效地结合了传统技术和基于学习的方法的优势。然而，现有方法到目前为止主要专注于改进跨信息预测，而没有对整体随机网络架构引入重大改进。在本文中，我们提出了Mixer，一种用于纹理表示学习的新型随机神经网络。其核心方法是利用超球面随机嵌入，结合双分支学习模块来捕获通道内和通道间的关系，并通过新制定的优化问题进一步增强，以构建丰富的纹理表示。实验结果表明，所提出的方法在几个纯纹理基准测试中取得了有趣的结果，每个基准测试都具有不同的特征和挑战。源代码将在发表后提供。


### 论文摘要

Randomized neural networks for representation learning have consistently achieved prominent results in texture recognition tasks, effectively combining the advantages of both traditional techniques and learning-based approaches. However, existing approaches have so far focused mainly on improving cross-information prediction, without introducing significant advancements to the overall randomized network architecture. In this paper, we propose Mixer, a novel randomized neural network for texture representation learning. At its core, the method leverages hyperspherical random embeddings coupled with a dual-branch learning module to capture both intra- and inter-channel relationships, further enhanced by a newly formulated optimization problem for building rich texture representations. Experimental results have shown the interesting results of the proposed approach across several pure texture benchmarks, each with distinct characteristics and challenges. The source code will be available upon publication.

---

## 10. CVSM: Contrastive Vocal Similarity Modeling

**论文链接:** [http://arxiv.org/abs/2510.03025v1](http://arxiv.org/abs/2510.03025v1)

**作者:** Christos Garoufis, Athanasia Zlatintsi, Petros Maragos

**发布时间:** 2025-10-03

**备注:** 13 pages, 3 tables, 8 figures. Submitted article at IEEE Trans. on  Audio, Speech and Language Proc. (pre-print version)

### GPT解析

### 总结

本文介绍了CVSM（对比声音相似性建模），一种用于音频领域音乐信号表示学习的对比自监督方法，可在音乐和声音相似性建模中有效应用。

### 背景

大型无标注数据集在各个领域的可用性促进了多种通过自监督预训练学习表示方法的发展，这些方法能够为多个目标（下游）任务学习表示。

### 目的

开发一种用于音频领域音乐信号表示学习的对比自监督程序，可用于音乐和声音相似性建模。

### 方法

CVSM在对比框架下运行，最大化相同声音片段和包含该声音的音乐混合体之间的相似性。设计了两种方案：基于标签的协议（利用艺术家身份信息采样对比对）和无标签方案（从随机采样的声音和伴奏片段创建人工混合体并与来自同一音频段的声音配对）。

### 主要发现

通过CVSM学习到的表示在音乐和声音相似性建模中有效，在独立声音和完整音乐混合体上都优于多个基线。在预训练期间使用艺术家身份标签会导致更一致的性能，但无标签CVSM变体结合混合预训练也能达到相当的性能。

### 结论

CVSM方法在声音相似性建模方面表现出色，即使在没有标签的情况下也能获得与有标签方法相当的性能。

### 翻译

各个领域大型无标注数据集的可用性促进了多种通过自监督预学习为多个目标（下游）任务学习表示方法的发展。在本工作中，我们介绍了CVSM（对比声音相似性建模），一种用于音频领域音乐信号表示学习的对比自监督程序，可用于音乐和声音相似性建模。我们的方法在对比框架下运行，最大化相同声音片段和包含该声音的音乐混合体之间的相似性；我们设计了基于标签的协议，利用艺术家身份信息来采样对比对，以及无标签方案，涉及从随机采样的声音和伴奏片段创建人工混合体，这些混合体与来自同一音频段的声音配对。我们通过在一系列适当的下游任务上进行线性探测来客观评估声音相似性，并通过在基于查询的推荐设置中进行不同模型之间的成对比较的用户研究来主观评估。我们的结果表明，通过CVSM学习到的表示在音乐和声音相似性建模中有效，在独立声音和完整音乐混合体上都优于多个基线。此外，虽然在预训练期间使用艺术家身份标签会导致评估的下游任务和用户研究中整体更一致的性能，但结合真实和人工混合体的混合预训练的无标签CVSM变体在艺术家识别和感知声音相似性方面达到了与基于标签的方案相当的性能。


### 论文摘要

The availability of large, unlabeled datasets across various domains has contributed to the development of a plethora of methods that learn representations for multiple target (downstream) tasks through self-supervised pre-training. In this work, we introduce CVSM (Contrastive Vocal Similarity Modeling), a contrastive self-supervised procedure for music signal representation learning in the audio domain that can be utilized for musical and vocal similarity modeling. Our method operates under a contrastive framework, maximizing the similarity between vocal excerpts and musical mixtures containing the same vocals; we devise both a label-informed protocol, leveraging artist identity information to sample the contrastive pairs, and a label-agnostic scheme, involving artificial mixture creation from randomly sampled vocal and accompaniment excerpts, which are paired with vocals from the same audio segment. We evaluate our proposed method in measuring vocal similarity both objectively, through linear probing on a suite of appropriate downstream tasks, and subjectively, via conducting a user study consisting of pairwise comparisons between different models in a recommendation-by-query setting. Our results indicate that the representations learned through CVSM are effective in musical and vocal similarity modeling, outperforming numerous baselines across both isolated vocals and complete musical mixtures. Moreover, while the availability of artist identity labels during pre-training leads to overall more consistent performance both in the evaluated downstream tasks and the user study, a label-agnostic CVSM variant incorporating hybrid pre-training with real and artificial mixtures achieves comparable performance to the label-informed one in artist identification and perceived vocal similarity.

---

## 11. Fusing Multi- and Hyperspectral Satellite Data for Harmful Algal Bloom Monitoring with Self-Supervised and Hierarchical Deep Learning

**论文链接:** [http://arxiv.org/abs/2510.02763v1](http://arxiv.org/abs/2510.02763v1)

**作者:** Nicholas LaHaye, Kelly M. Luis, Michelle M. Gierach

**发布时间:** 2025-10-03

### GPT解析

### 总结

本文介绍了一种名为SIT-FUSE的自监督机器学习框架，用于利用多传感器卫星数据检测和绘制有害藻华的严重程度和物种分布，无需每个仪器的标记数据集即可生成产品。

### 背景

有害藻华监测对于海洋生态系统和人类健康至关重要，但在标记数据稀缺的环境中面临挑战。

### 目的

开发一种自监督机器学习框架，用于检测和绘制有害藻华的严重程度和物种分布，减少对标记数据的依赖。

### 方法

融合VIIRS、MODIS、Sentinel-3和PACE等仪器的反射率数据与TROPOMI太阳诱导荧光数据；采用自监督表示学习和分层深度聚类方法；在墨西哥湾和南加利福尼亚(2018-2025)的实地数据上进行验证。

### 主要发现

结果与总浮游植物、Karenia brevis、Alexandrium spp.和Pseudo-nitzschia spp.的测量值高度一致；该框架能够在标记稀缺环境中实现可扩展的有害藻华监测。

### 结论

该工作通过分层嵌入实现了探索性分析，是迈向将自监督学习应用于全球水生生物地球化学操作化的重要一步。

### 翻译

我们提出了一种自监督机器学习框架，用于利用多传感器卫星数据检测和绘制有害藻华的严重程度和物种分布。通过融合操作仪器(VIIRS、MODIS、Sentinel-3、PACE)的反射率数据与TROPOMI太阳诱导荧光，我们的框架名为SIT-FUSE，能够生成有害藻华严重程度和物种分布产品，而无需每个仪器的标记数据集。该框架采用自监督表示学习，通过分层深度聚类将浮游植物浓度和物种分割为可解释的类别，并在墨西哥湾和南加利福尼亚(2018-2025)的实地数据上进行了验证。结果显示与总浮游植物、Karenia brevis、Alexandrium spp.和Pseudo-nitzschia spp.的测量值高度一致。这项工作在标记稀缺环境中推进了可扩展的有害藻华监测，同时通过分层嵌入实现了探索性分析：这是将自监督学习操作化应用于全球水生生物地球化学的关键步骤。


### 论文摘要

We present a self-supervised machine learning framework for detecting and mapping harmful algal bloom (HAB) severity and speciation using multi-sensor satellite data. By fusing reflectance data from operational instruments (VIIRS, MODIS, Sentinel-3, PACE) with TROPOMI solar-induced fluorescence (SIF), our framework, called SIT-FUSE, generates HAB severity and speciation products without requiring per-instrument labeled datasets. The framework employs self-supervised representation learning, hierarchical deep clustering to segment phytoplankton concentrations and speciations into interpretable classes, validated against in-situ data from the Gulf of Mexico and Southern California (2018-2025). Results show strong agreement with total phytoplankton, Karenia brevis, Alexandrium spp., and Pseudo-nitzschia spp. measurements. This work advances scalable HAB monitoring in label-scarce environments while enabling exploratory analysis via hierarchical embeddings: a critical step toward operationalizing self-supervised learning for global aquatic biogeochemistry.

---

## 12. Hybrid-Collaborative Augmentation and Contrastive Sample Adaptive-Differential Awareness for Robust Attributed Graph Clustering

**论文链接:** [http://arxiv.org/abs/2510.02731v1](http://arxiv.org/abs/2510.02731v1)

**作者:** Tianxiang Zhao, Youqing Wang, Jinlu Wang, Jiapu Wang, Mingliang Cui, Junbin Gao, Jipeng Guo

**发布时间:** 2025-10-03

### GPT解析

### 总结

该论文提出了一种新型鲁棒属性图聚类方法(RAGC)，结合混合协同增强(HCA)和对比样本自适应差异感知(CSADA)策略，解决了现有对比属性图聚类方法中只关注节点级嵌入增强、忽略边级增强和样本对差异性的问题。

### 背景

对比属性图聚类(CAGC)因其在自监督表示学习和聚类方面的强大能力取得了显著成功，主要依赖于有效的数据增强和对比目标设置。

### 目的

解决现有CAGC方法忽略边级嵌入增强、不同粒度下节点级和边级嵌入增强交互以及平等对待所有对比样本对的问题，以提高方法的判别能力。

### 方法

提出RAGC方法，同时执行节点级和边级嵌入表示和增强，建立更全面的相似性测量标准；利用高置信度伪标签信息设计CSADA策略，自适应识别并差异化处理对比样本对；HCA和CSADA模块在良性循环中相互强化。

### 主要发现

同时考虑节点级和边级嵌入增强可以建立更全面的相似性测量标准；差异化处理难易正负样本对可以增强表示学习的判别能力；HCA和CSADA模块相互强化形成良性循环。

### 结论

在六个基准数据集上的全面图聚类评估表明，所提出的RAGC方法相对于几种最先进的CAGC方法具有更好的效果。

### 翻译

由于其强大的自监督表示学习和聚类能力，对比属性图聚类(CAGC)已取得巨大成功，这主要依赖于有效的数据增强和对比目标设置。然而，大多数CAGC方法使用边作为辅助信息来获取节点级嵌入表示，并且只关注节点级嵌入增强。这种方法忽略了边级嵌入增强以及不同粒度下节点级和边级嵌入增强之间的交互。此外，它们通常平等对待所有对比样本对，忽视了难易正负样本对之间的显著差异，这最终限制了它们的判别能力。为解决这些问题，提出了一种新型鲁棒属性图聚类(RAGC)，结合了混合协同增强(HCA)和对比样本自适应差异感知(CSADA)。首先，同时执行节点级和边级嵌入表示和增强，为后续对比学习建立更全面的相似性测量标准。反过来，判别相似性进一步有意识地指导边增强。其次，通过利用高置信度的伪标签信息，精心设计了CSADA策略，该策略自适应地识别所有对比样本对，并通过创新的权重调制函数差异化处理它们。HCA和CSADA模块在良性循环中相互强化，从而增强了表示学习的判别能力。在六个基准数据集上进行的全面图聚类评估证明了所提出的RAGC相对于几种最先进的CAGC方法的有效性。


### 论文摘要

Due to its powerful capability of self-supervised representation learning and clustering, contrastive attributed graph clustering (CAGC) has achieved great success, which mainly depends on effective data augmentation and contrastive objective setting. However, most CAGC methods utilize edges as auxiliary information to obtain node-level embedding representation and only focus on node-level embedding augmentation. This approach overlooks edge-level embedding augmentation and the interactions between node-level and edge-level embedding augmentations across various granularity. Moreover, they often treat all contrastive sample pairs equally, neglecting the significant differences between hard and easy positive-negative sample pairs, which ultimately limits their discriminative capability. To tackle these issues, a novel robust attributed graph clustering (RAGC), incorporating hybrid-collaborative augmentation (HCA) and contrastive sample adaptive-differential awareness (CSADA), is proposed. First, node-level and edge-level embedding representations and augmentations are simultaneously executed to establish a more comprehensive similarity measurement criterion for subsequent contrastive learning. In turn, the discriminative similarity further consciously guides edge augmentation. Second, by leveraging pseudo-label information with high confidence, a CSADA strategy is elaborately designed, which adaptively identifies all contrastive sample pairs and differentially treats them by an innovative weight modulation function. The HCA and CSADA modules mutually reinforce each other in a beneficent cycle, thereby enhancing discriminability in representation learning. Comprehensive graph clustering evaluations over six benchmark datasets demonstrate the effectiveness of the proposed RAGC against several state-of-the-art CAGC methods.

---

## 13. PGMEL: Policy Gradient-based Generative Adversarial Network for Multimodal Entity Linking

**论文链接:** [http://arxiv.org/abs/2510.02726v1](http://arxiv.org/abs/2510.02726v1)

**作者:** KM Pooja, Cheng Long, Aixin Sun

**发布时间:** 2025-10-03

### GPT解析

### 总结

论文提出了PGMEL，一种基于策略梯度的生成对抗网络，用于多模态实体链接，通过生成高质量负样本来改进表示学习，实验证明该方法优于现有技术。

### 背景

实体链接任务因其众多潜在应用而受到广泛关注。最近提出了各种多模态实体链接技术，旨在利用文本和视觉模态学习全面的嵌入表示。

### 目的

填补现有文献中关于高质量负样本选择在多模态实体链接框架中未被探索的空白。

### 方法

在生成对抗框架下解决多模态实体链接问题，生成器负责生成高质量负样本，判别器负责度量学习任务。由于生成样本是离散过程，使用策略梯度技术优化生成器，提出了基于策略梯度的生成对抗网络用于多模态实体链接(PGMEL)。

### 主要发现

PGMEL通过选择具有挑战性的负样本来学习有意义的表示，在Wiki-MEL、Richpedia-MEL和WikiDiverse数据集上的实验结果证明PGMEL优于最先进的方法。

### 结论

PGMEL方法在多模态实体链接任务中表现优异，为负样本选择提供了新的解决方案。

### 翻译

实体链接任务涉及将提及与知识图中的相应实体关联起来，由于其众多潜在应用而受到显著关注。最近，提出了各种多模态实体链接技术，旨在利用文本和视觉模态学习全面的嵌入表示。高质量负样本的选择可能在度量/表示学习中发挥关键作用。然而，据我们所知，这种可能性在现有文献的多模态实体链接框架中尚未被探索。为填补这一空白，我们在生成对抗设置下解决多模态实体链接问题，其中生成器负责生成高质量负样本，判别者负责度量学习任务。由于生成器参与生成样本，这是一个离散过程，我们使用策略梯度技术对其进行优化，并提出了基于策略梯度的生成对抗网络用于多模态实体链接(PGMEL)。基于Wiki-MEL、Richpedia-MEL和WikiDiverse数据集的实验结果表明，PGMEL通过选择具有挑战性的负样本来学习有意义的表示，并优于最先进的方法。


### 论文摘要

The task of entity linking, which involves associating mentions with their respective entities in a knowledge graph, has received significant attention due to its numerous potential applications. Recently, various multimodal entity linking (MEL) techniques have been proposed, targeted to learn comprehensive embeddings by leveraging both text and vision modalities. The selection of high-quality negative samples can potentially play a crucial role in metric/representation learning. However, to the best of our knowledge, this possibility remains unexplored in existing literature within the framework of MEL. To fill this gap, we address the multimodal entity linking problem in a generative adversarial setting where the generator is responsible for generating high-quality negative samples, and the discriminator is assigned the responsibility for the metric learning tasks. Since the generator is involved in generating samples, which is a discrete process, we optimize it using policy gradient techniques and propose a policy gradient-based generative adversarial network for multimodal entity linking (PGMEL). Experimental results based on Wiki-MEL, Richpedia-MEL and WikiDiverse datasets demonstrate that PGMEL learns meaningful representation by selecting challenging negative samples and outperforms state-of-the-art methods.

---

## 14. STSM-FiLM: A FiLM-Conditioned Neural Architecture for Time-Scale Modification of Speech

**论文链接:** [http://arxiv.org/abs/2510.02672v1](http://arxiv.org/abs/2510.02672v1)

**作者:** Dyah A. M. G. Wisnu, Ryandhimas E. Zezario, Stefano Rini, Fo-Rui Li, Yan-Tsung Peng, Hsin-Min Wang, Yu Tsao

**发布时间:** 2025-10-03

### GPT解析

### 总结

本研究提出了一种名为STSM-FILM的全神经架构，用于语音时间尺度修改，通过Feature-Wise Linear Modulation (FiLM)实现基于连续速度因子的条件化模型，能够在广泛的时间尺度因子范围内产生感知上一致的音频输出。

### 背景

语音时间尺度修改(TSM)旨在改变音频播放速率而不改变音调。经典方法如WSOLA在非平稳或极端拉伸条件下常引入伪影。

### 目的

提出一种改进的神经TSM架构，利用FiLM条件化提高模型在多种时间尺度因子下的表现。

### 方法

使用WSOLA生成的输出来监督网络训练，使模型学习模仿对齐和合成行为；探索了四种编码器-解码器变体：STFT-HiFiGAN、WavLM-HiFiGAN、Whisper-HiFiGAN和EnCodec。

### 主要发现

STSM-FILM能够在广泛的时间尺度因子范围内产生感知上一致的输出，基于FiLM的条件化提高了神经TSM模型的泛化能力和灵活性。

### 结论

基于FiLM的条件化显示出改进神经TSM模型泛化能力和灵活性的潜力。

### 翻译

语音时间尺度修改(TSM)旨在改变音频的播放速率而不改变其音调。虽然像基于波形相似性的重叠相加(WSOLA)这样的经典方法提供了强大的基线，但它们在非平稳或极端拉伸条件下常常引入伪影。我们提出了STSM-FILM - 一种全神经架构，它结合了特征级线性调制(FiLM)，使模型能够根据连续的速度因子进行条件化。通过使用WSOLA生成的输出来监督网络，STSM-FILM学习模仿对齐和合成行为，同时受益于通过深度学习学习到的表示。我们探索了四种编码器-解码器变体：STFT-HiFiGAN、WavLM-HiFiGAN、Whisper-HiFiGAN和EnCodec，并证明STSM-FILM能够在广泛的时间尺度因子范围内产生感知上一致的输出。总体而言，我们的结果证明了基于FiLM的条件化在提高神经TSM模型的泛化能力和灵活性方面的潜力。


### 论文摘要

Time-Scale Modification (TSM) of speech aims to alter the playback rate of audio without changing its pitch. While classical methods like Waveform Similarity-based Overlap-Add (WSOLA) provide strong baselines, they often introduce artifacts under non-stationary or extreme stretching conditions. We propose STSM-FILM - a fully neural architecture that incorporates Feature-Wise Linear Modulation (FiLM) to condition the model on a continuous speed factor. By supervising the network using WSOLA-generated outputs, STSM-FILM learns to mimic alignment and synthesis behaviors while benefiting from representations learned through deep learning. We explore four encoder-decoder variants: STFT-HiFiGAN, WavLM-HiFiGAN, Whisper-HiFiGAN, and EnCodec, and demonstrate that STSM-FILM is capable of producing perceptually consistent outputs across a wide range of time-scaling factors. Overall, our results demonstrate the potential of FiLM-based conditioning to improve the generalization and flexibility of neural TSM models.

---

## 15. MINERVA: Mutual Information Neural Estimation for Supervised Feature Selection

**论文链接:** [http://arxiv.org/abs/2510.02610v1](http://arxiv.org/abs/2510.02610v1)

**作者:** Taurai Muvunzaa, Egor Kraev, Pere Planell-Morell, Alexander Y. Shestopaloff

**发布时间:** 2025-10-02

**备注:** 23 pages

### GPT解析

### 总结

论文提出了MINERVA，一种基于互信息神经估计的监督特征选择方法，能有效处理高阶特征交互关系。

### 背景

现有特征过滤器依赖统计成对依赖性指标建模特征-目标关系，但当目标依赖于高阶特征交互而非单个特征贡献时，这种方法可能失效。

### 目的

引入Mutual Information Neural Estimation Regularized Vetting Algorithm (MINERVA)，一种基于特征和目标间互信息神经估计的新型监督特征选择方法。

### 方法

使用神经网络参数化互信息近似，通过添加稀疏诱导正则化器的精心设计损失函数进行特征选择；采用两阶段流程解耦表示学习和特征选择；通过评估特征子集作为集成来捕获复杂关系。

### 主要发现

展示了文献中很少捕捉到的普遍依赖结构，证明所提方法能有效捕获这些复杂的特征-目标关系。

### 结论

在合成和真实欺诈数据集上的实验结果验证了该方法的有效性及其执行精确解的能力。

### 翻译

现有的特征过滤器依赖于统计成对依赖性指标来建模特征-目标关系，但当目标依赖于高阶特征交互而非单个特征贡献时，这种方法可能会失效。我们引入了互信息神经估计审查算法(MINERVA)，一种基于特征和目标之间互信息神经估计的新型监督特征选择方法。我们使用神经网络参数化互信息的近似，并使用精心设计的损失函数进行特征选择，该函数添加了稀疏诱导正则化器。我们的方法通过两阶段流程实现，将表示学习与特征选择解耦，确保更好的泛化能力和更准确的特征重要性表达。我们展示了文献中很少捕捉到的普遍依赖结构示例，并通过评估特征子集作为集成，证明我们提出的方法能有效捕获这些复杂的特征-目标关系。在合成和真实欺诈数据集上的实验结果证明了我们方法的有效性及其执行精确解的能力。


### 论文摘要

Existing feature filters rely on statistical pair-wise dependence metrics to model feature-target relationships, but this approach may fail when the target depends on higher-order feature interactions rather than individual contributions. We introduce Mutual Information Neural Estimation Regularized Vetting Algorithm (MINERVA), a novel approach to supervised feature selection based on neural estimation of mutual information between features and targets. We paramaterize the approximation of mutual information with neural networks and perform feature selection using a carefully designed loss function augmented with sparsity-inducing regularizers. Our method is implemented in a two-stage process to decouple representation learning from feature selection, ensuring better generalization and a more accurate expression of feature importance. We present examples of ubiquitous dependency structures that are rarely captured in literature and show that our proposed method effectively captures these complex feature-target relationships by evaluating feature subsets as an ensemble. Experimental results on synthetic and real-life fraud datasets demonstrate the efficacy of our method and its ability to perform exact solutions.

---

## 16. Latent Multi-view Learning for Robust Environmental Sound Representations

**论文链接:** [http://arxiv.org/abs/2510.02500v1](http://arxiv.org/abs/2510.02500v1)

**作者:** Sivan Sing, Julia Wilkins, Magdalena Fuentes, Juan Pablo Bello

**发布时间:** 2025-10-02

**备注:** Accepted to DCASE 2025 Workshop. 4+1 pages, 2 figures, 2 tables

### GPT解析

### 总结

本文提出了一种多视图学习框架，将对比原则整合到生成流程中，以捕捉声音源和设备信息，提高环境声音表征学习效果。

### 背景

自监督学习(SSL)方法如对比方法和生成方法已使用未标记数据推动了环境声音表征学习的发展，但这些方法如何在统一框架中互补仍然相对未被探索。

### 目的

开发一个多视图学习框架，整合对比原则与生成方法，以更有效地捕捉声音源和设备信息。

### 方法

该方法将压缩的音频潜在编码为视图特定和视图共享的子空间，由两个自监督目标引导：子空间间的对比学习以实现有针对性的信息流，以及整体信息保存的重建。

### 主要发现

在城市声音传感器网络数据集上的评估表明，该方法在声音源和传感器分类任务中展现出优于传统SSL技术的下游性能；此外，该模型在变化训练配置下具有在结构化潜在空间中解离环境声音属性的潜力。

### 结论

多视图学习框架结合对比学习和生成方法能够更好地捕捉声音源和设备信息，提高下游性能，并可能解离环境声音属性。

### 翻译

自监督学习(SSL)方法，如对比方法和生成方法，已经使用未标记数据推动了环境声音表征学习的发展。然而，这些方法如何在统一框架中互补仍然相对未被探索。在这项工作中，我们提出了一种多视图学习框架，将对比原则整合到生成流程中以捕捉声音源和设备信息。我们的方法将压缩的音频潜在编码为视图特定和视图共享的子空间，由两个自监督目标引导：子空间之间的对比学习以实现有针对性的信息流，以及整体信息保存的重建。我们在城市声音传感器网络数据集上评估了我们的方法，用于声音源和传感器分类，展示了与传统SSL技术相比改进的下游性能。此外，我们还研究了模型在变化训练配置下在结构化潜在空间中解离环境声音属性的潜力。


### 论文摘要

Self-supervised learning (SSL) approaches, such as contrastive and generative methods, have advanced environmental sound representation learning using unlabeled data. However, how these approaches can complement each other within a unified framework remains relatively underexplored. In this work, we propose a multi-view learning framework that integrates contrastive principles into a generative pipeline to capture sound source and device information. Our method encodes compressed audio latents into view-specific and view-common subspaces, guided by two self-supervised objectives: contrastive learning for targeted information flow between subspaces, and reconstruction for overall information preservation. We evaluate our method on an urban sound sensor network dataset for sound source and sensor classification, demonstrating improved downstream performance over traditional SSL techniques. Additionally, we investigate the model's potential to disentangle environmental sound attributes within the structured latent space under varied training configurations.

---

## 17. Learning to Interact in World Latent for Team Coordination

**论文链接:** [http://arxiv.org/abs/2509.25550v3](http://arxiv.org/abs/2509.25550v3)

**作者:** Dongsu Lee, Daehee Lee, Yaru Niu, Honguk Woo, Amy Zhang, Ding Zhao

**发布时间:** 2025-09-29

**备注:** Web: https://dongsuleetech.github.io/projects/IWoL/

### GPT解析

### 总结

本文提出了交互世界潜变量(IWoL)这一新型表示学习框架，用于促进多智能体强化学习(MARL)中的团队协调。该框架通过直接建模通信协议，构建能够同时捕获智能体间关系和任务特定世界信息的可学习表示空间，实现完全去中心化的执行和隐式协调，同时避免显式消息传递的固有缺点。

### 背景

在多智能体强化学习中，为团队协调构建有效表示具有挑战性，这源于多智能体交互产生的复杂动态以及由局部观察引起的不完整信息。

### 目的

开发一种表示学习框架，有效解决多智能体强化学习中的团队协调问题，同时避免显式消息传递的缺点。

### 方法

构建可学习的表示空间，通过直接建模通信协议，联合捕获智能体间关系和任务特定的世界信息。这种表示既可以作为每个智能体的隐式潜变量，也可以作为通信的显式消息。

### 主要发现

在四个具有挑战性的MARL基准测试中，评估了两种变体，结果表明IWoL为团队协调提供了简单而强大的关键。此外，该表示可以与现有MARL算法结合使用，进一步提高性能。

### 结论

IWoL框架有效解决了多智能体强化学习中的团队协调问题，通过隐式协调避免了显式消息传递的缺点，并具有良好的适应性和扩展性。

### 翻译

这项工作提出了一种新的表示学习框架——交互世界潜变量(IWoL)，以促进多智能体强化学习(MARL)中的团队协调。为团队协调构建有效的表示是一个具有挑战性的问题，这是由于多智能体交互产生的复杂动态以及由局部观察引起的不完整信息。我们的关键见解是构建一个可学习的表示空间，通过直接建模通信协议，联合捕获智能体间关系和任务特定的世界信息。我们保持完全去中心化的执行和隐式协调，同时避免了显式消息传递的固有缺点，例如决策较慢、易受恶意攻击者攻击以及对带宽限制的敏感性。在实践中，我们的表示不仅可以作为每个智能体的隐式潜变量，也可以作为通信的显式消息。在四个具有挑战性的MARL基准测试中，我们评估了两种变体，并证明IWoL为团队协调提供了一个简单而强大的关键。此外，我们证明了我们的表示可以与现有的MARL算法结合，以进一步提高它们的性能。


### 论文摘要

This work presents a novel representation learning framework, interactive world latent (IWoL), to facilitate team coordination in multi-agent reinforcement learning (MARL). Building effective representation for team coordination is a challenging problem, due to the intricate dynamics emerging from multi-agent interaction and incomplete information induced by local observations. Our key insight is to construct a learnable representation space that jointly captures inter-agent relations and task-specific world information by directly modeling communication protocols. This representation, we maintain fully decentralized execution with implicit coordination, all while avoiding the inherent drawbacks of explicit message passing, e.g., slower decision-making, vulnerability to malicious attackers, and sensitivity to bandwidth constraints. In practice, our representation can be used not only as an implicit latent for each agent, but also as an explicit message for communication. Across four challenging MARL benchmarks, we evaluate both variants and show that IWoL provides a simple yet powerful key for team coordination. Moreover, we demonstrate that our representation can be combined with existing MARL algorithms to further enhance their performance.

---

## 18. Wave-GMS: Lightweight Multi-Scale Generative Model for Medical Image Segmentation

**论文链接:** [http://arxiv.org/abs/2510.03216v1](http://arxiv.org/abs/2510.03216v1)

**作者:** Talha Ahmed, Nehal Ahmed Shaikh, Hassan Mohy-ud-Din

**发布时间:** 2025-10-03

**备注:** 5 pages, 1 figure, 4 tables; Submitted to IEEE Conference for  possible publication

### GPT解析

### 总结

本研究提出Wave-GMS，一种轻量级且高效的多尺度生成模型，用于医学图像分割，可在有限内存的GPU上实现高性能和大批量训练。

### 背景

为了在医疗机构公平部署AI工具，需要高性能的深度分割网络，这些网络可以在具有有限内存和大数据批处理规模的低成本GPU上进行训练。

### 目的

开发一种轻量级且高效的医学图像分割模型，能够在资源受限的环境中有效运行。

### 方法

Wave-GMS模型具有较少的可训练参数，不需要加载内存密集型的预训练视觉基础模型，并支持在内存有限的GPU上使用大数据批处理规模进行训练。

### 主要发现

在四个公开数据集上进行的实验表明，Wave-GMS实现了最先进的分割性能，具有出色的跨域泛化能力，只需要约260万可训练参数。

### 结论

Wave-GMS是一种轻量级且高效的医学图像分割解决方案，能够在有限的计算资源上实现高性能，有助于AI工具在医疗环境中的公平部署。

### 翻译

为了在医院和医疗机构公平部署AI工具，我们需要高性能的深度分割网络，这些网络可以在具有有限内存和大数据批处理规模的低成本GPU上进行训练。在这项工作中，我们提出了Wave-GMS，一种用于医学图像分割的轻量级且高效的多尺度生成模型。Wave-GMS的可训练参数数量显著减少，不需要加载内存密集型的预训练视觉基础模型，并支持在内存有限的GPU上使用大数据批处理规模进行训练。我们在四个公开可用的数据集上进行了广泛的实验，证明Wave-GMS实现了最先进的分割性能，具有出色的跨域泛化能力，只需要约260万可训练参数。代码可在https://github.com/ATPLab-LUMS/Wave-GMS获取。


### 论文摘要

For equitable deployment of AI tools in hospitals and healthcare facilities, we need Deep Segmentation Networks that offer high performance and can be trained on cost-effective GPUs with limited memory and large batch sizes. In this work, we propose Wave-GMS, a lightweight and efficient multi-scale generative model for medical image segmentation. Wave-GMS has a substantially smaller number of trainable parameters, does not require loading memory-intensive pretrained vision foundation models, and supports training with large batch sizes on GPUs with limited memory. We conducted extensive experiments on four publicly available datasets (BUS, BUSI, Kvasir-Instrument, and HAM10000), demonstrating that Wave-GMS achieves state-of-the-art segmentation performance with superior cross-domain generalizability, while requiring only ~2.6M trainable parameters. Code is available at https://github.com/ATPLab-LUMS/Wave-GMS.

---

## 19. Dynamic Prompt Generation for Interactive 3D Medical Image Segmentation Training

**论文链接:** [http://arxiv.org/abs/2510.03189v1](http://arxiv.org/abs/2510.03189v1)

**作者:** Tidiane Camaret Ndir, Alexander Pfefferle, Robin Tibor Schirrmeister

**发布时间:** 2025-10-03

### GPT解析

### 总结

本文提出了一种针对交互式3D生物医学图像分割的训练策略，结合动态体积提示生成和内容感知自适应裁剪，优化了图像编码器的使用，解决了单GPU上从顺序细化反馈学习的计算挑战。

### 背景

当前基础模型在交互式3D生物医学图像分割中存在局限，要么缺乏体积感知能力，要么交互能力有限，需要能够基于用户提示迭代优化预测的高效模型。

### 目的

开发一种训练策略，优化图像编码器的使用，模拟真实用户交互模式，并解决在单GPU上从顺序细化反馈学习的计算挑战。

### 方法

提出结合动态体积提示生成与内容感知自适应裁剪的训练策略，使用nnInteractive分割模型的公开可用权重初始化网络。

### 主要发现

在'交互式3D生物医学图像分割基础模型'竞赛中，方法表现强劲，平均最终Dice得分为0.6385，归一化表面距离为0.6614，Dice曲线下面积指标为2.4799，NSD曲线下面积指标为2.5671。

### 结论

所提出的方法在交互式3D生物医学图像分割任务中表现出色，能够有效处理体积数据并响应用户交互。

### 翻译

交互式3D生物医学图像分割需要能够基于用户提示迭代优化预测的高效模型。当前的基础模型要么缺乏体积感知能力，要么交互能力有限。我们提出了一种结合动态体积提示生成和内容感知自适应裁剪的训练策略，以优化图像编码器的使用。我们的方法在训练过程中模拟真实的用户交互模式，同时解决了在单GPU上从顺序细化反馈学习的计算挑战。为了高效训练，我们使用nnInteractive分割模型的公开可用权重初始化我们的网络。在'交互式3D生物医学图像分割基础模型'竞赛中的评估显示出强劲的性能，平均最终Dice得分为0.6385，归一化表面距离为0.6614，Dice曲线下面积指标为2.4799，NSD曲线下面积指标为2.5671。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决交互式3D医学图像分割中高效模型训练的问题。当前的基础模型要么缺乏体积感知能力，要么交互能力有限。这个问题在现实中非常重要，因为准确的3D医学图像分割对临床诊断和治疗规划至关重要，而生物医学数据集的复杂性和体积不断增加，需要能够有效整合用户反馈的交互式分割方法。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先分析了现有方法的局限性：SAM/MedSAM缺乏迭代细化能力；SegVol不支持点击修正；VISTA3D不支持边界框；SAM-Med3D分辨率过低；nnInteractive将边界框限制在2D切片。作者借鉴了nnInteractive的架构，但针对其局限性进行了改进。设计思路是解决交互式模型训练中的循环依赖问题：生成真实训练信号需要迭代预测，而模型又需要从这些自生成交互中学习，为此作者设计了无梯度计算和有梯度计算的两阶段训练策略。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是动态提示生成和内容感知自适应裁剪。动态提示生成通过识别预测和真实值之间的最大错误区域来模拟用户点击；内容感知裁剪根据解剖结构大小动态调整处理区域。整体流程：1)采用3D残差编码器U-Net架构；2)将图像和各类提示(边界框、点击、先前分割)作为多通道输入；3)两阶段交互模拟-先无梯度生成点击提示，再有梯度更新模型；4)基于错误分析生成点击位置；5)动态裁剪确保捕获完整结构；6)使用Dice损失和交叉熵的组合损失函数。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点：1)动态提示生成策略，模拟真实用户交互模式；2)内容感知动态裁剪，优化视场大小；3)两阶段交互模拟解决训练循环依赖问题。相比之前工作：不同于VISTA3D仅支持点击，支持边界框和迭代细化；不同于SegVol仅支持边界框，支持点击修正；不同于SAM-Med3D低分辨率，支持高精度3D分割；不同于nnInteractive的2D边界框，使用完整3D边界框充分利用体积信息。论文还解决了现有方法不充分利用所有可用提示信息的问题。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '这篇论文提出了一种动态提示生成策略，通过模拟真实用户交互和内容感知自适应裁剪，显著提高了交互式3D医学图像分割的准确性和效率，特别是在有边界框初始化的情况下。'}


### 论文摘要

Interactive 3D biomedical image segmentation requires efficient models that can iteratively refine predictions based on user prompts. Current foundation models either lack volumetric awareness or suffer from limited interactive capabilities. We propose a training strategy that combines dynamic volumetric prompt generation with content-aware adaptive cropping to optimize the use of the image encoder. Our method simulates realistic user interaction patterns during training while addressing the computational challenges of learning from sequential refinement feedback on a single GPU. For efficient training, we initialize our network using the publicly available weights from the nnInteractive segmentation model. Evaluation on the \textbf{Foundation Models for Interactive 3D Biomedical Image Segmentation} competition demonstrates strong performance with an average final Dice score of 0.6385, normalized surface distance of 0.6614, and area-under-the-curve metrics of 2.4799 (Dice) and 2.5671 (NSD).

---

## 20. Geometry Meets Vision: Revisiting Pretrained Semantics in Distilled Fields

**论文链接:** [http://arxiv.org/abs/2510.03104v1](http://arxiv.org/abs/2510.03104v1)

**作者:** Zhiting Mei, Ola Shorinwa, Anirudha Majumdar

**发布时间:** 2025-10-03

### GPT解析

### 总结

该研究探讨了在辐射场中使用几何基础语义特征的效果，与纯视觉特征进行了比较，并提出了一个名为SPINE的新型辐射场反转框架。

### 背景

语义蒸馏在辐射场中已显著推动了开放词汇机器人策略的发展，前人工作已证明视觉语义特征在高斯溅射和神经辐射场中的有效性，但几何基础在蒸馏场中的潜在优势仍不明确。

### 目的

研究几何基础语义特征在蒸馏场中的优势，并回答三个关键问题：空间基础是否能产生更高保真度的几何感知语义特征？几何基础是否能改善语义目标定位？几何基础是否能实现更高精度的辐射场反转？

### 方法

提出SPINE框架，用于在没有初始猜测的情况下反转辐射场，包含两个核心组件：使用蒸馏语义的粗反转和使用基于光度优化的细反转。

### 主要发现

几何基础骨干网络的图像特征包含更精细的结构细节；在语义目标定位任务中没有观察到显著差异；姿态估计精度随着几何基础特征的使用而降低；视觉特征为更广泛的下游任务提供了更大的通用性。

### 结论

虽然几何基础特征包含更多几何细节，但视觉-only特征在更广泛的下游任务中具有更好的通用性。未来需要探索有效的几何基础策略，以增强预训练语义特征的通用性和性能。

### 翻译

辐射场中的语义蒸馏推动了开放词汇机器人策略的重大进展，如在操作和导航方面，这基于来自大型视觉模型的预训练语义。虽然先前的工作已经证明了视觉语义特征（如DINO和CLIP）在高斯溅射和神经辐射场中的有效性，但在蒸馏场中几何基础的潜在优势仍然是一个开放问题。原则上，视觉-几何特征对于空间任务（如姿态估计）似乎非常有前景，这引发了一个问题：几何基础的语义特征在蒸馏场中是否具有优势？具体来说，我们提出了三个关键问题：首先，空间基础是否能产生更高保真度的几何感知语义特征？我们发现，与对应特征相比，来自几何基础骨干网络的图像特征包含更精细的结构细节。其次，几何基础是否能改善语义目标定位？我们观察到此任务没有显著差异。第三，几何基础是否能实现更高精度的辐射场反转？鉴于先前工作的局限性及其语义集成的缺乏，我们提出了一种新颖的框架SPINE，用于在没有初始猜测的情况下反转辐射场，包含两个核心组件：使用蒸馏语义的粗反转和使用基于光度优化的细反转。令人惊讶的是，我们发现姿态估计精度随着几何基础特征的使用而降低。我们的结果表明，视觉-only特征为更广泛的下游任务提供了更大的通用性，尽管几何基础特征包含更多几何细节。值得注意的是，我们的研究结果强调了未来研究的必要性，即研究有效的几何基础策略，以增强预训练语义特征的通用性和性能。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要研究在蒸馏辐射场中，基于几何的语义特征(visual-geometry semantic features)与纯视觉语义特征(visual-only semantic features)相比是否具有优势。这个问题在机器人学和计算机视觉领域非常重要，因为辐射场结合预训练语义模型为机器人提供了理解环境的能力，而理解哪种类型的语义特征更适合各种下游任务对于构建更有效的机器人系统至关重要。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者注意到现有工作主要使用纯视觉语义特征在辐射场中进行语义蒸馏，而基于几何的特征在空间任务中可能更有潜力，但其在蒸馏辐射场中的性能尚不清楚。他们设计了三个关键问题来系统比较这两种特征。该方法借鉴了现有的辐射场表示方法(如NeRF和Gaussian Splatting)、预训练的视觉模型(DINO、CLIP和VGGT)、语义蒸馏技术，以及视角n点(PnP)优化方法。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是研究视觉几何语义特征在蒸馏辐射场中的相对性能，评估它们在三个关键任务中的表现，并提出一种新的辐射场反转框架。整体流程包括：1)从纯视觉模型和视觉几何模型提取语义特征；2)将语义蒸馏到辐射场中，学习语义场映射3D点到特征空间；3)通过语义内容分析(PCA可视化和GFF指标)、语义目标定位(使用CLIP查询和评估指标)和辐射场反转(SPINE框架)来评估性能；4)在多个数据集上验证不同特征的表现。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)首次系统性地比较视觉几何语义特征与纯视觉语义特征在蒸馏辐射场中的性能；2)提出SPINE框架，实现无需初始猜测的辐射场反转；3)引入几何保真度因子(GFF)量化语义特征中的几何内容；4)发现视觉几何特征在大多数下游任务中并不总是优于纯视觉特征。相比之前工作，这项研究首次将几何增强与语义蒸馏结合，提供了对几何增强语义特征在辐射场中实际性能的深入理解。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '这项研究系统性地比较了视觉几何语义特征与纯视觉语义特征在蒸馏辐射场中的性能，发现尽管视觉几何特征包含更丰富的几何细节，但纯视觉特征在大多数下游任务中表现更好，并提出了无需初始猜测的辐射场反转新方法SPINE。'}


### 论文摘要

Semantic distillation in radiance fields has spurred significant advances in open-vocabulary robot policies, e.g., in manipulation and navigation, founded on pretrained semantics from large vision models. While prior work has demonstrated the effectiveness of visual-only semantic features (e.g., DINO and CLIP) in Gaussian Splatting and neural radiance fields, the potential benefit of geometry-grounding in distilled fields remains an open question. In principle, visual-geometry features seem very promising for spatial tasks such as pose estimation, prompting the question: Do geometry-grounded semantic features offer an edge in distilled fields? Specifically, we ask three critical questions: First, does spatial-grounding produce higher-fidelity geometry-aware semantic features? We find that image features from geometry-grounded backbones contain finer structural details compared to their counterparts. Secondly, does geometry-grounding improve semantic object localization? We observe no significant difference in this task. Thirdly, does geometry-grounding enable higher-accuracy radiance field inversion? Given the limitations of prior work and their lack of semantics integration, we propose a novel framework SPINE for inverting radiance fields without an initial guess, consisting of two core components: coarse inversion using distilled semantics, and fine inversion using photometric-based optimization. Surprisingly, we find that the pose estimation accuracy decreases with geometry-grounded features. Our results suggest that visual-only features offer greater versatility for a broader range of downstream tasks, although geometry-grounded features contain more geometric detail. Notably, our findings underscore the necessity of future research on effective strategies for geometry-grounding that augment the versatility and performance of pretrained semantic features.

---

## 21. Towards Scalable and Consistent 3D Editing

**论文链接:** [http://arxiv.org/abs/2510.02994v1](http://arxiv.org/abs/2510.02994v1)

**作者:** Ruihao Xia, Yang Tang, Pan Zhou

**发布时间:** 2025-10-03

### GPT解析

### 总结

该研究提出了一种新的3D编辑框架，通过改进数据集和模型方法解决了现有3D编辑技术的局限性，实现了更快速、精确且一致的3D资产编辑效果。

### 背景

3D编辑（修改3D资产几何形状或外观）在沉浸式内容创作、数字娱乐和AR/VR中有广泛应用，但与2D编辑不同，它面临跨视图一致性、结构完整性和细粒度可控性等挑战。现有方法通常速度慢、易产生几何失真，或依赖易出错且不切实际的手动精确3D掩码。

### 目的

解决现有3D编辑方法的局限性，提高编辑的速度、质量和可控性，实现实用且可扩展的3D编辑解决方案。

### 方法

在数据方面，引入3DEditVerse，最大的成对3D编辑基准，包含116,309个高质量训练对和1,500个精选测试对，通过姿势驱动的几何编辑和基础模型引导的外观编辑构建；在模型方面，提出3DEditFormer，一种3D结构保持的条件Transformer，通过双引导注意力和时间自适应门控增强图像到3D生成，无需辅助3D掩码即可分离可编辑区域与保留结构。

### 主要发现

提出的3DEditFormer框架在定量和定性评估上都优于现有最先进的基线方法，实现了更精确、一致的3D编辑效果，同时保持了原始3D资产的结构完整性。

### 结论

3DEditFormer和3DEditVerse为3D编辑领域建立了新的实用且可扩展的标准，相关数据集和代码将公开发布，项目地址为https://www.lv-lab.org/3DEditFormer/

### 翻译

3D编辑——对3D资产几何形状或外观进行局部修改的任务——在沉浸式内容创作、数字娱乐和AR/VR中有广泛应用。然而，与2D编辑不同，由于需要跨视图一致性、结构完整性和细粒度可控性，3D编辑仍然具有挑战性。现有方法通常速度慢，容易产生几何失真，或依赖于容易出错且不切实际的手动精确3D掩码。为应对这些挑战，我们在数据和模型两方面都进行了创新。在数据方面，我们引入了3DEditVerse，迄今为止最大的成对3D编辑基准，包含116,309个高质量训练对和1,500个精选测试对。通过姿势驱动的几何编辑和基础模型引导的外观编辑的互补流程构建，3DEditVerse确保了编辑的局部性、多视图一致性和语义对齐。在模型方面，我们提出了3DEditFormer，一种3D结构保持的条件Transformer。通过双引导注意力和时间自适应门控增强图像到3D生成，3DEditFormer将可编辑区域与保留结构分离，无需辅助3D掩码即可实现精确且一致的编辑。大量实验证明，我们的框架在定量和定性上都优于最先进的基线，为实用且可扩展的3D编辑建立了新标准。数据集和代码将发布。项目：https://www.lv-lab.org/3DEditFormer/

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决3D编辑的挑战，包括跨视角一致性、结构保真度和细粒度可控性。这个问题在现实中非常重要，因为3D编辑在沉浸式内容创作、数字娱乐和AR/VR等领域有广泛应用。与2D编辑相比，3D编辑更困难，而实用的3D编辑系统可以让用户像使用2D工具一样直观地进行修改，大大提高创作效率。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先分析了3D编辑的两个基本瓶颈：缺乏大规模配对3D编辑数据集和难以实现结构保持的编辑。他们创建了3DEditVerse数据集，通过姿态驱动的几何编辑和基础模型引导的外观编辑两个互补管道构建。模型方面，他们基于现有的图像到3D的Trellis模型扩展，设计了3DEditFormer，借鉴了现有基础模型如DeepSeek-R1、Flux、Qwen-VL等，并利用了掩码引导的重绘策略和扩散模型思想。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是通过注入源资产的多阶段特征到目标生成中，分离可编辑区域和保持的结构。使用双引导注意力机制（一个关注细粒度结构特征，另一个关注语义过渡特征）和时间自适应门控机制来平衡不同扩散阶段的影响。整体流程包括：使用3DEditVerse数据集训练；从源资产提取细粒度结构和语义过渡特征；通过双引导注意力块将源特征注入目标生成；使用时间自适应门控机制平衡特征影响；最终实现无需手动3D掩码的精确且一致的3D编辑。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1) 3DEditVerse数据集，最大的配对3D编辑基准；2) 3DEditFormer模型，具有双引导注意力块、多阶段特征提取和时间自适应门控机制；3) 无需辅助3D掩码即可实现精确编辑。相比之前的工作：比SDS方法更快；比多视图编辑方法更好地保持跨视图一致性；比端到端3D生成模型不依赖易出错的3D掩码；比VoxHammer对掩码精度不那么敏感，且在3D指标上平均提高13%。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '这篇论文通过提出大规模3DEditVerse数据集和3DEditFormer模型，解决了3D编辑中跨视角一致性和结构保持的挑战，实现了无需辅助3D掩码的精确、一致且结构保持的3D编辑，为实用且可扩展的3D编辑建立了新标准。'}


### 论文摘要

3D editing - the task of locally modifying the geometry or appearance of a 3D asset - has wide applications in immersive content creation, digital entertainment, and AR/VR. However, unlike 2D editing, it remains challenging due to the need for cross-view consistency, structural fidelity, and fine-grained controllability. Existing approaches are often slow, prone to geometric distortions, or dependent on manual and accurate 3D masks that are error-prone and impractical. To address these challenges, we advance both the data and model fronts. On the data side, we introduce 3DEditVerse, the largest paired 3D editing benchmark to date, comprising 116,309 high-quality training pairs and 1,500 curated test pairs. Built through complementary pipelines of pose-driven geometric edits and foundation model-guided appearance edits, 3DEditVerse ensures edit locality, multi-view consistency, and semantic alignment. On the model side, we propose 3DEditFormer, a 3D-structure-preserving conditional transformer. By enhancing image-to-3D generation with dual-guidance attention and time-adaptive gating, 3DEditFormer disentangles editable regions from preserved structure, enabling precise and consistent edits without requiring auxiliary 3D masks. Extensive experiments demonstrate that our framework outperforms state-of-the-art baselines both quantitatively and qualitatively, establishing a new standard for practical and scalable 3D editing. Dataset and code will be released. Project: https://www.lv-lab.org/3DEditFormer/

---

## 22. Training-Free Out-Of-Distribution Segmentation With Foundation Models

**论文链接:** [http://arxiv.org/abs/2510.02909v1](http://arxiv.org/abs/2510.02909v1)

**作者:** Laith Nayal, Hadi Salloum, Ahmad Taha, Yaroslav Kholodov, Alexander Gasnikov

**发布时间:** 2025-10-03

**备注:** 12 pages, 5 figures, 2 tables, ICOMP 2025

### GPT解析

### 总结

本研究探讨了基础模型在语义分割中检测分布外(OoD)区域的能力，提出了一种无需训练的简单方法，利用InternImage特征和K-Means聚类识别未知对象，并在基准测试上超越了监督和无监督基线。

### 背景

在自动驾驶等安全关键应用中，检测语义分割中的未知对象至关重要。大型视觉基础模型如DINOv2、InternImage和CLIP通过提供丰富的特征推动了视觉表征学习，但它们在检测分布外区域的能力尚未被充分探索。

### 目的

研究经过分割数据集微调的基础模型是否能够在没有任何异常监督的情况下，本质地区分分布内(ID)和分布外(OoD)区域。

### 方法

提出一种简单、无需训练的方法，利用InternImage主干网络的特征，在原始解码器logits上应用K-Means聚类和置信度阈值来识别OoD聚类。

### 主要发现

使用InternImage-L，在RoadAnomaly基准测试上达到50.02的平均精度，在ADE-OoD基准测试上达到48.77的平均精度，超越了多个监督和无监督基线。

### 结论

这些结果表明，对于需要最少假设或额外数据的通用OoD分割方法来说，这是一个有前景的方向。

### 翻译

在语义分割中检测未知对象对于自动驾驶等安全关键应用至关重要。包括DINOv2、InternImage和CLIP在内的大型视觉基础模型，通过提供能很好地跨不同任务泛化的丰富特征，推动了视觉表征学习的发展。虽然它们在闭集语义任务中的优势已经得到确立，但在语义分割中检测分布外(OoD)区域的能力尚未被充分探索。在本工作中，我们研究了经过分割数据集微调的基础模型是否能够在没有任何异常监督的情况下，本质地区分分布内(ID)和分布外(OoD)区域。我们提出了一种简单、无需训练的方法，利用InternImage主干网络的特征，并在原始解码器logits上应用K-Means聚类和置信度阈值来识别OoD聚类。我们的方法在使用InternImage-L的情况下，在RoadAnomaly基准测试上达到50.02的平均精度，在ADE-OoD基准测试上达到48.77的平均精度，超越了多个监督和无监督基线。这些结果表明，对于需要最少假设或额外数据的通用OoD分割方法来说，这是一个有前景的方向。


### 论文摘要

Detecting unknown objects in semantic segmentation is crucial for safety-critical applications such as autonomous driving. Large vision foundation models, including DINOv2, InternImage, and CLIP, have advanced visual representation learning by providing rich features that generalize well across diverse tasks. While their strength in closed-set semantic tasks is established, their capability to detect out-of-distribution (OoD) regions in semantic segmentation remains underexplored. In this work, we investigate whether foundation models fine-tuned on segmentation datasets can inherently distinguish in-distribution (ID) from OoD regions without any outlier supervision. We propose a simple, training-free approach that utilizes features from the InternImage backbone and applies K-Means clustering alongside confidence thresholding on raw decoder logits to identify OoD clusters. Our method achieves 50.02 Average Precision on the RoadAnomaly benchmark and 48.77 on the benchmark of ADE-OoD with InternImage-L, surpassing several supervised and unsupervised baselines. These results suggest a promising direction for generic OoD segmentation methods that require minimal assumptions or additional data.

---

## 23. Energy Efficiency in Cloud-Based Big Data Processing for Earth Observation: Gap Analysis and Future Directions

**论文链接:** [http://arxiv.org/abs/2510.02882v1](http://arxiv.org/abs/2510.02882v1)

**作者:** Adhitya Bhawiyuga, Serkan Girgin, Rolf A. de By, Raul Zurita-Milla

**发布时间:** 2025-10-03

### GPT解析

### 总结

这篇论文关注地球观测数据在云处理过程中的能源效率问题，指出了当前处理平台中的能源效率差距，并提出了改进的研究方向。

### 背景

地球观测数据量快速增长，云计算被用于处理大型数据集，但能源效率方面受到的关注较少。随着大数据处理中能源成本和碳足迹意识的提高，以及计算密集型基础模型的发展，这一问题变得尤为突出。

### 目的

识别基于云的地球观测大数据(EOBD)处理中能源效率实践的差距，并提出改进的研究方向。

### 方法

检查当前EOBD环境，关注需要基于云处理的要求，分析现有解决方案，研究其他大数据领域成功应用的能源效率策略。

### 主要发现

现有EOBD处理平台主要关注数据可访问性和计算可行性而非能源效率；关键差距包括：不足的能源监测机制、数据管理中缺乏能源意识、能源感知资源分配实施不足、任务调度缺乏能源效率标准。

### 结论

建议开发能源感知性能监测和基准测试框架，使用优化技术进行基础设施编排，采用能源高效的任务调度方法，以促进EOBD处理中的能源意识，减少能源消耗和环境影响，同时保持处理性能。

### 翻译

地球观测数据量正在快速增长。虽然云计算现在被用于处理大型地球观测数据集，但这种处理的能源效率方面受到的关注较少。考虑到大数据处理中能源成本和碳足迹意识的不断提高，特别是在计算密集型基础模型受到更多关注的情况下，这个问题尤为突出。在本文中，我们确定了基于云的地球观测大数据处理中能源效率实践的差距，并提出了几项改进的研究方向。


### 论文摘要

Earth observation (EO) data volumes are rapidly increasing. While cloud computing are now used for processing large EO datasets, the energy efficiency aspects of such a processing have received much less attention. This issue is notable given the increasing awareness of energy costs and carbon footprint in big data processing, particularly with increased attention on compute-intensive foundation models. In this paper we identify gaps in energy efficiency practices within cloud-based EO big data (EOBD) processing and propose several research directions for improvement. We first examine the current EOBD landscape, focus on the requirements that necessitate cloud-based processing and analyze existing cloud-based EOBD solutions. We then investigate energy efficiency strategies that have been successfully employed in well-studied big data domains. Through this analysis, we identify several critical gaps in existing EOBD processing platforms, which primarily focus on data accessibility and computational feasibility, instead of energy efficiency. These gaps include insufficient energy monitoring mechanisms, lack of energy awareness in data management, inadequate implementation of energy-aware resource allocation and lack of energy efficiency criteria on task scheduling. Based on these findings, we propose the development of energy-aware performance monitoring and benchmarking frameworks, the use of optimization techniques for infrastructure orchestration, and of energy-efficient task scheduling approaches for distributed cloud-based EOBD processing frameworks. These proposed approaches aim to foster more energy awareness in EOBD processing , potentially reducing power consumption and environmental impact while maintaining or minimally impacting processing performance.

---

## 24. From Tokens to Nodes: Semantic-Guided Motion Control for Dynamic 3D Gaussian Splatting

**论文链接:** [http://arxiv.org/abs/2510.02732v1](http://arxiv.org/abs/2510.02732v1)

**作者:** Jianing Chen, Zehao Li, Yujun Cai, Hao Jiang, Shuqin Gao, Honglong Zhao, Tianlu Mao, Yucheng Zhang

**发布时间:** 2025-10-03

### GPT解析

### 总结

本文提出了一种运动自适应框架，用于解决从单目视频中动态3D重建的挑战，通过使控制点密度与运动复杂度相匹配来提高重建质量和效率。

### 背景

从单目视频中进行动态3D重建仍然困难，原因是从有限视角推断3D运动存在歧义，以及对时变场景建模的计算需求大。

### 目的

开发一个运动自适应框架，使控制点密度与运动复杂度相匹配，解决现有方法中静态冗余和动态不足的问题。

### 方法

利用视觉基础模型的语义和运动先验，建立补丁-标记-节点对应关系，应用运动自适应压缩将控制点集中在动态区域同时抑制静态背景冗余，通过迭代体素化和运动倾向评分实现灵活的表示密度适应，并引入基于样条的轨迹参数化替代基于MLP的变形场。

### 主要发现

该方法直接解决了控制点分配与运动复杂度之间的根本不匹配问题，能够更有效地分配计算资源到动态区域。

### 结论

大量实验表明，与现有最先进的方法相比，该方法在重建质量和效率方面都有显著改进。

### 翻译

从单目视频进行动态3D重建仍然困难，这是由于从有限视角推断3D运动存在歧义以及对时变场景建模的计算需求。虽然最近的稀疏控制方法通过将数百万个高斯函数减少到数千个控制点来减轻计算负担，但它们存在一个关键局限：它们完全基于几何分配点，导致静态冗余和动态不足。我们提出了一种运动自适应框架，使控制密度与运动复杂度相匹配。利用视觉基础模型的语义和运动先验，我们建立了补丁-标记-节点对应关系，并应用运动自适应压缩，将控制点集中在动态区域，同时抑制静态背景的冗余。我们的方法通过迭代体素化和运动倾向评分实现了灵活的表示密度适应，直接解决了控制点分配与运动复杂度之间的根本不匹配问题。为了捕捉时间演化，我们引入了基于样条的轨迹参数化，由2D轨迹初始化，替代基于MLP的变形场，以实现更平滑的运动表示和更稳定的优化。大量实验证明了与现有最先进方法相比，在重建质量和效率方面的显著改进。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决动态3D重建中的两个核心问题：从有限视角推断3D运动的不确定性，以及建模时变场景的高计算需求。具体来说，现有稀疏控制方法仅基于几何分配控制点，导致静态区域冗余而动态区域表示不足。这个问题在虚拟现实、自主系统和内容创建等领域至关重要，因为这些应用需要从有限视角捕捉复杂物体运动和变形，同时保持实时渲染性能。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先识别出现有稀疏控制方法的局限性：控制点分配不均衡，浪费在静态区域而无法充分表示动态区域。然后提出利用语义和运动先验来指导控制点分配的思路，使控制密度与运动复杂度相匹配。作者借鉴了3D高斯泼溅技术、视觉基础模型用于语义提取、节点表示方法控制场景变形、样条参数化表示轨迹以及双四元数混合技术进行变形。这些现有技术被创新性地组合，解决了传统方法中的静态冗余和动态不足问题。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是利用语义和运动先验指导控制点分配，使控制点密度与运动复杂度相匹配，在动态区域放置更多控制点，在静态区域减少冗余。整体实现流程分为五个阶段：1) 基础阶段从单目视频中提取语义和运动先验；2) 运动自适应节点初始化，通过补丁到节点的生成和运动自适应压缩创建优化的节点集；3) 样条参数化节点轨迹，使用三次Hermite样条表示并从2D轨迹初始化；4) 高斯到节点变形，通过双四元数混合将节点运动传播到高斯点；5) 优化阶段，使用多视图约束进行联合优化。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点有三方面：1) 运动自适应节点初始化方法，利用视觉基础模型的语义和运动先验使控制密度与运动复杂度匹配；2) 基于样条的节点轨迹参数化，提供紧凑、平滑且可微的运动基础；3) 完整的优化框架，实现比现有方法更优的重建质量和效率。相比之前的工作，不同之处在于：传统方法(如SC-GS、4D-Scaffold)仅基于几何均匀性分配控制点，不考虑运动复杂性；之前的方法使用MLP或网格变形场，而本文使用样条参数化提供更平滑的表示；本文利用视觉基础模型的语义先验指导控制点分配，而之前方法主要依赖几何信息。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '本文提出了一种基于语义引导的运动自适应框架，通过视觉基础模型的先验知识优化控制点分配，并使用样条参数化实现更稳定、高效的动态3D高斯泼溅重建。'}


### 论文摘要

Dynamic 3D reconstruction from monocular videos remains difficult due to the ambiguity inferring 3D motion from limited views and computational demands of modeling temporally varying scenes. While recent sparse control methods alleviate computation by reducing millions of Gaussians to thousands of control points, they suffer from a critical limitation: they allocate points purely by geometry, leading to static redundancy and dynamic insufficiency. We propose a motion-adaptive framework that aligns control density with motion complexity. Leveraging semantic and motion priors from vision foundation models, we establish patch-token-node correspondences and apply motion-adaptive compression to concentrate control points in dynamic regions while suppressing redundancy in static backgrounds. Our approach achieves flexible representational density adaptation through iterative voxelization and motion tendency scoring, directly addressing the fundamental mismatch between control point allocation and motion complexity. To capture temporal evolution, we introduce spline-based trajectory parameterization initialized by 2D tracklets, replacing MLP-based deformation fields to achieve smoother motion representation and more stable optimization. Extensive experiments demonstrate significant improvements in reconstruction quality and efficiency over existing state-of-the-art methods.

---

## 25. AgenticRAG: Tool-Augmented Foundation Models for Zero-Shot Explainable Recommender Systems

**论文链接:** [http://arxiv.org/abs/2510.02668v1](http://arxiv.org/abs/2510.02668v1)

**作者:** Bo Ma, Hang Li, ZeHua Hu, XiaoFan Gui, LuYao Liu, Simon Liu

**发布时间:** 2025-10-03

### GPT解析

### 总结

AgenticRAG是一个结合工具增强基础模型和检索增强生成的新框架，用于零样本可解释推荐。它通过整合外部工具调用、知识检索和思维链推理，实现无需任务特定训练的透明决策，实验证明其在多个数据集上优于现有方法，同时保持可解释性和计算效率。

### 背景

基础模型已经彻底改变了人工智能领域，但在推荐系统中的应用受到推理不透明性和知识限制的约束。

### 目的

介绍AgenticRAG框架，结合工具增强的基础模型和检索增强生成，实现零样本可解释推荐。

### 方法

整合外部工具调用、知识检索和思维链推理，创建能够进行透明决策的自主推荐代理，无需任务特定训练。

### 主要发现

在三个真实世界数据集上的实验结果表明，AgenticRAG实现了持续改进：Amazon Electronics上NDCG@10提高0.4%，MovieLens-1M上提高0.8%，Yelp数据集上提高1.6%。框架展现出卓越的可解释性，同时保持与传统方法相当的计算效率。

### 结论

AgenticRAG框架解决了基础模型在推荐系统中应用的两个主要限制：推理不透明性和知识约束，为推荐系统提供了新的有效解决方案。

### 翻译

基础模型已经彻底改变了人工智能，然而它们在推荐系统中的应用仍然受到推理不透明性和知识限制的制约。本文介绍了AgenticRAG，一个结合工具增强基础模型和检索增强生成的新框架，用于零样本可解释推荐。我们的方法整合了外部工具调用、知识检索和思维链推理，创建了无需任务特定训练即可进行透明决策的自主推荐代理。在三个真实世界数据集上的实验结果表明，AgenticRAG与最先进的基线相比实现了持续改进，在Amazon Electronics上NDCG@10提高了0.4%，在MovieLens-1M上提高了0.8%，在Yelp数据集上提高了1.6%。该框架展现出卓越的可解释性，同时保持与传统方法相当的计算效率。


### 论文摘要

Foundation models have revolutionized artificial intelligence, yet their application in recommender systems remains limited by reasoning opacity and knowledge constraints. This paper introduces AgenticRAG, a novel framework that combines tool-augmented foundation models with retrieval-augmented generation for zero-shot explainable recommendations. Our approach integrates external tool invocation, knowledge retrieval, and chain-of-thought reasoning to create autonomous recommendation agents capable of transparent decision-making without task-specific training. Experimental results on three real-world datasets demonstrate that AgenticRAG achieves consistent improvements over state-of-the-art baselines, with NDCG@10 improvements of 0.4\% on Amazon Electronics, 0.8\% on MovieLens-1M, and 1.6\% on Yelp datasets. The framework exhibits superior explainability while maintaining computational efficiency comparable to traditional methods.

---

## 26. TabImpute: Accurate and Fast Zero-Shot Missing-Data Imputation with a Pre-Trained Transformer

**论文链接:** [http://arxiv.org/abs/2510.02625v1](http://arxiv.org/abs/2510.02625v1)

**作者:** Jacob Feitelberg, Dwaipayan Saha, Kyuseong Choi, Zaid Ahmad, Anish Agarwal, Raaz Dwivedi

**发布时间:** 2025-10-03

### GPT解析

### 总结

TabImpute是一种基于预训练transformer的零样本插补方法，解决了表格数据中缺失数据处理的挑战，它不需要在推理时进行拟合或超参数调整，且速度快、准确性高。

### 背景

缺失数据是表格设置中的一个普遍问题。现有的解决方案从简单平均到复杂的生成对抗网络不等。然而，由于在实际领域中性能差异很大，且超参数调整耗时，目前没有默认的插补方法。

### 目的

基于TabPFN（一种用于监督学习的表格基础模型）提出TabImpute，开发一个预训练的transformer模型，提供准确且快速的零样本插补，无需在推理时进行拟合或超参数调整。

### 方法

提出TabImpute，一种预训练的transformer模型；引入(i) 表格设置中的逐要素特征化，比之前的TabPFN插补方法快100倍；(ii) 合成训练数据生成管道，融入真实的缺失模式，提高测试时性能；(iii) MissBench，一个包含42个OpenML数据集和13种缺失模式的插补方法综合评估基准。

### 主要发现

TabImpute在医学、金融和工程等多个领域展现了强大的性能，与11种成熟的插补方法相比表现良好。

### 结论

TabImpute提供了一种准确、快速且无需超参数调整的缺失数据插补解决方案，通过MissBench基准测试验证了其有效性。

### 翻译

缺失数据是表格设置中的一个普遍问题。现有的解决方案从简单平均到复杂的生成对抗网络不等。然而，由于在实际领域中性能差异很大，且超参数调整耗时，目前没有默认的插补方法。基于TabPFN（一种用于监督学习的表格基础模型），我们提出了TabImpute，一种预训练的transformer模型，能够提供准确且快速的零样本插补，无需在推理时进行拟合或超参数调整。为了训练和评估TabImpute，我们引入了(i) 表格设置中的逐要素特征化，比之前的TabPFN插补方法快100倍，(ii) 合成训练数据生成管道，融入真实的缺失模式，提高测试时性能，以及(iii) MissBench，一个包含42个OpenML数据集和13种缺失模式的插补方法综合评估基准。MissBench涵盖医学、金融和工程等领域，展示了TabImpute与11种成熟插补方法相比的强大性能。


### 论文摘要

Missing data is a pervasive problem in tabular settings. Existing solutions range from simple averaging to complex generative adversarial networks. However, due to huge variance in performance across real-world domains and time-consuming hyperparameter tuning, no default imputation method exists. Building on TabPFN, a recent tabular foundation model for supervised learning, we propose TabImpute, a pre-trained transformer that delivers accurate and fast zero-shot imputations requiring no fitting or hyperparameter tuning at inference-time. To train and evaluate TabImpute, we introduce (i) an entry-wise featurization for tabular settings, which enables a $100\times$ speedup over the previous TabPFN imputation method, (ii) a synthetic training data generation pipeline incorporating realistic missingness patterns, which boosts test-time performance, and (iii) MissBench, a comprehensive benchmark for evaluation of imputation methods with $42$ OpenML datasets and $13$ missingness patterns. MissBench spans domains such as medicine, finance, and engineering, showcasing TabImpute's robust performance compared to $11$ established imputation methods.

---

## 27. Mitigating Modal Imbalance in Multimodal Reasoning

**论文链接:** [http://arxiv.org/abs/2510.02608v1](http://arxiv.org/abs/2510.02608v1)

**作者:** Chen Henry Wu, Neil Kale, Aditi Raghunathan

**发布时间:** 2025-10-02

**备注:** 10 pages, 10 figures, CoLM 2025

### GPT解析

### 总结

该研究探讨了基础模型在处理跨模态冲突时的表现，发现模型在单模态上下文中能较好识别冲突，但在跨模态上下文中表现极差，原因是存在跨模态注意力不平衡问题。通过改进训练方法，可以显著提高模型在跨模态任务中的表现。

### 背景

基础模型在真实世界任务（如计算机使用代理）中需要整合多种模态，特别是在模态间相互作用形成跨模态上下文时。

### 目的

研究基础模型在跨模态冲突场景下的表现，探究模型是优先考虑某一模态还是联合推理解决冲突。

### 方法

通过实验研究基础模型在跨模态冲突（不同模态间存在矛盾证据）场景下的表现，分析注意力机制，并提出改进的训练方法。

### 主要发现

基础模型在单模态上下文中能90%时间识别冲突，但在跨模态上下文中识别率降至3%；跨语言上下文中也有类似现象；失败原因是跨模态注意力不平衡，模型过度优先考虑某些模态；仅扩大数据集无法解决问题，但通过在每个训练实例中明确组合多种模态可以显著减少注意力不平衡并提高下游任务性能。

### 结论

系统地解决跨模态上下文对构建可靠的基础模型至关重要。

### 翻译

部署在计算机使用代理等真实世界任务中的基础模型必须整合多种模态。基础模型在执行联合推理（同时推理多种模态）方面的表现如何，特别是在模态相互作用并形成跨模态上下文时？为更好地理解这一问题，我们研究了基础模型在跨模态冲突场景下的表现：不同模态呈现矛盾证据的情况。这使我们能够检验基础模型是优先考虑某一模态还是联合推理来解决冲突。我们的实验显示，基础模型在单模态上下文中能90%时间识别冲突，但当证据分散在不同模态时，这一比例低至3%——在由多种语言组成的跨语言上下文中也有类似观察。我们将这一失败归因于跨模态注意力不平衡，表明基础模型在注意力分数上表现出极端不对称性，过度优先考虑某些模态。我们证明，仅盲目扩大多模态或多语言数据集无法解决跨模态注意力不平衡问题，因为它们缺乏明确要求跨模态推理的训练样本。我们证明，即使在每个训练实例中明确组合多种模态的简单可扩展方法也能显著减少注意力不平衡。减少注意力不平衡直接转化为多个视觉语言基准测试的下游性能提升。我们的发现强调了系统解决跨模态上下文对构建可靠基础模型的重要性。


### 论文摘要

Foundation models (FMs) deployed in real-world tasks such as computer-use agents must integrate diverse modalities. How good are FMs at performing joint reasoning, simultaneously reasoning over multiple modalities, especially when the modalities interact and relate to each other to form cross-modal context? To better understand this problem, we study FMs on cross-modal conflicts: scenarios where conflicting evidence is presented across modalities. This allows us to examine whether FMs prioritize one modality over another or reason jointly to reconcile the conflict. Our experiments reveal that FMs can recognize conflicts in unimodal contexts, composed of a single modality, 90% of the time, but the ratio falls as low as 3% when evidence is split across modalities -- similar observations hold in cross-lingual contexts, composed of multiple languages. We trace this failure to cross-modal attention imbalance, showing that FMs exhibit extreme asymmetry in attention scores, disproportionately prioritizing certain modalities. We show that cross-modal attention imbalance does not go away by simply scaling up multimodal or multilingual datasets blindly, since they lack training examples that explicitly require cross-modal reasoning. We demonstrate that even a simple and scalable method of explicitly combining multiple modalities within each training instance significantly reduces attention imbalance. Reduced attention imbalance directly translates to improved downstream performance on several vision-language benchmarks. Our findings underscore the importance of systematically addressing cross-modal contexts to build reliable foundation models.

---

## 28. FLOWR.root: A flow matching based foundation model for joint multi-purpose structure-aware 3D ligand generation and affinity prediction

**论文链接:** [http://arxiv.org/abs/2510.02578v1](http://arxiv.org/abs/2510.02578v1)

**作者:** Julian Cremer, Tuan Le, Mohammad M. Ghahremanpour, Emilia Sługocka, Filipe Menezes, Djork-Arné Clevert

**发布时间:** 2025-10-02

### GPT解析

### 总结

Flowr.root是一个等变流匹配模型，用于口袋感知的3D配体生成，具有结合亲和力预测和置信度估计功能。它支持从头生成、药效团条件采样、片段修饰和多端点亲和力预测。该模型结合大规模配体库和混合保真度的蛋白质-配体复合物进行训练，并在精选共晶数据集上精炼。

### 背景

基于结构的药物设计需要能够生成高质量配体并预测其结合亲和力的工具，以加速从命中识别到先导优化的过程。

### 目的

开发一个能够进行3D配体生成并预测结合亲和力的模型，为基于结构的药物设计提供全面基础。

### 方法

使用等变流匹配技术构建Flowr.root模型，结合大规模配体库和混合保真度的蛋白质-配体复合物进行训练，并在精选共晶数据集上精炼。使用参数高效微调进行项目特定适应。

### 主要发现

Flowr.root在无条件3D分子生成和口袋条件配体设计中达到最先进性能，产生几何上真实的低应变结构。其集成亲和力预测模块在SPINDR测试集上表现出卓越准确性，在Schrodinger FEP+/OpenFE基准测试中优于最近模型。案例研究显示预测与量子力学计算结果具有强相关性。

### 结论

Flowr.root通过整合结构感知生成、亲和力估计和属性引导采样，为基于结构的药物设计提供了全面基础，涵盖从命中识别到先导优化的整个过程。作为基础模型，它需要在项目特定数据集上进行微调以应对未见过的结构-活性景观。

### 翻译

我们提出了Flowr.root，一个用于口袋感知3D配体生成的等变流匹配模型，具有结合亲和力预测和置信度估计功能。该模型支持从头生成、药效团条件采样、片段修饰和多端点亲和力预测。训练结合大规模配体库和混合保真度的蛋白质-配体复合物，随后在精选的共晶数据集上进行精炼，并通过参数高效微调进行项目特定适应。Flowr.root在无条件3D分子生成和口袋条件配体设计中达到最先进性能，产生几何上真实的低应变结构。集成的亲和力预测模块在SPINDR测试集上表现出卓越的准确性，并在Schrodinger FEP+/OpenFE基准测试中优于最近的模型，具有显著的速度优势。作为基础模型，Flowr.root需要在项目特定数据集上进行微调以应对未见过的结构-活性景观，与实验数据产生强相关性。联合生成和亲和力预测通过重要性采样实现推理时扩展，引导分子设计朝向更高亲和力的化合物。案例研究验证了这一点：针对CLK3的选择性CK2alpha配体生成显示预测与量子力学结合能之间的显著相关性，而ERalpha和TYK2骨架修饰与量子力学计算显示出强一致性。通过整合结构感知生成、亲和力估计和属性引导采样，Flowr.root为基于结构的药物设计提供了全面基础，涵盖从命中识别到先导优化的整个过程。


### 论文摘要

We present Flowr.root, an equivariant flow-matching model for pocket-aware 3D ligand generation with joint binding affinity prediction and confidence estimation. The model supports de novo generation, pharmacophore-conditional sampling, fragment elaboration, and multi-endpoint affinity prediction (pIC50, pKi, pKd, pEC50). Training combines large-scale ligand libraries with mixed-fidelity protein-ligand complexes, followed by refinement on curated co-crystal datasets and parameter-efficient finetuning for project-specific adaptation. Flowr.root achieves state-of-the-art performance in unconditional 3D molecule generation and pocket-conditional ligand design, producing geometrically realistic, low-strain structures. The integrated affinity prediction module demonstrates superior accuracy on the SPINDR test set and outperforms recent models on the Schrodinger FEP+/OpenFE benchmark with substantial speed advantages. As a foundation model, Flowr.root requires finetuning on project-specific datasets to account for unseen structure-activity landscapes, yielding strong correlation with experimental data. Joint generation and affinity prediction enable inference-time scaling through importance sampling, steering molecular design toward higher-affinity compounds. Case studies validate this: selective CK2alpha ligand generation against CLK3 shows significant correlation between predicted and quantum-mechanical binding energies, while ERalpha and TYK2 scaffold elaboration demonstrates strong agreement with QM calculations. By integrating structure-aware generation, affinity estimation, and property-guided sampling, Flowr.root provides a comprehensive foundation for structure-based drug design spanning hit identification through lead optimization.

---

## 29. Geospatial Machine Learning Libraries

**论文链接:** [http://arxiv.org/abs/2510.02572v1](http://arxiv.org/abs/2510.02572v1)

**作者:** Adam J. Stewart, Caleb Robinson, Arindam Banerjee

**发布时间:** 2025-10-02

**备注:** Book chapter

### GPT解析

### 总结

本文提供了地理空间机器学习(GeoML)库的全面概述，分析了它们的演变、核心功能和当前生态系统，介绍了流行库如TorchGeo、eo-learn和Raster Vision的架构和数据类型支持，讨论了数据预处理等方法论，并通过作物类型映射案例研究展示了实际应用，同时提出了最佳实践和未来方向。

### 背景

机器学习的进步得益于特定领域软件库的出现，但地理空间机器学习领域面临地球观测数据可用性超过处理其独特挑战的领域库发展的问题，这些挑战包括变化的空间分辨率、光谱特性、时间频率、数据覆盖范围、坐标系统和文件格式。

### 目的

提供GeoML库的全面概述，分析它们的演变、核心功能和当前生态系统，介绍流行的GeoML库及其架构、数据类型支持和ML框架集成，讨论数据预处理等方法论，通过案例研究展示实际应用，强调最佳实践，并讨论开放挑战和未来方向，特别是基础模型和开源地理空间软件治理。

### 方法

分析GeoML库的演变、核心功能和生态系统；介绍并详细说明流行的GeoML库的架构、支持的数据类型和与ML框架的集成；讨论数据预处理、时空连接、基准测试和预训练模型使用的方法；通过作物类型映射的案例研究展示实际应用。

### 主要发现

存在多种流行的GeoML库，如TorchGeo、eo-learn和Raster Vision，它们具有不同的架构和数据类型支持；数据预处理、时空连接、基准测试和预训练模型使用是GeoML中的常见方法；作物类型映射是GeoML的一个实际应用案例；软件设计、许可和测试有最佳实践可遵循；基础模型的兴起和开源地理空间软件治理是未来的重要方向。

### 结论

本文旨在指导从业者、开发者和研究人员导航和贡献于快速发展的GeoML领域。

### 翻译

最近的机器学习进展得益于特定领域软件库的出现，使工作流程更加高效且提高了可重复性。对于地理空间机器学习(GeoML)，地球观测数据的可用性已经超过了处理其独特挑战的领域库的发展，如变化的空间分辨率、光谱特性、时间频率、数据覆盖范围、坐标系统和文件格式。本章提供了GeoML库的全面概述，分析了它们的演变、核心功能和当前生态系统。它还介绍了流行的GeoML库，如TorchGeo、eo-learn和Raster Vision，详细说明了它们的架构、支持的数据类型以及与ML框架的集成。此外，它讨论了数据预处理、时空连接、基准测试和使用预训练模型的常见方法。通过作物类型映射的案例研究，它展示了这些工具的实际应用。软件设计、许可和测试的最佳实践被强调，同时还有开放挑战和未来方向，特别是基础模型的兴起和开源地理空间软件治理的需求。我们的目标是指导从业者、开发者和研究人员导航并贡献于快速发展的GeoML领域。


### 论文摘要

Recent advances in machine learning have been supported by the emergence of domain-specific software libraries, enabling streamlined workflows and increased reproducibility. For geospatial machine learning (GeoML), the availability of Earth observation data has outpaced the development of domain libraries to handle its unique challenges, such as varying spatial resolutions, spectral properties, temporal cadence, data coverage, coordinate systems, and file formats. This chapter presents a comprehensive overview of GeoML libraries, analyzing their evolution, core functionalities, and the current ecosystem. It also introduces popular GeoML libraries such as TorchGeo, eo-learn, and Raster Vision, detailing their architecture, supported data types, and integration with ML frameworks. Additionally, it discusses common methodologies for data preprocessing, spatial--temporal joins, benchmarking, and the use of pretrained models. Through a case study in crop type mapping, it demonstrates practical applications of these tools. Best practices in software design, licensing, and testing are highlighted, along with open challenges and future directions, particularly the rise of foundation models and the need for governance in open-source geospatial software. Our aim is to guide practitioners, developers, and researchers in navigating and contributing to the rapidly evolving GeoML landscape.

---

## 30. Uncertainty-Guided Model Selection for Tabular Foundation Models in Biomolecule Efficacy Prediction

**论文链接:** [http://arxiv.org/abs/2510.02476v1](http://arxiv.org/abs/2510.02476v1)

**作者:** Jie Li, Andrew McCarthy, Zhizhuo Zhang, Stephen Young

**发布时间:** 2025-10-02

**备注:** NeurIPS 2025 workshop: 2nd Workshop on Multi-modal Foundation Models  and Large Language Models for Life Sciences

### GPT解析

### 总结

该研究探讨了基于不确定性的模型选择策略在生物分子功效预测中的应用，发现使用简单序列特征的TabPFN模型可以超越专业预测器，且模型的不确定性可作为无需标签的启发式方法优化预测性能。

### 背景

上下文学习器如TabPFN在生物分子功效预测方面前景广阔，其中已建立的分子特征集和相关实验结果可作为有力的上下文示例。然而，这些学习器的性能对提供的上下文高度敏感，导致在不同数据子集上训练的模型的后集成成为一种可行方法。

### 目的

研究在没有真实标签的情况下，如何选择最佳模型进行集成，特别关注基于不确定性的模型选择策略。

### 方法

在siRNA敲低功效任务上测试TabPFN模型，使用简单的基于序列的特征，评估模型预测的IQR（不确定性的度量）与真实预测误差的关系，并通过选择和平均具有最低平均IQR的模型进行集成。

### 主要发现

1) 使用简单序列特征的TabPFN模型超越了专业的最先进预测器；2) 模型预测的IQR与真实预测误差呈负相关；3) 通过选择最低平均IQR的模型集成，比简单集成或使用在所有可用数据上训练的单个模型性能更好。

### 结论

模型不确定性是优化生物分子功效预测的一个强大、无需标签的启发式方法，为模型选择提供了新思路。

### 翻译

上下文学习器如TabPFN在生物分子功效预测方面很有前景，其中已建立的分子特征集和相关实验结果可以作为有力的上下文示例。然而，它们的性能对提供的上下文高度敏感，使得在不同数据子集上训练的模型的后集成成为一种可行方法。一个开放问题是，在没有真实标签的情况下如何为集成选择最佳模型。在本研究中，我们研究了一种基于不确定性的模型选择策略。我们在siRNA敲低功效任务上证明，使用简单序列特征的TabPFN模型可以超越专业的最先进预测器。我们还表明，模型预测的四分位距，即其不确定性的度量，与真实预测误差呈负相关。通过选择并平均具有最低平均四分位距的模型集成，我们实现了比简单集成或使用在所有可用数据上训练的单个模型更好的性能。这一发现突显了模型不确定性作为优化生物分子功效预测的强大、无需标签的启发式方法的重要性。


### 论文摘要

In-context learners like TabPFN are promising for biomolecule efficacy prediction, where established molecular feature sets and relevant experimental results can serve as powerful contextual examples. However, their performance is highly sensitive to the provided context, making strategies like post-hoc ensembling of models trained on different data subsets a viable approach. An open question is how to select the best models for the ensemble without access to ground truth labels. In this study, we investigate an uncertainty-guided strategy for model selection. We demonstrate on an siRNA knockdown efficacy task that a TabPFN model using simple sequence-based features can surpass specialized state-of-the-art predictors. We also show that the model's predicted inter-quantile range (IQR), a measure of its uncertainty, has a negative correlation with true prediction error. By selecting and averaging an ensemble of models with the lowest mean IQR, we achieve superior performance compared to naive ensembling or using a single model trained on all available data. This finding highlights model uncertainty as a powerful, label-free heuristic for optimizing biomolecule efficacy predictions.

---

## 31. How to Train Your Advisor: Steering Black-Box LLMs with Advisor Models

**论文链接:** [http://arxiv.org/abs/2510.02453v1](http://arxiv.org/abs/2510.02453v1)

**作者:** Parth Asawa, Alan Zhu, Matei Zaharia, Alexandros G. Dimakis, Joseph E. Gonzalez

**发布时间:** 2025-10-02

### GPT解析

### 总结

该研究提出了顾问模型（Advisor Models），一种通过强化学习训练的轻量级参数化策略，用于动态优化黑盒基础模型的行为，克服了静态提示优化的局限性。

### 背景

基础模型越来越多地作为黑盒服务部署，模型权重无法修改，定制仅限于提示工程。静态提示优化虽然显示出前景，但产生的是单一固定提示，无法适应不同的输入、用户或环境。

### 目的

开发一种能够动态适应不同输入、用户和环境的方法，用于优化黑盒基础模型的行为。

### 方法

引入顾问模型，这是一种轻量化的参数化策略，通过强化学习训练，能够针对黑盒模型反应性地发出自然语言引导指令。顾问模型作为第二模型位于输入和目标模型之间，基于环境的奖励信号逐个实例地塑造行为。

### 主要发现

顾问模型在多个涉及推理和个性化的领域超越了静态提示优化器，能够发现环境动态并改进下游任务性能。顾问模型可以跨黑盒模型迁移，显示出良好的泛化能力，同时能够实现专业化并保持对分布外输入的鲁棒性。

### 结论

通过顾问模型对黑盒模型进行动态优化是实现个性化和环境适应性AI的有前景方向，能够为黑盒系统提供可学习的接口，顾问模型作为参数化、环境特定的记忆发挥作用。

### 翻译

基础模型越来越多地作为黑盒服务部署，其中模型权重无法修改，定制仅限于提示工程。虽然静态提示优化显示出前景，但它产生的单一固定提示无法适应不同的输入、用户或环境。我们引入了顾问模型，这是一种通过强化学习训练的轻量级参数化策略，能够针对黑盒模型反应性地发出自然语言引导指令。顾问是一个小型第二模型，位于输入和模型之间，使用来自环境的奖励信号逐个实例地塑造行为。在多个涉及推理和个性化的领域，我们表明顾问模型优于静态提示优化器，能够发现环境动态并改进下游任务性能。我们还通过跨黑盒模型迁移顾问模型证明了其泛化能力，以及框架在保持对分布外输入鲁棒性的同时实现专业化的能力。更广泛地看，顾问模型为黑盒系统提供了可学习的接口，其中顾问作为参数化、环境特定的记忆发挥作用。我们认为，通过顾问模型对黑盒模型进行动态优化是实现具有前沿能力的个性化和环境适应性AI的一个有前景的方向。


### 论文摘要

Foundation models are increasingly deployed as black-box services, where model weights cannot be modified and customization is limited to prompting. While static prompt optimization has shown promise, it produces a single fixed prompt that fails to adapt to different inputs, users, or environments. We introduce Advisor Models, lightweight parametric policies trained with reinforcement learning to reactively issue natural language steering instructions in-context to black-box models. The advisor is a second small model that sits between the input and the model, shaping behavior on a per-instance basis using reward signals from the environment. Across multiple domains involving reasoning and personalization, we show that Advisor Models outperform static prompt optimizers, discovering environment dynamics and improving downstream task performance. We also demonstrate the generalizability of advisors by transferring them across black-box models, as well as the framework's ability to achieve specialization while retaining robustness to out-of-distribution inputs. Viewed more broadly, Advisor Models provide a learnable interface to black-box systems where the advisor acts as a parametric, environment-specific memory. We argue that dynamic optimization of black-box models via Advisor Models is a promising direction for enabling personalization and environment-adaptable AI with frontier-level capabilities.

---

## 32. KAIROS: Unified Training for Universal Non-Autoregressive Time Series Forecasting

**论文链接:** [http://arxiv.org/abs/2510.02084v2](http://arxiv.org/abs/2510.02084v2)

**作者:** Kuiye Ding, Fanda Fan, Zheya Wang, Hongxiao Li, Yifan Wang, Lei Wang, Chunjie Luo, Jianfeng Zhan

**发布时间:** 2025-10-02

### GPT解析

### 总结

本文介绍了KAIROS，一个非自回归时间序列预测框架，直接对段级多峰分布进行建模，避免了误差累积并实现及时推理，在六个基准测试上表现出强大的零样本泛化能力，以较低推理成本提供与最先进模型相当的预测性能。

### 背景

在万维网中，可靠的时间序列预测提供前瞻性信号，驱动资源规划、缓存放置和异常响应，使平台能够随着用户行为和内容分布的演变而高效运行。与其他领域相比，Web应用的时间序列预测需要更快的响应速度以支持实时决策。

### 目的

开发一个能够快速响应的时间序列预测框架，以满足Web应用中实时决策的需求，同时避免自回归方法中的误差累积问题。

### 方法

提出KAIROS，一个非自回归时间序列预测框架，直接对段级多峰分布进行建模。与自回归方法不同，KAIROS避免了误差累积，实现了及时推理，同时改进了现有的非自回归模型，防止它们退化为过度平滑的预测。

### 主要发现

在大型语料库上训练后，KAIROS在六个广泛使用的基准测试上表现出强大的零样本泛化能力，以远低于最先进基础模型的推理成本提供了可比较的预测性能。

### 结论

KAIROS强调了非自回归设计作为时间序列基础模型可扩展范式的重要性。

### 翻译

在万维网中，可靠的时间序列预测提供了前瞻性信号，驱动资源规划、缓存放置和异常响应，使平台能够随着用户行为和内容分布的演变而高效运行。与其他领域相比，Web应用的时间序列预测需要更快的响应速度以支持实时决策。我们提出了KAIROS，一个非自回归时间序列预测框架，直接对段级多峰分布进行建模。与自回归方法不同，KAIROS避免了误差累积，实现了及时推理，同时改进了现有的非自回归模型，防止它们退化为过度平滑的预测。在大型语料库上训练后，KAIROS在六个广泛使用的基准测试上表现出强大的零样本泛化能力，以远低于最先进基础模型的推理成本提供了可比较的预测性能。除了实证结果外，KAIROS还强调了非自回归设计作为时间序列基础模型可扩展范式的重要性。


### 论文摘要

In the World Wide Web, reliable time series forecasts provide the forward-looking signals that drive resource planning, cache placement, and anomaly response, enabling platforms to operate efficiently as user behavior and content distributions evolve. Compared with other domains, time series forecasting for Web applications requires much faster responsiveness to support real-time decision making. We present KAIROS, a non-autoregressive time series forecasting framework that directly models segment-level multi-peak distributions. Unlike autoregressive approaches, KAIROS avoids error accumulation and achieves just-in-time inference, while improving over existing non-autoregressive models that collapse to over-smoothed predictions. Trained on the large-scale corpus, KAIROS demonstrates strong zero-shot generalization on six widely used benchmarks, delivering forecasting performance comparable to state-of-the-art foundation models with similar scale, at a fraction of their inference cost. Beyond empirical results, KAIROS highlights the importance of non-autoregressive design as a scalable paradigm for foundation models in time series.

---

## 33. InsideOut: An EfficientNetV2-S Based Deep Learning Framework for Robust Multi-Class Facial Emotion Recognition

**论文链接:** [http://arxiv.org/abs/2510.03066v1](http://arxiv.org/abs/2510.03066v1)

**作者:** Ahsan Farabi, Israt Khandaker, Ibrahim Khalil Shanto, Md Abdul Ahad Minhaz, Tanisha Zaman

**发布时间:** 2025-10-03

### GPT解析

### 总结

本文提出了InsideOut框架，一个基于EfficientNetV2-S的可重现面部情感识别系统，通过迁移学习、强数据增强和不平衡感知优化，在FER2013数据集上实现了62.8%的准确率和0.590的宏平均F1分数。

### 背景

面部情感识别是情感计算中的关键任务，应用于人机交互、电子学习、医疗和安全系统。尽管深度学习有所进步，FER仍面临遮挡、光照和姿势变化、类内差异细微以及数据不平衡等挑战，这些因素阻碍了对少数情感的识别。

### 目的

开发一个可重现的FER框架，解决数据不平衡问题，并在FER2013数据集上实现有竞争力的性能。

### 方法

构建基于EfficientNetV2-S的框架，采用迁移学习、强数据增强和不平衡感知优化技术。标准化FER2013图像，应用分层分割和增强，使用类别加权损失微调轻量级分类头以处理偏斜分布。

### 主要发现

InsideOut在FER2013上达到62.8%的准确率和0.590的宏平均F1分数，与传统CNN基线相比具有竞争力。

### 结论

高效架构与定制的不平衡处理相结合可以提供实用、透明和可重现的FER解决方案。

### 翻译

面部情感识别是情感计算中的关键任务，使人机交互、电子学习、医疗和安全系统的应用成为可能。尽管深度学习有所进步，FER仍然面临挑战，包括遮挡、光照和姿势变化、细微的类内差异以及数据不平衡，这些因素阻碍了对少数情感的识别。我们提出了InsideOut，这是一个基于EfficientNetV2-S的可重现FER框架，采用迁移学习、强数据增强和不平衡感知优化构建。该方法标准化FER2013图像，应用分层分割和增强，并使用类别加权损失微调轻量级分类头以解决偏斜分布问题。InsideOut在FER2013上实现了62.8%的准确率和0.590的宏平均F1分数，显示出与传统CNN基线相比具有竞争力的结果。新颖之处在于证明了高效架构与定制的不平衡处理相结合可以提供实用、透明和可重现的FER解决方案。


### 论文摘要

Facial Emotion Recognition (FER) is a key task in affective computing, enabling applications in human-computer interaction, e-learning, healthcare, and safety systems. Despite advances in deep learning, FER remains challenging due to occlusions, illumination and pose variations, subtle intra-class differences, and dataset imbalance that hinders recognition of minority emotions. We present InsideOut, a reproducible FER framework built on EfficientNetV2-S with transfer learning, strong data augmentation, and imbalance-aware optimization. The approach standardizes FER2013 images, applies stratified splitting and augmentation, and fine-tunes a lightweight classification head with class-weighted loss to address skewed distributions. InsideOut achieves 62.8% accuracy with a macro averaged F1 of 0.590 on FER2013, showing competitive results compared to conventional CNN baselines. The novelty lies in demonstrating that efficient architectures, combined with tailored imbalance handling, can provide practical, transparent, and reproducible FER solutions.

---

## 34. Modeling Quantum Geometry for Fractional Chern Insulators with unsupervised learning

**论文链接:** [http://arxiv.org/abs/2510.03018v1](http://arxiv.org/abs/2510.03018v1)

**作者:** Ang-Kun Wu, Louis Primeau, Jingtao Zhang, Kai Sun, Yang Zhang, Shi-Zeng Lin

**发布时间:** 2025-10-03

**备注:** 17 pages, 11 figures

### GPT解析

### 总结

研究引入了一种无监督机器学习框架，通过单粒子形状因子的分布直接建模相互作用的哈密顿量，成功区分并生成了具有不同拓扑特性的分数量子霍尔绝缘体态

### 背景

莫尔材料中的分数量子霍尔绝缘体为探索强关联拓扑相提供了独特平台，但现实莫尔模型缺乏量子度量和贝里曲率的直接可调性，限制了理论和数值探索

### 目的

开发一种机器学习方法来建模相互作用哈密顿量，克服传统方法在理想量子几何方面的局限性

### 方法

使用变分自编码器(VAE)进行无监督学习，分析单粒子形状因子的分布，并结合主成分分析(PCA)揭示形状因子中的主导模式

### 主要发现

无监督学习能区分FCI和非FCI态；能生成训练集中不存在的新形状因子；能生成和插值具有陈数|C|=1的拓扑平带形状因子；能发现如电荷密度波等未观测到的多体态；形状因子中的主导模式可分解为具有近似量化陈数的分量

### 结论

机器学习能够泛化和建模拓扑量子系统，为设计具有定制量子几何和平带材料中多体相的形状因子开辟了新途径

### 翻译

莫尔材料中的分数量子霍尔绝缘体(FCIs)为探索超越理想量子几何范式的强关联拓扑相提供了一个独特平台。虽然FCIs和分数量子霍尔态(FQHS)的解析方法通常依赖于理想的布洛赫波函数，但现实的莫尔模型缺乏量子度量和贝里曲率的直接可调性，限制了理论和数值探索。在这里，我们引入了一个无监督机器学习框架，通过单粒子形状因子的分布直接建模相互作用的哈密顿量。使用变分自编码器(VAE)，我们证明无监督学习不仅能区分FCI和非FCI态，还能生成训练集中不存在的新形状因子，具有不同的拓扑特性。这个潜在空间使得能够生成和插值具有陈数|C|=1的拓扑平带形状因子，能够发现如电荷密度波等未观测到的多体态。主成分分析(PCA)进一步揭示了形状因子中的主导模式——反映了布里渊区内的关联性——可以被分解为具有近似量化陈数的分量，为量子几何的全局和拓扑结构提供了新的见解。我们的结果强调了机器学习泛化和建模拓扑量子系统的能力，为设计具有定制量子几何和平带材料中多体相的形状因子开辟了新途径。


### 论文摘要

Fractional Chern insulators (FCIs) in moire materials present a unique platform for exploring strongly correlated topological phases beyond the paradigm of ideal quantum geometry. While analytical approaches to FCIs and fractional quantum Hall states (FQHS) often rely on idealized Bloch wavefunctions, realistic moire models lack direct tunability of quantum metric and Berry curvature, limiting theoretical and numerical exploration. Here, we introduce an unsupervised machine learning framework to model interacting Hamiltonians directly through the distribution of single-particle form factors. Using a variational autoencoder (VAE), we show that unsupervised learning can not only distinguish FCI and non-FCI states, but also generate new form factors with distinct topological character, not present in the training set. This latent space enables the generation and interpolation of form factors for topological flatbands with Chern number $|C|=1$, enabling the discovery of unobserved many-body states such as charge density waves. Principal component analysis (PCA) further reveals that the dominant patterns in the form factors-reflecting correlations across the Brillouin zone-can be decomposed into components with approximately quantized Chern numbers, providing new insights into the global and topological structure of quantum geometry. Our results highlight the ability of machine learning to generalize and model topological quantum systems, paving the way for the inverse design of form factors with tailored quantum geometry and many-body phases in flatband materials.

---

## 35. From high-frequency sensors to noon reports: Using transfer learning for shaft power prediction in maritime

**论文链接:** [http://arxiv.org/abs/2510.03003v1](http://arxiv.org/abs/2510.03003v1)

**作者:** Akriti Sharma, Dogan Altan, Dusica Marijan, Arnbjørn Maressa

**发布时间:** 2025-10-03

**备注:** Keywords: transfer learning, shaft power prediction, noon reports,  sensor data, maritime

### GPT解析

### 总结

本研究提出了一种基于迁移学习的方法来预测船舶轴功率，通过结合高频传感器数据和低频日间报告数据，显著提高了预测准确性。

### 背景

全球海运运输的增长使得能源优化对降低成本和确保运营效率变得至关重要。轴功率作为影响燃料消耗的关键因素，其准确预测对优化船舶性能具有重要意义。

### 目的

开发一种准确预测船舶轴功率的方法，以优化船舶性能和降低燃料消耗。

### 方法

提出一种基于迁移学习的轴功率预测方法，模型首先在高频数据上进行训练，然后使用来自其他船舶的低频日间报告进行微调。

### 主要发现

与仅使用日间报告数据训练的模型相比，该方法在不同类型船舶上的平均绝对百分比误差均有降低：姊妹船舶降低10.6%，相似船舶降低3.6%，不同船舶降低5.3%。

### 结论

基于迁移学习的方法能有效预测船舶轴功率，特别是在具有相似配置的姊妹船舶上表现最佳。

### 翻译

随着全球海运运输的增长，能源优化对于降低成本和确保运营效率已变得至关重要。轴功率是从发动机传递到轴的机械功率，直接影响燃料消耗，因此其准确预测是优化船舶性能的关键一步。功率消耗与船舶参数（如速度和每分钟轴转速）以及天气和海况高度相关。频繁获取这些运营数据可以提高预测准确性。然而，获取高质量传感器数据通常不可行且成本高昂，使得日间报告等替代来源成为可行选择。在本文中，我们提出了一种基于迁移学习的船舶轴功率预测方法，模型首先在一艘船舶的高频数据上进行初始训练，然后使用其他船舶的低频日间报告进行微调。我们在姊妹船舶（相同尺寸和配置）、相似船舶（稍大且发动机不同）和不同船舶（不同尺寸和配置）上测试了我们的方法。实验表明，与仅使用日间报告数据训练的模型相比，姊妹船舶的平均绝对百分比误差降低了10.6%，相似船舶降低了3.6%，不同船舶降低了5.3%。


### 论文摘要

With the growth of global maritime transportation, energy optimization has become crucial for reducing costs and ensuring operational efficiency. Shaft power is the mechanical power transmitted from the engine to the shaft and directly impacts fuel consumption, making its accurate prediction a paramount step in optimizing vessel performance. Power consumption is highly correlated with ship parameters such as speed and shaft rotation per minute, as well as weather and sea conditions. Frequent access to this operational data can improve prediction accuracy. However, obtaining high-quality sensor data is often infeasible and costly, making alternative sources such as noon reports a viable option. In this paper, we propose a transfer learning-based approach for predicting vessels shaft power, where a model is initially trained on high-frequency data from a vessel and then fine-tuned with low-frequency daily noon reports from other vessels. We tested our approach on sister vessels (identical dimensions and configurations), a similar vessel (slightly larger with a different engine), and a different vessel (distinct dimensions and configurations). The experiments showed that the mean absolute percentage error decreased by 10.6 percent for sister vessels, 3.6 percent for a similar vessel, and 5.3 percent for a different vessel, compared to the model trained solely on noon report data.

---

## 36. Hierarchical Generalized Category Discovery for Brain Tumor Classification in Digital Pathology

**论文链接:** [http://arxiv.org/abs/2510.02760v1](http://arxiv.org/abs/2510.02760v1)

**作者:** Matthias Perkonigg, Patrick Rockenschaub, Georg Göbel, Adelheid Wöhrer

**发布时间:** 2025-10-03

### GPT解析

### 总结

本文提出了一种名为HGCD-BT的新型脑肿瘤分类方法，结合层次聚类与对比学习，能够同时识别已知和未知类别的肿瘤类型。

### 背景

准确的脑肿瘤分类对神经肿瘤手术中的术中决策至关重要，但现有方法局限于预定义类别，无法捕捉训练时未提供的肿瘤类型模式。无监督学习缺乏整合标记数据先验知识的能力，而半监督方法通常假设所有潜在类别都包含在标记数据中。

### 目的

开发一种能够反映脑肿瘤分类层次结构的方法，解决现有分类方法无法识别未知类别的问题。

### 方法

引入Hierarchical Generalized Category Discovery for Brain Tumor Classification (HGCD-BT)，这是一种结合层次聚类与对比学习的方法，通过扩展基于对比学习的GCD方法，纳入新的半监督层次聚类损失。

### 主要发现

在OpenSRH数据集上评估，HGCD-BT与最先进的GCD方法相比，在补丁级别分类上准确率提高了28%，特别是在识别先前未见过的肿瘤类别方面表现突出。此外，该方法在Digital Brain Tumor Atlas的苏木精-伊红染色全幻灯片图像的幻灯片级别分类上也展示了良好的可推广性。

### 结论

HGCD-BT是一种创新的脑肿瘤分类方法，能够处理已知和未知类别，并在不同数据集和成像模式下都表现出良好的性能，为脑肿瘤分类提供了新的解决方案。

### 翻译

准确的脑肿瘤分类对于神经肿瘤手术中的术中决策至关重要。然而，现有方法局限于预定义的固定类别集合，因此无法捕捉训练时未提供的肿瘤类型模式。无监督学习可以提取通用特征，但缺乏整合标记数据先验知识的能力，而半监督方法通常假设所有潜在类别都包含在标记数据中。广义类别发现(GCD)旨在通过标记未标记数据中的已知和未知类别来弥合这一差距。为了反映脑肿瘤分类法的层次结构，在本文中，我们引入了用于脑肿瘤分类的层次广义类别发现(HGCD-BT)，这是一种结合层次聚类与对比学习的新方法。我们的方法通过纳入新颖的半监督层次聚类损失，扩展了基于对比学习的GCD方法。我们在OpenSRH(一个模拟拉曼组织学脑肿瘤图像数据集)上评估了HGCD-BT，与最先进的GCD方法相比，在补丁级别分类上实现了28%的准确率提升，特别是在识别先前未见过的肿瘤类别方面。此外，我们在数字脑肿瘤图谱的苏木精和伊红染色全幻灯片图像的幻灯片级别分类上证明了HGCD-BT的可推广性，确认了其在不同成像模式下的实用性。


### 论文摘要

Accurate brain tumor classification is critical for intra-operative decision making in neuro-oncological surgery. However, existing approaches are restricted to a fixed set of predefined classes and are therefore unable to capture patterns of tumor types not available during training. Unsupervised learning can extract general-purpose features, but it lacks the ability to incorporate prior knowledge from labelled data, and semi-supervised methods often assume that all potential classes are represented in the labelled data. Generalized Category Discovery (GCD) aims to bridge this gap by categorizing both known and unknown classes within unlabelled data. To reflect the hierarchical structure of brain tumor taxonomies, in this work, we introduce Hierarchical Generalized Category Discovery for Brain Tumor Classification (HGCD-BT), a novel approach that integrates hierarchical clustering with contrastive learning. Our method extends contrastive learning based GCD by incorporating a novel semi-supervised hierarchical clustering loss. We evaluate HGCD-BT on OpenSRH, a dataset of stimulated Raman histology brain tumor images, achieving a +28% improvement in accuracy over state-of-the-art GCD methods for patch-level classification, particularly in identifying previously unseen tumor categories. Furthermore, we demonstrate the generalizability of HGCD-BT on slide-level classification of hematoxylin and eosin stained whole-slide images from the Digital Brain Tumor Atlas, confirming its utility across imaging modalities.

---

## 37. AdaRD-key: Adaptive Relevance-Diversity Keyframe Sampling for Long-form Video understanding

**论文链接:** [http://arxiv.org/abs/2510.02778v1](http://arxiv.org/abs/2510.02778v1)

**作者:** Xian Zhang, Zexi Wu, Zinuo Li, Hongming Xu, Luqi Gong, Farid Boussaid, Naoufel Werghi, Mohammed Bennamoun

**发布时间:** 2025-10-03

### GPT解析

### 总结

这篇论文提出了AdaRD-Key，一个无需训练的关键帧采样模块，用于查询驱动的长视频理解。它结合了查询相关性和视觉多样性，能够在长视频中高效选择信息丰富且不冗余的帧，实现了最先进的性能。

### 背景

当前的多模态大语言模型（MLLMs）在理解长视频时面临挑战，因为长视频具有广泛的时域长度和高信息密度。现有的关键帧选择方法存在局限性：均匀采样方法忽略关键时刻；严格时域间隔方法错过重要事件附近的精细线索；强调视觉多样性的方法则忽略查询相关性。

### 目的

开发一个高效的关键帧采样模块，能够在长视频中智能选择与查询相关且具有视觉多样性的帧，提高视觉-语言模型对长视频的理解能力。

### 方法

提出AdaRD-Key，一个无需训练的关键帧采样模块，最大化'相关性-多样性最大体积'（RD-MV）目标，结合查询条件的相关性评分和对数行列式多样性组件。采用轻量级相关性感知门控机制，当相关性分布弱时自动切换到纯多样性模式。该方法计算效率高，可在单个GPU上实时运行，并能与现有VLMs即插即用。

### 主要发现

在LongVideoBench和Video-MME上的大量实验表明，AdaRD-Key在长视频理解任务上实现了最先进的性能。该方法能够在无需额外训练的情况下，有效平衡相关性和多样性，提高视频理解的准确性。

### 结论

AdaRD-Key为长视频理解提供了一个高效、无需训练的解决方案，通过智能选择关键帧，显著提升了视觉-语言模型对长视频的理解能力，同时保持了计算效率。

### 翻译

理解长视频对于视觉-语言模型（VLMs）来说仍然是一个重大挑战，因为它们具有广泛的时域长度和高信息密度。大多数当前的多模态大语言模型（MLLMs）依赖于均匀采样，这常常忽略关键时刻，导致对查询的错误回答。同时，许多关键帧选择方法施加严格的时域间隔：一旦选择了一个帧，就会抑制相邻时间戳以减少冗余。虽然这种方法在限制重叠方面有效，但它经常错过重要事件附近的短时精细线索。其他方法则强调视觉多样性而忽略查询相关性。我们提出了AdaRD-Key，一个用于查询驱动的长视频理解的无训练关键帧采样模块。AdaRD-Key最大化了一个统一的'相关性-多样性最大体积'（RD-MV）目标，结合了查询条件的相关性评分和对数行列式多样性组件，以产生信息丰富且不冗余的帧。为了处理与视频对齐性弱的广泛查询，AdaRD-Key采用了一个轻量级的相关性感知门控机制；当相关性分布表明对齐性弱时，该方法无缝切换到纯多样性模式，无需额外监督即可提高覆盖率。我们的流水线无需训练，计算效率高（可在单个GPU上实时运行），并且可以即插即式地与现有VLMs兼容。在LongVideoBench和Video-MME上的大量实验展示了最先进的性能，特别是在长视频方面。代码可在https://github.com/Xian867/AdaRD-Key获取。


### 论文摘要

Understanding long-form videos remains a significant challenge for vision--language models (VLMs) due to their extensive temporal length and high information density. Most current multimodal large language models (MLLMs) rely on uniform sampling, which often overlooks critical moments, leading to incorrect responses to queries. In parallel, many keyframe selection approaches impose rigid temporal spacing: once a frame is chosen, an exclusion window suppresses adjacent timestamps to reduce redundancy. While effective at limiting overlap, this strategy frequently misses short, fine-grained cues near important events. Other methods instead emphasize visual diversity but neglect query relevance. We propose AdaRD-Key, a training-free keyframe sampling module for query-driven long-form video understanding. AdaRD-Key maximizes a unified Relevance--Diversity Max-Volume (RD-MV) objective, combining a query-conditioned relevance score with a log-determinant diversity component to yield informative yet non-redundant frames. To handle broad queries with weak alignment to the video, AdaRD-Key employs a lightweight relevance-aware gating mechanism; when the relevance distribution indicates weak alignment, the method seamlessly shifts into a diversity-only mode, enhancing coverage without additional supervision. Our pipeline is training-free, computationally efficient (running in real time on a single GPU), and compatible with existing VLMs in a plug-and-play manner. Extensive experiments on LongVideoBench and Video-MME demonstrate state-of-the-art performance, particularly on long-form videos. Code available at https://github.com/Xian867/AdaRD-Key.

---

## 38. IntrusionX: A Hybrid Convolutional-LSTM Deep Learning Framework with Squirrel Search Optimization for Network Intrusion Detection

**论文链接:** [http://arxiv.org/abs/2510.00572v2](http://arxiv.org/abs/2510.00572v2)

**作者:** Ahsan Farabi, Muhaiminul Rashid Shad, Israt Khandaker

**发布时间:** 2025-10-01

### GPT解析

### 总结

IntrusionX是一种混合深度学习框架，通过结合CNN和LSTM网络，并使用松鼠搜索算法优化，解决了入侵检测系统面临的网络攻击演变、高维流量数据和类别不平衡问题。

### 背景

入侵检测系统面临持续挑战，包括不断演变的网络攻击、高维流量数据和基准数据集(如NSL-KDD)中严重的类别不平衡问题。

### 目的

解决入侵检测系统面临的挑战，提高检测性能，特别是对稀有类别的检测能力。

### 方法

提出IntrusionX混合深度学习框架，整合CNN用于局部特征提取和LSTM用于时序建模，使用松鼠搜索算法进行架构优化和超参数调优，并采用严格的预处理、分层数据分割和动态类别加权技术。

### 主要发现

在NSL-KDD数据集上，IntrusionX在二元分类中达到98%的准确率，在5类分类中达到87%的准确率，少数类别的召回率显著提升(U2R: 71%，R2L: 93%)。

### 结论

IntrusionX的创新点在于其可复制、不平衡感知的设计与元启发式优化相结合，有效解决了入侵检测系统中的关键挑战。

### 翻译

入侵检测系统由于不断演变的网络攻击、高维流量数据以及像NSL-KDD这样的基准数据集中的严重类别不平衡而面临持续挑战。为解决这些问题，我们提出了IntrusionX，一种混合深度学习框架，整合了用于局部特征提取的卷积神经网络和用于时序建模的长短期记忆网络。该架构使用松鼠搜索算法进一步优化，实现了有效的超参数调优同时保持计算效率。我们的流程包含严格的预处理、分层数据分割和动态类别加权，以增强稀有类别的检测。在NSL-KDD上的实验评估表明，IntrusionX在二元分类中达到98%的准确率，在5类分类中达到87%的准确率，少数类别的召回率有显著提升(U2R: 71%，R2L: 93%)。IntrusionX的创新点在于其可复制、不平衡感知的设计与元启发式优化。


### 论文摘要

Intrusion Detection Systems (IDS) face persistent challenges due to evolving cyberattacks, high-dimensional traffic data, and severe class imbalance in benchmark datasets such as NSL-KDD. To address these issues, we propose IntrusionX, a hybrid deep learning framework that integrates Convolutional Neural Networks (CNNs) for local feature extraction and Long Short-Term Memory (LSTM) networks for temporal modeling. The architecture is further optimized using the Squirrel Search Algorithm (SSA), enabling effective hyperparameter tuning while maintaining computational efficiency. Our pipeline incorporates rigorous preprocessing, stratified data splitting, and dynamic class weighting to enhance the detection of rare classes. Experimental evaluation on NSL-KDD demonstrates that IntrusionX achieves 98% accuracy in binary classification and 87% in 5-class classification, with significant improvements in minority class recall (U2R: 71%, R2L: 93%). The novelty of IntrusionX lies in its reproducible, imbalance-aware design with metaheuristic optimization.

---

## 39. XTRA: Cross-Lingual Topic Modeling with Topic and Representation Alignments

**论文链接:** [http://arxiv.org/abs/2510.02788v1](http://arxiv.org/abs/2510.02788v1)

**作者:** Tien Phat Nguyen, Vu Minh Ngo, Tung Nguyen, Linh Van Ngo, Duc Anh Nguyen, Sang Dinh, Trung Le

**发布时间:** 2025-10-03

**备注:** 2025 EMNLP Findings

### GPT解析

### 总结

这篇论文介绍了XTRA，一种新的跨语言主题建模框架，通过结合词袋建模和多语言嵌入，实现了主题和表示的双重对齐，显著提高了主题的连贯性、多样性和跨语言一致性。

### 背景

跨语言主题建模旨在揭示不同语言之间的共享语义主题。虽然已有方法在主题多样性方面取得了一定进展，但往往难以确保高主题连贯性和跨语言一致性。

### 目的

开发一种能够同时保证主题可解释性（连贯性和多样性）和跨语言良好对齐的跨语言主题建模方法。

### 方法

提出XTRA框架，统一了词袋建模与多语言嵌入，包含两个核心组件：(1) 表示对齐：通过对比学习在共享语义空间中对齐文档-主题分布；(2) 主题对齐：将主题-词分布投影到同一空间以强制跨语言一致性。

### 主要发现

在多语料库上的实验表明，XTRA在主题连贯性、多样性和对齐质量方面显著优于强基线方法。

### 结论

XTRA的双重机制能够学习到可解释且跨语言良好对齐的主题。

### 翻译

跨语言主题建模旨在揭示不同语言之间的共享语义主题。已有多种方法解决这个问题，利用传统和神经方法。虽然先前方法在主题多样性方面取得了一些改进，但它们往往难以确保高主题连贯性和跨语言一致性。我们提出XTRA（跨语言主题建模与主题和表示对齐），一种统一词袋建模与多语言嵌入的新框架。XTRA引入两个核心组件：(1) 表示对齐，通过在共享语义空间中的对比学习对齐文档-主题分布；(2) 主题对齐，将主题-词分布投影到同一空间以强制跨语言一致性。这种双重机制使XTRA能够学习到可解释（连贯且多样）且跨语言良好对齐的主题。多语料库上的实验证实，XTRA在主题连贯性、多样性和对齐质量方面显著优于强基线方法。代码和可重现脚本可在https://github.com/tienphat140205/XTRA获取。


### 论文摘要

Cross-lingual topic modeling aims to uncover shared semantic themes across languages. Several methods have been proposed to address this problem, leveraging both traditional and neural approaches. While previous methods have achieved some improvements in topic diversity, they often struggle to ensure high topic coherence and consistent alignment across languages. We propose XTRA (Cross-Lingual Topic Modeling with Topic and Representation Alignments), a novel framework that unifies Bag-of-Words modeling with multilingual embeddings. XTRA introduces two core components: (1) representation alignment, aligning document-topic distributions via contrastive learning in a shared semantic space; and (2) topic alignment, projecting topic-word distributions into the same space to enforce crosslingual consistency. This dual mechanism enables XTRA to learn topics that are interpretable (coherent and diverse) and well-aligned across languages. Experiments on multilingual corpora confirm that XTRA significantly outperforms strong baselines in topic coherence, diversity, and alignment quality. Code and reproducible scripts are available at https: //github.com/tienphat140205/XTRA.

---

## 40. IndiCASA: A Dataset and Bias Evaluation Framework in LLMs Using Contrastive Embedding Similarity in the Indian Context

**论文链接:** [http://arxiv.org/abs/2510.02742v1](http://arxiv.org/abs/2510.02742v1)

**作者:** Santhosh G S, Akshay Govind S, Gokul S Krishnan, Balaraman Ravindran, Sriraam Natarajan

**发布时间:** 2025-10-03

**备注:** Accepted at 8th AAAI/ACM Conference on AI, Ethics, and Society (AIES)  2025

### GPT解析

### 总结

研究提出了一个基于对比学习编码器的评估框架和针对印度文化背景的新数据集IndiCASA，评估发现开源LLMs存在不同程度的刻板印象偏见，特别是与残疾相关的偏见，强调了开发更公平模型的必要性。

### 背景

大型语言模型(LLMs)因其出色的上下文理解和生成能力而在关键领域获得广泛应用，这些模型越来越多地应用于高风险场景，需要严格评估其嵌入的偏见。在印度等文化多样性的背景下，现有的基于嵌入的偏见评估方法往往难以捕捉细微的刻板印象。

### 目的

提出一个评估框架，用于捕捉大型语言模型中的细微偏见，并创建一个专门针对印度文化背景的新数据集。

### 方法

提出了一个基于使用对比学习训练的编码器的评估框架，通过嵌入相似性来捕捉细粒度的偏见；引入了名为IndiCASA的新数据集，包含2575个人类验证的句子，涵盖五个人口统计维度：种姓、性别、宗教、残疾和社会经济地位。

### 主要发现

对多个开源LLMs的评估显示，所有模型都表现出一定程度的刻板印象偏见，与残疾相关的偏见尤为明显且持续存在，宗教偏见普遍较低，可能是由于全球去偏见努力的结果。

### 结论

研究表明需要开发更公平的模型，提出的框架和数据集有助于更准确地评估和减少模型偏见。

### 翻译

大型语言模型(LLMs)因其出色的上下文理解和生成能力而在关键领域获得了显著的关注。然而，它们在高风险应用中的日益部署需要对嵌入偏见进行严格评估，特别是在像印度这样的文化多样性背景下，现有的基于嵌入的偏见评估方法往往难以捕捉细微的刻板印象。我们提出了一个基于使用对比学习训练的编码器的评估框架，通过嵌入相似性捕捉细粒度的偏见。我们还引入了一个名为IndiCASA(基于印度偏见的上下文对齐的刻板印象和反刻板印象)的新数据集，包含2575个人类验证的句子，涵盖五个人口统计维度：种姓、性别、宗教、残疾和社会经济地位。我们对多个开源LLMs的评估显示，所有模型都表现出一定程度的刻板印象偏见，与残疾相关的偏见尤为明显且持续存在，宗教偏见普遍较低，可能是由于全球去偏见努力的结果，这展示了开发更公平模型的必要性。


### 论文摘要

Large Language Models (LLMs) have gained significant traction across critical domains owing to their impressive contextual understanding and generative capabilities. However, their increasing deployment in high stakes applications necessitates rigorous evaluation of embedded biases, particularly in culturally diverse contexts like India where existing embedding-based bias assessment methods often fall short in capturing nuanced stereotypes. We propose an evaluation framework based on a encoder trained using contrastive learning that captures fine-grained bias through embedding similarity. We also introduce a novel dataset - IndiCASA (IndiBias-based Contextually Aligned Stereotypes and Anti-stereotypes) comprising 2,575 human-validated sentences spanning five demographic axes: caste, gender, religion, disability, and socioeconomic status. Our evaluation of multiple open-weight LLMs reveals that all models exhibit some degree of stereotypical bias, with disability related biases being notably persistent, and religion bias generally lower likely due to global debiasing efforts demonstrating the need for fairer model development.

---

## 41. From Pixels to Factors: Learning Independently Controllable State Variables for Reinforcement Learning

**论文链接:** [http://arxiv.org/abs/2510.02484v1](http://arxiv.org/abs/2510.02484v1)

**作者:** Rafael Rodriguez-Sanchez, Cameron Allen, George Konidaris

**发布时间:** 2025-10-02

### GPT解析

### 总结

本文提出了一种名为动作可控因子化(ACF)的对比学习方法，用于解决从高维观测中学习因子化表示的问题，结合了因子化MDP的样本效率和深度强化学习处理高维输入的能力。

### 背景

基于因子化马尔可夫决策过程的算法比因子无关方法更具有样本效率，但它们假设因子化表示是预先已知的，这在智能体只能看到高维观测时失效；而深度强化学习能处理高维输入但无法利用因子化结构。

### 目的

解决表示问题，使算法能够从高维观测中学习因子化表示，发现可独立控制的潜在变量。

### 方法

提出动作可控因子化(ACF)对比学习方法，利用动作的稀疏性（动作通常只影响变量的一小部分，其余部分在环境动态下演化）为对比训练提供信息数据。

### 主要发现

ACF能够直接从像素观测中恢复真实的可控因子，在Taxi、FourRooms和MiniGrid-DoorKey三个基准测试上表现一致优于基解纠缠算法。

### 结论

ACF成功解决了因子化表示预先已知的要求与高维观测处理之间的矛盾，为强化学习中的表示学习提供了新方法。

### 翻译

利用因子化马尔可夫决策过程的算法比因子无关方法具有更高的样本效率，但它们假设因子化表示是预先已知的——这一要求在智能体只能看到高维观测时会失效。相反，深度强化学习能够处理此类输入，但无法受益于因子化结构。我们通过动作可控因子化(ACF)解决这一表示问题，ACF是一种对比学习方法，能够发现可独立控制的潜在变量——即每个动作可以单独影响的状态分量。ACF利用稀疏性：动作通常只影响变量的一小部分，而其余部分在环境动态下演化，为对比训练提供信息数据。ACF能够在三个具有已知因子化结构的基准测试(Taxi、FourRooms和MiniGrid-DoorKey)上直接从像素观测中恢复真实的可控因子，并且一致性地优于基解纠缠算法。


### 论文摘要

Algorithms that exploit factored Markov decision processes are far more sample-efficient than factor-agnostic methods, yet they assume a factored representation is known a priori -- a requirement that breaks down when the agent sees only high-dimensional observations. Conversely, deep reinforcement learning handles such inputs but cannot benefit from factored structure. We address this representation problem with Action-Controllable Factorization (ACF), a contrastive learning approach that uncovers independently controllable latent variables -- state components each action can influence separately. ACF leverages sparsity: actions typically affect only a subset of variables, while the rest evolve under the environment's dynamics, yielding informative data for contrastive training. ACF recovers the ground truth controllable factors directly from pixel observations on three benchmarks with known factored structure -- Taxi, FourRooms, and MiniGrid-DoorKey -- consistently outperforming baseline disentanglement algorithms.

---

## 42. VarCoNet: A variability-aware self-supervised framework for functional connectome extraction from resting-state fMRI

**论文链接:** [http://arxiv.org/abs/2510.02120v2](http://arxiv.org/abs/2510.02120v2)

**作者:** Charalampos Lamprou, Aamna Alshehhi, Leontios J. Hadjileontiadis, Mohamed L. Seghier

**发布时间:** 2025-10-02

### GPT解析

### 总结

VarCoNet是一个创新的自监督框架，通过将大脑功能的个体间变异性视为有意义数据而非噪声，实现了从静息态fMRI数据中稳健提取功能连接组。该框架在主体指纹识别和自闭症谱系障碍分类等下游任务上表现出优越性、稳健性、可解释性和泛化能力。

### 背景

精准医疗需要考虑大脑功能的个体间变异性，但传统方法通常将这种变异性视为噪声而非有价值的信息。

### 目的

开发一个增强的自监督框架VarCoNet，用于从静息态fMRI数据中稳健提取功能连接组，并将功能个体间变异性作为有意义的数据加以利用。

### 方法

VarCoNet采用自监督对比学习利用内在的功能个体间变异性，作为大脑功能编码器生成可直接应用于下游任务的FC嵌入。它通过基于分割rs-fMRI信号的新型增强策略促进对比学习，核心是集成了1D-CNN-Transformer编码器进行高级时间序列处理，并采用贝叶斯超参数优化增强稳健性。

### 主要发现

VarCoNet在两个下游任务上得到验证：使用人类连接组计划数据进行主体指纹识别，以及使用ABIDE I和II数据集进行自闭症谱系障碍分类。与包括13种深度学习方法在内的最先进方法相比，VarCoNet展现出优越性、稳健性、可解释性和泛化能力。

### 结论

VarCoNet为静息态fMRI中的功能连接组分析提供了一个通用且稳健的框架，能够有效利用大脑功能的个体间变异性。

### 翻译

考虑大脑功能的个体间变异性对精准医疗至关重要。通过将功能个体间变异性视为有意义的数据而非噪声，我们引入了VarCoNet，一个增强的自监督框架，用于从静息态fMRI数据中稳健提取功能连接组。VarCoNet采用自监督对比学习来利用内在的功能个体间变异性，作为大脑功能编码器生成可直接应用于下游任务的FC嵌入，即使在没有标记数据的情况下。对比学习通过一种基于分割rs-fMRI信号的新型增强策略促进。其核心是集成了一个1D-CNN-Transformer编码器用于高级时间序列处理，增强了稳健的贝叶斯超参数优化。我们的VarCoNet框架在两个下游任务上进行了评估：(i)使用人类连接组计划的rs-fMRI数据进行主体指纹识别，和(ii)使用ABIDE I和ABIDE II数据集的rs-fMRI数据进行自闭症谱系障碍分类。使用不同的脑区分割方法，我们与包括13种深度学习方法在内的最先进方法进行了广泛测试，证明了VarCoNet的优越性、稳健性、可解释性和泛化能力。总体而言，VarCoNet为rs-fMRI中的FC分析提供了一个通用且稳健的框架。


### 论文摘要

Accounting for inter-individual variability in brain function is key to precision medicine. Here, by considering functional inter-individual variability as meaningful data rather than noise, we introduce VarCoNet, an enhanced self-supervised framework for robust functional connectome (FC) extraction from resting-state fMRI (rs-fMRI) data. VarCoNet employs self-supervised contrastive learning to exploit inherent functional inter-individual variability, serving as a brain function encoder that generates FC embeddings readily applicable to downstream tasks even in the absence of labeled data. Contrastive learning is facilitated by a novel augmentation strategy based on segmenting rs-fMRI signals. At its core, VarCoNet integrates a 1D-CNN-Transformer encoder for advanced time-series processing, enhanced with a robust Bayesian hyperparameter optimization. Our VarCoNet framework is evaluated on two downstream tasks: (i) subject fingerprinting, using rs-fMRI data from the Human Connectome Project, and (ii) autism spectrum disorder (ASD) classification, using rs-fMRI data from the ABIDE I and ABIDE II datasets. Using different brain parcellations, our extensive testing against state-of-the-art methods, including 13 deep learning methods, demonstrates VarCoNet's superiority, robustness, interpretability, and generalizability. Overall, VarCoNet provides a versatile and robust framework for FC analysis in rs-fMRI.

---

## 43. Adaptive Node Feature Selection For Graph Neural Networks

**论文链接:** [http://arxiv.org/abs/2510.03096v1](http://arxiv.org/abs/2510.03096v1)

**作者:** Ali Azizpour, Madeline Navarro, Santiago Segarra

**发布时间:** 2025-10-03

### GPT解析

### 总结

提出了一种自适应节点特征选择方法，用于图神经网络，能够在训练过程中识别并移除不必要的特征，基于验证性能变化确定特征重要性，适用于不同图架构和挑战性图学习场景。

### 背景

测量特征如何贡献于模型输出对于解释决策、降低维度和通过消除无用变量提高性能至关重要。然而，图结构数据引入了复杂的依赖关系，可能不适合传统的特征重要性指标。

### 目的

开发一种模型和任务无关的方法，在训练过程中基于验证性能变化确定相关特征，提供特征重要性分数，并跟踪特征在连续被删除过程中相关性的演变。

### 方法

提出基于干预的方法，通过置换特征值并观察验证性能变化来确定特征重要性。不仅训练结束后返回特征重要性分数，还跟踪特征在连续被删除过程中相关性的演变。

### 主要发现

实证结果表明，该方法能够灵活适应不同的图架构，并且能够适应更具挑战性的图学习设置。

### 结论

该自适应节点特征选择方法能够有效识别和移除不必要的特征，提高图神经网络的性能和可解释性。

### 翻译

我们提出了一种用于图神经网络(GNNs)的自适应节点特征选择方法，该方法在训练过程中能够识别并移除不必要的特征。测量特征如何贡献于模型输出的能力对于解释决策、降低维度以及通过消除无用变量来提高性能至关重要。然而，图结构数据引入了复杂的依赖关系，这些关系可能不适合传统的特征重要性指标。受此挑战启发，我们提出了一种模型和任务无关的方法，该方法基于置换特征值后验证性能的变化来确定训练过程中的相关特征。我们通过表征GNN性能如何依赖于节点数据与图结构之间的关系，从理论上验证了我们基于干预的方法。我们不仅会在训练结束后返回特征重要性分数，还会跟踪特征在连续被删除过程中相关性的演变。因此，我们可以监控特征是否被有效消除，并使用此技术评估其他指标。我们的实证结果验证了我们的方法对不同图架构的灵活性，以及其对更具挑战性的图学习设置的适应性。


### 论文摘要

We propose an adaptive node feature selection approach for graph neural networks (GNNs) that identifies and removes unnecessary features during training. The ability to measure how features contribute to model output is key for interpreting decisions, reducing dimensionality, and even improving performance by eliminating unhelpful variables. However, graph-structured data introduces complex dependencies that may not be amenable to classical feature importance metrics. Inspired by this challenge, we present a model- and task-agnostic method that determines relevant features during training based on changes in validation performance upon permuting feature values. We theoretically motivate our intervention-based approach by characterizing how GNN performance depends on the relationships between node data and graph structure. Not only do we return feature importance scores once training concludes, we also track how relevance evolves as features are successively dropped. We can therefore monitor if features are eliminated effectively and also evaluate other metrics with this technique. Our empirical results verify the flexibility of our approach to different graph architectures as well as its adaptability to more challenging graph learning settings.

---

## 44. Bootstrap Learning for Combinatorial Graph Alignment with Sequential GNNs

**论文链接:** [http://arxiv.org/abs/2510.03086v1](http://arxiv.org/abs/2510.03086v1)

**作者:** Marc Lelarge

**发布时间:** 2025-10-03

**备注:** 27 pages, 10 figures, 12 tables

### GPT解析

### 总结

本文提出了一种新颖的链式图神经网络方法来解决图对齐问题，显著提高了图神经网络在组合问题上的性能。

### 背景

图神经网络在组合问题上一直难以超越传统优化方法，限制了其实际应用影响。

### 目的

解决图对齐这一NP-hard任务，仅使用结构信息在未标记图之间寻找最优节点对应关系。

### 方法

引入链式程序，训练一系列图神经网络迭代改进相似度矩阵；结合在节点对上操作的架构，捕获全局结构模式；与传统优化方法结合作为后处理。

### 主要发现

链式图神经网络在具有挑战性的实例上比现有方法提高3倍以上准确性；唯一解决了所有竞争方法都失效的正则图；与传统优化结合后显著优于最先进求解器。

### 结论

通过链式图神经网络架构和节点对操作，成功解决了图神经网络在组合问题上的局限性，在图对齐问题上取得显著进展。

### 翻译

图神经网络在组合问题上一直难以超越传统优化方法，限制了其实际应用。我们通过引入一种新颖的链式程序来解决图对齐问题，这是一个基础的NP-hard任务，仅使用结构信息在未标记图之间寻找最优节点对应关系。我们的方法训练一系列图神经网络，每个网络学习迭代改进前一个网络产生的相似度矩阵。在推理过程中，这创造了一种自举效应：每个图神经网络通过整合先前迭代中关于节点对齐质量的离散排名信息来改进部分解决方案。我们结合了一个强大的架构，该架构在节点对而非单个节点上操作，捕获了对齐至关重要的全局结构模式，这是标准消息传递网络无法表示的。在合成基准上的广泛实验表明了显著的改进：我们的链式图神经网络在具有挑战性的实例上比现有方法提高了3倍以上的准确性，并且唯一解决了所有竞争方法都失效的正则图。当与传统优化结合作为后处理时，我们的方法在图对齐基准上显著优于最先进的求解器。


### 论文摘要

Graph neural networks (GNNs) have struggled to outperform traditional optimization methods on combinatorial problems, limiting their practical impact. We address this gap by introducing a novel chaining procedure for the graph alignment problem, a fundamental NP-hard task of finding optimal node correspondences between unlabeled graphs using only structural information. Our method trains a sequence of GNNs where each network learns to iteratively refine similarity matrices produced by previous networks. During inference, this creates a bootstrap effect: each GNN improves upon partial solutions by incorporating discrete ranking information about node alignment quality from prior iterations. We combine this with a powerful architecture that operates on node pairs rather than individual nodes, capturing global structural patterns essential for alignment that standard message-passing networks cannot represent. Extensive experiments on synthetic benchmarks demonstrate substantial improvements: our chained GNNs achieve over 3x better accuracy than existing methods on challenging instances, and uniquely solve regular graphs where all competing approaches fail. When combined with traditional optimization as post-processing, our method substantially outperforms state-of-the-art solvers on the graph alignment benchmark.

---

## 45. BrainIB++: Leveraging Graph Neural Networks and Information Bottleneck for Functional Brain Biomarkers in Schizophrenia

**论文链接:** [http://arxiv.org/abs/2510.03004v1](http://arxiv.org/abs/2510.03004v1)

**作者:** Tianzheng Hu, Qiang Li, Shu Liu, Vince D. Calhoun, Guido van Wingen, Shujian Yu

**发布时间:** 2025-10-03

**备注:** This manuscript has been accepted by Biomedical Signal Processing and  Control and the code is available at  https://github.com/TianzhengHU/BrainIB_coding/tree/main/BrainIB_GIB

### GPT解析

### 总结

本研究提出了一种名为BrainIB++的端到端创新图神经网络框架，应用信息瓶颈原则识别最具信息量的脑区域作为子图，提高了精神障碍诊断模型的准确性和可解释性。

### 背景

精神障碍诊断模型正在快速发展，基于rs-fMRI的机器学习分类器被用于识别区分精神障碍与健康对照的脑生物标志物。传统机器学习模型依赖大量特征工程引入偏差，而深度学习模型虽无需人工干预但缺乏可解释性，限制了临床应用。

### 目的

开发一种端到端的创新图神经网络框架，在保持高诊断准确性的同时提高模型的可解释性，识别出与临床相关联的脑生物标志物。

### 方法

引入名为BrainIB++的图神经网络框架，应用信息瓶颈原则在模型训练过程中识别最具信息量的数据驱动脑区域作为子图进行解释。在三个多队列精神分裂症数据集上评估性能，并与九种已建立的脑网络分类方法进行比较。

### 主要发现

BrainIB++模型在诊断准确性上表现一致优异，对未见过的数据具有泛化能力。模型识别的子图与精神分裂症已建立的生物标志物相对应，特别强调了视觉、感觉运动和更高认知脑功能网络的异常。

### 结论

BrainIB++模型与临床生物标志物的一致性增强了其可解释性，强调了其在现实世界诊断应用中的相关性和潜力。

### 翻译

精神障碍诊断模型的发展正在该领域获得关注。最近，基于静息态功能磁共振成像的机器学习分类器已被开发用于识别区分精神障碍与健康对照的脑生物标志物。然而，传统基于机器学习的诊断模型通常依赖大量特征工程，通过人工干预引入偏差。虽然深度学习模型预期无需人工操作，但它们缺乏可解释性，在获取可解释和可靠的脑生物标志物以支持诊断决策方面面临重大挑战，最终限制了它们的临床适用性。在本研究中，我们引入了一种名为BrainIB++的端到端创新图神经网络框架，该框架应用信息瓶颈原则在模型训练期间识别最具信息量的数据驱动脑区域作为子图进行解释。我们在三个多队列精神分裂症数据集上评估了我们的模型与九种已建立的脑网络分类方法的性能。它始终表现出优越的诊断准确性，并显示出对未见数据的泛化能力。此外，我们模型识别的子图也与精神分裂症中已建立的临床生物标志物相对应，特别强调了视觉、感觉运动和更高认知脑功能网络的异常。这种一致性增强了模型的可解释性，并强调了其对现实世界诊断应用的相关性。


### 论文摘要

The development of diagnostic models is gaining traction in the field of psychiatric disorders. Recently, machine learning classifiers based on resting-state functional magnetic resonance imaging (rs-fMRI) have been developed to identify brain biomarkers that differentiate psychiatric disorders from healthy controls. However, conventional machine learning-based diagnostic models often depend on extensive feature engineering, which introduces bias through manual intervention. While deep learning models are expected to operate without manual involvement, their lack of interpretability poses significant challenges in obtaining explainable and reliable brain biomarkers to support diagnostic decisions, ultimately limiting their clinical applicability. In this study, we introduce an end-to-end innovative graph neural network framework named BrainIB++, which applies the information bottleneck (IB) principle to identify the most informative data-driven brain regions as subgraphs during model training for interpretation. We evaluate the performance of our model against nine established brain network classification methods across three multi-cohort schizophrenia datasets. It consistently demonstrates superior diagnostic accuracy and exhibits generalizability to unseen data. Furthermore, the subgraphs identified by our model also correspond with established clinical biomarkers in schizophrenia, particularly emphasizing abnormalities in the visual, sensorimotor, and higher cognition brain functional network. This alignment enhances the model's interpretability and underscores its relevance for real-world diagnostic applications.

---

## 46. Enhancing Photogrammetry Reconstruction For HRTF Synthesis Via A Graph Neural Network

**论文链接:** [http://arxiv.org/abs/2510.02813v1](http://arxiv.org/abs/2510.02813v1)

**作者:** Ludovic Pirard, Katarina C. Poole, Lorenzo Picinali

**发布时间:** 2025-10-03

**备注:** Accepted for poster presentation at Forum Acusticum Euronoise 2025,  Malaga, Spain

### GPT解析

### 总结

该研究提出使用图神经网络(GNN)提升摄影测量重建网格分辨率，解决传统HRTFs获取方法成本高和专业要求高的问题，实现个体化HRTFs合成。

### 背景

传统HRTFs获取方法依赖专业设备和声学专业知识，存在可及性挑战；高分辨率3D建模可通过数值方法合成HRTFs，但高级3D扫描仪成本高且可用性有限；摄影测量法生成3D头部网格虽可行，但分辨率不足限制了其在HRTF合成中的应用。

### 目的

研究使用图神经网络和神经细分技术将低分辨率摄影测量重建网格提升为高分辨率网格的可行性，进而用于合成个体化的HRTFs。

### 方法

使用Apple Photogrammetry API处理SONICOM数据集重建低分辨率头部网格；训练GNN网络通过基于Hausdorff距离的损失函数将低分辨率网格提升为高分辨率；通过几何验证和Mesh2HRTF生成的合成HRTFs评估性能；将合成HRTFs与高分辨率3D扫描计算结果、声学测量结果及KEMAR HRTF进行对比，使用感知相关数值分析和行为实验（包括定位和空间掩蔽释放任务）进行评估。

### 主要发现

图神经网络能有效提升低分辨率摄影测量网格的分辨率；合成的HRTFs与从高分辨率3D扫描计算出的HRTFs、声学测量的HRTFs以及KEMAR HRTF具有可比性；在定位和空间掩蔽释放任务中表现良好。

### 结论

使用图神经网络和神经细分技术提升低分辨率摄影测量重建网格的分辨率是可行的；这种方法可用于合成高质量的个体化HRTFs，无需依赖昂贵的3D扫描设备。

### 翻译

传统的头相关传递函数获取方法依赖专业设备和声学专业知识，带来了可及性挑战。 alternatively, 高分辨率3D建模提供了使用边界元法等数值方法合成HRTFs的途径。然而，高级3D扫描仪的高成本和有限可用性限制了它们的适用性。摄影测量已被提出作为生成3D头部网格的解决方案，但其分辨率限制阻碍了其在HRTF合成中的应用。为解决这些限制，本研究探讨了使用图神经网络和神经细分技术将低分辨率摄影测量重建网格提升为高分辨率网格的可行性，这些网格随后可用于合成个体化的HRTFs。使用Apple Photogrammetry API处理SONICOM数据集中的摄影测量数据，以重建低分辨率头部网格。然后使用成对的高低分辨率网格数据集训练GNN，使用基于Hausdorff距离的损失函数将低分辨率输入提升为高分辨率输出。通过几何验证和通过Mesh2HRTF生成的合成HRTFs来验证GNN在未见过的摄影测量数据上的性能。合成的HRTFs与从高分辨率3D扫描计算的HRTFs、声学测量的HRTFs以及KEMAR HRTF进行比较，使用感知相关的数值分析和行为实验，包括定位和空间掩蔽释放任务。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决如何通过提高摄影测量重建（Photogrammetry Reconstruction）的分辨率来改进头相关传输函数（HRTF）的合成质量问题。这个问题重要是因为传统HRTF获取方法需要专业设备和声学知识，而高分辨率3D扫描又成本高昂，限制了普通用户获得个体化HRTF的能力，而个体化HRTF能提供更准确的空间音频体验，减少前后混淆和高度感知障碍。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先分析了传统HRTF获取方法的局限性和摄影测量的优缺点，认识到提高摄影测量网格分辨率可能是解决方案。他们借鉴了Liu等人提出的数据驱动粗到细几何建模框架和Schmidt等人的表面同映射技术，设计了使用图神经网络（GNN）对低分辨率摄影测量网格进行上采样的方法，并使用基于Hausdorff距离的损失函数进行优化，最终通过HRTF合成和评估验证方法效果。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是使用图神经网络（GNN）对摄影测量重建的低分辨率头部网格进行上采样，提高其分辨率和耳部细节，从而合成更高质量的个体化HRTF。整体流程包括：1)使用iPhone采集摄影测量数据和专业3D扫描作为参考；2)使用Apple摄影测量API重建低分辨率网格；3)计算低-高分辨率网格之间的双射映射；4)训练GNN学习网格上采样；5)使用Mesh2HRTF从不同网格合成HRTF；6)通过数值指标（如LSD）和感知实验（如定位任务）评估效果。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)首次应用GNN提高摄影测量重建头部网格分辨率，特别是耳部细节；2)创新性地使用表面同胚映射技术处理不同获取技术间的网格差异；3)使用基于Hausdorff距离的损失函数优化网络；4)结合数值和感知评估全面验证效果。相比之前工作，本文方法大幅降低了获取个体化HRTF的成本和复杂性，使用消费级设备即可实现，同时保持了与高成本方法相当的质量。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '本文通过应用图神经网络提高摄影测量重建头部网格的分辨率，特别是改进耳部细节，为经济可及地获取高质量个体化头相关传输函数（HRTF）提供了创新解决方案，显著提升了普通消费者获得沉浸式音频体验的可能性。'}


### 论文摘要

Traditional Head-Related Transfer Functions (HRTFs) acquisition methods rely on specialised equipment and acoustic expertise, posing accessibility challenges. Alternatively, high-resolution 3D modelling offers a pathway to numerically synthesise HRTFs using Boundary Elements Methods and others. However, the high cost and limited availability of advanced 3D scanners restrict their applicability. Photogrammetry has been proposed as a solution for generating 3D head meshes, though its resolution limitations restrict its application for HRTF synthesis. To address these limitations, this study investigates the feasibility of using Graph Neural Networks (GNN) using neural subdivision techniques for upsampling low-resolution Photogrammetry-Reconstructed (PR) meshes into high-resolution meshes, which can then be employed to synthesise individual HRTFs. Photogrammetry data from the SONICOM dataset are processed using Apple Photogrammetry API to reconstruct low-resolution head meshes. The dataset of paired low- and high-resolution meshes is then used to train a GNN to upscale low-resolution inputs to high-resolution outputs, using a Hausdorff Distance-based loss function. The GNN's performance on unseen photogrammetry data is validated geometrically and through synthesised HRTFs generated via Mesh2HRTF. Synthesised HRTFs are evaluated against those computed from high-resolution 3D scans, to acoustically measured HRTFs, and to the KEMAR HRTF using perceptually-relevant numerical analyses as well as behavioural experiments, including localisation and Spatial Release from Masking (SRM) tasks.

---

## 47. VisitHGNN: Heterogeneous Graph Neural Networks for Modeling Point-of-Interest Visit Patterns

**论文链接:** [http://arxiv.org/abs/2510.02702v1](http://arxiv.org/abs/2510.02702v1)

**作者:** Lin Pang, Jidong J. Yang

**发布时间:** 2025-10-03

**备注:** 16 pages, 9 figures, 5 tables

### GPT解析

### 总结

研究开发了一个名为VisitHGNN的异构关系特定图神经网络，用于预测城市居民从社区到各个兴趣点(POIs)的访问概率，支持交通规划和公共卫生决策。

### 背景

了解城市居民如何在社区和目的地之间出行对于交通规划、出行管理和公共卫生至关重要。通过分析城市间的出行模式，可以估计社区对城市人流量的贡献。

### 目的

开发一种能够准确预测从社区到各个兴趣点访问概率的方法，支持需求估计、可达性评估和多模式规划，为城市规划和交通政策提供决策支持。

### 方法

引入VisitHGNN异构关系特定图神经网络，结合POIs的多种属性和人口普查区块组(CBGs)的社会经济人口变量，通过空间邻接和带距离注释的跨类型边缘连接，使用掩码Kullback-Leibler散度进行训练和预测。

### 主要发现

VisitHGNN在富尔顿县数据上表现优异，平均KL散度0.287，MAE 0.008，Top-1准确率0.853，R平方0.892，显著优于基线方法，并与实际访问模式高度一致(NDCG@50 = 0.966; Recall@5 = 0.611)。

### 结论

该模型能够高度准确地反映实际出行行为，在城市规划、交通政策、出行系统设计和公共卫生决策支持方面具有巨大潜力。

### 翻译

了解城市居民如何在社区和目的地之间出行对于交通规划、出行管理和公共卫生至关重要。通过挖掘城市地点间具有空间、时间和功能关系的历史起点到终点流动模式，我们估计了从社区到特定目的地的访问概率。这些概率捕捉了社区层面城市车辆和人流量贡献，支持需求估计、可达性评估和多模式规划。特别是，我们引入了VisitHGNN，这是一种异构关系特定的图神经网络，专为预测单个兴趣点(POIs)的访问概率而设计。POIs使用数值、JSON派生和文本属性进行表征，并增加了POI-POI空间邻近度、时间共活动和品牌亲和力的固定摘要，而人口普查区块组(CBGs)则用72个社会经济人口变量描述。CBGs通过空间邻接连接，POIs和CBGs通过带距离注释的跨类型边缘连接。推理被限制在基于距离的可能来源CBG候选集内，训练最小化了掩码Kullback-Leibler(KL)散度，以在候选集上产生概率分布。使用美国乔治亚州富尔顿县的每周移动数据，VisitHGNN取得了强大的预测性能，平均KL散度为0.287，平均绝对误差为0.008，Top-1准确率为0.853，R平方为0.892，显著优于成对MLP和仅距离的基线方法，并与实际访问模式紧密一致(NDCG@50 = 0.966; Recall@5 = 0.611)。所得分布高度忠实地反映了观察到的出行行为，突显了该模型在城市规划、交通政策、出行系统设计和公共卫生决策支持方面的潜力。


### 论文摘要

Understanding how urban residents travel between neighborhoods and destinations is critical for transportation planning, mobility management, and public health. By mining historical origin-to-destination flow patterns with spatial, temporal, and functional relations among urban places, we estimate probabilities of visits from neighborhoods to specific destinations. These probabilities capture neighborhood-level contributions to citywide vehicular and foot traffic, supporting demand estimation, accessibility assessment, and multimodal planning. Particularly, we introduce VisitHGNN, a heterogeneous, relation-specific graph neural network designed to predict visit probabilities at individual Points of interest (POIs). POIs are characterized using numerical, JSON-derived, and textual attributes, augmented with fixed summaries of POI--POI spatial proximity, temporal co-activity, and brand affinity, while census block groups (CBGs) are described with 72 socio-demographic variables. CBGs are connected via spatial adjacency, and POIs and CBGs are linked through distance-annotated cross-type edges. Inference is constrained to a distance-based candidate set of plausible origin CBGs, and training minimizes a masked Kullback-Leibler (KL) divergence to yield probability distribution across the candidate set. Using weekly mobility data from Fulton County, Georgia, USA, VisitHGNN achieves strong predictive performance with mean KL divergence of 0.287, MAE of 0.008, Top-1 accuracy of 0.853, and R-square of 0.892, substantially outperforming pairwise MLP and distance-only baselines, and aligning closely with empirical visitation patterns (NDCG@50 = 0.966); Recall@5 = 0.611). The resulting distributions closely mirror observed travel behavior with high fidelity, highlighting the model's potential for decision support in urban planning, transportation policy, mobility system design, and public health.

---

## 48. Identifying Asymptomatic Nodes in Network Epidemics using Graph Neural Networks

**论文链接:** [http://arxiv.org/abs/2510.02568v1](http://arxiv.org/abs/2510.02568v1)

**作者:** Conrado Catarcione Pinto, Amanda Camacho Novaes de Oliveira, Rodrigo Sapienza Luna, Daniel Ratton Figueiredo

**发布时间:** 2025-10-02

**备注:** Paper presented in the 35th Brazilian Conference on Intelligent  Systems (BRACIS)

### GPT解析

### 总结

本文提出了一种使用图神经网络(GNN)识别流行病中无症状个体的方法，该方法能够在不进行全面检测的情况下准确识别无症状感染者，且在不同网络环境下表现鲁棒。

### 背景

某些流行病中的感染者可能保持无症状状态但仍能传播感染，这增加了疫情控制的难度。识别这些无症状个体对疫情监测至关重要，但定期广泛检测健康人群成本过高。

### 目的

解决在SI(易感-感染)网络流行病学模型中识别无症状个体的问题，这些个体的观察状态与易感节点相同，难以区分。

### 方法

采用带有监督学习的图神经网络模型，从具有已观察感染节点的网络构建节点特征集，用于将健康节点分类为无症状或易感状态。

### 主要发现

所提出的方法在不同网络模型、网络规模和观察感染比例的场景下均表现出鲁棒性，能够准确识别无症状节点，并具有良好的泛化能力。

### 结论

基于图神经网络的识别方法能有效解决无症状感染者识别难题，为疫情监测和控制提供了一种经济有效的解决方案。

### 翻译

某些流行病中的感染者可能保持无症状状态，但仍能携带和传播感染。这些个体促进了疫情的传播，对公共卫生政策构成重大挑战。识别无症状个体对测量和控制疫情至关重要，但定期广泛检测健康人群往往成本过高。本文考虑经典的SI(易感-感染)网络流行病学模型，解决识别无症状个体的问题，其中部分感染节点未被观察到(即它们的观察状态与易感节点相同)。为将健康节点分类为无症状或易感，采用了一种基于监督学习的图神经网络模型，从具有已观察感染节点的网络构建节点特征集。该方法在不同网络模型、网络大小和观察感染比例下进行了评估。结果表明，所提出的方法在不同场景下具有鲁棒性，能准确识别无症状节点，同时推广到不同网络大小和观察感染比例的情况。


### 论文摘要

Infected individuals in some epidemics can remain asymptomatic while still carrying and transmitting the infection. These individuals contribute to the spread of the epidemic and pose a significant challenge to public health policies. Identifying asymptomatic individuals is critical for measuring and controlling an epidemic, but periodic and widespread testing of healthy individuals is often too costly. This work tackles the problem of identifying asymptomatic individuals considering a classic SI (Susceptible-Infected) network epidemic model where a fraction of the infected nodes are not observed as infected (i.e., their observed state is identical to susceptible nodes). In order to classify healthy nodes as asymptomatic or susceptible, a Graph Neural Network (GNN) model with supervised learning is adopted where a set of node features are built from the network with observed infected nodes. The approach is evaluated across different network models, network sizes, and fraction of observed infections. Results indicate that the proposed methodology is robust across different scenarios, accurately identifying asymptomatic nodes while also generalizing to different network sizes and fraction of observed infections.

---

## 49. On The Expressive Power of GNN Derivatives

**论文链接:** [http://arxiv.org/abs/2510.02565v1](http://arxiv.org/abs/2510.02565v1)

**作者:** Yam Eitan, Moshe Eliasof, Yoav Gelberg, Fabrizio Frasca, Guy Bar-Shalom, Haggai Maron

**发布时间:** 2025-10-02

**备注:** 30 pages, 3 figures

### GPT解析

### 总结

本文提出了一种高阶导数图神经网络(HOD-GNN)，通过利用基础模型的高阶节点导数来增强消息传递神经网络的表达能力，在多个图学习基准测试中表现出色。

### 背景

尽管图神经网络(GNNs)取得了显著进展，但其表达能力仍然是一个基本挑战。关于GNN表达能力的研究已经产生了多种具有不同表达能力的架构层次。同时，GNN对节点特征的导数研究已在过度挤压和过度平滑现象、GNN可解释性等方面得到广泛研究。

### 目的

探索利用GNN对节点特征的导数作为增强GNN表达能力的新方法，填补这一研究空白。

### 方法

引入高阶导数GNN(HOD-GNN)，一种新颖方法，通过利用基础模型的高阶节点导数增强消息传递神经网络(MPNNs)的表达能力。这些导数生成具有结构感知能力的节点嵌入，由第二个GNN在端到端可训练架构中处理。同时开发了利用图稀疏性和并行性的消息传递算法以提高计算效率。

### 主要发现

理论上，HOD-GNN架构家族的表达能力与WL层次结构一致；发现了HOD-GNN、子图GNN和流行的结构编码方案之间的深层联系；在多个图学习基准测试上表现出强劲性能。

### 结论

高阶导数GNN(HOD-GNN)为增强图神经网络表达能力提供了有效途径，通过利用高阶节点导数生成结构感知的节点嵌入，结合理论分析和实验验证，证明了其在图学习任务中的优越性能。

### 翻译

尽管图神经网络(GNNs)取得了显著进展，但其有限的表达能力仍然是一个基本挑战。关于GNN表达能力的研究已经产生了许多具有表达能力的架构，形成了具有越来越强表达能力模型的架构层次。同时，关于GNN对节点特征的导数研究已经在过度挤压和过度平滑现象、GNN可解释性等方面得到广泛研究。迄今为止，这些导数尚未被探索作为增强GNN表达能力的方法。在本文中，我们展示了这些导数为增强GNN表达能力提供了一种自然方式。我们引入了高阶导数GNN(HOD-GNN)，一种新颖方法，通过利用基础模型的高阶节点导数来增强消息传递神经网络(MPNNs)的表达能力。这些导数生成具有结构感知能力的节点嵌入，由第二个GNN在端到端可训练架构中处理。理论上，我们展示了所得到的架构家族的表达能力与WL层次结构一致。我们还发现了HOD-GNN、子图GNN和流行的结构编码方案之间的深层联系。为了计算效率，我们开发了一种用于计算MPNN高阶导数的消息传递算法，该算法利用了图稀疏性和并行性。在流行的图学习基准测试上的评估表明，HOD-GNN在流行的图学习任务上表现强劲。


### 论文摘要

Despite significant advances in Graph Neural Networks (GNNs), their limited expressivity remains a fundamental challenge. Research on GNN expressivity has produced many expressive architectures, leading to architecture hierarchies with models of increasing expressive power. Separately, derivatives of GNNs with respect to node features have been widely studied in the context of the oversquashing and over-smoothing phenomena, GNN explainability, and more. To date, these derivatives remain unexplored as a means to enhance GNN expressivity. In this paper, we show that these derivatives provide a natural way to enhance the expressivity of GNNs. We introduce High-Order Derivative GNN (HOD-GNN), a novel method that enhances the expressivity of Message Passing Neural Networks (MPNNs) by leveraging high-order node derivatives of the base model. These derivatives generate expressive structure-aware node embeddings processed by a second GNN in an end-to-end trainable architecture. Theoretically, we show that the resulting architecture family's expressive power aligns with the WL hierarchy. We also draw deep connections between HOD-GNN, Subgraph GNNs, and popular structural encoding schemes. For computational efficiency, we develop a message-passing algorithm for computing high-order derivatives of MPNNs that exploits graph sparsity and parallelism. Evaluations on popular graph learning benchmarks demonstrate HOD-GNN's strong performance on popular graph learning tasks.

---

## 50. Heterogeneous Graph Representation of Stiffened Panels with Non-Uniform Boundary Conditions and Loads

**论文链接:** [http://arxiv.org/abs/2510.02472v1](http://arxiv.org/abs/2510.02472v1)

**作者:** Yuecheng Cai, Jasmin Jelovica

**发布时间:** 2025-10-02

**备注:** This is a preprint and has been submitted to Engineering with  Computers

### GPT解析

### 总结

本研究提出了一种基于异构图神经网络(HGNNs)的加筋板异构图表示方法，能够有效考虑几何变异性、非均匀边界条件和不同加载场景。通过将结构划分为多个结构单元并使用三种不同类型的节点表示，结合异构图变换器(HGT)，该方法能够准确预测加筋板上的冯·米塞斯应力和位移场。数值测试表明，与同构对应物相比，该方法具有优越性能，能有效捕捉结构行为模式和最大值。

### 背景

代理模型在结构分析和优化中是必不可少的。

### 目的

提出一种能够考虑几何变异性、非均匀边界条件和不同加载场景的加筋板异构图表示方法，使用异构图神经网络(HGNNs)。

### 方法

将结构划分为多个结构单元，如加强筋和它们之间的板，每个单元由三种不同类型的节点表示：几何节点、边界节点和加载节点；通过引入连接节点的局部方向和空间关系来引入边异构性；提出几种具有不同异构程度的异构图表示；将这些表示实现到异构图变换器(HGT)中，以预测加筋板上的冯·米塞斯应力和位移场。

### 主要发现

对承受点载荷的板和由加筋板组成的箱梁在各种载荷条件下的数值测试表明，异构图表示与同构对应物相比表现出优越性能；消融分析评估了图异构性对HGT性能的影响；结果显示对位移和冯·米塞斯应力都具有强预测准确性，能有效捕捉结构行为模式和最大值。

### 结论

所提出的异构图表示方法能够准确预测加筋板的应力和位移，有效捕捉结构行为模式。

### 翻译

代理模型在结构分析和优化中是必不可少的。我们提出了一种加筋板的异构图表示方法，该方法考虑了几何变异性、非均匀边界条件和不同的加载场景，使用异构图神经网络(HGNNs)。结构被划分为多个结构单元，如加强筋和它们之间的板，每个单元由三种不同类型的节点表示：几何节点、边界节点和加载节点。通过引入连接节点的局部方向和空间关系来引入边异构性。提出了几种具有不同异构程度的异构图表示并进行了分析。这些表示被实现到异构图变换器(HGT)中，以预测加筋板上的冯·米塞斯应力和位移场，基于其边界上的载荷和自由度。为了评估我们方法的功效，我们对承受点载荷的板和由加筋板组成的在各种载荷条件下的箱梁进行了数值测试。将异构图表示与同构对应物进行了比较，显示出优越的性能。此外，进行了消融分析以评估图异构性对HGT性能的影响。结果显示对位移和冯·米塞斯应力都具有强预测准确性，有效捕捉了结构行为模式和最大值。


### 论文摘要

Surrogate models are essential in structural analysis and optimization. We propose a heterogeneous graph representation of stiffened panels that accounts for geometrical variability, non-uniform boundary conditions, and diverse loading scenarios, using heterogeneous graph neural networks (HGNNs). The structure is partitioned into multiple structural units, such as stiffeners and the plates between them, with each unit represented by three distinct node types: geometry, boundary, and loading nodes. Edge heterogeneity is introduced by incorporating local orientations and spatial relationships of the connecting nodes. Several heterogeneous graph representations, each with varying degrees of heterogeneity, are proposed and analyzed. These representations are implemented into a heterogeneous graph transformer (HGT) to predict von Mises stress and displacement fields across stiffened panels, based on loading and degrees of freedom at their boundaries. To assess the efficacy of our approach, we conducted numerical tests on panels subjected to patch loads and box beams composed of stiffened panels under various loading conditions. The heterogeneous graph representation was compared with a homogeneous counterpart, demonstrating superior performance. Additionally, an ablation analysis was performed to evaluate the impact of graph heterogeneity on HGT performance. The results show strong predictive accuracy for both displacement and von Mises stress, effectively capturing structural behavior patterns and maximum values.

---

