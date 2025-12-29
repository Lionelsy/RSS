# 今日论文推荐 - 2025-12-29

共 43 篇论文

---

## 1. 论文ID: 2512.22014v1

**错误:** 无法读取论文信息 - [Errno 2] No such file or directory: 'storage/result/2512.22014v1.json'

---

## 2. Knowledge Reasoning of Large Language Models Integrating Graph-Structured Information for Pest and Disease Control in Tobacco

**论文链接:** [http://arxiv.org/abs/2512.21837v1](http://arxiv.org/abs/2512.21837v1)

**作者:** Siyu Li, Chenwei Song, Wan Zhou, Xinyi Liu

**发布时间:** 2025-12-26

### GPT解析

### 总结

本文提出了一种将图结构信息整合到大语言模型中的方法，用于烟草病虫害防治的知识推理。基于GraphRAG框架，该方法通过明确整合领域知识图谱中的结构化信息来增强知识检索和推理能力。

### 背景

烟草病虫害防治需要专业知识和准确推理能力，而传统方法可能难以有效处理复杂的多跳和比较推理场景。

### 目的

提出一种结合图结构信息的大语言模型方法，以提高在烟草病虫害防治领域知识推理的准确性和深度。

### 方法

基于GraphRAG框架构建方法，利用LLM辅助构建烟草病虫害知识图谱，采用Transformer架构作为核心推理模型，使用图神经网络学习节点表示，并以ChatGLM为基础模型，使用LoRA进行参数高效微调。

### 主要发现

该方法在多个评估指标上一致优于基线方法，显著提高了推理的准确性和深度，特别是在复杂的多跳和比较推理场景中表现尤为突出。

### 结论

将图结构信息整合到大语言模型中可以有效增强特定领域知识推理能力，为烟草病虫害防治提供了更准确、更深入的解决方案。

### 翻译

本文提出了一种将图结构信息整合到大语言模型中的方法，用于烟草病虫害防治的知识推理。基于GraphRAG框架，该方法通过明确整合领域知识图谱中的结构化信息来增强知识检索和推理能力。具体来说，首先利用LLM辅助构建烟草病虫害知识图谱，组织疾病、症状、防治方法等关键实体及其关系。基于此图谱，检索相关知识并整合到推理过程中，以支持准确的答案生成。采用Transformer架构作为核心推理模型，同时使用图神经网络学习 expressive 节点表示，捕获知识图谱中的局部和全局关系信息。以ChatGLM为基础的模型作为骨干LLM，并使用LoRA进行微调，实现参数高效的适应性。大量实验结果表明，该方法在多个评估指标上一致优于基线方法，显著提高了推理的准确性和深度，特别是在复杂的多跳和比较推理场景中。


### 论文摘要

This paper proposes a large language model (LLM) approach that integrates graph-structured information for knowledge reasoning in tobacco pest and disease control. Built upon the GraphRAG framework, the proposed method enhances knowledge retrieval and reasoning by explicitly incorporating structured information from a domain-specific knowledge graph. Specifically, LLMs are first leveraged to assist in the construction of a tobacco pest and disease knowledge graph, which organizes key entities such as diseases, symptoms, control methods, and their relationships. Based on this graph, relevant knowledge is retrieved and integrated into the reasoning process to support accurate answer generation. The Transformer architecture is adopted as the core inference model, while a graph neural network (GNN) is employed to learn expressive node representations that capture both local and global relational information within the knowledge graph. A ChatGLM-based model serves as the backbone LLM and is fine-tuned using LoRA to achieve parameter-efficient adaptation. Extensive experimental results demonstrate that the proposed approach consistently outperforms baseline methods across multiple evaluation metrics, significantly improving both the accuracy and depth of reasoning, particularly in complex multi-hop and comparative reasoning scenarios.

---

## 3. Toward Generalizable Surrogate Models for Molecular Dynamics via Graph Neural Networks

**论文链接:** [http://arxiv.org/abs/2512.21822v1](http://arxiv.org/abs/2512.21822v1)

**作者:** Judah Immanuel, Avik Mahata, Aniruddha Maiti

**发布时间:** 2025-12-26

### GPT解析

### 总结

该研究介绍了一种基于图神经网络的分子动力学模拟替代框架，可直接预测原子位移并学习原子系统的演化算子，无需传统方法中的力计算和时间积分步骤。

### 背景

传统分子动力学模拟依赖于重复的力评估和数值时间积分，计算成本较高，需要更高效的替代方法。

### 目的

开发一种计算效率更高且保持准确性的分子动力学模拟替代框架，加速原子尺度模拟。

### 方法

将原子环境表示为图结构，结合消息传递层和注意力机制捕获金属系统中的局部配位和多体相互作用，使用块状铝的经典分子动力学轨迹进行训练。

### 主要发现

替代模型在训练范围内实现了亚埃级别的精度，在短期到中期时间外推中表现稳定，并通过径向分布函数和均方位移趋势验证了结构和动力学保真度。

### 结论

基于图神经网络的替代积分器可作为传统分子动力学的有效补充，在验证范围内提供计算效率更高的原子模拟方案。

### 翻译

我们提出了一种基于图神经网络(GNN)的替代框架，用于分子动力学模拟，可直接预测原子位移并学习原子系统潜在的演化算子。与传统分子动力学不同，传统分子动力学依赖于重复的力评估和数值时间积分，而所提出的替代模型可以在不明确计算力的情况下向前传播原子构型。该方法将原子环境表示为图，并结合消息传递层和注意力机制，以捕获金属系统中的局部配位和多体相互作用。使用块状铝的经典分子动力学轨迹进行训练后，替代模型在训练范围内实现了亚埃级别的精度，并在短期到中期时间范围的外推中表现出稳定行为。通过与参考径向分布函数和均方位移趋势的一致性验证了结构和动力学保真度，证明了该模型在点坐标精度之外保留了关键的物理特征。


### 论文摘要

We present a graph neural network (GNN) based surrogate framework for molecular dynamics simulations that directly predicts atomic displacements and learns the underlying evolution operator of an atomistic system. Unlike conventional molecular dynamics, which relies on repeated force evaluations and numerical time integration, the proposed surrogate model propagates atomic configurations forward in time without explicit force computation. The approach represents atomic environments as graphs and combines message-passing layers with attention mechanisms to capture local coordination and many-body interactions in metallic systems. Trained on classical molecular dynamics trajectories of bulk aluminum, the surrogate achieves sub angstrom level accuracy within the training horizon and exhibits stable behavior during short- to mid-horizon temporal extrapolation. Structural and dynamical fidelity are validated through agreement with reference radial distribution functions and mean squared displacement trends, demonstrating that the model preserves key physical signatures beyond pointwise coordinate accuracy. These results establish GNN-based surrogate integrators as a promising and computationally efficient complement to traditional molecular dynamics for accelerated atomistic simulations within a validated regime.

---

## 4. ALETHEIA: Combating Social Media Influence Campaigns with Graph Neural Networks

**论文链接:** [http://arxiv.org/abs/2512.21391v1](http://arxiv.org/abs/2512.21391v1)

**作者:** Mohammad Hammas Saeed, Isaiah J. King, Howie Huang

**发布时间:** 2025-12-24

### GPT解析

### 总结

本文提出了ALETHEIA系统，用于检测社交媒体网络中的恶意账户并预测其行为，通过图神经网络和时间链接预测机制提高了检测效率和准确性。

### 背景

影响力活动在网络空间中日益增长，成为政策制定者、版主和研究人员关注的重点问题。

### 目的

开发ALETHEIA系统，规范化检测恶意账户，并预测它们在社交媒体网络中的行为。

### 方法

分析Reddit和X平台上的影响力活动，建立基于图形的表示方法，结合拓扑和语言特征，使用图神经网络检测恶意用户，并通过在循环神经网络上堆叠GNN构建时间链接预测机制。

### 主要发现

基于图形的检测方法优于标准特征；ALETHEIA在大规模网络上可扩展，F1分数提高3.7%；时间链接预测机制平均AUC达到96.6%，能预测巨魔间的互动及巨魔对普通用户的影响。

### 结论

利用影响力操作的联网性质（结构信息）对于预测和检测在线空间中的恶意协调活动至关重要。

### 翻译

影响力活动是在网络空间中日益增长的问题。政策制定者、版主和研究人员采取了多种途径来应对这些活动，使在线系统对普通用户更安全。为此，我们的论文提出了ALETHEIA系统，它规范化了此类活动中使用的恶意账户（或称巨魔账户）的检测，并预测它们在社交媒体网络中的行为。我们分析了来自不同国家的Reddit和X平台上的影响力活动，并强调建立在活动图形表示基础上的检测管道，结合拓扑和语言特征，比标准的交互和用户特征有所改进。ALETHEIA使用最先进的图神经网络（GNNs）来检测恶意用户，这些网络可扩展到大规模网络，并在之前使用交互特征的标准分类工作中实现了3.7%的F1分数提升。此外，ALETHEIA采用了一种首次为影响力活动构建的时间链接预测机制，通过在循环神经网络（RNN）上堆叠GNN，可以预测未来巨魔之间以及巨魔与普通用户之间的互动，平均AUC为96.6%。ALETHEIA预测巨魔到巨魔的边缘（TTE）和巨魔到用户的边缘（TUE），这可以帮助识别受到恶意影响力影响的普通用户。总体而言，我们的结果突显了在预测和检测在线空间中的恶意协调活动时，利用影响力操作的联网性质（即结构信息）的重要性。


### 论文摘要

Influence campaigns are a growing concern in the online spaces. Policymakers, moderators and researchers have taken various routes to fight these campaigns and make online systems safer for regular users. To this end, our paper presents ALETHEIA, a system that formalizes the detection of malicious accounts (or troll accounts) used in such operations and forecasts their behaviors within social media networks. We analyze influence campaigns on Reddit and X from different countries and highlight that detection pipelines built over a graph-based representation of campaigns using a mix of topological and linguistic features offer improvement over standard interaction and user features. ALETHEIA uses state-of-the-art Graph Neural Networks (GNNs) for detecting malicious users that can scale to large networks and achieve a 3.7% F1-score improvement over standard classification with interaction features in prior work. Furthermore, ALETHEIA employs a first temporal link prediction mechanism built for influence campaigns by stacking a GNN over a Recurrent Neural Network (RNN), which can predict future troll interactions towards other trolls and regular users with an average AUC of 96.6%. ALETHEIA predicts troll-to-troll edges (TTE) and troll-to-user edges (TUE), which can help identify regular users being affected by malicious influence efforts. Overall, our results highlight the importance of utilizing the networked nature of influence operations (i.e., structural information) when predicting and detecting malicious coordinated activity in online spaces.

---

## 5. SENTINEL: A Multi-Modal Early Detection Framework for Emerging Cyber Threats using Telegram

**论文链接:** [http://arxiv.org/abs/2512.21380v1](http://arxiv.org/abs/2512.21380v1)

**作者:** Mohammad Hammas Saeed, Howie Huang

**发布时间:** 2025-12-24

### GPT解析

### 总结

本研究提出了SENTINEL框架，利用社交媒体信号早期检测网络攻击，通过结合大型语言模型和图神经网络技术，将网络安全讨论与现实世界威胁对齐。

### 背景

网络攻击对现代社会技术系统构成严重威胁，攻击者常通过恶意软件、勒索软件等技术手段实施攻击。传统防御机制多为事后响应而非主动预防，而社交媒体讨论可作为检测此类威胁的可靠指标。

### 目的

开发一种能够利用社交媒体信号早期检测网络攻击的框架，通过分析社交媒体上的讨论来识别潜在网络安全威胁。

### 方法

SENTINEL框架利用多模态信号，结合大型语言模型进行语言建模和图神经网络进行协调标记，将网络安全讨论与现实世界网络攻击对齐。研究使用了来自16个与网络安全和开源情报相关的Telegram公共频道的数据，涵盖365k条消息。

### 主要发现

社交媒体讨论涉及围绕网络威胁的积极对话，使用SENTINEL可将信号与现实世界威胁对齐，F1分数达到0.89。

### 结论

利用语言和网络信号预测在线威胁具有重要价值，SENTINEL框架为网络安全威胁的早期检测提供了有效方法。

### 翻译

网络攻击对现代社会技术系统构成严重威胁，通常导致严重的技术和社会后果。攻击者通常通过恶意软件、勒索软件或其他形式的技术利用方法来系统和基础设施。大多数传统的应对这些威胁的机制依赖于事后检测和缓解策略，在网络事件发生后才响应，而不是主动预防。最近的趋势显示，社交媒体讨论可以作为检测此类威胁的可靠指标。恶意行为者经常利用在线平台分发攻击工具、分享攻击知识和协调。专家们也经常预测正在进行的攻击，并在在线空间讨论潜在的漏洞。在这项工作中，我们提出了SENTINEL，一个利用社交媒体信号早期检测网络攻击的框架。SENTINEL利用多模态信号将网络安全讨论与现实世界网络攻击对齐，即通过大型语言模型进行语言建模，通过图神经网络进行协调标记。我们使用了来自16个与网络安全和开源情报相关的公共Telegram频道的数据，涵盖了365k条消息。我们强调，社交媒体讨论涉及围绕网络威胁的积极对话，并利用SENTINEL将信号与现实世界威胁对齐，F1分数为0.89。我们的工作强调了利用语言和网络信号预测在线威胁的重要性。


### 论文摘要

Cyberattacks pose a serious threat to modern sociotechnical systems, often resulting in severe technical and societal consequences. Attackers commonly target systems and infrastructure through methods such as malware, ransomware, or other forms of technical exploitation. Most traditional mechanisms to counter these threats rely on post-hoc detection and mitigation strategies, responding to cyber incidents only after they occur rather than preventing them proactively. Recent trends reveal social media discussions can serve as reliable indicators for detecting such threats. Malicious actors often exploit online platforms to distribute attack tools, share attack knowledge and coordinate. Experts too, often predict ongoing attacks and discuss potential breaches in online spaces. In this work, we present SENTINEL, a framework that leverages social media signals for early detection of cyber attacks. SENTINEL aligns cybersecurity discussions to realworld cyber attacks leveraging multi modal signals, i.e., combining language modeling through large language models and coordination markers through graph neural networks. We use data from 16 public channels on Telegram related to cybersecurity and open source intelligence (OSINT) that span 365k messages. We highlight that social media discussions involve active dialogue around cyber threats and leverage SENTINEL to align the signals to real-world threats with an F1 of 0.89. Our work highlights the importance of leveraging language and network signals in predicting online threats.

---

## 6. AstraNav-World: World Model for Foresight Control and Consistency

**论文链接:** [http://arxiv.org/abs/2512.21714v1](http://arxiv.org/abs/2512.21714v1)

**作者:** Junjun Hu, Jintao Chen, Haochen Bai, Minghua Luo, Shichao Xie, Ziyi Chen, Fei Liu, Zedong Chu, Xinda Xue, Botao Ren, Xiaolong Wu, Mu Xu, Shanghang Zhang

**发布时间:** 2025-12-25

### GPT解析

### 总结

论文提出了AstraNav-World，一种在开放动态环境中进行具身导航的端到端世界模型，通过统一的概率框架联合推理未来视觉状态和动作序列。

### 背景

在开放、动态环境中进行具身导航需要准确预测世界如何随时间演变以及行动如何展开。

### 目的

开发一个能够准确预测世界演变和行动展开的具身导航系统，提高在开放环境中的导航能力。

### 方法

AstraNav-World框架集成了基于扩散的视频生成器和视觉语言策略，实现预测场景和计划动作的同步更新。训练优化两个互补目标：生成条件动作的多步视觉预测和基于预测视觉导出轨迹。这种双向约束使视觉预测可执行，使决策基于物理一致且与任务相关的未来。

### 主要发现

在各种具身导航基准测试中，AstraNav-World提高了轨迹准确性和成功率。消融研究证实了紧密视觉-动作耦合和统一训练的必要性。真实世界测试显示其具有卓越的零样本能力，无需微调即可适应新场景。

### 结论

AstraNav-World捕捉了可转移的空间理解和规划相关的导航动态，而非仅拟合特定数据分布。通过在单个生成模型中统一远见视觉和控制，系统更接近于在开放真实世界环境中可靠、可解释和通用的具身智能体。

### 翻译

在开放、动态环境中的具身导航需要准确预测世界将如何演变以及行动将如何随时间展开。我们提出了AstraNav-World，一种端到端世界模型，在统一的概率框架内共同推理未来的视觉状态和动作序列。我们的框架将基于扩散的视频生成器与视觉语言策略相结合，使预测场景和计划动作能够同步更新。训练优化了两个互补目标：生成条件动作的多步视觉预测和基于预测视觉导出轨迹。这种双向约束使视觉预测可执行，并使决策保持基于物理一致且与任务相关的未来，减轻了解耦的'先想象后规划'管道中常见的累积错误。在各种具身导航基准测试中的实验显示提高了轨迹准确性和成功率。消融研究证实了紧密视觉-动作耦合和统一训练的必要性，移除任何一个分支都会降低预测质量和策略可靠性。在真实世界测试中，AstraNav-World表现出卓越的零样本能力，无需任何真实世界微调即可适应前所未见的场景。这些结果表明，AstraNav-World捕捉了可转移的空间理解和规划相关的导航动态，而不仅仅是模拟特定的数据分布。总体而言，通过在单个生成模型中统一远见视觉和控制，我们更接近于在开放真实世界环境中可靠、可解释和通用的具身智能体。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': "论文主要解决具身导航中的物理一致性和因果一致性问题。当前导航系统通常采用'先想象后行动'的松耦合范式，导致物理不确定性被放大、因果模糊性增加，以及误差随时间累积，最终影响全局规划效果。这个问题在现实中非常重要，因为导航是具身智能的核心能力，能让代理在复杂、未知的真实环境中自主行动；缺乏对物理规律和时间动态的建模是导航失败的主要原因；小的预测偏差会随时间累积，削弱全局规划的有效性；在开放、动态环境中，准确的预测能力对于可靠导航至关重要。", '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': "作者首先分析了现有'envision-then-plan'范式的局限性，识别出需要同时推进'想象未来'和'规划未来'两个能力管道。他们借鉴了多个现有工作：世界模型（如LaDi-WM、MoWM）用于预测未来状态；视觉-语言-动作模型（如WorldVLA、CoT-VLA）但改进其松耦合问题；基于扩散的视频生成模型（如Wan2.2-TI2V-5B）提供高质量视觉先验；视觉-语言模型（如Qwen2.5-VL-3B）处理指令和视觉历史。设计上，作者创建了一个统一的生成框架，通过双向约束和同步滚动进行联合优化，设计了3D-RoPE重新排列策略处理多视图输入，提出了稀疏预见调度平衡实时性，实现了两种策略头，并采用两阶段训练策略。", '这个方法的核心思想是什么？整体实现流程是怎样的？': "核心思想是创建一个统一的生成世界模型，将'想象未来'和'规划未来'紧密绑定在单一概率框架内，通过双向约束和同步滚动确保视觉预测与动作序列之间的一致性。整体流程：1)接收自然语言指令和历史视觉观察；2)VLM规划器处理这些输入并生成视觉-语言嵌入；3)视频生成器基于VLM嵌入预测未来视觉帧；4)动作策略头（Action Former或Diffusion Policy）生成动作序列；5)采用两阶段训练（组件预训练和联合微调）；6)推理时使用稀疏预见调度平衡实时性和计算效率；7)输出预测的未来视觉帧和对应的动作序列，确保两者一致性。", '论文的关键创新点有哪些？相比之前的工作，有什么不同？': "关键创新点：1)统一的生成框架，将想象和规划紧密绑定；2)VLM作为中央规划器提供高级指导；3)3D-RoPE重新排列策略处理多视图输入；4)多模态融合交叉注意力(MMFCA)实现双向信息流；5)稀疏预见调度(SFS)平衡实时性和计算效率；6)双策略头设计提供确定性/概率性动作生成。相比之前工作：不同于传统'envision-then-plan'方法的松耦合，AstraNav-World实现紧耦合减少误差累积；不同于WorldVLA和CoT-VLA仍遵循松耦合范式，AstraNav-World将视频生成和动作生成统一在单一生成过程中；不同于传统世界模型只关注短期预测，AstraNav-World结合VLM长程理解实现长期规划；通过SFS策略在保持性能的同时大幅减少计算开销。", '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'AstraNav-World通过统一视觉预测与动作规划的生成框架，实现了具身导航中物理一致性和因果一致性的显著提升，在多个导航基准测试中取得最先进性能，并展现出卓越的零样本迁移能力。'}


### 论文摘要

Embodied navigation in open, dynamic environments demands accurate foresight of how the world will evolve and how actions will unfold over time. We propose AstraNav-World, an end-to-end world model that jointly reasons about future visual states and action sequences within a unified probabilistic framework. Our framework integrates a diffusion-based video generator with a vision-language policy, enabling synchronized rollouts where predicted scenes and planned actions are updated simultaneously. Training optimizes two complementary objectives: generating action-conditioned multi-step visual predictions and deriving trajectories conditioned on those predicted visuals. This bidirectional constraint makes visual predictions executable and keeps decisions grounded in physically consistent, task-relevant futures, mitigating cumulative errors common in decoupled "envision-then-plan" pipelines. Experiments across diverse embodied navigation benchmarks show improved trajectory accuracy and higher success rates. Ablations confirm the necessity of tight vision-action coupling and unified training, with either branch removal degrading both prediction quality and policy reliability. In real-world testing, AstraNav-World demonstrated exceptional zero-shot capabilities, adapting to previously unseen scenarios without any real-world fine-tuning. These results suggest that AstraNav-World captures transferable spatial understanding and planning-relevant navigation dynamics, rather than merely overfitting to simulation-specific data distribution. Overall, by unifying foresight vision and control within a single generative model, we move closer to reliable, interpretable, and general-purpose embodied agents that operate robustly in open-ended real-world settings.

---

## 7. BertsWin: Resolving Topological Sparsity in 3D Masked Autoencoders via Component-Balanced Structural Optimization

**论文链接:** [http://arxiv.org/abs/2512.21769v1](http://arxiv.org/abs/2512.21769v1)

**作者:** Evgeny Alves Limarenko, Anastasiia Studenikina

**发布时间:** 2025-12-25

**备注:** Code available at https://github.com/AlevLab-dev/BertsWinMAE and https://github.com/AlevLab-dev/GCond. Zenodo repository (DOI: 10.5281/zenodo.17916932) contains source images, training logs, trained models, and code

### GPT解析

### 总结

本文提出了一种名为BertsWin的新型混合架构，结合了完整的BERT风格令牌屏蔽和Swin Transformer窗口，用于增强3D医学图像自监督学习中的空间上下文学习，解决了标准方法在处理3D体积图像时的局限性。

### 背景

自监督学习和Vision Transformers在2D医学成像领域表现优异，但在3D体积图像上的应用存在困难。标准的Masked Autoencoders作为2D领域的先进解决方案，在预训练过程中难以捕捉三维空间关系，尤其是当75%的令牌被丢弃时。

### 目的

开发一种能够有效处理3D体积图像的架构，保留空间拓扑结构，同时提高计算效率和收敛速度，用于医学图像处理中的自监督学习任务。

### 方法

提出BertsWin混合架构，结合：1)完整的BERT风格令牌屏蔽使用Swin Transformer窗口；2)引入完整3D令牌网格保留空间拓扑；3)使用单层局部Swin窗口平滑ViT的二次复杂度；4)引入结构优先损失函数；5)使用GradientConductor优化器。

### 主要发现

BertsWin通过保持完整三维空间拓扑，比标准ViT-MAE基线加速语义收敛5.8倍；结合GradientConductor优化器，达到最先进重建保真度所需的训练周期减少15倍(44对660)；实现加速的同时没有计算惩罚；在标准输入分辨率下保持与稀疏ViT基线理论上的FLOP对等，总计算资源显著减少。

### 结论

BertsWin架构有效解决了3D医学图像处理中的自监督学习挑战，通过保留空间拓扑结构和优化计算效率，显著提高了训练速度和模型性能，同时保持了与现有方法相当的计算资源需求。

### 翻译

自监督学习和Vision Transformers方法在2D医学成像领域的应用显示出有希望的结果，但这些方法在3D体积图像上的使用充满困难。作为2D最先进解决方案的标准Masked Autoencoders，在预训练过程中丢弃75%的令牌时，难以捕捉三维空间关系。我们提出了BertsWin，这是一种混合架构，结合使用Swin Transformer窗口的完整BERT风格令牌屏蔽，以在自监督预训练期间增强3D中的空间上下文学习。与仅处理可见区域的传统MAE不同，BertsWin引入了完整的3D令牌网格（屏蔽和可见的），保留了空间拓扑。并且为了平滑ViT的二次复杂度，使用了单层局部Swin窗口。我们引入了一种结构优先损失函数，并评估了颞下颌关节锥束计算断层扫描的结果。随后的评估包括3D CT扫描上的TMJ分割。我们证明，BertsWin架构通过保持完整的三维空间拓扑，本质上比标准ViT-MAE基线加速了语义收敛5.8倍。此外，当我们提出的GradientConductor优化器结合使用时，完整的BertsWin框架达到最先进重建保真度所需的训练周期减少了15倍（44对660）。分析显示，BertsWin实现了这种加速，而没有通常与密集体积处理相关的计算惩罚。在标准输入分辨率下，该架构保持与稀疏ViT基线理论上的FLOP对等，由于更快收敛，导致总计算资源显著减少。


### 论文摘要

The application of self-supervised learning (SSL) and Vision Transformers (ViTs) approaches demonstrates promising results in the field of 2D medical imaging, but the use of these methods on 3D volumetric images is fraught with difficulties. Standard Masked Autoencoders (MAE), which are state-of-the-art solution for 2D, have a hard time capturing three-dimensional spatial relationships, especially when 75% of tokens are discarded during pre-training. We propose BertsWin, a hybrid architecture combining full BERT-style token masking using Swin Transformer windows, to enhance spatial context learning in 3D during SSL pre-training. Unlike the classic MAE, which processes only visible areas, BertsWin introduces a complete 3D grid of tokens (masked and visible), preserving the spatial topology. And to smooth out the quadratic complexity of ViT, single-level local Swin windows are used. We introduce a structural priority loss function and evaluate the results of cone beam computed tomography of the temporomandibular joints. The subsequent assessment includes TMJ segmentation on 3D CT scans. We demonstrate that the BertsWin architecture, by maintaining a complete three-dimensional spatial topology, inherently accelerates semantic convergence by a factor of 5.8x compared to standard ViT-MAE baselines. Furthermore, when coupled with our proposed GradientConductor optimizer, the full BertsWin framework achieves a 15-fold reduction in training epochs (44 vs 660) required to reach state-of-the-art reconstruction fidelity. Analysis reveals that BertsWin achieves this acceleration without the computational penalty typically associated with dense volumetric processing. At canonical input resolutions, the architecture maintains theoretical FLOP parity with sparse ViT baselines, resulting in a significant net reduction in total computational resources due to faster convergence.

---

## 8. 论文ID: 2512.21544v1

**错误:** 无法读取论文信息 - [Errno 2] No such file or directory: 'storage/result/2512.21544v1.json'

---

## 9. Global-Graph Guided and Local-Graph Weighted Contrastive Learning for Unified Clustering on Incomplete and Noise Multi-View Data

**论文链接:** [http://arxiv.org/abs/2512.21516v1](http://arxiv.org/abs/2512.21516v1)

**作者:** Hongqing He, Jie Xu, Wenyuan Yang, Yonghua Zhu, Guoqiu Wen, Xiaofeng Zhu

**发布时间:** 2025-12-25

### GPT解析

### 总结

本文提出了一种统一的基于对比学习的多视图聚类框架，用于解决不完整和噪声多视图数据中的聚类问题。该方法通过全局图引导和局部图加权两种对比学习策略，分别处理稀疏配对和错误配对问题，在不进行数据填补的情况下取得了优越的聚类性能。

### 背景

对比学习在多视图聚类中发挥着重要作用，用于探索不同视图间的互补信息。然而，现实世界中的多视图数据常存在不完整或噪声问题，导致稀疏配对样本或错误配对样本，这严重影响了基于对比学习的多视图聚类方法的有效性。

### 目的

提出一个统一的基于对比学习的多视图聚类框架，以增强在不完整和噪声多视图数据上的聚类效果，解决稀疏配对和错误配对问题。

### 方法

首先，设计全局图引导的对比学习，让所有视图样本构建全局视图亲和图，形成新的样本对以充分探索互补信息；其次，提出局部图加权的对比学习，利用局部邻居生成成对权重，自适应地加强或削弱成对对比学习。该方法无需数据填补，可集成到统一的框架中。

### 主要发现

在不完整和噪声设置的多视图数据上进行了大量实验，结果表明该方法与最先进的方法相比取得了优越的性能。

### 结论

所提出的全局-局部图引导对比学习框架能够有效处理不完整和噪声多视图数据中的聚类问题，通过解决稀疏配对和错误配对挑战，显著提升了多视图聚类的效果。

### 翻译

最近，对比学习在探索多视图聚类的互补信息方面发挥着重要作用，并引起了越来越多的关注。然而，现实世界中的多视图数据存在不完整或噪声问题，导致稀疏配对样本或错误配对样本，这严重挑战了基于对比学习的多视图聚类的有效性。也就是说，稀疏配对问题阻碍了多视图聚类提取足够的互补信息，而错误配对问题导致对比学习向错误方向优化模型。为解决这些问题，我们提出了一个统一的基于对比学习的多视图聚类框架，以增强在不完整和噪声多视图数据上的聚类效果。首先，为了克服稀疏配对问题，我们设计了全局图引导的对比学习，其中所有视图样本构建全局视图亲和图，形成新的样本对以充分探索互补信息。其次，为了减轻错误配对问题，我们提出了局部图加权的对比学习，利用局部邻居生成成对权重，自适应地加强或削弱成对对比学习。我们的方法无需数据填补，可以集成到统一的全局-局部图引导对比学习框架中。在不完整和噪声设置的多视图数据上的大量实验表明，与最先进的方法相比，我们的方法取得了优越的性能。


### 论文摘要

Recently, contrastive learning (CL) plays an important role in exploring complementary information for multi-view clustering (MVC) and has attracted increasing attention. Nevertheless, real-world multi-view data suffer from data incompleteness or noise, resulting in rare-paired samples or mis-paired samples which significantly challenges the effectiveness of CL-based MVC. That is, rare-paired issue prevents MVC from extracting sufficient multi-view complementary information, and mis-paired issue causes contrastive learning to optimize the model in the wrong direction. To address these issues, we propose a unified CL-based MVC framework for enhancing clustering effectiveness on incomplete and noise multi-view data. First, to overcome the rare-paired issue, we design a global-graph guided contrastive learning, where all view samples construct a global-view affinity graph to form new sample pairs for fully exploring complementary information. Second, to mitigate the mis-paired issue, we propose a local-graph weighted contrastive learning, which leverages local neighbors to generate pair-wise weights to adaptively strength or weaken the pair-wise contrastive learning. Our method is imputation-free and can be integrated into a unified global-local graph-guided contrastive learning framework. Extensive experiments on both incomplete and noise settings of multi-view data demonstrate that our method achieves superior performance compared with state-of-the-art approaches.

---

## 10. UniTacHand: Unified Spatio-Tactile Representation for Human to Robotic Hand Skill Transfer

**论文链接:** [http://arxiv.org/abs/2512.21233v2](http://arxiv.org/abs/2512.21233v2)

**作者:** Chi Zhang, Penglin Cai, Haoqi Yuan, Chaoyi Xu, Zongqing Lu

**发布时间:** 2025-12-24

### GPT解析

### 总结

该研究提出了一种使用触觉手套收集人类操作数据的方法，并通过UniTacHand统一表示解决了人类与机器人触觉数据不匹配问题，实现了高效的触觉策略迁移。

### 背景

触觉感知对机器人手实现人类级灵巧操作至关重要，特别是在视觉遮挡场景中。然而，大规模真实世界机器人触觉数据收集困难，限制了其应用。

### 目的

提出使用触觉手套收集低成本的人类操作数据，用于基于触觉的机器人策略学习，并解决人类与机器人触觉数据不匹配问题，实现从人类到机器人的策略迁移。

### 方法

提出UniTacHand统一表示方法，将人类手和机器人手的触觉信号投影到MANO手模型的形态一致的2D表面空间上，并引入对比学习方法将它们对齐到统一的潜在空间，仅使用10分钟的配对数据进行训练。

### 主要发现

实现了从人类到真实机器人的零样本基于触觉的策略迁移，可推广到预训练数据中未见过的物体；通过UniTacHand在混合数据上进行共同训练，比仅使用机器人数据获得更好的性能和数据效率。

### 结论

UniTacHand为基于触觉的灵巧手学习提供了通用、可扩展和数据高效的路径。

### 翻译

触觉感知对机器人手实现人类级灵巧操作至关重要，特别是在视觉遮挡场景中。然而，大规模真实世界机器人触觉数据收集的困难常常限制了其应用。在本研究中，我们提出使用触觉手套收集低成本的人类操作数据，用于基于触觉的机器人策略学习。人类与机器人触觉数据之间的不匹配使得将从人类数据中学到的策略转移到机器人上具有挑战性。为了弥合这一差距，我们提出了UniTacHand，一种统一表示方法，用于对齐灵巧手捕获的机器人触觉信息与手套获取的人类手触觉。首先，我们将人类手和机器人手的触觉信号投影到MANO手模型的形态一致的2D表面空间上。这种统一标准化了异构数据结构，并内在地将触觉信号嵌入空间上下文。然后，我们引入了一种对比学习方法，将它们对齐到统一的潜在空间，仅使用我们数据收集系统中10分钟的配对数据进行训练。我们的方法实现了从人类到真实机器人的零样本基于触觉的策略迁移，可推广到预训练数据中未见过的物体。我们还证明，通过UniTacHand在混合数据（包括人类和机器人演示）上进行共同训练，比仅使用机器人数据能获得更好的性能和数据效率。UniTacHand为基于触觉的灵巧手学习通用、可扩展和数据高效的路径铺平了道路。


### 论文摘要

Tactile sensing is crucial for robotic hands to achieve human-level dexterous manipulation, especially in scenarios with visual occlusion. However, its application is often hindered by the difficulty of collecting large-scale real-world robotic tactile data. In this study, we propose to collect low-cost human manipulation data using haptic gloves for tactile-based robotic policy learning. The misalignment between human and robotic tactile data makes it challenging to transfer policies learned from human data to robots. To bridge this gap, we propose UniTacHand, a unified representation to align robotic tactile information captured by dexterous hands with human hand touch obtained from gloves. First, we project tactile signals from both human hands and robotic hands onto a morphologically consistent 2D surface space of the MANO hand model. This unification standardizes the heterogeneous data structures and inherently embeds the tactile signals with spatial context. Then, we introduce a contrastive learning method to align them into a unified latent space, trained on only 10 minutes of paired data from our data collection system. Our approach enables zero-shot tactile-based policy transfer from humans to a real robot, generalizing to objects unseen in the pre-training data. We also demonstrate that co-training on mixed data, including both human and robotic demonstrations via UniTacHand, yields better performance and data efficiency compared with using only robotic data. UniTacHand paves a path toward general, scalable, and data-efficient learning for tactile-based dexterous hands.

---

## 11. Scene-VLM: Multimodal Video Scene Segmentation via Vision-Language Models

**论文链接:** [http://arxiv.org/abs/2512.21778v1](http://arxiv.org/abs/2512.21778v1)

**作者:** Nimrod Berman, Adam Botach, Emanuel Ben-Baruch, Shunit Haviv Hakimi, Asaf Gendler, Ilan Naiman, Erez Yosef, Igor Kviatkovsky

**发布时间:** 2025-12-25

### GPT解析

### 总结

Scene-VLM是一个创新的视觉语言模型框架，用于视频场景分割，它联合处理视觉和文本线索，生成具有因果依赖关系的预测，并引入了上下文焦点窗口机制，同时能够提取置信度分数并生成自然语言解释，在标准基准测试中取得了最先进的表现。

### 背景

现有的基于编码器的方法在视频场景分割中存在局限性，包括以视觉为中心的偏见、孤立处理每个镜头而不利用序列依赖关系、缺乏叙事理解和可解释性。

### 目的

提出Scene-VLM，第一个针对视频场景分割进行微调的视觉语言模型(VLM)框架，以解决现有方法的局限性。

### 方法

Scene-VLM联合处理视觉和文本线索（包括帧、转录和可选元数据），实现跨连续镜头的多模态推理。模型顺序生成预测，镜头间存在因果依赖关系，并引入上下文焦点窗口机制确保每个镜头级别决策有充分的时序上下文。此外，提出一种从VLM的标记级logits中提取置信度分数的方案，实现可控制的精度-召回权衡。还展示了模型可以通过最少的有针对性监督，对边界决策生成连贯的自然语言解释。

### 主要发现

Scene-VLM在标准场景分割基准测试中取得了最先进的表现。在MovieNet上，相比之前领先的方法，Scene-VLM实现了+6 AP和+13.7 F1的显著提升。

### 结论

Scene-VLM通过整合视觉和语言信息，有效解决了现有视频场景分割方法的局限性，提供了更好的性能、可解释性和控制能力。

### 翻译

将长格式视频分割成语义连贯的场景是大规模视频理解中的基本任务。现有的基于编码器的方法受视觉中心偏见的限制，孤立地分类每个镜头而不利用序列依赖关系，并且缺乏叙事理解和可解释性。在本文中，我们提出了Scene-VLM，这是第一个针对视频场景分割进行微调的视觉语言模型(VLM)框架。Scene-VLM联合处理视觉和文本线索，包括帧、转录和可选元数据，以实现跨连续镜头的多模态推理。模型顺序生成预测，镜头间存在因果依赖关系，并引入上下文焦点窗口机制，确保每个镜头级别决策有充分的时序上下文。此外，我们提出了一种从VLM的标记级logits中提取置信度分数的方案，实现了以往仅限于基于编码器方法的可控精度-召回权衡。此外，我们证明了我们的模型可以通过最少的有针对性监督，对边界决策生成连贯的自然语言解释。我们的方法在标准场景分割基准测试中取得了最先进的性能。例如，在MovieNet上，Scene-VLM比之前领先的方法实现了+6 AP和+13.7 F1的显著提升。


### 论文摘要

Segmenting long-form videos into semantically coherent scenes is a fundamental task in large-scale video understanding. Existing encoder-based methods are limited by visual-centric biases, classify each shot in isolation without leveraging sequential dependencies, and lack both narrative understanding and explainability. In this paper, we present Scene-VLM, the first fine-tuned vision-language model (VLM) framework for video scene segmentation. Scene-VLM jointly processes visual and textual cues including frames, transcriptions, and optional metadata to enable multimodal reasoning across consecutive shots. The model generates predictions sequentially with causal dependencies among shots and introduces a context-focus window mechanism to ensure sufficient temporal context for each shot-level decision. In addition, we propose a scheme to extract confidence scores from the token-level logits of the VLM, enabling controllable precision-recall trade-offs that were previously limited to encoder-based methods. Furthermore, we demonstrate that our model can be aligned to generate coherent natural-language rationales for its boundary decisions through minimal targeted supervision. Our approach achieves state-of-the-art performance on standard scene segmentation benchmarks. On MovieNet, for example, Scene-VLM yields significant improvements of +6 AP and +13.7 F1 over the previous leading method.

---

## 12. Knot Forcing: Taming Autoregressive Video Diffusion Models for Real-time Infinite Interactive Portrait Animation

**论文链接:** [http://arxiv.org/abs/2512.21734v1](http://arxiv.org/abs/2512.21734v1)

**作者:** Steven Xiao, XIndi Zhang, Dechao Meng, Qi Wang, Peng Zhang, Bang Zhang

**发布时间:** 2025-12-25

### GPT解析

### 总结

本文提出了一种名为'Knot Forcing'的新型流式框架，用于实时人像动画，解决了现有方法在视觉质量、时间连贯性和实时性方面的挑战。

### 背景

实时人像动画对虚拟助手和实时头像等交互应用至关重要，需要高视觉保真度、时间连贯性、超低延迟和响应式控制。基于扩散的模型质量高但非因果，阻碍流式部署；因果自回归视频生成方法存在误差累积和长期一致性问题。

### 目的

开发一种能够实现高保真度、时间连贯性、交互式人像动画的流式框架，解决现有方法在误差累积、块间运动不连续性和长期一致性下降方面的问题。

### 方法

Knot Forcing框架包含三个关键设计：(1)块状生成策略，通过缓存参考图像KV状态保持全局身份特征，使用滑动窗口注意力进行局部时间建模；(2)时间节点模块，重叠相邻块并通过图像到视频条件传递空间时间线索，平滑块间运动过渡；(3)'提前运行'机制，动态更新参考帧时间坐标，使其语义内容领先于当前帧以支持长期连贯性。

### 主要发现

Knot Forcing能够在无限序列上实现高保真度、时间连贯性和交互式人像动画，在消费级GPU上达到实时性能并保持强大的视觉稳定性。

### 结论

该框架成功解决了实时人像动画中的关键挑战，实现了高质量、时间连贯的动画效果，同时保持实时性能，为交互应用提供了有效解决方案。

### 翻译

实时人像动画对于虚拟助手和实时头像等交互应用至关重要，需要高视觉保真度、时间连贯性、超低延迟以及从参考图像和驱动信号等动态输入获得响应式控制。虽然基于扩散的模型能实现强质量，但其非因果性质阻碍了流式部署。因果自回归视频生成方法能够高效的逐帧生成，但存在误差累积、块间运动不连续和长期一致性下降的问题。在这项工作中，我们提出了一种名为Knot Forcing的新型流式框架用于实时人像动画，通过三个关键设计解决这些挑战：(1)块状生成策略，通过缓存参考图像的KV状态保持全局身份特征，并使用滑动窗口注意力进行局部时间建模；(2)时间节点模块，重叠相邻块并通过图像到视频的条件传递空间时间线索，平滑块间运动过渡；(3)'提前运行'机制，在推理过程中动态更新参考帧的时间坐标，使其语义内容领先于当前展开帧以支持长期连贯性。Knot Forcing能够在无限序列上实现高保真度、时间连贯和交互式的人像动画，在消费级GPU上实现实时性能并保持强大的视觉稳定性。


### 论文摘要

Real-time portrait animation is essential for interactive applications such as virtual assistants and live avatars, requiring high visual fidelity, temporal coherence, ultra-low latency, and responsive control from dynamic inputs like reference images and driving signals. While diffusion-based models achieve strong quality, their non-causal nature hinders streaming deployment. Causal autoregressive video generation approaches enable efficient frame-by-frame generation but suffer from error accumulation, motion discontinuities at chunk boundaries, and degraded long-term consistency. In this work, we present a novel streaming framework named Knot Forcing for real-time portrait animation that addresses these challenges through three key designs: (1) a chunk-wise generation strategy with global identity preservation via cached KV states of the reference image and local temporal modeling using sliding window attention; (2) a temporal knot module that overlaps adjacent chunks and propagates spatio-temporal cues via image-to-video conditioning to smooth inter-chunk motion transitions; and (3) A "running ahead" mechanism that dynamically updates the reference frame's temporal coordinate during inference, keeping its semantic context ahead of the current rollout frame to support long-term coherence. Knot Forcing enables high-fidelity, temporally consistent, and interactive portrait animation over infinite sequences, achieving real-time performance with strong visual stability on consumer-grade GPUs.

---

## 13. 论文ID: 2512.21641v1

**错误:** 无法读取论文信息 - [Errno 2] No such file or directory: 'storage/result/2512.21641v1.json'

---

## 14. World-Coordinate Human Motion Retargeting via SAM 3D Body

**论文链接:** [http://arxiv.org/abs/2512.21573v1](http://arxiv.org/abs/2512.21573v1)

**作者:** Zhangzheng Tum, Kailun Su, Shaolong Zhu, Yukun Zheng

**发布时间:** 2025-12-25

### GPT解析

### 总结

本文提出了一种从单目视频中恢复世界坐标系人体运动并重定向到人形机器人的轻量级框架，通过冻结感知主干和机器人友好中间表示，实现了稳定的轨迹和可靠的重定向效果。

### 背景

从单目视频中恢复世界坐标系人体运动并重定向到人形机器人对于具身智能和机器人技术具有重要意义，但现有方法通常需要复杂的SLAM流程或重型时序模型。

### 目的

开发一个轻量级、面向工程的框架，避免复杂SLAM流程或重型时序模型，从单目视频中生成机器人可直接使用的运动数据。

### 方法

使用SAM 3D Body作为冻结感知主干，Momentum HumanRig作为机器人友好中间表示；锁定被追踪对象的身份和骨骼尺度参数确保时间一致性；通过低维MHR潜空间的滑动窗口优化平滑预测；使用可微分软足地接触模型和接触感知全局优化恢复物理合理的根轨迹；采用运动学感知的两阶段逆运动学管道将运动重定向到Unitree G1人形机器人。

### 主要发现

在真实单目视频上的测试表明，该方法具有稳定的世界轨迹和可靠的机器人重定向效果。

### 结论

具有轻量级物理约束的结构化人体表示可以从单目输入中产生机器人就绪的运动。

### 翻译

从单目视频中恢复世界坐标系人体运动并重定向到人形机器人对于具身智能和机器人技术具有重要意义。为避免复杂的SLAM流程或重型时序模型，我们提出了一种轻量级、面向工程的框架，利用SAM 3D Body (3DB)作为冻结感知主干，并使用Momentum HumanRig (MHR)表示作为机器人友好的中间表示。我们的方法(i)锁定每个被追踪对象的身份和骨骼尺度参数，强制执行时间一致的骨骼长度；(ii)通过在低维MHR潜空间中的高效滑动窗口优化来平滑每帧预测；(iii)使用可微分的软足地接触模型和接触感知的全局优化来恢复物理上合理的全局根轨迹。最后，我们使用运动学感知的两阶段逆运动学管道将重建的运动重定向到Unitree G1人形机器人。真实单目视频上的结果表明，我们的方法具有稳定的世界轨迹和可靠的机器人重定向效果，表明具有轻量级物理约束的结构化人体表示可以从单目输入中产生机器人就绪的运动。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决如何从单目视频中恢复世界坐标系下的人体运动并将其重定向到人形机器人上的问题。这个问题很重要，因为单目视频成本低且易于获取，而将人体运动转移到人形机器人对具身智能和机器人应用有重大意义。现有方法通常依赖复杂的SLAM系统或重型时序模型，工程开销大，限制了在实际机器人重定向场景中的应用。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者观察到，虽然单目3D人体重建已取得进展，但这些方法主要在相机坐标系中操作，关注视觉保真度而非全局运动。因此，他们探索如何将结构化人体表示与轻量级物理约束结合，生成机器人可用的运动。他们借鉴了SAM 3D Body作为感知骨干，Momentum Human Rig作为中间表示，以及检测-跟踪模块和滑动窗口优化等技术。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是利用结构化人体表示结合轻量物理约束，从单目视频中生成机器人可用运动。具体流程：1)视频预处理和跟踪：用3DB处理每帧，用卡尔曼滤波跟踪主体；2)轨迹级一致性处理：锁定身份和骨骼尺度参数，在MHR潜在空间中进行滑动窗口平滑；3)接触感知的全局根优化：使用软接触模型估计物理合理的根轨迹；4)重定向到机器人：通过两阶段逆运动学将运动映射到Unitree G1机器人。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)面向工程的3DB世界坐标系运动重定向管道；2)轨迹级身份和骨骼尺度锁定策略；3)接触感知的全局根优化方案；4)在实际人形机器人上验证的完整系统。相比之前工作，该方法避免了复杂SLAM或重型时序模型，使用MHR保持骨骼长度一致性，优先考虑鲁棒性和实际部署性而非绝对视觉准确性。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '这篇论文提出了一种轻量级、面向工程的管道，通过结构化人体表示和最小物理约束，从单目视频中生成机器人可用的世界坐标系人体运动，并在实际人形机器人上成功展示了其有效性。'}


### 论文摘要

Recovering world-coordinate human motion from monocular videos with humanoid robot retargeting is significant for embodied intelligence and robotics. To avoid complex SLAM pipelines or heavy temporal models, we propose a lightweight, engineering-oriented framework that leverages SAM 3D Body (3DB) as a frozen perception backbone and uses the Momentum HumanRig (MHR) representation as a robot-friendly intermediate. Our method (i) locks the identity and skeleton-scale parameters of per tracked subject to enforce temporally consistent bone lengths, (ii) smooths per-frame predictions via efficient sliding-window optimization in the low-dimensional MHR latent space, and (iii) recovers physically plausible global root trajectories with a differentiable soft foot-ground contact model and contact-aware global optimization. Finally, we retarget the reconstructed motion to the Unitree G1 humanoid using a kinematics-aware two-stage inverse kinematics pipeline. Results on real monocular videos show that our method has stable world trajectories and reliable robot retargeting, indicating that structured human representations with lightweight physical constraints can yield robot-ready motion from monocular input.

---

## 15. Understanding Virality: A Rubric based Vision-Language Model Framework for Short-Form Edutainment Evaluation

**论文链接:** [http://arxiv.org/abs/2512.21402v1](http://arxiv.org/abs/2512.21402v1)

**作者:** Arnav Gupta, Gurekas Singh Sahney, Hardik Rathi, Abhishek Chandwani, Ishaan Gupta, Pratik Narang, Dhruv Kumar

**发布时间:** 2025-12-24

**备注:** Under Review

### GPT解析

### 总结

本文提出了一种数据驱动的短视频内容评估框架，利用视觉语言模型提取视听特征并预测观众参与度，相比传统评估方法更具可解释性和可扩展性。

### 背景

现有短视频评估框架如VideoScore-2仅评估视觉和语义保真度，未能捕捉特定视听属性如何真正影响观众参与度。

### 目的

开发一种能够提取视听特征、聚类为可解释因素并预测短视频教育内容参与度的评估框架。

### 方法

使用视觉语言模型提取无监督视听特征，将其聚类为可解释因素，训练回归评估器预测参与度，并构建YouTube Shorts数据集分析特征与参与行为的关系。

### 主要发现

预测参与度与实际参与度之间存在强相关性，该轻量级基于特征的评估器相比传统指标（如SSIM、FID）提供了更可解释和可扩展的评估。

### 结论

通过基于多模态特征重要性和以人为中心的参与信号进行评估，这种方法推动了稳健和可解释的视频理解发展。

### 翻译

评估短视频内容需要超越表面质量指标，转向与人类一致的多模态推理。虽然现有的VideoScore-2等框架评估视觉和语义保真度，但它们没有捕捉特定视听属性如何真正驱动观众参与。在这项工作中，我们提出了一个数据驱动的评估框架，使用视觉语言模型提取无监督视听特征，将其聚类为可解释因素，并训练基于回归的评估器来预测短视频教育内容的参与度。我们精心策划的YouTube Shorts数据集使系统能分析VLM派生特征如何与人类参与行为相关。实验显示预测参与度和实际参与度之间存在强相关性，表明与传统指标相比，我们这种轻量级基于特征的评估器提供了可解释且可扩展的评估。通过将评估基于多模态特征重要性和以人为中心的参与信号，我们的方法朝着稳健和可解释的视频理解迈进。


### 论文摘要

Evaluating short-form video content requires moving beyond surface-level quality metrics toward human-aligned, multimodal reasoning. While existing frameworks like VideoScore-2 assess visual and semantic fidelity, they do not capture how specific audiovisual attributes drive real audience engagement. In this work, we propose a data-driven evaluation framework that uses Vision-Language Models (VLMs) to extract unsupervised audiovisual features, clusters them into interpretable factors, and trains a regression-based evaluator to predict engagement on short-form edutainment videos. Our curated YouTube Shorts dataset enables systematic analysis of how VLM-derived features relate to human engagement behavior. Experiments show strong correlations between predicted and actual engagement, demonstrating that our lightweight, feature-based evaluator provides interpretable and scalable assessments compared to traditional metrics (e.g., SSIM, FID). By grounding evaluation in both multimodal feature importance and human-centered engagement signals, our approach advances toward robust and explainable video understanding.

---

## 16. Modeling high dimensional point clouds with the spherical cluster model

**论文链接:** [http://arxiv.org/abs/2512.21960v1](http://arxiv.org/abs/2512.21960v1)

**作者:** Frédéric Cazals, Antoine Commaret, Louis Goldenberg

**发布时间:** 2025-12-26

**备注:** Main text: 4 figures, 15 pages

### GPT解析

### 总结

该研究介绍了一种称为球形聚类模型（SC）的参数化聚类模型，通过球体近似有限点集，并提出了一个精确求解器。实验显示该算法在多种维度数据集上表现优异，尤其在高维数据分析中有直接应用价值。

### 背景

参数化聚类模型是一种统计模型，为定义聚类的点提供几何洞察。球形聚类模型通过球体近似有限点集，其中η=0时退化为K均值聚类中使用的质心。

### 目的

研究球形聚类模型的拟合问题，开发精确求解器，并探索该模型在高维数据分析中的应用。

### 方法

1. 展示拟合球形聚类模型是一个严格凸但非光滑的组合优化问题；2. 提出使用Clarke梯度的精确求解器，基于从超球面排列定义的分层胞复形；3. 在多种维度（9到10,000）的数据集上进行实验。

### 主要发现

1. 对于小/中等维度的数据集和小η值，以及高维数据集（d>100），精确算法比基于BFGS的启发式方法快几个数量级；2. SC模型的中心表现为参数化高维中位数。

### 结论

球形聚类模型对高维多元数据分析有直接应用价值，将其应用于球形混合模型的设计将在后续论文中报告。

### 翻译

参数化聚类模型是一种统计模型，为定义聚类的点提供几何洞察。球形聚类模型（SC）通过球体S(c,r)来近似有限点集P⊂ℝ^d，具体如下：将r取为介于中心c和数据点之间距离的标准差的一个分数η∈(0,1)（超参数），SC模型的成本是所有位于球体S外部的数据点关于S的幂距离之和。SC模型的中心c是使该成本最小化的点。注意，η=0产生K均值聚类中使用的著名质心。我们做出了三项贡献。首先，我们展示了拟合球形聚类产生一个严格凸但不光滑的组合优化问题。其次，我们提出了一个精确求解器，使用Clarke梯度在从超球面排列定义的适当分层胞复形上工作。最后，我们在多种数据集上进行了实验，维度从d=9到d=10,000，主要有两个观察结果。首先，对于小/中等维度的数据集和小η值，以及对于高维数据集（d>100），精确算法比基于BFGS的启发式方法快几个数量级，无论η值如何。其次，SC模型的中心表现为参数化高维中位数。SC模型对高维多元数据分析有直接应用价值，将其应用于球形混合模型的设计将在后续论文中报告。


### 论文摘要

A parametric cluster model is a statistical model providing geometric insights onto the points defining a cluster. The {\em spherical cluster model} (SC) approximates a finite point set $P\subset \mathbb{R}^d$ by a sphere $S(c,r)$ as follows. Taking $r$ as a fraction $η\in(0,1)$ (hyper-parameter) of the std deviation of distances between the center $c$ and the data points, the cost of the SC model is the sum over all data points lying outside the sphere $S$ of their power distance with respect to $S$. The center $c$ of the SC model is the point minimizing this cost. Note that $η=0$ yields the celebrated center of mass used in KMeans clustering. We make three contributions.   First, we show fitting a spherical cluster yields a strictly convex but not smooth combinatorial optimization problem. Second, we present an exact solver using the Clarke gradient on a suitable stratified cell complex defined from an arrangement of hyper-spheres. Finally, we present experiments on a variety of datasets ranging in dimension from $d=9$ to $d=10,000$, with two main observations. First, the exact algorithm is orders of magnitude faster than BFGS based heuristics for datasets of small/intermediate dimension and small values of $η$, and for high dimensional datasets (say $d>100$) whatever the value of $η$. Second, the center of the SC model behave as a parameterized high-dimensional median.   The SC model is of direct interest for high dimensional multivariate data analysis, and the application to the design of mixtures of SC will be reported in a companion paper.

---

## 17. CrownGen: Patient-customized Crown Generation via Point Diffusion Model

**论文链接:** [http://arxiv.org/abs/2512.21890v1](http://arxiv.org/abs/2512.21890v1)

**作者:** Juyoung Bae, Moo Hyun Son, Jiale Peng, Wanting Qu, Wener Chen, Zelin Qiu, Kaixin Li, Xiaojuan Chen, Yifan Lin, Hao Chen

**发布时间:** 2025-12-26

### GPT解析

### 总结

CrownGen是一个自动化患者定制牙冠设计的生成框架，通过去噪扩散模型和新型牙齿级别点云表示解决了牙冠设计中的劳动密集型瓶颈问题。

### 背景

牙冠设计是修复牙科中的一个劳动密集型的瓶颈问题，需要大量专业时间和技能。

### 目的

开发一个自动化系统，用于生成患者定制的牙冠设计，以减少设计时间和成本，同时保持高质量。

### 方法

使用基于去噪扩散模型的生成框架，结合牙齿级别点云表示，包含边界预测模块和基于扩散的生成模块，可在单次推理中合成多颗牙齿的高保真形态。

### 主要发现

在496个外部扫描和26个修复病例的临床研究中验证，CrownGen在几何保真度上超越最先进模型，显著减少设计时间，且牙医评估确认其质量与专家手动设计相当。

### 结论

CrownGen通过自动化复杂假体建模，提供了可扩展的解决方案，可降低成本、缩短周转时间，提高患者获得高质量牙科护理的机会。

### 翻译

牙冠设计仍然是修复牙科中的一个劳动密集型瓶颈。我们提出了CrownGen，一个生成框架，使用一种新型的牙齿级别点云表示上的去噪扩散模型，自动化患者定制的牙冠设计。该系统采用两个核心组件：边界预测模块用于建立空间先验，以及基于扩散的生成模块用于在单次推理中合成多颗牙齿的高保真形态。我们通过在496个外部扫描上的定量基准测试和26个修复病例的临床研究验证了CrownGen。结果表明，CrownGen在几何保真度上超越了最先进的模型，并显著减少了主动设计时间。经过培训的牙医的临床评估确认，CrownGen辅助的牙冠在质量上与专家技术人员使用手动工作流程生产的牙冠相比没有统计学上的差异。通过自动化复杂的假体建模，CrownGen提供了一个可扩展的解决方案，可以降低成本、缩短周转时间，并增强患者获得高质量牙科护理的机会。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决牙冠设计自动化的问题。在牙科修复领域，牙冠设计目前仍是一个劳动密集型的瓶颈，技师需要手动调整模板，每个牙冠可能需要超过一小时，且随着修复数量增加而线性增长。这个问题很重要，因为它直接影响患者获得高质量牙科护理的成本、等待时间和可及性，同时全球口腔疾病负担持续增加，牙齿脱落严重影响咀嚼功能、面部美观和生活质量。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先分析了现有方法的局限性：早期2D图像合成方法无法捕捉完整3D解剖结构；3D体素方法在高分辨率临床保真度方面存在困难；而点云方法虽先进但只能生成单颗牙冠且依赖预备好的基牙。作者设计CrownGen时借鉴了去噪扩散模型技术，并创新性地提出将牙列分解为独立的牙齿级点云，而非单一整体。作者还提出了距离加权牙间注意力(DITA)机制来建模牙齿间关系，并采用两阶段伪牙冠训练策略来解决数据异质性问题，使模型能够利用部分无牙颌的临床数据。', '这个方法的核心思想是什么？整体实现流程是怎样的？': 'CrownGen的核心思想是将牙列分解为独立的牙齿级点云，使用去噪扩散模型生成患者定制的牙冠，并通过DITA机制建模牙齿间关系。整体流程包括：1)数据预处理和牙齿分割；2)边界预测模块为每个目标位置预测圆柱形边界作为空间先验；3)基于点云扩散的生成模块以上下文牙齿和预测边界为条件，通过去噪过程生成牙冠点云；4)使用泊松表面重建将点云转换为不透水网格；5)采用两阶段训练策略，先在完整牙列数据上训练，再利用生成伪牙冠扩展数据集进行微调。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': 'CrownGen的关键创新点包括：1)首次实现多牙冠一次性生成能力，可在一个推理过程中生成任意数量的解剖学协调修复体；2)创新的牙齿级表示方法，将牙列分解为独立牙齿对象集合；3)边界预测模块提供明确空间先验；4)距离加权牙间注意力(DITA)机制优先考虑局部上下文；5)可扩展的数据利用策略，能利用大规模部分无牙颌临床数据。相比之前工作，CrownGen在多牙修复场景中性能稳定，不依赖基牙预备，能利用不完美临床数据，且将牙列处理为独立牙齿对象而非单一整体。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'CrownGen通过创新的牙齿级点云表示和去噪扩散模型，首次实现了在一个推理过程中自动生成任意数量患者定制牙冠的能力，显著提高了牙科修复设计的效率和可及性。'}


### 论文摘要

Digital crown design remains a labor-intensive bottleneck in restorative dentistry. We present \textbf{CrownGen}, a generative framework that automates patient-customized crown design using a denoising diffusion model on a novel tooth-level point cloud representation. The system employs two core components: a boundary prediction module to establish spatial priors and a diffusion-based generative module to synthesize high-fidelity morphology for multiple teeth in a single inference pass. We validated CrownGen through a quantitative benchmark on 496 external scans and a clinical study of 26 restoration cases. Results demonstrate that CrownGen surpasses state-of-the-art models in geometric fidelity and significantly reduces active design time. Clinical assessments by trained dentists confirmed that CrownGen-assisted crowns are statistically non-inferior in quality to those produced by expert technicians using manual workflows. By automating complex prosthetic modeling, CrownGen offers a scalable solution to lower costs, shorten turnaround times, and enhance patient access to high-quality dental care.

---

## 18. End-to-End 3D Spatiotemporal Perception with Multimodal Fusion and V2X Collaboration

**论文链接:** [http://arxiv.org/abs/2512.21831v1](http://arxiv.org/abs/2512.21831v1)

**作者:** Zhenwei Yang, Yibo Ai, Weidong Zhang

**发布时间:** 2025-12-26

**备注:** 19 pages, 19 figures

### GPT解析

### 总结

XET-V2X是一种多模态融合端到端跟踪框架，通过双层空间交叉注意力模块和特征聚合策略，在V2X场景中实现了鲁棒且时间稳定的感知性能

### 背景

多视图协同感知和多模态融合对自动驾驶中可靠的三维时空理解至关重要，特别是在遮挡、有限视角和V2X场景中的通信延迟情况下

### 目的

提出XET-V2X，一种用于V2X协作的多模态融合端到端跟踪框架，在共享时空表示中统一多视图多模态感知

### 方法

引入基于多尺度可变形注意力的双层空间交叉注意力模块，首先聚合多视图图像特征增强语义一致性，然后通过更新的空间查询引导点云融合，实现有效跨模态交互同时减少计算开销

### 主要发现

在真实世界V2X-Seq-SPD数据集和模拟V2X-Sim-V2V、V2X-Sim-V2I基准测试上，XET-V2X在不同通信延迟条件下检测和跟踪性能有持续改进，定量结果和定性可视化表明其在复杂交通场景中实现了鲁棒且时间稳定的感知

### 结论

XET-V2X框架能够有效处理V2X场景中的挑战，提供可靠的3D时空理解

### 翻译

多视图协同感知和多模态融合对自动驾驶中可靠的三维时空理解至关重要，特别是在遮挡、有限视角和V2X场景中的通信延迟情况下。本文提出了XET-V2X，一种用于V2X协作的多模态融合端到端跟踪框架，在共享时空表示中统一了多视图多模态感知。为了高效对齐异构视角和模态，XET-V2X引入了基于多尺度可变形注意力的双层空间交叉注意力模块。多视图图像特征首先被聚合以增强语义一致性，然后由更新的空间查询引导的点云融合，实现有效的跨模态交互同时减少计算开销。在真实世界V2X-Seq-SPD数据集和模拟V2X-Sim-V2V、V2X-Sim-V2I基准测试上的实验表明，在不同通信延迟条件下，检测和跟踪性能有持续改进。定量结果和定性可视化都表明XET-V2X在复杂交通场景中实现了鲁棒且时间稳定的感知。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决自动驾驶中的3D时空感知问题，特别是在V2X环境下的多视角协同感知和多模态融合。这个问题很重要，因为在复杂交通场景中，单一车辆或单一传感器的感知能力有限，无法完全覆盖所有情况。通过V2X协同感知可以扩展感知范围，减少遮挡，提高检测和跟踪的可靠性，从而增强自动驾驶的安全性和效率。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者从三个维度进行思考：1)空间互补性：多视角可提供互补观察，扩展感知范围；2)模态多样性：激光雷达和摄像头具有互补特性，多模态融合可提高鲁棒性；3)时间连续性：需要时间建模保持身份一致性。作者借鉴了多视图协同感知(如V2VNet)、多模态融合(如PointPainting、Transformer架构)、时间信息融合(如循环机制)和端到端多目标跟踪(如MOTR)等现有工作，并将它们整合到一个统一框架中。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是通过统一的端到端框架，联合建模多视图协作、多模态融合和时间演化，解决V2X环境下的3D时空感知问题。整体流程：1)多视图多模态特征提取(点云用PointPillars，图像用ResNet-101)；2)特征级图像传输减少带宽需求；3)双层空间交叉注意力模块融合特征(先图像后点云)；4)基于MOTR框架进行端到端检测和跟踪。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点：1)统一的端到端框架整合多视图协作、多模态融合和时间建模；2)双层跨模态跨视图交互模块；3)共享BEV表示实现跨视图一致性；4)特征级传输策略。相比之前工作，XET-V2X将多视图、多模态和时间建模整合在一个框架中，采用端到端学习而非模块化方法，并引入了双层注意力机制和统一的时空建模。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '本文提出了XET-V2X，一个统一的端到端3D时空感知框架，通过多视图协同、多模态融合和V2X协作，显著提升了自动驾驶环境中的物体检测和跟踪性能，特别是在遮挡、有限视野和通信延迟等挑战场景下。'}


### 论文摘要

Multi-view cooperative perception and multimodal fusion are essential for reliable 3D spatiotemporal understanding in autonomous driving, especially under occlusions, limited viewpoints, and communication delays in V2X scenarios. This paper proposes XET-V2X, a multi-modal fused end-to-end tracking framework for v2x collaboration that unifies multi-view multimodal sensing within a shared spatiotemporal representation. To efficiently align heterogeneous viewpoints and modalities, XET-V2X introduces a dual-layer spatial cross-attention module based on multi-scale deformable attention. Multi-view image features are first aggregated to enhance semantic consistency, followed by point cloud fusion guided by the updated spatial queries, enabling effective cross-modal interaction while reducing computational overhead. Experiments on the real-world V2X-Seq-SPD dataset and the simulated V2X-Sim-V2V and V2X-Sim-V2I benchmarks demonstrate consistent improvements in detection and tracking performance under varying communication delays. Both quantitative results and qualitative visualizations indicate that XET-V2X achieves robust and temporally stable perception in complex traffic scenarios.

---

## 19. Spatiotemporal-Untrammelled Mixture of Experts for Multi-Person Motion Prediction

**论文链接:** [http://arxiv.org/abs/2512.21707v1](http://arxiv.org/abs/2512.21707v1)

**作者:** Zheng Yin, Chengjian Li, Xiangbo Shu, Meiqi Cao, Rui Yan, Jinhui Tang

**发布时间:** 2025-12-25

**备注:** 12 pages, 7 figures, Accepted by AAAI 2026 (oral)

### GPT解析

### 总结

本文提出了一种名为时空无限制专家混合模型(ST-MoE)的新方法，用于多人运动预测，能够灵活捕捉复杂时空依赖关系并降低计算成本。

### 背景

多人运动预测中全面灵活地捕捉复杂时空依赖关系至关重要，但现有方法存在两个主要局限性：时空表示不够灵活和计算成本高。

### 目的

克服现有方法的局限性，提出一种既能灵活探索人体运动中的复杂时空依赖关系又能显著降低计算成本的模型。

### 方法

提出时空无限制专家混合模型(ST-MoE)，包含四种不同类型的时空专家，每种专家专门捕捉不同的空间或时间依赖关系；引入双向时空Mamba作为专家，通过不同组合共享双向时间和空间Mamba，实现模型效率和参数经济。

### 主要发现

在四个多人基准数据集上的实验表明，该方法在准确性上优于最先进的方法，同时模型参数减少41.38%，训练速度提升3.6倍。

### 结论

ST-MoE模型成功解决了多人运动预测中的时空表示灵活性和计算效率问题，为未来研究提供了新的思路。

### 翻译

全面灵活地捕捉人体运动的复杂时空依赖关系对多人运动预测至关重要。现有方法面临两个主要局限性：i) 由于依赖位置编码来捕捉时空信息，导致时空表示不够灵活；ii) 传统注意力机制的二次时间复杂度导致高计算成本。为克服这些局限性，我们提出时空无限制专家混合模型(ST-MoE)，能灵活探索人体运动中的复杂时空依赖关系并显著降低计算成本。为自适应挖掘人体运动中的复杂时空模式，我们的模型纳入了四种不同类型的时空专家，每种专家专门捕捉不同的空间或时间依赖关系。为在整合多个专家时减少潜在计算开销，我们引入双向时空Mamba作为专家，通过不同组合共享双向时间和空间Mamba，实现模型效率和参数经济。在四个多人基准数据集上的广泛实验表明，我们的方法不仅在准确性上优于最先进水平，还减少了41.38%的模型参数，实现了3.6倍的训练加速。代码可在https://github.com/alanyz106/ST-MoE获取。


### 论文摘要

Comprehensively and flexibly capturing the complex spatio-temporal dependencies of human motion is critical for multi-person motion prediction. Existing methods grapple with two primary limitations: i) Inflexible spatiotemporal representation due to reliance on positional encodings for capturing spatiotemporal information. ii) High computational costs stemming from the quadratic time complexity of conventional attention mechanisms. To overcome these limitations, we propose the Spatiotemporal-Untrammelled Mixture of Experts (ST-MoE), which flexibly explores complex spatio-temporal dependencies in human motion and significantly reduces computational cost. To adaptively mine complex spatio-temporal patterns from human motion, our model incorporates four distinct types of spatiotemporal experts, each specializing in capturing different spatial or temporal dependencies. To reduce the potential computational overhead while integrating multiple experts, we introduce bidirectional spatiotemporal Mamba as experts, each sharing bidirectional temporal and spatial Mamba in distinct combinations to achieve model efficiency and parameter economy. Extensive experiments on four multi-person benchmark datasets demonstrate that our approach not only outperforms state-of-art in accuracy but also reduces model parameter by 41.38% and achieves a 3.6x speedup in training. The code is available at https://github.com/alanyz106/ST-MoE.

---

## 20. Unsupervised Anomaly Detection in Brain MRI via Disentangled Anatomy Learning

**论文链接:** [http://arxiv.org/abs/2512.21924v1](http://arxiv.org/abs/2512.21924v1)

**作者:** Tao Yang, Xiuying Wang, Hao Liu, Guanzhong Gong, Lian-Ming Wu, Yu-Ping Wang, Lisheng Wang

**发布时间:** 2025-12-26

**DOI:** 10.1016/j.media.2025.103922

**备注:** Accepted by Medical Image Analysis (2025)

### GPT解析

### 总结

该研究提出了一种新型PHI重建框架，通过解耦表示模块和边缘到图像恢复模块，解决了脑部MRI病变检测中无监督学习方法面临的泛化能力有限和性能受限问题，在多个数据集上显著优于现有方法。

### 背景

脑部MRI中各种病变的检测在临床上至关重要，但由于病变多样性和成像条件变化而具有挑战性。当前无监督学习方法主要通过正常样本学习将异常图像重建为伪健康图像(PHIs)，然后分析图像差异来检测异常。

### 目的

解决当前无监督模型在脑部MRI病变检测中的两个主要限制：对多模态和多中心MRI的泛化能力有限，以及由于异常残差传播导致的性能受限。

### 方法

提出两个新模块形成新的PHI重建框架：1)解耦表示模块，将脑部MRI解耦为成像信息和基本成像不变解剖图像，引入脑部解剖先验和可微的一热编码算子增强解耦稳定性；2)边缘到图像恢复模块，从解剖图像的高频边缘信息恢复解剖表示并重新耦合成像信息，通过仅边缘输入减少异常像素输入。

### 主要发现

在9个公共数据集(4,443名患者的MRI)上评估，该方法优于17种最先进方法，在AP和DSC指标上分别实现了+18.32%和+13.64%的绝对改进。

### 结论

提出的解耦表示和边缘到图像恢复模块有效解决了现有无监督学习方法在脑部MRI病变检测中的局限性，显著提高了检测性能和泛化能力。

### 翻译

脑部MRI中各种病变的检测在临床上至关重要，但由于病变多样性和成像条件变化而具有挑战性。当前无监督学习方法主要通过正常样本学习将异常图像重建为伪健康图像(PHIs)，然后分析图像差异来检测异常。然而，这些无监督模型面临两个显著限制：由于依赖正常训练数据中的特定成像信息，其泛化能力受限，难以适应多模态和多中心MRI；由于异常残差从输入图像传播到重建的PHIs，性能受限。为解决这些限制，提出了两个新模块，形成新的PHI重建框架。首先，提出了解耦表示模块，通过将脑部MRI解耦为成像信息和基本成像不变解剖图像来提高泛化能力，确保重建专注于解剖结构。具体而言，引入脑部解剖先验和可微的一热编码算子来约束解耦结果并增强解耦稳定性。其次，设计了边缘到图像恢复模块，通过从解剖图像的高频边缘信息恢复解剖表示，然后重新耦合解耦的成像信息，重建高质量PHIs。该模块通过仅边缘输入减少异常像素输入来抑制PHI中的异常残差，同时利用边缘中保留的结构细节有效重建正常区域。在9个公共数据集(来自多个中心的4,443名患者的MRI)上评估，我们的方法优于17种最先进方法，在AP和DSC指标上分别实现了+18.32%和+13.64%的绝对改进。


### 论文摘要

Detection of various lesions in brain MRI is clinically critical, but challenging due to the diversity of lesions and variability in imaging conditions. Current unsupervised learning methods detect anomalies mainly through reconstructing abnormal images into pseudo-healthy images (PHIs) by normal samples learning and then analyzing differences between images. However, these unsupervised models face two significant limitations: restricted generalizability to multi-modality and multi-center MRIs due to their reliance on the specific imaging information in normal training data, and constrained performance due to abnormal residuals propagated from input images to reconstructed PHIs. To address these limitations, two novel modules are proposed, forming a new PHI reconstruction framework. Firstly, the disentangled representation module is proposed to improve generalizability by decoupling brain MRI into imaging information and essential imaging-invariant anatomical images, ensuring that the reconstruction focuses on the anatomy. Specifically, brain anatomical priors and a differentiable one-hot encoding operator are introduced to constrain the disentanglement results and enhance the disentanglement stability. Secondly, the edge-to-image restoration module is designed to reconstruct high-quality PHIs by restoring the anatomical representation from the high-frequency edge information of anatomical images, and then recoupling the disentangled imaging information. This module not only suppresses abnormal residuals in PHI by reducing abnormal pixels input through edge-only input, but also effectively reconstructs normal regions using the preserved structural details in the edges. Evaluated on nine public datasets (4,443 patients' MRIs from multiple centers), our method outperforms 17 SOTA methods, achieving absolute improvements of +18.32% in AP and +13.64% in DSC.

---

## 21. Zero-Shot to Zero-Lies: Detecting Bengali Deepfake Audio through Transfer Learning

**论文链接:** [http://arxiv.org/abs/2512.21702v1](http://arxiv.org/abs/2512.21702v1)

**作者:** Most. Sharmin Sultana Samu, Md. Rakibul Islam, Md. Zahid Hossain, Md. Kamrozzaman Bhuiyan, Farhad Uz Zaman

**发布时间:** 2025-12-25

**备注:** Accepted for publication in 2025 28th International Conference on Computer and Information Technology (ICCIT)

### GPT解析

### 总结

该研究针对孟加拉语深度伪造音频检测问题进行了系统评估，通过零样本推理和微调两种方法比较了多种深度学习模型的性能，发现微调显著提升了检测效果。

### 背景

语音合成和语音转换系统的快速发展使深度伪造音频成为主要的安全威胁，而孟加拉语深度伪造检测领域尚未得到充分探索。

### 目的

研究使用BanglaFake数据集自动检测孟加拉语音频深度伪造，为这一低资源语言提供首个系统基准。

### 方法

评估了Wav2Vec2-XLSR-53、Whisper、PANNsCNN14、WavLM和Audio Spectrogram Transformer等预训练模型的零样本推理能力，并对Wav2Vec2-Base、LCNN、LCNN-Attention、ResNet18、ViT-B16和CNN-BiLSTM等架构进行了微调。

### 主要发现

零样本结果显示检测能力有限，最佳模型Wav2Vec2-XLSR-53仅达到53.80%准确率；微调后性能显著提升，ResNet18表现最优，达到79.17%准确率、79.12% F1分数、84.37% AUC和24.35% EER。

### 结论

微调深度学习模型在孟加拉语深度伪造音频检测中表现出色，为低资源语言的深度伪造检测提供了有效解决方案。

### 翻译

语音合成和语音转换系统的快速增长使深度伪造音频成为主要的安全问题。孟加拉语深度伪造检测在很大程度上仍未被探索。在这项工作中，我们使用BanglaFake数据集研究孟加拉语音频深度伪造的自动检测。我们评估了几个预训练模型的零样本推理能力，包括Wav2Vec2-XLSR-53、Whisper、PANNsCNN14、WavLM和音频频谱变换器。零样本结果显示检测能力有限。最佳模型Wav2Vec2-XLSR-53达到53.80%的准确率、56.60%的AUC和46.20%的EER。然后，我们为孟加拉语深度伪造检测微调了多种架构，包括Wav2Vec2-Base、LCNN、LCNN-Attention、ResNet18、ViT-B16和CNN-BiLSTM。微调后的模型显示出强大的性能提升。ResNet18达到最高的79.17%准确率、79.12%的F1分数、84.37%的AUC和24.35%的EER。实验结果证实微调显著优于零样本推理。这项研究提供了孟加拉语深度伪造音频检测的第一个系统基准。它强调了微调后的深度学习模型在低资源语言中的有效性。


### 论文摘要

The rapid growth of speech synthesis and voice conversion systems has made deepfake audio a major security concern. Bengali deepfake detection remains largely unexplored. In this work, we study automatic detection of Bengali audio deepfakes using the BanglaFake dataset. We evaluate zeroshot inference with several pretrained models. These include Wav2Vec2-XLSR-53, Whisper, PANNsCNN14, WavLM and Audio Spectrogram Transformer. Zero-shot results show limited detection ability. The best model, Wav2Vec2-XLSR-53, achieves 53.80% accuracy, 56.60% AUC and 46.20% EER. We then f ine-tune multiple architectures for Bengali deepfake detection. These include Wav2Vec2-Base, LCNN, LCNN-Attention, ResNet18, ViT-B16 and CNN-BiLSTM. Fine-tuned models show strong performance gains. ResNet18 achieves the highest accuracy of 79.17%, F1 score of 79.12%, AUC of 84.37% and EER of 24.35%. Experimental results confirm that fine-tuning significantly improves performance over zero-shot inference. This study provides the first systematic benchmark of Bengali deepfake audio detection. It highlights the effectiveness of f ine-tuned deep learning models for this low-resource language.

---

## 22. Cross-Semantic Transfer Learning for High-Dimensional Linear Regression

**论文链接:** [http://arxiv.org/abs/2512.21689v1](http://arxiv.org/abs/2512.21689v1)

**作者:** Jiancheng Jiang, Xuejun Jiang, Hongxia Jin

**发布时间:** 2025-12-25

### GPT解析

### 总结

本文提出了跨语义迁移学习(CSTL)框架，用于处理高维线性回归中的特征不对齐问题，通过比较目标域和源域系数实现更有效的知识迁移

### 背景

现有高维线性回归迁移学习方法假设域间特征对齐，限制了在语义匹配特征上的应用，而现实中目标域和源域的不同特征可能扮演相似预测角色

### 目的

利用更广泛的跨语义相似性，提出CSTL框架捕获域间潜在关系，保留可迁移信号同时过滤源特定噪声

### 方法

CSTL通过比较每个目标系数与所有源系数，利用加权融合惩罚捕获潜在关系，权重由SCAD惩罚导数推导，使用ADMM算法实现计算效率

### 主要发现

理论上在温和条件下CSTL能以压倒性概率实现oracle估计器，实证结果显示其在跨语义和部分信号相似性设置下均优于现有方法

### 结论

CSTL框架有效解决了特征不对齐的高维线性回归迁移学习问题，通过捕获跨语义相似性实现了更好的性能

### 翻译

当前的高维线性回归迁移学习方法假设域间特征对齐，限制了它们在语义匹配特征上的适用性。然而，在许多现实场景中，目标域和源域中的不同特征可以扮演相似的预测角色，形成一种跨语义相似性。为了利用这种更广泛的迁移能力，我们提出了跨语义迁移学习(CSTL)框架。它通过比较每个目标系数与所有源系数，利用加权融合惩罚来捕获潜在关系。权重由SCAD惩罚的导数推导得出，有效近似了理想的加权方案，保留了可迁移信号同时过滤掉源特定噪声。为了计算效率，我们使用交替方向乘子法(ADMM)实现了CSTL。理论上，我们建立了在温和条件下，CSTL能够以压倒性概率实现oracle估计器。来自模拟和真实数据应用的实证结果证实，CSTL在跨语义和部分信号相似性设置下都优于现有方法。


### 论文摘要

Current transfer learning methods for high-dimensional linear regression assume feature alignment across domains, restricting their applicability to semantically matched features. In many real-world scenarios, however, distinct features in the target and source domains can play similar predictive roles, creating a form of cross-semantic similarity. To leverage this broader transferability, we propose the Cross-Semantic Transfer Learning (CSTL) framework. It captures potential relationships by comparing each target coefficient with all source coefficients through a weighted fusion penalty. The weights are derived from the derivative of the SCAD penalty, effectively approximating an ideal weighting scheme that preserves transferable signals while filtering out source-specific noise. For computational efficiency, we implement CSTL using the Alternating Direction Method of Multipliers (ADMM). Theoretically, we establish that under mild conditions, CSTL achieves the oracle estimator with overwhelming probability. Empirical results from simulations and a real-data application confirm that CSTL outperforms existing methods in both cross-semantic and partial signal similarity settings.

---

## 23. Intelligent recognition of GPR road hidden defect images based on feature fusion and attention mechanism

**论文链接:** [http://arxiv.org/abs/2512.21452v1](http://arxiv.org/abs/2512.21452v1)

**作者:** Haotian Lv, Yuhui Zhang, Jiangbo Dai, Hanli Wu, Jiaji Wang, Dawei Wang

**发布时间:** 2025-12-25

**DOI:** 10.1109/TGRS.2025.3575293

**备注:** Accepted for publication in *IEEE Transactions on Geoscience and Remote Sensing*

### GPT解析

### 总结

该研究提出了一种全面的框架，用于自动化地面穿透雷达(GPR)图像中的道路缺陷检测，解决了传统方法依赖主观专业知识的局限性。

### 背景

地面穿透雷达(GPR)已成为评估地下道路缺陷的关键工具，但传统GPR图像解释主要依赖主观专业知识，导致效率低下和准确性不足。

### 目的

解决传统GPR图像解释的局限性，建立一个全面的框架来自动化GPR缺陷检测，平衡计算效率与高准确性。

### 方法

1) 基于DCGAN的数据增强策略生成高保真GPR图像缓解数据稀缺问题；2) 提出多模态链和全局注意力网络(MCGA-Net)，包括多模态链特征融合(MCFF)和全局注意力机制(GAM)；3) 使用MS COCO迁移学习微调骨干网络，加速收敛并提高泛化能力。

### 主要发现

MCGA-Net达到92.8%的精确率、92.5%的召回率和95.9%的mAP@50；在高斯噪声、弱信号和小目标检测中保持鲁棒性并优于其他模型。

### 结论

该工作建立了基于GPR的缺陷检测的新范式，在复杂地下环境中平衡了计算效率和高准确性。

### 翻译

地面穿透雷达(GPR)已成为评估地下道路缺陷的关键工具。然而，传统GPR图像解释仍然严重依赖主观专业知识，引入了低效和不准确。本研究引入了一个全面的框架来解决这些局限性：(1)基于DCGAN的数据增强策略合成高保真GPR图像，缓解数据稀缺问题，同时保持复杂背景下的缺陷形态；(2)提出了一种新的多模态链和全局注意力网络(MCGA-Net)，集成多模态链特征融合(MCFF)进行分层多尺度缺陷表示，以及全局注意力机制(GAM)进行上下文感知特征增强；(3)MS COCO迁移学习微调骨干网络，加速收敛并提高泛化能力。消融和比较实验验证了该框架的有效性。MCGA-Net实现了精确率(92.8%)、召回率(92.5%)和mAP@50(95.9%)。在高斯噪声、弱信号和小目标检测中，MCGA-Net保持鲁棒性并优于其他模型。这项工作建立了基于GPR的缺陷检测的新范式，在复杂地下环境中平衡计算效率与高准确性。


### 论文摘要

Ground Penetrating Radar (GPR) has emerged as a pivotal tool for non-destructive evaluation of subsurface road defects. However, conventional GPR image interpretation remains heavily reliant on subjective expertise, introducing inefficiencies and inaccuracies. This study introduces a comprehensive framework to address these limitations: (1) A DCGAN-based data augmentation strategy synthesizes high-fidelity GPR images to mitigate data scarcity while preserving defect morphology under complex backgrounds; (2) A novel Multi-modal Chain and Global Attention Network (MCGA-Net) is proposed, integrating Multi-modal Chain Feature Fusion (MCFF) for hierarchical multi-scale defect representation and Global Attention Mechanism (GAM) for context-aware feature enhancement; (3) MS COCO transfer learning fine-tunes the backbone network, accelerating convergence and improving generalization. Ablation and comparison experiments validate the framework's efficacy. MCGA-Net achieves Precision (92.8%), Recall (92.5%), and mAP@50 (95.9%). In the detection of Gaussian noise, weak signals and small targets, MCGA-Net maintains robustness and outperforms other models. This work establishes a new paradigm for automated GPR-based defect detection, balancing computational efficiency with high accuracy in complex subsurface environments.

---

## 24. Physics-Informed Neural Solvers for Periodic Quantum Eigenproblems

**论文链接:** [http://arxiv.org/abs/2512.21349v1](http://arxiv.org/abs/2512.21349v1)

**作者:** Haaris Mian

**发布时间:** 2025-12-20

**备注:** Master's thesis

### GPT解析

### 总结

本文提出了一种物理信息机器学习框架，用于求解二维周期势中粒子的Floquet-Bloch本征值问题，特别关注蜂巢晶格几何结构。

### 背景

蜂巢晶格具有独特的能带拓扑特性，包含狄拉克点，与石墨烯等材料相关。

### 目的

开发一种无监督的网格求解器，能够同时学习复杂的布洛赫函数及其相关的本征值（能量），恢复能带结构和布洛赫模式。

### 方法

利用神经网络学习布洛赫函数和本征值，通过复合损失函数强制执行薛定谔方程、布洛赫周期性和归一化约束，在布里渊区上训练模型，并采用迁移学习技术从近自由电子势适应到强变化势。

### 主要发现

模型能够准确恢复能带结构和布洛赫模式，并与传统平面波展开方法的结果相符；能够捕获能带结构拓扑变化的能力。

### 结论

该工作为量子本征问题的物理信息机器学习领域做出贡献，提供了关于对称性、能带结构和神经架构之间相互作用的理解。

### 翻译

本论文提出了一种物理信息机器学习框架，用于求解与二维周期势中粒子相关的Floquet-Bloch本征值问题，特别关注蜂巢晶格几何结构，因其具有包含狄拉克点的独特能带拓扑及其与石墨烯等材料的相关性。通过利用神经网络同时学习复杂的布洛赫函数及其相关的本征值（能量），我们开发了一种无监督的网格求解器，通过复合损失函数强制执行薛定谔方程、布洛赫周期性和归一化约束。模型在布里渊区上进行训练，以恢复能带结构和布洛赫模式，并通过与传统平面波展开方法的数值验证。我们进一步探索了迁移学习技术，使求解器从近自由电子势适应到强变化势，展示了其捕获能带结构拓扑变化的能力。这项工作为量子本征问题的物理信息机器学习不断发展的领域做出了贡献，提供了关于对称性、能带结构和神经架构之间相互作用的见解。


### 论文摘要

This thesis presents a physics-informed machine learning framework for solving the Floquet-Bloch eigenvalue problem associated with particles in two-dimensional periodic potentials, with a focus on honeycomb lattice geometry, due to its distinctive band topology featuring Dirac points and its relevance to materials such as graphene. By leveraging neural networks to learn complex Bloch functions and their associated eigenvalues (energies) simultaneously, we develop a mesh-free solver enforcing the governing Schrödinger equation, Bloch periodicity, and normalization constraints through a composite loss function without supervision. The model is trained over the Brillouin zone to recover band structures and Bloch modes, with numerical validation against traditional plane-wave expansion methods. We further explore transfer learning techniques to adapt the solver from nearly-free electron potentials to strongly varying potentials, demonstrating its ability to capture changes in band structure topology. This work contributes to the growing field of physics-informed machine learning for quantum eigenproblems, providing insights into the interplay between symmetry, band structure, and neural architectures.

---

## 25. Patch-Discontinuity Mining for Generalized Deepfake Detection

**论文链接:** [http://arxiv.org/abs/2512.22027v1](http://arxiv.org/abs/2512.22027v1)

**作者:** Huanhuan Yuan, Yang Ping, Zhengqin Xu, Junyi Cao, Shuai Jia, Chao Ma

**发布时间:** 2025-12-26

**备注:** Our paper was accepted by the IEEE Transactions on Multimedia

### GPT解析

### 总结

本文提出了一种名为GenDF的简单而有效的深度伪造检测框架，通过迁移大规模视觉模型，结合深度伪造特定表征学习、特征空间重新分配和分类不变特征增强策略，实现了在跨域和跨操作设置中的最先进泛化性能。

### 背景

生成式人工智能的快速发展使得创建高度逼真的假面部图像成为可能，对个人隐私和在线信息完整性构成严重威胁。

### 目的

开发一种能够有效检测深度伪造图像的方法，特别是在面对未见过的伪造模式时保持良好的泛化性能。

### 方法

GenDF框架将强大的大规模视觉模型转移到深度伪造检测任务中，采用紧凑的网络设计，并集成了深度伪造特定的表征学习、特征空间重新分配以减轻分布不匹配问题，以及分类不变特征增强策略来提高泛化能力而不增加额外参数。

### 主要发现

GenDF在跨域和跨操作设置中实现了最先进的泛化性能，同时只需要0.28M的可训练参数，证明了该方法的有效性和效率。

### 结论

GenDF框架通过简洁而有效的设计，成功解决了现有深度伪造检测方法在泛化能力方面的局限性，为深度伪造检测提供了一个高效且实用的解决方案。

### 翻译

生成式人工智能的快速发展使得创建高度逼真的假面部图像成为可能，对个人隐私和在线信息完整性构成严重威胁。现有的深度伪造检测方法通常依赖于手工制作的取证线索和复杂架构，在域内设置中表现良好，但在面对未见过的伪造模式时性能显著下降。在本文中，我们提出了GenDF，一个简单而有效的框架，它将强大的大规模视觉模型转移到深度伪造检测任务中，采用紧凑而简洁的网络设计。GenDF集成了深度伪造特定的表征学习，以捕捉真实和假面部图像之间的判别性模式，特征空间重新分配以减轻分布不匹配问题，以及分类不变特征增强策略，在不引入额外可训练参数的情况下提高泛化能力。大量实验表明，GenDF在跨域和跨操作设置中实现了最先进的泛化性能，同时只需要0.28M的可训练参数，验证了所提出框架的有效性和效率。


### 论文摘要

The rapid advancement of generative artificial intelligence has enabled the creation of highly realistic fake facial images, posing serious threats to personal privacy and the integrity of online information. Existing deepfake detection methods often rely on handcrafted forensic cues and complex architectures, achieving strong performance in intra-domain settings but suffering significant degradation when confronted with unseen forgery patterns. In this paper, we propose GenDF, a simple yet effective framework that transfers a powerful large-scale vision model to the deepfake detection task with a compact and neat network design. GenDF incorporates deepfake-specific representation learning to capture discriminative patterns between real and fake facial images, feature space redistribution to mitigate distribution mismatch, and a classification-invariant feature augmentation strategy to enhance generalization without introducing additional trainable parameters. Extensive experiments demonstrate that GenDF achieves state-of-the-art generalization performance in cross-domain and cross-manipulation settings while requiring only 0.28M trainable parameters, validating the effectiveness and efficiency of the proposed framework.

---

## 26. Patch as Node: Human-Centric Graph Representation Learning for Multimodal Action Recognition

**论文链接:** [http://arxiv.org/abs/2512.21916v1](http://arxiv.org/abs/2512.21916v1)

**作者:** Zeyu Liang, Hailun Xia, Naichuan Zheng

**发布时间:** 2025-12-26

### GPT解析

### 总结

本文提出PAN框架，首个用于多模态动作识别的以人为中心的图表示学习方法，有效解决了RGB和骨架模态间的异质性问题，实现了更有效的特征融合。

### 背景

人类动作识别虽已取得显著成就，但融合RGB和骨架模态的多模态方法仍受其固有异质性困扰，无法充分利用模态间的互补潜力。

### 目的

提出一种以人为中心的图表示学习框架，实现更有效的多模态特征融合，减少对高质量骨架数据的依赖。

### 方法

将包含人体关节的RGB块标记嵌入表示为时空图；采用人类中心图建模范式抑制RGB帧冗余；提出基于注意力的后校准减少对高质量骨架数据的依赖；开发两种变体：PAN-Ensemble（双路径图卷积网络加后期融合）和PAN-Unified（单网络内统一图表示学习）。

### 主要发现

在三个广泛使用的多模态动作识别数据集上，PAN-Ensemble和PAN-Unified在各自的多模态融合设置中实现了最先进性能。

### 结论

PAN框架通过以人为中心的图表示学习有效解决了多模态方法中的异质性问题，实现了RGB和骨架模态的更有效融合，并在标准数据集上取得最先进性能。

### 翻译

虽然人类动作识别已经取得了显著成就，但融合RGB和骨架模态的多模态方法仍然受到其固有异质性的困扰，无法充分利用它们之间的互补潜力。在本文中，我们提出了PAN，这是第一个用于多模态动作识别的以人为中心的图表示学习框架，其中包含人体关节的RGB块标记嵌入被表示为时空图。人类中心图建模范式抑制了RGB帧中的冗余，并与基于骨架的方法很好地对齐，从而实现更有效和语义一致的多模态特征融合。由于标记嵌入的采样严重依赖二维骨架数据，我们进一步提出了基于注意力的后校准，以在模型性能最小损失的情况下减少对高质量骨架数据的依赖。为了探索PAN与基于骨架方法的集成潜力，我们提出了两种变体：PAN-Ensemble，它采用双路径图卷积网络后接后期融合；以及PAN-Unified，它在单个网络内执行统一的图表示学习。在三个广泛使用的多模态动作识别数据集上，PAN-Ensemble和PAN-Unified在各自的多模态融合设置中分别实现了最先进性能。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决多模态动作识别中RGB模态和骨骼模态融合的挑战。RGB包含丰富外观信息但冗余且易受环境影响，骨骼数据简洁鲁棒但缺乏外观信息且依赖数据质量。两种模态存在异构性，现有方法无法有效利用它们的互补潜力。这个问题很重要，因为人类动作识别在智能监控、人机交互等领域有广泛应用，结合两种模态可提高识别准确性和鲁棒性。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者观察到可区分的视觉线索常位于人体骨骼关节周围，并受视觉图神经网络工作的启发，思考RGB帧是否可采用与骨骼数据相同的图建模范式。他们设计PAN框架，利用视觉基础模型编码RGB帧，基于骨骼数据采样包含人体关节的图像块标记，并通过注意力后校准减少对高质量骨骼数据的依赖。该方法借鉴了骨骼数据建模中的图卷积网络、视觉基础模型以及注意力机制，但创新性地将RGB帧直接建模为图结构。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是将RGB图像中的图像块视为图节点，构建以人为中心的图表示，与骨骼数据的图结构一致，实现结构一致性和语义对齐。整体流程：1)提取RGB帧和同步骨骼数据；2)用视觉基础模型编码RGB帧；3)基于骨骼数据采样包含人体关节的图像块标记；4)通过注意力后校准改进采样标记；5)将视觉标记表示为时空图输入GCN处理；6)设计两种变体实现多模态融合：PAN-Ensemble采用双路径晚期融合，PAN-Unified在单网络内统一学习。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点：1)首次提出人体中心图表示学习框架将RGB建模为图；2)引入注意力后校准减少对高质量骨骼数据的依赖；3)实现视觉标记图与骨骼图的结构一致性和语义对齐；4)设计PAN-Ensemble和PAN-Unified两种融合变体。不同之处：之前方法要么将骨骼数据转换为同构表示，要么在分离路径处理，而PAN直接将RGB建模为图；只采样关键图像块减少冗余；通过后校准增强鲁棒性；利用一致图结构实现更精细的跨模态融合；解决时间分辨率不匹配问题。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'PAN首次提出将RGB图像块建模为人体中心图表示，通过结构一致的图结构实现RGB和骨骼模态的有效融合，并在多模态动作识别任务上达到了最先进的性能。'}


### 论文摘要

While human action recognition has witnessed notable achievements, multimodal methods fusing RGB and skeleton modalities still suffer from their inherent heterogeneity and fail to fully exploit the complementary potential between them. In this paper, we propose PAN, the first human-centric graph representation learning framework for multimodal action recognition, in which token embeddings of RGB patches containing human joints are represented as spatiotemporal graphs. The human-centric graph modeling paradigm suppresses the redundancy in RGB frames and aligns well with skeleton-based methods, thus enabling a more effective and semantically coherent fusion of multimodal features. Since the sampling of token embeddings heavily relies on 2D skeletal data, we further propose attention-based post calibration to reduce the dependency on high-quality skeletal data at a minimal cost interms of model performance. To explore the potential of PAN in integrating with skeleton-based methods, we present two variants: PAN-Ensemble, which employs dual-path graph convolution networks followed by late fusion, and PAN-Unified, which performs unified graph representation learning within a single network. On three widely used multimodal action recognition datasets, both PAN-Ensemble and PAN-Unified achieve state-of-the-art (SOTA) performance in their respective settings of multimodal fusion: separate and unified modeling, respectively.

---

## 27. MMCTOP: A Multimodal Textualization and Mixture-of-Experts Framework for Clinical Trial Outcome Prediction

**论文链接:** [http://arxiv.org/abs/2512.21897v1](http://arxiv.org/abs/2512.21897v1)

**作者:** Carolina Aparício, Qi Shi, Bo Wen, Tesfaye Yadete, Qiwei Han

**发布时间:** 2025-12-26

**备注:** 15 pages, 3 figures, 5 tables

### GPT解析

### 总结

本文提出了MMCTOP，一个多模态临床试验结果预测框架，通过整合异构生物医学信号、结合模式引导的文本化和模态感知的表示学习，以及使用稀疏专家混合模型进行融合，解决了高维生物医学信息学中的多模态数据融合挑战。

### 背景

高维生物医学信息学中多模态数据融合面临挑战

### 目的

开发一个多模态临床试验结果预测框架，提高预测性能和可靠性

### 方法

MMCTOP整合三种异构生物医学信号（分子结构表示、协议元数据和资格叙述、疾病本体），使用模式引导的文本化和保真度验证，结合模态感知的表示学习，通过特定领域编码器生成对齐嵌入，并利用药物-疾病条件稀疏专家混合模型进行融合，采用top-k路由保持计算效率，应用温度缩放获得校准概率

### 主要发现

在基准数据集上，MMCTOP在精确度、F1和AUC方面始终优于单模态和多模态基线；消融研究表明模式引导的文本化和选择性专家路由对性能和稳定性有实质性贡献

### 结论

MMCTOP通过结合受控的叙述规范化、上下文条件的专家融合和操作保障，推进了多模态试验建模，提高了生物医学信息学中的可审计性和可重复性

### 翻译

针对高维生物医学信息学中多模态数据融合的挑战，我们提出了MMCTOP，一个多模态临床试验结果预测框架，整合了跨越分子结构表示、协议元数据和长格式资格叙述以及疾病本体的异构生物医学信号。MMCTOP将模式引导的文本化和保真度验证与模态感知的表示学习相结合，其中特定领域编码器生成对齐的嵌入，并通过药物-疾病条件稀疏专家混合增强的变压器主干进行融合。这种设计明确支持治疗和设计子空间的专业化，同时通过top-k路由保持可扩展计算。在基准数据集上，MMCTOP在精确度、F1和AUC方面始终优于单模态和多模态基线，消融研究表明模式引导的文本化和选择性专家路由对性能和稳定性有实质性贡献。我们 additionally应用温度缩放以获得校准概率，确保下游决策支持的可靠风险估计。总体而言，MMCTOP通过结合受控的叙述规范化、上下文条件的专家融合和旨在提高生物医学信息学中可审计性和可重复性的操作保障，推进了多模态试验建模。


### 论文摘要

Addressing the challenge of multimodal data fusion in high-dimensional biomedical informatics, we propose MMCTOP, a MultiModal Clinical-Trial Outcome Prediction framework that integrates heterogeneous biomedical signals spanning (i) molecular structure representations, (ii) protocol metadata and long-form eligibility narratives, and (iii) disease ontologies. MMCTOP couples schema-guided textualization and input-fidelity validation with modality-aware representation learning, in which domain-specific encoders generate aligned embeddings that are fused by a transformer backbone augmented with a drug-disease-conditioned sparse Mixture-of-Experts (SMoE). This design explicitly supports specialization across therapeutic and design subspaces while maintaining scalable computation through top-k routing. MMCTOP achieves consistent improvements in precision, F1, and AUC over unimodal and multimodal baselines on benchmark datasets, and ablations show that schema-guided textualization and selective expert routing contribute materially to performance and stability. We additionally apply temperature scaling to obtain calibrated probabilities, ensuring reliable risk estimation for downstream decision support. Overall, MMCTOP advances multimodal trial modeling by combining controlled narrative normalization, context-conditioned expert fusion, and operational safeguards aimed at auditability and reproducibility in biomedical informatics.

---

## 28. TICON: A Slide-Level Tile Contextualizer for Histopathology Representation Learning

**论文链接:** [http://arxiv.org/abs/2512.21331v2](http://arxiv.org/abs/2512.21331v2)

**作者:** Varun Belagali, Saarthak Kapse, Pierre Marza, Srijan Das, Zilinghan Li, Sofiène Boutaj, Pushpak Pati, Srikar Yellapragada, Tarak Nath Nandi, Ravi K Madduri, Joel Saltz, Prateek Prasanna, Stergios Christodoulidis, Maria Vakalopoulou, Dimitris Samaras

**发布时间:** 2025-12-24

### GPT解析

### 总结

这篇论文介绍了一种名为TICON的新型transformer-based tile representation contextualizer，用于计算病理学中的全切片图像(WSI)分析。该模型能够为各种应用提供丰富的、具有上下文信息的嵌入表示，解决了标准tile编码器无法充分利用图像上下文信息的问题。

### 背景

在计算病理学中，对全切片图像(WSI)中的小tile进行解释通常需要更大的图像上下文。然而，标准的基于tile编码器的管道在提取嵌入时会剥离tile的上下文信息，无法建模对局部和全局任务都至关重要的丰富的幻灯片级信息。此外，不同的tile编码器在不同下游任务上表现出色，因此需要一个统一的模型来对来自任何tile级基础模型的嵌入进行上下文化。

### 目的

开发一个统一的模型，能够对来自任何tile级基础模型的嵌入进行上下文化，解决现有方法无法充分利用图像上下文信息的问题，并在各种计算病理学任务上提高性能。

### 方法

作者提出了TICON，一个基于transformer的tile表示上下文化器，它使用单一的共享编码器，通过掩码建模目标进行预训练，同时统一和丰富来自不同tile级病理学基础模型的表示。此外，作者还在TICON上预训练了一个聚合器，形成一个仅使用11K个WSI的slide级基础模型。

### 主要发现

实验表明，TICON上下文化的嵌入显著提高了许多不同任务的性能，在tile级基准测试(如HEST-Bench, THUNDER, CATCH)和slide级基准测试(如Patho-Bench)上建立了新的最先进结果。此外，仅使用11K个WSI预训练的TICON基础模型，优于使用多达350K个WSI预训练的最先进的slide级基础模型。

### 结论

TICON有效地解决了计算病理学中tile表示缺乏上下文信息的问题，提供了一个统一的框架来丰富来自任何tile级基础模型的嵌入。它在多个任务上取得了最先进的结果，并证明即使在较少的训练数据下也能构建高性能的slide级基础模型。

### 翻译

在大型全切片图像(WSI)中解释小tile通常需要更大的图像上下文。我们介绍了TICON，这是一种基于transformer的tile表示上下文化器，可以为计算病理学中的'任何'应用生成丰富、具有上下文化的嵌入。标准的基于tile编码器的管道提取剥离了上下文的tile嵌入，无法建模对局部和全局任务都至关重要的丰富的幻灯片级信息。此外，不同的tile编码器在不同的下游任务上表现出色。因此，需要一个统一的模型来对来自'任何'tile级基础模型的嵌入进行上下文化。TICON通过一个单一的共享编码器解决了这一需求，该编码器使用掩码建模目标进行预训练，同时统一和丰富来自不同tile级病理学基础模型的表示。我们的实验证明，TICON上下文化的嵌入显著提高了许多不同任务的性能，在tile级基准测试(即HEST-Bench, THUNDER, CATCH)和slide级基准测试(即Patho-Bench)上建立了新的最先进结果。最后，我们在TICON上预训练了一个聚合器，形成一个slide级基础模型，仅使用11K个WSI，就优于使用多达350K个WSI预训练的最先进slide级基础模型。


### 论文摘要

The interpretation of small tiles in large whole slide images (WSI) often needs a larger image context. We introduce TICON, a transformer-based tile representation contextualizer that produces rich, contextualized embeddings for ''any'' application in computational pathology. Standard tile encoder-based pipelines, which extract embeddings of tiles stripped from their context, fail to model the rich slide-level information essential for both local and global tasks. Furthermore, different tile-encoders excel at different downstream tasks. Therefore, a unified model is needed to contextualize embeddings derived from ''any'' tile-level foundation model. TICON addresses this need with a single, shared encoder, pretrained using a masked modeling objective to simultaneously unify and contextualize representations from diverse tile-level pathology foundation models. Our experiments demonstrate that TICON-contextualized embeddings significantly improve performance across many different tasks, establishing new state-of-the-art results on tile-level benchmarks (i.e., HEST-Bench, THUNDER, CATCH) and slide-level benchmarks (i.e., Patho-Bench). Finally, we pretrain an aggregator on TICON to form a slide-level foundation model, using only 11K WSIs, outperforming SoTA slide-level foundation models pretrained with up to 350K WSIs.

---

## 29. SpidR: Learning Fast and Stable Linguistic Units for Spoken Language Models Without Supervision

**论文链接:** [http://arxiv.org/abs/2512.20308v2](http://arxiv.org/abs/2512.20308v2)

**作者:** Maxime Poli, Mahi Luthra, Youssef Benchekroun, Yosuke Higuchi, Martin Gleize, Jiayi Shen, Robin Algayres, Yu-An Chung, Mido Assran, Juan Pino, Emmanuel Dupoux

**发布时间:** 2025-12-23

**备注:** Published in Transactions on Machine Learning Research. 30 pages, 16 figures

### GPT解析

### 总结

论文介绍了一种名为SpidR的自监督语音表征模型，可以直接从语音中学习语义表示，无需文本作为中介。该模型通过掩码预测目标结合自蒸馏和在线聚类进行训练，在下游语言建模任务上表现优异，且显著减少了预训练时间。

### 背景

语言建模和语音表征学习的并行进展使得直接从语音学习语言而不需要文本中间层成为可能。这需要直接从语音中提取语义表征。

### 目的

开发一种能够高效学习具有高度可访问语音信息表征的自监督语音表征模型，特别适合无文本的口语语言建模。

### 方法

1. 引入SpidR，一种自监督语音表征模型；2. 使用原始波形进行训练；3. 采用掩码预测目标结合自蒸馏和在线聚类；4. 学生模型的中间层学习预测来自教师模型中间层的分配；5. 这种学习目标稳定了在线聚类过程，产生更高质量的码本；6. 开源训练代码和模型检查点。

### 主要发现

1. SpidR在下游语言建模基准测试(sWUGGY, sBLIMP, tSC)上优于wav2vec 2.0, HuBERT, WavLM和DinoSR；2. 系统评估了模型和层之间语音单元质量(ABX, PNMI)与语言建模性能之间的相关性，验证了这些指标作为可靠代理的有效性；3. SpidR显著减少了预训练时间，相比HuBERT只需要16 GPU上预训练一天，而不是一周。

### 结论

SpidR是一种高效的自监督语音表征模型，能够直接从语音中学习语义表示，在性能和训练效率上都优于现有方法。

### 翻译

语言建模和语音表征学习的并行进展提出了直接从语音学习语言而不需要文本中间层的前景。这需要直接从语音中提取语义表征。我们的贡献有三方面。首先，我们引入了SpidR，一种自监督语音表征模型，它高效地学习具有高度可访问语音信息的表征，这使其特别适合无文本的口语语言建模。它使用掩码预测目标结合自蒸馏和在线聚类在原始波形上进行训练。学生模型的中间层学习预测来自教师模型中间层的分配。与以往方法相比，这种学习目标稳定了在线聚类过程，产生了更高质量的码本。在下游语言建模基准测试(sWUGGY, sBLIMP, tSC)上，SpidR优于wav2vec 2.0、HuBERT、WavLM和DinoSR。其次，我们系统性地评估了模型和层之间语音单元质量(ABX, PNMI)与语言建模性能之间的相关性，验证了这些指标作为可靠代理的有效性。最后，与HuBERT相比，SpidR显著减少了预训练时间，只需要在16 GPU上预训练一天，而不是一周。这种加速是由预训练方法和高效代码库实现的，它允许更快的迭代和更容易的实验。我们在https://github.com/facebookresearch/spidr开源了训练代码和模型检查点。


### 论文摘要

The parallel advances in language modeling and speech representation learning have raised the prospect of learning language directly from speech without textual intermediates. This requires extracting semantic representations directly from speech. Our contributions are threefold. First, we introduce SpidR, a self-supervised speech representation model that efficiently learns representations with highly accessible phonetic information, which makes it particularly suited for textless spoken language modeling. It is trained on raw waveforms using a masked prediction objective combined with self-distillation and online clustering. The intermediate layers of the student model learn to predict assignments derived from the teacher's intermediate layers. This learning objective stabilizes the online clustering procedure compared to previous approaches, resulting in higher quality codebooks. SpidR outperforms wav2vec 2.0, HuBERT, WavLM, and DinoSR on downstream language modeling benchmarks (sWUGGY, sBLIMP, tSC). Second, we systematically evaluate across models and layers the correlation between speech unit quality (ABX, PNMI) and language modeling performance, validating these metrics as reliable proxies. Finally, SpidR significantly reduces pretraining time compared to HuBERT, requiring only one day of pretraining on 16 GPUs, instead of a week. This speedup is enabled by the pretraining method and an efficient codebase, which allows faster iteration and easier experimentation. We open-source the training code and model checkpoints at https://github.com/facebookresearch/spidr.

---

## 30. Meta-Learning-Based Handover Management in NextG O-RAN

**论文链接:** [http://arxiv.org/abs/2512.22022v1](http://arxiv.org/abs/2512.22022v1)

**作者:** Michail Kalntis, George Iosifidis, José Suárez-Varela, Andra Lutu, Fernando A. Kuipers

**发布时间:** 2025-12-26

### GPT解析

### 总结

CONTRA是一种创新的框架，首次在O-RAN架构中联合优化传统切换和条件切换，通过元学习算法实现自适应切换决策，提高网络性能，在实际网络环境中表现优于现有方法。

### 背景

传统切换(THOs)曾是移动连接的支柱，但在密集部署和高频段中越来越容易失败和延迟。3GPP引入了条件切换(CHOs)，可以主动预留小区和用户驱动的执行，但两种切换类型在信令、资源使用和可靠性方面存在复杂的权衡。

### 目的

提出新的全国性移动管理数据集，提供对这些问题的见解，并呼吁在下一代网络中采用自适应和鲁棒的切换控制。

### 方法

提出CONTRA框架，研究两种变体：一种预先分配用户到特定切换类型，另一种动态决定切换类型。使用元学习算法适应运行时观察，保证性能接近拥有完美未来信息的预言机。CONTRA设计为近实时部署为O-RAN xApp，符合6G灵活和智能控制的目标。

### 主要发现

利用众包数据集进行的广泛评估显示，CONTRA提高了用户吞吐量，减少了THO和CHO的切换成本，在动态和真实场景中优于3GPP兼容和强化学习基线。

### 结论

CONTRA框架有效地解决了传统切换和条件切换的权衡问题，通过自适应和智能的切换控制，提高了网络性能。

### 翻译

虽然传统切换(THOs)一直是移动连接的支柱，但它们在密集部署和高频段中越来越容易遭受故障和延迟。为解决这些限制，3GPP引入了条件切换(CHOs)，可以主动预留小区和用户驱动的执行。然而，两种切换类型在信令、资源使用和可靠性方面呈现复杂的权衡。本文提出了一家顶级移动网络运营商(MNO)提供的独特全国性移动管理数据集，为这些问题提供了新的见解，并呼吁在下一代网络中采用自适应和鲁棒的切换控制。受这些发现的启发，我们提出了CONTRA框架，首次在O-RAN架构中联合优化THOs和CHOs。我们研究了CONTRA的两种变体：一种用户预先分配到一种切换类型，反映不同的服务或用户特定需求，以及一种更动态的公式，控制器根据系统条件和需求即时决定切换类型。为此，它依靠一种实用的元学习算法，适应运行时观察，并保证性能接近拥有完美未来信息(无遗憾)的预言机。CONTRA专为近实时部署设计，作为O-RAN xApp，符合6G灵活和智能控制的目标。利用众包数据集进行的广泛评估显示，CONTRA提高了用户吞吐量并减少了THO和CHO的切换成本，在动态和真实场景中优于3GPP兼容和强化学习(RL)基线。


### 论文摘要

While traditional handovers (THOs) have served as a backbone for mobile connectivity, they increasingly suffer from failures and delays, especially in dense deployments and high-frequency bands. To address these limitations, 3GPP introduced Conditional Handovers (CHOs) that enable proactive cell reservations and user-driven execution. However, both handover (HO) types present intricate trade-offs in signaling, resource usage, and reliability. This paper presents unique, countrywide mobility management datasets from a top-tier mobile network operator (MNO) that offer fresh insights into these issues and call for adaptive and robust HO control in next-generation networks. Motivated by these findings, we propose CONTRA, a framework that, for the first time, jointly optimizes THOs and CHOs within the O-RAN architecture. We study two variants of CONTRA: one where users are a priori assigned to one of the HO types, reflecting distinct service or user-specific requirements, as well as a more dynamic formulation where the controller decides on-the-fly the HO type, based on system conditions and needs. To this end, it relies on a practical meta-learning algorithm that adapts to runtime observations and guarantees performance comparable to an oracle with perfect future information (universal no-regret). CONTRA is specifically designed for near-real-time deployment as an O-RAN xApp and aligns with the 6G goals of flexible and intelligent control. Extensive evaluations leveraging crowdsourced datasets show that CONTRA improves user throughput and reduces both THO and CHO switching costs, outperforming 3GPP-compliant and Reinforcement Learning (RL) baselines in dynamic and real-world scenarios.

---

## 31. MAD-NG: Meta-Auto-Decoder Neural Galerkin Method for Solving Parametric Partial Differential Equations

**论文链接:** [http://arxiv.org/abs/2512.21633v1](http://arxiv.org/abs/2512.21633v1)

**作者:** Qiuqi Li, Yiting Liu, Jin Zhao, Wencan Zhu

**发布时间:** 2025-12-25

### GPT解析

### 总结

本文提出了一种结合元自解码器(MAD)范式的神经Galerkin方法(NGM)增强框架，通过时空解耦和元学习驱动的适应，实现了对参数化偏微分方程的高效求解，显著降低了计算成本同时保持了高准确性。

### 背景

参数化偏微分方程(PDEs)是建模受不确定或变化参数影响的物理和工程系统的基本工具。传统基于神经网络的求解器（如物理信息神经网络PINNs和深度Galerkin方法）由于依赖全时空近似，在泛化能力和长时间预测效率方面面临挑战。

### 目的

解决传统神经网络求解器在泛化和长时间预测效率方面的问题，开发一种能够实现物理一致性长时程预测且计算开销显著降低的方法。

### 方法

提出一种新颖且可扩展的框架，通过结合元自解码器(MAD)范式增强神经Galerkin方法(NGM)。该方法利用时空解耦实现更稳定高效的时间积分，采用元学习驱动的适应实现快速泛化，并应用随机稀疏更新降低计算成本。

### 主要发现

所提出的方法能够以显著降低的计算开销实现复杂参数化演化方程的物理一致性、长时程预测。数值实验表明，该方法在准确性、鲁棒性和适应性方面表现良好。

### 结论

结合元自解码器范式、时空解耦、元学习驱动适应和随机稀疏更新的框架，有效解决了传统神经网络求解器在参数化偏微分方程求解中的泛化和效率问题，为复杂物理系统的建模提供了高效工具。

### 翻译

参数化偏微分方程(PDEs)是建模受不确定或变化参数影响的广泛物理和工程系统的基本工具。传统的基于神经网络的求解器，如物理信息神经网络(PINNs)和深度Galerkin方法，由于依赖全时空近似，常常面临泛化和长时间预测效率方面的挑战。为解决这些问题，我们提出了一种新颖且可扩展的框架，通过结合元自解码器(MAD)范式显著增强了神经Galerkin方法(NGM)。我们的方法利用时空解耦实现更稳定高效的时间积分，而元学习驱动的适应允许对未见过的参数配置进行快速泛化，且只需最少量的再训练。此外，随机稀疏更新能有效降低计算成本而不损害准确性。这些进步共同使该方法能够以显著降低的计算开销，实现复杂参数化演化方程的物理一致性、长时程预测。在基准问题上的数值实验表明，我们的方法在准确性、鲁棒性和适应性方面表现良好。


### 论文摘要

Parametric partial differential equations (PDEs) are fundamental for modeling a wide range of physical and engineering systems influenced by uncertain or varying parameters. Traditional neural network-based solvers, such as Physics-Informed Neural Networks (PINNs) and Deep Galerkin Methods, often face challenges in generalization and long-time prediction efficiency due to their dependence on full space-time approximations. To address these issues, we propose a novel and scalable framework that significantly enhances the Neural Galerkin Method (NGM) by incorporating the Meta-Auto-Decoder (MAD) paradigm. Our approach leverages space-time decoupling to enable more stable and efficient time integration, while meta-learning-driven adaptation allows rapid generalization to unseen parameter configurations with minimal retraining. Furthermore, randomized sparse updates effectively reduce computational costs without compromising accuracy. Together, these advancements enable our method to achieve physically consistent, long-horizon predictions for complex parameterized evolution equations with significantly lower computational overhead. Numerical experiments on benchmark problems demonstrate that our methods performs comparatively well in terms of accuracy, robustness, and adaptability.

---

## 32. Discovering Sparse Recovery Algorithms Using Neural Architecture Search

**论文链接:** [http://arxiv.org/abs/2512.21563v1](http://arxiv.org/abs/2512.21563v1)

**作者:** Patrick Yubeaton, Sarthak Gupta, M. Salman Asif, Chinmay Hegde

**发布时间:** 2025-12-25

**备注:** Presented at the 59th Asilomar Conference on Signals, Systems, and Computers

### GPT解析

### 总结

该论文探讨了利用元学习工具（如神经架构搜索NAS）来自动发现信号处理中反问题求解算法的方法，并以ISTA和FISTA算法为例验证了这一方法的可行性。

### 背景

设计用于解决信号处理中反问题的新算法是一项极其困难、依赖启发式方法且耗时的任务。

### 目的

探索在信号处理领域中通过元学习工具实现算法自动发现的思路。

### 方法

开发了一个元学习框架，在包含超过50,000个变量的搜索空间中重新发现ISTA和FISTA算法的关键元素，并验证了该框架的通用性。

### 主要发现

元学习框架能够成功地重新发现ISTA和FISTA算法的关键组成部分，证明自动化算法发现在信号处理领域是可行的。

### 结论

通过元学习工具可以实现信号处理中反问题求解算法的自动发现，这种方法具有通用性，可应用于不同的数据分布和算法类型。

### 翻译

为解决信号处理中的反问题设计新颖算法是一项极其困难、依赖启发式方法且耗时的任务。在这篇简短的文章中，我们探讨了通过元学习工具（如神经架构搜索NAS）在信号处理环境中实现算法自动发现的思路。具体而言，我们将迭代收缩阈值算法(ISTA)及其加速版本快速ISTA(FISTA)作为算法重新发现的候选对象。我们开发了一个元学习框架，在包含超过50,000个变量的搜索空间中，能够重新发现上述两种算法的几个关键元素。然后，我们展示了我们的框架如何应用于ISTA/FISTA之外的各种数据分布和算法。


### 论文摘要

The design of novel algorithms for solving inverse problems in signal processing is an incredibly difficult, heuristic-driven, and time-consuming task. In this short paper, we the idea of automated algorithm discovery in the signal processing context through meta-learning tools such as Neural Architecture Search (NAS). Specifically, we examine the Iterative Shrinkage Thresholding Algorithm (ISTA) and its accelerated Fast ISTA (FISTA) variant as candidates for algorithm rediscovery. We develop a meta-learning framework which is capable of rediscovering (several key elements of) the two aforementioned algorithms when given a search space of over 50,000 variables. We then show how our framework can apply to various data distributions and algorithms besides ISTA/FISTA.

---

## 33. Backdoor Attacks on Prompt-Driven Video Segmentation Foundation Models

**论文链接:** [http://arxiv.org/abs/2512.22046v1](http://arxiv.org/abs/2512.22046v1)

**作者:** Zongmin Zhang, Zhen Sun, Yifan Liao, Wenhan Dong, Xinlei He, Xingshuo Han, Shengmin Xu, Xinyi Huang

**发布时间:** 2025-12-26

### GPT解析

### 总结

研究提出了BadVSFM，第一个专门针对提示驱动视频分割基础模型的后门攻击框架，解决了传统后门攻击在VSFM上效果不佳的问题。

### 背景

提示驱动的视频分割基础模型如SAM2正被越来越多地应用于自动驾驶和数字病理等领域，引发了关于后门威胁的担忧。

### 目的

理解并解决传统后门攻击在VSFM上效果不佳的问题，开发专门针对VSFM的后门攻击框架。

### 方法

提出BadVSFM框架，采用两阶段策略：(1)引导图像编码器使触发帧映射到目标嵌入而干净帧保持对齐；(2)训练掩码解码器使触发帧-提示对产生共享目标掩码而干净输出接近参考解码器。

### 主要发现

BadVSFM在两个数据集和五个VSFM上实现了强大且可控的后门效果，同时保持干净分割质量；消融实验证实了两阶段设计的必要性；梯度冲突分析显示BadVSFM分离了触发和干净表示；四种代表性防御措施效果有限。

### 结论

BadVSFM揭示了当前VSFM中一个未被充分探索的安全漏洞，为理解VSFM的安全特性提供了重要见解。

### 翻译

提示驱动的视频分割基础模型如SAM2正越来越多地应用于自动驾驶和数字病理等领域，引发了对后门威胁的担忧。令人惊讶的是，我们发现直接将经典后门攻击(如BadNet)转移到VSFM上几乎无效，攻击成功率低于5%。为理解这一点，我们研究了编码器梯度和注意力图，观察到传统训练使得干净样本和触发样本的梯度保持大致对齐，而注意力仍然集中在真实对象上，阻止编码器学习与触发相关的表示。为应对这一挑战，我们提出了BadVSFM，这是第一个专门针对提示驱动VSFM的后门框架。BadVSFM采用两阶段策略：(1)引导图像编码器，使触发帧映射到指定的目标嵌入，同时干净帧与干净的参考编码器保持对齐；(2)训练掩码解码器，使得跨提示类型的触发帧-提示对产生共享的目标掩码，同时干净输出接近参考解码器。在两个数据集和五个VSFM上的大量实验表明，BadVSFM在各种触发器和提示下实现了强大且可控的后门效果，同时保持了干净分割的质量。对损失、阶段、目标、触发设置和中毒率的消融实验表明，对合理的超参数变化具有鲁棒性，并证实了两阶段设计的必要性。最后，梯度冲突分析和注意力可视化显示，BadVSFM分离了触发和干净表示，并将注意力转移到触发区域，而四种代表性防御措施在很大程度上仍然无效，揭示了当前VSFM中一个未被充分探索的漏洞。


### 论文摘要

Prompt-driven Video Segmentation Foundation Models (VSFMs) such as SAM2 are increasingly deployed in applications like autonomous driving and digital pathology, raising concerns about backdoor threats. Surprisingly, we find that directly transferring classic backdoor attacks (e.g., BadNet) to VSFMs is almost ineffective, with ASR below 5\%. To understand this, we study encoder gradients and attention maps and observe that conventional training keeps gradients for clean and triggered samples largely aligned, while attention still focuses on the true object, preventing the encoder from learning a distinct trigger-related representation. To address this challenge, we propose BadVSFM, the first backdoor framework tailored to prompt-driven VSFMs. BadVSFM uses a two-stage strategy: (1) steer the image encoder so triggered frames map to a designated target embedding while clean frames remain aligned with a clean reference encoder; (2) train the mask decoder so that, across prompt types, triggered frame-prompt pairs produce a shared target mask, while clean outputs stay close to a reference decoder. Extensive experiments on two datasets and five VSFMs show that BadVSFM achieves strong, controllable backdoor effects under diverse triggers and prompts while preserving clean segmentation quality. Ablations over losses, stages, targets, trigger settings, and poisoning rates demonstrate robustness to reasonable hyperparameter changes and confirm the necessity of the two-stage design. Finally, gradient-conflict analysis and attention visualizations show that BadVSFM separates triggered and clean representations and shifts attention to trigger regions, while four representative defenses remain largely ineffective, revealing an underexplored vulnerability in current VSFMs.

---

## 34. StereoVLA: Enhancing Vision-Language-Action Models with Stereo Vision

**论文链接:** [http://arxiv.org/abs/2512.21970v1](http://arxiv.org/abs/2512.21970v1)

**作者:** Shengliang Deng, Mi Yan, Yixin Zheng, Jiayi Su, Wenhao Zhang, Xiaoguang Zhao, Heming Cui, Zhizheng Zhang, He Wang

**发布时间:** 2025-12-26

### GPT解析

### 总结

本文提出了StereoVLA模型，一种利用立体视觉几何线索的视觉-语言-动作模型，通过几何-语义特征提取模块和交互区域深度估计任务提升空间感知和指令跟随能力。

### 背景

立体相机模仿人类双目视觉，为机器人精确操作提供丰富的空间线索，但立体视觉在视觉-语言-动作模型中的应用仍未被充分探索。

### 目的

开发一种能够利用立体视觉提供的丰富几何线索的视觉-语言-动作模型，提升机器人在精确操作任务中的性能。

### 方法

提出几何-语义特征提取模块，利用视觉基础模型提取并融合几何特征(来自立体视图差异)和语义特征(来自单视图)；同时引入辅助的交互区域深度估计任务增强空间感知并加速模型收敛。

### 主要发现

在立体设置下的多样化任务中，StereoVLA显著优于基线模型，且对相机姿态变化表现出强大的鲁棒性。

### 结论

通过有效利用立体视觉的几何信息，StereoVLA模型提升了视觉-语言-动作模型在机器人操作任务中的性能和鲁棒性。

### 翻译

立体相机 closely mimic human binocular vision, providing rich spatial cues critical for precise robotic manipulation. Despite their advantage, the adoption of stereo vision in vision-language-action models (VLAs) remains underexplored. In this work, we present StereoVLA, a VLA model that leverages rich geometric cues from stereo vision. We propose a novel Geometric-Semantic Feature Extraction module that utilizes vision foundation models to extract and fuse two key features: geometric features from subtle stereo-view differences for spatial perception; semantic-rich features from the monocular view for instruction following. Additionally, we propose an auxiliary Interaction-Region Depth Estimation task to further enhance spatial perception and accelerate model convergence. Extensive experiments show that our approach outperforms baselines by a large margin in diverse tasks under the stereo setting and demonstrates strong robustness to camera pose variations.

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决如何在视觉-语言-行动模型（VLAs）中有效利用立体视觉来增强机器人的空间感知和操作能力的问题。这个问题很重要，因为当前大多数VLAs依赖单目RGB图像，存在深度模糊问题，难以精确判断物体距离和空间关系，而这对精确的机器人操作至关重要。立体视觉能模拟人类双眼视觉，提供丰富的空间线索，且相比其他补充传感器（如手腕相机、深度传感器）具有视野更广、噪声更小、硬件更简单等优势。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先分析了现有VLAs模型在空间感知方面的局限性以及各种补充传感器的缺点，从人类视觉系统获得启发，认识到立体视觉能提供丰富的空间信息。在设计方法时，作者借鉴了FoundationStereo模型提取几何特征，借鉴了PrismaticVLM提取语义特征，参考了GraspVLA的训练和数据生成策略，并利用InternLM-1.8B作为大语言模型骨干。在此基础上，作者创新性地提出了几何-语义特征提取模块和交互区域深度估计任务，将几何特征和语义特征有效融合，使模型既具备几何精度又保持语义理解能力。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是通过立体视觉提供丰富的几何信息，同时保持强大的语义理解能力，实现精确的机器人操作。具体流程为：1) 接收一对立体图像，使用FoundationStereo提取几何特征，使用SigLIP和DINOv2提取语义特征；2) 将几何和语义特征在空间上对齐并通过通道级连接融合成混合视觉token；3) 将视觉token与语言token一起输入到大语言模型进行联合处理；4) 使用动作专家通过flow-matching预测末端执行器姿态；5) 在训练过程中引入交互区域深度估计作为辅助任务，仅在夹爪和物体周围采样点预测深度，提高训练效率。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1) 首个系统利用立体视觉增强的VLAs模型；2) 几何-语义特征提取模块，融合立体视觉的几何特征和单目视图的语义特征；3) 交互区域深度估计任务，专注于操作相关区域；4) 解决了直接将立体输入提供给现有多相机VLAs时性能不佳的问题。相比之前工作，StereoVLA使用立体相机而非单目或多相机设置，专门设计了特征提取与融合方法，引入了针对性的深度估计任务，在各种任务上表现更优，特别是在高精度任务上，且对相机姿态变化更具鲁棒性。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'StereoVLA通过创新的几何-语义特征提取方法和交互区域深度估计任务，首次将立体视觉系统性地整合到视觉-语言-行动模型中，显著提升了机器人在精确操作任务中的性能和对相机姿态变化的鲁棒性。'}


### 论文摘要

Stereo cameras closely mimic human binocular vision, providing rich spatial cues critical for precise robotic manipulation. Despite their advantage, the adoption of stereo vision in vision-language-action models (VLAs) remains underexplored. In this work, we present StereoVLA, a VLA model that leverages rich geometric cues from stereo vision. We propose a novel Geometric-Semantic Feature Extraction module that utilizes vision foundation models to extract and fuse two key features: 1) geometric features from subtle stereo-view differences for spatial perception; 2) semantic-rich features from the monocular view for instruction following. Additionally, we propose an auxiliary Interaction-Region Depth Estimation task to further enhance spatial perception and accelerate model convergence. Extensive experiments show that our approach outperforms baselines by a large margin in diverse tasks under the stereo setting and demonstrates strong robustness to camera pose variations.

---

## 35. SLIM-Brain: A Data- and Training-Efficient Foundation Model for fMRI Data Analysis

**论文链接:** [http://arxiv.org/abs/2512.21881v1](http://arxiv.org/abs/2512.21881v1)

**作者:** Mo Wang, Junfeng Xia, Wenhao Ye, Enyu Liu, Kaining Peng, Jianfeng Feng, Quanying Liu, Hongkai Wen

**发布时间:** 2025-12-26

**备注:** The code will be released after review

### GPT解析

### 总结

论文介绍了SLIM-Brain模型，一种新型无图谱基础模型，通过两阶段自适应设计同时提高fMRI分析中的数据和训练效率，在七个公共基准上取得最先进性能，仅需4千预训练会话和30%的GPU内存。

### 背景

基础模型正成为fMRI分析的强大范式，但当前方法面临双重瓶颈：基于图谱的方法降低数据维度但丢弃空间细节且需大训练数据；无图谱方法保留空间保真度但内存和计算强度过高，使大规模预训练不可行。

### 目的

引入SLIM-Brain(Sample-efficient, Low-memory fMRI Foundation Model for Human Brain)，一种新的无图谱基础模型，同时提高数据和训练效率。

### 方法

采用两阶段自适应设计：(i)轻量级时间提取器捕获完整序列的全局上下文并根据显著性对数据窗口排序；(ii)4D分层编码器(Hiera-JEPA)仅从前k个选定窗口学习精细体素级表示，同时删除约70%掩码块。

### 主要发现

在七个公共基准上的广泛实验表明，SLIM-Brain在各种任务上建立了新的最先进性能，仅需4千个预训练会话，相比传统体素级方法大约只需要30%的GPU内存。

### 结论

SLIM-Brain模型在保持高性能的同时显著提高了数据效率和训练效率，为fMRI分析提供了一种新的有效方法。

### 翻译

基础模型正在成为fMRI分析的一个强大范式，但当前方法面临数据和训练效率的双重瓶颈。基于图谱的方法将体素信号聚合成固定的感兴趣区域，降低了数据维度但丢弃了精细的空间细节，并且需要非常大的队列才能作为通用基础模型进行有效训练。另一方面，无图谱方法直接在体素级别上操作，保留了空间保真度，但内存和计算强度极高，使得大规模预训练不可行。我们引入SLIM-Brain(高效样本、低内存人脑fMRI基础模型)，一种新的无图谱基础模型，同时提高了数据和训练效率。SLIM-Brain采用两阶段自适应设计：(i)轻量级时间提取器捕获完整序列的全局上下文，并根据显著性对数据窗口进行排序；(ii)4D分层编码器(Hiera-JEPA)仅从前k个选定的窗口中学习精细的体素级表示，同时删除约70%的掩码块。在七个公共基准上的广泛实验表明，SLIM-Brain在各种任务上建立了新的最先进性能，同时仅需4千个预训练会话，并且相比传统的体素级方法，大约只需要30%的GPU内存。


### 论文摘要

Foundation models are emerging as a powerful paradigm for fMRI analysis, but current approaches face a dual bottleneck of data- and training-efficiency. Atlas-based methods aggregate voxel signals into fixed regions of interest, reducing data dimensionality but discarding fine-grained spatial details, and requiring extremely large cohorts to train effectively as general-purpose foundation models. Atlas-free methods, on the other hand, operate directly on voxel-level information - preserving spatial fidelity but are prohibitively memory- and compute-intensive, making large-scale pre-training infeasible. We introduce SLIM-Brain (Sample-efficient, Low-memory fMRI Foundation Model for Human Brain), a new atlas-free foundation model that simultaneously improves both data- and training-efficiency. SLIM-Brain adopts a two-stage adaptive design: (i) a lightweight temporal extractor captures global context across full sequences and ranks data windows by saliency, and (ii) a 4D hierarchical encoder (Hiera-JEPA) learns fine-grained voxel-level representations only from the top-$k$ selected windows, while deleting about 70% masked patches. Extensive experiments across seven public benchmarks show that SLIM-Brain establishes new state-of-the-art performance on diverse tasks, while requiring only 4 thousand pre-training sessions and approximately 30% of GPU memory comparing to traditional voxel-level methods.

---

## 36. Training-free Conditional Image Embedding Framework Leveraging Large Vision Language Models

**论文链接:** [http://arxiv.org/abs/2512.21860v1](http://arxiv.org/abs/2512.21860v1)

**作者:** Masayuki Kawarada, Kosuke Yamada, Antonio Tejero-de-Pablos, Naoto Inoue

**发布时间:** 2025-12-26

### GPT解析

### 总结

本文提出了DIOR方法，一种利用大型视觉-语言模型生成条件图像嵌入的无需训练方法。该方法通过提示模型用与给定条件相关的单个单词描述图像，提取最后一个隐藏状态向量作为条件图像嵌入。

### 背景

条件图像嵌入是指专注于图像中由给定文本条件(如颜色、类型)指出的特定方面的特征表示，这是一个具有挑战性的问题。虽然最近的视觉基础模型如CLIP提供了丰富的图像表示，但它们并非设计用来专注于指定条件。

### 目的

提出一种方法，利用大型视觉-语言模型生成条件图像嵌入，使其能够专注于图像中由给定文本条件指定的特定方面。

### 方法

DIOR是一种无需训练的方法，它通过提示大型视觉-语言模型(LVLM)用与给定条件相关的单个单词描述图像，然后提取LVLM最后一个隐藏状态向量作为条件图像嵌入。

### 主要发现

DIOR在条件图像相似度任务上表现优于现有无需训练的基线方法，包括CLIP；在多个设置下也优于需要额外训练的方法；提供了通用解决方案，可应用于任何图像和条件，无需额外训练或任务特定先验。

### 结论

DIOR是一种有效的条件图像嵌入生成方法，无需训练即可实现高性能，且具有广泛的适用性。

### 翻译

条件图像嵌入是专注于图像中由给定文本条件(如颜色、类型)指出的特定方面的特征表示，这是一个具有挑战性的问题。虽然最近的视觉基础模型，如CLIP，提供了丰富的图像表示，但它们并非设计用来专注于指定条件。在本文中，我们提出了DIOR，一种利用大型视觉-语言模型生成条件图像嵌入的方法。DIOR是一种无需训练的方法，它提示模型用与给定条件相关的单个单词描述图像，然后将最后一个标记的隐藏状态向量提取为条件图像嵌入。DIOR提供了通用解决方案，可应用于任何图像和条件，无需额外训练或任务特定先验。在条件图像相似度任务上的全面实验结果表明，DIOR优于现有无需训练的基线方法，包括CLIP。此外，DIOR在多个设置下也优于需要额外训练的方法。


### 论文摘要

Conditional image embeddings are feature representations that focus on specific aspects of an image indicated by a given textual condition (e.g., color, genre), which has been a challenging problem. Although recent vision foundation models, such as CLIP, offer rich representations of images, they are not designed to focus on a specified condition. In this paper, we propose DIOR, a method that leverages a large vision-language model (LVLM) to generate conditional image embeddings. DIOR is a training-free approach that prompts the LVLM to describe an image with a single word related to a given condition. The hidden state vector of the LVLM's last token is then extracted as the conditional image embedding. DIOR provides a versatile solution that can be applied to any image and condition without additional training or task-specific priors. Comprehensive experimental results on conditional image similarity tasks demonstrate that DIOR outperforms existing training-free baselines, including CLIP. Furthermore, DIOR achieves superior performance compared to methods that require additional training across multiple settings.

---

## 37. Hyperion: Low-Latency Ultra-HD Video Analytics via Collaborative Vision Transformer Inference

**论文链接:** [http://arxiv.org/abs/2512.21730v1](http://arxiv.org/abs/2512.21730v1)

**作者:** Linyi Jiang, Yifei Zhu, Hao Yin, Bo Li

**发布时间:** 2025-12-25

**备注:** Accepted for publication in IEEE INFOCOM 2026

### GPT解析

### 总结

本文提出了Hyperion，第一个云设备协作框架，用于在动态网络上使用现成的视觉transformer对超高清视觉数据进行低延迟推理。

### 背景

阵列相机摄影技术可实时捕捉超高清视频，提供广阔视野中的丰富视觉信息，但使用基于transformer的视觉基础模型处理此类数据时，在设备计算或云计算中面临显著的计算或传输开销。

### 目的

开发一个框架，解决超高清视觉数据处理中的计算和传输瓶颈问题，实现低延迟推理。

### 方法

Hyperion整合了三个关键组件：协作感知的重要性评分器识别关键区域；动态调度器自适应调整块传输质量以平衡延迟和准确性；加权集成器融合边缘和云结果提高准确性。

### 主要发现

实验结果表明，与各种网络环境下的最先进基线相比，Hyperion将帧处理速率提高了最多1.61倍，准确性提高了最多20.2%。

### 结论

Hyperion有效解决了超高清视觉transformer的计算和传输瓶颈，在保持高准确性的同时显著提高了处理速度。

### 翻译

最近阵列相机摄影技术的进步能够实时捕捉超高清视频，提供了广阔视野中的丰富视觉信息。然而，使用最先进的基于transformer的视觉基础模型及时处理此类数据，在设备计算中面临显著的计算开销，或在云计算中面临传输开销。在本文中，我们提出了Hyperion，这是第一个云设备协作框架，能够在动态网络上使用现成的视觉transformer对超高清视觉数据进行低延迟推理。Hyperion利用视觉Transformer模型中的固有特性，解决了超高清视觉transformer的计算和传输瓶颈。具体而言，Hyperion集成了一个协作感知的重要性评分器，用于识别块级别的关键区域；一个动态调度器，能够自适应调整块传输质量，在动态网络条件下平衡延迟和准确性；以及一个加权集成器，融合边缘和云结果以提高准确性。实验结果表明，与各种网络环境下的最先进基线相比，Hyperion将帧处理速率提高了最多1.61倍，准确性提高了最多20.2%。


### 论文摘要

Recent advancements in array-camera videography enable real-time capturing of ultra-high-definition (Ultra-HD) videos, providing rich visual information in a large field of view. However, promptly processing such data using state-of-the-art transformer-based vision foundation models faces significant computational overhead in on-device computing or transmission overhead in cloud computing. In this paper, we present Hyperion, the first cloud-device collaborative framework that enables low-latency inference on Ultra-HD vision data using off-the-shelf vision transformers over dynamic networks. Hyperion addresses the computational and transmission bottleneck of Ultra-HD vision transformers by exploiting the intrinsic property in vision Transformer models. Specifically, Hyperion integrates a collaboration-aware importance scorer that identifies critical regions at the patch level, a dynamic scheduler that adaptively adjusts patch transmission quality to balance latency and accuracy under dynamic network conditions, and a weighted ensembler that fuses edge and cloud results to improve accuracy. Experimental results demonstrate that Hyperion enhances frame processing rate by up to 1.61 times and improves the accuracy by up to 20.2% when compared with state-of-the-art baselines under various network environments.

---

## 38. Linear Foundation Model for Quantum Embedding: Data-Driven Compression of the Ghost Gutzwiller Variational Space

**论文链接:** [http://arxiv.org/abs/2512.21666v1](http://arxiv.org/abs/2512.21666v1)

**作者:** Samuele Giuli, Hasanat Hasan, Benedikt Kloss, Marius S. Frank, Tsung-Han Lee, Olivier Gingras, Yong-Xin Yao, Nicola Lanatà

**发布时间:** 2025-12-25

**备注:** 17 pages, 6 figures

### GPT解析

### 总结

研究引入了一种基于主成分分析的量子嵌入线性基础模型，通过数据驱动的主动学习方案显著降低了量子嵌入理论中的计算瓶颈，实现了强关联材料的高效模拟。

### 背景

Kohn-Sham密度泛函理论(DFT)在模拟量子物质时被广泛使用，但对强关联系统往往失效。量子嵌入(QE)理论通过映射系统到辅助嵌入哈密顿量(EH)来解决这一局限，但EH通常很大，其迭代解算是主要的计算瓶颈。

### 目的

开发一种QE的线性基础模型，利用主成分分析(PCA)压缩解决EH所需的量子态空间，在小变分子子空间内高效解决EH问题。

### 方法

采用数据驱动的主动学习方案从EH基态学习变分子空间，将嵌入解简化为约化空间中的确定性基态本征值问题。在鬼Gutzwiller近似(ghost-GA)下对三轨道Hubbard模型进行研究，并在钚元素上进行验证。

### 主要发现

在Bethe晶格上学习的变分空间可转移到方形和立方晶格无需额外训练；显著降低了EH步骤的计算成本；在钚元素上，单个变分空间可重现所有六个结晶相的能量学，同时将EH解决方案成本降低几个数量级。

### 结论

该方法为克服QE框架的主要计算瓶颈提供了实用途径，使强关联材料的高通量从头算模拟接近DFT成本，为量子物质研究开辟了新方向。

### 翻译

量子物质的模拟主要依赖Kohn-Sham密度泛函理论(DFT)，但该方法通常对强关联系统失效。量子嵌入(QE)理论通过将系统映射到描述片段-环境相互作用的辅助嵌入哈密顿量(EH)来解决这一局限，但EH通常很大，其迭代解算是主要的计算瓶颈。我们引入了一种QE的线性基础模型，利用主成分分析(PCA)压缩解决EH所需的量子态空间，从而在小变分子子空间内解决EH问题。通过数据驱动的主动学习方案，我们从EH基态学习这个子空间，将每个嵌入解简化为约化空间中的确定性基态本征值问题。在鬼Gutzwiller近似(ghost-GA)下，我们对三轨道Hubbard模型的研究表明，在Bethe晶格上学习的变分空间可以转移到方形和立方晶格上而无需额外训练，同时显著降低了EH步骤的成本。我们进一步在钚元素上验证了该方法，单个变分空间可重现所有六个结晶相的能量学，同时将EH解决方案的成本降低了几个数量级。这为克服QE框架的主要计算瓶颈提供了实用途径，为强关联材料的高通量从头算模拟开辟了道路，使其成本接近DFT。


### 论文摘要

Simulations of quantum matter rely mainly on Kohn-Sham density functional theory (DFT), which often fails for strongly correlated systems. Quantum embedding (QE) theories address this limitation by mapping the system onto an auxiliary embedding Hamiltonian (EH) describing fragment-environment interactions, but the EH is typically large and its iterative solution is the primary computational bottleneck. We introduce a linear foundation model for QE that utilizes principal component analysis (PCA) to compress the space of quantum states needed to solve the EH within a small variational subspace. Using a data-driven active-learning scheme, we learn this subspace from EH ground states and reduce each embedding solve to a deterministic ground-state eigenvalue problem in the reduced space. Within the ghost Gutzwiller approximation (ghost-GA), we show for a three-orbital Hubbard model that a variational space learned on a Bethe lattice is transferable to square and cubic lattices without additional training, while substantially reducing the cost of the EH step. We further validate the approach on plutonium, where a single variational space reproduces the energetics of all six crystalline phases while reducing the cost of the EH solution by orders of magnitude. This provides a practical route to overcome the main computational bottleneck of QE frameworks, paving the way for high-throughput ab initio simulations of strongly correlated materials at a near-DFT cost.

---

## 39. Enabling Ultra-Fast Cardiovascular Imaging Across Heterogeneous Clinical Environments with a Generalist Foundation Model and Multimodal Database

**论文链接:** [http://arxiv.org/abs/2512.21652v1](http://arxiv.org/abs/2512.21652v1)

**作者:** Zi Wang, Mingkai Huang, Zhang Shi, Hongjie Hu, Lan Lan, Hui Zhang, Yan Li, Xi Hu, Qing Lu, Zongming Zhu, Qiong Yao, Yuxiang Dai, Fanwen Wang, Yinzhe Wu, Jun Lyu, Qianqian Gao, Guangming Xu, Zhenxuan Zhang, Haosen Zhang, Qing Li, Guangming Wang, Tianxing He, Lizhen Lan, Siyue Li, Le Xue, Mengting Sun, Yuntong Lyu, Junpu Hu, Jiayu Zhu, Rizwan Ahmad, Zhengyu Bu, Xianling Qian, Guanke Cai, Ruiyu Cao, Weirui Cai, Chang Xu, Yuyang Ren, Feidan Yu, Siying Ma, Ziqiang Xu, Xinran Chen, Sha Hua, Daniel Kim, Yajing Zhang, Chen Ouyang, Wenjia Bai, Jing Qin, Yucheng Yang, Daniel Rueckert, He Wang, Qian Tao, Claudia Prieto, Michael Markl, Alistair Young, Lianming Wu, Shuo Wang, Chen Qin, Mengsu Zeng, Xihong Hu, Haibo Xu, Xiaobo Qu, Hao Li, Guang Yang, Chengyan Wang

**发布时间:** 2025-12-25

**备注:** Github: https://github.com/wangziblake/CardioMM_MMCMR-427K

### GPT解析

### 总结

本研究开发了一个通用的心血管磁共振成像重建基础模型CardioMM，并创建了最大的多模态CMR k空间数据库MMCMR-427K，以解决CMR临床应用中的扫描时间长和环境差异问题。

### 背景

多模态心血管磁共振成像(CMR)能提供全面且非侵入性的心血管疾病诊断和机制洞察，尽管有数十年的技术进步，其临床应用仍受限于扫描时间长和医疗环境差异。

### 目的

开发一个通用的重建基础模型用于超快速CMR成像，使其能适应不同的成像场景，并作为所有下游分析的基础。

### 方法

创建了MMCMR-427K数据库，包含427,465个多线圈k空间数据，配对结构化元数据，涵盖13个国际中心、12种CMR模式、15台扫描仪和17种CVD类别；基于此资源开发了CardioMM模型，该模型结合语义上下文理解和基于物理的数据一致性。

### 主要发现

CardioMM在内部中心实现最先进性能，对未见过的外部设置表现出强大的零样本泛化能力；即使在24倍成像加速下，也能可靠保留关键心脏表型、定量心肌生物标志物和诊断图像质量，显著提高CMR检查吞吐量而不损害临床完整性。

### 结论

开源的MMCMR-427K数据库和CardioMM框架建立了通往高吞吐量、高质量和临床可及的心血管成像的可扩展路径。

### 翻译

多模态心血管磁共振(CMR)成像为心血管疾病(CVD)诊断和潜在机制提供了全面且非侵入性的见解。尽管有数十年的进步，其广泛的临床应用仍然受到扫描时间长和医疗环境异质性的限制。这凸显了对超快速CMR成像的通用重建基础模型的迫切需求，该模型应能适应多样化的成像场景，并作为所有下游分析的基本基础。为实现这一目标，我们整理了MMCMR-427K，这是迄今为止最大、最全面的多模态CMR k空间数据库，包含427,465个多线圈k空间数据，配对来自13个国际中心、12种CMR模式、15台扫描仪和17种CVD类别的结构化元数据，覆盖三大洲人群。基于这一前所未有的资源，我们引入了CardioMM，一个通用重建基础模型，能够动态适应异构的快速CMR成像场景。CardioMM将语义上下文理解与基于物理的数据一致性相结合，提供跨不同扫描仪、协议和患者表现的稳健重建。全面评估表明，CardioMM在内部中心实现了最先进的性能，并对未见过的外部环境表现出强大的零样本泛化能力。即使在高达24倍的成像加速下，CardioMM也能可靠地保留关键心脏表型、定量心肌生物标志物和诊断图像质量，能够在不损害临床完整性的情况下显著提高CMR检查吞吐量。总之，我们的开源MMCMR-427K数据库和CardioMM框架为建立高吞吐量、高质量和临床可及的心血管成像提供了可扩展的途径。


### 论文摘要

Multimodal cardiovascular magnetic resonance (CMR) imaging provides comprehensive and non-invasive insights into cardiovascular disease (CVD) diagnosis and underlying mechanisms. Despite decades of advancements, its widespread clinical adoption remains constrained by prolonged scan times and heterogeneity across medical environments. This underscores the urgent need for a generalist reconstruction foundation model for ultra-fast CMR imaging, one capable of adapting across diverse imaging scenarios and serving as the essential substrate for all downstream analyses. To enable this goal, we curate MMCMR-427K, the largest and most comprehensive multimodal CMR k-space database to date, comprising 427,465 multi-coil k-space data paired with structured metadata across 13 international centers, 12 CMR modalities, 15 scanners, and 17 CVD categories in populations across three continents. Building on this unprecedented resource, we introduce CardioMM, a generalist reconstruction foundation model capable of dynamically adapting to heterogeneous fast CMR imaging scenarios. CardioMM unifies semantic contextual understanding with physics-informed data consistency to deliver robust reconstructions across varied scanners, protocols, and patient presentations. Comprehensive evaluations demonstrate that CardioMM achieves state-of-the-art performance in the internal centers and exhibits strong zero-shot generalization to unseen external settings. Even at imaging acceleration up to 24x, CardioMM reliably preserves key cardiac phenotypes, quantitative myocardial biomarkers, and diagnostic image quality, enabling a substantial increase in CMR examination throughput without compromising clinical integrity. Together, our open-access MMCMR-427K database and CardioMM framework establish a scalable pathway toward high-throughput, high-quality, and clinically accessible cardiovascular imaging.

---

## 40. Omni-Weather: Unified Multimodal Foundation Model for Weather Generation and Understanding

**论文链接:** [http://arxiv.org/abs/2512.21643v1](http://arxiv.org/abs/2512.21643v1)

**作者:** Zhiwang Zhou, Yuandong Pu, Xuming He, Yidi Liu, Yixin Chen, Junchao Gong, Xiang Zhuang, Wanghan Xu, Qinglong Cao, Shixiang Tang, Yihao Liu, Wenlong Zhang, Lei Bai

**发布时间:** 2025-12-25

**备注:** 25 pages, 12 figures. ICLR 2026 submission

### GPT解析

### 总结

本文提出了Omni-Weather，这是首个统一天气生成和理解的多模态基础模型，解决了现有方法将这两个目标孤立处理的问题。

### 背景

天气建模需要准确的预测和机制解释，而现有方法将生成与理解分开处理，导致两者无法协同工作。

### 目的

解决天气建模中预测和解释分离的问题，创建一个能够同时处理天气生成和理解任务的统一模型。

### 方法

Omni-Weather整合了雷达编码器用于天气生成任务，采用共享的自注意力机制进行统一处理，并构建了用于天气生成因果推理的Chain-of-Thought数据集，实现可解释输出和改进的感知质量。

### 主要发现

大量实验表明Omni-Weather在天气生成和理解方面都达到最先进性能；天气领域的生成和理解任务可以相互增强；统一天气生成和理解的方法具有可行性和价值。

### 结论

Omni-Weather成功实现了天气生成和理解的统一，证明了这种统一方法在性能和可解释性方面的优势。

### 翻译

天气建模需要准确的预测和机制解释，然而现有方法将这些目标孤立处理，将生成与理解分开。为解决这一差距，我们提出了Omni-Weather，这是第一个统一天气生成和理解的多模态基础模型。Omni-Weather整合了雷达编码器用于天气生成任务，随后使用共享的自注意力机制进行统一处理。此外，我们构建了用于天气生成因果推理的Chain-of-Thought数据集，使输出具有可解释性并提高了感知质量。大量实验表明，Omni-Weather在天气生成和理解方面都达到了最先进的性能。我们的研究进一步表明，天气领域的生成和理解任务可以相互促进。Omni-Weather还证明了统一天气生成和理解的可行性和价值。


### 论文摘要

Weather modeling requires both accurate prediction and mechanistic interpretation, yet existing methods treat these goals in isolation, separating generation from understanding. To address this gap, we present Omni-Weather, the first multimodal foundation model that unifies weather generation and understanding within a single architecture. Omni-Weather integrates a radar encoder for weather generation tasks, followed by unified processing using a shared self-attention mechanism. Moreover, we construct a Chain-of-Thought dataset for causal reasoning in weather generation, enabling interpretable outputs and improved perceptual quality. Extensive experiments show Omni-Weather achieves state-of-the-art performance in both weather generation and understanding. Our findings further indicate that generative and understanding tasks in the weather domain can mutually enhance each other. Omni-Weather also demonstrates the feasibility and value of unifying weather generation and understanding.

---

## 41. RefineBridge: Generative Bridge Models Improve Financial Forecasting by Foundation Models

**论文链接:** [http://arxiv.org/abs/2512.21572v1](http://arxiv.org/abs/2512.21572v1)

**作者:** Anthony Bolton, Wuyang Zhou, Zehua Chen, Giorgos Iacovides, Danilo Mandic

**发布时间:** 2025-12-25

### GPT解析

### 总结

提出了一种名为RefineBridge的新改进模块，基于Schrödinger Bridge生成框架，用于增强基于Transformer的时间序列基础模型(TSFMs)在金融时间序列预测中的性能。

### 背景

金融时间序列预测对于基于Transformer的时间序列基础模型(TSFMs)特别具有挑战性，因为数据中存在非平稳性、重尾分布和高频噪声。

### 目的

开发一种方法来改进TSFMs在金融数据上的预测性能，解决现有LoRA方法在金融数据上表现不佳的问题。

### 方法

RefineBridge模块基于可处理的Schrödinger Bridge生成框架构建，将TSFM的预测作为生成先验，观察到的真实值作为目标，学习上下文条件随机传输映射，逐步从低质量先验接近真实目标。

### 主要发现

在多个金融基准测试上的模拟表明，RefineBridge在不同预测时间范围内持续提高了最先进TSFMs的性能。

### 结论

RefineBridge是一种有效的方法，可以改进基于Transformer的时间序列基础模型在金融时间序列预测任务中的表现。

### 翻译

金融时间序列预测对于基于Transformer的时间序列基础模型(TSFMs)尤其具有挑战性，因为数据中存在非平稳性、重尾分布和高频噪声。低秩适应(LoRA)已成为一种流行的参数高效方法，用于将预训练的TSFMs适应到下游数据领域。然而，它在金融数据上仍然表现不佳，因为它保留了TSFMs的网络架构和训练目标，而不是补充基础模型。为了进一步增强TSFMs，我们提出了一种名为RefineBridge的新型改进模块，它构建在可处理的Schrödinger Bridge(SB)生成框架之上。将TSFM的预测作为生成先验，观察到的真实值作为目标，RefineBridge学习上下文条件随机传输映射，以改进TSFM预测，逐步从低质量先验接近真实目标。在多个金融基准测试上的模拟表明，RefineBridge在不同预测时间范围内持续提高了最先进TSFMs的性能。


### 论文摘要

Financial time series forecasting is particularly challenging for transformer-based time series foundation models (TSFMs) due to non-stationarity, heavy-tailed distributions, and high-frequency noise present in data. Low-rank adaptation (LoRA) has become a popular parameter-efficient method for adapting pre-trained TSFMs to downstream data domains. However, it still underperforms in financial data, as it preserves the network architecture and training objective of TSFMs rather than complementing the foundation model. To further enhance TSFMs, we propose a novel refinement module, RefineBridge, built upon a tractable Schrödinger Bridge (SB) generative framework. Given the forecasts of TSFM as generative prior and the observed ground truths as targets, RefineBridge learns context-conditioned stochastic transport maps to improve TSFM predictions, iteratively approaching the ground-truth target from even a low-quality prior. Simulations on multiple financial benchmarks demonstrate that RefineBridge consistently improves the performance of state-of-the-art TSFMs across different prediction horizons.

---

## 42. Perplexity-Aware Data Scaling Law: Perplexity Landscapes Predict Performance for Continual Pre-training

**论文链接:** [http://arxiv.org/abs/2512.21515v1](http://arxiv.org/abs/2512.21515v1)

**作者:** Lei Liu, Hao Zhu, Yue Shen, Zhixuan Chu, Jian Wang, Jinjie Gu, Kui Ren

**发布时间:** 2025-12-25

### GPT解析

### 总结

本文提出了一种困惑度感知的数据扩展定律，用于优化持续预训练过程中的数据选择，解决了简单增加数据导致的边际收益递减问题，通过量化候选训练样本的信息困惑度景观，实现了高效数据子集选择，在医学和通用领域基准测试上取得了优越性能。

### 背景

持续预训练是将基础模型适应特定领域应用的基本方法。预训练的扩展定律定义了数据集规模和大语言模型测试损失之间的幂律关系。然而，简单地增加持续预训练的数据会导致边际收益迅速减少，造成数据利用次优和训练效率低下。

### 目的

提出一种新的困惑度感知数据扩展定律，建立特定领域数据的困惑度景观与测试损失之间的预测关系，以优化数据选择过程。

### 方法

利用预训练模型在领域数据上推导的困惑度作为估计知识差距的代理，有效量化候选训练样本的信息困惑度景观。通过在不同困惑度范围内拟合这个扩展定律，实现自适应选择高效数据子集，优先选择能最大化知识吸收同时最小化冗余和噪声的内容。

### 主要发现

该方法能够识别接近最优的训练子集，在医学和通用领域基准测试上取得优越性能，证明了困惑度感知数据选择的有效性。

### 结论

通过困惑度感知的数据选择方法，可以优化持续预训练过程，提高数据利用效率和模型性能，解决了简单增加数据导致的边际收益递减问题。

### 翻译

持续预训练是将基础模型适应特定领域应用的基本方法。预训练的扩展定律定义了数据集规模和大语言模型测试损失之间的幂律关系。然而，简单地增加持续预训练数据的边际收益会迅速减少，导致数据利用次优和训练效率低下。为解决这一挑战，我们提出了一种新的困惑度感知数据扩展定律，建立了特定领域数据的困惑度景观与测试损失之间的预测关系。我们的方法利用预训练模型在领域数据上推导的困惑度作为估计知识差距的代理，有效量化了候选训练样本的信息困惑度景观。通过在不同困惑度范围内拟合这个扩展定律，我们能够自适应选择高效数据子集，优先选择能最大化知识吸收同时最小化冗余和噪声的内容。大量实验证明，我们的方法能够一致地识别接近最优的训练子集，并在医学和通用领域基准测试上取得优越性能。


### 论文摘要

Continual Pre-training (CPT) serves as a fundamental approach for adapting foundation models to domain-specific applications. Scaling laws for pre-training define a power-law relationship between dataset size and the test loss of an LLM. However, the marginal gains from simply increasing data for CPT diminish rapidly, yielding suboptimal data utilization and inefficient training. To address this challenge, we propose a novel perplexity-aware data scaling law to establish a predictive relationship between the perplexity landscape of domain-specific data and the test loss. Our approach leverages the perplexity derived from the pre-trained model on domain data as a proxy for estimating the knowledge gap, effectively quantifying the informational perplexity landscape of candidate training samples. By fitting this scaling law across diverse perplexity regimes, we enable adaptive selection of high-utility data subsets, prioritizing content that maximizes knowledge absorption while minimizing redundancy and noise. Extensive experiments demonstrate that our method consistently identifies near-optimal training subsets and achieves superior performance on both medical and general-domain benchmarks.

---

## 43. EVE: A Generator-Verifier System for Generative Policies

**论文链接:** [http://arxiv.org/abs/2512.21430v1](http://arxiv.org/abs/2512.21430v1)

**作者:** Yusuf Ali, Gryphon Patlin, Karthik Kothuri, Muhammad Zubair Irshad, Wuwei Liang, Zsolt Kira

**发布时间:** 2025-12-24

### GPT解析

### 总结

本文提出了一种名为EVE的模块化生成器-验证器交互框架，通过在测试时利用零样本VLM验证器来增强预训练的生成策略性能，无需额外训练。该框架在多种操作任务中一致提高了任务成功率。

### 背景

基于生成架构的视觉运动策略在分布变化下性能下降且恢复能力有限。语言建模领域通过测试时计算扩展和零样本验证模块改进了候选解决方案，但这种生成-验证框架在视觉运动策略领域尚未充分探索。

### 目的

研究生成策略如何通过额外推理时计算和零样本VLM验证器受益，构建一个系统化的生成-验证框架来提升预训练生成策略在测试时的性能。

### 方法

EVE框架将冻结的基础策略与多个零样本VLM验证器代理结合，验证器提出动作改进建议，动作融合器聚合验证器输出并与基础策略预测融合产生最终动作。研究了具有不同能力的验证器系统中生成器-验证器信息交互的设计选择。

### 主要发现

在多样化的操作任务套件中，EVE在无需任何额外策略训练的情况下，一致提高了任务成功率。通过消融实验分离了验证器能力和动作融合器策略的贡献。

### 结论

EVE为构建可扩展的模块化生成器-验证器系统提供了实用指南，可用于具身控制，展示了零样本验证器在增强预训练生成策略方面的潜力。

### 翻译

基于生成架构的视觉运动策略，如扩散和基于流的匹配，已表现出强大的性能，但在分布变化下性能下降，显示出有限的恢复能力，而无需昂贵的微调。在语言建模领域，测试时计算扩展通过利用额外的推理时计算来改进候选解决方案，从而革新了现代LLM的推理能力。这些方法通常以零样本方式利用基础模型作为验证模块来合成改进的候选解决方案。在这项工作中，我们假设生成策略可以通过采用零样本VLM验证器的额外推理时计算获得类似的好处。关于通过生成-验证框架改进策略性能的系统分析在当前文献中相对未被探索。为此，我们引入了EVE - 一个模块化的生成器-验证器交互框架 - 它可以在测试时提高预训练生成策略的性能，无需额外训练。EVE将一个冻结的基础策略与多个零样本VLM验证器代理包装在一起。每个验证器向基础策略候选动作提出动作改进建议，而动作融合器将聚合的验证器输出融合到基础策略动作预测中，以产生最终执行的动作。我们研究了具有不同能力的验证器系统中生成器-验证器信息交互的设计选择。在多样化的操作任务套件中，EVE一致提高了任务成功率，而无需任何额外的策略训练。通过大量的消融实验，我们分离了验证器能力和动作融合器策略的贡献，为构建可扩展的模块化生成器-验证器系统提供了实用指南，用于具身控制。


### 论文摘要

Visuomotor policies based on generative architectures such as diffusion and flow-based matching have shown strong performance but degrade under distribution shifts, demonstrating limited recovery capabilities without costly finetuning. In the language modeling domain, test-time compute scaling has revolutionized reasoning capabilities of modern LLMs by leveraging additional inference-time compute for candidate solution refinement. These methods typically leverage foundation models as verification modules in a zero-shot manner to synthesize improved candidate solutions. In this work, we hypothesize that generative policies can similarly benefit from additional inference-time compute that employs zero-shot VLM-based verifiers. A systematic analysis of improving policy performance through the generation-verification framework remains relatively underexplored in the current literature. To this end, we introduce EVE - a modular, generator-verifier interaction framework - that boosts the performance of pretrained generative policies at test time, with no additional training. EVE wraps a frozen base policy with multiple zero-shot, VLM-based verifier agents. Each verifier proposes action refinements to the base policy candidate actions, while an action incorporator fuses the aggregated verifier output into the base policy action prediction to produce the final executed action. We study design choices for generator-verifier information interfacing across a system of verifiers with distinct capabilities. Across a diverse suite of manipulation tasks, EVE consistently improves task success rates without any additional policy training. Through extensive ablations, we isolate the contribution of verifier capabilities and action incorporator strategies, offering practical guidelines to build scalable, modular generator-verifier systems for embodied control.

---

