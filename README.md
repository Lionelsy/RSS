# 今日论文推荐 - 2025-12-02

共 222 篇论文

---

## 1. Four Over Six: More Accurate NVFP4 Quantization with Adaptive Block Scaling

**论文链接:** [http://arxiv.org/abs/2512.02010v1](http://arxiv.org/abs/2512.02010v1)

**作者:** Jack Cook, Junxian Guo, Guangxuan Xiao, Yujun Lin, Song Han

**发布时间:** 2025-12-01

**备注:** 10 pages, 5 figures

### GPT解析

### 总结

本文提出了一种名为Four Over Six (4/6)的NVFP4量化算法修改方案，解决了使用低精度格式训练大语言模型时的发散和性能下降问题。

### 背景

随着大语言模型规模不断扩大，NVFP4等低精度数值格式因其速度和内存优势而日益流行，但使用该格式进行量化会导致训练发散和推理性能下降。

### 目的

开发一种改进的NVFP4量化算法，解决训练过程中的发散问题，提高推理性能，并使其能在NVIDIA Blackwell GPU上高效实现。

### 方法

提出4/6算法，对每个值块评估两个而非多个缩放因子，通过缩放到更小的FP4值使可表示值的分布更加均匀，改善对接近最大值的表示。

### 主要发现

在transformer和混合模型架构的预训练实验中，4/6在多种情况下防止了训练发散，使训练损失显著接近BF16；该方法可轻松整合到多种训练后量化方法中，提高下游准确性。

### 结论

4/6算法有效解决了NVFP4量化中的关键问题，可在训练LLM时使用NVFP4同时保持高效，为未来使用NVFP4训练和部署模型提供了新思路。

### 翻译

随着大语言模型规模不断扩大，NVFP4等低精度数值格式因其提供的速度和内存优势而日益流行。然而，为了使用NVFP4加速计算，所有矩阵乘法操作数（前向传播中的权重和激活，以及反向传播中的权重、激活和梯度）都必须量化为NVFP4，这通常会导致训练过程中的发散和推理过程中的性能下降。NVFP4通过评估每个值块的多个可能的缩放因子来解决这一问题。为解决此问题，本文介绍了Four Over Six (4/6)，这是对NVFP4量化算法的一种修改，对每个值块评估两个可能的缩放因子。与整数格式不同，FP4等浮点格式在每个块中接近最大值的值上具有最大的量化误差，我们发现这主要是导致下游性能下降的原因。我们发现，对于某些块，缩放到更小的FP4值可以使可表示值的分布更加均匀，改善对接近最大值的表示。重要的是，4/6可以在NVIDIA Blackwell GPU上高效实现，使其在训练LLM时使用NVFP4成为可能。在transformer和混合模型架构的预训练实验中，我们发现4/6在多种情况下防止了发散，与使用当前最先进的NVFP4训练方法训练的模型相比，使训练损失显著接近BF16。我们还发现，4/6可以轻松整合到许多不同的训练后量化方法中，并通常提高下游准确性。我们希望这能激励未来使用NVFP4训练和部署模型的工作。


### 论文摘要

As large language models have grown larger, low-precision numerical formats such as NVFP4 have become increasingly popular due to the speed and memory benefits they provide. However, to accelerate computation with NVFP4, all matrix multiplication operands--weights and activations in the forward pass, and weights, activations, and gradients in the backward pass--must be quantized to NVFP4, often leading to divergence during training and performance degradation during inference. NVFP4 by evaluating multiple potential scale factors for each block of values. To address this issue, in this work we introduce Four Over Six (4/6), a modification to the NVFP4 quantization algorithm that evaluates two potential scale factors for each block of values. Unlike integer formats, floating-point formats such as FP4 have the most quantization error on near-maximal values in each block, which we find to be primarily responsible for downstream performance degradation. We find that for some blocks, scaling to smaller FP4 values makes the distribution of representable values more uniform, improving representation of near-maximal values. Importantly, 4/6 can be implemented efficiently on NVIDIA Blackwell GPUs, making it viable to use while training LLMs with NVFP4. In pre-training experiments with transformer and hybrid model architectures, we find that 4/6 prevents divergence in several cases, bringing training loss significantly closer to BF16 compared to models trained with current state-of-the-art NVFP4 training recipes. We also find that 4/6 can be easily incorporated into many different post-training quantization methods and generally improves downstream accuracy. We hope this inspires future work in training and deploying models with NVFP4.

---

## 2. Artemis: Structured Visual Reasoning for Perception Policy Learning

**论文链接:** [http://arxiv.org/abs/2512.01988v1](http://arxiv.org/abs/2512.01988v1)

**作者:** Wei Tang, Yanpeng Sun, Shan Zhang, Xiaofan Li, Piotr Koniusz, Wei Li, Na Zhao, Zechao Li

**发布时间:** 2025-12-01

### GPT解析

### 总结

本文提出了一种名为Artemis的感知策略学习框架，通过结构化提案推理解决视觉感知中语言推理的局限性，在多种任务上表现出色并展现出强大的泛化能力。

### 背景

最近的视觉感知策略强化学习框架开始融入自然语言表达的中间推理链，但经验表明这种纯语言中间推理通常会降低感知任务性能。

### 目的

解决视觉感知中语言推理形式不当导致的性能下降问题，提出更适合视觉感知的推理框架。

### 方法

开发Artemis框架，采用结构化提案推理，将每个中间步骤表示为(标签，边界框)对，捕捉可验证的视觉状态，基于Qwen2.5-VL-3B构建。

### 主要发现

Artemis在定位和检测任务上表现出色，在计数和几何感知任务上展现出显著的泛化能力，空间基础推理为可扩展和通用的感知策略提供了原则性途径。

### 结论

将推理与空间表示对齐可以增强感知策略学习，空间基础推理是构建可扩展和通用感知策略的有效方法。

### 翻译

最近的视觉感知策略强化学习框架已开始融入自然语言表达的中间推理链。经验观察表明，这种纯语言中间推理通常会降低感知任务性能。我们认为核心问题不在于推理本身，而在于推理形式：虽然这些链在非结构化的语言空间中进行语义推理，但视觉感知需要在空间和以对象为中心的空间中进行推理。为此，我们引入了Artemis，一个执行结构化提案推理的感知策略学习框架，其中每个中间步骤表示为一个(标签，边界框)对，捕捉可验证的视觉状态。这种设计能够明确跟踪中间状态，直接监督提案质量，并避免基于语言推理带来的模糊性。Artemis基于Qwen2.5-VL-3B构建，在定位和检测任务上取得强性能，并在计数和几何感知任务上展现出显著的泛化能力。这些不同设置下的一致性改进证实，将推理与空间表示对齐可以增强感知策略学习。由于其增强的视觉推理能力，Artemis在通用MLLM基准测试上也取得了具有竞争力的性能，说明空间基础推理为构建可扩展和通用的感知策略提供了原则性途径。


### 论文摘要

Recent reinforcement-learning frameworks for visual perception policy have begun to incorporate intermediate reasoning chains expressed in natural language. Empirical observations indicate that such purely linguistic intermediate reasoning often reduces performance on perception tasks. We argue that the core issue lies not in reasoning per se but in the form of reasoning: while these chains perform semantic reasoning in an unstructured linguistic space, visual perception requires reasoning in a spatial and object-centric space. In response, we introduce Artemis, a perception-policy learning framework that performs structured proposal-based reasoning, where each intermediate step is represented as a (label, bounding-box) pair capturing a verifiable visual state. This design enables explicit tracking of intermediate states, direct supervision for proposal quality, and avoids ambiguity introduced by language-based reasoning. Artemis is built on Qwen2.5-VL-3B, achieves strong performance on grounding and detection task and exhibits substantial generalization to counting and geometric-perception tasks. The consistent improvements across these diverse settings confirm that aligning reasoning with spatial representations enhances perception-policy learning. Owing to its strengthened visual reasoning, Artemis also achieves competitive performance on general MLLM benchmarks, illustrating that spatially grounded reasoning provides a principled route toward scalable and general perception policies.

---

## 3. Real-World Robot Control by Deep Active Inference With a Temporally Hierarchical World Model

**论文链接:** [http://arxiv.org/abs/2512.01924v1](http://arxiv.org/abs/2512.01924v1)

**作者:** Kentaro Fujii, Shingo Murata

**发布时间:** 2025-12-01

**DOI:** 10.1109/LRA.2025.3636032

**备注:** Accepted for publication in IEEE Robotics and Automation Letters (RA-L)

### GPT解析

### 总结

本文提出了一种新颖的深度主动推理框架，包含世界模型、动作模型和抽象世界模型，用于机器人在不确定环境中执行目标导向和探索性行动，同时降低计算成本。

### 背景

大多数基于深度学习的控制方法忽视了探索行为，在不确定性条件下表现不佳。传统深度主动推理方法由于环境表示能力有限和动作选择计算成本高而面临挑战。

### 目的

解决现有深度学习控制方法在不确定性环境下表现不佳的问题，特别是它们忽视探索行为且动作选择计算成本高的问题。

### 方法

提出包含三个核心组件的框架：1)世界模型将环境动态编码为慢速和快速时间尺度的隐藏状态；2)动作模型使用向量量化将动作序列压缩为抽象动作；3)抽象世界模型基于抽象动作预测未来慢速状态，实现低成本动作选择。

### 主要发现

在实物机器人对象操作任务中，该框架在多样化任务中取得高成功率，能够在不确定环境中切换目标导向和探索性行动，同时使动作选择计算上可行。

### 结论

建模多时间尺度动态以及抽象动作和状态转换对于机器人在不确定环境中的有效控制至关重要。

### 翻译

在不确定的真实世界环境中的机器人必须执行目标导向和探索性行动。然而，大多数基于深度学习的控制方法忽视了探索行为，在不确定性条件下表现不佳。为了解决这个问题，我们采用了深度主动推理，这是一个考虑人类目标导向和探索性行动的框架。然而，传统的深度主动推理方法由于环境表示能力有限和动作选择计算成本高而面临挑战。我们提出了一种新颖的深度主动推理框架，包括世界模型、动作模型和抽象世界模型。世界模型将环境动态编码为慢速和快速时间尺度的隐藏状态表示。动作模型使用向量量化将动作序列压缩为抽象动作，抽象世界模型基于抽象动作预测未来慢速状态，实现低成本动作选择。我们在实物机器人的对象操作任务上评估了该框架。结果表明，它在多样化的操作任务中取得了高成功率，能够在不确定环境中切换目标导向和探索性行动，同时使动作选择计算上可行。这些发现强调了建模多时间尺度动态以及抽象动作和状态转换的重要性。


### 论文摘要

Robots in uncertain real-world environments must perform both goal-directed and exploratory actions. However, most deep learning-based control methods neglect exploration and struggle under uncertainty. To address this, we adopt deep active inference, a framework that accounts for human goal-directed and exploratory actions. Yet, conventional deep active inference approaches face challenges due to limited environmental representation capacity and high computational cost in action selection. We propose a novel deep active inference framework that consists of a world model, an action model, and an abstract world model. The world model encodes environmental dynamics into hidden state representations at slow and fast timescales. The action model compresses action sequences into abstract actions using vector quantization, and the abstract world model predicts future slow states conditioned on the abstract action, enabling low-cost action selection. We evaluate the framework on object-manipulation tasks with a real-world robot. Results show that it achieves high success rates across diverse manipulation tasks and switches between goal-directed and exploratory actions in uncertain settings, while making action selection computationally tractable. These findings highlight the importance of modeling multiple timescale dynamics and abstracting actions and state transitions.

---

## 4. SARL: Spatially-Aware Self-Supervised Representation Learning for Visuo-Tactile Perception

**论文链接:** [http://arxiv.org/abs/2512.01908v1](http://arxiv.org/abs/2512.01908v1)

**作者:** Gurmeher Khurana, Lan Wei, Dandan Zhang

**发布时间:** 2025-12-01

### GPT解析

### 总结

SARL是一种空间感知的自监督学习框架，通过地图级别目标保持跨视图的注意力焦点、部分组成和几何关系一致，在融合视觉-触觉数据的机器人操作任务中表现优异。

### 背景

接触密集型机器人操作需要编码局部几何的表示。视觉提供全局上下文但缺乏直接测量纹理和硬度等属性，而触觉提供这些线索。现代视觉-触觉传感器在单一融合图像中捕获两种模态，产生内在对齐的输入。

### 目的

解决大多数自监督学习框架将特征图压缩为全局向量而丢弃空间结构的问题，提出SARL框架以保持空间信息。

### 方法

SARL增强Bootstrap Your Own Latent (BYOL)架构，添加三个地图级别目标：显著性对齐(SAL)、补丁原型分布对齐(PPDA)和区域亲和匹配(RAM)，这些损失作用于中间特征图，补充全局目标。

### 主要发现

在六个融合视觉-触觉数据的下游任务中，SARL始终优于九个自监督学习基线。在几何敏感的边缘姿态回归任务中，SARL实现了0.3955的平均绝对误差，比次佳方法有30%的相对改进，接近监督上限。

### 结论

对于融合视觉-触觉数据，最有效的信号是结构化空间等变性，其中特征随物体几何可预测地变化，从而实现更强大的机器人感知。

### 翻译

摘要原文已提供，无需翻译。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决传统自监督学习(SSL)框架在处理融合视觉-触觉数据时丢失空间结构信息的问题。这个问题很重要，因为接触密集的机器人操作需要编码局部几何的表示，而传统SSL方法将特征图压缩为全局向量，丢弃了空间结构，无法满足操作任务的需求。视觉提供全局上下文但缺乏物理属性测量，触觉提供这些线索但缺乏全局上下文，融合传感器提供了理想的多模态输入，但现有方法无法充分利用其内在的空间对齐结构。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先认识到视觉和触觉信息的互补性，发现现有SSL方法在处理融合视觉-触觉数据时的局限性。他们假设在特征级别明确强制执行空间和语义一致性的SSL框架将产生更精细的表示。该方法借鉴了BYOL(Bootstrap Your Own Latent)架构作为基础，这是一种稳定的非对比学习方法。同时受到视觉-触觉感知领域现有工作的启发，如MViTac和Sparsh，但针对融合视觉-触觉数据的特点进行了改进，引入了三个新的空间感知损失函数。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是在自监督学习中保留空间结构信息，而不仅仅是全局特征。通过三个辅助损失函数在中间特征图上操作，保持注意力焦点、语义部分和几何关系的一致性。整体实现流程如下：1)使用BYOL作为基础架构，包含在线网络和目标网络；2)添加三个空间感知损失函数：SAL(显著性对齐)确保模型关注相同区域，PPDA(块原型分布对齐)确保语义部分一致，RAM(区域亲和匹配)保持几何关系；3)这些损失在中间特征图上操作，补充全局目标；4)使用加权组合的目标函数进行训练。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)提出SARL框架，专门为融合视觉-触觉数据设计；2)引入三个新的空间感知损失函数：SAL、PPDA和RAM；3)在中间特征图上操作，而非仅依赖全局特征；4)能够保留空间和几何信息，对精细机器人操作至关重要。相比之前的工作，SARL不同于传统SSL方法(如SimCLR、BYOL)将特征图压缩为全局向量；区别于多模态方法(如MViTac)只在全局级别对齐模态；也不同于单模态触觉方法(如Sparsh)操作在单一模态上。SARL是第一个专门为融合视觉-触觉数据设计的自监督框架，能在像素级别利用模态间的对应关系。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'SARL通过引入三个空间感知损失函数，解决了传统自监督学习在处理融合视觉-触觉数据时丢失空间结构信息的问题，显著提升了机器人在需要精细几何推理的任务上的表现。'}


### 论文摘要

Contact-rich robotic manipulation requires representations that encode local geometry. Vision provides global context but lacks direct measurements of properties such as texture and hardness, whereas touch supplies these cues. Modern visuo-tactile sensors capture both modalities in a single fused image, yielding intrinsically aligned inputs that are well suited to manipulation tasks requiring visual and tactile information. Most self-supervised learning (SSL) frameworks, however, compress feature maps into a global vector, discarding spatial structure and misaligning with the needs of manipulation. To address this, we propose SARL, a spatially-aware SSL framework that augments the Bootstrap Your Own Latent (BYOL) architecture with three map-level objectives, including Saliency Alignment (SAL), Patch-Prototype Distribution Alignment (PPDA), and Region Affinity Matching (RAM), to keep attentional focus, part composition, and geometric relations consistent across views. These losses act on intermediate feature maps, complementing the global objective. SARL consistently outperforms nine SSL baselines across six downstream tasks with fused visual-tactile data. On the geometry-sensitive edge-pose regression task, SARL achieves a Mean Absolute Error (MAE) of 0.3955, a 30% relative improvement over the next-best SSL method (0.5682 MAE) and approaching the supervised upper bound. These findings indicate that, for fused visual-tactile data, the most effective signal is structured spatial equivariance, in which features vary predictably with object geometry, which enables more capable robotic perception.

---

## 5. The Mean-Field Dynamics of Transformers

**论文链接:** [http://arxiv.org/abs/2512.01868v1](http://arxiv.org/abs/2512.01868v1)

**作者:** Philippe Rigollet

**发布时间:** 2025-12-01

**备注:** to appear as Proceedings of the ICM2026, Philadelphia, USA

### GPT解析

### 总结

本文开发了一个将Transformer注意力机制解释为相互作用粒子系统的数学框架，并研究了其连续极限（平均场极限），揭示了全局聚类现象和表示崩溃机制。

### 背景

Transformer注意力机制需要更深入的理论理解，特别是其在表示学习和聚类方面的行为。

### 目的

通过将注意力理想化为球面上的连续体，建立Transformer动力学与Wasserstein梯度流、同步模型（Kuramoto）和均值漂移聚类的理论联系。

### 方法

采用数学框架将Transformer注意力机制建模为相互作用粒子系统，分析其连续极限行为，并研究可处理的等角简化以获得精确聚类率。

### 主要发现

1) 存在全局聚类现象，令牌在经历长时间亚稳态后渐近聚类；2) 常用归一化方案会改变收缩速度；3) 长上下文注意力存在相变；4) 识别了导致表示崩溃的机制和保留多簇结构的条件。

### 结论

研究结果既揭示了导致表示崩溃的机制，也明确了在深度注意力架构中保留表达性、多簇结构的条件，为理解Transformer行为提供了新视角。

### 翻译

我们开发了一个将Transformer注意力解释为相互作用粒子系统的数学框架，并研究了其连续（平均场）极限。通过将注意力理想化为球面上的连续体，我们将Transformer动力学与Wasserstein梯度流、同步模型（Kuramoto）和均值漂移聚类联系起来。我们结果的核心是一个全局聚类现象，即令牌在经历长时间的亚稳态（在此期间它们被排列成多个簇）后渐近地聚类。我们进一步分析了可处理的等角简化以获得精确的聚类率，展示了常用归一化方案如何改变收缩速度，并确定了长上下文注意力的相变。这些结果既突出了导致表示崩溃的机制，也保留了深度注意力架构中表达性、多簇结构的范围。


### 论文摘要

We develop a mathematical framework that interprets Transformer attention as an interacting particle system and studies its continuum (mean-field) limits. By idealizing attention continuous on the sphere, we connect Transformer dynamics to Wasserstein gradient flows, synchronization models (Kuramoto), and mean-shift clustering. Central to our results is a global clustering phenomenon whereby tokens cluster asymptotically after long metastable states where they are arranged into multiple clusters. We further analyze a tractable equiangular reduction to obtain exact clustering rates, show how commonly used normalization schemes alter contraction speeds, and identify a phase transition for long-context attention. The results highlight both the mechanisms that drive representation collapse and the regimes that preserve expressive, multi-cluster structure in deep attention architectures.

---

## 6. JPEGs Just Got Snipped: Croppable Signatures Against Deepfake Images

**论文链接:** [http://arxiv.org/abs/2512.01845v1](http://arxiv.org/abs/2512.01845v1)

**作者:** Pericle Perazzo, Massimiliano Mattei, Giuseppe Anastasi, Marco Avvenuti, Gianluca Dini, Giuseppe Lettieri, Carlo Vallati

**发布时间:** 2025-12-01

**DOI:** 10.1109/IJCNN64981.2025.11227387

### GPT解析

### 总结

本文提出了一种基于BLS签名的图像认证方法，能够在图像被裁剪的情况下保持签名有效性，同时有效检测包括Deepfake在内的其他图像篡改操作。

### 背景

Deepfakes是一种使用人工智能和深度学习算法创建的合成媒体，可将面部和声音叠加到视频中，创造出超现实但虚假的内容，对信息真实性和公众信任构成威胁。

### 目的

开发一种签名方案，使图像在裁剪后签名仍然有效，但在其他类型的操作（如Deepfake创建）中签名无效，从而保护图像完整性并防止虚假信息传播。

### 方法

利用BLS签名（Boneh, Lynn, and Shacham 2004）实现一种特殊的签名机制，该机制对图像裁剪操作具有鲁棒性，但对其他类型的图像操作（包括Deepfake创建）敏感。

### 主要发现

该方法不需要裁剪图像的人知道签名私钥或被信任，签名大小为O(1)，使其成为通过Web服务器传播图像且裁剪是主要转换场景的实用解决方案。

### 结论

通过将签名方案适应JPEG标准并进行实验测试，该方法为检测Deepfake等图像篡改提供了有效工具，有助于维护信息真实性和公众信任。

### 翻译

深度伪造（Deepfakes）是一种使用人工智能特别是深度学习算法创建的合成媒体。例如，这种技术可以将面部和声音叠加到视频中，创造出超现实但人工的表示。Deepfakes在虚假信息和假新闻方面构成重大风险，因为它们可以通过描绘公众人物说或做他们从未做过的事情来传播虚假信息，从而破坏公众信任。在本文中，我们提出了一种利用BLS签名（Boneh, Lynn, and Shacham 2004）的方法，实现图像裁剪后仍然有效，但在其他类型的操作（包括Deepfake创建）中无效的签名。我们的方法不需要裁剪图像的人知道签名私钥或被信任，并且在签名大小方面是O(1)，使其成为通过Web服务器传播图像且裁剪是主要转换场景的实用解决方案。最后，我们将签名方案适应了JPEG标准，并实验测试了签名图像的大小。


### 论文摘要

Deepfakes are a type of synthetic media created using artificial intelligence, specifically deep learning algorithms. This technology can for example superimpose faces and voices onto videos, creating hyper-realistic but artificial representations. Deepfakes pose significant risks regarding misinformation and fake news, because they can spread false information by depicting public figures saying or doing things they never did, undermining public trust. In this paper, we propose a method that leverages BLS signatures (Boneh, Lynn, and Shacham 2004) to implement signatures that remain valid after image cropping, but are invalidated in all the other types of manipulation, including deepfake creation. Our approach does not require who crops the image to know the signature private key or to be trusted in general, and it is O(1) in terms of signature size, making it a practical solution for scenarios where images are disseminated through web servers and cropping is the primary transformation. Finally, we adapted the signature scheme for the JPEG standard, and we experimentally tested the size of a signed image.

---

## 7. Decision Tree Embedding by Leaf-Means

**论文链接:** [http://arxiv.org/abs/2512.01819v1](http://arxiv.org/abs/2512.01819v1)

**作者:** Cencheng Shen, Yuexiao Dong, Carey E. Priebe

**发布时间:** 2025-12-01

**备注:** 9 pages

### GPT解析

### 总结

本文提出了一种名为决策树嵌入(DTE)的快速有效方法，利用训练好的决策树的叶分区构建可解释的特征表示，在保持可解释性的同时提高计算效率，并在准确性和计算效率之间取得了良好平衡。

### 背景

决策树和随机森林在中等规模标准数据集上的分类任务中具有很强的竞争力，具有鲁棒性、最小预处理要求和可解释性。然而，单个决策树存在高估计方差，而大型集成方法虽减少方差却带来显著计算开销和可解释性降低。

### 目的

开发一种快速有效的方法，利用决策树的叶分区构建可解释的特征表示，规避决策树分割规则中的高方差问题，同时保持或提高分类性能并降低计算复杂度。

### 方法

利用每个叶区域内的样本均值作为锚点，将输入映射到由树的分区结构定义的嵌入空间；引入基于额外自举树的集成扩展；将得到的嵌入与线性判别分析配对进行分类；建立DTE的总体层面理论性质。

### 主要发现

DTE在温和条件下保持条件密度；对分类误差进行了表征；在合成和真实数据集上优于或匹配随机森林和浅层神经网络；在大多数情况下只需要它们一小部分的训练时间。

### 结论

DTE可被视为改进了标准分割规则的可扩展决策树分类器，或被视为从树衍生锚点学习权重的神经网络模型，实现了两种范式的有趣整合。

### 翻译

决策树和随机森林由于其鲁棒性、最小的预处理要求和可解释性，在中等规模标准数据集上的分类任务中仍然具有很高的竞争力。然而，单个决策树存在高估计方差，而大型集成方法在减少这种方差的同时带来了显著的计算开销和可解释性的降低。在本文中，我们提出了决策树嵌入(DTE)，一种快速有效的方法，它利用训练好的分类树的叶分区来构建可解释的特征表示。通过使用每个叶区域内的样本均值作为锚点，DTE将输入映射到由树的分区结构定义的嵌入空间，有效规避了决策树分割规则中固有的大方差问题。我们进一步引入了基于额外自举树的集成扩展，并将得到的嵌入与线性判别分析配对进行分类。我们建立了DTE的几个总体层面理论性质，包括在温和条件下对条件密度的保持以及对 resulting classification error 的表征。在合成和真实数据集上的实证研究表明，DTE在准确性和计算效率之间取得了良好的平衡，优于或匹配随机森林和浅层神经网络，同时在大多数情况下只需要它们一小部分的训练时间。总体而言，所提出的DTE方法可以被视为一种改进了标准分割规则的可扩展决策树分类器，或被视为一种权重从树衍生锚点中学习的神经网络模型，实现了两种范式的有趣整合。


### 论文摘要

Decision trees and random forest remain highly competitive for classification on medium-sized, standard datasets due to their robustness, minimal preprocessing requirements, and interpretability. However, a single tree suffers from high estimation variance, while large ensembles reduce this variance at the cost of substantial computational overhead and diminished interpretability. In this paper, we propose Decision Tree Embedding (DTE), a fast and effective method that leverages the leaf partitions of a trained classification tree to construct an interpretable feature representation. By using the sample means within each leaf region as anchor points, DTE maps inputs into an embedding space defined by the tree's partition structure, effectively circumventing the high variance inherent in decision-tree splitting rules. We further introduce an ensemble extension based on additional bootstrap trees, and pair the resulting embedding with linear discriminant analysis for classification. We establish several population-level theoretical properties of DTE, including its preservation of conditional density under mild conditions and a characterization of the resulting classification error. Empirical studies on synthetic and real datasets demonstrate that DTE strikes a strong balance between accuracy and computational efficiency, outperforming or matching random forest and shallow neural networks while requiring only a fraction of their training time in most cases. Overall, the proposed DTE method can be viewed either as a scalable decision tree classifier that improves upon standard split rules, or as a neural network model whose weights are learned from tree-derived anchor points, achieving an intriguing integration of both paradigms.

---

## 8. Generative Action Tell-Tales: Assessing Human Motion in Synthesized Videos

**论文链接:** [http://arxiv.org/abs/2512.01803v1](http://arxiv.org/abs/2512.01803v1)

**作者:** Xavier Thomas, Youngsun Lim, Ananya Srinivasan, Audrey Zheng, Deepti Ghadiyaram

**发布时间:** 2025-12-01

### GPT解析

### 总结

该研究提出了一种新的视频生成评估指标，解决了现有方法在评估复杂人类动作时的局限性，通过融合骨骼几何特征和外观特征，实现了更准确的动作质量评估。

### 背景

尽管视频生成模型快速发展，但评估复杂人类行为的视觉和时间正确性的稳健指标仍然难以实现。现有的纯视觉编码器和多模态大语言模型严重偏向外观，缺乏时间理解能力，难以辨别生成视频中复杂的运动动态和解剖学上的不合理性。

### 目的

解决现有评估方法的局限性，引入一种基于真实世界人类动作学习到的潜在空间的新评估指标，以更准确地评估生成视频中的动作质量。

### 方法

通过融合与外观无关的人体骨骼几何特征和基于外观的特征，捕捉真实世界运动的细微差别、约束和时间平滑性。该组合特征空间提供动作合理性的稳健表示。对于给定生成视频，通过测量其底层表示与学习到的真实世界动作分布之间的距离来量化动作质量。

### 主要发现

新指标在专门设计的基准测试上比现有最先进方法实现了超过68%的显著改进，在已建立的外部基准上表现具有竞争力，并且与人类感知有更强的相关性。

### 结论

深入分析揭示了当前视频生成模型的关键局限性，并为视频生成的高级研究建立了新标准。

### 翻译

尽管视频生成模型迅速发展，但评估复杂人类行为的视觉和时间正确性的稳健指标仍然难以实现。关键问题是，现有的纯视觉编码器和多模态大语言模型(MLLMs)严重偏向外观，缺乏时间理解能力，因此难以辨别生成视频中复杂的运动动态和解剖学上的不合理性。我们通过引入一种基于真实世界人类动作学习到的潜在空间的新评估指标来解决这一差距。我们的方法首先通过融合与外观无关的人体骨骼几何特征和基于外观的特征，捕捉真实世界运动的细微差别、约束和时间平滑性。我们认为这种组合特征空间提供了动作合理性的稳健表示。对于给定的生成视频，我们的指标通过测量其底层表示与学习到的真实世界动作分布之间的距离来量化其动作质量。为了严格验证，我们开发了一个新的多方面基准，专门用于探测人类动作保真度的具有挑战性的时间方面。通过大量实验，我们显示我们的指标在基准测试上比现有最先进方法实现了超过68%的显著改进，在已建立的外部基准上表现具有竞争力，并且与人类感知有更强的相关性。我们的深入分析揭示了当前视频生成模型的关键局限性，并为视频生成的高级研究建立了新标准。


### 论文摘要

Despite rapid advances in video generative models, robust metrics for evaluating visual and temporal correctness of complex human actions remain elusive. Critically, existing pure-vision encoders and Multimodal Large Language Models (MLLMs) are strongly appearance-biased, lack temporal understanding, and thus struggle to discern intricate motion dynamics and anatomical implausibilities in generated videos. We tackle this gap by introducing a novel evaluation metric derived from a learned latent space of real-world human actions. Our method first captures the nuances, constraints, and temporal smoothness of real-world motion by fusing appearance-agnostic human skeletal geometry features with appearance-based features. We posit that this combined feature space provides a robust representation of action plausibility. Given a generated video, our metric quantifies its action quality by measuring the distance between its underlying representations and this learned real-world action distribution. For rigorous validation, we develop a new multi-faceted benchmark specifically designed to probe temporally challenging aspects of human action fidelity. Through extensive experiments, we show that our metric achieves substantial improvement of more than 68% compared to existing state-of-the-art methods on our benchmark, performs competitively on established external benchmarks, and has a stronger correlation with human perception. Our in-depth analysis reveals critical limitations in current video generative models and establishes a new standard for advanced research in video generation.

---

## 9. IGen: Scalable Data Generation for Robot Learning from Open-World Images

**论文链接:** [http://arxiv.org/abs/2512.01773v1](http://arxiv.org/abs/2512.01773v1)

**作者:** Chenghao Gu, Haolan Kang, Junchao Lin, Jinghe Wang, Duo Wu, Shuzhao Xie, Fanding Huang, Junchen Ge, Ziyang Gong, Letian Li, Hongying Zheng, Changwei Lv, Zhi Wang

**发布时间:** 2025-12-01

**备注:** 8 pages, 8 figures

### GPT解析

### 总结

IGen框架能够从开放世界图像生成高质量的视觉运动数据，支持通用机器人策略训练，性能与使用真实世界数据相当。

### 背景

通用机器人策略的兴起导致对大规模训练数据的指数级需求，但在机器人上收集数据劳动密集且通常局限于特定环境。开放世界图像捕捉了与机器人操作任务自然对齐的丰富真实世界场景，为低成本、大规模机器人数据采集提供了有前景的途径。

### 目的

为了弥合开放世界图像与机器人学习之间的差距，作者提出了IGen框架，该框架可从开放世界图像可扩展地生成真实的视觉观察和可执行的动作。

### 方法

IGen首先将非结构化的2D像素转换为适合场景理解和操作的结构化3D场景表示。然后，它利用视觉-语言模型的推理能力，将场景特定的任务指令转换为高层计划，并生成低级动作作为末端执行器姿态序列。从这些姿态中，它合成动态场景演变并渲染时间连贯的视觉观察。

### 主要发现

实验验证了IGen生成的视觉运动数据的高质量，并显示仅使用IGen合成数据训练的策略实现了与使用真实世界数据训练的策略相当的性能。

### 结论

IGen框架有潜力支持从开放世界图像进行可扩展的数据生成，用于通用机器人策略训练。

### 翻译

通用机器人策略的兴起导致对大规模训练数据的指数级需求。然而，在机器人上收集数据劳动密集且通常局限于特定环境。相比之下，开放世界图像捕捉了与机器人操作任务自然对齐的丰富真实世界场景，为低成本、大规模机器人数据采集提供了有前景的途径。尽管有这种潜力，但缺乏相关的机器人动作阻碍了开放世界图像在机器人学习中的实际应用，使得这一丰富的视觉资源 largely 未被开发。为了弥合这一差距，我们提出了IGen，一个从开放世界图像可扩展地生成真实视觉观察和可执行动作的框架。IGen首先将非结构化的2D像素转换为适合场景理解和操作的结构化3D场景表示。然后，它利用视觉-语言模型的推理能力，将场景特定的任务指令转换为高层计划，并生成低级动作作为SE(3)末端执行器姿态序列。从这些姿态中，它合成动态场景演变并渲染时间连贯的视觉观察。实验验证了IGen生成的视觉运动数据的高质量，并显示仅使用IGen合成数据训练的策略实现了与使用真实世界数据训练的策略相当的性能。这突显了IGen从开放世界图像支持通用机器人策略训练的可扩展数据生成的潜力。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决机器人学习中大规模训练数据获取困难的问题。在真实机器人上收集数据既耗时又昂贵，而且通常只限于特定环境。随着通用机器人策略的发展，对大规模训练数据的需求呈指数级增长，这成为机器人策略在多样化现实场景中泛化的根本瓶颈。开放世界图像可以以极低成本获取，包含丰富的现实场景，为解决这一问题提供了有希望的途径。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先认识到开放世界图像资源丰富但缺乏机器人动作信息，而现有方法要么需要显式重建物理工作空间，要么无法提供明确的机器人动作。他们设计了一个统一框架，借鉴了多种现有技术：利用大型视觉模型重建场景为3D点云，使用Segment Anything Model获取对象掩码，采用DINOv2提取场景特征，并利用视觉-语言模型进行任务规划和动作转换。这些技术的有机结合使系统能够从非结构化图像生成结构化的机器人训练数据。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是将非结构化的开放世界图像转换为结构化的3D场景表示，利用视觉-语言模型进行任务规划和动作生成，然后合成动态场景演变和渲染视觉观察。整体流程分为三阶段：1)场景重建：将图像转换为3D点云和空间关键点；2)行动规划：利用视觉-语言模型进行任务分解和动作生成；3)观察合成：在仿真环境中执行动作，捕获点云序列并渲染成视觉观察，最终形成配对的视觉-动作数据。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)提出从开放世界图像生成可扩展视觉-动作数据集的框架；2)将非结构化图像转换为可操作的3D场景表示；3)引入无需仿真的点云合成方法；4)证明仅用生成数据训练的机器人策略可在现实世界成功执行任务。相比之前工作，IGen不依赖显式物理重建，可从任意图像生成数据；提供明确的机器人动作，能处理复杂任务；计算效率比基线方法高出30-200倍；不依赖精确物理属性估计，在复杂交互任务中更具优势。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'IGen提出了一种创新框架，能够将开放世界图像转化为高质量的机器人视觉-动作数据，使机器人能够仅从互联网图像中学习并成功执行现实世界操作任务，无需人工标注或真实机器人数据收集。'}


### 论文摘要

The rise of generalist robotic policies has created an exponential demand for large-scale training data. However, on-robot data collection is labor-intensive and often limited to specific environments. In contrast, open-world images capture a vast diversity of real-world scenes that naturally align with robotic manipulation tasks, offering a promising avenue for low-cost, large-scale robot data acquisition. Despite this potential, the lack of associated robot actions hinders the practical use of open-world images for robot learning, leaving this rich visual resource largely unexploited. To bridge this gap, we propose IGen, a framework that scalably generates realistic visual observations and executable actions from open-world images. IGen first converts unstructured 2D pixels into structured 3D scene representations suitable for scene understanding and manipulation. It then leverages the reasoning capabilities of vision-language models to transform scene-specific task instructions into high-level plans and generate low-level actions as SE(3) end-effector pose sequences. From these poses, it synthesizes dynamic scene evolution and renders temporally coherent visual observations. Experiments validate the high quality of visuomotor data generated by IGen, and show that policies trained solely on IGen-synthesized data achieve performance comparable to those trained on real-world data. This highlights the potential of IGen to support scalable data generation from open-world images for generalist robotic policy training.

---

## 10. Weight Space Representation Learning with Neural Fields

**论文链接:** [http://arxiv.org/abs/2512.01759v1](http://arxiv.org/abs/2512.01759v1)

**作者:** Zhuoqian Yang, Mathieu Salzmann, Sabine Süsstrunk

**发布时间:** 2025-12-01

**备注:** 12 pages body, 9 pages appendix

### GPT解析

### 总结

本研究探讨了权重作为有效表示的潜力，特别是在神经场领域，通过预训练基础模型和低秩适应（LoRA）约束优化空间，在权重空间中诱导结构，实现了高质量的表示。

### 背景

神经场作为表示学习的一种方法，其权重空间的结构和表示能力值得进一步研究，现有的权重空间方法在生成质量上可能存在局限。

### 目的

研究权重作为有效表示的潜力，探索通过约束优化空间在权重空间中诱导结构的方法，并评估其在不同任务中的表现。

### 方法

使用预训练基础模型和低秩适应（LoRA）来约束优化空间，采用乘性LoRA权重，在2D和3D数据的重建、生成和分析任务中进行测试，并与潜在扩散模型结合使用。

### 主要发现

乘性LoRA权重在多种任务中实现了高质量的表示，同时具有独特性和语义结构；与潜在扩散模型结合使用时，能够比现有权重空间方法实现更高质量的生成。

### 结论

通过预训练基础模型和低秩适应约束优化空间，可以在权重空间中诱导有用的结构；乘性LoRA权重是一种有效的表示方法，适用于多种任务，并能与潜在扩散模型结合以实现高质量的生成。

### 翻译

在这项工作中，我们研究了权重作为有效表示的潜力，重点关注神经场。我们的关键见解是，通过预训练的基础模型和低秩适应（LoRA）约束优化空间，可以在权重空间中诱导结构。在2D和3D数据的重建、生成和分析任务中，我们发现乘性LoRA权重实现了高质量的表示，同时展现出独特性和语义结构。当与潜在扩散模型一起使用时，乘性LoRA权重能够实现比现有权重空间方法更高质量的生成。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要研究神经网络权重是否能作为有效的数据表示。传统上，神经网络权重被视为优化的不透明副产品，难以解释或操作。这个问题很重要，因为如果权重可以作为有效的数据表示，将为数据表示提供新范式，可能具有语义结构，可用于重建、生成和分析任务，并提供更高效的数据压缩方式。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先认识到神经网络权重具有模糊性，功能相同的网络在权重空间中可能相距甚远。关键洞察是通过适当的归纳偏置约束网络权重，可将混乱参数转化为有组织的表示。作者借鉴了低秩自适应（LoRA）技术，发现标准加性LoRA不足，因此引入乘性LoRA（mLoRA），并采用非对称掩码技术处理排列对称性问题。方法还借鉴了隐式神经表示、扩散模型和变分自解码等现有技术。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是通过预训练基础模型和低秩自适应约束优化空间，在权重空间中诱导结构；使用乘性LoRA而非加性LoRA，因为它与生成神经场中的调制机制自然对齐；通过非对称掩码处理排列对称性。整体流程包括：1)基础模型训练；2)实例拟合；3)权重空间表示；4)扩散模型训练；5)生成新实例。每个实例由其网络权重表示，LoRA方法使用LoRA权重表示，扩散模型用于生成新实例。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)首次证明权重可作为具有语义结构的数据表示；2)引入乘性LoRA提供更好的表示质量；3)将非对称掩码应用于神经场解决排列对称性；4)设计分层LoRA层编码器架构；5)跨任务验证权重空间表示的可行性。相比之前工作，本文不是构建将权重作为输入的编码器，而是直接研究优化衍生的权重作为表示；不设计超网络预测权重，而是研究梯度下降优化的权重直接作为表示；不仅探索权重空间生成，还研究权重中的语义结构。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '本文通过引入乘性低秩自适应和结构化约束，证明了神经网络权重可以成为具有语义结构的高效数据表示，为重建、生成和分析任务提供了新的范式。'}


### 论文摘要

In this work, we investigate the potential of weights to serve as effective representations, focusing on neural fields. Our key insight is that constraining the optimization space through a pre-trained base model and low-rank adaptation (LoRA) can induce structure in weight space. Across reconstruction, generation, and analysis tasks on 2D and 3D data, we find that multiplicative LoRA weights achieve high representation quality while exhibiting distinctiveness and semantic structure. When used with latent diffusion models, multiplicative LoRA weights enable higher-quality generation than existing weight-space methods.

---

## 11. DiG-Flow: Discrepancy-Guided Flow Matching for Robust VLA Models

**论文链接:** [http://arxiv.org/abs/2512.01715v1](http://arxiv.org/abs/2512.01715v1)

**作者:** Wanpeng Zhang, Ye Wang, Hao Luo, Haoqi Yuan, Yicheng Feng, Sipeng Zheng, Qin Jin, Zongqing Lu

**发布时间:** 2025-12-01

### GPT解析

### 总结

本文提出了DiG-Flow框架，通过几何正则化增强Vision-Language-Action模型的鲁棒性，解决了模型在分布偏移和复杂多步骤任务上性能下降的问题。

### 背景

Vision-Language-Action模型使用flow matching训练后在机器人操作任务上表现出色，但在分布偏移和复杂多步骤任务上性能往往下降，表明学习到的表示可能无法稳健捕获任务相关语义。

### 目的

引入DiG-Flow框架，通过几何正则化增强VLA模型的鲁棒性，提高其在分布偏移和复杂多步骤任务上的性能。

### 方法

DiG-Flow利用观察和动作嵌入分布间的差异作为几何信号，计算嵌入经验分布间的差异度量，通过单调函数映射为调制权重，并在flow matching前对观察嵌入应用残差更新，这种干预在表示级别操作而不修改flow matching路径。

### 主要发现

理论保证显示差异引导的训练可减少训练目标，引导推理收敛具有收缩性；实证上DiG-Flow以可忽略开销集成现有架构并提高性能，在复杂多步骤任务和有限训练数据下改进尤为明显。

### 结论

DiG-Flow通过几何正则化有效提升了VLA模型在分布偏移和复杂多步骤任务上的鲁棒性和性能。

### 翻译

使用flow matching训练的Vision-Language-Action模型在机器人操作任务上展现出令人印象深刻的能力。然而，它们在分布偏移和复杂多步骤任务上的性能往往下降，这表明学习到的表示可能无法稳健地捕获任务相关的语义。我们引入了DiG-Flow，一个通过几何正则化增强VLA鲁棒性的原则性框架。我们的关键洞察是，观察和动作嵌入之间的分布差异提供了有意义的几何信号：较低的传输成本表示兼容的表示，而较高的成本则表明潜在的不对齐。DiG-Flow计算观察和动作嵌入的经验分布之间的差异度量，通过单调函数将其映射到调制权重，并在flow matching之前对观察嵌入应用残差更新。关键的是，这种干预在表示级别操作，而不修改flow matching路径或目标向量场。我们提供理论保证表明，差异引导的训练可证明减少训练目标，并且引导推理收敛具有收缩性。实证上，DiG-Flow以可忽略的开销集成到现有的VLA架构中，并持续提高性能，在复杂多步骤任务和有限的训练数据下改进尤为明显。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决Vision-Language-Action (VLA)模型在分布变化和复杂多步任务上性能下降的问题。这个问题很重要，因为机器人系统需要在真实环境中处理各种不可预测的变化（如光照变化、纹理变化或相机角度扰动），并且经常需要执行多步骤任务。如果模型对分布变化敏感，在实际应用中就会表现出脆弱性，限制了机器人在真实世界中的实用性和可靠性，特别是在早期步骤错误会级联导致最终失败的多步任务中。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者从几何正则化的角度出发，提出观察和动作嵌入之间的分布差异提供了有意义的几何信号：较低的传输成本表示兼容的表示，较高的成本表示潜在的失配。作者借鉴了现有的flow matching和最优传输理论，但创新性地将最优传输作为辅助信号来调制表示学习，而不是像OT-CFM那样使用最优传输修改flow matching轨迹。作者设计了包含三个主要组件的框架：差异函数、单调权重映射和轻量级残差算子，通过这种差异引导的方式增强VLA模型的鲁棒性。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是通过观察和动作嵌入之间的分布差异来增强VLA模型的鲁棒性。整体实现流程是：1)计算观察和动作嵌入的经验分布之间的差异度量(默认使用Wasserstein距离)；2)通过单调递减函数将差异转换为调制权重；3)应用轻量级残差更新调整观察特征，更新量由调制权重控制；4)将DiG-Block插入到预训练VLM骨干网络和flow matching头部之间；5)训练时使用带门控的目标函数；6)推理时可使用相同机制或可选的迭代细化方案(DiG-Refine)改进预测动作。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)首次引入基于几何正则化的框架增强VLA鲁棒性；2)在表示级别进行干预，不改变概率路径或目标向量场；3)提供理论保证证明差异引导的训练可减少目标，引导的推理可收敛；4)即插即用集成到现有架构，计算开销小。相比之前工作(如OT-CFM)，DiG-Flow使用最优传输作为辅助信号调制表示学习，而非直接修改flow matching动态过程，因此在保持底层动作生成机制不变的情况下提高鲁棒性。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'DiG-Flow通过引入基于观察和动作嵌入分布差异的几何正则化框架，显著提高了VLA模型在分布变化和复杂多步任务下的鲁棒性，同时保持与现有架构的兼容性和计算效率。'}


### 论文摘要

Vision-Language-Action (VLA) models trained with flow matching have demonstrated impressive capabilities on robotic manipulation tasks. However, their performance often degrades under distribution shift and on complex multi-step tasks, suggesting that the learned representations may not robustly capture task-relevant semantics. We introduce DiG-Flow, a principled framework that enhances VLA robustness through geometric regularization. Our key insight is that the distributional discrepancy between observation and action embeddings provides a meaningful geometric signal: lower transport cost indicates compatible representations, while higher cost suggests potential misalignment. DiG-Flow computes a discrepancy measure between empirical distributions of observation and action embeddings, maps it to a modulation weight via a monotone function, and applies residual updates to the observation embeddings before flow matching. Crucially, this intervention operates at the representation level without modifying the flow matching path or target vector field. We provide theoretical guarantees showing that discrepancy-guided training provably decreases the training objective, and that guided inference refinement converges with contraction. Empirically, DiG-Flow integrates into existing VLA architectures with negligible overhead and consistently improves performance, with particularly pronounced gains on complex multi-step tasks and under limited training data.

---

## 12. A unified framework for geometry-independent operator learning in cardiac electrophysiology simulations

**论文链接:** [http://arxiv.org/abs/2512.01702v1](http://arxiv.org/abs/2512.01702v1)

**作者:** Bei Zhou, Cesare Corrado, Shuang Qian, Maximilian Balmus, Angela W. C. Lee, Cristobal Rodero, Marco J. W. Gotte, Luuk H. G. A. Hopman, Mengyun Qiao, Steven Niederer

**发布时间:** 2025-12-01

### GPT解析

### 总结

研究人员开发了一种与几何无关的算子学习框架，能够快速准确地预测不同心房解剖结构中的电活动模式，为心律失常的实时临床治疗提供了新方法

### 背景

精确的心房电活动图谱对于心律失常的个性化治疗至关重要，然而生物物理详细模拟对于实时临床应用或人群规模分析仍然计算密集

### 目的

引入一种与几何无关的算子学习框架，预测不同左心房解剖结构中的局部激活时间场，实现接近即时的推理

### 方法

使用GPU加速的电生理学求解器生成308,700个模拟数据集，系统变化147个患者特异性几何结构上的起搏点和传导特性；所有数据在通用心房坐标系中表示；设计具有视觉变换器骨干的神经算子学习结构和电生理输入到LAT场的映射

### 主要发现

模型在455毫秒最大模拟时间内平均预测误差为5.1毫秒，优于既定算子学习方法，每个样本推理时间仅0.12毫秒

### 结论

该框架建立了跨可变解剖域学习域不变生物物理映射的通用策略，使计算电生理学能够集成到实时和大规模临床工作流中

### 翻译

精确的心房电活动图谱对于心律失常的个性化治疗至关重要，然而生物物理详细模拟对于实时临床使用或人群规模分析仍然计算密集。我们在此介绍了一种与几何无关的算子学习框架，能够预测不同左心房解剖结构中的局部激活时间场，并实现接近即时的推理。我们使用GPU加速的电生理学求解器生成了308,700个模拟数据集，系统性地变化了147个来自两个独立临床队列的患者特异性几何结构上的多个起搏点和生理变化的传导特性。所有解剖和功能数据都在通用心房坐标系中表示，提供了一致的表示，将电生理模式与网格拓扑解耦。在这个坐标空间中，我们设计了一个具有视觉变换器骨干的神经算子，学习从结构和电生理输入到LAT场的映射。在455毫秒的最大模拟时间内，模型平均预测误差为5.1毫秒，优于既定的算子学习方法，每个样本推理时间为0.12毫秒。我们的框架建立了跨可变解剖域学习域不变生物物理映射的通用策略，使计算电生理学能够集成到实时和大规模临床工作流中。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决心脏电生理学模拟中计算效率低的问题，传统有限元方法虽然准确但计算成本高，无法满足临床实时应用需求。这个问题很重要，因为房颤是最常见的心律失常，影响全球约4630万人，是中风、心力衰竭和心源性猝死的主要原因。准确的心脏电激活映射对个性化治疗至关重要，但传统方法无法在手术过程中实时分析电解剖映射数据。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先认识到传统有限元方法的局限性，然后考虑使用神经算子方法加速计算。发现现有神经算子框架难以处理不规则几何形状，因此提出使用通用心房坐标系统(UAC)将不同解剖结构映射到统一空间。借鉴了DeepONet和FNO等神经算子框架的基本思想，UAC系统作为标准化坐标表示，Vision Transformer架构用于空间特征学习，以及拉丁超立方采样方法生成多样化训练数据。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是创建一个与几何无关的神经算子学习框架，通过UAC系统将不同患者解剖结构映射到统一参考空间，使用大规模GPU加速模拟生成训练数据，并设计基于Vision Transformer的模型学习从解剖和电生理输入到局部激活时间(LAT)场的映射。流程包括：1)数据准备(获取患者解剖结构、图像分割、纤维建模)；2)模拟数据生成(使用GPU加速模拟器，从7个起搏点模拟300种电导率组合)；3)模型设计(编码器-解码器架构，处理多种输入特征)；4)训练与评估(使用特定损失函数，评估预测精度和空间结构相似性)。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)几何无关的神经算子框架，使用UAC系统统一不同解剖结构；2)大规模数据生成，创建308,700个模拟LAT图；3)基于Vision Transformer的架构，结合全局上下文和局部定位能力；4)跨域泛化能力，对不同数据采集协议鲁棒。相比之前工作，计算速度快5-6个数量级；比其他神经算子更好地处理不规则几何和局部梯度变化；比几何映射方法能处理更复杂拓扑结构；比卷积网络更好捕捉全局效应和长程依赖。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '这篇论文提出了一种基于通用心房坐标系统和Vision Transformer的几何无关神经算子学习框架，能够在毫秒级时间内准确预测不同患者的左心房电激活模式，为心脏电生理学的实时临床应用和大规模分析提供了新途径。'}


### 论文摘要

Accurate maps of atrial electrical activation are essential for personalised treatment of arrhythmias, yet biophysically detailed simulations remain computationally intensive for real-time clinical use or population-scale analyses. Here we introduce a geometry-independent operator-learning framework that predicts local activation time (LAT) fields across diverse left atrial anatomies with near-instantaneous inference. We generated a dataset of 308,700 simulations using a GPU-accelerated electrophysiology solver, systematically varying multiple pacing sites and physiologically varied conduction properties across 147 patient-specific geometries derived from two independent clinical cohorts. All anatomical and functional data are expressed in a Universal Atrium Coordinate system, providing a consistent representation that decouples electrophysiological patterns from mesh topology. Within this coordinate space, we designed a neural operator with a vision-transformer backbone to learn the mapping from structural and electrophysiological inputs to LAT fields. With a mean prediction error of 5.1 ms over a 455 ms maximum simulation time, the model outperforms established operator-learning approaches and performs inference in 0.12 ms per sample. Our framework establishes a general strategy for learning domain-invariant biophysical mappings across variable anatomical domains and enables integration of computational electrophysiology into real-time and large-scale clinical workflows.

---

## 13. Integrating Artificial Intelligence and Mixed Integer Linear Programming: Explainable Graph-Based Instance Space Analysis in Air Transportation

**论文链接:** [http://arxiv.org/abs/2512.01698v1](http://arxiv.org/abs/2512.01698v1)

**作者:** Artur Guerra Rosa, Felipe Tavares Loureiro, Marcus Vinicius Santos da Silva, Andréia Elizabeth Silva Barros, Silvia Araújo dos Reis, Victor Rafael Rezende Celestino

**发布时间:** 2025-12-01

**备注:** 25 pages, 6 figures, presented at XXII SITRAER 2025, in processes for submission to JATM

### GPT解析

### 总结

本研究探讨了人工智能与混合整数线性规划在航空运输复杂优化问题中的整合应用，特别关注可解释性。研究验证了使用图神经网络从MILP实例中提取结构特征嵌入的有效性，并通过不同神经网络架构和降维技术进行了分析。

### 背景

航空运输领域面临复杂的优化挑战，需要结合人工智能与传统优化方法，同时保持决策过程的可解释性。

### 目的

验证使用图神经网络从MILP实例中提取结构特征嵌入的有效性，以air05机组调度问题为案例研究。

### 方法

将MILP实例转换为异构二分图建模变量与约束关系；训练图卷积网络(GCN)和图注意力网络(GAT)生成节点嵌入；使用实例空间分析(ISA)通过线性和非线性降维技术评估表示效果。

### 主要发现

GCN架构表现更佳，能捕获全局拓扑并形成良好聚类；GAT模型未能有效组织约束空间；简单图架构可映射航空物流问题稀疏拓扑，无需手动特征工程。

### 结论

这种结构意识为开发能提高求解器在安全关键环境中性能的学习优化(L2O)代理提供了验证基础，同时增强了实例复杂性的可解释性。

### 翻译

本文分析了人工智能与混合整数线性规划的结合，以解决航空运输中的复杂优化挑战，并关注可解释性。研究旨在验证使用图神经网络从MILP实例中提取结构特征嵌入的有效性，以air05机组调度问题为例。MILP实例被转换为异构二分图，以建模变量和约束之间的关系。训练了图卷积网络和图注意力网络两种神经网络架构来生成节点嵌入。这些表示通过线性和非线性降维技术使用实例空间分析进行了评估。分析显示PCA无法区分聚类结构，需要非线性降维来可视化嵌入拓扑。GCN架构表现更优，能捕获全局拓扑并为变量和约束定义良好聚类。相比之下，GAT模型未能组织约束空间。研究证实，简单的图架构可以有效地映射航空物流问题的稀疏拓扑，无需手动特征工程，提高了实例复杂性的可解释性。这种结构意识为开发能够提高求解器在安全关键环境中性能的学习优化代理提供了验证基础。


### 论文摘要

This paper analyzes the integration of artificial intelligence (AI) with mixed integer linear programming (MILP) to address complex optimization challenges in air transportation with explainability. The study aims to validate the use of Graph Neural Networks (GNNs) for extracting structural feature embeddings from MILP instances, using the air05 crew scheduling problem. The MILP instance was transformed into a heterogeneous bipartite graph to model relationships between variables and constraints. Two neural architectures, Graph Convolutional Networks (GCN) and Graph Attention Networks (GAT) were trained to generate node embeddings. These representations were evaluated using Instance Space Analysis (ISA) through linear (PCA) and non-linear (UMAP, t-SNE) dimensionality reduction techniques. Analysis revealed that PCA failed to distinguish cluster structures, necessitating non-linear reductions to visualize the embedding topology. The GCN architecture demonstrated superior performance, capturing global topology with well-defined clusters for both variables and constraints. In contrast, the GAT model failed to organize the constraint space. The findings confirm that simpler graph architectures can effectively map the sparse topology of aviation logistics problems without manual feature engineering, contributing to explainability of instance complexity. This structural awareness provides a validated foundation for developing future Learning to Optimize (L2O) agents capable of improving solver performance in safety-critical environments.

---

## 14. Open-world Hand-Object Interaction Video Generation Based on Structure and Contact-aware Representation

**论文链接:** [http://arxiv.org/abs/2512.01677v1](http://arxiv.org/abs/2512.01677v1)

**作者:** Haodong Yan, Hang Yu, Zhide Zhong, Weilin Yuan, Xin Gong, Zehang Luo, Chengxi Heyu, Junfeng Li, Wenxuan Song, Shunbo Zhou, Haoang Li

**发布时间:** 2025-12-01

### GPT解析

### 总结

该研究提出了一种结构和接触感知的表示方法，用于生成真实的手物交互视频，解决了2D和3D表示之间的权衡问题，通过联合生成范式实现了物理真实且时间连贯的视频生成。

### 背景

生成真实的手物交互(HOI)视频具有挑战性，主要困难在于建模物理约束（如手物接触和遮挡）。当前方法将HOI表示作为辅助生成目标指导视频合成，但存在2D和3D表示之间的两难选择，无法同时保证可扩展性和交互保真度。

### 目的

解决2D和3D表示之间的局限性，提出一种能够捕捉手物接触、遮挡和整体结构上下文的无3D标注的表示方法，实现物理真实且可扩展的手物交互视频生成。

### 方法

提出结构和接触感知的表示方法，引入联合生成范式，采用共享和专业化策略生成面向交互的表示和视频，无需3D标注即可学习细粒度的交互物理特性。

### 主要发现

在两个真实世界数据集上的实验表明，该方法在生成物理真实且时间连贯的HOI视频方面优于最先进方法；在开放世界场景中表现出强大的泛化能力，突显了可扩展设计的好处。

### 结论

所提出的方法成功解决了手物交互视频生成的挑战，在物理真实性、时间连贯性和泛化能力方面表现优异，特别是对开放世界场景有良好的适应性。

### 翻译

生成真实的手物交互(HOI)视频是一个重大挑战，因为建模物理约束（例如手与操作物体之间的接触和遮挡）很困难。当前方法将HOI表示作为辅助生成目标来指导视频合成。然而，2D和3D表示之间存在两难选择，无法同时保证可扩展性和交互保真度。为了解决这一局限性，我们提出了一种结构和接触感知的表示方法，能够捕捉手物接触、手物遮挡和整体结构上下文，无需3D标注。这种面向交互且可扩展的监督信号使模型能够学习细粒度的交互物理特性并泛化到开放世界场景。为了充分利用所提出的表示方法，我们引入了一种联合生成范式，采用共享和专业化策略，生成面向交互的表示和视频。广泛实验证明，我们的方法在两个真实世界数据集上生成物理真实且时间连贯的HOI视频方面优于最先进的方法。此外，我们的方法在具有挑战性的开放世界场景中表现出强大的泛化能力，突显了可扩展设计的好处。我们的项目页面是https://hgzn258.github.io/SCAR/。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决生成真实且符合物理规律的手部-物体交互视频的问题。这个问题很重要，因为手部-物体交互视频生成在机器人学习、增强现实和人类行为分析等领域有广泛应用，而现有方法难以同时保证物理真实性和视觉质量，且难以推广到开放世界场景。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者分析了当前HOI表示的两难困境：2D表示缺乏结构上下文和接触信息，3D表示难以扩展。因此设计了结构化和接触感知表示，结合接触增强的手-物体轮廓和深度图。方法借鉴了VLM、SAM2、视频深度估计和Diffusion Transformer等现有工作，但提出了新的组合方式和联合生成范式。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是设计一种新的HOI表示方法作为监督信号，并采用联合生成范式同时生成视频和表示。流程包括：1)使用VLM和SAM2提取分割，估计接触区域，获取深度图构建表示；2)使用3D VAE编码到统一潜在空间；3)通过分层联合去噪器实现共享语义和专业化细节的生成；4)同时解码生成最终视频和HOI表示。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)结构化和接触感知表示，同时捕捉接触、定位和结构信息；2)联合生成范式避免多阶段累积误差；3)大规模数据集构建。相比之前工作，本文的表示结合了2D和3D表示的优点，生成范式更高效，且能推广到开放世界场景，生成更物理真实的视频。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '本文提出了一种基于结构化和接触感知表示的联合生成方法，实现了物理真实且具有强泛化能力的开放世界手-物体交互视频生成。'}


### 论文摘要

Generating realistic hand-object interactions (HOI) videos is a significant challenge due to the difficulty of modeling physical constraints (e.g., contact and occlusion between hands and manipulated objects). Current methods utilize HOI representation as an auxiliary generative objective to guide video synthesis. However, there is a dilemma between 2D and 3D representations that cannot simultaneously guarantee scalability and interaction fidelity. To address this limitation, we propose a structure and contact-aware representation that captures hand-object contact, hand-object occlusion, and holistic structure context without 3D annotations. This interaction-oriented and scalable supervision signal enables the model to learn fine-grained interaction physics and generalize to open-world scenarios. To fully exploit the proposed representation, we introduce a joint-generation paradigm with a share-and-specialization strategy that generates interaction-oriented representations and videos. Extensive experiments demonstrate that our method outperforms state-of-the-art methods on two real-world datasets in generating physics-realistic and temporally coherent HOI videos. Furthermore, our approach exhibits strong generalization to challenging open-world scenarios, highlighting the benefit of our scalable design. Our project page is https://hgzn258.github.io/SCAR/.

---

## 15. 论文ID: 2512.01626v1

**错误:** 无法读取论文信息 - [Errno 2] No such file or directory: 'storage/result/2512.01626v1.json'

---

## 16. 论文ID: 2512.01616v1

**错误:** 无法读取论文信息 - [Errno 2] No such file or directory: 'storage/result/2512.01616v1.json'

---

## 17. 论文ID: 2512.01591v1

**错误:** 无法读取论文信息 - [Errno 2] No such file or directory: 'storage/result/2512.01591v1.json'

---

## 18. 论文ID: 2512.01537v1

**错误:** 无法读取论文信息 - [Errno 2] No such file or directory: 'storage/result/2512.01537v1.json'

---

## 19. 论文ID: 2512.01519v1

**错误:** 无法读取论文信息 - [Errno 2] No such file or directory: 'storage/result/2512.01519v1.json'

---

## 20. Learning Reduced Representations for Quantum Classifiers

**论文链接:** [http://arxiv.org/abs/2512.01509v1](http://arxiv.org/abs/2512.01509v1)

**作者:** Patrick Odagiu, Vasilis Belis, Lennart Schulze, Panagiotis Barkoutsos, Michele Grossi, Florentin Reiter, Günther Dissertori, Ivano Tavernelli, Sofia Vallecorsa

**发布时间:** 2025-12-01

**DOI:** 10.1007/s42484-025-00331-y

### GPT解析

### 总结

该研究通过降维方法使具有大量特征的数据集适用于量子机器学习算法，特别关注希格斯玻色子产生的二元分类问题，设计的Sinkclass自动编码器方法表现优于基线方法40%。

### 背景

当前具有大量特征的数据集超出了量子机器学习算法的适用范围，限制了量子机器学习在更广泛数据集上的应用。

### 目的

通过降维方法扩展量子机器学习的适用范围，使其能够处理具有大量特征的数据集，并以粒子物理学数据集为例进行验证。

### 方法

研究六种传统特征提取算法和五种基于自动编码器的降维模型，应用于具有67个特征的粒子物理学数据集。使用这些模型生成的降维表示训练量子支持向量机，解决希格斯玻色子在LHC质子碰撞中是否产生的二元分类问题。

### 主要发现

自动编码器方法能够学习到更好的数据低维表示，其中设计的Sinkclass自动编码器方法比基线方法表现好40%。

### 结论

所开发的方法扩展了量子机器学习在更广泛数据集上的适用性，同时为在此背景下进行有效降维提供了指导方案。

### 翻译

当前由大量特征指定的数据集超出了量子机器学习算法的适用范围。解决这一困境的直接解决方案是在将数据传递给量子算法之前应用降维方法。我们研究了六种传统特征提取算法和五种基于自动编码器的降维模型，应用于具有67个特征的粒子物理学数据集。然后使用这些模型生成的降维表示来训练量子支持向量机，以解决二元分类问题：希格斯玻色子是否在LHC的质子碰撞中产生。我们表明，自动编码器方法能够学习到数据的更好的低维表示，其中我们设计的Sinkclass自动编码器方法比基线方法表现好40%。这里开发的方法扩展了量子机器学习在更广泛数据集上的适用性。此外，我们提供了在此背景下进行有效降维的方案。


### 论文摘要

Data sets that are specified by a large number of features are currently outside the area of applicability for quantum machine learning algorithms. An immediate solution to this impasse is the application of dimensionality reduction methods before passing the data to the quantum algorithm. We investigate six conventional feature extraction algorithms and five autoencoder-based dimensionality reduction models to a particle physics data set with 67 features. The reduced representations generated by these models are then used to train a quantum support vector machine for solving a binary classification problem: whether a Higgs boson is produced in proton collisions at the LHC. We show that the autoencoder methods learn a better lower-dimensional representation of the data, with the method we design, the Sinkclass autoencoder, performing 40% better than the baseline. The methods developed here open up the applicability of quantum machine learning to a larger array of data sets. Moreover, we provide a recipe for effective dimensionality reduction in this context.

---

## 21. CourtMotion: Learning Event-Driven Motion Representations from Skeletal Data for Basketball

**论文链接:** [http://arxiv.org/abs/2512.01478v1](http://arxiv.org/abs/2512.01478v1)

**作者:** Omer Sela, Michael Chertok, Lior Wolf

**发布时间:** 2025-12-01

### GPT解析

### 总结

本文介绍了CourtMotion，一个用于分析和预测职业篮球比赛中事件和战术发展的时空建模框架。

### 背景

预测篮球事件需要理解身体运动模式及其在比赛中的语义意义。传统仅使用球员位置的方法无法捕捉身体方向、防守姿势或投篮准备动作等关键指标。

### 目的

开发一个能够捕捉更细微运动模式和球员交互的框架，以提高篮球事件预测的准确性。

### 方法

采用两阶段方法，首先通过图神经网络处理骨骼跟踪数据以捕捉细微的运动模式，然后采用具有专门注意力机制的Transformer架构来建模球员交互。引入事件投影头，将球员运动明确连接到传球、投篮和抢断等篮球事件。

### 主要发现

在NBA跟踪数据上的实验表明，与仅基于位置的基线相比有显著改进：与最先进的基于位置的模型相比，轨迹预测误差减少了35%，并且在关键篮球分析任务中表现一致提升。

### 结论

预训练模型可以作为多个下游任务的强大基础，在捡球检测、投篮者识别、助攻预测、投篮位置分类和投篮类型识别等方面展现出比现有方法更显著的改进。

### 翻译

本文提出了CourtMotion，一个用于分析和预测职业篮球比赛中事件和战术发展的时空建模框架。预测篮球事件需要理解身体运动模式及其在比赛中的语义意义。传统仅使用球员位置的方法无法捕捉身体方向、防守姿势或投篮准备动作等关键指标。我们的两阶段方法首先通过图神经网络处理骨骼跟踪数据以捕捉细微的运动模式，然后采用具有专门注意力机制的Transformer架构来建模球员交互。我们引入了事件投影头，将球员运动明确连接到传球、投篮和抢断等篮球事件，训练模型将身体运动模式与其战术目的相关联。在NBA跟踪数据上的实验表明，与仅基于位置的基线相比有显著改进：与最先进的基于位置的模型相比，轨迹预测误差减少了35%，并且在关键篮球分析任务中表现一致提升。所得的预训练模型可作为多个下游任务的强大基础，在捡球检测、投篮者识别、助攻预测、投篮位置分类和投篮类型识别等方面展现出比现有方法更显著的改进。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决如何更好地分析和预测篮球比赛中的事件和战术问题。传统方法仅使用球员位置数据，无法捕捉身体姿态、防守姿势等关键指标。这个问题很重要，因为篮球是动态团队运动，理解球员间互动和预测关键事件对改进战术、提高球队表现和开发自动化分析系统都具有重要意义。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者思考的关键是有效篮球分析需要同时理解运动模式及其战术结果。他们借鉴了图神经网络处理骨架数据、Transformer架构建模序列关系，以及Baller2vec++的多实体transformer方法。在此基础上，作者设计了两阶段方法：先用GNN处理骨架数据捕捉细微动作，再用Transformer建模球员互动，并通过事件投影头将动作与篮球事件联系起来。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是同时理解运动模式及其在比赛背景下的战术结果。整体流程分为三部分：1) 用图神经网络处理30Hz的3D关节数据生成5Hz姿态嵌入；2) 将当前状态、前瞻轨迹等特征输入Transformer处理时空关系；3) 通过事件投影头将运动模式映射到过去、现在、未来三个时间窗口的篮球事件上，实现多任务训练。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1) 利用骨架数据而非仅位置数据捕捉球员姿态和意图；2) 设计两阶段架构结合GNN和Transformer；3) 引入多任务训练同时预测轨迹和事件；4) 使用肩部法线向量作为骨架数据的轻量替代。相比之前工作，CourtMotion能捕捉传球意图、投篮准备等微妙信号，突破了传统方法假设球员轨迹统计独立的限制，在预测准确性和理解比赛动态方面有显著提升。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'CourtMotion通过结合球员骨架数据和位置数据，利用图神经网络和Transformer架构，显著提升了篮球比赛中关键事件和战术的预测准确性。'}


### 论文摘要

This paper presents CourtMotion, a spatiotemporal modeling framework for analyzing and predicting game events and plays as they develop in professional basketball. Anticipating basketball events requires understanding both physical motion patterns and their semantic significance in the context of the game. Traditional approaches that use only player positions fail to capture crucial indicators such as body orientation, defensive stance, or shooting preparation motions. Our two-stage approach first processes skeletal tracking data through Graph Neural Networks to capture nuanced motion patterns, then employs a Transformer architecture with specialized attention mechanisms to model player interactions. We introduce event projection heads that explicitly connect player movements to basketball events like passes, shots, and steals, training the model to associate physical motion patterns with their tactical purposes. Experiments on NBA tracking data demonstrate significant improvements over position-only baselines: 35% reduction in trajectory prediction error compared to state-of-the-art position-based models and consistent performance gains across key basketball analytics tasks. The resulting pretrained model serves as a powerful foundation for multiple downstream tasks, with pick detection, shot taker identification, assist prediction, shot location classification, and shot type recognition demonstrating substantial improvements over existing methods.

---

## 22. A Nonlinear Low-rank Representation Model with Convolutional Neural Network for Imputing Water Quality Data

**论文链接:** [http://arxiv.org/abs/2512.01465v1](http://arxiv.org/abs/2512.01465v1)

**作者:** Hongnan Si, Tong Li, Yujie Chen, Xin Liao

**发布时间:** 2025-12-01

**备注:** 8 pages, 1 figure

### GPT解析

### 总结

本文提出了一种用于水质数据插补的神经Tucker卷积网络(NTCN)模型，能有效解决长期水质监测中的数据缺失问题，提高水质分析的准确性。

### 背景

水质监测是生态环境保护的核心组成部分，但在长期监测过程中，由于传感器故障或其他不可避免的因素，数据缺失现象普遍存在，给水质分析带来巨大挑战。

### 目的

开发一种有效的水质数据插补模型，以解决水质监测数据缺失问题。

### 方法

提出NTCN模型，包含两个关键组件：a)将不同模态实体编码为嵌入向量，通过外积操作构建Tucker交互张量以捕获复杂的模态特征交互；b)使用3D卷积从交互张量中提取细粒度的时空特征。

### 主要发现

在三个真实世界水质数据集上的实验表明，所提出的NTCN模型在准确性方面优于几种最先进的插补模型。

### 结论

NTCN模型能有效处理水质监测中的数据缺失问题，为水质分析提供更准确的数据支持。

### 翻译

水质监测是生态环境保护的核心组成部分。然而，由于传感器故障或其他不可避免的因素，长期监测中经常存在数据缺失，给水质分析带来巨大挑战。本文提出了一种用于水质数据插补的神经Tucker卷积网络(NTCN)模型，其关键特征包括：a)将不同模态实体编码为各自的嵌入向量，并通过外积操作构建Tucker交互张量以捕获复杂的模态特征交互；b)使用3D卷积从交互张量中提取细粒度的时空特征。在三个真实世界水质数据集上的实验表明，所提出的NTCN模型在准确性方面优于几种最先进的插补模型。


### 论文摘要

Water quality monitoring is a core component of ecological environmental protection. However, due to sensor failure or other inevitable factors, data missing often exists in long-term monitoring, posing great challenges in water quality analysis. This paper proposes a Neural Tucker Convolutional Network (NTCN) model for water quality data imputation, which features the following key components: a) Encode different mode entities into respective embedding vectors, and construct a Tucker interaction tensor by outer product operations to capture the complex mode-wise feature interactions; b) Use 3D convolution to extract fine-grained spatiotemporal features from the interaction tensor. Experiments on three real-world water quality datasets show that the proposed NTCN model outperforms several state-of-the-art imputation models in terms of accuracy.

---

## 23. Masked Symbol Modeling for Demodulation of Oversampled Baseband Communication Signals in Impulsive Noise-Dominated Channels

**论文链接:** [http://arxiv.org/abs/2512.01428v1](http://arxiv.org/abs/2512.01428v1)

**作者:** Oguz Bedir, Nurullah Sevim, Mostafa Ibrahim, Sabit Ekin

**发布时间:** 2025-12-01

**备注:** Accepted to the 39th Conference on Neural Information Processing Systems (NeurIPS 2025) Workshop on AI and ML for Next-Generation Wireless Communications and Networking (AI4NextG), non-archival

### GPT解析

### 总结

本文提出了掩码符号建模(MSM)框架，受Transformer双向编码器表示方法启发，应用于物理层通信系统。该框架通过掩码部分符号样本并利用Transformer预测缺失符号，使模型学习复基带波形的潜在语法。研究展示了MSM在处理受脉冲噪声干扰信号解调中的潜力，为上下文感知的物理层设计开辟新途径。

### 背景

自然语言处理的最新突破表明，通过掩码令牌预测训练的Transformer网络中的注意力机制使模型能够捕获令牌的语义上下文化和内化语言的语法。然而，尽管Transformer在通信系统中的应用是新兴领域，但物理波形中的上下文概念仍未得到充分探索。

### 目的

重新审视由脉冲整形重叠引起的符号间贡献(ISC)，并将其视为嵌入在过采样复基带信号中的确定性上下文信息来源，而非干扰因素。

### 方法

提出掩码符号建模(MSM)框架，这是一种受Transformer双向编码器表示方法启发的物理层框架。随机掩码一部分符号对齐样本，使用Transformer利用周围的'中间'样本预测缺失的符号标识符，使模型学习复基带波形的潜在语法。

### 主要发现

通过将MSM应用于受脉冲噪声干扰的信号解调任务，模型能够利用学习到的上下文推断受损的信号段，展示了MSM的潜力。

### 结论

研究结果表明，可以开发出能够解释而非仅检测通信信号的接收器，为上下文感知的物理层设计开辟新途径。

### 翻译

自然语言处理的最新突破表明，通过掩码令牌预测训练的Transformer网络中的注意力机制使模型能够捕获令牌的语义上下文化和内化语言的语法。尽管Transformer在通信系统中的应用是一个新兴领域，但物理波形中的上下文概念仍未得到充分探索。本文通过重新审视由脉冲整形重叠引起的符号间贡献(ISC)来填补这一空白。我们不将ISC视为干扰因素，而是将其视为嵌入在过采样复基带信号中的确定性上下文信息来源。我们提出了掩码符号建模(MSM)，这是一种受Transformer双向编码器表示方法启发的物理层框架。在MSM中，随机掩码一部分符号对齐样本，并使用Transformer利用周围的'中间'样本预测缺失的符号标识符。通过这一目标，模型学习复基带波形的潜在语法。我们通过将MSM应用于受脉冲噪声干扰的信号解调任务来说明其潜力，在该任务中，模型通过利用学习到的上下文推断受损的信号段。我们的研究结果表明，可以开发出能够解释而非仅检测通信信号的接收器，为上下文感知的物理层设计开辟了新途径。


### 论文摘要

Recent breakthroughs in natural language processing show that attention mechanism in Transformer networks, trained via masked-token prediction, enables models to capture the semantic context of the tokens and internalize the grammar of language. While the application of Transformers to communication systems is a burgeoning field, the notion of context within physical waveforms remains under-explored. This paper addresses that gap by re-examining inter-symbol contribution (ISC) caused by pulse-shaping overlap. Rather than treating ISC as a nuisance, we view it as a deterministic source of contextual information embedded in oversampled complex baseband signals. We propose Masked Symbol Modeling (MSM), a framework for the physical (PHY) layer inspired by Bidirectional Encoder Representations from Transformers methodology. In MSM, a subset of symbol aligned samples is randomly masked, and a Transformer predicts the missing symbol identifiers using the surrounding "in-between" samples. Through this objective, the model learns the latent syntax of complex baseband waveforms. We illustrate MSM's potential by applying it to the task of demodulating signals corrupted by impulsive noise, where the model infers corrupted segments by leveraging the learned context. Our results suggest a path toward receivers that interpret, rather than merely detect communication signals, opening new avenues for context-aware PHY layer design.

---

## 24. Language-Guided Open-World Anomaly Segmentation

**论文链接:** [http://arxiv.org/abs/2512.01427v1](http://arxiv.org/abs/2512.01427v1)

**作者:** Klara Reichard, Nikolas Brasch, Nassir Navab, Federico Tombari

**发布时间:** 2025-12-01

### GPT解析

### 总结

该论文提出了Clipomaly，第一个基于CLIP的开放世界和异常分割方法，用于自动驾驶系统。该方法能够在不使用特定异常训练数据的情况下，零样本地检测和分割已知及未知对象，并为未知对象分配人类可解释的名称，同时实现动态词汇表扩展。

### 背景

开放世界和异常分割方法旨在使自动驾驶系统能够检测和分割现实场景中已知和未知的对象。然而，现有方法不为未知区域分配有语义意义的标签，且难以区分和学习未知类的表示。开放词汇分割方法虽然可以推广到新类别，但需要固定的推理词汇表，无法直接应用于未知类别不受约束的异常分割任务。

### 目的

开发一种能够同时处理开放世界分割和异常分割的方法，能够检测和分割已知及未知对象，并为未知对象分配有意义的语义标签，同时保持灵活性和可解释性。

### 方法

作者提出了Clipomaly，一种基于CLIP的开放世界和异常分割方法。该方法采用零样本学习，不需要特定的异常训练数据，而是利用CLIP的共享图像-文本嵌入空间来分割未知对象并为它们分配人类可解释的名称。与开放词汇方法不同，Clipomaly能够在推理时动态扩展词汇表，无需重新训练，从而能够稳健地检测和命名超出常见类别定义的异常。

### 主要发现

Clipomaly在已建立的异常分割基准测试上取得了最先进的性能。该方法不仅能够有效检测和分割异常对象，还能为这些对象提供人类可理解的名称，同时具有实际部署所需的灵活性和可解释性。

### 结论

Clipomaly代表了自动驾驶领域中开放世界和异常分割方法的进步，它解决了现有方法在处理未知对象时的局限性，提供了更实用、更灵活的解决方案，为自动驾驶系统在实际复杂环境中的应用奠定了基础。

### 翻译

开放世界和异常分割方法旨在使自动驾驶系统能够检测和分割现实场景中已知和未知的对象。然而，现有方法不为未知区域分配有语义意义的标签，且难以区分和学习未知类的表示。虽然开放词汇分割方法在推广到新类别方面显示出潜力，但它们需要固定的推理词汇表，因此不能直接应用于未知类别不受约束的异常分割任务。我们提出了Clipomaly，这是第一个基于CLIP的用于自动驾驶的开放世界和异常分割方法。我们的零样本方法不需要特定的异常训练数据，而是利用CLIP的共享图像-文本嵌入空间来分割未知对象并为它们分配人类可解释的名称。与开放词汇方法不同，我们的模型在推理时动态扩展其词汇表而无需重新训练，从而能够稳健地检测和命名超出常见类别定义(如Cityscapes中的类别)的异常。Clipomaly在已建立的异常分割基准测试上取得了最先进的性能，同时提供了实际部署所必需的可解释性和灵活性。


### 论文摘要

Open-world and anomaly segmentation methods seek to enable autonomous driving systems to detect and segment both known and unknown objects in real-world scenes. However, existing methods do not assign semantically meaningful labels to unknown regions, and distinguishing and learning representations for unknown classes remains difficult. While open-vocabulary segmentation methods show promise in generalizing to novel classes, they require a fixed inference vocabulary and thus cannot be directly applied to anomaly segmentation where unknown classes are unconstrained. We propose Clipomaly, the first CLIP-based open-world and anomaly segmentation method for autonomous driving. Our zero-shot approach requires no anomaly-specific training data and leverages CLIP's shared image-text embedding space to both segment unknown objects and assign human-interpretable names to them. Unlike open-vocabulary methods, our model dynamically extends its vocabulary at inference time without retraining, enabling robust detection and naming of anomalies beyond common class definitions such as those in Cityscapes. Clipomaly achieves state-of-the-art performance on established anomaly segmentation benchmarks while providing interpretability and flexibility essential for practical deployment.

---

## 25. A Self-explainable Model of Long Time Series by Extracting Informative Structured Causal Patterns

**论文链接:** [http://arxiv.org/abs/2512.01412v1](http://arxiv.org/abs/2512.01412v1)

**作者:** Ziqian Wang, Yuxiao Cheng, Jinli Suo

**发布时间:** 2025-12-01

**备注:** Approximately 30 pages, 8 figures, and 5 tables. Preprint version. Includes theoretical analysis, model architecture, interpretability evaluation, and extensive benchmark experiments

### GPT解析

### 总结

EXCAP是一种统一框架，解决了长时间序列神经网络解释性的关键问题，能够捕获时间结构并提供连贯的因果解释。

### 背景

现有可解释AI方法仅生成点状重要性分数，无法捕获时间序列中的趋势、周期和制度变化等时间结构，削弱了人们对长期预测模型的可解释性和信任度。

### 目的

解决长时间序列神经网络解释性的局限，满足可解释时间序列建模的四个关键需求：时间连续性、以模式为中心的解释、因果解纠缠和对模型推理过程的忠实性。

### 方法

提出EXCAP框架，结合基于注意力的分段器提取连贯时间模式、由预训练因果图引导的因果结构解码器、以及强制表示稳定性的潜在聚合机制。

### 主要发现

EXCAP提供平滑且随时间稳定的解释，对因果掩码扰动具有鲁棒性，在保持强预测准确性的同时生成连贯且基于因果的解释。

### 结论

EXCAP为长时间序列的可解释建模提供了一种有原则且可扩展的方法，适用于医疗保健和金融等高风险领域。

### 翻译

可解释性对于建模长时间序列的神经网络至关重要，但大多数现有的可解释AI方法仅产生点状重要性分数，无法捕获趋势、周期和制度变化等时间结构。这一限制削弱了人们对长期预测模型的可解释性和信任。为解决这些问题，我们确定了可解释时间序列建模的四个关键需求：时间连续性、以模式为中心的解释、因果解纠缠以及对模型推理过程的忠实性。我们提出了EXCAP，这是一个满足所有四个需求的统一框架。EXCAP结合了基于注意力的分段器，用于提取连贯的时间模式；由预训练因果图引导的因果结构解码器；以及强制表示稳定性的潜在聚合机制。我们的理论分析表明，EXCAP提供随时间平滑且稳定的解释，并且对因果掩码的扰动具有鲁棒性。在分类和预测基准上的广泛实验表明，EXCAP在保持强预测准确性的同时，生成连贯且基于因果的解释。这些结果表明，EXCAP为长时间序列的可解释建模提供了一种有原则且可扩展的方法，与医疗保健和金融等高风险领域相关。


### 论文摘要

Explainability is essential for neural networks that model long time series, yet most existing explainable AI methods only produce point-wise importance scores and fail to capture temporal structures such as trends, cycles, and regime changes. This limitation weakens human interpretability and trust in long-horizon models. To address these issues, we identify four key requirements for interpretable time-series modeling: temporal continuity, pattern-centric explanation, causal disentanglement, and faithfulness to the model's inference process. We propose EXCAP, a unified framework that satisfies all four requirements. EXCAP combines an attention-based segmenter that extracts coherent temporal patterns, a causally structured decoder guided by a pre-trained causal graph, and a latent aggregation mechanism that enforces representation stability. Our theoretical analysis shows that EXCAP provides smooth and stable explanations over time and is robust to perturbations in causal masks. Extensive experiments on classification and forecasting benchmarks demonstrate that EXCAP achieves strong predictive accuracy while generating coherent and causally grounded explanations. These results show that EXCAP offers a principled and scalable approach to interpretable modeling of long time series with relevance to high-stakes domains such as healthcare and finance.

---

## 26. DyFuLM: An Advanced Multimodal Framework for Sentiment Analysis

**论文链接:** [http://arxiv.org/abs/2512.01410v1](http://arxiv.org/abs/2512.01410v1)

**作者:** Ruohan Zhou, Jiachen Yuan, Churui Yang, Wenzheng Huang, Guoyan Zhang, Shiyao Wei, Jiazhen Hu, Ning Xin, Md Maruf Hasan

**发布时间:** 2025-12-01

**备注:** 8 pages, 6 figures, preprint. Under review for a suitable AI conference

### GPT解析

### 总结

本文提出了一种动态融合学习模型(DyFuLM)，用于解决复杂文本表达中的情感理解挑战。该模型通过两个关键模块实现多模态特征融合，在多任务情感数据集上取得了优异的性能。

### 背景

理解复杂文本表达中的情感仍然是情感计算中的一个基本挑战。

### 目的

提出一个动态融合学习模型(DyFuLM)，捕获分层语义表示和细粒度情感细微差别。

### 方法

DyFuLM是一个多模态框架，包含两个关键模块：分层动态融合模块（自适应集成多级特征）和门控特征聚合模块（调节跨层信息流以实现平衡的表示学习）。

### 主要发现

在多任务情感数据集上，DyFuLM达到了82.64%的粗粒度和68.48%的细粒度准确率，产生最低的回归误差和最高的决定系数。消融研究验证了每个模块的有效性，移除不同模块会导致不同程度的性能下降。

### 结论

DyFuLM通过有效的分层特征融合增强了情感表示和整体性能，每个模块都对特征交互和任务平衡有显著贡献。

### 翻译

在复杂文本表达中理解情感仍然是情感计算中的一个基本挑战。为此，我们提出了动态融合学习模型(DyFuLM)，这是一个多模态框架，旨在捕获分层语义表示和细粒度情感细微差别。DyFuLM引入了两个关键模块：分层动态融合模块，自适应地集成多级特征；以及门控特征聚合模块，调节跨层信息流以实现平衡的表示学习。在多任务情感数据集上的综合实验表明，DyFuLM达到了82.64%的粗粒度和68.48%的细粒度准确率，产生了最低的回归误差和最高的决定系数。此外，消融研究验证了DyFuLM中每个模块的有效性。当所有模块都被移除时，粗粒度和细粒度任务的准确率分别下降了0.91%和0.68%。仅保留门控融合模块导致粗粒度和细粒度任务分别下降了0.75%和0.55%，而移除动态损失机制则导致粗粒度和细粒度情感分类分别下降了0.78%和0.26%。这些结果表明每个模块都对特征交互和任务平衡有显著贡献。总体而言，实验结果进一步验证了DyFuLM通过有效的分层特征融合增强了情感表示和整体性能。


### 论文摘要

Understanding sentiment in complex textual expressions remains a fundamental challenge in affective computing. To address this, we propose a Dynamic Fusion Learning Model (DyFuLM), a multimodal framework designed to capture both hierarchical semantic representations and fine-grained emotional nuances. DyFuLM introduces two key moodules: a Hierarchical Dynamic Fusion module that adaptively integrates multi-level features, and a Gated Feature Aggregation module that regulates cross-layer information ffow to achieve balanced representation learning. Comprehensive experiments on multi-task sentiment datasets demonstrate that DyFuLM achieves 82.64% coarse-grained and 68.48% fine-grained accuracy, yielding the lowest regression errors (MAE = 0.0674, MSE = 0.0082) and the highest R^2 coefficient of determination (R^2= 0.6903). Furthermore, the ablation study validates the effectiveness of each module in DyFuLM. When all modules are removed, the accuracy drops by 0.91% for coarse-grained and 0.68% for fine-grained tasks. Keeping only the gated fusion module causes decreases of 0.75% and 0.55%, while removing the dynamic loss mechanism results in drops of 0.78% and 0.26% for coarse-grained and fine-grained sentiment classification, respectively. These results demonstrate that each module contributes significantly to feature interaction and task balance. Overall, the experimental findings further validate that DyFuLM enhances sentiment representation and overall performance through effective hierarchical feature fusion.

---

## 27. Fantastic Features and Where to Find Them: A Probing Method to combine Features from Multiple Foundation Models

**论文链接:** [http://arxiv.org/abs/2512.01405v1](http://arxiv.org/abs/2512.01405v1)

**作者:** Benjamin Ramtoula, Pierre-Yves Lajoie, Paul Newman, Daniele De Martini

**发布时间:** 2025-12-01

**备注:** Published at NeurIPS 2025

### GPT解析

### 总结

论文提出ComBo方法，一种简单可扩展的基于探测的适配器，用于有效整合多个基础模型的特征，提升下游任务性能，同时评估各模型的任务相关性。

### 背景

基础模型(FMs)根据不同目标和数据训练，学习多样化表示，使某些模型在特定下游任务上更有效。现有适应策略专注于单模型，未利用模型间互补优势；探测方法虽能从冻结模型提取信息，但在大特征集上扩展性差且依赖数据集特定超参数调整。

### 目的

开发一种简单可扩展的基于探测的适配器，有效整合多模型和层特征，解决现有方法局限性，并允许评估各骨干模型的任务相关性。

### 方法

ComBo将一个或多个FM层的激活压缩为紧凑标记级表示，用轻量级transformer处理进行任务预测。不需要数据集特定调整或骨干模型反向传播。引入机制利用联合多骨干探测评估各模型任务相关性，实现模型比较和选择性适应。

### 主要发现

在VTAB-1k基准19个任务上，ComBo优于之前的探测方法，匹配或超过更昂贵的替代方案(如基于蒸馏的模型合并)，并支持对已微调模型的高效探测。

### 结论

ComBo提供了实用通用框架，用于组合来自多个FM的多样化表示，有效提升下游任务性能。

### 翻译

基础模型(FMs)通过不同目标和数据训练，学习多样化表示，使某些模型比其他模型在特定下游任务上更有效。现有适应策略如参数高效微调专注于单模型，未利用模型间互补优势。探测方法提供有前景替代方案，可从冻结模型提取信息，但当前技术在大特征集上扩展性差且依赖数据集特定超参数调整。我们提出Combined backBones (ComBo)，一种简单可扩展的基于探测适配器，能有效整合多模型和层特征。ComBo将一个或多个FM层激活压缩为紧凑标记级表示，用轻量级transformer处理进行任务特定预测。重要的是，ComBo不需要数据集特定调整或骨干模型反向传播。然而，并非所有模型对所有任务都同等相关。为解决此问题，我们引入机制利用ComBo的联合多骨干探测高效评估各骨干模型任务相关性，实现实用模型比较和通过选择性适应提高性能。在VTAB-1k基准19个任务上，ComBo优于之前探测方法，匹配或超过更昂贵替代方案如基于蒸馏的模型合并，并支持对已微调模型的高效探测。结果表明ComBo为组合来自多个FM的多样化表示提供了实用通用框架。


### 论文摘要

Foundation models (FMs) trained with different objectives and data learn diverse representations, making some more effective than others for specific downstream tasks. Existing adaptation strategies, such as parameter-efficient fine-tuning, focus on individual models and do not exploit the complementary strengths across models. Probing methods offer a promising alternative by extracting information from frozen models, but current techniques do not scale well with large feature sets and often rely on dataset-specific hyperparameter tuning. We propose Combined backBones (ComBo), a simple and scalable probing-based adapter that effectively integrates features from multiple models and layers. ComBo compresses activations from layers of one or more FMs into compact token-wise representations and processes them with a lightweight transformer for task-specific prediction. Crucially, ComBo does not require dataset-specific tuning or backpropagation through the backbone models. However, not all models are equally relevant for all tasks. To address this, we introduce a mechanism that leverages ComBo's joint multi-backbone probing to efficiently evaluate each backbone's task-relevance, enabling both practical model comparison and improved performance through selective adaptation. On the 19 tasks of the VTAB-1k benchmark, ComBo outperforms previous probing methods, matches or surpasses more expensive alternatives, such as distillation-based model merging, and enables efficient probing of tuned models. Our results demonstrate that ComBo offers a practical and general-purpose framework for combining diverse representations from multiple FMs.

---

## 28. InternVideo-Next: Towards General Video Foundation Models without Video-Text Supervision

**论文链接:** [http://arxiv.org/abs/2512.01342v1](http://arxiv.org/abs/2512.01342v1)

**作者:** Chenting Wang, Yuhan Zhu, Yicheng Xu, Jiange Yang, Ziang Yan, Yali Wang, Yi Wang, Limin Wang

**发布时间:** 2025-12-01

### GPT解析

### 总结

该研究提出了一种名为InternVideo-Next的新型视频预训练框架，通过分离传统编码器-解码器设计为Encoder-Predictor-Decoder(EPD)架构，并采用两阶段预训练方案，成功弥合了像素级重建与语义抽象之间的差距，减轻了捷径学习问题，在多个基准测试中取得了最先进的结果。

### 背景

大规模视频-文本预训练虽表现良好，但依赖于噪声大、语义覆盖有限的人工合成字幕，且忽略了隐式世界知识；而掩码视频建模虽直接利用时空结构，但在通用任务上落后于文本监督方法。

### 目的

解决视频建模中的架构问题，弥合像素级重建与语义抽象之间的差距，减少捷径学习，构建一个语义一致且保留细节的潜在空间。

### 方法

提出Encoder-Predictor-Decoder(EPD)框架，将预测器作为潜在世界模型；设计两阶段预训练方案：第一阶段使用条件扩散解码器注入图像级语义先验，增强语义和收敛性；第二阶段在潜在空间中预测冻结的第一阶段目标，进一步学习世界知识。

### 主要发现

像素级重建难以收敛且与语义要求冲突；潜在预测容易导致捷径学习；传统线性解码器使预测输出在像素空间可分离，与语义抽象产生冲突。

### 结论

在公共未标记视频上训练的InternVideo-Next在多个基准测试中取得最先进结果，为通用视频表示学习提供了可扩展路径。

### 翻译

大规模视频-文本预训练取得了强大的性能，但依赖于噪声大、语义覆盖有限的人工合成字幕，经常忽略隐式世界知识，如物体运动、3D几何和物理线索。相比之下，掩码视频建模(MVM)直接利用时空结构，但在通用任务上落后于文本监督方法。我们发现这种差距源于被忽视的架构问题：像素级重建难以收敛，且其低级要求常常与语义冲突，而潜在预测往往鼓励捷径学习。为解决这些问题，我们将传统的编码器-解码器设计分离为Encoder-Predictor-Decoder(EPD)框架，其中预测器充当潜在世界模型，并提出了InternVideo-Next，一种为此世界模型构建语义一致且保留细节的潜在空间的两阶段预训练方案。首先，像素MVM中的传统线性解码器强制预测器输出的潜变量可线性投影到像素空间，因此在像素空间中可分离，这与语义抽象产生冲突。我们的第一阶段提出了条件扩散解码器，并注入可靠的图像级语义先验以增强语义和收敛性，从而弥合像素级保真度与高级语义抽象之间的差距。第二阶段通过在此空间中预测冻结的第一阶段目标，进一步学习世界知识，减轻了捷径学习。在公共未标记视频上训练的InternVideo-Next在多个基准测试中取得了最先进的结果，并为通用视频表示学习提供了可扩展的路径。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决视频基础模型训练中过度依赖视频-文本监督的问题。现有方法依赖有噪声、语义覆盖有限的合成字幕，忽略了视频中的隐式世界知识（如物体运动、3D几何和物理线索）。这个问题很重要，因为视频是理解物理世界的重要窗口，包含时空动态、因果关系等关键信息，对于构建真正的视频理解模型、推进具身AI和下一代多模态大语言模型至关重要。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': "作者通过分析现有方法局限性发现：像素级重建难以收敛且与语义冲突，而潜在预测易导致'捷径学习'。他们借鉴了掩码视频建模（如VideoMAE）、潜在空间预测（如V-JEPA）和扩散模型等现有工作，但创新性地将传统编码器-解码器设计解耦为编码器-预测器-解码器（EPD）框架，使预测器成为'潜在世界模型'，并设计两阶段训练方案解决上述问题。", '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是通过EPD框架解耦传统架构，构建一个语义一致且保留细节的潜在空间。整体流程分为两阶段：第一阶段使用语义引导的像素重建，通过语义对齐损失、语义感知掩码和条件扩散解码器构建高质量潜在空间；第二阶段在这个空间中进行语义一致的潜在预测，使用多块掩码策略和冻结教师目标学习时空动态和因果关系，避免捷径学习。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1) EPD框架解耦设计，使预测器成为潜在世界模型；2) 两阶段训练方案，先构建语义一致空间再学习世界知识；3) 条件扩散解码器替代传统线性解码器；4) 语义感知掩码策略；5) 文本解码器初始化。相比之前工作，它不依赖有噪声的视频字幕，能捕捉更丰富的隐式世界知识，同时解决了像素级重建与语义冲突、潜在预测中的捷径学习问题，在保持高语义的同时保留了细节信息。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'InternVideo-Next通过创新的编码器-预测器-解码器框架和两阶段训练方案，首次在没有视频-文本监督的情况下实现了超越视频-文本预训练方法的性能，为构建通用的视频基础模型提供了新路径。'}


### 论文摘要

Large-scale video-text pretraining achieves strong performance but depends on noisy, synthetic captions with limited semantic coverage, often overlooking implicit world knowledge such as object motion, 3D geometry, and physical cues. In contrast, masked video modeling (MVM) directly exploits spatiotemporal structures but trails text-supervised methods on general tasks. We find this gap arises from overlooked architectural issues: pixel-level reconstruction struggles with convergence and its low-level requirement often conflicts with semantics, while latent prediction often encourages shortcut learning. To address these, we disentangle the traditional encoder-decoder design into an Encoder-Predictor-Decoder (EPD) framework, where the predictor acts as a latent world model, and propose InternVideo-Next, a two-stage pretraining scheme that builds a semantically consistent yet detail-preserving latent space for this world model. First, conventional linear decoder in pixel MVM enforces the predictor output latent to be linearly projected to, thus separable in pixel space, causing the conflict with semantic abstraction. Our Stage 1 proposes a conditional diffusion decoder and injects reliable image-level semantic priors to enhance semantics and convergence, thus bridging pixel-level fidelity with high-level semantic abstraction. Stage 2 further learns world knowledge by predicting frozen Stage 1 targets within this space, mitigating shortcut learning. Trained on public, unlabeled videos, InternVideo-Next achieves state-of-the-art results across benchmarks and provides a scalable path toward general video representation learning.

---

## 29. Panda: Self-distillation of Reusable Sensor-level Representations for High Energy Physics

**论文链接:** [http://arxiv.org/abs/2512.01324v1](http://arxiv.org/abs/2512.01324v1)

**作者:** Samuel Young, Kazuhiro Terao

**发布时间:** 2025-12-01

**备注:** 23 pages, 15 figures, preprint. Project page at https://youngsm.com/panda/

### GPT解析

### 总结

本文介绍了一种名为Panda的新模型，该模型可以直接从未标记的原始液氩时间投影室(LArTPC)数据中学习可重用的传感器级表示，显著提高了粒子重建效率和准确性。

### 背景

液氩时间投影室(LArTPCs)能够提供密集、高保真的粒子相互作用3D测量，支持当前和未来的中微子和稀有事件实验。然而，现有的物理重建方法依赖于复杂的探测器特定流程，使用大量手工设计的模式识别算法或特定任务的神经网络级联，需要大量标记的模拟数据和耗时的校准过程。

### 目的

开发一种能够从未标记的原始LArTPC数据中直接学习可重用表示的模型，以减少对大量标记数据和复杂校准过程的依赖，提高重建效率和准确性。

### 方法

引入Panda模型，该模型结合了分层稀疏3D编码器和多视图、基于原型的自蒸馏目标，直接从未标记的原始LArTPC数据中学习可重用的传感器级表示。

### 主要发现

在模拟数据集上，Panda显著提高了标签效率和重建质量，使用比之前最先进模型少1000倍的标签就能获得更好的性能；仅使用Panda冻结输出的单个小型集合预测头(无需物理先验训练)，就能实现与最先进重建工具相当的粒子识别能力；完整微调可进一步提高所有任务的性能。

### 结论

Panda模型为LArTPC数据分析提供了一种更高效、更自动化的方法，减少了对大量标记数据和复杂校准过程的依赖，为粒子物理实验提供了新的重建工具。

### 翻译

液氩时间投影室(LArTPCs)提供密集、高保真的粒子相互作用3D测量，支撑当前和未来的中微子及稀有事件实验。物理重建通常依赖于复杂的探测器特定流程，使用数十种手工设计的模式识别算法或特定任务神经网络的级联，这些需要大量标记的模拟数据和仔细、耗时的校准过程。我们引入Panda，一种直接从未标记的原始LArTPC数据中学习可重用传感器级表示的模型。Panda将分层稀疏3D编码器与多视图、基于原型的自蒸馏目标相结合。在模拟数据集上，Panda显著提高了标签效率和重建质量，使用比之前最先进的语义分割模型少1000倍的标签就能超越其性能。我们还表明，仅使用Panda冻结输出的单个大小为主干1/20的集合预测头，无需物理先验训练，就能实现与最先进重建工具相当的粒子识别能力。完整微调可进一步提高所有任务的性能。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决高能物理实验中液氩时间投影室(LArTPC)数据处理的问题。当前物理重建依赖于复杂的、特定于探测器的管道，使用大量手工设计的算法或神经网络级联，需要大量标记的模拟数据和耗时校准。这个问题的重要性在于：1) 数据效率低下，模拟数据生成和校准极其耗时；2) 系统复杂度高，难以维护和调整；3) 泛化能力差，难以适应不同探测器；4) 级联系统存在信息瓶颈，早期错误会传播到后续阶段。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者从LArTPC数据特性出发，观察到这些数据提供密集、高保真的3D测量，粒子轨迹有独特模式，与现代自监督学习目标匹配。设计上借鉴了多个领域：1) 自蒸馏方法如DINO、iBOT和Sonata；2) 点云处理如Point Transformer V3；3) 掩码建模如PoLAr-MAE；4) 分割任务如Mask2Former。针对物理数据特性进行了调整：不使用全局[cls]标记（因LArTPC图像包含因果不相关的粒子相互作用）；设计特定视图增强策略保留物理几何；为粒子识别设计特定损失函数和评估指标。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是通过自蒸馏从原始未标记LArTPC数据中学习可重用的传感器级表示，捕捉粒子轨迹物理特征而无需特定于探测器的算法或大量标记数据。实现流程：1) 数据表示：将LArTPC事件表示为3D点集，3mm网格分辨率；2) 编码器：五阶段点原生分层编码器，结合稀疏3D卷积和自注意力；3) 自蒸馏预训练：教师-学生架构，使用全局/局部/掩码视图，原型分布作为目标；4) 下游任务：语义分割用线性分类头或多尺度解码器，全景分割用集合预测头部识别粒子；5) 训练评估：在120万事件数据集上预训练，多任务微调评估数据效率和参数效率。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点：1) 传感器级自蒸馏框架，专门针对高能物理数据；2) 点原生分层编码器，直接处理3D电荷云；3) 基于原型的自蒸馏目标，学习物理变换不变表示；4) 统一多任务框架，单一主干支持多个下游任务。相比之前工作不同：1) 与PoLAr-MAE相比，Panda使用点原生方法直接操作3D电荷云，能更好捕捉细微特征；2) 与传统重建管道(如Pandora)相比，减少参数需求，避免级联错误传播；3) 与其他自监督方法相比，针对LArTPC数据特性优化；4) 与监督方法相比，显著提高数据效率，使用少1000倍标签获得更好性能。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'Panda通过自蒸馏从原始未标记LArTPC数据中学习可重用的传感器级表示，实现了高能物理中粒子重建的高效、准确且可泛化的解决方案，显著提高数据效率并减少对手工设计算法和大量模拟数据的依赖。'}


### 论文摘要

Liquid argon time projection chambers (LArTPCs) provide dense, high-fidelity 3D measurements of particle interactions and underpin current and future neutrino and rare-event experiments. Physics reconstruction typically relies on complex detector-specific pipelines that use tens of hand-engineered pattern recognition algorithms or cascades of task-specific neural networks that require extensive, labeled simulation that requires a careful, time-consuming calibration process. We introduce \textbf{Panda}, a model that learns reusable sensor-level representations directly from raw unlabeled LArTPC data. Panda couples a hierarchical sparse 3D encoder with a multi-view, prototype-based self-distillation objective. On a simulated dataset, Panda substantially improves label efficiency and reconstruction quality, beating the previous state-of-the-art semantic segmentation model with 1,000$\times$ fewer labels. We also show that a single set-prediction head 1/20th the size of the backbone with no physical priors trained on frozen outputs from Panda can result in particle identification that is comparable with state-of-the-art (SOTA) reconstruction tools. Full fine-tuning further improves performance across all tasks.

---

## 30. Experimental Methods, Health Indicators, and Diagnostic Strategies for Retired Lithium-ion Batteries: A Comprehensive Review

**论文链接:** [http://arxiv.org/abs/2512.01294v1](http://arxiv.org/abs/2512.01294v1)

**作者:** Song Zhang, Ruohan Guo, Xiaohua Ge, Perter Mahon, Weixiang Shen

**发布时间:** 2025-12-01

**备注:** Review article; 46 pages, 3 figures, 2 tables

### GPT解析

### 总结

这篇综述探讨了退役锂离子电池健康评估的挑战和最新进展，旨在解决传统方法在退役阶段分类应用中的局限性，并提出基于物理健康指标、实验测试方法、数据增强技术和多种学习范式的解决方案。

### 背景

退役锂离子电池的健康评估对于安全和经济可行的二次利用至关重要，但由于测量稀疏、历史记录不完整、化学成分异质以及电池健康标签有限或有噪声，这一评估仍然困难。

### 目的

综合最近的研究进展，解决退役锂离子电池健康评估中的限制，并提出可靠、可扩展且可部署的健康预测工具。

### 方法

通过物理健康指标、实验测试方法、数据生成和增强技术，以及涵盖监督、半监督、弱监督和无监督范式的多种基于学习的建模路线。

### 主要发现

最小测试特征、合成数据、领域不变表示和不确定性感知预测能够在有限或近似标签以及混合化学成分和操作历史条件下实现鲁棒推理；比较评估揭示了准确性、可解释性、可扩展性和计算负担之间的权衡。

### 结论

未来需要朝着物理约束生成模型、跨化学成分泛化、校准不确定性估计和标准化基准进展，以构建适应退役电池应用实际情况的可靠、可扩展且可部署的健康预测工具。

### 翻译

退役锂离子电池的可靠健康评估对于安全和经济可行的二次利用部署至关重要，但由于测量稀疏、历史记录不完整、化学成分异质以及电池健康标签有限或有噪声，这一评估仍然困难。传统的实验室诊断，如完整的充放电循环、脉冲测试、电化学阻抗谱测量和热表征，虽然能提供准确的退化信息，但在退役阶段分类应用时过于耗时、设备密集或条件敏感，无法大规模应用，导致现实世界数据集零散且不一致。这篇综述综合了最近的研究进展，通过物理健康指标、实验测试方法、数据生成和增强技术，以及涵盖监督、半监督、弱监督和无监督范式的多种基于学习的建模路线来解决这些限制。我们强调了如何通过最小测试特征、合成数据、领域不变表示和不确定性感知预测，在有限或近似标签以及混合化学成分和操作历史条件下实现鲁棒推理。比较评估进一步揭示了准确性、可解释性、可扩展性和计算负担之间的权衡。展望未来，朝着物理约束生成模型、跨化学成分泛化、校准不确定性估计和标准化基准进展，对于构建可靠、可扩展且可部署的健康预测工具至关重要，这些工具需要适应退役电池应用的实际情况。


### 论文摘要

Reliable health assessment of retired lithium-ion batteries is essential for safe and economically viable second-life deployment, yet remains difficult due to sparse measurements, incomplete historical records, heterogeneous chemistries, and limited or noisy battery health labels. Conventional laboratory diagnostics, such as full charge-discharge cycling, pulse tests, Electrochemical Impedance Spectroscopy (EIS) measurements, and thermal characterization, provide accurate degradation information but are too time-consuming, equipment-intensive, or condition-sensitive to be applied at scale during retirement-stage sorting, leaving real-world datasets fragmented and inconsistent. This review synthesizes recent advances that address these constraints through physical health indicators, experiment testing methods, data-generation and augmentation techniques, and a spectrum of learning-based modeling routes spanning supervised, semi-supervised, weakly supervised, and unsupervised paradigms. We highlight how minimal-test features, synthetic data, domain-invariant representations, and uncertainty-aware prediction enable robust inference under limited or approximate labels and across mixed chemistries and operating histories. A comparative evaluation further reveals trade-offs in accuracy, interpretability, scalability, and computational burden. Looking forward, progress toward physically constrained generative models, cross-chemistry generalization, calibrated uncertainty estimation, and standardized benchmarks will be crucial for building reliable, scalable, and deployment-ready health prediction tools tailored to the realities of retired-battery applications.

---

## 31. Register Any Point: Scaling 3D Point Cloud Registration by Flow Matching

**论文链接:** [http://arxiv.org/abs/2512.01850v1](http://arxiv.org/abs/2512.01850v1)

**作者:** Yue Pan, Tao Sun, Liyuan Zhu, Lucas Nunes, Iro Armeni, Jens Behley, Cyrill Stachniss

**发布时间:** 2025-12-01

**备注:** 22 pages

### GPT解析

### 总结

该工作提出了一种新颖的点云配准方法，通过条件生成直接生成配准后的点云，而非传统的对应点匹配方法。该方法在多种场景下表现优异，支持多种下游应用。

### 背景

点云配准是将多个未配准的点云对齐到同一坐标系中的过程，是3D重建和机器人定位的核心步骤。

### 目的

通过条件生成的方式解决点云配准问题，直接生成配准后的点云，避免传统方法中的对应点匹配和变换估计步骤。

### 方法

将配准问题表述为条件生成问题：学习到的连续逐点速度场将带噪点传输到配准场景，从中恢复每个视图的姿态。使用轻量级局部特征提取器和测试时刚性约束。

### 主要发现

在点对和多视图配准基准测试上取得最先进结果，特别是在低重叠情况下；能够跨尺度和传感器模态泛化；支持重定位、多机器人SLAM和多会话地图合并等下游任务。

### 结论

通过将配准视为条件生成问题，提出的方法相比传统方法具有优势，特别是在低重叠场景下表现优异，且具有良好的泛化能力。

### 翻译

点云配准将多个未配准的点云对齐到同一坐标系中，是3D重建和机器人定位的核心步骤。在这项工作中，我们将配准视为条件生成：学习到的连续逐点速度场将带噪点传输到配准的场景，从中恢复每个视图的姿态。与之前通过对应点匹配来估计点云对之间的变换，然后优化这些成对变换以实现多视图配准的方法不同，我们的模型直接生成配准后的点云。通过轻量级的局部特征提取器和测试时刚性约束，我们的方法在点对和多视图配准基准测试上取得了最先进的结果，特别是在低重叠情况下，并且能够跨尺度和传感器模态泛化。它还支持重定位、多机器人SLAM和多会话地图合并等下游任务。源代码可在 https://github.com/PRBonn/RAP 获取。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决多视图3D点云配准问题，即将多个未配准的点云对齐到共同坐标系中。这个问题在现实中非常重要，因为它是3D重建和机器人定位的核心步骤，广泛应用于自动驾驶、机器人导航和增强现实等领域。现实世界中的点云数据通常是稀疏、嘈杂且密度不均匀的，不同传感器具有不同模态和校准参数，点云间重叠可能很小，传统方法难以处理这些挑战。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者将配准问题重新定义为条件生成问题，借鉴了RPF工作中的流动匹配技术，使用一个连续的速度场来传输点云位置。与RPF不同，作者通过稀疏关键点和局部特征表示扩展到大型场景，并引入刚性强制采样和选择策略来确保每个视图的刚性运动。作者还收集了17个多样化数据集的超过10万个样本，使模型能够泛化到不同规模和传感器模态。方法采用了Diffusion Transformer架构，带有交替注意块来同时捕获局部结构和全局关系。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是将多视图点云配准视为条件生成问题，使用流动匹配直接从无序点云生成配准后的点云，绕过传统的点对应匹配和姿态图优化。整体流程包括：1)关键点采样与局部特征提取；2)输入和目标的规范化处理；3)使用条件流动模型学习速度场；4)刚性强制推理确保刚性约束；5)将结果提升回原始坐标系。这种方法直接生成配准结果，避免了传统方法的二次复杂度问题。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)单阶段配准模型，直接生成配准点云；2)刚性强制采样和选择策略，提高配准准确性；3)大规模多样化训练数据，增强泛化能力；4)关键点表示方法，使模型能扩展到大型场景。相比RPF工作，本文不依赖专门的重叠预测网络，并在流动过程中强制刚性约束。相比传统两阶段方法，本文绕过点对应匹配和姿态图优化，直接生成结果，在低重叠场景下表现更好。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '本文提出了一种基于流动匹配的单阶段生成式点云配准方法，通过强制刚性约束和大规模多样化训练，实现了在低重叠场景下超越现有方法的性能，并支持从物体级到地图级的多种应用场景。'}


### 论文摘要

Point cloud registration aligns multiple unposed point clouds into a common frame, and is a core step for 3D reconstruction and robot localization. In this work, we cast registration as conditional generation: a learned continuous, point-wise velocity field transports noisy points to a registered scene, from which the pose of each view is recovered. Unlike previous methods that conduct correspondence matching to estimate the transformation between a pair of point clouds and then optimize the pairwise transformations to realize multi-view registration, our model directly generates the registered point cloud. With a lightweight local feature extractor and test-time rigidity enforcement, our approach achieves state-of-the-art results on pairwise and multi-view registration benchmarks, particularly with low overlap, and generalizes across scales and sensor modalities. It further supports downstream tasks including relocalization, multi-robot SLAM, and multi-session map merging. Source code available at: https://github.com/PRBonn/RAP.

---

## 32. LAHNet: Local Attentive Hashing Network for Point Cloud Registration

**论文链接:** [http://arxiv.org/abs/2512.00927v1](http://arxiv.org/abs/2512.00927v1)

**作者:** Wentao Qu, Xiaoshui Huang, Liang Xiao

**发布时间:** 2025-11-30

### GPT解析

### 总结

本文提出了一种名为LAHNet的局部注意力哈希网络，用于点云配准，通过引入局部注意力机制和创新的窗口策略来增强特征的区分度。

### 背景

大多数现有的基于学习的点云描述符专注于感知点云的局部信息来生成特征，但合理的更广泛的感受野对于增强特征区分度至关重要。

### 目的

设计一个能够捕获更广泛感受野的点云描述符，以提高特征区分度和配准性能。

### 方法

设计了Group Transformer来捕捉点之间的长程上下文；采用线性邻域搜索策略和局部敏感哈希将点云划分为不重叠窗口；使用跨窗口策略扩展特征感受野；提出Interaction Transformer增强点云对重叠区域的特征交互；通过计算重叠矩阵匹配点云对之间的重叠区域。

### 主要发现

LAHNet能够学习鲁棒且具有区分度的特征，在真实世界室内和室外基准测试中取得了显著的配准结果。

### 结论

通过引入局部注意力机制和创新的窗口策略，LAHNet有效增强了点云特征的区分度，提高了配准性能。

### 翻译

大多数现有的基于学习的点云描述符用于点云配准，专注于感知点云的局部信息以生成显著特征。然而，合理的更广泛的感受域对于增强特征区分度是必不可少的。在本文中，我们提出了一种用于点云配准的局部注意力哈希网络，称为LAHNet，它将类似于卷积操作的局部性归纳偏置的局部注意力机制引入点云描述符。具体而言，设计了一个组Transformer来捕捉点之间的合理长程上下文。这采用线性邻域搜索策略，局部敏感哈希，能够将点云均匀划分为不重叠的窗口。同时，采用高效的跨窗口策略来进一步扩展合理的特征感受域。此外，基于这种有效的窗口策略，我们提出了一个交互式Transformer来增强点云对内重叠区域的特征交互。这通过将每个窗口表示为全局信号来计算重叠矩阵，以匹配点云对之间的重叠区域。大量结果表明，LAHNet能够学习鲁棒且具有区分度的特征，在真实世界的室内和室外基准测试中取得了显著的配准结果。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决点云配准中特征区分度不足的问题。现有方法大多只关注点云的局部信息，缺乏合理的广泛感受野，导致特征不够独特。这个问题在现实中非常重要，因为点云配准是自动驾驶、机器人技术和3D重建等关键任务的基础，只有特征足够独特，才能在不同点云片段间找到可靠对应关系，实现精确配准。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者受到现有方法局限性的启发：大多数方法优化了局部特征提取但无法建模长程依赖关系；全局Transformer虽能捕获长程依赖但计算复杂度高且可能引入冗余信息。作者借鉴了Swin Transformer的局部注意力机制和移位窗口策略，针对点云的无序特性，使用局部敏感哈希(LSH)进行高效窗口划分。设计思路包括：分析感受野重要性、借鉴Transformer注意力机制、参考Swin的局部窗口策略，并针对点云特性进行改进。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是通过局部注意力机制建立点云中点之间的合理长程依赖关系，扩大特征感受野以提高特征区分度。整体流程：1)使用LSH将点云划分为非重叠窗口；2)Group Transformer对窗口内点应用局部自注意力，通过跨窗口交互扩大感受野；3)Interaction Transformer在U-Net瓶颈阶段，将窗口编码为全局信号计算重叠矩阵，匹配重叠区域并通过交叉注意力增强特征交互；4)编码器-解码器架构处理多尺度信息；5)使用对比损失训练模型确保正样本接近，负样本远离。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点：1)Group Transformer：使用LSH引入局部注意力机制，设计跨窗口交互策略扩大感受野；2)Interaction Transformer：基于重叠矩阵匹配点云对中的重叠区域，通过交叉注意力增强特征交互；3)LSH窗口划分策略：线性成本划分点云，避免体素化空体素问题和KNN的二次复杂度。相比之前工作不同：LAHNet关注合理长程依赖而非仅局部特征；使用局部注意力降低计算复杂度；专注于重叠区域而非全局交互；实现高效点云处理同时保持高性能。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'LAHNet通过局部敏感哈希驱动的局部注意力机制和交互式重叠区域处理，显著提高了点云配准中的特征区分度，实现了在室内外场景下高效且精确的点云对齐。'}


### 论文摘要

Most existing learning-based point cloud descriptors for point cloud registration focus on perceiving local information of point clouds to generate distinctive features. However, a reasonable and broader receptive field is essential for enhancing feature distinctiveness. In this paper, we propose a Local Attentive Hashing Network for point cloud registration, called LAHNet, which introduces a local attention mechanism with the inductive bias of locality of convolution-like operators into point cloud descriptors. Specifically, a Group Transformer is designed to capture reasonable long-range context between points. This employs a linear neighborhood search strategy, Locality-Sensitive Hashing, enabling uniformly partitioning point clouds into non-overlapping windows. Meanwhile, an efficient cross-window strategy is adopted to further expand the reasonable feature receptive field. Furthermore, building on this effective windowing strategy, we propose an Interaction Transformer to enhance the feature interactions of the overlap regions within point cloud pairs. This computes an overlap matrix to match overlap regions between point cloud pairs by representing each window as a global signal. Extensive results demonstrate that LAHNet can learn robust and distinctive features, achieving significant registration results on real-world indoor and outdoor benchmarks.

---

## 33. S2AM3D: Scale-controllable Part Segmentation of 3D Point Cloud

**论文链接:** [http://arxiv.org/abs/2512.00995v1](http://arxiv.org/abs/2512.00995v1)

**作者:** Han Su, Tianyu Huang, Zichen Wan, Xiaohe Wu, Wangmeng Zuo

**发布时间:** 2025-11-30

### GPT解析

### 总结

本研究提出了S2AM3D模型，通过结合2D分割先验和3D一致性监督，解决了点云分割中的泛化性和视图一致性问题，并构建了大规模高质量数据集，实现了在复杂结构和尺寸变化大的部分上的卓越性能。

### 背景

部分级点云分割在3D计算机视觉领域受到广泛关注，但现有研究面临两大挑战：原生3D模型因数据稀疏而缺乏泛化能力，而引入2D预训练知识则常导致不同视图间分割结果不一致。

### 目的

解决现有点云分割研究中存在的泛化性不足和视图不一致性问题，开发一种能够处理复杂结构和尺寸变化大的部分的分割方法。

### 方法

提出S2AM3D模型，包含点一致的部分编码器，通过原生3D对比学习聚合多视图2D特征，生成全局一致的点特征；以及尺度感知的提示解码器，可通过连续尺度信号实时调整分割粒度；同时引入包含超过10万样本的大规模高质量部分级点云数据集，为模型训练提供充足的监督信号。

### 主要发现

大量实验表明，S2AM3D在多种评估设置下达到领先性能，在处理复杂结构和尺寸差异大的部分时表现出卓越的鲁棒性和可控性。

### 结论

S2AM3D通过有效结合2D先验知识和3D一致性监督，显著提升了点云分割的性能和适用性，为处理复杂3D对象提供了新的解决方案。

### 翻译

部分级点云分割最近在3D计算机视觉领域引起了广泛关注。然而，现有研究受限于两大挑战：原生3D模型因数据稀疏而缺乏泛化能力，而引入2D预训练知识常导致不同视图间分割结果不一致。为解决这些挑战，我们提出了S2AM3D，它将2D分割先验与3D一致性监督相结合。我们设计了一个点一致的部分编码器，通过原生3D对比学习聚合多视图2D特征，生成全局一致的点特征。随后提出了一个尺度感知的提示解码器，可通过连续尺度信号实时调整分割粒度。同时，我们引入了一个包含超过10万样本的大规模高质量部分级点云数据集，为模型训练提供充足的监督信号。大量实验证明，S2AM3D在多种评估设置下达到领先性能，在处理复杂结构和尺寸差异大的部分时表现出卓越的鲁棒性和可控性。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决两个挑战：原生3D模型因数据稀疏导致泛化能力不足，以及引入2D预训练知识导致不同视图间分割结果不一致。这个问题很重要，因为点云级别的分割在3D计算机视觉中扮演关键角色，连接精细几何细节和高级语义理解，支持3D内容创建、机器人操作和逆向工程等应用，并能实现灵活的粒度调整，影响下游任务的可行性和效率。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者认识到现有方法的两难困境：纯3D方法泛化能力差，基于2D的方法在复杂情况下表现不佳。因此设计了一个混合方案，结合2D预训练知识和3D一致性监督。具体包括：设计点一致部件编码器通过3D对比学习聚合多视图2D特征；提出尺度感知提示解码器实现分割粒度调整；并构建大规模高质量数据集。该方法借鉴了SAM等2D预训练模型、对比学习、特征提取和多视图融合等现有技术。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是结合2D预训练知识与3D一致性监督，实现全局一致的点特征和灵活的粒度控制。整体流程：1)输入点云数据；2)点一致部件编码器提取特征并应用3D对比学习增强全局一致性；3)尺度感知提示解码器通过尺度调制器和双向交叉注意力实现特征交互；4)通过MLP和Sigmoid生成概率掩码。训练采用解耦方案，先稳定编码器，再训练解码器。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点：1)提出2D-3D训练配方，重用2D预训练知识并进行原生3D监督；2)设计尺度感知提示解码器，实现灵活的3D部件分割；3)引入可扩展数据管道，收集超过10万个标注点云实例。相比之前工作，S2AM3D解决了跨视图不一致问题，实现了粒度的连续控制，提供了更大规模的数据支持，并在处理复杂结构和尺寸变化大的部件时表现出更优的性能和可控性。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'S2AM3D通过结合2D预训练知识与3D一致性监督，实现了可扩展控制的3D点云部件分割，解决了跨视图不一致问题并提供了粒度灵活的分割能力。'}


### 论文摘要

Part-level point cloud segmentation has recently attracted significant attention in 3D computer vision. Nevertheless, existing research is constrained by two major challenges: native 3D models lack generalization due to data scarcity, while introducing 2D pre-trained knowledge often leads to inconsistent segmentation results across different views. To address these challenges, we propose S2AM3D, which incorporates 2D segmentation priors with 3D consistent supervision. We design a point-consistent part encoder that aggregates multi-view 2D features through native 3D contrastive learning, producing globally consistent point features. A scale-aware prompt decoder is then proposed to enable real-time adjustment of segmentation granularity via continuous scale signals. Simultaneously, we introduce a large-scale, high-quality part-level point cloud dataset with more than 100k samples, providing ample supervision signals for model training. Extensive experiments demonstrate that S2AM3D achieves leading performance across multiple evaluation settings, exhibiting exceptional robustness and controllability when handling complex structures and parts with significant size variations.

---

## 34. GFT: Graph Feature Tuning for Efficient Point Cloud Analysis

**论文链接:** [http://arxiv.org/abs/2511.10799v2](http://arxiv.org/abs/2511.10799v2)

**作者:** Manish Dhakal, Venkat R. Dasari, Rajshekhar Sunderraman, Yi Ding

**发布时间:** 2025-11-13

**备注:** Accepted to WACV 2026

### GPT解析

### 总结

本文提出了一种针对点云数据的参数高效微调方法Graph Features Tuning (GFT)，通过学习动态图并使用轻量级图卷积网络和交叉注意力模块，显著减少了可训练参数的数量，同时在物体分类和分割任务上保持与现有方法相当的性能。

### 背景

参数高效微调(PEFT)通过只更新模型的一小部分参数，显著降低了计算和内存成本，使模型能够更快地适应新任务，同时保持最小的性能损失。先前的研究已经针对点云数据引入了专门的PEFT方法，因为通用方法效果不佳。

### 目的

为了进一步减少可训练参数的数量，作者提出了一种专门针对点云的PEFT方法。

### 方法

提出图特征调优(GFT)，使用轻量级图卷积网络从transformer的初始标记输入中学习动态图，并通过跳跃连接和高效的交叉注意力模块将这些图特征传递到更深的层次。

### 主要发现

在物体分类和分割任务上的大量实验表明，GFT在相同领域内与现有方法竞争，同时减少了可训练参数。

### 结论

GFT是一种有效的点云参数高效微调方法，能够在减少可训练参数的同时保持与现有方法相当的性能。

### 翻译

参数高效微调(PEFT)通过仅更新模型参数的一小部分，显著降低了计算和内存成本，使模型能够更快地适应新任务，同时性能损失最小。先前的研究已经引入了针对点云数据的PEFT方法，因为通用方法效果不佳。为了进一步减少可训练参数的数量，我们提出了一种专门针对点云的PEFT方法，称为图特征调优(GFT)，它使用轻量级图卷积网络从transformer的初始标记输入中学习动态图，并通过跳跃连接和高效的交叉注意力模块将这些图特征传递到更深的层次。在物体分类和分割任务上的大量实验表明，GFT在相同领域内与现有方法竞争，同时减少了可训练参数。代码可在https://github.com/manishdhakal/GFT获取。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决点云分析中的参数高效微调问题。现有的通用PEFT方法在点云任务中表现不佳，而现有的点云特定PEFT方法仍有参数效率不高的问题。这个问题很重要，因为点云数据在3D场景理解、自动驾驶等领域广泛应用，而高效的微调方法能显著减少计算和内存成本，使模型能更快适应新任务，同时降低存储多个任务模型的成本。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先分析了现有PEFT方法在点云任务中的局限性，注意到IDPT等方法在参数减少时性能下降明显。受Transformer中自注意力机制启发，将其视为动态图，并考虑用KNN图提高效率。作者借鉴了EdgeConv进行局部特征提取，借鉴了提示调优方法引入可学习标记，借鉴了跨注意力机制注入特征。与IDPT不同，GFT从早期就开始学习图特征并传递到深层，使用轻量级模块组合而非单一重型模块，并采用稀疏交互提高效率。', '这个方法的核心思想是什么？整体实现流程是怎样的？': 'GFT的核心思想是通过学习动态图来改进点云分析的参数高效微调，使用轻量级图卷积网络从Transformer初始输入中提取特征，并通过跳跃连接和跨注意力模块传递到更深层次。整体流程包括：1) 添加任务特定可学习提示到标记嵌入空间；2) 使用EdgeConv从标记嵌入空间提取图特征，通过KNN定义局部信息并构建多层图特征；3) 在特定层通过稀疏跨注意力模块将图特征注入到Transformer编码器；4) 使用更新后的特征进行下游任务。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1) 专门为点云设计的PEFT方法；2) 从Transformer初始输入学习动态图；3) 使用轻量级模块组合（总共仅0.73M参数）；4) 稀疏跨注意力交互机制。相比IDPT和DAPT，GFT从早期就开始学习图特征而非只在最后一层；参数效率更高（比IDPT减少57%，比DAPT减少34%）；使用多个轻量级模块而非单一重型模块；在参数预算减少时性能更稳定；引入任务特定提示提供更高自由度。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '本文提出了GFT，一种针对点云分析的参数高效微调方法，通过学习动态图特征并轻量级注入到Transformer中，显著减少了可训练参数数量同时保持了competitive的性能。'}


### 论文摘要

Parameter-efficient fine-tuning (PEFT) significantly reduces computational and memory costs by updating only a small subset of the model's parameters, enabling faster adaptation to new tasks with minimal loss in performance. Previous studies have introduced PEFTs tailored for point cloud data, as general approaches are suboptimal. To further reduce the number of trainable parameters, we propose a point-cloud-specific PEFT, termed Graph Features Tuning (GFT), which learns a dynamic graph from initial tokenized inputs of the transformer using a lightweight graph convolution network and passes these graph features to deeper layers via skip connections and efficient cross-attention modules. Extensive experiments on object classification and segmentation tasks show that GFT operates in the same domain, rivalling existing methods, while reducing the trainable parameters. Code is available at https://github.com/manishdhakal/GFT.

---

## 35. Domain-Decomposed Graph Neural Network Surrogate Modeling for Ice Sheets

**论文链接:** [http://arxiv.org/abs/2512.01888v1](http://arxiv.org/abs/2512.01888v1)

**作者:** Adrienne M. Propp, Mauro Perego, Eric C. Cyr, Anthony Gruber, Amanda A. Howard, Alexander Heinlein, Panos Stinis, Daniel M. Tartakovsky

**发布时间:** 2025-12-01

### GPT解析

### 总结

本文提出了一种基于物理启发的图神经网络代理模型，结合域分解策略和迁移学习技术，用于高效准确地模拟大规模偏微分方程系统，特别是在冰盖模拟和不确定性量化任务中表现出色。

### 背景

准确且高效的代理模型对于大规模偏微分方程模拟至关重要，特别是需要数百或数千次评估的不确定性量化任务。传统的全局代理模型在处理大规模系统时面临训练效率和泛化能力的挑战。

### 目的

开发一种直接处理非结构化网格的图神经网络代理模型，通过域分解策略提高训练效率和模型泛化能力，并利用迁移学习在数据有限条件下加速训练并提高准确性。

### 方法

引入域分解策略将网格划分为子域，并行训练局部GNN代理模型，聚合各子域预测结果，并应用迁移学习跨子域微调模型。将该方法应用于冰盖模拟，预测高分辨率网格上的全场速度。

### 主要发现

该方法能够准确预测高分辨率网格上的全场速度，相比训练单个全局代理模型显著减少训练时间，为不确定性量化目标提供了良好基础。图域分解与迁移学习相结合，为训练大规模PDE系统的GNN代理模型提供了可扩展且可靠的途径。

### 结论

基于图域分解和迁移学习的GNN代理模型训练方法为大规模PDE系统提供了可扩展且可靠的解决方案，在冰盖动力学之外具有广泛的应用潜力。

### 翻译

准确而高效的代理模型对于偏微分方程的大规模模拟至关重要，特别是在需要数百或数千次评估的不确定性量化任务中。我们开发了一种受物理启发的图神经网络代理模型，该模型直接处理非结构化网格并利用图注意力的灵活性。为了提高模型的训练效率和泛化能力，我们引入了域分解策略，将网格划分为子域，并行训练局部GNN代理模型，并聚合它们的预测。然后我们采用迁移学习来跨子域微调模型，在数据有限的情况下加速训练并提高准确性。应用于冰盖模拟，我们的方法能够准确预测高分辨率网格上的全场速度，相对于训练单个全局代理模型显著减少了训练时间，并为不确定性量化目标提供了良好的基础。我们的结果表明，图域分解与迁移学习相结合，为在大型PDE系统上训练GNN代理模型提供了可扩展且可靠的途径，在冰盖动力学之外具有广泛的应用潜力。


### 论文摘要

Accurate yet efficient surrogate models are essential for large-scale simulations of partial differential equations (PDEs), particularly for uncertainty quantification (UQ) tasks that demand hundreds or thousands of evaluations. We develop a physics-inspired graph neural network (GNN) surrogate that operates directly on unstructured meshes and leverages the flexibility of graph attention. To improve both training efficiency and generalization properties of the model, we introduce a domain decomposition (DD) strategy that partitions the mesh into subdomains, trains local GNN surrogates in parallel, and aggregates their predictions. We then employ transfer learning to fine-tune models across subdomains, accelerating training and improving accuracy in data-limited settings. Applied to ice sheet simulations, our approach accurately predicts full-field velocities on high-resolution meshes, substantially reduces training time relative to training a single global surrogate model, and provides a ripe foundation for UQ objectives. Our results demonstrate that graph-based DD, combined with transfer learning, provides a scalable and reliable pathway for training GNN surrogates on massive PDE-governed systems, with broad potential for application beyond ice sheet dynamics.

---

## 36. 论文ID: 2512.01878v1

**错误:** 无法读取论文信息 - [Errno 2] No such file or directory: 'storage/result/2512.01878v1.json'

---

## 37. Morphling: Fast, Fused, and Flexible GNN Training at Scale

**论文链接:** [http://arxiv.org/abs/2512.01678v1](http://arxiv.org/abs/2512.01678v1)

**作者:** Anubhab, Rupesh Nasre

**发布时间:** 2025-12-01

### GPT解析

### 总结

本文介绍了Morphling，一个专门为图神经网络设计的领域特定代码合成器，旨在解决图神经网络在硬件执行方面的根本性挑战。

### 背景

图神经网络面临一个基本硬件挑战，即将不规则的、内存限制的图遍历与规则的、计算密集的密集矩阵操作相融合。现有框架如PyTorch Geometric和Deep Graph Library优先考虑高级可用性，但未能解决这些不同的执行特性，导致它们依赖通用内核，存在缓存局部性差、内存移动过多和大量中间分配的问题。

### 目的

开发Morphling来解决现有框架的局限性，通过领域特定的代码合成来弥合图遍历和密集矩阵操作之间的执行差距。

### 方法

Morphling通过将高级GNN规范编译为针对OpenMP、CUDA和MPI的便携式、后端专门实现；实例化针对每个执行环境定制的优化、架构感知基元库；以及集成一个运行时稀疏感知执行引擎，该引擎使用输入特征统计信息动态选择密集或稀疏执行路径，减少对零值条目的不必要计算。

### 主要发现

在十一个真实世界数据集上评估显示，Morphling将每个epoch的训练吞吐量平均提高了20倍(CPU上)和19倍(GPU上)，超过PyG和DGL，峰值加速比达到66倍；Morphling的内存高效布局将峰值内存消耗减少了高达15倍，使得能够在商用硬件上进行大规模GNN训练。

### 结论

专门的、架构感知的代码合成提供了在多样化并行和分布式平台上实现高性能GNN执行的有效且可扩展的路径。

### 翻译

图神经网络(GNNs)通过将不规则的、内存限制的图遍历与规则的、计算密集的密集矩阵操作相结合，提出了一个基本的硬件挑战。虽然诸如PyTorch Geometric(PyG)和Deep Graph Library(DGL)之类的框架优先考虑高级可用性，但它们未能解决这些不同的执行特性。因此，它们依赖于遭受缓存局部性差、内存移动过多和大量中间分配的通用内核。为了解决这些限制，我们提出了Morphling，一个旨在弥合这一差距的领域特定代码合成器。Morphling通过实例化为每个执行环境量身定制的优化、架构感知基元库，将高级GNN规范编译为针对OpenMP、CUDA和MPI的便携式、后端专门实现。Morphling还集成了一个运行时稀疏感知执行引擎，该引擎使用输入特征统计信息动态选择密集或稀疏执行路径，减少对零值条目的不必要计算。我们在跨越不同图结构、特征维度和稀疏性的十一个真实世界数据集上评估Morphling。结果表明，Morphling将每个epoch的训练吞吐量平均提高了20倍(CPU上)和19倍(GPU上)，超过了PyG和DGL，峰值加速比达到66倍。Morphling的内存高效布局进一步将峰值内存消耗减少了高达15倍，使得能够在商用硬件上进行大规模GNN训练。这些发现表明，专门的、架构感知的代码合成提供了在多样化并行和分布式平台上实现高性能GNN执行的有效且可扩展的路径。


### 论文摘要

Graph Neural Networks (GNNs) present a fundamental hardware challenge by fusing irregular, memory-bound graph traversals with regular, compute-intensive dense matrix operations. While frameworks such as PyTorch Geometric (PyG) and Deep Graph Library (DGL) prioritize high-level usability, they fail to address these divergent execution characteristics. As a result, they rely on generic kernels that suffer from poor cache locality, excessive memory movement, and substantial intermediate allocations. To address these limitations, we present Morphling, a domain-specific code synthesizer designed to bridge this gap. Morphling compiles high-level GNN specifications into portable, backend-specialized implementations targeting OpenMP, CUDA, and MPI. It achieves this by instantiating a library of optimized, architecture-aware primitives tailored to each execution environment. Morphling also incorporates a runtime sparsity-aware execution engine that dynamically selects dense or sparse execution paths using input feature statistics, reducing unnecessary computation on zero-valued entries. We evaluate Morphling on eleven real-world datasets spanning diverse graph structures, feature dimensionalities, and sparsity regimes. The results show that Morphling improves per-epoch training throughput by an average of 20X on CPUs and 19X on GPUs over PyG and DGL, with peak speedups reaching 66X. Morphling's memory-efficient layouts further reduce peak memory consumption by up to 15X, enabling large-scale GNN training on commodity hardware. These findings demonstrate that specialized, architecture-aware code synthesis provides an effective and scalable path toward high-performance GNN execution across diverse parallel and distributed platforms.

---

## 38. 论文ID: 2512.01647v1

**错误:** 无法读取论文信息 - [Errno 2] No such file or directory: 'storage/result/2512.01647v1.json'

---

## 39. Efficiently Learning Branching Networks for Multitask Algorithmic Reasoning

**论文链接:** [http://arxiv.org/abs/2512.01113v1](http://arxiv.org/abs/2512.01113v1)

**作者:** Dongyue Li, Zhenshuo Zhang, Minxuan Duan, Edgar Dobriban, Hongyang R. Zhang

**发布时间:** 2025-11-30

**备注:** 31 pages. Preprint, to appear in KDD'26

### GPT解析

### 总结

该论文提出了分支神经网络（branching neural networks）和AutoBRANE算法，用于解决多任务算法推理中的负干扰问题，通过基于梯度的任务聚类和凸松弛优化，显著提高了模型在多个算法推理任务上的性能。

### 背景

算法推理已成为评估图神经网络和大语言模型推理能力的重要基准，但不同算法的执行步骤差异导致多任务训练时产生负干扰，影响模型性能。

### 目的

设计一个能够在多个算法推理任务上同时表现良好的单一模型架构，解决多任务训练中的负干扰问题。

### 方法

提出分支神经网络架构，并开发了AutoBRANE算法，通过基于梯度的亲和度分数对任务进行聚类，利用凸松弛优化将搜索复杂度从k^(nL)降低到O(nL)时间，可用于任何基础模型。

### 主要发现

AutoBRANE在CLRS基准测试上比最强单一多任务GNN提高3.7%，比最佳基线提高1.2%，同时减少48%运行时间和26%内存使用；在文本推理基准上提高3.2%；在大图数据集上提高28%准确率并减少4.5倍运行时间；学习到的分支结构揭示了相关算法的合理层次聚类。

### 结论

分支神经网络架构和AutoBRANE算法有效解决了多任务算法推理中的负干扰问题，显著提高了模型在多种算法推理任务上的性能和效率。

### 翻译

算法推理——执行逐步逻辑推理的能力——已成为评估图神经网络和大语言模型推理能力的重要基准。理想情况下，人们希望设计一个能够在多个算法推理任务上同时表现良好的单一模型。然而，当不同算法的执行步骤不同时，这具有挑战性，因为一起训练时会产生负干扰。我们提出了分支神经网络，这是一种用于多任务算法推理的原则性架构。在n个算法任务上搜索具有L层的最优k叉树是组合问题，需要探索多达k^(nL)种可能的结构。我们开发了AutoBRANE算法，通过在每层解决凸松弛问题来近似最优任务划分，将搜索复杂度降低到O(nL)时间。该方法使用基于梯度的亲和度分数对任务进行聚类，可用于任何基础模型，包括GNNs和LLMs。我们在广泛的基于图的算法和基于文本的推理基准测试上验证了AutoBRANE。我们表明，在四个GNNs和四个LLMs（多达340亿参数）上，梯度特征能够以5%的误差估计真实任务性能。在CLRS基准测试上，它比最强的单一多任务GNN提高3.7%，比最佳基线提高1.2%，同时运行时间减少48%，内存使用减少26%。学习到的分支结构揭示了相关算法的直观合理的层次聚类。在三个基于文本的图推理基准测试上，AutoBRANE比最佳的非分支多任务基线提高3.2%。最后，在一个包含2100万条边和500个任务的大图数据集上，AutoBRANE比现有的多任务和分支架构提高28%的准确率，同时运行时间减少4.5倍。


### 论文摘要

Algorithmic reasoning -- the ability to perform step-by-step logical inference -- has become a core benchmark for evaluating reasoning in graph neural networks (GNNs) and large language models (LLMs). Ideally, one would like to design a single model capable of performing well on multiple algorithmic reasoning tasks simultaneously. However, this is challenging when the execution steps of algorithms differ from one another, causing negative interference when they are trained together.   We propose branching neural networks, a principled architecture for multitask algorithmic reasoning. Searching for the optimal $k$-ary tree with $L$ layers over $n$ algorithmic tasks is combinatorial, requiring exploration of up to $k^{nL}$ possible structures. We develop AutoBRANE, an efficient algorithm that reduces this search to $O(nL)$ time by solving a convex relaxation at each layer to approximate an optimal task partition. The method clusters tasks using gradient-based affinity scores and can be used on top of any base model, including GNNs and LLMs.   We validate AutoBRANE on a broad suite of graph-algorithmic and text-based reasoning benchmarks. We show that gradient features estimate true task performance within 5% error across four GNNs and four LLMs (up to 34B parameters). On the CLRS benchmark, it outperforms the strongest single multitask GNN by 3.7% and the best baseline by 1.2%, while reducing runtime by 48% and memory usage by 26%. The learned branching structures reveal an intuitively reasonable hierarchical clustering of related algorithms. On three text-based graph reasoning benchmarks, AutoBRANE improves over the best non-branching multitask baseline by 3.2%. Finally, on a large graph dataset with 21M edges and 500 tasks, AutoBRANE achieves a 28% accuracy gain over existing multitask and branching architectures, along with a 4.5$\times$ reduction in runtime.

---

## 40. 论文ID: 2512.00936v1

**错误:** 无法读取论文信息 - [Errno 2] No such file or directory: 'storage/result/2512.00936v1.json'

---

## 41. City-Conditioned Memory for Multi-City Traffic and Mobility Forecasting

**论文链接:** [http://arxiv.org/abs/2512.00851v1](http://arxiv.org/abs/2512.00851v1)

**作者:** Wenzhang Du

**发布时间:** 2025-11-30

### GPT解析

### 总结

本文提出了CityCond，一个轻量级城市条件化记忆层，用于增强现有的时空预测骨干网络，通过城市ID编码器和共享记忆库提高多城市交通预测的准确性和适应性。

### 背景

在多个城市部署时空预测模型存在困难：交通网络在大小和拓扑结构上各不相同，数据可用性可能相差几个数量级，新城市可能只提供简短的历史日志记录。现有的深度交通模型通常是按城市和骨干网络分别训练的，导致高昂的维护成本和对数据稀缺城市的迁移能力差。

### 目的

研究是否可以设计一个单一、与骨干网络无关的层，能够根据'这个序列来自哪个城市'进行条件化，在全数据和少数据情况下提高准确性，并以最少的代码变更支持更好的跨城市适应性。

### 方法

提出CityCond，结合城市ID编码器和可选的共享记忆库(CityMem)，通过门控残差连接产生融合的城市条件化特征。将CityCond附加到五个代表性骨干网络(GRU, TCN, Transformer, GNN, STGCN)上，并在METR-LA、PEMS-BAY和SIND数据集上评估全数据、少数据和跨城市少样本迁移三种情况。

### 主要发现

在超过十四种模型变体和三个随机种子上，CityCond始终带来一致的改进，对于高容量骨干网络（如Transformer和STGCN）的改进最大。CityMem将Transformer的错误率减少了约三分之一，并在少数据和跨城市迁移中带来显著增益。简单的城市ID条件化适度改善了少数据LSTM性能。

### 结论

CityCond可以作为可重用的设计模式，用于在现实数据约束下可扩展的多城市预测。

### 翻译

在多个城市部署时空预测模型很困难：交通网络在大小和拓扑结构上各不相同，数据可用性可能相差几个数量级，新城市可能只提供简短的历史日志记录。现有的深度交通模型通常是按城市和骨干网络分别训练的，这导致高昂的维护成本和对数据稀缺城市的迁移能力差。我们研究是否可以设计一个单一、与骨干网络无关的层，能够根据'这个序列来自哪个城市'进行条件化，在全数据和少数据情况下提高准确性，并以最少的代码变更支持更好的跨城市适应性。我们提出了CityCond，一个轻量级城市条件化记忆层，用于增强现有的时空骨干网络。CityCond结合了城市ID编码器和可选的共享记忆库(CityMem)。给定城市索引和骨干网络隐藏状态，它通过门控残差连接产生融合的城市条件化特征。我们将CityCond附加到五个代表性骨干网络(GRU, TCN, Transformer, GNN, STGCN)上，并在METR-LA和PEMS-BAY数据集上评估了三种情况：全数据、少数据和跨城市少样本迁移。我们还在SIND数据集上进行了辅助实验。在超过十四种模型变体和三个随机种子上，CityCond始终带来一致的改进，对于高容量骨干网络的改进最大。在全数据设置中，CityMem将Transformer的错误率减少了约三分之一，并在少数据和跨城市迁移中带来了显著增益。在SIND数据集上，简单的城市ID条件化适度改善了少数据LSTM性能。因此，CityCond可以作为可重用的设计模式，用于在现实数据约束下可扩展的多城市预测。


### 论文摘要

Deploying spatio-temporal forecasting models across many cities is difficult: traffic networks differ in size and topology, data availability can vary by orders of magnitude, and new cities may provide only a short history of logs. Existing deep traffic models are typically trained per city and backbone, creating high maintenance cost and poor transfer to data-scarce cities. We ask whether a single, backbone-agnostic layer can condition on "which city this sequence comes from", improve accuracy in full- and low-data regimes, and support better cross-city adaptation with minimal code changes.   We propose CityCond, a light-weight city-conditioned memory layer that augments existing spatio-temporal backbones. CityCond combines a city-ID encoder with an optional shared memory bank (CityMem). Given a city index and backbone hidden states, it produces city-conditioned features fused through gated residual connections. We attach CityCond to five representative backbones (GRU, TCN, Transformer, GNN, STGCN) and evaluate three regimes: full-data, low-data, and cross-city few-shot transfer on METR-LA and PEMS-BAY. We also run auxiliary experiments on SIND, a drone-based multi-agent trajectory dataset from a signalized intersection in Tianjin (we focus on pedestrian tracks).   Across more than fourteen model variants and three random seeds, CityCond yields consistent improvements, with the largest gains for high-capacity backbones such as Transformers and STGCNs. CityMem reduces Transformer error by roughly one third in full-data settings and brings substantial gains in low-data and cross-city transfer. On SIND, simple city-ID conditioning modestly improves low-data LSTM performance. CityCond can therefore serve as a reusable design pattern for scalable, multi-city forecasting under realistic data constraints.

---

## 42. MS-PPO: Morphological-Symmetry-Equivariant Policy for Legged Robot Locomotion

**论文链接:** [http://arxiv.org/abs/2512.00727v1](http://arxiv.org/abs/2512.00727v1)

**作者:** Sizhe Wei, Xulin Chen, Fengze Xie, Garrett Ethan Katz, Zhenyu Gan, Lu Gan

**发布时间:** 2025-11-30

### GPT解析

### 总结

本文提出了一种名为MS-PPO的形态-对称等变策略学习框架，通过将机器人的运动学结构和形态对称性直接编码到策略网络中，显著提高了足式机器人运动学习的训练效率和泛化能力。

### 背景

强化学习最近使足式机器人实现了令人印象深刻的运动能力，但大多数策略架构仍然与形态和对称性无关，导致训练效率低下和泛化能力有限。

### 目的

引入一种能够编码机器人运动学结构和形态对称性的策略学习框架，以解决现有方法的训练效率和泛化问题。

### 方法

构建形态感知的图神经网络架构，该架构在机器人的形态对称群作用下具有可证明的等变性，确保对称状态下策略响应的一致性，同时保持价值估计的不变性，从而消除对奖励塑造或数据增强的需求。

### 主要发现

在Unitree Go2和小米CyberDog2机器人上的多种运动任务（包括小跑、pronking、斜坡行走和双足转向）实验表明，MS-PPO在训练稳定性、对称泛化能力和样本效率方面优于最先进的基线方法。

### 结论

将运动学结构和形态对称性嵌入策略学习为足式机器人运动控制提供了强大的归纳偏置，显著改善了学习效果。

### 翻译

强化学习最近使足式机器人实现了令人印象深刻的运动能力；然而，大多数策略架构仍然与形态和对称性无关，导致训练效率低下和泛化能力有限。本文介绍了MS-PPO，一种形态-对称等变策略学习框架，将机器人的运动学结构和形态对称性直接编码到策略网络中。我们构建了一个形态感知的图神经网络架构，该架构在机器人的形态对称群作用下具有可证明的等变性，确保在对称状态下策略响应的一致性，同时保持价值估计的不变性。这种设计消除了繁琐的奖励塑造或昂贵的数据增强的需求，这些通常是强制执行对称性所必需的。我们在Unitree Go2和小米CyberDog2机器人上对MS-PPO进行了仿真评估，包括小跑、pronking、斜坡行走和双足转向等多种运动任务，并在硬件上部署了学到的策略。大量实验表明，与最先进的基线方法相比，MS-PPO在具有挑战性的运动任务中实现了更好的训练稳定性、对称泛化能力和样本效率。这些发现表明，将运动学结构和形态对称性嵌入策略学习为足式机器人运动控制提供了强大的归纳偏置。我们的代码将在https://lunarlab-gatech.github.io/MS-PPO/上公开提供。


### 论文摘要

Reinforcement learning has recently enabled impressive locomotion capabilities on legged robots; however, most policy architectures remain morphology- and symmetry-agnostic, leading to inefficient training and limited generalization. This work introduces MS-PPO, a morphological-symmetry-equivariant policy learning framework that encodes robot kinematic structure and morphological symmetries directly into the policy network. We construct a morphology-informed graph neural architecture that is provably equivariant with respect to the robot's morphological symmetry group actions, ensuring consistent policy responses under symmetric states while maintaining invariance in value estimation. This design eliminates the need for tedious reward shaping or costly data augmentation, which are typically required to enforce symmetry. We evaluate MS-PPO in simulation on Unitree Go2 and Xiaomi CyberDog2 robots across diverse locomotion tasks, including trotting, pronking, slope walking, and bipedal turning, and further deploy the learned policies on hardware. Extensive experiments show that MS-PPO achieves superior training stability, symmetry generalization ability, and sample efficiency in challenging locomotion tasks, compared to state-of-the-art baselines. These findings demonstrate that embedding both kinematic structure and morphological symmetry into policy learning provides a powerful inductive bias for legged robot locomotion control. Our code will be made publicly available at https://lunarlab-gatech.github.io/MS-PPO/.

---

## 43. Graph Data Augmentation with Contrastive Learning on Covariate Distribution Shift

**论文链接:** [http://arxiv.org/abs/2512.00716v1](http://arxiv.org/abs/2512.00716v1)

**作者:** Fanlong Zeng, Wensheng Gan

**发布时间:** 2025-11-30

**备注:** 8 tables, 8 figures

### GPT解析

### 总结

本文提出了一种名为MPAIACL的新方法，用于解决图数据中的协变量分布偏移问题。该方法利用对比学习充分利用潜在空间中的信息，在各种公共OOD数据集上表现出强大的泛化能力和有效性。

### 背景

协变量分布偏移是当测试集中存在某些结构特征而训练集中不存在时发生的问题，是分布外（OOD）问题的一种常见类型，在具有复杂结构的现实图数据中经常遇到。

### 目的

解决现有图神经网络无法处理协变量偏移以及现有方法未能充分利用潜在空间信息的问题。

### 方法

提出名为MPAIACL（More Powerful Adversarial Invariant Augmentation using Contrastive Learning）的新方法，利用对比学习来充分利用向量表示的内在信息。

### 主要发现

通过大量实验验证，MPAIACL在各种公共OOD数据集上与其他基线方法相比表现良好，展示了其强大的泛化能力和有效性。

### 结论

MPAIACL是解决图数据中协变量分布偏移问题的有效方法，能够充分利用潜在空间中的信息。

### 翻译

当测试集中存在某些结构特征而训练集中不存在时，会发生协变量分布偏移。这是分布外（OOD）问题的一种常见类型，在具有复杂结构的现实图数据中经常遇到。现有研究表明，大多数现成的图神经网络（GNNs）无法处理协变量偏移。此外，我们还观察到，现有旨在解决协变量偏移的方法往往未能充分利用潜在空间中包含的丰富信息。受潜在空间的潜力启发，我们引入了一种名为MPAIACL的新方法，即使用对比学习的更强大对抗不变增强。MPAIACL利用对比学习通过利用其内在信息来充分释放向量表示的潜力。通过大量实验，MPAIACL展示了其强大的泛化能力和有效性，在各种公共OOD数据集上与其他基线方法相比表现良好。代码可在https://github.com/flzeng1/MPAIACL公开获取。


### 论文摘要

Covariate distribution shift occurs when certain structural features present in the test set are absent from the training set. It is a common type of out-of-distribution (OOD) problem, frequently encountered in real-world graph data with complex structures. Existing research has revealed that most out-of-the-box graph neural networks (GNNs) fail to account for covariate shifts. Furthermore, we observe that existing methods aimed at addressing covariate shifts often fail to fully leverage the rich information contained within the latent space. Motivated by the potential of the latent space, we introduce a new method called MPAIACL for More Powerful Adversarial Invariant Augmentation using Contrastive Learning. MPAIACL leverages contrastive learning to unlock the full potential of vector representations by harnessing their intrinsic information. Through extensive experiments, MPAIACL demonstrates its robust generalization and effectiveness, as it performs well compared with other baselines across various public OOD datasets. The code is publicly available at https://github.com/flzeng1/MPAIACL.

---

## 44. 论文ID: 2512.00696v1

**错误:** 无法读取论文信息 - [Errno 2] No such file or directory: 'storage/result/2512.00696v1.json'

---

## 45. Generalized Graph Transformer Variational Autoencoder

**论文链接:** [http://arxiv.org/abs/2512.00612v1](http://arxiv.org/abs/2512.00612v1)

**作者:** Siddhant Karki

**发布时间:** 2025-11-29

### GPT解析

### 总结

这篇论文提出了广义图变分自编码器（GGT-VAE）用于图链接预测，结合了广义图变换器架构和变分自编码器框架，利用transformer风格的全局自注意力机制和拉普拉斯位置编码在潜在空间中建模节点间的结构模式。

### 背景

图链接预测一直是网络分析和生成建模中图表示学习的核心问题。深度学习的进步引入了日益复杂的架构来捕捉图结构数据中的关系依赖。

### 目的

提出一种新的模型用于图链接预测，该模型能够有效捕捉图中的结构模式。

### 方法

提出广义图变分自编码器（GGT-VAE），结合广义图变换器架构与变分自编码器框架，利用transformer风格的全局自注意力机制和拉普拉斯位置编码，在潜在空间中建模节点间的结构模式，不依赖于消息传递。

### 主要发现

在几个基准数据集上，GGT-VAE在ROC-AUC和平均精度方面始终达到高于基线的性能。

### 结论

GGT-VAE是一种有效的图链接预测方法，据作者所知，这是首批在变分框架中使用广义图变换器主干探索图结构生成的研究之一。

### 翻译

图链接预测长期以来一直是网络分析和生成建模中图表示学习的核心问题。深度学习的最新进展引入了日益复杂的架构来捕捉图结构数据中的关系依赖。在这项工作中，我们提出了广义图变分自编码器（GGT-VAE）。我们的模型将广义图变换器架构与变分自编码器框架相结合用于链接预测。与之前的GraphVAE、GCN或GNN方法不同，GGT-VAE利用transformer风格的全局自注意力机制以及拉普拉斯位置编码，将节点间的结构模式建模到潜在空间中，而不依赖于消息传递。在几个基准数据集上的实验结果表明，GGT-VAE在ROC-AUC和平均精度方面始终达到高于基线的性能。据我们所知，这是首批在变分框架中使用广义图变换器主干探索图结构生成的研究之一。


### 论文摘要

Graph link prediction has long been a central problem in graph representation learning in both network analysis and generative modeling. Recent progress in deep learning has introduced increasingly sophisticated architectures for capturing relational dependencies within graph-structured data. In this work, we propose the Generalized Graph Transformer Variational Autoencoder (GGT-VAE). Our model integrates Generalized Graph Transformer Architecture with Variational Autoencoder framework for link prediction. Unlike prior GraphVAE, GCN, or GNN approaches, GGT-VAE leverages transformer style global self-attention mechanism along with laplacian positional encoding to model structural patterns across nodes into a latent space without relying on message passing. Experimental results on several benchmark datasets demonstrate that GGT-VAE consistently achieves above-baseline performance in terms of ROC-AUC and Average Precision. To the best of our knowledge, this is among the first studies to explore graph structure generation using a generalized graph transformer backbone in a variational framework.

---

## 46. TrojanLoC: LLM-based Framework for RTL Trojan Localization

**论文链接:** [http://arxiv.org/abs/2512.00591v1](http://arxiv.org/abs/2512.00591v1)

**作者:** Weihua Xiao, Zeng Wang, Minghao Shao, Raghu Vamshi Hemadri, Ozgur Sinanoglu, Muhammad Shafique, Johann Knechtel, Siddharth Garg, Ramesh Karri

**发布时间:** 2025-11-29

### GPT解析

### 总结

论文提出了一种基于大语言模型(LLM)的硬件木马(HT)检测框架TrojanLoC，能够在寄存器传输级(RTL)实现模块级检测和行级精确定位。

### 背景

硬件木马是对集成电路的持续威胁，特别是在RTL级插入时。现有方法将设计转换为图结构并使用图神经网络(GNN)，但存在三个问题：失去紧凑RTL语义、依赖浅层GNN感受野有限、局限于粗粒度模块级二进制检测。

### 目的

开发一种能够保留RTL语义、利用深度学习模型进行细粒度硬件木马检测和定位的方法，突破现有技术的局限性。

### 方法

提出TrojanLoC框架，使用RTL微调的LLM直接从RTL代码提取模块级和行级嵌入，捕获全局设计上下文和局部语义；训练任务特定分类器进行模块级检测、类型预测和行级定位；同时创建TrojanInS大型合成RTL设计数据集，包含四种基于效果类别的系统注入木马及精确行级注释。

### 主要发现

TrojanLoC在模块级检测上达到0.99的F1分数，比基线高0.68；在木马类型分类上达到0.84的macro-F1；在行级定位上达到0.93的macro-F1，能够实现与木马相关的RTL行的细粒度定位。

### 结论

基于LLM的框架能够有效解决现有硬件木马检测方法的局限性，实现高性能的模块级检测和细粒度的行级定位，为RTL级硬件木马安全提供了新途径。

### 翻译

硬件木马(HTs)是对集成电路的持续威胁，特别是在寄存器传输级(RTL)插入时。现有方法通常先将设计转换为图，如门级网表或RTL派生的数据流图(DFG)，然后使用图神经网络(GNN)获取该图的嵌入，这种方法(i)失去了紧凑的RTL语义，(ii)依赖于具有有限感受野的浅层GNN，(iii)主要局限于粗粒度的模块级二进制HT检测。我们提出了TrojanLoC，一种基于LLM的RTL级HT定位框架。我们使用RTL微调的LLM直接从RTL代码派生模块级和行级嵌入，同时捕获全局设计上下文和局部语义。接下来，我们在这些嵌入上训练任务特定的分类器，以执行模块级木马检测、类型预测和细粒度的行级定位。我们还介绍了TrojanInS，这是一个大型合成RTL设计数据集，包含从四种基于效果的类别中系统注入的木马，每个都配有精确的行级注释。我们的实验表明，TrojanLoC在模块级上取得了强大的性能，木马检测达到0.99的F1分数，比基线高0.68，木马类型分类达到0.84的macro-F1。在行级上，TrojanLoc进一步达到0.93的macro-F1，能够实现与木马相关的RTL行的细粒度定位。


### 论文摘要

Hardware Trojans (HT s) are a persistent threat to integrated circuits, especially when inserted at the register-transfer level (RTL). Existing methods typically first convert the design into a graph, such as a gate-level netlist or an RTL-derived dataflow graph (DFG), and then use a graph neural network (GNN ) to obtain an embedding of that graph, which (i) loses compact RTL semantics, (ii) relies on shallow GNNs with limited receptive field, and (iii) is largely restricted to coarse, module-level binary HT detection. We propose TrojanLoC, an LLM-based framework for RTL-level HT localization. We use an RTL-finetuned LLM to derive module-level and line-level embeddings directly from RTL code, capturing both global design context and local semantics. Next, we train task-specific classifiers on these embeddings to perform module-level Trojan detection, type prediction, and fine-grained line-level localization. We also introduce TrojanInS, a large synthetic dataset of RTL designs with systematically injected Trojans from four effect-based categories, each accompanied by precise line-level annotations. Our experiments show that TrojanLoC achieves strong module-level performance, reaching 0.99 F1-score for Trojan detection, up to 0.68 higher than baseline, and 0.84 macro-F1 for Trojan-type classification. At the line level, TrojanLoc further achieves up to 0.93 macro-F1, enabling fine-grained localization of Trojan-relevant RTL lines

---

## 47. A Graph Neural Network Approach for Localized and High-Resolution Temperature Forecasting

**论文链接:** [http://arxiv.org/abs/2512.00546v1](http://arxiv.org/abs/2512.00546v1)

**作者:** Joud El-Shawa, Elham Bagheri, Sedef Akinli Kocak, Yalda Mohsenzadeh

**发布时间:** 2025-11-29

**备注:** 6 pages, 2 figures. Accepted to the NeurIPS 2025 Tackling Climate Change with Machine Learning Workshop

### GPT解析

### 总结

该研究提出了一种基于图神经网络的局部高分辨率温度预测框架，能够生成长达48小时的预测，并在加拿大安大略省西南部的测试中取得了良好的预测效果。

### 背景

热浪在全球范围内加剧，是最致命的天气灾害之一。边缘化人口和全球南方国家承受不成比例的负担，这些地区资源不足的医疗系统、城市热岛暴露和适应性基础设施的缺乏放大了风险。然而，当前的数值天气预报模型往往无法捕捉微观尺度的极端情况，使得最脆弱的人群无法及时获得早期预警。

### 目的

开发一种用于局部、高分辨率温度预测的图神经网络框架。

### 方法

利用空间学习和高效计算，生成多个时间范围的温度预测，最长可达48小时。

### 主要发现

在加拿大安大略省西南部，该模型在1-48小时的预测中，平均绝对误差约为1.93度，在最大区域使用24小时输入窗口评估时，48小时预测的绝对误差约为2.93度。

### 结论

虽然在数据丰富的背景下进行了演示，但这项工作为迁移学习方法奠定了基础，可以在全球南方数据有限的地区实现局部、公平的预测。

### 翻译

热浪正在全球范围内加剧，是最致命的天气灾害之一。负担不成比例地落在边缘化人口和全球南方国家，这些地区资源不足的医疗系统、城市热岛暴露和适应性基础设施的缺乏放大了风险。然而，当前的数值天气预报模型往往无法捕捉微观尺度的极端情况，使得最脆弱的人群无法及时获得早期预警。我们提出了一种用于局部、高分辨率温度预测的图神经网络框架。通过利用空间学习和高效计算，我们的方法能够生成多个时间范围的预测，最长可达48小时。在加拿大安大略省西南部，该模型在1-48小时的预测中，平均绝对误差约为1.93度，在最大区域使用24小时输入窗口评估时，48小时预测的绝对误差约为2.93度。虽然在数据丰富的背景下进行了演示，但这项工作为迁移学习方法奠定了基础，可以在全球南方数据有限的地区实现局部、公平的预测。


### 论文摘要

Heatwaves are intensifying worldwide and are among the deadliest weather disasters. The burden falls disproportionately on marginalized populations and the Global South, where under-resourced health systems, exposure to urban heat islands, and the lack of adaptive infrastructure amplify risks. Yet current numerical weather prediction models often fail to capture micro-scale extremes, leaving the most vulnerable excluded from timely early warnings. We present a Graph Neural Network framework for localized, high-resolution temperature forecasting. By leveraging spatial learning and efficient computation, our approach generates forecasts at multiple horizons, up to 48 hours. For Southwestern Ontario, Canada, the model captures temperature patterns with a mean MAE of 1.93$^{\circ}$C across 1-48h forecasts and MAE@48h of 2.93$^{\circ}$C, evaluated using 24h input windows on the largest region. While demonstrated here in a data-rich context, this work lays the foundation for transfer learning approaches that could enable localized, equitable forecasts in data-limited regions of the Global South.

---

## 48. Hyperbolic Continuous Structural Entropy for Hierarchical Clustering

**论文链接:** [http://arxiv.org/abs/2512.00524v1](http://arxiv.org/abs/2512.00524v1)

**作者:** Guangjie Zeng, Hao Peng, Angsheng Li, Li Sun, Chunyang Liu, Shengze Li, Yicheng Pan, Philip S. Yu

**发布时间:** 2025-11-29

**备注:** 14 pages, accepted by AAAI 2026

### GPT解析

### 总结

本文提出了一种名为HypCSE的Hyperbolic Continuous Structural Entropy神经网络，用于解决层次聚类中的两个主要挑战，并通过双曲空间映射和图结构学习实现了优越性能。

### 背景

层次聚类是基础机器学习技术，用于将数据点分组为树状图。现有方法面临两个主要挑战：大多数方法指定树状图时没有全局目标；基于图的方法常忽视图结构重要性，在完全图或静态预定义图上优化目标。

### 目的

提出HypCSE方法，用于增强结构的连续层次聚类，解决现有方法中的不足。

### 方法

在双曲空间中映射数据点，并在增强结构的图上最小化松弛的连续结构熵。使用双曲图神经网络编码图顶点，通过树中最低公共祖先重新表述SE目标并松弛为连续SE，采用图结构学习策略在训练过程中更新图结构。

### 主要发现

在七个数据集上的大量实验证明HypCSE具有优越性能。

### 结论

HypCSE通过结合双曲空间表示、连续结构熵和图结构学习，有效解决了现有层次聚类方法的两个主要挑战。

### 翻译

层次聚类是一种基础机器学习技术，用于将数据点分组为树状图。然而，现有的层次聚类方法遇到两个主要挑战：1）大多数方法在没有全局目标的情况下指定树状图。2）基于图的方法常常忽视图结构的重要性，在完全图或静态预定义图上优化目标。在这项工作中，我们提出了用于增强结构连续层次聚类的双曲连续结构熵神经网络，即HypCSE。我们的核心思想是在双曲空间中映射数据点，并在增强结构的图上最小化松弛的连续结构熵（SE）。具体来说，我们使用双曲图神经网络在双曲空间中编码图顶点，并在图嵌入上最小化定义的近似SE。为了使SE目标可微分以进行优化，我们将其重新表述为使用树中最低公共祖先（LCA）的函数，然后通过双曲图嵌入和划分树的类比将其松弛为连续SE（CSE）。为确保图结构能有效捕捉数据点的层次结构以进行CSE计算，我们采用图结构学习（GSL）策略，在训练过程中更新图结构。在七个数据集上的大量实验证明了HypCSE的优越性能。


### 论文摘要

Hierarchical clustering is a fundamental machine-learning technique for grouping data points into dendrograms. However, existing hierarchical clustering methods encounter two primary challenges: 1) Most methods specify dendrograms without a global objective. 2) Graph-based methods often neglect the significance of graph structure, optimizing objectives on complete or static predefined graphs. In this work, we propose Hyperbolic Continuous Structural Entropy neural networks, namely HypCSE, for structure-enhanced continuous hierarchical clustering. Our key idea is to map data points in the hyperbolic space and minimize the relaxed continuous structural entropy (SE) on structure-enhanced graphs. Specifically, we encode graph vertices in hyperbolic space using hyperbolic graph neural networks and minimize approximate SE defined on graph embeddings. To make the SE objective differentiable for optimization, we reformulate it into a function using the lowest common ancestor (LCA) on trees and then relax it into continuous SE (CSE) by the analogy of hyperbolic graph embeddings and partitioning trees. To ensure a graph structure that effectively captures the hierarchy of data points for CSE calculation, we employ a graph structure learning (GSL) strategy that updates the graph structure during training. Extensive experiments on seven datasets demonstrate the superior performance of HypCSE.

---

## 49. TrendGNN: Towards Understanding of Epidemics, Beliefs, and Behaviors

**论文链接:** [http://arxiv.org/abs/2512.00421v1](http://arxiv.org/abs/2512.00421v1)

**作者:** Mulin Tian, Ajitesh Srivastava

**发布时间:** 2025-11-29

**备注:** 4 pages, 2 figures, 1 table

### GPT解析

### 总结

本文提出了一种基于图的预测框架，用于可解释地预测与人类信念和行为相关的流行病信号，通过构建信号关系图和应用图神经网络实现可解释分析。

### 背景

流行病结果与人类行为和信念之间存在复杂的相互作用。大多数预测文献专注于使用简单的机械模型或黑盒模型（如深度转换器）来预测流行病信号，但这些模型缺乏可解释性。

### 目的

为了更好地理解流行病机制并预测干预措施的影响，需要以可解释的方式预测与信念和行为相关的信号。

### 方法

提出了一种基于图的预测框架，首先基于趋势相似性构建相互关联信号的图，然后应用图神经网络进行预测。这种方法能够揭示哪些信号更可预测以及哪些关系对预测准确性贡献最大。

### 主要发现

该方法能够通过图结构分析提供可解释的洞察，识别出预测中的关键信号和重要关系，增强了模型的可解释性。

### 结论

该方法为在具有多个潜在相互依赖信号的领域中建立可解释模型提供了初步步骤，对构建整合行为、信念和观察的未来流行病模拟模型具有重要意义。

### 翻译

流行病结果与人类行为和信念之间存在复杂的相互作用。大多数预测文献专注于使用简单的机械模型或黑盒模型（如深度转换器）来预测流行病信号，但这些模型提供不了可解释性。然而，为了更好地理解机制并预测干预措施的影响，我们需要以可解释的方式预测与信念和行为相关的信号。在这项工作中，我们提出了一种基于图的预测框架，首先基于趋势相似性构建相互关联信号的图，然后应用图神经网络进行预测。这种方法通过揭示哪些信号更可预测以及哪些关系对预测准确性贡献最大，来实现可解释分析。我们认为我们的方法为在具有多个潜在相互依赖信号的领域中建立可解释模型提供了初步步骤，对构建整合行为、信念和观察的未来模拟模型具有重要意义。


### 论文摘要

Epidemic outcomes have a complex interplay with human behavior and beliefs. Most of the forecasting literature has focused on the task of predicting epidemic signals using simple mechanistic models or black-box models, such as deep transformers, that ingest all available signals without offering interpretability. However, to better understand the mechanisms and predict the impact of interventions, we need the ability to forecast signals associated with beliefs and behaviors in an interpretable manner. In this work, we propose a graph-based forecasting framework that first constructs a graph of interrelated signals based on trend similarity, and then applies graph neural networks (GNNs) for prediction. This approach enables interpretable analysis by revealing which signals are more predictable and which relationships contribute most to forecasting accuracy. We believe our method provides early steps towards a framework for interpretable modeling in domains with multiple potentially interdependent signals, with implications for building future simulation models that integrate behavior, beliefs, and observations.

---

## 50. Interpretable Graph Neural Networks for Classifying Structure and Magnetism in Delafossite Compounds

**论文链接:** [http://arxiv.org/abs/2512.00292v1](http://arxiv.org/abs/2512.00292v1)

**作者:** Jovin Ryan Joseph, Do Hoon Kiem, Sinchul Yeom, Mina Yoon

**发布时间:** 2025-11-29

**备注:** 6 main figures, 9 SI figures

### GPT解析

### 总结

该研究使用概念白化图神经网络对Delafossite材料结构进行分类，通过将AI模型与人类可解释的物理概念对齐，实现了对磁行为的准确预测和深入理解。

### 背景

Delafossites(ABC2，其中A和B是金属，C是硫族元素)是一类多功能量子材料和层状氧化物/硫族化合物，其性质对原子成分和堆叠几何结构高度敏感。它们的宽泛化学可调性使其成为大规模组合探索和高通量计算筛选的理想平台。

### 目的

使用概念白化图神经网络对Delafossite结构按堆叠序列和磁状态进行分类，通过将学习到的表示与人类可解释的物理概念对齐，实现准确预测并理解驱动磁行为的结构和化学特征。

### 方法

采用概念白化图神经网络(一种灰盒AI模型)，通过概念对齐分析学习物理上有意义的描述符，并将概念重要性映射到材料图表示上。

### 主要发现

磁序模型的验证准确率超过80%，包含最多概念的模型观察到轻微提升；概念对齐分析显示在九个物理上有意义的描述符上可测量的学习，其中d壳层价电子概念的确定系数约为0.6，磁耦合参数的确定系数在0.4-0.5之间；概念重要性映射阐明了可解释的物理趋势和训练过程中稳定概念对齐区域的进展。

### 结论

可解释的基于图的学习能够捕捉复杂材料系统的潜在物理学，为加速Delafossite和相关晶态材料的发现和理解提供了可解释的框架。

### 翻译

Delafossites(ABC2，其中A和B是金属，C是硫族元素)是一类多功能量子材料和层状氧化物/硫族化合物，其性质对原子成分和堆叠几何结构高度敏感。它们广泛的化学可调性使其成为大规模组合探索和高通量计算筛选的理想平台，具有理想的量子特性。在本工作中，我们采用概念白化图神经网络(一种灰盒AI模型)按堆叠序列和磁状态对Delafossite结构进行分类。通过将学习到的表示与人类可解释的物理概念对齐，这种灰盒方法实现了准确预测并深入了解驱动磁行为的结构和化学特征。磁序模型的验证准确率超过80%，在包含最多概念的模型中观察到轻微提升。概念对齐分析显示在九个物理上有意义的描述符上存在可测量的学习，其中d壳层价电子概念的确定系数约为0.6，磁耦合参数的确定系数在0.4-0.5之间。此外，我们将概念重要性映射到材料图表示上，阐明了可解释的物理趋势和训练过程中稳定概念对齐区域的进展。这些结果表明，可解释的基于图的学习具有捕捉复杂材料系统潜在物理学的潜力，并为加速Delafossite和相关晶态材料的发现和理解提供了可解释的框架。


### 论文摘要

Delafossites (ABC2, where A and B are metals and C is a chalcogen) are a versatile family of quantum materials and layered oxides/chalcogenides whose properties are highly sensitive to atomic composition and stacking geometry. Their broad chemical tunability makes them an ideal platform for large-scale combinatorial exploration and high-throughput computational screening with desirable quantum properties. In this work, we employ a Concept Whitening Graph Neural Network, a gray-box AI model, to classify delafossite structures by stacking sequence and magnetic states. By aligning learned representations with human-interpretable physical concepts, this gray-box approach enables both accurate prediction and insight into the structural and chemical features driving magnetic behavior. The magnetic-ordering models achieved validation accuracies exceeding 80 percent, with a further slight uptick observed in the model incorporating the largest number of concepts. Concept alignment analysis revealed measurable learning across nine physically meaningful descriptors, with coefficients of determination ranging from approximately 0.6 for the d-shell valence-electron concepts to 0.4-0.5 for the magnetic coupling parameters. Furthermore, we mapped the concept importances onto the material graph representation, elucidating interpretable physical trends and the progression of stable concept-aligned regions across training. These results demonstrate the potential of interpretable graph-based learning to capture the underlying physics of complex materials systems and provide an interpretable framework for accelerating the discovery and understanding of delafossites and related crystalline materials.

---

## 51. Polynomial Neural Sheaf Diffusion: A Spectral Filtering Approach on Cellular Sheaves

**论文链接:** [http://arxiv.org/abs/2512.00242v1](http://arxiv.org/abs/2512.00242v1)

**作者:** Alessio Borgi, Fabrizio Silvestri, Pietro Liò

**发布时间:** 2025-11-28

### GPT解析

### 总结

本文提出了一种称为多项式层化神经网络扩散(PolyNSD)的新方法，解决了传统层化神经网络扩散中存在的计算效率和稳定性问题。PolyNSD通过多项式递归和谱重缩放技术，实现了更高效的图神经网络，同时在同质性和异质性图上都取得了最先进的性能。

### 背景

层化神经网络为图结构配备了层化细胞层，这是一种几何结构，为节点和边分配局部向量空间(茎)和可学习的线性限制/传输映射，从而处理异质性和限制过度平滑问题。然而，常见的层化神经网络扩散实现依赖于基于SVD的归一化和密集的每边限制映射，这些方法与茎维度成比例，需要频繁重建拉普拉斯矩阵，并且产生脆弱的梯度。

### 目的

解决传统层化神经网络扩散方法中的计算效率和稳定性问题，特别是处理与茎维度相关的扩展性问题和脆弱梯度问题。

### 方法

PolyNSD是一种新的层化扩散方法，其传播算子是归一化层化拉普拉斯矩阵的K次多项式，通过在谱重缩放算子上的稳定三项递归来评估。这种方法提供了单层中的显式K跳感受野(独立于茎维度)，并获得作为K+1正交多项式基响应的凸混合的可训练谱响应。PolyNSD通过凸混合、谱重缩放和残差/门控路径强制稳定性。

### 主要发现

PolyNSD在同类和异类基准测试中达到了新的最先进结果，通过仅使用对角限制映射逆转了层化神经网络扩散的趋势，将性能与大茎维度解耦，同时减少了运行时间和内存需求。

### 结论

PolyNSD解决了传统层化神经网络扩散方法的计算效率和稳定性问题，通过多项式递归和谱重缩放技术实现了更高效的图神经网络，同时保持了或提高了在各类图数据上的性能。

### 翻译

层化神经网络为图结构配备了层化细胞层：一种几何结构，为节点和边分配局部向量空间(茎)和可学习的线性限制/传输映射，从而产生处理异质性和限制过度平滑的边缘感知归纳偏差。然而，常见的层化神经网络扩散实现依赖于基于SVD的层化归一化和密集的每边限制映射，这些方法与茎维度成比例，需要频繁重建拉普拉斯矩阵，并产生脆弱的梯度。为解决这些限制，我们引入了多项式层化神经网络扩散(PolyNSD)，一种新的层化扩散方法，其传播算子是归一化层化拉普拉斯矩阵的K次多项式，通过在谱重缩放算子上的稳定三项递归来评估。这提供了单层中的显式K跳感受野(独立于茎维度)，并获得作为K+1正交多项式基响应的凸混合的可训练谱响应。PolyNSD通过凸混合、谱重缩放和残差/门控路径强制稳定性，在同类和异类基准测试中达到了新的最先进结果，通过仅使用对角限制映射逆转了层化神经网络扩散的趋势，将性能与大茎维度解耦，同时减少了运行时间和内存需求。


### 论文摘要

Sheaf Neural Networks equip graph structures with a cellular sheaf: a geometric structure which assigns local vector spaces (stalks) and a linear learnable restriction/transport maps to nodes and edges, yielding an edge-aware inductive bias that handles heterophily and limits oversmoothing. However, common Neural Sheaf Diffusion implementations rely on SVD-based sheaf normalization and dense per-edge restriction maps, which scale with stalk dimension, require frequent Laplacian rebuilds, and yield brittle gradients. To address these limitations, we introduce Polynomial Neural Sheaf Diffusion (PolyNSD), a new sheaf diffusion approach whose propagation operator is a degree-K polynomial in a normalised sheaf Laplacian, evaluated via a stable three-term recurrence on a spectrally rescaled operator. This provides an explicit K-hop receptive field in a single layer (independently of the stalk dimension), with a trainable spectral response obtained as a convex mixture of K+1 orthogonal polynomial basis responses. PolyNSD enforces stability via convex mixtures, spectral rescaling, and residual/gated paths, reaching new state-of-the-art results on both homophilic and heterophilic benchmarks, inverting the Neural Sheaf Diffusion trend by obtaining these results with just diagonal restriction maps, decoupling performance from large stalk dimension, while reducing runtime and memory requirements.

---

## 52. NetDeTox: Adversarial and Efficient Evasion of Hardware-Security GNNs via RL-LLM Orchestration

**论文链接:** [http://arxiv.org/abs/2512.00119v1](http://arxiv.org/abs/2512.00119v1)

**作者:** Zeng Wang, Minghao Shao, Akashdeep Saha, Ramesh Karri, Johann Knechtel, Muhammad Shafique, Ozgur Sinanoglu

**发布时间:** 2025-11-27

### GPT解析

### 总结

本文提出了一种名为NetDeTox的自动化端到端框架，通过结合大型语言模型和强化学习，有效降低了图神经网络在硬件安全应用中的脆弱性，实现了更高效的对抗性网表重写。

### 背景

图神经网络(GNNs)通过学习网表图中的结构基序在硬件安全领域展现出潜力，但这种对基序的依赖使其容易受到对抗性网表重写的影响，即使小规模编辑也可能误导GNN预测。现有对抗性方法存在较高设计开销问题。

### 目的

开发一个自动化框架，能够系统化地编排大型语言模型与强化学习，实现有针对性的局部网表重写，同时降低设计开销。

### 方法

NetDeTox框架利用强化学习代理识别对GNN推理至关重要的网表组件，大型语言模型制定保留功能性的基序多样化重写计划，并通过RL与LLM间的迭代反馈优化对抗性重写，限制设计开销。

### 主要发现

与AttackGNN相比，NetDeTox用更少的重写显著降低了各安全方案的面积开销：GNN-RE降低54.50%，GNN4IP降低25.44%，OMLA降低41.04%。对于GNN4IP，NetDeTox甚至能优化原始基准的面积，特别是大电路，证明了其实用性和可扩展性。

### 结论

NetDeTox框架有效解决了GNN在硬件安全中的脆弱性问题，通过结合LLM和RL实现了高效且低开销的对抗性网表重写，具有实际应用价值。

### 翻译

图神经网络(GNNs)已通过从网表图中学习结构基序在硬件安全方面显示出潜力。然而，这种对基序的依赖使GNN容易受到对抗性网表重写的影响；即使是小规模的编辑也可能误导GNN预测。现有的对抗性方法，从综合配方扰动到门变换，都带来了较高的设计开销。我们提出了NetDeTox，一个自动化端到端框架，以系统化方式编排大型语言模型(LLMs)与强化学习(RL)，实现有针对性的局部重写。RL代理识别对基于GNN的推理至关重要的网表组件，而LLM制定重写计划以多样化保留功能性的基序。RL和LLM阶段之间的迭代反馈完善对抗性重写以限制开销。与最先进的AttackGNN相比，NetDeTox能用更少的重写和显著降低的面积开销来降低所有安全方案的有效性（分别降低GNN-RE的54.50%，GNN4IP的25.44%和OMLA的41.04%）。对于GNN4IP，我们的方法甚至可以优化/减少原始基准的面积，特别是对于较大的电路，证明了NetDeTox的实用性和可扩展性。


### 论文摘要

Graph neural networks (GNNs) have shown promise in hardware security by learning structural motifs from netlist graphs. However, this reliance on motifs makes GNNs vulnerable to adversarial netlist rewrites; even small-scale edits can mislead GNN predictions. Existing adversarial approaches, ranging from synthesis-recipe perturbations to gate transformations, come with high design overheads. We present NetDeTox, an automated end-to-end framework that orchestrates large language models (LLMs) with reinforcement learning (RL) in a systematic manner, enabling focused local rewriting. The RL agent identifies netlist components critical for GNN-based reasoning, while the LLM devises rewriting plans to diversify motifs that preserve functionality. Iterative feedback between the RL and LLM stages refines adversarial rewritings to limit overheads. Compared to the SOTA work AttackGNN, NetDeTox successfully degrades the effectiveness of all security schemes with fewer rewrites and substantially lower area overheads (reductions of 54.50% for GNN-RE, 25.44% for GNN4IP, and 41.04% for OMLA, respectively). For GNN4IP, ours can even optimize/reduce the original benchmarks' area, in particular for larger circuits, demonstrating the practicality and scalability of NetDeTox.

---

## 53. HeartFormer: Semantic-Aware Dual-Structure Transformers for 3D Four-Chamber Cardiac Point Cloud Reconstruction

**论文链接:** [http://arxiv.org/abs/2512.00264v1](http://arxiv.org/abs/2512.00264v1)

**作者:** Zhengda Ma, Abhirup Banerjee

**发布时间:** 2025-11-29

### GPT解析

### 总结

本研究提出了首个基于点云表示的几何深度学习框架，用于从动态MRI数据进行3D四腔心脏重建。研究团队开发了名为HeartFormer的新方法，并构建了大型数据集HeartCompv1，实验表明该方法性能优于现有技术。

### 背景

传统动态MRI通常仅提供心脏的2D切片图像，这限制了在健康和病理条件下对心脏形态和生理机制的全面理解。

### 目的

克服传统动态MRI的局限性，实现从2D切片到3D完整心脏模型的重建，提供更全面的心脏形态和生理信息。

### 方法

提出HeartFormer，一个多类别点云补全网络，包含两个关键组件：语义感知双结构Transformer网络(SA-DSTNet)生成初始粗略点云；语义感知几何特征细化Transformer网络(SA-GFRTNet)逐步细化输出，生成高保真和几何一致的重建结果。

### 主要发现

构建了HeartCompv1，首个包含17,000个高分辨率3D多类别心脏网格和点云的大型公开数据集。在HeartCompv1和UK Biobank上的跨域实验表明，HeartFormer实现了稳健、准确和可泛化的性能，持续超越最先进方法。

### 结论

HeartFormer通过创新的点云表示和多类别处理能力，为从动态MRI数据进行3D心脏重建提供了有效解决方案，克服了传统方法的局限性。

### 翻译

我们提出了首个基于点云表示的几何深度学习框架，用于从动态MRI数据进行3D四腔心脏重建。这项工作解决了传统动态MRI的一个长期局限性，传统动态MRI通常只提供心脏的2D切片图像，从而限制了对健康和病理条件下心脏形态和生理机制的全面理解。为此，我们提出了HeartFormer，一个新颖的多类别点云补全网络，将传统的单类别点云补全扩展到多类别。HeartFormer包含两个关键组件：语义感知双结构Transformer网络(SA-DSTNet)和语义感知几何特征细化Transformer网络(SA-GFRTNet)。SA-DSTNet生成包含全局几何特征和子结构几何特征的初始粗略点云。在这些语义几何表示的指导下，SA-GFRTNet逐步细化粗略输出，有效利用全局和子结构几何先验，生成高保真和几何一致的重建结果。我们进一步构建了HeartCompv1，这是第一个公开可用的包含17,000个高分辨率3D多类别心脏网格和点云的大型数据集，为这一新兴研究方向建立了通用基准。在HeartCompv1和UK Biobank上的广泛跨域实验表明，HeartFormer实现了稳健、准确和可泛化的性能，持续超越最先进(SOTA)方法。代码和数据集将在接受后发布于：https://github.com/10Darren/HeartFormer。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决从2D心脏磁共振成像(cine MRI)数据重建完整准确的3D四腔心脏结构的问题。这个问题在现实中非常重要，因为传统cine MRI只提供2D切片图像，限制了医生对心脏形态和生理机制的全面理解；而高保真的3D心脏表示对定量生物标志物分析、病理可视化和患者特异性心脏力学模拟至关重要，有助于心脏病诊断和治疗。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者基于点云表示的灵活性，设计出HeartFormer框架，将传统单类别点云完成扩展到多类别完成。他们借鉴了点云完成网络的coarse-to-fine生成策略，Transformer-based set translation技术，以及统计形状模型在医学图像分析中的应用。作者特别关注了如何处理MRI切片间的错位问题，以及如何同时建模全局心脏结构和局部子结构特征，从而实现更准确的心脏重建。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是结合全局心脏解剖上下文和局部子结构先验知识，在语义条件引导下自适应细化子结构表示，实现高保真心脏重建。整体流程包括：1)接收2D MRI数据并转换为稀疏点云；2)使用SA-DSTNet生成初始粗略点云，包含全局和子结构特征；3)通过SA-GFRTNet逐步细化几何细节和语义分布；4)使用语义感知Chamfer距离损失进行监督；5)最终输出高质量3D心脏点云和网格。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)首个基于点云表示的几何深度学习框架用于3D心脏重建；2)HeartFormer模型结合SA-DSTNet和SA-GFRTNet，同时处理全局和子结构特征；3)构建了HeartCompv1首个大规模多类别心脏点云完成数据集；4)设计了语义感知损失函数。相比之前工作，不同之处在于：从单类别扩展到多类别重建，有效处理了MRI切片错位问题，实现了解剖一致的高保真表面重建，同时建模全局与局部结构。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'HeartFormer首次引入基于点云的几何深度学习框架，通过语义感知双结构Transformer和首个大规模多类别心脏数据集，实现了从稀疏错位的2D MRI到高保真3D四腔心脏结构的准确重建，显著提升了心脏形态分析的准确性和临床实用性。'}


### 论文摘要

We present the first geometric deep learning framework based on point cloud representation for 3D four-chamber cardiac reconstruction from cine MRI data. This work addresses a long-standing limitation in conventional cine MRI, which typically provides only 2D slice images of the heart, thereby restricting a comprehensive understanding of cardiac morphology and physiological mechanisms in both healthy and pathological conditions. To overcome this, we propose \textbf{HeartFormer}, a novel point cloud completion network that extends traditional single-class point cloud completion to the multi-class. HeartFormer consists of two key components: a Semantic-Aware Dual-Structure Transformer Network (SA-DSTNet) and a Semantic-Aware Geometry Feature Refinement Transformer Network (SA-GFRTNet). SA-DSTNet generates an initial coarse point cloud with both global geometry features and substructure geometry features. Guided by these semantic-geometry representations, SA-GFRTNet progressively refines the coarse output, effectively leveraging both global and substructure geometric priors to produce high-fidelity and geometrically consistent reconstructions. We further construct \textbf{HeartCompv1}, the first publicly available large-scale dataset with 17,000 high-resolution 3D multi-class cardiac meshes and point-clouds, to establish a general benchmark for this emerging research direction. Extensive cross-domain experiments on HeartCompv1 and UK Biobank demonstrate that HeartFormer achieves robust, accurate, and generalizable performance, consistently surpassing state-of-the-art (SOTA) methods. Code and dataset will be released upon acceptance at: https://github.com/10Darren/HeartFormer.

---

## 54. Cross-Geometry Transfer Learning in Fast Electromagnetic Shower Simulation

**论文链接:** [http://arxiv.org/abs/2512.00187v1](http://arxiv.org/abs/2512.00187v1)

**作者:** Frank Gaede, Gregor Kasieczka, Lorenzo Valente

**发布时间:** 2025-11-28

### GPT解析

### 总结

本文提出了一种用于生成性量热器模拟模型的迁移学习框架，使模型能够高效适应不同的探测器几何结构，解决了传统方法计算成本高和现有机器学习方法需要针对每种几何结构完全重新训练的问题。

### 背景

精确的粒子簇射模拟仍是高能物理学中的关键计算瓶颈。传统蒙特卡洛方法（如Geant4）计算成本过高，而现有机器学习代理模型依赖于特定探测器几何结构，每次设计变更或替代探测器都需要完全重新训练。

### 目的

开发一种迁移学习框架，使生成性量热器模拟模型能够高效适应多种几何结构，提高数据效率并减少计算负担。

### 方法

使用点云表示方法，在国际大型探测器上进行预训练，使模型能够处理新配置而无需为每种几何结构重新对簇射进行体素化。采用参数高效微调，仅更新17%的模型参数，并使用仅100个目标域样本进行迁移学习。

### 主要发现

在CaloChallenge数据集上，迁移学习方法比从头开始训练实现了44%的Wasserstein距离几何平均改进。参数高效的微调方法在仅更新17%模型参数的情况下实现了具有竞争力的性能。

### 结论

该研究为粒子簇射发展的适应机制提供了见解，为点云方法在量热器模拟中的未来进展建立了基准。

### 翻译

精确的粒子簇射模拟仍然是高能物理学中的关键计算瓶颈。传统的蒙特卡洛方法，如Geant4，计算成本过高，而现有的机器学习代理模型依赖于特定的探测器几何结构，每次设计变更或替代探测器都需要完全重新训练。我们提出了一个用于生成性量热器模拟模型的迁移学习框架，使模型能够高效适应各种几何结构。使用点云表示并在国际大型探测器上进行预训练，我们的方法能够处理新配置而无需为每种几何结构重新对簇射进行体素化。在CaloChallenge数据集上，仅使用100个目标域样本进行迁移学习，比从头开始训练在Wasserstein距离的几何平均上实现了44%的改进。仅更新17%模型参数的偏置调整参数高效微调实现了具有竞争力的性能。我们的分析为粒子簇射发展的适应机制提供了见解，为点云方法在量热器模拟中的未来进展建立了基准。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决粒子探测器模拟中的几何依赖性问题。当探测器设计发生变化时，现有的机器学习模拟模型需要完全重新训练，这严重限制了高能物理研究中快速模拟方法的实用性。这个问题非常重要，因为随着高亮度大型强子对撞机等实验的推进，实验数据量将空前增长，而传统模拟方法(如Geant4)计算成本极高，无法满足需求。机器学习替代模型虽然能加速模拟，但它们通常与特定几何形状绑定，每次设计变更都需要重新训练，大大降低了效率。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者借鉴了自然语言处理和计算机视觉中的基础模型范式，探索单探测器预训练在点云表示下的可行性。他们参考了MetaHEP(通过元学习进行跨探测器迁移)和OmniJet(统一多任务架构)的工作，但指出这些方法存在局限性。作者基于CaloClouds网络架构(包含PointWise Net扩散模型和ShowerFlow归一化流)进行改进，特别关注参数高效的微调策略，如BitFit、Top2微调和LoRA，以减少适应新几何形状所需的计算资源。这种方法旨在解决单探测器预训练与多探测器预训练的平衡问题，以及点云表示与固定网格的比较问题。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是使用点云表示簇射数据，使其能够适应不同的探测器几何形状；在一个探测器(ILD)上预训练模型，然后通过迁移学习将其适应到新的探测器几何形状(CaloChallenge)；使用参数高效的微调策略来减少计算成本。整体流程：1)使用点云表示电磁簇射(空间坐标和能量沉积)；2)在ILD探测器上的光子簇射数据预训练CaloClouds模型；3)对于新的探测器几何形状，使用迁移学习策略(完整微调或参数高效微调)来适应预训练模型；4)评估模型在新几何形状上的性能，使用Wasserstein距离和KL散度等指标。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点：1)单探测器预训练：与需要多个探测器数据集预训练的CaloDiT-2不同，本文专注于单探测器预训练，这在只有单个探测器可用时更具实用性；2)点云表示：相比固定网格，点云提供了几何灵活性，直接匹配了量热器簇射的稀疏性质；3)参数高效微调：研究了BitFit、Top2微调和LoRA等方法，发现BitFit在更新仅17%参数的情况下实现了接近完整微调的性能；4)跨几何形状迁移：成功将模型从平面ILD几何形状迁移到圆柱形CaloChallenge几何形状，处理了多种域偏移(几何形状、能量分布、粒子类型)。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '这篇论文展示了单探测器预训练的点云模型可以通过迁移学习高效适应不同的探测器几何形状，在低数据量情况下显著提高性能，并提出了参数高效的微调策略来减少计算成本。'}


### 论文摘要

Accurate particle shower simulation remains a critical computational bottleneck for high-energy physics. Traditional Monte Carlo methods, such as Geant4, are computationally prohibitive, while existing machine learning surrogates are tied to specific detector geometries and require complete retraining for each design change or alternative detector. We present a transfer learning framework for generative calorimeter simulation models that enables adaptation across diverse geometries with high data efficiency. Using point cloud representations and pre-training on the International Large Detector detector, our approach handles new configurations without re-voxelizing showers for each geometry. On the CaloChallenge dataset, transfer learning with only 100 target-domain samples achieves a $44\%$ improvement on the geometric mean of Wasserstein distance over training from scratch. Parameter-efficient fine-tuning with bias-only adaptation achieves competitive performance while updating only $17\%$ of model parameters. Our analysis provides insight into adaptation mechanisms for particle shower development, establishing a baseline for future progress of point cloud approaches in calorimeter simulation.

---

## 55. Rethinking Multimodal Point Cloud Completion: A Completion-by-Correction Perspective

**论文链接:** [http://arxiv.org/abs/2511.12170v2](http://arxiv.org/abs/2511.12170v2)

**作者:** Wang Luo, Di Wu, Hengyuan Na, Yinlin Zhu, Miao Hu, Guocong Quan

**发布时间:** 2025-11-15

**备注:** Accepted by AAAI 2026

### GPT解析

### 总结

该论文针对点云补全任务提出了一种新的'Completion-by-Correction'范式，并开发了PGNet多阶段框架。该框架通过从预训练的图像到3D模型生成拓扑完整的形状先验，然后在特征空间进行校正，使其与部分观测对齐。实验表明，该方法在ShapeNetViPC数据集上优于最先进的基线，Chamfer距离降低23.5%，F-score提高7.1%。

### 背景

点云补全旨在从部分观测中重建完整的3D形状，但由于严重的遮挡和缺失几何结构，这是一个具有挑战性的问题。尽管最近的多模态技术利用互补的RGB图像来补偿缺失的几何结构，但大多数方法仍然遵循'Completion-by-Inpainting'范式，即从融合的潜在特征中合成缺失结构。

### 目的

解决现有方法中由于几何和语义约束有限而导致的结构不一致性和拓扑伪影问题，提出一种更鲁棒的点云补全范式和框架。

### 方法

提出'Completion-by-Correction'范式，从预训练的图像到3D模型生成拓扑完整的形状先验，然后在特征空间进行校正以与部分观测对齐。基于此范式，开发了PGNet多阶段框架，进行双特征编码以生成先验，合成结构对齐的粗略支架，并通过分层校正逐步细化几何细节。

### 主要发现

实验表明，'Completion-by-Inpainting'范式往往由于有限的几何和语义约束而导致结构不一致性和拓扑伪影。而'Completion-by-Correction'范式将补全从无约束的合成转变为引导的细化，能够实现结构一致且与观测对齐的重建。

### 结论

PGNet在ShapeNetViPC数据集上的实验证明了其优于最先进基线的性能，平均Chamfer距离降低了23.5%，F-score提高了7.1%。这表明将补全任务重新构思为校正问题可以显著提高点云补全的质量。

### 翻译

点云补全旨在从部分观测中重建完整的3D形状，由于严重的遮挡和缺失几何结构，这是一个具有挑战性的问题。尽管最近的多模态技术利用互补的RGB图像来补偿缺失的几何结构，但大多数方法仍然遵循'Completion-by-Inpainting'范式，从融合的潜在特征中合成缺失结构。我们经验性地表明，由于有限的几何和语义约束，这种范式通常会导致结构不一致性和拓扑伪影。为了解决这个问题，我们重新思考了任务，并提出了一种更鲁棒的范式，称为'Completion-by-Correction'，它从预训练的图像到3D模型生成拓扑完整的形状先验，并在特征空间进行校正以使其与部分观测对齐。这种范式将补全从无约束的合成转变为引导的细化，实现了结构一致且与观测对齐的重建。基于此范式，我们引入了PGNet，一个多阶段框架，它进行双特征编码以生成先验，合成结构对齐的粗略支架，并通过分层校正逐步细化几何细节。在ShapeNetViPC数据集上的实验表明，PGNet在平均Chamfer距离(-23.5%)和F-score(+7.1%)方面优于最先进的基线。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决多模态点云补全问题，即如何从不完整的3D点云观测中重建完整的3D形状。这个问题在现实中非常重要，因为点云是自动驾驶、增强现实和机器人等AI应用中的基本3D表示，但传感器捕获的点云常常因遮挡、光线反射和有限分辨率而稀疏和不完整，这种不完整性会严重影响下游任务的性能。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先识别出当前多模态点云补全方法（Completion-by-Inpainting）的局限性，即从不完整的融合表示中合成缺失几何结构会导致结构不一致和拓扑伪影。因此，作者提出了Completion-by-Correction范式，从一个预训练的图像到3D模型生成的完整形状先验开始，然后在特征空间进行修正。作者借鉴了现有的图像到3D模型、DGCNN用于局部特征聚合、Transformer架构进行特征编码和注意力机制，以及点云处理中的远点采样和逆距离加权等技术。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是Completion-by-Correction范式，将点云补全任务从不受约束的合成转变为对完整形状先验的引导修正。整体实现流程包括三个阶段：1）修正双特征编码：并行编码部分点云和生成先验，使用Salient Transformer和Grounding Transformer生成修正的观测感知表示；2）基于种子的生成：从修正特征生成结构化种子，创建完整但几何对齐的支架；3）分层基于修正的细化：使用多个Grounded Refinement Block逐步改进几何保真度，通过双源特征关联和结构感知上采样增强细节。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1）提出Completion-by-Correction范式，重新定义多模态点云补全任务；2）设计PGNet三阶段框架实现这一范式；3）在ShapeNetViPC数据集上实现最先进性能，平均Chamfer Distance降低23.5%，F-score提高7.1%。与之前工作的不同在于：之前方法从不完整表示中合成缺失几何，而本文从完整形状先验开始，通过修正确保结构一致性；本文引入了Salient Transformer和Grounding Transformer处理特征，并设计了双源特征关联和结构感知上采样机制。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '本文提出了Completion-by-Correction范式和PGNet框架，通过修正预训练图像到3D模型生成的完整形状先验，显著提高了多模态点云补全的结构一致性和几何保真度。'}


### 论文摘要

Point cloud completion aims to reconstruct complete 3D shapes from partial observations, which is a challenging problem due to severe occlusions and missing geometry. Despite recent advances in multimodal techniques that leverage complementary RGB images to compensate for missing geometry, most methods still follow a Completion-by-Inpainting paradigm, synthesizing missing structures from fused latent features. We empirically show that this paradigm often results in structural inconsistencies and topological artifacts due to limited geometric and semantic constraints. To address this, we rethink the task and propose a more robust paradigm, termed Completion-by-Correction, which begins with a topologically complete shape prior generated by a pretrained image-to-3D model and performs feature-space correction to align it with the partial observation. This paradigm shifts completion from unconstrained synthesis to guided refinement, enabling structurally consistent and observation-aligned reconstruction. Building upon this paradigm, we introduce PGNet, a multi-stage framework that conducts dual-feature encoding to ground the generative prior, synthesizes a coarse yet structurally aligned scaffold, and progressively refines geometric details via hierarchical correction. Experiments on the ShapeNetViPC dataset demonstrate the superiority of PGNet over state-of-the-art baselines in terms of average Chamfer Distance (-23.5%) and F-score (+7.1%).

---

## 56. LiNeXt: Revisiting LiDAR Completion with Efficient Non-Diffusion Architectures

**论文链接:** [http://arxiv.org/abs/2511.10209v2](http://arxiv.org/abs/2511.10209v2)

**作者:** Wenzhe He, Xiaojun Chen, Ruiqi Wang, Ruihui Li, Huilong Pi, Jiapeng Zhang, Zhuo Tang, Kenli Li

**发布时间:** 2025-11-13

**备注:** 18 pages, 13 figures, Accepted to AAAI 2026

### GPT解析

### 总结

论文提出了一种名为LiNeXt的轻量级非扩散网络，用于快速准确的3D LiDAR点云场景补全，解决了传统扩散模型计算开销大的问题。

### 背景

3D LiDAR场景补全是自动驾驶车辆感知系统的基本组成部分，先前方法主要采用扩散模型进行高保真重建。

### 目的

开发一种轻量级、非扩散网络，实现快速准确的点云补全，解决传统扩散模型多步迭代采样带来的计算开销问题。

### 方法

LiNeXt采用噪声到粗略(N2C)模块进行单次去噪，避免多步迭代；使用精炼模块进一步提高结构完整性；提出距离感知选择重复策略处理LiDAR点云的距离依赖空间分布特性。

### 主要发现

在SemanticKITTI数据集上，LiNeXt实现推理速度提升199.8倍，Chamfer Distance减少50.7%，仅使用LiDiff参数量的6.1%。

### 结论

LiNeXt在实时场景补全方面展现出卓越的效率和有效性，适合实际应用。

### 翻译

从点云进行3D LiDAR场景补全是自动驾驶车辆感知系统的基本组成部分。先前方法主要采用扩散模型进行高保真重建。然而，它们的多步迭代采样带来了巨大的计算开销，限制了其实时应用性。为解决这一问题，我们提出LiNeXt——一种为快速准确的点云补全而优化的轻量级非扩散网络。具体而言，LiNeXt首先应用噪声到粗略(N2C)模块对输入的噪声点云进行单次去噪，从而避免了基于扩散方法的多步迭代采样。然后，精炼模块接收来自N2C模块的粗略点云及其中间特征，执行更精确的精炼，进一步提高结构完整性。此外，我们观察到LiDAR点云表现出距离依赖的空间分布，在近距离处密集采样，远距离处稀疏采样。因此，我们提出距离感知选择重复策略，生成更均匀分布的噪声点云。在SemanticKITTI数据集上，LiNeXt实现推理速度提升199.8倍，Chamfer Distance减少50.7%，且仅使用LiDiff参数量的6.1%。这些结果证明了LiNeXt在实时场景补全方面的优越效率和有效性。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决3D LiDAR场景补全中的效率问题。当前主流的扩散模型方法虽然能实现高保真度重建，但由于多步迭代采样导致计算开销大，限制了实时应用。在自动驾驶领域，实时感知系统对安全导航至关重要，而LiDAR点云的补全是提高感知系统鲁棒性的关键组件。高效的场景补全对于自动驾驶等实时应用场景具有重要意义。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先分析了现有扩散模型的局限性：计算开销大、优化挑战复杂。他们决定放弃使用扩散模型，转而采用轻量级网络直接重建场景。设计思路包括：引入距离感知的点复制策略实现均匀分布；设计两阶段重建流程（粗略重建+精细处理）；开发高效的特征提取与融合机制。作者借鉴了点云表示优势、PointNet、稀疏卷积等现有技术，但针对点云处理特点设计了专门的Cross-Point Attention和Multi-Scale Sparse Convolution机制。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是通过轻量级非扩散架构实现高效、准确的3D LiDAR场景补全，避免扩散模型的迭代过程，采用距离感知的点复制策略，并通过两阶段重建流程平衡效率和质量。整体流程：1)输入不完整点云，应用距离感知策略复制点并添加噪声；2)Noise to Coarse模块通过多尺度稀疏卷积提取特征，层次生成种子点，然后重建粗略场景；3)Refine模块对粗略结果进行精细处理，生成最终完整点云。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)Distance-aware Selected Repeat策略根据距离调整复制因子，实现均匀空间分布；2)Noise to Coarse模块直接从噪声点云重建粗略结构，避免扩散模型迭代过程；3)Cross-Point Attention机制动态对齐特征并融合互补信息；4)Multi-Scale Sparse Convolution模块高效提取多尺度特征。相比之前工作，LiNeXt放弃了计算密集的扩散模型，实现了199.8倍推理加速，模型大小仅为LiDiff的6.1%，同时显著提高了重建质量。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'LiNeXt通过创新的轻量级非扩散架构和距离感知的点处理策略，实现了比现有扩散模型快199.8倍、模型小94%的LiDAR场景补全，同时显著提高了重建质量，为自动驾驶等实时应用提供了高效解决方案。'}


### 论文摘要

3D LiDAR scene completion from point clouds is a fundamental component of perception systems in autonomous vehicles. Previous methods have predominantly employed diffusion models for high-fidelity reconstruction. However, their multi-step iterative sampling incurs significant computational overhead, limiting its real-time applicability. To address this, we propose LiNeXt-a lightweight, non-diffusion network optimized for rapid and accurate point cloud completion. Specifically, LiNeXt first applies the Noise-to-Coarse (N2C) Module to denoise the input noisy point cloud in a single pass, thereby obviating the multi-step iterative sampling of diffusion-based methods. The Refine Module then takes the coarse point cloud and its intermediate features from the N2C Module to perform more precise refinement, further enhancing structural completeness. Furthermore, we observe that LiDAR point clouds exhibit a distance-dependent spatial distribution, being densely sampled at proximal ranges and sparsely sampled at distal ranges. Accordingly, we propose the Distance-aware Selected Repeat strategy to generate a more uniformly distributed noisy point cloud. On the SemanticKITTI dataset, LiNeXt achieves a 199.8x speedup in inference, reduces Chamfer Distance by 50.7%, and uses only 6.1% of the parameters compared with LiDiff. These results demonstrate the superior efficiency and effectiveness of LiNeXt for real-time scene completion.

---

## 57. ZIP-RC: Zero-overhead Inference-time Prediction of Reward and Cost for Adaptive and Interpretable Generation

**论文链接:** [http://arxiv.org/abs/2512.01457v1](http://arxiv.org/abs/2512.01457v1)

**作者:** Rohin Manvi, Joey Hong, Tim Seyde, Maxime Labonne, Mathias Lechner, Sergey Levine

**发布时间:** 2025-12-01

**备注:** Code coming soon

### GPT解析

### 总结

ZIP-RC是一种自适应推理方法，通过零开销的推理时预测奖励和成本，使大型语言模型具备实时内省能力，从而在数学推理任务中提高准确度并降低成本。

### 背景

大型语言模型在推理方面表现出色，但缺乏关键的内省能力，包括预测自身成功和所需计算的能力。现有测试时扩展方法不考虑边际收益，导致成本增加且缺乏置信度信号。

### 目的

开发一种使大型语言模型具备零开销推理时预测奖励和成本能力的方法，以实现自适应推理。

### 方法

ZIP-RC在每个token处重用保留或未使用的logits，在与下一个token预测相同的正向传递中输出最终奖励和剩余长度的联合分布。使用这种联合分布计算采样效用，并通过元动作最大化这种效用。

### 主要发现

在混合难度数学基准测试中，ZIP-RC在相等或更低平均成本的情况下，比多数投票提高准确度高达12%，并在质量、计算和延迟之间绘制出平滑的帕累托前沿。

### 结论

通过提供实时奖励-成本内省，ZIP-RC能够实现自适应、高效的推理。

### 翻译

大型语言模型在推理方面表现出色，但缺乏关键的内省方面，包括预测自身成功和实现成功所需的计算。人类使用实时内省来决定投入多少努力、何时多次尝试、何时停止以及何时表示成功或失败。没有这种能力，大型语言模型难以做出智能的元认知决策。测试时扩展方法如Best-of-N通过使用固定数量的样本而不考虑任何生成点每个样本的边际收益，从而提高了成本和延迟，缺乏置信度信号可能导致误导、无法适当地升级到更好的工具以及降低可信度。学习验证器或奖励模型可以提供置信度估计，但不能实现自适应推理，并且通过需要额外的模型或正向传递增加了大量成本。我们提出了ZIP-RC，一种自适应推理方法，使模型能够以零开销的方式在推理时预测奖励和成本。在每个token处，ZIP-RC在与下一个token预测相同的正向传递中重用保留或未使用的logits，输出最终奖励和剩余长度的联合分布——不需要额外的模型、架构更改或推理开销。使用这种完整的联合分布来计算采样效用，这是如果完成生成，一组样本的预期最大奖励、总计算和延迟的线性组合。在推理过程中，我们通过确定要继续的token前缀或从哪里开始采样的元动作来最大化这种效用。在混合难度数学基准测试中，ZIP-RC在相等或更低平均成本的情况下，比多数投票提高准确度高达12%，并在质量、计算和延迟之间绘制出平滑的帕累托前沿。通过提供实时奖励-成本内省，ZIP-RC能够实现自适应、高效的推理。


### 论文摘要

Large language models excel at reasoning but lack key aspects of introspection, including anticipating their own success and the computation required to achieve it. Humans use real-time introspection to decide how much effort to invest, when to make multiple attempts, when to stop, and when to signal success or failure. Without this, LLMs struggle to make intelligent meta-cognition decisions. Test-time scaling methods like Best-of-N drive up cost and latency by using a fixed budget of samples regardless of the marginal benefit of each one at any point in generation, and the absence of confidence signals can mislead people, prevent appropriate escalation to better tools, and undermine trustworthiness. Learned verifiers or reward models can provide confidence estimates, but do not enable adaptive inference and add substantial cost by requiring extra models or forward passes. We present ZIP-RC, an adaptive inference method that equips models with zero-overhead inference-time predictions of reward and cost. At every token, ZIP-RC reuses reserved or unused logits in the same forward pass as next-token prediction to output a joint distribution over final reward and remaining length -- no extra models, architecture change, or inference overhead. This full joint distribution is used to compute a sampling utility which is the linear combination of the expected maximum reward, total compute, and latency of set of samples if generated to completion. During inference, we maximize this utility with meta-actions that determine which prefix of tokens to continue or initiate sampling from. On mixed-difficulty mathematical benchmarks, ZIP-RC improves accuracy by up to 12% over majority voting at equal or lower average cost, and traces smooth Pareto frontiers between quality, compute, and latency. By providing real-time reward-cost introspection, ZIP-RC enables adaptive, efficient reasoning.

---

## 58. Know Thyself by Knowing Others: Learning Neuron Identity from Population Context

**论文链接:** [http://arxiv.org/abs/2512.01199v1](http://arxiv.org/abs/2512.01199v1)

**作者:** Vinam Arora, Divyansha Lachi, Ian J. Knight, Mehdi Azabou, Blake Richards, Cole L. Hurwitz, Josh Siegle, Eva L. Dyer

**发布时间:** 2025-12-01

**备注:** Accepted at Neurips 2025

### GPT解析

### 总结

本文介绍了NuCLR，一种用于学习神经活动表征的自监督框架，能够区分不同神经元身份，并在多个数据集上实现了细胞类型和脑区解码的最先进性能。

### 背景

神经元以依赖于其细胞类型、连接性和所在脑区的方式处理信息，但从神经活动中推断这些因素仍然是一个重大挑战。

### 目的

构建通用表征，能够解析神经元的身份信息，引入NuCLR框架来学习神经活动的表征，以区分一个神经元与其他神经元。

### 方法

NuCLR结合同一神经元在不同时间和不同刺激下观察到的视图，使用对比目标拉拢这些表征；构建时空transformer以排列等变方式整合活动，而不假设固定神经元排序；在多个电生理学和钙成像数据集上进行评估。

### 主要发现

NuCLR表征上的线性解码在细胞类型和脑区解码任务上达到新最先进水平；展示了对未见动物的强大零样本泛化能力；增加预训练动物数量持续提高下游性能；学习到的表征标签高效，只需少量标记样本即可实现有竞争力性能。

### 结论

大型、多样化的神经数据集使模型能够恢复关于神经元身份的信息，这些信息可以跨动物泛化。

### 翻译

神经元以依赖于其细胞类型、连接性和所在脑区的方式处理信息。然而，从神经活动中推断这些因素仍然是一个重大挑战。为了构建允许解析神经元身份信息的通用表征，我们引入了NuCLR，这是一种自监督框架，旨在学习神经活动的表征，以区分一个神经元与其他神经元。NuCLR将同一神经元在不同时间和不同刺激下观察到的视图结合起来，并使用对比目标来拉拢这些表征。为了在不假设任何固定神经元排序的情况下捕捉群体上下文，我们构建了一个时空transformer，以排列等变方式整合活动。在多个电生理学和钙成像数据集上，在NuCLR表征之上的线性解码评估在细胞类型和脑区解码任务上均达到了新的最先进水平，并展示了对未见过的动物的强大零样本泛化能力。我们首次对神经元级别表征学习进行了系统的扩展分析，显示增加预训练期间使用的动物数量持续提高下游性能。学习到的表征也是标签高效的，只需要一小部分标记样本就能实现有竞争力的性能。这些结果突显了大型、多样化的神经数据集如何使模型能够恢复跨动物泛化的神经元身份信息。代码可在https://github.com/nerdslab/nuclr获取。


### 论文摘要

Neurons process information in ways that depend on their cell type, connectivity, and the brain region in which they are embedded. However, inferring these factors from neural activity remains a significant challenge. To build general-purpose representations that allow for resolving information about a neuron's identity, we introduce NuCLR, a self-supervised framework that aims to learn representations of neural activity that allow for differentiating one neuron from the rest. NuCLR brings together views of the same neuron observed at different times and across different stimuli and uses a contrastive objective to pull these representations together. To capture population context without assuming any fixed neuron ordering, we build a spatiotemporal transformer that integrates activity in a permutation-equivariant manner. Across multiple electrophysiology and calcium imaging datasets, a linear decoding evaluation on top of NuCLR representations achieves a new state-of-the-art for both cell type and brain region decoding tasks, and demonstrates strong zero-shot generalization to unseen animals. We present the first systematic scaling analysis for neuron-level representation learning, showing that increasing the number of animals used during pretraining consistently improves downstream performance. The learned representations are also label-efficient, requiring only a small fraction of labeled samples to achieve competitive performance. These results highlight how large, diverse neural datasets enable models to recover information about neuron identity that generalize across animals. Code is available at https://github.com/nerdslab/nuclr.

---

## 59. Learning to Reconstruct Temperature Field from Sparse Observations with Implicit Physics Priors

**论文链接:** [http://arxiv.org/abs/2512.01196v1](http://arxiv.org/abs/2512.01196v1)

**作者:** Shihang Li, Zhiqiang Gong, Weien Zhou, Yue Gao, Wen Yao

**发布时间:** 2025-12-01

### GPT解析

### 总结

本文提出了一种名为IPTR的隐式物理引导的温度场重建框架，通过引入参考仿真数据作为先验，结合双物理嵌入模块，有效解决了热源系统温度场重建中的高成本测量和分布变化问题，实现了比现有方法更准确的重建结果和更强的泛化能力。

### 背景

热源系统温度场的准确重建对于电子设备和航空航天结构等工程应用中的热监控和可靠性评估至关重要。

### 目的

解决测量采集成本高和温度场在不同条件下存在显著分布变化的问题，开发具有强大泛化能力的温度场重建模型。

### 方法

提出IPTR框架，引入参考仿真数据作为先验，设计双物理嵌入模块，包含隐式物理引导分支(使用交叉注意力提取参考数据中的潜在物理)和辅助编码分支(基于傅里叶层捕获目标观测的空间特征)，最后将融合表示解码以重建完整温度场。

### 主要发现

在单条件、多条件和少样本设置下的实验表明，IPTR方法始终优于现有方法，实现了最先进的重建精度和强大的泛化能力。

### 结论

IPTR框架通过有效整合参考仿真数据和目标观测信息，解决了现有方法未能充分利用参考数据的问题，为热源系统温度场重建提供了新的有效解决方案。

### 翻译

热源系统温度场的准确重建对于电子设备和航空航天结构等工程应用中的热监控和可靠性评估至关重要。然而，测量采集成本高以及不同条件下温度场的显著分布变化，给开发具有强大泛化能力的重建模型带来了重大挑战。现有的基于深度神经网络的方法通常仅基于目标稀疏测量将温度场重建表述为一对一回归问题，而没有有效利用隐式编码热力学知识的参考仿真数据。为解决这一局限，我们提出IPTR，一种隐式物理引导的温度场重建框架，引入来自参考仿真的稀疏监测温度场对作为先验，以丰富物理理解。为整合参考和目标信息，我们设计了一个包含两个互补分支的双物理嵌入模块：一个使用交叉注意力从参考数据中提取潜在物理的隐式物理引导分支，和一个基于傅里叶层捕获目标观测空间特征的辅助编码分支。融合表示随后被解码以重建完整温度场。在单条件、多条件和少样本设置下的广泛实验表明，IPTR始终优于现有方法，实现了最先进的重建精度和强大的泛化能力。


### 论文摘要

Accurate reconstruction of temperature field of heat-source systems (TFR-HSS) is crucial for thermal monitoring and reliability assessment in engineering applications such as electronic devices and aerospace structures. However, the high cost of measurement acquisition and the substantial distributional shifts in temperature field across varying conditions present significant challenges for developing reconstruction models with robust generalization capabilities. Existing DNNs-based methods typically formulate TFR-HSS as a one-to-one regression problem based solely on target sparse measurements, without effectively leveraging reference simulation data that implicitly encode thermal knowledge. To address this limitation, we propose IPTR, an implicit physics-guided temperature field reconstruction framework that introduces sparse monitoring-temperature field pair from reference simulations as priors to enrich physical understanding. To integrate both reference and target information, we design a dual physics embedding module consisting of two complementary branches: an implicit physics-guided branch employing cross-attention to distill latent physics from the reference data, and an auxiliary encoding branch based on Fourier layers to capture the spatial characteristics of the target observation. The fused representation is then decoded to reconstruct the full temperature field. Extensive experiments under single-condition, multi-condition, and few-shot settings demonstrate that IPTR consistently outperforms existing methods, achieving state-of-the-art reconstruction accuracy and strong generalization capability.

---

## 60. LGDC: Latent Graph Diffusion via Spectrum-Preserving Coarsening

**论文链接:** [http://arxiv.org/abs/2512.01190v1](http://arxiv.org/abs/2512.01190v1)

**作者:** Nagham Osman, Keyue Jiang, Davide Buffelli, Xiaowen Dong, Laura Toni

**发布时间:** 2025-12-01

### GPT解析

### 总结

本研究分析了图生成领域的两种主要范式：自回归模型和一次性模型（如扩散模型），揭示了它们在捕捉局部结构和全局模式方面的权衡。基于此，作者提出了LGDC混合框架，结合两种方法的优势，通过谱保持粗化-反粗化在图和潜在空间间进行映射，利用扩散生成潜在图后再扩展恢复细节，有效捕捉局部和全局特性。

### 背景

图生成是各科学领域的关键任务，现有方法主要分为自回归模型（迭代扩展图）和一次性模型（如扩散模型，一次性生成完整图）。

### 目的

分析现有图生成方法的优缺点，并提出一种能够结合自回归模型和扩散模型优势的混合框架，以同时捕捉图的局部和全局特性。

### 方法

提出LGDC（通过谱保持粗化的潜在图扩散）混合框架，采用谱保持的粗化-反粗化技术在图和潜在空间间进行双向映射，在潜在空间中利用扩散模型高效生成潜在图，再通过扩展恢复细节。

### 主要发现

自回归模型擅长捕捉细粒度的局部结构（如度和聚类特性），而一次性模型擅长建模全局模式（如谱分布）；LGDC在局部结构数据集（Tree）上匹配自回归模型性能，在全局结构数据集（Planar, Community-20）上匹配扩散模型性能。

### 结论

混合生成方法能够有效结合自回归模型和扩散模型的优势，同时捕捉图的局部和全局特性，提高生成效率。

### 翻译

图生成是跨科学领域的关键任务。现有方法大致分为两类：自回归模型，迭代扩展图；以及一次性模型，如扩散模型，一次性生成完整图。在本工作中，我们分析了这两种范式，并揭示了一个关键权衡：自回归模型在捕捉细粒度局部结构（如度和聚类特性）方面突出，而一次性模型在建模全局模式（如谱分布）方面表现出色。基于此，我们提出了LGDC（通过谱保持粗化的潜在图扩散），这是一个结合两种方法优势的混合框架。LGDC采用谱保持的粗化-反粗化在图和潜在空间之间进行双向映射，在潜在空间中扩散高效生成潜在图，然后通过扩展恢复细节。这种设计同时捕捉局部和全局特性，提高了效率。实验上，LGDC在局部结构数据集（Tree）上匹配自回归模型，在全局结构数据集（Planar, Community-20）上匹配扩散模型，验证了混合生成的好处。


### 论文摘要

Graph generation is a critical task across scientific domains. Existing methods fall broadly into two categories: autoregressive models, which iteratively expand graphs, and one-shot models, such as diffusion, which generate the full graph at once. In this work, we provide an analysis of these two paradigms and reveal a key trade-off: autoregressive models stand out in capturing fine-grained local structures, such as degree and clustering properties, whereas one-shot models excel at modeling global patterns, such as spectral distributions. Building on this, we propose LGDC (latent graph diffusion via spectrum-preserving coarsening), a hybrid framework that combines strengths of both approaches. LGDC employs a spectrum-preserving coarsening-decoarsening to bidirectionally map between graphs and a latent space, where diffusion efficiently generates latent graphs before expansion restores detail. This design captures both local and global properties with improved efficiency. Empirically, LGDC matches autoregressive models on locally structured datasets (Tree) and diffusion models on globally structured ones (Planar, Community-20), validating the benefits of hybrid generation.

---

## 61. H-Zero: Cross-Humanoid Locomotion Pretraining Enables Few-shot Novel Embodiment Transfer

**论文链接:** [http://arxiv.org/abs/2512.00971v1](http://arxiv.org/abs/2512.00971v1)

**作者:** Yunfeng Lin, Minghuan Liu, Yufei Xue, Ming Zhou, Yong Yu, Jiangmiao Pang, Weinan Zhang

**发布时间:** 2025-11-30

**备注:** in submission, under review

### GPT解析

### 总结

该研究提出了H-Zero，一种跨人形机器人运动预训练管道，能够学习通用的人形基础策略，实现零样本和少样本迁移到新型机器人。

### 背景

人形机器人技术快速发展，但现有控制器针对特定机器人设计定制，需要大量调整奖励函数、物理参数和训练超参数，开发通用控制器仍是重大挑战。

### 目的

开发一种跨人形运动预训练方法，解决现有控制器需要针对每个机器人进行大量调整的问题，实现通用人形运动策略的学习。

### 方法

引入H-Zero预训练管道，在有限的人形机器人集合上进行预训练，然后迁移到新型机器人，只需进行少量微调。

### 主要发现

预训练策略在仿真中能在未见过的机器人上保持高达81%的完整episode持续时间，能够在30分钟微调后迁移到未见的人形机器人和直立四足机器人上。

### 结论

H-Zero预训练管道能够学习通用的人形运动策略，减少对新机器人控制器的大量调整工作，实现了跨平台的运动控制能力。

### 翻译

人形机器人技术的快速发展加剧了对强大且适应性强的控制器的需求，以实现跨不同平台的稳定高效运动。然而，开发此类控制器仍然是一个重大挑战，因为现有解决方案针对特定机器人设计定制，需要为每种形态大量调整奖励函数、物理参数和训练超参数。为应对这一挑战，我们引入了H-Zero，一种跨人形运动预训练管道，学习可泛化的人形基础策略。我们证明，在有限形态集合上进行预训练，能够实现零样本和少样本迁移到新型人形机器人，只需最小限度微调。评估显示，预训练策略在仿真中能在未见过的机器人上保持高达81%的完整episode持续时间，同时能够在30分钟微调后迁移到未见的人形机器人和直立四足机器人上。


### 论文摘要

The rapid advancement of humanoid robotics has intensified the need for robust and adaptable controllers to enable stable and efficient locomotion across diverse platforms. However, developing such controllers remains a significant challenge because existing solutions are tailored to specific robot designs, requiring extensive tuning of reward functions, physical parameters, and training hyperparameters for each embodiment. To address this challenge, we introduce H-Zero, a cross-humanoid locomotion pretraining pipeline that learns a generalizable humanoid base policy. We show that pretraining on a limited set of embodiments enables zero-shot and few-shot transfer to novel humanoid robots with minimal fine-tuning. Evaluations show that the pretrained policy maintains up to 81% of the full episode duration on unseen robots in simulation while enabling few-shot transfer to unseen humanoids and upright quadrupeds within 30 minutes of fine-tuning.

---

## 62. Goal-Driven Reward by Video Diffusion Models for Reinforcement Learning

**论文链接:** [http://arxiv.org/abs/2512.00961v1](http://arxiv.org/abs/2512.00961v1)

**作者:** Qi Wang, Mian Wu, Yuyang Zhang, Mingqi Yuan, Wenyao Zhang, Haoxiang You, Yunbo Wang, Xin Jin, Xiaokang Yang, Wenjun Zeng

**发布时间:** 2025-11-30

### GPT解析

### 总结

该研究利用预训练视频扩散模型中的世界知识为目标驱动的强化学习提供奖励信号，避免了对奖励函数的手动设计。

### 背景

强化学习在多个领域取得了显著成功，但通常依赖精心设计的程序化奖励函数来指导智能体行为，这种设计具有挑战性且可能无法很好地泛化到不同任务。

### 目的

解决传统奖励函数设计的局限性，利用预训练视频扩散模型中的丰富知识为强化学习智能体提供目标驱动的奖励信号，无需手动设计奖励函数。

### 方法

利用预训练在大型视频数据集上的视频扩散模型作为奖励函数；对于视频级奖励，在特定领域数据集上微调模型并使用视频编码器评估轨迹与目标视频的对齐；对于帧级目标，使用CLIP识别最相关帧作为目标状态，并使用学习的前向-后向表示作为帧级奖励。

### 主要发现

在各种Meta-World任务上的实验证明了该方法的有效性，能够促进更连贯和目标驱动的轨迹。

### 结论

通过利用预训练视频扩散模型的知识，可以有效地为目标驱动的强化学习提供奖励信号，无需手动设计奖励函数。

### 翻译

强化学习(RL)在各个领域取得了显著成功，但它通常依赖于精心设计的程序化奖励函数来指导智能体行为。设计此类奖励函数可能具有挑战性，并且可能无法很好地泛化到不同的任务中。为了解决这一局限性，我们利用预训练视频扩散模型中包含的丰富世界知识，为强化学习智能体提供目标驱动的奖励信号，而无需对奖励进行特殊设计。我们的核心思想是利用在大型视频数据集上预训练的现成视频扩散模型，作为视频级和帧级目标的信息丰富的奖励函数。对于视频级奖励，我们首先在特定领域数据集上微调预训练的视频扩散模型，然后使用其视频编码器评估智能体轨迹的潜在表示与生成的目标视频之间的对齐。为实现更细粒度的目标达成，我们使用CLIP从生成的视频中识别最相关的帧作为目标状态。然后，我们使用学习的前向-后向表示（表示从给定状态-动作对访问目标状态的概率）作为帧级奖励，促进更连贯和目标驱动的轨迹。在各种Meta-World任务上的实验证明了我们方法的有效性。


### 论文摘要

Reinforcement Learning (RL) has achieved remarkable success in various domains, yet it often relies on carefully designed programmatic reward functions to guide agent behavior. Designing such reward functions can be challenging and may not generalize well across different tasks. To address this limitation, we leverage the rich world knowledge contained in pretrained video diffusion models to provide goal-driven reward signals for RL agents without ad-hoc design of reward. Our key idea is to exploit off-the-shelf video diffusion models pretrained on large-scale video datasets as informative reward functions in terms of video-level and frame-level goals. For video-level rewards, we first finetune a pretrained video diffusion model on domain-specific datasets and then employ its video encoder to evaluate the alignment between the latent representations of agent's trajectories and the generated goal videos. To enable more fine-grained goal-achievement, we derive a frame-level goal by identifying the most relevant frame from the generated video using CLIP, which serves as the goal state. We then employ a learned forward-backward representation that represents the probability of visiting the goal state from a given state-action pair as frame-level reward, promoting more coherent and goal-driven trajectories. Experiments on various Meta-World tasks demonstrate the effectiveness of our approach.

---

## 63. Fine-tuning of lightweight large language models for sentiment classification on heterogeneous financial textual data

**论文链接:** [http://arxiv.org/abs/2512.00946v1](http://arxiv.org/abs/2512.00946v1)

**作者:** Alvaro Paredes Amorin, Andre Python, Christoph Weisser

**发布时间:** 2025-11-30

### GPT解析

### 总结

研究表明轻量级开源大型语言模型在金融文本分析中能够以较低成本取得与复杂模型相当的效果，即使在数据有限的情况下也能有效泛化情感理解。

### 背景

大型语言模型在金融市场分析中扮演着重要角色，能从复杂异构的文本数据源中提取信号，但它们的性能依赖于大量计算资源和专有数据集，这些资源成本高昂且受限，许多研究人员和从业者无法访问。

### 目的

研究轻量级开源大型语言模型能否从不同大小、来源、格式和语言的金融数据集中泛化情感理解，以反映现实情况。

### 方法

比较基准金融NLP模型FinBERT与三个开源轻量级LLMs（DeepSeek-LLM 7B、Llama3 8B Instruct和Qwen3 8B）在五个公开数据集上的表现：FinancialPhraseBank、Financial Question Answering、Gold News Sentiment、Twitter Sentiment和Chinese Finance Sentiment。

### 主要发现

LLMs，特别是Qwen3 8B和Llama3 8B，在大多数场景中表现最佳，即使仅使用5%的可用训练数据也能取得良好效果，这些发现在零样本和少样本学习场景中均成立。

### 结论

轻量级、开源的大型语言模型是一种经济有效的选择，它们可以在异构文本数据上实现具有竞争力的性能，即使只在广泛标注语料库的有限子集上进行训练也能取得良好效果。

### 翻译

大型语言模型（LLMs）通过捕捉来自复杂且异构的文本数据源（如推文、新闻文章、报告和微博）中的信号，在金融市场分析中扮演着越来越重要的角色。然而，它们的性能依赖于大量计算资源和专有数据集，这些资源成本高昂、受限，因此许多研究人员和从业者无法访问。为反映现实情况，我们研究了轻量级开源LLMs（较小且公开可用的模型，旨在有限计算资源下运行）从不同大小、来源、格式和语言的金融数据集中泛化情感理解的能力。我们在五个公开可用数据集上比较了基准金融自然语言处理（NLP）模型FinBERT和三个开源轻量级LLMs：DeepSeek-LLM 7B、Llama3 8B Instruct和Qwen3 8B。我们发现，LLMs，特别是Qwen3 8B和Llama3 8B，在大多数场景中表现最佳，即使仅使用5%的可用训练数据。这些发现在零样本和少样本学习场景中均成立。我们的研究结果表明，轻量级、开源的大型语言模型（LLMs）是一种经济有效的选择，因为即使只在通常被认为必要的广泛标注语料库的有限子集上进行训练，它们也能在异构文本数据上实现具有竞争力的性能。


### 论文摘要

Large language models (LLMs) play an increasingly important role in finan- cial markets analysis by capturing signals from complex and heterogeneous textual data sources, such as tweets, news articles, reports, and microblogs. However, their performance is dependent on large computational resources and proprietary datasets, which are costly, restricted, and therefore inacces- sible to many researchers and practitioners. To reflect realistic situations we investigate the ability of lightweight open-source LLMs - smaller and publicly available models designed to operate with limited computational resources - to generalize sentiment understanding from financial datasets of varying sizes, sources, formats, and languages. We compare the benchmark finance natural language processing (NLP) model, FinBERT, and three open-source lightweight LLMs, DeepSeek-LLM 7B, Llama3 8B Instruct, and Qwen3 8B on five publicly available datasets: FinancialPhraseBank, Financial Question Answering, Gold News Sentiment, Twitter Sentiment and Chinese Finance Sentiment. We find that LLMs, specially Qwen3 8B and Llama3 8B, perform best in most scenarios, even from using only 5% of the available training data. These results hold in zero-shot and few-shot learning scenarios. Our findings indicate that lightweight, open-source large language models (LLMs) consti- tute a cost-effective option, as they can achieve competitive performance on heterogeneous textual data even when trained on only a limited subset of the extensive annotated corpora that are typically deemed necessary.

---

## 64. Exchange-Correlation Functionals in 2D Materials: Applications, Challenges, and Limitations

**论文链接:** [http://arxiv.org/abs/2512.00921v1](http://arxiv.org/abs/2512.00921v1)

**作者:** Ahsan Javed, Mahvish Shaheen, Muhammad Shahbaz, M. Sufyan Ramzan, Rafi Ullah, Wei Jiang

**发布时间:** 2025-11-30

### GPT解析

### 总结

该综述探讨了在密度泛函理论框架下，交换关联泛函在预测二维材料关键性质（结构、光电子、磁性和热性质）中的关键作用，评估了高级计算方法的准确性，并讨论了当前限制和未来发展方向。

### 背景

二维材料的快速发展改变了现代纳米科学，其性质与块体材料有根本不同。随着实验发现的加速，对可靠计算技术的需求日益增长。

### 目的

在密度泛函理论框架下，探讨交换关联泛函在预测二维材料关键性质中的关键作用，评估不同计算方法的准确性。

### 方法

评估了meta-GGA、混合泛函和多体微扰理论（如GW和Bethe-Salpeter方程）等高级方法，讨论了机器学习在提高计算效率方面的作用。

### 主要发现

传统泛函在描述量子限制、各向异性屏蔽和范德华相互作用方面存在挑战；高级方法在捕捉电子结构和激子效应方面提高了准确性；不同二维材料家族中泛函具有非普适性。

### 结论

当前计算方法存在限制，需要发展新策略来推进交换关联泛函及超越，以实现二维材料的实际设计和应用。

### 翻译

二维材料的快速发展重塑了现代纳米科学，提供了与块体材料性质根本不同的特性。随着实验发现的加速，对可靠计算技术的需求日益增长。在密度泛函理论框架下，本综述探讨了交换关联泛函在预测关键材料性质（如结构、光电子、磁性和热性质）中的关键作用。我们考察了量子限制、各向异性和屏蔽以及范德华相互作用带来的挑战，这些是传统泛函常常无法描述的。评估了meta-GGA、混合泛函和多体微扰理论（如GW和Bethe-Salpeter方程）等高级方法在提高捕捉电子结构和激子效应准确性方面的表现。我们进一步讨论了不同二维材料家族中泛函的非普适性以及机器学习在提高计算效率方面的新兴作用。最后，本综述概述了当前限制和新兴策略，为推进交换关联泛函及超越提供了路线图，以实现二维材料的实际设计和应用。


### 论文摘要

The rapid development of two-dimensional (2D) materials has reshaped modern nanoscience, offering properties that differ fundamentally from their bulk counterparts. As experimental discovery accelerates, the need for reliable computational techniques has become increas- ingly important. Within the framework of density functional theory, this review explores the critical role of exchange-correlation functionals in predicting key material properties such as structural, optoelectronic, magnetic, and thermal. We examine the challenges posed by quantum confinement, anisotropic screening, and van der Waals interactions, which conventional functionals often fail to describe. Advanced approaches, including meta-GGA, hybrid functionals, and many-body perturbation theory (e.g., GW and Bethe-Salpeter equation), are assessed for their improved accuracy in capturing electronic structure and excitonic effects. We further discuss the non-universality of functionals across different 2D material families and the emerging role of machine learning to enhance computational efficiency. Finally, the review outlines current limitations and emerging strategies, providing a roadmap for advancing exchange-correlation functionals and beyond, to enable the practical design and application of 2D materials.

---

## 65. 论文ID: 2512.00849v1

**错误:** 无法读取论文信息 - [Errno 2] No such file or directory: 'storage/result/2512.00849v1.json'

---

## 66. Towards Precision Protein-Ligand Affinity Prediction Benchmark: A Complete and Modification-Aware DAVIS Dataset

**论文链接:** [http://arxiv.org/abs/2512.00708v1](http://arxiv.org/abs/2512.00708v1)

**作者:** Ming-Hsiu Wu, Ziqian Xie, Shuiwang Ji, Degui Zhi

**发布时间:** 2025-11-30

### GPT解析

### 总结

本研究通过扩充DAVIS数据集，创建了一个包含4,032种激酶-配体对且具有修饰感知能力的数据集，并提出了三种基准测试设置来评估AI模型在蛋白质修饰存在下的性能。研究发现基于对接的模型在零样本设置中泛化能力更好，而无对接模型在野生型蛋白质上过度拟合，但在少量修改示例微调后表现改善。

### 背景

AI在科学领域的进步为关键的药物发现任务（如蛋白质-配体结合亲和力预测）提供了能力。然而，当前模型过度拟合于现有简化的数据集，这些数据集不能代表具有修饰的自然存在和生物相关蛋白质。

### 目的

创建一个完整且具有修饰感知能力的DAVIS数据集版本，并基于此提出三种基准测试设置，以评估模型在蛋白质修饰存在下的鲁棒性，从而促进能够更好泛化到蛋白质修饰的模型发展。

### 方法

通过整合涉及替换、插入、删除和磷酸化事件的4,032种激酶-配体对，扩充了广泛使用的DAVIS数据集。基于这个新数据集，提出了三种基准测试设置：增强数据集预测、野生型到修饰泛化和少样本修饰泛化，并对无对接和基于对接的方法进行了广泛评估。

### 主要发现

基于对接的模型在零样本设置中泛化能力更好。无对接模型倾向于过度拟合野生型蛋白质，难以处理未见过的修饰，但在少量修改示例上进行微调后显示出显著改进。

### 结论

策划的数据集和基准为开发更好泛化到蛋白质修饰的模型提供了有价值的基础，最终推动药物发现中的精准医疗。相关基准可在https://github.com/ZhiGroup/DAVIS-complete获取。

### 翻译

AI在科学领域的进步为关键的药物发现任务（如蛋白质-配体结合亲和力预测）提供了能力。然而，当前模型过度拟合于现有简化的数据集，这些数据集不能代表具有修饰的自然存在和生物相关蛋白质。在本工作中，我们通过整合涉及替换、插入、删除和磷酸化事件的4,032种激酶-配体对，策划了广泛使用的DAVIS数据集的一个完整且具有修饰感知能力的版本。这个丰富的数据集使预测模型能够在生物现实条件下进行基准测试。基于这个新数据集，我们提出了三种基准设置-增强数据集预测、野生型到修饰泛化和少样本修饰泛化，旨在评估模型在存在蛋白质修饰时的鲁棒性。通过对无对接和基于对接方法的广泛评估，我们发现基于对接的模型在零样本设置中泛化能力更好。相比之下，无对接模型倾向于过度拟合野生型蛋白质，难以处理未见过的修饰，但在少量修改示例上进行微调后显示出显著改进。我们期望策划的数据集和基准为开发更好泛化到蛋白质修饰的模型提供了有价值的基础，最终推动药物发现中的精准医疗。基准测试可在以下网址获取：https://github.com/ZhiGroup/DAVIS-complete


### 论文摘要

Advancements in AI for science unlocks capabilities for critical drug discovery tasks such as protein-ligand binding affinity prediction. However, current models overfit to existing oversimplified datasets that does not represent naturally occurring and biologically relevant proteins with modifications. In this work, we curate a complete and modification-aware version of the widely used DAVIS dataset by incorporating 4,032 kinase-ligand pairs involving substitutions, insertions, deletions, and phosphorylation events. This enriched dataset enables benchmarking of predictive models under biologically realistic conditions. Based on this new dataset, we propose three benchmark settings-Augmented Dataset Prediction, Wild-Type to Modification Generalization, and Few-Shot Modification Generalization-designed to assess model robustness in the presence of protein modifications. Through extensive evaluation of both docking-free and docking-based methods, we find that docking-based model generalize better in zero-shot settings. In contrast, docking-free models tend to overfit to wild-type proteins and struggle with unseen modifications but show notable improvement when fine-tuned on a small set of modified examples. We anticipate that the curated dataset and benchmarks offer a valuable foundation for developing models that better generalize to protein modifications, ultimately advancing precision medicine in drug discovery. The benchmark is available at: https://github.com/ZhiGroup/DAVIS-complete

---

## 67. Pre-Generating Multi-Difficulty PDE Data for Few-Shot Neural PDE Solvers

**论文链接:** [http://arxiv.org/abs/2512.00564v1](http://arxiv.org/abs/2512.00564v1)

**作者:** Naman Choudhary, Vedant Singh, Ameet Talwalkar, Nicholas Matthew Boffi, Mikhail Khodak, Tanya Marwah

**发布时间:** 2025-11-29

**备注:** 10 Pages, 11 Figures

### GPT解析

### 总结

该研究解决了学习型偏微分方程(PDE)求解器中训练数据生成成本高昂的问题，通过系统性地研究难度转移，发现结合低和高难度数据可以显著减少计算成本。

### 背景

学习型PDE求解器的主要成本通常来自使用经典求解器生成训练数据，而非学习模型本身。问题沿着几何复杂度和雷诺数等难度轴变得对经典求解器更困难，也更有可能从神经加速中受益。

### 目的

解决'先有鸡还是先有蛋'的挑战，研究2D不可压缩Navier-Stokes方程上的难度转移，系统性地改变几何形状、物理特性及它们的组合来调整任务复杂度。

### 方法

类似于预训练基础模型的方法，通过经典求解器生成大量低和中等难度的例子并纳入训练集，从而从更少的样本中学习高难度物理问题。

### 主要发现

通过结合低和高难度数据，可以比仅使用高难度例子少花费8.9倍的计算量来预生成数据集，同时达到相同的误差水平。跨难度级别合理分配经典求解器计算资源与总体分配量同样重要。

### 结论

对预生成的PDE数据进行有原则的整理可以为神经求解器带来显著性能提升，研究代码已公开在GitHub上。

### 翻译

学习型偏微分方程(PDE)求解器的一个关键方面是，主要成本通常来自于使用经典求解器生成训练数据，而不是学习模型本身。另一个方面是存在明确的难度轴——例如更复杂的几何形状和更高的雷诺数——沿着这些轴，问题对经典求解器变得更困难，因此更有可能从神经加速中受益。为解决这一循环挑战，我们研究了2D不可压缩Navier-Stokes上的难度转移，系统性地改变几何形状（障碍物的数量和位置）、物理特性（雷诺数）以及它们的组合来调整任务复杂度。类似于如何通过计算预训练基础模型并改进其在下游任务上的性能，我们发现通过经典求解器（类比预生成）许多低和中等难度的例子并将它们包含在训练集中，可以从更少的样本中学习高难度物理。此外，我们证明通过结合低和高难度数据，我们可以比仅使用高难度例子少花费8.9倍的计算量来预生成数据集以达到相同的误差水平。我们的结果表明，我们如何跨难度级别分配经典求解器的计算资源与总体分配多少计算资源同样重要，并表明通过有原则地整理预生成的PDE数据可以为神经求解器带来显著收益。我们的代码可在https://github.com/Naman-Choudhary-AI-ML/pregenerating-pde获取。


### 论文摘要

A key aspect of learned partial differential equation (PDE) solvers is that the main cost often comes from generating training data with classical solvers rather than learning the model itself. Another is that there are clear axes of difficulty--e.g., more complex geometries and higher Reynolds numbers--along which problems become (1) harder for classical solvers and thus (2) more likely to benefit from neural speedups. Towards addressing this chicken-and-egg challenge, we study difficulty transfer on 2D incompressible Navier-Stokes, systematically varying task complexity along geometry (number and placement of obstacles), physics (Reynolds number), and their combination. Similar to how it is possible to spend compute to pre-train foundation models and improve their performance on downstream tasks, we find that by classically solving (analogously pre-generating) many low and medium difficulty examples and including them in the training set, it is possible to learn high-difficulty physics from far fewer samples. Furthermore, we show that by combining low and high difficulty data, we can spend 8.9x less compute on pre-generating a dataset to achieve the same error as using only high difficulty examples. Our results highlight that how we allocate classical-solver compute across difficulty levels is as important as how much we allocate overall, and suggest substantial gains from principled curation of pre-generated PDE data for neural solvers. Our code is available at https://github.com/Naman-Choudhary-AI-ML/pregenerating-pde

---

## 68. PEOAT: Personalization-Guided Evolutionary Question Assembly for One-Shot Adaptive Testing

**论文链接:** [http://arxiv.org/abs/2512.00439v1](http://arxiv.org/abs/2512.00439v1)

**作者:** Xiaoshan Yu, Ziwei Huang, Shangshang Yang, Ziwen Wang, Haiping Ma, Xingyi Zhang

**发布时间:** 2025-11-29

**备注:** AAAI-2026, 9 pages

### GPT解析

### 总结

该研究提出了一种面向一次性自适应测试(One-shot Adaptive Testing, OAT)的个性化引导进化问题组装框架(PEOAT)，旨在解决传统计算机化自适应测试(CAT)在实际应用中的局限性。

### 背景

随着智能教育的快速发展，计算机化自适应测试(CAT)结合教育心理学和深度学习技术受到关注。CAT通过自适应选择最适合的项目来评估考生能力，但其实时性和顺序性在大规模评估和高敏感度领域存在交互成本高和噪音干扰等问题。

### 目的

介绍一次性自适应测试(OAT)这一新任务，旨在为每位考生一次性选择一组最优项目；提出PEOAT框架，从组合优化角度解决OAT问题。

### 方法

设计个性化感知初始化策略，整合考生能力和练习难度差异，使用多策略采样构建初始种群；提出认知增强进化框架，包含模式保留交叉和认知引导变异；引入多样性感知环境选择机制以保持多样性而不损害适应度。

### 主要发现

PEOAT在两个数据集上通过大量实验验证了其有效性，案例研究发现了有价值的见解。

### 结论

PEOAT框架有效解决了传统CAT在实际应用中的局限性，为一次性自适应测试提供了新的解决方案。

### 翻译

随着智能教育的快速发展，计算机化自适应测试(CAT)通过整合教育心理学与深度学习技术日益受到关注。与传统纸笔测试不同，CAT旨在通过在评估过程中自适应选择最合适的项目来高效准确地评估考生能力。然而，其实时性和顺序性在实际场景中存在局限性，特别是在大规模评估中交互成本高，或在心理评估等敏感领域需要最小化噪音和干扰。这些挑战限制了传统CAT方法在时间敏感或资源受限环境中的适用性。为此，我们首先介绍了一种名为一次性自适应测试(OAT)的新任务，旨在为每位考生一次性选择一组最优项目。同时，我们提出了PEOAT，即从组合优化角度面向一次性自适应测试的个性化引导进化问题组装框架。具体而言，我们首先设计了一个个性化感知初始化策略，整合考生能力和练习难度差异，使用多策略采样构建多样化和信息丰富的初始种群。在此基础上，我们提出了一个包含模式保留交叉和认知引导变异的认知增强进化框架，通过信息信号实现高效探索。为了保持多样性而不损害适应度，我们进一步引入了多样性感知环境选择机制。PEOAT的有效性在两个数据集上通过大量实验得到验证，辅以案例研究发现了有价值的见解。


### 论文摘要

With the rapid advancement of intelligent education, Computerized Adaptive Testing (CAT) has attracted increasing attention by integrating educational psychology with deep learning technologies. Unlike traditional paper-and-pencil testing, CAT aims to efficiently and accurately assess examinee abilities by adaptively selecting the most suitable items during the assessment process. However, its real-time and sequential nature presents limitations in practical scenarios, particularly in large-scale assessments where interaction costs are high, or in sensitive domains such as psychological evaluations where minimizing noise and interference is essential. These challenges constrain the applicability of conventional CAT methods in time-sensitive or resourceconstrained environments. To this end, we first introduce a novel task called one-shot adaptive testing (OAT), which aims to select a fixed set of optimal items for each test-taker in a one-time selection. Meanwhile, we propose PEOAT, a Personalization-guided Evolutionary question assembly framework for One-shot Adaptive Testing from the perspective of combinatorial optimization. Specifically, we began by designing a personalization-aware initialization strategy that integrates differences between examinee ability and exercise difficulty, using multi-strategy sampling to construct a diverse and informative initial population. Building on this, we proposed a cognitive-enhanced evolutionary framework incorporating schema-preserving crossover and cognitively guided mutation to enable efficient exploration through informative signals. To maintain diversity without compromising fitness, we further introduced a diversity-aware environmental selection mechanism. The effectiveness of PEOAT is validated through extensive experiments on two datasets, complemented by case studies that uncovered valuable insights.

---

## 69. Adaptive prediction theory combining offline and online learning

**论文链接:** [http://arxiv.org/abs/2512.00342v1](http://arxiv.org/abs/2512.00342v1)

**作者:** Haizheng Li, Lei Guo

**发布时间:** 2025-11-29

### GPT解析

### 总结

本文研究了一种结合离线学习和在线适应的两阶段学习框架在一类非线性随机动力系统中的预测性能。在离线阶段，作者建立了强相关性和分布偏移数据下的泛化误差上界；在在线阶段，提出了meta-LMS算法处理参数漂移问题。研究表明，这种两阶段框架比单纯的离线或在线方法具有更好的预测性能，并通过理论和实验进行了验证。

### 背景

现实世界的智能系统通常结合离线学习和在线适应来处理高度相关和非平稳的系统数据或信号，但这种现象在文献中很少得到理论研究的关注。

### 目的

对结合离线和在线算法的两阶段学习框架在一类非线性随机动力系统中的预测性能进行理论研究。

### 方法

离线学习阶段：在强相关性和分布偏移的数据集上建立近似非线性最小二乘估计的泛化误差上界，使用KL散度量化分布差异；在线适应阶段：提出meta-LMS预测算法解决现实系统中可能的不确定参数漂移问题。

### 主要发现

整合离线学习和在线适应的两阶段框架比单纯的离线或在线方法具有更好的预测性能。

### 结论

该研究提供了理论保证和实证研究结果，证实了所提出框架的有效性。

### 翻译

现实世界的智能系统通常通过结合离线学习和在线适应来操作，使用高度相关和非平稳的系统数据或信号，然而，这在文献中很少被理论研究。本文对一类非线性随机动力系统的两阶段学习框架的预测性能进行了理论研究，该框架结合了离线和在线算法。对于离线学习阶段，我们在具有强相关性和分布偏移的一般数据集上，为近似非线性最小二乘估计建立了泛化误差的上界，利用Kullback-Leibler散度来量化分布差异。对于在线适应阶段，我们基于离线训练的模型，通过提出一种meta-LMS预测算法，解决了现实世界目标系统中可能存在的不确定参数漂移问题。这种整合离线学习和在线适应的两阶段框架，与单纯的离线或在线方法相比，表现出优越的预测性能。本文提供了理论保证和实证研究。


### 论文摘要

Real-world intelligence systems usually operate by combining offline learning and online adaptation with highly correlated and non-stationary system data or signals, which, however, has rarely been investigated theoretically in the literature. This paper initiates a theoretical investigation on the prediction performance of a two-stage learning framework combining offline and online algorithms for a class of nonlinear stochastic dynamical systems. For the offline-learning phase, we establish an upper bound on the generalization error for approximate nonlinear-least-squares estimation under general datasets with strong correlation and distribution shift, leveraging the Kullback-Leibler divergence to quantify the distributional discrepancies. For the online-adaptation phase, we address, on the basis of the offline-trained model, the possible uncertain parameter drift in real-world target systems by proposing a meta-LMS prediction algorithm. This two-stage framework, integrating offline learning with online adaptation, demonstrates superior prediction performances compared with either purely offline or online methods. Both theoretical guarantees and empirical studies are provided.

---

## 70. Behavioral Indicators of Loneliness: Predicting University Students' Loneliness Scores from Smartphone Sensing Data

**论文链接:** [http://arxiv.org/abs/2512.00326v1](http://arxiv.org/abs/2512.00326v1)

**作者:** Qianjie Wu, Tianyi Zhang, Hong Jia, Simon D'Alfonso

**发布时间:** 2025-11-29

**DOI:** 10.1145/3714394.3756345

### GPT解析

### 总结

本研究探索了使用被动智能手机传感数据预测大学生孤独水平的方法，整合机器学习和大型语言模型开发通用和个性化模型，发现智能手机使用行为和位置移动是孤独感的关键预测因素。

### 背景

孤独感是大学生中关键的心理健康问题，但传统监测方法主要依赖回顾性自我报告，缺乏实时行为背景。

### 目的

探索使用被动智能手机传感数据预测孤独水平，解决现有方法在捕捉孤独感动态特性方面的局限性。

### 方法

整合智能手机传感与机器学习和大型语言模型开发通用和个性化模型；使用随机森林开发通用模型；利用大型语言模型采用一次性方法；在UCLA孤独感量表（简表）上进行评估。

### 主要发现

随机森林通用模型在期中和期末分别达到3.29和3.98（满分32）的平均绝对误差；智能手机屏幕使用和位置移动是关键预测因素；大型语言模型一次性方法比零样本推理减少高达42%的预测误差；个性化模型突出了屏幕使用、应用程序使用、电池和位置转换作为显著行为指标。

### 结论

智能手机传感数据在可扩展和可解释的孤独感检测方面具有潜力，对数字心理健康领域具有重要意义。

### 翻译

孤独感是大学生中一个关键的心理健康问题，然而传统的监测方法主要依赖回顾性自我报告，并且常常缺乏实时的行为背景。本研究探索了使用被动智能手机传感数据来预测孤独水平，解决了现有方法在捕捉其动态特性方面的局限性。我们将智能手机传感分别与机器学习和大型语言模型相结合，开发了通用模型和个性化模型。我们的随机森林通用模型在UCLA孤独感量表（简表）上期中达到3.29、期末达到3.98（满分32）的平均绝对误差，识别出智能手机屏幕使用和位置移动是关键预测因素。利用大型语言模型的一次性方法与零样本推理相比减少了高达42%的预测误差。个性化模型的一次性结果突出了屏幕使用、应用程序使用、电池和位置转换作为显著的行为指标。这些发现证明了智能手机传感数据在数字心理健康中可扩展和可解释的孤独感检测的潜力。


### 论文摘要

Loneliness is a critical mental health issue among university students, yet traditional monitoring methods rely primarily on retrospective self-reports and often lack real-time behavioral context. This study explores the use of passive smartphone sensing data to predict loneliness levels, addressing the limitations of existing approaches in capturing its dynamic nature. We integrate smartphone sensing with machine learning and large language models respectively to develop generalized and personalized models. Our Random Forest generalized models achieved mean absolute errors of 3.29 at midterm and 3.98 (out of 32) at the end of semester on the UCLA Loneliness Scale (short form), identifying smartphone screen usage and location mobility to be key predictors. The one-shot approach leveraging large language models reduced prediction errors by up to 42% compared to zero-shot inference. The one-shot results from personalized models highlighted screen usage, application usage, battery, and location transitions as salient behavioral indicators. These findings demonstrate the potential of smartphone sensing data for scalable and interpretable loneliness detection in digital mental health.

---

## 71. PORTAL: Controllable Landscape Generator for Continuous Optimization-Part I: Framework

**论文链接:** [http://arxiv.org/abs/2512.00288v1](http://arxiv.org/abs/2512.00288v1)

**作者:** Danial Yazdani, Mai Peng, Delaram Yazdani, Shima F. Yazdi, Mohammad Nabi Omidvar, Yuan Sun, Trung Thanh Nguyen, Changhe Li, Xiaodong Li

**发布时间:** 2025-11-29

**备注:** 15 pages, 1 figure

### GPT解析

### 总结

PORTAL是一个创新的优化基准测试生成器，解决了现有测试套件的局限性，提供了对优化景观的精细控制，支持各种研究、教育和测试需求。

### 背景

基准测试在优化研究中至关重要，但现有的连续优化测试套件存在局限性：经典集合是固定且僵硬的，而之前的生成器仅涵盖景观的狭窄家族，可变性和对细节的控制有限。

### 目的

引入PORTAL（优化研究、测试、分析和学习的平台），一个通用的基准测试生成器，提供对盆曲率、条件、变量相互作用和表面粗糙度的细粒度、独立控制。

### 方法

PORTAL采用分层设计，从单个组件到多组件景观的块状组合，具有可控的部分可分离性和不平衡的块贡献；提供对每个组件在每个维度和方向上的形状的精确控制；通过多种转换模式保持组件中心和局部二次结构；设计中和机制防止意外组件支配；系统化引入复杂景观特征如多模态性、不对称性和异构粗糙度。

### 主要发现

PORTAL支持通过隔离特定挑战和渐进式难度扩展来进行系统的算法分析；有助于创建多样化的数据集，用于元算法研究、定制的基准测试套件设计和交互式教育用途。

### 结论

PORTAL的完整Python和MATLAB源代码已在GitHub上公开，为优化研究提供了强大而灵活的工具。

### 翻译

Benchmarking is central to optimization research, yet existing test suites for continuous optimization remain limited: classical collections are fixed and rigid, while previous generators cover only narrow families of landscapes with restricted variability and control over details. This paper introduces PORTAL (Platform for Optimization Research, Testing, Analysis, and Learning), a general benchmark generator that provides fine-grained, independent control over basin curvature, conditioning, variable interactions, and surface ruggedness. PORTAL's layered design spans from individual components to block-wise compositions of multi-component landscapes with controllable partial separability and imbalanced block contributions. It offers precise control over the shape of each component in every dimension and direction, and supports diverse transformation patterns through both element-wise and coupling operators with compositional sequencing. All transformations preserve component centers and local quadratic structure, ensuring stability and interpretability. A principled neutralization mechanism prevents unintended component domination caused by exponent or scale disparities, which addresses a key limitation of prior landscape generators. On this foundation, transformations introduce complex landscape characteristics, such as multimodality, asymmetry, and heterogeneous ruggedness, in a controlled and systematic way. PORTAL enables systematic algorithm analysis by supporting both isolation of specific challenges and progressive difficulty scaling. It also facilitates the creation of diverse datasets for meta-algorithmic research, tailored benchmark suite design, and interactive educational use. The complete Python and MATLAB source code for PORTAL is publicly available at [https://github.com/EvoMindLab/PORTAL].


### 论文摘要

Benchmarking is central to optimization research, yet existing test suites for continuous optimization remain limited: classical collections are fixed and rigid, while previous generators cover only narrow families of landscapes with restricted variability and control over details. This paper introduces PORTAL (Platform for Optimization Research, Testing, Analysis, and Learning), a general benchmark generator that provides fine-grained, independent control over basin curvature, conditioning, variable interactions, and surface ruggedness. PORTAL's layered design spans from individual components to block-wise compositions of multi-component landscapes with controllable partial separability and imbalanced block contributions. It offers precise control over the shape of each component in every dimension and direction, and supports diverse transformation patterns through both element-wise and coupling operators with compositional sequencing. All transformations preserve component centers and local quadratic structure, ensuring stability and interpretability. A principled neutralization mechanism prevents unintended component domination caused by exponent or scale disparities, which addresses a key limitation of prior landscape generators. On this foundation, transformations introduce complex landscape characteristics, such as multimodality, asymmetry, and heterogeneous ruggedness, in a controlled and systematic way. PORTAL enables systematic algorithm analysis by supporting both isolation of specific challenges and progressive difficulty scaling. It also facilitates the creation of diverse datasets for meta-algorithmic research, tailored benchmark suite design, and interactive educational use. The complete Python and MATLAB source code for PORTAL is publicly available at [https://github.com/EvoMindLab/PORTAL].

---

## 72. Orion-Bix: Bi-Axial Attention for Tabular In-Context Learning

**论文链接:** [http://arxiv.org/abs/2512.00181v1](http://arxiv.org/abs/2512.00181v1)

**作者:** Mohamed Bouadi, Pratinav Seth, Aditya Tanna, Vinay Kumar Sankarapu

**发布时间:** 2025-11-28

### GPT解析

### 总结

Orion-Bix是一种结合双轴注意力与元学习的上下文推理的表格基础模型，用于少样本表格学习，在公共基准测试上优于梯度提升基线方法，并与最先进的表格基础模型保持竞争力。

### 背景

表格数据驱动大多数现实世界的机器学习应用，但构建通用模型仍然困难。混合数值和分类字段、弱特征结构以及有限的标记数据使得扩展和泛化具有挑战性。

### 目的

引入Orion-Bix，一个表格基础模型，结合双轴注意力与元学习的上下文推理，用于解决少样本表格学习问题。

### 方法

编码器交替使用标准、分组、层次化和关系注意力，通过多CLS摘要融合输出以捕获局部和全局依赖关系；标签感知的ICL头能够即时适应并通过层次决策路由扩展到大型标签空间；在具有因果先验的合成生成的结构多样化的表格上进行元训练，学习跨异构数据的可迁移归纳偏置。

### 主要发现

作为与scikit-learn兼容的基础模型，Orion-Bix在公共基准测试上优于梯度提升基线方法，并与最先进的表格基础模型保持竞争力。

### 结论

双轴注意力与情节元训练相结合能够实现稳健的、适用于少样本的表格学习。

### 翻译

表格数据驱动大多数现实世界的机器学习应用，但构建通用模型仍然困难。混合数值和分类字段、弱特征结构以及有限的标记数据使得扩展和泛化具有挑战性。为此，我们引入了Orion-Bix，一个结合双轴注意力与元学习的上下文推理的表格基础模型，用于少样本表格学习。其编码器交替使用标准、分组、层次化和关系注意力，通过多CLS摘要融合它们的输出，以高效捕获局部和全局依赖关系。标签感知的ICL头能够即时适应，并通过层次决策路由扩展到大型标签空间。在具有因果先验的合成生成的结构多样化的表格上进行元训练，Orion-Bix学习跨异构数据的可迁移归纳偏置。作为与scikit-learn兼容的基础模型，它优于梯度提升基线方法，并在公共基准测试上与最先进的表格基础模型保持竞争力，这表明双轴注意力与情节元训练相结合能够实现稳健的、适用于少样本的表格学习。该模型可在https://github.com/Lexsi-Labs/Orion-BiX公开获取。


### 论文摘要

Tabular data drive most real-world machine learning applications, yet building general-purpose models for them remains difficult. Mixed numeric and categorical fields, weak feature structure, and limited labeled data make scaling and generalization challenging. To this end, we introduce Orion-Bix, a tabular foundation model that combines biaxial attention with meta-learned in-context reasoning for few-shot tabular learning. Its encoder alternates standard, grouped, hierarchical, and relational attention, fusing their outputs through multi-CLS summarization to capture both local and global dependencies efficiently. A label-aware ICL head adapts on the fly and scales to large label spaces via hierarchical decision routing. Meta-trained on synthetically generated, structurally diverse tables with causal priors, Orion-Bix learns transferable inductive biases across heterogeneous data. Delivered as a scikit-learn compatible foundation model, it outperforms gradient-boosting baselines and remains competitive with state-of-the-art tabular foundation models on public benchmarks, showing that biaxial attention with episodic meta-training enables robust, few-shot-ready tabular learning. The model is publicly available at https://github.com/Lexsi-Labs/Orion-BiX .

---

## 73. Measuring What LLMs Think They Do: SHAP Faithfulness and Deployability on Financial Tabular Classification

**论文链接:** [http://arxiv.org/abs/2512.00163v1](http://arxiv.org/abs/2512.00163v1)

**作者:** Saeed AlMarri, Mathieu Ravaut, Kristof Juhasz, Gautier Marti, Hamdan Al Ahbabi, Ibrahim Elfadel

**发布时间:** 2025-11-28

**备注:** 7 pages, 3 figures, 3 tables, AAAI 2026 Deployable AI Workshop

### GPT解析

### 总结

该研究系统评估了大型语言模型在金融分类任务上的表现，通过生成SHAP值分析其可靠性和可解释性，发现LLMs的自我解释与SHAP值存在差异，且与传统模型LightGBM的SHAP值也有显著不同。

### 背景

大型语言模型因其在分类任务中的零样本提示能力而受到关注，被视为传统机器学习模型(如LightGBM)的灵活替代方案。然而，它们在结构化表格数据上的可靠性，特别是在金融风险评估等高风险应用中仍不明确。

### 目的

系统评估LLMs在金融分类任务上的表现，并生成它们的SHAP值，以了解其作为独立分类器在结构化金融建模中的适用性和可靠性。

### 方法

通过系统评估大型语言模型在金融分类任务上的表现，并生成它们的SHAP值来分析模型特征影响和可解释性。

### 主要发现

1) LLMs的自我解释(关于特征影响的解释)与它们的SHAP值之间存在差异；2) LLMs和传统机器学习模型LightGBM的SHAP值之间存在显著差异。

### 结论

LLMs作为独立分类器在结构化金融建模方面存在局限性，但改进的解释性机制结合少样本提示将使LLMs能够在风险敏感领域使用。

### 翻译

大型语言模型因其在分类任务中的零样本提示能力而受到广泛关注，成为可信的传统机器学习模型(如LightGBM)的灵活替代方案。然而，它们在结构化表格数据上的可靠性仍不明确，特别是在金融风险评估等高风险应用中。我们的研究系统评估了LLMs在金融分类任务上的表现，并生成了它们的SHAP值。分析显示，LLMs的自我解释与SHAP值之间存在差异，且与LightGBM的SHAP值也有显著不同。这些发现突显了LLMs作为独立分类器在结构化金融建模中的局限性，但也让人乐观地认为，改进的解释性机制结合少样本提示将使LLMs能够在风险敏感领域使用。


### 论文摘要

Large Language Models (LLMs) have attracted significant attention for classification tasks, offering a flexible alternative to trusted classical machine learning models like LightGBM through zero-shot prompting. However, their reliability for structured tabular data remains unclear, particularly in high stakes applications like financial risk assessment. Our study systematically evaluates LLMs and generates their SHAP values on financial classification tasks. Our analysis shows a divergence between LLMs self-explanation of feature impact and their SHAP values, as well as notable differences between LLMs and LightGBM SHAP values. These findings highlight the limitations of LLMs as standalone classifiers for structured financial modeling, but also instill optimism that improved explainability mechanisms coupled with few-shot prompting will make LLMs usable in risk-sensitive domains.

---

## 74. Hybrid Synthetic Data Generation with Domain Randomization Enables Zero-Shot Vision-Based Part Inspection Under Extreme Class Imbalance

**论文链接:** [http://arxiv.org/abs/2512.00125v1](http://arxiv.org/abs/2512.00125v1)

**作者:** Ruo-Syuan Mei, Sixian Jia, Guangze Li, Soo Yeon Lee, Brian Musser, William Keller, Sreten Zakula, Jorge Arinez, Chenhui Shao

**发布时间:** 2025-11-28

**备注:** Submitted to the NAMRC 54

### GPT解析

### 总结

该研究提出了一种混合式合成数据生成框架，解决了工业质量检验中数据不足和类别不平衡的问题，实现了无需手动标注的高性能零样本学习，显著提升了工业零件检测和分类的准确率。

### 背景

机器学习特别是深度学习正在改变工业质量检验，但训练健壮模型需要大量高质量标注数据，而这些数据在制造业中获取成本高、耗时长、劳动密集。此外，缺陷样本稀少导致严重类别不平衡，限制了机器学习方法在实际生产中的应用。

### 目的

开发一种混合式合成数据生成(SDG)框架，通过创建大型、平衡且完全标注的数据集，实现无需手动标注的零样本学习，用于计算机视觉驱动的工业零件质量检验。

### 方法

集成基于模拟的渲染、域随机化和真实背景合成技术，通过改变零件几何形状、光照和表面特性每小时生成12,960个标注图像，并将合成零件合成到真实图像背景上。采用两阶段架构：YOLOv8n用于目标检测，MobileNetV3-small用于质量分类，仅使用合成数据训练模型。

### 主要发现

检测任务达到mAP@0.5为0.995，分类准确率达96%，平衡准确率达90.1%。在严重类别不平衡情况下，SDG方法达到90-91%的平衡准确率，而基线方法仅达50%的准确率，显示出显著优势。

### 结论

提出的SDG方法实现了无需标注、可扩展且稳健的质量检验，适用于实际制造业应用，有效解决了工业质量检验中的数据约束问题。

### 翻译

机器学习，特别是深度学习正在改变工业质量检验。然而，训练健壮的机器学习模型通常需要大量高质量标注数据，这些数据在制造业中获取成本高昂、耗时且劳动密集。此外，缺陷样本本质上是稀少的，导致严重的类别不平衡，降低模型性能。这些数据限制阻碍了机器学习为基础的质量检验方法在实际生产环境中的广泛应用。合成数据生成(SDG)提供了一种有前途的解决方案，能够以高效、经济和可扩展的方式创建大型、平衡且完全标注的数据集。本文提出了一种混合SDG框架，集成基于模拟的渲染、域随机化和真实背景合成，实现了无需手动标注的计算机视觉驱动的工业零件零样本学习。SDG流程通过改变零件几何形状、光照和表面特性，每小时生成12,960个标注图像，然后将合成零件合成到真实图像背景上。使用YOLOv8n骨干网络进行目标检测、MobileNetV3-small进行质量分类的两阶段架构仅在合成数据上训练，并在300个真实工业零件上评估。所提方法实现了0.995的检测mAP@0.5、96%的分类准确率和90.1%的平衡准确率。与少样本真实数据基线方法的比较评估显示出显著改进。在严重类别不平衡情况下，所提SDG方法达到90-91%的平衡准确率，而基线方法仅达到50%的准确率。这些结果表明，所提方法实现了无需标注、可扩展且稳健的质量检验，适用于实际制造业应用。


### 论文摘要

Machine learning, particularly deep learning, is transforming industrial quality inspection. Yet, training robust machine learning models typically requires large volumes of high-quality labeled data, which are expensive, time-consuming, and labor-intensive to obtain in manufacturing. Moreover, defective samples are intrinsically rare, leading to severe class imbalance that degrades model performance. These data constraints hinder the widespread adoption of machine learning-based quality inspection methods in real production environments. Synthetic data generation (SDG) offers a promising solution by enabling the creation of large, balanced, and fully annotated datasets in an efficient, cost-effective, and scalable manner. This paper presents a hybrid SDG framework that integrates simulation-based rendering, domain randomization, and real background compositing to enable zero-shot learning for computer vision-based industrial part inspection without manual annotation. The SDG pipeline generates 12,960 labeled images in one hour by varying part geometry, lighting, and surface properties, and then compositing synthetic parts onto real image backgrounds. A two-stage architecture utilizing a YOLOv8n backbone for object detection and MobileNetV3-small for quality classification is trained exclusively on synthetic data and evaluated on 300 real industrial parts. The proposed approach achieves an mAP@0.5 of 0.995 for detection, 96% classification accuracy, and 90.1% balanced accuracy. Comparative evaluation against few-shot real-data baseline approaches demonstrates significant improvement. The proposed SDG-based approach achieves 90-91% balanced accuracy under severe class imbalance, while the baselines reach only 50% accuracy. These results demonstrate that the proposed method enables annotation-free, scalable, and robust quality inspection for real-world manufacturing applications.

---

## 75. Objects in Generated Videos Are Slower Than They Appear: Models Suffer Sub-Earth Gravity and Don't Know Galileo's Principle...for now

**论文链接:** [http://arxiv.org/abs/2512.02016v1](http://arxiv.org/abs/2512.02016v1)

**作者:** Varun Varma Thozhiyoor, Shivam Tripathi, Venkatesh Babu Radhakrishnan, Anand Bhattad

**发布时间:** 2025-12-01

**备注:** https://gravity-eval.github.io/

### GPT解析

### 总结

研究视频生成器对重力物理定律的表示，发现它们生成的物体下落加速度较慢，但可通过专业化微调部分纠正这一物理差距

### 背景

视频生成器越来越多地被评估为潜在的世界模型，这要求它们能够编码和理解物理定律，特别是像重力这样的基本物理定律

### 目的

调查视频生成器对重力这一基本物理定律的表示方式，并探索如何改进这种表示

### 方法

首先研究观察到的物理错误是否由度量模糊性引起；引入无单位的、双物体协议测试时间比率关系；使用轻量级低秩适配器进行专业化微调

### 主要发现

开箱即用的视频生成器生成物体下落的有效加速度较慢；时间重缩放无法纠正高方差的重力伪影；相对测试揭示了伽利略等效原理的违反；专业化微调后有效重力加速度从1.81 m/s²提高到6.43 m/s²（达到地球重力的65%）

### 结论

特定的物理定律可以用最少的数据进行部分纠正，专业适配器还能零样本推广到其他物理场景

### 翻译

视频生成器越来越多地被评估为潜在的世界模型，这要求它们能够编码和理解物理定律。我们研究了它们对基本定律——重力的表示。开箱即用的视频生成器一致地生成物体以有效较慢的加速度下落。然而，这些物理测试常常因度量尺度模糊而变得复杂。我们首先研究观察到的物理错误是否是这些模糊性的伪影。我们发现即使时间重缩放也无法纠正高方差的重力伪影。为了严格分离这些混淆因素下的基本物理表示，我们引入了一种无单位的、双物体协议。然后我们证明，这种物理差距可以通过有针对性的专业化部分缓解。专业适配器提供了特定物理定律可以用最少数据纠正的初步证据。


### 论文摘要

Video generators are increasingly evaluated as potential world models, which requires them to encode and understand physical laws. We investigate their representation of a fundamental law: gravity. Out-of-the-box video generators consistently generate objects falling at an effectively slower acceleration. However, these physical tests are often confounded by ambiguous metric scale. We first investigate if observed physical errors are artifacts of these ambiguities (e.g., incorrect frame rate assumptions). We find that even temporal rescaling cannot correct the high-variance gravity artifacts. To rigorously isolate the underlying physical representation from these confounds, we introduce a unit-free, two-object protocol that tests the timing ratio $t_1^2/t_2^2 = h_1/h_2$, a relationship independent of $g$, focal length, and scale. This relative test reveals violations of Galileo's equivalence principle. We then demonstrate that this physical gap can be partially mitigated with targeted specialization. A lightweight low-rank adaptor fine-tuned on only 100 single-ball clips raises $g_{\mathrm{eff}}$ from $1.81\,\mathrm{m/s^2}$ to $6.43\,\mathrm{m/s^2}$ (reaching $65\%$ of terrestrial gravity). This specialist adaptor also generalizes zero-shot to two-ball drops and inclined planes, offering initial evidence that specific physical laws can be corrected with minimal data.

---

## 76. TUNA: Taming Unified Visual Representations for Native Unified Multimodal Models

**论文链接:** [http://arxiv.org/abs/2512.02014v1](http://arxiv.org/abs/2512.02014v1)

**作者:** Zhiheng Liu, Weiming Ren, Haozhe Liu, Zijian Zhou, Shoufa Chen, Haonan Qiu, Xiaoke Huang, Zhaochong An, Fanny Yang, Aditya Patel, Viktar Atliha, Tony Ng, Xiao Han, Chuyan Zhu, Chenyang Zhang, Ding Liu, Juan-Manuel Perez-Rua, Sen He, Jürgen Schmidhuber, Wenhu Chen, Ping Luo, Wei Liu, Tao Xiang, Jonas Schult, Yuren Cong

**发布时间:** 2025-12-01

**备注:** Project page: https://tuna-ai.org/

### GPT解析

### 总结

本文提出了TUNA，一种原生统一多模态模型，通过级联VAE编码器和表示编码器构建统一的连续视觉表示，实现了图像和视频的理解和生成任务的端到端处理。

### 背景

统一多模态模型(UMMs)旨在在单一框架内联合执行多模态理解和生成任务。

### 目的

提出一种原生统一多模态模型，解决现有UMMs中解耦表示导致的格式不匹配问题。

### 方法

通过级联VAE编码器和表示编码器构建统一的连续视觉表示空间，实现图像和视频的理解和生成任务的端到端处理。

### 主要发现

TUNA的统一视觉空间避免了单独编码器引入的表示格式不匹配问题；更强的预训练表示编码器在所有多模态任务中表现更好；在统一设置中，同时训练理解和生成数据使两个任务相互受益而非干扰。

### 结论

在多模态理解和生成基准上的广泛实验表明，TUNA在图像和视频理解、图像和视频生成以及图像编辑方面取得了最先进的结果，证明了其统一表示设计的有效性和可扩展性。

### 翻译

统一多模态模型(UMMs)旨在在单一框架内联合执行多模态理解和生成。我们提出了TUNA，一种原生UMMs，通过级联VAE编码器和表示编码器构建统一的连续视觉表示。这种统一的表示空间允许对图像和视频进行端到端处理，用于理解和生成任务。与具有解耦表示的先前UMMs相比，TUNA的统一视觉空间避免了单独编码器引入的表示格式不匹配问题，在理解和生成任务上都优于解耦替代方案。此外，我们观察到更强的预训练表示编码器在所有多模态任务中始终产生更好的性能，突显了表示编码器的重要性。最后，在这种统一设置中，同时在理解和生成数据上进行训练使两个任务能够相互受益而非干扰。我们在多模态理解和生成基准上的广泛实验表明，TUNA在图像和视频理解、图像和视频生成以及图像编辑方面取得了最先进的结果，证明了其统一表示设计的有效性和可扩展性。


### 论文摘要

Unified multimodal models (UMMs) aim to jointly perform multimodal understanding and generation within a single framework. We present TUNA, a native UMM that builds a unified continuous visual representation by cascading a VAE encoder with a representation encoder. This unified representation space allows end-to-end processing of images and videos for both understanding and generation tasks. Compared to prior UMMs with decoupled representations, TUNA's unified visual space avoids representation format mismatches introduced by separate encoders, outperforming decoupled alternatives in both understanding and generation. Moreover, we observe that stronger pretrained representation encoders consistently yield better performance across all multimodal tasks, highlighting the importance of the representation encoder. Finally, in this unified setting, jointly training on both understanding and generation data allows the two tasks to benefit from each other rather than interfere. Our extensive experiments on multimodal understanding and generation benchmarks show that TUNA achieves state-of-the-art results in image and video understanding, image and video generation, and image editing, demonstrating the effectiveness and scalability of its unified representation design.

---

## 77. MV-TAP: Tracking Any Point in Multi-View Videos

**论文链接:** [http://arxiv.org/abs/2512.02006v1](http://arxiv.org/abs/2512.02006v1)

**作者:** Jahyeok Koo, Inès Hyeonsu Kim, Mungyeom Kim, Junghyun Park, Seohyun Park, Jaeyeong Kim, Jung Yi, Seokju Cho, Seungryong Kim

**发布时间:** 2025-12-01

**备注:** Project Page: https://cvlab-kaist.github.io/MV-TAP/

### GPT解析

### 总结

本文提出了MV-TAP，一种新颖的多视图点跟踪器，通过利用跨视图信息实现更可靠的多视图轨迹估计。

### 背景

多摄像头系统能够对复杂现实世界场景进行丰富观察，在多视图设置中理解动态对象已成为各种应用的核心。

### 目的

开发一个能够利用跨视图信息跟踪多视图中动态场景点的跟踪器，实现更完整和可靠的轨迹估计。

### 方法

MV-TAP利用相机几何和跨视图注意力机制来聚合跨视图的时空信息，并构建了大规模合成训练数据集和针对多视图跟踪定制的真实世界评估集。

### 主要发现

MV-TAP在具有挑战性的基准测试中优于现有的点跟踪方法，能够实现更完整和可靠的轨迹估计。

### 结论

MV-TAP为推进多视图点跟踪研究建立了有效的基线。

### 翻译

多摄像头系统能够对复杂现实世界场景进行丰富观察，在多视图设置中理解动态对象已成为各种应用的核心。在本工作中，我们提出了MV-TAP，一种新颖的点跟踪器，通过利用跨视图信息来跟踪多视图视频中动态场景的点。MV-TAP利用相机几何和跨视图注意力机制来聚合跨视图的时空信息，从而能够在多视图中实现更完整和可靠的轨迹估计。为支持这一任务，我们构建了一个大规模合成训练数据集和针对多视图跟踪定制的真实世界评估集。大量实验表明，MV-TAP在具有挑战性的基准测试中优于现有的点跟踪方法，为推进多视图点跟踪研究建立了有效的基线。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决多视角视频中的任意点跟踪问题。现有单视角点跟踪方法存在遮挡、运动模糊和深度不确定性等固有局限，而多视角摄像头系统可以提供互补信息来克服这些限制。这个问题在动作捕捉、机器人操作和自动驾驶等实际应用中非常重要，因为这些应用需要精确理解动态场景中的物体运动。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先认识到单视角点跟踪的局限性，然后利用多视角信息可以解决这些问题的洞察。他们借鉴了单视角点跟踪方法（特别是CoTracker3）作为基础架构，同时引入了多视角几何信息和跨视角注意力机制。作者还构建了专门的多视角训练数据集，因为缺乏适合的训练数据。整体设计思路是在保留单视角跟踪优势的同时，增加多视角感知能力，通过相机编码和视图注意力来整合跨视角信息。', '这个方法的核心思想是什么？整体实现流程是怎样的？': 'MV-TAP的核心思想是利用多视角信息增强点跟踪的鲁棒性，通过跨视角信息融合、几何感知和注意力机制来解决单视角的局限性。整体流程包括：1)输入多视角视频、查询点和相机参数；2)计算局部4D相关性；3)将相关性编码为标记；4)使用Plücker坐标进行视图感知的相机编码；5)通过多维度Transformer(时间、空间和视图注意力)处理标记；6)迭代优化轨迹和遮挡状态；7)输出最终的多视角跟踪结果。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)首次定义像素空间中的多视角点跟踪任务；2)提出MV-TAP框架，融合相机几何和跨视角信息；3)设计视图感知的相机编码模块；4)引入跨视角注意力机制；5)构建专门的多视角训练和评估数据集。与之前工作不同，MV-TAP直接在2D像素空间工作，不依赖外部深度输入；同时结合了时间、空间和视图三个维度的注意力，而现有方法通常只考虑其中一到两个维度；在处理遮挡和快速运动等挑战性场景时表现更鲁棒。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '这篇论文提出了MV-TAP，一种创新的多视角点跟踪方法，通过融合跨视角信息和相机几何，显著提高了动态场景中点跟踪的鲁棒性和准确性，并为此任务创建了首个大规模数据集和评估基准。'}


### 论文摘要

Multi-view camera systems enable rich observations of complex real-world scenes, and understanding dynamic objects in multi-view settings has become central to various applications. In this work, we present MV-TAP, a novel point tracker that tracks points across multi-view videos of dynamic scenes by leveraging cross-view information. MV-TAP utilizes camera geometry and a cross-view attention mechanism to aggregate spatio-temporal information across views, enabling more complete and reliable trajectory estimation in multi-view videos. To support this task, we construct a large-scale synthetic training dataset and real-world evaluation sets tailored for multi-view tracking. Extensive experiments demonstrate that MV-TAP outperforms existing point-tracking methods on challenging benchmarks, establishing an effective baseline for advancing research in multi-view point tracking.

---

## 78. Learning Visual Affordance from Audio

**论文链接:** [http://arxiv.org/abs/2512.02005v1](http://arxiv.org/abs/2512.02005v1)

**作者:** Lidong Lu, Guo Chen, Zhu Wei, Yicheng Liu, Tong Lu

**发布时间:** 2025-12-01

**备注:** 15 pages, 10 figures

### GPT解析

### 总结

介绍视听可操作性定位(AV-AG)新任务，通过动作声音分割物体交互区域，构建首个数据集并提出AVAGFormer模型，实现最先进性能。

### 背景

现有可操作性定位方法依赖文本指令或演示视频，常受模糊性或遮挡限制，而音频能提供实时、语义丰富且视觉独立的线索。

### 目的

引入AV-AG任务，通过动作声音分割物体交互区域，克服现有方法局限性，构建相应数据集和模型。

### 方法

构建首个AV-AG数据集，包含动作声音、物体图像和像素级可操作性标注，并包含未见子集评估零样本泛化；提出AVAGFormer模型，配备语义条件化跨模态混合器和双头解码器，融合音频和视觉信号进行掩码预测。

### 主要发现

AVAGFormer在AV-AG任务上取得最先进性能；实验分析揭示AV-AG与AVS区别、端到端建模优势和各组件贡献。

### 结论

通过引入AV-AG任务、构建数据集和提出模型，展示了利用音频信号进行可操作性定位的有效性，为领域提供新研究方向。

### 翻译

我们引入了视听可操作性定位（Audio-Visual Affordance Grounding, AV-AG），这是一个新任务，能够从动作声音中分割物体交互区域。与依赖文本指令或演示视频的现有方法（通常受模糊性或遮挡限制）不同，音频为可操作性定位提供了实时、语义丰富且视觉独立的线索，使得对交互区域的理解更加直观。为了支持这一任务，我们构建了首个AV-AG数据集，包含大量动作声音、物体图像和像素级可操作性标注。该数据集还包括一个未见过的子集用于评估零样本泛化能力。此外，我们提出了AVAGFormer模型，该模型配备了语义条件化的跨模态混合器和双头解码器，能有效融合音频和视觉信号进行掩码预测。实验表明，AVAGFormer在AV-AG任务上取得了最先进的性能，超越了相关任务的基线。全面分析突出了AV-AG与AVS之间的区别、端到端建模的优势以及每个组件的贡献。代码和数据集已在https://jscslld.github.io/AVAGFormer/上发布。


### 论文摘要

We introduce Audio-Visual Affordance Grounding (AV-AG), a new task that segments object interaction regions from action sounds. Unlike existing approaches that rely on textual instructions or demonstration videos, which often limited by ambiguity or occlusion, audio provides real-time, semantically rich, and visually independent cues for affordance grounding, enabling more intuitive understanding of interaction regions. To support this task, we construct the first AV-AG dataset, comprising a large collection of action sounds, object images, and pixel-level affordance annotations. The dataset also includes an unseen subset to evaluate zero-shot generalization. Furthermore, we propose AVAGFormer, a model equipped with a semantic-conditioned cross-modal mixer and a dual-head decoder that effectively fuses audio and visual signals for mask prediction. Experiments show that AVAGFormer achieves state-of-the-art performance on AV-AG, surpassing baselines from related tasks. Comprehensive analyses highlight the distinctions between AV-AG and AVS, the benefits of end-to-end modeling, and the contribution of each component. Code and dataset have been released on https://jscslld.github.io/AVAGFormer/.

---

## 79. PAI-Bench: A Comprehensive Benchmark For Physical AI

**论文链接:** [http://arxiv.org/abs/2512.01989v1](http://arxiv.org/abs/2512.01989v1)

**作者:** Fengzhe Zhou, Jiannan Huang, Jialuo Li, Deva Ramanan, Humphrey Shi

**发布时间:** 2025-12-01

### GPT解析

### 总结

该研究引入了Physical AI Bench (PAI-Bench)，一个统一且全面的基准，用于评估物理人工智能模型的感知和预测能力，包含2808个真实案例和专门设计的指标。

### 背景

Physical AI旨在开发能够感知和预测现实世界动态的模型，但目前对当前多模态大语言模型和视频生成模型支持这些能力的程度理解不足。

### 目的

创建一个统一且全面的基准来评估视频生成、条件视频生成和视频理解方面的感知和预测能力。

### 方法

构建包含2808个真实案例的基准，设计与任务对齐的指标来捕捉物理合理性和领域特定推理能力，并对最近的模型进行系统评估。

### 主要发现

视频生成模型尽管视觉保真度高，但往往难以保持物理连贯的动态；多模态大语言模型在预测和因果解释方面表现有限。

### 结论

当前系统在处理Physical AI的感知和预测需求方面仍处于早期阶段；PAI-Bench为评估Physical AI建立了现实基础，并指出了未来系统必须解决的关键差距。

### 翻译

Physical AI旨在开发能够感知和预测现实世界动态的模型；然而，当前多模态大语言模型和视频生成模型支持这些能力的程度尚未得到充分理解。我们引入了Physical AI Bench (PAI-Bench)，一个统一且全面的基准，用于评估视频生成、条件视频生成和视频理解方面的感知和预测能力，包含2808个真实案例和与任务对齐的指标，这些指标旨在捕捉物理合理性和领域特定推理能力。我们的研究对最近的模型进行了系统评估，表明视频生成模型尽管具有强大的视觉保真度，但往往难以保持物理连贯的动态，而多模态大语言模型在预测和因果解释方面表现出有限性能。这些观察表明，当前系统在处理Physical AI的感知和预测需求方面仍处于早期阶段。总之，PAI-Bench为评估Physical AI建立了现实基础，并突出了未来系统必须解决的关键差距。


### 论文摘要

Physical AI aims to develop models that can perceive and predict real-world dynamics; yet, the extent to which current multi-modal large language models and video generative models support these abilities is insufficiently understood. We introduce Physical AI Bench (PAI-Bench), a unified and comprehensive benchmark that evaluates perception and prediction capabilities across video generation, conditional video generation, and video understanding, comprising 2,808 real-world cases with task-aligned metrics designed to capture physical plausibility and domain-specific reasoning. Our study provides a systematic assessment of recent models and shows that video generative models, despite strong visual fidelity, often struggle to maintain physically coherent dynamics, while multi-modal large language models exhibit limited performance in forecasting and causal interpretation. These observations suggest that current systems are still at an early stage in handling the perceptual and predictive demands of Physical AI. In summary, PAI-Bench establishes a realistic foundation for evaluating Physical AI and highlights key gaps that future systems must address.

---

## 80. Predicting Onsets and Dry Spells of the West African Monsoon Season Using Machine Learning Methods

**论文链接:** [http://arxiv.org/abs/2512.01965v1](http://arxiv.org/abs/2512.01965v1)

**作者:** Colin Bobocea, Yves Atchadé

**发布时间:** 2025-12-01

### GPT解析

### 总结

本研究探索了利用机器学习模型预测西非雨季开始和干旱期的有效方法，通过结合海表温度遥相关关系，开发了两种预测模型并评估了其预测效果。

### 背景

西非雨季开始和干旱期的预测非常困难，但这些指标是农民决定何时种植作物并影响整体产量的关键。虽然许多研究已经表明全球海表温度与西非季风季节特征之间存在相关性，但很少有研究能有效地将这些信息整合到机器学习预测模型中。

### 目的

探索定义目标变量（雨季开始和干旱期）的最佳方法，并利用海表温度遥相关关系预测未来季节的这些现象。

### 方法

结合了两种著名的雨季开始定义方法来定义目标变量，并在两个模型（线性模型和自适应阈值逻辑回归模型）上应用了定制统计技术——总变分正则化和预测变量选择。

### 主要发现

对于雨季开始的预测，结果混合，空间验证显示出显著技能，而时间验证则显示出很少或没有技能；对于干旱期，通过分析多个二元分类指标，发现了显著准确性。这些模型克服了当前方法的一些局限性，如计算密集和需要偏差校正。

### 结论

该研究引入了一个使用机器学习方法利用气候相关变量针对特定天气现象预测的框架，随着机器学习技术应用于更多问题，气象学等领域将看到明显益处，并提出了几个新的研究方向。

### 翻译

西非雨季的开始和干旱期的发生 notoriously 难以预测，然而这些是农民用来决定何时种植作物的关键指标，对他们的整体产量有重大影响。虽然许多研究已经表明全球海表温度与西非季风季节特征之间存在相关性，但很少有研究能有效地将这些信息整合到机器学习预测模型中。在本研究中，我们研究了定义目标变量（开始和干旱期）的最佳方法，并利用海表温度遥相关关系预测未来季节。定义我们的目标变量需要结合两种著名的开始定义。然后，我们在构建的两个模型上应用了定制统计技术——如总变分正则化和预测变量选择——第一个是线性模型，第二个是自适应阈值逻辑回归模型。我们发现雨季开始的预测结果混合，空间验证显示出显著技能的迹象，而时间验证则显示出很少或没有技能。然而，对于干旱期，通过分析多个二元分类指标，我们发现显著准确性。这些模型克服了当前方法的一些局限性，如计算密集和需要偏差校正。我们还引入本研究作为使用机器学习方法利用气候相关变量针对特定天气现象预测的框架。随着我们将机器学习技术应用于更多问题，我们看到气象学等领域的明显益处，并提出了几个进一步研究的新方向。


### 论文摘要

The beginning of the rainy season and the occurrence of dry spells in West Africa is notoriously difficult to predict, however these are the key indicators farmers use to decide when to plant crops, having a major influence on their overall yield. While many studies have shown correlations between global sea surface temperatures and characteristics of the West African monsoon season, there are few that effectively implementing this information into machine learning (ML) prediction models. In this study we investigated the best ways to define our target variables, onset and dry spell, and produced methods to predict them for upcoming seasons using sea surface temperature teleconnections. Defining our target variables required the use of a combination of two well known definitions of onset. We then applied custom statistical techniques -- like total variation regularization and predictor selection -- to the two models we constructed, the first being a linear model and the other an adaptive-threshold logistic regression model. We found mixed results for onset prediction, with spatial verification showing signs of significant skill, while temporal verification showed little to none. For dry spell though, we found significant accuracy through the analysis of multiple binary classification metrics. These models overcome some limitations that current approaches have, such as being computationally intensive and needing bias correction. We also introduce this study as a framework to use ML methods for targeted prediction of certain weather phenomenon using climatologically relevant variables. As we apply ML techniques to more problems, we see clear benefits for fields like meteorology and lay out a few new directions for further research.

---

## 81. SpriteHand: Real-Time Versatile Hand-Object Interaction with Autoregressive Video Generation

**论文链接:** [http://arxiv.org/abs/2512.01960v1](http://arxiv.org/abs/2512.01960v1)

**作者:** Zisu Li, Hengye Lyu, Jiaxin Shi, Yufeng Zeng, Mingming Fan, Hanwang Zhang, Chen Liang

**发布时间:** 2025-12-01

### GPT解析

### 总结

本文介绍了SpriteHand，一种用于实时合成各种手部-物体交互视频的自动回归视频生成框架，能够处理多种类型的物体包括非刚性或关节式实体，并在单个GPU上实现实时生成。

### 背景

建模和合成复杂的手部-物体交互仍然是一个重大挑战，即使是对于最先进的物理引擎也是如此。传统的基于模拟的方法依赖于明确定义的刚性物体模型和预编写的手势，无法捕捉与可变形织物、弹性材料、铰链结构、毛茸表面甚至生物等动态实体的交互。

### 目的

开发一种能够实时合成各种物体类型和运动模式的手部-物体交互视频的框架，以克服传统方法的局限性。

### 方法

SpriteHand采用因果推理架构进行自动回归生成，并利用混合后训练方法增强视觉真实性和时间一致性。该模型接收静态物体图像和手部与虚拟物体交互的视频流作为输入，实时生成相应的手部-物体交互效果。

### 主要发现

1.3B模型支持在单个NVIDIA RTX 5090 GPU上以约18 FPS和640x368分辨率进行实时流式生成，延迟约150毫秒，可生成超过一分钟连续输出；与生成式和基于引擎的基线相比，具有优越的视觉质量、物理合理性和交互保真度。

### 结论

SpriteHand为复杂手部-物体交互的实时合成提供了有效的解决方案，能够处理多种类型的物体并生成高质量、物理合理且时间一致的视频内容。

### 翻译

建模和合成复杂的手部-物体交互仍然是一个重大挑战，即使是对于最先进的物理引擎也是如此。传统的基于模拟的方法依赖于明确定义的刚性物体模型和预编写的手势，无法捕捉与可变形织物、弹性材料、铰链结构、毛茸表面甚至生物等动态实体的交互。在本文中，我们介绍了SpriteHand，这是一种用于实时合成各种物体类型和运动模式的手部-物体交互视频的自动回归视频生成框架。SpriteHand接收静态物体图像和手部与嵌入在现实世界场景中的虚拟物体交互的视频流作为输入，并实时生成相应的手部-物体交互效果。我们的模型采用因果推理架构进行自动回归生成，并利用混合后训练方法增强视觉真实性和时间一致性。我们的1.3B模型支持在单个NVIDIA RTX 5090 GPU上以约18 FPS和640x368分辨率进行实时流式生成，延迟约150毫秒，可生成超过一分钟连续输出。实验表明，与生成式和基于引擎的基线相比，该模型具有优越的视觉质量、物理合理性和交互保真度。


### 论文摘要

Modeling and synthesizing complex hand-object interactions remains a significant challenge, even for state-of-the-art physics engines. Conventional simulation-based approaches rely on explicitly defined rigid object models and pre-scripted hand gestures, making them inadequate for capturing dynamic interactions with non-rigid or articulated entities such as deformable fabrics, elastic materials, hinge-based structures, furry surfaces, or even living creatures. In this paper, we present SpriteHand, an autoregressive video generation framework for real-time synthesis of versatile hand-object interaction videos across a wide range of object types and motion patterns. SpriteHand takes as input a static object image and a video stream in which the hands are imagined to interact with the virtual object embedded in a real-world scene, and generates corresponding hand-object interaction effects in real time. Our model employs a causal inference architecture for autoregressive generation and leverages a hybrid post-training approach to enhance visual realism and temporal coherence. Our 1.3B model supports real-time streaming generation at around 18 FPS and 640x368 resolution, with an approximate 150 ms latency on a single NVIDIA RTX 5090 GPU, and more than a minute of continuous output. Experiments demonstrate superior visual quality, physical plausibility, and interaction fidelity compared to both generative and engine-based baselines.

---

## 82. GrndCtrl: Grounding World Models via Self-Supervised Reward Alignment

**论文链接:** [http://arxiv.org/abs/2512.01952v1](http://arxiv.org/abs/2512.01952v1)

**作者:** Haoyang He, Jay Patrikar, Dong-Ki Kim, Max Smith, Daniel McGann, Ali-akbar Agha-mohammadi, Shayegan Omidshafiei, Sebastian Scherer

**发布时间:** 2025-12-01

### GPT解析

### 总结

论文提出了RLWG框架，通过几何和感知奖励将预训练的世界模型与物理可验证结构对齐，解决了视频世界模型缺乏几何基础的问题，提高了导航任务中的空间一致性和长期稳定性。

### 背景

视频世界建模的最新进展使大规模生成模型能够以高视觉保真度模拟具身环境，为预测、规划和控制提供了强大的先验知识。然而，这些模型尽管具有真实感，但常常缺乏几何基础，限制了它们在需要空间一致性和长期稳定性的导航任务中的应用。

### 目的

引入一种名为'带世界基础的强化学习'(RLWG)的自监督后训练框架，通过几何和感知奖励将预训练的世界模型与可物理验证的结构对齐，提高模型在导航任务中的表现。

### 方法

RLWG类似于语言模型中的'从可验证反馈中强化学习'(RLVR)，使用多种奖励测量姿态循环一致性、深度重投影和时间一致性。作者实现了GrndCtrl，这是一种基于组相对策略优化(GRPO)的奖励对齐适应方法。

### 主要发现

GrndCtrl产生保持稳定轨迹、一致几何和可靠滚动的世界模型，用于具身导航。与监督微调相比，GrndCtrl在室外环境中实现了更好的空间一致性和导航稳定性。

### 结论

像大型语言模型的后训练对齐一样，GrndCtrl利用可验证奖励来弥合生成预训练和基础行为之间的差距，在导航任务中表现优于监督微调。

### 翻译

视频世界建模的最新进展使大规模生成模型能够以高视觉保真度模拟具身环境，为预测、规划和控制提供了强大的先验知识。然而，尽管这些模型具有真实感，但它们常常缺乏几何基础，限制了它们在需要空间一致性和长期稳定性的导航任务中的使用。我们引入了'带世界基础的强化学习'(RLWG)，这是一个自监督后训练框架，通过几何和感知奖励将预训练的世界模型与可物理验证的结构对齐。类似于语言模型中的'从可验证反馈中强化学习'(RLVR)，RLWG可以使用多种奖励来测量姿态循环一致性、深度重投影和时间一致性。我们通过GrndCtrl实例化了这个框架，这是一种基于组相对策略优化(GRPO)的奖励对齐适应方法，产生具有稳定轨迹、一致几何和可靠滚动的世界模型，用于具身导航。像大型语言模型中的后训练对齐一样，GrndCtrl利用可验证奖励来弥合生成预训练和基础行为之间的差距，在室外环境中实现了优于监督微调的空间一致性和导航稳定性。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决视频世界模型的几何一致性问题，即当前模型虽然能生成高视觉保真度的未来帧，但缺乏空间和时序上的一致性，导致姿态漂移、深度不稳定和轨迹失真。这个问题在机器人导航、自动驾驶等领域至关重要，因为这些应用需要物理一致的表示来进行可靠的定位、地图绘制和长期规划，而不仅仅是视觉上的逼真。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': "作者观察到预训练的视频世界模型在视觉上逼真但几何上不一致，限制了其在需要空间一致性的任务中的应用。他们借鉴了语言模型中的'基于可验证奖励的强化学习'(RLVR)思想，将其扩展到具身领域，用几何和时序验证替代文本逻辑验证。为了优化这些奖励，他们采用'组相对策略优化'(GRPO)作为训练机制，形成了GrndCtrl算法。此外，他们还借鉴了MapAnything和VideoAlign等现有评估工具来计算可验证的奖励。", '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是通过自我监督的强化学习框架，对预训练的世界模型进行后训练，使用可验证的几何和感知奖励来对齐模型，使其生成的轨迹在空间和时序上保持一致。整体流程包括：1)给定条件上下文(初始帧和动作序列)；2)预训练模型生成多个候选轨迹；3)使用冻结的3D评估器和视频评估器计算每个轨迹的平移、旋转、深度时序重投影和视频质量奖励；4)在组内计算归一化奖励和相对优势；5)使用GRPO优化模型参数，同时向预训练模型正则化；6)输出优化后的世界模型，能生成物理一致且视觉保真的轨迹。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)引入RLWG框架，使用可验证的几何和感知奖励对齐世界模型；2)提出GrndCtrl算法，基于GRPO进行多目标奖励对齐；3)实现完全自我监督的训练过程，无需人工标注或外部模拟器；4)在多个数据集上验证了方法的有效性。相比之前的工作，GrndCtrl与传统监督微调相比在反事实场景中表现更好；与其他世界模型控制方法相比，它能够评估和优化物理正确性；与基于重建损失的方法相比，它衡量的是物理正确性而非仅像素误差。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'GrndCtrl通过自我监督的强化学习框架，利用可验证的几何和感知奖励对预训练的视频世界模型进行后训练，显著提高了模型在导航任务中的空间一致性和长期稳定性，特别是在反事实场景中表现出色。'}


### 论文摘要

Recent advances in video world modeling have enabled large-scale generative models to simulate embodied environments with high visual fidelity, providing strong priors for prediction, planning, and control. Yet, despite their realism, these models often lack geometric grounding, limiting their use in navigation tasks that require spatial coherence and long-horizon stability. We introduce Reinforcement Learning with World Grounding (RLWG), a self-supervised post-training framework that aligns pretrained world models with a physically verifiable structure through geometric and perceptual rewards. Analogous to reinforcement learning from verifiable feedback (RLVR) in language models, RLWG can use multiple rewards that measure pose cycle-consistency, depth reprojection, and temporal coherence. We instantiate this framework with GrndCtrl, a reward-aligned adaptation method based on Group Relative Policy Optimization (GRPO), yielding world models that maintain stable trajectories, consistent geometry, and reliable rollouts for embodied navigation. Like post-training alignment in large language models, GrndCtrl leverages verifiable rewards to bridge generative pretraining and grounded behavior, achieving superior spatial coherence and navigation stability over supervised fine-tuning in outdoor environments.

---

## 83. Script: Graph-Structured and Query-Conditioned Semantic Token Pruning for Multimodal Large Language Models

**论文链接:** [http://arxiv.org/abs/2512.01949v1](http://arxiv.org/abs/2512.01949v1)

**作者:** Zhongyu Yang, Dannong Xu, Wei Pang, Yingfang Yuan

**发布时间:** 2025-12-01

**备注:** Published in Transactions on Machine Learning Research, Project in https://01yzzyu.github.io/script.github.io/

### GPT解析

### 总结

Script是一种即插即用的多模态大语言模型视觉token修剪方法，通过两个模块分别去除视觉冗余和保留查询相关信息，提高模型效率和准确性。

### 背景

多模态大语言模型中视觉token快速增长导致内存消耗和推理延迟增加，特别是在处理高分辨率图像和视频时。

### 目的

解决现有token修剪方法忽略用户查询相关性或受注意力机制局限的问题，提高模型的适应性和有效性。

### 方法

Script包含图结构修剪模块(去除视觉冗余token)和查询条件语义修剪模块(保留查询相关的视觉信息)，无需重新训练且可推广到各种MLLMs。

### 主要发现

在14个图像和视频理解基准测试中，Script比现有修剪方法实现更高模型效率和预测准确性；在LLaVA-NeXT-7B上实现6.8倍预填充加速和10倍FLOP减少，同时保留96.88%原始性能。

### 结论

Script是一种有效的即插即用修剪方法，无需重新训练且可推广到多种多模态大语言模型，显著提高模型效率同时保持高预测准确性。

### 翻译

多模态大语言模型(MLLMs)中视觉token的快速增长导致过度的内存消耗和推理延迟，特别是在处理高分辨率图像和视频时。Token修剪是一种通过去除冗余来缓解此问题的技术，但现有方法通常忽略了与用户查询的相关性或受到注意力机制的局限，降低了它们的适应性和有效性。为应对这些挑战，我们提出了Script，一种即插即用的修剪方法，无需重新训练且可推广到各种MLLMs。Script包含两个模块：去除视觉冗余token的图结构修剪模块，以及保留查询相关视觉信息的查询条件语义修剪模块。两者共同提高了多模态任务上的性能。在图像和视频理解任务的14个基准测试中，Script比现有修剪方法一致实现了更高的模型效率和预测准确性。在LLaVA-NeXT-7B上，它实现了高达6.8倍的预填充加速和10倍的FLOP减少，同时保留了96.88%的原始性能。


### 论文摘要

The rapid growth of visual tokens in multimodal large language models (MLLMs) leads to excessive memory consumption and inference latency, especially when handling high-resolution images and videos. Token pruning is a technique used to mitigate this issue by removing redundancy, but existing methods often ignore relevance to the user query or suffer from the limitations of attention mechanisms, reducing their adaptability and effectiveness. To address these challenges, we propose Script, a plug-and-play pruning method that requires no retraining and generalizes across diverse MLLMs. Script comprises two modules: a graph-structured pruning module that removes visually redundant tokens, and a query-conditioned semantic pruning module that preserves query-relevant visual information. Together, they enhance performance on multimodal tasks. Experiments on fourteen benchmarks across image and video understanding tasks show that Script consistently achieves higher model efficiency and predictive accuracy compared to existing pruning methods. On LLaVA-NeXT-7B, it achieves up to 6.8x prefill speedup and 10x FLOP reduction, while retaining 96.88% of the original performance.

---

## 84. Delays in Spiking Neural Networks: A State Space Model Approach

**论文链接:** [http://arxiv.org/abs/2512.01906v1](http://arxiv.org/abs/2512.01906v1)

**作者:** Sanja Karilanova, Subhrakanti Dey, Ayça Özçelikkale

**发布时间:** 2025-12-01

### GPT解析

### 总结

该研究提出了一种通过附加状态变量将延迟机制纳入脉冲神经网络(SNNs)的通用框架，使神经元能够访问有限的时间输入历史，实验证明该机制在保持计算效率的同时匹配了现有基于延迟的SNNs性能，并能显著提升小规模网络的表现。

### 背景

脉冲神经网络(SNNs)是受生物启发的、事件驱动的模型，适合处理时间数据，并在神经形态硬件上实现时能提供高效的计算能力。在SNNs中，更丰富的神经元动态可以捕获更复杂的时间依赖性，而延迟通过允许过去的输入直接影响当前的脉冲行为发挥着关键作用。

### 目的

提出一个通用框架，将延迟机制纳入SNNs，使每个神经元能够访问有限的时间输入历史，同时保持与标准脉冲神经元模型的兼容性。

### 方法

通过附加状态变量将延迟机制纳入SNNs。该框架与神经元模型无关，可以无缝集成到标准的脉冲神经元模型如LIF和adLIF中。研究人员分析了延迟持续时间以及与延迟相关的可学习参数如何影响性能，并研究了由延迟机制引入的附加状态变量在网络架构中的权衡。

### 主要发现

1) 延迟机制在Spiking Heidelberg Digits(SHD)数据集上表现优异，匹配了现有基于延迟的SNNs性能；2) 该机制保持了计算效率；3) 在较小的网络中，纳入延迟可以显著提高性能。

### 结论

提出的延迟机制能够有效地提升SNNs的性能，特别是在较小的网络中，同时保持计算效率。该框架为SNNs中利用时间信息提供了灵活且通用的方法。

### 翻译

脉冲神经网络(SNNs)是受生物启发的、事件驱动的模型，适合处理时间数据，并在神经形态硬件上实现时能提供高效的计算能力。在SNNs中，更丰富的神经元动态可以捕获更复杂的时间依赖性，而延迟通过允许过去的输入直接影响当前的脉冲行为发挥着关键作用。我们提出了一种通过附加状态变量将延迟纳入SNNs的通用框架。所提出的机制使每个神经元能够访问有限的时间输入历史。该框架与神经元模型无关，因此可以无缝集成到标准的脉冲神经元模型如LIF和adLIF中。我们分析了延迟的持续时间以及与延迟相关的可学习参数如何影响性能。我们研究了由延迟机制引入的附加状态变量在网络架构中的权衡。在Spiking Heidelberg Digits(SHD)数据集上的实验表明，所提出的机制在保持计算效率的同时匹配了现有基于延迟的SNNs的性能。此外，结果表明在较小的网络中，纳入延迟可以显著提高性能。


### 论文摘要

Spiking neural networks (SNNs) are biologically inspired, event-driven models that are suitable for processing temporal data and offer energy-efficient computation when implemented on neuromorphic hardware. In SNNs, richer neuronal dynamic allows capturing more complex temporal dependencies, with delays playing a crucial role by allowing past inputs to directly influence present spiking behavior. We propose a general framework for incorporating delays into SNNs through additional state variables. The proposed mechanism enables each neuron to access a finite temporal input history. The framework is agnostic to neuron models and hence can be seamlessly integrated into standard spiking neuron models such as LIF and adLIF. We analyze how the duration of the delays and the learnable parameters associated with them affect the performance. We investigate the trade-offs in the network architecture due to additional state variables introduced by the delay mechanism. Experiments on the Spiking Heidelberg Digits (SHD) dataset show that the proposed mechanism matches the performance of existing delay-based SNNs while remaining computationally efficient. Moreover, the results illustrate that the incorporation of delays may substantially improve performance in smaller networks.

---

## 85. Active chromospheric fibril singularity: Coordinated observations from Solar Orbiter, SST, and IRIS

**论文链接:** [http://arxiv.org/abs/2512.01886v1](http://arxiv.org/abs/2512.01886v1)

**作者:** Reetika Joshi, Luc Rouppe van der Voort, Guillaume Aulanier, Sanja Danilovic, Avijeet Prasad, Carlos J. Díaz Baso, Daniel Nóbrega-Siverio, Nicolas Poirier, Daniele Calchetti

**发布时间:** 2025-12-01

**备注:** 6 pages, 5 figures, Accepted for publication in A&A

### GPT解析

### 总结

本研究发现了一种新型的太阳色球纤维奇异性，该奇异性出现在喷流和耀斑环附近，研究者通过协调的高分辨率多波长观测数据分析了其磁性质和活动原因。

### 背景

太阳色球精细结构由光球层运动驱动，在太阳磁场动力学中起着关键作用。虽然已识别多种结构如纤维状物、丝状足和拱形纤维系统，但高分辨率观测仍显示许多未被理解的结构。

### 目的

理解这种色球纤维奇异性的磁性质及其活动原因。

### 方法

协调Solar Orbiter、SST、IRIS和SDO的数据集，将Solar Orbiter数据重新投影以匹配地面仪器视角，使用Solar Orbiter/PHI数据进行势场外推，分析等离子体结构与表面磁场的时空演化。

### 主要发现

发现了一种新特征：色球纤维模式中的奇异性，它形成于两个同号通量集中之间的弱磁场走廊中，位于倒Y形磁场线模式底部。结构一端出现耀斑环，另一端出现喷流，该喷流处存在日冕奇点，并与位于纤维奇异性上的色球鞍点相关联。

### 结论

这种纤维奇异性与太阳活动密切相关，可能是一个新的太阳活动触发点。

### 翻译

太阳色球的精细结构，由光球层运动驱动，在太阳磁场动力学中起着关键作用。已经识别了许多结构，如纤维状物、丝状足和拱形纤维系统。然而，高分辨率观测仍显示出大量未被理解的结构。我们在一个爆发性太阳喷流和一个耀斑环附近观察到一个令人困惑的、前所未有的色球纤维奇异性。我们旨在使用协调的高分辨率多波长观测数据来理解这种奇异性的磁性质及其活动原因。我们对齐了来自Solar Orbiter、SST、IRIS和SDO的数据集，并将Solar Orbiter数据重新投影以匹配地面仪器的视角。我们使用Solar Orbiter/PHI数据进行了势场外推，分析了等离子体结构的时空演化及其与表面磁场的联系。这使我们能够推导出所观测结构的模型和情景，并在一般示意图中进行了解释。我们发现了一个新特征：色球纤维模式中的一个奇异性。它形成于两个同号通量集中之间的弱磁场走廊中，位于倒Y形磁场线模式的底部。在这种特定情况下，结构沿某些方向发展活动。一端首先出现耀斑环，另一端出现爆发性喷流，该喷流处存在日冕奇点，并与位于纤维奇异性上的色球鞍点相关联。观测结果表明...


### 论文摘要

The fine structures of the solar chromosphere, driven by photospheric motions, play a crucial role in the dynamics of solar magnetic fields. Many have been already identified such as fibrils, filament feet, and arch filament systems. Still, high resolution observations show a wealth of structures that remain elusive. We have observed a puzzling, unprecedented chromospheric fibril singularity in close vicinity of a blow-out solar jet and a flaring loop. We aim to understand the magnetic nature of this singularity and the cause of its activity using coordinated high- resolution multi-wavelengths observations. We aligned datasets from Solar Orbiter, SST, IRIS, and SDO. We re-projected the Solar Orbiter datasets to match the perspective of the Earth-based instruments. We performed potential field extrapolations from Solar Orbiter/PHI data. We analysed the spatial and temporal evolution of the plasma structures and their link with the surface magnetic field. This leads us to derive a model and scenario for the observed structures which we explain in a general schematic representation. We have discovered a new feature, a singularity in the chromospheric fibril pattern. It is formed in a weak magnetic field corridor between two flux concentrations of equal sign, at the base of a vertically inverted-Y shape field line pattern. In this specific case some activity develops along the structure. Firstly a flaring loop at one end, secondly a blow-out jet at the other end, where a coronal null-point was present and associated with a chromospheric saddle point being located onto the fibril singularity. The observations sugge

---

## 86. Prejudiced Futures? Algorithmic Bias in Time Series Forecasting and Its Ethical Implications

**论文链接:** [http://arxiv.org/abs/2512.01877v1](http://arxiv.org/abs/2512.01877v1)

**作者:** Bagattini Alexander, Chen Shao

**发布时间:** 2025-12-01

**备注:** 22 pages

### GPT解析

### 总结

本文探讨了时间序列预测算法中的算法偏见问题，分析了其伦理基础和缓解策略，提出了将公平性融入算法设计的三方面贡献。

### 背景

时间序列预测算法在高风险决策领域日益重要，但这些系统往往会继承和放大历史数据、问题规范和社会技术设计决策中存在的偏见。

### 目的

批判性地审视时间序列预测中算法偏见的伦理基础和缓解策略，提出将公平性作为算法设计核心要素的框架。

### 方法

提出三方面贡献：将算法偏见重新定义为社会技术现象；提供跨管道偏见源的系统性诊断，强调因果建模和包容性设计；倡导通过参与式治理和法律保障嵌入公平性；提出多指标、时间感知的公平评估方法。

### 主要发现

预测模型通过代理变量和反馈循环可复制结构不平等；公平性不应被视为与性能的权衡，而应作为负责任创新的共同要求。

### 结论

需要采用'设计伦理'方法，开发不仅有效适应性强，而且与民主价值观和社会公平保持一致的预测系统。

### 翻译

时间序列预测算法越来越成为医疗保健、能源管理和经济规划等高风险领域决策的核心。然而，这些系统往往继承了历史数据、有缺陷的问题规范和社会技术设计决策中存在的偏见并加以放大。本文批判性地审视了时间序列预测中算法偏见的伦理基础和缓解策略。我们概述了预测模型，特别是在时间动态领域，如何通过代理变量和反馈循环复制结构不平等和新兴歧视。本文提出了三方面的贡献：首先，它将算法偏见重新定义为植根于规范选择和制度约束的社会技术现象。其次，它提供了跨管道偏见源的系统性诊断，强调了对因果建模、可解释系统和包容性设计实践的需要。第三，它倡导通过参与式治理、利益相关者参与和具有法律约束力的保障措施来嵌入公平性的结构性改革。特别关注动态环境中的公平验证，提出了多指标、时间感知和上下文敏感的评估方法。最终，我们呼吁采用集成的'设计伦理'方法，将公平性视为与性能的权衡，而是负责任创新的共同要求。这一框架对于开发不仅有效和适应性强，而且与民主价值观和社会公平保持一致的预测系统至关重要。


### 论文摘要

Time series prediction algorithms are increasingly central to decision-making in high-stakes domains such as healthcare, energy management, and economic planning. Yet, these systems often inherit and amplify biases embedded in historical data, flawed problem specifications, and socio-technical design decisions. This paper critically examines the ethical foundations and mitigation strategies for algorithmic bias in time series prediction. We outline how predictive models, particularly in temporally dynamic domains, can reproduce structural inequalities and emergent discrimination through proxy variables and feedback loops. The paper advances a threefold contribution: First, it reframes algorithmic bias as a socio- technical phenomenon rooted in normative choices and institutional constraints. Second, it offers a structured diagnosis of bias sources across the pipeline, emphasizing the need for causal modeling, interpretable systems, and inclusive design practices. Third, it advocates for structural reforms that embed fairness through participatory governance, stakeholder engagement, and legally enforceable safeguards. Special attention is given to fairness validation in dynamic environments, proposing multi-metric, temporally-aware, and context- sensitive evaluation methods. Ultimately, we call for an integrated ethics-by-design approach that positions fairness not as a trade-off against performance, but as a co-requisite of responsible innovation. This framework is essential to developing predictive systems that are not only effective and adaptive but also aligned with democratic values and social equity.

---

## 87. COACH: Collaborative Agents for Contextual Highlighting - A Multi-Agent Framework for Sports Video Analysis

**论文链接:** [http://arxiv.org/abs/2512.01853v1](http://arxiv.org/abs/2512.01853v1)

**作者:** Tsz-To Wong, Ching-Chun Huang, Hong-Han Shuai

**发布时间:** 2025-12-01

**备注:** Accepted by AAAI 2026 Workshop LaMAS

### GPT解析

### 总结

该研究提出了一种可重新配置的多智能体系统(MAS)作为体育视频理解的基础框架，通过专门化的智能体解决现有端到端模型在时间层次理解上的局限性，实现了灵活、可扩展和可解释的体育视频分析。

### 背景

智能体育视频分析需要全面理解时间上下文，从微观动作到宏观游戏策略，而现有的端到端模型难以处理这种时间层次结构，导致解决方案泛化能力差、新任务开发成本高、可解释性差等问题。

### 目的

克服现有模型的局限性，提出一个可重新配置的多智能体系统(MAS)作为体育视频理解的基础框架。

### 方法

构建一个多智能体系统，每个智能体作为不同的'认知工具'专门负责分析的一个特定方面；系统架构不局限于单一的时间维度或任务；通过迭代调用和灵活组合这些智能体，构建自适应管道，用于短期分析推理和长期生成性总结。

### 主要发现

使用羽毛球分析中的两个代表性任务展示了该框架的适应性，框架能够弥合细粒度事件检测和全局语义组织之间的差距。

### 结论

这项工作朝着灵活、可扩展和可解释的系统方向发展，用于稳健的跨任务体育视频智能，代表了一种范式转变。

### 翻译

智能体育视频分析需要对时间上下文有全面的理解，从微观层面的动作到宏观层面的游戏策略。现有的端到端模型通常难以处理这种时间层次结构，提供的解决方案缺乏泛化能力，为新任务带来高昂的开发成本，并且可解释性差。为了克服这些局限性，我们提出了一个可重新配置的多智能体系统(MAS)作为体育视频理解的基础框架。在我们的系统中，每个智能体作为一个不同的'认知工具'，专门负责分析的特定方面。系统的架构不局限于单一的时间维度或任务。通过迭代调用和灵活组合这些智能体，我们的框架可以为短期分析推理(如回合问答)和长期生成性总结(如比赛摘要)构建自适应管道。我们使用羽毛球分析中的两个代表性任务展示了该框架的适应性，展示了它弥合细粒度事件检测和全局语义组织之间差距的能力。这项工作朝着灵活、可扩展和可解释的系统方向发展，用于稳健的跨任务体育视频智能。项目主页可在 https://aiden1020.github.io/COACH-project-page 获取。


### 论文摘要

Intelligent sports video analysis demands a comprehensive understanding of temporal context, from micro-level actions to macro-level game strategies. Existing end-to-end models often struggle with this temporal hierarchy, offering solutions that lack generalization, incur high development costs for new tasks, and suffer from poor interpretability. To overcome these limitations, we propose a reconfigurable Multi-Agent System (MAS) as a foundational framework for sports video understanding. In our system, each agent functions as a distinct "cognitive tool" specializing in a specific aspect of analysis. The system's architecture is not confined to a single temporal dimension or task. By leveraging iterative invocation and flexible composition of these agents, our framework can construct adaptive pipelines for both short-term analytic reasoning (e.g., Rally QA) and long-term generative summarization (e.g., match summaries). We demonstrate the adaptability of this framework using two representative tasks in badminton analysis, showcasing its ability to bridge fine-grained event detection and global semantic organization. This work presents a paradigm shift towards a flexible, scalable, and interpretable system for robust, cross-task sports video intelligence.The project homepage is available at https://aiden1020.github.io/COACH-project-page

---

## 88. PhyDetEx: Detecting and Explaining the Physical Plausibility of T2V Models

**论文链接:** [http://arxiv.org/abs/2512.01843v1](http://arxiv.org/abs/2512.01843v1)

**作者:** Zeqing Wang, Keze Wang, Lei Zhang

**发布时间:** 2025-12-01

**备注:** 17 pages, 8 figures

### GPT解析

### 总结

本研究评估了文本到视频生成模型对物理定律的理解和遵守能力，构建了PID数据集并提出了PhyDetEx方法，发现当前T2V模型在物理合理性方面仍有改进空间，尤其是开源模型。

### 背景

文本到视频生成模型在视频质量、长度和指令跟随能力方面取得了显著进展，但它们是否能够理解物理并生成物理上合理的视频仍然未知。视觉语言模型虽被广泛用作通用评估器，但难以识别生成视频中的物理上不可能的内容。

### 目的

研究T2V模型理解和遵守物理定律的能力，构建评估数据集并开发检测物理不合理内容的方法。

### 方法

构建了PID数据集，包含500个手动标注视频的测试集和2588对视频的训练集；引入轻量级微调方法使VLMs能检测物理不合理事件并生成文本解释；将微调后的VLM命名为PhyDetEx并基准测试了多个最先进T2V模型。

### 主要发现

尽管最近的T2V模型在生成物理上合理内容方面取得显著进展，但理解和遵守物理定律仍具挑战性，尤其对于开源模型。

### 结论

T2V模型在物理合理性方面仍有改进空间，研究提供了数据集、训练代码和检查点供进一步研究使用。

### 翻译

随着容量和训练规模的不断增长，文本到视频生成模型最近在视频质量、长度和指令跟随能力方面取得了实质性进展。然而，这些模型是否能够理解物理并生成物理上合理的视频仍然是一个问题。虽然视觉语言模型已被广泛用作各种应用中的通用评估器，但它们难以从生成的视频中识别出物理上不可能的内容。为了研究这个问题，我们构建了一个PID数据集，其中包括一个包含500个手动标注视频的测试集和一个包含2588对视频的训练集，其中每个不合理视频是通过仔细重写其对应真实世界视频的标题来生成的，以诱导T2V模型产生物理上不合理的内容。利用构建的数据集，我们引入了一种轻量级的微调方法，使VLMs不仅能够检测物理上不合理的事件，还能生成关于违反物理原理的文本解释。将微调后的VLM作为物理合理性和解释器，即PhyDetEx，我们对一系列最先进的T2V模型进行了基准测试，以评估它们对物理定律的遵守情况。我们的研究结果表明，尽管最近的T2V模型在生成物理上合理的内容方面取得了显著进展，但理解和遵守物理定律仍然是一个具有挑战性的问题，特别是对于开源模型。我们的数据集、训练代码和检查点可在https://github.com/Zeqing-Wang/PhyDetEx获取。


### 论文摘要

Driven by the growing capacity and training scale, Text-to-Video (T2V) generation models have recently achieved substantial progress in video quality, length, and instruction-following capability. However, whether these models can understand physics and generate physically plausible videos remains a question. While Vision-Language Models (VLMs) have been widely used as general-purpose evaluators in various applications, they struggle to identify the physically impossible content from generated videos. To investigate this issue, we construct a \textbf{PID} (\textbf{P}hysical \textbf{I}mplausibility \textbf{D}etection) dataset, which consists of a \textit{test split} of 500 manually annotated videos and a \textit{train split} of 2,588 paired videos, where each implausible video is generated by carefully rewriting the caption of its corresponding real-world video to induce T2V models producing physically implausible content. With the constructed dataset, we introduce a lightweight fine-tuning approach, enabling VLMs to not only detect physically implausible events but also generate textual explanations on the violated physical principles. Taking the fine-tuned VLM as a physical plausibility detector and explainer, namely \textbf{PhyDetEx}, we benchmark a series of state-of-the-art T2V models to assess their adherence to physical laws. Our findings show that although recent T2V models have made notable progress toward generating physically plausible content, understanding and adhering to physical laws remains a challenging issue, especially for open-source models. Our dataset, training code, and checkpoints are available at \href{https://github.com/Zeqing-Wang/PhyDetEx}{https://github.com/Zeqing-Wang/PhyDetEx}.

---

## 89. Seeing through Imagination: Learning Scene Geometry via Implicit Spatial World Modeling

**论文链接:** [http://arxiv.org/abs/2512.01821v1](http://arxiv.org/abs/2512.01821v1)

**作者:** Meng Cao, Haokun Lin, Haoyuan Li, Haoran Tang, Rongtao Xu, Dong An, Xue Liu, Ian Reid, Xiaodan Liang

**发布时间:** 2025-12-01

### GPT解析

### 总结

本文介绍了MILO，一种隐式空间世界建模范式，用于增强多模态大语言模型的空间推理能力。该方法通过视觉生成器提供几何感知反馈，并提出了RePE编码方案来捕获相对相机姿态变换。研究还构建了GeoGen数据集来支持训练，实验证明该方法显著提升了空间推理能力。

### 背景

空间推理（理解和解释世界3D结构的能力）是多模态大语言模型中关键但尚未充分发展的能力。当前方法主要依赖语言描述调优，存在视觉不识字问题，即它们仅通过文本符号学习空间概念，缺乏与视觉表现的连接。

### 目的

弥合语言符号与视觉表现之间的差距，增强MLLM的空间推理能力，使其能够模拟类似人类的空间想象力。

### 方法

引入MILO（隐式空间世界建模范式），模拟类似人类的空间想象力；集成视觉生成器提供几何感知反馈，将MLLM的符号推理隐式地锚定在感知经验上；提出RePE（相对位置编码），一种捕获相对相机姿态变换的新型编码方案；构建GeoGen数据集，包含约2,241个视频和67,827个观察-动作-结果三元组。

### 主要发现

MILO方法显著增强了多个基线和基准测试中的空间推理能力；RePE编码方案在性能上优于绝对坐标系统；提供了对3D空间更全面的理解。

### 结论

通过MILO范式和RePE编码方案，成功增强了MLLM的空间推理能力，使其能够更好地理解和解释3D空间结构。

### 翻译

空间推理，即理解和解释世界三维结构的能力，是多模态大语言模型中关键但尚未充分发展的能力。当前方法主要依赖语言描述调优，存在视觉不识字问题，即它们仅通过文本符号学习空间概念，缺乏与视觉表现的连接。为弥合这一差距，本文引入了MILO，一种模拟人类空间想象力的隐式空间世界建模范式。MILO集成了视觉生成器提供几何感知反馈，从而将MLLM的符号推理隐式地锚定在感知经验上。作为这一范式的补充，我们提出了RePE（相对位置编码），一种捕获相对相机姿态变换的新型编码方案，其在性能上优于绝对坐标系统。为支持训练，我们构建了GeoGen，一个包含约2,241个视频和67,827个观察-动作-结果三元组的大规模几何感知生成数据集。实验证明，我们的方法显著增强了多个基线和基准测试中的空间推理能力，提供了对3D空间更全面的理解。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': "论文主要解决多模态大语言模型（MLLMs）在空间推理能力方面的不足，特别是'视觉文盲'问题——模型仅通过文本符号学习空间概念，而没有将这些概念与视觉表现联系起来。这个问题很重要，因为空间推理是理解和解释世界3D结构的关键能力，对自动驾驶、具身导航和机器人操作等现实应用至关重要。当前MLLMs难以准确理解空间方向、距离和布局等概念，限制了它们在实际应用中的表现。", '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者观察到人类的空间认知是一个直观过程，通过想象和模拟空间结构将推理基于感知经验而非符号抽象。这启发他们设计了一个类似人类的空间想象方法。他们借鉴了现有工作：Ross3D的视觉生成调优思想，世界模型关于学习视觉动力学的概念，以及位置编码方法。作者首先识别了MLLMs的视觉文盲问题，然后结合视觉生成调优与语言描述调优，设计了相对位置编码（RePE）来处理相机姿态，并构建了GeoGen数据集支持训练。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是提出MILO（Implicit spatIaL wOrld modeling）范式，模拟人类的空间想象能力，通过整合视觉生成器提供几何感知反馈，将MLLM的符号推理隐式建立在感知经验基础上，并使用相对位置编码（RePE）捕获相对相机姿态变换。整体流程包括：1）相对位置编码：计算相邻帧间的相对几何变换并投影到高维表示；2）隐式空间世界建模：第一阶段进行视觉生成调优，使用视频扩散模型重建目标并提供视觉反馈；第二阶段进行语言微调，处理空间指令并生成响应；3）构建GeoGen数据集，包含观察-动作-结果三元组。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1）MILO范式：首次将视觉生成调优引入空间推理，通过视觉反馈连接符号推理与感知基础；2）相对位置编码（RePE）：捕获相对相机姿态变换，不依赖特定全局坐标系，提高泛化能力；3）GeoGen数据集：大规模几何感知生成数据集。相比之前工作，MILO结合了视觉生成调优而非仅使用语言描述调优；与Ross3D不同，MILO明确提供几何变换指令使模型感知底层几何变换；与依赖绝对3D坐标的方法不同，RePE提高了跨数据集和相机设置的泛化能力。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'MILO通过引入视觉生成调优和相对位置编码，使多模态大语言模型获得了类似人类的空间想象能力，显著提升了其在3D空间推理任务上的表现。'}


### 论文摘要

Spatial reasoning, the ability to understand and interpret the 3D structure of the world, is a critical yet underdeveloped capability in Multimodal Large Language Models (MLLMs). Current methods predominantly rely on verbal descriptive tuning, which suffers from visual illiteracy, i.e., they learn spatial concepts through textual symbols alone, devoid of connection to their visual manifestations. To bridge this gap, this paper introduces MILO, an Implicit spatIaL wOrld modeling paradigm that simulates human-like spatial imagination. MILO integrates a visual generator to provide geometry-aware feedback, thereby implicitly grounding the MLLM's symbolic reasoning in perceptual experience. Complementing this paradigm, we propose RePE (Relative Positional Encoding), a novel encoding scheme that captures relative camera-pose transformations, offering superior performance over absolute coordinate systems. To support the training, we construct GeoGen, a large-scale Geometry-aware Generative dataset with approximately 2,241 videos and 67,827 observation-action-outcome triplets. Experiments demonstrate that our approach significantly enhances spatial reasoning capabilities across multiple baselines and benchmarks, offering a more holistic understanding of 3D space.

---

## 90. Forget Less, Retain More: A Lightweight Regularizer for Rehearsal-Based Continual Learning

**论文链接:** [http://arxiv.org/abs/2512.01818v1](http://arxiv.org/abs/2512.01818v1)

**作者:** Lama Alssum, Hasan Abed Al Kader Hammoud, Motasem Alfarra, Juan C Leon Alcazar, Bernard Ghanem

**发布时间:** 2025-12-01

### GPT解析

### 总结

本文提出了一种名为信息最大化(IM)正则化的新方法，用于解决深度神经网络中的灾难性遗忘问题。该方法基于期望的标签分布，与类别无关，可直接集成到各种基于重放的持续学习方法中，减少遗忘并促进更快收敛。

### 背景

深度神经网络面临灾难性遗忘问题，即在训练新任务后，模型在先前任务上的性能会下降。这是因为模型倾向于用新信息覆盖已获取的知识。

### 目的

提出一种新方法来应对灾难性遗忘挑战，专注于基于记忆的方法和正则化方法的交叉点。

### 方法

提出了一种名为信息最大化(IM)正则化的正则化策略，专门用于基于记忆的持续学习方法。这种策略仅基于期望的标签分布，因此与类别无关，可直接集成到各种基于重放的持续学习方法中。

### 主要发现

1)在多个数据集上，无论任务数量多少，IM正则化策略都能一致性地提高基线性能，且计算开销最小；2)IM的轻量级特性确保它保持实用和可扩展，适用于对效率要求高的实际持续学习场景；3)通过在视频数据上应用，证明了IM正则化的数据无关性，尽管视频数据带来额外挑战，它仍能提高视频持续学习方法的性能。

### 结论

IM正则化是一种有效解决灾难性遗忘问题的方法，具有计算效率高、可扩展性强、适用于多种数据类型等优点，是实际持续学习场景中的实用解决方案。

### 翻译

深度神经网络遭受灾难性遗忘的困扰，即在训练新任务后，在先前任务上的性能会下降。这个问题是由于模型倾向于用新信息覆盖先前获取的知识而产生的。我们提出了一种新颖的方法来应对这一挑战，专注于基于记忆的方法和正则化方法的交叉点。我们为基于记忆的持续学习方法制定了一种正则化策略，称为信息最大化(IM)正则化，它完全基于期望的标签分布，因此与类别无关。因此，IM正则化可以直接集成到各种基于重放的持续学习方法中，减少遗忘并促进更快收敛。我们的经验验证表明，在多个数据集上，无论任务数量如何，我们提出的正则化策略都能以最小的计算开销一致性地提高基线性能。IM的轻量级特性确保它保持实用和可扩展，使其适用于效率至关重要的实际持续学习场景。最后，我们通过将其应用于视频数据，展示了我们正则化的数据无关性，视频数据由于其时间结构和更高的内存需求而带来额外挑战。尽管存在显著的领域差距，我们的实验表明，IM正则化也能提高视频持续学习方法的性能。


### 论文摘要

Deep neural networks suffer from catastrophic forgetting, where performance on previous tasks degrades after training on a new task. This issue arises due to the model's tendency to overwrite previously acquired knowledge with new information. We present a novel approach to address this challenge, focusing on the intersection of memory-based methods and regularization approaches. We formulate a regularization strategy, termed Information Maximization (IM) regularizer, for memory-based continual learning methods, which is based exclusively on the expected label distribution, thus making it class-agnostic. As a consequence, IM regularizer can be directly integrated into various rehearsal-based continual learning methods, reducing forgetting and favoring faster convergence. Our empirical validation shows that, across datasets and regardless of the number of tasks, our proposed regularization strategy consistently improves baseline performance at the expense of a minimal computational overhead. The lightweight nature of IM ensures that it remains a practical and scalable solution, making it applicable to real-world continual learning scenarios where efficiency is paramount. Finally, we demonstrate the data-agnostic nature of our regularizer by applying it to video data, which presents additional challenges due to its temporal structure and higher memory requirements. Despite the significant domain gap, our experiments show that IM regularizer also improves the performance of video continual learning methods.

---

## 91. Envision: Benchmarking Unified Understanding & Generation for Causal World Process Insights

**论文链接:** [http://arxiv.org/abs/2512.01816v1](http://arxiv.org/abs/2512.01816v1)

**作者:** Juanxi Tian, Siyuan Li, Conghui He, Lijun Wu, Cheng Tan

**发布时间:** 2025-12-01

**备注:** 35 pages, 12 figures, 10 tables

### GPT解析

### 总结

论文提出Envision因果事件进展基准和Envision-Score评估指标，用于评估多模态模型在链式文本到多图像生成任务上的表现。研究评估了15个模型，发现统一多模态模型在因果叙事连贯性方面优于专门T2I模型，但仍面临时空一致性挑战。

### 背景

当前多模态模型通过统一理解和生成超越单一模态限制，使用文本到图像任务校准语义一致性。然而，训练和评估中依赖静态单图像生成，导致过拟合静态模式匹配，阻碍了建模动态过程的能力。

### 目的

解决模型过度依赖静态单图像生成的问题，提出新评估框架以评估模型理解和生成随时间发展的因果事件序列的能力。

### 方法

提出Envision基准，基于世界知识和时空因果关系构建，包含1000个四阶段提示；引入Envision-Score指标整合多维度一致性、物理性和美学性；评估15个模型（10个专门T2I模型和5个统一模型）。

### 主要发现

专门T2I模型美学渲染出色但缺乏世界知识；统一多模态模型在因果叙事连贯性上优于专门模型；统一架构仍无法超越闭源模型且难以克服时空一致性挑战；专注静态图像阻碍多帧推理和生成。

### 结论

专注于因果隔离的单图像阻碍多帧推理和生成，促进静态模式匹配而非动态世界建模，最终限制世界知识的内化和生成能力。

### 翻译

当前的多模态模型旨在通过统一理解和生成来超越单一模态表示的局限性，通常使用文本到图像（T2I）任务来校准语义一致性。然而，它们在训练和评估中依赖于静态的单图像生成，导致对静态模式匹配和语义融合的过拟合，同时从根本上阻碍了它们建模随时间展开的动态过程的能力。为了解决这些限制，我们提出了Envision——一个用于链式文本到多图像生成的因果事件进展基准。它基于世界知识并由时空因果关系构建，重新组织了现有的评估维度，包含1000个跨越六个科学和人文学科领域的四阶段提示。为了将评估从单图像转变为连续帧，并评估模型是否真正内化了世界知识同时遵守因果时间约束，我们引入了Envision-Score，一个整合了多维度一致性、物理性和美学性的整体指标。对15个模型（10个专门的T2I模型，5个统一模型）的全面评估发现：专门的T2I模型在美学渲染方面表现出色，但缺乏内在的世界知识。统一的多模态模型弥补了这一差距，在因果叙事连贯性方面一致优于专门的模型。然而，即使是这些统一架构仍然无法超越闭源模型，并且难以克服时空一致性的核心挑战。这表明，专注于因果隔离的单图像阻碍了多帧推理和生成，促进了静态模式匹配而非动态世界建模——最终限制了世界知识的内化和生成。


### 论文摘要

Current multimodal models aim to transcend the limitations of single-modality representations by unifying understanding and generation, often using text-to-image (T2I) tasks to calibrate semantic consistency. However, their reliance on static, single-image generation in training and evaluation leads to overfitting to static pattern matching and semantic fusion, while fundamentally hindering their ability to model dynamic processes that unfold over time. To address these constraints, we propose Envision-a causal event progression benchmark for chained text-to-multi-image generation. Grounded in world knowledge and structured by spatiotemporal causality, it reorganizes existing evaluation dimensions and includes 1,000 four-stage prompts spanning six scientific and humanities domains. To transition evaluation from single images to sequential frames and assess whether models truly internalize world knowledge while adhering to causal-temporal constraints, we introduce Envision-Score, a holistic metric integrating multi-dimensional consistency, physicality, and aesthetics. Comprehensive evaluation of 15 models (10 specialized T2I models, 5 unified models) uncovers: specialized T2I models demonstrate proficiency in aesthetic rendering yet lack intrinsic world knowledge. Unified multimodal models bridge this gap, consistently outperforming specialized counterparts in causal narrative coherence. However, even these unified architectures remain subordinate to closed-source models and struggle to overcome the core challenge of spatiotemporal consistency. This demonstrates that a focus on causally-isolated single images impedes multi-frame reasoning and generation, promoting static pattern matching over dynamic world modeling-ultimately limiting world knowledge internalization, generation.

---

## 92. Evaluating SAM2 for Video Semantic Segmentation

**论文链接:** [http://arxiv.org/abs/2512.01774v1](http://arxiv.org/abs/2512.01774v1)

**作者:** Syed Hesham Syed Ariff, Yun Liu, Guolei Sun, Jing Yang, Henghui Ding, Xue Geng, Xudong Jiang

**发布时间:** 2025-12-01

**备注:** 17 pages, 3 figures and 7 tables

### GPT解析

### 总结

本研究探讨了如何将SAM2模型扩展到视频语义分割(VSS)任务，提出了两种主要方法并评估了其效果。

### 背景

SAM2是一个强大的基础模型，可用于图像和视频中可提示的视觉对象分割，能够存储对象感知的记忆并通过记忆块在时间上传递。虽然SAM2在视频对象分割方面表现出色，但将其扩展到密集视频语义分割(VSS)仍面临挑战。

### 目的

探索将SAM2扩展到VSS的方法，关注两种主要方法，并在这个过程中的一手观察和常见挑战。

### 方法

第一种方法使用SAM2从图像中提取唯一对象作为掩码，同时使用分割网络并行生成和细化初始预测；第二种方法使用预测的掩码提取特征向量，输入到简单网络中进行分类，然后将分类结果和掩码组合产生最终分割。

### 主要发现

实验表明，利用SAM2可以增强VSS的整体性能，主要归因于SAM2对对象边界的精确预测。

### 结论

SAM2在VSS任务中具有潜力，其精确的对象边界预测能力可以提高整体分割性能。

### 翻译

分割任意模型2(SAM2)已被证明是图像和视频中可提示的视觉对象分割的强大基础模型，能够存储对象感知的记忆并通过记忆块在时间上传递这些记忆。虽然SAM2通过基于提示提供密集分割掩码在视频对象分割方面表现出色，但由于需要空间准确性、时间一致性以及跟踪具有复杂边界和不同尺度多个对象的能力，将其扩展到密集视频语义分割(VSS)仍存在挑战。本文探讨了SAM2在VSS方面的扩展，重点关注两种主要方法，并强调了在这个过程中的一手观察和常见挑战。第一种方法涉及使用SAM2从给定图像中提取唯一对象作为掩码，同时使用分割网络并行生成和细化初始预测。第二种方法利用预测的掩码提取唯一特征向量，然后将其输入到简单网络中进行分类。随后将分类结果和掩码组合以产生最终分割。我们的实验表明，利用SAM2可以增强VSS的整体性能，主要归因于其对对象边界的精确预测。


### 论文摘要

The Segmentation Anything Model 2 (SAM2) has proven to be a powerful foundation model for promptable visual object segmentation in both images and videos, capable of storing object-aware memories and transferring them temporally through memory blocks. While SAM2 excels in video object segmentation by providing dense segmentation masks based on prompts, extending it to dense Video Semantic Segmentation (VSS) poses challenges due to the need for spatial accuracy, temporal consistency, and the ability to track multiple objects with complex boundaries and varying scales. This paper explores the extension of SAM2 for VSS, focusing on two primary approaches and highlighting firsthand observations and common challenges faced during this process. The first approach involves using SAM2 to extract unique objects as masks from a given image, with a segmentation network employed in parallel to generate and refine initial predictions. The second approach utilizes the predicted masks to extract unique feature vectors, which are then fed into a simple network for classification. The resulting classifications and masks are subsequently combined to produce the final segmentation. Our experiments suggest that leveraging SAM2 enhances overall performance in VSS, primarily due to its precise predictions of object boundaries.

---

## 93. VideoScoop: A Non-Traditional Domain-Independent Framework For Video Analysis

**论文链接:** [http://arxiv.org/abs/2512.01769v1](http://arxiv.org/abs/2512.01769v1)

**作者:** Hafsa Billah

**发布时间:** 2025-12-01

**备注:** This is a report submitted as part of PhD proposal defense of Hafsa Billah

### GPT解析

### 总结

该研究提出了一种通用的视频情况分析(VSA)框架，能够自动理解视频内容并识别有意义的活动或情况，克服了现有方法需要人工干预或针对特定情况设计专用算法的局限性。

### 背景

自动理解视频内容对公民监控、一般监控和辅助生活等应用很重要。虽然图像和视频分析研究已推进内容提取任务，但识别有意义的活动或情况仍然困难，现有VSA方法要么依赖人工，要么需要为每种新情况开发专用算法。

### 目的

开发一个通用的VSA框架，能够处理不同领域和类型的视频情况，无需为每种新情况或新领域设计特定算法。

### 方法

使用最先进的视频内容提取技术提取内容，并通过两种模型表示：扩展关系模型(R++)和图模型。R++支持连续查询处理，图模型则检测难以用关系模型发现的情况。通过参数化模板实现领域独立性，支持跨领域的基本情况变体识别。

### 主要发现

在辅助生活、公民监控和一般监控三个领域的多个有趣情况下进行的实验表明，该框架能够有效检测各种情况，具有良好的准确性、效率和鲁棒性。

### 结论

所提出的通用VSA框架克服了现有方法的局限性，通过结合关系模型和图模型的优势，实现了对不同领域和类型视频情况的自动化分析，无需为每种新情况开发专用算法。

### 翻译

自动理解视频内容对于公民监控、一般监控和辅助生活等几个应用很重要。数十年的图像和视频分析研究已推进了内容提取等任务。识别有意义的活动或情况仍然困难，仅靠内容提取无法实现。目前视频情况分析通过人工进行或通过针对特定视频类型或情况设计的自定义算法。这些算法不是通用目的的，对于每种新情况或来自新领域的视频需要新的算法/软件。本报告提出了一个通用的VSA框架，克服了上述局限性。视频内容使用最先进的视频内容提取技术提取一次。它们使用两种替代模型表示——扩展关系模型(R++)和图模型。当使用R++表示时，提取的内容可用作数据流，通过提出的视频分析连续查询语言实现连续查询处理。图模型通过检测使用关系模型难以或不可能检测的情况来补充这一点。现有图算法和新开发的算法支持各种情况检测。为支持领域独立性，识别了跨领域的基本情况变体并将其表示为参数化模板。在AL、CM和SL三个领域的多个有趣情况下进行了广泛实验，使用这些领域不同长度的视频数据集评估了所提出方法的准确性、效率和鲁棒性。


### 论文摘要

Automatically understanding video contents is important for several applications in Civic Monitoring (CM), general Surveillance (SL), Assisted Living (AL), etc. Decades of Image and Video Analysis (IVA) research have advanced tasks such as content extraction (e.g., object recognition and tracking). Identifying meaningful activities or situations (e.g., two objects coming closer) remains difficult and cannot be achieved by content extraction alone. Currently, Video Situation Analysis (VSA) is done manually with a human in the loop, which is error-prone and labor-intensive, or through custom algorithms designed for specific video types or situations. These algorithms are not general-purpose and require a new algorithm/software for each new situation or video from a new domain.   This report proposes a general-purpose VSA framework that overcomes the above limitations. Video contents are extracted once using state-of-the-art Video Content Extraction technologies. They are represented using two alternative models -- the extended relational model (R++) and graph models. When represented using R++, the extracted contents can be used as data streams, enabling Continuous Query Processing via the proposed Continuous Query Language for Video Analysis. The graph models complement this by enabling the detection of situations that are difficult or impossible to detect using the relational model alone. Existing graph algorithms and newly developed algorithms support a wide variety of situation detection. To support domain independence, primitive situation variants across domains are identified and expressed as parameterized templates. Extensive experiments were conducted across several interesting situations from three domains -- AL, CM, and SL-- to evaluate the accuracy, efficiency, and robustness of the proposed approach using a dataset of videos of varying lengths from these domains.

---

## 94. Beyond Scaffold: A Unified Spatio-Temporal Gradient Tracking Method

**论文链接:** [http://arxiv.org/abs/2512.01732v1](http://arxiv.org/abs/2512.01732v1)

**作者:** Yan Huang, Jinming Xu, Jiming Chen, Karl Henrik Johansson

**发布时间:** 2025-12-01

**备注:** 13 pages

### GPT解析

### 总结

本文提出了一种时空梯度跟踪算法ST-GT，用于解决分布式和联邦学习中的通信开销问题，同时处理数据异质性和梯度噪声问题。

### 背景

在分布式和联邦学习算法中，通常通过在通信轮次之间执行多次本地更新来减少通信开销，但这种方法可能导致局部模型偏离全局最优解。

### 目的

解决由于数据异质性和局部梯度噪声导致的模型漂移问题，提高分布式学习算法的收敛性和通信效率。

### 方法

重新审视Scaffold联邦学习方法，从梯度跟踪角度出发，提出统一的时空梯度跟踪算法ST-GT，用于时变图上的分布式随机优化。ST-GT跟踪相邻节点间的全局梯度减轻数据异质性，同时保持局部梯度的运行平均值抑制噪声。

### 主要发现

ST-GT在不假设有界数据异质性的情况下，对强凸问题实现线性收敛速率，对非凸问题实现次线性收敛速率；首次实现关于每个轮次本地更新数量的线性通信复杂度加速；将拓扑相关噪声项从σ²降低到σ²/τ，提高了通信效率。

### 结论

ST-GT算法有效解决了分布式学习中的数据异质性和噪声问题，实现了更好的收敛性能和通信效率，存储开销略有增加。

### 翻译

在分布式和联邦学习算法中，通常通过在通信轮次之间执行多次本地更新来减少通信开销。然而，由于节点间的数据异质性和每个节点内的局部梯度噪声，这种策略可能导致局部模型偏离全局最优。为了解决这个问题，我们从梯度跟踪的角度重新审视了著名的联邦学习方法Scaffold（Karimireddy等人，2020），并提出了一种统一的时空梯度跟踪算法ST-GT，用于时变图上的分布式随机优化。ST-GT跟踪相邻节点间的全局梯度以减轻数据异质性，同时保持局部梯度的运行平均值以显著抑制噪声，存储开销略有增加。在不假设有界数据异质性的情况下，我们证明了ST-GT对于强凸问题能够获得线性收敛速率，对于非凸情况能够获得次线性收敛速率。值得注意的是，ST-GT在强凸设置中首次实现了关于每个轮次本地更新数量τ的线性通信复杂度加速。与传统梯度跟踪方法相比，ST-GT将拓扑相关的噪声项从σ²降低到σ²/τ，其中σ²表示噪声水平，从而提高了通信效率。


### 论文摘要

In distributed and federated learning algorithms, communication overhead is often reduced by performing multiple local updates between communication rounds. However, due to data heterogeneity across nodes and the local gradient noise within each node, this strategy can lead to the drift of local models away from the global optimum. To address this issue, we revisit the well-known federated learning method Scaffold (Karimireddy et al., 2020) under a gradient tracking perspective, and propose a unified spatio-temporal gradient tracking algorithm, termed ST-GT, for distributed stochastic optimization over time-varying graphs. ST-GT tracks the global gradient across neighboring nodes to mitigate data heterogeneity, while maintaining a running average of local gradients to substantially suppress noise, with slightly more storage overhead. Without assuming bounded data heterogeneity, we prove that ST-GT attains a linear convergence rate for strongly convex problems and a sublinear rate for nonconvex cases. Notably, ST-GT achieves the first linear speed-up in communication complexity with respect to the number of local updates per round $τ$ for the strongly-convex setting. Compared to traditional gradient tracking methods, ST-GT reduces the topology-dependent noise term from $σ^2$ to $σ^2/τ$, where $σ^2$ denotes the noise level, thereby improving communication efficiency.

---

## 95. StreamGaze: Gaze-Guided Temporal Reasoning and Proactive Understanding in Streaming Videos

**论文链接:** [http://arxiv.org/abs/2512.01707v1](http://arxiv.org/abs/2512.01707v1)

**作者:** Daeun Lee, Subhojyoti Mukherjee, Branislav Kveton, Ryan A. Rossi, Viet Dac Lai, Seunghyun Yoon, Trung Bui, Franck Dernoncourt, Mohit Bansal

**发布时间:** 2025-12-01

**备注:** Project page: https://streamgaze.github.io/

### GPT解析

### 总结

该研究引入了StreamGaze，第一个用于评估多模态大语言模型在流式视频中如何有效使用视线信号进行时序和前瞻性推理的基准测试。研究发现在所有任务中，先进模型与人类性能存在显著差距，揭示了当前模型在基于视线的时序推理、意图建模和前瞻性预测方面的局限性。

### 背景

流式视频理解不仅需要模型处理时间上连续传入的帧，还需预测用户意图，适用于AR眼镜等实际应用。虽然之前的流式基准测试评估了时序推理能力，但没有衡量MLLMs是否能在流式环境中解释或利用人类视线信号。

### 目的

填补这一空白，引入StreamGaze作为第一个基准测试，用于评估MLLMs在流式视频中如何有效地使用视线进行时序和前瞻性推理。

### 方法

StreamGaze引入了视线引导的过去、现在和前瞻性任务，全面评估流式视频理解能力。研究团队开发了视线-问答生成管道，通过注视提取、特定区域的视觉提示和扫描路径构建，将第一人称视频与原始视线轨迹对齐，产生时空定位的问答对。

### 主要发现

在所有StreamGaze任务中，观察到最先进的MLLMs与人类性能之间存在显著差距，揭示了基于视线的时序推理、意图建模和前瞻性预测方面的基本局限性。研究还提供了视线提示策略、推理行为和特定任务失败模式的详细分析。

### 结论

当前MLLMs在视线引导的流式视频理解方面存在基本限制，未来的模型需要发展基于视线的时序推理、意图建模和前瞻性预测能力。所有数据和代码将公开发布，以支持视线引导的流式视频理解的持续研究。

### 翻译

流式视频理解不仅需要模型处理时间上连续传入的帧，还需要预测用户意图，适用于AR眼镜等实际应用。虽然之前的流式基准测试评估了时序推理能力，但没有衡量MLLMs是否能在流式环境中解释或利用人类视线信号。为填补这一空白，我们引入StreamGaze，这是第一个基准测试，用于评估MLLMs在流式视频中如何有效地使用视线进行时序和前瞻性推理。StreamGaze引入了视线引导的过去、现在和前瞻性任务，全面评估流式视频理解能力。这些任务评估模型是否能使用实时视线来跟随转移的注意力，并仅从过去和当前观察到的帧中推断用户意图。为构建StreamGaze，我们开发了一个视线-问答生成管道，通过注视提取、特定区域的视觉提示和扫描路径构建，将第一人称视频与原始视线轨迹对齐。该管道产生时空定位的问答对，反映人类感知动态。在所有StreamGaze任务中，我们观察到最先进的MLLMs与人类性能之间存在显著差距，揭示了基于视线的时序推理、意图建模和前瞻性预测方面的基本局限性。我们进一步提供了视线提示策略、推理行为和特定任务失败模式的详细分析，深入了解当前MLLMs的不足以及未来模型必须发展的能力。所有数据和代码将公开发布，以支持视线引导的流式视频理解的持续研究。


### 论文摘要

Streaming video understanding requires models not only to process temporally incoming frames, but also to anticipate user intention for realistic applications like AR glasses. While prior streaming benchmarks evaluate temporal reasoning, none measure whether MLLMs can interpret or leverage human gaze signals within a streaming setting. To fill this gap, we introduce StreamGaze, the first benchmark designed to evaluate how effectively MLLMs use gaze for temporal and proactive reasoning in streaming videos. StreamGaze introduces gaze-guided past, present, and proactive tasks that comprehensively evaluate streaming video understanding. These tasks assess whether models can use real-time gaze to follow shifting attention and infer user intentions from only past and currently observed frames. To build StreamGaze, we develop a gaze-video QA generation pipeline that aligns egocentric videos with raw gaze trajectories via fixation extraction, region-specific visual prompting, and scanpath construction. This pipeline produces spatio-temporally grounded QA pairs that closely reflect human perceptual dynamics. Across all StreamGaze tasks, we observe substantial performance gaps between state-of-the-art MLLMs and human performance, revealing fundamental limitations in gaze-based temporal reasoning, intention modeling, and proactive prediction. We further provide detailed analyses of gaze-prompting strategies, reasoning behaviors, and task-specific failure modes, offering deeper insight into why current MLLMs struggle and what capabilities future models must develop. All data and code will be publicly released to support continued research in gaze-guided streaming video understanding.

---

## 96. Revisiting Direct Encoding: Learnable Temporal Dynamics for Static Image Spiking Neural Networks

**论文链接:** [http://arxiv.org/abs/2512.01687v1](http://arxiv.org/abs/2512.01687v1)

**作者:** Huaxu He

**发布时间:** 2025-12-01

### GPT解析

### 总结

本研究解决了脉冲神经网络(SNNs)处理静态图像的挑战，重新审视了直接编码和基于速率编码之间的性能差距，并提出了一个最小可学习的时序编码方法来从静态输入中诱导有意义的时变变化。

### 背景

静态图像缺乏内在的时间动态性，这是脉冲神经网络面临的一个基本挑战。在直接训练的SNNs中，静态输入通常在时间步上重复，导致时间维度坍缩为类似速率的表示，从而阻止有意义的时序建模。

### 目的

本研究旨在重新评估直接编码和基于速率编码之间报告的性能差距，并揭示这种差距的根本原因，同时提出一种新的编码方法来改善静态输入在SNNs中的处理。

### 方法

作者引入了一个最小可学习的时序编码，通过添加自适应相位偏移来从静态输入中诱导有意义的时变变化。这种方法机制上澄清了卷积可学习性和替代梯度公式对性能的影响。

### 主要发现

研究发现直接编码和基于速率编码之间的性能差距主要源于卷积可学习性和替代梯度公式，而非编码方案本身。这为理解SNNs处理静态输入提供了新的见解。

### 结论

通过机制层面的澄清和提出的新编码方法，本研究解决了SNNs处理静态图像的挑战，为改进静态输入在SNNs中的处理提供了新的思路和方法。

### 翻译

处理缺乏固有时间动态性的静态图像对脉冲神经网络(SNNs)来说仍然是一个基本挑战。在直接训练的SNNs中，静态输入通常在时间步上重复，导致时间维度坍缩为类似速率的表示，从而阻止有意义的时序建模。这项工作重新审视了直接编码和基于速率编码之间报告的性能差距，表明这种差距主要源于卷积可学习性和替代梯度公式，而非编码方案本身。为了说明这种机制层面的澄清，我们引入了一个最小可学习的时序编码，它添加自适应相位偏移，从静态输入中诱导有意义的时变变化。


### 论文摘要

Handling static images that lack inherent temporal dynamics remains a fundamental challenge for spiking neural networks (SNNs). In directly trained SNNs, static inputs are typically repeated across time steps, causing the temporal dimension to collapse into a rate like representation and preventing meaningful temporal modeling. This work revisits the reported performance gap between direct and rate based encodings and shows that it primarily stems from convolutional learnability and surrogate gradient formulations rather than the encoding schemes themselves. To illustrate this mechanism level clarification, we introduce a minimal learnable temporal encoding that adds adaptive phase shifts to induce meaningful temporal variation from static inputs.

---

## 97. Improved Disease Outbreak Detection from Out-of-sequence measurements Using Markov-switching Fixed-lag Particle Filters

**论文链接:** [http://arxiv.org/abs/2512.01639v1](http://arxiv.org/abs/2512.01639v1)

**作者:** Conor Rosato, Joshua Murphy, Siân E. Jenkins, Paul Horridge, Alessandro Varsi, Martyn Bull, Alessandro Gerada, Alex Howard, Veronica Bowman, Simon Maskell

**发布时间:** 2025-12-01

**备注:** 23 Pages

### GPT解析

### 总结

本文提出了一种马尔可夫转换固定延迟粒子滤波器(FL-PF)，用于处理疾病监测中的乱序测量数据，提高了爆发检测的准确性和及时性，减少了误报。

### 背景

粒子滤波器已成为疾病监测的重要工具，能够估计非线性非高斯模型中的隐藏流行病状态。在现实监测系统中，数据经常延迟到达或时间顺序错误，产生乱序(OOS)测量值，而现有PF方法在完全调整过去潜在轨迹方面存在局限性。

### 目的

开发一种能够更好地处理乱序测量的粒子滤波方法，提高疾病监测中爆发检测的准确性和及时性。

### 方法

提出马尔可夫转换固定延迟粒子滤波器(FL-PF)，在用户指定的延迟窗口内重新采样粒子轨迹，允许乱序测量值回溯更新状态和模型估计。通过明确重新评估历史样本改进检测性能，并在框架内计算对数似然，使用顺序蒙特卡洛平方(SMC²)进行参数估计。

### 主要发现

FL-PF通过重新评估历史样本，提高了爆发检测的准确性和及时性，减少了误报。能够计算对数似然，支持参数估计。

### 结论

这些贡献扩展了粒子滤波器在常见回顾性数据的监测系统中的适用性，为监测疾病爆发和参数推断提供了更强大的框架。

### 翻译

粒子滤波器(PFs)已成为疾病监测的基本工具，因为它们可以估计非线性非高斯模型中的隐藏流行病状态。在流行病学建模中，人口动态可能由不同状态（如地方病或爆发阶段）控制，这些状态可以用马尔可夫转换状态空间模型表示。在许多现实监测系统中，数据经常延迟到达或时间顺序错误，产生与过去时间点而非当前时间点相关的乱序(OOS)测量值。虽然现有PF方法可以通过粒子重加权纳入乱序测量值，但这些方法在完全调整过去潜在轨迹方面的能力有限。为此，我们引入了一种马尔可夫转换固定延迟粒子滤波器(FL-PF)，它在用户指定的延迟窗口内重新采样粒子轨迹，允许乱序测量值回溯更新状态和模型估计。通过明确重新评估历史样本，FL-PF提高了爆发检测的准确性和及时性，减少了误报。我们还展示了如何在FL-PF框架内计算对数似然，使用顺序蒙特卡洛平方(SMC²)进行参数估计。这些贡献共同扩展了PFs在常见回顾性数据的监测系统中的适用性，为监测疾病爆发和参数推断提供了更强大的框架。


### 论文摘要

Particle filters (PFs) have become an essential tool for disease surveillance, as they can estimate hidden epidemic states in nonlinear and non-Gaussian models. In epidemic modelling, population dynamics may be governed by distinct regimes such as endemic or outbreak phases which can be represented using Markov-switching state-space models. In many real-world surveillance systems, data often arrives with delays or in the wrong temporal order, producing out-of-sequence (OOS) measurements that pertain to past time points rather than the current one. While existing PF methods can incorporate OOS measurements through particle reweighting, these approaches are limited in their ability to fully adjust past latent trajectories. To address this, we introduce a Markov-switching fixed-lag particle filter (FL-PF) that resimulates particle trajectories within a user-specified lag window, allowing OOS measurements to retroactively update both state and model estimates. By explicitly reevaluating historical samples, the FL-PF improves the accuracy and timeliness of outbreak detection and reduces false alarms. We also show how to compute the log-likelihood within the FL-PF framework, enabling parameter estimation using Sequential Monte Carlo squared (SMC$^2$). Together, these contributions extend the applicability of PFs to surveillance systems where retrospective data are common, offering a more robust framework for monitoring disease outbreaks and parameter inference.

---

## 98. Exploring Scavenging Strategies and Cognitive Problem-Solving in Indian Free-Ranging Dogs

**论文链接:** [http://arxiv.org/abs/2512.01637v1](http://arxiv.org/abs/2512.01637v1)

**作者:** Tuhin Subhra Pal, Srijaya Nandi, Hindoli Gope, Aniket Malakar, Rohan Sarkar, Sagarika Biswas, Anindita Bhadra

**发布时间:** 2025-12-01

**备注:** 5 figures

### GPT解析

### 总结

该研究探讨了印度流浪狗在面临污染食物时的觅食策略，发现它们采用灵活的多层次策略来平衡营养收益与风险，展现了在人类主导环境中的生态适应性。

### 背景

动物在自然环境中会权衡营养益处与厌恶性或有害刺激带来的风险。在印度，流浪狗主要依赖人类产生的废弃物为食，经常遇到被柠檬汁等令人不快的物质污染的食物。

### 目的

了解流浪狗应对食物污染挑战的策略，理解它们在人类主导环境中的生态适应性和生存能力。

### 方法

在西孟加拉邦Nadia区的15个地点随机测试了156只成年流浪狗，让它们接触分别浸泡在柠檬汁、稀释柠檬溶液或水中的鸡肉食物，通过录像记录并分析它们的行为序列。

### 主要发现

流浪狗使用灵活的、多方面的策略处理相对不太可口的食物，通常避免最不可口的食物选择；它们展现出层次结构和上下文依赖的觅食策略，由感官评估、风险-回报平衡和行为灵活性推动。

### 结论

城市清道夫能够在令人厌恶的条件下动态适应觅食行为，这体现了支持它们在人类主导环境中生存的认知机制。

### 翻译

动物在其自然环境中进行战略决策时，会仔细权衡营养益处与厌恶性或有害刺激带来的风险，以最大化觅食效率。在印度，流浪狗主要依靠人类产生的废弃物生存，在觅食过程中经常遇到被柠檬汁等令人不快或有毒物质污染的食物。这些狗应对此类挑战的策略仍知之甚少，但对于理解它们在人类主导环境中的生态适应性和生存能力至关重要。研究者在西孟加拉邦Nadia区的15个地点随机测试了156只成年流浪狗。每只个体被暴露在含有鸡肉的食物源前，这些鸡肉分别浸泡在柠檬汁、稀释的柠檬溶液或水中。所有试验都被录像记录，狗的行为序列包括嗅闻、舔舐、进食和食物操作都被编码分析，以量化在不可口条件下的战略觅食反应。研究发现，狗使用灵活的、多方面的策略来处理相对不太可口的食物选择，并通常避免最不可口的选择，以最大化它们的获取选择。总体而言，这项研究揭示了流浪狗层次结构和上下文依赖的觅食策略，由感官评估、风险-回报平衡和行为灵活性推动。这些发现展示了城市清道夫如何在令人厌恶的条件下动态适应觅食行为，强调了支持它们在人类主导环境中生存的认知机制。


### 论文摘要

Animals employ strategic decision-making while carefully weighing nutritional benefits against the risks presented by aversive or harmful stimuli in their natural environment, to maximize foraging efficiency, In India, free-ranging dogs subsist predominantly on human-generated waste, where they often encounter food contaminated with unpalatable or noxious substances such as lemon juice while scavenging. The strategies these dogs use to navigate such challenges remain poorly understood, yet are critical for understanding their ecological adaptability and survival in human-dominated environments. A total of 156 randomly encountered free-ranging adult dogs were tested across 15 sites in Nadia district, West Bengal. Each individual was exposed to a single food source containing chicken placed in either lemon juice, diluted lemon solution, or water. All trials were video-recorded, and the behavioural sequences of the dogs, including sniffing, licking, eating, and food manipulation were coded and analysed to quantify strategic foraging responses under unpalatable conditions. They were found to use a flexible, multi-pronged strategy to manipulate the comparatively less palatable food option, and typically avoid the most unpalatable one, to maximize their acquiring options. Overall, this study revealed a hierarchically structured and context-dependent foraging strategy of free-ranging dogs, propelled by sensory evaluation, risk-reward balancing, and behavioural flexibility. These findings demonstrated how urban scavengers dynamically adapt to aversive conditions while scavenging, underscoring the cognitive mechanisms that support their survival in human-dominated environments.

---

## 99. BlinkBud: Detecting Hazards from Behind via Sampled Monocular 3D Detection on a Single Earbud

**论文链接:** [http://arxiv.org/abs/2512.01366v1](http://arxiv.org/abs/2512.01366v1)

**作者:** Yunzhe Li, Jiajun Yan, Yuzhou Wei, Kechen Liu, Yize Zhao, Chong Zhang, Hongzi Zhu, Li Lu, Shan Chang, Minyi Guo

**发布时间:** 2025-12-01

**备注:** This is the author-accepted version of the paper published in Proceedings of the ACM on Interactive, Mobile, Wearable and Ubiquitous Technologies (IMWUT), Vol. 9, No. 4, Article 191, 2025. Final published version: https://doi.org/10.1145/3770707

### GPT解析

### 总结

本文提出了一种名为BlinkBud的系统，利用单个耳机和配对手机检测用户后方接近的危险物体，具有低功耗和高准确性特点。

### 背景

未能注意到从后方接近的超速车辆对行人和骑行者的道路安全构成巨大威胁。

### 目的

开发一个系统来在线检测用户后方接近的危险物体，提高道路安全性。

### 方法

使用耳机拍摄少量采样图像跟踪物体；设计结合卡尔曼滤波轨迹估计和强化学习图像采样的3D跟踪算法；利用俯仰和偏航角修正深度估计并校正坐标系，消除头部移动影响。

### 主要发现

实现BlinkBud原型系统；耳机功耗29.8 mW，手机功耗702.6 mW；假阳性率4.90%，假阴性率1.47%，能够准确检测危险。

### 结论

BlinkBud是一种轻量级、低功耗的有效解决方案，能提高行人和骑行者的道路安全。

### 翻译

未能注意到从后方接近的超速车辆对行人和骑行者的道路安全构成巨大威胁。在本文中，我们提出了BlinkBud，它利用单个耳机和配对的手机在线检测用户后方接近的危险物体。核心思想是利用从耳机拍摄的小量采样图像来准确跟踪视觉识别的物体。为了最小化耳机和手机的功耗同时保证最佳跟踪精度，设计了一种新颖的3D物体跟踪算法，结合了基于卡尔曼滤波的轨迹估计方案和基于强化学习的最优图像采样策略。此外，通过利用估计的俯仰和偏航角分别修正物体深度估计并将相机坐标系与用户身体坐标系对齐，显著消除了用户头部持续移动对跟踪精度的影响。我们实现了BlinkBud系统原型并进行了大量真实世界实验。结果表明，BlinkBud轻量级，耳机和智能手机的平均功耗极低，分别为29.8 mW和702.6 mW，并且能够以4.90%和1.47%的低平均假阳性率和假阴性率准确检测危险。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决行人和骑行者无法察觉从后方接近的危险车辆（尤其是安静的电动汽车）的安全隐患。这个问题非常重要，因为根据美国国家公路交通安全管理局数据，2022年美国有7522名行人和1105名骑自行车者死亡；使用耳机等设备时行走或骑自行车会增加碰撞风险；电动汽车的安静特性使传统声音预警方法失效，因此需要一种新的解决方案来提高道路安全。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': "作者首先分析了现有方法的局限性：基础设施方法成本高且覆盖有限，无线方法依赖车辆兼容模块，声学方法对电动汽车无效。然后设计了一个结合耳塞摄像头和智能手机的系统，采用间歇性'眨眼'机制而非连续视频流。作者借鉴了YOLOv5s进行2D目标检测，结合单目深度估计和相机模型进行3D检测，使用卡尔曼滤波器进行目标跟踪，并创新性地应用强化学习优化采样策略。针对头部运动问题，作者利用IMU数据进行坐标系统校正，实现了在资源受限设备上的高效检测。", '这个方法的核心思想是什么？整体实现流程是怎样的？': "核心思想是利用配备摄像头的耳塞和配对智能手机，通过间歇性'眨眼'（捕获图像并分析）检测后方危险物体，同时最小化功耗。整体流程包括：1)接收耳塞的灰度图像和IMU数据；2)进行2D目标检测并利用俯仰角估计深度信息；3)将3D坐标从相机系转换到用户系；4)用卡尔曼滤波跟踪物体轨迹；5)基于强化学习决定何时进行新采样；6)评估碰撞时间并在风险过高时触发警报。整个过程平衡了准确性和能效，解决了头部运动带来的干扰问题。", '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)轻量级3D目标检测方案，使用灰度图像和优化的深度估计；2)结合卡尔曼滤波和强化学习的自适应采样策略；3)利用IMU数据校正头部运动影响；4)完整的原型系统和真实世界验证。相比之前工作，BlinkBud不依赖基础设施或车辆通信模块，不受电动汽车安静特性影响，采用间歇性采样大幅降低功耗，并通过IMU校正解决了头部运动问题，实现了在资源受限设备上的高效后方危险检测。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'BlinkBud通过在耳塞上实现轻量级3D目标检测和智能采样策略，结合智能手机处理，有效解决了行人和骑行者无法感知后方接近车辆的安全隐患，同时显著降低了系统功耗。'}


### 论文摘要

Failing to be aware of speeding vehicles approaching from behind poses a huge threat to the road safety of pedestrians and cyclists. In this paper, we propose BlinkBud, which utilizes a single earbud and a paired phone to online detect hazardous objects approaching from behind of a user. The core idea is to accurately track visually identified objects utilizing a small number of sampled camera images taken from the earbud. To minimize the power consumption of the earbud and the phone while guaranteeing the best tracking accuracy, a novel 3D object tracking algorithm is devised, integrating both a Kalman filter based trajectory estimation scheme and an optimal image sampling strategy based on reinforcement learning. Moreover, the impact of constant user head movements on the tracking accuracy is significantly eliminated by leveraging the estimated pitch and yaw angles to correct the object depth estimation and align the camera coordinate system to the user's body coordinate system, respectively. We implement a prototype BlinkBud system and conduct extensive real-world experiments. Results show that BlinkBud is lightweight with ultra-low mean power consumptions of 29.8 mW and 702.6 mW on the earbud and smartphone, respectively, and can accurately detect hazards with a low average false positive ratio (FPR) and false negative ratio (FNR) of 4.90% and 1.47%, respectively.

---

## 100. OpenBox: Annotate Any Bounding Boxes in 3D

**论文链接:** [http://arxiv.org/abs/2512.01352v1](http://arxiv.org/abs/2512.01352v1)

**作者:** In-Jae Lee, Mungyeom Kim, Kwonyoung Ryu, Pierre Musacchio, Jaesik Park

**发布时间:** 2025-12-01

**备注:** Accepted by NeurIPS 2025

### GPT解析

### 总结

本文提出了一种名为OpenBox的两阶段自动标注流程，用于无监督和开放词汇3D目标检测，能够生成高质量的3D边界框标注而不需要自训练。

### 背景

无监督和开放词汇3D目标检测在自动驾驶领域受到关注，减少标注成本和识别未见对象对安全性和可扩展性至关重要。

### 目的

解决现有方法中统一标注3D边界框、忽略物体物理状态、需要多次自训练迭代导致的次优质量和大量计算开销问题。

### 方法

OpenBox利用2D视觉基础模型，通过两阶段流程：第一阶段进行跨模态实例对齐，将2D图像中的实例级线索与3D点云关联；第二阶段按刚性和运动状态分类实例，使用类别特定尺寸统计生成自适应边界框。

### 主要发现

在Waymo开放数据集、Lyft Level 5感知数据集和nuScenes数据集上的实验表明，OpenBox与基线方法相比提高了准确性和效率。

### 结论

OpenBox能够产生高质量的3D边界框标注，无需自训练，在多个数据集上优于现有方法。

### 翻译

无监督和开放词汇3D目标检测最近受到关注，特别是在自动驾驶领域，减少标注成本和识别未见对象对安全性和可扩展性至关重要。然而，大多数现有方法统一标注3D边界框，忽略物体的物理状态，并需要多次自训练迭代来优化标注，导致次优质量和大量计算开销。为解决这些挑战，我们提出OpenBox，一种利用2D视觉基础模型的两阶段自动标注流程。在第一阶段，OpenBox通过跨模态实例对齐，将视觉基础模型处理的2D图像中的实例级线索与对应的3D点云关联。在第二阶段，它按刚性和运动状态对实例分类，然后使用类别特定的尺寸统计生成自适应边界框。结果，OpenBox产生高质量的3D边界框标注，无需自训练。在Waymo开放数据集、Lyft Level 5感知数据集和nuScenes数据集上的实验表明，与基线相比提高了准确性和效率。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决无监督和开放词汇表3D目标检测中的挑战，特别是降低标注成本和识别未见对象的需求。这个问题在自动驾驶领域至关重要，因为3D目标检测直接关系到系统的安全性和可靠性，而创建大规模3D标注数据集是主要瓶颈。LiDAR点云虽提供精确几何结构，但缺乏语义上下文，难以与文本对齐或手动标注。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者分析了现有方法不考虑物体物理状态、需要多次自训练迭代的局限性，借鉴了2D视觉基础模型（如Grounding DINO和SAM2）的能力，参考了PP分数和SDF等现有技术。设计思路是通过两阶段管道：首先跨模态对齐将2D实例信息与3D点云关联，然后根据物体物理类型自适应生成边界框，避免了自训练需求。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是利用2D视觉基础模型生成高质量实例信息作为监督信号，通过跨模态对齐与3D点云关联，并根据物体物理属性自适应生成3D边界框。整体流程分为两阶段：1)跨模态实例对齐——提取2D实例信息并反投影到3D点云，应用上下文感知细化；2)自适应3D边界框生成——根据物理类型（静态刚体、动态刚体、可变形物体）选择不同策略生成边界框。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)两阶段自动标注管道；2)两点云细化过程（上下文感知细化和表面感知噪声过滤）；3)物理状态特定的边界框生成；4)无需自训练。相比纯LiDAR方法，OpenBox利用2D视觉模型提供语义信息并考虑物体物理状态；相比多模态方法，它在输出前进行几何对齐而非简单融合，更充分利用视觉线索提高3D标注质量。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'OpenBox提出了一种创新的无需自训练的两阶段自动标注方法，通过结合2D视觉基础模型和考虑物体物理状态的自适应边界框生成，显著提高了3D目标检测的标注质量和效率，使系统能够识别任意类别的3D物体并降低标注成本。'}


### 论文摘要

Unsupervised and open-vocabulary 3D object detection has recently gained attention, particularly in autonomous driving, where reducing annotation costs and recognizing unseen objects are critical for both safety and scalability. However, most existing approaches uniformly annotate 3D bounding boxes, ignore objects' physical states, and require multiple self-training iterations for annotation refinement, resulting in suboptimal quality and substantial computational overhead. To address these challenges, we propose OpenBox, a two-stage automatic annotation pipeline that leverages a 2D vision foundation model. In the first stage, OpenBox associates instance-level cues from 2D images processed by a vision foundation model with the corresponding 3D point clouds via cross-modal instance alignment. In the second stage, it categorizes instances by rigidity and motion state, then generates adaptive bounding boxes with class-specific size statistics. As a result, OpenBox produces high-quality 3D bounding box annotations without requiring self-training. Experiments on the Waymo Open Dataset, the Lyft Level 5 Perception dataset, and the nuScenes dataset demonstrate improved accuracy and efficiency over baselines.

---

## 101. Programmable Switching of Molecular Transitions via Plasmonic Toroidal Nanoantennae

**论文链接:** [http://arxiv.org/abs/2512.01303v1](http://arxiv.org/abs/2512.01303v1)

**作者:** Arda Gulucu, Emre Ozan Polat

**发布时间:** 2025-12-01

**备注:** 18 Pages, 4 Figures

### GPT解析

### 总结

该研究展示了通过环状等离子体纳米天线实现量子对象分子转换能量的完全切换，具有高调制深度和辐射增强效果，为光谱检测和量子计算提供了新方法。

### 背景

等离子体纳米天线能够通过环矩聚焦大量三维局部电场，同时允许在量子发射器周围进行前后定位，为从生物传感器到量子计算的各种应用提供了机会。

### 目的

研究等离子体环状纳米天线如何实现对量子对象分子转换能量的完全控制，展示其在光谱检测和量子模式开关方面的应用潜力。

### 方法

使用优化的环状纳米天线几何结构，通过等离子体连续谱与量子对象窄量子跃迁之间的法诺干涉，抑制辐射和非辐射衰减通道，实现能量捕获在混合模式中。

### 主要发现

实现了量子对象分子转换能量的完全切换，调制深度达99.9%，辐射增强2840倍；优化几何结构下，法诺干涉在850纳米附近同时抑制辐射和非辐射衰减通道；在多个量子对象系统中，简并性增强了透明带宽，失谐则产生不同的最小值，实现了可单独寻址的光谱响应。

### 结论

等离子体环状纳米天线是单分子或多分子构型光谱检测的有前景架构，具有高灵敏度，并为实现用于光子处理的量子模式开关提供了能力。

### 翻译

通过确定性定位的等离子体纳米天线切换和编程分子转换的能力，从生物传感器到量子计算等广泛应用领域提供了机会。由于其拓扑结构，环状纳米天线通过环矩聚焦大量三维局部电场，同时允许在量子发射器周围进行前后定位。在此，我们报道了量子对象分子转换能量的完全切换，调制深度为99.9%，辐射增强2840倍。在优化的几何结构下，等离子体连续谱与量子对象窄量子跃迁之间的法诺干涉在850纳米附近同时抑制辐射和非辐射衰减通道，产生可观察到的完全切换，将能量捕获在混合模式中而不是重新发射。为展示该概念的潜力，我们进一步演示了具有多个量子对象的系统，其中简并性增强了透明带宽，而失谐则产生不同的最小值，实现了可单独寻址的光谱响应。这些结果将等离子体环状纳米天线确立为具有高灵敏度的单分子或多分子构型光谱检测的有前景架构，并为用户实现用于光子处理的量子模式开关提供了能力。


### 论文摘要

The ability to switch and program molecular transitions via deterministically located plasmonic nanoantennae presents opportunities for wide spectrum of applications from biosensors to quantum computing. Due to its topology, toroidal nanoantenna (TNA) focuses immense amount of three-dimensional (3D) local electric field by toroidal moment while allowing pre and post positioning around quantum emitters (QEs). Here, we report complete switching of molecular transition energies of quantum objects (QOs) with modulation depth of 99.9% over 2840-fold radiative enhancement. At optimized TNA geometries, Fano interference between the broadband plasmonic continuum and narrow quantum transitions of QOs suppresses both radiative and non-radiative decay channels near 850 nm, yielding an observable full switching that traps energy within the hybrid mode instead of re-emitting it. To show the promises of the concept, we further demonstrate systems with multiple QOs where spectral degeneracy enhances the transparency bandwidth, while detuning generates distinct minima, enabling individually addressable spectral responses. These results establish plasmonic TNA as a promising architecture for spectral detection of single or multi-molecule configurations with high sensitivity and empowers the user for the implementation of quantum mode switches to be used in photonic processing.

---

## 102. VSRD++: Autolabeling for 3D Object Detection via Instance-Aware Volumetric Silhouette Rendering

**论文链接:** [http://arxiv.org/abs/2512.01178v1](http://arxiv.org/abs/2512.01178v1)

**作者:** Zihua Liu, Hiroki Sakuma, Masatoshi Okutomi

**发布时间:** 2025-12-01

**备注:** arXiv admin note: text overlap with arXiv:2404.00149

### GPT解析

### 总结

本文提出了一种名为VSRD++的新型弱监督框架，用于单目3D目标检测，该框架消除了对3D标注的依赖，并利用基于神经场的体积渲染和弱2D监督。该框架包含多视图3D自动标注和单目3D检测器训练两个阶段。在KITTI-360数据集上的实验表明，VSRD++在静态和动态场景中显著优于现有的弱监督方法。

### 背景

单目3D目标检测是3D场景理解中的基础且具有挑战性的任务。现有方法严重依赖监督学习，需要大量通过激光雷达点云经过密集标注过程获取的3D标注数据，这些标注过程劳动强度大且成本高昂。

### 目的

提出一种新的弱监督框架VSRD++，消除对3D标注的依赖，并利用基于神经场的体积渲染和弱2D监督进行单目3D目标检测。

### 方法

VSRD++采用两阶段流程：1）多视图3D自动标注阶段，将物体表面表示为符号距离场(SDFs)，通过实例感知的体积轮廓渲染为实例掩码，并将SDF分解为立方体SDF和残差距离场(RDF)优化3D边界框；2）处理动态物体的几何不一致性，通过包含速度、分配伪标签置信度和使用3D属性初始化模块；3）单目3D目标检测阶段，使用优化的3D边界框作为伪标签训练检测器。

### 主要发现

在KITTI-360数据集上的大量实验表明，VSRD++在静态和动态场景中显著优于现有的弱监督单目3D目标检测方法。

### 结论

VSRD++是一种有效的弱监督框架，能够在不依赖3D标注的情况下进行单目3D目标检测，代码已开源。

### 翻译

单目3D目标检测是3D场景理解中的一个基础但具有挑战性的任务。现有方法严重依赖监督学习，需要大量从激光雷达点云通过劳动密集型标注过程获取的3D标注。为了解决这个问题，我们提出了VSRD++，一种用于单目3D目标检测的新型弱监督框架，它消除了对3D标注的依赖，并利用基于神经场的体积渲染和弱2D监督。VSRD++包含一个两阶段流程：多视图3D自动标注和后续的单目3D检测器训练。在多视图自动标注阶段，物体表面被表示为符号距离场(SDFs)，并通过提出的实例感知的体积轮廓渲染为实例掩码。为了优化3D边界框，我们将每个实例的SDF分解为一个立方体SDF和一个捕捉与立方体偏差的残差距离场(RDF)。为了解决体积渲染方法应用于动态物体时常见的几何不一致性问题，我们通过将速度包含在边界框属性中以及为每个伪标签分配置信度来建模动态物体。此外，我们还采用了一个3D属性初始化模块来初始化动态边界框参数。在单目3D目标检测阶段，优化的3D边界框作为伪标签用于训练单目3D目标检测器。在KITTI-360数据集上的大量实验表明，VSRD++在静态和动态场景的单目3D目标检测弱监督方法上显著优于现有方法。代码可在https://github.com/Magicboomliu/VSRD_plus_plus获取。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决单目3D目标检测中依赖昂贵且耗时的3D标注问题。现有方法严重依赖基于LiDAR点云的3D标注，这些标注需要大量人工工作，限制了3D目标检测技术的发展和实际应用。这个问题在现实中很重要，因为自动驾驶等应用需要精确的3D感知能力，而高昂的标注成本阻碍了技术的广泛应用和迭代。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者从2D标注比3D标注更丰富、更便宜的事实出发，受到基于神经场的体积渲染技术(如NeRF、NeuS)的启发，这些技术能从多视图2D图像中重建3D场景而不需要3D标签。作者将3D检测任务转化为表面重建任务，使用有符号距离场(SDF)表示物体表面。借鉴了NeuS的SDF体积渲染方法，但创新性地设计了实例感知的体积轮廓渲染和SDF分解技术。同时借鉴了弱监督学习思想，利用2D监督替代3D监督，并引入深度估计技术用于3D属性初始化。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是利用多视图图像和2D实例掩码作为弱监督，通过体积渲染生成高质量的3D边界框伪标签，从而训练单目3D目标检测器。整体流程分为两个阶段：第一阶段是多视图3D自动标注，使用SDF表示物体表面，分解为立方体SDF和残差距离场(RDF)，通过实例感知的体积轮廓渲染比较渲染掩码与真实掩码来优化3D边界框，并针对动态物体引入速度属性和3D属性初始化；第二阶段是单目3D目标检测器训练，使用优化的3D边界框作为伪标签，并为每个伪标签分配置信度分数作为训练权重。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1) VSRD++框架，第一个基于体积渲染的弱监督3D目标检测框架；2) 实例感知的体积轮廓渲染，能渲染每个实例轮廓并考虑几何关系；3) SDF分解技术，将SDF分解为立方体SDF和残差距离场；4) 动态物体处理，引入速度属性和动态掩码；5) 置信度分配方法，提高伪标签质量。相比之前的工作，VSRD++完全不需要LiDAR点云或3D标注，仅使用2D实例掩码；与VSRD相比，VSRD++显式扩展边界框包含速度属性，构建时变SDF建模动态物体；与基于神经场的方法相比，专为3D目标检测设计而非场景重建。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'VSRD++提出了一种创新的弱监督单目3D目标检测框架，通过实例感知的体积轮廓渲染技术，仅利用多视图图像和2D实例掩码生成高质量的3D边界框伪标签，显著降低了3D标注依赖，并在静态和动态场景中均取得了优异性能。'}


### 论文摘要

Monocular 3D object detection is a fundamental yet challenging task in 3D scene understanding. Existing approaches heavily depend on supervised learning with extensive 3D annotations, which are often acquired from LiDAR point clouds through labor-intensive labeling processes. To tackle this problem, we propose VSRD++, a novel weakly supervised framework for monocular 3D object detection that eliminates the reliance on 3D annotations and leverages neural-field-based volumetric rendering with weak 2D supervision. VSRD++ consists of a two-stage pipeline: multi-view 3D autolabeling and subsequent monocular 3D detector training. In the multi-view autolabeling stage, object surfaces are represented as signed distance fields (SDFs) and rendered as instance masks via the proposed instance-aware volumetric silhouette rendering. To optimize 3D bounding boxes, we decompose each instance's SDF into a cuboid SDF and a residual distance field (RDF) that captures deviations from the cuboid. To address the geometry inconsistency commonly observed in volume rendering methods applied to dynamic objects, we model the dynamic objects by including velocity into bounding box attributes as well as assigning confidence to each pseudo-label. Moreover, we also employ a 3D attribute initialization module to initialize the dynamic bounding box parameters. In the monocular 3D object detection phase, the optimized 3D bounding boxes serve as pseudo labels for training monocular 3D object detectors. Extensive experiments on the KITTI-360 dataset demonstrate that VSRD++ significantly outperforms existing weakly supervised approaches for monocular 3D object detection on both static and dynamic scenes. Code is available at https://github.com/Magicboomliu/VSRD_plus_plus

---

## 103. Words into World: A Task-Adaptive Agent for Language-Guided Spatial Retrieval in AR

**论文链接:** [http://arxiv.org/abs/2512.00294v1](http://arxiv.org/abs/2512.00294v1)

**作者:** Lixing Guo, Tobias Höllerer

**发布时间:** 2025-11-29

### GPT解析

### 总结

提出了一种整合多模态大语言模型与视觉模型的模块化AR智能体系统，能够处理复杂自然语言查询，构建动态场景图，并支持人在回路的交互理解。

### 背景

传统增强现实系统主要依赖固定类别检测器或标记物，在解释复杂、开放词汇的自然语言查询方面能力有限。

### 目的

开发一个模块化AR智能体系统，整合多模态大语言模型与视觉模型，实现空间关系推理和语言条件下的空间检索。

### 方法

通过自适应任务智能体协调MLLMs和坐标感知工具处理不同复杂度查询；构建动态AR场景图编码九种类型关系；使用任务自适应区域高亮和上下文空间检索；动态调用坐标感知工具处理复杂查询。

### 主要发现

系统能理解物体在3D空间中的存在、关联和互动；能引导人类注意力到信息密集区域；支持人在回路的细化；模块化架构支持即插即用视觉语言模型无需重新训练。

### 结论

AR智能体可作为中介，通过现实世界空间智能增强MLLMs用于交互场景理解；引入了GroundedAR-Bench评估框架用于跨环境评估。

### 翻译

传统增强现实系统主要依赖固定类别检测器或标记物，限制了它们解释复杂、开放词汇自然语言查询的能力。我们提出了一种模块化AR智能体系统，将多模态大语言模型与视觉模型结合，实现空间关系推理和物理环境中语言条件下的空间检索。自适应任务智能体协调MLLMs和坐标感知工具，处理从简单物体识别到多物体关系推理的不同复杂度查询，同时返回米级精确的3D锚点。它构建动态AR场景图，编码九种类型关系（空间、结构语义、因果功能），使MLLMs不仅理解存在什么物体，还理解它们如何在3D空间中关联和互动。通过任务自适应感兴趣区域高亮和上下文空间检索，系统引导人类注意力到信息密集区域，同时支持人在回路的细化。智能体动态调用坐标感知工具处理复杂查询-选择、测量、比较和执行，将语言理解与物理操作相结合。模块化架构支持即插即用视觉语言模型无需重新训练，确立AR智能体为通过现实世界空间智能增强MLLMs用于交互场景理解的中介。我们还引入了GroundedAR-Bench，一个用于跨不同环境语言驱动现实世界定位和关系基础的评估框架。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': "这篇论文主要解决传统增强现实(AR)系统无法有效处理复杂、开放词汇自然语言查询的问题。当前AR系统主要依赖固定类别的检测器或标记，无法理解用户如'桌子上的哪个盒子在工具箱后面但比打印机离我更近'这样的复杂空间查询。这个问题很重要，因为它限制了AR设备真正理解物理世界并与之交互的能力，而用户期望AR系统能回答关于周围环境的空间问题，不仅仅是显示悬浮的窗口。", '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者认识到现有多模态大语言模型(MLLMs)主要基于2D理解，缺乏几何感知能力；而3D感知框架又需要密集扫描或离线处理。作者借鉴了开放词汇检测器(如Grounding DINO)、场景图表示法和任务适应控制器等现有技术，但采用了独特思路：保持MLLM在2D+语言领域，同时通过深度感知的2D到3D接地模块弥补其空间盲点。设计了一个分层代理系统，包括感知执行层、工具层、世界模型层和编排层，实现了模块化架构，支持即插即用的视觉-语言模型。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是创建一个任务自适应的AR代理系统，将多模态大语言模型与基于深度的感知相结合，支持开放词汇、关系查询，并将2D检测结果提升到精确的3D坐标。系统构建动态AR场景图，编码九种类型的关系(空间、结构-语义、因果-功能)，使系统能理解对象在3D空间中如何存在、关联和交互。整体流程包括：1)捕获RGB、深度和地理信息；2)对话代理解析自然语言查询并制定计划；3)使用MLLM提出候选对象标签；4)通过2D检测器获取边界框；5)通过深度射线投射将2D检测提升到3D锚点；6)构建场景图并推断对象关系；7)生成AR覆盖层和多模态反馈；8)维护会话和跨会话记忆。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)任务自适应AR代理系统，统一了MLLM语义与深度感知；2)轻量级3D场景图，避免了昂贵的网格重建；3)语言条件关系推理，支持复杂查询；4)GroundedAR-Bench评估框架。相比之前工作，不同之处在于：与2D VLMs不同，本文提供精确3D定位而非仅2D理解；与3D重建方法不同，避免了全局重建需求；与放置导向方法不同，专注于查询现有物理场景而非放置虚拟内容；与其他AR代理系统不同，更紧密结合MLLM与深度感知，并将场景表示为显式关系图。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': "这篇论文提出了'Words into World'系统，一个任务自适应的AR代理，通过结合多模态大语言模型与深度感知的3D接地，实现了开放词汇、关系查询的实时增强现实理解，并构建了动态场景图来编码对象间的空间和语义关系。"}


### 论文摘要

Traditional augmented reality (AR) systems predominantly rely on fixed class detectors or fiducial markers, limiting their ability to interpret complex, open-vocabulary natural language queries. We present a modular AR agent system that integrates multimodal large language models (MLLMs) with grounded vision models to enable relational reasoning in space and language-conditioned spatial retrieval in physical environments. Our adaptive task agent coordinates MLLMs and coordinate-aware perception tools to address varying query complexities, ranging from simple object identification to multi-object relational reasoning, while returning meter-accurate 3D anchors. It constructs dynamic AR scene graphs encoding nine typed relations (spatial, structural-semantic, causal-functional), enabling MLLMs to understand not just what objects exist, but how they relate and interact in 3D space. Through task-adaptive region-of-interest highlighting and contextual spatial retrieval, the system guides human attention to information-dense areas while supporting human-in-the-loop refinement. The agent dynamically invokes coordinate-aware tools for complex queries-selection, measurement, comparison, and actuation-grounding language understanding in physical operations. The modular architecture supports plug-and-use vision-language models without retraining, establishing AR agents as intermediaries that augment MLLMs with real-world spatial intelligence for interactive scene understanding. We also introduce GroundedAR-Bench, an evaluation framework for language-driven real world localization and relation grounding across diverse environments.

---

## 104. PEFT-DML: Parameter-Efficient Fine-Tuning Deep Metric Learning for Robust Multi-Modal 3D Object Detection in Autonomous Driving

**论文链接:** [http://arxiv.org/abs/2512.00060v1](http://arxiv.org/abs/2512.00060v1)

**作者:** Abdolazim Rezaei, Mehdi Sookhak

**发布时间:** 2025-11-23

### GPT解析

### 总结

本研究介绍了PEFT-DML，一种用于自动驾驶中鲁棒多模态3D目标检测的参数高效深度度量学习框架。

### 背景

传统模型假设固定的传感器可用性，而实际情况中传感器可能会失效或出现未见过的模态组合。

### 目的

开发一个能够在传感器失效或未见过的模态组合情况下仍能可靠检测目标的框架。

### 方法

将多种模态（LiDAR、雷达、摄像头、IMU、GNSS）映射到共享的潜在空间，并集成低秩适应（LoRA）和适配器层，实现训练效率的同时提高对快速运动、天气变化和域变化的鲁棒性。

### 主要发现

在nuScenes基准测试上展示了优越的准确性。

### 结论

PEFT-DML是一种有效的框架，能够在传感器失效或未见过的模态组合情况下实现可靠的多模态3D目标检测。

### 翻译

本研究介绍了PEFT-DML，一种用于自动驾驶中鲁棒多模态3D目标检测的参数高效深度度量学习框架。与假设固定传感器可用性的传统模型不同，PEFT-DML将多种模态（LiDAR、雷达、摄像头、IMU、GNSS）映射到共享的潜在空间，即使在传感器失效或未见过的模态组合情况下也能实现可靠检测。通过集成低秩适应（LoRA）和适配器层，PEFT-DML在显著提高训练效率的同时，增强了对快速运动、天气变化和域变化的鲁棒性。在nuScenes基准测试上的实验证明了其优越的准确性。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决自动驾驶中的鲁棒多模态3D目标检测问题，特别是在传感器缺失或模态不可用情况下的检测能力。这个问题在现实中非常重要，因为自动驾驶系统需要应对各种挑战，如传感器故障、恶劣天气条件、快速运动场景等。传统方法假设所有传感器都可用，但实际驾驶环境中传感器可能失效或数据质量下降，因此需要一种能够适应不同传感器组合和条件的鲁棒检测方法来确保安全。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先分析了现有方法的局限性，如3ML-DML需要固定模态可用性，CRKD在传感器失效时很脆弱，RoboFusion计算效率低等。作者借鉴了低秩适应(LoRA)和适配器层的参数高效微调方法，以及深度度量学习技术。设计思路是将不同传感器模态(LiDAR、雷达、相机、IMU、GNSS)映射到共享潜在空间，保持主干编码器冻结，只微调轻量级层，并通过多目标损失函数(检测损失、度量对齐损失和一致性损失)来训练模型，使其能够处理传感器缺失和模态变化的情况。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是创建一个统一的潜在空间，将不同传感器模态映射到这个共享空间，使用参数高效的微调方法处理模态差异，实现零样本跨模态检测。整体流程为：1)各模态通过冻结的主干编码器处理；2)每个模态通过投影头映射到标准化的嵌入向量；3)通过交叉注意力和门控模块融合不同模态的嵌入；4)检测头从融合特征中预测3D边界框和类别；5)使用多目标损失函数进行训练，包括检测损失、度量对齐损失(三元组损失)和一致性损失(时间稳定性和跨模态一致性)。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)参数高效微调，结合LoRA和适配器层，只微调不到10%的参数即可达到或超过全微调性能；2)共享潜在空间，实现零样本跨模态检测；3)在传感器缺失和不同天气条件下保持高鲁棒性；4)多目标训练策略，结合检测、度量和一致性损失。相比之前工作的不同：与3ML-DML相比，PEFT-DML不需要固定模态可用性；与CRKD相比，支持任何传感器子集；与RoboFusion相比，通过轻量级微调实现相当鲁棒性；与Chae等人的方法相比，在部分模态缺失时仍能工作。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'PEFT-DML通过将多种传感器模态统一到共享潜在空间并采用参数高效微调方法，实现了在传感器缺失和恶劣天气条件下仍能保持高精度的自动驾驶3D目标检测，同时大幅降低了训练成本。'}


### 论文摘要

This study introduces PEFT-DML, a parameter-efficient deep metric learning framework for robust multi-modal 3D object detection in autonomous driving. Unlike conventional models that assume fixed sensor availability, PEFT-DML maps diverse modalities (LiDAR, radar, camera, IMU, GNSS) into a shared latent space, enabling reliable detection even under sensor dropout or unseen modality class combinations. By integrating Low-Rank Adaptation (LoRA) and adapter layers, PEFT-DML achieves significant training efficiency while enhancing robustness to fast motion, weather variability, and domain shifts. Experiments on benchmarks nuScenes demonstrate superior accuracy.

---

## 105. Data-Centric Visual Development for Self-Driving Labs

**论文链接:** [http://arxiv.org/abs/2512.02018v1](http://arxiv.org/abs/2512.02018v1)

**作者:** Anbang Liu, Guanzhong Hu, Jiayi Wang, Ping Guo, Han Liu

**发布时间:** 2025-12-01

**备注:** 11 pages, 4 figures

### GPT解析

### 总结

该研究提出了一种混合数据生成方法，结合真实数据和虚拟数据来解决自动驾驶实验室中移液操作训练数据稀缺的问题，实现了高精度的气泡检测模型训练。

### 背景

自动驾驶实验室为减少生物科学中劳动密集型、耗时且通常不可重复的工作流程提供了前景，但其需要高度稳健的模型，这些模型的训练依赖于大量标注数据，而这种数据在日常实践中难以获得，特别是负样本数据。

### 目的

解决自动驾驶实验室中移液操作这一关键且精度敏感动作的训练数据稀缺问题，特别是负样本数据的缺乏。

### 方法

构建了一个混合数据生成管道：1)真实数据轨道采用人机循环方案，结合自动采集和选择性人工验证；2)虚拟数据轨道使用基于参考条件和提示引导的图像生成来增强真实数据，并进一步筛选验证；3)结合这两条轨道产生类平衡数据集用于气泡检测训练。

### 主要发现

在保留的真实测试集上，完全在自动采集的真实图像上训练的模型达到99.6%的准确率；在训练过程中混合真实数据和生成数据，在减轻收集和审查负担的同时，保持了99.4%的准确率。

### 结论

该方法为向SDL工作流程提供视觉反馈数据提供了一种可扩展且具有成本效益的策略，并为稀有事件检测和更广泛的视觉任务中的数据稀缺问题提供了实用的解决方案。

### 翻译

自动驾驶实验室为减少生物科学中劳动密集型、耗时且通常不可重复的工作流程提供了一条有前景的路径。然而，它们严格的精度要求需要高度稳健的模型，而这些模型的训练依赖于大量标注数据。然而，这类数据在日常实践中难以获得，尤其是负样本。在这项工作中，我们专注于移液操作，这是SDL中最关键且对精度敏感的操作。为了克服训练数据的稀缺性，我们构建了一个融合真实和虚拟数据生成的混合管道。真实数据轨道采用人机循环方案，将自动采集与选择性人工验证相结合，以最小努力最大化准确性。虚拟数据轨道使用基于参考条件和提示引导的图像生成来增强真实数据，并进一步筛选和验证其可靠性。这两条轨道共同产生了一个类平衡的数据集，使稳健的气泡检测训练成为可能。在一个保留的真实测试集上，完全在自动采集的真实图像上训练的模型达到99.6%的准确率，而在训练过程中混合真实数据和生成数据则在减轻收集和审查负担的同时保持了99.4%的准确率。我们的方法为向SDL工作流程提供视觉反馈数据提供了一种可扩展且具有成本效益的策略，并为稀有事件检测和更广泛的视觉任务中的数据稀缺问题提供了实用的解决方案。


### 论文摘要

Self-driving laboratories offer a promising path toward reducing the labor-intensive, time-consuming, and often irreproducible workflows in the biological sciences. Yet their stringent precision requirements demand highly robust models whose training relies on large amounts of annotated data. However, this kind of data is difficult to obtain in routine practice, especially negative samples. In this work, we focus on pipetting, the most critical and precision sensitive action in SDLs. To overcome the scarcity of training data, we build a hybrid pipeline that fuses real and virtual data generation. The real track adopts a human-in-the-loop scheme that couples automated acquisition with selective human verification to maximize accuracy with minimal effort. The virtual track augments the real data using reference-conditioned, prompt-guided image generation, which is further screened and validated for reliability. Together, these two tracks yield a class-balanced dataset that enables robust bubble detection training. On a held-out real test set, a model trained entirely on automatically acquired real images reaches 99.6% accuracy, and mixing real and generated data during training sustains 99.4% accuracy while reducing collection and review load. Our approach offers a scalable and cost-effective strategy for supplying visual feedback data to SDL workflows and provides a practical solution to data scarcity in rare event detection and broader vision tasks.

---

## 106. Forecasting in Offline Reinforcement Learning for Non-stationary Environments

**论文链接:** [http://arxiv.org/abs/2512.01987v1](http://arxiv.org/abs/2512.01987v1)

**作者:** Suzan Ece Ada, Georg Martius, Emre Ugur, Erhan Oztop

**发布时间:** 2025-12-01

**备注:** The Thirty-Ninth Annual Conference on Neural Information Processing Systems, NeurIPS 2025

### GPT解析

### 总结

FORL是一个框架，通过条件扩散的候选状态生成和零样本时间序列基础模型，解决了离线强化学习在非平稳环境中的挑战，提高了代理性能。

### 背景

离线强化学习从预收集数据集中训练策略，但现有方法假设环境平稳或仅在测试时考虑合成扰动，这些假设在现实世界中常因突然的、时变偏移而失效，导致部分可观测性和性能下降。

### 目的

克服现有离线RL方法在非平稳环境中的局限性，处理意外、可能非马尔可夫偏移，确保代理在每个剧集开始时就能保持稳健性能。

### 方法

引入FORL框架，统一条件扩散的候选状态生成(训练时不预设特定非平稳性模式)和零样本时间序列基础模型，以适应非平稳环境。

### 主要发现

在增强真实世界时间序列数据的离线RL基准测试中，FORL相比竞争基线方法持续提高性能，弥合了离线RL与现实世界非平稳环境复杂性之间的差距。

### 结论

FORL通过整合零样本预测与代理经验，有效解决了离线RL在非平稳环境中的挑战，提高了代理性能。

### 翻译

离线强化学习在无法收集额外交互数据时，为从预收集数据集中训练策略提供了一条有前途的途径。然而，现有的离线强化学习方法通常假设平稳性或仅在测试时考虑合成扰动，这些假设在现实世界中常因突然的、时变偏移而失效。这些偏移会导致部分可观测性，使代理错误感知其真实状态并降低性能。为了克服这一挑战，我们引入了非平稳离线强化学习中的预测框架，该框架统一了基于条件扩散的候选状态生成(训练时不预设任何特定的未来非平稳性模式)和零样本时间序列基础模型。FORL针对易受意外、可能非马尔可夫偏移影响的环境，要求代理在每个剧集开始时就保持稳健性能。在使用真实世界时间序列数据增强以模拟真实非平稳性的离线强化学习基准测试上的实证评估表明，FORL相比竞争基线方法持续提高性能。通过将零样本预测与代理经验相结合，我们旨在弥合离线强化学习与现实世界非平稳环境复杂性之间的差距。


### 论文摘要

Offline Reinforcement Learning (RL) provides a promising avenue for training policies from pre-collected datasets when gathering additional interaction data is infeasible. However, existing offline RL methods often assume stationarity or only consider synthetic perturbations at test time, assumptions that often fail in real-world scenarios characterized by abrupt, time-varying offsets. These offsets can lead to partial observability, causing agents to misperceive their true state and degrade performance. To overcome this challenge, we introduce Forecasting in Non-stationary Offline RL (FORL), a framework that unifies (i) conditional diffusion-based candidate state generation, trained without presupposing any specific pattern of future non-stationarity, and (ii) zero-shot time-series foundation models. FORL targets environments prone to unexpected, potentially non-Markovian offsets, requiring robust agent performance from the onset of each episode. Empirical evaluations on offline RL benchmarks, augmented with real-world time-series data to simulate realistic non-stationarity, demonstrate that FORL consistently improves performance compared to competitive baselines. By integrating zero-shot forecasting with the agent's experience, we aim to bridge the gap between offline RL and the complexities of real-world, non-stationary environments.

---

## 107. Low-Rank Prehab: Preparing Neural Networks for SVD Compression

**论文链接:** [http://arxiv.org/abs/2512.01980v1](http://arxiv.org/abs/2512.01980v1)

**作者:** Haoran Qin, Shansita Sharma, Ali Abbasi, Chayne Thrash, Soheil Kolouri

**发布时间:** 2025-12-01

### GPT解析

### 总结

本文提出了一种名为'Low-Rank Prehab'的新方法，通过在压缩前进行微调，明确鼓励权重矩阵具有低秩结构，同时保持任务性能。这种方法将权重引导到参数空间的光谱紧凑区域，实现更平滑的低秩近似和更好的恢复效果。实验表明，该方法显著减少了压缩后的精度下降，并持续改善了微调后的性能。

### 背景

低秩近似方法，如奇异值分解(SVD)及其变体(如Fisher加权SVD、激活SVD)，最近已成为神经网络压缩的有效工具。在这种方法中，分解作为一种'外科手术'干预，随后的微调则作为'康复'阶段来恢复准确性。

### 目的

受外科手术中预康复的启发，作者引入了一个压缩前微调阶段——低秩预康复(Low-Rank Prehab)，旨在明确鼓励权重矩阵具有低秩结构，同时保持任务性能。

### 方法

Low-Rank Prehab是一种在SVD压缩之前进行的预微调方法。它通过条件化模型，在SVD之前将权重引导到参数空间的光谱紧凑区域，从而实现更平滑的低秩近似和更好的恢复效果。

### 主要发现

在大型语言模型(LLMs)和其他基于Transformer的架构(包括视觉变换器ViTs)上的实验表明，Prehab显著减少了压缩后的精度下降，并持续改善了微调后的性能。在各种压缩比率下，该方法都优于最先进的基于SVD的技术，如SVD-LLM。

### 结论

研究强调了为压缩做准备的重要性，而不仅仅是改进压缩和恢复阶段。通过在压缩前准备模型，可以实现更好的压缩效果和性能恢复。

### 翻译

低秩近似方法，如奇异值分解(SVD)及其变体(例如，Fisher加权SVD、激活SVD)最近已成为神经网络压缩的有效工具。在这种设置中，分解作为一种'外科手术'干预，随后的微调则作为'康复'阶段来恢复准确性。受外科手术中预康复的启发，我们引入了一个压缩前微调阶段——低秩预康复(Low-Rank Prehab)，它明确鼓励权重矩阵具有低秩结构，同时保持任务性能。通过在SVD之前对模型进行条件化，Prehab将权重引导到参数空间的光谱紧凑区域，实现更平滑的低秩近似和更好的恢复效果。在大型语言模型(LLMs)和其他基于Transformer的架构(包括视觉变换器ViTs)上的实验表明，Prehab显著减少了压缩后的精度下降，并持续改善了微调后的性能。在各种压缩比率下，我们的方法都优于最先进的基于SVD的技术，如SVD-LLM，这强调了为压缩做准备而非仅改进压缩和恢复阶段的重要性。源代码可在https://github.com/niqretnuh/PREHAB-SVD获取。


### 论文摘要

Low-rank approximation methods such as singular value decomposition (SVD) and its variants (e.g., Fisher-weighted SVD, Activation SVD) have recently emerged as effective tools for neural network compression. In this setting, decomposition acts as a "surgical" intervention, followed by fine-tuning that serves as "rehab" to recover accuracy. Inspired by prehabilitation in surgery, we introduce a pre-compression fine-tuning stage, Low-Rank Prehab, that explicitly encourages low-rank structure in weight matrices while preserving task performance. By conditioning the model before SVD, Prehab steers weights toward spectrally compact regions of the parameter space, enabling smoother low-rank approximation and improved recovery. Experiments on large language models (LLMs) and other Transformer-based architectures, including Vision Transformers (ViTs), show that Prehab substantially reduces the immediate accuracy drop after compression and consistently improves post-finetuning performance. Across a wide range of compression ratios, our method outperforms state-of-the-art SVD-based techniques such as SVD-LLM, highlighting the importance of preparing models for compression rather than only improving the compression and recovery stages. Source code is available at https://github.com/niqretnuh/PREHAB-SVD

---

## 108. Chain-of-Ground: Improving GUI Grounding via Iterative Reasoning and Reference Feedback

**论文链接:** [http://arxiv.org/abs/2512.01979v1](http://arxiv.org/abs/2512.01979v1)

**作者:** Aiden Yiliu Li, Bizhi Yu, Daoan Lei, Tianhe Ren, Shilong Liu

**发布时间:** 2025-12-01

### GPT解析

### 总结

该研究提出了一种名为Chain of Ground (CoG)的多步grounding框架，利用多模态大语言模型进行迭代视觉推理和优化，无需额外训练即可提高GUI定位准确性。该方法在基准测试和现实世界数据集上都表现出色，展示了通过结构化迭代优化释放grounding潜力的有效性。

### 背景

GUI grounding旨在将自然语言指令与复杂用户界面中的精确区域对齐。尽管先进的多模态大语言模型在视觉GUI grounding方面表现出强大能力，但在处理小型或视觉相似目标以及现实世界布局中的模糊性方面仍然存在困难。这些局限性源于有限的grounding能力和对现有推理潜力的利用不足。

### 目的

开发一种无需训练的多步grounding框架，利用多模态大语言模型进行迭代视觉推理和优化，以解决现有方法在处理小型或视觉相似目标以及现实世界布局中的模糊性方面的局限性。

### 方法

研究提出了Chain of Ground (CoG)，一种无需训练的多步grounding框架。该框架利用多模态大语言模型进行迭代视觉推理和优化，不是直接预测，而是逐步反思和调整假设，从而实现更准确和可解释的定位。同时，研究还引入了TPanel UI数据集，这是一个包含420个带有视觉失真（如模糊和遮挡）的标记工业控制面板的数据集，用于衡量现实世界的泛化能力。

### 主要发现

1. CoG方法在ScreenSpot Pro基准测试上达到了68.4%的准确率，比基线提高了4.8个百分点。2. 在TPanel UI数据集上，CoG比强大的基线模型Qwen3 VL 235B提高了6.9个百分点。3. 多步无需训练的grounding方法在现实世界和数字界面之间都表现出有效性。

### 结论

通过结构化的迭代优化而非额外训练可以释放grounding潜力。CoG框架展示了如何通过利用多模态大语言模型的推理能力进行迭代视觉推理和优化，来提高GUI grounding的准确性和可解释性，特别是在处理小型或视觉相似目标以及现实世界布局中的模糊性方面。

### 翻译

GUI grounding旨在将自然语言指令与复杂用户界面中的精确区域对齐。先进的多模态大语言模型在视觉GUI grounding方面表现出强大的能力，但在处理小型或视觉相似目标以及现实世界布局中的模糊性方面仍然存在困难。这些局限性源于有限的grounding能力和对现有推理潜力的利用不足。我们提出了Chain of Ground (CoG)，这是一种无需训练的多步grounding框架，利用多模态大语言模型进行迭代视觉推理和优化。模型不是直接预测，而是逐步反思和调整假设，从而实现更准确和可解释的定位。我们的方法在ScreenSpot Pro基准测试上达到了68.4%的准确率，提高了4.8个百分点。为了衡量现实世界的泛化能力，我们引入了TPanel UI数据集，这是一个包含420个带有模糊和遮挡等视觉失真的标记工业控制面板的数据集。在TPanel UI上，Chain of Ground比强大的基线模型Qwen3 VL 235B提高了6.9个百分点，展示了多步无需训练的grounding在现实世界和数字界面之间的有效性。这些结果突显了一个方向，即通过结构化的迭代优化而非额外训练来释放grounding潜力。


### 论文摘要

GUI grounding aims to align natural language instructions with precise regions in complex user interfaces. Advanced multimodal large language models show strong ability in visual GUI grounding but still struggle with small or visually similar targets and ambiguity in real world layouts. These limitations arise from limited grounding capacity and from underuse of existing reasoning potential. We present Chain of Ground CoG a training free multi step grounding framework that uses multimodal large language models for iterative visual reasoning and refinement. Instead of direct prediction the model progressively reflects and adjusts its hypotheses leading to more accurate and interpretable localization. Our approach achieves 68.4 accuracy on the ScreenSpot Pro benchmark an improvement of 4.8 points. To measure real world generalization we introduce TPanel UI a dataset of 420 labeled industrial control panels with visual distortions such as blur and masking. On TPanel UI Chain of Ground improves over the strong baseline Qwen3 VL 235B by 6.9 points showing the effectiveness of multi step training free grounding across real world and digital interfaces. These results highlight a direction for unlocking grounding potential through structured iterative refinement instead of additional training.

---

## 109. From Atomic to Composite: Reinforcement Learning Enables Generalization in Complementary Reasoning

**论文链接:** [http://arxiv.org/abs/2512.01970v1](http://arxiv.org/abs/2512.01970v1)

**作者:** Sitao Cheng, Xunjian Yin, Ruiwen Zhou, Yuxuan Li, Xinyi Wang, Liangming Pan, William Yang Wang, Victor Zhong

**发布时间:** 2025-12-01

**备注:** Work in Progress. Code and data will be available at https://github.com/sitaocheng/from_atomic_to_composite

### GPT解析

### 总结

本研究探讨了强化学习(RL)对推理能力的贡献机制，发现RL充当推理合成器而非概率放大器，但需要先通过监督微调(SFT)掌握基础原子技能。

### 背景

强化学习(RL)对推理能力的贡献机制——是激励新技能合成还是仅放大现有行为——仍是学术争论焦点。

### 目的

通过互补推理任务探究RL在推理能力形成中的作用机制，该任务需要整合内部参数知识与外部上下文信息。

### 方法

使用人类传记的受控合成数据集，将互补推理能力解耦为参数推理和上下文推理两个原子技能，并在I.I.D.、组合和零样本三种难度设置下评估泛化能力。

### 主要发现

1) SFT足以处理分布内性能但分布外泛化能力弱；2) 发现SFT泛化悖论：模型在分布内准确率高但分布外泛化崩溃；3) RL是推理合成器而非概率放大器；4) RL合成复杂策略需先通过SFT掌握基础原子技能。

### 结论

RL可从学习到的基元中主动合成复杂推理策略，无需明确监督，解耦原子训练后接RL为复杂推理任务提供可扩展的泛化路径。

### 翻译

强化学习(RL)促进推理能力的机制——它是激励新技能合成还是仅放大现有行为——仍是激烈争论的主题。本研究通过互补推理这一复杂任务的视角探究此问题，该任务需要整合内部参数知识与外部上下文信息。我们使用人类传记的受控合成数据集，将此能力严格解耦为两个原子技能：参数推理(依赖内部知识)和上下文推理(依赖外部信息)。为严格评估能力边界，我们在三个不同难度级别评估泛化能力：I.I.D.、组合和零样本设置。我们发现，虽然SFT足以处理分布内性能，但在分布外泛化方面存在困难，特别是在零样本设置中。关键地，我们发现了SFT泛化悖论：仅通过复合任务监督的模型在分布内准确率接近完美，但在分布外泛化方面崩溃，表明它们依赖于死记硬背路径捷径。相比之下，我们发现RL充当推理合成器而非概率放大器。然而，我们发现一个严格的原子先决条件：只有当基础模型首先通过SFT掌握了独立的原子技能(参数和上下文推理)时，RL才能合成这些复杂策略。这些发现挑战了RL仅作为放大器的观点，表明在有足够的原子基础的情况下，RL可以从学习到的基元中主动合成复杂的推理策略，而无需对这种复杂策略进行明确监督。这表明解耦的原子训练后接RL为复杂推理任务提供了一种可扩展的泛化路径。


### 论文摘要

The mechanism by which RL contributes to reasoning capabilities-whether it incentivizes the synthesis of new skills or merely amplifies existing behaviors-remains a subject of intense debate. In this work, we investigate this question through the lens of Complementary Reasoning, a complex task that requires integrating internal parametric knowledge with external contextual information. Using a controlled synthetic dataset of human biographies, we strictly decouple this ability into two atomic skills: Parametric Reasoning (relying on internal knowledge) and Contextual Reasoning (depending on external information). To rigorously assess capability boundaries, we evaluate generalization across three distinct levels of difficulty: I.I.D., Composition, and Zero-shot settings. We find that while SFT is sufficient for in-distribution performance, it struggles with O.O.D. generalization, particularly in Zero-shot settings where relational combinations are novel. Crucially, we identify the SFT Generalization Paradox: Models supervised solely on the composite task achieve near-perfect in-distribution accuracy but collapse on out-of-distribution generalization, indicating their reliance on rote memorization of path shortcuts. In contrast, we find that RL acts as a reasoning synthesizer rather than a probability amplifier. However, we uncover a strict atomic prerequisite: RL can only synthesize these complex strategies if the base model has first mastered the independent atomic skills (Parametric and Contextual) via SFT. These findings challenge the view of RL as a mere amplifier, suggesting that given sufficient atomic foundations, RL can actively synthesize complex reasoning strategies from learned primitives without explicit supervision on such complex strategies. This indicates that decoupled atomic training followed by RL offers a scalable path to generalization for complex reasoning tasks.

---

## 110. A framework for disentangling spatial and visual neural representations

**论文链接:** [http://arxiv.org/abs/2512.01962v1](http://arxiv.org/abs/2512.01962v1)

**作者:** Mai M. Morimoto, Julien Fournier, Aman B. Saleem

**发布时间:** 2025-12-01

**备注:** 14 pages, 5 figures

### GPT解析

### 总结

研究提出了一种新框架，用于分离视觉和空间信号，并成功应用于V1神经元研究，发现了空间信号的异质性和多峰特征

### 背景

皮层区域的神经元通常整合来自不同来源的信号。在初级视觉皮层(V1)中，神经反应受到非视觉上下文的调节，如动物的位置，但这些位置信号在环境中的空间分布仍然未知。

### 目的

提出一种新框架来分离虚拟现实中的视觉和空间贡献，研究位置信号在环境中的空间分布特征

### 方法

提出基于两个原则的新框架：1) 虚拟走廊设计，通过有针对性的线索重复和操作使视觉和空间去相关；2) 广义线性模型(GLM)，在视网膜坐标中明确估计视觉贡献。在模拟中测试框架特性，并应用于在虚拟走廊中导航的小鼠V1记录。

### 主要发现

框架具有高度特异性，能有效捕获空间增益场的轮廓和权重；模型分离了大量V1神经元中的显著空间成分，这些成分表现出异质性，通常是多峰值的轮廓

### 结论

将该框架应用于大规模记录可能为表征调节跨脑区感觉处理的信号性质提供一种稳健的方法

### 翻译

皮层区域的神经元通常整合来自不同来源的信号。在初级视觉皮层(V1)中，神经反应受到非视觉上下文的调节，如动物的位置。然而，这些位置信号在环境中的空间分布仍然未知。在此，我们提出了一种新框架，用于分离虚拟现实中的视觉和空间贡献。该方法基于两个原则：1) 虚拟走廊设计，通过有针对性的线索重复和操作使视觉和空间去相关；2) 广义线性模型(GLM)，在视网膜坐标而非环境坐标中明确估计视觉贡献。在模拟中，我们证明该框架具有高度特异性(仅在存在空间调制时才能恢复空间调制)，并有效捕获了环境中空间增益场的轮廓和权重。当应用于在虚拟走廊中导航的小鼠V1记录时，该模型分离了大量V1神经元中的显著空间成分。恢复的空间成分表现出异质性，通常是多峰值的轮廓。将此框架应用于大规模记录可能为表征调节跨脑区感觉处理的信号性质提供一种稳健的方法。


### 论文摘要

Neurons in cortical areas often integrate signals from different origins. In the primary visual cortex (V1), neural responses are modulated by non-visual context such as the animal's position. However, the spatial profile of these position signals across the environment remains unknown. Here, we propose a new framework to disentangle visual and spatial contributions in virtual reality. This method relies on two principles: 1) a virtual corridor design that decorrelates vision and space through targeted cue repetitions and manipulations and 2) a Generalized Linear Model (GLM) that explicitly estimates visual contributions in retinotopic rather than environmental coordinates. In simulations, we demonstrate that this framework is highly specific (recovering spatial modulation only when present) and effectively captures the profile and weight of spatial gain fields across the environment. When applied to V1 recordings from mice navigating the virtual corridor, the model isolated significant spatial components in a substantial fraction of V1 neurons. The recovered spatial components exhibited heterogeneous, often multi-peaked, profiles. Application of this framework to large-scale recordings may provide a robust approach to characterize the nature of spatial signals modulating sensory processing across brain areas.

---

## 111. SVRG and Beyond via Posterior Correction

**论文链接:** [http://arxiv.org/abs/2512.01930v1](http://arxiv.org/abs/2512.01930v1)

**作者:** Nico Daheim, Thomas Möllenhoff, Ming Liang Ang, Mohammad Emtiyaz Khan

**发布时间:** 2025-12-01

**备注:** Preprint. Under review

### GPT解析

### 总结

本文建立了随机方差减少梯度(SVRG)与贝叶斯后验校正方法之间的新联系，展示SVRG可作为各向同性高斯族后验校正的特例，并通过使用更灵活的指数族获得SVRG的新扩展，包括一个类似牛顿的变体和一个类似Adam的扩展，改进了Transformer语言模型的训练效果。

### 背景

随机方差减少梯度(SVRG)及其变体旨在通过梯度校正加速训练，但在深度学习中的应用成功有限。

### 目的

揭示SVRG与贝叶斯后验校正方法之间的新基础联系，并利用这种联系开发新的SVRG变体以提升深度网络训练效果。

### 方法

将SVRG视为各向同性高斯族后验校正的特例，通过使用更灵活的指数族自动获得SVRG的新扩展，特别是推导出两个使用高斯族的变体：一个采用新颖海塞校正的类似牛顿的变体，以及一个改进Transformer语言模型预训练和微调的类似Adam的扩展。

### 主要发现

SVRG可作为后验校正的一个特例；通过使用更灵活的指数族可获得SVRG的新变体；提出的两个新变体在深度网络训练中表现出色。

### 结论

这是首个将SVRG与贝叶斯方法联系起来并利用其提升深度网络变分训练的研究工作。

### 翻译

随机方差减少梯度(SVRG)及其变体旨在通过使用梯度校正来加速训练，但在深度学习中取得了有限的成功。在这里，我们展示了SVRG与最近提出的贝叶斯方法——后验校正之间的惊人新基础联系。具体来说，我们证明SVRG可以视为各向同性高斯族后验校正的一个特例，而通过使用更灵活的指数族自动获得了SVRG的新扩展。我们通过使用高斯族推导出两个新的SVRG变体：首先，一个采用新颖海塞校正的类似牛顿的变体；其次，一个改进Transformer语言模型预训练和微调的类似Adam的扩展。这是首个将SVRG与贝叶斯联系起来并利用它提升深度网络变分训练的工作。


### 论文摘要

Stochastic Variance Reduced Gradient (SVRG) and its variants aim to speed-up training by using gradient corrections, but have seen limited success in deep learning. Here, we show surprising new foundational connections of SVRG to a recently proposed Bayesian method called posterior correction. Specifically, we show that SVRG is recovered as a special case of posterior correction over the isotropic-Gaussian family, while novel extensions are automatically obtained by using more flexible exponential families. We derive two new SVRG variants by using Gaussian families: First, a Newton-like variant that employs novel Hessian corrections, and second, an Adam-like extension that improves pretraining and finetuning of Transformer language models. This is the first work to connect SVRG to Bayes and use it to boost variational training for deep networks.

---

## 112. Med-VCD: Mitigating Hallucination for Medical Large Vision Language Models through Visual Contrastive Decoding

**论文链接:** [http://arxiv.org/abs/2512.01922v1](http://arxiv.org/abs/2512.01922v1)

**作者:** Zahra Mahdavi, Zahra Khodakaramimaghsoud, Hooman Khaloo, Sina Bakhshandeh Taleshani, Erfan Hashemi, Javad Mirzapour Kaleybar, Omid Nejati Manzari

**发布时间:** 2025-12-01

**DOI:** 10.1016/j.compbiomed.2025.111347

### GPT解析

### 总结

Med-VCD是一种稀疏视觉对比解码方法，能够在不增加时间开销的情况下有效减轻医疗大型视觉语言模型中的幻觉问题，同时保持高效率和可靠性。

### 背景

大型视觉语言模型已成为医疗应用的核心，如医学视觉问答和影像报告生成，但这些模型容易出现看似合理但实际上不正确的幻觉输出。

### 目的

开发一种能够减轻医疗LVLMs中幻觉问题，同时避免二次解码带来时间开销的方法。

### 方法

提出Med-VCD，一种稀疏视觉对比解码方法，采用新颖的token稀疏化策略，即时选择视觉信息丰富的token，减少冗余同时保留关键视觉上下文，平衡效率与可靠性。

### 主要发现

在八个涵盖眼科、放射学和病理学任务的医学数据集上评估显示，Med-VCD将事实准确性平均提高13%，将幻觉准确性提高6%，优于基线医疗LVLMs。

### 结论

Med-VCD是一种有效解决医疗LVLMs中幻觉问题的方法，无需二次解码的时间开销即可显著提高模型的准确性和可靠性。

### 翻译

大型视觉语言模型现已成为医疗应用的核心，如医学视觉问答和影像报告生成。然而，这些模型仍然容易出现看似合理但实际上不正确的幻觉输出。在自然图像领域，虽然已经提出了几种通过强化视觉证据来减轻幻觉的解码策略，但大多数依赖于二次解码或回滚程序，这会显著降低推理速度。此外，现有解决方案通常是特定领域的，并可能导致模态之间或生成内容与真实内容之间的不一致。我们介绍了Med-VCD，一种稀疏视觉对比解码方法，能够在不增加二次解码时间开销的情况下减轻医疗LVLMs中的幻觉问题。Med-VCD采用了一种新颖的token稀疏化策略，可以即时选择视觉信息丰富的token，在减少冗余的同时保留关键视觉上下文，从而平衡效率与可靠性。在八个涵盖眼科、放射学和病理学任务的医学数据集上的评估，包括视觉问答、报告生成和专门的幻觉基准测试，表明Med-VCD将事实准确性平均提高了13%，将幻觉准确性提高了6%，相对于基线医疗LVLMs。


### 论文摘要

Large vision-language models (LVLMs) are now central to healthcare applications such as medical visual question answering and imaging report generation. Yet, these models remain vulnerable to hallucination outputs that appear plausible but are in fact incorrect. In the natural image domain, several decoding strategies have been proposed to mitigate hallucinations by reinforcing visual evidence, but most rely on secondary decoding or rollback procedures that substantially slow inference. Moreover, existing solutions are often domain-specific and may introduce misalignment between modalities or between generated and ground-truth content. We introduce Med-VCD, a sparse visual-contrastive decoding method that mitigates hallucinations in medical LVLMs without the time overhead of secondary decoding. Med-VCD incorporates a novel token-sparsification strategy that selects visually informed tokens on the fly, trimming redundancy while retaining critical visual context and thus balancing efficiency with reliability. Evaluations on eight medical datasets, spanning ophthalmology, radiology, and pathology tasks in visual question answering, report generation, and dedicated hallucination benchmarks, show that Med-VCD raises factual accuracy by an average of 13\% and improves hallucination accuracy by 6\% relative to baseline medical LVLMs.

---

## 113. Disentangling Progress in Medical Image Registration: Beyond Trend-Driven Architectures towards Domain-Specific Strategies

**论文链接:** [http://arxiv.org/abs/2512.01913v1](http://arxiv.org/abs/2512.01913v1)

**作者:** Bailiang Jian, Jiazhen Pan, Rohit Jena, Morteza Ghahremani, Hongwei Bran Li, Daniel Rueckert, Christian Wachinger, Benedikt Wiestler

**发布时间:** 2025-12-01

**备注:** Submitted to Medical Image Analysis. Journal Extension of arXiv:2407.19274

### GPT解析

### 总结

该研究通过系统性的评估表明，在医学图像配准中，领域特定的设计原则比通用的深度学习架构趋势更为重要，研究团队发布了一个透明、模块化的基准测试平台，便于新架构和配准任务的即插即用比较。

### 背景

医学图像配准推动跨器官、模态和患者群体的定量分析。近期的深度学习方法通常结合计算机视觉中的低级'趋势驱动'计算块（如大核CNN、Transformer和状态空间模型）与高级配准特定设计（如运动金字塔、相关层和迭代细化）。

### 目的

明确不同设计方法的相对贡献，解决核心问题：未来的配准研究应该专注于导入通用架构趋势，还是专注于改进领域特定设计原则？

### 方法

通过一个跨越脑部、肺部、心脏和腹部配准的模块化框架，系统性地分离这两种范式的影响。

### 主要发现

低级'趋势驱动'计算块仅带来边际或不一致的改进，而高级配准特定设计始终提供更准确、更平滑、更稳健的变形。领域先验显著提升标准U-Net基线的性能，远超融入'趋势驱动'块的变体，实现了约3%的平均相对改进。

### 结论

研究重点应从遵循架构趋势转向拥抱领域特定设计原则，这些才是基于学习的医学图像配准进步的真正驱动力。

### 翻译

医学图像配准推动跨器官、模态和患者群体的定量分析。近期的深度学习方法通常结合计算机视觉中的低级'趋势驱动'计算块，如大核CNN、Transformer和状态空间模型，与高级配准特定设计，如运动金字塔、相关层和迭代细化。然而，它们的相对贡献仍然不明确且相互纠缠。这提出了一个核心问题：未来的配准进步应该专注于导入通用架构趋势，还是专注于改进领域特定的设计原则？通过一个跨越脑部、肺部、心脏和腹部配准的模块化框架，我们系统地分离了这两种范式的影响。我们的评估显示，低级'趋势驱动'计算块仅带来边际或不一致的改进，而高级配准特定设计始终提供更准确、更平滑、更稳健的变形。这些领域先验显著提升了标准U-Net基线的性能，远超过融入'趋势驱动'块的变体，实现了约3%的平均相对改进。所有模型和实验都在一个透明、模块化的基准测试中发布，便于新架构和配准任务的即插即用比较（https://github.com/BailiangJ/rethink-reg）。这个动态且可扩展的平台建立了可复现和公平评估的共同基础，邀请社区将真正的方法学贡献与领域先验分离开来。我们的研究结果主张研究重点的转变：从遵循架构趋势到拥抱领域特定设计原则，作为基于学习的医学图像配准进步的真正驱动力。


### 论文摘要

Medical image registration drives quantitative analysis across organs, modalities, and patient populations. Recent deep learning methods often combine low-level "trend-driven" computational blocks from computer vision, such as large-kernel CNNs, Transformers, and state-space models, with high-level registration-specific designs like motion pyramids, correlation layers, and iterative refinement. Yet, their relative contributions remain unclear and entangled. This raises a central question: should future advances in registration focus on importing generic architectural trends or on refining domain-specific design principles? Through a modular framework spanning brain, lung, cardiac, and abdominal registration, we systematically disentangle the influence of these two paradigms. Our evaluation reveals that low-level "trend-driven" computational blocks offer only marginal or inconsistent gains, while high-level registration-specific designs consistently deliver more accurate, smoother, and more robust deformations. These domain priors significantly elevate the performance of a standard U-Net baseline, far more than variants incorporating "trend-driven" blocks, achieving an average relative improvement of $\sim3\%$. All models and experiments are released within a transparent, modular benchmark that enables plug-and-play comparison for new architectures and registration tasks (https://github.com/BailiangJ/rethink-reg). This dynamic and extensible platform establishes a common ground for reproducible and fair evaluation, inviting the community to isolate genuine methodological contributions from domain priors. Our findings advocate a shift in research emphasis: from following architectural trends to embracing domain-specific design principles as the true drivers of progress in learning-based medical image registration.

---

## 114. Provably Safe Model Updates

**论文链接:** [http://arxiv.org/abs/2512.01899v1](http://arxiv.org/abs/2512.01899v1)

**作者:** Leo Elmecker-Plakolm, Pierre Fasterling, Philip Sosnin, Calvin Tsay, Matthew Wicker

**发布时间:** 2025-12-01

**备注:** 12 pages, 9 figures, submitted to IEEE SaTML 2026

### GPT解析

### 总结

论文提出了一种可证明安全的机器学习模型更新框架，通过计算局部不变域(LID)来确保模型更新满足关键性能规范。

### 背景

关键环境本质上是动态的，分布偏移、新漏洞和变化的需求要求模型持续更新，但现有启发式方法无法保证更新后的模型继续满足性能规范。

### 目的

开发一个框架，能够证明更新后的模型仍然满足所需的性能规范，解决现有方法无法提供正式安全保证的问题。

### 方法

将问题形式化为计算最大的局部不变域(LID)，通过参数化抽象域(正交体、zonotope)得到可行的原始对偶形式，通过将更新投影到安全域实现高效认证，并支持计算多个近似最优LID、整合正则化偏差和使用前瞻性数据缓冲区。

### 主要发现

在持续学习和基础模型微调基准测试中，该方法匹配或超过了启发式基线，有效避免了遗忘问题，同时提供了形式化的安全保证。

### 结论

该框架通过局部不变域概念，为机器学习模型的安全更新提供了形式化保证，确保模型更新不会违反关键规范。

### 翻译

关键环境本质上是动态的。分布偏移、新出现的漏洞和不断变化的需求要求机器学习模型持续更新。然而，即使是良性的参数更新也可能产生意外后果，如经典模型中的灾难性遗忘或基础模型中的对齐漂移。现有的启发式方法（如正则化、参数隔离）可以缓解这些影响，但不能保证更新的模型继续满足所需的性能规范。我们通过引入一个可证明安全模型更新的框架来解决这个问题。我们的方法首先将问题形式化为计算最大的局部不变域（LID）：参数空间中的一个连通区域，其中所有点都被证明满足给定规范。虽然精确的最大LID计算是不可行的，但我们表明将问题放松到参数化抽象域（正交体、zonotope）可以得到一个可行的原始对偶形式。这通过将更新投影到安全域实现了高效的认证，独立于使用的数据或算法。我们的形式化进一步允许计算多个近似最优的LID，整合受正则化启发的偏差，并使用前瞻性数据缓冲区。在持续学习和基础模型微调基准测试中，我们的方法匹配或超过了启发式基线，在避免遗忘方面提供了形式化的安全保证。


### 论文摘要

Safety-critical environments are inherently dynamic. Distribution shifts, emerging vulnerabilities, and evolving requirements demand continuous updates to machine learning models. Yet even benign parameter updates can have unintended consequences, such as catastrophic forgetting in classical models or alignment drift in foundation models. Existing heuristic approaches (e.g., regularization, parameter isolation) can mitigate these effects but cannot certify that updated models continue to satisfy required performance specifications. We address this problem by introducing a framework for provably safe model updates. Our approach first formalizes the problem as computing the largest locally invariant domain (LID): a connected region in parameter space where all points are certified to satisfy a given specification. While exact maximal LID computation is intractable, we show that relaxing the problem to parameterized abstract domains (orthotopes, zonotopes) yields a tractable primal-dual formulation. This enables efficient certification of updates - independent of the data or algorithm used - by projecting them onto the safe domain. Our formulation further allows computation of multiple approximately optimal LIDs, incorporation of regularization-inspired biases, and use of lookahead data buffers. Across continual learning and foundation model fine-tuning benchmarks, our method matches or exceeds heuristic baselines for avoiding forgetting while providing formal safety guarantees.

---

## 115. OPOR-Bench: Evaluating Large Language Models on Online Public Opinion Report Generation

**论文链接:** [http://arxiv.org/abs/2512.01896v1](http://arxiv.org/abs/2512.01896v1)

**作者:** Jinzheng Yu, Yang Xu, Haozhen Li, Junqi Li, Yifan Feng, Ligu Zhu, Hao Shen, Lei Shi

**发布时间:** 2025-12-01

**备注:** 27 pages, accepted by CMC-Computers, Materials & Continua, 2025

### GPT解析

### 总结

研究定义了自动化在线舆情报告生成任务，构建了数据集和评估框架，为舆情报告自动生成领域奠定了研究基础。

### 背景

在线舆情报告整合新闻和社交媒体信息，为政府和企业的危机管理提供及时支持。虽然大型语言模型已使自动化报告生成在技术上可行，但该领域缺乏系统研究，特别是缺乏正式任务定义和基准。

### 目的

填补在线舆情报告自动生成领域的空白，定义任务，构建数据集，提出评估框架，为未来研究奠定基础。

### 方法

1. 定义了自动化在线舆情报告生成(OPOR-GEN)任务；2. 构建了OPOR-BENCH数据集，包含463个危机事件及其对应的新闻文章、社交媒体帖子和参考摘要；3. 提出了OPOR-EVAL评估框架，通过基于代理的框架模拟人类专家评估。

### 主要发现

使用前沿模型的实验表明，所提出的框架与人类判断具有高度相关性。

### 结论

全面的任务定义、基准数据集和评估框架为这一关键领域的未来研究提供了坚实的基础。

### 翻译

在线舆情报告整合新闻和社交媒体信息，为政府和企业的及时危机管理提供支持。虽然大型语言模型已使自动化报告生成在技术上变得可行，但这一特定领域的系统研究仍然明显缺乏，特别是缺乏正式的任务定义和相应的基准。为填补这一空白，我们定义了自动化在线舆情报告生成(OPOR-GEN)任务，并构建了OPOR-BENCH数据集，该数据集以事件为中心，涵盖了463个危机事件及其对应的新闻文章、社交媒体帖子和参考摘要。为了评估报告质量，我们提出了OPOR-EVAL，这是一种新颖的基于代理的框架，通过分析上下文中的生成报告来模拟人类专家评估。使用前沿模型的实验表明，我们的框架与人类判断具有高度相关性。我们全面的任务定义、基准数据集和评估框架为这一关键领域的未来研究提供了坚实的基础。


### 论文摘要

Online Public Opinion Reports consolidate news and social media for timely crisis management by governments and enterprises. While large language models have made automated report generation technically feasible, systematic research in this specific area remains notably absent, particularly lacking formal task definitions and corresponding benchmarks. To bridge this gap, we define the Automated Online Public Opinion Report Generation (OPOR-GEN) task and construct OPOR-BENCH, an event-centric dataset covering 463 crisis events with their corresponding news articles, social media posts, and a reference summary. To evaluate report quality, we propose OPOR-EVAL, a novel agent-based framework that simulates human expert evaluation by analyzing generated reports in context. Experiments with frontier models demonstrate that our framework achieves high correlation with human judgments. Our comprehensive task definition, benchmark dataset, and evaluation framework provide a solid foundation for future research in this critical domain.

---

## 116. Storage capacity of perceptron with variable selection

**论文链接:** [http://arxiv.org/abs/2512.01861v1](http://arxiv.org/abs/2512.01861v1)

**作者:** Yingying Xu, Masayuki Ohzeki, Yoshiyuki Kabashima

**发布时间:** 2025-12-01

**备注:** 21 pages, 3 figures

### GPT解析

### 总结

研究机器学习中高维数据区分真实结构与偶然相关性的挑战，针对感知机模型，通过最优变量选择超越Cover-Gardner理论设定的界限。

### 背景

机器学习中的核心挑战是如何在高维数据中区分真实结构与偶然相关性。感知机作为神经网络的基础模型，是研究这一问题的重要对象。

### 目的

研究模式负载α与变量选择比率ρ之间的关系，探究简单感知机如何通过从N个变量中优选M=ρN个变量，来完美分类P=αN个随机模式。

### 方法

基于统计力学中的复制方法，开发了一种枚举能够实现完美模式分类的变量组合的方法。

### 主要发现

最优变量选择可以超越Cover-Gardner理论设定的α<2ρ界限，实现更高效的分类能力。

### 结论

研究结果为区分数据中的真实结构与虚假规律提供了定量标准，同时也给出了具有稀疏非对称耦合的联想记忆模型的存储容量。

### 翻译

机器学习中的一个中心挑战是在高维数据中区分真实结构与偶然相关性。在这项工作中，我们针对感知机这一神经网络基础模型解决了这个问题。具体来说，我们研究了模式负载α与变量选择比率ρ之间的关系，对于简单感知机可以通过从N个变量中优选M=ρN个变量来完美分类P=αN个随机模式。虽然Cover-Gardner理论确立了随机选择的ρN维度可以在且仅在α<2ρ的条件下分离αN个随机模式，但我们通过开发一种基于统计力学中复制方法的方法来枚举能够实现完美模式分类的变量组合，证明了最优变量选择可以超越这一界限。这不仅为区分数据中的真实结构与虚假规律提供了定量标准，而且也给出了具有稀疏非对称耦合的联想记忆模型的存储容量。


### 论文摘要

A central challenge in machine learning is to distinguish genuine structure from chance correlations in high-dimensional data. In this work, we address this issue for the perceptron, a foundational model of neural computation. Specifically, we investigate the relationship between the pattern load $α$ and the variable selection ratio $ρ$ for which a simple perceptron can perfectly classify $P = αN$ random patterns by optimally selecting $M = ρN$ variables out of $N$ variables. While the Cover--Gardner theory establishes that a random subset of $ρN$ dimensions can separate $αN$ random patterns if and only if $α< 2ρ$, we demonstrate that optimal variable selection can surpass this bound by developing a method, based on the replica method from statistical mechanics, for enumerating the combinations of variables that enable perfect pattern classification. This not only provides a quantitative criterion for distinguishing true structure in the data from spurious regularities, but also yields the storage capacity of associative memory models with sparse asymmetric couplings.

---

## 117. OpenREAD: Reinforced Open-Ended Reasoing for End-to-End Autonomous Driving with LLM-as-Critic

**论文链接:** [http://arxiv.org/abs/2512.01830v1](http://arxiv.org/abs/2512.01830v1)

**作者:** Songyan Zhang, Wenhui Huang, Zhan Chen, Chua Jiahao Collister, Qihang Huang, Chen Lv

**发布时间:** 2025-12-01

### GPT解析

### 总结

OpenREAD是一种基于开放推理强化视觉语言模型的自动驾驶框架，通过端到端的强化微调实现了从高级推理到低级轨迹规划的全面优化，在推理和规划基准测试中达到了最先进的性能。

### 背景

当前两阶段微调策略（监督微调SFT和强化微调RFT）在知识驱动的自动驾驶范式中有很大潜力，但SFT的学习性质限制了推理的泛化能力，而当前的RFT方法主要应用于下游任务，因为场景理解是一个开放性问题，相应的奖励难以量化。

### 目的

解决现有自动驾驶系统中SFT限制推理泛化能力以及RFT难以应用于场景理解等开放性问题的挑战，实现端到端的RFT从高级推理到低级轨迹规划的全面应用。

### 方法

OpenREAD框架通过以下方法实现：1)在开源驾驶相关知识数据集上构建大规模思维链(CoT)注释；2)使用强大的Qwen3大语言模型作为RFT中的评判者，在奖励建模过程中对开放性问题的推理质量进行量化。

### 主要发现

联合端到端RFT在上游和下游任务中都带来了显著改进，使OpenREAD在推理和规划基准测试中实现了最先进的性能。

### 结论

OpenREAD框架通过结合开放推理和强化学习，有效解决了自动驾驶系统中推理泛化和奖励量化的问题，实现了从高级推理到低级轨迹规划的全面优化，代表了自动驾驶技术的重要进步。

### 翻译

最近，两阶段微调策略，例如通过监督微调(SFT)获取基本驾驶知识，并通过强化微调(RFT)进一步增强决策和规划，在推进知识驱动的自动驾驶(AD)范式方面显示出巨大潜力。然而，SFT的学习性质仍然限制了推理的泛化能力，从而限制了驾驶性能的全部潜力。同时，当前的RFT方法主要应用于下游任务，因为场景理解是一个开放性问题，相应的奖励难以量化。为了解决这些局限性，我们提出了OpenREAD，一个基于开放推理强化视觉语言模型(VLM)的自动驾驶(AD)框架，能够实现从高级推理到低级轨迹规划的端到端RFT。具体而言，我们首先在开源驾驶相关知识数据集上构建大规模思维链(CoT)注释，并使用强大的Qwen3大语言模型(LLM)作为RFT中的评判者，在奖励建模过程中对开放性问题的推理质量进行量化。大量实验证实，联合端到端RFT在上游和下游任务中都带来了显著改进，使OpenREAD能够在推理和规划基准测试上实现最先进的性能。


### 论文摘要

Recently, two-stage fine-tuning strategies, e.g., acquiring essential driving knowledge through supervised fine-tuning (SFT) and further enhancing decision-making and planning via reinforcement fine-tuning (RFT), have shown strong potential in advancing the knowledge-driven autonomous driving (AD) paradigm. However, the learning nature of SFT still limits the generalization of reasoning, thereby constraining the full potential of driving performance. Meanwhile, current RFT approaches are primarily applied to downstream tasks, since scene understanding is an open-ended problem where corresponding rewards are difficult to quantify. To address these limitations, we propose OpenREAD, an OPEN-ended REasoning reinforced vision-language model (VLM)-based autonomous driving (AD) framework that enables end-to-end RFT across the full spectrum from high-level reasoning to low-level trajectory planning. Specifically, we begin by constructing large-scale Chain-of-Thought (CoT) annotations on open-source driving-related knowledge datasets, and employ the powerful Qwen3 large language model (LLM) as the critic in RFT to quantify reasoning quality for open-ended questions during reward modeling. Extensive experiments confirm that joint end-to-end RFT yields substantial improvements in both upstream and downstream tasks, enabling OpenREAD to achieve state-of-the-art performance on reasoning and planning benchmarks.

---

## 118. CauSight: Learning to Supersense for Visual Causal Discovery

**论文链接:** [http://arxiv.org/abs/2512.01827v1](http://arxiv.org/abs/2512.01827v1)

**作者:** Yize Zhang, Meiqi Chen, Sirui Chen, Bo Peng, Yanxi Zhang, Tianyu Li, Chaochao Lu

**发布时间:** 2025-12-01

**备注:** project page: https://github.com/OpenCausaLab/CauSight

### GPT解析

### 总结

研究引入了视觉因果发现任务，使AI系统能够理解视觉实体间的因果关系而不仅仅是识别它们的存在。研究团队构建了大规模视觉因果图数据集(VCG-32K)，并开发了名为CauSight的视觉语言模型，通过因果感知推理执行视觉因果发现。实验表明，CauSight在视觉因果发现任务上超越了GPT-4.1，性能提升了三倍多。

### 背景

人类具有因果思维能力，能够理解事物发生的原因而不仅仅是观察到什么。现代AI系统缺乏这种能力，需要通过引入视觉因果发现任务来模拟这种能力。

### 目的

开发能够推断视觉实体间因果关系的AI模型，而不仅仅是感知它们的存在。

### 方法

1. 构建了视觉因果图数据集(VCG-32K)，包含超过32,000张标注了实体级因果图的图像；2. 开发了名为CauSight的视觉语言模型；3. 训练方法整合了三个组件：从VCG-32K中筛选训练数据、使用因果思维树(ToCT)合成推理轨迹、使用设计的因果奖励进行强化学习以优化推理策略。

### 主要发现

CauSight在视觉因果发现任务上超越了GPT-4.1，实现了超过三倍的性能提升(绝对增益21%)。

### 结论

通过引入视觉因果发现任务和开发CauSight模型，研究成功地使AI系统能够理解视觉实体间的因果关系，而不仅仅是识别它们的存在。研究团队已开源了代码、模型和数据集。

### 翻译

因果思维使人类不仅能理解所见之物，还能理解其发生的原因。为了在现代AI系统中复制这种能力，我们引入了视觉因果发现任务。它要求模型在不同场景中推断视觉实体间的因果关系，而不仅仅是感知它们的存在。为此，我们首先构建了视觉因果图数据集(VCG-32K)，这是一个包含超过32,000张图像的大型集合，这些图像标注了实体级因果图，并进一步开发了CauSight，这是一种新颖的视觉语言模型，通过因果感知推理执行视觉因果发现。我们的训练方法整合了三个组件：(1)从VCG-32K中筛选训练数据，(2)因果思维树(Tree-of-Causal-Thought, ToCT)用于合成推理轨迹，(3)使用设计的因果奖励进行强化学习以优化推理策略。实验表明，CauSight在视觉因果发现上优于GPT-4.1，实现了超过三倍的性能提升(绝对增益21%)。我们的代码、模型和数据集已在项目页面完全开源：https://github.com/OpenCausaLab/CauSight。


### 论文摘要

Causal thinking enables humans to understand not just what is seen, but why it happens. To replicate this capability in modern AI systems, we introduce the task of visual causal discovery. It requires models to infer cause-and-effect relations among visual entities across diverse scenarios instead of merely perceiving their presence. To this end, we first construct the Visual Causal Graph dataset (VCG-32K), a large-scale collection of over 32,000 images annotated with entity-level causal graphs, and further develop CauSight, a novel vision-language model to perform visual causal discovery through causally aware reasoning. Our training recipe integrates three components: (1) training data curation from VCG-32K, (2) Tree-of-Causal-Thought (ToCT) for synthesizing reasoning trajectories, and (3) reinforcement learning with a designed causal reward to refine the reasoning policy. Experiments show that CauSight outperforms GPT-4.1 on visual causal discovery, achieving over a threefold performance boost (21% absolute gain). Our code, model, and dataset are fully open-sourced at project page: https://github.com/OpenCausaLab/CauSight.

---

## 119. Benchmarking Distributed Quantum Computing Emulators

**论文链接:** [http://arxiv.org/abs/2512.01807v1](http://arxiv.org/abs/2512.01807v1)

**作者:** Guillermo Díaz-Camacho, Iago F. Llovo, F. Javier Cardama, Irais Bautista, Daniel Faílde, Mariamo Mussa Juane, Jorge Vázquez-Pérez, Natalia Costas, Tomás F. Pena, Andrés Gómez

**发布时间:** 2025-12-01

### GPT解析

### 总结

该研究引入了一个评估分布式量子计算仿真器的基准测试框架，使用分布式逆量子傅里叶变换作为测试案例，通过比较四个代表性仿真器的性能，揭示了架构保真度和仿真可扩展性之间的权衡。

### 背景

可扩展的量子计算需要超越单片处理器的架构解决方案，分布式量子计算通过量子通信协议互连较小的量子节点实现协作计算，而仿真平台是在现实条件下探索其可行性的必要工具。

### 目的

引入一个基准测试框架来评估分布式量子计算仿真器，使用分布式实现的逆量子傅里叶变换作为代表性测试案例，从而为未来仿真器开发和分布式量子协议验证提供基础。

### 方法

使用基于传态的协议将量子傅里叶变换跨节点分区，并根据执行时间、内存使用和与单片基线的保真度分析性能。审查多种仿真器并选择四个代表性仿真器（Qiskit Aer、SquidASM、Interlin-q和SQUANCH）进行基准测试，这些仿真器在离散事件仿真、量子网络、噪声建模和并行执行方面存在显著差异。

### 主要发现

许多仿真器平台缺乏对传态协议的支持或需要复杂的变通方案；所选的四个仿真器在架构保真度和仿真可扩展性之间存在权衡；该框架为未来仿真器开发和分布式量子协议验证提供了基础。

### 结论

所提出的基准测试框架可以扩展以支持额外的算法和仿真器，有助于推动分布式量子计算领域的发展。

### 翻译

可扩展的量子计算需要超越单片处理器的架构解决方案。分布式量子计算通过量子通信协议互连较小的量子节点来解决这一挑战，实现协作计算。虽然存在分布式量子计算的多种实验和理论提案，但仿真平台是在现实条件下探索其可行性的必要工具。在这项工作中，我们引入了一个基准测试框架，使用分布式实现的逆量子傅里叶变换作为代表性测试案例来评估分布式量子计算仿真器，该变换能够从预编码的傅里叶状态中高效恢复相位。QFT通过基于传态的协议跨节点分区，并根据执行时间、内存使用和与单片基线的保真度分析性能。作为这项工作的一部分，我们审查了广泛的仿真器，确定了它们在编程分布式量子算法方面的能力和局限性。许多平台要么缺乏对传态协议的支持，要么需要复杂的变通方案。因此，我们选择并测试了四个代表性的仿真器：Qiskit Aer、SquidASM、Interlin-q和SQUANCH。它们在离散事件仿真、量子网络、噪声建模和并行执行方面的支持存在显著差异。我们的结果突显了架构保真度和仿真可扩展性之间的权衡，为未来仿真器开发和分布式量子协议的验证提供了基础。该框架可以扩展以支持额外的算法和仿真器。


### 论文摘要

Scalable quantum computing requires architectural solutions beyond monolithic processors. Distributed quantum computing (DQC) addresses this challenge by interconnecting smaller quantum nodes through quantum communication protocols, enabling collaborative computation. While several experimental and theoretical proposals for DQC exist, emulator platforms are essential tools for exploring their feasibility under realistic conditions. In this work, we introduce a benchmarking framework to evaluate DQC emulators using a distributed implementation of the inverse Quantum Fourier Transform ($\mathrm{QFT}^{\dagger}$) as a representative test case, which enables efficient phase recovery from pre-encoded Fourier states. The QFT is partitioned across nodes using teleportation-based protocols, and performance is analyzed in terms of execution time, memory usage, and fidelity with respect to a monolithic baseline.   As part of this work, we review a broad range of emulators, identifying their capabilities and limitations for programming distributed quantum algorithms. Many platforms either lacked support for teleportation protocols or required complex workarounds. Consequently, we select and benchmark four representative emulators: Qiskit Aer, SquidASM, Interlin-q, and SQUANCH. They differ significantly in their support for discrete-event simulation, quantum networking, noise modeling, and parallel execution. Our results highlight the trade-offs between architectural fidelity and simulation scalability, providing a foundation for future emulator development and the validation of distributed quantum protocols. This framework can be extended to support additional algorithms and emulators.

---

## 120. GR-RL: Going Dexterous and Precise for Long-Horizon Robotic Manipulation

**论文链接:** [http://arxiv.org/abs/2512.01801v1](http://arxiv.org/abs/2512.01801v1)

**作者:** Yunfei Li, Xiao Ma, Jiafeng Xu, Yu Cui, Zhongren Cui, Zhigang Han, Liqun Huang, Tao Kong, Yuxiao Liu, Hao Niu, Wanli Peng, Jingchao Qiao, Zeyu Ren, Haixin Shi, Zhi Su, Jiawen Tian, Yuyang Xiao, Shenyu Zhang, Liwei Zheng, Hang Li, Yonghui Wu

**发布时间:** 2025-12-01

### GPT解析

### 总结

GR-RL是一种机器人学习框架，可将通用视觉-语言-行动策略转变为专业化的长视野灵巧操作专家。

### 背景

现有VLA策略假设人类演示是最优的，但在高度灵巧和精确的操作任务中，人类演示实际上是有噪声和次优的。

### 目的

开发一种方法，将通用VLA策略转变为高度专业化的长视野灵巧操作专家。

### 方法

GR-RL采用多阶段训练管道：1)学习基于视觉-语言条件的任务进度并过滤演示轨迹；2)引入形态对称性增强提高泛化能力；3)通过学习潜在空间噪声预测器进行在线RL，使VLA策略与部署行为对齐。

### 主要发现

GR-RL是首个能够自主系鞋带的基于学习的策略，成功率达83.3%，能够处理需要长视野推理、毫米级精度和柔性软体交互的复杂任务。

### 结论

GR-RL为使通用机器人基础模型专业化为可靠的现实世界专家提供了重要步骤。

### 翻译

我们提出了GR-RL，一种机器人学习框架，可将通用视觉-语言-行动策略转变为高度专业化的长视野灵巧操作专家。现有VLA策略的核心假设是人类演示的最优性。然而，我们认为在高度灵巧和精确的操作任务中，人类演示是有噪声和次优的。GR-RL提出了一个多阶段训练管道，通过强化学习来过滤、增强和强化演示。首先，GR-RL学习基于视觉-语言条件的任务进度，过滤演示轨迹，只保留对进度有积极贡献的转换。具体来说，我们展示通过直接应用稀疏奖励的离线RL，得到的Q值可以作为鲁棒的进度函数。接下来，我们引入形态对称性增强，显著提高了GR-RL的泛化能力和性能。最后，为了使VLA策略与其部署行为更好地对齐以实现高精度控制，我们通过学习潜在空间噪声预测器进行在线RL。通过这一管道，据我们所知，GR-RL是首个能够自主系鞋带的基于学习的策略，成功率为83.3%，这是一项需要长视野推理、毫米级精度和柔性软体交互的任务。我们希望GR-RL为使通用机器人基础模型专业化为可靠的现实世界专家提供了第一步。


### 论文摘要

We present GR-RL, a robotic learning framework that turns a generalist vision-language-action (VLA) policy into a highly capable specialist for long-horizon dexterous manipulation. Assuming the optimality of human demonstrations is core to existing VLA policies. However, we claim that in highly dexterous and precise manipulation tasks, human demonstrations are noisy and suboptimal. GR-RL proposes a multi-stage training pipeline that filters, augments, and reinforces the demonstrations by reinforcement learning. First, GR-RL learns a vision-language-conditioned task progress, filters the demonstration trajectories, and only keeps the transitions that contribute positively to the progress. Specifically, we show that by directly applying offline RL with sparse reward, the resulting $Q$-values can be treated as a robust progress function. Next, we introduce morphological symmetry augmentation that greatly improves the generalization and performance of GR-RL. Lastly, to better align the VLA policy with its deployment behaviors for high-precision control, we perform online RL by learning a latent space noise predictor. With this pipeline, GR-RL is, to our knowledge, the first learning-based policy that can autonomously lace up a shoe by threading shoelaces through multiple eyelets with an 83.3% success rate, a task requiring long-horizon reasoning, millimeter-level precision, and compliant soft-body interaction. We hope GR-RL provides a step toward enabling generalist robot foundations models to specialize into reliable real-world experts.

---

## 121. Search for Peak Structures in the Stochastic Gravitational-Wave Background in LIGO-Virgo-KAGRA O1-O4a Datasets

**论文链接:** [http://arxiv.org/abs/2512.01776v1](http://arxiv.org/abs/2512.01776v1)

**作者:** Catalina-Ana Miritescu, Mario Martinez, Oriol Pujolas

**发布时间:** 2025-12-01

### GPT解析

### 总结

本研究使用LIGO-Virgo-KAGRA网络前三轮观测和第四轮观测初期数据，专门搜索具有非平凡峰结构的引力波背景。研究基于多种早期宇宙模型，这些模型具有多峰信号特征。

### 背景

早期宇宙模型可能产生具有多个峰值的引力波背景信号，这超出了传统的单峰结构假设。LIGO-Virgo-KAGRA合作组织积累了足够的数据来探索这种更复杂的信号结构。

### 目的

研究目的是探测具有非平凡峰结构的引力波背景，特别是双峰光谱结构，扩展对引力波信号的理解。

### 方法

研究者引入了一种基于两个归一化断点幂律叠加的双峰光谱模型无关参数化方法。他们使用LIGO-Virgo-KAGRA各向同性交叉相关数据进行贝叶斯推断研究。

### 主要发现

虽然没有发现多峰背景的统计显著证据，但分析提供了对峰间斜率的约束，并与信号幅度相关。这些结果展示了LIGO-Virgo-KAGRA探测单峰结构之外信号的能力。

### 结论

LIGO-Virgo-KAGRA网络有能力探测超越单峰结构的信号，为未来观测轮次和先进探测器时代针对非平凡引力波背景光谱形状的定向搜索奠定了基础。

### 翻译

我们提出使用LIGO-Virgo-KAGRA网络前三轮观测和第四轮观测初期数据，专门搜索具有非平凡峰结构的引力波背景。分析动机是多种早期宇宙模型，这些模型以具有多个峰值的信号为特征。我们引入了一种基于两个归一化断点幂律叠加的双峰光谱模型无关参数化方法，并使用LIGO-Virgo-KAGRA各向同性交叉相关数据进行贝叶斯推断研究。虽然没有发现多峰背景的统计显著证据，但分析提供了对峰间斜率的约束，并与信号幅度相关。这些结果展示了LIGO-Virgo-KAGRA探测单峰结构之外信号的能力，并为未来观测轮次和先进探测器时代针对非平凡引力波背景光谱形状的定向搜索奠定了基础。


### 论文摘要

We present a dedicated search for gravitational-wave backgrounds with nontrivial peak structures using data from the first three and the initial part of the fourth observing runs of the LIGO-Virgo-KAGRA network. The analysis is motivated by a variety of early-Universe models characterized by signals with multiple peaks. We introduce a model independent parameterization of double-peaked spectra based on the superposition of two normalized broken power laws and perform a Bayesian inference study using the LIGO-Virgo-KAGRA isotropic cross-correlation data. While no statistically significant evidence for a multi-peak background is found, the analysis provides constraints on the inter-peak slopes in correlation with the signal amplitude. These results exhibit LIGO-Virgo-KAGRA's ability to probe signals beyond a single peak structure and establish a foundation for future targeted searches for nontrivial gravitational waves background spectral shapes in future observing runs and the advanced detector era.

---

## 122. MMAG: Mixed Memory-Augmented Generation for Large Language Models Applications

**论文链接:** [http://arxiv.org/abs/2512.01710v1](http://arxiv.org/abs/2512.01710v1)

**作者:** Stefano Zeppieri

**发布时间:** 2025-12-01

### GPT解析

### 总结

本文提出了一种混合记忆增强生成(MMAG)模式，通过五种记忆层次提升大型语言代理的连贯性、个性化和长期交互能力。

### 背景

大型语言模型在单个提示中生成连贯文本表现出色，但在长时间交互中维持相关性、个性化和连续性方面存在不足，而人类交流依赖于多种形式的记忆。

### 目的

引入混合记忆增强生成(MMAG)模式，为基于LLM的代理组织记忆，使其更接近人类的交流方式。

### 方法

提出一个框架，将记忆组织为五个相互作用的层次：会话记忆、长期用户记忆、情境和事件关联记忆、感官和情境感知记忆、短期工作记忆，借鉴认知心理学映射到技术组件，并概述协调、优先级和冲突解决策略。

### 主要发现

在Heero对话代理中的实现表明，加密的长期用户资料和对话历史已经提高了参与度和留存率。

### 结论

MMAG为构建具有丰富记忆的语言代理提供了基础，这些代理更加连贯、主动，并且与人类需求更加一致。

### 翻译

大型语言模型(LLMs)在单个提示中生成连贯文本方面表现出色，但在长时间交互中维持相关性、个性化和连续性方面存在不足。然而，人类交流依赖于多种形式的记忆，从回忆过去的对话到适应个人特质和情境背景。本文引入了混合记忆增强生成(MMAG)模式，这是一个将基于LLM的代理记忆组织为五个相互作用的层次的框架：会话记忆、长期用户记忆、情境和事件关联记忆、感官和情境感知记忆、短期工作记忆。借鉴认知心理学，我们将这些层次映射到技术组件，并概述了协调、优先级和冲突解决策略。我们通过在Heero对话代理中的实现来证明该方法，其中加密的长期用户资料和对话历史已经提高了参与度和留存率。我们进一步讨论了关于存储、检索、隐私和延迟的实现问题，并指出了开放性挑战。MMAG为构建具有丰富记忆的语言代理提供了基础，这些代理更加连贯、主动，并且与人类需求更加一致。


### 论文摘要

Large Language Models (LLMs) excel at generating coherent text within a single prompt but fall short in sustaining relevance, personalization, and continuity across extended interactions. Human communication, however, relies on multiple forms of memory, from recalling past conversations to adapting to personal traits and situational context. This paper introduces the Mixed Memory-Augmented Generation (MMAG) pattern, a framework that organizes memory for LLM-based agents into five interacting layers: conversational, long-term user, episodic and event-linked, sensory and context-aware, and short-term working memory. Drawing inspiration from cognitive psychology, we map these layers to technical components and outline strategies for coordination, prioritization, and conflict resolution. We demonstrate the approach through its implementation in the Heero conversational agent, where encrypted long-term bios and conversational history already improve engagement and retention. We further discuss implementation concerns around storage, retrieval, privacy, and latency, and highlight open challenges. MMAG provides a foundation for building memory-rich language agents that are more coherent, proactive, and aligned with human needs.

---

## 123. SGDiff: Scene Graph Guided Diffusion Model for Image Collaborative SegCaptioning

**论文链接:** [http://arxiv.org/abs/2512.01975v1](http://arxiv.org/abs/2512.01975v1)

**作者:** Xu Zhang, Jin Yuan, Hanwang Zhang, Guojin Zhong, Yongsheng Zang, Jiacheng Lin, Zhiyong Li

**发布时间:** 2025-12-01

**备注:** Accept by AAAI-2025

### GPT解析

### 总结

本文提出了一种新的'图像协同分割与标注'(SegCaptioning)任务，能够将简单的用户提示转化为多样的语义解释，使用户能够灵活选择结果。

### 背景

可控图像语义理解任务（如标注或分割）需要用户输入提示来预测唯一结果，带来了高成本提示输入或有限信息输出等挑战。

### 目的

解决传统图像理解任务中提示成本高和信息输出有限的问题，通过简单提示提供多样语义解释。

### 方法

提出场景图引导扩散模型，包括提示中心场景图适配器映射用户意图，场景图引导双模态转换器预测相关标注-掩码对，以及多实体对比学习损失确保准确对齐。

### 主要发现

在两个数据集上的实验表明，SGDiff在SegCaptioning任务上取得优越性能，仅用最少提示输入就能实现良好的标注和分割效果。

### 结论

SegCaptioning任务有效解决了传统图像理解任务中的提示成本高和信息输出有限问题，为用户提供更灵活的图像理解方式。

### 翻译

可控图像语义理解任务，如标注或分割，需要用户输入提示（例如文本或边界框）来预测唯一结果，带来了诸如高成本提示输入或有限信息输出等挑战。本文介绍了一个新任务'图像协同分割与标注'(SegCaptioning)，旨在将简单的提示（如物体周围的边界框）转化为由（标注，掩码）对表示的多样语义解释，允许用户灵活选择结果。此任务带来了重大挑战，包括从最小提示中准确捕捉用户意图，同时预测多个语义一致的标注词和掩码。技术上，我们提出了一种新颖的场景图引导扩散模型，利用结构化场景图特征进行相关掩码-标注预测。首先，我们引入了一个提示中心场景图适配器，将用户提示映射到场景图，有效捕获其意图。随后，我们采用包含场景图引导双模态转换器的扩散过程，通过揭示它们之间的复杂相关性来预测相关的标注-掩码对。为确保准确对齐，我们设计了一个多实体对比学习损失，通过考虑跨模态相似性显式对齐视觉和文本实体，从而产生良好对齐的标注-掩码对。在两个数据集上进行的大量实验表明，SGDiff在SegCaptioning方面取得了优越的性能，仅用最少的提示输入就为标注和分割任务带来了有希望的结果。


### 论文摘要

Controllable image semantic understanding tasks, such as captioning or segmentation, necessitate users to input a prompt (e.g., text or bounding boxes) to predict a unique outcome, presenting challenges such as high-cost prompt input or limited information output. This paper introduces a new task ``Image Collaborative Segmentation and Captioning'' (SegCaptioning), which aims to translate a straightforward prompt, like a bounding box around an object, into diverse semantic interpretations represented by (caption, masks) pairs, allowing flexible result selection by users. This task poses significant challenges, including accurately capturing a user's intention from a minimal prompt while simultaneously predicting multiple semantically aligned caption words and masks. Technically, we propose a novel Scene Graph Guided Diffusion Model that leverages structured scene graph features for correlated mask-caption prediction. Initially, we introduce a Prompt-Centric Scene Graph Adaptor to map a user's prompt to a scene graph, effectively capturing his intention. Subsequently, we employ a diffusion process incorporating a Scene Graph Guided Bimodal Transformer to predict correlated caption-mask pairs by uncovering intricate correlations between them. To ensure accurate alignment, we design a Multi-Entities Contrastive Learning loss to explicitly align visual and textual entities by considering inter-modal similarity, resulting in well-aligned caption-mask pairs. Extensive experiments conducted on two datasets demonstrate that SGDiff achieves superior performance in SegCaptioning, yielding promising results for both captioning and segmentation tasks with minimal prompt input.

---

## 124. Robust Rigid and Non-Rigid Medical Image Registration Using Learnable Edge Kernels

**论文链接:** [http://arxiv.org/abs/2512.01771v1](http://arxiv.org/abs/2512.01771v1)

**作者:** Ahsan Raza Siyal, Markus Haltmeier, Ruth Steiger, Malik Galijasevic, Elke Ruth Gizewski, Astrid Ellen Grams

**发布时间:** 2025-12-01

### GPT解析

### 总结

该研究提出了一种结合可学习边缘核与基于学习的刚性和非刚性配准技术的方法，用于解决医学图像配准中的挑战，在多个实验中表现优于现有技术。

### 背景

医学图像配准对临床和研究应用至关重要，包括疾病诊断和治疗规划，需要配准来自不同模态、时间点或受试者的图像。传统配准技术常面临对比度差异、空间失真和模态特定变化等挑战。

### 目的

开发一种能够有效处理医学图像配准中各种挑战的方法，提高多模态图像配准和结构分析的准确性。

### 方法

提出一种整合可学习边缘核与基于学习的刚性和非刚性配准技术的方法。该方法从预定义的边缘检测核开始，然后用随机噪声扰动这些核，在训练过程中学习提取针对任务优化的边缘特征。还引入了四种刚性配准的变体模型和四种非刚性配准的变体模型。

### 主要发现

在三个设置（无头骨移除的刚性配准、有头骨移除的刚性配准和非刚性配准）中使用医科大学提供的数据集进行了评估，并在两个公开可用数据集上评估了性能。在所有实验中，该方法都优于最先进的技术。

### 结论

该方法具有提高多模态图像配准和结构分析能力的潜力，为医学图像处理提供了有效的解决方案。

### 翻译

医学图像配准对于各种临床和研究应用至关重要，包括疾病诊断或治疗规划，这些应用需要对来自不同模态、时间点或受试者的图像进行配准。传统配准技术常面临对比度差异、空间失真和模态特定变化等挑战。为解决这些局限性，我们提出了一种将可学习边缘核与基于学习的刚性和非刚性配准技术相结合的方法。与学习所有特征而没有特定偏见的传统层不同，我们的方法从预定义的边缘检测核开始，然后用随机噪声扰动这些核。这些核在训练过程中被学习，以提取针对任务优化的边缘特征。这种自适应边缘检测通过捕捉医学成像中重要的多样化结构特征，增强了配准过程。为了更清晰地了解设计中每个组件的贡献，我们为刚性配准引入了四种变体模型，为非刚性配准引入了四种变体模型。我们使用医科大学提供的数据集在三种设置下评估了我们的方法：无头骨移除的刚性配准、有头骨移除的刚性配准和非刚性配准。此外，我们还评估了在两个公开可用数据集上的性能。在所有实验中，我们的方法都持续优于最先进的技术，展示了其提高多模态图像配准和结构分析的潜力。


### 论文摘要

Medical image registration is crucial for various clinical and research applications including disease diagnosis or treatment planning which require alignment of images from different modalities, time points, or subjects. Traditional registration techniques often struggle with challenges such as contrast differences, spatial distortions, and modality-specific variations. To address these limitations, we propose a method that integrates learnable edge kernels with learning-based rigid and non-rigid registration techniques. Unlike conventional layers that learn all features without specific bias, our approach begins with a predefined edge detection kernel, which is then perturbed with random noise. These kernels are learned during training to extract optimal edge features tailored to the task. This adaptive edge detection enhances the registration process by capturing diverse structural features critical in medical imaging. To provide clearer insight into the contribution of each component in our design, we introduce four variant models for rigid registration and four variant models for non-rigid registration. We evaluated our approach using a dataset provided by the Medical University across three setups: rigid registration without skull removal, with skull removal, and non-rigid registration. Additionally, we assessed performance on two publicly available datasets. Across all experiments, our method consistently outperformed state-of-the-art techniques, demonstrating its potential to improve multi-modal image alignment and anatomical structure analysis.

---

## 125. SSR: Semantic and Spatial Rectification for CLIP-based Weakly Supervised Segmentation

**论文链接:** [http://arxiv.org/abs/2512.01701v1](http://arxiv.org/abs/2512.01701v1)

**作者:** Xiuli Bi, Die Xiao, Junchao Fan, Bin Xiao

**发布时间:** 2025-12-01

**备注:** Accepted in AAAI 2026

### GPT解析

### 总结

本文提出了一种新颖的语义和空间校正(SSR)方法，用于解决基于CLIP的弱监督语义分割方法中存在的非目标前景区域和背景区域过度激活的问题。该方法通过跨模态原型对齐和超像素引导校正两个技术手段，在PASCAL VOC和MS COCO数据集上取得了优于现有方法的性能。

### 背景

CLIP(对比语言-图像预训练)因其强大的跨模态语义理解能力，近年来被广泛应用于弱监督语义分割(WSSS)任务。

### 目的

解决现有基于CLIP的弱监督语义分割方法中存在的两个主要局限性：在非目标前景区域过度激活和在背景区域过度激活的问题。

### 方法

提出语义和空间校正(SSR)方法，包含两个层面：1)语义层面：跨模态原型对齐(CMPA)，建立对比学习机制强制跨模态特征空间对齐；2)空间层面：超像素引导校正(SGC)，利用基于超像素的空间先验过滤非目标区域干扰。

### 主要发现

在PASCAL VOC和MS COCO数据集上的实验表明，该方法优于所有单阶段方法以及更复杂的多阶段方法，分别达到了79.5%和50.6%的mIoU分数。

### 结论

通过语义和空间两个层面的校正，有效解决了现有CLIP-based弱监督语义分割方法中的过度激活问题，显著提升了分割性能。

### 翻译

近年来，对比语言-图像预训练(Clip)因其强大的跨模态语义理解能力，已被广泛应用于弱监督语义分割(WSSS)任务。本文提出了一种新颖的语义和空间校正(SSR)方法，以解决现有基于Clip的弱监督语义分割方法的局限性：在非目标前景区域和背景区域的过度激活问题。具体而言，在语义层面，跨模态原型对齐(CMPA)建立了对比学习机制，强制跨模态特征空间对齐，减少类间重叠同时增强语义相关性，有效校正非目标前景区域的过度激活；在空间层面，超像素引导校正(SGC)利用基于超像素的空间先验，在相似性传播过程中精确过滤非目标区域的干扰，显著校正背景过度激活。在PASCAL VOC和MS COCO数据集上的大量实验表明，我们的方法优于所有单阶段方法以及更复杂的多阶段方法，分别实现了79.5%和50.6%的mIoU分数。


### 论文摘要

In recent years, Contrastive Language-Image Pretraining (CLIP) has been widely applied to Weakly Supervised Semantic Segmentation (WSSS) tasks due to its powerful cross-modal semantic understanding capabilities. This paper proposes a novel Semantic and Spatial Rectification (SSR) method to address the limitations of existing CLIP-based weakly supervised semantic segmentation approaches: over-activation in non-target foreground regions and background areas. Specifically, at the semantic level, the Cross-Modal Prototype Alignment (CMPA) establishes a contrastive learning mechanism to enforce feature space alignment across modalities, reducing inter-class overlap while enhancing semantic correlations, to rectify over-activation in non-target foreground regions effectively; at the spatial level, the Superpixel-Guided Correction (SGC) leverages superpixel-based spatial priors to precisely filter out interference from non-target regions during affinity propagation, significantly rectifying background over-activation. Extensive experiments on the PASCAL VOC and MS COCO datasets demonstrate that our method outperforms all single-stage approaches, as well as more complex multi-stage approaches, achieving mIoU scores of 79.5% and 50.6%, respectively.

---

## 126. Cuffless Blood Pressure Estimation from Six Wearable Sensor Modalities in Multi-Motion-State Scenarios

**论文链接:** [http://arxiv.org/abs/2512.01653v1](http://arxiv.org/abs/2512.01653v1)

**作者:** Yiqiao Chen, Fazheng Xu, Zijian Huang, Juchi He, Zhenghui Feng

**发布时间:** 2025-12-01

**备注:** 13 pages, 7 figures

### GPT解析

### 总结

本研究提出了一种六模态血压估计框架，通过结合多种生理信号和运动数据，提高了在多种运动状态下的血压监测准确性，达到了临床标准要求。

### 背景

心血管疾病是全球发病率和死亡率的主要原因，持续的高血压是一个常见的沉默风险因素。现有无袖带血压估计方法主要使用光电容积描记信号和心电图信号，但这些方法在静态条件下开发，难以在多种运动状态下保持准确。

### 目的

开发一种能够在多种运动状态下保持准确性的血压监测方法，解决现有方法在动态环境中的局限性。

### 方法

提出六模态血压估计框架，联合利用心电图、多通道光电容积描记信号、附着压力、传感器温度和三轴加速度及角速度数据。每种模态通过轻量级分支编码器处理，对比学习强制跨模态语义对齐，专家混合回归头自适应映射融合特征到不同运动状态的血压值。

### 主要发现

该方法在收缩压(SBP)上平均绝对误差为3.60 mmHg，在舒张压(DBP)上平均绝对误差为3.01 mmHg。根据英国高血压协会(BHS)协议，该方法在收缩压、舒张压和平均动脉压(MAP)上达到A级，满足美国医疗器械促进协会(AAMI)标准的数值要求。

### 结论

所提出的六模态血压估计框架能够在多种运动状态下提供准确的血压监测，符合临床标准，为心血管疾病的早期筛查和长期管理提供了有效工具。

### 翻译

心血管疾病是全球发病率和死亡率的主要原因，持续的高血压是一个常见的沉默风险因素，这使得使用可穿戴设备进行无袖带连续血压监测对早期筛查和长期管理非常重要。大多数现有的无袖带血压估计方法仅使用光电容积描记信号和心电图信号，单独或组合使用。这些模型通常在休息或准静态条件下开发，难以在多种运动状态下保持稳健的准确性。在这项研究中，我们提出了一种六模态血压估计框架，联合利用心电图、多通道光电容积描记信号、附着压力、传感器温度和三轴加速度及角速度。每种模态通过轻量级分支编码器处理，对比学习强制跨模态语义对齐，而专家混合回归头自适应地将融合特征映射到不同运动状态的血压。在包含22名受试者跑步、行走和坐姿数据的公共脉搏传导时间光电容积描记数据集上的全面实验表明，所提出的方法在收缩压(SBP)上的平均绝对误差为3.60 mmHg，在舒张压(DBP)上的平均绝对误差为3.01 mmHg。从临床角度看，根据英国高血压协会(BHS)协议，该方法在收缩压、舒张压和平均动脉压(MAP)上达到A级，并且满足美国医疗器械促进协会(AAMI)标准的平均误差(ME)和误差标准差(SDE)的数值标准。


### 论文摘要

Cardiovascular disease (CVD) is a leading cause of morbidity and mortality worldwide, and sustained hypertension is an often silent risk factor, making cuffless continuous blood pressure (BP) monitoring with wearable devices important for early screening and long-term management. Most existing cuffless BP estimation methods use only photoplethysmography (PPG) and electrocardiography (ECG) signals, alone or in combination. These models are typically developed under resting or quasi-static conditions and struggle to maintain robust accuracy in multi-motion-state scenarios. In this study, we propose a six-modal BP estimation framework that jointly leverages ECG, multi-channel PPG, attachment pressure, sensor temperature, and triaxial acceleration and angular velocity. Each modality is processed by a lightweight branch encoder, contrastive learning enforces cross-modal semantic alignment, and a mixture-of-experts (MoE) regression head adaptively maps the fused features to BP across motion states. Comprehensive experiments on the public Pulse Transit Time PPG Dataset, which includes running, walking, and sitting data from 22 subjects, show that the proposed method achieves mean absolute errors (MAE) of 3.60 mmHg for systolic BP (SBP) and 3.01 mmHg for diastolic BP (DBP). From a clinical perspective, it attains Grade A for SBP, DBP, and mean arterial pressure (MAP) according to the British Hypertension Society (BHS) protocol and meets the numerical criteria of the Association for the Advancement of Medical Instrumentation (AAMI) standard for mean error (ME) and standard deviation of error (SDE).

---

## 127. TimePred: efficient and interpretable offline change point detection for high volume data - with application to industrial process monitoring

**论文链接:** [http://arxiv.org/abs/2512.01562v1](http://arxiv.org/abs/2512.01562v1)

**作者:** Simon Leszek

**发布时间:** 2025-12-01

**备注:** 6 pages, 3 figures

### GPT解析

### 总结

TimePred是一种自监督框架，通过预测归一化时间索引将高维大规模时间序列的多变量变化点检测简化为单变量均值偏移检测，提高了检测效率和可解释性。

### 背景

高维、大规模时间序列中的变化点检测在统计一致性、可扩展性和可解释性方面面临挑战。

### 目的

开发一种高效且可解释的变化点检测方法，适用于高维大规模时间序列数据。

### 方法

TimePred框架通过自监督学习预测每个样本的归一化时间索引，将多变量CPD问题转化为单变量均值偏移检测，并支持XAI归因方法的集成以提供特征级别的解释。

### 主要发现

TimePred实现了具有竞争力的变化点检测性能，同时将计算成本降低了高达两个数量级；在工业制造案例中展示了改进的检测精度和可解释变化点洞察的实际价值。

### 结论

TimePred为高维大规模时间序列的变化点检测提供了一种高效且可解释的解决方案，显著降低了计算成本并提高了检测准确性。

### 翻译

在高维、大规模时间序列中进行变化点检测(CPD)在统计一致性、可扩展性和可解释性方面具有挑战性。我们引入了TimePred，一个自监督框架，通过预测每个样本的归一化时间索引，将多变量CPD简化为单变量均值偏移检测。这使得使用现有算法进行高效的离线CPD成为可能，并支持集成XAI归因方法以提供特征级别的解释。我们的实验显示了具有竞争力的CPD性能，同时将计算成本降低了高达两个数量级。在一个工业制造案例研究中，我们展示了改进的检测精度，并阐明了可解释变化点洞察的实际价值。


### 论文摘要

Change-point detection (CPD) in high-dimensional, large-volume time series is challenging for statistical consistency, scalability, and interpretability. We introduce TimePred, a self-supervised framework that reduces multivariate CPD to univariate mean-shift detection by predicting each sample's normalized time index. This enables efficient offline CPD using existing algorithms and supports the integration of XAI attribution methods for feature-level explanations. Our experiments show competitive CPD performance while reducing computational cost by up to two orders of magnitude. In an industrial manufacturing case study, we demonstrate improved detection accuracy and illustrate the practical value of interpretable change-point insights.

---

## 128. Data-Driven Learnability Transition of Measurement-Induced Entanglement

**论文链接:** [http://arxiv.org/abs/2512.01317v1](http://arxiv.org/abs/2512.01317v1)

**作者:** Dongheng Qian, Jing Wang

**发布时间:** 2025-12-01

**备注:** 7 pages, 4 figures

### GPT解析

### 总结

本研究提出了一种基于数据驱动学习的方法来估计测量诱导纠缠(MIE)，解决了实验评估MIE的挑战。通过将MIE检测重新框架化为数据驱动学习问题，研究人员使用测量记录训练神经网络来预测MIE的不确定性指标。该方法在一维全连通和二维最近邻耦合的随机电路上展示了可学习性转变，并在当前嘈杂的量子设备上进行了实验验证。

### 背景

测量诱导纠缠(MIE)描述了局部测量如何在多体系统中产生长程量子关联并驱动动力学相变。然而，实验估计MIE仍然具有挑战性，因为直接评估需要对测量结果进行大量后选择，这引发了MIE是否仅通过多项式资源就可达到的问题。

### 目的

解决实验估计MIE的挑战，通过将MIE检测重新框架化为数据驱动学习问题，无需预先了解状态制备，仅使用测量记录来训练神经网络，以预测MIE的不确定性指标。

### 方法

将MIE检测重新框架化为数据驱动学习问题，使用测量记录以自监督方式训练神经网络，预测MIE的不确定性指标（即后测量双纠缠上下界之间的差距）。该方法应用于具有一维全连通性和二维最近邻耦合的随机电路。

### 主要发现

随着电路深度的增加，方法揭示了一个可学习性转变：在阈值以下，不确定性小且随多项式测量数据和模型参数增加而减小；在阈值以上，尽管资源增加，不确定性仍然很大。在当前嘈杂的量子设备上实验验证了这种转变，证明了其对实际噪声的鲁棒性。

### 结论

这些结果强调了数据驱动方法在学习MIE方面的能力，并界定了其经典可学习性的实际界限。

### 翻译

测量诱导纠缠(MIE)描述了局部测量如何在多体系统中产生长程量子关联并驱动动力学相变。然而，实验估计MIE仍然具有挑战性：直接评估需要对测量结果进行大量后选择，这引发了MIE是否仅通过多项式资源就可达到的问题。我们通过将MIE检测重新框架化为数据驱动学习问题来解决这一挑战，该方法假设不预先了解状态制备。仅使用测量记录，我们以自监督方式训练神经网络来预测MIE的不确定性指标——后测量双纠缠上下界之间的差距。将其应用于具有一维全连通性和二维最近邻耦合的随机电路，我们的方法揭示了随着电路深度增加的可学习性转变：在阈值以下，不确定性小且随多项式测量数据和模型参数增加而减小，而在阈值以上，尽管资源增加，不确定性仍然很大。我们在当前嘈杂的量子设备上进一步实验验证了这种转变，证明了其对实际噪声的鲁棒性。这些结果强调了数据驱动方法在学习MIE方面的能力，并界定了其经典可学习性的实际界限。


### 论文摘要

Measurement-induced entanglement (MIE) captures how local measurements generate long-range quantum correlations and drive dynamical phase transitions in many-body systems. Yet estimating MIE experimentally remains challenging: direct evaluation requires extensive post-selection over measurement outcomes, raising the question of whether MIE is accessible with only polynomial resources. We address this challenge by reframing MIE detection as a data-driven learning problem that assumes no prior knowledge of state preparation. Using measurement records alone, we train a neural network in a self-supervised manner to predict the uncertainty metric for MIE--the gap between upper and lower bounds of the average post-measurement bipartite entanglement. Applied to random circuits with one-dimensional all-to-all connectivity and two-dimensional nearest-neighbor coupling, our method reveals a learnability transition with increasing circuit depth: below a threshold, the uncertainty is small and decreases with polynomial measurement data and model parameters, while above it the uncertainty remains large despite increasing resources. We further verify this transition experimentally on current noisy quantum devices, demonstrating its robustness to realistic noise. These results highlight the power of data-driven approaches for learning MIE and delineate the practical limits of its classical learnability.

---

## 129. Samplability makes learning easier

**论文链接:** [http://arxiv.org/abs/2512.01276v1](http://arxiv.org/abs/2512.01276v1)

**作者:** Guy Blanc, Caleb Koch, Jane Lange, Carmen Strassle, Li-Yang Tan

**发布时间:** 2025-12-01

**备注:** ITCS 2026

### GPT解析

### 总结

该研究探讨了标准PAC学习与可采样PAC学习之间的区别，证明了可采样PAC学习显著扩展了高效学习者的能力。作者构建了一个概念类，在标准PAC学习中需要指数级样本复杂度，但在可采样PAC学习中只需多项式样本复杂度。研究还引入了一种新的复杂性原语——显式回避集合，并将结果扩展到在线学习场景。

### 背景

标准PAC学习（Valiant 1984）要求学习者在所有分布下都能成功，即使这些分布难以从中采样。这与可采样PAC学习（Blum, Furst, Kearns, 和 Lipton 1993）形成对比，后者只要求学习者在可采样分布下成功。

### 目的

研究标准PAC学习与可采样PAC学习之间的区别，并证明可采样PAC学习如何显著扩展高效学习者的能力。

### 方法

1. 构建一个在标准PAC学习中需要指数级样本复杂度，但在可采样PAC学习中只需多项式样本复杂度的概念类；2. 将这种统计分离提升到计算设置中，获得相对于随机预言机的分离；3. 引入并研究一种新的复杂性原语——显式回避集合；4. 将结果扩展到在线学习场景。

### 主要发现

1. 可采样PAC学习显著扩展了高效学习者的能力；2. 存在一个概念类，在标准PAC学习中需要指数级样本复杂度，但在可采样PAC学习中只需多项式样本复杂度；3. 在计算设置中获得了相对于随机预言机的分离；4. 引入了显式回避集合这一新的复杂性原语，其成员资格容易确定但极难从中采样；5. 当假设对手是高效的而非计算无界时，在线学习的景观会发生变化。

### 结论

可采样PAC学习比标准PAC学习具有更强的能力，特别是在样本复杂度方面。通过引入显式回避集合这一新概念，研究者能够证明两种学习范式之间的显著差异，并将这一结果从统计学习扩展到计算学习和在线学习领域。

### 翻译

标准PAC学习的定义（Valiant 1984）要求学习者在所有分布下都能成功——即使是难以从中采样的分布。这与可采样PAC学习（Blum, Furst, Kearns, 和 Lipton 1993）形成对比，后者只要求学习者在可采样分布下成功。我们研究这种区别并证明可采样PAC学习显著扩展了高效学习者的能力。我们首先构建了一个概念类，它在标准PAC学习中需要指数级样本复杂度，但在可采样PAC学习中可以用多项式样本复杂度学习。然后我们将这种统计分离提升到计算设置中，获得了相对于随机预言机的分离。我们的证明围绕一种新的复杂性原语——显式回避集合展开，这是我们引入并研究的。这些集合的成员资格容易确定但极难从中采样。我们的结果还扩展到在线设置，类似地展示了当假设对手是高效的而非计算无界时，其景观如何变化。


### 论文摘要

The standard definition of PAC learning (Valiant 1984) requires learners to succeed under all distributions -- even ones that are intractable to sample from. This stands in contrast to samplable PAC learning (Blum, Furst, Kearns, and Lipton 1993), where learners only have to succeed under samplable distributions. We study this distinction and show that samplable PAC substantially expands the power of efficient learners.   We first construct a concept class that requires exponential sample complexity in standard PAC but is learnable with polynomial sample complexity in samplable PAC. We then lift this statistical separation to the computational setting and obtain a separation relative to a random oracle. Our proofs center around a new complexity primitive, explicit evasive sets, that we introduce and study. These are sets for which membership is easy to determine but are extremely hard to sample from.   Our results extend to the online setting to similarly show how its landscape changes when the adversary is assumed to be efficient instead of computationally unbounded.

---

## 130. TRivia: Self-supervised Fine-tuning of Vision-Language Models for Table Recognition

**论文链接:** [http://arxiv.org/abs/2512.01248v1](http://arxiv.org/abs/2512.01248v1)

**作者:** Junyuan Zhang, Bin Wang, Qintong Zhang, Fan Wu, Zichen Wen, Jialin Lu, Junjie Shan, Ziqi Zhao, Shuya Yang, Ziling Wang, Ziyang Miao, Huaping Zhong, Yuhang Zang, Xiaoyi Dong, Ka-Ho Chow, Conghui He

**发布时间:** 2025-12-01

### GPT解析

### 总结

论文介绍了一种名为TRivia的自监督微调方法，使预训练的视觉语言模型能够直接从未标记的表格图像中学习表格识别，并提出了一个名为TRivia-3B的开源模型，在三个流行基准测试上超越了现有系统。

### 背景

表格识别(TR)旨在将表格图像转换为半结构化表示(如HTML或Markdown)。作为文档解析的核心组成部分，TR长期以来一直依赖监督学习，最近的努力主要是使用标记数据微调视觉语言模型(VLMs)。虽然VLMs已将TR提升到新水平，但进一步推动性能需要大规模标记数据，而这些数据获取成本高昂。因此，专有模型不断推动性能边界，而开源模型由于资源有限且受隐私法规限制，仍远远落后。

### 目的

为了缩小专有模型和开源模型之间的性能差距，作者提出了TRivia方法，使预训练的VLMs能够直接从未标记的表格图像中学习表格识别，无需人工标记。

### 方法

TRivia基于组相对策略优化，能够自动识别最有效促进学习的未标记样本，并通过基于问答的奖励机制消除了对人工注释的需求。一个注意力引导的模块为每个表格图像生成多样化的问题，能够解释识别结果并正确回答这些问题，为优化TR模型提供反馈。这种闭环过程使TR模型能够在没有标记数据的情况下自主学习识别、结构和推理表格。

### 主要发现

利用此流程，作者提出了TRivia-3B，这是一个开源的、紧凑的、最先进的TR模型，在三个流行的基准测试上超越了现有系统(如Gemini 2.5 Pro、MinerU2.5)。

### 结论

TRivia方法成功地缩小了专有模型和开源模型在表格识别任务上的性能差距，提供了一个无需大量标记数据就能训练高性能TR模型的途径。

### 翻译

表格识别(TR)旨在将表格图像转换为半结构化表示，如HTML或Markdown。作为文档解析的核心组成部分，TR长期以来一直依赖监督学习，最近的努力主要是使用标记数据微调视觉语言模型(VLMs)。虽然VLMs已将TR提升到新水平，但进一步推动性能需要大规模标记数据，而这些数据获取成本高昂。因此，虽然专有模型不断推动性能边界，但开源模型通常受限于有限资源，且在实践上是许多人的唯一可行选择(由于隐私法规)，仍远远落后。为了缩小这一差距，我们引入了TRivia，一种自监督微调方法，使预训练的VLMs能够直接从未标记的表格图像中学习TR。基于组相对策略优化，TRivia自动识别最有效促进学习的未标记样本，并通过基于问答的奖励机制消除了对人工注释的需求。一个注意力引导的模块为每个表格图像生成多样化的问题，能够解释识别结果并正确回答这些问题，为优化TR模型提供反馈。这种闭环过程使TR模型能够在没有标记数据的情况下自主学习识别、结构和推理表格。利用此流程，我们提出了TRivia-3B，一个开源的、紧凑的、最先进的TR模型，在三个流行的基准测试上超越了现有系统(如Gemini 2.5 Pro、MinerU2.5)。模型和代码发布于：https://github.com/opendatalab/TRivia


### 论文摘要

Table recognition (TR) aims to transform table images into semi-structured representations such as HTML or Markdown. As a core component of document parsing, TR has long relied on supervised learning, with recent efforts dominated by fine-tuning vision-language models (VLMs) using labeled data. While VLMs have brought TR to the next level, pushing performance further demands large-scale labeled data that is costly to obtain. Consequently, although proprietary models have continuously pushed the performance boundary, open-source models, often trained with limited resources and, in practice, the only viable option for many due to privacy regulations, still lag far behind. To bridge this gap, we introduce TRivia, a self-supervised fine-tuning method that enables pretrained VLMs to learn TR directly from unlabeled table images in the wild. Built upon Group Relative Policy Optimization, TRivia automatically identifies unlabeled samples that most effectively facilitate learning and eliminates the need for human annotations through a question-answering-based reward mechanism. An attention-guided module generates diverse questions for each table image, and the ability to interpret the recognition results and answer them correctly provides feedback to optimize the TR model. This closed-loop process allows the TR model to autonomously learn to recognize, structure, and reason over tables without labeled data. Leveraging this pipeline, we present TRivia-3B, an open-sourced, compact, and state-of-the-art TR model that surpasses existing systems (e.g., Gemini 2.5 Pro, MinerU2.5) on three popular benchmarks. Model and code are released at: https://github.com/opendatalab/TRivia

---

## 131. Pay Attention Later: From Vector Space Diffusion to Linearithmic Spectral Phase-Locking

**论文链接:** [http://arxiv.org/abs/2512.01208v1](http://arxiv.org/abs/2512.01208v1)

**作者:** Alper Yıldırım, İbrahim Yücedağ

**发布时间:** 2025-12-01

**备注:** 12 pages, 5 figures

### GPT解析

### 总结

本研究探讨了标准Transformer模型在语义对齐方面存在的问题，并提出了一种新的PRISM模型来解决模型适应新概念时的灾难性遗忘问题。

### 背景

标准Transformer模型遭受'语义对齐税'，即通过局部梯度扩散将混乱初始化组织成连贯几何图所需的高昂优化成本。这种对扩散学习的依赖导致'灾难性刚性'，使模型无法在不破坏预训练推理能力的情况下适应新概念。

### 目的

隔离并解决Transformer模型中的'灾难性刚性'现象，实现模型在保留原有知识的同时有效学习新概念，解决实时知识适应中的可塑性与稳定性困境。

### 方法

引入迭代语义图细化(ISMR)作为诊断协议，并提出相位共振智能谱模型(PRISM)。PRISM将语义身份编码为复域中的共振频率，用线性复杂度的门控谐波卷积替代二次自注意力机制。

### 主要发现

对齐是固定的几何障碍，模型规模扩展无法解决；在WMT14翻译任务上，标准Transformer在静态基准测试上略优(23.88 vs 21.40 BLEU)，但在可塑性-稳定性压力测试中完全失败，遭受灾难性遗忘(BLEU下降10.55点，仅60%概念获取)；相比之下，PRISM实现无损可塑性(96%的5-shot获取，BLEU仅下降0.84点)。

### 结论

谐波表示能有效将记忆与推理解耦，为实时知识适应中的可塑性与稳定性困境提供了结构化解决方案。

### 翻译

标准Transformer模型遭受'语义对齐税'，这是一种通过局部梯度扩散将混乱初始化组织成连贯几何图所需的高昂优化成本。我们假设这种对扩散学习的依赖导致'灾难性刚性'，使模型无法在不破坏预训练推理能力的情况下适应新概念。为隔离这一现象，我们引入迭代语义图细化(ISMR)，这是一种诊断协议，揭示对齐是一个固定的几何障碍，规模扩展无法解决；20层模型克服这一障碍的速度并不比1层模型快。我们提出相位共振智能谱模型(PRISM)。PRISM将语义身份编码为复域中的共振频率，并用线性复杂度O(N log N)的门控谐波卷积替代二次自注意力。我们在WMT14翻译任务上验证了PRISM。虽然标准Transformer在静态基准测试上保持轻微优势(23.88对21.40 BLEU)，但它完全未能通过'可塑性-稳定性'压力测试。当注入新概念时，Transformer遭受灾难性遗忘，BLEU下降10.55点，仅实现60%的概念获取。相比之下，PRISM展示了无损可塑性，实现了96%的5-shot获取，BLEU仅下降0.84点。这些结果表明，谐波表示能有效将记忆与推理解耦，为实时知识适应中的可塑性与稳定性困境提供了结构化解决方案。


### 论文摘要

Standard Transformers suffer from a "Semantic Alignment Tax", a prohibitive optimization cost required to organize a chaotic initialization into a coherent geometric map via local gradient diffusion. We hypothesize that this reliance on diffusive learning creates "Catastrophic Rigidity", rendering models unable to adapt to novel concepts without destroying their pre-trained reasoning capabilities. To isolate this phenomenon, we introduce Iterative Semantic Map Refinement (ISMR), a diagnostic protocol revealing that alignment is a fixed geometric barrier that scaling cannot solve; a 20-layer model overcomes this barrier no faster than a 1-layer model. We introduce the Phase-Resonant Intelligent Spectral Model (PRISM). PRISM encodes semantic identity as resonant frequencies in the complex domain (C^d) and replaces quadratic self-attention with linearithmic O(N log N) Gated Harmonic Convolutions. We validate PRISM on the WMT14 translation task. While the Standard Transformer maintains a slight edge in general competence on static benchmarks (23.88 vs 21.40 BLEU), it fails the "Plasticity-Stability" stress test completely. When injected with novel concepts, the Transformer suffers Catastrophic Forgetting, degrading by -10.55 BLEU points while achieving only 60% acquisition. In contrast, PRISM demonstrates Lossless Plasticity, achieving 96% 5-shot acquisition with negligible degradation (-0.84 BLEU). These results suggest that harmonic representations effectively decouple memory from reasoning, offering a structural solution to the plasticity-stability dilemma in real-time knowledge adaptation.

---

## 132. fMRI2GES: Co-speech Gesture Reconstruction from fMRI Signal with Dual Brain Decoding Alignment

**论文链接:** [http://arxiv.org/abs/2512.01189v1](http://arxiv.org/abs/2512.01189v1)

**作者:** Chunzheng Zhu, Jialin Shao, Jianxin Lin, Yijun Wang, Jing Wang, Jinhui Tang, Kenli Li

**发布时间:** 2025-12-01

**DOI:** 10.1109/TCSVT.2025.3558125

**备注:** IEEE Transactions on Circuits and Systems for Video Technology (TCSVT) 2025

### GPT解析

### 总结

本文提出了一种名为fMRI2GES的新方法，通过双脑解码对齐技术，利用非配对数据训练fMRI到手势重建网络，成功实现了从脑部活动记录中重建与言语相关的手势。

### 背景

理解大脑如何响应外部刺激并解码这一过程是神经科学中的重大挑战。以往研究主要集中在脑到图像和脑到语言的重建，而缺乏配对的{大脑、言语、手势}数据阻碍了深度学习模型在此领域的应用。

### 目的

重建与言语刺激相关联的手势，探索大脑活动与手势表达之间的关系。

### 方法

提出fMRI2GES方法，利用双脑解码对齐技术，在非配对数据上训练fMRI到手势重建网络。该方法依赖于引发大脑反应的观察文本和与手势相关联的文本描述，建立两种fMRI到手势重建模式，并对齐两个输出以自监督方式训练模型。

### 主要发现

所提方法可以直接从fMRI记录重建表达性手势；研究了大脑皮层中不同ROI的fMRI信号及其对生成结果的影响。

### 结论

为解码伴随言语的手势提供了新的见解，推进了神经科学和认知科学领域的发展。

### 翻译

理解大脑如何响应外部刺激并解码这一过程一直是神经科学中的一个重大挑战。虽然以往的研究通常集中在脑到图像和脑到语言的重建上，但我们的工作致力于重建与言语刺激相关联的手势。遗憾的是，缺乏配对的{大脑、言语、手势}数据阻碍了深度学习模型在此目的上的部署。在本文中，我们介绍了一种新方法fMRI2GES，它允许使用双脑解码对齐技术在非配对数据上训练fMRI到手势重建网络。该方法依赖于两个关键组成部分：(i)引发大脑反应的观察文本，和(ii)与手势相关联的文本描述。然后，我们不是以完全监督的方式训练模型来寻找三种模态之间的映射关系，而是利用一个fMRI到文本模型、一个有配对数据的文本到手势模型和一个有非配对数据的fMRI到手势模型，建立两种fMRI到手势重建模式。之后，我们明确对齐两个输出并以自监督方式训练我们的模型。我们表明，所提出的方法可以直接从fMRI记录重建表达性手势。我们还研究了大脑皮层中不同ROI的fMRI信号及其对生成结果的影响。总体而言，我们为解码伴随言语的手势提供了新的见解，从而推进了我们对神经科学和认知科学的理解。


### 论文摘要

Understanding how the brain responds to external stimuli and decoding this process has been a significant challenge in neuroscience. While previous studies typically concentrated on brain-to-image and brain-to-language reconstruction, our work strives to reconstruct gestures associated with speech stimuli perceived by brain. Unfortunately, the lack of paired \{brain, speech, gesture\} data hinders the deployment of deep learning models for this purpose. In this paper, we introduce a novel approach, \textbf{fMRI2GES}, that allows training of fMRI-to-gesture reconstruction networks on unpaired data using \textbf{Dual Brain Decoding Alignment}. This method relies on two key components: (i) observed texts that elicit brain responses, and (ii) textual descriptions associated with the gestures. Then, instead of training models in a completely supervised manner to find a mapping relationship among the three modalities, we harness an fMRI-to-text model, a text-to-gesture model with paired data and an fMRI-to-gesture model with unpaired data, establishing dual fMRI-to-gesture reconstruction patterns. Afterward, we explicitly align two outputs and train our model in a self-supervision way. We show that our proposed method can reconstruct expressive gestures directly from fMRI recordings. We also investigate fMRI signals from different ROIs in the cortex and how they affect generation results. Overall, we provide new insights into decoding co-speech gestures, thereby advancing our understanding of neuroscience and cognitive science.

---

## 133. Diffusion-Based Synthesis of 3D T1w MPRAGE Images from Multi-Echo GRE with Multi-Parametric MRI Integration

**论文链接:** [http://arxiv.org/abs/2512.01135v1](http://arxiv.org/abs/2512.01135v1)

**作者:** Sizhe Fang, Deqiang Qiu

**发布时间:** 2025-11-30

### GPT解析

### 总结

研究提出了一种基于扩散模型的深度学习方法，可以直接从mGRE数据合成高对比度的T1w MPRAGE图像，无需额外的扫描时间。该方法整合了QSM和R2*图作为物理先验，特别处理富含铁质的深灰质区域的对比度问题。在175名健康受试者上的实验表明，该方法在感知质量和分割精度上优于传统方法，且保持了重要的生物依赖性，如年龄和性别相关的变化模式。

### 背景

Multi-echo Gradient Echo (mGRE)序列提供有价值的定量参数图，如定量磁敏感成像(QSM)和横向弛豫率(R2*)，这些对组织铁和髓鞘敏感。然而，结构形态学通常依赖于单独的T1加权MPRAGE采集，延长了扫描时间。

### 目的

提出一个深度学习框架，直接从mGRE数据合成高对比度的3D T1w MPRAGE图像，简化神经成像方案。

### 方法

开发了一种基于Fast-DDPM架构的新型多参数条件扩散模型。与传统基于强度的合成不同，该方法整合了对铁敏感的QSM和R2*图作为物理先验，解决富含铁质的深灰质中的对比度模糊问题。在175名健康受试者上训练和验证模型，使用感知指标和下游分割精度与U-Net和基于GAN的基线比较评估性能。通过复制与年龄和性别的人群水平统计关联，评估合成图像的生物合理性。

### 主要发现

所提出的框架显著优于基线，实现了卓越的感知质量和分割精度，特别是在丘脑和苍白球等皮层下区域。关键的是，合成的图像保留了必要的生物依赖性：回归分析显示，与真实数据相比，在年龄相关的萎缩率、衰老效应大小和性别二态性模式方面高度一致。

### 结论

通过有效利用定量MRI先验，基于扩散的方法生成严格生物合理的T1w图像，适用于可靠的临床形态学分析。这种方法通过从定量mGRE序列回顾性获取结构对比度，为减少采集时间提供了有前景的途径。

### 翻译

多回波梯度回波(mGRE)序列提供有价值的定量参数图，如定量磁敏感成像(QSM)和横向弛豫率(R2*)，对组织铁和髓鞘敏感。然而，结构形态学通常依赖于单独的T1加权MPRAGE采集，延长了扫描时间。我们提出一个深度学习框架，直接从mGRE数据合成高对比度的3D T1w MPRAGE图像，简化神经成像方案。我们开发了一种基于Fast-DDPM架构的新型多参数条件扩散模型。与传统的基于强度的合成不同，我们的方法整合了对铁敏感的QSM和R2*图作为物理先验，解决富含铁质的深灰质中的对比度模糊问题。我们在175名健康受试者上训练和验证了模型。使用感知指标和下游分割精度与已建立的U-Net和基于GAN的基线比较评估性能。独特的是，我们通过复制与年龄和性别的人群水平统计关联，评估了合成图像的生物合理性。所提出的框架显著优于基线，实现了卓越的感知质量和分割精度，特别是在丘脑和苍白球等皮层下区域。关键的是，合成的图像保留了必要的生物依赖性：回归分析显示，与真实数据相比，在年龄相关的萎缩率、衰老效应大小和性别二态性模式方面高度一致。通过有效利用定量MRI先验，我们的基于扩散的方法生成严格生物合理的T1w图像，适用于可靠的临床形态学分析。这种方法通过从定量mGRE序列回顾性获取结构对比度，为减少采集时间提供了有前景的途径。


### 论文摘要

Multi-echo Gradient Echo (mGRE) sequences provide valuable quantitative parametric maps, such as Quantitative Susceptibility Mapping (QSM) and transverse relaxation rate (R2*), sensitive to tissue iron and myelin. However, structural morphometry typically relies on separate T1-weighted MPRAGE acquisitions, prolonging scan times. We propose a deep learning framework to synthesize high-contrast 3D T1w MPRAGE images directly from mGRE data, streamlining neuroimaging protocols. We developed a novel multi-parametric conditional diffusion model based on the Fast-DDPM architecture. Unlike conventional intensity-based synthesis, our approach integrates iron-sensitive QSM and R2* maps as physical priors to address contrast ambiguity in iron-rich deep gray matter. We trained and validated the model on 175 healthy subjects. Performance was evaluated against established U-Net and GAN-based baselines using perceptual metrics and downstream segmentation accuracy. Uniquely, we assessed the biological plausibility of synthesized images by replicating population-level statistical associations with age and sex. The proposed framework significantly outperformed baselines, achieving superior perceptual quality and segmentation accuracy, particularly in subcortical regions like the thalamus and pallidum. Crucially, synthesized images preserved essential biological dependencies: regression analyses showed high concordance in age-related atrophy rates, aging effect sizes, and sexual dimorphism patterns compared to ground truth. By effectively leveraging quantitative MRI priors, our diffusion-based method generates strictly biologically plausible T1w images suitable for reliable clinical morphometric analysis. This approach offers a promising pathway to reduce acquisition time by deriving structural contrasts retrospectively from quantitative mGRE sequences.

---

## 134. Bayesian dynamic scheduling of multipurpose batch processes under incomplete look-ahead information

**论文链接:** [http://arxiv.org/abs/2512.01093v1](http://arxiv.org/abs/2512.01093v1)

**作者:** Taicheng Zheng, Dan Li, Jie Li

**发布时间:** 2025-11-30

### GPT解析

### 总结

本文提出了一种贝叶斯动态调度方法，用于处理多批次制造过程中的动态环境干扰，通过学习干扰的概率分布构建贝叶斯网络，在线执行时根据观察到的干扰更新后验分布，指导重调度策略。

### 背景

多批次流程在制造业中越来越受欢迎，因为它们适应小批量、高价值产品和变化的需求。这些流程通常在动态环境中运行，面临加工延迟和需求变化等干扰。现有方法假设在调度范围内具有完整的先验信息，这与实际情况不符。

### 目的

为了最小化长期成本和系统紧张度（即对计划的破坏性变化），设计有效的重调度策略来应对干扰，解决现有方法在处理不完整先验信息时的局限性。

### 方法

提出一种贝叶斯动态调度方法，从干扰的概率分布中学习贝叶斯网络，表示每个操作可能受到干扰影响的可能性。在线执行过程中，当观察到新的干扰时，更新后验分布，从而指导重调度策略。

### 主要发现

在四个基准问题上，该方法与周期性重调度策略相比，在统计上实现了更好的长期成本和系统紧张度。理论上证明了如果干扰相互独立，影响量化变量本质上满足贝叶斯网络所需的独立性假设。

### 结论

所提出的贝叶斯动态调度方法能够有效处理多批次制造过程中的干扰，优于传统的周期性重调度策略，并可扩展到其他调度问题，只要定义了操作之间的特定依赖关系。

### 翻译

多批次流程在制造业中越来越受欢迎，因为它们适应小批量、高价值产品和变化的需求。这些流程通常在动态环境中运行，面临加工延迟和需求变化等干扰。为了最小化长期成本和系统紧张度（即对计划的破坏性变化），调度者必须设计重调度策略来有效应对这些干扰。现有方法通常假设在调度范围内具有完整的先验信息，这一假设与实际情况不符。在这项工作中，我们提出了一种贝叶斯动态调度方法，依赖于从干扰的概率分布中学习贝叶斯网络，在线执行时根据观察到的干扰更新后验分布，指导重调度策略。计算结果表明，我们的方法在统计上实现了更好的长期成本和系统紧张度。理论上证明了如果干扰相互独立，影响量化变量本质上满足贝叶斯网络所需的独立性假设。作为一个推论，实践者可以将该方法扩展到其他调度问题，只要他们定义了操作之间的特定依赖关系。


### 论文摘要

Multipurpose batch processes become increasingly popular in manufacturing industries since they adapt to low-volume, high-value products and shifting demands. These processes often operate in a dynamic environment, which faces disturbances such as processing delays and demand changes. To minimise long-term cost and system nervousness (i.e., disruptive changes to schedules), schedulers must design rescheduling strategies to address such disturbances effectively. Existing methods often assume complete look-ahead information over the scheduling horizon. This assumption contrasts with realistic situations where schedulers can only access incomplete look-ahead information. Sticking with existing methods may lead to suboptimal long-term costs and high-level system nervousness. In this work we propose a Bayesian dynamic scheduling method. Our method relies on learning a Bayesian Network from the probability distribution of disturbances. Specifically, the Bayesian Network represents how likely each operation will be impacted by disturbances. During the online execution, when new disturbances become observed, this method updates the posterior distribution and therefore guides the rescheduling strategy. We compare our method with the existing periodic rescheduling strategy (which generates new schedules from scratch at fixed intervals) on four benchmark problems. Computational results show that our method achieves statistically better long-term costs and system nervousness. In the theoretical aspect, we prove that if disturbances are mutually independent, the impact-quantifying variables inherently satisfy the independence assumptions required by Bayesian Networks. As an implication, practitioners can extend the method to other scheduling problems (such as job shop scheduling and continuous processes), given that they define the problem-specific dependencies between operations.

---

## 135. Outcome-Aware Spectral Feature Learning for Instrumental Variable Regression

**论文链接:** [http://arxiv.org/abs/2512.00919v1](http://arxiv.org/abs/2512.00919v1)

**作者:** Dimitri Meunier, Jakub Wornbard, Vladimir R. Kostic, Antoine Moulin, Alek Fröhlich, Karim Lounici, Massimiliano Pontil, Arthur Gretton

**发布时间:** 2025-11-30

### GPT解析

### 总结

本文提出了一种增强谱特征学习方法，用于在存在隐藏混杂因素的情况下进行因果效应估计，解决了传统基于谱特征的估计器不考虑结果变量的问题。

### 背景

在存在隐藏混杂因素的情况下，使用非参数工具变量(IV)回归进行因果效应估计是一个重要问题。现有方法基于学习到的谱特征，但这些特征不考虑结果变量。

### 目的

开发一种对结果变量敏感的特征学习方法，使因果效应估计在真实因果函数不能很好地被主导奇异函数表示的情况下仍然有效。

### 方法

提出增强谱特征学习框架，通过最小化从融入结果信息的增强算子导出的新型对比损失来学习特定任务的特征。

### 主要发现

通过学习这些特定任务的特征，即使在谱失配的情况下，该方法仍然有效。

### 结论

增强谱特征学习框架在理论上得到分析，并在具有挑战性的基准上验证了其有效性。

### 翻译

我们使用非参数工具变量回归解决了存在隐藏混杂因素时的因果效应估计问题。一种既定方法是使用基于学习到的谱特征的估计器，即跨越工具变量与治疗之间算子的前奇异子空间的特征。虽然这些功能很强大，但它们不考虑结果变量。因此，当真实因果函数不能很好地被这些主导奇异函数表示时，该方法可能会失败。为减轻这一问题，我们引入了增强谱特征学习，这是一个使特征学习过程对结果变量敏感的框架。我们的方法通过最小化从融入结果信息的增强算子导出的新型对比损失来学习特征。通过学习这些特定任务的特征，即使在谱失配的情况下，我们的方法仍然有效。我们提供了该框架的理论分析，并在具有挑战性的基准上验证了我们的方法。


### 论文摘要

We address the problem of causal effect estimation in the presence of hidden confounders using nonparametric instrumental variable (IV) regression. An established approach is to use estimators based on learned spectral features, that is, features spanning the top singular subspaces of the operator linking treatments to instruments. While powerful, such features are agnostic to the outcome variable. Consequently, the method can fail when the true causal function is poorly represented by these dominant singular functions. To mitigate, we introduce Augmented Spectral Feature Learning, a framework that makes the feature learning process outcome-aware. Our method learns features by minimizing a novel contrastive loss derived from an augmented operator that incorporates information from the outcome. By learning these task-specific features, our approach remains effective even under spectral misalignment. We provide a theoretical analysis of this framework and validate our approach on challenging benchmarks.

---

## 136. Limitations of Using Identical Distributions for Training and Testing When Learning Boolean Functions

**论文链接:** [http://arxiv.org/abs/2512.00791v1](http://arxiv.org/abs/2512.00791v1)

**作者:** Jordi Pérez-Guijarro

**发布时间:** 2025-11-30

### GPT解析

### 总结

研究训练分布与测试分布不一致情况下的泛化问题，发现匹配分布并不总是最优策略

### 背景

当训练数据和测试数据的分布不匹配时，理解泛化问题变得更为复杂

### 目的

探究训练分布是否总是应该与测试分布相同这一基本问题

### 方法

基于单向函数存在性的假设进行分析

### 主要发现

在存在单向函数的前提下，匹配分布并不总是最佳情况，这与大多数学习方法的行为形成对比

### 结论

当对目标函数施加某些规律性条件时，在均匀分布的情况下，标准结论仍然成立

### 翻译

当训练数据和测试数据的分布不一致时，理解泛化问题会变得更加复杂，引发了许多问题。在这项工作中，我们关注一个基本问题：训练分布是否总是应该与测试分布相同？令人惊讶的是，在假设存在单向函数的前提下，我们发现答案是否定的。也就是说，匹配分布并不总是最佳情况，这与大多数学习方法的行为形成对比。尽管如此，我们也表明，当对目标函数施加某些规律性条件时，在均匀分布的情况下，标准结论仍然成立。


### 论文摘要

When the distributions of the training and test data do not coincide, the problem of understanding generalization becomes considerably more complex, prompting a variety of questions. In this work, we focus on a fundamental one: Is it always optimal for the training distribution to be identical to the test distribution? Surprisingly, assuming the existence of one-way functions, we find that the answer is no. That is, matching distributions is not always the best scenario, which contrasts with the behavior of most learning methods. Nonetheless, we also show that when certain regularities are imposed on the target functions, the standard conclusion is recovered in the case of the uniform distribution.

---

## 137. Sign Language Recognition using Bidirectional Reservoir Computing

**论文链接:** [http://arxiv.org/abs/2512.00777v1](http://arxiv.org/abs/2512.00777v1)

**作者:** Nitin Kumar Singh, Arie Rachmad Syulistyo, Yuichiro Tanaka, Hakaru Tamukoh

**发布时间:** 2025-11-30

### GPT解析

### 总结

该研究提出了一种基于MediaPipe和双向储备计算(BRC)架构的高效手语识别系统，显著降低了计算需求，使其适用于资源受限的边缘设备。

### 背景

手语识别(SLR)促进聋人和听力正常人士之间的交流。深度学习虽被广泛用于开发SLR系统，但计算密集且需要大量资源，不适合资源受限设备。

### 目的

开发一种高效的手语识别系统，解决深度学习方法在资源受限设备上的适用性问题。

### 方法

使用MediaPipe提取手部关节坐标作为输入，结合基于回声状态网络(ESN)的双向储备计算(BRC)架构，通过前后双向处理特征捕获时间依赖性，并将结果状态连接形成鲁棒表示用于分类。

### 主要发现

在WLASL视频数据集上达到57.71%的竞争性准确率，训练时间仅需9秒，而深度学习的Bi-GRU方法需要55分38秒。

### 结论

基于BRC的SLR系统计算效率高，非常适合在边缘设备上部署。

### 翻译

手语识别促进聋人与听力人士之间的交流。深度学习被广泛用于开发基于手语识别的系统；然而，它计算密集且需要大量计算资源，使其不适合资源受限设备。为解决这一问题，我们提出了一种使用MediaPipe和基于回声状态网络的双向储备计算架构的高效手语识别系统。MediaPipe提取手部关节坐标，作为基于ESN的双向储备计算架构的输入。双向储备计算向前和向后处理这些特征，有效捕获时间依赖性。将双向储备计算的结果状态连接起来，形成用于分类的鲁棒表示。我们在词级美国手语视频数据集上评估了我们的方法，实现了57.71%的竞争性准确率，训练时间仅为9秒，而基于深度学习的双向门控循环单元方法需要55分38秒。因此，基于双向储备计算的手语识别系统非常适合边缘设备。


### 论文摘要

Sign language recognition (SLR) facilitates communication between deaf and hearing individuals. Deep learning is widely used to develop SLR-based systems; however, it is computationally intensive and requires substantial computational resources, making it unsuitable for resource-constrained devices. To address this, we propose an efficient sign language recognition system using MediaPipe and an echo state network (ESN)-based bidirectional reservoir computing (BRC) architecture. MediaPipe extracts hand joint coordinates, which serve as inputs to the ESN-based BRC architecture. The BRC processes these features in both forward and backward directions, efficiently capturing temporal dependencies. The resulting states of BRC are concatenated to form a robust representation for classification. We evaluated our method on the Word-Level American Sign Language (WLASL) video dataset, achieving a competitive accuracy of 57.71% and a significantly lower training time of only 9 seconds, in contrast to the 55 minutes and $38$ seconds required by the deep learning-based Bi-GRU approach. Consequently, the BRC-based SLR system is well-suited for edge devices.

---

## 138. Cross-Domain Federated Semantic Communication with Global Representation Alignment and Domain-Aware Aggregation

**论文链接:** [http://arxiv.org/abs/2512.00711v1](http://arxiv.org/abs/2512.00711v1)

**作者:** Loc X. Nguyen, Ji Su Yoon, Huy Q. Le, Yu Qiao, Avi Deb Raha, Eui-Nam Huh, Walid Saad, Dusit Niyato, Zhu Han, Choong Seon Hong

**发布时间:** 2025-11-30

**备注:** 13 pages, 7 figures, 6 tables

### GPT解析

### 总结

这篇论文提出了一种新的联邦学习框架，用于解决语义通信系统中的领域偏移问题，通过构建全局表示和领域感知的聚合方法，提高了图像重建任务的性能。

### 背景

语义通信可以通过利用原始数据背后的意义来显著提高无线系统的带宽利用率，但其进步依赖于深度学习模型进行联合信源信道编码技术，而这些模型需要大量数据进行训练。联邦学习已被提出以分布式方式训练模型，但传统联邦学习方法在客户端数据来自不同领域时会出现灾难性退化。

### 目的

解决深度学习模型训练数据密集型问题，处理联邦学习中客户端数据来自不同领域时的灾难性退化问题，以及在语义通信系统中首次考虑领域偏移问题用于图像重建任务。

### 方法

提出了一种新的联邦学习框架，通过构建与客户端局部特征保持一致的全局表示来解决领域偏移问题，并识别解决了样本数量多的客户端领域的主导问题，采用领域感知的聚合方法。

### 主要发现

在三个领域、信噪比为1 dB的情况下，所提出的方法比模型对比联邦学习框架的PSNR值高0.5，且随着信道质量的改善，这种性能差距继续扩大。

### 结论

该工作是首次在语义通信系统的图像重建任务训练中考虑领域偏移问题，所提出的方法有效解决了领域偏移和样本数量不均衡的问题。

### 翻译

语义通信可以通过利用原始数据背后的意义来显著提高无线系统的带宽利用率。然而，语义通信取得的进步在很大程度上依赖于用于联合信源信道编码编码器/解码器技术的深度学习模型的发展，这些模型需要大量数据进行训练。为了解决深度学习模型的这种数据密集型特性，联邦学习被提出以分布式方式训练模型，服务器将深度学习模型广播给网络中的客户端，让他们用本地数据进行训练。然而，当客户端数据来自不同领域时，传统的联邦学习方法会出现灾难性退化。相比之下，本文提出了一种新的联邦学习框架，通过构建与客户端局部特征保持一致的全局表示来解决这种领域偏移问题，从而保留不同数据领域的语义。此外，本文还识别并解决了样本数量多的客户端领域的主导问题，采用了一种领域感知的聚合方法。这项工作是首次在语义通信系统的图像重建任务训练中考虑领域偏移问题。最后，仿真结果表明，在信噪比为1 dB的三个领域条件下，所提出的方法比模型对比联邦学习框架的PSNR值高0.5，随着信道质量的改善，这一差距继续扩大。


### 论文摘要

Semantic communication can significantly improve bandwidth utilization in wireless systems by exploiting the meaning behind raw data. However, the advancements achieved through semantic communication are closely dependent on the development of deep learning (DL) models for joint source-channel coding (JSCC) encoder/decoder techniques, which require a large amount of data for training. To address this data-intensive nature of DL models, federated learning (FL) has been proposed to train a model in a distributed manner, where the server broadcasts the DL model to clients in the network for training with their local data. However, the conventional FL approaches suffer from catastrophic degradation when client data are from different domains. In contrast, in this paper, a novel FL framework is proposed to address this domain shift by constructing the global representation, which aligns with the local features of the clients to preserve the semantics of different data domains. In addition, the dominance problem of client domains with a large number of samples is identified and, then, addressed with a domain-aware aggregation approach. This work is the first to consider the domain shift in training the semantic communication system for the image reconstruction task. Finally, simulation results demonstrate that the proposed approach outperforms the model-contrastive FL (MOON) framework by 0.5 for PSNR values under three domains at an SNR of 1 dB, and this gap continues to widen as the channel quality improves.

---

## 139. Controlling weak-lensing shear biases from undetected galaxies in the era of Stage IV Surveys

**论文链接:** [http://arxiv.org/abs/2512.00666v1](http://arxiv.org/abs/2512.00666v1)

**作者:** Lisa M Voigt

**发布时间:** 2025-11-29

**备注:** 22 pages, 18 figures

### GPT解析

### 总结

该研究探讨了引力透镜效应中微弱星系对剪切测量的影响，发现低于检测阈值的星系污染是第四阶段巡天时代的重要偏差来源。

### 背景

引力透镜效应是研究宇宙学模型的有力工具，但在第四阶段巡天时代，低于检测阈值的星系污染已成为显著的偏差来源。

### 目的

采用无噪声偏差的机器学习方法估计剪切，量化微弱星系对类似欧几里得巡天的影响，并确定校准模拟所需参数。

### 方法

通过基线模拟研究微弱星系混合体对剪切偏差的影响，并分析不同微弱星系属性对偏差的贡献。

### 主要发现

微弱星系混合体导致乘法剪切偏差为-0.008，超过欧几里得要求；校准模拟需包括视星等暗至27.0、距离明亮样本星系约1.0角秒的邻近星系；偏差受微弱星系密度分布、与明亮星系的排列方式及位置各向异性影响，而对剪切相干性和平行方向排列不敏感。

### 结论

研究结果指导校准模拟设计，强调深观测在测量微弱星系属性中的关键作用。

### 翻译

背景星系被中间物质引力透镜是研究宇宙学模型的有力探针。在第四阶段巡天时代，来自低于检测阈值的星系的污染已成为一个显著的偏差来源。我们采用一种无噪声偏差的机器学习方法来估计剪切，量化了微弱星系对类似欧几里得巡天的影响。在我们的基线模拟中，微弱星系混合体导致乘法剪切偏差为-0.008，远高于欧几里得的要求。与之前的研究类似，我们发现校准模拟必须包括邻近星系，其视星等暗至27.0 (+2.1, -0.9)且距离每个明亮样本星系(BSG；测量剪切的星系)约1.0 (+0.2, -0.2)角秒。通过改变微弱星系属性，我们确定了哪些属性显著影响剪切偏差，并量化了需要约束的程度。关键的是，我们发现偏差不仅取决于微弱星系的平均投影密度和视星等分布，还取决于这些量如何随BSG的观测亮度变化。此外，偏差对微弱星系相对于BSG的径向和切向排列以及位置各向异性敏感。相比之下，在探索的参数范围内，BSG和微弱星系之间的剪切相干性、平行方向排列以及微弱星系大小-星等关系的变化影响可以忽略。我们的结果指导校准模拟，并强调了深观测在测量微弱星系属性中的关键作用。


### 论文摘要

Gravitational lensing of background galaxies by intervening matter is a powerful probe of the cosmological model. In the era of Stage IV surveys, contamination from galaxies below the detection threshold has emerged as a significant source of bias. Adopting a noise-bias-free machine-learning method to estimate shear, we quantify the impact of faint galaxies for a Euclid-like survey. In our baseline simulations, faint blends induce a multiplicative shear bias of -0.008, well above Euclid's requirement. Similar to previous studies, we find that calibration simulations must include neighbouring galaxies to AB apparent magnitudes as faint as 27.0 (+2.1, -0.9) and within approximately 1.0 (+0.2, -0.2) arcsec of each bright sample galaxy (BSG; the galaxy for which shear is measured). By varying faint galaxy properties, we identify which ones significantly affect shear biases and quantify how well they must be constrained. Crucially, we find that biases not only depend on the mean projected faint-galaxy density and apparent-magnitude distribution across the sample, but also on how these quantities vary with the observed brightness of the BSG. Furthermore, biases are sensitive to radial and tangential alignments and positional anisotropy of faint galaxies relative to BSGs. By contrast, shear coherence between BSGs and faint galaxies, parallel orientation alignments, and variations in the faint galaxy size-magnitude relation have negligible impact within the parameter ranges explored. Our results guide calibration simulations and highlight the critical role of deep observations in measuring the properties of faint galaxies.

---

## 140. Self-sufficient Independent Component Analysis via KL Minimizing Flows

**论文链接:** [http://arxiv.org/abs/2512.00665v1](http://arxiv.org/abs/2512.00665v1)

**作者:** Song Liu

**发布时间:** 2025-11-29

### GPT解析

### 总结

该研究提出了一种基于非线性ICA的解耦信号学习方法，通过自足信号的概念实现信号解耦，无需先验假设和观测模型，避免了不稳定的对抗训练问题。

### 背景

自监督学习领域的进展为信号解耦提供了新的思路，传统ICA方法通常需要先验假设和观测模型，限制了模型的灵活性。

### 目的

开发一种无需先验假设和观测模型的解耦信号学习方法，学习自足信号，即能够仅依靠其他成分重建自身缺失值的信号。

### 方法

将问题表述为条件KL散度的最小化，提出一种顺序算法，在每次迭代中减少KL散度并学习最优的去混流模型。

### 主要发现

所提出的方法完全避免了不稳定的对抗训练问题，在玩具和真实世界数据集上表现出有效性。

### 结论

通过最小化条件KL散度并采用顺序算法，成功实现了无需先验假设的解耦信号学习，证明了该方法的有效性和实用性。

### 翻译

我们研究使用非线性独立成分分析(ICA)从数据中学习解耦信号的问题。受自监督学习进展的启发，我们提出学习自足信号：恢复的信号应该能够仅依靠所有其他成分来重建自身的缺失值，而不依赖任何其他信号。我们将此问题表述为条件KL散度的最小化。与传统最大似然估计相比，我们的算法是无先验和无似然的，意味着我们不需要对原始信号或观测模型施加任何先验条件，这些条件通常会限制模型的灵活性。为解决KL散度最小化问题，我们提出一种顺序算法，该算法在每次迭代中减少KL散度并学习最优的去混流模型。这种方法完全避免了最小化KL散度时的常见问题——不稳定的对抗训练。在玩具和真实世界数据集上的实验证明了我们方法的有效性。


### 论文摘要

We study the problem of learning disentangled signals from data using non-linear Independent Component Analysis (ICA). Motivated by advances in self-supervised learning, we propose to learn self-sufficient signals: A recovered signal should be able to reconstruct a missing value of its own from all remaining components without relying on any other signals. We formulate this problem as the minimization of a conditional KL divergence. Compared to traditional maximum likelihood estimation, our algorithm is prior-free and likelihood-free, meaning that we do not need to impose any prior on the original signals or any observational model, which often restricts the model's flexibility. To tackle the KL divergence minimization problem, we propose a sequential algorithm that reduces the KL divergence and learns an optimal de-mixing flow model at each iteration. This approach completely avoids the unstable adversarial training, a common issue in minimizing the KL divergence. Experiments on toy and real-world datasets show the effectiveness of our method.

---

## 141. Melody or Machine: Detecting Synthetic Music with Dual-Stream Contrastive Learning

**论文链接:** [http://arxiv.org/abs/2512.00621v1](http://arxiv.org/abs/2512.00621v1)

**作者:** Arnesh Batra, Dev Sharma, Krish Thukral, Ruhani Bhatia, Naman Batra, Aditya Gautam

**发布时间:** 2025-11-29

**备注:** Accepted at Transactions on Machine Learning Research (TMLR)

### GPT解析

### 总结

研究者提出了新的基准和检测架构来解决AI生成音乐检测的挑战，CLAM模型在多样化生成器和分布外内容上表现优异

### 背景

端到端AI音乐生成技术快速发展对艺术真实性和版权构成威胁，现有检测方法如SpecTTTra在面对多样化新型生成器时性能下降，在分布外内容上表现不佳

### 目的

开发更强大、更具泛化能力的AI生成音乐检测方法，解决现有检测方法的局限性

### 方法

引入MoM基准（13万首歌曲，6,665小时）和CLAM双流检测架构；CLAM使用MERT和Wave2Vec2两种音频编码器创建并行表示，通过可学习跨聚合模块融合，采用双损失目标训练（二元交叉熵损失和对比三元组损失）

### 主要发现

CLAM在合成音乐取证领域建立新最先进水平，在MoM基准上实现0.925的F1分数

### 结论

新的基准和检测架构能有效应对AI生成音乐的检测挑战，CLAM模型在检测AI生成音乐方面表现出色

### 翻译

端到端AI音乐生成的快速发展对艺术真实性和版权构成日益增长的威胁，需要能够跟上这种步伐的检测方法。虽然具有基础性，但现有模型如SpecTTTra在面对多样化且快速发展的新型生成器生态系统时表现不佳，在分布外内容上表现出显著的性能下降。这种泛化失败突显了一个关键差距：需要更具挑战性的基准和更强大的检测架构。为了解决这个问题，我们首先引入Melody or Machine (MoM)，一个包含超过13万首歌曲（6,665小时）的新的大规模基准。MoM是迄今为止最多样化的数据集，由开源和闭源模型混合构建，并包含一个专门设计的OOD测试集，旨在促进真正可泛化检测器的发展。除了这个基准，我们还引入了CLAM，一种新颖的双流检测架构。我们假设声乐和乐器元素之间的微妙机器诱导不一致性是合成的有力标志。CLAM通过使用两个不同的预训练音频编码器创建音频的并行表示来测试这一假设。这些表示通过可学习的跨聚合模块融合，该模块建模它们之间的相互依赖关系。该模型使用双损失目标进行训练：标准的二元交叉熵损失用于分类，辅以对比三元组损失，增强其对合成伪影的敏感性。CLAM在合成音乐取证领域建立了新的最先进水平，在我们的具有挑战性的MoM基准上，它实现了0.925的F1分数。


### 论文摘要

The rapid evolution of end-to-end AI music generation poses an escalating threat to artistic authenticity and copyright, demanding detection methods that can keep pace. While foundational, existing models like SpecTTTra falter when faced with the diverse and rapidly advancing ecosystem of new generators, exhibiting significant performance drops on out-of-distribution (OOD) content. This generalization failure highlights a critical gap: the need for more challenging benchmarks and more robust detection architectures. To address this, we first introduce Melody or Machine (MoM), a new large-scale benchmark of over 130,000 songs (6,665 hours). MoM is the most diverse dataset to date, built with a mix of open and closed-source models and a curated OOD test set designed specifically to foster the development of truly generalizable detectors. Alongside this benchmark, we introduce CLAM, a novel dual-stream detection architecture. We hypothesize that subtle, machine-induced inconsistencies between vocal and instrumental elements, often imperceptible in a mixed signal, offer a powerful tell-tale sign of synthesis. CLAM is designed to test this hypothesis by employing two distinct pre-trained audio encoders (MERT and Wave2Vec2) to create parallel representations of the audio. These representations are fused by a learnable cross-aggregation module that models their inter-dependencies. The model is trained with a dual-loss objective: a standard binary cross-entropy loss for classification, complemented by a contrastive triplet loss which trains the model to distinguish between coherent and artificially mismatched stream pairings, enhancing its sensitivity to synthetic artifacts without presuming a simple feature alignment. CLAM establishes a new state-of-the-art in synthetic music forensics. It achieves an F1 score of 0.925 on our challenging MoM benchmark.

---

## 142. DLRREC: Denoising Latent Representations via Multi-Modal Knowledge Fusion in Deep Recommender Systems

**论文链接:** [http://arxiv.org/abs/2512.00596v1](http://arxiv.org/abs/2512.00596v1)

**作者:** Jiahao Tian, Zhenkai Wang

**发布时间:** 2025-11-29

### GPT解析

### 总结

本文提出了一种新的框架，通过深度融合多模态和协作知识进行表示去噪，解决了现代推荐系统难以有效利用大型语言模型生成的高维度嘈杂多模态特征的问题。

### 背景

现代推荐系统难以有效利用大型语言模型生成的丰富但高维度且嘈杂的多模态特征。将这些特征作为静态输入会使它们与核心推荐任务脱节。

### 目的

解决上述局限性，提出一个新颖的框架，通过深度融合多模态和协作知识进行表示去噪。

### 方法

统一的架构引入了两个主要技术创新：首先，将降维直接集成到推荐模型中，实现端到端共同训练，使降维过程能够感知最终的排序目标；其次，引入对比学习目标，将协作过滤信号显式地融入潜在空间。

### 主要发现

这种协同过程可以优化原始的大型语言模型嵌入，过滤噪声同时增强任务相关信号。大量实验证实了该方法具有更强的判别能力。

### 结论

这种集成融合和去噪策略对于实现最先进性能至关重要，为在推荐系统中有效利用大型语言模型提供了基础范式。

### 翻译

现代推荐系统难以有效利用大型语言模型生成的丰富但高维度且嘈杂的多模态特征。将这些特征作为静态输入会使它们与核心推荐任务脱节。我们通过一个基于关键洞察的新颖框架解决这一局限性：深度融合多模态和协作知识进行表示去噪。我们的统一架构引入了两个主要技术创新。首先，我们将降维直接集成到推荐模型中，实现端到端共同训练，使降维过程能够感知最终的排序目标。其次，我们引入了对比学习目标，将协作过滤信号显式地融入潜在空间。这种协同过程优化了原始的大型语言模型嵌入，过滤噪声同时增强任务相关信号。大量实验证实了我们方法的优越判别能力，证明这种集成融合和去噪策略对于实现最先进性能至关重要。我们的工作为在推荐系统中有效利用大型语言模型提供了基础范式。


### 论文摘要

Modern recommender systems struggle to effectively utilize the rich, yet high-dimensional and noisy, multi-modal features generated by Large Language Models (LLMs). Treating these features as static inputs decouples them from the core recommendation task. We address this limitation with a novel framework built on a key insight: deeply fusing multi-modal and collaborative knowledge for representation denoising. Our unified architecture introduces two primary technical innovations. First, we integrate dimensionality reduction directly into the recommendation model, enabling end-to-end co-training that makes the reduction process aware of the final ranking objective. Second, we introduce a contrastive learning objective that explicitly incorporates the collaborative filtering signal into the latent space. This synergistic process refines raw LLM embeddings, filtering noise while amplifying task-relevant signals. Extensive experiments confirm our method's superior discriminative power, proving that this integrated fusion and denoising strategy is critical for achieving state-of-the-art performance. Our work provides a foundational paradigm for effectively harnessing LLMs in recommender systems.

---

## 143. ECO: Energy-Constrained Operator Learning for Chaotic Dynamics with Boundedness Guarantees

**论文链接:** [http://arxiv.org/abs/2512.01984v1](http://arxiv.org/abs/2512.01984v1)

**作者:** Andrea Goertzen, Sunbochen Tang, Navid Azizan

**发布时间:** 2025-12-01

### GPT解析

### 总结

本文提出了一种名为能量约束算子(ECO)的新方法，用于学习混沌系统的动力学特性，同时确保预测的有界性，这是首个为数据驱动的混沌动力学模型提供形式化保证的工作。

### 背景

混沌是许多复杂动态系统的基本特征，包括天气系统和流体湍流。由于对初始条件的极端敏感性，这些系统本质上难以预测。许多混沌系统是耗散和遍历的，促使开发旨在学习长时间范围内不变统计特性的数据驱动模型。

### 目的

克服现有模型生成无界预测的局限性，这些无界预测阻碍了有意义的统计评估，同时学习系统动力学并强制预测有界性。

### 方法

引入能量约束算子(ECO)，利用控制理论的概念，基于可学习的能量函数开发代数条件确保学习到的动力学是耗散的。ECO通过高效的闭式二次投影层强制执行这些代数条件，提供可证明的轨迹有界性。

### 主要发现

ECO能够生成稳定的长期预测，在受混沌偏微分方程（包括Kuramoto-Sivashinsky和Navier-Stokes方程）支配的系统中捕获不变统计特性，同时学习到的不变水平集为奇怪吸引子提供了外估计。

### 结论

ECO是首个为数据驱动的混沌动力学模型建立此类形式化保证的方法，能够同时学习系统动力学并强制预测有界性，在混沌系统的长期预测中表现出色。

### 翻译

混沌是许多复杂动态系统的基本特征，包括天气系统和流体湍流。由于这些系统对初始条件具有极端敏感性，因此本质上难以预测。许多混沌系统是耗散和遍历的，这促使了旨在学习长时间范围内不变统计特性的数据驱动模型。尽管最近的模型在保持不变统计特性方面显示了经验上的成功，但它们容易产生无界预测，这阻碍了有意义的统计评估。为了克服这一点，我们引入了能量约束算子(ECO)，它同时学习系统动力学并强制预测的有界性。我们利用控制理论的概念，基于可学习的能量函数开发代数条件，确保学习到的动力学是耗散的。ECO通过一个高效的闭式二次投影层强制执行这些代数条件，提供可证明的轨迹有界性。据我们所知，这是首个为数据驱动的混沌动力学模型建立此类形式化保证的工作。此外，学习到的不变水平集为奇怪吸引子提供了一个外估计，奇怪吸引子是一个计算上难以表征的复杂结构。我们证明了ECO在生成稳定的长期预测方面的经验成功，能够捕获受混沌偏微分方程（包括Kuramoto-Sivashinsky和Navier-Stokes方程）支配的系统中的不变统计特性。


### 论文摘要

Chaos is a fundamental feature of many complex dynamical systems, including weather systems and fluid turbulence. These systems are inherently difficult to predict due to their extreme sensitivity to initial conditions. Many chaotic systems are dissipative and ergodic, motivating data-driven models that aim to learn invariant statistical properties over long time horizons. While recent models have shown empirical success in preserving invariant statistics, they are prone to generating unbounded predictions, which prevent meaningful statistics evaluation. To overcome this, we introduce the Energy-Constrained Operator (ECO) that simultaneously learns the system dynamics while enforcing boundedness in predictions. We leverage concepts from control theory to develop algebraic conditions based on a learnable energy function, ensuring the learned dynamics is dissipative. ECO enforces these algebraic conditions through an efficient closed-form quadratic projection layer, which provides provable trajectory boundedness. To our knowledge, this is the first work establishing such formal guarantees for data-driven chaotic dynamics models. Additionally, the learned invariant level set provides an outer estimate for the strange attractor, a complex structure that is computationally intractable to characterize. We demonstrate empirical success in ECO's ability to generate stable long-horizon forecasts, capturing invariant statistics on systems governed by chaotic PDEs, including the Kuramoto--Sivashinsky and the Navier--Stokes equations.

---

## 144. Spontaneous Symmetry Breaking in Two-dimensional Long-range Heisenberg Model

**论文链接:** [http://arxiv.org/abs/2512.01956v1](http://arxiv.org/abs/2512.01956v1)

**作者:** Dingyun Yao, Tianning Xiao, Zhijie Fan, Youjin Deng

**发布时间:** 2025-12-01

### GPT解析

### 总结

本研究探讨了长程相互作用系统中的相变和低温行为，通过大规模Monte Carlo模拟和长程随机行走方法，提出了适用于任意空间维度的一般判据。

### 背景

衰减长程相互作用系统性质随参数变化的研究一直受到关注，Sak's criterion和扩展的Mermin-Wagner定理被广泛用于预测这类系统的临界和低温行为。

### 目的

理解长程相互作用系统的相变性质和低温行为，并提出适用于任意空间维度的一般判据。

### 方法

进行大规模Monte Carlo模拟研究二维长程海森堡模型，并引入长程简单随机行走方法来表征系统的低温标度行为。

### 主要发现

当参数小于等于2时，系统通过单一连续相变表现出自发对称性破缺并形成普遍长程有序；长程简单随机行走可以准确表征由Goldstone模波动引起的二维和三维长程海森堡模型的低温标度行为。

### 结论

基于长程随机行走的见解，提出了具有连续对称性的任意空间维度中长程统计系统的相变和低温性质的一般判据。

### 翻译

衰减长程相互作用的引入使得理解系统性质如何随参数变化的研究持续受到关注。Sak's criterion和扩展的Mermin-Wagner定理已被广泛接受用于预测此类系统的临界和低温行为。我们对二维长程海森堡模型进行了大规模Monte Carlo模拟，线性尺寸达到最大值，表明当参数小于等于2时，系统通过单一连续相变表现出自发对称性破缺，并形成普遍长程有序。随后，我们引入了总行走长度固定的长程简单随机行走，满足统计系统的广延性要求，并观察到该方法可以准确表征由Goldstone模波动引起的二维和三维长程海森堡模型的低温标度行为。最后，基于长程随机行走的见解，我们提出了适用于任意空间维度中具有连续对称性的长程统计系统的相变和低温性质的一般判据。


### 论文摘要

The introduction of decaying long-range (LR) interactions $1/r^{d+σ}$ has drawn persistent interest in understanding how system properties evolve with $σ$. The Sak's criterion and the extended Mermin-Wagner theorem have gained broad acceptance in predicting the critical and low-temperature (low-T) behaviors of such systems. We perform large-scale Monte Carlo simulations for the LR-Heisenberg model in two dimensions (2D) up to linear size $L=8192$, and show that, as long as for $σ\leq 2$, the system exhibits spontaneous symmetry breaking, via a single continuous phase transition, and develops a generic long-range order. We then introduce an LR simple random walk (LR-SRW) with the total walk length fixed at O($L^d$), satisfying the extensivity of statistical systems, and observe that the LR-SRW can faithfully characterize the low-T scaling behaviors of the LR-Heisenberg model in both 2D and 3D, as induced by Goldstone-mode fluctuations. Finally, based on insights from LR-SRW, we propose a general criterion for the phase transition and the low-T properties of LR statistical systems with continuous symmetry in any spatial dimension.

---

## 145. Predicting Human Chess Moves: An AI Assisted Analysis of Chess Games Using Skill-group Specific n-gram Language Models

**论文链接:** [http://arxiv.org/abs/2512.01880v1](http://arxiv.org/abs/2512.01880v1)

**作者:** Daren Zhong, Dingcheng Huang, Clayton Greenberg

**发布时间:** 2025-12-01

### GPT解析

### 总结

本研究提出了一种新颖且计算高效的走法预测框架，将国际象棋走法预测作为行为分析任务，通过n-gram语言模型捕捉不同技能水平棋手的走法特征，并实现动态模型选择以提高预测准确率。

### 背景

国际象棋作为一种具有完全信息的确定性游戏，长期以来一直是研究战略决策和人工智能的基准。然而，传统的国际象棋引擎或分析工具主要专注于计算最优走法，往往忽视了人类棋手之间存在的变异性，特别是不同技能水平棋手之间的差异。

### 目的

克服传统国际象棋分析工具的局限性，开发一种能够捕捉不同技能水平棋手走法特征的计算高效走法预测框架，并实现动态模型选择以提高预测准确率。

### 方法

研究采用n-gram语言模型来捕捉特定棋手技能水平的走法特征。通过将棋手分为从新手到专家的七个不同技能组，使用来自开源国际象棋平台Lichess的数据训练单独的模型。框架能够动态选择最适合的模型进行预测任务，并根据先前序列生成棋手走法。

### 主要发现

在真实游戏数据上的评估表明，当使用早期游戏信息（16半步）时，框架内的模型选择器模块可以将技能水平分类的准确率高达31.7%。走法预测框架也显示出显著的准确率提升，选择器辅助准确率比基准准确率高出39.1%。

### 结论

该框架的计算效率进一步增强了其进行实时国际象棋分析的适用性，为理解和预测人类棋手行为提供了有效工具，同时考虑了不同技能水平的差异性。

### 翻译

国际象棋，一种具有完全信息的确定性游戏，长期以来一直是研究战略决策和人工智能的基准。传统的国际象棋引擎或分析工具主要专注于计算最优走法，往往忽视了人类棋手之间存在的变异性，特别是不同技能水平棋手之间的差异。为了克服这一限制，我们提出了一种新颖且计算高效的走法预测框架，将国际象棋走法预测作为行为分析任务。该框架采用n-gram语言模型来捕捉特定棋手技能水平的走法特征。通过将棋手分为从新手到专家的七个不同技能组，我们使用来自开源国际象棋平台Lichess的数据训练了单独的模型。该框架动态选择最适合的模型进行预测任务，并根据先前序列生成棋手走法。在真实游戏数据上的评估表明，当使用早期游戏信息（16半步）时，框架内的模型选择器模块可以将技能水平分类的准确率高达31.7%。走法预测框架也显示出显著的准确率提升，我们的选择器辅助准确率比基准准确率高出39.1%。该框架的计算效率进一步增强了其进行实时国际象棋分析的适用性。


### 论文摘要

Chess, a deterministic game with perfect information, has long served as a benchmark for studying strategic decision-making and artificial intelligence. Traditional chess engines or tools for analysis primarily focus on calculating optimal moves, often neglecting the variability inherent in human chess playing, particularly across different skill levels.   To overcome this limitation, we propose a novel and computationally efficient move prediction framework that approaches chess move prediction as a behavioral analysis task. The framework employs n-gram language models to capture move patterns characteristic of specific player skill levels. By dividing players into seven distinct skill groups, from novice to expert, we trained separate models using data from the open-source chess platform Lichess. The framework dynamically selects the most suitable model for prediction tasks and generates player moves based on preceding sequences.   Evaluation on real-world game data demonstrates that the model selector module within the framework can classify skill levels with an accuracy of up to 31.7\% when utilizing early game information (16 half-moves). The move prediction framework also shows substantial accuracy improvements, with our Selector Assisted Accuracy being up to 39.1\% more accurate than our benchmark accuracy. The computational efficiency of the framework further enhances its suitability for real-time chess analysis.

---

## 146. Anomalous Eigenstates of a Doped Hole in the Ising Antiferromagnet

**论文链接:** [http://arxiv.org/abs/2512.01815v1](http://arxiv.org/abs/2512.01815v1)

**作者:** Piotr Wrzosek, Krzysztof Wohlfeld, Eugene A. Demler, Annabelle Bohrdt, Fabian Grusdt

**发布时间:** 2025-12-01

**备注:** 15 pages, 3 appendices

### GPT解析

### 总结

研究掺入反铁磁莫特绝缘体的移动空穴问题，发现了单个空穴局域谱中存在一系列异常的长寿命量子多体瘢痕态，这些态具有与J成线性比例的激发能，导致异常缓慢的热化行为。

### 背景

掺入一个移动空穴的反铁磁莫特绝缘体问题被认为是几个典型强关联电子系统丰富物理的基础，包括重费米子系统和高温超导性。最简单的对应于掺入伊辛反铁磁体问题，该问题在过去60年中被认为已基本解决。

### 目的

重新研究掺入伊辛反铁磁体问题，探索其中可能存在的新现象，特别是单个空穴在经典伊辛-奈尔态中的局域谱特性。

### 方法

通过精确对角化和自避路径近似方法进行研究，并计算不同的局域旋转谱以确定异常态的起源。

### 主要发现

在经典伊辛-奈尔态中单个空穴的局域谱包含一系列异常的长寿命态，这些态的激发能约与J成线性比例，与已知的梯状谱产生避免交叉。这些异常态源于问题中近似出现的局域C3对称性，导致异常缓慢的热化行为。

### 结论

发现的异常态代表了一种新型的量子多体瘢痕态，可能与晶格规范理论中预测的多体瘢痕相关。

### 翻译

掺入反铁磁莫特绝缘体的移动空穴问题被认为是几个典型强关联电子系统丰富物理的基础，范围从重费米子到高温超导性。可以说，这个问题的最简单形式对应于掺入伊辛反铁磁体，这个问题在过去60年中通过一个流行但近似的映射到贝特格点上的单粒子问题而被广泛认为基本解决。在这里我们表明，尽管其表面简单，但在经典伊辛-奈尔态中单个空穴的局域谱包含一系列异常的长寿命态，这些态超出了众所周知的梯状谱（激发能级间隔为J的2/3次方乘以t的1/3次方）。我们通过精确对角化和自避路径近似发现的异常态具有与J成近似线性比例的激发能，并与更明显的梯状谱产生一系列避免交叉。通过计算不同的局域旋转谱，我们解释了这些异常态的根源在于问题中近似出现的局域C3对称性。从它们直接的光谱特征我们进一步得出结论，这些态导致异常缓慢的热化行为——因此代表了一种新型的量子多体瘢痕态，可能与晶格规范理论中预测的多体瘢痕相关。


### 论文摘要

The problem of a mobile hole doped into an antiferromagnet Mott insulator is believed to underly the rich physics of several paradigmatic strongly correlated electron systems, ranging from heavy fermions to high-Tc superconductivity. Arguably the simplest incarnation of this problem corresponds to a doped Ising antiferromagnet, a problem widely considered essentially solved since almost 60 years by a popular yet approximate mapping to a single-particle problem on the Bethe lattice. Here we show that, despite its deceptive simplicity, the local spectrum of a single hole in a classical Ising-Néel state contains a series of anomalous, long-lived states that go beyond the well-known ladder-like spectrum with excited energies spaced as $J^{2/3} t^{1/3}$. The anomalous states we find through exact diagonalization and within the self-avoiding path approximation have excitation energies scaling approximately linear with $J$ and lead to a series of avoided crossings with the more pronounced ladder spectrum. By also computing different local, rotational spectra we explain the origin of the anomalous states as rooted in an approximate emergent local $C_3$ symmetry of the problem. From their direct spectral signatures we further conclude that these states lead to anomalously slow thermalization behavior -- hence representing a new type of quantum many-body scar state, potentially related to many-body scars predicted in lattice gauge theories.

---

## 147. H-Neurons: On the Existence, Impact, and Origin of Hallucination-Associated Neurons

**论文链接:** [http://arxiv.org/abs/2512.01797v1](http://arxiv.org/abs/2512.01797v1)

**作者:** Cheng Gao, Huimin Chen, Chaojun Xiao, Zhiyi Chen, Zhiyuan Liu, Maosong Sun

**发布时间:** 2025-12-01

**备注:** 20 pages, 4 figures

### GPT解析

### 总结

本研究从神经元角度探索大型语言模型中的幻觉机制，发现与幻觉相关的神经元(H-Neurons)非常稀疏但可预测幻觉发生，这些神经元在预训练阶段就已出现并与过度服从行为有因果联系。

### 背景

大型语言模型经常产生幻觉（看似合理但内容不正确的输出），影响其可靠性。先前研究主要从宏观角度（如训练数据和目标）探讨幻觉，而神经元层面的机制尚未被充分探索。

### 目的

从三个角度对LLMs中的与幻觉相关的神经元(H-Neurons)进行系统性研究：识别、行为影响和起源。

### 方法

1) 识别：分析可预测幻觉发生的神经元子集；2) 行为影响：通过受控干预研究这些神经元与过度服从行为的因果关系；3) 起源：追溯这些神经元到预训练基础模型，确定它们的出现时间。

### 主要发现

1) 与幻觉相关的神经元非常稀疏（不到总神经元的0.1%）；2) 这些神经元可以可靠预测幻觉发生，并在各种场景中具有强大泛化能力；3) 这些神经元与过度服从行为有因果联系；4) 这些神经元在预训练阶段就已出现，对幻觉检测仍然具有预测性。

### 结论

研究结果将宏观行为模式与微观神经机制联系起来，为开发更可靠的LLMs提供了见解。

### 翻译

大型语言模型(LLMs)经常产生幻觉——看似合理但内容不正确的输出——这损害了它们的可靠性。虽然先前的工作已经从宏观角度（如训练数据和目标）研究了幻觉，但潜在的神经元级机制仍然 largely未被探索。在本文中，我们从三个角度对LLMs中的与幻觉相关的神经元(H-Neurons)进行了系统性研究：识别、行为影响和起源。关于它们的识别，我们展示了一个非常稀疏的神经元子集（不到总神经元的0.1%）可以可靠地预测幻觉的发生，并在各种场景中具有强大的泛化能力。在行为影响方面，受控干预显示这些神经元与过度服从行为有因果关系。关于它们的起源，我们将这些神经元追溯到预训练的基础模型，发现这些神经元对幻觉检测仍然具有预测性，表明它们是在预训练过程中出现的。我们的研究结果将宏观行为模式与微观神经机制联系起来，为开发更可靠的LLMs提供了见解。


### 论文摘要

Large language models (LLMs) frequently generate hallucinations -- plausible but factually incorrect outputs -- undermining their reliability. While prior work has examined hallucinations from macroscopic perspectives such as training data and objectives, the underlying neuron-level mechanisms remain largely unexplored. In this paper, we conduct a systematic investigation into hallucination-associated neurons (H-Neurons) in LLMs from three perspectives: identification, behavioral impact, and origins. Regarding their identification, we demonstrate that a remarkably sparse subset of neurons (less than $0.1\%$ of total neurons) can reliably predict hallucination occurrences, with strong generalization across diverse scenarios. In terms of behavioral impact, controlled interventions reveal that these neurons are causally linked to over-compliance behaviors. Concerning their origins, we trace these neurons back to the pre-trained base models and find that these neurons remain predictive for hallucination detection, indicating they emerge during pre-training. Our findings bridge macroscopic behavioral patterns with microscopic neural mechanisms, offering insights for developing more reliable LLMs.

---

## 148. Dynamic Log-Gaussian Process Control Barrier Function for Safe Robotic Navigation in Dynamic Environments

**论文链接:** [http://arxiv.org/abs/2512.01668v1](http://arxiv.org/abs/2512.01668v1)

**作者:** Xin Yin, Chenyang Liang, Yanning Guo, Jie Mei

**发布时间:** 2025-12-01

**备注:** To be presented in the 64th IEEE Conference on Decision and Control (CDC 2025)

### GPT解析

### 总结

本文提出了一种动态对数高斯过程控制屏障函数(DLGP-CBF)，用于解决机器人应用中的安全导航问题，特别是在未知和动态场景中。

### 背景

控制屏障函数(CBFs)已成为解决机器人安全导航问题的有效工具，但使用实时传感器数据在线合成信息丰富且对障碍物运动感知的CBFs在动态场景中仍然具有挑战性。

### 目的

提出一种新型的高斯过程基CBF公式，实现实时构建既具有空间信息性又能响应障碍物运动的控制屏障函数。

### 方法

首先，利用高斯过程回归的对数变换生成平滑且信息丰富的屏障值和梯度；其次，将DLGP-CBF明确建模为障碍物位置的函数，使导出的安全约束能够集成预测的障碍物速度，实现控制器对动态障碍物的主动响应。

### 主要发现

模拟结果表明，与基线方法相比，DLGP-CBF在障碍物避障性能方面有显著改进，包括增加安全裕度、产生更平滑的轨迹和增强响应能力。

### 结论

DLGP-CBF方法能够有效解决动态环境中的机器人安全导航问题，提高了避障性能和系统安全性。

### 翻译

控制屏障函数(CBFs)已成为解决机器人应用中安全导航问题的有效工具。然而，使用实时传感器数据在线合成信息丰富且对障碍物运动感知的CBFs仍然具有挑战性，特别是在未知和动态场景中。受此挑战启发，本文旨在提出一种基于高斯过程的新型CBF公式，称为动态对数高斯过程控制屏障函数(DLGP-CBF)，以实现实时构建既具有空间信息性又能响应障碍物运动的CBF。首先，DLGP-CBF利用高斯过程回归的对数变换来生成平滑且信息丰富的屏障值和梯度，即使在稀疏数据区域也是如此。其次，通过将DLGP-CBF明确建模为障碍物位置的函数，导出的安全约束集成了预测的障碍物速度，使控制器能够主动响应动态障碍物的运动。模拟结果表明，与基线方法相比，在障碍物避障性能方面有显著改进，包括增加安全裕度、更平滑的轨迹和增强的响应能力。


### 论文摘要

Control Barrier Functions (CBFs) have emerged as efficient tools to address the safe navigation problem for robot applications. However, synthesizing informative and obstacle motion-aware CBFs online using real-time sensor data remains challenging, particularly in unknown and dynamic scenarios. Motived by this challenge, this paper aims to propose a novel Gaussian Process-based formulation of CBF, termed the Dynamic Log Gaussian Process Control Barrier Function (DLGP-CBF), to enable real-time construction of CBF which are both spatially informative and responsive to obstacle motion. Firstly, the DLGP-CBF leverages a logarithmic transformation of GP regression to generate smooth and informative barrier values and gradients, even in sparse-data regions. Secondly, by explicitly modeling the DLGP-CBF as a function of obstacle positions, the derived safety constraint integrates predicted obstacle velocities, allowing the controller to proactively respond to dynamic obstacles' motion. Simulation results demonstrate significant improvements in obstacle avoidance performance, including increased safety margins, smoother trajectories, and enhanced responsiveness compared to baseline methods.

---

## 149. Do Large Language Models Walk Their Talk? Measuring the Gap Between Implicit Associations, Self-Report, and Behavioral Altruism

**论文链接:** [http://arxiv.org/abs/2512.01568v1](http://arxiv.org/abs/2512.01568v1)

**作者:** Sandro Andric

**发布时间:** 2025-12-01

**备注:** 14 pages, 7 figures, 7 tables. Code and data available at https://github.com/sandroandric/LLMs_Altruism_Study_Code

### GPT解析

### 总结

本研究探讨了大型语言模型是否表现出利他倾向，以及它们的内隐联想和自我报告是否能预测实际利他行为。

### 背景

大型语言模型在人类社会中的行为表现及其与自我认知的关系尚未充分研究。

### 目的

检验大型语言模型的利他行为模式及其与自我认知的一致性。

### 方法

采用受人类社会心理学启发的多方法研究，测试了24个前沿大型语言模型，通过三种范式：(1)内隐联想测试测量内隐利他偏见；(2)强制二元选择任务测量行为利他性；(3)自我评估量表测量明确的利他信念。

### 主要发现

所有模型都表现出强烈的亲利他内隐偏见；模型的利他行为表现高于随机概率但存在显著差异；内隐联想不能预测行为；模型系统地高估了自己的利他性，这种'美德信号差距'影响了75%的测试模型。

### 结论

建议采用校准差距(自我报告和行为价值之间的差异)作为标准化的对齐指标，只有12.5%的模型实现了高亲社会行为和准确自我认知的理想组合。

### 翻译

我们研究大型语言模型是否表现出利他倾向，以及它们的内隐联想和自我报告是否能预测实际的利他行为。采用受人类社会心理学启发的多方法方法，我们测试了24个前沿大型语言模型，通过三种范式：(1)内隐联想测试测量内隐利他偏见，(2)强制二元选择任务测量行为利他性，(3)自我评估量表测量明确的利他信念。我们的主要发现是：(1)所有模型都表现出强烈的亲利他内隐偏见，证实模型'知道'利他主义是好的。(2)模型的利他行为表现高于随机概率，但存在显著差异。(3)内隐联想不能预测行为。(4)最重要的是，模型系统地高估了自己的利他性，声称77.5%的利他性而实际表现为65.6%。这种'美德信号差距'影响了75%的测试模型。基于这些发现，我们建议采用校准差距作为标准化的对齐指标。校准良好的模型更具可预测性和行为一致性；只有12.5%的模型实现了高亲社会行为和准确自我理想的理想组合。


### 论文摘要

We investigate whether Large Language Models (LLMs) exhibit altruistic tendencies, and critically, whether their implicit associations and self-reports predict actual altruistic behavior. Using a multi-method approach inspired by human social psychology, we tested 24 frontier LLMs across three paradigms: (1) an Implicit Association Test (IAT) measuring implicit altruism bias, (2) a forced binary choice task measuring behavioral altruism, and (3) a self-assessment scale measuring explicit altruism beliefs. Our key findings are: (1) All models show strong implicit pro-altruism bias (mean IAT = 0.87, p < .0001), confirming models "know" altruism is good. (2) Models behave more altruistically than chance (65.6% vs. 50%, p < .0001), but with substantial variation (48-85%). (3) Implicit associations do not predict behavior (r = .22, p = .29). (4) Most critically, models systematically overestimate their own altruism, claiming 77.5% altruism while acting at 65.6% (p < .0001, Cohen's d = 1.08). This "virtue signaling gap" affects 75% of models tested. Based on these findings, we recommend the Calibration Gap (the discrepancy between self-reported and behavioral values) as a standardized alignment metric. Well-calibrated models are more predictable and behaviorally consistent; only 12.5% of models achieve the ideal combination of high prosocial behavior and accurate self-knowledge.

---

## 150. X-CME: From In Situ Flux-Rope Reconstruction to CME Propagation Forecasting

**论文链接:** [http://arxiv.org/abs/2512.01561v1](http://arxiv.org/abs/2512.01561v1)

**作者:** Marti Masso Moreno, Carlos Arturo Perez-Alanis, P. K. Manoharan

**发布时间:** 2025-12-01

**备注:** 9 pages, 13 figures, 1 table. Work developed at NASA Goddard Space Flight Center

### GPT解析

### 总结

X-CME框架通过结合中间距离的磁场重建与几何一致的传播模型，显著提高了日冕物质抛射(CME)到达时间和影响几何结构的预测准确性，将误差从传统的约10小时降低到2-4小时。

### 背景

准确预测日冕物质抛射(CME)的到达时间和影响几何结构仍然是空间天气操作的主要挑战。基于日冕观测的技术通常能达到约10小时的平均绝对误差，而在L1点的实地测量虽能提供优秀的磁场信息，但只有几十分钟的预警时间。

### 目的

引入X-CME框架，该框架将中间日心距离的实地磁绳重建与基于物理的CME传播模型联系起来，以提高CME到达时间和影响几何结构的预测准确性。

### 方法

使用椭圆形圆柱、径向极向磁绳模型获取内部磁结构，并将其嵌入到锥形环面CME几何结构中；通过在帕克太阳风背景中求解基于阻力的运动方程计算后续传播，包括重力减速、横截面自相似膨胀以及对黄道面中CME润湿面积和扫掠面积的显式计算；将X-CME应用于由帕克太阳探测器和太阳轨道器航天器观测的两个事件，并将重建的结构传播到地球和火星。

### 主要发现

模型成功重现了L1点观测到的实地特征；预测地球上的CME到达时间误差为2-4小时；正确区分了中心遭遇和擦边撞击。

### 结论

将中间距离的磁场重建与几何一致的传播方案相结合可以显著改善内日球层中的CME到达时间预测和影响评估。

### 翻译

日冕物质抛射(CME)到达时间和影响几何结构的准确预测仍然是空间天气操作的主要挑战。基于日冕观测的技术通常能达到约十小时的平均绝对误差，而在L1点的实地测量能提供优秀的磁场信息，但只有几十分钟的预警时间。在本工作中，我们引入了X-CME框架，该框架将中间日心距离的实地磁绳重建与基于物理的CME传播模型联系起来。内部磁结构通过椭圆形圆柱、径向极向磁绳模型获得，并嵌入到锥形环面CME几何结构中。后续传播通过在帕克太阳风背景中求解基于阻力的运动方程计算，包括重力减速、横截面的自相似膨胀以及对黄道面中CME润湿面积和扫掠面积的显式计算。我们将X-CME应用于帕克太阳探测器和太阳轨道器航天器分别观测的两个事件，并将重建的结构传播到地球和火星。对于两种情况，该模型都重现了L1点观测到的实地特征，并预测地球上的CME到达时间误差为几小时（通常约2-4小时），同时正确区分了中心遭遇和擦边撞击。这些结果表明，将中间距离的磁场重建与几何一致的传播方案相结合可以显著改善内日球层中的CME到达时间预测和影响评估。


### 论文摘要

Accurate forecasts of Coronal Mass Ejection (CME) arrival times and impact geometry remain a major challenge for space-weather operations. Coronagraph-based techniques typically achieve mean absolute errors of order ten hours, while in situ measurements at L1 provide excellent magnetic-field information but only tens of minutes of warning. In this work we introduce X-CME, a framework that links in situ flux-rope reconstructions at intermediate heliocentric distances with a physics-based CME propagation model. The internal magnetic structure is obtained with an elliptical cylindrical, radial poloidal flux-rope model and embedded into a tapered torus CME geometry. The subsequent propagation is computed by solving a drag-based equation of motion in a Parker solar-wind background, including gravitational deceleration, self-similar expansion of the cross section, and an explicit calculation of the CME wetted area and swept area in the ecliptic plane. We apply X-CME to two events observed by the Parker Solar Probe and Solar Orbiter spacecraft, respectively, and propagate the reconstructed structures to Earth and Mars. For both cases, the model reproduces the observed in situ signatures at L1 and predicts the CME arrival time at Earth with errors of a few hours (typically about 2-4 hours), while correctly distinguishing between central encounters and glancing blows. These results demonstrate that combining intermediate-distance magnetic reconstructions with a geometrically consistent propagation scheme can substantially improve CME arrival-time forecasts and impact assessment in the inner heliosphere.

---

## 151. A Unified Bayesian Framework for Stochastic Data-Driven Smoothing, Prediction, and Control

**论文链接:** [http://arxiv.org/abs/2512.01475v1](http://arxiv.org/abs/2512.01475v1)

**作者:** Mingzhou Yin, Andrea Iannelli, Seyed Ali Nazari, Matthias A. Müller

**发布时间:** 2025-12-01

### GPT解析

### 总结

该研究提出了一种统一的贝叶斯框架，用于处理随机数据驱动任务，包括平滑、预测和控制，通过最大后验估计实现。

### 背景

基于Willems基本引理的数据驱动算法扩展到随机数据通常需要经验和定制的解决方案。

### 目的

提供一种系统化和通用的方法来处理随机数据驱动任务，避免需要定制的工作区。

### 方法

通过最大后验估计处理平滑、预测和控制任务；制定统一的轨迹估计问题；解决贝叶斯问题，将轨迹知识与来自离线数据的轨迹数据驱动表征最优结合，以处理一般类的随机干扰。

### 主要发现

在特定条件下，该问题可以推广现有的数据驱动预测和控制算法。

### 结论

数值示例证明了统一方法在所有三项任务中相对于其他数据驱动和系统识别方法的性能。

### 翻译

将基于Willems基本引理的数据驱动算法扩展到随机数据通常需要经验和定制的解决方案。这项工作提出了一个统一的贝叶斯框架，通过最大后验估计为处理随机数据驱动任务（包括平滑、预测和控制）提供了一种系统化和通用的方法。该框架通过指定不同类型的轨迹知识，为三项任务制定了统一的轨迹估计问题。然后，解决了一个贝叶斯问题，将轨迹知识与来自离线数据的轨迹数据驱动表征最优结合，以处理一般类的随机干扰。在特定条件下，该问题被证明可以推广现有的数据驱动预测和控制算法。数值示例展示了统一方法在所有三项任务中相对于其他数据驱动和系统识别方法的性能。


### 论文摘要

Extending data-driven algorithms based on Willems' fundamental lemma to stochastic data often requires empirical and customized workarounds. This work presents a unified Bayesian framework that provides a systematic and general method for handling stochastic data-driven tasks, including smoothing, prediction, and control, via maximum a posteriori estimation. %This extends behavioral systems theory to stochastic data. This framework formulates a unified trajectory estimation problem for the three tasks by specifying different types of trajectory knowledge. Then, a Bayesian problem is solved that optimally combines trajectory knowledge with a data-driven characterization of the trajectory from offline data for a general class of stochastic disturbances. Under specific conditions, this problem is shown to generalize existing data-driven prediction and control algorithms. Numerical examples demonstrate the performance of the unified approach for all three tasks against other data-driven and system identification approaches.

---

## 152. MDiff4STR: Mask Diffusion Model for Scene Text Recognition

**论文链接:** [http://arxiv.org/abs/2512.01422v1](http://arxiv.org/abs/2512.01422v1)

**作者:** Yongkun Du, Miaomiao Zhao, Songlin Fan, Zhineng Chen, Caiyan Jia, Yu-Gang Jiang

**发布时间:** 2025-12-01

**备注:** Accepted by AAAI 2026 (Oral)

### GPT解析

### 总结

本研究首次将掩码扩散模型(MDMs)引入场景文本识别(STR)任务，提出MDiff4STR模型，通过两项关键改进策略解决了MDMs在STR任务中准确率低于自回归模型(ARMs)的问题，在保持高效推理的同时超越了最先进的ARMs。

### 背景

掩码扩散模型(MDMs)最近成为视觉语言任务中自回归模型(ARMs)的有前景替代方案，因其能在效率和准确性间提供灵活平衡。

### 目的

将MDMs应用于STR任务，解决原始MDM在STR任务中准确率落后于ARMs的问题，提高模型性能。

### 方法

提出MDiff4STR，针对STR任务的两项关键挑战开发改进策略：(1)开发六种噪声策略解决训练和推理之间的噪声差距；(2)提出令牌替换噪声机制解决推理过程中过度自信的预测问题。

### 主要发现

原始MDM在STR任务中虽然提高了识别效率，但在准确性方面落后于ARMs；MDiff4STR在多种场景下(包括不规则、艺术、遮挡和中文文本等)均优于流行STR模型，在准确性上超越最先进的ARMs，同时仅通过三个去噪步骤保持快速推理。

### 结论

MDiff4STR成功将MDMs应用于STR任务，通过针对性改进策略解决了MDMs在STR任务中的关键挑战，实现了高效准确的文本识别。

### 翻译

掩码扩散模型(MDMs)最近已成为视觉语言任务中自回归模型(ARMs)的一个有前景的替代方案，因为它们在效率和准确性之间提供了灵活的平衡。在本文中，我们首次将MDMs引入场景文本识别(STR)任务。我们发现，尽管原始MDM提高了识别效率，但在准确性方面落后于ARMs。为了缩小这一差距，我们提出了MDiff4STR，这是一个针对STR任务进行了两项关键改进策略增强的掩码扩散模型。具体来说，我们确定了将MDMs应用于STR任务时的两个关键挑战：训练和推理之间的噪声差距，以及推理过程中过度自信的预测。两者都显著阻碍了MDMs的性能。为缓解第一个问题，我们开发了六种噪声策略，使训练与推理行为更好地保持一致。对于第二个问题，我们提出了一种令牌替换噪声机制，提供非掩码噪声类型，鼓励模型重新考虑和修正过度自信但错误的预测。我们在标准且具有挑战性的STR基准测试上对MDiff4STR进行了广泛评估，涵盖了各种场景，包括不规则、艺术、遮挡和中文文本，以及是否使用预训练。在这些设置中，MDiff4STR始终优于流行的STR模型，在准确性上超越了最先进的ARMs，同时仅通过三个去噪步骤保持快速推理。代码：https://github.com/Topdu/OpenOCR。


### 论文摘要

Mask Diffusion Models (MDMs) have recently emerged as a promising alternative to auto-regressive models (ARMs) for vision-language tasks, owing to their flexible balance of efficiency and accuracy. In this paper, for the first time, we introduce MDMs into the Scene Text Recognition (STR) task. We show that vanilla MDM lags behind ARMs in terms of accuracy, although it improves recognition efficiency. To bridge this gap, we propose MDiff4STR, a Mask Diffusion model enhanced with two key improvement strategies tailored for STR. Specifically, we identify two key challenges in applying MDMs to STR: noising gap between training and inference, and overconfident predictions during inference. Both significantly hinder the performance of MDMs. To mitigate the first issue, we develop six noising strategies that better align training with inference behavior. For the second, we propose a token-replacement noise mechanism that provides a non-mask noise type, encouraging the model to reconsider and revise overly confident but incorrect predictions. We conduct extensive evaluations of MDiff4STR on both standard and challenging STR benchmarks, covering diverse scenarios including irregular, artistic, occluded, and Chinese text, as well as whether the use of pretraining. Across these settings, MDiff4STR consistently outperforms popular STR models, surpassing state-of-the-art ARMs in accuracy, while maintaining fast inference with only three denoising steps. Code: https://github.com/Topdu/OpenOCR.

---

## 153. CLAPS: Posterior-Aware Conformal Intervals via Last-Layer Laplace

**论文链接:** [http://arxiv.org/abs/2512.01384v1](http://arxiv.org/abs/2512.01384v1)

**作者:** Dongseok Kim, Hyoungsun Choi, Mohamed Jismy Aashik Rasool, Gisung Oh

**发布时间:** 2025-12-01

**备注:** 19 pages, 2 figures

### GPT解析

### 总结

CLAPS是一种结合了最后一层拉普拉斯近似和分割校准的自适应回归方法，通过定义双侧后验CDF评分使一致性度量与完整预测形状对齐，从而在相同目标覆盖率下产生更窄的预测区间，特别是在数据稀缺的小到中型表格数据集中表现优异。

### 背景

在回归问题中，不确定性建模对于小到中型表格数据集尤为重要，传统的基于残差的校准方法可能无法充分利用预测分布的全部信息。

### 目的

开发一种能够产生更窄预测区间同时保持目标覆盖率的回归方法，并提供工具帮助理解预测区间变化的原因。

### 方法

CLAPS方法结合了最后一层拉普拉斯近似和分割校准，从高斯后验分布中定义双侧后验CDF评分，并包含一个轻量级诊断工具用于分离偶然性和认知性不确定性成分。

### 主要发现

CLAPS在相同目标覆盖率下能产生更窄的预测区间，这种优势在小到中型表格数据集中尤为明显；在多个基准测试中始终实现了标称覆盖率，相比基于残差的校准基线提高了效率且开销最小。

### 结论

CLAPS为回归问题提供了一个清晰、实用的升级方案，通过将一致性度量与完整预测形状对齐，实现了更高效的预测区间估计，同时提供了诊断工具帮助理解预测行为。

### 翻译

我们提出了CLAPS，一种后验感知的自适应回归方法，结合了最后一层拉普拉斯近似和分割校准。从得到的高斯后验分布中，CLAPS定义了一个简单的双侧后验CDF评分，使一致性度量与完整预测形状对齐，而不仅仅是点估计。这种对齐在相同目标覆盖率下产生更窄的预测区间，特别是在数据稀缺的小到中型表格数据集中，不确定性建模尤为重要。我们还提供了一个轻量级诊断工具，用于分离偶然性和认知性成分，并可视化后验行为，帮助从业者理解预测区间何时以及为何收缩。在使用相同MLP主干网络的多个基准测试中，CLAPS始终实现了标称覆盖率，同时提高了效率且开销最小，为基于残差的自适应基线提供了清晰、实用的升级。


### 论文摘要

We present CLAPS, a posterior-aware conformal regression method that pairs a Last-Layer Laplace Approximation with split-conformal calibration. From the resulting Gaussian posterior, CLAPS defines a simple two-sided posterior CDF score that aligns the conformity metric with the full predictive shape, not just a point estimate. This alignment yields narrower prediction intervals at the same target coverage, especially on small to medium tabular datasets where data are scarce and uncertainty modeling matters. We also provide a lightweight diagnostic suite that separates aleatoric and epistemic components and visualizes posterior behavior, helping practitioners understand why intervals shrink when they do. Across multiple benchmarks using the same MLP backbone, CLAPS consistently attains nominal coverage with improved efficiency and minimal overhead, offering a clear, practical upgrade to residual-based conformal baselines.

---

## 154. Floquet Chern Insulators and Radiation-Induced Zero Resistance in Irradiated Graphene

**论文链接:** [http://arxiv.org/abs/2512.01346v1](http://arxiv.org/abs/2512.01346v1)

**作者:** Youngjae Kim, Kwon Park

**发布时间:** 2025-12-01

**备注:** 16 pages, 6 figures

### GPT解析

### 总结

研究预测在圆偏振光照射下，石墨烯可以承载两种具有零电阻的非平衡稳态物质：Floquet Chern绝缘体和辐射诱导的零电阻状态。

### 背景

光学和时间分辨技术的进步使得人们能够在非平衡条件下探索物质的新状态。

### 目的

预测在圆偏振光照射下石墨烯可能存在的非平衡稳态物质。

### 方法

计算非平衡异常霍尔电导率和纵向电导率，绘制照射石墨烯的非平衡相图，作为驱动频率和圆偏振光电场强度的函数。

### 主要发现

1. Floquet Chern绝缘体发生在高驱动频率下；2. 低驱动频率下出现负电阻和不规则电导行为；3. 负电阻引发灾难性崩溃，形成具有非均匀电流分布的零电阻状态。

### 结论

预测的零电阻状态与量子霍尔系统中观察到的辐射诱导零电阻状态相似。

### 翻译

最近光学和时间分辨技术的进步使得在非平衡条件下探索物质的新状态成为可能。本文预测，在圆偏振光照射下，石墨烯可以承载两种具有零电阻的新型非平衡稳态物质：(i) Floquet Chern绝缘体和(ii)具有自发形成非均匀电流分布的辐射诱导零电阻状态。具体而言，我们计算了非平衡异常霍尔电导率和纵向电导率，以绘制照射石墨烯的非平衡相图，作为圆偏振光驱动频率和电场强度的函数。结果表明，Floquet Chern绝缘体发生在高于石墨烯带宽的高驱动频率下。相比之下，在低于石墨烯带宽的低驱动频率下，非平衡异常霍尔电导率偏离预期的量化值，非平衡纵向电导率表现出高度不规则的行为，包括负电阻。预测热力学不稳定的负电阻将引发灾难性崩溃，诱导出具有自发形成非均匀电流分布的零电阻状态，类似于量子霍尔系统中观察到的辐射诱导零电阻状态。


### 论文摘要

Recent advances in optics and time-resolved techniques have facilitated the exploration of new states of matter under nonequilibrium conditions. Here, we predict that irradiated graphene can host two novel nonequilibrium steady states of matter with zero resistance when exposed to circularly polarized light: (i) Floquet Chern insulators and (ii) a radiation-induced zero-resistance state with spontaneous formation of an inhomogeneous current distribution. Specifically, we calculate nonequilibrium anomalous Hall and longitudinal conductivities to map the nonequilibrium phase diagram of irradiated graphene as a function of the driving frequency and the electric-field strength of circularly polarized light. As a result, Floquet Chern insulators are found to occur at high driving frequencies above the graphene band width. By contrast, at low driving frequencies below the graphene band width, the nonequilibrium anomalous Hall conductivity deviates from the expected quantized values, and the nonequilibrium longitudinal conductivity exhibits highly irregular behavior, including negative resistance. It is predicted that the thermodynamically unstable negative resistance will trigger a catastrophic breakdown, inducing a zero-resistance state with spontaneous formation of an inhomogeneous current distribution, similar to the radiation-induced zero-resistance state observed in quantum Hall systems.

---

## 155. On the importance of numerical integration details for homogeneous flow simulation

**论文链接:** [http://arxiv.org/abs/2512.01318v1](http://arxiv.org/abs/2512.01318v1)

**作者:** Stephen Sanderson, Debra J. Searles

**发布时间:** 2025-12-01

**备注:** 16 pages, 4 figures

### GPT解析

### 总结

本文介绍了一种可逆且能量守恒的Sllod运动方程积分方案，解决了现有代码中缺乏可逆数值积分方案的问题，提高了模拟准确性，特别是在高流速条件下。

### 背景

Sllod运动方程能够模拟原子尺度的均匀流动并预测流体性质如粘度，但公开可用的支持此类模拟的代码很少，且现有代码通常没有实现可逆的数值积分方案或有其他问题。

### 目的

实现一个可逆且能量守恒的Sllod运动方程积分方案，提高模拟的准确性，特别是在高流速条件下的流体性质预测。

### 方法

开发了一个可逆且能量守恒的积分方案，误差约为δt^3量级，与标准分子动力学模拟中使用的典型算子分裂积分器一致，并在LAMMPS软件中实现了该方案。

### 主要发现

新方案能够更准确地模拟瞬态响应、混合流动和稳态，特别是在高流速下；缺乏能量守恒会导致压力张量的系综平均值产生系统误差，使计算出的粘度在高流速下出现显著误差。

### 结论

新实现方案解决了Sllod运动方程模拟中的能量守恒问题，提高了模拟准确性，特别是在高流速条件下，避免了压力张量计算的系统误差，从而更准确地预测流体粘度。

### 翻译

Sllod运动方程能够在原子尺度上模拟均匀流动，常用于预测流体性质如粘度。然而，很少有公开可用的代码支持此类模拟，而现有的代码通常没有实现可逆的数值积分方案或有其他细微问题。本文展示了一种可逆且能量守恒的Sllod运动方程积分方案，误差约为δt^3量级，与标准分子动力学模拟中使用的典型算子分裂积分器一致。我们讨论了各种实现细节，并在LAMMPS中实现了该方案，发现我们的改进能够更准确地模拟瞬态响应、混合流动和稳态，特别是在高流速下。重要的是，我们表明能量守恒的缺失可能导致压力张量的系综平均值出现系统误差，从而在高流速下导致计算出的粘度出现显著误差。


### 论文摘要

The Sllod equations of motion enable modeling of homogeneous flow at the atomic scale, and are commonly used to predict fluid properties such as viscosity. However, few publicly available codes support such simulations, and those that do often do not implement a reversible numerical integration scheme or have other subtle problems. Here, we demonstrate a reversible and energy-conserving integration scheme for the Sllod equations of motion with error on the order of $δt^3$, in line with typical operator splitting integrators used in standard molecular dynamics simulations. We discuss various implementation details, and implement the scheme in LAMMPS where we find that our changes enable more accurate simulation of transient responses, mixed flows, and steady states, especially at high rates of flow. Importantly, we show that a lack of energy conservation can manifest as a systematic error in the direct ensemble average of the pressure tensor, leading to an error in the calculated viscosity which becomes significant at high flow rates.

---

## 156. Virtual Observability in Sequential Play

**论文链接:** [http://arxiv.org/abs/2512.01244v1](http://arxiv.org/abs/2512.01244v1)

**作者:** C. Monica Capra, Charles A. Holt, Po-Hsuan Lin

**发布时间:** 2025-12-01

**备注:** 27 pages, 5 figures, 2 tables

### GPT解析

### 总结

本研究探讨了在没有信息揭示的情况下，行动时机如何影响玩家行为，特别是在有唯一均衡的博弈中。

### 背景

当玩家做出彼此不可观察的顺序决策时，他们的行为可能受到谁先行动的影响，这种现象被称为'虚拟可观察性'。然而，虚拟可观察性的原始概念在有唯一均衡的博弈中没有作用。

### 目的

实验检验在具有唯一均衡的博弈中，时机是否仍然影响玩家行为。

### 方法

使用旅行者困境和信任博弈进行实验研究，比较顺序决策和同时决策条件下玩家行为的差异。

### 主要发现

在不可观察的顺序旅行者困境中，先行者的行为比同时版本更接近均衡预测；而在信任博弈中，没有观察性的时机对行为没有影响。

### 结论

时机在没有信息揭示的博弈中确实会影响行为，但这种影响可能因博弈类型而异。

### 翻译

当玩家做出彼此不可观察的顺序决策时，他们的行为仍然可能受到谁先行动的影响。这种被称为'虚拟可观察性'的顺序结构表明，即使没有信息被揭示，时机本身也能塑造期望和选择。然而，虚拟可观察性的原始概念是一种基于时间结构的均衡精炼，在有唯一均衡的博弈中没有作用。在本文中，我们使用旅行者困境和信任博弈实验性地检验在这样的博弈中时机是否仍然影响行为。我们发现，在不可观察的顺序旅行者困境中，先行者的行为比同时版本更接近均衡预测。相比之下，在信任博弈中，没有观察性的时机对行为没有影响。


### 论文摘要

When players make sequential decisions that are unobservable to one another, their behavior can nonetheless be influenced by knowing who moves first. This sequential structure, often referred to as "virtual observability," suggests that timing alone can shape expectations and choices, even when no information is revealed. The original notion of virtual observability, however, is an equilibrium refinement based on the timing structure and has no bite in games with a unique equilibrium. In this paper, we experimentally examine whether timing still affects behavior in such games, using the Traveler's Dilemma and the Trust Game. We find that in the sequential Traveler's Dilemma without observability, first movers tend to behave closer to the equilibrium prediction than in the simultaneous version. In contrast, timing without observability has no effect on behavior in the Trust Game.

---

## 157. Autonomous Navigation and Station-Keeping on Near-Rectilinear Halo Orbits

**论文链接:** [http://arxiv.org/abs/2512.01182v1](http://arxiv.org/abs/2512.01182v1)

**作者:** Yuri Shimane, Karl Berntorp, Stefano Di Cairano, Avishai Weiss

**发布时间:** 2025-12-01

### GPT解析

### 总结

该研究开发了一种用于近直线晕轨道的光学导航和定点保持管道，结合非迭代地平线OPNAV算法和扩展卡尔曼滤波器，并引入了改进的目标预测方案和滞后机制，有效降低了定点保持成本。

### 背景

在高精度星历模型动力学环境下，航天器在近直线晕轨道上的导航和定点保持面临挑战，需要开发有效的导航和控制系统。

### 目的

开发一个光学导航和定点保持管道，使航天器能够在近直线晕轨道附近保持精确位置，同时优化燃料消耗。

### 方法

使用合成图像的非迭代基于地平线的OPNAV算法，结合扩展卡尔曼滤波器进行状态估计；研究基于微分校正和基于最小化的控制方案；提出将滤波器状态协方差与无迹变换结合的预测方案；引入滞后机制；通过蒙特卡洛实验评估性能。

### 主要发现

滤波器性能随传感器视场和测量位置变化；无迹变换预测和滞后机制有效降低了定点保持成本；累积ΔV值因OPNAV滤波器估计精度的周期性结构而随机动位置变化。

### 结论

所开发的光学导航和定点保持管道在高精度星历模型下能有效工作，改进的预测方案和滞后机制显著提高了性能，为近直线晕轨道任务提供了实用的导航和定点保持解决方案。

### 翻译

本文开发了一种用于高精度星历模型动力学中近直线晕轨道的光学导航(OPNAV)和定点保持管道。该管道涉及使用非迭代基于地平线的OPNAV算法的合成图像，输入到扩展卡尔曼滤波器中。状态估计被控制器用于保持航天器在参考NRHO附近的运动。我们研究了基于微分校正和基于最小化的x轴穿越控制方案实现，并提出了一种改进的目标预测方案，通过将滤波器的状态协方差与无迹变换相结合。我们还引入了一种滞后机制，提高了定点保持成本，并提供了对基于微分校正和基于最小化的方法之间性能差异的见解。我们进行了蒙特卡洛实验来评估该管道的跟踪和ΔV性能。我们报告了几个关键发现，包括滤波器性能随传感器视场和测量位置的变化、通过无迹变换预测和滞后实现的定点保持成本降低，以及由于OPNAV基础滤波器估计精度的周期性结构，累积ΔV作为机动位置函数的变化。


### 论文摘要

This article develops an optical navigation (OPNAV) and station-keeping pipeline for the near-rectilinear halo orbit (NRHO) in high-fidelity ephemeris model dynamics. The pipeline involves synthetic images used by the non-iterative horizon-based OPNAV algorithm, fed into an extended Kalman filter. The state estimate is used by a controller to maintain the spacecraft's motion within the vicinity of a reference NRHO. We study differential correction-based and minimization-based implementations of the x-axis crossing control scheme, and propose an improved targeting prediction scheme by incorporating the filter's state covariance with an unscented transform. We also introduce a hysteresis mechanism, which improves station-keeping cost and provides insight into the difference in performance between the differential correction-based and minimization-based approaches. We perform Monte-Carlo experiments to assess the pipeline's tracking and ΔV performances. We report several key findings, including the variability of the filter performance with the sensor field of view and measurement locations, station-keeping cost reduction achieved by the unscented transform-based prediction and hysteresis, as well as variability of the cumulative ΔV as a function of maneuver location due to the periodic structure in the OPNAV-based filter's estimation accuracy.

---

## 158. From Regression to Classification: Exploring the Benefits of Categorical Representations of Energy in MLIPs

**论文链接:** [http://arxiv.org/abs/2512.01160v1](http://arxiv.org/abs/2512.01160v1)

**作者:** Ahmad Ali

**发布时间:** 2025-12-01

**备注:** 11th Annual Conference on Vision and Intelligent Systems (CVIS 2025)

### GPT解析

### 总结

本文提出了一种基于多类分类的机器学习原子间势方法，替代传统的标量回归方法，通过预测能量/力值的分类分布来提供更丰富的监督和模型不确定性量化。

### 背景

密度泛函理论(DFT)是广泛用于估算分子能量和行为的计算方法，但计算成本高。机器学习原子间势(MLIPs)是训练用来近似DFT级别能量和力的模型，计算成本显著降低。

### 目的

探索一种多类分类公式来预测能量/力值的分类分布，提供更丰富的监督，并实现模型不确定性的原则性量化。

### 方法

提出预测能量/力分布直方图的方法，将标量目标转换为直方图，并使用交叉熵损失训练模型。这种方法通过预测分布的熵来认识论不确定性量化。

### 主要发现

分类公式可以实现与回归基线相当的绝对误差性能，同时能够量化模型置信度，这是标量回归方法所缺乏的。

### 结论

基于多类分类的MLIPs不仅能够保持与回归方法相当的准确性，还能提供模型不确定性的量化，为分子模拟提供了更可靠的工具。

### 翻译

密度泛函理论(DFT)是一种广泛用于估算分子能量和行为的计算方法。机器学习原子间势(MLIPs)是经过训练以近似DFT级别能量和力的模型，计算成本显著降低。许多现代MLIPs依赖于标量回归公式；给定分子的信息，它们预测单一的能量值和相应的力，同时最小化与DFT计算之间的绝对误差。在这项工作中，我们探索了一种多类分类公式，预测能量/力值的分类分布，通过多个目标提供更丰富的监督。最重要的是，这种方法提供了量化模型不确定性的原则性方法。特别是，我们的方法预测能量/力分布的直方图，将标量目标转换为直方图，并使用交叉熵损失训练模型。我们的结果表明，这种分类公式可以实现与回归基线相当的绝对误差性能。此外，这种表示可以通过预测分布的熵来认识论不确定性量化，提供了标量回归方法中缺乏的模型置信度度量。


### 论文摘要

Density Functional Theory (DFT) is a widely used computational method for estimating the energy and behavior of molecules. Machine Learning Interatomic Potentials (MLIPs) are models trained to approximate DFT-level energies and forces at dramatically lower computational cost. Many modern MLIPs rely on a scalar regression formulation; given information about a molecule, they predict a single energy value and corresponding forces while minimizing absolute error with DFT's calculations. In this work, we explore a multi-class classification formulation that predicts a categorical distribution over energy/force values, providing richer supervision through multiple targets. Most importantly, this approach offers a principled way to quantify model uncertainty.   In particular, our method predicts a histogram of the energy/force distribution, converts scalar targets into histograms, and trains the model using cross-entropy loss. Our results demonstrate that this categorical formulation can achieve absolute error performance comparable to regression baselines. Furthermore, this representation enables the quantification of epistemic uncertainty through the entropy of the predicted distribution, offering a measure of model confidence absent in scalar regression approaches.

---

## 159. Market Sensitivities and Growth Differentials Across Australian Housing Markets

**论文链接:** [http://arxiv.org/abs/2512.01139v1](http://arxiv.org/abs/2512.01139v1)

**作者:** Willem P Sijp

**发布时间:** 2025-11-30

### GPT解析

### 总结

澳大利亚房价自1990年代中期以来大幅上涨但区域增长不均衡，研究通过三因素模型分析这种差异是结构性趋势还是周期性波动

### 背景

澳大利亚房价自1990年代中期以来大幅上涨，但各地区增长极不均衡

### 目的

确定房价区域差异是反映持续的结构性趋势还是周期性波动

### 方法

使用三因素模型对1995-2024年区域重复销售价格指数进行水平估计，将每个区域指数分解为全国市场因素、两个平稳的价差（矿业和生活方式）以及城市特定的残差

### 主要发现

矿业价差反映资源驱动的相对表现波动；生活方式价差捕捉设施驱动的沿海和区域周期；市场负载隔离各地区对全国增长的基本敏感性；市场贝塔值在重大冲击中保持稳定；墨尔本放大全国增长，悉尼紧密跟随全国趋势，区域地区则减弱增长

### 结论

该框架提供了一种简单的基于因素的工具来解释区域增长差异及其持续性

### 翻译

澳大利亚房价自1990年代中期以来大幅上涨，但各地区增长极不均衡。原始增长数据掩盖了这些差异是反映持续的结构性趋势还是周期性波动。我们通过估计1995-2024年区域重复销售价格指数水平的三因素模型来解决这个问题。该模型将每个区域指数分解为全国市场因素、两个平稳的价差（矿业和生活方式）以及城市特定的残差。矿业价差以珀斯-悉尼指数差异为代表，反映了资源驱动的相对表现波动；生活方式价差捕捉了设施驱动的沿海和区域周期。市场负载隔离了各地区对全国增长的基本敏感性，贝塔值，因此在假设全国变化下，一个城市的增长是通过其贝塔值计算得出的，一旦扣除了均值回归的价差。将实际路径与这些因素暗示的轨迹进行比较，可以表明一个城市在历史上是处于高位还是低位，并将这一差距归因于矿业或生活方式价差。扩展窗口ARIMAX估计显示，市场贝塔值在重大冲击（矿业繁荣、全球金融危机和COVID-19）中保持稳定，而矿业和生活方式行为作为平稳的价差，扩大了预测通道，而没有推翻贝塔值暗示的横截面排名。墨尔本放大全国增长，悉尼紧密跟随全国趋势，而区域地区则减弱增长。因此，该框架提供了一个简单的基于因素的工具，用于解释区域增长差异及其持续性。


### 论文摘要

Australian house prices have risen strongly since the mid-1990s, but growth has been highly uneven across regions. Raw growth figures obscure whether these differences reflect persistent structural trends or cyclical fluctuations. We address this by estimating a three-factor model in levels for regional repeat-sales log price indexes over 1995-2024. The model decomposes each regional index into a national Market factor, two stationary spreads (Mining and Lifestyle) that capture mean-reverting geographic cycles, and a city-specific residual. The Mining spread, proxied by a Perth-Sydney index differential, reflects resource-driven oscillations in relative performance; the Lifestyle spread captures amenity-driven coastal and regional cycles. The Market loading isolates each region's fundamental sensitivity, beta, to national growth, so that a city's growth under an assumed national change is calculated from its beta once mean-reverting spreads are netted out. Comparing realised paths to these factor-implied trajectories indicates when a city is historically elevated or depressed, and attributes the gap to Mining or Lifestyle spreads.   Expanding-window ARIMAX estimation reveals that Market betas are stable across major shocks (the mining boom, the Global Financial Crisis, and COVID-19), while Mining and Lifestyle behave as stationary spreads that widen forecast funnels without overturning the cross-sectional ranking implied by beta. Melbourne amplifies national growth, Sydney tracks the national trend closely, and regional areas dampen it. The framework thus provides a simple, factor-based tool for interpreting regional growth differentials and their persistence.

---

## 160. A Neuromodulable Current-Mode Silicon Neuron for Robust and Adaptive Neuromorphic Systems

**论文链接:** [http://arxiv.org/abs/2512.01133v1](http://arxiv.org/abs/2512.01133v1)

**作者:** Loris Mendolia, Chenxi Wen, Elisabetta Chicca, Giacomo Indiveri, Rodolphe Sepulchre, Jean-Michel Redouté, Alessio Franci

**发布时间:** 2025-11-30

**备注:** 18 pages, 13 figures

### GPT解析

### 总结

本文提出了一种新型电流模式神经元设计，支持稳健的神经调制，具有模型简单、与标准CMOS技术兼容的特点。通过理论和实验验证，该神经元表现出类似生物的神经调制适应能力，具有高度的鲁棒性、灵活性和可扩展性，适合实际神经形态应用。

### 背景

神经形态工程利用混合信号模拟和数字电路直接模拟生物大脑的计算原理。这类电子系统在从边缘计算到机器人技术的广泛任务中表现出高度的适应性、鲁棒性和能效。生物神经元的一个重要特征是它们能够通过神经调节根据上下文调整其输入响应和尖峰模式，从而执行稳健可靠的计算。

### 目的

在神经形态电路中通过调节机制实现类似生物水平的鲁棒性和适应性是一个 largely 未被探索的领域。本研究旨在设计一种支持稳健神经调制的电流模式神经元，具有最小模型复杂度，并与标准CMOS技术兼容。

### 方法

作者首先介绍了一种电路的数学模型，并提供了分析和调整神经元行为的工具。然后通过理论和实验验证了电路在大参数范围内的生物合理神经调制适应能力。所有理论预测都在所提出的神经元电路的低功耗180纳米CMOS实现实验中得到验证。

### 主要发现

由于底层的模拟反馈结构，所提出的自适应可调制神经元表现出高度的鲁棒性、灵活性和可扩展性，在电流和温度的操作范围内都能正常工作。这使得它成为实际神经形态应用的理想候选者。

### 结论

这种新型电流模式神经元设计成功实现了类似生物的神经调制功能，具有简单模型和CMOS兼容性，在多种环境条件下表现出稳定的性能，为实际神经形态应用提供了有价值的解决方案。

### 翻译

神经形态工程利用混合信号模拟和数字电路来直接模拟生物大脑的计算原理。这类电子系统在从边缘计算到机器人技术的广泛任务中表现出高度的适应性、鲁棒性和能效。在此背景下，我们研究了生物神经元的一个关键特征：它们能够通过神经调节根据上下文调整输入响应和尖峰模式，从而执行稳健可靠的计算。在神经形态电路中通过调节机制实现类似生物水平的鲁棒性和适应性是一个 largely 未被探索的领域。我们提出了一种新型电流模式神经元设计，支持稳健的神经调制，具有最小模型复杂度，与标准CMOS技术兼容。我们首先介绍了电路的数学模型，并提供了分析和调整神经元行为的工具；然后通过理论和实验证明了该电路在大参数范围内的生物合理神经调制适应能力。所有理论预测都在所提出的神经元电路的低功耗180纳米CMOS实现实验中得到验证。由于底层的模拟反馈结构，所提出的自适应可调制神经元表现出高度的鲁棒性、灵活性和可扩展性，在电流和温度的操作范围内都能正常工作，使其成为实际神经形态应用的完美候选者。


### 论文摘要

Neuromorphic engineering makes use of mixed-signal analog and digital circuits to directly emulate the computational principles of biological brains. Such electronic systems offer a high degree of adaptability, robustness, and energy efficiency across a wide range of tasks, from edge computing to robotics. Within this context, we investigate a key feature of biological neurons: their ability to carry out robust and reliable computation by adapting their input response and spiking pattern to context through neuromodulation. Achieving analogous levels of robustness and adaptation in neuromorphic circuits through modulatory mechanisms is a largely unexplored path. We present a novel current-mode neuron design that supports robust neuromodulation with minimal model complexity, compatible with standard CMOS technologies. We first introduce a mathematical model of the circuit and provide tools to analyze and tune the neuron behavior; we then demonstrate both theoretically and experimentally the biologically plausible neuromodulation adaptation capabilities of the circuit over a wide range of parameters. All the theoretical predictions were verified in experiments on a low-power 180 nm CMOS implementation of the proposed neuron circuit. Due to the analog underlying feedback structure, the proposed adaptive neuromodulable neuron exhibits a high degree of robustness, flexibility, and scalability across operating ranges of currents and temperatures, making it a perfect candidate for real-world neuromorphic applications.

---

## 161. Estimation of Kinematic Motion from Dashcam Footage

**论文链接:** [http://arxiv.org/abs/2512.01104v1](http://arxiv.org/abs/2512.01104v1)

**作者:** Evelyn Zhang, Alex Richardson, Jonathan Sprinkle

**发布时间:** 2025-11-30

**备注:** 8 pages, 10 figures

### GPT解析

### 总结

本研究评估了使用行车记录仪预测车辆运动参数的准确性，并提供了数据收集和实验方法。

### 背景

研究使用来自车辆车载数据流的真实信息，通过控制器局域网和时间同步的仪表板摄像头，收集了18小时的录像和驾驶数据。

### 目的

探索使用行车记录仪录像来预测类似汽车车辆的实际运动学运动的准确性。

### 方法

论文贡献了神经网络模型，使研究人员能够量化预测车辆速度和偏航角度的准确性，以及前方车辆的存在性及其相对距离和速度。

### 主要发现

研究成功量化了使用行车记录仪预测车辆速度、偏航角度、前方车辆存在性及其相对距离和速度的准确性。

### 结论

论文描述了其他研究人员如何使用开源工具和现成技术收集自己的数据来进行类似的实验。

### 翻译

本文的目的是探索使用行车记录仪录像来预测类似汽车车辆的实际运动学运动的准确性。我们的方法使用来自车辆车载数据流的真实信息，通过控制器局域网，以及一个安装在消费级车辆上的时间同步仪表板摄像头，收集了18小时的录像和驾驶数据。论文的贡献包括神经网络模型，使我们能够量化预测车辆速度和偏航角度的准确性，以及前方车辆的存在性及其相对距离和速度。此外，论文描述了其他研究人员如何使用开源工具和现成技术收集自己的数据来进行类似的实验。


### 论文摘要

The goal of this paper is to explore the accuracy of dashcam footage to predict the actual kinematic motion of a car-like vehicle. Our approach uses ground truth information from the vehicle's on-board data stream, through the controller area network, and a time-synchronized dashboard camera, mounted to a consumer-grade vehicle, for 18 hours of footage and driving. The contributions of the paper include neural network models that allow us to quantify the accuracy of predicting the vehicle speed and yaw, as well as the presence of a lead vehicle, and its relative distance and speed. In addition, the paper describes how other researchers can gather their own data to perform similar experiments, using open-source tools and off-the-shelf technology.

---

## 162. Euclid Structural-Thermal-Optical Performance

**论文链接:** [http://arxiv.org/abs/2512.01075v1](http://arxiv.org/abs/2512.01075v1)

**作者:** Euclid Collaboration, A. Anselmi, R. Laureijs, G. D. Racca, G. Costa, L. Courcould Mifsud, J. -C. Cuillandre, M. Gottero, H. Hoekstra, K. Kuijken, V. Mareschi, L. Miller, S. Mottini, D. Stramaccioni, B. Altieri, A. Amara, S. Andreon, N. Auricchio, C. Baccigalupi, M. Baldi, A. Balestra, S. Bardelli, R. Bender, A. Biviano, E. Branchini, M. Brescia, S. Camera, G. Canas-Herrera, V. Capobianco, C. Carbone, J. Carretero, M. Castellano, G. Castignani, S. Cavuoti, A. Cimatti, C. Colodro-Conde, G. Congedo, C. J. Conselice, L. Conversi, Y. Copin, F. Courbin, H. M. Courtois, M. Cropper, A. Da Silva, H. Degaudenzi, G. De Lucia, H. Dole, F. Dubath, F. Ducret, C. A. J. Duncan, X. Dupac, S. Dusini, S. Escoffier, M. Fabricius M. Farina, R. Farinelli, F. Faustini, S. Ferriol, F. Finelli, N. Fourmanoit, M. Frailis, E. Franceschi, M. Fumana, S. Galeotta, K. George, B. Gillis, C. Giocoli, J. Gracia-Carpio, A. Grazian, F. Grupp, S. V. H. Haugan, J. Hoar, W. Holmes, F. Hormuth, A. Hornstrup, K. Jahnke, M. Jhabvala, E. Keihanen, S. Kermiche, A. Kiessling, R. Kohley, B. Kubik, M. Kunz, H. Kurki-Suonio, A. M. C. Le Brun, S. Ligori, P. B. Lilje, V. Lindholm, I. Lloro, G. Mainetti, D. Maino, E. Maiorano, O. Mansutti, O. Marggraf, M. Martinelli, N. Martinet, F. Marulli, R. J. Massey, E. Medinaceli, S. Mei, Y. Mellier, M. Meneghetti, E. Merlin, G. Meylan, A. Mora, M. Moresco, L. Moscardini, R. Nakajima, C. Neissner, R. C. Nichol, S. -M. Niemi C. Padilla, S. Paltani, F. Pasian, K. Pedersen, W. J. Percival, V. Pettorino, S. Pires, G. Polenta, M. Poncet, L. A. Popa, F. Raison, R. Rebolo, A. Renzi, J. Rhodes, G. Riccio, E. Romelli, M. Roncarelli, C. Rosset, E. Rossetti, R. Saglia, Z. Sakr, J. -C. Salvignol, A. G. Sanchez, D. Sapone, B. Sartoris, M. Schirmer, P. Schneider, T. Schrabback, A. Secroun, G. Seidel, S. Serrano, C. Sirignano, G. Sirri, J. Skottfelt, L. Stanco, J. Steinwagner, P. Tallada-Cresp, D. Tavagnacco, A. N. Taylor, H. I. Teplitz, I. Tereno, N. Tessore, S. Toft, R. Toledo-Moreo, F. Torradeflot, I. Tutusaus, E. A. Valentijn, L. Valenziano, J. Valiviita, T. Vassallo, G. Verdoes Kleijn, A. Veropalumbo, Y. Wang, J. Weller, A. Zacchei, G. Zamorani, E. Zucca, M. Ballardini, M. Bolzonella, E. Bozzo, C. Burigana, R. Cabanac, A. Cappi, J. A. Escartin Vigo, L. Gabarra W. G. Hartley, J. Martin Fleitas, S. Matthew, N. Mauri, R. B. Metcalf, A. Pezzotta, M. Pontinen, I. Risso, V. Scottez, M. Sereno, M. Tenti, M. Viel, M. Wiesmann, Y. Akrami, I. T. Andika, S. Anselmi, M. Archidiacono, F. Atrio-Barandela, D. Bertacca, M. Bethermin, A. Blanchard, L. Blot, M. Bonici, S. Borgani, M. L. Brown, S. Bruton, A. Calabro, B. Camacho Quevedo, F. Caro, C. S. Carvalho, T. Castro, F. Cogato, S. Conseil, A. R. Cooray, O. Cucciati, S. Davini, G. Desprez, A. Diaz-Sanchez, J. J. Diaz, S. Di Domizio, J. M. Diego M. Y. Elkhashab, A. Enia, Y. Fang, A. G. Ferrari, A. Finoguenov, A. Franco, K. Ganga, J. Garcia-Bellido, T. Gasparetto, E. Gaztanaga, F. Giacomini, F. Gianotti, G. Gozaliasl, M. Guidi, C. M. Gutierrez, A. Hall, H. Hildebrandt, J. Hjorth, J. J. E. Kajava, Y. Kang, V. Kansal, D. Karagiannis, K. Kiiveri, J. Kim, C. C. Kirkpatrick, S. Kruk, J. Le Graet, L. Legrand, M. Lembo, F. Lepori, G. Leroy, G. F. Lesci, J. Lesgourgues, L. Leuzzi, T. I. Liaudat, S. J. Liu, A. Loureiro, J. Macias-Perez, M. Magliocchetti, F. Mannucci, R. Maoli, C. J. A. P. Martins, L. Maurin, M. Miluzio, P. Monaco, A. Montoro, C. Moretti, G. Morgante, S. Nadathur, K. Naidoo, A. Navarro-Alsina, S. Nesseris, D. Paoletti, F. Passalacqua, K. Paterson, L. Patrizii, A. Pisani, D. Potter, S. Quai, M. Radovich, S. Sacquegna, M. Sahlen, D. B. Sanders, E. Sarpa, A. Schneider, D. Sciotti, E. Sellentin, L. C. Smith, K. Tanidis, G. Testera, R. Teyssier, S. Tosi, A. Troja, M. Tucci, C. Valieri, A. Venhola, D. Vergani, G. Verza, P. Vielzeuf, N. A. Walton

**发布时间:** 2025-11-30

**备注:** 21 pages, 15 figures, submitted to A&A

### GPT解析

### 总结

欧几里得系统性能通过针对弱引力透镜宇宙学探测调整的图像质量指标来定义。研究通过结构-热-光学性能分析验证了望远镜及其航天器接口满足在轨稳态和瞬态图像质量要求，并证实了望远镜的出色整体性能。

### 背景

弱引力透镜(WL)对VIS仪器系统的点扩散函数(PSF)的形状和稳定性提出了严格要求。PSF受到来自望远镜、焦平面和图像运动的误差贡献影响，并通过全局误差预算进行控制，对每个贡献者进行误差分配。

### 目的

在航天器开发期间，通过结构-热-光学性能(STOP)分析验证已建造和验证的望远镜及其航天器接口满足在轨稳态和瞬态图像质量要求。

### 方法

为了进行STOP分析，建立了一个详细的有限元数学模型，并定义了一组标准的测试用例，包括稳态和瞬态情况，由最坏情况的边界条件组合构成。

### 主要发现

STOP分析解决了所有航天器组件在传递导致光学系统变形的温度载荷时的相互作用。发射前分析结果表明，温度引起的光学扰动将远低于所有允许观测条件的限制。在轨第一年，使用STOP分析预测帮助解释了作为环境变量函数的测量性能，发现了未预测到的干扰和意外的敏感性。轨道上的温度变化很小(<300 mK)，对望远镜结构影响不大，但在图像质量指标的时间历程中可检测到，且是弱引力透镜科学所要求的PSF稳定性预算中不可忽视的因素。

### 结论

总体而言，分析证实了望远镜的出色整体性能。

### 翻译

欧几里得系统性能通过针对弱引力透镜(WL)宇宙学探测调整的图像质量指标来定义。弱引力透镜对VIS仪器系统的点扩散函数(PSF)的形状和稳定性提出了严格要求。PSF受到来自望远镜、焦平面和图像运动的误差贡献影响，并通过全局误差预算进行控制，对每个贡献者进行误差分配。目的：在航天器开发期间，我们通过结构-热-光学性能(STOP)分析验证了已建造和验证的望远镜及其航天器接口满足在轨稳态和瞬态图像质量要求。方法：为了进行STOP分析，建立了一个详细的有限元数学模型，并定义了一组标准的测试用例，包括稳态和瞬态情况，由最坏情况的边界条件组合构成。结果：STOP分析解决了所有航天器组件在传递导致光学系统变形的温度载荷时的相互作用。发射前分析结果表明，温度引起的光学扰动将远低于所有允许观测条件的限制。在轨第一年，我们使用STOP分析预测帮助解释了作为环境变量函数的测量性能。发现了未预测到的干扰和意外的敏感性。轨道上的温度变化很小(<300 mK)，对望远镜结构影响不大，但在图像质量指标的时间历程中可检测到，且是弱引力透镜科学所要求的PSF稳定性预算中不可忽视的因素。总而言之，我们的分析证实了望远镜的出色整体性能。


### 论文摘要

The Euclid system performance is defined in terms of image quality metrics tuned to the weak gravitational lensing (WL) cosmological probe. WL induces stringent requirements on the shape and stability of the VIS instrument system point spread function (PSF). The PSF is affected by error contributions from the telescope, the focal plane and image motion, and is controlled by a global error budget with error allocations to each contributor. Aims. During spacecraft development, we verified through a structural-thermal-optical performance (STOP) analysis that the built and verified telescope with its spacecraft interface meets the in-orbit steady-state and transient image quality requirements. Methods. For the purposes of the STOP analysis, a detailed finite-element mathematical model was set up and a standard set of test cases, both steady-state and transient, was defined, comprising combinations of worst-case boundary conditions. Results. The STOP analysis addressed the interaction of all spacecraft components in transmitting temperature-induced loads that lead to optical train deformation. The results of the prelaunch analysis demonstrated that temperature-induced optical perturbations will be well below the allowable limits for all permitted observing conditions. During the first year in orbit, we used the STOP analysis predictions to help interpret the measured performance as a function of environmental variables. Unpredicted disturbances were discovered and unexpected sensitivities were revealed. In-orbit temperature variations are small (<300 mK) and so are their effects on the telescope structure, but they are detected in the time histories of the image quality metrics and are a non-negligible factor in the PSF stability budget demanded by the WL science. Taking everything into account, our analysis confirms the excellent overall performance of the telescope.

---

## 163. Newsvendor Decisions under Stochastic and Strategic Uncertainties: Theory and Experimental Evidence

**论文链接:** [http://arxiv.org/abs/2512.00994v1](http://arxiv.org/abs/2512.00994v1)

**作者:** Hang Wu, Qin Wu, Yue Liu, Mengmeng Shi

**发布时间:** 2025-11-30

### GPT解析

### 总结

研究数字商务平台中竞争零售商的协调定价和库存管理决策，分析序列双头垄断报童博弈，发现理论与实践存在差异。

### 背景

数字商务平台的快速发展增强了竞争零售商之间协调定价和库存管理决策的战略重要性。

### 目的

分析零售商在序列双头垄断报童博弈中的定价和库存决策行为，探索理论与实践的差异。

### 方法

研究序列双头垄断报童博弈，零售商首先公开定价，然后在需求不确定性下做出私人库存决策，并通过实验室实验验证理论预测。

### 主要发现

1) 理论预测：更高的利润率和需求不确定性会加剧价格竞争，最优库存反应受利润率影响；2) 实验证据：参与者不愿在价格上竞争，常协调在焦点价格上，对需求不确定性不敏感，库存决策对价格不敏感且存在'拉向中心'偏见。

### 结论

竞争下定价和库存决策之间存在脱节，零售运营中需考虑持久的行为倾向。

### 翻译

数字商务平台的快速扩张增强了竞争零售商之间协调定价和库存管理决策的战略重要性。受领先电商平台实践的启发，我们分析了一个序列双头垄断报童博弈，其中零售商首先公开定价，然后在需求不确定性下做出私人库存决策。我们的理论预测，更高的利润率和需求不确定性会加剧价格竞争，而对需求不确定性的最优库存反应则受利润率的影响。然而，实验室证据显示，参与者通常不愿意在价格上竞争，经常协调在显著的焦点（保留）价格上，特别是在低利润率环境中，并且在定价方面对需求不确定性不敏感。在库存方面，参与者的订购数量对所选价格不敏感，并继续表现出广为人知的'拉向中心'偏见。这些发现揭示了竞争下定价和库存决策之间的脱节，并强调了在零售运营中考虑持久行为倾向的重要性。


### 论文摘要

The rapid expansion of digital commerce platforms has amplified the strategic importance of coordinated pricing and inventory management decisions among competing retailers. Motivated by practices on leading e-commerce platforms, we analyze a sequential duopolistic newsvendor game where retailers first publicly set prices and subsequently make private inventory decisions under demand uncertainty. Our theory predicts that higher profit margins and demand uncertainty intensify price competition, while optimal inventory responses to demand uncertainty are shaped by profit margins. Laboratory evidence, however, reveals that participants are generally reluctant to compete on price, frequently coordinating on salient focal (reserve) prices, particularly in low-margin settings, and show little sensitivity to demand uncertainty in pricing. On the inventory side, participants' order quantities are largely insensitive to chosen prices and continue to exhibit well-documented Pull-to-Center biases. These findings reveal a disconnect between pricing and inventory decisions under competition and highlight the importance of accounting for persistent behavioral tendencies in retail operations.

---

## 164. Efficient and Scalable Monocular Human-Object Interaction Motion Reconstruction

**论文链接:** [http://arxiv.org/abs/2512.00960v1](http://arxiv.org/abs/2512.00960v1)

**作者:** Boran Wen, Ye Lu, Keyan Wan, Sirui Wang, Jiahong Zhou, Junxuan Liang, Xinpeng Liu, Bang Xiao, Dingbang Huang, Ruiyang Liu, Yong-Lu Li

**发布时间:** 2025-11-30

### GPT解析

### 总结

本文提出了一种从互联网视频中提取4D人-物交互数据的新方法，并构建了一个大规模数据集，为通用机器人学习提供了新的数据源。

### 背景

通用机器人需要从多样化、大规模的人-物交互中学习以在真实世界中稳健运行，但单目互联网视频中准确且可扩展地提取4D交互数据仍是一个重大挑战。

### 目的

解决从自然视频中准确且可扩展地提取4D人-物交互数据的挑战。

### 方法

提出4DHOISolver优化框架，利用稀疏的、有人类参与的接触点注释约束4D HOI重建问题，并构建了包含144种物体类型和103种动作的Open4DHOI数据集。

### 主要发现

通过基于RL的代理证明了重建的有效性，但现有3D基础模型在自动预测精确人-物接触对应关系方面仍有不足。

### 结论

人类参与策略对于解决4D HOI重建问题是必要的，同时这一领域仍存在开放挑战。数据和代码将公开。

### 翻译

通用机器人必须从多样化、大规模的人-物交互中学习，才能在真实世界中稳健运行。单目互联网视频提供了近乎无限且易于获取的数据源，捕捉了人类活动、物体和环境的无与伦比的多样性。然而，从这些自然视频中准确且可扩展地提取4D交互数据仍然是一个重大且未解决的挑战。因此，在这项工作中，我们介绍了4DHOISolver，这是一个新颖且高效的优化框架，它通过利用稀疏的、有人类参与的接触点注释来约束不适定的4D HOI重建问题，同时保持高时空相干性和物理合理性。利用这一框架，我们引入了Open4DHOI，一个新的4D HOI数据集，包含144种物体类型和103种动作。此外，我们通过使基于RL的代理能够模仿恢复的运动，证明了我们重建的有效性。然而，对现有3D基础模型的全面基准测试表明，自动预测精确的人-物接触对应关系仍然是一个未解决的问题，这强调了我们的有人类参与策略的必要性，同时也向社区提出了一个开放的挑战。数据和代码将在https://wenboran2002.github.io/open4dhoi/上公开。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决如何从单目视频中高效、可扩展地重建高质量的人-物交互4D运动数据的问题。这个问题很重要，因为通用机器人需要从多样化、大规模的人-物交互数据中学习才能在真实世界中稳健操作，而现有的多传感器系统成本高昂无法大规模采集数据，从单目图像重建的方法在视频序列上应用时又变得昂贵且耗时，难以保证跨帧时空一致性。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者认识到现有多传感器系统成本高昂且无法大规模采集数据，而单目图像重建方法在视频序列上应用时效率低下且难以保证时空一致性。作者的核心想法是用轻量级的稀疏接触点注释代替昂贵密集的标注，并设计了两阶段优化框架：第一阶段使用最小二乘匹配和逆运动学进行快速几何对齐，第二阶段基于梯度优化改进物理合理性。作者借鉴了现有的3D重建工具（如GVHMR、HAMER、TRELLIS）和图像修复技术，以及点匹配方法来优化重建过程。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是用轻量级的稀疏接触点注释代替密集标注，通过两阶段优化框架实现高效且高质量的重建，确保时空一致性和物理合理性。整体流程包括：1)数据收集（从在线平台或手机录制视频）；2)4D重建（跟踪掩码、修复遮挡区域、重建人体和物体）；3)标注接触点对；4)4DHOISolver优化（快速几何对齐和物理合理性优化）；5)构建Open4DHOI数据集；6)通过强化学习代理评估重建效果。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)提出4DHOISolver优化框架，利用稀疏接触点注释约束重建；2)构建大规模Open4DHOI数据集，包含144种物体类型和103种动作；3)开发基于接触的奖励函数用于稳健的交互运动模仿学习；4)揭示现有3D模型在预测人-物接触对应关系上的局限性。相比之前工作，本文方法成本更低、可扩展性更强，能保证跨帧时空一致性，标注成本大大降低，数据集规模更大更多样化，并证明了重建数据可用于下游任务如机器人学习。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '这篇论文提出了一种高效、可扩展的单目人-物交互运动重建方法，通过稀疏接触点注释和两阶段优化框架，构建了大规模、多样化的Open4DHOI数据集，并证明了重建数据可用于下游任务如机器人学习。'}


### 论文摘要

Generalized robots must learn from diverse, large-scale human-object interactions (HOI) to operate robustly in the real world. Monocular internet videos offer a nearly limitless and readily available source of data, capturing an unparalleled diversity of human activities, objects, and environments. However, accurately and scalably extracting 4D interaction data from these in-the-wild videos remains a significant and unsolved challenge. Thus, in this work, we introduce 4DHOISolver, a novel and efficient optimization framework that constrains the ill-posed 4D HOI reconstruction problem by leveraging sparse, human-in-the-loop contact point annotations, while maintaining high spatio-temporal coherence and physical plausibility. Leveraging this framework, we introduce Open4DHOI, a new large-scale 4D HOI dataset featuring a diverse catalog of 144 object types and 103 actions. Furthermore, we demonstrate the effectiveness of our reconstructions by enabling an RL-based agent to imitate the recovered motions. However, a comprehensive benchmark of existing 3D foundation models indicates that automatically predicting precise human-object contact correspondences remains an unsolved problem, underscoring the immediate necessity of our human-in-the-loop strategy while posing an open challenge to the community. Data and code will be publicly available at https://wenboran2002.github.io/open4dhoi/

---

## 165. Robust Probabilistic Load Forecasting for a Single Household: A Comparative Study from SARIMA to Transformers on the REFIT Dataset

**论文链接:** [http://arxiv.org/abs/2512.00856v1](http://arxiv.org/abs/2512.00856v1)

**作者:** Midhun Manoj

**发布时间:** 2025-11-30

**备注:** 12 pages, 8 figures, 1 table. This work includes a rigorous comparative study of imputation methods and presents results submitted to PAKDD 2026. Source code and analysis notebooks are available on GitHub: [https://github.com/middhun-31/Robust-Probabilistic-Load-Forecasting-for-a-Single-Household]

### GPT解析

### 总结

该研究针对具有大量结构数据缺口的REFIT家庭数据集，通过季节性插补方法解决数据缺失问题，并评估从经典到深度学习的一系列预测模型，发现时间融合变换器(TFT)模型在点预测准确率和预测区间质量方面表现最佳。

### 背景

概率预测对现代风险管理至关重要，能够帮助决策者量化关键系统中的不确定性。REFIT家庭数据集因其波动性和大量结构数据缺口而具有挑战性。

### 目的

解决具有大量结构数据缺口的复杂数据集的概率预测挑战，选择合适的季节性插补方法，并评估不同预测模型的性能表现。

### 方法

进行严格的比较实验选择季节性插补方法；系统评估模型层次，包括经典基线模型(SARIMA, Prophet)、机器学习模型(XGBoost)和深度学习架构(LSTM)；使用RMSE等指标评估预测性能。

### 主要发现

季节性插补方法在保留数据底层分布方面优于线性插补；经典模型无法捕捉数据的非线性和regime-switching行为；LSTM提供了校准最好的概率预测；TFT实现了最佳点预测准确率(RMSE 481.94)并产生更安全、更谨慎的预测区间。

### 结论

时间融合变换器(TFT)是最佳全能模型，在点预测准确率和预测区间质量方面均表现优异，能有效捕捉极端波动性。

### 翻译

概率预测对现代风险管理至关重要，它使决策者能够量化关键系统中的不确定性。本文使用波动性的REFIT家庭数据集应对这一挑战，该数据集因存在大量结构数据缺口而变得复杂。我们首先通过进行严格的比较实验来选择季节性插补方法，证明了其在保留数据底层分布方面优于线性插补。然后我们系统地评估了一个模型层次，从经典基线模型(SARIMA, Prophet)到机器学习(XGBoost)和高级深度学习架构(LSTM)。我们的研究结果表明，经典模型无法捕捉数据的非线性、regime-switching行为。虽然LSTM提供了校准最好的概率预测，但时间融合变换器(TFT)成为最佳全能模型，实现了最好的点预测准确率(RMSE 481.94)，并产生了更安全、更谨慎的预测区间，有效捕捉了极端波动性。


### 论文摘要

Probabilistic forecasting is essential for modern risk management, allowing decision-makers to quantify uncertainty in critical systems. This paper tackles this challenge using the volatile REFIT household dataset, which is complicated by a large structural data gap. We first address this by conducting a rigorous comparative experiment to select a Seasonal Imputation method, demonstrating its superiority over linear interpolation in preserving the data's underlying distribution. We then systematically evaluate a hierarchy of models, progressing from classical baselines (SARIMA, Prophet) to machine learning (XGBoost) and advanced deep learning architectures (LSTM). Our findings reveal that classical models fail to capture the data's non-linear, regime-switching behavior. While the LSTM provided the most well-calibrated probabilistic forecast, the Temporal Fusion Transformer (TFT) emerged as the superior all-round model, achieving the best point forecast accuracy (RMSE 481.94) and producing safer, more cautious prediction intervals that effectively capture extreme volatility.

---

## 166. TrajDiff: End-to-end Autonomous Driving without Perception Annotation

**论文链接:** [http://arxiv.org/abs/2512.00723v1](http://arxiv.org/abs/2512.00723v1)

**作者:** Xingtai Gui, Jianbo Zhao, Wencheng Han, Jikai Wang, Jiahao Gong, Feiyang Tan, Cheng-zhong Xu, Jianbing Shen

**发布时间:** 2025-11-30

### GPT解析

### 总结

TrajDiff是一种面向轨迹的BEV条件扩散框架，实现了完全无需感知标注的端到端自动驾驶方法。

### 背景

端到端自动驾驶系统可直接从原始传感器输入生成驾驶策略，但依赖于辅助感知任务。由于手动感知标注成本高，开发无需感知标注的规划范式变得至关重要。

### 目的

提出一种完全无需感知标注的端到端自动驾驶生成方法，降低对昂贵标注数据的依赖。

### 方法

TrajDiff框架仅需原始传感器输入和未来轨迹，构建高斯BEV热图目标；设计面向轨迹的BEV编码器提取TrajBEV特征；引入TB-DiT利用自车状态信息和预测特征生成多样化合理轨迹，无需手工制作运动先验。

### 主要发现

TrajDiff在NAVSIM基准测试上实现87.5 PDMS，在无需标注方法中达到最先进性能；通过数据扩展提升至88.5 PDMS，可与先进感知方法相媲美。

### 结论

TrajDiff成功实现了无需感知标注的端到端自动驾驶，代码和模型将公开可用。

### 翻译

端到端自动驾驶系统直接从原始传感器输入生成驾驶策略。虽然这些系统可以提取有效的环境特征进行规划，但依赖于辅助感知任务，由于手动感知标注的高成本，开发无需感知标注的规划范式变得越来越重要。在这项工作中，我们提出了TrajDiff，一种面向轨迹的BEV条件扩散框架，为端到端自动驾驶建立了完全无需感知标注的生成方法。TrajDiff仅需原始传感器输入和未来轨迹，构建固有捕获驾驶模态的高斯BEV热图目标。我们设计了一种简单而有效的面向轨迹的BEV编码器，无需感知监督即可提取TrajBEV特征。此外，我们引入了面向轨迹的BEV扩散变换器(TB-DiT)，利用自车状态信息和预测的TrajBEV特征直接生成多样化且合理的轨迹，消除了对手工制作运动先验的需求。除了架构创新外，TrajDiff还能够在无需标注设置下探索数据扩展的好处。在NAVSIM基准测试上评估，TrajDiff实现了87.5 PDMS，在所有无需标注的方法中建立了最先进的性能。通过数据扩展，它进一步提升至88.5 PDMS，可与先进的基于感知的方法相媲美。我们的代码和模型将公开提供。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决端到端自动驾驶系统对感知标注的依赖问题。这一问题在现实中非常重要，因为人工标注自动驾驶数据成本极高（1000个场景需要超过7000小时标注工作），限制了数据规模和模型性能；同时依赖感知标注也增加了系统复杂度，违背了端到端方法直接优化规划性能的核心原则。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先认识到现有无标注方法依赖未来传感器帧和专用架构来预测环境特征，且未明确利用未来轨迹构建规划目标。因此，他们提出通过未来轨迹构建高斯BEV热力图目标实现自监督学习，设计轨迹导向的BEV编码器提取特征，并利用扩散模型处理规划的不确定性和多模态特性。该方法借鉴了扩散模型在图像生成和轨迹预测中的成功应用，以及BEV表示在车辆规划中的有效性，同时参考了Transfuser的编码器结构。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是利用未来轨迹构建高斯BEV热力图目标实现自监督学习，设计轨迹导向的BEV特征编码潜在驾驶模式，并通过扩散模型在无锚点情况下生成轨迹。整体流程：1)接收原始传感器输入和自车状态；2)通过轨迹导向的BEV编码器生成TrajBEV特征；3)使用TB-DiT模块将噪声轨迹去噪为清晰轨迹；4)结合高斯BEV热力图损失和轨迹扩散损失进行训练；5)使用DDIM采样器生成最终轨迹。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)完全无感知标注的端到端框架；2)轨迹导向的BEV扩散变换器(TB-DiT)；3)无标注训练策略带来的数据缩放优势；4)在NAVSIM上达到87.5 PDMS的最先进性能。相比之前工作，该方法不依赖未来传感器帧和专用架构预测环境特征，明确利用未来轨迹构建规划目标，无需预定义轨迹锚点，直接在连续轨迹上执行去噪操作，且系统设计更简洁。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'TrajDiff通过轨迹导向的BEV条件扩散框架，实现了无需感知标注的端到端自动驾驶，仅利用原始传感器输入和未来轨迹就能生成多样化的合理驾驶轨迹，并通过数据缩放进一步提升了规划性能。'}


### 论文摘要

End-to-end autonomous driving systems directly generate driving policies from raw sensor inputs. While these systems can extract effective environmental features for planning, relying on auxiliary perception tasks, developing perception annotation-free planning paradigms has become increasingly critical due to the high cost of manual perception annotation. In this work, we propose TrajDiff, a Trajectory-oriented BEV Conditioned Diffusion framework that establishes a fully perception annotation-free generative method for end-to-end autonomous driving. TrajDiff requires only raw sensor inputs and future trajectory, constructing Gaussian BEV heatmap targets that inherently capture driving modalities. We design a simple yet effective trajectory-oriented BEV encoder to extract the TrajBEV feature without perceptual supervision. Furthermore, we introduce Trajectory-oriented BEV Diffusion Transformer (TB-DiT), which leverages ego-state information and the predicted TrajBEV features to directly generate diverse yet plausible trajectories, eliminating the need for handcrafted motion priors. Beyond architectural innovations, TrajDiff enables exploration of data scaling benefits in the annotation-free setting. Evaluated on the NAVSIM benchmark, TrajDiff achieves 87.5 PDMS, establishing state-of-the-art performance among all annotation-free methods. With data scaling, it further improves to 88.5 PDMS, which is comparable to advanced perception-based approaches. Our code and model will be made publicly available.

---

## 167. Evolution of Flare Ribbon Bead-like Structures in a Solar Flare

**论文链接:** [http://arxiv.org/abs/2512.00710v1](http://arxiv.org/abs/2512.00710v1)

**作者:** Ryan J. French, Maria D. Kazachenko, David Berghmans, Elke D'Huys, Marie Dominique, Ritesh Patel, Dana-Camelia Talpeanu, Cole A. Tamburri, Rahul Yadav

**发布时间:** 2025-11-30

**备注:** 11 pages, 5 figures, accepted to ApJL

### GPT解析

### 总结

利用太阳轨道器极紫外成像仪对C9.9级太阳耀斑进行高分辨率观测，发现耀斑带中存在小尺度动态结构，其行为与撕裂模式不稳定性理论预测一致。

### 背景

使用太阳轨道器极紫外成像仪（EUI）进行快速节奏和高分辨率观测，研究太阳耀斑期间耀斑带中的精细结构。

### 目的

研究太阳耀斑带中小尺度块状/珠状核心结构的动态行为及其形成机制。

### 方法

利用EUI高分辨率极紫外成像仪（HRIEUV）进行短时间曝光观测，分析耀斑带结构的平面视天速度和分离度演化，并进行快速傅里叶变换分析不同空间尺度的功率谱。

### 主要发现

1) 在耀斑带钩状末端发现小尺度块状/珠状核心结构；2) 这些珠状结构动态变化，空间分辨率低至420-840公里；3) 耀斑带中同时存在多种过程：类准周期脉动增亮、缓慢来回锯齿运动、快速表观运动（600+公里/秒）和静止块状结构；4) 带状珠形成时具有1.7-1.9兆米的关键空间分离度。

### 结论

观测结果与撕裂模式不稳定性的理论预测一致，为理解太阳耀斑的精细物理过程提供了重要依据。

### 翻译

我们展示了来自太阳轨道器极紫外成像仪（EUI）的快速节奏和高分辨率耀斑带观测。利用EUI高分辨率极紫外成像仪（HRIEUV）的短时间曝光观测，我们在C9.9级太阳耀斑的脉冲相期间，发现小尺度的块状/珠状核心结构在耀斑带末端的钩状区域内传播。这些珠状结构是动态的，具有低至约420-840公里（3-6像素）的良好分辨空间距离——这低于全日面太阳成像仪的可观测极限。我们分析了耀斑带结构平面视天速度和分离度的演化，发现在耀斑带中同时存在多种过程。这些过程包括类准周期脉动（QPP）的增亮、沿带状的缓慢来回锯齿运动、沿带状的快速表观运动（600+公里/秒）以及静止的块状结构。最后，我们进行了快速傅里叶变换分析，并分析了不同空间尺度下耀斑带功率谱指数增长的开始时间。我们的分析显示，带状珠形成时具有1.7-1.9兆米的关键空间分离度，随后在更大和更小的空间尺度上发展成更复杂的结构。这一观测结果与撕裂模式不稳定性的预测一致。


### 论文摘要

We present fast cadence and high resolution observations of flare ribbons from the Solar Orbiter Extreme Ultraviolet Imager (EUI). Utilizing the short-exposure observations from the EUI High Resolution Imager in EUV (HRIEUV), we find small-scale blob/bead-like kernel structures propagating within a hook at the end of a flare ribbon, during the impulsive phase of a C9.9-class solar flare. These bead structures are dynamic, with well-resolved spatial separations as low as ~420-840 kilometers (3-6 pixels) - below the observable limit of full-disk solar imagers. We analyze the evolution of the plane-of-sky apparent velocity and separation of the flare ribbon structures, finding evidence for multiple processes occurring simultaneously within the flare ribbon. These processes include - quasi-periodic pulsation (QPP)-like brightenings, slow back-and-forth zig-zag motions along the ribbon, rapid apparent motions along the ribbon (600+ km/s), and stationary blob-like structures. Finally, we conduct Fast Fourier Transform analysis and analyze the start times of exponential growth in the power spectrum at different spatial scales across the flare ribbon. Our analysis reveals that the ribbon beads form with a key spatial separation of 1.7-1.9 Mm, before developing into more complex structures at progressively larger and smaller spatial scales. This observation is consistent with predictions of the tearing mode instability.

---

## 168. Learning Dexterous Manipulation Skills from Imperfect Simulations

**论文链接:** [http://arxiv.org/abs/2512.02011v1](http://arxiv.org/abs/2512.02011v1)

**作者:** Elvis Hsieh, Wen-Han Hsieh, Yen-Jen Wang, Toru Lin, Jitendra Malik, Koushil Sreenath, Haozhi Qi

**发布时间:** 2025-12-01

### GPT解析

### 总结

该研究提出了一种名为\ours的仿真到现实框架，解决了灵巧操作中复杂接触动力学和触觉反馈模拟的困难，通过三阶段方法实现了在螺母-螺栓紧固和螺丝刀操作任务上的高效性能。

### 背景

强化学习和仿真到现实转换在灵巧操作方面取得了显著进展，但仍受限于复杂接触动力学和多重感官信号（特别是触觉反馈）的模拟困难。

### 目的

提出一个解决这些限制的仿真到现实框架，并在多指手上展示其在螺母-螺栓紧固和螺丝刀操作任务上的有效性。

### 方法

框架包含三个阶段：1)在仿真中使用简化物体模型训练强化学习策略；2)将学习策略作为遥操作系统中的技能原语收集真实世界演示；3)训练整合触觉传感的行为克隆策略并推广到不同几何形状的物体。

### 主要发现

与直接仿真到现实转换相比，在两项任务中表现出高的任务进展比率，即使在未见过的物体形状和外部扰动下也能保持稳健性能。

### 结论

通过结合仿真训练、真实世界演示和行为克隆，该框架成功解决了复杂接触动力学和触觉反馈模拟的挑战，实现了有效的仿真到现实转换。

### 翻译

强化学习和仿真到现实转换在灵巧操作方面取得了显著进展。然而，由于难以模拟复杂的接触动力学和多重感官信号，特别是触觉反馈，进展仍然受到限制。在这项工作中，我们提出了\ours，一个解决这些限制的仿真到现实框架，并在多指手上展示了其在螺母-螺栓紧固和螺丝刀操作任务上的有效性。该框架包含三个阶段。首先，我们在仿真中使用简化的物体模型训练强化学习策略，这导致正确的手指步态的出现。然后，我们将学习到的策略作为遥操作系统中的技能原语，收集包含触觉和本体感觉信息的真实世界演示。最后，我们训练了一个整合触觉传感的行为克隆策略，并证明它可以推广到具有不同几何形状的螺母和螺丝刀。两项任务中的实验表明，与直接仿真到现实转换相比，具有高的任务进展比率，即使在未见过的物体形状和外部扰动下也能保持稳健性能。视频和代码可在https://dexscrew.github.io获取。


### 论文摘要

Reinforcement learning and sim-to-real transfer have made significant progress in dexterous manipulation. However, progress remains limited by the difficulty of simulating complex contact dynamics and multisensory signals, especially tactile feedback. In this work, we propose \ours, a sim-to-real framework that addresses these limitations and demonstrates its effectiveness on nut-bolt fastening and screwdriving with multi-fingered hands. The framework has three stages. First, we train reinforcement learning policies in simulation using simplified object models that lead to the emergence of correct finger gaits. We then use the learned policy as a skill primitive within a teleoperation system to collect real-world demonstrations that contain tactile and proprioceptive information. Finally, we train a behavior cloning policy that incorporates tactile sensing and show that it generalizes to nuts and screwdrivers with diverse geometries. Experiments across both tasks show high task progress ratios compared to direct sim-to-real transfer and robust performance even on unseen object shapes and under external perturbations. Videos and code are available on https://dexscrew.github.io.

---

## 169. 论文ID: 2512.02004v1

**错误:** 无法读取论文信息 - [Errno 2] No such file or directory: 'storage/result/2512.02004v1.json'

---

## 170. How Does RL Post-training Induce Skill Composition? A Case Study on Countdown

**论文链接:** [http://arxiv.org/abs/2512.01775v1](http://arxiv.org/abs/2512.01775v1)

**作者:** Simon Park, Simran Kaur, Sanjeev Arora

**发布时间:** 2025-12-01

### GPT解析

### 总结

研究强化学习(RL)后训练对大型语言模型组合泛化的影响，专注于Countdown任务，通过分析表达式树揭示技能合成和转移机制。

### 背景

强化学习成功增强了大型语言模型的推理能力，但其促进组合泛化的作用常常与长度泛化相混淆。

### 目的

研究RL后训练在技能合成方面教给了模型什么，以及合成的结构如何影响技能转移。

### 方法

专注于Countdown任务，将模型解决方案分析为表达式树，其中每个子树对应一个可重用的子任务(视为技能)，通过跟踪训练过程中树形状及其成功率的变化进行研究。

### 主要发现

(i) 分布外泛化到更大的n和未见过的树形状，表明子任务的可重用组合；(ii) 存在结构相关的可学习性层次结构，模型先掌握浅层平衡树，然后是深层不平衡树，但在右重结构上存在持久脆弱性。

### 结论

该诊断揭示了学习的内容、顺序以及泛化失败的地方，阐明了仅通过RL后训练如何诱导标准指标未能揭示的分布外泛化。

### 翻译

虽然强化学习(RL)成功增强了大型语言模型的推理能力，但其在促进组合泛化(从已知组件合成新技能的能力)方面常常与简单的长度泛化相混淆。为此，我们研究了RL后训练关于技能合成的内容以及合成结构如何影响技能转移。我们专注于Countdown任务(给定n个数字和一个目标，构造一个表达式求值为目标)，并将模型解决方案分析为表达式树，其中每个子树对应一个可重用的子任务，可以视为一个'技能'。通过跟踪训练过程中树形状及其成功率的变化，我们发现：(i) 分布外(OOD)泛化到更大的n和未见过的树形状，表明子任务的可重用组合；(ii) 存在一种结构相关的可学习性层次结构——模型先掌握浅层平衡树(子任务间工作负载平衡)，然后是深层不平衡树，但在右重结构上存在持久脆弱性(即使组合深度与某些左重结构相同)。我们的诊断揭示了学习的内容、顺序以及泛化失败的地方，阐明了仅通过RL后训练如何诱导标准指标(如pass@k)未能揭示的分布外泛化。


### 论文摘要

While reinforcement learning (RL) successfully enhances reasoning in large language models, its role in fostering compositional generalization (the ability to synthesize novel skills from known components) is often conflated with mere length generalization. To this end, we study what RL post-training teaches about skill composition and how the structure of the composition affects the skill transfer. We focus on the Countdown task (given n numbers and a target, form an expression that evaluates to the target) and analyze model solutions as expression trees, where each subtree corresponds to a reusable subtask and thus can be viewed as a ``skill.'' Tracking tree shapes and their success rates over training, we find: (i) out-of-distribution (OOD) generalization to larger n and to unseen tree shapes, indicating compositional reuse of subtasks; (ii) a structure-dependent hierarchy of learnability -- models master shallow balanced trees (workload is balanced between subtasks) before deep unbalanced ones, with persistent fragility on right-heavy structures (even when the composition depth is the same as some left-heavy structures). Our diagnostic reveals what is learned, in what order, and where generalization fails, clarifying how RL-only post-training induces OOD generalization beyond what standard metrics such as pass@k reveal.

---

## 171. RoMe: Row Granularity Access Memory System for Large Language Models

**论文链接:** [http://arxiv.org/abs/2512.01541v1](http://arxiv.org/abs/2512.01541v1)

**作者:** Hwayong Nam, Seungmin Baek, Jumin Kim, Michael Jaemin Kim, Jung Ho Ahn

**发布时间:** 2025-12-01

**备注:** 15 pages, 14 figures, accepted at HPCA 2026

### GPT解析

### 总结

RoMe是一种创新的内存系统设计，通过改变内存访问粒度和简化接口，能够在不显著增加硬件开销的情况下提高带宽，特别适合大型语言模型等连续数据块处理的工作负载。

### 背景

现代基于HBM的内存系统在保留缓存行粒度访问的同时已经发展了多代。为了保持这种细粒度，引入了bank groups和pseudo channels等结构，这些结构扩展了时序参数和控制开销，显著增加了内存控制器调度复杂性。

### 目的

解决传统HBM内存系统在处理大型语言模型工作负载时的效率低下问题，通过简化内存调度来提高带宽。

### 方法

RoMe以行粒度访问DRAM，并从内存接口中移除了列、bank groups和pseudo channels。这种设计简化了内存调度，从而每个通道需要的引脚更少。释放的引脚被聚合以形成额外的通道，仅用最少的额外引脚就使整体带宽增加了12.5%。

### 主要发现

RoMe能够显著简化内存调度逻辑，特别适合代表性的LLM工作负载，同时实现更高的带宽和最小的硬件开销。

### 结论

RoMe为下一代基于HBM的内存系统提供了一种替代方法，通过改变内存访问粒度和简化接口，能够在不显著增加硬件开销的情况下提高带宽，特别适合大型语言模型等连续数据块处理的工作负载。

### 翻译

现代基于HBM的内存系统在保留缓存行粒度访问的同时已经发展了多代。为了保持这种细粒度，引入了bank groups和pseudo channels等结构，这些结构扩展了时序参数和控制开销，显著增加了内存控制器调度复杂性。大型语言模型(LLMs)现在主导深度学习工作负载，每次操作会流式传输从几千字节到几兆字节的连续数据块。在传统的基于HBM的内存系统中，这些传输被分割成数百个32字节的缓存行事务，这迫使内存控制器采用不必要的复杂调度，导致效率低下。为了解决这个问题，我们提出了RoMe。RoMe以行粒度访问DRAM，并从内存接口中移除了列、bank groups和pseudo channels。这种设计简化了内存调度，从而每个通道需要的引脚更少。释放的引脚被聚合以形成额外的通道，仅用最少的额外引脚就使整体带宽增加了12.5%。RoMe展示了如何针对代表性的LLM工作负载显著简化内存调度逻辑，并为下一代基于HBM的内存系统提供了一种替代方法，以实现更高的带宽和最小的硬件开销。


### 论文摘要

Modern HBM-based memory systems have evolved over generations while retaining cache line granularity accesses. Preserving this fine granularity necessitated the introduction of bank groups and pseudo channels. These structures expand timing parameters and control overhead, significantly increasing memory controller scheduling complexity. Large language models (LLMs) now dominate deep learning workloads, streaming contiguous data blocks ranging from several kilobytes to megabytes per operation. In a conventional HBM-based memory system, these transfers are fragmented into hundreds of 32B cache line transactions. This forces the memory controller to employ unnecessarily intricate scheduling, leading to growing inefficiency.   To address this problem, we propose RoMe. RoMe accesses DRAM at row granularity and removes columns, bank groups, and pseudo channels from the memory interface. This design simplifies memory scheduling, thereby requiring fewer pins per channel. The freed pins are aggregated to form additional channels, increasing overall bandwidth by 12.5% with minimal extra pins. RoMe demonstrates how memory scheduling logic can be significantly simplified for representative LLM workloads, and presents an alternative approach for next-generation HBM-based memory systems achieving increased bandwidth with minimal hardware overhead.

---

## 172. PromptBridge: Cross-Model Prompt Transfer for Large Language Models

**论文链接:** [http://arxiv.org/abs/2512.01420v1](http://arxiv.org/abs/2512.01420v1)

**作者:** Yaxuan Wang, Quan Liu, Zhenting Wang, Zichao Li, Wei Wei, Yang Liu, Yujia Bao

**发布时间:** 2025-12-01

### GPT解析

### 总结

本文提出PromptBridge框架，用于解决大型语言模型间的提示词迁移问题。当在不同模型(如GPT、Claude、Llama)间切换时，针对一个模型优化的提示词在另一模型上效果显著下降，这种现象称为'Model Drifting'。PromptBridge能在不进行昂贵重新优化的情况下实现跨模型提示词迁移，保持提示词有效性。

### 背景

大型语言模型支撑代码生成、数学推理和基于代理的工作流程等应用。系统通过商业API或开源部署访问LLMs，而模型格局快速演变。频繁的模型切换由能力、成本、部署限制和隐私考虑驱动。然而，提示词对模型高度敏感：针对一个模型设计的提示词在另一个模型上重用，通常导致性能显著下降。

### 目的

解决'Model Drifting'现象，开发无需训练的框架，在模型切换时保持提示词有效性，实现跨模型提示词迁移，避免昂贵的任务或模型特定重新优化。

### 方法

PromptBridge框架使用少量对齐任务进行校准。首先应用'Model-Adaptive Reflective Prompt Evolution (MAP-RPE)'，通过迭代反思性完善和定量评估获取任务和模型特定的最优提示词。然后学习源模型和目标模型提示词间的映射关系。测试时，给定源模型提示词，直接生成目标模型的优化提示词。

### 主要发现

通过多样化LLM配置的实证分析表明，'Model Drifting'现象既普遍又严重。在单代理和多代理设置中，PromptBridge能提高下游准确性，同时减少迁移工作量。

### 结论

PromptBridge有效解决了大型语言模型间的提示词迁移问题，使用户能够在不进行昂贵重新优化的情况下，将针对一个模型的提示词迁移到另一个模型，保持提示词有效性，提高下游任务准确性。

### 翻译

大型语言模型支撑着代码生成、数学推理和基于代理的工作流程等应用。在实践中，系统通过商业API或开源部署访问LLMs，而模型格局(如GPT、Claude、Llama)正在迅速发展。这种快速演变迫使系统基于能力、成本、部署限制和隐私考虑频繁切换模型。然而，提示词对模型高度敏感：针对一个模型设计的提示词在另一个模型上重用，通常会导致性能显著低于针对目标模型优化的提示词。我们将这种现象称为'Model Drifting'。通过对多样化LLM配置进行广泛的实证分析，我们表明Model Drifting既普遍又严重。为应对这一挑战，我们引入了PromptBridge，这是一个无需训练的框架，能够在模型切换时保持提示词的有效性，实现跨模型提示词迁移，而无需昂贵的任务或模型特定重新优化。PromptBridge只需要一组小的对齐任务进行校准。它首先应用'Model-Adaptive Reflective Prompt Evolution (MAP-RPE)'，通过迭代式反思性完善和定量评估获取任务和模型特定的最优提示词。使用为源模型和目标模型生成的校准提示词对，PromptBridge学习一个跨模型提示词映射。在测试时，即对于未见过的任务，给定一个源模型提示词，该映射直接为目标模型生成一个优化后的提示词。在单代理和多代理设置中的实验表明，PromptBridge在提高下游准确性的同时减少了迁移工作量。代码将很快可用。


### 论文摘要

Large language models (LLMs) underpin applications in code generation, mathematical reasoning, and agent-based workflows. In practice, systems access LLMs via commercial APIs or open-source deployments, and the model landscape (e.g., GPT, Claude, Llama) evolves rapidly. This rapid evolution forces frequent model switches driven by capability, cost, deployment constraints, and privacy. Yet prompts are highly model-sensitive: reusing a prompt engineered for one model on another often yields substantially worse performance than a prompt optimized for the target model. We term this phenomenon Model Drifting. Through extensive empirical analysis across diverse LLM configurations, we show that model drifting is both common and severe. To address this challenge, we introduce PromptBridge, a training-free framework that preserves prompt effectiveness under model switches, enabling cross-model prompt transfer without costly per-task or per-model re-optimization. PromptBridge requires only a small set of alignment tasks for calibration. It first applies Model-Adaptive Reflective Prompt Evolution (MAP-RPE) to obtain task- and model-specific optimal prompts via iterative reflective refinement and quantitative evaluation. Using the resulting calibrated prompt pairs for the source and target models, PromptBridge learns a cross-model prompt mapping. At test time, i.e., for an unseen task, given a source-model prompt, this mapping directly produces an optimized prompt for the target model. Experiments in single-agent and multi-agent settings show that PromptBridge consistently improves downstream accuracy while reducing migration effort. The code will be available soon.

---

## 173. On Global Applicability and Location Transferability of Generative Deep Learning Models for Precipitation Downscaling

**论文链接:** [http://arxiv.org/abs/2512.01400v1](http://arxiv.org/abs/2512.01400v1)

**作者:** Paula Harder, Christian Lessig, Matthew Chantry, Francis Pelletier, David Rolnick

**发布时间:** 2025-12-01

### GPT解析

### 总结

深度学习为气候和天气预报统计降尺度提供有前景能力，生成式方法在捕捉精细降水模式方面特别成功，但现有模型泛化能力未知。

### 背景

深度学习在气候和天气预报统计降尺度方面显示出潜力，生成式方法尤其擅长捕捉精细尺度降水模式。然而，大多数现有模型是区域特定的，它们对未见过的地理区域的泛化能力尚未得到充分探索。

### 目的

评估生成式降尺度模型在不同地理区域的泛化性能。

### 方法

使用全球框架，采用ERA5再分析数据作为预测因子，使用0.1°分辨率的IMERG降水估计作为目标。通过基于位置的层次数据分割，系统评估模型在全球15个区域的性能。

### 主要发现

摘要中未明确提及具体的主要发现

### 结论

摘要中未明确提及具体结论

### 翻译

深度学习为气候和天气预报的统计降尺度提供了有前景的能力，生成式方法在捕捉精细尺度降水模式方面显示出特别成功。然而，大多数现有模型是区域特定的，它们对未见过的地理区域的泛化能力尚未得到充分探索。在本研究中，我们评估了生成式降尺度模型在不同区域的泛化性能。使用全球框架，我们采用ERA5再分析数据作为预测因子，使用0.1°分辨率的IMERG降水估计作为目标。基于位置的层次数据分割使我们能够系统评估模型在全球15个区域的性能。


### 论文摘要

Deep learning offers promising capabilities for the statistical downscaling of climate and weather forecasts, with generative approaches showing particular success in capturing fine-scale precipitation patterns. However, most existing models are region-specific, and their ability to generalize to unseen geographic areas remains largely unexplored. In this study, we evaluate the generalization performance of generative downscaling models across diverse regions. Using a global framework, we employ ERA5 reanalysis data as predictors and IMERG precipitation estimates at $0.1^\circ$ resolution as targets. A hierarchical location-based data split enables a systematic assessment of model performance across 15 regions around the world.

---

## 174. 论文ID: 2512.01358v1

**错误:** 无法读取论文信息 - [Errno 2] No such file or directory: 'storage/result/2512.01358v1.json'

---

## 175. Discovering Self-Protective Falling Policy for Humanoid Robot via Deep Reinforcement Learning

**论文链接:** [http://arxiv.org/abs/2512.01336v1](http://arxiv.org/abs/2512.01336v1)

**作者:** Diyuan Shi, Shangke Lyu, Donglin Wang

**发布时间:** 2025-12-01

### GPT解析

### 总结

该研究针对人形机器人跌倒问题，提出了一种基于深度强化学习和课程学习的方法，训练人形智能体探索跌倒保护行为，发现通过形成'三角形'结构可显著减少跌倒损害，并在真实世界成功应用。

### 背景

人形机器人近年来受到广泛关注，但由于其形态、动力学特性和控制策略限制，相比四足或轮式机器人更易跌倒。人形机器人重量大、质心高、自由度高，跌倒时可能对自身和周围物体造成严重硬件损坏。

### 目的

开发适合人形机器人自身特性的跌倒保护策略，减少跌倒造成的损害，避免硬件损坏。

### 方法

使用大规模深度强化学习和课程学习技术，通过精心设计的奖励函数和领域多样化课程，训练人形智能体探索跌倒保护行为。

### 主要发现

人形机器人通过形成'三角形'结构，可以利用其刚性材质的身体显著减少跌倒造成的损害。

### 结论

所提出的方法能够有效训练人形机器人进行跌倒保护，并通过全面的指标和实验验证了其性能，成功将训练结果转移到真实世界平台。

### 翻译

人形机器人在近年来受到了显著的研究关注和进展。尽管取得了许多成功，但由于其形态、动力学特性和控制策略的局限性，与其他形态如四足或轮式机器人相比，人形机器人更容易跌倒。其大重量、高质心和高度自由度会在失控跌倒时对自身和周围物体造成严重的硬件损坏。该领域现有研究大多基于控制方法，难以应对多样化的跌倒场景，并可能引入不合适的人类先验。另一方面，大规模深度强化学习和课程学习可用于激励人形智能体发现适合其自身特性的跌倒保护策略。在本工作中，通过精心设计的奖励函数和领域多样化课程，我们成功训练人形智能体探索跌倒保护行为，并发现通过形成'三角形'结构，可以利用其刚性材质的身体显著减少跌倒造成的损害。通过全面的指标和实验，我们量化了其性能并与其它方法进行了比较，可视化了其跌倒行为，并成功将其转移到真实世界平台。


### 论文摘要

Humanoid robots have received significant research interests and advancements in recent years. Despite many successes, due to their morphology, dynamics and limitation of control policy, humanoid robots are prone to fall as compared to other embodiments like quadruped or wheeled robots. And its large weight, tall Center of Mass, high Degree-of-Freedom would cause serious hardware damages when falling uncontrolled, to both itself and surrounding objects. Existing researches in this field mostly focus on using control based methods that struggle to cater diverse falling scenarios and may introduce unsuitable human prior. On the other hand, large-scale Deep Reinforcement Learning and Curriculum Learning could be employed to incentivize humanoid agent discovering falling protection policy that fits its own nature and property. In this work, with carefully designed reward functions and domain diversification curriculum, we successfully train humanoid agent to explore falling protection behaviors and discover that by forming a `triangle' structure, the falling damages could be significantly reduced with its rigid-material body. With comprehensive metrics and experiments, we quantify its performance with comparison to other methods, visualize its falling behaviors and successfully transfer it to real world platform.

---

## 176. FOD-S2R: A FOD Dataset for Sim2Real Transfer Learning based Object Detection

**论文链接:** [http://arxiv.org/abs/2512.01315v1](http://arxiv.org/abs/2512.01315v1)

**作者:** Ashish Vashist, Qiranul Saadiyean, Suresh Sundaram, Chandra Sekhar Seelamantula

**发布时间:** 2025-12-01

**备注:** 8 pages, 11 figures

### GPT解析

### 总结

这篇论文介绍了一个名为FOD-S2R的新数据集，用于检测飞机燃料箱内的外来碎片(FOD)。研究展示了合成数据如何提高检测模型的准确性和泛化能力，为航空维护中的自动化FOD检测系统提供了基础。

### 背景

飞机燃料箱内的外来碎片(FOD)构成严重的安全隐患，包括燃料污染、系统故障和维护成本增加。然而，目前缺乏专门针对燃料箱这类复杂封闭环境的数据集。

### 目的

创建一个专门用于评估在封闭结构中使用合成数据提高真实世界FOD检测效果的数据集，以弥补现有数据集只关注外部或开放环境的不足。

### 方法

构建了一个名为FOD-S2R的数据集，包含3,114张在受控燃料箱复制品中拍摄的高清真实图像，以及3,137张使用Unreal Engine生成的合成图像。研究对几种最先进的物体检测模型进行了基准测试，评估合成数据对检测性能的影响。

### 主要发现

引入合成数据可以提高检测模型的准确性和对真实条件的泛化能力，有效缩小了模拟到现实(Sim2Real)的差距。

### 结论

合成数据在增强模型性能和缩小Sim2Real差距方面是有效的，为开发航空维护中的自动化FOD检测系统提供了有价值的基础。

### 翻译

飞机燃料箱内的外来碎片(FOD)构成严重的安全隐患，包括燃料污染、系统故障和维护成本增加。尽管这些风险的严重性，但在燃料箱内部复杂封闭环境中，仍然缺乏专门的数据集。为了弥补这一差距，我们提出了一个名为FOD-S2R的新数据集，由模拟飞机燃料箱内FOD的真实和合成图像组成。与关注外部或露天环境的数据集不同，我们的数据集首次系统性地评估了合成数据在增强真实世界中封闭结构内FOD检测性能方面的效果。真实世界子集包含在受控燃料箱复制品中拍摄的3,114张高分辨率高清图像，而合成子集则包括使用Unreal Engine生成的3,137张图像。该数据集包含各种视野、物体距离、光照条件、颜色和物体尺寸。先前的研究表明，合成数据可以减少对大量真实世界标注的依赖，并提高视觉模型的泛化能力。因此，我们对几种最先进的物体检测模型进行了基准测试，证明引入合成数据可以提高检测准确性和对真实条件的泛化能力。这些实验证明了合成数据在增强模型性能和缩小Sim2Real差距方面的有效性，为开发航空维护中的自动化FOD检测系统提供了宝贵的基础。


### 论文摘要

Foreign Object Debris (FOD) within aircraft fuel tanks presents critical safety hazards including fuel contamination, system malfunctions, and increased maintenance costs. Despite the severity of these risks, there is a notable lack of dedicated datasets for the complex, enclosed environments found inside fuel tanks. To bridge this gap, we present a novel dataset, FOD-S2R, composed of real and synthetic images of the FOD within a simulated aircraft fuel tank. Unlike existing datasets that focus on external or open-air environments, our dataset is the first to systematically evaluate the effectiveness of synthetic data in enhancing the real-world FOD detection performance in confined, closed structures. The real-world subset consists of 3,114 high-resolution HD images captured in a controlled fuel tank replica, while the synthetic subset includes 3,137 images generated using Unreal Engine. The dataset is composed of various Field of views (FOV), object distances, lighting conditions, color, and object size. Prior research has demonstrated that synthetic data can reduce reliance on extensive real-world annotations and improve the generalizability of vision models. Thus, we benchmark several state-of-the-art object detection models and demonstrate that introducing synthetic data improves the detection accuracy and generalization to real-world conditions. These experiments demonstrate the effectiveness of synthetic data in enhancing the model performance and narrowing the Sim2Real gap, providing a valuable foundation for developing automated FOD detection systems for aviation maintenance.

---

## 177. 论文ID: 2512.01219v1

**错误:** 无法读取论文信息 - [Errno 2] No such file or directory: 'storage/result/2512.01219v1.json'

---

## 178. How do trout regulate patterns of muscle contraction to optimize propulsive efficiency during steady swimming

**论文链接:** [http://arxiv.org/abs/2512.01218v1](http://arxiv.org/abs/2512.01218v1)

**作者:** Tao Li, Chunze Zhang, Weiwei Yao, Junzhao He, Ji Hou, Qin Zhou, Lu Zhang

**发布时间:** 2025-12-01

### GPT解析

### 总结

研究高效鱼类游泳可为生物力学、流体动力学和工程学提供见解。通过创建仿生数字鳟鱼模型，结合多体动力学、希尔型肌肉建模和高保真流固耦合算法，研究揭示了肌肉激活策略对游泳速度和能量使用的影响。轴向肌节耦合、适中的肌肉收缩时间和适当的激活相位滞后对高效游泳至关重要。

### 背景

传统研究常常错过神经肌肉控制与全身运动之间的联系，需要更深入理解鱼类游泳的机制。

### 目的

探索鲹形游泳中的能量传递，创建一个仿生数字鳟鱼模型研究肌肉激活策略对游泳性能的影响。

### 方法

结合多体动力学、希尔型肌肉建模和高保真流固耦合算法创建仿生数字鳟鱼模型，使用深度强化学习实现神经系统的时空分层控制，系统检查激活策略对速度和能量使用的影响。

### 主要发现

轴向肌节耦合（激活跨越超过0.5个身体长度）对稳定身体波传播至关重要；适中的肌肉收缩时间（尾拍周期的[0.1,0.3]）使身体和流体作为被动阻尼系统，减少能量使用；肌节的激活相位滞后塑造身体波，过大的相位滞后会导致拮抗收缩阻碍推力。

### 结论

这些发现推进了对仿生运动的理解，有助于设计节能的水下系统。

### 翻译

理解高效的鱼类游泳可为生物力学、流体动力学和工程学提供见解。传统研究常常忽略了神经肌肉控制与全身运动之间的联系。为了探索鲹形游泳中的能量传递，我们创建了一个仿生数字鳟鱼。该模型结合了多体动力学、希尔型肌肉建模和高保真流固耦合算法，准确复制了真实鳟鱼的形式和特性。使用深度强化学习，鳟鱼的神经系统实现了对肌肉激活的时空分层控制。我们系统检查了激活策略如何影响速度和能量使用。结果表明，轴向肌节耦合（激活跨越超过0.5个身体长度）对于稳定身体波传播至关重要。适中的肌肉收缩时间（尾拍周期的[0.1,0.3]）使身体和流体作为被动阻尼系统，减少能量使用。此外，肌节的激活相位滞后塑造了身体波；如果过大，会导致拮抗收缩阻碍推力。这些发现推进了仿生运动的理解，并有助于设计节能的水下系统。


### 论文摘要

Understanding efficient fish locomotion offers insights for biomechanics, fluid dynamics, and engineering. Traditional studies often miss the link between neuromuscular control and whole-body movement. To explore energy transfer in carangiform swimming, we created a bio-inspired digital trout. This model combined multibody dynamics, Hill-type muscle modeling, and a high-fidelity fluid-structure interaction algorithm, accurately replicating a real trout's form and properties. Using deep reinforcement learning, the trout's neural system achieved hierarchical spatiotemporal control of muscle activation. We systematically examined how activation strategies affect speed and energy use. Results show that axial myomere coupling-with activation spanning over 0.5 body lengths-is crucial for stable body wave propagation. Moderate muscle contraction duration ([0.1,0.3] of a tail-beat cycle) lets the body and fluid act as a passive damping system, cutting energy use. Additionally, the activation phase lag of myomeres shapes the body wave; if too large, it causes antagonistic contractions that hinder thrust. These findings advance bio-inspired locomotion understanding and aid energy-efficient underwater system design.

---

## 179. Physics-Constrained Neural Dynamics: A Unified Manifold Framework for Large-Scale Power Flow Computation

**论文链接:** [http://arxiv.org/abs/2512.01207v1](http://arxiv.org/abs/2512.01207v1)

**作者:** Xuezhi Liu

**发布时间:** 2025-12-01

### GPT解析

### 总结

本文提出了一种基于流形几何和梯度流的神经物理潮流求解方法，通过将潮流方程转化为约束流形并构建能量函数与梯度流，将潮流求解转化为动力学系统平衡点寻找问题，实现了无需标记数据的物理约束学习。

### 背景

潮流分析是电力系统分析、规划和运行控制的基本工具，但传统牛顿-拉夫森方法存在初值敏感性和批量计算效率低等问题，而现有基于深度学习的潮流求解器大多依赖监督学习，需要预先求解大量案例且难以保证物理一致性。

### 目的

开发一种无需大量预计算案例且能保证物理一致性的潮流求解方法。

### 方法

将潮流方程描述为约束流形，构建能量函数和梯度流，将潮流求解转化为动力学系统平衡点寻找问题，通过直接最小化物理残差以无监督方式训练神经网络。

### 主要发现

所提出的方法可以实现真正的'端到端'物理约束学习，无需标记数据即可训练。

### 结论

基于流形几何和梯度流的神经物理潮流求解方法克服了传统方法和现有深度学习方法的一些局限性，为电力系统潮流分析提供了新思路。

### 翻译

潮流分析是电力系统分析、规划和运行控制的基本工具。传统的牛顿-拉夫森方法存在初值敏感性和批量计算效率低等局限性，而现有的基于深度学习的潮流求解器大多依赖监督学习，需要预先求解大量案例且难以保证物理一致性。本文提出了一种基于流形几何和梯度流的神经物理潮流求解方法，通过将潮流方程描述为约束流形，构建能量函数和梯度流，将潮流求解转化为动力学系统平衡点寻找问题。神经网络通过直接最小化物理残差以无监督方式进行训练，不需要标记数据，实现了真正的'端到端'物理约束学习。


### 论文摘要

Power flow analysis is a fundamental tool for power system analysis, planning, and operational control. Traditional Newton-Raphson methods suffer from limitations such as initial value sensitivity and low efficiency in batch computation, while existing deep learning-based power flow solvers mostly rely on supervised learning, requiring pre-solving of numerous cases and struggling to guarantee physical consistency. This paper proposes a neural physics power flow solving method based on manifold geometry and gradient flow, by describing the power flow equations as a constraint manifold, and constructing an energy function \(V(\mathbf{x}) = \frac{1}{2}\|\mathbf{F}(\mathbf{x})\|^2\) and gradient flow \(\frac{d\mathbf{x}}{dt} = -\nabla V(\mathbf{x})\), transforming power flow solving into an equilibrium point finding problem for dynamical systems. Neural networks are trained in an unsupervised manner by directly minimizing physical residuals, requiring no labeled data, achieving true "end-to-end" physics-constrained learning.

---

## 180. 论文ID: 2512.01148v1

**错误:** 无法读取论文信息 - [Errno 2] No such file or directory: 'storage/result/2512.01148v1.json'

---

## 181. OmniFD: A Unified Model for Versatile Face Forgery Detection

**论文链接:** [http://arxiv.org/abs/2512.01128v1](http://arxiv.org/abs/2512.01128v1)

**作者:** Haotian Liu, Haoyu Chen, Chenhui Pan, You Hu, Guoying Zhao, Xiaobai Li

**发布时间:** 2025-11-30

### GPT解析

### 总结

OmniFD是一个统一的面部伪造检测框架，能够在单一模型中同时处理图像和视频分类、空间定位和时间定位四个核心任务，相比传统方法更高效且性能更优。

### 背景

当前的面部伪造检测方法通常使用特定任务的独立模型，导致计算冗余并忽略了相关任务间的潜在关联。

### 目的

开发一个统一框架，在单一模型中联合解决四个核心面部伪造检测任务，提高效率并捕捉任务间的关联。

### 方法

OmniFD架构包含三个主要组件：(1)共享的Swin Transformer编码器，提取统一的4D时空表示；(2)跨任务交互模块，通过基于注意力的推理捕获任务间依赖；(3)轻量级解码头，将表示转换为各任务的预测。

### 主要发现

OmniFD优于特定任务模型；统一设计利用多任务学习实现细粒度知识转移；融入图像数据后视频分类准确率提高4.63%；统一框架减少63%模型参数和50%训练时间，同时保持优越性能。

### 结论

OmniFD为现实世界应用中的全面面部伪造检测提供了实用且可推广的解决方案。

### 翻译

面部伪造检测包括多个关键任务，包括识别伪造图像和视频以及定位操作区域和时间片段。当前方法通常采用具有独立架构的任务特定模型，导致计算冗余并忽略了相关任务间的潜在关联。我们引入了OmniFD，这是一个统一框架，在单一模型中共同解决四个核心面部伪造检测任务，即图像和视频分类、空间定位和时间定位。我们的架构包含三个主要组件：(1)共享的Swin Transformer编码器，从图像和视频输入中提取统一的4D时空表示；(2)跨任务交互模块，使用可学习的查询通过基于注意力的推理动态捕获任务间依赖关系；(3)轻量级解码头，将精细表示转换为所有FFD任务的相应预测。大量实验证明了OmniFD相比特定任务模型的优势。其统一设计利用多任务学习捕获跨任务的通用表示，特别是实现细粒度知识转移，促进其他任务。例如，当融入图像数据时，视频分类准确率提高了4.63%。此外，通过在一个框架中统一图像、视频和四个任务，OmniFD在各种基准测试中实现了优越的性能，同时具有高效率和可扩展性，例如减少63%的模型参数和50%的训练时间。它为现实世界应用中的全面面部伪造检测提供了一个实用且可推广的解决方案。源代码可在https://github.com/haotianll/OmniFD获取。


### 论文摘要

Face forgery detection encompasses multiple critical tasks, including identifying forged images and videos and localizing manipulated regions and temporal segments. Current approaches typically employ task-specific models with independent architectures, leading to computational redundancy and ignoring potential correlations across related tasks. We introduce OmniFD, a unified framework that jointly addresses four core face forgery detection tasks within a single model, i.e., image and video classification, spatial localization, and temporal localization. Our architecture consists of three principal components: (1) a shared Swin Transformer encoder that extracts unified 4D spatiotemporal representations from both images and video inputs, (2) a cross-task interaction module with learnable queries that dynamically captures inter-task dependencies through attention-based reasoning, and (3) lightweight decoding heads that transform refined representations into corresponding predictions for all FFD tasks. Extensive experiments demonstrate OmniFD's advantage over task-specific models. Its unified design leverages multi-task learning to capture generalized representations across tasks, especially enabling fine-grained knowledge transfer that facilitates other tasks. For example, video classification accuracy improves by 4.63% when image data are incorporated. Furthermore, by unifying images, videos and the four tasks within one framework, OmniFD achieves superior performance across diverse benchmarks with high efficiency and scalability, e.g., reducing 63% model parameters and 50% training time. It establishes a practical and generalizable solution for comprehensive face forgery detection in real-world applications. The source code is made available at https://github.com/haotianll/OmniFD.

---

## 182. Learning Eigenstructures of Unstructured Data Manifolds

**论文链接:** [http://arxiv.org/abs/2512.01103v1](http://arxiv.org/abs/2512.01103v1)

**作者:** Roy Velich, Arkadi Piven, David Bensaïd, Daniel Cremers, Thomas Dagès, Ron Kimmel

**发布时间:** 2025-11-30

### GPT解析

### 总结

提出一种新框架，直接从非结构化数据中学习用于形状和流形分析的光谱基，无需传统算子选择、离散化和特征值求解器。

### 背景

传统形状和流形分析方法需要选择算子、离散化和特征值求解器，这些步骤繁琐且需要专业知识。

### 目的

开发一个直接从非结构化数据中学习光谱基的框架，消除传统方法中的算子选择、离散化和特征值求解需求。

### 方法

基于最优逼近理论，训练网络分解隐式逼近算子，通过在选定探测函数分布上最小化重建误差来学习光谱基。

### 主要发现

对于合适分布可视为拉普拉斯算子近似；能统一恢复光谱基、采样密度和特征值；无监督方法不假设数据流形，可扩展到任意维度；在点云和图像流形上产生有意义的光谱基，无需显式构建算子。

### 结论

用基于学习的方法替代传统算子处理流程，为非结构化数据几何处理提供数据驱动方案，特别是在高维空间开辟新可能性。

### 翻译

我们引入了一种新框架，直接从非结构化数据中学习用于形状和流形分析的光谱基，消除了传统算子选择、离散化和特征值求解器的需求。基于最优逼近理论，我们训练网络分解隐式逼近算子，通过在选定探测函数分布上最小化重建误差来学习光谱基。对于合适的分布，它们可以被视为拉普拉斯算子及其特征分解的近似，这在几何处理中是基础。此外，我们的方法能统一恢复光谱基、隐式度量的采样密度和底层算子的特征值。值得注意的是，我们的无监督方法不假设数据流形，如网格化或流形维度，使其可扩展到任意维度的数据集。在3D表面上的点云和高维图像流形上，我们的方法能产生有意义的光谱基，类似于拉普拉斯算子的光谱基，无需显式构建算子。通过用基于学习的方法替代传统算子选择、构建和特征分解，我们的框架为非结构化数据的几何处理提供了原则性、数据驱动的替代方案，特别是在高维空间中开辟了新的可能性。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决如何直接从非结构化数据中学习谱基用于形状和流形分析，无需传统的算子选择、离散化和特征分解步骤。这个问题很重要，因为现实世界数据多为非结构化（如点云），传统方法难以处理高维流形，且计算成本高，限制了几何处理在复杂场景中的应用。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者从最优逼近理论出发，认识到对于受约束的信号类，最优正交基由约束算子的特征向量给出。他们不针对特定算子，而是学习一个隐式算子的谱分解。借鉴了最优逼近理论、拉普拉斯-贝尔特拉米算子应用、图谱方法和神经特征值求解等工作，但创新性地整合这些思想，设计出直接从非结构化数据学习谱基的框架。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是基于最优逼近理论，学习一个能最优重建探测函数的基，同时隐式学习流形度量和算子特征值。流程包括：1)神经网络预测点云特征向量；2)QR分解得到正交基；3)生成探测函数；4)计算重建误差；5)通过最小化重建误差训练网络；6)推理时预测谱基并估计特征值。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：直接学习谱基、算子无关框架、统一学习谱基和度量、高维扩展性、基于最优逼近的理论基础。相比之前工作，本方法无需显式选择离散化算子，不依赖网格结构，能处理任意维度流形，且能同时学习谱基和度量，而传统方法需要分别处理这些组件。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '这篇论文提出了一种基于最优逼近理论的神经网络框架，能够直接从任意维度的非结构化数据中学习谱基，绕过了传统几何处理中算子选择、离散化和特征分解的步骤，为高维流形分析提供了新的数据驱动方法。'}


### 论文摘要

We introduce a novel framework that directly learns a spectral basis for shape and manifold analysis from unstructured data, eliminating the need for traditional operator selection, discretization, and eigensolvers. Grounded in optimal-approximation theory, we train a network to decompose an implicit approximation operator by minimizing the reconstruction error in the learned basis over a chosen distribution of probe functions. For suitable distributions, they can be seen as an approximation of the Laplacian operator and its eigendecomposition, which are fundamental in geometry processing. Furthermore, our method recovers in a unified manner not only the spectral basis, but also the implicit metric's sampling density and the eigenvalues of the underlying operator. Notably, our unsupervised method makes no assumption on the data manifold, such as meshing or manifold dimensionality, allowing it to scale to arbitrary datasets of any dimension. On point clouds lying on surfaces in 3D and high-dimensional image manifolds, our approach yields meaningful spectral bases, that can resemble those of the Laplacian, without explicit construction of an operator. By replacing the traditional operator selection, construction, and eigendecomposition with a learning-based approach, our framework offers a principled, data-driven alternative to conventional pipelines. This opens new possibilities in geometry processing for unstructured data, particularly in high-dimensional spaces.

---

## 183. Unsupervised Machine Learning for Experimental Detection of Quantum-Many-Body Phase Transitions

**论文链接:** [http://arxiv.org/abs/2512.01091v1](http://arxiv.org/abs/2512.01091v1)

**作者:** Ron Ziv, David Wei, Antonio Rubio-Abadal, Daniel Adler, Anna Keselman, Eran Lustig, Ronen Talmon, Johannes Zeiher, Immanuel Bloch, Mordechai Segev

**发布时间:** 2025-11-30

### GPT解析

### 总结

本文提出了一种无监督机器学习方法，用于直接从原始实验测量中检测量子多体系统的相变和交叉现象，无需对系统有任何模型特定的先验知识。

### 背景

量子多体系统通常计算上难以精确模拟，所需资源往往远超现有计算能力。费曼提出的量子模拟器概念虽解决了量子动力学模拟问题，但无法解决从实验中有限的可观测量推断基础物理的同样基本问题。识别QMB系统中的相变（尤其是在没有简单序参量时）仍然是一个重大挑战，且数值模拟常因系统过大或有限尺寸效应而不可行。

### 目的

开发一种无监督机器学习方法，专门用于研究QMB实验，直接从原始实验测量中检测相变和交叉现象。

### 方法

采用无监督机器学习方法，直接分析原始实验数据，无需对系统有特定模型假设，适用于大规模复杂系统。

### 主要发现

该方法在多体局域化交叉和莫特到超流体相变的系统上得到验证，能够从非常有限的实验数据中揭示集体现象，且无需系统先验知识。

### 结论

该研究为复杂量子多体系统中涌现现象的数据驱动发现提供了一条通用且可扩展的途径。

### 翻译

量子多体系统通常计算上难以处理：精确模拟它们所需的计算资源往往超出现有计算资源数个数量级。为此，理查德·费曼提出了量子模拟器的概念：设计遵循特定演化方程的量子系统并重复实验多次。然而，正如下文所述，实验中描述系统的大多数可观测量是无法获取的。因此，虽然费曼的想法解决了量子动力学模拟问题，但它同样未解决从实验中有限可观测量推断基础物理的基本问题。确实，与QMB系统相关的许多复杂现象仍然难以捉摸。或许，最重要的例子是在没有简单序参量存在的情况下识别QMB系统中的相变，这至今仍构成重大挑战。进一步复杂化的是，在大多数情况下，无法从数值模拟中学习，因为底层系统通常太大而无法计算，且小的QMB系统可能表现出强烈的有限尺寸效应，掩盖了相变的存在。在此，我们提出了一种无监督机器学习方法来研究QMB实验，专门旨在直接从原始实验测量中检测相变和交叉现象。我们在经历多体局域化交叉和莫特到超流体相变的系统上展示了我们的方法，表明它从非常有限的实验数据中揭示了集体现象，且无需任何系统特定的先验知识。这种方法为复杂量子多体系统中涌现现象的数据驱动发现提供了一条通用且可扩展的途径。


### 论文摘要

Quantum many-body (QMB) systems are generally computationally hard: the computing resources necessary to simulate them exactly can often exceed the existing computation resources by orders of magnitude. For this reason, Richard Feynman proposed the concept of a quantum simulator: quantum systems engineered to obey a prescribed evolution equation and repeating the experiment multiple times. Experimentally, however, as we explain below, the vast majority of observables describing the system are inaccessible. Thus, while Feynman's idea addresses the problem of simulating quantum dynamics, it leaves unsolved the equally fundamental problem of inferring the underlying physics from the limited observables accessible in experiments. Indeed, many complex phenomena associated with QMB systems remain elusive. Perhaps, the most important example is identifying phase transitions in QMB systems when no simple order-parameter exists, which poses major challenges to this day. Complicating the problem further is the fact that, in most cases, it is impossible to learn from numerical simulations, as the underlying systems are often too large to be computable, and small QMB can show strong finite size effects, masking the presence of the transition. Here, we present an unsupervised machine learning approach to study QMB experiments, specifically aimed at detecting phase transitions and crossovers directly from raw experimental measurements. We demonstrate our methodology on systems undergoing Many-Body Localization cross-over and Mott-to-Superfluid phase-transition, showing that it reveals collective phenomena from the very partial experimental data and without any model-specific prior knowledge of the system. This approach offers a general and scalable route for data-driven discovery of emergent phenomena in complex quantum many-body systems.

---

## 184. Building Trustworthy AI for Materials Discovery: From Autonomous Laboratories to Z-scores

**论文链接:** [http://arxiv.org/abs/2512.01080v1](http://arxiv.org/abs/2512.01080v1)

**作者:** Benhour Amirian, Ashley S. Dale, Sergei Kalinin, Jason Hattrick-Simpers

**发布时间:** 2025-11-30

### GPT解析

### 总结

该研究提出了GIFTERS框架，用于评估材料科学中人工智能和机器学习方法的可信赖性，包括泛化性、可解释性、公平性、透明度、可解释性、鲁棒性和稳定性。研究发现现有方法在可信赖性方面存在不足，并提出了改进方向。

### 背景

加速材料发现越来越依赖人工智能和机器学习（AI/ML），但使用AI的关键挑战在于确保人类科学家相信模型的可靠性和有效性。

### 目的

定义一个名为GIFTERS的可信赖AI框架，用于评估材料科学和发现领域中的机器学习方法是否具有可信赖性，并探索改进可信赖性方法的方向。

### 方法

通过批判性文献综述，确定材料发现社区最重视的可信赖性原则，评估现有AI/ML方法在材料科学中的可信赖性，并借鉴其他科学领域（如医疗保健、气候科学和自然语言处理）的工作。

### 主要发现

材料发现社区最重视的可信赖性原则包括泛化性、可解释性、公平性、透明度、可解释性、鲁棒性和稳定性；全面处理可信赖性的方法很少被报告（中位数GIFTERS得分为5/7）；贝叶斯研究经常省略公平的数据实践，而非贝叶斯研究最常省略可解释性；确定了改进材料科学中AI/ML可信赖性方法的方法。

### 结论

需要人类参与循环和综合方法来弥合可信赖性与不确定性量化之间的差距；未来材料科学研究应确保AI/ML方法不仅加速发现，而且符合材料发现社区建立的伦理和科学规范；该研究为开发能够准确自信地促进材料发现的可信赖人工智能系统提供了路线图。

### 翻译

加速材料发现越来越多地依赖人工智能和机器学习，统称为'AI/ML'。使用AI的一个关键挑战是确保人类科学家相信模型是有效和可靠的。因此，我们为材料科学和发现定义了一个名为GIFTERS的可信赖AI框架，以评估报告的机器学习方法是否具有泛化性、可解释性、公平性、透明度、可解释性、鲁棒性和稳定性。通过批判性文献综述，我们强调这些是材料发现社区最重视的可信赖性原则。然而，我们也发现全面处理可信赖性的方法很少被报告；这通过5/7的中位数GIFTERS分数来量化。我们观察到贝叶斯研究经常省略公平的数据实践，而非贝叶斯研究最常省略可解释性。最后，我们确定了改进材料科学中人工智能和机器学习可信赖性方法的方法，通过考虑在其他科学领域（如医疗保健、气候科学和自然语言处理）完成的工作，特别强调可能转移到材料发现实验的方法。通过结合这些观察，我们强调了人类参与循环的必要性，以及综合方法来弥合可信赖性与不确定性量化之间的差距，作为材料科学研究的未来方向。这确保AI/ML方法不仅加速发现，而且符合材料发现社区建立的伦理和科学规范。这项工作为开发能够准确自信地促进材料发现的可信赖人工智能系统提供了路线图。


### 论文摘要

Accelerated material discovery increasingly relies on artificial intelligence and machine learning, collectively termed "AI/ML". A key challenge in using AI is ensuring that human scientists trust the models are valid and reliable. Accordingly, we define a trustworthy AI framework GIFTERS for materials science and discovery to evaluate whether reported machine learning methods are generalizable, interpretable, fair, transparent, explainable, robust, and stable. Through a critical literature review, we highlight that these are the trustworthiness principles most valued by the materials discovery community. However, we also find that comprehensive approaches to trustworthiness are rarely reported; this is quantified by a median GIFTERS score of 5/7. We observe that Bayesian studies frequently omit fair data practices, while non-Bayesian studies most frequently omit interpretability. Finally, we identify approaches for improving trustworthiness methods in artificial intelligence and machine learning for materials science by considering work accomplished in other scientific disciplines such as healthcare, climate science, and natural language processing with an emphasis on methods that may transfer to materials discovery experiments. By combining these observations, we highlight the necessity of human-in-the-loop, and integrated approaches to bridge the gap between trustworthiness and uncertainty quantification for future directions of materials science research. This ensures that AI/ML methods not only accelerate discovery, but also meet ethical and scientific norms established by the materials discovery community. This work provides a road map for developing trustworthy artificial intelligence systems that will accurately and confidently enable material discovery.

---

## 185. Opening the Sim-to-Real Door for Humanoid Pixel-to-Action Policy Transfer

**论文链接:** [http://arxiv.org/abs/2512.01061v1](http://arxiv.org/abs/2512.01061v1)

**作者:** Haoru Xue, Tairan He, Zi Wang, Qingwei Ben, Wenli Xiao, Zhengyi Luo, Xingye Da, Fernando Castañeda, Guanya Shi, Shankar Sastry, Linxi "Jim" Fan, Yuke Zhu

**发布时间:** 2025-11-30

**备注:** https://doorman-humanoid.github.io/

### GPT解析

### 总结

该研究开发了一种基于GPU加速的、照片级真实感模拟的机器人学习框架，针对视觉类人机器人的定位操作任务，通过教师-学生-自举学习方法和分阶段重置探索策略，实现了在不同类型门上的零样本性能，超越了人类远程操作员的表现。

### 背景

GPU加速的、照片级真实感模拟技术的最新进展为机器人学习提供了可扩展的数据生成路径，通过大量物理和视觉随机化，使策略能够超越精心设计的环境限制。

### 目的

开发一种能够处理高难度关节化物体交互任务的视觉类人机器人定位操作学习框架，实现模拟到现实的策略迁移，提高任务完成效率。

### 方法

研究采用了教师-学生-自举学习框架，引入分阶段重置探索策略以稳定长期特权的策略训练，并使用基于GRPO的微调程序减轻部分可观察性问题，提高模拟到现实RL中的闭环一致性。

### 主要发现

完全在模拟数据上训练的策略在不同类型门上实现了强大的零样本性能，在相同全身控制堆栈下，任务完成时间比人类远程操作员提高高达31.7%。

### 结论

该研究首次实现了能够使用纯RGB感知进行多样化关节化定位操作的类人机器人模拟到现实策略，为机器人学习领域提供了新的解决方案。

### 翻译

GPU加速、照片级真实感模拟的最新进展为机器人学习开辟了一条可扩展的数据生成路径，其中大量的物理和视觉随机化使策略能够超越精心设计的环境限制。在这些进展的基础上，我们为视觉类人机器人定位操作开发了一种教师-学生-自举学习框架，使用关节化物体交互作为代表性高难度基准。我们的方法引入了分阶段重置探索策略，稳定了长期特权的策略训练，以及基于GRPO的微调程序，减轻了部分可观察性问题，并提高了模拟到现实RL中的闭环一致性。完全在模拟数据上训练的策略在不同类型门上实现了强大的零样本性能，并且在相同的全身控制堆栈下，任务完成时间比人类远程操作员提高高达31.7%。这是第一个能够使用纯RGB感知进行多样化关节化定位操作的类人机器人模拟到现实策略。


### 论文摘要

Recent progress in GPU-accelerated, photorealistic simulation has opened a scalable data-generation path for robot learning, where massive physics and visual randomization allow policies to generalize beyond curated environments. Building on these advances, we develop a teacher-student-bootstrap learning framework for vision-based humanoid loco-manipulation, using articulated-object interaction as a representative high-difficulty benchmark. Our approach introduces a staged-reset exploration strategy that stabilizes long-horizon privileged-policy training, and a GRPO-based fine-tuning procedure that mitigates partial observability and improves closed-loop consistency in sim-to-real RL. Trained entirely on simulation data, the resulting policy achieves robust zero-shot performance across diverse door types and outperforms human teleoperators by up to 31.7% in task completion time under the same whole-body control stack. This represents the first humanoid sim-to-real policy capable of diverse articulated loco-manipulation using pure RGB perception.

---

## 186. Associative Syntax and Maximal Repetitions reveal context-dependent complexity in fruit bat communication

**论文链接:** [http://arxiv.org/abs/2512.01033v1](http://arxiv.org/abs/2512.01033v1)

**作者:** Luigi Assom

**发布时间:** 2025-11-30

**备注:** Accepted for a lightning talk at the NeurIPS 2025 Workshop: "AI for Non-Human Animal Communication"

### GPT解析

### 总结

本研究提出了一种无监督方法来推断果蝠叫声的离散性、句法和时间结构，并评估与行为背景相关的交流模式复杂性。

### 背景

研究果蝠叫声的分级发声系统，分析其交流模式与行为背景的关系。

### 目的

开发无监督方法来分析果蝠发声的离散性、句法和时间结构，并探究交流复杂性与行为背景的关联。

### 方法

通过流形学习改进发声单元的无监督标记；研究梅尔频谱图降维对标记的影响；将发声编码为音节序列分析句法；提取最大重复评估句法结构。

### 主要发现

1) 存在关联性而非组合性句法；2) 音节使用依赖于上下文；3) 最大重复呈重尾分布，表明存在编码组合复杂性的机制；4) 母子互动以重复为特征，冲突背景下交流复杂性更高；5) 分歧场景下交流复杂性更高，信息可压缩性更低。

### 结论

在存在分歧的情境中，果蝠交流复杂性更高，反映了信息的可压缩性较低。

### 翻译

本研究提出了一种无监督方法来推断果蝠叫声的离散性、句法和时间结构，作为分级发声系统的案例研究，并评估与行为背景相关的交流模式复杂性。该方法通过流形学习改进了发声单元（即音节）的无监督标记基线，通过研究梅尔频谱图上的降维如何影响标记，并将其与基于声学相似性的无监督标记进行比较。然后我们将发声编码为音节序列来分析句法类型，并提取最大重复（MR）来评估句法结构。我们发现：i) 存在关联性句法，而非组合性句法（序列置换不影响上下文分类，F1 > 0.9）；ii) 音节使用依赖于上下文（Wilcoxon秩和检验，p值 < 0.05）；iii) MR呈重尾分布（截断幂律，指数α < 2），表明存在编码组合复杂性的机制。MR分析和音节转换网络显示，母子互动以重复为特征，而冲突背景下的交流比非攻击性背景具有更高的复杂性（更长的MR和更互联的发声序列）。我们提出在分歧场景下交流复杂性更高，反映了信息的可压缩性较低。


### 论文摘要

This study presents an unsupervised method to infer discreteness, syntax and temporal structures of fruit-bats vocalizations, as a case study of graded vocal systems, and evaluates the complexity of communication patterns in relation with behavioral context. The method improved the baseline for unsupervised labeling of vocal units (i.e. syllables) through manifold learning, by investigating how dimen- sionality reduction on mel-spectrograms affects labeling, and comparing it with unsupervised labels based on acoustic similarity. We then encoded vocalizations as syllabic sequences to analyze the type of syntax, and extracted the Maximal Repetitions (MRs) to evaluate syntactical structures. We found evidence for: i) associative syntax, rather than combinatorial (context classification is unaffected by permutation of sequences, F 1 > 0.9); ii) context-dependent use of syllables (Wilcoxon rank-sum tests, p-value < 0.05); iii) heavy-tail distribution of MRs (truncated power-law, exponent α < 2), indicative of mechanism encoding com- binatorial complexity. Analysis of MRs and syllabic transition networks revealed that mother-pupil interactions were characterized by repetitions, while commu- nication in conflict-contexts exhibited higher complexity (longer MRs and more interconnected vocal sequences) than non-agonistic contexts. We propose that communicative complexity is higher in scenarios of disagreement, reflecting lower compressibility of information.

---

## 187. Operator-Theoretic Framework for Gradient-Free Federated Learning

**论文链接:** [http://arxiv.org/abs/2512.01025v1](http://arxiv.org/abs/2512.01025v1)

**作者:** Mohit Kumar, Mathias Brucker, Alexander Valentinitsch, Adnan Husakovic, Ali Abbas, Manuela Geiß, Bernhard A. Moser

**发布时间:** 2025-11-30

### GPT解析

### 总结

该研究提出了一种基于算子理论的联邦学习框架，解决了异构性、通信限制和隐私保护问题，通过核机器方法实现了高效的知识传输和差分隐私保护，同时提供了与全同态加密兼容的预测规则。

### 背景

联邦学习需要解决数据异构性、严格的通信和计算限制以及隐私保护问题，同时确保模型性能。现有的梯度联邦学习方法在这些方面存在局限性。

### 目的

设计一种能够处理异构数据、减少通信开销、保护隐私且性能优异的联邦学习框架，并提供理论保证。

### 方法

提出基于算子理论的框架，将L²-最优解映射到再生核希尔伯特空间，利用核仿射包机器的空间折叠特性设计高效核机器，客户端通过标量空间折叠度量传输知识，实现差分私有协议，并开发与全同态加密兼容的预测规则。

### 主要发现

在四个基准测试中，该方法匹配或优于强梯度微调，增益高达23.7分；在差分私有实验中，核平滑减轻了高隐私设置下的精度损失；全局规则采用全同态加密实现时具有实用延迟。

### 结论

该框架提供了可证明的保证且通信量低，支持通过标量摘要进行私有知识传输，为异构性下的梯度联邦学习提供了数学基础上的替代方案。

### 翻译

联邦学习必须解决异构性、严格的通信和计算限制以及隐私问题，同时确保性能。我们提出了一种基于算子理论的框架，通过前向算子将L²-最优解映射到再生核希尔伯特空间，利用可用数据对其进行近似，然后通过逆算子映射回来，产生一种无梯度方案。使用算子范数上的集中不等式推导有限样本界限，该框架确定了一个与数据相关的假设空间，并对风险、误差、鲁棒性和近似性提供保证。在该空间内，我们设计利用核仿射包机器空间折叠特性的高效核机器。客户端通过标量空间折叠度量传输知识，减少通信量并实现简单的差分私有协议：从噪声扰动的数据矩阵中一次性计算摘要，避免了每轮的裁剪和隐私计算。诱导的全局规则每个测试点只需要整数最小值和等式比较操作，使其与全同态加密兼容。在四个基准测试中，具有固定编码器嵌入的无梯度联邦学习方法匹配或优于强梯度微调，增益高达23.7分。在差分私有实验中，核平滑减轻了高隐私设置下的精度损失。全局规则采用每个测试点Q×C加密最小值和C等式比较操作实现全同态加密，操作级基准测试显示了实际延迟。总的来说，该框架提供了可证明的保证且通信量低，支持通过标量摘要进行私有知识传输，并产生与全同态加密兼容的预测规则，为异构性下的梯度联邦学习提供了数学基础上的替代方案。


### 论文摘要

Federated learning must address heterogeneity, strict communication and computation limits, and privacy while ensuring performance. We propose an operator-theoretic framework that maps the $L^2$-optimal solution into a reproducing kernel Hilbert space (RKHS) via a forward operator, approximates it using available data, and maps back with the inverse operator, yielding a gradient-free scheme. Finite-sample bounds are derived using concentration inequalities over operator norms, and the framework identifies a data-dependent hypothesis space with guarantees on risk, error, robustness, and approximation. Within this space we design efficient kernel machines leveraging the space folding property of Kernel Affine Hull Machines. Clients transfer knowledge via a scalar space folding measure, reducing communication and enabling a simple differentially private protocol: summaries are computed from noise-perturbed data matrices in one step, avoiding per-round clipping and privacy accounting. The induced global rule requires only integer minimum and equality-comparison operations per test point, making it compatible with fully homomorphic encryption (FHE). Across four benchmarks, the gradient-free FL method with fixed encoder embeddings matches or outperforms strong gradient-based fine-tuning, with gains up to 23.7 points. In differentially private experiments, kernel smoothing mitigates accuracy loss in high-privacy regimes. The global rule admits an FHE realization using $Q \times C$ encrypted minimum and $C$ equality-comparison operations per test point, with operation-level benchmarks showing practical latencies. Overall, the framework provides provable guarantees with low communication, supports private knowledge transfer via scalar summaries, and yields an FHE-compatible prediction rule offering a mathematically grounded alternative to gradient-based federated learning under heterogeneity.

---

## 188. ForamDeepSlice: A High-Accuracy Deep Learning Framework for Foraminifera Species Classification from 2D Micro-CT Slices

**论文链接:** [http://arxiv.org/abs/2512.00912v1](http://arxiv.org/abs/2512.00912v1)

**作者:** Abdelghafour Halimi, Ali Alibrahim, Didier Barradas-Bautista, Ronell Sicat, Abdulkader M. Afifi

**发布时间:** 2025-11-30

### GPT解析

### 总结

本研究开发了一个全面的深度学习管道，用于自动分类12种有孔虫物种，使用来自3D扫描的2D显微CT切片。研究创建了一个包含97个显微CT扫描标本的数据集，评估了七种先进的卷积神经网络架构，最终的集成模型结合了ConvNeXt-Large和EfficientNetV2-Small，实现了95.64%的测试准确率。研究还开发了一个交互式高级仪表板，支持实时切片分类和3D切片匹配。

### 背景

有孔虫是微体古生物学中的重要研究对象，传统的分类方法可能存在主观性和效率低下的问题。本研究旨在利用深度学习技术实现有孔虫物种的自动分类，提高分类的准确性和效率。

### 目的

开发一个全面的深度学习管道，用于自动分类12种有孔虫物种，使用来自3D扫描的2D显微CT切片，并提供一个实用的交互式工具用于实际应用。

### 方法

创建了一个包含97个显微CT扫描标本的数据集，涵盖27个物种，选择了12个代表性足够的物种。使用标本级数据分割防止数据泄漏，总共使用了109,617个高质量2D切片（44,103个用于训练，14,046个用于验证，51,468个用于测试）。评估了七种最先进的2D卷积神经网络架构，使用迁移学习。最终的集成模型结合了ConvNeXt-Large和EfficientNetV2-Small。开发了一个交互式高级仪表板，支持实时切片分类和3D切片匹配，使用SSIM、NCC和Dice系数等高级相似度指标。

### 主要发现

最终的集成模型实现了95.64%的测试准确率，前3准确率达到99.6%，所有物种的ROC曲线下面积为0.998。这表明深度学习方法在有孔虫分类任务中具有很高的准确性和可靠性。

### 结论

本研究为AI辅助的微体古生物学鉴定建立了新的基准，为有孔虫分类研究提供了完全可复现的框架，成功弥合了深度学习和应用地球科学之间的差距。开发的交互式仪表板为实际应用提供了便利工具。

### 翻译

本研究提出了一个全面的深度学习管道，用于使用从3D扫描衍生的2D显微CT切片对12种有孔虫物种进行自动分类。我们整理了一个科学严谨的数据集，包含97个显微CT扫描标本，涵盖27个物种，选择了12个具有足够代表性的物种用于稳健的机器学习。为确保方法完整性和防止数据泄漏，我们采用了标本级数据分割，产生了109,617个高质量2D切片（44,103个用于训练，14,046个用于验证，51,468个用于测试）。我们使用迁移学习评估了七种最先进的2D卷积神经网络架构。我们的最终集成模型，结合了ConvNeXt-Large和EfficientNetV2-Small，实现了95.64%的测试准确率，所有物种的前3准确率为99.6%，ROC曲线下面积为0.998。为了便于实际部署，我们开发了一个交互式高级仪表板，支持实时切片分类和使用高级相似度指标（包括SSIM、NCC和Dice系数）进行3D切片匹配。这项工作为AI辅助的微体古生物学鉴定建立了新基准，并为有孔虫分类研究提供了完全可复现的框架，弥合了深度学习和应用地球科学之间的差距。


### 论文摘要

This study presents a comprehensive deep learning pipeline for the automated classification of 12 foraminifera species using 2D micro-CT slices derived from 3D scans. We curated a scientifically rigorous dataset comprising 97 micro-CT scanned specimens across 27 species, selecting 12 species with sufficient representation for robust machine learning. To ensure methodological integrity and prevent data leakage, we employed specimen-level data splitting, resulting in 109,617 high-quality 2D slices (44,103 for training, 14,046 for validation, and 51,468 for testing). We evaluated seven state-of-the-art 2D convolutional neural network (CNN) architectures using transfer learning. Our final ensemble model, combining ConvNeXt-Large and EfficientNetV2-Small, achieved a test accuracy of 95.64%, with a top-3 accuracy of 99.6% and an area under the ROC curve (AUC) of 0.998 across all species. To facilitate practical deployment, we developed an interactive advanced dashboard that supports real-time slice classification and 3D slice matching using advanced similarity metrics, including SSIM, NCC, and the Dice coefficient. This work establishes new benchmarks for AI-assisted micropaleontological identification and provides a fully reproducible framework for foraminifera classification research, bridging the gap between deep learning and applied geosciences.

---

## 189. Deep learning-based dynamic error correction and uncertainty estimation for digital twin-assisted fringe projection profilometry of rotating gears

**论文链接:** [http://arxiv.org/abs/2512.00859v1](http://arxiv.org/abs/2512.00859v1)

**作者:** Zhangsheng Li, Jiancheng Qiu, Gao Xu Wu

**发布时间:** 2025-11-30

### GPT解析

### 总结

这篇论文提出了一种基于深度学习的动态齿轮测量和不确定性估计方法，通过Unity平台生成模拟数据，使用Concrete Dropout网络进行像素级不确定性估计，并通过迁移学习策略提高性能。

### 背景

实际齿轮测量数据稀少，难以验证网络性能。

### 目的

开发一种动态齿轮测量和不确定性估计方法，解决数据稀缺问题并提高测量精度。

### 方法

使用Unity平台上的双系统生成多样化模拟数据集；设计Concrete Dropout-Pixel wise Uncertainty Network进行像素级不确定性估计；在输出层使用两个轻量级层提高预测结果的空间连续性；采用迁移学习策略进行网络训练。

### 主要发现

相比传统三步相移方法，该方法在相位预测精度、三维重建精度、动态误差校正能力和不确定性估计可靠性方面都有显著提高。

### 结论

该研究为基于条纹投影的动态齿轮测量提供了一种实用高效的技术解决方案。

### 翻译

本文提出了一种基于深度学习的动态齿轮测量和不确定性估计方法。在Unity平台上提出的双系统被用来灵活生成多样化的模拟数据集。这有效解决了实际齿轮测量数据稀少的问题，并便于验证网络性能。所设计的Concrete Dropout-Pixel wise Uncertainty Network集成了Concrete Dropout机制进行像素级不确定性估计。输出层采用了两个轻量级层来增强预测结果的空间连续性。在网络训练过程中，采用迁移学习策略：模型先用少量三步相移(3-PS)数据预训练，然后在目标齿轮测量数据集上进行微调。实验结果表明，与传统三步相移(3-PS)方法相比，所提出的方法在相位预测精度、三维重建精度、动态误差校正能力和不确定性估计可靠性方面都取得了显著改进。这项工作为基于条纹投影的动态齿轮测量提供了一种实用高效的技术解决方案。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决旋转齿轮在动态测量中的运动引起的相位误差问题，同时估计重建结果的不确定性。这个问题很重要，因为齿轮具有不连续表面、重复特征和显著高度变化的特点，使得全齿三维测量极具挑战性。传统方法在处理动态旋转齿轮时存在局限性：单帧方法(FTP)在大高度变化区精度损失，多帧方法(PSP)则遭受旋转引起的相位误差。现有方法难以在高动态、强非均匀运动或复杂表面场景下实现高精度3D重建，且缺乏统一的科学评估机制，高质量的动态齿轮3D真实数据难以获取。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先分析了四类现有方法在动态齿轮测量中的局限性，然后注意到深度学习在FPP中的应用潜力。作者借鉴了U-Net架构的基础，但进行了创新改进。设计过程中，作者提出使用数字孪生系统生成模拟数据解决真实数据获取困难的问题，设计了CD-DCU-Net网络集成Concrete Dropout进行不确定性估计，采用迁移学习策略（先模拟数据预训练再目标数据微调），并引入加权损失函数平衡不同学习目标。这些设计参考了现有的深度学习技术、贝叶斯神经网络和迁移学习策略，但进行了针对性创新以适应齿轮测量的特殊挑战。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是使用深度学习网络校正旋转引起的动态相位误差，同时估计重建结果的不确定性，并利用数字孪生技术解决数据获取问题。整体流程包括：1)构建物理和虚拟FPP系统，使用Unity数字孪生生成模拟数据并收集真实数据；2)设计CD-DCU-Net网络，采用编码器-解码器结构，集成Concrete Dropout和双3×3卷积输出层；3)采用迁移学习策略训练网络，先用模拟数据预训练再用真实数据微调；4)使用蒙特卡洛采样量化预测不确定性，计算模型和数据不确定性；5)与传统方法比较性能，测试不同运动条件下的表现，验证不确定性估计准确性。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)提出CD-DCU-Net网络，集成Concrete Dropout进行逐像素不确定性估计；2)在输出层使用双3×3卷积增强预测空间连续性；3)采用迁移学习策略提高模型泛化能力；4)引入加权损失函数平衡相位预测和不确定性估计；5)构建Unity数字孪生系统生成多样化模拟数据；6)提出虚拟-现实混合训练模式。相比之前工作，本文方法不仅关注相位校正，还同时估计不确定性；解决了齿轮复杂重复表面导致的相位恢复难题；通过迁移学习策略解决了真实数据稀缺问题；在高动态、强非均匀运动条件下表现更好，且提供了重建结果可靠性的量化评估。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '这篇论文提出了一种结合深度学习和数字孪生技术的创新方法，有效解决了旋转齿轮动态测量中的相位误差问题，同时提供了可靠的不确定性估计，显著提高了三维测量的精度和可靠性。'}


### 论文摘要

This paper presents a deep learning-based method for dynamic gear measurement and uncertainty estimation. A twin-system proposed on the Unity platform is utilized to flexibly generate diverse simulated datasets. This effectively addresses the scarcity of real-world gear measurement data and facilitates verification of network performance.The designed Concrete Dropout-Pixel wise Uncertainty Network integrates the Concrete Dropout mechanism for pixel-level uncertainty estimation. Two lightweight layers are employed in the output layer to enhance the spatial continuity of prediction results.During network training, a transfer learning strategy is adopted: the model is first pretrained with a small amount of three-phase-shifting (3-PS) data, then fine-tuned on the target gear measurement dataset. Experimental results demonstrate that, compared with the traditional three-step phase-shifting (3-PS) method, the proposed approach achieves significant improvements in phase prediction accuracy, three-dimensional reconstruction accuracy, dynamic error correction capability, and uncertainty estimation reliability.This work provides a practical and efficient technical solution for fringe projection-based dynamic gear measurement.

---

## 190. One Swallow Does Not Make a Summer: Understanding Semantic Structures in Embedding Spaces

**论文链接:** [http://arxiv.org/abs/2512.00852v1](http://arxiv.org/abs/2512.00852v1)

**作者:** Yandong Sun, Qiang Huang, Ziwei Xu, Yiqun Sun, Yixuan Tang, Anthony K. H. Tung

**发布时间:** 2025-11-30

### GPT解析

### 总结

该研究提出了语义场子空间(SFS)和SAFARI算法，用于解决嵌入空间内部结构不透明的问题，实现了高效且可解释的语义分析。

### 背景

嵌入空间是现代AI的基础，将原始数据转换为编码丰富语义关系的高维向量，但现有方法通常在语义一致性和结构规则性之间难以平衡，或因提高可解释性而导致高计算开销。

### 目的

开发一种既能保持几何结构又能感知上下文的表示方法，以及一种无监督的算法来发现分层的语义结构，解决嵌入空间内部结构不透明的问题。

### 方法

提出语义场子空间(SFS)捕获嵌入空间中的局部语义邻域；开发SAFARI算法使用'语义偏移'指标揭示分层语义结构；实现语义偏移的高效近似方法，用低成本计算替代昂贵的SVD计算，实现15-30倍加速且误差低于0.01。

### 主要发现

在六个真实世界的文本和图像数据集上评估显示，SFS在分类任务和如政治偏见检测等微妙任务上均优于标准分类器；SAFARI能提供一致可解释和可推广的语义层次结构。

### 结论

该研究为结构化、分析和扩展嵌入空间中的语义理解提供了一个统一的框架。

### 翻译

嵌入空间是现代AI的基础，将原始数据转换为编码丰富语义关系的高维向量。然而，它们的内部结构仍然不透明，现有方法通常为了结构规则性而牺牲语义一致性，或者为提高可解释性而带来高计算开销。为解决这些挑战，我们引入了语义场子空间(SFS)，这是一种保持几何结构、感知上下文的表示，用于捕获嵌入空间中的局部语义邻域。我们还提出了SAFARI(语义场子空间确定)算法，这是一种无监督的、模态无关的算法，使用称为'语义偏移'的新指标来揭示分层的语义结构，该指标量化了语义如何随着SFS的演变而演变。为确保可扩展性，我们开发了语义偏移的高效近似方法，用低成本计算替代了昂贵的SVD计算，实现了15-30倍的加速，平均误差低于0.01。在六个真实世界的文本和图像数据集上的广泛评估表明，SFS不仅在分类任务上优于标准分类器，还在政治偏见检测等微妙任务上表现更好，同时SAFARI一致地提供了可解释和可推广的语义层次结构。这项工作为结构化、分析和扩展嵌入空间中的语义理解提供了一个统一的框架。


### 论文摘要

Embedding spaces are fundamental to modern AI, translating raw data into high-dimensional vectors that encode rich semantic relationships. Yet, their internal structures remain opaque, with existing approaches often sacrificing semantic coherence for structural regularity or incurring high computational overhead to improve interpretability. To address these challenges, we introduce the Semantic Field Subspace (SFS), a geometry-preserving, context-aware representation that captures local semantic neighborhoods within the embedding space. We also propose SAFARI (SemAntic Field subspAce deteRmInation), an unsupervised, modality-agnostic algorithm that uncovers hierarchical semantic structures using a novel metric called Semantic Shift, which quantifies how semantics evolve as SFSes evolve. To ensure scalability, we develop an efficient approximation of Semantic Shift that replaces costly SVD computations, achieving a 15~30x speedup with average errors below 0.01. Extensive evaluations across six real-world text and image datasets show that SFSes outperform standard classifiers not only in classification but also in nuanced tasks such as political bias detection, while SAFARI consistently reveals interpretable and generalizable semantic hierarchies. This work presents a unified framework for structuring, analyzing, and scaling semantic understanding in embedding spaces.

---

## 191. Exploiting Function-Family Structure in Analog Circuit Optimization

**论文链接:** [http://arxiv.org/abs/2512.00712v1](http://arxiv.org/abs/2512.00712v1)

**作者:** Zhuohua Liu, Kaiqi Huang, Qinxin Mei, Yuanqi Hu, Wei W. Xing

**发布时间:** 2025-11-30

### GPT解析

### 总结

该研究提出了一种名为电路先验网络（CPN）的新方法，通过预训练的表格模型编码器件物理的基本原理，实现了在模拟电路优化中的可靠性能，无需针对每个电路进行特定工程调整。

### 背景

模拟电路优化通常被视为在任意平滑函数上的黑盒搜索，但器件物理限制了性能映射到结构化家族（如指数器件定律、有理传递函数和依赖于区域的动态）。现有高斯过程代理模型施加的全局平滑、静止先验与这些区域切换基本原理不一致，在实际样本量下可能导致高度非线性电路的严重拟合问题。

### 目的

展示预训练的表格模型能够编码这些基本原理，实现可靠的优化，无需针对每个电路进行工程调整，并探索从手工制作模型作为先验转向系统物理信息结构识别的可能性。

### 方法

提出电路先验网络（CPN），结合表格基础模型（TabPFN v2）和直接期望改进（DEI）算法，在离散后验而非高斯近似下精确计算期望改进，利用结构匹配的先验进行优化。

### 主要发现

在6个电路和25个基线测试中，结构匹配的先验在小样本区域实现了约0.99的R²值（GP-Matérn仅达0.16），以3.34-11.89倍的迭代次数实现了1.05-3.81倍更高的品质因数（FoM）。

### 结论

研究结果表明，结构匹配的先验方法在模拟电路优化中显著优于传统方法，为从手工制作模型转向系统物理信息结构识别提供了新思路，代码将在论文接受后公开发布。

### 翻译

模拟电路优化通常被视为在任意平滑函数上的黑盒搜索，然而器件物理限制了性能映射到结构化家族：指数器件定律、有理传递函数和依赖于区域的动态。现成的表格模型编码这些基本原理，实现可靠的优化，无需针对每个电路进行工程调整。电路先验网络（CPN）结合表格基础模型（TabPFN v2）和直接期望改进（DEI），在离散后验而非高斯近似下精确计算期望改进。在6个电路和25个基线上，结构匹配的先验在小样本区域实现了约0.99的R²值，而GP-Matérn在Bandgap上仅达到0.16的R²，以3.34-11.89倍的迭代次数实现了1.05-3.81倍更高的品质因数，并提出了从手工制作模型作为先验转向系统物理信息结构识别的转变。我们的代码将在论文接受后公开发布。


### 论文摘要

Analog circuit optimization is typically framed as black-box search over arbitrary smooth functions, yet device physics constrains performance mappings to structured families: exponential device laws, rational transfer functions, and regime-dependent dynamics. Off-the-shelf Gaussian-process surrogates impose globally smooth, stationary priors that are misaligned with these regime-switching primitives and can severely misfit highly nonlinear circuits at realistic sample sizes (50--100 evaluations). We demonstrate that pre-trained tabular models encoding these primitives enable reliable optimization without per-circuit engineering. Circuit Prior Network (CPN) combines a tabular foundation model (TabPFN v2) with Direct Expected Improvement (DEI), computing expected improvement exactly under discrete posteriors rather than Gaussian approximations. Across 6 circuits and 25 baselines, structure-matched priors achieve $R^2 \approx 0.99$ in small-sample regimes where GP-Matérn attains only $R^2 = 0.16$ on Bandgap, deliver $1.05$--$3.81\times$ higher FoM with $3.34$--$11.89\times$ fewer iterations, and suggest a shift from hand-crafting models as priors toward systematic physics-informed structure identification. Our code will be made publicly available upon paper acceptance.

---

## 192. Bridging FR1 to FR3: Frequency-Continuous Urban Macro/Microcellular Channel Parameterization Anchored at 4.85 GHz

**论文链接:** [http://arxiv.org/abs/2512.00707v1](http://arxiv.org/abs/2512.00707v1)

**作者:** Inocent Calist, Minseok Kim

**发布时间:** 2025-11-30

### GPT解析

### 总结

该研究开发了一个统一的无线电信道模型框架，解决了从5G到6G过渡中频率连续性问题，特别是在4-8GHz频段，填补了测量空白，为宽带5G/6G仿真、移动性评估和频谱规划提供了实用基础。

### 背景

从5G到6G的过渡需要覆盖整个FR1-FR3频段的无线电信道模型，特别是在WRC-27关注的4-8GHz频段。现有的3GPP风格模型通常在离散频率上指定，在7.125GHz边界处引入了大尺度参数(LSPs)的不连续性。

### 目的

开发一个频率连续的统一框架，解决现有模型在7.125GHz边界处的不连续性问题，填补4.85GHz附近的测量空白，为宽带5G/6G系统提供实用的信道模型基础。

### 方法

通过在4.85GHz进行严格的双向测量，在三种代表性的UMa/UMi布局中收集数据。使用稳健的距离分箱和bootstrap处理推导特定路线的统计信息，包括路径损耗、延迟和角度扩展、K因子和空间一致性。然后将这些局部统计信息与补充的高频数据集相结合，构建从4.85到28GHz的对数-对数频率连续LSP模型。

### 主要发现

所提出的模型确保了7.125GHz处的平滑性，并与3GPP趋势系统性地偏离，捕捉到了城市宏小区(UMa)中较弱的色散和城市微小区(UMi)中较强的频率依赖压缩。

### 结论

该研究提出的参数集填补了4.85GHz附近的测量空白，为宽带5G/6G仿真、移动性评估和FR1-FR3接口的频谱规划活动提供了实用、可实施的基础。

### 翻译

从5G到6G的过渡需要覆盖整个FR1-FR3频段的无线电信道模型，特别是在WRC-27关注的4-8GHz区域。现有的3GPP风格模型通常在离散频率上指定，在7.125GHz边界处引入了大尺度参数(LSPs)的不连续性。为解决这个问题，我们开发了一个统一框架，以4.85GHz在三种代表性的UMa/UMi布局中进行的严格双向测量为基础。我们使用稳健的距离分箱和bootstrap处理推导出特定路线的路径损耗、延迟和角度扩展、K因子以及空间一致性的统计信息。随后，我们通过将这些局部统计信息锚定到互补的高频数据集，构建了一个从4.85到28GHz的对数-对数频率连续LSP模型。所得模型确保了7.125GHz处的平滑性，并系统性地偏离了3GPP趋势，捕捉到了城市宏小区(UMa)中较弱的色散和城市微小区(UMi)中较强的频率依赖压缩。所提出的参数集填补了4.85GHz附近的测量空白，为宽带5G/6G仿真、移动性评估和FR1-FR3接口的频谱规划活动提供了实用、可实施的基础。


### 论文摘要

The transition from 5G to 6G requires radio channel models that are frequency-continuous across the entire FR1--FR3 span, particularly in the under-explored 4--8 GHz region targeted by WRC-27. Existing 3GPP-style models are often specified at discrete frequencies, introducing discontinuities in large-scale parameters (LSPs) at the 7.125 GHz boundary. To address this, we develop a unified framework anchored by rigorous double-directional measurements at 4.85 GHz in three representative UMa/UMi layouts. We derive route-specific statistics for path loss, delay and angular spreads, K-factor, and spatial consistency using robust distance-binning and bootstrap processing. Subsequently, we construct a log-log frequency-continuous LSP model spanning 4.85 to 28 GHz by anchoring these local statistics to complementary high-frequency datasets. The resulting models ensure smoothness across 7.125 GHz and systematically deviate from 3GPP trends, capturing weaker dispersion in urban macrocells (UMa) and stronger frequency-dependent compaction in urban microcells (UMi). The proposed parameter set fills the measurement gap around 4.85 GHz and provides a practical, implementation-ready basis for wideband 5G/6G simulations, mobility evaluation, and spectrum-planning activities across the FR1--FR3 interface.

---

## 193. HAVEN: Hierarchical Adversary-aware Visibility-Enabled Navigation with Cover Utilization using Deep Transformer Q-Networks

**论文链接:** [http://arxiv.org/abs/2512.00592v1](http://arxiv.org/abs/2512.00592v1)

**作者:** Mihir Chauhan, Damon Conover, Aniket Bera

**发布时间:** 2025-11-29

### GPT解析

### 总结

该研究提出了一种分层导航框架，结合深度Transformer Q网络(DTQN)作为高层子目标选择器和模块化低级控制器，用于在部分可观测环境和遮挡条件下实现安全高效的自主导航。

### 背景

在部分可观测环境中进行自主导航需要智能体超越即时传感器输入、利用遮挡并确保安全地朝目标前进。这些挑战出现在许多机器人领域，从城市驾驶、仓库自动化到国防和监控。传统的路径规划方法和无记忆强化学习在视野有限和有遮挡的情况下常常失败，可能导致不安全或低效的操作。

### 目的

开发一种能够在部分可观测环境和遮挡条件下实现安全、高效自主导航的框架，解决传统方法在有限视野和遮挡情况下的局限性。

### 方法

提出分层导航框架，包括：1)使用深度Transformer Q网络(DTQN)作为高层子目标选择器；2)模块化低级控制器用于路径点执行；3)DTQN处理包含里程计、目标方向、障碍物接近度和可见性提示的任务感知特征短序列历史；4)引入可见性感知的候选生成，包含掩蔽和暴露惩罚机制；5)低层使用势场控制器跟踪所选子目标，确保短时障碍物避免；6)在2D仿真中验证方法，并直接扩展到3D Unity-ROS环境。

### 主要发现

该方法在成功率、安全裕度和到达目标时间方面相比经典规划者和强化学习基线有持续改进；消融研究证实了时间记忆和可见性感知候选设计的重要性；该框架具有可推广性，适用于各种机器人平台。

### 结论

该研究提出了一个在不确定性下安全导航的可推广框架，具有广泛的机器人平台相关性，通过结合深度学习和传统控制方法，有效解决了部分可观测环境中的导航挑战。

### 翻译

部分可观测环境中的自主导航需要智能体超越即时传感器输入，利用遮挡，并在朝目标前进时确保安全。这些挑战出现在许多机器人领域，从城市驾驶和仓库自动化到国防和监控。传统的路径规划方法和无记忆强化学习在视野有限和有遮挡的情况下往往失败，导致不安全或低效的操作。我们提出了一种分层导航框架，将深度Transformer Q网络(DTQN)作为高层子目标选择器与用于路径点执行的模块化低级控制器集成。DTQN处理任务感知特征的短历史序列，包括里程计、目标方向、障碍物接近度和可见性提示，并输出Q值来排序候选子目标。可见性感知的候选生成引入掩蔽和暴露惩罚，奖励使用掩护和预期安全。然后，低层势场控制器跟踪所选子目标，确保平滑的短时障碍物避免。我们在2D仿真中验证了我们的方法，并通过将点云感知投影到相同的特征架构中，直接将其扩展到3D Unity-ROS环境，实现了无需架构更改的迁移。结果表明，在成功率、安全裕度和到达目标时间方面，该方法相比经典规划者和强化学习基线有持续改进，消融研究证实了时间记忆和可见性感知候选设计的价值。这些发现突显了一个在不确定性下安全导航的可推广框架，具有广泛的机器人平台相关性。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决自主机器人在部分可观测环境中的导航问题，特别是在有遮挡和敌方视野约束下的安全导航。这个问题在国防、监视、城市驾驶和仓库自动化等多个领域都非常重要，因为在这些场景中，机器人不仅需要避开障碍物，还需要利用遮挡隐藏自己以避免被敌方检测，否则可能导致任务失败或危险。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者将复杂的导航问题分解为高层子目标选择和低层轨迹执行两个层次。他们借鉴了分层强化学习(HRL)的思想，以及Esslinger等人的深度Transformer Q网络(DTQN)来处理部分可观测性问题。同时，他们采用了经典的势场控制器进行低层轨迹执行，并受到CoverNav等工作的启发关注覆盖物利用。作者的创新在于设计了特定的任务感知特征向量，引入可见性感知的候选子目标生成，以及将2D框架直接扩展到3D环境的能力。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是分层架构、记忆增强、可见性感知和特定特征设计。整体流程为：1)高层使用DTQN处理历史观测信息，从候选子目标中选择最优子目标；2)低层使用势场控制器执行选定的子目标，结合吸引力、排斥力、敌方回避力和逃生行为；3)通过特定奖励函数平衡进展、安全和隐蔽性；4)将2D训练的模型直接应用于3D环境，通过点云投影保持相同的特征架构。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)可见性和敌方感知的子目标选择，设计编码几何、目标进展和敌方可见性的特征向量；2)与低层控制的分层集成，结合DTQN和势场控制器；3)统一的2D-3D扩展能力。相比之前工作，本文显式处理部分可观测性和敌方视野约束，使用Transformer捕获长期时间依赖，专门设计包含敌方可见性线索的特征向量，并通过分层架构减少状态空间复杂性，同时在保持竞争力的同时提高了安全性和隐蔽性。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '这篇论文提出了一种分层导航框架，结合深度Transformer Q网络和势场控制器，使机器人能够在部分可观测和敌方存在的环境中安全高效地导航，同时利用遮挡物来最小化暴露风险。'}


### 论文摘要

Autonomous navigation in partially observable environments requires agents to reason beyond immediate sensor input, exploit occlusion, and ensure safety while progressing toward a goal. These challenges arise in many robotics domains, from urban driving and warehouse automation to defense and surveillance. Classical path planning approaches and memoryless reinforcement learning often fail under limited fields of view (FoVs) and occlusions, committing to unsafe or inefficient maneuvers. We propose a hierarchical navigation framework that integrates a Deep Transformer Q-Network (DTQN) as a high-level subgoal selector with a modular low-level controller for waypoint execution. The DTQN consumes short histories of task-aware features, encoding odometry, goal direction, obstacle proximity, and visibility cues, and outputs Q-values to rank candidate subgoals. Visibility-aware candidate generation introduces masking and exposure penalties, rewarding the use of cover and anticipatory safety. A low-level potential field controller then tracks the selected subgoal, ensuring smooth short-horizon obstacle avoidance. We validate our approach in 2D simulation and extend it directly to a 3D Unity-ROS environment by projecting point-cloud perception into the same feature schema, enabling transfer without architectural changes. Results show consistent improvements over classical planners and RL baselines in success rate, safety margins, and time to goal, with ablations confirming the value of temporal memory and visibility-aware candidate design. These findings highlight a generalizable framework for safe navigation under uncertainty, with broad relevance across robotic platforms.

---

## 194. Cross-Temporal 3D Gaussian Splatting for Sparse-View Guided Scene Update

**论文链接:** [http://arxiv.org/abs/2512.00534v1](http://arxiv.org/abs/2512.00534v1)

**作者:** Zeyuan An, Yanghang Xiao, Zhiying Leng, Frederick W. B. Li, Xiaohui Liang

**发布时间:** 2025-11-29

**备注:** AAAI2026 accepted

### GPT解析

### 总结

本文提出了Cross-Temporal 3D Gaussian Splatting框架，用于从稀疏观测中高效重建和更新跨时间周期的3D场景，支持非连续捕获，能够利用历史先验信息提高重建质量。

### 背景

在计算机视觉中，保持3D场景表示的一致性是一个重大挑战。从稀疏视图更新3D场景对于城市规划、灾害评估和历史遗迹保护等实际应用至关重要，因为这些场景中密集扫描通常不可用或不切实际。

### 目的

开发一种新框架，能够高效重建和更新不同时间周期的3D场景，使用稀疏图像和先前捕获的场景先验信息，支持非连续捕获，实现场景版本控制、跨时间数字孪生和长期空间文档记录。

### 方法

Cross-Temporal 3DGS框架包含三个阶段：1)跨时间相机对齐，用于估计和校准不同时间戳的相机姿态；2)基于干扰的置信度初始化，用于识别不同时间戳之间未变化的区域，从而指导更新；3)渐进式跨时间优化，迭代地将历史先验信息集成到3D场景中以提高重建质量。

### 主要发现

实验结果表明，与基线方法相比，该方法在重建质量和数据效率方面有显著提高。该方法仅使用稀疏图像即可实现时间变化，并可根据需要重建为详细的3D表示。

### 结论

Cross-Temporal 3DGS是一种有前途的解决方案，适用于场景版本控制、跨时间数字孪生和长期空间文档记录，能够从稀疏观测中高效重建和更新3D场景。

### 翻译

在计算机视觉中，随时间保持一致的3D场景表示是一个重大挑战。从稀疏观测更新3D场景对于各种实际应用至关重要，包括城市规划、灾害评估和历史遗迹保护，在这些应用中，密集扫描通常不可用或不切实际。在本文中，我们提出了跨时间3D高斯飞溅，这是一个新框架，用于使用稀疏图像和先前捕获的场景先验信息高效重建和更新不同时间周期的3D场景。我们的方法包括三个阶段：1)跨时间相机对齐，用于估计和校准不同时间戳的相机姿态；2)基于干扰的置信度初始化，用于识别不同时间戳之间未变化的区域，从而指导更新；3)渐进式跨时间优化，迭代地将历史先验信息集成到3D场景中以提高重建质量。我们的方法支持非连续捕获，不仅能够使用新的稀疏视图更新以优化现有场景，还能在当前捕获的帮助下从有限数据中恢复过去的场景。此外，我们展示了这种方法仅使用稀疏图像实现时间变化的潜力，可以根据需要重建为详细的3D表示。实验结果表明，与基线方法相比，重建质量和数据效率有显著提高，使这种方法成为场景版本控制、跨时间数字孪生和长期空间文档记录的有前途的解决方案。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': "这篇论文主要解决如何在不同时间点之间更新3D场景表示的问题，特别是在只有稀疏视图观察的情况下。这个问题在现实中非常重要，因为它能支持城市规划、灾害评估和历史遗迹保护等应用，这些场景中往往无法获得密集扫描数据。此外，它还能实现'按需重建'的长期空间文档和分析模式，对于城市数字孪生、文化遗产保护和灾害响应等应用至关重要。", '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先认识到独立估计不同时间点的相机姿态会导致坐标系错位，因此设计了跨时间相机对齐策略。同时，他们发现直接转移原始3DGS模型会在动态区域引入伪影，于是提出了基于干扰的置信度初始化方法。此外，他们还认识到在稀疏视图条件下3DGS缺乏足够约束会导致伪影，因此设计了渐进式优化策略。作者确实借鉴了现有工作，包括使用DUSt3R进行相机姿态估计，使用Levenberg-Marquardt和ICP算法进行点云配准，以及基于修改的SSIM指标量化区域稳定性。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是利用历史时间点的3D场景作为先验知识，通过自动识别时间点之间的稳定区域，选择性传播可靠的先验信息，并采用渐进式优化策略逐步整合历史先验到当前场景重建中。整体流程分为三个阶段：1)跨时间相机对齐：通过点云配准将不同时间点的场景对齐到统一坐标系；2)基于干扰的置信度初始化：通过评估扰动后的渲染质量变化，生成标记稳定区域和变化区域的置信度图；3)渐进式跨时间优化：使用融合点云初始化模型，结合稀疏视图和高置信度区域进行优化，迭代细化置信度图并最终固定静态区域，优化动态区域。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)首个基于3DGS的框架能从稀疏观察中进行跨时间场景更新；2)设计了新颖的置信度引导优化策略，选择性传播可靠先验同时适应新场景内容；3)构建了基准数据集，在质量、鲁棒性和效率方面显著优于基线方法。相比之前的工作，不同之处在于：与动态3D重建方法不同，不需要密集时间观察和显式运动建模；与变化检测技术不同，能够重建更新的场景级几何结构；与3D场景编辑器不同，不依赖大量人工干预，提供更高的结构准确性；解决了现有方法中缺乏统一框架的问题，能够从稀疏跨时间图像更新3D场景。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '这篇论文提出了Cross-Temporal 3D Gaussian Splatting，一种创新的框架，能够利用稀疏视图和历史先验信息高效、准确地重建和更新跨时间点的3D场景，为长期空间文档和分析提供了实用的数据高效解决方案。'}


### 论文摘要

Maintaining consistent 3D scene representations over time is a significant challenge in computer vision. Updating 3D scenes from sparse-view observations is crucial for various real-world applications, including urban planning, disaster assessment, and historical site preservation, where dense scans are often unavailable or impractical. In this paper, we propose Cross-Temporal 3D Gaussian Splatting (Cross-Temporal 3DGS), a novel framework for efficiently reconstructing and updating 3D scenes across different time periods, using sparse images and previously captured scene priors. Our approach comprises three stages: 1) Cross-temporal camera alignment for estimating and aligning camera poses across different timestamps; 2) Interference-based confidence initialization to identify unchanged regions between timestamps, thereby guiding updates; and 3) Progressive cross-temporal optimization, which iteratively integrates historical prior information into the 3D scene to enhance reconstruction quality. Our method supports non-continuous capture, enabling not only updates using new sparse views to refine existing scenes, but also recovering past scenes from limited data with the help of current captures. Furthermore, we demonstrate the potential of this approach to achieve temporal changes using only sparse images, which can later be reconstructed into detailed 3D representations as needed. Experimental results show significant improvements over baseline methods in reconstruction quality and data efficiency, making this approach a promising solution for scene versioning, cross-temporal digital twins, and long-term spatial documentation.

---

## 195. Data-Driven Multi-Emitter Localization Using Spatially Distributed Power Measurements

**论文链接:** [http://arxiv.org/abs/2512.00510v1](http://arxiv.org/abs/2512.00510v1)

**作者:** H. Nazim Bicer

**发布时间:** 2025-11-29

**备注:** This is a report submitted for 1st year PhD Qualifier. Not Submitted to a conference

### GPT解析

### 总结

该研究提出两种基于卷积神经网络的稀疏采样功率图多发射器检测与定位方法，适用于频谱监测场景，不需要精确时间同步且使用低成本传感器。

### 背景

随着更多设备竞争有限频谱资源，动态频谱共享面临未授权发射器干扰问题，需要快速检测和定位这些干扰源。

### 目的

开发一种使用低成本、分布式传感器进行未授权发射器快速检测和定位的方法，无需精确时间同步。

### 方法

提出两种卷积神经网络方法：第一种是单阶段预测，直接预测存在概率和位置；第二种是两阶段方法，先估计占用图作为中间表示，再定位发射器。使用统一训练目标结合二元交叉熵和坐标回归损失，可处理未知发射器数量，并在模拟环境中训练和评估了约70k参数的小型网络。

### 主要发现

两种方法都能从稀疏测量中定位多个发射器，适用于不同环境；基于logits的两阶段变体在极端传感器稀疏条件下保持竞争力，在某些情况下表现更优。

### 结论

具有统一训练目标的小型CNN可有效部署于频谱监测和定位应用，即使在稀疏采样条件下也能正常工作。

### 翻译

随着更多设备竞争有限的频谱资源，动态频谱共享越来越容易受到未授权发射器的干扰。这促使使用低成本、分布式传感器进行快速检测和定位这些发射器，且这些传感器不需要精确的时间同步。本文提出了两种卷积神经网络方法，用于从稀疏采样功率图中检测和定位多个发射器。第一种方法直接预测存在概率和位置。另一种两阶段方法首先估计占用图作为可解释的中间表示，然后定位发射器。统一的训练目标结合了二元交叉熵和坐标回归损失，可以处理未知的发射器数量。在模拟的自由空间和城市场景中训练和评估了约70k参数的小型网络。实验证明，这两种方法都能从稀疏测量中定位多个发射器，适用于不同环境，基于logits的两阶段变体在极端传感器稀疏条件下保持竞争力，在某些情况下甚至更优。研究表明，具有统一目标的小型CNN可以部署用于频谱监测和定位。


### 论文摘要

With more devices competing for limited spectrum, dynamic spectrum sharing is increasingly vulnerable to interference from unauthorized emitters. This motivates fast detection and localization of these emitters using low-cost, distributed sensors that do not require precise time synchronization. This paper presents two convolutional neural network (CNN) approaches for multi-emitter detection and localization from sparsely sampled power maps. The first method performs single-stage prediction of existence probabilities and positions. The alternative two-stage method first estimates an occupancy map as an interpretable intermediate representation and then localizes emitters. A unified training objective combines binary cross entropy with coordinate regression loss and can handle an unknown emitter count. Small footprint networks, on the order of 70\,k parameters, are trained and evaluated on simulated free-space and urban scenes. Experiments demonstrate that both approaches localize multiple emitters from sparse measurements across diverse environments, with the logits based two-stage variant remaining competitive, and in some cases superior, under extreme sensor sparsity. The findings indicate that small CNNs with a unified objective can be deployed for spectrum monitoring and localization.

---

## 196. PointNet4D: A Lightweight 4D Point Cloud Video Backbone for Online and Offline Perception in Robotic Applications

**论文链接:** [http://arxiv.org/abs/2512.01383v1](http://arxiv.org/abs/2512.01383v1)

**作者:** Yunze Liu, Zifan Wang, Peiran Wu, Jiayang Ao

**发布时间:** 2025-12-01

**备注:** Accepted by WACV2026

### GPT解析

### 总结

研究提出了一种名为PointNet4D的轻量级4D骨干网络，针对在线和离线场景优化，能够高效处理动态4D环境中的点云视频数据。

### 背景

理解动态4D环境（随时间演化的3D空间）对机器人和交互系统至关重要，这些应用需要系统能够实时处理流式点云视频，通常在资源受限的情况下，同时能够利用过去和现在的观察结果。

### 目的

开发一种轻量级的4D骨干网络，解决当前基于时空卷积和Transformer的4D网络计算密集、不适合实时应用的问题。

### 方法

提出PointNet4D，其核心是混合Mamba-Transformer时间融合块，结合了Mamba的高效状态空间建模和Transformer的双向建模能力；同时引入4DMAP，一种帧级掩码自回归预训练策略，用于捕获跨帧的运动线索。

### 主要发现

在7个数据集的9个任务上进行了广泛评估，展示了PointNet4D在不同领域的一致性改进；在RoboTwin和HandoverSim基准测试上，基于PointNet4D构建的4D扩散策略和4D模仿学习系统取得了显著提升。

### 结论

PointNet4D是一种有效的轻量级4D骨干网络，能够高效处理可变长度的在线序列，适用于不同部署场景，为机器人和交互系统提供了强大的4D环境理解能力。

### 翻译

理解随时间演化的3D空间构成的动态4D环境对机器人和交互系统至关重要。这些应用需要系统能够实时处理流式点云视频，通常在资源受限的情况下，同时也能在可能的情况下利用过去和现在的观察结果。然而，当前的4D骨干网络严重依赖于时空卷积和Transformer，这些方法计算密集，不太适合实时应用。我们提出了PointNet4D，一种针对在线和离线场景优化的轻量级4D骨干网络。其核心是混合Mamba-Transformer时间融合块，它结合了Mamba的高效状态空间建模和Transformer的双向建模能力。这使得PointNet4D能够高效处理不同部署场景中可变长度的在线序列。为了增强时间理解，我们引入了4DMAP，一种帧级掩码自回归预训练策略，用于捕获跨帧的运动线索。我们在7个数据集的9个任务上进行了广泛评估，展示了不同领域的一致性改进。我们通过构建两个机器人应用系统进一步证明了PointNet4D的实用性：4D扩散策略和4D模仿学习，在RoboTwin和HandoverSim基准测试上取得了显著提升。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决动态4D环境（随时间变化的3D空间）的理解问题，特别是针对机器人应用中的在线和离线感知任务。这个问题很重要，因为机器人、AR/VR和具身AI等领域需要在资源受限条件下实时处理流式点云视频，既需要在线感知（实时处理当前和历史数据）也需要离线感知（利用完整序列进行回顾分析），而现有方法要么计算量大不适合实时应用，要么无法同时处理这两种场景。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者通过观察现有方法的局限性进行设计：Transformer适合离线场景但计算量大，Mamba适合在线场景但无法利用双向信息。作者选择PointNet++作为特征提取器（平衡性能和效率），设计混合Mamba-Transformer时间融合模块，并开发4DMAP预训练策略。该方法借鉴了PointNet++作为基础特征提取器，受到MambaVision等混合架构的启发，基于MAP预训练框架但针对时间数据调整，并参考了NSM4D处理在线场景的方法但避免了其单独内存模块的复杂性。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是构建一个轻量级、统一的4D主干网络，通过混合Mamba-Transformer时间融合模块，结合Mamba的高效单向建模能力和Transformer的双向推理能力，使网络能够在不同场景中处理可变长度的序列。整体流程：1)使用PointNet++提取每帧空间特征；2)使用混合Mamba-Transformer层处理特征序列捕捉时间动态；3)通过任务特定层进行预测；4)使用4DMAP预训练增强时间理解能力，包括帧级掩码、编码和按帧自回归重建。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点：1)统一的在线/离线4D感知框架；2)混合Mamba-Transformer时间融合模块；3)4DMAP预训练策略；4)实际应用验证（DP4和4DIL）。相比之前工作：与P4Transformer/P4Mamba相比，PointNet4D在两种场景中都表现更好；与NSM4D相比，无需单独内存模块；与LeaF相比，计算开销更小；与VideoMAE相比，4DMAP更适合时间数据特性。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'PointNet4D通过混合Mamba-Transformer架构和4DMAP预训练策略，首次实现了轻量级统一的在线/离线4D点云视频感知，显著提升了机器人在动态环境中的感知能力。'}


### 论文摘要

Understanding dynamic 4D environments-3D space evolving over time-is critical for robotic and interactive systems. These applications demand systems that can process streaming point cloud video in real-time, often under resource constraints, while also benefiting from past and present observations when available. However, current 4D backbone networks rely heavily on spatiotemporal convolutions and Transformers, which are often computationally intensive and poorly suited to real-time applications. We propose PointNet4D, a lightweight 4D backbone optimized for both online and offline settings. At its core is a Hybrid Mamba-Transformer temporal fusion block, which integrates the efficient state-space modeling of Mamba and the bidirectional modeling power of Transformers. This enables PointNet4D to handle variable-length online sequences efficiently across different deployment scenarios. To enhance temporal understanding, we introduce 4DMAP, a frame-wise masked auto-regressive pretraining strategy that captures motion cues across frames. Our extensive evaluations across 9 tasks on 7 datasets, demonstrating consistent improvements across diverse domains. We further demonstrate PointNet4D's utility by building two robotic application systems: 4D Diffusion Policy and 4D Imitation Learning, achieving substantial gains on the RoboTwin and HandoverSim benchmarks.

---

## 197. Data assimilation and discrepancy modeling with shallow recurrent decoders

**论文链接:** [http://arxiv.org/abs/2512.01170v1](http://arxiv.org/abs/2512.01170v1)

**作者:** Yuxuan Bao, J. Nathan Kutz

**发布时间:** 2025-12-01

**备注:** 27 pages, 11 figures

### GPT解析

### 总结

该研究提出了一种名为DA-SHRED的机器学习框架，用于数据同化，旨在弥合计算建模与实验传感器数据之间的差距。该方法通过利用从简化模拟模型中学习的潜在空间，并使用真实传感器数据更新这些潜在变量，实现了复杂物理系统全状态的准确重建。此外，该算法在潜在空间中整合了基于稀疏识别非线性动力学的回归模型，以识别模拟模型中缺失的动力学对应函数。

### 背景

现代传感需求正在迅速发展，对数据效率、实时处理和有限传感覆盖下的部署要求不断提高。复杂物理系统通常通过有限数量的点传感器与科学计算相结合来表征，这些计算近似主导的全状态动力学。然而，模拟模型不可避免地会忽略小规模或隐藏过程，对扰动敏感，或过度简化参数相关性，导致重建结果常常与传感器测量的现实情况偏离。

### 目的

解决数据同化的关键需求，即结合观测数据与预测模拟模型，以产生复杂物理系统全状态的一致且准确的估计，弥合计算建模与实验传感器数据之间的模拟到现实（SIM2REAL）差距。

### 方法

提出DA-SHRED框架，利用SHRED从简化的模拟模型中学习的潜在空间，使用真实传感器数据更新这些潜在变量以准确重建完整系统状态。在潜在空间中结合基于稀疏识别非线性动力学的回归模型，以识别模拟模型中缺失的动力学对应函数。

### 主要发现

DA-SHRED成功弥合了SIM2REAL差距，并在高度复杂系统中额外恢复了缺失的动力学。有效的时间编码与物理信息校正的结合实现了强大的数据同化能力。

### 结论

DA-SHRED框架结合了高效的时间编码和物理信息校正，实现了复杂物理系统的强大数据同化能力，能够准确重建系统状态并恢复模拟模型中缺失的动力学。

### 翻译

现代传感的需求正在迅速发展，受到对数据效率、实时处理和有限传感覆盖下部署需求的推动。复杂物理系统通常通过有限数量的点传感器与科学计算相结合来表征，这些计算近似主导的全状态动力学。然而，模拟模型不可避免地会忽略小规模或隐藏过程，对扰动敏感，或过度简化参数相关性，导致重建结果常常与传感器测量的现实情况偏离。这创造了对数据同化的关键需求，即结合观测数据与预测模拟模型，以产生复杂物理系统全状态的一致且准确的估计。我们提出了一个具有浅层循环解码器的数据同化机器学习框架DA-SHRED，它弥合了计算建模与实验传感器数据之间的模拟到现实（SIM2REAL）差距。对于建模高维时空场且全状态无法直接观测必须从稀疏传感器测量推断的现实物理系统，我们利用通过SHRED从简化模拟模型中学习的潜在空间，并使用真实传感器数据更新这些潜在变量，以准确重建完整系统状态。此外，我们的算法在潜在空间中结合了基于稀疏识别非线性动力学的回归模型，以识别模拟模型中缺失的动力学对应函数。我们证明DA-SHRED成功弥合了SIM2REAL差距，并在高度复杂系统中额外恢复了缺失的动力学，表明有效的时间编码与物理信息校正的结合 enables robust data assimilation。


### 论文摘要

The requirements of modern sensing are rapidly evolving, driven by increasing demands for data efficiency, real-time processing, and deployment under limited sensing coverage. Complex physical systems are often characterized through the integration of a limited number of point sensors in combination with scientific computations which approximate the dominant, full-state dynamics. Simulation models, however, inevitably neglect small-scale or hidden processes, are sensitive to perturbations, or oversimplify parameter correlations, leading to reconstructions that often diverge from the reality measured by sensors. This creates a critical need for data assimilation, the process of integrating observational data with predictive simulation models to produce coherent and accurate estimates of the full state of complex physical systems. We propose a machine learning framework for Data Assimilation with a SHallow REcurrent Decoder (DA-SHRED) which bridges the simulation-to-real (SIM2REAL) gap between computational modeling and experimental sensor data. For real-world physics systems modeling high-dimensional spatiotemporal fields, where the full state cannot be directly observed and must be inferred from sparse sensor measurements, we leverage the latent space learned from a reduced simulation model via SHRED, and update these latent variables using real sensor data to accurately reconstruct the full system state. Furthermore, our algorithm incorporates a sparse identification of nonlinear dynamics based regression model in the latent space to identify functionals corresponding to missing dynamics in the simulation model. We demonstrate that DA-SHRED successfully closes the SIM2REAL gap and additionally recovers missing dynamics in highly complex systems, demonstrating that the combination of efficient temporal encoding and physics-informed correction enables robust data assimilation.

---

## 198. SwiftVLA: Unlocking Spatiotemporal Dynamics for Lightweight VLA Models at Minimal Overhead

**论文链接:** [http://arxiv.org/abs/2512.00903v1](http://arxiv.org/abs/2512.00903v1)

**作者:** Chaojun Ni, Cheng Chen, Xiaofeng Wang, Zheng Zhu, Wenzhao Zheng, Boyuan Wang, Tianrun Chen, Guosheng Zhao, Haoyun Li, Zhehao Dong, Qiang Zhang, Yun Ye, Yang Wang, Guan Huang, Wenjun Mei

**发布时间:** 2025-11-30

### GPT解析

### 总结

SwiftVLA是一种创新的架构，通过4D理解和设计效率提升，解决了VLA模型参数量大、实用性受限的问题，在保持高性能的同时显著减少了计算资源需求。

### 背景

基于预训练VLM的VLA模型潜力巨大但参数量过大，使用轻量级VLM会损害时空推理能力，而结合3D输入的方法通常依赖大型VLM且缺乏时间理解。

### 目的

提出SwiftVLA架构，在保持设计效率的同时增强紧凑模型的4D理解能力，解决VLA模型实用性受限的问题。

### 方法

采用预训练4D视觉几何变换器与时间缓存从2D图像提取4D特征；引入Fusion Tokens通过未来预测目标训练生成统一表示；采用掩码-重建策略使VLM学习有效4D表示，推理时可丢弃4D分支。

### 主要发现

SwiftVLA在真实和模拟环境中优于轻量级基线，性能媲美大7倍的VLA，在边缘设备上运行速度快18倍，内存占用减少12倍。

### 结论

SwiftVLA有效平衡了模型性能与计算效率，为在资源受限设备上部署高性能VLA模型提供了可行解决方案。

### 翻译

基于预训练视觉语言模型(VLM)构建的视觉-语言-动作(VLA)模型显示出强大的潜力，但由于其庞大的参数量而实用性受限。为缓解此问题，已探索使用轻量级VLM，但这会损害时空推理能力。尽管一些方法表明结合额外的3D输入有所帮助，但它们通常依赖大型VLM来融合3D和2D输入，仍然缺乏时间理解。因此，我们提出了SwiftVLA，一种在保持设计效率的同时增强紧凑模型4D理解能力的架构。具体而言，我们的方法采用带有时间缓存的预训练4D视觉几何变换器，从2D图像中提取4D特征。然后，为了增强VLM利用2D图像和4D特征的能力，我们引入了Fusion Tokens，这是一组通过未来预测目标训练的可学习token，用于动作为生成统一表示。最后，我们引入了一种掩码和重建策略，将4D输入掩码到VLM中，并训练VLA重建它们，使VLM能够学习有效的4D表示，并在推理时可以丢弃4D分支而性能损失最小。在真实和模拟环境中的实验表明，SwiftVLA优于轻量级基线，并能与体积大7倍的VLA相媲美，在边缘设备上实现了可比的性能，同时速度快18倍，内存占用减少12倍。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决轻量级视觉-语言-动作（VLA）模型在保持高效推理的同时缺乏时空理解能力的问题。这个问题很重要，因为机器人需要在资源受限的边缘设备上实时运行，大型模型的高延迟和高内存消耗不适合实际部署，而轻量级模型虽然速度快但无法有效捕捉对机器人精确动作规划至关重要的3D空间信息，导致任务成功率低。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先分析了现有方法的局限性：仅使用2D输入的模型时空感知能力有限；直接融合3D特征的方法需要依赖大型VLM；引入专用3D分支的方法参数开销大。然后设计了轻量级解决方案，使用预训练的4D视觉几何变换器提取时空特征，引入Fusion Tokens融合2D和4D特征，采用mask-and-reconstruct策略使模型在推理时可以丢弃4D输入。作者借鉴了现有工作中的预训练4D视觉几何变换器、轻量级VLM作为基础模型、扩散模型用于动作生成以及时间缓存机制处理时序信息。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是通过4D特征增强轻量级VLM的时空理解能力，使用Fusion Tokens实现2D和4D特征的有效融合，采用mask-and-reconstruct策略使模型在训练时学习时空知识，但在推理时可以丢弃4D输入，保持轻量级。整体流程：1)接收多视角图像、语言指令和状态；2)提取2D视觉特征和4D时空特征；3)通过Fusion Tokens融合多种特征；4)预测末端执行器轨迹并生成动作；5)训练时随机掩码特征并要求重建；6)推理时移除4D特征提取器，只保留核心模块。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)4D特征提取与轻量级VLM的有效集成，使用预训练4D视觉几何变换器和时间缓存机制；2)Fusion Tokens设计，融合多模态特征并通过末端执行器轨迹监督；3)Mask-and-reconstruct训练策略，使模型在推理时可丢弃4D输入。相比之前工作，SwiftVLA不需要大型VLM来融合多模态输入，避免了参数开销，同时通过训练策略使模型在推理时只需2D输入，保持了轻量级特性，在性能上媲美参数量多达7倍的大型模型。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'SwiftVLA通过创新的4D特征提取、Fusion Tokens设计和mask-and-reconstruct训练策略，实现了轻量级VLA模型在保持高效推理的同时获得强大的时空理解能力，在边缘设备上实现了18倍的加速和12倍的内存减少。'}


### 论文摘要

Vision-Language-Action (VLA) models built on pretrained Vision-Language Models (VLMs) show strong potential but are limited in practicality due to their large parameter counts. To mitigate this issue, using a lightweight VLM has been explored, but it compromises spatiotemporal reasoning. Although some methods suggest that incorporating additional 3D inputs can help, they usually rely on large VLMs to fuse 3D and 2D inputs and still lack temporal understanding. Therefore, we propose SwiftVLA, an architecture that enhances a compact model with 4D understanding while preserving design efficiency. Specifically, our approach features a pretrained 4D visual geometry transformer with a temporal cache that extracts 4D features from 2D images. Then, to enhance the VLM's ability to exploit both 2D images and 4D features, we introduce Fusion Tokens, a set of learnable tokens trained with a future prediction objective to generate unified representations for action generation. Finally, we introduce a mask-and-reconstruct strategy that masks 4D inputs to the VLM and trains the VLA to reconstruct them, enabling the VLM to learn effective 4D representations and allowing the 4D branch to be dropped at inference with minimal performance loss. Experiments in real and simulated environments show that SwiftVLA outperforms lightweight baselines and rivals VLAs up to 7 times larger, achieving comparable performance on edge devices while being 18 times faster and reducing memory footprint by 12 times.

---

## 199. S^2-KD: Semantic-Spectral Knowledge Distillation Spatiotemporal Forecasting

**论文链接:** [http://arxiv.org/abs/2512.00366v1](http://arxiv.org/abs/2512.00366v1)

**作者:** Wenshuo Wang, Yaomin Shen, Yingjie Tan, Yihao Chen

**发布时间:** 2025-11-29

### GPT解析

### 总结

该论文提出了一种名为S²-KD的新型知识蒸馏框架，结合语义先验和频谱表示，用于时空预测任务，使轻量级学生模型能够实现既频谱准确又语义一致的预测。

### 背景

时空预测通常依赖计算密集型模型捕捉复杂动态。现有知识蒸馏方法如频感知KD虽能保留频谱特性，但受限于像素级信号操作，无法捕捉视觉模式背后的语义和因果上下文。

### 目的

克服现有知识蒸馏方法的局限性，通过统一语义先验和频谱表示，使学生模型学习到既频谱准确又语义一致的预测能力。

### 方法

训练一个特权化的多模态教师模型，利用大型多模态模型的文本叙述推理事件原因，同时在潜在空间解耦频谱成分，并通过新的蒸馏目标将统一的语义-频谱知识转移到轻量级纯视觉学生模型中。

### 主要发现

在WeatherBench和TaxiBJ+等基准测试上，S²-KD显著提高了简单学生模型的性能，使其能够超越最先进方法，特别是在长距离和复杂非平稳场景中表现更优。

### 结论

S²-KD成功将语义和频谱知识整合到轻量级学生模型中，使其在无需额外文本输入或架构开销的情况下，实现高质量预测。

### 翻译

时空预测通常依赖计算密集型模型来捕捉复杂动态。知识蒸馏(KD)已成为创建轻量级学生模型的关键技术，最近的进展如频感知KD成功保留了频谱特性（即高频细节和低频趋势）。然而，这些方法基本受限于在像素级信号上操作，使它们无法看到视觉模式背后丰富的语义和因果上下文。为了克服这一限制，我们引入了S²-KD，这是一个统一语义先验和频谱表示进行蒸馏的新型框架。我们的方法首先训练一个特权的多模态教师模型。该教师模型利用来自大型多模态模型(LMM)的文本叙述来推理事件的基本原因，同时其架构在潜在空间中解耦频谱成分。我们框架的核心是一种新的蒸馏目标，将这种统一的语义-频谱知识转移到轻量级的纯视觉学生中。因此，学生学会做出的预测不仅是频谱准确的，而且是语义连贯的，无需任何文本输入或推理时的架构开销。在WeatherBench和TaxiBJ+等基准上的广泛实验表明，S²-KD显著提高了简单学生模型的性能，使它们能够超越最先进的方法，特别是在长距离和复杂非平稳场景中。


### 论文摘要

Spatiotemporal forecasting often relies on computationally intensive models to capture complex dynamics. Knowledge distillation (KD) has emerged as a key technique for creating lightweight student models, with recent advances like frequency-aware KD successfully preserving spectral properties (i.e., high-frequency details and low-frequency trends). However, these methods are fundamentally constrained by operating on pixel-level signals, leaving them blind to the rich semantic and causal context behind the visual patterns. To overcome this limitation, we introduce S^2-KD, a novel framework that unifies Semantic priors with Spectral representations for distillation. Our approach begins by training a privileged, multimodal teacher model. This teacher leverages textual narratives from a Large Multimodal Model (LMM) to reason about the underlying causes of events, while its architecture simultaneously decouples spectral components in its latent space. The core of our framework is a new distillation objective that transfers this unified semantic-spectral knowledge into a lightweight, vision-only student. Consequently, the student learns to make predictions that are not only spectrally accurate but also semantically coherent, without requiring any textual input or architectural overhead at inference. Extensive experiments on benchmarks like WeatherBench and TaxiBJ+ show that S^2-KD significantly boosts the performance of simple student models, enabling them to outperform state-of-the-art methods, particularly in long-horizon and complex non-stationary scenarios.

---

## 200. New Spiking Architecture for Multi-Modal Decision-Making in Autonomous Vehicles

**论文链接:** [http://arxiv.org/abs/2512.01882v1](http://arxiv.org/abs/2512.01882v1)

**作者:** Aref Ghoreishee, Abhishek Mishra, Lifeng Zhou, John Walsh, Nagarajan Kandasamy

**发布时间:** 2025-12-01

### GPT解析

### 总结

该研究提出了一种端到端的多模态强化学习框架，用于自动驾驶车辆的高层决策，通过尖峰时间感知类Transformer架构解决了多模态融合在资源受限环境中的计算效率问题。

### 背景

Transformer已成为现代多模态架构的骨干，但其高计算成本限制了其在资源受限的边缘环境中的部署。

### 目的

克服Transformer高计算成本的挑战，提出一种计算高效的多模态融合方法，适用于自动驾驶的实时决策。

### 方法

提出了一种尖峰时间感知类Transformer架构，使用三元尖峰神经元进行计算高效的多模态融合。该框架整合了摄像头图像、LiDAR点云和车辆航向信息等异构传感器输入，通过基于交叉注意力的Transformer感知模块进行处理。

### 主要发现

在高速公路环境中的多个任务上的全面评估表明，所提出的方法在实时自动驾驶决策中既有效又高效。

### 结论

该研究成功解决了多模态融合在资源受限环境中的计算效率问题，为自动驾驶提供了实时决策解决方案。

### 翻译

这项工作提出了一种端到端的多模态强化学习框架，用于自动驾驶车辆的高层决策。该框架通过基于交叉注意力的Transformer感知模块整合了异构传感器输入，包括摄像头图像、LiDAR点云和车辆航向信息。尽管Transformer已成为现代多模态架构的骨干，但其高计算成本限制了其在资源受限的边缘环境中的部署。为克服这一挑战，我们提出了一种尖峰时间感知类Transformer架构，使用三元尖峰神经元进行计算高效的多模态融合。在高速公路环境中的多个任务上的全面评估表明，所提出的方法在实时自动驾驶决策中既有效又高效。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决自动驾驶车辆中多模态决策-making的计算效率问题。传统transformer架构虽然能有效融合摄像头、LiDAR等多种传感器数据，但计算成本高，难以部署在资源受限的边缘设备上。这一问题在现实中至关重要，因为自动驾驶系统需要在有限计算资源和能耗下实时处理大量异构数据，确保安全高效的决策。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先认识到传统多模态融合方法在边缘设备上的计算局限性，然后受生物神经元启发的尖峰神经网络吸引，因其提供事件驱动的计算范式。他们分析了现有尖峰注意力机制忽略时间依赖性的缺陷，借鉴了transformer架构的跨注意力设计、尖峰神经网络的基本原理以及深度Q网络用于强化学习的思想，最终将尖峰神经网络与跨注意力机制结合，特别关注时间维度处理。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是使用尖峰神经网络替代传统神经网络，并提出时间感知的三元尖峰注意力机制，通过三元尖峰神经元（-1, 0, 1）而非二元（0, 1）增强表示能力，在保持计算效率的同时捕捉时间依赖性和负值信息。实现流程：1)将多模态传感器数据转换为图像表示；2)用卷积层提取特征；3)将特征投影到嵌入空间；4)使用TTSA机制融合多模态特征；5)通过全连接层生成动作值函数用于强化学习决策。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点：1)提出端到端多模态Q学习网络架构(MM-DQN)融合BEV图像、LiDAR和IMU数据；2)设计时间感知三元尖峰注意力(TTSA)机制；3)首次将尖峰跨注意力融合集成到自动驾驶端到端强化学习中。不同之处：相比传统transformer大幅降低计算成本；相比现有尖峰自注意力机制，TTSA使用三元神经元增强表示能力，考虑时间依赖性，更好处理负值信息；相比单模态方法，在更少观察下实现更好性能；显著缩小尖峰模型与传统模型间的性能差距。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '这篇论文提出了一种基于时间感知三元尖峰注意力机制的高效多模态融合框架，显著降低了自动驾驶决策的计算成本，同时保持了与传统方法相当或更好的性能。'}


### 论文摘要

This work proposes an end-to-end multi-modal reinforcement learning framework for high-level decision-making in autonomous vehicles. The framework integrates heterogeneous sensory input, including camera images, LiDAR point clouds, and vehicle heading information, through a cross-attention transformer-based perception module. Although transformers have become the backbone of modern multi-modal architectures, their high computational cost limits their deployment in resource-constrained edge environments. To overcome this challenge, we propose a spiking temporal-aware transformer-like architecture that uses ternary spiking neurons for computationally efficient multi-modal fusion. Comprehensive evaluations across multiple tasks in the Highway Environment demonstrate the effectiveness and efficiency of the proposed approach for real-time autonomous decision-making.

---

## 201. S$^2$-MLLM: Boosting Spatial Reasoning Capability of MLLMs for 3D Visual Grounding with Structural Guidance

**论文链接:** [http://arxiv.org/abs/2512.01223v1](http://arxiv.org/abs/2512.01223v1)

**作者:** Beining Xu, Siting Zhu, Zhao Jin, Junxian Li, Hesheng Wang

**发布时间:** 2025-12-01

**备注:** 18 pages, 9 figures

### GPT解析

### 总结

本文提出了S²-MLLM框架，通过隐式空间推理增强多模态大语言模型在3D视觉定位任务中的空间推理能力，解决了现有方法效率低下和空间推理有限的问题。

### 背景

3D视觉定位(3DVG)是在3D场景中基于自然语言描述定位物体的基础任务，对具身AI和机器人至关重要。多模态大语言模型(MLLMs)的最新进展激发了将其扩展到3DVG的研究，但MLLMs主要处理2D视觉输入，难以从有限视角理解3D场景空间结构。

### 目的

开发一个高效的框架，通过隐式空间推理增强MLLMs的空间推理能力，解决现有方法依赖视点相关渲染导致的效率低下和空间推理有限问题。

### 方法

提出S²-MLLM框架，引入空间指导策略利用前馈3D重建的结构感知能力；设计结构增强模块(SE)，采用视图内和视图间注意力机制捕获依赖性和对应关系，整合多级位置编码将视觉表示与空间位置和视点信息关联。

### 主要发现

模型通过训练过程获取3D结构理解，可在不依赖低效点云重建的情况下隐式推理3D场景；在多个数据集上实现了显著优于现有方法的性能。

### 结论

S²-MLLM在性能、泛化能力和效率方面表现出色，在ScanRefer、Nr3D和Sr3D数据集上显著优于现有方法，代码将在接受后提供。

### 翻译

3D视觉定位(3DVG)专注于基于自然语言描述在3D场景中定位物体，是具身AI和机器人的基础任务。多模态大语言模型(MLLMs)的最新进展激发了将其扩展到3DVG的研究。然而，MLLMs主要处理2D视觉输入，仅从这些有限视角难以理解3D场景的空间结构。现有方法主要利用重建点云的视点相关渲染为3DVG任务中的MLLMs提供显式结构指导，导致效率低下和空间推理有限。为解决这一问题，我们提出了S²-MLLM，这是一个通过隐式空间推理增强MLLMs空间推理能力的高效框架。我们引入了一种空间指导策略，利用前馈3D重建的结构感知能力。通过在训练过程中获取3D结构理解，我们的模型可以在不依赖低效点云重建的情况下隐式推理3D场景。此外，我们提出了一个结构增强模块(SE)，该模块首先采用视图内和视图间注意力机制来捕获视图内的依赖性和跨视图的对应关系。该模块进一步整合多级位置编码，将视觉表示与空间位置和视点信息关联起来，实现更准确的结构理解。广泛的实验表明，S²-MLLM在性能、泛化能力和效率方面都表现出色，在ScanRefer、Nr3D和Sr3D数据集上显著优于现有方法。代码将在接受后提供。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决多模态大语言模型（MLLMs）在3D视觉定位任务中的空间推理能力不足问题。MLLMs主要处理2D图像数据，难以理解3D场景的空间结构，而这对于具身AI和机器人技术至关重要，因为它们需要根据自然语言描述在3D环境中准确定位物体。这一限制阻碍了AI系统在真实物理世界中的应用。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先分析了现有方法（如通过重建点云并渲染为2D表示）的局限性，包括效率低下和受视角影响等问题。他们提出了一种新思路：通过隐式空间推理而非显式重建来增强MLLM的空间理解能力。该方法借鉴了前馈3D重建技术（如Fast3R）的空间结构理解能力，以及视频建模中的时空分解思想设计了视图内和视图间注意力机制。作者通过端到端联合优化，使模型能够在训练过程中内化3D结构理解，从而在推理时无需显式重建。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是通过隐式空间推理增强MLLMs的3D理解能力，避免推理时的显式点云重建。整体流程包括：1) 接收多视角RGB-D图像、相机参数和自然语言描述；2) 使用视频编码器和位置编码器提取特征；3) 通过结构增强模块（SE）应用视图内和视图间注意力机制，并集成多级位置编码；4) 在训练过程中利用空间引导策略通过前馈3D重建提供结构监督；5) 视频LLM处理视觉和文本输入，预测目标物体的3D边界框和类别；6) 推理时禁用重建分支，仅使用学习到的空间推理能力。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1) 隐式空间推理框架（S2-MLLM），避免推理时的显式重建；2) 空间引导策略，利用前馈3D重建的结构感知能力；3) 结构增强模块（SE），结合视图内/间注意力和多级位置编码。相比之前工作，S2-MLLM在效率上显著提升（无需推理时重建），空间理解更全面（不受视角限制），泛化能力更强（在分布外场景表现更好）。实验表明，该方法在多个数据集上实现了最先进的性能，同时保持了高效率。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'S2-MLLM通过引入隐式空间推理和结构增强模块，显著提升了多模态大语言模型在3D视觉定位任务中的空间推理能力，同时实现了更高的效率和更好的泛化性能。'}


### 论文摘要

3D Visual Grounding (3DVG) focuses on locating objects in 3D scenes based on natural language descriptions, serving as a fundamental task for embodied AI and robotics. Recent advances in Multi-modal Large Language Models (MLLMs) have motivated research into extending them to 3DVG. However, MLLMs primarily process 2D visual inputs and struggle with understanding 3D spatial structure of scenes solely from these limited perspectives. Existing methods mainly utilize viewpoint-dependent rendering of reconstructed point clouds to provide explicit structural guidance for MLLMs in 3DVG tasks, leading to inefficiency and limited spatial reasoning. To address this issue, we propose S$^2$-MLLM, an efficient framework that enhances spatial reasoning in MLLMs through implicit spatial reasoning. We introduce a spatial guidance strategy that leverages the structure awareness of feed-forward 3D reconstruction. By acquiring 3D structural understanding during training, our model can implicitly reason about 3D scenes without relying on inefficient point cloud reconstruction. Moreover, we propose a structure-enhanced module (SE), which first employs intra-view and inter-view attention mechanisms to capture dependencies within views and correspondences across views. The module further integrates multi-level position encoding to associate visual representations with spatial positions and viewpoint information, enabling more accurate structural understanding. Extensive experiments demonstrate that S$^2$-MLLM unifies superior performance, generalization, and efficiency, achieving significant performance over existing methods across the ScanRefer, Nr3D, and Sr3D datasets. Code will be available upon acceptance.

---

## 202. SpeContext: Enabling Efficient Long-context Reasoning with Speculative Context Sparsity in LLMs

**论文链接:** [http://arxiv.org/abs/2512.00722v1](http://arxiv.org/abs/2512.00722v1)

**作者:** Jiaming Xu, Jiayi Pan, Hanzhen Wang, Yongkang Zhou, Jiancai Ye, Yu Wang, Guohao Dai

**发布时间:** 2025-11-30

**备注:** Accepted by ASPLOS 2026

### GPT解析

### 总结

本文提出了一种名为SpeContext的长上下文推理算法和系统协同设计方案，通过利用蒸馏语言模型(DLM)作为检索算法，在算法、系统和编译三个层面进行了优化，显著提高了在资源受限环境中的性能。

### 背景

检索算法的目标与大型语言模型(LLM)对齐，这与LLM中的知识蒸馏目标相似。从信息理论角度看，蒸馏语言模型(DLM)与原始LLM在信息焦点上存在相似性。

### 目的

提出一种利用DLM作为检索算法的新范式，并基于此设计SpeContext系统，以提升长上下文推理的效率和性能。

### 方法

SpeContext在三个层面进行设计：(1)算法层面：提出基于DLM头部注意力权重的轻量级检索头，通过修剪冗余实现>90%的参数减少；(2)系统层面：设计异步预取数据流，通过弹性加载策略将KV缓存检索与LLM计算重叠；(3)编译层面：构建理论内存模型并实现自适应内存管理系统，最大化GPU内存利用率。

### 主要发现

在云和边缘两种资源受限环境中，SpeContext相比Huggingface框架实现了显著性能提升：云环境中吞吐量提高24.89倍，边缘环境中加速10.06倍，且精度损失可忽略不计，推动了精度和吞吐量Pareto前沿的发展。

### 结论

SpeContext通过算法、系统和编译层面的协同设计，有效解决了资源受限环境中的长上下文推理问题，实现了性能与精度的平衡。

### 翻译

在本文中，我们指出检索算法的目标是与大型语言模型对齐，这与LLM中的知识蒸馏目标相似。我们从信息理论角度分析了蒸馏语言模型与原始LLM在信息焦点上的相似性，因此提出了一种利用DLM作为检索算法的新范式。基于这一见解，我们提出了SpeContext，一个用于长上下文推理的算法和系统协同设计方案。(1)在算法层面，SpeContext提出了基于DLM头部注意力权重的轻量级检索头，通过修剪冗余实现了>90%的参数减少。(2)在系统层面，SpeContext通过弹性加载策略设计了异步预取数据流，有效将KV缓存检索与LLM计算重叠。(3)在编译层面，SpeContext构建了理论内存模型并实现了自适应内存管理系统，通过最大化GPU内存利用率实现加速。我们在云和边缘两种资源受限环境中部署和评估了SpeContext。大量实验表明，与Huggingface框架相比，SpeContext在云环境中实现了高达24.89倍的吞吐量提升，在边缘环境中实现了10.06倍的加速，且精度损失可忽略不计，推动了精度和吞吐量Pareto前沿的发展。


### 论文摘要

In this paper, we point out that the objective of the retrieval algorithms is to align with the LLM, which is similar to the objective of knowledge distillation in LLMs. We analyze the similarity in information focus between the distilled language model(DLM) and the original LLM from the perspective of information theory, and thus propose a novel paradigm that leverages a DLM as the retrieval algorithm. Based on the insight, we present SpeContext, an algorithm and system co-design for long-context reasoning. (1) At the algorithm level, SpeContext proposes lightweight retrieval head based on the head-level attention weights of DLM, achieving > 90% parameters reduction by pruning the redundancy. (2) At the system level, SpeContext designs an asynchronous prefetch dataflow via the elastic loading strategy, effectively overlapping KV cache retrieval with the LLM computation. (3) At the compilation level, SpeContext constructs the theoretical memory model and implements an adaptive memory management system to achieve acceleration by maximizing GPU memory utilization. We deploy and evaluate SpeContext in two resourceconstrained environments, cloud and edge. Extensive experiments show that, compared with the Huggingface framework, SpeContext achieves up to 24.89x throughput improvement in cloud and 10.06x speedup in edge with negligible accuracy loss, pushing the Pareto frontier of accuracy and throughput.

---

## 203. SplatFont3D: Structure-Aware Text-to-3D Artistic Font Generation with Part-Level Style Control

**论文链接:** [http://arxiv.org/abs/2512.00413v1](http://arxiv.org/abs/2512.00413v1)

**作者:** Ji Gan, Lingxu Chen, Jiaxu Leng, Xinbo Gao

**发布时间:** 2025-11-29

### GPT解析

### 总结

研究提出SplatFont3D框架，结合3D高斯散射技术实现从文本提示生成3D艺术字体，提供精确的部分级别风格控制。

### 背景

大多数艺术字体生成研究关注2D平面设计，个性化3D艺术字体生成领域探索不足；3D字体具有精确语义和强结构约束，需要细粒度部分级别控制。

### 目的

解决3D艺术字体生成中的挑战，实现精确部分级别风格控制，处理3D字体的语义和结构约束问题。

### 方法

提出SplatFont3D框架，包含Glyph2Cloud模块逐步增强2D字形并生成3D点云，通过预训练2D扩散模型优化3D高斯，采用动态组件分配策略实现部分级别控制。

### 主要发现

SplatFont3D比NeRF提供更明确有效的部分级别风格控制，渲染效率更高；在风格文本一致性、视觉质量和渲染效率方面优于现有3D艺术字体生成模型。

### 结论

SplatFont3D是一种有效的结构感知文本到3D艺术字体生成框架，能从多样化风格文本提示创建3D艺术字体并实现精确的部分级别风格控制。

### 翻译

艺术字体生成(AFG)可以协助人类设计师创建创新的艺术字体。然而，大多数先前的研究主要关注平面设计中的2D艺术字体，而个性化的3D-AFG在很大程度上仍未被探索。3D-AFG不仅能在视频游戏和动画等沉浸式3D环境中实现应用，还可能通过渲染新视角的2D字体来增强2D-AFG。此外，与普通3D对象不同，3D字体表现出精确的语义和强烈的结构约束，同时需要细粒度的部分级别风格控制。为了解决这些挑战，我们提出了SplatFont3D，这是一种基于3D高斯散射的新型结构感知文本到3D AFG框架，能够从多样化的风格文本提示创建3D艺术字体，并实现精确的部分级别风格控制。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决3D艺术字体生成的问题，特别是具有结构感知和部分级别样式控制的3D艺术字体生成。这个问题很重要，因为3D艺术字体比2D艺术字体有更广泛的应用前景，可以在视频游戏、动画和虚拟现实等沉浸式环境中使用，同时还能从任意视角渲染2D字体。此外，3D字体具有精确的语义信息和严格的形状约束，需要保持语义正确性的同时融入准确的风格属性，这比一般3D对象生成更具挑战性。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先认识到3D艺术字体生成面临的三大挑战：语义和风格约束、部分级别样式控制、以及数据获取成本高。作者借鉴了现有工作，包括使用3D高斯溅射(3DGS)提高渲染效率，利用预训练2D扩散模型避免依赖大量3D数据，以及采用Score Distillation Sampling(SDS)优化3D表示。基于这些思考，作者设计了三个主要模块：Glyph2Cloud用于生成初始化3D点云，3D高斯溅射优化模块，以及动态组件分配策略来解决部分级别样式控制问题。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是利用3D高斯溅射的几何优势和预训练2D扩散模型的强先验知识，实现具有精确部分级别样式控制的结构感知3D艺术字体生成。整体流程包括：1)Glyph2Cloud模块通过2D扩散模型生成风格化的2D字体并进行分割，构建用于初始化的3D点云；2)通过Score Distillation Sampling优化3D高斯，使投影的2D图像匹配文本条件；3)采用动态组件分配策略，利用3D高斯的几何先验划分组件，解决优化过程中的漂移问题，实现精确的部分级别样式控制。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)提出SplatFont3D框架，专注于 largely unexplored 的3D艺术字体生成问题；2)引入Glyph2Cloud模块，结合2D字形几何先验和扩散模型，生成高质量的初始化3D点云，避免了对真实3D字体数据的依赖；3)提出动态组件分配策略，实现显式的部分级别样式控制，解决了3D高斯优化过程中的漂移问题；4)实验证明在风格文本一致性、视觉质量和渲染效率方面优于现有方法。相比之前的工作，SplatFont3D专注于3D艺术字体而非一般3D对象，提供了显式的部分级别控制，并且渲染效率显著高于NeRF等传统方法。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'SplatFont3D通过结合3D高斯溅射的几何优势和预训练2D扩散模型的先验知识，实现了具有精确部分级别样式控制的高效3D艺术字体生成，解决了3D字体生成中的语义约束、部分控制和数据稀缺等关键挑战。'}


### 论文摘要

Artistic font generation (AFG) can assist human designers in creating innovative artistic fonts. However, most previous studies primarily focus on 2D artistic fonts in flat design, leaving personalized 3D-AFG largely underexplored. 3D-AFG not only enables applications in immersive 3D environments such as video games and animations, but also may enhance 2D-AFG by rendering 2D fonts of novel views. Moreover, unlike general 3D objects, 3D fonts exhibit precise semantics with strong structural constraints and also demand fine-grained part-level style control. To address these challenges, we propose SplatFont3D, a novel structure-aware text-to-3D AFG framework with 3D Gaussian splatting, which enables the creation of 3D artistic fonts from diverse style text prompts with precise part-level style control. Specifically, we first introduce a Glyph2Cloud module, which progressively enhances both the shapes and styles of 2D glyphs (or components) and produces their corresponding 3D point clouds for Gaussian initialization. The initialized 3D Gaussians are further optimized through interaction with a pretrained 2D diffusion model using score distillation sampling. To enable part-level control, we present a dynamic component assignment strategy that exploits the geometric priors of 3D Gaussians to partition components, while alleviating drift-induced entanglement during 3D Gaussian optimization. Our SplatFont3D provides more explicit and effective part-level style control than NeRF, attaining faster rendering efficiency. Experiments show that our SplatFont3D outperforms existing 3D models for 3D-AFG in style-text consistency, visual quality, and rendering efficiency.

---

## 204. DenoiseGS: Gaussian Reconstruction Model for Burst Denoising

**论文链接:** [http://arxiv.org/abs/2511.22939v2](http://arxiv.org/abs/2511.22939v2)

**作者:** Yongsen Cheng, Yuanhao Cai, Yulun Zhang

**发布时间:** 2025-11-28

**备注:** Update Abstract

### GPT解析

### 总结

DenoiseGS是首个利用3D Gaussian Splatting效率进行burst denoising的框架，通过Gaussian self-consistency损失和log-weighted frequency损失解决了将前馈高斯重建模型应用于噪声输入时的两个关键挑战：高斯点云退化和细节丢失。实验表明，DenoiseGS在burst denoising和噪声条件下的novel view synthesis方面显著优于最先进的NeRF-based方法，同时实现了250倍更快的推理速度。

### 背景

Burst denoising方法对增强手持设备拍摄的图像至关重要，但现有方法通常在大运动情况下表现不佳或计算成本过高。

### 目的

提出DenoiseGS，首个利用3D Gaussian Splatting效率进行burst denoising的框架。

### 方法

1) 提出Gaussian self-consistency (GSC)损失，使用从干净输入生成的高质量高斯点云规范从噪声输入预测的几何形状；2) 引入log-weighted frequency (LWF)损失，在频谱域内加强监督，以对数方式自适应地加权频率差异，强调具有挑战性的高频细节。

### 主要发现

DenoiseGS在burst denoising和噪声条件下的novel view synthesis方面显著超越了最先进的NeRF-based方法，同时实现了250倍更快的推理速度。

### 结论

DenoiseGS是一种高效且有效的burst denoising方法，优于现有方法，代码和模型已公开发布。

### 翻译

Burst denoising方法对于增强手持设备拍摄的图像至关重要，但它们通常在大运动情况下表现不佳或计算成本过高。在本文中，我们提出了DenoiseGS，这是首个利用3D Gaussian Splatting效率进行burst denoising的框架。我们的方法解决了将前馈高斯重建模型应用于噪声输入时的两个关键挑战：高斯点云退化和细节丢失。为此，我们提出了Gaussian self-consistency (GSC)损失，它使用从干净输入生成的相同模型训练的高质量高斯点云来规范从噪声输入预测的几何形状，从而减轻潜在的偏差或域差距。此外，我们引入了log-weighted frequency (LWF)损失，以在频谱域内加强监督，有效保留精细细节。LWF损失以对数方式自适应地加权频率差异，强调具有挑战性的高频细节。大量实验表明，DenoiseGS在burst denoising和噪声条件下的novel view synthesis方面显著超越了最先进的NeRF-based方法，同时实现了250倍的更快推理速度。代码和模型已在https://github.com/yscheng04/DenoiseGS发布。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决手持设备拍摄图像的连拍降噪问题，特别是在处理大运动场景时现有方法效果不佳以及计算成本过高的问题。这个问题在现实中非常重要，因为智能手机摄影普及但传感器尺寸有限，导致图像易受噪声影响；连拍降噪能有效利用多帧信息恢复单帧中丢失的细节，提升图像质量，但现有方法在复杂运动条件下表现不佳或计算效率低下。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先分析了传统2D连拍降噪方法在大运动情况下的局限性，以及基于NeRF的3D方法计算成本高的问题。他们借鉴了3D高斯飞溅(3DGS)的高效特性，特别是前馈高斯模型如GS-LRM，将其作为基础模型。针对直接应用3DGS到连拍降噪时遇到的点云质量下降和细节丢失问题，作者设计了两种创新损失函数：GSC损失利用模型从干净输入生成高质量点云的能力来指导噪声输入的点云重建；LWF损失在频域中自适应强调高频细节。此外，作者还改进了相机条件表示，用RPPC替代标准Plücker射线以提高几何感知能力。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是利用3D高斯飞溅的高效性解决连拍降噪问题，通过GSC损失和LWF损失分别提升点云质量和细节保留能力。整体流程包括：1)接收噪声图像序列和相机条件作为输入；2)使用Transformer块处理输入；3)预测每个像素的高斯参数并计算3D位置；4)通过可微3DGS渲染管道生成目标视图；5)训练时同时输入干净图像生成高质量点云作为指导，应用GSC损失规范噪声输入的点云，用LWF损失保留高频细节；6)推理时只需输入噪声图像，通过前馈网络快速生成降噪结果。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)首次将3D高斯飞溅引入连拍降噪任务；2)提出GSC损失，利用模型自身从干净输入生成高质量点云的能力，避免对外部深度估计网络的依赖；3)提出LWF损失，在频域中自适应对频率差异进行对数加权，强调高频细节；4)使用RPPC替代标准Plücker射线，提高几何感知能力。相比之前工作，DenoiseGS不依赖精确帧对齐或光流估计，能处理大运动场景；比基于NeRF的方法快250倍，实现实时推理；解决了直接应用3DGS时的点云质量下降和细节丢失问题。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'DenoiseGS首次将3D高斯飞溅引入连拍降噪领域，通过创新的GSC和LWF损失函数，实现了比现有方法更高质量的降噪效果，同时保持了极快的推理速度，显著提升了手持设备在低光条件下的图像质量。'}


### 论文摘要

Burst denoising methods are crucial for enhancing images captured on handheld devices, but they often struggle with large motion or suffer from prohibitive computational costs. In this paper, we propose DenoiseGS, the first framework to leverage the efficiency of 3D Gaussian Splatting for burst denoising. Our approach addresses two key challenges when applying feedforward Gaussian reconsturction model to noisy inputs: the degradation of Gaussian point clouds and the loss of fine details. To this end, we propose a Gaussian self-consistency (GSC) loss, which regularizes the geometry predicted from noisy inputs with high-quality Gaussian point clouds. These point clouds are generated from clean inputs by the same model that we are training, thereby alleviating potential bias or domain gaps. Additionally, we introduce a log-weighted frequency (LWF) loss to strengthen supervision within the spectral domain, effectively preserving fine-grained details. The LWF loss adaptively weights frequency discrepancies in a logarithmic manner, emphasizing challenging high-frequency details. Extensive experiments demonstrate that DenoiseGS significantly exceeds the state-of-the-art NeRF-based methods on both burst denoising and novel view synthesis under noisy conditions, while achieving 250$\times$ faster inference speed. Code and models are released at https://github.com/yscheng04/DenoiseGS.

---

## 205. RS-ISRefiner: Towards Better Adapting Vision Foundation Models for Interactive Segmentation of Remote Sensing Images

**论文链接:** [http://arxiv.org/abs/2512.00718v1](http://arxiv.org/abs/2512.00718v1)

**作者:** Deliang Wang, Peng Liu

**发布时间:** 2025-11-30

### GPT解析

### 总结

本文提出了一种名为RS-ISRefiner的新型点击式交互式图像分割框架，专门针对遥感图像设计，解决了现有方法在遥感领域难以推广的问题，通过基于适配器的调优策略、混合注意力机制和改进的概率图调制方案，实现了更高的分割精度、效率和更低的交互成本。

### 背景

交互式图像分割在遥感影像精确实标注中起关键作用，但遥感影像中的物体通常表现出尺度变化、不规则边界和复杂背景。现有的IIS方法主要针对自然图像设计，难以推广到遥感领域，原因是标注数据有限和计算开销大。

### 目的

解决现有IIS方法在遥感领域的局限性，提出一种专门针对遥感图像的点击式IIS框架，提高分割精度和效率。

### 方法

提出了RS-ISRefiner框架，采用基于适配器的调优策略保留视觉基础模型的通用表示能力，同时高效学习遥感特定的空间和边界特征；使用混合注意力机制结合卷积局部建模和基于Transformer的全局推理；改进的概率图调制方案有效整合历史用户交互，实现更稳定的迭代精细化和更高的边界保真度。

### 主要发现

在六个遥感数据集上的实验表明，RS-ISRefiner在分割精度、效率和交互成本方面持续优于最先进的IIS方法。

### 结论

RS-ISRefiner框架的有效性和通用性得到证实，非常适合实际遥感场景中的高质量实例分割。

### 翻译

交互式图像分割在为遥感影像生成精确标注中起着关键作用，其中物体常常表现出尺度变化、不规则边界和复杂背景。然而，现有的IIS方法主要针对自然图像设计，由于标注数据有限和计算开销大，难以推广到遥感领域。为应对这些挑战，我们提出了RS-ISRefiner，一种专为遥感图像设计的新型基于点击的IIS框架。该框架采用基于适配器的调优策略，保留视觉基础模型的通用表示，同时实现对遥感特定的空间和边界特征的高效学习。结合卷积局部建模和基于Transformer的全局推理的混合注意力机制，增强了对抗尺度多样性和场景复杂性的鲁棒性。此外，改进的概率图调制方案有效整合了历史用户交互，产生更稳定的迭代精细化和更高的边界保真度。在包括iSAID、ISPRS Potsdam、SandBar、NWPU、LoveDA Urban和WHUBuilding在内的六个遥感数据集上的综合实验表明，RS-ISRefiner在分割精度、效率和交互成本方面持续优于最先进的IIS方法。这些结果证实了我们框架的有效性和通用性，使其非常适合实际遥感场景中的高质量实例分割。


### 论文摘要

Interactive image segmentation(IIS) plays a critical role in generating precise annotations for remote sensing imagery, where objects often exhibit scale variations, irregular boundaries and complex backgrounds. However, existing IIS methods, primarily designed for natural images, struggle to generalize to remote sensing domains due to limited annotated data and computational overhead. To address these challenges, we proposed RS-ISRefiner, a novel click-based IIS framework tailored for remote sensing images. The framework employs an adapter-based tuning strategy that preserves the general representations of Vision Foundation Models while enabling efficient learning of remote sensing-specific spatial and boundary characteristics. A hybrid attention mechanism integrating convolutional local modeling with Transformer-based global reasoning enhances robustness against scale diversity and scene complexity. Furthermore, an improved probability map modulation scheme effectively incorporates historical user interactions, yielding more stable iterative refinement and higher boundary fidelity. Comprehensive experiments on six remote sensing datasets, including iSAID, ISPRS Potsdam, SandBar, NWPU, LoveDA Urban and WHUBuilding, demonstrate that RS-ISRefiner consistently outperforms state-of-the-art IIS methods in terms of segmentation accuracy, efficiency and interaction cost. These results confirm the effectiveness and generalizability of our framework, making it highly suitable for high-quality instance segmentation in practical remote sensing scenarios.

---

## 206. ManualVLA: A Unified VLA Model for Chain-of-Thought Manual Generation and Robotic Manipulation

**论文链接:** [http://arxiv.org/abs/2512.02013v1](http://arxiv.org/abs/2512.02013v1)

**作者:** Chenyang Gu, Jiaming Liu, Hao Chen, Runzhong Huang, Qingpo Wuwu, Zhuoyang Liu, Xiaoqi Li, Ying Li, Renrui Zhang, Peng Jia, Pheng-Ann Heng, Shanghang Zhang

**发布时间:** 2025-12-01

### GPT解析

### 总结

本文提出了ManualVLA，一个基于Mixture-of-Transformers架构的统一Vision-Language-Action框架，通过多模态手册生成和动作执行的协同工作，解决了现有VLA模型在长时程任务中协调高层规划和精确操作的挑战。

### 背景

Vision-Language-Action (VLA)模型最近在机器人场景理解和操作方面表现出强大的泛化能力，但在面对需要定义目标状态的长时程任务（如乐高组装或物体重新排列）时，仍难以协调高层规划和精确操作。

### 目的

赋予VLA模型从'什么'结果推断'如何'过程的能力，将目标状态转化为可执行的程序。

### 方法

引入ManualVLA框架，配备规划专家生成包含图像、位置提示和文本指令的中间手册；设计ManualCoT推理过程，将手册输入到动作专家中，每个步骤提供明确控制条件和隐式指导；开发基于3D高斯溅射的数字孪生工具包自动生成训练数据。

### 主要发现

ManualVLA在现实世界中表现出强大性能，在乐高组装和物体重新排列任务上，比之前的分层SOTA基线平均成功率高出32%。

### 结论

ManualVLA通过将目标状态转化为可执行程序，并利用多模态手册和ManualCoT推理过程，有效解决了VLA模型在长时程任务中的协调挑战。

### 翻译

视觉-语言-行动（VLA）模型最近出现，在机器人场景理解和操作方面表现出强大的泛化能力。然而，当面临需要定义目标状态的长时程任务，如乐高组装或物体重新排列时，现有的VLA模型在协调高层规划和精确操作方面仍面临挑战。因此，我们旨在赋予VLA模型从'什么'结果推断'如何'过程的能力，将目标状态转化为可执行的程序。在本文中，我们引入了ManualVLA，一个基于混合变换器（MoT）架构构建的统一VLA框架，使多模态手册生成和动作执行能够协同工作。与直接将感官输入映射到动作的先前VLA模型不同，我们首先为ManualV配备一个规划专家，生成包含图像、位置提示和文本指令的中间手册。基于这些多模态手册，我们设计了一个手册思维链（ManualCoT）推理过程，将它们输入到动作专家中，每个手册步骤提供明确的控制条件，而其潜在表示则为精确操作提供隐式指导。为减轻数据收集负担，我们开发了一个基于3D高斯溅射的高保真数字孪生工具包，自动生成规划专家训练的手册数据。ManualVLA在现实世界中表现出强大的性能，在乐高组装和物体重新排列任务上，比之前的分层SOTA基线平均成功率高出32%。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决VLA（视觉-语言-动作）模型在面对长时程任务（如乐高积木组装或物体重新排列）时的挑战，即如何协调高层规划与精确操作。这个问题很重要，因为长时程任务是机器人执行复杂现实任务的基础能力，解决这个问题将使机器人能更自主地完成复杂任务，减少人类干预，对于实现真正的通用机器人助手至关重要。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': "作者受人类行为启发，观察到人类能从目标状态推断中间步骤并执行精确操作。他们反思了现有VLA模型直接映射感官输入到动作的局限性，以及分层方法依赖详细手册或人类演示视频的不足。设计上，他们引入了'手册生成'概念和'思维链'推理过程，采用Mixture-of-Transformers架构整合规划和行动专家。他们借鉴了Janus-Pro作为基础模型，使用3D Gaussian Splatting技术构建数字孪生工具包，并采用扩散模型进行动作建模。", '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是将长时程任务分解为手册生成和基于手册执行动作两个阶段，通过思维链推理将目标状态转化为可执行程序。整体流程：1)接收语言指令、当前状态和目标状态图像；2)规划专家生成多模态手册（文本描述、位置提示和子目标图像）；3)行动专家基于手册执行动作，使用显式思维链（视觉提示）和隐式思维链（共享注意力）指导操作；4)采用三阶段训练策略（行动专家预训练、手册专家预训练、联合微调）。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)统一的VLA框架整合手册生成和动作执行；2)ManualCoT推理过程将手册转化为精确动作；3)跨任务共享注意力机制促进信息交换；4)基于3D Gaussian Splatting的数字孪生工具包。相比之前工作，不同之处在于：从直接映射到间接推理，在单一框架内整合规划与行动能力，减少人工依赖，并具备更好的泛化能力，能处理未见过的目标状态。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'ManualVLA通过引入统一的多模态手册生成与执行框架，使机器人能够自主推断长时程任务的中间步骤，实现了比现有方法高32%的成功率，显著提升了机器人完成复杂目标导向任务的能力。'}


### 论文摘要

Vision-Language-Action (VLA) models have recently emerged, demonstrating strong generalization in robotic scene understanding and manipulation. However, when confronted with long-horizon tasks that require defined goal states, such as LEGO assembly or object rearrangement, existing VLA models still face challenges in coordinating high-level planning with precise manipulation. Therefore, we aim to endow a VLA model with the capability to infer the "how" process from the "what" outcomes, transforming goal states into executable procedures. In this paper, we introduce ManualVLA, a unified VLA framework built upon a Mixture-of-Transformers (MoT) architecture, enabling coherent collaboration between multimodal manual generation and action execution. Unlike prior VLA models that directly map sensory inputs to actions, we first equip ManualVLA with a planning expert that generates intermediate manuals consisting of images, position prompts, and textual instructions. Building upon these multimodal manuals, we design a Manual Chain-of-Thought (ManualCoT) reasoning process that feeds them into the action expert, where each manual step provides explicit control conditions, while its latent representation offers implicit guidance for accurate manipulation. To alleviate the burden of data collection, we develop a high-fidelity digital-twin toolkit based on 3D Gaussian Splatting, which automatically generates manual data for planning expert training. ManualVLA demonstrates strong real-world performance, achieving an average success rate 32% higher than the previous hierarchical SOTA baseline on LEGO assembly and object rearrangement tasks.

---

## 207. AirSim360: A Panoramic Simulation Platform within Drone View

**论文链接:** [http://arxiv.org/abs/2512.02009v1](http://arxiv.org/abs/2512.02009v1)

**作者:** Xian Ge, Yuling Pan, Yuhang Zhang, Xiang Li, Weijun Zhang, Dizhe Zhang, Zhaoliang Wan, Xin Lin, Xiangkai Zhang, Juntao Liang, Jason Li, Wenjie Jiang, Bo Du, Ming-Hsuan Yang, Lu Qi

**发布时间:** 2025-12-01

**备注:** Project Website: https://insta360-research-team.github.io/AirSim360-website/

### GPT解析

### 总结

本文提出了AirSim360，一个用于从空中视角获取全景数据的仿真平台，解决了360度全景理解领域缺乏大规模多样化数据的问题。

### 背景

360度全景理解领域正在获得越来越多的关注，用于推进空间智能发展，但缺乏大规模多样化的数据仍然是一个主要限制。

### 目的

开发一个能够通过无人机进行广泛场景采样的全景数据仿真平台，以支持空间智能研究。

### 方法

AirSim360专注于三个关键方面：用于像素级几何、语义和实体级理解的渲染对齐数据和标记范式；用于建模人类行为的交互式行人感知系统；以及支持导航任务的自动轨迹生成范式。

### 主要发现

收集了超过6万个全景样本，并在各种任务上进行了广泛实验，证明了仿真器的有效性。与现有仿真器不同，这项工作是第一个在全景设置下系统建模4D真实世界的。

### 结论

整个平台，包括工具包、插件和收集的数据集，将在https://insta360-research-team.github.io/AirSim360-website公开提供。

### 翻译

360度全景理解领域正在获得越来越多的关注，用于推进空间智能发展。然而，缺乏大规模和多样化的数据仍然是一个主要限制。在这项工作中，我们提出了AirSim360，一个用于从空中视角获取全景数据的仿真平台，能够通过无人机进行广泛的场景采样。具体来说，AirSim360专注于三个关键方面：用于像素级几何、语义和实体级理解的渲染对齐数据和标记范式；用于建模人类行为的交互式行人感知系统；以及支持导航任务的自动轨迹生成范式。此外，我们收集了超过6万个全景样本，并在各种任务上进行了广泛实验，证明了我们仿真器的有效性。与现有仿真器不同，我们的工作是第一个在全景设置下系统建模4D真实世界的。整个平台，包括工具包、插件和收集的数据集，将在https://insta360-research-team.github.io/AirSim360-website公开提供。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决缺乏大规模、多样化的360度全景数据集的问题，这是推动空间智能发展的主要限制因素。这个问题很重要，因为360度全景理解对于空间智能发展至关重要，可以应用于各种机器人应用如导航中的全向避障，而现有数据集主要针对透视图像设计，全景数据稀缺限制了全景方法的发展。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者受近期模拟平台进步启发，意识到旋转代理在模拟器中捕获全景视图是直接解决方案，但存在计算效率低和真实信号定义不匹配的问题。他们选择无人机作为代理，因其能探索更广泛空间采样更多数据。基于Unreal Engine 5构建平台，借鉴了AirSim、CARLA等现有模拟平台的设计理念，采用equirectangular projection作为全景表示方法，并使用了Minimum Snap轨迹规划方法。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是构建一个基于Unreal Engine的全景模拟平台，支持无人机视角的全向感知，提供完整工具链用于智能数据采集，模拟真实世界中人类行为，并自动生成符合物理规律的飞行轨迹。整体流程包括：1)平台架构构建，包含飞行控制模块、渲染引擎和推理引擎；2)数据采集工具包，包括渲染对齐的数据生成、交互式行人感知系统和自动化轨迹生成；3)大规模数据集收集；4)多任务实验验证。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)首个系统建模4D真实世界的全景模拟平台；2)基于RHI的GPU端纹理复制机制实现快速无缝拼接；3)交互式行人感知系统模拟各种行人行为；4)Minimum Snap轨迹规划实现自动数据采集；5)收集超过60K全景样本的多模态数据集。相比之前工作，AirSim360支持全景图像和视频级全景分割，提供实体级分割和3D关键点标注，支持完整API套件和实时运行时交互，且具有向后兼容性。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'AirSim360是一个基于Unreal Engine的全景无人机模拟平台，提供了高效的全景数据采集工具和交互式行人感知系统，填补了大规模全景数据集的空白，推动了空间智能和全向感知研究的发展。'}


### 论文摘要

The field of 360-degree omnidirectional understanding has been receiving increasing attention for advancing spatial intelligence. However, the lack of large-scale and diverse data remains a major limitation. In this work, we propose AirSim360, a simulation platform for omnidirectional data from aerial viewpoints, enabling wide-ranging scene sampling with drones. Specifically, AirSim360 focuses on three key aspects: a render-aligned data and labeling paradigm for pixel-level geometric, semantic, and entity-level understanding; an interactive pedestrian-aware system for modeling human behavior; and an automated trajectory generation paradigm to support navigation tasks. Furthermore, we collect more than 60K panoramic samples and conduct extensive experiments across various tasks to demonstrate the effectiveness of our simulator. Unlike existing simulators, our work is the first to systematically model the 4D real world under an omnidirectional setting. The entire platform, including the toolkit, plugins, and collected datasets, will be made publicly available at https://insta360-research-team.github.io/AirSim360-website.

---

## 208. Transition from Outside-in to Inside-Out at $z\sim 2$: Evidence from Radial Profiles of Specific Star Formation Rate based on JWST/HST

**论文链接:** [http://arxiv.org/abs/2512.01684v1](http://arxiv.org/abs/2512.01684v1)

**作者:** Jie Song, Enci Wang, Cheng Jia, Cheqiu Lyu, Yangyao Chen, Jinyang Wang, Fujia Li, Weiyu Ding, Guanwen Fang, Xu Kong

**发布时间:** 2025-12-01

**备注:** submitted to ApJS, comments are welcomed

### GPT解析

### 总结

研究结合JWST和HST的高分辨率观测数据，测量了CANDELS场中星系的恒星质量、恒星形成率及多波长形态，并推导出46,313个星系的恒星质量和SFR表面密度轮廓。研究发现高红移星系(z>2.5)的特定SFR呈现负梯度，表明其不能仅通过原位恒星形成增长尺寸，挑战了传统认知；而低红移星系(z<2.0)则呈现正梯度，符合内部向外增长情景。

### 背景

利用JWST和HST的高分辨率观测数据研究CANDELS场中的星系

### 目的

测量星系的恒星质量、恒星形成率(SFRs)和多波长形态

### 方法

基于静止框架1微米形态，推导出46,313个星系的恒星质量和SFR表面密度轮廓

### 主要发现

1) 对于恒星形成星系，结果与先前研究在恒星形成主序和尺寸-质量关系方面表现出良好一致性；2) 在较高红移(z>2.5)，Σ_SFR的中值径向轮廓几乎平行于但略陡于Σ_*轮廓；3) 这导致所有考虑的恒星质量区间中的特定SFR(sSFR)轮廓呈现轻微负梯度；4) 在z<2.0时，sSFR轮廓转变为在较低红移处呈现越来越正的梯度，与内部向外增长情景一致

### 结论

z>2.5的星系不能仅通过原位恒星形成来增长尺寸，这对宇宙正午之后的星系尺寸演变的理解提出了挑战；而低红移星系则呈现正梯度，符合内部向外增长情景

### 翻译

通过结合JWST和HST的高分辨率观测，我们测量了CANDELS场中星系的恒星质量、恒星形成率(SFRs)和多波长形态。此外，基于静止框架1微米的形态，我们为46,313个星系推导了空间分辨的恒星质量和SFR表面密度轮廓，这些星系在0<z<4范围内具有可靠的结构测量，并提供了相应的目录。对于恒星形成星系(SFGs)，我们的结果在恒星形成主序和尺寸-质量关系方面与先前研究表现出极好的一致性，证明了我们恒星质量和SFR测量的稳健性。对于空间分辨的轮廓，我们发现在高红移(z>2.5)时，Σ_SFR的中值径向轮廓几乎平行于但略陡于Σ_*轮廓。这导致在所有考虑的恒星质量区间中，特定SFR(sSFR)轮廓呈现轻微负梯度。这些发现表明，z>2.5的星系不能仅通过原位恒星形成来增长尺寸，这对宇宙正午之后的星系尺寸演变的理解提出了挑战。相比之下，在z<2.0时，sSFR轮廓转变为在较低红移处呈现越来越正的梯度，与恒星形成优先扩展星系外围的内部向外增长情景一致。


### 论文摘要

By combining high-resolution observations from JWST and HST, we have measured the stellar masses, star formation rates (SFRs), and multi-wavelength morphologies of galaxies in the CANDELS fields. Furthermore, based on rest-frame 1 $μ$m morphologies, we have derived spatially resolved stellar mass and SFR surface density ($Σ_*$ and $Σ_{\rm SFR}$) profiles for 46,313 galaxies with reliable structural measurements at $0<z<4$ and $\log(M_\ast /M_{\odot})>8$, and provide the corresponding catalogue. For star-forming galaxies (SFGs), our results show excellent consistency with previous studies in terms of the star formation main sequence and the size-mass relation, demonstrating the robustness of our stellar mass and SFR measurements. For spatially resolved profiles, we find that at higher redshifts ($z>2.5$), the median radial profile of $Σ_{\rm SFR}$ is nearly parallel to but slightly steeper than that of $Σ_*$. This results in mildly negative gradients in the specific SFR (sSFR) profiles across all stellar mass bins considered. These findings indicate that galaxies at $z>2.5$ cannot grow in size via only in-situ star formation, challenging the understanding of galaxy size evolution beyond the cosmic noon. In contrast, at $z<2.0$, the sSFR profiles transition to exhibit more and more positive gradients at lower redshifts, consistent with an inside-out growth scenario where star formation preferentially expands the galactic outskirts.

---

## 209. SPARK: Sim-ready Part-level Articulated Reconstruction with VLM Knowledge

**论文链接:** [http://arxiv.org/abs/2512.01629v1](http://arxiv.org/abs/2512.01629v1)

**作者:** Yumeng He, Ying Jiang, Jiayin Lu, Yin Yang, Chenfanfu Jiang

**发布时间:** 2025-12-01

### GPT解析

### 总结

SPARK是一种从单张RGB图像重建物理一致的运动学部件级关节式对象的框架，通过VLMs提取参数、生成参考图像，并利用生成式扩散变换器合成一致形状，最终优化URDF参数以创建高质量的模拟资产。

### 背景

关节式3D对象对具身人工智能、机器人和交互式场景理解至关重要，但创建模拟准备好的资产仍然是劳动密集型的，需要专家对部件层次结构和运动结构进行建模。

### 目的

介绍SPARK框架，用于从单张RGB图像重建物理一致的运动学部件级关节式对象。

### 方法

利用VLMs提取粗略的URDF参数并生成部件级参考图像；将部件图像指导和推断的结构图集成到生成式扩散变换器中合成关节式对象的部件一致形状和完整形状；结合可微分前向运动学和可微分渲染优化关节类型、轴和原点，在VLM生成的开放状态监督下。

### 主要发现

大量实验表明，SPARK能够跨不同类别生成高质量的、模拟准备好的关节式资产。

### 结论

SPARK能够支持下游应用，如机器人操作和交互建模。

### 翻译

关节式3D对象对具身人工智能、机器人和交互式场景理解至关重要，但创建模拟准备好的资产仍然是劳动密集型的，需要专家对部件层次结构和运动结构进行建模。我们介绍了SPARK，这是一个从单张RGB图像重建物理一致的运动学部件级关节式对象的框架。给定输入图像，我们首先利用VLMs提取粗略的URDF参数并生成部件级参考图像。然后，我们将部件图像指导和推断的结构图集成到生成式扩散变换器中，以合成关节式对象的部件一致形状和完整形状。为了进一步优化URDF参数，我们结合了可微分前向运动学和可微分渲染，在VLM生成的开放状态监督下优化关节类型、轴和原点。大量实验表明，SPARK能够在不同类别中产生高质量的、模拟准备好的关节式资产，支持机器人操作和交互建模等下游应用。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决如何从单个RGB图像中重建高质量、可模拟的关节式3D对象的问题。这个问题很重要，因为关节式物体在日常环境中无处不在，创建高保真、可交互的3D资产对具身AI、机器人和交互场景理解至关重要，但当前创建过程仍很费力且需要专业知识。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者结合了多种现有技术：利用视觉语言模型(VLMs)提取结构信息，借鉴扩散模型进行3D生成，采用部件级生成方法处理多部件合成，并应用可微分优化技术。这些方法分别来自Articulate-Anything、TripoSG、PartCrafter等领域的工作，作者将它们创新性地整合成一个统一的框架，专门针对关节式对象重建的特殊挑战。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是利用VLM提供的先验知识指导关节式对象重建，结合部件级图像和结构图，通过生成模型合成一致几何形状，并优化关节参数。流程分为三步：1) VLM引导结构推理，提取URDF参数和生成部件参考图像；2) 使用扩散变压器结合多级注意力机制生成部件级和完整关节式网格；3) 通过可微分正向运动学和渲染优化关节参数。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1) 新框架结合VLM先验和扩散变压器生成高质量关节式对象及URDF参数；2) 部件图像指导和多级注意力实现一致多部件合成；3) 关节优化组件在VLM指导下优化参数。相比之前工作，SPARK无需多视图图像或模板网格，能从单张图像重建完整关节式对象，生成的部件是可分离、可操作的，而非融合的整体形状。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'SPARK通过结合视觉语言模型、扩散模型和可微分优化技术，实现了从单张图像中重建高质量、可模拟的关节式3D对象，为机器人操作和交互建模提供了新解决方案。'}


### 论文摘要

Articulated 3D objects are critical for embodied AI, robotics, and interactive scene understanding, yet creating simulation-ready assets remains labor-intensive and requires expert modeling of part hierarchies and motion structures. We introduce SPARK, a framework for reconstructing physically consistent, kinematic part-level articulated objects from a single RGB image. Given an input image, we first leverage VLMs to extract coarse URDF parameters and generate part-level reference images. We then integrate the part-image guidance and the inferred structure graph into a generative diffusion transformer to synthesize consistent part and complete shapes of articulated objects. To further refine the URDF parameters, we incorporate differentiable forward kinematics and differentiable rendering to optimize joint types, axes, and origins under VLM-generated open-state supervision. Extensive experiments show that SPARK produces high-quality, simulation-ready articulated assets across diverse categories, enabling downstream applications such as robotic manipulation and interaction modeling.

---

## 210. Fixed Points in Quantum Metric Spaces: A Structural Advantage over Fuzzy Frameworks

**论文链接:** [http://arxiv.org/abs/2512.01583v1](http://arxiv.org/abs/2512.01583v1)

**作者:** Nicola Fabiano

**发布时间:** 2025-12-01

**备注:** 7 pages

### GPT解析

### 总结

本研究在量子度量空间框架下证明了收缩映射不动点的存在性和唯一性定理，适用于归一化实值高斯波函数。研究通过比较量子度量空间与模糊度量空间，揭示了量子框架在结构上的一致性和与希尔伯特空间几何丰富性的兼容性，并将模糊逻辑批判扩展到内在不确定性下的动态推理。

### 背景

量子度量空间和模糊度量空间是研究不动点定理的重要框架。模糊度量空间虽然承认类似的不动点定理，但在量子特性方面存在局限，如缺乏干涉、相位敏感性和拓扑保护。

### 目的

证明量子度量空间中收缩映射不动点的存在性和唯一性，比较量子度量空间与模糊度量空间的差异，揭示量子框架的结构优势，并将模糊逻辑批判扩展到动态推理领域。

### 方法

在量子度量空间框架下，使用L²范数定义可区分性，研究归一化实值高斯波函数在保持函数形式的连续收缩演化下的性质。

### 主要发现

量子度量空间框架下的收缩映射存在唯一不动点；量子框架具有更深层次的结构一致性，与希尔伯特空间的几何丰富性相容；量子特性如干涉、相位敏感性和拓扑保护在模糊度量空间中缺失。

### 结论

量子度量空间不仅在技术上优越于模糊度量空间，而且在结构上与希尔伯特空间的几何丰富性相容，为内在不确定性下的动态推理提供了更合适的框架。

### 翻译

我们在量子度量空间的框架下证明了收缩映射不动点的存在性和唯一性定理，其中可区分性由L²范数定义。该结果适用于在保持函数形式的连续收缩演化下的归一化实值高斯波函数。相比之下，虽然模糊度量空间也承认类似的不动点定理，但它们缺乏干涉、相位敏感性和拓扑保护。这种比较揭示了量子框架中更深层次的结构一致性——不仅仅是技术上的优越性，还与希尔伯特空间的几何丰富性相容。我们的工作将模糊逻辑的批判扩展到了内在不确定性下的动态推理。


### 论文摘要

We prove an existence and uniqueness theorem for fixed points of contraction maps in the framework of quantum metric spaces, where distinguishability is defined by the $L^2$ norm: $d_Q(ψ_1,ψ_2) = \|ψ_1 - ψ_2\|$. The result applies to normalized real-valued Gaussian wavefunctions under continuous contractive evolution preserving the functional form. In contrast, while fuzzy metric spaces admit analogous fixed point theorems, they lack interference, phase sensitivity, and topological protection. This comparison reveals a deeper structural coherence in the quantum framework -- not merely technical superiority, but compatibility with the geometric richness of Hilbert space. Our work extends the critique of fuzzy logic into dynamical reasoning under intrinsic uncertainty.

---

## 211. Euclid: The first statistical census of dusty and massive objects in the ERO/Perseus field

**论文链接:** [http://arxiv.org/abs/2512.01489v1](http://arxiv.org/abs/2512.01489v1)

**作者:** G. Girardi, A. Grazian, G. Rodighiero, L. Bisigello, G. Gandolfi, E. Bañados, S. Belladitta, J. R. Weaver, S. Eales, C. C. Lovell, K. I. Caputi, A. Enia, A. Bianchetti, E. Dalla Bontà, T. Saifollahi, A. Vietri, N. Aghanim, B. Altieri, S. Andreon, N. Auricchio, H. Aussel, C. Baccigalupi, M. Baldi, A. Balestra, S. Bardelli, P. Battaglia, A. Biviano, E. Branchini, M. Brescia, J. Brinchmann, S. Camera, G. Cañas-Herrera, V. Capobianco, C. Carbone, J. Carretero, S. Casas, M. Castellano, G. Castignani, S. Cavuoti, K. C. Chambers, A. Cimatti, C. Colodro-Conde, G. Congedo, C. J. Conselice, L. Conversi, Y. Copin, F. Courbin, H. M. Courtois, M. Cropper, A. Da Silva, H. Degaudenzi, G. De Lucia, A. M. Di Giorgio, H. Dole, M. Douspis, F. Dubath, C. A. J. Duncan, X. Dupac, S. Dusini, S. Escoffier, M. Farina, R. Farinelli, F. Faustini, S. Ferriol, S. Fotopoulou, M. Frailis, E. Franceschi, M. Fumana, S. Galeotta, K. George, B. Gillis, C. Giocoli, J. Gracia-Carpio, F. Grupp, S. V. H. Haugan, J. Hoar, W. Holmes, I. M. Hook, F. Hormuth, A. Hornstrup, P. Hudelot, K. Jahnke, M. Jhabvala, E. Keihänen, S. Kermiche, A. Kiessling, B. Kubik, M. Kümmel, M. Kunz, H. Kurki-Suonio, A. M. C. Le Brun, D. Le Mignant, P. Liebing, S. Ligori, P. B. Lilje, V. Lindholm, I. Lloro, G. Mainetti, D. Maino, E. Maiorano, O. Mansutti, S. Marcin, O. Marggraf, M. Martinelli, N. Martinet, F. Marulli, R. Massey, S. Maurogordato, E. Medinaceli, S. Mei, Y. Mellier, M. Meneghetti, E. Merlin, G. Meylan, A. Mora, M. Moresco, L. Moscardini, R. Nakajima, C. Neissner, R. C. Nichol, S. -M. Niemi, C. Padilla, S. Paltani, F. Pasian, K. Pedersen, W. J. Percival, V. Pettorino, G. Polenta, M. Poncet, L. A. Popa, L. Pozzetti, F. Raison, R. Rebolo, A. Renzi, J. Rhodes, G. Riccio, E. Romelli, M. Roncarelli, E. Rossetti, B. Rusholme, R. Saglia, Z. Sakr, D. Sapone, B. Sartoris, J. A. Schewtschenko, P. Schneider, T. Schrabback, A. Secroun, G. Seidel, M. Seiffert, S. Serrano, P. Simon, C. Sirignano, G. Sirri, L. Stanco, J. Steinwagner, P. Tallada-Crespí, D. Tavagnacco, A. N. Taylor, I. Tereno, R. Toledo-Moreo, F. Torradeflot, I. Tutusaus, L. Valenziano, J. Valiviita, T. Vassallo, G. Verdoes Kleijn, A. Veropalumbo, Y. Wang, J. Weller, G. Zamorani, F. M. Zerbi, E. Zucca, M. Bolzonella, C. Burigana, L. Gabarra, J. Martín-Fleitas, V. Scottez

**发布时间:** 2025-12-01

**备注:** 23 pages, 8 figures. Accepted for publication in A\&A

### GPT解析

### 总结

本研究利用Euclid早期释放观测数据，在英仙座领域发现了42个稳健的HIEROs星系样本，通过SED拟合分析计算了3.5<z<5.5范围内的星系恒星质量函数，发现高质量端与之前研究相当，表明真实数量可能更高，强调了研究被尘埃隐藏的恒星形成群体的重要性。

### 背景

我们对z>3恒星形成历史的理解主要依赖于静止帧紫外观测，但这种观测方式会遗漏最尘埃化和大质量的星系，导致早期宇宙的星系普查不完整。红外设施如斯皮策望远镜和詹姆斯·韦伯太空望远镜已经在z=3-6范围内发现了一个具有极端红色特征的隐藏星系群体，被称为HIEROs，其识别标准是H_E-ch2>2.25。

### 目的

利用Euclid早期释放观测数据，结合辅助的Spitzer/IRAC成像，进一步研究这些具有极端红色特征的星系群体，评估它们在宇宙恒星形成率密度中的作用，以及它们与星系形成模型的一致性。

### 方法

在英仙座领域232平方角分区域使用VIS和NISP测光数据，结合四个斯皮策通道和地面MegaCam波段数据。应用颜色标准筛选出121个HIEROs，通过多波段切图视觉检查排除球状星团、褐矮星和不可靠案例后，获得42个稳健的HIEROs样本。使用SED拟合代码Bagpipes估算光测红移和物理性质，并计算星系恒星质量函数。

### 主要发现

即使排除了可能的AGN宿主系统或恒星质量可能被高估的星系，高质量端仍与之前的研究结果相当，表明真实数量可能更高。这些结果突显了被尘埃隐藏的恒星形成群体在宇宙恒星形成率密度中的重要作用。

### 结论

研究这种被尘埃隐藏的恒星形成群体对于评估它们在宇宙恒星形成率密度中的作用以及与星系形成模型一致性的重要性，展示了Euclid望远镜推进我们对早期宇宙尘埃隐藏恒星形成理解的能力。

### 翻译

我们对z>3恒星形成历史的理解主要依赖于静止帧紫外观测，但这种观测方式会遗漏最尘埃化和大质量的星系，导致早期宇宙的星系普查不完整。红外设施如斯皮策望远镜和詹姆斯·韦伯太空望远镜已经在z=3-6范围内发现了一个具有极端红色特征的隐藏星系群体，被称为HIEROs，通过H_E-ch2>2.25的标准识别。最近，Euclid早期释放观测使得通过比较Euclid数据与辅助的Spitzer/IRAC成像能够进一步研究这类天体成为可能。我们使用VIS和NISP测光数据，结合四个斯皮策通道和地面MegaCam波段，在英仙座领域研究了一个232平方角分的区域。应用颜色标准筛选出121个HIEROs；通过多波段切图视觉检查排除球状星团、褐矮星和不可靠案例后，我们获得42个稳健的HIEROs的最终样本。使用SED拟合代码Bagpipes估算光测红移和物理性质。从得到的光测红移和恒星质量值，我们计算了3.5<z<5.5范围内的星系恒星质量函数。即使排除了可能的AGN宿主系统或恒星质量可能被高估的系统，高质量端仍与之前的研究结果相当，表明真实数量可能更高。这些结果突显了进一步研究这种被隐藏群体以评估其在宇宙恒星形成率密度中的作用及其与星系形成模型一致性的重要性，展示了Euclid望远镜推进我们对跨早期时代尘埃隐藏恒星形成理解的能力。


### 论文摘要

Our comprehension of the history of star formation at $z>3$ relies on rest-frame UV observations, yet this selection misses the most dusty and massive sources, yielding an incomplete census at early times. Infrared facilities such as Spitzer and the James Webb Space Telescope have revealed a hidden population at $z=3$-$6$ with extreme red colours, named HIEROs (HST-to-IRAC extremely red objects), identified by the criterion $H_{\mathrm{E}}-\mathrm{ch2}>2.25$. Recently, Euclid Early Release Observations (ERO) have made it possible to further study such objects by comparing Euclid data with ancillary Spitzer/IRAC imaging. We investigate a $232$ arcmin$^2$ area in the Perseus field using VIS and NISP photometry, complemented by the four Spitzer channels and ground-based MegaCam bands ($u$, $g$, $r$, ${\rm H}α$, $i$, $z$). Applying the colour cut yields $121$ HIEROs; after removing globular clusters, brown dwarfs, and unreliable cases through visual inspection of multiband cutouts, we obtain a final sample of $42$ robust HIEROs. Photometric redshifts and physical properties are estimated with the SED-fitting code Bagpipes. From the resulting $z_{\mathrm{phot}}$ and $M_*$ values, we compute the galaxy stellar mass function at $3.5<z<5.5$. Even after excluding possible AGN hosts or systems where the stellar mass may be overestimated, the high-mass end remains comparable to previous determinations, suggesting the true abundance could be higher. These results highlight the importance of further study of this obscured population to assess its role in the cosmic star-formation rate density and its consistency with galaxy-formation models, demonstrating Euclid's capability to advance our understanding of dust-hidden star formation across early epochs.

---

## 212. Rice-VL: Evaluating Vision-Language Models for Cultural Understanding Across ASEAN Countries

**论文链接:** [http://arxiv.org/abs/2512.01419v1](http://arxiv.org/abs/2512.01419v1)

**作者:** Tushar Pranav, Eshan Pandey, Austria Lyka Diane Bala, Aman Chadha, Indriyati Atmosukarto, Donny Soh Cheng Lock

**发布时间:** 2025-12-01

**备注:** 14 pages

### GPT解析

### 总结

本研究提出RICE-VL基准和SEA-LAVE指标，评估视觉语言模型在东南亚文化理解方面的能力，发现模型在资源匮乏国家和抽象文化领域存在显著性能差距。

### 背景

视觉语言模型在多模态任务中表现出色，但往往存在西方中心偏见，限制了模型在东南亚等文化多元化地区的有效性。

### 目的

引入RICE-VL基准，评估VLM对11个东盟国家的文化理解能力，解决西方中心偏见问题。

### 方法

创建包含28,000个人类策划的视觉问答样本(覆盖真/假、填空和开放式格式)和1,000个图像-边界框对用于视觉定位的基准；提出SEA-LAVE指标，扩展LAVE指标以评估文本准确性、文化一致性和国家识别能力。

### 主要发现

六个开源和闭源VLM在资源匮乏国家和抽象文化领域存在显著性能差距；视觉定位任务测试了模型在复杂场景中定位文化重要元素的能力；RICE-VL揭示了VLMs文化理解的局限性。

### 结论

需要包容性模型开发以更好地服务多元化的全球人口。

### 翻译

视觉语言模型在多模态任务中表现出色，但往往存在西方中心偏见，限制了它们在东南亚等文化多元化地区的有效性。为此，我们引入了RICE-VL，这是一个新颖的基准，用于评估VLM对11个东盟国家的文化理解能力。RICE-VL包含超过28,000个人类策划的视觉问答样本——涵盖真/假、填空和开放式格式——以及1,000个用于视觉定位的图像-边界框对，由14个子类别的文化专家进行标注。我们提出了SEA-LAVE，作为LAVE指标的扩展，评估文本准确性、文化一致性和国家识别能力。对六个开源和闭源VLM的评估显示，在资源匮乏国家和抽象文化领域存在显著的性能差距。视觉定位任务测试了模型在复杂场景中定位文化重要元素的能力，探索空间和上下文准确性。RICE-VL揭示了VLMs文化理解的局限性，并强调了包容性模型开发的必要性，以更好地服务多元化的全球人口。


### 论文摘要

Vision-Language Models (VLMs) excel in multimodal tasks but often exhibit Western-centric biases, limiting their effectiveness in culturally diverse regions like Southeast Asia (SEA). To address this, we introduce RICE-VL, a novel benchmark evaluating VLM cultural understanding across 11 ASEAN countries. RICE-VL includes over 28,000 human-curated Visual Question Answering (VQA) samples -- covering True or False, Fill-in-the-Blank, and open-ended formats -- and 1,000 image-bounding box pairs for Visual Grounding, annotated by culturally informed experts across 14 sub-ground categories. We propose SEA-LAVE, an extension of the LAVE metric, assessing textual accuracy, cultural alignment, and country identification. Evaluations of six open- and closed-source VLMs reveal significant performance gaps in low-resource countries and abstract cultural domains. The Visual Grounding task tests models' ability to localize culturally significant elements in complex scenes, probing spatial and contextual accuracy. RICE-VL exposes limitations in VLMs' cultural comprehension and highlights the need for inclusive model development to better serve diverse global populations.

---

## 213. RoboDriveVLM: A Novel Benchmark and Baseline towards Robust Vision-Language Models for Autonomous Driving

**论文链接:** [http://arxiv.org/abs/2512.01300v1](http://arxiv.org/abs/2512.01300v1)

**作者:** Dacheng Liao, Mengshi Qi, Peng Shu, Zhining Zhang, Yuxin Lin, Liang Liu, Huadong Ma

**发布时间:** 2025-12-01

### GPT解析

### 总结

本文提出了首个评估视觉-语言模型在端到端自动驾驶系统中鲁棒性的基准测试RoboDriveBench，以及相应的增强框架RoboDriveVLM和测试时适应方法，旨在提高自动驾驶系统在真实世界场景中的可靠性。

### 背景

当前基于视觉-语言模型的端到端自动驾驶系统利用大型语言模型根据场景理解直接生成驾驶决策，但在真实驾驶场景中存在多种风险。

### 目的

评估VLM是否真正适用于自动驾驶，并开发提高系统鲁棒性的方法。

### 方法

创建了RoboDriveBench基准测试，包含11个模拟场景评估传感器损坏和提示损坏两类挑战；提出RoboDriveVLM框架将多模态数据映射到统一潜在空间；引入基于跨模态知识蒸馏的测试时适应方法提高鲁棒性。

### 主要发现

当前VLM端到端自动驾驶系统在真实世界挑战下存在明显局限性，需要改进以提高可靠性。

### 结论

通过提出的框架和方法，可以显著提高VLM自动驾驶系统在复杂真实场景中的鲁棒性和可靠性，为实际部署提供更可靠的解决方案。

### 翻译

当前基于视觉-语言模型的端到端自动驾驶系统通常利用大型语言模型根据对当前场景的理解直接生成驾驶决策。然而，这类系统在真实驾驶场景中会引入多种风险。为了评估VLM是否真正适用于自动驾驶，我们引入了RoboDriveBench，这是首个专注于端到端轨迹预测任务的鲁棒性基准测试。该基准通过11个模拟场景系统评估了VLM端到端自动驾驶系统面临的两个关键类别的现实挑战，包括6种由环境变化引起的传感器损坏场景和5种由人为干预和数据传输故障导致的提示损坏场景。每种损坏类型包含250个独特的驾驶场景和5,689帧，每次评估总共包含64,559个轨迹预测案例。为了克服这些现实挑战，我们提出了名为RoboDriveVLM的新型VLM自动驾驶框架，通过将更多多模态数据(如激光雷达和雷达)映射到统一的潜在空间来增强鲁棒性。此外，我们引入了一种基于跨模态知识蒸馏的新型测试时适应方法，以提高VLM自动驾驶系统的鲁棒性。通过大量实验，我们的工作突显了当前VLM端到端自动驾驶系统的局限性，并为实际部署提供了更可靠的解决方案。源代码和数据集将公开发布。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决基于视觉-语言模型(VLM)的端到端自动驾驶系统在现实世界场景中的鲁棒性问题。这个问题在现实中非常重要，因为自动驾驶系统的鲁棒性直接关系到行车安全，系统在面对各种干扰和挑战时表现不稳定可能导致严重事故。在研究中也很重要，因为随着VLM在自动驾驶中的应用越来越广泛，评估这些系统在真实世界条件下的鲁棒性变得至关重要。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先分析了当前VLM-based端到端自动驾驶系统的局限性，包括语义理解缺乏底层视觉证据、轨迹预测不确定性高、面对复杂多模态数据时鲁棒性不足等问题。作者设计了一个系统性的方法：创建RoboDriveBench基准测试、提出RoboDriveVLM框架和基于跨模态知识蒸馏的TTA方法。作者借鉴了现有工作，如ImageNet-C等图像鲁棒性测试基准、DriveLM和OpenEMMA等VLM在自动驾驶中的应用，以及测试时适应(TTA)的概念，但针对自动驾驶场景进行了改进。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是通过多模态数据融合和跨模态知识蒸馏来增强VLM-based自动驾驶系统在现实世界复杂场景中的鲁棒性。整体实现流程包括：1)收集并处理来自不同传感器(摄像头、激光雷达、雷达)的数据；2)将激光雷达和雷达数据投影到统一坐标系并转换为鸟瞰图(BEV)图像；3)通过提示工程和多任务微调方法融合不同模态信息；4)使用VLM模型基于融合的特征预测未来轨迹；5)在不同模态间进行轨迹蒸馏，实现推理时的模态解耦，增强模型鲁棒性。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)提出RoboDriveBench基准测试，系统评估VLM-based自动驾驶系统面临的现实挑战；2)提出RoboDriveVLM框架，有效融合激光雷达和雷达模态；3)引入基于跨模态知识蒸馏的测试时适应(TTA)方法；4)提出新的评估指标MCC和MCL2，考虑VLM输出中的不确定性。相比之前的工作，不同之处在于：现有基准测试未能充分考虑VLM特性；当前系统通常只处理图像和文本模态，而RoboDriveVLM融合了更多模态；针对VLM在提示损坏场景中的脆弱性提出专门解决方案；新评估指标更准确反映VLM在实际应用中的表现。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '这篇论文通过提出RoboDriveBench基准测试、RoboDriveVLM多模态框架和跨模态知识蒸馏的TTA方法，显著提高了视觉-语言模型在自动驾驶系统中的鲁棒性和安全性，为自动驾驶技术在真实世界环境中的部署提供了更可靠的解决方案。'}


### 论文摘要

Current Vision-Language Model (VLM)-based end-to-end autonomous driving systems often leverage large language models to generate driving decisions directly based on their understanding of the current scene. However, such systems introduce multiple risks in real-world driving scenarios. To evaluate whether VLMs are truly viable for autonomous driving, we introduce RoboDriveBench, the first robustness benchmark focused on end-to-end trajectory prediction tasks. This benchmark systematically evaluates two critical categories of real-world challenges for VLM-based end-to-end autonomous driving systems through 11 simulated scenarios encompassing various corruption types, including 6 scenarios of sensor corruption caused by environmental variations, along with 5 cases of prompt corruption resulting from human intervention and data transmission failures. Each corruption type includes 250 unique driving scenarios and 5,689 frames, resulting in 64,559 total trajectory prediction cases per evaluation. To overcome these real-world challenges, we propose a novel VLM-based autonomous driving framework called RoboDriveVLM, which enhances robustness by mapping more multimodal data-e.g., lidar and radar-into a unified latent space. Furthermore, we introduce a new Test-Time Adaptation (TTA) method based on cross-modal knowledge distillation to improve the robustness of VLM-based autonomous driving systems. Through extensive experiments, our work highlights the limitations of current VLM-based end-to-end autonomous driving systems and provides a more reliable solution for real-world deployment. Source code and datasets will be released.

---

## 214. Generative Adversarial Gumbel MCTS for Abstract Visual Composition Generation

**论文链接:** [http://arxiv.org/abs/2512.01242v1](http://arxiv.org/abs/2512.01242v1)

**作者:** Zirui Zhao, Boye Niu, David Hsu, Wee Sun Lee

**发布时间:** 2025-12-01

### GPT解析

### 总结

研究抽象视觉合成，提出约束引导框架结合几何推理和神经语义，在七巧板组装任务中表现优异

### 背景

抽象视觉合成中身份主要由几何基元的空间配置决定，从固定组件在几何约束下组合这些结构存在组合放置选择、有限数据和离散可行性等问题

### 目的

开发一种能够处理组合放置选择、有限数据和离散可行性问题的框架，提高抽象视觉合成结果的有效性和语义保真度

### 方法

提出约束引导框架，结合AlphaGo风格搜索强制执行可行性，微调视觉语言模型评估语义一致性，使用策略网络作为蒙特卡洛树搜索启发式，通过搜索生成的计划微调网络，并采用对抗性奖励 refinement

### 主要发现

在七巧板组装任务中，该方法比扩散和自回归基线模型产生更高的有效性和语义保真度，特别是在约束严格的情况下

### 结论

约束引导框架结合显式几何推理和神经语义的方法在抽象视觉合成任务中表现优异

### 翻译

我们研究抽象视觉合成，其中身份主要由少量几何基元（如部分、对称性、拓扑）之间的空间配置和关系决定。它们主要对纹理和逼真细节具有不变性。在几何约束和模糊目标规范（如文本）下从固定组件组合这些结构是困难的，因为存在组合放置选择、有限数据和离散可行性（无重叠、允许方向）等问题，这导致了解决方案流形稀疏，不适合纯统计像素空间生成器。我们提出了一种约束引导的框架，结合显式几何推理和神经语义。类似AlphaGo的搜索强制执行可行性，而微调的视觉语言模型将语义一致性作为奖励信号进行评分。我们的算法使用策略网络作为蒙特卡洛树搜索中的启发式方法，并通过搜索生成的计划微调网络。受生成对抗网络的启发，我们使用生成的实例进行对抗性奖励细化。随着时间的推移，当奖励模型无法区分生成的实例和真实数据时，生成应更接近实际数据。在七巧板组装任务中，我们的方法比扩散和自回归基线模型产生了更高的有效性和语义保真度，特别是在约束更加严格的情况下。


### 论文摘要

We study abstract visual composition, in which identity is primarily determined by the spatial configuration and relations among a small set of geometric primitives (e.g., parts, symmetry, topology). They are invariant primarily to texture and photorealistic detail. Composing such structures from fixed components under geometric constraints and vague goal specification (such as text) is non-trivial due to combinatorial placement choices, limited data, and discrete feasibility (overlap-free, allowable orientations), which create a sparse solution manifold ill-suited to purely statistical pixel-space generators. We propose a constraint-guided framework that combines explicit geometric reasoning with neural semantics. An AlphaGo-style search enforces feasibility, while a fine-tuned vision-language model scores semantic alignment as reward signals. Our algorithm uses a policy network as a heuristic in Monte-Carlo Tree Search and fine-tunes the network via search-generated plans. Inspired by the Generative Adversarial Network, we use the generated instances for adversarial reward refinement. Over time, the generation should approach the actual data more closely when the reward model cannot distinguish between generated instances and ground-truth. In the Tangram Assembly task, our approach yields higher validity and semantic fidelity than diffusion and auto-regressive baselines, especially as constraints tighten.

---

## 215. Near-field perturbation of laser filament enabling simultaneous far-field THz diagnosis and broadband calculus processing

**论文链接:** [http://arxiv.org/abs/2512.01215v1](http://arxiv.org/abs/2512.01215v1)

**作者:** Jiayu Zhao, Yifu Tian, Linlin Yuan, Jiajun Yang, Xiaofeng Li, Li Lao, Alexander Shkurinov, Yan Peng, Yiming Zhu

**发布时间:** 2025-12-01

### GPT解析

### 总结

该研究提出了一种基于飞秒激光诱导空气电离产生的激光丝-等离子体通道的太赫兹波操控方法，通过引入非侵入式近场调制方案，实现了对太赫兹模式的诊断和新型全光学信号处理。

### 背景

基于飞秒激光诱导空气电离产生的激光丝-等离子体通道的太赫兹波操控已成为自由空间太赫兹应用的有前景的平台。然而，由于等离子体超高强度，对丝内空间受限太赫兹模式的原位表征面临重大挑战，这不仅阻碍了直接近场探测，也限制了对间接远场重建的依赖。

### 目的

开发一种能够诊断近场太赫兹模式限制的方法，结合近场调制效率和远场检测的稳健性，推进对等离子体-太赫兹相互作用的基本理解，并为基于丝的太赫兹技术实现新型全光学信号处理。

### 方法

引入了一种非侵入式近场调制方案，其中金属板以亚毫米距离（相当于太赫兹波长）接近丝，通过扰动介电环境将对称环形太赫兹模式转换为非对称状态，从而实现远场检测宽带演算行为和特征谱传递函数。

### 主要发现

通过控制对称环形太赫兹模式向非对称状态的转换，能够诊断近场太赫兹模式的限制，并实现宽带演算行为和具有特定频率依赖性的特征谱传递函数的远场检测。

### 结论

所提出的方法结合了近场调制效率和远场检测稳健性，推进了等离子体-太赫兹相互作用的基本理解，并为基于丝的太赫兹技术实现了新型全光学信号处理。

### 翻译

基于飞秒激光诱导空气电离产生的激光丝-等离子体通道的太赫兹波操控已成为自由空间太赫兹应用的有前景的平台。然而，由于等离子体超高强度，对丝内空间受限太赫兹模式的原位表征面临重大挑战，这不仅阻碍了直接近场探测，也限制了对间接远场重建的依赖。在此，我们引入了一种非侵入式近场调制方案，其中金属板以亚毫米距离（相当于太赫兹波长）接近丝，通过扰动介电环境将对称环形太赫兹模式转换为非对称状态。这种受控转换使得能够远场检测时域太赫兹波形上的宽带演算行为以及具有特定频率依赖性的特征谱传递函数，从而诊断近场太赫兹模式的限制。因此，所提出的方法结合了近场调制效率和远场检测稳健性，推进了等离子体-太赫兹相互作用的基本理解，并为基于丝的太赫兹技术实现了新型全光学信号处理。


### 论文摘要

Terahertz (THz) wave manipulation based on laser filaments-plasma channels formed by femtosecond laser-induced air ionization-has emerged as a promising platform for free-space THz applications. However, in-situ characterization of the spatially confined THz modes within filaments faces significant challenges due to the plasma's ultra-high intensity, which not only hinders direct near-field probing but also limits reliance on indirect far-field reconstruction. Here, we introduce a non-invasive near-field modulation scheme where a metal plate approaches the filament at submillimeter distances (comparable to THz wavelengths), perturbing the dielectric environment to convert the symmetric annular THz mode into an asymmetric state. This controlled transition enables far-field detection of broadband calculus behaviors (first- and second-order differentiation/integration) on time-domain THz waveforms and characteristic spectral transfer functions with 1/f, 1/f^2, f or f^2 dependency (where f is the THz frequency), thereby diagnosing the near-field THz mode confinement. Hence, the proposed approach synergizes near-field modulation efficiency with far-field detection robustness, advancing fundamental understanding of plasma-THz interactions and enabling novel all-optical signal processing for filament-based THz technologies.

---

## 216. CycliST: A Video Language Model Benchmark for Reasoning on Cyclical State Transitions

**论文链接:** [http://arxiv.org/abs/2512.01095v1](http://arxiv.org/abs/2512.01095v1)

**作者:** Simon Kohaut, Daniel Ochs, Shun Zhang, Benedict Flade, Julian Eggert, Kristian Kersting, Devendra Singh Dhami

**发布时间:** 2025-11-30

### GPT解析

### 总结

CycliST是一个新型基准数据集，用于评估视频语言模型在循环状态转换方面的文本推理能力，通过生成具有周期性模式的合成视频序列来测试模型性能。

### 背景

现有视频语言模型在理解现实世界中的周期性过程方面存在局限性，需要专门的数据集来评估它们在循环状态转换方面的能力。

### 目的

评估当前视频语言模型对周期性模式的理解和推理能力，揭示其在时空认知方面的技术差距。

### 方法

CycliST采用分层评估系统，通过改变循环物体数量、场景杂乱程度和光照条件逐步增加难度，对当前最先进的开源和专有VLM进行全面实验。

### 主要发现

现有VLM难以可靠检测和利用循环模式，缺乏时间理解概念，无法提取定量洞察；没有单一模型在所有任务上表现一致，模型大小和架构与性能无强相关性。

### 结论

CycliST为开发能更好理解周期性模式的视觉推理模型提供了基础，有助于推动视频语言模型在时空认知方面的发展。

### 翻译

我们提出了CycliST，这是一个新颖的基准数据集，旨在评估视频语言模型(VLM)在循环状态转换方面的文本推理能力。CycliST通过生成合成的、结构丰富的视频序列来捕捉现实世界过程的基本方面，这些视频序列中物体的运动和视觉属性具有周期性模式。CycliST采用分层评估系统，通过改变循环物体数量、场景杂乱程度和光照条件逐步增加难度，挑战最先进模型在时空认知方面的能力。我们对当前最先进的VLM进行了广泛实验，包括开源和专有模型，揭示了它们在推广到循环动力学（如线性和轨道运动）方面的局限性，以及在视觉属性（如颜色和比例）时依赖性变化方面的不足。我们的结果表明，当今的VLM难以可靠地检测和利用循环模式，缺乏时间理解概念，无法从场景中提取定量洞察（如运动物体的数量），突显了需要解决的重大技术差距。更具体地说，我们发现没有单一模型在性能上始终领先：模型大小和架构与结果没有强相关性，没有模型在所有任务上都同样成功。通过提供有针对性的挑战和全面的评估框架，CycliST为超越最先进水平、理解周期性模式的视觉推理模型铺平了道路。


### 论文摘要

We present CycliST, a novel benchmark dataset designed to evaluate Video Language Models (VLM) on their ability for textual reasoning over cyclical state transitions. CycliST captures fundamental aspects of real-world processes by generating synthetic, richly structured video sequences featuring periodic patterns in object motion and visual attributes. CycliST employs a tiered evaluation system that progressively increases difficulty through variations in the number of cyclic objects, scene clutter, and lighting conditions, challenging state-of-the-art models on their spatio-temporal cognition. We conduct extensive experiments with current state-of-the-art VLMs, both open-source and proprietary, and reveal their limitations in generalizing to cyclical dynamics such as linear and orbital motion, as well as time-dependent changes in visual attributes like color and scale. Our results demonstrate that present-day VLMs struggle to reliably detect and exploit cyclic patterns, lack a notion of temporal understanding, and are unable to extract quantitative insights from scenes, such as the number of objects in motion, highlighting a significant technical gap that needs to be addressed. More specifically, we find no single model consistently leads in performance: neither size nor architecture correlates strongly with outcomes, and no model succeeds equally well across all tasks. By providing a targeted challenge and a comprehensive evaluation framework, CycliST paves the way for visual reasoning models that surpass the state-of-the-art in understanding periodic patterns.

---

## 217. Lotus-2: Advancing Geometric Dense Prediction with Powerful Image Generative Model

**论文链接:** [http://arxiv.org/abs/2512.01030v1](http://arxiv.org/abs/2512.01030v1)

**作者:** Jing He, Haodong Li, Mingzhi Sheng, Ying-Cong Chen

**发布时间:** 2025-11-30

**备注:** Work done at the Hong Kong University of Science and Technology (Guangzhou). Project page: https://lotus-2.github.io/. 15 Pages, 12 Figures, 3 Tables

### GPT解析

### 总结

Lotus-2是一个两阶段确定性框架，通过充分利用预训练的生成先验，实现了稳定、准确和细粒度的几何密集预测，在单目深度估计和表面法线预测任务上取得了最先进的结果。

### 背景

从单幅图像恢复像素级几何属性是根本不适定问题，由于外观模糊性和2D观测与3D结构间的非单射映射导致。判别回归模型虽表现良好但受限于数据规模和质量，扩散模型虽包含强大世界先验但随机生成公式不适合确定性几何推理。

### 目的

提出Lotus-2框架，提供最佳适应协议以充分利用预训练的生成先验，实现稳定、准确和细粒度的几何密集预测。

### 方法

第一阶段：核心预测器采用单步确定性公式和轻量级局部连续性模块生成全局连贯结构；第二阶段：细节锐化器在核心预测器定义的流形内进行约束多步修正流细化，通过确定性流匹配增强细粒度几何。

### 主要发现

仅使用59K训练样本（不到现有大规模数据集的1%），Lotus-2在单目深度估计方面建立了新的最先进结果，在表面法线预测方面也具有高度竞争力。

### 结论

扩散模型可以作为确定性世界先验，实现超越传统判别和生成范式的高质量几何推理。

### 翻译

从单幅图像中恢复像素级几何属性从根本上是不适定问题，这是由于外观模糊性和2D观测与3D结构之间的非单射映射。虽然判别回归模型通过大规模监督实现了强大性能，但其成功受限于可用数据的规模、质量和多样性以及有限的物理推理能力。最近的扩散模型表现出强大的世界先验，这些先验是从大规模图像-文本数据中学习到的几何和语义，然而直接重用它们的随机生成公式对于确定性几何推理不是最优的：前者针对多样化和高保真图像生成进行优化，而后者需要稳定和准确的预测。在这项工作中，我们提出了Lotus-2，一个用于稳定、准确和细粒度几何密集预测的两阶段确定性框架，旨在提供最佳适应协议以充分利用预训练的生成先验。具体而言，在第一阶段，核心预测器采用单步确定性公式，结合干净数据目标和轻量级局部连续性模块(LCM)，生成无网格伪影的全局连贯结构。在第二阶段，细节锐化器在核心预测器定义的流形内执行约束多步修正流细化，通过无噪声确定性流匹配增强细粒度几何。仅使用59K训练样本（不到现有大规模数据集的1%），Lotus-2在单目深度估计方面建立了新的最先进结果，在表面法线预测方面也具有高度竞争力。这些结果表明，扩散模型可以作为确定性世界先验，实现超越传统判别和生成范式的高质量几何推理。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决从单张图像中恢复像素级几何属性（如深度、表面法线）的问题，这是一个本质上不适定的问题，因为2D图像到3D结构的映射存在歧义和非单射性。这个问题在计算机视觉中至关重要，它是现代视觉理解的基础，广泛应用于可控图像生成、3D/4D重建和自动驾驶等领域，使机器能够从单目图像中理解物理世界的几何结构。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先分析了现有方法的局限性：判别性回归模型受限于数据规模和质量，而扩散模型直接重用其随机生成公式对于确定性几何推理存在根本性不匹配。作者借鉴了扩散模型（如FLUX）强大的世界先验能力，但重新思考了其角色，将其视为结构化的世界先验而非随机生成器。设计上，作者系统分析了随机生成公式的关键设计（随机性、多步采样、参数化类型和局部连续性），并针对性地提出了改进方案，构建了两阶段确定性框架：核心预测器和细节锐化器。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是将预训练的扩散模型从随机图像生成器转变为结构化的世界先验，通过确定性两阶段框架实现准确且高保真的几何密集预测。整体流程：1) 输入图像通过编码器进入VAE潜空间；2) 第一阶段核心预测器使用单步确定性公式和干净数据预测生成准确但粗糙的几何结构，结合局部连续模块消除网格伪影；3) 第二阶段细节锐化器在核心预测器定义的流形内执行受约束的多步修正流细化，增强高频细节；4) 最终结果通过解码器输出到像素空间。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点：1) 重新定义扩散模型在密集预测中的角色，从随机生成器转变为结构化世界先验；2) 提出两阶段确定性框架，解耦结构预测和细节细化；3) 系统分析并改进随机生成公式，提出单步确定性公式、干净数据预测和局部连续模块；4) 仅使用59K训练样本（不到现有数据集的1%），实现最先进性能。相比之前工作：不同于直接使用随机生成公式的方法（如Marigold），Lotus-2采用确定性方法确保结构一致性；不同于单步加速方法（如Diffusion-E2E-FT），它同时实现结构准确性和细节保真度；不同于粗到细策略（如StableNormal），其第二阶段完全确定性，避免了几何推理的稳定性问题。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'Lotus-2通过将扩散模型重新定义为确定性世界先验并采用两阶段框架，在极少训练数据下实现了几何密集预测的最先进性能，同时保证了结构准确性和细节保真度。'}


### 论文摘要

Recovering pixel-wise geometric properties from a single image is fundamentally ill-posed due to appearance ambiguity and non-injective mappings between 2D observations and 3D structures. While discriminative regression models achieve strong performance through large-scale supervision, their success is bounded by the scale, quality and diversity of available data and limited physical reasoning. Recent diffusion models exhibit powerful world priors that encode geometry and semantics learned from massive image-text data, yet directly reusing their stochastic generative formulation is suboptimal for deterministic geometric inference: the former is optimized for diverse and high-fidelity image generation, whereas the latter requires stable and accurate predictions. In this work, we propose Lotus-2, a two-stage deterministic framework for stable, accurate and fine-grained geometric dense prediction, aiming to provide an optimal adaption protocol to fully exploit the pre-trained generative priors. Specifically, in the first stage, the core predictor employs a single-step deterministic formulation with a clean-data objective and a lightweight local continuity module (LCM) to generate globally coherent structures without grid artifacts. In the second stage, the detail sharpener performs a constrained multi-step rectified-flow refinement within the manifold defined by the core predictor, enhancing fine-grained geometry through noise-free deterministic flow matching. Using only 59K training samples, less than 1% of existing large-scale datasets, Lotus-2 establishes new state-of-the-art results in monocular depth estimation and highly competitive surface normal prediction. These results demonstrate that diffusion models can serve as deterministic world priors, enabling high-quality geometric reasoning beyond traditional discriminative and generative paradigms.

---

## 218. FOM-Nav: Frontier-Object Maps for Object Goal Navigation

**论文链接:** [http://arxiv.org/abs/2512.01009v1](http://arxiv.org/abs/2512.01009v1)

**作者:** Thomas Chabal, Shizhe Chen, Jean Ponce, Cordelia Schmid

**发布时间:** 2025-11-30

**备注:** Project page: https://www.di.ens.fr/willow/research/fom-nav/

### GPT解析

### 总结

本文提出了FOM-Nav框架，通过结合边界-物体地图和视觉语言模型来解决物体目标导航问题，在未知环境中高效寻找目标物体，并在多个基准测试上取得了最先进性能

### 背景

现有基于隐式记忆的方法在长期记忆保留和规划方面存在困难，而基于显式地图的方法缺乏丰富的语义信息，限制了物体目标导航的效率

### 目的

开发一种能够提高机器人探索效率的物体目标导航方法，解决现有方法的局限性

### 方法

提出FOM-Nav模块化框架，包含在线构建的边界-物体地图(联合编码空间边界和细粒度物体信息)、视觉语言模型进行多模态场景理解和高层目标预测、低级规划器执行高效轨迹生成，并从真实世界扫描环境自动构建大规模导航数据集进行训练

### 主要发现

FOM-Nav在MP3D和HM3D基准测试上取得最先进性能，特别是在导航效率指标SPL上表现突出，在真实机器人上也展示了有希望的结果

### 结论

FOM-Nav框架有效解决了现有物体目标导航方法的局限性，显著提高了导航效率，并在多种环境中验证了其有效性

### 翻译

这篇论文解决了物体目标导航问题，即机器人在未知环境中高效寻找目标物体的问题。现有的基于隐式记忆的方法在长期记忆保留和规划方面存在困难，而基于显式地图的方法缺乏丰富的语义信息。为了应对这些挑战，我们提出了FOM-Nav，一个通过边界-物体地图和视觉语言模型提高探索效率的模块化框架。我们的边界-物体地图是在线构建的，联合编码空间边界和细粒度物体信息。使用这种表示，视觉语言模型执行多模态场景理解和高层目标预测，由低级规划器执行以实现高效轨迹生成。为了训练FOM-Nav，我们从真实世界扫描环境中自动构建了大规模导航数据集。大量实验验证了我们模型设计和构建数据集的有效性。FOM-Nav在MP3D和HM3D基准测试上取得了最先进的性能，特别是在导航效率指标SPL方面，并在真实机器人上取得了有希望的结果

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': "这篇论文主要解决'物体目标导航'问题，即机器人在未知环境中高效找到指定目标物体的挑战。这个问题在现实中非常重要，因为它是移动操作系统执行多样化任务的基础能力，需要机器人具备长距离多模态场景理解和高效探索的能力，同时保持对已访问区域的长期记忆和学习环境先验知识。", '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者通过分析现有方法的局限性来设计新方法：现有隐式模型方法难以处理长期空间记忆，而显式地图方法缺乏丰富的语义信息。作者借鉴了前沿探索概念、视觉语言模型和模块化管道等现有技术，但创新性地将它们结合，提出了前沿-物体地图这一新表示方法，以克服现有方法的不足。', '这个方法的核心思想是什么？整体实现流程是怎样的？': "核心思想是提出'前沿-物体地图'这一丰富表示方法，联合编码空间前沿和细粒度物体信息，并利用视觉语言模型进行多模态场景理解和高级目标预测。整体流程分为三步：1)在线前沿-物体映射，维护3D障碍物地图和2D探索地图，同时分割物体并存储其特征；2)高级目标预测，使用VLM处理前沿、物体和历史路径信息来预测下一个导航目标；3)低级路径规划，将高级目标转换为可执行的机器人动作序列。", '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)在线前沿-物体映射，显式捕获具有丰富空间-语义信息的细粒度记忆；2)利用视觉语言模型架构进行高级目标预测；3)自动构建大规模机器人导航数据集；4)模块化框架设计。相比之前工作，该方法克服了隐式模型长期记忆不足和显式地图语义信息贫乏的问题，地图可重用且不局限于单一目标物体，能够从机器人数据中持续改进。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '本文提出了FOM-Nav框架，通过前沿-物体地图和视觉语言模型实现了物体目标导航任务中的高效探索和语义理解，在多个基准测试上达到了最先进的性能。'}


### 论文摘要

This paper addresses the Object Goal Navigation problem, where a robot must efficiently find a target object in an unknown environment. Existing implicit memory-based methods struggle with long-term memory retention and planning, while explicit map-based approaches lack rich semantic information. To address these challenges, we propose FOM-Nav, a modular framework that enhances exploration efficiency through Frontier-Object Maps and vision-language models. Our Frontier-Object Maps are built online and jointly encode spatial frontiers and fine-grained object information. Using this representation, a vision-language model performs multimodal scene understanding and high-level goal prediction, which is executed by a low-level planner for efficient trajectory generation. To train FOM-Nav, we automatically construct large-scale navigation datasets from real-world scanned environments. Extensive experiments validate the effectiveness of our model design and constructed dataset. FOM-Nav achieves state-of-the-art performance on the MP3D and HM3D benchmarks, particularly in navigation efficiency metric SPL, and yields promising results on a real robot.

---

## 219. Accelerating Streaming Video Large Language Models via Hierarchical Token Compression

**论文链接:** [http://arxiv.org/abs/2512.00891v1](http://arxiv.org/abs/2512.00891v1)

**作者:** Yiyu Wang, Xuyang Liu, Xiyan Gui, Xinying Lin, Boxue Yang, Chenfei Liao, Tailai Chen, Linfeng Zhang

**发布时间:** 2025-11-30

**备注:** Code is avaliable at \url{https://github.com/lern-to-write/STC}

### GPT解析

### 总结

本文提出了一种名为Streaming Token Compression (STC)的即插即用分层框架，用于优化流式视频大语言模型的性能，通过减少ViT编码开销和压缩视觉标记序列，在保持高达99%准确性的同时，分别将ViT编码延迟和LLM预填充延迟减少了24.5%和45.3%。

### 背景

流式视频大语言模型(VideoLLMs)在多种视频理解任务上表现出色，但由于处理连续视频流中密集视觉标记的高计算成本，在实时部署方面面临重大挑战。

### 目的

解决流式VideoLLMs在实时部署中的计算效率问题，优化处理速度同时保持准确性。

### 方法

提出STC框架，包含两个标记级加速器：STC-Cacher通过缓存和重用时间相似帧的特征减少ViT编码开销；STC-Pruner在视觉标记序列进入LLM前进行压缩，仅保留基于空间和时间相关性的最显著标记。

### 主要发现

STC在四个基线流式VideoLLMs和五个基准测试上优于其他压缩方法，在ReKV框架上保留了高达99%的准确性，同时将ViT编码延迟减少24.5%，将LLM预填充延迟减少45.3%。

### 结论

STC是一种有效的解决方案，可以无缝集成到现有流式VideoLLMs中，显著提高处理效率同时保持高准确性。

### 翻译

流式视频大语言模型(VideoLLMs)在各种视频理解任务上已经展示了令人印象深刻的性能，但由于处理来自连续视频流的密集视觉标记的高计算成本，它们在实时部署方面面临重大挑战。在流视频场景中，主要瓶颈在于Vision Transformer (ViT)编码阶段，其中时间上相似帧的重复处理导致效率低下。此外，在LLM预填充过程中膨胀的标记序列进一步加剧了延迟和内存开销。为解决这些挑战，我们提出了Streaming Token Compression (STC)，一种即插即用的分层框架，可无缝集成到现有的流式VideoLLMs中，优化ViT编码和LLM预填充阶段以加速处理。STC引入了两个标记级加速器：STC-Cacher，通过缓存和重用时间上相似帧的特征来减少ViT编码开销；以及STC-Pruner，在视觉标记序列进入LLM之前对其进行压缩，仅保留基于空间和时间相关性的最显著标记。在五个基准测试上对四个基线流式VideoLLMs进行的广泛实验表明，STC优于其他压缩方法。值得注意的是，STC在ReKV框架上保留了高达99%的准确性，同时将ViT编码延迟和LLM预填充延迟分别减少了24.5%和45.3%。


### 论文摘要

Streaming Video Large Language Models (VideoLLMs) have demonstrated impressive performance across various video understanding tasks, but they face significant challenges in real-time deployment due to the high computational cost of processing dense visual tokens from continuous video streams. In streaming video scenarios, the primary bottleneck lies in the Vision Transformer (ViT) encoding stage, where redundant processing of temporally similar frames leads to inefficiency. Additionally, inflated token sequences during LLM pre-filling further exacerbate latency and memory overhead. To address these challenges, we propose \textbf{S}treaming \textbf{T}oken \textbf{C}ompression (\textbf{STC}), a plug-and-play hierarchical framework that seamlessly integrates into existing streaming VideoLLMs, optimizing both ViT encoding and LLM pre-filling stages to accelerate processing. STC introduces two token-level accelerators: \textbf{STC-Cacher}, which reduces ViT encoding overhead by caching and reusing features from temporally similar frames, and \textbf{STC-Pruner}, which compresses the visual token sequence before it enters the LLM, preserving only the most salient tokens based on both spatial and temporal relevance. Extensive experiments on four baseline streaming VideoLLMs across five benchmarks demonstrate that STC outperforms other compression methods. Notably, STC retains up to \textbf{99\%} of accuracy on the ReKV framework while reducing ViT encoding latency and LLM pre-filling latency by \textbf{24.5\%} and \textbf{45.3\%}.

---

## 220. HanDyVQA: A Video QA Benchmark for Fine-Grained Hand-Object Interaction Dynamics

**论文链接:** [http://arxiv.org/abs/2512.00885v1](http://arxiv.org/abs/2512.00885v1)

**作者:** Masatoshi Tateno, Gido Kato, Hirokatsu Kataoka, Yoichi Sato, Takuma Yagi

**发布时间:** 2025-11-30

**备注:** Project page: https://masatate.github.io/HanDyVQA-project-page/

### GPT解析

### 总结

本文提出了HanDyVQA，一个细粒度的视频问答基准测试，用于全面评估手部-物体交互(HOI)中的操作和效应两个方面，并发现现有模型在理解HOI动力学方面仍有显著差距。

### 背景

手部-物体交互(HOI)本质上涉及动态过程，人类操作会对物体产生特定的时空效应。然而，现有的语义HOI基准测试要么专注于操作，要么专注于粗粒度的结果效应，缺乏细粒度的时空推理来捕捉HOI中的潜在动态。

### 目的

引入HanDyVQA，一个细粒度的视频问答基准测试，全面涵盖HOI的操作和效应两个方面，以促进对HOI动力力的更深入理解。

### 方法

HanDyVQA包含六种互补的问题类型（动作、过程、物体、位置、状态变化和物体部分），总共11.1K个多项选择问答对。收集了识别操作风格、手/物体运动和部分级别状态变化的问答对，并包含10.3K个针对物体和物体部分问题的分割掩码，用于评估视频对象分割中的物体/部分级别推理。

### 主要发现

在基准测试上评估了最近的视频基础模型，发现即使表现最好的模型Gemini-2.5-Pro也仅达到73%的平均准确率，远低于人类表现(97%)。进一步分析显示在空间关系、运动和部分级别几何理解方面仍有挑战。将明确的HOI相关线索整合到视觉特征中可以提高性能。

### 结论

HanDyVQA为评估和改进手部-物体交互理解提供了新基准，并为开发具有更深层HOI动力学理解的未来模型提供了见解。

### 翻译

手部-物体交互(HOI)本质上涉及动态过程，人类操作会对物体产生特定的时空效应。然而，现有的语义HOI基准测试要么专注于操作，要么专注于粗粒度的结果效应，缺乏细粒度的时空推理来捕捉HOI中的潜在动态。我们引入HanDyVQA，一个细粒度的视频问答基准测试，全面涵盖HOI的操作和效应两个方面。HanDyVQA包含六种互补的问题类型（动作、过程、物体、位置、状态变化和物体部分），总共11.1K个多项选择问答对。收集了识别操作风格、手/物体运动和部分级别状态变化的问答对。HanDyVQA还包括10.3K个针对物体和物体部分问题的分割掩码，用于评估视频对象分割中的物体/部分级别推理。我们在基准测试上评估了最近的视频基础模型，发现即使表现最好的模型Gemini-2.5-Pro也仅达到73%的平均准确率，远低于人类表现(97%)。进一步分析显示在空间关系、运动和部分级别几何理解方面仍有挑战。我们还发现，将明确的HOI相关线索整合到视觉特征中可以提高性能，为开发具有更深层HOI动力学理解的未来模型提供了见解。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决的问题是现有手部-物体交互(HOI)基准测试的局限性，它们要么关注操作(manipulation)，要么关注结果效应(effect)，但缺乏对HOI中精细时空动态的全面评估。这个问题在现实中很重要，因为准确识别HOI中的时空动态可以应用于工人辅助、机器人灵巧操作和运动功能分析等场景；在研究中很重要，因为它填补了现有基准测试的空白，提供了一个更全面的评估框架来理解和建模手部-物体交互的完整动态过程。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先识别出现有HOI基准测试的局限性，它们孤立了HOI的单一方面，忽略了从操作到结果的动态特性。在设计方法时，作者借鉴了现有的HOI识别基准(包括低级别的手部和物体检测、3D姿态估计和高级别的动作识别)、指称视频对象分割(RVOS)和推理视频对象分割(ReasoningVOS)的方法，以及现有的视频语言模型。作者设计了两个任务：多项选择题(MCQ)和推理视频对象分割(ReasoningVOS)，定义了六种问题类型，并基于Ego4D数据集通过协作框架收集QA对，使用LLMs提出初始候选，再由人工完善和验证，确保质量和多样性。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是创建一个全面的视频问答基准，用于评估模型对手部-物体交互(HOI)动态的理解能力，特别是从操作到效应的精细时空动态。整体实现流程包括：1)基于Ego4D数据集筛选包含手部-物体交互的视频片段；2)使用模板自动生成候选问题；3)人工验证和修改问题，提供正确答案和干扰项；4)为物体和物体部分问题添加分割掩码注释；5)定义MCQ和ReasoningVOS两种任务类型；6)使用多种视频语言模型进行评估，分析模型在理解空间关系、运动和部分级几何理解方面的挑战。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)全面覆盖HOI的操作和效应两个方面；2)定义六种互补问题类型(动作、过程、物体、位置、状态变化和物体部分)；3)引入推理视频对象分割(ReasoningVOS)任务，要求隐式像素级推理；4)覆盖多样化的场景和活动领域；5)提供精细的人工注释；6)深入分析现有模型的局限性。相比之前的工作，HanDyVQA的主要不同在于它提供了一个更全面、更精细的基准测试，不仅关注高级别的动作或物体状态，还关注精细的时空动态和组件级别的变化，并引入了需要推理动态手部-物体关系的新任务。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'HanDyVQA是一个全新的视频问答基准，通过六种互补问题类型和推理视频对象分割任务，全面评估了对手部-物体交互精细时空动态的理解能力，揭示了现有模型的局限性，并为未来模型开发指明了方向。'}


### 论文摘要

Hand-object interaction (HOI) inherently involves dynamics where human manipulations produce distinct spatio-temporal effects on objects. However, existing semantic HOI benchmarks focused either on manipulation or on the resulting effects at a coarse level, lacking fine-grained spatio-temporal reasoning to capture the underlying dynamics in HOI. We introduce HanDyVQA, a fine-grained video question-answering benchmark that comprehensively covers both the manipulation and effect aspects of HOI. HanDyVQA comprises six complementary question types (Action, Process, Objects, Location, State Change, and Object Parts), totalling 11.1K multiple-choice QA pairs. Collected QA pairs recognizing manipulation styles, hand/object motions, and part-level state changes. HanDyVQA also includes 10.3K segmentation masks for Objects and Object Parts questions, enabling the evaluation of object/part-level reasoning in video object segmentation. We evaluated recent video foundation models on our benchmark and found that even the best-performing model, Gemini-2.5-Pro, reached only 73% average accuracy, which is far from human performance (97%). Further analysis shows the remaining challenges in spatial relationship, motion, and part-level geometric understanding. We also found that integrating explicit HOI-related cues into visual features improves performance, offering insights for developing future models with a deeper understanding of HOI dynamics.

---

## 221. Smol-GS: Compact Representations for Abstract 3D Gaussian Splatting

**论文链接:** [http://arxiv.org/abs/2512.00850v1](http://arxiv.org/abs/2512.00850v1)

**作者:** Haishan Wang, Mohammad Hassan Vali, Arno Solin

**发布时间:** 2025-11-30

### GPT解析

### 总结

Smol-GS是一种新颖的3D高斯飞溅紧凑表示学习方法，通过整合空间和语义信息实现高效编码，在保持高质量渲染的同时实现显著压缩。

### 背景

3D高斯飞溅(3DGS)是一种3D场景表示方法，需要学习紧凑的表示以减少存储和计算开销。

### 目的

提出Smol-GS方法，用于学习3D高斯飞溅的紧凑表示，实现高效率的3D空间编码，整合空间和语义信息。

### 方法

通过递归体素层次结构捕捉飞溅坐标，飞溅特征存储抽象线索，包括颜色、不透明度、变换和材质属性，这种设计允许模型压缩3D场景几个数量级而不损失灵活性。

### 主要发现

Smol-GS在标准基准测试上实现了最先进的压缩，同时保持高渲染质量。

### 结论

除了视觉保真度，离散表示可以作为下游任务的基础，如导航、规划和更广泛的3D场景理解。

### 翻译

我们提出了Smol-GS，一种用于学习3D高斯飞溅(3DGS)紧凑表示的新颖方法。我们的方法在3D空间中学习高度高效的编码，整合了空间和语义信息。该模型通过递归体素层次结构捕捉飞溅的坐标，而飞溅特征存储抽象线索，包括颜色、不透明度、变换和材质属性。这种设计允许模型压缩3D场景几个数量级而不损失灵活性。Smol-GS在标准基准测试上实现了最先进的压缩，同时保持高渲染质量。除了视觉保真度，离散表示还可以作为下游任务的基础，如导航、规划和更广泛的3D场景理解。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决3D Gaussian Splatting (3DGS)的内存效率低下问题。3DGS虽然能实现实时渲染和高图像质量，但通常需要数GB存储空间来表示复杂场景，这限制了它在资源受限设备上的应用，阻碍了3D场景的存储、传输和广泛应用，如VR/AR和移动设备等场景。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先分析了现有3DGS压缩方法的两个主要方向(信号处理方法和基于学习的方法)及其局限性，然后提出新思路：保持空间结构显式但高效打包，同时抽象化splat级别的视觉特征。作者借鉴了神经高斯生成系统和点云几何压缩方法，并使用了类似HAC的熵编码技术，设计了Smol-GS方法。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是将3D场景表示为紧凑的神经高斯splat，使用八叉树结构存储坐标，用低维抽象特征表示视觉属性，并通过小型MLP解码器重建视图相关属性。整体流程包括：初始化坐标和特征；通过梯度引导的分裂和剪枝调整splat数量；对特征进行量化和算术编码；使用占用八叉树和熵编码存储坐标；训练分为预热、密度化、压缩、特征压缩和坐标压缩五个阶段。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：创新的八叉树坐标编码和低维抽象特征表示；同时压缩坐标和特征(大多数方法避免坐标压缩)；使用学习到的量化和算术编码特征；自适应密度控制策略。相比之前工作，Smol-GS不使用锚点-偏移设计，避免了透明偏移问题，实现了更高的压缩比(约150倍)，同时保持了高质量的渲染结果和实时渲染能力。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'Smol-GS通过创新的八叉树坐标编码和低维抽象特征表示，实现了3D Gaussian Splatting的高效压缩，在保持高质量渲染的同时将模型大小减少了两个数量级，为3D场景的存储、传输和应用提供了新的可能性。'}


### 论文摘要

We present Smol-GS, a novel method for learning compact representations for 3D Gaussian Splatting (3DGS). Our approach learns highly efficient encodings in 3D space that integrate both spatial and semantic information. The model captures the coordinates of the splats through a recursive voxel hierarchy, while splat-wise features store abstracted cues, including color, opacity, transformation, and material properties. This design allows the model to compress 3D scenes by orders of magnitude without loss of flexibility. Smol-GS achieves state-of-the-art compression on standard benchmarks while maintaining high rendering quality. Beyond visual fidelity, the discrete representations could potentially serve as a foundation for downstream tasks such as navigation, planning, and broader 3D scene understanding.

---

## 222. AFRAgent : An Adaptive Feature Renormalization Based High Resolution Aware GUI agent

**论文链接:** [http://arxiv.org/abs/2512.00846v1](http://arxiv.org/abs/2512.00846v1)

**作者:** Neeraj Anand, Rishabh Jain, Sohan Patnaik, Balaji Krishnamurthy, Mausoom Sarkar

**发布时间:** 2025-11-30

**备注:** Accepted at WACV 2026 Conference

### GPT解析

### 总结

本文介绍了AFRAgent，一种基于instruct-BLIP的小型多模态架构，在GUI自动化任务中实现卓越性能，同时模型大小不到最近竞争对手的四分之一。

### 背景

移动用户界面(UI)自动化需求日益增长，视觉语言模型(VLMs)的发展使GUI自动化从生成基于文本的指令转变为自主执行任务。现有方法利用VLMs直接处理屏幕内容、独立于设备API以及应用现实世界上下文知识的优势。

### 目的

解决现有VLMs在准确识别小部件和确定动作方面的困难，以及大型模型需要大量训练和推理延迟的问题。

### 方法

提出AFRAgent架构，并引入基于自适应特征重归一化(一种标记级仿射变换)的技术，增强大语言模型(LLM)管道中的图像嵌入，有效丰富低分辨率图像嵌入并融合高分辨率细节。

### 主要发现

AFRAgent在GUI自动化方面取得了卓越性能，模型大小不到最近竞争对手的四分之一，在Meta-GUI和AITW基准测试上建立了智能手机自动化的新最先进基线。

### 结论

通过创新的图像嵌入增强技术，AFRAgent实现了高效且准确的GUI自动化，解决了现有模型在性能和效率之间的权衡问题。

### 翻译

移动用户界面(UI)自动化的需求日益增长，这得益于其在各行业的广泛应用。随着视觉语言模型(VLMs)的出现，GUI自动化已从为人类生成基于文本的指令发展为自主执行任务，从而优化了自动化工作流程。由于VLMs能够1)直接处理屏幕内容，2)通过利用人类动作(如点击、输入)独立于设备特定API，以及3)应用现实世界上下文知识进行任务理解，最近的方法利用VLMs解决此问题。然而，由于视觉编码器特征中的空间信息有限，这些模型在准确识别小部件和确定动作方面经常遇到困难。此外，性能最佳的模型通常较大，需要大量训练并导致推理延迟。在这项工作中，我们介绍了AFRAgent，一种基于instruct-BLIP的多模态架构，在GUI自动化方面取得了卓越性能，同时大小不到最近竞争对手的四分之一。为了增强大语言模型(LLM)管道中的图像嵌入，我们提出了一种基于自适应特征重归一化(一种标记级仿射变换)的技术，有效丰富了低分辨率图像嵌入并融合了高分辨率细节。我们在Meta-GUI和AITW基准测试上评估了AFRAgent，为智能手机自动化建立了新的最先进基线。


### 论文摘要

There is a growing demand for mobile user interface (UI) automation, driven by its broad applications across industries. With the advent of visual language models (VLMs), GUI automation has progressed from generating text-based instructions for humans to autonomously executing tasks, thus optimizing automation workflows. Recent approaches leverage VLMs for this problem due to their ability to 1) process on-screen content directly, 2) remain independent of device-specific APIs by utilizing human actions (e.g., clicks, typing), and 3) apply real-world contextual knowledge for task understanding. However, these models often have trouble accurately identifying widgets and determining actions due to limited spatial information in vision encoder features. Additionally, top-performing models are often large, requiring extensive training and resulting in inference delays. In this work, we introduce AFRAgent, an instruct-BLIP-based multimodal architecture that achieves superior performance in GUI automation while being less than one-fourth the size of its nearest competitor. To enhance image embeddings in the large language model (LLM) pipeline, we propose an adaptive feature renormalization-based (a token-level affine transformation) technique that effectively enriches low-resolution image embeddings and fuses high-resolution details. We evaluate AFRAgent on Meta-GUI and AITW benchmarks, establishing a new state-of-the-art baseline for smartphone automation.

---

