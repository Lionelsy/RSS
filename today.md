# 今日论文推荐 - 2025-09-25

共 66 篇论文

---

## 1. CapStARE: Capsule-based Spatiotemporal Architecture for Robust and Efficient Gaze Estimation

**论文链接:** [http://arxiv.org/abs/2509.19936v1](http://arxiv.org/abs/2509.19936v1)

**作者:** Miren Samaniego, Igor Rodriguez, Elena Lazkano

**发布时间:** 2025-09-24

### GPT解析

### 总结

CapStARE是一种基于胶囊的时空架构，用于凝视估计，结合了ConvNeXt主干网络、注意力路由的胶囊形成和专门处理慢速和快速凝视动态的双GRU解码器，实现了高性能和实时推理能力。

### 背景

凝视估计是交互系统中的关键任务，需要准确估计人的凝视方向，同时处理不同凝视动态和复杂环境条件。

### 目的

开发一种高效、准确且实时运行的凝视估计模型，能够处理不同凝视动态，并在各种条件下具有良好的泛化能力。

### 方法

采用模块化设计的CapStARE架构，包含ConvNeXt主干网络、注意力路由的胶囊形成和双GRU解码器，支持高效的部件-整体推理和解耦的时序建模。

### 主要发现

在ETH-XGaze(3.36)和MPIIFaceGaze(2.65)数据集上达到最先进性能，实现实时推理(<10ms)，在Gaze360(9.06)和RT-GENE(4.76)等不同场景下表现良好，且使用更少参数并具有更高可解释性。

### 结论

CapStARE为交互系统中的实时凝视估计提供了一种实用且稳健的解决方案，在保持高性能的同时实现了高效推理和良好的泛化能力。

### 翻译

我们介绍了CapStARE，一种基于胶囊的时空架构用于凝视估计，它集成了ConvNeXt主干网络、注意力路由的胶囊形成和专门处理慢速与快速凝视动态的双GRU解码器。这种模块化设计能够实现高效的部件-整体推理和解耦的时序建模，在ETH-XGaze(3.36)和MPIIFaceGaze(2.65)上达到最先进的性能，同时保持实时推理(<10ms)。该模型在Gaze360(9.06)的无约束条件和RT-GENE(4.76)的人机交互场景中也具有良好的泛化能力，以更少的参数和更高的可解释性优于或匹配现有方法。这些结果表明CapStARE为交互系统中的实时凝视估计提供了实用且稳健的解决方案。相关代码和结果可在https://github.com/toukapy/capsStare找到。


### 论文摘要

We introduce CapStARE, a capsule-based spatio-temporal architecture for gaze estimation that integrates a ConvNeXt backbone, capsule formation with attention routing, and dual GRU decoders specialized for slow and rapid gaze dynamics. This modular design enables efficient part-whole reasoning and disentangled temporal modeling, achieving state-of-the-art performance on ETH-XGaze (3.36) and MPIIFaceGaze (2.65) while maintaining real-time inference (< 10 ms). The model also generalizes well to unconstrained conditions in Gaze360 (9.06) and human-robot interaction scenarios in RT-GENE (4.76), outperforming or matching existing methods with fewer parameters and greater interpretability. These results demonstrate that CapStARE offers a practical and robust solution for real-time gaze estimation in interactive systems. The related code and results for this article can be found on: https://github.com/toukapy/capsStare

---

## 2. iFinder: Structured Zero-Shot Vision-Based LLM Grounding for Dash-Cam Video Reasoning

**论文链接:** [http://arxiv.org/abs/2509.19552v1](http://arxiv.org/abs/2509.19552v1)

**作者:** Manyi Yao, Bingbing Zhuang, Sparsh Garg, Amit Roy-Chowdhury, Christian Shelton, Manmohan Chandraker, Abhishek Aich

**发布时间:** 2025-09-23

### GPT解析

### 总结

iFinder是一个结构化的语义基础框架，通过将驾驶摄像头视频转换为分层、可解释的数据结构，使大型语言模型能够进行驾驶视频分析。它使用预训练视觉模型提取关键线索，并结合三块提示策略，显著提高了事故推理准确性，最高达39%。

### 背景

大型语言模型在特定领域任务如驾驶视频分析中面临挑战，因为它们是通用训练的，缺乏结构化的归纳偏见。在驾驶分析中，视觉通常是唯一可用的模态，现有的基于视频的视觉-语言模型在空间推理、因果推断和事件解释性方面存在问题。

### 目的

介绍iFinder，一个结构化的语义基础框架，解决V-VLMs在驾驶视频分析中的局限性，使大型语言模型能够更好地处理驾驶视频分析任务。

### 方法

iFinder将感知与推理解耦，通过将驾驶摄像头视频转换为分层、可解释的数据结构供LLMs使用。它作为一个模块化、无需训练的流水线运行，使用预训练的视觉模型提取物体姿态、车道位置和物体轨迹等关键线索，并将这些线索分层组织成帧级和视频级结构。结合三块提示策略，使LLM能够进行逐步的、基于基础的推理，精炼另一个V-VLM的输出并提供准确的推理。

### 主要发现

在四个公共驾驶摄像头视频基准测试上的评估显示，iFinder提出的领域特定线索的基础，特别是物体方向和全局上下文，在四个零样本驾驶基准测试上显著优于端到端的V-VLMs，事故推理准确性提高了高达39%。

### 结论

通过使用驾驶领域特定表示来基础化LLMs，iFinder为端到端V-VLMs提供了一个零样本、可解释且可靠的替代方案，用于后置驾驶视频理解。

### 翻译

将大型语言模型应用于特定领域任务（如后置摄像头驾驶视频分析）具有挑战性，因为它们是通用训练的，缺乏结构化的归纳偏见。由于视觉通常是此类分析中唯一可用的模态（即没有LiDAR、GPS等），现有的基于视频的视觉-语言模型在空间推理、因果推断和输入视频中事件的解释性方面存在困难。为此，我们介绍了iFinder，一个结构化的语义基础框架，它通过将驾驶摄像头视频转换为分层的、可解释的数据结构供LLMs使用，从而将感知与推理解耦。iFinder作为一个模块化、无需训练的流水线运行，使用预训练的视觉模型提取关键线索——物体姿态、车道位置和物体轨迹——这些线索被分层组织成帧级和视频级结构。结合三块提示策略，它使LLM能够进行逐步的、基于基础的推理，以精炼另一个V-VLM的输出并提供准确的推理。在四个公共驾驶摄像头视频基准测试上的评估显示，iFinder提出的领域特定线索的基础，特别是物体方向和全局上下文，在四个零样本驾驶基准测试上显著优于端到端的V-VLMs，事故推理准确性提高了高达39%。通过使用驾驶领域特定表示来基础化LLMs，iFinder为端到端V-VLMs提供了一个零样本、可解释且可靠的替代方案，用于后置驾驶视频理解。


### 论文摘要

Grounding large language models (LLMs) in domain-specific tasks like post-hoc dash-cam driving video analysis is challenging due to their general-purpose training and lack of structured inductive biases. As vision is often the sole modality available for such analysis (i.e., no LiDAR, GPS, etc.), existing video-based vision-language models (V-VLMs) struggle with spatial reasoning, causal inference, and explainability of events in the input video. To this end, we introduce iFinder, a structured semantic grounding framework that decouples perception from reasoning by translating dash-cam videos into a hierarchical, interpretable data structure for LLMs. iFinder operates as a modular, training-free pipeline that employs pretrained vision models to extract critical cues -- object pose, lane positions, and object trajectories -- which are hierarchically organized into frame- and video-level structures. Combined with a three-block prompting strategy, it enables step-wise, grounded reasoning for the LLM to refine a peer V-VLM's outputs and provide accurate reasoning. Evaluations on four public dash-cam video benchmarks show that iFinder's proposed grounding with domain-specific cues, especially object orientation and global context, significantly outperforms end-to-end V-VLMs on four zero-shot driving benchmarks, with up to 39% gains in accident reasoning accuracy. By grounding LLMs with driving domain-specific representations, iFinder offers a zero-shot, interpretable, and reliable alternative to end-to-end V-VLMs for post-hoc driving video understanding.

---

## 3. Transformer Modeling for Both Scalability and Performance in Multivariate Time Series

**论文链接:** [http://arxiv.org/abs/2509.19471v1](http://arxiv.org/abs/2509.19471v1)

**作者:** Hunjae Lee, Corey Clark

**发布时间:** 2025-09-23

### GPT解析

### 总结

变量数量是多元时间序列Transformer建模的主要可扩展性瓶颈，无差别变量间混合会导致噪声累积和性能下降。DELTAformer通过委托令牌约束变量间混合，实现线性扩展的同时提高性能，在基准测试中表现最佳，并在嘈杂环境中展现出更强的鲁棒性。

### 背景

变量数量是多元时间序列(MTS)数据中Transformer建模的主要可扩展性瓶颈之一。该领域日益达成共识，认为无差别的变量间混合可能是噪声累积和性能下降的潜在来源。许多MTS系统具有信息信号稀疏性特征，加上来自异构变量间无差别信息混合的表示错位，可能会加剧这一问题。

### 目的

在MTS中同时提高可扩展性和性能，通过策略性地限制变量间混合的表示能力来实现这一目标。

### 方法

提出了一种名为DELTAformer的方法（具有委托令牌注意力的Transformer）。通过所谓的委托令牌约束变量间建模，然后使用这些委托令牌执行完全无约束的时间间建模。委托令牌充当隐式正则化器，强制模型对允许通过网络传播的变量间信息高度选择性。

### 主要发现

DELTAformer随变量数量线性扩展，实际上优于标准Transformer，在基准测试和基线中实现了最先进的性能。在嘈杂的MTS环境中比标准Transformer更好地关注相关信号，总体表现出更强的噪声鲁棒性。

### 结论

通过将模型设计与利用MTS领域特定挑战相结合，DELTAformer可以同时实现线性扩展，同时与标准的二次方Transformer相比，实际上提高了其性能。

### 翻译

变量数量是多元时间序列Transformer建模中的主要可扩展性瓶颈之一。此外，该领域日益达成的共识表明，无差别的变量间混合可能是噪声累积和性能下降的潜在来源。这很可能被许多MTS系统固有的信息信号稀疏性所加剧，同时也源于异构变量间无差别信息混合导致的表示错位。虽然可扩展性和性能在Transformer设计中通常被视为相互竞争的目标，但我们证明在MTS中，通过策略性地约束变量间混合的表示能力，可以同时提高两者。我们提出的方法是具有委托令牌注意力的Transformer（DELTAformer），通过我们所谓的委托令牌约束变量间建模，然后使用这些委托令牌执行完全无约束的时间间建模。委托令牌充当隐式正则化器，强制模型对允许通过网络传播的变量间信息高度选择性。我们的结果表明，DELTAformer随变量数量线性扩展，同时实际上优于标准Transformer，在基准测试和基线中实现了最先进的性能。此外，DELTAformer在嘈杂的MTS环境中比标准Transformer能更好地关注相关信号，总体表现出更强的噪声鲁棒性。总体而言，各种实验的结果证实，通过将我们的模型设计与利用MTS中的领域特定挑战相结合，DELTAformer可以同时实现线性扩展，同时实际上提高了其性能，与标准的二次方Transformer相比。


### 论文摘要

Variable count is among the main scalability bottlenecks for transformer modeling in multivariate time series (MTS) data. On top of this, a growing consensus in the field points to indiscriminate inter-variable mixing as a potential source of noise-accumulation and performance degradation. This is likely exacerbated by sparsity of informative signals characteristic of many MTS systems coupled with representational misalignment stemming from indiscriminate information mixing between (heterogeneous) variables. While scalability and performance are often seen as competing interests in transformer design, we show that both can be improved simultaneously in MTS by strategically constraining the representational capacity of inter-variable mixing. Our proposed method, transformer with Delegate Token Attention (DELTAformer), constrains inter-variable modeling through what we call delegate tokens which are then used to perform full, unconstrained, inter-temporal modeling. Delegate tokens act as an implicit regularizer that forces the model to be highly selective about what inter-variable information is allowed to propagate through the network. Our results show that DELTAformer scales linearly with variable-count while actually outperforming standard transformers, achieving state-of-the-art performance across benchmarks and baselines. In addition, DELTAformer can focus on relevant signals better than standard transformers in noisy MTS environments and overall exhibit superior noise-resilience. Overall, results across various experiments confirm that by aligning our model design to leverage domain-specific challenges in MTS to our advantage, DELTAformer can simultaneously achieve linear scaling while actually improving its performance against standard, quadratic transformers.

---

## 4. COLT: Enhancing Video Large Language Models with Continual Tool Usage

**论文链接:** [http://arxiv.org/abs/2509.18754v2](http://arxiv.org/abs/2509.18754v2)

**作者:** Yuyang Liu, Xinyuan Shi, Xiaondan Liang

**发布时间:** 2025-09-23

**备注:** 16 pages

### GPT解析

### 总结

本研究提出了一种名为COLT（COntinuaL Tool usage）的方法，用于增强开源视频大型语言模型，使其能够在持续变化和流动的工具数据中自动获取工具使用能力而不会遗忘已学习的工具。同时，作者还收集了VideoToolBench数据集，并通过实验验证了该方法的有效性。

### 背景

大型语言模型的成功显著推动了视频理解领域的研究。现有的视频LLMs主要探索工具使用能力，但现有方法依赖于固定的工具库，难以适应现实世界中不断变化和流动的工具数据。

### 目的

增强开源视频LLMs，使其能够在持续变化和流动的工具数据中自动获取工具使用能力，同时避免对已学习工具的'灾难性遗忘'。

### 方法

作者提出了COLT方法，它包含一个可学习的工具代码本作为特定工具的记忆系统。然后，基于用户指令与工具代码本中工具特征的相似性动态选择相关工具。此外，作者还收集了一个名为VideoToolBench的视频中心工具使用指令微调数据集。

### 主要发现

在先前视频LLM基准和特定工具使用的VideoToolBench数据集上的大量实验表明，所提出的COLT达到了最先进的性能。

### 结论

COLT方法能够有效增强开源视频LLMs的工具使用能力，使其能够适应不断变化和流动的工具数据，同时保持对已学习工具的记忆。

### 翻译

大型语言模型的成功显著推动了视频理解的研究。为了利用已训练专家模型（即工具）的优势，视频LLMs优先探索工具使用能力。然而，现有方法要么提示闭源LLMs，要么采用指令微调范式进行工具使用微调。这些方法假设有一个固定的工具库，难以推广到工具数据不断变化和流动的现实世界环境中。为此，我们提出通过持续工具使用（称为COLT）增强开源视频LLMs，该方法能够在连续工具流中自动获取工具使用能力，而不会对过去学习的工具造成'灾难性遗忘'。具体来说，我们的COLT包含一个可学习的工具代码本作为特定工具的记忆系统。然后，基于用户指令与代码本内工具特征的相似性动态选择相关工具。为了释放视频LLMs的工具使用潜力，我们收集了一个以视频为中心的工具使用指令微调数据集VideoToolBench。在先前视频LLM基准和特定工具使用的VideoToolBench数据集上的大量实验证明了我们提出的COLT的最先进性能。


### 论文摘要

The success of Large Language Models (LLMs) has significantly propelled the research of video understanding. To harvest the benefits of well-trained expert models (i.e., tools), video LLMs prioritize the exploration of tool usage capabilities. Existing methods either prompt closed-source LLMs or employ the instruction tuning paradigm for tool-use fine-tuning. These methods, however, assume an established repository of fixed tools and struggle to generalize to real-world environments where tool data is perpetually evolving and streaming in. To this end, we propose to enhance open-source video LLMs with COntinuaL Tool usage (termed COLT), which automatically acquires tool-use ability in a successive tool stream without suffering 'catastrophic forgetting' of the past learned tools. Specifically, our COLT incorporates a learnable tool codebook as a tool-specific memory system. Then relevant tools are dynamically selected based on the similarity between user instruction and tool features within the codebook. To unleash the tool usage potential of video LLMs, we collect a video-centric tool-use instruction tuning dataset VideoToolBench. Extensive experiments on both previous video LLM benchmarks and the tool-use-specific VideoToolBench dataset demonstrate the state-of-the-art performance of our proposed COLT.

---

## 5. Predictive Coding-based Deep Neural Network Fine-tuning for Computationally Efficient Domain Adaptation

**论文链接:** [http://arxiv.org/abs/2509.20269v1](http://arxiv.org/abs/2509.20269v1)

**作者:** Matteo Cardoni, Sam Leroux

**发布时间:** 2025-09-24

**备注:** 20 pages, 4 figures

### GPT解析

### 总结

本研究提出了一种结合反向传播和预测编码的混合训练方法，用于在动态环境中实现高效的设备端域适应，解决了深度神经网络在真实场景中因数据分布变化导致的性能下降问题。

### 背景

深度神经网络越来越多地部署在动态的真实环境中，仅依靠单一静态模型往往不够。传感器漂移或光照变化导致的输入数据分布变化需要模型持续适应。

### 目的

提出一种混合训练方法，通过结合反向传播和预测编码的优势，实现高效的设备端域适应。

### 方法

该方法首先使用反向传播离线训练深度神经网络以获得高初始性能，然后使用预测编码进行在线适应，使模型能够恢复因输入数据分布变化而损失的准确性。

### 主要发现

这种方法利用反向传播在初始表示学习方面的稳健性和预测编码在持续学习方面的计算效率，特别适合资源受限的边缘设备或未来的神经形态加速器。在MNIST和CIFAR-10数据集上的实验结果表明，这种混合策略能够实现有效的适应，同时减少计算开销。

### 结论

该混合策略为在动态环境中保持模型性能提供了一种有前景的解决方案。

### 翻译

随着深度神经网络越来越多地部署在动态的真实环境中，依靠单一静态模型往往是不够的。传感器漂移或光照变化引起的输入数据分布变化需要模型持续适应。在本文中，我们提出了一种混合训练方法，通过结合反向传播和预测编码的优势，实现高效的设备端域适应。该方法首先使用反向传播离线训练深度神经网络以获得高初始性能。随后，使用预测编码进行在线适应，使模型能够恢复因输入数据分布变化而损失的准确性。这种方法利用了反向传播在初始表示学习方面的稳健性和预测编码在持续学习方面的计算效率，使其特别适合资源受限的边缘设备或未来的神经形态加速器。在MNIST和CIFAR-10数据集上的实验结果表明，这种混合策略能够实现有效的适应，同时减少计算开销，为在动态环境中保持模型性能提供了一种有前景的解决方案。


### 论文摘要

As deep neural networks are increasingly deployed in dynamic, real-world environments, relying on a single static model is often insufficient. Changes in input data distributions caused by sensor drift or lighting variations necessitate continual model adaptation. In this paper, we propose a hybrid training methodology that enables efficient on-device domain adaptation by combining the strengths of Backpropagation and Predictive Coding. The method begins with a deep neural network trained offline using Backpropagation to achieve high initial performance. Subsequently, Predictive Coding is employed for online adaptation, allowing the model to recover accuracy lost due to shifts in the input data distribution. This approach leverages the robustness of Backpropagation for initial representation learning and the computational efficiency of Predictive Coding for continual learning, making it particularly well-suited for resource-constrained edge devices or future neuromorphic accelerators. Experimental results on the MNIST and CIFAR-10 datasets demonstrate that this hybrid strategy enables effective adaptation with a reduced computational overhead, offering a promising solution for maintaining model performance in dynamic environments.

---

## 6. Diffusion-Augmented Contrastive Learning: A Noise-Robust Encoder for Biosignal Representations

**论文链接:** [http://arxiv.org/abs/2509.20048v1](http://arxiv.org/abs/2509.20048v1)

**作者:** Rami Zewail

**发布时间:** 2025-09-24

### GPT解析

### 总结

本研究提出了一种名为扩散增强对比学习(DACL)的新型混合框架，用于学习生物信号的稳健表示，通过融合扩散模型和监督对比学习的概念，解决了传统数据增强方法无法捕捉生理数据复杂变化的问题。

### 背景

学习生物信号的稳健表示常常受到有效数据增强设计挑战的阻碍。传统方法无法捕捉生理数据中固有的复杂变化，这限制了表示学习的有效性。

### 目的

开发一种新的混合框架，能够更好地捕捉生理数据中的复杂变化，学习具有鲁棒性的生物信号表示。

### 方法

提出了一种名为扩散增强对比学习(DACL)的混合框架，该框架在由轻量级变分自编码器(VAE)创建的潜在空间上运行，VAE在新型散射变换器(ST)特征上训练。该方法利用扩散前向过程作为数据增强技术生成潜在嵌入的多个噪声视图，并使用监督对比目标训练U-Net风格编码器，以学习在不同扩散时间步上平衡类判别性和噪声鲁棒性的表示。

### 主要发现

在PhysioNet 2017 ECG数据集上评估该方法，取得了0.7815的竞争性AUROC值，证明了该方法的有效性。

### 结论

这项工作通过使用扩散过程本身来驱动对比目标，为表示学习建立了一种新范式，创建了噪声不变的嵌入，这些嵌入展示了类可分性的坚实基础，为生物信号表示学习提供了新的思路。

### 翻译

学习生物信号的稳健表示通常受到设计有效数据增强技术的挑战阻碍。传统方法可能无法捕捉生理数据中固有的复杂变化。在此背景下，我们提出了一种新的混合框架——扩散增强对比学习(DACL)，该框架融合了扩散模型和监督对比学习的概念。DACL框架在由轻量级变分自编码器(VAE)创建的潜在空间上运行，该VAE在我们的新型散射变换器(ST)特征[12]上训练。它利用扩散前向过程作为有原则的数据增强技术，生成这些潜在嵌入的多个噪声视图。然后，使用监督对比目标训练U-Net风格的编码器，以学习一种表示，该表示在不同扩散时间步上平衡了类判别性和对噪声的鲁棒性。我们在PhysioNet 2017 ECG数据集上评估了这个概念验证方法，取得了0.7815的竞争性AUROC。这项工作通过使用扩散过程本身来驱动对比目标，为表示学习建立了一种新范式，创建了噪声不变的嵌入，这些嵌入展示了类可分性的坚实基础。


### 论文摘要

Learning robust representations for biosignals is often hampered by the challenge of designing effective data augmentations.Traditional methods can fail to capture the complex variations inherent in physiological data. Within this context, we propose a novel hybrid framework, Diffusion-Augmented Contrastive Learning (DACL), that fuses concepts from diffusion models and supervised contrastive learning. The DACL framework operates on a latent space created by a lightweight Variational Autoencoder (VAE) trained on our novel Scattering Transformer (ST) features [12]. It utilizes the diffusion forward process as a principled data augmentation technique to generate multiple noisy views of these latent embeddings. A U-Net style encoder is then trained with a supervised contrastive objective to learn a representation that balances class discrimination with robustness to noise across various diffusion time steps. We evaluated this proof-of-concept method on the PhysioNet 2017 ECG dataset, achieving a competitive AUROC of 0.7815. This work establishes a new paradigm for representation learning by using the diffusion process itself to drive the contrastive objective, creating noise-invariant embeddings that demonstrate a strong foundation for class separability.

---

## 7. Ricci Flow on Weighted Digraphs with Balancing Factor

**论文链接:** [http://arxiv.org/abs/2509.19989v1](http://arxiv.org/abs/2509.19989v1)

**作者:** Shuliang Bai, Rui Li, Shuang Liu, Xin Lai

**发布时间:** 2025-09-24

### GPT解析

### 总结

该论文介绍了有向加权图上的里奇流的严格公式化，提出了一种保持距离同时演化边权重的里奇流，并建立了其解的存在性和唯一性。为了捕捉有向网络中的不对称性，作者引入了节点级别的平衡因子来调节流出和流入。基于连续里奇流演化框架，提出了适用于数值计算的离散里奇流算法，数值研究表明该算法能够揭示结构不对称性和动态演化。

### 背景

里奇曲率和里奇流已被证明是分析离散结构几何特性的强大工具，特别是在无向图中，它们已被应用于从社区检测到图表示学习的各种任务。然而，在有向图上的发展仍然有限，特别是里奇流的研究尤其不足。

### 目的

在有向加权图上引入里奇流的严格公式化，解决有向网络上里奇流发展有限的问题，特别是探索里奇流在有向图中的应用。

### 方法

引入了一种在有向加权图上的里奇流公式，这种流在演化边权重的过程中保持距离；建立了这种流解的存在性和唯一性；引入了节点级别的平衡因子来调节流出和流入；基于连续里奇流演化框架，提出了适用于数值计算的离散里奇流算法。

### 主要发现

数值研究表明，所提出的里奇流算法能够揭示有向图的结构不对称性和动态演化特性。

### 结论

该工作成功地将里奇流理论扩展到了有向加权图领域，为分析有向网络的结构特性提供了新的数学工具和方法。

### 翻译

里奇曲率和里奇流已被证明是分析离散结构几何特性的强大工具，特别是在无向图中，它们已被应用于从社区检测到图表示学习的各种任务。然而，在有向图上的发展仍然有限，特别是里奇流的研究尤其不足。在这项工作中，我们引入了有向加权图上里奇流的严格公式化，这种流在演化边权重的过程中保持距离，并建立了其解的存在性和唯一性。为了捕捉有向网络中的不对称本质并增强建模更灵活结构的能力，我们引入了一个节点级别的平衡因子来调节流出和流入。基于连续里奇流演化框架，我们提出了一个适用于数值计算的离散里奇流算法。在各种有向图示例上的数值研究表明，所提出的流能够揭示结构不对称性和动态演化。


### 论文摘要

Ricci curvature and Ricci flow have proven to be powerful tools for analyzing the geometry of discrete structures, particularly on undirected graphs, where they have been applied to tasks ranging from community detection to graph representation learning. However, their development on directed graphs remains limited, with Ricci flow being especially underexplored. In this work, we introduce a rigorous formulation of Ricci flow on directed weighted graphs, which evolves edge weights while preserving distances, and establish both the existence and uniqueness of its solutions. To capture the essence of asymmetry in directed networks and to enhance the capability of modeling more flexible structures, we incorporate a node-wise balancing factor that regulates between outflow and inflow. Building on the continuous Ricci flow evolution framework, we propose a discrete Ricci flow algorithm that is applicable to numerical computing. Numerical studies on various directed graph examples demonstrate the capacity of the proposed flow to reveal structural asymmetry and dynamic evolutions.

---

## 8. Multimodal-enhanced Federated Recommendation: A Group-wise Fusion Approach

**论文链接:** [http://arxiv.org/abs/2509.19955v1](http://arxiv.org/abs/2509.19955v1)

**作者:** Chunxu Zhang, Weipeng Zhang, Guodong Long, Zhiheng Xue, Riting Xia, Bo Yang

**发布时间:** 2025-09-24

### GPT解析

### 总结

论文提出了一种名为GFMFR的新型多模态融合机制，用于联邦推荐系统，通过将多模态表示学习转移到服务器端并采用感知群体的项目表示融合方法，解决了整合多模态特征的效率、分布异质性和细粒度对齐等挑战。

### 背景

联邦推荐(FR)是一种新的学习范式，以隐私保护方式解决学习排序问题。然而，如何将多模态特征整合到联邦推荐中仍然是一个开放的挑战，主要面临效率、分布异质性和细粒度对齐等问题。

### 目的

解决联邦推荐中整合多模态特征面临的效率、分布异质性和细粒度对齐等挑战，提出一种新的多模态融合机制。

### 方法

提出GFMFR机制，将多模态表示学习转移到服务器端，服务器存储项目内容并使用高容量编码器生成丰富表示以减轻客户端开销；采用感知群体的项目表示融合方法，使相似用户间进行细粒度知识共享同时保留个人偏好；该融合损失可插入任何现有联邦推荐系统以增强多模态特征处理能力。

### 主要发现

在五个公共基准数据集上的大量实验表明，GFMFR持续优于最先进的多模态联邦推荐基线方法。

### 结论

GFMFR有效地解决了联邦推荐中整合多模态特征的挑战，通过将表示学习转移到服务器端和采用群体感知的融合方法，提高了系统的性能和效率。

### 翻译

联邦推荐(FR)是一种新的学习范式，以隐私保护的方式解决学习排序问题。如何将多模态特征整合到联邦推荐中，在效率、分布异质性和细粒度对齐方面仍然是一个开放的挑战。为了应对这些挑战，我们提出了联邦推荐环境中的新型多模态融合机制(GFMFR)。具体来说，它将多模态表示学习转移到服务器，服务器存储项目内容并使用高容量编码器生成丰富的表示，减轻了客户端的开销。此外，感知群体的项目表示融合方法使相似用户之间能够进行细粒度的知识共享，同时保留个人偏好。所提出的融合损失可以简单地插入到任何现有的联邦推荐系统中，通过添加多模态特征增强其能力。在五个公共基准数据集上的大量实验表明，GFMFR持续优于最先进的多模态联邦推荐基线方法。


### 论文摘要

Federated Recommendation (FR) is a new learning paradigm to tackle the learn-to-rank problem in a privacy-preservation manner. How to integrate multi-modality features into federated recommendation is still an open challenge in terms of efficiency, distribution heterogeneity, and fine-grained alignment. To address these challenges, we propose a novel multimodal fusion mechanism in federated recommendation settings (GFMFR). Specifically, it offloads multimodal representation learning to the server, which stores item content and employs a high-capacity encoder to generate expressive representations, alleviating client-side overhead. Moreover, a group-aware item representation fusion approach enables fine-grained knowledge sharing among similar users while retaining individual preferences. The proposed fusion loss could be simply plugged into any existing federated recommender systems empowering their capability by adding multi-modality features. Extensive experiments on five public benchmark datasets demonstrate that GFMFR consistently outperforms state-of-the-art multimodal FR baselines.

---

## 9. Efficient Cell Painting Image Representation Learning via Cross-Well Aligned Masked Siamese Network

**论文链接:** [http://arxiv.org/abs/2509.19896v1](http://arxiv.org/abs/2509.19896v1)

**作者:** Pin-Jui Huang, Yu-Hsuan Liao, SooHeon Kim, NoSeong Park, JongBae Park, DongMyung Shin

**发布时间:** 2025-09-24

**备注:** 9 pages, 3 figures, reference 4 pages

### GPT解析

### 总结

本研究提出了跨孔对齐掩码孪生网络（CWA-MSN），一种新的细胞图像表征学习框架，能够在不同批次中对相同干扰处理的细胞嵌入进行对齐，克服批次效应问题，同时保持数据和参数效率。

### 背景

计算模型预测细胞对化学和遗传干扰的表型反应可加速药物发现，但提取具有生物学意义且批次稳健的细胞绘画表征具有挑战性。传统自监督和对比学习方法需要大规模模型和大量数据，仍难以处理批次效应。

### 目的

开发一种新的表征学习框架，能够在不同培养孔中对相同干扰处理的细胞嵌入进行对齐，强制实现语义一致性，克服批次效应问题。

### 方法

提出跨孔对齐掩码孪生网络（CWA-MSN），集成到掩码孪生架构中，捕获细粒度形态特征，同时保持数据和参数效率。

### 主要发现

在基因-关系检索基准测试中，CWA-MSN优于OpenPhenom和CellCLIP方法，分别提高29%和9%的基准分数，同时使用更少数据（0.2M图像vs 2.2M图像）和更小模型（22M参数vs 1.48B参数）。

### 结论

CWA-MSN是一种简单有效的方法，用于学习细胞图像表征，能够在有限数据和参数预算下实现高效的表型建模。

### 翻译

能够预测细胞对化学和遗传干扰的表型反应的计算模型可以通过优先选择治疗假设和减少昂贵的湿实验室迭代来加速药物发现。然而，提取具有生物学意义且批次稳健的细胞绘画表征仍然具有挑战性。传统的自监督和对比学习方法通常需要大规模模型和/或大量精心策划的数据，仍然难以处理批次效应。我们提出了跨孔对齐掩码孪生网络（CWA-MSN），这是一种新的表征学习框架，在不同培养孔中对相同干扰处理的细胞嵌入进行对齐，强制实现语义一致性，尽管存在批次效应。集成到掩码孪生架构中，这种对齐产生了能够捕获细粒度形态特征的特征，同时保持数据和参数效率。例如，在基因-关系检索基准测试中，CWA-MSN优于最先进的公开可用自监督（OpenPhenom）和对比学习（CellCLIP）方法，分别提高了29%和9%的基准分数，同时在显著较少的数据（例如CWA-MSN使用0.2M图像vs OpenPhenom使用2.2M图像）或更小的模型大小（例如CWA-MSN使用22M参数vs CellCLIP使用1.48B参数）上进行训练。大量实验表明，CWA-MSN是学习细胞图像表征的一种简单有效的方法，能够在有限数据和参数预算下实现高效的表型建模。


### 论文摘要

Computational models that predict cellular phenotypic responses to chemical and genetic perturbations can accelerate drug discovery by prioritizing therapeutic hypotheses and reducing costly wet-lab iteration. However, extracting biologically meaningful and batch-robust cell painting representations remains challenging. Conventional self-supervised and contrastive learning approaches often require a large-scale model and/or a huge amount of carefully curated data, still struggling with batch effects. We present Cross-Well Aligned Masked Siamese Network (CWA-MSN), a novel representation learning framework that aligns embeddings of cells subjected to the same perturbation across different wells, enforcing semantic consistency despite batch effects. Integrated into a masked siamese architecture, this alignment yields features that capture fine-grained morphology while remaining data- and parameter-efficient. For instance, in a gene-gene relationship retrieval benchmark, CWA-MSN outperforms the state-of-the-art publicly available self-supervised (OpenPhenom) and contrastive learning (CellCLIP) methods, improving the benchmark scores by +29\% and +9\%, respectively, while training on substantially fewer data (e.g., 0.2M images for CWA-MSN vs. 2.2M images for OpenPhenom) or smaller model size (e.g., 22M parameters for CWA-MSN vs. 1.48B parameters for CellCLIP). Extensive experiments demonstrate that CWA-MSN is a simple and effective way to learn cell image representation, enabling efficient phenotype modeling even under limited data and parameter budgets.

---

## 10. Adaptive von Mises-Fisher Likelihood Loss for Supervised Deep Time Series Hashing

**论文链接:** [http://arxiv.org/abs/2509.19625v1](http://arxiv.org/abs/2509.19625v1)

**作者:** Juan Manuel Perez, Kevin Garcia, Brooklyn Berry, Dongjin Song, Yifeng Gao

**发布时间:** 2025-09-23

**备注:** 6 pages, 6 figures, Conference: ICMLA 2025

### GPT解析

### 总结

本文提出了一种基于von Mises-Fisher (vMF)哈希损失的时间序列深度哈希方法，通过将数据映射到M维超球面空间来减少信息损失，并将每个数据类建模为遵循不同vMF分布的点，实验结果表明该方法优于现有基线方法。

### 背景

时间序列数据挖掘中的基本任务是创建紧凑的二进制表示。最近，基于深度学习的哈希方法已被证明可以根据语义意义而非原始相似性来索引时间序列。与其他监督表示学习方法不同，监督深度哈希需要一个离散化步骤将实值表示转换为二进制码，但这可能导致显著的信息损失。

### 目的

解决监督深度哈希方法中离散化步骤导致的信息损失问题，提高时间序列索引的效率和准确性。

### 方法

提出了一种von Mises-Fisher (vMF)哈希损失。该深度哈希模型将数据映射到M维超球面空间以有效减少信息损失，并将每个数据类建模为遵循不同vMF分布的点。设计的损失函数旨在最大化每个建模的vMF分布之间的分离，为最大化每个语义不同的数据样本之间的间距提供更好的方法。

### 主要发现

实验结果表明，所提出的方法优于现有的基线方法。

### 结论

通过vMF哈希损失方法，可以有效减少深度哈希中的信息损失，提高时间序列索引的性能。实现已在https://github.com/jmpq97/vmf-hashing公开。

### 翻译

通过创建紧凑的二进制表示来索引时间序列是时间序列数据挖掘中的基本任务。最近，基于深度学习的哈希方法已被证明可以根据语义意义而非原始相似性来索引时间序列。深度哈希的目的是将具有相同语义含义的样本映射到相同的二进制哈希码，从而实现更高效的搜索和检索。与其他监督表示学习方法不同，监督深度哈希需要一个离散化步骤将实值表示转换为二进制码，但这可能导致显著的信息损失。在本文中，我们提出了一种von Mises-Fisher (vMF)哈希损失。所提出的深度哈希模型将数据映射到M维超球面空间以有效减少信息损失，并将每个数据类建模为遵循不同vMF分布的点。设计的损失旨在最大化每个建模的vMF分布之间的分离，为最大化每个语义不同的数据样本之间的间距提供更好的方法。实验结果表明，我们的方法优于现有的基线方法。实现已在https://github.com/jmpq97/vmf-hashing公开。


### 论文摘要

Indexing time series by creating compact binary representations is a fundamental task in time series data mining. Recently, deep learning-based hashing methods have proven effective for indexing time series based on semantic meaning rather than just raw similarity. The purpose of deep hashing is to map samples with the same semantic meaning to identical binary hash codes, enabling more efficient search and retrieval. Unlike other supervised representation learning methods, supervised deep hashing requires a discretization step to convert real-valued representations into binary codes, but this can induce significant information loss. In this paper, we propose a von Mises-Fisher (vMF) hashing loss. The proposed deep hashing model maps data to an M-dimensional hyperspherical space to effectively reduce information loss and models each data class as points following distinct vMF distributions. The designed loss aims to maximize the separation between each modeled vMF distribution to provide a better way to maximize the margin between each semantically different data sample. Experimental results show that our method outperforms existing baselines. The implementation is publicly available at https://github.com/jmpq97/vmf-hashing

---

## 11. Can LLMs Reason Over Non-Text Modalities in a Training-Free Manner? A Case Study with In-Context Representation Learning

**论文链接:** [http://arxiv.org/abs/2509.17552v2](http://arxiv.org/abs/2509.17552v2)

**作者:** Tianle Zhang, Wanlong Fang, Jonathan Woo, Paridhi Latawa, Deepak A. Subramanian, Alvin Chan

**发布时间:** 2025-09-22

**备注:** NeurIPS 2025

### GPT解析

### 总结

本文提出了一种名为上下文表征学习（ICRL）的新方法，允许大型语言模型（LLMs）无需训练即可整合非文本模态表征，实现多模态推理。

### 背景

大型语言模型（LLMs）的性能可通过测试时计算得到增强，但现有将非文本模态表征整合到LLMs中的方法通常需要额外的昂贵监督训练，限制了模型对新领域和模态的即时适应能力。

### 目的

探索以无需训练的方式将非文本基础模型（FMs）的表征整合到文本基础的LLMs中的可行性，并提出一种方法使LLMs能够通过少样本学习自适应地利用非文本模态表征。

### 方法

提出上下文表征学习（ICRL）作为概念验证，用FM表征替换传统上下文学习中的文本输入，使LLM能够在不进行微调的情况下执行多模态推理。在分子领域的一系列任务上评估ICRL，并探讨三个核心研究问题。

### 主要发现

研究探讨了三个核心问题：(i) 如何以无需训练的方式将FM表征映射到LLMs中；(ii) 哪些因素影响ICRL性能；(iii) ICRL有效性的潜在机制。ICRL是首个无需训练即可将非文本模态表征整合到文本基础LLMs中的框架。

### 结论

ICRL为可适应的多模态泛化提供了有前景的方向，使大型语言模型能够无需额外训练即可整合和使用非文本模态信息。

### 翻译

大型语言模型（LLMs）的卓越性能可以通过测试时计算得到增强，这依赖于外部工具甚至其他深度学习模型。然而，现有将非文本模态表征整合到LLMs中的方法通常需要额外的昂贵监督训练，限制了模型对新领域和模态的即时适应能力。在本工作中，我们探索以无需训练的方式将非文本基础模型（FMs）的表征整合到文本基础的LLMs中的可行性。我们提出上下文表征学习（ICRL）作为概念验证，允许LLMs通过少样本学习自适应地利用非文本模态表征。与传统上下文学习不同（后者整合文本-标签对），ICRL用FM表征替换文本输入，使LLM能够在不进行微调的情况下执行多模态推理。我们在分子领域的一系列任务上评估了ICRL，研究了三个核心研究问题：(i) 如何以无需训练的方式将FM表征映射到LLMs中；(ii) 哪些因素影响ICRL性能；(iii) ICRL有效性的潜在机制。据我们所知，ICRL是首个无需训练即可将非文本模态表征整合到文本基础LLMs中的框架，为可适应的多模态泛化提供了有前景的方向。


### 论文摘要

The remarkable performance of Large Language Models (LLMs) can be enhanced with test-time computation, which relies on external tools and even other deep learning models. However, existing approaches for integrating non-text modality representations into LLMs typically require additional costly supervised training, restricting on-the-fly adaptation to new domains and modalities. In this work, we explore the feasibility of integrating representations from non-text foundational models (FMs) into text-based LLMs in a training-free manner. We propose In-Context Representation Learning (ICRL) as a proof-of-concept to allow LLMs to adaptively utilize non-text modality representations with few-shot learning. Unlike traditional in-context learning, which incorporates text-label pairs, ICRL replaces text inputs with FM representations, enabling the LLM to perform multi-modal inference without fine-tuning. We evaluate ICRL on a suite of tasks in the molecular domain, investigating three core research questions: (i) how to map FM representations into LLMs in a training-free manner, (ii) what factors influence ICRL performance, and (iii) what mechanisms underlie the effectiveness of ICRL. To the best of our knowledge, ICRL is the first training-free framework for integrating non-text modality representations into text-based LLMs, presenting a promising direction for adaptable, multi-modal generalization.

---

## 12. Causality-Induced Positional Encoding for Transformer-Based Representation Learning of Non-Sequential Features

**论文链接:** [http://arxiv.org/abs/2509.16629v2](http://arxiv.org/abs/2509.16629v2)

**作者:** Kaichen Xu, Yihang Du, Mianpeng Liu, Zimu Yu, Xiaobo Sun

**发布时间:** 2025-09-20

**备注:** Accepted by NeurIPS 2025

### GPT解析

### 总结

CAPE是一种新颖的位置编码方法，通过识别非顺序特征下的潜在因果结构并嵌入双曲空间，生成因果感知的位置编码，有效增强transformer对非顺序特征数据的处理能力。

### 背景

位置编码对补充transformer中token的位置信息至关重要，但现有方法需要预定义的token/特征顺序，不适合处理现实世界中非顺序但因果相关的特征数据。

### 目的

解决现有位置编码方法的局限性，提出一种适用于非顺序特征数据的位置编码方法。

### 方法

提出CAPE方法，识别非顺序特征下的潜在因果结构并建模为加权有向无环图(DAG)，使用广义结构方程建模方法实现，然后将DAG嵌入双曲空间保留其几何结构，捕捉因果强度和因果特异性两个重要属性，生成因果感知的位置编码并转换为旋转形式以与transformer的自注意力机制集成。

### 主要发现

理论分析表明CAPE生成的旋转位置编码具有三个有价值特性：因果距离引起的衰减、因果普遍性引起的衰减以及对位置扰动的鲁棒性；在合成和真实世界数据集上的实验验证了其理论特性和有效性。

### 结论

CAPE能够有效增强transformer对非顺序特征数据的处理能力，相关代码已公开在https://github.com/Catchxu/CAPE。

### 翻译

位置编码对于补充transformer中token的位置信息至关重要。现有的位置编码方法需要预定义的token/特征顺序，使其不适合处理现实世界中非顺序但因果相关的特征数据。为了解决这一局限性，我们提出了CAPE，一种新颖的方法，它使用广义结构方程建模识别非顺序特征下的潜在因果结构，并将其建模为加权有向无环图(DAG)。然后将DAG嵌入双曲空间，使用基于双曲模型的方法有效保留其几何结构，捕捉两个重要的因果图属性（因果强度和因果特异性）。这一步为特征生成了因果感知的位置编码，然后将其转换为旋转形式以与transformer的自注意力机制集成。理论分析表明，CAPE生成的旋转位置编码具有三个增强自注意力的宝贵特性，包括因果距离引起的衰减、因果普遍性引起的衰减以及对位置扰动的鲁棒性。我们在合成和真实世界数据集上评估了CAPE， empirically证明了其理论特性和在增强transformer处理非顺序特征数据方面的有效性。我们的代码可在https://github.com/Catchxu/CAPE获取。


### 论文摘要

Positional encoding is essential for supplementing transformer with positional information of tokens. Existing positional encoding methods demand predefined token/feature order, rendering them unsuitable for real-world data with non-sequential yet causally-related features. To address this limitation, we propose CAPE, a novel method that identifies underlying causal structure over non-sequential features as a weighted directed acyclic graph (DAG) using generalized structural equation modeling. The DAG is then embedded in hyperbolic space where its geometric structure is well-preserved using a hyperboloid model-based approach that effectively captures two important causal graph properties (causal strength & causal specificity). This step yields causality-aware positional encodings for the features, which are converted into their rotary form for integrating with transformer's self-attention mechanism. Theoretical analysis reveals that CAPE-generated rotary positional encodings possess three valuable properties for enhanced self-attention, including causal distance-induced attenuation, causal generality-induced attenuation, and robustness to positional disturbances. We evaluate CAPE over both synthetic and real-word datasets, empirically demonstrating its theoretical properties and effectiveness in enhancing transformer for data with non-sequential features. Our code is available at https://github.com/Catchxu/CAPE.

---

## 13. Queryable 3D Scene Representation: A Multi-Modal Framework for Semantic Reasoning and Robotic Task Planning

**论文链接:** [http://arxiv.org/abs/2509.20077v1](http://arxiv.org/abs/2509.20077v1)

**作者:** Xun Li, Rodrigo Santa Cruz, Mingze Xi, Hu Zhang, Madhawa Perera, Ziwei Wang, Ahalya Ravendran, Brandon J. Matthews, Feng Xu, Matt Adcock, Dadong Wang, Jiajun Liu

**发布时间:** 2025-09-24

### GPT解析

### 总结

本文提出了一种3D可查询场景表示(3D QSR)框架，帮助机器人理解高级人类指令并在复杂3D环境中执行任务。

### 背景

机器人理解高级人类指令并执行复杂任务的关键挑战在于实现全面场景理解：有意义地解释和与3D环境交互。

### 目的

开发一个智能地图，将精确的几何结构与丰富、人类可理解的语义融合，使机器人能够更好地理解和执行复杂任务。

### 方法

引入3D可查询场景表示(3D QSR)框架，统一三种互补的3D表示：3D一致的新视角渲染和分割、来自3D点云的精确几何、通过3D场景图进行结构化组织。该框架基于对象中心设计，与大型视觉语言模型集成，支持语义查询和对象级信息检索。

### 主要发现

该框架能够促进场景理解，整合空间和语义推理，有效将高级人类指令转化为复杂3D环境中的精确机器人任务规划。

### 结论

3D QSR框架成功解决了机器人理解高级人类指令并执行复杂任务的关键挑战，在模拟环境和真实湿实验室环境中均表现出色。

### 翻译

为了使机器人能够理解高级人类指令并执行复杂任务，关键挑战在于实现全面的场景理解：以有意义的方式解释和与3D环境交互。这需要一个智能地图，将精确的几何结构与丰富、人类可理解的语义融合。为此，我们引入了3D可查询场景表示(3D QSR)，这是一个基于多媒体数据的新框架，统一了三种互补的3D表示：(1)从全景重建中获得的3D一致的新视角渲染和分割，(2)来自3D点云的精确几何，(3)通过3D场景图进行结构化、可扩展的组织。基于对象中心设计，该框架与大型视觉语言模型集成，通过链接多模态对象嵌入实现语义可查询性，并支持几何、视觉和语义信息的对象级检索。检索到的数据随后被加载到机器人任务规划器中用于下游执行。我们在Unity中通过模拟机器人任务规划场景评估了该方法，这些场景由抽象语言指令指导，并使用了室内公共数据集Replica。此外，我们在真实湿实验室环境的数字副本中应用了该方法，以测试QSR支持的机器人任务规划用于应急响应。结果表明，该框架能够促进场景理解，整合空间和语义推理，有效将高级人类指令转化为复杂3D环境中的精确机器人任务规划。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': "论文主要解决机器人如何理解高级人类指令并执行复杂任务的问题，特别是在3D环境中实现全面的场景理解。这个问题很重要，因为机器人需要将抽象的人类指令（如'我渴了'）转化为具体行动，包括推断意图、定位相关物品、评估可用性和规划路径，这对实现人机协作和机器人自主操作至关重要。", '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者通过分析传统3D地图的局限性（主要是几何性，缺乏语义信息）和语义理解与3D几何对齐的挑战，设计了一个多模态框架。他们借鉴了多项现有工作：NeRF等3D重建技术、Panoptic Lifting方法进行语义分割、CLIP等大型视觉-语言模型实现开放词汇理解、以及3D场景图作为结构化表示。作者将这些方法有机结合，创建了一个统一的、可查询的3D场景表示框架。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是创建一个统一的、可查询的3D场景表示（3D QSR），融合几何、语义和结构信息，支持自然语言查询，将人类高层指令转化为机器人可执行的精确任务计划。整体流程包括：1) 构建场景表示（3D全景重建、点云分割、多视角描述生成、场景图构建）；2) 查询场景表示（点云查询、NeRF查询、场景图查询）；3) 连接场景理解与行动（在Unity模拟器中实现机器人导航和任务执行）。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1) 统一的多模态3D表示，结合全景辐射场、精确几何点云和结构化场景图；2) 多视角描述生成，利用大型视觉-语言模型纠正分割错误；3) 两阶段查询方法，显著提高复杂查询（如否定查询）的成功率；4) 场景整合与任务规划机制。相比之前工作，3D QSR克服了传统3D地图缺乏语义、单一模态系统表示不全面、现有查询地图方法忽视几何结构以及3D场景图缺乏视觉几何信息等局限。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '该论文提出了3D可查询场景表示（3D QSR）多模态框架，通过统一几何、语义和结构信息，使机器人能够理解自然语言查询并执行复杂任务，实现了从高层人类指令到精确机器人行动的转化。'}


### 论文摘要

To enable robots to comprehend high-level human instructions and perform complex tasks, a key challenge lies in achieving comprehensive scene understanding: interpreting and interacting with the 3D environment in a meaningful way. This requires a smart map that fuses accurate geometric structure with rich, human-understandable semantics. To address this, we introduce the 3D Queryable Scene Representation (3D QSR), a novel framework built on multimedia data that unifies three complementary 3D representations: (1) 3D-consistent novel view rendering and segmentation from panoptic reconstruction, (2) precise geometry from 3D point clouds, and (3) structured, scalable organization via 3D scene graphs. Built on an object-centric design, the framework integrates with large vision-language models to enable semantic queryability by linking multimodal object embeddings, and supporting object-level retrieval of geometric, visual, and semantic information. The retrieved data are then loaded into a robotic task planner for downstream execution. We evaluate our approach through simulated robotic task planning scenarios in Unity, guided by abstract language instructions and using the indoor public dataset Replica. Furthermore, we apply it in a digital duplicate of a real wet lab environment to test QSR-supported robotic task planning for emergency response. The results demonstrate the framework's ability to facilitate scene understanding and integrate spatial and semantic reasoning, effectively translating high-level human instructions into precise robotic task planning in complex 3D environments.

---

## 14. OmniScene: Attention-Augmented Multimodal 4D Scene Understanding for Autonomous Driving

**论文链接:** [http://arxiv.org/abs/2509.19973v1](http://arxiv.org/abs/2509.19973v1)

**作者:** Pei Liu, Hongliang Lu, Haichao Liu, Haipeng Liu, Xin Liu, Ruoyu Yao, Shengbo Eben Li, Jun Ma

**发布时间:** 2025-09-24

### GPT解析

### 总结

本文提出了一种名为OmniScene的新型类人框架，通过模拟人类视觉系统的场景理解能力，显著提升了自动驾驶系统对复杂环境的认知能力。

### 背景

人类视觉能够将二维观察转化为以自我为中心的三维场景理解，支持翻译复杂场景和展示适应性行为。然而，当前自动驾驶系统缺乏这种能力，主流方法主要依赖基于深度的3D重建而非真正的场景理解。

### 目的

解决自动驾驶系统中缺乏类人场景理解能力的问题，提出一种更接近人类认知方式的框架。

### 方法

提出OmniScene框架，包含：1) OmniVLM视觉语言模型整合多视图和时间感知实现4D场景理解；2) 利用教师-学生架构和知识蒸馏将文本表示嵌入3D实例特征；3) 将特征与人类驾驶行为对齐形成类人感知-理解-行动架构；4) 提出分层融合策略解决多模态整合中的模态贡献不平衡问题。

### 主要发现

在nuScenes数据集上与十多种最先进模型比较，OmniScene在感知、预测、规划和视觉问答任务上均取得优异结果，建立了新的基准。

### 结论

OmniScene通过模拟人类视觉系统的工作方式，有效提升了自动驾驶系统的场景理解能力，使其能更接近人类认知方式处理复杂环境。

### 翻译

人类视觉能够将二维观察转化为以自我为中心的三维场景理解，这支持了翻译复杂场景和展示适应性行为的能力。然而，当前的自动驾驶系统仍缺乏这种能力，主流方法主要依赖基于深度的3D重建，而非真正的场景理解。为了解决这一局限性，我们提出了一种名为OmniScene的新型类人框架。首先，我们引入了OmniScene视觉语言模型，这是一个整合多视图和时间感知的视觉语言框架，用于整体的4D场景理解。然后，利用教师-学生OmniVLM架构和知识蒸馏，我们将文本表示嵌入到3D实例特征中，用于语义监督，丰富特征学习并明确捕获类人注意语义。这些特征表示进一步与人类驾驶行为对齐，形成更类人的感知-理解-行动架构。此外，我们提出了分层融合策略来解决多模态整合过程中模态贡献不平衡的问题。我们的方法能够在多个抽象层次上自适应校准几何和语义特征的相对重要性，实现视觉和文本模态互补线索的协同使用。这种可学习的动态融合能够更细致有效地利用异构信息。我们在nuScenes数据集上对OmniScene进行了全面评估，在各种任务上与十多种最先进的模型进行基准测试。我们的方法始终取得优异结果，在感知、预测、规划和视觉问答方面建立了新的基准。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决自动驾驶系统缺乏真正场景理解能力的问题。当前系统主要依赖基于深度的3D重建，无法像人类那样将二维观察转化为三维场景理解并做出适应性决策。这个问题很重要，因为真正的场景理解是确保自动驾驶系统安全性、适应性和决策质量的关键，特别是在复杂和动态的交通环境中。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者通过分析人类视觉系统的特点（将二维观察转化为三维场景理解）和当前自动驾驶系统的不足（缺乏有效整合感知和场景理解）来设计方法。他们借鉴了多模态信息融合机制、端到端自动驾驶系统和视觉语言模型在自动驾驶中的应用经验。具体来说，他们结合了基于注意力的融合机制、可学习融合策略、稀疏查询范式和知识蒸馏技术，设计了OmniScene框架。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是将人类视觉的注意力机制和场景理解能力引入自动驾驶系统，通过多模态融合实现更全面的4D场景理解。整体流程是：1)接收多视角图像流、操作命令和用户提示；2)学生OmniVLM生成场景文本注释，视觉编码层提取视觉特征；3)使用CLIP模型将文本转换为特征表示；4)通过分层融合策略融合3D实例特征、视觉特征和文本特征；5)教师OmniVLM生成丰富文本描述，学生模型学习关注关键区域；6)提供全面场景表示支持下游任务。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)OmniScene框架实现类人感知-理解-行动架构；2)OmniVLM模型通过教师-学生架构实现注意力知识蒸馏；3)分层融合策略(HFS)解决模态贡献不平衡问题；4)全面的4D场景理解结合几何和语义特征。相比之前工作，传统方法依赖3D重建缺乏场景理解，现有端到端系统未有效整合感知与理解，多模态方法未深度整合视觉和文本，且缺乏明确的类人注意力建模。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'OmniScene通过引入类人注意力机制和分层多模态融合策略，显著提升了自动驾驶系统的4D场景理解能力，在感知、预测、规划和视觉问答等任务上实现了最先进的性能。'}


### 论文摘要

Human vision is capable of transforming two-dimensional observations into an egocentric three-dimensional scene understanding, which underpins the ability to translate complex scenes and exhibit adaptive behaviors. This capability, however, remains lacking in current autonomous driving systems, where mainstream approaches primarily rely on depth-based 3D reconstruction rather than true scene understanding. To address this limitation, we propose a novel human-like framework called OmniScene. First, we introduce the OmniScene Vision-Language Model (OmniVLM), a vision-language framework that integrates multi-view and temporal perception for holistic 4D scene understanding. Then, harnessing a teacher-student OmniVLM architecture and knowledge distillation, we embed textual representations into 3D instance features for semantic supervision, enriching feature learning, and explicitly capturing human-like attentional semantics. These feature representations are further aligned with human driving behaviors, forming a more human-like perception-understanding-action architecture. In addition, we propose a Hierarchical Fusion Strategy (HFS) to address imbalances in modality contributions during multimodal integration. Our approach adaptively calibrates the relative significance of geometric and semantic features at multiple abstraction levels, enabling the synergistic use of complementary cues from visual and textual modalities. This learnable dynamic fusion enables a more nuanced and effective exploitation of heterogeneous information. We evaluate OmniScene comprehensively on the nuScenes dataset, benchmarking it against over ten state-of-the-art models across various tasks. Our approach consistently achieves superior results, establishing new benchmarks in perception, prediction, planning, and visual question answering.

---

## 15. Terra: Hierarchical Terrain-Aware 3D Scene Graph for Task-Agnostic Outdoor Mapping

**论文链接:** [http://arxiv.org/abs/2509.19579v1](http://arxiv.org/abs/2509.19579v1)

**作者:** Chad R. Samuelson, Abigail Austin, Seth Knoop, Blake Romrell, Gabriel R. Slade, Timothy W. McLain, Joshua G. Mangelson

**发布时间:** 2025-09-23

### GPT解析

### 总结

本研究提出了一种结合室内3D场景图技术与户外几何映射和地形感知推理的新方法，用于户外智能自主机器人操作。该方法生成了任务无关的度量-语义稀疏地图，并构建了3D场景图，同时保持轻量级特性以适应自主机器人操作。

### 背景

户外智能自主机器人操作依赖于充分表达的环境地图。传统几何映射方法保留了基本的环境结构信息，但缺乏语义理解和组织，无法支持高级机器人推理。3D场景图(3DSGs)通过整合几何、拓扑和语义关系到多级基于图的地图中解决了这一局限性。户外自主操作通常依赖地形信息，这既与任务需求相关，也与机器人平台的可通行性有关。

### 目的

开发一种结合室内3D场景图技术与户外几何映射和地形感知推理的新方法，为户外环境生成地形感知的位置节点和层次化区域组织，并构建轻量级的3D场景图以支持下游规划任务。

### 方法

提出了一种新方法，将室内3D场景图技术与标准户外几何映射和地形感知推理相结合，生成地形感知的位置节点和层次化组织的户外区域。该方法生成任务无关的度量-语义稀疏地图，并从中构建3D场景图，同时保持轻量级特性以适应自主机器人操作。

### 主要发现

全面评估表明，所提出的3D场景图方法在物体检索方面与最先进的基于相机的3D场景图方法性能相当，在区域分类方面超越了它们，同时保持内存效率。在模拟和真实世界环境中的物体检索和区域监测等多样化机器人任务中，该方法展示了其有效性。

### 结论

该方法成功地将室内3D场景图技术扩展到户外环境，通过整合地形信息提高了地图的语义理解和组织能力，同时保持轻量级特性，使其适合自主机器人操作。在多种任务中展示了其有效性和优越性。

### 翻译

户外智能自主机器人操作依赖于充分表达的环境地图。传统几何映射方法保留了基本的环境结构信息，但缺乏语义理解和组织，无法支持高级机器人推理。3D场景图(3DSGs)通过整合几何、拓扑和语义关系到多级基于图的地图中解决了这一局限性。户外自主操作通常依赖地形信息，这既与任务需求相关，也与机器人平台的可通行性有关。我们提出了一种新方法，将室内3D场景图技术与标准户外几何映射和地形感知推理相结合，为户外环境生成地形感知的位置节点和层次化区域组织。我们的方法生成了任务无关的度量-语义稀疏地图，并从中构建3D场景图以支持下游规划任务，同时保持轻量级特性以适应自主机器人操作。我们的全面评估表明，所提出的3D场景图方法在物体检索方面与最先进的基于相机的3D场景图方法性能相当，在区域分类方面超越了它们，同时保持内存效率。我们在模拟和真实世界环境中的物体检索和区域监测等多样化机器人任务中展示了其有效性。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决户外自主机器人在大规模环境中缺乏有效地图表示的问题，特别是缺乏语义理解和地形感知能力。这个问题很重要，因为户外自主机器人可用于搜索救援、森林火灾监测、食品递送等多种社会应用场景，而现有室内3D场景图方法无法有效扩展到户外环境，且缺乏对地形这一关键导航因素的处理。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先分析了现有3D场景图方法在户外环境中的局限性，如计算密集、内存需求大、缺乏地形感知等。他们借鉴了室内3DSG技术、LiDAR SLAM方法、视觉语言模型(VLM)和基础模型的思想，但针对户外环境特点进行了创新设计。特别是结合了Hydra的GVD方法用于位置节点构建，借鉴了Clio的任务驱动思想但扩展为任务无关的地图构建，并引入专门的地形识别模型解决VLM在识别地形方面的不足。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是将3D场景图生成分为三个阶段：任务无关的度量-语义映射、任务无关的地形感知3DSG构建、任务驱动的3DSG查询和导航。整体流程是：1)通过LiDAR和相机数据构建稀疏几何地图，提取地形和物体语义特征；2)基于地形信息构建层次化位置和区域节点；3)根据任务需求在3DSG上进行查询和导航，支持对象检索、区域监控和地形感知路径规划。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)任务无关的轻量级度量-语义映射方法；2)专门的地形感知3DSG层；3)层次化区域组织；4)使用稀疏点云而非密集网格；5)开放集语义理解；6)任务无关与任务驱动分离的设计。相比之前工作，不同之处在于：专门处理地形信息而非将其视为普通语义类别；使用稀疏表示减少计算和内存需求；支持开放集语义理解；分离地图构建和任务查询阶段，增强地图通用性。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'Terra提出了一种结合地形感知和层次化组织的轻量级3D场景图方法，实现了大规模户外环境中任务无关的度量-语义映射，支持多样化的自主机器人任务。'}


### 论文摘要

Outdoor intelligent autonomous robotic operation relies on a sufficiently expressive map of the environment. Classical geometric mapping methods retain essential structural environment information, but lack a semantic understanding and organization to allow high-level robotic reasoning. 3D scene graphs (3DSGs) address this limitation by integrating geometric, topological, and semantic relationships into a multi-level graph-based map. Outdoor autonomous operations commonly rely on terrain information either due to task-dependence or the traversability of the robotic platform. We propose a novel approach that combines indoor 3DSG techniques with standard outdoor geometric mapping and terrain-aware reasoning, producing terrain-aware place nodes and hierarchically organized regions for outdoor environments. Our method generates a task-agnostic metric-semantic sparse map and constructs a 3DSG from this map for downstream planning tasks, all while remaining lightweight for autonomous robotic operation. Our thorough evaluation demonstrates our 3DSG method performs on par with state-of-the-art camera-based 3DSG methods in object retrieval and surpasses them in region classification while remaining memory efficient. We demonstrate its effectiveness in diverse robotic tasks of object retrieval and region monitoring in both simulation and real-world environments.

---

## 16. EditVerse: Unifying Image and Video Editing and Generation with In-Context Learning

**论文链接:** [http://arxiv.org/abs/2509.20360v1](http://arxiv.org/abs/2509.20360v1)

**作者:** Xuan Ju, Tianyu Wang, Yuqian Zhou, He Zhang, Qing Liu, Nanxuan Zhao, Zhifei Zhang, Yijun Li, Yuanhao Cai, Shaoteng Liu, Daniil Pakhomov, Zhe Lin, Soo Ye Kim, Qiang Xu

**发布时间:** 2025-09-24

### GPT解析

### 总结

EditVerse是一个统一的图像和视频生成与编辑框架，通过将文本、图像和视频表示为统一令牌序列，利用自注意力机制实现跨模态知识转移和灵活处理任意分辨率和持续时间的输入输出。

### 背景

基础模型正朝着统一化和规模化方向发展，图像生成和编辑已从特定任务框架转变为统一框架，但视频生成和编辑因架构限制和数据稀缺性仍处于分散状态。

### 目的

开发一个单一模型框架，能够同时处理图像和视频的生成与编辑任务，克服当前视频生成领域的碎片化问题。

### 方法

将所有模态表示为统一令牌序列，利用自注意力机制实现上下文学习；设计可扩展数据管道收集232K视频编辑样本；结合大规模图像和视频数据集进行联合训练；提出首个基于指令的视频编辑基准测试EditVerseBench。

### 主要发现

EditVerse实现了最先进的性能，超越了现有的开源和商业模型；在跨模态方面表现出涌现的编辑和生成能力。

### 结论

EditVerse成功统一了图像和视频生成与编辑框架，通过创新的表示方法和数据管道解决了视频编辑领域的挑战，为未来多模态统一模型的发展提供了新方向。

### 翻译

近期基础模型的进展清晰地表明了向统一化和规模化发展的趋势，展示了跨不同领域的涌现能力。虽然图像生成和编辑已经从特定任务框架迅速转变为统一框架，但由于架构限制和数据稀缺性，视频生成和编辑仍然处于分散状态。在本工作中，我们介绍了EditVerse，一个用于图像和视频生成和编辑的统一框架，在单一模型内完成。通过将所有模态（即文本、图像和视频）表示为统一的令牌序列，EditVerse利用自注意力实现强大的上下文学习、自然的跨模态知识转移，以及对具有任意分辨率和持续时间的输入和输出的灵活处理。为解决视频编辑训练数据的缺乏，我们设计了一个可扩展的数据管道，整理了232K个视频编辑样本，并将它们与大规模图像和视频数据集结合进行联合训练。此外，我们提出了EditVerseBench，这是第一个涵盖多种任务和分辨率的基于指令的视频编辑基准测试。大量实验和用户研究表明，EditVerse实现了最先进的性能，超越了现有的开源和商业模型，同时在跨模态方面表现出涌现的编辑和生成能力。


### 论文摘要

Recent advances in foundation models highlight a clear trend toward unification and scaling, showing emergent capabilities across diverse domains. While image generation and editing have rapidly transitioned from task-specific to unified frameworks, video generation and editing remain fragmented due to architectural limitations and data scarcity. In this work, we introduce EditVerse, a unified framework for image and video generation and editing within a single model. By representing all modalities, i.e., text, image, and video, as a unified token sequence, EditVerse leverages self-attention to achieve robust in-context learning, natural cross-modal knowledge transfer, and flexible handling of inputs and outputs with arbitrary resolutions and durations. To address the lack of video editing training data, we design a scalable data pipeline that curates 232K video editing samples and combines them with large-scale image and video datasets for joint training. Furthermore, we present EditVerseBench, the first benchmark for instruction-based video editing covering diverse tasks and resolutions. Extensive experiments and user studies demonstrate that EditVerse achieves state-of-the-art performance, surpassing existing open-source and commercial models, while exhibiting emergent editing and generation abilities across modalities.

---

## 17. Video models are zero-shot learners and reasoners

**论文链接:** [http://arxiv.org/abs/2509.20328v1](http://arxiv.org/abs/2509.20328v1)

**作者:** Thaddäus Wiedemer, Yuxuan Li, Paul Vicol, Shixiang Shane Gu, Nick Matarese, Kevin Swersky, Been Kim, Priyank Jaini, Robert Geirhos

**发布时间:** 2025-09-24

**备注:** Project page: https://video-zero-shot.github.io/

### GPT解析

### 总结

大型语言模型的零样本能力推动了自然语言处理向通用基础模型转变，研究表明视频模型可能沿着类似路径发展成为通用视觉基础模型。

### 背景

大型语言模型的零样本能力推动了自然语言处理从特定任务模型向统一的通用基础模型转变，这种转变源于大规模生成模型在网页规模数据上的训练。

### 目的

探究视频模型是否可能沿着类似LLMs的路径发展，成为通用的视觉理解基础模型。

### 方法

研究团队展示了Veo 3模型能够解决多种它没有明确训练过的任务，包括物体分割、边缘检测、图像编辑、理解物理属性、识别物体可供性、模拟工具使用等。

### 主要发现

Veo模型能够感知、建模和操作视觉世界，实现了早期的视觉推理能力，如解决迷宫和对称性问题等。Veo的零样本能力表明视频模型正在成为统一的、通用的视觉基础模型。

### 结论

视频模型可能正朝着成为通用视觉基础模型的方向发展，类似于LLMs在语言领域的发展路径。

### 翻译

大型语言模型(LLMs)显著的零样本能力推动了自然语言处理从特定任务模型向统一的、通用的基础模型转变。这种转变源于简单的原则：在网页规模数据上训练的大规模生成模型。有趣的是，同样的原则也适用于当今的生成式视频模型。视频模型是否可能沿着与LLMs发展通用语言理解能力相似的路径，发展成为通用的视觉理解模型？我们证明Veo 3能够解决多种它没有明确训练过的任务：分割物体、检测边缘、编辑图像、理解物理属性、识别物体可供性、模拟工具使用等等。这些感知、建模和操作视觉世界的能力实现了早期的视觉推理，如解决迷宫和对称性问题。Veo的零样本能力表明视频模型正在成为统一的、通用的视觉基础模型的路径上。


### 论文摘要

The remarkable zero-shot capabilities of Large Language Models (LLMs) have propelled natural language processing from task-specific models to unified, generalist foundation models. This transformation emerged from simple primitives: large, generative models trained on web-scale data. Curiously, the same primitives apply to today's generative video models. Could video models be on a trajectory towards general-purpose vision understanding, much like LLMs developed general-purpose language understanding? We demonstrate that Veo 3 can solve a broad variety of tasks it wasn't explicitly trained for: segmenting objects, detecting edges, editing images, understanding physical properties, recognizing object affordances, simulating tool use, and more. These abilities to perceive, model, and manipulate the visual world enable early forms of visual reasoning like maze and symmetry solving. Veo's emergent zero-shot capabilities indicate that video models are on a path to becoming unified, generalist vision foundation models.

---

## 18. A Versatile Foundation Model for AI-enabled Mammogram Interpretation

**论文链接:** [http://arxiv.org/abs/2509.20271v1](http://arxiv.org/abs/2509.20271v1)

**作者:** Fuxiang Huang, Jiayi Zhu, Yunfang Yu, Yu Xie, Yuan Guo, Qingcong Kong, Mingxiang Wu, Xinrui Jiang, Shu Yang, Jiabo Ma, Ziyi Liu, Zhe Xu, Zhixuan Chen, Yujie Tan, Zifan He, Luhui Mao, Xi Wang, Junlin Hou, Lei Zhang, Qiong Luo, Zhenhui Li, Herui Yao, Hao Chen

**发布时间:** 2025-09-24

**备注:** 64 pages, 7 figures, 40 tables

### GPT解析

### 总结

本文介绍了VersaMammo，一个为乳腺X光片设计的多功能基础模型，通过创建最大的多机构乳腺X光数据集和采用两阶段预训练策略，克服了现有模型的局限性，并在92个临床相关任务中取得了最先进的性能。

### 背景

乳腺癌是全球女性最常见的诊断癌症和癌症相关死亡的主要原因。乳腺X光摄影术对于乳腺病变的早期检测和诊断至关重要。

### 目的

介绍VersaMammo，一个为乳腺X光片设计的多功能基础模型，旨在克服现有基础模型的局限性，包括训练数据多样性不足、模型泛化能力有限，以及缺乏对临床相关任务的全面评估。

### 方法

创建包含706,239张图像来自21个来源的最大多机构乳腺X光数据集；提出两阶段预训练策略：首先通过自监督学习训练教师模型提取可转移特征，然后结合监督学习和知识蒸馏将特征和临床知识转移到VersaMammo；建立包含92个特定任务的基准测试，涵盖病变检测、分割、分类、图像检索和视觉问答五大临床任务类别。

### 主要发现

VersaMammo达到了最先进的性能，在68个内部任务中排名前50，在24个外部验证任务中排名前20，平均排名分别为1.5和1.2，证明了其卓越的泛化能力和临床实用性。

### 结论

VersaMammo为可靠和可扩展的乳腺癌筛查和诊断提供了重大进展，其性能和泛化能力显著优于现有模型。

### 翻译

乳腺癌是全球女性最常见的诊断癌症和癌症相关死亡的主要原因。乳腺X光摄影术对于乳腺病变的早期检测和诊断至关重要。尽管在乳腺X光片分析的基础模型方面取得了进展，但它们的临床转化仍受到几个基本限制的制约，包括训练数据多样性不足、模型泛化能力有限，以及缺乏对临床相关任务的全面评估。在此，我们介绍了VersaMammo，一个为乳腺X光片设计的多功能基础模型，旨在克服这些限制。我们整理了迄今为止最大的多机构乳腺X光数据集，包含来自21个来源的706,239张图像。为了提高泛化能力，我们提出两阶段预训练策略来开发VersaMammo这个乳腺X光基础模型。首先，通过自监督学习训练教师模型，从未标记的乳腺X光片中提取可转移的特征。然后，结合监督学习和知识蒸馏将特征和临床知识转移到VersaMammo中。为确保全面评估，我们建立了一个包含92个特定任务的基准测试，包括68个内部任务和24个外部验证任务，涵盖5个主要临床任务类别：病变检测、分割、分类、图像检索和视觉问答。VersaMammo达到了最先进的性能，在68个特定内部任务中排名前50，在24个外部验证任务中排名前20，平均排名分别为1.5和1.2。这些结果证明了其卓越的泛化能力和临床实用性，为可靠和可扩展的乳腺癌筛查和诊断提供了重大进展。


### 论文摘要

Breast cancer is the most commonly diagnosed cancer and the leading cause of cancer-related mortality in women globally. Mammography is essential for the early detection and diagnosis of breast lesions. Despite recent progress in foundation models (FMs) for mammogram analysis, their clinical translation remains constrained by several fundamental limitations, including insufficient diversity in training data, limited model generalizability, and a lack of comprehensive evaluation across clinically relevant tasks. Here, we introduce VersaMammo, a versatile foundation model for mammograms, designed to overcome these limitations. We curated the largest multi-institutional mammogram dataset to date, comprising 706,239 images from 21 sources. To improve generalization, we propose a two-stage pre-training strategy to develop VersaMammo, a mammogram foundation model. First, a teacher model is trained via self-supervised learning to extract transferable features from unlabeled mammograms. Then, supervised learning combined with knowledge distillation transfers both features and clinical knowledge into VersaMammo. To ensure a comprehensive evaluation, we established a benchmark comprising 92 specific tasks, including 68 internal tasks and 24 external validation tasks, spanning 5 major clinical task categories: lesion detection, segmentation, classification, image retrieval, and visual question answering. VersaMammo achieves state-of-the-art performance, ranking first in 50 out of 68 specific internal tasks and 20 out of 24 external validation tasks, with average ranks of 1.5 and 1.2, respectively. These results demonstrate its superior generalization and clinical utility, offering a substantial advancement toward reliable and scalable breast cancer screening and diagnosis.

---

## 19. Discovering Association Rules in High-Dimensional Small Tabular Data

**论文链接:** [http://arxiv.org/abs/2509.20113v1](http://arxiv.org/abs/2509.20113v1)

**作者:** Erkan Karabulut, Daniel Daza, Paul Groth, Victoria Degeler

**发布时间:** 2025-09-24

**备注:** This paper was accepted at ECAI 2025 Workshop: 1st International  Workshop on Advanced Neuro-Symbolic Applications (ANSyA)

### GPT解析

### 总结

本文提出了一种基于表格基础模型的Aerial+微调方法，用于解决高维低数据环境下的关联规则挖掘问题，显著提高了规则发现的质量和效率。

### 背景

关联规则挖掘(ARM)旨在发现数据集中特征之间的模式，用于知识发现和可解释机器学习。在高维设置中，规则爆炸和计算开销使得传统算法方法不切实际，神经符号方法如Aerial+虽解决了规则爆炸问题，但在低数据环境下性能有限。

### 目的

解决高维数据中的关联规则挖掘问题，特别是在低数据环境下的挑战，提高规则发现的质量和效率。

### 方法

提出两种基于表格基础模型的Aerial+微调方法，用于高维低数据环境下的关联规则挖掘。

### 主要发现

1) Aerial+在五个真实世界数据集上比最先进的算法和神经符号基线好一到两个数量级；2) 引入了高维低数据设置下的ARM新问题；3) 提出的微调方法在五个真实世界数据集上显著提高了规则质量。

### 结论

提出的基于表格基础模型的Aerial+微调方法能有效处理高维低数据环境下的关联规则挖掘问题，显著提高规则质量。

### 翻译

关联规则挖掘(ARM)旨在以命题规则的形式发现数据集中的特征模式，支持高风险决策中的知识发现和可解释机器学习。然而，在高维设置中，规则爆炸和计算开销使得没有有效搜索空间缩减的流行算法方法不切实际，这些挑战会传播到下游任务。神经符号方法如Aerial+最近被提出以解决ARM中的规则爆炸问题。虽然它们处理了数据的高维性，但也继承了神经网络的局限性，特别是在低数据环境中的性能降低。本文对高维表格数据中的关联规则发现做出了三个关键贡献：首先，我们在五个真实世界数据集上实证表明Aerial+比最先进的算法和神经符号基线好一到两个数量级。其次，我们引入了高维、低数据设置下的ARM新问题，如生物医学领域中具有约18k特征和50个样本的基因表达数据。第三，我们提出了使用表格基础模型对Aerial+进行微调的两种方法。我们的方法在五个真实世界数据集上被证明显著提高了规则质量，证明了它们在低数据、高维场景中的有效性。


### 论文摘要

Association Rule Mining (ARM) aims to discover patterns between features in datasets in the form of propositional rules, supporting both knowledge discovery and interpretable machine learning in high-stakes decision-making. However, in high-dimensional settings, rule explosion and computational overhead render popular algorithmic approaches impractical without effective search space reduction, challenges that propagate to downstream tasks. Neurosymbolic methods, such as Aerial+, have recently been proposed to address the rule explosion in ARM. While they tackle the high dimensionality of the data, they also inherit limitations of neural networks, particularly reduced performance in low-data regimes.   This paper makes three key contributions to association rule discovery in high-dimensional tabular data. First, we empirically show that Aerial+ scales one to two orders of magnitude better than state-of-the-art algorithmic and neurosymbolic baselines across five real-world datasets. Second, we introduce the novel problem of ARM in high-dimensional, low-data settings, such as gene expression data from the biomedicine domain with around 18k features and 50 samples. Third, we propose two fine-tuning approaches to Aerial+ using tabular foundation models. Our proposed approaches are shown to significantly improve rule quality on five real-world datasets, demonstrating their effectiveness in low-data, high-dimensional scenarios.

---

## 20. Hyperspectral Adapter for Semantic Segmentation with Vision Foundation Models

**论文链接:** [http://arxiv.org/abs/2509.20107v1](http://arxiv.org/abs/2509.20107v1)

**作者:** JuanaJuana Valeria Hurtado, Rohit Mohan, Abhinav Valada

**发布时间:** 2025-09-24

### GPT解析

### 总结

本文提出了一种新型高光谱适配器，利用预训练视觉基础模型有效学习高光谱数据，在自动驾驶场景中实现了最先进的语义分割性能。

### 背景

高光谱成像(HSI)捕获空间信息和密集的光谱测量，跨越许多窄波长波段，这种丰富的光谱内容有可能促进机器人的鲁棒感知，特别是在具有复杂成分、变化光照或其他视觉挑战性条件的环境中。

### 目的

解决当前高光谱语义分割方法表现不佳的问题，因为这些方法依赖于针对RGB输入优化的架构和学习框架。

### 方法

提出一种包含光谱转换器和光谱感知空间先验模块的架构，用于提取丰富的空间-光谱特征；并引入模态感知交互块，通过专门的提取和注入机制促进高光谱表示和冻结视觉Transformer特征的有效集成。

### 主要发现

在三个基准自动驾驶数据集上的广泛评估表明，该架构直接使用HSI输入实现了最先进的语义分割性能，优于基于视觉和高光谱的分割方法。

### 结论

新型高光谱适配器能够有效利用高光谱数据，提升在复杂环境中的语义分割性能，为机器人感知提供了新的解决方案。

### 翻译

高光谱成像(HSI)捕获空间信息和密集的光谱测量，跨越众多窄波长波段。这种丰富的光谱内容有可能促进强大的机器人感知，特别是在具有复杂成分、变化光照或其他视觉挑战性条件的环境中。然而，当前的高光谱语义分割方法表现不佳，因为它们依赖于针对RGB输入优化的架构和学习框架。在这项工作中，我们提出了一种新型高光谱适配器，利用预训练的视觉基础模型从高光谱数据中有效学习。我们的架构包含一个光谱转换器和一个光谱感知的空间先验模块，用于提取丰富的空间-光谱特征。此外，我们引入了一种模态感知交互块，通过专门的提取和注入机制促进高光谱表示和冻结视觉Transformer特征的有效集成。在三个基准自动驾驶数据集上的广泛评估表明，我们的架构直接使用HSI输入实现了最先进的语义分割性能，优于基于视觉和高光谱的分割方法。我们在 https://hyperspectraladapter.cs.uni-freiburg.de 提供代码。


### 论文摘要

Hyperspectral imaging (HSI) captures spatial information along with dense spectral measurements across numerous narrow wavelength bands. This rich spectral content has the potential to facilitate robust robotic perception, particularly in environments with complex material compositions, varying illumination, or other visually challenging conditions. However, current HSI semantic segmentation methods underperform due to their reliance on architectures and learning frameworks optimized for RGB inputs. In this work, we propose a novel hyperspectral adapter that leverages pretrained vision foundation models to effectively learn from hyperspectral data. Our architecture incorporates a spectral transformer and a spectrum-aware spatial prior module to extract rich spatial-spectral features. Additionally, we introduce a modality-aware interaction block that facilitates effective integration of hyperspectral representations and frozen vision Transformer features through dedicated extraction and injection mechanisms. Extensive evaluations on three benchmark autonomous driving datasets demonstrate that our architecture achieves state-of-the-art semantic segmentation performance while directly using HSI inputs, outperforming both vision-based and hyperspectral segmentation methods. We make the code available at https://hyperspectraladapter.cs.uni-freiburg.de.

---

## 21. One Filters All: A Generalist Filter for State Estimation

**论文链接:** [http://arxiv.org/abs/2509.20051v1](http://arxiv.org/abs/2509.20051v1)

**作者:** Shiqi Liu, Wenhan Cao, Chang Liu, Zeyu He, Tianyi Zhang, Shengbo Eben Li

**发布时间:** 2025-09-24

**备注:** NeurIPS 2025

### GPT解析

### 总结

本文提出了一个名为LLM-Filter的通用滤波框架，利用大型语言模型进行动态系统中的状态估计，通过将带有噪声的观测值嵌入到文本原型中实现。

### 背景

估计动态系统中的隐藏状态（最优滤波）是科学和工程领域一个长期存在的问题。

### 目的

引入一个利用大型语言模型进行状态估计的通用滤波框架LLM-Filter。

### 方法

LLM-Filter通过将带有噪声的观测值嵌入到文本原型中，利用预训练大型语言模型中的推理知识进行状态估计，并设计了System-as-Prompt (SaP)提示结构。

### 主要发现

1) 状态估计可以从预训练LLMs中的推理知识显著受益；2) 通过与冻结的LLM进行模态对齐，LLM-Filter优于最先进的学习方法；3) 设计的SaP提示结构使LLM能够理解估计任务；4) LLM-Filter表现出卓越的泛化能力，能在变化甚至未见环境中准确执行滤波任务；5) 模型准确度随模型大小和训练时间增加而提高。

### 结论

这些发现使LLM-Filter成为滤波领域的一个有前景的基础模型。

### 翻译

估计动态系统中的隐藏状态，也称为最优滤波，是科学和工程领域一个长期存在的问题。在本文中，我们引入了一个通用滤波框架LLM-Filter，它通过将带有噪声的观测值嵌入到文本原型中，利用大型语言模型进行状态估计。在各种经典动态系统的实验中，我们发现首先，状态估计可以从预训练LLMs中嵌入的推理知识中显著受益。通过与冻结的LLM进行适当的模态对齐，LLM-Filter优于最先进的学习方法。其次，我们仔细设计了提示结构System-as-Prompt (SaP)，包含使LLM能够理解估计任务的任务指令。在这些提示的指导下，LLM-Filter表现出卓越的泛化能力，能够在变化甚至未见过的环境中准确执行滤波任务。我们还观察到LLM-Filter中存在扩展定律行为，其中准确度随模型大小和训练时间的增加而提高。这些发现使LLM-Filter成为滤波领域的一个有前景的基础模型。


### 论文摘要

Estimating hidden states in dynamical systems, also known as optimal filtering, is a long-standing problem in various fields of science and engineering. In this paper, we introduce a general filtering framework, \textbf{LLM-Filter}, which leverages large language models (LLMs) for state estimation by embedding noisy observations with text prototypes. In various experiments for classical dynamical systems, we find that first, state estimation can significantly benefit from the reasoning knowledge embedded in pre-trained LLMs. By achieving proper modality alignment with the frozen LLM, LLM-Filter outperforms the state-of-the-art learning-based approaches. Second, we carefully design the prompt structure, System-as-Prompt (SaP), incorporating task instructions that enable the LLM to understand the estimation tasks. Guided by these prompts, LLM-Filter exhibits exceptional generalization, capable of performing filtering tasks accurately in changed or even unseen environments. We further observe a scaling-law behavior in LLM-Filter, where accuracy improves with larger model sizes and longer training times. These findings make LLM-Filter a promising foundation model of filtering.

---

## 22. Anomaly Detection by Clustering DINO Embeddings using a Dirichlet Process Mixture

**论文链接:** [http://arxiv.org/abs/2509.19997v1](http://arxiv.org/abs/2509.19997v1)

**作者:** Nico Schulthess, Ender Konukoglu

**发布时间:** 2025-09-24

**备注:** Paper accepted at MICCAI 2025

### GPT解析

### 总结

该研究利用基础模型的信息嵌入进行医学影像的无监督异常检测，提出使用狄利克雷过程混合模型(DPMM)替代传统记忆库方法，显著提高了大型医疗数据集的异常检测效率。

### 背景

对于小型数据集，可以使用正常特征的记忆库进行异常检测，但这种方法对于大型医疗数据集计算负担过重，不适用。

### 目的

提出一种适用于大型医疗数据集的高效异常检测方法，减少计算负担同时保持检测性能。

### 方法

使用狄利克雷过程混合模型(DPMM)建模正常DINOv2嵌入的分布，利用成分中心和嵌入之间的相似性作为异常分数函数，创建粗略的异常分割掩码。

### 主要发现

DINOv2的DPMM嵌入在医学影像基准测试中实现了有竞争力的异常检测性能，同时至少将推理时间减少了一半；归一化的DINOv2嵌入比未归一化的特征更符合解剖结构，即使在存在异常的情况下也是如此。

### 结论

该方法有效解决了大型医疗数据集异常检测的计算效率问题，代码已在GitHub平台公开。

### 翻译

在这项工作中，我们利用基础模型的信息嵌入进行医学影像的无监督异常检测。对于小型数据集，最近的研究表明可以使用正常特征的记忆库直接进行异常检测。然而，对于大型医疗数据集，这种方法不适用，因为计算负担显著增加。因此，我们提出使用狄利克雷过程混合模型(DPMM)来建模正常DINOv2嵌入的分布，这是一种非参数混合模型，能根据数据自动调整混合成分的数量。我们不使用记忆库，而是利用成分中心和嵌入之间的相似性作为异常分数函数来创建粗略的异常分割掩码。我们的实验表明，通过DPMM处理的DINOv2嵌入尽管是在自然图像上训练的，但在医学影像基准测试中实现了非常有竞争力的异常检测性能，同时至少将推理时间减少了一半。我们的进一步分析表明，即使在存在异常的情况下，归一化的DINOv2嵌入通常比未归一化的特征更符合解剖结构，使它们成为异常检测的优秀表示。代码可在https://github.com/NicoSchulthess/anomalydino-dpmm获取。


### 论文摘要

In this work, we leverage informative embeddings from foundational models for unsupervised anomaly detection in medical imaging. For small datasets, a memory-bank of normative features can directly be used for anomaly detection which has been demonstrated recently. However, this is unsuitable for large medical datasets as the computational burden increases substantially. Therefore, we propose to model the distribution of normative DINOv2 embeddings with a Dirichlet Process Mixture model (DPMM), a non-parametric mixture model that automatically adjusts the number of mixture components to the data at hand. Rather than using a memory bank, we use the similarity between the component centers and the embeddings as anomaly score function to create a coarse anomaly segmentation mask. Our experiments show that through DPMM embeddings of DINOv2, despite being trained on natural images, achieve very competitive anomaly detection performance on medical imaging benchmarks and can do this while at least halving the computation time at inference. Our analysis further indicates that normalized DINOv2 embeddings are generally more aligned with anatomical structures than unnormalized features, even in the presence of anomalies, making them great representations for anomaly detection. The code is available at https://github.com/NicoSchulthess/anomalydino-dpmm.

---

## 23. Geometric Autoencoder Priors for Bayesian Inversion: Learn First Observe Later

**论文链接:** [http://arxiv.org/abs/2509.19929v1](http://arxiv.org/abs/2509.19929v1)

**作者:** Arnaud Vadeboncoeur, Gregory Duthé, Mark Girolami, Eleni Chatzi

**发布时间:** 2025-09-24

### GPT解析

### 总结

该论文提出了GABI（用于贝叶斯反演的几何自编码器）框架，用于处理具有复杂几何形状的工程系统中的不确定性量化问题。该方法采用'先学习后观察'的范式，从具有不同几何形状的大数据集中提取信息，生成几何感知的生成模型作为贝叶斯反演的先验信息。在多种工程应用中表现出色，预测精度与确定性监督学习方法相当，且在复杂几何形状的挑战性问题上具有良好校准和鲁棒性。

### 背景

不确定性量化（UQ）对于工程应用中的推断至关重要。常见的推断任务是从少量噪声观测中恢复物理系统的全场信息，这通常是一个高度不适定的问题。关键在于，工程系统通常具有复杂且可变的几何形状，这限制了标准贝叶斯UQ方法的使用。

### 目的

开发一种能够处理复杂几何形状的工程系统不确定性量化问题的框架，特别适用于从少量噪声观测中恢复物理系统全场信息的任务。

### 方法

提出GABI框架，学习几何感知的物理响应生成模型，作为贝叶斯反演的几何条件先验。采用'先学习后观察'的范式，从具有不同几何形状的大数据集中提取信息，不需要了解控制PDE、边界条件或观测过程。该框架与架构无关，创造性地使用近似贝叶斯计算（ABC）采样，实现了利用现代GPU硬件的高效实现。

### 主要发现

在监督学习方法可适用的受限情况下，预测精度与确定性监督学习方法相当；在具有复杂几何形状的挑战性问题上，不确定性量化具有良好的校准性和鲁棒性；该方法提供了一个灵活的几何感知'一次训练随处可用'的基础模型，独立于任何特定的观测过程。

### 结论

GABI方法为具有复杂几何形状的工程系统提供了一种有效的不确定性量化解决方案，能够从少量噪声观测中恢复物理系统的全场信息，同时提供可靠的不确定性估计。

### 翻译

不确定性量化（UQ）对于工程应用中的推断至关重要。常见的推断任务是从少量噪声观测中恢复物理系统的全场信息，这通常是一个高度不适定的问题。关键在于，工程系统通常具有复杂且可变的几何形状，这限制了标准贝叶斯UQ方法的使用。在这项工作中，我们引入了用于贝叶斯反演的几何自编码器（GABI），这是一个学习物理响应的几何感知生成模型的框架，作为贝叶斯反演的高度信息丰富的几何条件先验。遵循'先学习后观察'的范式，GABI将从具有不同几何形状的大数据集中提取的信息，不需要了解控制PDE、边界条件或观测过程，转化为丰富的潜在先验。在推断时，该先验与特定观测过程的可能性无缝结合，产生一个适应几何形状的后验分布。我们提出的框架与架构无关。创造性地使用近似贝叶斯计算（ABC）采样产生了一种高效的实现，利用现代GPU硬件。我们在以下测试了我们的方法：矩形域上的稳态热传导；翼型周围的雷诺平均纳维-斯托克斯（RANS）流动；3D车身上的亥姆霍兹共振和源定位；地形上的RANS气流。我们发现：在监督学习方法可适用的受限情况下，预测精度与确定性监督学习方法相当；在具有复杂几何形状的挑战性问题上，不确定性量化具有良好的校准性和鲁棒性。该方法提供了一个灵活的几何感知'一次训练随处可用'的基础模型，独立于任何特定的观测过程。


### 论文摘要

Uncertainty Quantification (UQ) is paramount for inference in engineering applications. A common inference task is to recover full-field information of physical systems from a small number of noisy observations, a usually highly ill-posed problem. Critically, engineering systems often have complicated and variable geometries prohibiting the use of standard Bayesian UQ. In this work, we introduce Geometric Autoencoders for Bayesian Inversion (GABI), a framework for learning geometry-aware generative models of physical responses that serve as highly informative geometry-conditioned priors for Bayesian inversion. Following a ''learn first, observe later'' paradigm, GABI distills information from large datasets of systems with varying geometries, without requiring knowledge of governing PDEs, boundary conditions, or observation processes, into a rich latent prior. At inference time, this prior is seamlessly combined with the likelihood of the specific observation process, yielding a geometry-adapted posterior distribution. Our proposed framework is architecture agnostic. A creative use of Approximate Bayesian Computation (ABC) sampling yields an efficient implementation that utilizes modern GPU hardware. We test our method on: steady-state heat over rectangular domains; Reynold-Averaged Navier-Stokes (RANS) flow around airfoils; Helmholtz resonance and source localization on 3D car bodies; RANS airflow over terrain. We find: the predictive accuracy to be comparable to deterministic supervised learning approaches in the restricted setting where supervised learning is applicable; UQ to be well calibrated and robust on challenging problems with complex geometries. The method provides a flexible geometry-aware train-once-use-anywhere foundation model which is independent of any particular observation process.

---

## 24. Exploration with Foundation Models: Capabilities, Limitations, and Hybrid Approaches

**论文链接:** [http://arxiv.org/abs/2509.19924v1](http://arxiv.org/abs/2509.19924v1)

**作者:** Remo Sasso, Michelangelo Conserva, Dominik Jeurissen, Paulo Rauber

**发布时间:** 2025-09-24

**备注:** 16 pages, 7 figures. Accepted for presentation at the 39th Conference  on Neural Information Processing Systems (NeurIPS 2025) Workshop on the  Foundations of Reasoning in Language Models (FoRLM)

### GPT解析

### 总结

研究基础模型在强化学习探索任务中的能力，特别是在稀疏奖励环境中的零样本探索，发现视觉语言模型能理解高级目标但在低级控制上存在困难，提出混合框架可提高早期学习效率。

### 背景

强化学习中的探索具有挑战性，特别是在稀疏奖励设置中。基础模型拥有强大的语义先验，但它们作为经典强化学习基准中零样本探索代理的能力尚未被充分理解。

### 目的

测试大型语言模型和视觉语言模型在多臂老虎机、网格世界和稀疏奖励Atari游戏中的零样本探索能力，并分析基础模型引导探索而非端到端控制的潜力和限制。

### 方法

在多臂老虎机、网格世界和稀疏奖励Atari上对大型语言模型和视觉语言模型进行基准测试，并在受控的最佳情况下研究一个简单的在线策略混合框架。

### 主要发现

视觉语言模型可以从视觉输入中推断高级目标，但在精确的低级控制上持续失败，存在'知行差距'。在理想化设置中，视觉语言模型指导可以显著提高早期阶段的样本效率。

### 结论

基础模型更适合引导探索而非端到端控制，视觉语言模型指导在理想化环境中能有效提高早期学习效率。

### 翻译

强化学习中的探索仍然具有挑战性，特别是在稀疏奖励设置中。虽然基础模型拥有强大的语义先验，但它们作为经典强化学习基准中零样本探索代理的能力尚未被充分理解。我们在多臂老虎机、网格世界和稀疏奖励Atari上对大型语言模型和视觉语言模型进行基准测试，以测试零样本探索。我们的研究揭示了一个关键局限：虽然视觉语言模型可以从视觉输入中推断高级目标，但它们在精确的低级控制上持续失败：即'知行差距'。为了分析这一差距的潜在桥梁，我们在受控的最佳情况下研究了一个简单的在线策略混合框架。我们在这种理想化设置中的结果表明，视觉语言模型指导可以显著提高早期阶段的样本效率，这为使用基础模型引导探索而非端到端控制的潜力和限制提供了清晰的分析。


### 论文摘要

Exploration in reinforcement learning (RL) remains challenging, particularly in sparse-reward settings. While foundation models possess strong semantic priors, their capabilities as zero-shot exploration agents in classic RL benchmarks are not well understood. We benchmark LLMs and VLMs on multi-armed bandits, Gridworlds, and sparse-reward Atari to test zero-shot exploration. Our investigation reveals a key limitation: while VLMs can infer high-level objectives from visual input, they consistently fail at precise low-level control: the "knowing-doing gap". To analyze a potential bridge for this gap, we investigate a simple on-policy hybrid framework in a controlled, best-case scenario. Our results in this idealized setting show that VLM guidance can significantly improve early-stage sample efficiency, providing a clear analysis of the potential and constraints of using foundation models to guide exploration rather than for end-to-end control.

---

## 25. Towards Self-Supervised Foundation Models for Critical Care Time Series

**论文链接:** [http://arxiv.org/abs/2509.19885v1](http://arxiv.org/abs/2509.19885v1)

**作者:** Katja Naasunnguaq Jagd, Rachael DeVries, Ole Winther

**发布时间:** 2025-09-24

**备注:** Accepted to NeurIPS 2025 workshop Learning from Time Series for  Health (TS4H)

### GPT解析

### 总结

本文介绍了一种基于双轴变换器(BAT)的重症监护时间序列早期预训练基础模型，展示了其在小数据集上优于监督基线的迁移学习能力，为资源有限环境下的临床应用提供了可能。

### 背景

医疗领域的基础模型近年来发展迅速，但重症监护时间序列的基础模型由于数据集规模有限且可用性低，研究相对较少。

### 目的

开发一种基于双轴变换器(BAT)的早期预训练基础模型，用于处理重症监护时间序列数据。

### 方法

使用汇集的电子健康记录数据集对模型进行预训练，然后在不同于训练源的数据集上进行微调，用于死亡率预测任务。

### 主要发现

该模型在小数据集（少于5,000个样本）的情况下表现出色，通过迁移学习超越了传统的监督学习方法。

### 结论

自监督基础模型在重症监护时间序列领域具有潜力，能够支持资源有限环境下的可推广和稳健的临床应用。

### 翻译

近年来，医疗领域的基础模型发展迅速，但重症监护时间序列的基础模型由于数据集规模有限且可用性低，研究相对不足。在本工作中，我们介绍了一种基于双轴变换器(BAT)的重症监护时间序列早期预训练基础模型，该模型在汇集的电子健康记录数据集上进行训练。我们在不同于训练源的数据集上对模型进行微调以进行死亡率预测，在小数据集（少于5,000个样本）的情况下，它优于监督基线。这些贡献强调了自监督基础模型在支持资源有限环境下的可推广和稳健的临床应用方面的潜力。


### 论文摘要

Domain-specific foundation models for healthcare have expanded rapidly in recent years, yet foundation models for critical care time series remain relatively underexplored due to the limited size and availability of datasets. In this work, we introduce an early-stage pre-trained foundation model for critical care time-series based on the Bi-Axial Transformer (BAT), trained on pooled electronic health record datasets. We demonstrate effective transfer learning by fine-tuning the model on a dataset distinct from the training sources for mortality prediction, where it outperforms supervised baselines, particularly for small datasets ($<5,000$). These contributions highlight the potential of self-supervised foundation models for critical care times series to support generalizable and robust clinical applications in resource-limited settings.

---

## 26. 论文ID: 2509.19604v1

**错误:** 无法读取论文信息 - [Errno 2] No such file or directory: 'storage/result/2509.19604v1.json'

---

## 27. A Foundation Chemical Language Model for Comprehensive Fragment-Based Drug Discovery

**论文链接:** [http://arxiv.org/abs/2509.19586v1](http://arxiv.org/abs/2509.19586v1)

**作者:** Alexander Ho, Sukyeong Lee, Francis T. F. Tsai

**发布时间:** 2025-09-23

### GPT解析

### 总结

介绍FragAtlas-62M，一个在最大片段数据集上训练的专业基础模型，覆盖了广泛的片段化学空间

### 背景

化学片段在药物发现中很重要，但缺乏大规模、高质量的片段数据集和专门的基础模型

### 目的

开发一个能生成大量化学有效片段的基础模型，加速药物发现过程

### 方法

构建基于GPT-2的模型（42.7M参数），使用包含6200多万个分子的ZINC-22片段子集进行训练，并通过12个描述符和三种指纹方法进行验证

### 主要发现

模型生成99.90%化学有效的片段；生成的片段与训练分布高度匹配（效应大小<0.4）；保留了53.6%的已知ZINC片段，同时产生22%具有实际相关性的新结构

### 结论

FragAtlas-62M能有效生成多样化的化学片段，模型及相关资源（训练代码、数据、文档和权重）的发布将加速该领域的采用

### 翻译

我们引入了FragAtlas-62M，这是一个专业的基础模型，迄今为止在最大的片段数据集上进行训练。它建立在完整的ZINC-22片段子集上，包含超过6200万个分子，实现了对片段化学空间的前所未有的覆盖。我们基于GPT-2的模型（42.7M参数）生成了99.90%化学有效的片段。跨越12个描述符和三种指纹方法的验证显示，生成的片段与训练分布高度匹配（所有效应大小<0.4）。该模型保留了53.6%的已知ZINC片段，同时产生了22%具有实际相关性的新结构。我们发布了FragAtlas-62M及其训练代码、预处理数据、文档和模型权重，以加速其采用。


### 论文摘要

We introduce FragAtlas-62M, a specialized foundation model trained on the largest fragment dataset to date. Built on the complete ZINC-22 fragment subset comprising over 62 million molecules, it achieves unprecedented coverage of fragment chemical space. Our GPT-2 based model (42.7M parameters) generates 99.90% chemically valid fragments. Validation across 12 descriptors and three fingerprint methods shows generated fragments closely match the training distribution (all effect sizes < 0.4). The model retains 53.6% of known ZINC fragments while producing 22% novel structures with practical relevance. We release FragAtlas-62M with training code, preprocessed data, documentation, and model weights to accelerate adoption.

---

## 28. Mouse-Guided Gaze: Semi-Supervised Learning of Intention-Aware Representations for Reading Detection

**论文链接:** [http://arxiv.org/abs/2509.19574v1](http://arxiv.org/abs/2509.19574v1)

**作者:** Seongsil Heo, Roberto Manduchi

**发布时间:** 2025-09-23

**备注:** Accepted at NeurIPS 2025 Foundation Models for the Brain and Body  NeurIPS 2025 Workshop

### GPT解析

### 总结

这篇论文介绍了一个半监督框架，用于学习放大阅读过程中具有意图意识的视线表示，通过利用鼠标轨迹作为弱监督来解决放大导致的视觉上下文丢失和视线碎片化问题。

### 背景

在放大阅读过程中，用户意图的理解对无障碍界面设计至关重要。然而，放大缩小了视觉上下文，并迫使持续拖动视口，产生碎片化、嘈杂的视线，掩盖了阅读意图。

### 目的

开发一个能够准确识别用户在放大阅读过程中的意图（阅读与浏览）的系统，以改善无障碍界面设计。

### 方法

提出一个半监督框架，首先使用未标记的视线进行预训练以预测鼠标速度，然后进行微调以区分阅读与浏览。同时，联合建模放大视口内的原始视线和重新映射到原始屏幕的补偿视图，以解决放大引起的失真问题。

### 主要发现

在文本和网页数据集上，该方法持续优于监督基线，半监督预训练在具有挑战性的设置中实现了高达7.5%的F1值提升。

### 结论

行为驱动预训练对于稳健的纯视线交互具有重要价值，为开发自适应的无手动辅助工具铺平了道路。

### 翻译

理解放大阅读过程中的用户意图对无障碍界面设计至关重要。然而，放大缩小了视觉上下文，并迫使持续拖动视口，产生碎片化、嘈杂的视线，掩盖了阅读意图。我们提出了一个半监督框架，通过利用鼠标轨迹作为弱监督来学习具有意图意识的视线表示。该模型首先使用未标记的视线进行预训练以预测鼠标速度，然后进行微调以区分阅读与浏览。为了解决放大引起的失真，我们联合建模放大视口内的原始视线和重新映射到原始屏幕的补偿视图，从而恢复了行和段落之间的空间连续性。在文本和网页数据集上，我们的方法持续优于监督基线，半监督预训练在具有挑战性的设置中实现了高达7.5%的F1值提升。这些研究结果强调了行为驱动预训练对于稳健的纯视线交互的价值，为自适应的无手动辅助工具铺平了道路。


### 论文摘要

Understanding user intent during magnified reading is critical for accessible interface design. Yet magnification collapses visual context and forces continual viewport dragging, producing fragmented, noisy gaze and obscuring reading intent. We present a semi-supervised framework that learns intention-aware gaze representations by leveraging mouse trajectories as weak supervision. The model is first pretrained to predict mouse velocity from unlabeled gaze, then fine-tuned to classify reading versus scanning. To address magnification-induced distortions, we jointly model raw gaze within the magnified viewport and a compensated view remapped to the original screen, which restores spatial continuity across lines and paragraphs. Across text and webpage datasets, our approach consistently outperforms supervised baselines, with semi-supervised pretraining yielding up to 7.5% F1 improvement in challenging settings. These findings highlight the value of behavior-driven pretraining for robust, gaze-only interaction, paving the way for adaptive, hands-free accessibility tools.

---

## 29. OmniVLA: An Omni-Modal Vision-Language-Action Model for Robot Navigation

**论文链接:** [http://arxiv.org/abs/2509.19480v1](http://arxiv.org/abs/2509.19480v1)

**作者:** Noriaki Hirose, Catherine Glossop, Dhruv Shah, Sergey Levine

**发布时间:** 2025-09-23

**备注:** 9 pages, 7 figures, 6 tables

### GPT解析

### 总结

本文提出了一种名为OmniVLA的机器人基础模型训练框架，实现了基于视觉导航的全模态目标条件，能够处理2D姿态、自我中心图像和自然语言等多种目标模态及其组合。

### 背景

人类能够灵活解释和组合不同目标规范（如语言指令、空间坐标或视觉参考）进行导航，而现有机器人导航策略多基于单一模态训练，限制了在现实世界场景中的适应性。

### 目的

开发一种训练框架，使机器人基础模型能够支持基于视觉导航的全模态目标条件，提高机器人在复杂环境中的导航能力。

### 方法

利用高容量的视觉-语言-动作（VLA）主干网络，通过随机模态融合策略训练三种主要目标模态：2D姿态、自我中心图像和自然语言，以及它们的组合。

### 主要发现

OmniVLA模型在未见环境中表现出强大泛化能力，对稀缺模态具有鲁棒性，能遵循新自然语言指令，在各种模态上优于专业基线模型，并为调整新模态和新任务提供了灵活基础。

### 结论

OmniVLA为广泛可泛化和灵活的导航策略提供了发展路径，并为构建全模态机器人基础模型展示了可扩展的方法。

### 翻译

人类在导航到目的地时能够灵活地解释和组合不同的目标规范，如语言指令、空间坐标或视觉参考。相比之下，大多数现有的机器人导航策略只针对单一模态进行训练，限制了它们在现实世界场景中的适应性，因为在这些场景中不同形式的目标规范是自然且互补的。在这项工作中，我们提出了一个机器人基础模型的训练框架，使基于视觉的导航能够实现全模态目标条件。我们的方法利用高容量的视觉-语言-动作（VLA）主干网络，并通过随机模态融合策略训练三种主要目标模态：2D姿态、自我中心图像和自然语言，以及它们的组合。这种设计不仅扩大了可用数据集的范围，还鼓励策略发展更丰富的几何、语义和视觉表示。由此产生的OmniVLA模型在未见过的环境中表现出强大的泛化能力，对稀缺模态具有鲁棒性，并且能够遵循新的自然语言指令。我们证明，OmniVVA在各种模态上都优于专业基线模型，并为调整新模态和新任务提供了灵活的基础。我们相信OmniVVA朝着广泛可泛化和灵活的导航策略迈出了一步，并为构建全模态机器人基础模型提供了一条可扩展的路径。我们展示了展示OmniVVA性能的视频，并将在项目页面上发布其检查点和训练代码。


### 论文摘要

Humans can flexibly interpret and compose different goal specifications, such as language instructions, spatial coordinates, or visual references, when navigating to a destination. In contrast, most existing robotic navigation policies are trained on a single modality, limiting their adaptability to real-world scenarios where different forms of goal specification are natural and complementary. In this work, we present a training framework for robotic foundation models that enables omni-modal goal conditioning for vision-based navigation. Our approach leverages a high-capacity vision-language-action (VLA) backbone and trains with three primary goal modalities: 2D poses, egocentric images, and natural language, as well as their combinations, through a randomized modality fusion strategy. This design not only expands the pool of usable datasets but also encourages the policy to develop richer geometric, semantic, and visual representations. The resulting model, OmniVLA, achieves strong generalization to unseen environments, robustness to scarce modalities, and the ability to follow novel natural language instructions. We demonstrate that OmniVLA outperforms specialist baselines across modalities and offers a flexible foundation for fine-tuning to new modalities and tasks. We believe OmniVLA provides a step toward broadly generalizable and flexible navigation policies, and a scalable path for building omni-modal robotic foundation models. We present videos showcasing OmniVLA performance and will release its checkpoints and training code on our project page.

---

## 30. A Realistic Evaluation of Cross-Frequency Transfer Learning and Foundation Forecasting Models

**论文链接:** [http://arxiv.org/abs/2509.19465v1](http://arxiv.org/abs/2509.19465v1)

**作者:** Kin G. Olivares, Malcolm Wolff, Tatiana Konstantinova, Shankar Ramasubramanian, Andrew Gordon Wilson, Andres Potapczynski, Willa Potosnak, Mengfei Cao, Boris Oreshkin, Dmitry Efimov

**发布时间:** 2025-09-23

**备注:** Thirty-Ninth Annual Conference on Neural Information Processing  Systems {NeurIPS 2025}. Recent Advances in Time Series Foundation Models Have  We Reached the 'BERT Moment'?

### GPT解析

### 总结

本文研究了跨频域迁移学习(CFTL)在基础预测模型(FFMs)预训练中的性能评估问题，发现当前评估方法存在严重缺陷，并提出了改进方案。研究证实统计模型及其集合在多个指标上显著优于现有FFMs，同时发现合成数据预训练对FFM性能有积极影响。

### 背景

跨频域迁移学习(CFTL)已成为构建大规模时间序列数据集以预训练基础预测模型(FFMs)的流行框架。然而，当前的基准测试实践无法准确评估CFTL的性能。

### 目的

解决现有CFTL评估实践的局限性，提出更准确的性能评估方法。

### 方法

引入广泛采用的神经预测网络的统一重新实现，使其适应CFTL设置；仅在专有和合成数据上进行预训练，防止测试数据泄露；在15个大型、多样化的公共预测竞赛数据集上进行评估。

### 主要发现

统计模型的准确性经常被低估；统计模型及其集合在所有数据集上的表现明显优于现有的FFMs，sCRPS指标提高超过8.2%，MASE指标提高超过20%；合成数据集预训练确实将FFM的准确性提高了7%。

### 结论

统计模型在时间序列预测任务中表现优于基础预测模型；合成数据预训练对FFM性能有积极影响，但效果不如统计模型显著。

### 翻译

跨频域迁移学习(CFTL)已成为一种流行的框架，用于构建大规模时间序列数据集以预训练基础预测模型(FFMs)。尽管CFTL显示出前景，但当前的基准测试实践无法准确评估其性能。这一不足源于多个因素：过度依赖小规模评估数据集；计算汇总统计量时对样本量处理不当；报告了次优的统计模型；以及未能充分考虑预训练和测试数据集之间重叠的非 negligible 风险。为解决这些局限性，我们引入了广泛采用的神经预测网络的统一重新实现，使其适应CFTL设置；我们仅在专有和合成数据上进行预训练，小心防止测试数据泄露；我们在15个大型、多样化的公共预测竞赛数据集上进行了评估。我们的经验分析表明，统计模型的准确性经常被低估。值得注意的是，我们确认统计模型及其集合在所有数据集上的表现一致优于现有FFMs，sCRPS指标提高超过8.2%，MASE指标提高超过20%。然而，我们也发现合成数据集预训练确实将FFM的准确性提高了7%。


### 论文摘要

Cross-frequency transfer learning (CFTL) has emerged as a popular framework for curating large-scale time series datasets to pre-train foundation forecasting models (FFMs). Although CFTL has shown promise, current benchmarking practices fall short of accurately assessing its performance. This shortcoming stems from many factors: an over-reliance on small-scale evaluation datasets; inadequate treatment of sample size when computing summary statistics; reporting of suboptimal statistical models; and failing to account for non-negligible risks of overlap between pre-training and test datasets. To address these limitations, we introduce a unified reimplementation of widely-adopted neural forecasting networks, adapting them for the CFTL setup; we pre-train only on proprietary and synthetic data, being careful to prevent test leakage; and we evaluate on 15 large, diverse public forecast competition datasets. Our empirical analysis reveals that statistical models' accuracy is frequently underreported. Notably, we confirm that statistical models and their ensembles consistently outperform existing FFMs by more than 8.2% in sCRPS, and by more than 20% MASE, across datasets. However, we also find that synthetic dataset pre-training does improve the accuracy of a FFM by 7% percent.

---

## 31. The Platonic Universe: Do Foundation Models See the Same Sky?

**论文链接:** [http://arxiv.org/abs/2509.19453v1](http://arxiv.org/abs/2509.19453v1)

**作者:** UniverseTBD, :, Kshitij Duraphe, Michael J. Smith, Shashwat Sourav, John F. Wu

**发布时间:** 2025-09-23

**备注:** 9 pages, 3 tables, 1 figure. Accepted as a workshop paper to Machine  Learning and the Physical Sciences at NeurIPS 2025

### GPT解析

### 总结

本研究通过测试柏拉图表示假说(PRH)，探索了不同数据类型训练的基础模型在天文学中的表示收敛性，发现模型容量增加时表示对齐程度提高，支持共享星系天体物理学表示的收敛。

### 背景

柏拉图表示假说(PRH)在天文学中的应用，探索基础模型是否能收敛到共享的天体物理学表示。

### 目的

测量在不同数据类型上训练的基础模型的表示收敛性，测试柏拉图表示假说在天文学中的适用性。

### 方法

使用JWST、HSC、Legacy Survey和DESI的光谱和成像观测数据，比较视觉变换器、自监督模型和天文学特定架构的表示，采用互k近邻分析方法进行评估。

### 主要发现

表示对齐程度通常随着模型容量增加而提高，这支持了模型收敛于共享星系天体物理学表示的假说。

### 结论

天文学基础模型可以采用预训练的通用架构，从而能够利用机器学习社区已经投入的计算资源。

### 翻译

我们通过测量在不同数据类型上训练的一系列基础模型的表示收敛性，在天文学中测试柏拉图表示假说。使用JWST、HSC、Legacy Survey和DESI的光谱和成像观测数据，我们通过互k近邻分析比较了视觉变换器、自监督模型和天文学特定架构的表示。我们观察到一致的扩展趋势：在我们测试的架构中，表示对齐通常随着模型容量增加而提高，支持了向共享星系天体物理学表示收敛的观点。我们的结果表明，天文学基础模型可以使用预训练的通用架构，使我们能够利用更广泛的机器学习社区已经投入的计算投资。


### 论文摘要

We test the Platonic Representation Hypothesis (PRH) in astronomy by measuring representational convergence across a range of foundation models trained on different data types. Using spectroscopic and imaging observations from JWST, HSC, Legacy Survey, and DESI, we compare representations from vision transformers, self-supervised models, and astronomy-specific architectures via mutual $k$-nearest neighbour analysis. We observe consistent scaling: representational alignment generally increases with model capacity across our tested architectures, supporting convergence toward a shared representation of galaxy astrophysics. Our results suggest that astronomical foundation models can use pre-trained general-purpose architectures, allowing us to capitalise on the broader machine learning community's already-spent computational investment.

---

## 32. PPG-Distill: Efficient Photoplethysmography Signals Analysis via Foundation Model Distillation

**论文链接:** [http://arxiv.org/abs/2509.19215v1](http://arxiv.org/abs/2509.19215v1)

**作者:** Juntong Ni, Saurabh Kataria, Shengpu Tang, Carl Yang, Xiao Hu, Wei Jin

**发布时间:** 2025-09-23

**备注:** Accepted at NeurIPS 2025 Workshop on Learning from Time Series for  Health

### GPT解析

### 总结

研究提出了PPG-Distill知识蒸馏框架，用于在资源有限的设备上高效部署大型光电容积脉搏波基础模型，提升性能并减少资源消耗。

### 背景

光电容积描记法(PPG)在可穿戴健康监测中被广泛使用，但大型PPG基础模型难以在资源有限的设备上部署。

### 目的

开发一个知识蒸馏框架，能够在资源受限的可穿戴设备上实现高效的PPG分析。

### 方法

PPG-Distill通过预测级、特征级和补丁级蒸馏来转移全局和局部知识，包含形态学蒸馏以保留局部波形模式，以及节律蒸馏以捕获补丁间的时间结构。

### 主要发现

在心率和房颤检测任务上，PPG-Distill将学生模型的性能提高了高达21.8%，同时实现了7倍的推理速度提升和19倍的内存使用减少。

### 结论

PPG-Distill使在可穿戴设备上高效的PPG分析成为可能，解决了大型模型在资源受限设备上的部署挑战。

### 翻译

光电容积描记法(PPG)在可穿戴健康监测中被广泛使用，但大型PPG基础模型难以在资源有限的设备上部署。我们提出了PPG-Distill，这是一个通过预测级、特征级和补丁级蒸馏来转移全局和局部知识的知识蒸馏框架。PPG-Distill包含形态学蒸馏以保留局部波形模式，以及节律蒸馏以捕获补丁间的时间结构。在心率和房颤检测方面，PPG-Distill将学生模型的性能提高了高达21.8%，同时实现了7倍的推理速度和19倍的内存使用减少，从而实现了在可穿戴设备上的高效PPG分析。


### 论文摘要

Photoplethysmography (PPG) is widely used in wearable health monitoring, yet large PPG foundation models remain difficult to deploy on resource-limited devices. We present PPG-Distill, a knowledge distillation framework that transfers both global and local knowledge through prediction-, feature-, and patch-level distillation. PPG-Distill incorporates morphology distillation to preserve local waveform patterns and rhythm distillation to capture inter-patch temporal structures. On heart rate estimation and atrial fibrillation detection, PPG-Distill improves student performance by up to 21.8% while achieving 7X faster inference and reducing memory usage by 19X, enabling efficient PPG analysis on wearables

---

## 33. An Empirical Study of Testing Practices in Open Source AI Agent Frameworks and Agentic Applications

**论文链接:** [http://arxiv.org/abs/2509.19185v2](http://arxiv.org/abs/2509.19185v2)

**作者:** Mohammed Mehedi Hasan, Hao Li, Emad Fallahzadeh, Gopi Krishnan Rajbahadur, Bram Adams, Ahmed E. Hassan

**发布时间:** 2025-09-23

### GPT解析

### 总结

本研究对基于基础模型的AI代理测试实践进行了首次大规模实证研究，分析了39个开源代理框架和439个代理应用，揭示了测试努力的分布不均和关键盲点。

### 背景

基于基础模型的AI代理在多个领域迅速普及，但其内在的非确定性和不可重现性给测试和质量保证带来挑战。现有基准主要提供任务级别评估，但对开发者在开发过程中如何验证代理内部正确性的理解有限。

### 目的

解决AI代理生态系统中测试实践的知识空白，提供首个大规模实证研究，以了解开发者如何验证AI代理的内部正确性。

### 方法

分析39个开源代理框架和439个代理应用，识别10种不同的测试模式，并将这些模式映射到代理框架和应用的典型架构组件。

### 主要发现

新的代理特定测试方法(如DeepEval)很少使用(约1%)，传统测试模式被广泛采用以管理基础模型不确定性；测试努力分布不均，确定性组件消耗70%以上测试努力，而基于基础模型的计划部分获得不到5%；触发组件(提示)被严重忽视，仅出现在约1%的测试中。

### 结论

研究提供了基础模型代理框架和应用的第一个实证测试基线，揭示了对非确定性的理性但不完整的适应；建议框架开发者改进对新型测试方法的支持，应用开发者采用提示回归测试，研究人员探索采用障碍，以构建更强大可靠的AI代理。

### 翻译

基于基础模型的AI代理正迅速在各个领域获得采用，但其固有的非确定性和不可重现性给测试和质量保证带来挑战。虽然最近的基准提供了任务级别的评估，但对于开发者在开发过程中如何验证这些代理的内部正确性，理解仍然有限。为解决这一空白，我们对AI代理生态系统中的测试实践进行了首次大规模实证研究，分析了39个开源代理框架和439个代理应用。我们确定了十种不同的测试模式，发现新的、代理特定的方法(如DeepEval)很少使用(约1%)，而传统模式(如负面测试和成员测试)被广泛采用以管理基础模型的不确定性。通过将这些模式映射到代理框架和代理应用的典型架构组件，我们揭示了测试努力的根本性倒置：确定性组件(如资源工件(工具)和协调工件(工作流))消耗了70%以上的测试努力，而基于基础模型的计划部分获得不到5%。关键的是，这揭示了一个关键盲点，因为触发组件(提示)仍然被忽视，仅出现在约1%的所有测试中。我们的研究为基于基础模型的代理框架和代理应用提供了第一个实证测试基线，揭示了对非确定性的理性但不完整的适应。为解决这一问题，框架开发者应改进对新型测试方法的支持，应用开发者必须采用提示回归测试，研究人员应探索采用的障碍。加强这些实践对于构建更强大和可靠的AI代理至关重要。


### 论文摘要

Foundation model (FM)-based AI agents are rapidly gaining adoption across diverse domains, but their inherent non-determinism and non-reproducibility pose testing and quality assurance challenges. While recent benchmarks provide task-level evaluations, there is limited understanding of how developers verify the internal correctness of these agents during development.   To address this gap, we conduct the first large-scale empirical study of testing practices in the AI agent ecosystem, analyzing 39 open-source agent frameworks and 439 agentic applications. We identify ten distinct testing patterns and find that novel, agent-specific methods like DeepEval are seldom used (around 1%), while traditional patterns like negative and membership testing are widely adapted to manage FM uncertainty. By mapping these patterns to canonical architectural components of agent frameworks and agentic applications, we uncover a fundamental inversion of testing effort: deterministic components like Resource Artifacts (tools) and Coordination Artifacts (workflows) consume over 70% of testing effort, while the FM-based Plan Body receives less than 5%. Crucially, this reveals a critical blind spot, as the Trigger component (prompts) remains neglected, appearing in around 1% of all tests.   Our findings offer the first empirical testing baseline in FM-based agent frameworks and agentic applications, revealing a rational but incomplete adaptation to non-determinism. To address it, framework developers should improve support for novel testing methods, application developers must adopt prompt regression testing, and researchers should explore barriers to adoption. Strengthening these practices is vital for building more robust and dependable AI agents.

---

## 34. RoSe: Robust Self-supervised Stereo Matching under Adverse Weather Conditions

**论文链接:** [http://arxiv.org/abs/2509.19165v1](http://arxiv.org/abs/2509.19165v1)

**作者:** Yun Wang, Junjie Hu, Junhui Hou, Chenghao Zhang, Renwei Yang, Dapeng Oliver Wu

**发布时间:** 2025-09-23

### GPT解析

### 总结

本文提出了一种在恶劣天气条件下鲁棒的自监督立体匹配方法，通过注入视觉基础模型先验和场景对应先验来解决恶劣天气下性能下降的问题

### 背景

现有的自监督立体匹配方法在良好条件下表现优异，但在夜间、雨天、雾天等恶劣天气条件下性能显著下降

### 目的

解决恶劣天气条件下自监督立体匹配方法性能下降的问题，提高模型在恶劣天气条件下的视差估计能力

### 方法

1) 将视觉基础模型推导的鲁棒先验注入CNN特征提取器；2) 引入场景对应先验构建鲁棒监督信号；3) 创建具有真实天气退化的合成立体数据集；4) 提出包含鲁棒自监督场景对应学习和恶劣天气蒸馏两步骤的训练范式

### 主要发现

恶劣天气条件下性能下降的主要原因是：1) 恶劣天气引入噪声降低可见度，使CNN特征提取器难以处理退化区域；2) 退化区域破坏像素对应关系，导致基于光度一致性假设的监督效果不佳

### 结论

所提出的方法通过注入鲁棒先验和构建新的训练范式，显著提高了恶劣天气条件下的立体匹配性能，实验证明其优于现有最先进方法

### 翻译

最近的自监督立体匹配方法已取得显著进展，但在夜间、雨天、雾天等恶劣天气条件下，其性能显著下降。我们确定了导致性能下降的两个主要弱点。首先，恶劣天气会引入噪声并降低可见度，使基于CNN的特征提取器难以处理反光和无纹理等退化区域。其次，这些退化区域会破坏准确的像素对应关系，导致基于光度一致性假设的监督效果不佳。为解决这些挑战，我们提出将视觉基础模型推导的鲁棒先验注入基于CNN的特征提取器中，以改善恶劣天气条件下的特征表示。然后引入场景对应先验来构建鲁棒的监督信号，而不是仅仅依赖光度一致性假设。具体而言，我们创建了具有真实天气退化的合成立体数据集。这些数据集包含清晰和恶劣的图像对，保持相同的语义上下文和视差，保留了场景对应属性。基于此，我们提出了一种鲁棒的自监督训练范式，包含两个关键步骤：鲁棒自监督场景对应学习和恶劣天气蒸馏。这两个步骤都旨在对齐来自清晰和恶劣图像对的基础场景结果，从而提高模型在恶劣天气影响下的视差估计能力。大量实验证明了我们提出的解决方案的有效性和通用性，其性能优于现有的最先进自监督方法。代码可在GitHub上获取

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决在恶劣天气条件（如夜间、雨天、雾天）下的立体匹配性能下降问题。这个问题在现实中非常重要，因为自动驾驶系统需要在各种天气条件下可靠运行，恶劣天气会影响视觉传感器的性能，可能导致安全隐患。在研究方面，这个问题也很重要，因为现有的自监督立体匹配方法在良好天气条件下表现良好，但在恶劣天气条件下性能显著下降，限制了自监督学习在实际场景中的应用。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先分析了现有自监督立体匹配方法在恶劣天气条件下性能下降的两个主要原因：CNN特征提取器难以处理退化区域，以及基于光度一致性假设的监督失效。作者借鉴了多个现有工作：视觉基础模型（如SAM和DAMv2）提供鲁棒特征，特征金字塔网络（FPN）捕捉细节，CycleGAN生成合成恶劣天气图像对，以及知识蒸馏的思想。作者的创新在于将这些技术整合到一个统一框架中，并专门针对恶劣天气条件下的立体匹配问题设计了反恶劣天气特征增强模块（AFEM）和场景对应学习机制。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是通过引入场景对应先验和视觉基础模型的鲁棒先验，提高立体匹配模型在恶劣天气条件下的鲁棒性。具体来说，作者认为清晰图像对和对应的恶劣天气图像对应该具有相同的语义上下文和视差，这一先验可用于构建鲁棒监督信号。整体流程包括两个步骤：首先进行自监督场景对应学习，构建两个分支处理清晰和恶劣天气图像，通过特征一致性和视差一致性损失确保模型学习不受天气影响的特征；然后进行恶劣天气蒸馏，使用高质量伪标签训练模型，提高在恶劣条件下的性能。此外，还设计了AFEM模块在空间、通道和频域上处理特征，提取退化不变特征。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)首次解决各种恶劣条件下的自监督立体匹配问题；2)提出结合视觉基础模型、FPN和AFEM的鲁棒特征提取器；3)设计包含两种一致性损失的两步自监督训练管道；4)创建保持场景对应属性的合成恶劣天气数据集。相比之前工作，RoSe专门针对多种恶劣天气条件设计，结合了视觉基础模型和CNN的优势，引入场景对应先验替代单纯的光度一致性假设，并采用两步训练策略提高模型在恶劣条件下的鲁棒性。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'RoSe通过结合视觉基础模型的鲁棒特征提取和基于场景对应先验的自监督学习，显著提高了立体匹配模型在夜间、雨天和雾天等恶劣天气条件下的性能，为自动驾驶和场景重建等应用提供了更可靠的深度估计解决方案。'}


### 论文摘要

Recent self-supervised stereo matching methods have made significant progress, but their performance significantly degrades under adverse weather conditions such as night, rain, and fog. We identify two primary weaknesses contributing to this performance degradation. First, adverse weather introduces noise and reduces visibility, making CNN-based feature extractors struggle with degraded regions like reflective and textureless areas. Second, these degraded regions can disrupt accurate pixel correspondences, leading to ineffective supervision based on the photometric consistency assumption. To address these challenges, we propose injecting robust priors derived from the visual foundation model into the CNN-based feature extractor to improve feature representation under adverse weather conditions. We then introduce scene correspondence priors to construct robust supervisory signals rather than relying solely on the photometric consistency assumption. Specifically, we create synthetic stereo datasets with realistic weather degradations. These datasets feature clear and adverse image pairs that maintain the same semantic context and disparity, preserving the scene correspondence property. With this knowledge, we propose a robust self-supervised training paradigm, consisting of two key steps: robust self-supervised scene correspondence learning and adverse weather distillation. Both steps aim to align underlying scene results from clean and adverse image pairs, thus improving model disparity estimation under adverse weather effects. Extensive experiments demonstrate the effectiveness and versatility of our proposed solution, which outperforms existing state-of-the-art self-supervised methods. Codes are available at \textcolor{blue}{https://github.com/cocowy1/RoSe-Robust-Self-supervised-Stereo-Matching-under-Adverse-Weather-Conditions}.

---

## 35. Towards Practical Multi-label Causal Discovery in High-Dimensional Event Sequences via One-Shot Graph Aggregation

**论文链接:** [http://arxiv.org/abs/2509.19112v1](http://arxiv.org/abs/2509.19112v1)

**作者:** Hugo Math, Rainer Lienhart

**发布时间:** 2025-09-23

**备注:** Accepted at NeuRIPS2025 Workshop on Structured Probabilistic  Inference and Generative Modeling

### GPT解析

### 总结

CARGO是一种可扩展的多标签因果发现方法，用于处理稀疏、高维事件序列，通过预训练的因果Transformer和两阶段推理方法实现高效的结构化推理

### 背景

理解事件序列中的因果关系是一个跨领域挑战，特别是在医疗保健或车辆诊断等领域，其中疾病或系统故障等结果标签源于症状或错误代码等先前事件

### 目的

开发一种可扩展的多标签因果发现方法，用于处理包含数千种独特事件类型的稀疏、高维事件序列

### 方法

CARGO使用两个预训练的因果Transformer作为领域特定基础模型，并行推断每个序列的一次性因果图，并使用自适应频率融合聚合它们，以重建标签的全局马尔可夫边界，实现高效的大规模概率推理

### 主要发现

在具有超过29,100种独特事件类型和474个不平衡标签的真实世界汽车故障预测数据集上，CARGO展示了其执行结构化推理的能力

### 结论

CARGO能够有效处理高维稀疏事件序列中的因果关系推断，绕过全数据集条件独立性测试的不可承受成本

### 翻译

理解事件序列中的因果关系，其中疾病或系统故障等结果标签源于症状或错误代码等先前事件，跨医疗保健或车辆诊断等领域仍是一个未解决的挑战。我们介绍了CARGO，一种用于处理包含数千种独特事件类型的稀疏、高维事件序列的可扩展多标签因果发现方法。使用两个预训练的因果Transformer作为事件序列的领域特定基础模型，CARGO并行推断每个序列的一次性因果图，并使用自适应频率融合聚合它们，以重建标签的全局马尔可夫边界。这种两阶段方法能够实现高效的大规模概率推理，同时绕过全数据集条件独立性测试的不可承受成本。我们在一个具有超过29,100种独特事件类型和474个不平衡标签的具有挑战性的真实世界汽车故障预测数据集上的结果证明了CARGO执行结构化推理的能力


### 论文摘要

Understanding causality in event sequences where outcome labels such as diseases or system failures arise from preceding events like symptoms or error codes is critical. Yet remains an unsolved challenge across domains like healthcare or vehicle diagnostics. We introduce CARGO, a scalable multi-label causal discovery method for sparse, high-dimensional event sequences comprising of thousands of unique event types. Using two pretrained causal Transformers as domain-specific foundation models for event sequences. CARGO infers in parallel, per sequence one-shot causal graphs and aggregates them using an adaptive frequency fusion to reconstruct the global Markov boundaries of labels. This two-stage approach enables efficient probabilistic reasoning at scale while bypassing the intractable cost of full-dataset conditional independence testing. Our results on a challenging real-world automotive fault prediction dataset with over 29,100 unique event types and 474 imbalanced labels demonstrate CARGO's ability to perform structured reasoning.

---

## 36. Citrus-V: Advancing Medical Foundation Models with Unified Medical Image Grounding for Clinical Reasoning

**论文链接:** [http://arxiv.org/abs/2509.19090v2](http://arxiv.org/abs/2509.19090v2)

**作者:** Guoxin Wang, Jun Zhao, Xinyi Liu, Yanbo Liu, Xuyang Cao, Chao Li, Zhuoyun Liu, Qintian Sun, Fangru Zhou, Haoqiang Xing, Zhenhong Yang

**发布时间:** 2025-09-23

### GPT解析

### 总结

Citrus-V是一种多模态医学基础模型，整合了图像分析与文本推理能力，在单一框架内实现了病变检测、分割和诊断推理，在多个医学影像任务中表现出色，优于现有开源模型和专家级系统。

### 背景

医学影像为临床诊断提供关键证据，但现有模型通常专注于单一任务，需要多个专业网络，限制了泛化能力。虽然大规模语言和多模态模型有强大推理能力，但临床应用需要精确的视觉定位、多模态整合和思维链推理。

### 目的

开发一个结合图像分析与文本推理的多模态医学基础模型，整合检测、分割和多模态思维链推理，实现像素级病变定位、结构化报告生成和类医师诊断推理。

### 方法

提出新颖的多模态训练方法，发布涵盖推理、检测、分割和文档理解任务的开源数据套件，Citrus-V模型整合了检测、分割和多模态思维链推理能力。

### 主要发现

Citrus-V在多个基准测试中优于现有开源医学模型和专家级影像系统，提供从视觉定位到临床推理的统一流程，支持精确病变量化、自动报告生成和可靠第二意见。

### 结论

Citrus-V代表了一种多模态医学基础模型的新方法，有效结合图像分析与文本推理，在多个医学影像任务中表现出色，通过开源方式发布促进了医学AI领域发展。

### 翻译

医学影像为临床诊断、治疗计划和手术决策提供关键证据，但大多数现有影像模型专注于狭窄领域，需要多个专业网络，限制了其泛化能力。虽然大规模语言和多模态模型展现出强大的推理和多任务能力，但现实世界的临床应用需要精确的视觉定位、多模态整合和思维链推理。我们引入了Citrus-V，这是一个结合图像分析与文本推理的多模态医学基础模型。该模型整合了检测、分割和多模态思维链推理，能够在单一框架内实现像素级病变定位、结构化报告生成和类医师诊断推理。我们提出了一种新颖的多模态训练方法，并发布了一个精心策划的开源数据套件，涵盖推理、检测、分割和文档理解任务。评估表明，Citrus-V在多个基准测试中优于现有的开源医学模型和专家级影像系统，提供了从视觉定位到临床推理的统一流程，支持精确的病变量化、自动报告生成和可靠的第二意见。


### 论文摘要

Medical imaging provides critical evidence for clinical diagnosis, treatment planning, and surgical decisions, yet most existing imaging models are narrowly focused and require multiple specialized networks, limiting their generalization. Although large-scale language and multimodal models exhibit strong reasoning and multi-task capabilities, real-world clinical applications demand precise visual grounding, multimodal integration, and chain-of-thought reasoning. We introduce Citrus-V, a multimodal medical foundation model that combines image analysis with textual reasoning. The model integrates detection, segmentation, and multimodal chain-of-thought reasoning, enabling pixel-level lesion localization, structured report generation, and physician-like diagnostic inference in a single framework. We propose a novel multimodal training approach and release a curated open-source data suite covering reasoning, detection, segmentation, and document understanding tasks. Evaluations demonstrate that Citrus-V outperforms existing open-source medical models and expert-level imaging systems across multiple benchmarks, delivering a unified pipeline from visual grounding to clinical reasoning and supporting precise lesion quantification, automated reporting, and reliable second opinions.

---

## 37. ColorBlindnessEval: Can Vision-Language Models Pass Color Blindness Tests?

**论文链接:** [http://arxiv.org/abs/2509.19070v1](http://arxiv.org/abs/2509.19070v1)

**作者:** Zijian Ling, Han Zhang, Yazhuo Zhou, Jiahao Cui

**发布时间:** 2025-09-23

**备注:** Accepted at the Open Science for Foundation Models (SCI-FM) Workshop  at ICLR 2025

### GPT解析

### 总结

本文提出了ColorBlindnessEval，一个受石原色盲测试启发的视觉对抗场景中评估视觉语言模型(VLMs)鲁棒性的新基准测试。

### 背景

视觉语言模型在复杂视觉环境中的鲁棒性有待提高，特别是在对抗性场景中。

### 目的

评估VLMs在受石原色盲测试启发的视觉对抗场景中的鲁棒性和准确性。

### 方法

创建了一个包含500个类似石原图像的数据集，图像中包含0到99的数字并具有不同的颜色组合；评估了9个VLMs，使用是/否和开放式提示，并与人类参与者进行比较。

### 主要发现

模型在解释对抗环境中的数字方面存在局限性，存在普遍的幻觉问题。

### 结论

需要提高VLMs在复杂视觉环境中的鲁棒性，以确保在现实应用中的准确性。

### 翻译

本文提出了ColorBlindnessEval，一个新颖的基准测试，旨在评估视觉语言模型(VLMs)在受石原色盲测试启发的视觉对抗场景中的鲁棒性。我们的数据集包含500个类似石原的图像，图像中包含0到99的数字并具有不同的颜色组合，挑战VLMs准确识别嵌入在复杂视觉模式中的数字信息。我们使用是/否和开放式提示评估了9个VLMs，并将其性能与人类参与者进行比较。我们的实验揭示了模型在解释对抗环境中数字的能力方面的局限性，突显了普遍存在的幻觉问题。这些发现强调了需要提高VLMs在复杂视觉环境中的鲁棒性。ColorBlindnessEval作为基准测试工具，对于提高VLMs在准确性至关重要的现实应用中的可靠性具有重要价值。


### 论文摘要

This paper presents ColorBlindnessEval, a novel benchmark designed to evaluate the robustness of Vision-Language Models (VLMs) in visually adversarial scenarios inspired by the Ishihara color blindness test. Our dataset comprises 500 Ishihara-like images featuring numbers from 0 to 99 with varying color combinations, challenging VLMs to accurately recognize numerical information embedded in complex visual patterns. We assess 9 VLMs using Yes/No and open-ended prompts and compare their performance with human participants. Our experiments reveal limitations in the models' ability to interpret numbers in adversarial contexts, highlighting prevalent hallucination issues. These findings underscore the need to improve the robustness of VLMs in complex visual environments. ColorBlindnessEval serves as a valuable tool for benchmarking and improving the reliability of VLMs in real-world applications where accuracy is critical.

---

## 38. MAPO: Mixed Advantage Policy Optimization

**论文链接:** [http://arxiv.org/abs/2509.18849v2](http://arxiv.org/abs/2509.18849v2)

**作者:** Wenke Huang, Quan Zhang, Yiyang Fang, Jian Liang, Xuankun Rong, Huanjin Yao, Guancheng Wan, Ke Liang, Wenwen He, Mingjun Li, Leszek Rutkowski, Mang Ye, Bo Du, Dacheng Tao

**发布时间:** 2025-09-23

### GPT解析

### 总结

本文提出了一种名为Mixed Advantage Policy Optimization (MAPO)的简单而有效的GRPO策略，解决了现有方法中的优势反转和优势镜像问题，通过动态重新加权优势函数来适应不同轨迹确定性的样本。

### 背景

基础模型在推理任务上的性能通过强化学习方法（如Group Relative Policy Optimization, GRPO）得到了显著提升，其中优势函数作为GRPO的核心机制用于排序轨迹重要性。

### 目的

解决现有GRPO方法中遇到的优势反转和优势镜像问题，实现不同查询样本间的合理优势分配。

### 方法

揭示轨迹具有不同的确定性，为高确定性轨迹的样本提出优势百分比偏差，并动态重新加权具有不同轨迹确定性的样本的优势函数，使其自适应地考虑样本特定特征。

### 主要发现

通过与相关最先进方法的比较以及不同优势变体的消融研究，验证了MAPO方法的有效性。

### 结论

MAPO是一种简单但有效的GRPO策略，能够解决现有方法中的问题，提高基础模型在推理任务上的性能。

### 翻译

基础模型的强化学习最新进展，如群体相对策略优化（GRPO），显著提高了基础模型在推理任务上的性能。值得注意的是，优势函数在GRPO中作为核心机制用于排序轨迹重要性。然而，现有探索遇到了优势反转和优势镜像问题，这阻碍了不同查询样本间的合理优势分配。在这项工作中，我们提出了一种简单但有效的GRPO策略，混合优势策略优化（MAPO）。我们揭示轨迹具有不同的确定性，并为高确定性轨迹的样本提出了优势百分比偏差。此外，我们动态重新加权具有不同轨迹确定性的样本的优势函数，从而自适应地配置优势函数以考虑样本特定特征。与相关最先进方法的比较以及不同优势变体的消融研究验证了我们方法的有效性。


### 论文摘要

Recent advances in reinforcement learning for foundation models, such as Group Relative Policy Optimization (GRPO), have significantly improved the performance of foundation models on reasoning tasks. Notably, the advantage function serves as a central mechanism in GRPO for ranking the trajectory importance. However, existing explorations encounter both advantage reversion and advantage mirror problems, which hinder the reasonable advantage allocation across different query samples. In this work, we propose an easy but effective GRPO strategy, Mixed Advantage Policy Optimization (MAPO). We reveal that the trajectory appears with different certainty and propose the advantage percent deviation for samples with high-certainty trajectories. Furthermore, we dynamically reweight the advantage function for samples with varying trajectory certainty, thereby adaptively configuring the advantage function to account for sample-specific characteristics. Comparison with related state-of-the-art methods, along with ablation studies on different advantage variants, validates the effectiveness of our approach.

---

## 39. MOMEMTO: Patch-based Memory Gate Model in Time Series Foundation Model

**论文链接:** [http://arxiv.org/abs/2509.18751v1](http://arxiv.org/abs/2509.18751v1)

**作者:** Samuel Yoon, Jongwon Kim, Juyoung Ha, Young Myoung Ko

**发布时间:** 2025-09-23

### GPT解析

### 总结

本文提出MOMEMTO，一种基于时间序列基础模型(TFM)的异常检测方法，通过引入基于补丁的内存模块来缓解过拟合问题。

### 背景

基于重建的深度模型被广泛用于时间序列异常检测，但随着模型容量增加，这些模型往往会过拟合，能准确重建未见过的异常。先前工作通过引入存储正常模式原型的内存架构来缓解，但这些方法训练成本高，且尚未与时间序列基础模型有效集成。

### 目的

解决基于重建的深度模型在时间序列异常检测中的过拟合问题，并降低训练成本，同时与时间序列基础模型有效集成。

### 方法

MOMEMTO包含一个内存模块，用于捕获来自多个域的代表性正常模式，并通过多域训练策略使单个模型能在多个数据集上联合微调。内存模块使用预训练编码器的潜在表示初始化，组织成补丁级别单元，并通过注意力机制更新。

### 主要发现

使用23个单变量基准数据集评估，MOMEMTO作为单一模型在AUC和VUS指标上优于基线方法，并且在少样本学习场景中显著提升了骨干TFM的性能。

### 结论

MOMEMTO成功解决了基于重建的深度模型的过拟合问题，通过基于补丁的内存模块和多域训练策略实现了更好的异常检测性能，特别是在少样本学习场景中表现优异。

### 翻译

最近基于重建的深度模型已被广泛用于时间序列异常检测，但随着其容量和表示能力的增加，这些模型往往会过拟合，经常能准确重建未见过的异常。先前的工作试图通过引入存储正常模式原型的内存架构来缓解这一问题。然而，这些方法存在训练成本高的问题，并且尚未与时间序列基础模型有效集成。为了解决这些挑战，我们提出了MOMEMTO，一种用于异常检测的TFM，通过基于补丁的内存模块增强，以减轻过拟合。该内存模块设计用于捕获来自多个域的代表性正常模式，并通过多域训练策略使单个模型能够在多个数据集上进行联合微调。MOMEMTO使用预训练编码器的潜在表示初始化内存项，将它们组织成补丁级别单元，并通过注意力机制进行更新。我们使用23个单变量基准数据集评估了我们的方法。实验结果表明，MOMEMTO作为单一模型，在AUC和VUS指标上比基线方法获得了更高的分数，并且在少样本学习场景中进一步增强了其骨干TFM的性能。


### 论文摘要

Recently reconstruction-based deep models have been widely used for time series anomaly detection, but as their capacity and representation capability increase, these models tend to over-generalize, often reconstructing unseen anomalies accurately. Prior works have attempted to mitigate this by incorporating a memory architecture that stores prototypes of normal patterns. Nevertheless, these approaches suffer from high training costs and have yet to be effectively integrated with time series foundation models (TFMs). To address these challenges, we propose \textbf{MOMEMTO}, a TFM for anomaly detection, enhanced with a patch-based memory module to mitigate over-generalization. The memory module is designed to capture representative normal patterns from multiple domains and enables a single model to be jointly fine-tuned across multiple datasets through a multi-domain training strategy. MOMEMTO initializes memory items with latent representations from a pre-trained encoder, organizes them into patch-level units, and updates them via an attention mechanism. We evaluate our method using 23 univariate benchmark datasets. Experimental results demonstrate that MOMEMTO, as a single model, achieves higher scores on AUC and VUS metrics compared to baseline methods, and further enhances the performance of its backbone TFM, particularly in few-shot learning scenarios.

---

## 40. Knowledge Transfer from Interaction Learning

**论文链接:** [http://arxiv.org/abs/2509.18733v1](http://arxiv.org/abs/2509.18733v1)

**作者:** Yilin Gao, Kangyi Chen, Zhongxing Peng, Hengjie Lu, Shugong Xu

**发布时间:** 2025-09-23

**备注:** Accepted by ICCV2025

### GPT解析

### 总结

本文提出了一种名为'从交互中学习'(LFI)的认知启发框架，通过显式建模视觉理解作为交互过程，解决了视觉基础模型(VFMs)从视觉语言模型(VLMs)转移知识时的局限性。该方法通过交互查询和基于交互的监督两个技术创新，有效捕获VLMs中的动态交互模式，实现了更忠实和高效的知识转移。

### 背景

当前视觉基础模型(VFMs)在从视觉语言模型(VLMs)转移知识方面存在根本性限制。虽然VLMs在通过统一表征空间建模跨模态交互方面表现出色，但现有的VFMs主要采用结果导向的范式，忽略了底层的交互过程。这种表征差异阻碍了有效的知识转移，限制了VFMs在多样化视觉任务上的泛化能力。

### 目的

解决VFMs从VLMs转移知识时的表征差异问题，通过显式建模视觉理解作为交互过程，实现更忠实和高效的知识转移，提升VFMs在多样化视觉任务上的泛化能力，特别是在跨域设置中的表现。

### 方法

提出'从交互中学习'(LFI)框架，包含两个技术创新：1) 交互查询(Interaction Queries)：在网络层之间保持持久的结构关系；2) 基于交互的监督(interaction-based supervision)：源自VLMs的跨模态注意力机制。通过捕捉预训练VLMs中编码的动态交互模式，实现更忠实和高效的知识转移到VFMs。

### 主要发现

实验结果显示在多个基准测试中取得了一致的改进：在TinyImageNet分类和COCO检测/分割任务上分别实现了3.3和1.6/2.4的mAP/AP绝对提升，参数开销小且收敛更快。该框架在跨域设置中表现尤为出色，在PACS和VLCS上分别实现了2.4和9.3的零样本提升。人类评估进一步证实了其认知一致性，在语义一致性指标上比结果导向方法高出2.7倍。

### 结论

通过显式建模视觉理解作为交互过程，LFI框架有效解决了VFMs从VLMs转移知识时的表征差异问题，实现了更忠实和高效的知识转移，显著提升了VFMs在多样化视觉任务上的性能，特别是在跨域设置中的表现。该方法具有参数开销小、收敛快的特点，且在语义一致性方面表现出色。

### 翻译

当前视觉基础模型(VFMs)在从视觉语言模型(VLMs)转移知识方面面临根本性限制，而VLMs在通过统一表征空间建模跨模态交互方面表现出色，现有的VFMs主要采用结果导向的范式，忽略了底层的交互过程。这种表征差异阻碍了有效的知识转移，限制了VFMs在多样化视觉任务上的泛化能力。我们提出了'从交互中学习'(LFI)，一种认知启发框架，通过将视觉理解明确建模为交互过程来解决这一差距。我们的核心见解是，捕捉预训练VLMs中编码的动态交互模式，可以实现更忠实和高效的知识转移到VFMs。该方法围绕两个技术创新展开：交互查询，在网络层之间保持持久的结构关系；以及基于交互的监督，源自VLMs的跨模态注意力机制。全面的实验证明在多个基准测试中取得了一致的改进，在TinyImageNet分类和COCO检测/分割任务上分别实现了3.3和1.6/2.4的mAP/AP绝对提升，参数开销最小且收敛更快。该框架在跨域设置中表现尤为出色，在PACS和VLCS上分别实现了2.4和9.3的零样本提升。人类评估进一步证实了其认知一致性，在语义一致性指标上比结果导向方法高出2.7倍。


### 论文摘要

Current visual foundation models (VFMs) face a fundamental limitation in transferring knowledge from vision language models (VLMs), while VLMs excel at modeling cross-modal interactions through unified representation spaces, existing VFMs predominantly adopt result-oriented paradigms that neglect the underlying interaction processes. This representational discrepancy hinders effective knowledge transfer and limits generalization across diverse vision tasks. We propose Learning from Interactions (LFI), a cognitive-inspired framework that addresses this gap by explicitly modeling visual understanding as an interactive process. Our key insight is that capturing the dynamic interaction patterns encoded in pre-trained VLMs enables more faithful and efficient knowledge transfer to VFMs. The approach centers on two technical innovations, Interaction Queries, which maintain persistent relational structures across network layers, and interaction-based supervision, derived from the cross-modal attention mechanisms of VLMs. Comprehensive experiments demonstrate consistent improvements across multiple benchmarks, achieving 3.3 and 1.6mAP/2.4AP absolute gains on TinyImageNet classification and COCO detection/segmentation respectively, with minimal parameter overhead and faster convergence. The framework particularly excels in cross-domain settings, delivering 2.4 and 9.3 zero-shot improvements on PACS and VLCS. Human evaluations further confirm its cognitive alignment, outperforming result-oriented methods by 2.7 times in semantic consistency metrics.

---

## 41. RSVG-ZeroOV: Exploring a Training-Free Framework for Zero-Shot Open-Vocabulary Visual Grounding in Remote Sensing Images

**论文链接:** [http://arxiv.org/abs/2509.18711v1](http://arxiv.org/abs/2509.18711v1)

**作者:** Ke Li, Di Wang, Ting Wang, Fuyu Dong, Yiming Zhang, Luyao Zhang, Xiangyu Wang, Shaofeng Li, Quan Wang

**发布时间:** 2025-09-23

### GPT解析

### 总结

RSVG-ZeroOV是一种无需训练的框架，利用冻结的通用基础模型实现零样本开放词汇遥感视觉定位，包含概述、聚焦和演进三个阶段，在无需任务特定训练的情况下提供高效可扩展的解决方案，性能优于现有方法。

### 背景

现有遥感视觉定位方法通常受限于封闭式词汇集，在开放世界场景中应用有限。最近的尝试虽然利用了通用基础模型，但过度依赖昂贵的高质量数据集和耗时的微调。

### 目的

提出一种名为RSVG-ZeroOV的无训练框架，探索冻结的通用基础模型在零样本开放词汇遥感视觉定位中的潜力。

### 方法

RSVG-ZeroOV包含三个关键阶段：(i)概述：利用视觉语言模型获取交叉注意图，捕捉文本查询与视觉区域间的语义相关性；(ii)聚焦：利用扩散模型的细粒度建模先验，填充VLM忽略的对象结构和形状信息；(iii)演进：引入注意力演进模块抑制不相关激活，生成所指对象的纯化分割掩码。

### 主要发现

无需繁琐的任务特定训练，RSVG-ZeroOV提供了一种高效且可扩展的解决方案。大量实验表明，所提出的框架持续优于现有的弱监督和零样本方法。

### 结论

RSVG-ZeroOV框架在开放词汇遥感视觉定位任务中表现出色，无需额外训练即可实现高性能。

### 翻译

遥感视觉定位(RSVG)旨在基于自由形式的自然语言表达式在遥感图像中定位对象。现有方法通常受限于封闭式词汇集，限制了它们在开放世界场景中的适用性。虽然最近尝试利用通用基础模型进行开放词汇RSVG，但它们过度依赖昂贵的高质量数据集和耗时的微调。为解决这些局限性，我们提出了RSVG-ZeroOV，一种无需训练的框架，旨在探索冻结的通用基础模型在零样本开放词汇RSVG中的潜力。具体而言，RSVG-ZeroOV包含三个关键阶段：(i)概述：我们利用视觉语言模型(VLM)获取交叉注意图，捕捉文本查询与视觉区域之间的语义相关性。(ii)聚焦：通过利用扩散模型(DM)的细粒度建模先验，我们填充了对象的结构和形状信息中的空白，这些信息经常被VLM忽略。(iii)演进：引入了一个简单而有效的注意力演进模块，以抑制不相关的激活，从而在所指对象上产生纯化的分割掩码。无需繁琐的任务特定训练，RSVG-ZeroOV提供了一种高效且可扩展的解决方案。大量实验表明，所提出的框架持续优于现有的弱监督和零样本方法。


### 论文摘要

Remote sensing visual grounding (RSVG) aims to localize objects in remote sensing images based on free-form natural language expressions. Existing approaches are typically constrained to closed-set vocabularies, limiting their applicability in open-world scenarios. While recent attempts to leverage generic foundation models for open-vocabulary RSVG, they overly rely on expensive high-quality datasets and time-consuming fine-tuning. To address these limitations, we propose \textbf{RSVG-ZeroOV}, a training-free framework that aims to explore the potential of frozen generic foundation models for zero-shot open-vocabulary RSVG. Specifically, RSVG-ZeroOV comprises three key stages: (i) Overview: We utilize a vision-language model (VLM) to obtain cross-attention\footnote[1]{In this paper, although decoder-only VLMs use self-attention over all tokens, we refer to the image-text interaction part as cross-attention to distinguish it from pure visual self-attention.}maps that capture semantic correlations between text queries and visual regions. (ii) Focus: By leveraging the fine-grained modeling priors of a diffusion model (DM), we fill in gaps in structural and shape information of objects, which are often overlooked by VLM. (iii) Evolve: A simple yet effective attention evolution module is introduced to suppress irrelevant activations, yielding purified segmentation masks over the referred objects. Without cumbersome task-specific training, RSVG-ZeroOV offers an efficient and scalable solution. Extensive experiments demonstrate that the proposed framework consistently outperforms existing weakly-supervised and zero-shot methods.

---

## 42. PU-Gaussian: Point Cloud Upsampling using 3D Gaussian Representation

**论文链接:** [http://arxiv.org/abs/2509.20207v1](http://arxiv.org/abs/2509.20207v1)

**作者:** Mahmoud Khater, Mona Strauss, Philipp von Olshausen, Alexander Reiterer

**发布时间:** 2025-09-24

**备注:** Accepted for the ICCV 2025 e2e3D Workshop. To be published in the  Proceedings of the IEEE/CVF International Conference on Computer Vision  Workshops (ICCVW)

### GPT解析

### 总结

本文提出了一种名为PU-Gaussian的新型点云上采样网络，通过各向异性三维高斯分布建模局部邻域，实现了高效且保持几何可解释性的点云上采样。

### 背景

三维传感器产生的点云通常稀疏且带有噪声，这对需要密集和高保真三维表示的任务构成了挑战。现有方法往往以牺牲几何可解释性或对输入稀疏性的鲁棒性为代价。

### 目的

开发一种能够保持几何可解释性且对输入稀疏性具有鲁棒性的点云上采样方法，解决现有方法的局限性。

### 方法

PU-Gaussian网络使用各向异性三维高斯分布建模每个点的局部邻域，捕获底层几何结构，在局部几何域中通过直接点采样执行上采样。采样生成密集但粗糙的点云后，再通过细化网络调整输出，产生更均匀的分布和更锐利的边缘。

### 主要发现

在PU1K和PUGAN数据集上的广泛测试表明，PU-Gaussian实现了最先进的性能，证明了该方法的有效性。

### 结论

PU-Gaussian成功解决了点云上采样中的几何可解释性和鲁棒性问题，代码和模型权重已公开供社区使用。

### 翻译

三维传感器产生的点云通常是稀疏和有噪声的，这对需要密集和高保真三维表示的任务构成了挑战。先前的工作已经探索了基于隐式特征的上采样和距离函数学习来解决这一问题，但往往以牺牲几何可解释性或对输入稀疏性的鲁棒性为代价。为了克服这些限制，我们提出了PU-Gaussian，一种新型上采样网络，它使用各向异性三维高斯分布建模每个点周围的局部邻域。这些高斯分布捕获了底层几何结构，使我们能够在局部几何域中通过直接点采样明确执行上采样。采样过程生成密集但粗糙的点云。随后的细化网络调整粗糙输出，以产生更均匀的分布和更锐利的边缘。我们在PU1K和PUGAN数据集上进行了广泛测试，证明PU-Gaussian实现了最先进的性能。我们在https://github.com/mvg-inatech/PU-Gaussian.git公开了代码和模型权重。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决点云上采样问题，即如何将稀疏、有噪声的点云转换为密集、高质量的点云。这个问题在现实中非常重要，因为点云是自动驾驶、机器人、增强现实和物体识别等应用的基础数据表示，而稀疏或不规则的点集会显著降低下游任务的性能，特别是当需要细粒度几何细节进行准确解释时。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先分析了现有方法的局限性，如缺乏几何可解释性、对输入稀疏性不够鲁棒、计算开销大等。他们借鉴了Gaussian splatting技术，使用各向异性3D高斯分布来表示局部表面几何。同时采用了类似PU-Net的三阶段框架，但使用Point Transformer作为特征提取器，因为它轻量且能建模局部几何结构。作者将问题形式化为局部分布拟合，通过学习预测稀疏点云中每个点周围的高斯基元，然后直接从这些分布中采样点，从而生成更密集的点云。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是使用各向异性3D高斯分布来建模每个点周围的局部邻域，通过直接在高斯分布中采样来显式地进行上采样，这种方法能够捕捉底层的几何结构。整体实现流程分为三部分：1）高斯预测网络：输入稀疏点云，使用Point Transformer生成特征，预测每个点的尺度、旋转和偏移参数，计算高斯分布的均值和协方差；2）采样模块：从每个预测的高斯分布中采样多个点，使用重参数化技巧进行训练；3）精炼网络：对粗略点云进行精炼，提高空间精度和几何一致性，获得最终的高质量点云。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1）引入PU-Gaussian，使用高斯表示进行点云上采样；2）显式局部几何建模，直接在3D空间中采样；3）提供可解释的上采样过程；4）两阶段架构，先生成粗略点云再精炼；5）轻量级设计，避免计算昂贵的K近邻操作。相比之前的工作，不同之处在于：不依赖特征空间上采样技术；不需要迭代精炼过程；高斯表示比离散体素更灵活；直接在3D空间中拟合高斯，无需解码阶段；在多个数据集上实现了最先进的性能，产生更少的异常值和更细粒度的细节。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '本文提出了PU-Gaussian，一种基于各向异性3D高斯表示的点云上采样方法，通过直接在局部几何域中采样并精炼，实现了比现有方法更高质量、更鲁棒且更可解释的点云上采样效果。'}


### 论文摘要

Point clouds produced by 3D sensors are often sparse and noisy, posing challenges for tasks requiring dense and high-fidelity 3D representations. Prior work has explored both implicit feature-based upsampling and distance-function learning to address this, but often at the expense of geometric interpretability or robustness to input sparsity. To overcome these limitations, we propose PU-Gaussian, a novel upsampling network that models the local neighborhood around each point using anisotropic 3D Gaussian distributions. These Gaussians capture the underlying geometric structure, allowing us to perform upsampling explicitly in the local geometric domain by direct point sampling. The sampling process generates a dense, but coarse, point cloud. A subsequent refinement network adjusts the coarse output to produce a more uniform distribution and sharper edges. We perform extensive testing on the PU1K and PUGAN datasets, demonstrating that PU-Gaussian achieves state-of-the-art performance. We make code and model weights publicly available at https://github.com/mvg-inatech/PU-Gaussian.git.

---

## 43. LidarScout: Direct Out-of-Core Rendering of Massive Point Clouds

**论文链接:** [http://arxiv.org/abs/2509.20198v1](http://arxiv.org/abs/2509.20198v1)

**作者:** Philipp Erler, Lukas Herzberger, Michael Wimmer, Markus Schütz

**发布时间:** 2025-09-24

**DOI:** 10.2312/hpg.20251170

**备注:** Published at High-Performance Graphics 2025

### GPT解析

### 总结

本文提出了一种能够即时可视化大规模地形扫描点云数据集的方法，无需预处理和额外磁盘空间。

### 背景

大规模地形扫描是许多重要任务的基础，如地形测绘、林业、农业和基础设施规划。然而，点云数据集规模巨大，即使是查看这样的基本任务也需要花费数小时到数天的预处理时间。

### 目的

开发一种能够即时可视化包含数百亿点的大规模国家扫描数据集的方法。

### 方法

通过分层加载和渲染策略：首先加载稀疏点子样本初始化概览；然后进行表面重建生成高质量高程图；根据用户导航优先处理视点区域的高程图构建；用户放大时加载完整分辨率数据；不再需要的区域数据被卸载但保留更新后的高程图纹理作为中等细节。

### 主要发现

该方法构成了一种直接的核心外渲染方法，适用于处理TB级压缩的大规模点云数据集，无需预处理和额外磁盘空间。

### 结论

该方法实现了大规模点云数据的即时可视化，显著提高了大规模地形数据的使用效率。

### 翻译

大规模地形扫描是许多重要任务的基础，如地形测绘、林业、农业和基础设施规划。 resulting point cloud data sets are massive in size，以至于即使是查看这样的基本任务也需要花费数小时到数天的预处理时间，才能创建允许实时检查整个数据集的细节层次结构。在本文中，我们提出一种方法，能够即时可视化包含数百亿点的大规模国家扫描数据集。打开数据集时，我们首先加载稀疏的点子样本并初始化整个点云的概览，随后立即进行表面重建过程以生成更高质量、无空洞的高程图。当用户开始导航到感兴趣区域时，我们继续优先处理用户视点的高程图构建。一旦用户放大查看，我们加载该区域的完整分辨率点云数据，并用完整分辨率数据更新相应的高程图纹理。当用户导航到其他地方时，不再需要的完整分辨率点数据会被卸载，但更新后的高程图纹理会保留作为中等细节层次的形式。总体而言，我们的方法构成了一种直接的核心外渲染方法，用于处理大规模点云数据集（TB级，压缩），无需预处理和额外的磁盘空间。源代码、可执行文件、预训练模型和数据集可在以下网址获取：https://github.com/cg-tuwien/lidarscout

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决大规模激光雷达扫描点云数据的快速可视化问题。随着激光扫描技术进步，产生的点云数据集越来越大（包含数百亿到数万亿个点），传统方法需要数小时到数天的预处理才能创建层次细节结构以便实时查看。这个问题很重要，因为这些大规模地形扫描是地形测绘、林业、农业和基础设施规划等任务的基础，快速查看对于寻找数据中的问题、查找特定区域和传输数据都是基本需求。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': "作者思考的核心是如何避免传统方法中耗时的预处理步骤。他们注意到LAZ压缩格式将点存储在块中（每块50,000点），且每个块中的第一个点是不压缩的，这使他们能够快速访问这些'块点'作为稀疏子样本。他们借鉴了点云渲染技术、表面重建技术和神经渲染方法，特别是从稀疏点样本构建高程图的方法。作者采用了类似U-Net的神经网络架构，但进行了修改以适应他们的特定需求。然而，他们指出现有方法大多针对密集点云且需要相机姿态信息，这在他们的应用中不可用，且高程图重建在航空激光雷达扫描方面被忽视了。", '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是通过直接从压缩点云中快速提取稀疏子样本，并利用神经网络从这个稀疏样本生成高质量高程图，实现大规模点云的即时可视化，避免预处理和额外存储需求。整体流程包括：1)快速加载稀疏子样本（读取瓦片边界框，从LAZ文件中加载块点）；2)生成粗糙高程图（将地图分成补丁，使用块点构建插值高程图）；3)神经网络优化（使用小型神经网络优化粗糙高程图）；4)用户交互和动态更新（优先处理用户视角区域，放大时加载完整分辨率数据）；5)渲染（使用CUDA软件光栅化器渲染点云和高程图）。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)交互式点云查看器，无需预处理和额外磁盘空间；2)从压缩点云中高效提取稀疏子样本；3)使用神经网络从稀疏样本预测高质量高程图；4)实现直接核心渲染方法。相比之前的工作，LidarScOUT的不同之处在于：无需预处理（传统方法需数小时到数天）；能处理任意大的数据集（而不仅限于内存容量）；优先处理用户视角区域（而非未定义顺序）；在稀疏样本重建高程图方面表现更好；能直接从压缩LAZ文件访问块点，提高效率。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'LidarScOUT通过直接从压缩点云数据中快速提取稀疏子样本并利用神经网络生成高质量高程图，实现了大规模点云数据的即时可视化，无需预处理或额外存储空间。'}


### 论文摘要

Large-scale terrain scans are the basis for many important tasks, such as topographic mapping, forestry, agriculture, and infrastructure planning. The resulting point cloud data sets are so massive in size that even basic tasks like viewing take hours to days of pre-processing in order to create level-of-detail structures that allow inspecting the data set in their entirety in real time. In this paper, we propose a method that is capable of instantly visualizing massive country-sized scans with hundreds of billions of points. Upon opening the data set, we first load a sparse subsample of points and initialize an overview of the entire point cloud, immediately followed by a surface reconstruction process to generate higher-quality, hole-free heightmaps. As users start navigating towards a region of interest, we continue to prioritize the heightmap construction process to the user's viewpoint. Once a user zooms in closely, we load the full-resolution point cloud data for that region and update the corresponding height map textures with the full-resolution data. As users navigate elsewhere, full-resolution point data that is no longer needed is unloaded, but the updated heightmap textures are retained as a form of medium level of detail. Overall, our method constitutes a form of direct out-of-core rendering for massive point cloud data sets (terabytes, compressed) that requires no preprocessing and no additional disk space. Source code, executable, pre-trained model, and dataset are available at: https://github.com/cg-tuwien/lidarscout

---

## 44. DB-TSDF: Directional Bitmask-based Truncated Signed Distance Fields for Efficient Volumetric Mapping

**论文链接:** [http://arxiv.org/abs/2509.20081v1](http://arxiv.org/abs/2509.20081v1)

**作者:** Jose E. Maese, Luis Merino, Fernando Caballero

**发布时间:** 2025-09-24

### GPT解析

### 总结

该研究提出了一种基于截断符号距离场(TSDF)的高效、仅CPU的体积映射框架，能够实现实时3D重建。

### 背景

现有的TSDF/ESDF方法大多依赖GPU加速，而本研究旨在开发一种完全在CPU上运行的体积映射方法。

### 目的

开发一种高效、仅CPU的体积映射框架，能够在不牺牲运行时性能的情况下实现高分辨率3D重建。

### 方法

使用基于方向位掩码的集成方案，将原始激光雷达点云数据增量融合到体素网格中，生成TSDF表示。

### 主要发现

处理每个点云的时间保持恒定，不随体素网格分辨率变化；完全在CPU上运行的方法能够达到与GPU加速方法相当的速度；生成的地图精度与当代映射技术相当。

### 结论

该CPU-only的体积映射框架能够实现高效、高精度的3D重建，不依赖GPU加速，同时保持实时性能。

### 翻译

本文提出了一种基于截断符号距离场(TSDF)的高效、仅CPU的体积映射框架。该系统使用基于方向位掩码的集成方案，将原始激光雷达点云数据增量融合到体素网格中，生成适合实时3D重建的密集且一致的TSDF表示。该方法的一个关键特点是每个点云的处理时间保持恒定，无论体素网格分辨率如何，能够在不牺牲运行时性能的情况下实现高分辨率映射。与大多数依赖GPU加速的最新TSDF/ESDF方法不同，该方法完全在CPU上运行，在速度上取得了具有竞争力的结果。在真实世界开放数据集上的实验表明，生成的地图精度与当代映射技术相当。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决在仅使用CPU的情况下实现高效高分辨率体积映射的问题。当前大多数TSDF/ESDF方法严重依赖GPU加速，在CPU上运行时计算成本会随着地图分辨率增加而显著增加，这限制了在资源受限的机器人平台上的应用。这个问题很重要，因为许多机器人平台可能没有GPU资源或GPU需要用于其他任务，而CPU-only解决方案可以更广泛地部署在各种平台上。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者借鉴了D-LIO框架中的Truncated Distance Field(TDF)映射后端，但进行了重要改进。他们观察到LiDAR返回的各向异性更新模式，包括每个测量表面点后面的阴影区域，因此设计了方向性核来建模这种模式。作者使用位掩码编码来简化距离计算和更新过程，实现恒定时间操作。他们扩展了原始TDF表示，从无符号变为有符号距离，引入方向性证据积累，并优化内存布局以提高缓存效率。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是使用基于方向性位掩码的截断符号距离场(TSDF)表示，结合预计算的方向性核，实现高效的CPU体积映射。每个体素存储三个字段：距离掩码(32位)、符号标志(1位)和命中计数器(8位)。实现流程：1)初始化体素网格；2)对每个LiDAR点，变换到全局坐标系；3)根据点的方位和仰角选择预计算的方向性核；4)通过按位AND操作更新体素的距离掩码；5)对核阴影区域中的体素递增命中计数器；6)当计数器超过阈值时更新符号位；7)并行处理多个点以提高效率。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点：1)方向性位掩码表示，仅需8字节/体素；2)方向性核，建模各向异性更新和阴影区域；3)恒定时间更新，处理时间与地图分辨率无关；4)完全CPU实现，不依赖GPU；5)有符号距离表示，明确区分自由和占用空间。相比之前的工作不同之处在于：大多数TSDF方法依赖GPU加速，而DB-TSDF完全在CPU上运行；传统方法计算成本随分辨率增加，而DB-TSDF保持恒定；使用方向性核而非各向同性更新；使用位掩码编码简化更新过程。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'DB-TSDF提出了一种基于方向性位掩码的CPU高效TSDF体积映射方法，实现了与GPU方法相媲美的精度，同时保持恒定的计算成本，使其适合资源受限的机器人平台。'}


### 论文摘要

This paper presents a high-efficiency, CPU-only volumetric mapping framework based on a Truncated Signed Distance Field (TSDF). The system incrementally fuses raw LiDAR point-cloud data into a voxel grid using a directional bitmask-based integration scheme, producing dense and consistent TSDF representations suitable for real-time 3D reconstruction. A key feature of the approach is that the processing time per point-cloud remains constant, regardless of the voxel grid resolution, enabling high resolution mapping without sacrificing runtime performance. In contrast to most recent TSDF/ESDF methods that rely on GPU acceleration, our method operates entirely on CPU, achieving competitive results in speed. Experiments on real-world open datasets demonstrate that the generated maps attain accuracy on par with contemporary mapping techniques.

---

## 45. An Overview of Meshfree Collocation Methods

**论文链接:** [http://arxiv.org/abs/2509.20056v1](http://arxiv.org/abs/2509.20056v1)

**作者:** Tomas Halada, Serhii Yaskovets, Abhinav Singh, Ludek Benes, Pratik Suchde, Ivo F. Sbalzarini

**发布时间:** 2025-09-24

**备注:** 55 pages, 259 references, Supplementary Material

### GPT解析

### 总结

论文提供了无网格配置方法的全面概述，用于在非结构化点云上数值逼近微分算子。

### 背景

无网格配置方法不需要计算网格或网格，而是在可能不规则分布的配置点(粒子)上逼近光滑函数及其导数到所需的一致性阶数。

### 目的

回顾文献中的无网格配置方法，追踪关键概念的历史发展，提出方法分类，统一表述这些方法，并提出未来广义推导。

### 方法

回顾文献中的无网格配置方法，根据推导原理进行分类，提出统一表述，展示每种方法如何从统一表述中推导，提出广义推导方法。

### 主要发现

许多无网格配置方法之间存在微妙但重要的差异，这些差异通过统一表述变得明显。

### 结论

提出了无网格配置方法的统一表述和广义推导，为未来研究提供了框架。

### 翻译

我们提供了关于在连续标记的非结构化点云上数值逼近微分算子的无网格配置方法的全面概述。无网格配置方法不需要计算网格或网格。相反，它们在可能不规则分布的配置点(通常称为粒子)上逼近光滑函数及其导数，达到所需的一致性阶数。我们从文献中回顾了几种无网格配置方法，追踪了关键概念的历史发展，并根据推导原理提出了方法分类。尽管我们回顾的一些方法相似或相同，但许多方法之间存在微妙但重要的差异，我们强调了这些差异并进行了讨论。我们提出了无网格配置方法的统一表述，使这些差异变得明显，并展示了每种方法如何从这种表述中推导出来。最后，我们提出了未来无网格配置方法的广义推导。


### 论文摘要

We provide a comprehensive overview of meshfree collocation methods for numerically approximating differential operators on continuously labeled unstructured point clouds. Meshfree collocation methods do not require a computational grid or mesh. Instead, they approximate smooth functions and their derivatives at potentially irregularly distributed collocation points, often called particles, to a desired order of consistency. We review several meshfree collocation methods from the literature, trace the historical development of key concepts, and propose a classification of methods according to their principle of derivation. Although some of the methods reviewed are similar or identical, there are subtle yet important differences between many, which we highlight and discuss. We present a unifying formulation of meshfree collocation methods that renders these differences apparent and show how each method can be derived from this formulation. Finally, we propose a generalized derivation for meshfree collocation methods going forward.

---

## 46. Generalist Robot Manipulation beyond Action Labeled Data

**论文链接:** [http://arxiv.org/abs/2509.19958v1](http://arxiv.org/abs/2509.19958v1)

**作者:** Alexander Spiridonov, Jan-Nico Zaech, Nikolay Nikolov, Luc Van Gool, Danda Pani Paudel

**发布时间:** 2025-09-24

**备注:** Accepted at Conference on Robot Learning 2025

### GPT解析

### 总结

这篇论文提出了一种创新的机器人学习方法，通过利用无标签的人类和机器人演示数据，结合3D动态预测和自监督技术，使机器人能够在没有动作标签的情况下学习新任务，并在真实世界和模拟环境中表现出良好的泛化能力。

### 背景

最近通用机器人操作技术的进步利用了预训练的视觉语言模型和大规模机器人演示数据，能够以零样本方式处理多样化任务。然而，扩展高质量、带动作标签的机器人演示数据仍然是一个关键挑战，现有方法依赖这些数据来获得鲁棒性和泛化能力。

### 目的

提出一种方法，利用没有动作标签的视频(包含人类和/或机器人的动作)，增强开放词汇性能，实现新任务的数据高效学习。

### 方法

在手部或夹持器位置提取密集的动态3D点云，使用提出的3D动态预测器进行自监督，然后使用较小的带标签数据集对预测器进行微调，以实现动作对齐。

### 主要发现

该方法不仅能够从无标签的人类和机器人演示中学习，改进下游通用机器人策略，还使机器人能够在没有动作标签的情况下学习新任务(即动作外泛化)，在真实世界和模拟环境中都有效。

### 结论

该方法解决了高质量、带动作标签的机器人演示数据扩展的挑战，通过利用无标签视频数据，实现了更高效的机器人学习。

### 翻译

最近通用机器人操作技术的进步利用了预训练的视觉语言模型和大规模机器人演示数据，以零样本方式处理多样化任务。一个关键挑战仍然存在：扩展高质量、带动作标签的机器人演示数据，现有方法依赖这些数据来获得鲁棒性和泛化能力。为解决这一问题，我们提出了一种方法，利用没有动作标签的视频(包含人类和/或机器人的动作)，增强开放词汇性能，并实现新任务的数据高效学习。我们的方法在手部或夹持器位置提取密集的动态3D点云，并使用提出的3D动态预测器进行自监督。然后使用较小的带标签数据集对预测器进行微调，以实现动作对齐。我们证明，我们的方法不仅能够从无标签的人类和机器人演示中学习，改进下游通用机器人策略，还使机器人能够在没有动作标签的情况下学习新任务(即动作外泛化)，在真实世界和模拟环境中都有效。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决如何从没有动作标签的人类和机器人演示视频中学习通用机器人操作能力的问题。这个问题很重要，因为现有方法依赖大量带有精确动作标签的机器人演示数据，收集这些数据成本高昂且难以扩展；而互联网上有丰富的人类操作视频资源却未被充分利用；此外，现有方法在训练分布外的任务上表现不佳，限制了机器人在真实世界中的泛化能力。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者认识到现有VLA模型在领域外泛化能力有限，并注意到互联网上有大量无标签人类操作视频蕴含丰富操作知识。他们借鉴了Vision-Language Models（特别是Paligemma）的语义理解能力、自监督学习思想、flow matching技术和Transformer架构。创新点在于将这些技术结合，设计了两阶段训练框架：先从无标签视频中学习通用运动表征，再在有标签数据上进行动作对齐。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是使用动态点云作为人类和机器人操作的通用表征，通过两阶段训练方法学习通用操作能力。第一阶段从无标签视频中提取手部或夹爪的动态点云序列，训练3D动态预测器；第二阶段在有标签机器人数据上训练动作预测器，将动态点云与机器人动作对齐。推理时，模型处理视觉和语言输入，预测机器人动作序列，通过欧拉积分计算具体控制命令。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1) MotoVLA，第一个能利用无标签数据学习通用机器人操作的端到端VLA模型；2) 两阶段训练方法，使用动态点云作为通用表征；3) 首次实现从无标签人类演示到机器人操作的直接迁移。相比之前工作，本文方法不依赖大规模有标签机器人数据，突破了特定任务和小规模策略的限制，避免了逆向动力学模型的需求，解决了重定向方法中的领域差距问题，是首个将无标签数据用于端到端通用VLA架构的研究。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '本文提出了MotoVLA，一种通过两阶段训练方法利用无标签人类和机器人演示视频学习通用机器人操作能力的端到端视觉-语言-行动模型，显著提升了机器人在领域内和领域外任务上的泛化能力。'}


### 论文摘要

Recent advances in generalist robot manipulation leverage pre-trained Vision-Language Models (VLMs) and large-scale robot demonstrations to tackle diverse tasks in a zero-shot manner. A key challenge remains: scaling high-quality, action-labeled robot demonstration data, which existing methods rely on for robustness and generalization. To address this, we propose a method that benefits from videos without action labels - featuring humans and/or robots in action - enhancing open-vocabulary performance and enabling data-efficient learning of new tasks. Our method extracts dense, dynamic 3D point clouds at the hand or gripper location and uses a proposed 3D dynamics predictor for self-supervision. This predictor is then tuned to an action predictor using a smaller labeled dataset for action alignment. We show that our method not only learns from unlabeled human and robot demonstrations - improving downstream generalist robot policies - but also enables robots to learn new tasks without action labels (i.e., out-of-action generalization) in both real-world and simulated settings.

---

## 47. The Impact of 2D Segmentation Backbones on Point Cloud Predictions Using 4D Radar

**论文链接:** [http://arxiv.org/abs/2509.19644v1](http://arxiv.org/abs/2509.19644v1)

**作者:** William L. Muckelroy III, Mohammed Alsakabi, John M. Dolan, Ozan K. Tonguz

**发布时间:** 2025-09-23

### GPT解析

### 总结

该研究探讨了使用更高容量的分割骨干网络改进从4D雷达生成类LiDAR点云的质量，发现最优分割骨干可比当前最先进方法提高23.7%性能。

### 背景

LiDAR能提供密集、精确的点云表示，实现准确感知并提高道路安全，但其高成本限制了高级自动驾驶系统在商业车辆中的广泛应用。

### 目的

研究更高容量的分割骨干网络对生成点云质量的影响，探索如何在不使用LiDAR的情况下生成类似LiDAR的3D点云。

### 方法

使用神经网络，以LiDAR点云作为地面真实值(GT)，仅使用4D雷达生成类似LiDAR的3D点云，研究分割骨干网络的效果，使用RaDelft数据集进行训练。

### 主要发现

容量极高的模型实际上可能损害性能，而最优的分割骨干网络可以比当前最先进(SOTA)方法提高23.7%的性能。

### 结论

通过选择适当的分割骨干网络，可以显著提高从4D雷达生成类LiDAR点云的质量，有助于降低自动驾驶系统的成本。

### 翻译

LiDAR对周围环境的密集、精确点云表示能够实现准确感知，并通过提供更大的场景感知和理解能力显著提高道路安全。然而，LiDAR的高成本继续限制高级自动驾驶系统在商业可用车辆中的广泛采用。先前的研究已经取得了进展，通过训练神经网络（使用LiDAR点云作为地面真实值）来绕过对LiDAR的需求，仅使用4D雷达生成类似LiDAR的3D点云。最好的例子之一是一个创建的神经网络，它使用模块化的二维卷积神经网络(CNN)骨干网络和一个以时间一致性网络为核心，使用RaDelff数据集进行训练(见arXiv:2406.04723)。在这项工作中，我们研究了更高容量的分割骨干网络对生成点云质量的影响。我们的结果表明，虽然容量极高的模型实际上可能损害性能，但最优的分割骨干网络可以比当前最先进(SOTA)方法提供23.7%的改进。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决如何提高4D雷达生成的点云质量问题，使其能够替代昂贵的激光雷达。这个问题很重要，因为激光雷达虽然能提供高质量的环境点云，但价格昂贵（中端约4000美元，高端可达70000美元），限制了自动驾驶技术在商业车辆中的广泛应用。4D雷达成本低且性能有竞争力，但生成的点云更稀疏、噪声更大。如果能通过神经网络处理4D雷达数据生成高质量点云，就能大幅降低自动驾驶系统成本，促进其商业化应用。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者基于现有工作进行思考和设计。他们借鉴了之前使用深度学习改进4D雷达成像质量的研究，特别是[1]中的工作，该工作使用包含2D卷积神经网络分割骨干网络和时间一致性网络的神经网络。作者在此基础上，研究了更高容量的分割骨干网络（ResNet50、101、152）对点云质量的影响，并测试了不同数量的3D卷积层在时间一致性网络中的作用。作者使用了焦点损失函数进行训练，并调整了训练超参数以适应更大容量的模型，包括增加训练周期、使用正则化防止学习噪声、调整批量大小等。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是通过优化分割骨干网络和时间一致性网络的结构，提高4D雷达生成的点云质量，使其更接近激光雷达生成的点云。整体流程包括：1)使用RaDelft数据集，包含雷达数据和对应激光雷达点云；2)构建网络架构，包括多普勒编码器、分割骨干网络（不同容量ResNet模型）和时间一致性网络（不同数量3D卷积层）；3)使用焦点损失函数训练模型，调整超参数适应不同容量模型；4)使用检测概率、虚警概率和双向Chamfer距离评估性能。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)系统研究了不同容量分割骨干网络对点云质量的影响；2)发现了时间一致性网络中3D卷积层数量的最优配置（4层）；3)发现更大容量模型容易学习噪声导致性能下降；4)证明ResNet50结合4层时间一致性网络可提高23.7%的性能。相比之前工作[1]，本文不仅研究了时间维度改进，还深入探索了分割骨干网络的选择和容量对性能的影响，并找到了最优组合。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '通过实验发现，使用中等容量的ResNet50作为分割骨干网络并结合4层时间一致性网络，可以显著提高4D雷达生成的点云质量，比现有方法提升23.7%，为低成本自动驾驶系统提供了更可靠的感知方案。'}


### 论文摘要

LiDAR's dense, sharp point cloud (PC) representations of the surrounding environment enable accurate perception and significantly improve road safety by offering greater scene awareness and understanding. However, LiDAR's high cost continues to restrict the broad adoption of high-level Autonomous Driving (AD) systems in commercially available vehicles. Prior research has shown progress towards circumventing the need for LiDAR by training a neural network, using LiDAR point clouds as ground truth (GT), to produce LiDAR-like 3D point clouds using only 4D Radars. One of the best examples is a neural network created to train a more efficient radar target detector with a modular 2D convolutional neural network (CNN) backbone and a temporal coherence network at its core that uses the RaDelft dataset for training (see arXiv:2406.04723). In this work, we investigate the impact of higher-capacity segmentation backbones on the quality of the produced point clouds. Our results show that while very high-capacity models may actually hurt performance, an optimal segmentation backbone can provide a 23.7% improvement over the state-of-the-art (SOTA).

---

## 48. Human-Interpretable Uncertainty Explanations for Point Cloud Registration

**论文链接:** [http://arxiv.org/abs/2509.18786v2](http://arxiv.org/abs/2509.18786v2)

**作者:** Johannes A. Gaus, Loris Schneider, Yitian Shi, Jongseok Lee, Rania Rayyes, Rudolph Triebel

**发布时间:** 2025-09-23

### GPT解析

### 总结

本研究提出了一种名为高斯过程概念归因（GP-CA）的新方法，用于解决点云配准中的不确定性问题。该方法不仅能够量化配准不确定性，还能通过将不确定性归因于已知的误差来源来解释不确定性，并利用主动学习发现新的不确定性来源。实验表明，GP-CA在运行时间、样本效率和准确性方面优于其他最先进方法，并能实现有效的故障恢复行为，提高机器人感知的鲁棒性。

### 背景

在点云配准问题中，知名方法如ICP在传感器噪声、姿态估计误差和由遮挡引起的部分重叠等不确定性情况下表现不佳，需要一种能够处理这些不确定性的新方法。

### 目的

开发一种不仅能量化配准不确定性，还能通过将不确定性归因于配准问题中已知的误差来源来解释不确定性的方法，同时利用主动学习发现新的不确定性来源。

### 方法

提出高斯过程概念归因（GP-CA）方法，结合主动学习技术，通过查询信息量丰富的实例来发现实际环境中的新不确定性来源。

### 主要发现

GP-CA在三个公开数据集和真实机器人实验中得到了验证；与最先进方法相比，在运行时间、主动学习的高样本效率和准确性方面表现更好；真实实验展示了其适用性；能够实现有效的故障恢复行为，提高机器人感知的鲁棒性。

### 结论

GP-CA是一种有效的点云配准方法，能够在不确定性情况下表现良好，不仅能够量化不确定性，还能解释不确定性来源，并通过主动学习发现新的不确定性来源，从而提高机器人感知的鲁棒性。

### 翻译

在本文中，我们解决了点云配准问题，其中知名方法如ICP在传感器噪声、姿态估计误差和由遮挡引起的部分重叠等不确定性情况下表现不佳。我们开发了一种新方法——高斯过程概念归因（GP-CA），它不仅量化配准不确定性，还通过将不确定性归因于配准问题中已知的误差来源来解释不确定性。我们的方法利用主动学习通过查询信息量丰富的实例来发现实际环境中的新不确定性来源。我们在三个公开可用数据集和我们的真实机器人实验中验证了GP-CA。大量的消融实验证实了我们的设计选择。我们的方法在运行时间、主动学习的高样本效率和准确性方面优于其他最先进方法。我们的真实实验清楚地展示了其适用性。我们的视频还表明，GP-CA能够实现有效的故障恢复行为，从而提供更强大的机器人感知能力。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决点云配准中的不确定性解释问题。在机器人感知任务中，如SLAM、3D重建和物体姿态估计，ICP等常用配准方法在传感器噪声、姿态估计误差和部分重叠等情况下容易失败。虽然现有方法能量化不确定性，但很少能解释不确定性产生的原因，这使得机器人无法理解为什么配准失败以及如何采取适当的恢复行动，限制了实际应用效果。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先分析了点云配准中的不确定性问题，并指出现有不确定性量化方法只提供大小而不提供原因，而现有可解释AI方法计算量大且不适合实时应用。作者借鉴了三方面工作：1)使用DGCNN进行3D点云表示学习；2)应用高斯过程进行分类；3)采用BALD进行主动学习。基于这些，作者设计了GP-CA方法，通过将点云编码为潜在向量，再映射到不同不确定性概念的概率，并集成主动学习机制来发现新不确定性来源。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是将点云配准中的不确定性归因于人类可理解的概念(如传感器噪声、姿态误差、部分重叠等)，使机器人不仅知道不确定性大小，还知道原因并采取针对性恢复行动。整体流程：1)使用ICP对齐点云；2)用DGCNN将对齐点云编码为潜在向量；3)用高斯过程分类器映射到不同概念的概率；4)选择概率最高的概念作为主要不确定性来源；5)当不确定性高时，通过BALD标准选择样本进行标注并更新模型。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点：1)首次实现点云配准中人类可理解的不确定性解释；2)将不确定性归因于特定语义概念；3)集成主动学习机制适应新不确定性来源；4)实现高效的实时应用。相比之前工作：与传统不确定性量化方法不同，它不仅提供大小还提供原因；与SHAP/SA相比，计算效率更高且针对点云配准优化；与TCAV相比，不依赖神经网络且支持在线整合新概念；与纯监督方法相比，减少标注数据需求并能适应新环境。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '本文提出的GP-CA方法首次实现了点云配准中人类可理解的不确定性解释，通过将不确定性归因于语义概念并集成主动学习，使机器人能理解配准失败原因并采取针对性恢复行动，显著提高了机器人感知系统的鲁棒性。'}


### 论文摘要

In this paper, we address the point cloud registration problem, where well-known methods like ICP fail under uncertainty arising from sensor noise, pose-estimation errors, and partial overlap due to occlusion. We develop a novel approach, Gaussian Process Concept Attribution (GP-CA), which not only quantifies registration uncertainty but also explains it by attributing uncertainty to well-known sources of errors in registration problems. Our approach leverages active learning to discover new uncertainty sources in the wild by querying informative instances. We validate GP-CA on three publicly available datasets and in our real-world robot experiment. Extensive ablations substantiate our design choices. Our approach outperforms other state-of-the-art methods in terms of runtime, high sample-efficiency with active learning, and high accuracy. Our real-world experiment clearly demonstrates its applicability. Our video also demonstrates that GP-CA enables effective failure-recovery behaviors, yielding more robust robotic perception.

---

## 49. Process-Informed Forecasting of Complex Thermal Dynamics in Pharmaceutical Manufacturing

**论文链接:** [http://arxiv.org/abs/2509.20349v1](http://arxiv.org/abs/2509.20349v1)

**作者:** Ramona Rubini, Siavash Khodakarami, Aniruddha Bora, George Em Karniadakis, Michele Dassisti

**发布时间:** 2025-09-24

### GPT解析

### 总结

本文提出了一种用于制药冷冻干燥过程中温度预测的过程信息引导预测(PIF)模型，解决了深度学习模型在物理一致性和鲁棒性方面的局限性，提高了预测的准确性、物理合理性和噪声鲁棒性。

### 背景

复杂物理系统的时间序列预测是现代工业监控和控制的基础，尽管深度学习模型在捕捉复杂动态方面表现出色，但其部署受到物理一致性和鲁棒性问题的限制，影响了在受控环境中的可靠性。

### 目的

开发一种过程信息引导预测(PIF)模型，用于制药冷冻干燥过程中的温度预测，提高预测的准确性和可靠性。

### 方法

研究从经典模型(ARIMA、ETS)到现代深度学习架构(KANs)的多种模型；比较三种整合过程信息轨迹先验的损失函数形式(固定权重损失、动态不确定性损失、残差注意力机制)；评估模型的准确性、物理一致性和传感器噪声鲁棒性；在新过程的迁移学习场景中测试最佳模型的泛化能力。

### 主要发现

PIF模型在准确性、物理合理性和噪声鲁棒性方面均优于传统的数据驱动模型。

### 结论

这项工作为制药制造领域关键应用开发可靠且可泛化的预测解决方案提供了路线图。

### 翻译

复杂物理系统的准确时间序列预测是现代工业监控和控制的基础。虽然深度学习模型在捕捉复杂动态方面表现出色，但目前由于物理一致性和鲁棒性问题，其部署受到限制，从而限制了它们在受控环境中的可靠性。我们针对制药冷冻干燥过程中的温度引入了过程信息引导(PIF)预测模型。我们研究了从经典模型(如自回归积分移动平均模型ARIMA和指数平滑模型ETS)到现代深度学习架构(包括Kolmogorov-Arnold网络KANs)的广泛模型。我们比较了三种不同的整合过程信息轨迹先验的损失函数形式：固定权重损失、基于动态不确定性的损失和基于残差的注意力(RBA)机制。我们不仅从准确性和物理一致性角度，还从对传感器噪声的鲁棒性角度评估所有模型。此外，我们在新过程的迁移学习场景中测试了最佳模型的实际泛化能力。我们的结果表明，PIF模型在准确性、物理合理性和噪声鲁棒性方面优于其数据驱动的对应模型。这项工作为在制药制造领域的关键应用中开发可靠且可泛化的预测解决方案提供了路线图。


### 论文摘要

Accurate time-series forecasting for complex physical systems is the backbone of modern industrial monitoring and control. While deep learning models excel at capturing complex dynamics, currently, their deployment is limited due to physical inconsistency and robustness, hence constraining their reliability in regulated environments. We introduce process-informed forecasting (PIF) models for temperature in pharmaceutical lyophilization. We investigate a wide range of models, from classical ones such as Autoregressive Integrated Moving Average Model (ARIMA) and Exponential Smoothing Model (ETS), to modern deep learning architectures, including Kolmogorov-Arnold Networks (KANs). We compare three different loss function formulations that integrate a process-informed trajectory prior: a fixed-weight loss, a dynamic uncertainty-based loss, and a Residual-Based Attention (RBA) mechanism. We evaluate all models not only for accuracy and physical consistency but also for robustness to sensor noise. Furthermore, we test the practical generalizability of the best model in a transfer learning scenario on a new process. Our results show that PIF models outperform their data-driven counterparts in terms of accuracy, physical plausibility and noise resilience. This work provides a roadmap for developing reliable and generalizable forecasting solutions for critical applications in the pharmaceutical manufacturing landscape.

---

## 50. Transfer Learning in Regression with Influential Points

**论文链接:** [http://arxiv.org/abs/2509.20272v1](http://arxiv.org/abs/2509.20272v1)

**作者:** Bingbing Wang, Jiaqi Wang, Yu Tang

**发布时间:** 2025-09-24

### GPT解析

### 总结

该研究针对迁移学习中影响点导致的模型性能下降问题，提出了Trans-CO框架，通过影响点检测和回归模型拟合的协同优化，提高了模型在存在影响点情况下的预测性能和鲁棒性。

### 背景

回归预测在实际应用中至关重要，但依赖于数据标注。由于标注成本高或领域特定约束，目标域中的标记数据通常稀缺，因此迁移学习成为利用源域知识解决这一问题的重要方法。

### 目的

创新性地引入一种迁移学习协同优化(Trans-CO)框架，用于影响点检测和回归模型拟合，解决迁移学习中影响点导致的性能下降问题。

### 方法

提出Trans-CO算法，通过协同优化进行影响点检测和回归模型拟合，以应对迁移学习中的鲁棒性挑战。

### 主要发现

广泛的模拟实验表明，Trans-CO算法在模型拟合性能和影响点识别准确性方面优于竞争方法，在真实数据集上也实现了优越的预测准确性。

### 结论

Trans-CO为存在影响点的回归迁移学习提供了新的解决方案，提高了模型在目标域中的预测性能和鲁棒性。

### 翻译

回归预测在实际应用中起着至关重要的作用，并强烈依赖于数据标注。然而，由于高昂的标注成本或领域特定的约束，目标域中的标记数据通常稀缺，这使得迁移学习成为通过利用资源丰富的源域知识来解决这一问题的关键方案。在实际目标场景中，尽管迁移学习已被广泛应用，但影响点会显著扭曲目标域模型的参数估计。当源域中也存在影响点时，这一问题会进一步加剧，导致性能下降，并对现有迁移学习框架的鲁棒性提出关键挑战。在本研究中，我们创新性地引入了一种用于影响点检测和回归模型拟合的迁移学习协同优化(Trans-CO)框架。广泛的模拟实验表明，所提出的Trans-CO算法在模型拟合性能和影响点识别准确性方面优于竞争方法。此外，它在真实数据集上实现了优越的预测准确性，为存在影响点的回归迁移学习提供了新的解决方案。


### 论文摘要

Regression prediction plays a crucial role in practical applications and strongly relies on data annotation. However, due to prohibitive annotation costs or domain-specific constraints, labeled data in the target domain is often scarce, making transfer learning a critical solution by leveraging knowledge from resource-rich source domains. In the practical target scenario, although transfer learning has been widely applied, influential points can significantly distort parameter estimation for the target domain model. This issue is further compounded when influential points are also present in source domains, leading to aggravated performance degradation and posing critical robustness challenges for existing transfer learning frameworks. In this study, we innovatively introduce a transfer learning collaborative optimization (Trans-CO) framework for influential point detection and regression model fitting. Extensive simulation experiments demonstrate that the proposed Trans-CO algorithm outperforms competing methods in terms of model fitting performance and influential point identification accuracy. Furthermore, it achieves superior predictive accuracy on real-world datasets, providing a novel solution for transfer learning in regression with influential points

---

## 51. Low-Resource English-Tigrinya MT: Leveraging Multilingual Models, Custom Tokenizers, and Clean Evaluation Benchmarks

**论文链接:** [http://arxiv.org/abs/2509.20209v1](http://arxiv.org/abs/2509.20209v1)

**作者:** Hailay Kidu Teklehaymanot, Gebrearegawi Gidey, Wolfgang Nejdl

**发布时间:** 2025-09-24

**备注:** This submission is 8 pages long, includes 4 tables, and contains all  required conference details

### GPT解析

### 总结

本研究探讨了使用迁移学习技术提高低资源语言提格里尼亚语的神经机器翻译质量，通过整合语言特定分词、有信息嵌入初始化和领域自适应微调，并构建了高质量评估数据集。

### 背景

尽管神经机器翻译取得进展，低资源语言如提格里尼亚语仍面临有限语料库、不充分分词策略和缺乏标准化评估基准等挑战。

### 目的

研究使用多语言预训练模型的迁移学习技术，以提高形态丰富、低资源语言的翻译质量。

### 方法

提出改进方法整合语言特定分词、有信息嵌入初始化和领域自适应微调；构建高质量人工对齐的英语-提格里尼亚语多领域评估数据集。

### 主要发现

使用自定义分词器的迁移学习显著优于零样本基线，通过BLEU、chrF和人工评估验证；应用Bonferroni校正确保统计显著性；错误分析揭示关键局限性并指导改进。

### 结论

语言感知建模和可重现基准对缩小代表性语言性能差距至关重要。

### 翻译

尽管神经机器翻译取得了进展，但低资源语言如提格里尼亚语仍然面临持续挑战，包括有限的语料库、不充分的分词策略以及缺乏标准化的评估基准。本文研究使用多语言预训练模型的迁移学习技术，以提高形态丰富、低资源语言的翻译质量。我们提出了一种改进的方法，整合了语言特定的分词、有信息的嵌入初始化和领域自适应微调。为进行严格评估，我们构建了一个高质量、人工对齐的英语-提格里尼亚语评估数据集，涵盖多个领域。实验结果表明，使用自定义分词器的迁移学习显著优于零样本基线，这一优势通过BLEU、chrF和定性人工评估得到验证。应用了Bonferroni校正以确保不同配置间的统计显著性。错误分析揭示了关键局限性并指导了有针对性的改进。这项研究强调了语言感知建模和可重现基准在缩小代表性语言性能差距中的重要性。资源可在https://github.com/hailaykidu/MachineT_TigEng和https://huggingface.co/Hailay/MachineT_TigEng获取。


### 论文摘要

Despite advances in Neural Machine Translation (NMT), low-resource languages like Tigrinya remain underserved due to persistent challenges, including limited corpora, inadequate tokenization strategies, and the lack of standardized evaluation benchmarks. This paper investigates transfer learning techniques using multilingual pretrained models to enhance translation quality for morphologically rich, low-resource languages. We propose a refined approach that integrates language-specific tokenization, informed embedding initialization, and domain-adaptive fine-tuning. To enable rigorous assessment, we construct a high-quality, human-aligned English-Tigrinya evaluation dataset covering diverse domains. Experimental results demonstrate that transfer learning with a custom tokenizer substantially outperforms zero-shot baselines, with gains validated by BLEU, chrF, and qualitative human evaluation. Bonferroni correction is applied to ensure statistical significance across configurations. Error analysis reveals key limitations and informs targeted refinements. This study underscores the importance of linguistically aware modeling and reproducible benchmarks in bridging the performance gap for underrepresented languages. Resources are available at https://github.com/hailaykidu/MachineT_TigEng   and https://huggingface.co/Hailay/MachineT_TigEng

---

## 52. CorIL: Towards Enriching Indian Language to Indian Language Parallel Corpora and Machine Translation Systems

**论文链接:** [http://arxiv.org/abs/2509.19941v1](http://arxiv.org/abs/2509.19941v1)

**作者:** Soham Bhattacharjee, Mukund K Roy, Yathish Poojary, Bhargav Dave, Mihir Raj, Vandan Mujadia, Baban Gain, Pruthwik Mishra, Arafat Ahsan, Parameswari Krishnamurthy, Ashwath Rao, Gurpreet Singh Josan, Preeti Dubey, Aadil Amin Kak, Anna Rao Kulkarni, Narendra VG, Sunita Arora, Rakesh Balbantray, Prasenjit Majumdar, Karunesh K Arora, Asif Ekbal, Dipti Mishra Sharma

**发布时间:** 2025-09-24

### GPT解析

### 总结

本文介绍了一个大规模、高质量标注的平行语料库CorIL，涵盖11种印度语言，包含772,000个双语句子对，分类为政府、健康和一般三个领域，用于支持领域感知的机器翻译研究。

### 背景

印度拥有超过120种主要语言和约1600种其他语言，宪法中承认22种预定语言。尽管多语言神经机器翻译取得进展，但印度语言的高质量平行语料仍然稀缺，特别是在不同领域。

### 目的

创建一个大规模高质量平行语料库，支持领域感知的机器翻译研究，建立未来研究的基准，并提高印度语言高质量训练数据的可用性。

### 方法

构建包含772,000个双语句子对的数据集，涵盖英语、泰卢固语、印地语、旁遮普语、奥里亚语、克什米尔语、信德语、多格拉语、卡纳达语、乌尔都语和古吉拉特语，并系统分类为三个领域。微调和评估了IndicTrans2、NLLB和BhashaVerse等先进NMT模型。

### 主要发现

结果显示不同语言脚本有不同的性能模式，大规模多语言模型在波斯-阿拉伯语系上表现更佳，而其他模型在印度语系上更优秀。研究提供了详细的领域性能分析，揭示了领域敏感性和跨脚本迁移学习的特性。

### 结论

通过公开发布CorIL，显著提高了印度语言高质量训练数据的可用性，为机器翻译研究社区提供了宝贵资源。

### 翻译

印度的语言景观是世界上最具多样性的之一，包含120多种主要语言和约1600种其他语言，其中22种被印度宪法正式承认为预定语言。尽管多语言神经机器翻译最近取得了进展，但印度语言的高质量平行语料仍然稀缺，特别是在不同领域。在本文中，我们介绍了一个大规模、高质量标注的平行语料库，涵盖其中11种语言：英语、泰卢固语、印地语、旁遮普语、奥里亚语、克什米尔语、信德语、多格拉语、卡纳达语、乌尔都语和古吉拉特语，总共包含772,000个双语句子对。该数据集经过精心策划并系统分类为三个关键领域：政府、健康和一般，以支持领域感知的机器翻译研究并促进有效的领域适应。为了展示CorIL的效用并为未来研究建立强有力的基准，我们微调并评估了几种最先进的NMT模型，包括IndicTrans2、NLLB和BhashaVerse。我们的分析揭示了重要的性能趋势，并突显了该语料库在探测模型能力方面的价值。例如，结果显示基于语言脚本存在不同的性能模式，大规模多语言模型在波斯-阿拉伯语系（乌尔都语、信德语）上表现出优势，而其他模型在印度语系上表现出色。本文提供了详细的领域性能分析，对领域敏感性和跨脚本迁移学习提供了见解。通过公开发布CorIL，我们旨在显著提高印度语言高质量训练数据的可用性，并为机器翻译研究社区提供宝贵资源。


### 论文摘要

India's linguistic landscape is one of the most diverse in the world, comprising over 120 major languages and approximately 1,600 additional languages, with 22 officially recognized as scheduled languages in the Indian Constitution. Despite recent progress in multilingual neural machine translation (NMT), high-quality parallel corpora for Indian languages remain scarce, especially across varied domains. In this paper, we introduce a large-scale, high-quality annotated parallel corpus covering 11 of these languages : English, Telugu, Hindi, Punjabi, Odia, Kashmiri, Sindhi, Dogri, Kannada, Urdu, and Gujarati comprising a total of 772,000 bi-text sentence pairs. The dataset is carefully curated and systematically categorized into three key domains: Government, Health, and General, to enable domain-aware machine translation research and facilitate effective domain adaptation. To demonstrate the utility of CorIL and establish strong benchmarks for future research, we fine-tune and evaluate several state-of-the-art NMT models, including IndicTrans2, NLLB, and BhashaVerse. Our analysis reveals important performance trends and highlights the corpus's value in probing model capabilities. For instance, the results show distinct performance patterns based on language script, with massively multilingual models showing an advantage on Perso-Arabic scripts (Urdu, Sindhi) while other models excel on Indic scripts. This paper provides a detailed domain-wise performance analysis, offering insights into domain sensitivity and cross-script transfer learning. By publicly releasing CorIL, we aim to significantly improve the availability of high-quality training data for Indian languages and provide a valuable resource for the machine translation research community.

---

## 53. Beyond Language Barriers: Multi-Agent Coordination for Multi-Language Code Generation

**论文链接:** [http://arxiv.org/abs/2509.19918v1](http://arxiv.org/abs/2509.19918v1)

**作者:** Micheline Bénédicte Moumoula, Serge Lionel Nikiema, Albérick Euraste Djire, Abdoul Kader Kabore, Jacques Klein, Tegawendé F. Bissyande

**发布时间:** 2025-09-24

### GPT解析

### 总结

XL-CoGen是一种创新的跨语言代码生成系统，通过协调多智能体架构和基于数据驱动的桥接语言选择机制，显著提高了在多种编程语言上生成高质量代码的能力，特别是在训练数据有限的语言上表现优异。

### 背景

现代软件系统建立在异构技术栈上，需要跨多种编程语言生成高质量代码。大型语言模型在自动化编程方面取得了进展，但在不同语言上的能力差异很大，特别是在训练数据有限的语言(如Rust、Perl、OCaml和Erlang)上表现不佳。当前解决方案仍然孤立地处理每种目标语言，错失了共享知识或利用跨语言模式重复出现的机会。

### 目的

解决大型语言模型在不同编程语言间能力差异大的问题，打破现有解决方案孤立处理每种目标语言的局限，实现跨语言知识的共享和利用。

### 方法

提出了XL-CoGen系统，采用协调的多智能体架构，整合了中间表示、代码生成、翻译和自动修复功能。创新点是基于数据驱动的机制选择桥接语言，通过经验推导的转移矩阵识别最佳中间语言，基于已证实的翻译成功率而非原始生成准确性。系统执行早期输出验证，迭代纠正错误，并重用中间工件作为后续翻译的上下文支架。

### 主要发现

XL-CoGen取得了显著改进，比最强微调基线提高13个百分点，比现有单语言多智能体方法提高多达30个百分点。消融研究进一步证明，兼容性引导的桥接显著优于基于大型语言模型的启发式方法，确认了累积跨语言知识转移的价值。

### 结论

XL-CoGen通过协调的多智能体架构有效解决了跨语言代码生成的挑战。数据驱动的桥接语言选择机制比传统方法更有效，跨语言知识共享和重用对提高代码生成质量至关重要。

### 翻译

随着当今软件系统建立在异构技术栈上，跨多种编程语言生成高质量代码变得越来越重要。大型语言模型已经推动了自动化编程的发展，但它们在不同语言上的熟练程度差异很大，特别是对于那些训练数据有限的语言，如Rust、Perl、OCaml和Erlang。许多当前解决方案，包括语言特定的微调、多智能体协调、迁移学习和中间表示管道，仍然孤立地处理每种目标语言，错失了共享知识或利用重复出现的跨语言模式的机会。XL-CoGen通过协调的多智能体架构应对这一挑战，该架构整合了中间表示、代码生成、翻译和自动修复。其显著特点是用于选择桥接语言的数据驱动机制：经验推导的转移矩阵基于已证实的翻译成功率而非原始生成准确性来识别最佳中间语言。系统执行早期输出验证，迭代纠正错误，并重用中间工件作为后续翻译的上下文支架。大量实验表明，XL-CoGen取得了显著改进，比最强的微调基线提高了13个百分点，比现有的单语言多智能体方法提高了多达30个百分点。消融研究进一步证明，兼容性引导的桥接显著优于基于大型语言模型的启发式方法，确认了累积跨语言知识转移的价值。


### 论文摘要

Producing high-quality code across multiple programming languages is increasingly important as today's software systems are built on heterogeneous stacks. Large language models (LLMs) have advanced the state of automated programming, yet their proficiency varies sharply between languages, especially those with limited training data such as Rust, Perl, OCaml, and Erlang. Many current solutions including language-specific fine-tuning, multi-agent orchestration, transfer learning, and intermediate-representation pipelines still approach each target language in isolation, missing opportunities to share knowledge or exploit recurring cross-language patterns.   XL-CoGen tackles this challenge with a coordinated multi-agent architecture that integrates intermediate representation, code generation, translation, and automated repair. Its distinguishing feature is a data-driven mechanism for selecting bridging languages: empirically derived transfer matrices identify the best intermediate languages based on demonstrated translation success rather than raw generation accuracy. The system performs early output validation, iteratively corrects errors, and reuses intermediate artifacts as contextual scaffolds for subsequent translations.   Extensive experiments show that XL-CoGen yields notable improvements with 13 percentage-point gains over the strongest fine-tuned baseline and as much as 30 percentage points over existing single-language multi-agent methods. Ablation studies further demonstrate that compatibility-guided bridging significantly outperforms LLM-based heuristics, confirming the value of cumulative cross-language knowledge transfer.

---

## 54. SMILES-Inspired Transfer Learning for Quantum Operators in Generative Quantum Eigensolver

**论文链接:** [http://arxiv.org/abs/2509.19715v1](http://arxiv.org/abs/2509.19715v1)

**作者:** Zhi Yin, Xiaoran Li, Shengyu Zhang, Xin Li, Xiaojin Zhang

**发布时间:** 2025-09-24

**备注:** 7 pages, 5 figures

### GPT解析

### 总结

本文提出了一种基于文本表示的量子算子构建方法，用于在生成量子特征求解器(GQE)框架下实现不同分子系统间的知识转移，从而减少计算资源需求。

### 背景

传统变分量子特征求解器(VQE)算法存在固有局限性。在量子化学中，广泛使用的UCCSD方法需要为不同分子系统构建不同量子算子，增加了计算成本。

### 目的

利用不同分子系统间的相似性减少量子算子构建的计算成本，受SMILES表示方法启发，开发基于文本的UCCSD量子算子表示方法。

### 方法

开发文本表示方法利用分子系统间固有的表示相似性，探索量子算子中的文本模式相似性，使用文本相似度度量建立迁移学习框架。

### 主要发现

在朴素基线设置下，该方法在GQE范式中实现了不同分子系统间的知识转移，用于基态能量计算。

### 结论

这一发现对分子基态能量的混合量子-经典计算具有重要意义，可显著减少计算资源需求。

### 翻译

鉴于传统变分量子特征求解器(VQE)算法的固有局限性，将深度生成模型集成到混合量子-经典框架中，特别是生成量子特征求解器(GQE)，代表了一种有前景的创新方法。然而，以量子化学中广泛使用的UCCSD(具有单激发和双激发的酉耦合簇)为例，不同的分子系统需要构建不同的量子算子。考虑到不同分子的相似性，利用这种相似性构建量子算子可以显著降低计算成本。受计算化学中的SMILES表示方法启发，我们利用不同分子系统之间固有的表示相似性，开发了一种用于UCCSD量子算子的基于文本的表示方法。该框架探索量子算子中的文本模式相似性，并使用文本相似度度量来建立迁移学习框架。我们在朴素基线设置下的方法证明了在GQE范式中不同分子系统间的知识转移可用于基态能量计算。这一发现为分子基态能量的混合量子-经典计算带来了显著好处，大大降低了计算资源要求。


### 论文摘要

Given the inherent limitations of traditional Variational Quantum Eigensolver(VQE) algorithms, the integration of deep generative models into hybrid quantum-classical frameworks, specifically the Generative Quantum Eigensolver(GQE), represents a promising innovative approach. However, taking the Unitary Coupled Cluster with Singles and Doubles(UCCSD) ansatz which is widely used in quantum chemistry as an example, different molecular systems require constructions of distinct quantum operators. Considering the similarity of different molecules, the construction of quantum operators utilizing the similarity can reduce the computational cost significantly. Inspired by the SMILES representation method in computational chemistry, we developed a text-based representation approach for UCCSD quantum operators by leveraging the inherent representational similarities between different molecular systems. This framework explores text pattern similarities in quantum operators and employs text similarity metrics to establish a transfer learning framework. Our approach with a naive baseline setting demonstrates knowledge transfer between different molecular systems for ground-state energy calculations within the GQE paradigm. This discovery offers significant benefits for hybrid quantum-classical computation of molecular ground-state energies, substantially reducing computational resource requirements.

---

## 55. Parameter-Efficient Multi-Task Learning via Progressive Task-Specific Adaptation

**论文链接:** [http://arxiv.org/abs/2509.19602v1](http://arxiv.org/abs/2509.19602v1)

**作者:** Neeraj Gangwar, Anshuka Rangi, Rishabh Deshmukh, Holakou Rahmanian, Yesh Dattatreya, Nickvash Kani

**发布时间:** 2025-09-23

### GPT解析

### 总结

本文提出了一种渐进式任务特定多任务适应方法，用于解决参数高效微调在多任务学习中的任务干扰和负迁移问题。该方法通过在预训练模型中添加适配器模块，在初始层实现跨任务共享，在后续层实现任务特定学习，并使用基于梯度的任务相似性计算来优化任务分配。

### 背景

参数高效微调方法是将预训练模型适应到各种下游任务的有前景的解决方案，但在单任务学习中表现良好，扩展到多任务学习时会加剧任务干扰和负迁移等问题，这是由于可训练参数数量有限导致的。

### 目的

解决多任务学习中的任务干扰和负迁移问题，提出一种新的参数高效多任务学习方法，能够在保持性能的同时减少可训练参数数量。

### 方法

引入渐进式任务特定多任务适应方法，在预训练模型中添加适配器模块，这些模块在初始层跨所有任务共享，在后续层逐渐变得任务特定。此外，提出了一种基于梯度的任务相似性计算方法，用于将相似任务分配到共享的适配器模块，以最小化管道开销。

### 主要发现

在PASCAL和NYUD-v2数据集上的实验表明，该方法优于完全微调的多任务模型，而只需要五分之一的可训练参数。相比单任务微调有更好的相对改进，同时减少了可训练参数数量，超越了当前参数高效多任务学习的最先进方法。

### 结论

渐进式任务特定多任务适应方法是一种有效的参数高效多任务学习解决方案，能够在减少可训练参数的同时保持或提高模型性能。

### 翻译

参数高效微调方法已成为将预训练模型适应到各种下游任务的有前景的解决方案。虽然这些方法在单任务学习中表现良好，但将它们扩展到多任务学习会加剧常见的挑战，如任务干扰和负迁移，这是由于可训练参数数量有限。为解决这些问题，我们引入了渐进式任务特定多任务适应，这是一种用于多任务学习的新的参数高效方法。这种方法在预训练模型中引入适配器模块，使得这些模块在初始层跨所有任务共享，并在后续层逐渐变得更具任务特定性。其动机是通过允许在初始层跨所有任务进行迁移学习，并在预测头方向上实现任务特定学习，来减少任务之间的冲突。此外，我们提出了一种基于梯度的任务相似性计算方法，并利用这一度量将相似任务分配到共享的适配器模块。我们的任务相似性方法在管道中引入了最小的开销。我们通过将Swin Transformer适应于密集预测任务来评估我们的方法。在PASCAL和NYUD-v2数据集上的实验表明，我们的方法优于完全微调的多任务模型，而只需要五分之一的可训练参数。这种方法相比单任务微调有更好的相对改进，同时减少了可训练参数数量，并超越了当前参数高效多任务学习的最先进方法。


### 论文摘要

Parameter-efficient fine-tuning methods have emerged as a promising solution for adapting pre-trained models to various downstream tasks. While these methods perform well in single-task learning, extending them to multi-task learning exacerbates common challenges, such as task interference and negative transfer, due to the limited number of trainable parameters. To address these issues, we introduce progressive task-specific multi-task adaptation, a novel parameter-efficient approach for multi-task learning. This approach introduces adapter modules in a pre-trained model such that these modules are shared across all tasks in the initial layers and become progressively more task-specific in the later layers. The motivation is to reduce the conflicts among tasks by allowing transfer learning across all tasks in the initial layers and enabling task-specific learning toward the prediction heads. Additionally, we propose a gradient-based approach for computing task similarity and use this measure to allocate similar tasks to the shared adapter modules. Our task similarity method introduces minimal overhead in the pipeline. We evaluate our approach by adapting the Swin Transformer for dense prediction tasks. Experiments on the PASCAL and NYUD-v2 datasets demonstrate that our approach outperforms a fully fine-tuned multi-task model while requiring only one-fifth of the trainable parameters. This approach achieves better relative improvement to single-task fine-tuning while reducing the number of trainable parameters and surpasses the current state-of-the-art methods for parameter-efficient multi-task learning.

---

## 56. A Generative Conditional Distribution Equality Testing Framework and Its Minimax Analysis

**论文链接:** [http://arxiv.org/abs/2509.17729v2](http://arxiv.org/abs/2509.17729v2)

**作者:** Siming Zheng, Meifang Lan, Tong Wang, Yuanyuan Lin

**发布时间:** 2025-09-22

### GPT解析

### 总结

本文提出了一种测试双样本问题中条件分布相等性的通用框架，该框架基于神经网络生成方法和样本分割技术，将条件分布测试问题转化为无条件分布问题，并提出了两种特殊测试方法。理论上建立了最小最大下界，证明了测试方法的一致性和收敛速率，实验验证了方法的有效性。

### 背景

该问题与协变量转移下的迁移学习密切相关，需要测试两个条件分布是否相等。

### 目的

开发一种通用框架来测试双样本问题中条件分布的相等性，特别是在协变量转移场景下。

### 方法

构建基于神经网络生成方法和样本分割技术的框架，将条件分布测试转化为无条件分布问题，并提出了基于生成排列和生成分类精度的两种特殊测试方法。

### 主要发现

在特定光滑性条件下建立了条件分布相等性统计推断的最小最大下界；生成排列测试及其修改版本可以达到这个下界；生成分类精度测试具有一致性；学习条件生成器具有收敛速率。

### 结论

所提出的框架和方法在理论上和实验上都证明了其有效性，为条件分布相等性测试提供了新的解决方案。

### 翻译

在本文中，我们提出了一种用于测试双样本问题中条件分布相等性的通用框架。这个问题与协变量转移下的迁移学习最为相关。我们的框架基于神经网络生成方法和样本分割技术，通过将条件分布测试问题转化为无条件分布问题而构建。我们引入了两种特殊测试：基于生成排列的条件分布相等性测试和基于生成分类精度的条件分布相等性测试。理论上，我们在特定的光滑性条件下，建立了测试两个条件分布相等性时统计推断的最小最大下界。我们证明，基于生成排列的条件分布相等性测试及其修改版本可以精确地或达到某些迭代对数因子地达到这个下界。此外，我们证明了基于生成分类精度的条件分布相等性测试的一致性。我们还通过推导与最近开发的偏置Rademacher复杂性和神经网络近似性质相关的新结果，建立了学习条件生成器的收敛速率。经验上，我们进行了包括合成数据集和两个真实数据集在内的数值研究，证明了我们方法的有效性。


### 论文摘要

In this paper, we propose a general framework for testing the equality of the conditional distributions in a two-sample problem. This problem is most relevant to transfer learning under covariate shift. Our framework is built on neural network-based generative methods and sample splitting techniques by transforming the conditional distribution testing problem into an unconditional one. We introduce two special tests: the generative permutation-based conditional distribution equality test and the generative classification accuracy-based conditional distribution equality test. Theoretically, we establish a minimax lower bound for statistical inference in testing the equality of two conditional distributions under certain smoothness conditions. We demonstrate that the generative permutation-based conditional distribution equality test and its modified version can attain this lower bound precisely or up to some iterated logarithmic factor. Moreover, we prove the testing consistency of the generative classification accuracy-based conditional distribution equality test. We also establish the convergence rate for the learned conditional generator by deriving new results related to the recently-developed offset Rademacher complexity and approximation properties using neural networks. Empirically, we conduct numerical studies including synthetic datasets and two real-world datasets, demonstrating the effectiveness of our approach.

---

## 57. Minimal Semantic Sufficiency Meets Unsupervised Domain Generalization

**论文链接:** [http://arxiv.org/abs/2509.15791v2](http://arxiv.org/abs/2509.15791v2)

**作者:** Tan Pan, Kaiyu Guo, Dongli Xu, Zhaorui Tan, Chen Jiang, Deshu Chen, Xin Guo, Brian C. Lovell, Limei Han, Yuan Cheng, Mahsa Baktashmotlagh

**发布时间:** 2025-09-19

**备注:** Accepted by NeurIPS 2025

### GPT解析

### 总结

该论文提出了MS-UDG方法，通过学习最小充分语义表示来解决无监督域泛化问题，实现了在无类别和域标签情况下的高性能泛化。

### 背景

深度学习的泛化能力在监督设置中已被广泛研究，但在无监督场景中研究较少。无监督域泛化（UDG）任务面临在没有类别标签的情况下区分语义与变化的挑战，而实际应用中域标签通常不可用。

### 目的

将UDG形式化为学习最小充分语义表示的任务，该表示应保留增强视图间共享的所有语义信息（充分性），并最大化移除与语义无关的信息（最小性）。

### 方法

提出最小充分UDG（MS-UDG）模型，通过基于InfoNCE的目标实现充分性，并采用两个互补组件促进最小性：新颖的语义-变化解纠缠损失和基于重建的机制来捕获足够的变异。

### 主要发现

优化表示以实现充分性和最小性可以直接减少分布外风险；MS-UDG在流行的无监督域泛化基准上建立了新的最先进水平，一致优于现有的SSL和UDG方法，且在表示学习过程中不需要类别或域标签。

### 结论

所提出的方法有效地解决了UDG任务中的挑战，通过学习最小充分语义表示，模型能够在没有类别或域标签的情况下实现更好的泛化性能。

### 翻译

深度学习的泛化能力在监督设置中已被广泛研究，但在无监督场景中研究较少。最近，无监督域泛化（UDG）任务被提出以增强使用常见无监督学习技术（如自监督学习SSL）训练的模型的泛化能力。UDG面临在没有类别标签的情况下区分语义与变化的挑战。尽管一些最近的方法使用域标签来解决这个问题，但在实际情况下这些域标签通常不可用。在本文中，我们将UDG形式化为学习最小充分语义表示的任务：一种保留增强视图间共享的所有语义信息（充分性）并最大化移除与语义无关的信息（最小性）的表示。我们从信息论角度理论上支持这些目标，证明优化表示以实现充分性和最小性可以直接减少分布外风险。实际上，我们通过最小充分UDG（MS-UDG）实现这一优化，这是一种可学习模型，通过整合（a）基于InfoNCE的目标来实现充分性；（b）两个互补组件来促进最小性：新颖的语义-变化解纠缠损失和基于重建的机制来捕获足够的变异。实验上，MS-UDG在流行的无监督域泛化基准上建立了新的最先进水平，一致优于现有的SSL和UDG方法，在表示学习过程中不需要类别或域标签。


### 论文摘要

The generalization ability of deep learning has been extensively studied in supervised settings, yet it remains less explored in unsupervised scenarios. Recently, the Unsupervised Domain Generalization (UDG) task has been proposed to enhance the generalization of models trained with prevalent unsupervised learning techniques, such as Self-Supervised Learning (SSL). UDG confronts the challenge of distinguishing semantics from variations without category labels. Although some recent methods have employed domain labels to tackle this issue, such domain labels are often unavailable in real-world contexts. In this paper, we address these limitations by formalizing UDG as the task of learning a Minimal Sufficient Semantic Representation: a representation that (i) preserves all semantic information shared across augmented views (sufficiency), and (ii) maximally removes information irrelevant to semantics (minimality). We theoretically ground these objectives from the perspective of information theory, demonstrating that optimizing representations to achieve sufficiency and minimality directly reduces out-of-distribution risk. Practically, we implement this optimization through Minimal-Sufficient UDG (MS-UDG), a learnable model by integrating (a) an InfoNCE-based objective to achieve sufficiency; (b) two complementary components to promote minimality: a novel semantic-variation disentanglement loss and a reconstruction-based mechanism for capturing adequate variation. Empirically, MS-UDG sets a new state-of-the-art on popular unsupervised domain-generalization benchmarks, consistently outperforming existing SSL and UDG methods, without category or domain labels during representation learning.

---

## 58. Unsupervised Outlier Detection in Audit Analytics: A Case Study Using USA Spending Data

**论文链接:** [http://arxiv.org/abs/2509.19366v1](http://arxiv.org/abs/2509.19366v1)

**作者:** Buhe Li, Berkay Kaplan, Maksym Lazirko, Aleksandr Kogan

**发布时间:** 2025-09-19

### GPT解析

### 总结

研究探讨无监督异常检测方法在审计分析中的有效性，使用美国卫生与公共服务部(DHHS)的支出数据作为案例。研究比较了多种异常检测算法，并发现混合方法能提高复杂金融数据中异常识别的稳健性和准确性。

### 背景

政府大规模数据集中需要高效准确的异常检测，而传统审计方法可能无法满足这一需求。

### 目的

评估和比较多种无监督异常检测算法在联邦支出模式异常识别中的有效性，以提高审计质量和效率。

### 方法

使用数据准备、算法实施和性能评估（包括精确度、召回率和F1分数）的方法论。比较的算法包括基于直方图的异常分数(HBOS)、稳健主成分分析(PCA)、最小协方差行列式(MCD)和K近邻(KNN)。

### 主要发现

结合多种检测策略的混合方法能提高复杂金融数据中异常识别的稳健性和准确性。

### 结论

该研究通过提供各种异常检测模型比较有效性的见解，展示了无监督学习技术在提高审计质量和效率方面的潜力，对审计人员、政策制定者和研究人员在利用先进分析进行政府财务监督和风险管理方面具有启示意义。

### 翻译

本研究探讨了无监督异常检测方法在审计分析中的有效性，利用美国卫生与公共服务部(DHHS)的支出数据作为案例示例。我们采用并比较了多种异常检测算法，包括基于直方图的异常分数(HBOS)、稳健主成分分析(PCA)、最小协方差行列式(MCD)和K近邻(KNN)，以识别联邦支出模式中的异常。研究解决了在政府大规模数据集中高效准确异常检测的日益增长的需求，而传统审计方法可能无法满足这一需求。我们的方法论包括数据准备、算法实施以及使用精确度、召回率和F1分数进行性能评估。结果表明，结合多种检测策略的混合方法提高了复杂金融数据中异常识别的稳健性和准确性。该研究通过提供各种异常检测模型比较有效性的见解，展示了无监督学习技术在提高审计质量和效率方面的潜力，对寻求利用先进分析进行政府财务监督和风险管理的审计人员、政策制定者和研究人员具有启示意义。


### 论文摘要

This study investigates the effectiveness of unsupervised outlier detection methods in audit analytics, utilizing USA spending data from the U.S. Department of Health and Human Services (DHHS) as a case example. We employ and compare multiple outlier detection algorithms, including Histogram-based Outlier Score (HBOS), Robust Principal Component Analysis (PCA), Minimum Covariance Determinant (MCD), and K-Nearest Neighbors (KNN) to identify anomalies in federal spending patterns. The research addresses the growing need for efficient and accurate anomaly detection in large-scale governmental datasets, where traditional auditing methods may fall short. Our methodology involves data preparation, algorithm implementation, and performance evaluation using precision, recall, and F1 scores. Results indicate that a hybrid approach, combining multiple detection strategies, enhances the robustness and accuracy of outlier identification in complex financial data. This study contributes to the field of audit analytics by providing insights into the comparative effectiveness of various outlier detection models and demonstrating the potential of unsupervised learning techniques in improving audit quality and efficiency. The findings have implications for auditors, policymakers, and researchers seeking to leverage advanced analytics in governmental financial oversight and risk management.

---

## 59. Graph Variate Neural Networks

**论文链接:** [http://arxiv.org/abs/2509.20311v1](http://arxiv.org/abs/2509.20311v1)

**作者:** Om Roy, Yashar Moshfeghi, Keith Smith

**发布时间:** 2025-09-24

### GPT解析

### 总结

GVNNs是一种创新的神经网络架构，专门用于处理动态时空信号，通过结合稳定长期支持和瞬时数据驱动交互，有效捕获动态统计相关性，在多个任务中表现优异，特别是在脑机接口应用方面。

### 背景

动态时空信号建模是图神经网络(GNN)文献中的一个突出挑战。GNN假设存在一个潜在的图结构，但这种潜在结构可能并不总是存在，或者是从信号中独立推导出来的。从多通道数据中总是可以构建一个随时间演化的功能网络。

### 目的

提出一种新的方法来处理动态时空信号建模问题，特别是当潜在图结构不存在或独立于信号的情况。

### 方法

基于GVSA(图变分信号分析)和图信号处理工具，引入GVNNs(图变分神经网络)。设计了能够将时空信号与信号相关的连接张量进行卷积的层，该连接张量结合了稳定的长期支持和瞬时数据驱动的交互。这种设计在每个时间步捕获动态统计相关性，无需使用滑动窗口，并实现了序列长度上的线性复杂度的高效实现。

### 主要发现

在预测基准测试中，GVNNs始终优于强大的基于图的基线方法，与广泛使用的序列模型(如LSTMs和Transformers)具有竞争力。在EEG运动想象分类中，GVNNs实现了高精度。

### 结论

GVNNs在脑机接口应用方面显示出潜力，为动态时空信号建模提供了一个有效的新方法。

### 翻译

对动态演化的时空信号进行建模是图神经网络(GNN)文献中的一个突出挑战。值得注意的是，GNN假设存在一个潜在的图结构。虽然这种潜在结构可能并不总是存在，或者是从信号中独立推导出来的，但从多通道数据中总是可以构建一个随时间演化的功能网络。图变分信号分析(GVSA)定义了一个统一框架，包含一个瞬时连接剖面的网络张量，通常基于信号本身构建一个稳定的支撑。基于GVSA和图信号处理工具，我们引入了图变分神经网络(GVNNs)：这些层将时空信号与信号相关的连接张量进行卷积，该张量结合了稳定的长期支撑和瞬时的数据驱动交互。这种设计在每个时间步捕获动态统计相关性，无需使用滑动窗口，并且允许在序列长度上实现线性复杂度的高效实现。在预测基准测试中，GVNNs始终优于强大的基于图的基线方法，并且与广泛使用的序列模型(如LSTMs和Transformers)具有竞争力。在EEG运动想象分类中，GVNNs实现了高精度，突显了它们在脑机接口应用中的潜力。


### 论文摘要

Modelling dynamically evolving spatio-temporal signals is a prominent challenge in the Graph Neural Network (GNN) literature. Notably, GNNs assume an existing underlying graph structure. While this underlying structure may not always exist or is derived independently from the signal, a temporally evolving functional network can always be constructed from multi-channel data. Graph Variate Signal Analysis (GVSA) defines a unified framework consisting of a network tensor of instantaneous connectivity profiles against a stable support usually constructed from the signal itself. Building on GVSA and tools from graph signal processing, we introduce Graph-Variate Neural Networks (GVNNs): layers that convolve spatio-temporal signals with a signal-dependent connectivity tensor combining a stable long-term support with instantaneous, data-driven interactions. This design captures dynamic statistical interdependencies at each time step without ad hoc sliding windows and admits an efficient implementation with linear complexity in sequence length. Across forecasting benchmarks, GVNNs consistently outperform strong graph-based baselines and are competitive with widely used sequence models such as LSTMs and Transformers. On EEG motor-imagery classification, GVNNs achieve strong accuracy highlighting their potential for brain-computer interface applications.

---

## 60. Stochastically Evolving Graphs via Edit Semigroups

**论文链接:** [http://arxiv.org/abs/2509.19678v1](http://arxiv.org/abs/2509.19678v1)

**作者:** Fan Chung, Sawyer Jack Robertson

**发布时间:** 2025-09-24

**备注:** 23 pages, 5 figures

### GPT解析

### 总结

研究了一种基于Tsetlin库和超平面排列半群谱理论的子图随机演化过程，该过程通过随机编辑操作生成宿主图子图上的随机游走。

### 背景

演化图出现在深度学习、图神经网络、流行病学建模和社会网络等多个领域。

### 目的

开发一种从给定图中采样随机子图的一般随机模型。

### 方法

从初始子图开始，每次迭代应用随机选择的编辑操作（简单编辑如添加/删除边，或复合编辑同时影响多条边），生成子图集合上的随机游走，并使用半群谱理论进行分析。

### 主要发现

1) 随机游走的特征值可由宿主图的边子集自然索引；2) 简单编辑情况下提供了转移概率矩阵特征向量的闭式公式；3) 提供了随机游走收敛速率的精确界限。

### 结论

该随机演化过程可作为从给定图中采样随机子图的一般随机模型，适用于多种应用场景。

### 翻译

我们使用与Tsetlin库和超平面排列相关的半群谱理论，研究了基础宿主图中子图的随机演化过程。从初始子图开始，每次迭代对当前子图应用随机选择的编辑操作。这些编辑操作从简单的添加或删除一条边，到可以同时影响多条边的复合编辑不等。这种演化过程在宿主图的所有可能子图集合上生成一个随机游走。我们证明了这个随机游走的特征值可以自然地由宿主图的边子集索引。在简单编辑的情况下，我们还提供了转移概率矩阵特征向量的闭式公式以及这个随机游走收敛速率的精确界限。我们考虑了向复合编辑情况的扩展；该模型的例子包括先前研究的Moran森林模型和动态随机相交图模型。演化图出现在从深度学习和图神经网络到流行病学建模和社会网络的各个领域。我们的随机演化过程作为从给定图中采样随机子图的一般随机模型。


### 论文摘要

We investigate a randomly evolving process of subgraphs in an underlying host graph using the spectral theory of semigroups related to the Tsetlin library and hyperplane arrangements. Starting with some initial subgraph, at each iteration, we apply a randomly selected edit to the current subgraph. Such edits vary in nature from simple edits consisting of adding or deleting an edge, or compound edits which can affect several edges at once. This evolving process generates a random walk on the set of all possible subgraphs of the host graph. We show that the eigenvalues of this random walk can be naturally indexed by subsets of edges of the host graph. We also provide, in the case of simple edits, a closed-form formula for the eigenvectors of the transition probability matrix and a sharp bound for the rate of convergence of this random walk. We consider extensions to the case of compound edits; examples of this model include the previously studied Moran forest model and a dynamic random intersection graph model. Evolving graphs arise in a variety of fields ranging from deep learning and graph neural networks to epidemic modeling and social networks. Our random evolving process serves as a general stochastic model for sampling random subgraphs from a given graph.

---

## 61. EngravingGNN: A Hybrid Graph Neural Network for End-to-End Piano Score Engraving

**论文链接:** [http://arxiv.org/abs/2509.19412v1](http://arxiv.org/abs/2509.19412v1)

**作者:** Emmanouil Karystinaios, Francesco Foscarin, Gerhard Widmer

**发布时间:** 2025-09-23

**备注:** Accepted at the International Conference on Technologies for Music  Notation and Representation (TENOR) 2025

### GPT解析

### 总结

本文提出了一种统一的图神经网络框架用于自动音乐雕版，通过多任务学习同时处理多个相互关联的子任务，实现了从量化符号输入到可打印乐谱的转换。

### 背景

自动音乐雕版是从音乐内容创建人类可读乐谱的关键步骤，对所有涉及人类演奏者的应用都至关重要，但在符号音乐处理领域仍是一个大多未被探索的主题。

### 目的

解决自动音乐雕版问题，通过一个统一的框架处理多个相互关联的子任务，包括声部连接、谱表分配、音高拼写、调号、符干方向、八度移位和谱号等。

### 方法

采用多任务图神经网络联合预测多个音乐雕版相关子任务，并使用专门的后续处理流程生成可打印的MusicXML/MEI输出。

### 主要发现

在J-Pop和DCML Romantic两个多样化钢琴曲库上的评估表明，统一模型在所有子任务上都取得了良好的准确性，优于仅专精于特定子任务的现有系统。

### 结论

在多任务设置中使用共享的GNN编码器和轻量级的任务特定解码器，为自动音乐雕版提供了一种可扩展且有效的解决方案。

### 翻译

这篇论文关注自动音乐雕版，即从音乐内容创建人类可读的乐谱。这一步骤对所有包含人类演奏者的应用都是基础性的，但在符号音乐处理领域仍是一个大多未被探索的主题。在这项工作中，我们将问题形式化为一系列相互依赖的子任务集合，并提出了一种统一的图神经网络框架，专门针对钢琴音乐和量化符号输入的情况。我们的方法采用多任务GNN来联合预测声部连接、谱表分配、音高拼写、调号、符干方向、八度移位和谱号等。专门的后续处理流程生成可打印的MusicXML/MEI输出。在两个多样化的钢琴曲库上的全面评估表明，与仅专精于特定子任务的现有系统相比，我们的统一模型在所有子任务上都取得了良好的准确性。这些结果表明，在多任务设置中，使用共享的GNN编码器和轻量级的任务特定解码器，为自动音乐雕版提供了一种可扩展且有效的解决方案。


### 论文摘要

This paper focuses on automatic music engraving, i.e., the creation of a humanly-readable musical score from musical content. This step is fundamental for all applications that include a human player, but it remains a mostly unexplored topic in symbolic music processing. In this work, we formalize the problem as a collection of interdependent subtasks, and propose a unified graph neural network (GNN) framework that targets the case of piano music and quantized symbolic input. Our method employs a multi-task GNN to jointly predict voice connections, staff assignments, pitch spelling, key signature, stem direction, octave shifts, and clef signs. A dedicated postprocessing pipeline generates print-ready MusicXML/MEI outputs. Comprehensive evaluation on two diverse piano corpora (J-Pop and DCML Romantic) demonstrates that our unified model achieves good accuracy across all subtasks, compared to existing systems that only specialize in specific subtasks. These results indicate that a shared GNN encoder with lightweight task-specific decoders in a multi-task setting offers a scalable and effective solution for automatic music engraving.

---

## 62. PGCLODA: Prompt-Guided Graph Contrastive Learning for Oligopeptide-Infectious Disease Association Prediction

**论文链接:** [http://arxiv.org/abs/2509.20290v1](http://arxiv.org/abs/2509.20290v1)

**作者:** Dayu Tan, Jing Chen, Xiaoping Zhou, Yansen Su, Chunhou Zheng

**发布时间:** 2025-09-24

**备注:** 12page and 8 figures

### GPT解析

### 总结

本研究提出了一种名为PGCLODA的提示引导的基于图对比学习的框架，用于发现寡肽与感染性疾病之间的潜在关联。该方法构建三元图，结合双编码器架构，在基准测试中表现优异。

### 背景

传染病持续威胁公共健康，亟需有效计算方法筛选新型抗感染剂。寡肽因结构简单、生物利用度高且不易产生耐药性，成为抗菌研究的有前景候选物，但专门预测寡肽与感染性疾病关联的计算模型仍然稀缺。

### 目的

开发一种计算方法来预测寡肽与感染性疾病之间的关联，促进新型抗感染剂的发现。

### 方法

构建包含寡肽、微生物和疾病节点的三元图，整合结构和语义信息。采用提示引导的图增强策略保留关键区域，使用结合图卷积网络(GCN)和Transformer的双编码器架构捕获特征，最后通过多层感知器(MLP)分类器进行预测。

### 主要发现

PGCLODA在AUROC、AUPRC和准确率方面持续优于最先进模型。消融研究和超参数研究确认了各模块的贡献。案例研究验证了其泛化能力和发现新型生物相关关联的潜力。

### 结论

该研究为机制驱动的发现和基于寡肽的药物开发提供了有价值的见解，PGCLODA的源代码已公开可用。

### 翻译

传染病继续对公共健康构成严重威胁，凸显了有效计算方法筛选新型抗感染剂的迫切需求。寡肽因其结构简单、生物利用度高且不易产生耐药性，已成为抗菌研究中的有前景的候选物。尽管有这些潜力，但专门用于预测寡肽与感染性疾病之间关联的计算模型仍然稀缺。本研究引入了一种提示引导的基于图对比学习的框架(PGCLODA)来发现潜在关联。构建了一个包含寡肽、微生物和疾病节点的三元图，整合了结构和语义信息。为了在对比学习中保留关键区域，采用提示引导的图增强策略来生成有意义的配对视图。使用整合图卷积网络(GCN)和Transformer的双编码器架构来联合捕获局部和全局特征。融合的嵌入随后被输入多层感知器(MLP)分类器进行最终预测。在基准数据集上的实验结果表明，PGCLODA在AUROC、AUPRC和准确率方面持续优于最先进的模型。消融研究和超参数研究确认了每个模块的贡献。案例研究进一步验证了PGCLODA的泛化能力及其发现新型、生物相关关联的潜力。这些发现为机制驱动的发现和基于寡肽的药物开发提供了有价值的见解。PGCLODA的源代码可在https://github.com/jjnlcode/PGCLODA在线获取。


### 论文摘要

Infectious diseases continue to pose a serious threat to public health, underscoring the urgent need for effective computational approaches to screen novel anti-infective agents. Oligopeptides have emerged as promising candidates in antimicrobial research due to their structural simplicity, high bioavailability, and low susceptibility to resistance. Despite their potential, computational models specifically designed to predict associations between oligopeptides and infectious diseases remain scarce. This study introduces a prompt-guided graph-based contrastive learning framework (PGCLODA) to uncover potential associations. A tripartite graph is constructed with oligopeptides, microbes, and diseases as nodes, incorporating both structural and semantic information. To preserve critical regions during contrastive learning, a prompt-guided graph augmentation strategy is employed to generate meaningful paired views. A dual encoder architecture, integrating Graph Convolutional Network (GCN) and Transformer, is used to jointly capture local and global features. The fused embeddings are subsequently input into a multilayer perceptron (MLP) classifier for final prediction. Experimental results on a benchmark dataset indicate that PGCLODA consistently outperforms state-of-the-art models in AUROC, AUPRC, and accuracy. Ablation and hyperparameter studies confirm the contribution of each module. Case studies further validate the generalization ability of PGCLODA and its potential to uncover novel, biologically relevant associations. These findings offer valuable insights for mechanism-driven discovery and oligopeptide-based drug development. The source code of PGCLODA is available online at https://github.com/jjnlcode/PGCLODA.

---

## 63. C$^2$MIL: Synchronizing Semantic and Topological Causalities in Multiple Instance Learning for Robust and Interpretable Survival Analysis

**论文链接:** [http://arxiv.org/abs/2509.20152v1](http://arxiv.org/abs/2509.20152v1)

**作者:** Min Cen, Zhenfeng Zhuang, Yuzhe Zhang, Min Zeng, Baptiste Magnier, Lequan Yu, Hong Zhang, Liansheng Wang

**发布时间:** 2025-09-24

### GPT解析

### 总结

本文提出了一种名为C²MIL的双重因果图MIL模型，用于解决H&E染色全切片图像生存分析中的语义偏差和拓扑噪声问题，提高了模型的泛化能力和可解释性。

### 背景

基于图的多个实例学习（Graph-based MIL）在H&E染色全切片图像生存分析中被广泛使用，但染色和扫描的变化会引入语义偏差，与因果关系无关的拓扑子图会产生噪声，导致有偏差的切片级表示，影响模型的可解释性和泛化能力。

### 目的

解决现有Graph-based MIL方法中存在的语义偏差和拓扑噪声问题，提高生存分析模型的可解释性和泛化能力。

### 方法

引入双重结构因果模型作为理论基础，提出C²MIL模型，包含跨尺度自适应特征解缠模块用于语义因果干预，以及伯努利可微分因果子图采样方法用于拓扑因果发现，并通过结合解缠监督和对比学习的联合优化策略同时优化语义和拓扑因果性。

### 主要发现

C²MIL在泛化和可解释性方面持续优于现有方法，可作为各种MIL基线的因果增强，代码已公开在GitHub上。

### 结论

C²MIL模型有效解决了现有方法中的语义偏差和拓扑噪声问题，提高了生存分析的性能和可解释性，为医学图像分析提供了新的因果推理框架。

### 翻译

基于图的多个实例学习（MIL）在苏木精和伊红（H&E）染色的全切片图像（WSIs）生存分析中被广泛使用，因为它能够捕获拓扑信息。然而，染色和扫描的变化会引入语义偏差，而与因果关系无关的拓扑子图会产生噪声，导致有偏差的切片级表示。这些问题可能会阻碍分析的可解释性和泛化能力。为了解决这个问题，我们引入双重结构因果模型作为理论基础，并提出了一种新颖且可解释的双重因果图MIL模型C²MIL。C²MIL包含一个新的跨尺度自适应特征解缠模块用于语义因果干预，以及一个新的伯努利可微分因果子图采样方法用于拓扑因果发现。结合解缠监督和对比学习的联合优化策略能够同时优化语义和拓扑因果性。实验证明，C²MIL在泛化和可解释性方面持续优于现有方法，并可作为各种MIL基线的因果增强。代码可在https://github.com/mimic0127/C2MIL获取。


### 论文摘要

Graph-based Multiple Instance Learning (MIL) is widely used in survival analysis with Hematoxylin and Eosin (H\&E)-stained whole slide images (WSIs) due to its ability to capture topological information. However, variations in staining and scanning can introduce semantic bias, while topological subgraphs that are not relevant to the causal relationships can create noise, resulting in biased slide-level representations. These issues can hinder both the interpretability and generalization of the analysis. To tackle this, we introduce a dual structural causal model as the theoretical foundation and propose a novel and interpretable dual causal graph-based MIL model, C$^2$MIL. C$^2$MIL incorporates a novel cross-scale adaptive feature disentangling module for semantic causal intervention and a new Bernoulli differentiable causal subgraph sampling method for topological causal discovery. A joint optimization strategy combining disentangling supervision and contrastive learning enables simultaneous refinement of both semantic and topological causalities. Experiments demonstrate that C$^2$MIL consistently improves generalization and interpretability over existing methods and can serve as a causal enhancement for diverse MIL baselines. The code is available at https://github.com/mimic0127/C2MIL.

---

## 64. CoMelSinger: Discrete Token-Based Zero-Shot Singing Synthesis With Structured Melody Control and Guidance

**论文链接:** [http://arxiv.org/abs/2509.19883v1](http://arxiv.org/abs/2509.19883v1)

**作者:** Junchuan Zhao, Wei Zeng, Tianle Lyu, Ye Wang

**发布时间:** 2025-09-24

**备注:** 13 pages, 5 figures, 5 tables

### GPT解析

### 总结

CoMelSinger是一种零样本歌唱声音合成框架，解决了韵律泄漏问题，实现了结构和分离的旋律控制，在音高准确性、音色一致性和零样本可转移性方面表现优异。

### 背景

离散编解码语音合成技术已通过上下文学习实现零样本生成，但直接应用于歌唱声音合成(SVS)仍面临挑战，特别是在需要精确旋律控制的情况下。

### 目的

开发一种零样本SVS框架，能够在离散编码建模范式中实现结构和分离的旋律控制，同时解决韵律泄漏问题。

### 方法

CoMelSinger基于非自回归MaskGCT架构，用歌词和音高令牌替换传统文本输入；提出从粗到细的对比学习策略抑制韵律泄漏；集成轻量级仅编码器歌唱声音转录模块提供细粒度帧级监督。

### 主要发现

基于提示的生成常常引入韵律泄漏，音高信息会无意中与音色提示纠缠；CoMelSinger通过对比学习策略和SVT模块有效解决了这一问题。

### 结论

CoMelSinger在音高准确性、音色一致性和零样本可转移性方面相较于竞争性基线方法取得了显著改进。

### 翻译

歌唱声音合成(SVS)旨在从结构化的音乐输入（如歌词和音高序列）生成富有表现力的声乐表演。虽然最近在离散编码语音合成方面的进展已通过上下文学习实现了零样本生成，但由于需要精确的旋律控制，直接将这些技术扩展到SVS仍然非同寻常。特别是，基于提示的生成常常引入韵律泄漏，音高信息会无意中与音色提示纠缠在一起，损害可控性。我们提出了CoMelSinger，一种零样本SVS框架，在离散编码建模范式中实现结构和分离的旋律控制。基于非自回归MaskGCT架构，CoMelSinger用歌词和音高令牌替换传统文本输入，保持上下文泛化能力同时增强旋律条件化。为了抑制韵律泄漏，我们提出了从粗到细的对比学习策略，明确调节声学提示和旋律输入之间的音高冗余。此外，我们还整合了一个轻量级的仅编码器歌唱声音转录(SVT)模块，将声学令牌与音高和持续时间对齐，提供细粒度的帧级监督。实验结果表明，CoMelSinger在音高准确性、音色一致性和零样本可转移性方面相较于竞争性基线方法取得了显著改进。


### 论文摘要

Singing Voice Synthesis (SVS) aims to generate expressive vocal performances from structured musical inputs such as lyrics and pitch sequences. While recent progress in discrete codec-based speech synthesis has enabled zero-shot generation via in-context learning, directly extending these techniques to SVS remains non-trivial due to the requirement for precise melody control. In particular, prompt-based generation often introduces prosody leakage, where pitch information is inadvertently entangled within the timbre prompt, compromising controllability. We present CoMelSinger, a zero-shot SVS framework that enables structured and disentangled melody control within a discrete codec modeling paradigm. Built on the non-autoregressive MaskGCT architecture, CoMelSinger replaces conventional text inputs with lyric and pitch tokens, preserving in-context generalization while enhancing melody conditioning. To suppress prosody leakage, we propose a coarse-to-fine contrastive learning strategy that explicitly regularizes pitch redundancy between the acoustic prompt and melody input. Furthermore, we incorporate a lightweight encoder-only Singing Voice Transcription (SVT) module to align acoustic tokens with pitch and duration, offering fine-grained frame-level supervision. Experimental results demonstrate that CoMelSinger achieves notable improvements in pitch accuracy, timbre consistency, and zero-shot transferability over competitive baselines.

---

## 65. MoTiC: Momentum Tightness and Contrast for Few-Shot Class-Incremental Learning

**论文链接:** [http://arxiv.org/abs/2509.19664v1](http://arxiv.org/abs/2509.19664v1)

**作者:** Zeyu He, Shuai Huang, Yuwu Lu, Ming Zhao

**发布时间:** 2025-09-24

### GPT解析

### 总结

这篇论文提出了名为MoTiC（Momentum Tightness and Contrast）的框架，用于解决少样本增量学习中的双重挑战：从稀缺样本中学习新类别同时保留旧类别知识。

### 背景

Few-Shot Class-Incremental Learning (FSCIL)需要同时处理两个挑战：从稀缺样本中学习新类别并保留旧类别知识。现有方法使用冻结特征提取器和类别平均原型来缓解灾难性遗忘和过拟合问题。

### 目的

减少新类别原型的估计偏差，提高原型准确性，并构建具有丰富表示能力和增强类别间内聚性的特征空间。

### 方法

通过贝叶斯分析将新类别先验与旧类别统计信息对齐以减少方差；提出大规模对比学习强制执行跨类别特征紧密性；集成动量自监督和虚拟类别到MoTiC框架中，以丰富特征多样性并为新类别原型注入先验信息。

### 主要发现

新类别先验与旧类别统计信息对齐可减少方差并提高原型准确性；大规模对比学习可增强跨类别特征紧密性；动量自监督和虚拟类别的集成可丰富特征多样性并提升增量学习性能。

### 结论

在三个FSCIL基准测试上取得最先进性能，特别是在细粒度任务CUB-200上，验证了该方法减少估计偏差和提高增量学习鲁棒性的能力。

### 翻译

少样本增量学习必须应对双重挑战：从稀缺样本中学习新类别同时保留旧类别知识。现有方法使用冻结特征提取器和类别平均原型来缓解灾难性遗忘和过拟合。然而，由于数据极度稀缺，新类别原型存在显著估计偏差，而基础类别原型则受益于充足的数据。在这项工作中，我们从理论上证明通过贝叶斯分析将新类别先验与旧类别统计信息对齐可以减少方差并提高原型准确性。此外，我们提出大规模对比学习来强制执行跨类别特征紧密性。为进一步丰富特征多样性并为新类别原型注入先验信息，我们将动量自监督和虚拟类别集成到动量紧密性和对比框架（MoTiC）中，构建具有丰富表示能力和增强类别间内聚性的特征空间。在三个FSCIL基准测试上的实验产生了最先进的性能，特别是在细粒度任务CUB-200上，验证了我们方法减少估计偏差和提高增量学习鲁棒性的能力。


### 论文摘要

Few-Shot Class-Incremental Learning (FSCIL) must contend with the dual challenge of learning new classes from scarce samples while preserving old class knowledge. Existing methods use the frozen feature extractor and class-averaged prototypes to mitigate against catastrophic forgetting and overfitting. However, new-class prototypes suffer significant estimation bias due to extreme data scarcity, whereas base-class prototypes benefit from sufficient data. In this work, we theoretically demonstrate that aligning the new-class priors with old-class statistics via Bayesian analysis reduces variance and improves prototype accuracy. Furthermore, we propose large-scale contrastive learning to enforce cross-category feature tightness. To further enrich feature diversity and inject prior information for new-class prototypes, we integrate momentum self-supervision and virtual categories into the Momentum Tightness and Contrast framework (MoTiC), constructing a feature space with rich representations and enhanced interclass cohesion. Experiments on three FSCIL benchmarks produce state-of-the-art performances, particularly on the fine-grained task CUB-200, validating our method's ability to reduce estimation bias and improve incremental learning robustness.

---

## 66. Symbol-Temporal Consistency Self-supervised Learning for Robust Time Series Classification

**论文链接:** [http://arxiv.org/abs/2509.19654v1](http://arxiv.org/abs/2509.19654v1)

**作者:** Kevin Garcia, Cassandra Garza, Brooklyn Berry, Yifeng Gao

**发布时间:** 2025-09-24

**备注:** 4 pages, 2 figures, IEEE-EMBS BSN 2025

### GPT解析

### 总结

本文提出了一种基于符号袋表示的自监督学习框架，用于处理数字健康领域中时间序列数据的分布偏移问题，特别是在人类行为差异导致的数据变化情况下表现出优越性能。

### 背景

数字健康领域的时间序列数据日益重要，但这类数据通常具有高度噪声、概念漂移等特性，这给训练具有良好泛化能力的深度学习模型带来了挑战。自监督对比学习作为一种从原始数据中学习有意义模式和表示的先进方法，在处理这些问题时展现出潜力。

### 目的

解决数字健康时间序列数据中因不同人类行为引起的数据分布偏移问题，提出一种能够抵抗数据偏移的自监督学习框架。

### 方法

提出了一种基于符号袋表示的自监督学习框架。符号袋表示对数据变形、位置偏移和时间序列数据中存在的噪声具有不敏感性，这被认为可能对指导深度学习获取抵抗此类数据偏移的表示具有关键作用。

### 主要发现

在存在显著数据偏移的情况下，所提出的方法能够实现显著更好的性能。

### 结论

基于符号袋表示的自监督学习框架能够有效处理数字健康领域中时间序列数据的分布偏移问题，特别是在人类行为差异导致的数据变化情况下表现出优越性能。

### 翻译

数字健康领域中时间序列重要性的激增，需要先进的方法来提取有意义的模式和表示。自监督对比学习已成为一种直接从原始数据中学习的有前景的方法。然而，数字健康中的时间序列数据已知具有高度噪声，本质上涉及概念漂移，并且对训练可泛化的深度学习模型提出了挑战。在本文中，我们特别关注由不同人类行为引起的数据分布偏移，并提出了一种意识到符号袋表示的自监督学习框架。符号袋表示以其对时间序列数据中存在的数据变形、位置偏移和噪声的不敏感性而闻名，这使其可能在指导深度学习获取抵抗此类数据偏移的表示方面具有关键作用。我们证明，在存在显著数据偏移的情况下，所提出的方法可以实现显著更好的性能。


### 论文摘要

The surge in the significance of time series in digital health domains necessitates advanced methodologies for extracting meaningful patterns and representations. Self-supervised contrastive learning has emerged as a promising approach for learning directly from raw data. However, time series data in digital health is known to be highly noisy, inherently involves concept drifting, and poses a challenge for training a generalizable deep learning model. In this paper, we specifically focus on data distribution shift caused by different human behaviors and propose a self-supervised learning framework that is aware of the bag-of-symbol representation. The bag-of-symbol representation is known for its insensitivity to data warping, location shifts, and noise existed in time series data, making it potentially pivotal in guiding deep learning to acquire a representation resistant to such data shifting. We demonstrate that the proposed method can achieve significantly better performance where significant data shifting exists.

---

