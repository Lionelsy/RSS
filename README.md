# 今日论文推荐 - 2025-10-17

共 70 篇论文

---

## 1. Programmatic Representation Learning with Language Models

**论文链接:** [http://arxiv.org/abs/2510.14825v1](http://arxiv.org/abs/2510.14825v1)

**作者:** Gabriel Poesia, Georgia Gabriela Sampaio

**发布时间:** 2025-10-16

**备注:** Code available at https://github.com/gpoesia/leapr/

### GPT解析

### 总结

该论文提出了一种称为'学习程序化表示'(LeaPR)的模型，它结合了决策树和通过大型语言模型(LLMs)合成的特征函数，能够在不依赖神经网络的情况下实现高质量的预测，同时保持模型的可解释性。

### 背景

传统监督机器学习模型（如决策树）是高效且可解释的预测器，但其质量高度依赖于输入特征的选择。虽然神经网络可以直接从原始数据（如图像或文本）学习有用的表示，但这以牺牲可解释性和需要专门硬件高效运行为代价。

### 目的

探索一种新的模型类LeaPR，它将表示为代码（从数据点到标量的函数）的任意特征与决策树预测器堆叠，从而在保持可解释性的同时实现高质量的预测。

### 方法

1. 使用大型语言模型(LLMs)合成特征函数，利用它们在广泛领域的丰富先验知识和使用现有领域特定库编写代码的能力；2. 提出两种算法从监督数据中学习LeaPR模型：设计了FunSearch的适配版本来学习特征而非直接生成预测器；开发了经典ID3算法的新变体用于决策树学习，在分割叶节点时按需生成新特征。

### 主要发现

在从国际象棋位置评估到图像和文本分类的实验中，该方法学习了高质量的无神经网络预测器，通常可与神经网络相媲美。

### 结论

该研究提出了一种灵活的范式，用于端到端学习可解释的表示，其中特征和预测可以轻松检查和理解。

### 翻译

传统监督机器学习的经典模型，如决策树，是高效且可解释的预测器，但其质量高度依赖于特定输入特征的选择。虽然神经网络可以直接从原始数据（例如图像或文本）学习有用的表示，但这以牺牲可解释性和需要专门硬件高效运行为代价。在本文中，我们探索了一个称为学习程序化表示的假设类，它将表示为代码的任意特征（从数据点到标量的函数）与决策树预测器堆叠。我们使用大型语言模型合成特征函数，这些模型在广泛领域拥有丰富的先验知识，并且使用现有领域特定库编写代码的能力令人瞩目。我们提出了两种算法从监督数据中学习LeaPR模型。首先，我们设计了FunSearch的适配版本来学习特征而非直接生成预测器。然后，我们开发了经典ID3算法用于决策树学习的新变体，其中在分割叶节点时按需生成新特征。从国际象棋位置评估到图像和文本分类的实验中，我们的方法学习了高质量的无神经网络预测器，通常可与神经网络相媲美。我们的研究提出了一种灵活的范式，用于端到端学习可解释的表示，其中特征和预测可以轻松检查和理解。


### 论文摘要

Classical models for supervised machine learning, such as decision trees, are efficient and interpretable predictors, but their quality is highly dependent on the particular choice of input features. Although neural networks can learn useful representations directly from raw data (e.g., images or text), this comes at the expense of interpretability and the need for specialized hardware to run them efficiently. In this paper, we explore a hypothesis class we call Learned Programmatic Representations (LeaPR) models, which stack arbitrary features represented as code (functions from data points to scalars) and decision tree predictors. We synthesize feature functions using Large Language Models (LLMs), which have rich prior knowledge in a wide range of domains and a remarkable ability to write code using existing domain-specific libraries. We propose two algorithms to learn LeaPR models from supervised data. First, we design an adaptation of FunSearch to learn features rather than directly generate predictors. Then, we develop a novel variant of the classical ID3 algorithm for decision tree learning, where new features are generated on demand when splitting leaf nodes. In experiments from chess position evaluation to image and text classification, our methods learn high-quality, neural network-free predictors often competitive with neural networks. Our work suggests a flexible paradigm for learning interpretable representations end-to-end where features and predictions can be readily inspected and understood.

---

## 2. Unifying Environment Perception and Route Choice Modeling for Trajectory Representation Learning

**论文链接:** [http://arxiv.org/abs/2510.14819v1](http://arxiv.org/abs/2510.14819v1)

**作者:** Ji Cao, Yu Wang, Tongya Zheng, Zujie Ren, Canghong Jin, Gang Chen, Mingli Song

**发布时间:** 2025-10-16

### GPT解析

### 总结

PRTraj是一种新颖的轨迹表示学习框架，通过统一环境感知和路线选择建模来有效学习轨迹表示，解决了现有方法将轨迹视为孤立时空序列的局限。

### 背景

现有轨迹表示学习方法将轨迹视为孤立的时空序列，忽略了形成轨迹的外部环境和内部路线选择行为。

### 目的

开发一种能够综合考虑外部环境和内部路线选择行为的轨迹表示学习框架，以生成更准确、更有效的轨迹嵌入表示。

### 方法

PRTraj框架包含环境感知模块和路线选择编码器：环境感知模块通过捕获周围POI分布的多粒度环境语义增强道路网络；路线选择编码器将轨迹的组成路段转换建模为决策序列来捕获路线选择行为；最后将路线选择感知表示聚合形成全局轨迹嵌入。

### 主要发现

在3个真实世界数据集的5个下游任务上的广泛实验验证了PRTraj的有效性和泛化能力；PRTraj展现出强大的数据效率，在少样本场景下仍能保持稳健性能。

### 结论

PRTraj通过结合环境感知和路线选择建模，显著提升了轨迹表示学习的效果，为各种下游任务提供了更高质量的轨迹嵌入。

### 翻译

轨迹表示学习旨在将原始轨迹编码为低维向量，这些向量可在各种下游任务中利用，包括行程时间估计、位置预测和轨迹相似性分析。然而，现有的轨迹表示学习方法存在一个关键疏忽：将轨迹视为孤立的时空序列，而没有考虑支配其形成的外部环境和内部路线选择行为。为了弥合这一差距，我们提出了一种新颖的框架，统一了全面的环境感知和明确的路线选择建模，用于有效的轨迹表示学习，称为PRTraj。具体而言，PRTraj首先引入环境感知模块，通过捕获周围POI分布的多粒度环境语义来增强道路网络。基于这种环境感知骨干网络，路线选择编码器通过将轨迹的组成路段转换建模为决策序列来捕获每条轨迹固有的路线选择行为。这些路线选择感知表示最终被聚合形成全局轨迹嵌入。在3个真实世界数据集的5个下游任务上的广泛实验验证了PRTraj的有效性和泛化能力。此外，PRTraj展现出强大的数据效率，在少样本场景下保持稳健性能。我们的代码可在以下网址获取：https://anonymous.4open.science/r/PRTraj。


### 论文摘要

Trajectory Representation Learning (TRL) aims to encode raw trajectories into low-dimensional vectors, which can then be leveraged in various downstream tasks, including travel time estimation, location prediction, and trajectory similarity analysis. However, existing TRL methods suffer from a key oversight: treating trajectories as isolated spatio-temporal sequences, without considering the external environment and internal route choice behavior that govern their formation. To bridge this gap, we propose a novel framework that unifies comprehensive environment \textbf{P}erception and explicit \textbf{R}oute choice modeling for effective \textbf{Traj}ectory representation learning, dubbed \textbf{PRTraj}. Specifically, PRTraj first introduces an Environment Perception Module to enhance the road network by capturing multi-granularity environmental semantics from surrounding POI distributions. Building on this environment-aware backbone, a Route Choice Encoder then captures the route choice behavior inherent in each trajectory by modeling its constituent road segment transitions as a sequence of decisions. These route-choice-aware representations are finally aggregated to form the global trajectory embedding. Extensive experiments on 3 real-world datasets across 5 downstream tasks validate the effectiveness and generalizability of PRTraj. Moreover, PRTraj demonstrates strong data efficiency, maintaining robust performance under few-shot scenarios. Our code is available at: https://anonymous.4open.science/r/PRTraj.

---

## 3. Acquisition of interpretable domain information during brain MR image harmonization for content-based image retrieval

**论文链接:** [http://arxiv.org/abs/2510.14535v1](http://arxiv.org/abs/2510.14535v1)

**作者:** Keima Abe, Hayato Muraki, Shuhei Tomoshige, Kenichi Oishi, Hitoshi Iyatomi

**发布时间:** 2025-10-16

**备注:** 6 pages,3 figures, 3 tables. Accepted at 2025 IEEE International  Conference on Systems, Man, and Cybernetics (IEEE SMC 2025)

### GPT解析

### 总结

本文提出了一种名为PL-SE-ADA的域调和框架，通过双编码器结构和对抗训练实现医学图像的域调和与可解释表示学习，同时保留与疾病相关的信息。

### 背景

医学图像（如磁共振扫描）常因扫描仪和协议差异在不同成像站点间表现出域偏移，降低了机器学习在疾病分类等任务中的性能。现有方法虽能提取域不变和域特定特征，但缺乏医学应用所需的可解释性。

### 目的

开发一种通用的域调和框架，实现可解释的表示学习，同时保留脑磁共振图像中与疾病相关的信息。

### 方法

提出PL-SE-ADA框架，包含两个编码器分别提取域不变和域特定特征，一个解码器用于重建图像，以及一个域预测器。模型通过对抗训练学习，并通过将域不变和域特定特征的重建求和来重构输入图像。

### 主要发现

PL-SE-ADA在图像重建、疾病分类和域识别方面实现了与先前方法相当或更好的性能，同时能够可视化域独立的脑特征和域特定成分，提供了高可解释性。

### 结论

PL-SE-ADA是一种有效的域调和框架，不仅提高了医学图像处理任务的性能，还提供了必要的可解释性，解决了医学应用中的实际问题。

### 翻译

医学图像如磁共振扫描通常因扫描仪和协议差异在不同成像站点间表现出域偏移，这降低了机器学习在疾病分类等任务中的性能。域调和因此成为关键研究焦点。近期方法将脑图像编码到低维潜在空间并分离为域不变和域特定成分，但往往缺乏医学应用所需的可解释性。我们提出PL-SE-ADA框架，包含两个编码器提取域不变和域特定特征，一个解码器用于重建图像，以及一个域预测器。模型通过对抗训练学习，并通过将域不变和域特定特征的重建求和来重构输入图像，确保调和效果和信息保留。与先前方法相比，PL-SE-ADA在图像重建、疾病分类和域识别方面表现相当或更好，同时提供了高可解释性。


### 论文摘要

Medical images like MR scans often show domain shifts across imaging sites due to scanner and protocol differences, which degrade machine learning performance in tasks such as disease classification. Domain harmonization is thus a critical research focus. Recent approaches encode brain images $\boldsymbol{x}$ into a low-dimensional latent space $\boldsymbol{z}$, then disentangle it into $\boldsymbol{z_u}$ (domain-invariant) and $\boldsymbol{z_d}$ (domain-specific), achieving strong results. However, these methods often lack interpretability$-$an essential requirement in medical applications$-$leaving practical issues unresolved. We propose Pseudo-Linear-Style Encoder Adversarial Domain Adaptation (PL-SE-ADA), a general framework for domain harmonization and interpretable representation learning that preserves disease-relevant information in brain MR images. PL-SE-ADA includes two encoders $f_E$ and $f_{SE}$ to extract $\boldsymbol{z_u}$ and $\boldsymbol{z_d}$, a decoder to reconstruct the image $f_D$, and a domain predictor $g_D$. Beyond adversarial training between the encoder and domain predictor, the model learns to reconstruct the input image $\boldsymbol{x}$ by summing reconstructions from $\boldsymbol{z_u}$ and $\boldsymbol{z_d}$, ensuring both harmonization and informativeness. Compared to prior methods, PL-SE-ADA achieves equal or better performance in image reconstruction, disease classification, and domain recognition. It also enables visualization of both domain-independent brain features and domain-specific components, offering high interpretability across the entire framework.

---

## 4. 论文ID: 2510.14486v1

**错误:** 无法读取论文信息 - [Errno 2] No such file or directory: 'storage/result/2510.14486v1.json'

---

## 5. Revisit Modality Imbalance at the Decision Layer

**论文链接:** [http://arxiv.org/abs/2510.14411v1](http://arxiv.org/abs/2510.14411v1)

**作者:** Xiaoyu Ma, Hao Chen

**发布时间:** 2025-10-16

**备注:** Some Insights in Balanced Multimodal Learning

### GPT解析

### 总结

多模态学习面临模态不平衡问题，这种不平衡不仅存在于表示学习阶段，也在决策层显著存在。研究表明，即使在充分预训练后，模型仍表现出对某些模态的系统偏见，这种偏见源于特征空间和决策权重分布的内在差异，而非仅由优化动态导致。作者建议在决策层引入自适应权重分配机制以实现更平衡的模态融合。

### 背景

多模态学习整合不同模态信息以增强模型性能，但常面临模态不平衡问题，即主导模态在联合优化过程中掩盖较弱模态。

### 目的

揭示模态不平衡不仅在表示学习阶段存在，也在决策层显著表现，并提出解决方案。

### 方法

在音频-视觉数据集（CREMAD和Kinetic-Sounds）上进行实验，分析模型在预训练和平衡优化后对模态的偏见，研究特征空间和决策权重分布的差异。

### 主要发现

1) 模态不平衡不仅存在于表示学习阶段，也在决策层显著存在；2) 即使在充分预训练和平衡优化后，模型仍表现出对某些模态（如音频）的系统偏见；3) 这种偏见源于特征空间和决策权重分布的内在差异，而非仅由优化动态导致；4) 在融合阶段聚合未校准的模态输出会导致决策层的加权偏差。

### 结论

未来的多模态系统应该在决策层更多地纳入自适应权重分配机制，使各模态能够根据其能力实现相对平衡，从而有效利用较弱模态的贡献。

### 翻译

多模态学习整合来自不同模态的信息以增强模型性能，但它常常遭受模态不平衡的影响，在联合优化过程中主导模态会掩盖较弱的模态。本文揭示这种不平衡不仅发生在表示学习阶段，而且在决策层也显著表现。在音频-视觉数据集（CREMAD和Kinetic-Sounds）上的实验表明，即使在广泛的预训练和平衡优化后，模型仍然表现出对某些模态（如音频）的系统偏见。进一步分析表明，这种偏见源于特征空间和决策权重分布的内在差异，而不仅仅是优化动态。我们认为，在融合阶段聚合未校准的模态输出会导致决策层的加权偏差，阻碍较弱模态的有效贡献。为此，我们建议未来的多模态系统应该更注重在决策层纳入自适应权重分配机制，使各模态能够根据其能力实现相对平衡。


### 论文摘要

Multimodal learning integrates information from different modalities to enhance model performance, yet it often suffers from modality imbalance, where dominant modalities overshadow weaker ones during joint optimization. This paper reveals that such an imbalance not only occurs during representation learning but also manifests significantly at the decision layer. Experiments on audio-visual datasets (CREMAD and Kinetic-Sounds) show that even after extensive pretraining and balanced optimization, models still exhibit systematic bias toward certain modalities, such as audio. Further analysis demonstrates that this bias originates from intrinsic disparities in feature-space and decision-weight distributions rather than from optimization dynamics alone. We argue that aggregating uncalibrated modality outputs at the fusion stage leads to biased decision-layer weighting, hindering weaker modalities from contributing effectively. To address this, we propose that future multimodal systems should focus more on incorporate adaptive weight allocation mechanisms at the decision layer, enabling relative balanced according to the capabilities of each modality.

---

## 6. DCMIL: A Progressive Representation Learning Model of Whole Slide Images for Cancer Prognosis Analysis

**论文链接:** [http://arxiv.org/abs/2510.14403v1](http://arxiv.org/abs/2510.14403v1)

**作者:** Chao Tu, Kun Huang, Jie Zhang, Qianjin Feng, Yu Zhang, Zhenyuan Ning

**发布时间:** 2025-10-16

### GPT解析

### 总结

该研究提出了DCMIL模型，用于处理全切片图像(WSI)进行癌症预后预测，解决了计算瓶颈和注释稀缺的问题，并在多种癌症类型中表现出色。

### 背景

计算病理学新兴学科利用WSI量化形态异性和开发癌症预后模型，但受千兆像素级输入的计算瓶颈和密集手动注释稀缺的阻碍，且当前方法忽视了多倍率WSI中的细粒度信息和肿瘤微环境变异。

### 目的

开发一个易于到难的正向表示学习模型(DCMIL)，高效处理WSI用于癌症预后预测，不依赖密集注释，并能直接将千兆像素级WSI转化为结果预测。

### 方法

提出名为双课程对比多实例学习(DCMIL)的模型，是一种正向表示学习模型，能高效处理WSI，不需要密集注释，可直接将大型WSI图像转化为预后预测。

### 主要发现

在12种癌症类型(5,954名患者，1,254万张图像块)的实验中，DCMIL优于标准WSI预后模型；能识别细粒度预后显著区域；提供稳健实例不确定性估计；捕获正常与肿瘤组织形态差异；有潜力产生新生物学见解。

### 结论

DCMIL模型在癌症预后预测方面表现出色，不需要密集注释，能直接处理大型WSI图像，代码已在GitHub公开。

### 翻译

蓬勃发展的计算病理学学科显示出利用全切片图像(WSIs)量化形态异质性并为人类癌症开发客观预后模型的希望。然而，千兆像素级输入的计算瓶颈和密集手动注释的稀缺阻碍了进展。当前方法常常忽视了多倍率WSI中的细粒度信息和肿瘤微环境的变异。在这里，我们提出一个易于到难的正向表示学习模型，称为双课程对比多实例学习(DCMIL)，以高效处理WSI用于癌症预后。该模型不依赖于密集注释，并能将千兆像素级WSI直接转化为结果预测。在十二种癌症类型(5,954名患者，1,254万张图像块)的大量实验中证明，DCMIL优于标准的基于WSI的预后模型。此外，DCMIL能识别细粒度的预后显著区域，提供稳健的实例不确定性估计，并捕获正常组织和肿瘤组织之间的形态差异，有潜力产生新的生物学见解。所有代码已在https://github.com/tuuuc/DCMIL上公开。


### 论文摘要

The burgeoning discipline of computational pathology shows promise in harnessing whole slide images (WSIs) to quantify morphological heterogeneity and develop objective prognostic modes for human cancers. However, progress is impeded by the computational bottleneck of gigapixel-size inputs and the scarcity of dense manual annotations. Current methods often overlook fine-grained information across multi-magnification WSIs and variations in tumor microenvironments. Here, we propose an easy-to-hard progressive representation learning model, termed dual-curriculum contrastive multi-instance learning (DCMIL), to efficiently process WSIs for cancer prognosis. The model does not rely on dense annotations and enables the direct transformation of gigapixel-size WSIs into outcome predictions. Extensive experiments on twelve cancer types (5,954 patients, 12.54 million tiles) demonstrate that DCMIL outperforms standard WSI-based prognostic models. Additionally, DCMIL identifies fine-grained prognosis-salient regions, provides robust instance uncertainty estimation, and captures morphological differences between normal and tumor tissues, with the potential to generate new biological insights. All codes have been made publicly accessible at https://github.com/tuuuc/DCMIL.

---

## 7. BinCtx: Multi-Modal Representation Learning for Robust Android App Behavior Detection

**论文链接:** [http://arxiv.org/abs/2510.14344v1](http://arxiv.org/abs/2510.14344v1)

**作者:** Zichen Liu, Shao Yang, Xusheng Xiao

**发布时间:** 2025-10-16

### GPT解析

### 总结

论文提出了BINCTX，一种多模态学习方法，用于检测移动应用中的不良行为，通过结合代码级语义、行为触发方式和第三方库使用信息，实现了高准确率的检测。

### 背景

移动应用市场有数百万个应用，但不良行为（如干扰性广告、非法重定向、支付欺诈）难以被发现，因为这些行为通常不依赖权限保护的API，且可通过UI或元数据编辑轻易伪装。

### 目的

开发一种能够有效检测移动应用中不良行为的机器学习方法，提高检测的准确性和鲁棒性。

### 方法

BINCTX构建应用的三种视图：全局字节码图像视图（捕获代码级语义和家族模式）、上下文视图（显示行为触发方式）和第三方库使用视图（总结组件间调用路径上的调用频率），然后将这三种视图嵌入并融合，训练上下文感知分类器。

### 主要发现

在真实世界恶意软件和良性应用上，BINCTX达到94.73%的宏观F1值，比强大基线方法至少高出14.92%；在商业混淆下保持84%的F1值；比最先进的仅字节码系统更能抵抗对抗样本。

### 结论

BINCTX通过多模态表示学习，有效结合了代码级语义、行为上下文和第三方库使用信息，显著提高了移动应用不良行为的检测性能，并增强了对混淆技术和对抗攻击的抵抗力。

### 翻译

移动应用市场托管着数百万个应用，但不良行为（例如干扰性广告、非法重定向、支付欺诈）仍然难以被发现，因为它们通常不依赖于权限保护的API，并且可以通过UI或元数据编辑轻松伪装。我们提出了BINCTX，一种学习方法，它从(i)全局字节码图像视图捕获代码级语义和家族模式，(ii)上下文视图（显示的操作、组件、声明的权限、URL/IP常量）指示行为如何被触发，以及(iii)第三方库使用视图总结组件间调用路径上的调用频率，构建应用的多模态表示。这三个视图被嵌入并融合，训练一个上下文感知分类器。在真实世界的恶意软件和良性应用上，BINCTX实现了94.73%的宏观F1值，比强大的基线方法至少高出14.92%。它在商业混淆下保持鲁棒性（混淆后F1为84%），并且比最先进的仅字节码系统更能抵抗对抗样本。


### 论文摘要

Mobile app markets host millions of apps, yet undesired behaviors (e.g., disruptive ads, illegal redirection, payment deception) remain hard to catch because they often do not rely on permission-protected APIs and can be easily camouflaged via UI or metadata edits. We present BINCTX, a learning approach that builds multi-modal representations of an app from (i) a global bytecode-as-image view that captures code-level semantics and family-style patterns, (ii) a contextual view (manifested actions, components, declared permissions, URL/IP constants) indicating how behaviors are triggered, and (iii) a third-party-library usage view summarizing invocation frequencies along inter-component call paths. The three views are embedded and fused to train a contextual-aware classifier. On real-world malware and benign apps, BINCTX attains a macro F1 of 94.73%, outperforming strong baselines by at least 14.92%. It remains robust under commercial obfuscation (F1 84% post-obfuscation) and is more resistant to adversarial samples than state-of-the-art bytecode-only systems.

---

## 8. Large Reasoning Embedding Models: Towards Next-Generation Dense Retrieval Paradigm

**论文链接:** [http://arxiv.org/abs/2510.14321v1](http://arxiv.org/abs/2510.14321v1)

**作者:** Jianting Tang, Dongshuai Li, Tao Wen, Fuyu Lv, Dan Ou, Linli Xu

**发布时间:** 2025-10-16

### GPT解析

### 总结

本文提出了一种名为大型推理嵌入模型(LREM)的新方法，通过将推理过程整合到表示学习中，解决了电子商务搜索系统中困难查询的语义匹配问题，显著提高了检索准确性。

### 背景

在现代电子商务搜索系统中，密集检索是重要组成部分。主流嵌入模型已从BERT转向大型语言模型(LLMs)，但仍采用直接嵌入方法，语义准确性不足。对比学习虽被使用，但模型倾向于捕获统计共现模式，偏向浅层词汇和语义匹配，导致对困难查询的性能下降。

### 目的

提出大型推理嵌入模型(LREM)，创新地将推理过程整合到表示学习中，以解决困难查询与目标物品之间的语义匹配问题，提高检索准确性。

### 方法

LREM对困难查询先进行推理以深入理解查询，然后生成推理增强的查询嵌入用于检索。采用两阶段训练：第一阶段在Query-CoT-Item三元组上使用SFT和InfoNCE损失优化LLM；第二阶段通过强化学习进一步优化推理轨迹。

### 主要发现

推理过程有效桥接了原始查询和目标物品间的语义差距，显著提高了检索准确性。大量离线和在线实验验证了LREM的有效性。

### 结论

LREM已被成功部署在中国最大的电子商务平台上，自2025年8月起，证明了其在实际应用中的价值。

### 翻译

在现代电子商务搜索系统中，密集检索已成为不可或缺的组成部分。通过计算查询和物品(产品)嵌入之间的相似性，它能够从大规模存储库中高效地选择候选产品。随着大型语言模型(LLMs)的突破，主流嵌入模型已逐渐从BERT转向LLMs以实现更准确的文本建模。然而，这些模型仍采用直接嵌入方法，嵌入的语义准确性仍然不足。因此，对比学习被大量使用来实现正对之间的紧密语义对齐。结果，这些模型倾向于捕获训练数据中的统计共现模式，偏向于浅层词汇和语义匹配。对于与目标物品存在明显词汇差异的困难查询，性能显著下降。在这项工作中，我们提出了大型推理嵌入模型(LREM)，创新地将推理过程整合到表示学习中。对于困难查询，LREM首先进行推理以实现对原始查询的深入理解，然后生成推理增强的查询嵌入用于检索。这一推理过程有效地桥接了原始查询和目标物品之间的语义差距，显著提高了检索准确性。具体而言，我们采用两阶段训练过程：第一阶段在精心策划的查询-思维链-物品(Query-CoT-Item)三元组上使用SFT和InfoNCE损失优化LLM，建立初步推理和嵌入能力；第二阶段通过强化学习(RL)进一步优化推理轨迹。大量的离线和在线实验验证了LREM的有效性，使其自2025年8月起被部署在中国最大的电子商务平台上。


### 论文摘要

In modern e-commerce search systems, dense retrieval has become an indispensable component. By computing similarities between query and item (product) embeddings, it efficiently selects candidate products from large-scale repositories. With the breakthroughs in large language models (LLMs), mainstream embedding models have gradually shifted from BERT to LLMs for more accurate text modeling. However, these models still adopt direct-embedding methods, and the semantic accuracy of embeddings remains inadequate. Therefore, contrastive learning is heavily employed to achieve tight semantic alignment between positive pairs. Consequently, such models tend to capture statistical co-occurrence patterns in the training data, biasing them toward shallow lexical and semantic matches. For difficult queries exhibiting notable lexical disparity from target items, the performance degrades significantly. In this work, we propose the Large Reasoning Embedding Model (LREM), which novelly integrates reasoning processes into representation learning. For difficult queries, LREM first conducts reasoning to achieve a deep understanding of the original query, and then produces a reasoning-augmented query embedding for retrieval. This reasoning process effectively bridges the semantic gap between original queries and target items, significantly improving retrieval accuracy. Specifically, we adopt a two-stage training process: the first stage optimizes the LLM on carefully curated Query-CoT-Item triplets with SFT and InfoNCE losses to establish preliminary reasoning and embedding capabilities, and the second stage further refines the reasoning trajectories via reinforcement learning (RL). Extensive offline and online experiments validate the effectiveness of LREM, leading to its deployment on China's largest e-commerce platform since August 2025.

---

## 9. Inferred global dense residue transition graphs from primary structure sequences enable protein interaction prediction via directed graph convolutional neural networks

**论文链接:** [http://arxiv.org/abs/2510.14139v1](http://arxiv.org/abs/2510.14139v1)

**作者:** Islam Akef Ebeid, Haoteng Tang, Pengfei Gu

**发布时间:** 2025-10-15

**DOI:** 10.3389/fbinf.2025.1651623

**备注:** under review in Frontiers in Bioinformatics

### GPT解析

### 背景

蛋白质-蛋白质相互作用的准确预测对于理解细胞功能和推进药物开发至关重要。现有的计算方法使用蛋白质语言模型(PLMs)的直接序列嵌入，或使用图神经网络(GNNs)处理3D蛋白质结构。本研究探索计算密集度较低的替代方法。

### 目的

开发一种新的框架用于下游PPI预测，通过链接预测实现。

### 方法

引入一个两阶段的图表示学习框架ProtGram-DirectGCN。第一阶段开发ProtGram，将蛋白质的一级结构建模为全局推断的n-gram图层次结构，其中残基转移概率定义边权重。第二阶段提出DirectGCN，一种定制的有向图卷积神经网络，通过入向、出向和无向路径的转换处理信息，并通过可学习的门控机制结合这些路径。

### 主要发现

DirectGCN在标准节点分类基准上表现良好，性能与已建立的方法相当，尤其在具有密集、异质结构的有向复杂图中表现出色。完整的ProtGram-DirectGCN框架应用于PPI预测时提供了强大的预测能力，即使在有限的训练数据下也能保持。

### 结论

ProtGram-DirectGCN框架是一种有效的PPI预测方法，在计算资源有限的情况下也能保持良好的性能。

### 翻译

引言：准确预测蛋白质相互作用对于理解细胞功能和推进药物开发至关重要。现有的计算方法使用蛋白质语言模型的直接序列嵌入，或使用图神经网络处理3D蛋白质结构。本研究探索计算密集度较低的替代方法。我们引入了一种通过链接预测进行下游PPI预测的新框架。方法：我们引入了一个两阶段的图表示学习框架ProtGram-DirectGCN。首先，我们开发了ProtGram，该方法将蛋白质的一级结构建模为全局推断的n-gram图层次结构。在这些图中，残基转移概率定义边权重，每条边在有向图中连接一对残基，这些概率从大量序列集合中聚合。其次，我们提出了DirectGCN，一种定制的有向图卷积神经网络，该模型具有独特的卷积层，通过入向、出向和无向的特定路径转换处理信息，同时应用共享转换，这些路径通过可学习的门控机制结合。我们将DirectGCN应用于ProtGram图以学习残基级嵌入，并通过注意力池化生成蛋白质级嵌入进行预测。结果：我们首先在标准节点分类基准上建立了DirectGCN的有效性，其在一般数据集上的性能与已建立的方法相当，该模型在具有密集、异质结构的有向复杂图中表现出色。当应用于PPI预测时，完整的ProtGram-DirectGCN框架提供了强大的预测能力，即使在有限的训练数据下，这种强大的性能仍然保持。


### 论文摘要

Introduction Accurate prediction of protein-protein interactions (PPIs) is crucial for understanding cellular functions and advancing drug development. Existing in-silico methods use direct sequence embeddings from Protein Language Models (PLMs). Others use Graph Neural Networks (GNNs) for 3D protein structures. This study explores less computationally intensive alternatives. We introduce a novel framework for downstream PPI prediction through link prediction. Methods We introduce a two-stage graph representation learning framework, ProtGram-DirectGCN. First, we developed ProtGram. This approach models a protein's primary structure as a hierarchy of globally inferred n-gram graphs. In these graphs, residue transition probabilities define edge weights. Each edge connects a pair of residues in a directed graph. The probabilities are aggregated from a large corpus of sequences. Second, we propose DirectGCN, a custom directed graph convolutional neural network. This model features a unique convolutional layer. It processes information through separate path-specific transformations: incoming, outgoing, and undirected. A shared transformation is also applied. These paths are combined via a learnable gating mechanism. We apply DirectGCN to ProtGram graphs to learn residue-level embeddings. These embeddings are pooled via attention to generate protein-level embeddings for prediction. Results We first established the efficacy of DirectGCN on standard node classification benchmarks. Its performance matches established methods on general datasets. The model excels at complex, directed graphs with dense, heterophilic structures. When applied to PPI prediction, the full ProtGram-DirectGCN framework delivers robust predictive power. This strong performance holds even with limited training data.

---

## 10. STEMS: Spatial-Temporal Enhanced Safe Multi-Agent Coordination for Building Energy Management

**论文链接:** [http://arxiv.org/abs/2510.14112v1](http://arxiv.org/abs/2510.14112v1)

**作者:** Huiliang Zhang, Di Wu, Arnaud Zinflou, Benoit Boulet

**发布时间:** 2025-10-15

### GPT解析

### 总结

本文提出了一种名为STEMS的新型安全约束多智能体强化学习框架，用于协调建筑能源管理，有效解决了多建筑系统中时空依赖关系利用和操作安全性的挑战。

### 背景

建筑能源管理对于实现碳减排目标、提高居住者舒适度和降低能源成本至关重要。当前多建筑能源系统面临三个关键挑战：时空信息利用不足、缺乏严格的安全保证以及系统复杂性。

### 目的

提出一种新的安全约束多智能体强化学习框架，解决多建筑协调能源管理中的挑战，特别是在利用时空依赖关系和确保操作安全方面。

### 方法

STEMS框架整合了两个核心组件：(1)时空图表示学习框架，使用GCN-Transformer融合架构捕捉建筑间关系和时间模式；(2)安全约束多智能体RL算法，结合控制屏障函数提供数学安全保证。

### 主要发现

实验表明STEMS相比现有方法具有优越性能，实现了21%的成本降低，18%的排放减少，将安全违规从35.1%大幅降低到5.6%，并保持最优舒适度，仅有0.13%的不舒适比例。该框架在极端天气条件下表现出强大的鲁棒性，并在不同类型建筑中保持有效性。

### 结论

STEMS框架成功解决了多建筑能源管理中的关键挑战，通过整合时空信息利用和安全约束，实现了显著的能源成本降低和排放减少，同时确保了系统安全和居住者舒适度。

### 翻译

建筑能源管理对于实现碳减排目标、提高居住者舒适度和降低能源成本至关重要。协调建筑能源管理在利用时空依赖关系的同时确保多建筑系统运行安全方面面临关键挑战。当前多建筑能源系统面临三个关键挑战：时空信息利用不足、缺乏严格的安全保证以及系统复杂性。本文提出STEMS，一种用于协调建筑能源管理的新型安全约束多智能体强化学习框架。STEMS整合了两个核心组件：(1)使用GCN-Transformer融合架构的时空图表示学习框架，用于捕捉建筑间关系和时间模式；(2)结合控制屏障函数的安全约束多智能体RL算法，提供数学安全保证。在真实建筑数据集上的大量实验表明STEMS相比现有方法具有优越性能，实现了21%的成本降低，18%的排放减少，同时将安全违规从35.1%大幅降低到5.6%，并保持最优舒适度，仅有0.13%的不舒适比例。该框架在极端天气条件下也表现出强大的鲁棒性，并且在不同类型建筑中保持有效性。


### 论文摘要

Building energy management is essential for achieving carbon reduction goals, improving occupant comfort, and reducing energy costs. Coordinated building energy management faces critical challenges in exploiting spatial-temporal dependencies while ensuring operational safety across multi-building systems. Current multi-building energy systems face three key challenges: insufficient spatial-temporal information exploitation, lack of rigorous safety guarantees, and system complexity. This paper proposes Spatial-Temporal Enhanced Safe Multi-Agent Coordination (STEMS), a novel safety-constrained multi-agent reinforcement learning framework for coordinated building energy management. STEMS integrates two core components: (1) a spatial-temporal graph representation learning framework using a GCN-Transformer fusion architecture to capture inter-building relationships and temporal patterns, and (2) a safety-constrained multi-agent RL algorithm incorporating Control Barrier Functions to provide mathematical safety guarantees. Extensive experiments on real-world building datasets demonstrate STEMS's superior performance over existing methods, showing that STEMS achieves 21% cost reduction, 18% emission reduction, and dramatically reduces safety violations from 35.1% to 5.6% while maintaining optimal comfort with only 0.13 discomfort proportion. The framework also demonstrates strong robustness during extreme weather conditions and maintains effectiveness across different building types.

---

## 11. CausalVerse: Benchmarking Causal Representation Learning with Configurable High-Fidelity Simulations

**论文链接:** [http://arxiv.org/abs/2510.14049v1](http://arxiv.org/abs/2510.14049v1)

**作者:** Guangyi Chen, Yunlong Deng, Peiyuan Zhu, Yan Li, Yifan Sheng, Zijian Li, Kun Zhang

**发布时间:** 2025-10-15

### GPT解析

### 总结

该研究引入了一个新的因果表征学习(CRL)基准，使用高保真模拟视觉数据，既保留真实视觉复杂性又能访问真实因果生成过程，包含约20万张图像和300万视频帧，涵盖四个领域的24个子场景。

### 背景

因果表征学习旨在揭示数据生成过程并识别潜在因果变量和关系，但评估具有挑战性，因为需要已知的真实因果变量和结构。现有评估方法要么依赖简化合成数据集，要么依赖现实世界任务中的下游性能，在真实性和评估精度间面临两难困境。

### 目的

创建一个既保留真实视觉复杂性又能访问真实因果生成过程的新CRL基准，解决现有评估方法在真实性和评估精度之间的两难困境。

### 方法

构建包含约20万张图像和300万视频帧的数据集，涵盖静态图像生成、动态物理模拟、机器人操作和交通情况分析四个领域的24个子场景，提供对底层因果结构的灵活访问，允许用户修改或配置以符合CRL假设要求。

### 主要发现

利用此基准评估了不同范式的代表性CRL方法，提供了实证见解，帮助实践者和新手选择或扩展适当的CRL框架，以解决可以从CRL视角受益的现实问题。

### 结论

该基准有望弥合严格评估和实际应用之间的差距，为CRL研究提供更全面、更真实的测试平台。

### 翻译

摘要内容已为中文，无需额外翻译。


### 论文摘要

Causal Representation Learning (CRL) aims to uncover the data-generating process and identify the underlying causal variables and relations, whose evaluation remains inherently challenging due to the requirement of known ground-truth causal variables and causal structure. Existing evaluations often rely on either simplistic synthetic datasets or downstream performance on real-world tasks, generally suffering a dilemma between realism and evaluative precision. In this paper, we introduce a new benchmark for CRL using high-fidelity simulated visual data that retains both realistic visual complexity and, more importantly, access to ground-truth causal generating processes. The dataset comprises around 200 thousand images and 3 million video frames across 24 sub-scenes in four domains: static image generation, dynamic physical simulations, robotic manipulations, and traffic situation analysis. These scenarios range from static to dynamic settings, simple to complex structures, and single to multi-agent interactions, offering a comprehensive testbed that hopefully bridges the gap between rigorous evaluation and real-world applicability. In addition, we provide flexible access to the underlying causal structures, allowing users to modify or configure them to align with the required assumptions in CRL, such as available domain labels, temporal dependencies, or intervention histories. Leveraging this benchmark, we evaluated representative CRL methods across diverse paradigms and offered empirical insights to assist practitioners and newcomers in choosing or extending appropriate CRL frameworks to properly address specific types of real problems that can benefit from the CRL perspective. Welcome to visit our: Project page:https://causal-verse.github.io/, Dataset:https://huggingface.co/CausalVerse.

---

## 12. Stealthy Dual-Trigger Backdoors: Attacking Prompt Tuning in LM-Empowered Graph Foundation Models

**论文链接:** [http://arxiv.org/abs/2510.14470v1](http://arxiv.org/abs/2510.14470v1)

**作者:** Xiaoyu Xue, Yuni Lai, Chenxi Huang, Yulin Zhu, Gaolei Li, Xiaoge Zhang, Kai Zhou

**发布时间:** 2025-10-16

### GPT解析

### 总结

该研究探讨了结合语言模型的图基础模型在文本属性图上的安全漏洞，特别是后门攻击问题，并提出了一种双触发攻击框架。

### 背景

图基础模型，特别是结合语言模型的模型，已革新图学习并在文本属性图上表现出色，但相比传统GNN引入了新的安全漏洞。

### 目的

解决LM赋能的GFMs在无安全提示调整阶段的安全漏洞问题，特别是在属性不可访问的约束TAG系统中的后门攻击挑战。

### 方法

提出一种新的双触发后门攻击框架，在文本层面和结构层面同时运作，通过利用预先建立的文本池实现无需显式优化触发节点文本属性的有效攻击。

### 主要发现

传统图后门攻击在属性不可访问的约束TAG系统中性能显著下降；所提双触发攻击框架能保持优越的干净准确率并取得出色的攻击成功率。

### 结论

LM赋能的GFMs在网络部署中存在关键后门风险，研究为基础模型时代开源平台开发更强大的监督机制提供了贡献。

### 翻译

图基础模型的出现，特别是那些结合语言模型的模型，已经革新了图学习并在文本属性图上表现出色。然而，与传统GNN相比，这些由语言模型赋能的图基础模型在无安全提示调整阶段引入了独特的安全漏洞，这些漏洞在当前研究中尚未得到充分研究。通过实证研究，我们发现在属性不可访问的约束文本属性图系统中，当没有显式优化触发节点属性时，传统图后门攻击的性能会显著下降。为此，我们提出了一种新的双触发后门攻击框架，在文本层面和结构层面同时运作，通过战略性地利用预先建立的文本池，无需显式优化触发节点文本属性即可实现有效攻击。大量实验评估表明，我们的攻击方法在保持优越的干净准确率的同时，取得了出色的攻击成功率，包括在高度隐蔽的单触发节点场景中。我们的工作强调了在网络上部署的由语言模型赋能的图基础模型中的关键后门风险，并为基础模型时代开源平台开发更强大的监督机制做出了贡献。


### 论文摘要

The emergence of graph foundation models (GFMs), particularly those incorporating language models (LMs), has revolutionized graph learning and demonstrated remarkable performance on text-attributed graphs (TAGs). However, compared to traditional GNNs, these LM-empowered GFMs introduce unique security vulnerabilities during the unsecured prompt tuning phase that remain understudied in current research. Through empirical investigation, we reveal a significant performance degradation in traditional graph backdoor attacks when operating in attribute-inaccessible constrained TAG systems without explicit trigger node attribute optimization. To address this, we propose a novel dual-trigger backdoor attack framework that operates at both text-level and struct-level, enabling effective attacks without explicit optimization of trigger node text attributes through the strategic utilization of a pre-established text pool. Extensive experimental evaluations demonstrate that our attack maintains superior clean accuracy while achieving outstanding attack success rates, including scenarios with highly concealed single-trigger nodes. Our work highlights critical backdoor risks in web-deployed LM-empowered GFMs and contributes to the development of more robust supervision mechanisms for open-source platforms in the era of foundation models.

---

## 13. DARTS-GT: Differentiable Architecture Search for Graph Transformers with Quantifiable Instance-Specific Interpretability Analysis

**论文链接:** [http://arxiv.org/abs/2510.14336v1](http://arxiv.org/abs/2510.14336v1)

**作者:** Shruti Sarika Chakraborty, Peter Minary

**发布时间:** 2025-10-16

### GPT解析

### 总结

本文提出了一种改进的图变换器架构DARTS-GT，通过不对称注意力和可微分架构搜索实现深度异质性，并开发了首个图变换器的定量可解释性框架。实验表明该方法在多个数据集上达到最先进水平，且发现的异构架构比基线更可解释，证明图变换器无需在性能和可解释性间做出取舍。

### 背景

图变换器(GTs)是处理图结构数据的有力架构，但受限于刚性设计和缺乏可量化可解释性。当前最先进的GT在所有层中固定使用相同的GNN类型，错过了深度特定组件选择的优势，且复杂架构变得不透明，无法区分性能提升中的有意义模式和虚假相关性。

### 目的

重新设计GT注意力机制通过不对称性解耦结构编码与特征表示；使用DARTS在每层选择最优GNN算子；开发首个GT的定量可解释性框架；探索GT是否需要在性能和可解释性之间做出选择。

### 方法

重新设计GT注意力：查询来自节点特征，键和值来自GNN变换；使用DARTS在transformer注意力内部实现深度异质性(DARTS-GT)；通过因果消融开发GT的定量可解释性框架；提出Head-deviation、Specialization和Focus指标；在8个基准数据集上进行实验。

### 主要发现

DARTS-GT在4个数据集上达到最先进水平，在其他数据集上保持竞争力；发现的架构揭示了数据集特定模式；可视化注意力和因果重要性并不总是相关，表明常用可视化方法可能忽略真正重要的组件；DARTS-GT发现的异构架构比基线产生更可解释的模型。

### 结论

Graph Transformers不需要在性能和可解释性之间做出选择。异构架构可以同时提高性能和可解释性，证明性能和可解释性并非相互排斥的目标。

### 翻译

图变换器(GTs)已成为处理图结构数据的有力架构，但仍受限于刚性设计且缺乏可量化可解释性。当前最先进的GT在所有层中固定使用相同的GNN类型，错过了深度特定组件选择的优势，同时其复杂架构变得不透明，无法区分性能提升中的有意义模式和虚假相关性。我们通过不对称性重新设计GT注意力，解耦结构编码与特征表示：查询来自节点特征，而键和值来自GNN变换。在此框架内，我们使用可微分架构搜索(DARTS)在每层选择最优GNN算子，在transformer注意力内部实现深度异质性(DARTS-GT)。为了理解发现的架构，我们通过因果消融开发了首个GT的定量可解释性框架。我们的指标(Head-deviation、Specialization和Focus)识别出哪些头和节点驱动预测，同时实现模型比较。在八个基准数据集上的实验显示，DARTS-GT在四个数据集上达到最先进水平，在其他数据集上保持竞争力，且发现的架构揭示了数据集特定模式。我们的可解释性分析表明，可视化注意力和因果重要性并不总是相关，表明广泛使用的可视化方法可能忽略实际重要的组件。重要的是，DARTS-GT发现的异构架构始终比基线产生更可解释的模型，证明图变换器无需在性能和可解释性之间做出选择。


### 论文摘要

Graph Transformers (GTs) have emerged as powerful architectures for graph-structured data, yet remain constrained by rigid designs and lack quantifiable interpretability. Current state-of-the-art GTs commit to fixed GNN types across all layers, missing potential benefits of depth-specific component selection, while their complex architectures become opaque where performance gains cannot be distinguished between meaningful patterns and spurious correlations. We redesign GT attention through asymmetry, decoupling structural encoding from feature representation: queries derive from node features while keys and values come from GNN transformations. Within this framework, we use Differentiable ARchiTecture Search (DARTS) to select optimal GNN operators at each layer, enabling depth-wise heterogeneity inside transformer attention itself (DARTS-GT). To understand discovered architectures, we develop the first quantitative interpretability framework for GTs through causal ablation. Our metrics (Head-deviation, Specialization, and Focus), identify which heads and nodes drive predictions while enabling model comparison. Experiments across eight benchmarks show DARTS-GT achieves state-of-the-art on four datasets while remaining competitive on others, with discovered architectures revealing dataset-specific patterns. Our interpretability analysis reveals that visual attention salience and causal importance do not always correlate, indicating widely used visualization approaches may miss components that actually matter. Crucially, heterogeneous architectures found by DARTS-GT consistently produced more interpretable models than baselines, establishing that Graph Transformers need not choose between performance and interpretability.

---

## 14. Spatial Computing Communications for Multi-User Virtual Reality in Distributed Mobile Edge Computing Network

**论文链接:** [http://arxiv.org/abs/2510.14243v1](http://arxiv.org/abs/2510.14243v1)

**作者:** Caolu Xu, Zhiyong Chen, Meixia Tao, Li Song, Wenjun Zhang

**发布时间:** 2025-10-16

**备注:** submited to IEEE journal

### GPT解析

### 总结

本文提出了一种名为空间计算通信(SCC)的框架，用于解决多用户沉浸式VR应用在分布式MEC网络中的延迟和能源效率问题。通过MO-CMPO算法，结合监督学习和强化学习，实现了帕累托最优的资源部署方案。

### 背景

沉浸式VR应用对延迟、能源效率和计算资源有严格要求，特别是在多用户交互场景中。现有的分布式移动边缘计算(MEC)网络难以满足这些需求。

### 目的

开发一种框架来满足多用户VR在分布式MEC网络上的延迟和能源需求，并实现资源的高效部署。

### 方法

提出空间计算通信(SCC)框架，将资源部署任务表述为多目标组合优化问题，并设计MO-CMPO算法，结合监督学习和强化学习，利用稀疏图神经网络生成帕累托最优解。

### 主要发现

MO-CMPO比基线方法实现了更好的超体积性能和显著更低的推理延迟。以延迟为导向的解决方案倾向于本地MEC执行，而以能源为导向的解决方案则最小化冗余部署。

### 结论

SCC框架和MO-CMPO算法能够有效解决多用户VR应用在分布式MEC网络中的资源部署问题，平衡延迟和能源消耗。

### 翻译

沉浸式虚拟现实(VR)应用对延迟、能源效率和计算资源有严格要求，特别是在多用户交互场景中。为应对这些挑战，我们引入了空间计算通信(SCC)的概念，这是一个旨在满足分布式移动边缘计算(MEC)网络上多用户VR延迟和能源需求的框架。SCC使用用户动态和资源需求的概率模型，联合表示由用户和基站定义的物理空间，以及代表共享沉浸式环境的虚拟空间。然后，资源部署任务被表述为多目标组合优化(MOCO)问题，同时最小化分布式MEC资源上的系统延迟和能源消耗。为解决这个问题，我们提出了MO-CMPO，这是一种基于策略优化的多目标一致性模型，集成了监督学习和由偏好权重引导的强化学习(RL)微调。利用稀疏图神经网络(GNN)，MO-CMPO有效生成帕累托最优解。使用真实的新无线电基站数据集进行的模拟表明，MO-CMPO比基线方法实现了更好的超体积性能和显著更低的推理延迟。此外，分析揭示了实际的部署模式：以延迟为导向的解决方案倾向于本地MEC执行以减少传输延迟，而以能源为导向的解决方案则最小化冗余部署以节省能源。


### 论文摘要

Immersive virtual reality (VR) applications impose stringent requirements on latency, energy efficiency, and computational resources, particularly in multi-user interactive scenarios. To address these challenges, we introduce the concept of spatial computing communications (SCC), a framework designed to meet the latency and energy demands of multi-user VR over distributed mobile edge computing (MEC) networks. SCC jointly represents the physical space, defined by users and base stations, and the virtual space, representing shared immersive environments, using a probabilistic model of user dynamics and resource requirements. The resource deployment task is then formulated as a multi-objective combinatorial optimization (MOCO) problem that simultaneously minimizes system latency and energy consumption across distributed MEC resources. To solve this problem, we propose MO-CMPO, a multi-objective consistency model with policy optimization that integrates supervised learning and reinforcement learning (RL) fine-tuning guided by preference weights. Leveraging a sparse graph neural network (GNN), MO-CMPO efficiently generates Pareto-optimal solutions. Simulations with real-world New Radio base station datasets demonstrate that MO-CMPO achieves superior hypervolume performance and significantly lower inference latency than baseline methods. Furthermore, the analysis reveals practical deployment patterns: latency-oriented solutions favor local MEC execution to reduce transmission delay, while energy-oriented solutions minimize redundant placements to save energy.

---

## 15. Learning Wireless Interference Patterns: Decoupled GNN for Throughput Prediction in Heterogeneous Multi-Hop p-CSMA Networks

**论文链接:** [http://arxiv.org/abs/2510.14137v1](http://arxiv.org/abs/2510.14137v1)

**作者:** Faezeh Dehghan Tarzjani, Bhaskar Krishnamachari

**发布时间:** 2025-10-15

### GPT解析

### 总结

论文提出了解耦图卷积网络(D-GCN)来解决异构多跳无线网络中吞吐量预测的挑战。D-GCN通过分离节点自身传输概率与邻居干扰效应，使用可学习注意力替代平均聚合，实现了更准确的预测和可解释性，实验表明其显著优于现有方法。

### 背景

p持续CSMA协议是随机接入MAC分析的核心，但在异构多跳无线网络中预测饱和吞吐量仍是一个难题。简化的单一共享干扰域模型会低估吞吐量48-62%，而精确的马尔可夫链分析计算复杂度高，对大型网络不实用。

### 目的

开发可扩展的吞吐量预测方法，解决异构多跳无线网络中的计算障碍，适用于一般网络拓扑的结构化机器学习方法。

### 方法

提出解耦图卷积网络(D-GCN)，一种新型架构，明确分离节点自身的传输概率与邻居干扰效应的处理。用可学习的注意力替代平均聚合，产生可解释的每邻居贡献权重，同时捕获复杂的多跳干扰模式。

### 主要发现

D-GCN实现了3.3%的归一化平均绝对误差(NMAE)，显著优于标准GCN的63.94% NMAE。D-GCN性能优于强基线方法，即使在精确分析方法计算上不可行的情况下仍然可扩展，且使基于梯度的网络优化达到理论最优值的1%以内。

### 结论

D-GCN通过解耦处理和注意力机制，能够更准确地捕获网络中的复杂干扰模式，有效解决了异构多跳无线网络吞吐量预测问题。

### 翻译

p持续CSMA协议、饱和吞吐量、异构多跳无线网络、干扰域、马尔可夫链分析、结构化机器学习、图卷积网络(GNNs)、图卷积网络(GCN)、归一化平均绝对误差(NMAE)、对称归一化、级联效应、解耦图卷积网络(D-GCN)、可学习注意力、多跳干扰模式、基于梯度的网络优化


### 论文摘要

The p-persistent CSMA protocol is central to random-access MAC analysis, but predicting saturation throughput in heterogeneous multi-hop wireless networks remains a hard problem. Simplified models that assume a single, shared interference domain can underestimate throughput by 48--62\% in sparse topologies. Exact Markov-chain analyses are accurate but scale exponentially in computation time, making them impractical for large networks. These computational barriers motivate structural machine learning approaches like GNNs for scalable throughput prediction in general network topologies. Yet off-the-shelf GNNs struggle here: a standard GCN yields 63.94\% normalized mean absolute error (NMAE) on heterogeneous networks because symmetric normalization conflates a node's direct interference with higher-order, cascading effects that pertain to how interference propagates over the network graph.   Building on these insights, we propose the Decoupled Graph Convolutional Network (D-GCN), a novel architecture that explicitly separates processing of a node's own transmission probability from neighbor interference effects. D-GCN replaces mean aggregation with learnable attention, yielding interpretable, per-neighbor contribution weights while capturing complex multihop interference patterns. D-GCN attains 3.3\% NMAE, outperforms strong baselines, remains tractable even when exact analytical methods become computationally infeasible, and enables gradient-based network optimization that achieves within 1\% of theoretical optima.

---

## 16. On the expressivity of sparse maxout networks

**论文链接:** [http://arxiv.org/abs/2510.14068v1](http://arxiv.org/abs/2510.14068v1)

**作者:** Moritz Grillo, Tobias Hofmann

**发布时间:** 2025-10-15

### GPT解析

### 总结

本研究探讨了稀疏maxout网络的表达能力，建立了这类网络与虚拟多面体之间的对偶关系，分析了网络深度和宽度对表达能力的影响。

### 背景

研究聚焦于稀疏maxout网络，其中每个神经元从前一层接收固定数量的输入并采用maxout激活函数，这种结构类似于卷积神经网络或图神经网络的关键特征。

### 目的

目的是理解稀疏maxout网络的表达能力，特别是网络深度、宽度和稀疏性如何影响其计算能力。

### 方法

通过建立稀疏maxout网络可计算函数与虚拟多面体之间的对偶关系，推导出相关多面体维度的紧界，并基于此构建深度层次结构序列。

### 主要发现

研究发现足够深的稀疏maxout网络具有通用性，但如果未达到所需深度，仅靠宽度无法弥补固定入度约束的稀疏性。

### 结论

稀疏maxout网络的表达能力不仅取决于宽度，还与深度密切相关，深度不足时宽度无法完全补偿稀疏性的限制。

### 翻译

我们研究了稀疏maxout网络的表达能力，其中每个神经元从前一层接收固定数量的输入，并采用可能有多参数的maxout激活函数。这种设置捕捉了卷积神经网络或图神经网络的关键特征。我们建立了此类网络可计算的函数与一类虚拟多面体之间的对偶关系，将它们的几何形状与网络表达能力的问题联系起来。特别是，我们推导出相关多面体维度的紧界，作为我们分析的中心工具。在此基础上，我们构建了一个深度层次结构序列。虽然足够深的稀疏maxout网络是通用的，但我们证明，如果未达到所需深度，仅靠宽度无法弥补固定入度约束的稀疏性。


### 论文摘要

We study the expressivity of sparse maxout networks, where each neuron takes a fixed number of inputs from the previous layer and employs a, possibly multi-argument, maxout activation. This setting captures key characteristics of convolutional or graph neural networks. We establish a duality between functions computable by such networks and a class of virtual polytopes, linking their geometry to questions of network expressivity. In particular, we derive a tight bound on the dimension of the associated polytopes, which serves as the central tool for our analysis. Building on this, we construct a sequence of depth hierarchies. While sufficiently deep sparse maxout networks are universal, we prove that if the required depth is not reached, width alone cannot compensate for the sparsity of a fixed indegree constraint.

---

## 17. GammaZero: Learning To Guide POMDP Belief Space Search With Graph Representations

**论文链接:** [http://arxiv.org/abs/2510.14035v1](http://arxiv.org/abs/2510.14035v1)

**作者:** Rajesh Mangannavar, Prasad Tadepalli

**发布时间:** 2025-10-15

**备注:** 10 pages content. 2 pages references

### GPT解析

### 总结

GammaZero是一种以动作为中心的图表示框架，用于在部分可观察马尔可夫决策过程(POMDPs)中指导规划学习，解决了现有方法在可扩展性和泛化能力方面的局限性。

### 背景

现有方法需要特定领域的神经网络架构，并且难以处理大规模问题，限制了在POMDPs中的规划学习能力。

### 目的

开发一种统一的图表示框架，使学习到的策略能够在不同规模的问题间泛化，并减少对领域特定架构的需求。

### 方法

GammaZero将信念状态转换为以动作为中心的图，使用图神经网络结合解码器架构从专家演示中学习价值函数和策略，然后应用这些启发式指导蒙特卡洛树搜索。

### 主要发现

在相同规模问题上，GammaZero性能与BetaZero相当；同时能够实现零样本泛化，处理比训练时所见大2-4倍的问题，并在减少搜索需求的同时保持解决方案质量。

### 结论

GammaZero通过统一的图表示框架有效解决了POMDPs中的规划学习问题，实现了更好的泛化能力和可扩展性，为处理大规模部分可观察环境提供了新思路。

### 翻译

我们介绍了一种以动作为中心的图表示框架，用于学习在部分可观察马尔可夫决策过程(POMDPs)中指导规划。与需要特定领域神经网络架构且难以扩展的现有方法不同，GammaZero利用统一的基于图的信念表示，使问题能够在领域内跨规模泛化。我们的关键见解是信念状态可以系统地转换为以动作为中心的图，其中在小问题上学习的结构模式可以转移到更大的实例上。我们采用具有解码器架构的图神经网络，从计算可行问题上的专家演示中学习价值函数和策略，然后将这些学习到的启发式应用于指导更大问题上的蒙特卡洛树搜索。在标准POMDP基准测试上的实验结果表明，当在相同规模问题上训练和测试时，GammaZero与BetaZero相当，同时能够独特地实现零样本泛化到比训练时所见大2-4倍的问题，在减少搜索需求的同时保持解决方案质量。


### 论文摘要

We introduce an action-centric graph representation framework for learning to guide planning in Partially Observable Markov Decision Processes (POMDPs). Unlike existing approaches that require domain-specific neural architectures and struggle with scalability, GammaZero leverages a unified graph-based belief representation that enables generalization across problem sizes within a domain. Our key insight is that belief states can be systematically transformed into action-centric graphs where structural patterns learned on small problems transfer to larger instances. We employ a graph neural network with a decoder architecture to learn value functions and policies from expert demonstrations on computationally tractable problems, then apply these learned heuristics to guide Monte Carlo tree search on larger problems. Experimental results on standard POMDP benchmarks demonstrate that GammaZero achieves comparable performance to BetaZero when trained and tested on the same-sized problems, while uniquely enabling zero-shot generalization to problems 2-4 times larger than those seen during training, maintaining solution quality with reduced search requirements.

---

## 18. A Physics Prior-Guided Dual-Stream Attention Network for Motion Prediction of Elastic Bragg Breakwaters

**论文链接:** [http://arxiv.org/abs/2510.14250v1](http://arxiv.org/abs/2510.14250v1)

**作者:** Lianzi Jiang, Jianxin Zhang, Xinyu Han, Huanhe Dong, Xiangrong Wang

**发布时间:** 2025-10-16

### GPT解析

### 总结

该研究提出了一种名为PhysAttnNet的新型物理先验引导双流注意力网络，通过引入衰减双向自注意力和相位差引导的双向交叉注意力模块，有效解决了传统深度学习模型在预测弹性Bragg防波堤运动响应时面临的泛化能力有限问题。实验证明该模型在波浪槽数据集上表现优异，且对未见环境具有良好的适应性和鲁棒性。

### 背景

准确预测弹性Bragg防波堤的运动响应对于其在海洋环境中的结构安全和运行完整性至关重要。然而，传统的深度学习模型在面对未见过的海况时往往表现出有限的泛化能力。这些缺陷源于忽视了海洋系统中自然衰减现象，以及对波-结构相互作用的不充分建模。

### 目的

克服传统深度学习模型在预测弹性Bragg防波堤运动响应时面临的泛化能力有限问题，开发一种能够更好处理未见海况的预测模型。

### 方法

提出了一种名为PhysAttnNet的物理先验引导双流注意力网络，包含三个关键模块：1)衰减双向自注意力(DBSA)模块，通过可学习的时间衰减模拟自然衰减现象；2)相位差引导的双向交叉注意力(PDG-BCA)模块，明确捕获波与结构之间的双向相互作用和相位关系；3)全局上下文融合(GCF)模块，协同整合两个流。模型使用混合时频损失函数进行训练，同时最小化时域预测误差和频域频谱差异。

### 主要发现

在波浪槽数据集上的综合实验表明，PhysAttnNet显著优于主流模型。此外，跨场景泛化测试验证了模型对未见环境的鲁棒性和适应性。

### 结论

PhysAttnNet有潜力作为开发海洋工程复杂系统预测模型的框架，能够有效解决传统深度学习模型在海洋环境预测中面临的泛化能力有限问题。

### 翻译

准确预测弹性Bragg防波堤的运动响应对于其在海洋环境中的结构安全和运行完整性至关重要。然而，传统的深度学习模型在面对未见过的海况时往往表现出有限的泛化能力。这些缺陷源于忽视了海洋系统中自然衰减现象，以及对波-结构相互作用的不充分建模。为克服这些挑战，本研究提出了一种新颖的物理先验引导双流注意力网络(PhysAttnNet)。首先，衰减双向自注意力(DBSA)模块纳入了可学习的时间衰减，为最近的状态分配更高的权重，旨在模拟自然衰减现象。同时，相位差引导的双向交叉注意力(PDG-BCA)模块使用基于余弦的偏差在双向交叉计算范式中明确捕获波与结构之间的双向相互作用和相位关系。这些流通过全局上下文融合(GCF)模块协同整合。最后，PhysAttnNet使用混合时频损失进行训练，该损失函数同时最小化时域预测误差和频域频谱差异。在波浪槽数据集上的综合实验表明，PhysAttnNet显著优于主流模型。此外，跨场景泛化测试验证了模型对未见环境的鲁棒性和适应性，突显了其作为开发海洋工程复杂系统预测模型的框架的潜力。


### 论文摘要

Accurate motion response prediction for elastic Bragg breakwaters is critical for their structural safety and operational integrity in marine environments. However, conventional deep learning models often exhibit limited generalization capabilities when presented with unseen sea states. These deficiencies stem from the neglect of natural decay observed in marine systems and inadequate modeling of wave-structure interaction (WSI). To overcome these challenges, this study proposes a novel Physics Prior-Guided Dual-Stream Attention Network (PhysAttnNet). First, the decay bidirectional self-attention (DBSA) module incorporates a learnable temporal decay to assign higher weights to recent states, aiming to emulate the natural decay phenomenon. Meanwhile, the phase differences guided bidirectional cross-attention (PDG-BCA) module explicitly captures the bidirectional interaction and phase relationship between waves and the structure using a cosine-based bias within a bidirectional cross-computation paradigm. These streams are synergistically integrated through a global context fusion (GCF) module. Finally, PhysAttnNet is trained with a hybrid time-frequency loss that jointly minimizes time-domain prediction errors and frequency-domain spectral discrepancies. Comprehensive experiments on wave flume datasets demonstrate that PhysAttnNet significantly outperforms mainstream models. Furthermore,cross-scenario generalization tests validate the model's robustness and adaptability to unseen environments, highlighting its potential as a framework to develop predictive models for complex systems in ocean engineering.

---

## 19. DPRF: A Generalizable Dynamic Persona Refinement Framework for Optimizing Behavior Alignment Between Personalized LLM Role-Playing Agents and Humans

**论文链接:** [http://arxiv.org/abs/2510.14205v1](http://arxiv.org/abs/2510.14205v1)

**作者:** Bingsheng Yao, Bo Sun, Yuanzhe Dong, Yuxuan Lu, Dakuo Wang

**发布时间:** 2025-10-16

**备注:** In Submission

### GPT解析

### 总结

该研究提出了一种名为动态人格完善框架(DPRF)的新方法，用于优化大型语言模型角色扮演代理的行为与目标个体行为的一致性，通过迭代识别认知差异并完善人格配置文件，显著提高了行为对齐度。

### 背景

新兴的大型语言模型角色扮演代理旨在模拟个体人类行为，但其人格保真度常因手动创建的配置文件(例如，精心挑选的信息和人格特征)而受损，这些配置文件未经验证是否与目标个体保持一致。

### 目的

解决LLM RPAs行为与目标个体行为不一致的问题，通过优化LLM RPAs的行为与目标个体行为的对齐度。

### 方法

提出动态人格完善框架(DPRF)，通过迭代识别生成行为与人类真实行为之间的认知差异(无论是自由形式还是基于理论的结构化分析)，并完善人格配置文件以减轻这些差异。在四个多样化的行为预测场景(正式辩论、心理健康问题的社交媒体帖子、公开采访和电影评论)中使用五个大型语言模型评估DPRF。

### 主要发现

DPRF能够一致地显著提高基线人格的行为一致性，并且在模型和场景方面具有通用性。

### 结论

该研究为创建高保真度人格配置文件和增强下游应用(如用户模拟、社会研究和个性化AI)的有效性提供了稳健的方法论。

### 翻译

新兴的大型语言模型角色扮演代理旨在模拟个体人类行为，但其人格保真度常因手动创建的配置文件(例如，精心挑选的信息和人格特征)而受损，这些配置文件未经验证是否与目标个体保持一致。为解决这一限制，我们的工作引入了动态人格完善框架(DPRF)。DPRF旨在通过迭代识别生成行为与人类真实行为之间的认知差异(无论是自由形式还是基于理论的结构化分析)，并完善人格配置文件以减轻这些差异，从而优化LLM RPAs的行为与目标个体行为的一致性。我们在四个多样化的行为预测场景中使用五个大型语言模型评估DPRF：正式辩论、心理健康问题的社交媒体帖子、公开采访和电影评论。DPRF能够一致地显著提高基线人格的行为一致性，并且在模型和场景方面具有通用性。我们的研究为创建高保真度人格配置文件和增强下游应用(如用户模拟、社会研究和个性化AI)的有效性提供了稳健的方法论。


### 论文摘要

The emerging large language model role-playing agents (LLM RPAs) aim to simulate individual human behaviors, but the persona fidelity is often undermined by manually-created profiles (e.g., cherry-picked information and personality characteristics) without validating the alignment with the target individuals. To address this limitation, our work introduces the Dynamic Persona Refinement Framework (DPRF).DPRF aims to optimize the alignment of LLM RPAs' behaviors with those of target individuals by iteratively identifying the cognitive divergence, either through free-form or theory-grounded, structured analysis, between generated behaviors and human ground truth, and refining the persona profile to mitigate these divergences.We evaluate DPRF with five LLMs on four diverse behavior-prediction scenarios: formal debates, social media posts with mental health issues, public interviews, and movie reviews.DPRF can consistently improve behavioral alignment considerably over baseline personas and generalizes across models and scenarios.Our work provides a robust methodology for creating high-fidelity persona profiles and enhancing the validity of downstream applications, such as user simulation, social studies, and personalized AI.

---

## 20. Energy-Guided Diffusion Sampling for Long-Term User Behavior Prediction in Reinforcement Learning-based Recommendation

**论文链接:** [http://arxiv.org/abs/2510.12815v1](http://arxiv.org/abs/2510.12815v1)

**作者:** Xiaocong Chen, Siyu Wang, Lina Yao

**发布时间:** 2025-10-09

**备注:** CIKM'25

### GPT解析

### 总结

基于强化学习的推荐系统(RL4RS)在离线设置下面临数据效率低和依赖预收集轨迹的挑战。本文提出了一种名为DAC4Rec的新框架，整合扩散过程与强化学习，有效解决噪声数据处理和长期用户偏好捕捉问题。

### 背景

基于强化学习的推荐系统能够适应用户的动态偏好，但在离线设置下面临数据效率低和依赖预收集轨迹的挑战。离线强化学习方法利用大量数据解决这些问题，但往往难以处理嘈杂数据且无法捕捉长期用户偏好。

### 目的

克服现有离线强化学习推荐系统的局限性，提出一种新的框架来更有效地建模复杂的用户偏好。

### 方法

提出了一种名为Diffusion-enhanced Actor-Critic for Offline RL4RS (DAC4Rec)的新框架，该框架整合了扩散过程与强化学习。DAC4Rec利用扩散模型的去噪能力增强离线强化学习算法的鲁棒性，并采用Q值引导的策略优化策略来更好地处理次优轨迹。此外，还引入了一种基于能量的采样策略来减少推荐生成过程中的随机性。

### 主要发现

通过在六个真实世界离线数据集和在线模拟环境中的大量实验验证了DAC4Rec的有效性，证明其能够优化长期用户偏好。此外，提出的扩散策略可以无缝集成到RL4RS中其他常用的强化学习算法中，展示了其多功能性和广泛的适用性。

### 结论

DAC4Rec框架通过整合扩散过程与强化学习，有效解决了离线强化学习推荐系统中的数据效率、噪声处理和长期偏好捕捉等问题。

### 翻译

基于强化学习的推荐系统(RL4RS)因其能够适应动态用户偏好而受到关注。然而，这些系统面临挑战，特别是在离线设置中，数据效率低下和对预收集轨迹的依赖限制了它们的广泛应用。虽然离线强化学习方法利用大量数据来解决这些问题，但它们通常难以处理嘈杂数据且无法捕捉长期用户偏好，导致次优的推荐策略。为了克服这些局限性，我们提出了用于离线RL4RS的扩散增强型Actor-Critic(DAC4Rec)，这是一个将扩散过程与强化学习相结合的新颖框架，能够更有效地建模复杂的用户偏好。DAC4Rec利用扩散模型的去噪能力增强离线强化学习算法的鲁棒性，并采用Q值引导的策略优化策略来更好地处理次优轨迹。此外，我们引入了一种基于能量的采样策略来减少推荐生成过程中的随机性，确保更有针对性和可靠的结果。我们在六个真实世界的离线数据集和在线模拟环境中通过大量实验验证了DAC4Rec的有效性，证明了其优化长期用户偏好的能力。此外，我们表明所提出的扩散策略可以无缝集成到RL4RS中其他常用的强化学习算法中，突显了其多功能性和广泛的适用性。


### 论文摘要

Reinforcement learning-based recommender systems (RL4RS) have gained attention for their ability to adapt to dynamic user preferences. However, these systems face challenges, particularly in offline settings, where data inefficiency and reliance on pre-collected trajectories limit their broader applicability. While offline reinforcement learning methods leverage extensive datasets to address these issues, they often struggle with noisy data and fail to capture long-term user preferences, resulting in suboptimal recommendation policies. To overcome these limitations, we propose Diffusion-enhanced Actor-Critic for Offline RL4RS (DAC4Rec), a novel framework that integrates diffusion processes with reinforcement learning to model complex user preferences more effectively. DAC4Rec leverages the denoising capabilities of diffusion models to enhance the robustness of offline RL algorithms and incorporates a Q-value-guided policy optimization strategy to better handle suboptimal trajectories. Additionally, we introduce an energy-based sampling strategy to reduce randomness during recommendation generation, ensuring more targeted and reliable outcomes. We validate the effectiveness of DAC4Rec through extensive experiments on six real-world offline datasets and in an online simulation environment, demonstrating its ability to optimize long-term user preferences. Furthermore, we show that the proposed diffusion policy can be seamlessly integrated into other commonly used RL algorithms in RL4RS, highlighting its versatility and wide applicability.

---

## 21. UrbanTwin: Synthetic LiDAR Datasets (LUMPI, V2X-Real-IC, and TUMTraf-I)

**论文链接:** [http://arxiv.org/abs/2509.06781v2](http://arxiv.org/abs/2509.06781v2)

**作者:** Muhammad Shahbaz, Shaurya Agarwal

**发布时间:** 2025-09-08

### GPT解析

### 总结

本文介绍了UrbanTwin数据集，这是三个公共路边激光雷达数据集的高保真真实副本，每个包含10K个注释帧，具有丰富的标注信息，能够有效支持深度学习模型训练。

### 背景

激光雷达感知任务需要大量高质量数据集进行模型训练，但真实数据集获取和标注成本高，且场景多样性有限。

### 目的

创建高保真合成数据集，能够独立使用或增强现有数据集，用于激光雷达感知任务，并探索其能否替代同领域真实世界数据集。

### 方法

基于实际位置的几何特征、道路对齐和交通模式构建数字孪生环境，使用模拟激光雷达传感器合成数据，添加3D边界框、实例分割和语义分割等标注，并通过统计和结构相似性分析评估数据质量。

### 主要发现

合成数据集与真实数据高度相似，仅使用合成数据训练的模型在真实未见数据上表现优于使用真实数据训练的模型，数据集通过增加样本量和场景多样性有效增强了基准数据集。

### 结论

UrbanTwin数据集是首批能够替换同领域真实世界数据集的数字合成数据集，提供了高保真数据副本，支持自定义场景测试，已公开可供使用。

### 翻译

这篇文章介绍了UrbanTwin数据集，这是三个公共路边激光雷达数据集的高保真真实副本。每个UrbanTwin数据集包含10K个注释帧，对应一个公共数据集。注释包括6个类别的3D边界框、实例分割标签和跟踪ID，以及9个类别的语义分割标签。这些数据集使用模拟激光雷达传感器在真实数字孪生中合成，基于实际位置的周围几何形状、车道级别的道路对齐以及交叉口的车道拓扑和车辆移动模式进行建模。由于精确的数字孪生建模，合成数据集与真实数据集很好地对齐，为训练深度学习模型提供了强大的独立和增强价值。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决高质量激光雷达数据集创建困难的问题。真实世界数据收集和标注成本高、耗时长，限制了智能交通系统感知算法的发展。这个问题很重要，因为激光雷达是智能交通系统中的关键技术，而高质量数据集对于训练和评估3D感知算法至关重要。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者认识到现有模拟环境虽然功能强大，但与真实世界存在差距。他们借鉴了数字孪生概念，结合了CARLA模拟器和现有路边激光雷达数据集的特点。作者强调需要同时建模静态元素(如几何结构)和动态行为(如交通模式)，而非仅依赖手工制作的3D资产和简化的物理假设。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是使用数字孪生技术创建真实世界场景的高保真虚拟副本，模拟激光雷达传感器生成与真实数据相似的点云，并提供丰富一致的标注。实现流程包括：1)使用卫星图像和真实位置数据构建环境；2)配置虚拟传感器匹配真实规格；3)随机生成符合交通规则的动态元素；4)在CARLA模拟器中生成10K帧带标注的合成数据。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：首次专门为路边激光雷达应用创建合成数据集；采用高保真数字孪生同时整合静态和动态元素；合成数据与真实数据高度相似；证明完全在模拟数据上训练的模型可匹敌真实数据训练效果。相比之前工作，UrbanTwin专门增强真实世界基准而非通用模拟，在模拟过程中而非事后缩小sim-to-real差距，提供完整标注支持多种感知任务。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'UrbanTwin通过创建基于数字孪生的高保真合成激光雷达数据集，成功解决了真实世界数据集创建成本高昂且sim-to-real差距大的问题，使模型能在合成数据上训练并有效应用于真实世界感知任务。'}


### 论文摘要

This article presents UrbanTwin datasets, high-fidelity, realistic replicas of three public roadside lidar datasets: LUMPI, V2X-Real-IC}}, and TUMTraf-I. Each UrbanTwin dataset contains 10K annotated frames corresponding to one of the public datasets. Annotations include 3D bounding boxes, instance segmentation labels, and tracking IDs for six object classes, along with semantic segmentation labels for nine classes. These datasets are synthesized using emulated lidar sensors within realistic digital twins, modeled based on surrounding geometry, road alignment at lane level, and the lane topology and vehicle movement patterns at intersections of the actual locations corresponding to each real dataset. Due to the precise digital twin modeling, the synthetic datasets are well aligned with their real counterparts, offering strong standalone and augmentative value for training deep learning models on tasks such as 3D object detection, tracking, and semantic and instance segmentation. We evaluate the alignment of the synthetic replicas through statistical and structural similarity analysis with real data, and further demonstrate their utility by training 3D object detection models solely on synthetic data and testing them on real, unseen data. The high similarity scores and improved detection performance, compared to the models trained on real data, indicate that the UrbanTwin datasets effectively enhance existing benchmark datasets by increasing sample size and scene diversity. In addition, the digital twins can be adapted to test custom scenarios by modifying the design and dynamics of the simulations. To our knowledge, these are the first digitally synthesized datasets that can replace in-domain real-world datasets for lidar perception tasks. UrbanTwin datasets are publicly available at https://dataverse.harvard.edu/dataverse/ucf-ut.

---

## 22. Backdoor Unlearning by Linear Task Decomposition

**论文链接:** [http://arxiv.org/abs/2510.14845v1](http://arxiv.org/abs/2510.14845v1)

**作者:** Amel Abdelraheem, Alessandro Favero, Gerome Bovet, Pascal Frossard

**发布时间:** 2025-10-16

### GPT解析

### 总结

这项研究解决了基础模型中后门攻击的安全问题，提出了一种基于后门与良性任务解耦特性的简单遗忘方法，能够在不损害模型通用能力的情况下有效移除后门。

### 背景

基础模型通过在多样化任务中实现广泛的泛化能力彻底改变了计算机视觉领域。然而，它们仍然容易受到对抗性扰动和定向后门攻击的影响。缓解此类脆弱性仍然是一个开放的挑战，特别是考虑到模型的大规模性质使得重新训练以确保安全性变得不可行。

### 目的

回答后门是否可以在不损害模型通用能力的情况下被移除这一问题，并研究后门如何在模型权重空间中被编码。

### 方法

研究后门与良性任务在模型权重空间中的解耦特性，基于这种分离开发一种简单的遗忘方法，能够隔离和擦除后门对模型的影响，同时保持干净性能。通过基于CLIP的模型和常见对抗触发器进行大量实验验证。

### 主要发现

后门与其他良性任务是解耦的；给定攻击知识的情况下，方法实现了近乎完美的遗忘，同时平均保留了96%的干净准确率；即使当攻击及其存在未知时，方法也能通过反向工程触发器的适当估计成功遗忘后门；与当前最先进的防御相比，方法始终产生更好的遗忘和干净准确率权衡。

### 结论

该方法在移除后门的同时，有效保留了模型的通用能力，为解决基础模型的安全问题提供了新的思路。

### 翻译

基础模型通过在多样化任务中实现广泛的泛化能力彻底改变了计算机视觉领域。然而，它们仍然容易受到对抗性扰动和定向后门攻击的影响。缓解此类脆弱性仍然是一个开放的挑战，特别是考虑到模型的大规模性质使得重新训练以确保安全性变得不可行。现有的后门移除方法依赖于昂贵的微调来覆盖有害行为，并且通常会降低在其他不相关任务上的性能。这引发了一个问题：后门是否可以在不损害模型通用能力的情况下被移除。在本研究中，我们解决了这个问题，并研究了后门如何在模型权重空间中被编码，发现它们与其他良性任务是解耦的。具体而言，这种分离使得能够隔离和擦除后门对模型的影响，同时对干净性能的影响最小。基于这一见解，我们引入了一种利用这种解耦特性的简单遗忘方法。通过对基于CLIP的模型和常见对抗触发器的大量实验，我们表明，给定攻击知识的情况下，我们的方法实现了近乎完美的遗忘，同时平均保留了96%的干净准确率。此外，我们证明即使当攻击及其存在未知时，我们的方法也能通过使用反向工程触发器的适当估计成功遗忘后门。总体而言，与当前最先进的防御相比，我们的方法始终产生更好的遗忘和干净准确率权衡。


### 论文摘要

Foundation models have revolutionized computer vision by enabling broad generalization across diverse tasks. Yet, they remain highly susceptible to adversarial perturbations and targeted backdoor attacks. Mitigating such vulnerabilities remains an open challenge, especially given that the large-scale nature of the models prohibits retraining to ensure safety. Existing backdoor removal approaches rely on costly fine-tuning to override the harmful behavior, and can often degrade performance on other unrelated tasks. This raises the question of whether backdoors can be removed without compromising the general capabilities of the models. In this work, we address this question and study how backdoors are encoded in the model weight space, finding that they are disentangled from other benign tasks. Specifically, this separation enables the isolation and erasure of the backdoor's influence on the model with minimal impact on clean performance. Building on this insight, we introduce a simple unlearning method that leverages such disentanglement. Through extensive experiments with CLIP-based models and common adversarial triggers, we show that, given the knowledge of the attack, our method achieves approximately perfect unlearning, while retaining, on average, 96% of clean accuracy. Additionally, we demonstrate that even when the attack and its presence are unknown, our method successfully unlearns backdoors by proper estimation using reverse-engineered triggers. Overall, our method consistently yields better unlearning and clean accuracy tradeoffs when compared to present state-of-the-art defenses.

---

## 23. Morphology-Aware Prognostic model for Five-Year Survival Prediction in Colorectal Cancer from H&E Whole Slide Images

**论文链接:** [http://arxiv.org/abs/2510.14800v1](http://arxiv.org/abs/2510.14800v1)

**作者:** Usama Sajjad, Abdul Rehman Akbar, Ziyu Su, Deborah Knight, Wendy L. Frankel, Metin N. Gurcan, Wei Chen, Muhammad Khalid Khan Niazi

**发布时间:** 2025-10-16

### GPT解析

### 总结

本研究开发了一种名为PRISM的新型可解释AI模型，用于结直肠癌预后预测。该模型通过整合连续变异性谱的形态学信息，能够更准确地捕捉肿瘤的渐进式进化过程，并在III期结直肠癌患者中展现出优异的预后预测性能。

### 背景

结直肠癌是全球第三大常见恶性肿瘤，预计2025年将有约154,000新病例和54,000例死亡。当前计算机病理学中的基础模型主要采用任务无关的方法学，可能忽略器官特定的关键形态学模式，而这些模式对肿瘤行为、治疗反应和患者结局有重要影响。

### 目的

开发一种新型、可解释的AI模型PRISM（预后性整合空间形态表征），纳入每种不同形态内的连续变异性谱以表征表型多样性，反映恶性肿瘤转化是通过渐进式进化过程而非表型急剧转变发生的原理。

### 方法

PRISM模型在874万张组织学图像上进行训练，这些图像来自424名III期结直肠癌患者的手术切除标本。模型整合了空间形态学信息，以捕捉肿瘤的形态学变异性。

### 主要发现

PRISM在五年总生存期(OS)预后方面表现优越：AUC = 0.70 ± 0.04；准确率 = 68.37% ± 4.75%；风险比(HR) = 3.34，95% CI = 2.28-4.90，p < 0.0001。模型优于现有的结直肠癌特异性方法15%，比AI基础模型高约23%的准确率。PRISM显示性别无关的稳健性，在临床病理亚组中表现稳定，在不同治疗方案间的准确率波动最小（差值 = 1.44%），复现了Alliance队列的研究结果，即两种治疗方案之间无生存差异。

### 结论

PRISM模型在结直肠癌预后预测方面表现优异，能够更好地捕捉肿瘤的形态学变异性，对不同治疗方案的患者预后有稳定的预测能力，为临床决策提供了有价值的工具。

### 翻译

结直肠癌(CRC)仍然是全球第三大常见恶性肿瘤，预计2025年将有约154,000新病例和54,000例死亡。最近，计算病理学中基础模型的进展主要是由任务无关的方法学推动的，这些方法可能忽略器官特定的关键形态学模式，这些模式代表不同的生物学过程，能从根本上影响肿瘤行为、治疗反应和患者结局。本研究旨在开发一种新型、可解释的AI模型PRISM（预后性整合空间形态表征），该模型纳入了每种不同形态内的连续变异性谱，以表征表型多样性，并反映恶性肿瘤转化是通过渐进式进化过程而非表型急剧转变发生的原理。PRISM在从424名III期CRC患者的手术切除标本中提取的874万张组织学图像上进行训练。PRISM在五年OS预后方面取得了优异的性能（AUC = 0.70 ± 0.04；准确率 = 68.37% ± 4.75%；HR = 3.34，95% CI = 2.28-4.90；p < 0.0001），比现有的CRC特异性方法高出15%，比AI基础模型高出约23%的准确率。它显示出性别无关的稳健性（AUC差值 = 0.02；准确率差值 = 0.15%），并在临床病理亚组中表现稳定，在5FU/LV和CPT-11/5FU/LV治疗方案之间的准确率波动最小（差值 = 1.44%），复现了Alliance队列的研究结果，即两种治疗方案之间无生存差异。


### 论文摘要

Colorectal cancer (CRC) remains the third most prevalent malignancy globally, with approximately 154,000 new cases and 54,000 projected deaths anticipated for 2025. The recent advancement of foundation models in computational pathology has been largely propelled by task agnostic methodologies that can overlook organ-specific crucial morphological patterns that represent distinct biological processes that can fundamentally influence tumor behavior, therapeutic response, and patient outcomes. The aim of this study is to develop a novel, interpretable AI model, PRISM (Prognostic Representation of Integrated Spatial Morphology), that incorporates a continuous variability spectrum within each distinct morphology to characterize phenotypic diversity and reflecting the principle that malignant transformation occurs through incremental evolutionary processes rather than abrupt phenotypic shifts. PRISM is trained on 8.74 million histological images extracted from surgical resection specimens of 424 patients with stage III CRC. PRISM achieved superior prognostic performance for five-year OS (AUC = 0.70 +- 0.04; accuracy = 68.37% +- 4.75%; HR = 3.34, 95% CI = 2.28-4.90; p < 0.0001), outperforming existing CRC-specific methods by 15% and AI foundation models by ~23% accuracy. It showed sex-agnostic robustness (AUC delta = 0.02; accuracy delta = 0.15%) and stable performance across clinicopathological subgroups, with minimal accuracy fluctuation (delta = 1.44%) between 5FU/LV and CPT-11/5FU/LV regimens, replicating the Alliance cohort finding of no survival difference between treatments.

---

## 24. COIG-Writer: A High-Quality Dataset for Chinese Creative Writing with Thought Processes

**论文链接:** [http://arxiv.org/abs/2510.14763v1](http://arxiv.org/abs/2510.14763v1)

**作者:** Yunwen Li, Shuangshuang Ying, Xingwei Qu, Xin Li, Sheng Jin, Minghao Liu, Zhoufutu Wen, Tianyu Zheng, Xeron Du, Qiguang Chen, Jiajun Shi, Wangchunshu Zhou, Jiazhan Feng, Wanjun Zhong, Libo Qin, Stephen Huang, Wanxiang Che, Chenghua Lin, Eli Zhang

**发布时间:** 2025-10-16

### GPT解析

### 总结

该研究针对大型语言模型在创意写作方面的局限性，特别是在非英语环境中的不足，提出了一个新颖的中文创意写作数据集COIG-Writer，并通过实验确定了创意写作的双组分模型及其关键发现。

### 背景

大型语言模型在创意写作方面存在系统性缺陷，特别是在非英语语境中，训练数据稀缺且缺乏过程层面的监督。

### 目的

开发一个能够捕捉多样化输出及其背后思维过程的中文创意写作数据集，并研究创意写作的构成要素和优化方法。

### 方法

创建了COIG-Writer数据集，包含1665个精心挑选的三元组，涵盖51个体裁，每个三元组包含逆向工程提示、详细创意推理和最终文本。通过全面实验分析创意写作的构成要素和优化方法。

### 主要发现

1. 过程监督非常有效，但需要通用数据稳定化，至少需要一个创意样本对应十二个通用样本才能实现最佳性能；2. 创意能力具有文化局限性，没有跨语言迁移能力，中文和英文表现之间有89.26百分点的差距；3. 词汇多样性与创意质量呈负相关（TTR悖论），高多样性信号表明对逻辑缺陷的补偿行为。

### 结论

创意卓越来自于逻辑支架和语言基础的相互作用，类似于数学推理如何增强但不能替代基础模型中的语言能力。创意写作需要过程监督和通用数据的适当平衡。

### 翻译

大型语言模型在创意写作方面表现出系统性缺陷，特别是在非英语环境中，训练数据稀缺且缺乏过程层面的监督。我们提出了COIG-Writer，这是一个新颖的中文创意写作数据集，通过对高质量文本进行系统性的逆向工程，捕捉多样化的输出及其背后的思维过程。与仅提供输入-输出对的数据集不同，COIG-Writer包含1665个精心挑选的三元组，涵盖51个体裁，每个三元组包含：(1)逆向工程提示，(2)详细创意推理记录决策过程，(3)最终文本。通过全面实验，我们确定了创意写作的双组分模型：叙事逻辑（由过程监督提供）和语言表达（由通用数据维持）。我们的研究揭示了三个关键见解：(1)过程监督非常有效，但需要通用数据稳定化。至少需要一个创意样本对应十二个通用样本的比例才能实现最佳性能；低于此阈值，胜率会逐渐下降（从62.75%降至35.78%）；(2)创意能力具有文化局限性，没有跨语言迁移能力（中文和英文表现之间有89.26百分点的差距）；(3)词汇多样性与创意质量呈负相关（TTR悖论），表明高多样性信号表明对逻辑缺陷的补偿行为。这些发现表明，创意卓越来自于逻辑支架和语言基础的相互作用，类似于数学推理如何增强但不能替代基础模型中的语言能力。


### 论文摘要

Large language models exhibit systematic deficiencies in creative writing, particularly in non-English contexts where training data is scarce and lacks process-level supervision. We present COIG-Writer, a novel Chinese creative writing dataset that captures both diverse outputs and their underlying thought processes through systematic reverse-engineering of high-quality texts. Unlike existing datasets that provide only input-output pairs, COIG-Writer comprises 1,665 meticulously curated triplets spanning 51 genres, each containing: (1) a reverse-engineered prompt, (2) detailed creative reasoning documenting decision-making processes, and (3) the final text. Through comprehensive experiments, we identify a two-component model of creative writing: narrative logic (provided by process supervision) and linguistic expression (maintained by general-purpose data). Our findings reveal three critical insights: (1) Process supervision is highly effective but requires stabilization with general data. A ratio of at least one creative sample to twelve general samples is needed to achieve optimal performance; below this threshold, the win rate progressively degrades (from 62.75% down to 35.78%)., (2) creative capabilities are culturally-bound with no cross-lingual transfer (89.26pp gap between Chinese and English performance), and (3) lexical diversity inversely correlates with creative quality (TTR paradox), suggesting high diversity signals compensatory behavior for logical deficiencies. These findings establish that creative excellence emerges from the interaction between logical scaffolding and linguistic grounding, analogous to how mathematical reasoning enhances but cannot replace linguistic competence in foundation models.

---

## 25. State-Space Models for Tabular Prior-Data Fitted Networks

**论文链接:** [http://arxiv.org/abs/2510.14573v1](http://arxiv.org/abs/2510.14573v1)

**作者:** Felix Koch, Marcel Wever, Fabian Raisch, Benjamin Tischler

**发布时间:** 2025-10-16

### GPT解析

### 总结

本研究探讨了使用Hydra（一种双向线性时间结构状态空间模型）替代TabPFN中的Transformer架构，以解决Transformer的二次复杂度问题，同时保持预测性能。

### 背景

基础模型在表格数据领域取得了进展，如TabPFN展示了预训练Transformer架构可以高预测性能近似贝叶斯推断。然而，Transformer在序列长度上具有二次复杂度，促使人们探索更高效的序列模型。

### 目的

研究Hydra作为TabPFN中Transformer替代方案的潜力，解决SSM对输入标记顺序的固有敏感性这一关键挑战，特别是在表格数据集中行顺序语义无意义的情况下。

### 方法

研究双向方法能在多大程度上保持效率并实现对称上下文聚合，以减少SSM对输入顺序的依赖性。

### 主要发现

实验表明，这种方法减少了顺序依赖性，实现了与原始TabPFN模型相当的预测性能。

### 结论

双向Hydra模型可以作为TabPFN中Transformer的有效替代方案，在保持预测性能的同时提高效率。

### 翻译

最近在表格数据基础模型方面的进展，如TabPFN，表明预训练的Transformer架构可以以高预测性能近似贝叶斯推断。然而，Transformer在序列长度上具有二次复杂度，促使人们探索更高效的序列模型。在这项工作中，我们研究了使用Hydra（一种双向线性时间结构状态空间模型SSM）作为TabPFN中Transformer替代方案的潜力。一个关键挑战在于SSM对输入标记顺序的固有敏感性——对于行顺序在语义上无意义的表格数据集来说，这是一个不希望有的特性。我们研究了双向方法在多大程度上可以保持效率并实现对称上下文聚合。我们的实验表明，这种方法减少了顺序依赖性，实现了与原始TabPFN模型具有竞争力的预测性能。


### 论文摘要

Recent advancements in foundation models for tabular data, such as TabPFN, demonstrated that pretrained Transformer architectures can approximate Bayesian inference with high predictive performance. However, Transformers suffer from quadratic complexity with respect to sequence length, motivating the exploration of more efficient sequence models. In this work, we investigate the potential of using Hydra, a bidirectional linear-time structured state space model (SSM), as an alternative to Transformers in TabPFN. A key challenge lies in SSM's inherent sensitivity to the order of input tokens - an undesirable property for tabular datasets where the row order is semantically meaningless. We investigate to what extent a bidirectional approach can preserve efficiency and enable symmetric context aggregation. Our experiments show that this approach reduces the order-dependence, achieving predictive performance competitive to the original TabPFN model.

---

## 26. Towards Generalist Intelligence in Dentistry: Vision Foundation Models for Oral and Maxillofacial Radiology

**论文链接:** [http://arxiv.org/abs/2510.14532v1](http://arxiv.org/abs/2510.14532v1)

**作者:** Xinrui Huang, Fan Xiao, Dongming He, Anqi Gao, Dandan Li, Xiaofan Zhang, Shaoting Zhang, Xudong Wang

**发布时间:** 2025-10-16

### GPT解析

### 总结

DentVFM是首个专为牙科设计的视觉基础模型系列，解决了现有牙科AI系统的局限性，通过自监督学习和大规模多模态数据集训练，展现出卓越的泛化能力和跨模态诊断性能。

### 背景

口腔颌面放射学在牙科医疗中至关重要，但受专业人才短缺限制。现有牙科AI系统因单一模态关注、任务特定设计和依赖标记数据而泛化能力有限。

### 目的

开发一种能够克服现有AI系统局限性的牙科视觉基础模型，实现更广泛的应用和更好的泛化能力。

### 方法

创建DentVFM模型系列，使用DentVista数据集(约160万多模态放射图像)进行自监督学习，基于Vision Transformer架构开发2D和3D变体，并建立DentBench基准测试涵盖8个牙科亚专科。

### 主要发现

DentVFM表现出通用智能，能推广到多种牙科任务；显著优于各类基线模型；提供更好的泛化能力、标签效率和可扩展性；在跨模态诊断中表现优于经验丰富的牙医。

### 结论

DentVFM为牙科AI树立新范式，提供可扩展、适应性强且标签高效的模型，有助于改善智能牙科医疗保健并解决全球口腔医疗保健差距。

### 翻译

口腔颌面放射学在牙科医疗保健中起着重要作用，但放射图像解读受到训练专业人员短缺的限制。虽然AI方法显示出前景，但现有牙科AI系统受限于其单一模态关注、任务特定设计和依赖昂贵的标记数据，阻碍了它们在多样化临床场景中的泛化能力。为解决这些挑战，我们引入了DentVFM，这是首个为牙科设计的视觉基础模型系列。DentVFM为广泛的牙科应用生成任务无关的视觉表示，并在DentVista上使用自监督学习，这是一个精心策划的大型牙科成像数据集，包含来自不同医疗中心的约160万张多模态放射图像。DentVFM基于Vision Transformer架构包含2D和3D变体。为解决牙科智能评估和基准测试的空白，我们引入了DentBench，这是一个全面的基准测试，涵盖八个牙科亚专科、更多疾病、成像方式和广泛的地理分布。DentVFM表现出令人印象深刻的通用智能，展示了向多样化牙科任务的稳健泛化能力，如疾病诊断、治疗分析、生物标志物识别以及解剖标志物检测和分割。实验结果表明，DentVFM显著优于监督、自监督和弱监督基线，提供更好的泛化能力、标签效率和可扩展性。此外，DentVFM实现跨模态诊断，在常规成像不可用的情况下提供比经验丰富的牙医更可靠的结果。DentVFM为牙科AI树立了新范式，提供可扩展、适应性强且标签高效的模型，以改善智能牙科医疗保健并解决全球口腔医疗保健中的关键差距。


### 论文摘要

Oral and maxillofacial radiology plays a vital role in dental healthcare, but radiographic image interpretation is limited by a shortage of trained professionals. While AI approaches have shown promise, existing dental AI systems are restricted by their single-modality focus, task-specific design, and reliance on costly labeled data, hindering their generalization across diverse clinical scenarios. To address these challenges, we introduce DentVFM, the first family of vision foundation models (VFMs) designed for dentistry. DentVFM generates task-agnostic visual representations for a wide range of dental applications and uses self-supervised learning on DentVista, a large curated dental imaging dataset with approximately 1.6 million multi-modal radiographic images from various medical centers. DentVFM includes 2D and 3D variants based on the Vision Transformer (ViT) architecture. To address gaps in dental intelligence assessment and benchmarks, we introduce DentBench, a comprehensive benchmark covering eight dental subspecialties, more diseases, imaging modalities, and a wide geographical distribution. DentVFM shows impressive generalist intelligence, demonstrating robust generalization to diverse dental tasks, such as disease diagnosis, treatment analysis, biomarker identification, and anatomical landmark detection and segmentation. Experimental results indicate DentVFM significantly outperforms supervised, self-supervised, and weakly supervised baselines, offering superior generalization, label efficiency, and scalability. Additionally, DentVFM enables cross-modality diagnostics, providing more reliable results than experienced dentists in situations where conventional imaging is unavailable. DentVFM sets a new paradigm for dental AI, offering a scalable, adaptable, and label-efficient model to improve intelligent dental healthcare and address critical gaps in global oral healthcare.

---

## 27. Vision Mamba for Permeability Prediction of Porous Media

**论文链接:** [http://arxiv.org/abs/2510.14516v1](http://arxiv.org/abs/2510.14516v1)

**作者:** Ali Kashefi, Tapan Mukerji

**发布时间:** 2025-10-16

### GPT解析

### 总结

本文首次引入使用Vision Mamba作为主干网络来预测三维多孔介质渗透率的神经网络，并证明了其相比ViTs和CNNs的优势。

### 背景

Vision Mamba最近作为Vision Transformers(ViTs)的替代方案在图像分类领域受到关注。Vision Mamba的网络规模随输入图像分辨率线性增长，而ViTs则是二次增长，这使得Vision Mamba在计算和内存效率方面更具优势。此外，Vision Mamba比传统卷积神经网络(CNN)需要更少的可训练参数，因此内存效率更高。

### 目的

首次引入使用Vision Mamba作为主干网络来预测三维多孔介质渗透率的神经网络，比较Vision Mamba与ViT和CNN模型在渗透率预测多个方面的性能，并进行消融研究以评估其组件对准确性的影响。

### 方法

构建了一个使用Vision Mamba作为主干网络的神经网络来预测三维多孔介质的渗透率，并与ViT和CNN模型进行了性能比较，进行了消融研究评估组件对准确性的影响。

### 主要发现

实践证明了Vision Mamba在三维多孔介质渗透率预测方面相比ViTs和CNNs具有计算效率高、内存占用少、参数量少等优势。

### 结论

作者认为提出的框架有潜力集成到使用Vision Mamba替代ViTs的大型视觉模型中，并已公开源代码以促进可重复性并使其他研究人员能够在此基础上进行扩展。

### 翻译

Vision Mamba最近作为Vision Transformers(ViTs)的替代方案在图像分类领域受到关注。Vision Mamba的网络规模随输入图像分辨率线性增长，而ViTs则是二次增长，这一特性提高了计算和内存效率。此外，Vision Mamba比传统卷积神经网络(CNN)需要少得多的可训练参数，因此可以更节省内存。由于这些特性，我们首次引入了一个使用Vision Mamba作为主干网络来预测三维多孔介质渗透率的神经网络。我们在渗透率预测的多个方面比较了Vision Mamba与ViT和CNN模型的性能，并进行了消融研究以评估其组件对准确性的影响。我们通过实践证明了Vision Mamba在三维多孔介质渗透率预测方面相比ViTs和CNNs具有上述优势。我们公开源代码以促进可重复性，并使其他研究人员能够在此基础上进行扩展和延伸。我们认为，在Vision Mamba替代ViTs的大型视觉模型中，所提出的框架具有集成潜力。


### 论文摘要

Vision Mamba has recently received attention as an alternative to Vision Transformers (ViTs) for image classification. The network size of Vision Mamba scales linearly with input image resolution, whereas ViTs scale quadratically, a feature that improves computational and memory efficiency. Moreover, Vision Mamba requires a significantly smaller number of trainable parameters than traditional convolutional neural networks (CNNs), and thus, they can be more memory efficient. Because of these features, we introduce, for the first time, a neural network that uses Vision Mamba as its backbone for predicting the permeability of three-dimensional porous media. We compare the performance of Vision Mamba with ViT and CNN models across multiple aspects of permeability prediction and perform an ablation study to assess the effects of its components on accuracy. We demonstrate in practice the aforementioned advantages of Vision Mamba over ViTs and CNNs in the permeability prediction of three-dimensional porous media. We make the source code publicly available to facilitate reproducibility and to enable other researchers to build on and extend this work. We believe the proposed framework has the potential to be integrated into large vision models in which Vision Mamba is used instead of ViTs.

---

## 28. Unsupervised Deep Generative Models for Anomaly Detection in Neuroimaging: A Systematic Scoping Review

**论文链接:** [http://arxiv.org/abs/2510.14462v1](http://arxiv.org/abs/2510.14462v1)

**作者:** Youwan Mahé, Elise Bannier, Stéphanie Leplaideur, Elisa Fromont, Francesca Galassi

**发布时间:** 2025-10-16

### GPT解析

### 总结

这篇PRISMA指导的范围综述综合了无监督深度生成模型在神经影像学中异常检测的最新研究进展，涵盖了2018-2025年间的49项研究，表明这些模型在大局灶性病变检测和微妙异常识别方面取得了显著进展。

### 背景

无监督深度生成模型正在成为脑成像异常检测和分割的有前景方法，与需要大量体素级标注数据且仅限于已表征病理的完全监督方法不同，这些模型可以仅使用健康数据进行训练，并将异常识别为从学习到的正常脑结构中出现的偏差。

### 目的

综合关于无监督深度生成模型在神经影像学中异常检测的最新工作，包括自编码器、变分自编码器、生成对抗网络和去噪扩散模型，并比较其性能指标和架构设计选择。

### 方法

采用PRISMA指导的范围综述方法，系统检索并分析了2018-2025年间发表的49项研究，这些研究应用了各种生成模型于脑MRI和CT影像，用于检测肿瘤、中风、多发性硬化和小血管疾病等多种病理。

### 主要发现

生成模型在大局灶性病变方面取得了令人鼓舞的性能，并在处理更微妙的异常方面取得了进展；其关键优势是能够产生可解释的伪健康重建，这在注释数据稀缺的情况下（如罕见或异质性疾病）特别有价值。

### 结论

这些模型为异常检测提供了有吸引力的方向，能够实现半监督学习，支持新成像生物标志物的发现，并促进统一端到端框架内的疾病内和跨疾病偏差映射；未来工作应优先考虑解剖感知建模、基础模型开发、任务适当的评估指标和严格的临床验证。

### 翻译

无监督深度生成模型正在成为脑成像异常检测和分割的替代性有前景方法，与需要大量体素级标注数据且仅限于已表征病理的完全监督方法不同，这些模型可以仅使用健康数据进行训练，并将异常识别为从学习到的正常脑结构中出现的偏差。这篇PRISMA指导的范围综述综合了无监督深度生成模型在神经影像学中异常检测的最新工作，包括自编码器、变分自编码器、生成对抗网络和去噪扩散模型。共确定了2018-2025年间发表的49项研究，涵盖了脑MRI和较少见的CT应用，应用于肿瘤、中风、多发性硬化和小血管疾病等多种病理。报告的性能指标与架构设计选择进行了比较。在纳入的研究中，生成模型在大局灶性病变方面取得了令人鼓舞的性能，并在处理更微妙的异常方面取得了进展。生成模型的一个关键优势是它们能够产生可解释的伪健康（也称为反事实）重建，这在注释数据稀缺时（如罕见或异质性疾病）特别有价值。展望未来，这些模型为异常检测提供了有吸引力的方向，能够实现半监督学习，支持新成像生物标志物的发现，并促进统一端到端框架内的疾病内和跨疾病偏差映射。为实现临床影响，未来工作应优先考虑解剖感知建模、基础模型开发、任务适当的评估指标和严格的临床验证。


### 论文摘要

Unsupervised deep generative models are emerging as a promising alternative to supervised methods for detecting and segmenting anomalies in brain imaging. Unlike fully supervised approaches, which require large voxel-level annotated datasets and are limited to well-characterised pathologies, these models can be trained exclusively on healthy data and identify anomalies as deviations from learned normative brain structures. This PRISMA-guided scoping review synthesises recent work on unsupervised deep generative models for anomaly detection in neuroimaging, including autoencoders, variational autoencoders, generative adversarial networks, and denoising diffusion models. A total of 49 studies published between 2018 - 2025 were identified, covering applications to brain MRI and, less frequently, CT across diverse pathologies such as tumours, stroke, multiple sclerosis, and small vessel disease. Reported performance metrics are compared alongside architectural design choices. Across the included studies, generative models achieved encouraging performance for large focal lesions and demonstrated progress in addressing more subtle abnormalities. A key strength of generative models is their ability to produce interpretable pseudo-healthy (also referred to as counterfactual) reconstructions, which is particularly valuable when annotated data are scarce, as in rare or heterogeneous diseases. Looking ahead, these models offer a compelling direction for anomaly detection, enabling semi-supervised learning, supporting the discovery of novel imaging biomarkers, and facilitating within- and cross-disease deviation mapping in unified end-to-end frameworks. To realise clinical impact, future work should prioritise anatomy-aware modelling, development of foundation models, task-appropriate evaluation metrics, and rigorous clinical validation.

---

## 29. 论文ID: 2510.14438v1

**错误:** 无法读取论文信息 - [Errno 2] No such file or directory: 'storage/result/2510.14438v1.json'

---

## 30. Vision-Centric Activation and Coordination for Multimodal Large Language Models

**论文链接:** [http://arxiv.org/abs/2510.14349v1](http://arxiv.org/abs/2510.14349v1)

**作者:** Yunnan Wang, Fan Lu, Kecheng Zheng, Ziyuan Huang, Ziqiang Li, Wenjun Zeng, Xin Jin

**发布时间:** 2025-10-16

**备注:** Under Review

### GPT解析

### 总结

这篇论文提出了VaCo方法，通过视觉中心激活和多视觉基础模型的协调来优化多模态大语言模型(MLLMs)的表示，提高模型在视觉理解方面的性能。

### 背景

多模态大语言模型(MLLMs)通过整合视觉编码器的图像特征与LLMs，展现出先进的理解能力。然而，主流MLLMs仅通过文本标记的下一个标记预测进行监督，忽略了分析能力所需的关键视觉中心信息。

### 目的

解决主流MLLMs忽视关键视觉中心信息的问题，通过引入视觉中心激活和协调机制，优化MLLMs的表示，提高其视觉理解能力。

### 方法

作者提出了VaCo方法，包括：视觉判别对齐整合从多个视觉基础模型(VFMs)中提取的任务感知特征；可学习的模块化任务查询(MTQs)在多种VFMs的监督下激活特定视觉信号；视觉对齐层(VALs)整合到MLLMs中；标记网关掩码(TGM)限制多组MTQs之间的信息流，协调VFMs之间的表示冲突。

### 主要发现

大量实验表明，VaCo显著提高了不同MLLMs在各种基准测试上的性能，展示了其在视觉理解方面的卓越能力。

### 结论

VaCo通过有效整合多种视觉基础模型的特征，解决了主流MLLMs忽视视觉中心信息的问题，显著提升了模型在视觉理解任务上的表现。

### 翻译

多模态大语言模型(MLLMs)整合视觉编码器中的图像特征与LLMs，展现出先进的理解能力。然而，主流MLLMs仅通过文本标记的下一个标记预测进行监督，忽略了分析能力所需的关键视觉中心信息。为了解决这一困境，我们引入了VaCo，它通过多个视觉基础模型(VFMs)的视觉中心激活和协调来优化MLLM表示。VaCo引入视觉判别对齐来整合从VFMs中提取的任务感知特征，从而统一MLMs中文本和视觉输出的优化。具体来说，我们将可学习的模块化任务查询(MTQs)和视觉对齐层(VALs)整合到MLLMs中，在多种VFMs的监督下激活特定的视觉信号。为了协调VFMs之间的表示冲突，精心设计的标记网关掩码(TGM)限制了多组MTQs之间的信息流。大量实验证明，VaCo显著提高了不同MLLMs在各种基准测试上的性能，展示了其在视觉理解方面的卓越能力。


### 论文摘要

Multimodal large language models (MLLMs) integrate image features from visual encoders with LLMs, demonstrating advanced comprehension capabilities. However, mainstream MLLMs are solely supervised by the next-token prediction of textual tokens, neglecting critical vision-centric information essential for analytical abilities. To track this dilemma, we introduce VaCo, which optimizes MLLM representations through Vision-Centric activation and Coordination from multiple vision foundation models (VFMs). VaCo introduces visual discriminative alignment to integrate task-aware perceptual features extracted from VFMs, thereby unifying the optimization of both textual and visual outputs in MLLMs. Specifically, we incorporate the learnable Modular Task Queries (MTQs) and Visual Alignment Layers (VALs) into MLLMs, activating specific visual signals under the supervision of diverse VFMs. To coordinate representation conflicts across VFMs, the crafted Token Gateway Mask (TGM) restricts the information flow among multiple groups of MTQs. Extensive experiments demonstrate that VaCo significantly improves the performance of different MLLMs on various benchmarks, showcasing its superior capabilities in visual comprehension.

---

## 31. GauSSmart: Enhanced 3D Reconstruction through 2D Foundation Models and Geometric Filtering

**论文链接:** [http://arxiv.org/abs/2510.14270v1](http://arxiv.org/abs/2510.14270v1)

**作者:** Alexander Valverde, Brian Xu, Yuyin Zhou, Meng Xu, Hongyun Wang

**发布时间:** 2025-10-16

### GPT解析

### 总结

本文提出了一种名为GauSSmart的混合方法，通过结合2D基础模型和3D高斯飞溅重建技术，解决了Gaussian Splatting在捕捉精细细节和稀疏覆盖区域保持真实感方面的局限性。

### 背景

场景重建是计算机视觉中的核心挑战，NeRF和Gaussian Splatting等方法取得了显著进展。但Gaussian Splatting在大规模数据集上表现良好时，往往难以捕捉精细细节或在稀疏覆盖区域保持真实感，这主要是由于稀疏3D训练数据的固有局限性。

### 目的

提出GauSSmart，一种有效桥接2D基础模型和3D高斯飞溅重建的混合方法，以提升场景重建的质量和细节表现。

### 方法

集成成熟的2D计算机视觉技术，包括凸滤波和来自基础模型(如DINO)的语义特征监督，利用2D分割先验和高维特征嵌入，指导高斯飞溅的密集化和细化，改善代表性不足区域的覆盖，并保持复杂的结构细节。

### 主要发现

在三个数据集上的验证表明，GauSSmart在大多数评估场景中一致性地优于现有的高斯飞溅方法，能够更好地捕捉场景细节并提高稀疏覆盖区域的重建质量。

### 结论

混合2D-3D方法具有巨大潜力，将2D基础模型与3D重建管道的巧妙结合可以克服单独使用任何一种方法的固有局限性。

### 翻译

场景重建已成为计算机视觉中的一个核心挑战，诸如神经辐射场和高斯飞溅等方法已取得显著进展。虽然高斯飞溅在大规模数据集上表现出色，但它往往难以捕捉精细细节或在稀疏覆盖区域保持真实感，这主要是由于稀疏3D训练数据的固有局限性。在本工作中，我们提出了GauSSmart，一种有效桥接2D基础模型和3D高斯飞溅重建的混合方法。我们的方法集成了成熟的2D计算机视觉技术，包括凸滤波和来自基础模型(如DINO)的语义特征监督，以增强基于高斯的场景重建。通过利用2D分割先验和高维特征嵌入，我们的方法指导高斯飞溅的密集化和细化，改善了代表性不足区域的覆盖并保持了复杂的结构细节。我们在三个数据集上验证了我们的方法，其中GauSSmart在大多数评估场景中一致性地优于现有的高斯飞溅方法。我们的结果证明了混合2D-3D方法的巨大潜力，强调了如何将2D基础模型与3D重建管道的巧妙结合可以克服单独使用任何一种方法所固有的局限性。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决3D场景重建中细节捕捉不足和稀疏覆盖区域真实感差的问题。这个问题很重要，因为高质量的3D重建对虚拟现实、增强现实、自动驾驶等应用至关重要，而现有方法在处理细节和稀疏区域时存在局限性，限制了重建质量和技术应用范围。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者思考如何结合2D基础模型和3D重建的优势，认识到2D视觉技术（如分割和特征提取）成熟而3D方法擅长空间建模。他们借鉴了DINO等基础模型的语义特征、SAM的图像分割能力，以及凸包过滤技术，并将这些2D方法与3D高斯溅射流程巧妙融合，形成互补优势。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是利用2D基础模型的语义信息指导3D高斯溅射优化，改善点云质量并增强稀疏区域。整体流程包括：1)使用凸包过滤去除点云异常值；2)通过相机聚类选择代表性图像；3)应用SAM进行图像分割并关联3D点；4)基于分割掩码有针对性地增强点云密度；5)引入DINOv3特征嵌入损失提高语义一致性。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)凸包引导的异常值去除方法；2)感知的点云增强策略，考虑语义区域重要性；3)基于DINOv3的嵌入对齐训练损失。相比之前工作，不同之处在于：不是简单拼接2D和3D方法，而是设计真正融合框架；利用语义先验指导3D重建；点云增强考虑语义区域重要性；使用特征嵌入损失而非仅传统光度损失。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'GauSSmart通过融合2D基础模型的语义理解与3D高斯溅射的空间建模能力，有效提升了3D场景重建中的细节捕捉和稀疏区域真实感。'}


### 论文摘要

Scene reconstruction has emerged as a central challenge in computer vision, with approaches such as Neural Radiance Fields (NeRF) and Gaussian Splatting achieving remarkable progress. While Gaussian Splatting demonstrates strong performance on large-scale datasets, it often struggles to capture fine details or maintain realism in regions with sparse coverage, largely due to the inherent limitations of sparse 3D training data.   In this work, we propose GauSSmart, a hybrid method that effectively bridges 2D foundational models and 3D Gaussian Splatting reconstruction. Our approach integrates established 2D computer vision techniques, including convex filtering and semantic feature supervision from foundational models such as DINO, to enhance Gaussian-based scene reconstruction. By leveraging 2D segmentation priors and high-dimensional feature embeddings, our method guides the densification and refinement of Gaussian splats, improving coverage in underrepresented areas and preserving intricate structural details.   We validate our approach across three datasets, where GauSSmart consistently outperforms existing Gaussian Splatting in the majority of evaluated scenes. Our results demonstrate the significant potential of hybrid 2D-3D approaches, highlighting how the thoughtful combination of 2D foundational models with 3D reconstruction pipelines can overcome the limitations inherent in either approach alone.

---

## 32. Generalist vs Specialist Time Series Foundation Models: Investigating Potential Emergent Behaviors in Assessing Human Health Using PPG Signals

**论文链接:** [http://arxiv.org/abs/2510.14254v1](http://arxiv.org/abs/2510.14254v1)

**作者:** Saurabh Kataria, Yi Wu, Zhaoliang Chen, Hyunjung Gloria Kwak, Yuhao Xu, Lovely Yeswanth Panchumarthi, Ran Xiao, Jiaying Lu, Ayca Ermis, Anni Zhao, Runze Yan, Alex Federov, Zewen Liu, Xu Wu, Wei Jin, Carl Yang, Jocelyn Grunwell, Stephanie R. Brown, Amit Shah, Craig Jabaley, Tim Buchman, Sivasubramanium V Bhavani, Randall J. Lee, Xiao Hu

**发布时间:** 2025-10-16

### GPT解析

### 总结

这篇论文研究了基础模型在时间序列分析中的应用，特别是比较了专家模型和通用模型在生理信号处理（特别是PPG信号）上的性能差异。

### 背景

基础模型是大规模机器学习模型，在大规模数据上预训练后可适应各种下游任务，已广泛应用于自然语言处理和计算机视觉领域。时间序列分析领域，特别是生理信号处理，正逐渐受到关注，但大多数时间序列基础模型是专家模型，只在同类型数据上预训练和测试，如心电图、脑电图和光电容积脉搏波(PPG)。最近的工作如MOMENT尝试训练跨多个领域的通用时间序列基础模型。

### 目的

进行全面的基准测试研究，比较专家模型和通用模型的性能，特别关注PPG信号。

### 方法

通过总共51个任务组成的测试套件进行评估，包括心脏状态评估、实验室值估计和跨模态推理。在七个维度上全面评估两种模型：获胜分数、平均性能、特征质量、调优增益、性能方差、可转移性和可扩展性。这些指标共同捕捉模型在不同微调策略下的能力、适应性和效率。在完整微调场景下比较模型性能，并提供泛化、公平性、注意力可视化和训练数据选择重要性的进一步分析。

### 主要发现

在完整微调场景下，专家模型的获胜分数比通用模型高27%。

### 结论

论文提供了专家模型和通用模型在多样化下游场景中的优势和局限性的全面理解。

### 翻译

基础模型是大规模机器学习模型，在大规模数据上预训练，并可适应各种下游任务。它们已广泛应用于自然语言处理和计算机视觉任务，如GPT、BERT和CLIP等模型。现在，时间序列分析领域，特别是生理信号处理，也日益受到关注。然而，大多数时间序列基础模型是专家模型，其预训练和测试使用相同类型的数据，如心电图、脑电图和光电容积脉搏波(PPG)。最近的工作如MOMENT，使用来自多个领域（如天气、交通和电力）的数据训练通用时间序列基础模型。本文旨在进行全面的基准测试研究，比较专家模型和通用模型的性能，特别关注PPG信号。通过涵盖心脏状态评估、实验室值估计和跨模态推理的51个任务，我们在七个维度上全面评估了两种模型，包括获胜分数、平均性能、特征质量、调优增益、性能方差、可转移性和可扩展性。这些指标共同捕捉了模型在不同微调策略下的能力、适应性和效率，为它们在多样化下游场景中的优势和局限性提供了全面理解。在完整微调场景下，我们证明专家模型的获胜分数高出27%。最后，我们对泛化、公平性、注意力可视化和训练数据选择的重要性进行了进一步分析。


### 论文摘要

Foundation models are large-scale machine learning models that are pre-trained on massive amounts of data and can be adapted for various downstream tasks. They have been extensively applied to tasks in Natural Language Processing and Computer Vision with models such as GPT, BERT, and CLIP. They are now also increasingly gaining attention in time-series analysis, particularly for physiological sensing. However, most time series foundation models are specialist models - with data in pre-training and testing of the same type, such as Electrocardiogram, Electroencephalogram, and Photoplethysmogram (PPG). Recent works, such as MOMENT, train a generalist time series foundation model with data from multiple domains, such as weather, traffic, and electricity. This paper aims to conduct a comprehensive benchmarking study to compare the performance of generalist and specialist models, with a focus on PPG signals. Through an extensive suite of total 51 tasks covering cardiac state assessment, laboratory value estimation, and cross-modal inference, we comprehensively evaluate both models across seven dimensions, including win score, average performance, feature quality, tuning gain, performance variance, transferability, and scalability. These metrics jointly capture not only the models' capability but also their adaptability, robustness, and efficiency under different fine-tuning strategies, providing a holistic understanding of their strengths and limitations for diverse downstream scenarios. In a full-tuning scenario, we demonstrate that the specialist model achieves a 27% higher win score. Finally, we provide further analysis on generalization, fairness, attention visualizations, and the importance of training data choice.

---

## 33. Spectral Analysis of Molecular Kernels: When Richer Features Do Not Guarantee Better Generalization

**论文链接:** [http://arxiv.org/abs/2510.14217v1](http://arxiv.org/abs/2510.14217v1)

**作者:** Asma Jamali, Tin Sum Cheng, Rodrigo A. Vargas-Hernández

**发布时间:** 2025-10-16

**备注:** 14 pages, 5 figures, 3 tables, SI: 8 pages, 7 figures

### GPT解析

### 总结

本研究首次对QM9数据集上的核岭回归进行了全面的谱分析，研究了不同分子表示方法（分子指纹、预训练Transformer、全局和局部3D表示）在七种分子属性上的谱特性，发现更丰富的谱特征并不总能提高准确性，主要特征值捕获了最有信息量的特征。

### 背景

理解核的谱特性为泛化和表示质量提供了原则性的视角。虽然深度模型在分子属性预测中实现了最先进的准确性，但核方法因其在小数据环境下的鲁棒性和透明的理论基础而被广泛使用。然而，对分子核的系统性谱分析仍然稀缺。

### 目的

提供对QM9数据集上核岭回归的首次全面谱分析，研究不同分子表示方法在七种分子属性上的谱特性，探索谱特性与预测性能之间的关系。

### 方法

使用四种不同的谱指标测量谱丰富度，实施截断核方法探究谱与预测性能的关系，分析七种分子属性，比较分子指纹、预训练Transformer、全局和局部3D表示等不同表示方法。

### 主要发现

1) 更丰富的谱特征并不一致地提高准确性；2) 对于基于Transformer和局部3D表示，谱丰富度甚至可能与性能呈负相关；3) 在许多核中，仅保留前2%的特征值就能恢复几乎所有性能；4) 主要特征值捕获了最有信息量的特征。

### 结论

研究结果表明表示、核特征和预测性能之间存在微妙的关系，挑战了关于谱丰富度与性能关系的传统观点。这些发现对如何在数据有限的科学和实际任务中评估核方法和自监督学习方法提供了指导。

### 翻译

理解核的谱特性为泛化和表示质量提供了原则性的视角。虽然深度模型在分子属性预测中实现了最先进的准确性，但核方法因其在小数据环境下的鲁棒性和透明的理论基础而被广泛使用。尽管机器学习中核谱的研究广泛，但对分子核的系统性谱分析仍然稀缺。在这项工作中，我们首次对QM9数据集上的核岭回归进行了全面的谱分析，研究了分子指纹、预训练Transformer、全局和局部3D表示在七种分子属性上的谱特性。令人惊讶的是，通过四种不同的谱指标测量的更丰富的谱特征并不一致地提高准确性。皮尔逊相关性测试进一步表明，对于基于Transformer和局部3D表示，谱丰富度甚至可能与性能呈负相关。我们还实现了截断核来探究谱与预测性能之间的关系：在许多核中，仅保留前2%的特征值就能恢复几乎所有性能，这表明主要特征值捕获了最有信息量的特征。我们的结果挑战了'更丰富的谱产生更好的泛化'这一常见启发式方法，并突出了表示、核特征和预测性能之间的微妙关系。除了分子属性预测外，这些发现还指导了如何在数据有限的科学和实际任务中评估核方法和自监督学习方法。


### 论文摘要

Understanding the spectral properties of kernels offers a principled perspective on generalization and representation quality. While deep models achieve state-of-the-art accuracy in molecular property prediction, kernel methods remain widely used for their robustness in low-data regimes and transparent theoretical grounding. Despite extensive studies of kernel spectra in machine learning, systematic spectral analyses of molecular kernels are scarce. In this work, we provide the first comprehensive spectral analysis of kernel ridge regression on the QM9 dataset, molecular fingerprint, pretrained transformer-based, global and local 3D representations across seven molecular properties. Surprisingly, richer spectral features, measured by four different spectral metrics, do not consistently improve accuracy. Pearson correlation tests further reveal that for transformer-based and local 3D representations, spectral richness can even have a negative correlation with performance. We also implement truncated kernels to probe the relationship between spectrum and predictive performance: in many kernels, retaining only the top 2% of eigenvalues recovers nearly all performance, indicating that the leading eigenvalues capture the most informative features. Our results challenge the common heuristic that "richer spectra yield better generalization" and highlight nuanced relationships between representation, kernel features, and predictive performance. Beyond molecular property prediction, these findings inform how kernel and self-supervised learning methods are evaluated in data-limited scientific and real-world tasks.

---

## 34. ARM-FM: Automated Reward Machines via Foundation Models for Compositional Reinforcement Learning

**论文链接:** [http://arxiv.org/abs/2510.14176v1](http://arxiv.org/abs/2510.14176v1)

**作者:** Roger Creus Castanyer, Faisal Mohamed, Pablo Samuel Castro, Cyrus Neary, Glen Berseth

**发布时间:** 2025-10-16

### GPT解析

### 总结

ARM-FM是一种利用基础模型高级推理能力的框架，用于强化学习中自动化、组合式的奖励设计，解决了强化学习算法对奖励函数设定敏感的核心挑战。

### 背景

强化学习算法对奖励函数的设定高度敏感，这仍然是限制其广泛应用的核心挑战。

### 目的

提出ARM-FM框架，实现强化学习中自动化、组合式的奖励设计，利用基础模型的高级推理能力来自动构建奖励机。

### 方法

使用奖励机(RMs)作为强化学习目标设定的机制，通过基础模型自动构建奖励机；将语言嵌入与每个奖励机自动机状态相关联以实现跨任务泛化；在多样化挑战性环境中评估框架效果。

### 主要发现

ARM-FM框架在多样化的挑战性环境中展现出有效性，包括实现零样本泛化的能力；基础模型能够从自然语言规范自动生成奖励机；结构化的奖励机形式化方法能实现有效的任务分解。

### 结论

基础模型与奖励机的结构化形式化方法相结合，能够实现有效的自动化奖励设计，促进强化学习在更广泛领域的应用。

### 翻译

强化学习(RL)算法对奖励函数的设定高度敏感，这仍然是限制其广泛适用性的核心挑战。我们提出了ARM-FM：基于基础模型的自动奖励机，这是一个用于强化学习中自动化、组合式奖励设计的框架，利用了基础模型(FMs)的高级推理能力。奖励机(RMs)——一种基于自动机的奖励规范形式化方法——被用作强化学习目标设定的机制，并通过基础模型的使用自动构建。奖励机的结构化形式化方法能够实现有效的任务分解，而基础模型的使用则允许用自然语言进行目标规范。具体而言，我们(i)使用基础模型从自然语言规范自动生成奖励机；(ii)将语言嵌入与每个奖励机自动机状态相关联，以实现跨任务泛化；(iii)在一系列多样化的挑战性环境中提供了ARM-FM有效性的实证证据，包括零样本泛化的证据。


### 论文摘要

Reinforcement learning (RL) algorithms are highly sensitive to reward function specification, which remains a central challenge limiting their broad applicability. We present ARM-FM: Automated Reward Machines via Foundation Models, a framework for automated, compositional reward design in RL that leverages the high-level reasoning capabilities of foundation models (FMs). Reward machines (RMs) -- an automata-based formalism for reward specification -- are used as the mechanism for RL objective specification, and are automatically constructed via the use of FMs. The structured formalism of RMs yields effective task decompositions, while the use of FMs enables objective specifications in natural language. Concretely, we (i) use FMs to automatically generate RMs from natural language specifications; (ii) associate language embeddings with each RM automata-state to enable generalization across tasks; and (iii) provide empirical evidence of ARM-FM's effectiveness in a diverse suite of challenging environments, including evidence of zero-shot generalization.

---

## 35. Formalizing the Safety, Security, and Functional Properties of Agentic AI Systems

**论文链接:** [http://arxiv.org/abs/2510.14133v1](http://arxiv.org/abs/2510.14133v1)

**作者:** Edoardo Allegrini, Ananth Shreekumar, Z. Berkay Celik

**发布时间:** 2025-10-15

### GPT解析

### 总结

论文提出了一种代理AI系统的统一建模框架，由主机代理模型和任务生命周期模型组成，解决了当前代理间通信生态系统碎片化的问题，为多AI代理系统提供了形式化验证基础。

### 背景

代理AI系统利用多个自主代理和大语言模型解决复杂多步骤任务，在高风险应用中安全性和功能性至关重要。当前代理间通信生态系统碎片化，各种协议被孤立分析，造成语义鸿沟，阻碍系统属性严格分析并引入架构不协调等风险。

### 目的

解决代理AI系统中因通信碎片化导致的语义鸿沟问题，提供统一语义框架实现多AI代理系统行为的推理，支持系统化分析、设计和部署正确、可靠、稳健的代理AI系统。

### 方法

引入由两个基础模型组成的框架：主机代理模型（正式化顶层实体与用户交互、任务分解和执行协调）和任务生命周期模型（详细说明子任务状态和转换）。基于此框架定义31个属性（主机代理17个，任务生命周期14个），分为活性、安全性、完整性和公平性四类，用时态逻辑表达以实现形式化验证。

### 主要发现

两个基础模型共同为多AI代理系统行为推理提供统一语义框架，定义的属性能实现系统行为形式化验证，检测协调边缘情况，防止死锁和安全漏洞。

### 结论

引入了第一个严格基础、领域无关的框架，用于代理AI系统的系统化分析、设计和部署，确保系统正确性、可靠性和稳健性。

### 翻译

代理AI系统，即利用多个自主代理和大语言模型的系统，正被越来越多地用于解决复杂的多步骤任务。这些系统的安全性、安全性和功能性至关重要，特别是在高风险应用中。然而，当前代理间通信生态系统是碎片化的，诸如用于工具访问的模型上下文协议和用于协调的代理到代理等协议被孤立地分析。这种碎片化造成了语义鸿沟，阻碍了对系统属性的严格分析，并引入了架构不协调和可利用的协调问题等风险。为应对这些挑战，我们引入了一个由两个基础模型组成的代理AI系统建模框架。第一个是主机代理模型，它正式化与用户交互、分解任务并通过利用外部代理和工具协调执行的最高级别实体。第二个是任务生命周期模型，它详细说明从创建到完成的各个子任务的状态和转换，提供细粒度的任务管理和错误处理视图。这两个模型共同为多AI代理系统行为推理提供了统一的语义框架。基于此框架，我们为主机代理定义了17个属性，为任务生命周期定义了14个属性，分为活性、安全性、完整性和公平性四类。用时态逻辑表达的这些属性，能够实现系统行为的正式验证，检测协调边缘情况，并防止死锁和安全漏洞。通过这项工作，我们引入了第一个严格基础、领域无关的框架，用于代理AI系统的系统化分析、设计和部署，以确保正确、可靠和稳健的系统。


### 论文摘要

Agentic AI systems, which leverage multiple autonomous agents and Large Language Models (LLMs), are increasingly used to address complex, multi-step tasks. The safety, security, and functionality of these systems are critical, especially in high-stakes applications. However, the current ecosystem of inter-agent communication is fragmented, with protocols such as the Model Context Protocol (MCP) for tool access and the Agent-to-Agent (A2A) protocol for coordination being analyzed in isolation. This fragmentation creates a semantic gap that prevents the rigorous analysis of system properties and introduces risks such as architectural misalignment and exploitable coordination issues. To address these challenges, we introduce a modeling framework for agentic AI systems composed of two foundational models. The first, the host agent model, formalizes the top-level entity that interacts with the user, decomposes tasks, and orchestrates their execution by leveraging external agents and tools. The second, the task lifecycle model, details the states and transitions of individual sub-tasks from creation to completion, providing a fine-grained view of task management and error handling. Together, these models provide a unified semantic framework for reasoning about the behavior of multi-AI agent systems. Grounded in this framework, we define 17 properties for the host agent and 14 for the task lifecycle, categorized into liveness, safety, completeness, and fairness. Expressed in temporal logic, these properties enable formal verification of system behavior, detection of coordination edge cases, and prevention of deadlocks and security vulnerabilities. Through this effort, we introduce the first rigorously grounded, domain-agnostic framework for the systematic analysis, design, and deployment of correct, reliable, and robust agentic AI systems.

---

## 36. Shadow Molecular Dynamics for Flexible Multipole Models

**论文链接:** [http://arxiv.org/abs/2510.14132v1](http://arxiv.org/abs/2510.14132v1)

**作者:** Rae A. Corrigan Grove, Robert Stanton, Michael E. Wall, Anders M. N. Niklasson

**发布时间:** 2025-10-15

### GPT解析

### 总结

该研究将阴影分子动力学扩展到柔性多极模型，处理长程静电相互作用，提供稳定高效的原子模拟框架。

### 背景

阴影分子动力学是处理具有长程静电相互作用的柔性电荷模型的高效稳定原子模拟框架，但之前实现仅限于原子单极电荷分布。

### 目的

扩展阴影分子动力学方法以支持柔性多极模型，实现更准确的原子相互作用模拟。

### 方法

推导阴影能量函数、势能和力项的详细表达式，明确包含单极-单极、偶极-单极和偶极-偶极相互作用；将原子单极和偶极视为扩展动力学变量；提出单极固定而偶极柔性的分子动力学方案。

### 主要发现

引入额外偶极自由度保留了仅单极阴影分子动力学模拟的稳定性和准确性；扩展的阴影动力学为涉及柔性多极长程相互作用的稳定、计算高效且多功能的分子动力学模拟提供了框架。

### 结论

该方法与现代人工智能和机器学习技术结合特别有意义，有助于开发可转移的高精度原子相互作用表示，适用于各种分子系统。

### 翻译

阴影分子动力学为具有长程静电相互作用的柔性电荷模型提供了一种高效稳定的原子模拟框架。虽然之前的实现仅限于原子单极电荷分布，但我们将这种方法扩展到了柔性多极模型。我们推导了阴影能量函数、势能和力项的详细表达式，明确包含了单极-单极、偶极-单极和偶极-偶极相互作用。在我们的公式中，原子单极和原子偶极都被视为扩展的动力学变量，与核自由度的传播一起处理。我们证明引入额外的偶极自由度保留了之前在仅单极阴影分子动力学模拟中看到的稳定性和准确性。此外，我们提出了一种阴影分子动力学方案，其中单极电荷保持固定，而偶极保持柔性。我们的扩展阴影动力学为涉及柔性多极之间长程相互作用的稳定、计算高效且多功能的分子动力学模拟提供了框架。这与现代人工智能和机器学习技术结合特别有意义，这些技术越来越多地用于开发原子模拟的物理信息驱动和数据驱动的基础模型。这些模型旨在提供可转移的高精度原子相互作用表示，适用于各种分子系统，这需要准确处理长程电荷相互作用。


### 论文摘要

Shadow molecular dynamics provide an efficient and stable atomistic simulation framework for flexible charge models with long-range electrostatic interactions. While previous implementations have been limited to atomic monopole charge distributions, we extend this approach to flexible multipole models. We derive detailed expressions for the shadow energy functions, potentials, and force terms, explicitly incorporating monopole-monopole, dipole-monopole, and dipole-dipole interactions. In our formulation, both atomic monopoles and atomic dipoles are treated as extended dynamical variables alongside the propagation of the nuclear degrees of freedom. We demonstrate that introducing the additional dipole degrees of freedom preserves the stability and accuracy previously seen in monopole-only shadow molecular dynamics simulations. Additionally, we present a shadow molecular dynamics scheme where the monopole charges are held fixed while the dipoles remain flexible. Our extended shadow dynamics provide a framework for stable, computationally efficient, and versatile molecular dynamics simulations involving long-range interactions between flexible multipoles. This is of particular interest in combination with modern artificial intelligence and machine learning techniques, which are increasingly used to develop physics-informed and data-driven foundation models for atomistic simulations. These models aim to provide transferable, high-accuracy representations of atomic interactions that are applicable across diverse sets of molecular systems, which requires accurate treatment of long-range charge interactions.

---

## 37. Exploratory Causal Inference in SAEnce

**论文链接:** [http://arxiv.org/abs/2510.14073v1](http://arxiv.org/abs/2510.14073v1)

**作者:** Tommaso Mencattini, Riccardo Cadei, Francesco Locatello

**发布时间:** 2025-10-15

### GPT解析

### 总结

研究提出了一种名为Neural Effect Search的新方法，可以直接从数据中发现未知的因果效应，解决了传统随机对照试验的局限性。

### 背景

随机对照试验是科学的重要支柱，但它们依赖于手工制作的假设和昂贵的分析。这些限制阻碍了大规模因果效应估计，可能导致依赖于流行但不完整的假设。

### 目的

直接从数据中发现治疗的未知效应。

### 方法

使用预训练的基础模型将试验中的非结构化数据转换为有意义的表示，通过稀疏自编码器解释这些表示，并引入Neural Effect Search这一新颖的递归过程，通过渐进分层解决多重测试问题和效应纠缠问题。

### 主要发现

在半合成实验中评估了算法的稳健性，并在实验生态学背景下展示了在真实世界科学试验中首次成功的无监督因果效应识别。

### 结论

Neural Effect Search方法成功解决了在神经水平发现显著因果效应的挑战。

### 翻译

随机对照试验是科学的重要支柱；然而，它们依赖于手工制作的假设和昂贵的分析。这些限制阻碍了大规模因果效应估计，可能导致依赖于流行但不完整的假设。我们提出直接从数据中发现治疗的未知效应。为此，我们通过预训练的基础模型将试验中的非结构化数据转换为有意义的表示，并通过稀疏自编码器解释它们。然而，由于多重测试问题和效应纠缠，在神经水平发现显著的因果效应并不简单。为了解决这些挑战，我们引入了Neural Effect Search，这是一种新颖的递归过程，通过渐进分层解决了这两个问题。在半合成实验中评估了我们算法的稳健性后，我们在实验生态学的背景下展示了在真实世界科学试验中首次成功的无监督因果效应识别。


### 论文摘要

Randomized Controlled Trials are one of the pillars of science; nevertheless, they rely on hand-crafted hypotheses and expensive analysis. Such constraints prevent causal effect estimation at scale, potentially anchoring on popular yet incomplete hypotheses. We propose to discover the unknown effects of a treatment directly from data. For this, we turn unstructured data from a trial into meaningful representations via pretrained foundation models and interpret them via a sparse autoencoder. However, discovering significant causal effects at the neural level is not trivial due to multiple-testing issues and effects entanglement. To address these challenges, we introduce Neural Effect Search, a novel recursive procedure solving both issues by progressive stratification. After assessing the robustness of our algorithm on semi-synthetic experiments, we showcase, in the context of experimental ecology, the first successful unsupervised causal effect identification on a real-world scientific trial.

---

## 38. Context-Selective State Space Models: Feedback is All You Need

**论文链接:** [http://arxiv.org/abs/2510.14027v1](http://arxiv.org/abs/2510.14027v1)

**作者:** Riccardo Zattra, Giacomo Baggio, Umberto Casti, Augusto Ferrante, Francesco Ticozzi

**发布时间:** 2025-10-15

### GPT解析

### 总结

本文提出了COFFEE模型，一种新颖的时变状态空间模型，通过状态反馈实现上下文相关的选择性，有效捕获长距离依赖关系，并在多项任务上取得了优于现有S6模型的结果。

### 背景

Transformers模型基于注意力机制，是大多数基础模型的骨干，但它们具有二次复杂度，并且在处理输入序列中的长距离依赖关系时存在困难。状态空间模型(SSMs)提供了一种高效的替代方案，其中S6模块在长序列基准测试上取得了最先进的结果。

### 目的

开发一种能够有效处理长距离依赖关系的高效序列模型，解决Transformers模型的二次复杂度问题，并超越现有状态空间模型的性能。

### 方法

提出COFFEE(COntext From FEEdback)模型，一种新颖的时变SSM，结合状态反馈以实现上下文相关的选择性。与S6不同，COFFEE从内部状态计算选择性，该状态作为序列历史的紧凑表示，使模型能够根据积累的上下文调节其动态。此外，采用高效的模型参数化方法消除冗余，实现更紧凑和可训练的公式。

### 主要发现

在归纳头任务上，COFFEE与S6相比，使用少两个数量级的参数和训练序列实现了接近完美的准确性；在MNIST上，仅用3585个参数就达到了97%的准确率，大大优于S6在相同架构上的表现。

### 结论

状态反馈是构建可扩展和高效序列模型的关键机制，COFFEE模型通过结合状态反馈和高效参数化，显著提升了序列建模能力，特别是在处理长距离依赖关系方面。

### 翻译

Transformers模型由注意力机制驱动，是大多数基础模型的骨干，但它们受二次复杂度的困扰，并且在处理输入序列中的长距离依赖关系时存在困难。最近的研究表明，状态空间模型(SSMs)提供了一种高效的替代方案，其中S6模块作为Mamba架构的核心，在长序列基准测试上取得了最先进的结果。在本文中，我们介绍了COFFEE(COntext From FEEdback)模型，一种新颖的时变SSM，它结合了状态反馈以实现上下文相关的选择性，同时仍允许并行实现。而S6的选择性机制仅依赖于当前输入，COFFEE从内部状态计算选择性，该状态作为序列历史的紧凑表示。这种转变使模型能够根据积累的上下文调节其动态，提高其捕获长距离依赖关系的能力。除了状态反馈外，我们还采用了一种高效的模型参数化方法，消除了S6中存在的冗余，导致更紧凑和可训练的公式。在归纳头任务上，COFFEE与S6相比，使用少两个数量级的参数和训练序列实现了接近完美的准确性。在MNIST上，COFFEE在相同架构上大大优于S6，仅用3585个参数就达到了97%的准确率。这些结果展示了状态反馈作为构建可扩展和高效序列模型的关键机制的作用。


### 论文摘要

Transformers, powered by the attention mechanism, are the backbone of most foundation models, yet they suffer from quadratic complexity and difficulties in dealing with long-range dependencies in the input sequence. Recent work has shown that state space models (SSMs) provide an efficient alternative, with the S6 module at the core of the Mamba architecture achieving state-of-the-art results on long-sequence benchmarks. In this paper, we introduce the COFFEE (COntext From FEEdback) model, a novel time-varying SSM that incorporates state feedback to enable context-dependent selectivity, while still allowing for parallel implementation. Whereas the selectivity mechanism of S6 only depends on the current input, COFFEE computes it from the internal state, which serves as a compact representation of the sequence history. This shift allows the model to regulate its dynamics based on accumulated context, improving its ability to capture long-range dependencies. In addition to state feedback, we employ an efficient model parametrization that removes redundancies present in S6 and leads to a more compact and trainable formulation. On the induction head task, COFFEE achieves near-perfect accuracy with two orders of magnitude fewer parameters and training sequences compared to S6. On MNIST, COFFEE largely outperforms S6 within the same architecture, reaching 97% accuracy with only 3585 parameters. These results showcase the role of state feedback as a key mechanism for building scalable and efficient sequence models.

---

## 39. NExT-OMNI: Towards Any-to-Any Omnimodal Foundation Models with Discrete Flow Matching

**论文链接:** [http://arxiv.org/abs/2510.13721v2](http://arxiv.org/abs/2510.13721v2)

**作者:** Run Luo, Xiaobo Xia, Lu Wang, Longze Chen, Renke Shan, Jing Luo, Min Yang, Tat-Seng Chua

**发布时间:** 2025-10-15

### GPT解析

### 总结

NExT-OMNI是一个开源的全模态基础模型，通过离散流范式实现统一建模，支持任何到任何的跨模态生成和多轮交互，克服了现有自回归架构的局限性。

### 背景

下一代多模态基础模型将成为人工通用智能系统的核心，但现有多模态模型受限于自回归架构，无法平衡整合理解与生成能力。混合和解耦策略虽被探索，但其冗余设计限制了在广泛场景如跨模态检索中的应用。

### 目的

引入NExT-OMNI，一个开源的全模态基础模型，通过离散流范式实现统一建模，支持任何到任何的理解和生成，并扩展应用场景。

### 方法

利用度量诱导的概率路径和动力学最优速度，原生支持任何到任何的理解和生成，增强响应效率；通过简洁的统一表示而非任务解耦设计实现更广泛应用；在大规模交错文本、图像、视频和音频数据上训练。

### 主要发现

NExT-OMNI在多模态生成和理解基准测试中具有竞争力，在多模态交互和跨模态检索方面优于之前的统一模型，展现了其作为下一代多模态基础模型的架构优势。

### 结论

发布训练细节、数据协议，并开源代码和模型检查点，以促进多模态基础模型领域的进一步研究和发展。

### 翻译

能够进行任何到任何跨模态生成和多轮交互的下一代多模态基础模型将成为人工通用智能系统的核心组成部分，在人机交互中发挥关键作用。然而，大多数现有多模态模型仍受限于自回归架构，其固有局限性阻碍了理解与生成能力的平衡整合。虽然混合和解耦策略已被探索用于在统一框架内分别解决这些问题，但它们的冗余、非集成设计限制了它们在更广泛场景（如跨模态检索）中的适用性。在这项工作中，我们引入了NExT-OMNI，一个开源的全模态基础模型，通过离散流范式实现统一建模。通过利用度量诱导的概率路径和动力学最优速度，NExT-OMNI原生支持任何到任何的理解和生成，同时通过简洁的统一表示而非任务解耦设计，实现更广泛的应用场景，增强响应效率。在大规模交错文本、图像、视频和音频数据上训练后，NExT-OMNI在多模态生成和理解基准测试中具有竞争力，同时在多模态交互和跨模态检索方面优于之前的统一模型，凸显了其作为下一代多模态基础模型的架构优势。为进一步推进研究，我们发布了训练细节、数据协议，并开源了代码和模型检查点。


### 论文摘要

Next-generation multimodal foundation models capable of any-to-any cross-modal generation and multi-turn interaction will serve as core components of artificial general intelligence systems, playing a pivotal role in human-machine interaction. However, most existing multimodal models remain constrained by autoregressive architectures, whose inherent limitations prevent a balanced integration of understanding and generation capabilities. Although hybrid and decoupling strategies have been explored to address these tasks within unified frameworks separately, their redundant, non-integrated designs limit their applicability to broader scenarios, such as cross-modal retrieval. In this work, we introduce NExT-OMNI, an open-source omnimodal foundation model that achieves unified modeling through discrete flow paradigms. By leveraging metric-induced probability paths and kinetic optimal velocities, NExT-OMNI natively supports any-to-any understanding and generation with enhanced response efficiency, while enabling broader application scenarios through concise unified representations rather than task-decoupled designs. Trained on large-scale interleaved text, image, video, and audio data, NExT-OMNI delivers competitive performance on multimodal generation and understanding benchmarks, while outperforming prior unified models in multi-turn multimodal interaction and cross-modal retrieval, highlighting its architectural advantages as a next-generation multimodal foundation model. To advance further research, we release training details, data protocols, and open-source both the code and model checkpoints.

---

## 40. Knowledge Reasoning Language Model: Unifying Knowledge and Language for Inductive Knowledge Graph Reasoning

**论文链接:** [http://arxiv.org/abs/2510.13909v1](http://arxiv.org/abs/2510.13909v1)

**作者:** Xingrui Zhuo, Jiapu Wang, Gongqing Wu, Zhongyuan Wang, Jichen Zhang, Shirui Pan, Xindong Wu

**发布时间:** 2025-10-15

### GPT解析

### 总结

本文提出了一种知识推理语言模型(KRLM)，用于解决归纳知识图谱推理中LLM知识与KG上下文协调的问题。通过设计KRL指令格式、KRL分词器、KRL注意力层和结构感知的下一个实体预测器，模型能够在KGR过程中实现LLM知识与KG上下文的统一协调，有效约束LLM的生成幻觉，提高推理结果的可信度。

### 背景

归纳知识图谱推理旨在发现包含未知实体和关系的开放域知识图谱中的事实，这给KGR模型在理解不确定的KG组件方面带来了挑战。现有研究提出了知识图谱基础模型来处理这种不确定性，而大型语言模型在开放域知识推理方面展示了强大能力。最新的研究集中在基于LLM的KGFMs上，这些模型整合了LLM知识与KG上下文进行归纳KGR。

### 目的

解决现有基于LLM的KGR方法中LLM知识被稀疏KG上下文掩盖导致知识扭曲的问题，以及难以完全约束LLM生成幻觉的问题，提出一个知识推理语言模型(KRLM)，在KGR过程中实现LLM知识与KG上下文的统一协调。

### 方法

设计了一种知识推理语言(KRL)指令格式和KRL分词器，以对齐LLM知识与KG表示；提出了一种KRL注意力层，通过动态知识记忆机制协调内在的LLM知识与额外的KG上下文；提出了一种结构感知的下一个实体预测器，将推理结果严格限制在可信的知识域内。

### 主要发现

在25个真实世界的归纳KGR数据集上进行了广泛的实验，结果表明所提出的KRLM在零样本推理和微调场景下都具有显著的优越性。

### 结论

KRLM模型有效地解决了LLM知识与KG上下文协调的问题，通过结构感知的下一个实体预测器提高了推理结果的可信度，在多个数据集上表现优异，证明了其有效性。

### 翻译

归纳知识图谱推理旨在发现包含未知实体和关系的开放域知识图谱中的事实，这给KGR模型在理解不确定的KG组件方面带来了挑战。现有研究提出了知识图谱基础模型，这些模型学习跨知识图谱的结构不变性来处理这种不确定性。最近，大型语言模型在开放域知识推理方面展示了强大的能力。因此，最新的研究集中在基于LLM的知识图谱基础模型上，这些模型整合了LLM知识与KG上下文进行归纳KGR。然而，LLM的内在知识可能被稀疏的KG上下文掩盖，导致LLM知识扭曲，这可能对模型推理造成不可逆的损害。此外，现有的基于LLM的KGR方法仍然难以完全约束LLM中的生成幻觉，严重限制了推理结果的可信度。为解决这些局限性，我们提出了一种知识推理语言模型(KRLM)，在KGR过程中实现LLM知识与KG上下文的统一协调。具体来说，我们设计了一种知识推理语言(KRL)指令格式和KRL分词器，以对齐LLM知识与KG表示。然后，我们提出了一种KRL注意力层，通过动态知识记忆机制协调内在的LLM知识与额外的KG上下文。最后，提出了一种结构感知的下一个实体预测器，将推理结果严格限制在可信的知识域内。在25个真实世界的归纳KGR数据集上的广泛实验结果表明，所提出的KRLM在零样本推理和微调场景下都具有显著的优越性。


### 论文摘要

Inductive Knowledge Graph Reasoning (KGR) aims to discover facts in open-domain KGs containing unknown entities and relations, which poses a challenge for KGR models in comprehending uncertain KG components. Existing studies have proposed Knowledge Graph Foundation Models (KGFMs) that learn structural invariances across KGs to handle this uncertainty. Recently, Large Language Models (LLMs) have demonstrated strong capabilities for open-domain knowledge reasoning. As a result, the latest research has focused on LLM-based KGFMs that integrate LLM knowledge with KG context for inductive KGR. However, the intrinsic knowledge of LLMs may be overshadowed by sparse KG context, leading to LLM knowledge distortion, which can cause irreversible damage to model reasoning. Moreover, existing LLM-based KGR methods still struggle to fully constrain generative hallucinations in LLMs, severely limiting the credibility of reasoning results. To address these limitations, we propose a Knowledge Reasoning Language Model (KRLM) that achieves unified coordination between LLM knowledge and KG context throughout the KGR process. Specifically, we design a Knowledge Reasoning Language (KRL) instruction format and a KRL tokenizer to align LLM knowledge with KG representations. Then, we propose a KRL attention layer that coordinates intrinsic LLM knowledge with additional KG context through a dynamic knowledge memory mechanism. Finally, a structure-aware next-entity predictor is proposed, which strictly constrains the reasoning results within a trustworthy knowledge domain. Extensive experimental results on 25 real-world inductive KGR datasets demonstrate the significant superiority of the proposed KRLM\footnote{Our source codes are available at https://anonymous.4open.science/r/KRLM-EA36 in both zero-shot reasoning and fine-tuning scenarios.

---

## 41. Rethinking Hebbian Principle: Low-Dimensional Structural Projection for Unsupervised Learning

**论文链接:** [http://arxiv.org/abs/2510.14810v1](http://arxiv.org/abs/2510.14810v1)

**作者:** Shikuang Deng, Jiayuan Zhang, Yuhang Wu, Ting Chen, Shi Gu

**发布时间:** 2025-10-16

### GPT解析

### 总结

论文提出了SPHeRe（结构投影Hebbian表示）方法，一种新型无监督学习技术，通过整合正交性和结构信息保留解决了传统Hebbian学习在机器学习中的局限性，在多个任务中取得了优异表现。

### 背景

Hebbian学习是一种描述神经元通过重复刺激调整连接的生物原理，但在机器学习应用中存在连接更新无约束和缺乏反馈中介考虑等问题，限制了其在复杂网络架构和任务中的有效扩展。

### 目的

开发一种能够克服传统Hebbian学习局限性的无监督学习方法，使其能够有效扩展到复杂网络架构和任务中。

### 方法

SPHeRe通过局部的辅助非线性块整合正交性和结构信息保留，结构信息保留的损失通过辅助轻量级投影反向传播到输入（充当反馈中介），正交性约束则确保更新幅度的有界性。

### 主要发现

SPHeRe在CIFAR-10、CIFAR-100和Tiny-ImageNet等标准图像分类基准测试中达到无监督突触可塑性方法的最新性能；在持续学习和迁移学习场景中表现有效；图像重建任务证明了提取特征的鲁棒性和泛化能力。

### 结论

该研究证明了Hebbian无监督学习规则在现代深度学习框架中的竞争力和潜力，展示了不依赖严格反向传播的高效且受生物启发的学习算法的可能性，代码已在GitHub上公开。

### 翻译

Hebbian学习是一种生物原理，直观地描述了神经元如何通过重复刺激来调整其连接。然而，当应用于机器学习时，由于连接更新的无约束性和缺乏对反馈中介的考虑，它存在严重问题。这些缺点限制了其在复杂网络架构和任务中的有效扩展。为此，我们在此引入结构投影Hebbian表示（SPHeRe），一种新型无监督学习方法，它通过一个局部的辅助非线性块整合了正交性和结构信息保留。结构信息保留的损失通过一个辅助的轻量级投影反向传播到输入，这个投影在概念上充当反馈中介，而正交性约束则考虑了更新幅度的有界性。大量实验结果表明，SPHeRe在CIFAR-10、CIFAR-100和Tiny-ImageNet等标准图像分类基准测试的无监督突触可塑性方法中达到了最先进性能。此外，该方法在持续学习和迁移学习场景中表现出强大的有效性，图像重建任务显示了所提取特征的鲁棒性和泛化能力。这项工作证明了Hebbian无监督学习规则在现代深度学习框架中的竞争力和潜力，展示了不依赖于严格反向传播的高效且受生物启发的学习算法的可能性。我们的代码可在https://github.com/brain-intelligence-lab/SPHeRe获取。


### 论文摘要

Hebbian learning is a biological principle that intuitively describes how neurons adapt their connections through repeated stimuli. However, when applied to machine learning, it suffers serious issues due to the unconstrained updates of the connections and the lack of accounting for feedback mediation. Such shortcomings limit its effective scaling to complex network architectures and tasks. To this end, here we introduce the Structural Projection Hebbian Representation (SPHeRe), a novel unsupervised learning method that integrates orthogonality and structural information preservation through a local auxiliary nonlinear block. The loss for structural information preservation backpropagates to the input through an auxiliary lightweight projection that conceptually serves as feedback mediation while the orthogonality constraints account for the boundedness of updating magnitude. Extensive experimental results show that SPHeRe achieves SOTA performance among unsupervised synaptic plasticity approaches on standard image classification benchmarks, including CIFAR-10, CIFAR-100, and Tiny-ImageNet. Furthermore, the method exhibits strong effectiveness in continual learning and transfer learning scenarios, and image reconstruction tasks show the robustness and generalizability of the extracted features. This work demonstrates the competitiveness and potential of Hebbian unsupervised learning rules within modern deep learning frameworks, demonstrating the possibility of efficient and biologically inspired learning algorithms without the strong dependence on strict backpropagation. Our code is available at https://github.com/brain-intelligence-lab/SPHeRe.

---

## 42. Unsupervised Learning to Recognize Quantum Phases of Matter

**论文链接:** [http://arxiv.org/abs/2510.14742v1](http://arxiv.org/abs/2510.14742v1)

**作者:** Mehran Khosrojerdi, Alessandro Cuccoli, Paola Verrucchi, Leonardo Banchi

**发布时间:** 2025-10-16

**备注:** 10 pages, 6 figures

### GPT解析

### 总结

本文提出使用无监督学习方法来确定多体系统的量子相图，该方法能够自主识别并可能揭示量子物质的新相位。

### 背景

将多体系统的量子相图绘制视为学习问题，需要根据某种分类标准对其基态进行标记以定义不同的相。

### 目的

采用无监督学习方法来确定多体系统的量子相图，算法无需访问任何预先标记的状态。

### 方法

算法直接处理量子态，基于量子态之间的保真度相似性标准对基态配置进行分组。使用基于谱聚类的无监督学习算法，并结合'轮廓'和'肘部'方法来确定相位的最佳数量。

### 主要发现

通过两个具体的自旋-1/2链进行基准测试，发现基于谱聚类的无监督学习算法能够准确重现相图。

### 结论

无监督学习可以自主识别并可能揭示量子物质的新相位，为量子相图的确定提供了新方法。

### 翻译

在哈密顿量参数空间中绘制多体系统的量子相图可以被视为一个学习问题，这需要根据定义相位的某种分类标准来标记相应的基态。在本工作中，我们采用无监督学习方法，其中算法无法访问任何预先标记的状态，作为确定多体系统量子相图的一种工具。该算法直接处理量子态：给定不同哈密顿量参数的基态配置，该过程基于量子态之间保真度的相似性标准揭示了对它们进行分组的最重要的方式，这种标准即使通过实验也容易估计。我们使用两个特定的自旋-1/2链来基准测试我们的方法，其状态通过张量网络技术确定。我们发现，基于谱聚类的无监督学习算法，结合用于确定相位最佳数量的'轮廓'和'肘部'方法，可以准确重现相图。我们的结果表明，无监督学习如何能够自主识别并可能揭示量子物质的新相位。


### 论文摘要

Drawing the quantum phase diagram of a many-body system in the parameter space of its Hamiltonian can be seen as a learning problem, which implies labelling the corresponding ground states according to some classification criterium that defines the phases. In this work we adopt unsupervised learning, where the algorithm has no access to any priorly labeled states, as a tool for determining quantum phase diagrams of many-body systems. The algorithm directly works with quantum states: given the ground-state configurations for different values of the Hamiltonian parameters, the process uncovers the most significant way of grouping them based on a similarity criterion that refers to the fidelity between quantum states, that can be easily estimated, even experimentally. We benchmark our method with two specific spin-$\frac{1}{2}$ chains, with states determined via tensor network techniques. We find that unsupervised learning algorithms based on spectral clustering, combined with ``silhouette'' and ``elbow'' methods for determining the optimal number of phases, can accurately reproduce the phase diagrams. Our results show how unsupervised learning can autonomously recognize and possibly unveil novel phases of quantum matter.

---

## 43. Evaluating Policy Effects under Network Interference without Network Information: A Transfer Learning Approach

**论文链接:** [http://arxiv.org/abs/2510.14415v1](http://arxiv.org/abs/2510.14415v1)

**作者:** Tadao Hoshino

**发布时间:** 2025-10-16

### GPT解析

### 总结

该论文开发了一个敏感性分析框架，将完全观测网络中的源数据的平均总处理效应转移到网络完全未知的目标数据中，以估计政策的平均社会影响。

### 背景

研究假设源数据和目标数据共享相同的条件均值结果（协变量漂移类型假设），但由于目标网络未被观测，这一假设本身不足以确定目标数据的ATTE。

### 目的

解决目标网络未观测情况下如何估计ATTE的问题，通过基于目标网络度分布不确定性的敏感性分析来构建ATTE的界限。

### 方法

考虑基于目标网络度分布不确定性的敏感性分析，不确定性程度由给定参考度分布的Wasserstein距离衡量；使用基于线性规划的估计量构建目标ATTE的界限；通过函数delta方法推导界限估计量的极限分布；开发wild bootstrap方法来近似该分布。

### 主要发现

构建了目标ATTE的界限估计量，推导了其极限分布，并开发了wild bootstrap方法来近似该分布。

### 结论

该框架允许在目标网络完全未知的情况下，通过敏感性分析来估计政策的平均社会影响。

### 翻译

这篇论文开发了一个敏感性分析框架，将具有完全观测网络的源数据中的平均总处理效应（ATTE）转移到网络完全未知的目标数据中。ATTE代表了对数据集中每个个体实施政策的平均社会影响。我们提出了一个协变量漂移类型的假设，即源数据和目标数据共享相同的条件均值结果。然而，由于目标网络未被观测，这一假设本身不足以确定目标数据的ATTE。为了解决这个问题，我们考虑了基于目标网络度分布不确定性的敏感性分析，其中不确定性程度由给定参考度分布的Wasserstein距离来衡量。然后，我们使用基于线性规划的估计量构建了目标ATTE的界限。通过函数delta方法推导了界限估计量的极限分布，并开发了wild bootstrap方法来近似该分布。作为一个实证说明，我们重新研究了Cai等人（2015）关于中国农民天气保险采用的社会网络实验。


### 论文摘要

This paper develops a sensitivity analysis framework that transfers the average total treatment effect (ATTE) from source data with a fully observed network to target data whose network is completely unknown. The ATTE represents the average social impact of a policy that assigns the treatment to every individual in the dataset. We postulate a covariate-shift type assumption that both source and target datasets share the same conditional mean outcome. However, because the target network is unobserved, this assumption alone is not sufficient to pin down the ATTE for the target data. To address this issue, we consider a sensitivity analysis based on the uncertainty of the target network's degree distribution, where the extent of uncertainty is measured by the Wasserstein distance from a given reference degree distribution. We then construct bounds on the target ATTE using a linear programming-based estimator. The limiting distribution of the bound estimator is derived via the functional delta method, and we develop a wild bootstrap approach to approximate the distribution. As an empirical illustration, we revisit the social network experiment on farmers' weather insurance adoption in China by Cai et al. (2015).

---

## 44. Glitch noise classification in KAGRA O3GK observing data using unsupervised machine learning

**论文链接:** [http://arxiv.org/abs/2510.14291v1](http://arxiv.org/abs/2510.14291v1)

**作者:** Shoichi Oshino, Yusuke Sakai, Marco Meyer-Conde, Takashi Uchiyama, Yousuke Itoh, Yutaka Shikano, Yoshikazu Terada, Hirotaka Takahashi

**发布时间:** 2025-10-16

**备注:** 9 pages, 7 figures, accepted to Physics Letters B

### GPT解析

### 总结

本研究展示了使用无监督机器学习方法对KAGRA O3GK数据中的非平稳噪声进行图像分类的有效性，成功识别出八种不同的故障噪声类别，提高了引力波观测的可靠性。

### 背景

引力波干涉仪受到各种非平稳噪声（称为故障噪声）的干扰，这些噪声影响数据分析和干涉仪的灵敏度。

### 目的

准确识别和分类故障噪声，以提高引力波观测的可靠性。

### 方法

使用变分自编码器(VAE)结合谱聚类的无监督机器学习方法，对KAGRA O3GK数据中的非平稳噪声图像进行分类，将潜在变量降维后在三维空间中可视化并进行分类。

### 主要发现

成功识别出八种不同的故障噪声类别，并更好地理解了KAGRA在O3GK期间的故障噪声特征。

### 结论

无监督学习在故障噪声分类方面显示出潜力，这有助于干涉仪升级和未来第三代引力波天文台的发展。

### 翻译

引力波干涉仪受到各种类型的非平稳噪声干扰，称为故障噪声，这些噪声影响数据分析和干涉仪灵敏度。准确识别和分类故障噪声对于提高引力波观测的可靠性至关重要。在本研究中，我们展示了无监督机器学习在KAGRA O3GK数据中分类含有非平稳噪声图像的有效性。使用变分自编码器(VAE)结合谱聚类，我们识别出八种不同的故障噪声类别。从VAE获得的潜在变量被降维，在三维空间中进行可视化，并使用谱聚类进行分类，以便更好地理解KAGRA在O3GK期间的故障噪声特征。我们的结果强调了无监督学习在高效故障噪声分类方面的潜力，这可能反过来促进干涉仪升级和未来第三代引力波天文台的发展。


### 论文摘要

Gravitational wave interferometers are disrupted by various types of nonstationary noise, referred to as glitch noise, that affect data analysis and interferometer sensitivity. The accurate identification and classification of glitch noise are essential for improving the reliability of gravitational wave observations. In this study, we demonstrated the effectiveness of unsupervised machine learning for classifying images with nonstationary noise in the KAGRA O3GK data. Using a variational autoencoder (VAE) combined with spectral clustering, we identified eight distinct glitch noise categories. The latent variables obtained from VAE were dimensionally compressed, visualized in three-dimensional space, and classified using spectral clustering to better understand the glitch noise characteristics of KAGRA during the O3GK period. Our results highlight the potential of unsupervised learning for efficient glitch noise classification, which may in turn potentially facilitate interferometer upgrades and the development of future third-generation gravitational wave observatories.

---

## 45. High-Dimensional BWDM: A Robust Nonparametric Clustering Validation Index for Large-Scale Data

**论文链接:** [http://arxiv.org/abs/2510.14145v1](http://arxiv.org/abs/2510.14145v1)

**作者:** Mohammed Baragilly, Hend Gabr

**发布时间:** 2025-10-15

### GPT解析

### 总结

本文提出了一种新的稳健非参数聚类验证框架HD-BWDM，用于解决高维或受污染数据中确定适当聚类数量的问题。

### 背景

确定无监督学习中适当的聚类数量是统计学和数据科学中的核心问题。传统的有效性指标如Calinski-Harabasz、Silhouette和Davies-Bouldin依赖于基于质心的距离，在高维或受污染数据中表现不佳。

### 目的

提出一个新的稳健的非参数聚类验证框架HD-BWDM，将BWDM标准扩展到高维空间，解决传统方法在高维数据中的局限性。

### 方法

HD-BWDM整合随机投影和主成分分析缓解维度诅咒，应用修剪聚类和基于medoid的距离确保对离群点的稳健性。作者推导了理论结果，证明在Johnson-Lindenstrauss嵌入下的一致性和收敛性。

### 主要发现

广泛的模拟表明，HD-BWDM在高维投影和污染情况下保持稳定性和可解释性，为传统基于质心的验证标准提供了稳健的替代方案。

### 结论

所提出的方法为现代高维应用中的非参数聚类提供了理论基础充分、计算效率高的停止规则。

### 翻译

确定无监督学习中适当的聚类数量是统计学和数据科学中的核心问题。传统的有效性指标如Calinski-Harabasz、Silhouette和Davies-Bouldin依赖于基于质心的距离，因此在高维或受污染数据中表现不佳。本文提出了一种新的稳健的非参数聚类验证框架，即高维组内组间距离中位数（HD-BWDM），将最近引入的BWDM标准扩展到高维空间。HD-BWDM整合了随机投影和主成分分析来缓解维度诅咒，并应用修剪聚类和基于medoid的距离以确保对离群点的稳健性。作者推导了理论结果，证明了在Johnson-Lindenstrauss嵌入下的一致性和收敛性。广泛的模拟表明，在高维投影和污染情况下，HD-BWDM保持稳定性和可解释性，为传统的基于质心的验证标准提供了一个稳健的替代方案。所提出的方法为现代高维应用中的非参数聚类提供了理论基础充分、计算效率高的停止规则。


### 论文摘要

Determining the appropriate number of clusters in unsupervised learning is a central problem in statistics and data science. Traditional validity indices such as Calinski-Harabasz, Silhouette, and Davies-Bouldin-depend on centroid-based distances and therefore degrade in high-dimensional or contaminated data. This paper proposes a new robust, nonparametric clustering validation framework, the High-Dimensional Between-Within Distance Median (HD-BWDM), which extends the recently introduced BWDM criterion to high-dimensional spaces. HD-BWDM integrates random projection and principal component analysis to mitigate the curse of dimensionality and applies trimmed clustering and medoid-based distances to ensure robustness against outliers. We derive theoretical results showing consistency and convergence under Johnson-Lindenstrauss embeddings. Extensive simulations demonstrate that HD-BWDM remains stable and interpretable under high-dimensional projections and contamination, providing a robust alternative to traditional centroid-based validation criteria. The proposed method provides a theoretically grounded, computationally efficient stopping rule for nonparametric clustering in modern high-dimensional applications.

---

## 46. RoboGPT-R1: Enhancing Robot Planning with Reinforcement Learning

**论文链接:** [http://arxiv.org/abs/2510.14828v1](http://arxiv.org/abs/2510.14828v1)

**作者:** Jinrui Liu, Bingyan Nie, Boyu Li, Yaran Chen, Yuze Wang, Shunsen He, Haoran Li

**发布时间:** 2025-10-16

### GPT解析

### 总结

本研究提出了一种名为RoboGPT-R1的两阶段微调框架，用于提升具身智能体的推理能力，使其能够更好地完成复杂环境中的长时程操作任务。

### 背景

大型语言模型和基于监督微调的视觉语言模型在规划任务中取得成功，但在复杂现实环境中执行长时程操作任务时仍面临挑战，原因是它们有限的常识和推理能力。将通用视觉语言模型通过监督微调对齐到机器人规划任务存在泛化能力差和对物理理解不足的问题。

### 目的

开发一种框架，提升具身智能体在复杂环境中的推理和规划能力，特别是完成长时程操作任务的能力。

### 方法

提出RoboGPT-R1框架，包含两个阶段：首先通过监督训练使用专家序列获取基础知识，然后利用强化学习解决模型在视觉空间理解和推理方面的不足。同时设计了基于规则的奖励函数，考虑长时程性能和环境动作约束，并在Qwen2.5-VL-3B上训练推理模型。

### 主要发现

在EmbodiedBench基准测试上，训练的推理模型显著优于更大规模的GPT-4o-mini模型，性能高出21.33%；同时超越了在Qwen2.5-VL-7B上训练的其他工作，高出20.33%。

### 结论

RoboGPT-R1框架有效提升了具身智能体的推理能力和规划能力，使其能够更好地完成复杂环境中的长时程操作任务。

### 翻译

提升具身智能体的推理能力对于机器人在长时程操作任务中成功完成复杂的人类指令至关重要。尽管基于监督微调的大型语言模型和视觉语言模型在规划任务中取得了成功，但由于常识和推理能力的限制，它们在复杂现实环境中执行长时程操作任务时仍面临挑战。考虑到通过监督微调将通用视觉语言模型对齐到机器人规划任务存在泛化能力差和物理理解不足的问题，我们提出了RoboGPT-R1，这是一个用于具身规划的两阶段微调框架。在该框架中，监督训练通过专家序列获取基础知识，随后使用强化学习解决模型在视觉空间理解和推理方面的不足。为了在多步推理任务中实现物理理解和动作序列一致性，我们设计了一个基于规则的奖励函数，同时考虑长时程性能和环境中的动作约束。在Qwen2.5-VL-3B上训练的推理模型在EmbodiedBench基准测试上显著优于更大规模的GPT-4o-mini模型，高出21.33%，并超越了在Qwen2.5-VL-7B上训练的其他工作，高出20.33%。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': "论文主要解决提升具身智能体在复杂长视野操作任务中的推理能力问题。当前基于监督微调的大语言模型在真实世界环境中执行长期任务时面临泛化能力不足和物理理解有限的问题。这一问题在现实中非常重要，因为机器人需要处理如'打扫厨房'或'准备晚餐'等复杂、长期的指令，而现有方法难以在动态环境中有效适应和自我纠正。", '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': "作者首先分析了现有SFT-only范式的局限性，包括缺乏环境适应能力和奖励函数设计不足。他们借鉴了强化学习在其他领域（如视频推理、数学推理）的成功应用，以及DeepSeek-R1中的'aha moment'概念。具体设计上，作者结合了REBP项目中的数据集和GRPO算法，同时创新性地设计了包含格式奖励和LCS准确奖励的奖励函数，以解决多步推理任务中的物理理解和动作序列一致性问题。", '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是通过两阶段训练框架结合监督微调和强化学习的优势，并设计针对具身任务的奖励函数。整体流程包括：1)数据准备阶段，使用从Gemini-2.0-flash提炼的SFT数据集和增强的RFT数据集；2)两阶段训练，第一阶段SFT赋予模型基础规划能力，第二阶段使用GRPO进行强化微调提升推理和泛化能力；3)奖励设计，结合格式奖励(评估结构完整性和动作有效性)和LCS准确奖励(关注动作序列顺序)；4)在EmbodiedBench基准上评估性能，包括域内(EB-ALFRED)和域外(EB-Habitat)场景。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)两阶段训练框架，结合SFT和GRPO强化学习；2)创新的奖励函数设计，包含格式奖励和LCS准确奖励；3)仅使用3B参数的小型模型实现高性能；4)采用零样本处理提高训练效率和泛化能力。相比之前的工作，RoboGPT-R1在EB-ALFRED基准上比GPT-4o-mini高21.33%，比其他基于Qwen2.5-VL-7B的工作高20.33%，特别是在长视野任务上达到50%的准确率，显著优于现有方法。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'RoboGPT-R1通过结合监督微调和强化学习的两阶段训练框架，以及针对具身任务设计的基于规则的奖励函数，显著提升了小型视觉语言模型在复杂长视野机器人规划任务中的性能和泛化能力。'}


### 论文摘要

Improving the reasoning capabilities of embodied agents is crucial for robots to complete complex human instructions in long-view manipulation tasks successfully. Despite the success of large language models and vision language models based on Supervised Fine-Tuning (SFT) in planning tasks, they continue facing challenges in performing long-horizon manipulation tasks in complex real-world environments, owing to their restricted common sense and reasoning capabilities. Considering that aligning general-purpose vision language models to robotic planning tasks via supervised fine-tuning suffers from poor generalization and insufficient physical understanding, we propose RoboGPT-R1, a two-stage fine-tuning framework for embodied planning. In this framework, supervised training acquires foundational knowledge through expert sequences, followed by RL to address the model's shortcomings in visual-spatial understanding and reasoning. To achieve physical understanding and action sequence consistency in multi-step reasoning tasks, we design a rule-based reward function that simultaneously considers long-horizon performance and action constraint in the environment. The reasoning model, trained on Qwen2.5-VL-3B, significantly outperforms the larger-scale model, GPT-4o-mini, by 21.33% and surpasses other work trained on Qwen2.5-VL-7B by 20.33% on the EmbodiedBench benchmark.

---

## 47. Spatially anchored Tactile Awareness for Robust Dexterous Manipulation

**论文链接:** [http://arxiv.org/abs/2510.14647v1](http://arxiv.org/abs/2510.14647v1)

**作者:** Jialei Huang, Yang Ye, Yuanqing Gong, Xuezhou Zhu, Yang Gao, Kaifeng Zhang

**发布时间:** 2025-10-16

**备注:** 8 pages

### GPT解析

### 总结

本研究提出了一种名为SaTA的空间锚定触觉感知方法，用于解决灵巧操作中的高精度几何推理问题。该方法通过将触觉特征锚定到手部运动学框架，实现了无需物体模型或显式姿态估计的精确几何推理。

### 背景

灵巧操作需要精确的几何推理，但现有的视觉-触觉学习方法在处理亚毫米精度任务时存在困难，而传统基于模型的方法可以轻松处理这些任务。

### 目的

开发一种能够有效利用触觉信号的感知丰富性及其与手部运动学空间关系的框架，以实现高精度的灵巧操作。

### 方法

提出了SaTA（Spatially-anchored Tactile Awareness for dexterous manipulation）框架，一种端到端策略框架，通过正向运动学将触觉特征锚定到手部运动学框架中。

### 主要发现

空间锚定的触觉表示使策略不仅能够检测接触发生，还能在手部坐标系中精确推断物体几何形状。SaTA在多个基准测试中显著优于强视觉-触觉基线，成功率提高高达30个百分点，任务完成时间减少27%。

### 结论

SaTA通过将触觉特征锚定到手部运动学框架，成功解决了现有学习框架在处理高精度灵巧操作任务时的局限性，实现了无需物体模型或显式姿态估计的精确几何推理。

### 翻译

灵巧操作需要精确的几何推理，然而现有的视觉-触觉学习方法在处理亚毫米精度任务时存在困难，而这些任务对于传统基于模型的方法来说则是常规操作。我们确定了一个关键限制：虽然触觉传感器提供了丰富的接触信息，但现有学习框架未能有效利用触觉信号的感知丰富性及其与手部运动学的空间关系。我们认为理想的触觉表示应将接触测量明确地锚定在稳定的参考框架中，同时保留详细的感官信息，使策略不仅能够检测接触发生，还能在手部坐标系中精确推断物体几何形状。我们引入了SaTA（用于灵巧操作的空间锚定触觉感知），一种端到端策略框架，通过正向运动学将触觉特征明确锚定到手部运动学框架，无需物体模型或显式姿态估计即可实现准确的几何推理。我们的关键见解是空间锚定的触觉表示使策略不仅能够检测接触发生，还能在手部坐标系中精确推断物体几何形状。我们在具有挑战性的灵巧操作任务上验证了SaTA，包括自由空间中的双臂USB-C连接（需要亚毫米级对齐精度）、需要精确螺纹啮合和旋转控制的灯泡安装，以及需要精细力调制和角度精度的卡片滑动。这些任务由于其严格的精度要求，对基于学习的方法构成了重大挑战。在多个基准测试中，SaTA显著优于强视觉-触觉基线，成功率提高高达30个百分点，同时任务完成时间减少27%。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决灵巧操作中如何有效利用触觉信号进行精确几何推理的问题，特别是在需要亚毫米级精度的任务中。这个问题很重要，因为在多指多接触场景中，毫米级误差就可能导致任务失败（如USB连接器无法插入），而在接触关键时刻，视觉信息常被遮挡或失效，精确的几何信息对成功操作至关重要。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先识别出触觉传感器虽提供丰富信息但现有学习框架未能有效利用其感知丰富性和空间关系这一关键限制。他们认为理想的触觉表示应将接触测量稳定在参考框架中，同时保留详细感官信息。设计方法借鉴了ACT框架作为基础架构，使用FiLM机制整合空间上下文与触觉特征，应用Fourier特征编码捕获多尺度几何变化，并采用模仿学习策略使用专家演示数据进行训练。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是将触觉特征锚定到手部运动学坐标系中，同时保留完整几何信息，使策略能准确推断接触状态和物体几何形状，直接输出精确操作动作。整体流程是：接收多模态输入（RGB图像、触觉图像、关节角度）；通过正向运动学计算触觉传感器6D姿态；用Fourier特征编码空间信息；通过FiLM整合空间上下文与触觉特征；将多模态信息通过Transformer处理；生成动作序列实现亚毫米精度操作。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：空间锚定触觉表示方法；端到端操作策略框架SaTA；高精度灵巧操作任务的成功验证。相比之前工作，SaTA将触觉测量显式锚定到手部坐标系而非处理为抽象特征；保留了完整触觉图像特征而非转换为简化几何形式；直接输出操作动作而非专注于感知重建；不依赖显式物体模型或离线优化过程。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'SaTA通过空间锚定触觉表示，使基于学习的方法能够实现亚毫米级精度的灵巧操作，成功解决了传统视觉-触觉学习方法在需要高精度几何推理任务中的局限性。'}


### 论文摘要

Dexterous manipulation requires precise geometric reasoning, yet existing visuo-tactile learning methods struggle with sub-millimeter precision tasks that are routine for traditional model-based approaches. We identify a key limitation: while tactile sensors provide rich contact information, current learning frameworks fail to effectively leverage both the perceptual richness of tactile signals and their spatial relationship with hand kinematics. We believe an ideal tactile representation should explicitly ground contact measurements in a stable reference frame while preserving detailed sensory information, enabling policies to not only detect contact occurrence but also precisely infer object geometry in the hand's coordinate system. We introduce SaTA (Spatially-anchored Tactile Awareness for dexterous manipulation), an end-to-end policy framework that explicitly anchors tactile features to the hand's kinematic frame through forward kinematics, enabling accurate geometric reasoning without requiring object models or explicit pose estimation. Our key insight is that spatially grounded tactile representations allow policies to not only detect contact occurrence but also precisely infer object geometry in the hand's coordinate system. We validate SaTA on challenging dexterous manipulation tasks, including bimanual USB-C mating in free space, a task demanding sub-millimeter alignment precision, as well as light bulb installation requiring precise thread engagement and rotational control, and card sliding that demands delicate force modulation and angular precision. These tasks represent significant challenges for learning-based methods due to their stringent precision requirements. Across multiple benchmarks, SaTA significantly outperforms strong visuo-tactile baselines, improving success rates by up to 30 percentage while reducing task completion times by 27 percentage.

---

## 48. QuASH: Using Natural-Language Heuristics to Query Visual-Language Robotic Maps

**论文链接:** [http://arxiv.org/abs/2510.14546v1](http://arxiv.org/abs/2510.14546v1)

**作者:** Matti Pekkanen, Francesco Verdoja, Ville Kyrki

**发布时间:** 2025-10-16

**备注:** Submitted to ICRA 2026

### GPT解析

### 总结

本文提出了一种利用视觉语言模型嵌入表示机器人地图语义的方法，通过自然语言同义词和反义词来训练分类器，解决机器人确定环境中与查询相关部分的挑战。

### 背景

视觉语言模型的嵌入表示被越来越多地用于表示机器人地图中的语义，提供开放词汇的场景理解，超越了传统有限标签的表示方法。

### 目的

解决机器人确定环境中与查询相关部分的关键挑战，提高地图和图像的查询能力。

### 方法

利用嵌入空间中与查询相关的自然语言同义词和反义词，应用启发式方法估计与查询相关的语言空间，并使用该语言空间训练分类器来将环境划分为匹配和不匹配的部分。

### 主要发现

通过大量实验表明，该方法能够显著提高地图和图像的查询能力，且该查询技术与表示和编码器无关，只需要有限的训练。

### 结论

所提出的方法有效解决了机器人确定环境中与查询相关部分的挑战，提高了地图和图像的查询能力，具有广泛的适用性。

### 翻译

视觉语言模型的嵌入表示越来越多地被用于表示机器人地图中的语义，提供开放词汇的场景理解，超越了传统的有限标签。嵌入表示通过相似度比较将嵌入的用户文本提示与地图嵌入，实现按需查询。执行查询任务的关键挑战是机器人必须确定环境中与查询相关的部分。本文提出了这一挑战的解决方案。我们利用嵌入空间中与查询相关的自然语言同义词和反义词，应用启发式方法估计与查询相关的语言空间，并使用该语言空间训练分类器将环境划分为匹配和不匹配的部分。我们通过大量实验评估了该方法，包括对地图和标准图像基准的查询。结果表明地图和图像的查询能力得到了提高。我们的查询技术与表示和编码器无关，只需要有限的训练。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决的是如何更有效地从视觉-语言模型(VLM)的嵌入表示中查询相关信息的问题。具体来说，当机器人需要根据自然语言查询在地图或图像中找到相关物体或区域时，现有方法无法准确确定环境中与查询相关的部分。这个问题很重要，因为随着视觉-语言模型的发展，机器人地图能够包含更丰富的语义信息，开放词汇的场景理解能力对机器人执行复杂任务至关重要，而现有方法在匹配查询与地图嵌入时性能有限，限制了机器人对环境的理解和交互能力。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': "作者首先分析了现有方法的局限性：现有方法主要采用阈值化余弦相似度或使用单个互补查询（如'other'）的策略，但这些方法假设所有维度对查询的重要性相同，且仅使用单个查询和单个负例无法准确估计相关区域的范围。基于这些分析，作者设计了QuASH方法，利用自然语言同义词和反义词来估计与查询相关的语言空间，通过启发式方法生成语义相关的同义词和反义词，并基于这些样本训练一个分类器。该方法借鉴了现有工作中的视觉-语言嵌入表示和查询机制，但改进了查询策略，不再依赖简单的阈值或单个负例比较。", '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是利用自然语言语义知识，通过生成查询的同义词和反义词样本来训练一个分类器，从而更准确地确定嵌入空间中与查询相关的区域。整体实现流程包括：1) 给定文本查询，生成一组语义同义词和反义词，并添加通用的负例查询；2) 使用嵌入函数将所有文本转换为嵌入表示；3) 使用这些嵌入表示作为训练数据，训练一个分类器；4) 给定一个地图，使用训练好的分类器对地图进行分类，得到与查询匹配的区域。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1) 提出了一种新的查询形式化方法，将查询过程视为在嵌入空间中的分类问题；2) 设计了QuASH方法，利用自然语言启发式方法生成同义词和反义词样本来训练分类器；3) 该方法不依赖于特定的嵌入表示或编码器，具有通用性；4) 通过非线性分类器而非简单的相似度阈值或线性分割来估计相关区域。相比之前的工作，不同之处在于不再依赖单一的查询嵌入和单一的负例嵌入进行比较，不使用固定的相似度阈值，而是通过训练的分类器动态确定决策边界，考虑了嵌入空间中不同维度可能具有不同语义重要性的事实，方法更加灵活，可以适应不同的视觉-语言模型和编码器。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'QuASH通过利用自然语言启发式方法生成同义词和反义词样本来训练分类器，显著提高了机器人地图和图像中基于自然语言查询的准确性，同时保持了方法的通用性和灵活性。'}


### 论文摘要

Embeddings from Visual-Language Models are increasingly utilized to represent semantics in robotic maps, offering an open-vocabulary scene understanding that surpasses traditional, limited labels. Embeddings enable on-demand querying by comparing embedded user text prompts to map embeddings via a similarity metric. The key challenge in performing the task indicated in a query is that the robot must determine the parts of the environment relevant to the query.   This paper proposes a solution to this challenge. We leverage natural-language synonyms and antonyms associated with the query within the embedding space, applying heuristics to estimate the language space relevant to the query, and use that to train a classifier to partition the environment into matches and non-matches. We evaluate our method through extensive experiments, querying both maps and standard image benchmarks. The results demonstrate increased queryability of maps and images. Our querying technique is agnostic to the representation and encoder used, and requires limited training.

---

## 49. Spatial Preference Rewarding for MLLMs Spatial Understanding

**论文链接:** [http://arxiv.org/abs/2510.14374v1](http://arxiv.org/abs/2510.14374v1)

**作者:** Han Qiu, Peng Gao, Lewei Lu, Xiaoqin Zhang, Ling Shao, Shijian Lu

**发布时间:** 2025-10-16

**备注:** ICCV 2025

### GPT解析

### 总结

本文提出SPR（空间偏好奖励）方法，通过奖励多模态大语言模型生成具有精确物体定位的详细响应，增强其细粒度空间理解能力，实验证明该方法有效且训练开销小。

### 背景

多模态大语言模型已展现出空间理解能力，但在细粒度空间感知方面仍有不足，如无法生成详细区域描述或准确定位物体，且常无法满足用户对细粒度空间理解的需求。

### 目的

解决现有MLLM方法缺乏对实际响应直接监督的问题，通过SPR方法提升MLLM的细粒度空间理解能力。

### 方法

SPR方法通过随机选择图像区域和描述，引入语义和定位分数评估MLLM生成描述的质量；使用高定位精度描述完善MLLM输出，并将最佳完善与初始最低分描述配对进行直接偏好优化，增强与视觉输入的细粒度对齐。

### 主要发现

在标准引用和定位基准上的大量实验表明，SPR有效提高了MLLM的空间理解能力，同时训练开销最小。

### 结论

SPR方法能够显著增强MLLM的细粒度空间理解能力，相关数据和代码将在https://github.com/hanqiu-hq/SPR发布。

### 翻译

多模态大语言模型已展现出有希望的空间理解能力，如引用和定位物体描述。尽管取得了成功，MLLMs在细粒度空间感知能力方面仍有不足，例如生成详细的区域描述或准确定位物体。此外，它们经常无法响应用户对所需细粒度空间理解的要求。这个问题可能是因为现有方法主要专注于调整MLLMs以建模预标注的指令数据来注入空间知识，而没有直接监督MLLMs的实际响应。我们通过SPR（空间偏好奖励）方法解决这个问题，通过奖励MLLMs具有精确物体定位的详细响应，而不是模糊或不准确的响应，从而增强MLLMs的空间能力。使用从MLLMs中随机选择的图像区域和区域描述，SPR引入语义和定位分数来全面评估MLLM生成描述中的文本质量和定位质量。我们还使用更好的定位精度来完善MLLM描述，并将得分最高的完善与初始得分最低的描述配对，用于直接偏好优化，从而增强与视觉输入的细粒度对齐。在标准引用和基准测试上的大量实验表明，SPR有效地提高了MLLM的空间理解能力，同时训练开销最小。数据和代码将在https://github.com/hanqiu-hq/SPR发布。


### 论文摘要

Multimodal large language models~(MLLMs) have demonstrated promising spatial understanding capabilities, such as referencing and grounding object descriptions. Despite their successes, MLLMs still fall short in fine-grained spatial perception abilities, such as generating detailed region descriptions or accurately localizing objects. Additionally, they often fail to respond to the user's requirements for desired fine-grained spatial understanding. This issue might arise because existing approaches primarily focus on tuning MLLMs to model pre-annotated instruction data to inject spatial knowledge, without direct supervision of MLLMs' actual responses. We address this issue by SPR, a Spatial Preference Rewarding~(SPR) approach that enhances MLLMs' spatial capabilities by rewarding MLLMs' detailed responses with precise object localization over vague or inaccurate responses. With randomly selected image regions and region descriptions from MLLMs, SPR introduces semantic and localization scores to comprehensively evaluate the text quality and localization quality in MLLM-generated descriptions. We also refine the MLLM descriptions with better localization accuracy and pair the best-scored refinement with the initial descriptions of the lowest score for direct preference optimization, thereby enhancing fine-grained alignment with visual input. Extensive experiments over standard referring and grounding benchmarks show that SPR improves MLLM spatial understanding capabilities effectively with minimal overhead in training. Data and code will be released at https://github.com/hanqiu-hq/SPR

---

## 50. SUM-AgriVLN: Spatial Understanding Memory for Agricultural Vision-and-Language Navigation

**论文链接:** [http://arxiv.org/abs/2510.14357v1](http://arxiv.org/abs/2510.14357v1)

**作者:** Xiaobei Zhao, Xingqi Lyu, Xiang Li

**发布时间:** 2025-10-16

### GPT解析

### 总结

本文提出了SUM-AgriVLN方法，通过空间理解记忆模块改进农业视觉语言导航，解决了现有方法忽略过去经验提供空间上下文的问题，在A2A基准测试上取得了最先进的性能。

### 背景

农业机器人正在成为各种农业任务的有力助手，但仍严重依赖人工操作或固定轨道系统进行移动。AgriVLN方法和A2A基准率先将视觉语言导航扩展到农业领域，使机器人能够遵循自然语言指令导航到目标位置。

### 目的

解决现有AgriVLN方法将每个导航指令视为独立片段而忽略过去经验提供空间上下文的问题，特别是在农业场景中经常出现重复导航指令的情况下。

### 方法

提出空间理解记忆用于农业视觉语言导航(SUM-AgriVLN)方法，其中SUM模块利用空间理解并通过三维重建和表示保存空间记忆。

### 主要发现

在A2A基准测试上，SUM-AgriVLN成功将成功率从0.47提高到0.54，导航误差仅从2.91米略微增加到2.93米，展示了在农业领域最先进的性能。

### 结论

SUM-AgriVLN方法有效利用了空间记忆来改进农业视觉语言导航性能，证明了在农业机器人导航中考虑历史经验的重要性。

### 翻译

农业机器人正在成为各种农业任务的有力助手，然而，它们仍然严重依赖人工操作或固定轨道系统进行移动。AgriVLN方法和A2A基准率先将视觉语言导航扩展到农业领域，使机器人能够遵循自然语言指令导航到目标位置。在实际农业场景中，导航指令经常重复出现，但AgriVLN将每个指令视为独立片段，忽略了过去经验为后续指令提供空间上下文的潜力。为了弥合这一差距，我们提出了用于农业视觉语言导航的空间理解记忆方法，其中SUM模块利用空间理解并通过三维重建和表示保存空间记忆。在A2A基准测试上评估时，我们的SUM-AgriVLN成功将成功率从0.47提高到0.54，导航误差仅从2.91米略微增加到2.93米，展示了在农业领域最先进的性能。代码：https://github.com/AlexTraveling/SUM-AgriVLN。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决农业机器人在视觉语言导航任务中缺乏空间长期记忆的问题。这个问题很重要，因为实际农业场景中经常需要重复执行相似导航指令，而现有方法将每个指令视为独立事件，无法利用过去经验提供的空间上下文，导致机器人导航效率低下。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者观察到人类导航时会自发形成空间记忆并在后续任务中利用，而现有农业机器人缺乏这种能力。他们受日常生活中第一次和第二次去陌生地方的差异启发，设计出空间理解记忆模块。该方法借鉴了VGGT视觉编码器用于3D重建，参考了结构运动和多视图立体等3D重建技术，并在AgriVLN基础上进行了改进。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是引入空间理解记忆模块，使机器人能够保存和利用空间记忆。整体流程包括：1)空间理解：从相机图像集中采样10帧，用VGGT生成3D重建；2)空间记忆：将3D重建渲染为点云，提取正面和倾斜两种视角的2D RGB表示并存储；3)基础模型集成：将SUM模块融入AgriVLN，在每一步加载空间记忆，结合语言指令和视觉输入预测行动；4)任务执行：持续更新子任务列表直到满足结束条件。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)首次将空间记忆引入农业视觉语言导航；2)通过3D重建和表示保存空间记忆，而非传统基于图的方法；3)提取多视角空间表示提供丰富上下文；4)能够利用任务间经验。相比之前工作，SUM-AgriVLN不同于传统VLN方法专注于农业场景，不同于AgriVLN将任务视为独立事件，不同于现有空间记忆方法依赖图结构，也不同于传统3D重建只关注几何准确性。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '本文提出的SUM-AgriVLN方法通过引入空间理解记忆模块，使农业机器人能够保存和利用空间记忆，显著提高了在重复导航任务中的成功率，从0.47提升到0.54。'}


### 论文摘要

Agricultural robots are emerging as powerful assistants across a wide range of agricultural tasks, nevertheless, still heavily rely on manual operation or fixed rail systems for movement. The AgriVLN method and the A2A benchmark pioneeringly extend Vision-and-Language Navigation (VLN) to the agricultural domain, enabling robots to navigate to the target positions following the natural language instructions. In practical agricultural scenarios, navigation instructions often repeatedly occur, yet AgriVLN treat each instruction as an independent episode, overlooking the potential of past experiences to provide spatial context for subsequent ones. To bridge this gap, we propose the method of Spatial Understanding Memory for Agricultural Vision-and-Language Navigation (SUM-AgriVLN), in which the SUM module employs spatial understanding and save spatial memory through 3D reconstruction and representation. When evaluated on the A2A benchmark, our SUM-AgriVLN effectively improves Success Rate from 0.47 to 0.54 with slight sacrifice on Navigation Error from 2.91m to 2.93m, demonstrating the state-of-the-art performance in the agricultural domain. Code: https://github.com/AlexTraveling/SUM-AgriVLN.

---

## 51. Leveraging Cycle-Consistent Anchor Points for Self-Supervised RGB-D Registration

**论文链接:** [http://arxiv.org/abs/2510.14354v1](http://arxiv.org/abs/2510.14354v1)

**作者:** Siddharth Tourani, Jayaram Reddy, Sarvesh Thakur, K Madhava Krishna, Muhammad Haris Khan, N Dinesh Reddy

**发布时间:** 2025-10-16

**DOI:** 10.1109/ICRA57147.2024.10610738

**备注:** 8 pages, accepted at ICRA 2024 (International Conference on Robotics  and Automation)

### GPT解析

### 总结

本文提出了一种利用未标记RGB-D数据进行场景几何推理的新方法，通过循环一致的关键点和结合GRU循环单元的姿态模块，提高了RGB-D配准的准确性。

### 背景

随着消费级深度相机的普及，大量未标记的RGB-D数据变得可用，如何有效利用这些数据进行场景几何推理成为一个重要问题。

### 目的

探索如何利用未标记的RGB-D数据进行场景的几何推理，提高RGB-D配准的准确性。

### 方法

不同于传统的基于几何和特征相似性的RGB-D配准方法，作者使用循环一致的关键点作为显著点强制执行空间一致性约束，并引入结合GRU循环单元和变换同步的姿态模块来融合历史和多视图数据。

### 主要发现

在ScanNet和3DMatch数据集上，该方法超越了之前的自监督配准方法，甚至优于一些旧的监督方法；将组件集成到现有方法中也证明了其有效性。

### 结论

通过创新的循环一致关键点和姿态模块设计，有效提高了RGB-D配准的准确性，为未标记RGB-D数据的利用提供了新思路。

### 翻译

随着消费级深度相机的兴起，大量未标记的RGB-D数据变得可用。这引发了一个问题：如何利用这些数据进行场景的几何推理。虽然许多RGB-D配准方法依赖于几何和基于特征的相似性，我们采取了不同的方法。我们使用循环一致的关键点作为显著点，在匹配过程中强制执行空间一致性约束，提高对应点准确性。此外，我们引入了一个新的姿态模块，将GRU循环单元与变换同步相结合，融合历史和多视图数据。我们的方法在ScanNet和3DMatch上超越了之前的自监督配准方法，甚至优于一些旧的监督方法。我们还将我们的组件集成到现有方法中，证明了它们的有效性。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决如何利用大量无标签的RGB-D数据进行场景几何推理，特别是RGB-D配准问题。这个问题很重要，因为随着消费级深度相机的普及，有大量无标签RGB-D数据可用，而RGB-D数据在机器人任务(如SLAM、无人机导航和物体姿态估计)中非常关键。传统配准方法在有噪声或特征稀少环境下表现不佳，且现有自监督方法未充分利用场景中的显著点信息。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者思考利用场景中的显著点作为锚点，这些点在多视角下易被识别且循环一致。通过空间一致性约束改善对应关系搜索，并结合GRU循环单元和变换同步融合历史和多视图信息。作者借鉴了多项现有工作：使用ResNet-18作为特征提取网络，采用LofTr的匹配策略，利用Sinkhorn归一化，参考矩阵分解方法获得循环一致匹配，受启发于GRU单元和变换同步方法。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是利用循环一致的显著点(锚点)施加空间一致性约束提高对应准确性，结合GRU和变换同步进行姿态估计。整体流程：1)使用ResNet-18提取特征；2)通过Sinkhorn归一化获得软匹配并转换为循环一致的锚点；3)使用锚点距离编码修改自注意力模块；4)定义空间一致性成本函数；5)迭代进行像素级匹配和姿态更新；6)结合GRU和变换同步改进姿态估计；7)通过内部迭代(20次)和外部迭代(3次)优化结果。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点：1)循环一致关键点匹配模块，施加空间约束；2)无RANSAC的姿态估计方法，结合GRU和变换同步；3)空间一致性成本函数；4)迭代优化框架。不同之处：大多数自监督方法依赖特征相似性或几何信息，而本文利用场景显著点；之前循环一致性方法应用于所有像素，本文仅用于定位显著点；与[24]不同，本文用空间约束学习锚点而非修剪离群值；结合GRU和变换同步，而非仅使用一种方法。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '本文通过循环一致锚点和空间一致性约束提出新自监督RGB-D配准方法，显著提高配准精度，超越之前自监督方法并接近一些有监督方法的性能。'}


### 论文摘要

With the rise in consumer depth cameras, a wealth of unlabeled RGB-D data has become available. This prompts the question of how to utilize this data for geometric reasoning of scenes. While many RGB-D registration meth- ods rely on geometric and feature-based similarity, we take a different approach. We use cycle-consistent keypoints as salient points to enforce spatial coherence constraints during matching, improving correspondence accuracy. Additionally, we introduce a novel pose block that combines a GRU recurrent unit with transformation synchronization, blending historical and multi-view data. Our approach surpasses previous self- supervised registration methods on ScanNet and 3DMatch, even outperforming some older supervised methods. We also integrate our components into existing methods, showing their effectiveness.

---

## 52. Efficient Few-Shot Learning in Remote Sensing: Fusing Vision and Vision-Language Models

**论文链接:** [http://arxiv.org/abs/2510.13993v1](http://arxiv.org/abs/2510.13993v1)

**作者:** Jia Yun Chua, Argyrios Zolotas, Miguel Arana-Catania

**发布时间:** 2025-10-15

**备注:** 11 pages, 7 figures, 8 tables. To be published in Applied AI Letters

### GPT解析

### 总结

本研究探索了结合传统视觉模型与视觉语言模型(VLMs)以增强遥感图像分析，特别是在飞机检测和场景理解方面的应用。通过集成YOLO与LLaVA、ChatGPT和Gemini等VLMs，实现了更准确和具有上下文意识的图像解释。

### 背景

遥感已成为城市规划、环境监测和灾害响应等领域的关键工具，数据量显著增加。然而，传统视觉模型受限于需要大量领域特定标记数据且在理解复杂环境上下文方面能力有限。视觉语言模型虽能整合视觉和文本数据，但在遥感领域的应用尚未充分探索。

### 目的

研究视觉模型与VLMs的结合，以增强遥感图像分析，专注于飞机检测和场景理解任务。

### 方法

集成YOLO与VLMs(如LLaVA、ChatGPT和Gemini)，在标记和未标记的遥感数据以及退化图像场景上评估性能，旨在实现更准确和具有上下文意识的图像解释。

### 主要发现

在原始和退化场景中，特别是在具有挑战性的条件下，飞机检测和计数的准确性平均提高了48.46%。在遥感图像的全面理解方面，CLIPScore提高了6.17%。

### 结论

结合传统视觉模型和VLMs的方法为更先进和高效的遥感图像分析铺平了道路，特别在少样本学习场景中表现优异。

### 翻译

遥感已成为城市规划、环境监测和灾害响应等跨领域的关键工具。尽管生成的数据量显著增加，但传统视觉模型通常受限于需要大量领域特定标记数据及其在理解复杂环境中上下文能力的有限性。视觉语言模型通过整合视觉和文本数据提供了一种互补方法；然而，它们在遥感领域的应用仍未得到充分探索，特别是考虑到它们的通用性质。本研究探讨了结合视觉模型和VLMs以增强遥感图像分析，专注于飞机检测和场景理解。将YOLO与LLaVA、ChatGPT和Gemini等VLMs的集成旨在实现更准确和具有上下文意识的图像解释。性能在标记和未标记的遥感数据以及退化图像场景上进行了评估，这些场景对遥感至关重要。研究显示，在原始和退化场景中，特别是在具有挑战性的条件下，飞机检测和计数的准确性在各类模型中平均提高了48.46%。在遥感图像的全面理解方面，获得了6.17%的CLIPScore提升。结合传统视觉模型和VLMs的方法为更先进和高效的遥感图像分析铺平了道路，特别是在少样本学习场景中。


### 论文摘要

Remote sensing has become a vital tool across sectors such as urban planning, environmental monitoring, and disaster response. While the volume of data generated has increased significantly, traditional vision models are often constrained by the requirement for extensive domain-specific labelled data and their limited ability to understand the context within complex environments. Vision Language Models offer a complementary approach by integrating visual and textual data; however, their application to remote sensing remains underexplored, particularly given their generalist nature. This work investigates the combination of vision models and VLMs to enhance image analysis in remote sensing, with a focus on aircraft detection and scene understanding. The integration of YOLO with VLMs such as LLaVA, ChatGPT, and Gemini aims to achieve more accurate and contextually aware image interpretation. Performance is evaluated on both labelled and unlabelled remote sensing data, as well as degraded image scenarios which are crucial for remote sensing. The findings show an average MAE improvement of 48.46% across models in the accuracy of aircraft detection and counting, especially in challenging conditions, in both raw and degraded scenarios. A 6.17% improvement in CLIPScore for comprehensive understanding of remote sensing images is obtained. The proposed approach combining traditional vision models and VLMs paves the way for more advanced and efficient remote sensing image analysis, especially in few-shot learning scenarios.

---

## 53. CVD-STORM: Cross-View Video Diffusion with Spatial-Temporal Reconstruction Model for Autonomous Driving

**论文链接:** [http://arxiv.org/abs/2510.07944v2](http://arxiv.org/abs/2510.07944v2)

**作者:** Tianrui Zhang, Yichen Liu, Zilin Guo, Yuxin Guo, Jingcheng Ni, Chenjing Ding, Dan Xu, Lewei Lu, Zehuan Wu

**发布时间:** 2025-10-09

### GPT解析

### 总结

CVD-STORM是一个跨视图视频扩散模型，利用空间-时间重建变分自编码器生成长期多视图视频并具备4D重建能力。

### 背景

生成模型已被广泛应用于世界建模和环境模拟、未来状态预测。随着自动驾驶的发展，对高质量视频生成以及深度估计等多样化有意义信息的需求日益增长。

### 目的

提出CVD-STORM模型，能够在各种控制输入下生成长期多视图视频，具备4D重建能力。

### 方法

首先使用辅助的4D重建任务对VAE进行微调，增强其编码3D结构和时间动态的能力；然后将这个VAE集成到视频扩散过程中提高生成质量；联合训练的高斯溅射解码器有效重建动态场景，为场景理解提供几何信息。

### 主要发现

实验结果表明，该模型在FID和FVD指标上都取得了显著改进。

### 结论

CVD-STORM模型能够在各种控制条件下生成高质量的多视图视频，并有效重建动态场景，为场景理解提供几何信息。

### 翻译

生成模型已被广泛应用于世界建模和环境模拟以及未来状态预测。随着自动驾驶的发展，不仅需要高质量的视频生成，还需要产生多样化和有意义的信息如深度估计。为此，我们提出了CVD-STORM，这是一个利用空间-时间重建变分自编码器的跨视图视频扩散模型，能够在各种控制输入下生成具有4D重建能力的长期多视图视频。我们的方法首先使用辅助的4D重建任务对VAE进行微调，增强其编码3D结构和时间动态的能力。随后，我们将这个VAE集成到视频扩散过程中，显著提高了生成质量。实验结果表明，我们的模型在FID和FVD指标上都取得了显著改进。此外，联合训练的高斯溅射解码器有效地重建动态场景，为全面场景理解提供了有价值的几何信息。我们的项目页面是https://sensetime-fvg.github.io/CVD-STORM。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决自动驾驶领域中高质量视频生成和4D场景重建的问题。具体来说，现有方法难以同时生成长期、多视角的视频并提供准确的深度信息，这限制了自动驾驶系统对环境的模拟和未来状态的预测能力。这个问题在现实中非常重要，因为自动驾驶需要准确的环境模拟来训练决策算法和验证规划输出，而深度信息对于理解场景的3D结构至关重要。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先认识到现有视频扩散模型在生成长期、多视角视频方面的局限性，以及缺乏明确3D信息的问题。他们借鉴了多项现有工作：基于现有的扩散模型架构（如DiT），参考了STORM模型的空间-时间重建方法，采用了UniMLVG的多模态DiT架构和训练策略，利用了3D高斯溅射技术进行场景重建，并结合了VAE进行表示学习。作者通过整合这些技术，设计了一个两阶段训练策略：先学习场景重建，再训练条件世界模型，以实现高质量的视频生成和4D场景重建。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是通过STORM-VAE（一个扩展的VAE模型，集成了高斯溅射解码器）进行4D场景重建，并利用CVD-STORM框架同时生成多视角视频和重建4D场景。整体实现流程分为三部分：1）STORM-VAE训练：使用预训练的图像VAE，添加高斯溅射解码器分支，通过多视图图像和相机姿态进行训练；2）CVD-STORM训练：使用STORM-VAE作为潜在编码器，在扩散模型中集成STORM-VAE，使用三个不同的transformer块处理不同维度；3）推理过程：生成长期六视角视频，高斯溅射解码器直接从生成的潜在表示重建4D场景。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1）STORM-VAE：一个能进行4D场景重建的扩展VAE模型；2）CVD-STORM：统一框架同时生成多视角视频和重建4D场景；3）两阶段训练策略：先学习场景重建，再训练条件世界模型；4）增强的表示学习：通过空间-时间重建模型提高生成质量；5）单阶段训练策略：简化训练过程，降低计算成本。相比之前的工作，CVD-STORM实现了真正的端到端交互（不同于MagicDrive3D的两阶段流水线），提供绝对深度估计（不同于UniFuture和GEM的相对深度），重建过程对生成模型有直接影响，并能同时完成多视角视频生成和4D场景重建。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'CVD-STORM通过引入STORM-VAE和统一的生成-重建框架，实现了高质量的多视角视频生成和准确的4D场景重建，为自动驾驶提供了更强大的世界模型。'}


### 论文摘要

Generative models have been widely applied to world modeling for environment simulation and future state prediction. With advancements in autonomous driving, there is a growing demand not only for high-fidelity video generation under various controls, but also for producing diverse and meaningful information such as depth estimation. To address this, we propose CVD-STORM, a cross-view video diffusion model utilizing a spatial-temporal reconstruction Variational Autoencoder (VAE) that generates long-term, multi-view videos with 4D reconstruction capabilities under various control inputs. Our approach first fine-tunes the VAE with an auxiliary 4D reconstruction task, enhancing its ability to encode 3D structures and temporal dynamics. Subsequently, we integrate this VAE into the video diffusion process to significantly improve generation quality. Experimental results demonstrate that our model achieves substantial improvements in both FID and FVD metrics. Additionally, the jointly-trained Gaussian Splatting Decoder effectively reconstructs dynamic scenes, providing valuable geometric information for comprehensive scene understanding. Our project page is https://sensetime-fvg.github.io/CVD-STORM.

---

## 54. VTimeCoT: Thinking by Drawing for Video Temporal Grounding and Reasoning

**论文链接:** [http://arxiv.org/abs/2510.14672v1](http://arxiv.org/abs/2510.14672v1)

**作者:** Jinglei Zhang, Yuanfan Guo, Rolandos Alexandros Potamias, Jiankang Deng, Hang Xu, Chao Ma

**发布时间:** 2025-10-16

**备注:** Accepted by ICCV 2025

### GPT解析

### 总结

论文提出了VTimeCoT框架，一种无需训练的方法，用于解决多模态大语言模型在视频时序定位和推理方面的缺陷，通过引入进度条视觉工具和跨模态推理过程，实现了显著的性能提升和可解释的推理过程。

### 背景

视频问答基于多模态大语言模型近年来受到关注，但这类模型在视频时序定位和推理方面存在明显缺陷，对有效现实世界视频理解系统的发展构成挑战。

### 目的

设计一个简单而有效的无需训练框架，用于高性能的视频时序定位和推理，解决现有模型的局限性。

### 方法

提出VTimeCoT框架，包含两个新颖的视觉工具：即插即用的进度条集成工具和高效率高亮工具，同时引入整合视频和文本跨模态推理的视觉时序思考链过程。

### 主要发现

在视频时序定位和基于推理的问答任务中，该方法对Qwen2VL-7B和GPT4o基线模型显示出显著的性能改进。

### 结论

所提出的框架实现了可组合且可解释的推理过程，有效提升了视频理解系统的性能。

### 翻译

近年来，基于多模态大语言模型(MLLM)的视频问答因其受益于LLMs的显著进步而受到广泛关注。然而，这些模型在视频时序定位和推理领域存在明显缺陷，对有效现实世界视频理解系统的发展构成挑战。受人类使用视频播放器与进度条交互以理解视频的启发，我们引入了VTimeCoT，一个简单而有效的无需训练框架，专为高性能视频定位和推理而设计。该框架包含两个新颖的进度条视觉工具：即插即用的进度条集成工具和高效率高亮工具。此外，为解决传统基于文本的思考链(CoT)方法的局限性，我们引入了一个整合视频和文本跨模态推理的视觉时序思考链过程。我们的方法在视频时序定位和基于推理的问答任务中，对Qwen2VL-7B和GPT4o基线模型均显示出显著的性能改进。最后，我们展示了所提出的框架实现了可组合且可解释的推理过程。项目页面：https://vtimecot.github.io


### 论文摘要

In recent years, video question answering based on multimodal large language models (MLLM) has garnered considerable attention, due to the benefits from the substantial advancements in LLMs. However, these models have a notable deficiency in the domains of video temporal grounding and reasoning, posing challenges to the development of effective real-world video understanding systems. Inspired by how humans use video players to interact with the progress bar for video comprehension, we introduce VTimeCoT, a simple yet effective training-free framework, designed for high-performance video grounding and reasoning. The proposed framework incorporates two novel visual tools of the progress bar: a plug-and-play progress bar integration tool and a high-efficiency highlighting tool. In addition, to address the limitations of conventional text-based chain-of-thought (CoT) approaches, we introduce a visuotemporal CoT process that integrates cross-modality reasoning across both video and text. Our approach demonstrates significant performance improvements on both Qwen2VL-7B and GPT4o baselines in tasks of video temporal grounding and reasoning-based question answering. Finally, we showcase that the proposed framework achieves a compositional and interpretable reasoning process. Project page: https://vtimecot.github.io

---

## 55. Vgent: Graph-based Retrieval-Reasoning-Augmented Generation For Long Video Understanding

**论文链接:** [http://arxiv.org/abs/2510.14032v1](http://arxiv.org/abs/2510.14032v1)

**作者:** Xiaoqian Shen, Wenxuan Zhang, Jun Chen, Mohamed Elhoseiny

**发布时间:** 2025-10-15

**备注:** NeurIPS 2025 (Spotlight). Webpage at  https://xiaoqian-shen.github.io/Vgent

### GPT解析

### 总结

本文提出了Vgent，一种基于图的检索-推理-增强生成框架，用于增强大型视频语言模型对长视频的理解能力。通过结构化图表示视频和引入中间推理步骤，有效解决了长视频处理中的时间依赖性问题和检索噪声问题，在多个基准测试上取得了显著性能提升。

### 背景

理解和推理长视频对大型视频语言模型(LVLMs)构成重大挑战，主要因为难以处理超出上下文窗口密集的视频token，并保留长期顺序信息。检索增强生成(RAG)虽然对处理长上下文有效，但应用于长视频时面临时间依赖性被打乱和包含无关信息等问题。

### 目的

开发一种框架增强LVLMs对长视频的理解能力，解决长视频处理中的时间依赖性问题和检索噪声问题，提高模型在长视频理解任务中的准确性和上下文感知能力。

### 方法

提出Vgent框架，包含两个关键创新：(1)使用结构化图表示视频，保留视频片段间的语义关系以提高检索效果；(2)引入中间推理步骤，利用结构化验证减少检索噪声，促进相关信息片段的显式聚合。

### 主要发现

在MLVU基准测试上，与基础模型相比，总体性能提升了3.0%~5.4%，并比最先进的视频RAG方法高出8.6%。代码已在https://xiaoqian-shen.github.io/Vgent公开。

### 结论

Vgent框架通过结构化图表示和中间推理步骤，有效解决了长视频理解中的关键挑战，显著提升了LVLMs的性能，为长视频理解任务提供了新的解决方案。

### 翻译

理解和推理长视频对大型视频语言模型(LVLMs)构成重大挑战，因为难以处理超出上下文窗口密集的视频token并保留长期顺序信息。检索增强生成(RAG)在处理大型语言模型(LLMs)的长上下文方面已显示出有效性；然而，将RAG应用于长视频面临时间依赖性被打乱和包含无关信息等挑战，这些都会妨碍准确推理。为解决这些局限性，我们提出了Vgent，一种新颖的基于图的检索-推理-增强生成框架，用于增强LVLMs对长视频的理解能力。我们的方法引入了两个关键创新：(i)它通过保留视频片段间的语义关系，使用结构化图表示视频，以提高检索效果。(ii)它引入中间推理步骤，缓解LVLMs的推理局限性，利用结构化验证减少检索噪声，促进相关信息的显式聚合，从而产生更准确和上下文感知的响应。我们在三个长视频理解基准测试上使用各种开源LVLMs全面评估了我们的框架。与基础模型相比，我们的方法在MLVU上总体性能提升了3.0%~5.4%，并比最先进的视频RAG方法高出8.6%。我们的代码已在https://xiaoqian-shen.github.io/Vgent公开。


### 论文摘要

Understanding and reasoning over long videos pose significant challenges for large video language models (LVLMs) due to the difficulty in processing intensive video tokens beyond context window and retaining long-term sequential information. Retrieval-Augmented Generation (RAG) has demonstrated effectiveness in processing long context for Large Language Models (LLMs); however, applying RAG to long video faces challenges such as disrupted temporal dependencies and inclusion of irrelevant information that can hinder accurate reasoning. To address these limitations, we propose Vgent, a novel graph-based retrieval-reasoning-augmented generation framework to enhance LVLMs for long video understanding. Our approach introduces two key innovations: (i) It represents videos by structured graphs with semantic relationships across video clips preserved to improve retrieval effectiveness. (ii) It introduces an intermediate reasoning step to mitigate the reasoning limitation of LVLMs, which leverages structured verification to reduce retrieval noise and facilitate the explicit aggregation of relevant information across clips, resulting in more accurate and context-aware responses. We comprehensively evaluate our framework with various open-source LVLMs on three long-video understanding benchmarks. Our approach yielded an overall performance improvement of $3.0\%\sim 5.4\%$ over base models on MLVU, and outperformed state-of-the-art video RAG methods by $8.6\%$. Our code is publicly available at https://xiaoqian-shen.github.io/Vgent.

---

## 56. SVAG-Bench: A Large-Scale Benchmark for Multi-Instance Spatio-temporal Video Action Grounding

**论文链接:** [http://arxiv.org/abs/2510.13016v2](http://arxiv.org/abs/2510.13016v2)

**作者:** Tanveer Hannan, Shuaicong Wu, Mark Weber, Suprosanna Shit, Jindong Gu, Rajat Koner, Aljoša Ošep, Laura Leal-Taixé, Thomas Seidl

**发布时间:** 2025-10-14

### GPT解析

### 总结

这项研究提出了时空视频动作定位(SVAG)任务，要求模型同时检测、跟踪和基于自然语言描述对视频中的相关对象进行时空定位。研究团队构建了SVAG-Bench基准数据集，提出了SVAGFormer基线框架，并开发了SVAGEVal评估工具。实验表明现有模型在SVAG任务上表现不佳，特别是在密集或复杂场景中。

### 背景

细粒度动作理解和准确时空定位是推进下一代AI系统的基本能力，包括具身智能体、自主平台和人机交互框架。尽管视频理解最近有所进展，但现有方法主要解决粗粒度动作识别或通用目标跟踪问题，忽略了根据动作联合检测和跟踪多个对象并进行时空定位的挑战。

### 目的

解决现有方法在联合检测、跟踪和时空定位视频中的相关对象方面的不足，推进细粒度动作理解和对象-动作交互的推理能力。

### 方法

提出了时空视频动作定位(SVAG)任务，构建了SVAG-Bench大型基准数据集（包含688个视频、19,590条标注记录和903个独特动词），提出了SVAGFormer基线框架（适应最先进的视觉语言模型进行联合时空定位），并开发了SVAGEVal标准化评估工具包。

### 主要发现

实验结果表明，现有模型在SVAG任务上表现不佳，特别是在密集或复杂场景中，这凸显了对长视频中细粒度对象-动作交互进行更高级推理的必要性。

### 结论

该研究为细粒度视频理解和对象-动作交互建立了新的基准和评估框架，强调了开发能够处理复杂场景和长视频中高级推理能力的模型的重要性。

### 翻译

理解细粒度动作并准确定位其在空间和时间中对应的执行者是推进下一代AI系统的基本能力，包括具身智能体、自主平台和人机交互框架。尽管视频理解最近有所进展，但现有方法主要解决粗粒度动作识别或通用目标跟踪问题，因此忽略了根据动作联合检测和跟踪多个对象并进行时空定位的挑战。为解决这一差距，我们引入了时空视频动作定位(SVAG)，这是一个新任务，要求模型基于自然语言描述的动作同时检测、跟踪和时空定位视频中所有相关对象。为支持此任务，我们构建了SVAG-Bench，这是一个大规模基准，包含688个视频、19,590条标注记录和903个独特动词，涵盖了多样化的对象、动作和现实世界场景。我们进一步提出了SVAGFormer，这是一个基线框架，它适应了最先进的视觉语言模型进行联合时空定位，并引入了SVAGEVal，这是一个标准化的评估工具包，用于公平和可复现的基准测试。实验结果表明，现有模型在SVAG上表现不佳，特别是在密集或复杂场景中，这凸显了需要对长视频中细粒度对象-动作交互进行更高级推理的必要性。


### 论文摘要

Understanding fine-grained actions and accurately localizing their corresponding actors in space and time are fundamental capabilities for advancing next-generation AI systems, including embodied agents, autonomous platforms, and human-AI interaction frameworks. Despite recent progress in video understanding, existing methods predominantly address either coarse-grained action recognition or generic object tracking, thereby overlooking the challenge of jointly detecting and tracking multiple objects according to their actions while grounding them temporally. To address this gap, we introduce Spatio-temporal Video Action Grounding (SVAG), a novel task that requires models to simultaneously detect, track, and temporally localize all referent objects in videos based on natural language descriptions of their actions. To support this task, we construct SVAG-Bench, a large-scale benchmark comprising 688 videos, 19,590 annotated records, and 903 unique verbs, covering a diverse range of objects, actions, and real-world scenes. We further propose SVAGFormer, a baseline framework that adapts state of the art vision language models for joint spatial and temporal grounding, and introduce SVAGEval, a standardized evaluation toolkit for fair and reproducible benchmarking. Empirical results show that existing models perform poorly on SVAG, particularly in dense or complex scenes, underscoring the need for more advanced reasoning over fine-grained object-action interactions in long videos.

---

## 57. K-frames: Scene-Driven Any-k Keyframe Selection for long video understanding

**论文链接:** [http://arxiv.org/abs/2510.13891v1](http://arxiv.org/abs/2510.13891v1)

**作者:** Yifeng Yao, Yike Yun, Jing Wang, Huishuai Zhang, Dongyan Zhao, Ke Tian, Zhihao Wang, Minghui Qiu, Tao Wang

**发布时间:** 2025-10-14

### GPT解析

### 总结

多模态大语言模型在长视频理解方面面临上下文窗口和计算成本限制，现有关键帧选择方法存在信息丢失和场景连续性问题。作者提出K-frames方法，通过预测语义连贯的视频片段而非单个帧，保持时间连续性，支持灵活的多尺度关键帧选择。

### 背景

多模态大语言模型在图像理解方面表现出色，但在长视频理解方面受到上下文窗口和计算成本的限制。均匀采样通常会导致大量信息丢失。

### 目的

解决现有关键帧选择方法的问题，提出一种能够保持时间连续性的场景驱动关键帧选择方法。

### 方法

K-frames方法预测语义连贯、与查询相关的视频片段而非单个帧，支持任意数量的关键帧选择。作者构建了包含20万个基于查询条件的视频亮点的PeakClips数据集，并采用三阶段渐进式课程学习：两个监督微调阶段（时间定位和关键片段感知）和一个强化学习阶段（优化场景驱动的预测策略）。

### 主要发现

在主要的长视频理解基准上的大量实验表明，K-frames在各种规模的关键帧选择方面提供了有效、可解释且即插即用的解决方案。

### 结论

K-frames方法解决了长视频理解中的关键帧选择问题，作者公开的数据集和模型将为该领域提供支持。

### 翻译

多模态大语言模型在图像理解方面已展现出显著能力，但在长视频处理中受限于上下文窗口和计算成本。均匀采样常导致大量信息丢失。同时，现有的关键帧选择方法如文本-帧检索或基于强化学习的帧优化通常产生稀疏且时间上不连续的帧，忽略了场景连续性，缺乏多尺度帧选择的灵活性。为解决这些问题，我们引入K-frames，一种保持时间连续性的场景驱动关键帧选择新范式。K-frames不选择单个帧，而是预测语义连贯、与查询相关的片段，支持任意数量的关键帧选择以满足不同用户需求。为实现这一方法，我们首先引入PeakClips数据集，包含20万个基于查询条件的视频亮点。基于此数据集，K-frames使用三阶段渐进式课程学习clip2frame选择，包括两个监督微调阶段（用于时间定位和关键片段感知）和一个强化学习阶段（直接优化场景驱动的预测策略，无需额外注释）。在主要长视频理解基准上的大量实验表明，K-frames为各种规模的关键帧选择提供了有效、可解释且即插即用的解决方案。我们的数据集和模型将会公开。


### 论文摘要

Multimodal Large Language Models (MLLMs) have demonstrated significant capabilities in image understanding, but long-video are constrained by context windows and computational cost. Uniform frame sampling often leads to substantial information loss. Meanwhile existing keyframe selection methods such as text-frame retrieval or RL-based frame optimization typically yield sparse and temporally disjointed frames, overlooking scene continuity and lacking flexibility for multi-scale frame selection. To address these limitations, we introduce K-frames, a novel paradigm for scene-driven keyframe selection that preserves temporal continuity. Instead of selecting individual frames, K-frames predicts semantically coherent, query-relevant clips, which enables any-k keyframes selection to meet diverse user budgets. To achieve this approach, we first introduce PeakClips, a dataset of 200K video highlights conditioned by query. Building on this dataset, K-frames learns clip2frame selection using a three-stage progressive curriculum. It involves two Supervised Fine-Tuning stages for temporal grounding and key-clip perception, followed by a Reinforcement Learning stage that directly optimizes the scene-driven prediction policy for downstream task without further annotations. Extensive experiments on major long-video understanding benchmarks demonstrate that K-frames provides an effective, interpretable, and plug-and-play solution for keyframe selection at various scales. Our dataset and model will be available.

---

## 58. ChangingGrounding: 3D Visual Grounding in Changing Scenes

**论文链接:** [http://arxiv.org/abs/2510.14965v1](http://arxiv.org/abs/2510.14965v1)

**作者:** Miao Hu, Zhiwei Huang, Tai Wang, Jiangmiao Pang, Dahua Lin, Nanning Zheng, Runsen Xu

**发布时间:** 2025-10-16

**备注:** 30 pages

### GPT解析

### 总结

论文提出了ChangingGrounding基准和Mem-ChangingGrounder方法，用于解决动态场景中3D视觉目标定位问题，通过利用过去观察信息减少探索成本。

### 背景

现实世界机器人需要从自然语言指令中定位物体，同时周围场景不断变化。现有3D视觉目标定位方法假设已重建且最新的点云，这导致需要昂贵的重新扫描，阻碍了实际部署。

### 目的

将3DVG表述为主动的、内存驱动的问题，引入ChangingGrounding基准来衡量代理如何有效利用过去观察、只在需要处探索，并在变化场景中提供精确3D边界框。

### 方法

提出Mem-ChangingGrounder零样本方法，结合跨模态检索与轻量级多视图融合：识别物体类型、检索相关记忆指导动作、高效探索目标、操作无效时回退、多视图扫描目标、融合多视图证据获取准确边界框。

### 主要发现

在ChangingGrounding基准上评估不同基线方法，Mem-ChangingGrounder实现最高定位精度，同时显著降低探索成本。

### 结论

希望该基准和方法能推动面向实际应用的、以内存为中心的3DVG研究转变。

### 翻译

现实世界中的机器人从自然语言指令中定位物体，同时周围场景不断变化。然而，大多数现有的3D视觉目标定位方法仍然假设已重建且最新的点云，这种假设迫使昂贵的重新扫描并阻碍部署。我们认为3DVG应表述为主动的、内存驱动的问题，并引入ChangingGrounding，这是第一个明确衡量代理如何有效利用过去观察、只在需要处探索并在变化场景中提供精确3D边界框的基准。为设定强参考点，我们还提出了Mem-ChangingGrounder，这是一种针对此任务的零样本方法，它结合了跨模态检索与轻量级多视图融合：识别查询暗示的物体类型，检索相关记忆指导动作，然后在场景中高效探索目标，在先前操作无效时回退，对目标进行多视图扫描，并将多视图扫描的融合证据投影以获得准确的物体边界框。我们在ChangingGrounding上评估了不同的基线方法，我们的Mem-ChangingGrounder实现了最高的定位精度，同时大大减少了探索成本。我们希望这个基准和方法能够推动面向实际应用的、以内存为中心的3DVG研究转变。项目页面：https://hm123450.github.io/CGB/。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决3D视觉定位在动态变化场景中的挑战。现有方法假设场景静态且拥有完整点云，但真实环境中物体会移动或被遮挡，导致机器人需要频繁重新扫描整个场景，这非常耗时且成本高昂。这个问题在现实中很重要，因为它限制了机器人在动态环境（如家庭、办公室）中的实用性，增加了能耗并降低了效率，而人类却能利用过去记忆快速适应变化环境。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': "作者从人类认知方式获得灵感，人类在动态环境中会利用过去记忆高效定位目标。作者将3D视觉定位重新定义为'主动的、记忆驱动的问题'。他们借鉴了VLM-Grounder的框架（使用2D图像而非点云）、3RScan数据集（提供不同时间点的场景扫描和物体对应关系），以及视觉语言模型和开放词汇检测器等现有技术。设计的Mem-ChangingGrounder方法结合了记忆检索、智能探索和回退策略，以应对场景变化。", '这个方法的核心思想是什么？整体实现流程是怎样的？': "核心思想是利用机器人对过去场景的记忆来指导在当前变化场景中的高效探索，避免盲目扫描整个场景。整体流程包括：1)查询分类：将查询分为'可验证'（即使目标移动，记忆中的目标仍匹配查询）和'不可验证'（记忆中的目标可能不再匹配）；2)记忆检索与定位：根据查询类型选择策略，使用全景扫描或空间关系感知扫描寻找目标；3)回退策略：当主策略失败时，从记忆检索目标类别并进行360度搜索；4)多视图投影：结合多视图信息生成精确3D边界框。", '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)首次定义动态场景中的3D视觉定位任务，强调利用过去记忆；2)提出ChangingGrounding基准数据集，包含267K个参照性描述，评估定位准确性和探索成本；3)设计Mem-ChangingGrounder方法，结合记忆检索和智能探索；4)引入探索成本指标，强调效率。相比之前工作，本文不再假设场景静态，而是设计基于智能体的方法，利用2D图像和记忆避免昂贵的点云重建，同时关注准确性和效率的平衡。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '本文提出了首个面向动态场景的3D视觉定位基准和方法，通过结合记忆检索和智能探索策略，实现了在变化环境中高效且准确的物体定位。'}


### 论文摘要

Real-world robots localize objects from natural-language instructions while scenes around them keep changing. Yet most of the existing 3D visual grounding (3DVG) method still assumes a reconstructed and up-to-date point cloud, an assumption that forces costly re-scans and hinders deployment. We argue that 3DVG should be formulated as an active, memory-driven problem, and we introduce ChangingGrounding, the first benchmark that explicitly measures how well an agent can exploit past observations, explore only where needed, and still deliver precise 3D boxes in changing scenes. To set a strong reference point, we also propose Mem-ChangingGrounder, a zero-shot method for this task that marries cross-modal retrieval with lightweight multi-view fusion: it identifies the object type implied by the query, retrieves relevant memories to guide actions, then explores the target efficiently in the scene, falls back when previous operations are invalid, performs multi-view scanning of the target, and projects the fused evidence from multi-view scans to get accurate object bounding boxes. We evaluate different baselines on ChangingGrounding, and our Mem-ChangingGrounder achieves the highest localization accuracy while greatly reducing exploration cost. We hope this benchmark and method catalyze a shift toward practical, memory-centric 3DVG research for real-world applications. Project page: https://hm123450.github.io/CGB/ .

---

## 59. RL-100: Performant Robotic Manipulation with Real-World Reinforcement Learning

**论文链接:** [http://arxiv.org/abs/2510.14830v1](http://arxiv.org/abs/2510.14830v1)

**作者:** Kun Lei, Huanyu Li, Dongjie Yu, Zhenyu Wei, Lingxiao Guo, Zhennan Jiang, Ziyu Wang, Shiyu Liang, Huazhe Xu

**发布时间:** 2025-10-16

**备注:** https://lei-kun.github.io/RL-100/

### GPT解析

### 总结

论文提出了RL-100，一个基于扩散视觉运动策略的真实世界强化学习训练框架，通过三阶段流程实现高效可靠的机器人操作，并在多个任务上达到100%成功率。

### 背景

家庭和工厂中的真实世界机器人操作需要可靠性、效率和鲁棒性，达到或超越熟练人类操作员的水平。

### 目的

开发一个真实世界的强化学习训练框架，实现高效、可靠且通用的机器人操作能力。

### 方法

RL-100框架采用三阶段流程：首先通过模仿学习利用人类先验知识；其次使用离线策略评估(OPE)进行迭代离线强化学习；最后通过在线强化学习消除剩余失败模式。此外，添加轻量级一致性蒸馏头将多步采样压缩为单步策略，实现高频控制并降低延迟。

### 主要发现

在七个真实机器人任务上评估，包括动态刚体控制、流体倾倒、布料折叠、拧螺丝和橙汁制作等，RL-100实现了900/900的100%成功率，包括连续250次试验全部成功。该方法达到接近人类远程操作或更好的时间效率，并展示了多小时的鲁棒性，可连续运行长达两小时。

### 结论

RL-100是一个与任务、具身和表示无关的通用框架，支持多种输入和机器人平台，能够实现与人类相当或更好的机器人操作性能。

### 翻译

家庭和工厂中的真实世界机器人操作需要可靠性、效率和鲁棒性，达到或超越熟练人类操作员的水平。我们提出了RL-100，一个基于通过监督学习训练的扩散视觉运动策略构建的真实世界强化学习训练框架。RL-100引入了一个三阶段流程。首先，模仿学习利用人类先验知识。其次，迭代离线强化学习使用离线策略评估(OPE)程序来筛选PPO风格的更新，并在去噪过程中应用这些更新，以实现保守可靠的改进。第三，在线强化学习消除剩余的失败模式。此外，添加的轻量级一致性蒸馏头将扩散中的多步采样过程压缩为单步策略，实现了高频控制，同时延迟减少一个数量级，并保留了任务性能。该框架与任务、具身和表示无关，支持3D点云和2D RGB输入，各种机器人平台，以及单步和动作块策略。我们在七个真实机器人任务上评估了RL-100，包括动态刚体控制（如推-T和敏捷保龄球）、流体和颗粒倾倒、可变形布料折叠、精确灵巧拧螺丝和多阶段橙汁制作。RL-100在总共900个评估试验中实现了100%成功率，包括在一个任务上连续250次试验全部成功。该方法实现了接近人类远程操作或更好的时间效率，并展示了多小时的鲁棒性，不间断运行时间长达两小时。


### 论文摘要

Real-world robotic manipulation in homes and factories demands reliability, efficiency, and robustness that approach or surpass skilled human operators. We present RL-100, a real-world reinforcement learning training framework built on diffusion visuomotor policies trained bu supervised learning. RL-100 introduces a three-stage pipeline. First, imitation learning leverages human priors. Second, iterative offline reinforcement learning uses an Offline Policy Evaluation procedure, abbreviated OPE, to gate PPO-style updates that are applied in the denoising process for conservative and reliable improvement. Third, online reinforcement learning eliminates residual failure modes. An additional lightweight consistency distillation head compresses the multi-step sampling process in diffusion into a single-step policy, enabling high-frequency control with an order-of-magnitude reduction in latency while preserving task performance. The framework is task-, embodiment-, and representation-agnostic and supports both 3D point clouds and 2D RGB inputs, a variety of robot platforms, and both single-step and action-chunk policies. We evaluate RL-100 on seven real-robot tasks spanning dynamic rigid-body control, such as Push-T and Agile Bowling, fluids and granular pouring, deformable cloth folding, precise dexterous unscrewing, and multi-stage orange juicing. RL-100 attains 100\% success across evaluated trials for a total of 900 out of 900 episodes, including up to 250 out of 250 consecutive trials on one task. The method achieves near-human teleoperation or better time efficiency and demonstrates multi-hour robustness with uninterrupted operation lasting up to two hours.

---

## 60. Leveraging Neural Descriptor Fields for Learning Contact-Aware Dynamic Recovery

**论文链接:** [http://arxiv.org/abs/2510.14768v1](http://arxiv.org/abs/2510.14768v1)

**作者:** Fan Yang, Zixuan Huang, Abhinav Kumar, Sergio Aguilera Marinovic, Soshi Iba, Rana Soltani Zarrin, Dmitry Berenson

**发布时间:** 2025-10-16

### GPT解析

### 总结

本研究提出了一种名为接触感知动态恢复(CADRE)的强化学习框架，用于在灵巧操作中处理意外错误和干扰，特别是接住下落物体并系统重置以恢复主要任务。

### 背景

现实世界中的灵巧操作经常遇到意外错误和干扰，可能导致灾难性故障，如掉落被操作物体。

### 目的

开发一种方法，在物体仍在抓取范围内时接住下落物体，并将系统重置为有利于恢复主要操作任务的配置。

### 方法

提出接触感知动态恢复(CADRE)框架，这是一个强化学习框架，集成了受神经描述场(NDF)启发的模块来提取隐式接触特征，直接推理手指-物体对应关系并适应不同物体几何形状。

### 主要发现

整合接触特征提高了训练效率，增强了强化学习的收敛性能，并最终导致更成功的恢复操作。

### 结论

CADRE框架可以零样本泛化到具有不同几何形状的未见物体上，证明了其在实际应用中的有效性和通用性。

### 翻译

现实世界中的灵巧操作经常遇到意外错误和干扰，可能导致灾难性故障，如掉落被操作物体。为了应对这一挑战，我们专注于在物体仍在抓取范围内时接住下落物体，并将系统重置为有利于恢复主要操作任务的配置。我们提出了接触感知动态恢复(CADRE)，这是一个强化学习框架，集成了受神经描述场(NDF)启发的模块来提取隐式接触特征。与仅依赖物体姿态或点云输入的方法相比，NDF可以直接推理手指-物体对应关系并适应不同的物体几何形状。实验表明，整合接触特征提高了训练效率，增强了强化学习的收敛性能，并最终导致更成功的恢复操作。此外，我们证明了CADRE可以零样本泛化到具有不同几何形状的未见物体上。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决机器人在灵巧操作中遇到意外错误和干扰时的恢复问题，特别是如何抓住掉落的物体并恢复到有利于继续主要操作任务的状态。这个问题在现实中很重要，因为机器人执行实际任务时经常遇到意外干扰，如螺丝卡住导致产生意外扭矩，可能导致物体掉落。仅仅抓住物体是不够的，还需要恢复到适合继续主要任务的状态，例如从强力抓取切换到精确抓取，这对实际应用至关重要。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先认识到动态恢复问题的重要性，特别是抓住掉落物体并恢复到有利于继续主要操作任务的状态。他们观察到接触在灵巧操作中的重要性，并认为保持不同物体几何形状间一致的接触行为是成功泛化的基本因素。作者借鉴了Neural Descriptor Fields (NDF)来提取隐式接触特征，NDF能够捕获3D坐标和物体点云之间的几何对应关系。他们设计了Contact-Aware Dynamic Recovery (CADRE)框架，将NDF特征作为隐式接触信息整合到强化学习中。同时，他们借鉴了强化学习(特别是PPO算法)、点云表示方法以及DexPoint中利用接触信息提高泛化的思想。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是利用神经描述场(NDF)提取隐式接触特征，使机器人能够推理手指-物体对应关系并适应不同物体几何形状，同时不仅关注抓住掉落物体，还关注恢复到有利于继续主要操作任务的状态。整体实现流程包括：1)使用预训练的NDF模型提取接触特征，在机器人手上预定义关键点并查询这些点的NDF特征形成抓取特征；2)设计强化学习框架，将观察(机器人关节角度、物体姿态、物体速度)和抓取特征结合作为输入，使用PPO算法优化策略，并设计多目标奖励函数；3)在螺丝刀和螺母恢复任务上评估方法，测试泛化能力，并在真实机器人硬件上部署验证。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)提出通过抓取进行恢复的问题，机器人不仅要抓住掉落的物体，还要实现能够无缝恢复主要操作任务的抓取配置；2)开发了基于NDF的隐式接触表示，用于接触丰富的灵巧操作，有效捕获手部和操作物体之间的几何对应关系；3)提出了用于动态恢复的强化学习框架，利用这种表示实现成功的抓取和有利于后续操作任务的状态；4)证明了这种接触表示能够在不同几何形状的动态恢复任务中实现有效的泛化。相比之前工作，我们的方法不仅关注稳定抓取，还考虑后续操作任务的需求；NDF可以直接推理手指-物体对应关系并适应不同物体几何形状；提供了比点云方法更全面的接触建模；相比DexPoint，我们的方法区分了应该接触和应该避免接触的区域。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '本文提出了CADRE，一种利用神经描述场提取隐式接触特征的强化学习框架，使机器人能够在灵巧操作中从掉落物体中恢复并回到有利于继续主要操作任务的状态，同时实现了对不同物体几何形状的零样本泛化。'}


### 论文摘要

Real-world dexterous manipulation often encounters unexpected errors and disturbances, which can lead to catastrophic failures, such as dropping the manipulated object. To address this challenge, we focus on the problem of catching a falling object while it remains within grasping range and, importantly, resetting the system to a configuration favorable for resuming the primary manipulation task. We propose Contact-Aware Dynamic Recovery (CADRE), a reinforcement learning framework that incorporates a Neural Descriptor Field (NDF)-inspired module to extract implicit contact features. Compared to methods that rely solely on object pose or point cloud input, NDFs can directly reason about finger-object correspondence and adapt to different object geometries. Our experiments show that incorporating contact features improves training efficiency, enhances convergence performance for RL training, and ultimately leads to more successful recoveries. Additionally, we demonstrate that CADRE can generalize zero-shot to unseen objects with different geometries.

---

## 61. A Generalized Placeability Metric for Model-Free Unified Pick-and-Place Reasoning

**论文链接:** [http://arxiv.org/abs/2510.14584v1](http://arxiv.org/abs/2510.14584v1)

**作者:** Benno Wingender, Nils Dengler, Rohit Menon, Sicong Pan, Maren Bennewitz

**发布时间:** 2025-10-16

### GPT解析

### 总结

本文提出了一种通用的可放置性度量方法，可以直接从嘈杂点云评估放置姿态，无需任何形状先验知识。该方法联合评分稳定性、可抓取性和间隙，实现无需模型的统一抓取-放置推理。

### 背景

在现实世界的传感噪声下可靠地抓取和放置未知物体仍然具有挑战性。现有方法依赖于强物体先验（如CAD模型）或平面支撑假设，限制了抓取和放置之间的泛化和统一推理能力。

### 目的

引入一种通用的可放置性度量方法，直接从嘈杂点云评估放置姿态，无需任何形状先验；实现统一抓取-放置推理；在未见过的真实物体和非平面物体支撑上提供准确的稳定性预测和物理合理的放置结果。

### 方法

引入通用的可放置性度量方法；从原始几何形状中提取物体的支撑表面；生成多样化的多方向放置候选；采样满足碰撞和稳定性约束的接触点；将抓取分数与每个候选放置相关联，实现无需模型的统一抓取-放置推理。

### 主要发现

在未见过的真实物体和非平面物体支撑上，该方法与CAD模型相当的准确性预测稳定性损失；比基于学习的方法产生更物理合理的放置结果。

### 结论

提出的方法能够实现无需模型的统一抓取-放置推理；在现实世界的噪声条件下，能够准确预测稳定性损失并产生物理合理的放置结果；克服了现有方法对强物体先验和平面支撑假设的依赖。

### 翻译

在现实世界的传感噪声下可靠地抓取和放置未知物体仍然是一项具有挑战性的任务，因为现有方法依赖于强物体先验（如CAD模型）或平面支撑假设，限制了抓取和放置之间的泛化和统一推理能力。在这项工作中，我们引入了一种通用的可放置性度量方法，直接从嘈杂点云评估放置姿态，无需任何形状先验。该度量方法联合评分稳定性、可抓取性和间隙。从原始几何形状中，我们提取物体的支撑表面，生成多样化的多方向放置候选，并采样满足碰撞和稳定性约束的接触点。通过将抓取分数与每个候选放置相关联，我们提出的方法实现了无需模型的统一抓取-放置推理，并选择导致稳定、无碰撞放置的抓取-放置对。在未见过的真实物体和非平面物体支撑上，我们的度量方法在预测稳定性损失方面提供了与CAD模型相当的准确性，并且通常比基于学习方法的预测器产生更物理合理的放置结果。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决在现实世界感知噪声下可靠地抓取和放置未知物体的问题。这个问题很重要，因为抓取和放置能力对仓库物流、家庭辅助和医疗保健等机器人应用至关重要，而现有方法依赖于强物体先验（如CAD模型）或平面支撑假设，限制了在复杂和噪声环境中的泛化能力。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先分析了现有方法的局限性：大多数方法使用强物体先验评估放置稳定性，或只评估少量预定义的放置姿态。作者借鉴了统一抓取和放置推理的思想，从点云处理中学习物体重建方法，并改进了稳定性评估以处理部分和噪声观测。新方法设计了一个通用的可放置性度量，直接从嘈杂点云评估放置姿态，融合物理可行性和机器人约束，实现无模型统一的抓取和放置推理。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是提出一个通用的可放置性度量，直接从传感器数据评估放置姿态，融合稳定性、放置条件下的可抓取性和间隙三个因素，通过统一评分抓取和放置候选，选择最佳组合。整体流程包括：1)感知阶段重建工作空间和物体点云；2)生成和评分候选抓取；3)生成多样化放置候选；4)计算可放置性评分（稳定性、PCG、间隙）；5)统一推理选择最佳抓取-放置组合进行执行。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点：1)通用可放置性度量，直接从嘈杂点云评估放置姿态；2)无模型统一抓取和放置推理，在共同物体框架中评估；3)物理有效性验证，在非平面支撑和边缘情况表现优异；4)任务驱动的放置评估，支持操作偏好。相比之前工作：不需要CAD模型或平面支撑假设；能处理边缘附近的物体；提供通用分数而非单一稳定平面；适合在线部署，不依赖重型预测网络。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '这篇论文提出了一种通用的可放置性度量，使机器人能够直接从嘈杂点云评估未知物体的稳定放置，实现无模型统一的抓取和放置推理，显著提高了在复杂环境中的操作成功率。'}


### 论文摘要

To reliably pick and place unknown objects under real-world sensing noise remains a challenging task, as existing methods rely on strong object priors (e.g., CAD models), or planar-support assumptions, limiting generalization and unified reasoning between grasping and placing. In this work, we introduce a generalized placeability metric that evaluates placement poses directly from noisy point clouds, without any shape priors. The metric jointly scores stability, graspability, and clearance. From raw geometry, we extract the support surfaces of the object to generate diverse candidates for multi-orientation placement and sample contacts that satisfy collision and stability constraints. By conditioning grasp scores on each candidate placement, our proposed method enables model-free unified pick-and-place reasoning and selects grasp-place pairs that lead to stable, collision-free placements. On unseen real objects and non-planar object supports, our metric delivers CAD-comparable accuracy in predicting stability loss and generally produces more physically plausible placements than learning-based predictors.

---

## 62. CALM-Net: Curvature-Aware LiDAR Point Cloud-based Multi-Branch Neural Network for Vehicle Re-Identification

**论文链接:** [http://arxiv.org/abs/2510.14576v1](http://arxiv.org/abs/2510.14576v1)

**作者:** Dongwook Lee, Sol Han, Jinwhan Kim

**发布时间:** 2025-10-16

**备注:** 10 pages, 7 figures

### GPT解析

### 总结

这篇论文提出了CALM-Net，一种基于曲率感知的激光雷达点云多分支神经网络，用于车辆重识别任务。

### 背景

车辆重识别面临的主要挑战是从三维点云中学习判别性和互补性特征来区分不同车辆。

### 目的

提出CALM-Net模型，通过整合曲率感知信息提高车辆重识别的准确性。

### 方法

采用多分支架构，整合了边缘卷积、点注意力和曲率嵌入（用于表征点云中的局部表面变化），学习丰富的几何和上下文特征。

### 主要发现

在nuScenes数据集上的实验表明，CALM-Net相比最强基线方法，平均重识别准确率提高了约1.97个百分点。

### 结论

将曲率信息整合到深度学习架构中并采用多分支特征学习，能有效提升基于激光雷达点云的车辆重识别性能。

### 翻译

这篇论文提出了CALM-Net，一种基于曲率感知的激光雷达点云多分支神经网络，用于车辆重识别。所提出的模型解决了从三维点云中学习判别性和互补性特征以区分车辆这一挑战。CALM-Net采用多分支架构，整合了边缘卷积、点注意力和曲率嵌入，后者用于表征点云中的局部表面变化。通过结合这些机制，模型学习更适合重识别任务的丰富几何和上下文特征。在大型nuScenes数据集上的实验评估表明，CALM-Net比我们研究中最强的基线方法平均提高了约1.97个百分点的重识别准确率。结果证实了将曲率信息整合到深度学习架构中的有效性，并突出了多分支特征学习对基于激光雷达点云的车辆重识别的益处。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决基于LiDAR点云的车辆重识别问题，即如何从三维点云数据中学习判别性特征来区分不同车辆。这个问题在智能交通系统中至关重要，因为它支持跨摄像头跟踪、交通分析和自动驾驶安全，能够解决传统运动跟踪在遮挡、轨迹碎片化等情况下的失败问题，同时点云数据提供准确的3D几何信息，相比摄像头数据对光照变化和视角变化具有更好的鲁棒性。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先分析了基于摄像头的方法在复杂环境下的局限性，认识到点云数据的优势。他们借鉴了车辆重识别领域的多种方法，包括视角感知学习、多分支特征融合和注意力机制，同时借鉴了机器人学中利用特征值确定车辆方向的思想。基于这些现有工作，作者设计了CALM-Net，整合边缘卷积（处理局部几何）、点注意力（捕获全局上下文）和曲率嵌入（编码表面变化）三种机制，并通过混合采样策略（训练时随机采样，推理时最远点采样）进一步提升了性能。', '这个方法的核心思想是什么？整体实现流程是怎样的？': 'CALM-Net的核心思想是通过多分支神经网络同时捕捉点云数据中的局部几何结构、全局上下文信息和表面曲率特征，学习对视角和环境变化鲁棒的车辆嵌入。整体流程包括：1)输入点云进行下采样；2)并行处理三个分支-边缘卷支提取局部几何、点注意力捕获全局依赖、曲率嵌入编码表面变化；3)将各分支特征融合并通过卷积和批归一化处理；4)通过ReLU激活得到最终嵌入；5)使用二元交叉熵损失进行训练，判断两个点云是否对应同一车辆。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)首次在LiDAR点云重识别中引入曲率感知机制，通过可学习的曲率嵌入实现细粒度几何推理；2)设计多分支架构，同时处理局部、上下文和结构特征；3)提出混合点下采样策略，结合随机采样的数据增强和FPS的结构保持优势；4)精心设计特征融合方法。相比之前工作，本文专注于点云而非图像数据，引入曲率嵌入这一新特征，采用多分支而非单一架构，并使用混合采样策略，在nuScenes数据集上实现了比最强基线高1.97%的准确率。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '本文提出了CALM-Net，一个结合边缘卷积、点注意力和曲率嵌入的多分支神经网络，通过从LiDAR点云中学习判别性和几何驱动的特征，显著提高了车辆重识别的准确率，特别是在复杂城市环境中的视角变化和光照变化情况下。'}


### 论文摘要

This paper presents CALM-Net, a curvature-aware LiDAR point cloud-based multi-branch neural network for vehicle re-identification. The proposed model addresses the challenge of learning discriminative and complementary features from three-dimensional point clouds to distinguish between vehicles. CALM-Net employs a multi-branch architecture that integrates edge convolution, point attention, and a curvature embedding that characterizes local surface variation in point clouds. By combining these mechanisms, the model learns richer geometric and contextual features that are well suited for the re-identification task. Experimental evaluation on the large-scale nuScenes dataset demonstrates that CALM-Net achieves a mean re-identification accuracy improvement of approximately 1.97\% points compared with the strongest baseline in our study. The results confirms the effectiveness of incorporating curvature information into deep learning architectures and highlight the benefit of multi-branch feature learning for LiDAR point cloud-based vehicle re-identification.

---

## 63. High-Order Meshfree Surface Integration, Including Singular Integrands

**论文链接:** [http://arxiv.org/abs/2510.14236v1](http://arxiv.org/abs/2510.14236v1)

**作者:** Daniel R. Venn, Steven J. Ruuth

**发布时间:** 2025-10-16

### GPT解析

### 总结

本研究开发并测试了针对表面点云的高阶积分方法，解决了在任意分段光滑表面上进行精确积分的问题。

### 背景

表面积分在工程和科学领域的多种应用中至关重要，特别是在涉及偏微分方程的各种积分方法中。基于网格的方法需要曲面网格才能实现高阶收敛，这在许多表面上难以可靠获得；而无网格方法通常需要在感兴趣域上精确积分一组函数，但这些积分在大多数表面上没有闭式形式。

### 目的

开发能够在任意、分段光滑表面（有边界或无边界）上进行高精度积分的方法，且不需要特定的点排列或初始三角剖分。

### 方法

作者提出了两种完全无网格的积分方法，适用于任意分段光滑表面。这些方法不需要特定的点排列或表面的初始三角剖分。此外，作者还展示了如何扩展这些方法以处理奇异积分。

### 主要发现

1. 开发了两种在任意分段光滑表面上进行积分的方法；2. 这些方法完全无网格，不需要特定的点排列或初始三角剖分；3. 方法可以处理奇异积分，同时保持高精度；4. 无需在奇点附近改变点密度即可维持高精度。

### 结论

所提出的方法为在任意表面上进行高阶积分提供了有效解决方案，克服了传统网格方法和无网格方法的局限性，并能处理奇异积分情况。

### 翻译

我们开发并测试了针对表面点云的高阶积分方法。在表面上积分函数的任务在工程和科学的一系列应用中出现，特别是在涉及偏微分方程的各种积分方法中。基于网格的方法需要曲面网格才能实现高阶收敛，这在许多表面上难以可靠获得，而大多数无网格方法需要在感兴趣域上精确积分一组函数（如径向基函数）；这些积分在大多数表面上通常没有闭式形式。我们描述了两种在任意、分段光滑表面（有边界或无边界）上进行积分的方法。我们的方法不需要特定的点排列或表面的初始三角剖分，使它们完全无网格。我们还展示了如何扩展这些方法以处理奇异积分，同时保持高精度，而无需在奇点附近改变点密度。


### 论文摘要

We develop and test high-order methods for integration on surface point clouds. The task of integrating a function on a surface arises in a range of applications in engineering and the sciences, particularly those involving various integral methods for partial differential equations. Mesh-based methods require a curved mesh for high-order convergence, which can be difficult to reliably obtain on many surfaces, and most meshfree methods require the ability to integrate a set of functions (such as radial basis functions) exactly on the domain of interest; these integrals are generally not known in closed form on most surfaces. We describe two methods for integrating on arbitrary, piecewise-smooth surfaces with or without boundary. Our approaches do not require a particular arrangement of points or an initial triangulation of the surface, making them completely meshfree. We also show how the methods can be extended to handle singular integrals while maintaining high accuracy without changing the point density near singularities.

---

## 64. Prescribed Performance Control of Deformable Object Manipulation in Spatial Latent Space

**论文链接:** [http://arxiv.org/abs/2510.14234v1](http://arxiv.org/abs/2510.14234v1)

**作者:** Ning Han, Gu Gong, Bin Zhang, Yuexuan Xu, Bohan Yang, Yunhui Liu, David Navarro-Alarcon

**发布时间:** 2025-10-16

### GPT解析

### 总结

本文提出了一种新型的无模型方法，用于带有关键点约束的三维可变形物体形状控制。该方法通过深度学习从点云中提取关键点作为特征向量，保留了物体的空间信息同时降低了特征空间维度。将操控问题简化为视觉伺服问题，使用变形雅可比矩阵描述形状动力学，并通过结合障碍李雅普诺夫函数的预设性能控制方法提高控制精度。实验验证了该方法的有效性和鲁棒性。

### 背景

操控三维可变形物体对机器人系统具有显著挑战，主要因为可变形物体具有无限维状态空间和复杂的变形动力学特性。

### 目的

提出一种新型的无模型方法，用于带有关键点约束的可变形物体形状控制，提高操控的准确性和鲁棒性。

### 方法

不同于依赖特征降维的现有方法，所提出的控制器利用从可变形物体点云中通过深度学习方法提取的关键点坐标作为特征向量。通过提取关键点，将可变形物体操控简化为视觉伺服问题，使用变形雅可比矩阵描述形状动力学。同时，开发了一种结合障碍李雅普诺夫函数的预设性能控制方法，以强制执行关键点的约束，提高控制精度。

### 主要发现

通过提取关键点，成功降低了特征空间维度同时保留了物体空间信息；结合障碍李雅普诺夫函数的预设性能控制方法有效提高了控制精度；实验结果验证了所提出方法的有效性和鲁棒性。

### 结论

所提出的无模型方法通过深度学习提取关键点并结合预设性能控制，有效解决了三维可变形物体形状控制问题，具有较好的应用前景。

### 翻译

操控三维可变形物体对机器人系统具有显著挑战，因为它们具有无限维状态空间和复杂的变形动力学。本文提出了一种新型的带有关键点约束的无模型形状控制方法。与依赖特征降维的现有方法不同，所提出的控制器利用从可变形物体点云中通过深度学习方法提取的关键点坐标作为特征向量。这种方法不仅降低了特征空间的维度，还保留了物体的空间信息。通过提取关键点，可变形物体的操控被简化为一个视觉伺服问题，其中形状动力学使用变形雅可比矩阵描述。为了提高控制精度，开发了一种结合障碍李雅普诺夫函数的预设性能控制方法，以强制执行关键点的约束。使用李雅普诺夫方法严格分析并验证了闭环系统的稳定性。实验结果进一步证明了所提出方法的有效性和鲁棒性。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决机器人操作三维可变形物体的控制挑战。可变形物体（如海绵、布料等）由于其形状可以无限变化，状态空间维度极高，且具有复杂的变形动力学特性，使得传统的机器人控制方法难以有效处理。这个问题在现实中非常重要，因为可变形物体操作在医疗手术、工业焊接、自动折叠衣物等领域有广泛应用，提高机器人对这类物体的操作能力可以扩展机器人在这些领域的应用，提高自动化水平，减少人工干预，并提高任务执行的精度和效率。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者在设计方法时，首先分析了现有可变形物体操作方法的局限性，包括基于模型的方法依赖物理模型但参数估计困难，以及无模型方法面临高维状态空间的挑战。作者借鉴了现有工作中的深度学习方法提取关键点、基于Jacobian的视觉伺服控制以及规定性能控制（PPC）和障碍Lyapunov函数（BLF）等技术。作者的创新点在于将PPC方法从已知Jacobian矩阵的视觉伺服任务迁移到Jacobian矩阵完全未知的可变形物体操作任务中，并结合关键点提取方法，在保留空间信息的同时降低特征维度。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是从可变形物体的3D点云中提取关键点作为特征向量，这些关键点保留了物体的空间信息同时降低了维度；将可变形物体操作简化为视觉伺服问题，使用变形Jacobian矩阵描述形状动力学；设计规定性能控制器，通过障碍Lyapunov函数强制执行关键点的约束；使用神经网络近似未知的Jacobian矩阵。整体实现流程包括：使用Key-Grid神经网络从点云中提取关键点；计算关键点误差；使用规定性能函数定义误差边界；将误差转换为转换误差；设计基于Jacobian的控制器，使用神经网络近似Jacobian矩阵；应用自适应律更新神经网络权重；通过Lyapunov分析确保系统稳定性。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：提出一种新的无模型方法，使用关键点坐标作为特征向量，保留了空间信息同时降低了维度；将规定性能控制方法从视觉伺服任务迁移到可变形物体操作任务，其中Jacobian矩阵完全未知；设计障碍Lyapunov函数来确保关键点误差的边界约束；结合深度学习和自适应控制方法，提高了控制精度和鲁棒性。相比之前工作，该方法直接从3D点云提取关键点，而不是使用手动标记的关键点；使用改进的PPC框架，而不是基于图网络的MPC控制器；与传统降维方法相比，保留了物理和空间信息；与其他避免潜在抽象的方法相比，不局限于二维结构、刚性假设或强模型依赖。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '这篇论文提出了一种基于空间潜在空间的关键点约束规定性能控制方法，有效解决了三维可变形物体操作中的高维状态空间和复杂变形动力学挑战，显著提高了控制精度和鲁棒性。'}


### 论文摘要

Manipulating three-dimensional (3D) deformable objects presents significant challenges for robotic systems due to their infinite-dimensional state space and complex deformable dynamics. This paper proposes a novel model-free approach for shape control with constraints imposed on key points. Unlike existing methods that rely on feature dimensionality reduction, the proposed controller leverages the coordinates of key points as the feature vector, which are extracted from the deformable object's point cloud using deep learning methods. This approach not only reduces the dimensionality of the feature space but also retains the spatial information of the object. By extracting key points, the manipulation of deformable objects is simplified into a visual servoing problem, where the shape dynamics are described using a deformation Jacobian matrix. To enhance control accuracy, a prescribed performance control method is developed by integrating barrier Lyapunov functions (BLF) to enforce constraints on the key points. The stability of the closed-loop system is rigorously analyzed and verified using the Lyapunov method. Experimental results further demonstrate the effectiveness and robustness of the proposed method.

---

## 65. Distributed-Memory Parallel Algorithms for Fixed-Radius Near Neighbor Graph Construction

**论文链接:** [http://arxiv.org/abs/2510.14147v1](http://arxiv.org/abs/2510.14147v1)

**作者:** Gabriel Raulet, Dmitriy Morozov, Aydin Buluc, Katherine Yelick

**发布时间:** 2025-10-15

**备注:** 11 pages, 5 figures, 3 tables

### GPT解析

### 总结

该研究提出了一种使用覆盖树的可扩展稀疏感知分布式内存算法，用于计算一般度量空间中的近邻图，在各种数据集和度量标准下表现出卓越的性能和并行扩展性。

### 背景

计算固定半径近邻图是许多数据分析算法的重要第一步。近邻图在某种度量下连接接近的点，为点云赋予组合结构。随着计算能力和数据获取方法的进步，各种大型科学数据集需要可扩展的解决方案来处理下游分析中的常见子程序。

### 目的

解决现有并行近邻搜索工作在精确解和非欧几里得度量方面的局限性，提供一个可扩展的稀疏感知分布式内存算法，用于计算一般度量空间中的近邻图。

### 方法

提出了一种使用覆盖树的可扩展稀疏感知分布式内存算法。提供了覆盖树构建的共享内存算法，并展示了其与最先进的固定半径搜索数据结构的竞争力。然后介绍了两种分布式内存算法：简单的点分区策略和空间分区策略，它们利用每个节点上的覆盖树算法。

### 主要发现

算法在各种真实和合成数据集上表现出并行扩展性，适用于传统和非传统度量。在包含一百万个点的真实世界高维数据集上，对于每个顶点平均70个邻居的图，使用1024个核心实现了高达678.34倍的速度提升；对于每个顶点平均500个邻居的图，使用4096个核心实现了高达1590.99倍的速度提升。

### 结论

该算法能够有效处理大规模数据集的近邻图计算，在多种数据集和度量标准下表现出良好的并行扩展性。

### 翻译

计算固定半径近邻图是许多数据分析算法的重要第一步。近邻图在某种度量下连接接近的点，为点云赋予组合结构。随着计算能力和数据获取方法的进步，各种大型科学数据集需要可扩展的解决方案来处理下游分析中的常见子程序。现有的并行近邻搜索工作在最近邻和近似最近邻搜索问题上取得了很大进展，特别关注欧几里得空间。然而，许多应用程序需要精确解和非欧几里得度量。本文提出了一种使用覆盖树的可扩展稀疏感知分布式内存算法，用于计算一般度量空间中的近邻图。我们提供了覆盖树构建的共享内存算法，并展示了其与最先进的固定半径搜索数据结构的竞争力。然后，我们介绍了用于近邻图问题的两种分布式内存算法：一种简单的点分区策略和一种空间分区策略，它们利用每个节点上的覆盖树算法。我们的算法在各种真实和合成数据集上表现出并行扩展性，适用于传统和非传统度量。在包含一百万个点的真实世界高维数据集上，对于每个顶点平均70个邻居的图，使用1024个核心实现了比最先进方法高达678.34倍的速度提升；对于每个顶点平均500个邻居的图，使用4096个核心实现了高达1590.99倍的速度提升。


### 论文摘要

Computing fixed-radius near-neighbor graphs is an important first step for many data analysis algorithms. Near-neighbor graphs connect points that are close under some metric, endowing point clouds with a combinatorial structure. As computing power and data acquisition methods advance, diverse sources of large scientific datasets would greatly benefit from scalable solutions to this common subroutine for downstream analysis. Prior work on parallel nearest neighbors has made great progress in problems like k-nearest and approximate nearest neighbor search problems, with particular attention on Euclidean spaces. Yet many applications need exact solutions and non-Euclidean metrics. This paper presents a scalable sparsity-aware distributed memory algorithm using cover trees to compute near-neighbor graphs in general metric spaces. We provide a shared-memory algorithm for cover tree construction and demonstrate its competitiveness with state-of-the-art fixed-radius search data structures. We then introduce two distributed-memory algorithms for the near-neighbor graph problem, a simple point-partitioning strategy and a spatial-partitioning strategy, which leverage the cover tree algorithm on each node. Our algorithms exhibit parallel scaling across a variety of real and synthetic datasets for both traditional and non-traditional metrics. On real world high dimensional datasets with one million points, we achieve speedups up to 678.34x over the state-of-the-art using 1024 cores for graphs with 70 neighbors per vertex (on average), and up to 1590.99x using 4096 cores for graphs with 500 neighbors per vertex (on average).

---

## 66. Geometric local parameterization for solving Hele-Shaw problems with surface tension

**论文链接:** [http://arxiv.org/abs/2510.14088v1](http://arxiv.org/abs/2510.14088v1)

**作者:** Zengyan Zhang, Wenrui Hao, John Harlim

**发布时间:** 2025-10-15

**备注:** 22 pages, 5 figures

### GPT解析

### 总结

本文介绍了一种解决二维Hele-Shaw自由边界问题（带有表面张力）的新型计算框架。该方法使用点云表示移动边界，无需全局参数化，并通过广义移动最小二乘法构建局部几何图表，实现高阶几何量近似。研究提供了严格的收敛分析，并通过数值实验验证了方法的有效性，展示了复杂形状在表面张力作用下向圆形平衡状态的正确演化。

### 背景

Hele-Shaw自由边界问题是流体力学中的重要问题，特别是在研究具有表面张力的界面动力学时。传统方法通常需要全局参数化来表示移动边界，这在处理复杂几何形状时存在局限性。

### 目的

开发一种新的计算框架，能够高效、准确地解决带有表面张力的二维Hele-Shaw自由边界问题，克服传统方法在处理复杂几何形状时的局限性，并实现高阶收敛精度。

### 方法

1. 使用点云表示移动边界，消除全局参数化的需求；2. 应用广义移动最小二乘法构建局部几何图表；3. 直接从点云数据高阶近似几何量（如曲率）；4. 使用局部参数化离散化控制边界积分方程；5. 包含奇异积分的解析公式；6. 进行严格的收敛分析，建立一致性和稳定性条件。

### 主要发现

1. 所提出的方法实现了高阶空间收敛；2. 获得了预期的时域收敛率；3. 误差界限与均匀采样点云数据大小、边界光滑度和数值积分规则阶数相关；4. 复杂初始形状在表面张力作用下正确演变为圆形平衡状态。

### 结论

该新型计算框架为解决二维Hele-Shaw自由边界问题提供了有效方法，点云表示和局部几何图表的构建使得方法能够处理复杂几何形状，同时保持高阶收敛精度。数值实验验证了理论分析的正确性和方法的有效性。

### 翻译

在这项工作中，我们介绍了一种解决带有表面张力的二维Hele-Shaw自由边界问题的新型计算框架。移动边界由点云表示，消除了对全局参数化的需求。我们的方法利用广义移动最小二乘法构建局部几何图表，能够直接从点云数据高阶近似几何量（如曲率）。这种局部参数化被系统地用于离散化控制边界积分方程，包括奇异积分的解析公式。我们为所提出的空间离散化提供了严格的收敛分析，在特定条件下建立了一致性和稳定性。导出的误差界限基于移动边界上均匀采样点云数据的大小、边界的光滑度和数值积分规则的阶数。数值实验验证了理论结果，展示了高阶空间收敛和预期的时域收敛率。通过复杂初始形状的模拟进一步说明了该方法的有效性，这些形状在表面张力的影响下正确地演变为圆形平衡状态。


### 论文摘要

In this work, we introduce a novel computational framework for solving the two-dimensional Hele-Shaw free boundary problem with surface tension. The moving boundary is represented by point clouds, eliminating the need for a global parameterization. Our approach leverages Generalized Moving Least Squares (GMLS) to construct local geometric charts, enabling high-order approximations of geometric quantities such as curvature directly from the point cloud data. This local parameterization is systematically employed to discretize the governing boundary integral equation, including an analytical formula of the singular integrals. We provide a rigorous convergence analysis for the proposed spatial discretization, establishing consistency and stability under certain conditions. The resulting error bound is derived in terms of the size of the uniformly sampled point cloud data on the moving boundary, the smoothness of the boundary, and the order of the numerical quadrature rule. Numerical experiments confirm the theoretical findings, demonstrating high-order spatial convergence and the expected temporal convergence rates. The method's effectiveness is further illustrated through simulations of complex initial shapes, which correctly evolve towards circular equilibrium states under the influence of surface tension.

---

## 67. 论文ID: 2510.14824v1

**错误:** 无法读取论文信息 - [Errno 2] No such file or directory: 'storage/result/2510.14824v1.json'

---

## 68. Contrastive Diffusion Alignment: Learning Structured Latents for Controllable Generation

**论文链接:** [http://arxiv.org/abs/2510.14190v1](http://arxiv.org/abs/2510.14190v1)

**作者:** Ruchi Sandilya, Sumaira Perez, Charles Lynch, Lindsay Victoria, Benjamin Zebley, Derrick Matthew Buchanan, Mahendra T. Bhati, Nolan Williams, Timothy J. Spellman, Faith M. Gunning, Conor Liston, Logan Grosenick

**发布时间:** 2025-10-16

### GPT解析

### 总结

本研究提出了ConDA(对比扩散对齐)框架，通过对比学习在扩散嵌入中组织潜在空间，使其与系统动力学对齐，从而实现更可控和可解释的生成操作。

### 背景

扩散模型在生成任务上表现出色，但其潜在空间没有被明确组织用于可解释的控制，限制了其在需要精确控制的应用中的使用。

### 目的

开发一种方法来组织扩散模型的潜在空间，使其能够支持忠实插值、外推和可控生成，同时保持生成质量。

### 方法

提出ConDA框架，应用对比学习于扩散嵌入中，将潜在几何结构与系统动力学对齐，使遍历方向反映潜在的动力学因素，并支持非线性轨迹遍历。

### 主要发现

在流体动力学、神经钙成像、治疗性神经刺激和面部表情等多个基准测试中，ConDA产生了比线性遍历和基于条件的基线更具可解释性的潜在表示，同时提高了可控性。

### 结论

扩散潜变量编码了与动力学相关的结构，但要有效利用这种结构，需要沿着潜在流形进行潜在组织和遍历。

### 翻译

扩散模型在生成方面表现出色，但它们的潜在空间没有被明确组织用于可解释的控制。我们引入了ConDA(对比扩散对齐)，这是一个在扩散嵌入中应用对比学习的框架，将潜在几何结构与系统动力学对齐。受最近进展的启发，这些进展表明对比目标可以恢复更多解缠和结构化的表示，ConDA组织扩散潜变量，使得遍历方向反映潜在的动力学因素。在这个对比结构化的空间中，ConDA支持非线性轨迹遍历，实现忠实插值、外推和可控生成。在流体动力学、神经钙成像、治疗性神经刺激和面部表情的基准测试中，ConDA与线性遍历和基于条件的基线相比，产生了具有改进可解释性的潜在表示。这些结果表明扩散潜变量编码了动力学相关的结构，但利用这种结构需要沿着潜在流形进行潜在组织和遍历。


### 论文摘要

Diffusion models excel at generation, but their latent spaces are not explicitly organized for interpretable control. We introduce ConDA (Contrastive Diffusion Alignment), a framework that applies contrastive learning within diffusion embeddings to align latent geometry with system dynamics. Motivated by recent advances showing that contrastive objectives can recover more disentangled and structured representations, ConDA organizes diffusion latents such that traversal directions reflect underlying dynamical factors. Within this contrastively structured space, ConDA enables nonlinear trajectory traversal that supports faithful interpolation, extrapolation, and controllable generation. Across benchmarks in fluid dynamics, neural calcium imaging, therapeutic neurostimulation, and facial expression, ConDA produces interpretable latent representations with improved controllability compared to linear traversals and conditioning-based baselines. These results suggest that diffusion latents encode dynamics-relevant structure, but exploiting this structure requires latent organization and traversal along the latent manifold.

---

## 69. ViTacGen: Robotic Pushing with Vision-to-Touch Generation

**论文链接:** [http://arxiv.org/abs/2510.14117v1](http://arxiv.org/abs/2510.14117v1)

**作者:** Zhiyuan Wu, Yijiong Lin, Yongqiang Zhao, Xuyang Zhang, Zhuo Chen, Nathan Lepora, Shan Luo

**发布时间:** 2025-10-15

### GPT解析

### 总结

ViTacGen是一个创新的机器人操作框架，通过视觉到触觉的生成解决了触觉传感器限制问题，结合视觉和生成触觉数据通过强化学习实现高性能的机器人推操作，在模拟和真实实验中表现出色，成功率高达86%。

### 背景

机器人推操作需要触觉反馈来捕捉末端执行器和物体之间的接触力和动力学，但真实触觉传感器面临高成本、脆弱性、校准困难和传感器差异等挑战，而仅基于视觉的策略难以获得满意性能。

### 目的

提出ViTacGen框架，用于视觉机器人推操作，在强化学习中实现视觉到触觉的生成，消除对高分辨率真实触觉传感器的依赖，实现视觉系统上的有效零样本部署。

### 方法

ViTacGen包含一个编码器-解码器视觉到触觉生成网络，直接从视觉图像序列生成接触深度图像（标准化触觉表示），以及一个基于视觉和生成触觉观察的对比学习融合视觉-触觉数据的强化学习策略。

### 主要发现

在模拟和真实世界实验中验证了方法的有效性，展示了其卓越的性能，成功率达到86%。

### 结论

ViTacGen框架能够在不依赖高分辨率触觉传感器的情况下实现有效的机器人推操作，通过视觉到触觉的生成实现了在视觉系统上的零样本部署。

### 翻译

机器人推操作是一种基础的操作任务，需要触觉反馈来捕捉末端执行器和物体之间的微妙接触力和动力学。然而，真实的触觉传感器通常面临硬件限制，如高成本和脆弱性，以及部署挑战，包括校准和不同传感器之间的差异，而仅基于视觉的策略难以获得令人满意的性能。受人类从视觉推断触觉状态能力的启发，我们提出了ViTacGen，一个新颖的机器人操作框架，专为视觉机器人推操作设计，在强化学习中实现视觉到触觉的生成，以消除对高分辨率真实触觉传感器的依赖，实现仅在视觉系统上的有效零样本部署。具体而言，ViTacGen包含一个编码器-解码器视觉到触觉生成网络，直接从视觉图像序列生成接触深度图像（标准化的触觉表示），随后是一个基于视觉和生成触觉观察的对比学习融合视觉-触觉数据的强化学习策略。我们在模拟和真实世界实验中都验证了我们方法的有效性，展示了其卓越的性能，成功率达到86%。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文解决的是机器人推动任务中依赖昂贵且脆弱的触觉传感器的问题。这个问题很重要，因为触觉反馈对捕捉物体间细微接触力和动态至关重要，但真实触觉传感器成本高、易损坏、需要精确校准，且不同传感器间存在差异，限制了高性能机器人操作系统的实际部署。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者受人类能从视觉推断触觉状态的启发，观察到现有方法要么依赖昂贵触觉传感器，要么仅使用视觉但性能不足。他们设计了编码器-解码器的视觉到触觉生成网络(VT-Gen)和强化学习策略网络(VT-Con)。借鉴了人类视觉-触觉交互能力、Soft Actor-Critic强化学习算法、MoCo对比学习框架、Tactile Gym模拟平台、注意力机制和VGG损失函数等现有工作。', '这个方法的核心思想是什么？整体实现流程是怎样的？': "核心思想是模拟人类从视觉推断触觉的能力，让机器人仅通过视觉'感知'触觉，使用生成的触觉接触深度图像作为标准化表示，并通过对比学习对齐视觉和触觉特征。整体流程：1)在模拟环境中收集配对的视觉和触觉数据；2)训练VT-Gen网络从视觉生成触觉深度图像；3)冻结VT-Gen，训练VT-Con强化学习策略；4)零样本部署到真实视觉-only机器人系统。", '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点：1)提出ViTacGen框架消除对触觉传感器的依赖；2)设计VT-Gen生成标准化触觉表示；3)提出VT-Con通过对比学习融合视觉-触觉特征；4)实现零样本部署；5)使用接触深度图解决传感器差异问题。不同之处：相比仅视觉方法提供更丰富感知；相比触觉传感器方法降低成本复杂度；相比简单特征拼接实现更有效跨模态对齐；相比校准方法具有更好通用性。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'ViTacGen通过模拟人类从视觉推断触觉的能力，实现了仅使用视觉信息的机器人精确推动，消除了对昂贵触觉传感器的依赖，同时保持了高性能操作能力。'}


### 论文摘要

Robotic pushing is a fundamental manipulation task that requires tactile feedback to capture subtle contact forces and dynamics between the end-effector and the object. However, real tactile sensors often face hardware limitations such as high costs and fragility, and deployment challenges involving calibration and variations between different sensors, while vision-only policies struggle with satisfactory performance. Inspired by humans' ability to infer tactile states from vision, we propose ViTacGen, a novel robot manipulation framework designed for visual robotic pushing with vision-to-touch generation in reinforcement learning to eliminate the reliance on high-resolution real tactile sensors, enabling effective zero-shot deployment on visual-only robotic systems. Specifically, ViTacGen consists of an encoder-decoder vision-to-touch generation network that generates contact depth images, a standardized tactile representation, directly from visual image sequence, followed by a reinforcement learning policy that fuses visual-tactile data with contrastive learning based on visual and generated tactile observations. We validate the effectiveness of our approach in both simulation and real world experiments, demonstrating its superior performance and achieving a success rate of up to 86\%.

---

## 70. CymbaDiff: Structured Spatial Diffusion for Sketch-based 3D Semantic Urban Scene Generation

**论文链接:** [http://arxiv.org/abs/2510.13245v2](http://arxiv.org/abs/2510.13245v2)

**作者:** Li Liang, Bo Miao, Xinyu Wang, Naveed Akhtar, Jordan Vice, Ajmal Mian

**发布时间:** 2025-10-15

**备注:** Accepted by NeurIPS 2025

### GPT解析

### 总结

本文提出了SketchSem3D数据集和Cylinder Mamba Diffusion (CymbaDiff)方法，用于从手绘草图生成高质量的3D室外语义场景。

### 背景

室外3D语义场景生成在都市仿真和自动驾驶等领域有重要应用，但该领域的发展受到缺乏公开可用、良好注释的数据集的制约。

### 目的

引入首个大规模基准数据集SketchSem3D，用于从抽象手绘草图和卫星图像的伪标记注释生成3D室外语义场景，并提出一种增强空间一致性的新方法。

### 方法

提出了Cylinder Mamba Diffusion (CymbaDiff)模型，该方法施加结构化的空间排序，明确捕获圆柱连续性和垂直层次结构，并保留生成的场景中的物理邻域关系和全局上下文。

### 主要发现

在SketchSem3D上的大量实验表明，CymbaDiff实现了优越的语义一致性、空间真实性和跨数据集泛化能力。

### 结论

SketchSem3数据集和CymbaDiff方法为室外3D语义场景生成提供了新的基准和解决方案，有助于推动该领域的发展。

### 翻译

室外3D语义场景生成为都市仿真和自动驾驶等应用生成真实且语义丰富的环境。然而，这一方向的发展受到缺乏公开可用、良好注释的数据集的限制。我们引入了SketchSem3D，这是第一个大规模基准，用于从抽象手绘草图和卫星图像的伪标记注释生成3D室外语义场景。SketchSem3D包含两个子集：基于语义的KITTI草图和基于KITTI-360的草图（包含LiDAR体素及其相应的草图和注释卫星图像），以实现标准化、严格和多样化的评估。我们还提出了圆柱形Mamba扩散模型（CymbaDiff），显著增强了室外3D场景生成的空间一致性。CymbaDiff施加结构化的空间排序，明确捕获圆柱连续性和垂直层次结构，并保留生成的场景中的物理邻域关系和全局上下文。在SketchSem3D上的大量实验表明，CymbaDiff实现了优越的语义一致性、空间真实性和跨数据集泛化能力。代码和数据集将在https://github.com/Lillian-research-hub/CymbaDiff上提供。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决基于草图的3D户外语义场景生成问题，特别是缺乏公开的大规模标注数据集和现有方法在户外场景中的局限性。这个问题在现实中非常重要，因为高质量的城市场景生成对自动驾驶模拟、城市规划等应用至关重要，而传统方法要么依赖昂贵的传感器数据，要么无法生成复杂且语义丰富的户外环境。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先分析了现有3D户外场景生成方法的局限性，如鸟瞰图(BEV)方法缺乏3D结构信息，多尺度方法计算复杂。他们借鉴了状态空间模型(SSMs)在图像处理和点云分析中的成功应用，结合扩散模型在生成任务中的优势。作者还利用了CLIP和SAM等现有模型进行数据集构建，但创新性地将这些技术整合到一个专门针对户外场景的框架中，通过圆柱坐标系统改进了空间表示。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是通过结构化空间扩散模型，结合笛卡尔和圆柱坐标系统的优势，增强3D场景生成的空间连贯性。整体流程包括：1)构建SketchSem3D数据集，包含草图、卫星图像和3D体素；2)使用场景结构估计网络(SSEN)提取结构信息；3)通过潜在映射网络(LMN)压缩输入条件；4)利用CymbaDiff去噪网络，结合三重Mamba模块和圆柱Mamba层进行生成；5)从噪声逐步去噪，最终生成高质量的3D语义场景。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': "关键创新点包括：1)提出'基于草图的3D户外场景生成'新任务；2)构建首个专门的大规模基准数据集SketchSem3D；3)提出CymbaDiff模型，结合圆柱Mamba块增强空间连贯性；4)引入圆柱坐标系统来更好地表示户外场景的空间关系。相比之前工作，CymbaDiff避免了多尺度方法的计算复杂性，解决了BEV方法缺乏3D结构信息的问题，并首次将基于草本的3D生成扩展到复杂户外场景。", '如果要用一句话总结这篇论文的贡献，你会怎么说？': '本文通过CymbaDiff方法和SketchSem3D数据集，首次实现了从简单草图和卫星图像生成高质量、语义连贯的大规模3D城市场景，为自动驾驶和城市规划等应用提供了新的解决方案。'}


### 论文摘要

Outdoor 3D semantic scene generation produces realistic and semantically rich environments for applications such as urban simulation and autonomous driving. However, advances in this direction are constrained by the absence of publicly available, well-annotated datasets. We introduce SketchSem3D, the first large-scale benchmark for generating 3D outdoor semantic scenes from abstract freehand sketches and pseudo-labeled annotations of satellite images. SketchSem3D includes two subsets, Sketch-based SemanticKITTI and Sketch-based KITTI-360 (containing LiDAR voxels along with their corresponding sketches and annotated satellite images), to enable standardized, rigorous, and diverse evaluations. We also propose Cylinder Mamba Diffusion (CymbaDiff) that significantly enhances spatial coherence in outdoor 3D scene generation. CymbaDiff imposes structured spatial ordering, explicitly captures cylindrical continuity and vertical hierarchy, and preserves both physical neighborhood relationships and global context within the generated scenes. Extensive experiments on SketchSem3D demonstrate that CymbaDiff achieves superior semantic consistency, spatial realism, and cross-dataset generalization. The code and dataset will be available at https://github.com/Lillian-research-hub/CymbaDiff

---

