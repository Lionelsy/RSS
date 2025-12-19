# 今日论文推荐 - 2025-12-19

共 76 篇论文

---

## 1. Deep learning directed synthesis of fluid ferroelectric materials

**论文链接:** [http://arxiv.org/abs/2512.16671v1](http://arxiv.org/abs/2512.16671v1)

**作者:** Charles Parton-Barr, Stuart R. Berrow, Calum J. Gibb, Jordan Hobbs, Wanhe Jiang, Caitlin O'Brien, Will C. Ogle, Helen F. Gleeson, Richard J. Mandle

**发布时间:** 2025-12-18

**备注:** 13 pages, 4 figures

### GPT解析

### 总结

研究团队开发并验证了一种深度学习数据到分子的管道，实现了有机流体铁电体的靶向设计和合成，实验验证了11个候选材料，展示了发现可合成流体铁电体的实用闭环方法，标志着向功能软材料的自主设计迈进了一步。

### 背景

流体铁电体是一类新发现的液晶材料，具有可切换的长程极化有序性，在超快电光技术、响应性软物质和下一代能源材料方面有应用前景。目前这一领域的发现主要依赖于直觉和偶然性，限制了进展。

### 目的

开发并实验验证一个深度学习数据到分子的管道，实现有机流体铁电体的靶向设计和合成。

### 方法

整理所有已知纵向极化液晶材料的综合数据集；训练图神经网络预测铁电行为（准确率高达95%）和转变温度（均方根误差低至11K）；使用图变分自编码器生成新分子结构；通过高性能分类器和回归器组合过滤候选物；与计算逆合成引擎和数字化化学库存集成缩小设计空间；合成并表征11个候选物，通过基于混合的外推方法比较实验结果与神经网络预测。

### 主要发现

实验验证了新型流体铁电体材料；通过质量反馈数据增强了原始数据集；展示了发现可合成流体铁电体的实用闭环方法。

### 结论

该研究代表了一种发现可合成流体铁电体的实用闭环方法，标志着向功能软材料的自主设计迈进了一步。

### 翻译

流体铁电体是一类最近发现的液晶，表现出可切换的长程极化有序性，为超快电光技术、响应性软物质和下一代能源材料提供了机会。然而，它们的发现几乎完全依赖于直觉和偶然性，限制了该领域的进展。在这里，我们开发并实验验证了一种深度学习数据到分子的管道，使新型有机流体铁电体的靶向设计和合成成为可能。我们整理了所有已知的纵向极化液晶材料的综合数据集，并训练了图神经网络，以高达95%的准确率预测铁电行为，并实现低至11K的转变温度均方根误差。图变分自编码器生成全新的分子结构，使用高性能分类器和回归器组合进行过滤，以识别具有预测铁电向列行为和可及转变温度的候选物。与计算逆合成引擎和数字化化学库存的集成进一步将设计空间缩小到可合成的长名单。合成了11个候选物并通过既定的基于混合的外推方法进行了表征。从中，外推的铁电向列转变与神经网络预测进行了比较。新型材料的实验验证通过质量反馈数据增强了原始数据集，从而有助于未来的研究。这些结果展示了一种发现可合成流体铁电体的实用闭环方法，标志着向功能软材料的自主设计迈进了一步。


### 论文摘要

Fluid ferroelectrics, a recently discovered class of liquid crystals that exhibit switchable, long-range polar order, offer opportunities in ultrafast electro-optic technologies, responsive soft matter, and next-generation energy materials. Yet their discovery has relied almost entirely on intuition and chance, limiting progress in the field. Here we develop and experimentally validate a deep-learning data-to-molecule pipeline that enables the targeted design and synthesis of new organic fluid ferroelectrics. We curate a comprehensive dataset of all known longitudinally polar liquid-crystal materials and train graph neural networks that predict ferroelectric behaviour with up to 95% accuracy and achieve root mean square errors as low as 11 K for transition temperatures. A graph variational autoencoder generates de novo molecular structures which are filtered using an ensemble of high-performing classifiers and regressors to identify candidates with predicted ferroelectric nematic behaviour and accessible transition temperatures. Integration with a computational retrosynthesis engine and a digitised chemical inventory further narrows the design space to a synthesis-ready longlist. 11 candidates were synthesised and characterized through established mixture-based extrapolation methods. From which extrapolated ferroelectric nematic transitions were compared against neural network predictions. The experimental verification of novel materials augments the original dataset with quality feedback data thus aiding future research. These results demonstrate a practical, closed-loop approach to discovering synthesizable fluid ferroelectrics, marking a step toward autonomous design of functional soft materials.

---

## 2. Microsoft Academic Graph Information Retrieval for Research Recommendation and Assistance

**论文链接:** [http://arxiv.org/abs/2512.16661v1](http://arxiv.org/abs/2512.16661v1)

**作者:** Jacob Reiss, Shikshya Shiwakoti, Samuel Goldsmith, Ujjwal Pandit

**发布时间:** 2025-12-18

**备注:** 5 pages, 3 figures

### GPT解析

### 总结

这篇论文提出了一种基于注意力的子图检索器模型，结合图神经网络和大型语言模型，通过注意力剪枝技术提取精炼子图，用于高级知识推理。

### 背景

在信息驱动的时代，获取科学出版物变得容易，但从海量研究中筛选信息却面临前所未有的挑战。

### 目的

开发一种有效处理大规模信息数据库检索的模型，解决科研文献筛选困难的问题。

### 方法

提出基于注意力的子图检索器，应用图神经网络作为检索器，通过注意力剪枝提取精炼子图，再传递给大型语言模型进行知识推理。

### 主要发现

图神经网络和图注意力机制在搜索大规模信息数据库方面表现出强大有效性，特别是与现代大型语言模型结合时。

### 结论

结合图神经网络和大型语言模型的优势，并应用注意力机制进行子图提取，可有效解决大规模科研文献筛选挑战。

### 翻译

在当今信息驱动的世界中，获取科学出版物变得越来越容易。与此同时，从海量可用研究文献中进行筛选比以往任何时候都更具挑战性。图神经网络(GNNs)和图注意力机制在搜索大规模信息数据库方面显示出强大的有效性，特别是与现代大型语言模型结合时。在本文中，我们提出了一种基于注意力的子图检索器，这是一种将图神经网络作为检索器的模型，它应用基于注意力的剪枝技术提取精炼的子图，然后将该子图传递给大型语言模型进行高级知识推理。


### 论文摘要

In today's information-driven world, access to scientific publications has become increasingly easy. At the same time, filtering through the massive volume of available research has become more challenging than ever. Graph Neural Networks (GNNs) and graph attention mechanisms have shown strong effectiveness in searching large-scale information databases, particularly when combined with modern large language models. In this paper, we propose an Attention-Based Subgraph Retriever, a GNN-as-retriever model that applies attention-based pruning to extract a refined subgraph, which is then passed to a large language model for advanced knowledge reasoning.

---

## 3. Riemannian Stochastic Interpolants for Amorphous Particle Systems

**论文链接:** [http://arxiv.org/abs/2512.16607v1](http://arxiv.org/abs/2512.16607v1)

**作者:** Louis Grenioux, Leonardo Galliano, Ludovic Berthier, Giulio Biroli, Marylou Gabrié

**发布时间:** 2025-12-18

### GPT解析

### 总结

本文提出了一种用于生成无定形材料平衡构型的等变黎曼随机插值框架，结合了几何约束和对称性，显著提高了生成性能。

### 背景

现代生成模型在物理系统模拟任务中潜力巨大，但需要针对特定领域约束进行适应。生物分子和晶体材料已取得进展，而无定形材料（玻璃）作为缺乏原子周期性的无序粒子系统，其平衡构型采样极为困难。

### 目的

开发能够产生具有明确定义似率的平衡构型的生成框架，以克服玻璃形成材料平衡构型采样缓慢困难的问题。

### 方法

利用等变黎曼随机插值框架，结合黎曼随机插值和等变流匹配。严格纳入周期边界条件和多组分粒子系统的对称性，调整等变图神经网络以直接在环面上操作。

### 主要发现

在模型无定形系统上的数值实验表明，强制执行几何和对称约束显著提高了生成性能。

### 结论

通过结合等变黎曼随机插值框架和几何对称约束，可以有效生成无定形材料的平衡构型，为无序粒子系统模拟提供了新方法。

### 翻译

现代生成模型在加速涉及物理系统模拟的多样化任务方面具有巨大潜力，但它们必须针对每个领域的特定约束进行适应。在生物分子和晶体材料方面已取得显著进展。在此，我们解决无定形材料（玻璃）的问题，它们是缺乏原子周期性的无序粒子系统。采样玻璃形成材料的平衡构型是一个 notoriously 缓慢且困难的任务。通过开发能够产生具有明确定义似率的平衡构型的生成框架，可以克服这一障碍。在这项工作中，我们通过利用等变黎曼随机插值框架来解决这一挑战，该框架结合了黎曼随机插值和等变流匹配。我们的方法严格纳入周期边界条件和多组分粒子系统的对称性，调整等变图神经网络以直接在环面上操作。我们在模型无定形系统上的数值实验表明，强制执行几何和对称约束显著提高了生成性能。


### 论文摘要

Modern generative models hold great promise for accelerating diverse tasks involving the simulation of physical systems, but they must be adapted to the specific constraints of each domain. Significant progress has been made for biomolecules and crystalline materials. Here, we address amorphous materials (glasses), which are disordered particle systems lacking atomic periodicity. Sampling equilibrium configurations of glass-forming materials is a notoriously slow and difficult task. This obstacle could be overcome by developing a generative framework capable of producing equilibrium configurations with well-defined likelihoods. In this work, we address this challenge by leveraging an equivariant Riemannian stochastic interpolation framework which combines Riemannian stochastic interpolant and equivariant flow matching. Our method rigorously incorporates periodic boundary conditions and the symmetries of multi-component particle systems, adapting an equivariant graph neural network to operate directly on the torus. Our numerical experiments on model amorphous systems demonstrate that enforcing geometric and symmetry constraints significantly improves generative performance.

---

## 4. GFLAN: Generative Functional Layouts

**论文链接:** [http://arxiv.org/abs/2512.16275v1](http://arxiv.org/abs/2512.16275v1)

**作者:** Mohamed Abouagour, Eleftherios Garyfallidis

**发布时间:** 2025-12-18

**备注:** 21 pages, 15 figures

### GPT解析

### 总结

本文提出了一种名为GFLAN的生成框架，通过将平面图生成分解为拓扑规划和几何实现两个阶段来解决现有方法在捕捉建筑推理方面的局限性。

### 背景

自动平面图生成涉及组合搜索、几何约束满足和功能设计要求的交叉领域，这一领域一直难以实现统一的计算处理。虽然最近的深度学习方法提高了技术水平，但它们往往难以捕捉建筑推理的关键方面。

### 目的

为了解决现有方法难以捕捉建筑推理（拓扑关系优先、功能约束传播、循环模式形成）的基本挑战，引入GFLAN框架重构平面图合成过程。

### 方法

给定一个外部边界和前门位置，采用两阶段分解：阶段A使用具有双编码器的特殊卷积架构，通过离散概率图顺序分配房间质心；阶段B构建异构图并应用增强Transformer的图神经网络联合回归房间边界。

### 主要发现

现有深度学习方法虽然提高了技术水平，但往往难以捕捉建筑推理的关键方面，包括拓扑关系优先于几何实例化、功能约束通过邻接网络传播以及从局部连接决策中出现循环模式。

### 结论

GFLAN框架通过明确分解为拓扑规划和几何实现两个阶段，能够更好地处理平面图生成中的建筑推理问题。

### 翻译

自动平面图生成位于组合搜索、几何约束满足和功能设计要求的交叉点 - 这一交汇点历史上一直抵制统一的计算处理。虽然最近的深度学习方法提高了技术水平，但它们往往难以捕捉建筑推理：拓扑关系优先于几何实例化、功能约束通过邻接网络传播、以及从局部连接决策中出现循环模式。为了解决这些基本挑战，本文引入了GFLAN，一个生成框架，通过明确分解为拓扑规划和几何实现来重构平面图合成。给定单个外部边界和前门位置，我们的方法不采用直接的像素到像素或墙壁追踪生成，而是采用原则性的两阶段分解。阶段A采用具有双编码器的特殊卷积架构 - 将不变的空间上下文与演变的布局状态分离 - 通过在可行放置上的离散概率图顺序分配房间质心。阶段B构建一个将房间节点链接到边界顶点的异构图，然后应用增强Transformer的图神经网络(GNN)，该网络联合回归房间边界。


### 论文摘要

Automated floor plan generation lies at the intersection of combinatorial search, geometric constraint satisfaction, and functional design requirements -- a confluence that has historically resisted a unified computational treatment. While recent deep learning approaches have improved the state of the art, they often struggle to capture architectural reasoning: the precedence of topological relationships over geometric instantiation, the propagation of functional constraints through adjacency networks, and the emergence of circulation patterns from local connectivity decisions. To address these fundamental challenges, this paper introduces GFLAN, a generative framework that restructures floor plan synthesis through explicit factorization into topological planning and geometric realization. Given a single exterior boundary and a front-door location, our approach departs from direct pixel-to-pixel or wall-tracing generation in favor of a principled two-stage decomposition. Stage A employs a specialized convolutional architecture with dual encoders -- separating invariant spatial context from evolving layout state -- to sequentially allocate room centroids within the building envelope via discrete probability maps over feasible placements. Stage B constructs a heterogeneous graph linking room nodes to boundary vertices, then applies a Transformer-augmented graph neural network (GNN) that jointly regresses room boundaries.

---

## 5. Sharpness-aware Federated Graph Learning

**论文链接:** [http://arxiv.org/abs/2512.16247v1](http://arxiv.org/abs/2512.16247v1)

**作者:** Ruiyu Li, Peige Zhao, Guangxia Li, Pengcheng Wu, Xingyu Gao, Zhiqiang Xu

**发布时间:** 2025-12-18

**备注:** Accepted by WSDM'26

### GPT解析

### 总结

本文提出了一种名为SEAL的联邦图学习算法，通过同时最小化损失函数及其尖锐度，并引入基于本地表示相关矩阵的正则化器，解决了联邦图学习中数据异构性问题导致的本地模型泛化能力弱和维度崩塌问题。

### 背景

图神经网络(GNNs)应用于大规模真实世界图数据的主要障碍是集中式训练的挑战，这需要聚合来自不同组织的数据并引发隐私问题。联邦图学习(FGL)通过在不共享私有数据的情况下协作训练GNN模型来解决这一问题。

### 目的

提出一种新颖的优化目标，使其意识到本地GNN模型的尖锐度，通过同时最小化损失函数及其尖锐度寻找在损失值均匀较低的区域中的模型参数，提高对异构数据的泛化能力；并通过引入基于本地表示相关矩阵的正则化器，缓解学习模型的维度崩塌问题。

### 方法

提出了名为SEAL(Sharpness-aware fEderated grAph Learning)的算法，该方法结合了损失尖锐度感知和表示相关性正则化，通过最小化损失函数及其尖锐度同时放松由单个本地图样本生成的表示之间的相关性。

### 主要发现

SEAL算法能够增强联邦图学习中本地GNN模型的分类准确性和泛化能力。在多个图分类基准上的实验研究表明，SEAL始终优于最先进的FGL基线，并为更多参与者带来收益。

### 结论

SEAL算法通过同时考虑损失尖锐度和表示相关性，有效解决了联邦图学习中的数据异构性问题，提高了本地GNN模型的泛化能力和分类准确性。

### 翻译

将图神经网络(GNNs)应用于大规模真实世界图数据的许多障碍之一是集中式训练的挑战，这需要聚合来自不同组织的数据，引发隐私问题。联邦图学习(FGL)通过在不共享私有数据的情况下协作训练GNN模型来解决这一问题。然而，FGL系统中的一个核心挑战是客户端间本地训练数据分布的差异，即数据异构性问题。大多数现有解决方案存在两个问题：(1)基于经验风险最小化的典型优化器容易导致本地模型陷入尖锐的谷底，削弱其对分布外图数据的泛化能力。(2)本地图数据学习表示中的普遍维度崩塌对GNN模型的分类能力产生不利影响。为此，我们提出了一种新颖的优化目标，使其意识到本地GNN模型的尖锐度(即损失曲面的曲率)。通过同时最小化损失函数及其尖锐度，我们寻找在损失值均匀较低的区域中的模型参数，从而提高对异构数据的泛化能力。通过引入基于本地表示相关矩阵的正则化器，我们放松由单个本地图样本生成的表示之间的相关性，以缓解学习模型的维度崩塌。所提出的SEAL(Sharpness-aware fEderated grAph Learning)算法能够增强联邦图学习中本地GNN模型的分类准确性和泛化能力。在多个图分类基准上的实验研究表明，SEAL始终优于最先进的FGL基线，并为更多参与者带来收益。


### 论文摘要

One of many impediments to applying graph neural networks (GNNs) to large-scale real-world graph data is the challenge of centralized training, which requires aggregating data from different organizations, raising privacy concerns. Federated graph learning (FGL) addresses this by enabling collaborative GNN model training without sharing private data. However, a core challenge in FGL systems is the variation in local training data distributions among clients, known as the data heterogeneity problem. Most existing solutions suffer from two problems: (1) The typical optimizer based on empirical risk minimization tends to cause local models to fall into sharp valleys and weakens their generalization to out-of-distribution graph data. (2) The prevalent dimensional collapse in the learned representations of local graph data has an adverse impact on the classification capacity of the GNN model. To this end, we formulate a novel optimization objective that is aware of the sharpness (i.e., the curvature of the loss surface) of local GNN models. By minimizing the loss function and its sharpness simultaneously, we seek out model parameters in a flat region with uniformly low loss values, thus improving the generalization over heterogeneous data. By introducing a regularizer based on the correlation matrix of local representations, we relax the correlations of representations generated by individual local graph samples, so as to alleviate the dimensional collapse of the learned model. The proposed \textbf{S}harpness-aware f\textbf{E}derated gr\textbf{A}ph \textbf{L}earning (SEAL) algorithm can enhance the classification accuracy and generalization ability of local GNN models in federated graph learning. Experimental studies on several graph classification benchmarks show that SEAL consistently outperforms SOTA FGL baselines and provides gains for more participants.

---

## 6. Coarse-to-Fine Open-Set Graph Node Classification with Large Language Models

**论文链接:** [http://arxiv.org/abs/2512.16244v1](http://arxiv.org/abs/2512.16244v1)

**作者:** Xueqi Ma, Xingjun Ma, Sarah Monazam Erfani, Danilo Mandic, James Bailey

**发布时间:** 2025-12-18

**备注:** Accepted to AAAI 2026

### GPT解析

### 总结

本文提出了CFC框架，利用大语言模型解决图神经网络在开放世界场景中的OOD分类问题，通过粗到细的分类方法显著提高了OOD检测和分类性能。

### 背景

开集分类方法对于在开放世界场景中部署图神经网络至关重要，但现有方法通常将所有OOD样本视为单一类别，而真实世界应用需要对OOD样本有更深入的了解。

### 目的

解决在没有真实标签信息的情况下，能否将OOD检测扩展到OOD分类的问题，并提出一种利用大语言模型的从粗到细的开集分类框架。

### 方法

CFC框架包含三个关键组件：粗分类器使用LLM提示进行OOD检测和异常标签生成；基于GNN的细分类器使用识别的OOD样本进行训练；通过LLM提示和后处理的OOD标签实现精细的OOD分类。该方法采用基于内在意义的语义OOD实例，而非合成或辅助样本。

### 主要发现

实验结果表明，CFC在图和文本领域比最先进方法提高了10%的OOD检测性能，在图数据集上实现了高达70%的OOD分类准确率。

### 结论

CFC框架能够有效地进行OOD检测和分类，不需要真实标签信息，提高了可解释性和实用性。

### 翻译

开发能够对分布内(ID)数据进行分类同时检测分布外(OOD)样本的开集分类方法，对于在开放世界场景中部署图神经网络(GNNs)至关重要。现有方法通常将所有OOD样本视为单一类别，尽管在现实世界的应用中，特别是高风险设置如欺诈检测和医疗诊断，需要对OOD样本有更深入的了解，包括它们可能的标签。这引出了一个关键问题：在没有真实标签信息的情况下，能否将OOD检测扩展到OOD分类？为解决这一问题，我们提出了一个利用大语言模型(LLMs)处理图数据集的从粗到细的开集分类(CFC)框架。CFC包含三个关键组件：使用LLM提示进行OOD检测和异常标签生成的粗分类器；使用粗分类器识别的OOD样本进行训练以增强OOD检测和ID分类的基于GNN的细分类器；以及通过LLM提示和后处理的OOD标签实现的精细OOD分类。与依赖合成或辅助OOD样本的方法不同，CFC采用基于其内在意义的语义OOD实例，提高了可解释性和实用性。实验结果表明，CFC在图和文本领域比最先进方法提高了10%的OOD检测性能，并在图数据集上实现了高达70%的OOD分类准确率。


### 论文摘要

Developing open-set classification methods capable of classifying in-distribution (ID) data while detecting out-of-distribution (OOD) samples is essential for deploying graph neural networks (GNNs) in open-world scenarios. Existing methods typically treat all OOD samples as a single class, despite real-world applications, especially high-stake settings such as fraud detection and medical diagnosis, demanding deeper insights into OOD samples, including their probable labels. This raises a critical question: can OOD detection be extended to OOD classification without true label information? To address this question, we propose a Coarse-to-Fine open-set Classification (CFC) framework that leverages large language models (LLMs) for graph datasets. CFC consists of three key components: a coarse classifier that uses LLM prompts for OOD detection and outlier label generation, a GNN-based fine classifier trained with OOD samples identified by the coarse classifier for enhanced OOD detection and ID classification, and refined OOD classification achieved through LLM prompts and post-processed OOD labels. Unlike methods that rely on synthetic or auxiliary OOD samples, CFC employs semantic OOD instances that are genuinely out-of-distribution based on their inherent meaning, improving interpretability and practical utility. Experimental results show that CFC improves OOD detection by ten percent over state-of-the-art methods on graph and text domains and achieves up to seventy percent accuracy in OOD classification on graph datasets.

---

## 7. The Evolution of Reranking Models in Information Retrieval: From Heuristic Methods to Large Language Models

**论文链接:** [http://arxiv.org/abs/2512.16236v1](http://arxiv.org/abs/2512.16236v1)

**作者:** Tejul Pandit, Sakshi Mahendru, Meet Raval, Dhvani Upadhyay

**发布时间:** 2025-12-18

**备注:** 15 pages, 1 figure, Accepted in CLNLP'25

### GPT解析

### 总结

这篇论文是对信息检索系统中重排序技术的全面综述，涵盖了从基础方法到先进神经网络架构的发展历程，以及在现代检索增强生成(RAG)管道中的应用。文章还探讨了提高重排序效率的技术，特别是知识蒸馏方法，以及大型语言模型在重排序中的新兴应用。

### 背景

重排序是当代信息检索系统中的关键阶段，通过优化初始候选集来提高用户呈现最终结果的相关性。特别是在现代检索增强生成(RAG)管道中，检索到的文档对输出质量有显著影响。

### 目的

提供对不断发展的重排序技术格局的全面指南，清晰展示重排序方法的进展。阐明各种重排序策略的基本思想、相对有效性、计算特性和实际权衡。

### 方法

作者进行了历史轨迹的梳理，从基础方法开始，探索了各种复杂的神经网络架构，包括交叉编码器、类似T5的序列生成模型和用于结构信息的图神经网络(GNNs)。分析了提高神经重排序器效率的技术，特别是知识蒸馏方法。还研究了在重排序中集成大型语言模型(LLMs)的新兴领域，包括新的提示策略和微调技术。

### 主要发现

论文展示了重排序技术的多样性和发展历程，认识到先进神经重排序器的计算成本问题，分析了提高效率的方法，特别是在知识蒸馏方面。发现了大型语言模型在重排序中的新兴应用和潜力。

### 结论

这篇综述提供了各种重排序范式的结构化综合，突出了它们的基本原理和相对优缺点，为理解和应用重排序技术提供了全面的视角。

### 翻译

重排序是当代信息检索系统中的关键阶段，通过优化初始候选集来提高用户呈现最终结果的相关性。本文是对不断发展的重排序技术格局的全面指南，清晰展示了重排序方法的进展。我们介绍了信息检索中使用的重排序模型的综合调查，特别是在现代检索增强生成(RAG)管道中，检索到的文档显著影响输出质量。我们按时间顺序梳理了重排序技术的历史轨迹，从基础方法开始，然后探索了各种复杂的神经网络架构，如交叉编码器、类似T5的序列生成模型以及用于结构信息的图神经网络(GNNs)。认识到先进神经重排序器的计算成本，我们分析了提高效率的技术，特别是知识蒸馏，用于创建具有竞争力的轻量级替代方案。此外，我们探索了在重排序中集成大型语言模型的新兴领域，研究了新的提示策略和微调技术。本综述旨在阐明各种重排序策略的基本思想、相对有效性、计算特性和实际权衡。该综述提供了各种重排序范式的结构化综合，突出了它们的基本原理和相对优缺点。


### 论文摘要

Reranking is a critical stage in contemporary information retrieval (IR) systems, improving the relevance of the user-presented final results by honing initial candidate sets. This paper is a thorough guide to examine the changing reranker landscape and offer a clear view of the advancements made in reranking methods. We present a comprehensive survey of reranking models employed in IR, particularly within modern Retrieval Augmented Generation (RAG) pipelines, where retrieved documents notably influence output quality.   We embark on a chronological journey through the historical trajectory of reranking techniques, starting with foundational approaches, before exploring the wide range of sophisticated neural network architectures such as cross-encoders, sequence-generation models like T5, and Graph Neural Networks (GNNs) utilized for structural information. Recognizing the computational cost of advancing neural rerankers, we analyze techniques for enhancing efficiency, notably knowledge distillation for creating competitive, lighter alternatives. Furthermore, we map the emerging territory of integrating Large Language Models (LLMs) in reranking, examining novel prompting strategies and fine-tuning tactics. This survey seeks to elucidate the fundamental ideas, relative effectiveness, computational features, and real-world trade-offs of various reranking strategies. The survey provides a structured synthesis of the diverse reranking paradigms, highlighting their underlying principles and comparative strengths and weaknesses.

---

## 8. A Multi-scale Fused Graph Neural Network with Inter-view Contrastive Learning for Spatial Transcriptomics Data Clustering

**论文链接:** [http://arxiv.org/abs/2512.16188v1](http://arxiv.org/abs/2512.16188v1)

**作者:** Jianping Mei, Siqi Ai, Ye Yuan

**发布时间:** 2025-12-18

**备注:** 15 pages, 3 figures

### GPT解析

### 总结

本文提出了一种名为stMFG的多尺度交互融合图网络，用于解决空间转录组学中空间域识别的挑战，通过引入层级跨视图注意力和跨视图对比学习，实现了空间和基因特征的有效整合，并在实验中取得了显著性能提升。

### 背景

空间转录组学能够在原生组织环境中进行全基因组表达分析，但由于基因-空间相互作用的复杂性，识别空间域仍然具有挑战性。现有方法通常采用'分别编码，后期融合'的方式，分别处理空间和特征视图，仅在最输出层进行融合，这限制了多尺度语义捕获和跨视图交互。

### 目的

开发一种能够更好地捕获多尺度语义并促进跨视图交互的方法，以改进空间转录组学中的空间域识别性能。

### 方法

提出了stMFG（多尺度交互融合图网络），该方法引入了层级跨视图注意力机制，在每个卷积后动态整合空间和基因特征。模型结合了跨视图对比学习和空间约束，以提高区分性同时保持空间连续性。

### 主要发现

在DLPFC和乳腺癌数据集上，stMFG超越了最先进的方法，在某些切片上实现了高达14%的ARI（调整兰德指数）改进。

### 结论

stMFG通过有效整合空间和基因特征，显著提高了空间域识别的性能，为空间转录组学分析提供了更强大的工具。

### 翻译

空间转录组学能够在原生组织环境中进行全基因组表达分析，但由于基因-空间相互作用的复杂性，识别空间域仍然具有挑战性。现有方法通常分别处理空间和特征视图，仅在最输出层进行融合——这是一种'分别编码，后期融合'的范式，限制了多尺度语义捕获和跨视图交互。因此，我们提出了stMFG，这是一种多尺度交互融合图网络，它引入了层级跨视图注意力，在每个卷积后动态整合空间和基因特征。该模型结合了跨视图对比学习和空间约束，以提高区分性同时保持空间连续性。在DLPFC和乳腺癌数据集上，stMFG超越了最先进的方法，在某些切片上实现了高达14%的ARI改进。


### 论文摘要

Spatial transcriptomics enables genome-wide expression analysis within native tissue context, yet identifying spatial domains remains challenging due to complex gene-spatial interactions. Existing methods typically process spatial and feature views separately, fusing only at output level - an "encode-separately, fuse-late" paradigm that limits multi-scale semantic capture and cross-view interaction. Accordingly, stMFG is proposed, a multi-scale interactive fusion graph network that introduces layer-wise cross-view attention to dynamically integrate spatial and gene features after each convolution. The model combines cross-view contrastive learning with spatial constraints to enhance discriminability while maintaining spatial continuity. On DLPFC and breast cancer datasets, stMFG outperforms state-of-the-art methods, achieving up to 14% ARI improvement on certain slices.

---

## 9. A Multimodal Approach to Alzheimer's Diagnosis: Geometric Insights from Cube Copying and Cognitive Assessments

**论文链接:** [http://arxiv.org/abs/2512.16184v1](http://arxiv.org/abs/2512.16184v1)

**作者:** Jaeho Yang, Kijung Yoon

**发布时间:** 2025-12-18

### GPT解析

### 总结

这项研究提出了一种创新的多模态机器学习方法，通过分析手绘立方体草图来辅助阿尔茨海默病的早期检测。该方法利用图神经网络捕捉立方体绘制的几何和拓扑特征，并结合人口统计和神经心理学数据，为AD筛查提供了一种可解释、非侵入性和可扩展的新工具。

### 背景

阿尔茨海默病的早期和可及检测仍然是一个关键的临床挑战，立方体临摹任务提供了一种简单但信息丰富的视空间功能评估方法。

### 目的

提出一个多模态框架，将手绘立方体草图转换为图结构表示，捕捉几何和拓扑特性，并将这些特征与人口统计信息和神经心理学测试分数结合用于AD分类。

### 方法

将立方体绘制建模为具有节点特征的图，节点特征编码空间坐标、局部基于图let的拓扑和角度几何，使用图神经网络处理这些特征，并在晚期融合模型中与年龄、教育和NPT特征融合。

### 主要发现

基于图的表示提供了强大的单模态基线，明显优于基于像素的卷积模型；多模态融合进一步提高了性能和对类别不平衡的鲁棒性；基于SHAP的可解释性分析确定了特定的图let基序和几何失真作为关键预测因子，与临床观察到的AD患者立方体绘制组织紊乱现象高度一致。

### 结论

基于图的立方体临摹分析成为一种可解释、非侵入性和可扩展的阿尔茨海默病筛查方法。

### 翻译

阿尔茨海默病的早期和可及检测仍然是一个关键的临床挑战，立方体临摹任务提供了一种简单但信息丰富的视空间功能评估方法。这项工作提出了一个多模态框架，将手绘立方体草图转换为捕获几何和拓扑特性的图结构表示，并将这些特征与人口统计信息和神经心理学测试分数结合用于AD分类。立方体绘制被建模为具有节点特征的图，这些节点特征编码空间坐标、局部基于图let的拓扑和角度几何，使用图神经网络处理这些特征，并在晚期融合模型中与年龄、教育和NPT特征融合。实验结果表明，基于图的表示提供了强大的单模态基线，明显优于基于像素的卷积模型，而多模态融合进一步提高了性能和对类别不平衡的鲁棒性。基于SHAP的可解释性分析确定了特定的图let基序和几何失真作为关键预测因子，与临床观察到的AD患者立方体绘制组织紊乱现象高度一致。总的来说，这些结果确立了基于图的立方体临摹分析作为一种可解释、非侵入性和可扩展的阿尔茨海默病筛查方法。


### 论文摘要

Early and accessible detection of Alzheimer's disease (AD) remains a critical clinical challenge, and cube-copying tasks offer a simple yet informative assessment of visuospatial function. This work proposes a multimodal framework that converts hand-drawn cube sketches into graph-structured representations capturing geometric and topological properties, and integrates these features with demographic information and neuropsychological test (NPT) scores for AD classification. Cube drawings are modeled as graphs with node features encoding spatial coordinates, local graphlet-based topology, and angular geometry, which are processed using graph neural networks and fused with age, education, and NPT features in a late-fusion model. Experimental results show that graph-based representations provide a strong unimodal baseline and substantially outperform pixel-based convolutional models, while multimodal integration further improves performance and robustness to class imbalance. SHAP-based interpretability analysis identifies specific graphlet motifs and geometric distortions as key predictors, closely aligning with clinical observations of disorganized cube drawings in AD. Together, these results establish graph-based analysis of cube copying as an interpretable, non-invasive, and scalable approach for Alzheimer's disease screening.

---

## 10. SegGraph: Leveraging Graphs of SAM Segments for Few-Shot 3D Part Segmentation

**论文链接:** [http://arxiv.org/abs/2512.16143v1](http://arxiv.org/abs/2512.16143v1)

**作者:** Yueyang Hu, Haiyong Jiang, Haoxuan Song, Jun Xiao, Hao Pan

**发布时间:** 2025-12-18

### GPT解析

### 总结

本文提出了一种名为SegGraph的新框架，用于解决少样本3D部件分割中有效聚合2D基础模型知识到3D的问题。通过基于SAM分割图的传播方法，显式学习几何特征并保持语义一致性，显著提高了分割性能。

### 背景

最近的进展表明2D基础模型在低样本3D部件分割方面有很大潜力，但如何有效聚合2D基础模型的3D知识仍是一个开放问题。现有方法要么忽略3D特征学习的几何结构，要么忽视SAM的高质量分组线索，导致分割不足和不一致的部件标签。

### 目的

解决如何有效聚合2D基础模型知识到3D的问题，提高3D部件分割的性能，特别是在小部件和部件边界上的表现。

### 方法

设计了一种基于SAM分割图的传播方法SegGraph，通过建模分割之间的相互重叠和邻接关系编码几何特征，构建分割图表示空间关系，使用图神经网络传播特征，并采用视图方向加权的融合将分割特征映射到3D点以保持语义一致性。

### 主要发现

在PartNet-E上的实验表明，SegGraph比所有竞争基线至少高出6.9%的mIoU，特别在小部件和部件边界上表现优异，展示了优越的几何理解能力。

### 结论

SegGraph方法通过显式学习几何特征和保持语义一致性，有效解决了2D基础模型知识到3D的聚合问题，显著提高了3D部件分割的性能。

### 翻译

本文提出了一种用于少样本3D部件分割的新框架。最近的进展已经证明2D基础模型在低样本3D部件分割方面具有巨大潜力。然而，如何有效聚合2D基础模型的3D知识仍然是一个开放问题。现有方法要么忽略3D特征学习的几何结构，要么忽视SAM的高质量分组线索，导致分割不足和不一致的部件标签。我们设计了一种基于SAM分割图的传播方法，名为SegGraph，用于显式学习SAM分割掩码中编码的几何特征。我们的方法通过建模分割之间的相互重叠和邻接关系来编码几何特征，同时保持分割内的语义一致性。我们构建了一个分割图，概念上类似于地图集，其中节点代表分割，边捕获它们的空间关系（重叠/邻接）。每个节点自适应调制2D基础模型特征，然后通过图神经网络传播以学习全局几何结构。为了强制执行分割内的语义一致性，我们使用一种新的视图方向加权的融合将分割特征映射到3D点，减弱来自低质量分割的贡献。在PartNet-E上的大量实验表明，我们的方法比所有竞争基线至少高出6.9%的mIoU。进一步分析显示，SegGraph在小部件和部件边界上实现了特别强的性能，展示了其优越的几何理解能力。代码可在以下网址获取：https://github.com/YueyangHu2000/SegGraph。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决如何有效将2D基础模型（如SAM）的知识聚合到3D空间，用于少样本3D部件分割的问题。这个问题很重要，因为3D部件分割是计算机视觉和图形学的基础任务，广泛应用于形状分析、3D建模和机器人操作等领域。许多实际应用涉及新颖形状，需要仅用少量标注就能实现高质量的3D部件分割，而2D基础模型虽有强大泛化能力，但2D图像和3D几何形状间的模态差距限制了它们在3D领域的直接应用。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者基于三个关键洞察设计了SegGraph方法：1)SAM的高质量分割为3D点提供了一致的段内分组线索；2)段之间的空间关系（重叠和相邻）为部件分割提供了重要先验；3)SAM段的质量与观察方向相关。作者借鉴了现有工作，如利用SAM进行2D分割、使用多视图渲染技术、采用图神经网络（GATv2）进行特征传播，并受微分几何中图册概念的启发，将段图视为3D形状的图册表示。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是构建基于SAM分割的段图，通过图神经网络传播特征，将2D基础模型的知识有效聚合到3D空间，同时保留几何结构和语义一致性。整体流程包括：1)特征编码：将3D点云渲染为多视图图像，提取并聚合特征；2)段生成：使用SAM进行分割并聚合为3D段；3)段编码：计算自适应点贡献和段特征；4)构建段图：以段为节点，空间关系为边构建图；5)图传播：使用GATv2传播特征；6)视角质量感知特征池化：根据视角质量将段特征映射回3D点；7)部件预测：结合特征预测部件标签。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)提出SAM段图框架，显式学习SAM分割掩码编码的几何特征；2)通过建模段间重叠和相邻关系编码几何特征；3)构建类似于图册的段图结构；4)提出视角方向加权的融合方法。相比之前工作的不同：相比3D标签聚合方法，考虑了几何结构；相比3D特征聚合方法，使用段图而非简单KNN传播；相比蒸馏方法，不需要大量3D数据训练，更适合少样本场景；显式利用SAM分割质量与观察方向的相关性提高鲁棒性。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'SegGraph通过构建SAM段图并利用图神经网络传播特征，有效解决了2D基础模型到3D部件分割的知识转移问题，在少样本场景下实现了最先进的性能，特别是在小部件和边界区域表现出色。'}


### 论文摘要

This work presents a novel framework for few-shot 3D part segmentation. Recent advances have demonstrated the significant potential of 2D foundation models for low-shot 3D part segmentation. However, it is still an open problem that how to effectively aggregate 2D knowledge from foundation models to 3D. Existing methods either ignore geometric structures for 3D feature learning or neglects the high-quality grouping clues from SAM, leading to under-segmentation and inconsistent part labels. We devise a novel SAM segment graph-based propagation method, named SegGraph, to explicitly learn geometric features encoded within SAM's segmentation masks. Our method encodes geometric features by modeling mutual overlap and adjacency between segments while preserving intra-segment semantic consistency. We construct a segment graph, conceptually similar to an atlas, where nodes represent segments and edges capture their spatial relationships (overlap/adjacency). Each node adaptively modulates 2D foundation model features, which are then propagated via a graph neural network to learn global geometric structures. To enforce intra-segment semantic consistency, we map segment features to 3D points with a novel view-direction-weighted fusion attenuating contributions from low-quality segments. Extensive experiments on PartNet-E demonstrate that our method outperforms all competing baselines by at least 6.9 percent mIoU. Further analysis reveals that SegGraph achieves particularly strong performance on small components and part boundaries, demonstrating its superior geometric understanding. The code is available at: https://github.com/YueyangHu2000/SegGraph.

---

## 11. Graph Neural Networks for Interferometer Simulations

**论文链接:** [http://arxiv.org/abs/2512.16051v1](http://arxiv.org/abs/2512.16051v1)

**作者:** Sidharth Kannan, Pooyan Goodarzi, Evangelos E. Papalexakis, Jonathan W. Richardson

**发布时间:** 2025-12-18

### GPT解析

### 总结

本研究将图神经网络(GNNs)应用于物理科学中的仪器设计领域，以LIGO模拟为例，展示了GNNs在准确捕捉复杂光学物理的同时，实现了比现有模拟技术快815倍的运行效率，并提供了高质量的数据集作为未来研究的基准。

### 背景

图神经网络(GNNs)近年来在高能物理、材料科学和流体动力学问题中显示出巨大潜力。

### 目的

探索GNNs在物理科学仪器设计领域的新应用，以LIGO模拟为案例研究。

### 方法

应用图神经网络技术模拟激光干涉引力波天文台(LIGO)的模型，捕捉复杂的光学物理特性。

### 主要发现

GNNs能够准确模拟复杂的光学物理，同时运行速度比现有最先进的模拟软件包快815倍。

### 结论

该研究讨论了仪器设计问题为机器学习模型带来的独特挑战，并提供了三种干涉仪拓扑结构的高保真光学物理模拟数据集，可作为未来研究的基准测试套件。

### 翻译

近年来，图神经网络(GNNs)在解决高能物理、材料科学和流体动力学问题方面显示出巨大潜力。在本研究中，我们介绍了GNNs在物理科学中的新应用：仪器设计。作为案例研究，我们将GNNs应用于激光干涉引力波天文台(LIGO)的模型模拟，表明它们能够准确捕捉其中复杂的光学物理，同时运行速度比最先进的模拟软件包快815倍。我们讨论了该问题为机器学习模型带来的独特挑战。此外，我们提供了三种干涉仪拓扑结构的高保真光学物理模拟数据集，可作为未来研究方向的基准测试套件。


### 论文摘要

In recent years, graph neural networks (GNNs) have shown tremendous promise in solving problems in high energy physics, materials science, and fluid dynamics. In this work, we introduce a new application for GNNs in the physical sciences: instrumentation design. As a case study, we apply GNNs to simulate models of the Laser Interferometer Gravitational-Wave Observatory (LIGO) and show that they are capable of accurately capturing the complex optical physics at play, while achieving runtimes 815 times faster than state of the art simulation packages. We discuss the unique challenges this problem provides for machine learning models. In addition, we provide a dataset of high-fidelity optical physics simulations for three interferometer topologies, which can be used as a benchmarking suite for future work in this direction.

---

## 12. VERM: Leveraging Foundation Models to Create a Virtual Eye for Efficient 3D Robotic Manipulation

**论文链接:** [http://arxiv.org/abs/2512.16724v1](http://arxiv.org/abs/2512.16724v1)

**作者:** Yixiang Chen, Yan Huang, Keji He, Peiyan Li, Liang Wang

**发布时间:** 2025-12-18

**备注:** Accepted at RA-L 2025

### GPT解析

### 总结

本文提出了VERM方法，通过利用基础模型知识从3D点云中生成虚拟任务自适应视图，过滤冗余信息，提高机器人3D操作任务的效率和性能。

### 背景

机器人在执行3D操作任务时，需要基于多个固定摄像头的感知进行动作规划。多摄像头设置引入了大量冗余和无关信息，增加了计算成本，并迫使模型花费额外的训练时间来提取关键的任务相关细节。

### 目的

过滤冗余信息并准确提取任务相关特征，提高机器人3D操作任务的效率和性能。

### 方法

提出了VERM（Virtual Eye for Robotic Manipulation）方法，利用基础模型中的知识，从构建的3D点云中想象一个虚拟任务自适应视图，有效捕获必要信息并减轻遮挡。同时设计了深度感知模块和动态粗到细的过程以促进3D动作规划和精细操作。

### 主要发现

在模拟基准RLBench和现实世界评估中的大量实验结果证明了该方法的有效性，超越了之前的先进方法，同时实现了训练时间1.89倍加速和推理速度1.54倍加速。

### 结论

VERM方法能够有效解决多摄像头系统中的冗余信息问题，提高机器人在3D操作任务中的性能和效率。

### 翻译

当执行3D操作任务时，机器人必须基于多个固定摄像头的感知来执行动作规划。多摄像头设置引入了大量冗余和无关信息，增加了计算成本，并迫使模型花费额外的训练时间来提取关键的任务相关细节。为了过滤冗余信息并准确提取任务相关特征，我们提出了VERM（用于机器人操作的虚拟眼睛）方法，利用基础模型中的知识，从构建的3D点云中想象一个虚拟任务自适应视图，有效捕获必要信息并减轻遮挡。为了促进3D动作规划和精细操作，我们进一步设计了一个深度感知模块和一个动态粗到细的过程。在模拟基准RLBench和现实世界评估中的大量实验结果证明了我们方法的有效性，超越了之前的先进方法，同时实现了训练时间1.89倍加速和推理速度1.54倍加速。更多结果可以在我们的项目网站https://verm-ral.github.io上找到。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文解决的问题是机器人在执行3D操作任务时，多摄像头系统引入大量冗余和不相关信息，增加了计算成本，并迫使模型花费额外训练时间提取关键任务相关信息。这个问题很重要，因为它限制了机器人系统的效率和实时性，在实际应用中计算资源有限的环境下尤为重要，同时现有方法要么处理整个3D表示（计算量大），要么依赖专家知识选择视角（缺乏灵活性）。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者受人类视觉注意机制的启发，思考如何让机器人像人类一样只需一眼和想象就能确定最佳视角。他们借鉴了认知科学对人类视觉选择的研究，并利用GPT-4o等基础模型的空间推理能力。该方法借鉴了现有工作中的多视图融合（如C2F-ARM、Peract）、虚拟相机平面投影（如RVT系列）和粗到细策略（如C2F-ARM、RVT-2），但创新性地将基础模型用于自动选择任务自适应视角，而非依赖预定义视角或专家知识。', '这个方法的核心思想是什么？整体实现流程是怎样的？': "核心思想是利用基础模型创建一个'虚拟眼睛'，从3D点云中生成任务自适应的虚拟视角，高效捕获必要信息并减少遮挡。整体流程包括：1)使用结构化提示引导GPT-4o预测最佳相机姿态；2)将多摄像头RGB-D输入转换为统一3D点云并投影到虚拟相机平面上；3)通过深度感知模块将2D动作预测扩展到3D空间；4)采用动态粗到细过程，在任务关键阶段自动触发视角放大以细化动作，只在必要时进行高精度计算。", '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)结构化提示框架将基础模型转化为空间推理代理；2)验证了方法在不同基础模型上的通用性；3)深度感知模块支持3D动作规划；4)动态粗到细机制只在必要时细化动作。相比之前工作，VERM不依赖预定义视角或专家知识，而是自动生成任务自适应视角；简化输入为单图像而非处理整个3D表示或多个视图；显著提升效率（训练加速1.89倍，推理加速1.54倍）；揭示并利用了大型多模态模型的3D空间推理能力。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'VERM通过利用基础模型创建虚拟眼睛，为机器人3D操作提供了高效、自适应的视觉感知方法，显著提升了操作效率并减少了计算负担。'}


### 论文摘要

When performing 3D manipulation tasks, robots have to execute action planning based on perceptions from multiple fixed cameras. The multi-camera setup introduces substantial redundancy and irrelevant information, which increases computational costs and forces the model to spend extra training time extracting crucial task-relevant details. To filter out redundant information and accurately extract task-relevant features, we propose the VERM (Virtual Eye for Robotic Manipulation) method, leveraging the knowledge in foundation models to imagine a virtual task-adaptive view from the constructed 3D point cloud, which efficiently captures necessary information and mitigates occlusion. To facilitate 3D action planning and fine-grained manipulation, we further design a depth-aware module and a dynamic coarse-to-fine procedure. Extensive experimental results on both simulation benchmark RLBench and real-world evaluations demonstrate the effectiveness of our method, surpassing previous state-of-the-art methods while achieving 1.89x speedup in training time and 1.54x speedup in inference speed. More results can be found on our project website at https://verm-ral.github.io .

---

## 13. Fully Dynamic Algorithms for Chamfer Distance

**论文链接:** [http://arxiv.org/abs/2512.16639v1](http://arxiv.org/abs/2512.16639v1)

**作者:** Gramoz Goranci, Shaofeng Jiang, Peter Kiss, Eva Szilagyi, Qiaoyuan Yang

**发布时间:** 2025-12-18

**备注:** NeurIPS 2025

### GPT解析

### 总结

研究动态环境下计算Chamfer距离的问题，提出首个在l_p范数下维护Chamfer距离近似的动态算法，算法简化为近似最近邻搜索，具有高效更新时间

### 背景

Chamfer距离是点云之间广泛使用的不相似性度量，有多个实际应用，特别是在机器学习中作为损失函数时需要重复评估动态变化的数据集

### 目的

高效维护两个动态变化点集之间的Chamfer距离近似值，支持点插入和删除操作

### 方法

提出一种动态算法，将Chamfer距离计算简化为近似最近邻搜索，使用标准ANN界限来保证近似性能

### 主要发现

算法能够实现(1+ε)-近似，更新时间为Õ(ε^{-d})；或实现O(1/ε)-近似，更新时间为Õ(d n^{ε^2} ε^{-4})

### 结论

该方法在真实数据集上评估表现良好，与自然基线相比具有竞争力

### 翻译

我们研究在完全动态环境下计算Chamfer距离的问题，其中两个点集A和B（每个大小最多为n）通过点的插入或删除动态变化，目标是在高效维护对dist_CH(A,B)的近似，其中dist是一种距离度量。Chamfer距离是点云之间广泛使用的不相似性度量，有许多实际应用需要重复评估动态变化的数据集，例如在机器学习中用作损失函数时。本文首次提出了在l_p范数下（p∈{1,2}）维护Chamfer距离近似的动态算法。我们的算法简化为近似最近邻搜索，开销很小。代入标准ANN界限，我们在Õ(ε^{-d})更新时间内获得(1+ε)-近似，在Õ(d n^{ε^2} ε^{-4})更新时间内获得O(1/ε)-近似。我们在真实数据集上评估了我们的方法，并证明其与自然基线相比具有竞争力。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决在动态变化环境下高效计算两个点集之间的Chamfer距离问题。当点集通过插入或删除操作动态变化时，如何维持对Chamfer距离的近似值。这个问题在现实中非常重要，因为Chamfer距离是点云分析中常用的不相似性度量，广泛应用于机器学习（如作为损失函数）、计算机视觉（如3D点云重建）和医学成像（如解剖结构跟踪）等领域，这些应用都需要在动态变化的数据集上重复计算Chamfer距离。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先认识到Chamfer距离计算可以转化为最近邻搜索问题，这是算法设计的基础。他们借鉴了静态算法中的重要性采样框架，特别是[BIJ+23]的工作，但在动态环境中显式维护近似分配变得困难且成本高。因此，作者设计了一个动态采样器，通过隐式表示距离估计来获得采样保证，避免了显式维护每个点的匹配单元。他们使用随机平移四叉树结构来隐式维护距离估计，并为每个节点维护动态采样器，允许根据单元大小和匹配点数量比例高效采样点。这种设计解决了动态环境中维护匹配单元的高成本问题。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是将Chamfer距离计算转化为最近邻搜索问题，使用动态最近邻oracle作为子程序，并通过重要性采样框架结合隐式距离表示来避免显式维护近似分配。整体实现流程包括：1) 维护动态四叉树结构，每个节点存储单元中A点和B点的数量；2) 为每个节点维护动态采样器；3) 处理更新时沿四叉树路径更新相关节点的匹配点计数；4) 回答查询时使用重要性采样从A集中采样点，查询近似最近邻oracle估计距离，计算加权平均值作为Chamfer距离估计；5) 通过多次采样和平均提高估计准确性。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1) 首个动态Chamfer距离算法，适用于ℓp范数(p∈{1,2})；2) 显著优于线性时间更新障碍的效率，低维达˜O(ε⁻ᵈ)，高维达˜O(dnε²ε⁻⁴)；3) 通过隐式表示距离估计而非显式维护近似分配，解决动态环境中的高成本问题；4) 基于随机平移四叉树的动态采样器设计；5) 与现有最近邻oracle的集成。相比之前工作，不同之处在于：之前工作主要关注静态设置，而本文首次解决动态环境下的维护问题；时间复杂度显著优于线性时间障碍；将问题转化为最近邻搜索；适用于任意维度；并在真实数据集上进行了实验验证。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '这篇论文首次提出了在动态变化点集上高效维护Chamfer距离的算法，通过问题转化和创新的动态采样框架，显著优于线性时间更新障碍，并在实际应用中表现出色。'}


### 论文摘要

We study the problem of computing Chamfer distance in the fully dynamic setting, where two set of points $A, B \subset \mathbb{R}^{d}$, each of size up to $n$, dynamically evolve through point insertions or deletions and the goal is to efficiently maintain an approximation to $\mathrm{dist}_{\mathrm{CH}}(A,B) = \sum_{a \in A} \min_{b \in B} \textrm{dist}(a,b)$, where $\textrm{dist}$ is a distance measure. Chamfer distance is a widely used dissimilarity metric for point clouds, with many practical applications that require repeated evaluation on dynamically changing datasets, e.g., when used as a loss function in machine learning. In this paper, we present the first dynamic algorithm for maintaining an approximation of the Chamfer distance under the $\ell_p$ norm for $p \in \{1,2 \}$. Our algorithm reduces to approximate nearest neighbor (ANN) search with little overhead. Plugging in standard ANN bounds, we obtain $(1+ε)$-approximation in $\tilde{O}(ε^{-d})$ update time and $O(1/ε)$-approximation in $\tilde{O}(d n^{ε^2} ε^{-4})$ update time. We evaluate our method on real-world datasets and demonstrate that it performs competitively against natural baselines.

---

## 14. SNOW: Spatio-Temporal Scene Understanding with World Knowledge for Open-World Embodied Reasoning

**论文链接:** [http://arxiv.org/abs/2512.16461v1](http://arxiv.org/abs/2512.16461v1)

**作者:** Tin Stribor Sohn, Maximilian Dillitzer, Jason J. Corso, Eric Sax

**发布时间:** 2025-12-18

### GPT解析

### 总结

SNOW是一个无需训练且与主干无关的框架，用于统一的4D场景理解，整合了视觉语言模型衍生的语义与点云几何和时间一致性，通过处理同步RGB图像和3D点云，生成对象级提议，进行分割，并通过时空标记化块编码产生多模态标记，最终形成4D场景图作为下游推理的先验。

### 背景

自主机器人系统需要对动态环境进行时空理解以确保可靠导航和交互。视觉语言模型提供开放世界语义先验但缺乏3D几何和时间动态基础，而几何感知捕捉结构和运动但语义稀疏。

### 目的

提出SNOW框架，实现统一的4D场景理解，整合VLM衍生的语义与点云几何和时间一致性。

### 方法

SNOW处理同步RGB图像和3D点云，使用HDBSCAN聚类生成对象级提议引导SAM2分割，通过STEP编码产生捕获语义、几何和时间属性的多模态标记，增量整合到4D场景图中，轻量级SLAM后端提供空间锚定和全局参考对齐。

### 主要发现

SNOW在多样化基准测试上实现精确的4D场景理解和空间基础推理，在多个设置中达到最先进性能，证明结构化4D先验对具身推理和自主机器人的重要性。

### 结论

SNOW成功整合VLM语义与几何感知，为自主机器人提供统一的4D场景理解能力，使视觉语言模型能够直接解释空间场景结构和时间动态。

### 翻译

自主机器人系统需要对动态环境进行时空理解，以确保可靠的导航和交互。虽然视觉语言模型提供开放世界语义先验，但它们缺乏3D几何和时间动态的基础。相反，几何感知捕捉结构和运动，但仍然语义稀疏。我们提出了SNOW(使用开放世界知识的场景理解)，一个无需训练且与主干无关的框架，用于统一的4D场景理解，将VLM衍生的语义与点云几何和时间一致性相结合。SNOW处理同步的RGB图像和3D点云，使用HDBSCAN聚类生成对象级提议，引导基于SAM2的分割。每个分割区域通过我们提出的时空标记化块编码(STEP)进行编码，产生捕获局部语义、几何和时间属性的多模态标记。这些标记被增量整合到4D场景图(4DSG)中，作为下游推理的4D先验。轻量级SLAM后端将所有STEP标记在环境中空间锚定，提供全局参考对齐，并确保跨越时间的明确空间基础。 resulting 4DSG形成了一个可查询的统一世界模型，通过VLMs可以直接解释空间场景结构和时间动态。在多样化的基准测试上的实验表明，SNOW能够实现精确的4D场景理解和空间基础推理，从而在多个设置中设置了新的最先进性能，突显了结构化4D先验对于具身推理和自主机器人的重要性。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决自主机器人在动态环境中进行时空理解的问题，即如何将视觉语言模型的开放世界语义知识与精确的3D几何和时间动态信息相结合。这个问题在现实中非常重要，因为自主机器人需要在非结构化、动态环境中导航和交互，不仅要识别物体，还要理解它们在3D空间中的位置以及如何随时间变化，而这正是现有方法的不足之处。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者通过分析现有方法的三个主要限制（依赖大量训练、特定主干架构、缺乏显式几何）来设计方法。他们借鉴了HDBSCAN聚类进行点云分组、SAM2进行分割、SLAM后端维持空间对齐等现有技术，但创新性地将这些技术组合成一个无需训练且与主干无关的框架。作者特别强调了解决VLMs缺乏几何定位和几何感知缺乏语义表达的问题，通过统一4D表示连接语义抽象与空间时间定位。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是创建一个统一的4D场景表示，将视觉语言模型的语义知识与3D几何和时间动态信息融合。整体流程包括：1)点云聚类和采样，使用HDBSCAN识别高密度区域；2)掩码生成和STEP编码，将分割区域编码为包含语义、几何和时间信息的多模态标记；3)4D场景图构建，通过滑动窗口聚合时空信息；4)使用VLM在统一场景图上进行推理。整个流程通过SLAM后端确保空间对齐，形成持久的4D世界模型。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)SNOW框架，无需训练即可融合VLM语义与3D感知；2)STEP编码，联合编码语义、几何和时间信息；3)4D场景图，提供结构化时空表示；4)在多个基准上实现最先进性能。相比之前工作，SNOW无需训练和微调，支持多种传感器模态，显式建模几何和时间信息，且能提供长期一致的场景表示，而不仅仅是2D图像处理或短时推理。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'SNOW通过创新的STEP编码和4D场景图构建方法，成功将视觉语言模型的开放世界语义知识与精确的3D几何和时间动态信息相融合，为自主机器人在复杂环境中的时空理解和推理提供了无需训练且通用的解决方案。'}


### 论文摘要

Autonomous robotic systems require spatio-temporal understanding of dynamic environments to ensure reliable navigation and interaction. While Vision-Language Models (VLMs) provide open-world semantic priors, they lack grounding in 3D geometry and temporal dynamics. Conversely, geometric perception captures structure and motion but remains semantically sparse. We propose SNOW (Scene Understanding with Open-World Knowledge), a training-free and backbone-agnostic framework for unified 4D scene understanding that integrates VLM-derived semantics with point cloud geometry and temporal consistency. SNOW processes synchronized RGB images and 3D point clouds, using HDBSCAN clustering to generate object-level proposals that guide SAM2-based segmentation. Each segmented region is encoded through our proposed Spatio-Temporal Tokenized Patch Encoding (STEP), producing multimodal tokens that capture localized semantic, geometric, and temporal attributes. These tokens are incrementally integrated into a 4D Scene Graph (4DSG), which serves as 4D prior for downstream reasoning. A lightweight SLAM backend anchors all STEP tokens spatially in the environment, providing the global reference alignment, and ensuring unambiguous spatial grounding across time. The resulting 4DSG forms a queryable, unified world model through which VLMs can directly interpret spatial scene structure and temporal dynamics. Experiments on a diverse set of benchmarks demonstrate that SNOW enables precise 4D scene understanding and spatially grounded inference, thereby setting new state-of-the-art performance in several settings, highlighting the importance of structured 4D priors for embodied reasoning and autonomous robotics.

---

## 15. Enhanced 3D Shape Analysis via Information Geometry

**论文链接:** [http://arxiv.org/abs/2512.16213v1](http://arxiv.org/abs/2512.16213v1)

**作者:** Amit Vishwakarma, K. S. Subrahamanian Moosath

**发布时间:** 2025-12-18

### GPT解析

### 总结

本文提出了一种基于信息几何框架的三维点云形状分析方法，通过将点云表示为统计流形上的高斯混合模型，并引入具有理论保证边界的修改的对称Kullback-Leibler散度，解决了传统点云比较方法中的局限性。

### 背景

三维点云为物体提供了高度精确的数字表示，在计算机图形学、摄影测量、计算机视觉和机器人学等领域有重要应用。然而，由于点云的非结构特性和所表示表面的复杂几何形状，比较点云面临重大挑战。

### 目的

引入一个信息几何框架用于三维点云形状分析，通过将点云表示为统计流形上的高斯混合模型，并提供一种稳定的散度度量方法。

### 方法

证明了高斯混合模型空间形成一个统计流形，并提出了修改的对称Kullback-Leibler(MSKL)散度，具有理论上保证的上界和下界，确保所有GMM比较的数值稳定性。

### 主要发现

通过在人体姿态辨别(MPI-FAUST数据集)和动物形状比较(G-PCD数据集)的综合实验，证明MSKL提供了稳定且单调变化的值，直接反映几何变化，优于传统距离和现有的KL近似。

### 结论

MSKL散度在点云比较中表现优异，解决了传统方法无法捕捉全局统计结构、对异常值敏感以及数值不稳定等问题。

### 翻译

三维点云为物体提供了高度精确的数字表示，对于计算机图形学、摄影测量、计算机视觉和机器人学中的应用至关重要。然而，由于点云的非结构性质和它们所表示表面的复杂几何形状，比较点云面临重大挑战。传统的几何度量，如Hausdorff距离和Chamfer距离，通常无法捕捉全局统计结构，并且对异常值敏感，而现有的高斯混合模型的Kullback-Leibler(KL)散度近似可能产生无界或数值不稳定的结果。本文通过将点云表示为统计流形上的高斯混合模型(GMMs)，引入了一种用于三维点云形状分析的信息几何框架。我们证明了GMM空间形成一个统计流形，并提出了具有理论上保证的上界和下界的修改的对称Kullback-Leibler(MSKL)散度，确保所有GMM比较的数值稳定性。通过在人体姿态辨别(MPI-FAUST数据集)和动物形状比较(G-PCD数据集)的综合实验，我们证明MSKL提供了稳定且单调变化的值，直接反映几何变化，优于传统距离和现有的KL近似。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决三维点云比较的挑战问题。点云是物体高度准确的数字表示，广泛应用于计算机图形学、摄影测量、计算机视觉和机器人等领域。然而，由于点云的非结构化特性和表面复杂几何形状，比较点云面临重大挑战。传统几何度量方法无法捕捉全局统计结构且对异常值敏感，而现有的高斯混合模型KL散度近似方法可能产生无界或数值不稳定的结果。解决这个问题对于形状分析任务如姿态估计、变形跟踪、配准和检索至关重要。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者基于信息几何框架设计方法，将点云表示为统计流形上的高斯混合模型(GMM)。首先分析了现有方法的局限性，然后建立了点云作为GMM的统计流形表示，证明了GMM空间形成统计流形。接着提出修改的对称KL散度(MSKL)解决现有KL近似的不稳定性问题。最后设计了完整的计算流程。作者借鉴了信息几何理论、GMM用于点云表示、Isomap流形学习方法以及现有的KL散度近似方法(如KLWA和KLMB)。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是将三维点云表示为统计流形上的概率分布(特别是高斯混合模型)，利用信息几何中的散度度量来比较这些分布，从而捕捉点云的全局统计结构而非仅关注点对点距离。整体流程包括：1)数据预处理(最远点采样下采样和归一化)；2)几何特征提取(计算局部协方差矩阵和11个几何特征)；3)流形嵌入(Isomap将14维描述符嵌入二维潜在空间)；4)统计流形建模(BIC确定混合组件数，EM算法拟合GMM)；5)散度计算(在规则网格上评估MSKL等散度度量)。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)建立点云作为GMM的统计流形表示框架；2)提出具有理论保证边界的MSKL散度，解决数值稳定性问题；3)推导MSKL散度的可计算上界和下界，确保理论适定性；4)在多个数据集上全面验证方法有效性。相比传统几何距离(如Hausdorff和Chamfer)仅捕捉点对点距离，MSKL能捕捉全局统计结构；相比现有KL近似方法可能低估或高估真实散度且可能无界，MSKL具有理论保证的边界，提供数值稳定性并更好反映几何变化。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '本文通过将点云表示为统计流形上的高斯混合模型并引入具有理论保证边界的修改对称KL散度，为3D点云形状分析提供了一种稳定、可解释且高效的比较方法。'}


### 论文摘要

Three-dimensional point clouds provide highly accurate digital representations of objects, essential for applications in computer graphics, photogrammetry, computer vision, and robotics. However, comparing point clouds faces significant challenges due to their unstructured nature and the complex geometry of the surfaces they represent. Traditional geometric metrics such as Hausdorff and Chamfer distances often fail to capture global statistical structure and exhibit sensitivity to outliers, while existing Kullback-Leibler (KL) divergence approximations for Gaussian Mixture Models can produce unbounded or numerically unstable values. This paper introduces an information geometric framework for 3D point cloud shape analysis by representing point clouds as Gaussian Mixture Models (GMMs) on a statistical manifold. We prove that the space of GMMs forms a statistical manifold and propose the Modified Symmetric Kullback-Leibler (MSKL) divergence with theoretically guaranteed upper and lower bounds, ensuring numerical stability for all GMM comparisons. Through comprehensive experiments on human pose discrimination (MPI-FAUST dataset) and animal shape comparison (G-PCD dataset), we demonstrate that MSKL provides stable and monotonically varying values that directly reflect geometric variation, outperforming traditional distances and existing KL approximations.

---

## 16. Secure AI-Driven Super-Resolution for Real-Time Mixed Reality Applications

**论文链接:** [http://arxiv.org/abs/2512.15823v1](http://arxiv.org/abs/2512.15823v1)

**作者:** Mohammad Waquas Usmani, Sankalpa Timilsina, Michael Zink, Susmit Shannigrahi

**发布时间:** 2025-12-17

### GPT解析

### 总结

该论文提出了一种解决沉浸式格式(360°和6DoF点云视频)在实时AR/VR流媒体中带宽和延迟问题的系统，通过降采样、部分加密和机器学习超分辨率技术实现。

### 背景

沉浸式格式如360°和6DoF点云视频需要高带宽和低延迟，这对实时AR/VR流媒体构成挑战，带宽消耗和加解密延迟是总体延迟的主要因素。

### 目的

减少带宽消耗和加解密延迟，这两个导致总体延迟的关键因素，以改善沉浸式内容的实时流媒体体验。

### 方法

设计一个系统，在源服务器端对点云内容进行降采样并应用部分加密，在客户端使用基于机器学习的超分辨率模型进行解密和放大。

### 主要发现

评估显示，随着降采样分辨率的降低，带宽/延迟和加解密开销呈近似线性减少，超分辨率模型能以最小误差和适度推理时间有效重建原始全分辨率点云。

### 结论

结合降采样、部分加密和基于机器学习的超分辨率技术可有效解决沉浸式内容流媒体中的带宽和延迟问题，同时保持内容质量。

### 翻译

沉浸式格式如360°和6DoF点云视频需要高带宽和低延迟，这对实时AR/VR流媒体构成了挑战。这项工作专注于减少带宽消耗和加解密延迟，这是总体延迟的两个主要贡献因素。我们设计了一个系统，在源服务器端对点云内容进行降采样并应用部分加密。在客户端，内容被解密并使用基于机器学习的超分辨率模型进行放大。我们的评估表明，随着降采样分辨率的降低，带宽/延迟和加解密开销呈近似线性减少，同时超分辨率模型能够以最小误差和适度的推理时间有效重建原始全分辨率点云。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决沉浸式媒体格式（如360°视频和6DoF点云视频）的高带宽需求和低延迟挑战，这对实时AR/VR流媒体传输至关重要。这个问题在现实中非常重要，因为高延迟会导致用户运动 sickness 并降低沉浸感，而高带宽需求则可能导致网络拥塞和数据包丢失，进一步增加延迟。随着Apple Vision Pro和Meta Quest 3等设备的普及，用户能够以高质量消费这些内容，因此需要有效的解决方案来处理这些高要求格式。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先认识到沉浸式媒体格式的带宽和延迟挑战，然后提出结合超分辨率和属性基加密(ABE)的新方法。他们设计了一个系统，在源服务器对点云内容进行下采样并应用部分加密，在客户端进行解密和基于机器学习的上采样。作者借鉴了多项现有工作，包括点云超分辨率研究（如PU-Net）、属性基加密技术、以及他们之前在视频流中使用ABE的研究，特别是针对点云数据的选择性坐标加密方法。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是将点云下采样与基于机器学习的超分辨率和属性基加密相结合，以减少带宽使用和加密/解密开销，同时通过训练AI/ML模型从下采样数据重建全分辨率点云。整体流程包括：1)在源服务器处对点云进行下采样（50%、25%、12.5%）；2)使用ABE对下采样内容进行部分加密；3)通过CDN分发加密内容；4)客户端获取并解密内容；5)使用基于AI/ML的超分辨率模型进行上采样；6)渲染重建的全分辨率点云。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)将安全性集成到点云超分辨率管道中；2)结合ABE与点云下采样和ML超分辨率；3)开发高效AI/ML模型重建全分辨率点云；4)实验证明近线性减少网络延迟、带宽和加密时间。相比之前的工作，本研究首次将安全性与点云超分辨率结合，而之前的研究主要关注带宽优化而忽略安全性；同时，虽然之前有ABE在视频流中的应用，但本研究将其扩展到点云数据并结合了下采样和超分辨率技术。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '本研究通过结合点云下采样、基于机器学习的超分辨率与属性基加密，为实时混合现实应用提供了一种安全高效的解决方案，显著降低了带宽需求和加密开销，同时以最小误差重建原始点云质量。'}


### 论文摘要

Immersive formats such as 360° and 6DoF point cloud videos require high bandwidth and low latency, posing challenges for real-time AR/VR streaming. This work focuses on reducing bandwidth consumption and encryption/decryption delay, two key contributors to overall latency. We design a system that downsamples point cloud content at the origin server and applies partial encryption. At the client, the content is decrypted and upscaled using an ML-based super-resolution model. Our evaluation demonstrates a nearly linear reduction in bandwidth/latency, and encryption/decryption overhead with lower downsampling resolutions, while the super-resolution model effectively reconstructs the original full-resolution point clouds with minimal error and modest inference time.

---

## 17. Diffusion-Based Restoration for Multi-Modal 3D Object Detection in Adverse Weather

**论文链接:** [http://arxiv.org/abs/2512.13107v2](http://arxiv.org/abs/2512.13107v2)

**作者:** Zhijian He, Feifei Liu, Yuwei Li, Zhanpeng Luo, Jintao Cheng, Xieyuanli Chen, Xiaoyu Tang

**发布时间:** 2025-12-15

### GPT解析

### 总结

本文提出了DiffFusion框架，通过基于扩散的恢复和自适应跨模态融合，增强多模态3D目标检测在恶劣天气条件下的鲁棒性。

### 背景

多模态3D目标检测对机器人和自动驾驶中的可靠感知至关重要，但在恶劣天气条件下，其有效性受到天气引起的失真和不同数据模态间错位的限制。

### 目的

开发一个能够增强在挑战性天气条件下鲁棒性的多模态3D目标检测框架。

### 方法

DiffFusion框架利用扩散模型强大的去噪和数据生成能力，引入Diffusion-IR恢复因天气影响退化的图像，使用点云恢复(PCR)利用图像对象线索补偿损坏的LiDAR数据，并开发双向自适应融合和对齐模块(BAFAM)处理模态间错位，实现动态多模态融合和双向鸟瞰图对齐。

### 主要发现

在三个公共数据集上的广泛实验表明，DiffFusion在恶劣天气条件下实现了最先进的鲁棒性，同时保持了强大的干净数据性能；在真实世界DENSE数据集上的零样本结果进一步验证了其泛化能力。

### 结论

DiffFusion框架有效解决了恶劣天气条件下的多模态3D目标检测挑战，实现将作为开源发布。

### 翻译

多模态3D目标检测对机器人和自动驾驶中的可靠感知很重要。然而，由于天气引起的失真和不同数据模态之间的错位，其在恶劣天气条件下的有效性仍然有限。在这项工作中，我们提出了DiffFusion，一个通过基于扩散的恢复和自适应跨模态融合来增强在挑战性天气条件下鲁棒性的新框架。我们的关键见解是扩散模型具有强大的去噪和生成数据的能力，可以适应各种天气条件。基于此，DiffFusion引入了Diffusion-IR来恢复因天气影响而退化的图像，并使用点云恢复(PCR)利用图像对象线索补偿损坏的LiDAR数据。为了解决两种模态之间的错位问题，我们开发了双向自适应融合和对齐模块(BAFAM)。它实现了动态多模态融合和双向鸟瞰图(BEV)对齐，以保持一致的空间对应关系。在三个公共数据集上的广泛实验表明，DiffFusion在恶劣天气条件下实现了最先进的鲁棒性，同时保持了强大的干净数据性能。在真实世界DENSE数据集上的零样本结果进一步验证了其泛化能力。我们的DiffFusion实现将作为开源发布。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决多模态3D目标检测在恶劣天气条件下性能下降的问题，特别是图像和LiDAR传感器数据因天气影响产生的失真和模态间错位。这个问题在自动驾驶和机器人领域非常重要，因为这些系统需要在各种天气条件下可靠运行，而现有方法在雨、雾、强光等条件下性能显著下降，可能导致安全隐患。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先观察到多模态3D检测在恶劣天气下效果不佳，分析现有方法主要针对干净数据优化，缺乏处理天气失真的机制。他们发现扩散模型在去噪和生成数据方面有强大能力，可以适应各种天气条件。方法设计借鉴了扩散模型、特征金字塔网络、CenterNet等现有技术，但创新性地将它们整合到一个专门处理恶劣天气的统一框架中，并设计了新的双向自适应融合和对齐模块来解决模态间错位问题。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是利用扩散模型恢复恶劣天气下的传感器数据，并通过图像检测结果指导点云修复，实现跨模态信息互补，同时设计双向融合机制解决模态间错位。整体流程分为三部分：1)扩散基础恢复模块，包含图像分支(Diffusion-IR)和点云分支(PCR)；2)双向自适应融合和对齐模块(BAFAM)，包含交叉注意力自适应融合和双向BEV对齐；3)将融合后的特征输入3D检测头进行目标检测。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)DiffFusion统一框架系统地解决天气引起的传感器退化；2)双分支恢复结构同时处理图像和点云恢复；3)BAFAM模块实现动态多模态融合和双向BEV对齐。相比之前工作，不同之处在于：传统方法通常针对单一天气条件训练，而DiffFusion可处理多种天气；现有方法缺乏处理模态间错位的机制；大多数方法只关注单一模态恢复，而DiffFusion同时处理两种模态；扩散模型在3D目标检测中的应用是新颖的，特别是与多模态融合的结合。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'DiffFusion通过基于扩散模型的多模态恢复和自适应融合机制，显著提升了3D目标检测在恶劣天气条件下的鲁棒性，同时保持了在正常天气条件下的高性能。'}


### 论文摘要

Multi-modal 3D object detection is important for reliable perception in robotics and autonomous driving. However, its effectiveness remains limited under adverse weather conditions due to weather-induced distortions and misalignment between different data modalities. In this work, we propose DiffFusion, a novel framework designed to enhance robustness in challenging weather through diffusion-based restoration and adaptive cross-modal fusion. Our key insight is that diffusion models possess strong capabilities for denoising and generating data that can adapt to various weather conditions. Building on this, DiffFusion introduces Diffusion-IR restoring images degraded by weather effects and Point Cloud Restoration (PCR) compensating for corrupted LiDAR data using image object cues. To tackle misalignments between two modalities, we develop Bidirectional Adaptive Fusion and Alignment Module (BAFAM). It enables dynamic multi-modal fusion and bidirectional bird's-eye view (BEV) alignment to maintain consistent spatial correspondence. Extensive experiments on three public datasets show that DiffFusion achieves state-of-the-art robustness under adverse weather while preserving strong clean-data performance. Zero-shot results on the real-world DENSE dataset further validate its generalization. The implementation of our DiffFusion will be released as open-source.

---

## 18. Unsupervised Thematic Clustering Of hadith Texts Using The Apriori Algorithm

**论文链接:** [http://arxiv.org/abs/2512.16694v1](http://arxiv.org/abs/2512.16694v1)

**作者:** Wisnu Uriawan, Achmad Ajie Priyajie, Angga Gustian, Fikri Nur Hidayat, Sendi Ahmad Rafiudin, Muhamad Fikri Zaelani

**发布时间:** 2025-12-18

### GPT解析

### 总结

这项研究旨在自动化圣训主题分组，使用无监督学习方法和Apriori算法识别未标记文本数据中的关联模式和语义关系。

### 背景

随着伊斯兰文本数字化进程加速，自动化圣训主题分组的需求日益紧迫。

### 目的

自动化圣训主题分组，促进伊斯兰文本数字化处理。

### 方法

使用布哈里圣训印尼语翻译数据集，经过大小写转换、标点清理、分词、停用词移除和词干提取等预处理，然后应用Apriori算法进行关联规则挖掘分析，使用支持度、置信度和提升度参数。

### 主要发现

发现了拜次数-祈祷、经文-启示和圣训-故事等有意义的关联模式，分别描述了崇拜、启示和圣训叙述的主题。

### 结论

Apriori算法能自动发现潜在语义关系，为数字伊斯兰研究和基于技术的学习系统发展做出贡献。

### 翻译

这项研究源于自动化圣训主题分组的紧迫性，以符合伊斯兰文本日益数字化的发展。基于文献综述，使用Apriori算法的无监督学习方法已被证明在识别未标记文本数据中的关联模式和语义关系方面有效。使用的数据集是布哈里圣训的印尼语翻译，首先经过包括大小写转换、标点符号清理、分词、停用词移除和词干提取的预处理阶段。接下来，使用具有支持度、置信度和提升度参数的Apriori算法进行了关联规则挖掘分析。结果表明存在有意义的关联模式，如拜次数-祈祷、经文-启示和圣训-故事之间的关系，这些关系描述了崇拜、启示和圣训叙述的主题。这些发现表明，Apriori算法能够自动发现潜在的语义关系，同时为数字伊斯兰研究和基于技术的学习系统的发展做出贡献。


### 论文摘要

This research stems from the urgency to automate the thematic grouping of hadith in line with the growing digitalization of Islamic texts. Based on a literature review, the unsupervised learning approach with the Apriori algorithm has proven effective in identifying association patterns and semantic relations in unlabeled text data. The dataset used is the Indonesian Translation of the hadith of Bukhari, which first goes through preprocessing stages including case folding, punctuation cleaning, tokenization, stopword removal, and stemming. Next, an association rule mining analysis was conducted using the Apriori algorithm with support, confidence, and lift parameters. The results show the existence of meaningful association patterns such as the relationship between rakaat-prayer, verse-revelation, and hadith-story, which describe the themes of worship, revelation, and hadith narration. These findings demonstrate that the Apriori algorithm has the ability to automatically uncover latent semantic relationships, while contributing to the development of digital Islamic studies and technology-based learning systems.

---

## 19. Skeleton-Snippet Contrastive Learning with Multiscale Feature Fusion for Action Localization

**论文链接:** [http://arxiv.org/abs/2512.16504v1](http://arxiv.org/abs/2512.16504v1)

**作者:** Qiushuo Cheng, Jingjing Liu, Catherine Morgan, Alan Whone, Majid Mirmehdi

**发布时间:** 2025-12-18

### GPT解析

### 总结

本文提出了一种基于骨架的时间动作定位的自监督学习方法，通过片段判别预任务和特征融合技术提高了动作边界检测的准确性。

### 背景

自监督预训练范式在基于骨架的动作识别中已取得成功，但在时间动作定位方面仍面临挑战。

### 目的

开发能够捕捉时间敏感特征的有效表示学习方法，用于检测动作边界和定位动作片段。

### 方法

提出片段判别预任务，将骨架序列投影到不重叠片段并通过对比学习区分；使用U形模块融合中间特征以增强帧级定位的特征分辨率。

### 主要发现

该方法在BABEL数据集上持续改进了现有方法；在PKUMMD上实现了最先进的迁移学习性能。

### 结论

自监督预训练结合时间敏感的特征学习可以有效解决基于骨架的时间动作定位问题。

### 翻译

自监督预训练范式在使用对比学习学习基于骨架的动作识别的3D动作表示方面取得了巨大成功。然而，学习用于基于骨架的时间动作定位的有效表示仍然具有挑战性且研究不足。与视频级动作识别不同，检测动作边界需要时间敏感的特征，能够捕捉标签变化处相邻帧之间的细微差别。为此，我们提出了一种用于自监督预训练的片段判别预任务，该方法将骨架序列密集投影到不重叠的片段中，并通过对比学习促进跨视频区分这些片段的特征。此外，我们通过使用U形模块融合中间特征，构建了基于骨架的动作识别模型的强大骨干网络，以增强帧级定位的特征分辨率。我们的方法在BABEL数据集上，针对不同的子集和评估协议，持续改进了现有的基于骨架的对比学习方法用于动作定位。在NTU RGB+D和BABEL上进行预训练后，我们在PKUMMD上实现了最先进的迁移学习性能。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决基于骨架的自监督表示学习在动作定位（action localization）任务中的挑战。这个问题很重要，因为获得帧级别的动作标签比视频级别标签更昂贵、耗时，而动作定位不仅是分类动作，还需要检测视频中的时间边界，这对医疗保健（如帕金森病步态分析）和监控等实际应用至关重要。现有骨架对比学习方法通过全局平均池化产生序列级表示，导致学习到的表示对帧级变化不敏感，而这对于需要时间敏感特征的定位任务至关重要。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先分析了现有骨架对比学习方法的局限性，即它们主要关注视频级别动作识别，使用全局投影导致学习到的表示对帧级变化不敏感。作者从图像密集预测任务的对比学习框架中汲取灵感，设计了片段判别代理任务来促进时间敏感特征学习。具体设计包括密集片段对比学习和多尺度特征融合。作者借鉴了MoCo-based视频级别骨架对比学习框架、RGB研究的密集对比学习（DenseCL）以及U-Net和特征金字塔网络的设计。方法采用两阶段范式：第一阶段在自监督对比预训练中集成密集片段级对比目标；第二阶段在微调阶段将U形模块插入预训练编码器以融合多尺度特征。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是：1）片段级对比学习，将骨架序列细分为多个片段并在片段级别进行对比学习，使模型能够学习更细粒度的时间敏感特征；2）多尺度特征融合，使用U形模块融合不同尺度特征，恢复时间分辨率并保留语义信息和细粒度细节。整体流程分为两阶段：第一阶段是密集片段对比预训练，输入骨架序列后通过不同增强获得锚点-正样本对，使用编码器获取骨架编码，同时应用全局投影和密集投影模块构建视频级和片段级嵌入，通过基于相似性的匹配确定片段对应关系，计算视频级和片段级对比损失；第二阶段是多尺度特征融合微调，保留预训练编码器作为骨干，插入U形模块通过跳跃连接融合不同尺度特征并逐步上采样，最后添加分类器生成帧级预测并进行后处理。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1）片段级判别代理任务，首次在未修剪、无标签骨架序列上学习时间敏感表示；2）密集片段对比学习，将骨架序列密集投影到非重叠片段并通过对比学习促进跨视频区分；3）多尺度特征融合的U形模块，增强现有骨干网络的特征分辨率；4）两阶段训练范式，专门针对动作定位任务。相比之前工作的不同：1）对比粒度从视频级别提升到片段级别，提高时间敏感性；2）应用任务从动作识别转向更具挑战性的动作定位；3）能从未修剪、无标签骨架序列中学习；4）引入U形模块进行多尺度特征融合；5）使用基于相似性的匹配策略确定片段对应关系，而非假设相同时间位置的片段在增强后仍然语义相关。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '这篇论文通过引入片段级对比学习和多尺度特征融合，首次实现了在未修剪骨架序列上学习时间敏感表示，显著提高了骨架动作定位的性能，并减少了对密集帧级标签的依赖。'}


### 论文摘要

The self-supervised pretraining paradigm has achieved great success in learning 3D action representations for skeleton-based action recognition using contrastive learning. However, learning effective representations for skeleton-based temporal action localization remains challenging and underexplored. Unlike video-level {action} recognition, detecting action boundaries requires temporally sensitive features that capture subtle differences between adjacent frames where labels change. To this end, we formulate a snippet discrimination pretext task for self-supervised pretraining, which densely projects skeleton sequences into non-overlapping segments and promotes features that distinguish them across videos via contrastive learning. Additionally, we build on strong backbones of skeleton-based action recognition models by fusing intermediate features with a U-shaped module to enhance feature resolution for frame-level localization. Our approach consistently improves existing skeleton-based contrastive learning methods for action localization on BABEL across diverse subsets and evaluation protocols. We also achieve state-of-the-art transfer learning performance on PKUMMD with pretraining on NTU RGB+D and BABEL.

---

## 20. Advantages and limitations in the use of transfer learning for individual treatment effects in causal machine learning

**论文链接:** [http://arxiv.org/abs/2512.16489v1](http://arxiv.org/abs/2512.16489v1)

**作者:** Seyda Betul Aydin, Holger Brandt

**发布时间:** 2025-12-18

### GPT解析

### 总结

研究展示了如何通过迁移学习改进个体治疗效果(ITE)的估计，特别是在小样本情境下，通过利用源数据集知识并将其适应到新环境中。

### 背景

将因果知识推广到不同环境具有挑战性，尤其是当需要将大规模数据集的估计应用于较小或系统性不同的情境时。基于机器学习的ITE估计器需要大样本量，限制了其在行为科学等拥有较小数据集领域的应用。

### 目的

展示如何利用源数据集知识并通过迁移学习将其适应到新环境中，以改进ITE的估计。

### 方法

使用Treatment Agnostic Representation Networks (TARNet)估计ITE，并通过迁移学习改进为TL-TARNet。在模拟中改变源样本量和目标样本量，考虑随机化和非随机化干预设置。在印度人类发展调查(IHDS-II)中实证应用，估计母亲收集柴火时间对孩子学习时间的影响。

### 主要发现

迁移学习扩展TL-TARNet优于标准TARNet，当存在大的无偏源样本且目标样本量小时，减少了ITE误差并减弱了偏差。在实证应用中，迁移学习将目标平均ITE拉向源ITE估计，减少了无迁移获得的估计中的偏差。

### 结论

因果模型的迁移学习可以改进小样本中的ITE估计。

### 翻译

将因果知识推广到不同环境具有挑战性，特别是当必须将大规模数据集的估计应用于较小或系统性不同的情境时，外部有效性至关重要。来自机器学习的个体治疗效果(ITE)的基于模型的估计器需要大样本量，限制了其在行为科学等拥有较小数据集的领域的适用性。我们展示了如何通过利用源数据集的知识并通过迁移学习将其适应到新环境中来改进使用Treatment Agnostic Representation Networks (TARNet; Shalit et al., 2017)的ITE估计(TL-TARNet; Aloui et al., 2023)。在改变源样本量和目标样本量并考虑随机化和非随机化干预目标设置的模拟中，迁移学习扩展TL-TARNet优于标准TARNet，当存在大的无偏源且目标样本量小时，减少了ITE误差并减弱了偏差。在使用印度人类发展调查(IHDS-II)的实证应用中，我们估计母亲收集柴火时间对孩子每周学习时间的影响；迁移学习将目标平均ITE拉向源ITE估计，减少了无迁移获得的估计中的偏差。这些结果表明，因果模型的迁移学习可以改进小样本中的ITE估计。


### 论文摘要

Generalizing causal knowledge across diverse environments is challenging, especially when estimates from large-scale datasets must be applied to smaller or systematically different contexts, where external validity is critical. Model-based estimators of individual treatment effects (ITE) from machine learning require large sample sizes, limiting their applicability in domains such as behavioral sciences with smaller datasets. We demonstrate how estimation of ITEs with Treatment Agnostic Representation Networks (TARNet; Shalit et al., 2017) can be improved by leveraging knowledge from source datasets and adapting it to new settings via transfer learning (TL-TARNet; Aloui et al., 2023). In simulations that vary source and sample sizes and consider both randomized and non-randomized intervention target settings, the transfer-learning extension TL-TARNet improves upon standard TARNet, reducing ITE error and attenuating bias when a large unbiased source is available and target samples are small. In an empirical application using the India Human Development Survey (IHDS-II), we estimate the effect of mothers' firewood collection time on children's weekly study time; transfer learning pulls the target mean ITEs toward the source ITE estimate, reducing bias in the estimates obtained without transfer. These results suggest that transfer learning for causal models can improve the estimation of ITE in small samples.

---

## 21. Machine Learning-based Optimal Control for Colloidal Self-Assembly

**论文链接:** [http://arxiv.org/abs/2512.16402v1](http://arxiv.org/abs/2512.16402v1)

**作者:** Andres Lizano-Villalobos, Fangyuan Ma, Wentao Tang, Wei Sun, Xun Tang

**发布时间:** 2025-12-18

**备注:** 19 pages, 5 figures, 1 table

### GPT解析

### 总结

研究展示了机器学习在胶体自组装控制中的应用潜力，通过结合无监督学习和深度强化学习，实现了高精度的胶体自组装控制。

### 背景

精确控制胶体自组装成特定图案一直是一个长期存在的挑战，因为过程动力学复杂。

### 目的

采用机器学习为基础的最优控制框架，提供一种数据驱动的控制方法，可以推广到其他多体自组装系统。

### 方法

采用机器学习为基础的最优控制框架，结合无监督学习和图卷积神经网络进行状态观测，使用基于深度强化学习的最优控制策略计算，并通过布朗动力学模拟进行验证。

### 主要发现

与传统的基于序参数的状态描述相比表现出优越的性能，在电场介导的系统中获得有序的二维球形胶体自组装，实际成功率达到97%。

### 结论

该方法为实现可自动化和可推广的图案化胶体组装提供了有前景的途径。

### 翻译

实现胶体自组装成特定图案的精确控制长期以来一直是一个挑战，这源于复杂的过程动力学。最近，基于机器学习的状态表示和基于强化学习的控制策略在该领域开始积累人气，显示出在实现可自动化和可推广的图案化胶体组装方法方面具有巨大潜力。在这项工作中，我们采用了基于机器学习的最优控制框架，结合无监督学习和图卷积神经网络进行状态观测，并使用基于深度强化学习的最优控制策略计算，提供了一种数据驱动的控制方法，该方法可能推广到其他多体自组装系统。通过布朗动力学模拟，我们证明了其与传统基于序参数的状态描述相比具有优越的性能，以及在电场介导系统中获得有序二维球形胶体自组装的有效性，实际成功率达到97%。


### 论文摘要

Achieving precise control of colloidal self-assembly into specific patterns remains a longstanding challenge due to the complex process dynamics. Recently, machine learning-based state representation and reinforcement learning-based control strategies have started to accumulate popularity in the field, showing great potential in achieving an automatable and generalizable approach to producing patterned colloidal assembly. In this work, we adopted a machine learning-based optimal control framework, combining unsupervised learning and graph convolutional neural work for state observation with deep reinforcement learning-based optimal control policy calculation, to provide a data-driven control approach that can potentially be generalized to other many-body self-assembly systems. With Brownian dynamics simulations, we demonstrated its superior performance as compared to traditional order parameter-based state description, and its efficacy in obtaining ordered 2-dimensional spherical colloidal self-assembly in an electric field-mediated system with an actual success rate of 97%.

---

## 22. Pretrained Battery Transformer (PBT): A battery life prediction foundation model

**论文链接:** [http://arxiv.org/abs/2512.16334v1](http://arxiv.org/abs/2512.16334v1)

**作者:** Ruifeng Tan, Weixiang Hong, Jia Li, Jiaqiang Huang, Tong-Yi Zhang

**发布时间:** 2025-12-18

**备注:** 5 figures in the main content

### GPT解析

### 总结

研究人员开发了Pretrained Battery Transformer (PBT)，这是首个用于电池寿命预测的基础模型，通过领域知识编码的专家混合层实现，在多个锂离子电池数据集上表现出色。

### 背景

电池循环寿命的早期预测对加速电池研究、制造和部署至关重要，但机器学习方法因数据稀缺性和不同老化条件导致的异质性而进展受阻。

### 目的

开发一种基础模型来解决电池寿命预测中的数据稀缺性和异质性问题，实现更准确的预测。

### 方法

提出Pretrained Battery Transformer (PBT)，通过领域知识编码的专家混合层开发，并在最大的公共电池寿命数据库上验证，从13个锂离子电池数据集中学习可迁移的表示。

### 主要发现

PBT比现有模型平均性能提高19.8%，通过迁移学习在涵盖各种操作条件、形成协议和锂离子电池化学成分的15个不同数据集上实现了最先进的性能。

### 结论

这项工作为电池寿命预测建立了基础模型途径，为开发通用电池寿命预测系统铺平了道路。

### 翻译

电池循环寿命的早期预测对于加速电池研究、制造和部署至关重要。尽管机器学习方法已显示出令人鼓舞的结果，但由于不同老化条件导致的数据稀缺性和异质性，进展受到阻碍。在其他领域，通过迁移学习在多样化数据集上训练的基础模型已实现了广泛泛化，但尚未有报道用于电池循环寿命预测的基础模型。在此，我们提出了Pretrained Battery Transformer (PBT)，这是首个用于电池寿命预测的基础模型，通过领域知识编码的专家混合层开发。在最大的公共电池寿命数据库上验证，PBT从13个锂离子电池数据集中学习可迁移的表示，比现有模型平均高出19.8%。通过迁移学习，PBT在涵盖各种操作条件、形成协议和锂离子电池化学成分的15个不同数据集上实现了最先进的性能。这项工作为电池寿命预测建立了基础模型途径，为通用电池寿命预测系统铺平了道路。


### 论文摘要

Early prediction of battery cycle life is essential for accelerating battery research, manufacturing, and deployment. Although machine learning methods have shown encouraging results, progress is hindered by data scarcity and heterogeneity arising from diverse aging conditions. In other fields, foundation models (FMs) trained on diverse datasets have achieved broad generalization through transfer learning, but no FMs have been reported for battery cycle life prediction yet. Here we present the Pretrained Battery Transformer (PBT), the first FM for battery life prediction, developed through domain-knowledge-encoded mixture-of-expert layers. Validated on the largest public battery life database, PBT learns transferable representations from 13 lithium-ion battery (LIB) datasets, outperforming existing models by an average of 19.8%. With transfer learning, PBT achieves state-of-the-art performance across 15 diverse datasets encompassing various operating conditions, formation protocols, and chemistries of LIBs. This work establishes a foundation model pathway for battery lifetime prediction, paving the way toward universal battery lifetime prediction systems.

---

## 23. Towards Fine-Tuning-Based Site Calibration for Knowledge-Guided Machine Learning: A Summary of Results

**论文链接:** [http://arxiv.org/abs/2512.16013v1](http://arxiv.org/abs/2512.16013v1)

**作者:** Ruolei Zeng, Arun Sharma, Shuai An, Mingzhou Yang, Shengya Zhang, Licheng Liu, David Mulla, Shashi Shekhar

**发布时间:** 2025-12-17

### GPT解析

### 总结

研究提出了FTBSC-KGML框架，一种基于预训练和微调的、具有空间变异性感知能力的、知识引导的机器学习方法，用于准确量化农业生态系统碳循环。

### 背景

准确且经济有效地量化决策相关尺度的农业生态系统碳循环对气候缓解和可持续农业至关重要，但迁移学习和空间变异性的利用面临挑战，传统方法依赖位置无关参数化和独立训练，未能充分利用空间异质性。

### 目的

开发一种能够利用迁移学习和空间异质性的机器学习框架，以准确量化农业生态系统碳循环，特别是在数据有限的情况下提高局部准确性。

### 方法

提出FTBSC-KGML框架，通过预训练-微调过程结合遥感GPP、气候和土壤协变量，采用空间异质性感知的迁移学习方案，在每个州或站点对全局预训练模型进行微调，学习位置感知表示。

### 主要发现

FTBSC-KGML比纯全局模型具有更低的验证误差和更大的解释能力一致性，能够更好地捕捉各州之间的空间变异性，在有限数据情况下提高局部准确性而不牺牲可解释性。

### 结论

这项工作扩展了先前的SDSA-KGML框架，为农业生态系统碳循环的准确量化提供了新的机器学习方法。

### 翻译

准确且经济有效地量化决策相关尺度的农业生态系统碳循环对于气候缓解和可持续农业至关重要。然而，在这个领域中，迁移学习和空间变异性的利用具有挑战性，因为它们涉及异构数据和复杂的跨尺度依赖关系。传统方法通常依赖于位置无关的参数化和独立训练，未能充分利用迁移学习和输入数据的空间异质性，限制了其在具有显著变异性区域的应用。我们提出了FTBSC-KGML（基于微调的站点校准-知识引导机器学习），一种基于预训练和微调的、具有空间变异性感知能力的、知识引导的机器学习框架，通过预训练-微调过程增强KGML-ag，并加入站点特定参数。使用从多个中西部站点收集的遥感GPP、气候和土壤协变量进行预训练-微调过程，FTBSC-KGML估计土地排放，同时利用迁移学习和空间异质性。关键组件是一个空间异质性感知的迁移学习方案，这是一个全局预训练的模型，在每个州或站点进行微调，以学习位置感知的表示，从而在有限数据的情况下提高局部准确性，同时不牺牲可解释性。实证表明，FTBSC-KGML比纯全局模型具有更低的验证误差和更大的解释能力一致性，从而更好地捕捉各州之间的空间变异性。这项工作扩展了先前的SDSA-KGML框架。


### 论文摘要

Accurate and cost-effective quantification of the agroecosystem carbon cycle at decision-relevant scales is essential for climate mitigation and sustainable agriculture. However, both transfer learning and the exploitation of spatial variability in this field are challenging, as they involve heterogeneous data and complex cross-scale dependencies. Conventional approaches often rely on location-independent parameterizations and independent training, underutilizing transfer learning and spatial heterogeneity in the inputs, and limiting their applicability in regions with substantial variability. We propose FTBSC-KGML (Fine-Tuning-Based Site Calibration-Knowledge-Guided Machine Learning), a pretraining- and fine-tuning-based, spatial-variability-aware, and knowledge-guided machine learning framework that augments KGML-ag with a pretraining-fine-tuning process and site-specific parameters. Using a pretraining-fine-tuning process with remote-sensing GPP, climate, and soil covariates collected across multiple midwestern sites, FTBSC-KGML estimates land emissions while leveraging transfer learning and spatial heterogeneity. A key component is a spatial-heterogeneity-aware transfer-learning scheme, which is a globally pretrained model that is fine-tuned at each state or site to learn place-aware representations, thereby improving local accuracy under limited data without sacrificing interpretability. Empirically, FTBSC-KGML achieves lower validation error and greater consistency in explanatory power than a purely global model, thereby better capturing spatial variability across states. This work extends the prior SDSA-KGML framework.

---

## 24. An updated efficient galaxy morphology classification model based on ConvNeXt encoding with UMAP dimensionality reduction

**论文链接:** [http://arxiv.org/abs/2512.15137v2](http://arxiv.org/abs/2512.15137v2)

**作者:** Guanwen Fang, Shiwei Zhu, Jun Xu, Shiying Lu, Chichun Zhou, Yao Dai, Zesen Lin, Xu Kong

**发布时间:** 2025-12-17

**备注:** Accepted for publication in AJ

### GPT解析

### 总结

研究团队在USmorph分类框架中引入了增强的无监督机器学习模块，结合预训练ConvNeXt神经网络和UMAP降维技术，成功优化了星系形态分类，实现了计算效率的大幅提升。

### 背景

研究人员之前已开发USmorph分类框架，现需要改进该框架以提高星系形态分类的效率和准确性，以适应未来大规模巡天观测的需求。

### 目的

开发一个增强的无监督机器学习模块，用于更高效、准确地分类星系形态，使其适合中国空间站望远镜(CSST)等计划进行的大规模巡天观测。

### 方法

使用预训练的ConvNeXt卷积神经网络进行分层特征提取，结合UMAP进行非线性流形学习实现拓扑感知降维；应用于99,806个红移0.2<z<1.2的COSMOS星系I波段图像；将聚类数量从50优化为20；将20个算法识别聚类合并为五种物理形态类型；测试质量大于10^9太阳质量的巨大星系形态参数。

### 主要发现

约51%的星系(50,056个)成功分类；分类结果与星系演化理论高度一致；聚类数量减少实现了显著计算节省；改进算法显著提高了星系形态分类效率。

### 结论

改进的算法显著提高了星系形态分类效率，使其适合大规模巡天观测，如中国空间站望远镜(CSST)计划进行的观测。

### 翻译

我们在之前的USmorph分类框架中提出了一个增强的无监督机器学习模块，包含两个组件：(1)通过使用迁移学习的预训练ConvNeXt卷积神经网络进行分层特征提取，以及(2)使用均匀流形近似和投影进行非线性流形学习，实现拓扑感知的降维。这种双阶段设计能够从大规模视觉数据集高效迁移知识，同时通过UMAP的邻域保持保留形态模式几何结构。我们将升级后的UML应用于99,806个红移0.2<z<1.2(确保静止帧光学形态)且I波段星等小于25的COSMOS星系的I波段图像。预定义的聚类数量被优化为20(从原始框架中的50减少)，实现了显著的计算节省。20个算法识别的聚类被合并为五种物理形态类型。约51%的星系(50,056个)被成功分类。为了评估分类效果，我们测试了质量大于10^9太阳质量的巨大星系的形态参数。我们的分类结果与星系演化理论高度一致。这种改进的算法显著提高了星系形态分类效率，使其适合中国空间站望远镜等计划进行的大规模巡天观测。


### 论文摘要

We present an enhanced unsupervised machine learning (UML) module within our previous \texttt{USmorph} classification framework featuring two components: (1) hierarchical feature extraction via a pre-trained ConvNeXt convolutional neural network (CNN) with transfer learning, and (2) nonlinear manifold learning using Uniform Manifold Approximation and Projection (UMAP) for topology-aware dimensionality reduction. This dual-stage design enables efficient knowledge transfer from large-scale visual datasets while preserving morphological pattern geometry through UMAP's neighborhood preservation. We apply the upgraded UML on I-band images of 99,806 COSMOS galaxies at redshift $0.2<z<1.2$ (to ensure rest-frame optical morphology) with $I_{\mathrm{mag}}<25$. The predefined cluster number is optimized to 20 (reduced from 50 in the original framework), achieving significant computational savings. The 20 algorithmically identified clusters are merged into five physical morphology types. About 51\% of galaxies (50,056) were successfully classified. To assess classification effectiveness, we tested morphological parameters for massive galaxies with $M_{*}>10^{9}~M_{\odot}$. Our classification results align well with galaxy evolution theory. This improved algorithm significantly enhances galaxy morphology classification efficiency, making it suitable for large-scale sky surveys such as those planned with the China Space Station Telescope (CSST).

---

## 25. Probabilistic Predictions of Process-Induced Deformation in Carbon/Epoxy Composites Using a Deep Operator Network

**论文链接:** [http://arxiv.org/abs/2512.13746v2](http://arxiv.org/abs/2512.13746v2)

**作者:** Elham Kiyani, Amit Makarand Deshpande, Madhura Limaye, Zhiwei Gao, Sai Aditya Pradeep, Srikanth Pilla, Gang Li, Zhen Li, George Em Karniadakis

**发布时间:** 2025-12-15

**备注:** 21 pages, 13 figures

### GPT解析

### 总结

该研究解决了复合材料制造中的工艺诱导变形问题，通过结合物理模型和深度学习方法，实现了PID的准确预测和优化。

### 背景

纤维增强体和聚合物基体对制造条件的响应不同，由于热膨胀系数不匹配和热固性树脂固化过程中的基体收缩，这些异质性在多个长度尺度上产生残余应力，部分释放导致工艺诱导变形(PID)。

### 目的

需要准确预测和减轻PID，通过优化的非等温固化周期来实现。

### 方法

研究考虑单向AS4碳纤维/胺双功能环氧预浸料，使用双机制框架模拟PID；开发基于深度算子网络(DeepONets)的数据驱动替代模型；扩展到特征线性调制(FiLM) DeepONet；使用迁移学习解决实验数据有限问题；使用集合卡尔曼反演(EKI)量化不确定性。

### 主要发现

物理模型和非等温固化周期可预测和减轻PID；DeepONet和FiLM DeepONet能有效预测PID；迁移学习方法可解决实验数据有限问题；EKI可量化不确定性并支持优化固化计划。

### 结论

开发了结合物理模型和数据驱动方法的综合框架，能够预测、量化和优化复合材料制造过程中的PID。

### 翻译

纤维增强和聚合物基体由于热膨胀系数不匹配和热固性树脂固化过程中的基体收缩，对制造条件的响应不同。这些异质性在多个长度尺度上产生残余应力，其部分释放导致工艺诱导变形(PID)，需要通过优化的非等温固化周期进行准确预测和缓解。本研究考虑单向AS4碳纤维/胺双功能环氧预浸料，并使用考虑热膨胀/收缩和固化收缩的双机制框架模拟PID。该模型经过制造试验验证以确定初始和边界条件，然后用于为多样化的非等温固化周期(时间-温度曲线)生成PID响应。基于这一物理基础，我们开发了基于深度算子网络(DeepONets)的数据驱动替代模型。DeepONet在一个结合高保真模拟和PID目标实验测量的数据集上进行训练。我们将其扩展到特征线性调制(FiLM) DeepONet，其中分支网络特征由包括初始固化程度的外部参数调制，从而能够预测固化程度、粘度和变形的时间历程。由于实验数据仅在有限时间点可用(例如最终变形)，我们使用迁移学习：模拟训练的主干和分支网络保持固定，仅使用测量的最终变形更新最后一层。最后，我们使用集合卡尔曼反演(EKI)增强该框架，量化实验条件下的不确定性，并支持优化复合材料中减少PID的固化计划。


### 论文摘要

Fiber reinforcement and polymer matrix respond differently to manufacturing conditions due to mismatch in coefficient of thermal expansion and matrix shrinkage during curing of thermosets. These heterogeneities generate residual stresses over multiple length scales, whose partial release leads to process-induced deformation (PID), requiring accurate prediction and mitigation via optimized non-isothermal cure cycles. This study considers a unidirectional AS4 carbon fiber/amine bi-functional epoxy prepreg and models PID using a two-mechanism framework that accounts for thermal expansion/shrinkage and cure shrinkage. The model is validated against manufacturing trials to identify initial and boundary conditions, then used to generate PID responses for a diverse set of non-isothermal cure cycles (time-temperature profiles). Building on this physics-based foundation, we develop a data-driven surrogate based on Deep Operator Networks (DeepONets). A DeepONet is trained on a dataset combining high-fidelity simulations with targeted experimental measurements of PID. We extend this to a Feature-wise Linear Modulation (FiLM) DeepONet, where branch-network features are modulated by external parameters, including the initial degree of cure, enabling prediction of time histories of degree of cure, viscosity, and deformation. Because experimental data are available only at limited time instances (for example, final deformation), we use transfer learning: simulation-trained trunk and branch networks are fixed and only the final layer is updated using measured final deformation. Finally, we augment the framework with Ensemble Kalman Inversion (EKI) to quantify uncertainty under experimental conditions and to support optimization of cure schedules for reduced PID in composites.

---

## 26. R3ST: A Synthetic 3D Dataset With Realistic Trajectories

**论文链接:** [http://arxiv.org/abs/2512.16784v1](http://arxiv.org/abs/2512.16784v1)

**作者:** Simone Teglia, Claudia Melis Tonti, Francesco Pro, Leonardo Russo, Andrea Alfarano, Leonardo Pentassuglia, Irene Amerini

**发布时间:** 2025-12-18

**DOI:** 10.1007/978-3-032-05060-1_30

### GPT解析

### 总结

这篇论文介绍了R3ST（Realistic 3D Synthetic Trajectories）合成数据集，它通过结合合成3D环境和来自真实世界的轨迹，解决了现有合成数据集中车辆运动不够真实的问题，为道路车辆轨迹预测研究提供了准确的地面真实注释和真实的人类驾驶车辆轨迹。

### 背景

数据集对于训练和评估用于交通分析和提高道路安全的计算机视觉模型至关重要。现有的真实数据集适合真实场景，捕捉真实的道路物体行为，但通常缺乏精确的地面真实注释。相比之下，合成数据集可以在没有额外成本或时间的情况下标注大量帧，但通常缺乏真实的车辆运动，因为轨迹是使用AI模型或基于规则的系统生成的。

### 目的

创建一个合成数据集，克服现有合成数据集缺乏真实车辆运动的局限性，缩小合成数据与真实轨迹之间的差距，推进道路车辆轨迹预测的研究。

### 方法

引入R3ST（Realistic 3D Synthetic Trajectories）合成数据集，生成合成3D环境，并整合来自SinD（无人机航拍记录的鸟瞰图数据集）的真实世界轨迹。

### 主要发现

所提出的数据集成功缩小了合成数据与真实轨迹之间的差距，提供了准确的多模态地面真实注释和真实的人类驾驶车辆轨迹。

### 结论

R3ST数据集通过结合合成环境和真实轨迹，解决了现有数据集的局限性，为交通分析和道路安全研究提供了更好的工具。

### 翻译

数据集对于训练和评估用于交通分析和提高道路安全的计算机视觉模型至关重要。现有的真实数据集适合真实场景，捕捉真实的道路物体行为，但通常缺乏精确的地面真实注释。相比之下，合成数据集可以在没有额外成本或时间的情况下标注大量帧，扮演着重要角色。然而，合成数据集的一个普遍缺点是缺乏真实的车辆运动，因为轨迹是使用AI模型或基于规则的系统生成的。在这项工作中，我们引入了R3ST（Realistic 3D Synthetic Trajectories），一个合成数据集，它通过生成合成3D环境并整合来自SinD（无人机航拍记录的鸟瞰图数据集）的真实世界轨迹，克服了这一局限性。所提出的数据集缩小了合成数据与真实轨迹之间的差距，推进了道路车辆轨迹预测的研究，同时提供了准确的多模态地面真实注释和真实的人类驾驶车辆轨迹。


### 论文摘要

Datasets are essential to train and evaluate computer vision models used for traffic analysis and to enhance road safety. Existing real datasets fit real-world scenarios, capturing authentic road object behaviors, however, they typically lack precise ground-truth annotations. In contrast, synthetic datasets play a crucial role, allowing for the annotation of a large number of frames without additional costs or extra time. However, a general drawback of synthetic datasets is the lack of realistic vehicle motion, since trajectories are generated using AI models or rule-based systems. In this work, we introduce R3ST (Realistic 3D Synthetic Trajectories), a synthetic dataset that overcomes this limitation by generating a synthetic 3D environment and integrating real-world trajectories derived from SinD, a bird's-eye-view dataset recorded from drone footage. The proposed dataset closes the gap between synthetic data and realistic trajectories, advancing the research in trajectory forecasting of road vehicles, offering both accurate multimodal ground-truth annotations and authentic human-driven vehicle trajectories.

---

## 27. Seeing is Believing (and Predicting): Context-Aware Multi-Human Behavior Prediction with Vision Language Models

**论文链接:** [http://arxiv.org/abs/2512.15957v1](http://arxiv.org/abs/2512.15957v1)

**作者:** Utsav Panchal, Yuchen Liu, Luigi Palmieri, Ilche Georgievski, Marco Aiello

**发布时间:** 2025-12-17

**备注:** Accepted at IEEE/CVF Winter Conference on Applications of Computer Vision (WACV) 2026

### GPT解析

### 总结

本文提出了一种名为CAMP-VLM的视觉语言模型框架，用于从第三人称视角预测多人类行为，该框架通过整合视觉输入的上下文特征和场景图的空间感知来增强人类-场景交互预测能力。

### 背景

准确预测人类行为对于在人类环境中运行的移动机器人至关重要。然而，先前的研究主要关注从第一人称视角预测单人场景中的动作，而多个机器人应用需要从第三人称视角理解多个人类行为。

### 目的

开发一种能够从第三人称视角预测多人类行为的框架，弥补现有研究的不足，满足多个机器人应用的需求。

### 方法

提出CAMP-VLM框架，整合视觉输入的上下文特征和场景图的空间感知。由于缺乏适合的数据集，研究人员使用逼真模拟器生成的合成人类行为数据对模型进行微调，并在合成和真实世界序列上评估模型性能。

### 主要发现

CAMP-VLM在预测准确性上比最佳基线模型提高了高达66.9%，证明了其在多人类行为预测任务上的优越性能和良好的泛化能力。

### 结论

CAMP-VLM框架有效地解决了从第三人称视角预测多人类行为的挑战，通过结合视觉语言模型、上下文特征和空间感知，显著提高了预测准确性。

### 翻译

准确预测人类行为对于在人类环境中运行的移动机器人至关重要。虽然先前的研究主要关注从第一人称视角预测单人场景中的动作，但多个机器人应用需要从第三人称视角理解多个人类行为。为此，我们提出了CAMP-VLM（上下文感知的多人类行为预测）：一种基于视觉语言模型的框架，它整合了视觉输入的上下文特征和场景图的空间感知，以增强人类-场景交互的预测能力。由于缺乏适合从观察者视角进行多人类行为预测的数据集，我们使用逼真模拟器生成的合成人类行为数据对CAMP-VLM进行微调，并在合成和真实世界序列上评估生成的模型，以评估其泛化能力。利用监督微调（SFT）和直接偏好优化（DPO），CAMP-VLM在预测准确性上比最佳基线模型提高了高达66.9%。


### 论文摘要

Accurately predicting human behaviors is crucial for mobile robots operating in human-populated environments. While prior research primarily focuses on predicting actions in single-human scenarios from an egocentric view, several robotic applications require understanding multiple human behaviors from a third-person perspective. To this end, we present CAMP-VLM (Context-Aware Multi-human behavior Prediction): a Vision Language Model (VLM)-based framework that incorporates contextual features from visual input and spatial awareness from scene graphs to enhance prediction of humans-scene interactions. Due to the lack of suitable datasets for multi-human behavior prediction from an observer view, we perform fine-tuning of CAMP-VLM with synthetic human behavior data generated by a photorealistic simulator, and evaluate the resulting models on both synthetic and real-world sequences to assess their generalization capabilities. Leveraging Supervised Fine-Tuning (SFT) and Direct Preference Optimization (DPO), CAMP-VLM outperforms the best-performing baseline by up to 66.9% in prediction accuracy.

---

## 28. Alchemist: Unlocking Efficiency in Text-to-Image Model Training via Meta-Gradient Data Selection

**论文链接:** [http://arxiv.org/abs/2512.16905v1](http://arxiv.org/abs/2512.16905v1)

**作者:** Kaixin Ding, Yang Zhou, Xi Chen, Miao Yang, Jiarong Ou, Rui Chen, Xin Tao, Hengshuang Zhao

**发布时间:** 2025-12-18

**备注:** project page: https://kxding.github.io/project/Alchemist/

### GPT解析

### 总结

该论文提出了Al框架，一种基于元梯度的数据选择方法，用于提高文本到图像生成模型的训练效率和效果。

### 背景

文本到图像生成模型如Imagen、Stable Diffusion和FLUX虽在视觉质量上取得进步，但其性能受训练数据质量限制。网络爬取和合成数据集常含低质量或冗余样本，导致视觉保真度降低、训练不稳定和计算效率低下。现有方法依赖昂贵的人工筛选或基于单维特征的启发式评分。

### 目的

开发一种自动、可扩展的数据选择框架，提高文本到图像模型训练的数据效率，并解决图像模态下的元学习应用问题。

### 方法

提出Alchemist框架，包含数据评级和数据修剪两个阶段。训练轻量级评分器基于梯度信息估计样本影响，并增强多粒度感知；使用Shift-G采样策略选择信息丰富的子集进行高效模型训练。

### 主要发现

Alchemist是首个用于文本到图像模型训练的自动、可扩展、基于元梯度的数据选择框架。实验表明，使用Alchemist选择的50%数据进行训练可优于使用完整数据集的效果，持续提高视觉质量和下游性能。

### 结论

Alchemist通过智能选择数据子集，有效解决了文本到图像生成模型中的数据质量问题，在减少计算资源的同时提高了模型性能。

### 翻译

最近的文本到图像生成模型进展，如Imagen、Stable Diffusion和FLUX，已在视觉质量方面取得了显著改进。然而，它们的性能从根本上受限于训练数据的质量。网络爬取和合成图像数据集通常包含低质量或冗余样本，导致视觉保真度降低、训练不稳定和计算效率低下。因此，有效的数据选择对提高数据效率至关重要。现有方法依赖于昂贵的人工筛选或基于文本到图像数据过滤中单维特征的启发式评分。尽管基于元学习的方法已在大型语言模型中得到探索，但尚未有针对图像模态的适应。为此，我们提出了Alchemist，一个基于元梯度的框架，用于从大规模文本-图像数据对中选择合适的子集。我们的方法从数据中心视角迭代优化模型，自动学习评估每个样本的影响。Alchemist包含两个关键阶段：数据评级和数据修剪。我们训练一个轻量级评分器，基于梯度信息估计每个样本的影响，并增强多粒度感知。然后我们使用Shift-G采样策略选择信息丰富的子集以进行高效模型训练。Alchemist是首个用于文本到图像模型训练的自动、可扩展、基于元梯度的数据选择框架。在合成和网络爬取数据集上的实验表明，Alchemist持续提高了视觉质量和下游性能。使用Alchemist选择的50%数据进行训练可以优于使用完整数据集进行训练的效果。


### 论文摘要

Recent advances in Text-to-Image (T2I) generative models, such as Imagen, Stable Diffusion, and FLUX, have led to remarkable improvements in visual quality. However, their performance is fundamentally limited by the quality of training data. Web-crawled and synthetic image datasets often contain low-quality or redundant samples, which lead to degraded visual fidelity, unstable training, and inefficient computation. Hence, effective data selection is crucial for improving data efficiency. Existing approaches rely on costly manual curation or heuristic scoring based on single-dimensional features in Text-to-Image data filtering. Although meta-learning based method has been explored in LLM, there is no adaptation for image modalities. To this end, we propose **Alchemist**, a meta-gradient-based framework to select a suitable subset from large-scale text-image data pairs. Our approach automatically learns to assess the influence of each sample by iteratively optimizing the model from a data-centric perspective. Alchemist consists of two key stages: data rating and data pruning. We train a lightweight rater to estimate each sample's influence based on gradient information, enhanced with multi-granularity perception. We then use the Shift-Gsampling strategy to select informative subsets for efficient model training. Alchemist is the first automatic, scalable, meta-gradient-based data selection framework for Text-to-Image model training. Experiments on both synthetic and web-crawled datasets demonstrate that Alchemist consistently improves visual quality and downstream performance. Training on an Alchemist-selected 50% of the data can outperform training on the full dataset.

---

## 29. Few-Shot Inference of Human Perceptions of Robot Performance in Social Navigation Scenarios

**论文链接:** [http://arxiv.org/abs/2512.16019v1](http://arxiv.org/abs/2512.16019v1)

**作者:** Qiping Zhang, Nathan Tsoi, Mofeed Nagib, Hao-Tien Lewis Chiang, Marynel Vázquez

**发布时间:** 2025-12-17

### GPT解析

### 总结

研究利用大型语言模型的小样本学习能力，在社交导航任务中改进机器人预测用户对其性能感知的能力，并通过实验验证了其有效性。

### 背景

理解人类在人机交互过程中如何评价机器人行为对开发符合人类期望的社交感知机器人至关重要。传统方法是进行用户研究，而最近研究提出使用机器学习替代。然而，现有数据驱动方法需要大量标记数据，限制了实际应用。

### 目的

利用大型语言模型的小样本学习能力，改进机器人预测用户对其性能感知的能力，并在社交导航任务中实验验证这一想法。

### 方法

扩展SEAN TOGETHER数据集，增加真实人机导航情境和参与者反馈；评估多种LLMs基于时空线索预测人类对机器人性能感知的能力；进行消融研究确定LLMs依赖的传感器信息类型；探索个性化示例在上下文学习中的应用。

### 主要发现

LLMs能够匹配或超越传统监督学习模型的性能，同时需要的标记实例数量少一个数量级；随着上下文示例增加，预测性能提高，证实了方法的可扩展性；来自同一被评估用户的个性化示例可以进一步提高预测准确性。

### 结论

这项工作为通过用户中心反馈以可扩展方式改进机器人行为铺平了道路。

### 翻译

理解人类在人机交互过程中如何评价机器人行为对于开发符合人类期望的社交感知机器人至关重要。虽然捕捉这些评价的传统方法是进行用户研究，但最近的工作提出利用机器学习来替代。然而，现有的数据驱动方法需要大量标记数据，这限制了它们在实际中的应用。为了解决这一差距，我们提出利用大型语言模型的小样本学习能力来提高机器人预测用户对其性能感知的能力，并在社交导航任务中实验性地研究了这一想法。为此，我们扩展了SEAN TOGETHER数据集，增加了更多真实世界的人机导航情境和参与者反馈。使用这个增强的数据集，我们评估了多种LLMs基于观察到的机器人和周围人类运动的时空线索，从少量上下文示例中预测人类对机器人性能感知的能力。我们的结果表明，LLMs能够匹配或超越传统监督学习模型的性能，同时需要的标记实例数量少一个数量级。我们进一步证明，预测性能随着更多上下文示例的增加而提高，证实了我们方法的可扩展性。此外，我们通过进行消融研究，研究了LLMs依赖哪种基于传感器的信息来进行这些推断，消融研究针对用于性能预测的输入特征。最后，我们探索了个性化示例在上下文学习中的新颖应用，即从被评估的同一用户中抽取示例，发现它们可以进一步提高预测准确性。这项工作为通过用户中心反馈以可扩展方式改进机器人行为铺平了道路。


### 论文摘要

Understanding how humans evaluate robot behavior during human-robot interactions is crucial for developing socially aware robots that behave according to human expectations. While the traditional approach to capturing these evaluations is to conduct a user study, recent work has proposed utilizing machine learning instead. However, existing data-driven methods require large amounts of labeled data, which limits their use in practice. To address this gap, we propose leveraging the few-shot learning capabilities of Large Language Models (LLMs) to improve how well a robot can predict a user's perception of its performance, and study this idea experimentally in social navigation tasks. To this end, we extend the SEAN TOGETHER dataset with additional real-world human-robot navigation episodes and participant feedback. Using this augmented dataset, we evaluate the ability of several LLMs to predict human perceptions of robot performance from a small number of in-context examples, based on observed spatio-temporal cues of the robot and surrounding human motion. Our results demonstrate that LLMs can match or exceed the performance of traditional supervised learning models while requiring an order of magnitude fewer labeled instances. We further show that prediction performance can improve with more in-context examples, confirming the scalability of our approach. Additionally, we investigate what kind of sensor-based information an LLM relies on to make these inferences by conducting an ablation study on the input features considered for performance prediction. Finally, we explore the novel application of personalized examples for in-context learning, i.e., drawn from the same user being evaluated, finding that they further enhance prediction accuracy. This work paves the path to improving robot behavior in a scalable manner through user-centered feedback.

---

## 30. 论文ID: 2512.15748v1

**错误:** 无法读取论文信息 - [Errno 2] No such file or directory: 'storage/result/2512.15748v1.json'

---

## 31. DenseBEV: Transforming BEV Grid Cells into 3D Objects

**论文链接:** [http://arxiv.org/abs/2512.16818v1](http://arxiv.org/abs/2512.16818v1)

**作者:** Marius Dähling, Sebastian Krebs, J. Marius Zöllner

**发布时间:** 2025-12-18

**备注:** 15 pages, 8 figures, accepted by WACV 2026

### GPT解析

### 总结

本文提出了一种名为DenseBEV的新方法，通过直接使用鸟瞰图(BEV)特征单元作为锚点，改进了多摄像头3D目标检测的性能。该方法采用两阶段锚点生成和基于BEV的非极大值抑制技术，有效解决了注意力计算扩展问题，并利用混合时间建模进一步提升了检测效果。

### 背景

当前研究越来越多地使用基于鸟瞰图(BEV)的transformer进行多摄像头3D目标检测。传统模型通常使用随机查询作为锚点并进行连续优化，而最近的进展则使用辅助网络的检测结果来补充或替代这些随机查询。

### 目的

提出一种更直观、高效的方法，直接使用BEV特征单元作为锚点；开发专门用于多摄像头3D目标检测的两阶段锚点生成方法；解决大量查询导致的注意力计算扩展问题；利用BEV特征中固有的时间信息增强检测性能。

### 方法

使用BEV特征单元直接作为锚点；提出新的两阶段锚点生成方法；应用基于BEV的非极大值抑制，允许梯度仅通过未被抑制的对象流动；使用编码器的BEV特征直接作为对象查询，嵌入时间信息；引入混合时间建模方法，整合先前的检测结果以进一步提高检测性能。

### 主要发现

在nuScenes数据集上与基线相比，NDS和mAP有持续且显著的改进；即使使用更稀疏的BEV网格，方法仍然有效；对小物体特别有效，在nuScenes上行人检测mAP提高了3.8%，在Waymo上LET-mAP提高了8%；应用于Waymo Open数据集实现了最先进的性能，LET-mAP达到60.7%，比之前的最佳结果提高了5.4%。

### 结论

直接使用BEV特征单元作为锚点是一种更直观且高效的3D目标检测方法；所提出的方法有效解决了计算扩展问题；混合时间建模进一步增强了检测性能；DenseBEV在多个数据集上实现了最先进的性能。

### 翻译

在当前研究中，基于鸟瞰图(BEV)的transformer越来越多地用于多摄像头3D目标检测。传统模型通常采用随机查询作为锚点，并对其进行连续优化。最近的进展通过辅助网络的检测结果来补充或替代这些随机查询。我们提出了一种更直观、高效的方法，直接使用BEV特征单元作为锚点。这种端到端方法利用BEV查询的密集网格，将每个单元视为最终检测任务中潜在的对象。因此，我们引入了一种专门为多摄像头3D目标设计的新型两阶段锚点生成方法。为了解决大量查询导致的注意力计算扩展问题，我们应用了基于BEV的非极大值抑制，允许梯度仅通过未被抑制的对象流动。这确保了高效训练，无需后处理。通过使用编码器(如BEVFormer)的BEV特征直接作为对象查询，时间BEV信息被自然嵌入。基于我们对象查询中已经嵌入的时间BEV信息，我们通过整合先前的检测结果引入了混合时间建模方法，以进一步提高检测性能。在nuScenes数据集上评估我们的方法，显示与基线相比，NDS和mAP有持续且显著的改进，即使使用更稀疏的BEV网格(因此初始锚点更少)。该方法对小物体特别有效，在nuScenes上行人检测mAP提高了3.8%，在Waymo上LET-mAP提高了8%。将我们的方法(命名为DenseBEV)应用于具有挑战性的Waymo Open数据集，实现了最先进的性能，LET-mAP达到60.7%，比之前的最佳结果提高了5.4%。代码可在https://github.com/mdaehl/DenseBEV获取。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决基于鸟瞰图(BEV)的transformer模型在多摄像头3D物体检测中如何更有效地初始化物体查询的问题。这个问题很重要，因为3D物体检测对自动驾驶和机器人应用至关重要，需要准确检测各种大小的物体(特别是小型物体如行人、自行车等)，同时保持计算效率以实现实际部署。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者受到Deformable DETR的两阶段物体查询初始化方法的启发，考虑直接使用BEV特征网格中的每个单元作为潜在物体的锚点。为了解决密集锚点带来的计算问题和重复检测问题，作者引入了非极大值抑制(NMS)到训练过程中。作者还借鉴了DDQ在2D检测中使用NMS过滤特征的概念，以及StreamPETR的时间建模方法，设计了混合时间建模方案。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是将BEV特征网格中的每个网格单元直接作为潜在物体的锚点，而不是使用随机查询或辅助网络生成的查询。整体流程是：1)使用BEVFormer编码器处理多摄像头图像生成BEV特征网格；2)将每个网格单元通过辅助检测头解码为边界框和置信度分数；3)应用BEV-NMS过滤冗余查询；4)将过滤后的查询输入解码器进一步优化；5)对于时间增强版本，将当前BEV查询与先前检测到的物体结合实现混合时间建模。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)提出DenseBEV，第一个通过NMS实现密集先验的端到端多视图3D物体检测方法；2)实现物体中心时间建模和BEV时间建模的合并；3)在Waymo数据集上实现最先进性能。相比之前的工作，不同之处在于：不再依赖辅助网络生成初始锚点；将NMS集成到训练过程；使用BEV平面而非3D空间进行NMS；实现混合时间建模；特别擅长检测小型物体。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'DenseBEV通过直接利用BEV特征网格作为密集锚点并集成非极大值抑制到训练过程中，显著提高了多摄像头3D物体检测的效率和准确性，特别是在检测小型物体方面，同时实现了最先进的性能。'}


### 论文摘要

In current research, Bird's-Eye-View (BEV)-based transformers are increasingly utilized for multi-camera 3D object detection. Traditional models often employ random queries as anchors, optimizing them successively. Recent advancements complement or replace these random queries with detections from auxiliary networks. We propose a more intuitive and efficient approach by using BEV feature cells directly as anchors. This end-to-end approach leverages the dense grid of BEV queries, considering each cell as a potential object for the final detection task. As a result, we introduce a novel two-stage anchor generation method specifically designed for multi-camera 3D object detection. To address the scaling issues of attention with a large number of queries, we apply BEV-based Non-Maximum Suppression, allowing gradients to flow only through non-suppressed objects. This ensures efficient training without the need for post-processing. By using BEV features from encoders such as BEVFormer directly as object queries, temporal BEV information is inherently embedded. Building on the temporal BEV information already embedded in our object queries, we introduce a hybrid temporal modeling approach by integrating prior detections to further enhance detection performance. Evaluating our method on the nuScenes dataset shows consistent and significant improvements in NDS and mAP over the baseline, even with sparser BEV grids and therefore fewer initial anchors. It is particularly effective for small objects, enhancing pedestrian detection with a 3.8% mAP increase on nuScenes and an 8% increase in LET-mAP on Waymo. Applying our method, named DenseBEV, to the challenging Waymo Open dataset yields state-of-the-art performance, achieving a LET-mAP of 60.7%, surpassing the previous best by 5.4%. Code is available at https://github.com/mdaehl/DenseBEV.

---

## 32. Auto-Vocabulary 3D Object Detection

**论文链接:** [http://arxiv.org/abs/2512.16077v1](http://arxiv.org/abs/2512.16077v1)

**作者:** Haomeng Zhang, Kuan-Chuan Peng, Suhas Lohit, Raymond A. Yeh

**发布时间:** 2025-12-18

**备注:** technical report

### GPT解析

### 总结

本文提出了一种自动词汇3D目标检测方法(AV3DOD)，能够自动生成检测对象的类别名称而无需用户输入，在多个数据集上实现了最先进的性能。

### 背景

现有的开放词汇3D目标检测方法虽然名称上暗示可以检测训练过程中未见过的类别，但实际上仍需用户在训练和推理时指定类别。

### 目的

研究自动词汇3D目标检测(AV3DOD)，其中检测到的对象的类别名称可以自动生成，无需任何用户输入。

### 方法

引入语义分数(SS)评估生成类别名称的质量；开发AV3DOD框架，利用2D视觉-语言模型通过图像标注、伪3D框生成和特征空间语义扩展来生成丰富的语义候选。

### 主要发现

在ScanNetV2和SUNRGB-D数据集上，AV3DOD在定位(mAP)和语义质量(SS)方面都达到了最先进的性能；在ScanNetV2上超越了CoDA方法，整体mAP提高3.48，SS相对提升24.5%。

### 结论

AV3DOD方法实现了自动词汇3D目标检测，无需用户指定类别，同时保持了高水平的检测性能和语义质量。

### 翻译

开放词汇3D目标检测方法能够定位训练过程中未见过的类别的3D边界框。尽管名称如此，现有方法在训练和推理时都依赖于用户指定的类别。我们研究了自动词汇3D目标检测(AV3DOD)，其中检测到的对象的类别名称无需任何用户输入即可自动生成。为此，我们引入语义分数(SS)来评估生成的类别名称的质量。然后我们开发了一个新框架AV3DOD，它利用2D视觉-语言模型(VLMs)通过图像标注、伪3D框生成和特征空间语义扩展来生成丰富的语义候选。在ScanNetV2和SUNRGB-D数据集上，AV3DOD在定位(mAP)和语义质量(SS)方面都达到了最先进的性能。值得注意的是，它超越了SOTA方法CoDA，整体mAP提高了3.48，在ScanNetV2上SS相对提升了24.5%。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决自动词汇3D物体检测问题，即让3D物体检测模型能够自主发现和生成物体类别名称，而不依赖于用户预定义的词汇。这个问题在现实中很重要，因为真实世界中的物体种类繁多且不断变化，无法预先定义所有可能的类别；在研究中很重要，因为现有方法限制了3D感知系统的开放世界特性，无法完全捕捉3D场景中的语义多样性，这对于实现真正的开放世界3D理解、应用于自动驾驶、机器人和embodied AI等领域至关重要。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先明确指出现有开放词汇3D物体检测方法的局限性，然后设计了新任务和评估指标。方法设计上借鉴了2D视觉-语言模型(VLMs)的能力，特别是图像描述功能；引入了伪3D框生成技术，从2D图像生成3D物体提案；设计了特征空间语义扩展策略(FSSE)，在连续嵌入空间中采样新的语义原型。确实借鉴了现有工作，包括3DETR用于3D物体检测、Florence2用于图像描述、CLIP用于跨模态特征对齐等，但在伪3D框生成和特征空间语义扩展方面进行了创新和扩展。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是让3D物体检测模型能够自主发现和生成物体类别，不依赖用户预定义词汇，通过利用2D视觉-语言模型能力和特征空间语义扩展实现。整体流程分为：1)物体定位模块：使用3DETR生成类无关的3D物体提案和特征；2)新颖语义探索模块：整合基础类别特征、VLM描述特征、伪标签特征和扩展特征；3)语义对齐模块：将检测到的3D物体特征与超词汇特征匹配预测类别；4)训练过程：结合定位损失和语义损失联合优化；5)推理过程：仅使用定位和语义对齐模块，排除扩展特征提高标签可读性。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)提出自动词汇3D物体检测新任务；2)设计语义分数(SS)评估指标；3)开发伪3D框生成技术；4)提出特征空间语义扩展(FSSE)策略；5)整合多源语义构建丰富语义空间。相比之前工作不同：传统OV3DOD需要预定义词汇，AV3DOD完全自主生成；现有自动词汇方法依赖离散文本标签，AV3DOD支持连续特征空间扩展；许多方法需要2D图像输入，AV3DOD推理时直接在3D点云上进行；性能上显著提升，如ScanNetV2上mAP高出3.48，SS提升24.5%。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '这篇论文提出了自动词汇3D物体检测新任务和框架，通过整合2D视觉-语言模型和特征空间语义扩展，使3D物体检测能够自主发现和生成物体类别，无需预定义词汇，显著提升了定位精度和语义理解能力。'}


### 论文摘要

Open-vocabulary 3D object detection methods are able to localize 3D boxes of classes unseen during training. Despite the name, existing methods rely on user-specified classes both at training and inference. We propose to study Auto-Vocabulary 3D Object Detection (AV3DOD), where the classes are automatically generated for the detected objects without any user input. To this end, we introduce Semantic Score (SS) to evaluate the quality of the generated class names. We then develop a novel framework, AV3DOD, which leverages 2D vision-language models (VLMs) to generate rich semantic candidates through image captioning, pseudo 3D box generation, and feature-space semantics expansion. AV3DOD achieves the state-of-the-art (SOTA) performance on both localization (mAP) and semantic quality (SS) on the ScanNetV2 and SUNRGB-D datasets. Notably, it surpasses the SOTA, CoDA, by 3.48 overall mAP and attains a 24.5% relative improvement in SS on ScanNetV2.

---

## 33. AG-MPBS: a Mobility-Aware Prediction and Behavior-Based Scheduling Framework for Air-Ground Unmanned Systems

**论文链接:** [http://arxiv.org/abs/2512.16454v1](http://arxiv.org/abs/2512.16454v1)

**作者:** Tianhao Shao, Kaixing Zhao, Feng Liu, Lixin Yang, Bin Guo

**发布时间:** 2025-12-18

### GPT解析

### 总结

本文提出了一种名为MPBS的可扩展任务招募框架，通过行为感知和移动性预测来优化无人系统的任务分配。

### 背景

随着无人机和无人地面车辆等无人系统在城市场景感知和应急响应等应用中的重要性日益增加，高效招募这些自主设备来执行时间敏感任务已成为关键挑战。

### 目的

提出一个可扩展的任务招募框架MPBS(Mobility-aware Prediction and Behavior-based Scheduling)，将每个设备视为可招募的'用户'，以实现高效的任务分配。

### 方法

MPBS集成了三个关键模块：行为感知的KNN分类器、用于预测设备移动性的时变马尔可夫预测模型，以及考虑任务紧急程度和基站性能的动态优先级调度机制。通过结合行为分类与时空预测，MPBS能够实时地将任务分配给最合适的设备。

### 主要发现

在真实世界的GeoLife数据集上的实验评估表明，MPBS显著提高了任务完成效率和资源利用率。

### 结论

所提出的框架为无人系统中的智能协作调度提供了预测性、行为感知的解决方案。

### 翻译

随着无人机和无人地面车辆等无人系统在城市场景感知和应急响应等应用中的重要性日益增加，高效招募这些自主设备来执行时间敏感任务已成为一个关键挑战。本文提出了MPBS（移动感知预测和行为调度），一种可扩展的任务招募框架，将每个设备视为可招募的'用户'。MPBS集成了三个关键模块：行为感知的KNN分类器、用于预测设备移动性的时变马尔可夫预测模型，以及考虑任务紧急程度和基站性能的动态优先级调度机制。通过结合行为分类与时空预测，MPBS能够实时地将任务分配给最合适的设备。在真实世界的GeoLife数据集上的实验评估显示，MPBS显著提高了任务完成效率和资源利用率。所提出的框架为无人系统中的智能协作调度提供了预测性、行为感知的解决方案。


### 论文摘要

As unmanned systems such as Unmanned Aerial Vehicles (UAVs) and Unmanned Ground Vehicles (UGVs) become increasingly important to applications like urban sensing and emergency response, efficiently recruiting these autonomous devices to perform time-sensitive tasks has become a critical challenge. This paper presents MPBS (Mobility-aware Prediction and Behavior-based Scheduling), a scalable task recruitment framework that treats each device as a recruitable "user". MPBS integrates three key modules: a behavior-aware KNN classifier, a time-varying Markov prediction model for forecasting device mobility, and a dynamic priority scheduling mechanism that considers task urgency and base station performance. By combining behavioral classification with spatiotemporal prediction, MPBS adaptively assigns tasks to the most suitable devices in real time. Experimental evaluations on the real-world GeoLife dataset show that MPBS significantly improves task completion efficiency and resource utilization. The proposed framework offers a predictive, behavior-aware solution for intelligent and collaborative scheduling in unmanned systems.

---

## 34. KineST: A Kinematics-guided Spatiotemporal State Space Model for Human Motion Tracking from Sparse Signals

**论文链接:** [http://arxiv.org/abs/2512.16791v1](http://arxiv.org/abs/2512.16791v1)

**作者:** Shuting Zhao, Zeyu Xiao, Xinrong Chen

**发布时间:** 2025-12-18

**备注:** Accepted by AAAI 2026

### GPT解析

### 总结

KineST是一种新颖的运动学引导的状态空间模型，用于解决AR/VR应用中基于稀疏信号重建全身姿势的问题，在准确性和时间一致性方面表现优异。

### 背景

全身运动追踪在AR/VR应用中连接物理和虚拟交互至关重要，但基于头戴式显示器的稀疏信号重建真实多样的全身姿势具有挑战性。现有方法计算成本高或分别建模时空依赖性，难以平衡准确性、时间一致性和效率。

### 目的

开发一种能够有效提取时空依赖性、整合局部和全局姿势感知的全身姿势重建方法，同时平衡准确性、时间一致性和效率。

### 方法

提出KineST模型，包含两个核心创新：1)将状态空间对偶框架内的扫描策略重新公式化为运动学引导的双向扫描，嵌入运动学先验；2)采用混合时空表示学习方法，紧密耦合空间和时间上下文；3)引入几何角速度损失，对旋转变化施加物理约束提高运动稳定性。

### 主要发现

KineST在轻量级框架内实现了优越的性能，在准确性和时间一致性方面均表现出色。

### 结论

KineST是一种有效的全身姿势重建解决方案，能够在保持计算效率的同时提供高质量、时间一致的运动重建结果。

### 翻译

全身运动追踪在AR/VR应用中扮演着重要角色，连接物理和虚拟交互。然而，基于AR/VR场景中的主要设备头戴式显示器获得的稀疏信号来重建真实多样的全身姿势具有挑战性。现有的姿势重建方法通常计算成本高或依赖分别建模空间和时间依赖性，难以平衡准确性、时间一致性和效率。为解决这一问题，我们提出了KineST，一种新颖的运动学引导的状态空间模型，它能够有效提取时空依赖性，同时整合局部和全局姿势感知。创新来自两个核心想法。首先，为了更好地捕捉复杂的关节关系，将状态空间对偶框架内的扫描策略重新公式化为运动学引导的双向扫描，嵌入运动学先验。其次，采用混合时空表示学习方法，紧密耦合空间和时间上下文，平衡准确性和平滑度。此外，引入几何角速度损失对旋转变化施加物理上有意义的约束，进一步提高运动稳定性。大量实验证明，KineST在轻量级框架内具有优越的性能，在准确性和时间一致性方面均表现出色。项目页面：https://kaka-1314.github.io/KineST/

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决从稀疏信号（如头戴显示器HMD捕获的头部和手部信号）重建真实、多样且平滑的全身运动姿势的问题。这个问题在AR/VR应用中至关重要，因为它连接了物理和虚拟交互，影响用户体验的沉浸感。现有的方法往往计算成本高、参数量大，或难以平衡姿势准确性和运动平滑性，限制了实际应用。在真实场景中，HMD只能提供非常有限的信息，使得从这些稀疏信号推断全身运动变得极具挑战性。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先分析了现有方法的局限性：高计算成本、参数量大、难以平衡准确性和平滑性。他们选择了状态空间对偶（SSD）框架作为基础，因为该框架在高效时间序列建模方面显示出潜力，但发现直接应用效果不佳。作者创新性地设计了运动学引导的双向扫描策略和混合时空表示学习机制，借鉴了状态空间模型（如Mamba和SSD）、SMPL人体表示方法以及常用的损失函数设计，但将这些元素进行了创新性整合和改进，以适应全身运动重建的特殊需求。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是通过运动学引导的时空建模，从稀疏信号中重建准确且平滑的全身运动。具体包括：1）将人体骨骼的运动学先验整合到模型设计中；2）通过混合时空表示学习方法紧密耦合空间和时间上下文；3）采用基于运动学树的双向扫描策略捕捉更完整的人体运动特征；4）使用物理约束的损失函数提高运动稳定性。整体流程：首先接收HMD的稀疏跟踪信号并嵌入为姿势特征；然后通过时间流模块（TFM）处理时间依赖关系；接着通过时空运动学流模块（SKFM）整合空间和时间信息；最后使用线性回归器估计全身运动姿势，并用组合损失函数进行训练。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1）运动学引导的双向扫描策略（KTSS），基于人体运动学树重新设计扫描顺序；2）时空混合机制（STMM），紧密耦合空间和时间上下文；3）几何角速度损失（L_geo_angvel），在流形空间中对旋转变化施加物理约束；4）轻量级高效架构，参数量仅11M。相比之前的工作，KineST不再分别建模空间和时间依赖关系，而是通过STMM实现联合建模；显式整合运动学先验，而大多数现有方法忽略人体骨骼结构；使用符合旋转几何特性的损失函数，而非简单的欧几里得距离；在轻量级框架内同时实现了高准确性和良好的时间一致性。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'KineST通过运动学引导的时空状态空间模型和创新的几何角速度损失，实现了从稀疏信号中高效重建准确且平滑的全身运动，为AR/VR应用提供了轻量级但高性能的解决方案。'}


### 论文摘要

Full-body motion tracking plays an essential role in AR/VR applications, bridging physical and virtual interactions. However, it is challenging to reconstruct realistic and diverse full-body poses based on sparse signals obtained by head-mounted displays, which are the main devices in AR/VR scenarios. Existing methods for pose reconstruction often incur high computational costs or rely on separately modeling spatial and temporal dependencies, making it difficult to balance accuracy, temporal coherence, and efficiency. To address this problem, we propose KineST, a novel kinematics-guided state space model, which effectively extracts spatiotemporal dependencies while integrating local and global pose perception. The innovation comes from two core ideas. Firstly, in order to better capture intricate joint relationships, the scanning strategy within the State Space Duality framework is reformulated into kinematics-guided bidirectional scanning, which embeds kinematic priors. Secondly, a mixed spatiotemporal representation learning approach is employed to tightly couple spatial and temporal contexts, balancing accuracy and smoothness. Additionally, a geometric angular velocity loss is introduced to impose physically meaningful constraints on rotational variations for further improving motion stability. Extensive experiments demonstrate that KineST has superior performance in both accuracy and temporal consistency within a lightweight framework. Project page: https://kaka-1314.github.io/KineST/

---

## 35. Depth Any Panoramas: A Foundation Model for Panoramic Depth Estimation

**论文链接:** [http://arxiv.org/abs/2512.16913v1](http://arxiv.org/abs/2512.16913v1)

**作者:** Xin Lin, Meixi Song, Dizhe Zhang, Wenxuan Lu, Haodong Li, Bo Du, Ming-Hsuan Yang, Truong Nguyen, Lu Qi

**发布时间:** 2025-12-18

**备注:** Project Page: https://insta360-research-team.github.io/DAP_website/

### GPT解析

### 总结

这项研究提出了一种全景度量深度基础模型，能够泛化到不同场景距离，通过数据闭环范式结合多种数据源构建大规模数据集，并采用三阶段伪标签整理管道减少域差距，同时引入优化的模型架构提高鲁棒性和几何一致性。

### 背景

深度估计在计算机视觉中具有重要意义，但现有方法在不同场景距离和类型上泛化能力有限。缺乏能够处理全景图像且在各种真实场景中保持稳定性能的基础模型。

### 目的

开发一种能够泛化到不同场景距离的全景度量深度基础模型，提高在室内外、合成与真实数据上的性能，并确保跨视图的几何一致性。

### 方法

1. 数据收集：结合公共数据集、UE5模拟器和文本到图像模型生成的合成数据，以及网络上的真实全景图像；2. 数据处理：引入三阶段伪标签整理管道，减少室内/室外和合成/真实数据间的域差距；3. 模型设计：采用DINOv3-Large作为骨干网络，引入即插即用的范围掩码头、以清晰度为中心的优化和以几何为中心的优化

### 主要发现

模型在多个基准测试（Stanford2D3D、Matterport3D和Deep360）上表现出强大的性能和零样本泛化能力，特别是在各种真实场景中能够提供稳健和稳定的度量深度预测。

### 结论

该全景度量深度基础模型通过创新的数据闭环范式和优化的模型架构，成功解决了在不同场景距离和类型上的泛化问题，为计算机视觉中的深度估计提供了新的解决方案。

### 翻译

在这项工作中，我们提出了一种全景度量深度基础模型，能够泛化到不同的场景距离。我们从数据构建和框架设计的角度探索了数据闭环范式。我们通过结合公共数据集、来自我们UE5模拟器和文本到图像模型的高质量合成数据，以及来自网络的真实全景图像，收集了一个大规模数据集。为了减少室内/室外和合成/真实数据之间的域差距，我们引入了一个三阶段伪标签整理管道，为未标记图像生成可靠的真值。对于模型，我们采用DINOv3-Large作为骨干网络，因为它具有强大的预训练泛化能力，并引入了即插即用的范围掩码头、以清晰度为中心的优化和以几何为中心的优化，以提高对不同距离的鲁棒性并强制执行跨视图的几何一致性。在多个基准测试（如Stanford2D3D、Matterport3D和Deep360）上的实验展示了强大的性能和零样本泛化能力，特别是在各种真实场景中具有稳健和稳定的度量预测。项目页面可以在以下网址找到：https://insta360-research-team.github.io/DAP_website/

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决全景深度估计的泛化能力不足问题，特别是现有方法难以适应多样化的真实世界场景（尤其是室外场景）。这个问题在现实中非常重要，因为全景深度估计能提供360°×180°的完整环境覆盖，对机器人导航中的全向避障、增强现实和空间智能等应用至关重要。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': "作者采用'数据在回路'范式，从数据构建和框架设计两方面思考。数据方面整合了公共数据集、UE5模拟器生成的合成数据及网络收集的真实图像；模型方面借鉴了DINOv3-Large作为骨干网络，并引入即插即用的范围掩码头和多种优化损失函数。作者借鉴了现有数据集和模拟平台，以及传统深度估计方法中的损失函数设计，但进行了创新性组合和改进。", '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是构建一个能统一跨不同域和场景类型的大规模全景深度基础模型。整体流程包括：1)构建包含2M+样本的多源数据集；2)实施三阶段训练管道（场景不变标签器→真实感不变标签器→DAP模型训练）；3)设计即插即用范围掩码头和多损失优化框架。这种方法通过大规模数据扩展和渐进式训练策略，实现了对室内外场景的强泛化能力。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)大规模数据引擎(2M+跨域样本)；2)三阶段伪标签精炼管道；3)即插即用范围掩码与几何-锐度双导向优化；4)强大的跨场景泛化能力。相比之前工作，DAP在数据规模上大幅提升(如PanDA仅12万样本，DAC仅80万样本)，训练方法从单一阶段改进为渐进式三阶段，并首次引入可适应不同距离的场景范围掩码机制，在室外场景表现尤为突出。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '本文通过构建大规模全景数据集和设计三阶段训练管道，提出了DAP全景深度基础模型，实现了跨室内-室外场景的强泛化能力和最先进的深度估计性能。'}


### 论文摘要

In this work, we present a panoramic metric depth foundation model that generalizes across diverse scene distances. We explore a data-in-the-loop paradigm from the view of both data construction and framework design. We collect a large-scale dataset by combining public datasets, high-quality synthetic data from our UE5 simulator and text-to-image models, and real panoramic images from the web. To reduce domain gaps between indoor/outdoor and synthetic/real data, we introduce a three-stage pseudo-label curation pipeline to generate reliable ground truth for unlabeled images. For the model, we adopt DINOv3-Large as the backbone for its strong pre-trained generalization, and introduce a plug-and-play range mask head, sharpness-centric optimization, and geometry-centric optimization to improve robustness to varying distances and enforce geometric consistency across views. Experiments on multiple benchmarks (e.g., Stanford2D3D, Matterport3D, and Deep360) demonstrate strong performance and zero-shot generalization, with particularly robust and stable metric predictions in diverse real-world scenes. The project page can be found at: \href{https://insta360-research-team.github.io/DAP_website/} {https://insta360-research-team.github.io/DAP\_website/}

---

## 36. PolaRiS: Scalable Real-to-Sim Evaluations for Generalist Robot Policies

**论文链接:** [http://arxiv.org/abs/2512.16881v1](http://arxiv.org/abs/2512.16881v1)

**作者:** Arhan Jain, Mingtong Zhang, Kanav Arora, William Chen, Marcel Torne, Muhammad Zubair Irshad, Sergey Zakharov, Yue Wang, Sergey Levine, Chelsea Finn, Wei-Chiu Ma, Dhruv Shah, Abhishek Gupta, Karl Pertsch

**发布时间:** 2025-12-18

**备注:** Website: https://polaris-evals.github.io/

### GPT解析

### 总结

这篇论文介绍了一个名为PolaRiS的可扩展真实到模拟框架，用于高保真模拟机器人评估。该框架利用神经重建方法将真实世界场景的短视频扫描转换为交互式模拟环境，并开发了模拟数据共训练方法，以弥合真实到模拟的差距，实现零样本评估。研究表明PolaRiS评估与真实世界通用策略性能的相关性比现有模拟基准更强。

### 背景

机器人学习研究面临准确测量和比较机器人策略性能的挑战。由于随机性、可重复性差和真实世界部署耗时长的特点，机器人领域的基准测试历来困难。对于需要在各种场景和任务中评估的通用策略，这一挑战更加突出。现有模拟评估与真实世界之间存在视觉和物理域差距，导致评估结果不可靠。此外，构建真实多样的模拟环境需要大量专业知识和人力投入。

### 目的

为了弥合模拟与真实世界之间的差距，研究者引入了PolaRiS框架，旨在提供一种可扩展的高保真模拟机器人评估方法，能够准确反映真实世界中的机器人策略性能。

### 方法

PolaRiS利用神经重建方法将真实世界场景的短视频扫描转换为交互式模拟环境。同时，开发了一种简单的模拟数据共训练方法，用于弥合剩余的真实到模拟差距，并实现在未见过的模拟环境中的零样本评估。通过在模拟和真实世界之间进行大量成对评估来验证方法的有效性。

### 主要发现

研究表明，PolaRiS评估与真实世界通用策略性能的相关性比现有模拟基准强得多。此外，该方法的简单性使得能够快速创建多样化的模拟环境，降低了高质量模拟环境创建的门槛。

### 结论

这项工作朝着为下一代机器人基础模型实现分布式和民主化评估迈出了一步。PolaRiS框架不仅提供了一种更可靠的机器人策略评估方法，还加速了高质量模拟环境的创建，有望推动机器人学习领域的发展。

### 翻译

机器人学习研究的一个重大挑战是我们准确测量和比较机器人策略性能的能力。由于随机性、可重复性和真实世界部署的耗时性，机器人领域的基准测试历来具有挑战性。对于最近的通用策略，这种挑战更加突出，因为它们需要在各种场景和任务中进行评估。模拟评估为真实世界评估提供了可扩展的补充，但现有模拟基准与真实世界之间的视觉和物理域差距使得它们成为策略改进的不可靠信号。此外，构建真实且多样化的模拟环境传统上需要大量的人力投入和专业知识。为了弥合这一差距，我们介绍了PolaRiS（模拟中的策略评估和环境重建），这是一个用于高保真模拟机器人评估的可扩展真实到模拟框架。PolaRiS利用神经重建方法将真实世界场景的短视频扫描转换为交互式模拟环境。此外，我们开发了一种简单的模拟数据共训练方法，用于弥合剩余的真实到模拟差距，并实现在未见过的模拟环境中的零样本评估。通过在模拟和真实世界之间进行大量成对评估，我们证明PolaRiS评估与真实世界通用策略性能的相关性比现有模拟基准强得多。其简单性也使得能够快速创建多样化的模拟环境。因此，这项工作朝着为下一代机器人基础模型实现分布式和民主化评估迈出了一步。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决如何准确评估通用机器人策略性能的问题。这个问题很重要，因为随着通用机器人策略的发展，需要大规模经验验证，而直接在真实机器人上评估既耗时又昂贵；现有模拟环境与真实世界之间存在视觉和物理差距，无法可靠预测真实世界中的策略性能；此外，构建逼真模拟环境需要大量专业知识，限制了评估的可扩展性。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先识别出现有模拟评估方法与真实世界之间存在差距的问题，特别是对于通用机器人策略。他们借鉴了神经重建技术（如2D高斯飞溅）和3D生成模型（如TRELLIS）来创建高保真环境，同时结合行为克隆等现有技术进行策略微调。创新性地将这些技术整合到一个专门针对通用机器人策略评估的框架中，并解决了支持手腕相机和零样本评估等特定挑战。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是通过神经重建将真实世界场景转换为高保真模拟环境，再通过简单模拟数据共训练弥合真实到模拟差距，实现策略准确评估。流程包括：1)拍摄真实场景视频并用2D高斯飞溅重建环境和机器人；2)自动处理机器人关节使高斯飞溅组件随物理运动学移动；3)用TRELLIS生成模型创建3D物体；4)组合场景并设置物理参数；5)用少量模拟演示共训练策略，使其在未见环境中也能准确评估。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)使用神经重建创建高保真3D模拟环境，支持完整3D渲染而非2D背景复制；2)通过少量共训练实现零样本评估能力；3)支持手腕相机评估（之前方法如SIMPLER不支持）；4)快速环境创建（<20分钟人力）；5)简单有效的共训练方法。相比之前工作，PolaRiS解决了移动相机支持问题，提供了比Libero等基准更高的保真度，比视频模型更可靠的物理交互，并首次实现了在未见环境中评估通用策略的能力。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'PolaRiS提出了一种可扩展的真实到模拟框架，通过神经重建技术和简单的模拟数据共训练方法，实现了对通用机器人策略的高保真评估，在未见过的模拟环境中与真实世界性能表现出强相关性。'}


### 论文摘要

A significant challenge for robot learning research is our ability to accurately measure and compare the performance of robot policies. Benchmarking in robotics is historically challenging due to the stochasticity, reproducibility, and time-consuming nature of real-world rollouts. This challenge is exacerbated for recent generalist policies, which has to be evaluated across a wide variety of scenes and tasks. Evaluation in simulation offers a scalable complement to real world evaluations, but the visual and physical domain gap between existing simulation benchmarks and the real world has made them an unreliable signal for policy improvement. Furthermore, building realistic and diverse simulated environments has traditionally required significant human effort and expertise. To bridge the gap, we introduce Policy Evaluation and Environment Reconstruction in Simulation (PolaRiS), a scalable real-to-sim framework for high-fidelity simulated robot evaluation. PolaRiS utilizes neural reconstruction methods to turn short video scans of real-world scenes into interactive simulation environments. Additionally, we develop a simple simulation data co-training recipe that bridges remaining real-to-sim gaps and enables zero-shot evaluation in unseen simulation environments. Through extensive paired evaluations between simulation and the real world, we demonstrate that PolaRiS evaluations provide a much stronger correlation to real world generalist policy performance than existing simulated benchmarks. Its simplicity also enables rapid creation of diverse simulated environments. As such, this work takes a step towards distributed and democratized evaluation for the next generation of robotic foundation models.

---

## 37. Task-Oriented Data Synthesis and Control-Rectify Sampling for Remote Sensing Semantic Segmentation

**论文链接:** [http://arxiv.org/abs/2512.16740v1](http://arxiv.org/abs/2512.16740v1)

**作者:** Yunkai Yang, Yudong Zhang, Kunquan Zhang, Jinxiao Zhang, Xinying Chen, Haohuan Fu, Runmin Dong

**发布时间:** 2025-12-18

### GPT解析

### 总结

该研究提出了一种任务导向的数据合成框架(TODSynth)，用于解决遥感图像语义分割中合成数据的效用限制问题。

### 背景

随着可控生成技术的快速发展，训练数据合成成为扩大遥感领域标记数据集和缓解人工标注的有效途径。然而，语义掩模控制的复杂性和采样质量的不确定性限制了合成数据在下游语义分割任务中的应用。

### 目的

解决语义掩模控制复杂性和采样质量不确定性带来的挑战，提高合成数据在遥感语义分割任务中的有效性。

### 方法

提出TODSynth框架，包含具有统一三重注意力的多模态扩散变换器(MM-DiT)和任务引导的即插即用采样策略；基于DiT生成基础模型系统评估不同控制方案；提出控制-修正流匹配(CRFM)方法，在早期高塑性阶段动态调整采样方向。

### 主要发现

文本-图像-掩模联合注意力方案结合图像和掩模分支的完全微调显著增强了遥感语义分割数据合成的效果，特别是在少样本和复杂场景下；CRFM方法减轻了生成图像的不稳定性，缩小了合成数据与下游任务之间的差距。

### 结论

该方法一致优于最先进的可控生成方法，能产生更稳定和任务导向的合成数据用于遥感语义分割任务。

### 翻译

随着可控生成的快速进展，训练数据合成已成为扩大标记数据集和缓解遥感领域人工标注的一种有前景的方法。然而，语义掩模控制的复杂性和采样质量的不确定性常常限制了合成数据在下游语义分割任务中的效用。为应对这些挑战，我们提出了一个任务导向的数据合成框架(TODSynth)，包括具有统一三重注意力的多模态扩散变换器(MM-DiT)和由任务引导的即插即用采样策略。基于强大的基于DiT的生成基础模型，我们系统评估了不同的控制方案，表明文本-图像-掩模联合注意力方案结合图像和掩模分支的完全微调显著增强了遥感语义分割数据合成的有效性，特别是在少样本和复杂场景场景下。此外，我们提出了控制-修正流匹配(CRFM)方法，在早期高塑性阶段由语义损失引导动态调整采样方向，减轻了生成图像的不稳定性，并缩小了合成数据与下游分割任务之间的差距。大量实验证明，我们的方法一致优于最先进的可控生成方法，为遥感语义分割产生了更稳定和任务导向的合成数据。


### 论文摘要

With the rapid progress of controllable generation, training data synthesis has become a promising way to expand labeled datasets and alleviate manual annotation in remote sensing (RS). However, the complexity of semantic mask control and the uncertainty of sampling quality often limit the utility of synthetic data in downstream semantic segmentation tasks. To address these challenges, we propose a task-oriented data synthesis framework (TODSynth), including a Multimodal Diffusion Transformer (MM-DiT) with unified triple attention and a plug-and-play sampling strategy guided by task feedback. Built upon the powerful DiT-based generative foundation model, we systematically evaluate different control schemes, showing that a text-image-mask joint attention scheme combined with full fine-tuning of the image and mask branches significantly enhances the effectiveness of RS semantic segmentation data synthesis, particularly in few-shot and complex-scene scenarios. Furthermore, we propose a control-rectify flow matching (CRFM) method, which dynamically adjusts sampling directions guided by semantic loss during the early high-plasticity stage, mitigating the instability of generated images and bridging the gap between synthetic data and downstream segmentation tasks. Extensive experiments demonstrate that our approach consistently outperforms state-of-the-art controllable generation methods, producing more stable and task-oriented synthetic data for RS semantic segmentation.

---

## 38. Discovering and Learning Probabilistic Models of Black-Box AI Capabilities

**论文链接:** [http://arxiv.org/abs/2512.16733v1](http://arxiv.org/abs/2512.16733v1)

**作者:** Daniel Bramblett, Rushang Karia, Adrian Ciotinga, Ruthvick Suresh, Pulkit Verma, YooJung Choi, Siddharth Srivastava

**发布时间:** 2025-12-18

### GPT解析

### 总结

本文提出了一种使用PDDL风格表示来学习和建模黑盒AI系统规划能力的方法，通过蒙特卡洛树搜索系统创建测试任务、获取数据并修剪假设空间，能够提供对黑盒AI能力的健全且可解释的表示。

### 背景

黑盒AI系统如基础模型越来越多地用于顺序决策，为确保这些系统的安全操作和部署，需要开发高效方法来提供对黑盒AI能力的健全且可解释的表示。

### 目的

开发一种方法，能够有效地学习和建模输入黑盒AI的规划能力，并提供对黑盒AI能力的健全且可解释的表示。

### 方法

使用PDDL风格的表示来学习和建模黑盒AI的规划能力，采用蒙特卡洛树搜索范式系统性地创建测试任务、获取数据并修剪可能的符号模型的假设空间，学习的模型描述黑盒AI的能力、执行条件以及执行的可能结果和相关概率。

### 主要发现

理论结果表明学习到的模型具有健全性、完整性和收敛性，多个黑盒AI系统的实证结果展示了所提出方法的范围、效率和准确性。

### 结论

PDDL风格的表示可以有效地用于学习和建模黑盒AI的规划能力，所提出的方法能够提供对黑盒AI能力的健全且可解释的表示。

### 翻译

黑盒AI系统如基础模型正越来越多地用于顺序决策。为确保此类系统的安全操作和部署，必须开发能够提供黑盒AI能力的健全且可解释表示的高效方法。本文表明，PDDL风格的表示可用于有效地学习和建模输入黑盒AI的规划能力。它采用蒙特卡洛树搜索范式系统性地创建测试任务、获取数据并修剪可能符号模型的假设空间。学习到的模型描述了黑盒AI的能力、执行条件以及执行的可能结果及其相关概率。理论结果表明学习到的模型的健全性、完整性和收敛性。多个黑盒AI系统的实证结果展示了所提出方法的范围、效率和准确性。


### 论文摘要

Black-box AI (BBAI) systems such as foundational models are increasingly being used for sequential decision making. To ensure that such systems are safe to operate and deploy, it is imperative to develop efficient methods that can provide a sound and interpretable representation of the BBAI's capabilities. This paper shows that PDDL-style representations can be used to efficiently learn and model an input BBAI's planning capabilities. It uses the Monte-Carlo tree search paradigm to systematically create test tasks, acquire data, and prune the hypothesis space of possible symbolic models. Learned models describe a BBAI's capabilities, the conditions under which they can be executed, and the possible outcomes of executing them along with their associated probabilities. Theoretical results show soundness, completeness and convergence of the learned models. Empirical results with multiple BBAI systems illustrate the scope, efficiency, and accuracy of the presented methods.

---

## 39. REGLUE Your Latents with Global and Local Semantics for Entangled Diffusion

**论文链接:** [http://arxiv.org/abs/2512.16636v1](http://arxiv.org/abs/2512.16636v1)

**作者:** Giorgos Petsangourakis, Christos Sgouropoulos, Bill Psomas, Theodoros Giannakopoulos, Giorgos Sfikas, Ioannis Kakogeorgiou

**发布时间:** 2025-12-18

### GPT解析

### 总结

本文介绍了REGLUE（表示纠缠与全局-局部统一编码），一种统一的潜在扩散框架，通过联合建模VAE图像潜在表示、局部VFM语义和全局[CLS]令牌，有效改善了潜在扩散模型的训练效率和样本质量。

### 背景

潜在扩散模型(LDMs)在图像合成方面表现优异，但其重建风格去噪目标仅提供间接语义监督，高级语义出现缓慢，需要更长训练时间并限制样本质量。现有方法要么通过表示对齐从外部注入语义，要么在扩散过程中仅联合建模VFM特征的窄片，未能充分利用VFM中丰富、非线性、多层空间语义。

### 目的

开发一种统一的潜在扩散框架，充分利用视觉基础模型(VFMs)中的丰富、非线性、多层空间语义，以改善潜在扩散模型的训练效率和样本质量。

### 方法

REGLUE框架在单个SiT主干网络中联合建模三种表示：(i) VAE图像潜在表示，(ii) 紧凑的局部（块级）VFM语义，和(iii) 全局（图像级）[CLS]令牌。使用轻量级卷积语义压缩器将多层VFM特征非线性聚合为低维空间结构化表示，并在扩散过程中与VAE潜在表示纠缠。同时，通过外部对齐损失将内部表示向冻结的VFM目标正则化。

### 主要发现

实验表明：(a) 空间VFM语义对性能至关重要，(b) 非线性压缩是解锁VFM特征全部益处的关键，(c) 全局令牌和外部对齐作为互补的轻量级增强，在全局-局部-潜在联合建模框架中发挥作用。

### 结论

REGLUE在ImageNet 256x256上持续改进FID指标并加速收敛，超过了多个基线方法。该方法通过有效结合全局和局部语义表示，充分利用了VFM特征，并通过外部对齐进一步优化了表示，为潜在扩散模型提供了新的研究方向。

### 翻译

潜在扩散模型(LDMs)实现了最先进的图像合成，但它们的重建风格去噪目标仅提供间接的语义监督：高级语义出现缓慢，需要更长的训练时间并限制了样本质量。最近的工作通过表示对齐从视觉基础模型(VFMs)外部注入语义，或者在扩散过程中仅联合建模VFM特征的一个窄片，未能充分利用丰富、非线性、多层空间语义的可用性。我们引入了REGLUE（Representation Entanglement with Global-Local Unified Encoding），这是一种统一的潜在扩散框架，在单个SiT主干网络中联合建模了(i) VAE图像潜在表示，(ii) 紧凑的局部（块级）VFM语义，和(iii) 全局（图像级）[CLS]令牌。一个轻量级卷积语义压缩器将多层VFM特征非线性聚合为低维空间结构化表示，该表示在扩散过程中与VAE潜在表示纠缠。外部对齐损失进一步将内部表示向冻结的VFM目标正则化。在ImageNet 256x256上，REGLUE在FID方面持续改进并加速收敛，超过了SiT-B/2和SiT-XL/2基线，以及REPA、ReDi和REG。大量实验表明，(a)空间VFM语义至关重要，(b)非线性压缩是解锁其全部益处的关键，(c)全局令牌和外部对齐在我们的全局-局部-潜在联合建模框架中作为互补的轻量级增强。代码可在https://github.com/giorgospets/reglue获取。


### 论文摘要

Latent diffusion models (LDMs) achieve state-of-the-art image synthesis, yet their reconstruction-style denoising objective provides only indirect semantic supervision: high-level semantics emerge slowly, requiring longer training and limiting sample quality. Recent works inject semantics from Vision Foundation Models (VFMs) either externally via representation alignment or internally by jointly modeling only a narrow slice of VFM features inside the diffusion process, under-utilizing the rich, nonlinear, multi-layer spatial semantics available. We introduce REGLUE (Representation Entanglement with Global-Local Unified Encoding), a unified latent diffusion framework that jointly models (i) VAE image latents, (ii) compact local (patch-level) VFM semantics, and (iii) a global (image-level) [CLS] token within a single SiT backbone. A lightweight convolutional semantic compressor nonlinearly aggregates multi-layer VFM features into a low-dimensional, spatially structured representation, which is entangled with the VAE latents in the diffusion process. An external alignment loss further regularizes internal representations toward frozen VFM targets. On ImageNet 256x256, REGLUE consistently improves FID and accelerates convergence over SiT-B/2 and SiT-XL/2 baselines, as well as over REPA, ReDi, and REG. Extensive experiments show that (a) spatial VFM semantics are crucial, (b) non-linear compression is key to unlocking their full benefit, and (c) global tokens and external alignment act as complementary, lightweight enhancements within our global-local-latent joint modeling framework. The code is available at https://github.com/giorgospets/reglue .

---

## 40. Causal-Tune: Mining Causal Factors from Vision Foundation Models for Domain Generalized Semantic Segmentation

**论文链接:** [http://arxiv.org/abs/2512.16567v1](http://arxiv.org/abs/2512.16567v1)

**作者:** Yin Zhang, Yongqiang Zhang, Yaoyue Zheng, Bogdan Raducanu, Dan Liu

**发布时间:** 2025-12-18

**备注:** Accepted by AAAI 2026

### GPT解析

### 总结

这篇论文提出了一种名为Causal-Tune的新型微调策略，用于从视觉基础模型中提取因果因素并抑制非因果因素，从而提高领域泛化语义分割的性能。

### 背景

现有方法主要关注训练轻量级适配器或改进中间特征来实现对未见领域的更好泛化，但忽视了长期预训练的视觉基础模型中存在的伪影(artifacts)，这些伪影阻碍了有价值表征的利用，最终降低了领域泛化语义分割的性能。

### 目的

明确检查视觉基础模型中特征的因果和非因果因素，并提出一种简单有效的方法来识别和解耦它们，实现更强大的领域泛化能力。

### 方法

作者提出了Causal-Tune方法：1)使用离散余弦变换(DCT)提取每层特征的频谱；2)应用高斯带通滤波器分离频谱为因果和非因果成分；3)引入因果感知的可学习令牌在频域操作并丢弃非因果成分；4)通过逆DCT将精炼特征转换回空间域并传递到下一层。

### 主要发现

通过因果机制观察发现，视觉基础模型中的伪影与非因果因素相关，这些因素通常位于模型频谱的低频和高频分量中。实验证明Causal-Tune在恶劣天气条件下表现优异，特别是在雪天条件下比基线方法提高了4.8%的mIoU。

### 结论

Causal-Tune方法通过识别和解耦视觉基础模型中的因果与非因果因素，显著提高了领域泛化语义分割的性能，特别是在恶劣天气条件下表现优异。

### 翻译

使用少量参数微调视觉基础模型在领域泛化语义分割任务中表现出显著性能。大多数现有工作要么训练轻量级适配器，要么改进中间特征，以实现对未见领域的更好泛化。然而，它们都忽视了长期预训练的视觉基础模型常常存在伪影，这阻碍了有价值表征的利用，最终降低了领域泛化语义分割性能。受因果机制的启发，我们观察到这些伪影与非因果因素相关，这些因素通常位于视觉基础模型频谱的低频和高频分量中。在本文中，我们明确检查了视觉基础模型中特征的因果和非因果因素，并提出了一种简单有效的方法来识别和解耦它们，从而实现更强大的领域泛化能力。具体来说，我们提出了Causal-Tune，一种新型微调策略，旨在从视觉基础模型的特征中提取因果因素并抑制非因果因素。首先，我们使用离散余弦变换提取每层特征的频谱。然后应用高斯带通滤波器将频谱分离为因果和非因果成分。为了进一步精炼因果成分，我们引入了一组因果感知的可学习令牌在频域操作，而非因果成分则被丢弃。最后，通过逆离散余弦变换将精炼的特征转换回空间域并传递到下一层。在各种跨域任务上进行的广泛实验证明了Causal-Tune的有效性。特别是，我们的方法在恶劣天气条件下实现了优越性能，在雪天条件下比基线提高了4.8%的平均交并比。


### 论文摘要

Fine-tuning Vision Foundation Models (VFMs) with a small number of parameters has shown remarkable performance in Domain Generalized Semantic Segmentation (DGSS). Most existing works either train lightweight adapters or refine intermediate features to achieve better generalization on unseen domains. However, they both overlook the fact that long-term pre-trained VFMs often exhibit artifacts, which hinder the utilization of valuable representations and ultimately degrade DGSS performance. Inspired by causal mechanisms, we observe that these artifacts are associated with non-causal factors, which usually reside in the low- and high-frequency components of the VFM spectrum. In this paper, we explicitly examine the causal and non-causal factors of features within VFMs for DGSS, and propose a simple yet effective method to identify and disentangle them, enabling more robust domain generalization. Specifically, we propose Causal-Tune, a novel fine-tuning strategy designed to extract causal factors and suppress non-causal ones from the features of VFMs. First, we extract the frequency spectrum of features from each layer using the Discrete Cosine Transform (DCT). A Gaussian band-pass filter is then applied to separate the spectrum into causal and non-causal components. To further refine the causal components, we introduce a set of causal-aware learnable tokens that operate in the frequency domain, while the non-causal components are discarded. Finally, refined features are transformed back into the spatial domain via inverse DCT and passed to the next layer. Extensive experiments conducted on various cross-domain tasks demonstrate the effectiveness of Causal-Tune. In particular, our method achieves superior performance under adverse weather conditions, improving +4.8% mIoU over the baseline in snow conditions.

---

## 41. Hearing to Translate: The Effectiveness of Speech Modality Integration into LLMs

**论文链接:** [http://arxiv.org/abs/2512.16378v1](http://arxiv.org/abs/2512.16378v1)

**作者:** Sara Papi, Javier Garcia Gilabert, Zachary Hopton, Vilém Zouhar, Carlos Escolano, Gerard I. Gállego, Jorge Iranzo-Sánchez, Ahrii Kim, Dominik Macháček, Patricia Schmidtova, Maike Züfle

**发布时间:** 2025-12-18

**备注:** Project available at https://github.com/sarapapi/hearing2translate

### GPT解析

### 总结

本文评估了将大型语言模型与语音结合的SpeechLLMs在语音翻译任务上的表现，并与传统的级联架构进行了比较。

### 背景

大型语言模型正在扩展到文本以外的领域，将语音作为原生模态整合，催生了SpeechLLMs，这些模型旨在直接翻译口语，绕过传统的基于转录的流水线架构。

### 目的

探究SpeechLLMs的整合是否比现有的级联架构提高了语音到文本翻译的质量。

### 方法

提出了第一个全面的测试套件"Hearing to Translate"，严格地将5种最先进的SpeechLLMs与16个强大的直接和级联系统进行了基准测试，这些系统结合了领先的语音基础模型和多语言LLMs。

### 主要发现

级联系统仍然是最可靠的，当前的SpeechLLMs只在选定的设置中与级联系统相匹配，语音基础模型则落后于两者。

### 结论

将大型语言模型整合到模型中或流水线中，对于高质量的语音翻译是必不可少的。

### 翻译

随着大型语言模型扩展到文本以外的领域，将语音作为原生模态进行整合催生了SpeechLLMs，这些模型旨在直接翻译口语，从而绕过传统的基于转录的流水线。然而，这种整合是否比已建立的级联架构提高了语音到文本翻译的质量，仍然是一个开放性问题。我们提出了"Hearing to Translate"，这是第一个全面的测试套件，严格地将5种最先进的SpeechLLMs与16个强大的直接和级联系统进行了基准测试，这些系统结合了领先的语音基础模型和多语言LLMs。我们的分析涵盖了16个基准测试、13种语言对和9个具有挑战性的条件，包括不流畅、嘈杂和长篇语音。在这广泛的评估中，我们发现级联系统仍然是最可靠的，而当前的SpeechLLMs只在选定的设置中与级联系统相匹配，语音基础模型则落后于两者，这表明将大型语言模型整合到模型中或流水线中，对于高质量的语音翻译是必不可少的。


### 论文摘要

As Large Language Models (LLMs) expand beyond text, integrating speech as a native modality has given rise to SpeechLLMs, which aim to translate spoken language directly, thereby bypassing traditional transcription-based pipelines. Whether this integration improves speech-to-text translation quality over established cascaded architectures, however, remains an open question. We present Hearing to Translate, the first comprehensive test suite rigorously benchmarking 5 state-of-the-art SpeechLLMs against 16 strong direct and cascade systems that couple leading speech foundation models (SFM), with multilingual LLMs. Our analysis spans 16 benchmarks, 13 language pairs, and 9 challenging conditions, including disfluent, noisy, and long-form speech. Across this extensive evaluation, we find that cascaded systems remain the most reliable overall, while current SpeechLLMs only match cascades in selected settings and SFMs lag behind both, highlighting that integrating an LLM, either within the model or in a pipeline, is essential for high-quality speech translation.

---

## 42. Adaptation of Agentic AI

**论文链接:** [http://arxiv.org/abs/2512.16301v1](http://arxiv.org/abs/2512.16301v1)

**作者:** Pengcheng Jiang, Jiacheng Lin, Zhiyi Shi, Zifeng Wang, Luxi He, Yichen Wu, Ming Zhong, Peiyang Song, Qizheng Zhang, Heng Wang, Xueqiang Xu, Hanwen Xu, Pengrui Han, Dylan Zhang, Jiashuo Sun, Chaoqi Yang, Kun Qian, Tian Wang, Changran Hu, Manling Li, Quanzheng Li, Hao Peng, Sheng Wang, Jingbo Shang, Chao Zhang, Jiaxuan You, Liyuan Liu, Pan Lu, Yu Zhang, Heng Ji, Yejin Choi, Dawn Song, Jimeng Sun, Jiawei Han

**发布时间:** 2025-12-18

### GPT解析

### 总结

该论文提出了一个系统性框架，统一了智能体AI系统中的适应机制，包括智能体适应和工具适应两大类，并进一步细分为不同形式。该框架有助于明确适应策略的设计空间，权衡利弊，并为系统设计提供实用指导。

### 背景

前沿的智能体AI系统建立在基础模型之上，这些模型可以被调整以进行规划、推理和与外部工具交互，从而执行日益复杂和专业的任务。随着这些系统能力和范围的扩大，适应机制成为提高性能、可靠性和泛化能力的关键。

### 目的

统一快速扩展的研究领域，构建涵盖智能体适应和工具适应的系统性框架；明确适应策略的设计空间，权衡利弊，为系统设计提供实用指导；回顾各类代表性方法，分析优势和局限，突出关键开放挑战和未来机遇。

### 方法

构建了一个系统性框架，将适应机制分为智能体适应和工具适应两大类；将智能体适应细分为工具执行信号驱动的智能体适应和智能体输出信号驱动的智能体适应；将工具适应细分为智能体无关的工具适应和智能体监督的工具适应。

### 主要发现

该框架有助于明确智能体AI中适应策略的设计空间，使权衡关系更加明确，并为系统设计过程中选择或切换策略提供实用指导。

### 结论

该论文旨在为寻求构建更强大、高效和可靠的智能体AI系统的研究人员和从业者提供概念基础和实用路线图。

### 翻译

前沿的智能体AI系统建立在基础模型之上，这些模型可以被调整以进行规划、推理和与外部工具交互，从而执行日益复杂和专业的任务。随着这些系统能力和范围的扩大，适应机制成为提高性能、可靠性和泛化能力的关键。在本文中，我们将快速扩展的研究领域统一为一个系统性框架，涵盖了智能体适应和工具适应。我们进一步将智能体适应细分为工具执行信号驱动的智能体适应和智能体输出信号驱动的智能体适应，以及将工具适应细分为智能体无关的工具适应和智能体监督的工具适应。我们证明，该框架有助于明确智能体AI中适应策略的设计空间，使权衡关系明确，并为系统设计过程中选择或切换策略提供实用指导。然后，我们回顾了各类中的代表性方法，分析了它们的优势和局限性，并突出了关键的开放挑战和未来机遇。总体而言，本文旨在为寻求构建更强大、高效和可靠的智能体AI系统的研究人员和从业者提供概念基础和实用路线图。


### 论文摘要

Cutting-edge agentic AI systems are built on foundation models that can be adapted to plan, reason, and interact with external tools to perform increasingly complex and specialized tasks. As these systems grow in capability and scope, adaptation becomes a central mechanism for improving performance, reliability, and generalization. In this paper, we unify the rapidly expanding research landscape into a systematic framework that spans both agent adaptations and tool adaptations. We further decompose these into tool-execution-signaled and agent-output-signaled forms of agent adaptation, as well as agent-agnostic and agent-supervised forms of tool adaptation. We demonstrate that this framework helps clarify the design space of adaptation strategies in agentic AI, makes their trade-offs explicit, and provides practical guidance for selecting or switching among strategies during system design. We then review the representative approaches in each category, analyze their strengths and limitations, and highlight key open challenges and future opportunities. Overall, this paper aims to offer a conceptual foundation and practical roadmap for researchers and practitioners seeking to build more capable, efficient, and reliable agentic AI systems.

---

## 43. Sigma-Moe-Tiny Technical Report

**论文链接:** [http://arxiv.org/abs/2512.16248v1](http://arxiv.org/abs/2512.16248v1)

**作者:** Qingguo Hu, Zhenghao Lin, Ziyue Yang, Yucheng Ding, Xiao Liu, Yuting Jiang, Ruizhe Wang, Tianyu Chen, Zhongxin Guo, Yifan Xiong, Rui Gao, Lei Qu, Jinsong Su, Peng Cheng, Yeyun Gong

**发布时间:** 2025-12-18

### GPT解析

### 总结

本文提出了Sigma-MoE-Tiny，一种高度稀疏的混合专家语言模型，通过细粒度专家分段和渐进式稀疏化方案实现了高效的参数利用和稳定的训练过程，同时保持了顶尖的性能。

### 背景

混合专家模型已成为基础模型的一种有前途的范式，因为它具有高效且强大的可扩展性。

### 目的

开发一种MoE语言模型，实现比现有开源模型更高的稀疏度，同时保持卓越的性能。

### 方法

采用细粒度专家分段，每层最多96个专家，每个标记只激活一个专家；提出渐进式稀疏化调度方案解决专家负载均衡问题；在多样化和高质量语料库上进行预训练，然后进行后训练以释放更多能力。

### 主要发现

尽管只激活50亿参数（占总参数的2.5%），Sigma-MoE-Tiny在规模相当或显著更大的同类模型中实现了顶尖性能；训练过程保持稳定，没有发生不可恢复的损失峰值；常用的负载均衡损失在较低层变得无效，需要新的解决方案。

### 结论

通过创新的专家分段和渐进式稀疏化方案，Sigma-MoE-Tiny成功解决了极端稀疏性带来的负载均衡挑战，为未来MoE架构的稀疏化发展提供了有价值的见解。

### 翻译

混合专家模型因其高效且强大的可扩展性已成为基础模型的一种有前途的范式。在这项工作中，我们提出了Sigma-MoE-Tiny，一种MoE语言模型，与现有开源模型相比实现了最高的稀疏度。Sigma-MoE-Tiny采用细粒度专家分段，每层最多96个专家，同时只为每个标记激活一个专家，总计200亿参数中只有50亿被激活。这种极端稀疏性带来的主要挑战在于专家负载均衡。我们发现，在这种设置下，广泛使用的负载均衡损失在较低层往往变得无效。为了解决这个问题，我们提出了一种渐进式稀疏化调度方案，旨在平衡专家利用率和训练稳定性。Sigma-MoE-Tiny在多样化和高质量的语料库上进行预训练，随后进行后训练以进一步释放其能力。整个训练过程保持显著稳定，没有发生不可恢复的损失峰值。全面的评估表明，尽管只激活50亿参数，Sigma-MoE-Tiny在规模相当或显著更大的同类模型中实现了顶尖性能。此外，我们对高度稀疏MoE模型中的负载平衡进行了深入讨论，为未来MoE架构的稀疏化进展提供了见解。


### 论文摘要

Mixture-of-Experts (MoE) has emerged as a promising paradigm for foundation models due to its efficient and powerful scalability. In this work, we present Sigma-MoE-Tiny, an MoE language model that achieves the highest sparsity compared to existing open-source models. Sigma-MoE-Tiny employs fine-grained expert segmentation with up to 96 experts per layer, while activating only one expert for each token, resulting in 20B total parameters with just 0.5B activated. The major challenge introduced by such extreme sparsity lies in expert load balancing. We find that the widely-used load balancing loss tends to become ineffective in the lower layers under this setting. To address this issue, we propose a progressive sparsification schedule aiming to balance expert utilization and training stability. Sigma-MoE-Tiny is pre-trained on a diverse and high-quality corpus, followed by post-training to further unlock its capabilities. The entire training process remains remarkably stable, with no occurrence of irrecoverable loss spikes. Comprehensive evaluations reveal that, despite activating only 0.5B parameters, Sigma-MoE-Tiny achieves top-tier performance among counterparts of comparable or significantly larger scale. In addition, we provide an in-depth discussion of load balancing in highly sparse MoE models, offering insights for advancing sparsity in future MoE architectures.   Project page: https://qghuxmu.github.io/Sigma-MoE-Tiny   Code: https://github.com/microsoft/ltp-megatron-lm

---

## 44. AlignMerge - Alignment-Preserving Large Language Model Merging via Fisher-Guided Geometric Constraints

**论文链接:** [http://arxiv.org/abs/2512.16245v1](http://arxiv.org/abs/2512.16245v1)

**作者:** Aniruddha Roy, Jyoti Patel, Aman Chadha, Vinija Jain, Amitava Das

**发布时间:** 2025-12-18

### GPT解析

### 总结

AlignMerge是一种几何感知的大型语言模型合并方法，能够在组合多个模型能力的同时保持对齐性，通过在Fisher-Rao几何空间中优化合并过程，确保模型在保持性能的同时不损害安全性和对齐性。

### 背景

合并大型语言模型(LLMs)是一种实用的方法，可以从多个微调检查点组合能力而无需重新训练。然而，标准方案（线性权重混合、任务向量和Fisher加权平均）可以在保持损失的同时悄悄破坏对齐。

### 目的

提出一种几何感知的合并框架，使对齐成为显式不变量，解决模型合并过程中对齐性被破坏的问题。

### 方法

引入AlignMerge框架，在基于指令微调的基础模型的局部Fisher图中估计对齐子空间，使用投影器P_A，并优化目标函数L_AlignMerge = L_geo + lambda_align * L_align + lambda_bud * L_bud，其中L_geo保持合并结果接近专家模型，L_align惩罚沿对齐敏感方向的移动，L_bud执行软对齐预算。使用解码不变的对齐质量指数(AQI)作为对齐函数。

### 主要发现

在五个模型家族（LLaMA-3 8B、Mistral 7B、Qwen 2、Phi-3.5、Gemma 2）上测试，AlignMerge提高了对齐指标（AQI、毒性、LLM判断对齐），在指令遵循、推理和帮助性方面匹配或超过最佳专家，且与Fisher混合、TIES、SafeMerge和MergeAlign相比，对齐子空间漂移更小，预算违规更少。

### 结论

使保持对齐的合并成为一流的设计目标，为未来基础模型的几何感知组合提供了一条路径。

### 翻译

合并大型语言模型(LLMs)是一种实用的方法，可以从多个微调检查点组合能力而无需重新训练。然而，标准方案（线性权重混合、任务向量和Fisher加权平均）可以在保持损失的同时悄悄破坏对齐。我们认为合并不是数字技巧，而是围绕已对齐锚点的几何约束操作：融合必须引导以尊重安全几何，而不是事后验证。我们引入了AlignMerge，一种几何感知的合并框架，使对齐成为显式不变量。在基于指令微调的基础模型的局部Fisher图中，我们使用投影器P_A估计一个对齐子空间，并优化：L_AlignMerge = L_geo + lambda_align * L_align + lambda_bud * L_bud，其中L_geo使合并结果在Fisher-Rao几何中接近其专家模型，L_align惩罚沿对齐敏感方向的移动，L_bud强制执行软对齐预算。作为对齐函数，我们使用解码不变的对齐质量指数(AQI)，这是一个潜在空间标准，捕捉表示空间中对齐和非对齐行为的分离程度。在五个模型家族（LLaMA-3 8B、Mistral 7B、Qwen 2、Phi-3.5、Gemma 2）上，将安全锚点与任务专家合并时，AlignMerge提高了对齐指标（AQI、毒性、LLM判断对齐），同时在指令遵循、推理和帮助性方面匹配或超过最佳专家。与Fisher混合、TIES、SafeMerge和MergeAlign相比，它显示出更小的对齐子空间漂移和更少的预算违规。这些结果使保持对齐的合并成为一流的设计目标，并暗示了未来基础模型几何感知组合的路径。


### 论文摘要

Merging large language models (LLMs) is a practical way to compose capabilities from multiple fine-tuned checkpoints without retraining. Yet standard schemes (linear weight soups, task vectors, and Fisher-weighted averaging) can preserve loss while quietly destroying alignment. We argue that merging is not a numerical trick but a geometry-constrained operation around an already-aligned anchor: fusion must be steered to respect safety geometry, not validated post hoc.   We introduce AlignMerge, a geometry-aware merging framework that makes alignment an explicit invariant. In a local Fisher chart around an instruction-tuned base, we estimate an alignment subspace with projector P_A and optimize:   L_AlignMerge = L_geo + lambda_align * L_align + lambda_bud * L_bud,   where L_geo keeps the merge close to its experts in Fisher-Rao geometry, L_align penalizes motion along alignment-sensitive directions, and L_bud enforces a soft alignment budget. As the alignment functional we use the decoding-invariant Alignment Quality Index (AQI), a latent-space criterion that captures how cleanly aligned and misaligned behaviors separate in representation space.   Across five model families (LLaMA-3 8B, Mistral 7B, Qwen 2, Phi-3.5, Gemma 2), merging safety anchors with task experts, AlignMerge improves alignment metrics (AQI, toxicity, LLM-judge alignment) while matching or exceeding the best expert on instruction-following, reasoning, and helpfulness. It also exhibits smaller alignment-subspace drift and fewer budget violations than Fisher soups, TIES, SafeMerge, and MergeAlign. These results make alignment-preserving merging a first-class design goal and suggest a path to geometry-aware composition of future foundation models.

---

## 45. Conversational Time Series Foundation Models: Towards Explainable and Effective Forecasting

**论文链接:** [http://arxiv.org/abs/2512.16022v1](http://arxiv.org/abs/2512.16022v1)

**作者:** Defu Cao, Michael Gee, Jinbo Liu, Hengxuan Wang, Wei Yang, Rui Wang, Yan Liu

**发布时间:** 2025-12-17

**备注:** 31Pages

### GPT解析

### 总结

该研究提出了一种新方法，将大型语言模型(LLM)重新定位为智能评判者，用于评估、解释和协调时间序列基础模型的集成。通过R1风格的微调过程和基于SHAP的忠实度分数，使LLM能够理解时间序列数据的动态特性，并通过多轮对话优化集成策略。该方法在GIFT-Eval基准上取得了最先进的结果。

### 背景

时间序列基础模型众多，但没有一种方法能够持续表现出优越性。大型语言模型虽然具有强大的推理能力，但直接应用于时间序列预测效果不佳。

### 目的

解决时间序列预测中的模型选择和集成问题，通过将LLM作为智能评判者来优化基础模型集成，提高预测性能和可解释性。

### 方法

1. 将LLM重新定位为智能评判者，评估和协调基础模型集成
2. 引入R1风格的微调过程，由基于SHAP的忠实度分数指导
3. 教会模型将集成权重解释为关于时间动态的因果陈述
4. 通过迭代的多轮对话进行前瞻性评估和策略优化

### 主要发现

在GIFT-Eval基准上，该方法在23个数据集的97种设置中，在CRPS和MASE指标上显著领先于其他时间序列基础模型。

### 结论

将LLM作为智能评判者来协调时间序列基础模型集成是一种有效的方法，能够显著提高预测性能，同时提供基于因果的解释，增强了模型的可解释性。

### 翻译

时间序列基础模型的激增创造了一种局面，没有单一方法能够持续表现出优越性，将核心挑战框定为寻找最佳模型，而是编排具有可解释性的最优集成。虽然大型语言模型(LLMs)提供了强大的推理能力，但它们在时间序列预测中的直接应用已被证明无效。我们通过将LLM重新定位为智能评判者来解决这一差距，该评判者评估、解释和战略性地协调基础模型集成。为了克服LLM在时间序列领域特定知识的固有缺乏，我们引入了一种R1风格的微调过程，由基于SHAP的忠实度分数指导，教会模型将集成权重解释为关于时间动态的有意义因果陈述。训练后的代理随后进行迭代的多轮对话，进行前瞻性评估，为其权重决策提供基于因果的解释，并自适应地优化策略。在GIFT-Eval基准上验证，涵盖23个数据集的97种设置，我们的方法在CRPS和MASE指标上显著领先于领先的时间序列基础模型，建立了新的最先进结果。


### 论文摘要

The proliferation of time series foundation models has created a landscape where no single method achieves consistent superiority, framing the central challenge not as finding the best model, but as orchestrating an optimal ensemble with interpretability. While Large Language Models (LLMs) offer powerful reasoning capabilities, their direct application to time series forecasting has proven ineffective. We address this gap by repositioning the LLM as an intelligent judge that evaluates, explains, and strategically coordinates an ensemble of foundation models. To overcome the LLM's inherent lack of domain-specific knowledge on time series, we introduce an R1-style finetuning process, guided by SHAP-based faithfulness scores, which teaches the model to interpret ensemble weights as meaningful causal statements about temporal dynamics. The trained agent then engages in iterative, multi-turn conversations to perform forward-looking assessments, provide causally-grounded explanations for its weighting decisions, and adaptively refine the optimization strategy. Validated on the GIFT-Eval benchmark on 23 datasets across 97 settings, our approach significantly outperforms leading time series foundation models on both CRPS and MASE metrics, establishing new state-of-the-art results.

---

## 46. Are vision-language models ready to zero-shot replace supervised classification models in agriculture?

**论文链接:** [http://arxiv.org/abs/2512.15977v1](http://arxiv.org/abs/2512.15977v1)

**作者:** Earl Ranario, Mason J. Earles

**发布时间:** 2025-12-17

**备注:** Draft version

### GPT解析

### 总结

该研究评估了视觉语言模型(VLMs)在农业分类任务中的表现，发现当前现成的VLMs不适合作为独立农业诊断系统，但可作为辅助组件与特定策略结合使用。

### 背景

视觉语言模型(VLMs)被越来越多地提议作为视觉识别任务的通用解决方案，但它们在农业决策支持中的可靠性仍不清楚。

### 目的

评估VLMs在农业分类任务中的适用性和性能，了解它们是否适合作为农业诊断系统。

### 方法

在AgML集合中的27个农业分类数据集上测试多种开源和闭源VLMs，涵盖162个类别；比较零样本VLMs与监督基线(YOLO11)的性能；测试多选题和开放式提示方法；应用LLM-based语义判断；进行任务级别分析。

### 主要发现

零样本VLMs显著低于监督基线；多选题提示下最佳模型(Gemini-3 Pro)达62%准确率，开放式提示低于25%；语义判断可提高开放式准确率；开源模型中Qwen-VL-72B表现最佳；植物分类比虫害识别更容易。

### 结论

当前现成的VLMs不适合作为独立农业诊断系统，但当与受限界面、显式标签本体和领域感知评估策略结合时，可作为有效辅助组件。

### 翻译

视觉语言模型(VLMs)越来越多地被提议作为视觉识别任务的通用解决方案，但它们在农业决策支持中的可靠性仍不明确。我们在AgML集合中的27个农业分类数据集上对多种开源和闭源VLMs进行了基准测试，涵盖162个类别，包括植物病害、虫害和损害以及植物和杂草物种识别。在所有任务中，零样本VLMs的表现显著低于监督任务特定基线(YOLO11)，后者始终比任何基础模型获得明显更高的准确率。在多选题提示下，表现最佳的VLM(Gemini-3 Pro)达到约62%的平均准确率，而开放式提示的表现要低得多，原始准确率通常低于25%。应用基于LLM的语义判断提高了开放式准确率(例如，顶级模型从21%提高到30%)并改变了模型排名，证明评估方法显著影响报告的结论。在开源模型中，Qwen-VL-72B表现最好，在受限提示下接近闭源性能，但仍落后于顶级专有系统。任务级别分析显示，植物和杂草物种分类始终比虫害和损害识别更容易，后者是所有模型中最具挑战性的类别。总体而言，这些结果表明当前的现成VLMs还不适合作为独立的农业诊断系统，但当与受限界面、显式标签本体和领域感知评估策略配对时，可以作为辅助组件发挥作用。


### 论文摘要

Vision-language models (VLMs) are increasingly proposed as general-purpose solutions for visual recognition tasks, yet their reliability for agricultural decision support remains poorly understood. We benchmark a diverse set of open-source and closed-source VLMs on 27 agricultural classification datasets from the AgML collection, spanning 162 classes across plant disease, pest and damage, and plant and weed species identification. Across all tasks, zero-shot VLMs substantially underperform a supervised task-specific baseline (YOLO11), which consistently achieves markedly higher accuracy than any foundation model. Under multiple-choice prompting, the best-performing VLM (Gemini-3 Pro) reaches approximately 62% average accuracy, while open-ended prompting yields much lower performance, with raw accuracies typically below 25%. Applying LLM-based semantic judging increases open-ended accuracy (for example, from 21% to 30% for top models) and alters model rankings, demonstrating that evaluation methodology meaningfully affects reported conclusions. Among open-source models, Qwen-VL-72B performs best, approaching closed-source performance under constrained prompting but still trailing top proprietary systems. Task-level analysis shows that plant and weed species classification is consistently easier than pest and damage identification, which remains the most challenging category across models. Overall, these results indicate that current off-the-shelf VLMs are not yet suitable as standalone agricultural diagnostic systems, but can function as assistive components when paired with constrained interfaces, explicit label ontologies, and domain-aware evaluation strategies.

---

## 47. Small Language Models for Efficient Agentic Tool Calling: Outperforming Large Models with Targeted Fine-tuning

**论文链接:** [http://arxiv.org/abs/2512.15943v1](http://arxiv.org/abs/2512.15943v1)

**作者:** Polaris Jhandi, Owais Kazi, Shreyas Subramanian, Neel Sendas

**发布时间:** 2025-12-17

### GPT解析

### 总结

该研究探讨了通过优化小型语言模型(SLM)替代大型语言模型(LLM)驱动的工作流程，以实现生成式AI的成本优化和运营效率提升。研究训练了一个领域适应的SLM来执行传统上由LLM处理的任务，实验结果显示微调后的SLM在ToolBench评估中取得77.55%的通过率，显著优于所有基线模型。

### 背景

随着组织扩展采用生成式AI，模型成本优化和运营效率已成为决定可持续性和可访问性的关键因素。大型语言模型虽然在各种任务上展现出令人印象深刻的能力，但其广泛的计算需求使其成为常规企业使用的成本障碍，这促使研究人员探索小型语言模型。

### 目的

探索用优化的小型语言模型替代大型语言模型驱动的工作流程的可行性，训练一个领域适应的SLM来执行传统上由LLM处理的代表性任务，并评估其性能。

### 方法

采用领域适应方法训练小型语言模型，使用facebook/opt-350m模型仅通过一个epoch进行微调，采用Hugging Face TRL中的监督微调训练器。该模型执行文档摘要、查询回答和结构数据解释等任务。

### 主要发现

实验结果表明，微调后的SLM在ToolBench评估中取得了77.55%的通过率，显著优于所有基线模型，包括ChatGPT-CoT(26.00%)、ToolLLaMA-DFS(30.18%)和ToolLLaMA-CoT(16.27%)。这表明即使在350M参数规模，模型也可以对指令微调管道做出有意义贡献。

### 结论

小型语言模型的精心设计和针对性训练可以显著降低采用障碍，使生成式AI能够以经济有效的方式大规模集成到生产系统中。

### 翻译

随着组织扩展采用生成式AI，模型成本优化和运营效率已成为决定可持续性和可访问性的关键因素。虽然大型语言模型在各种任务上展现出令人印象深刻的能力，但其广泛的计算需求使其成为常规企业使用的成本障碍。这一局限性促使研究人员探索小型语言模型，它们可以在特定应用中提供可比较的性能，同时显著降低基础设施开销。在本工作中，我们研究了用优化的小型语言模型替代大型语言模型驱动的工作流程的可行性。我们训练了一个领域适应的SLM来执行传统上由LLM处理的代表性任务，如文档摘要、查询回答和结构数据解释。作为实验的一部分，我们使用Hugging Face TRL研究了facebook/opt-350m模型的微调，特别是监督微调训练器。实验结果表明，我们微调的SLM在ToolBench评估中取得了77.55%的通过率，显著优于所有基线模型。这些发现强调，小型语言模型的精心设计和针对性训练可以显著降低采用障碍，使生成式AI能够以经济有效的方式大规模集成到生产系统中。


### 论文摘要

As organizations scale adoption of generative AI, model cost optimization and operational efficiency have emerged as critical factors determining sustainability and accessibility. While Large Language Models (LLMs) demonstrate impressive capabilities across diverse tasks, their extensive computational requirements make them cost-prohibitive for routine enterprise use. This limitation motivates the exploration of Small Language Models (SLMs), which can deliver comparable performance in targeted applications while drastically reducing infrastructure overhead (Irugalbandara et al., 2023). In this work, we investigate the feasibility of replacing LLM-driven workflows with optimized SLMs. We trained a domain-adapted SLM to execute representative tasks traditionally handled by LLMs, such as document summarization, query answering, and structured data interpretation. As part of the experiment, we investigated the fine-tuning of facebook/opt-350m model (single epoch only) using the Hugging Face TRL (Transformer Reinforcement Learning), specifically the Supervised Fine-Tuning (SFT) trainer. The OPT-350M model was released by Meta AI in 2022 as part of the OPT (Open Pretrained Transformer) family of models. Similar studies demonstrate that even models at the 350M parameter scale can meaningfully contribute to instruction-tuning pipelines (Mekala et al., 2024). Experimental results demonstrated that our fine-tuned SLM achieves exceptional performance with a 77.55\% pass rate on ToolBench evaluation, significantly outperforming all baseline models including ChatGPT-CoT (26.00\%), ToolLLaMA-DFS (30.18\%), and ToolLLaMA-CoT (16.27\%). These findings emphasize that thoughtful design and targeted training of SLMs can significantly lower barriers to adoption, enabling cost-effective, large-scale integration of generative AI into production systems.

---

## 48. BarcodeMamba+: Advancing State-Space Models for Fungal Biodiversity Research

**论文链接:** [http://arxiv.org/abs/2512.15931v1](http://arxiv.org/abs/2512.15931v1)

**作者:** Tiancheng Gao, Scott C. Lowe, Brendan Furneaux, Angel X Chang, Graham W. Taylor

**发布时间:** 2025-12-17

**备注:** 11 pages, accepted at the 3rd Workshop on Imageomics: Discovering Biological Knowledge from Images Using AI (NeurIPS 2025)

### GPT解析

### 总结

该研究介绍了BarcodeMamba+，一种基于状态空间模型架构的真菌条形码分类基础模型，采用预训练和微调范式，结合多种增强技术解决了真菌分类中的挑战。

### 背景

从DNA条形码进行准确分类是全球生物多样性监测的基础，但真菌因标记稀疏和长尾分类群分布而面临极大挑战。传统监督学习方法难以泛化到未见物种且无法捕捉数据层次结构。

### 目的

开发一种有效的基础模型解决真菌分类挑战，特别是在标记稀疏环境下提高分类性能并捕捉数据层次性质。

### 方法

构建BarcodeMamba+基础模型；采用预训练和微调范式利用部分标记数据；在微调过程中整合层次标签平滑、加权损失函数和来自MycoAI的多头输出层等增强技术。

### 主要发现

预训练和微调范式在数据稀疏环境中比传统全监督方法更有效；每种增强技术都带来显著性能提升；在具有挑战性的真菌分类基准测试上，最终模型在各种分类水平上都优于现有方法。

### 结论

该研究为基于基因组的生物多样性研究提供了强大新工具，并为这一具有挑战性的领域建立了有效且可扩展的训练范式。

### 翻译

从DNA条形码进行准确的分类是全球生物多样性监测的基石，但由于标记稀疏和长尾分类群分布，真菌面临着极端挑战。传统的监督学习方法在这个领域往往表现不佳，难以泛化到未见过的物种，也无法捕捉数据的层次性质。为解决这些限制，我们引入了BarcodeMamba+，这是一种基于强大高效的状态空间模型架构构建的真菌条形码分类基础模型。我们采用预训练和微调范式，利用部分标记数据，并证明在数据稀疏环境中，这比传统全监督方法更有效。在微调过程中，我们系统地整合和评估了一系列增强技术，包括层次标签平滑、加权损失函数以及来自MycoAI的多头输出层，以专门解决真菌分类学的挑战。我们的实验表明，每个组件都带来了显著的性能提升。在一个具有挑战性的真菌分类基准测试上，该基准测试的训练集与测试集之间存在明显的分类分布差异，我们的最终模型在各种分类水平上都优于一系列现有方法。我们的工作为基于基因组的生物多样性研究提供了强大的新工具，并为这一具有挑战性的领域建立了有效且可扩展的训练范式。我们的代码已在https://github.com/bioscan-ml/BarcodeMamba上公开。


### 论文摘要

Accurate taxonomic classification from DNA barcodes is a cornerstone of global biodiversity monitoring, yet fungi present extreme challenges due to sparse labelling and long-tailed taxa distributions. Conventional supervised learning methods often falter in this domain, struggling to generalize to unseen species and to capture the hierarchical nature of the data. To address these limitations, we introduce BarcodeMamba+, a foundation model for fungal barcode classification built on a powerful and efficient state-space model architecture. We employ a pretrain and fine-tune paradigm, which utilizes partially labelled data and we demonstrate this is substantially more effective than traditional fully-supervised methods in this data-sparse environment. During fine-tuning, we systematically integrate and evaluate a suite of enhancements--including hierarchical label smoothing, a weighted loss function, and a multi-head output layer from MycoAI--to specifically tackle the challenges of fungal taxonomy. Our experiments show that each of these components yields significant performance gains. On a challenging fungal classification benchmark with distinct taxonomic distribution shifts from the broad training set, our final model outperforms a range of existing methods across all taxonomic levels. Our work provides a powerful new tool for genomics-based biodiversity research and establishes an effective and scalable training paradigm for this challenging domain. Our code is publicly available at https://github.com/bioscan-ml/BarcodeMamba.

---

## 49. Seeing Beyond Words: Self-Supervised Visual Learning for Multimodal Large Language Models

**论文链接:** [http://arxiv.org/abs/2512.15885v1](http://arxiv.org/abs/2512.15885v1)

**作者:** Davide Caffagni, Sara Sarto, Marcella Cornia, Lorenzo Baraldi, Pier Luigi Dovesi, Shaghayegh Roohi, Mark Granroth-Wilding, Rita Cucchiara

**发布时间:** 2025-12-17

### GPT解析

### 总结

本文介绍了JARVIS，一种受JEPA启发的框架，用于提高多模态大语言模型(MLLMs)的视觉理解能力，特别是在基础视觉推理任务方面。

### 背景

多模态大语言模型(MLLMs)在连接视觉和语言方面展现了令人印象深刻的能力，但在基础视觉推理任务方面的能力仍然有限。

### 目的

解决MLLMs在视觉理解方面的局限性，提高其在视觉推理任务中的表现。

### 方法

引入JARVIS框架，将I-JEPA学习范式整合到MLLMs训练的标准视觉语言对齐流程中。利用冻结的视觉基础模型作为上下文和目标编码器，训练LLM的早期层作为预测器，从图像中学习结构和语义规律，减少对语言监督的依赖。

### 主要发现

在标准MLLM基准上的大量实验表明，JARVIS在不同LLM家族的以视觉为中心的基准上一致提高了性能，同时没有降低多模态推理能力。

### 结论

JARVIS框架有效提升了MLLMs的视觉理解能力，特别是在视觉推理任务方面，为多模态模型的发展提供了新的思路。

### 翻译

多模态大语言模型(MLLMs)最近在连接视觉和语言方面展示了令人印象深刻的能力，然而它们在基础视觉推理任务方面的熟练程度仍然有限。这种限制可以归因于MLLMs主要从文本描述中学习视觉理解，而文本描述构成了主观且本质上不完整的监督信号。此外，与大规模仅文本预训练相比，多模态指令调整的规模较小，导致MLLMs过度拟合语言先验而忽视视觉细节。为了解决这些问题，我们引入了JARVIS，一种受JEPA启发的框架，用于MLLMs中的自监督视觉增强。具体来说，我们将I-JEPA学习范式整合到MLLMs训练的标准视觉语言对齐流程中。我们的方法利用冻结的视觉基础模型作为上下文和目标编码器，同时训练预测器(实现为LLM的早期层)，从图像中学习结构和语义规律，而不完全依赖语言监督。在标准MLLM基准上的大量实验表明，JARVIS在不同LLM家族的以视觉为中心的基准上一致提高了性能，同时没有降低多模态推理能力。我们的源代码已在https://github.com/aimagelab/JARVIS公开可用。


### 论文摘要

Multimodal Large Language Models (MLLMs) have recently demonstrated impressive capabilities in connecting vision and language, yet their proficiency in fundamental visual reasoning tasks remains limited. This limitation can be attributed to the fact that MLLMs learn visual understanding primarily from textual descriptions, which constitute a subjective and inherently incomplete supervisory signal. Furthermore, the modest scale of multimodal instruction tuning compared to massive text-only pre-training leads MLLMs to overfit language priors while overlooking visual details. To address these issues, we introduce JARVIS, a JEPA-inspired framework for self-supervised visual enhancement in MLLMs. Specifically, we integrate the I-JEPA learning paradigm into the standard vision-language alignment pipeline of MLLMs training. Our approach leverages frozen vision foundation models as context and target encoders, while training the predictor, implemented as the early layers of an LLM, to learn structural and semantic regularities from images without relying exclusively on language supervision. Extensive experiments on standard MLLM benchmarks show that JARVIS consistently improves performance on vision-centric benchmarks across different LLM families, without degrading multimodal reasoning abilities. Our source code is publicly available at: https://github.com/aimagelab/JARVIS.

---

## 50. Reusable theory representations for colliders: a demonstrator SMEFT foundation model

**论文链接:** [http://arxiv.org/abs/2512.15862v1](http://arxiv.org/abs/2512.15862v1)

**作者:** Supratim Das Bakshi, T. J. Hobbs, Brandon Kriesten

**发布时间:** 2025-12-17

**备注:** 39 pages, 12 figures, 2 tables

### GPT解析

### 总结

这篇论文开发了一个用于探索标准模型有效场理论的基础模型，通过对比表示理论模拟的中性电流Drell-Yan截面构建而成。

### 背景

在高能对撞机上进行新物理搜索需要有效的理论分析方法，标准模型有效场理论(SMEFT)是描述超出标准模型物理的理论框架。

### 目的

开发一个可重用、物理对齐的基础表示，用于在高能对撞机上探索新物理理论，特别是通过分析SMEFT引起的Drell-Yan谱变形。

### 方法

通过Warsaw基中维度-6 Wilson系数空间的受控采样生成高分辨率微分分布，使用监督对比损失训练最小参数化编码器网络，创建低维潜在流形来表示SMEFT引起的变形。

### 主要发现

潜在方向与特征性的SMEFT形状畸变相关；嵌入中的簇对应于具有相似现象学影响的Wilson系数配置族；学习到的表示支持分类、异常检测和最近邻检索等下游任务。

### 结论

这项研究为高能对撞机上的新物理搜索理论提供了基础表示的第一步，尽管目前仅限于leading-order SMEFT和简化的不确定性建模。

### 翻译

我们为对撞机尺度的标准模型有效场理论(SMEFT)探索开发了一个演示基础模型，该模型由理论模拟的中性电流Drell-Yan截面的对比表示构建而成。通过在Warsaw基维度-6 Wilson系数空间O(Λ^-2)阶进行受控采样，我们生成了m_ℓℓ和p_T中的高分辨率微分分布，并添加了具有相关不确定性的物理动机蒙特卡罗副本。使用具有监督对比损失的最小参数化编码器网络进行训练，以产生低维潜在流形，SMEFT引起的Drell-Yan谱变形在此流形上获得明确的几何结构。我们分析得到的嵌入并证明：(i)潜在方向与特征性的SMEFT形状畸变相关，包括能量增长的四费米子贡献和电弱顶点修正；(ii)嵌入中的簇对应于具有相似现象学影响的Wilson系数配置族；(iii)学习到的表示支持带不确定性量化的分类、异常检测和最近邻检索等下游任务。虽然仅限于leading-order SMEFT和简化的不确定性建模，这项研究为高能对撞机上新物理搜索理论提供了第一个可重用、物理对齐的基础表示步骤。我们概述了向完整全局分析扩展的方案，包括多进程训练语料库、高阶修正和多目标预训练。


### 论文摘要

We develop a demonstrator foundation model for collider-scale explorations of the Standard Model Effective Field Theory (SMEFT), constructed from contrastive representations of theoretically simulated neutral-current Drell-Yan cross sections. Using a controlled sampling of the Warsaw-basis dimension-6 Wilson-coefficient space at $O(Λ^{-2})$, we generate a corpus of high-resolution differential distributions in $m_{\ell\ell}$ and $p_{T}$, augmented by physics-motivated Monte Carlo replicas with correlated uncertainties. A minimally parameterized encoder network is trained with a supervised contrastive loss to produce a low-dimensional latent manifold on which SMEFT-induced deformations of the Drell-Yan spectrum acquire a well-defined geometric structure. We analyze the resulting embedding and demonstrate that (i) latent directions correlate with characteristic SMEFT shape distortions, including energy-growing four-fermion contributions and electroweak vertex corrections; (ii) clusters in the embedding correspond to families of Wilson-coefficient configurations with similar phenomenological impact; and (iii) the learned representation supports downstream tasks such as classification with uncertainty quantification, anomaly detection, and nearest-neighbor retrieval. While restricted to leading-order SMEFT and simplified uncertainty modeling, this study provides the first step toward a reusable, physics-aligned foundational representation for the theory of New-Physics searches at high-energy colliders. We outline extensions towards a complete global analyses, including multi-process training corpora, higher-order corrections, and multi-objective pretraining.

---

## 51. Large Video Planner Enables Generalizable Robot Control

**论文链接:** [http://arxiv.org/abs/2512.15840v1](http://arxiv.org/abs/2512.15840v1)

**作者:** Boyuan Chen, Tianyuan Zhang, Haoran Geng, Kiwhan Song, Caiyi Zhang, Peihao Li, William T. Freeman, Jitendra Malik, Pieter Abbeel, Russ Tedrake, Vincent Sitzmann, Yilun Du

**发布时间:** 2025-12-17

**备注:** 29 pages, 16 figures

### GPT解析

### 总结

该研究探索了一种替代范式，使用大规模视频预训练作为构建机器人基础模型的主要模态，而非传统的多模态大语言模型扩展方法。研究团队收集了互联网规模的视频数据集，训练了生成式机器人规划模型，并通过真实机器人实验验证了其任务泛化能力和物理执行可行性。

### 背景

通用机器人需要能在不同任务和环境中泛化的决策模型。最近的工作通过将多模态大语言模型(MLLMs)扩展为动作输出来构建机器人基础模型，创建视觉-语言-动作(VLA)系统，这些工作基于MLLMs的大规模语言和图像预训练可以有效地转移到动作输出模态的直觉。

### 目的

探索使用大规模视频预训练作为构建机器人基础模型的主要模态的替代范式，而非依赖静态图像和语言。

### 方法

收集互联网规模的人类活动和任务演示视频数据集；首次在基础模型规模上训练开放视频模型用于生成式机器人规划；生成零样本视频计划并后处理提取可执行机器人动作；通过第三方选择的野外任务和真实机器人实验评估任务级别的泛化能力。

### 主要发现

模型能够成功执行物理任务；展示了强大的指令跟随能力、强大的泛化能力和现实世界的可行性。

### 结论

视频预训练可以作为构建机器人基础模型的有效替代方法；研究团队发布了模型和数据集以支持开放、可复现的基于视频的机器人学习。

### 翻译

通用机器人需要能在不同任务和环境中泛化的决策模型。最近的工作通过将多模态大语言模型(MLLMs)扩展为动作输出来构建机器人基础模型，创建视觉-语言-动作(VLA)系统。这些工作的动机是直觉认为MLLMs的大规模语言和图像预训练可以有效地转移到动作输出模态。在这项工作中，我们探索了使用大规模视频预训练作为构建机器人基础模型的主要模态的替代范式。与静态图像和语言不同，视频捕获了物理世界中状态和动作的时空序列，这些序列与机器人行为自然对齐。我们收集了一个互联网规模的人类活动和任务演示视频数据集，并首次在基础模型规模上训练了一个开放视频模型，用于生成式机器人规划。该模型为新颖场景和任务生成零样本视频计划，我们对其进行后处理以提取可执行的机器人动作。我们通过第三方选择的野外任务和真实机器人实验评估任务级别的泛化能力，展示了成功的物理执行。总之，这些结果表明了强大的指令跟随能力、强大的泛化能力和现实世界的可行性。我们发布了模型和数据集以支持开放、可复现的基于视频的机器人学习。我们的网站可在https://www.boyuan.space/large-video-planner/获取。


### 论文摘要

General-purpose robots require decision-making models that generalize across diverse tasks and environments. Recent works build robot foundation models by extending multimodal large language models (MLLMs) with action outputs, creating vision-language-action (VLA) systems. These efforts are motivated by the intuition that MLLMs' large-scale language and image pretraining can be effectively transferred to the action output modality. In this work, we explore an alternative paradigm of using large-scale video pretraining as a primary modality for building robot foundation models. Unlike static images and language, videos capture spatio-temporal sequences of states and actions in the physical world that are naturally aligned with robotic behavior. We curate an internet-scale video dataset of human activities and task demonstrations, and train, for the first time at a foundation-model scale, an open video model for generative robotics planning. The model produces zero-shot video plans for novel scenes and tasks, which we post-process to extract executable robot actions. We evaluate task-level generalization through third-party selected tasks in the wild and real-robot experiments, demonstrating successful physical execution. Together, these results show robust instruction following, strong generalization, and real-world feasibility. We release both the model and dataset to support open, reproducible video-based robot learning. Our website is available at https://www.boyuan.space/large-video-planner/.

---

## 52. Foundation Models in Biomedical Imaging: Turning Hype into Reality

**论文链接:** [http://arxiv.org/abs/2512.15808v1](http://arxiv.org/abs/2512.15808v1)

**作者:** Amgad Muneer, Kai Zhang, Ibraheem Hamdi, Rizwan Qureshi, Muhammad Waqas, Shereen Fouad, Hazrat Ali, Syed Muhammad Anwar, Jia Wu

**发布时间:** 2025-12-17

**备注:** 5 figures and 3 tables

### GPT解析

### 总结

基础模型正在推动人工智能在生物医学成像等领域的显著变革，但临床评估和部署面临重大挑战。文章评估了当前技术状态，提供了推理分类法，讨论了因果推断的重要性，以及部署中的信任、偏见和安全问题。未来发展方向是开发混合的、具有因果意识且可验证安全的系统，增强而非替代人类专业知识。

### 背景

基础模型正在推动人工智能在生物医学成像等不同领域的变革，这些模型旨在超越狭隘的模式识别，模拟复杂的临床推理、理解复杂的空间关系，并以前所未有的灵活性整合多模态数据。然而，这种潜力与现实之间存在显著差距。

### 目的

批判性评估基础模型在生物医学领域的现状，分析其核心能力和局限性；提供推理分类法，评估这些模型是否表现出真正的认知或仅模仿表面模式；讨论部署中的关键问题，如可信度、偏见和安全；呼吁更包容、严谨且临床相关的验证框架。

### 方法

通过批判性评估当前技术状态，分析炒作现象；提供从模拟顺序逻辑和空间理解到显式符号知识整合的推理分类法；讨论因果推断的重要性；剖析部署中的信任、偏见和安全问题。

### 主要发现

基础模型在生物医学领域的临床评估和部署面临重大挑战；需要超越统计相关性，追求因果推断；部署中存在算法偏见、数据偏见和隐私以及模型幻觉等关键问题；需要更包容、严谨且临床相关的验证框架。

### 结论

虽然自主AI医生的愿景仍然遥远，但当前现实是出现了将受益于临床实践的有力技术和辅助工具。基础模型在生物医学成像的未来不仅取决于规模，还取决于开发混合的、具有因果意识且可验证安全的系统，增强而非替代人类专业知识。

### 翻译

基础模型正在推动人工智能在不同领域（包括生物医学成像）的重大转变。这些模型旨在超越狭隘的模式识别，模拟复杂的临床推理，理解复杂的空间关系，并以前所未有的灵活性整合多模态数据。然而，这种潜力与现实之间存在显著差距，基础模型的临床评估和部署受到重大挑战的阻碍。在此，我们批判性评估当前技术状态，通过检查基础模型在生物医学领域的核心能力和局限性来分析炒作现象。我们还提供了推理分类法，范围从模拟顺序逻辑和空间理解到显式符号知识的整合，以评估这些模型是否表现出真正的认知或仅模仿表面模式。我们认为，一个关键的前沿领域在于超越统计相关性，追求因果推断，这对于构建理解因果关系和效果的稳健模型至关重要。此外，我们讨论了源于可信度、偏见和安全性的部署关键问题，剖析了算法偏见、数据偏见和隐私以及模型幻觉的挑战。我们还呼吁需要更包容、严谨且临床相关的验证框架，以确保其安全且合乎道德的应用。我们得出结论，虽然自主AI医生的愿景仍然遥远，但当前现实是出现了将受益于临床实践的有力技术和辅助工具。基础模型在生物医学成像的未来不仅取决于规模，还取决于开发混合的、具有因果意识且可验证安全的系统，增强而非替代人类专业知识。


### 论文摘要

Foundation models (FMs) are driving a prominent shift in artificial intelligence across different domains, including biomedical imaging. These models are designed to move beyond narrow pattern recognition towards emulating sophisticated clinical reasoning, understanding complex spatial relationships, and integrating multimodal data with unprecedented flexibility. However, a critical gap exists between this potential and the current reality, where the clinical evaluation and deployment of FMs are hampered by significant challenges. Herein, we critically assess the current state-of-the-art, analyzing hype by examining the core capabilities and limitations of FMs in the biomedical domain. We also provide a taxonomy of reasoning, ranging from emulated sequential logic and spatial understanding to the integration of explicit symbolic knowledge, to evaluate whether these models exhibit genuine cognition or merely mimic surface-level patterns. We argue that a critical frontier lies beyond statistical correlation, in the pursuit of causal inference, which is essential for building robust models that understand cause and effect. Furthermore, we discuss the paramount issues in deployment stemming from trustworthiness, bias, and safety, dissecting the challenges of algorithmic bias, data bias and privacy, and model hallucinations. We also draw attention to the need for more inclusive, rigorous, and clinically relevant validation frameworks to ensure their safe and ethical application. We conclude that while the vision of autonomous AI-doctors remains distant, the immediate reality is the emergence of powerful technology and assistive tools that would benefit clinical practice. The future of FMs in biomedical imaging hinges not on scale alone, but on developing hybrid, causally aware, and verifiably safe systems that augment, rather than replace, human expertise.

---

## 53. Next-Embedding Prediction Makes Strong Vision Learners

**论文链接:** [http://arxiv.org/abs/2512.16922v1](http://arxiv.org/abs/2512.16922v1)

**作者:** Sihan Xu, Ziqiao Ma, Wenhao Chai, Xuweiyi Chen, Weiyang Jin, Joyce Chai, Saining Xie, Stella X. Yu

**发布时间:** 2025-12-18

**备注:** Project Page: https://sihanxu.me/nepa

### GPT解析

### 总结

这篇论文提出了一种名为NEPA的自监督视觉学习方法，通过预测未来嵌入而非传统方法中的特征表示，实现了简单而有效的视觉模型预训练。

### 背景

受自然语言生成式预训练成功的启发，探索相同原则是否能应用于视觉领域。

### 目的

探索从学习表征到学习模型的转变，开发一种简单、可扩展的自监督视觉学习方法。

### 方法

提出Next-Embedding Predictive Autoregression (NEPA)，训练模型基于过去的patch嵌入预测未来的patch嵌入，使用因果掩码和stop gradient技术。仅使用next embedding prediction作为学习目标，无需像素重建、离散token、对比损失或任务特定头。

### 主要发现

简单的Transformer在ImageNet-1k上仅以next embedding prediction为学习目标进行预训练是有效的。NEPA在ImageNet-1K上达到83.8%和85.3%的top-1准确率，并能有效转移到ADE20K上的语义分割任务。

### 结论

生成式预训练从嵌入提供了一种简单、可扩展且可能独立于模态的视觉自监督学习替代方案。

### 翻译

受自然语言生成式预训练成功的启发，我们探索相同原则是否能产生强大的自监督视觉学习器。我们训练模型生成嵌入来直接执行预测任务，而不是输出特征供下游使用。这项工作探索了从学习表征到学习模型的转变。具体来说，模型学习使用因果掩码和stop gradient基于过去的patch嵌入预测未来的patch嵌入，我们称之为Next-Embedding Predictive Autoregression (NEPA)。我们证明，在ImageNet-1k上仅以next embedding prediction为唯一学习目标进行预训练的简单Transformer是有效的-不需要像素重建、离散token、对比损失或任务特定头。这种 formulation保留了架构简单性和可扩展性，不需要额外的设计复杂性。NEPA在多个任务上取得了良好的结果，使用ViT-B和ViT-L骨干网络在ImageNet-1K上微调后达到83.8%和85.3%的top-1准确率，并能有效转移到ADE20K上的语义分割。我们相信从嵌入进行生成式预训练为视觉自监督学习提供了一种简单、可扩展且可能独立于模态的替代方案。


### 论文摘要

Inspired by the success of generative pretraining in natural language, we ask whether the same principles can yield strong self-supervised visual learners. Instead of training models to output features for downstream use, we train them to generate embeddings to perform predictive tasks directly. This work explores such a shift from learning representations to learning models. Specifically, models learn to predict future patch embeddings conditioned on past ones, using causal masking and stop gradient, which we refer to as Next-Embedding Predictive Autoregression (NEPA). We demonstrate that a simple Transformer pretrained on ImageNet-1k with next embedding prediction as its sole learning objective is effective - no pixel reconstruction, discrete tokens, contrastive loss, or task-specific heads. This formulation retains architectural simplicity and scalability, without requiring additional design complexity. NEPA achieves strong results across tasks, attaining 83.8% and 85.3% top-1 accuracy on ImageNet-1K with ViT-B and ViT-L backbones after fine-tuning, and transferring effectively to semantic segmentation on ADE20K. We believe generative pretraining from embeddings provides a simple, scalable, and potentially modality-agnostic alternative to visual self-supervised learning.

---

## 54. InfoDCL: Informative Noise Enhanced Diffusion Based Contrastive Learning

**论文链接:** [http://arxiv.org/abs/2512.16576v1](http://arxiv.org/abs/2512.16576v1)

**作者:** Xufeng Liang, Zhida Qin, Chong Zhang, Tianyu Huang, Gangyi Ding

**发布时间:** 2025-12-18

### GPT解析

### 总结

本文提出了一种名为InfoDCL的新型基于扩散的对比学习框架，用于解决推荐系统中现有方法无法捕获足够语义信息的问题。

### 背景

对比学习在推荐系统中显示出良好的潜力，但现有方法通常通过随机扰动原始交互图构建更稀疏视图，由于不了解真实用户偏好和推荐数据的稀疏性，只能捕获不足的语义信息。

### 目的

解决现有方法在推荐系统中捕获语义信息不足的问题，提高推荐性能。

### 方法

提出InfoDCL框架，采用单步扩散过程将噪声与辅助语义信息集成，生成真实的用户偏好作为对比视图；构建协作训练目标策略将生成和偏好学习之间的干扰转化为协作；仅在推理阶段使用多层GCN融入高阶共现信息，同时保持训练效率。

### 主要发现

在五个真实世界数据集上的大量实验表明，InfoDCL显著优于最先进的方法。

### 结论

InfoDCL为提高推荐性能提供了有效的解决方案，并为在对比学习框架中应用扩散方法提供了新的范式。

### 翻译

对比学习在推荐系统中显示出良好的潜力。现有方法通常通过随机扰动原始交互图构建更稀疏的视图，因为它们不了解真实的用户偏好。由于推荐数据的稀疏性，这种范式只能捕获不足的语义信息。为解决这一问题，我们提出了InfoDCL，一种用于推荐的新型基于扩散的对比学习框架。我们不是注入随机采样的高斯噪声，而是采用单步扩散过程，将噪声与辅助语义信息集成以生成信号，并将它们输入到标准扩散过程中，生成真实的用户偏好作为对比视图。此外，基于对InfoDCL中生成和偏好学习之间相互影响的全面分析，我们构建了协作训练目标策略，将它们之间的干扰转化为相互协作。此外，我们仅在推理阶段使用多层GCN，以融入高阶共现信息，同时保持训练效率。在五个真实世界数据集上的大量实验表明，InfoDCL显著优于最先进的方法。我们的InfoDCL为提高推荐性能提供了有效的解决方案，并为在对比学习框架中应用扩散方法提出了新的范式。


### 论文摘要

Contrastive learning has demonstrated promising potential in recommender systems. Existing methods typically construct sparser views by randomly perturbing the original interaction graph, as they have no idea about the authentic user preferences. Owing to the sparse nature of recommendation data, this paradigm can only capture insufficient semantic information. To address the issue, we propose InfoDCL, a novel diffusion-based contrastive learning framework for recommendation. Rather than injecting randomly sampled Gaussian noise, we employ a single-step diffusion process that integrates noise with auxiliary semantic information to generate signals and feed them to the standard diffusion process to generate authentic user preferences as contrastive views. Besides, based on a comprehensive analysis of the mutual influence between generation and preference learning in InfoDCL, we build a collaborative training objective strategy to transform the interference between them into mutual collaboration. Additionally, we employ multiple GCN layers only during inference stage to incorporate higher-order co-occurrence information while maintaining training efficiency. Extensive experiments on five real-world datasets demonstrate that InfoDCL significantly outperforms state-of-the-art methods. Our InfoDCL offers an effective solution for enhancing recommendation performance and suggests a novel paradigm for applying diffusion method in contrastive learning frameworks.

---

## 55. BrepLLM: Native Boundary Representation Understanding with Large Language Models

**论文链接:** [http://arxiv.org/abs/2512.16413v1](http://arxiv.org/abs/2512.16413v1)

**作者:** Liyuan Deng, Hao Guo, Yunpeng Bai, Yongkang Dai, Huaxi Huang, Yilei Shi

**发布时间:** 2025-12-18

### GPT解析

### 总结

BrepLLM是一种新型框架，使大语言模型能够解析和推理原始3D边界表示数据，弥合结构化3D几何与自然语言之间的模态差距。

### 背景

当前基于token序列的大语言模型不适合直接处理包含复杂几何和拓扑信息的3D边界表示模型，存在结构化3D几何与自然语言之间的模态差距。

### 目的

提出BrepLLM框架，使大语言模型能够直接处理和理解3D边界表示数据，实现3D几何与自然语言之间的有效沟通。

### 方法

采用两阶段训练流程：第一阶段使用自适应UV采样策略将Brep转换为图表示，设计分层BrepEncoder提取特征并通过对比学习与文本嵌入对齐；第二阶段将预训练的BrepEncoder集成到LLM中，使用三阶段渐进式训练策略对节点token序列进行对齐，并构建了包含269,444个Brep-文本问答对的数据集。

### 主要发现

实验表明，BrepLLM在3D物体分类和字幕任务上取得了最先进的结果。

### 结论

BrepLLM成功地使大语言模型能够处理和理解3D边界表示数据，有效地连接了3D几何表示和自然语言理解。

### 翻译

当前基于token序列的大语言模型不太适合直接处理包含复杂几何和拓扑信息的3D边界表示模型。我们提出了BrepLLM，这是第一个使大语言模型能够解析和推理原始Brep数据的框架，弥合了结构化3D几何与自然语言之间的模态差距。BrepLLM采用两阶段训练流程：跨模态对齐预训练和多阶段LLM微调。在第一阶段，自适应UV采样策略将Brep转换为包含几何和拓扑信息的图表示。我们设计了一个分层BrepEncoder从几何(即面和边)和拓扑中提取特征，生成单个全局token和一系列节点token。然后通过对比学习将全局token与冻结的CLIP文本编码器(ViT-L/14)的文本嵌入对齐。在第二阶段，我们将预训练的BrepEncoder集成到LLM中。然后使用三阶段渐进式训练策略对其节点token序列进行对齐：(1)训练基于MLP的从Brep表示到2D的语义映射，利用2D-LLM先验。(2)对LLM进行微调。(3)设计查询专家混合(MQE)以增强几何多样性建模。我们还构建了Brep2Text数据集，包含269,444个Brep-文本问答对。实验表明，BrepLLM在3D物体分类和字幕任务上取得了最先进的结果。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文解决现有基于标记序列的大语言模型无法直接处理包含复杂几何和拓扑信息的3D边界表示（Brep）模型的问题。这一问题在工业设计中至关重要，因为CAD模型由Brep精确定义，直接理解Brep将推动工业模型设计和智能制造的重大进步，使AI系统能够真正理解CAD模型的几何结构和拓扑关系，而不仅仅是生成命令序列。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者认识到现有方法只能通过生成CAD命令序列间接处理Brep，无法捕获固有形状结构。他们借鉴了CLIP的跨模态对齐思想，将其应用于几何-语言对齐；参考了BLIP-2的Q-Former架构处理视觉-语言对齐；并受到PointLLM和ShapeLLM等3D-LLM工作的启发，但针对Brep数据进行了专门改进。作者采用两阶段训练管道，先进行跨模态对齐预训练，再进行多阶段LLM微调，以实现Brep与自然语言的深度融合。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是使大语言模型能够直接解析和推理原始Brep数据，弥合结构化3D几何和自然语言之间的模态差距。整体流程分为两个阶段：1）跨模态对齐预训练：使用自适应UV采样将Brep转换为图表示，通过分层BrepEncoder提取几何和拓扑特征，生成全局标记和节点标记序列，然后使用对比学习将全局标记与CLIP文本嵌入对齐；2）多阶段LLM微调：先训练Brep表示到2D的语义映射，再微调LLM，最后引入混合查询专家（MQE）增强几何多样性建模。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1）首个使大语言模型直接解析原始Brep模型的框架；2）自适应UV采样策略和分层BrepEncoder实现多粒度几何和拓扑特征表示；3）三阶段策略将2D-LLM先验适应于Brep领域，形成残差MQE；4）建立首个Brep为中心的语言理解任务的大规模基准（269,444个Brep-语言对）。相比之前工作，BrepLLM直接处理原始Brep数据而非命令序列，能够执行真正的几何和拓扑推理，同时使用分层编码器同时处理几何和拓扑信息，并通过MQE增强几何多样性建模。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'BrepLLM首次实现了大语言模型对CAD边界表示数据的直接理解和推理，通过创新的分层编码器和三阶段训练策略，弥合了结构化3D几何与自然语言之间的模态差距，为工业设计和智能制造提供了新的可能性。'}


### 论文摘要

Current token-sequence-based Large Language Models (LLMs) are not well-suited for directly processing 3D Boundary Representation (Brep) models that contain complex geometric and topological information. We propose BrepLLM, the first framework that enables LLMs to parse and reason over raw Brep data, bridging the modality gap between structured 3D geometry and natural language. BrepLLM employs a two-stage training pipeline: Cross-modal Alignment Pre-training and Multi-stage LLM Fine-tuning. In the first stage, an adaptive UV sampling strategy converts Breps into graphs representation with geometric and topological information. We then design a hierarchical BrepEncoder to extract features from geometry (i.e., faces and edges) and topology, producing both a single global token and a sequence of node tokens. Then we align the global token with text embeddings from a frozen CLIP text encoder (ViT-L/14) via contrastive learning. In the second stage, we integrate the pretrained BrepEncoder into an LLM. We then align its sequence of node tokens using a three-stage progressive training strategy: (1) training an MLP-based semantic mapping from Brep representation to 2D with 2D-LLM priors. (2) performing fine-tuning of the LLM. (3) designing a Mixture-of-Query Experts (MQE) to enhance geometric diversity modeling. We also construct Brep2Text, a dataset comprising 269,444 Brep-text question-answer pairs. Experiments show that BrepLLM achieves state-of-the-art (SOTA) results on 3D object classification and captioning tasks.

---

## 56. MACL: Multi-Label Adaptive Contrastive Learning Loss for Remote Sensing Image Retrieval

**论文链接:** [http://arxiv.org/abs/2512.16294v1](http://arxiv.org/abs/2512.16294v1)

**作者:** Amna Amir, Erchan Aptoula

**发布时间:** 2025-12-18

### GPT解析

### 总结

本文提出了一种名为多标签自适应对比学习(MACL)的方法，解决了多标签遥感图像检索中的语义重叠、标签分布不平衡和类间共现模式复杂等挑战。

### 背景

土地覆盖类别间的语义重叠、高度不平衡的标签分布以及复杂的类间共现模式构成了多标签遥感图像检索的重大挑战。

### 目的

开发一种能够有效处理语义不平衡问题的方法，实现常见和稀有类别之间的平衡表示学习。

### 方法

提出多标签自适应对比学习(MACL)，作为对比学习的扩展，整合了标签感知采样、频率敏感加权和动态温度缩放技术。

### 主要发现

在三个基准数据集(DLRSD、ML-AID和WHDLD)上的大量实验表明，MACL持续优于基于对比损失的基线方法，有效减轻了语义不平衡问题，并在大规模遥感档案中提供了更可靠的检索性能。

### 结论

MACL是一种有效的解决方案，能够应对多标签遥感图像检索中的挑战，作者将在论文接受后通过GitHub发布代码、预训练模型和评估脚本。

### 翻译

土地覆盖类别之间的语义重叠、高度不平衡的标签分布以及复杂的类间共现模式构成了多标签遥感图像检索的重大挑战。本文引入了多标签自适应对比学习(MACL)，作为对比学习的扩展来解决这些问题。它整合了标签感知采样、频率敏感加权和动态温度缩放，以实现常见和稀有类别之间的平衡表示学习。在三个基准数据集(DLRSD、ML-AID和WHDLD)上的大量实验表明，MACL持续优于基于对比损失的基线方法，有效减轻了语义不平衡问题，并在大规模遥感档案中提供了更可靠的检索性能。代码、预训练模型和评估脚本将在接受后发布于https://github.com/amna/MACL。


### 论文摘要

Semantic overlap among land-cover categories, highly imbalanced label distributions, and complex inter-class co-occurrence patterns constitute significant challenges for multi-label remote-sensing image retrieval. In this article, Multi-Label Adaptive Contrastive Learning (MACL) is introduced as an extension of contrastive learning to address them. It integrates label-aware sampling, frequency-sensitive weighting, and dynamic-temperature scaling to achieve balanced representation learning across both common and rare categories. Extensive experiments on three benchmark datasets (DLRSD, ML-AID, and WHDLD), show that MACL consistently outperforms contrastive-loss based baselines, effectively mitigating semantic imbalance and delivering more reliable retrieval performance in large-scale remote-sensing archives. Code, pretrained models, and evaluation scripts will be released at https://github.com/amna/MACL upon acceptance.

---

## 57. Interaction-via-Actions: Cattle Interaction Detection with Joint Learning of Action-Interaction Latent Space

**论文链接:** [http://arxiv.org/abs/2512.16133v1](http://arxiv.org/abs/2512.16133v1)

**作者:** Ren Nakagawa, Yang Yang, Risa Shinoda, Hiroaki Santo, Kenji Oyama, Fumio Okura, Takenao Ohkawa

**发布时间:** 2025-12-18

**备注:** Accepted to WACV 2026

### GPT解析

### 总结

本文介绍了一种从单张图像自动检测牛群行为交互的方法及其应用，这对智能畜牧管理（如发情检测）至关重要。

### 背景

人类行为交互检测已被积极研究，但牛的行为交互检测面临挑战，特别是缺乏包含交互行为的综合数据集，因为牛群交互是稀有事件。

### 目的

开发一种数据高效的交互检测方法，用于智能畜牧管理。

### 方法

提出CattleAct方法，将交互分解为个体牛的行为组合；从大规模牛行为数据集中学习动作潜在空间；使用对比学习微调预训练的潜在空间以嵌入稀有交互；构建动作和交互的统一潜在空间；开发整合视频和GPS输入的实用工作系统。

### 主要发现

在商业牧场上的实验表明，该方法相比基线实现了准确的交互检测。

### 结论

CattleAct方法有效解决了牛群交互检测的挑战，为智能畜牧管理提供了实用工具。

### 翻译

本文介绍了一种从单张图像自动检测牛群行为交互的方法及其应用，这对畜牧业的智能畜牧管理（如发情检测）至关重要。尽管人类行为交互检测已被积极研究，但牛交互检测存在一个非平凡的挑战，特别是缺乏包含交互的综合行为数据集，因为牛群交互是稀有事件。因此，我们提出了CattleAct，一种通过将交互分解为个体牛的行为组合来实现数据高效的交互检测方法。具体来说，我们首先从大规模牛行为数据集中学习动作潜在空间。然后，我们通过使用对比学习微调预训练的潜在空间来嵌入稀有交互，从而构建动作和交互的统一潜在空间。在提出的方法基础上，我们开发了一个整合视频和GPS输入的实用工作系统。在商业牧场上的实验表明，与基线相比，我们的方法实现了准确的交互检测。我们的实现可在https://github.com/rakawanegan/CattleAct获取。


### 论文摘要

This paper introduces a method and application for automatically detecting behavioral interactions between grazing cattle from a single image, which is essential for smart livestock management in the cattle industry, such as for detecting estrus. Although interaction detection for humans has been actively studied, a non-trivial challenge lies in cattle interaction detection, specifically the lack of a comprehensive behavioral dataset that includes interactions, as the interactions of grazing cattle are rare events. We, therefore, propose CattleAct, a data-efficient method for interaction detection by decomposing interactions into the combinations of actions by individual cattle. Specifically, we first learn an action latent space from a large-scale cattle action dataset. Then, we embed rare interactions via the fine-tuning of the pre-trained latent space using contrastive learning, thereby constructing a unified latent space of actions and interactions. On top of the proposed method, we develop a practical working system integrating video and GPS inputs. Experiments on a commercial-scale pasture demonstrate the accurate interaction detection achieved by our method compared to the baselines. Our implementation is available at https://github.com/rakawanegan/CattleAct.

---

## 58. From Minutes to Days: Scaling Intracranial Speech Decoding with Supervised Pretraining

**论文链接:** [http://arxiv.org/abs/2512.15830v1](http://arxiv.org/abs/2512.15830v1)

**作者:** Linnea Evanson, Mingfang, Zhang, Hubert Banville, Saarang Panchavati, Pierre Bourdillon, Jean-Rémi King

**发布时间:** 2025-12-17

**备注:** Linnea Evanson* and Mingfang (Lucy) Zhang* are joint first authors. Pierre Bourdillon** and Jean-Rémi King** are joint last authors

### GPT解析

### 总结

研究引入了一种利用长期临床监测数据训练语音解码模型的新框架，显著提高了解码性能，并发现大脑活动存在跨日变异性。

### 背景

传统的脑电活动解码语音研究依赖于在短暂且高度受控的实验中收集的有限神经记录数据，限制了模型的训练效果和泛化能力。

### 目的

开发能够利用长期临床监测数据训练语音解码模型的方法，提高解码性能，并探索大脑活动随时间的变异性。

### 方法

引入一个框架，利用患者临床监测期间长达一周的颅内和音频记录，将训练数据集大小增加了两个数量级以上；使用对比学习模型进行预训练，并分析学习到的表示。

### 主要发现

1. 使用预训练的对比学习模型显著优于仅使用传统实验数据训练的模型；2. 性能提升与数据集大小呈对数线性关系；3. 大脑活动虽然表示语音特征，但其全局结构在几天内会漂移；4. 需要明确考虑跨日变异性的模型。

### 结论

该方法为在真实生活和受控任务环境中解码和建模大脑表征提供了一条可扩展的途径，为脑机接口和神经科学研究提供了新的可能性。

### 翻译

从大脑活动中解码语音通常依赖于在短暂且高度受控的实验中收集的有限神经记录。在这里，我们引入了一个框架，利用接受临床监测的患者长达一周的颅内和音频记录，有效地将训练数据集的大小增加了两个数量级以上。通过这种预训练，我们的对比学习模型明显优于仅使用传统实验数据训练的模型，性能提升与数据集大小呈对数线性关系。对学习到的表示的分析表明，虽然大脑活动表示语音特征，但其全局结构在几天内会漂移，突显了需要明确考虑跨日变异性的模型。总体而言，我们的方法为在真实生活和受控任务环境中解码和建模大脑表征开辟了一条可扩展的途径。


### 论文摘要

Decoding speech from brain activity has typically relied on limited neural recordings collected during short and highly controlled experiments. Here, we introduce a framework to leverage week-long intracranial and audio recordings from patients undergoing clinical monitoring, effectively increasing the training dataset size by over two orders of magnitude. With this pretraining, our contrastive learning model substantially outperforms models trained solely on classic experimental data, with gains that scale log-linearly with dataset size. Analysis of the learned representations reveals that, while brain activity represents speech features, its global structure largely drifts across days, highlighting the need for models that explicitly account for cross-day variability. Overall, our approach opens a scalable path toward decoding and modeling brain representations in both real-life and controlled task settings.

---

## 59. Dual-coding contrastive learning based on ConvNeXt and ViT models for morphological classification of galaxies in COSMOS-Web

**论文链接:** [http://arxiv.org/abs/2512.15129v2](http://arxiv.org/abs/2512.15129v2)

**作者:** Shiwei Zhu, Guanwen Fang, Chichun Zhou, Jie Song, Zesen Lin, Yao Dai, Xu Kong

**发布时间:** 2025-12-17

**备注:** Published in APJS

### GPT解析

### 总结

本研究提出了一种基于对比学习的自监督方法来升级USmorph框架中的无监督机器学习部分，提高了星系形态分类的效率和准确性。

### 背景

作者之前提出了一个名为USmorph的机器学习框架用于星系形态分类，本研究是对该框架中无监督机器学习部分的升级。

### 目的

提出一种自监督方法来升级USmorph框架中的无监督机器学习部分，以提高特征提取步骤的效率。

### 方法

升级的UML方法包括：1)使用卷积自编码器去噪和自适应极坐标变换增强旋转不变性；2)采用基于ConvNeXt和ViT的预训练双编码器卷积神经网络进行图像编码，并用对比学习降低特征维度；3)使用基于Bagging的聚类模型对星系进行分组。应用于红移范围0.5 < z < 6.0的COSMOS-Web场域星系。

### 主要发现

改进的UML方法成功分类了73%的星系，剩余27%使用GoogleNet算法分类。分类结果与其他星系形态参数比较显示与星系演化有良好一致性。

### 结论

由于该算法具有更高的效率，非常适合应用于未来的中国空间望远镜任务。

### 翻译

在我们之前的工作中，我们提出了一个名为USmorph的机器学习框架用于高效分类星系形态。在本研究中，我们提出了一种称为对比学习的自监督方法来升级USmorph框架中的无监督机器学习部分，旨在提高该步骤中特征提取的效率。升级的UML方法主要包括以下三个方面。(1)我们采用卷积自编码器对星系图像去噪，并使用自适应极坐标变换增强模型的旋转不变性。(2)使用基于ConvNeXt和ViT的预训练双编码器卷积神经网络对图像数据进行编码，然后应用对比学习降低特征维度。(3)我们采用基于Bagging的聚类模型将具有相似特征的星系聚集成不同的组。通过仔细划分红移区间，我们将该模型应用于红移范围0.5 < z < 6.0的COSMOS-Web场域星系的光学图像。与之前的算法相比，改进的UML方法成功分类了73%的星系。使用GoogleNet算法，我们对剩余27%的星系进行形态分类。为了验证更新算法的可靠性，我们将分类结果与其他星系形态参数进行了比较，发现与星系演化有良好的一致性。得益于其更高的效率，该更新算法非常适合应用于未来的中国空间望远镜任务。


### 论文摘要

In our previous works, we proposed a machine learning framework named \texttt{USmorph} for efficiently classifying galaxy morphology. In this study, we propose a self-supervised method called contrastive learning to upgrade the unsupervised machine learning (UML) part of the \texttt{USmorph} framework, aiming to improve the efficiency of feature extraction in this step. The upgraded UML method primarily consists of the following three aspects. (1) We employ a Convolutional Autoencoder to denoise galaxy images and the Adaptive Polar Coordinate Transformation to enhance the model's rotational invariance. (2) A pre-trained dual-encoder convolutional neural network based on ConvNeXt and ViT is used to encode the image data, while contrastive learning is then applied to reduce the dimension of the features. (3) We adopt a Bagging-based clustering model to cluster galaxies with similar features into distinct groups. By carefully dividing the redshift bins, we apply this model to the rest-frame optical images of galaxies in the COSMOS-Web field within the redshift range of $0.5 < z < 6.0$. Compared to the previous algorithm, the improved UML method successfully classifies 73\% galaxies. Using the GoogleNet algorithm, we classify the morphology of the remaining 27\% galaxies. To validate the reliability of our updated algorithm, we compared our classification results with other galaxy morphological parameters and found a good consistency with galaxy evolution. Benefiting from its higher efficiency, this updated algorithm is well-suited for application in future China Space Station Telescope missions.

---

## 60. 论文ID: 2512.15793v1

**错误:** 无法读取论文信息 - [Errno 2] No such file or directory: 'storage/result/2512.15793v1.json'

---

## 61. SARMAE: Masked Autoencoder for SAR Representation Learning

**论文链接:** [http://arxiv.org/abs/2512.16635v1](http://arxiv.org/abs/2512.16635v1)

**作者:** Danxu Liu, Di Wang, Hebaixu Wang, Haoyang Chen, Wentao Jiang, Yilin Cheng, Haonan Guo, Wei Cui, Jing Zhang

**发布时间:** 2025-12-18

**备注:** Code and models will be available at https://github.com/MiliLab/SARMAE

### GPT解析

### 总结

该研究提出了SARMAE，一种用于自监督SAR表征学习的噪声感知掩码自编码器，通过构建百万级SAR数据集和设计斑点感知表征增强方法，解决了SAR图像处理中的数据稀缺和斑点噪声挑战。

### 背景

SAR图像在全天候、日夜遥感应用中起关键作用，但现有面向SAR的深度学习受限于数据稀缺，且SAR图像中固有的斑点噪声阻碍了细粒度语义表征学习。

### 目的

解决SAR图像处理中的数据稀缺问题，应对斑点噪声挑战，实现噪声感知和鲁棒的表征学习。

### 方法

1) 构建SAR-1M，首个百万级SAR数据集并配有配对光学图像；2) 设计斑点感知表征增强(SARE)，将SAR特定斑点噪声注入掩码自编码器；3) 引入语义锚定表征约束(SARC)，利用配对光学先验对齐SAR特征并确保语义一致性。

### 主要发现

在多个SAR数据集上的实验表明，SARMAE在分类、检测和分割任务上达到最先进性能，代码和模型已开源。

### 结论

SARMAE有效解决了SAR图像处理中的数据稀缺和斑点噪声问题，通过自监督学习实现了高性能的SAR表征学习。

### 翻译

合成孔径雷达(SAR)图像在全天候、日夜遥感应用中起着关键作用。然而，现有的面向SAR的深度学习受限于数据稀缺，而SAR图像中基于物理的斑点噪声进一步阻碍了细粒度语义表征学习。为应对这些挑战，我们提出了SARMAE，一种用于自监督SAR表征学习的噪声感知掩码自编码器。具体来说，我们构建了SAR-1M，首个百万级SAR数据集，并配有额外的配对光学图像，以实现大规模预训练。基于此，我们设计了斑点感知表征增强(SARE)，将SAR特定的斑点噪声注入掩码自编码器，以促进噪声感知和鲁棒的表征学习。此外，我们引入了语义锚定表征约束(SARC)，利用配对的光学先验对齐SAR特征并确保语义一致性。在多个SAR数据集上的广泛实验表明，SARMAE在分类、检测和分割任务上实现了最先进的性能。代码和模型将在https://github.com/MiliLab/SARMAE上提供。


### 论文摘要

Synthetic Aperture Radar (SAR) imagery plays a critical role in all-weather, day-and-night remote sensing applications. However, existing SAR-oriented deep learning is constrained by data scarcity, while the physically grounded speckle noise in SAR imagery further hampers fine-grained semantic representation learning. To address these challenges, we propose SARMAE, a Noise-Aware Masked Autoencoder for self-supervised SAR representation learning. Specifically, we construct SAR-1M, the first million-scale SAR dataset, with additional paired optical images, to enable large-scale pre-training. Building upon this, we design Speckle-Aware Representation Enhancement (SARE), which injects SAR-specific speckle noise into masked autoencoders to facilitate noise-aware and robust representation learning. Furthermore, we introduce Semantic Anchor Representation Constraint (SARC), which leverages paired optical priors to align SAR features and ensure semantic consistency. Extensive experiments across multiple SAR datasets demonstrate that SARMAE achieves state-of-the-art performance on classification, detection, and segmentation tasks. Code and models will be available at https://github.com/MiliLab/SARMAE.

---

## 62. Sharpness-aware Second-order Latent Factor Model for High-dimensional and Incomplete Data

**论文链接:** [http://arxiv.org/abs/2512.16277v1](http://arxiv.org/abs/2512.16277v1)

**作者:** Jialiang Wang, Xueyan Bao, Hao Wu

**发布时间:** 2025-12-18

### GPT解析

### 总结

本文提出了一种锐度感知的二阶潜在因子(SSLF)模型，通过Hessian-向量积获取二阶信息并将锐度项注入到曲率中，解决了传统SLF模型优化困难的问题，在多个工业数据集上表现出优于最先进基线的性能。

### 背景

二阶潜在因子(SLF)模型作为一种低秩表示学习方法，能够从高维且不完整的数据中提取节点间交互模式，但其优化过程因双线性和非凸特性而异常困难。

### 目的

解决SLF模型优化困难的问题，提高表示学习模型的泛化能力。

### 方法

提出锐度感知的SLF(SSLF)模型，包含两个关键思想：(1)通过Hessian-向量积获取二阶信息；(2)通过设计的Hessian-向量积将锐度项注入到曲率(Hessian)中。

### 主要发现

在多个工业数据集上的实验表明，所提出的SSLF模型性能始终优于现有的最先进基线方法。

### 结论

SSLF模型通过结合锐度感知最小化技术与二阶潜在因子模型，有效解决了传统SLF模型的优化难题，提高了表示学习效果。

### 翻译

二阶潜在因子模型是一种低秩表示学习方法，已证明能够从高维且不完整的数据中提取节点间交互模式。然而，由于其双线性和非凸特性，其优化过程异常困难。锐度感知最小化最近被提出用于在最小化非凸目标时寻找平坦的局部最小值，从而提高表示学习模型的泛化能力。为了应对这一挑战，我们提出了一个锐度感知的二阶潜在因子模型。该模型体现了两个关键思想：(1)通过Hessian-向量积获取二阶信息；(2)通过设计的Hessian-向量积将锐度项注入到曲率中。在多个工业数据集上的实验表明，所提出的模型性能始终优于最先进的基线方法。


### 论文摘要

Second-order Latent Factor (SLF) model, a class of low-rank representation learning methods, has proven effective at extracting node-to-node interaction patterns from High-dimensional and Incomplete (HDI) data. However, its optimization is notoriously difficult due to its bilinear and non-convex nature. Sharpness-aware Minimization (SAM) has recently proposed to find flat local minima when minimizing non-convex objectives, thereby improving the generalization of representation-learning models. To address this challenge, we propose a Sharpness-aware SLF (SSLF) model. SSLF embodies two key ideas: (1) acquiring second-order information via Hessian-vector products; and (2) injecting a sharpness term into the curvature (Hessian) through the designed Hessian-vector products. Experiments on multiple industrial datasets demonstrate that the proposed model consistently outperforms state-of-the-art baselines.

---

## 63. Domain-Agnostic Causal-Aware Audio Transformer for Infant Cry Classification

**论文链接:** [http://arxiv.org/abs/2512.16271v1](http://arxiv.org/abs/2512.16271v1)

**作者:** Geofrey Owino, Bernard Shibwabo Kasamani, Ahmed M. Abdelmoniem, Edem Wornyo

**发布时间:** 2025-12-18

**DOI:** 10.1109/IC2IE67206.2025.11283358

**备注:** This paper has been published in the IEEE proceedings of the 8th International Conference of Computer and Informatics Engineering (IC2IE)

### GPT解析

### 总结

本文提出了DACH-TIC模型，一种领域无关因果感知分层音频Transformer，用于稳健的婴儿哭声分类，通过集成因果注意力和多任务学习等方法，在准确率和泛化能力上优于现有方法。

### 背景

准确且可解释的婴儿哭声分类对新生儿早期distress检测和临床决策支持至关重要，但现有深度学习方法易受噪声、伪线索和环境变化的影响。

### 目的

开发一种稳健的婴儿哭声分类模型，能够抵抗环境变化和噪声干扰，同时保持高准确性和可解释性。

### 方法

DACH-TIC模型整合了因果注意力机制、分层表示学习、多任务监督和对抗性域泛化，使用结构化Transformer主干，结合局部令牌级和全局语义编码器，并通过因果注意力掩码和受控扰动训练来近似反事实声学变化。

### 主要发现

DACH-TIC在Baby Chillanto和Donate-a-Cry数据集上表现优异，准确率提高2.6%，宏F1分数提高2.2点，同时增强了因果保真度；模型能有效泛化到未见过的声学环境，域性能差距仅为2.4%。

### 结论

DACH-TIC模型适合应用于现实世界的新生儿声学监测系统，能够提供准确且可解释的婴儿哭声分类结果。

### 翻译

准确且可解释的婴儿哭声音韵学分类对新生儿早期distress检测和临床决策支持至关重要。然而，许多现有的深度学习方法依赖于相关性驱动的声学表示，这使得它们容易受到噪声、伪线索和不同录制环境域转移的影响。我们提出了DACH-TIC，一种领域无关因果感知分层音频Transformer，用于稳健的婴儿哭声分类。该模型在一个统一框架内集成了因果注意力、分层表示学习、多任务监督和对抗性域泛化。DACH-TIC采用具有局部令牌级和全局语义编码器的结构化Transformer主干，通过因果注意力掩码和受控扰动训练来近似反事实声学变化。领域对抗性目标促进环境不变的表示，而多任务学习联合优化哭声类型识别、痛苦强度估计和因果相关性预测。该模型在Baby Chillanto和Donate-a-Cry数据集上进行了评估，并使用ESC-50环境噪声叠加进行域增强。实验结果表明，DACH-TIC优于最先进的基线方法，包括HTS-AT和SE-ResNet Transformer，在准确率上提高了2.6%，在宏F1分数上提高了2.2点，同时增强了因果保真度。模型能有效泛化到未见过的声学环境，域性能差距仅为2.4%，证明了其适用于现实世界的新生儿声学监测系统。


### 论文摘要

Accurate and interpretable classification of infant cry paralinguistics is essential for early detection of neonatal distress and clinical decision support. However, many existing deep learning methods rely on correlation-driven acoustic representations, which makes them vulnerable to noise, spurious cues, and domain shifts across recording environments. We propose DACH-TIC, a Domain-Agnostic Causal-Aware Hierarchical Audio Transformer for robust infant cry classification. The model integrates causal attention, hierarchical representation learning, multi-task supervision, and adversarial domain generalization within a unified framework.   DACH-TIC employs a structured transformer backbone with local token-level and global semantic encoders, augmented by causal attention masking and controlled perturbation training to approximate counterfactual acoustic variations. A domain-adversarial objective promotes environment-invariant representations, while multi-task learning jointly optimizes cry type recognition, distress intensity estimation, and causal relevance prediction. The model is evaluated on the Baby Chillanto and Donate-a-Cry datasets, with ESC-50 environmental noise overlays for domain augmentation.   Experimental results show that DACH-TIC outperforms state-of-the-art baselines, including HTS-AT and SE-ResNet Transformer, achieving improvements of 2.6 percent in accuracy and 2.2 points in macro-F1 score, alongside enhanced causal fidelity. The model generalizes effectively to unseen acoustic environments, with a domain performance gap of only 2.4 percent, demonstrating its suitability for real-world neonatal acoustic monitoring systems.

---

## 64. CauSTream: Causal Spatio-Temporal Representation Learning for Streamflow Forecasting

**论文链接:** [http://arxiv.org/abs/2512.16046v1](http://arxiv.org/abs/2512.16046v1)

**作者:** Shu Wan, Reepal Shah, John Sabo, Huan Liu, K. Selçuk Candan

**发布时间:** 2025-12-18

**备注:** Accepted by IEEE Big Data 2025

### GPT解析

### 总结

提出CauStream框架，联合学习径流因果图和路由图，实现可解释且泛化能力强的径流预测，在美国三个主要流域的实验中优于现有方法。

### 背景

径流预测对水资源管理和风险缓解至关重要。现有深度学习模型忽略物理过程，限制可解释性；因果学习方法虽整合领域知识，但依赖固定因果图无法适应数据变化。

### 目的

开发CauStream框架，解决现有方法中固定因果图无法适应数据的问题，提高模型可解释性和泛化能力。

### 方法

CauStream联合学习(i)气象强迫因素间的径流因果图，和(ii)捕捉站点间动态依赖关系的路由图；在非参数条件下建立因果结构可识别性条件；在美国三个主要流域的三个预测时间跨度上评估。

### 主要发现

模型持续优于最先进方法，长期预测窗口上性能差距扩大，表明泛化能力强；学习到的因果图与领域知识高度一致，提供流域动力学可解释见解。

### 结论

CauStream为因果时空建模提供有原则基础，可扩展到广泛的科学和环境应用。

### 翻译

径流预测对水资源管理和风险缓解至关重要。虽然深度学习模型已经取得了强大的预测性能，但它们往往忽略了潜在的物理过程，限制了可解释性和泛化能力。最近的因果学习方法通过整合领域知识解决了这些问题，但它们通常依赖于固定的因果图，无法适应数据变化。我们提出了CauStream，一个用于因果时空径流预测的统一框架。CauStream联合学习(i)气象强迫因素之间的径流因果图，和(ii)捕捉站点间动态依赖关系的路由图。我们进一步在非参数条件下建立了这些因果结构的可识别性条件。我们在美国三个主要流域的三个预测时间跨度上评估了CauStream。该模型持续优于先前最先进的方法，性能差距在更长的预测窗口上扩大，表明对未见条件的泛化能力更强。除了预测功能外，CauStream还学习捕捉水文因素和站点之间关系的因果图。推断出的结构与已建立的领域知识高度一致，为流域动力学提供了可解释的见解。CauStream为因果时空建模提供了有原则的基础，有潜力扩展到广泛的科学和环境应用中。


### 论文摘要

Streamflow forecasting is crucial for water resource management and risk mitigation. While deep learning models have achieved strong predictive performance, they often overlook underlying physical processes, limiting interpretability and generalization. Recent causal learning approaches address these issues by integrating domain knowledge, yet they typically rely on fixed causal graphs that fail to adapt to data. We propose CauStream, a unified framework for causal spatiotemporal streamflow forecasting. CauSTream jointly learns (i) a runoff causal graph among meteorological forcings and (ii) a routing graph capturing dynamic dependencies across stations. We further establish identifiability conditions for these causal structures under a nonparametric setting. We evaluate CauSTream on three major U.S. river basins across three forecasting horizons. The model consistently outperforms prior state-of-the-art methods, with performance gaps widening at longer forecast windows, indicating stronger generalization to unseen conditions. Beyond forecasting, CauSTream also learns causal graphs that capture relationships among hydrological factors and stations. The inferred structures align closely with established domain knowledge, offering interpretable insights into watershed dynamics. CauSTream offers a principled foundation for causal spatiotemporal modeling, with the potential to extend to a wide range of scientific and environmental applications.

---

## 65. In-Context Semi-Supervised Learning

**论文链接:** [http://arxiv.org/abs/2512.15934v1](http://arxiv.org/abs/2512.15934v1)

**作者:** Jiashuo Fan, Paul Rosu, Aaron T. Wang, Michael Li, Lawrence Carin, Xiang Cheng

**发布时间:** 2025-12-17

### GPT解析

### 总结

本文研究了Transformer在上下文半监督学习中的能力，展示了Transformer如何利用未标记上下文学习鲁棒表示，从而在标签稀少的情况下提高预测性能。

### 背景

近期对Transformer在上下文学习能力的研究兴趣浓厚，但大多数理论研究集中在有明确标签对的有监督设置上。

### 目的

引入并研究上下文半监督学习，探索Transformer如何利用未标记上下文学习表示。

### 方法

研究包含少量标记示例和许多未标记点的上下文半监督学习场景，分析Transformer如何利用未标记上下文学习鲁棒且依赖于上下文的表示。

### 主要发现

Transformer能够利用未标记上下文学习鲁棒的上下文相关表示，实现准确预测，并在低标签情况下显著提高性能。

### 结论

这些发现为Transformer如何在ICL框架内利用未标记上下文进行表示学习提供了基础性见解。

### 翻译

近期对理解Transformer在上下文学习能力方面有显著兴趣，但大多数理论聚焦于具有明确标签对的有监督设置。在实践中，即使标签稀少或不存在，Transformer通常表现良好，这表明未标记的上下文演示中存在重要结构。我们引入并研究了上下文半监督学习，其中一小部分标记示例伴随着许多未标记点，并证明Transformer可以利用未标记上下文学习鲁棒且依赖于上下文的表示。这种表示能够实现准确预测，并在低标签情况下显著提高性能，为Transformer如何在ICL框架内利用未标记上下文进行表示学习提供了基础性见解。


### 论文摘要

There has been significant recent interest in understanding the capacity of Transformers for in-context learning (ICL), yet most theory focuses on supervised settings with explicitly labeled pairs. In practice, Transformers often perform well even when labels are sparse or absent, suggesting crucial structure within unlabeled contextual demonstrations. We introduce and study in-context semi-supervised learning (IC-SSL), where a small set of labeled examples is accompanied by many unlabeled points, and show that Transformers can leverage the unlabeled context to learn a robust, context-dependent representation. This representation enables accurate predictions and markedly improves performance in low-label regimes, offering foundational insights into how Transformers exploit unlabeled context for representation learning within the ICL framework.

---

## 66. 论文ID: 2512.16909v1

**错误:** 无法读取论文信息 - [Errno 2] No such file or directory: 'storage/result/2512.16909v1.json'

---

## 67. N3D-VLM: Native 3D Grounding Enables Accurate Spatial Reasoning in Vision-Language Models

**论文链接:** [http://arxiv.org/abs/2512.16561v1](http://arxiv.org/abs/2512.16561v1)

**作者:** Yuxin Wang, Lei Ke, Boqiang Zhang, Tianyuan Qu, Hanxun Yu, Zhenpeng Huang, Meng Yu, Dan Xu, Dong Yu

**发布时间:** 2025-12-18

**备注:** Project Page: https://n3d-vlm.github.io

### GPT解析

### 总结

本文提出了N3D-VLM框架，通过原生3D物体感知和3D感知视觉推理的集成，解决了多模态模型在3D场景理解方面的局限性，实现了精确的3D定位和可解释的空间理解。

### 背景

当前多模态模型能够基于2D图像回答问题，但缺乏内在的3D物体感知能力，限制了它们对3D场景中空间关系和深度线索的理解能力。

### 目的

提出N3D-VLM，一个新颖的统一框架，无缝集成原生3D物体感知和3D感知视觉推理，实现精确的3D定位和可解释的空间理解。

### 方法

赋予模型原生3D物体感知能力，使其能直接根据文本描述在3D空间中定位物体；开发可扩展的数据构建管道，利用深度估计将大规模2D标注提升到3D空间，生成空间问答数据集促进3D物体定位和3D空间推理的联合训练。

### 主要发现

实验结果表明，该统一框架不仅在3D定位任务上取得了最先进的性能，而且在视觉语言模型的3D空间推理中也持续超越现有方法。

### 结论

N3D-VLM通过原生3D物体感知和3D感知视觉推理的集成，有效解决了多模态模型在3D场景理解方面的局限性，显著提升了3D物体定位和空间推理能力。

### 翻译

虽然当前的多模态模型可以基于2D图像回答问题，但它们缺乏内在的3D物体感知能力，限制了它们对3D场景中空间关系和深度线索的理解能力。在这项工作中，我们提出了N3D-VLM，一个新颖的统一框架，无缝集成原生3D物体感知与3D感知视觉推理，实现精确的3D定位和可解释的空间理解。与直接从RGB/RGB-D输入预测答案的传统端到端模型不同，我们的方法赋予模型原生3D物体感知能力，使其能够直接根据文本描述在3D空间中定位物体。基于准确的3D物体定位，模型进一步在3D中进行显式推理，实现更可解释和结构化的空间理解。为支持这些能力的稳健训练，我们开发了一个可扩展的数据构建管道，利用深度估计将大规模2D标注提升到3D空间，显著增加了3D物体定位数据的多样性和覆盖率，生成的数据量比现有最大的单图像3D检测数据集大六倍以上。此外，该管道生成针对3D中思维链(CoT)推理的空间问答数据集，促进3D物体定位和3D空间推理的联合训练。实验结果表明，我们的统一框架不仅在3D定位任务上取得了最先进的性能，而且在视觉语言模型的3D空间推理中也持续超越现有方法。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决当前多模态模型缺乏内在3D物体感知能力的问题，限制了它们对3D场景中空间关系和深度线索的理解。这个问题在现实中很重要，因为真实世界的应用往往需要深入理解3D结构和空间关系，而有效的3D空间推理需要准确的物体级3D感知能力。没有这种能力，模型很难推断空间配置或推理物理环境。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者将3D空间理解分解为两个核心能力：3D物体定位和随后的3D空间推理。他们借鉴了现有工作，如集成外部感知模型获取物体信息、假设预定义空间信息、以及在点云中定位物体等方法，但这些方法各有局限。作者的创新在于设计了一个统一的视觉语言模型，它具有原生3D物体感知能力，能够准确定位物体并捕获深度线索，然后基于这些结构化的3D表示进行空间推理。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是原生3D定位与基于定位的空间推理相结合的统一框架。整体实现流程包括：1) 3D数据构建，通过深度估计将2D注释提升到3D空间，生成大规模3D检测和空间推理数据；2) 3D感知视觉编码，将RGB图像和深度图作为输入，通过深度估计和位置编码融合空间信息；3) 两阶段训练，先训练3D物体定位，再混合训练空间推理；4) 支持两种推理模式，直接回答空间问题或先进行3D定位再回答后续问题。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1) 统一的视觉语言模型架构，集成3D检测、定位和推理；2) 大规模3D数据构建管道，将2D注释提升到3D空间；3) 空间推理基准，包含显式推理过程；4) 3D感知视觉编码，注入明确深度线索。相比之前工作，该方法不依赖外部模块或预定义信息，不局限于狭窄场景，能处理多样化环境和广泛物体类别，支持显式空间推理而不仅是物体检测，能输出完整3D边界框而非仅位置或方向。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'N3D-VLM通过原生3D物体定位和显式3D感知推理的统一框架，显著提升了视觉语言模型在3D空间理解方面的能力，同时通过创新的数据构建方法解决了3D训练数据稀缺的问题。'}


### 论文摘要

While current multimodal models can answer questions based on 2D images, they lack intrinsic 3D object perception, limiting their ability to comprehend spatial relationships and depth cues in 3D scenes. In this work, we propose N3D-VLM, a novel unified framework that seamlessly integrates native 3D object perception with 3D-aware visual reasoning, enabling both precise 3D grounding and interpretable spatial understanding. Unlike conventional end-to-end models that directly predict answers from RGB/RGB-D inputs, our approach equips the model with native 3D object perception capabilities, enabling it to directly localize objects in 3D space based on textual descriptions. Building upon accurate 3D object localization, the model further performs explicit reasoning in 3D, achieving more interpretable and structured spatial understanding. To support robust training for these capabilities, we develop a scalable data construction pipeline that leverages depth estimation to lift large-scale 2D annotations into 3D space, significantly increasing the diversity and coverage for 3D object grounding data, yielding over six times larger than the largest existing single-image 3D detection dataset. Moreover, the pipeline generates spatial question-answering datasets that target chain-of-thought (CoT) reasoning in 3D, facilitating joint training for both 3D object localization and 3D spatial reasoning. Experimental results demonstrate that our unified framework not only achieves state-of-the-art performance on 3D grounding tasks, but also consistently surpasses existing methods in 3D spatial reasoning in vision-language model.

---

## 68. Privacy-Aware Sharing of Raw Spatial Sensor Data for Cooperative Perception

**论文链接:** [http://arxiv.org/abs/2512.16265v1](http://arxiv.org/abs/2512.16265v1)

**作者:** Bangya Liu, Chengpo Yan, Chenghao Jiang, Suman Banerjee, Akarsh Prabhakara

**发布时间:** 2025-12-18

### GPT解析

### 总结

本文探讨了车辆协同感知中的隐私问题，并提出了解决方案

### 背景

车辆间的协同感知有望提供强大可靠的场景理解能力，实验系统研究正在建立测试平台共享原始空间传感器数据

### 目的

解决原始数据共享带来的隐私问题，推动利益相关者采用基于原始数据的协同感知

### 方法

提出SHARP研究框架，旨在最小化隐私泄露

### 主要发现

原始数据共享会引发新的隐私问题，阻碍利益相关者参与

### 结论

讨论了实现SHARP框架所需的跨领域合作和研究方向

### 翻译

车辆间的协同感知有望提供强大可靠的场景理解。最近，我们正在见证实验系统研究建立测试平台，用于共享原始空间传感器数据进行协同感知。虽然这种方法在准确性方面有显著提高，并且是自然的发展方向，但我们有必要考虑这种方法最终被汽车制造商采用时可能存在的问题。在本文中，我们首先指出，新的隐私问题会出现，这会阻碍利益相关者分享原始传感器数据。接下来，我们提出了SHARP研究框架，旨在最小化隐私泄露，并推动利益相关者朝着基于原始数据的协同感知这一宏伟目标前进。最后，我们讨论了网络系统、移动计算、感知研究人员、行业和政府在实现我们提出的框架时需要考虑的开放性问题。


### 论文摘要

Cooperative perception between vehicles is poised to offer robust and reliable scene understanding. Recently, we are witnessing experimental systems research building testbeds that share raw spatial sensor data for cooperative perception. While there has been a marked improvement in accuracies and is the natural way forward, we take a moment to consider the problems with such an approach for eventual adoption by automakers. In this paper, we first argue that new forms of privacy concerns arise and discourage stakeholders to share raw sensor data. Next, we present SHARP, a research framework to minimize privacy leakage and drive stakeholders towards the ambitious goal of raw data based cooperative perception. Finally, we discuss open questions for networked systems, mobile computing, perception researchers, industry and government in realizing our proposed framework.

---

## 69. Scaling Spatial Reasoning in MLLMs through Programmatic Data Synthesis

**论文链接:** [http://arxiv.org/abs/2512.16237v1](http://arxiv.org/abs/2512.16237v1)

**作者:** Zhi Helu, Huang Jingjing, Xu Wang, Xu Yangbin, Zhang Wanyue, Jiang Baoyang, Deng Shirui, Zhu Liang, Li Fangfang, Zhao Tiejun, Lin Yankai, Yao Yuan

**发布时间:** 2025-12-18

### GPT解析

### 总结

SPRITE是一种新型框架，通过将真实值生成视为代码生成任务，利用模拟器和大型语言模型合成高质量、多样化的空间推理数据。该方法克服了传统方法的困境，提供了可扩展、语言多样且计算精确的数据集，实验证明能有效提升视觉语言模型的空间推理能力。

### 背景

身体智能是人工智能的重大挑战，当前模型的空间理解和推理能力有限。现有的通过增强视觉语言模型来解决这一问题的努力面临困境：基于模板的数据集可扩展但结构僵化，而手工注释语言多样但不可扩展且计算上不精确。

### 目的

引入SPRITE框架，克服现有方法的困境，利用模拟器和大型模型可编程地合成可扩展、多样且高质量的空间推理数据。

### 方法

SPRITE的核心创新是将真实值生成重新定义为代码生成任务。使用大型语言模型将复杂的空间问题编译成可执行程序，然后与从模拟器中提取的高精度场景元信息进行验证。确保真实值在计算上精确且可验证，同时大型语言模型的生成能力提供了广泛的语言多样性。利用此流程整理了包含3个模拟器、11k+场景和300k+图像/视频指令微调对的数据集。

### 主要发现

在SPRITE数据上训练的视觉语言模型在多个空间基准测试上取得了显著的性能提升，优于同等大小的其他开源数据集。可扩展性分析证实了假设：克服传统模板方法的低多样性性质对于构建强大、可泛化的空间智能至关重要。

### 结论

将SPRITE框架代码和完整的300k+数据集公开，以促进未来空间智能研究。

### 翻译

身体智能是人工智能的重大挑战，从根本上受到当前模型有限的空间理解和推理能力的制约。通过增强视觉语言模型来解决这一问题的现有努力陷入两难：基于模板的数据集可扩展但结构僵化，而手工注释语言多样但不可扩展，关键的是，计算上不精确。我们引入了SPRITE，一个新型框架，通过利用模拟器和大型模型可编程地合成可扩展、多样且高质量的空间推理数据，从而克服这一困境。SPRITE的核心创新是将真实值生成重新定义为代码生成任务。我们利用大型语言模型将复杂的空间问题编译成可执行程序，然后与从模拟器中提取的高精度场景元信息进行验证。这确保了我们的真实值在计算上精确且可验证，同时大型语言模型的生成能力提供了广泛的语言多样性。利用此流程，我们整理了一个包含3个模拟器、11k+场景和300k+图像/视频指令微调对的数据集。我们证明，在我们的数据上训练的视觉语言模型在多个空间基准测试上取得了显著的性能提升，并优于同等大小的其他开源数据集。此外，可扩展性分析证实了我们的假设：克服传统模板方法的低多样性性质对于构建强大、可泛化的空间智能至关重要。我们将公开SPRITE框架代码和完整的300k+数据集，以促进未来空间智能研究。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决多模态大语言模型在空间理解和推理方面的局限性问题。这个问题很重要，因为空间理解和推理能力是AI系统与现实世界交互的基础，它超越了简单的物体识别，需要模型对物体关系、3D姿态和场景有深刻理解。没有这种能力，AI模型无法在复杂真实世界中可靠地规划、导航或操作物体，这限制了具身智能的发展。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先认识到当前多模态大语言模型在空间理解方面的不足，并分析了现有数据生成方法的局限性：基于模板的方法虽然可扩展但结构僵化，缺乏多样性；人工标注虽然语言多样但不可扩展且计算不精确。作者设计了SPRITE框架，借鉴了现有的模拟器技术、视觉-语言模型的成功经验以及大语言模型的生成能力，将真实标签生成重新定义为代码生成任务，同时利用模拟器提供高精度场景元信息和大语言模型提供语言多样性。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是将真实标签生成重新定义为代码生成任务，利用大语言模型将复杂空间问题编译成可执行程序，然后与从模拟器中提取的高精度场景元信息进行验证，确保真实标签在计算上精确且可验证，同时利用大语言模型的生成能力提供丰富的语言多样性。整体流程包括：1)从模拟器收集多模态数据和对象元信息；2)使用VLM解决对象命名歧义；3)使用GPT-4o生成多样化结构化问题；4)使用代码大语言模型生成可执行代码获取真实标签；5)通过自动质量控制确保数据质量。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)将真实标签生成重新定义为代码生成任务，确保计算精确性和可验证性；2)提出SPRITE框架，同时解决数据生成的多样性、可扩展性和精确性三难困境；3)构建大规模数据集，覆盖多种空间推理任务。相比之前的工作，SPRITE克服了基于模板方法的僵化结构和语言多样性不足的问题，解决了人工标注的计算不精确和不可扩展性问题，同时保证了数据的多样性、可扩展性和精确性。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'SPRITE框架通过将真实标签生成重新定义为代码生成任务，解决了多模态大语言模型在空间推理数据生成中的多样性、可扩展性和精确性难以兼顾的问题，显著提升了模型在空间理解任务上的性能。'}


### 论文摘要

Embodied intelligence, a grand challenge in artificial intelligence, is fundamentally constrained by the limited spatial understanding and reasoning capabilities of current models. Prevailing efforts to address this through enhancing Vision-Language Models (VLMs) are trapped in a dilemma: template-based datasets are scalable but structurally rigid, while manual annotation is linguistically diverse but unscalable and, critically, computationally imprecise. We introduce SPRITE, a novel framework that overcomes this dilemma by leveraging simulators and large models to programmatically synthesize scalable, diverse, and high-quality spatial reasoning data. The core innovation of SPRITE is to reframe ground-truth generation as a code-generation task. We utilize LLMs to compile complex spatial questions into executable programs, which are then verified against high-precision scene meta-information extracted from simulators. This ensures our ground truth is both computationally precise and verifiable, while the generative power of LLMs provides vast linguistic diversity. Leveraging this pipeline, we have curated a dataset encompassing 3 simulators, 11k+ scenes, and 300k+ image/video instruction-tuning pairs. We demonstrate that a VLM trained on our data achieves significant performance gains on multiple spatial benchmarks and outperforms other open-source datasets of equivalent size. Furthermore, a scalability analysis confirms our hypothesis that overcoming the low-diversity nature of traditional template methods is essential for building robust, generalizable spatial intelligence. We will make the SPRITE framework code and the full 300k+ dataset publicly available to facilitate future research in spatial intelligence.

---

## 70. Unified Semantic Transformer for 3D Scene Understanding

**论文链接:** [http://arxiv.org/abs/2512.14364v2](http://arxiv.org/abs/2512.14364v2)

**作者:** Sebastian Koch, Johanna Wald, Hidenobu Matsuki, Pedro Hermosilla, Timo Ropinski, Federico Tombari

**发布时间:** 2025-12-16

**备注:** Project page: https://unite-page.github.io/

### GPT解析

### 总结

本文介绍了UNITE，一种用于3D场景理解的统一语义Transformer，是一种新颖的前馈神经网络，可以在单一模型中统一多种3D语义任务。

### 背景

全面的3D场景理解涉及捕获和解析非结构化的3D环境。由于现实世界的固有复杂性，现有模型主要是为特定任务开发和限制的。

### 目的

开发一种能够在单一模型中统一多种3D语义任务的方法，实现对未见场景的快速、全面的3D语义理解。

### 方法

UNITE是一种前馈神经网络，以完全端到端的方式处理未见过的场景，仅需几秒钟即可推断完整的3D语义几何。该方法能够直接从RGB图像预测多个语义属性，包括3D场景分割、实例嵌入、开放词汇特征，以及功能和关节运动。训练采用2D蒸馏结合自监督，并利用新颖的多视图损失确保3D视图一致性。

### 主要发现

UNITE在几种不同的语义任务上实现了最先进的性能，在许多情况下甚至超过了特定任务模型的性能，超越了在真实3D几何上运行的方法。

### 结论

UNITE是一种统一多种3D语义任务的有效方法，可以在未见过的场景上快速工作并实现高性能，为3D场景理解提供了新的解决方案。

### 翻译

全面的3D场景理解涉及捕获和解析非结构化的3D环境。由于现实世界的固有复杂性，现有模型主要被开发并局限于特定任务。我们引入了UNITE，一种用于3D场景理解的统一语义Transformer，这是一种新颖的前馈神经网络，可以在单一模型中统一多种3D语义任务。我们的模型以完全端到端的方式处理未见过的场景，只需几秒钟即可推断完整的3D语义几何。我们的方法能够直接从RGB图像预测多个语义属性，包括3D场景分割、实例嵌入、开放词汇特征，以及功能和关节运动。该方法使用2D蒸馏进行训练，严重依赖自监督，并利用新颖的多视图损失来确保3D视图一致性。我们证明UNITE在几种不同的语义任务上实现了最先进的性能，甚至在许多情况下超过了特定任务模型的性能，超越了在真实3D几何上运行的方法。项目网站见unite-page.github.io

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决如何实现一个统一的3D场景理解模型，能够同时处理多种3D语义任务（语义分割、实例分割、开放词汇搜索和关节预测）的问题。这个问题在现实和研究中很重要，因为3D场景理解是AR/VR和机器人的基础应用，使系统能够感知周围环境并构建丰富的3D表示。现有方法大多是任务特定的，缺乏通用性，或者依赖于复杂的后处理步骤，无法实现真正的端到端学习，限制了它们在复杂现实环境中的应用。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先分析了现有方法的局限性：辐射场方法依赖于已知相机姿态和场景特定训练；蒸馏方法需要3D重建；基于提升的技术依赖不可微分的后处理。作者借鉴了VGGT作为几何基础，使用DPT学习开放词汇语义，并利用SAM进行实例分割和CLIP作为视觉语言特征提取器。设计思路是创建一个统一的前馈transformer，通过联合学习几何和语义实现原生3D一致性，使用多视图一致性损失确保不同视图中相同3D点的特征一致，并利用对比学习学习度量嵌入空间。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是创建一个统一的前馈神经网络(UNITE)，能够在单一模型中统一多种3D语义任务，通过联合学习几何和语义实现原生3D一致性，避免手工设计的提升步骤。整体流程：1)输入多视图RGB图像；2)使用基于VGGT的预训练编码器处理图像，预测相机姿态和点图；3)使用DPT头预测语义特征；4)使用另一个DPT头预测实例特征；5)使用专门DPT头预测关节；6)使用2D蒸馏损失和多视图一致性损失进行端到端训练；7)输出带密集语义、实例和关节特征的3D点云。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)统一模型架构，首次在单一前馈模型中统一多种3D语义任务；2)原生3D语义一致性，通过联合学习几何和语义避免手工提升步骤；3)多视图一致性损失，确保不同视图中相同3D点的特征一致；4)对比学习实例分割，学习度量嵌入空间；5)端到端训练，无需手动标注。相比之前的工作，UNITE是单一前馈模型而非多模型组合，避免了不可微分的后处理，不需要已知相机姿态和场景特定训练，且能泛化到新环境。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'UNITE是一个统一的前馈transformer模型，能够在单一网络中从RGB图像实现几何和语义的联合3D场景理解，通过多视图一致性损失确保不同视角间的语义一致性，并在多种3D语义任务上实现了最先进的性能。'}


### 论文摘要

Holistic 3D scene understanding involves capturing and parsing unstructured 3D environments. Due to the inherent complexity of the real world, existing models have predominantly been developed and limited to be task-specific. We introduce UNITE, a Unified Semantic Transformer for 3D scene understanding, a novel feed-forward neural network that unifies a diverse set of 3D semantic tasks within a single model. Our model operates on unseen scenes in a fully end-to-end manner and only takes a few seconds to infer the full 3D semantic geometry. Our approach is capable of directly predicting multiple semantic attributes, including 3D scene segmentation, instance embeddings, open-vocabulary features, as well as affordance and articulations, solely from RGB images. The method is trained using a combination of 2D distillation, heavily relying on self-supervision and leverages novel multi-view losses designed to ensure 3D view consistency. We demonstrate that UNITE achieves state-of-the-art performance on several different semantic tasks and even outperforms task-specific models, in many cases, surpassing methods that operate on ground truth 3D geometry. See the project website at unite-page.github.io

---

## 71. LinkedOut: Linking World Knowledge Representation Out of Video LLM for Next-Generation Video Recommendation

**论文链接:** [http://arxiv.org/abs/2512.16891v1](http://arxiv.org/abs/2512.16891v1)

**作者:** Haichao Zhang, Yao Lu, Lichen Wang, Yunzhe Li, Daiwei Chen, Yunpeng Xu, Yun Fu

**发布时间:** 2025-12-18

### GPT解析

### 总结

论文提出了LinkedOut，一种新的表示方法，用于从视频中直接提取VLLM世界知识，实现快速推理，支持多视频历史，并消除语言瓶颈。它是首个基于VLLM的、无需手工标注标签即可在原始帧上运行的视频推荐方法，在标准基准测试中取得了最先进的结果。

### 背景

视频大语言模型(VLLMs)通过在互联网规模数据上的预训练，实现了具有世界知识感知的视频理解，并在电影分析和视频问答等任务上显示出潜力。然而，将VLLMs部署到下游任务（如视频推荐）仍然具有挑战性，因为真实系统需要多视频输入、轻量级骨干网络、低延迟顺序推理和快速响应。

### 目的

解决VLLMs在视频推荐等下游任务部署中面临的挑战，包括：解码器生成导致顺序推理高延迟、典型接口不支持多视频输入、以及将输出限制为语言会丢弃下游视觉任务重要的细粒度视觉细节。

### 方法

提出了LinkedOut表示方法，它使用VLLMs从原始帧中提取语义基础、知识感知的标记，由可提示的查询和可选的辅助模态引导。引入了一个跨层知识融合的MoE（专家混合模型），从丰富的VLLM特征中选择适当的抽象层次，实现个性化、可解释和低延迟的推荐。

### 主要发现

LinkedOut是首个基于VLLM的、无需手工标注标签即可在原始帧上运行的视频推荐方法，在标准基准测试中取得了最先进的结果。可解释性研究和消融实验证实了层多样性和逐层融合的益处。

### 结论

LinkedOut为下游视觉任务（如推荐）提供了一个实用路径，能够充分利用VLLM的世界知识先验和视觉推理能力。

### 翻译

视频大语言模型(VLLMs)通过在互联网规模数据上的预训练，实现了具有世界知识感知的视频理解，并在电影分析和视频问答等任务上显示出潜力。然而，将VLLMs部署到下游任务（如视频推荐）仍然具有挑战性，因为真实系统需要多视频输入、轻量级骨干网络、低延迟顺序推理和快速响应。在实践中，(1)仅解码生成会导致顺序推理的高延迟，(2)典型接口不支持多视频输入，以及(3)将输出限制为语言会丢弃下游视觉任务重要的细粒度视觉细节。我们认为这些局限性源于缺乏一种既能保留像素级细节又能利用世界知识的表示。我们提出了LinkedOut，一种表示方法，它直接从视频中提取VLLM世界知识，以实现快速推理，支持多视频历史，并消除语言瓶颈。LinkedOut使用VLLMs从原始帧中提取语义基础、知识感知的标记，由可提示的查询和可选的辅助模态引导。我们引入了一个跨层知识融合MoE，从丰富的VLLM特征中选择适当的抽象层次，实现个性化、可解释和低延迟的推荐。据我们所知，LinkedOut是首个基于VLLM的、无需手工标注标签即可在原始帧上运行的视频推荐方法，在标准基准测试中取得了最先进的结果。可解释性研究和消融实验证实了层多样性和逐层融合的益处，指出了一个充分利用VLLM世界知识先验和视觉推理能力用于下游视觉任务（如推荐）的实用路径。


### 论文摘要

Video Large Language Models (VLLMs) unlock world-knowledge-aware video understanding through pretraining on internet-scale data and have already shown promise on tasks such as movie analysis and video question answering. However, deploying VLLMs for downstream tasks such as video recommendation remains challenging, since real systems require multi-video inputs, lightweight backbones, low-latency sequential inference, and rapid response. In practice, (1) decode-only generation yields high latency for sequential inference, (2) typical interfaces do not support multi-video inputs, and (3) constraining outputs to language discards fine-grained visual details that matter for downstream vision tasks. We argue that these limitations stem from the absence of a representation that preserves pixel-level detail while leveraging world knowledge. We present LinkedOut, a representation that extracts VLLM world knowledge directly from video to enable fast inference, supports multi-video histories, and removes the language bottleneck. LinkedOut extracts semantically grounded, knowledge-aware tokens from raw frames using VLLMs, guided by promptable queries and optional auxiliary modalities. We introduce a cross-layer knowledge fusion MoE that selects the appropriate level of abstraction from the rich VLLM features, enabling personalized, interpretable, and low-latency recommendation. To our knowledge, LinkedOut is the first VLLM-based video recommendation method that operates on raw frames without handcrafted labels, achieving state-of-the-art results on standard benchmarks. Interpretability studies and ablations confirm the benefits of layer diversity and layer-wise fusion, pointing to a practical path that fully leverages VLLM world-knowledge priors and visual reasoning for downstream vision tasks such as recommendation.

---

## 72. CPMamba: Selective State Space Models for MIMO Channel Prediction in High-Mobility Environments

**论文链接:** [http://arxiv.org/abs/2512.16315v1](http://arxiv.org/abs/2512.16315v1)

**作者:** Sheng Luo, Jiashu Xie, Yueling Che, Junmei Yao, Jian Tian, Daquan Feng, Kaishun Wu

**发布时间:** 2025-12-18

### GPT解析

### 总结

本文提出CPMamba，一种基于选择性状态空间模型的高效信道预测框架，通过特征提取网络和堆叠残差Mamba模块有效捕获信道长程依赖关系，在保持线性计算复杂度的同时实现高精度预测，并显著减少参数数量。

### 背景

信道预测是提高MIMO-OFDM系统中预编码、自适应调制和资源分配等性能的关键技术。在高移动性场景下，面对快速时变信道，信道预测对抵抗信道老化、确保通信质量至关重要。

### 目的

解决现有信道预测方法的高复杂度和无法准确建模信道时间变化的问题，开发一种高效且准确的信道预测框架。

### 方法

提出CPMamba框架，包含专门设计的特征提取和嵌入网络用于从历史CSI中提取特征，以及堆叠的残差Mamba模块进行时间建模，利用输入相关的选择性机制动态调整状态转换。

### 主要发现

在3GPP标准信道模型下，CPMamba在所有场景实现最先进预测精度，具有优越泛化能力和鲁棒性，相比现有基线模型减少约50%参数数量，同时实现相当或更好的性能。

### 结论

CPMamba是解决MIMO-OFDM系统高移动性场景下信道预测问题的有效方案，显著降低了实际部署门槛，为通信系统性能提升提供了新途径。

### 翻译

信道预测是提高MIMO-OFDM系统中预编码、自适应调制和资源分配等各项功能性能的关键技术。特别是在具有快速时变信道的移动性高的场景下，抵抗信道老化和确保通信质量至关重要。然而，现有方法存在高复杂度和无法准确建模信道时间变化的问题。为解决这一问题，本文提出了CPMamba——一种基于选择性状态空间模型的高效信道预测框架。所提出的CPMamba架构通过专门设计的特征提取和嵌入网络从历史信道状态信息中提取特征，并使用堆叠的残差Mamba模块进行时间建模。通过利用输入相关的选择性机制动态调整状态转换，它能够有效捕获CSI之间的长程依赖关系，同时保持线性计算复杂度。在3GPP标准信道模型下的模拟结果表明，CPMamba在所有场景下实现了最先进的预测精度，同时具有优越的泛化能力和鲁棒性。与现有基线模型相比，CPMamba减少了约50%的参数数量，同时实现了相当或更好的性能，从而显著降低了实际部署的门槛。


### 论文摘要

Channel prediction is a key technology for improving the performance of various functions such as precoding, adaptive modulation, and resource allocation in MIMO-OFDM systems. Especially in high-mobility scenarios with fast time-varying channels, it is crucial for resisting channel aging and ensuring communication quality. However, existing methods suffer from high complexity and the inability to accurately model the temporal variations of channels. To address this issue, this paper proposes CPMamba -- an efficient channel prediction framework based on the selective state space model. The proposed CPMamba architecture extracts features from historical channel state information (CSI) using a specifically designed feature extraction and embedding network and employs stacked residual Mamba modules for temporal modeling. By leveraging an input-dependent selective mechanism to dynamically adjust state transitions, it can effectively capture the long-range dependencies between the CSIs while maintaining a linear computational complexity. Simulation results under the 3GPP standard channel model demonstrate that CPMamba achieves state-of-the-art prediction accuracy across all scenarios, along with superior generalization and robustness. Compared to existing baseline models, CPMamba reduces the number of parameters by approximately 50 percent while achieving comparable or better performance, thereby significantly lowering the barrier for practical deployment.

---

## 73. LaverNet: Lightweight All-in-one Video Restoration via Selective Propagation

**论文链接:** [http://arxiv.org/abs/2512.16313v1](http://arxiv.org/abs/2512.16313v1)

**作者:** Haiyu Zhao, Yiwen Shan, Yuanbiao Gou, Xi Peng

**发布时间:** 2025-12-18

### GPT解析

### 总结

本文提出了一种轻量级一体化视频修复网络LaverNet，解决了现有方法在处理时变退化时的两个主要挑战：退化主导时间建模和依赖大型模型。LaverNet仅有362K参数，通过引入选择性传播退化无关特征的机制，实现了与现有模型相当甚至更优的性能。

### 背景

最近的研究探索了一体化视频修复，使用统一模型处理多种退化问题，但这些方法在处理时变退化时面临挑战。

### 目的

解决一体化视频修复中时变退化处理的两个挑战：退化主导时间建模和依赖大型模型的问题。

### 方法

提出轻量级一体化视频修复网络LaverNet（362K参数），引入选择性传播机制，仅传输退化无关特征跨帧，减轻退化对时间建模的影响。

### 主要发现

LaverNet尽管体积小（现有模型的参数不到1%），但在基准测试中实现了相当甚至更优的性能，证明紧凑网络可以实现强大的一体化修复。

### 结论

通过LaverNet，作者展示了使用紧凑网络可以实现强大的一体化视频修复，解决了现有方法在处理时变退化时的主要问题。

### 翻译

最近的研究探索了一体化视频修复，使用统一模型处理多种退化问题。然而，这些方法在处理时变退化时仍面临两个挑战。首先，退化可能主导时间建模，使模型专注于伪影而非视频内容。其次，当前方法通常依赖大型模型处理一体化修复，掩盖了潜在的困难。为解决这些挑战，我们提出了一个轻量级的一体化视频修复网络LaverNet，只有362K参数。为减轻退化对时间建模的影响，我们引入了一种新的传播机制，选择性地仅传输与退化无关的特征跨帧。通过LaverNet，我们证明了一个紧凑的网络可以实现强大的一体化修复。尽管体积小（现有模型的参数不到1%），LaverNet在基准测试中实现了相当甚至更优的性能。


### 论文摘要

Recent studies have explored all-in-one video restoration, which handles multiple degradations with a unified model. However, these approaches still face two challenges when dealing with time-varying degradations. First, the degradation can dominate temporal modeling, confusing the model to focus on artifacts rather than the video content. Second, current methods typically rely on large models to handle all-in-one restoration, concealing those underlying difficulties. To address these challenges, we propose a lightweight all-in-one video restoration network, LaverNet, with only 362K parameters. To mitigate the impact of degradations on temporal modeling, we introduce a novel propagation mechanism that selectively transmits only degradation-agnostic features across frames. Through LaverNet, we demonstrate that strong all-in-one restoration can be achieved with a compact network. Despite its small size, less than 1\% of the parameters of existing models, LaverNet achieves comparable, even superior performance across benchmarks.

---

## 74. AMUSE: Audio-Visual Benchmark and Alignment Framework for Agentic Multi-Speaker Understanding

**论文链接:** [http://arxiv.org/abs/2512.16250v1](http://arxiv.org/abs/2512.16250v1)

**作者:** Sanjoy Chowdhury, Karren D. Yang, Xudong Liu, Fartash Faghri, Pavan Kumar Anasosalu Vasu, Oncel Tuzel, Dinesh Manocha, Chun-Liang Li, Raviteja Vemulapalli

**发布时间:** 2025-12-18

### GPT解析

### 总结

本文介绍了AMUSE基准测试和RAFT框架，用于评估和改进多模态大语言模型在多说话人对话场景中的代理推理能力。

### 背景

当前多模态大语言模型如GPT-4o和Qwen3-Omni在感知能力上表现出色，但在需要代理推理的多说话人、对话中心化场景中表现不佳，这些场景要求模型跟踪说话者、保持角色并在时间上锚定事件，这对多模态音频视频理解至关重要。

### 目的

设计专门的基准测试和框架来评估和改进多模态大语言模型在多说话人对话场景中的推理能力。

### 方法

引入AMUSE基准测试，围绕本质上需要代理的任务设计，要求模型将复杂音频视频交互分解为规划、锚定和反思步骤；评估MLLMs在零样本、引导式和代理式三种模式下的表现；提出RAFT框架，整合奖励优化和内在多模态自评估作为奖励，以及选择性参数适应。

### 主要发现

当前模型在所有模式下都表现出较弱的多说话人推理能力，在非代理和代理评估下行为不一致；使用RAFT框架可实现高达39.52%的相对准确率提升。

### 结论

AMUSE和RAFT共同提供了一个实用平台，用于检验多模态模型中的代理推理能力并改进其性能。

### 翻译

最近的多模态大语言模型如GPT-4o和Qwen3-Omni展现出强大的感知能力，但在需要代理推理的多说话人、对话中心化场景中表现挣扎，这些场景要求模型跟踪谁在说话、保持角色并在时间上锚定事件。这些场景对于多模态音频视频理解至关重要，在对话式视频助手和会议分析等应用中，模型必须共同推理音频和视频流。我们引入了AMUSE，一个围绕本质上需要代理的任务设计的基准测试，要求模型将复杂的音频视频交互分解为规划、锚定和反思步骤。它评估了MLLMs在零样本、引导式和代理式三种模式以及六个任务家族上的表现，包括时空说话者锚定和多模态对话摘要。在所有模式下，当前模型都表现出较弱的多说话人推理能力，并且在非代理和代理评估下行为不一致。受这些任务本质上具有的代理性质和最近LLM代理进展的启发，我们提出了RAFT，一个数据高效的代理对齐框架，将奖励优化与内在多模态自评估作为奖励相结合，并采用选择性参数适应实现数据和参数高效更新。使用RAFT，我们在基准测试上实现了高达39.52%的相对准确率提升。AMUSE和RAFT共同提供了一个实用平台，用于检验多模态模型中的代理推理能力并改进其性能。


### 论文摘要

Recent multimodal large language models (MLLMs) such as GPT-4o and Qwen3-Omni show strong perception but struggle in multi-speaker, dialogue-centric settings that demand agentic reasoning tracking who speaks, maintaining roles, and grounding events across time. These scenarios are central to multimodal audio-video understanding, where models must jointly reason over audio and visual streams in applications such as conversational video assistants and meeting analytics. We introduce AMUSE, a benchmark designed around tasks that are inherently agentic, requiring models to decompose complex audio-visual interactions into planning, grounding, and reflection steps. It evaluates MLLMs across three modes zero-shot, guided, and agentic and six task families, including spatio-temporal speaker grounding and multimodal dialogue summarization. Across all modes, current models exhibit weak multi-speaker reasoning and inconsistent behavior under both non-agentic and agentic evaluation. Motivated by the inherently agentic nature of these tasks and recent advances in LLM agents, we propose RAFT, a data-efficient agentic alignment framework that integrates reward optimization with intrinsic multimodal self-evaluation as reward and selective parameter adaptation for data and parameter efficient updates. Using RAFT, we achieve up to 39.52\% relative improvement in accuracy on our benchmark. Together, AMUSE and RAFT provide a practical platform for examining agentic reasoning in multimodal models and improving their capabilities.

---

## 75. Explainable AI in Big Data Fraud Detection

**论文链接:** [http://arxiv.org/abs/2512.16037v1](http://arxiv.org/abs/2512.16037v1)

**作者:** Ayush Jain, Rahul Kulkarni, Siyi Lin

**发布时间:** 2025-12-17

**备注:** 7 pages, 3 figures, research project

### GPT解析

### 总结

这篇论文探讨了如何将可解释人工智能(XAI)集成到大数据分析流程中，用于欺诈检测和风险管理。作者回顾了大数据特征、分析工具和XAI方法，并提出了一个结合大数据基础设施与上下文感知解释机制的框架，同时指出了未来研究方向。

### 背景

大数据已成为现代金融、保险和网络安全应用的核心，使机器学习系统能够进行大规模风险评估和欺诈检测。然而，对自动分析日益增长的关注引发了关于透明度、监管合规性和信任的重要问题。

### 目的

本文旨在研究如何将可解释人工智能(XAI)整合到大数据分析流程中，用于欺诈检测和风险管理，以提高透明度、监管合规性和信任度。

### 方法

作者回顾了大数据特征和主要分析工具，包括分布式存储系统、流处理平台和先进的欺诈检测模型（如异常检测器、基于图的方法和集成分类器）。此外，作者还介绍了广泛使用的XAI方法，包括LIME、SHAP、反事实解释和注意力机制，并分析了它们在规模化部署时的优势和局限性。

### 主要发现

研究确定了与可扩展性、实时处理以及图模型和时序模型可解释性相关的主要研究差距。为解决这些挑战，作者概述了一个概念框架，将可扩展的大数据基础设施与上下文感知的解释机制和人类反馈相结合。

### 结论

论文以可扩展XAI、隐私保护解释和可解释欺诈检测系统标准化评估方法的开放研究方向作为结尾。

### 翻译

大数据已成为现代金融、保险和网络安全应用的核心，使机器学习系统能够执行大规模风险评估和欺诈检测。然而，对自动分析日益增长的依赖引入了关于透明度、监管合规性和信任的重要问题。本文研究了如何将可解释人工智能(XAI)集成到大数据分析流程中，用于欺诈检测和风险管理。我们回顾了关键的大数据特征，并调查了主要分析工具，包括分布式存储系统、流处理平台和先进的欺诈检测模型，如异常检测器、基于图的方法和集成分类器。我们还介绍了广泛使用的XAI方法的系统回顾，包括LIME、SHAP、反事实解释和注意力机制，并分析了它们在规模化部署时的优势和局限性。基于这些发现，我们确定了与可扩展性、实时处理以及图模型和时序模型可解释性相关的主要研究差距。为解决这些挑战，我们概述了一个概念框架，将可扩展的大数据基础设施与上下文感知的解释机制和人类反馈相结合。论文以可扩展XAI、隐私保护解释和可解释欺诈检测系统标准化评估方法的开放研究方向作为结尾。


### 论文摘要

Big Data has become central to modern applications in finance, insurance, and cybersecurity, enabling machine learning systems to perform large-scale risk assessments and fraud detection. However, the increasing dependence on automated analytics introduces important concerns about transparency, regulatory compliance, and trust. This paper examines how explainable artificial intelligence (XAI) can be integrated into Big Data analytics pipelines for fraud detection and risk management. We review key Big Data characteristics and survey major analytical tools, including distributed storage systems, streaming platforms, and advanced fraud detection models such as anomaly detectors, graph-based approaches, and ensemble classifiers. We also present a structured review of widely used XAI methods, including LIME, SHAP, counterfactual explanations, and attention mechanisms, and analyze their strengths and limitations when deployed at scale. Based on these findings, we identify key research gaps related to scalability, real-time processing, and explainability for graph and temporal models. To address these challenges, we outline a conceptual framework that integrates scalable Big Data infrastructure with context-aware explanation mechanisms and human feedback. The paper concludes with open research directions in scalable XAI, privacy-aware explanations, and standardized evaluation methods for explainable fraud detection systems.

---

## 76. LADY: Linear Attention for Autonomous Driving Efficiency without Transformers

**论文链接:** [http://arxiv.org/abs/2512.15038v2](http://arxiv.org/abs/2512.15038v2)

**作者:** Jihao Huang, Xi Xia, Zhiyuan Li, Tianle Liu, Jingke Wang, Junbo Chen, Tengju Ye

**发布时间:** 2025-12-17

**备注:** Under review

### GPT解析

### 总结

本文提出了LADY，第一个完全基于线性注意力机制的端到端自动驾驶生成模型，能够有效融合长程时间上下文并支持跨模态信息交换，在保持高性能的同时显著降低计算复杂度。

### 背景

端到端范式在自动驾驶领域显示出巨大潜力，但大多数现有方法基于Transformer架构，其二次方注意力成本限制了建模长时空序列的能力，特别是在资源受限的边缘平台上。现有的线性注意力架构仅限于自注意力，缺乏对自动驾驶至关重要的跨模态和跨时间交互支持。

### 目的

开发一种完全基于线性注意力机制的端到端自动驾驶生成模型，实现长程时间上下文融合，同时保持恒定的计算和内存成本，并引入轻量级线性交叉注意力机制实现有效的跨模态信息交换。

### 方法

提出了LADY模型，该模型能够在推理时融合长程时间上下文，具有恒定的计算和内存成本，无论相机和LiDAR特征的历史长度如何。同时引入了轻量级线性交叉注意力机制，实现有效的跨模态信息交换。

### 主要发现

在NAVSIM和Bench2Drive基准测试上，LADY实现了最先进的性能，具有恒定时间和内存复杂度，提供改进的规划性能并显著降低计算成本。该模型已在边缘设备上部署和验证，展示了在资源有限场景下的实用性。

### 结论

线性注意力机制可以有效解决Transformer架构在自动驾驶中的局限性，LADY模型能够在保持高性能的同时显著降低计算复杂度，并在资源受限的边缘设备上具有良好的实用性和部署能力。

### 翻译

端到端范式已经展示了自动驾驶的巨大潜力。此外，大多数现有方法都基于Transformer架构构建。然而，Transformer会产生二次方注意力成本，限制了它们建模长时空序列的能力，特别是在资源受限的边缘平台上。由于自动驾驶本质上需要高效的时间建模，这一挑战严重限制了它们的部署和实时性能。最近，线性注意力机制由于其优越的时空复杂度而获得了越来越多的关注。然而，现有的线性注意力架构仅限于自注意力，缺乏对自动驾驶至关重要的跨模态和跨时间交互的支持。在这项工作中，我们提出了LADY，这是第一个完全基于线性注意力机制的端到端自动驾驶生成模型。LADY能够在推理时融合长程时间上下文，具有恒定的计算和内存成本，无论相机和LiDAR特征的历史长度如何。此外，我们引入了一种轻量级线性交叉注意力机制，实现了有效的跨模态信息交换。在NAVSIM和Bench2Drive基准上的实验表明，LADY以恒定的时间和内存复杂度实现了最先进的性能，提供了改进的规划性能并显著降低了计算成本。此外，该模型已在边缘设备上部署和验证，展示了其在资源有限场景下的实用性。


### 论文摘要

End-to-end paradigms have demonstrated great potential for autonomous driving. Additionally, most existing methods are built upon Transformer architectures. However, transformers incur a quadratic attention cost, limiting their ability to model long spatial and temporal sequences-particularly on resource-constrained edge platforms. As autonomous driving inherently demands efficient temporal modeling, this challenge severely limits their deployment and real-time performance. Recently, linear attention mechanisms have gained increasing attention due to their superior spatiotemporal complexity. However, existing linear attention architectures are limited to self-attention, lacking support for cross-modal and cross-temporal interactions-both crucial for autonomous driving. In this work, we propose LADY, the first fully linear attention-based generative model for end-to-end autonomous driving. LADY enables fusion of long-range temporal context at inference with constant computational and memory costs, regardless of the history length of camera and LiDAR features. Additionally, we introduce a lightweight linear cross-attention mechanism that enables effective cross-modal information exchange. Experiments on the NAVSIM and Bench2Drive benchmarks demonstrate that LADY achieves state-of-the-art performance with constant-time and memory complexity, offering improved planning performance and significantly reduced computational cost. Additionally, the model has been deployed and validated on edge devices, demonstrating its practicality in resource-limited scenarios.

---

