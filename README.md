# 今日论文推荐 - 2025-10-03

共 100 篇论文

---

## 1. Fine-Grained Urban Traffic Forecasting on Metropolis-Scale Road Networks

**论文链接:** [http://arxiv.org/abs/2510.02278v1](http://arxiv.org/abs/2510.02278v1)

**作者:** Fedor Velikonivtsev, Oleg Platonov, Gleb Bazhenov, Liudmila Prokhorenkova

**发布时间:** 2025-10-02

### GPT解析

### 总结

这项研究提供了更完整、更真实、更具挑战性的交通预测基准，通过发布两个主要城市的道路网络数据集，其中最大的包含近100,000个路段。研究还提出了一种替代方法，使用不包含专门时间序列处理模块的图神经网络，实现更好的可扩展性和更强的预测性能。

### 背景

交通预测是机器学习社区关注的重点，时空图神经网络已成为最流行方法。然而，当前公开基准数据集存在显著缺点：缺乏道路连通性信息、道路属性信息有限、路段数量较少，且主要包含城际高速公路信息，而非更具挑战性的城市道路网络。

### 目的

提供更完整、更真实、更具挑战性的交通预测基准，并开发能够处理大规模数据集的神经交通预测方法。

### 方法

发布两个主要城市的道路网络数据集，包含丰富的道路特征和细粒度的交通流量与速度数据。提出一种不包含专门时间序列处理模块的图神经网络方法，以提升可扩展性。

### 主要发现

大多数当前神经时空模型无法扩展到大规模数据集。提出的替代方法（不包含专门时间序列处理模块的GNN）实现了更好的可扩展性，同时展示了更强的预测性能。

### 结论

研究团队的数据集和模型洞察将成为交通预测研究的有价值资源。

### 翻译

道路网络上的交通预测是一项具有重大实际意义的复杂任务，最近引起了机器学习社区的极大关注，时空图神经网络已成为最流行的方法。当前公开可用的基准存在显著缺点，包括缺乏道路连通性信息、道路属性信息有限，以及路段数量相对较少。此外，当前数据集主要包含城际高速公路信息，而城市道路网络由于道路更密集和交通模式更复杂，实际上提出了更具挑战性的预测任务。在这项工作中，我们通过发布两个主要城市的道路网络数据集，为交通预测提供了一个更完整、更真实、更具挑战性的基准。我们的数据集包含丰富的道路特征，并提供关于交通流量和交通速度的细粒度数据。我们提出的替代方法使用不包含专门时间序列处理模块的GNN，实现了更好的可扩展性和更强的预测性能。


### 论文摘要

Traffic forecasting on road networks is a complex task of significant practical importance that has recently attracted considerable attention from the machine learning community, with spatiotemporal graph neural networks (GNNs) becoming the most popular approach. The proper evaluation of traffic forecasting methods requires realistic datasets, but current publicly available benchmarks have significant drawbacks, including the absence of information about road connectivity for road graph construction, limited information about road properties, and a relatively small number of road segments that falls short of real-world applications. Further, current datasets mostly contain information about intercity highways with sparsely located sensors, while city road networks arguably present a more challenging forecasting task due to much denser roads and more complex urban traffic patterns. In this work, we provide a more complete, realistic, and challenging benchmark for traffic forecasting by releasing datasets representing the road networks of two major cities, with the largest containing almost 100,000 road segments (more than a 10-fold increase relative to existing datasets). Our datasets contain rich road features and provide fine-grained data about both traffic volume and traffic speed, allowing for building more holistic traffic forecasting systems. We show that most current implementations of neural spatiotemporal models for traffic forecasting have problems scaling to datasets of our size. To overcome this issue, we propose an alternative approach to neural traffic forecasting that uses a GNN without a dedicated module for temporal sequence processing, thus achieving much better scalability, while also demonstrating stronger forecasting performance. We hope our datasets and modeling insights will serve as a valuable resource for research in traffic forecasting.

---

## 2. Transformers Discover Molecular Structure Without Graph Priors

**论文链接:** [http://arxiv.org/abs/2510.02259v1](http://arxiv.org/abs/2510.02259v1)

**作者:** Tobias Kreiman, Yutong Bai, Fadi Atieh, Elizabeth Weaver, Eric Qu, Aditi S. Krishnapriyan

**发布时间:** 2025-10-02

### GPT解析

### 总结

本研究探讨了纯Transformer模型在分子机器学习中的应用，挑战了图神经网络(GNNs)中硬编码图的必要性。研究表明，直接在笛卡尔坐标上训练的Transformer可以近似分子能量和力，并且学习到物理上一致的注意力模式。

### 背景

图神经网络(GNNs)是分子机器学习的主导架构，特别用于分子性质预测和机器学习原子间势能。GNNs在预定义的图上进行消息传递，这些图通常由固定半径截断或k近邻方案诱导。这种设计虽然与许多分子任务中的局部性一致，但硬编码的图可能限制表达能力并减慢推理速度。

### 目的

研究纯的、未经修改的Transformer（直接在笛卡尔坐标上训练，没有预定义图或物理先验）是否可以近似分子能量和力。

### 方法

在匹配的训练计算预算下训练Transformer，使其在OMol25数据集上与最先进的等变GNN具有竞争力的能量和力平均绝对误差。使用标准Transformer架构进行分析。

### 主要发现

Transformer学习到物理上一致的模式，例如注意力权重随原子间距离呈反比衰减；由于没有硬编码的偏见，Transformer能够灵活地适应不同的分子环境；使用标准Transformer实现了可预测的性能提升，与在其他领域观察到的经验缩放定律一致。

### 结论

许多GNN的有利特性可以在Transformer中自适应地出现，挑战了硬编码图归纳偏见的必要性，指向分子建模的标准化、可扩展架构。

### 翻译

图神经网络(GNNs)是分子机器学习的主导架构，特别用于分子性质预测和机器学习原子间势能(MLIPs)。GNNs在预定义的图上进行消息传递，这些图通常由固定半径截断或k近邻方案诱导。虽然这种设计符合许多分子任务中的局部性，但硬编码的图可能由于固定的感受野而限制表达能力，并且稀疏图操作会减慢推理速度。在本工作中，我们研究纯的、未经修改的Transformer（直接在笛卡尔坐标上训练，没有预定义图或物理先验）是否可以近似分子能量和力。作为分析的起点，我们展示如何在匹配的训练计算预算下训练Transformer，使其在OMol25数据集上与最先进的等变GNN具有竞争力的能量和力平均绝对误差。我们发现Transformer学习到物理上一致的模式，例如注意力权重随原子间距离呈反比衰减，并且由于没有硬编码的偏见，能够灵活地适应不同的分子环境。使用标准Transformer还实现了可预测的性能提升，与其他领域观察到的经验缩放定律一致。我们的结果表明，GNN的许多有利特性可以在Transformer中自适应地出现，挑战了硬编码图归纳偏见的必要性，并指向分子建模的标准化、可扩展架构。


### 论文摘要

Graph Neural Networks (GNNs) are the dominant architecture for molecular machine learning, particularly for molecular property prediction and machine learning interatomic potentials (MLIPs). GNNs perform message passing on predefined graphs often induced by a fixed radius cutoff or k-nearest neighbor scheme. While this design aligns with the locality present in many molecular tasks, a hard-coded graph can limit expressivity due to the fixed receptive field and slows down inference with sparse graph operations. In this work, we investigate whether pure, unmodified Transformers trained directly on Cartesian coordinates$\unicode{x2013}$without predefined graphs or physical priors$\unicode{x2013}$can approximate molecular energies and forces. As a starting point for our analysis, we demonstrate how to train a Transformer to competitive energy and force mean absolute errors under a matched training compute budget, relative to a state-of-the-art equivariant GNN on the OMol25 dataset. We discover that the Transformer learns physically consistent patterns$\unicode{x2013}$such as attention weights that decay inversely with interatomic distance$\unicode{x2013}$and flexibly adapts them across different molecular environments due to the absence of hard-coded biases. The use of a standard Transformer also unlocks predictable improvements with respect to scaling training resources, consistent with empirical scaling laws observed in other domains. Our results demonstrate that many favorable properties of GNNs can emerge adaptively in Transformers, challenging the necessity of hard-coded graph inductive biases and pointing toward standardized, scalable architectures for molecular modeling.

---

## 3. Are LLMs Better GNN Helpers? Rethinking Robust Graph Learning under Deficiencies with Iterative Refinement

**论文链接:** [http://arxiv.org/abs/2510.01910v1](http://arxiv.org/abs/2510.01910v1)

**作者:** Zhaoyan Wang, Zheng Gao, Arogya Kharel, In-Young Ko

**发布时间:** 2025-10-02

**备注:** 14 pages

### GPT解析

### 总结

该研究针对图神经网络(GNNs)在真实场景下面临的复合缺陷问题，首次系统比较了传统方法和基于大型语言模型(LLMs)的增强方法，挑战了LLM增强始终优于传统方法的假设，并提出了RoGRAD框架通过检索增强对比学习迭代改进图表示，实验证明其性能显著优于基线方法。

### 背景

图神经网络(GNNs)被广泛应用于网络相关应用，作为学习图结构数据的核心技术。然而，在真实场景中，这些图往往存在缺陷，严重影响了GNN的性能。先前研究虽然探索了对单个缺陷的鲁棒性，但对于图原生方法和LLM增强方法在复合缺陷下的行为，仍缺乏系统理解。

### 目的

填补现有研究空白，首次进行实证研究对比传统方法和LLM-on-graph框架在各种图缺陷下的表现，揭示被忽视的脆弱性，并挑战LLM增强始终优于传统方法的假设。

### 方法

提出了Robust Graph Learning via Retrieval-Augmented Contrastive Refinement (RoGRAD)框架。这是一种迭代范式，利用检索增强生成(RAG)通过提供类别一致、多样化的增强，并通过迭代图对比学习强制区分性表示，将图的LLM增强从静态信号注入转变为动态改进。

### 主要发现

LLM增强并不总是优于传统方法，挑战了这一假设。RoGRAD框架在性能上显著优于传统GNN基线和LLM增强基线，平均提升了高达82.43%。

### 结论

RoGRAD框架通过迭代地检索增强对比学习，有效地解决了图神经网络在复合缺陷下的鲁棒性问题，为图学习领域提供了新的思路和方法。

### 翻译

图神经网络(GNNs)被广泛应用于网络相关应用，作为学习图结构数据的核心技术。然而，在真实场景中，这些图存在缺陷，严重影响了GNN的性能。虽然先前基于GNN的增强研究已经探索了对单个缺陷的鲁棒性，但对于图原生方法和大型语言模型增强方法在复合缺陷下的行为，仍然缺乏系统理解。为填补这一空白，我们进行了首次实证研究，在多种图缺陷下对这些方法进行基准测试，揭示了被忽视的脆弱性，并挑战了LLM增强始终优于传统方法的假设。基于实证发现，我们提出了RoGRAD框架，这是一种迭代范式，利用检索增强生成通过提供类别一致、多样化的增强，并通过迭代图对比学习来强制区分性表示，将图的LLM增强从静态信号注入转变为动态改进。广泛的实验证明了RoGRAD在传统GNN基线和LLM增强基线上的优越性，平均提升了高达82.43%。


### 论文摘要

Graph Neural Networks (GNNs) are widely adopted in Web-related applications, serving as a core technique for learning from graph-structured data, such as text-attributed graphs. Yet in real-world scenarios, such graphs exhibit deficiencies that substantially undermine GNN performance. While prior GNN-based augmentation studies have explored robustness against individual imperfections, a systematic understanding of how graph-native and Large Language Models (LLMs) enhanced methods behave under compound deficiencies is still missing. Specifically, there has been no comprehensive investigation comparing conventional approaches and recent LLM-on-graph frameworks, leaving their merits unclear. To fill this gap, we conduct the first empirical study that benchmarks these two lines of methods across diverse graph deficiencies, revealing overlooked vulnerabilities and challenging the assumption that LLM augmentation is consistently superior. Building on empirical findings, we propose Robust Graph Learning via Retrieval-Augmented Contrastive Refinement (RoGRAD) framework. Unlike prior one-shot LLM-as-Enhancer designs, RoGRAD is the first iterative paradigm that leverages Retrieval-Augmented Generation (RAG) to inject retrieval-grounded augmentations by supplying class-consistent, diverse augmentations and enforcing discriminative representations through iterative graph contrastive learning. It transforms LLM augmentation for graphs from static signal injection into dynamic refinement. Extensive experiments demonstrate RoGRAD's superiority over both conventional GNN- and LLM-enhanced baselines, achieving up to 82.43% average improvement.

---

## 4. Detecting LLM-Generated Spam Reviews by Integrating Language Model Embeddings and Graph Neural Network

**论文链接:** [http://arxiv.org/abs/2510.01801v1](http://arxiv.org/abs/2510.01801v1)

**作者:** Xin Liu, Rongwu Xu, Xinyi Jia, Jason Liao, Jiao Sun, Ling Huang, Wei Xu

**发布时间:** 2025-10-02

### GPT解析

### 总结

该研究针对大型语言模型生成的高说服力垃圾评论提出了一种混合检测模型FraudSquad，该模型在LLM生成的垃圾评论数据集上表现出色，比现有方法提高精确度44.22%，召回率43.01%，同时保持模型大小适中且训练资源需求低。

### 背景

大型语言模型的发展使高度逼真、模仿人类写作的垃圾评论生成成为可能，这些评论对现有检测系统构成挑战，威胁在线平台的可信度。

### 目的

开发一种能够有效检测由大型语言模型生成的高说服力垃圾评论的检测系统。

### 方法

创建三个使用不同LLM生成的垃圾评论数据集，由产品元数据和真实参考评论指导；提出FraudSquad混合检测模型，整合预训练语言模型的文本嵌入和门控图transformer进行垃圾节点分类，捕获语义和行为信号。

### 主要发现

FraudSquad在三个LLM生成的数据集上比最先进基线模型提高精确度44.22%，召回率43.01%；在两个人类编写的垃圾评论数据集上也取得良好结果；模型大小适中，只需少量标记训练数据。

### 结论

FraudSquad是现实世界应用的实用解决方案，贡献包括新的合成数据集、实用的检测框架以及强调适应LLM时代垃圾检测紧迫性的实证证据。

### 翻译

大型语言模型(LLMs)的兴起使得能够生成高度说服力的垃圾评论，这些评论紧密模仿人类写作。这些评论对现有检测系统构成重大挑战，威胁在线平台的可信度。在这项工作中，我们首先使用三个不同的LLM创建了三个逼真的LLM生成垃圾评论数据集，每个数据集都由产品元数据和真实的参考评论指导。GPT-4.1的评估确认了这些评论的高说服力和欺骗潜力。为应对这一威胁，我们提出了FraudSquad，一个混合检测模型，集成了预训练语言模型的文本嵌入和用于垃圾节点分类的门控图transformer。FraudSquad捕获语义和行为信号，无需依赖手动特征工程或大量训练资源。实验表明，FraudSquad在三个LLM生成的数据集上比最先进的基线模型在精确度上提高高达44.22%，在召回率上提高43.01%，同时在两个人类编写的垃圾评论数据集上也取得了有希望的结果。此外，FraudSquad保持适中的模型大小，只需要最少的标记训练数据，使其成为现实世界应用的实用解决方案。我们的贡献包括新的合成数据集、实用的检测框架以及强调适应LLM时代垃圾检测紧迫性的实证证据。我们的代码和数据集可在以下网址获取：https://anonymous.4open.science/r/FraudSquad-5389/。


### 论文摘要

The rise of large language models (LLMs) has enabled the generation of highly persuasive spam reviews that closely mimic human writing. These reviews pose significant challenges for existing detection systems and threaten the credibility of online platforms. In this work, we first create three realistic LLM-generated spam review datasets using three distinct LLMs, each guided by product metadata and genuine reference reviews. Evaluations by GPT-4.1 confirm the high persuasion and deceptive potential of these reviews. To address this threat, we propose FraudSquad, a hybrid detection model that integrates text embeddings from a pre-trained language model with a gated graph transformer for spam node classification. FraudSquad captures both semantic and behavioral signals without relying on manual feature engineering or massive training resources. Experiments show that FraudSquad outperforms state-of-the-art baselines by up to 44.22% in precision and 43.01% in recall on three LLM-generated datasets, while also achieving promising results on two human-written spam datasets. Furthermore, FraudSquad maintains a modest model size and requires minimal labeled training data, making it a practical solution for real-world applications. Our contributions include new synthetic datasets, a practical detection framework, and empirical evidence highlighting the urgency of adapting spam detection to the LLM era. Our code and datasets are available at: https://anonymous.4open.science/r/FraudSquad-5389/.

---

## 5. BioBlobs: Differentiable Graph Partitioning for Protein Representation Learning

**论文链接:** [http://arxiv.org/abs/2510.01632v1](http://arxiv.org/abs/2510.01632v1)

**作者:** Xin Wang, Carlos Oliver

**发布时间:** 2025-10-02

### GPT解析

### 总结

BioBlobs是一种即插即用、完全可微分的蛋白质表征学习模块，通过动态分割蛋白质为灵活大小的非重叠亚结构（blobs），并将其量化到共享码本中，生成与功能相关的蛋白质亚结构离散词汇表，从而提高蛋白质编码器性能并提供功能机制性见解。

### 背景

蛋白质功能由大小和拓扑各异的相干亚结构驱动，而当前蛋白质表征学习模型依赖k-hop和固定半径邻域等刚性亚结构，扭曲了这些信号。

### 目的

开发一种能够准确捕捉蛋白质功能相关亚结构的表征方法，提高预测性能并提供功能机制性见解。

### 方法

引入BioBlobs模块，通过动态分割蛋白质结构为灵活大小、不重叠的亚结构（blobs），将这些blobs量化到共享且可解释的码本中，生成与功能相关的蛋白质亚结构离散词汇表，用于计算蛋白质嵌入。

### 主要发现

BioBlobs表征显著提高了GVP-GNN等广泛使用的蛋白质编码器在各种蛋白质表征学习任务中的性能，证明了直接捕获功能相关蛋白质亚结构的架构价值。

### 结论

直接捕获功能相关蛋白质亚结构的架构不仅能提高预测性能，还能为蛋白质功能提供机制性见解，具有重要的研究价值。

### 翻译

蛋白质功能是由相干的亚结构驱动的，这些亚结构在大小和拓扑上各不相同，然而当前的蛋白质表征学习模型（PRL）通过依赖k-hop和固定半径邻域等刚性亚结构而扭曲了这些信号。我们引入了BioBlobs，一个即插即用、完全可微分的模块，它通过动态地将结构分割成灵活大小、非重叠的亚结构（'blobs'）来表示蛋白质。生成的blobs被量化到一个共享且可解释的码本中，产生一个与功能相关的蛋白质亚结构的离散词汇表，用于计算蛋白质嵌入。我们证明，BioBlobs表征提高了广泛使用的蛋白质编码器（如GVP-GNN）在各种PRL任务中的性能。我们的方法强调了直接捕获功能相关蛋白质亚结构的架构的价值，既能提高预测性能，又能对蛋白质功能提供机制性见解。


### 论文摘要

Protein function is driven by coherent substructures which vary in size and topology, yet current protein representation learning models (PRL) distort these signals by relying on rigid substructures such as k-hop and fixed radius neighbourhoods. We introduce BioBlobs, a plug-and-play, fully differentiable module that represents proteins by dynamically partitioning structures into flexibly-sized, non-overlapping substructures ("blobs"). The resulting blobs are quantized into a shared and interpretable codebook, yielding a discrete vocabulary of function-relevant protein substructures used to compute protein embeddings. We show that BioBlobs representations improve the performance of widely used protein encoders such as GVP-GNN across various PRL tasks. Our approach highlights the value of architectures that directly capture function-relevant protein substructures, enabling both improved predictive performance and mechanistic insight into protein function.

---

## 6. Equivariant Geometric Scattering Networks via Vector Diffusion Wavelets

**论文链接:** [http://arxiv.org/abs/2510.01022v1](http://arxiv.org/abs/2510.01022v1)

**作者:** David R. Johnson, Rishabh Anand, Smita Krishnaswamy, Michael Perlmutter

**发布时间:** 2025-10-01

**备注:** Accepted for presentation at the NeurIPS workshop on New Perspectives  in Advancing Graph Machine Learning

### GPT解析

### 总结

该研究介绍了一种新型的几何散射变换版本，用于处理包含标量和向量节点特征的几何图，该变换具有关于刚体旋转平移的对称性，可整合到几何GNN框架中，并且实证表明其基于等变性散射的GNN在参数数量大幅减少的情况下，与其他等变性消息传递GNN性能相当。

### 背景

几何图处理是图神经网络领域的重要研究方向，特别是对于具有标量和向量节点特征的几何图。

### 目的

开发一种具有理想对称性的新型几何散射变换，并证明其在几何GNN框架中的有效性。

### 方法

提出了一种新型的几何散射变换版本，该变换具有关于刚体旋转平移（SE(3)等变性）的对称性，并将其整合到几何GNN框架中。

### 主要发现

基于等变性散射的GNN在参数数量大幅减少的情况下，与其他基于等变性消息传递的GNN实现了相当的性能。

### 结论

所提出的等变性散射变换是一种有效的几何图处理方法，可以在保持性能的同时显著减少参数数量。

### 翻译

我们介绍了一种用于处理包含标量和向量节点特征的几何图的新型几何散射变换版本。这种新的散射变换具有关于刚体旋转平移（即SE(3)等变性）的理想对称性，并且可以整合到几何GNN框架中。我们通过实证表明，我们基于等变性散射的GNN在参数数量大幅减少的情况下，与其他基于等变性消息传递的GNN实现了相当的性能。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决几何图神经网络中处理向量值节点特征的旋转等变性问题，以及传统消息传递网络存在的过平滑和下达问题。这个问题在现实中非常重要，因为许多应用（如分子结构分析、3D点云处理）需要处理具有几何对称性的数据，而向量特征（如位置、速度）的旋转等变性对于保持物理和几何意义至关重要。同时，解决过平滑和下达问题可以构建更深、更强大的图神经网络，提高模型性能。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者借鉴了现有的几何散射变换工作（如Zou and Lerman 2019, Gama et al. 2018），这些工作使用扩散小波捕获图的多尺度几何信息，避免了消息传递网络的局限性。作者意识到现有方法主要针对标量特征，因此扩展其处理向量特征的能力，通过向量扩散映射（Singer and Wu 2012）设计向量扩散矩阵Q。为每个节点构建局部正交基，设计特殊的激活函数和扩散操作确保旋转等变性，并将标量和向量特征处理分离后结合，充分利用两种特征信息。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是设计向量扩散小波来处理向量值节点特征，同时保持旋转等变性；为每个节点构建局部正交基，实现不同局部坐标系间的转换；分离处理标量和向量特征后结合结果。整体流程包括：1)输入几何图和节点特征；2)构建向量扩散矩阵Q，通过SVD为每个节点创建局部坐标系；3)定义向量扩散小波；4)计算标量和向量特征的散射系数；5)分别应用MLP进行特征混合，向量特征还使用门控机制；6)提取向量不变量；7)根据任务类型应用预测头。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)首次将几何散射变换扩展到处理向量值节点特征；2)通过理论证明确保旋转等变性；3)设计局部坐标系转换机制处理向量特征；4)以显著更少的参数实现与传统等变性GNN相当的性能。相比之前工作，不同之处在于：避免了消息传递网络的过平滑和下达问题；不依赖底层流形离散化假设；参数效率更高（少90%以上）；使用扩散小波而非局部捕获多尺度信息；分离处理标量和向量特征。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '这篇论文提出了一种基于向量扩散小波的等变性几何散射网络，首次将几何散射变换扩展到处理向量值节点特征，在保持旋转等变性的同时，以显著更少的参数实现了与传统等变性图神经网络相当的性能。'}


### 论文摘要

We introduce a novel version of the geometric scattering transform for geometric graphs containing scalar and vector node features. This new scattering transform has desirable symmetries with respect to rigid-body roto-translations (i.e., $SE(3)$-equivariance) and may be incorporated into a geometric GNN framework. We empirically show that our equivariant scattering-based GNN achieves comparable performance to other equivariant message-passing-based GNNs at a fraction of the parameter count.

---

## 7. Graph Neural Networks in Large Scale Wireless Communication Networks: Scalability Across Random Geometric Graphs

**论文链接:** [http://arxiv.org/abs/2510.00896v1](http://arxiv.org/abs/2510.00896v1)

**作者:** Romina Garcia Camargo, Zhiyang Wang, Alejandro Ribeiro

**发布时间:** 2025-10-01

### GPT解析

### 总结

本文研究了无线系统中图神经网络(GNNs)的可转移性特性，为随机几何图(RGGs)上的可转移性提供了正式的理论基础，并通过功率分配任务的数值实验进行了验证。

### 背景

无线系统日益复杂，推动了从传统方法向基于学习解决方案的转变。图神经网络特别适合无线系统，因为无线网络可自然表示为图。尽管实证研究表明GNN-based无线策略能有效转移，但现有理论保证无法解释这一现象，因为多数研究假设密集图，而无线系统实际上是稀疏的。

### 目的

为随机几何图(RGGs)上的GNN可转移性提供正式的理论基础，RGGs是无线网络的一种稀疏且广泛使用的模型。

### 方法

通过理论分析和随机几何图模型建立GNN可转移性的理论基础，并通过功率分配这一基本资源管理任务的数值实验验证结果。

### 主要发现

GNN模型在一个图上训练后可以有效地泛化到更大的图上，性能损失很小，这种现象在稀疏的无线网络模型中得到了理论支持。

### 结论

本文为无线系统中GNN的可转移性提供了理论保证，填补了现有理论与实证观察之间的差距，证明了GNN在无线资源管理任务中的有效性。

### 翻译

随着无线系统复杂性的日益增加，传统方法向基于学习解决方案的转变正在加速。图神经网络(GNNs)特别适合于此，因为无线网络可以自然地表示为图。GNN的一个关键特性是可转移性：在一个图上训练的模型通常可以很好地泛化到更大的图上，性能损失很小。尽管实证研究表明基于GNN的无线策略能够有效转移，但现有的理论保证并不能解释这一现象。大多数工作关注密集图，其中节点度随网络规模扩展，这一假设在无线系统中不成立。在本工作中，我们为随机几何图(RGGs)上的可转移性提供了正式的理论基础，RGGs是无线网络的一种稀疏且广泛使用的模型。我们进一步通过功率分配这一基本资源管理任务的数值实验验证了我们的结果。


### 论文摘要

The growing complexity of wireless systems has accelerated the move from traditional methods to learning-based solutions. Graph Neural Networks (GNNs) are especially well-suited here, since wireless networks can be naturally represented as graphs. A key property of GNNs is transferability: models trained on one graph often generalize to much larger graphs with little performance loss. While empirical studies have shown that GNN-based wireless policies transfer effectively, existing theoretical guarantees do not capture this phenomenon. Most works focus on dense graphs where node degrees scale with network size, an assumption that fails in wireless systems. In this work, we provide a formal theoretical foundation for transferability on Random Geometric Graphs (RGGs), a sparse and widely used model of wireless networks. We further validate our results through numerical experiments on power allocation, a fundamental resource management task.

---

## 8. LEAP: Local ECT-Based Learnable Positional Encodings for Graphs

**论文链接:** [http://arxiv.org/abs/2510.00757v1](http://arxiv.org/abs/2510.00757v1)

**作者:** Juan Amboage, Ernst Röell, Patrick Schnider, Bastian Rieck

**发布时间:** 2025-10-01

### GPT解析

### 总结

本文提出了一种名为LEAP的新型图位置编码方法，结合了欧拉特征变换的可微分近似及其局部变体，用于解决标准图神经网络在理论和实践上的局限性。

### 背景

图神经网络(GNNs)主要依赖于消息传递范式，其中节点迭代地从邻居聚合信息。然而，标准消息传递神经网络(MPNNs)面临着理论和实践上的局限性。图位置编码(PE)已成为解决这些局限性的有前途的方向。

### 目的

开发一种新的端到端可训练的图局部结构位置编码方法，以克服标准图神经网络的局限性。

### 方法

结合欧拉特征变换的可微分近似(DECT)及其局部变体(ℓ-ECT)，提出名为LEAP的新型图位置编码方法。

### 主要发现

基于LEAP的编码在多个真实世界数据集和合成任务上表现出了潜力，特别是在提取拓扑特征方面。

### 结论

LEAP-based编码可以作为图表示学习流水线的强大组件，具有实际应用价值。

### 翻译

图神经网络(GNNs)主要依赖于消息传递范式，其中节点迭代地从邻居聚合信息。然而，标准消息传递神经网络(MPNNs)面临着众所周知的理论和实践局限性。图位置编码(PE)已成为解决这些局限性的一个有前途的方向。欧拉特征变换(Euler Characteristic Transform, ECT)是一种可有效计算的几何-拓扑不变量，用于表征形状和图形。在这项工作中，我们将ECT的可微分近似(DECT)及其局部变体(ℓ-ECT)相结合，提出了LEAP，这是一种新的端到端可训练的图局部结构位置编码。我们在多个真实世界数据集以及一个旨在测试其提取拓扑特征能力的合成任务上评估了我们的方法。我们的结果强调了基于LEAP的编码作为图表示学习流水线强大组件的潜力。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决图神经网络(GNNs)在处理图数据时面临的理论和实践限制，特别是消息传递神经网络(MPNNs)的局限性，包括在高直径图中丢失信号和无法有效利用子结构信息等问题。这个问题在现实中非常重要，因为图是许多科学领域处理二元关系的主要模态，而改进的图表示学习方法对图表示学习领域的发展至关重要，能够帮助更好地理解和分析各种复杂网络结构。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者从MPNNs的局限性出发，受到Transformer架构中位置编码的启发，开始关注图的位置编码(PEs)和结构编码(SEs)。注意到现有方法大多基于几何或拓扑单一方面的信息，表达能力有限，因此决定结合两者优势。作者借鉴了现有的Euler Characteristic Transform (ECT)这一几何-拓扑不变量，以及可微分ECT近似(DECT)和局部变体(ℓ-ECT)，同时参考了Random Walk Positional Encoding (RWPE)和Laplacian Positional Encoding (LaPE)等现有图位置编码方法，最终设计出LEAP方法。', '这个方法的核心思想是什么？整体实现流程是怎样的？': 'LEAP的核心思想是利用局部欧拉特征变换(ℓ-ECT)创建一种可学习的图位置编码，结合了几何和拓扑信息。整体实现流程为：1)对图中的每个节点，计算其m跳子图；2)对子图中的节点特征进行归一化处理(均值中心化并除以最大范数)；3)计算不同方向和阈值下的可微分ECT近似，得到矩阵表示；4)使用可学习的投影函数将ECT矩阵映射到低维空间，得到节点的位置编码向量。这种方法能够捕捉图的局部结构信息，并支持端到端训练。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)提出LEAP，一种基于局部ECT的可学习图位置编码；2)结合几何和拓扑信息提高表达能力；3)实现端到端可训练性；4)提出多种ECT投影策略(线性投影、一维卷积、DeepSets、注意力机制等)；5)支持学习方向向量而非仅使用固定方向。相比之前的工作，LEAP是可学习的而非静态预处理，专门设计为局部结构编码而非全局，能够在节点特征不具信息性时仍捕获结构信息，并提供多种投影策略选择。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '这篇论文提出了LEAP，一种基于局部欧拉特征变换的可学习图位置编码方法，它结合了几何和拓扑信息，能够在多种图神经网络架构和任务中提供强大的结构表示，并实现了端到端的可训练性。'}


### 论文摘要

Graph neural networks (GNNs) largely rely on the message-passing paradigm, where nodes iteratively aggregate information from their neighbors. Yet, standard message passing neural networks (MPNNs) face well-documented theoretical and practical limitations. Graph positional encoding (PE) has emerged as a promising direction to address these limitations. The Euler Characteristic Transform (ECT) is an efficiently computable geometric-topological invariant that characterizes shapes and graphs. In this work, we combine the differentiable approximation of the ECT (DECT) and its local variant ($\ell$-ECT) to propose LEAP, a new end-to-end trainable local structural PE for graphs. We evaluate our approach on multiple real-world datasets as well as on a synthetic task designed to test its ability to extract topological features. Our results underline the potential of LEAP-based encodings as a powerful component for graph representation learning pipelines.

---

## 9. Hierarchy-Aware Neural Subgraph Matching with Enhanced Similarity Measure

**论文链接:** [http://arxiv.org/abs/2510.00402v1](http://arxiv.org/abs/2510.00402v1)

**作者:** Zhouyang Liu, Ning Liu, Yixin Chen, Jiezhong He, Menghan Jia, Dongsheng Li

**发布时间:** 2025-10-01

**备注:** Accepted by IEEE Transactions on Knowledge and Data Engineering

### GPT解析

### 总结

本文提出了一种名为NC-Iso的新型GNN架构，用于解决子图匹配问题，解决了现有方法在处理图对尺度差异和特征相对位置方面的局限性。

### 背景

子图匹配具有挑战性，需要耗时的组合搜索。基于图神经网络的方法虽然显著缩短了响应时间，但存在两个主要问题：1) 编码过程中图对间存在尺度差异，因只关注特征计数而忽略节点根子树中特征的相对位置；2) 铰链距离度量对匹配图对缺乏判别力，影响排名应用。

### 目的

提出NC-Iso架构，解决现有子图匹配方法的局限性，提供更具判别力的神经子图匹配解决方案，用于子图检索任务。

### 方法

NC-Iso通过构建节点根子树内相邻层级间的层次依赖关系保留特征相对位置，确保匹配图对保持一致层次结构并符合特征计数包含约束。同时引入相似优势比率增强度量，量化图对间相似性对差异性的优势，提升匹配对的排名能力。

### 主要发现

在九个数据集上的经验结果验证了NC-Iso的有效性、泛化能力、可扩展性和可转移性，同时保持时间效率，为子图检索提供了更具判别力的神经子图匹配解决方案。

### 结论

NC-Iso通过保留特征相对位置和引入新的相似性度量，解决了现有子图匹配方法中的关键问题，在保持时间效率的同时提高了子图匹配的准确性。

### 翻译

子图匹配具有挑战性，因为它需要耗时的组合搜索。最近的基于图神经网络（GNN）的方法通过使用GNN编码器提取图信息和使用铰链距离度量确保嵌入空间中的包含约束来解决这个问题。这些方法显著缩短了响应时间，使它们成为子图检索的 promising 解决方案。然而，它们在编码过程中图对之间存在尺度差异，因为它们只关注特征计数而忽略了节点根子树中特征的相对位置，导致包含约束被干扰和错误预测。此外，它们的铰链距离度量对匹配的图对缺乏判别力，阻碍了排名应用。我们提出了NC-Iso，一种用于神经子图匹配的新型GNN架构。NC-Iso通过构建节点根子树内相邻层级之间的层次依赖关系来保留特征的相对位置，确保匹配的图对保持一致的层次结构，同时符合特征计数中的包含约束。为了增强匹配对的排名能力，我们引入了一种新颖的相似优势比率增强度量，它量化了图对之间相似性对差异性的优势。在九个数据集上的经验结果验证了NC-Iso的有效性、泛化能力、可扩展性和可转移性，同时保持时间效率，为子图检索提供了更具判别力的神经子图匹配解决方案。代码可在https://github.com/liuzhouyang/NC-Iso获取。


### 论文摘要

Subgraph matching is challenging as it necessitates time-consuming combinatorial searches. Recent Graph Neural Network (GNN)-based approaches address this issue by employing GNN encoders to extract graph information and hinge distance measures to ensure containment constraints in the embedding space. These methods significantly shorten the response time, making them promising solutions for subgraph retrieval. However, they suffer from scale differences between graph pairs during encoding, as they focus on feature counts but overlook the relative positions of features within node-rooted subtrees, leading to disturbed containment constraints and false predictions. Additionally, their hinge distance measures lack discriminative power for matched graph pairs, hindering ranking applications. We propose NC-Iso, a novel GNN architecture for neural subgraph matching. NC-Iso preserves the relative positions of features by building the hierarchical dependencies between adjacent echelons within node-rooted subtrees, ensuring matched graph pairs maintain consistent hierarchies while complying with containment constraints in feature counts. To enhance the ranking ability for matched pairs, we introduce a novel similarity dominance ratio-enhanced measure, which quantifies the dominance of similarity over dissimilarity between graph pairs. Empirical results on nine datasets validate the effectiveness, generalization ability, scalability, and transferability of NC-Iso while maintaining time efficiency, offering a more discriminative neural subgraph matching solution for subgraph retrieval. Code available at https://github.com/liuzhouyang/NC-Iso.

---

## 10. GDLNN: Marriage of Programming Language and Neural Networks for Accurate and Easy-to-Explain Graph Classification

**论文链接:** [http://arxiv.org/abs/2510.00374v1](http://arxiv.org/abs/2510.00374v1)

**作者:** Minseok Jeon, Seunghyun Park

**发布时间:** 2025-10-01

### GPT解析

### 总结

本文提出GDLNN，一种结合领域特定编程语言GDL与神经网络的新型图机器学习架构，用于图分类任务。GDLNN的核心优势在于其GDL层能够生成具有表现力和可解释性的图表示，在多个基准数据集上表现出色，且解释成本较低。

### 背景

图机器学习在图分类任务中面临可解释性和性能平衡的挑战，现有方法如GNNs虽然性能良好但解释性不足。

### 目的

开发一种新的图机器学习架构，既能保持高分类准确率，又能提供可解释的图表示，使现有模型解释技术可直接应用。

### 方法

提出GDLNN架构，结合名为GDL的领域特定编程语言与神经网络，核心是GDL层，用于生成表达性强且可解释的图表示。

### 主要发现

基于GDL的表示在大多数图分类基准数据集上实现高准确率，优于主流图学习方法如GNNs；应用现有模型解释技术能产生高质量的预测解释；包含解释成本时，GDLNN的成本较低。

### 结论

GDLNN通过结合领域特定编程语言与神经网络，成功实现了高准确率和良好可解释性的图分类，为图机器学习提供了新的有效方法。

### 翻译

我们提出了GDLNN，一种用于图分类任务的新型图机器学习架构。GDLNN将一种名为GDL的领域特定编程语言与神经网络相结合。GDLNN的主要优势在于其GDL层，能够生成具有表现力和可解释性的图表示。由于图表示具有可解释性，现有的模型解释技术可以直接应用于解释GDLNN的预测。我们的评估显示，基于GDL的表示在大多数图分类基准数据集上实现了高准确率，优于主流的图学习方法如GNNs。应用现有的模型解释技术也能产生高质量的GDLNN预测解释。此外，当包含解释成本时，GDLNN的成本较低。


### 论文摘要

We present GDLNN, a new graph machine learning architecture, for graph classification tasks. GDLNN combines a domain-specific programming language, called GDL, with neural networks. The main strength of GDLNN lies in its GDL layer, which generates expressive and interpretable graph representations. Since the graph representation is interpretable, existing model explanation techniques can be directly applied to explain GDLNN's predictions. Our evaluation shows that the GDL-based representation achieves high accuracy on most graph classification benchmark datasets, outperforming dominant graph learning methods such as GNNs. Applying an existing model explanation technique also yields high-quality explanations of GDLNN's predictions. Furthermore, the cost of GDLNN is low when the explanation cost is included.

---

## 11. SoREX: Towards Self-Explainable Social Recommendation with Relevant Ego-Path Extraction

**论文链接:** [http://arxiv.org/abs/2510.00080v1](http://arxiv.org/abs/2510.00080v1)

**作者:** Hanze Guo, Yijun Ma, Xiao Zhou

**发布时间:** 2025-09-30

**备注:** 27 pages, 10 figures

### GPT解析

### 总结

SoREX是一种自解释的基于图神经网络的社交推荐框架，通过双塔结构和朋友推荐增强，独立建模社交关系和用户-项交互，并提供新颖的自我路径提取方法来生成解释，在多个基准数据集上验证了其预测准确性和解释有效性。

### 背景

社交推荐已被证明可以通过利用社交网络解决用户-项交互建模中的数据稀疏性问题。近年来，图神经网络（GNNs）的整合提高了社交推荐算法的预测准确性，但许多基于GNN的社交推荐方法缺乏为其预测提供有意义解释的能力。

### 目的

引入SoREX，一种自解释的基于GNN的社交推荐框架，以解决现有方法缺乏预测解释能力的问题。

### 方法

SoREX采用由朋友推荐增强的双塔框架，独立建模社交关系和用户-项交互，同时联合优化辅助任务以强化社交信号。为提供解释，提出了一种新颖的自我路径提取方法，将目标用户的自我网络转换为多跳自我路径集合，提取特定因素和候选感知的自我路径子集作为解释，并进行解释重新聚合将解释与下游预测明确关联。

### 主要发现

在四个广泛采用的基准数据集上的实验验证了SoREX在预测准确性方面的有效性。定性和定量分析确认了SoREX中提取的解释的有效性。

### 结论

SoREX是一个有效的自解释GNN社交推荐框架，不仅具有高预测准确性，还能提供有意义的解释，使推荐结果更加透明和可信。

### 翻译

社交推荐已被证明通过利用社交网络解决用户-项交互建模中的数据稀疏性问题。最近图神经网络（GNNs）的整合进一步提高了当代社交推荐算法的预测准确性。然而，社交推荐中的许多基于GNN的方法缺乏为其预测提供有意义解释的能力。在本研究中，我们通过引入SoREX（一种自解释的基于GNN的社交推荐框架）来应对这一挑战。SoREX采用一个由朋友推荐增强的双塔框架，独立建模社交关系和用户-项交互，同时联合优化辅助任务以强化社交信号。为提供解释，我们提出了一种新颖的自我路径提取方法。此方法涉及将目标用户的自我网络转换为多跳自我路径集合，从中我们提取特定因素和候选感知的自我路径子集作为解释。这一过程通过复杂的子结构分析，促进了不同候选项目之间详细比较解释的总结。此外，我们进行了解释重新聚合，将解释与下游预测明确关联，赋予我们的框架内在的自解释能力。在四个广泛采用的基准数据集上进行的大量实验验证了SoREX在预测准确性方面的有效性。此外，定性和定量分析确认了SoREX中提取解释的有效性。我们的代码和数据可在https://github.com/antman9914/SoREX获取。


### 论文摘要

Social recommendation has been proven effective in addressing data sparsity in user-item interaction modeling by leveraging social networks. The recent integration of Graph Neural Networks (GNNs) has further enhanced prediction accuracy in contemporary social recommendation algorithms. However, many GNN-based approaches in social recommendation lack the ability to furnish meaningful explanations for their predictions. In this study, we confront this challenge by introducing SoREX, a self-explanatory GNN-based social recommendation framework. SoREX adopts a two-tower framework enhanced by friend recommendation, independently modeling social relations and user-item interactions, while jointly optimizing an auxiliary task to reinforce social signals. To offer explanations, we propose a novel ego-path extraction approach. This method involves transforming the ego-net of a target user into a collection of multi-hop ego-paths, from which we extract factor-specific and candidate-aware ego-path subsets as explanations. This process facilitates the summarization of detailed comparative explanations among different candidate items through intricate substructure analysis. Furthermore, we conduct explanation re-aggregation to explicitly correlate explanations with downstream predictions, imbuing our framework with inherent self-explainability. Comprehensive experiments conducted on four widely adopted benchmark datasets validate the effectiveness of SoREX in predictive accuracy. Additionally, qualitative and quantitative analyses confirm the efficacy of the extracted explanations in SoREX. Our code and data are available at https://github.com/antman9914/SoREX.

---

## 12. AGNOMIN -- Architecture Agnostic Multi-Label Function Name Prediction

**论文链接:** [http://arxiv.org/abs/2509.25514v2](http://arxiv.org/abs/2509.25514v2)

**作者:** Yonatan Gizachew Achamyeleh, Tongtao Zhang, Joshua Hyunki Kim, Gabriel Garcia, Shih-Yuan Yu, Anton Kocheturov, Mohammad Abdullah Al Faruque

**发布时间:** 2025-09-29

### GPT解析

### 总结

论文提出了AGNOMIN，一种用于剥离二进制文件中多标签函数名预测的新型架构无关方法。该方法通过构建功能增强的分层图和分层图神经网络，实现了跨架构一致的函数表示，显著提高了函数名预测的精度和召回率，并在实际安全应用中得到了验证。

### 背景

函数名预测对于理解软件逆向工程中的剥离二进制文件至关重要，是进行后续漏洞分析和修复的关键步骤。然而，现有方法通常面临特定架构限制、数据稀缺和多样化命名约定的挑战。

### 目的

提出一种名为AGNOMIN的新型架构无关方法，用于剥离二进制文件中的多标签函数名预测，以克服现有方法的局限性。

### 方法

AGNOMIN构建了功能增强的分层图（FEHGs），结合控制流图、函数调用图和动态学习的PCode特征。分层图神经网络处理这种增强结构，生成跨架构一致的函数表示。对于函数名预测，采用受Renée启发的解码器，增强了基于注意力的头部层和算法改进。

### 主要发现

在包含三种架构的9000个ELF可执行文件的综合性数据集上评估AGNOMIN，结果显示其性能优于最先进的方法，在测试数据集中精度提高了27.17%，召回率提高了55.86%。此外，AGNOMIN在未见过的架构上泛化良好，召回率比最接近的基线高出5.89%。

### 结论

AGNOMIN的实际效用已通过安全黑客马拉松得到验证，成功帮助逆向工程师分析和修补不同架构上的易受攻击的二进制文件，证明了其在实际安全应用中的价值。

### 翻译

函数名预测对于理解软件逆向工程中的剥离二进制文件至关重要，这是进行后续漏洞分析和修复的关键步骤。然而，现有方法通常面临特定架构限制、数据稀缺和多样化命名约定的挑战。我们提出了AGNOMIN，一种用于剥离二进制文件中多标签函数名预测的新型架构无关方法。AGNOMIN构建了功能增强的分层图（FEHGs），结合了控制流图、函数调用图和动态学习的PCode特征。分层图神经网络处理这种增强结构，以生成跨架构一致的函数表示，这对可扩展的安全评估至关重要。对于函数名预测，AGNOMIN采用了受Renée启发的解码器，增强了基于注意力的头部层和算法改进。我们在包含三种架构的9000个ELF可执行文件的综合性数据集上评估AGNOMIN，展示了其相比最先进方法的优越性能，在测试数据集中精度提高了27.17%，召回率提高了55.86%。此外，AGNOMIN在未见过的架构上泛化良好，召回率比最接近的基线高出5.89%。AGNOMIN的实际效用已通过安全黑客马拉松得到验证，成功帮助逆向工程师分析和修补不同架构上的易受攻击的二进制文件。


### 论文摘要

Function name prediction is crucial for understanding stripped binaries in software reverse engineering, a key step for \textbf{enabling subsequent vulnerability analysis and patching}. However, existing approaches often struggle with architecture-specific limitations, data scarcity, and diverse naming conventions. We present AGNOMIN, a novel architecture-agnostic approach for multi-label function name prediction in stripped binaries. AGNOMIN builds Feature-Enriched Hierarchical Graphs (FEHGs), combining Control Flow Graphs, Function Call Graphs, and dynamically learned \texttt{PCode} features. A hierarchical graph neural network processes this enriched structure to generate consistent function representations across architectures, vital for \textbf{scalable security assessments}. For function name prediction, AGNOMIN employs a Ren\'ee-inspired decoder, enhanced with an attention-based head layer and algorithmic improvements.   We evaluate AGNOMIN on a comprehensive dataset of 9,000 ELF executable binaries across three architectures, demonstrating its superior performance compared to state-of-the-art approaches, with improvements of up to 27.17\% in precision and 55.86\% in recall across the testing dataset. Moreover, AGNOMIN generalizes well to unseen architectures, achieving 5.89\% higher recall than the closest baseline. AGNOMIN's practical utility has been validated through security hackathons, where it successfully aided reverse engineers in analyzing and patching vulnerable binaries across different architectures.

---

## 13. Demystifying the Roles of LLM Layers in Retrieval, Knowledge, and Reasoning

**论文链接:** [http://arxiv.org/abs/2510.02091v1](http://arxiv.org/abs/2510.02091v1)

**作者:** Xinyuan Song, Keyu Wang, PengXiang Li, Lu Yin, Shiwei Liu

**发布时间:** 2025-10-02

**备注:** ICASSP 2025

### GPT解析

### 总结

本研究系统分析了大型语言模型不同深度的层在各种评估协议、任务类别和模型架构下的利用情况，发现深度利用具有高度异质性和上下文依赖性。

### 背景

最近的研究表明，大型语言模型的深层对表征学习的贡献很小，通常可以被移除而不会造成显著的性能损失。然而，这些说法通常是基于狭隘的评估得出的，可能忽略了模型行为的重要方面。

### 目的

对大型语言模型在不同维度上的深度利用进行系统研究，包括评估协议、任务类别和模型架构。

### 方法

通过分析不同评估协议下模型各层的表现，研究深度利用的异质性和上下文依赖性。

### 主要发现

1. 在基于似然的指标且不涉及生成的评估中，修剪大部分层可以保持性能，只有最初几个层是关键的；2. 在基于生成的评估中，中间层和深层在促进推理和保持长程连贯性中不可或缺；3. 知识和检索集中在浅层组件中，而推理准确性则严重依赖于深层，但可以通过蒸馏来重塑。

### 结论

大型语言模型中的深度利用高度异质且依赖于上下文，因此在解释和压缩大型模型时需要考虑任务、指标和模型感知的视角。

### 翻译

最近的研究表明，大型语言模型的深层对表征学习的贡献很小，通常可以被移除而不会造成显著的性能损失。然而，这些说法通常是基于狭隘的评估得出的，可能忽略了模型行为的重要方面。在这项工作中，我们对不同维度上的深度利用进行了系统研究，包括评估协议、任务类别和模型架构。我们的分析证实，非常深的层通常比早期层效果较差，但它们的贡献会随着评估设置而有显著变化。在基于似然的指标且不涉及生成的评估中，修剪大部分层可以保持性能，只有最初几个层是关键的。相比之下，基于生成的评估揭示了中间层和深层在促进推理和保持长程连贯性中不可或缺的作用。我们还发现，知识和检索集中在浅层组件中，而推理准确性则严重依赖于深层——但可以通过蒸馏来重塑。这些结果表明，大型语言模型中的深度利用高度异质且依赖于上下文，强调了在解释和压缩大型模型时需要考虑任务、指标和模型感知的视角。


### 论文摘要

Recent studies suggest that the deeper layers of Large Language Models (LLMs) contribute little to representation learning and can often be removed without significant performance loss. However, such claims are typically drawn from narrow evaluations and may overlook important aspects of model behavior. In this work, we present a systematic study of depth utilization across diverse dimensions, including evaluation protocols, task categories, and model architectures. Our analysis confirms that very deep layers are generally less effective than earlier ones, but their contributions vary substantially with the evaluation setting. Under likelihood-based metrics without generation, pruning most layers preserves performance, with only the initial few being critical. By contrast, generation-based evaluation uncovers indispensable roles for middle and deeper layers in enabling reasoning and maintaining long-range coherence. We further find that knowledge and retrieval are concentrated in shallow components, whereas reasoning accuracy relies heavily on deeper layers -- yet can be reshaped through distillation. These results highlight that depth usage in LLMs is highly heterogeneous and context-dependent, underscoring the need for task-, metric-, and model-aware perspectives in both interpreting and compressing large models.

---

## 14. FairContrast: Enhancing Fairness through Contrastive learning and Customized Augmenting Methods on Tabular Data

**论文链接:** [http://arxiv.org/abs/2510.02017v1](http://arxiv.org/abs/2510.02017v1)

**作者:** Aida Tayebi, Ali Khodabandeh Yalabadi, Mehdi Yazdani-Jahromi, Ozlem Ozmen Garibay

**发布时间:** 2025-10-02

**备注:** Accepted to NeurIPS 2025 - Reliable ML Workshop

### GPT解析

### 总结

研究提出了一种针对表格数据的对比学习框架，通过战略性地选择正样本对并结合监督和自监督对比学习，显著减少了偏见，同时保持了较高的准确率，并在各种下游任务中表现良好。

### 背景

随着AI系统日益融入日常生活，开发公平无偏的模型变得至关重要。学习公平稳健的表示已被证明是有效消除算法偏见并提高公平性的有力方法，尽管这些方法在表格数据应用中的公平性问题仍未得到充分探索。

### 目的

引入一个专门设计用于解决表格数据中偏见和学习公平表示的对比学习框架。

### 方法

通过战略性地选择正样本对，并结合监督和自监督对比学习技术。

### 主要发现

与现有表格数据对比学习模型相比，该方法显著减少了偏见，在最小化准确率权衡方面有效减轻偏见，且利用学习到的公平表示在各种下游任务中表现良好。

### 结论

对比学习框架能有效解决表格数据中的偏见问题，学习公平表示可在保持预测任务所需基本信息的同时提高算法公平性。

### 翻译

随着AI系统越来越融入日常生活，开发公平无偏的模型变得至关重要。考虑AI系统的社会影响不仅是技术挑战，也是道德义务。正如多项研究所示，学习公平稳健的表示已被证明是有效消除算法偏见并提高公平性的有力方法，同时保持预测任务所需的基本信息。表示学习框架，特别是利用自监督和对比学习的框架，已在各种领域表现出更强的稳健性和泛化能力。尽管越来越有兴趣将这些方法应用于表格数据，但这些学习表示中的公平性问题仍未得到充分探索。在本研究中，我们引入了一个专门设计用于解决表格数据中偏见和学习公平表示的对比学习框架。通过战略性地选择正样本对并采用监督和自监督对比学习，我们显著减少了与现有表格数据最先进对比学习模型相比的偏见。我们的结果证明了该方法在最小化准确率权衡方面减轻偏见的有效性，以及利用学习到的公平表示在各种下游任务中的优势。


### 论文摘要

As AI systems become more embedded in everyday life, the development of fair and unbiased models becomes more critical. Considering the social impact of AI systems is not merely a technical challenge but a moral imperative. As evidenced in numerous research studies, learning fair and robust representations has proven to be a powerful approach to effectively debiasing algorithms and improving fairness while maintaining essential information for prediction tasks. Representation learning frameworks, particularly those that utilize self-supervised and contrastive learning, have demonstrated superior robustness and generalizability across various domains. Despite the growing interest in applying these approaches to tabular data, the issue of fairness in these learned representations remains underexplored. In this study, we introduce a contrastive learning framework specifically designed to address bias and learn fair representations in tabular datasets. By strategically selecting positive pair samples and employing supervised and self-supervised contrastive learning, we significantly reduce bias compared to existing state-of-the-art contrastive learning models for tabular data. Our results demonstrate the efficacy of our approach in mitigating bias with minimum trade-off in accuracy and leveraging the learned fair representations in various downstream tasks.

---

## 15. Hybrid Quantum-Classical Walks for Graph Representation Learning in Community Detection

**论文链接:** [http://arxiv.org/abs/2510.01918v1](http://arxiv.org/abs/2510.01918v1)

**作者:** Adrián Marın, Mauricio Soto-Gomez, Giorgio Valentini, Elena Casiraghi, Carlos Cano, Daniel Manzano

**发布时间:** 2025-10-02

**备注:** 6 pages. Accepted at the 2025 IEEE International Conference on  Quantum Artificial Intelligence

### GPT解析

### 总结

本文提出了一种基于混合量子-经典行走的量子启发式图表示学习算法，用于解决传统方法在处理复杂图关系时的局限性。

### 背景

图表示学习已成为分析生物系统、社交网络和数据分析等领域复杂网络化数据的核心技术，但传统方法难以捕捉具有幂律分布或层次结构等非平凡结构特性的复杂图中的关系。

### 目的

开发一种新型量子启发式算法，以克服传统图表示学习方法在捕捉复杂图内 intricate 关系方面的局限性。

### 方法

提出利用混合量子-经典行走的算法，结合量子和经典动力学的优势，使行走者能够同时探索图中的高度局部和远距离连接。

### 主要发现

网络社区检测案例研究的初步结果表明，这种混合动态使算法能够有效适应复杂的图拓扑结构。

### 结论

该混合量子-经典行走算法为图表示学习任务提供了强大且通用的解决方案，特别适用于处理具有复杂结构特性的图。

### 翻译

图表示学习已成为分析生物系统、社交网络和数据分析等不同领域中复杂网络化数据的核心技术。传统图表示学习方法通常难以捕捉复杂图中的 intricate 关系，特别是那些表现出幂律分布或层次结构等非平凡结构特性的图。本文介绍了一种用于图表示学习的量子启发式新算法，利用混合量子-经典行走来克服这些局限性。我们的方法结合了量子和经典动力学的优点，使行走者能够同时探索图中的高度局部和远距离连接。在网络社区检测的案例研究中，初步结果表明这种混合动态使算法能够有效适应复杂的图拓扑结构，为图表示学习任务提供了强大且通用的解决方案。


### 论文摘要

Graph Representation Learning (GRL) has emerged as a cornerstone technique for analysing complex, networked data across diverse domains, including biological systems, social networks, and data analysis. Traditional GRL methods often struggle to capture intricate relationships within complex graphs, particularly those exhibiting non-trivial structural properties such as power-law distributions or hierarchical structures. This paper introduces a novel quantum-inspired algorithm for GRL, utilizing hybrid Quantum-Classical Walks to overcome these limitations. Our approach combines the benefits of both quantum and classical dynamics, allowing the walker to simultaneously explore both highly local and far-reaching connections within the graph. Preliminary results for a case study in network community detection shows that this hybrid dynamic enables the algorithm to adapt effectively to complex graph topologies, offering a robust and versatile solution for GRL tasks.

---

## 16. Learning Representations Through Contrastive Neural Model Checking

**论文链接:** [http://arxiv.org/abs/2510.01853v1](http://arxiv.org/abs/2510.01853v1)

**作者:** Vladimir Krsmanovic, Matthias Cosler, Mohamed Ghanem, Bernd Finkbeiner

**发布时间:** 2025-10-02

### GPT解析

### 总结

本文提出了一种对比神经模型检查(CNML)方法，将模型检查任务作为指导信号来学习对齐表征，在形式验证领域探索了表征学习的应用。

### 背景

模型检查是验证安全关键系统符合形式规范的关键技术，最近深度学习的应用显示出了前景。然而，表征学习在形式验证领域仍然探索不足。

### 目的

利用模型检查任务作为学习对齐表征的指导信号，探索表征学习在形式验证领域的应用。

### 方法

提出对比神经模型检查(CNML)方法，通过自监督对比目标将逻辑规范和系统共同嵌入到一个共享的潜在空间中。

### 主要发现

在受工业启发的检索任务中，CNML在跨模态和模态内部设置中都明显优于算法和神经基线；学习到的表征能有效地迁移到下游任务并推广到更复杂的公式。

### 结论

模型检查可以作为学习形式语言表征的目标。

### 翻译

模型检查是验证安全关键系统是否符合形式规范的关键技术，最近深度学习的应用显示出了前景。然而，尽管表征学习在视觉和语言领域无处不在，但在形式验证领域仍然探索不足。作者引入了对比神经模型检查(CNML)这一新方法，利用模型检查任务作为学习对齐表征的指导信号。CNML通过自监督对比目标将逻辑规范和系统共同嵌入到一个共享的潜在空间中。在受工业启发的检索任务中，CNML在跨模态和模态内部设置中都明显优于算法和神经基线。作者进一步展示了学习到的表征有效地迁移到下游任务并推广到更复杂的公式。这些发现表明模型检查可以作为学习形式语言表征的目标。


### 论文摘要

Model checking is a key technique for verifying safety-critical systems against formal specifications, where recent applications of deep learning have shown promise. However, while ubiquitous for vision and language domains, representation learning remains underexplored in formal verification. We introduce Contrastive Neural Model Checking (CNML), a novel method that leverages the model checking task as a guiding signal for learning aligned representations. CNML jointly embeds logical specifications and systems into a shared latent space through a self-supervised contrastive objective. On industry-inspired retrieval tasks, CNML considerably outperforms both algorithmic and neural baselines in cross-modal and intra-modal settings.We further show that the learned representations effectively transfer to downstream tasks and generalize to more complex formulas. These findings demonstrate that model checking can serve as an objective for learning representations for formal languages.

---

## 17. Contrastive Representation Regularization for Vision-Language-Action Models

**论文链接:** [http://arxiv.org/abs/2510.01711v1](http://arxiv.org/abs/2510.01711v1)

**作者:** Taeyoung Kim, Jimin Lee, Myungkyu Koo, Dongyoung Kim, Kyungmin Lee, Changyeon Kim, Younggyo Seo, Jinwoo Shin

**发布时间:** 2025-10-02

**备注:** 20 pages, 12 figures

### GPT解析

### 总结

研究提出了一种名为Robot State-aware Contrastive Loss (RS-CL)的表示正则化方法，用于增强VLA模型在机器人操作中的性能，通过将视觉语言模型表示与机器人信号对齐来提高操作准确性。

### 背景

VLA模型已经通过利用预训练视觉语言模型的丰富表示展示了在机器人操作方面的能力，但它们的表示仍然不够优化，缺乏对机器人信号（如控制动作和本体感受状态）的敏感性。

### 目的

解决VLA模型表示不够优化的问题，通过引入RS-CL来弥合视觉语言模型表示与机器人信号之间的差距，使表示更接近机器人的本体感受状态。

### 方法

提出了Robot State-aware Contrastive Loss (RS-CL)，一种简单有效的VLA模型表示正则化方法。它使用状态之间的相对距离作为软监督，将表示更紧密地与机器人的本体感受状态对齐。RS-CL补充了原始的动作预测目标，同时保持轻量级且完全兼容标准的VLA训练流程。

### 主要发现

RS-CL显著提高了最先进VLA模型的操作性能；在RoboCasa-Kitchen的拾取和放置任务中，将最先进的结果从30.8%提高到41.5%，通过在抓取和放置过程中实现更精确的定位；在具有挑战性的真实机器人操作任务中，将成功率从45.0%提高到58.3%。

### 结论

RS-CL是一种简单而有效的方法，能够增强控制相关表示的学习，显著提升VLA模型在机器人操作任务中的性能，同时保持轻量级和与标准训练流程的兼容性。

### 翻译

视觉-语言-动作模型已经通过利用预训练视觉语言模型的丰富表示展示了在机器人操作方面的能力。然而，它们的表示被认为仍然不够优化，缺乏对机器人信号（如控制动作和本体感受状态）的敏感性。为了解决这个问题，我们引入了机器人状态感知对比损失，这是一种简单有效的视觉-语言-动作模型表示正则化方法，旨在弥合视觉语言模型表示与机器人信号之间的差距。特别是，该损失通过使用状态之间的相对距离作为软监督，使表示更紧密地与机器人的本体感受状态对齐。除了原始的动作预测目标外，机器人状态感知对比损失有效地增强了控制相关表示的学习，同时保持轻量级并完全兼容标准的视觉-语言-动作训练流程。我们的实证结果表明，机器人状态感知对比损失显著提高了最先进视觉-语言-动作模型的操作性能；它通过在抓取和放置过程中实现更精确的定位，将RoboCasa-Kitchen中拾取和放置任务的最先进结果从30.8%提高到41.5%，并将具有挑战性的真实机器人操作任务的成功率从45.0%提高到58.3%。


### 论文摘要

Vision-Language-Action (VLA) models have shown its capabilities in robot manipulation by leveraging rich representations from pre-trained Vision-Language Models (VLMs). However, their representations arguably remain suboptimal, lacking sensitivity to robotic signals such as control actions and proprioceptive states. To address the issue, we introduce Robot State-aware Contrastive Loss (RS-CL), a simple and effective representation regularization for VLA models, designed to bridge the gap between VLM representations and robotic signals. In particular, RS-CL aligns the representations more closely with the robot's proprioceptive states, by using relative distances between the states as soft supervision. Complementing the original action prediction objective, RS-CL effectively enhances control-relevant representation learning, while being lightweight and fully compatible with standard VLA training pipeline. Our empirical results demonstrate that RS-CL substantially improves the manipulation performance of state-of-the-art VLA models; it pushes the prior art from 30.8% to 41.5% on pick-and-place tasks in RoboCasa-Kitchen, through more accurate positioning during grasping and placing, and boosts success rates from 45.0% to 58.3% on challenging real-robot manipulation tasks.

---

## 18. Discrete Facial Encoding: : A Framework for Data-driven Facial Display Discovery

**论文链接:** [http://arxiv.org/abs/2510.01662v1](http://arxiv.org/abs/2510.01662v1)

**作者:** Minh Tran, Maksim Siniukov, Zhangyu Jin, Mohammad Soleymani

**发布时间:** 2025-10-02

### GPT解析

### 总结

本文提出了一种名为离散面部编码(DFE)的无监督、数据驱动方法，用于从3D网格序列中学习面部表情的紧凑且可解释的字典。

### 背景

面部表情分析对理解人类行为至关重要，但现有的面部动作编码系统(FACS)存在覆盖范围有限和人工标注成本高的问题。

### 目的

开发一种替代方案，克服FACS的局限性，提供更精确、更高效的面部表情分析方法。

### 方法

使用3D Morphable Model提取身份不变的表情特征，然后通过Residual Vector Quantized Variational Autoencoder(RVQ-VAE)编码这些特征，生成离散令牌序列，每个令牌捕捉特定的面部变形模式。

### 主要发现

DFE比FACS和其他面部编码替代方案捕获更精确的面部行为；在压力检测、性格预测和抑郁检测三个心理任务上表现优于基于FACS的流程和先进的图像视频表征模型；覆盖更广泛的面部表情显示。

### 结论

DFE是FACS在心理和情感计算应用中的一种可扩展且有效的替代方案。

### 翻译

面部表情分析对理解人类行为至关重要，但现有的编码系统如面部动作编码系统(FACS)受限于有限的覆盖范围和昂贵的手工标注。在这项工作中，我们引入了离散面部编码(DFE)，这是一种无监督、数据驱动的替代方案，通过残差向量量化变分自编码器(RVQ-VAE)从3D网格序列中学习紧凑且可解释的面部表情字典。我们的方法首先使用3D可变形模型(3DMM)从图像中提取身份不变的表情特征，有效地分离了头部姿态和面部几何等因素。然后我们使用RVQ-VAE对这些特征进行编码，从共享码本中生成离散令牌序列，其中每个令牌捕获特定的、可重用的面部变形模式，这些模式共同构成整体表情。通过大量实验，我们证明离散面部编码比FACS和其他面部编码替代方案捕获更精确的面部行为。我们在三个高级心理任务上评估了我们表征的实用性：压力检测、性格预测和抑郁检测。使用在学习的令牌之上构建的简单词袋模型，我们的系统持续优于基于FACS的流程以及强大的图像和视频表征学习模型，如掩码自编码器。进一步的分析显示，我们的表征覆盖了更广泛的面部表情显示，突显了其作为心理和情感计算应用中FACS的可扩展且有效替代方案的潜力。


### 论文摘要

Facial expression analysis is central to understanding human behavior, yet existing coding systems such as the Facial Action Coding System (FACS) are constrained by limited coverage and costly manual annotation. In this work, we introduce Discrete Facial Encoding (DFE), an unsupervised, data-driven alternative of compact and interpretable dictionary of facial expressions from 3D mesh sequences learned through a Residual Vector Quantized Variational Autoencoder (RVQ-VAE). Our approach first extracts identity-invariant expression features from images using a 3D Morphable Model (3DMM), effectively disentangling factors such as head pose and facial geometry. We then encode these features using an RVQ-VAE, producing a sequence of discrete tokens from a shared codebook, where each token captures a specific, reusable facial deformation pattern that contributes to the overall expression. Through extensive experiments, we demonstrate that Discrete Facial Encoding captures more precise facial behaviors than FACS and other facial encoding alternatives. We evaluate the utility of our representation across three high-level psychological tasks: stress detection, personality prediction, and depression detection. Using a simple Bag-of-Words model built on top of the learned tokens, our system consistently outperforms both FACS-based pipelines and strong image and video representation learning models such as Masked Autoencoders. Further analysis reveals that our representation covers a wider variety of facial displays, highlighting its potential as a scalable and effective alternative to FACS for psychological and affective computing applications.

---

## 19. Self-Supervised Representation Learning as Mutual Information Maximization

**论文链接:** [http://arxiv.org/abs/2510.01345v1](http://arxiv.org/abs/2510.01345v1)

**作者:** Akhlaqur Rahman Sabby, Yi Sui, Tongzi Wu, Jesse C. Cresswell, Ga Wu

**发布时间:** 2025-10-01

### GPT解析

### 总结

该研究从第一性原理出发，探讨自监督表示学习(SSRL)算法的学习目标如何决定其优化策略和模型设计选择，通过变分互信息下界推导出SDMI和JMI两种训练范式，为现有SSRL方法的架构组件选择提供理论解释。

### 背景

自监督表示学习已取得显著成功但基本原理尚未充分理解，现有研究多从信息论目标或防止表示崩溃的启发式方法角度统一SSRL方法，而架构元素如预测器网络、stop-gradient操作和统计正则化器常被视为经验驱动的补充。

### 目的

采用第一性原理方法，探究SSRL算法的学习目标是否决定其可能的优化策略和模型设计选择。

### 方法

从变分互信息(MI)下界出发，推导出自蒸馏互信息(SDMI)和联合互信息(JMI)两种训练范式，分析它们施加的不同结构约束及其与现有SSRL算法的关系。

### 主要发现

SDMI需要交替优化使stop-gradient操作理论上成为必需；JMI可通过对称架构进行联合优化无需此类组件；预测器网络和统计正则化器分别是MI目标的可替代代理；许多现有SSRL方法可视为这两种范式的特定实例或近似。

### 结论

本文为现有SSRL方法不同架构组件的选择提供了超越启发式便利性的理论解释。

### 翻译

自监督表示学习(SSRL)已经取得了显著的实证成功，但其基本原理仍未被充分理解。虽然最近的工作试图通过检查其信息论目标或总结防止表示崩溃的启发式方法来统一SSRL方法，但预测器网络、stop-gradient操作和统计正则化器等架构元素通常被视为经验驱动的补充。在本文中，我们采用第一性原理方法，研究SSRL算法的学习目标是否决定其可能的优化策略和模型设计选择。特别是，从变分互信息(MI)下界出发，我们推导出两种训练范式，即自蒸馏互信息(SDMI)和联合互信息(JMI)，每种范式施加不同的结构约束并涵盖一系列现有的SSRL算法。SDMI本质上需要交替优化，使stop-gradient操作在理论上成为必需。相比之下，JMI可以通过对称架构进行联合优化，无需此类组件。在提出的公式中，SDMI中的预测器网络和JMI中的统计正则化器成为MI目标的可替代代理。我们表明，许多现有的SSRL方法都是这两种范式的特定实例或近似。本文为现有SSRL方法不同架构组件的选择提供了理论解释，超越了启发式便利性的范畴。


### 论文摘要

Self-supervised representation learning (SSRL) has demonstrated remarkable empirical success, yet its underlying principles remain insufficiently understood. While recent works attempt to unify SSRL methods by examining their information-theoretic objectives or summarizing their heuristics for preventing representation collapse, architectural elements like the predictor network, stop-gradient operation, and statistical regularizer are often viewed as empirically motivated additions. In this paper, we adopt a first-principles approach and investigate whether the learning objective of an SSRL algorithm dictates its possible optimization strategies and model design choices. In particular, by starting from a variational mutual information (MI) lower bound, we derive two training paradigms, namely Self-Distillation MI (SDMI) and Joint MI (JMI), each imposing distinct structural constraints and covering a set of existing SSRL algorithms. SDMI inherently requires alternating optimization, making stop-gradient operations theoretically essential. In contrast, JMI admits joint optimization through symmetric architectures without such components. Under the proposed formulation, predictor networks in SDMI and statistical regularizers in JMI emerge as tractable surrogates for the MI objective. We show that many existing SSRL methods are specific instances or approximations of these two paradigms. This paper provides a theoretical explanation behind the choices of different architectural components of existing SSRL methods, beyond heuristic conveniences.

---

## 20. Deep Learning-Based Approach for Improving Relational Aggregated Search

**论文链接:** [http://arxiv.org/abs/2510.00966v1](http://arxiv.org/abs/2510.00966v1)

**作者:** Sara Saad Soliman, Ahmed Younes, Islam Elkabani, Ashraf Elsayed

**发布时间:** 2025-10-01

### GPT解析

### 总结

本研究探讨了高级自然语言处理技术在阿拉伯语文本数据聚类中的应用，特别是堆叠自编码器和AraBERT嵌入技术，以改善聚合搜索环境中的搜索结果质量和相关性。

### 背景

互联网信息爆炸导致需要开发能够提升各种格式内容检索和管理的聚合搜索系统。传统搜索引擎存在不精确、缺乏上下文相关性和个性化的问题。

### 目的

改进阿拉伯语文本数据在聚合搜索环境中的聚类效果，提供更丰富、具有上下文感知能力的搜索结果表征。

### 方法

研究应用了堆叠自编码器和AraBERT嵌入等高级自然语言处理技术，并使用K-means聚类算法来发现搜索结果中的特征和关系。通过不同阿拉伯语查询评估了该方法的有效性。

### 主要发现

堆叠自编码器在表示学习中适合聚类任务，能够显著改善聚类搜索结果。该方法提高了搜索结果的准确性和相关性。

### 结论

通过超越传统搜索引擎的局限性，该研究提供了一种更有效的方法来聚类和表征阿拉伯语文本搜索结果，显著提升了搜索质量和用户体验。

### 翻译

由于互联网上的信息爆炸，需要开发聚合搜索系统来提升各种格式内容的检索和管理。为了进一步改进聚合搜索环境中阿拉伯语文本数据的聚类，本研究调查了高级自然语言处理技术的应用，即堆叠自编码器和AraBERT嵌入。通过超越传统搜索引擎的不精确、缺乏上下文相关性和个性化等局限，我们提供了更丰富、具有上下文感知能力的搜索结果表征，因此我们使用K-means聚类算法来发现这些结果中的特征和关系，然后在不同阿拉伯语查询上使用我们的方法评估其有效性。我们的模型表明，在表示学习中使用堆叠自编码器适合聚类任务，并能显著改善聚类搜索结果。它还展示了搜索结果的准确性和相关性的提高。


### 论文摘要

Due to an information explosion on the internet, there is a need for the development of aggregated search systems that can boost the retrieval and management of content in various formats. To further improve the clustering of Arabic text data in aggregated search environments, this research investigates the application of advanced natural language processing techniques, namely stacked autoencoders and AraBERT embeddings. By transcending the limitations of traditional search engines, which are imprecise, not contextually relevant, and not personalized, we offer more enriched, context-aware characterizations of search results, so we used a K-means clustering algorithm to discover distinctive features and relationships in these results, we then used our approach on different Arabic queries to evaluate its effectiveness. Our model illustrates that using stacked autoencoders in representation learning suits clustering tasks and can significantly improve clustering search results. It also demonstrates improved accuracy and relevance of search results.

---

## 21. LLM Routing with Dueling Feedback

**论文链接:** [http://arxiv.org/abs/2510.00841v1](http://arxiv.org/abs/2510.00841v1)

**作者:** Chao-Kai Chiang, Takashi Ishida, Masashi Sugiyama

**发布时间:** 2025-10-01

### GPT解析

### 总结

本研究提出了一种基于上下文对决老虎机的LLM路由方法，通过成对偏好反馈学习，并引入了CCFT表示学习方法和FGTS.CDB算法，实现了在用户满意度、模型专业性和推理成本之间的平衡。

### 背景

在大型语言模型(LLM)应用中，为每个查询选择最佳模型是一个关键挑战，需要同时考虑用户满意度、模型专业性和推理成本等因素。

### 目的

开发一种能够有效选择最佳LLM的路由方法，平衡用户满意度、模型专业性和推理成本，并通过成对偏好反馈实现标签效率和动态适应。

### 方法

将路由问题表述为上下文对决老虎机，引入Category-Calibrated Fine-Tuning (CCFT)表示学习方法，使用带类别权重的对比微从未标记数据中导出模型嵌入，并实现Feel-Good Thompson Sampling for Contextual Dueling Bandits (FGTS.CDB)算法，提出了四种整合模型质量和成本的类别权重变体。

### 主要发现

在RouterBench和MixInstruct数据集上的实验表明，所提出的方法实现了比强基线更低的累积遗憾和更快的收敛速度，具有更好的鲁棒性和性能-成本平衡。

### 结论

基于上下文对决老虎机和CCFT的LLM路由方法能够有效平衡用户满意度、模型专业性和推理成本，是一种标签效率和动态适应的解决方案。

### 翻译

我们研究LLM路由问题，即在平衡用户满意度、模型专业性和推理成本的同时为每个查询选择最佳模型。我们将路由表述为上下文对决老虎机，从成对偏好反馈而非绝对分数中学习，从而实现标签效率和动态适应。基于这一表述，我们引入了类别校准微调(CCFT)，一种表示学习方法，使用带类别权重的对比从未标记数据中导出模型嵌入。这些嵌入使Feel-Good Thompson Sampling for Contextual Dueling Bandits(FGTS.CDB)成为可能，这是一个理论上合理的后验采样算法。我们提出了四种明确整合模型质量和成本的类别权重变体，并在RouterBench和MixInstruct数据集上对所提出的方法进行了实证评估。在两个基准测试中，我们的方法实现了比使用通用OpenAI嵌入模型构建的强基线更低的累积遗憾和更快的收敛速度，具有更好的鲁棒性和性能-成本平衡。


### 论文摘要

We study LLM routing, the problem of selecting the best model for each query while balancing user satisfaction, model expertise, and inference cost. We formulate routing as contextual dueling bandits, learning from pairwise preference feedback rather than absolute scores, thereby yielding label-efficient and dynamic adaptation. Building on this formulation, we introduce Category-Calibrated Fine-Tuning (CCFT), a representation-learning method that derives model embeddings from offline data using contrastive fine-tuning with categorical weighting. These embeddings enable the practical instantiation of Feel-Good Thompson Sampling for Contextual Dueling Bandits (FGTS.CDB), a theoretically grounded posterior-sampling algorithm. We propose four variants of the categorical weighting that explicitly integrate model quality and cost, and we empirically evaluate the proposed methods on the RouterBench and MixInstruct datasets. Across both benchmarks, our methods achieve lower cumulative regret and faster convergence, with better robustness and performance-cost balance than strong baselines built with a general-purpose OpenAI embedding model.

---

## 22. FAME: Adaptive Functional Attention with Expert Routing for Function-on-Function Regression

**论文链接:** [http://arxiv.org/abs/2510.00621v1](http://arxiv.org/abs/2510.00621v1)

**作者:** Yifei Gao, Yong Chen, Chen Zhang

**发布时间:** 2025-10-01

### GPT解析

### 总结

FAME是一种创新的函数数据处理方法，结合了注意力机制和专家混合模型，能够有效处理函数数据的连续性和函数间依赖关系。

### 背景

函数数据在科学和工程中起着关键作用，但其无限维特性使得表示学习具有挑战性。传统统计模型依赖于预选择的基展开或核函数，限制了数据驱动发现的灵活性；而许多深度学习方法将函数视为固定网格向量，忽略了固有的连续性。

### 目的

引入一种端到端的、完全数据驱动的框架，用于函数对函数回归任务。

### 方法

提出Functional Attention with a Mixture-of-Experts (FAME)方法，通过双向神经控制微分方程与MoE驱动的向量场耦合形成连续注意力以捕获函数内连续性，并通过多头交叉注意力进一步融合变化以捕获函数间依赖关系。

### 主要发现

在合成和真实世界的函数回归基准测试中，FAME实现了最先进的准确性，并且对函数的任意采样离散观测具有强大的鲁棒性。

### 结论

FAME方法为函数数据表示学习提供了一个有效的解决方案，能够处理函数的连续性和函数间依赖关系。

### 翻译

函数数据在科学和工程中起着关键作用，但其无限维特性使得表示学习具有挑战性。传统统计模型依赖于预选择的基展开或核函数，限制了数据驱动发现的灵活性，而许多深度学习管道将函数视为固定网格向量，忽略了固有的连续性。在本文中，我们引入了具有专家混合的函数注意力(FAME)，这是一种用于函数对函数回归的端到端、完全数据驱动框架。FAME通过双向神经控制微分方程与MoE驱动的向量场耦合形成连续注意力，以捕获函数内连续性，并通过多头交叉注意力进一步融合变化以捕获函数间依赖关系。在合成和真实世界函数回归基准上的广泛实验表明，FAME实现了最先进的准确性，并对函数的任意采样离散观测具有强大的鲁棒性。


### 论文摘要

Functional data play a pivotal role across science and engineering, yet their infinite-dimensional nature makes representation learning challenging. Conventional statistical models depend on pre-chosen basis expansions or kernels, limiting the flexibility of data-driven discovery, while many deep-learning pipelines treat functions as fixed-grid vectors, ignoring inherent continuity. In this paper, we introduce Functional Attention with a Mixture-of-Experts (FAME), an end-to-end, fully data-driven framework for function-on-function regression. FAME forms continuous attention by coupling a bidirectional neural controlled differential equation with MoE-driven vector fields to capture intra-functional continuity, and further fuses change to inter-functional dependencies via multi-head cross attention. Extensive experiments on synthetic and real-world functional-regression benchmarks show that FAME achieves state-of-the-art accuracy, strong robustness to arbitrarily sampled discrete observations of functions.

---

## 23. VIRTUE: Visual-Interactive Text-Image Universal Embedder

**论文链接:** [http://arxiv.org/abs/2510.00523v1](http://arxiv.org/abs/2510.00523v1)

**作者:** Wei-Yao Wang, Kazuya Tateishi, Qiyu Wu, Shusuke Takahashi, Yuki Mitsufuji

**发布时间:** 2025-10-01

**备注:** 25 pages

### GPT解析

### 总结

本文提出了一种新颖的视觉交互式文本图像通用嵌入器（VIRTUE），它将分割模型和视觉语言模型的能力扩展到表示学习领域，使嵌入模型能够处理视觉提示并精确定位图像中的特定区域，从而在复杂和模糊场景中实现更精确的处理。

### 背景

多模态表示学习模型在复杂任务中表现出色，视觉语言模型（VLMs）的集成为嵌入模型提供了指令遵循能力。然而，现有嵌入模型缺乏视觉交互能力，无法处理用户指定的感兴趣区域（如点、边界框、掩码），而这些能力在生成模型中已被探索以扩大人机交互适用性。

### 目的

为嵌入模型配备视觉交互能力，解锁基于用户意图局部化的新应用，并使模型能够学习图像中的实体级信息，以补充传统嵌入任务的全局表示。

### 方法

提出VIRTUE模型，将分割模型和视觉语言模型的能力扩展到表示学习领域。分割模型可以处理视觉提示，精确定位图像中的特定区域，使嵌入器能够更精确地处理复杂和模糊的场景。

### 主要发现

引入了一个大规模分割和场景标题检索（SCaR）基准，包含100万个样本，旨在通过同时考虑特定对象的实体和图像场景来检索文本标题。VIRTUE在36个通用MMEB任务上取得了显著改进（3.1%-8.5%），在五个视觉交互式SCaR任务上也取得了显著改进（15.2%-20.3%）。

### 结论

VIRTUE成功地将视觉交互能力集成到嵌入模型中，使模型能够更好地处理复杂场景和用户指定的感兴趣区域，并在多个任务上取得了最先进的性能。

### 翻译

多模态表示学习模型已在复杂任务中展现出成功操作，并且视觉语言模型（VLMs）的集成进一步使嵌入模型具备了指令遵循能力。然而，现有的嵌入模型缺乏视觉交互能力，无法处理用户指定的感兴趣区域（例如点、边界框、掩码），这些能力在生成模型中已被探索，以扩大其人机交互适用性。为嵌入模型配备视觉交互能力不仅能够解锁基于用户意图局部化的新应用，这些应用尚未被探索，而且还能使模型学习图像中的实体级信息，以补充其在传统嵌入任务中的全局表示。在本文中，我们提出了一种新颖的视觉交互式文本图像通用嵌入器（VIRTUE），它将分割模型和视觉语言模型的能力扩展到表示学习领域。在VIRTUE中，分割模型可以处理视觉提示，精确定位图像中的特定区域，从而使嵌入器能够更精确地处理复杂和模糊的场景。为了评估VIRTUE的视觉交互能力，我们引入了一个包含100万个样本的大规模分割和场景标题检索（SCaR）基准，旨在通过同时考虑特定对象的实体和图像场景来检索文本标题。VIRTUE在36个通用MMEB任务（3.1%-8.5%）和五个视觉交互式SCaR任务（15.2%-20.3%）上持续取得了最先进的性能。


### 论文摘要

Multimodal representation learning models have demonstrated successful operation across complex tasks, and the integration of vision-language models (VLMs) has further enabled embedding models with instruction-following capabilities. However, existing embedding models lack visual-interactive capabilities to specify regions of interest from users (e.g., point, bounding box, mask), which have been explored in generative models to broaden their human-interactive applicability. Equipping embedding models with visual interactions not only would unlock new applications with localized grounding of user intent, which remains unexplored, but also enable the models to learn entity-level information within images to complement their global representations for conventional embedding tasks. In this paper, we propose a novel Visual-InteRactive Text-Image Universal Embedder (VIRTUE) that extends the capabilities of the segmentation model and the vision-language model to the realm of representation learning. In VIRTUE, the segmentation model can process visual prompts that pinpoint specific regions within an image, thereby enabling the embedder to handle complex and ambiguous scenarios more precisely. To evaluate the visual-interaction ability of VIRTUE, we introduce a large-scale Segmentation-and-Scene Caption Retrieval (SCaR) benchmark comprising 1M samples that aims to retrieve the text caption by jointly considering the entity with a specific object and image scene. VIRTUE consistently achieves a state-of-the-art performance with significant improvements across 36 universal MMEB (3.1%-8.5%) and five visual-interactive SCaR (15.2%-20.3%) tasks.

---

## 24. Cutting the Skip: Training Residual-Free Transformers

**论文链接:** [http://arxiv.org/abs/2510.00345v1](http://arxiv.org/abs/2510.00345v1)

**作者:** Yiping Ji, James Martens, Jianqiao Zheng, Ziqin Zhou, Peyman Moghadam, Xinyu Zhang, Hemanth Saratchandran, Simon Lucey

**发布时间:** 2025-09-30

### GPT解析

### 总结

本文提出了一种方法，使无跳跃连接的transformers能够稳定高效地训练，证明了跳跃连接并非训练ViTs的基本要求。

### 背景

Transformers在各种应用中取得了显著成功，通常归因于其可扩展性。然而，没有跳跃（残差）连接的训练仍然非常困难。

### 目的

解决transformers在没有跳跃连接情况下的训练问题。

### 方法

通过分析无跳跃transformer块的雅可比矩阵，解释跳跃连接如何改善条件，并揭示可以通过原则性初始化策略恢复跳跃连接的稳定益处。基于此，引入了第一种能够在不改变标准架构的情况下实现无跳跃transformers稳定高效训练的方法。

### 主要发现

使用其初始化方法训练的无跳跃ViTs克服了通常的优化障碍，学习了更丰富的层次化表示，并在密集预测基准上优于包含跳跃连接的强基线模型。

### 结论

跳跃连接不是训练ViTs的基本要求，并为视觉模型中的层次化表示学习开辟了新的途径。

### 翻译

Transformers在广泛的应用中取得了显著成功，这一成就通常归因于其可扩展性。然而，没有跳跃（残差）连接的训练仍然非常困难。虽然跳跃连接稳定了优化过程，但它们也破坏了表示的层次结构，引发了一个长期存在的问题：transformers是否可以在没有跳跃连接的情况下被有效训练。在本工作中，我们通过分析无跳跃transformer块的雅可比矩阵来解决这个问题，解释了跳跃连接如何改善条件，并揭示可以通过一种原则性的初始化策略来恢复跳跃连接的稳定益处。基于这一见解，我们引入了第一种方法，可以在不改变标准架构的情况下实现无跳跃transformers的稳定高效训练。我们在监督和自监督设置下的Vision Transformers (ViTs)上验证了我们的方法，证明使用我们的初始化方法训练的无跳跃ViTs克服了通常的优化障碍，学习了更丰富的层次化表示，并在密集预测基准上优于包含跳跃连接的强基线模型。这些结果表明，跳跃连接不是训练ViTs的基本要求，并为视觉模型中的层次化表示学习开辟了新的途径。


### 论文摘要

Transformers have achieved remarkable success across a wide range of applications, a feat often attributed to their scalability. Yet training them without skip (residual) connections remains notoriously difficult. While skips stabilize optimization, they also disrupt the hierarchical structure of representations, raising the long-standing question of whether transformers can be trained efficiently without them. In this work, we address this problem by analyzing the Jacobian of a skipless transformer block, showing why skips improve conditioning and revealing that their stabilization benefits can be recovered through a principled initialization strategy. Building on this insight, we introduce the first method that enables stable and efficient training of skipless transformers without altering the standard architecture. We validate our approach on Vision Transformers (ViTs) in both supervised and self-supervised settings, demonstrating that skipless ViTs trained with our initialization overcome the usual optimization barriers, learn richer hierarchical representations, and outperform strong baselines, that incorporate skip connections, on dense prediction benchmarks. These results show that skip connections are not a fundamental requirement for training ViTs and open new avenues for hierarchical representation learning in vision models.

---

## 25. Looking Beyond the Known: Towards a Data Discovery Guided Open-World Object Detection

**论文链接:** [http://arxiv.org/abs/2510.00303v1](http://arxiv.org/abs/2510.00303v1)

**作者:** Anay Majee, Amitesh Gangrade, Rishabh Iyer

**发布时间:** 2025-09-30

**备注:** Accepted to NeurIPS'25. 22 pages, 6 figures

### GPT解析

### 总结

论文提出了名为CROWD的组合开放世界检测框架，解决了现有OWOD方法中的语义混淆和灾难性遗忘问题，显著提高了已知类准确率和未知召回率。

### 背景

开放世界目标检测(OWOD)通过人工指导能够持续发现和集成未知目标来丰富传统目标检测器。然而，现有的OWOD方法经常面临已知类和未知类之间的语义混淆，以及灾难性遗忘问题，导致未知目标召回率下降和已知类准确率降低。

### 目的

克服OWOD方法中的语义混淆和灾难性遗忘挑战，提高未知目标召回率和已知类准确率。

### 方法

提出CROWD统一框架，将未知目标发现和适应重新表述为交织的组合数据发现(CROWD-Discover)和表征学习(CROWD-Learn)任务。CROWD-Discover通过最大化子模态条件增益函数策略性挖掘未知实例；CROWD-Learn采用新颖组合目标，分离已知和未知表征同时保持已知类间的判别一致性。

### 主要发现

在OWOD基准评估中，CROWD在M-OWODB和S-OWODB上的已知类准确率分别提高了2.83%和2.05%，与最先进基线相比，未知召回率提高了约2.4倍。

### 结论

CROWD框架有效解决了开放世界目标检测中的关键挑战，显著提升了检测性能。

### 翻译

开放世界目标检测(OWOD)通过人工指导实现未知目标的持续发现和集成，从而丰富了传统目标检测器。然而，现有的OWOD方法经常遭受已知类与未知类之间的语义混淆，以及灾难性遗忘问题，导致未知召回率降低和已知类准确率下降。为克服这些挑战，我们提出了组合开放世界检测(CROWD)，这是一个统一框架，将未知目标发现和适应重新表述为一个交织的组合(基于集合)的数据发现(CROWD-Discover)和表征学习(CROWD-Learn)任务。CROWD-Discover通过最大化子模态条件增益(SCG)函数策略性地挖掘未知实例，选择与已知目标明显不同的代表性示例。随后，CROWD-Learn采用新颖的组合目标，同时分离已知和未知表征，同时保持已知类之间的判别一致性，从而减轻混淆和遗忘。在OWOD基准上的广泛评估表明，CROWD在M-OWODB和S-OWODB上的已知类准确率分别提高了2.83%和2.05%，并且与最先进的基线相比，未知召回率提高了近2.4倍。


### 论文摘要

Open-World Object Detection (OWOD) enriches traditional object detectors by enabling continual discovery and integration of unknown objects via human guidance. However, existing OWOD approaches frequently suffer from semantic confusion between known and unknown classes, alongside catastrophic forgetting, leading to diminished unknown recall and degraded known-class accuracy. To overcome these challenges, we propose Combinatorial Open-World Detection (CROWD), a unified framework reformulating unknown object discovery and adaptation as an interwoven combinatorial (set-based) data-discovery (CROWD-Discover) and representation learning (CROWD-Learn) task. CROWD-Discover strategically mines unknown instances by maximizing Submodular Conditional Gain (SCG) functions, selecting representative examples distinctly dissimilar from known objects. Subsequently, CROWD-Learn employs novel combinatorial objectives that jointly disentangle known and unknown representations while maintaining discriminative coherence among known classes, thus mitigating confusion and forgetting. Extensive evaluations on OWOD benchmarks illustrate that CROWD achieves improvements of 2.83% and 2.05% in known-class accuracy on M-OWODB and S-OWODB, respectively, and nearly 2.4x unknown recall compared to leading baselines.

---

## 26. Uncertainty-Aware Generative Oversampling Using an Entropy-Guided Conditional Variational Autoencoder

**论文链接:** [http://arxiv.org/abs/2509.25334v2](http://arxiv.org/abs/2509.25334v2)

**作者:** Amirhossein Zare, Amirhessam Zare, Parmida Sadat Pezeshki, Herlock, Rahimi, Ali Ebrahimi, Ignacio Vázquez-García, Leo Anthony Celi

**发布时间:** 2025-09-29

**备注:** 16 pages, 2 figures

### GPT解析

### 总结

本研究提出了一种名为LEO-CVAE的局部熵引导过采样方法，用于解决高维生物医学数据中的类别不平衡问题，通过整合局部不确定性信息提高生成样本质量，从而改善分类性能。

### 背景

类别不平衡是机器学习中的主要挑战，特别是在高维生物医学数据中，其非线性流形结构占主导地位。传统过采样方法如SMOTE依赖局部线性插值，产生不合理样本；而深度生成模型如CVAE虽能捕捉非线性分布，但未特别关注边界区域样本的重要性。

### 目的

开发一种能够显式地将局部不确定性纳入表示学习和数据生成的生成过采样框架，以改善不平衡数据的学习效果，特别是在复杂非线性结构的数据中。

### 方法

提出LEO-CVAE框架，通过计算样本邻域类分布的香农熵来量化不确定性，并利用两种机制：(1)局部熵加权损失(LEWL)强调在不确定区域的鲁棒学习；(2)熵引导的采样策略在类重叠区域集中生成样本。

### 主要发现

在ADNI和TCGA肺癌临床基因组数据集上应用LEO-CVAE，结果显示该方法一致提高了分类器性能，优于传统过采样和生成基线方法。

### 结论

在受复杂非线性结构（如组学数据）支配的领域中进行不平衡学习时，感知不确定性的生成过采样具有重要价值，LEO-CVAE通过整合局部不确定性信息有效提高了生成样本质量和分类性能。

### 翻译

类别不平衡仍然是机器学习中的一个主要挑战，特别是在高维生物医学数据中，非线性流形结构占主导地位。传统的过采样方法如SMOTE依赖于局部线性插值，常常产生不合理的合成样本。深度生成模型如条件变分自编码器(CVAE)能够更好地捕捉非线性分布，但标准变体对所有少数类样本同等对待，忽略了边界区域样本的重要性，如Borderline-SMOTE和ADASYN等启发式方法所强调的那样。我们提出了一个结合了CVAE的局部熵引导过采样方法(LEO-CVAE)，这是一个生成过采样框架，明确地将局部不确定性纳入表示学习和数据生成中。为了量化不确定性，我们计算样本邻域中类分布的香农熵：高熵表示更大的类重叠，作为不确定性的代理。LEO-CVAE通过两种机制利用这一信号：(i)一种局部熵加权损失(LEWL)，强调在不确定区域的鲁棒学习；(ii)一种熵引导的采样策略，在这些信息丰富、类重叠的区域集中生成。应用于临床基因组数据集(ADNI和TCGA肺癌)时，LEO-CVAE一致地提高了分类器性能，优于传统的过采样和生成基线。这些结果强调了在受复杂非线性结构（如组学数据）支配的领域中进行不平衡学习时，感知不确定性的生成过采样的价值。


### 论文摘要

Class imbalance remains a major challenge in machine learning, especially for high-dimensional biomedical data where nonlinear manifold structures dominate. Traditional oversampling methods such as SMOTE rely on local linear interpolation, often producing implausible synthetic samples. Deep generative models like Conditional Variational Autoencoders (CVAEs) better capture nonlinear distributions, but standard variants treat all minority samples equally, neglecting the importance of uncertain, boundary-region examples emphasized by heuristic methods like Borderline-SMOTE and ADASYN.   We propose Local Entropy-Guided Oversampling with a CVAE (LEO-CVAE), a generative oversampling framework that explicitly incorporates local uncertainty into both representation learning and data generation. To quantify uncertainty, we compute Shannon entropy over the class distribution in a sample's neighborhood: high entropy indicates greater class overlap, serving as a proxy for uncertainty. LEO-CVAE leverages this signal through two mechanisms: (i) a Local Entropy-Weighted Loss (LEWL) that emphasizes robust learning in uncertain regions, and (ii) an entropy-guided sampling strategy that concentrates generation in these informative, class-overlapping areas.   Applied to clinical genomics datasets (ADNI and TCGA lung cancer), LEO-CVAE consistently improves classifier performance, outperforming both traditional oversampling and generative baselines. These results highlight the value of uncertainty-aware generative oversampling for imbalanced learning in domains governed by complex nonlinear structures, such as omics data.

---

## 27. Diffusion^2: Turning 3D Environments into Radio Frequency Heatmaps

**论文链接:** [http://arxiv.org/abs/2510.02274v1](http://arxiv.org/abs/2510.02274v1)

**作者:** Kyoungjun Park, Yifan Yang, Changhan Ge, Lili Qiu, Shiqi Jiang

**发布时间:** 2025-10-02

### GPT解析

### 总结

Diffusion^2是一种基于扩散的方法，使用3D点云建模从Wi-Fi到毫米波的广泛频率范围内的RF信号传播，通过RF-3D Encoder和多尺度嵌入有效捕获RF相关特征，实现了高精度和高效的RF信号预测。

### 背景

RF信号传播建模对于理解环境至关重要，因为RF信号能提供RGB相机无法提供的宝贵见解，后者受限于可见光谱、镜头覆盖和遮挡。RF建模对于支持无线诊断、部署和优化也很有用，但在复杂环境中准确预测RF信号仍然是一个挑战，因为信号与障碍物存在相互作用，如吸收和反射。

### 目的

开发一种能够准确预测复杂环境中RF信号传播的方法，克服现有方法的局限性，提高预测精度和效率。

### 方法

提出Diffusion^2，一种基于扩散的方法，使用3D点云来建模RF信号传播。引入RF-3D Encoder来从3D数据中有效捕获RF相关特征，结合多尺度嵌入来模拟实际的RF信号传播过程。

### 主要发现

基于合成和真实世界测量的评估表明，Diffusion^2能够准确估计不同频段和环境条件下RF信号的行为，误差仅为1.9分贝，比现有方法快27倍。

### 结论

Diffusion^2代表了RF信号预测领域的重大进展，为无线诊断、部署和优化提供了更强大的工具。

### 翻译

射频信号传播建模对于理解环境至关重要，因为RF信号提供了超越RGB相机能力的宝贵见解，而RGB相机受限于可见光谱、镜头覆盖和遮挡。RF建模对于支持无线诊断、部署和优化也很有用。然而，由于与障碍物的相互作用（如吸收和反射），准确预测复杂环境中的RF信号仍然是一个挑战。我们引入了Diffusion^2，一种基于扩散的方法，使用3D点云来建模从Wi-Fi到毫米波的广泛频率范围内的RF信号传播。为了从3D数据中有效捕获RF相关特征，我们提出了RF-3D Encoder，它封装了3D几何的复杂性以及信号特定的细节。这些特征经过多尺度嵌入，以模拟实际的RF信号传播过程。基于合成和真实世界测量的评估表明，Diffusion^2能够准确估计不同频段和环境条件下RF信号的行为，误差仅为1.9分贝，比现有方法快27倍，标志着该领域的重大进展。更多信息请访问https://rfvision-project.github.io/。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决如何从3D环境模型生成准确的射频(RF)信号热力图问题。这个问题很重要，因为RF信号能提供超越RGB相机能力的环境洞察，可用于无线网络优化、设备部署、智能环境建设等领域，而现有方法要么计算成本高，要么需要大量预测量数据，难以实际应用。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者受生成式AI(如Sora)启发，思考能否让AI理解可见光谱外的RF信号。他们分析了现有射线追踪方法(计算复杂、需材料信息)和机器学习方法(需大量数据)的局限性，借鉴了扩散模型在图像生成中的成功应用，结合NeRF的场景表示思想，设计了条件引导的扩散框架，并创新性地提出了RF-3D编码器和RF-3D配对块来处理多模态信息。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是使用扩散模型将3D环境模型转换为RF信号热力图，通过条件引导确保生成结果符合物理规律。流程包括：1)用智能手机捕获3D环境模型和少量预测量数据；2)RF-3D编码器提取3D点云、2D图像和RF信号特征；3)扩散过程(前向加噪、反向去噪)由RF-3D特征引导；4)RF-3D配对块融合噪声预测与环境特征；5)训练时使用信号损失函数；6)推理时生成静态热力图或动态热力图视频。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点：1)首个基于扩散模型的RF热力图生成方法；2)RF-3D编码器提取多模态特征；3)RF-3D配对块实现跨模态融合；4)支持动态场景的视频扩散。相比不同：1)数据效率高(只需15个测量点vs.数千点)；2)计算速度快(27倍于AUTOMS)；3)支持多频率生成；4)环境变化时无需重新训练；5)同时支持静态和动态场景。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'Diffusion2创新性地将扩散模型与3D环境建模相结合，仅需少量预测量数据就能快速准确地生成多频率的RF信号热力图，显著提高了无线网络规划和优化的效率和实用性。'}


### 论文摘要

Modeling radio frequency (RF) signal propagation is essential for understanding the environment, as RF signals offer valuable insights beyond the capabilities of RGB cameras, which are limited by the visible-light spectrum, lens coverage, and occlusions. It is also useful for supporting wireless diagnosis, deployment, and optimization. However, accurately predicting RF signals in complex environments remains a challenge due to interactions with obstacles such as absorption and reflection. We introduce Diffusion^2, a diffusion-based approach that uses 3D point clouds to model the propagation of RF signals across a wide range of frequencies, from Wi-Fi to millimeter waves. To effectively capture RF-related features from 3D data, we present the RF-3D Encoder, which encapsulates the complexities of 3D geometry along with signal-specific details. These features undergo multi-scale embedding to simulate the actual RF signal dissemination process. Our evaluation, based on synthetic and real-world measurements, demonstrates that Diffusion^2 accurately estimates the behavior of RF signals in various frequency bands and environmental conditions, with an error margin of just 1.9 dB and 27x faster than existing methods, marking a significant advancement in the field. Refer to https://rfvision-project.github.io/ for more information.

---

## 28. GeoPurify: A Data-Efficient Geometric Distillation Framework for Open-Vocabulary 3D Segmentation

**论文链接:** [http://arxiv.org/abs/2510.02186v1](http://arxiv.org/abs/2510.02186v1)

**作者:** Weijia Dou, Xu Zhang, Yi Bin, Jian Liu, Bo Peng, Guoqing Wang, Yang Yang, Heng Tao Shen

**发布时间:** 2025-10-02

### GPT解析

### 总结

本研究提出了一种名为GeoPurify的新方法，用于解决将2D视觉语言模型特征转移到3D语义分割中的权衡问题，通过利用潜在的几何信息实现高效的数据利用和优越的性能。

### 背景

当前将2D视觉语言模型(VLMs)的特征转移到3D语义分割中存在一个持续的权衡：直接投影2D特征到3D会产生嘈杂和碎片化的预测，而强制几何一致性需要昂贵的训练流程和大规模的3D标注数据。

### 目的

开发一种方法，能够有效利用2D视觉语言模型的特征进行3D语义分割，同时避免传统方法中的权衡问题，实现更高的数据效率和性能。

### 方法

提出GeoPurify方法，应用一个小型学生亲和网络来净化2D VLM生成的3D点特征，使用从3D自监督教师模型中提炼的几何先验；在推理阶段，设计了几何引导池化模块来进一步去噪点云并确保语义和结构一致性。

### 主要发现

在主要3D基准上的广泛实验表明，GeoPurify在使用仅约1.5%训练数据的情况下达到或超过了最先进的性能，有效缓解了传统方法中的权衡问题。

### 结论

GeoPurify通过利用潜在的几何信息和学习的亲和网络，成功缓解了2D到3D特征转移中的权衡问题，实现了优越的数据效率和性能，为3D语义分割提供了一种新的有效方法。

### 翻译

最近尝试将2D视觉语言模型(VLMs)的特征转移到3D语义分割中暴露了一个持续的权衡。直接将2D特征投影到3D会产生嘈杂和碎片化的预测，而强制几何一致性需要昂贵的训练流程和大规模的3D标注数据。我们认为这种限制源于主导的分割和匹配范式，该范式无法调和2D语义与3D几何结构。几何线索在2D到3D转移过程中并未被消除，而是保留在嘈杂和视图聚合的特征中。为了利用这一特性，我们提出了GeoPurify，它应用一个小型学生亲和网络，使用从3D自监督教师模型中提炼的几何先验来净化2D VLM生成的3D点特征。在推理过程中，我们设计了一个几何引导池化模块来进一步去噪点云并确保语义和结构一致性。受益于潜在的几何信息和学习的亲和网络，GeoPurify有效缓解了权衡问题并实现了优越的数据效率。在主要3D基准上的广泛实验表明，GeoPurify在使用仅约1.5%训练数据的情况下达到或超过了最先进的性能。我们的代码和检查点可在https://github.com/tj12323/GeoPurify获取。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决开放词汇3D语义分割中的基本权衡问题：直接将2D视觉-语言模型(VLM)的特征投影到3D会产生噪声和碎片化的预测，而强制几何一致性则需要昂贵的训练流程和大规模的3D标注数据。这个问题很重要，因为传统3D场景理解受限于封闭世界假设，无法处理真实世界中多样复杂的物体；同时手动3D标注成本极高，劳动强度大；开放词汇3D理解对自动驾驶、机器人和增强现实等应用至关重要，但当前方法要么产生噪声输出，要么需要大量计算资源。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': "作者观察到当前'分割和匹配'范式的局限性，它将几何和语义视为分离的问题。作者假设2D到3D的转换不会破坏几何信息，而是使其变得潜在(latent)，因此可以更有效地恢复潜在结构，而不是从头学习3D几何。作者借鉴了生物感知中的'分割即理解'范式。方法借鉴了现有工作：使用2D视觉-语言模型(X-Decoder)获取语义特征；利用3D自监督教师模型(如Sonata)提供几何先验；应用知识蒸馏技术让学生网络从教师模型学习几何关系；采用对比学习来学习点之间的几何亲和性。", '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是通过几何对比蒸馏，利用小型学生亲和网络来净化2D VLM生成的3D点特征，使用从3D自监督教师模型中提炼的几何先验。在推理时，使用几何引导池化模块去噪点云并确保语义和结构一致性。整体流程：1)语义初始化：用冻结的2D VLM从多视图RGB图像生成初始3D特征；2)几何对比蒸馏：训练学生网络学习几何亲和性，使用教师模型作为指导，通过对比目标优化；3)几何引导池化：在推理时，用训练好的学生网络构建亲和矩阵，通过迭代池化传播语义信息，产生净化的特征集。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': "关键创新点：1)范式转变：从'分割和匹配'转向'分割即理解'；2)数据效率框架：仅用1.5%训练数据达到或超过最先进性能；3)几何对比蒸馏：让学生从未标记3D扫描中学习几何亲和性；4)几何引导池化：在推理时确保结构一致性。不同之处：不同于训练自由方法导致的噪声问题；不同于需要密集标注的高成本方法；不依赖大规模数据集隐式学习几何先验；采用解耦训练过程学习与语义无关的几何先验，提高跨域泛化能力。", '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'GeoPurify通过几何对比蒸馏和几何引导池化，解决了开放词汇3D语义分割中语义丰富性与几何一致性的基本权衡，实现了在极少量数据上达到或超过最先进性能的数据高效3D场景理解框架。'}


### 论文摘要

Recent attempts to transfer features from 2D Vision-Language Models (VLMs) to 3D semantic segmentation expose a persistent trade-off. Directly projecting 2D features into 3D yields noisy and fragmented predictions, whereas enforcing geometric coherence necessitates costly training pipelines and large-scale annotated 3D data. We argue that this limitation stems from the dominant segmentation-and-matching paradigm, which fails to reconcile 2D semantics with 3D geometric structure. The geometric cues are not eliminated during the 2D-to-3D transfer but remain latent within the noisy and view-aggregated features. To exploit this property, we propose GeoPurify that applies a small Student Affinity Network to purify 2D VLM-generated 3D point features using geometric priors distilled from a 3D self-supervised teacher model. During inference, we devise a Geometry-Guided Pooling module to further denoise the point cloud and ensure the semantic and structural consistency. Benefiting from latent geometric information and the learned affinity network, GeoPurify effectively mitigates the trade-off and achieves superior data efficiency. Extensive experiments on major 3D benchmarks demonstrate that GeoPurify achieves or surpasses state-of-the-art performance while utilizing only about 1.5% of the training data. Our codes and checkpoints are available at [https://github.com/tj12323/GeoPurify](https://github.com/tj12323/GeoPurify).

---

## 29. LangGrasp: Leveraging Fine-Tuned LLMs for Language Interactive Robot Grasping with Ambiguous Instructions

**论文链接:** [http://arxiv.org/abs/2510.02104v1](http://arxiv.org/abs/2510.02104v1)

**作者:** Yunhan Lin, Wenqi Wu, Zhijie Zhang, Huasong Min

**发布时间:** 2025-10-02

**备注:** 8 pages, 6 figures

### GPT解析

### 总结

本文提出LangGrasp框架，通过整合微调的大语言模型和点云定位模块，解决语言驱动抓取中处理模糊指令隐含意图的问题，实现从对象级到部件级的高精度抓取。

### 背景

现有的语言驱动抓取方法难以处理包含隐含意图的模糊指令。

### 目的

提出LangGrasp框架，解决模糊指令中隐含意图的处理问题。

### 方法

提出LangGrasp框架，整合微调的大语言模型(LLMs)利用其常识理解和环境感知能力，从语言指令中推断隐含意图；设计了由2D部件分割引导的点云定位模块，实现场景中部分点云定位。

### 主要发现

实验结果表明，LangGrasp框架能够准确解决模糊指令中的隐含意图，识别出任务完成所需的关键操作和目标信息；通过整合环境信息动态选择最优抓取姿态。

### 结论

LangGrasp实现了从对象级到部件级的高精度抓取，显著提高了机器人在非结构化环境中的适应性和任务执行效率。

### 翻译

现有的语言驱动抓取方法难以处理包含隐含意图的模糊指令。为了应对这一挑战，我们提出了LangGrasp，一种新颖的语言交互式机器人抓取框架。该框架整合了微调的大语言模型(LLMs)，利用其强大的常识理解和环境感知能力，从而从语言指令中推断隐含意图，并阐明任务要求以及目标操作对象。此外，我们设计的由2D部件分割引导的点云定位模块，能够在场景中进行部分点云定位，从而将抓取操作从粗粒度的对象级扩展到细粒度的部件级操作。实验结果表明，LangGrasp框架能够准确解决模糊指令中的隐含意图，识别出任务完成所需的关键操作和目标信息，而这些信息在指令中未明确说明但对任务完成至关重要。此外，它通过整合环境信息动态选择最优抓取姿态，实现了从对象级到部件级操作的高精度抓取，显著提高了机器人在非结构化环境中的适应性和任务执行效率。更多信息 and code are available here: https://github.com/wu467/LangGrasp.

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决现有语言驱动抓取方法无法处理包含隐含意图的模糊指令，以及现有框架仅限于对象级别抓取而忽略部件功能区别的问题。这个问题很重要，因为随着机器人在日常环境中的部署增加，基于自然语言的人机交互能显著提高效率，但在动态非结构化环境中准确解释模糊指令，特别是包含隐含意图的指令，仍是关键挑战，机器人需要具备高效的语言理解和视觉感知能力来实现常识驱动的意图消除歧义。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先分析了现有方法在处理模糊指令方面的局限性，注意到大型语言模型具有强大的常识理解和环境感知能力，可以利用这些能力推断隐含意图。他们设计了三个主要模块：感知与推理、点云定位和抓取姿态检测。通过微调LLMs来保留其广泛知识同时增强推理能力。作者借鉴了LLMs作为机器人任务规划器的现有工作，语言驱动机器人抓取的研究，以及任务导向抓取(TOG)的相关方法，但针对抓取任务进行了专门优化和创新。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是利用微调的大型语言模型处理模糊复杂指令，推断隐含意图，并将抓取从对象级别扩展到部件级别，通过结构化输出直接提取任务信息。整体流程分为三阶段：1)感知与推理：使用微调LLM基于场景和多轮对话生成JSON格式动作序列；2)点云定位：用预训练2D部件分割模型定位目标，应用扩展策略确保覆盖边缘和背景信息，将深度图转换为点云并聚焦目标区域；3)抓取姿态检测：对定位点云预测6-DoF抓取姿态，基于置信度选择最优姿态，生成执行轨迹。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)提出LangGrasp框架，利用微调LLMs处理模糊指令并实现部件级别抓取；2)构建专门用于机器人抓取任务的LLMs微调数据集；3)设计点云定位模块支持精细抓取。相比之前工作，LangGrasp能处理含隐含意图的模糊指令，实现从对象到部件的细粒度抓取，通过微调生成结构化输出无需额外解析，并使用更简洁方法实现任务导向抓取，避免了大规模数据集训练的需求。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'LangGrasp通过微调大型语言模型并设计创新的点云定位策略，使机器人能够准确理解模糊语言指令并实现从对象级别到部件级别的精细抓取操作，显著提高了机器人在非结构化环境中的适应性和任务执行效率。'}


### 论文摘要

The existing language-driven grasping methods struggle to fully handle ambiguous instructions containing implicit intents. To tackle this challenge, we propose LangGrasp, a novel language-interactive robotic grasping framework. The framework integrates fine-tuned large language models (LLMs) to leverage their robust commonsense understanding and environmental perception capabilities, thereby deducing implicit intents from linguistic instructions and clarifying task requirements along with target manipulation objects. Furthermore, our designed point cloud localization module, guided by 2D part segmentation, enables partial point cloud localization in scenes, thereby extending grasping operations from coarse-grained object-level to fine-grained part-level manipulation. Experimental results show that the LangGrasp framework accurately resolves implicit intents in ambiguous instructions, identifying critical operations and target information that are unstated yet essential for task completion. Additionally, it dynamically selects optimal grasping poses by integrating environmental information. This enables high-precision grasping from object-level to part-level manipulation, significantly enhancing the adaptability and task execution efficiency of robots in unstructured environments. More information and code are available here: https://github.com/wu467/LangGrasp.

---

## 30. GaussianMorphing: Mesh-Guided 3D Gaussians for Semantic-Aware Object Morphing

**论文链接:** [http://arxiv.org/abs/2510.02034v1](http://arxiv.org/abs/2510.02034v1)

**作者:** Mengtian Li, Yunshu Bai, Yimin Chu, Yijun Shen, Zhongmei Li, Weifeng Ge, Zhifeng Xie, Chaofeng Chen

**发布时间:** 2025-10-02

**备注:** Project page: https://baiyunshu.github.io/GAUSSIANMORPHING.github.io/

### GPT解析

### 总结

本文介绍了一种名为GaussianMorphing的新框架，用于从多视角图像进行语义感知的3D形状和纹理变形。

### 背景

以往的方法通常依赖于点云或需要为未纹理数据预定义同胚映射，存在一定的局限性。

### 目的

克服现有方法的限制，实现高质量的3D形状和纹理变形，同时保持几何一致性和纹理保真度。

### 方法

利用网格引导的3D高斯溅射技术进行高保真几何和外观建模，通过统一的变形策略将3D高斯锚定到重建的网格补片上，并利用网格拓扑作为几何先验建立无监督语义对应。

### 主要发现

在提出的TexMorph基准测试上，GaussianMorphing显著优于之前的2D/3D方法，将颜色一致性误差降低了22.2%，将EI指标降低了26.2%。

### 结论

这种集成方法在不需要标记数据的情况下，在整个变形过程中保持了局部细节和全局语义一致性。

### 翻译

我们引入GaussianMorphing，一种用于从多视角图像进行语义感知的3D形状和纹理变形的新框架。以往的方法通常依赖于点云或需要为未纹理数据预定义同胚映射。我们的方法通过利用网格引导的3D高斯溅射技术进行高保真几何和外观建模，克服了这些限制。我们框架的核心是一个统一的变形策略，将3D高斯锚定到重建的网格补片上，确保几何一致的变换，并通过拓扑感知约束保持纹理保真度。同时，我们的框架利用网格拓扑作为几何先验建立无监督语义对应，并通过物理合理的点轨迹保持结构完整性。这种集成方法在整个变形过程中保持了局部细节和全局语义一致性，而无需标记数据。在我们提出的TexMorph基准测试上，GaussianMorphing显著优于之前的2D/3D方法，将颜色一致性误差降低了22.2%，将EI降低了26.2%。项目页面：https://baiyunshu.github.io/GAUSSIANMORPHING.github.io/

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决语义感知的3D形状和纹理变形问题，即如何从多视角图像生成高质量、结构一致且纹理连贯的3D物体变形序列。这个问题在计算机动画、几何建模、形状分析和电影视觉特效等领域非常重要，因为变形技术是连接计算机视觉与计算机图形学的桥梁，能够实现物体之间的平滑过渡，为创意内容制作提供关键技术支持。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者通过分析现有方法的局限性进行思考：图像导向方法缺乏3D几何推理，3D方法需要高质量网格输入且难以处理复杂纹理。他们借鉴了3D高斯泼溅(3DGS)的高效渲染能力和网格的结构优势，结合图卷积网络学习语义特征，并使用测地距离计算几何关系。通过将离散的高斯点与结构化网格桥接，作者设计出一种混合表示方法，既保留了高斯表示的渲染优势，又利用网格提供了必要的结构约束。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是使用网格作为拓扑支架来引导无结构高斯点的变形，确保几何一致性的同时保持纹理保真度。整体流程包括：1)从多视角图像生成混合网格-高斯表示，将高斯锚定到网格面上；2)通过图卷积网络学习网格顶点间的语义对应关系；3)使用神经网络预测连续变形场；4)随着网格变形更新绑定高斯的位置；5)通过多目标优化平衡几何结构、外观一致性和语义对齐，最终生成高质量的3D变形序列。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)首个使用3D高斯进行联合3D几何和纹理变形的框架；2)网格引导的变形机制，同时感知拓扑和语义；3)双域优化策略结合几何约束和纹理插值。相比之前的工作，本方法不需要预定义的3D数据或同胚映射，直接从图像生成完全纹理化的3D输出；解决了现有方法在几何鲁棒性、纹理一致性和输入可访问性间的权衡问题；在复杂拓扑和纹理丰富场景中表现出色，同时减少了高质量3D数据的依赖。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'GaussianMorphing通过结合网格引导的3D高斯泼溅与语义感知变形，实现了从多视角图像生成高质量纹理化3D变形的创新框架，显著优于现有方法在几何准确性、纹理一致性和结构完整性方面的表现。'}


### 论文摘要

We introduce GaussianMorphing, a novel framework for semantic-aware 3D shape and texture morphing from multi-view images. Previous approaches usually rely on point clouds or require pre-defined homeomorphic mappings for untextured data. Our method overcomes these limitations by leveraging mesh-guided 3D Gaussian Splatting (3DGS) for high-fidelity geometry and appearance modeling. The core of our framework is a unified deformation strategy that anchors 3DGaussians to reconstructed mesh patches, ensuring geometrically consistent transformations while preserving texture fidelity through topology-aware constraints. In parallel, our framework establishes unsupervised semantic correspondence by using the mesh topology as a geometric prior and maintains structural integrity via physically plausible point trajectories. This integrated approach preserves both local detail and global semantic coherence throughout the morphing process with out requiring labeled data. On our proposed TexMorph benchmark, GaussianMorphing substantially outperforms prior 2D/3D methods, reducing color consistency error ($\Delta E$) by 22.2% and EI by 26.2%. Project page: https://baiyunshu.github.io/GAUSSIANMORPHING.github.io/

---

## 31. LiLa-Net: Lightweight Latent LiDAR Autoencoder for 3D Point Cloud Reconstruction

**论文链接:** [http://arxiv.org/abs/2510.02028v1](http://arxiv.org/abs/2510.02028v1)

**作者:** Mario Resino, Borja Pérez, Jaime Godoy, Abdulla Al-Kaff, Fernando García

**发布时间:** 2025-10-02

**备注:** 7 pages, 3 figures, 7 tables, Submitted to ICRA

### GPT解析

### 总结

本研究提出了一种名为LiLa-Net的3D自编码器架构，仅使用LiDAR点云从真实交通环境中编码高效特征，并成功实现了原始点云的准确重建。

### 背景

研究使用配备Velodyne LiDAR的真实半自动驾驶车辆作为数据来源，针对交通环境中的点云数据处理需求。

### 目的

开发一种资源高效的3D自编码器架构，能够从交通环境中提取有效特征并准确重建原始点云。

### 方法

提出LiLa-Net架构，利用跳跃连接概念提高性能，同时减少编码器层数和简化跳跃连接结构，以在保持高效性的同时产生具有代表性的潜在空间。

### 主要发现

在跳跃连接信息和潜在编码之间实现了有效平衡，提高了重建质量而不影响性能；模型展示了强大的泛化能力，能够重建与原始交通环境无关的物体。

### 结论

LiLa-Net是一种资源高效的3D自编码器架构，能够在不使用大量资源的情况下，从交通环境中提取有效特征并准确重建点云，同时具备良好的泛化能力。

### 翻译

这项工作提出了一种名为LiLa-Net的3D自编码器架构，它仅使用LiDAR的点云从真实交通环境中编码高效特征。为此，我们使用了一辆配备有Velodyne LiDAR的真实半自动驾驶车辆。该系统利用跳跃连接概念来提高性能，而不像最先进的架构那样使用大量资源。关键变化包括减少编码器层数和简化跳跃连接，同时仍然产生高效的潜在空间，允许准确重建原始点云。此外，在跳跃连接和潜在编码携带的信息之间实现了有效平衡，提高了重建质量而不影响性能。最后，该模型展示了强大的泛化能力，成功重建了与原始交通环境无关的物体。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决如何高效处理自动驾驶车辆中LiDAR传感器生成的大量3D点云数据问题。这个问题很重要，因为自动驾驶需要准确理解周围环境，而点云数据量大，需要能高效提取特征同时减少计算和内存需求的模型，现有的基于Transformer的架构虽然有效但计算成本高，限制了实时系统中的部署。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者意识到现有方法计算资源需求高，因此设计了一种轻量级框架。他们借鉴了自监督学习和掩码建模策略，但简化了架构；采用了编码器-解码器结构类似传统自编码器；利用跳跃连接概念但简化了实现；使用Chamfer Distance作为重建目标。这些借鉴帮助他们创建了一个更高效、更轻量的点云处理模型。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是设计轻量级自编码器架构，直接在稀疏3D点上操作，通过减少编码器层数和简化跳跃连接提高性能，在跳跃连接信息和潜在编码间实现平衡，学习紧凑且具有表达能力的潜在表示。流程包括：1)数据预处理(移除地面点、范围过滤、下采样)；2)编码器(1D卷积层处理点云，生成特征)；3)潜在空间(生成1024维潜在表示)；4)跳跃连接(仅保留最后编码层连接)；5)解码器(将特征转换回3D坐标重建点云)。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点：1)提出LiLa-Net直接在稀疏3D点上操作；2)能高效处理大型密集点云；3)成功重建复杂交通环境；4)证明潜在表示强大泛化能力；5)消除预训练或掩码策略。不同之处：简化了跳跃连接结构；减少编码器层数使模型更轻量；实现跳跃连接和潜在编码的平衡；不需要预训练训练流程更直接；保持高质量重建同时减少计算和内存需求。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'LiLa-Net是一种轻量级自编码器，能够高效压缩和重建3D点云数据，同时学习具有强大泛化能力的紧凑潜在表示，解决了自动驾驶环境中处理大规模点云数据的计算和内存挑战。'}


### 论文摘要

This work proposed a 3D autoencoder architecture, named LiLa-Net, which encodes efficient features from real traffic environments, employing only the LiDAR's point clouds. For this purpose, we have real semi-autonomous vehicle, equipped with Velodyne LiDAR. The system leverage skip connections concept to improve the performance without using extensive resources as the state-of-the-art architectures. Key changes include reducing the number of encoder layers and simplifying the skip connections, while still producing an efficient and representative latent space which allows to accurately reconstruct the original point cloud. Furthermore, an effective balance has been achieved between the information carried by the skip connections and the latent encoding, leading to improved reconstruction quality without compromising performance. Finally, the model demonstrates strong generalization capabilities, successfully reconstructing objects unrelated to the original traffic environment.

---

## 32. Efficient manifold evolution algorithm using adaptive B-Spline interpolation

**论文链接:** [http://arxiv.org/abs/2510.01790v1](http://arxiv.org/abs/2510.01790v1)

**作者:** Muhammad Ammad, Leevan Ling

**发布时间:** 2025-10-02

**DOI:** 10.1016/j.enganabound.2025.106488

### GPT解析

### 总结

该论文提出了一种在光滑流形上演化点云数据的高效拉格朗日方法，使用B样条作为基础函数，替代传统径向基函数方法，实现点云数据的灵活操作和高效更新。

### 背景

传统径向基函数(RBF)方法在高维流形上处理点云数据存在局限性，需要频繁重新插值，特别是在点密度波动较大的区域。

### 目的

开发一种替代传统RBF方法的高效拉格朗日方法，用于在光滑流形上演化点云数据，实现点云数据的无缝添加和删除，特别是在点密度变化大的区域。

### 方法

使用B样条作为所有局部插值的基础函数，利用其系数具有几何意义的特性，使系数可以像点一样被操作，实现快速更新插值函数，避免频繁重新插值。

### 主要发现

数值结果证明了所提方法在几何量收敛方面的有效性，以及点云数据添加和删除操作的无缝性，特别是在点密度波动较大的区域表现尤为突出。

### 结论

B样条作为拉格朗日方法的基础函数能够有效替代传统RBF方法，提供更高效、更灵活的点云数据处理方式，特别适合处理点密度变化大的场景。

### 翻译

本文探讨了一种在光滑流形上演化点云数据的高效拉格朗日方法。在这项初步研究中，我们专注于分析平面曲线，最终目标是为高维流形提供一种替代传统径向基函数(RBF)的方法。特别是，我们使用B样条作为所有局部插值的基础函数。与RBF和其他光滑基函数一样，B样条能够近似法向量和曲率等几何特征。一旦正确设置，使用B样条的优势在于其系数具有几何意义，这使得系数可以像点一样被操作，便于快速更新插值函数，并消除了频繁重新插值的需要。因此，点云数据的添加和删除变得无缝进行，特别是在点密度波动较大的区域特别有利。数值结果证明了几何量的收敛性和该方法的有效性。最后，我们展示了曲率流的模拟，其速度取决于耦合反应-扩散系统模式形成的解。


### 论文摘要

This paper explores an efficient Lagrangian approach for evolving point cloud data on smooth manifolds. In this preliminary study, we focus on analyzing plane curves, and our ultimate goal is to provide an alternative to the conventional radial basis function (RBF) approach for manifolds in higher dimensions. In particular, we use the B-Spline as the basis function for all local interpolations. Just like RBF and other smooth basis functions, B-Splines enable the approximation of geometric features such as normal vectors and curvature. Once properly set up, the advantage of using B-Splines is that their coefficients carry geometric meanings. This allows the coefficients to be manipulated like points, facilitates rapid updates of the interpolant, and eliminates the need for frequent re-interpolation. Consequently, the removal and insertion of point cloud data become seamless processes, particularly advantageous in regions experiencing significant fluctuations in point density. The numerical results demonstrate the convergence of geometric quantities and the effectiveness of our approach. Finally, we show simulations of curvature flows whose speeds depend on the solutions of coupled reaction--diffusion systems for pattern formation.

---

## 33. Reducing Simulation Dependence in Neutrino Telescopes with Masked Point Transformers

**论文链接:** [http://arxiv.org/abs/2510.01733v1](http://arxiv.org/abs/2510.01733v1)

**作者:** Felix J. Yu, Nicholas Kamp, Carlos A. Argüelles

**发布时间:** 2025-10-02

**备注:** 8 pages, 3 figures, presented at the 39th International Cosmic Ray  Conference (ICRC2025)

### GPT解析

### 总结

本文提出了中微子望远镜的首个自监督学习训练流程，利用点云变换器和掩码自编码器，减少对模拟数据的依赖，从而提高事件重建和分类的准确性。

### 背景

传统中微子物理中的机器学习技术依赖模拟数据获取真实标签，但模拟数据的准确性以及模拟与真实数据间的差异仍是重大问题，特别是在复杂自然介质中运行的大型中微子望远镜。

### 目的

减少对标记数据集的依赖，降低模拟数据带来的系统不确定性。

### 方法

开发首个中微子望远镜的自监督训练流程，利用点云变换器和掩码自编码器，将大部分训练转移到真实数据上。

### 主要发现

通过将训练重点转向真实数据，显著减少了对模拟数据的依赖，从而减轻了相关的系统不确定性问题。

### 结论

这代表了中微子望远镜中机器学习应用的范式转变，为事件重建和分类技术的重大改进开辟了新途径。

### 翻译

中微子物理中的机器学习技术传统上依赖于模拟数据，这可以提供真实标签的访问。然而，这些模拟的准确性以及模拟数据和真实数据之间的差异仍然是一个重大问题，特别是在复杂自然介质中运行的大型中微子望远镜。近年来，自监督学习已成为减少对标记数据集依赖的强大范式。在这里，我们提出了中微子望远镜的第一个自监督训练流程，利用点云变换器和掩码自编码器。通过将大部分训练转移到真实数据，这种方法最小化了对模拟的依赖，从而减轻了相关的系统不确定性。这代表了中微子望远镜中机器学习应用的根本性转变，为事件重建和分类的重大改进铺平了道路。


### 论文摘要

Machine learning techniques in neutrino physics have traditionally relied on simulated data, which provides access to ground-truth labels. However, the accuracy of these simulations and the discrepancies between simulated and real data remain significant concerns, particularly for large-scale neutrino telescopes that operate in complex natural media. In recent years, self-supervised learning has emerged as a powerful paradigm for reducing dependence on labeled datasets. Here, we present the first self-supervised training pipeline for neutrino telescopes, leveraging point cloud transformers and masked autoencoders. By shifting the majority of training to real data, this approach minimizes reliance on simulations, thereby mitigating associated systematic uncertainties. This represents a fundamental departure from previous machine learning applications in neutrino telescopes, paving the way for substantial improvements in event reconstruction and classification.

---

## 34. Non-Rigid Structure-from-Motion via Differential Geometry with Recoverable Conformal Scale

**论文链接:** [http://arxiv.org/abs/2510.01665v1](http://arxiv.org/abs/2510.01665v1)

**作者:** Yongbo Chen, Yanhao Zhang, Shaifali Parashar, Liang Zhao, Shoudong Huang

**发布时间:** 2025-10-02

### GPT解析

### 总结

该论文介绍了一种名为Con-NRSfM的新方法，用于解决单目视觉可变形SLAM中的非刚性结构从运动问题。该方法通过基于图框架优化的2D选择图像变形进行逐点重建，消除了现有方法的严格假设，能够准确计算局部共形尺度并解耦深度和共形尺度的约束，实现了更精确的深度估计。实验证明该方法在重建精度和鲁棒性上优于现有方法。

### 背景

非刚性结构从运动技术是解决单目视觉可变形同时定位与地图构建中映射挑战的一种有前景的方法，近年来引起了越来越多的关注。

### 目的

引入一种名为Con-NRSfM的新方法，用于处理共形变形下的非刚性结构从运动问题，消除现有方法的严格假设限制，准确计算局部共形尺度，解耦深度和共形尺度的约束，实现更精确的深度估计。

### 方法

使用基于图框架优化的2D选择图像变形进行逐点重建，采用并行可分离迭代优化策略解决公式化问题的敏感性，集成自监督学习框架，使用编码器-解码器网络生成带纹理的密集3D点云。

### 主要发现

该方法消除了对局部平面表面或局部线性变形的严格假设，能够恢复共形尺度，解耦了深度和共形尺度的约束使深度估计更精确，在合成和真实数据集上的实验结果表明，该方法在重建精度和鲁棒性方面超越了现有方法。

### 结论

Con-NRSfM方法在非刚性结构从运动领域表现出色，代码将在项目网站上公开：https://sites.google.com/view/con-nrsfm。

### 翻译

Non-rigid structure-from-motion (NRSfM): 非刚性结构从运动；monocular visual deformable simultaneous localization and mapping (SLAM): 单目视觉可变形同时定位与地图构建；conformal deformations: 共形变形；isometric deformations: 等距变形；point-wise reconstruction: 逐点重建；image warps: 图像变形；graph-based framework: 基于图框架；locally planar surfaces: 局部平面表面；locally linear deformations: 局部线性变形；conformal scale: 共形尺度；depth estimation: 深度估计；parallel separable iterative optimization strategy: 并行可分离迭代优化策略；self-supervised learning framework: 自监督学习框架；encoder-decoder network: 编码器-解码器网络；dense 3D point clouds: 密集3D点云；reconstruction accuracy: 重建精度；robustness: 鲁棒性。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决非刚性结构运动恢复（NRSfM）中的挑战，特别是在共形变形条件下的3D形状重建问题。这个问题在现实中非常重要，因为它对于单目视觉SLAM在变形环境中的应用至关重要，能够帮助机器人在动态变化的环境中（如医疗手术、软体物体交互等）进行精确导航和地图构建，解决了传统SLAM方法无法处理的非刚性场景问题。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先分析了现有NRSfM方法的局限性，包括统计约束方法对复杂变形效果不佳、全局物理约束方法计算复杂度高、以及局部物理约束方法依赖不切实际的假设（如局部平面表面和局部线性变形）。基于微分几何理论，作者发现了共形变形下连接（connections）和活动标架（moving frames）的旋转不变性性质，证明了共形尺度和深度估计可以解耦。他们借鉴了现有局部物理约束方法（如[11]、[12]、[13]和[37]），但通过理论创新克服了这些方法的局限性，设计了并行可分离的迭代优化算法和自监督学习框架。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是利用微分几何理论中的旋转不变性性质，放弃现有方法中的局部平面表面和局部线性变形假设，将共形尺度和深度估计解耦，使用并行可分离的迭代优化策略提高鲁棒性。整体流程包括：1)构建完整加权图并选择连接良好的子图；2)基于选定的图像变换进行点级重建；3)通过并行可分离迭代优化（包括预步骤和四个主要步骤）优化深度、法线和共形尺度；4)可选地使用自监督编码器-解码器网络生成带纹理的密集3D点云；5)输出稀疏点云或密集带纹理的表面。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)理论创新，证明了共形变形下连接的旋转不变性，实现了共形尺度和深度估计的解耦；2)方法创新，放弃了局部平面表面和局部线性变形假设，设计了并行可分离的迭代优化算法，提出了基于共形尺度的物理约束；3)结合自监督学习框架生成密集点云。相比之前的工作，不同之处在于：不需要依赖不切实际的假设，能恢复共形尺度，将深度和共形尺度的约束解耦，使用更鲁棒的优化策略，并能生成更全面的密集点云表示。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '这篇论文通过微分几何理论和并行可分离优化框架，提出了一种能够在共形变形条件下准确恢复3D形状并估计共形尺度的非刚性结构运动恢复方法，显著提高了变形物体重建的精度和鲁棒性。'}


### 论文摘要

Non-rigid structure-from-motion (NRSfM), a promising technique for addressing the mapping challenges in monocular visual deformable simultaneous localization and mapping (SLAM), has attracted growing attention. We introduce a novel method, called Con-NRSfM, for NRSfM under conformal deformations, encompassing isometric deformations as a subset. Our approach performs point-wise reconstruction using 2D selected image warps optimized through a graph-based framework. Unlike existing methods that rely on strict assumptions, such as locally planar surfaces or locally linear deformations, and fail to recover the conformal scale, our method eliminates these constraints and accurately computes the local conformal scale. Additionally, our framework decouples constraints on depth and conformal scale, which are inseparable in other approaches, enabling more precise depth estimation. To address the sensitivity of the formulated problem, we employ a parallel separable iterative optimization strategy. Furthermore, a self-supervised learning framework, utilizing an encoder-decoder network, is incorporated to generate dense 3D point clouds with texture. Simulation and experimental results using both synthetic and real datasets demonstrate that our method surpasses existing approaches in terms of reconstruction accuracy and robustness. The code for the proposed method will be made publicly available on the project website: https://sites.google.com/view/con-nrsfm.

---

## 35. Real-time Multi-Plane Segmentation Based on GPU Accelerated High-Resolution 3D Voxel Mapping for Legged Robot Locomotion

**论文链接:** [http://arxiv.org/abs/2510.01592v1](http://arxiv.org/abs/2510.01592v1)

**作者:** Shun Niijima, Ryoichi Tsuzaki, Noriaki Takasugi, Masaya Kinoshita

**发布时间:** 2025-10-02

**备注:** 8 pages, 12 figures, This work has been submitted to the IEEE for  possible publication. Copyright may be transfered without notice, after which  this version may no longer be accessible

### GPT解析

### 总结

本文提出了一种基于GPU加速的高分辨率3D体素映射的实时多平面分割方法，用于足式机器人运动控制。

### 背景

现有的在线平面映射方法难以平衡准确性和计算效率：直接深度图像分割存在时间积分差问题；基于高度图的方法无法表示悬挑等复杂3D结构；基于体素的平面分割在实时应用中尚未被探索。

### 目的

解决现有平面映射方法的局限性，开发一种能够快速准确提取3D平面区域的新框架。

### 方法

结合基于顶点的连通分量标记与基于随机样本一致性的平面检测和凸包，利用GPU并行计算从高分辨率3D体素图中积累的点云中快速提取平面区域。

### 主要发现

提出的方法能够在0.01米的分辨率下实现超过30Hz更新率的快速准确的3D多平面分割，使检测到的平面能够实时用于运动控制任务。

### 结论

通过在模拟环境和物理足式机器人平台上的实验验证了该方法的有效性，考虑3D平面结构时能够实现稳健的运动性能。

### 翻译

本文提出了一种基于GPU加速的高分辨率3D体素映射的实时多平面分割方法，用于足式机器人运动。现有的在线平面映射方法难以平衡准确性和计算效率：直接从特定传感器进行深度图像分割存在时间积分差的问题，基于高度图的方法无法表示悬挑等复杂3D结构，而基于体素的平面分割在实时应用中尚未被探索。为解决这些限制，我们开发了一种新的框架，结合基于顶点的连通分量标记与基于随机样本一致性的平面检测和凸包，利用GPU并行计算从高分辨率3D体素图中积累的点云中快速提取平面区域。实验结果表明，提出的方法能够在0.01米的分辨率下实现超过30Hz更新率的快速准确的3D多平面分割，使检测到的平面能够实时用于运动控制任务。此外，我们通过在模拟环境和物理足式机器人平台上的实验验证了该方法的有效性，考虑3D平面结构时能够实现稳健的运动性能。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决足式机器人在复杂3D环境中运动时，如何实现实时、高精度的多平面分割问题。这个问题很重要，因为足式机器人需要准确识别可站立区域和稳定平面表面才能安全运动，而现有方法要么无法表示悬挑等复杂结构（如高度图方法），要么计算效率太低无法实时处理（如传统体素方法），导致机器人在开放式楼梯、桌下等环境中容易发生碰撞和运动失败。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者通过分析现有方法的局限性来设计新方法：直接传感器分割方法时间积分差，高度图方法无法表示复杂3D结构，传统体素分割无法实时处理。作者借鉴了GPU加速3D体素映射技术、基于高度图的环境感知方法、RANSAC平面检测和GPU图像分割等现有工作，但创新性地将它们整合到一个专门针对足式机器人运动优化的框架中，通过并行计算解决了准确性和实时性的平衡问题。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是利用GPU并行计算能力实现高分辨率三维体素映射基础上的实时多平面分割。整体流程分为两大部分：1) 3D体素映射模块：接收传感器点云和机器人姿态，高效累积到体素地图中并通过射线投射移除动态物体；2) 多平面分割模块：将体素分类为可站立点和物体点，对可站立点进行聚类，对每个聚类进行平面参数估计，最后生成边界多边形。整个流程充分利用GPU并行能力，实现了超过30Hz的更新速率。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1) GPU加速的3D多平面分割，利用连通分量标记聚类和并行平面边界估计；2) 综合框架整合3D体素映射与多平面分割；3) 在5m×5m×5m范围内实现0.01米分辨率的实时处理。相比之前工作，这篇论文完全基于GPU实现（而非CPU-GPU混合），支持多种传感器配置，并引入簇级并行处理显著提高了多平面分割效率，同时解决了准确性和实时性的平衡问题。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '这篇论文提出了一种基于GPU加速的高分辨率3D体素映射的实时多平面分割方法，使足式机器人能够准确感知复杂3D环境中的多层平面表面，从而安全高效地完成运动任务。'}


### 论文摘要

This paper proposes a real-time multi-plane segmentation method based on GPU-accelerated high-resolution 3D voxel mapping for legged robot locomotion. Existing online planar mapping approaches struggle to balance accuracy and computational efficiency: direct depth image segmentation from specific sensors suffers from poor temporal integration, height map-based methods cannot represent complex 3D structures like overhangs, and voxel-based plane segmentation remains unexplored for real-time applications. To address these limitations, we develop a novel framework that integrates vertex-based connected component labeling with random sample consensus based plane detection and convex hull, leveraging GPU parallel computing to rapidly extract planar regions from point clouds accumulated in high-resolution 3D voxel maps. Experimental results demonstrate that the proposed method achieves fast and accurate 3D multi-plane segmentation at over 30 Hz update rate even at a resolution of 0.01 m, enabling the detected planes to be utilized in real time for locomotion tasks. Furthermore, we validate the effectiveness of our approach through experiments in both simulated environments and physical legged robot platforms, confirming robust locomotion performance when considering 3D planar structures.

---

## 36. AFFORD2ACT: Affordance-Guided Automatic Keypoint Selection for Generalizable and Lightweight Robotic Manipulation

**论文链接:** [http://arxiv.org/abs/2510.01433v1](http://arxiv.org/abs/2510.01433v1)

**作者:** Anukriti Singh, Kasra Torshizi, Khuzema Habib, Kelin Yu, Ruohan Gao, Pratap Tokekar

**发布时间:** 2025-10-01

### GPT解析

### 总结

本文提出了AFFORD2ACT框架，一种从文本提示和单张图像中提取语义2D关键点的功能引导方法，实现了高效且可扩展的视觉机器人学习。

### 背景

基于视觉的机器人学习通常依赖密集图像或点云输入，计算量大且包含不相关背景特征；现有关键点方法虽轻量但依赖手动启发式或与任务耦合选择，限制了可扩展性和语义理解。

### 目的

开发一个能够从文本提示和单张图像中提取最小语义2D关键点的框架，解决现有方法的局限性，提高数据效率和泛化能力。

### 方法

AFFORD2ACT框架采用三阶段流程：功能过滤、类别级别关键点构建和基于transformer的策略学习（带有嵌入式门控机制），生成38维状态策略，15分钟内完成训练。

### 主要发现

该方法在实时执行中表现良好，无需本体感受或密集表示；在多样化现实操作任务中持续提高数据效率，在未见物体、新类别、背景和干扰物上达到82%成功率。

### 结论

AFFORD2ACT是一个有效的框架，能够从文本提示和单张图像中提取语义关键点，在各种现实操作任务中表现出色，具有高数据效率和泛化能力。

### 翻译

基于视觉的机器人学习通常依赖于密集图像或点云输入，这些计算量大且包含不相关的背景特征。现有关键点方法虽然可以专注于操作相关的特征并且轻量级，但要么依赖于手动启发式规则，要么与任务耦合选择，限制了可扩展性和语义理解。为解决这一问题，我们提出了AFFORD2ACT，一个由功能引导的框架，可以从文本提示和单张图像中提取最小语义2D关键点。AFFORD2ACT遵循三阶段流程：功能过滤、类别级别关键点构建和基于transformer的策略学习（带有嵌入式门控机制），能够推理最相关的关键点，生成一个紧凑的38维状态策略，可以在15分钟内训练完成，在无需本体感受或密集表示的情况下实时表现良好。在多样化的现实操作任务中，AFFORD2ACT持续提高了数据效率，在未见过的物体、新类别、背景和干扰物上达到82%的成功率。


### 论文摘要

Vision-based robot learning often relies on dense image or point-cloud inputs, which are computationally heavy and entangle irrelevant background features. Existing keypoint-based approaches can focus on manipulation-centric features and be lightweight, but either depend on manual heuristics or task-coupled selection, limiting scalability and semantic understanding. To address this, we propose AFFORD2ACT, an affordance-guided framework that distills a minimal set of semantic 2D keypoints from a text prompt and a single image. AFFORD2ACT follows a three-stage pipeline: affordance filtering, category-level keypoint construction, and transformer-based policy learning with embedded gating to reason about the most relevant keypoints, yielding a compact 38-dimensional state policy that can be trained in 15 minutes, which performs well in real-time without proprioception or dense representations. Across diverse real-world manipulation tasks, AFFORD2ACT consistently improves data efficiency, achieving an 82% success rate on unseen objects, novel categories, backgrounds, and distractors.

---

## 37. To Augment or Not to Augment? Diagnosing Distributional Symmetry Breaking

**论文链接:** [http://arxiv.org/abs/2510.01349v1](http://arxiv.org/abs/2510.01349v1)

**作者:** Hannah Lawrence, Elyssa Hofgard, Vasco Portilheiro, Yuxuan Chen, Tess Smidt, Robin Walters

**发布时间:** 2025-10-01

**备注:** A short version of this paper appeared at the ICLR AI4Mat workshop in  April 2025

### GPT解析

### 总结

本文研究了对称性感知方法在机器学习中的应用，提出了一种评估数据集中对称性假设的方法，通过量化各向异性来分析对称性感知方法的效果，发现其效果取决于数据集特性。

### 背景

对称性感知方法（如数据增强和等变架构）鼓励模型对所有原始数据集的变换（如旋转或排列）表现出正确行为，可提高泛化能力和样本效率，但其有效性依赖于变换后的数据点在测试分布下具有高概率或'重要性'的假设。

### 目的

开发一种方法来批判性评估对称性感知方法所依赖的假设，即量化数据集中的各向异性或对称性破坏程度。

### 方法

提出一种度量指标，通过两样本神经分类器测试区分原始数据集及其随机增强的等效版本，在合成数据集上验证该指标，并应用于分析几个基准点云数据集。

### 主要发现

1) 理论上证明分布对称性破坏可阻止不变方法达到最优性能，即使底层标签真正不变；2) 实证研究发现对称性感知方法的效果是数据依赖的；3) 在几个基准点云数据集中发现了出人意料的高度对齐程度。

### 结论

理解等变性（包括它何时有效以及为何有效）可能需要重新思考数据中的对称性偏差，对称性感知方法的效果取决于数据集特性。

### 翻译

对称性感知的机器学习方法，如数据增强和等变架构，鼓励模型对所有原始数据集的变换（例如旋转或排列）表现出正确的行为。这些方法可以提高泛化能力和样本效率，其假设是变换后的数据点在测试分布下具有高概率或'重要性'。在本工作中，我们开发了一种方法来批判性评估这一假设。特别是，我们提出了一种度量指标，通过两样本神经分类器测试来区分原始数据集及其随机增强的等效版本，从而量化数据集中的各向异性或对称性破坏程度。我们在合成数据集上验证了我们的指标，然后使用它来揭示几个基准点云数据集中出人意料的高度对齐程度。我们从理论上证明，分布对称性破坏实际上可以阻止不变方法达到最优性能，即使底层标签真正是不变的，正如我们在无限特征限制下的不变岭回归中所展示的那样。从经验上看，我们发现对称性感知方法的含义是数据依赖的：等变方法仍然在某些各向异性数据集上带来好处，但在其他数据集上则没有。总的来说，这些发现表明，理解等变性——无论它何时有效以及为何有效——可能需要重新思考数据中的对称性偏差。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': "论文主要解决数据增强和等变方法在应用中的核心假设问题，即变换后的数据点在测试分布下是否具有高概率或'重要性'。这个问题很重要，因为错误地应用对称性假设可能导致模型性能下降，理解数据中的对称性破坏有助于更有效地应用数据增强和等变方法，提高模型泛化能力和样本效率，同时指导模型设计。", '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者通过识别对称感知方法中隐含的假设局限性，从理论角度分析对称性破坏对等变方法的影响，进而提出基于双样本分类器测试的度量方法。作者借鉴了Lopez-Paz & Oquab的双样本分类器测试方法用于分布差异检测，参考了Chiu & Bloem-Reddy的对称性检测工作但避免了核选择问题，同时基于现有的等变机器学习方法进行研究但关注其在对称性破坏情况下的局限性。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是通过训练一个分类器来区分原始数据点和经过随机变换的数据点，分类器的准确率作为对称性破坏程度的度量。实现流程包括：1) 输入未标记数据集和群组；2) 将数据集分成两部分并对一部分应用随机变换；3) 构建二分类数据集(原始样本标记0，变换样本标记1)；4) 训练二分类器；5) 返回分类器在测试集上的准确率，值越接近1表示对称性破坏程度越高，越接近0.5表示数据越对称。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1) 提出无需特定领域知识的分布对称性破坏度量方法；2) 提供在对称性破坏情况下不变性岭回归的理论分析；3) 在多个基准数据集上发现高度对称性破坏并评估等变方法效果；4) 探索局部对称性破坏概念。相比之前工作，作者的方法避免了核选择问题，扩展了理论分析到非对称数据分布，在更广泛数据集上验证了方法，明确区分了分布对称性破坏和功能对称性破坏，并引入了任务相关的对称性破坏度量。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '这篇论文提出了一种量化数据集中分布对称性破坏的新方法，并通过理论和实验揭示了等变机器学习方法在对称性破坏数据集上的复杂行为，为何时以及如何应用对称性假设提供了重要见解。'}


### 论文摘要

Symmetry-aware methods for machine learning, such as data augmentation and equivariant architectures, encourage correct model behavior on all transformations (e.g. rotations or permutations) of the original dataset. These methods can improve generalization and sample efficiency, under the assumption that the transformed datapoints are highly probable, or "important", under the test distribution. In this work, we develop a method for critically evaluating this assumption. In particular, we propose a metric to quantify the amount of anisotropy, or symmetry-breaking, in a dataset, via a two-sample neural classifier test that distinguishes between the original dataset and its randomly augmented equivalent. We validate our metric on synthetic datasets, and then use it to uncover surprisingly high degrees of alignment in several benchmark point cloud datasets. We show theoretically that distributional symmetry-breaking can actually prevent invariant methods from performing optimally even when the underlying labels are truly invariant, as we show for invariant ridge regression in the infinite feature limit. Empirically, we find that the implication for symmetry-aware methods is dataset-dependent: equivariant methods still impart benefits on some anisotropic datasets, but not others. Overall, these findings suggest that understanding equivariance -- both when it works, and why -- may require rethinking symmetry biases in the data.

---

## 38. Approximate mean curvature flows of a general varifold, and their limit spacetime Brakke flow

**论文链接:** [http://arxiv.org/abs/2510.00746v1](http://arxiv.org/abs/2510.00746v1)

**作者:** Blanche Buet, Gian Paolo Leonardi, Simon Masnou, Abdelmouksit Sagueni

**发布时间:** 2025-10-01

### GPT解析

### 总结

提出了一种基于变分流形理论的近似方法，用于构建适用于非常一般初始数据的平均曲率流。

### 背景

基于Brakke和Kim & Tonegawa的工作，利用变分流形理论构建平均曲率流。

### 目的

为任意维度和余维度的非常一般结构（从连续表面到离散点云）提供一种近似平均曲率流的概念。

### 方法

通过迭代前向构造一个依赖于给定时间步长和近似参数的近似时间离散平均曲率流。

### 主要发现

当时间步长趋于0时，时间离散流收敛到唯一的近似平均曲率流；该流满足稳定性、唯一性、Brakke型等式和质量衰减等性质；当近似参数趋于0时，与规范时间测度耦合收敛到广义平均曲率有界的时空极限测度。

### 结论

在可求长性假设下，极限测度是一个时空Brakke流，为一般结构的平均曲率流提供了理论基础。

### 翻译

我们提出了一种通过近似方法构建平均曲率流的构造方法，适用于非常一般的初始数据，遵循Brakke和Kim & Tonegawa基于变分流形理论的工作精神。给定一个一般的变分流形，我们通过迭代前向构造一个依赖于给定时间步长和近似参数的近似时间离散平均曲率流。我们证明，当时间步长趋于0时，这个时间离散流收敛到一个唯一的极限流，我们称之为近似平均曲率流。我们方法的一个有趣特点是它的普适性，因为它为任意维度和余维度的非常一般的结构提供了平均曲率流的近似概念，从连续表面到离散点云。我们证明，我们的近似平均曲率流满足几个性质：稳定性、唯一性、Brakke型等式、质量衰减。通过将这种近似流与规范时间测度耦合，我们证明，当近似参数趋于0时，收敛到一个时空极限测度，其广义平均曲率有界。在额外的可求长性假设下，我们进一步证明这个极限测度是一个时空Brakke流。


### 论文摘要

We propose a construction of mean curvature flows by approximation for very general initial data, in the spirit of the works of Brakke and of Kim & Tonegawa based on the theory of varifolds. Given a general varifold, we construct by iterated push-forwards an approximate time-discrete mean curvature flow depending on both a given time step and an approximation parameter. We show that, as the time step tends to $0$, this time-discrete flow converges to a unique limit flow, which we call the approximate mean curvature flow. An interesting feature of our approach is its generality, as it provides an approximate notion of mean curvature flow for very general structures of any dimension and codimension, ranging from continuous surfaces to discrete point clouds. We prove that our approximate mean curvature flow satisfies several properties: stability, uniqueness, Brakke-type equality, mass decay. By coupling this approximate flow with the canonical time measure, we prove convergence, as the approximation parameter tends to $0$, to a spacetime limit measure whose generalized mean curvature is bounded. Under an additional rectifiability assumption, we further prove that this limit measure is a spacetime Brakke flow.

---

## 39. From 2D to 3D, Deep Learning-based Shape Reconstruction in Magnetic Resonance Imaging: A Review

**论文链接:** [http://arxiv.org/abs/2510.01296v1](http://arxiv.org/abs/2510.01296v1)

**作者:** Emma McMillian, Abhirup Banerjee, Alfonso Bueno-Orovio

**发布时间:** 2025-10-01

### GPT解析

### 总结

这篇综述论文调查了基于深度学习的3D MRI重建方法，重点关注四种主要方法：点云、基于网格、形状感知和体积模型，分析了它们的技术基础、限制和应用领域。

### 背景

基于深度学习的3D形状重建从2D磁共振成像技术在医学疾病诊断、治疗计划和计算建模中变得越来越重要。

### 目的

为研究人员提供当前3D重建方法的结构化概述，以识别推进深度学习向更强大、可推广和具有临床影响力的解决方案的机会。

### 方法

分析四种主要3D重建方法（点云、基于网格、形状感知和体积模型）的最新技术、方法论基础和限制；考察这些方法在心脏、神经和肺部等不同解剖结构中的应用；评估模型对病理解剖的临床适用性及其训练和测试数据的影响；检查公开数据集、计算需求和评估指标。

### 主要发现

各种3D重建方法在不同解剖结构和疾病应用中各有优势和限制；训练数据的选择和模型计算需求对重建质量有显著影响；多模态集成和跨模态框架是新兴的研究方向。

### 结论

深度学习在3D MRI重建领域展现出巨大潜力，未来的研究应关注提高方法的鲁棒性、可推广性和临床实用性，同时探索多模态融合等创新方向。

### 翻译

基于深度学习的从二维磁共振成像到三维形状重建在医学疾病诊断、治疗计划和计算建模中变得越来越重要。本综述调查了3D MRI重建的方法论现状，重点关注四种主要方法：点云、基于网格、形状感知和体积模型。对于每个类别，我们分析了当前最先进的技术、其方法论基础、限制以及在不同解剖结构中的应用。我们提供了从心脏、神经到肺部成像的广泛概述。我们还关注模型对病理解剖的临床适用性及其训练和测试数据的影响。我们检查了公开可用的数据集、计算需求和评估指标。最后，我们突出了包括多模态集成和跨模态框架在内的新兴研究方向。本综述旨在为研究人员提供当前3D重建方法的概述，以识别推进深度学习向更强大、可推广和具有临床影响力的解决方案的机会。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决如何从2D MRI图像重建出准确的3D形状的问题。这个问题在现实中非常重要，因为准确的3D器官表示能为疾病状态和功能提供无与伦比的见解，使医生能够进行更精确、有效和个性化的治疗。虽然医生能够解释2D MRI图像并在脑海中构建3D解剖结构，但使用深度学习直接生成3D模型可以提高效率和准确性，特别是在处理扫描质量差异、解剖多样性和数据稀缺性等挑战时。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '这是一篇综述文章，作者通过系统性地回顾和分析现有研究来组织内容。作者将现有的3D MRI重建方法分为四类：点云方法、网格方法、形状感知方法和体积模型方法。对于每类方法，作者分析了最先进技术、方法基础、局限性和应用场景。作者借鉴了大量现有工作，包括CNN、GAN和扩散模型等架构，并参考了其他综述文章，同时指出了这些综述的局限性，如专注于特定器官（如心脏）或缺乏对临床适用性的讨论。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '由于这是一篇综述文章，没有单一的核心思想或实现流程。相反，论文综述了多种方法的核心思想和实现流程：1）点云方法将解剖结构表示为3D空间中的点集合，通过编码器-解码器架构生成全局和局部特征；2）网格方法生成连续表面表示，使用互连的顶点、边和面形成多边形网格；3）形状感知方法利用统计或学习先验约束输出空间，确保解剖合理性；4）体积方法生成基于体素的完整3D表示，通常结合数据驱动先验和基于模型的重建技术。这些方法通常从2D MRI切片开始，通过深度学习模型生成3D表示，最终输出可以是点云、网格或体积数据。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '作为一篇综述文章，其创新点在于综述的方法和视角：1）提供了更全面的综述范围，涵盖从心脏到神经到肺部成像的广泛应用，而非专注于特定器官；2）特别关注模型对病理解剖的临床适用性和训练数据的影响；3）强调多模态集成和跨模态框架等新兴研究方向；4）评估了当前评估指标的有效性并提供了未来研究建议；5）探讨了非MRI重建方法（如动画和自然图像处理）在MRI重建中的潜在应用。相比其他综述，本文更注重不同器官间的技术转移机会和临床实用性。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '这篇论文提供了基于深度学习的3D MRI重建技术的全面综述，系统分析了四类主要方法，为研究人员指出了推进深度学习向更稳健、可泛化和具有临床影响力的3D重建解决方案的机会。'}


### 论文摘要

Deep learning-based 3-dimensional (3D) shape reconstruction from 2-dimensional (2D) magnetic resonance imaging (MRI) has become increasingly important in medical disease diagnosis, treatment planning, and computational modeling. This review surveys the methodological landscape of 3D MRI reconstruction, focusing on 4 primary approaches: point cloud, mesh-based, shape-aware, and volumetric models. For each category, we analyze the current state-of-the-art techniques, their methodological foundation, limitations, and applications across anatomical structures. We provide an extensive overview ranging from cardiac to neurological to lung imaging. We also focus on the clinical applicability of models to diseased anatomy, and the influence of their training and testing data. We examine publicly available datasets, computational demands, and evaluation metrics. Finally, we highlight the emerging research directions including multimodal integration and cross-modality frameworks. This review aims to provide researchers with a structured overview of current 3D reconstruction methodologies to identify opportunities for advancing deep learning towards more robust, generalizable, and clinically impactful solutions.

---

## 40. VarCoNet: A variability-aware self-supervised framework for functional connectome extraction from resting-state fMRI

**论文链接:** [http://arxiv.org/abs/2510.02120v1](http://arxiv.org/abs/2510.02120v1)

**作者:** Charalampos Lamprou, Aamna Alshehhi, Leontios J. Hadjileontiadis, Mohamed L. Seghier

**发布时间:** 2025-10-02

**备注:** My preview .pdf was not loading. Can you please share with me a  compiled .pdf file so I can confirm that the result is correct?

### GPT解析

### 总结

论文介绍了一种名为VarCoNet的新型自监督框架，用于从静息态功能磁共振成像数据中稳健提取功能连接组，该框架将个体间功能变异性视为有价值信息而非噪声。

### 背景

个体间脑功能变异性是精准医疗的关键因素，但传统方法通常将其视为噪声而非有意义的数据。

### 目的

开发能够有效利用个体间功能变异性来提取功能连接组的框架，生成适用于下游任务的FC嵌入，即使在缺乏标记数据的情况下也能工作。

### 方法

VarCoNet采用自监督对比学习框架，结合基于静息态fMRI信号分割的新型增强策略，核心是1D-CNN-Transformer编码器用于时间序列处理，并集成了贝叶斯超参数优化。

### 主要发现

在受试者指纹识别和自闭症谱系障碍分类两个下游任务上，使用不同脑区分割方法，与13种最先进方法相比，VarCoNet显示出优越性、稳健性、可解释性和泛化能力。

### 结论

VarCoNet为静息态fMRI中的功能连接组分析提供了一个多功能且稳健的框架。

### 翻译

考虑个体间脑功能变异性是精准医疗的关键。在这里，通过将功能个体间变异性视为有意义的数据而非噪声，我们引入了VarCoNet，这是一种增强的自监督框架，用于从静息态功能磁共振成像数据中稳健地提取功能连接组。VarCoNet采用自监督对比学习来利用内在的功能个体间变异性，充当脑功能编码器，生成可直接适用于下游任务的FC嵌入，即使在没有标记数据的情况下也是如此。对比学习通过一种基于静息态fMRI信号分割的新型增强策略得到促进。其核心是，VarCoNet集成了一个1D-CNN-Transformer编码器用于高级时间序列处理，并增强了稳健的贝叶斯超参数优化。我们的VarCoNet框架在两个下游任务上进行了评估：（i）使用人类连接组计划的rs-fMRI数据进行受试者指纹识别，以及（ii）使用ABIDE I和ABIDE II数据集的rs-fMRI数据进行自闭症谱系障碍分类。使用不同的脑区分割方法，我们与包括13种深度学习方法在内的最先进方法进行了广泛测试，证明了VarCoNet的优越性、稳健性、可解释性和泛化能力。总体而言，VarCoNet为rs-fMRI中的FC分析提供了一个多功能且稳健的框架。


### 论文摘要

Accounting for inter-individual variability in brain function is key to precision medicine. Here, by considering functional inter-individual variability as meaningful data rather than noise, we introduce VarCoNet, an enhanced self-supervised framework for robust functional connectome (FC) extraction from resting-state fMRI (rs-fMRI) data. VarCoNet employs self-supervised contrastive learning to exploit inherent functional inter-individual variability, serving as a brain function encoder that generates FC embeddings readily applicable to downstream tasks even in the absence of labeled data. Contrastive learning is facilitated by a novel augmentation strategy based on segmenting rs-fMRI signals. At its core, VarCoNet integrates a 1D-CNN-Transformer encoder for advanced time-series processing, enhanced with a robust Bayesian hyperparameter optimization. Our VarCoNet framework is evaluated on two downstream tasks: (i) subject fingerprinting, using rs-fMRI data from the Human Connectome Project, and (ii) autism spectrum disorder (ASD) classification, using rs-fMRI data from the ABIDE I and ABIDE II datasets. Using different brain parcellations, our extensive testing against state-of-the-art methods, including 13 deep learning methods, demonstrates VarCoNet's superiority, robustness, interpretability, and generalizability. Overall, VarCoNet provides a versatile and robust framework for FC analysis in rs-fMRI.

---

## 41. SLAP: Learning Speaker and Health-Related Representations from Natural Language Supervision

**论文链接:** [http://arxiv.org/abs/2510.01860v1](http://arxiv.org/abs/2510.01860v1)

**作者:** Angelika Ando, Auguste Crabeil, Adrien Lesage, Rachid Riad

**发布时间:** 2025-10-02

### GPT解析

### 总结

本研究提出了SLAP模型，这是首个通过对比学习将语音与说话者和健康元数据的自然语言描述对齐的音频基础模型。该模型在多种语言和任务上展示了强大的零样本和分布外泛化能力，特别是在健康相关任务上表现优异。

### 背景

语音编码了副语言学信息如人口统计、语音质量和健康信息，但目前没有音频基础模型支持这些任务的零样本或分布外（OOD）泛化。

### 目的

开发一个能够将语音与自然语言描述对齐的模型，实现对副语言学信息的零样本和分布外泛化，特别是在人口统计、语音特征和健康评估任务上。

### 方法

提出SLAP（Speaker contrastive Language-Audio Pretraining）模型，结合Vision Transformer音频编码器和文本编码器，通过对比学习对齐语音与自然语言描述。在9个数据集上使用超过3400小时的数据进行训练，涵盖多样化的说话者注释。

### 主要发现

SLAP在零样本评估中实现62.9%的平均F1分数，比CLAP相对提高48%；展示了在未见语言和临床人群上的强OOD泛化能力；通过线性探测微调后总体达到69.3%的F1分数，在健康任务上实现57.9%的F1分数，表现优于更大的基础模型。

### 结论

SLAP是首个能够有效对齐语音与自然语言描述的音频基础模型，在多种副语言学任务上展示了强大的零样本和分布外泛化能力，特别是在健康相关任务上实现了最先进的性能。

### 翻译

语音编码了副语言学信息，如人口统计、语音质量和健康信息。然而，目前没有音频基础模型支持这些任务的零样本或分布外（OOD）泛化。我们介绍了SLAP（Speaker contrastive Language-Audio Pretraining），这是第一个通过对比学习将语音与说话者和健康元数据的自然语言描述对齐的模型。SLAP结合了Vision Transformer音频编码器和文本编码器，在9个具有不同说话者注释的数据集上进行了超过3400小时的训练。我们在7种语言的14个数据集上的38个二元分类任务（涵盖人口统计、语音特征和临床评估）上进行了评估。SLAP在零样本评估中实现了62.9%的平均F1分数，比CLAP（42.4%）相对提高了48%，同时展示了在未见过的语言和临床人群上的强OOD泛化能力。当通过线性探测微调时，SLAP总体上达到69.3%的F1分数，并在健康任务上实现了最先进的性能（57.9% F1），超过了更大的基础模型。


### 论文摘要

Speech encodes paralinguistic information such as demographics, voice quality, and health. Yet no audio foundation model supports zero-shot or out-of-distribution (OOD) generalization to these tasks. We introduce SLAP (Speaker contrastive Language-Audio Pretraining), the first model aligning speech with natural language descriptions of speaker and health metadata through contrastive learning. SLAP combines a Vision Transformer audio encoder with text encoders, trained on more than 3400 hours across 9 datasets with diverse speaker annotations. We evaluated on 38 binary classification tasks spanning demographics, voice characteristics, and clinical assessments across 14 datasets in 7 languages. SLAP achieves 62.9% average F1 in zero-shot evaluation, a 48% relative improvement over CLAP (42.4%), while demonstrating strong OOD generalization to unseen languages and clinical populations. When fine-tuned with linear probing, SLAP reaches 69.3% F1 overall and achieves best-in-class performance on health tasks (57.9% F1), surpassing larger foundation models.

---

## 42. Learning to Look at the Other Side: A Semantic Probing Study of Word Embeddings in LLMs with Enabled Bidirectional Attention

**论文链接:** [http://arxiv.org/abs/2510.01652v1](http://arxiv.org/abs/2510.01652v1)

**作者:** Zhaoxin Feng, Jianfei Ma, Emmanuele Chersoni, Xiaojing Zhao, Xiaoyi Bao

**发布时间:** 2025-10-02

### GPT解析

### 总结

本研究探讨了在大型语言模型中启用双向注意力机制，以克服单向注意力机制在文本嵌入任务和语义表示分析方面的局限性。

### 背景

自回归大型语言模型在语言理解和生成方面表现出色，但由于单向注意力机制的约束，在文本嵌入任务中的应用发展较慢，并且在探测任务中的语义表示分析也相对滞后。

### 目的

探索是否可以通过在大型语言模型中启用双向注意力来克服这些约束。

### 方法

通过对Llama架构的不同变体进行额外的训练步骤，逐步启用双向注意力，并结合无监督/监督对比学习进行测试。

### 主要发现

摘要中未明确提及具体发现。

### 结论

摘要中未明确提及具体结论。

### 翻译

自回归大型语言模型在语言理解和生成方面表现出色。然而，由于单向注意力机制的约束，它们在文本嵌入任务中的应用相对较慢，同时在探测任务中对其语义表示的分析也较为滞后。本文旨在探索是否可以通过在大型语言模型中启用双向注意力来克服这些约束。我们通过对Llama架构的不同变体进行额外的训练步骤，逐步启用双向注意力，并结合无监督/监督对比学习进行测试。


### 论文摘要

Autoregressive Large Language Models (LLMs) demonstrate exceptional performance in language understanding and generation. However, their application in text embedding tasks has been relatively slow, along with the analysis of their semantic representation in probing tasks, due to the constraints of the unidirectional attention mechanism.   This paper aims to explore whether such constraints can be overcome by enabling bidirectional attention in LLMs. We tested different variants of the Llama architecture through additional training steps, progressively enabling bidirectional attention and unsupervised/supervised contrastive learning.

---

## 43. It Takes Two: Your GRPO Is Secretly DPO

**论文链接:** [http://arxiv.org/abs/2510.00977v1](http://arxiv.org/abs/2510.00977v1)

**作者:** Yihong Wu, Liheng Ma, Lei Ding, Muzhi Li, Xinyu Wang, Kejia Chen, Zhan Su, Zhanguang Zhang, Chenyang Huang, Yingxue Zhang, Mark Coates, Jian-Yun Nie

**发布时间:** 2025-10-01

### GPT解析

### 总结

本文挑战了GRPO算法需要大组大小的传统假设，通过将GRPO重新构建为对比学习形式，揭示了其与DPO的联系，并验证了仅需两个轮次的2-GRPO可以达到与16-GRPO相当的性能，同时大幅减少计算开销。

### 背景

GRPO是一种用于训练后大型语言模型(LLMs)的重要强化学习算法。传统观点认为，GRPO需要较大的组大小以确保通过精确的统计估计实现稳定训练，但这会导致显著的计算开销。

### 目的

挑战GRPO需要大组大小的假设，重新构建GRPO为对比学习形式，揭示其与DPO的联系，并调查之前被认为不可行的最小两轮案例(2-GRPO)的可行性。

### 方法

将GRPO重新构建为对比学习形式，提供严格的理论分析验证2-GRPO，并进行实证研究比较2-GRPO与16-GRPO的性能差异。

### 主要发现

2-GRPO(使用最小化的两个轮次)与16-GRPO性能相当，仅使用1/8的轮次，同时将训练时间减少了70%以上。

### 结论

GRPO不一定需要大组大小也能实现良好性能，2-GRPO是一种高效可行的替代方案，显著降低了计算成本。

### 翻译

组相对策略优化(GRPO)是一种用于训练后大型语言模型(LLMs)的突出强化学习算法。人们普遍认为，GRPO需要较大的组大小来确保通过精确的统计估计实现稳定训练，这会导致大量的计算开销。在这项工作中，我们通过将GRPO重新构建为对比学习形式来挑战这一假设，这揭示了GRPO与直接偏好优化(DPO)的基本联系。受DPO实证成功的启发，我们调查了最小化的两轮案例(2-GRPO)，这是一种之前被认为不可行的配置。我们提供了严格的理论分析来验证2-GRPO，并通过实证证明，尽管仅使用1/8的轮次并将训练时间减少70%以上，2-GRPO的性能与16-GRPO相当。


### 论文摘要

Group Relative Policy Optimization (GRPO) is a prominent reinforcement learning algorithm for post-training Large Language Models (LLMs). It is commonly believed that GRPO necessitates a large group size to ensure stable training via precise statistical estimation, which incurs substantial computational overhead. In this work, we challenge this assumption by reframing GRPO as a form of contrastive learning, which reveals a fundamental connection to Direct Preference Optimization (DPO). Motivated by DPO's empirical success, we investigate the minimal two-rollout case (2-GRPO), a configuration previously deemed infeasible. We provide a rigorous theoretical analysis to validate 2-GRPO and demonstrate empirically that it achieves performance on par with 16-GRPO, despite using only 1/8 of the rollouts and reducing training time by over 70%.

---

## 44. Span-level Detection of AI-generated Scientific Text via Contrastive Learning and Structural Calibration

**论文链接:** [http://arxiv.org/abs/2510.00890v1](http://arxiv.org/abs/2510.00890v1)

**作者:** Zhen Yin, Shenghua Wang

**发布时间:** 2025-10-01

### GPT解析

### 总结

该论文提出了Sci-SpanDet框架，一个用于检测学术文本中AI生成内容的结构感知方法。该方法结合了基于章节的条件风格建模和多层次对比学习，捕捉人类与AI文本的细微差异，同时减少主题依赖性。此外，它集成了BIO-CRF序列标记和置信度校准，实现精确的片段级检测。实验表明，该框架在跨学科数据集上达到最先进性能，并在对抗性重写下表现出强鲁棒性。

### 背景

大型语言模型在科学写作中的快速采用引发了关于作者完整性和学术出版物可靠性的严重担忧。现有检测方法主要依赖文档级分类或表面统计线索，但忽略了细粒度片段定位，校准能力弱，且难以跨学科和跨生成器泛化。

### 目的

解决现有AI生成文本检测方法的局限性，包括忽视细粒度片段定位、校准能力弱以及跨学科和跨生成器泛化能力差的问题。

### 方法

提出Sci-SpanDet框架，结合基于章节的条件风格建模与多层次对比学习，捕捉人类与AI文本的细微差异，同时缓解主题依赖。集成BIO-CRF序列标记、基于指针的边界解码和置信度校准，实现精确的片段级检测和可靠概率估计。

### 主要发现

在包含100,000个标注样本的跨学科数据集上，Sci-SpanDet达到最先进性能：F1(AI)为80.17，AUROC为92.63，Span-F1为74.36。该方法在对抗性重写下表现出强鲁棒性，在IMRaD部分和不同学科中保持平衡准确率，显著优于现有基线。

### 结论

Sci-SpanDet是有效的AI生成学术文本检测框架，能精确识别AI生成内容并在各种条件下保持高准确率。为促进该领域研究，整理好的数据集和源代码将在发表后公开。

### 翻译

大型语言模型在科学写作中的快速采用引发了关于作者完整性和学术出版物可靠性的严重担忧。现有的检测方法主要依赖于文档级分类或表面级别的统计线索；然而，它们忽略了细粒度的片段定位，校准能力弱，并且往往无法跨学科和跨生成器泛化。为了解决这些局限性，我们提出了Sci-SpanDet，一个用于检测AI生成学术文本的结构感知框架。该方法结合了基于章节的条件风格建模与多层次对比学习，捕捉人类与AI生成文本的细微差异，同时缓解主题依赖，从而增强跨领域鲁棒性。此外，它还集成了BIO-CRF序列标记、基于指针的边界解码和置信度校准，以实现精确的片段级检测和可靠的概率估计。


### 论文摘要

The rapid adoption of large language models (LLMs) in scientific writing raises serious concerns regarding authorship integrity and the reliability of scholarly publications. Existing detection approaches mainly rely on document-level classification or surface-level statistical cues; however, they neglect fine-grained span localization, exhibit weak calibration, and often fail to generalize across disciplines and generators. To address these limitations, we present Sci-SpanDet, a structure-aware framework for detecting AI-generated scholarly texts. The proposed method combines section-conditioned stylistic modeling with multi-level contrastive learning to capture nuanced human-AI differences while mitigating topic dependence, thereby enhancing cross-domain robustness. In addition, it integrates BIO-CRF sequence labeling with pointer-based boundary decoding and confidence calibration to enable precise span-level detection and reliable probability estimates. Extensive experiments on a newly constructed cross-disciplinary dataset of 100,000 annotated samples generated by multiple LLM families (GPT, Qwen, DeepSeek, LLaMA) demonstrate that Sci-SpanDet achieves state-of-the-art performance, with F1(AI) of 80.17, AUROC of 92.63, and Span-F1 of 74.36. Furthermore, it shows strong resilience under adversarial rewriting and maintains balanced accuracy across IMRaD sections and diverse disciplines, substantially surpassing existing baselines. To ensure reproducibility and to foster further research on AI-generated text detection in scholarly documents, the curated dataset and source code will be publicly released upon publication.

---

## 45. Feature Identification for Hierarchical Contrastive Learning

**论文链接:** [http://arxiv.org/abs/2510.00837v1](http://arxiv.org/abs/2510.00837v1)

**作者:** Julius Ott, Nastassia Vysotskaya, Huawei Sun, Lorenzo Servadei, Robert Wille

**发布时间:** 2025-10-01

**备注:** Submitted to ICASSP 2026

### GPT解析

### 总结

本文提出两种新颖的分层对比学习方法，通过建模类别间关系和不均衡分布，在多个层级实现细粒度聚类，并在标准数据集上取得最先进性能。

### 背景

分层分类是许多应用中的关键任务，对象被组织成多个类别的层次结构。

### 目的

解决传统分类方法忽视不同层次类别间内在关系的问题，开发能充分利用层次结构信息的新方法。

### 方法

提出两种分层对比学习方法：第一种利用高斯混合模型（G-HMLC），第二种使用注意力机制捕获层次特定特征（A-HMLC），模仿人类处理过程。

### 主要发现

所提方法明确建模了类别间关系和高层级的不均衡类别分布，实现了所有层级的细粒度聚类，在CIFAR100和ModelNet40数据集上线性评估准确率比现有方法提高2个百分点。

### 结论

方法通过定量和定性结果证明了有效性，突显了其在计算机视觉及其他领域应用的潜力。

### 翻译

分层分类是许多应用中的关键任务，其中对象被组织成多个类别的层次结构。然而，传统分类方法常常忽视不同层次类别间的内在关系，从而错过了重要的监督信号。因此，我们提出了两种新颖的分层对比学习方法。第一种利用高斯混合模型，第二种使用注意力机制捕获层次特定特征，模仿人类处理过程。我们的方法明确建模了类别间关系和高层级的不均衡类别分布，实现了所有层级的细粒度聚类。在具有竞争力的CIFAR100和ModelNet40数据集上，我们的方法在线性评估中取得了最先进的性能，在准确率上比现有的分层对比学习方法提高了2个百分点。我们方法的有效性得到了定量和定性结果的支持，突显了其在计算机视觉及其他领域应用的潜力。


### 论文摘要

Hierarchical classification is a crucial task in many applications, where objects are organized into multiple levels of categories. However, conventional classification approaches often neglect inherent inter-class relationships at different hierarchy levels, thus missing important supervisory signals. Thus, we propose two novel hierarchical contrastive learning (HMLC) methods. The first, leverages a Gaussian Mixture Model (G-HMLC) and the second uses an attention mechanism to capture hierarchy-specific features (A-HMLC), imitating human processing. Our approach explicitly models inter-class relationships and imbalanced class distribution at higher hierarchy levels, enabling fine-grained clustering across all hierarchy levels. On the competitive CIFAR100 and ModelNet40 datasets, our method achieves state-of-the-art performance in linear evaluation, outperforming existing hierarchical contrastive learning methods by 2 percentage points in terms of accuracy. The effectiveness of our approach is backed by both quantitative and qualitative results, highlighting its potential for applications in computer vision and beyond.

---

## 46. HAMLET: Switch your Vision-Language-Action Model into a History-Aware Policy

**论文链接:** [http://arxiv.org/abs/2510.00695v2](http://arxiv.org/abs/2510.00695v2)

**作者:** Myungkyu Koo, Daewon Choi, Taeyoung Kim, Kyungmin Lee, Changyeon Kim, Younggyo Seo, Jinwoo Shin

**发布时间:** 2025-10-01

**备注:** Project page: https://myungkyukoo.github.io/hamlet/

### GPT解析

### 总结

HAMLET是一种可扩展的框架，使视觉-语言-动作模型(VLAs)能够在动作预测过程中关注历史上下文，通过引入时刻令牌和轻量级记忆模块，显著提高了在历史依赖任务上的性能。

### 背景

机器人操作任务本质上是依赖于历史的，利用过去的上下文可能是有益的。然而，大多数现有的VLAs在设计时没有考虑这一方面，它们仅依赖于当前观察，而忽略先前的上下文。

### 目的

提出HAMLET框架，使VLAs能够在动作预测过程中关注历史上下文，提高在历史依赖任务上的性能。

### 方法

引入时刻令牌紧凑编码每个时间步的感知信息，通过时间对比学习初始化这些表示，并采用轻量级记忆模块将过去时间步的时刻令牌整合为记忆特征，用于动作预测。

### 主要发现

HAMLET成功将最先进的VLA转变为历史感知策略，在历史依赖的真实世界任务上实现了76.4%的平均成功率，比基线提高了47.2%；在RoboCasa Kitchen上将性能从64.1%提高到66.4%；在LIBERO上将性能从95.6%提高到97.7%。

### 结论

HAMLET是一个有效的框架，能够使VLAs关注历史上下文，在各种任务上取得了显著的性能提升，特别是在长周期和历史依赖任务上表现出色。

### 翻译

本质上，机器人操作任务是依赖于历史的：利用过去的上下文可能是有益的。然而，大多数现有的视觉-语言-动作模型(VLAs)在设计时没有考虑这一方面，即它们仅依赖于当前观察，而忽略先前的上下文。在本文中，我们提出了HAMLET，一个可扩展的框架，使VLAs能够在动作预测过程中关注历史上下文。具体来说，我们引入了时刻令牌，它们紧凑地编码每个时间步的感知信息。它们的表示通过时间对比学习进行初始化，使其能够更好地捕捉时间上独特的方面。接下来，我们采用一个轻量级记忆模块，将过去时间步的时刻令牌整合为记忆特征，然后利用这些特征进行动作预测。通过经验评估，我们表明HAMLET成功地将最先进的VLA转变为历史感知策略，特别是在需要历史上下文的长周期任务上显示出显著改进。特别是在基于GR00T N1.5的实验中，HAMLET在历史依赖的真实世界任务上实现了76.4%的平均成功率，超过了基线性能47.2%。此外，HAMLET将RoboCasa Kitchen(100-demo设置)上的先前艺术性能从64.1%提高到66.4%，将LIBERO上的性能从95.6%提高到97.7%，突显了其在通用机器人操作基准测试中的有效性。


### 论文摘要

Inherently, robotic manipulation tasks are history-dependent: leveraging past context could be beneficial. However, most existing Vision-Language-Action models (VLAs) have been designed without considering this aspect, i.e., they rely solely on the current observation, ignoring preceding context. In this paper, we propose HAMLET, a scalable framework to adapt VLAs to attend to the historical context during action prediction. Specifically, we introduce moment tokens that compactly encode perceptual information at each timestep. Their representations are initialized with time-contrastive learning, allowing them to better capture temporally distinctive aspects. Next, we employ a lightweight memory module that integrates the moment tokens across past timesteps into memory features, which are then leveraged for action prediction. Through empirical evaluation, we show that HAMLET successfully transforms a state-of-the-art VLA into a history-aware policy, especially demonstrating significant improvements on long-horizon tasks that require historical context. In particular, on top of GR00T N1.5, HAMLET achieves an average success rate of 76.4% on history-dependent real-world tasks, surpassing the baseline performance by 47.2%. Furthermore, HAMLET pushes prior art performance from 64.1% to 66.4% on RoboCasa Kitchen (100-demo setup) and from 95.6% to 97.7% on LIBERO, highlighting its effectiveness even under generic robot-manipulation benchmarks.

---

## 47. ARIONet: An Advanced Self-supervised Contrastive Representation Network for Birdsong Classification and Future Frame Prediction

**论文链接:** [http://arxiv.org/abs/2510.00522v2](http://arxiv.org/abs/2510.00522v2)

**作者:** Md. Abdur Rahman, Selvarajah Thuseethan, Kheng Cher Yeo, Reem E. Mohamed, Sami Azam

**发布时间:** 2025-10-01

### GPT解析

### 总结

本文提出了一种名为ARIONet的自监督对比网络，用于自动鸟类声音分类，通过联合优化对比分类和未来帧预测来学习具有区分性的物种特定表示，无需大规模标注数据。

### 背景

自动鸟类声音分类对生态监测和生物多样性研究至关重要，但现有方法存在严重依赖标记数据、使用有限特征表示、忽略物种识别所需时间动态等问题。

### 目的

设计一个自监督对比网络ARIONet，通过增强音频表示联合优化对比分类和未来帧预测，并在基于transformer的编码器模型中集成多种互补音频特征。

### 方法

ARIONet实现两个关键目标：(1)通过最大化同一音频段增强视图间的相似性同时分离不同样本来学习区分性物种表示；(2)通过预测未来音频帧建模时间动态，两者均无需大规模标注。

### 主要发现

在四个数据集上验证了框架性能，分类准确率分别为98.41%、93.07%、91.89%和91.58%，F1分数分别为97.84%、94.10%、91.29%和90.94%，未来帧预测任务中余弦相似度高达95%。

### 结论

自监督学习策略能有效捕捉复杂声学模式和时序依赖，该方法在生态保护和监测方面具有实际应用潜力。

### 翻译

自动鸟类声音分类对推进生态监测和生物多样性研究至关重要。尽管最近取得了进展，但现有方法通常严重依赖标记数据，使用有限的特征表示，并忽略了准确物种识别所必需的时间动态。在这项工作中，我们提出了一个自监督对比网络ARIONet（帧间目标网络的声音表示），通过增强音频表示联合优化对比分类和未来帧预测。该模型在基于transformer的编码器模型中同时集成多种互补音频特征。我们的框架设计有两个关键目标：(1)通过最大化同一音频段增强视图之间的相似性同时推开不同样本来学习具有区分性的物种特定表示用于对比学习；(2)通过预测未来音频帧来建模时间动态，两者都不需要大规模标注。我们在四个多样化的鸟类声音数据集上验证了我们的框架，包括英国鸟类声音数据集、鸟类声音数据集和两个扩展的Xeno-Canto子集（A-M和N-Z）。我们的方法一致优于现有基线，分别实现了98.41%、93.07%、91.89%和91.58%的分类准确率，以及97.84%、94.10%、91.29%和90.94%的F1分数。此外，它在未来帧预测任务中显示出低的平均绝对误差和高余弦相似度，高达95%。大量实验进一步证实了我们的自监督学习策略在捕捉复杂声学模式和时序依赖方面的有效性，以及其在生态保护和监测中的实际应用潜力。


### 论文摘要

Automated birdsong classification is essential for advancing ecological monitoring and biodiversity studies. Despite recent progress, existing methods often depend heavily on labeled data, use limited feature representations, and overlook temporal dynamics essential for accurate species identification. In this work, we propose a self-supervised contrastive network, ARIONet (Acoustic Representation for Interframe Objective Network), that jointly optimizes contrastive classification and future frame prediction using augmented audio representations. The model simultaneously integrates multiple complementary audio features within a transformer-based encoder model. Our framework is designed with two key objectives: (1) to learn discriminative species-specific representations for contrastive learning through maximizing similarity between augmented views of the same audio segment while pushing apart different samples, and (2) to model temporal dynamics by predicting future audio frames, both without requiring large-scale annotations. We validate our framework on four diverse birdsong datasets, including the British Birdsong Dataset, Bird Song Dataset, and two extended Xeno-Canto subsets (A-M and N-Z). Our method consistently outperforms existing baselines and achieves classification accuracies of 98.41%, 93.07%, 91.89%, and 91.58%, and F1-scores of 97.84%, 94.10%, 91.29%, and 90.94%, respectively. Furthermore, it demonstrates low mean absolute errors and high cosine similarity, up to 95%, in future frame prediction tasks. Extensive experiments further confirm the effectiveness of our self-supervised learning strategy in capturing complex acoustic patterns and temporal dependencies, as well as its potential for real-world applicability in ecological conservation and monitoring.

---

## 48. Learning Domain-Robust Bioacoustic Representations for Mosquito Species Classification with Contrastive Learning and Distribution Alignment

**论文链接:** [http://arxiv.org/abs/2510.00346v1](http://arxiv.org/abs/2510.00346v1)

**作者:** Yuanbo Hou, Zhaoyi Liu, Xin Shen, Stephen Roberts

**发布时间:** 2025-09-30

### GPT解析

### 总结

该研究提出了一种域鲁棒生物声学学习(DR-BioL)框架，结合对比学习和分布对齐，解决了蚊子物种分类中域特征干扰问题，提高了跨域分类的准确性和鲁棒性。

### 背景

蚊子物种分类对媒介监测和疾病控制至关重要，但蚊子生物声学数据收集受活动季节和野外工作限制。不同区域、栖息地和实验室的蚊子录音包含非生物学变化的域特征，干扰分类准确性。

### 目的

开发一种域鲁棒生物声学学习方法，解决直接在含域特征的音频上训练的模型依赖域信息而非物种声学线索的问题，提高跨域泛化能力。

### 方法

提出域鲁棒生物声学学习(DR-BioL)框架，结合对比学习促进同一物种内部凝聚力和减轻域间差异，以及物种条件分布对齐增强跨域物种表示。

### 主要发现

在多域蚊子生物声学数据集上的实验表明，DR-BioL提高了基线模型的准确性和鲁棒性，展现了在现实世界中进行可靠跨域蚊子物种分类的潜力。

### 结论

DR-BioL框架能有效解决域特征干扰问题，提高跨域蚊子物种分类的准确性和鲁棒性，具有实际应用价值。

### 翻译

蚊子物种分类(MSC)对媒介监测和疾病控制至关重要。蚊子生物声学数据的收集通常受蚊子活动季节和野外工作的限制。来自不同区域、栖息地和实验室的蚊子录音通常包含记录环境带来的非生物学变化，我们称之为域特征。研究发现，直接在包含域特征的音频录音上训练的模型往往依赖域信息而非物种声学线索进行识别，导致看似良好的性能但实际上跨域泛化能力差。为此，我们提出了一种域鲁棒生物声学学习(DR-BioL)框架，结合对比学习和分布对齐。对比学习旨在促进同一物种内部的凝聚力并减轻域间差异，物种条件分布对齐进一步增强了跨域物种表示。在来自不同环境的多域蚊子生物声学数据集上的实验表明，DR-BioL提高了基线模型的准确性和鲁棒性，突显了其在现实世界中进行可靠的跨域蚊子物种分类的潜力。


### 论文摘要

Mosquito Species Classification (MSC) is crucial for vector surveillance and disease control. The collection of mosquito bioacoustic data is often limited by mosquito activity seasons and fieldwork. Mosquito recordings across regions, habitats, and laboratories often show non-biological variations from the recording environment, which we refer to as domain features. This study finds that models directly trained on audio recordings with domain features tend to rely on domain information rather than the species' acoustic cues for identification, resulting in illusory good performance while actually performing poor cross-domain generalization. To this end, we propose a Domain-Robust Bioacoustic Learning (DR-BioL) framework that combines contrastive learning with distribution alignment. Contrastive learning aims to promote cohesion within the same species and mitigate inter-domain discrepancies, and species-conditional distribution alignment further enhances cross-domain species representation. Experiments on a multi-domain mosquito bioacoustic dataset from diverse environments show that the DR-BioL improves the accuracy and robustness of baselines, highlighting its potential for reliable cross-domain MSC in the real world.

---

## 49. VL-KnG: Visual Scene Understanding for Navigation Goal Identification using Spatiotemporal Knowledge Graphs

**论文链接:** [http://arxiv.org/abs/2510.01483v1](http://arxiv.org/abs/2510.01483v1)

**作者:** Mohamad Al Mdfaa, Svetlana Lukina, Timur Akhtyamov, Arthur Nigmatzyanov, Dmitrii Nalberskii, Sergey Zagoruyko, Gonzalo Ferrer

**发布时间:** 2025-10-01

**备注:** This work has been submitted to the IEEE for possible publication

### GPT解析

### 总结

本文提出了VL-KnG，一个视觉场景理解系统，通过时空知识图谱构建和计算高效查询处理，解决了视觉语言模型在机器人导航中的基本局限性。

### 背景

视觉语言模型在机器人导航方面显示出潜力，但存在基本局限性：缺乏持久的场景记忆，空间推理能力有限，且随着视频持续时间增加无法有效扩展以实现实时应用。

### 目的

开发一个能够克服VLMs在机器人导航中局限性的系统，特别是解决场景记忆缺失、空间推理不足和实时应用扩展性问题。

### 方法

提出VL-KnG系统，将视频序列分块处理利用现代VLMs，创建持久知识图谱以保持对象身份随时间变化，通过可查询的图结构实现可解释的空间推理，并引入WalkieKnowledge基准测试包含8条不同轨迹的约200个手动标注问题。

### 主要发现

在差速驱动机器人上的实际部署展示了实际应用能力，方法实现了77.27%的成功率和76.92%的答案准确率，与Gemini 2.5 Pro性能相匹配，同时提供知识图谱支持的可解释推理和计算效率。

### 结论

VL-KnG系统有效解决了VLMs在机器人导航中的基本局限性，提供了可解释的推理能力和计算效率，适用于实时应用。代码和数据集将在接受后发布。

### 翻译

视觉语言模型(VLMs)在机器人导航方面显示出潜力，但遇到基本局限性：它们缺乏持久的场景记忆，空间推理能力有限，并且随着视频持续时间增加无法有效扩展以实现实时应用。我们提出了VL-KnG，一个视觉场景理解系统，它通过时空知识图谱构建和用于导航目标识别的计算高效查询处理来解决这些挑战。我们的方法利用现代VLMs处理视频序列块，创建保持对象身份随时间变化的持久知识图谱，并通过可查询的图结构实现可解释的空间推理。我们还引入了WalkieKnowledge，一个新的基准测试，包含约200个手动标注问题，跨越8条不同轨迹，约100分钟的视频数据，使结构化方法和通用VLMs之间的公平比较成为可能。在差速驱动机器人上的实际部署展示了实际应用能力，我们的方法实现了77.27%的成功率和76.92%的答案准确率，与Gemini 2.5 Pro性能相匹配，同时提供知识图谱支持的可解释推理，具有计算效率，可实现不同任务(如定位、导航和规划)的实时部署。代码和数据集将在接受后发布。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决机器人导航中视觉场景理解的挑战，具体是让机器人通过视觉语言模型更好地识别导航目标。这个问题很重要，因为机器人导航是机器人融入人类日常生活的关键能力，需要理解复杂的空间关系和时间对象动态，以实现自然语言引导的目标导向行为。当前视觉语言模型存在持久场景记忆缺乏、空间推理有限和无法随视频时长扩展以实现实时应用的三大基本限制。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者通过识别现有方法不足（顺序处理失去时间一致性或直接VLM推理缺乏结构化能力）提出关键见解：持久结构化表示（如知识图谱）能提供直接VLM推理无法比拟的优势，特别是在可解释性、计算效率和跨任务适应性方面。作者确实借鉴了现有工作，包括视觉语言导航研究、多模态3D映射方法（VLMaps、ConceptFusion）、场景图构建（ConceptGraphs）、3D图表示（Hydra、Clio）、基于图像的拓扑图、CLIP检索和检索增强生成(RAG)技术，并综合这些方法的最佳实践。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是构建时空知识图谱来表示和理解环境，通过结构化知识表示和高效查询处理支持导航目标识别。整体流程包括：1)时空知识图谱构建：将视频分块处理，用视觉语言模型提取对象描述符，迭代构建知识图谱；2)时空对象关联：使用基于语义的关联机制维持跨时间对象身份，利用大语言模型推理建立对象对应关系；3)导航查询处理：采用基于GraphRAG的方法进行子图检索和推理，包括查询分解、子图检索和推理定位三步；4)实际部署：与机器人导航系统集成，提供目标位置估计，支持实时应用。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)基于语义的对象关联机制，维持跨时间唯一对象身份；2)综合对象描述符系统，捕获颜色、材质、大小等丰富语义信息；3)时空知识图谱系统，实现持久场景表示和可查询空间推理；4)WalkieKnowledge评估基准，提供结构化方法和通用VLM的公平比较；5)实时部署能力，在实际机器人平台上验证实用性。相比之前工作，VL-KnG结合了结构化知识表示和VLM推理优势，提供可解释推理过程，在保持高性能同时实现实时计算效率，通过知识图谱实现跨时间对象一致性跟踪，并引入新评估基准。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'VL-KnG通过构建时空知识图谱和高效查询处理机制，解决了视觉语言模型在机器人导航中的持久场景记忆、空间推理和实时性挑战，实现了与最先进VLM相当的性能但提供了更好的可解释性和计算效率。'}


### 论文摘要

Vision-language models (VLMs) have shown potential for robot navigation but encounter fundamental limitations: they lack persistent scene memory, offer limited spatial reasoning, and do not scale effectively with video duration for real-time application. We present VL-KnG, a Visual Scene Understanding system that tackles these challenges using spatiotemporal knowledge graph construction and computationally efficient query processing for navigation goal identification. Our approach processes video sequences in chunks utilizing modern VLMs, creates persistent knowledge graphs that maintain object identity over time, and enables explainable spatial reasoning through queryable graph structures. We also introduce WalkieKnowledge, a new benchmark with about 200 manually annotated questions across 8 diverse trajectories spanning approximately 100 minutes of video data, enabling fair comparison between structured approaches and general-purpose VLMs. Real-world deployment on a differential drive robot demonstrates practical applicability, with our method achieving 77.27% success rate and 76.92% answer accuracy, matching Gemini 2.5 Pro performance while providing explainable reasoning supported by the knowledge graph, computational efficiency for real-time deployment across different tasks, such as localization, navigation and planning. Code and dataset will be released after acceptance.

---

## 50. KeySG: Hierarchical Keyframe-Based 3D Scene Graphs

**论文链接:** [http://arxiv.org/abs/2510.01049v1](http://arxiv.org/abs/2510.01049v1)

**作者:** Abdelrhman Werby, Dennis Rotondi, Fabio Scaparro, Kai O. Arras

**发布时间:** 2025-10-01

### GPT解析

### 总结

KeySG是一种新的框架，通过层次化图表示和多模态信息增强，解决了现有3D场景图构建方法的语义局限性和上下文窗口问题，在多个基准测试上表现优异。

### 背景

3D场景图作为强大的世界表示方法，结合大型语言模型可让机器人在复杂环境中推理和导航，但当前方法语义局限于预定义关系，且大环境序列化易超出LLM上下文窗口。

### 目的

开发一种更通用的、与任务无关的3D场景表示方法，克服现有方法的语义局限性和扩展性问题，使系统能处理复杂模糊查询。

### 方法

KeySG将3D场景表示为包含楼层、房间、物体和功能元素的层次化图，节点通过优化选择的关键帧提取多模态信息增强，利用VLM提取场景信息避免显式建模物体关系，采用层次化RAG管道处理大规模场景图。

### 主要发现

KeySG在四个基准测试(包括3D物体分割和复杂查询检索)上优于先前方法，展示了卓越的语义丰富性和效率。

### 结论

KeySG通过创新的设计有效解决了现有3D场景图方法的局限性，能够处理复杂模糊查询并缓解大规模场景图的扩展性问题，为机器人在复杂环境中的推理和导航提供了新的解决方案。

### 翻译

近年来，3D场景图已成为一种强大的世界表示方法，既提供几何准确性又具备语义丰富性。将3D场景图与大型语言模型结合，使机器人在以人为中心的环境中能够推理、规划和导航。然而，当前构建3D场景图的方法在语义上仅限于预定义的关系集合，且在大环境中的序列化容易超出LLM的上下文窗口。我们引入了KeySG框架，将3D场景表示为由楼层、房间、物体和功能元素组成的层次化图，其中节点通过从关键帧中提取的多模态信息进行增强，这些关键帧经过选择以优化几何和视觉覆盖率。关键帧使我们能够有效利用VLM提取场景信息，避免了对物体间关系边进行显式建模的需求，从而实现更通用、与任务无关的推理和规划。我们的方法可以处理复杂和模糊的查询，同时通过利用层次化检索增强生成(RAG)管道从图中提取相关上下文，缓解与大规模场景图相关的扩展性问题。在四个不同的基准测试上(包括3D物体分割和复杂查询检索)评估，KeySG在大多数指标上优于先前的方法，展示了其卓越的语义丰富性和效率。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决当前3D场景图方法的两个关键问题：一是语义局限性，即只能支持预定义的关系集合，限制了任务多样性；二是可扩展性问题，即大型环境场景图可能超过大型语言模型的上下文窗口限制。这些问题在机器人领域尤为重要，因为要让机器人在复杂人类环境中有效工作，需要一种既能提供精确几何细节又支持高级推理的世界表示方法。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先识别出3D场景图的局限性，然后设计了一个分层结构来捕捉环境的多层次抽象。他们借鉴了关键帧视觉SLAM的思想，但将其应用于视觉覆盖而非几何重建；使用了现有的场景分割算法和视觉语言模型；并引入了分层检索增强生成(RAG)管道来解决可扩展性问题。这些设计共同构成了KeySG框架，它通过关键帧采样和多模态信息增强来构建一个更灵活、可扩展的场景表示。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是通过关键帧采样和多模态信息增强构建分层3D场景图，不显式建模物体关系边，而是将丰富场景信息存储在关键帧及其描述中。整体流程包括：1)分层场景分割(重建点云并分割为楼层和房间)；2)关键帧采样(选择能提供全面视觉覆盖的代表帧)；3)物体和功能元素分割(利用VLM提取标签并进行3D分割)；4)场景描述生成(用LLM创建房间和楼层摘要)；5)分层RAG查询(处理用户查询并返回相关上下文)。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)五层分层的3D场景图结构(建筑、楼层、房间、物体和功能元素)；2)关键帧驱动的多模态信息增强；3)分层场景描述生成；4)分层检索增强生成(RAG)管道。相比之前的工作，KeySG不显式建模预定义关系边，而是通过关键帧隐式捕获关系；解决了大型环境中的可扩展性问题；支持更通用、任务无关的推理；能处理更复杂的查询；提供了更丰富的语义信息。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'KeySG提出了一种基于关键帧的分层3D场景图框架，通过多模态信息增强和分层检索增强生成技术，解决了现有3D场景图在语义表达和可扩展性方面的局限性，实现了更通用、高效的机器人环境表示和推理能力。'}


### 论文摘要

In recent years, 3D scene graphs have emerged as a powerful world representation, offering both geometric accuracy and semantic richness. Combining 3D scene graphs with large language models enables robots to reason, plan, and navigate in complex human-centered environments. However, current approaches for constructing 3D scene graphs are semantically limited to a predefined set of relationships, and their serialization in large environments can easily exceed an LLM's context window. We introduce KeySG, a framework that represents 3D scenes as a hierarchical graph consisting of floors, rooms, objects, and functional elements, where nodes are augmented with multi-modal information extracted from keyframes selected to optimize geometric and visual coverage. The keyframes allow us to efficiently leverage VLM to extract scene information, alleviating the need to explicitly model relationship edges between objects, enabling more general, task-agnostic reasoning and planning. Our approach can process complex and ambiguous queries while mitigating the scalability issues associated with large scene graphs by utilizing a hierarchical retrieval-augmented generation (RAG) pipeline to extract relevant context from the graph. Evaluated across four distinct benchmarks -- including 3D object segmentation and complex query retrieval -- KeySG outperforms prior approaches on most metrics, demonstrating its superior semantic richness and efficiency.

---

## 51. L4P: Towards Unified Low-Level 4D Vision Perception

**论文链接:** [http://arxiv.org/abs/2502.13078v3](http://arxiv.org/abs/2502.13078v3)

**作者:** Abhishek Badki, Hang Su, Bowen Wen, Orazio Gallo

**发布时间:** 2025-02-18

### GPT解析

### 总结

L4P是一种通用的前馈架构，通过预训练的ViT视频编码器和轻量级任务头部，在统一框架下高效解决多种低级4D感知任务，性能与专业化方法相当，且可同时处理所有任务。

### 背景

视频像素之间的时空关系对低级4D感知任务至关重要。目前大多数最先进的方法依赖于针对特定任务的专业化架构。

### 目的

提出一种通用的前馈架构L4P，能够在统一框架下解决低级4D感知任务。

### 方法

L4P利用预训练的基于ViT的视频编码器，结合轻量级的特定任务头部，因此不需要大量训练。

### 主要发现

尽管L4P是通用和前馈的，但它在密集任务（如深度或光流估计）和稀疏任务（如2D/3D跟踪）上都能与现有的专业化方法相媲美。

### 结论

L4P能够一次性解决所有任务，所需时间与单任务方法相当。

### 翻译

视频像素之间的时空关系对低级4D感知任务携带关键信息。能够推理这种关系的单一模型应该能够很好地解决多种此类任务。然而，大多数最先进的方法依赖于针对特定任务的专业化架构。我们提出了L4P，一种前馈、通用架构，在统一框架下解决低级4D感知任务。L4P利用预训练的基于ViT的视频编码器，并结合轻量级的特定任务头部，因此不需要大量训练。尽管其通用和前馈的表述，我们的方法在密集任务（如深度或光流估计）和稀疏任务（如2D/3D跟踪）上都能与现有专业化方法相媲美。此外，它一次性解决了所有任务，所需时间与单任务方法相当。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决低层次4D视觉感知任务缺乏统一框架的问题。现有方法通常需要针对每个任务（如深度估计、光流估计、跟踪等）设计专门的架构，无法有效利用视频数据中的时空关系信息。这个问题很重要，因为视频像素间的时空关系对理解动态场景至关重要，统一的框架能更高效地处理多种任务，减少重复计算，并增强模型在不同任务间的知识迁移能力。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先认识到预训练视频模型（如VideoMAE）能捕捉丰富的时空特征，但其在低层次4D感知任务中的应用未被充分探索。他们设计了一个通用架构，结合预训练视频编码器与轻量级任务特定头：对于密集任务（如深度估计），扩展了DPT架构；对于稀疏任务（如跟踪），借鉴了SAM的提示机制并添加记忆机制。作者采用了多阶段训练策略，先在单窗口上端到端训练，再通过冻结部分参数和展开窗口训练优化跟踪性能。该方法明显借鉴了VideoMAE、DPT和CoTracker等现有工作，但进行了创新性整合。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是使用预训练的视频编码器作为通用特征提取器，配合轻量级任务特定头，实现统一的低层次4D视觉感知。整体流程为：1) 视频输入经VideoMAE编码器生成时空特征；2) 密集任务（如深度估计）通过扩展的DPT头部处理，将视频标记映射到3D特征图并引入时间推理；3) 稀疏任务（如跟踪）通过特殊头部处理，使用双向注意力机制和记忆机制实现长时间跟踪；4) 采用多阶段训练策略，先端到端训练再优化特定任务；5) 推理时使用滑动窗口处理长视频，确保重叠区域一致性。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1) 统一框架同时处理密集和稀疏任务，打破传统方法需专门架构的限制；2) 轻量级任务特定头设计，只需少量训练参数即可扩展新任务；3) 记忆机制解决视频编码器有限上下文问题，支持长时间跟踪；4) 多阶段训练策略平衡不同任务性能；5) 强大的跨数据集泛化能力。相比之前工作，L4P无需针对每个任务重新训练整个模型，计算效率更高（处理16帧视频约300ms），且能同时输出多种任务结果，而传统方法通常只能处理单一任务。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'L4P提出了一种统一的低层次4D视觉感知框架，通过结合预训练视频编码器和轻量级任务特定头，实现了在密集和稀疏任务上的高性能，同时支持长时间跟踪和新任务的轻松扩展。'}


### 论文摘要

The spatio-temporal relationship between the pixels of a video carries critical information for low-level 4D perception tasks. A single model that reasons about it should be able to solve several such tasks well. Yet, most state-of-the-art methods rely on architectures specialized for the task at hand. We present L4P, a feedforward, general-purpose architecture that solves low-level 4D perception tasks in a unified framework. L4P leverages a pre-trained ViT-based video encoder and combines it with per-task heads that are lightweight and therefore do not require extensive training. Despite its general and feedforward formulation, our method is competitive with existing specialized methods on both dense tasks, such as depth or optical flow estimation, and sparse tasks, such as 2D/3D tracking. Moreover, it solves all tasks at once in a time comparable to that of single-task methods.

---

## 52. Moon: A Modality Conversion-based Efficient Multivariate Time Series Anomaly Detection

**论文链接:** [http://arxiv.org/abs/2510.01970v1](http://arxiv.org/abs/2510.01970v1)

**作者:** Yuanyuan Yao, Yuhan Shi, Lu Chen, Ziquan Fang, Yunjun Gao, Leong Hou U, Yushuai Li, Tianyi Li

**发布时间:** 2025-10-02

### GPT解析

### 总结

Moon是一种监督模态转换的多元时间序列异常检测框架，通过多变量马尔可夫转移场技术将时间序列数据转换为图像表示，结合多模态CNN融合数值和图像数据，并使用SHAP解释器提高异常检测的可解释性。

### 背景

多元时间序列异常检测旨在识别每个时间戳包含多个变量的异常模式。现有方法分为基于重建、基于预测和基于分类器三类，但面临无监督方法依赖误差阈值导致不准确、半监督方法未充分利用异常标签、监督方法无法捕获局部关系且计算成本高等挑战。

### 目的

解决现有多元时间序列异常检测方法的局限性，提高异常检测的效率和准确性，并提供详细的异常分析报告。

### 方法

Moon框架包含三个关键组件：1)多变量马尔可夫转移场技术将数值时间序列转换为图像表示；2)多模态CNN通过参数共享的特征融合模型整合数值和图像数据；3)基于SHAP的异常解释器识别导致异常的关键变量。

### 主要发现

在六个真实世界多元时间序列数据集上的实验表明，Moon在效率上比六种最先进的方法高93%，在准确性上高4%，在解释性能上高10.8%。

### 结论

Moon通过模态转换和多模态融合有效提高了异常检测的效率和准确性，同时提供详细的异常分析报告，解决了现有方法的关键挑战。

### 翻译

多元时间序列异常检测识别每个时间戳包含多个变量的异常模式。现有的多元时间序列异常检测方法分为三类：基于重建的方法、基于预测的方法和基于分类器的方法。然而，这些方法面临两个关键挑战：(1)无监督学习方法，如基于重建和预测的方法，依赖误差阈值，可能导致不准确；(2)半监督方法主要建模正常数据且常常未充分利用异常标签，限制了细微异常的检测；(3)监督学习方法，如基于分类器的方法，往往无法捕获局部关系，计算成本高，且受限于标记数据的稀缺性。为解决这些局限性，我们提出了Moon，一种监督模态转换的多元时间序列异常检测框架。Moon提高了异常检测的效率和准确性，同时提供详细的异常分析报告。首先，Moon引入了一种新颖的多变量马尔可夫转移场技术，将数值时间序列数据转换为图像表示，捕获变量和时间戳之间的关系。由于数值数据保留了无法仅通过图像转换完全捕获的独特模式，Moon采用多模态CNN通过参数共享的特征融合模型整合数值和图像数据，提高训练效率。最后，基于SHAP的异常解释器识别导致异常的关键变量，提高可解释性。在六个真实世界多元时间序列数据集上的大量实验表明，Moon在效率上比六种最先进的方法高93%，在准确性上高4%，在解释性能上高10.8%。


### 论文摘要

Multivariate time series (MTS) anomaly detection identifies abnormal patterns where each timestamp contains multiple variables. Existing MTS anomaly detection methods fall into three categories: reconstruction-based, prediction-based, and classifier-based methods. However, these methods face two key challenges: (1) Unsupervised learning methods, such as reconstruction-based and prediction-based methods, rely on error thresholds, which can lead to inaccuracies; (2) Semi-supervised methods mainly model normal data and often underuse anomaly labels, limiting detection of subtle anomalies;(3) Supervised learning methods, such as classifier-based approaches, often fail to capture local relationships, incur high computational costs, and are constrained by the scarcity of labeled data. To address these limitations, we propose Moon, a supervised modality conversion-based multivariate time series anomaly detection framework. Moon enhances the efficiency and accuracy of anomaly detection while providing detailed anomaly analysis reports. First, Moon introduces a novel multivariate Markov Transition Field (MV-MTF) technique to convert numeric time series data into image representations, capturing relationships across variables and timestamps. Since numeric data retains unique patterns that cannot be fully captured by image conversion alone, Moon employs a Multimodal-CNN to integrate numeric and image data through a feature fusion model with parameter sharing, enhancing training efficiency. Finally, a SHAP-based anomaly explainer identifies key variables contributing to anomalies, improving interpretability. Extensive experiments on six real-world MTS datasets demonstrate that Moon outperforms six state-of-the-art methods by up to 93% in efficiency, 4% in accuracy and, 10.8% in interpretation performance.

---

## 53. Automatic Speech Recognition (ASR) for African Low-Resource Languages: A Systematic Literature Review

**论文链接:** [http://arxiv.org/abs/2510.01145v1](http://arxiv.org/abs/2510.01145v1)

**作者:** Sukairaj Hafiz Imam, Tadesse Destaw Belay, Kedir Yassin Husse, Ibrahim Said Ahmad, Idris Abdulmumin, Hadiza Ali Umar, Muhammad Yahuza Bello, Joyce Nakatumba-Nabende, Seid Muhie Yimam, Shamsuddeen Hassan Muhammad

**发布时间:** 2025-10-01

### GPT解析

### 总结

这篇系统文献综述探讨了非洲低资源语言的自动语音识别研究现状，分析了数据集、模型、训练方法、评估技术和面临的挑战，并提出了未来发展方向。

### 背景

自动语音识别技术在全球取得了显著进展，但非洲低资源语言仍然严重缺乏代表性，阻碍了非洲大陆的数字化包容性，非洲大陆拥有超过2000种语言。

### 目的

探索非洲语言自动语音识别的研究现状，重点关注数据集、模型和训练方法、评估技术、挑战，并提出未来研究方向。

### 方法

采用PRISMA 2020程序，在多个学术数据库中检索2020年1月至2025年7月发表的研究，筛选出71篇高质量论文，记录了74个数据集，涵盖111种语言和约11,206小时的语音数据。

### 主要发现

1) 少于15%的研究提供了可复现材料；2) 数据集许可不明确；3) 自监督和迁移学习技术有前景但受限于预训练数据和方言覆盖；4) 研究主要使用词错误率，很少考虑声调和形态丰富的语言特性；5) ASR系统研究证据不一致，受到数据集质量和基准测试限制。

### 结论

社区驱动倡议和方法进步为改进提供了途径，未来可持续发展需要利益相关方合作、创建伦理平衡的数据集、使用轻量级建模技术和加强基准测试。

### 翻译

ASR已取得显著的全球进展，然而非洲低资源语言仍然严重缺乏代表性，阻碍了非洲大陆的数字化包容，非洲大陆有超过2000种语言。这篇系统文献综述探索了非洲语言ASR的研究，重点关注数据集、模型和训练方法、评估技术、挑战，并推荐未来方向。我们采用PRISMA 2020程序，在DBLP、ACM Digital Library、Google Scholar、Semantic Scholar和arXiv中搜索2020年1月至2025年7月发表的研究。我们纳入与非洲语言ASR数据集、模型或指标相关的研究，同时排除非非洲研究、重复研究和低质量研究(评分<3/5)。我们从2,062条记录中筛选出71条，记录了涵盖111种语言的74个数据集，包含约11,206小时的语音。少于15%的研究提供了可复现材料，数据集许可不明确。自监督和迁移学习技术有前景，但受到预训练数据有限、方言覆盖不足和资源可用性的阻碍。大多数研究人员使用词错误率(WER)，很少使用语言学评分如字符错误率(CER)或变音符号错误率(DER)，因此在声调和形态丰富的语言中应用有限。关于ASR系统的现有证据不一致，受到数据集可用性、注释质量、许可不确定性和有限基准测试等问题的影响。尽管如此，社区驱动倡议和方法进步的兴起表明了改进的途径。该领域的可持续发展还将包括利益相关方合作、创建伦理平衡的数据集、使用轻量级建模技术和积极的基准测试。


### 论文摘要

ASR has achieved remarkable global progress, yet African low-resource languages remain rigorously underrepresented, producing barriers to digital inclusion across the continent with more than +2000 languages. This systematic literature review (SLR) explores research on ASR for African languages with a focus on datasets, models and training methods, evaluation techniques, challenges, and recommends future directions. We employ the PRISMA 2020 procedures and search DBLP, ACM Digital Library, Google Scholar, Semantic Scholar, and arXiv for studies published between January 2020 and July 2025. We include studies related to ASR datasets, models or metrics for African languages, while excluding non-African, duplicates, and low-quality studies (score <3/5). We screen 71 out of 2,062 records and we record a total of 74 datasets across 111 languages, encompassing approximately 11,206 hours of speech. Fewer than 15% of research provided reproducible materials, and dataset licensing is not clear. Self-supervised and transfer learning techniques are promising, but are hindered by limited pre-training data, inadequate coverage of dialects, and the availability of resources. Most of the researchers use Word Error Rate (WER), with very minimal use of linguistically informed scores such as Character Error Rate (CER) or Diacritic Error Rate (DER), and thus with limited application in tonal and morphologically rich languages. The existing evidence on ASR systems is inconsistent, hindered by issues like dataset availability, poor annotations, licensing uncertainties, and limited benchmarking. Nevertheless, the rise of community-driven initiatives and methodological advancements indicates a pathway for improvement. Sustainable development for this area will also include stakeholder partnership, creation of ethically well-balanced datasets, use of lightweight modelling techniques, and active benchmarking.

---

## 54. Bridging the Gap Between Simulated and Real Network Data Using Transfer Learning

**论文链接:** [http://arxiv.org/abs/2510.00956v1](http://arxiv.org/abs/2510.00956v1)

**作者:** Carlos Güemes-Palau, Miquel Ferriol-Galmés, Jordi Paillisse-Vilanova, Albert López-Brescó, Pere Barlet-Ros, Albert Cabellos-Aparicio

**发布时间:** 2025-10-01

**备注:** This paper was submitted to IEEE ICC 2026. 7 Pages, 5 Figures

### GPT解析

### 总结

本研究提出了一种结合迁移学习的混合方法，通过微调预训练模型将模拟数据与少量真实数据结合，显著提高了网络行为预测的准确性。

### 背景

基于机器学习的网络模型能够快速准确地预测复杂网络行为，但需要大量训练数据。从真实网络收集这些数据通常成本高昂且有限，特别是在故障等关键场景中。

### 目的

解决模型在真实环境中部署时因依赖模拟数据而导致的准确性降低问题。

### 方法

提出一种利用迁移学习结合模拟和真实数据的混合方法，使用RouteNet-Fermi模型，通过少量真实数据微调预训练模型来提高性能。

### 主要发现

使用OMNeT++和自定义测试台的实验将数据包延迟预测的平均绝对百分比误差降低了高达88%。仅使用10个真实场景，MAPE下降37%；使用50个场景，MAPE下降48%。

### 结论

通过迁移学习结合少量真实数据与模拟数据，可以显著提高网络行为预测模型在真实环境中的准确性。

### 翻译

基于机器学习的网络模型能够为复杂的网络行为提供快速而准确的预测，但需要大量的训练数据。从真实网络收集此类数据通常成本高昂且有限，特别是在故障等关键场景中。因此，研究人员通常依赖模拟数据，但这会导致模型在真实环境中部署时准确性降低。我们提出了一种利用迁移学习结合模拟和真实数据的混合方法。使用RouteNet-Fermi，我们展示了用少量真实数据微调预训练模型能显著提高性能。我们在OMNeT++和自定义测试台上的实验将数据包延迟预测的平均绝对百分比误差降低了高达88%。仅使用10个真实场景，MAPE就下降了37%，使用50个场景则下降了48%。


### 论文摘要

Machine Learning (ML)-based network models provide fast and accurate predictions for complex network behaviors but require substantial training data. Collecting such data from real networks is often costly and limited, especially for critical scenarios like failures. As a result, researchers commonly rely on simulated data, which reduces accuracy when models are deployed in real environments. We propose a hybrid approach leveraging transfer learning to combine simulated and real-world data. Using RouteNet-Fermi, we show that fine-tuning a pre-trained model with a small real dataset significantly improves performance. Our experiments with OMNeT++ and a custom testbed reduce the Mean Absolute Percentage Error (MAPE) in packet delay prediction by up to 88%. With just 10 real scenarios, MAPE drops by 37%, and with 50 scenarios, by 48%.

---

## 55. Intuitions of Machine Learning Researchers about Transfer Learning for Medical Image Classification

**论文链接:** [http://arxiv.org/abs/2510.00902v1](http://arxiv.org/abs/2510.00902v1)

**作者:** Yucheng Lu, Hubert Dariusz Zając, Veronika Cheplygina, Amelia Jiménez-Sánchez

**发布时间:** 2025-10-01

**备注:** Under review

### GPT解析

### 总结

本研究探讨了医学影像迁移学习中源数据集选择的决策过程，发现选择是任务依赖性的，受社区实践、数据集属性和相似性影响，且相似性与预期性能并不总是对齐。

### 背景

迁移学习对医学影像至关重要，但源数据集的选择通常基于研究者直觉而非系统原则，这会影响算法的可推广性和患者结果。

### 目的

从人类中心的人机交互(HCI)角度了解机器学习从业者如何选择源数据集，为更系统的源选择提供实用见解。

### 方法

对机器学习从业者进行基于任务的调查，采用人类中心的HCI视角研究源数据集选择决策，而非传统的模型和实验设置基准测试方法。

### 主要发现

1)选择是任务依赖性的；2)受社区实践、数据集属性和计算或感知相似性影响；3)相似性评级与预期性能并不总是对齐，挑战了'越相似越好'的传统观点；4)参与者常使用模糊术语，表明需要更清晰的定义和HCI工具。

### 结论

通过阐明这些启发式方法，该工作为迁移学习中更系统的源数据集选择提供了实用见解，有助于改善算法的可推广性和患者结果。

### 翻译

迁移学习对医学影像至关重要，然而源数据集的选择——这可能影响算法的可推广性，进而影响患者结果——通常依赖于研究者的直觉而非系统原则。本研究通过一项针对机器学习从业者的基于任务的调查来研究这些决策。与之前对模型和实验设置进行基准测试的工作不同，我们采用人类中心的HCI视角来了解从业者如何选择源数据集。我们的发现表明，选择是任务依赖性的，并受到社区实践、数据集属性以及计算(数据嵌入)或感知视觉或语义相似性的影响。然而，相似性评级和预期性能并不总是对齐，挑战了传统的'越相似越好'的观点。参与者经常使用模糊的术语，这表明需要更清晰的定义和HCI工具来使其明确和可用。通过阐明这些启发式方法，这项工作为迁移学习中更系统的源选择提供了实用见解。


### 论文摘要

Transfer learning is crucial for medical imaging, yet the selection of source datasets - which can impact the generalizability of algorithms, and thus patient outcomes - often relies on researchers' intuition rather than systematic principles. This study investigates these decisions through a task-based survey with machine learning practitioners. Unlike prior work that benchmarks models and experimental setups, we take a human-centered HCI perspective on how practitioners select source datasets. Our findings indicate that choices are task-dependent and influenced by community practices, dataset properties, and computational (data embedding), or perceived visual or semantic similarity. However, similarity ratings and expected performance are not always aligned, challenging a traditional "more similar is better" view. Participants often used ambiguous terminology, which suggests a need for clearer definitions and HCI tools to make them explicit and usable. By clarifying these heuristics, this work provides practical insights for more systematic source selection in transfer learning.

---

## 56. LVLMs as inspectors: an agentic framework for category-level structural defect annotation

**论文链接:** [http://arxiv.org/abs/2510.00603v1](http://arxiv.org/abs/2510.00603v1)

**作者:** Sheng Jiang, Yuanmin Ning, Bingxi Huang, Peiyin Chen, Zhaohui Chen

**发布时间:** 2025-10-01

### GPT解析

### 总结

本文介绍了一种名为ADPT的新型自动结构缺陷标注框架，该框架结合大视觉语言模型、语义模式匹配和迭代自提问完善机制，能够在无需人工监督的情况下将原始视觉数据转换为高质量语义标注的缺陷数据集。

### 背景

自动化结构缺陷标注对于确保基础设施安全至关重要，同时可以最大限度地减少人工标注的高成本和低效率问题。

### 目的

开发一种无需人工监督的高效、准确的结构缺陷自动标注方法，构建高质量数据集，支持结构损伤评估中的迁移学习和领域适应等下游任务。

### 方法

提出了ADPT框架，整合了大视觉语言模型(LVLMs)、语义模式匹配模块和迭代自提问完善机制，通过优化的领域特定提示和递归验证流程实现自动标注。

### 主要发现

实验结果表明，ADPT在区分缺陷和非缺陷图像时准确率高达98%，在四类缺陷的平衡类别设置下标注准确率为85%-98%，在不平衡类别数据集上准确率为80%-92%。

### 结论

ADPT框架为高保真数据集构建提供了可扩展且经济高效的解决方案，为结构损伤评估等下游任务提供了强有力的支持。

### 翻译

自动化结构缺陷标注对于确保基础设施安全至关重要，同时可以最大限度地减少人工标注的高成本和低效率问题。本文介绍了一种新颖的代理标注框架——基于代理的缺陷模式标记器(ADPT)，该框架整合了大视觉语言模型、语义模式匹配模块和迭代自提问完善机制。通过利用优化的领域特定提示和递归验证流程，ADPT能够在没有任何人工监督的情况下将原始视觉数据转换为高质量的语义标注缺陷数据集。实验结果表明，ADPT在区分缺陷图像和非缺陷图像时准确率高达98%，在四类缺陷的平衡类别设置下标注准确率为85%-98%，在不平衡类别数据集上准确率为80%-92%。该框架为高保真数据集构建提供了可扩展且经济高效的解决方案，为迁移学习和领域适应等下游任务提供了强有力的支持，特别是在结构损伤评估领域。


### 论文摘要

Automated structural defect annotation is essential for ensuring infrastructure safety while minimizing the high costs and inefficiencies of manual labeling. A novel agentic annotation framework, Agent-based Defect Pattern Tagger (ADPT), is introduced that integrates Large Vision-Language Models (LVLMs) with a semantic pattern matching module and an iterative self-questioning refinement mechanism. By leveraging optimized domain-specific prompting and a recursive verification process, ADPT transforms raw visual data into high-quality, semantically labeled defect datasets without any manual supervision. Experimental results demonstrate that ADPT achieves up to 98% accuracy in distinguishing defective from non-defective images, and 85%-98% annotation accuracy across four defect categories under class-balanced settings, with 80%-92% accuracy on class-imbalanced datasets. The framework offers a scalable and cost-effective solution for high-fidelity dataset construction, providing strong support for downstream tasks such as transfer learning and domain adaptation in structural damage assessment.

---

## 57. Deep Learning Approaches with Explainable AI for Differentiating Alzheimer Disease and Mild Cognitive Impairment

**论文链接:** [http://arxiv.org/abs/2510.00048v1](http://arxiv.org/abs/2510.00048v1)

**作者:** Fahad Mostafa, Kannon Hossain, Hafiz Khan

**发布时间:** 2025-09-27

**备注:** 18 pages, 4 figures

### GPT解析

### 总结

这项研究开发了一种混合深度学习集成框架，使用结构磁共振成像数据来区分阿尔茨海默病、轻度认知障碍和正常对照组，通过结合多种预训练CNN和集成学习策略，实现了高准确率，并通过可解释AI技术提高了诊断的可解释性。

### 背景

早期准确诊断阿尔茨海默病对有效临床干预至关重要，需要将阿尔茨海默病与轻度认知障碍区分开，后者是一种以细微结构变化为特征的早期阶段。

### 目的

提出一种混合深度学习集成框架，用于阿尔茨海默病分类，使用结构磁共振成像作为数据源。

### 方法

使用灰质和白质切片作为输入，采用三种预训练的卷积神经网络（ResNet50、NASNet和MobileNet），通过端到端过程对每个模型进行微调，并采用堆叠集成学习策略，包含元学习器和加权平均，以最优方式组合基础模型。

### 主要发现

在阿尔茨海默病神经影像计划数据集上评估，所提出的方法实现了最先进的准确率：阿尔茨海默病与轻度认知障碍之间为99.21%，轻度认知障碍与正常对照组之间为91.0%，优于传统的迁移学习和基线集成方法；集成了可解释人工智能技术，通过梯度加权类激活生成热图和归因图，突出灰质和白质切片中的关键区域，揭示了影响模型决策的结构生物标志物。

### 结论

结果突显了该框架在神经退行性疾病诊断中具有稳健和可扩展的临床决策支持潜力。

### 翻译

阿尔茨海默病的早期准确诊断对有效临床干预至关重要，特别是在将其与轻度认知障碍区分方面，后者是一种以细微结构变化为特征的早期阶段。在本研究中，我们提出了一种混合深度学习集成框架，使用结构磁共振成像进行阿尔茨海默病分类。灰质和白质切片被用作三种预训练的卷积神经网络的输入，如ResNet50、NASNet和MobileNet，每个网络都通过端到端过程进行微调。为进一步提高性能，我们采用堆叠集成学习策略，结合元学习器和加权平均，以最优方式组合基础模型。在阿尔茨海默病神经影像计划数据集上评估，所提出的方法实现了最先进的准确率：阿尔茨海默病与轻度认知障碍之间为99.21%，轻度认知障碍与正常对照组之间为91.0%，优于传统的迁移学习和基线集成方法。为了提高基于图像诊断的可解释性，我们通过梯度加权类激活集成了可解释人工智能技术，生成热图和归因图，突出灰质和白质切片中的关键区域，揭示了影响模型决策的结构生物标志物。这些结果突显了该框架在神经退行性疾病诊断中具有稳健和可扩展的临床决策支持潜力。


### 论文摘要

Early and accurate diagnosis of Alzheimer Disease is critical for effective clinical intervention, particularly in distinguishing it from Mild Cognitive Impairment, a prodromal stage marked by subtle structural changes. In this study, we propose a hybrid deep learning ensemble framework for Alzheimer Disease classification using structural magnetic resonance imaging. Gray and white matter slices are used as inputs to three pretrained convolutional neural networks such as ResNet50, NASNet, and MobileNet, each fine tuned through an end to end process. To further enhance performance, we incorporate a stacked ensemble learning strategy with a meta learner and weighted averaging to optimally combine the base models. Evaluated on the Alzheimer Disease Neuroimaging Initiative dataset, the proposed method achieves state of the art accuracy of 99.21% for Alzheimer Disease vs. Mild Cognitive Impairment and 91.0% for Mild Cognitive Impairment vs. Normal Controls, outperforming conventional transfer learning and baseline ensemble methods. To improve interpretability in image based diagnostics, we integrate Explainable AI techniques by Gradient weighted Class Activation, which generates heatmaps and attribution maps that highlight critical regions in gray and white matter slices, revealing structural biomarkers that influence model decisions. These results highlight the frameworks potential for robust and scalable clinical decision support in neurodegenerative disease diagnostics.

---

## 58. VideoNSA: Native Sparse Attention Scales Video Understanding

**论文链接:** [http://arxiv.org/abs/2510.02295v1](http://arxiv.org/abs/2510.02295v1)

**作者:** Enxin Song, Wenhao Chai, Shusheng Yang, Ethan Armand, Xiaojun Shan, Haiyang Xu, Jianwen Xie, Zhuowen Tu

**发布时间:** 2025-10-02

**备注:** Project Page: https://enxinsong.com/VideoNSA-web/, Code:  https://github.com/Espere-1119-Song/VideoNSA

### GPT解析

### 总结

VideoNSA是一种针对视频语言模型优化的方法，通过本地稀疏注意力技术解决了视频理解中的上下文长度限制问题，在长视频理解、时间推理和空间分析任务中表现出色。

### 背景

多模态语言模型中的视频理解受到上下文长度的限制，模型经常错过关键转换帧，难以在长时间尺度上保持连贯性。

### 目的

解决多模态语言模型中视频理解的上下文长度限制问题，提高模型在长时间视频上的理解能力。

### 方法

提出VideoNSA方法，通过在包含21.6万个视频指令的数据集上进行端到端训练，将本地稀疏注意力（NSA）适配到视频语言模型Qwen2.5-VL中。采用硬件感知的混合注意力方法，为文本保留密集注意力，同时为视频使用NSA。

### 主要发现

与基于token压缩和无需训练的稀疏基线方法相比，VideoNSA在长视频理解、时间推理和空间基准测试中实现了改进的性能。消融分析揭示了四个关键发现：(1)可靠扩展到12.8万个token；(2)在固定预算下的最优全局-局部注意力分配；(3)任务相关的分支使用模式；(4)可学习的组合稀疏注意力有助于诱导动态注意力汇。

### 结论

VideoNSA通过适配本地稀疏注意力到视频语言模型中，有效解决了多模态语言模型中视频理解的上下文长度限制问题，并在多个基准测试中表现出色。

### 翻译

多模态语言模型中的视频理解受到上下文长度的限制：模型经常错过关键转换帧，难以在长时间尺度上保持连贯性。为解决这一问题，我们将本地稀疏注意力（NSA）适配到视频语言模型中。我们的方法VideoNSA通过在21.6万个视频指令数据集上进行端到端训练来适配Qwen2.5-VL。我们采用了一种硬件感知的混合注意力方法，为文本保留密集注意力，同时为视频使用NSA。与基于token压缩和无需训练的稀疏基线方法相比，VideoNSA在长视频理解、时间推理和空间基准测试中实现了改进的性能。进一步的消融分析揭示了四个关键发现：(1)可靠扩展到12.8万个token；(2)在固定预算下的最优全局-局部注意力分配；(3)任务相关的分支使用模式；(4)可学习的组合稀疏注意力有助于诱导动态注意力汇。


### 论文摘要

Video understanding in multimodal language models remains limited by context length: models often miss key transition frames and struggle to maintain coherence across long time scales. To address this, we adapt Native Sparse Attention (NSA) to video-language models. Our method, VideoNSA, adapts Qwen2.5-VL through end-to-end training on a 216K video instruction dataset. We employ a hardware-aware hybrid approach to attention, preserving dense attention for text, while employing NSA for video. Compared to token-compression and training-free sparse baselines, VideoNSA achieves improved performance on long-video understanding, temporal reasoning, and spatial benchmarks. Further ablation analysis reveals four key findings: (1) reliable scaling to 128K tokens; (2) an optimal global-local attention allocation at a fixed budget; (3) task-dependent branch usage patterns; and (4) the learnable combined sparse attention help induce dynamic attention sinks.

---

## 59. From Frames to Clips: Efficient Key Clip Selection for Long-Form Video Understanding

**论文链接:** [http://arxiv.org/abs/2510.02262v1](http://arxiv.org/abs/2510.02262v1)

**作者:** Guangyu Sun, Archit Singhal, Burak Uzkent, Mubarak Shah, Chen Chen, Garin Kessler

**发布时间:** 2025-10-02

### GPT解析

### 总结

这项研究提出了一种名为F2C的训练-free方法，通过从孤立的关键帧扩展到关键片段来改善视频理解，同时采用自适应分辨率策略保持固定的计算预算。实验表明该方法在三个长视频基准测试中显著优于均匀采样。

### 背景

视频大语言模型(VLMs)在各种视觉语言任务上取得了显著成果，但实际应用受到'大海捞针'问题的限制：从原始视频帧产生的大量视觉tokens耗尽了模型的上下文窗口。现有解决方案通过选择稀疏帧集缓解此问题，但这种帧级选择丢弃了基本的时间动态，导致对运动和事件连续性的次优推理。

### 目的

系统探索时间信息的影响，证明从孤立的关键帧扩展到关键片段可以改善视频理解，同时保持固定计算预算并适应更大的片段token占用空间。

### 方法

提出了一种自适应分辨率策略，动态平衡空间分辨率和片段长度，确保每个视频的token数量恒定。该方法从帧级选择扩展到片段级选择，保留了时间连贯性，同时通过调整分辨率来控制计算成本。

### 主要发现

在三个长视频基准测试(Video-MME、LongVideoBench和MLVU)上，F2C方法分别比均匀采样高出8.1%、5.6%和10.3%。这些结果突显了在帧选择中保持时间连贯性的重要性。

### 结论

帧选择中保持时间连贯性对视频理解至关重要，从关键帧到关键片段的转变结合自适应分辨率策略，为扩展视频LLM到实际应用提供了实用路径。

### 翻译

视频大语言模型(VLMs)在各种视觉语言任务上取得了显著成果，但它们的实际应用受到'大海捞针'问题的限制：从原始视频帧产生的大量视觉tokens耗尽了模型的上下文窗口。现有的解决方案通过选择稀疏的帧集来缓解这一问题，但这种帧级选择丢弃了基本的时间动态，导致对运动和事件连续性的次优推理。在本工作中，我们系统性地探索了时间信息的影响，并证明将选择从孤立的关键帧扩展到关键片段(短时间连贯的片段)可以改善视频理解。为了在适应片段更大的token占用空间的同时保持固定的计算预算，我们提出了一种自适应分辨率策略，动态平衡空间分辨率和片段长度，确保每个视频的token数量恒定。在三个长视频基准测试上的实验表明，我们的训练-free方法F2C在Video-MME、LongVideoBench和MLVU基准测试上分别比均匀采样高出8.1%、5.6%和10.3%。这些结果突显了在帧选择中保持时间连贯性的重要性，并为扩展视频LLM到真实世界视频理解应用提供了实用途径。项目网页可在https://guangyusun.com/f2c获取。


### 论文摘要

Video Large Language Models (VLMs) have achieved remarkable results on a variety of vision language tasks, yet their practical use is limited by the "needle in a haystack" problem: the massive number of visual tokens produced from raw video frames exhausts the model's context window. Existing solutions alleviate this issue by selecting a sparse set of frames, thereby reducing token count, but such frame-wise selection discards essential temporal dynamics, leading to suboptimal reasoning about motion and event continuity. In this work we systematically explore the impact of temporal information and demonstrate that extending selection from isolated key frames to key clips, which are short, temporally coherent segments, improves video understanding. To maintain a fixed computational budget while accommodating the larger token footprint of clips, we propose an adaptive resolution strategy that dynamically balances spatial resolution and clip length, ensuring a constant token count per video. Experiments on three long-form video benchmarks demonstrate that our training-free approach, F2C, outperforms uniform sampling up to 8.1%, 5.6%, and 10.3% on Video-MME, LongVideoBench and MLVU benchmarks, respectively. These results highlight the importance of preserving temporal coherence in frame selection and provide a practical pathway for scaling Video LLMs to real world video understanding applications. Project webpage is available at https://guangyusun.com/f2c .

---

## 60. TimeGazer: Temporal Modeling of Predictive Gaze Stabilization for AR Interaction

**论文链接:** [http://arxiv.org/abs/2510.01561v1](http://arxiv.org/abs/2510.01561v1)

**作者:** Yaozheng Xia, Zaiping Zhu, Bo Pang, Shaorong Wang, Sheng Li

**发布时间:** 2025-10-02

### GPT解析

### 总结

这篇论文提出了TimeGazer，一种用于增强现实环境中视线稳定的新方法，通过时间回归模型预测理想注视轨迹，提高交互准确性和效率。

### 背景

在沉浸式AR环境中，视线稳定对实现流畅、准确和高效的交互至关重要。然而，活动视线任务中的注视序列常表现出不规则分散和系统性偏差，主要由人眼运动生理学、AR设备精度不足和环境干扰引起，损害了交互性能。

### 目的

解决AR环境中视线不稳定的问题，提高交互准确性和视觉参与度，增强任务驱动AR交互中的注意力一致性和响应性。

### 方法

将视线稳定重新表述为序列到序列的时间回归问题，从搜索阶段的历史视线动态中预测目标注视阶段的理想化注视轨迹。提出合成数据生成和混合策略，产生空间集中、以目标为中心的注视参考，丰富训练空间并增强模型泛化能力。在54名参与者通过Microsoft HoloLens 2收集的数据集上训练和评估TimeGazer。

### 主要发现

统计结果表明TimeGazer显著提高了交互准确性并减少了完成时间，证实了预测性视线稳定的时间建模可以增强任务驱动AR交互中的注意力一致性和响应性。

### 结论

TimeGazer在推进自适应基于视线的界面和沉浸式系统时间建模研究方面具有广泛潜力，为解决AR环境中视线不稳定问题提供了有效解决方案。

### 翻译

视线稳定对于在沉浸式增强现实(AR)环境中实现流畅、准确和高效的交互至关重要，特别是在任务导向的视觉行为期间。然而，在活动视线任务中捕获的注视序列通常表现出不规则分散和从目标位置的系统性偏差，这种变异性主要由人眼运动生理学、AR头戴式设备跟踪和校准精度不足以及环境干扰的综合效应引起，这损害了交互性能和视觉参与度。为了解决这个问题，我们提出了TimeGazer，它将视线稳定重新表述为序列到序列的时间回归问题，从搜索阶段的历史视线动态中预测目标注视阶段的理想化注视轨迹。我们提出了一种合成数据生成和混合策略，产生空间集中、以目标为中心的注视参考，与任务目标保持一致，大大丰富了训练空间并增强了模型泛化能力。我们在54名参与者通过Microsoft HoloLens 2收集的真实和增强视线序列的混合数据集上训练和评估了TimeGazer，并在多个预测时间范围内进行了测试。通过用户研究，统计结果表明TimeGazer显著提高了交互准确性并减少了完成时间，证实了预测性视线稳定的时间建模可以增强任务驱动AR交互中的注意力一致性和响应性。这些发现突显了TimeGazer在推进自适应基于视线的界面和沉浸式系统时间建模研究方面的更广泛潜力。


### 论文摘要

Gaze stabilization is critical for enabling fluid, accurate, and efficient interaction in immersive augmented reality (AR) environments, particularly during task-oriented visual behaviors. However, fixation sequences captured in active gaze tasks often exhibit irregular dispersion and systematic deviations from target locations, a variability primarily caused by the combined effects of human oculomotor physiology, insufficient AR headset tracking and calibration accuracy, and environmental disturbances, undermining interaction performance and visual engagement. To address this issue, we propose TimeGazer, which reformulates gaze stabilization as a sequence-to-sequence temporal regression problem, predicting idealized fixation trajectories for the target-fixation phase from historical gaze dynamics in the search phase. We present a synthetic data generation and blending strategy that produces spatially concentrated, target-centered fixation references aligned with task objectives, substantially enriching the training space and enhancing model generalization. We train and evaluate TimeGazer on a hybrid dataset of real and augmented gaze sequences collected via Microsoft HoloLens 2 from 54 participants across multiple prediction horizons. Through the user study, statistical results demonstrate that TimeGazer significantly improves interaction accuracy and reduces completion time, confirming that temporal modeling of predictive gaze stabilization can strengthen attentional consistency and responsiveness in task-driven AR interaction. These findings highlight the broader potential of TimeGazer for advancing adaptive gaze-based interfaces and temporal modeling research in immersive systems.

---

## 61. TAG-EQA: Text-And-Graph for Event Question Answering via Structured Prompting Strategies

**论文链接:** [http://arxiv.org/abs/2510.01391v1](http://arxiv.org/abs/2510.01391v1)

**作者:** Maithili Kadam, Francis Ferraro

**发布时间:** 2025-10-01

**备注:** Accepted in *sem 2025

### GPT解析

### 总结

本文介绍了一种名为TAG-EQA的提示框架，通过将因果事件图注入大型语言模型输入，增强模型对事件问答特别是因果和时间推理的能力。

### 背景

大型语言模型在通用语言任务上表现出色，但在处理基于事件的问答问题时，尤其是需要因果或时间推理的问题时，往往表现不佳。

### 目的

开发一个提示框架，通过注入因果事件图来增强大型语言模型对事件推理的能力，无需微调即可提高问答准确性。

### 方法

提出TAG-EQA框架，将结构化关系转换为自然语言语句，包含九种提示配置，结合三种策略（零样本、少样本、思维链）和三种输入模态（纯文本、纯图、文本+图），系统分析结构化知识何时以及如何辅助推理。

### 主要发现

在TORQUESTRA基准测试中，TAG-EQA相比纯文本基线平均提高5%的准确率，零样本设置中最高提高12%，图增强的思维链提示有效时提高幅度可达18%。

### 结论

因果图可以在不进行微调的情况下增强大型语言模型中的事件推理能力，为在基于提示的问答中编码结构提供了一种灵活的方式。

### 翻译

大型语言模型在通用语言任务上表现出色，但在处理基于事件的问答问题时，尤其是那些需要因果或时间推理的问题时，往往表现不佳。我们引入了TAG-EQA（Text-And-Graph for Event Question Answering），一种提示框架，通过将结构化关系转换为自然语言语句，将因果事件图注入到大型语言模型的输入中。TAG-EQA包含九种提示配置，结合了三种策略（零样本、少样本、思维链）和三种输入模态（纯文本、纯图、文本+图），能够系统分析结构化知识何时以及如何辅助推理。在TORQUESTRA基准测试中，TAG-EQA相比纯文本基线平均提高了5%的准确率，在零样本设置中最高可提高12%，当图增强的思维链提示有效时，提高幅度可达18%。虽然性能因模型和配置而异，但我们的研究表明，因果图可以在不进行微调的情况下增强大型语言模型中的事件推理能力，为在基于提示的问答中编码结构提供了一种灵活的方式。


### 论文摘要

Large language models (LLMs) excel at general language tasks but often struggle with event-based questions-especially those requiring causal or temporal reasoning. We introduce TAG-EQA (Text-And-Graph for Event Question Answering), a prompting framework that injects causal event graphs into LLM inputs by converting structured relations into natural-language statements. TAG-EQA spans nine prompting configurations, combining three strategies (zero-shot, few-shot, chain-of-thought) with three input modalities (text-only, graph-only, text+graph), enabling a systematic analysis of when and how structured knowledge aids inference. On the TORQUESTRA benchmark, TAG-EQA improves accuracy by 5% on average over text-only baselines, with gains up to 12% in zero-shot settings and 18% when graph-augmented CoT prompting is effective. While performance varies by model and configuration, our findings show that causal graphs can enhance event reasoning in LLMs without fine-tuning, offering a flexible way to encode structure in prompt-based QA.

---

## 62. Augmenting LLMs for General Time Series Understanding and Prediction

**论文链接:** [http://arxiv.org/abs/2510.01111v1](http://arxiv.org/abs/2510.01111v1)

**作者:** Felix Parker, Nimeesha Chan, Chi Zhang, Kimia Ghobadi

**发布时间:** 2025-10-01

### GPT解析

### 总结

本文提出了一种新型的时间序列增强大型语言模型(TsLLM)，结合了时间序列数据处理能力和自然语言理解能力，解决了传统时间序列模型无法处理文本信息以及大型语言模型难以处理数值型时间序列数据的问题。

### 背景

时间序列数据在医疗保健、金融和环境科学等关键领域的决策中非常重要，但传统时间序列模型缺乏处理文本信息的能力，无法整合非结构化上下文信息、回答领域特定问题或生成自然语言解释。同时，大型语言模型虽然擅长上下文推理和知识整合，却因基于文本的表示效率低下及预训练中时间数据接触有限而难以处理数值型时间序列数据。

### 目的

解决传统时间序列模型和大型语言模型之间的能力差距，通过专门的感知能力增强LLM以处理时间序列数据，创建一种能够同时处理数值计算和自然语言理解的新型时间序列分析范式。

### 方法

采用基于补丁的编码器-解码器架构增强大型语言模型，创建时间序列增强的大型语言模型(TsLLM)。在包含超过200万个交错时间序列和文本示例的大型语料库上训练，涵盖多种分析任务：具有上下文信息的预测、时间序列问答、模式解释、具有自然语言输出的分类和报告生成。

### 主要发现

TsLLM能够同时利用其语言理解能力和新获得的时序推理能力。虽然在传统基准测试中不是为了超越专业模型，但在需要时间序列分析与自然语言整合的任务上表现出色，这是现有方法无法提供的。

### 结论

这项工作建立了一种新的时间序列分析范式，弥合了数值计算和自然语言理解之间的鸿沟，通过自然语言交互使复杂的时序推理变得普及。

### 翻译

时间序列数据是医疗保健、金融和环境科学等许多关键领域决策的基础。然而，分析这些数据通常需要整合非结构化的上下文信息，回答领域特定问题，以及生成自然语言解释——这些能力是传统时间序列模型所缺乏的，因为它们无法处理文本。虽然大型语言模型擅长上下文推理和知识整合，但由于基于文本的表示效率低下以及在预训练过程中对时间数据的接触有限，它们难以处理数值型时间序列数据。我们通过基于补丁的编码器-解码器架构增强LLM，专门的时间序列感知能力来解决这一差距。我们在包含超过200万个交错时间序列和文本示例的大型语料库上训练这个时间序列增强的LLM(TsLLM)，涵盖各种分析任务：具有上下文信息的预测、时间序列问答、模式解释、具有自然语言输出的分类和报告生成。这种训练使TsLLM能够同时利用其语言理解能力和新获得的时序推理能力。虽然不是为了在传统基准测试中超越专业模型而设计的，但TsLLM在需要时间序列分析与自然语言整合的任务上表现出色，这是现有方法无法提供的。我们的工作建立了一种新的时间序列分析范式，弥合了数值计算和自然语言理解之间的鸿沟，通过自然语言交互使复杂的时序推理变得普及。


### 论文摘要

Time series data is fundamental to decision-making in many crucial domains including healthcare, finance, and environmental science. However, analyzing this data often requires incorporating unstructured contextual information, answering domain-specific questions, and generating natural language explanations -- capabilities that traditional time series models lack due to their inability to process text. While Large Language Models (LLMs) excel at contextual reasoning and knowledge integration, they struggle with numerical time series due to inefficient text-based representations and limited exposure to temporal data during pretraining. We address this gap by augmenting an LLM with specialized time series perception through a patch-based encoder-decoder architecture. We train this Time Series-augmented LLM (TsLLM) on a large corpus of over 2 million interleaved time series and text examples spanning diverse analysis tasks: forecasting with contextual information, time series question-answering, pattern explanation, classification with natural language outputs, and report generation. This training enables TsLLM to leverage both its language understanding and newly acquired temporal reasoning capabilities. While not designed to surpass specialized models on traditional benchmarks, TsLLM demonstrates strong performance on tasks requiring the integration of time series analysis with natural language -- capabilities that existing approaches cannot provide. Our work establishes a new paradigm for time series analysis that bridges numerical computation and natural language understanding, democratizing access to sophisticated temporal reasoning through natural language interaction.

---

## 63. Shape Happens: Automatic Feature Manifold Discovery in LLMs via Supervised Multi-Dimensional Scaling

**论文链接:** [http://arxiv.org/abs/2510.01025v1](http://arxiv.org/abs/2510.01025v1)

**作者:** Federico Tiblias, Irina Bigoulaeva, Jingcheng Niu, Simone Balloccu, Iryna Gurevych

**发布时间:** 2025-10-01

### GPT解析

### 总结

本研究引入了监督多维尺度变换（SMDS）方法，用于自动发现语言模型中的特征流形，并通过时间推理案例研究揭示了这些几何结构的多方面特性。

### 背景

线性表征假设认为语言模型将概念编码为其潜在空间中的方向，形成有组织的多维流形。先前的研究专注于为特定特征发现特定几何结构，因此缺乏泛化性。

### 目的

引入一种模型无关的方法来自动发现特征流形，并应用于时间推理以揭示这些结构的特性。

### 方法

开发了监督多维尺度变换（SMDS）方法，这是一种模型无关的方法，用于自动发现语言模型中的特征流形。将其应用于时间推理作为案例研究。

### 主要发现

不同特征形成各种几何结构，如圆形、线和簇。这些结构一致地反映所表示概念的特性；在不同模型族和规模中保持稳定；主动支持模型中的推理；并根据上下文变化动态重塑。

### 结论

这些发现揭示了特征流形的功能作用，支持了一种基于实体的推理模型，在该模型中，语言模型编码和转换结构化表示。

### 翻译

线性表征假设认为语言模型将概念编码为其潜在空间中的方向，形成有组织的、多维流形。先前的研究专注于为特定特征发现特定几何结构，因此缺乏泛化性。我们引入了监督多维尺度变换（SMDS），这是一种模型无关的方法，用于自动发现特征流形。我们将SMDS应用于时间推理作为案例研究，发现不同特征形成各种几何结构，如圆形、线和簇。SMDS揭示了这些结构的多个见解：它们一致地反映所表示概念的特性；在不同模型族和规模中保持稳定；主动支持模型中的推理；并根据上下文变化动态重塑。总之，我们的发现揭示了特征流形的功能作用，支持了一种基于实体的推理模型，在该模型中，语言模型编码和转换结构化表示。


### 论文摘要

The linear representation hypothesis states that language models (LMs) encode concepts as directions in their latent space, forming organized, multidimensional manifolds. Prior efforts focus on discovering specific geometries for specific features, and thus lack generalization. We introduce Supervised Multi-Dimensional Scaling (SMDS), a model-agnostic method to automatically discover feature manifolds. We apply SMDS to temporal reasoning as a case study, finding that different features form various geometric structures such as circles, lines, and clusters. SMDS reveals many insights on these structures: they consistently reflect the properties of the concepts they represent; are stable across model families and sizes; actively support reasoning in models; and dynamically reshape in response to context changes. Together, our findings shed light on the functional role of feature manifolds, supporting a model of entity-based reasoning in which LMs encode and transform structured representations.

---

## 64. CroSTAta: Cross-State Transition Attention Transformer for Robotic Manipulation

**论文链接:** [http://arxiv.org/abs/2510.00726v1](http://arxiv.org/abs/2510.00726v1)

**作者:** Giovanni Minelli, Giulio Turrisi, Victor Barasuol, Claudio Semini

**发布时间:** 2025-10-01

**备注:** Code and data available at https://github.com/iit-DLSLab/croSTAta

### GPT解析

### 总结

论文提出了一种交叉状态转换注意力Transformer模型，通过新颖的状态转换注意力机制(STA)来调节标准注意力权重，使机器人策略能够更好地根据执行历史调整行为，并在训练中结合结构化注意力和时序掩码以提高鲁棒性。

### 背景

通过监督学习从演示中学习机器人操作策略在遇到训练中未明确涵盖的执行变化时仍然具有挑战性。虽然通过注意力机制整合历史上下文可以提高鲁棒性，但标准方法处理序列中的所有过去状态，没有明确建模演示可能包含的时序结构，如失败和恢复模式。

### 目的

解决机器人策略在遇到训练中未明确涵盖的执行变化时的适应性问题，更好地建模演示中的时序结构，特别是失败和恢复模式。

### 方法

提出交叉状态转换注意力Transformer，使用新颖的状态转换注意力(STA)机制基于学习到的状态演化模式调节标准注意力权重，在训练中结合结构化注意力和时序掩码，从最近的timesteps中随机移除视觉信息，鼓励从历史上下文进行时序推理。

### 主要发现

在模拟评估中，STA在所有任务中一致优于标准的交叉注意力和时序建模方法(如TCN和LSTM网络)，在精度关键任务上比交叉注意力提高了2倍以上。

### 结论

所提出的STA机制能够有效建模状态之间的时序关系，特别是失败和恢复模式，通过结合结构化注意力和时序掩码训练，提高了机器人策略的鲁棒性和适应性。

### 翻译

通过从演示中进行监督学习来学习机器人操作策略，当策略遇到训练中未明确涵盖的执行变化时仍然具有挑战性。虽然通过注意力机制整合历史上下文可以提高鲁棒性，但标准方法处理序列中的所有过去状态，没有明确建模演示可能包含的时序结构，如失败和恢复模式。我们提出了一种交叉状态转换注意力Transformer，它采用了一种新颖的状态转换注意力(STA)机制，基于学习到的状态演化模式来调节标准注意力权重，使策略能够更好地根据执行历史调整其行为。我们的方法在训练中将这种结构化注意力与时序掩码相结合，从最近的timesteps中随机移除视觉信息，鼓励从历史上下文进行时序推理。在模拟中的评估表明，STA在所有任务中一致优于标准的交叉注意力和时序建模方法，如TCN和LSTM网络，在精度关键任务上比交叉注意力提高了2倍以上。


### 论文摘要

Learning robotic manipulation policies through supervised learning from demonstrations remains challenging when policies encounter execution variations not explicitly covered during training. While incorporating historical context through attention mechanisms can improve robustness, standard approaches process all past states in a sequence without explicitly modeling the temporal structure that demonstrations may include, such as failure and recovery patterns. We propose a Cross-State Transition Attention Transformer that employs a novel State Transition Attention (STA) mechanism to modulate standard attention weights based on learned state evolution patterns, enabling policies to better adapt their behavior based on execution history. Our approach combines this structured attention with temporal masking during training, where visual information is randomly removed from recent timesteps to encourage temporal reasoning from historical context. Evaluation in simulation shows that STA consistently outperforms standard cross-attention and temporal modeling approaches like TCN and LSTM networks across all tasks, achieving more than 2x improvement over cross-attention on precision-critical tasks.

---

## 65. Training-free Uncertainty Guidance for Complex Visual Tasks with MLLMs

**论文链接:** [http://arxiv.org/abs/2510.00705v1](http://arxiv.org/abs/2510.00705v1)

**作者:** Sanghwan Kim, Rui Xiao, Stephan Alaniz, Yongqin Xian, Zeynep Akata

**发布时间:** 2025-10-01

### GPT解析

### 总结

本研究提出了一种无需训练的框架，利用多模态大语言模型的内在不确定性作为主动指导信号，解决了MLLMs在细粒度感知任务中的局限性。

### 背景

多模态大语言模型(MLLMs)在细粒度感知方面存在困难，如在高分辨率图像中识别小物体或在长视频中找到关键时刻。现有方法通常依赖于复杂且任务特定的微调，限制了泛化能力并增加了模型复杂性。

### 目的

提出一种有效的、无需训练的框架，利用MLLMs的内在不确定性作为主动指导信号，增强其在细粒度多模态任务中的表现。

### 方法

基于模型输出熵在接收到相关视觉信息时会降低的核心见解，引入了一种统一的机制，通过响应不确定性对候选视觉输入进行评分，使模型能够自主关注最显著的数据。

### 主要发现

将这一简单原则应用于视觉搜索、长视频理解和时间定位三种复杂视觉任务，使现成的MLLMs能够实现与专门的微调方法相竞争的性能。

### 结论

利用内在不确定性是增强细粒度多模态性能的一种强大且通用的策略。

### 翻译

多模态大语言模型(MLLMs)通常在细粒度感知方面存在困难，例如在高分辨率图像中识别小物体或在长视频中找到关键时刻。现有工作通常依赖于复杂、任务特定的微调，这限制了它们的泛化能力并增加了模型复杂性。在本工作中，我们提出了一种有效的、无需训练的框架，利用MLLMs的内在不确定性作为主动指导信号。我们的核心见解是，当模型接收到相关的视觉信息时，其输出熵会降低。我们引入了一种统一的机制，通过响应不确定性对候选视觉输入进行评分，使模型能够自主关注最显著的数据。我们将这一简单原则应用于三种复杂的视觉任务：视觉搜索、长视频理解和时间定位，使现成的MLLMs能够实现与专门的微调方法相竞争的性能。我们的工作验证了利用内在不确定性是增强细粒度多模态性能的一种强大且通用的策略。


### 论文摘要

Multimodal Large Language Models (MLLMs) often struggle with fine-grained perception, such as identifying small objects in high-resolution images or finding key moments in long videos. Existing works typically rely on complicated, task-specific fine-tuning, which limits their generalizability and increases model complexity. In this work, we propose an effective, training-free framework that uses an MLLM's intrinsic uncertainty as a proactive guidance signal. Our core insight is that a model's output entropy decreases when presented with relevant visual information. We introduce a unified mechanism that scores candidate visual inputs by response uncertainty, enabling the model to autonomously focus on the most salient data. We apply this simple principle to three complex visual tasks: Visual Search, Long Video Understanding, and Temporal Grounding, allowing off-the-shelf MLLMs to achieve performance competitive with specialized, fine-tuned methods. Our work validates that harnessing intrinsic uncertainty is a powerful, general strategy for enhancing fine-grained multimodal performance.

---

## 66. IntrusionX: A Hybrid Convolutional-LSTM Deep Learning Framework with Squirrel Search Optimization for Network Intrusion Detection

**论文链接:** [http://arxiv.org/abs/2510.00572v1](http://arxiv.org/abs/2510.00572v1)

**作者:** Ahsan Farabi, Muhaiminul Rashid Shad, Israt Khandaker

**发布时间:** 2025-10-01

### GPT解析

### 总结

IntrusionX是一种混合深度学习框架，通过结合CNN和LSTM网络，并使用松鼠搜索算法进行优化，有效解决了入侵检测系统面临的网络攻击演变、高维流量数据和严重类别不平衡问题。

### 背景

入侵检测系统（IDS）面临持续挑战，包括不断演变的网络攻击、高维流量数据，以及基准数据集如NSL-KDD中存在的严重类别不平衡问题。

### 目的

解决入侵检测系统面临的挑战，提高检测性能，特别是对罕见类别的检测能力。

### 方法

提出IntrusionX，一种混合深度学习框架，集成CNN进行局部特征提取，LSTM进行时序建模，使用松鼠搜索算法进行超参数优化，并采用严格的预处理、分层数据分割和动态类别加权技术。

### 主要发现

在NSL-KDD数据集上，IntrusionX在二元分类中达到98%的准确率，在5类分类中达到87%的准确率，显著提高了少数类别的召回率（U2R: 71%，R2L: 93%）。

### 结论

IntrusionX的创新点在于其可复现的、不平衡感知的设计和元启发式优化，能有效处理入侵检测系统中的复杂挑战。

### 翻译

入侵检测系统（IDS）由于网络攻击的不断演变、高维流量数据以及基准数据集（如NSL-KDD）中严重的类别不平衡问题而面临持续挑战。为解决这些问题，我们提出了IntrusionX，一种混合深度学习框架，集成了卷积神经网络（CNN）用于局部特征提取和长短期记忆（LSTM）网络用于时序建模。该架构使用松鼠搜索算法（SSA）进一步优化，实现了有效的超参数调同时保持计算效率。我们的流程包含严格的预处理、分层数据分割和动态类别加权，以增强对罕见类别的检测。在NSL-KDD上的实验评估表明，IntrusionX在二元分类中达到98%的准确率，在5类分类中达到87%的准确率，少数类别召回率显著提高（U2R: 71%，R2L: 93%）。IntrusionX的创新点在于其可复现的、不平衡感知的设计和元启发式优化。


### 论文摘要

Intrusion Detection Systems (IDS) face persistent challenges due to evolving cyberattacks, high-dimensional traffic data, and severe class imbalance in benchmark datasets such as NSL-KDD. To address these issues, we propose IntrusionX, a hybrid deep learning framework that integrates Convolutional Neural Networks (CNNs) for local feature extraction and Long Short-Term Memory (LSTM) networks for temporal modeling. The architecture is further optimized using the Squirrel Search Algorithm (SSA), enabling effective hyperparameter tuning while maintaining computational efficiency. Our pipeline incorporates rigorous preprocessing, stratified data splitting, and dynamic class weighting to enhance the detection of rare classes. Experimental evaluation on NSL-KDD demonstrates that IntrusionX achieves 98% accuracy in binary classification and 87% in 5-class classification, with significant improvements in minority class recall (U2R: 71%, R2L: 93%). The novelty of IntrusionX lies in its reproducible, imbalance-aware design with metaheuristic optimization.

---

## 67. CardioBench: Do Echocardiography Foundation Models Generalize Beyond the Lab?

**论文链接:** [http://arxiv.org/abs/2510.00520v1](http://arxiv.org/abs/2510.00520v1)

**作者:** Darya Taratynova, Ahmed Aly, Numan Saeed, Mohammad Yaqub

**发布时间:** 2025-10-01

### GPT解析

### 总结

该研究介绍了CardioBench，一个用于超声心动图基础模型的全面基准，解决了该领域缺乏标准化评估工具的问题。通过整合八个公共数据集并涵盖多种任务类型，研究人员评估了不同类型的基础模型，发现了各模型的互补优势和局限性，为未来超声心动图基础模型的设计提供了指导。

### 背景

基础模型正在重塑医学影像领域，但在超声心动图中的应用仍然有限。虽然最近出现了几种专门针对超声心动图的基础模型，但缺乏标准化的基准来评估它们。超声心动图存在独特挑战，包括噪声采集、高帧冗余和有限的公共数据集，且大多数现有解决方案在私有数据上评估，限制了结果的可比性。

### 目的

引入CardioBench，一个全面的超声心动图基础模型基准，以解决该领域缺乏标准化评估工具的问题，促进不同模型之间的公平比较和未来发展。

### 方法

CardioBench将八个公共数据集统一为一个标准化套件，涵盖四个回归和五个分类任务，涉及功能、结构、诊断和视图识别终点。研究在一致的零样本、探测和校准协议下评估了多种领先的基础模型，包括心脏特定、生物医学和通用目的编码器，并发布了预处理、分割和公共评估流程。

### 主要发现

不同模型家族具有互补优势：时间建模对功能回归至关重要，检索在分布变化下提供鲁棒性，领域特定的文本编码器能捕获生理上有意义的轴。通用编码器迁移能力强，通常缩小与探测的差距，但在细粒度区分(如视图分类和细微病理识别)方面存在困难。

### 结论

通过发布预处理、分割和公共评估流程，CardioBench建立了可重现的参考点，并提供了有价值的见解，以指导未来超声心动图基础模型的设计，推动该领域的发展。

### 翻译

基础模型(FMs)正在重塑医学影像领域，但其在超声心动图中的应用仍然有限。虽然最近已引入几种专门针对超声心动图的基础模型，但尚无标准化基准来评估它们。超声心动图存在独特挑战，包括噪声采集、高帧冗余和有限的公共数据集。大多数现有解决方案在私有数据上进行评估，限制了可比性。为解决这一问题，我们引入了CardioBench，一个用于超声心动图基础模型的全面基准。CardioBench将八个公共数据集统一为一个标准化套件，涵盖四个回归和五个分类任务，涉及功能、结构、诊断和视图识别终点。我们在一致的零样本、探测和校准协议下评估了几种领先的基础模型，包括心脏特定、生物医学和通用目的编码器。我们的结果突显了不同模型家族的互补优势：时间建模对功能回归至关重要，检索在分布变化下提供鲁棒性，领域特定的文本编码器捕获生理上有意义的轴。通用编码器迁移能力强，通常缩小与探测的差距，但在细粒度区分如视图分类和细微病理识别方面存在困难。通过发布预处理、分割和公共评估流程，CardioBench建立了可重现的参考点，并提供了有价值的见解，以指导未来超声心动图基础模型的设计。


### 论文摘要

Foundation models (FMs) are reshaping medical imaging, yet their application in echocardiography remains limited. While several echocardiography-specific FMs have recently been introduced, no standardized benchmark exists to evaluate them. Echocardiography poses unique challenges, including noisy acquisitions, high frame redundancy, and limited public datasets. Most existing solutions evaluate on private data, restricting comparability. To address this, we introduce CardioBench, a comprehensive benchmark for echocardiography FMs. CardioBench unifies eight publicly available datasets into a standardized suite spanning four regression and five classification tasks, covering functional, structural, diagnostic, and view recognition endpoints. We evaluate several leading FM, including cardiac-specific, biomedical, and general-purpose encoders, under consistent zero-shot, probing, and alignment protocols. Our results highlight complementary strengths across model families: temporal modeling is critical for functional regression, retrieval provides robustness under distribution shift, and domain-specific text encoders capture physiologically meaningful axes. General-purpose encoders transfer strongly and often close the gap with probing, but struggle with fine-grained distinctions like view classification and subtle pathology recognition. By releasing preprocessing, splits, and public evaluation pipelines, CardioBench establishes a reproducible reference point and offers actionable insights to guide the design of future echocardiography foundation models.

---

## 68. From Seeing to Predicting: A Vision-Language Framework for Trajectory Forecasting and Controlled Video Generation

**论文链接:** [http://arxiv.org/abs/2510.00806v1](http://arxiv.org/abs/2510.00806v1)

**作者:** Fan Yang, Zhiyang Chen, Yousong Zhu, Xin Li, Jinqiao Wang

**发布时间:** 2025-10-01

### GPT解析

### 总结

论文提出了TrajVLM-Gen，一个两阶段的物理感知图像到视频生成框架，通过视觉语言模型预测运动轨迹并基于注意力机制进行视频生成，解决了现有视频生成模型中物理不一致的问题。

### 背景

当前的视频生成模型产生的运动在物理上不一致，违反了真实世界的动力学规律。

### 目的

开发一个能够生成符合物理规律的动态视频的框架，解决现有视频生成模型中物理不一致的问题。

### 方法

提出TrajVLM-Gen两阶段框架：首先使用视觉语言模型预测与真实世界物理保持一致的粗粒度运动轨迹；其次，通过基于注意力的机制引导视频生成，进行细粒度运动优化。同时，基于具有真实运动模式的视频跟踪数据构建了轨迹预测数据集。

### 主要发现

在UCF-101和MSR-VTT数据集上的实验表明，TrajVLM-Gen优于现有方法，在UCF-101上实现了545的FVD分数，在MSR-VTT上实现了539的FVD分数。

### 结论

TrajVLM-Gen框架通过结合物理感知的运动轨迹预测和基于注意力的视频生成，能够生成更符合物理规律的动态视频，性能优于现有方法。

### 翻译

当前的视频生成模型产生的运动在物理上不一致，违反了真实世界的动力学规律。我们提出了TrajVLM-Gen，一个用于物理感知图像到视频生成的两阶段框架。首先，我们采用视觉语言模型预测与真实世界物理保持一致的粗粒度运动轨迹。其次，这些轨迹通过基于注意力的机制引导视频生成，进行细粒度运动优化。我们基于具有真实运动模式的视频跟踪数据构建了轨迹预测数据集。在UCF-101和MSR-VTT上的实验表明，TrajVLM-Gen优于现有方法，在UCF-101上实现了545的竞争性FVD分数，在MSR-VTT上实现了539的FVD分数。


### 论文摘要

Current video generation models produce physically inconsistent motion that violates real-world dynamics. We propose TrajVLM-Gen, a two-stage framework for physics-aware image-to-video generation. First, we employ a Vision Language Model to predict coarse-grained motion trajectories that maintain consistency with real-world physics. Second, these trajectories guide video generation through attention-based mechanisms for fine-grained motion refinement. We build a trajectory prediction dataset based on video tracking data with realistic motion patterns. Experiments on UCF-101 and MSR-VTT demonstrate that TrajVLM-Gen outperforms existing methods, achieving competitive FVD scores of 545 on UCF-101 and 539 on MSR-VTT.

---

## 69. EgoTraj-Bench: Towards Robust Trajectory Prediction Under Ego-view Noisy Observations

**论文链接:** [http://arxiv.org/abs/2510.00405v1](http://arxiv.org/abs/2510.00405v1)

**作者:** Jiayi Liu, Jiaming Zhou, Ke Ye, Kun-Yu Lin, Allan Wang, Junwei Liang

**发布时间:** 2025-10-01

### GPT解析

### 总结

该研究提出了EgoTraj-Bench基准和BiFlow模型，用于解决从自我中心视角进行可靠轨迹预测的问题，通过考虑真实世界中的感知约束，提高了模型的鲁棒性和预测精度。

### 背景

从自我中心视角进行可靠的轨迹预测对于在以人为中心的环境中机器人导航至关重要。然而，现有方法通常假设理想化的观察历史，没有考虑第一视觉中固有的感知伪影，如遮挡、ID切换和跟踪漂移。

### 目的

弥合训练假设与部署现实之间的差距，开发能够应对真实世界自我中心感知挑战的轨迹预测系统。

### 方法

引入EgoTraj-Bench基准，将嘈杂的第一视觉历史与清晰的鸟瞰图未来轨迹相关联；提出BiFlow双流流匹配模型，通过共享潜在表示同时去噪历史观察和预测未来运动；引入EgoAnchor机制，通过特征调制将预测解码器条件化为提取的历史特征。

### 主要发现

BiFlow实现了最先进的性能，将minADE和minFDE平均降低了10-15%，并展示了卓越的鲁棒性。

### 结论

EgoTraj-Bench基准和BiFlow模型为开发真正能够抵御真实世界自我中心感知挑战的轨迹预测系统提供了关键基础。

### 翻译

从以自我为中心的视角进行可靠的轨迹预测对于在以人为中心的环境中机器人导航至关重要。然而，现有方法通常假设理想化的观察历史，未能考虑第一视觉中固有的感知伪影，如遮挡、ID切换和跟踪漂移。这种训练假设与部署现实之间的差异严重限制了模型的鲁棒性。为了弥合这一差距，我们引入了EgoTraj-Bench，这是第一个将嘈杂的第一视觉历史与清晰的鸟瞰图未来轨迹相关联的真实世界基准，使模型能够在真实的感知约束下进行鲁棒学习。基于此基准，我们提出了BiFlow，一种双流流匹配模型，通过利用共享的潜在表示同时去噪历史观察和预测未来运动。为了更好地建模智能体意图，BiFlow融入了我们的EgoAnchor机制，该机制通过特征调制将预测解码器条件化为提取的历史特征。大量实验表明，BiFlow实现了最先进的性能，将minADE和minFDE平均降低了10-15%，并展示了卓越的鲁棒性。我们期望，我们的基准和模型将为开发真正能够抵御真实世界自我中心感知挑战的轨迹预测系统提供关键基础。


### 论文摘要

Reliable trajectory prediction from an ego-centric perspective is crucial for robotic navigation in human-centric environments. However, existing methods typically assume idealized observation histories, failing to account for the perceptual artifacts inherent in first-person vision, such as occlusions, ID switches, and tracking drift. This discrepancy between training assumptions and deployment reality severely limits model robustness. To bridge this gap, we introduce EgoTraj-Bench, the first real-world benchmark that grounds noisy, first-person visual histories in clean, bird's-eye-view future trajectories, enabling robust learning under realistic perceptual constraints. Building on this benchmark, we propose BiFlow, a dual-stream flow matching model that concurrently denoises historical observations and forecasts future motion by leveraging a shared latent representation. To better model agent intent, BiFlow incorporates our EgoAnchor mechanism, which conditions the prediction decoder on distilled historical features via feature modulation. Extensive experiments show that BiFlow achieves state-of-the-art performance, reducing minADE and minFDE by 10-15% on average and demonstrating superior robustness. We anticipate that our benchmark and model will provide a critical foundation for developing trajectory forecasting systems truly resilient to the challenges of real-world, ego-centric perception.

---

## 70. Physics-Informed Neural Controlled Differential Equations for Scalable Long Horizon Multi-Agent Motion Forecasting

**论文链接:** [http://arxiv.org/abs/2510.00401v1](http://arxiv.org/abs/2510.00401v1)

**作者:** Shounak Sural, Charles Kekeh, Wenliang Liu, Federico Pecora, Mouhacine Benosman

**发布时间:** 2025-10-01

### GPT解析

### 总结

本文提出了一种名为PINCoDE的物理信息神经控制微分方程模型，用于长时间范围的多机器人运动预测，该模型能够处理大规模多机器人系统并显著提高预测准确性。

### 背景

长时间范围的多机器人运动预测面临非线性智能体交互、累积预测误差和动力学连续时间演化的挑战。学习此类系统的动力学对行程时间预测、预测引导规划和生成式仿真等应用具有重要意义。

### 目的

开发一个基于多智能体目标条件的高效轨迹预测模型，实现长时间范围的多机器人运动预测。

### 方法

基于神经控制微分方程(CDEs)构建模型，与离散时间方法不同，它在连续时间运行，能够结合物理约束和偏差共同建模多机器人动力学。PINCoDE学习微分方程参数，从初始条件预测多智能体系统轨迹，并基于未来目标条件强制执行物理约束。采用可扩展策略使模型从10个机器人扩展到100个机器人，无需额外参数，并使用课程学习进行渐进式训练。

### 主要发现

对于1分钟的时间范围，模型预测的平均位移误差低于0.5米；与分析模型相比，PINCoDE在4分钟时间范围内的预测姿态误差减少了2.7倍。

### 结论

PINCoDE是一种有效的长时间范围多机器人运动预测方法，能够处理大规模多机器人系统，并通过结合物理约束和课程学习显著提高预测准确性。

### 翻译

对于多个自主机器人的长时间范围运动预测具有挑战性，这是由于非线性智能体交互、累积预测误差和动力学的连续时间演化。此类系统的学习动力学在各种应用中很有用，如行程时间预测、预测引导规划和生成式仿真。在本工作中，我们的目标是开发一个基于多智能体目标条件的高效轨迹预测模型。受最近物理学引导深度学习在部分已知动力学系统中成功的启发，我们开发了一种基于神经控制微分方程(CDEs)的模型，用于长时间范围的运动预测。与RNNs和transformers等离散时间方法不同，神经CDEs在连续时间运行，使我们能够结合物理信息约束和偏差来共同建模多机器人动力学。我们的方法名为PINCoDE(物理信息神经控制微分方程)，学习微分方程参数，可用于从初始条件预测多智能体系统的轨迹。PINCoDE基于未来目标条件，并强制执行机器人运动在长时间内的物理约束。我们采用一种策略，使模型能够从10个机器人扩展到100个机器人，而无需额外模型参数，同时产生1分钟时间范围内平均位移误差低于0.5米的预测。此外，使用课程学习对我们的PINCoDE模型进行渐进式训练，与分析模型相比，在4分钟时间范围内的预测姿态误差减少了2.7倍。


### 论文摘要

Long-horizon motion forecasting for multiple autonomous robots is challenging due to non-linear agent interactions, compounding prediction errors, and continuous-time evolution of dynamics. Learned dynamics of such a system can be useful in various applications such as travel time prediction, prediction-guided planning and generative simulation. In this work, we aim to develop an efficient trajectory forecasting model conditioned on multi-agent goals. Motivated by the recent success of physics-guided deep learning for partially known dynamical systems, we develop a model based on neural Controlled Differential Equations (CDEs) for long-horizon motion forecasting. Unlike discrete-time methods such as RNNs and transformers, neural CDEs operate in continuous time, allowing us to combine physics-informed constraints and biases to jointly model multi-robot dynamics. Our approach, named PINCoDE (Physics-Informed Neural Controlled Differential Equations), learns differential equation parameters that can be used to predict the trajectories of a multi-agent system starting from an initial condition. PINCoDE is conditioned on future goals and enforces physics constraints for robot motion over extended periods of time. We adopt a strategy that scales our model from 10 robots to 100 robots without the need for additional model parameters, while producing predictions with an average ADE below 0.5 m for a 1-minute horizon. Furthermore, progressive training with curriculum learning for our PINCoDE model results in a 2.7X reduction of forecasted pose error over 4 minute horizons compared to analytical models.

---

## 71. Accelerating Long-Term Molecular Dynamics with Physics-Informed Time-Series Forecasting

**论文链接:** [http://arxiv.org/abs/2510.01206v1](http://arxiv.org/abs/2510.01206v1)

**作者:** Hung Le, Sherif Abbas, Minh Hoang Nguyen, Van Dai Do, Huu Hiep Nguyen, Dung Nguyen

**发布时间:** 2025-09-16

**备注:** 16 pages, preprint

### GPT解析

### 总结

提出了一种将分子动力学模拟作为时间序列预测问题的新方法，通过位移而非绝对位置预测原子轨迹，结合物理信息损失机制，显著提高了模拟效率与准确性。

### 背景

传统的密度泛函理论方法计算成本高，限制了长期模拟的可行性，而分子动力学模拟对于理解材料科学和生物物理学中的原子尺度过程至关重要。

### 目的

开发一种高效的分子动力学模拟方法，克服传统DFT方法的计算限制，实现长期原子尺度过程的准确模拟。

### 方法

将MD模拟表述为时间序列预测问题，基于DFT参数化的成对Morse势函数构建物理信息损失和推理机制，通过惩罚非物理原子接近来确保物理合理性。

### 主要发现

该方法在多种材料的模拟准确性上一致优于标准基线方法，能够在几分钟内稳定建模数千个MD步骤，为昂贵的DFT模拟提供了一种可扩展的替代方案。

### 结论

融入物理知识对于提高原子轨迹预测的可靠性和精确度至关重要，该方法为材料科学和生物物理学中的原子尺度过程研究提供了高效工具。

### 翻译

高效的分子动力学模拟对于理解材料科学和生物物理学中的原子尺度过程至关重要。传统的密度泛函理论方法计算成本高，限制了长期模拟的可行性。我们提出了一种新方法，将MD模拟表述为时间序列预测问题，使先进的预测模型能够通过位移而非绝对位置来预测原子轨迹。我们基于DFT参数化的成对Morse势函数，融入了物理信息损失和推理机制，通过惩罚非物理原子接近来确保物理合理性。我们的方法在多种材料的模拟准确性上一致优于标准基线方法。结果强调了融入物理知识以提高原子轨迹预测可靠性和精确度的重要性。值得注意的是，它能够在几分钟内稳定建模数千个MD步骤，为昂贵的DFT模拟提供了一种可扩展的替代方案。


### 论文摘要

Efficient molecular dynamics (MD) simulation is vital for understanding atomic-scale processes in materials science and biophysics. Traditional density functional theory (DFT) methods are computationally expensive, which limits the feasibility of long-term simulations. We propose a novel approach that formulates MD simulation as a time-series forecasting problem, enabling advanced forecasting models to predict atomic trajectories via displacements rather than absolute positions. We incorporate a physics-informed loss and inference mechanism based on DFT-parametrised pair-wise Morse potential functions that penalize unphysical atomic proximity to enforce physical plausibility. Our method consistently surpasses standard baselines in simulation accuracy across diverse materials. The results highlight the importance of incorporating physics knowledge to enhance the reliability and precision of atomic trajectory forecasting. Remarkably, it enables stable modeling of thousands of MD steps in minutes, offering a scalable alternative to costly DFT simulations.

---

## 72. Calibrating the Full Predictive Class Distribution of 3D Object Detectors for Autonomous Driving

**论文链接:** [http://arxiv.org/abs/2510.01829v1](http://arxiv.org/abs/2510.01829v1)

**作者:** Cornelius Schröder, Marius-Raphael Schlüter, Markus Lienkamp

**发布时间:** 2025-10-02

**DOI:** 10.1109/IV64158.2025.11097526

### GPT解析

### 总结

该研究针对三维目标检测器的分类任务提出了一种置信度校准方法，通过引入辅助正则化损失项来改善预测置信度的校准效果。

### 背景

在自主系统中，精确的目标检测和不确定性估计对于系统的自我感知和安全运行至关重要。然而，三维目标检测器的置信度校准仍存在挑战。

### 目的

研究旨在解决三维目标检测器分类任务的置信度校准问题，提出能够捕捉主要和次要类别预测校准情况的度量指标，并开发相应的校准方法。

### 方法

作者提出了两种辅助正则化损失项：一种针对主要预测的校准，另一种针对完整预测向量的校准。将这些方法与等度规回归相结合，应用于CenterPoint、PillarNet和DSVT-Pillar三种三维目标检测模型。

### 主要发现

将完整类别预测的正则化损失项与等度规回归相结合，对CenterPoint和PillarNet的主要和次要类别预测实现了最佳校准效果。然而，DSVT-Pillar无法使用相同的方法同时校准主要和次要预测。

### 结论

通过提出的正则化损失项与等度规回归的结合，可以有效改善三维目标检测器的置信度校准，但不同模型可能需要采用不同的校准策略才能达到最佳效果。

### 翻译

在自主系统中，精确的目标检测和不确定性估计对于自我感知和安全运行至关重要。这项工作解决了三维目标检测器分类任务的置信度校准问题。我们认为有必要对所有类别的完整预测置信度分布进行校准，并提出了一种能够捕捉主要和次要类别预测校准情况的度量指标。我们提出了两个辅助的正则化损失项，它们分别引入主要预测的校准或完整预测向量的校准作为训练目标。我们评估了一系列应用于CenterPoint、PillarNet和DSVT-Pillar的事后和训练时间方法，发现将完整类别预测的正则化损失项与等度规回归相结合，对于CenterPoint和PillarNet的主要和次要类别预测都能实现最佳校准。我们还发现DSVT-Pillar无法使用相同的方法同时校准主要和次要预测。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决3D目标检测器的置信度校准问题，特别是针对自动驾驶场景中目标分类的不确定性估计。这个问题在现实中非常重要，因为自动驾驶系统需要准确评估其预测的不确定性，以便在规划行为时考虑可能的环境感知错误。如果系统对自己的预测置信度估计不准确，可能会导致危险决策。此外，完整预测分布（包括主导和次要类别）的校准对安全关键应用尤为重要，例如当物体被分类为车辆或弱势道路使用者的置信度相近时，系统需要了解这种不确定性。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者从强校准条件出发，认为理想的校准应该考虑所有类别的预测概率分布，而不仅仅是主导类别。他们借鉴了现有的置信度校准方法，包括分箱方法（直方图分箱、等度回归）和缩放方法（Platt缩放、温度缩放），以及训练时校准方法如标签平滑和Focal Loss。在此基础上，作者设计了新的校准度量（Full D-ECE）和两种辅助损失函数（L_DECE和L_FullDECE），用于在训练过程中引入校准目标。作者还系统地评估了这些方法与现有训练后校准方法的组合效果。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是全面校准3D目标检测器的预测分布，不仅校准主导类别的置信度，还校准所有类别（包括次要类别）的预测分布。整体实现流程包括：1) 使用Waymo数据集训练三种3D目标检测器；2) 将D-ECE或Full D-ECE作为辅助损失函数添加到标准分类损失中进行训练时校准；3) 应用训练后校准方法（温度缩放、Platt缩放和等度回归）；4) 评估不同组合方法的校准性能（使用D-ECE和Full D-ECE指标）和检测性能（mAP和mAPH指标）。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1) 提出完整检测期望校准误差（Full D-ECE）指标，评估所有类别预测的校准质量；2) 设计两种辅助损失函数（L_DECE和L_FullDECE），在训练过程中引入校准目标；3) 提出全面校准策略，同时考虑主导和次要类别预测；4) 系统性评估多种训练时和训练后校准方法的组合效果。相比之前的工作，这篇论文专门针对3D目标检测器，而不仅仅是2D检测器或一般分类器；它关注完整预测分布而非仅主导类别；提出的校准度量专门为3D检测场景设计；作者还分析了不同架构（如Transformer架构）对校准方法的影响，发现需要特定策略。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '这篇论文提出了一种针对3D目标检测器的全面置信度校准方法，通过新的校准度量和辅助损失函数，结合训练时和训练后校准技术，显著提高了自动驾驶系统中目标检测的不确定性估计质量，同时保持了检测性能。'}


### 论文摘要

In autonomous systems, precise object detection and uncertainty estimation are critical for self-aware and safe operation. This work addresses confidence calibration for the classification task of 3D object detectors. We argue that it is necessary to regard the calibration of the full predictive confidence distribution over all classes and deduce a metric which captures the calibration of dominant and secondary class predictions. We propose two auxiliary regularizing loss terms which introduce either calibration of the dominant prediction or the full prediction vector as a training goal. We evaluate a range of post-hoc and train-time methods for CenterPoint, PillarNet and DSVT-Pillar and find that combining our loss term, which regularizes for calibration of the full class prediction, and isotonic regression lead to the best calibration of CenterPoint and PillarNet with respect to both dominant and secondary class predictions. We further find that DSVT-Pillar can not be jointly calibrated for dominant and secondary predictions using the same method.

---

## 73. Inferring Dynamic Physical Properties from Video Foundation Models

**论文链接:** [http://arxiv.org/abs/2510.02311v1](http://arxiv.org/abs/2510.02311v1)

**作者:** Guanqi Zhan, Xianzheng Ma, Weidi Xie, Andrew Zisserman

**发布时间:** 2025-10-02

### GPT解析

### 总结

论文研究从视频中预测动态物理属性的方法，收集了新数据集并探索了三种不同的推断方法。

### 背景

动态物理属性（如弹性、粘度和摩擦力）需要时间信息才能推断，传统方法可能难以准确捕捉这些属性。

### 目的

开发能够从视频中准确预测动态物理属性的方法，包括弹性、粘度和摩擦力。

### 方法

收集了新的视频数据集，并探索了三种推断方法：基于计算机视觉的oracle方法、基于预训练视频模型的提示机制、以及多模态大型语言模型的提示策略。

### 主要发现

视频基础模型在生成或自监督训练下表现相似，但略低于oracle方法；MLLMs目前表现较差，但通过适当提示可改善。

### 结论

视频基础模型在动态物理属性预测方面具有潜力，而MLLMs需要进一步优化才能达到类似性能。

### 翻译

我们研究从视频中预测动态物理属性的任务。更具体地说，我们考虑需要时间信息才能推断的物理属性：弹跳物体的弹性、流动液体的粘度和物体在表面上滑动的动态摩擦力。为此，我们做出以下贡献：(i)我们为每个物理属性收集了一个新的视频数据集，包括合成的训练和测试分割，以及用于现实世界评估的真实分割。(ii)我们探索了三种从视频中推断物理属性的方法：(a)oracle方法，我们使用经典计算机视觉技术提供反映该属性固有视觉线索；(b)使用视觉提示和可训练提示向量在预训练的视频生成和自监督模型上进行交叉注意力的简单读取机制；(c)多模态大型语言模型(MLLMs)的提示策略。(iii)我们表明，以生成或自监督方式训练的视频基础模型实现了类似oracle方法的性能，尽管略低于oracle方法，而MLLMs目前劣于其他模型，但通过适当的提示可以提高其性能。


### 论文摘要

We study the task of predicting dynamic physical properties from videos. More specifically, we consider physical properties that require temporal information to be inferred: elasticity of a bouncing object, viscosity of a flowing liquid, and dynamic friction of an object sliding on a surface. To this end, we make the following contributions: (i) We collect a new video dataset for each physical property, consisting of synthetic training and testing splits, as well as a real split for real world evaluation. (ii) We explore three ways to infer the physical property from videos: (a) an oracle method where we supply the visual cues that intrinsically reflect the property using classical computer vision techniques; (b) a simple read out mechanism using a visual prompt and trainable prompt vector for cross-attention on pre-trained video generative and self-supervised models; and (c) prompt strategies for Multi-modal Large Language Models (MLLMs). (iii) We show that video foundation models trained in a generative or self-supervised manner achieve a similar performance, though behind that of the oracle, and MLLMs are currently inferior to the other models, though their performance can be improved through suitable prompting.

---

## 74. F2LLM Technical Report: Matching SOTA Embedding Performance with 6 Million Open-Source Data

**论文链接:** [http://arxiv.org/abs/2510.02294v1](http://arxiv.org/abs/2510.02294v1)

**作者:** Ziyin Zhang, Zihan Liao, Hang Yu, Peng Di, Rui Wang

**发布时间:** 2025-10-02

### GPT解析

### 总结

F2LLM是一种新型的嵌入模型套件，包含0.6B、1.7B和4B三种参数规模版本，通过从基础模型直接微调而非使用对比预训练和合成数据的方式训练，在保持高性能的同时降低了训练成本。

### 背景

现有的顶级嵌入模型通常需要大规模对比预训练、复杂的训练流程和昂贵的合成训练数据，这限制了其广泛应用和研究。

### 目的

开发一种在训练成本、模型大小和嵌入性能之间取得平衡的高效嵌入模型，并提供可复现且经济实惠的基线。

### 方法

从基础模型直接微调F2LLM，使用从开源非合成数据集中筛选的600万查询-文档-负元组进行训练，避免了复杂的对比预训练和昂贵的合成数据需求。

### 主要发现

在MTEB英语排行榜上，F2LLM-4B在约40亿参数模型中排名第2，总体排名第7；F2LLM-1.7B在10-20亿参数规模模型中排名第1，证明了其高效性能。

### 结论

F2LLM通过创新的训练方法，成功在训练成本、模型大小和性能之间取得了平衡，为嵌入模型领域提供了一个强大、可复现且经济实惠的基线。

### 翻译

我们引入F2LLM - 基础到特性大型语言模型，这是一套包含三种尺寸的最先进嵌入模型：0.6B、1.7B和4B。与之前需要大量对比预训练、复杂训练流程和昂贵合成训练数据的顶级嵌入模型不同，F2LLM是从基础模型直接微调而来，使用了从开源非合成数据集中筛选的600万查询-文档-负元组，在训练成本、模型大小和嵌入性能之间取得了良好平衡。在MTEB英语排行榜上，F2LLM-4B在约40亿参数模型中排名第2，总体排名第7，而F2LLM-1.7B在10-20亿参数规模模型中排名第1。为了促进该领域未来的研究，我们发布了模型、训练数据集和代码，使F2LLM成为未来工作的强大、可复现且经济实惠的基线。


### 论文摘要

We introduce F2LLM - Foundation to Feature Large Language Models, a suite of state-of-the-art embedding models in three sizes: 0.6B, 1.7B, and 4B. Unlike previous top-ranking embedding models that require massive contrastive pretraining, sophisticated training pipelines, and costly synthetic training data, F2LLM is directly finetuned from foundation models on 6 million query-document-negative tuples curated from open-source, non-synthetic datasets, striking a strong balance between training cost, model size, and embedding performance. On the MTEB English leaderboard, F2LLM-4B ranks 2nd among models with approximately 4B parameters and 7th overall, while F2LLM-1.7B ranks 1st among models in the 1B-2B size range. To facilitate future research in the field, we release the models, training dataset, and code, positioning F2LLM as a strong, reproducible, and budget-friendly baseline for future works.

---

## 75. Test-Time Anchoring for Discrete Diffusion Posterior Sampling

**论文链接:** [http://arxiv.org/abs/2510.02291v1](http://arxiv.org/abs/2510.02291v1)

**作者:** Litu Rout, Andreas Lugmayr, Yasamin Jafarian, Srivatsan Varadharajan, Constantine Caramanis, Sanjay Shakkottai, Ira Kemelmacher-Shlizerman

**发布时间:** 2025-10-02

**备注:** Preprint

### GPT解析

### 总结

研究使用预训练的离散扩散基础模型进行后验采样，从噪声测量中恢复图像而无需重新训练特定任务模型

### 背景

扩散模型在生成建模方面取得显著成功，但大多依赖连续高斯扩散；离散扩散为文本和图像等分类数据提供统一建模框架，具有更快推理、更精细控制和无需训练的贝叶斯推断优势

### 目的

解决离散扩散后验采样面临的挑战：无导数引导产生稀疏信号、连续松弛限制适用性、分裂吉布斯采样器受维度诅咒影响

### 方法

引入锚定后验采样(APS)，用于掩码扩散基础模型，基于两个关键创新：离散嵌入空间中的量化期望（用于梯度类引导）和锚定重新掩码（用于自适应解码）

### 主要发现

该方法在标准基准上针对线性和非线性逆问题在离散扩散采样器中实现了最先进的性能

### 结论

该方法在无需训练的风格化和文本引导编辑中展示了其优势

### 翻译

我们研究使用预训练的离散扩散基础模型进行后验采样的问题，旨在从噪声测量中恢复图像而无需重新训练特定任务模型。虽然扩散模型在生成建模方面取得了显著成功，但大多数进展依赖于连续高斯扩散。相比之下，离散扩散为文本和图像等分类数据提供了统一的建模框架。除了统一性，离散扩散提供了更快的推理、更精细的控制和无需训练的贝叶斯推断，使其特别适合后验采样。然而，现有的离散扩散后验采样方法面临严重挑战：无导数引导产生稀疏信号，连续松弛限制了适用性，分裂吉布斯采样器受到维度诅咒的影响。为了克服这些限制，我们为掩码扩散基础模型引入了锚定后验采样(APS)，基于两个关键创新——离散嵌入空间中的量化期望（用于梯度类引导）和锚定重新掩码（用于自适应解码）。我们的方法在标准基准上针对线性和非线性逆问题在离散扩散采样器中实现了最先进的性能。我们进一步展示了该方法在无需训练的风格化和文本引导编辑中的优势。


### 论文摘要

We study the problem of posterior sampling using pretrained discrete diffusion foundation models, aiming to recover images from noisy measurements without retraining task-specific models. While diffusion models have achieved remarkable success in generative modeling, most advances rely on continuous Gaussian diffusion. In contrast, discrete diffusion offers a unified framework for jointly modeling categorical data such as text and images. Beyond unification, discrete diffusion provides faster inference, finer control, and principled training-free Bayesian inference, making it particularly well-suited for posterior sampling. However, existing approaches to discrete diffusion posterior sampling face severe challenges: derivative-free guidance yields sparse signals, continuous relaxations limit applicability, and split Gibbs samplers suffer from the curse of dimensionality. To overcome these limitations, we introduce Anchored Posterior Sampling (APS) for masked diffusion foundation models, built on two key innovations -- quantized expectation for gradient-like guidance in discrete embedding space, and anchored remasking for adaptive decoding. Our approach achieves state-of-the-art performance among discrete diffusion samplers across linear and nonlinear inverse problems on the standard benchmarks. We further demonstrate the benefits of our approach in training-free stylization and text-guided editing.

---

## 76. BioX-Bridge: Model Bridging for Unsupervised Cross-Modal Knowledge Transfer across Biosignals

**论文链接:** [http://arxiv.org/abs/2510.02276v1](http://arxiv.org/abs/2510.02276v1)

**作者:** Chenqi Li, Yu Liu, Timothy Denison, Tingting Zhu

**发布时间:** 2025-10-02

### GPT解析

### 总结

该研究提出了一种名为BioX-Bridge的新框架，用于生物信号的无监督跨模态知识转移，通过训练轻量级桥接网络对齐不同模态生物信号的中间表示，实现信息流动。

### 背景

生物信号能提供人体生理状态的宝贵见解，不同模态信号相互关联但功能各异。现有方法基于知识蒸馏，需要同时运行教师模型，计算和内存开销大，且基础模型虽性能优越但尺寸大。

### 目的

解决生物信号跨模态知识转移中的高计算开销问题，开发一种高效的无监督跨模态知识转移方法，减少参数数量同时保持或提高性能。

### 方法

训练轻量级桥接网络对齐中间表示，促进基础模型间和跨模态信息流动；引入高效策略选择桥接位置；采用灵活的原型网络作为桥接架构。

### 主要发现

BioX-Bridge在多个生物信号模态、任务和数据集上实验表明，可将可训练参数数量减少88-99%，同时与最先进方法相比保持或提高了转移性能。

### 结论

BioX-Bridge框架有效解决了生物信号跨模态知识转移中的计算效率问题，显著减少了参数数量同时保持了或提高了性能，为健康监测系统提供了更高效、更可访问的解决方案。

### 翻译

生物信号为人体生理状态提供了宝贵的洞察。尽管生物信号模态在功能、信号保真度、传感器舒适度和成本方面有所不同，但它们通常是相互关联的，反映了人体生理的整体性和互联性。这使得使用替代的生物信号模态执行相同任务成为可能，从而提高健康监测系统的可访问性、可用性和适应性。然而，针对特定任务和感兴趣的模态训练模型的挑战在于标记数据集的有限可用性。无监督跨模态知识转移提供了一种有前途的解决方案，它利用现有模态的知识来支持新模态的模型训练。现有方法通常基于知识蒸馏，这需要 alongside 学生模型训练运行教师模型，导致高计算和内存开销。这个挑战随着最近基础模型的发展而进一步加剧，这些模型在跨任务方面表现出卓越的性能和泛化能力，但代价是模型尺寸大。为此，作者探索了一种新的生物信号无监督跨模态知识转移框架，通过训练轻量级桥接网络来对齐中间表示，并促进基础模型之间和跨模态的信息流动。具体来说，他们引入了一种选择桥接位置的策略，以及一个灵活的原型网络作为桥接架构。在多个生物信号模态、任务和数据集上的广泛实验表明，BioX-Bridge将可训练参数数量减少了88-99%，同时与最先进的方法相比保持了甚至提高了转移性能。


### 论文摘要

Biosignals offer valuable insights into the physiological states of the human body. Although biosignal modalities differ in functionality, signal fidelity, sensor comfort, and cost, they are often intercorrelated, reflecting the holistic and interconnected nature of human physiology. This opens up the possibility of performing the same tasks using alternative biosignal modalities, thereby improving the accessibility, usability, and adaptability of health monitoring systems. However, the limited availability of large labeled datasets presents challenges for training models tailored to specific tasks and modalities of interest. Unsupervised cross-modal knowledge transfer offers a promising solution by leveraging knowledge from an existing modality to support model training for a new modality. Existing methods are typically based on knowledge distillation, which requires running a teacher model alongside student model training, resulting in high computational and memory overhead. This challenge is further exacerbated by the recent development of foundation models that demonstrate superior performance and generalization across tasks at the cost of large model sizes. To this end, we explore a new framework for unsupervised cross-modal knowledge transfer of biosignals by training a lightweight bridge network to align the intermediate representations and enable information flow between foundation models and across modalities. Specifically, we introduce an efficient strategy for selecting alignment positions where the bridge should be constructed, along with a flexible prototype network as the bridge architecture. Extensive experiments across multiple biosignal modalities, tasks, and datasets show that BioX-Bridge reduces the number of trainable parameters by 88--99\% while maintaining or even improving transfer performance compared to state-of-the-art methods.

---

## 77. Efficiently Generating Correlated Sample Paths from Multi-step Time Series Foundation Models

**论文链接:** [http://arxiv.org/abs/2510.02224v1](http://arxiv.org/abs/2510.02224v1)

**作者:** Ethan Baron, Boris Oreshkin, Ruijun Ma, Hanyu Zhang, Kari Torkkola, Michael W. Mahoney, Andrew Gordon Wilson, Tatiana Konstantinova

**发布时间:** 2025-10-02

### GPT解析

### 总结

这项研究提出了一种基于copula的方法，用于从现有的多步时间序列基础模型中高效生成准确的相关样本路径，只需一次前向传递。该方法比自回归采样快几个数量级，并通过减轻误差累积现象提高了样本路径质量。

### 背景

时间序列应用通常需要访问样本路径形式的多步预测轨迹。最近，时间序列基础模型利用多步前向预测来提高多步预测的质量和效率。然而，这些模型只预测每个时间步的独立边际分布，而不是完整的联合预测分布。

### 目的

为了解决现有方法在生成具有相关结构的预测样本路径时效率低下的问题，提出一种比自回归采样更快且质量更高的方法。

### 方法

开发了一种基于copula的方法，可以从现有的多步时间序列基础模型中高效生成准确的相关样本路径，只需一次前向传递。

### 主要发现

copula方法比自回归采样快几个数量级，并且通过减轻误差累积现象提高了样本路径质量。

### 结论

基于copula的方法是生成相关时间序列预测样本路径的高效替代方案，在速度和质量上都优于传统的自回归采样方法。

### 翻译

许多时间序列应用需要访问样本路径形式的多步预测轨迹。最近，时间序列基础模型利用多步前向预测来提高多步预测的质量和效率。然而，这些模型只预测每个时间步的独立边际分布，而不是完整的联合预测分布。为了生成具有相关结构的预测样本路径，通常采用自回归采样，但这可能非常昂贵。在本文中，我们提出了一种基于copula的方法，可以从现有的多步时间序列基础模型中高效生成准确的相关样本路径，只需一次前向传递。我们的copula方法比自回归采样快几个数量级，并且通过减轻误差累积现象提高了样本路径质量。


### 论文摘要

Many time series applications require access to multi-step forecast trajectories in the form of sample paths. Recently, time series foundation models have leveraged multi-step lookahead predictions to improve the quality and efficiency of multi-step forecasts. However, these models only predict independent marginal distributions for each time step, rather than a full joint predictive distribution. To generate forecast sample paths with realistic correlation structures, one typically resorts to autoregressive sampling, which can be extremely expensive. In this paper, we present a copula-based approach to efficiently generate accurate, correlated sample paths from existing multi-step time series foundation models in one forward pass. Our copula-based approach generates correlated sample paths orders of magnitude faster than autoregressive sampling, and it yields improved sample path quality by mitigating the snowballing error phenomenon.

---

## 78. FRIEREN: Federated Learning with Vision-Language Regularization for Segmentation

**论文链接:** [http://arxiv.org/abs/2510.02114v1](http://arxiv.org/abs/2510.02114v1)

**作者:** Ding-Ruei Shen

**发布时间:** 2025-10-02

**备注:** Master Thesis

### GPT解析

### 总结

本研究提出了FFREEDG任务和FRIEREN框架，解决联邦学习在语义分割中面临的域偏移挑战，特别是在客户端仅有未标记数据的情况下，通过整合视觉和语言模态来提升性能。

### 背景

联邦学习为语义分割任务提供隐私保护解决方案，但面临域偏移挑战，尤其是客户端数据未标记时。现有方法假设可访问客户端标记数据或未能充分利用现代视觉基础模型。

### 目的

解决FFREEDG任务：模型在服务器标记源数据集预训练后，仅使用客户端未标记数据在客户端间训练，且不再重新访问源数据。

### 方法

提出FRIEREN框架，利用视觉基础模型知识，整合视觉和语言模态。采用由CLIP文本嵌入引导的视觉语言解码器改善语义消歧，并使用弱到强一致性学习策略对伪标签进行鲁棒局部训练。

### 主要发现

在合成到真实和清晰到恶劣天气的基准测试中，该框架有效解决了新任务，与现有领域泛化和适应方法相比取得有竞争力性能，为未来研究设定了强基线。

### 结论

该研究解决了联邦学习中的重要挑战，提出的方法有效利用视觉基础模型知识，为处理未标记数据场景下的语义分割提供了创新解决方案。

### 翻译

联邦学习(FL)为语义分割(SS)任务提供了一种隐私保护解决方案，可以适应新领域，但当客户端数据未标记时，面临着来自这些域偏移的显著挑战。然而，大多数现有的联邦学习方法不切实际地假设可以访问远程客户端的标记数据，或者未能利用现代视觉基础模型(VFMs)的力量。在这里，我们提出了一个新颖且具有挑战性的任务FFREEDG，其中模型在服务器的标记源数据集上预训练，随后仅使用客户端的未标记数据在客户端间进行训练，且不再重新访问源数据。为了解决FFREEDG，我们提出了FRIEREN框架，该框架通过整合视觉和语言模态来利用视觉基础模型的知识。我们的方法采用由基于CLIP的文本嵌入引导的视觉语言解码器来改善语义消歧，并使用弱到强一致性学习策略对伪标签进行鲁棒局部训练。我们在合成到真实和清晰到恶劣天气的基准测试中的实验表明，我们的框架有效地解决了这一新任务，与现有的领域泛化和适应方法相比取得了有竞争力的性能，并为未来研究设定了强有力的基线。


### 论文摘要

Federeated Learning (FL) offers a privacy-preserving solution for Semantic Segmentation (SS) tasks to adapt to new domains, but faces significant challenges from these domain shifts, particularly when client data is unlabeled. However, most existing FL methods unrealistically assume access to labeled data on remote clients or fail to leverage the power of modern Vision Foundation Models (VFMs). Here, we propose a novel and challenging task, FFREEDG, in which a model is pretrained on a server's labeled source dataset and subsequently trained across clients using only their unlabeled data, without ever re-accessing the source. To solve FFREEDG, we propose FRIEREN, a framework that leverages the knowledge of a VFM by integrating vision and language modalities. Our approach employs a Vision-Language decoder guided by CLIP-based text embeddings to improve semantic disambiguation and uses a weak-to-strong consistency learning strategy for robust local training on pseudo-labels. Our experiments on synthetic-to-real and clear-to-adverse-weather benchmarks demonstrate that our framework effectively tackles this new task, achieving competitive performance against established domain generalization and adaptation methods and setting a strong baseline for future research.

---

## 79. KAIROS: Unified Training for Universal Non-Autoregressive Time Series Forecasting

**论文链接:** [http://arxiv.org/abs/2510.02084v1](http://arxiv.org/abs/2510.02084v1)

**作者:** Kuiye Ding, Fanda Fan, Zheya Wang, Hongxiao Li, Yifan Wang, Lei Wang, Chunjie Luo, Jianfeng Zhan

**发布时间:** 2025-10-02

### GPT解析

### 总结

KAIROS是一种非自回归时间序列预测框架，直接对段级多峰分布进行建模，避免了误差累积并实现了及时推理，在保持高性能的同时大幅降低了推理成本。

### 背景

在万维网中，可靠的时间序列预测为资源规划、缓存放置和异常响应提供前瞻性信号，使平台能够随着用户行为和内容分布的演变而高效运行。与其他领域相比，Web应用的时间序列预测需要更快的响应速度以支持实时决策。

### 目的

开发一种能够快速响应的时间序列预测框架，以满足Web应用实时决策的需求，同时避免自回归方法的误差累积问题。

### 方法

提出KAIROS框架，一种非自回归时间序列预测方法，直接对段级多峰分布进行建模，避免了自回归方法的误差累积，实现了及时推理，并改进了现有非自回归模型过度平滑的问题。

### 主要发现

在大规模语料库上训练的KAIROS在六个广泛使用的基准测试上展示了强大的零样本泛化能力，其预测性能可与规模相当的最先进基础模型相媲美，但推理成本仅为这些模型的一小部分。

### 结论

非自回归设计可作为基础模型在时间序列领域的一种重要可扩展范式，KAIROS框架证明了这一设计在保持高性能的同时显著降低计算成本的有效性。

### 翻译

在万维网中，可靠的时间序列预测为资源规划、缓存放置和异常响应提供前瞻性信号，使平台能够随着用户行为和内容分布的演变而高效运行。与其他领域相比，Web应用的时间序列预测需要更快的响应速度以支持实时决策。我们提出了KAIROS，一种非自回归时间序列预测框架，直接对段级多峰分布进行建模。与自回归方法不同，KAIROS避免了误差累积并实现了及时推理，同时改进了现有的会退化为过度平滑预测的非自回归模型。在大规模语料库上训练后，KAIROS在六个广泛使用的基准测试上展示了强大的零样本泛化能力，以远低于这些模型推理成本的价格，提供了与规模相当的最先进基础模型相当的预测性能。除了实证结果外，KAIROS还强调了非自回归设计作为基础模型在时间序列中可扩展范式的重要性。


### 论文摘要

In the World Wide Web, reliable time series forecasts provide the forward-looking signals that drive resource planning, cache placement, and anomaly response, enabling platforms to operate efficiently as user behavior and content distributions evolve. Compared with other domains, time series forecasting for Web applications requires much faster responsiveness to support real-time decision making. We present KAIROS, a non-autoregressive time series forecasting framework that directly models segment-level multi-peak distributions. Unlike autoregressive approaches, KAIROS avoids error accumulation and achieves just-in-time inference, while improving over existing non-autoregressive models that collapse to over-smoothed predictions. Trained on the large-scale corpus, KAIROS demonstrates strong zero-shot generalization on six widely used benchmarks, delivering forecasting performance comparable to state-of-the-art foundation models with similar scale, at a fraction of their inference cost. Beyond empirical results, KAIROS highlights the importance of non-autoregressive design as a scalable paradigm for foundation models in time series.

---

## 80. Multimodal Foundation Models for Early Disease Detection

**论文链接:** [http://arxiv.org/abs/2510.01899v1](http://arxiv.org/abs/2510.01899v1)

**作者:** Md Talha Mohsin, Ismail Abdulrashid

**发布时间:** 2025-10-02

**备注:** 6 pages

### GPT解析

### 总结

本研究提出了一种多模态基础模型，通过基于注意力机制的transformer框架整合多样化医疗数据，用于早期疾病诊断。

### 背景

医疗健康领域产生多种数据流（电子健康记录、医学影像、基因数据和可穿戴设备监测数据），传统诊断模型通常单独分析这些数据源，限制了识别跨模态关联的能力，而这些关联对早期疾病诊断至关重要。

### 目的

提出一个多模态基础模型，整合多样化患者数据，用于提高早期疾病诊断的准确性。

### 方法

提出一个基于注意力机制的transformer框架的多模态基础模型。该模型首先通过专用编码器将每种模态数据转换到共享的潜在空间，然后使用多头注意力和残差归一化进行组合。该架构支持在多项任务上进行预训练，使其能够轻松适应新的疾病和数据集。研究还提供了一种实验策略，使用肿瘤学、心脏病学和神经病学基准数据集测试早期检测任务。

### 主要发现

该框架包括数据治理和模型管理工具，提高了透明度、可靠性和临床可解释性。

### 结论

所提出的方法旨在为精准诊断建立单一基础模型，可以提高预测准确性并帮助医生做出决策。

### 翻译

医疗健康领域产生多样化的数据流，包括电子健康记录、医学影像、基因数据以及来自可穿戴设备的持续监测数据。传统的诊断模型经常单独分析这些数据源，这限制了它们识别跨模态关联的能力，而这些关联对早期疾病诊断至关重要。我们的研究提出了一个多模态基础模型，通过基于注意力的transformer框架整合多样化的患者数据。首先，专用编码器将每种模态数据转换到共享的潜在空间。然后，使用多头注意力和残差归一化将它们组合。该架构设计用于在多项任务上进行预训练，使其能够轻松适应新的疾病和数据集，只需额外少量工作。我们提供了一种实验策略，使用肿瘤学、心脏病学和神经病学基准数据集，以测试早期检测任务。除了技术性能外，该框架还包括数据治理和模型管理工具，以提高透明度、可靠性和临床可解释性。所提出的方法旨在为精准诊断建立一个单一基础模型，这可以提高预测准确性并帮助医生做出决策。


### 论文摘要

Healthcare generates diverse streams of data, including electronic health records (EHR), medical imaging, genetics, and ongoing monitoring from wearable devices. Traditional diagnostic models frequently analyze these sources in isolation, which constrains their capacity to identify cross-modal correlations essential for early disease diagnosis. Our research presents a multimodal foundation model that consolidates diverse patient data through an attention-based transformer framework. At first, dedicated encoders put each modality into a shared latent space. Then, they combine them using multi-head attention and residual normalization. The architecture is made for pretraining on many tasks, which makes it easy to adapt to new diseases and datasets with little extra work. We provide an experimental strategy that uses benchmark datasets in oncology, cardiology, and neurology, with the goal of testing early detection tasks. The framework includes data governance and model management tools in addition to technological performance to improve transparency, reliability, and clinical interpretability. The suggested method works toward a single foundation model for precision diagnostics, which could improve the accuracy of predictions and help doctors make decisions.

---

## 81. AI Foundation Model for Time Series with Innovations Representation

**论文链接:** [http://arxiv.org/abs/2510.01560v1](http://arxiv.org/abs/2510.01560v1)

**作者:** Lang Tong, Xinyi Wang

**发布时间:** 2025-10-02

### GPT解析

### 总结

本文提出了一种基于创新表示的生成式预训练转换器（TS-GPT），专门用于工程应用中的时间序列分析，解决了基于大型语言模型的AI基础模型在遵循物理规律而非语言规律的时间序列数据中的局限性。

### 背景

工程时间序列由物理规律而非语言规律支配，因此大型语言模型为基础的AI基础模型在工程应用中可能效果不佳或效率不高。

### 目的

开发一种专门针对工程时间序列的AI基础模型，用于需要因果操作的实时监控和控制任务。

### 方法

基于Wiener、Kallianpur和Rosenblatt的经典创新表示理论，提出时间序列GPT（TS-GPT）——一种基于创新表示的生成式预训练转换器，并采用概率生成预测方法，从条件概率分布中生成未来时间序列样本。

### 主要发现

通过使用美国独立系统运营商的历史数据进行实时位置边际价格预测，证明了TS-GPT的有效性。

### 结论

TS-GPT作为一种专门为工程时间序列设计的基础模型，能够有效处理物理规律支配的数据，为工程监控和控制任务提供可靠的解决方案。

### 翻译

本文介绍了一种用于工程应用中时间序列的人工智能基础模型，在实时监控和控制中需要因果操作。由于工程时间序列遵循物理规律而非语言规律，基于大型语言模型的AI基础模型可能效果不佳或效率不高。基于Wiener、Kallianpur和Rosenblatt的经典创新表示理论，我们提出了时间序列GPT（TS-GPT）——一种基于创新表示的生成式预训练转换器，用于工程监控和控制。作为基础模型适应的例子，我们考虑了概率生成预测，它根据给定的过去实现，从条件概率分布中生成未来时间序列样本。我们通过使用美国独立系统运营商的历史数据进行实时位置边际价格预测，证明了TS-GPT的有效性。


### 论文摘要

This paper introduces an Artificial Intelligence (AI) foundation model for time series in engineering applications, where causal operations are required for real-time monitoring and control. Since engineering time series are governed by physical, rather than linguistic, laws, large-language-model-based AI foundation models may be ineffective or inefficient. Building on the classical innovations representation theory of Wiener, Kallianpur, and Rosenblatt, we propose Time Series GPT (TS-GPT) -- an innovations-representation-based Generative Pre-trained Transformer for engineering monitoring and control. As an example of foundation model adaptation, we consider Probabilistic Generative Forecasting, which produces future time series samples from conditional probability distributions given past realizations. We demonstrate the effectiveness of TS-GPT in forecasting real-time locational marginal prices using historical data from U.S. independent system operators.

---

## 82. Round-trip Reinforcement Learning: Self-Consistent Training for Better Chemical LLMs

**论文链接:** [http://arxiv.org/abs/2510.01527v1](http://arxiv.org/abs/2510.01527v1)

**作者:** Lecheng Kong, Xiyuan Wang, Yixin Chen, Muhan Zhang

**发布时间:** 2025-10-01

**备注:** 19 pages

### GPT解析

### 总结

本文提出了一种名为往返强化学习(RTRL)的新框架，用于提高大型语言模型在计算化学任务中的往返一致性，实验证明该方法能有效提升模型性能。

### 背景

大型语言模型正在成为计算化学的多功能基础模型，能够处理反应预测和逆合成分析等双向任务，但这些模型通常缺乏往返一致性，即无法从自身生成的文本中准确重建原始分子结构。

### 目的

将往返一致性重新定义为模型改进的直接目标，开发一种能够提高模型一致性的训练方法。

### 方法

提出往返强化学习(RTRL)框架，使用往返变换的成功作为奖励信号来训练模型；进一步提出迭代变体，让前向和反向映射在自我改进循环中交替训练，这种方法对数据效率很高，特别适用于化学中常见的海量未标记数据。

### 主要发现

实验表明，RTRL在监督、自监督和合成数据方案中都显著提高了性能和一致性，证明往返一致性不仅是一个理想属性，而且是一个可训练的目标。

### 结论

往返一致性为构建更强大可靠的基础模型提供了一条新路径，通过将一致性作为训练目标可以显著提升模型表现。

### 翻译

大型语言模型(LLMs)正在成为计算化学的多功能基础模型，能够处理反应预测和逆合成分析等双向任务。然而，这些模型通常缺乏往返一致性。例如，最先进的化学LLM可能能够成功描述一个分子，但无法从其自身生成的文本中准确重建原始结构。这种不一致性表明模型正在学习单向记忆而非灵活掌握。确实，最近的研究已经证明模型的往返一致性与其主要任务表现之间存在强相关性。这种强相关性将一致性重新定义为模型改进的直接目标。因此，我们引入了往返强化学习(RTRL)，这是一个新框架，通过使用往返变换的成功作为奖励信号来训练模型以提高其一致性。我们进一步提出了一种迭代变体，其中前向和反向映射在自我改进循环中交替训练，这个过程对数据效率很高，并且在化学中常见的海量未标记数据上特别有效。实验证明，RTRL在监督、自监督和合成数据方案中都显著提高了性能和一致性。这项工作表明，往返一致性不仅是一个理想属性，而且是一个可训练的目标，为构建更强大可靠的基础模型提供了新路径。


### 论文摘要

Large Language Models (LLMs) are emerging as versatile foundation models for computational chemistry, handling bidirectional tasks like reaction prediction and retrosynthesis. However, these models often lack round-trip consistency. For instance, a state-of-the-art chemical LLM may successfully caption a molecule, yet be unable to accurately reconstruct the original structure from its own generated text. This inconsistency suggests that models are learning unidirectional memorization rather than flexible mastery. Indeed, recent work has demonstrated a strong correlation between a model's round-trip consistency and its performance on the primary tasks. This strong correlation reframes consistency into a direct target for model improvement. We therefore introduce Round-Trip Reinforcement Learning (RTRL), a novel framework that trains a model to improve its consistency by using the success of a round-trip transformation as a reward signal. We further propose an iterative variant where forward and reverse mappings alternately train each other in a self-improvement loop, a process that is highly data-efficient and notably effective with the massive amount of unlabelled data common in chemistry. Experiments demonstrate that RTRL significantly \textbf{boosts performance and consistency} over strong baselines across supervised, self-supervised, and synthetic data regimes. This work shows that round-trip consistency is not just a desirable property but a trainable objective, offering a new path toward more robust and reliable foundation models.

---

## 83. CarbonX: An Open-Source Tool for Computational Decarbonization Using Time Series Foundation Models

**论文链接:** [http://arxiv.org/abs/2510.01521v1](http://arxiv.org/abs/2510.01521v1)

**作者:** Diptyaroop Maji, Kang Yang, Prashant Shenoy, Ramesh K Sitaraman, Mani Srivastava

**发布时间:** 2025-10-01

### GPT解析

### 总结

本文介绍了CarbonX，一个利用时间序列基础模型(TSFMs)进行多种脱碳任务的开源工具，能够仅使用历史碳强度数据在全球范围内提供准确的碳强度预测，克服了现有工具的局限性。

### 背景

计算脱碳旨在减少计算系统和社会系统（如数据中心、交通和建筑环境）的碳排放，但现有工具存在三大局限性：需要特定电网的电力组合数据、依赖单独的特定电网模型难以提供全球覆盖、以及提供的预测没有不确定性估计。

### 目的

开发一个能够克服现有工具局限性的碳强度预测工具，实现全球范围内的准确碳强度预测，为下游碳感知应用提供可靠支持。

### 方法

提出CarbonX工具，利用时间序列基础模型(TSFMs)的多功能性，仅使用历史碳强度数据和单个通用模型进行碳强度预测和插补任务。

### 主要发现

CarbonX在全球214个电网中实现了15.82%的零样本预测平均绝对百分比误差；在13个基准电网上的性能与当前最先进技术相当，平均MAPE为9.59%，尾部预测MAPE为16.54%，并提供95%覆盖率的预测区间；可提供长达21天的预测，精度下降最小；完全微调后，在插补任务上优于统计基线1.2-3.9倍。

### 结论

CarbonX可以在有限数据的情况下轻松应用于任何电网并仍能提供强大性能，是全球规模脱碳的实用工具。

### 翻译

计算脱碳旨在减少计算系统和社会系统（如数据中心、交通和建筑环境）中的碳排放。这需要准确、细粒度的碳强度预测，然而现有工具有几个关键局限性：(i)它们需要特定电网的电力组合数据，限制了在无法获取此类信息的地方的使用；(ii)它们依赖于单独的特定电网模型，难以提供全球覆盖；以及(iii)它们提供的预测没有不确定性估计，限制了下游碳感知应用的可靠性。在本文中，我们提出了CarbonX，一个利用时间序列基础模型(TSFMs)进行多种脱碳任务的开源工具。CarbonX利用TSFMs的多功能性，在多种任务（如碳强度预测和插补）和不同电网间提供强大性能。仅使用历史碳强度数据和单个通用模型，我们的工具在全球214个电网中实现了15.82%的零样本预测平均绝对百分比误差。在13个基准电网中，CarbonX的性能与当前最先进技术相当，平均MAPE为9.59%，尾部预测MAPE为16.54%，同时还提供95%覆盖率的预测区间。CarbonX可以提供长达21天的预测，精度下降最小。此外，当完全微调后，CarbonX在插补任务上优于统计基线1.2-3.9倍。总体而言，这些结果表明CarbonX可以在有限数据的情况下轻松应用于任何电网并仍能提供强大性能，是全球规模脱碳的实用工具。


### 论文摘要

Computational decarbonization aims to reduce carbon emissions in computing and societal systems such as data centers, transportation, and built environments. This requires accurate, fine-grained carbon intensity forecasts, yet existing tools have several key limitations: (i) they require grid-specific electricity mix data, restricting use where such information is unavailable; (ii) they depend on separate grid-specific models that make it challenging to provide global coverage; and (iii) they provide forecasts without uncertainty estimates, limiting reliability for downstream carbon-aware applications.   In this paper, we present CarbonX, an open-source tool that leverages Time Series Foundation Models (TSFMs) for a range of decarbonization tasks. CarbonX utilizes the versatility of TSFMs to provide strong performance across multiple tasks, such as carbon intensity forecasting and imputation, and across diverse grids. Using only historical carbon intensity data and a single general model, our tool achieves a zero-shot forecasting Mean Absolute Percentage Error (MAPE) of 15.82% across 214 grids worldwide. Across 13 benchmark grids, CarbonX performance is comparable with the current state-of-the-art, with an average MAPE of 9.59% and tail forecasting MAPE of 16.54%, while also providing prediction intervals with 95% coverage. CarbonX can provide forecasts for up to 21 days with minimal accuracy degradation. Further, when fully fine-tuned, CarbonX outperforms the statistical baselines by 1.2--3.9X on the imputation task. Overall, these results demonstrate that CarbonX can be used easily on any grid with limited data and still deliver strong performance, making it a practical tool for global-scale decarbonization.

---

## 84. Flock: A Knowledge Graph Foundation Model via Learning on Random Walks

**论文链接:** [http://arxiv.org/abs/2510.01510v1](http://arxiv.org/abs/2510.01510v1)

**作者:** Jinwoo Kim, Xingyue Huang, Krzysztof Olejniczak, Kyungbin Min, Michael Bronstein, Seunghoon Hong, İsmail İlkan Ceylan

**发布时间:** 2025-10-01

### GPT解析

### 总结

论文提出了Flock，一种基于概率节点-关系等变性的知识图谱基础模型，用于解决零次链接预测问题，在多个知识图谱上实现了最先进的性能。

### 背景

知识图谱基础模型(KGFMs)通过强制节点和关系的等变性来解决零次链接预测问题，但传统确定性等变性限制了模型的表达能力，无法区分结构相似但语义不同的关系。

### 目的

克服传统确定性等变性的内在限制，引入概率节点-关系等变性，提高知识图谱基础模型的表达能力，使其能够区分结构相似但语义不同的关系。

### 方法

提出Flock模型，它迭代采样随机游走，通过记录协议将它们编码为序列，使用序列模型嵌入，并通过学习的池化聚合节点和关系的表示。Flock尊重概率节点-关系等变性，是知识图谱上同构不变链接级函数的通用逼近器。

### 主要发现

Flock完美解决了当前KGFMs失败的新诊断数据集Petals，并在54个不同领域的知识图谱上的实体和关系预测任务上实现了最先进的性能。

### 结论

概率节点-关系等变性可以有效克服传统确定性等变性的限制，提高知识图谱基础模型的表达能力和泛化能力。

### 翻译

我们研究了知识图谱上的零次链接预测问题，这要求模型能够泛化到新的实体和新的关系。知识图谱基础模型通过强制节点和关系的等变性来解决这一任务，学习节点和关系的结构特性，然后将其转移到具有相似结构特性的新图谱中。然而，传统确定性等变性的概念对知识图谱基础模型的表达能力有内在限制，使它们无法区分结构相似但语义不同的关系。为了克服这一限制，我们引入了概率节点-关系等变性，在保持分布等变性的同时，引入了合理的随机化来打破推理过程中的对称性。基于这一原理，我们提出了Flock，一种知识图谱基础模型，它通过迭代采样随机游走，通过记录协议将它们编码为序列，使用序列模型嵌入它们，并通过学习的池化聚合节点和关系的表示。关键是，Flock尊重概率节点-关系等变性，并且是知识图谱上同构不变链接级函数的通用逼近器。实验上，Flock完美解决了当前知识图谱基础模型失败的新诊断数据集Petals，并在54个不同领域的知识图谱上的实体和关系预测任务上实现了最先进的性能。


### 论文摘要

We study the problem of zero-shot link prediction on knowledge graphs (KGs), which requires models to generalize over novel entities and novel relations. Knowledge graph foundation models (KGFMs) address this task by enforcing equivariance over both nodes and relations, learning from structural properties of nodes and relations, which are then transferable to novel graphs with similar structural properties. However, the conventional notion of deterministic equivariance imposes inherent limits on the expressive power of KGFMs, preventing them from distinguishing structurally similar but semantically distinct relations. To overcome this limitation, we introduce probabilistic node-relation equivariance, which preserves equivariance in distribution while incorporating a principled randomization to break symmetries during inference. Building on this principle, we present Flock, a KGFM that iteratively samples random walks, encodes them into sequences via a recording protocol, embeds them with a sequence model, and aggregates representations of nodes and relations via learned pooling. Crucially, Flock respects probabilistic node-relation equivariance and is a universal approximator for isomorphism-invariant link-level functions over KGs. Empirically, Flock perfectly solves our new diagnostic dataset Petals where current KGFMs fail, and achieves state-of-the-art performances on entity- and relation prediction tasks on 54 KGs from diverse domains.

---

## 85. BioVERSE: Representation Alignment of Biomedical Modalities to LLMs for Multi-Modal Reasoning

**论文链接:** [http://arxiv.org/abs/2510.01428v1](http://arxiv.org/abs/2510.01428v1)

**作者:** Ching-Huei Tsou, Michal Ozery-Flato, Ella Barkan, Diwakar Mahajan, Ben Shapira

**发布时间:** 2025-10-01

### GPT解析

### 总结

BIOVERSE是一种两阶段方法，通过预训练的生物医学基础模型作为模态编码器，并使用轻量级的模态特定投影层将它们与大型语言模型对齐，实现跨模态生物医学推理。

### 背景

大型语言模型和生物医学基础模型在生物文本推理、分子建模和单细胞分析方面取得了显著成果，但它们仍然被隔离在分离的嵌入空间中，限制了跨模态推理能力。

### 目的

将生物医学数据与嵌入在大型语言模型中的知识统一起来，实现零样本标注、跨模态问答和交互式可解释对话。

### 方法

BIOVERSE采用两阶段方法：首先将每个模态通过独立训练的投影对齐到共享的大型语言模型空间，使它们能够自然互操作；然后使用多模态数据应用标准指令调优，将它们结合用于下游推理。

### 主要发现

在细胞类型注释、分子描述和蛋白质功能推理等跨任务中，紧凑的BIOVERSE配置超过了更大的大型语言模型基线，同时能够生成比现有生物医学基础模型更丰富、更具生成性的输出。

### 结论

BIOVERSE为有原则的多模态生物医学推理奠定了基础，通过统一原始生物医学数据与大型语言模型中嵌入的知识，实现了跨模态推理能力。

### 翻译

大型语言模型和生物医学基础模型的最新进展已在生物文本推理、分子建模和单细胞分析方面取得了显著成果，但它们仍然被隔离在分离的嵌入空间中，限制了跨模态推理能力。我们提出了BIOVERSE（生物医学向量嵌入重新对齐以实现语义参与），这是一种两阶段方法，它将预训练的生物医学基础模型作为模态编码器，并通过轻量级的模态特定投影层将它们与大型语言模型对齐。该方法首先通过独立训练的投影将每个模态对齐到共享的大型语言模型空间，使它们能够自然互操作，然后使用多模态数据应用标准指令调优，将它们结合用于下游推理。通过将原始生物医学数据与嵌入在大型语言模型中的知识统一起来，该方法实现了零样本标注、跨模态问答和交互式可解释对话。在跨越细胞类型注释、分子描述和蛋白质功能推理的任务中，紧凑的BIOVERSE配置超过了更大的大型语言模型基线，同时能够生成比现有生物医学基础模型更丰富、更具生成性的输出，为有原则的多模态生物医学推理奠定了基础。


### 论文摘要

Recent advances in large language models (LLMs) and biomedical foundation models (BioFMs) have achieved strong results in biological text reasoning, molecular modeling, and single-cell analysis, yet they remain siloed in disjoint embedding spaces, limiting cross-modal reasoning. We present BIOVERSE (Biomedical Vector Embedding Realignment for Semantic Engagement), a two-stage approach that adapts pretrained BioFMs as modality encoders and aligns them with LLMs through lightweight, modality-specific projection layers. The approach first aligns each modality to a shared LLM space through independently trained projections, allowing them to interoperate naturally, and then applies standard instruction tuning with multi-modal data to bring them together for downstream reasoning. By unifying raw biomedical data with knowledge embedded in LLMs, the approach enables zero-shot annotation, cross-modal question answering, and interactive, explainable dialogue. Across tasks spanning cell-type annotation, molecular description, and protein function reasoning, compact BIOVERSE configurations surpass larger LLM baselines while enabling richer, generative outputs than existing BioFMs, establishing a foundation for principled multi-modal biomedical reasoning.

---

## 86. VENTURA: Adapting Image Diffusion Models for Unified Task Conditioned Navigation

**论文链接:** [http://arxiv.org/abs/2510.01388v1](http://arxiv.org/abs/2510.01388v1)

**作者:** Arthur Zhang, Xiangyun Meng, Luca Calliari, Dong-Ki Kim, Shayegan Omidshafiei, Joydeep Biswas, Ali Agha, Amirreza Shaban

**发布时间:** 2025-10-01

**备注:** 9 pages, 6 figures, 3 tables

### GPT解析

### 总结

VENTURA是一个视觉-语言导航系统，通过微调互联网预训练的图像扩散模型进行路径规划，将自然语言指令转化为多样化的机器人行为，在真实世界评估中表现出色。

### 背景

机器人必须适应多样化的人类指令并在非结构化的开放世界环境中安全操作。近期的视觉-语言模型(VLMs)为语言和感知提供了强大的先验知识，但由于动作空间和预训练目标的差异，这些模型难以用于导航任务，限制了其在机器人任务中的可转移性。

### 目的

解决VLMs在机器人导航中的局限性，开发一个能够处理自然语言指令并生成多样化机器人行为的系统。

### 方法

引入VENTURA，微调互联网预训练的图像扩散模型进行路径规划，在图像空间生成路径掩码(视觉计划)而非直接预测低级行动，使用轻量级行为克隆策略将这些视觉计划转化为可执行的轨迹，通过自监督跟踪模型与VLM增强的标题相结合生成的路径掩码进行监督训练，避免了手动像素级注释。

### 主要发现

在真实世界评估中，VENTURA在物体抓取、障碍物避让和地形偏好任务上优于最先进的基础模型基线，在已见和未见的场景中，成功率提高了33%，碰撞减少了54%，并且能够推广到未见过的不同任务的组合，展现出组合能力。

### 结论

VENTURA是一个有效的视觉-语言导航系统，通过微调图像扩散模型并使用路径掩码作为中间表示，系统能够将自然语言指令转化为多样化的机器人行为，并在多种任务上表现出色，具有泛化能力。

### 翻译

机器人必须适应多样化的人类指令并在非结构化的开放世界环境中安全操作。近期的视觉-语言模型(VLMs)为语言和感知提供了强大的先验知识，但由于动作空间和预训练目标的差异，这些模型难以用于导航任务，限制了其在机器人任务中的可转移性。为此，我们引入了VENTURA，一个视觉-语言导航系统，它微调了互联网预训练的图像扩散模型进行路径规划。VENTURA不在图像空间直接预测低级行动，而是生成路径掩码(即视觉计划)，捕捉细粒度的、具有上下文感知的导航行为。轻量级行为克隆策略将这些视觉计划转化为可执行的轨迹，形成一个遵循自然语言指令生成多样化机器人行为的界面。为扩展训练，我们使用自监督跟踪模型与VLM增强的标题相结合衍生的路径掩码进行监督，避免了手动像素级注释或高度工程化的数据收集设置。在广泛的真实世界评估中，VENTURA在物体抓取、障碍物避让和地形偏好任务上优于最先进的基础模型基线，在已见和未见的场景中，成功率提高了33%，碰撞减少了54%。值得注意的是，我们发现VENTURA能够推广到未见过的不同任务的组合，展现出组合能力。视频、代码和额外材料：https://venturapath.github.io

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': "这篇论文主要解决机器人如何理解和执行多样化语言指令在非结构化环境中进行安全导航的问题。这个问题在现实中非常重要，因为机器人需要在建筑检查、城市维护和配送等场景中适应不断变化的人类偏好和环境上下文，而现有视觉-语言导航系统难以将复杂语言指令（如'保持与儿童的安全距离'）精确转化为机器人运动，限制了机器人在开放世界中的适应性和安全性。", '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者通过分析现有视觉-语言模型的局限性，认识到需要一种能将互联网规模先验知识转化为精确导航规划的方法。他们借鉴了多个现有工作：利用预训练图像扩散模型（如Stable Diffusion）的强大视觉生成能力；采用现成的点跟踪技术（Co-Tracker）自动生成路径标签；使用轻量级行为克隆策略将视觉计划转换为动作；并借鉴了图像扩散模型中的分类器自由引导训练方法。作者设计了两阶段训练流程，首先训练扩散模型生成路径掩码，然后训练策略网络将其转换为可执行动作。', '这个方法的核心思想是什么？整体实现流程是怎样的？': 'VENTURA的核心思想是将导航规划问题重新表述为图像生成问题，利用预训练扩散模型根据语言指令生成路径掩码（视觉计划），再通过轻量级策略转换为机器人动作。整体流程：1)接收相机图像和语言指令；2)使用预训练编码器提取特征；3)扩散模型逐步去噪生成路径掩码；4)ResNet-34结合当前观测和路径掩码预测xyz路径点；5)转换为机器人可执行动作。训练分两阶段：先训练扩散模型生成路径掩码，再训练策略网络将掩码转换为动作。数据收集方面，使用点跟踪技术从演示视频中自动生成路径标签，并用VLM生成多样化语言描述增强数据集。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)将互联网预训练的图像扩散模型首次用于多任务路径规划；2)引入路径掩码作为视觉计划表示，捕获细粒度导航行为；3)设计可扩展的自动标注管道，从非结构化演示中生成高质量路径标签；4)采用混合训练策略，结合任务无关和任务导向演示。相比之前工作，VENTURA能处理更复杂语言指令而非仅定位目标；生成更精细视觉计划而非粗略规划；不需要机器人里程计监督，更具可扩展性；直接在图像空间规划完整轨迹而非生成子目标或直接合成动作，支持多样化语言条件任务。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'VENTURA创新性地将互联网预训练的图像扩散模型适应用于机器人导航，通过生成语言条件化的视觉计划并转换为精确动作，显著提升了机器人在开放环境中理解和执行多样化语言指令的能力。'}


### 论文摘要

Robots must adapt to diverse human instructions and operate safely in unstructured, open-world environments. Recent Vision-Language models (VLMs) offer strong priors for grounding language and perception, but remain difficult to steer for navigation due to differences in action spaces and pretraining objectives that hamper transferability to robotics tasks. Towards addressing this, we introduce VENTURA, a vision-language navigation system that finetunes internet-pretrained image diffusion models for path planning. Instead of directly predicting low-level actions, VENTURA generates a path mask (i.e. a visual plan) in image space that captures fine-grained, context-aware navigation behaviors. A lightweight behavior-cloning policy grounds these visual plans into executable trajectories, yielding an interface that follows natural language instructions to generate diverse robot behaviors. To scale training, we supervise on path masks derived from self-supervised tracking models paired with VLM-augmented captions, avoiding manual pixel-level annotation or highly engineered data collection setups. In extensive real-world evaluations, VENTURA outperforms state-of-the-art foundation model baselines on object reaching, obstacle avoidance, and terrain preference tasks, improving success rates by 33% and reducing collisions by 54% across both seen and unseen scenarios. Notably, we find that VENTURA generalizes to unseen combinations of distinct tasks, revealing emergent compositional capabilities. Videos, code, and additional materials: https://venturapath.github.io

---

## 87. SPUS: A Lightweight and Parameter-Efficient Foundation Model for PDEs

**论文链接:** [http://arxiv.org/abs/2510.01370v1](http://arxiv.org/abs/2510.01370v1)

**作者:** Abu Bucker Siddik, Diane Oyen, Alexander Most, Michal Kucer, Ayan Biswas

**发布时间:** 2025-10-01

### GPT解析

### 总结

介绍了一种名为Small PDE U-Net Solver (SPUS)的紧凑高效基础模型，作为统一的神经算子用于求解广泛的偏微分方程。相比基于大型Transformer架构的现有方法，SPUS采用轻量级残差U-Net架构，并通过自回归预训练策略学习底层物理规律。

### 背景

现有的PDE基础模型主要基于大型复杂Transformer架构，具有高计算和参数开销。残差U-Net架构作为基础模型架构在PDE求解领域尚未得到充分探索。

### 目的

开发一种紧凑高效的神经算子基础模型，用于求解广泛的偏微分方程，同时减少参数需求和计算开销。

### 方法

提出基于残差U-Net架构的轻量级SPUS模型；使用简单而强大的自回归预训练策略模拟数值求解器行为；在多样化流体动力学PDEs上预训练；在6个具有挑战性的未见过的下游PDEs上评估。

### 主要发现

SPUS在下游任务上实现了最先进的泛化能力；需要显著更少的参数；只需要最少的微调数据；作为高度参数效率的PDE求解基础模型具有巨大潜力。

### 结论

轻量级残差U-Net架构可作为PDE求解的基础模型，在保持高性能的同时显著减少参数需求和计算开销，为PDE求解提供了高效的新方法。

### 翻译

我们引入了小型PDE U-Net求解器（SPUS），这是一种紧凑高效的基础模型（FM），被设计为用于求解广泛偏微分方程（PDE）的统一神经算子。与现有的最先进的PDE基础模型（主要基于具有高计算和参数开销的大型复杂Transformer架构）不同，SPUS利用了一种轻量级的残差U-Net架构，该架构在这一领域作为基础模型架构在很大程度上尚未得到探索。为了在这个最小化框架中实现有效学习，我们使用了一种简单而强大的自回归预训练策略，该策略紧密复制了数值求解器的行为以学习底层物理规律。SPUS在多样化的流体动力学PDE集上进行了预训练，并在跨越各种物理系统的6个具有挑战性的未见过的下游PDE上进行了评估。实验结果表明，使用基于残差U-Net架构的SPUS在这些下游任务上实现了最先进的泛化能力，同时需要显著更少的参数和最少的微调数据，突显了其作为求解多样化PDE系统的高度参数效率的FM的潜力。


### 论文摘要

We introduce Small PDE U-Net Solver (SPUS), a compact and efficient foundation model (FM) designed as a unified neural operator for solving a wide range of partial differential equations (PDEs). Unlike existing state-of-the-art PDE FMs-primarily based on large complex transformer architectures with high computational and parameter overhead-SPUS leverages a lightweight residual U-Net-based architecture that has been largely underexplored as a foundation model architecture in this domain. To enable effective learning in this minimalist framework, we utilize a simple yet powerful auto-regressive pretraining strategy which closely replicates the behavior of numerical solvers to learn the underlying physics. SPUS is pretrained on a diverse set of fluid dynamics PDEs and evaluated across 6 challenging unseen downstream PDEs spanning various physical systems. Experimental results demonstrate that SPUS using residual U-Net based architecture achieves state-of-the-art generalization on these downstream tasks while requiring significantly fewer parameters and minimal fine-tuning data, highlighting its potential as a highly parameter-efficient FM for solving diverse PDE systems.

---

## 88. Benchmarking Foundation Models with Retrieval-Augmented Generation in Olympic-Level Physics Problem Solving

**论文链接:** [http://arxiv.org/abs/2510.00919v2](http://arxiv.org/abs/2510.00919v2)

**作者:** Shunfeng Zheng, Yudi Zhang, Meng Fang, Zihan Zhang, Zhitan Wu, Mykola Pechenizkiy, Ling Chen

**发布时间:** 2025-10-01

**备注:** Accepted to EMNLP 2025 (Findings)

### GPT解析

### 总结

本研究探讨了检索增强生成（RAG）基础模型在解决奥林匹克级别物理问题方面的能力，提出了PhoPile多模态数据集，并对多种检索增强的基础模型进行了基准测试，结果表明检索与物理语料库的集成可以提升模型性能。

### 背景

检索增强生成（RAG）与基础模型已在多种任务上展现出强大性能，但在专家级推理能力（如解决奥林匹克级别的物理问题）方面的应用仍 largely unexplored。受学生通过复习往届问题准备竞赛的启发，研究者探索了RAG增强基础模型物理推理能力的可能性。

### 目的

研究旨在探索RAG技术能否增强基础模型在奥林匹克级别物理问题上的推理能力，并系统研究基于检索的物理推理方法。

### 方法

研究者引入了PhoPile，一个专门为奥林匹克级别物理设计的高质量多模态数据集，包含图表、图形和方程，捕捉了物理问题解决的多模态本质。使用PhoPile，他们对多种检索增强的基础模型进行了基准测试，包括大型语言模型（LLMs）和大视觉语言模型（LMMs），并使用了多种检索器。

### 主要发现

研究结果表明，将检索与物理语料库集成可以提高模型在奥林匹克级别物理问题上的性能，同时也指出了需要进一步研究的挑战。

### 结论

检索增强生成技术有潜力提升基础模型在物理推理方面的能力，特别是在解决专家级别物理问题上，为未来研究提供了方向。

### 翻译

检索增强生成（RAG）与基础模型已在各种任务上取得强大性能，但它们在专家级推理方面的能力——如解决奥林匹克级别的物理问题——在很大程度上尚未被探索。受学生通过复习往届问题准备竞赛的方式启发，我们研究了RAG增强基础模型物理推理能力的潜力。我们引入了PhoPile，一个专门为奥林匹克级别物理设计的高质量多模态数据集，使基于检索的推理能够得到系统研究。PhoPile包含图表、图形和方程，捕捉了物理问题解决的内在多模态特性。使用PhoPile，我们对检索增强的基础模型进行了基准测试，涵盖了使用多种检索器的大型语言模型（LLMs）和大视觉语言模型（LMMs）。我们的结果表明，将检索与物理语料库集成可以改进模型性能，同时也指出了激励检索增强物理推理进一步研究的挑战。


### 论文摘要

Retrieval-augmented generation (RAG) with foundation models has achieved strong performance across diverse tasks, but their capacity for expert-level reasoning-such as solving Olympiad-level physics problems-remains largely unexplored. Inspired by the way students prepare for competitions by reviewing past problems, we investigate the potential of RAG to enhance physics reasoning in foundation models. We introduce PhoPile, a high-quality multimodal dataset specifically designed for Olympiad-level physics, enabling systematic study of retrieval-based reasoning. PhoPile includes diagrams, graphs, and equations, capturing the inherently multimodal nature of physics problem solving. Using PhoPile, we benchmark RAG-augmented foundation models, covering both large language models (LLMs) and large multimodal models (LMMs) with multiple retrievers. Our results demonstrate that integrating retrieval with physics corpora can improve model performance, while also highlighting challenges that motivate further research in retrieval-augmented physics reasoning.

---

## 89. MorphGen: Controllable and Morphologically Plausible Generative Cell-Imaging

**论文链接:** [http://arxiv.org/abs/2510.01298v1](http://arxiv.org/abs/2510.01298v1)

**作者:** Berker Demirel, Marco Fumero, Theofanis Karaletsos, Francesco Locatello

**发布时间:** 2025-10-01

### GPT解析

### 总结

MorphGen是一种先进的基于扩散的生成模型，用于荧光显微镜，能够在多种细胞类型和干预条件下生成高质量图像，保留细胞器特定细节，支持细粒度形态分析，并且与真实图像具有生物一致性。

### 背景

计算模拟细胞对干预的反应是加速基于高含量图像的筛选的有前途的方向，这对药物发现和基因编辑的推进至关重要。

### 目的

介绍MorphGen，一种能够在多种细胞类型和干预条件下进行可控生成的荧光显微镜生成模型。

### 方法

MorphGen使用对齐损失进行训练，使其表示与OpenPhenom的表型嵌入相匹配，以捕获有生物学意义的模式。与之前将多通道染色压缩为RGB图像的方法不同，MorphGen联合生成完整的荧光通道集，保留每个细胞器的结构。

### 主要发现

通过CellProfiler特征证明与真实图像的生物一致性，MorphGen的FID分数比之前最先进的MorphoDiff低35%以上，后者仅生成单一细胞类型的RGB图像。

### 结论

MorphGen是一种先进的生成模型，能够在多种细胞类型和干预条件下生成高质量的荧光显微镜图像，保留细胞器特定细节，支持细粒度形态分析，并且与真实图像具有生物一致性。

### 翻译

计算模拟细胞对干预的反应是加速基于高含量图像的筛选的一个有前途的方向，对推进药物发现和基因编辑至关重要。为此，我们介绍了MorphGen，这是一种最先进的基于扩散的生成模型，用于荧光显微镜，能够在多种细胞类型和干预条件下进行可控生成。为了捕获与已知细胞形态一致的有生物学意义的模式，MorphGen使用对齐损失进行训练，使其表示与OpenPhenom（一种最先进的生物基础模型）的表型嵌入相匹配。与之前将多通道染色压缩为RGB图像（从而牺牲细胞器特定细节）的方法不同，MorphGen联合生成完整的荧光通道集，保留每个细胞器的结构，并支持对生物解释至关重要的细粒度形态分析。我们通过CellProfiler特征证明了与真实图像的生物一致性，并且MorphGen的FID分数比之前最先进的MorphoDiff低35%以上，后者仅生成单一细胞类型的RGB图像。代码可在https://github.com/czi-ai/MorphGen获取。


### 论文摘要

Simulating in silico cellular responses to interventions is a promising direction to accelerate high-content image-based assays, critical for advancing drug discovery and gene editing. To support this, we introduce MorphGen, a state-of-the-art diffusion-based generative model for fluorescent microscopy that enables controllable generation across multiple cell types and perturbations. To capture biologically meaningful patterns consistent with known cellular morphologies, MorphGen is trained with an alignment loss to match its representations to the phenotypic embeddings of OpenPhenom, a state-of-the-art biological foundation model. Unlike prior approaches that compress multichannel stains into RGB images -- thus sacrificing organelle-specific detail -- MorphGen generates the complete set of fluorescent channels jointly, preserving per-organelle structures and enabling a fine-grained morphological analysis that is essential for biological interpretation. We demonstrate biological consistency with real images via CellProfiler features, and MorphGen attains an FID score over $35\%$ lower than the prior state-of-the-art MorphoDiff, which only generates RGB images for a single cell type. Code is available at https://github.com/czi-ai/MorphGen.

---

## 90. Can World Models Benefit VLMs for World Dynamics?

**论文链接:** [http://arxiv.org/abs/2510.00855v1](http://arxiv.org/abs/2510.00855v1)

**作者:** Kevin Zhang, Kuangzhi Ge, Xiaowei Chi, Renrui Zhang, Shaojun Shi, Zhen Dong, Sirui Han, Shanghang Zhang

**发布时间:** 2025-10-01

**备注:** Project page: https://dyva-worldlm.github.io

### GPT解析

### 总结

研究提出了一种新型视觉语言模型架构，称为世界语言模型（WorldLMs），通过将视频扩散模型作为生成编码器，显著提升了视觉推理能力，特别是在空间推理和多帧任务上。

### 背景

在互联网规模视频数据上训练的世界模型能够生成一致且合理的动态，这引发了它们是否可能取代传统视觉编码器范式的问题。然而，现有研究缺乏对世界模型在通用多模态任务上的系统探索。

### 目的

研究世界模型先验知识转移到视觉语言模型时的能力，探索生成编码器在下游视觉理解任务中的应用潜力。

### 方法

将视频扩散模型重新用作生成编码器，执行单步去噪，并将得到的潜变量作为视觉嵌入。提出了一种称为动态视觉对齐器（DyVA）的最佳变体，并通过一系列视觉推理任务进行评估。

### 主要发现

1. 生成编码器可以捕获对下游理解有用的潜变量，这些潜变量与传统编码器有所不同；2. DyVA方法显著增强了空间推理能力，使单图像模型能够执行多帧推理；3. DyVA超越了开源和专有基线，达到了最先进或可比的性能；4. 这些改进归因于WorldLM从视频预训练中继承的运动一致性内化。

### 结论

研究为利用世界模型先验的新一代视觉语言模型铺平了道路，这些模型朝着通用视觉学习者的方向发展，具有广阔的前景。

### 翻译

在互联网规模的视频数据上训练生成的世界模型正日益被视为强大的世界模拟器，能够在结构、运动和物理方面生成一致且合理的动态。这自然引发了一个问题：随着强大的视频基础模型的兴起，它们是否会取代传统的视觉编码器范式，用于通用多模态理解？尽管最近的研究开始探索世界模型在常见视觉任务上的潜力，但这些探索通常缺乏对通用多模态任务的系统研究。在这项工作中，我们努力研究将世界模型先验知识转移到视觉语言模型时的能力：我们将视频扩散模型重新用作生成编码器，执行单步去噪，并将得到的潜变量视为一组视觉嵌入。我们经验性地研究了这类模型，称之为世界语言模型（WorldLMs），发现生成编码器可以捕获对下游理解有用的潜变量，这些潜变量与传统编码器有所不同。我们将性能最佳的变体命名为动态视觉对齐器（DyVA），进一步发现这种方法显著增强了空间推理能力，并使单图像模型能够执行多帧推理。通过一系列视觉推理任务的整理，我们发现DyVA超越了开源和专有基线，达到了最先进或可比的性能。我们将这些改进归因于WorldLM从视频预训练中继承的运动一致性内化。最后，我们系统性地探索了广泛的模型设计，以突出未来工作的有前途方向。我们希望我们的研究能够为利用世界模型先验的新一代VLMs铺平道路，并朝着通用视觉学习者的有希望前进。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要探讨世界模型（特别是视频生成模型）能否增强视觉语言模型（VLMs）对世界动态的理解能力。这个问题很重要，因为当前主流VLMs依赖静态图像编码器（如CLIP），在处理动态场景、空间推理和时序理解方面存在局限，而世界模型通过大规模视频数据训练，能够理解和预测物体的运动、空间布局和物理规律，如果能将这种动态理解能力整合到VLMs中，将大大增强模型对现实世界动态场景的理解和推理能力。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者观察到世界模型具有强大的动态建模能力，能够生成连贯、合理的未来场景预测，思考这种生成能力是否也意味着对视觉动态的语义理解，并假设这种理解可以迁移到其他任务中。他们设计将视频扩散模型（Stable Video Diffusion）重新用作生成编码器，通过单步去噪提取动态特征，并与静态语义特征融合。该方法借鉴了VLMs的基本架构、视频扩散模型的U-Net网络以及Prismatic-VLMs的单阶段训练策略，但创新性地将生成模型作为编码器使用。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是利用世界模型的动态理解能力提取包含动态信息的视觉特征，并将其与静态语义特征融合，使VLMs既能理解静态内容，又能理解动态变化和空间关系。整体流程包括：1) 使用SigLIP提取图像语义特征，同时用SVD进行单步去噪提取动态特征；2) 将两种特征通过投影层映射到语言模型嵌入空间并融合；3) 将融合特征与文本提示拼接输入语言模型生成输出；4) 采用单阶段训练策略训练投影层和语言模型；5) 推理时处理单图像或多图像输入，融合静态和动态特征。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1) 范式转变，从静态描述转向动态预测；2) 实现零样本多帧适应，仅用单图像训练获得多帧推理能力；3) 提出动态视觉对齐器（DyVA）架构，有效融合静态和动态特征；4) 建立系统研究框架，从范式比较、基准诊断到设计空间探索全面研究。相比之前的工作，本文创新性地使用生成编码器而非静态编码器，融合动态和静态特征而非仅使用静态特征，训练效率更高（仅需10.3小时），且显著增强了空间推理和多帧理解能力。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '本文提出了一种新的World-Language Model架构，通过融合世界模型的动态特征与传统静态视觉特征，显著增强了视觉语言模型对空间关系和动态场景的理解与推理能力，实现了仅用单图像训练即可获得多帧推理的突破。'}


### 论文摘要

Trained on internet-scale video data, generative world models are increasingly recognized as powerful world simulators that can generate consistent and plausible dynamics over structure, motion, and physics. This raises a natural question: with the advent of strong video foundational models, might they supplant conventional vision encoder paradigms for general-purpose multimodal understanding? While recent studies have begun to explore the potential of world models on common vision tasks, these explorations typically lack a systematic investigation of generic, multimodal tasks. In this work, we strive to investigate the capabilities when world model priors are transferred into Vision-Language Models: we re-purpose a video diffusion model as a generative encoder to perform a single denoising step and treat the resulting latents as a set of visual embedding. We empirically investigate this class of models, which we refer to as World-Language Models (WorldLMs), and we find that generative encoders can capture latents useful for downstream understanding that show distinctions from conventional encoders. Naming our best-performing variant Dynamic Vision Aligner (DyVA), we further discover that this method significantly enhances spatial reasoning abilities and enables single-image models to perform multi-frame reasoning. Through the curation of a suite of visual reasoning tasks, we find DyVA to surpass both open-source and proprietary baselines, achieving state-of-the-art or comparable performance. We attribute these gains to WorldLM's inherited motion-consistency internalization from video pre-training. Finally, we systematically explore extensive model designs to highlight promising directions for future work. We hope our study can pave the way for a new family of VLMs that leverage priors from world models and are on a promising path towards generalist vision learners.

---

## 91. Are Time Series Foundation Models Susceptible to Catastrophic Forgetting?

**论文链接:** [http://arxiv.org/abs/2510.00809v2](http://arxiv.org/abs/2510.00809v2)

**作者:** Nouha Karaouli, Denis Coquenet, Elisa Fromont, Martial Mermillod, Marina Reyboz

**发布时间:** 2025-10-01

### GPT解析

### 总结

本研究探讨了时间序列基础模型(TSFMs)在持续适应过程中的鲁棒性问题，特别关注了灾难性遗忘现象。

### 背景

时间序列基础模型(TSFMs)已在多种预测任务中展现出有前途的零样本泛化能力，但它们对持续适应的鲁棒性尚未被充分探索。

### 目的

调查TSFMs在连续微调多个数据集时是否会遭受灾难性遗忘，并衡量适应新数据与保留先前知识之间的权衡。

### 方法

使用具有不同程度周期结构的合成数据集，通过实验测量模型在新任务上的适应能力与对先前知识的保留程度。

### 主要发现

虽然微调提高了TSFMs在新任务上的性能，但它常常导致先前学习任务的性能显著下降，体现了基本的稳定性-可塑性困境。

### 结论

TSFMs在持续适应过程中存在稳定性-可塑性困境，需要在适应新数据和保留旧知识之间找到平衡。

### 翻译

时间序列基础模型(TSFMs)已在多样化的预测任务中展现出有前途的零样本泛化能力。然而，它们对持续适应的鲁棒性仍处于探索不足的状态。在本工作中，我们研究了TSFMs在连续微调多个数据集时遭受灾难性遗忘的程度。使用具有不同程度周期结构的合成数据集，我们衡量了适应新数据与保留先前知识之间的权衡。我们的实验揭示，虽然微调提高了在新任务上的性能，但它常常导致先前学习任务的显著性能下降，说明了基本的稳定性-可塑性困境。


### 论文摘要

Time Series Foundation Models (TSFMs) have shown promising zero-shot generalization across diverse forecasting tasks. However, their robustness to continual adaptation remains underexplored. In this work, we investigate the extent to which TSFMs suffer from catastrophic forgetting when fine-tuned sequentially on multiple datasets. Using synthetic datasets designed with varying degrees of periodic structure, we measure the trade-off between adaptation to new data and retention of prior knowledge. Our experiments reveal that, while fine-tuning improves performance on new tasks, it often causes significant degradation on previously learned ones, illustrating a fundamental stability-plasticity dilemma.

---

## 92. Solar PV Installation Potential Assessment on Building Facades Based on Vision and Language Foundation Models

**论文链接:** [http://arxiv.org/abs/2510.00797v1](http://arxiv.org/abs/2510.00797v1)

**作者:** Ruyu Liu, Dongxu Zhuang, Jianhua Zhang, Arega Getaneh Abate, Per Sieverts Nielsen, Ben Wang, Xiufeng Liu

**发布时间:** 2025-10-01

### GPT解析

### 总结

该研究提出了SF-SPA框架，通过计算机视觉和人工智能技术自动将街景照片转化为光伏部署的定量评估，解决了城市建筑立面光伏潜力评估的挑战。

### 背景

城市建筑立面是密集城市环境中太阳能发电的重要未开发资源，但由于复杂几何形状和语义组件，评估其光伏潜力具有挑战性。

### 目的

引入SF-SPA（语义立面光伏评估）自动化框架，将街景照片转换为光伏部署的定量评估，提高评估效率和准确性。

### 方法

结合计算机视觉和人工智能技术，采用四阶段流程：几何校正、零样本语义分割、大型语言模型引导的空间推理和能量模拟，解决透视失真校正、立面元素语义理解和光伏布局优化三个关键挑战。

### 主要发现

在四个国家的80栋建筑上验证，平均面积估计误差为6.2%±2.8%，每栋建筑评估约需100秒，效率远高于手动方法；模拟发电量预测证实了该方法在区域潜力研究、城市能源规划和建筑一体化光伏部署中的可靠性和适用性。

### 结论

SF-SPA框架能够有效评估城市建筑立面的光伏潜力，为城市能源规划和光伏部署提供支持。

### 翻译

建筑立面代表了密集城市环境中太阳能发电的重要未开发资源，但由于复杂几何形状和语义组件，评估其光伏潜力仍然具有挑战性。本研究引入了SF-SPA（语义立面光伏评估），这是一个自动化框架，将街景照片转换为光伏部署的定量评估。该方法结合计算机视觉和人工智能技术，解决三个关键挑战：透视失真校正、立面元素的语义理解和光伏布局优化的空间推理。我们的四阶段流程通过几何校正、零样本语义分割、大型语言模型引导的空间推理和能量模拟来处理图像。在四个国家的80栋建筑上的验证表明，与专家注释相比，平均面积估计误差为6.2%±2.8%，性能稳健。自动化评估每栋建筑约需100秒，效率比手动方法大幅提高。模拟的发电量预测证实了该方法在区域潜力研究、城市能源规划和建筑一体化光伏部署中的可靠性和适用性。代码可在https://github.com/CodeAXu/Solar-PV-Installation获取。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决建筑物立面太阳能光伏安装潜力的评估问题。这个问题很重要，因为在密集城市环境中，建筑物立面是未被充分利用的太阳能资源，其面积往往超过屋顶面积的3-5倍。传统方法主要关注屋顶安装，忽略立面潜力会导致对城市总可再生能源容量的严重低估，同时立面评估面临复杂几何形状、语义组件和透视失真等独特挑战。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先识别出现有方法的局限性：屋顶导向方法忽略立面潜力，整体建筑评估方法将立面简化为均匀表面，现有方法需要昂贵的3D数据或专门训练的模型。基于这些局限，作者设计了结合几何校正、语义理解和空间推理的解决方案。方法借鉴了计算机视觉和人工智能技术，利用了现有基础模型如Grounding-DINO和Segment Anything Model，以及光伏性能模拟工具pvlib，但创新性地将它们组合用于立面光伏评估。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是将街道级别建筑物立面照片自动转换为可量化的光伏部署评估，结合计算机视觉和人工智能技术解决透视失真校正、立面元素语义理解和光伏布局优化三个关键挑战。整体流程分为四个阶段：1)立面图像获取和几何校正，使用语义关键点进行单应性变换校正透视失真并建立物理尺度；2)视觉基础模型的语义感知，使用零样本语义分割区分建筑元素；3)大型语言模型驱动的光伏布局推理，通过提示链将语义掩码转换为符合规范的布局；4)辐照度和能量模拟，使用pvlib估算年发电量。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': "关键创新点包括：1)自动化立面光伏评估框架，从单个街道视图图像量化可安装面积并预测发电量；2)失真感知校正，使用语义关键点执行单应性变换校正透视失真；3)零样本立面解析，利用视觉-语言模型进行像素级分割无需专门训练数据；4)提示引导的空间推理，通过'描述→分区→过滤→总结'提示链引导大型语言模型生成实用光伏布局。相比之前工作，此方法不再依赖屋顶导向或简化立面模型，不需要昂贵3D数据，结合了最新基础模型，提供了完全自动化的从2D图像到精确光伏布局和能量估计的流程。", '如果要用一句话总结这篇论文的贡献，你会怎么说？': '本文提出了一种创新的自动化框架SF-SPA，通过结合视觉-语言基础模型和大型语言模型，能够从单个街道级别建筑物立面照片中准确评估太阳能光伏安装潜力，为城市能源规划和建筑集成光伏系统部署提供了高效、可靠的工具。'}


### 论文摘要

Building facades represent a significant untapped resource for solar energy generation in dense urban environments, yet assessing their photovoltaic (PV) potential remains challenging due to complex geometries and semantic com ponents. This study introduces SF-SPA (Semantic Facade Solar-PV Assessment), an automated framework that transforms street-view photographs into quantitative PV deployment assessments. The approach combines com puter vision and artificial intelligence techniques to address three key challenges: perspective distortion correction, semantic understanding of facade elements, and spatial reasoning for PV layout optimization. Our four-stage pipeline processes images through geometric rectification, zero-shot semantic segmentation, Large Language Model (LLM) guided spatial reasoning, and energy simulation. Validation across 80 buildings in four countries demonstrates ro bust performance with mean area estimation errors of 6.2% &#177; 2.8% compared to expert annotations. The auto mated assessment requires approximately 100 seconds per building, a substantial gain in efficiency over manual methods. Simulated energy yield predictions confirm the method's reliability and applicability for regional poten tial studies, urban energy planning, and building-integrated photovoltaic (BIPV) deployment. Code is available at: https:github.com/CodeAXu/Solar-PV-Installation

---

## 93. How Foundational are Foundation Models for Time Series Forecasting?

**论文链接:** [http://arxiv.org/abs/2510.00742v2](http://arxiv.org/abs/2510.00742v2)

**作者:** Nouha Karaouli, Denis Coquenet, Elisa Fromont, Martial Mermillod, Marina Reyboz

**发布时间:** 2025-10-01

**备注:** Accepted at NeurIPS 2025 Workshop on Recent Advances in Time Series  Foundation Models (BERT2S)

### GPT解析

### 总结

该论文探讨了时间序列数据构建Foundation模型的适用性问题，指出时间序列数据的内在特性使其不适合作为Foundation模型的基础，并通过预测任务验证了这一观点。

### 背景

Foundation Models通常作为多功能嵌入机器设计，具有强大的零样本能力和在微调后的优越泛化性能，这在语言和视觉Foundation模型中表现尤为突出。

### 目的

作者旨在论证时间序列数据的内在多样性使其不适合构建有效的Foundation模型，并通过预测任务来验证这一观点。

### 方法

作者使用预测作为下游任务，评估时间序列Foundation模型的零样本能力和微调后的性能，并将其与针对特定预测任务定制的小型模型进行比较。

### 主要发现

时间序列Foundation模型的零样本能力受到其预训练领域的显著影响；当应用于未见过的真实世界时间序列数据时，微调后的Foundation模型并不比针对特定任务定制的小型模型一致地产生更好的结果，尽管参数数量和内存占用更大。

### 结论

时间序列数据由于其内在多样性，不太适合构建有效的Foundation模型，至少在预测任务方面是这样。

### 翻译

基础模型被设计为多功能的嵌入机器，具有强大的零样本能力和在多样化下游任务上微调后的优越泛化性能。虽然这在语言和视觉基础模型中很大程度上成立，但我们认为时间序列数据的内在多样性使得它们不太适合构建有效的基础模型。我们使用预测作为下游任务来证明这一点。我们展示了时间序列基础模型的零样本能力受到其预训练领域的显著影响和限制。此外，当应用于未见过的真实世界时间序列数据时，微调后的基础模型并没有比针对特定预测任务定制的小型模型一致地产生明显更好的结果，考虑到其增加的参数数量和内存占用。


### 论文摘要

Foundation Models are designed to serve as versatile embedding machines, with strong zero shot capabilities and superior generalization performance when fine-tuned on diverse downstream tasks. While this is largely true for language and vision foundation models, we argue that the inherent diversity of time series data makes them less suited for building effective foundation models. We demonstrate this using forecasting as our downstream task. We show that the zero-shot capabilities of a time series foundation model are significantly influenced and tied to the specific domains it has been pretrained on. Furthermore, when applied to unseen real-world time series data, fine-tuned foundation models do not consistently yield substantially better results, relative to their increased parameter count and memory footprint, than smaller, dedicated models tailored to the specific forecasting task at hand.

---

## 94. Flexible Uncertainty Calibration for Machine-Learned Interatomic Potentials

**论文链接:** [http://arxiv.org/abs/2510.00721v1](http://arxiv.org/abs/2510.00721v1)

**作者:** Cheuk Hin Ho, Christoph Ortner, Yangshuai Wang

**发布时间:** 2025-10-01

### GPT解析

### 总结

该研究提出了一种灵活的不确定性校准框架，用于机器学习原子势能(MLIPs)中的不确定性量化，解决了现有共形预测技术在准确性、可扩展性和适应性方面的局限性。

### 背景

可靠的不确定性量化(UQ)对开发用于预测原子模拟的机器学习原子势能至关重要。共形预测(CP)虽能提供覆盖保证，但现有技术常缺乏准确性、可扩展性及对原子环境复杂性的适应性。

### 目的

为MLIPs开发一个灵活的不确定性校准框架，提高预测区间的准确性、可扩展性和适应性，同时保持计算效率。

### 方法

提出一种受CP启发但重新表述为参数化优化问题的框架，直接学习环境相关的分位数函数，以最小计算成本产生更锐利和自适应的预测区间。使用MACE-MP-0基础模型在离子晶体、催化表面和分子系统等多种基准中进行验证。

### 主要发现

实现了不确定性-误差相关性数量级的改进，提高了主动学习中的数据效率，展现出强大的泛化性能，并能可靠地在不同交换关联泛函间转移校准的不确定性。

### 结论

该工作为MLIPs中的不确定性校准建立了原则性和数据高效的方法，为更可靠和可转移的原子模拟提供了实用途径。

### 翻译

可靠的不确定性量化(UQ)对于开发用于预测原子模拟的机器学习原子势能(MLIPs)至关重要。共形预测(CP)是一种统计框架，能在最小假设下构建具有覆盖保证的预测区间，使其成为UQ的有吸引力的工具。然而，现有的CP技术虽然提供了正式的覆盖保证，但通常缺乏准确性、可扩展性以及对原子环境复杂性的适应性。在这项工作中，我们为MLIPs提出了一个灵活的不确定性校准框架，受CP启发但重新表述为参数化优化问题。这种表述能够直接学习环境相关的分位数函数，以最小的计算成本产生更锐利和自适应的预测区间。使用MACE-MP-0基础模型作为代表性案例，我们在离子晶体、催化表面和分子系统等多种基准中展示了该框架。我们的结果显示不确定性-误差相关性有数量级的改进，主动学习中的数据效率得到增强，同时具有强大的泛化性能，以及在不同的交换关联泛函之间可靠转移校准的不确定性。这项工作为MLIPs中的不确定性校准建立了一个原则性和数据高效的方法，为更可靠和可转移的原子模拟提供了实用途径。


### 论文摘要

Reliable uncertainty quantification (UQ) is essential for developing machine-learned interatomic potentials (MLIPs) in predictive atomistic simulations. Conformal prediction (CP) is a statistical framework that constructs prediction intervals with guaranteed coverage under minimal assumptions, making it an attractive tool for UQ. However, existing CP techniques, while offering formal coverage guarantees, often lack accuracy, scalability, and adaptability to the complexity of atomic environments. In this work, we present a flexible uncertainty calibration framework for MLIPs, inspired by CP but reformulated as a parameterized optimization problem. This formulation enables the direct learning of environment-dependent quantile functions, producing sharper and more adaptive predictive intervals at negligible computational cost. Using the foundation model MACE-MP-0 as a representative case, we demonstrate the framework across diverse benchmarks, including ionic crystals, catalytic surfaces, and molecular systems. Our results show order-of-magnitude improvements in uncertainty-error correlation, enhanced data efficiency in active learning, and strong generalization performance, together with reliable transfer of calibrated uncertainties across distinct exchange-correlation functionals. This work establishes a principled and data-efficient approach to uncertainty calibration in MLIPs, providing a practical route toward more trustworthy and transferable atomistic simulations.

---

## 95. ProtoMask: Segmentation-Guided Prototype Learning

**论文链接:** [http://arxiv.org/abs/2510.00683v1](http://arxiv.org/abs/2510.00683v1)

**作者:** Steffen Meinert, Philipp Schlinge, Nils Strodthoff, Martin Atzmueller

**发布时间:** 2025-10-01

### GPT解析

### 总结

本文介绍了一种名为ProtoMask的新型模型架构，利用图像分割基础模型提高可解释性，通过将注意力图计算限制在预定义的语义图像块上，减少可视化的不确定性。

### 背景

XAI(可解释人工智能)近年来获得了相当大的重要性。基于原型案例推理的方法在可解释性方面显示出有希望的提升，但这些方法通常依赖于额外的后期显著性技术来解释学习到的原型的语义，而这些技术的可靠性和质量受到了多方批评。

### 目的

研究使用突出的图像分割基础模型来提高嵌入空间与输入空间之间映射的真实性，旨在将显著性图的计算区域限制在预定义的语义图像块上，以减少此类可视化的不确定性。

### 方法

使用每个生成的分割掩码的边界框来裁剪图像，以便感知整个图像的信息。每个掩码在名为ProtoMask的新型模型架构中产生一个单独的输入。

### 主要发现

在三个流行的细粒度分类数据集上使用广泛的指标进行实验，提供了关于可解释性特征的详细概述。与其他流行模型的比较表明，该模型具有竞争性的性能和独特的可解释性特征。

### 结论

ProtoMask模型通过利用图像分割基础模型改进了可解释性，特别是在减少可视化不确定性方面表现出色，同时保持了与其他模型相当的分类性能。

### 翻译

XAI近年来获得了相当大的重要性。基于原型案例推理的方法在可解释性方面显示出有希望的提升。然而，这些方法通常依赖于额外的后期显著性技术来解释学习到的原型的语义。关于此类技术的可靠性和质量已有多方批评。因此，我们研究了使用突出的图像分割基础模型来提高嵌入空间与输入空间之间映射的真实性。我们旨在将显著性图的计算区域限制在预定义的语义图像块上，以减少此类可视化的不确定性。为了感知整个图像的信息，我们使用每个生成的分割掩码的边界框来裁剪图像。每个掩码在名为ProtoMask的新型模型架构中产生一个单独的输入。我们在三个流行的细粒度分类数据集上使用广泛的指标进行实验，提供了关于可解释性特征的详细概述。与其他流行模型的比较表明，该模型具有竞争性的性能和独特的可解释性特征。


### 论文摘要

XAI gained considerable importance in recent years. Methods based on prototypical case-based reasoning have shown a promising improvement in explainability. However, these methods typically rely on additional post-hoc saliency techniques to explain the semantics of learned prototypes. Multiple critiques have been raised about the reliability and quality of such techniques. For this reason, we study the use of prominent image segmentation foundation models to improve the truthfulness of the mapping between embedding and input space. We aim to restrict the computation area of the saliency map to a predefined semantic image patch to reduce the uncertainty of such visualizations. To perceive the information of an entire image, we use the bounding box from each generated segmentation mask to crop the image. Each mask results in an individual input in our novel model architecture named ProtoMask. We conduct experiments on three popular fine-grained classification datasets with a wide set of metrics, providing a detailed overview on explainability characteristics. The comparison with other popular models demonstrates competitive performance and unique explainability features of our model. https://github.com/uos-sis/quanproto

---

## 96. U-DFA: A Unified DINOv2-Unet with Dual Fusion Attention for Multi-Dataset Medical Segmentation

**论文链接:** [http://arxiv.org/abs/2510.00585v1](http://arxiv.org/abs/2510.00585v1)

**作者:** Zulkaif Sajjad, Furqan Shaukat, Junaid Mir

**发布时间:** 2025-10-01

### GPT解析

### 总结

本文提出了一种名为U-DFA的医学图像分割方法，通过整合DINOv2和Unet架构，并引入局部-全局融合适配器(LGFA)来有效融合局部和全局特征，在多个数据集上实现了最先进的性能，同时显著减少了可训练参数。

### 背景

准确的医学图像分割在诊断中至关重要，但CNN模型存在局部感受野限制，无法捕获全局上下文；结合CNN和transformer的方法未能有效融合局部和全局特征；而现有的VLMs和基础模型虽可用于医学影像任务，但存在领域差距和高计算成本问题。

### 目的

提出U-DFA架构，通过集成局部-全局融合适配器(LGFA)来增强医学图像分割性能，有效融合高级语义特征和空间特征。

### 方法

U-DFA采用统一的DINOv2-Unet编码器-解码器架构，LGFA模块将基于CNN的空间模式适配器(SPA)的空间特征注入到多个阶段的冻结DINOv2块中，实现局部和全局特征的有效融合。

### 主要发现

该方法在Synapse和ACDC数据集上实现了最先进的性能，同时仅需33%的可训练模型参数，证明了其高效性。

### 结论

U-DFA是跨多种模态的医学图像分割的稳健且可扩展的框架，能够在保持高性能的同时显著减少计算资源需求。

### 翻译

准确的医学图像分割在整体诊断中起着至关重要的作用，是诊断流程中最关键的任务之一。尽管CNN-based模型被广泛使用，但它们存在局部感受野问题，无法捕获全局上下文。结合CNN和transformer的常见方法试图弥补这一差距，但未能有效融合局部和全局特征。随着最近VLMs和基础模型的出现，它们已被适应用于下游医学影像任务；然而，它们存在固有的领域差距和高计算成本。为此，我们提出了U-DFA，一个统一的DINOv2-Unet编码器-解码器架构，集成了新的局部-全局融合适配器(LGFA)以增强分割性能。LGFA模块将基于CNN的空间模式适配器(SPA)模块的空间特征注入到多个阶段的冻结DINOv2块中，实现了高级语义特征和空间特征的有效融合。我们的方法在Synapse和ACDC数据集上仅使用33%的可训练模型参数就实现了最先进的性能。这些结果表明，U-DFA是跨多种模态的医学图像分割的稳健且可扩展的框架。


### 论文摘要

Accurate medical image segmentation plays a crucial role in overall diagnosis and is one of the most essential tasks in the diagnostic pipeline. CNN-based models, despite their extensive use, suffer from a local receptive field and fail to capture the global context. A common approach that combines CNNs with transformers attempts to bridge this gap but fails to effectively fuse the local and global features. With the recent emergence of VLMs and foundation models, they have been adapted for downstream medical imaging tasks; however, they suffer from an inherent domain gap and high computational cost. To this end, we propose U-DFA, a unified DINOv2-Unet encoder-decoder architecture that integrates a novel Local-Global Fusion Adapter (LGFA) to enhance segmentation performance. LGFA modules inject spatial features from a CNN-based Spatial Pattern Adapter (SPA) module into frozen DINOv2 blocks at multiple stages, enabling effective fusion of high-level semantic and spatial features. Our method achieves state-of-the-art performance on the Synapse and ACDC datasets with only 33\% of the trainable model parameters. These results demonstrate that U-DFA is a robust and scalable framework for medical image segmentation across multiple modalities.

---

## 97. Assessing Foundation Models for Mold Colony Detection with Limited Training Data

**论文链接:** [http://arxiv.org/abs/2510.00561v1](http://arxiv.org/abs/2510.00561v1)

**作者:** Henrik Pichler, Janis Keuper, Matthew Copping

**发布时间:** 2025-10-01

**备注:** 17 pages, 2 figures, accepted as oral presentation at GCPR 2025

### GPT解析

### 总结

该研究展示了如何使用少量标注数据训练视觉基础模型来量化培养皿上的霉菌菌落，替代了传统上需要大量标注数据的方法。

### 背景

量化培养皿上的霉菌菌落对评估室内空气质量至关重要，高菌落数量可能表示潜在健康风险和通风系统缺陷。传统自动化方法依赖于大型数据集的手动标注和模型训练。

### 目的

证明在处理新的视觉任务时，详尽的标注不再是先决条件，展示数据高效的基础模型如何匹配传统方法。

### 方法

编译包含5000张培养皿图像的数据集，用边界框标注，模拟传统数据收集方法以及少样本和低样本场景，对比三种视觉基础模型与传统基线的性能。

### 主要发现

MaskDINO模型在仅150张图像上微调后达到与 extensively 训练的YoloV9模型几乎相当的性能；即使使用仅25张图像，仍能在约70%的样本上保持可靠性能。

### 结论

数据高效的基础模型只需传统方法所需数据的一小部分就能匹配传统方法，使自动化微生物系统能更早开发并更快迭代改进，且具有更高的上限性能。

### 翻译

量化培养皿样本上霉菌菌落的过程对评估室内空气质量至关重要，因为高菌落数量可能表示潜在的健康风险和通风系统缺陷。传统上，这种劳动密集型过程的自动化以及微生物学中的其他任务依赖于大型数据集的手动标注和后续的模型训练（如YoloV9）。为了证明在处理新的视觉任务时详尽的标注不再是先决条件，我们编译了一个包含5000张培养皿图像的代表数据集，用边界框进行标注，模拟了传统数据收集方法以及少样本和低样本场景，具有精心策划的子集和实例级掩码。我们在反映现实世界实际需求的特定任务指标上，将三种视觉基础模型与传统基线进行对比。值得注意的是，MaskDINO在仅150张图像上微调后就达到了与 extensively 训练的YoloV9模型几乎相当的性能，即使使用仅25张图像仍能保持有竞争力的性能，在约70%的样本上仍然可靠。我们的结果表明，数据高效的基础模型只需传统方法所需数据的一小部分就能匹配传统方法，使自动化微生物系统能够更早开发并更快迭代改进，且具有比传统模型更高的上限性能。


### 论文摘要

The process of quantifying mold colonies on Petri dish samples is of critical importance for the assessment of indoor air quality, as high colony counts can indicate potential health risks and deficiencies in ventilation systems. Conventionally the automation of such a labor-intensive process, as well as other tasks in microbiology, relies on the manual annotation of large datasets and the subsequent extensive training of models like YoloV9. To demonstrate that exhaustive annotation is not a prerequisite anymore when tackling a new vision task, we compile a representative dataset of 5000 Petri dish images annotated with bounding boxes, simulating both a traditional data collection approach as well as few-shot and low-shot scenarios with well curated subsets with instance level masks. We benchmark three vision foundation models against traditional baselines on task specific metrics, reflecting realistic real-world requirements. Notably, MaskDINO attains near-parity with an extensively trained YoloV9 model while finetuned only on 150 images, retaining competitive performance with as few as 25 images, still being reliable on $\approx$ 70% of the samples. Our results show that data-efficient foundation models can match traditional approaches with only a fraction of the required data, enabling earlier development and faster iterative improvement of automated microbiological systems with a superior upper-bound performance than traditional models would achieve.

---

## 98. Black-Box Time-Series Domain Adaptation via Cross-Prompt Foundation Models

**论文链接:** [http://arxiv.org/abs/2510.00487v1](http://arxiv.org/abs/2510.00487v1)

**作者:** M. T. Furqon, Mahardhika Pratama, Igor Skrjanc, Lin Liu, Habibullah Habibullah, Kutluyil Dogancay

**发布时间:** 2025-10-01

### GPT解析

### 总结

这篇论文提出了跨提示基础模型(CPFM)，用于解决黑盒时间序列领域适应(BBTSDA)问题，通过双分支网络结构和独特的提示设计，有效捕捉时间序列数据的时空特征，并在多个数据集上优于现有方法。

### 背景

黑盒领域适应(BBDA)旨在解决隐私安全问题，其中只有源模型的API可用。现有研究多集中于视觉应用，不适用于具有独特时空特征的时间序列应用，且尚未探索基础模型在黑盒时间序列领域适应中的应用。

### 目的

提出一种针对黑盒时间序列领域适应(BBTSDA)问题的跨提示基础模型(CPFM)，构建能处理时间序列数据时空动态的基础模型。

### 方法

CPFM采用双分支网络结构构建，每个分支配备独特提示以捕捉数据分布的不同特征；在领域适应阶段，开发提示级别和输入级别的重建学习阶段；所有构建都基于时间序列基础模型，以克服时空动态问题。

### 主要发现

通过严格实验证明CPFM的优势，在三个不同应用领域的时间序列数据集上，以明显优势优于竞争对手，取得改进结果。

### 结论

CPFM是有效的黑盒时间序列领域适应方法，基础模型在处理时间序列数据的时空特征方面具有显著优势。

### 翻译

黑盒领域适应(BBDA)主题的发展是为了解决隐私和安全问题，其中只有源模型的应用程序接口(API)可用于领域适应。尽管BBDA主题已引起越来越多的研究关注，但现有工作大多针对视觉应用，不适用于具有独特时空特征的时间序列应用。此外，现有方法中没有任何一种探索了基础模型在黑盒时间序列领域适应(BBTSDA)中的优势。本文提出了针对BBTSDA问题的跨提示基础模型(CPFM)概念。CPFM在双分支网络结构下构建，每个分支配备独特提示以捕捉数据分布的不同特征。在领域适应阶段，开发了提示级别和输入级别的重建学习阶段。所有这些都构建在时间序列基础模型之上，以克服时空动态。我们的严格实验证明了CPFM的优势，在三个不同应用领域的时间序列数据集上，以明显优势优于竞争对手，取得了改进的结果。


### 论文摘要

The black-box domain adaptation (BBDA) topic is developed to address the privacy and security issues where only an application programming interface (API) of the source model is available for domain adaptations. Although the BBDA topic has attracted growing research attentions, existing works mostly target the vision applications and are not directly applicable to the time-series applications possessing unique spatio-temporal characteristics. In addition, none of existing approaches have explored the strength of foundation model for black box time-series domain adaptation (BBTSDA). This paper proposes a concept of Cross-Prompt Foundation Model (CPFM) for the BBTSDA problems. CPFM is constructed under a dual branch network structure where each branch is equipped with a unique prompt to capture different characteristics of data distributions. In the domain adaptation phase, the reconstruction learning phase in the prompt and input levels is developed. All of which are built upon a time-series foundation model to overcome the spatio-temporal dynamic. Our rigorous experiments substantiate the advantage of CPFM achieving improved results with noticeable margins from its competitors in three time-series datasets of different application domains.

---

## 99. Learning a Zeroth-Order Optimizer for Fine-Tuning LLMs

**论文链接:** [http://arxiv.org/abs/2510.00419v1](http://arxiv.org/abs/2510.00419v1)

**作者:** Kairun Zhang, Haoyu Li, Yanjun Zhao, Yifan Sun, Huan Zhang

**发布时间:** 2025-10-01

### GPT解析

### 总结

本文提出了一种名为ZO Fine-tuner的基于学习的零阶优化器，用于微调大型语言模型，通过自动学习高效的扰动策略，显著减少了GPU内存消耗，并可在不同下游任务间重用。

### 背景

零阶优化器作为微调大型语言模型的实用方法，比传统一阶方法显著减少GPU内存消耗，但现有方法依赖手工制作的静态采样策略，无法适应特定模型结构。

### 目的

开发一种学习型的零阶优化器，能够自动学习高效的扰动策略，通过紧凑且内存高效的设计解决现有方法的局限性，并支持一次训练后跨任务重用。

### 方法

基于只有少数基础模型及其变体在实践中被广泛采用的观察，设计ZO Fine-tuner将学习到学习(L2L)扩展到基础模型时代，支持每个LLM只需一次训练，开销最小。

### 主要发现

在4个LLMs和7个数据集上的实验表明，ZO Fine-tuner在82.1%的任务-模型组合中优于先前的零阶基线，展示了高效LLMs微调的强大性能和可扩展性。

### 结论

ZO Fine-tuner是一种有效的零阶优化器，通过学习高效的扰动策略，比传统方法更节省内存，并可在不同下游任务间重用，提高了效率。

### 翻译

零阶优化器最近已成为微调大型语言模型(LLMs)的一种实用方法，与传统一阶方法相比显著减少了GPU内存消耗。然而，现有的零阶方法依赖于手工制作的静态采样策略，这些策略不能适应特定模型的结构。为此，我们提出了ZO Fine-tuner，一种基于学习的LLMs零阶优化器，通过紧凑且内存高效的设计自动学习高效的扰动策略。关键的是，我们的方法基于一个观察：在实践中只有少数基础模型及其变体被广泛采用。因此，为给定的LLM学习一次优化器并在各种下游任务中重用是可行且高度可取的。相应地，ZO Fine-tuner通过支持每个LLM只需一次训练且开销最小，将学习到学习(L2L)扩展到基础模型时代。在4个LLMs和7个数据集上的实验表明，ZO Fine-tuner在82.1%的任务-模型组合中优于先前的零阶基线，从而展示了高效LLMs微调的强大性能和可扩展性。我们的代码可在https://github.com/ASTRAL-Group/ZO_Fine_tuner.git获取。


### 论文摘要

Zeroth-order optimizers have recently emerged as a practical approach for fine-tuning large language models (LLMs), significantly reducing GPU memory consumption compared to traditional first-order methods. Yet, existing zeroth-order methods rely on hand-crafted, static sampling strategies that are not adaptable to model-specific structures. To address this, we propose ZO Fine-tuner, a learning-based zeroth-order optimizer for LLMs that automatically learns efficient perturbation strategies through a compact and memory-efficient design. Crucially, our approach is motivated by the observation that only a small number of foundation models and their derivatives are widely adopted in practice. Therefore, learning the optimizer once for a given LLM and reusing it across diverse downstream tasks is both feasible and highly desirable. Accordingly, ZO Fine-tuner is designed to scale learning to learn (L2L) to the foundation-model era by supporting one-time training per LLM with minimal overhead. Experiments on 4 LLMs and 7 datasets show that ZO Fine-tuner outperforms prior zeroth-order baselines in 82.1\% of task-model combinations, thereby demonstrating strong performance and scalability for efficient LLM fine-tuning. Our code is available at https://github.com/ASTRAL-Group/ZO_Fine_tuner.git.

---

## 100. Evaluating New AI Cell Foundation Models on Challenging Kidney Pathology Cases Unaddressed by Previous Foundation Models

**论文链接:** [http://arxiv.org/abs/2510.01287v1](http://arxiv.org/abs/2510.01287v1)

**作者:** Runchen Wang, Junlin Guo, Siqi Lu, Ruining Deng, Zhengyi Lu, Yanfan Zhu, Yuechen Yang, Chongyu Qu, Yu Wang, Shilin Zhao, Catie Chang, Mitchell Wilkes, Mengmeng Yin, Haichun Yang, Yuankai Huo

**发布时间:** 2025-10-01

### GPT解析

### 总结

本研究评估了最新AI细胞基础模型在肾脏病理学细胞核分割任务上的性能，通过比较多种模型和融合方法，发现最新模型显著提高了分割准确性，融合模型尤其有效，解决了大多数具有挑战性的案例。

### 背景

准确的细胞核分割对肾脏病理学的下游任务至关重要，但由于肾脏组织的形态多样性和成像变异性，这仍然是一个重大挑战。之前的研究评估了早期一代的AI细胞基础模型，但最近细胞基础模型的有效性尚不清楚。

### 目的

基准测试先进的AI细胞基础模型（2025年），包括CellViT++变体和Cellpose-SAM，与2024年之前开发的三种广泛使用的细胞基础模型进行比较，评估它们在肾脏图像分割任务上的性能。

### 方法

使用包含人类参与循环评分框架的多样化大规模肾脏图像补丁集进行评估，进行基于融合的集成评估和模型一致性分析，使用2,091个具有挑战性的样本进行测试。

### 主要发现

CellViT++ [Virchow]在独立性能方面表现最佳，40.3%的预测被评为'好'；融合模型实现了62.2%的'好'预测和仅0.4%的'坏'预测，显著减少了分割错误；融合模型成功解决了之前研究中大多数未解决的具有挑战性案例。

### 结论

这些发现证明了AI细胞基础模型在肾脏病理学中的开发潜力，提供了一个具有挑战性样本的精选数据集，以支持未来肾脏特定模型的改进。

### 翻译

准确的细胞核分割对肾脏病理学的下游任务至关重要，并由于肾脏组织的形态多样性和成像变异性，仍然是一个重大挑战。虽然我们之前的工作已经评估了该领域中早期一代的AI细胞基础模型，但最近的细胞基础模型的有效性仍然不清楚。在这项研究中，我们使用包含人类参与循环评分框架的多样化大规模肾脏图像补丁集，基准测试了先进的AI细胞基础模型（2025年），包括CellViT++变体和Cellpose-SAM，与2024年之前开发的三种广泛使用的细胞基础模型进行比较。我们进一步进行了基于融合的集成评估和模型一致性分析，以评估不同模型的分割能力。我们的结果显示，CellViT++ [Virchow]在独立性能方面表现最佳，在2,091个精选的具有挑战性样本中，40.3%的预测被评为'好'，优于所有之前的模型。此外，我们的融合模型实现了62.2%的'好'预测和仅0.4%的'坏'预测，显著减少了分割错误。值得注意的是，融合模型（2025年）成功解决了我们之前研究中大多数未解决的具有挑战性案例。这些发现证明了AI细胞基础模型在肾脏病理学中的开发潜力，并提供了一个具有挑战性样本的精选数据集，以支持未来肾脏特定模型的改进。


### 论文摘要

Accurate cell nuclei segmentation is critical for downstream tasks in kidney pathology and remains a major challenge due to the morphological diversity and imaging variability of renal tissues. While our prior work has evaluated early-generation AI cell foundation models in this domain, the effectiveness of recent cell foundation models remains unclear. In this study, we benchmark advanced AI cell foundation models (2025), including CellViT++ variants and Cellpose-SAM, against three widely used cell foundation models developed prior to 2024, using a diverse large-scale set of kidney image patches within a human-in-the-loop rating framework. We further performed fusion-based ensemble evaluation and model agreement analysis to assess the segmentation capabilities of the different models. Our results show that CellViT++ [Virchow] yields the highest standalone performance with 40.3% of predictions rated as "Good" on a curated set of 2,091 challenging samples, outperforming all prior models. In addition, our fused model achieves 62.2% "Good" predictions and only 0.4% "Bad", substantially reducing segmentation errors. Notably, the fusion model (2025) successfully resolved the majority of challenging cases that remained unaddressed in our previous study. These findings demonstrate the potential of AI cell foundation model development in renal pathology and provide a curated dataset of challenging samples to support future kidney-specific model refinement.

---

