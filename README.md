# 今日论文推荐 - 2025-12-25

共 38 篇论文

---

## 1. A Community-Enhanced Graph Representation Model for Link Prediction

**论文链接:** [http://arxiv.org/abs/2512.21166v1](http://arxiv.org/abs/2512.21166v1)

**作者:** Lei Wang, Darong Lai

**发布时间:** 2025-12-24

### GPT解析

### 总结

该研究提出了一种社区增强链接预测(CELP)框架，通过结合社区结构同时建模局部和全局图拓扑，解决了现有图神经网络在链接预测任务中的局限性。

### 背景

图神经网络(GNNs)已成为图表示学习的主导方法，但在链接预测任务上并不总是优于传统启发式方法。现有GNNs倾向于学习局部节点表示，难以有效捕获节点对间的结构关系，且过度依赖局部邻域信息会导致过平滑问题。

### 目的

解决现有GNNs在链接预测中的局限性，通过引入社区结构来同时建模局部和全局图拓扑，提高链接预测的准确性。

### 方法

提出社区增强链接预测(CELP)框架，通过社区感知、置信度引导的边完成和剪枝来增强图，并集成多尺度结构特征以实现更准确的链接预测。

### 主要发现

在多个基准数据集上的实验结果表明，CELP实现了卓越的性能，验证了社区结构在提高链接预测准确性中的关键作用。

### 结论

社区结构对于改进链接预测准确性至关重要，CELP框架通过结合社区感知和多尺度特征，有效解决了现有GNNs在链接预测中的局限性。

### 翻译

尽管图神经网络(GNNs)已成为图表示学习的主导方法，但它们在链接预测任务上的表现并不总是优于传统的启发式方法，如共同邻居和Jaccard系数。这主要是因为现有的GNNs倾向于学习局部节点表示，难以有效捕获节点对之间的结构关系。此外，过度依赖局部邻域信息可能导致过平滑。先前的研究表明，引入全局结构编码可以部分缓解这一问题。为解决这些局限性，我们提出了一个社区增强链接预测(CELP)框架，该框架结合社区结构来共同建模局部和全局图拓扑。具体而言，CELP通过社区感知、置信度引导的边完成和剪枝来增强图，同时集成多尺度结构特征以实现更准确的链接预测。在多个基准数据集上的实验结果表明，CELP实现了卓越的性能，验证了社区结构在提高链接预测准确性中的关键作用。


### 论文摘要

Although Graph Neural Networks (GNNs) have become the dominant approach for graph representation learning, their performance on link prediction tasks does not always surpass that of traditional heuristic methods such as Common Neighbors and Jaccard Coefficient. This is mainly because existing GNNs tend to focus on learning local node representations, making it difficult to effectively capture structural relationships between node pairs. Furthermore, excessive reliance on local neighborhood information can lead to over-smoothing. Prior studies have shown that introducing global structural encoding can partially alleviate this issue. To address these limitations, we propose a Community-Enhanced Link Prediction (CELP) framework that incorporates community structure to jointly model local and global graph topology. Specifically, CELP enhances the graph via community-aware, confidence-guided edge completion and pruning, while integrating multi-scale structural features to achieve more accurate link prediction. Experimental results across multiple benchmark datasets demonstrate that CELP achieves superior performance, validating the crucial role of community structure in improving link prediction accuracy.

---

## 2. Semantic Refinement with LLMs for Graph Representations

**论文链接:** [http://arxiv.org/abs/2512.21106v1](http://arxiv.org/abs/2512.21106v1)

**作者:** Safal Thapaliya, Zehong Wang, Jiazheng Li, Ziming Li, Yanfang Ye, Chuxu Zhang

**发布时间:** 2025-12-24

### GPT解析

### 总结

论文提出DAS（数据自适应语义细化）框架，通过将固定图神经网络和大语言模型结合在闭环反馈系统中，解决图结构数据中的结构-语义异质性问题。

### 背景

图结构数据在预测信号来源上存在显著异质性：某些领域节点级语义占主导，其他领域结构模式起核心作用。这种异质性意味着没有固定归纳偏置的图学习模型能在多样化图领域最优泛化。现有方法从模型侧解决此问题，但面对现实世界图的开放多样性存在根本局限。

### 目的

从数据中心视角出发，将节点语义视为任务自适应变量，提出DAS框架用于图表示学习，以应对图数据中的结构-语义异质性挑战。

### 方法

DAS框架将固定图神经网络(GNN)和大语言模型(LLM)耦合在闭环反馈系统中。GNN提供隐式监督信号指导LLM的语义细化，细化后的语义又反馈回更新同一图学习器。

### 主要发现

在文本丰富和文本稀疏的图上评估结果显示，该方法在结构主导的图上取得持续改进，同时在语义丰富的图上保持竞争力，证明了在结构-语义异质性下数据中心语义适应的有效性。

### 结论

通过数据中心的语义适应方法可以有效处理图数据中的结构-语义异质性，使模型能够在不同类型的图数据上表现更好。

### 翻译

图结构数据在预测信号的来源上表现出显著的异质性：在某些领域中，节点级语义占主导地位，而在其他领域中，结构模式则起着核心作用。这种结构-语义异质性意味着没有具有固定归纳偏置的图学习模型能够在多样化的图领域上最优地泛化。然而，大多数现有方法从模型侧解决这个问题，通过逐步注入新的归纳偏置，这在面对现实世界中图的开端多样性时仍然存在根本性限制。在这项工作中，我们采用数据中心的视角，将节点语义视为任务自适应变量。我们提出了一个用于图表示学习的数据自适应语义细化框架DAS，该框架将固定的图神经网络（GNN）和大语言模型（LLM）耦合在闭环反馈系统中。GNN提供隐式监督信号来指导LLM的语义细化，而细化的语义又被反馈回更新同一个图学习器。我们在文本丰富和文本稀疏的图上都评估了我们的方法。结果表明，在结构主导的图上取得了持续改进，同时在语义丰富的图上保持竞争力，证明了在结构-语义异质性下数据中心语义适应的有效性。


### 论文摘要

Graph-structured data exhibit substantial heterogeneity in where their predictive signals originate: in some domains, node-level semantics dominate, while in others, structural patterns play a central role. This structure-semantics heterogeneity implies that no graph learning model with a fixed inductive bias can generalize optimally across diverse graph domains. However, most existing methods address this challenge from the model side by incrementally injecting new inductive biases, which remains fundamentally limited given the open-ended diversity of real-world graphs. In this work, we take a data-centric perspective and treat node semantics as a task-adaptive variable. We propose a Data-Adaptive Semantic Refinement framework DAS for graph representation learning, which couples a fixed graph neural network (GNN) and a large language model (LLM) in a closed feedback loop. The GNN provides implicit supervisory signals to guide the semantic refinement of LLM, and the refined semantics are fed back to update the same graph learner. We evaluate our approach on both text-rich and text-free graphs. Results show consistent improvements on structure-dominated graphs while remaining competitive on semantics-rich graphs, demonstrating the effectiveness of data-centric semantic adaptation under structure-semantics heterogeneity.

---

## 3. A Multi-fidelity Double-Delta Wing Dataset and Empirical Scaling Laws for GNN-based Aerodynamic Field Surrogate

**论文链接:** [http://arxiv.org/abs/2512.20941v1](http://arxiv.org/abs/2512.20941v1)

**作者:** Yiren Shen, Juan J. Alonso

**发布时间:** 2025-12-24

### GPT解析

### 总结

本研究探讨了基于图神经网络的代理模型中训练数据大小与预测精度的关系，发布了开源多保真度气动数据集，并发现测试误差随数据大小呈幂律减少，指数为-0.6122，表明数据利用效率较高。

### 背景

数据驱动代理模型正越来越多地被采用以加速车辆设计，然而开源的多保真度数据集以及将数据集大小与模型性能关联的经验指导仍然有限。

### 目的

研究训练数据大小与基于图神经网络的代理模型预测精度之间的关系，并发布一个开源的多保真度气动数据集用于双三角翼。

### 方法

创建包含2448个流场快照的多保真度气动数据集，覆盖272种几何构型，攻角从11度到19度，马赫数为0.3；使用涡格法和雷诺平均纳维-斯托克斯求解器评估；构建六个不同大小的训练数据集；在固定训练预算下训练参数量从0.1到240万的模型。

### 主要发现

测试误差随数据大小呈幂律减少，指数为-0.6122；根据缩放定律估计d维设计空间中的最优采样密度约为每维八个样本；较大的代理模型显示出改进的数据利用效率。

### 结论

研究提供了关于数据集大小对图神经网络代理模型性能影响的实证指导，并发布开源数据集支持未来研究；模型训练预算与数据集生成成本之间存在潜在权衡。

### 翻译

数据驱动的代理模型正越来越多地被采用以加速车辆设计。然而，开源的多保真度数据集以及将数据集大小与模型性能关联的经验指导仍然有限。本研究探讨了用于气动场预测的基于图神经网络的代理模型的训练数据大小与预测精度之间的关系。我们发布了一个开源的、用于双三角翼的多保真度气动数据集，包含2448个流场快照，覆盖272种几何构型，在马赫数为0.3的情况下，使用涡格法和雷诺平均纳维-斯托克斯求解器评估了从11度到19度的攻角。几何构型使用嵌套的Saltelli采样方案生成，以支持未来数据集扩展和基于方差的敏感性分析。使用此数据集，我们通过构建六个训练数据集并在固定训练预算下训练不同参数量的模型，进行了初步的实证缩放研究。我们发现测试误差随数据大小呈幂律减少，指数为-0.6122，表明数据利用效率高。基于这一缩放定律，我们估计最优采样密度约为每维八个样本。研究结果表明，较大的代理模型具有改进的数据利用效率，暗示数据集生成成本与模型训练预算之间可能存在权衡。


### 论文摘要

Data-driven surrogate models are increasingly adopted to accelerate vehicle design. However, open-source multi-fidelity datasets and empirical guidelines linking dataset size to model performance remain limited. This study investigates the relationship between training data size and prediction accuracy for a graph neural network (GNN) based surrogate model for aerodynamic field prediction. We release an open-source, multi-fidelity aerodynamic dataset for double-delta wings, comprising 2448 flow snapshots across 272 geometries evaluated at angles of attack from 11 (degree) to 19 (degree) at Ma=0.3 using both Vortex Lattice Method (VLM) and Reynolds-Averaged Navier-Stokes (RANS) solvers. The geometries are generated using a nested Saltelli sampling scheme to support future dataset expansion and variance-based sensitivity analysis. Using this dataset, we conduct a preliminary empirical scaling study of the MF-VortexNet surrogate by constructing six training datasets with sizes ranging from 40 to 1280 snapshots and training models with 0.1 to 2.4 million parameters under a fixed training budget. We find that the test error decreases with data size with a power-law exponent of -0.6122, indicating efficient data utilization. Based on this scaling law, we estimate that the optimal sampling density is approximately eight samples per dimension in a d-dimensional design space. The results also suggest improved data utilization efficiency for larger surrogate models, implying a potential trade-off between dataset generation cost and model training budget.

---

## 4. Towards a General Framework for Predicting and Explaining the Hardness of Graph-based Combinatorial Optimization Problems using Machine Learning and Association Rule Mining

**论文链接:** [http://arxiv.org/abs/2512.20915v1](http://arxiv.org/abs/2512.20915v1)

**作者:** Bharat Sharman, Elkafi Hassini

**发布时间:** 2025-12-24

### GPT解析

### 总结

本研究介绍了GCO-HPIF，一个基于机器学习的通用框架，用于预测和解释可在图上表示的组合优化问题的计算难度。该框架包含两个阶段：第一阶段创建包含问题无关图特征和难度分类的数据集，并训练机器学习分类算法；第二阶段使用关联规则挖掘算法解释预测结果，并训练回归模型预测计算时间。

### 背景

组合优化问题的计算难度预测对于算法选择和性能优化至关重要，需要一种能够同时提供预测和解释功能的通用框架。

### 目的

开发一个通用的机器学习框架，用于预测和解释组合优化问题的计算难度，特别关注可在图上表示的问题。

### 方法

构建包含图特征和难度分类的数据集；训练分类算法将图特征映射到难度类别；使用FP-Growth算法进行关联规则挖掘以解释预测；训练回归模型预测计算时间；在3287个来自COLLAB、IMDB和TWITTER图数据集的最大团问题实例上测试，使用五种算法：Gurobi、CliSAT、MOMC、EGN和HGS。

### 主要发现

框架在难度预测方面表现出色，仅使用三个图特征就实现了0.9921的加权F1分数、0.878的少数类F1分数和0.9083的ROC-AUC分数；FP-Growth算法找到的最佳关联规则对困难实例的支持度为0.8829，总体准确率为87.64%；最佳回归模型实现了5.12的百分比RMSE和0.991的R2值。

### 结论

GCO-HPIF框架是一个有效的方法，用于预测和解释组合优化问题的计算难度，特别是在图上表示的问题。该框架不仅实现了高精度的预测，还提供了可解释性，有助于理解问题难度背后的原因。

### 翻译

本研究引入了GCO-HPIF，一个基于机器学习的通用框架，用于预测和解释可在图上表示的组合优化问题的计算难度。该框架包含两个阶段。在第一阶段，创建包含问题无关图特征和问题实例难度分类的数据集。基于机器学习的分类算法被训练以将图特征映射到难度类别。在第二阶段，框架使用关联规则挖掘算法解释预测结果。此外，基于机器学习的回归模型被训练以预测算法计算时间。GCO-HPIF框架应用于从COLLAB、IMDB和TWITTER图数据集编译的3287个最大团问题实例，使用了五种最先进的算法，即三种精确分支定界算法（Gurobi、CliSAT和MOMC）和两种基于图神经网络的算法（EGN和HGS）。该框架在预测实例难度方面表现出色，仅使用三个图特征就实现了0.9921的加权F1分数、0.878的少数类F1分数和0.9083的ROC-AUC分数。FP-Growth算法找到的用于解释难度预测的最佳关联规则对困难实例的支持度为0.8829，总体准确率为87.64%，强调了该框架在预测和解释方面的实用性。此外，用于预测计算时间的最佳回归模型实现了5.12的百分比RMSE和0.991的R2值。


### 论文摘要

This study introduces GCO-HPIF, a general machine-learning-based framework to predict and explain the computational hardness of combinatorial optimization problems that can be represented on graphs. The framework consists of two stages. In the first stage, a dataset is created comprising problem-agnostic graph features and hardness classifications of problem instances. Machine-learning-based classification algorithms are trained to map graph features to hardness categories. In the second stage, the framework explains the predictions using an association rule mining algorithm. Additionally, machine-learning-based regression models are trained to predict algorithmic computation times. The GCO-HPIF framework was applied to a dataset of 3287 maximum clique problem instances compiled from the COLLAB, IMDB, and TWITTER graph datasets using five state-of-the-art algorithms, namely three exact branch-and-bound-based algorithms (Gurobi, CliSAT, and MOMC) and two graph-neural-network-based algorithms (EGN and HGS). The framework demonstrated excellent performance in predicting instance hardness, achieving a weighted F1 score of 0.9921, a minority-class F1 score of 0.878, and an ROC-AUC score of 0.9083 using only three graph features. The best association rule found by the FP-Growth algorithm for explaining the hardness predictions had a support of 0.8829 for hard instances and an overall accuracy of 87.64 percent, underscoring the framework's usefulness for both prediction and explanation. Furthermore, the best-performing regression model for predicting computation times achieved a percentage RMSE of 5.12 and an R2 value of 0.991.

---

## 5. From GNNs to Symbolic Surrogates via Kolmogorov-Arnold Networks for Delay Prediction

**论文链接:** [http://arxiv.org/abs/2512.20885v1](http://arxiv.org/abs/2512.20885v1)

**作者:** Sami Marouani, Kamal Singh, Baptiste Jeudy, Amaury Habrard

**发布时间:** 2025-12-24

### GPT解析

### 总结

研究提出了三种建模方法来提高现代通信网络中流量延迟预测的准确性，包括异构GNN基线模型、FlowKANet模型和符号代理模型。

### 背景

准确预测流量延迟对于优化和管理现代通信网络至关重要。

### 目的

通过三种不同级别的建模方法提高流量延迟预测的准确性和效率。

### 方法

首先实现基于注意力的异构GNN建立神经基线；其次提出FlowKANet，用Kolmogorov-Arnold网络替代标准MLP层，减少参数同时保持性能；最后通过块状回归将模型蒸馏为符号代理模型，产生闭式方程。

### 主要发现

KAN层在效率和准确性之间提供了有利权衡；符号代理模型强调了轻量级部署和增强透明度的潜力。

### 结论

FlowKANet和符号代理模型在流量延迟预测中表现出色，提供了从高效计算到透明解释的多种优势。

### 翻译

准确预测流量延迟对于优化和管理现代通信网络至关重要。我们研究了此任务的三种建模级别。首先，我们实现了一个基于注意力的异构GNN，建立了强大的神经基线。其次，我们提出了FlowKANet，其中Kolmogorov-Arnold网络替代了标准MLP层，减少了可训练参数，同时保持了竞争性的预测性能。FlowKANet集成了KAMP-Attn（带有注意力的Kolmogorov-Arnold消息传递），将KAN运算符直接嵌入到消息传递和注意力计算中。最后，我们使用块状回归将模型蒸馏为符号代理模型，产生消除可训练权重但保留图结构依赖性的闭式方程。结果表明，KAN层在效率和准确性之间提供了有利的权衡，而符号代理模型则强调了轻量级部署和增强透明度的潜力。


### 论文摘要

Accurate prediction of flow delay is essential for optimizing and managing modern communication networks. We investigate three levels of modeling for this task. First, we implement a heterogeneous GNN with attention-based message passing, establishing a strong neural baseline. Second, we propose FlowKANet in which Kolmogorov-Arnold Networks replace standard MLP layers, reducing trainable parameters while maintaining competitive predictive performance. FlowKANet integrates KAMP-Attn (Kolmogorov-Arnold Message Passing with Attention), embedding KAN operators directly into message-passing and attention computation. Finally, we distill the model into symbolic surrogate models using block-wise regression, producing closed-form equations that eliminate trainable weights while preserving graph-structured dependencies. The results show that KAN layers provide a favorable trade-off between efficiency and accuracy and that symbolic surrogates emphasize the potential for lightweight deployment and enhanced transparency.

---

## 6. GraphFire-X: Physics-Informed Graph Attention Networks and Structural Gradient Boosting for Building-Scale Wildfire Preparedness at the Wildland-Urban Interface

**论文链接:** [http://arxiv.org/abs/2512.20813v1](http://arxiv.org/abs/2512.20813v1)

**作者:** Miguel Esparza, Vamshi Battal, Ali Mostafavi

**发布时间:** 2025-12-23

### GPT解析

### 总结

本研究提出了一种新型双专家集成框架，将野火风险脆弱性分解为环境传播和结构脆弱性两个向量，结合图神经网络和XGBoost模型，实现了对野火在城市环境中传播路径的精确预测和风险评估。

### 背景

随着野火日益演变成城市大火灾，传统风险模型将结构视为孤立资产，无法捕捉野地城市界面(WUI)特有的非线性传播动力学特性。

### 目的

填补机械物理学与数据驱动学习之间的空白，建立一种新的双专家集成框架，更准确地预测野火在城市环境中的传播路径，为社区韧性提供数据驱动的决策支持。

### 方法

建立双专家集成框架：环境专家使用图神经网络(GNN)实现，将社区视为有向传播图，权重基于物理信息化的对流、辐射和飞火概率，并融入高维Google AlphaEarth Foundation嵌入；结构专家通过XGBoost实现，隔离细粒度的资产级弹性；最后通过逻辑堆叠整合不同信号，生成诊断风险拓扑。

### 主要发现

应用于2025年伊顿火灾的框架揭示了风险驱动因素的关键二分法：社区规模的环境压力在定义传播路径方面压倒性地胜过内在结构特征；而屋檐是微观尺度上的主要侵入向量。

### 结论

通过整合环境传播和结构脆弱性两种信号，该模型使决策者能够超越二元损失预测，精确确定缓解措施优先级：对高连接集群进行植被管理，对建筑结构脆弱的节点进行结构加固，从而实施主动的、数据驱动的社区韧性方法。

### 翻译

随着野火日益演变成城市大火灾，传统风险模型将结构视为孤立资产，无法捕捉野地城市界面(WUI)特有的非线性传播动力学特性。本研究通过建立一种新的双专家集成框架弥合了机械物理学与数据驱动学习之间的差距，将脆弱性分解为环境传播和结构脆弱性两个不同的向量。该架构集成了两个专门的预测流：环境专家，实现为图神经网络(GNN)，将社区操作化为有向传播图，权重基于物理信息化的对流、辐射和飞火概率，并融入高维Google AlphaEarth Foundation嵌入；以及结构专家，通过XGBoost实现，隔离细粒度的资产级弹性。应用于2025年伊顿火灾，该框架揭示了风险驱动因素的关键二分法。GNN表明，社区规模的环境压力在定义传播路径方面压倒性地胜过内在结构特征，而XGBoost模型则发现屋檐是微观尺度上的主要侵入向量。通过逻辑堆叠整合这些不同的信号，集成模型实现了稳健的分类，并生成了诊断风险拓扑。这种能力使决策者能够超越二元损失预测，精确地确定缓解措施的优先级：对高连接集群进行植被管理，对建筑结构脆弱的节点进行结构加固，从而实施主动的、数据驱动的社区弹性方法。


### 论文摘要

As wildfires increasingly evolve into urban conflagrations, traditional risk models that treat structures as isolated assets fail to capture the non-linear contagion dynamics characteristic of the wildland urban interface (WUI). This research bridges the gap between mechanistic physics and data driven learning by establishing a novel dual specialist ensemble framework that disentangles vulnerability into two distinct vectors, environmental contagion and structural fragility. The architecture integrates two specialized predictive streams, an environmental specialist, implemented as a graph neural network (GNN) that operationalizes the community as a directed contagion graph weighted by physics informed convection, radiation, and ember probabilities, and enriched with high dimensional Google AlphaEarth Foundation embeddings, and a Structural Specialist, implemented via XGBoost to isolate granular asset level resilience. Applied to the 2025 Eaton Fire, the framework reveals a critical dichotomy in risk drivers. The GNN demonstrates that neighborhood scale environmental pressure overwhelmingly dominates intrinsic structural features in defining propagation pathways, while the XGBoost model identifies eaves as the primary micro scale ingress vector. By synthesizing these divergent signals through logistic stacking, the ensemble achieves robust classification and generates a diagnostic risk topology. This capability empowers decision makers to move beyond binary loss prediction and precisely target mitigation prioritizing vegetation management for high connectivity clusters and structural hardening for architecturally vulnerable nodes thereby operationalizing a proactive, data driven approach to community resilience.

---

## 7. Symbolic regression for defect interactions in 2D materials

**论文链接:** [http://arxiv.org/abs/2512.20785v1](http://arxiv.org/abs/2512.20785v1)

**作者:** Mikhail Lazarev, Andrey Ustyuzhanin

**发布时间:** 2025-12-23

### GPT解析

### 总结

机器学习模型已在所有科学领域广泛应用。神经网络方法虽能获得高精度，但存在缺点。符号回归技术可发现描述数据的解析方程，提供可解释和可推广的模型，随着神经网络技术进步获得新动力。本研究应用深度符号回归算法SEGVAE确定二维材料性质，并与图神经网络方法比较。

### 背景

机器学习模型已在所有科学领域得到广泛应用。神经网络方法虽能获得高精度，但存在一些缺点。符号回归技术因其可解释性和结果可推广性而具有优势，随着神经网络技术的发展，符号回归方法获得了新的动力。

### 目的

研究深度符号回归算法SEGVAE在确定具有缺陷的二维材料性质方面的应用，并与基于图神经网络的最先进方法进行比较。

### 方法

使用深度符号回归算法SEGVAE来确定具有缺陷的二维材料的性质，并将结果与基于图神经网络的最先进方法进行比较。

### 主要发现

SEGVAE算法与最先进的基于图神经网络的方法相比，显示出可比性，在某些情况下甚至产生了相同的结果。

### 结论

符号回归方法，特别是结合了神经网络技术的深度符号回归，在自然科学领域具有广泛的应用前景，特别是在需要可解释结果的场景中。

### 翻译

机器学习模型已在所有科学领域得到牢固确立。从数据中提取特征并使用神经网络模型基于这些特征进行推断通常能获得高精度；然而，这种方法有几个缺点。符号回归是一种强大的技术，可以发现描述数据的解析方程，提供可解释和可推广的模型，能够预测未见过的数据。随着神经网络技术的进步，符号回归方法获得了新的动力，并提供了几个优势，主要结果是结果的可解释性。在本工作中，我们研究了深度符号回归算法SEGVAE在确定具有缺陷的二维材料性质方面的应用。与最先进的基于图神经网络的方法相比，结果显示出可比性，在某些情况下甚至相同的结果。我们还讨论了这类方法在自然科学中的适用性。


### 论文摘要

Machine learning models have become firmly established across all scientific fields. Extracting features from data and making inferences based on them with neural network models often yields high accuracy; however, this approach has several drawbacks. Symbolic regression is a powerful technique for discovering analytical equations that describe data, providing interpretable and generalizable models capable of predicting unseen data. Symbolic regression methods have gained new momentum with the advancement of neural network technologies and offer several advantages, the main one being the interpretability of results. In this work, we examined the application of the deep symbolic regression algorithm SEGVAE to determine the properties of two-dimensional materials with defects. Comparing the results with state-of-the-art graph neural network-based methods shows comparable or, in some cases, even identical outcomes. We also discuss the applicability of this class of methods in natural sciences.

---

## 8. Fast SAM2 with Text-Driven Token Pruning

**论文链接:** [http://arxiv.org/abs/2512.21333v1](http://arxiv.org/abs/2512.21333v1)

**作者:** Avilasha Mandal, Chaoning Zhang, Fachrina Dewi Puspitasari, Xudong Wang, Jiaquan Zhang, Caiyan Qin, Guoqing Wang, Yang Yang, Heng Tao Shen

**发布时间:** 2025-12-24

**备注:** 28 pages, 9 figures

### GPT解析

### 总结

本文提出了一种文本引导的token修剪框架，通过选择性减少视觉token密度来提高SAM2模型的推理效率，同时保持分割性能。

### 背景

SAM2模型在提示驱动的视频对象分割方面取得显著进展，但其实际部署受限于处理密集视觉token的高计算和内存成本。SAM2管道通常传播所有视觉token，无论是否与目标对象相关，导致可扩展性降低。

### 目的

开发一种方法提高SAM2模型的推理效率，减少计算和内存需求，同时保持分割质量，使其更适合实时和资源受限的应用。

### 方法

引入文本引导的token修剪框架，在视觉编码后和基于内存的传播前操作。使用轻量级路由机制对token进行排序，该机制整合局部视觉上下文、从文本描述派生的语义相关性以及不确定性提示，保留最具信息量的token进行下游处理。

### 主要发现

在多个视频分割基准上的实验表明，该方法与未修剪的基线SAM2相比，实现了42.50%的更快推理速度和37.41%的更低GPU内存使用，同时保持了有竞争力的J和F性能。

### 结论

早期token选择显著提高了基于transformer的视频分割系统的可扩展性，为实时和资源受限应用中的高效视频分割提供了实用且有效的途径。

### 翻译

Segment Anything Model 2 (SAM2)作为一种视觉基础模型，在提示驱动的视频对象分割方面取得了显著进展，然而其实际部署仍受限于处理密集视觉token的高计算和内存成本。SAM2管道通常将图像编码器产生的所有视觉token传播到下游时间推理模块，无论它们是否与目标对象相关，导致由于二次内存注意力开销而降低了可扩展性。在这项工作中，我们引入了一个文本引导的token修剪框架，通过在时间传播前选择性减少token密度来提高推理效率，同时不修改底层分割架构。在视觉编码后和基于内存的传播前操作，我们的方法使用轻量级路由机制对token进行排序，该机制集成了局部视觉上下文、从以对象为中心的文本描述中派生的语义相关性（用户提供或自动生成）以及不确定性提示，有助于保留模糊或边界关键区域。通过仅保留最具信息量的token进行下游处理，所提出的方法减少了冗余计算，同时保持了分割保真度。在多个具有挑战性的视频分割基准上的广泛实验表明，编码器后token修剪提供了一种实用且高效的提示感知视频分割途径，与未修剪的基线SAM2相比，实现了42.50%的更快推理速度和37.41%的更低GPU内存使用，同时保持了有竞争力的J和F性能。这些结果突出了早期token选择在提高基于transformer的视频分割系统可扩展性方面的潜力，适用于实时和资源受限应用。


### 论文摘要

Segment Anything Model 2 (SAM2), a vision foundation model has significantly advanced in prompt-driven video object segmentation, yet their practical deployment remains limited by the high computational and memory cost of processing dense visual tokens across time. The SAM2 pipelines typically propagate all visual tokens produced by the image encoder through downstream temporal reasoning modules, regardless of their relevance to the target object, resulting in reduced scalability due to quadratic memory attention overhead. In this work, we introduce a text-guided token pruning framework that improves inference efficiency by selectively reducing token density prior to temporal propagation, without modifying the underlying segmentation architecture. Operating after visual encoding and before memory based propagation, our method ranks tokens using a lightweight routing mechanism that integrates local visual context, semantic relevance derived from object-centric textual descriptions (either user-provided or automatically generated), and uncertainty cues that help preserve ambiguous or boundary critical regions. By retaining only the most informative tokens for downstream processing, the proposed approach reduces redundant computation while maintaining segmentation fidelity. Extensive experiments across multiple challenging video segmentation benchmarks demonstrate that post-encoder token pruning provides a practical and effective pathway to efficient, prompt-aware video segmentation, achieving up to 42.50 percent faster inference and 37.41 percent lower GPU memory usage compared to the unpruned baseline SAM2, while preserving competitive J and F performance. These results highlight the potential of early token selection to improve the scalability of transformer-based video segmentation systems for real-time and resource-constrained applications.

---

## 9. TICON: A Slide-Level Tile Contextualizer for Histopathology Representation Learning

**论文链接:** [http://arxiv.org/abs/2512.21331v1](http://arxiv.org/abs/2512.21331v1)

**作者:** Varun Belagali, Saarthak Kapse, Pierre Marza, Srijan Das, Zilinghan Li, Sofiène Boutaj, Pushpak Pati, Srikar Yellapragada, Tarak Nath Nandi, Ravi K Madduri, Joel Saltz, Prateek Prasanna, Stergios Christodoulidis Maria Vakalopoulou, Dimitris Samaras

**发布时间:** 2025-12-24

### GPT解析

### 总结

TICON是一种基于Transformer的瓦片表示上下文化器，能够为计算病理学中的各种应用生成丰富、上下文化的嵌入，显著提高了多个任务的性能，并在多个基准测试上建立了新的最先进结果。

### 背景

在计算病理学中，解释大型全幻灯片图像(WSI)中的小瓦片通常需要更大的图像上下文，但标准的瓦片编码器管道无法建模幻灯片级别的丰富信息，且不同瓦片编码器在不同下游任务上表现各异。

### 目的

开发一个统一的模型来上下文化来自任何瓦片级基础模型的嵌入，解决现有方法无法充分利用幻灯片级信息的问题。

### 方法

TICON使用单个共享编码器，通过掩模建模目标进行预训练，同时统一和上下文化来自多样化瓦片级病理学基础模型的表示，并在TICON上预训练聚合器形成幻灯片级基础模型。

### 主要发现

TICON上下文化的嵌入显著提高了多个不同任务上的性能，在瓦片级基准测试(HEST-Bench, THUNDER, CATCH)和幻灯片级基准测试(Patho-Bench)上建立了新的最先进结果；仅使用11K个WSI预训练的幻灯片级基础模型就超过了使用多达350K个WSI预训练的最先进模型。

### 结论

TICON有效解决了计算病理学中瓦片表示缺乏上下文信息的问题，提供了一个强大而高效的解决方案，能够统一和增强来自不同瓦片编码器的表示，在各种病理学任务中取得了显著性能提升。

### 翻译

在大型全幻灯片图像(WSI)中解释小瓦片通常需要更大的图像上下文。我们引入了TICON，一种基于Transformer的瓦片表示上下文化器，它能为计算病理学中的'任何'应用程序生成丰富、上下文化的嵌入。标准的基于瓦片编码器的管道提取的是剥离了上下文的瓦片嵌入，无法建模对本地和全局任务都至关重要的幻灯片级别丰富信息。此外，不同的瓦片编码器在不同的下游任务上表现出色。因此，需要一个统一的模型来上下文化来自'任何'瓦片级基础模型的嵌入。TICON通过使用单个共享编码器满足了这一需求，该编码器使用掩模建模目标进行预训练，同时统一和上下文化来自多样化瓦片级病理学基础模型的表示。我们的实验证明，TICON上下文化的嵌入显著提高了许多不同任务的性能，在瓦片级基准测试(即HEST-Bench, THUNDER, CATCH)和幻灯片级基准测试(即Patho-Bench)上建立了新的最先进结果。最后，我们在TICON上预训练了一个聚合器，形成一个幻灯片级基础模型，仅使用11K个WSI，就超过了使用多达350K个WSI预训练的最先进幻灯片级基础模型。


### 论文摘要

The interpretation of small tiles in large whole slide images (WSI) often needs a larger image context. We introduce TICON, a transformer-based tile representation contextualizer that produces rich, contextualized embeddings for ''any'' application in computational pathology. Standard tile encoder-based pipelines, which extract embeddings of tiles stripped from their context, fail to model the rich slide-level information essential for both local and global tasks. Furthermore, different tile-encoders excel at different downstream tasks. Therefore, a unified model is needed to contextualize embeddings derived from ''any'' tile-level foundation model. TICON addresses this need with a single, shared encoder, pretrained using a masked modeling objective to simultaneously unify and contextualize representations from diverse tile-level pathology foundation models. Our experiments demonstrate that TICON-contextualized embeddings significantly improve performance across many different tasks, establishing new state-of-the-art results on tile-level benchmarks (i.e., HEST-Bench, THUNDER, CATCH) and slide-level benchmarks (i.e., Patho-Bench). Finally, we pretrain an aggregator on TICON to form a slide-level foundation model, using only 11K WSIs, outperforming SoTA slide-level foundation models pretrained with up to 350K WSIs.

---

## 10. Surgical Scene Segmentation using a Spike-Driven Video Transformer with Real-Time Potential

**论文链接:** [http://arxiv.org/abs/2512.21284v1](http://arxiv.org/abs/2512.21284v1)

**作者:** Shihao Zou, Jingjing Li, Wei Ji, Jincai Huang, Kai Wang, Guo Dan, Weixin Si, Yi Pan

**发布时间:** 2025-12-24

### GPT解析

### 总结

本文提出了SpikeSurgSeg，第一个为手术场景分割设计的脉冲驱动的视频Transformer框架，在非GPU平台上实现了实时潜力，同时保持了高分割精度。

### 背景

现代外科手术系统依赖智能场景理解提供及时感知，手术场景分割是核心。尽管深度学习模型取得显著精度，但其计算需求和功耗阻碍了在资源受限环境中的实时部署。

### 目的

解决深度学习模型在资源受限环境中实时部署的问题，探索SNN作为高效外科智能的范式。

### 方法

提出SpikeSurgSeg框架，采用外科场景掩码自编码预训练策略解决标注数据有限问题，并通过轻量级脉冲驱动的分割头保持低延迟特性。

### 主要发现

在EndoVis18和SurgBleed数据集上，SpikeSurgSeg实现了与最先进ANN模型相当的mIoU，推理延迟减少至少8倍，比大多数基础模型基线加速超过20倍。

### 结论

SpikeSurgSeg在时间关键的手术场景分割中显示出显著潜力。

### 翻译

现代外科手术系统越来越依赖智能场景理解来提供及时的场景感知，以提高手术中的安全性。在这个流程中，手术场景分割在准确感知手术事件方面起着核心作用。虽然最近的深度学习模型，特别是大规模基础模型，取得了显著的分割精度，但其巨大的计算需求和功耗阻碍了在资源受限的手术环境中的实时部署。为解决这一限制，我们探索了新兴的SNN作为一种有前途的高效外科智能范式。然而，其性能仍然受到标注外科数据稀缺性和外科视频表示固有稀疏性的限制。为此，我们提出了SpikeSurgSeg，这是第一个为手术场景分割量身定制的脉冲驱动的视频Transformer框架，具有在非GPU平台上实时运行的潜力。为解决外科标注可用性有限的问题，我们引入了一种针对SNN的外科场景掩码自编码预训练策略，通过逐层管状掩码实现强大的时空表征学习。基于这个预训练骨干网络，我们进一步采用了一个轻量级脉冲驱动的分割头，在保持SNN低延迟特性的同时产生时间一致的预测。在EndoVis18和我们内部的SurgBleed数据集上的大量实验表明，SpikeSurgSeg实现了与最先进的基于ANN的模型相当的mIoU，同时将推理延迟减少了至少8倍。值得注意的是，与大多数基础模型基线相比，它提供了超过20倍的加速，凸显了其在时间关键的手术场景分割中的潜力。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决如何在资源受限的手术环境中实现高效且实时的手术场景分割问题。这个问题在现实中非常重要，因为现代手术系统依赖智能场景理解来提供及时的环境感知以提高手术安全性，而现有的深度学习模型虽然精度高，但计算需求和功耗大，难以在手术环境中实时部署。手术环境有物理和安全限制（如空间有限、散热限制、电源预算），使得高端GPU集群无法使用，而仅使用CPU或嵌入式平台又会导致高延迟，阻碍了智能手术系统在临床实践中的应用。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先认识到手术场景分割面临的主要障碍是计算效率和实时性能的限制，特别是在资源受限的环境中。他们探索了脑启发计算范式，特别关注脉冲神经网络（SNN），因为其通过模仿大脑神经元动态和二进制脉冲通信实现高效计算。针对SNN在手术场景应用中的两大挑战（标注数据稀缺和领域特定特征捕捉不足），作者设计了SpikeSurgSeg框架，结合了手术场景掩码自动编码预训练策略和轻量级脉冲驱动分割头。该方法借鉴了现有工作包括SNN基本理论、Transformer架构、掩码视觉建模、自监督学习、知识蒸馏和特征金字塔网络等技术，但进行了针对性的创新和整合。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是利用脉冲神经网络（SNN）的脑启发计算范式实现高效、实时的手术场景分割，通过模仿大脑神经元动态和二进制脉冲通信实现仅加法的前向计算，结合自监督学习和知识蒸馏解决SNN在手术场景中的数据稀缺和表征能力不足问题。整体实现流程分为三个阶段：1）脉冲驱动视频编码器：结合CNN块和时空Transformer块；2）手术场景掩码自动编码预训练：应用层级管状掩码和知识蒸馏；3）手术视频分割微调：集成记忆读出模块和特征金字塔网络，使用交叉熵和焦点损失进行训练。整个流程在非GPU平台上实现低延迟和低能耗的实时分割。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1）SpikeSurgSeg框架：首个专为手术场景分割设计的SNN框架；2）手术场景掩码自动编码预训练策略：通过层级管状掩码解决数据稀缺问题；3）轻量级脉冲驱动分割头：确保时间一致性和低延迟；4）语义知识蒸馏：增强SNN的语义理解能力。相比之前的工作，不同之处在于：实现了与ANN模型相当的分割精度但显著降低延迟和能耗；不需要用户提示输入；解决了SNN在手术数据方面的挑战；结合了时空建模能力扩展了SNN应用；保持了SNN的仅加法计算特性。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'SpikeSurgSeg通过创新的脉冲驱动视频Transformer和手术场景特定预训练，在非GPU平台上实现了与最先进ANN模型相当的手术分割精度，同时将推理延迟降低至少8倍，能耗降低5倍以上，为实时可部署的智能手术系统提供了实用解决方案。'}


### 论文摘要

Modern surgical systems increasingly rely on intelligent scene understanding to provide timely situational awareness for enhanced intra-operative safety. Within this pipeline, surgical scene segmentation plays a central role in accurately perceiving operative events. Although recent deep learning models, particularly large-scale foundation models, achieve remarkable segmentation accuracy, their substantial computational demands and power consumption hinder real-time deployment in resource-constrained surgical environments. To address this limitation, we explore the emerging SNN as a promising paradigm for highly efficient surgical intelligence. However, their performance is still constrained by the scarcity of labeled surgical data and the inherently sparse nature of surgical video representations. To this end, we propose \textit{SpikeSurgSeg}, the first spike-driven video Transformer framework tailored for surgical scene segmentation with real-time potential on non-GPU platforms. To address the limited availability of surgical annotations, we introduce a surgical-scene masked autoencoding pretraining strategy for SNNs that enables robust spatiotemporal representation learning via layer-wise tube masking. Building on this pretrained backbone, we further adopt a lightweight spike-driven segmentation head that produces temporally consistent predictions while preserving the low-latency characteristics of SNNs. Extensive experiments on EndoVis18 and our in-house SurgBleed dataset demonstrate that SpikeSurgSeg achieves mIoU comparable to SOTA ANN-based models while reducing inference latency by at least $8\times$. Notably, it delivers over $20\times$ acceleration relative to most foundation-model baselines, underscoring its potential for time-critical surgical scene segmentation.

---

## 11. Learning from Next-Frame Prediction: Autoregressive Video Modeling Encodes Effective Representations

**论文链接:** [http://arxiv.org/abs/2512.21004v1](http://arxiv.org/abs/2512.21004v1)

**作者:** Jinghan Li, Yang Jin, Hao Jiang, Yadong Mu, Yang Song, Kun Xu

**发布时间:** 2025-12-24

### GPT解析

### 总结

本研究提出了NExT-Vid，一种新型的自回归视觉生成预训练框架，通过掩码下一帧预测联合建模图像和视频，解决了现有视觉生成预训练方法中忽略时间信息和生成质量差的问题。

### 背景

预训练基础模型在多样化下游任务中显著提升了性能。虽然自回归生成模型如GPT在自然语言处理中取得了革命性进展，但大多数视觉生成预训练方法仍依赖BERT风格的掩码建模，忽视了视频分析中至关重要的时间信息。现有的自回归视觉预训练方法存在语义定位不准确和生成质量差等问题，导致语义表现不佳。

### 目的

开发一种能够有效建模图像和视频的自回归视觉生成预训练框架，解决现有方法中忽视时间信息和生成质量差的问题，提升视觉表示学习的能力。

### 方法

提出了NExT-Vid框架，利用掩码下一帧预测来联合建模图像和视频。该框架引入了上下文隔离的自回归预测器来解耦语义表示与目标解码，以及一个条件流匹配解码器来增强生成质量和多样性。通过上下文隔离的流匹配预训练，该方法获得了强大的表示能力。

### 主要发现

在大规模预训练模型上的广泛实验表明，所提出的方法通过注意力探测在下游分类任务中，始终优于之前的视觉表示学习的生成预训练方法。

### 结论

NExT-Vid通过创新的架构设计，成功地将自回归生成模型的优势引入视觉领域，特别是在处理视频数据时考虑了时间信息，显著提升了视觉表示学习的性能。

### 翻译

最近预训练通用基础模型的进展显著提高了多样化下游任务中的性能。虽然像GPT这样的自回归生成模型革新了自然语言处理，但大多数视觉生成预训练方法仍然依赖于BERT风格的掩码建模，这通常忽视了视频分析中必不可少的时间信息。少数现有的自回归视觉预训练方法存在语义定位不准确和生成质量差等问题，导致语义表现不佳。在这项工作中，我们提出了NExT-Vid，一种新型的自回归视觉生成预训练框架，利用掩码下一帧预测来联合建模图像和视频。NExT-Vid引入了上下文隔离的自回归预测器来解耦语义表示与目标解码，以及一个条件流匹配解码器来增强生成质量和多样性。通过上下文隔离的流匹配预训练，我们的方法获得了强大的表示能力。在大规模预训练模型上的广泛实验表明，我们提出的方法通过注意力探测在下游分类任务中，始终优于之前的用于视觉表示学习的生成预训练方法。


### 论文摘要

Recent advances in pretraining general foundation models have significantly improved performance across diverse downstream tasks. While autoregressive (AR) generative models like GPT have revolutionized NLP, most visual generative pretraining methods still rely on BERT-style masked modeling, which often disregards the temporal information essential for video analysis. The few existing autoregressive visual pretraining methods suffer from issues such as inaccurate semantic localization and poor generation quality, leading to poor semantics. In this work, we propose NExT-Vid, a novel autoregressive visual generative pretraining framework that utilizes masked next-frame prediction to jointly model images and videos. NExT-Vid introduces a context-isolated autoregressive predictor to decouple semantic representation from target decoding, and a conditioned flow-matching decoder to enhance generation quality and diversity. Through context-isolated flow-matching pretraining, our approach achieves strong representations. Extensive experiments on large-scale pretrained models demonstrate that our proposed method consistently outperforms previous generative pretraining methods for visual representation learning via attentive probing in downstream classification.

---

## 12. Deadline-Aware Online Scheduling for LLM Fine-Tuning with Spot Market Predictions

**论文链接:** [http://arxiv.org/abs/2512.20967v1](http://arxiv.org/abs/2512.20967v1)

**作者:** Linggao Kong, Yuedong Xu, Lei Jiao, Chuan Xu

**发布时间:** 2025-12-24

### GPT解析

### 总结

研究提出了一种结合现货和按需GPU实例的在线调度框架，用于降低大规模基础模型微调成本，通过预测算法和策略选择算法适应现货市场动态，实验表明可提高效用达54.8%。

### 背景

随着基础模型规模增长，微调成本日益增加。GPU现货实例虽为低成本替代方案，但其价格和可用性的波动性使截止时间感知调度面临挑战。

### 目的

解决现货实例调度难题，通过混合使用现货和按需实例实现成本高效的调度策略。

### 方法

1) 分析现货市场价格和可用性的可预测性；2) 建立整数规划模型捕捉混合实例使用；3) 提出基于预测的在线分配算法；4) 设计无预测的补充算法；5) 开发在线策略选择算法学习最佳策略。

### 主要发现

预测算法随误差减小可获更紧性能界限；策略选择算法具有O(√T)遗憾界限；在线框架能根据市场动态自适应选择最佳策略。

### 结论

提出的在线框架能持续优于基线方法，最多提高54.8%的效用，有效解决了基础模型微调中的成本优化问题。

### 翻译

随着基础模型规模的扩大，对其进行微调变得越来越昂贵。虽然GPU现货实例提供了按需资源的低成本替代方案，但其价格和可用性的波动使得具有截止时间意识的调度特别具有挑战性。我们通过混合使用现货和按需实例来解决这个问题。特别地，我们展示了现货实例市场中价格和可用性的可预测性，预测在实现成本高效调度方面的能力及其对估计误差的敏感性。我们制定了一个整数规划问题，以捕捉在价格和可用性动态下使用混合实例的情况。我们提出了一种基于预测的在线分配算法，采用基于提交范围控制的方法，利用提交级别来执行决策的部分序列。当预测不准确时，我们进一步提出了一种无需预测的补充在线算法。开发了一种在线策略选择算法，它通过变化两个算法的参数从构建的池中学习最佳策略。我们证明，随着预测误差的减小，基于预测的算法可以实现更紧的性能界限，而策略选择算法具有O(√T)的遗憾界限。实验结果表明，我们的在线框架可以根据变化的现货市场动态和预测质量自适应选择最佳策略，持续优于基线方法，最多提高54.8%的效用。


### 论文摘要

As foundation models grow in size, fine-tuning them becomes increasingly expensive. While GPU spot instances offer a low-cost alternative to on-demand resources, their volatile prices and availability make deadline-aware scheduling particularly challenging. We tackle this difficulty by using a mix of spot and on-demand instances. Distinctively, we show the predictability of prices and availability in a spot instance market, the power of prediction in enabling cost-efficient scheduling and its sensitivity to estimation errors. An integer programming problem is formulated to capture the use of mixed instances under both the price and availability dynamics. We propose an online allocation algorithm with prediction based on the committed horizon control approach that leverages a \emph{commitment level} to enforce the partial sequence of decisions. When this prediction becomes inaccurate, we further present a complementary online algorithm without predictions. An online policy selection algorithm is developed that learns the best policy from a pool constructed by varying the parameters of both algorithms. We prove that the prediction-based algorithm achieves tighter performance bounds as prediction error decreases, while the policy selection algorithm possesses a regret bound of $\mathcal{O}(\sqrt{T})$. Experimental results demonstrate that our online framework can adaptively select the best policy under varying spot market dynamics and prediction quality, consistently outperforming baselines and improving utility by up to 54.8\%.

---

## 13. Foundation Model-based Evaluation of Neuropsychiatric Disorders: A Lifespan-Inclusive, Multi-Modal, and Multi-Lingual Study

**论文链接:** [http://arxiv.org/abs/2512.20948v1](http://arxiv.org/abs/2512.20948v1)

**作者:** Zhongren Dong, Haotian Guo, Weixiang Xu, Huan Zhao, Zixing Zhang

**发布时间:** 2025-12-24

### GPT解析

### 总结

该研究提出了FEND框架，一个综合多模态系统，整合语音和文本模态用于跨生命周期的阿尔茨海默病、抑郁症和自闭症谱系障碍检测，并在13种多语言数据集上进行了系统评估。

### 背景

神经精神障碍如阿尔茨海默病、抑郁症和自闭症谱系障碍具有语言和声学异常特征，可作为早期检测的生物标志物。然而，多模态方法面临多语言泛化和缺乏统一评估框架等挑战。

### 目的

解决多语言泛化和缺乏统一评估框架的挑战，提出一个综合多模态框架用于检测跨生命周期的神经精神障碍。

### 方法

提出FEND框架，整合语音和文本模态。利用13种多语言数据集（英语、中文、希腊语、法语和荷兰语）系统评估多模态融合性能。

### 主要发现

多模态融合在AD和抑郁症检测中表现优异，但在ASD检测中表现不佳，原因是数据集异质性；模态不平衡是普遍问题，多模态融合未能超越最佳单模态模型；跨语料库实验显示在任务和语言一致场景中表现稳健，但在多语言和任务异构设置中性能下降。

### 结论

FEND通过提供广泛基准和性能影响因素详细分析，推动了自动化、包容全生命周期和多语言神经精神障碍评估领域发展，鼓励研究人员采用该框架进行公平比较和可重复研究。

### 翻译

神经精神障碍，如阿尔茨海默病、抑郁症和自闭症谱系障碍，其特征是语言和声学异常，为早期检测提供了潜在生物标志物。尽管多模态方法很有前景，但多语言泛化和缺乏统一评估框架等挑战仍然存在。为解决这些差距，我们提出了FEND，这是一个综合多模态框架，整合了语音和文本模态，用于跨生命周期检测AD、抑郁症和ASD。利用涵盖英语、中文、希腊语、法语和荷兰语的13种多语言数据集，我们系统评估了多模态融合性能。我们的结果表明，多模态融合在AD和抑郁症检测中表现出色，但由于数据集异质性，在ASD检测中表现不佳。我们还发现模态不平衡是一个普遍问题，多模态融合未能超越最佳单模态模型。跨语料库实验显示，在任务和语言一致的场景中表现稳健，但在多语言和任务异构设置中性能明显下降。


### 论文摘要

Neuropsychiatric disorders, such as Alzheimer's disease (AD), depression, and autism spectrum disorder (ASD), are characterized by linguistic and acoustic abnormalities, offering potential biomarkers for early detection. Despite the promise of multi-modal approaches, challenges like multi-lingual generalization and the absence of a unified evaluation framework persist. To address these gaps, we propose FEND (Foundation model-based Evaluation of Neuropsychiatric Disorders), a comprehensive multi-modal framework integrating speech and text modalities for detecting AD, depression, and ASD across the lifespan. Leveraging 13 multi-lingual datasets spanning English, Chinese, Greek, French, and Dutch, we systematically evaluate multi-modal fusion performance. Our results show that multi-modal fusion excels in AD and depression detection but underperforms in ASD due to dataset heterogeneity. We also identify modality imbalance as a prevalent issue, where multi-modal fusion fails to surpass the best mono-modal models. Cross-corpus experiments reveal robust performance in task- and language-consistent scenarios but noticeable degradation in multi-lingual and task-heterogeneous settings. By providing extensive benchmarks and a detailed analysis of performance-influencing factors, FEND advances the field of automated, lifespan-inclusive, and multi-lingual neuropsychiatric disorder assessment. We encourage researchers to adopt the FEND framework for fair comparisons and reproducible research.

---

## 14. Decoding Predictive Inference in Visual Language Processing via Spatiotemporal Neural Coherence

**论文链接:** [http://arxiv.org/abs/2512.20929v1](http://arxiv.org/abs/2512.20929v1)

**作者:** Sean C. Borneman, Julia Krebs, Ronnie B. Wilbur, Evie A. Malaia

**发布时间:** 2025-12-24

**备注:** 39th Conference on Neural Information Processing Systems (NeurIPS 2025) Workshop: Foundation Models for the Brain and Body

### GPT解析

### 总结

该研究开发了一种机器学习框架，通过分析聋人手语使用者的脑电图反应，揭示了大脑处理语言时的预测神经动力学机制，发现左侧半球和额叶的低频神经活动是语言理解的关键，且神经特征与年龄相关。

### 背景

人类语言处理依赖于大脑的预测推理能力，特别是聋人手语使用者处理动态视觉语言刺激的方式。

### 目的

开发一种机器学习框架，用于解码聋人对动态视觉语言刺激的神经（EEG）反应。

### 方法

使用神经信号和光流推导的运动特征之间的一致性构建预测神经动力学的时空表示，通过基于熵的特征选择，识别区分可解释语言输入和语言紊乱刺激的频率特异性神经特征。

### 主要发现

结果显示，左侧半球和额叶低频一致性是语言理解的关键特征，经验依赖性神经特征与年龄相关。

### 结论

这项工作展示了一种新颖的多模态方法，用于探索大脑中经验驱动的感知生成模型。

### 翻译

人类语言处理依赖于大脑的预测推理能力。我们提出了一种机器学习框架，用于解码聋人手语使用者对动态视觉语言刺激的神经（EEG）反应。利用神经信号与光流推导的运动特征之间的一致性，我们构建了预测神经动力学的时空表示。通过基于熵的特征选择，我们能够区分可解释的语言输入和语言紊乱（时间反转）刺激的频率特异性神经特征。我们的研究结果表明，左侧半球和额叶的低频一致性是语言理解的关键特征，且经验依赖性的神经特征与年龄相关。这项工作展示了一种新颖的多模态方法，用于探索大脑中经验驱动的感知生成模型。


### 论文摘要

Human language processing relies on the brain's capacity for predictive inference. We present a machine learning framework for decoding neural (EEG) responses to dynamic visual language stimuli in Deaf signers. Using coherence between neural signals and optical flow-derived motion features, we construct spatiotemporal representations of predictive neural dynamics. Through entropy-based feature selection, we identify frequency-specific neural signatures that differentiate interpretable linguistic input from linguistically disrupted (time-reversed) stimuli. Our results reveal distributed left-hemispheric and frontal low-frequency coherence as key features in language comprehension, with experience-dependent neural signatures correlating with age. This work demonstrates a novel multimodal approach for probing experience-driven generative models of perception in the brain.

---

## 15. Beyond Weight Adaptation: Feature-Space Domain Injection for Cross-Modal Ship Re-Identification

**论文链接:** [http://arxiv.org/abs/2512.20892v1](http://arxiv.org/abs/2512.20892v1)

**作者:** Tingfeng Xian, Wenlve Zhou, Zhiheng Zhou, Zhelin Li

**发布时间:** 2025-12-24

### GPT解析

### 总结

本文提出了一种名为领域表示注入(DRI)的新型参数高效微调策略，用于解决跨模态船舶重识别中的模态差异问题。该方法通过视觉基础模型和轻量级偏移编码器实现，在保持模型预训练权重不变的情况下，有效提取和注入领域特定表示，实现了最先进的性能。

### 背景

跨模态船舶重识别(CMS Re-ID)对实现全天候海事目标跟踪至关重要，但面临显著的模态差异挑战。主流解决方案依赖显式模态对齐策略，这严重依赖于构建大规模配对数据集进行预训练。

### 目的

解决跨模态船舶重识别中模态差异导致的挑战，减少对大规模配对数据集的依赖，提高模型在有限参数下的性能。

### 方法

基于柏拉图表示假设，将优化视角从权重空间转向特征空间，提出领域表示注入(DRI)策略。具体包括：保持视觉基础模型(VFM)完全冻结；设计轻量级偏移编码器提取领域特定表示；通过调制器自适应转换这些表示；通过加性融合将表示注入中间层，动态重塑特征分布以适应下游任务。

### 主要发现

DRI方法在HOSS-ReID数据集上实现了最先进的性能，仅使用1.54M和7.05M参数分别达到57.9%和60.5%的mAP。该方法有效弥合了模态差距，同时保持了模型的通用知识。

### 结论

领域表示注入(DRI)是一种有效的跨模态船舶重识别方法，通过在特征空间进行优化，实现了高性能与低参数消耗的平衡，为解决模态差异问题提供了新思路。

### 翻译

跨模态船舶重识别(CMS Re-ID)对于实现全天候和全天气海事目标跟踪至关重要，但其根本上受到显著模态差异的挑战。主流解决方案通常依赖显式模态对齐策略；然而，这种范式严重依赖于构建大规模配对数据集进行预训练。为此，基于柏拉图表示假设，我们探索了视觉基础模型(VFMs)在弥合模态差距方面的潜力。认识到现有通用的参数高效微调(PEFT)方法在权重空间操作的表现不佳，尤其是在有限容量模型上，我们将优化视角转向特征空间，并提出了一种称为领域表示注入(DRI)的新型PEFT策略。具体而言，在保持VFM完全冻结以最大化保留通用知识的同时，我们设计了一个轻量级、可学习的偏移编码器，从原始输入中提取富含模态和身份属性的领域特定表示。根据不同层中间特征的上下文信息，调制器自适应地转换这些表示。随后，它们通过加性融合注入到中间层，动态重塑特征分布以适应下游任务，同时不改变VFM的预训练权重。大量实验结果表明我们方法的优越性，仅使用最少的可训练参数就实现了最先进的(SOTA)性能。例如，在HOSS-ReID数据集上，我们仅使用1.54M和7.05M参数，分别实现了57.9%和60.5%的mAP。代码可在https://github.com/TingfengXian/DRI获取。


### 论文摘要

Cross-Modality Ship Re-Identification (CMS Re-ID) is critical for achieving all-day and all-weather maritime target tracking, yet it is fundamentally challenged by significant modality discrepancies. Mainstream solutions typically rely on explicit modality alignment strategies; however, this paradigm heavily depends on constructing large-scale paired datasets for pre-training. To address this, grounded in the Platonic Representation Hypothesis, we explore the potential of Vision Foundation Models (VFMs) in bridging modality gaps. Recognizing the suboptimal performance of existing generic Parameter-Efficient Fine-Tuning (PEFT) methods that operate within the weight space, particularly on limited-capacity models, we shift the optimization perspective to the feature space and propose a novel PEFT strategy termed Domain Representation Injection (DRI). Specifically, while keeping the VFM fully frozen to maximize the preservation of general knowledge, we design a lightweight, learnable Offset Encoder to extract domain-specific representations rich in modality and identity attributes from raw inputs. Guided by the contextual information of intermediate features at different layers, a Modulator adaptively transforms these representations. Subsequently, they are injected into the intermediate layers via additive fusion, dynamically reshaping the feature distribution to adapt to the downstream task without altering the VFM's pre-trained weights. Extensive experimental results demonstrate the superiority of our method, achieving State-of-the-Art (SOTA) performance with minimal trainable parameters. For instance, on the HOSS-ReID dataset, we attain 57.9\% and 60.5\% mAP using only 1.54M and 7.05M parameters, respectively. The code is available at https://github.com/TingfengXian/DRI.

---

## 16. Proprioception Enhances Vision Language Model in Generating Captions and Subtask Segmentations for Robot Task

**论文链接:** [http://arxiv.org/abs/2512.20876v1](http://arxiv.org/abs/2512.20876v1)

**作者:** Kanata Suzuki, Shota Shimizu, Tetsuya Ogata

**发布时间:** 2025-12-24

### GPT解析

### 总结

本研究评估了视觉语言模型(VLMs)在理解机器人运动方面的能力，通过视频字幕任务测试了两种能力：机器人任务自动字幕生成和任务序列分割。研究提出了一种结合图像字幕和机器人轨迹数据的方法，以提高VLMs在机器人任务理解和分割方面的性能。

### 背景

从机器人未来发展角度看，验证仅通过离线数据（如图像和语言）训练的基础模型能否理解机器人运动至关重要。特别是，视觉语言模型(VLMs)的训练数据中不包含机器人的低级运动信息，因此包含轨迹信息的视频理解仍是一个重大挑战。

### 目的

评估VLMs通过包含低级机器人运动信息的视频字幕任务的两种能力：(1)机器人任务自动字幕生成和(2)任务序列分割。这两种能力旨在通过连接语言和运动来提高机器人模仿学习效率，并作为基础模型性能的衡量标准。

### 方法

提出的方法利用机器人任务的图像字幕和轨迹数据生成多个'场景'字幕，然后总结这些单个字幕生成完整任务字幕。此外，通过比较图像字幕的文本嵌入相似性执行子任务分割。在两种字幕任务中，将机器人运动数据（关节和末端执行器状态）作为输入提供给VLMs以提高性能。

### 主要发现

通过模拟实验验证了所提出方法的有效性，表明提供机器人运动数据作为输入可以增强VLMs在机器人任务理解和分割方面的能力。

### 结论

结合机器人运动数据与视觉语言模型可以改善对机器人任务的理解和分割，为机器人模仿学习提供了新的可能性。

### 翻译

从机器人未来发展的角度来看，验证仅离线数据（如图像和语言）训练的基础模型能否理解机器人运动至关重要。特别是，视觉语言模型(VLMs)在其训练数据中不包括机器人的低级运动信息，因此包含轨迹信息的视频理解仍然是一个重大挑战。在本研究中，我们通过包含低级机器人运动信息的视频字幕任务评估了VLMs的两种能力：(1)机器人任务自动字幕生成和(2)任务序列分割。这两种能力都旨在通过连接语言和运动来提高机器人模仿学习的效率，并作为基础模型性能的衡量标准。所提出的方法利用机器人任务的图像字幕和轨迹数据生成多个'场景'字幕，然后通过总结这些单个字幕生成完整任务字幕。此外，该方法通过比较图像字幕的文本嵌入相似性执行子任务分割。在两种字幕任务中，该方法旨在通过将机器人运动数据（关节和末端执行器状态）作为输入提供给VLMs来提高性能。进行了模拟实验以验证所提出方法的有效性。


### 论文摘要

From the perspective of future developments in robotics, it is crucial to verify whether foundation models trained exclusively on offline data, such as images and language, can understand the robot motion. In particular, since Vision Language Models (VLMs) do not include low-level motion information from robots in their training datasets, video understanding including trajectory information remains a significant challenge. In this study, we assess two capabilities of VLMs through a video captioning task with low-level robot motion information: (1) automatic captioning of robot tasks and (2) segmentation of a series of tasks. Both capabilities are expected to enhance the efficiency of robot imitation learning by linking language and motion and serve as a measure of the foundation model's performance. The proposed method generates multiple "scene" captions using image captions and trajectory data from robot tasks. The full task caption is then generated by summarizing these individual captions. Additionally, the method performs subtask segmentation by comparing the similarity between text embeddings of image captions. In both captioning tasks, the proposed method aims to improve performance by providing the robot's motion data - joint and end-effector states - as input to the VLM. Simulator experiments were conducted to validate the effectiveness of the proposed method.

---

## 17. Memory-Efficient Acceleration of Block Low-Rank Foundation Models on Resource Constrained GPUs

**论文链接:** [http://arxiv.org/abs/2512.20861v1](http://arxiv.org/abs/2512.20861v1)

**作者:** Pierre Abillama, Changwoo Lee, Juechu Dong, David Blaauw, Dennis Sylvester, Hun-Seok Kim

**发布时间:** 2025-12-24

### GPT解析

### 总结

本研究针对基于Transformer的基础模型在GPU上部署的内存和计算效率问题，通过优化的块低秩(BLR)压缩技术实现了显著加速和模型压缩。

### 背景

基于Transformer的基础模型已成为许多任务的默认选择，但其快速增长使得在单个GPU上完整运行变得困难，计算成本过高。

### 目的

解决BLR方法在多令牌推理中面临的内存限制问题，提高计算效率。

### 方法

引入带有部分融合和内存布局优化的自定义Triton内核，应用于Monarch和BLAST两种BLR方法。

### 主要发现

在内存受限的NVIDIA GPU上，优化后的内核相比PyTorch密集基线实现了高达3.76倍的加速和3倍模型大小压缩，同时支持多种模型如Llama-7/1B、GPT2-S、DiT-XL/2和ViT-B。

### 结论

通过优化的BLR方法和自定义Triton内核，可以在保持模型精度的同时显著提高内存效率和计算速度。

### 翻译

最近基于Transformer的基础模型的进展使它们成为许多任务的默认选择，但它们快速增长的大小使得在单个GPU上完整运行模型变得越来越困难，且计算成本过高。块低秩(BLR)压缩技术通过学习权重矩阵的紧凑表示来应对这一挑战。虽然传统低秩(LR)方法通常会导致精度急剧下降，但Monarch和BLAST等BLR方法能更好地捕捉底层结构，从而在减少计算和内存占用的同时保持精度。在这项工作中，我们使用roofline分析表明，尽管BLR方法在单令牌推理中实现了理论节省和实际加速，但多令牌推理在实践中往往成为内存限制，尽管在PyTorch中进行了编译器级别的优化。为解决这一问题，我们为Monarch和BLAST引入了带有部分融合和内存布局优化的自定义Triton内核。在内存受限的NVIDIA GPU（如Jetson Orin Nano和A40）上，我们的内核相比使用CUDA后端和编译器级优化的PyTorch密集基线，实现了高达3.76倍的加速和3倍模型大小压缩，同时支持包括Llama-7/1B、GPT2-S、DiT-XL/2和ViT-B在内的各种模型。我们的代码可在https://github.com/pabillam/mem-efficient-blr获取。


### 论文摘要

Recent advances in transformer-based foundation models have made them the default choice for many tasks, but their rapidly growing size makes fitting a full model on a single GPU increasingly difficult and their computational cost prohibitive. Block low-rank (BLR) compression techniques address this challenge by learning compact representations of weight matrices. While traditional low-rank (LR) methods often incur sharp accuracy drops, BLR approaches such as Monarch and BLAST can better capture the underlying structure, thus preserving accuracy while reducing computations and memory footprints. In this work, we use roofline analysis to show that, although BLR methods achieve theoretical savings and practical speedups for single-token inference, multi-token inference often becomes memory-bound in practice, increasing latency despite compiler-level optimizations in PyTorch. To address this, we introduce custom Triton kernels with partial fusion and memory layout optimizations for both Monarch and BLAST. On memory-constrained NVIDIA GPUs such as Jetson Orin Nano and A40, our kernels deliver up to $3.76\times$ speedups and $3\times$ model size compression over PyTorch dense baselines using CUDA backend and compiler-level optimizations, while supporting various models including Llama-7/1B, GPT2-S, DiT-XL/2, and ViT-B. Our code is available at https://github.com/pabillam/mem-efficient-blr .

---

## 18. TS-Arena Technical Report -- A Pre-registered Live Forecasting Platform

**论文链接:** [http://arxiv.org/abs/2512.20761v1](http://arxiv.org/abs/2512.20761v1)

**作者:** Marcel Meyer, Sascha Kaltenpoth, Kevin Zalipski, Henrik Albers, Oliver Müller

**发布时间:** 2025-12-23

### GPT解析

### 总结

时间序列基础模型(TSFMs)在预测方面具有变革性能力，但同时也引发了评估危机，主要源于信息泄露和全局模式的非法转移。

### 背景

时间序列基础模型能够学习共享的时间动态，这是它们的主要优势。然而，它们在历史档案上的评估往往允许利用观察到的全局冲击，这违反了有效基准测试所需的独立性。

### 目的

引入TS-Arena平台，通过将真正未知的未来作为 definitive 测试环境，恢复预测的操作完整性。

### 方法

通过在实时数据流上实施预注册机制，确保评估目标在推理期间物理上不存在，从而强制执行严格的全局时间分割。这种方法建立了一个移动的时间前沿，防止历史污染并提供模型泛化的真实评估。

### 主要发现

时间序列基础模型的评估存在问题，包括不同模型间训练和测试集重叠导致的信息泄露，以及全局模式向测试数据的非法转移。

### 结论

TS-Arena平台为在真实世界约束下比较基础模型提供了可持续的基础设施，最初应用于能源部门，确保了评估的完整性和真实性。

### 翻译

虽然时间序列基础模型(TSFMs)为预测提供了变革性能力，但它们同时也可能引发根本性的评估危机。这种危机是由不同模型间训练和测试集重叠导致的信息泄露，以及全局模式向测试数据的非法转移所驱动的。虽然学习共享时间动态的能力代表了这些模型的主要优势，但它们在历史档案上的评估往往允许利用观察到的全局冲击，这违反了有效基准测试所需的独立性。我们引入了TS-Arena平台，它通过将真正未知的未来作为 definitive 测试环境，恢复了预测的操作完整性。通过在实时数据流上实施预注册机制，该平台确保评估目标在推理期间物理上不存在，从而强制执行严格的全局时间分割。这种方法建立了一个移动的时间前沿，防止历史污染并提供模型泛化的真实评估。最初应用于能源部门，TS-Arena为在真实世界约束下比较基础模型提供了可持续的基础设施。该平台的原型可在 https://huggingface.co/spaces/DAG-UPB/TS-Arena 获取。


### 论文摘要

While Time Series Foundation Models (TSFMs) offer transformative capabilities for forecasting, they simultaneously risk triggering a fundamental evaluation crisis. This crisis is driven by information leakage due to overlapping training and test sets across different models, as well as the illegitimate transfer of global patterns to test data. While the ability to learn shared temporal dynamics represents a primary strength of these models, their evaluation on historical archives often permits the exploitation of observed global shocks, which violates the independence required for valid benchmarking. We introduce TS-Arena, a platform that restores the operational integrity of forecasting by treating the genuinely unknown future as the definitive test environment. By implementing a pre-registration mechanism on live data streams, the platform ensures that evaluation targets remain physically non-existent during inference, thereby enforcing a strict global temporal split. This methodology establishes a moving temporal frontier that prevents historical contamination and provides an authentic assessment of model generalization. Initially applied within the energy sector, TS-Arena provides a sustainable infrastructure for comparing foundation models under real-world constraints. A prototype of the platform is available at https://huggingface.co/spaces/DAG-UPB/TS-Arena.

---

## 19. Emergent temporal abstractions in autoregressive models enable hierarchical reinforcement learning

**论文链接:** [http://arxiv.org/abs/2512.20605v2](http://arxiv.org/abs/2512.20605v2)

**作者:** Seijin Kobayashi, Yanick Schimpf, Maximilian Schlegel, Angelika Steger, Maciej Wolczyk, Johannes von Oswald, Nino Scherrer, Kaitlin Maile, Guillaume Lajoie, Blake A. Richards, Rif A. Saurous, James Manyika, Blaise Agüera y Arcas, Alexander Meulemans, João Sacramento

**发布时间:** 2025-12-23

### GPT解析

### 总结

本文提出了一种名为'内部RL'的方法，通过在自回归模型的内部表示中进行动作探索，解决了逐个标记采样导致的学习效率低下问题，特别是在奖励稀疏的情况下。

### 背景

大型自回归模型通过下一词预测预训练并使用强化学习微调，在许多问题领域取得了前所未有的成功。然而，在RL过程中，逐个标记采样动作可能导致学习效率低下，特别是在奖励稀疏的情况下。

### 目的

克服逐个标记采样导致的学习效率低下问题，特别是在奖励稀疏的情况下。

### 方法

在自回归模型的内部表示中进行动作探索；引入一个高阶非因果序列模型，其输出控制基础自回归模型的残差流激活；该高阶模型学习将长激活序列块压缩到内部控制器上。

### 主要发现

在具有层次结构的网格世界和基于MuJoCo的任务中，高阶模型学习将长激活序列块压缩到内部控制器上；每个控制器执行一系列行为上有意义的动作，这些动作在长时间尺度上展开，并带有学习到的终止条件；通过组合多个控制器可以实现在新任务上的高效探索；'内部RL'在标准RL微调失败的情况下，能够从稀疏奖励中学习。

### 结论

自回归模型中的潜在动作生成和强化具有优势；'内部RL'是在基础模型中实现层次化RL的有前途的方向。

### 翻译

大型基于下一词预测预训练并通过强化学习(RL)微调的自回归模型在许多问题领域取得了前所未有的成功。在RL期间，这些模型通过一次生成一个新标记来进行探索。然而，逐个标记采样动作可能导致学习效率低下，特别是在奖励稀疏的情况下。在这里，我们表明通过在自回归模型的内部表示中进行动作和探索可以克服这个问题。具体来说，为了发现时间上抽象的动作，我们引入了一个高阶非因果序列模型，其输出控制基础自回归模型的残差流激活。在具有层次结构的网格世界和基于MuJoCo的任务中，我们发现高阶模型学习将长激活序列块压缩到内部控制器上。关键的是，每个控制器执行一系列行为上有意义的动作，这些动作在长时间尺度上展开，并带有学习到的终止条件，使得随时间组合多个控制器能够在新任务上实现高效探索。我们表明，直接内部控制器强化，这一过程我们称为'内部RL'，能够在标准RL微调失败的情况下从稀疏奖励中学习。我们的结果证明了自回归模型中潜在动作生成和强化的优势，表明内部RL是在基础模型中实现层次化RL的有前途的途径。


### 论文摘要

Large-scale autoregressive models pretrained on next-token prediction and finetuned with reinforcement learning (RL) have achieved unprecedented success on many problem domains. During RL, these models explore by generating new outputs, one token at a time. However, sampling actions token-by-token can result in highly inefficient learning, particularly when rewards are sparse. Here, we show that it is possible to overcome this problem by acting and exploring within the internal representations of an autoregressive model. Specifically, to discover temporally-abstract actions, we introduce a higher-order, non-causal sequence model whose outputs control the residual stream activations of a base autoregressive model. On grid world and MuJoCo-based tasks with hierarchical structure, we find that the higher-order model learns to compress long activation sequence chunks onto internal controllers. Critically, each controller executes a sequence of behaviorally meaningful actions that unfold over long timescales and are accompanied with a learned termination condition, such that composing multiple controllers over time leads to efficient exploration on novel tasks. We show that direct internal controller reinforcement, a process we term "internal RL", enables learning from sparse rewards in cases where standard RL finetuning fails. Our results demonstrate the benefits of latent action generation and reinforcement in autoregressive models, suggesting internal RL as a promising avenue for realizing hierarchical RL within foundation models.

---

## 20. CoDrone: Autonomous Drone Navigation Assisted by Edge and Cloud Foundation Models

**论文链接:** [http://arxiv.org/abs/2512.19083v2](http://arxiv.org/abs/2512.19083v2)

**作者:** Pengyu Chen, Tao Ouyang, Ke Luo, Weijie Hong, Xu Chen

**发布时间:** 2025-12-22

**备注:** This paper is accepted by the IEEE Internet of Things Journal (IoT-J) for publication in the Special Issue on "Augmented Edge Sensing Intelligence for Low-Altitude IoT Systems"

### GPT解析

### 总结

CoDrone是一个创新的云-边缘-端协同计算框架，将基础模型集成到自主无人机导航中，通过灰度图像、深度估计和一维占据网格导航方法解决计算资源限制问题，并利用深度强化学习神经调度器和视觉语言交互模块提升导航性能。

### 背景

无人机自主导航面临机载计算资源有限的挑战，限制了深度神经网络的复杂度；任务卸载到边缘服务器会导致高延迟，造成系统设计中的固有权衡。

### 目的

解决资源受限无人机平台的性能问题，提出CoDrone框架作为首个将基础模型集成到自主无人机巡航场景中的协同计算框架。

### 方法

CoDrone使用灰度图像减少计算开销；利用Depth Anything V2进行深度估计；引入一维占据网格导航方法实现细粒度场景理解；采用基于深度强化学习的神经调度器集成深度估计与导航决策；开发无人机特定的视觉语言交互模块实现云基础模型与无人机间的有效交互。

### 主要发现

实验表明CoDrone在不同飞行速度和网络条件下优于基线方法，平均飞行距离增加40%，平均导航质量提高5%。

### 结论

CoDrone框架有效解决了无人机自主导航中的计算资源限制问题，通过协同计算和基础模型集成提升了性能，新方法和模块增强了环境感知、导航决策和复杂场景推理能力。

### 翻译

无人机自主导航面临来自有限机载计算资源的关键挑战，这限制了部署的深度神经网络只能采用浅层架构，无法处理复杂环境。将任务卸载到远程边缘服务器会引入高延迟，造成系统设计中的固有权衡。为解决这些限制，我们提出了CoDrone——首个将基础模型集成到自主无人机巡航场景中的云-边缘-端协同计算框架——有效利用基础模型提升资源受限无人机平台的性能。为减少机载计算和数据传输开销，CoDrone对导航模型使用灰度图像。当需要增强环境感知时，CoDrone利用边缘辅助的基础模型Depth Anything V2进行深度估计，并引入了一种新颖的一维占据网格导航方法——在提升效率和表示简单性的同时实现细粒度的场景理解。CoDrone的一个关键组件是基于深度强化学习的神经调度器，将深度估计与自主导航决策无缝集成，实现对动态环境的实时适应。此外，该框架引入了无人机特定的视觉语言交互模块，融入领域定制的低级飞行原语，实现云基础模型与无人机之间的有效交互。VLM的引入增强了在复杂未见场景中的开集推理能力。实验结果表明，CoDrone在不同飞行速度和网络条件下优于基线方法，实现了平均飞行距离增加40%和平均导航质量提高5%。


### 论文摘要

Autonomous navigation for Unmanned Aerial Vehicles faces key challenges from limited onboard computational resources, which restrict deployed deep neural networks to shallow architectures incapable of handling complex environments. Offloading tasks to remote edge servers introduces high latency, creating an inherent trade-off in system design. To address these limitations, we propose CoDrone - the first cloud-edge-end collaborative computing framework integrating foundation models into autonomous UAV cruising scenarios - effectively leveraging foundation models to enhance performance of resource-constrained unmanned aerial vehicle platforms. To reduce onboard computation and data transmission overhead, CoDrone employs grayscale imagery for the navigation model. When enhanced environmental perception is required, CoDrone leverages the edge-assisted foundation model Depth Anything V2 for depth estimation and introduces a novel one-dimensional occupancy grid-based navigation method - enabling fine-grained scene understanding while advancing efficiency and representational simplicity of autonomous navigation. A key component of CoDrone is a Deep Reinforcement Learning-based neural scheduler that seamlessly integrates depth estimation with autonomous navigation decisions, enabling real-time adaptation to dynamic environments. Furthermore, the framework introduces a UAV-specific vision language interaction module incorporating domain-tailored low-level flight primitives to enable effective interaction between the cloud foundation model and the UAV. The introduction of VLM enhances open-set reasoning capabilities in complex unseen scenarios. Experimental results show CoDrone outperforms baseline methods under varying flight speeds and network conditions, achieving a 40% increase in average flight distance and a 5% improvement in average Quality of Navigation.

---

## 21. SpidR-Adapt: A Universal Speech Representation Model for Few-Shot Adaptation

**论文链接:** [http://arxiv.org/abs/2512.21204v1](http://arxiv.org/abs/2512.21204v1)

**作者:** Mahi Luthra, Jiayi Shen, Maxime Poli, Angelo Ortiz, Yosuke Higuchi, Youssef Benchekroun, Martin Gleize, Charles-Eric Saint-James, Dongyan Lin, Phillip Rust, Angel Villar, Surya Parimi, Vanessa Stark, Rashel Moritz, Juan Pino, Yann LeCun, Emmanuel Dupoux

**发布时间:** 2025-12-24

### GPT解析

### 总结

本文介绍了一种名为SpidR-Adapt的新方法，用于在极少量的未标记数据下快速适应新语言。该方法通过元学习框架和多任务自适应预训练协议实现，并提出了首阶双层优化解决方案来降低计算成本。实验表明，该方法比标准训练提高了一百多倍的数据效率。

### 背景

人类婴儿仅通过几百小时的语音暴露就能获取新语言的基本单位，这与需要大量数据的自监督语音模型之间存在显著效率差距。现有自监督语音模型在数据效率方面远不如人类婴儿的语言学习能力。

### 目的

解决现有自监督语音模型与人类婴儿语言学习能力之间的效率差距，开发一种能够在极少未标记数据下快速适应新语言的方法。

### 方法

将低资源语音表征学习构造成元学习问题；构建多任务自适应预训练协议，将适应过程表述为双层优化框架；提出首阶双层优化解决方案避免沉重计算成本；通过交错监督稳定元训练，交替使用自监督和监督目标进行鲁棒初始化。

### 主要发现

SpidR-Adapt在音素区分度和口语语言建模方面取得快速提升；在目标语言音频训练时间不足1小时的情况下超过了领域内语言模型的性能；比标准训练方法提高了一百多倍的数据效率；为生物启发、数据高效的表征提供了实用且与架构无关的路径。

### 结论

SpidR-Adapt方法成功地缩小了人类婴儿语言学习能力与现有自监督语音模型之间的效率差距，通过创新的元学习框架和优化策略，实现了在极少数据下的高效语言适应。相关代码和模型已在GitHub开源。

### 翻译

人类婴儿仅通过几百小时的语音暴露就能获取新语言的基本单位，这凸显了与数据饥渴的自监督语音模型相比的惊人效率差距。为解决这一差距，本文介绍了SpidR-Adapt，用于使用最少的未标记数据快速适应新语言。我们将这种低资源语音表征学习构造成元学习问题，并构建了多任务自适应预训练协议，该协议将适应过程表述为双层优化框架。为了在此框架下实现可扩展的元训练，我们提出了一种新颖的启发式解决方案，即首阶双层优化，避免了沉重的计算成本。最后，我们通过交错监督使用鲁棒初始化来稳定元训练，该方法交替使用自监督和监督目标。实验表明，SpidR-Adapt在音素区分度和口语语言建模方面取得快速提升，在目标语言音频训练时间不足1小时的情况下，超过了领域内语言模型的性能，比标准训练提高了100倍以上的数据效率。这些发现强调了实现生物启发、数据高效表征的一种实用、与架构无关的路径。我们在GitHub开源了训练代码和模型检查点。


### 论文摘要

Human infants, with only a few hundred hours of speech exposure, acquire basic units of new languages, highlighting a striking efficiency gap compared to the data-hungry self-supervised speech models. To address this gap, this paper introduces SpidR-Adapt for rapid adaptation to new languages using minimal unlabeled data. We cast such low-resource speech representation learning as a meta-learning problem and construct a multi-task adaptive pre-training (MAdaPT) protocol which formulates the adaptation process as a bi-level optimization framework. To enable scalable meta-training under this framework, we propose a novel heuristic solution, first-order bi-level optimization (FOBLO), avoiding heavy computation costs. Finally, we stabilize meta-training by using a robust initialization through interleaved supervision which alternates self-supervised and supervised objectives. Empirically, SpidR-Adapt achieves rapid gains in phonemic discriminability (ABX) and spoken language modeling (sWUGGY, sBLIMP, tSC), improving over in-domain language models after training on less than 1h of target-language audio, over $100\times$ more data-efficient than standard training. These findings highlight a practical, architecture-agnostic path toward biologically inspired, data-efficient representations. We open-source the training code and model checkpoints at https://github.com/facebookresearch/spidr-adapt.

---

## 22. SparScene: Efficient Traffic Scene Representation via Sparse Graph Learning for Large-Scale Trajectory Generation

**论文链接:** [http://arxiv.org/abs/2512.21133v1](http://arxiv.org/abs/2512.21133v1)

**作者:** Xiaoyu Mo, Jintian Ge, Zifan Wang, Chen Lv, Karl Henrik Johansson

**发布时间:** 2025-12-24

**备注:** 13 pages, 7 figures, 5 tables

### GPT解析

### 总结

SparScene是一种稀疏图学习框架，专为高效可扩展的交通场景表示而设计，解决了现有方法在大规模复杂交通场景中的效率问题。

### 背景

多智能体轨迹生成是自动驾驶和智能交通系统的核心问题，但在复杂场景中高效建模众多道路用户和基础设施之间的动态交互仍然是一个开放性问题。

### 目的

克服现有方法的局限性，提出一种高效且可扩展的交通场景表示框架，解决大规模复杂交通场景中的轨迹生成问题。

### 方法

SparScene利用车道图拓扑在智能体和车道之间构建结构感知的稀疏连接，采用轻量级图编码器高效聚合智能体-地图和智能体-智能体交互，产生紧凑的场景表示。

### 主要发现

SparScene在Waymo开放运动数据集的运动预测基准测试中取得了具有竞争力的性能和显著的效率，能在5毫秒内为200多个智能体生成轨迹，使用2.9GB GPU内存，推理时间仅为54毫秒，可扩展到5000多个智能体和17000多条车道。

### 结论

SparScene通过稀疏图学习框架实现了高效且可扩展的交通场景表示，在大规模交通场景中展现出卓越的可扩展性。

### 翻译

多智能体轨迹生成是自动驾驶和智能交通系统的核心问题。然而，在复杂场景中高效建模众多道路用户和基础设施之间的动态交互仍然是一个开放性问题。现有方法通常采用基于距离或全连接密集图结构来捕获交互信息，这不仅引入了大量冗余边，还需要复杂且高度参数化的网络进行编码，从而导致训练和推理效率低下，限制了在大规模复杂交通场景中的可扩展性。为了克服现有方法的局限性，我们提出了SparScene，一种专为高效可扩展的交通场景表示而设计的稀疏图学习框架。SparScene不依赖于距离阈值，而是利用车道图拓扑在智能体和车道之间构建结构感知的稀疏连接，实现了高效且信息丰富的场景图表示。SparScene采用轻量级图编码器，有效聚合智能体-地图和智能体-智能体交互，产生紧凑的场景表示，显著提高了效率和可扩展性。在Waymo开放运动数据集的运动预测基准测试中，SparScene取得了具有竞争力的性能和显著的效率。它能在5毫秒内为场景中200多个智能体生成轨迹，仅使用2.9GB GPU内存，推理时间仅为54毫秒，可扩展到5000多个智能体和17000多条车道，突出了其在大规模交通场景中的卓越可扩展性。


### 论文摘要

Multi-agent trajectory generation is a core problem for autonomous driving and intelligent transportation systems. However, efficiently modeling the dynamic interactions between numerous road users and infrastructures in complex scenes remains an open problem. Existing methods typically employ distance-based or fully connected dense graph structures to capture interaction information, which not only introduces a large number of redundant edges but also requires complex and heavily parameterized networks for encoding, thereby resulting in low training and inference efficiency, limiting scalability to large and complex traffic scenes. To overcome the limitations of existing methods, we propose SparScene, a sparse graph learning framework designed for efficient and scalable traffic scene representation. Instead of relying on distance thresholds, SparScene leverages the lane graph topology to construct structure-aware sparse connections between agents and lanes, enabling efficient yet informative scene graph representation. SparScene adopts a lightweight graph encoder that efficiently aggregates agent-map and agent-agent interactions, yielding compact scene representations with substantially improved efficiency and scalability. On the motion prediction benchmark of the Waymo Open Motion Dataset (WOMD), SparScene achieves competitive performance with remarkable efficiency. It generates trajectories for more than 200 agents in a scene within 5 ms and scales to more than 5,000 agents and 17,000 lanes with merely 54 ms of inference time with a GPU memory of 2.9 GB, highlighting its superior scalability for large-scale traffic scenes.

---

## 23. AI-Driven Green Cognitive Radio Networks for Sustainable 6G Communication

**论文链接:** [http://arxiv.org/abs/2512.20739v1](http://arxiv.org/abs/2512.20739v1)

**作者:** Anshul Sharma, Shujaatali Badami, Biky Chouhan, Pushpanjali Pandey, Brijeena Rana, Navneet Kaur

**发布时间:** 2025-12-23

**备注:** 10 pages, 8 figures. Full research article with MATLAB and NS-3 simulations

### GPT解析

### 总结

本文提出了一种基于人工智能的绿色认知无线电网络框架，通过结合多种先进技术优化6G网络的能效和性能。

### 背景

6G无线通信需要实现Tb/s峰值数据率、亚毫秒级延迟和大规模物联网/车辆连接，要求可持续的空中音频接入和节能功能。认知无线电网络虽能缓解频谱稀缺问题，但传统方法能耗高且对频谱变化敏感。

### 目的

开发一个以人工智能驱动的绿色CRN框架，通过优化组合感知时间表、发射功率、带宽分配和RIS相位选择，提高网络能效和性能。

### 方法

该框架整合了深度强化学习与迁移学习、能量收集、可重构智能表面(RIS)以及轻量级遗传优化操作，实现智能资源分配和能效优化。

### 主要发现

与传统方法相比，该框架减少了25-30%的能量消耗，感知AUC超过0.90，数据包交付率(PDR)提高了6-13个百分点。

### 结论

该集成框架可轻松扩展到大型物联网和车辆应用，为6G认知无线电网络提供了可行且可持续的发展路径。

### 翻译

6G无线通信旨在实现Tb/s的峰值数据速率、亚毫秒级延迟以及大规模物联网/车辆连接，这要求可持续的空中音频接入和节能功能。认知无线电网络(CRNs)有助于缓解频谱稀缺问题，但传统的感知和分配方法仍然能耗高，且对快速频谱变化敏感。我们的框架以人工智能驱动的绿色CRN为核心，旨在将深度强化学习(DRL)与迁移学习、能量收集(EH)、可重构智能表面(RIS)以及其他轻量级遗传优化操作相结合，优化组合感知时间表、发射功率、带宽分配和RIS相位选择。与两个基线相比(密集负载下的MATLAB + NS-3、固定策略下的传统CRN、启发式资源分配下的混合CRN)，该框架使用了(25-30%)更少的能量储备，感知AUC大于0.90，PDR提高了6-13个百分点。该集成框架可轻松扩展到大型物联网和车辆应用，为6G CRNs提供了可行且可持续的发展路线图。


### 论文摘要

The 6G wireless aims at the Tb/s peak data rates are expected, a sub-millisecond latency, massive Internet of Things/vehicle connectivity, which requires sustainable access to audio over the air and energy-saving functionality. Cognitive Radio Networks CCNs help in alleviating the problem of spectrum scarcity, but classical sensing and allocation are still energy-consumption intensive, and sensitive to rapid spectrum variations. Our framework which centers on AI driven green CRN aims at integrating deep reinforcement learning (DRL) with transfer learning, energy harvesting (EH), reconfigurable intelligent surfaces (RIS) with other light-weight genetic refinement operations that optimally combine sensing timelines, transmit power, bandwidth distribution and RIS phase selection. Compared to two baselines, the utilization of MATLAB + NS-3 under dense loads, a traditional CRN with energy sensing under fixed policies, and a hybrid CRN with cooperative sensing under heuristic distribution of resource, there are (25-30%) fewer energy reserves used, sensing AUC greater than 0.90 and +6-13 p.p. higher PDR. The integrated framework is easily scalable to large IoT and vehicular applications, and it provides a feasible and sustainable roadmap to 6G CRNs.   Index Terms--Cognitive Radio Networks (CRNs), 6G, Green Communication, Energy Efficiency, Deep Reinforcement Learning (DRL), Spectrum Sensing, RIS, Energy Harvesting, QoS, IoT.

---

## 24. PUFM++: Point Cloud Upsampling via Enhanced Flow Matching

**论文链接:** [http://arxiv.org/abs/2512.20988v1](http://arxiv.org/abs/2512.20988v1)

**作者:** Zhi-Song Liu, Chenhang He, Roland Maier, Andreas Rupp

**发布时间:** 2025-12-24

**备注:** 21 pages, 15 figures

### GPT解析

### 总结

本文提出了PUFM++，一种增强的流匹配框架，用于从稀疏、噪声和部分观测中重建密集且准确的点云。该框架在几何保真度、输入鲁棒性和下游任务一致性三方面进行了改进。

### 背景

生成模型在高质量点云上采样方面已显示出强大潜力，但仍存在改进空间。

### 目的

开发一个增强的流匹配框架PUFM++，用于从稀疏、噪声和部分观测中重建密集且准确的点云。

### 方法

PUFM++引入了两阶段流匹配策略，首先学习从稀疏输入到密集目标的直接流，然后使用噪声扰动样本细化；提出数据驱动的自适应时间调度器提高采样效率；施加流形约束确保生成点与底层表面一致；集成循环接口网络增强层次特征交互。

### 主要发现

在合成基准和真实世界扫描上的广泛实验表明，PUFM++在点云上采样方面建立了新的最先进水平，在各种任务中提供了卓越的视觉保真度和定量准确性。

### 结论

PUFM++通过三个关键方面的改进（几何保真度、输入鲁棒性和下游任务一致性）显著提升了点云上采样质量，代码和预训练模型已公开。

### 翻译

生成建模的最新进展已显示出高质量点云上采样的强大潜力。在这项工作中，我们提出了PUFM++，一种增强的流匹配框架，用于从稀疏、噪声和部分观测中重建密集且准确的点云。PUFM++在三个关键方面改进了流匹配：(i) 几何保真度，(ii) 对不完美输入的鲁棒性，以及(iii) 与下游基于表面的任务的一致性。我们引入了一种两阶段流匹配策略，首先学习从稀疏输入到密集目标的直接直线路径流，然后使用噪声扰动样本对其进行细化，以更好地逼近终端边际分布。为了加速和稳定推理，我们提出了一种数据驱动的自适应时间调度器，基于插值行为提高采样效率。我们在采样过程中进一步施加流形约束，以确保生成的点与底层表面保持一致。最后，我们集成了循环接口网络(RIN)以增强层次特征交互并提高重建质量。在合成基准和真实世界扫描上的广泛实验表明，PUFM++在点云上采样方面建立了新的最先进水平，在各种任务中提供了卓越的视觉保真度和定量准确性。代码和预训练模型可在https://github.com/Holmes-Alan/Enhanced_PUFM公开获取。


### 论文摘要

Recent advances in generative modeling have demonstrated strong promise for high-quality point cloud upsampling. In this work, we present PUFM++, an enhanced flow-matching framework for reconstructing dense and accurate point clouds from sparse, noisy, and partial observations. PUFM++ improves flow matching along three key axes: (i) geometric fidelity, (ii) robustness to imperfect input, and (iii) consistency with downstream surface-based tasks. We introduce a two-stage flow-matching strategy that first learns a direct, straight-path flow from sparse inputs to dense targets, and then refines it using noise-perturbed samples to approximate the terminal marginal distribution better. To accelerate and stabilize inference, we propose a data-driven adaptive time scheduler that improves sampling efficiency based on interpolation behavior. We further impose on-manifold constraints during sampling to ensure that generated points remain aligned with the underlying surface. Finally, we incorporate a recurrent interface network~(RIN) to strengthen hierarchical feature interactions and boost reconstruction quality. Extensive experiments on synthetic benchmarks and real-world scans show that PUFM++ sets a new state of the art in point cloud upsampling, delivering superior visual fidelity and quantitative accuracy across a wide range of tasks. Code and pretrained models are publicly available at https://github.com/Holmes-Alan/Enhanced_PUFM.

---

## 25. OccuFly: A 3D Vision Benchmark for Semantic Scene Completion from the Aerial Perspective

**论文链接:** [http://arxiv.org/abs/2512.20770v1](http://arxiv.org/abs/2512.20770v1)

**作者:** Markus Gross, Sai B. Matha, Aya Fahmy, Rui Song, Daniel Cremers, Henri Meess

**发布时间:** 2025-12-23

### GPT解析

### 总结

本文介绍了OccuFly，第一个基于相机的真实世界空中语义场景完成基准数据集，解决了空中场景中SSC研究的不足，并提出了一种无需LiDAR的数据生成框架。

### 背景

语义场景完成(SSC)对移动机器人3D感知至关重要，在地面领域已被广泛研究，但在空中场景探索不足。LiDAR作为主要数据生成模态，对无人机存在飞行规定、质量和能量限制以及点云稀疏性等挑战。

### 目的

解决空中场景中SSC研究的局限性，创建基于相机的真实世界空中SSC基准数据集，提出减少手动3D标注工作量的LiDAR-free数据生成框架。

### 方法

引入OccuFly基准数据集，在50m、40m和30米高度采集，涵盖四季，包括城市、工业和农村场景，提供22个语义类别。提出基于相机模态的数据生成框架，利用传统3D重建将2D掩码提升到点云中实现自动化标签转移。

### 主要发现

在OccuFly上对最先进方法进行了基准测试，强调了高处视角特有的挑战，为空中3D场景理解提供了全面的视觉基准。

### 结论

OccuFly填补了空中场景中SSC研究的空白，基于相机的方法为无人机提供了避开LiDAR局限性的实用解决方案，该基准和框架将促进空中3D场景理解领域的发展。

### 翻译

语义场景完成(SSC)对移动机器人中的3D感知至关重要，它通过联合估计密集体积占用率和体素级语义来实现整体场景理解。尽管SSC已在地面领域（如自动驾驶）中得到广泛研究，但像自主飞行这样的空中场景仍 largely 未被探索，从而限制了下游应用的进展。此外，LiDAR传感器代表SSC数据生成的主要模态，由于飞行规定、质量和能量限制，以及从高处视角获取的LiDAR点云的稀疏性，这对大多数无人机(UAVs)构成了挑战。为解决这些限制，我们引入了OccuFly，这是第一个基于相机的真实世界空中SSC基准，在春季、夏季、秋季和冬季分别在50米、40米和30米的高度采集。OccuFly涵盖城市、工业和农村场景，提供22个语义类别，数据格式遵循既定惯例，便于与现有研究的无缝集成。重要的是，我们提出了一个基于相机模态的LiDAR-free数据生成框架，现代无人机普遍采用这种模态。利用传统的3D重建，我们的框架通过将部分标注的2D掩码提升到重建的点云中来自动化标签转移，从而显著减少了手动3D标注的工作量。最后，我们在OccuFly上对最先进的方法进行了基准测试，并强调了高处视角特有的挑战，为全面的空中3D场景理解提供了一个全面的视觉基准。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决空中视角（如无人机飞行）的语义场景完成(SSC)缺乏专用基准数据集的问题。这个问题很重要，因为现有SSC研究主要集中在地面场景，而空中场景理解对自主飞行至关重要；同时，传统基于LiDAR的数据生成方法在无人机应用中面临飞行限制、质量和能量约束以及点云稀疏性等问题，限制了相关技术的发展。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先分析了现有SSC数据集的局限性，特别是LiDAR在无人机应用中的挑战，然后设计了一个基于相机模态的数据生成框架。他们借鉴了传统3D重建技术(SfM和MVS)、现有SSC数据集的组织结构以及深度估计方法。创新点在于设计了高效的标注转移策略，只需标注少量图像(<10%)，通过2D-3D对应关系将语义标签自动提升到3D点云，大幅减少了标注工作量。', '这个方法的核心思想是什么？整体实现流程是怎样的？': "核心思想是使用相机而非LiDAR作为主要传感器，通过'少量标注，自动扩展'策略解决空中场景的语义场景完成问题。整体流程包括：1)使用地理参考图像进行3D重建，生成点云和深度图；2)对少量图像进行手动语义标注，然后通过2D-3D对应关系将标签提升到3D点云；3)将语义类分为实例类、地面类和其他类，分别采用DBSCAN聚类、泊松表面重建和直接体素化进行处理；4)通过视锥体裁剪提取每帧地面真值，并构建二进制掩码。", '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)首个空中视角的SSC基准数据集，包含9个场景、20,000+样本和22个语义类别；2)LiDAR-free数据生成框架，完全基于相机模态；3)高效的标注转移方法，只需标注<10%图像即可覆盖99%的3D点；4)全面的实验评估和针对空中场景的深度估计模型。相比之前工作，不同之处在于专注于空中视角而非地面视角，采用相机而非LiDAR作为主要传感器，提供了更大规模和更多样化的数据，并揭示了空中视角与地面视角的差异和挑战。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'OccuFly引入了首个基于相机的空中视角语义场景完成基准数据集，通过创新的LiDAR-free数据生成框架和高效的标注转移方法，大幅减少了3D标注工作量，为空中3D场景理解研究提供了重要资源。'}


### 论文摘要

Semantic Scene Completion (SSC) is crucial for 3D perception in mobile robotics, as it enables holistic scene understanding by jointly estimating dense volumetric occupancy and per-voxel semantics. Although SSC has been widely studied in terrestrial domains such as autonomous driving, aerial scenarios like autonomous flying remain largely unexplored, thereby limiting progress on downstream applications. Furthermore, LiDAR sensors represent the primary modality for SSC data generation, which poses challenges for most uncrewed aerial vehicles (UAVs) due to flight regulations, mass and energy constraints, and the sparsity of LiDAR-based point clouds from elevated viewpoints. To address these limitations, we introduce OccuFly, the first real-world, camera-based aerial SSC benchmark, captured at altitudes of 50m, 40m, and 30m during spring, summer, fall, and winter. OccuFly covers urban, industrial, and rural scenarios, provides 22 semantic classes, and the data format adheres to established conventions to facilitate seamless integration with existing research. Crucially, we propose a LiDAR-free data generation framework based on camera modality, which is ubiquitous on modern UAVs. By utilizing traditional 3D reconstruction, our framework automates label transfer by lifting a subset of annotated 2D masks into the reconstructed point cloud, thereby substantially minimizing manual 3D annotation effort. Finally, we benchmark the state-of-the-art on OccuFly and highlight challenges specific to elevated viewpoints, yielding a comprehensive vision benchmark for holistic aerial 3D scene understanding.

---

## 26. SegMo: Segment-aligned Text to 3D Human Motion Generation

**论文链接:** [http://arxiv.org/abs/2512.21237v1](http://arxiv.org/abs/2512.21237v1)

**作者:** Bowen Dang, Lin Wu, Xiaohang Yang, Zheng Yuan, Zhixiang Chen

**发布时间:** 2025-12-24

**备注:** The IEEE/CVF Winter Conference on Applications of Computer Vision 2026

### GPT解析

### 总结

SegMo是一种新的分段对齐的文本条件人体运动生成框架，通过将文本和运动分解为语义连贯的片段并使用对比学习进行对齐，实现了细粒度的文本-运动对应关系。

### 背景

从文本描述生成3D人体运动是视频游戏、虚拟现实和增强现实中的重要研究问题。现有方法在序列级别对齐文本和运动，忽略了模态的内部语义结构。

### 目的

提出SegMo框架，实现细粒度的文本-运动对齐，以改进现有的文本到3D人体运动生成方法。

### 方法

SegMo包含三个模块：(1)文本分段提取，将复杂描述分解为时序短语；(2)运动分段提取，将运动序列划分为对应片段；(3)细粒度文本-运动对齐，使用对比学习对齐文本和运动片段。

### 主要发现

SegMo在两个广泛使用的数据集上改进了强基线，在HumanML3D测试集上实现了0.553的改进TOP 1分数。

### 结论

由于学习了文本和运动片段的共享嵌入空间，SegMo也可应用于运动定位和运动到文本检索等检索式任务。

### 翻译

从文本描述生成3D人体运动是一个重要的研究问题，在视频游戏、虚拟现实和增强现实中有广泛应用。最近的方法在序列级别对齐文本描述和人体运动，忽略了模态的内部语义结构。然而，运动描述和运动序列都可以自然地分解为更小的语义连贯的片段，这些片段可以作为原子对齐单元来实现更细粒度的对应关系。受此启发，我们提出了SegMo，一种新的分段对齐的文本条件人体运动生成框架，以实现细粒度的文本-运动对齐。我们的框架包含三个模块：(1)文本分段提取，将复杂的文本描述分解为时序短语，每个短语代表一个简单的原子动作；(2)运动分段提取，将完整的运动序列划分为相应的运动片段；(3)细粒度文本-运动对齐，使用对比学习对齐文本和运动片段。大量实验表明，SegMo在两个广泛使用的数据集上改进了强基线，在HumanML3D测试集上实现了0.553的改进TOP 1分数。此外，由于为文本和运动片段学习了共享的嵌入空间，SegMo也可以应用于检索式任务，如运动定位和运动到文本检索。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决从文本描述生成3D人体运动时文本与运动对齐不够精细的问题。现有方法只在序列级别对齐文本和运动，忽略了两种模态内部的语义结构，导致生成的运动可能出现缺失动作、重复动作或动作顺序错误等问题。这个问题在现实世界中很重要，因为3D人体运动生成在视频游戏、虚拟现实和增强现实等领域有广泛应用，而更准确、自然的运动生成能显著提升这些技术的用户体验。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先分析了现有方法的局限性，即只在序列级别对齐文本和运动。然后基于事件分割理论(表明人类自然将连续流分解为有意义片段)的启发，提出将文本和运动都分解为更小的语义片段作为原子对齐单元。作者借鉴了MoMask作为基础框架，使用CLIP文本编码器和对比学习技术，但创新性地设计了三个新模块：文本片段提取(利用LLMs)、运动片段提取(分割运动序列)和精细文本-运动对齐(通过对比学习实现片段级对齐)。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是将文本描述和运动序列都分解为更小的、语义连贯的片段，作为原子对齐单元，实现更精细的文本-运动对齐。整体流程包括：1)文本片段提取 - 利用大型语言模型将复杂文本分解为时序有序的短语；2)运动片段提取 - 将完整运动序列分割为对应片段；3)精细文本-运动对齐 - 在共享嵌入空间中对齐文本和运动片段；4)运动生成 - 使用残差VQ-VAE和transformer生成运动序列。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)首次引入片段级对齐到文本条件的人体运动生成；2)设计了三个核心模块(文本片段提取、运动片段提取和精细对齐)；3)由于学习共享嵌入空间，支持了检索类任务应用。相比之前工作，不同之处在于：对齐粒度从序列级别提升到片段级别；明确考虑了文本和运动的内部结构；在每个样本内而非跨样本进行对比学习；扩展了模型在检索类任务的能力。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'SegMo通过片段级对齐框架将文本和运动分解为语义片段并实现精细对齐，显著提高了从文本生成3D人体运动的准确性和自然度，同时扩展了模型在检索类任务中的应用能力。'}


### 论文摘要

Generating 3D human motions from textual descriptions is an important research problem with broad applications in video games, virtual reality, and augmented reality. Recent methods align the textual description with human motion at the sequence level, neglecting the internal semantic structure of modalities. However, both motion descriptions and motion sequences can be naturally decomposed into smaller and semantically coherent segments, which can serve as atomic alignment units to achieve finer-grained correspondence. Motivated by this, we propose SegMo, a novel Segment-aligned text-conditioned human Motion generation framework to achieve fine-grained text-motion alignment. Our framework consists of three modules: (1) Text Segment Extraction, which decomposes complex textual descriptions into temporally ordered phrases, each representing a simple atomic action; (2) Motion Segment Extraction, which partitions complete motion sequences into corresponding motion segments; and (3) Fine-grained Text-Motion Alignment, which aligns text and motion segments with contrastive learning. Extensive experiments demonstrate that SegMo improves the strong baseline on two widely used datasets, achieving an improved TOP 1 score of 0.553 on the HumanML3D test set. Moreover, thanks to the learned shared embedding space for text and motion segments, SegMo can also be applied to retrieval-style tasks such as motion grounding and motion-to-text retrieval.

---

## 27. UniTacHand: Unified Spatio-Tactile Representation for Human to Robotic Hand Skill Transfer

**论文链接:** [http://arxiv.org/abs/2512.21233v1](http://arxiv.org/abs/2512.21233v1)

**作者:** Chi Zhang, Penglin Cai, Haoqi Yuan, Chaoyi Xu, Zongqing Lu

**发布时间:** 2025-12-24

### GPT解析

### 总结

本文提出UniTacHand统一表示方法，解决人类和机器人触觉数据不对齐问题，实现从人类到机器人的零样本触觉策略迁移，提高数据效率和泛化能力。

### 背景

触觉感知对机器人实现人类灵巧操作至关重要，特别是在视觉遮挡场景下。然而，大规模真实世界机器人触觉数据收集困难限制了其应用。

### 目的

解决人类和机器人触觉数据间的差距，实现从人类到机器人的触觉策略迁移，提升数据效率和泛化性能。

### 方法

1) 将人类和机器人触觉信号投影到MANO手模型的2D表面空间；2) 引入对比学习方法，仅需10分钟配对数据训练，将异构数据对齐到统一潜在空间；3) 提出UniTacHand统一表示方法桥接触觉信息差距。

### 主要发现

1) 实现从人类到真实机器人的零样本触觉策略迁移，可推广到未见物体；2) 混合数据（人类+机器人）联合训练比仅用机器人数据获得更好性能和数据效率。

### 结论

UniTacHand为触觉灵巧手的学习提供了通用、可扩展和数据高效的途径。

### 翻译

触觉感知对于机器人手实现人类级别的灵巧操作至关重要，特别是在视觉遮挡的场景中。然而，其应用常常因为难以收集大规模真实世界机器人触觉数据而受到阻碍。在本研究中，我们提议使用触觉手套收集低成本的人类操作数据，用于基于触觉的机器人策略学习。人类和机器人触觉数据之间的不对齐使得将从人类数据中学到的策略迁移到机器人上变得具有挑战性。为了弥合这一差距，我们提出了UniTacHand，一种统一表示方法，用于对齐由灵巧手捕获的机器人触觉信息与从手套获得的人类手触觉信息。首先，我们将人类手和机器人手的触觉信号投影到MANO手模型的形态一致的2D表面空间上。这种统一标准化了异构数据结构，并内在地将触觉信号嵌入空间上下文。然后，我们引入了一种对比学习方法，将它们对齐到统一的潜在空间中，仅使用我们数据收集系统中10分钟的配对数据进行训练。我们的方法实现了从人类到真实机器人的零样本触觉策略迁移，并推广到预训练数据中未见过的物体。我们还证明，通过UniTacHand对混合数据（包括人类和机器人演示）进行联合训练，比仅使用机器人数据能获得更好的性能和数据效率。UniTacHand为基于触觉的灵巧手的学习铺就了一条通用、可扩展和数据高效的途径。


### 论文摘要

Tactile sensing is crucial for robotic hands to achieve human-level dexterous manipulation, especially in scenarios with visual occlusion. However, its application is often hindered by the difficulty of collecting large-scale real-world robotic tactile data. In this study, we propose to collect low-cost human manipulation data using haptic gloves for tactile-based robotic policy learning. The misalignment between human and robotic tactile data makes it challenging to transfer policies learned from human data to robots. To bridge this gap, we propose UniTacHand, a unified representation to align robotic tactile information captured by dexterous hands with human hand touch obtained from gloves. First, we project tactile signals from both human hands and robotic hands onto a morphologically consistent 2D surface space of the MANO hand model. This unification standardizes the heterogeneous data structures and inherently embeds the tactile signals with spatial context. Then, we introduce a contrastive learning method to align them into a unified latent space, trained on only 10 minutes of paired data from our data collection system. Our approach enables zero-shot tactile-based policy transfer from humans to a real robot, generalizing to objects unseen in the pre-training data. We also demonstrate that co-training on mixed data, including both human and robotic demonstrations via UniTacHand, yields better performance and data efficiency compared with using only robotic data. UniTacHand paves a path toward general, scalable, and data-efficient learning for tactile-based dexterous hands.

---

## 28. ElfCore: A 28nm Neural Processor Enabling Dynamic Structured Sparse Training and Online Self-Supervised Learning with Activity-Dependent Weight Update

**论文链接:** [http://arxiv.org/abs/2512.21153v1](http://arxiv.org/abs/2512.21153v1)

**作者:** Zhe Su, Giacomo Indiveri

**发布时间:** 2025-12-24

**DOI:** 10.1109/ESSERC66193.2025.11214101

**备注:** This paper has been published in the proceedings of the 2025 IEEE European Solid-State Electronics Research Conference (ESSERC)

### GPT解析

### 总结

ElfCore是一种28nm数字脉冲神经网络处理器，专门用于事件驱动的感官信号处理，集成了自监督学习、稀疏训练和权重更新机制，在多个任务上表现出色。

### 背景

事件驱动的感官信号处理领域需要高效能的处理器来处理脉冲神经网络，同时减少功耗和内存需求。

### 目的

开发一种能够高效处理事件驱动信号、支持自监督学习和稀疏训练的神经网络处理器。

### 方法

设计并实现ElfCore处理器，集成本地在线自监督学习引擎、动态结构化稀疏训练引擎和活动依赖的稀疏权重更新机制。

### 主要发现

ElfCore在多个任务上表现出色，功耗降低16倍，内存需求减少3.8倍，网络容量效率提高5.9倍。

### 结论

ElfCore通过创新的架构设计，在保持高性能的同时显著降低了功耗和内存需求，为事件驱动的感官信号处理提供了高效解决方案。

### 翻译

在这篇论文中，我们提出了ElfCore，一种28nm数字脉冲神经网络处理器，专门用于事件驱动的感官信号处理。ElfCore首次高效集成了：(1)本地在线自监督学习引擎，能够在没有标记输入的情况下进行多层时序学习；(2)动态结构化稀疏训练引擎，支持高精度的稀疏到稀疏学习；(3)活动依赖的稀疏权重更新机制，仅基于输入活动和网络动态选择性更新权重。在手势识别、语音和生物医学信号处理等任务上的演示表明，ElfCore的性能优于最先进的解决方案，功耗降低高达16倍，芯片上内存需求减少3.8倍，网络容量效率提高5.9倍。


### 论文摘要

In this paper, we present ElfCore, a 28nm digital spiking neural network processor tailored for event-driven sensory signal processing. ElfCore is the first to efficiently integrate: (1) a local online self-supervised learning engine that enables multi-layer temporal learning without labeled inputs; (2) a dynamic structured sparse training engine that supports high-accuracy sparse-to-sparse learning; and (3) an activity-dependent sparse weight update mechanism that selectively updates weights based solely on input activity and network dynamics. Demonstrated on tasks including gesture recognition, speech, and biomedical signal processing, ElfCore outperforms state-of-the-art solutions with up to 16X lower power consumption, 3.8X reduced on-chip memory requirements, and 5.9X greater network capacity efficiency.

---

## 29. Encrypted Traffic Detection in Resource Constrained IoT Networks: A Diffusion Model and LLM Integrated Framework

**论文链接:** [http://arxiv.org/abs/2512.21144v1](http://arxiv.org/abs/2512.21144v1)

**作者:** Hongjuan Li, Hui Kang, Chenbang Liu, Ruolin Wang, Jiahui Li, Geng Sun, Jiacheng Wang, Shuang Liang, Shiwen Mao

**发布时间:** 2025-12-24

**备注:** This paper is accepted by IEEE Transactions on Network Science and Engineering

### GPT解析

### 总结

DMLITE是一种集成了扩散模型和大语言模型的流量嵌入框架，用于资源受限的IoT环境中的网络流量检测。该框架通过三阶段架构实现了高准确率的流量分类，同时减少了训练时间。

### 背景

物联网基础设施的普及和流量加密的广泛采用带来了显著挑战，特别是在具有动态流量模式、有限计算能力和严格延迟限制的环境中。

### 目的

提出DMLITE框架，用于资源受限的IoT环境中的网络流量检测，解决动态流量模式、有限计算能力和严格延迟限制带来的挑战。

### 方法

DMLITE采用三阶段架构：流量视觉预处理、基于扩散的多级特征提取和LLM引导的特征优化。框架利用自监督扩散模型通过多级特征融合和代表性样本选择的对比学习来捕获加密流量中的细粒度和抽象模式，并集成了LLM来动态调整粒子群优化参数，实现双重目标函数最小化分类错误和数据分布的方差。

### 主要发现

在USTC-TFC、ISCX-VPN和Edge-IIoTset数据集上分别实现了98.87%、92.61%和99.83%的分类准确率。与代表性深度学习模型相比，平均提高了3.7%的分类准确率，平均减少了41.9%的训练时间。

### 结论

DMLITE框架有效地解决了资源受限的IoT环境中的网络流量检测挑战，通过结合扩散模型和LLM实现了高准确率和效率。

### 翻译

物联网基础设施的普及和流量加密的广泛采用带来了显著挑战，特别是在具有动态流量模式、有限计算能力和严格延迟限制的环境中。本文提出了DMLITE，一种用于资源受限物联网环境中网络流量检测的扩散模型和大语言模型集成的流量嵌入框架。DMLITE通过三阶段架构克服了这些挑战，包括流量视觉预处理、基于扩散的多级特征提取和LLM引导的特征优化。具体而言，该框架利用自监督扩散模型通过多级特征融合和代表性样本选择的对比学习来捕获加密流量中的细粒度和抽象模式，从而能够用最少的标记数据快速适应新的流量模式。此外，DMLITE集成了LLM，通过实现双重目标函数来动态调整粒子群优化参数，该函数最小化分类错误和数据分布的方差。在基准数据集上的全面实验验证证实了DMLITE的有效性，在USTC-TFC、ISCX-VPN和Edge-IIoTset数据集上分别实现了98.87%、92.61%和99.83%的分类准确率。与代表性深度学习模型相比，这平均提高了3.7%的分类准确率，平均减少了41.9%的训练时间。


### 论文摘要

The proliferation of Internet-of-things (IoT) infrastructures and the widespread adoption of traffic encryption present significant challenges, particularly in environments characterized by dynamic traffic patterns, constrained computational capabilities, and strict latency constraints. In this paper, we propose DMLITE, a diffusion model and large language model (LLM) integrated traffic embedding framework for network traffic detection within resource-limited IoT environments. The DMLITE overcomes these challenges through a tri-phase architecture including traffic visual preprocessing, diffusion-based multi-level feature extraction, and LLM-guided feature optimization. Specifically, the framework utilizes self-supervised diffusion models to capture both fine-grained and abstract patterns in encrypted traffic through multi-level feature fusion and contrastive learning with representative sample selection, thus enabling rapid adaptation to new traffic patterns with minimal labeled data. Furthermore, DMLITE incorporates LLMs to dynamically adjust particle swarm optimization parameters for intelligent feature selection by implementing a dual objective function that minimizes both classification error and variance across data distributions. Comprehensive experimental validation on benchmark datasets confirms the effectiveness of DMLITE, achieving classification accuracies of 98.87\%, 92.61\%, and 99.83\% on USTC-TFC, ISCX-VPN, and Edge-IIoTset datasets, respectively. This improves classification accuracy by an average of 3.7\% and reduces training time by an average of 41.9\% compared to the representative deep learning model.

---

## 30. MultiMind at SemEval-2025 Task 7: Crosslingual Fact-Checked Claim Retrieval via Multi-Source Alignment

**论文链接:** [http://arxiv.org/abs/2512.20950v1](http://arxiv.org/abs/2512.20950v1)

**作者:** Mohammad Mahdi Abootorabi, Alireza Ghahramani Kure, Mohammadali Mohammadkhani, Sina Elahimanesh, Mohammad Ali Ali Panah

**发布时间:** 2025-12-24

**备注:** 11 pages Published at the SemEval-2025 workshop

### GPT解析

### 总结

论文介绍了TriAligner系统，用于多语言和跨语言事实核查主张检索

### 背景

错误信息迅速传播的时代，有效事实核查至关重要

### 目的

开发一个能够跨多种语言有效检索事实核查主张的系统

### 方法

TriAligner采用双编码器架构和对比学习，结合不同模态下的原生和英语翻译，学习不同来源的相对重要性，并使用大型语言模型进行数据预处理和增强，采用困难负采样改进表示学习

### 主要发现

在单语言和跨语言基准上，TriAligner相比基线在检索准确性和事实核查性能上有显著改进

### 结论

TriAligner是一种有效的多语言和跨语言事实核查主张检索方法

### 翻译

本文介绍了我们为SemEval-2025任务7：多语言和跨语言事实核查主张检索设计的系统。在错误信息迅速传播的时代，有效的事实核查变得越来越重要。我们引入了TriAligner，一种新颖的方法，它利用双编码器架构和对比学习，并结合不同模态下的原生和英语翻译。我们的方法通过学习不同来源在排列中的相对重要性，有效地跨多种语言检索主张。为了增强鲁棒性，我们使用大型语言模型进行高效的数据预处理和增强，同时采用困难负采样来改进表示学习。我们在单语言和跨语言基准上评估了我们的方法，显示出相比基线在检索准确性和事实核查性能上有显著改进。


### 论文摘要

This paper presents our system for SemEval-2025 Task 7: Multilingual and Crosslingual Fact-Checked Claim Retrieval. In an era where misinformation spreads rapidly, effective fact-checking is increasingly critical. We introduce TriAligner, a novel approach that leverages a dual-encoder architecture with contrastive learning and incorporates both native and English translations across different modalities. Our method effectively retrieves claims across multiple languages by learning the relative importance of different sources in alignment. To enhance robustness, we employ efficient data preprocessing and augmentation using large language models while incorporating hard negative sampling to improve representation learning. We evaluate our approach on monolingual and crosslingual benchmarks, demonstrating significant improvements in retrieval accuracy and fact-checking performance over baselines.

---

## 31. Self-supervised Multiplex Consensus Mamba for General Image Fusion

**论文链接:** [http://arxiv.org/abs/2512.20921v1](http://arxiv.org/abs/2512.20921v1)

**作者:** Yingying Wang, Rongjin Zhuang, Hui Zheng, Xuanhua He, Ke Cao, Xiaotong Tu, Xinghao Ding

**发布时间:** 2025-12-24

**备注:** Accepted by AAAI 2026, 9 pages, 4 figures

### GPT解析

### 总结

该论文提出了SMC-Mamba框架，一种用于通用图像融合的自监督多路共识Mamba方法，通过创新的模块设计和损失函数实现了高质量的多模态图像融合，在多种任务中优于现有方法。

### 背景

图像融合技术整合不同模态的互补信息以生成高质量融合图像，增强下游任务如目标检测和语义分割。现有任务特定技术主要关注模态间信息整合，而通用图像融合需解决广泛任务同时提高性能而不增加复杂度。

### 目的

开发一种通用图像融合框架，能够在不增加计算复杂度的情况下提高多种下游任务的性能，有效整合不同模态的互补信息。

### 方法

提出了SMC-Mamba框架，包含：1)模态无关特征增强(MAFE)模块，通过自适应门控保留细节，通过空间-通道和频率-旋转扫描增强全局表示；2)多路共识跨模态Mamba(MCCM)模块，实现专家间动态协作达成共识，高效整合多模态信息；3)双层自监督对比学习损失(BSCL)，保留高频信息同时提高下游任务性能。

### 主要发现

大量实验表明，该方法在红外-可见光、医学、多焦点、多曝光融合以及下游视觉任务中均优于最先进的图像融合算法。

### 结论

SMC-Mamba框架通过创新的模块设计和损失函数，有效解决了通用图像融合中的关键挑战，在不增加计算复杂度的情况下实现了高性能的图像融合，为多种下游视觉任务提供了更高质量的输入。

### 翻译

图像融合整合来自不同模态的互补信息，以生成高质量的融合图像，从而增强目标检测和语义分割等下游任务。与主要关注整合模态间信息的任务特定技术不同，通用图像融合需要解决广泛任务的同时提高性能而不增加复杂度。为此，我们提出了SMC-Mamba，一种用于通用图像融合的自监督多路共识Mamba框架。具体而言，模态无关特征增强(MAFE)模块通过自适应门控保留精细细节，并通过空间-通道和频率-旋转扫描增强全局表示。多路共识跨模态Mamba(MCCM)模块使专家之间能够动态协作，达成共识以高效整合多模态的互补信息。MCCM内的跨模态扫描进一步加强了模态间的特征交互，促进来自两个源的关键信息的无缝整合。此外，我们引入了双层自监督对比学习损失(BSCL)，它在不增加计算开销的同时保留高频信息，同时提高下游任务性能。大量实验证明，我们的方法在红外-可见光、医学、多焦点、多曝光融合以及下游视觉任务中优于最先进的图像融合算法。


### 论文摘要

Image fusion integrates complementary information from different modalities to generate high-quality fused images, thereby enhancing downstream tasks such as object detection and semantic segmentation. Unlike task-specific techniques that primarily focus on consolidating inter-modal information, general image fusion needs to address a wide range of tasks while improving performance without increasing complexity. To achieve this, we propose SMC-Mamba, a Self-supervised Multiplex Consensus Mamba framework for general image fusion. Specifically, the Modality-Agnostic Feature Enhancement (MAFE) module preserves fine details through adaptive gating and enhances global representations via spatial-channel and frequency-rotational scanning. The Multiplex Consensus Cross-modal Mamba (MCCM) module enables dynamic collaboration among experts, reaching a consensus to efficiently integrate complementary information from multiple modalities. The cross-modal scanning within MCCM further strengthens feature interactions across modalities, facilitating seamless integration of critical information from both sources. Additionally, we introduce a Bi-level Self-supervised Contrastive Learning Loss (BSCL), which preserves high-frequency information without increasing computational overhead while simultaneously boosting performance in downstream tasks. Extensive experiments demonstrate that our approach outperforms state-of-the-art (SOTA) image fusion algorithms in tasks such as infrared-visible, medical, multi-focus, and multi-exposure fusion, as well as downstream visual tasks.

---

## 32. MODE: Multi-Objective Adaptive Coreset Selection

**论文链接:** [http://arxiv.org/abs/2512.21152v1](http://arxiv.org/abs/2512.21152v1)

**作者:** Tanmoy Mukherjee, Pierre Marquis, Zied Bouraoui

**发布时间:** 2025-12-24

### GPT解析

### 总结

本文提出了MODE（多目标自适应数据效率）框架，该框架根据核心集选择策略对模型性能的不断演变的贡献动态结合这些策略。

### 背景

静态方法无法适应不同训练阶段的数据选择需求。

### 目的

开发一个能够根据不同训练阶段动态调整选择标准的框架，以提高数据效率和模型性能。

### 方法

MODE框架根据训练阶段调整选择标准：早期强调类别平衡，表示学习阶段强调多样性，收敛阶段强调不确定性。

### 主要发现

MODE实现了理论近似保证，具有高效的计算复杂度，在保持竞争力的准确率的同时，提供了对数据效用演化的可解释性见解，并减少了内存需求。

### 结论

MODE框架通过动态调整选择策略，有效提高了数据效率和模型性能。

### 翻译

我们提出了MODE（多目标自适应数据效率）框架，该框架根据核心集选择策略对模型性能的不断演变的贡献动态结合这些策略。与静态方法不同，MODE将选择标准适应到训练阶段：早期强调类别平衡，表示学习期间强调多样性，收敛时强调不确定性。我们证明MODE实现了理论近似保证，具有高效复杂度，并在保持竞争力的准确率的同时，提供了对数据效用演化的可解释性见解。实验表明MODE减少了内存需求。


### 论文摘要

We present Mode(Multi-Objective adaptive Data Efficiency), a framework that dynamically combines coreset selection strategies based on their evolving contribution to model performance. Unlike static methods, \mode adapts selection criteria to training phases: emphasizing class balance early, diversity during representation learning, and uncertainty at convergence. We show that MODE achieves (1-1/e)-approximation with O(n \log n) complexity and demonstrates competitive accuracy while providing interpretable insights into data utility evolution. Experiments show \mode reduces memory requirements

---

## 33. Shared Representation Learning for High-Dimensional Multi-Task Forecasting under Resource Contention in Cloud-Native Backends

**论文链接:** [http://arxiv.org/abs/2512.21102v1](http://arxiv.org/abs/2512.21102v1)

**作者:** Zixiao Huang, Jixiao Yang, Sijia Li, Chi Zhang, Jinyu Chen, Chengda Xu

**发布时间:** 2025-12-24

### GPT解析

### 总结

该研究提出了一种用于高维多任务时间序列的统一预测框架，旨在满足云原生后端系统在高动态负载、耦合指标和并行任务条件下的预测需求。该方法通过共享编码结构、状态融合机制、跨任务结构传播模块和动态调整机制，实现了对复杂系统行为的准确预测，并在实验验证中表现出优越性能。

### 背景

云原生后端系统在高动态负载、耦合指标和并行任务条件下运行，面临着复杂的预测挑战。这些系统需要能够处理资源竞争、链路交互和服务拓扑变化等因素形成的复杂结构模式。

### 目的

开发一个统一预测框架，能够处理高维多任务时间序列，为云原生后端系统提供可靠的预测能力，支持智能后端管理。

### 方法

1. 构建共享编码结构统一表示不同的监控指标；2. 采用状态融合机制捕获不同时间尺度的趋势变化和局部扰动；3. 引入跨任务结构传播模块建模节点间的潜在依赖关系；4. 包含动态调整机制根据系统状态变化自动调节内部特征流，确保在负载突变、拓扑漂移和资源抖动情况下保持稳定预测。

### 主要发现

所提出的方法在多个误差指标上表现出优越性能，能够为不同运行条件下的未来状态提供更准确的表示。实验通过超参数敏感性、环境敏感性和数据敏感性分析验证了框架的有效性。

### 结论

统一预测框架为云原生系统中的高维、多任务和强动态环境提供了可靠的预测能力，为智能后端管理提供了必要的技术支持。

### 翻译

本研究提出了一种适用于高维多任务时间序列的统一预测框架，以满足在高度动态负载、耦合指标和并行任务条件下运行的云原生后端系统的预测需求。该方法构建了一个共享编码结构，以统一方式表示多样化的监控指标，并采用状态融合机制来捕获不同时间尺度上的趋势变化和局部扰动。引入了跨任务结构传播模块来建模节点间的潜在依赖关系，使模型能够理解由资源竞争、链路交互和服务拓扑变化形成的复杂结构模式。为了增强对非平稳行为的适应性，该框架集成了动态调整机制，可根据系统状态变化自动调节内部特征流，确保在负载突变、拓扑漂移和资源抖动情况下保持稳定预测。实验评估比较了多个模型在不同指标上的表现，并通过超参数敏感性、环境敏感性和数据敏感性分析验证了框架的有效性。结果表明，所提出的方法在多个误差指标上实现了优越性能，并能针对不同运行条件提供更准确的未来状态表示。总体而言，统一预测框架为云原生系统中的高维、多任务和强动态环境提供了可靠的预测能力，为智能后端管理提供了必要的技术支持。


### 论文摘要

This study proposes a unified forecasting framework for high-dimensional multi-task time series to meet the prediction demands of cloud native backend systems operating under highly dynamic loads, coupled metrics, and parallel tasks. The method builds a shared encoding structure to represent diverse monitoring indicators in a unified manner and employs a state fusion mechanism to capture trend changes and local disturbances across different time scales. A cross-task structural propagation module is introduced to model potential dependencies among nodes, enabling the model to understand complex structural patterns formed by resource contention, link interactions, and changes in service topology. To enhance adaptability to non-stationary behaviors, the framework incorporates a dynamic adjustment mechanism that automatically regulates internal feature flows according to system state changes, ensuring stable predictions in the presence of sudden load shifts, topology drift, and resource jitter. The experimental evaluation compares multiple models across various metrics and verifies the effectiveness of the framework through analyses of hyperparameter sensitivity, environmental sensitivity, and data sensitivity. The results show that the proposed method achieves superior performance on several error metrics and provides more accurate representations of future states under different operating conditions. Overall, the unified forecasting framework offers reliable predictive capability for high-dimensional, multi-task, and strongly dynamic environments in cloud native systems and provides essential technical support for intelligent backend management.

---

## 34. Multimodal Skeleton-Based Action Representation Learning via Decomposition and Composition

**论文链接:** [http://arxiv.org/abs/2512.21064v1](http://arxiv.org/abs/2512.21064v1)

**作者:** Hongsong Wang, Heng Fei, Bingxuan Dai, Jie Gui

**发布时间:** 2025-12-24

**备注:** Accepted by Machine Intelligence Research (Journal Impact Factor 8.7, 2024)

### GPT解析

### 总结

这是一项关于多模态人体动作理解的研究，提出了一种名为'分解与组合'的自监督框架，通过分解和组合策略有效平衡了计算效率和模型性能。

### 背景

多模态人体动作理解是计算机视觉中的重要问题，主要挑战是如何有效利用不同模态之间的互补性，同时保持模型效率。现有方法要么使用简单的后期融合（导致大量计算开销），要么使用早期融合（效率高但性能有限）。

### 目的

解决在效率和有效性之间平衡的困境。

### 方法

提出了一种名为'分解与组合'的自监督多模态骨架动作表示学习框架。该方法包含两个策略：分解策略将融合的多模态特征分解为不同的单模态特征，并与相应的真实单模态特征进行对齐；组合策略整合多个单模态特征，利用它们作为自监督指导来增强多模态表示的学习。

### 主要发现

在NTU RGB+D 60、NTU RGB+D 120和PKU-MMD II数据集上的大量实验表明，所提出的方法在计算成本和模型性能之间取得了很好的平衡。

### 结论

该方法成功解决了多模态人体动作理解中效率和有效性平衡的难题。

### 翻译

多模态人体动作理解是计算机视觉中的一个重要问题，核心挑战在于有效利用不同模态之间的互补性，同时保持模型效率。然而，大多数现有方法依赖简单的后期融合来提升性能，这导致了大量的计算开销。虽然对所有模态使用共享骨干网络的早期融合是高效的，但它难以实现优异的性能。为了解决效率和有效性之间的平衡困境，我们引入了一种自监督的多模态骨架动作表示学习框架，名为'分解与组合'。分解策略将融合的多模态特征细致地分解为不同的单模态特征，随后将它们与各自的真实单模态对应特征进行对齐。另一方面，组合策略整合多个单模态特征，利用它们作为自监督指导来增强多模态表示的学习。在NTU RGB+D 60、NTU RGB+D 120和PKU-MMD II数据集上的大量实验表明，所提出的方法在计算成本和模型性能之间取得了出色的平衡。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决多模态骨架动作理解中有效利用不同模态之间的互补性同时保持模型效率的挑战。这个问题重要是因为骨架数据相比传统图像或视频数据有诸多优势（消除背景干扰、保护隐私、计算效率高），而多模态数据可提供互补信息提高识别准确性，在实际应用（如人机交互、视频监控）中需要高效且准确的系统，且自监督学习能克服标记数据获取困难的限制。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先分析多模态动作理解中效率和性能的权衡问题，认识到现有方法要么后期融合（计算量大）要么早期融合（性能受限）。他们借鉴了UmURL[21]的统一编码器思想、对比学习的实例区分任务、时空解耦概念以及VICReg[45]的正则化方法。在此基础上，设计了嵌入融合策略在特征空间整合多模态信息，提出分解策略确保多模态特征包含各模态信息，以及组合策略整合单模态特征作为自监督指导增强多模态表示学习，并引入视角不变训练策略提升泛化能力。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是通过分解与组合的自监督训练策略，在嵌入融合框架下利用模态间的互补性，同时保持效率和性能。整体流程包括：1)多模态嵌入融合，将不同模态输入映射到共同嵌入空间并整合；2)时空解耦编码，将输入分为时间视图和空间视图分别处理；3)单模态特征分解，将融合特征分解为各模态特征并与真实特征对齐；4)多模态特征组合，使用后期融合组合各模态特征作为监督信号；5)训练过程，应用正则化防止模型崩溃，结合多视角数据增强视角不变性。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)分解与组合训练框架，确保多模态特征包含各模态信息并直接优化多模态表示；2)嵌入融合策略，在特征空间融合多模态信息平衡效率和性能；3)时空解耦的双流学习框架，针对时空特征分别设计损失函数；4)视角不变训练策略，利用多视角数据作为无监督信号。相比之前工作，与UmURL[21]相比增加了对多模态特征的直接优化；与CrosSCLR[26]和CMD[27]相比融合策略更高效；与传统早期/后期融合相比在特征空间融合平衡了优缺点；将时空解耦应用于多模态自监督学习进一步提升特征质量。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '这篇论文提出了分解与组合的自监督训练框架，通过嵌入融合策略和时空解耦的双流学习，实现了高效且高性能的多模态骨架动作表示学习，在多种任务和基准数据集上达到了最先进的结果。'}


### 论文摘要

Multimodal human action understanding is a significant problem in computer vision, with the central challenge being the effective utilization of the complementarity among diverse modalities while maintaining model efficiency. However, most existing methods rely on simple late fusion to enhance performance, which results in substantial computational overhead. Although early fusion with a shared backbone for all modalities is efficient, it struggles to achieve excellent performance. To address the dilemma of balancing efficiency and effectiveness, we introduce a self-supervised multimodal skeleton-based action representation learning framework, named Decomposition and Composition. The Decomposition strategy meticulously decomposes the fused multimodal features into distinct unimodal features, subsequently aligning them with their respective ground truth unimodal counterparts. On the other hand, the Composition strategy integrates multiple unimodal features, leveraging them as self-supervised guidance to enhance the learning of multimodal representations. Extensive experiments on the NTU RGB+D 60, NTU RGB+D 120, and PKU-MMD II datasets demonstrate that the proposed method strikes an excellent balance between computational cost and model performance.

---

## 35. Towards Better Search with Domain-Aware Text Embeddings for C2C Marketplaces

**论文链接:** [http://arxiv.org/abs/2512.21021v1](http://arxiv.org/abs/2512.21021v1)

**作者:** Andre Rusli, Miao Cao, Shoma Ishimoto, Sho Akiyama, Max Frenzel

**发布时间:** 2025-12-24

**备注:** 5 pages, AAAI 2026 Workshop on New Frontiers in Information Retrieval

### GPT解析

### 总结

该论文提出了一种领域感知的日语文本嵌入方法，以提高日本最大C2C市场Mercari的搜索质量。通过微调、角色特定前缀和俄罗斯套娃表示学习，该方法在离线评估和在线A/B测试中均显示出显著改善。

### 背景

C2C市场面临特殊检索挑战，包括简短模糊的查询、嘈杂的用户生成商品列表以及严格的生产约束。

### 目的

构建一个领域感知的日语文本嵌入方法，提高Mercari平台的搜索质量。

### 方法

基于购买驱动的查询-标题对进行微调，使用角色特定前缀建模查询-项目不对称性，应用俄罗斯套娃表示学习获得紧凑、截断鲁棒的嵌入，并通过离线评估、手动评估和在线A/B测试进行验证。

### 主要发现

与通用编码器相比显示一致改进，特别是用俄罗斯套娃截断替换PCA压缩时；更好处理专有名词、市场特定语义和术语重要性对齐；每用户收入和搜索流程效率有统计学显著改善，同时保持交易频率。

### 结论

领域感知的嵌入提高了大规模相关性和效率，为更丰富的LLM时代搜索体验提供了实用基础。

### 翻译

消费者对消费者(C2C)市场平台带来特殊的检索挑战：简短、模糊的查询；嘈杂的、用户生成的商品列表；以及严格的生产约束。本文报告了我们在Mercari（日本最大的C2C市场）构建领域感知日语文本嵌入方法的实验，以改善搜索质量。我们尝试基于购买驱动的查询-标题对进行微调，使用角色特定前缀来建模查询-项目不对称性。为满足生产约束，我们应用俄罗斯套娃表示学习来获得紧凑、截断鲁棒的嵌入。基于历史搜索日志的离线评估显示，相比强大的通用编码器有稳定提升，特别是在用俄罗斯套娃截断替换PCA压缩时改进显著。手动评估进一步突显了在专有名词、市场特定语义和术语重要性对齐方面的更好处理。此外，初步的在线A/B测试证明了每用户收入和搜索流程效率的统计学显著改善，同时保持了交易频率。结果表明，领域感知的嵌入提高了大规模的相关性和效率，并为更丰富的LLM时代搜索体验形成了实用基础。


### 论文摘要

Consumer-to-consumer (C2C) marketplaces pose distinct retrieval challenges: short, ambiguous queries; noisy, user-generated listings; and strict production constraints. This paper reports our experiment to build a domain-aware Japanese text-embedding approach to improve the quality of search at Mercari, Japan's largest C2C marketplace. We experimented with fine-tuning on purchase-driven query-title pairs, using role-specific prefixes to model query-item asymmetry. To meet production constraints, we apply Matryoshka Representation Learning to obtain compact, truncation-robust embeddings. Offline evaluation on historical search logs shows consistent gains over a strong generic encoder, with particularly large improvements when replacing PCA compression with Matryoshka truncation. A manual assessment further highlights better handling of proper nouns, marketplace-specific semantics, and term-importance alignment. Additionally, an initial online A/B test demonstrates statistically significant improvements in revenue per user and search-flow efficiency, with transaction frequency maintained. Results show that domain-aware embeddings improve relevance and efficiency at scale and form a practical foundation for richer LLM-era search experiences.

---

## 36. 论文ID: 2512.20963v1

**错误:** 无法读取论文信息 - [Errno 2] No such file or directory: 'storage/result/2512.20963v1.json'

---

## 37. ReACT-Drug: Reaction-Template Guided Reinforcement Learning for de novo Drug Design

**论文链接:** [http://arxiv.org/abs/2512.20958v1](http://arxiv.org/abs/2512.20958v1)

**作者:** R Yadunandan, Nimisha Ghosh

**发布时间:** 2025-12-24

### GPT解析

### 总结

这篇论文介绍了一个名为ReACT-Drug的全新药物设计框架，该框架基于强化学习，能够生成具有高结合亲和力和高合成可及性的全新药物候选分子，同时保证100%的化学有效性和新颖性。

### 背景

从头药物设计是现代药物开发的关键组成部分，但在巨大的化学空间中寻找具有合成可行性和高亲和力的候选分子仍然是一个重大挑战。传统的监督学习方法缺乏多目标优化和探索新型化学空间的能力。

### 目的

开发一个名为ReACT-Drug的完全集成、目标无关的分子设计框架，该框架基于强化学习，能够克服传统方法的局限性，自动加速理性药物设计过程。

### 方法

ReACT-Drug采用通用方法，利用ESM-2蛋白质嵌入从知识库中识别与给定目标相似的蛋白质，将对应蛋白质的已知药物配体分解以初始化基于片段的搜索空间，使代理偏向生物学相关的子空间。然后采用近端策略优化代理，引导经过ChemBERTa编码的分子通过动态的、基于反应模板的化学有效变换动作空间。

### 主要发现

ReACT-Drug能够生成具有竞争性结合亲和力和高合成可及性的全新药物候选分子，同时确保100%的化学有效性和新颖性（根据MOSES基准测试）。

### 结论

该架构展示了整合结构生物学、深度表示学习和化学合成规则以自动化和加速理性药物设计的潜力。

### 翻译

从头药物设计是现代药物开发的关键组成部分，然而在巨大的化学空间中寻找具有合成可行性和高亲和力的候选分子仍然是一个重大挑战。强化学习通过实现多目标优化和新型化学空间的探索增强了这一过程——这些能力是传统监督学习方法所缺乏的。在这项工作中，我们介绍了ReACT-Drug，这是一个基于强化学习的完全集成、目标无关的分子设计框架。与需要针对特定目标进行微调的模型不同，ReACT-Drug利用通用方法，通过利用ESM-2蛋白质嵌入从知识库中识别与给定目标相似的蛋白质。此后，对应于这些蛋白质的已知药物配体被分解，以初始化基于片段的搜索空间，使代理偏向生物学相关的子空间。对于每个这样的片段，该管道采用近端策略优化代理，引导经过ChemBERTa编码的分子通过动态的、基于反应模板的化学有效变换动作空间。这导致了生成具有竞争性结合亲和力和高合成可及性的全新药物候选分子，同时根据MOSES基准测试确保100%的化学有效性和新颖性。该架构突出了整合结构生物学、深度表示学习和化学合成规则以自动化和加速理性药物设计的潜力。


### 论文摘要

De novo drug design is a crucial component of modern drug development, yet navigating the vast chemical space to find synthetically accessible, high-affinity candidates remains a significant challenge. Reinforcement Learning (RL) enhances this process by enabling multi-objective optimization and exploration of novel chemical space - capabilities that traditional supervised learning methods lack. In this work, we introduce \textbf{ReACT-Drug}, a fully integrated, target-agnostic molecular design framework based on Reinforcement Learning. Unlike models requiring target-specific fine-tuning, ReACT-Drug utilizes a generalist approach by leveraging ESM-2 protein embeddings to identify similar proteins for a given target from a knowledge base such as Protein Data Base (PDB). Thereafter, the known drug ligands corresponding to such proteins are decomposed to initialize a fragment-based search space, biasing the agent towards biologically relevant subspaces. For each such fragment, the pipeline employs a Proximal Policy Optimization (PPO) agent guiding a ChemBERTa-encoded molecule through a dynamic action space of chemically valid, reaction-template-based transformations. This results in the generation of \textit{de novo} drug candidates with competitive binding affinities and high synthetic accessibility, while ensuring 100\% chemical validity and novelty as per MOSES benchmarking. This architecture highlights the potential of integrating structural biology, deep representation learning, and chemical synthesis rules to automate and accelerate rational drug design. The dataset and code are available at https://github.com/YadunandanRaman/ReACT-Drug/.

---

## 38. Streaming Video Instruction Tuning

**论文链接:** [http://arxiv.org/abs/2512.21334v1](http://arxiv.org/abs/2512.21334v1)

**作者:** Jiaer Xia, Peixian Chen, Mengdan Zhang, Xing Sun, Kaiyang Zhou

**发布时间:** 2025-12-24

### GPT解析

### 总结

Streamo是一个实时流式视频大语言模型，可作为通用交互式助手，能够执行广泛的流式视频任务，包括实时叙述、动作理解、事件标注等。

### 背景

现有的在线视频模型主要专注于问答或字幕生成等狭窄任务，而离线视频感知模型与实时多模态助手之间存在差距。

### 目的

开发一个能够执行多种流式视频任务的通用交互式助手，统一处理连续视频流中的各种理解任务。

### 方法

构建大规模指令跟随数据集Streamo-Instruct-465K，涵盖多样化时间上下文和多任务监督，通过简化流水线进行端到端训练。

### 主要发现

Streamo在时间推理、响应交互和跨多种流式基准的广泛泛化方面表现出色，能够弥合离线视频感知模型与实时多模态助手之间的差距。

### 结论

Streamo代表了向统一、智能的视频理解迈出的重要一步，能够处理连续视频流中的多种任务。

### 翻译

我们提出了Streamo，一个实时流式视频大语言模型，可作为通用交互式助手。与现有专注于问答或字幕生成的在线视频模型不同，Streamo执行广泛的流式视频任务，包括实时叙述、动作理解、事件标注、时间事件定位和时间敏感问答。为了实现这种多功能性，我们构建了Streamo-Instruct-465K，这是一个专门用于流式视频理解的大规模指令跟随数据集。该数据集涵盖多样化的时间上下文和多任务监督，使异构流式任务能够进行统一训练。通过简化的流水线在指令跟随数据集上进行端到端训练后，Streamo在时间推理、响应交互和跨各种流式基准的广泛泛化方面表现出色。大量实验表明，Streamo弥合了离线视频感知模型与实时多模态助手之间的差距，在连续视频流中实现统一、智能的视频理解方面迈出了一步。


### 论文摘要

We present Streamo, a real-time streaming video LLM that serves as a general-purpose interactive assistant. Unlike existing online video models that focus narrowly on question answering or captioning, Streamo performs a broad spectrum of streaming video tasks, including real-time narration, action understanding, event captioning, temporal event grounding, and time-sensitive question answering. To develop such versatility, we construct Streamo-Instruct-465K, a large-scale instruction-following dataset tailored for streaming video understanding. The dataset covers diverse temporal contexts and multi-task supervision, enabling unified training across heterogeneous streaming tasks. After training end-to-end on the instruction-following dataset through a streamlined pipeline, Streamo exhibits strong temporal reasoning, responsive interaction, and broad generalization across a variety of streaming benchmarks. Extensive experiments show that Streamo bridges the gap between offline video perception models and real-time multimodal assistants, making a step toward unified, intelligent video understanding in continuous video streams.

---

