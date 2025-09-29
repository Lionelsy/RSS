# 今日论文推荐 - 2025-09-29

共 84 篇论文

---

## 1. Transfer Learning under Group-Label Shift: A Semiparametric Exponential Tilting Approach

**论文链接:** [http://arxiv.org/abs/2509.22268v1](http://arxiv.org/abs/2509.22268v1)

**作者:** Manli Cheng, Subha Maity, Qinglong Tian, Pengfei Li

**发布时间:** 2025-09-26

### GPT解析

### 总结

本文提出了一种适用于迁移学习场景的二分类新框架，能够处理源域和目标域间协变量分布和标签分布同时变化的情况。通过引入组标签偏移假设，该方法能够处理子群体不平衡问题并减少虚假相关性，从而提高对现实世界分布变化的鲁棒性。

### 背景

传统迁移学习方法通常假设只有协变量分布或标签分布发生变化，而现实世界中这两种分布可能同时发生变化，导致现有方法效果不佳。

### 目的

开发一种新的二分类框架，能够在协变量分布和标签分布同时变化的情况下有效进行迁移学习，提高目标域的分类准确性。

### 方法

采用指数倾斜公式建模联合分布差异，通过工具变量策略建立识别条件；开发两步似然估计程序，结合源结果逻辑回归和协变量条件似然估计；推导估计量的一致性和渐近正态性，并将理论扩展到接收者操作特征曲线和曲线下面积等指标。

### 主要发现

在子群体偏移场景下，该方法优于现有替代方法；使用waterbirds数据集的半合成应用证实了该方法能够有效传递信息并提高目标域分类准确性。

### 结论

所提出的组标签偏移框架能够有效处理迁移学习中的分布变化问题，提高分类准确性，对现实世界应用具有更好的鲁棒性。

### 翻译

我们提出了一种适用于迁移学习场景中二分类的新框架，其中源域和目标域的协变量分布和标签分布都可能发生变化。与传统协变量偏移或标签偏移假设不同，我们引入了组标签偏移假设，该假设能够处理子群体不平衡并减少虚假相关性，从而提高对现实世界分布变化的鲁棒性。为了建模联合分布差异，我们采用灵活的指数倾斜公式，并通过工具变量策略建立了温和且可验证的识别条件。我们开发了一种计算效率高的两步似然估计程序，将源结果模型逻辑回归与使用源域和目标域协变量的条件似然估计相结合。我们推导了所得估计量的一致性和渐近正态性，并将理论扩展到接收者操作特征曲线、曲线下面积和其他目标函数，解决了插入式分类器带来的非标准挑战。模拟研究表明，在子群体偏移场景下，我们的方法优于现有替代方法。使用waterbirds数据集的半合成应用进一步证实了所提出方法有效传递信息和提高目标域分类准确性的能力。


### 论文摘要

We propose a new framework for binary classification in transfer learning settings where both covariate and label distributions may shift between source and target domains. Unlike traditional covariate shift or label shift assumptions, we introduce a group-label shift assumption that accommodates subpopulation imbalance and mitigates spurious correlations, thereby improving robustness to real-world distributional changes. To model the joint distribution difference, we adopt a flexible exponential tilting formulation and establish mild, verifiable identification conditions via an instrumental variable strategy. We develop a computationally efficient two-step likelihood-based estimation procedure that combines logistic regression for the source outcome model with conditional likelihood estimation using both source and target covariates. We derive consistency and asymptotic normality for the resulting estimators, and extend the theory to receiver operating characteristic curves, the area under the curve, and other target functionals, addressing the nonstandard challenges posed by plug-in classifiers. Simulation studies demonstrate that our method outperforms existing alternatives under subpopulation shift scenarios. A semi-synthetic application using the waterbirds dataset further confirms the proposed method's ability to transfer information effectively and improve target-domain classification accuracy.

---

## 2. Towards Understanding Feature Learning in Parameter Transfer

**论文链接:** [http://arxiv.org/abs/2509.22056v1](http://arxiv.org/abs/2509.22056v1)

**作者:** Hua Yuan, Xuran Meng, Qiufeng Wang, Shiyu Xia, Ning Xu, Xu Yang, Jing Wang, Xin Geng, Yong Rui

**发布时间:** 2025-09-26

### GPT解析

### 总结

本研究探讨了迁移学习中部分参数迁移的理论基础，分析了在ReLU卷积神经网络中部分参数重用的条件和影响因素，并通过实验验证了理论发现。

### 背景

参数迁移是迁移学习的中心范式，通过在上游和下游模型间共享参数实现知识和任务复用。然而，当仅迁移上游模型的部分参数时，缺乏对这种部分参数重用有益条件的理论理解。

### 目的

填补理论空白，理解部分参数重用的条件和影响因素，分析ReLU卷积神经网络设置下参数迁移的机制。

### 方法

在理论框架下分析ReLU卷积神经网络的上游和下游模型，表征继承参数如何作为通用知识的载体，识别放大其对目标任务有益影响的关键因素。

### 主要发现

分析了为什么在某些情况下，迁移参数会导致目标任务测试准确率低于从头开始训练新模型的原因，并识别了影响参数迁移效果的关键因素。

### 结论

通过数值实验和真实世界数据实验验证了理论发现，为部分参数迁移提供了理论基础和实践指导。

### 翻译

参数迁移是迁移学习中的中心范式，通过在上游和下游模型之间共享模型参数，实现跨任务和跨域的知识重用。然而，当只有上游模型的一部分参数被转移到下游模型时，我们仍然缺乏对这种部分参数重用有益条件的理论理解，以及影响其有效性的因素。为填补这一空白，我们分析了上游和下游模型都是ReLU卷积神经网络(CNN)的设置。在这一理论框架内，我们表征了继承的参数如何作为通用知识的载体，并确定了放大其对目标任务有益影响的关键因素。此外，我们的分析提供了为什么在某些情况下，转移参数会导致目标任务测试准确率低于从头开始训练新模型的理解。进行了数值实验和真实世界数据实验，以经验性地验证我们的理论发现。


### 论文摘要

Parameter transfer is a central paradigm in transfer learning, enabling knowledge reuse across tasks and domains by sharing model parameters between upstream and downstream models. However, when only a subset of parameters from the upstream model is transferred to the downstream model, there remains a lack of theoretical understanding of the conditions under which such partial parameter reuse is beneficial and of the factors that govern its effectiveness. To address this gap, we analyze a setting in which both the upstream and downstream models are ReLU convolutional neural networks (CNNs). Within this theoretical framework, we characterize how the inherited parameters act as carriers of universal knowledge and identify key factors that amplify their beneficial impact on the target task. Furthermore, our analysis provides insight into why, in certain cases, transferring parameters can lead to lower test accuracy on the target task than training a new model from scratch. Numerical experiments and real-world data experiments are conducted to empirically validate our theoretical findings.

---

## 3. Coreset selection based on Intra-class diversity

**论文链接:** [http://arxiv.org/abs/2509.21380v1](http://arxiv.org/abs/2509.21380v1)

**作者:** Imran Ashraf, Mukhtar Ullah, Muhammad Faisal Nadeem, Muhammad Nouman Noor

**发布时间:** 2025-09-23

### GPT解析

### 总结

该研究提出了一种智能的轻量级机制用于选择深度学习训练的代表性数据子集（corset），解决了深度学习模型训练中计算资源需求大的问题，通过提取类内多样性形成聚类进行采样，在保持模型性能的同时减少了计算负担。

### 背景

深度学习模型已改变医疗保健领域，特别是在生物医学图像分类方面。训练深度学习模型（从头开始或迁移学习）需要大量计算资源和时间，尤其是超参数设计空间探索需要多次训练。随着数据集增长，寻找解决方案已成为研究热点。

### 目的

开发一种方法选择数据集的代表性子集（corset）用于训练和超参数搜索，以减少计算需求同时保持模型性能。

### 方法

提出一种智能机制提取类内多样性，形成每个类的聚类，然后利用这些聚类进行最终采样，而非简单的随机采样。这种方法避免了随机采样对主导类的偏见和无法捕获类内多样性的问题。

### 主要发现

在著名生物成像数据集上的分类实验表明，所提出的方案在多个性能指标上优于随机采样方法，能够有效减少计算需求同时保持模型性能。

### 结论

通过智能选择代表性数据子集，可以在不牺牲模型性能的情况下显著减少深度学习训练所需的计算资源和时间，为处理大规模生物医学图像数据集提供了有效解决方案。

### 翻译

深度学习模型已经改变了各个领域，包括医疗保健部门，特别是在生物医学图像分类方面，通过学习复杂特征，能够对复杂疾病进行准确诊断。最近的研究采用两种不同的方法训练深度学习模型：从头开始训练和迁移学习。由于训练涉及大规模数据集，这两种方法都需要大量的计算时间和资源。由于选择最佳超参数所需的设计空间探索，这些计算需求进一步增加，这通常需要多次训练。随着数据集大小的增长，解决这一问题最近引起了研究界的关注。一个可行的解决方案是为训练和超参数搜索选择数据集的一个子集。这个子集称为corset，必须是原始数据集的代表性集合。选择corset的直接方法可能是采用随机采样，但代价是损害原始数据集的代表性。随机采样的一个关键局限性是对不平衡数据集中主导类的偏见。即使数据集具有类间平衡，这种随机采样也无法捕获类内多样性。本研究通过引入一种智能的轻量级机制来选择corset解决了这个问题。具体来说，它提出了一种提取类内多样性的方法，形成每个类的聚类用于最终采样。我们在著名的生物成像数据集上进行了广泛的分类实验，证明了所提出方法的有效性。结果表明，在相同条件下，所提出的方案在多个性能指标上优于随机采样方法。


### 论文摘要

Deep Learning models have transformed various domains, including the healthcare sector, particularly biomedical image classification by learning intricate features and enabling accurate diagnostics pertaining to complex diseases. Recent studies have adopted two different approaches to train DL models: training from scratch and transfer learning. Both approaches demand substantial computational time and resources due to the involvement of massive datasets in model training. These computational demands are further increased due to the design-space exploration required for selecting optimal hyperparameters, which typically necessitates several training rounds. With the growing sizes of datasets, exploring solutions to this problem has recently gained the research community's attention. A plausible solution is to select a subset of the dataset for training and hyperparameter search. This subset, referred to as the corset, must be a representative set of the original dataset. A straightforward approach to selecting the coreset could be employing random sampling, albeit at the cost of compromising the representativeness of the original dataset. A critical limitation of random sampling is the bias towards the dominant classes in an imbalanced dataset. Even if the dataset has inter-class balance, this random sampling will not capture intra-class diversity. This study addresses this issue by introducing an intelligent, lightweight mechanism for coreset selection. Specifically, it proposes a method to extract intra-class diversity, forming per-class clusters that are utilized for the final sampling. We demonstrate the efficacy of the proposed methodology by conducting extensive classification experiments on a well-known biomedical imaging dataset. Results demonstrate that the proposed scheme outperforms the random sampling approach on several performance metrics for uniform conditions.

---

## 4. Multilingual Vision-Language Models, A Survey

**论文链接:** [http://arxiv.org/abs/2509.22123v1](http://arxiv.org/abs/2509.22123v1)

**作者:** Andrei-Alexandru Manea, Jindřich Libovický

**发布时间:** 2025-09-26

### GPT解析

### 总结

这是一项关于多语言视觉-语言模型的综述研究，考察了能够处理多种语言文本和图像的模型，分析了不同架构模型的特点，并探讨了语言中立性与文化意识之间的张力。

### 背景

多语言视觉-语言模型是一个正在发展的研究领域，涉及跨语言处理文本和图像的能力，目前存在语言中立性与文化意识之间的关键张力。

### 目的

综述多语言视觉-语言模型，分析不同架构模型的特点，研究语言中立性与文化意识之间的张力，并评估当前训练方法与评估基准的匹配程度。

### 方法

综述了31个模型和21个基准测试，涵盖了仅编码器和生成式架构，分析了训练方法和评估基准，识别了当前研究中的优势和不足。

### 主要发现

当前训练方法通过对比学习倾向于语言中立性，而文化意识依赖于多样化数据；三分之二的评估基准使用基于翻译的方法优先考虑语义一致性，但最近工作开始融入基于文化的内容；跨语言能力存在差异，训练目标与评估目标之间存在差距。

### 结论

多语言视觉-语言模型需要在保持语言中立性的同时增强文化意识，当前训练方法与评估基准之间存在不匹配，需要进一步研究以弥合这一差距。

### 翻译

该研究强调了多语言视觉-语言模型在全球化应用中的重要性，指出模型需要在保持跨语言一致性的同时适应不同文化背景，这对开发真正具有包容性的AI系统具有重要意义。


### 论文摘要

This survey examines multilingual vision-language models that process text and images across languages. We review 31 models and 21 benchmarks, spanning encoder-only and generative architectures, and identify a key tension between language neutrality (consistent cross-lingual representations) and cultural awareness (adaptation to cultural contexts). Current training methods favor neutrality through contrastive learning, while cultural awareness depends on diverse data. Two-thirds of evaluation benchmarks use translation-based approaches prioritizing semantic consistency, though recent work incorporates culturally grounded content. We find discrepancies in cross-lingual capabilities and gaps between training objectives and evaluation goals.

---

## 5. Enriching Knowledge Distillation with Intra-Class Contrastive Learning

**论文链接:** [http://arxiv.org/abs/2509.22053v1](http://arxiv.org/abs/2509.22053v1)

**作者:** Hua Yuan, Ning Xu, Xin Geng, Yong Rui

**发布时间:** 2025-09-26

### GPT解析

### 总结

该研究提出了一种改进知识蒸馏的方法，通过在教师模型训练中引入类内对比损失来丰富软标签中的类内信息，并解决了训练不稳定和收敛慢的问题。

### 背景

自知识蒸馏出现以来，研究重点在于如何有效利用教师模型生成的软标签。现有研究表明，软标签中的隐式知识源于数据中的多视图结构，同类样本内的特征变化有助于学生模型学习多样化表示并提高泛化能力。

### 目的

解决现有蒸馏方法中教师模型主要遵循真实标签而忽视同一类内多样化表示的问题，通过丰富软标签中的类内信息来提升知识蒸馏效果。

### 方法

在教师训练过程中引入类内对比损失，并将边界损失整合到类内对比学习中以提高训练稳定性和收敛速度。同时理论上分析了该损失对类内距离和类间距离的影响。

### 主要发现

类内对比损失能够有效丰富类内多样性，实验结果证明了所提出方法的有效性。

### 结论

通过改进教师模型训练过程，引入类内对比损失和边界损失，可以显著提升知识蒸馏的效果，使学生模型能够更好地学习多样化表示。

### 翻译

自从知识蒸馏出现以来，许多研究都集中在如何有效利用教师模型生成的软标签。现有研究表明，软标签中的隐式知识源于数据中存在的多视图结构。同类样本内的特征变化使学生模型能够通过学习多样化表示来更好地泛化。然而，在现有的蒸馏方法中，教师模型主要遵循真实标签作为目标，而没有考虑同一类内的多样化表示。因此，我们提出在教师训练过程中加入类内对比损失，以丰富软标签中包含的类内信息。在实践中，我们发现类内损失会导致训练不稳定并减慢收敛速度。为了缓解这些问题，将边界损失整合到类内对比学习中以提高训练稳定性和收敛速度。同时，我们理论上分析了这种损失对类内距离和类间距离的影响。已经证明类内对比损失可以丰富类内多样性。实验结果证明了所提出方法的有效性。


### 论文摘要

Since the advent of knowledge distillation, much research has focused on how the soft labels generated by the teacher model can be utilized effectively. Existing studies points out that the implicit knowledge within soft labels originates from the multi-view structure present in the data. Feature variations within samples of the same class allow the student model to generalize better by learning diverse representations. However, in existing distillation methods, teacher models predominantly adhere to ground-truth labels as targets, without considering the diverse representations within the same class. Therefore, we propose incorporating an intra-class contrastive loss during teacher training to enrich the intra-class information contained in soft labels. In practice, we find that intra-class loss causes instability in training and slows convergence. To mitigate these issues, margin loss is integrated into intra-class contrastive learning to improve the training stability and convergence speed. Simultaneously, we theoretically analyze the impact of this loss on the intra-class distances and inter-class distances. It has been proved that the intra-class contrastive loss can enrich the intra-class diversity. Experimental results demonstrate the effectiveness of the proposed method.

---

## 6. GRAM-TDI: adaptive multimodal representation learning for drug target interaction prediction

**论文链接:** [http://arxiv.org/abs/2509.21971v1](http://arxiv.org/abs/2509.21971v1)

**作者:** Feng Jiang, Amina Mollaysa, Hehuan Ma, Tommaso Mansi, Junzhou Huang, Mangal Prakash, Rui Liao

**发布时间:** 2025-09-26

### GPT解析

### 总结

本文提出了一种名为GRAMDTI的药物靶点相互作用(DTI)预测预训练框架，通过整合多模态分子和蛋白质输入，实现了超越现有方法的性能表现。

### 背景

药物靶点相互作用(DTI)预测是计算药物发现的核心，能够促进合理药物设计、药物重定位和机制洞察。然而，现有的深度学习方法主要依赖SMILES蛋白对，未能充分利用小分子和蛋白质的丰富多模态信息。

### 目的

开发一种能够整合多模态分子和蛋白质输入的预训练框架，提高DTI预测的性能和泛化能力。

### 方法

GRAMDTI框架将基于体积的对比学习扩展到四种模态，捕获高阶语义对齐；提出自适应模态丢弃机制，动态调节各模态在预训练中的贡献；整合IC50活性测量作为弱监督，将表示锚定在生物学上有意义的相互作用强度上。

### 主要发现

在四个公开数据集上的实验表明，GRAMDTI框架在DTI预测任务中始终优于最先进的基线方法，证明了高阶多模态对齐、自适应模态利用和辅助监督的有效性。

### 结论

高阶多模态对齐、自适应模态利用和辅助监督对于构建稳健和可推广的DTI预测模型至关重要，为药物发现提供了新的计算方法。

### 翻译

药物靶点相互作用(DTI)预测是计算药物发现的基础，能够促进合理设计、药物重定位和机制洞察。虽然深度学习已推动DTI建模发展，但现有方法主要依赖SMILES蛋白对，未能充分利用小分子和蛋白质的丰富多模态信息。我们引入了GRAMDTI，一个将多模态分子和蛋白质输入整合为统一表示的预训练框架。GRAMDTI将基于体积的对比学习扩展到四种模态，捕获超越传统成对方法的高阶语义对齐。为处理模态信息量，我们提出自适应模态丢弃，在预训练过程中动态调节各模态的贡献。此外，当可用时，IC50活性测量被整合为弱监督，将表示锚定在生物学上有意义的相互作用强度上。在四个公开数据集上的实验表明，GRAMDTI始终优于最先进的基线方法。我们的结果突显了高阶多模态对齐、自适应模态利用和辅助监督对稳健和可推广的DTI预测的益处。


### 论文摘要

Drug target interaction (DTI) prediction is a cornerstone of computational drug discovery, enabling rational design, repurposing, and mechanistic insights. While deep learning has advanced DTI modeling, existing approaches primarily rely on SMILES protein pairs and fail to exploit the rich multimodal information available for small molecules and proteins. We introduce GRAMDTI, a pretraining framework that integrates multimodal molecular and protein inputs into unified representations. GRAMDTI extends volume based contrastive learning to four modalities, capturing higher-order semantic alignment beyond conventional pairwise approaches. To handle modality informativeness, we propose adaptive modality dropout, dynamically regulating each modality's contribution during pre-training. Additionally, IC50 activity measurements, when available, are incorporated as weak supervision to ground representations in biologically meaningful interaction strengths. Experiments on four publicly available datasets demonstrate that GRAMDTI consistently outperforms state of the art baselines. Our results highlight the benefits of higher order multimodal alignment, adaptive modality utilization, and auxiliary supervision for robust and generalizable DTI prediction.

---

## 7. Enhancing Vehicle Detection under Adverse Weather Conditions with Contrastive Learning

**论文链接:** [http://arxiv.org/abs/2509.21916v1](http://arxiv.org/abs/2509.21916v1)

**作者:** Boying Li, Chang Liu, Petter Kyösti, Mattias Öhman, Devashish Singha Roy, Sofia Plazzi, Hamam Mokayed, Olle Hagner

**发布时间:** 2025-09-26

### GPT解析

### 总结

本文提出了一种sideload-CL-adaptation框架，利用未标注数据改进北欧地区无人机图像中的车辆检测性能。

### 背景

北欧地区从无人机图像检测车辆面临特殊挑战，包括强能见度问题和不同雪覆盖水平引起的域偏移。此外，遥感中的常见挑战如小目标、稀疏目标和计算成本限制也同时存在。标注数据获取成本高，而未标注数据通过无人机飞行可更经济地获取。

### 目的

开发一种能够利用未标注数据提高车辆检测性能的框架，同时使用轻量级模型以满足计算成本限制。

### 方法

提出一个基于CNN的表示提取器，通过在未标注数据上进行对比学习进行预训练，然后在微调阶段将预训练的表示提取器sideload到冻结的YOLO11n骨干网络中。通过广泛实验比较各种融合方法和粒度，以找到最优的sideload-CL-adaptation方案。

### 主要发现

提出的sideload-CL-adaptation模型在NVD数据集上将检测性能提高了3.8%至9.5%（以mAP50衡量），证明了该方法的有效性。

### 结论

该框架成功利用易获取的未标注数据显著提升了北欧地区复杂环境下无人机图像的车辆检测性能。

### 翻译

除了遥感中的常见挑战，如小目标、稀疏目标和计算成本限制外，从北欧地区的无人机图像中检测车辆还面临强能见度挑战和由不同雪覆盖水平引起的域偏移。虽然标注数据昂贵，但未标注数据可以通过简单飞行无人机更便宜地获取。在这项工作中，我们提出了一个sideload-CL-adaptation框架，使未标注数据能够用于改进使用轻量级模型的车辆检测。具体而言，我们建议在预训练阶段通过未标注数据上的对比学习训练一个基于CNN的表示提取器，然后在微调阶段将其sideload到冻结的YOLO11n骨干网络中。为了找到稳健的sideload-CL-adaptation，我们进行了广泛实验以比较各种融合方法和粒度。我们提出的sideload-CL-adaptation模型在NVD数据集上将检测性能提高了3.8%至9.5%（以mAP50衡量）。


### 论文摘要

Aside from common challenges in remote sensing like small, sparse targets and computation cost limitations, detecting vehicles from UAV images in the Nordic regions faces strong visibility challenges and domain shifts caused by diverse levels of snow coverage. Although annotated data are expensive, unannotated data is cheaper to obtain by simply flying the drones. In this work, we proposed a sideload-CL-adaptation framework that enables the use of unannotated data to improve vehicle detection using lightweight models. Specifically, we propose to train a CNN-based representation extractor through contrastive learning on the unannotated data in the pretraining stage, and then sideload it to a frozen YOLO11n backbone in the fine-tuning stage. To find a robust sideload-CL-adaptation, we conducted extensive experiments to compare various fusion methods and granularity. Our proposed sideload-CL-adaptation model improves the detection performance by 3.8% to 9.5% in terms of mAP50 on the NVD dataset.

---

## 8. EMG-UP: Unsupervised Personalization in Cross-User EMG Gesture Recognition

**论文链接:** [http://arxiv.org/abs/2509.21589v1](http://arxiv.org/abs/2509.21589v1)

**作者:** Nana Wang, Gen Li, Zhaoxin Fan, Suli Wang

**发布时间:** 2025-09-25

### GPT解析

### 总结

本研究提出了一种名为EMG-UP的新型无监督个性化框架，用于解决跨用户肌电手势识别中的泛化问题，通过两阶段自适应策略实现了最先进的性能。

### 背景

跨用户肌电(EMG)手势识别是实现可扩展和个性化人机交互的基本挑战。现有方法由于EMG信号的内在生物变异性（源于解剖异质性和多样的任务执行风格）而难以有效泛化到不同用户。

### 目的

开发一种能够有效解决跨用户手势识别中泛化问题的无监督个性化框架。

### 方法

采用两阶段自适应策略：(1)序列交叉视角对比学习，通过捕捉对用户间变化不变的内在信号模式，解耦鲁棒和用户特定的特征表示；(2)伪标签引导微调，使模型能够针对单个用户进行优化，无需访问源域数据。

### 主要发现

大量评估显示，EMG-UP达到了最先进的性能，在准确率上比先前方法至少高出2.0%。

### 结论

EMG-UP框架有效地解决了跨用户EMG手势识别中的泛化问题，为可扩展和个性化人机交互提供了新思路。

### 翻译

跨用户肌电(EMG)手势识别代表了在现实应用中实现可扩展和个性化人机交互的基本挑战。尽管付出了大量努力，但由于EMG信号的内在生物变异性（源于解剖异质性和多样的任务执行风格），现有方法难以有效跨用户泛化。为解决这一限制，我们引入了EMG-UP，一种用于跨用户手势识别中无监督个性化的新颖有效框架。所提出的框架利用两阶段自适应策略：(1)序列交叉视角对比学习，旨在通过捕捉对用户间变化不变的内在信号模式，解耦鲁健和用户特定的特征表示；(2)伪标签引导微调，使模型能够在无需访问源域数据的情况下针对单个用户进行优化。大量评估表明，EMG-UP实现了最先进的性能，在准确率上至少比先前方法高出2.0%。


### 论文摘要

Cross-user electromyography (EMG)-based gesture recognition represents a fundamental challenge in achieving scalable and personalized human-machine interaction within real-world applications. Despite extensive efforts, existing methodologies struggle to generalize effectively across users due to the intrinsic biological variability of EMG signals, resulting from anatomical heterogeneity and diverse task execution styles. To address this limitation, we introduce EMG-UP, a novel and effective framework for Unsupervised Personalization in cross-user gesture recognition. The proposed framework leverages a two-stage adaptation strategy: (1) Sequence-Cross Perspective Contrastive Learning, designed to disentangle robust and user-specific feature representations by capturing intrinsic signal patterns invariant to inter-user variability, and (2) Pseudo-Label-Guided Fine-Tuning, which enables model refinement for individual users without necessitating access to source domain data. Extensive evaluations show that EMG-UP achieves state-of-the-art performance, outperforming prior methods by at least 2.0% in accuracy.

---

## 9. Enhancing Contrastive Learning for Geolocalization by Discovering Hard Negatives on Semivariograms

**论文链接:** [http://arxiv.org/abs/2509.21573v1](http://arxiv.org/abs/2509.21573v1)

**作者:** Boyi Chen, Zhangyu Wang, Fabian Deuser, Johann Maximilian Zollner, Martin Werner

**发布时间:** 2025-09-25

### GPT解析

### 总结

提出了一种新颖的空间正则化对比学习策略，通过结合半变异函数来解决基于图像的全球地理定位中的挑战，特别是假阴性和困难负样本问题。

### 背景

基于图像的全球地理定位面临环境多样性、视觉模糊场景和许多地区缺乏明显地标的挑战，导致定位困难。

### 目的

解决对比学习方法中忽略地理空间潜在空间依赖性的问题，以及由此产生的假阴性和困难负样本难以区分的问题。

### 方法

提出一种空间正则化对比学习策略，结合半变异函数建模空间相关性随距离的变化，通过将特征空间中图像距离与地理距离关联来拟合半变异函数，并基于此识别困难负样本和假阴性样本。

### 主要发现

明确建模空间先验知识可以提高基于图像的地理定位性能，特别是在更细粒度的层面上。

### 结论

将提出的空间正则化策略集成到GeoCLIP中并在OSV5M数据集上评估，证实了该策略的有效性。

### 翻译

准确的全球规模图像地理定位具有挑战性，原因包括环境多样、视觉模糊场景以及许多地区缺乏明显地标。虽然对比学习方法通过将街景图像与相应位置的特征对齐显示出良好的性能，但它们忽略了地理空间中的潜在空间依赖性。因此，它们无法解决假阴性问题——即视觉和地理相似但被标记为负面的图像对，并且难以有效区分困难负样本——即视觉相似但地理距离远的样本。为解决这一问题，我们提出了一种新颖的空间正则化对比学习策略，结合了半变异函数，这是一种用于建模空间相关性如何随距离变化的地理统计工具。我们通过将特征空间中图像的距离与其地理距离相关联来拟合半变异函数，捕捉空间相关性中的预期视觉内容。利用拟合的半变异函数，我们将给定空间距离处的预期视觉差异定义为参考，以识别困难负样本和假阴性样本。我们将此策略集成到GeoCLIP中并在OSV5M数据集上评估，证明明确建模空间先验知识可以提高基于图像的地理定位性能，特别是在更细粒度的层面上。


### 论文摘要

Accurate and robust image-based geo-localization at a global scale is challenging due to diverse environments, visually ambiguous scenes, and the lack of distinctive landmarks in many regions. While contrastive learning methods show promising performance by aligning features between street-view images and corresponding locations, they neglect the underlying spatial dependency in the geographic space. As a result, they fail to address the issue of false negatives -- image pairs that are both visually and geographically similar but labeled as negatives, and struggle to effectively distinguish hard negatives, which are visually similar but geographically distant. To address this issue, we propose a novel spatially regularized contrastive learning strategy that integrates a semivariogram, which is a geostatistical tool for modeling how spatial correlation changes with distance. We fit the semivariogram by relating the distance of images in feature space to their geographical distance, capturing the expected visual content in a spatial correlation. With the fitted semivariogram, we define the expected visual dissimilarity at a given spatial distance as reference to identify hard negatives and false negatives. We integrate this strategy into GeoCLIP and evaluate it on the OSV5M dataset, demonstrating that explicitly modeling spatial priors improves image-based geo-localization performance, particularly at finer granularity.

---

## 10. Contrastive Mutual Information Learning: Toward Robust Representations without Positive-Pair Augmentations

**论文链接:** [http://arxiv.org/abs/2509.21511v1](http://arxiv.org/abs/2509.21511v1)

**作者:** Micha Livne

**发布时间:** 2025-09-25

**备注:** Preprint. 9 pages main manuscript, 23 pages with appendix

### GPT解析

### 总结

本文提出了对比互信息机(cMIM)，一种结合了对比学习和互信息机(MIM)的表示学习框架，旨在同时提升判别式和生成式任务的表现。

### 背景

表示学习中的核心挑战是学习能够良好迁移到多样化下游任务的表示。现有范式包括对比学习、自监督掩码和去噪自编码器，它们在应对这一挑战时存在不同权衡。互信息机(MIM)最大化输入和潜在表示之间的互信息并促进代码聚类，但在判别式任务上表现不佳。

### 目的

开发一种能够同时保留MIM的生成保真度并引入全局判别结构的框架，从而在判别式和生成式任务上都取得良好表现。

### 方法

提出cMIM(对比互信息机)，这是MIM的对比扩展，结合了对比目标和MIM的互信息最大化方法。同时引入'信息丰富嵌入'技术，用于从编码器-解码器模型中提取增强特征，提高判别性能而无需额外训练。

### 主要发现

1. cMIM不需要正面数据增强，并且对批次大小的敏感性显著低于InfoNCE；2. '信息丰富嵌入'技术可以提升判别性能，无需额外训练，并且可以广泛应用于MIM以外的模型；3. 在视觉和分子基准测试中，cMIM在分类和回归任务上均优于MIM和InfoNCE，同时保持了竞争性的重建质量。

### 结论

cMIM被定位为一个统一的表示学习框架，推进了模型能够有效服务于判别式和生成式应用的目标。

### 翻译

学习能够良好迁移到多样化下游任务的表示仍然是表示学习中的一个核心挑战。现有范式——对比学习、自监督掩码和去噪自编码器——通过不同的权衡来应对这一挑战。我们引入了对比互信息机(cMIM)，这是一个将互信息机(MIM)与对比目标相结合的概率框架。虽然MIM最大化输入和潜在表示之间的互信息并促进代码聚类，但在判别式任务上表现不佳。cMIM通过引入全局判别结构同时保留MIM的生成保真度来解决这一差距。我们的贡献有三方面。首先，我们提出了cMIM，这是MIM的对比扩展，它不需要正面数据增强，并且对批次大小的敏感性显著低于InfoNCE。其次，我们引入了'信息丰富嵌入'，这是一种从编码器-解码器模型中提取增强特征的通用技术，可以在无需额外训练的情况下提高判别性能，并且可以广泛应用于MIM以外的领域。第三，我们在视觉和分子基准测试中提供了经验证据，表明cMIM在分类和回归任务上均优于MIM和InfoNCE，同时保持了竞争性的重建质量。这些结果将cMIM定位为表示学习的统一框架，推进了模型能够有效服务于判别式和生成式应用的目标。


### 论文摘要

Learning representations that transfer well to diverse downstream tasks remains a central challenge in representation learning. Existing paradigms -- contrastive learning, self-supervised masking, and denoising auto-encoders -- balance this challenge with different trade-offs. We introduce the {contrastive Mutual Information Machine} (cMIM), a probabilistic framework that extends the Mutual Information Machine (MIM) with a contrastive objective. While MIM maximizes mutual information between inputs and latents and promotes clustering of codes, it falls short on discriminative tasks. cMIM addresses this gap by imposing global discriminative structure while retaining MIM's generative fidelity. Our contributions are threefold. First, we propose cMIM, a contrastive extension of MIM that removes the need for positive data augmentation and is substantially less sensitive to batch size than InfoNCE. Second, we introduce {informative embeddings}, a general technique for extracting enriched features from encoder-decoder models that boosts discriminative performance without additional training and applies broadly beyond MIM. Third, we provide empirical evidence across vision and molecular benchmarks showing that cMIM outperforms MIM and InfoNCE on classification and regression tasks while preserving competitive reconstruction quality. These results position cMIM as a unified framework for representation learning, advancing the goal of models that serve both discriminative and generative applications effectively.

---

## 11. Diffusion-Augmented Contrastive Learning: A Noise-Robust Encoder for Biosignal Representations

**论文链接:** [http://arxiv.org/abs/2509.20048v2](http://arxiv.org/abs/2509.20048v2)

**作者:** Rami Zewail

**发布时间:** 2025-09-24

### GPT解析

### 总结

研究提出了一种名为DACL的新型混合框架，结合扩散模型和监督对比学习，用于生物信号表示学习，通过扩散过程生成噪声视图并学习鲁棒表示，在ECG数据集上取得了0.7815的竞争性AUROC值。

### 背景

学习生物信号的鲁棒表示通常面临设计有效数据增强的挑战，传统方法无法捕捉生理数据中固有的复杂变化。

### 目的

提出一种新的混合框架，解决传统数据增强方法在生理数据上的局限性，学习对噪声具有鲁棒性的生物信号表示。

### 方法

提出名为Diffusion-Augmented Contrastive Learning (DACL)的混合框架，融合扩散模型和监督对比学习概念；在基于Scattering Transformer特征训练的轻量级变分自编码器创建的潜在空间上运行；利用扩散前向过程作为数据增强技术生成噪声视图；使用监督对比目标训练U-Net风格编码器学习跨不同扩散时间步骤的鲁棒表示。

### 主要发现

在PhysioNet 2017 ECG数据集上评估该方法，达到了0.7815的竞争性AUROC值。

### 结论

通过使用扩散过程本身驱动对比目标，为表示学习建立了新范式，创建了对噪声不变的嵌入，显示出良好的类别可分离性基础。

### 翻译

学习生物信号的鲁棒表示通常面临设计有效数据增强的挑战。传统方法可能无法捕捉生理数据中固有的复杂变化。在此背景下，我们提出了一种新的混合框架Diffusion-Augmented Contrastive Learning (DACL)，该框架融合了扩散模型和监督对比学习的概念。DACL框架在我们新颖的Scattering Transformer (ST)特征[12]训练的轻量级变分自编码器(VAE)创建的潜在空间上运行。它利用扩散前向过程作为数据增强技术，生成这些潜在嵌入的多个噪声视图。然后使用监督对比目标训练U-Net风格的编码器，学习一个在不同扩散时间步骤上对噪声具有鲁棒性的表示。我们在PhysioNet 2017 ECG数据集上评估了这个概念验证方法，取得了0.7815的竞争性AUROC值。这项工作通过使用扩散过程本身来驱动对比目标，为表示学习建立了新范式，创建了对噪声不变的嵌入，显示出良好的类别可分离性基础。


### 论文摘要

Learning robust representations for biosignals is often hampered by the challenge of designing effective data augmentations.Traditional methods can fail to capture the complex variations inherent in physiological data. Within this context, we propose a novel hybrid framework, Diffusion-Augmented Contrastive Learning (DACL), that fuses concepts from diffusion models and supervised contrastive learning. The DACL framework operates on a latent space created by a lightweight Variational Autoencoder (VAE) trained on our novel Scattering Transformer (ST) features [12]. It utilizes the diffusion forward process as a principled data augmentation technique to generate multiple noisy views of these latent embeddings. A U-Net style encoder is then trained with a supervised contrastive objective to learn a representation that balances class discrimination with robustness to noise across various diffusion time steps. We evaluated this proof-of-concept method on the PhysioNet 2017 ECG dataset, achieving a competitive AUROC of 0.7815. This work establishes a new paradigm for representation learning by using the diffusion process itself to drive the contrastive objective, creating noise-invariant embeddings that demonstrate a strong foundation for class separability.

---

## 12. Transport Based Mean Flows for Generative Modeling

**论文链接:** [http://arxiv.org/abs/2509.22592v1](http://arxiv.org/abs/2509.22592v1)

**作者:** Elaheh Akbari, Ping He, Ahmadreza Moradipari, Yikun Bai, Soheil Kolouri

**发布时间:** 2025-09-26

### GPT解析

### 总结

本研究通过整合基于最优传输的采样策略到Mean Flow框架中，解决了Mean Flows在连续数据生成中无法忠实近似原始多步flow-matching过程的问题，实现了在保持生成性能的同时显著提高推理速度的单步生成模型。

### 背景

Flow-matching生成模型已成为连续数据生成的强大范式，在图像、3D形状和点云等领域取得了最先进的结果。然而，这些模型由于需要大量顺序采样步骤而面临推理速度慢的问题。最近的工作试图通过减少采样步骤数量来加速推理，其中Mean Flows提供了一种单步生成方法，在保持强大生成性能的同时带来了显著的速度提升。

### 目的

解决Mean Flows在许多连续领域中无法忠实地近似原始多步flow-matching过程的行为，开发能够更好保留原始多步flow过程保真度和多样性的单步生成器。

### 方法

将基于最优传输(optimal transport)的采样策略整合到Mean Flow框架中，创建能够更好保留原始多步flow过程保真度和多样性的单步生成器。

### 主要发现

在受控的低维设置以及图像生成、图像到图像转换和点云生成等高维任务上的实验表明，该方法在单步生成建模中实现了卓越的推理准确性，优于现有的Mean Flows方法。

### 结论

通过整合基于最优传输的采样策略，Mean Flow框架能够生成更忠实于原始多步过程的单步生成器，解决了flow-matching模型的主要瓶颈，在保持生成性能的同时显著提高了推理速度。

### 翻译

Flow-matching生成模型已成为连续数据生成的强大范式，在图像、3D形状和点云等领域取得了最先进的结果。尽管取得了成功，但这些模型由于需要大量顺序采样步骤而面临推理速度慢的问题。最近的工作试图通过减少采样步骤数量来加速推理。特别是，Mean Flows提供了一种单步生成方法，在保持强大生成性能的同时带来了显著的速度提升。然而，在许多连续领域中，Mean Flows无法忠实地近似原始多步flow-matching过程的行为。在本工作中，我们通过将基于最优传输的采样策略整合到Mean Flow框架中，解决了这一局限性，使单步生成器能够更好地保留原始多步flow过程的保真度和多样性。在受控的低维设置以及图像生成、图像到图像转换和点云生成等高维任务上的实验表明，我们的方法在单步生成建模中实现了卓越的推理准确性。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决Flow-matching生成模型推理速度慢的问题。传统Flow-matching模型需要大量顺序采样步骤，限制了实际应用。虽然MeanFlow方法提供了一步生成方案，但在许多连续领域中无法很好地近似原始多步过程。这个问题很重要，因为生成模型在图像、3D形状和点云等领域有广泛应用，提高推理速度同时保持生成质量是推动这些模型实际部署的关键。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者观察到MeanFlow方法虽能实现快速推理，但在连续数据中无法很好地保留原始多步Flow-matching的行为。他们思考如何结合最优传输(OT)的轨迹拉直原则与MeanFlow的时间平均公式，以创建更高效的一步生成轨迹。该方法借鉴了多个现有工作：Flow Matching和Diffusion Models两种生成框架、最优传输理论(特别是Benamou-Brenier动态公式)、MeanFlow的平均速度场学习，以及Mini-batch OT Flow Matching的耦合策略。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是将最优传输的轨迹拉直原则与MeanFlow的时间平均公式相结合，产生更直效的一步生成轨迹，通过OT基础的耦合与mean-flow监督相结合，实现几何感知和高效的一步生成。实现流程分为训练和推理两阶段：训练时，从源和目标分布采样小批次数据，计算OT计划，采样时间点，计算中间点和目标速度，计算目标平均速度场，通过最小化学习速度场与目标速度场的差异来更新参数；推理时，直接使用学习到的平均速度场一步生成目标数据，x1 ≈ x0 + u1,0(x0)。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1) 统一框架，将条件flow matching、mini-batch OT flow matching和mean flow方法统一于OT-MF下；2) 改进的效率和准确性，在点云生成和图像转换中保留一步生成能力同时提高质量；3) 可扩展训练，集成线性OT和分层OT等加速方法。相比之前工作，OT-MF通过OT策略实现更直效轨迹提高生成质量，相比MeanFlow更好地保留原始多步过程的行为，相比传统Flow Matching大幅提升推理速度，同时引入近似OT变体在保持性能的同时提高计算效率。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '本文提出OT-MF方法，结合最优传输与平均流实现一步高质量生成，在保持推理速度的同时显著提升了生成模型的性能。'}


### 论文摘要

Flow-matching generative models have emerged as a powerful paradigm for continuous data generation, achieving state-of-the-art results across domains such as images, 3D shapes, and point clouds. Despite their success, these models suffer from slow inference due to the requirement of numerous sequential sampling steps. Recent work has sought to accelerate inference by reducing the number of sampling steps. In particular, Mean Flows offer a one-step generation approach that delivers substantial speedups while retaining strong generative performance. Yet, in many continuous domains, Mean Flows fail to faithfully approximate the behavior of the original multi-step flow-matching process. In this work, we address this limitation by incorporating optimal transport-based sampling strategies into the Mean Flow framework, enabling one-step generators that better preserve the fidelity and diversity of the original multi-step flow process. Experiments on controlled low-dimensional settings and on high-dimensional tasks such as image generation, image-to-image translation, and point cloud generation demonstrate that our approach achieves superior inference accuracy in one-step generative modeling.

---

## 13. The Flood Complex: Large-Scale Persistent Homology on Millions of Points

**论文链接:** [http://arxiv.org/abs/2509.22432v1](http://arxiv.org/abs/2509.22432v1)

**作者:** Florian Graf, Paolo Pellizzoni, Martin Uray, Stefan Huber, Roland Kwitt

**发布时间:** 2025-09-26

### GPT解析

### 总结

本文提出了一种名为Flood复形的新方法，用于计算大规模欧几里得点云数据的持续同调(PH)，解决了传统方法在计算效率和可扩展性方面的挑战。

### 背景

计算大规模点云数据的持续同调面临严重计算限制，因为常用的Vietoris-Rips复形会呈指数增长。虽然Alpha复形和稀疏Rips近似等替代方法存在，但它们仍会产生大量单纯形，限制了在大规模点云上的应用。

### 目的

开发一种可扩展的方法，能够高效计算大规模点云数据的持续同调，特别是在下游机器学习任务中的应用。

### 方法

引入Flood复形，受Alpha复形和Witness复形的优势启发。在给定过滤值r时，Flood复形包含点云小子集的Delaunay三角剖分中被半径为r的球体完全覆盖的所有单纯形，这一过程称为'flooding'。

### 主要发现

Flood复形允许高效的PH计算，具有理想的理论性质，适合GPU并行化。实验表明，该方法可在数百万个3D点上计算到2维的PH。在对象分类任务中，对于几何或拓扑复杂对象，该方法性能优于其他基于PH的方法和用于点云数据的神经网络。

### 结论

Flood复形为大规模点云数据的持续同调计算提供了一种有效解决方案，其扩展能力对于处理复杂几何或拓扑对象至关重要，在机器学习应用中展现出优越性能。

### 翻译

我们考虑为大规模欧几里得点云数据计算持续同调(PH)的问题，旨在用于下游机器学习任务，其中最广泛使用的Vietoris-Rips复形的指数增长带来了严重的计算限制。虽然存在更可扩展的替代方法，如Alpha复形或稀疏Rips近似，但它们通常仍会产生大量单纯形，这给复形的构建和随后的PH计算带来了挑战，限制了它们在大规模点云上的使用。为缓解这些问题，我们引入了Flood复形，受Alpha复形和Witness复形构建的优势启发。非正式地说，在给定的过滤值r≥0时，Flood复形包含点云X的一个小子集的Delaunay三角剖分中的所有单纯形，这些单纯形完全被从X发出的半径为r的球体覆盖，这个过程我们称为flooding。我们的构造允许高效的PH计算，具有几种理想的理论性质，并且适合GPU并行化。在3D点云数据上的扩展实验表明，我们可以在数百万个点上计算到2维的PH。重要的是，在真实世界和合成数据上评估对象分类性能时，我们提供了证据表明这种扩展能力是必要的，特别是当对象在几何或拓扑上复杂时，性能优于其他基于PH的方法和用于点云数据的神经网络。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决计算大规模欧几里得点云数据的持久同调(Persistent Homology, PH)时面临的计算挑战。这个问题很重要，因为持久同调是拓扑数据分析的核心工具，能揭示数据的拓扑和几何特性；随着数据规模增大，传统方法变得不可行；许多机器学习应用(如图分类、时间序列预测)依赖于PH提取特征；大规模点云数据在科学计算(如冷冻电镜图像)、3D扫描等领域很常见。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者分析了现有方法的局限性：Vietoris-Rips复形计算复杂度高；Alpha复形对大规模点云仍效率不足；子采样会丢失小拓扑特征；Witness复形构造过程脆弱。作者借鉴了Alpha复形(基于Delaunay三角剖分)和Witness复形(使用小地标集)的优点，设计出Flood复形：在给定过滤值r时，包含从点云小子集L的Delaunay三角剖分中所有被半径为r的球完全覆盖的单纯形，结合了两者的优势同时避免了它们的缺点。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是：使用小地标点集构建Delaunay三角剖分，为每个单纯形计算被整个点云覆盖所需的最小半径作为过滤值。实现流程：1)选择地标点集(通常用最远点采样)；2)计算地标集的Delaunay三角剖分；3)对每个单纯形：计算包围球、创建掩码、在单纯形上离散采样点、计算过滤值；4)构建过滤的Flood复形；5)计算持久同调；6)将持久图转换为向量表示用于下游任务。作者通过GPU并行化和掩码技术提高效率。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点：1)Flood复形构造结合了Alpha和Witness复形的优点；2)提供了近似质量的稳定性和理论保证；3)设计了高效的GPU并行算法；4)能够处理数百万点的点云。相比之前工作的不同：与子采样相比保留了小拓扑特征；与Witness复形相比更稳定且不需精细控制距离截止值；与Alpha复形相比使用更小的地标集效率更高；与稀疏Rips相比有更好的理论保证且更适合GPU并行化。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'Flood复形通过结合Alpha复形的拓扑特性和Witness复形的数据驱动方法，实现了对大规模点云数据的高效持久同调计算，在保持拓扑信息完整性的同时显著提升了计算效率，为拓扑数据分析在机器学习等领域的应用提供了新的可能性。'}


### 论文摘要

We consider the problem of computing persistent homology (PH) for large-scale Euclidean point cloud data, aimed at downstream machine learning tasks, where the exponential growth of the most widely-used Vietoris-Rips complex imposes serious computational limitations. Although more scalable alternatives such as the Alpha complex or sparse Rips approximations exist, they often still result in a prohibitively large number of simplices. This poses challenges in the complex construction and in the subsequent PH computation, prohibiting their use on large-scale point clouds. To mitigate these issues, we introduce the Flood complex, inspired by the advantages of the Alpha and Witness complex constructions. Informally, at a given filtration value $r\geq 0$, the Flood complex contains all simplices from a Delaunay triangulation of a small subset of the point cloud $X$ that are fully covered by balls of radius $r$ emanating from $X$, a process we call flooding. Our construction allows for efficient PH computation, possesses several desirable theoretical properties, and is amenable to GPU parallelization. Scaling experiments on 3D point cloud data show that we can compute PH of up to dimension 2 on several millions of points. Importantly, when evaluating object classification performance on real-world and synthetic data, we provide evidence that this scaling capability is needed, especially if objects are geometrically or topologically complex, yielding performance superior to other PH-based methods and neural networks for point cloud data.

---

## 14. Wavelet-Induced Rotary Encodings: RoPE Meets Graphs

**论文链接:** [http://arxiv.org/abs/2509.22259v1](http://arxiv.org/abs/2509.22259v1)

**作者:** Isaac Reid, Arijit Sehanobish, Cedrik Höfs, Bruno Mlodozeniec, Leonhard Vulpius, Federico Barbero, Adrian Weller, Krzysztof Choromanski, Richard E. Turner, Petar Veličković

**发布时间:** 2025-09-26

### GPT解析

### 总结

本文介绍了一种名为WIRE（小波诱导的旋转编码）的新方法，它扩展了在大型语言模型和视觉Transformer中流行的RoPE算法，使其能够应用于图结构数据。

### 背景

RoPE（RotaryPosition Encodings）是一种在大型语言模型和视觉Transformer中广泛使用的算法，但主要针对序列数据设计，难以直接应用于图结构数据。

### 目的

扩展RoPE算法，使其能够处理图结构数据，同时保持或提高其在各种任务中的性能，特别是在底层图结构重要的场景中。

### 方法

作者提出了WIRE（Wavelet-Induced Rotary Encodings）方法，这是一种基于小波的旋转编码方法，专门设计用于图结构数据。WIRE是RoPE的泛化，在网格图的特例中可以恢复RoPE。

### 主要发现

WIRE具有多种理想的理论特性，包括节点排序置换下的等变性、与线性注意力机制的兼容性，以及在特定假设下对图电阻距离的渐近依赖性。在多种合成和真实世界任务中，WIRE被证明是有效的，特别是在底层图结构重要的场景中。

### 结论

WIRE是一种有效的图结构数据处理方法，它成功地将RoPE扩展到图数据领域，并在多种任务中展现出良好的性能，特别是在需要考虑图结构信息的应用场景中。

### 翻译

我们介绍WIRE：小波诱导的旋转编码。WIRE将RotaryPosition Encodings（RoPE）——一种在大型语言模型和视觉Transformer中流行的算法——扩展到图结构数据。我们证明WIRE比RoPE更通用，在网格图的特例中可以恢复后者。WIRE还具有许多理想的理论特性，包括在节点排序置换下的等变性、与线性注意力的兼容性，以及（在特定假设下）对图电阻距离的渐近依赖性。我们在一系列合成和真实世界任务上测试了WIRE，包括识别单色子图、点云语义分割以及更标准的图基准测试。我们发现，在底层图结构重要的场景中，WIRE是有效的。


### 论文摘要

We introduce WIRE: Wavelet-Induced Rotary Encodings. WIRE extends Rotary Position Encodings (RoPE), a popular algorithm in LLMs and ViTs, to graph-structured data. We demonstrate that WIRE is more general than RoPE, recovering the latter in the special case of grid graphs. WIRE also enjoys a host of desirable theoretical properties, including equivariance under node ordering permutation, compatibility with linear attention, and (under select assumptions) asymptotic dependence on graph resistive distance. We test WIRE on a range of synthetic and real-world tasks, including identifying monochromatic subgraphs, semantic segmentation of point clouds, and more standard graph benchmarks. We find it to be effective in settings where the underlying graph structure is important.

---

## 15. Joint graph entropy knowledge distillation for point cloud classification and robustness against corruptions

**论文链接:** [http://arxiv.org/abs/2509.22150v1](http://arxiv.org/abs/2509.22150v1)

**作者:** Zhiqiang Tian, Weigang Li, Junwei Hu, Chunhua Deng

**发布时间:** 2025-09-26

### GPT解析

### 总结

该研究提出了一种名为联合图熵知识蒸馏(JGEKD)的分类策略，适用于非独立同分布的3D点云数据，通过基于联合图熵构建损失函数实现类别相关性的知识迁移。

### 背景

3D点云分类任务通常假设类别事件是独立同分布的(IID)，但这种假设破坏了类别之间的相关性。

### 目的

提出一种适用于非独立同分布的3D点云数据的分类策略，实现类别相关性的知识迁移。

### 方法

提出联合图熵知识蒸馏(JGEKD)策略，通过基于联合图熵构建损失函数实现知识迁移；使用联合图捕获类别间的隐藏关系；构建孪生结构处理空间变换不变的3D点云；开发自知识蒸馏和教师知识蒸馏两种框架促进信息传递；利用框架在点云及其损坏形式间实现知识迁移提高鲁棒性。

### 主要发现

在ScanObject、ModelNet40、ScanntV2_cls和ModelNet-C数据集上的实验证明，JGEKD策略能够取得具有竞争力的结果。

### 结论

JGEKD策略能有效处理非独立同分布的3D点云数据，通过联合图熵知识蒸馏实现类别相关性迁移，并在多个数据集上展现优异性能。

### 翻译

分类任务在三维点云中通常假设类别事件遵循独立同分布，尽管这种假设破坏了类别之间的相关性。本研究提出了一种分类策略——联合图熵知识蒸馏(JGEKD)，适用于非独立同分布的三维点云数据，该策略通过基于联合图熵构建损失函数实现类别相关性的知识迁移。首先，我们使用联合图捕获类别之间的隐藏关系，通过计算图的熵实现知识蒸馏来训练我们的模型。随后，为处理空间变换不变的三维点云，我们构建孪生结构并开发了两种框架：自知识蒸馏和教师知识蒸馏，以促进相同数据不同变换形式之间的信息传递。此外，我们利用上述框架在点云及其损坏形式之间实现知识迁移，提高模型对损坏的鲁棒性。在ScanObject、ModelNet40、ScanntV2_cls和ModelNet-C上的大量实验证明，所提出的策略能够取得具有竞争力的结果。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决3D点云分类任务中的三个关键问题：1) 传统方法假设类别事件独立同分布(IID)，忽略了类别间的相关性；2) 点云数据对空间变换敏感，不同变换下同一物体可能呈现相似表示；3) 点云数据易受噪声、遮挡等因素损坏，模型鲁棒性不足。这些问题在现实中很重要，因为点云数据规模小、标注成本高，忽略类别关系导致模型泛化能力差，而对变换和损坏的敏感性限制了模型在真实场景中的应用效果。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先通过分析指出IID假设在点云分类中的局限性，并通过类别相关性矩阵证明不同类别间存在关联。然后借鉴了知识蒸馏技术来建模类别间关系，使用图结构捕捉潜在联系。针对空间变换不变性，作者构建了孪生网络结构；为增强鲁棒性，开发了对抗训练策略。该方法综合借鉴了现有工作：包括基于特征层和logits的知识蒸馏方法、多视图和基于点的点云处理方法、多种数据增强策略以及对抗训练和输入随机化技术等，但创新性地将这些技术结合起来解决非IID点云分类问题。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是通过联合图建模类别间潜在关系，利用图熵量化关系信息，并通过知识蒸馏实现类别间知识传递，同时结合孪生网络处理点云的不同表示，最终增强模型对损坏的鲁棒性。整体流程包括：1) 构建联合图表示类别关系；2) 计算联合图熵作为损失函数；3) 设计JGEsKD和JGEtKD两种蒸馏框架；4) 应用对抗训练处理标准点云和损坏点云；5) 结合交叉熵和蒸馏损失进行模型训练。这种方法使模型能够学习类别间的隐含关系，适应不同变换，并对损坏点云保持鲁棒性。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1) 提出联合图熵知识蒸馏(JGEKD)，首次使用联合图建模类别关系并通过图熵实现知识传递；2) 设计JGEsKD和JGEtKD两种框架，分别处理自蒸馏和教师蒸馏场景；3) 开发基于JGEKD的对抗训练策略增强模型鲁棒性。相比之前工作，不同之处在于：突破了传统IID假设限制，显式建模类别间关系；将知识蒸馏应用于类别间而非仅模型间知识传递；通过蒸馏而非简单数据增强增强鲁棒性；结合交叉熵和蒸馏损失全面学习特征表示。这些创新使模型在非IID点云数据上表现更稳定，对变换和损坏更具鲁棒性。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '这篇论文提出了一种基于联合图熵知识蒸馏的创新方法，有效解决了点云分类任务中类别关系建模、空间变换不变性和损坏鲁棒性三大挑战，显著提升了模型在非独立同分布点云数据上的分类性能和鲁棒性。'}


### 论文摘要

Classification tasks in 3D point clouds often assume that class events \replaced{are }{follow }independent and identically distributed (IID), although this assumption destroys the correlation between classes. This \replaced{study }{paper }proposes a classification strategy, \textbf{J}oint \textbf{G}raph \textbf{E}ntropy \textbf{K}nowledge \textbf{D}istillation (JGEKD), suitable for non-independent and identically distributed 3D point cloud data, \replaced{which }{the strategy } achieves knowledge transfer of class correlations through knowledge distillation by constructing a loss function based on joint graph entropy. First\deleted{ly}, we employ joint graphs to capture add{the }hidden relationships between classes\replaced{ and}{,} implement knowledge distillation to train our model by calculating the entropy of add{add }graph.\replaced{ Subsequently}{ Then}, to handle 3D point clouds \deleted{that is }invariant to spatial transformations, we construct \replaced{S}{s}iamese structures and develop two frameworks, self-knowledge distillation and teacher-knowledge distillation, to facilitate information transfer between different transformation forms of the same data. \replaced{In addition}{ Additionally}, we use the above framework to achieve knowledge transfer between point clouds and their corrupted forms, and increase the robustness against corruption of model. Extensive experiments on ScanObject, ModelNet40, ScanntV2\_cls and ModelNet-C demonstrate that the proposed strategy can achieve competitive results.

---

## 16. Self-Supervised Point Cloud Completion based on Multi-View Augmentations of Single Partial Point Cloud

**论文链接:** [http://arxiv.org/abs/2509.22132v1](http://arxiv.org/abs/2509.22132v1)

**作者:** Jingjing Lu, Huilong Pi, Yunchuan Qin, Zhuo Tang, Ruihui Li

**发布时间:** 2025-09-26

### GPT解析

### 总结

该研究提出了一种新颖的自监督点云补全方法，通过多视图增强和引入Mamba模型，解决了现有方法在真实数据集上泛化能力有限的问题，实现了在合成和真实数据集上的最先进结果。

### 背景

点云补全旨在从部分观测中重建完整形状。当前方法存在局限性：监督方法依赖真实标签，受合成到真实域差距影响；无监督方法需要完整点云构建非配对数据；弱监督方法需要多视图观测；现有自监督方法因信号能力有限而效果不佳。

### 目的

克服现有点云补全方法的局限性，提出一种新的自监督点云补全方法，提高模型在真实世界数据集上的性能。

### 方法

1) 基于单个部分点云的多视图增强设计新自监督信号；2) 首次将Mamba模型引入自监督点云补全任务；3) 鼓励模型生成更高质量的点云。

### 主要发现

在合成和真实世界数据集上的实验表明，该方法实现了最先进的结果。

### 结论

所提出的自监督点云补全方法通过多视图增强和Mamba模型的整合，有效克服了现有方法的局限性，在点云补全任务上取得了显著改进。

### 翻译

点云补全旨在从部分观测中重建完整形状。尽管当前方法已取得了显著性能，但仍存在一些局限性：监督方法严重依赖真实标签，由于合成到真实的域差距，限制了它们在真实世界数据集上的泛化能力。无监督方法需要完整的点云来构建非配对训练数据，而弱监督方法需要物体的多视图观测。现有的自监督方法由于其自监督信号的有限能力，经常产生不令人满意的预测结果。为了克服这些挑战，我们提出了一种新颖的自监督点云补全方法。我们基于单个部分点云的多视图增强设计了一套新的自监督信号。此外，为了增强模型的学习能力，我们首次将Mamba引入自监督点云补全任务，鼓励模型生成更高质量的点云。在合成和真实世界数据集上的实验表明，我们的方法实现了最先进的结果。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文解决的是点云补全问题，即从不完整的点云数据中重建完整的3D形状。这个问题在现实中很重要，因为现实世界中的点云数据常因传感器限制或遮挡而缺失部分区域，而完整的点云对机器人导航、自动驾驶、增强现实等领域至关重要。在研究中，解决这一问题可以克服现有方法对大量标注数据或完整参考数据的依赖，提高模型的泛化能力和实用性。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先分析了现有方法的局限性：监督方法依赖真实标签导致域差距问题，无监督方法需要完整点云，弱监督方法需要多视角观测，而现有自监督方法因自监督信号弱和模型架构不适合而效果不佳。基于这些问题，作者设计了两种主要创新：1)利用单部分点云的多视图增强创建新的自监督信号；2)首次将Mamba模型应用于此任务。作者借鉴了点云处理中的FPS采样、Hilbert曲线序列化、KNN局部划分等技术，以及Mamba的高效全局建模能力，但将这些技术以创新方式组合应用于自监督点云补全任务。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是通过单部分点云的多视图增强创建多样化的不完整性，提高模型的鲁棒性和泛化能力，同时利用Mamba模型增强特征提取。整体流程为：1)使用FPS选择关键点并用Hilbert曲线序列化；2)应用KNN划分为局部块并通过patch嵌入层生成tokens；3)使用8个Mamba块组成的编码器提取全局特征；4)通过多视图增强生成器从8个随机视角创建部分点云；5)使用包含三个全连接层的生成器生成完整点云；6)结合初始点云监督损失和多视图一致性损失进行训练。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)基于单部分点云多视图增强的新型自监督信号，引入不完整性多样性；2)首次将Mamba模型应用于自监督点云补全任务，设计专门的Mamba编码器；3)仅使用单部分点云无需任何先验信息的整体框架。相比之前的工作，本方法不依赖成对训练数据(监督方法)、不需要完整点云(无监督方法)、不需要多视角观测(弱监督方法)，且相比现有自监督方法使用了更强的自监督信号和更适合任务的Mamba架构，实验证明在合成和真实数据集上都取得了最先进结果。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '本文提出了一种基于单部分点云多视图增强的自监督点云补全方法，首次将Mamba模型应用于此任务，显著提高了点云补全的准确性和细节表现，在合成和真实世界数据集上都取得了最先进的结果。'}


### 论文摘要

Point cloud completion aims to reconstruct complete shapes from partial observations. Although current methods have achieved remarkable performance, they still have some limitations: Supervised methods heavily rely on ground truth, which limits their generalization to real-world datasets due to the synthetic-to-real domain gap. Unsupervised methods require complete point clouds to compose unpaired training data, and weakly-supervised methods need multi-view observations of the object. Existing self-supervised methods frequently produce unsatisfactory predictions due to the limited capabilities of their self-supervised signals. To overcome these challenges, we propose a novel self-supervised point cloud completion method. We design a set of novel self-supervised signals based on multi-view augmentations of the single partial point cloud. Additionally, to enhance the model's learning ability, we first incorporate Mamba into self-supervised point cloud completion task, encouraging the model to generate point clouds with better quality. Experiments on synthetic and real-world datasets demonstrate that our method achieves state-of-the-art results.

---

## 17. Large Material Gaussian Model for Relightable 3D Generation

**论文链接:** [http://arxiv.org/abs/2509.22112v1](http://arxiv.org/abs/2509.22112v1)

**作者:** Jingrui Ye, Lingting Zhu, Runze Zhang, Zeyu Hu, Yingda Yin, Lanjiong Li, Lequan Yu, Qingmin Liao

**发布时间:** 2025-09-26

### GPT解析

### 总结

本文提出了大型材质高斯模型(MGM)，解决了现有3D重建模型无法生成材质属性的问题，实现了具有物理基础渲染(PBR)材质的高质量3D内容生成。

### 背景

各行业对3D资产的需求不断增长，需要高效和自动化的3D内容创建方法。现有的3D高斯溅射技术虽然能实现高质量3D渲染，但无法生成资产的材质属性，这对不同光照环境中的真实感渲染至关重要。

### 目的

引入大型材质高斯模型(MGM)框架，生成具有基于物理渲染(PBR)材质(反照率、粗糙度和金属属性)的高质量3D内容，而非仅生成具有不受控光照烘焙的RGB纹理。

### 方法

首先微调基于输入深度和法线图条件化的多视图材质扩散模型；然后探索与2D高斯溅射对齐的材质表示方法，建模PBR材料的每个通道；最后利用重建的点云获取PBR属性，实现通过环境光照图的动态重新照明。

### 主要发现

实验证明，MGM方法产生的材料不仅比基线方法具有更大的视觉吸引力，而且增强了材质建模，能够实现实用的下游渲染应用。

### 结论

MGM成功解决了现有3D重建模型的材质生成缺陷，支持动态重新照明，为3D内容创建提供了实用解决方案。

### 翻译

各行业对3D资产不断增长的需求需要高效和自动化的3D内容创建方法。利用3D高斯溅射，最近的大型重建模型(LRMs)已经展示了通过集成多视图扩散进行生成和可扩展变换器进行重建，高效实现高质量3D渲染的能力。然而，现有模型无法生成资产的材质属性，这对于在不同光照环境中的真实感渲染至关重要。在本文中，我们引入了大型材质高斯模型(MGM)，这是一个新框架，旨在生成具有基于物理渲染(PBR)材质(即反照率、粗糙度和金属属性)的高质量3D内容，而不仅仅是生成具有不受控光照烘焙的RGB纹理。具体而言，我们首先微调了一个基于输入深度和法线图条件化的新多视图材质扩散模型。利用生成的多视图PBR图像，我们探索了一种与2D高斯溅射对齐的材质表示，同时建模PBR材料的每个通道。然后，重建的点云可以被渲染以获取PBR属性，通过应用各种环境光照图实现动态重新照明。大量实验证明，我们方法产生的材料不仅比基线方法具有更大的视觉吸引力，而且增强了材质建模，从而能够实现实用的下游渲染应用。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决现有3D生成方法无法产生物体材料属性的问题，包括反照率、粗糙度和金属度等PBR(基于物理渲染)材料特性。这个问题很重要，因为缺乏材料属性限制了3D资产在不同光照环境下的真实感渲染能力，使其无法适应动态光照变化，严重影响了3D内容在游戏、电影、虚拟现实等领域的应用效果和实用性。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先分析了现有3D生成方法的局限性：优化方法如DreamFusion存在计算成本高和视图不一致问题；基于transformer的方法如LRM无法重建精细细节；而基于高斯溅射的方法虽然高效但无法控制光照和材料属性。作者借鉴了多项现有技术：MVDream等多视图扩散模型用于文本到多视图图像生成；2D高斯溅射作为3D表示；ControlNet控制深度和法线图提供几何先验；LaRa的体积解码器结构。主要创新在于首次提出生成具有PBR材料属性的高斯表示，并设计了专门的多视图PBR扩散模型和重建流程。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是生成具有物理渲染材料属性(反照率、粗糙度、金属度)的高斯表示，而不仅仅是RGB纹理，并支持动态重照明。整体流程包括：1)多视图PBR扩散：训练两个子模型分别生成反照率和粗糙度/金属度图像，使用深度和法线图作为条件；2)材料高斯重建：设计统一的高斯表示包含几何和材料属性，使用体积解码器预测3D高斯体积，通过重建损失、几何正则化和两阶段训练策略优化；3)重照明：使用Cook-Torrance微面模型将生成的PBR属性与环境光照图结合，实现不同光照条件下的真实感渲染。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)材料高斯表示：首次提出包含PBR材料属性的高斯表示；2)多视图PBR扩散模型：专门设计生成无光照影响的材料图像；3)几何先验注入：在生成和重建阶段都使用深度和法线图确保一致性；4)两阶段训练策略：先训练反照率再训练所有材料组件。相比之前工作，不同之处在于：与优化方法相比更快速且避免了视图不一致；与LRM方法相比支持高分辨率渲染和材料生成；与其他高斯溅射方法相比能生成PBR材料属性；与其他PBR方法相比直接生成3D资产且速度更快。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '这篇论文首次提出了一种能够生成具有物理渲染材料属性的高斯表示方法，实现了从文本提示到可重照明的3D资产的高效生成，显著提升了3D内容创建的灵活性和真实感。'}


### 论文摘要

The increasing demand for 3D assets across various industries necessitates efficient and automated methods for 3D content creation. Leveraging 3D Gaussian Splatting, recent large reconstruction models (LRMs) have demonstrated the ability to efficiently achieve high-quality 3D rendering by integrating multiview diffusion for generation and scalable transformers for reconstruction. However, existing models fail to produce the material properties of assets, which is crucial for realistic rendering in diverse lighting environments. In this paper, we introduce the Large Material Gaussian Model (MGM), a novel framework designed to generate high-quality 3D content with Physically Based Rendering (PBR) materials, ie, albedo, roughness, and metallic properties, rather than merely producing RGB textures with uncontrolled light baking. Specifically, we first fine-tune a new multiview material diffusion model conditioned on input depth and normal maps. Utilizing the generated multiview PBR images, we explore a Gaussian material representation that not only aligns with 2D Gaussian Splatting but also models each channel of the PBR materials. The reconstructed point clouds can then be rendered to acquire PBR attributes, enabling dynamic relighting by applying various ambient light maps. Extensive experiments demonstrate that the materials produced by our method not only exhibit greater visual appeal compared to baseline methods but also enhance material modeling, thereby enabling practical downstream rendering applications.

---

## 18. An Adaptive ICP LiDAR Odometry Based on Reliable Initial Pose

**论文链接:** [http://arxiv.org/abs/2509.22058v1](http://arxiv.org/abs/2509.22058v1)

**作者:** Qifeng Wang, Weigang Li, Lei Nie, Xin Xu, Wenping Liu, Zhe Xu

**发布时间:** 2025-09-26

**DOI:** 10.1109/TIM.2025.3571148

### GPT解析

### 总结

该研究提出了一种基于可靠初始位姿的自适应ICP LiDAR里程计方法，通过分布式粗配准获得可靠初始位姿，并结合动态阈值调整机制，显著提高了点云配准精度，在KITTI数据集上表现优于现有方法。

### 背景

LiDAR里程计是移动机器人自主导航和定位的关键技术，在自动驾驶领域广泛应用。基于迭代最近点（ICP）的方法因能有效且准确地进行点云配准而成为LiDAR里程计的核心技术。

### 目的

解决现有基于ICP的方法不考虑初始位姿可靠性以及缺乏自适应机制处理复杂动态环境的问题，提高LiDAR里程计的精度。

### 方法

1) 使用基于密度滤波的分布式粗配准获得初始位姿估计；2) 通过与运动预测位姿比较选择可靠初始位姿，减少源点云和目标点云间的初始误差；3) 结合当前和历史误差动态调整自适应阈值；4) 基于可靠初始位姿和自适应阈值执行点对平面自适应ICP配准。

### 主要发现

所提出的方法在KITTI数据集上的实验表明，它优于现有方法，能够显著提高LiDAR里程计的精度，特别是在复杂动态环境中表现更佳。

### 结论

通过可靠的初始位姿选择和自适应阈值调整机制，该方法有效解决了现有ICP方法在动态环境中的局限性，实现了高精度的点云配准。

### 翻译

作为移动机器人自主导航和定位的关键技术，激光雷达里程计在自动驾驶应用中得到广泛应用。基于迭代最近点的方法因其在点云配准方面的高效性和准确性已成为LiDAR里程计的核心技术。然而，一些现有的基于ICP的方法没有考虑初始位姿的可靠性，可能导致方法收敛到局部最优。此外，缺乏自适应机制阻碍了对复杂动态环境的有效处理，导致配准精度显著下降。为解决这些问题，本文提出了一种基于可靠初始位姿的自适应ICP LiDAR里程计方法。首先，采用基于密度滤波的分布式粗配准来获得初始位姿估计。通过与运动预测位姿比较选择可靠的初始位姿，减少源点云和目标点云之间的初始误差。随后，通过结合当前和历史误差，动态调整自适应阈值以适应动态环境的实时变化。最后，基于可靠的初始位姿和自适应阈值，执行从当前帧到局部地图的点对平面自适应ICP配准，实现源点云和目标点云的高精度对齐。在公开KITTI数据集上的大量实验表明，所提出的方法优于现有方法，并显著提高了LiDAR里程计的精度。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决现有基于ICP的激光雷达里程计方法中初始位姿不可靠和缺乏自适应机制的问题。这个问题很重要，因为激光雷达里程计是自主导航和定位的关键技术，广泛应用于自动驾驶领域。初始位姿的可靠性直接影响配准精度，而动态环境下的精确定位对自动驾驶和机器人系统至关重要。解决这些问题可以提高定位精度和鲁棒性，特别是在复杂动态场景中。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先分析了现有基于ICP的激光雷达里程计方法的局限性，包括初始位姿不可靠和缺乏自适应机制。作者借鉴了现有的点云配准技术，如传统的ICP算法和GICP，并参考了特征点云处理方法，如密度过滤和协方差矩阵计算。作者引入了自适应机制来处理动态环境变化，并设计了一个四步流程：分布式粗配准、初始位姿确定、自适应阈值和自适应ICP配准。这些设计基于对现有方法的深入理解和改进。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是通过密度过滤的分布式粗配准获取初始位姿估计，通过比较初始位姿与运动预测位姿选择可靠的初始位姿，结合当前和历史误差动态调整自适应阈值，基于可靠的初始位姿和自适应阈值执行点到平面自适应ICP配准。整体流程包括：1)分布式粗配准：计算点密度、过滤低密度点、计算协方差矩阵、查找最近点对并优化变换矩阵；2)初始位姿确定：生成预测位姿、比较与选择可靠位姿；3)自适应阈值：计算加速度变化率、模型偏差矩阵和加权误差；4)自适应ICP配准：应用位姿变换、查找最近邻、计算残差和权重、优化变换矩阵。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)可靠初始位姿获取方法：使用密度过滤的分布式粗配准和运动预测比较；2)自适应帧到局部地图ICP配准方法：结合当前和历史误差动态调整参数；3)自适应阈值机制：根据运动状态和点云环境变化动态调整阈值。相比之前的工作，不同之处在于传统ICP方法没有充分考虑初始位姿可靠性，容易陷入局部最优，且缺乏自适应机制难以处理动态环境。本方法通过可靠初始位姿和自适应机制显著提高了在复杂动态环境中的鲁棒性和精度。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '本文提出了一种基于可靠初始位姿的自适应ICP激光雷达里程计方法，通过可靠的初始位姿估计和动态自适应机制，显著提高了在复杂动态环境中的定位精度和鲁棒性。'}


### 论文摘要

As a key technology for autonomous navigation and positioning in mobile robots, light detection and ranging (LiDAR) odometry is widely used in autonomous driving applications. The Iterative Closest Point (ICP)-based methods have become the core technique in LiDAR odometry due to their efficient and accurate point cloud registration capability. However, some existing ICP-based methods do not consider the reliability of the initial pose, which may cause the method to converge to a local optimum. Furthermore, the absence of an adaptive mechanism hinders the effective handling of complex dynamic environments, resulting in a significant degradation of registration accuracy. To address these issues, this paper proposes an adaptive ICP-based LiDAR odometry method that relies on a reliable initial pose. First, distributed coarse registration based on density filtering is employed to obtain the initial pose estimation. The reliable initial pose is then selected by comparing it with the motion prediction pose, reducing the initial error between the source and target point clouds. Subsequently, by combining the current and historical errors, the adaptive threshold is dynamically adjusted to accommodate the real-time changes in the dynamic environment. Finally, based on the reliable initial pose and the adaptive threshold, point-to-plane adaptive ICP registration is performed from the current frame to the local map, achieving high-precision alignment of the source and target point clouds. Extensive experiments on the public KITTI dataset demonstrate that the proposed method outperforms existing approaches and significantly enhances the accuracy of LiDAR odometry.

---

## 19. Convexity-Driven Projection for Point Cloud Dimensionality Reduction

**论文链接:** [http://arxiv.org/abs/2509.22043v1](http://arxiv.org/abs/2509.22043v1)

**作者:** Suman Sanyal

**发布时间:** 2025-09-26

### GPT解析

### 总结

提出了一种名为凸性驱动投影(CDP)的新型点云降维方法，通过构建k-NN图并识别可接受点对来保持局部非凸性结构。

### 背景

点云降维领域需要处理局部非凸性特征，传统方法可能无法有效保持这种特性。

### 目的

开发一种能够保持点云中由绕行引起的局部非凸性的降维方法。

### 方法

构建k-NN图，识别欧几里得到最短路径比率低于阈值的可接受点对，聚合这些点对的归一化方向形成半正定非凸性结构矩阵，使用该结构矩阵的前k个特征向量进行投影。

### 主要发现

提供了两种可验证保证：点对的后验证书，用于限制每个可接受点对的投影后失真；以及平均情况谱边界，将捕获的方向能量的期望与结构矩阵的谱联系起来，得出典型失真的分位数陈述。

### 结论

评估协议报告了固定点和重新选择点的绕行误差以及证书分位数，使实践者能够检查其数据上的保证。

### 翻译

我们提出了一种凸性驱动投影(CDP)，这是一种无边界的线性方法，用于点云的降维，旨在保持由绕行引起的局部非凸性。CDP构建一个k-NN图，识别欧几里得到最短路径比率低于阈值的可接受点对，并聚合它们的归一化方向以形成一个半正定非凸性结构矩阵。投影使用该结构矩阵的前k个特征向量。我们提供了两种可验证的保证：一种点对的后验证书，用于限制每个可接受点对的投影后失真；以及一种平均情况谱边界，它将捕获的方向能量的期望与结构矩阵的谱联系起来，从而得出典型失真的分位数陈述。我们的评估协议报告了固定点和重新选择点的绕行误差以及证书分位数，使实践者能够检查其数据上的保证。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': "这篇论文主要解决点云在高维空间中的降维问题，特别是如何保留由障碍物或曲率引起的'绕行(detour)'导致的局部非凸性结构。在3D建模、机器人和可视化等领域，点云数据中两点间的图最短路径可能远大于它们的欧几里得距离，这种绕行几何结构对于路径规划、形状分析等任务至关重要，而标准降维方法如PCA、t-SNE等无法有效保留这种结构。", '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': "作者分析了现有降维方法的局限性：PCA最大化解释方差但可能忽略测地线结构；非线性方法如t-SNE和UMAP缺乏对绕行几何的明确控制；基于图的线性投影方法不能突出显示非凸性方向。作者借鉴了基于图的降维方法思路，但针对绕行几何进行了专门设计。通过构建k-NN图，识别具有低欧几里得到最短路径比率的'可接受对'，并利用这些对的方向信息构建非凸性结构矩阵，从而专注于保留非凸结构而非保持邻近点接近。", '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是通过识别和保留那些显示显著绕行(即非凸性)的方向对来降维，同时最小化绕行结构的扭曲。整体流程：1)标准化坐标；2)构建互惠k-NN图；3)计算所有点对的最短路径距离；4)形成满足rij = 欧几里得距离/最短路径距离 ≤ 阈值τ的可接受对集合；5)构建非凸性结构矩阵，聚合这些对的归一化方向；6)计算该矩阵的前k个特征向量；7)将原始点投影到这些特征向量张成的子空间；8)为投影图分配边权重。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点：1)提出边界无关的线性降维方法，专注于保留绕行引起的局部非凸性；2)构建正半定非凸性结构矩阵；3)提供成对后验证书和平均情况谱界限两种可验证保证；4)提出评估协议报告固定和重新选择的绕行错误。相比之前工作，CDP不最大化方差(PCA)，不优化概率邻域(t-SNE/UMAP)，也不最小化平滑性(LPP/NPP/OLPP)，而是突出显示非凸性方向，并提供理论保证和评估协议。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '这篇论文提出了凸性驱动投影(CDP)方法，一种线性降维技术，通过构建非凸性结构矩阵并利用其特征向量进行投影，有效保留了点云中的绕行诱导局部非凸结构，同时提供了可验证的理论保证和评估协议。'}


### 论文摘要

We propose Convexity-Driven Projection (CDP), a boundary-free linear method for dimensionality reduction of point clouds that targets preserving detour-induced local non-convexity. CDP builds a $k$-NN graph, identifies admissible pairs whose Euclidean-to-shortest-path ratios are below a threshold, and aggregates their normalized directions to form a positive semidefinite non-convexity structure matrix. The projection uses the top-$k$ eigenvectors of the structure matrix. We give two verifiable guarantees. A pairwise a-posteriori certificate that bounds the post-projection distortion for each admissible pair, and an average-case spectral bound that links expected captured direction energy to the spectrum of the structure matrix, yielding quantile statements for typical distortion. Our evaluation protocol reports fixed- and reselected-pairs detour errors and certificate quantiles, enabling practitioners to check guarantees on their data.

---

## 20. TDEdit: A Unified Diffusion Framework for Text-Drag Guided Image Manipulation

**论文链接:** [http://arxiv.org/abs/2509.21905v1](http://arxiv.org/abs/2509.21905v1)

**作者:** Qihang Wang, Yaxiong Wang, Lechao Cheng, Zhun Zhong

**发布时间:** 2025-09-26

### GPT解析

### 总结

本文提出了一种基于扩散的统一框架，实现了文本和拖拽交互共同控制下的图像编辑，结合了文本驱动和拖拽驱动两种方法的优点。

### 背景

文本驱动和拖拽驱动的图像编辑方法最近取得了显著进展，但存在互补限制：文本驱动方法擅长纹理处理但缺乏精确空间控制，拖拽驱动方法主要修改形状结构而缺乏细粒度纹理指导。

### 目的

解决现有图像编辑方法的局限性，实现结合文本和拖拽优势的高保真联合图像编辑。

### 方法

提出了两个关键创新：(1)点云确定性拖拽，通过3D特征映射增强潜在空间布局控制；(2)拖拽-文本引导去噪，动态平衡拖拽和文本条件的影响。支持仅文本、仅拖拽或组合条件的灵活编辑模式。

### 主要发现

通过大量定量和定性实验证明，该方法实现了高保真的联合编辑，性能匹配或超过了专门的仅文本或仅拖拽方法，建立了可控图像操作的通用可推广解决方案。

### 结论

代码将公开可用，以重现本文呈现的所有结果。

### 翻译

本文探索了在文本和拖拽交互共同控制下的图像编辑。虽然最近在文本驱动和拖拽驱动编辑方面取得了显著进展，但它们存在互补的限制：文本驱动方法擅长纹理处理但缺乏精确的空间控制，而拖拽驱动方法主要修改形状和结构，没有细粒度的纹理指导。为解决这些限制，我们提出了一个统一的基于扩散的框架，用于联合拖拽-文本图像编辑，整合了两种范式的优势。我们的框架引入了两个关键创新：(1)点云确定性拖拽，通过3D特征映射增强潜在空间布局控制；(2)拖拽-文本引导去噪，在去噪过程中动态平衡拖拽和文本条件的影响。值得注意的是，我们的模型支持灵活的编辑模式 - 可以在仅文本、仅拖拽或组合条件下运行 - 同时在每种设置中保持强性能。大量的定量和定性实验证明，我们的方法不仅实现了高保真的联合编辑，而且匹配或超过了专门的仅文本或仅拖拽方法的性能，为可控图像操作建立了一个多功能且可推广的解决方案。代码将公开可用，以重现本文呈现的所有结果。


### 论文摘要

This paper explores image editing under the joint control of text and drag interactions. While recent advances in text-driven and drag-driven editing have achieved remarkable progress, they suffer from complementary limitations: text-driven methods excel in texture manipulation but lack precise spatial control, whereas drag-driven approaches primarily modify shape and structure without fine-grained texture guidance. To address these limitations, we propose a unified diffusion-based framework for joint drag-text image editing, integrating the strengths of both paradigms. Our framework introduces two key innovations: (1) Point-Cloud Deterministic Drag, which enhances latent-space layout control through 3D feature mapping, and (2) Drag-Text Guided Denoising, dynamically balancing the influence of drag and text conditions during denoising. Notably, our model supports flexible editing modes - operating with text-only, drag-only, or combined conditions - while maintaining strong performance in each setting. Extensive quantitative and qualitative experiments demonstrate that our method not only achieves high-fidelity joint editing but also matches or surpasses the performance of specialized text-only or drag-only approaches, establishing a versatile and generalizable solution for controllable image manipulation. Code will be made publicly available to reproduce all results presented in this work.

---

## 21. Generating Stable Placements via Physics-guided Diffusion Models

**论文链接:** [http://arxiv.org/abs/2509.21664v1](http://arxiv.org/abs/2509.21664v1)

**作者:** Philippe Nadeau, Miguel Rogel, Ivan Bilić, Ivan Petrović, Jonathan Kelly

**发布时间:** 2025-09-25

**备注:** Submitted to the IEEE International Conference on Robotics and  Automation 2026, Vienna, Austria, June 1-5, 2026

### GPT解析

### 总结

本研究提出了一种将稳定性直接整合到扩散模型采样过程中的新方法，用于解决多物体场景中机器人放置物体的稳定性挑战。通过训练扩散模型生成稳定的放置，并结合几何感知先验和稳定性感知损失，实现了无需额外训练的高效稳定物体放置。

### 背景

在多物体场景中稳定放置物体是机器人操作的基本挑战，放置必须满足无穿透、精确表面接触和力平衡等条件。现有方法依赖运行仿真引擎或基于启发式、外观的评估来评估稳定性，这些方法存在效率或准确性问题。

### 目的

开发一种能够直接生成稳定放置的方法，无需依赖仿真引擎或启发式评估，提高放置稳定性的同时减少计算时间。

### 方法

作者查询离线基于采样的规划器收集多模态放置标签，训练扩散模型生成稳定的放置。该模型基于场景和物体点云条件化，作为几何感知先验。利用基于分数的生成模型的组合特性，将学习到的先验与稳定性感知损失结合，增加从高稳定性区域采样的可能性。此方法无需额外的再训练或微调，可直接应用于现成模型。

### 主要发现

在四个基准场景上的评估表明，物理引导的模型相比最先进的几何方法，放置对强力扰动的鲁棒性提高了56%，同时运行时间减少了47%。

### 结论

将稳定性直接整合到扩散模型采样过程中的方法，能够有效提高物体放置的稳定性和效率，且无需额外的训练步骤，可直接应用于现有模型，为机器人操作中的物体放置问题提供了新的解决方案。

### 翻译

在多物体场景中稳定放置物体是机器人操作中的一个基本挑战，因为放置必须无穿透、建立精确表面接触，并达到力平衡。为了评估稳定性，现有方法依赖运行仿真引擎或采用启发式、基于外观的评估。相比之下，我们的方法将稳定性直接整合到扩散模型的采样过程中。为此，我们查询离线基于采样的规划器以收集多模态放置标签，并训练扩散模型来生成稳定的放置。扩散模型基于场景和物体点云条件化，并作为几何感知先验。我们利用基于分数的生成模型的组合特性，将这种学习到的先验与稳定性感知损失相结合，从而增加从高稳定性区域采样的可能性。重要的是，这种策略不需要额外的再训练或微调，可以直接应用于现成模型。我们在四个稳定性可以准确计算的基准场景上评估了我们的方法。我们的物理引导模型相比最先进的几何方法，实现了对强力扰动56%更强的鲁棒性，同时将运行时间减少了47%。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决在多物体场景中稳定放置物体的问题，放置必须满足无穿透、建立精确表面接触和实现力平衡三个条件。这个问题很重要，因为它是机器人操作中的基本挑战，在建筑施工、场景重组和密集包装等高级任务中至关重要。现有方法要么依赖耗时的模拟引擎，要么依赖基于外观的启发式评估，效率低下且准确性有限。只有很少的工作空间姿态会导致有效放置，使得随机采样或搜索方法效率不高。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先分析了稳定放置规划需要满足的严格几何和物理约束，将其分解为三个需求：几何感知的放置算法、场景平衡推理和稳定性验证器。他们评估了现有方法的局限性，然后设计了一种基于扩散模型的物理引导方法。作者借鉴了扩散模型在条件生成方面的能力，利用了装配鲁棒性概念，并采用采样规划器生成训练数据。他们使用U-Net架构进行点云编码，并通过组合学习到的几何先验与稳定性感知损失来实现物理引导。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是将物理稳定性直接整合到扩散模型的采样过程中，利用扩散模型生成稳定的放置姿态，并通过鲁棒性引导指导采样过程朝向高稳定性区域。整体流程包括：1)使用采样规划器生成训练数据集；2)处理观测数据为点云表示；3)训练U-Net模型预测放置姿态；4)在推理过程中应用鲁棒性引导来生成稳定放置；5)评估放置的鲁棒性、非穿透性和稳定性。整个方法不需要额外训练或微调，可直接应用于现成模型。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)首次将物理稳定性直接整合到扩散模型采样过程中；2)提出鲁棒性引导方案指导模型朝向高稳定性区域；3)利用基于分数的生成模型组合性质，结合几何先验与稳定性损失；4)引导方案独立于模型训练，可直接应用于现成模型；5)将几何感知和物理推理统一在一个框架中。相比之前工作，该方法不依赖模拟引擎（减少47%计算时间），不依赖外观评估或惯性参数假设，在未知场景上表现出更好的泛化能力，生成的放置具有更高的鲁棒性（平均高56%）。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '本文提出了一种物理引导的扩散模型方法，通过将稳定性直接整合到采样过程中，实现了在未知场景中快速生成具有更高鲁棒性的物体放置方案，相比现有方法显著提高了稳定性和效率。'}


### 论文摘要

Stably placing an object in a multi-object scene is a fundamental challenge in robotic manipulation, as placements must be penetration-free, establish precise surface contact, and result in a force equilibrium. To assess stability, existing methods rely on running a simulation engine or resort to heuristic, appearance-based assessments. In contrast, our approach integrates stability directly into the sampling process of a diffusion model. To this end, we query an offline sampling-based planner to gather multi-modal placement labels and train a diffusion model to generate stable placements. The diffusion model is conditioned on scene and object point clouds, and serves as a geometry-aware prior. We leverage the compositional nature of score-based generative models to combine this learned prior with a stability-aware loss, thereby increasing the likelihood of sampling from regions of high stability. Importantly, this strategy requires no additional re-training or fine-tuning, and can be directly applied to off-the-shelf models. We evaluate our method on four benchmark scenes where stability can be accurately computed. Our physics-guided models achieve placements that are 56% more robust to forceful perturbations while reducing runtime by 47% compared to a state-of-the-art geometric method.

---

## 22. SeamCrafter: Enhancing Mesh Seam Generation for Artist UV Unwrapping via Reinforcement Learning

**论文链接:** [http://arxiv.org/abs/2509.20725v2](http://arxiv.org/abs/2509.20725v2)

**作者:** Duoteng Xu, Yuguang Chen, Jing Li, Xinhai Liu, Xueqi Ma, Zhuo Chen, Dongyu Zhang, Chunchao Guo

**发布时间:** 2025-09-25

### GPT解析

### 总结

本文提出了一种名为SeamCrafter的自回归GPT风格接缝生成器，用于解决3D表面UV参数化和纹理映射中的接缝放置问题。该方法通过双分支点云编码器和直接偏好优化技术，显著降低了接缝的UV变形和碎片化程度，同时保持拓扑一致性和视觉保真度。

### 背景

网格接缝在3D表面的UV参数化和纹理映射中起着关键作用。不良的接缝放置会导致严重的UV变形或过度碎片化，从而阻碍纹理合成和艺术家工作流程。

### 目的

解决现有接缝生成方法中高变形和碎片化之间的权衡问题，开发一种能够同时降低变形和碎片化的接缝生成方法。

### 方法

SeamCrafter是一种基于点云输入的自回归GPT风格接缝生成器。它采用双分支点云编码器在预训练过程中解耦并捕获互补的拓扑和几何线索。此外，研究人员使用基于新型接缝评估框架的偏好数据集，通过直接偏好优化(DPO)微调模型，该框架主要通过UV变形和碎片化评估接缝并提供成对偏好标签。

### 主要发现

SeamCrafter产生的接缝比先前方法的变形和碎片化程度低得多，同时保持拓扑一致性和视觉保真度。

### 结论

SeamCrafter通过结合自回归GPT架构、双分支点云编码器和直接偏好优化技术，有效解决了3D表面UV参数化和纹理映射中的接缝放置问题，显著提高了接缝质量。

### 翻译

网格接缝在3D表面的UV参数化和纹理映射的分区中起着关键作用。放置不当的接缝通常会导致严重的UV变形或过度碎片化，从而阻碍纹理合成并干扰艺术家工作流程。现有方法经常以一种失败模式换取另一种失败模式——要么产生高变形，要么产生许多分散的碎片。为此，我们引入了SeamCrafter，一个基于点云输入的自回归GPT风格接缝生成器。SeamCrafter采用双分支点云编码器，在预训练过程中解耦并捕获互补的拓扑和几何线索。为了进一步提高接缝质量，我们使用基于新型接缝评估框架的偏好数据集，通过直接偏好优化(DPO)对模型进行微调。该框架主要通过UV变形和碎片化评估接缝，并提供成对偏好标签来指导优化。大量实验表明，SeamCrafter产生的接缝比先前方法的变形和碎片化程度低得多，同时保持拓扑一致性和视觉保真度。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决3D模型UV展开过程中的接缝生成问题。不当的接缝放置会导致严重的UV变形或过度碎片化，影响纹理映射质量并阻碍艺术家工作流程。这个问题在3D建模、游戏开发、电影特效等领域非常重要，因为高质量的UV展开能确保纹理正确映射到3D模型上，避免拉伸、压缩和可见不连续性，提高艺术家工作效率和最终作品质量。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先分析了现有方法的局限性：传统方法（如区域增长）产生碎片化UV图谱，对参数敏感；基于学习的方法（如SeamGPT）过度依赖拓扑线索而缺乏几何感知。作者借鉴了SeamGPT的自回归方法、VecSet点云编码器和直接偏好优化（DPO）等技术，但通过双分支编码器同时捕获拓扑和几何信息，并设计了专门的接缝评估框架来优化变形与碎片化之间的权衡。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是同时利用网格的拓扑和几何信息生成高质量接缝，并通过直接偏好优化将模型输出与人类偏好对齐。整体流程分为三阶段：1）预训练阶段，使用双分支点云编码器分别从顶点边和表面采样点捕获拓扑和几何信息，用沙漏transformer解码器生成接缝；2）后训练阶段，构建接缝评估系统生成偏好数据集，用DPO微调模型；3）推理阶段，预测接缝端点坐标，映射到网格表面并标记接缝路径，完成UV展开。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1）双分支点云编码器，分别捕获拓扑结构和几何细节；2）接缝评估框架，提供基于UV变形和碎片化的偏好信号；3）直接偏好优化应用，将模型与人类判断对齐。相比之前工作，与传统方法相比避免了过度碎片化；与其他学习方法相比更好地保持了网格结构约束；与SeamGPT相比同时利用拓扑和几何信息，生成更连贯、结构合理的接缝布局。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'SeamCrafter通过结合双分支几何-拓扑编码器和直接偏好优化技术，显著提高了3D模型UV展开的接缝生成质量，同时降低了UV变形和碎片化，为艺术家和3D内容创作者提供了更高效的工作流程。'}


### 论文摘要

Mesh seams play a pivotal role in partitioning 3D surfaces for UV parametrization and texture mapping. Poorly placed seams often result in severe UV distortion or excessive fragmentation, thereby hindering texture synthesis and disrupting artist workflows. Existing methods frequently trade one failure mode for another-producing either high distortion or many scattered islands. To address this, we introduce SeamCrafter, an autoregressive GPT-style seam generator conditioned on point cloud inputs. SeamCrafter employs a dual-branch point-cloud encoder that disentangles and captures complementary topological and geometric cues during pretraining. To further enhance seam quality, we fine-tune the model using Direct Preference Optimization (DPO) on a preference dataset derived from a novel seam-evaluation framework. This framework assesses seams primarily by UV distortion and fragmentation, and provides pairwise preference labels to guide optimization. Extensive experiments demonstrate that SeamCrafter produces seams with substantially lower distortion and fragmentation than prior approaches, while preserving topological consistency and visual fidelity.

---

## 23. TUN3D: Towards Real-World Scene Understanding from Unposed Images

**论文链接:** [http://arxiv.org/abs/2509.21388v1](http://arxiv.org/abs/2509.21388v1)

**作者:** Anton Konushin, Nikita Drozdov, Bulat Gabdullin, Alexey Zakharov, Anna Vorontsova, Danila Rukhovich, Maksim Kolodiazhnyi

**发布时间:** 2025-09-23

### GPT解析

### 总结

本研究提出了一种名为TUN3D的新方法，首次实现了仅从多视图图像输入进行室内场景的联合布局估计和3D物体检测，无需真实相机位姿或深度监督。该方法在多个基准测试中取得了最先进的性能，显著提升了室内场景理解能力。

### 背景

布局估计和3D物体检测是室内场景理解的两大基本任务。现有方法通常依赖于点云输入，但大多数消费者相机缺乏深度传感器，视觉数据更为常见，这限制了现有方法的应用范围。

### 目的

开发一种能够仅从多视图图像输入进行联合布局估计和3D物体检测的方法，不需要真实的相机位姿或深度监督，从而解决点云输入的限制问题。

### 方法

TUN3D基于轻量级稀疏卷积骨干网络，采用两个专用头部分别处理3D物体检测和布局估计任务。布局估计部分利用了一种新颖且有效的参数化墙体表示方法。

### 主要发现

TUN3D在三个具有挑战性的场景理解基准测试中取得了最先进的性能：(1)使用真实点云，(2)使用有位姿的图像，(3)使用无位姿的图像。与专门的3D物体检测方法性能相当的同时，显著提升了布局估计能力。

### 结论

TUN3D为整体室内场景理解树立了新基准，证明了仅从视觉输入进行联合布局估计和3D物体检测的可行性，为缺乏深度传感器的消费级相机应用提供了有效解决方案。

### 翻译

布局估计和3D物体检测是室内场景理解的两大基本任务。当结合使用时，它们能够创建一个紧凑且语义丰富的场景空间表示。现有方法通常依赖于点云输入，这构成了一个主要限制，因为大多数消费者相机缺乏深度传感器，而纯视觉数据则更为常见。我们通过TUN3D解决了这一问题，这是首个在真实扫描中处理联合布局估计和3D物体检测的方法，以多视图图像为输入，且不需要真实的相机位姿或深度监督。我们的方法基于轻量级稀疏卷积骨干网络，并采用两个专用头部分别进行3D物体检测和布局估计，利用了一种新颖且有效的参数化墙体表示。大量实验表明，TUN3D在三个具有挑战性的场景理解基准测试中取得了最先进的性能：(i)使用真实点云，(ii)使用有位姿的图像，以及(iii)使用无位姿的图像。虽然性能与专门的3D物体检测方法相当，但TUN3D显著改进了布局估计，为整体室内场景理解树立了新基准。代码可在https://github.com/col14m/tun3d获取。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决室内场景理解中的联合布局估计和3D物体检测问题，特别是减少对点云数据的依赖。这个问题很重要，因为大多数消费级相机没有深度传感器，而现有方法通常需要点云输入，限制了在普通设备上的应用。室内场景理解在机器人、AR/VR、室内设计等领域有广泛应用，而紧凑的空间表示比密集3D重建更适合在设备上运行。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先分析了现有方法的局限性：点云输入要求高、联合模型要么速度慢要么精度低。他们决定构建在实时3D物体检测模型之上，并扩展输入模态从点云到带姿态图像再到无姿态图像。借鉴了TR3D的稀疏卷积骨干网络、PQ-Transformer的联合检测思路，以及DUSt3R的图像到点云转换方法。作者还提出了一种新的墙壁参数化方法，将3D问题简化为2D表示以提高效率。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是使用轻量级稀疏卷积网络处理多种输入模态（点云、带姿态图像、无姿态图像），通过新的墙壁参数化方法实现高效的联合布局估计和3D物体检测。整体流程：1)输入处理（点云直接体素化，图像通过DUSt3R转换为点云）；2)网络结构（稀疏卷积骨干+颈部+检测头+布局头）；3)墙壁参数化（2D偏移+高度）；4)训练（多组件损失函数）。这种方法无需深度或相机姿态监督，却能实现高性能场景理解。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点：1)首个从多视角图像（带或不带姿态）进行联合布局估计和3D物体检测的方法；2)新的墙壁参数化方法（2D偏移+高度，仅需5个参数）；3)高效的稀疏卷积架构；4)灵活处理三种输入模态。相比之前工作，TUN3D不需要深度传感器或相机姿态，比现有方法快4-160倍，在布局估计上显著优于现有方法（比PQ-Transformer高23.6 F1分数），且能在真实场景上工作，而不仅仅是合成数据。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'TUN3D首次实现了从多视角图像（带或不带相机姿态）进行无需深度监督的联合室内布局估计和3D物体检测，显著提高了场景理解的效率和准确性，为在普通设备上运行的空间理解应用开辟了新可能。'}


### 论文摘要

Layout estimation and 3D object detection are two fundamental tasks in indoor scene understanding. When combined, they enable the creation of a compact yet semantically rich spatial representation of a scene. Existing approaches typically rely on point cloud input, which poses a major limitation since most consumer cameras lack depth sensors and visual-only data remains far more common. We address this issue with TUN3D, the first method that tackles joint layout estimation and 3D object detection in real scans, given multi-view images as input, and does not require ground-truth camera poses or depth supervision. Our approach builds on a lightweight sparse-convolutional backbone and employs two dedicated heads: one for 3D object detection and one for layout estimation, leveraging a novel and effective parametric wall representation. Extensive experiments show that TUN3D achieves state-of-the-art performance across three challenging scene understanding benchmarks: (i) using ground-truth point clouds, (ii) using posed images, and (iii) using unposed images. While performing on par with specialized 3D object detection methods, TUN3D significantly advances layout estimation, setting a new benchmark in holistic indoor scene understanding. Code is available at https://github.com/col14m/tun3d .

---

## 24. Rate-Distortion Optimized Communication for Collaborative Perception

**论文链接:** [http://arxiv.org/abs/2509.21994v1](http://arxiv.org/abs/2509.21994v1)

**作者:** Genjia Liu, Anning Hu, Yue Hu, Wenjun Zhang, Siheng Chen

**发布时间:** 2025-09-26

### GPT解析

### 总结

该研究提出了一种名为RDcomm的通信高效协同感知框架，通过信息理论指导，优化多智能体间的信息共享，在保持高准确度的同时大幅减少通信量。

### 背景

协同感知强调通过多智能体共享视觉信息来增强环境理解，但受限于带宽资源。先前工作探索了任务性能与通信量之间的权衡，但缺乏理论基础。

### 目的

填补协同感知领域理论基础的空白，提出一种专门用于分析目标导向多智能体系统中性能-通信权衡的实用速率失真理论。

### 方法

提出RDcomm框架，包含两个关键创新：任务熵离散编码，为特征分配任务相关码字长度以最大化实用信息效率；互信息驱动的消息选择，利用互信息神经估计实现无冗余传输。

### 主要发现

在DAIR-V2X和OPV2V数据集上的3D目标检测和BEV分割实验表明，RDcomm实现了最先进的准确度，同时将通信量减少了高达108倍。

### 结论

RDcomm框架通过理论指导的通信策略优化，有效解决了协同感知中的性能-通信权衡问题，为多智能体系统的高效协作提供了新思路。

### 翻译

协同感知强调通过使多个智能体能够共享视觉信息来增强环境理解，同时受限于带宽资源。虽然之前的工作已经探索了任务性能和通信量之间的经验权衡，但理论基础仍存在显著差距。为了填补这一空白，我们借鉴信息理论，提出了一种面向多智能体协作的实用速率失真理论，专门用于分析目标导向多智能体系统中的性能-通信权衡。该理论具体化了两条设计最优通信策略的关键条件：提供实用相关信息和传输无冗余消息。基于这两个条件，我们提出了RDcomm，一种通信高效的协同感知框架，引入了两个关键创新：i) 任务熵离散编码，为具有任务相关码字长度的特征分配，以最大化提供实用信息的效率；ii) 互信息驱动的消息选择，利用互信息神经估计来接近最优无冗余条件。在DAIR-V2X和OPV2V上的3D目标检测和BEV分割实验表明，RDcomm实现了最先进的准确度，同时将通信量减少了高达108倍。代码将公开发布。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决多智能体协作感知中的通信优化问题，即在有限带宽资源下如何平衡感知任务性能和通信量。这个问题在自动驾驶、机器人协作等领域至关重要，因为这些场景中多个智能体需要共享视觉信息来增强环境理解，但通信带宽有限。如果通信量过大会导致网络拥塞和延迟，过度压缩又可能丢失关键信息影响任务性能，因此研究如何在性能和通信效率间取得平衡具有重要理论和实践意义。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者从信息论角度出发，借鉴经典率失真理论并针对多智能体协作特点进行扩展。他们首先分析现有工作局限性：大多数方法都是启发式的，缺乏理论基础。作者设计过程分三步：1) 提出多智能体协作的实用率失真理论，明确两个最优条件；2) 基于条件设计RDcomm框架；3) 通过实验验证有效性。作者借鉴了现有工作中的空间选择和特征压缩技术，但将其置于理论框架下，提供了更系统的方法。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是通过率失优化的方式，在保证感知任务性能前提下最小化通信量，即只传输对任务有用的信息且避免传输接收方已拥有的信息。整体流程：1) 感知管道将传感器输入转换为鸟瞰图特征；2) 任务熵离散编码：使用分层向量量化映射特征到码本，根据任务相关性分配不同长度编码；3) 互信息驱动消息选择：评估特征间冗余，选择互补性强的特征传输；4) 消息平滑和融合：对稀疏选择的消息平滑处理后与接收方本地特征融合。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点：1) 提出多智能体协作的实用率失真理论，明确最优通信策略的两个条件；2) 设计任务熵离散编码模块，根据任务相关性分配不同长度编码；3) 提出互信息驱动的消息选择模块，减少智能体间冗余；4) 设计RDcomm框架实现性能与通信效率平衡。相比之前工作，不同在于：提供了理论基础而非仅启发式方法；同时优化消息选择和编码；采用任务相关编码策略而非通用压缩方法；实验显示在大幅减少通信量同时保持或提升任务性能。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '本文提出基于率失真理论的RDcomm框架，通过任务相关编码和互信息驱动的消息选择，在多智能体协作感知中实现通信量大幅降低（最高108倍）同时保持或提升感知任务性能。'}


### 论文摘要

Collaborative perception emphasizes enhancing environmental understanding by enabling multiple agents to share visual information with limited bandwidth resources. While prior work has explored the empirical trade-off between task performance and communication volume, a significant gap remains in the theoretical foundation. To fill this gap, we draw on information theory and introduce a pragmatic rate-distortion theory for multi-agent collaboration, specifically formulated to analyze performance-communication trade-off in goal-oriented multi-agent systems. This theory concretizes two key conditions for designing optimal communication strategies: supplying pragmatically relevant information and transmitting redundancy-less messages. Guided by these two conditions, we propose RDcomm, a communication-efficient collaborative perception framework that introduces two key innovations: i) task entropy discrete coding, which assigns features with task-relevant codeword-lengths to maximize the efficiency in supplying pragmatic information; ii) mutual-information-driven message selection, which utilizes mutual information neural estimation to approach the optimal redundancy-less condition. Experiments on 3D object detection and BEV segmentation demonstrate that RDcomm achieves state-of-the-art accuracy on DAIR-V2X and OPV2V, while reducing communication volume by up to 108 times. The code will be released.

---

## 25. GeoSketch: A Neural-Symbolic Approach to Geometric Multimodal Reasoning with Auxiliary Line Construction and Affine Transformation

**论文链接:** [http://arxiv.org/abs/2509.22460v1](http://arxiv.org/abs/2509.22460v1)

**作者:** Shichao Weng, Zhiqiang Wang, Yuhua Zhou, Rui Lu, Ting Liu, Zhiyang Teng, Xiaozhang Liu, Hanmeng Liu

**发布时间:** 2025-09-26

### GPT解析

### 总结

该论文提出了GeoSketch，一个神经符号框架，用于解决几何问题。该框架将几何推理重新定义为交互式的感知-推理-行动循环，整合了感知模块、符号推理模块和草图行动模块。通过两阶段训练方法和GeoSketch基准测试，GeoSketch显著提高了多模态大语言模型在几何问题解决上的性能。

### 背景

几何问题解决（GPS）对多模态大语言模型（MLLMs）构成独特挑战，不仅需要联合解释文本和图表，还需要迭代的空间推理。现有方法将图表处理为静态图像，缺乏动态操作能力，而这正是人类几何推理的核心方面，涉及辅助线构建和仿射变换。

### 目的

开发一个能够处理几何问题解决中动态操作需求的框架，特别是辅助线构建和仿射变换，以提升多模态大语言模型在几何问题上的推理能力。

### 方法

GeoSketch是一个神经符号框架，包含三个主要模块：1) 感知模块：将图表抽象为结构化逻辑形式；2) 符号推理模块：应用几何定理决定下一个推理步骤；3) 草图行动模块：执行绘制辅助线或应用变换等操作，更新图表。训练采用两阶段流程：首先在2000个符号化整理的轨迹上进行监督微调，然后使用密集的符号奖励进行强化学习，以提高鲁棒性和策略探索能力。评估使用了GeoSketch基准测试，包含390个需要辅助构建或仿射变换的高质量几何问题。

### 主要发现

在强大的MLLM基线模型上的实验表明，GeoSketch显著提高了逐步推理准确性和问题解决成功率，优于静态感知方法。

### 结论

GeoSketch通过统一分层决策、可执行视觉动作和符号验证，将多模态推理从静态解释提升到动态、可验证的交互，为解决复杂空间视觉问题建立了新基础。

### 翻译

几何问题解决（GPS）对多模态大语言模型（MLLMs）提出了独特挑战，不仅需要联合解释文本和图表，还需要迭代的空间推理。虽然现有方法将图表处理为静态图像，但它们缺乏动态操作能力——这是人类几何推理的一个核心方面，涉及辅助线构建和仿射变换。我们提出了GeoSketch，一个神经符号框架，将几何推理重新定义为交互式的感知-推理-行动循环。GeoSketch整合：（1）将图表抽象为结构化逻辑形式的感知模块，（2）应用几何定理决定下一个推理步骤的符号推理模块，以及（3）执行绘制辅助线或应用变换等操作的草图行动模块，从而在闭环中更新图表。为了训练这个智能体，我们开发了一个两阶段流程：首先在2000个符号化整理的轨迹上进行监督微调，然后使用密集的符号奖励进行强化学习，以提高鲁棒性和策略探索。为了评估这一范式，我们引入了GeoSketch基准测试，一个包含390个需要辅助构建或仿射变换的高质量几何问题集合。在强大的MLLM基线模型上的实验表明，GeoSketch显著提高了逐步推理准确性和问题解决成功率，优于静态感知方法。通过统一分层决策、可执行视觉动作和符号验证，GeoSketch将多模态推理从静态提升到动态、可验证的交互，为解决复杂空间视觉问题建立了新基础。


### 论文摘要

Geometric Problem Solving (GPS) poses a unique challenge for Multimodal Large Language Models (MLLMs), requiring not only the joint interpretation of text and diagrams but also iterative visuospatial reasoning. While existing approaches process diagrams as static images, they lack the capacity for dynamic manipulation - a core aspect of human geometric reasoning involving auxiliary line construction and affine transformations. We present GeoSketch, a neural-symbolic framework that recasts geometric reasoning as an interactive perception-reasoning-action loop. GeoSketch integrates: (1) a Perception module that abstracts diagrams into structured logic forms, (2) a Symbolic Reasoning module that applies geometric theorems to decide the next deductive step, and (3) a Sketch Action module that executes operations such as drawing auxiliary lines or applying transformations, thereby updating the diagram in a closed loop. To train this agent, we develop a two-stage pipeline: supervised fine-tuning on 2,000 symbolic-curated trajectories followed by reinforcement learning with dense, symbolic rewards to enhance robustness and strategic exploration. To evaluate this paradigm, we introduce the GeoSketch Benchmark, a high-quality set of 390 geometry problems requiring auxiliary construction or affine transformations. Experiments on strong MLLM baselines demonstrate that GeoSketch significantly improves stepwise reasoning accuracy and problem-solving success over static perception methods. By unifying hierarchical decision-making, executable visual actions, and symbolic verification, GeoSketch advances multimodal reasoning from static interpretation to dynamic, verifiable interaction, establishing a new foundation for solving complex visuospatial problems.

---

## 26. UrbanFeel: A Comprehensive Benchmark for Temporal and Perceptual Understanding of City Scenes through Human Perspective

**论文链接:** [http://arxiv.org/abs/2509.22228v1](http://arxiv.org/abs/2509.22228v1)

**作者:** Jun He, Yi Lin, Zilong Huang, Jiacong Yin, Junyan Ye, Yuchuan Zhou, Weijia Li, Xiang Zhang

**发布时间:** 2025-09-26

**备注:** 13 pages, 6 figures

### GPT解析

### 总结

UrbanFeel是一个全面的基准测试，用于评估多模态大语言模型(MLLMs)在城市发展理解和主观环境感知方面的表现。通过评估20个最先进的模型，发现Gemini-2.5 Pro总体表现最佳，准确度接近人类专家水平，但在需要时间推理的城市发展任务中性能显著下降。

### 背景

城市发展影响着全球一半以上的人口，对城市结构和感知变化的人本主义理解对可持续发展至关重要。尽管MLLMs在各个领域展现出显著能力，但现有探索MLLMs在城市环境中表现的基准测试有限，缺乏对城市环境时间演变和主观感知的系统探索。

### 目的

提出UrbanFeel，一个全面的基准测试，用于评估MLLMs在城市发展理解和主观环境感知方面的表现，填补现有评估空白。

### 方法

UrbanFeel包含14.3K个精心构建的视觉问题，跨越三个认知渐进维度：静态场景感知、时间变化理解和主观环境感知。研究从全球11个代表性城市收集多时序单视图和全景街景图像，并通过空间聚类、基于规则的生成、模型辅助提示和人工注释的混合流程生成高质量问答对。

### 主要发现

Gemini-2.5 Pro总体表现最佳，准确度接近人类专家水平，平均差距仅1.5%。大多数模型在基于场景理解的任务上表现良好，一些模型甚至在像素级变化检测中超越了人类注释者。然而，在需要时间推理的城市发展任务中，性能显著下降。在主观感知维度，几个模型在美丽和安全等评估维度上达到或超过了人类水平的一致性。

### 结论

UrbanFill基准测试揭示了当前MLLMs在城市环境理解中的优势与局限，特别是在时间推理方面的不足，为未来模型改进提供了明确方向。

### 翻译

城市发展影响着全球一半以上的人口，使人本主义理解其结构和感知变化对可持续发展至关重要。虽然多模态大语言模型(MLLMs)已在各个领域展现出显著能力，但现有探索它们在城市环境中表现的基准测试仍然有限，缺乏对城市环境时间演变和主观感知的系统探索，这些探索应与人类感知相一致。为解决这些限制，我们提出了UrbanFeel，一个全面的基准测试，旨在评估MLLMs在城市发展理解和主观环境感知方面的表现。UrbanFeel包含14.3K个精心构建的视觉问题，跨越三个认知渐进维度：静态场景感知、时间变化理解和主观环境感知。我们从全球11个代表性城市收集多时序单视图和全景街景图像，并通过空间聚类、基于规则的生成、模型辅助提示和人工注释的混合流程生成高质量问答对。通过对20个最先进的MLLMs进行广泛评估，我们观察到Gemini-2.5 Pro取得了最佳总体表现，其准确度接近人类专家水平，将平均差距缩小至仅1.5%。大多数模型在基于场景理解的任务上表现良好。特别是，一些模型甚至在像素级变化检测中超越了人类注释者。然而，在需要时间推理的城市发展任务中，性能显著下降。此外，在主观感知维度，几个模型在美丽和安全等评估维度上达到或超过了人类水平的一致性。


### 论文摘要

Urban development impacts over half of the global population, making human-centered understanding of its structural and perceptual changes essential for sustainable development. While Multimodal Large Language Models (MLLMs) have shown remarkable capabilities across various domains, existing benchmarks that explore their performance in urban environments remain limited, lacking systematic exploration of temporal evolution and subjective perception of urban environment that aligns with human perception. To address these limitations, we propose UrbanFeel, a comprehensive benchmark designed to evaluate the performance of MLLMs in urban development understanding and subjective environmental perception. UrbanFeel comprises 14.3K carefully constructed visual questions spanning three cognitively progressive dimensions: Static Scene Perception, Temporal Change Understanding, and Subjective Environmental Perception. We collect multi-temporal single-view and panoramic street-view images from 11 representative cities worldwide, and generate high-quality question-answer pairs through a hybrid pipeline of spatial clustering, rule-based generation, model-assisted prompting, and manual annotation. Through extensive evaluation of 20 state-of-the-art MLLMs, we observe that Gemini-2.5 Pro achieves the best overall performance, with its accuracy approaching human expert levels and narrowing the average gap to just 1.5\%. Most models perform well on tasks grounded in scene understanding. In particular, some models even surpass human annotators in pixel-level change detection. However, performance drops notably in tasks requiring temporal reasoning over urban development. Additionally, in the subjective perception dimension, several models reach human-level or even higher consistency in evaluating dimension such as beautiful and safety.

---

## 27. Lightweight Structured Multimodal Reasoning for Clinical Scene Understanding in Robotics

**论文链接:** [http://arxiv.org/abs/2509.22014v1](http://arxiv.org/abs/2509.22014v1)

**作者:** Saurav Jha, Stefan K. Ehrlich

**发布时间:** 2025-09-26

**备注:** 11 pages, 3 figures

### GPT解析

### 总结

研究提出了一种轻量级智能体多模态框架，用于医疗机器人的视频场景理解，结合Qwen2.5-VL-3B-Instruct模型与SmolAgent编排层，支持思维链推理、语音视觉融合和动态工具调用，在Video-MME基准测试和临床数据集上展现出竞争性准确性和改进的鲁棒性。

### 背景

医疗机器人需要在动态临床环境中具有强大的多模态感知和推理能力以确保安全。当前的视觉-语言模型虽然展示了强大的通用能力，但在时间推理、不确定性估计和机器人规划所需的结构化输出方面仍然存在局限性。

### 目的

提出一个轻量级的智能体多模态框架，用于基于视频的场景理解，以解决当前VLMs在医疗机器人应用中的限制。

### 方法

将Qwen2.5-VL-3B-Instruct模型与基于SmolAgent的编排层相结合，支持思维链推理、语音视觉融合和动态工具调用。该框架生成结构化场景图，并利用混合检索模块进行可解释和自适应的推理。

### 主要发现

在Video-MME基准测试和自定义临床数据集上的评估表明，与最先进的VLMs相比，该框架具有竞争性的准确性和改进的鲁棒性。

### 结论

该框架展示了在机器人辅助手术、患者监测和决策支持中应用的潜力，为医疗机器人提供了更安全可靠的多模态感知和推理能力。

### 翻译

医疗机器人需要在动态临床环境中具有强大的多模态感知和推理能力以确保安全。当前的视觉-语言模型展示了强大的通用能力，但在时间推理、不确定性估计和机器人规划所需的结构化输出方面仍然存在局限性。我们提出了一个用于视频场景理解的轻量级智能体多模态框架。将Qwen2.5-VL-3B-Instruct模型与基于SmolAgent的编排层相结合，它支持思维链推理、语音视觉融合和动态工具调用。该框架生成结构化场景图，并利用混合检索模块进行可解释和自适应的推理。在Video-MME基准测试和自定义临床数据集上的评估显示，与最先进的VLMs相比具有竞争性的准确性和改进的鲁棒性，展示了其在机器人辅助手术、患者监测和决策支持中的应用潜力。


### 论文摘要

Healthcare robotics requires robust multimodal perception and reasoning to ensure safety in dynamic clinical environments. Current Vision-Language Models (VLMs) demonstrate strong general-purpose capabilities but remain limited in temporal reasoning, uncertainty estimation, and structured outputs needed for robotic planning. We present a lightweight agentic multimodal framework for video-based scene understanding. Combining the Qwen2.5-VL-3B-Instruct model with a SmolAgent-based orchestration layer, it supports chain-of-thought reasoning, speech-vision fusion, and dynamic tool invocation. The framework generates structured scene graphs and leverages a hybrid retrieval module for interpretable and adaptive reasoning. Evaluations on the Video-MME benchmark and a custom clinical dataset show competitive accuracy and improved robustness compared to state-of-the-art VLMs, demonstrating its potential for applications in robot-assisted surgery, patient monitoring, and decision support.

---

## 28. Spatial Reasoning in Foundation Models: Benchmarking Object-Centric Spatial Understanding

**论文链接:** [http://arxiv.org/abs/2509.21922v1](http://arxiv.org/abs/2509.21922v1)

**作者:** Vahid Mirjalili, Ramin Giahi, Sriram Kollipara, Akshay Kekuda, Kehui Yao, Kai Zhao, Jianpeng Xu, Kaushiki Nag, Sinduja Subramaniam, Topojoy Biswas, Evren Korpeoglu, Kannan Achan

**发布时间:** 2025-09-26

**备注:** 4 pages, NeurIPS Workshop SpaVLE

### GPT解析

### 总结

本研究评估了基础模型对物体为中心的空间推理能力，发现现有模型在定位精度和空间理解之间存在权衡，真正的空间理解能力仍然不足。

### 背景

空间理解是视觉基础模型的关键能力。最近大型视觉模型或视觉语言模型(VLMs)的发展扩展了识别能力，但大多数基准测试强调定位精度，而非模型是否捕捉到场景中物体的排列和关系。

### 目的

提出一个针对基础模型中物体为中心的空间推理的系统性基准测试，评估模型是否真正理解物体在场景中的相对位置、分组和深度。

### 方法

使用受控合成数据集，评估最先进的视觉模型(如GroundingDINO, Florence-2, OWLv2)和大型VLMs(如InternVL, LLaVA, GPT-4o)在三个任务上的表现：空间定位、空间推理和下游检索任务。

### 主要发现

存在稳定的权衡关系：检测器如GroundingDINO和OWLv2提供精确的边界框但有限的推理能力，而VLMs如SmolVLM和GPT-4o提供粗略的布局线索和流畅的描述，但在精细的空间上下文中表现不佳。

### 结论

研究突显了定位和真正空间理解之间的差距，指向社区需要开发具有空间感知能力的基础模型。

### 翻译

空间理解是视觉基础模型的关键能力。虽然最近大型视觉模型或视觉语言模型(VLMs)的进展扩展了识别能力，但大多数基准测试强调定位精度，而非模型是否捕捉到场景中物体的排列和关系。这一差距很重要；有效的场景理解不仅需要识别物体，还需要推理它们的相对位置、分组和深度。在本文中，我们提出了一个针对基础模型中物体为中心的空间推理的系统性基准。使用受控合成数据集，我们评估了最先进的视觉模型(如GroundingDINO, Florence-2, OWLv2)和大型VLMs(如InternVL, LLaVA, GPT-4o)在三个任务上的表现：空间定位、空间推理和下游检索任务。我们发现存在稳定的权衡关系：检测器如GroundingDINO和OWLv2提供精确的边界框但有限的推理能力，而VLMs如SmolVLM和GPT-4o提供粗略的布局线索和流畅的描述，但在精细的空间上下文中表现不佳。我们的研究突显了定位和真正空间理解之间的差距，并指向社区需要开发具有空间感知能力的基础模型。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决基础模型（特别是视觉模型和视觉-语言模型）中的空间推理能力评估问题，即模型是否能理解物体在场景中的排列方式和相互关系，而不仅仅是精确定位物体。这个问题很重要，因为有效的场景理解需要识别物体并理解其相对位置、分组和深度，这对电商推荐、人机交互、场景检索和具身AI等应用至关重要。例如，在购物场景中，沙发与咖啡桌的相对位置会影响推荐系统提供的相关商品建议。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者通过分析现有基准测试的局限性（主要强调定位准确性而非空间关系推理）来设计这个方法。他们创建了一个受控的合成数据集，包含9个家具类别的3D渲染图像，通过随机旋转、位移和缩放增加多样性，并与背景场景合成。他们借鉴了现有的开放词汇检测模型（如OWL-ViT、OWLv2）和视觉-语言模型（如LLaVA、InternVL）的评估方法，但创新性地设计了专门针对空间推理的评估任务和指标，包括空间定位、空间推理和下游检索任务。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是评估模型不仅能否精确定位物体，还能否理解物体之间的空间关系和上下文。整体流程包括：1) 数据生成：创建合成数据集，包含数据库图像（正面视图）和查询图像（角度视图）；2) 模型评估：评估14个模型（分为任务特定视觉模型和通用视觉-语言模型）在三种任务上的表现；3) 使用多种指标评估结果：空间定位使用准确率、macro-F1、MCC，空间推理使用准确率、精确率、召回率、F1，检索任务使用Precision@k和Hit@k。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1) 提出首个系统化的空间推理基准测试；2) 使用受控合成数据集精确控制变量；3) 同时评估多种类型的模型（视觉检测模型和视觉-语言模型）；4) 揭示精确定位与真实空间理解之间的差距；5) 提供标准化任务和指标。相比之前的工作，这篇论文不局限于定位准确性评估，而是专注于空间关系推理；不依赖真实世界数据集的不确定性，而是使用合成数据集进行受控评估；不仅评估单一模型类型，而是全面比较不同模型家族的优势和局限。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '这篇论文通过提出一个系统化的空间推理基准测试，揭示了当前基础模型在理解物体空间关系方面的局限性，为开发能够结合精确定位与空间上下文理解的新型基础模型奠定了基础。'}


### 论文摘要

Spatial understanding is a critical capability for vision foundation models. While recent advances in large vision models or vision-language models (VLMs) have expanded recognition capabilities, most benchmarks emphasize localization accuracy rather than whether models capture how objects are arranged and related within a scene. This gap is consequential; effective scene understanding requires not only identifying objects, but reasoning about their relative positions, groupings, and depth. In this paper, we present a systematic benchmark for object-centric spatial reasoning in foundation models. Using a controlled synthetic dataset, we evaluate state-of-the-art vision models (e.g., GroundingDINO, Florence-2, OWLv2) and large VLMs (e.g., InternVL, LLaVA, GPT-4o) across three tasks: spatial localization, spatial reasoning, and downstream retrieval tasks. We find a stable trade-off: detectors such as GroundingDINO and OWLv2 deliver precise boxes with limited relational reasoning, while VLMs like SmolVLM and GPT-4o provide coarse layout cues and fluent captions but struggle with fine-grained spatial context. Our study highlights the gap between localization and true spatial understanding, and pointing toward the need for spatially-aware foundation models in the community.

---

## 29. Text2Move: Text-to-moving sound generation via trajectory prediction and temporal alignment

**论文链接:** [http://arxiv.org/abs/2509.21919v1](http://arxiv.org/abs/2509.21919v1)

**作者:** Yunyi Liu, Shaofan Yang, Kai Li, Xu Li

**发布时间:** 2025-09-26

### GPT解析

### 总结

本文介绍了一个根据文本提示可控生成移动声音的框架，通过构建合成数据集训练文本到轨迹预测模型，并结合预训练的文本到音频生成模型，实现了空间音频的生成。

### 背景

人类听觉感知受到3D空间中移动声源的影响，但之前在生成式声音建模方面的工作主要局限于单声道信号或静态空间音频。

### 目的

引入一个框架，可以根据文本提示以可控方式生成移动声音。

### 方法

构建一个合成数据集，记录双耳格式的移动声音、它们的空间轨迹和文本描述；训练文本到轨迹预测模型；微调预训练的文本到音频生成模型以输出与轨迹时间对齐的单声道声音；使用预测的时间对齐轨迹模拟空间音频。

### 主要发现

文本到轨迹模型展现出合理的空间理解能力。

### 结论

该方法可以轻松集成到现有的文本到音频生成工作流程中，并可以扩展到其他空间音频格式中的移动声音生成。

### 翻译

人类的听觉感知受到3D空间中移动声源的影响，然而在生成式声音建模方面之前的工作主要局限于单声道信号或静态空间音频。在这项工作中，我们引入了一个框架，可以根据文本提示以可控方式生成移动声音。为了使训练成为可能，我们构建了一个合成数据集，记录了双耳格式的移动声音、它们的空间轨迹以及关于声音事件和空间运动的文本描述。使用这个数据集，我们训练了一个文本到轨迹预测模型，该模型根据文本提示输出移动声源的三维轨迹。为了生成空间音频，我们首先微调了一个预训练的文本到音频生成模型，使其输出与轨迹时间对齐的单声道声音。然后使用预测的时间对齐轨迹来模拟空间音频。实验评估表明文本到轨迹模型具有合理的空间理解能力。这种方法可以轻松集成到现有的文本到音频生成工作流程中，并扩展到其他空间音频格式中的移动声音生成。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决如何根据文本描述生成具有空间移动特性的声音问题。这个问题很重要，因为人类听觉感知受到3D空间中移动声源的影响，而现有生成声音模型主要局限于单声道信号或静态空间音频，无法处理现实中许多不断移动的声源。空间音频在导航、媒体沉浸体验等日常应用中扮演着重要角色。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者将问题分解为两个组件：文本到轨迹预测模型和同步的文本到音频管道。他们借鉴了文本到空间声音建模、对象音频方法、轨迹预测和声音合成等领域的现有工作。特别是借鉴了机器人、自动驾驶等领域中从语言预测运动轨迹的方法，以及预训练的文本到音频生成模型。作者还构建了专门的合成数据集来支持训练和评估。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是通过文本预测声源在3D空间中的移动轨迹，生成与轨迹时间对齐的单声道声音，然后根据预测轨迹模拟空间音频。实现流程包括：1)文本到轨迹预测模型，包含文本语义编码器、时间编码器和轨迹解码器；2)微调预训练的文本到音频生成模型并添加时间调整机制；3)使用预测轨迹对生成的单声道声音进行空间化处理；4)构建包含双耳音频、空间轨迹和文本描述的合成数据集。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)第一个明确将文本、轨迹和音频统一在空间音频生成框架中的工作；2)将问题分解为轨迹预测和时间对齐两个组件，而非端到端生成；3)构建了专门的合成数据集，包含双耳音频、空间轨迹和文本描述；4)引入时间调整机制，使生成的音频与预测轨迹精确对齐；5)实现了对移动声音的精确控制和生成，而之前的工作主要关注静态空间音频。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'Text2Move通过文本预测声源空间轨迹并生成与轨迹时间对齐的空间音频，实现了对移动声音的精确控制和生成。'}


### 论文摘要

Human auditory perception is shaped by moving sound sources in 3D space, yet prior work in generative sound modelling has largely been restricted to mono signals or static spatial audio. In this work, we introduce a framework for generating moving sounds given text prompts in a controllable fashion. To enable training, we construct a synthetic dataset that records moving sounds in binaural format, their spatial trajectories, and text captions about the sound event and spatial motion. Using this dataset, we train a text-to-trajectory prediction model that outputs the three-dimensional trajectory of a moving sound source given text prompts. To generate spatial audio, we first fine-tune a pre-trained text-to-audio generative model to output temporally aligned mono sound with the trajectory. The spatial audio is then simulated using the predicted temporally-aligned trajectory. Experimental evaluation demonstrates reasonable spatial understanding of the text-to-trajectory model. This approach could be easily integrated into existing text-to-audio generative workflow and extended to moving sound generation in other spatial audio formats.

---

## 30. Real-Time Indoor Object SLAM with LLM-Enhanced Priors

**论文链接:** [http://arxiv.org/abs/2509.21602v1](http://arxiv.org/abs/2509.21602v1)

**作者:** Yang Jiao, Yiding Qiu, Henrik I. Christensen

**发布时间:** 2025-09-25

### GPT解析

### 总结

本文提出了一种利用大型语言模型（LLMs）提供对象几何属性常识知识作为先验因素的方法，解决了对象级SLAM中因稀疏观测导致的优化约束不足问题，并在TUM RGB-D和3RScan数据集上将映射精度提高了36.8%。

### 背景

对象级SLAM结合语义信息进行高级场景理解，但由于稀疏观测面临优化约束不足的挑战。之前的工作使用常识知识引入额外约束，但获取这些先验知识劳动密集且缺乏跨对象类别的泛化能力。

### 目的

解决获取常识先验知识劳动密集且泛化能力有限的问题，利用大型语言模型（LLMs）提供对象几何属性的常识知识作为先验因素，提高对象级SLAM的性能。

### 方法

利用大型语言模型（LLMs）提供对象几何属性（大小和方向）的常识知识，在基于图的SLAM框架中将这些知识作为先验因素，实现完整流程整合这些先验知识，在稀疏对象级特征上实现鲁棒的数据关联，支持实时对象SLAM。

### 主要发现

这些先验知识在对象观测有限的初始阶段特别有益；在TUM RGB-D和3RScan数据集上评估的系统比最新基线提高映射精度36.8%；实验展示了系统的实时性能。

### 结论

使用大型语言模型提供常识先验知识可以显著提高对象级SLAM的性能；该方法解决了传统获取先验知识的劳动密集和泛化能力有限的问题。

### 翻译

对象级同步定位与地图构建（SLAM）结合语义信息进行高级场景理解，但由于稀疏观测面临优化约束不足的挑战。先前的工作使用常识知识引入额外约束，但获取此类先验知识传统上劳动密集且缺乏跨不同对象类别的泛化能力。我们通过利用大型语言模型（LLMs）提供对象几何属性的常识知识，作为基于图的SLAM框架中的先验因素，来解决这一局限性。这些先验知识在对象观测有限的初始阶段特别有益。我们实现了一个整合这些先验因素的完整流程，实现了在稀疏对象级特征上的鲁棒数据关联，并支持实时对象SLAM。我们的系统在TUM RGB-D和3RScan数据集上评估，比最新基线提高映射精度36.8%。此外，我们在补充视频中展示了真实世界实验，证明了其实时性能。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决物体级SLAM中因观测稀疏导致的优化约束不足问题。这个问题很重要，因为在实际应用中，当相机帧率低或机器人移动快时，物体观测往往有限，这会影响初始建图质量，阻碍后续优化和数据关联，进而影响机器人导航、物体搜索等下游任务的性能。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先分析了物体级SLAM面临的核心问题是观测稀疏导致的约束不足，然后考虑引入常识知识作为额外约束。他们创新性地利用大型语言模型(LLM)自动提供这些先验知识，而不是依赖传统的人工标注方法。该方法借鉴了现有的QuadricSLAM和CubeSLAM框架、因子图优化技术、视觉里程计(ORB-SLAM3)和物体检测(YOLO)等现有工作，但将它们与LLM先验知识进行了创新性整合。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是利用大型语言模型提供的常识知识(特别是物体的大小和方向)作为先验约束，整合到因子图SLAM框架中，解决稀疏观测下的优化问题。整体流程包括：1)RGB-D输入；2)ORB-SLAM3视觉里程计；3)YOLO物体检测；4)短期物体跟踪；5)长期物体关联；6)向LLM查询物体常识先验；7)构建包含里程计、观测和先验的因子图；8)使用iSAM2增量优化；9)输出物体级地图和相机轨迹。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)首次将LLM提供的常识知识直接整合到SLAM系统；2)利用LLM生成物体几何先验作为因子图约束；3)设计了完整的从先验嵌入到优化的流程；4)实现了实时物体级SLAM系统。相比之前工作，不同之处在于：先验知识获取从人工标注变为LLM自动生成；先验应用从仅用于初始化扩展到整个优化过程；提供了完整的实时系统而非单一组件；在数据集上实现了36.8%的精度提升。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '本文提出了一种利用大型语言模型提供的常识知识作为先验约束的实时物体级SLAM方法，显著提高了稀疏观测场景下的建图精度。'}


### 论文摘要

Object-level Simultaneous Localization and Mapping (SLAM), which incorporates semantic information for high-level scene understanding, faces challenges of under-constrained optimization due to sparse observations. Prior work has introduced additional constraints using commonsense knowledge, but obtaining such priors has traditionally been labor-intensive and lacks generalizability across diverse object categories. We address this limitation by leveraging large language models (LLMs) to provide commonsense knowledge of object geometric attributes, specifically size and orientation, as prior factors in a graph-based SLAM framework. These priors are particularly beneficial during the initial phase when object observations are limited. We implement a complete pipeline integrating these priors, achieving robust data association on sparse object-level features and enabling real-time object SLAM. Our system, evaluated on the TUM RGB-D and 3RScan datasets, improves mapping accuracy by 36.8\% over the latest baseline. Additionally, we present real-world experiments in the supplementary video, demonstrating its real-time performance.

---

## 31. Residual Vector Quantization For Communication-Efficient Multi-Agent Perception

**论文链接:** [http://arxiv.org/abs/2509.21464v1](http://arxiv.org/abs/2509.21464v1)

**作者:** Dereje Shenkut, B. V. K Vijaya Kumar

**发布时间:** 2025-09-25

**备注:** 5 pages

### GPT解析

### 总结

ReVQom是一种创新的多智能体协同感知特征压缩方法，通过学习到的特征编解码器在保持空间身份的同时大幅压缩特征数据，解决了通信带宽限制问题，实现了高效且准确的多智能体协同感知。

### 背景

多智能体协同感知(CP)通过连接的智能体(如自动驾驶汽车、无人机和机器人)共享信息来提高场景理解能力，但通信带宽限制了其可扩展性。

### 目的

提出ReVQom，一种学习到的特征编解码器，在压缩中间特征的同时保持空间身份，以解决通信带宽限制问题。

### 方法

ReVQom是一种端到端方法，通过简单的瓶颈网络压缩特征维度，然后进行多阶段残差向量量化(RVQ)，只传输每个像素的代码索引来减少数据负载。

### 主要发现

ReVQom将未压缩的32位浮点特征从每像素8192位压缩到每智能体6-30位，精度损失最小；在DAIR-V2X数据集上，30 bpp时实现273倍压缩，6 bpp时实现1365倍压缩；18 bpp时匹配或优于原始特征CP；6-12 bpp时实现超低带宽操作，性能优雅降级。

### 结论

ReVQom实现了高效且准确的多智能体协同感知，为V2X(车对万物)的实际部署提供了可能性。

### 翻译

多智能体协同感知(CP)通过连接的智能体(如自动驾驶汽车、无人机和机器人)共享信息来提高场景理解能力。然而，通信带宽限制了其可扩展性。我们提出了ReVQom，一种学习到的特征编解码器，在压缩中间特征的同时保持空间身份。ReVQom是一种端到端方法，通过简单的瓶颈网络压缩特征维度，然后进行多阶段残差向量量化(RVQ)。这使得只需传输每个像素的代码索引，将未压缩的32位浮点特征从每像素8192位减少到每智能体6-30位，精度损失最小。在DAIR-V2X真实世界CP数据集上，ReVQom在30 bpp时实现273倍压缩，在6 bpp时实现1365倍压缩。在18 bpp(455倍)时，ReVQom匹配或优于原始特征CP，在6-12 bpp时，它实现了超低带宽操作，性能优雅降级。ReVQom实现了高效且准确的多智能体协同感知，为V2X的实际部署迈出了一步。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决多智能体协作感知中的通信带宽限制问题。在自动驾驶、无人机和机器人等场景中，多个智能体需要共享信息来提高场景理解能力，但原始特征数据量巨大（每像素8192位），严重限制了这种协作的扩展性和实际应用。这个问题在现实中非常重要，因为它阻碍了多智能体感知技术的规模化部署，特别是在带宽受限的V2X（车对万物）通信环境中。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': "作者首先分析了现有方法的局限性，发现它们主要关注'传输什么特征'而非'如何高效压缩特征'。然后提出使用残差向量量化（RVQ）来实现高效压缩。该方法借鉴了向量量化的基本思想，但创新性地将其扩展为多阶段残差量化；同时采用指数移动平均（EMA）更新码本技术，这与一些自学习方法类似。作者还保留了鸟瞰图（BEV）特征表示，与CoBEVT等多智能体融合框架一致，但将这些技术组合创新，形成了专门针对多智能体协作感知的高效压缩方案。", '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是通过多阶段残差向量量化将高维特征压缩为少量码本索引，只传输这些索引而非完整特征，从而大幅减少通信量，同时保持空间身份信息确保重构特征的准确性。整体流程：发送方先通过1×1卷积减少通道数，然后进行多阶段量化，每阶段找到最接近残差的码本向量，记录索引并更新残差，最后传输所有索引；接收方根据索引查找码本向量，累积这些向量，应用后处理重构特征，用于多智能体融合和检测。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)空间身份保持的编解码器，通过1×1瓶颈网络和多阶段RVQ保持空间结构；2)仅传输索引的通信方案，将通信量从8192bpp降至6-30bpp；3)实用化的多智能体BEV融合集成，在极低带宽下仍能工作；4)系统化的码本自适应机制，使用EMA更新码本。相比之前的工作，ReVQom同时解决了高压缩率和保持空间信息两个关键问题，在18bpp时性能甚至优于原始特征传输的协作感知方法，而Where2comm和What2comm等方法主要关注选择传输哪些特征而非如何压缩。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'ReVQom通过创新的残差向量量化方法，实现了多智能体协作感知中特征数据的高效压缩，在保持空间结构的同时将通信量减少到原来的1/273至1/1365，为实际V2X部署铺平了道路。'}


### 论文摘要

Multi-agent collaborative perception (CP) improves scene understanding by sharing information across connected agents such as autonomous vehicles, unmanned aerial vehicles, and robots. Communication bandwidth, however, constrains scalability. We present ReVQom, a learned feature codec that preserves spatial identity while compressing intermediate features. ReVQom is an end-to-end method that compresses feature dimensions via a simple bottleneck network followed by multi-stage residual vector quantization (RVQ). This allows only per-pixel code indices to be transmitted, reducing payloads from 8192 bits per pixel (bpp) of uncompressed 32-bit float features to 6-30 bpp per agent with minimal accuracy loss. On DAIR-V2X real-world CP dataset, ReVQom achieves 273x compression at 30 bpp to 1365x compression at 6 bpp. At 18 bpp (455x), ReVQom matches or outperforms raw-feature CP, and at 6-12 bpp it enables ultra-low-bandwidth operation with graceful degradation. ReVQom allows efficient and accurate multi-agent collaborative perception with a step toward practical V2X deployment.

---

## 32. Syncphony: Synchronized Audio-to-Video Generation with Diffusion Transformers

**论文链接:** [http://arxiv.org/abs/2509.21893v1](http://arxiv.org/abs/2509.21893v1)

**作者:** Jibin Song, Mingi Kwon, Jaeseok Jeong, Youngjung Uh

**发布时间:** 2025-09-26

**备注:** Project page: https://jibin86.github.io/syncphony_project_page

### GPT解析

### 总结

Syncphony是一种音频到视频生成模型，能够生成与音频精确同步的高质量视频，解决了现有方法在时序控制方面的局限性。

### 背景

文本到视频和图像到视频生成在视觉质量方面取得了进展，但在控制运动时序方面仍然有限。音频与视频运动时间线索对齐，是时序控制视频生成的有前景的条件。然而，现有的音频到视频模型由于间接的条件机制或有限的时序建模能力，难以实现细粒度的同步。

### 目的

开发一个能够生成与音频精确同步的高质量视频的模型，提高音频与视频之间的同步精度。

### 方法

Syncphony构建在预训练的视频骨干模型之上，并包含两个关键组件：1) 运动感知损失，强调在高运动区域的学习；2) 音频同步引导，使用没有音频层的视觉对齐不同步模型引导整个模型，在推理时更好地利用音频线索，同时保持视觉质量。

### 主要发现

提出了CycleSync评估指标，这是一种基于视频到音频的度量，用于测量生成视频中运动线索的数量以重建原始音频。在AVSync15和The Greatest Hits数据集上的实验表明，Syncphony在同步准确性和视觉质量方面都优于现有方法。

### 结论

Syncphony是一种有效的音频到视频生成方法，能够生成380x640分辨率、24fps的高质量视频，并与各种音频输入精确同步，在同步准确性和视觉质量方面表现优异。

### 翻译

文本到视频和图像到视频生成在视觉质量方面取得了快速进展，但它们在控制运动精确时序方面仍然有限。相比之下，音频提供了与视频运动对齐的时间线索，使其成为时序控制视频生成的有前景的条件。然而，现有的音频到视频模型由于间接的条件机制或有限的时序建模能力，在细粒度同步方面存在困难。我们提出了Syncphony，它可以生成与各种音频输入同步的380x640分辨率、24fps视频。我们的方法基于预训练的视频骨干模型，并包含两个关键组件来提高同步性：1) 运动感知损失，强调在高运动区域的学习；2) 音频同步引导，使用没有音频层的视觉对齐不同步模型引导整个模型，在推理时更好地利用音频线索，同时保持视觉质量。为了评估同步性，我们提出了CycleSync，这是一种基于视频到音频的指标，用于测量生成视频中运动线索的数量以重建原始音频。在AVSync15和The Greatest Hits数据集上的实验表明，Syncphony在同步准确性和视觉质量方面都优于现有方法。项目页面可在：https://jibin86.github.io/syncphony_project_page 获取。


### 论文摘要

Text-to-video and image-to-video generation have made rapid progress in visual quality, but they remain limited in controlling the precise timing of motion. In contrast, audio provides temporal cues aligned with video motion, making it a promising condition for temporally controlled video generation. However, existing audio-to-video (A2V) models struggle with fine-grained synchronization due to indirect conditioning mechanisms or limited temporal modeling capacity. We present Syncphony, which generates 380x640 resolution, 24fps videos synchronized with diverse audio inputs. Our approach builds upon a pre-trained video backbone and incorporates two key components to improve synchronization: (1) Motion-aware Loss, which emphasizes learning at high-motion regions; (2) Audio Sync Guidance, which guides the full model using a visually aligned off-sync model without audio layers to better exploit audio cues at inference while maintaining visual quality. To evaluate synchronization, we propose CycleSync, a video-to-audio-based metric that measures the amount of motion cues in the generated video to reconstruct the original audio. Experiments on AVSync15 and The Greatest Hits datasets demonstrate that Syncphony outperforms existing methods in both synchronization accuracy and visual quality. Project page is available at: https://jibin86.github.io/syncphony_project_page

---

## 33. A Comprehensive Evaluation of Transformer-Based Question Answering Models and RAG-Enhanced Design

**论文链接:** [http://arxiv.org/abs/2509.21845v1](http://arxiv.org/abs/2509.21845v1)

**作者:** Zichen Zhang, Kunlong Zhang, Hongwei Ruan, Yiming Luo

**发布时间:** 2025-09-26

### GPT解析

### 总结

本研究评估了多跳问答中的检索策略，提出了一种结合密集嵌入与词汇重叠的混合方法，并通过优化EfficientRAG管道提高了检索效率。实验证明，该方法在HotpotQA数据集上显著优于传统方法，为多跳问答提供了准确、高效且可解释的零样本解决方案。

### 背景

基于Transformer的模型推动了问答领域的发展，但多跳推理（需要结合多个段落证据来回答问题）仍然具有挑战性。

### 目的

本文对检索增强生成框架下的多跳问答检索策略进行了全面评估。

### 方法

比较了余弦相似度、最大边际相关性和一种结合密集嵌入与词汇重叠及重排序的混合方法。为了进一步提高检索效果，调整了EfficientRAG管道进行查询优化，引入了标记标记和迭代优化，同时保持效率。

### 主要发现

在HotpotQA数据集上的实验显示，混合方法显著优于基线方法，与余弦相似度相比，精确匹配相对提高了50%，F1分数相对提高了47%。错误分析表明，混合检索提高了实体召回率和证据互补性，但在处理干扰项和时间推理方面仍有局限。

### 结论

总体而言，结果表明混合检索增强生成为多跳问答提供了一个实用的零样本解决方案，平衡了准确性、效率和可解释性。

### 翻译

基于Transformer的模型推动了问答领域的发展，但多跳推理（需要结合多个段落证据来回答问题）仍然具有挑战性。本文对检索增强生成框架下的多跳问答检索策略进行了全面评估。我们比较了余弦相似度、最大边际相关性和一种结合密集嵌入与词汇重叠及重排序的混合方法。为了进一步提高检索效果，我们调整了EfficientRAG管道进行查询优化，引入了标记标记和迭代优化，同时保持效率。在HotpotQA数据集上的实验显示，混合方法显著优于基线方法，与余弦相似度相比，精确匹配相对提高了50%，F1分数相对提高了47%。错误分析表明，混合检索提高了实体召回率和证据互补性，但在处理干扰项和时间推理方面仍有局限。总体而言，结果表明混合检索增强生成为多跳问答提供了一个实用的零样本解决方案，平衡了准确性、效率和可解释性。


### 论文摘要

Transformer-based models have advanced the field of question answering, but multi-hop reasoning, where answers require combining evidence across multiple passages, remains difficult. This paper presents a comprehensive evaluation of retrieval strategies for multi-hop question answering within a retrieval-augmented generation framework. We compare cosine similarity, maximal marginal relevance, and a hybrid method that integrates dense embeddings with lexical overlap and re-ranking. To further improve retrieval, we adapt the EfficientRAG pipeline for query optimization, introducing token labeling and iterative refinement while maintaining efficiency. Experiments on the HotpotQA dataset show that the hybrid approach substantially outperforms baseline methods, achieving a relative improvement of 50 percent in exact match and 47 percent in F1 score compared to cosine similarity. Error analysis reveals that hybrid retrieval improves entity recall and evidence complementarity, while remaining limited in handling distractors and temporal reasoning. Overall, the results suggest that hybrid retrieval-augmented generation provides a practical zero-shot solution for multi-hop question answering, balancing accuracy, efficiency, and interpretability.

---

## 34. Prompt-guided Representation Disentanglement for Action Recognition

**论文链接:** [http://arxiv.org/abs/2509.21783v1](http://arxiv.org/abs/2509.21783v1)

**作者:** Tianci Wu, Guangming Zhu, Jiang Lu, Siyuan Wang, Ning Wang, Nuoye Xiong, Zhang Liang

**发布时间:** 2025-09-26

### GPT解析

### 总结

本文提出了一种名为ProDA的新框架，可以从多动作场景中分离指定动作，通过时空场景图和动态提示模块引导图解析神经网络生成动作特定表示，在视频动作识别实验中证明了其有效性。

### 背景

动作识别是视频理解的基本任务，现有方法通常提取统一特征来处理一个视频中的所有动作，这使得在多动作场景中建模不同对象之间的交互变得具有挑战性。

### 目的

探索从复杂场景中分离指定动作作为有效解决方案，以解决多动作场景中不同对象间交互建模的挑战。

### 方法

提出了提示引导的解耦表示用于动作识别（ProDA）框架，利用时空场景图（SSGs）和引入动态提示模块（DPM）来引导图解析神经网络（GPNN）生成动作特定的表示，并设计了视频适配的GPNN，使用动态权重聚合信息。

### 主要发现

在视频动作识别实验中，与最先进的方法相比，ProDA方法证明了其有效性。

### 结论

ProDA框架能够有效处理多动作场景中的动作识别任务，通过分离指定动作并生成动作特定表示，解决了现有方法的局限性。

### 翻译

动作识别是视频理解的基本任务。现有方法通常提取统一特征来处理一个视频中的所有动作，这使得在多动作场景中建模不同对象之间的交互变得具有挑战性。为了缓解这一问题，我们探索从复杂场景中分离任何指定动作作为有效解决方案。在本文中，我们提出了提示引导的解耦表示用于动作识别（ProDA），这是一个新框架，可以从多动作场景中分离任何指定的动作。ProDA利用时空场景图（SSGs）并引入动态提示模块（DPM）来引导图解析神经网络（GPNN）生成动作特定的表示。此外，我们设计了一个视频适配的GPNN，使用动态权重聚合信息。在视频动作识别实验中，与最先进的方法相比，我们的方法证明了其有效性。我们的代码可以在https://github.com/iamsnaping/ProDA.git找到。


### 论文摘要

Action recognition is a fundamental task in video understanding. Existing methods typically extract unified features to process all actions in one video, which makes it challenging to model the interactions between different objects in multi-action scenarios. To alleviate this issue, we explore disentangling any specified actions from complex scenes as an effective solution. In this paper, we propose Prompt-guided Disentangled Representation for Action Recognition (ProDA), a novel framework that disentangles any specified actions from a multi-action scene. ProDA leverages Spatio-temporal Scene Graphs (SSGs) and introduces Dynamic Prompt Module (DPM) to guide a Graph Parsing Neural Network (GPNN) in generating action-specific representations. Furthermore, we design a video-adapted GPNN that aggregates information using dynamic weights. Experiments in video action recognition demonstrate the effectiveness of our approach when compared with the state-of-the-art methods. Our code can be found in https://github.com/iamsnaping/ProDA.git

---

## 35. Temporal vs. Spatial: Comparing DINOv3 and V-JEPA2 Feature Representations for Video Action Analysis

**论文链接:** [http://arxiv.org/abs/2509.21595v1](http://arxiv.org/abs/2509.21595v1)

**作者:** Sai Varun Kodathala, Rakesh Vunnam

**发布时间:** 2025-09-25

### GPT解析

### 总结

本研究比较了DINOv3和V-JEPA2两种视频动作识别的自监督学习架构，在UCF Sports数据集上评估了它们的特征质量。

### 背景

研究关注两种用于视频动作识别的自监督学习架构：DINOv3（通过空间特征提取独立处理帧）和V-JEPA2（在视频序列中采用联合时间建模）。

### 目的

评估这两种方法在UCF Sports数据集上的特征质量，包括分类准确率、聚类性能、类内一致性和类间判别能力。

### 方法

在UCF Sports数据集上评估DINOv3和V-JEPA2两种架构，通过多个维度分析特征质量。

### 主要发现

DINOv3在聚类性能（轮廓分数：0.31比0.21）和判别能力（6.16倍分离比）上表现更好，特别是对于姿态可识别的动作；V-JEPA2在所有动作类型上表现出一致可靠性，性能方差显著更低（0.094比0.288）；DINOv3在静态姿态识别上表现优异，但在依赖于动作的任务上表现下降；V-JEPA2的时间建模提供了跨不同动作类别的平衡表示质量。

### 结论

这些发现有助于理解视频分析系统中的架构设计选择，并根据任务要求和可靠性约束提供选择适当特征提取方法的实证指导。

### 翻译

本研究对两种用于视频动作识别的 prominent 自监督学习架构进行了全面的比较分析：DINOv3，它通过空间特征提取独立处理帧；以及 V-JEPA2，它在视频序列中采用联合时间建模。我们在 UCF Sports 数据集上评估了这两种方法，通过多个维度检查特征质量，包括分类准确率、聚类性能、类内一致性和类间判别能力。我们的分析揭示了基本的架构权衡：DINOv3 实现了更好的聚类性能（轮廓分数：0.31 比 0.21）并表现出卓越的判别能力（6.16 倍分离比），特别是对于姿态可识别的动作，而 V-JEPA2 在所有动作类型上表现出一致的可靠性，性能方差显著更低（0.094 比 0.288）。通过针对特定动作的评估，我们发现 DINOv3 的空间处理架构在静态姿态识别方面表现出色，但在依赖于动作的任务上表现出性能下降，而 V-JEPA2 的时间建模提供了跨不同动作类别的平衡表示质量。这些发现有助于理解视频分析系统中的架构设计选择，并根据任务要求和可靠性约束为选择适当的特征提取方法提供实证指导。


### 论文摘要

This study presents a comprehensive comparative analysis of two prominent self-supervised learning architectures for video action recognition: DINOv3, which processes frames independently through spatial feature extraction, and V-JEPA2, which employs joint temporal modeling across video sequences. We evaluate both approaches on the UCF Sports dataset, examining feature quality through multiple dimensions including classification accuracy, clustering performance, intra-class consistency, and inter-class discrimination. Our analysis reveals fundamental architectural trade-offs: DINOv3 achieves superior clustering performance (Silhouette score: 0.31 vs 0.21) and demonstrates exceptional discrimination capability (6.16x separation ratio) particularly for pose-identifiable actions, while V-JEPA2 exhibits consistent reliability across all action types with significantly lower performance variance (0.094 vs 0.288). Through action-specific evaluation, we identify that DINOv3's spatial processing architecture excels at static pose recognition but shows degraded performance on motion-dependent actions, whereas V-JEPA2's temporal modeling provides balanced representation quality across diverse action categories. These findings contribute to the understanding of architectural design choices in video analysis systems and provide empirical guidance for selecting appropriate feature extraction methods based on task requirements and reliability constraints.

---

## 36. VideoJudge: Bootstrapping Enables Scalable Supervision of MLLM-as-a-Judge for Video Understanding

**论文链接:** [http://arxiv.org/abs/2509.21451v1](http://arxiv.org/abs/2509.21451v1)

**作者:** Abdul Waheed, Zhen Wu, Dareen Alharthi, Seungone Kim, Bhiksha Raj

**发布时间:** 2025-09-25

**备注:** Work in progress

### GPT解析

### 总结

本文提出了VideoJudge，一个专门用于评估视频理解模型输出的3B和7B大小的多模态大型语言模型评估器，在多个基准测试中表现优于更大的评估器基线。

### 背景

精确评估视频理解模型具有挑战性，常用指标如BLEU、ROUGE和BERTScore无法捕捉人类判断的细微差别，而人工评估成本高昂。

### 目的

探索使用大型语言模型或多模态大型语言模型作为视频理解模型评估工具的可能性。

### 方法

提出VideoJudge评估器，通过生成器和评估器之间的交互进行训练：生成器根据目标评分提示生成响应，不符合评估器评分的响应被丢弃。

### 主要发现

VideoJudge-7B在四个元评估基准中的三个上表现优于Qwen2.5-VL(32B和72B)等更大的MLLM评估器基线；LLM评估器表现不如MLLM评估器；长链式推理不会提高性能。

### 结论

VideoJudge是一个有效的视频理解模型评估工具，比更大的模型表现更好，并且强调了视频输入在评估中的重要性。

### 翻译

精确评估视频理解模型仍然具有挑战性：常用的指标如BLEU、ROUGE和BERTScore无法捕捉人类判断的细微差别，而通过人工评估获得这种判断成本高昂。最近的工作已经探索使用大型语言模型(LLMs)或多模态大型语言模型(MLLMs)作为评估工具，但它们在视频理解领域的扩展仍然相对未被探索。在这项工作中，我们引入了VideoJudge，一个专门用于评估视频理解模型输出的3B和7B大小的MLLM评估器(即基于视频条件的文本响应)。为了训练VideoJudge，我们的方法建立在生成器和评估者之间的互动基础上：生成器被提示根据目标评分产生响应，不符合评估者评分的响应被丢弃。在四个元评估基准中的三个上，VideoJudge-7B优于更大的MLLM评估器基线，如Qwen2.5-VL(32B和72B)。值得注意的是，我们发现LLM评估器(Qwen3)模型表现不如MLLM评估器(Qwen2.5-VL)，且长链式推理不会提高性能，这表明提供视频输入对于评估视频理解任务至关重要。


### 论文摘要

Precisely evaluating video understanding models remains challenging: commonly used metrics such as BLEU, ROUGE, and BERTScore fail to capture the fineness of human judgment, while obtaining such judgments through manual evaluation is costly. Recent work has explored using large language models (LLMs) or multimodal LLMs (MLLMs) as evaluators, but their extension to video understanding remains relatively unexplored. In this work, we introduce VideoJudge, a 3B and 7B-sized MLLM judge specialized to evaluate outputs from video understanding models (\textit{i.e.}, text responses conditioned on videos). To train VideoJudge, our recipe builds on the interplay between a generator and an evaluator: the generator is prompted to produce responses conditioned on a target rating, and responses not matching the evaluator's rating are discarded. Across three out of four meta-evaluation benchmarks, VideoJudge-7B outperforms larger MLLM judge baselines such as Qwen2.5-VL (32B and 72B). Notably, we find that LLM judges (Qwen3) models perform worse than MLLM judges (Qwen2.5-VL) and long chain-of-thought reasoning does not improve performance, indicating that providing video inputs is crucial for evaluation of video understanding tasks.

---

## 37. MOSS-ChatV: Reinforcement Learning with Process Reasoning Reward for Video Temporal Reasoning

**论文链接:** [http://arxiv.org/abs/2509.21113v2](http://arxiv.org/abs/2509.21113v2)

**作者:** Sicheng Tao, Jungang Li, Yibo Yan, Junyan Zhang, Yubo Gao, Hanqian Li, ShuHang Xun, Yuxuan Fan, Hong Chen, Jianxiang He, Xuming Hu

**发布时间:** 2025-09-25

### GPT解析

### 总结

论文提出MOSS-ChatV框架，通过基于动态时间规整的过程奖励解决了多模态大语言模型在视频推理过程中的不一致性问题，显著提升了模型对时间动态的理解能力和推理轨迹的稳定性。

### 背景

视频推理已成为多模态大语言模型的关键能力，但现有模型往往表现出过程不一致性，即使最终答案正确，中间推理也会偏离视频动态，损害了模型的可解释性和鲁棒性。

### 目的

解决现有多模态大语言模型在视频推理过程中的不一致性问题，提高模型对视频中时间动态的理解能力和推理轨迹的稳定性。

### 方法

引入MOSS-ChatV强化学习框架，采用基于动态时间规整的过程奖励机制，无需辅助奖励模型即可实现高效的过程监督；同时构建MOSS-Video基准数据集，包含标注的推理轨迹，用于模型训练和评估。

### 主要发现

MOSS-ChatV在MOSS-Video测试集上达到87.2%的性能，在MVBench和MMVU等通用视频基准上表现提升；该框架在不同架构(包括Qwen2.5-VL和Phi-2)上均能带来性能提升；评估显示MOSS-ChatV产生更一致和稳定的推理轨迹。

### 结论

MOSS-ChatV框架有效解决了多模态大语言模型在视频推理过程中的不一致性问题，具有广泛的架构适用性，通过提高推理过程的一致性增强了模型的可解释性和鲁棒性。

### 翻译

视频推理已成为多模态大语言模型的关键能力，要求模型超越静态感知，转向对复杂场景中时间动态的连贯理解。然而，现有的多模态大语言模型通常表现出过程不一致性，即使最终答案正确，中间推理也会偏离视频动态，损害了可解释性和鲁棒性。为解决这一问题，我们引入了MOSS-ChatV，这是一个基于强化学习的框架，采用基于动态时间规整的过程奖励。这种基于规则的奖励将推理轨迹与时间上锚定的参考对齐，无需辅助奖励模型即可实现高效的过程监督。我们进一步将动态状态预测确定为视频推理的关键度量，并构建了MOSS-Video基准，其中包含标注的推理轨迹，训练集用于微调MOSS-ChatV，保留集用于评估。MOSS-ChatV在MOSS-Video(测试)上达到87.2%，并提高了在MVBench和MMVU等通用视频基准上的性能。该框架在不同架构(包括Qwen2.5-VL和Phi-2)上 consistently 带来提升，证实了其广泛的适用性。使用GPT-4o-as-judge的进一步评估表明，MOSS-ChatV产生更一致和稳定的推理轨迹。


### 论文摘要

Video reasoning has emerged as a critical capability for multimodal large language models (MLLMs), requiring models to move beyond static perception toward coherent understanding of temporal dynamics in complex scenes. Yet existing MLLMs often exhibit process inconsistency, where intermediate reasoning drifts from video dynamics even when the final answer is correct, undermining interpretability and robustness. To address this issue, we introduce MOSS-ChatV, a reinforcement learning framework with a Dynamic Time Warping (DTW)-based process reward. This rule-based reward aligns reasoning traces with temporally grounded references, enabling efficient process supervision without auxiliary reward models. We further identify dynamic state prediction as a key measure of video reasoning and construct MOSS-Video, a benchmark with annotated reasoning traces, where the training split is used to fine-tune MOSS-ChatV and the held-out split is reserved for evaluation. MOSS-ChatV achieves 87.2\% on MOSS-Video (test) and improves performance on general video benchmarks such as MVBench and MMVU. The framework consistently yields gains across different architectures, including Qwen2.5-VL and Phi-2, confirming its broad applicability. Evaluations with GPT-4o-as-judge further show that MOSS-ChatV produces more consistent and stable reasoning traces.

---

## 38. Linear Causal Representation Learning by Topological Ordering, Pruning, and Disentanglement

**论文链接:** [http://arxiv.org/abs/2509.22553v1](http://arxiv.org/abs/2509.22553v1)

**作者:** Hao Chen, Lin Liu, Yu Guang Wang

**发布时间:** 2025-09-26

### GPT解析

### 总结

本文提出了一种新型线性因果表征学习算法，在较弱的假设条件下仍能恢复潜在因果特征，并通过实验验证了其优越性和在大型语言模型可解释性分析中的应用潜力。

### 背景

因果表征学习(CRL)在因果推理和人工智能领域受到越来越多的关注，因为它能够利用现代数据集的异质性，将复杂数据生成机制解构为可因果解释的潜在特征。

### 目的

为CRL文献做出进一步贡献，专注于潜在特征上的风格化线性结构因果模型，并假设线性混合函数将潜在特征映射到观测数据。

### 方法

提出一种新型线性CRL算法，与现有方法不同，它在关于环境异质性和数据生成分布的较弱假设下运行，同时仍能恢复到等价类的潜在因果特征。

### 主要发现

新算法在有限样本中优于竞争方法，且在大型语言模型的可解释性分析中展现出将因果性集成到AI中的潜力。

### 结论

该算法克服了现有线性CRL方法对单节点干预数据或潜在特征和外生测量噪声分布的严格依赖要求，为因果表征学习提供了更实用的解决方案。

### 翻译

因果表征学习(CRL)因其能够利用现代数据集的异质性，将潜在复杂的数据生成机制解构为可因果解释的潜在特征，从而在因果推理和人工智能社区获得了越来越多的关注。在本文中，我们通过关注潜在特征上的风格化线性结构因果模型并假设将潜在特征映射到观测数据或测量值的线性混合函数，进一步为CRL文献做出贡献。现有的线性CRL方法通常依赖于严格的假设，例如访问单节点干预数据或对潜在特征和外生测量噪声的限制性分布约束。然而，这些前提在某些情况下可能难以满足。在本工作中，我们提出了一种新颖的线性CRL算法，与大多数现有的线性CRL方法不同，它在关于环境异质性和数据生成分布的较弱假设下运行，同时仍然能够恢复到等价类的潜在因果特征。我们通过合成实验和对大型语言模型(LLMs)的可解释性分析进一步验证了我们的新算法，展示了其在有限样本中优于竞争方法的潜力以及将因果性集成到AI中的潜力。


### 论文摘要

Causal representation learning (CRL) has garnered increasing interests from the causal inference and artificial intelligence community, due to its capability of disentangling potentially complex data-generating mechanism into causally interpretable latent features, by leveraging the heterogeneity of modern datasets. In this paper, we further contribute to the CRL literature, by focusing on the stylized linear structural causal model over the latent features and assuming a linear mixing function that maps latent features to the observed data or measurements. Existing linear CRL methods often rely on stringent assumptions, such as accessibility to single-node interventional data or restrictive distributional constraints on latent features and exogenous measurement noise. However, these prerequisites can be challenging to satisfy in certain scenarios. In this work, we propose a novel linear CRL algorithm that, unlike most existing linear CRL methods, operates under weaker assumptions about environment heterogeneity and data-generating distributions while still recovering latent causal features up to an equivalence class. We further validate our new algorithm via synthetic experiments and an interpretability analysis of large language models (LLMs), demonstrating both its superiority over competing methods in finite samples and its potential in integrating causality into AI.

---

## 39. Category Discovery: An Open-World Perspective

**论文链接:** [http://arxiv.org/abs/2509.22542v1](http://arxiv.org/abs/2509.22542v1)

**作者:** Zhenqi He, Yuanpei Liu, Kai Han

**发布时间:** 2025-09-26

### GPT解析

### 总结

这篇论文综述了类别发现(Category Discovery, CD)这一新兴的开世界学习任务，提供了全面的文献回顾、详细的方法分析和深入的讨论。

### 背景

类别发现是一个新兴的开世界学习任务，旨在给定一些已标记的已知类别数据的情况下，自动对包含未知类别实例的无标记数据进行分类。这个任务近年来受到了显著关注，并产生了丰富的文献。

### 目的

提供对类别发现文献的全面回顾，对不同方法提供详细分析和深入讨论，并指出未来研究方向。

### 方法

1) 引入文献分类法，考虑新颖类别发现(NCD)和广义类别发现(GCD)两个基本设置，以及持续类别发现、倾斜数据分布、联邦类别发现等派生设置；2) 对每种设置的方法进行分析，包括表示学习、标签分配和类别数量估计三个组成部分；3) 对所有方法进行基准测试；4) 讨论关键见解并指出未来研究方向。

### 主要发现

大规模预训练骨干网络、层次化和辅助线索、课程式训练都有利于类别发现；但在标签分配设计、类别数量估计以及扩展到复杂多目标场景方面仍存在挑战。

### 结论

讨论了文献中的关键见解，指出了有前途的未来研究方向，并提供了类别发现文献的动态调查页面(https://github.com/Visual-AI/Category-Discovery)。

### 翻译

类别发现(CD)是一个新兴的开世界学习任务，旨在给定一些来自已知类别的标记数据的情况下，自动对包含未知类别实例的无标记数据进行分类。多年来，这个任务引起了广泛关注，并产生了大量从不同角度尝试解决该问题的文献。在本综述中，我们对文献进行了全面回顾，并对不同方法提供了详细分析和深入讨论。首先，我们通过考虑两个基本设置(即新颖类别发现(NCD)和广义类别发现(GCD))以及为应对不同实际应用场景中的额外挑战而设计的几个派生设置(包括持续类别发现、倾斜数据分布、联邦类别发现等)，为文献引入了一个分类法。其次，对每种设置，我们提供了包含三个基本组成部分(表示学习、标签分配和类别数量估计)的方法详细分析。第三，我们对所有方法进行了基准测试并提炼了关键见解，表明大规模预训练骨干网络、层次化和辅助线索以及课程式训练都有利于类别发现，但在标签分配设计、类别数量估计以及扩展到复杂多目标场景方面仍存在挑战。最后，我们讨论了迄今为止文献中的关键见解，并指出了有前途的未来研究方向。我们在https://github.com/Visual-AI/Category-Discovery上整理了类别发现文献的动态调查。


### 论文摘要

Category discovery (CD) is an emerging open-world learning task, which aims at automatically categorizing unlabelled data containing instances from unseen classes, given some labelled data from seen classes. This task has attracted significant attention over the years and leads to a rich body of literature trying to address the problem from different perspectives. In this survey, we provide a comprehensive review of the literature, and offer detailed analysis and in-depth discussion on different methods. Firstly, we introduce a taxonomy for the literature by considering two base settings, namely novel category discovery (NCD) and generalized category discovery (GCD), and several derived settings that are designed to address the extra challenges in different real-world application scenarios, including continual category discovery, skewed data distribution, federated category discovery, etc. Secondly, for each setting, we offer a detailed analysis of the methods encompassing three fundamental components, representation learning, label assignment, and estimation of class number. Thirdly, we benchmark all the methods and distill key insights showing that large-scale pretrained backbones, hierarchical and auxiliary cues, and curriculum-style training are all beneficial for category discovery, while challenges remain in the design of label assignment, the estimation of class numbers, and scaling to complex multi-object scenarios.Finally, we discuss the key insights from the literature so far and point out promising future research directions. We compile a living survey of the category discovery literature at \href{https://github.com/Visual-AI/Category-Discovery}{https://github.com/Visual-AI/Category-Discovery}.

---

## 40. We Think, Therefore We Align LLMs to Helpful, Harmless and Honest Before They Go Wrong

**论文链接:** [http://arxiv.org/abs/2509.22510v1](http://arxiv.org/abs/2509.22510v1)

**作者:** Gautam Siddharth Kashyap, Mark Dras, Usman Naseem

**发布时间:** 2025-09-26

### GPT解析

### 总结

论文提出了自适应多分支转向(AMBS)方法，解决了大型语言模型多目标对齐中的灾难性遗忘和推理碎片化问题。

### 背景

大型语言模型在有用性、无害性和诚实性(HHH)目标上的对齐对安全可靠部署至关重要。先前的一对一转向向量方法会导致灾难性遗忘，而一对多方法虽然缓解了这一问题，但可能造成推理碎片化。

### 目的

开发一个统一且高效的多目标对齐框架，能够在保持跨目标一致性的同时实现目标特定控制。

### 方法

AMBS是一个两阶段的一对多框架：第一阶段计算Transformer层的注意力后隐藏状态形成共享表示；第二阶段将该表示克隆到并行分支中，通过策略引用机制进行目标特定控制。

### 主要发现

在Alpaca、BeaverTails和TruthfulQA上的评估显示，AMBS在多个7B LLM主干上显著改善了HHH对齐。在DeepSeek-7B上，与简单1-to-N基线相比，平均对齐分数提高32.4%，不安全输出减少11.0%，同时保持与最先进方法的竞争力。

### 结论

AMBS有效解决了多目标对齐中的关键挑战，为大型语言模型的安全可靠部署提供了新思路。

### 翻译

大型语言模型(LLMs)在多个目标上的对齐-有用性、无害性和诚实性(HHH)-对于安全可靠的部署至关重要。先前工作使用转向向量-注入到隐藏状态中的小型控制信号-来引导LLM输出，通常通过一对一(1-to-1)Transformer解码器。在这种设置下，优化单一对齐目标可能会无意中覆盖为其他目标学习的表示，导致灾难性遗忘。更新的方法通过一对多(1-to-N)Transformer解码器扩展转向向量。虽然这缓解了灾难性遗忘，但简单的多分支设计独立优化每个目标，可能导致推理碎片化-跨HHH目标的输出可能变得不一致。我们提出了自适应多分支转向(AMBS)，这是一个两阶段的一对多框架，用于统一和高效的多目标对齐。在第一阶段，计算Transformer层的注意力后隐藏状态一次形成共享表示。在第二阶段，将该表示克隆到并行分支中，并通过策略引用机制进行转向，实现目标特定控制同时保持跨目标一致性。在Alpaca、BeaverTails和TruthfulQA上的经验评估表明，AMBS在多个7B LLM主干上持续改进HHH对齐。例如，在DeepSeek-7B上，与简单1-to-N基线相比，AMBS将平均对齐分数提高32.4%，将不安全输出减少11.0%，同时保持与最先进方法的竞争力。


### 论文摘要

Alignment of Large Language Models (LLMs) along multiple objectives-helpfulness, harmlessness, and honesty (HHH)-is critical for safe and reliable deployment. Prior work has used steering vector-small control signals injected into hidden states-to guide LLM outputs, typically via one-to-one (1-to-1) Transformer decoders. In this setting, optimizing a single alignment objective can inadvertently overwrite representations learned for other objectives, leading to catastrophic forgetting. More recent approaches extend steering vectors via one-to-many (1-to-N) Transformer decoders. While this alleviates catastrophic forgetting, naive multi-branch designs optimize each objective independently, which can cause inference fragmentation-outputs across HHH objectives may become inconsistent. We propose Adaptive Multi-Branch Steering (AMBS), a two-stage 1-to-N framework for unified and efficient multi-objective alignment. In Stage I, post-attention hidden states of the Transformer layer are computed once to form a shared representation. In Stage II, this representation is cloned into parallel branches and steered via a policy-reference mechanism, enabling objective-specific control while maintaining cross-objective consistency. Empirical evaluations on Alpaca, BeaverTails, and TruthfulQA show that AMBS consistently improves HHH alignment across multiple 7B LLM backbones. For example, on DeepSeek-7B, AMBS improves average alignment scores by +32.4% and reduces unsafe outputs by 11.0% compared to a naive 1-to-N baseline, while remaining competitive with state-of-the-art methods.

---

## 41. PSTTS: A Plug-and-Play Token Selector for Efficient Event-based Spatio-temporal Representation Learning

**论文链接:** [http://arxiv.org/abs/2509.22481v1](http://arxiv.org/abs/2509.22481v1)

**作者:** Xiangmo Zhao, Nan Yang, Yang Wang, Zhanwen Liu

**发布时间:** 2025-09-26

### GPT解析

### 总结

本文提出了一种渐进式时空token选择(PSTTS)方法，用于解决事件数据中的时空冗余问题，无需额外参数即可显著提高计算效率

### 背景

主流事件时空表征学习方法将事件流转换为事件帧序列处理，但忽略了事件帧序列中固有的高空间稀疏性和帧间运动冗余性，导致计算开销大

### 目的

开发一种即插即用模块，有效识别并丢弃时空冗余token，在保持任务精度的同时提高计算效率

### 方法

提出渐进式时空token选择(PSTTS)方法，包含空间token净化和时间token选择两个阶段，前者评估事件帧内事件的时空一致性以丢弃噪声，后者评估相邻帧运动模式相似性以移除冗余时间信息

### 主要发现

PSTTS在保持任务准确性的同时，在DailyDVS-200数据集上减少了29-43.6%的计算量(FLOPs)，提高了21.6-41.3%的处理速度(FPS)

### 结论

PSTTS是一种有效的即插即用模块，可应用于多种骨干网络，显著提升事件数据处理效率

### 翻译

主流事件时空表征学习方法通常通过将事件流转换为事件帧序列来处理，取得显著性能。然而，它们忽略了事件帧序列中固有的高空间稀疏性和帧间运动冗余性，导致大量计算开销。现有的RGB视频token稀疏化方法依赖于不可靠的中间token表示，忽略了事件噪声的影响，因此不能直接应用于事件数据。本文提出渐进式时空token选择(PSTTS)，一种无需引入额外参数的事件数据即插即用模块。PSTTS利用原始事件数据中嵌入的时空分布特征，有效识别并丢弃时空冗余token，实现精度和效率之间的最佳平衡。具体而言，PSTTS包含两个阶段：空间token净化和时间token选择。空间token净化通过评估每个事件帧内事件的时空一致性，丢弃噪声和非事件区域，防止干扰后续的冗余性评估。时间token选择评估相邻事件帧之间的运动模式相似性，精确识别并移除冗余时间信息。我们将PSTTS应用于四个代表性骨干网络UniformerV2、VideoSwin、EVMamba和ExACT，在HARDVS、DailyDVS-200和SeACT数据集上进行测试。实验结果表明PSTTS实现了显著的效率提升。具体而言，在DailyDVS-200数据集上，PSTTS减少了29-43.6%的FLOPs，提高了21.6-41.3%的FPS，同时保持了任务准确性。我们的代码将会公开。


### 论文摘要

Mainstream event-based spatio-temporal representation learning methods typically process event streams by converting them into sequences of event frames, achieving remarkable performance. However, they neglect the high spatial sparsity and inter-frame motion redundancy inherent in event frame sequences, leading to significant computational overhead. Existing token sparsification methods for RGB videos rely on unreliable intermediate token representations and neglect the influence of event noise, making them ineffective for direct application to event data. In this paper, we propose Progressive Spatio-Temporal Token Selection (PSTTS), a Plug-and-Play module for event data without introducing any additional parameters. PSTTS exploits the spatio-temporal distribution characteristics embedded in raw event data to effectively identify and discard spatio-temporal redundant tokens, achieving an optimal trade-off between accuracy and efficiency. Specifically, PSTTS consists of two stages, Spatial Token Purification and Temporal Token Selection. Spatial Token Purification discards noise and non-event regions by assessing the spatio-temporal consistency of events within each event frame to prevent interference with subsequent temporal redundancy evaluation. Temporal Token Selection evaluates the motion pattern similarity between adjacent event frames, precisely identifying and removing redundant temporal information. We apply PSTTS to four representative backbones UniformerV2, VideoSwin, EVMamba, and ExACT on the HARDVS, DailyDVS-200, and SeACT datasets. Experimental results demonstrate that PSTTS achieves significant efficiency improvements. Specifically, PSTTS reduces FLOPs by 29-43.6% and increases FPS by 21.6-41.3% on the DailyDVS-200 dataset, while maintaining task accuracy. Our code will be available.

---

## 42. Learning the Neighborhood: Contrast-Free Multimodal Self-Supervised Molecular Graph Pretraining

**论文链接:** [http://arxiv.org/abs/2509.22468v1](http://arxiv.org/abs/2509.22468v1)

**作者:** Boshra Ariguib, Mathias Niepert, Andrei Manolache

**发布时间:** 2025-09-26

### GPT解析

### 总结

C-FREE是一种新的分子表示学习框架，通过整合2D图和3D构象信息，在MoleculeNet上取得了最先进的结果，证明3D信息对高质量分子表示至关重要。

### 背景

高质量的分子表示对属性预测和分子设计至关重要，但大型标记数据集稀缺。现有自监督方法依赖手工增强或复杂生成目标，且通常只使用2D拓扑，忽视了3D结构信息。

### 目的

开发一种简单框架，能够同时利用2D拓扑和3D结构信息，提高分子表示学习质量，解决现有方法的局限性。

### 方法

提出C-FREE（基于ego-net的无对比表示学习）框架，集成2D图和3D构象集合，通过预测子图嵌入学习分子表示，使用固定半径的ego-net作为建模单元，结合几何和拓扑信息，采用混合GNN-Transformer骨干网络，无需负样本、位置编码或昂贵的预处理。

### 主要发现

在GEOM数据集上预训练后，C-FREE在MoleculeNet上超越对比学习、生成和其他多模态自监督方法，取得最先进结果。在不同大小和分子类型的数据集上微调，证明预训练能有效迁移到新的化学领域。

### 结论

3D信息感知的分子表示对高质量分子表示学习至关重要，C-FREE框架有效解决了现有方法的局限性，为分子属性预测和设计提供了更好的表示。

### 翻译

高质量的分子表示对于属性预测和分子设计至关重要，然而大型标记数据集仍然稀缺。虽然基于分子图的自监督预训练已显示出潜力，但许多现有方法要么依赖手工增强或复杂的生成目标，要么仅依赖2D拓扑，导致有价值的3D结构信息未被充分利用。为解决这一差距，我们引入了C-FREE（基于ego-net的无对比表示学习），这是一个整合2D图和3D构象集合的简单框架。C-FREE通过在潜在空间中预测子图嵌入来学习分子表示，使用不同构象中的固定半径ego-net作为建模单元。这种设计使我们能够在混合图神经网络(GNN)-Transformer骨干网络中整合几何和拓扑信息，无需负样本、位置编码或昂贵的预处理。在提供丰富3D构象多样性的GEOM数据集上进行预训练，C-FREE在MoleculeNet上取得了最先进的结果，超越了对比学习、生成和其他多模态自监督方法。在不同大小和分子类型的数据集上进行微调进一步证明，预训练能有效迁移到新的化学领域，突显了3D信息感知的分子表示的重要性。


### 论文摘要

High-quality molecular representations are essential for property prediction and molecular design, yet large labeled datasets remain scarce. While self-supervised pretraining on molecular graphs has shown promise, many existing approaches either depend on hand-crafted augmentations or complex generative objectives, and often rely solely on 2D topology, leaving valuable 3D structural information underutilized. To address this gap, we introduce C-FREE (Contrast-Free Representation learning on Ego-nets), a simple framework that integrates 2D graphs with ensembles of 3D conformers. C-FREE learns molecular representations by predicting subgraph embeddings from their complementary neighborhoods in the latent space, using fixed-radius ego-nets as modeling units across different conformers. This design allows us to integrate both geometric and topological information within a hybrid Graph Neural Network (GNN)-Transformer backbone, without negatives, positional encodings, or expensive pre-processing. Pretraining on the GEOM dataset, which provides rich 3D conformational diversity, C-FREE achieves state-of-the-art results on MoleculeNet, surpassing contrastive, generative, and other multimodal self-supervised methods. Fine-tuning across datasets with diverse sizes and molecule types further demonstrates that pretraining transfers effectively to new chemical domains, highlighting the importance of 3D-informed molecular representations.

---

## 43. FreqDebias: Towards Generalizable Deepfake Detection via Consistency-Driven Frequency Debiasing

**论文链接:** [http://arxiv.org/abs/2509.22412v1](http://arxiv.org/abs/2509.22412v1)

**作者:** Hossein Kashiani, Niloufar Alipour Talemi, Fatemeh Afghah

**发布时间:** 2025-09-26

**备注:** Accepted to the IEEE/CVF Conference on Computer Vision and Pattern  Recognition (CVPR 2025)

### GPT解析

### 总结

本文提出了一种名为FreqDebias的频率去偏差框架，用于解决Deepfake检测器在面对新型伪造类型时的泛化能力受限问题。

### 背景

Deepfake检测器通常难以泛化到新型伪造类型，这是由于从有限训练数据中学习到的偏差导致的。

### 目的

识别并解决频域中的一种新型模型偏差（频谱偏差），该偏差导致检测器过度依赖特定频带，限制了其对未见伪造类型的泛化能力。

### 方法

FreqDebias框架采用两种互补策略：1）引入伪造混合增强（Fo-Mixup），动态增加训练样本的频率特性多样性；2）结合双一致性正则化（CR），使用类激活图强制局部一致性，并通过冯·米塞斯-费舍尔分布在超球嵌入空间上强制全局一致性。

### 主要发现

实验证明FreqDebias显著增强了跨域泛化能力，并在跨域和域内设置中都优于最先进的方法。

### 结论

通过减轻频谱偏差，FreqDebias框架提高了Deepfake检测器对新型伪造类型的泛化能力。

### 翻译

摘要翻译：Deepfake检测器通常难以泛化到新型伪造类型，这是由于从有限训练数据中学习到的偏差导致的。在本文中，我们识别了频域中的一种新型模型偏差，称为频谱偏差，其中检测器过度依赖特定频带，限制了它们对未见伪造类型的泛化能力。为此，我们提出了FreqDebias，一个频率去偏差框架，通过两种互补策略减轻频谱偏差。首先，我们引入了一种新颖的伪造混合增强（Fo-Mixup），动态增加训练样本的频率特性多样性。其次，我们结合了双一致性正则化（CR），使用类激活图（CAMs）强制局部一致性，并通过超球嵌入空间上的冯·米塞斯-费舍尔（vMF）分布强制全局一致性。这种双CR通过在局部和全局监督下促进一致的表征学习，减轻了对特定频率成分的过度依赖。大量实验表明，FreqDebias显著增强了跨域泛化能力，并在跨域和域内设置中都优于最先进的方法。


### 论文摘要

Deepfake detectors often struggle to generalize to novel forgery types due to biases learned from limited training data. In this paper, we identify a new type of model bias in the frequency domain, termed spectral bias, where detectors overly rely on specific frequency bands, restricting their ability to generalize across unseen forgeries. To address this, we propose FreqDebias, a frequency debiasing framework that mitigates spectral bias through two complementary strategies. First, we introduce a novel Forgery Mixup (Fo-Mixup) augmentation, which dynamically diversifies frequency characteristics of training samples. Second, we incorporate a dual consistency regularization (CR), which enforces both local consistency using class activation maps (CAMs) and global consistency through a von Mises-Fisher (vMF) distribution on a hyperspherical embedding space. This dual CR mitigates over-reliance on certain frequency components by promoting consistent representation learning under both local and global supervision. Extensive experiments show that FreqDebias significantly enhances cross-domain generalization and outperforms state-of-the-art methods in both cross-domain and in-domain settings.

---

## 44. Pushing Toward the Simplex Vertices: A Simple Remedy for Code Collapse in Smoothed Vector Quantization

**论文链接:** [http://arxiv.org/abs/2509.22161v1](http://arxiv.org/abs/2509.22161v1)

**作者:** Takashi Morita

**发布时间:** 2025-09-26

### GPT解析

### 总结

本文提出了一种新的平滑向量量化方法，通过简单直观的正则化技术同时满足量化器接近one-hot向量和充分利用所有码本条目的需求，在图像自编码和语音表示学习等任务中取得了优于先前方法的性能。

### 背景

向量量化将连续向量空间离散化为有限个代表性向量(码本)，在现代机器学习中广泛应用，但其非可微的量化步骤阻碍了梯度反向传播。

### 目的

解决向量量化中的非可微问题，同时确保平滑量化器接近one-hot向量并充分利用所有码本条目，防止码本坍塌。

### 方法

引入一种简单直观的正则化方法，通过最小化每个单纯形顶点与其K个最近平滑量化器之间的距离，同时满足两个关键特性：量化器接近one-hot向量和所有码本条目被充分利用。

### 主要发现

在离散图像自编码和对比语音表示学习等代表性基准测试中，所提出的方法实现了更可靠的码本利用并提高了性能。

### 结论

该正则化方法能够同时解决向量量化中的两个关键挑战，为平滑向量量化提供了有效解决方案。

### 翻译

向量量化是将连续向量空间离散化为有限个代表性向量(码本)的方法，在现代机器学习中已被广泛采用。尽管其有效，但向量量化存在一个基本挑战：非可微的量化步骤阻碍了梯度反向传播。平滑向量量化通过将码本向量的硬分配放松为码本条目的加权和来解决这个问题，表示为单纯形向量与码本的矩阵乘积。有效平滑需要两个特性：(1)平滑量化器应接近one-hot向量，确保紧密近似；(2)应使用所有码本条目，防止码本坍塌。现有方法通常分别处理这些需求。相比之下，本研究引入了一种简单直观的正则化方法，通过最小化每个单纯形顶点与其K个最近平滑量化器之间的距离，同时促进这两个特性。在离散图像自编码和对比语音表示学习等代表性基准测试中的实验表明，与先前方法相比，所提出的方法实现了更可靠的码本利用并提高了性能。


### 论文摘要

Vector quantization, which discretizes a continuous vector space into a finite set of representative vectors (a codebook), has been widely adopted in modern machine learning. Despite its effectiveness, vector quantization poses a fundamental challenge: the non-differentiable quantization step blocks gradient backpropagation. Smoothed vector quantization addresses this issue by relaxing the hard assignment of a codebook vector into a weighted combination of codebook entries, represented as the matrix product of a simplex vector and the codebook. Effective smoothing requires two properties: (1) smoothed quantizers should remain close to a onehot vector, ensuring tight approximation, and (2) all codebook entries should be utilized, preventing code collapse. Existing methods typically address these desiderata separately. By contrast, the present study introduces a simple and intuitive regularization that promotes both simultaneously by minimizing the distance between each simplex vertex and its $K$-nearest smoothed quantizers. Experiments on representative benchmarks, including discrete image autoencoding and contrastive speech representation learning, demonstrate that the proposed method achieves more reliable codebook utilization and improves performance compared to prior approaches.

---

## 45. Mind the Missing: Variable-Aware Representation Learning for Irregular EHR Time Series using Large Language Models

**论文链接:** [http://arxiv.org/abs/2509.22121v1](http://arxiv.org/abs/2509.22121v1)

**作者:** Jeong Eul Kwon, Joo Heung Yoon, Hyo Kyung Lee

**发布时间:** 2025-09-26

### GPT解析

### 总结

VITAL是一种基于大型语言模型的变量感知框架，专门用于从不规则采样的生理时间序列中学习。它通过区分生命体征和实验室检查的不同特性，并采用相应的处理方法，有效解决了不规则采样和高缺失率的挑战。在基准测试中，VITAL表现优于现有方法，并且在处理高缺失率数据时保持稳健。

### 背景

从电子健康记录（EHRs）导出的时间序列数据面临不规则采样和高缺失率的挑战。临床变量根据工作流程和干预时间以不均匀的时间间隔进行测量，这使得传统的时间序列分析方法难以有效应用。

### 目的

提出一种名为VITAL的框架，用于从不规则采样的生理时间序列中有效学习，并解决临床数据中常见的高缺失率问题。

### 方法

VITAL框架区分两种临床变量：1）生命体征，将其重新编程到语言空间，使LLM能够捕获时间上下文并推理缺失值；2）实验室检查，根据可用性使用代表性汇总值或可学习的'未测量'令牌进行嵌入。这种方法使模型能够处理不同特性的临床变量。

### 主要发现

在PhysioNet的基准数据集上，VITAL优于为不规则时间序列设计的最先进方法。此外，在高缺失率情况下（现实临床场景中常见），VITAL保持稳健的性能，即使关键变量不可用。

### 结论

VITAL框架能够有效处理来自电子健康记录的不规则时间序列数据，特别是在高缺失率情况下表现优异，为临床数据分析提供了新的解决方案。

### 翻译

不规则采样和高缺失率是从电子健康记录（EHRs）导出的时间序列建模中的内在挑战，其中临床变量根据工作流程和干预时间以不均匀间隔进行测量。为解决这一问题，我们提出了VITAL，这是一种变量感知的基于大型语言模型（LLM）的框架，专为从不规则采样的生理时间序列中学习而设计。VITAL区分两种不同类型的临床变量：生命体征，它们被频繁记录并表现出时间模式；以及实验室检查，它们被零星测量且缺乏时间结构。它将生命体征重新编程到语言空间，使LLM能够通过显式编码捕获时间上下文并对缺失值进行推理。相比之下，实验室变量根据其可用性，使用代表性汇总值或可学习的'未测量'令牌进行嵌入。在PhysioNet的基准数据集上进行的大量评估表明，VITAL优于为不规则时间序列设计的最先进方法。此外，在高缺失率情况下（现实临床场景中常见，关键变量通常不可用），它保持稳健的性能。


### 论文摘要

Irregular sampling and high missingness are intrinsic challenges in modeling time series derived from electronic health records (EHRs),where clinical variables are measured at uneven intervals depending on workflow and intervention timing. To address this, we propose VITAL, a variable-aware, large language model (LLM) based framework tailored for learning from irregularly sampled physiological time series. VITAL differentiates between two distinct types of clinical variables: vital signs, which are frequently recorded and exhibit temporal patterns, and laboratory tests, which are measured sporadically and lack temporal structure. It reprograms vital signs into the language space, enabling the LLM to capture temporal context and reason over missing values through explicit encoding. In contrast, laboratory variables are embedded either using representative summary values or a learnable [Not measured] token, depending on their availability. Extensive evaluations on the benchmark datasets from the PhysioNet demonstrate that VITAL outperforms state of the art methods designed for irregular time series. Furthermore, it maintains robust performance under high levels of missingness, which is prevalent in real world clinical scenarios where key variables are often unavailable.

---

## 46. BrainPro: Towards Large-scale Brain State-aware EEG Representation Learning

**论文链接:** [http://arxiv.org/abs/2509.22050v1](http://arxiv.org/abs/2509.22050v1)

**作者:** Yi Ding, Muyun Jiang, Weibang Jiang, Shuailei Zhang, Xinliang Zhou, Chenyu Liu, Shanglin Li, Yong Li, Cuntai Guan

**发布时间:** 2025-09-26

**备注:** 26 pages, 9 figures

### GPT解析

### 总结

BrainPro是一种创新的EEG基础模型，通过引入空间学习和状态解耦机制，解决了现有模型在捕捉通道间和区域间相互作用以及学习状态感知表示方面的挑战，在多个数据集上表现出色，具有广泛的适用性。

### 背景

EEG是一种无创技术，用于记录大脑电活动，广泛应用于脑机接口和医疗保健领域。近期基于大规模数据集训练的EEG基础模型相比传统解码方法表现出更好的性能和泛化能力，但仍存在显著挑战。

### 目的

解决现有EEG基础模型无法明确捕捉通道间和区域间相互作用、难以适应不同电极布局、以及很少学习状态感知表示的问题，开发一种能够灵活适应多样化任务和硬件设置的大型EEG模型。

### 方法

提出BrainPro模型，引入基于检索的空间学习块以灵活捕捉不同电极布局下的通道和区域级相互作用，同时引入大脑状态解耦块，通过具有解耦和区域感知重建损失的并行编码器实现状态感知的表示学习，并在大规模EEG语料库上进行预训练。

### 主要发现

BrainPro在九个公共BCI数据集上实现了最先进的性能和强大的泛化能力，证明了该模型在处理多样化EEG数据时的有效性。

### 结论

BrainPro通过创新的空间学习和状态解耦机制，成功解决了现有EEG基础模型的局限性，能够无缝适应多样化的任务和硬件设置，代码和预训练权重将公开发布。

### 翻译

脑电图是一种记录大脑电活动的无创技术，广泛应用于脑机接口和医疗保健领域。最近在大型数据集上训练的EEG基础模型相比传统解码方法显示出改进的性能和泛化能力，但仍然存在重大挑战。现有模型通常无法明确捕捉通道间和区域间的相互作用，这些是EEG信号中固有的关键信息源。由于不同数据集的通道配置不同，现有模型要么用自注意力近似空间结构，要么将训练限制在有限的通用通道集合，牺牲了灵活性和有效性。此外，尽管EEG数据集反映了情绪、运动等多种大脑状态，但当前模型在自监督预训练过程中很少学习状态感知的表示。为解决这些差距，我们提出BrainPro，一种大型EEG模型，引入了基于检索的空间学习块，以灵活捕捉不同电极布局下的通道和区域级相互作用，以及大脑状态解耦块，通过具有解耦和区域感知重建损失的并行编码器实现状态感知的表示学习。这种设计使BrainPro能够无缝适应多样化的任务和硬件设置。在大型EEG语料库上预训练后，BrainPro在九个公共BCI数据集上实现了最先进的性能和强大的泛化能力。我们的代码和预训练权重将公开发布。


### 论文摘要

Electroencephalography (EEG) is a non-invasive technique for recording brain electrical activity, widely used in brain-computer interface (BCI) and healthcare. Recent EEG foundation models trained on large-scale datasets have shown improved performance and generalizability over traditional decoding methods, yet significant challenges remain. Existing models often fail to explicitly capture channel-to-channel and region-to-region interactions, which are critical sources of information inherently encoded in EEG signals. Due to varying channel configurations across datasets, they either approximate spatial structure with self-attention or restrict training to a limited set of common channels, sacrificing flexibility and effectiveness. Moreover, although EEG datasets reflect diverse brain states such as emotion, motor, and others, current models rarely learn state-aware representations during self-supervised pre-training. To address these gaps, we propose BrainPro, a large EEG model that introduces a retrieval-based spatial learning block to flexibly capture channel- and region-level interactions across varying electrode layouts, and a brain state-decoupling block that enables state-aware representation learning through parallel encoders with decoupling and region-aware reconstruction losses. This design allows BrainPro to adapt seamlessly to diverse tasks and hardware settings. Pre-trained on an extensive EEG corpus, BrainPro achieves state-of-the-art performance and robust generalization across nine public BCI datasets. Our codes and the pre-trained weights will be released.

---

## 47. 论文ID: 2509.21965v1

**错误:** 无法读取论文信息 - [Errno 2] No such file or directory: 'storage/result/2509.21965v1.json'

---

## 48. TRACE: Learning to Compute on Graphs

**论文链接:** [http://arxiv.org/abs/2509.21886v1](http://arxiv.org/abs/2509.21886v1)

**作者:** Ziyang Zheng, Jiaying Zhu, Jingyi Zhou, Qiang Xu

**发布时间:** 2025-09-26

### GPT解析

### 总结

TRACE是一种新的图计算学习范式，通过分层Transformer架构和函数位移学习目标，解决了主流消息传递神经网络在计算图建模中的架构不匹配问题

### 背景

图表示学习中学习计算（建模计算图的功能行为）是一个基本挑战，但主流架构与这个任务不匹配

### 目的

解决主流消息传递神经网络和基于Transformer的方法在捕捉计算位置感知和层次性质方面的局限性

### 方法

TRACE采用分层Transformer镜像计算的逐步流动，取代排列不变聚合；引入函数位移学习，将学习问题解耦，只预测函数位移而非复杂全局函数

### 主要发现

在电子电路这一复杂且经济重要的计算图类别上，TRACE在全面基准测试中明显优于所有先前架构

### 结论

架构对齐骨干和解耦学习目标形成了一个更强大的范式，用于解决图上学习计算的基本挑战

### 翻译

学习计算，即建模计算图功能行为的能力，是图表示学习的一个基本挑战。然而，主流范式在架构上与这个任务不匹配。这个有缺陷的假设是主流消息传递神经网络及其传统基于Transformer的对应方法的核心，阻碍了模型捕捉计算的位置感知和层次性质。为了解决这个问题，我们引入TRACE，这是一个基于架构合理的骨干和原则性学习目标的新范式。首先，TRACE采用分层Transformer，镜像计算的逐步流动，提供了一个忠实的架构骨干，取代了有缺陷的排列不变聚合。其次，我们引入函数位移学习，一个新颖的目标，将学习问题解耦。模型不是直接预测复杂全局函数，而是被训练为只预测函数位移，即真实全局函数与假设输入独立的简单局部近似之间的差异。我们在电子电路上验证了这个范式，电子电路是最复杂和经济上最重要的计算图类别之一。在全面的基准测试中，TRACE明显优于所有先前架构。这些结果表明，我们的架构对齐骨干和解耦学习目标形成了一个更强大的范式，用于解决图上学习计算的基本挑战。


### 论文摘要

Learning to compute, the ability to model the functional behavior of a computational graph, is a fundamental challenge for graph representation learning. Yet, the dominant paradigm is architecturally mismatched for this task. This flawed assumption, central to mainstream message passing neural networks (MPNNs) and their conventional Transformer-based counterparts, prevents models from capturing the position-aware, hierarchical nature of computation. To resolve this, we introduce \textbf{TRACE}, a new paradigm built on an architecturally sound backbone and a principled learning objective. First, TRACE employs a Hierarchical Transformer that mirrors the step-by-step flow of computation, providing a faithful architectural backbone that replaces the flawed permutation-invariant aggregation. Second, we introduce \textbf{function shift learning}, a novel objective that decouples the learning problem. Instead of predicting the complex global function directly, our model is trained to predict only the \textit{function shift}, the discrepancy between the true global function and a simple local approximation that assumes input independence. We validate this paradigm on electronic circuits, one of the most complex and economically critical classes of computational graphs. Across a comprehensive suite of benchmarks, TRACE substantially outperforms all prior architectures. These results demonstrate that our architecturally-aligned backbone and decoupled learning objective form a more robust paradigm for the fundamental challenge of learning to compute on graphs.

---

## 49. IndiSeek learns information-guided disentangled representations

**论文链接:** [http://arxiv.org/abs/2509.21584v1](http://arxiv.org/abs/2509.21584v1)

**作者:** Yu Gui, Cong Ma, Zongming Ma

**发布时间:** 2025-09-25

### GPT解析

### 总结

本研究提出了一种名为IndiSeek的新解耦表示学习方法，用于多模态学习中的解耦表示学习问题。该方法通过结合独立性强制目标和计算高效的重建损失，有效平衡了特征独立性和完整性，能够在单细胞多组学等应用中提取模态特定特征。

### 背景

解耦表示学习是多模态学习的基本任务。在现代应用如单细胞多组学中，共享特征和模态特定特征对表征细胞状态和支持下游分析都至关重要。然而，基于互信息的目标难以可靠估计，其变分替代在实践中表现不佳。

### 目的

解决多模态学习中解耦表示学习的挑战，特别是如何使模态特定特征独立于共享特征同时捕获每个模态内的互补信息，以及如何平衡特征独立性和完整性。

### 方法

作者提出了IndiSeek方法，它结合了独立性强制目标和计算高效的重建损失，该损失限制了条件互信息。这种公式明确平衡了独立性和完整性，实现了模态特定特征的原则性提取。

### 主要发现

IndiSeek在合成模拟、CITE-seq数据集和多个真实世界多模态基准测试上表现出色，证明了其有效性和实用性。

### 结论

IndiSeek为多模态学习中的解耦表示学习提供了新的解决方案，能够有效提取模态特定特征，支持下游分析任务。

### 翻译

学习解耦表示是多模态学习中的一个基本任务。在现代应用中，如单细胞多组学，共享特征和模态特定特征对于表征细胞状态和支持下游分析都至关重要。理想情况下，模态特定特征应该独立于共享特征，同时捕获每个模态内的所有互补信息。这种权衡自然地通过信息论标准表达，但基于互信息的目标难以可靠估计，其变分替代在实践中通常表现不佳。在本文中，我们介绍了IndiSeek，这是一种新的解耦表示学习方法，它通过结合独立性强制目标和计算高效的重建损失来解决这一挑战，该损失限制了条件互信息。这种公式明确平衡了独立性和完整性，实现了模态特定特征的原则性提取。我们在合成模拟、CITE-seq数据集和多个真实世界多模态基准测试上证明了IndiSeek的有效性。


### 论文摘要

Learning disentangled representations is a fundamental task in multi-modal learning. In modern applications such as single-cell multi-omics, both shared and modality-specific features are critical for characterizing cell states and supporting downstream analyses. Ideally, modality-specific features should be independent of shared ones while also capturing all complementary information within each modality. This tradeoff is naturally expressed through information-theoretic criteria, but mutual-information-based objectives are difficult to estimate reliably, and their variational surrogates often underperform in practice. In this paper, we introduce IndiSeek, a novel disentangled representation learning approach that addresses this challenge by combining an independence-enforcing objective with a computationally efficient reconstruction loss that bounds conditional mutual information. This formulation explicitly balances independence and completeness, enabling principled extraction of modality-specific features. We demonstrate the effectiveness of IndiSeek on synthetic simulations, a CITE-seq dataset and multiple real-world multi-modal benchmarks.

---

## 50. Physics-informed GNN for medium-high voltage AC power flow with edge-aware attention and line search correction operator

**论文链接:** [http://arxiv.org/abs/2509.22458v1](http://arxiv.org/abs/2509.22458v1)

**作者:** Changhun Kim, Timon Conrad, Redwanul Karim, Julian Oelhaf, David Riebesel, Tomás Arias-Vergara, Andreas Maier, Johann Jäger, Siming Bayer

**发布时间:** 2025-09-26

**备注:** 5 pages, 2 figures. Submitted to ICASSP 2026. Code available at  https://github.com/Kimchangheon/PIGNN-Attn-LS

### GPT解析

### 总结

物理信息图神经网络（PIGNNs）作为交流潮流计算求解器，通过结合边感知注意力机制和基于回溯线搜索的校正算子，显著提高了计算精度和速度，在保持高速的同时超越了传统牛顿-拉夫逊方法的性能。

### 背景

物理信息图神经网络（PIGNNs）已发展为快速的交流潮流计算求解器，可替代经典牛顿-拉夫逊（NR）求解器，特别是在需要评估数千种场景的应用中。

### 目的

解决当前PIGNNs在保持速度的同时需要提高准确性问题，特别是物理损失在推理阶段不起作用可能阻碍其实际应用的问题。

### 方法

提出PIGNN-Attn-LS模型，结合边感知注意力机制通过每条边的偏置显式编码线路物理特性捕捉电网各向异性，以及基于回溯线搜索的全局化校正算子在推理阶段恢复有效下降准则。

### 主要发现

在4-32总线电网的高压案例测试中，PIGNN-Attn-LS实现了电压均方根误差0.00033 p.u.和角度误差0.08°，分别比PIGNN-MLP基线提高99.5%和87.1%；在4-1024总线网格上，批处理推理速度比牛顿-拉夫逊方法快2-5倍。

### 结论

PIGNN-Attn-LS通过结合物理信息与深度学习技术，显著提高了潮流计算的准确性和效率，为电网分析提供了一种快速且精确的替代方案。

### 翻译

物理信息图神经网络已成为快速的交流潮流计算求解器，可以替代经典的牛顿-拉夫逊求解器，特别是在需要评估数千种场景时。然而，当前的PIGNNs在保持速度的同时仍需提高准确性；特别是物理损失在推理阶段不起作用，这可能阻碍其实际应用。我们通过PIGNN-Attn-LS解决了这一问题，它结合了边感知注意力机制，通过每条边的偏置显式编码线路物理特性，捕捉电网的各向异性，以及基于回溯线搜索的全局化校正算子，在推理阶段恢复有效的下降准则。训练和测试使用真实的 高/中压场景生成器，仅使用NR构造参考状态。在包含4-32总线电网的保留高压案例中，PIGNN-Attn-LS实现了电压测试均方根误差0.00033 p.u.和角度误差0.08°，分别比PIGNN-MLP基线好99.5%和87.1%。使用流式微批处理，它在4-1024总线网格上的批处理推理速度比NR快2-5倍。


### 论文摘要

Physics-informed graph neural networks (PIGNNs) have emerged as fast AC power-flow solvers that can replace classic Newton--Raphson (NR) solvers, especially when thousands of scenarios must be evaluated. However, current PIGNNs still need accuracy improvements at parity speed; in particular, the physics loss is inoperative at inference, which can deter operational adoption. We address this with PIGNN-Attn-LS, combining an edge-aware attention mechanism that explicitly encodes line physics via per-edge biases, capturing the grid's anisotropy, with a backtracking line-search-based globalized correction operator that restores an operative decrease criterion at inference. Training and testing use a realistic High-/Medium-Voltage scenario generator, with NR used only to construct reference states. On held-out HV cases consisting of 4--32-bus grids, PIGNN-Attn-LS achieves a test RMSE of 0.00033 p.u. in voltage and 0.08$^\circ$ in angle, outperforming the PIGNN-MLP baseline by 99.5\% and 87.1\%, respectively. With streaming micro-batches, it delivers 2--5$\times$ faster batched inference than NR on 4--1024-bus grids.

---

## 51. SHAKE-GNN: Scalable Hierarchical Kirchhoff-Forest Graph Neural Network

**论文链接:** [http://arxiv.org/abs/2509.22100v1](http://arxiv.org/abs/2509.22100v1)

**作者:** Zhipu Cui, Johannes Lutzeyer

**发布时间:** 2025-09-26

### GPT解析

### 总结

研究提出了SHAKE-GNN，一种基于基尔霍夫森林层次结构的新型可扩展图级GNN框架，能够生成多尺度表示，在效率和性能之间提供灵活权衡，并在大规模图分类任务中取得具有竞争力的性能。

### 背景

图神经网络(GNNs)在各种学习任务中取得了显著成功，但将GNN扩展到大型图仍然是一个重大挑战，特别是对于图级任务。

### 目的

解决GNNs在大规模图上扩展的挑战，特别是针对图级任务设计一种可扩展的框架。

### 方法

引入SHAKE-GNN框架，基于基尔霍夫森林层次结构构建图的多分辨率随机分解；提出改进的、数据驱动的策略来选择权衡参数；分析SHAKE-GNN的时间复杂度。

### 主要发现

SHAKE-GNN能够生成多尺度表示，在多个大规模图分类基准测试中实现了具有竞争力的性能，同时提供了改进的可扩展性。

### 结论

SHAKE-GNN为解决GNN在大规模图上的扩展问题提供了一种有效方法，通过多尺度表示实现了效率和性能的良好平衡。

### 翻译

图神经网络(GNNs)在各种学习任务中取得了显著成功。然而，将GNN扩展到大型图仍然是一个重大挑战，特别是对于图级任务。在这项工作中，我们介绍了SHAKE-GNN，一种基于基尔霍夫森林层次结构的新型可扩展图级GNN框架，基尔霍夫森林是一类用于构建图多分辨率随机分解的随机生成森林。SHAKE-GNN生成多尺度表示，能够灵活地在效率和性能之间进行权衡。我们引入了一种改进的、数据驱动的策略来选择权衡参数，并分析了SHAKE-GNN的时间复杂度。在多个大规模图分类基准测试上的实验结果表明，SHAKE-GNN实现了具有竞争力的性能，同时提供了改进的可扩展性。


### 论文摘要

Graph Neural Networks (GNNs) have achieved remarkable success across a range of learning tasks. However, scaling GNNs to large graphs remains a significant challenge, especially for graph-level tasks. In this work, we introduce SHAKE-GNN, a novel scalable graph-level GNN framework based on a hierarchy of Kirchhoff Forests, a class of random spanning forests used to construct stochastic multi-resolution decompositions of graphs. SHAKE-GNN produces multi-scale representations, enabling flexible trade-offs between efficiency and performance. We introduce an improved, data-driven strategy for selecting the trade-off parameter and analyse the time-complexity of SHAKE-GNN. Experimental results on multiple large-scale graph classification benchmarks demonstrate that SHAKE-GNN achieves competitive performance while offering improved scalability.

---

## 52. Stable and Interpretable Jet Physics with IRC-Safe Equivariant Feature Extraction

**论文链接:** [http://arxiv.org/abs/2509.22059v1](http://arxiv.org/abs/2509.22059v1)

**作者:** Partha Konar, Vishal S. Ngairangbam, Michael Spannowsky, Deepanshu Srivastava

**发布时间:** 2025-09-26

**备注:** 30 pages, 3 tables, 7 figures

### GPT解析

### 总结

深度学习在喷注分类中取得成功，但理解模型学习内容和特征与已知QCD可观测量之间的关系仍具挑战。本研究通过图神经网络和物理学动机的归纳偏置提高可解释性，展示了嵌入对称性和安全性约束如何提高模型鲁棒性并使网络表示与已知QCD结构对应。

### 背景

深度学习在喷注分类任务中已取得显著成功，但理解这些模型学习的内容以及它们的特征如何与已知的量子色动力学(QCD)可观测量相关联仍然是一个关键挑战。提高可解释性对于在对撞机物理学中构建强大可靠的机器学习工具至关重要。

### 目的

研究图神经网络用于夸克-胶子鉴别，系统性地融入物理学动机的归纳偏置，以提高模型的可解释性和鲁棒性。

### 方法

设计强制执行红外和共线性(IRC)安全，以及在快度-方位角平面上的E(2)和O(2)等变性的消息传递架构。使用模拟的喷注数据集，将这些网络与无约束基线在分类性能、对软发射的鲁棒性和潜在表示结构方面进行比较。通过将能量流多项式回归到主要主成分上，建立学习表示与已建立的IRC安全喷注可观测量之间的对应关系。

### 主要发现

物理感知的网络在不同训练实例中更稳定，并将潜在方差分布在多个可解释的方向上。通过回归分析，建立了学习表示与已建立的IRC安全喷注可观测量之间的直接对应关系。

### 结论

嵌入对称性和安全性约束不仅提高了模型的鲁棒性，还将网络表示建立在已知的QCD结构上，为对撞机物理学中的可解释深度学习提供了原则性方法。

### 翻译

深度学习在喷注分类任务中已取得显著成功，但一个关键挑战仍然存在：理解这些模型学习的内容以及它们的特征如何与已知的量子色动力学(QCD)可观测量相关。提高可解释性对于在对撞机物理学中构建强大可靠的机器学习工具至关重要。为了应对这一挑战，我们研究了用于夸克-胶子鉴别的图神经网络，系统性地融入了物理学动机的归纳偏置。特别是，我们设计了强制执行红外和共线性(IRC)安全，以及在快度-方位角平面上的E(2)和O(2)等变性的消息传递架构。使用模拟的喷注数据集，我们在分类性能、对软发射的鲁棒性和潜在表示结构方面将这些网络与无约束基线进行了比较。我们的分析表明，物理感知的网络在训练实例中更稳定，并将它们的潜在方差分布在多个可解释的方向上。通过将能量流多项式回归到主要主成分上，我们建立了学习表示与已建立的IRC安全喷注可观测量之间的直接对应关系。这些结果表明，嵌入对称性和安全性约束不仅提高了鲁棒性，还将网络表示建立在已知的QCD结构上，为对撞机物理学中的可解释深度学习提供了原则性方法。


### 论文摘要

Deep learning has achieved remarkable success in jet classification tasks, yet a key challenge remains: understanding what these models learn and how their features relate to known QCD observables. Improving interpretability is essential for building robust and trustworthy machine learning tools in collider physics. To address this challenge, we investigate graph neural networks for quark-gluon discrimination, systematically incorporating physics-motivated inductive biases. In particular, we design message-passing architectures that enforce infrared and collinear (IRC) safety, as well as E(2) and O(2) equivariance in the rapidity-azimuth plane. Using simulated jet datasets, we compare these networks against unconstrained baselines in terms of classification performance, robustness to soft emissions, and latent representation structures. Our analysis shows that physics-aware networks are more stable across training instances and distribute their latent variance across multiple interpretable directions. By regressing Energy Flow Polynomials onto the leading principal components, we establish a direct correspondence between learned representations and established IRC-safe jet observables. These results demonstrate that embedding symmetry and safety constraints not only improves robustness but also grounds network representations in known QCD structures, providing a principled approach toward interpretable deep learning in collider physics.

---

## 53. MCGM: Multi-stage Clustered Global Modeling for Long-range Interactions in Molecules

**论文链接:** [http://arxiv.org/abs/2509.22028v1](http://arxiv.org/abs/2509.22028v1)

**作者:** Haodong Pan, Yusong Wang, Nanning Zheng, Caijui Jiang

**发布时间:** 2025-09-26

**备注:** 27 pages, 1 figures

### GPT解析

### 总结

该论文提出了多阶段聚类全局建模(MCGM)方法，解决了几何图神经网络在建模长距离相互作用时的局限性，实现了高效的全局信息捕获和特征增强。

### 背景

几何图神经网络(GNNs)在捕捉分子几何方面表现出色，但其局部偏向的消息传递阻碍了长距离相互作用的建模。当前解决方案存在计算成本高、系统特定性强、参数调复杂等问题。

### 目的

开发一种轻量级、即插即用的模块，使几何GNNs能够通过高效的聚类操作获得分层全局上下文，以更好地建模长距离相互作用。

### 方法

MCGM方法构建原子簇的多分辨率层次结构，通过动态层次聚类提炼全局信息，通过学习转换传播此上下文，并通过残差连接最终强化原子特征。该方法可无缝集成到不同的骨干架构中。

### 主要发现

MCGM将OE62能量预测误差平均降低26.2%；在AQM上实现了最先进的准确性(能量17.0 meV，力4.9 meV/Å)；同时使用比Neural P3M少20%的参数。

### 结论

MCGM是一种有效的方法，能够增强几何GNNs建模长距离相互作用的能力，同时保持计算效率，具有广泛的应用潜力。

### 翻译

几何图神经网络(GNNs)在捕捉分子几何方面表现出色，但其局部偏向的消息传递阻碍了长距离相互作用的建模。当前解决方案存在根本性限制：扩展截止半径会导致计算成本随距离立方增长；物理启发的核(如库仑、色散)通常是系统特定的，缺乏通用性；傅里叶空间方法需要仔细调整多个参数，并增加计算开销。我们引入了多阶段聚类全局建模(MCGM)，这是一种轻量级、即插即用的模块，通过高效的聚类操作赋予几何GNNs分层全局上下文。MCGM构建原子簇的多分辨率层次结构，通过动态层次聚类提炼全局信息，并通过学习转换传播此上下文，最终通过残差连接强化原子特征。无缝集成到四种不同的骨干架构中，MCGM将OE62能量预测误差平均降低26.2%。在AQM上，MCGM实现了最先进的准确性(能量17.0 meV，力4.9 meV/Å)，同时使用的参数比Neural P3M少20%。代码将在接受后公开。


### 论文摘要

Geometric graph neural networks (GNNs) excel at capturing molecular geometry, yet their locality-biased message passing hampers the modeling of long-range interactions. Current solutions have fundamental limitations: extending cutoff radii causes computational costs to scale cubically with distance; physics-inspired kernels (e.g., Coulomb, dispersion) are often system-specific and lack generality; Fourier-space methods require careful tuning of multiple parameters (e.g., mesh size, k-space cutoff) with added computational overhead. We introduce Multi-stage Clustered Global Modeling (MCGM), a lightweight, plug-and-play module that endows geometric GNNs with hierarchical global context through efficient clustering operations. MCGM builds a multi-resolution hierarchy of atomic clusters, distills global information via dynamic hierarchical clustering, and propagates this context back through learned transformations, ultimately reinforcing atomic features via residual connections. Seamlessly integrated into four diverse backbone architectures, MCGM reduces OE62 energy prediction error by an average of 26.2%. On AQM, MCGM achieves state-of-the-art accuracy (17.0 meV for energy, 4.9 meV/{\AA} for forces) while using 20% fewer parameters than Neural P3M. Code will be made available upon acceptance.

---

## 54. Uncovering Alzheimer's Disease Progression via SDE-based Spatio-Temporal Graph Deep Learning on Longitudinal Brain Networks

**论文链接:** [http://arxiv.org/abs/2509.21735v1](http://arxiv.org/abs/2509.21735v1)

**作者:** Houliang Zhou, Rong Zhou, Yangying Liu, Kanhao Zhao, Li Shen, Brian Y. Chen, Yu Zhang, Lifang He, Alzheimer's Disease Neuroimaging Initiative

**发布时间:** 2025-09-26

### GPT解析

### 总结

该研究开发了一种可解释的时空图神经网络框架，利用双随机微分方程对不规则采样的纵向功能磁共振成像数据进行建模，以预测阿尔茨海默病进展。该框架在两个独立队列上得到验证，能够识别与疾病进展相关的大脑回路异常，并发现新的生物标志物。

### 背景

识别能够预测阿尔茨海默病进展的神经影像生物标志物对于及时干预至关重要，但由于现有方法往往忽略了大脑网络时空特性的复杂功能障碍，这一任务仍然具有挑战性。

### 目的

开发一个可解释的时空图神经网络框架来预测未来的阿尔茨海默病进展。

### 方法

利用双随机微分方程对不规则采样的纵向功能磁共振成像数据进行建模，并在OASIS-3和ADNI两个独立队列上验证该方法。

### 主要发现

框架能够学习稀疏的区域和连接重要性概率，识别出海马旁皮层、前额叶皮层和顶小叶为显著区域，在腹侧注意、背侧注意和默认模式网络中存在显著紊乱；这些异常与AD临床症状有强相关性；解释性策略揭示了已知的和新的神经系统水平以及性别特异性的生物标志物。

### 结论

时空图学习方法在不规则采样的纵向成像数据背景下，对AD进展的早期、个体化预测具有潜力。

### 翻译

识别用于预测阿尔茨海默病进展的客观神经影像生物标志物对于及时干预至关重要。然而，由于潜在大脑网络时空特性的复杂功能障碍，这一任务仍然具有挑战性，现有方法往往忽视了这些特点。为解决这些局限性，我们开发了一个可解释的时空图神经网络框架来预测未来的AD进展，利用双随机微分方程对不规则采样的纵向功能磁共振成像数据进行建模。我们在两个独立队列上验证了我们的方法，包括开放获取影像研究系列(OASIS-3)和阿尔茨海默病神经影像倡议(ADNI)。我们的框架有效地学习稀疏的区域和连接重要性概率，能够识别与疾病进展相关的大脑回路异常。值得注意的是，我们检测到海马旁皮层、前额叶皮层和顶小叶为显著区域，在腹侧注意、背侧注意和默认模式网络中存在显著紊乱。这些异常与纵向AD相关临床症状有很强的相关性。此外，我们的解释性策略揭示了已知的和新的神经系统水平和性别特异性的生物标志物，为理解AD进展的神经生物学机制提供了新的见解。我们的研究结果强调了时空图学习方法在不规则采样纵向成像数据背景下对AD进展早期、个体化预测的潜力。


### 论文摘要

Identifying objective neuroimaging biomarkers to forecast Alzheimer's disease (AD) progression is crucial for timely intervention. However, this task remains challenging due to the complex dysfunctions in the spatio-temporal characteristics of underlying brain networks, which are often overlooked by existing methods. To address these limitations, we develop an interpretable spatio-temporal graph neural network framework to predict future AD progression, leveraging dual Stochastic Differential Equations (SDEs) to model the irregularly-sampled longitudinal functional magnetic resonance imaging (fMRI) data. We validate our approach on two independent cohorts, including the Open Access Series of Imaging Studies (OASIS-3) and the Alzheimer's Disease Neuroimaging Initiative (ADNI). Our framework effectively learns sparse regional and connective importance probabilities, enabling the identification of key brain circuit abnormalities associated with disease progression. Notably, we detect the parahippocampal cortex, prefrontal cortex, and parietal lobule as salient regions, with significant disruptions in the ventral attention, dorsal attention, and default mode networks. These abnormalities correlate strongly with longitudinal AD-related clinical symptoms. Moreover, our interpretability strategy reveals both established and novel neural systems-level and sex-specific biomarkers, offering new insights into the neurobiological mechanisms underlying AD progression. Our findings highlight the potential of spatio-temporal graph-based learning for early, individualized prediction of AD progression, even in the context of irregularly-sampled longitudinal imaging data.

---

## 55. Exact Subgraph Isomorphism Network for Predictive Graph Mining

**论文链接:** [http://arxiv.org/abs/2509.21699v1](http://arxiv.org/abs/2509.21699v1)

**作者:** Taiga Kojima, Masayuki Karasuyama

**发布时间:** 2025-09-25

### GPT解析

### 总结

论文提出了精确子图同构网络（EIN），结合子图枚举、神经网络和稀疏正则化，实现了高判别能力和可解释性的图级别预测。

### 背景

在图级别预测任务中，输入图的子图信息起着关键作用，构建具有高判别能力和可解释性的图级别预测模型仍然是一个挑战。

### 目的

提出一种能够实现高判别能力和可解释性的图级别预测模型。

### 方法

提出精确子图同构网络（EIN），结合精确子图枚举、神经网络和稀疏正则化，通过子图枚举和神经网络的组合提高判别能力，利用稀疏正则化实现有效剪枝和重要子图识别。

### 主要发现

EIN相比标准图神经网络模型具有足够高的预测性能，并能通过所选子图进行有效的事后分析。

### 结论

EIN通过结合子图枚举、神经网络和稀疏正则化，成功实现了高判别能力和可解释性的图级别预测。

### 翻译

在图级别预测任务（为给定图预测标签）中，输入图的子图信息起着关键作用。在本文中，我们提出了精确子图同构网络（EIN），它结合了精确子图枚举、神经网络和稀疏正则化。通常，构建具有高判别能力和可解释性的图级别预测模型仍然是一个具有挑战性的问题。我们的子图枚举和神经网络的组合有助于提高对输入图子图结构的高判别能力。此外，EIN中的稀疏正则化使我们能够：1)推导出一种有效的剪枝策略，减轻枚举的计算难度，同时保持预测性能；2)识别对高可解释性有贡献的重要子图。我们经验性地证明，与标准图神经网络模型相比，EIN具有足够高的预测性能，同时，我们还展示了基于所选子图的事后分析示例。


### 论文摘要

In the graph-level prediction task (predict a label for a given graph), the information contained in subgraphs of the input graph plays a key role. In this paper, we propose Exact subgraph Isomorphism Network (EIN), which combines the exact subgraph enumeration, neural network, and a sparse regularization. In general, building a graph-level prediction model achieving high discriminative ability along with interpretability is still a challenging problem. Our combination of the subgraph enumeration and neural network contributes to high discriminative ability about the subgraph structure of the input graph. Further, the sparse regularization in EIN enables us 1) to derive an effective pruning strategy that mitigates computational difficulty of the enumeration while maintaining the prediction performance, and 2) to identify important subgraphs that contributes to high interpretability. We empirically show that EIN has sufficiently high prediction performance compared with standard graph neural network models, and also, we show examples of post-hoc analysis based on the selected subgraphs.

---

## 56. Shoot from the HIP: Hessian Interatomic Potentials without derivatives

**论文链接:** [http://arxiv.org/abs/2509.21624v1](http://arxiv.org/abs/2509.21624v1)

**作者:** Andreas Burger, Luca Thiede, Nikolaj Rønne, Varinia Bernales, Nandita Vijaykumar, Tejs Vegge, Arghya Bhowmik, Alan Aspuru-Guzik

**发布时间:** 2025-09-25

**备注:** https://github.com/BurgerAndreas/hip

### GPT解析

### 总结

本文提出了一种名为HIP的深度学习方法，可以直接预测分子Hessian矩阵，无需依赖自动微分或有限差分，在计算化学任务中表现出显著优势。

### 背景

计算化学中的基本任务，如过渡态搜索和振动分析，依赖于分子Hessian矩阵（势能的二阶导数）。然而，传统方法计算成本高，且随着系统规模增大扩展性差，无论是量子力学方法还是神经网络方法都存在这一问题。

### 目的

展示可以直接通过深度学习模型预测Hessian矩阵，提高计算效率并解决扩展性问题。

### 方法

从图神经网络消息传递过程中计算出的不可约表示特征（最高到l=2度）构建SE(3)-等变、对称的Hessian矩阵。这种方法被称为HIP Hessians。

### 主要发现

HIP Hessians比传统方法快一到两个数量级，更准确，内存效率更高，更容易训练，且随系统规模增大具有更好的扩展性。在过渡态搜索、加速几何优化、零点能校正和振动分析基准测试中均表现出一致优越的性能。

### 结论

HIP方法为计算化学中的Hessian矩阵计算提供了高效解决方案，已开源代码库和模型权重以促进进一步发展。

### 翻译

计算化学中的基本任务，从过渡态搜索到振动分析，都依赖于分子Hessian矩阵，即势能的二阶导数。然而，Hessian矩阵的计算成本很高，并且随着系统规模增大而扩展性差，无论是量子力学方法还是神经网络方法都是如此。在这项工作中，我们展示Hessian矩阵可以直接从深度学习模型预测，而不依赖于自动微分或有限差分。我们观察到可以从图神经网络消息传递过程中计算出的不可约表示特征（最高到l=2度）构建SE(3)-等变、对称的Hessian矩阵。这使得HIP Hessians比传统方法快一到两个数量级，更准确，内存效率更高，更容易训练，并实现了随系统规模增大更有利的扩展性。我们在广泛的下游任务中验证了我们的预测，证明在过渡态搜索、加速几何优化、零点能校正和振动分析基准测试中均表现出一致优越的性能。我们在https://github.com/BurgerAndreas/hip开源了HIP代码库和模型权重，以促进Hessian直接预测的进一步发展。


### 论文摘要

Fundamental tasks in computational chemistry, from transition state search to vibrational analysis, rely on molecular Hessians, which are the second derivatives of the potential energy. Yet, Hessians are computationally expensive to calculate and scale poorly with system size, with both quantum mechanical methods and neural networks. In this work, we demonstrate that Hessians can be predicted directly from a deep learning model, without relying on automatic differentiation or finite differences. We observe that one can construct SE(3)-equivariant, symmetric Hessians from irreducible representations (irrep) features up to degree $l$=2 computed during message passing in graph neural networks. This makes HIP Hessians one to two orders of magnitude faster, more accurate, more memory efficient, easier to train, and enables more favorable scaling with system size. We validate our predictions across a wide range of downstream tasks, demonstrating consistently superior performance for transition state search, accelerated geometry optimization, zero-point energy corrections, and vibrational analysis benchmarks. We open-source the HIP codebase and model weights to enable further development of the direct prediction of Hessians at https://github.com/BurgerAndreas/hip

---

## 57. EEG-Based Consumer Behaviour Prediction: An Exploration from Classical Machine Learning to Graph Neural Networks

**论文链接:** [http://arxiv.org/abs/2509.21567v1](http://arxiv.org/abs/2509.21567v1)

**作者:** Mohammad Parsa Afshar, Aryan Azimi

**发布时间:** 2025-09-25

### GPT解析

### 总结

本研究利用脑电图(EEG)数据和机器学习模型预测消费者行为，比较了图神经网络(GNN)与经典机器学习模型的性能

### 背景

消费者行为预测是营销、认知神经科学和人机交互领域的重要目的，EEG数据可通过提供大脑神经活动的详细信息来帮助分析决策过程

### 目的

利用比较方法通过EEG数据预测消费者行为

### 方法

提取并清理Neuma数据集的EEG特征，为GNN模型创建脑连接特征，实现不同架构的GNN模型，应用广泛的经典模型如集成模型进行对比

### 主要发现

总体结果无显著差异，但GNN模型在某些基本标准上表现优于经典模型

### 结论

结合EEG信号分析和机器学习模型可提供更深入理解消费者行为的方法，同时为EEG神经营销领域常用模型与新兴模型(如GNN)提供了全面比较

### 翻译

消费者行为预测是营销、认知神经科学和人机交互领域的重要目的之一。脑电图(EEG)数据可以通过提供大脑神经活动的详细信息来帮助分析决策过程。在本研究中，采用比较方法通过EEG数据预测消费者行为。首先，提取并清理了来自Neuma数据集的EEG数据特征。对于图神经网络(GNN)模型，创建了脑连接特征。使用了不同的机器学习模型，如经典模型和图神经网络，并进行比较。实现了具有不同架构的GNN模型进行全面比较；此外，应用了广泛的经典模型，如集成模型，这些模型可以非常有帮助地展示每个模型在数据集上的差异和性能。尽管总体结果没有显示出显著差异，但GNN模型在某些基本标准上通常表现更好，而经典模型在这些标准上表现不令人满意。这项研究不仅表明结合EEG信号分析和机器学习模型可以提供一种更深入理解消费者行为的方法，还提供了先前在基于EEG的神经营销研究中广泛使用的机器学习模型(如支持向量机SVM)与在该领域未使用或很少使用的模型(如图神经网络)之间的全面比较。


### 论文摘要

Prediction of consumer behavior is one of the important purposes in marketing, cognitive neuroscience, and human-computer interaction. The electroencephalography (EEG) data can help analyze the decision process by providing detailed information about the brain's neural activity. In this research, a comparative approach is utilized for predicting consumer behavior by EEG data. In the first step, the features of the EEG data from the NeuMa dataset were extracted and cleaned. For the Graph Neural Network (GNN) models, the brain connectivity features were created. Different machine learning models, such as classical models and Graph Neural Networks, are used and compared. The GNN models with different architectures are implemented to have a comprehensive comparison; furthermore, a wide range of classical models, such as ensemble models, are applied, which can be very helpful to show the difference and performance of each model on the dataset. Although the results did not show a significant difference overall, the GNN models generally performed better in some basic criteria where classical models were not satisfactory. This study not only shows that combining EEG signal analysis and machine learning models can provide an approach to deeper understanding of consumer behavior, but also provides a comprehensive comparison between the machine learning models that have been widely used in previous studies in the EEG-based neuromarketing such as Support Vector Machine (SVM), and the models which are not used or rarely used in the field, like Graph Neural Networks.

---

## 58. GraphPFN: A Prior-Data Fitted Graph Foundation Model

**论文链接:** [http://arxiv.org/abs/2509.21489v1](http://arxiv.org/abs/2509.21489v1)

**作者:** Dmitry Eremeev, Oleg Platonov, Gleb Bazhenov, Artem Babenko, Liudmila Prokhorenkova

**发布时间:** 2025-09-25

### GPT解析

### 总结

本文提出了GraphPFN，一种用于节点级预测的先验数据拟合网络，通过在合成图上预训练实现了图基础模型的性能突破。

### 背景

基础模型在大规模数据集上的预训练已改变自然语言处理和计算机视觉领域，但在图数据上的应用仍然有限。现有的图基础模型如G2T-FM虽优于之前尝试，但主要依赖手工特征，限制了学习复杂图模式的能力。

### 目的

设计一种能够捕捉图结构依赖关系的图基础模型，克服现有方法对手工特征的依赖。

### 方法

1) 设计合成属性图的先验分布；2) 使用随机块模型和优先连接过程的组合生成图结构；3) 应用图感知结构因果模型生成节点属性和目标；4) 将表格基础模型LimiX与图邻域聚合层结合；5) 在合成图上训练模型以捕获图结构依赖。

### 主要发现

GraphPFN在多达50,000个节点的多样化真实世界图数据集上展现出强大的上下文学习能力，微调后达到最先进结果，在大多数数据集上优于G2T-FM和任务特定图神经网络。

### 结论

在精心设计的先验分布的合成图上进行预训练是构建图基础模型的有效策略。

### 翻译

在大型数据集上预训练的基础模型已经改变了自然语言处理和计算机视觉等领域，但它们在图数据上的应用仍然有限。最近出现的图基础模型，如G2T-FM，利用表格基础模型处理图任务，并被证明显著优于之前创建图基础模型的尝试。然而，这些模型主要依赖手工制作的图特征，限制了它们学习复杂图特定模式的能力。在这项工作中，我们提出了GraphPFN：一种用于节点级预测的先验数据拟合网络。首先，我们设计了一种合成属性图的先验分布。对于图结构生成，我们使用多个随机块模型和优先连接过程的新组合。然后，我们应用图感知结构因果模型生成节点属性和目标。这个过程使我们能够高效生成各种真实的图数据集。接着，我们将表格基础模型LimiX与基于注意力的图邻域聚合层相结合，并在从我们的先验分布采样的合成图上训练它，使模型能够捕获表格数据中不存在的图结构依赖关系。在多达50,000个节点的多样化真实世界图数据集上，GraphPFN展现了强大的上下文学习能力，并且在微调后达到了最先进的结果，在大多数数据集上都优于G2T-FM和从零开始训练的任务特定图神经网络。更广泛地说，我们的工作证明了在精心设计的先验分布的合成图上进行预训练是构建图基础模型的有效策略。


### 论文摘要

Foundation models pretrained on large-scale datasets have transformed such fields as natural language processing and computer vision, but their application to graph data remains limited. Recently emerged graph foundation models, such as G2T-FM, utilize tabular foundation models for graph tasks and were shown to significantly outperform prior attempts to create GFMs. However, these models primarily rely on hand-crafted graph features, limiting their ability to learn complex graph-specific patterns. In this work, we propose GraphPFN: a prior-data fitted network for node-level prediction. First, we design a prior distribution of synthetic attributed graphs. For graph structure generation, we use a novel combination of multiple stochastic block models and a preferential attachment process. We then apply graph-aware structured causal models to generate node attributes and targets. This procedure allows us to efficiently generate a wide range of realistic graph datasets. Then, we augment the tabular foundation model LimiX with attention-based graph neighborhood aggregation layers and train it on synthetic graphs sampled from our prior, allowing the model to capture graph structural dependencies not present in tabular data. On diverse real-world graph datasets with up to 50,000 nodes, GraphPFN shows strong in-context learning performance and achieves state-of-the-art results after finetuning, outperforming both G2T-FM and task-specific GNNs trained from scratch on most datasets. More broadly, our work demonstrates that pretraining on synthetic graphs from a well-designed prior distribution is an effective strategy for building graph foundation models.

---

## 59. LABELING COPILOT: A Deep Research Agent for Automated Data Curation in Computer Vision

**论文链接:** [http://arxiv.org/abs/2509.22631v1](http://arxiv.org/abs/2509.22631v1)

**作者:** Debargha Ganguly, Sumit Kumar, Ishwar Balappanawar, Weicong Chen, Shashank Kambhatla, Srinivasan Iyengar, Shivkumar Kalyanaraman, Ponnurangam Kumaraguru, Vipin Chaudhary

**发布时间:** 2025-09-26

### GPT解析

### 总结

Labeling Copilot是一个用于计算机视觉的数据策展深度研究代理，通过大型多模态语言模型驱动的中央编排器代理，结合三个核心能力工具解决了高质量数据集构建的瓶颈问题。

### 背景

高质量、特定领域的数据集是部署强大视觉系统的主要瓶颈，在研究大量未标记数据时，需要在数据质量、多样性和成本之间进行复杂的权衡。

### 目的

介绍Labeling Copilot，这是第一个用于计算机视觉的数据策展深度研究代理，旨在解决数据集构建的瓶颈问题。

### 方法

一个由大型多模态语言模型驱动的中央编排器代理，使用多步推理执行三个核心能力的专业工具：(1)校准发现：从大型存储库中获取相关、分布内的数据；(2)可控合成：通过强大的过滤为罕见场景生成新数据；(3)共识标注：通过新颖的共识机制编排多个基础模型产生准确的标注。

### 主要发现

共识标注模块在COCO数据集上平均每张图像产生14.2个候选提案，几乎是真实物体数量的两倍，达到37.1%的标注mAP；在Open Images数据集上发现了903个新类别；校准发现工具比同类方法高效40倍。

### 结论

具有优化、可扩展工具的代理工作流程为策展工业规模数据集提供了强大的基础。

### 翻译

策划高质量、特定领域的数据集是部署强大视觉系统的主要瓶颈，在研究大量未标记数据时，需要在数据质量、多样性和成本之间进行复杂的权衡。我们介绍了Labeling Copilot，这是第一个用于计算机视觉的数据策展深度研究代理。一个由大型多模态语言模型驱动的中央编排器代理，使用多步推理执行三个核心能力的专业工具：(1)校准发现从大型存储库中获取相关、分布内的数据；(2)可控合成通过强大的过滤为罕见场景生成新数据；(3)共识标注通过新颖的共识机制编排多个基础模型产生准确的标注，该机制结合了非极大值抑制和投票。我们的大规模验证证明了Labeling Copilot组件的有效性。共识标注模块在物体发现方面表现出色：在密集的COCO数据集上，平均每张图像有14.2个候选提案，几乎是7.4个真实物体的两倍，最终达到37.1%的标注mAP。在网络规模的Open Images数据集上，它处理了极度的类别不平衡，发现了903个新的边界框类别。同时，校准发现工具在1000万样本规模测试中，采用主动学习策略，比具有相同样本效率的替代方法高效40倍。这些实验验证了具有优化、可扩展工具的代理工作流程为策展工业规模数据集提供了强大的基础。


### 论文摘要

Curating high-quality, domain-specific datasets is a major bottleneck for deploying robust vision systems, requiring complex trade-offs between data quality, diversity, and cost when researching vast, unlabeled data lakes. We introduce Labeling Copilot, the first data curation deep research agent for computer vision. A central orchestrator agent, powered by a large multimodal language model, uses multi-step reasoning to execute specialized tools across three core capabilities: (1) Calibrated Discovery sources relevant, in-distribution data from large repositories; (2) Controllable Synthesis generates novel data for rare scenarios with robust filtering; and (3) Consensus Annotation produces accurate labels by orchestrating multiple foundation models via a novel consensus mechanism incorporating non-maximum suppression and voting. Our large-scale validation proves the effectiveness of Labeling Copilot's components. The Consensus Annotation module excels at object discovery: on the dense COCO dataset, it averages 14.2 candidate proposals per image-nearly double the 7.4 ground-truth objects-achieving a final annotation mAP of 37.1%. On the web-scale Open Images dataset, it navigated extreme class imbalance to discover 903 new bounding box categories, expanding its capability to over 1500 total. Concurrently, our Calibrated Discovery tool, tested at a 10-million sample scale, features an active learning strategy that is up to 40x more computationally efficient than alternatives with equivalent sample efficiency. These experiments validate that an agentic workflow with optimized, scalable tools provides a robust foundation for curating industrial-scale datasets.

---

## 60. Guiding Evolution of Artificial Life Using Vision-Language Models

**论文链接:** [http://arxiv.org/abs/2509.22447v1](http://arxiv.org/abs/2509.22447v1)

**作者:** Nikhil Baid, Hannah Erlebach, Paul Hellegouarch, Frederico Wieser

**发布时间:** 2025-09-26

**备注:** 9 pages, 6 figures. Accepted for publication in the Proceedings of  the Artificial Life Conference 2025 (MIT Press)

### GPT解析

### 总结

研究提出ASAL++方法，利用多模态基础模型引导的类开放式搜索在人工生命模拟中实现进化目标

### 背景

基础模型(FMs)为人工生命(ALife)领域提供了通过自动化搜索ALife模拟的新工具，先前工作使用视觉语言模型(VLMs)将ALife模拟与自然语言目标提示对齐

### 目的

在ASAL方法基础上开发ASAL++，实现具有开放式特征的基础模型驱动的人工生命发现

### 方法

使用第二个基础模型基于模拟的视觉历史提出新的进化目标，诱导出具有越来越复杂目标的进化轨迹；探索两种策略：(1)每次迭代进化模拟以匹配单个新提示(EST)；(2)进化模拟以匹配生成的提示序列(ETT)；在Lenia基质中使用Gemma-3测试

### 主要发现

EST策略促进更大的视觉新颖性，ETT策略促进更连贯和可解释的进化序列

### 结论

ASAL++指向具有开放式特征的基础模型驱动人工生命发现的新方向

### 翻译

基础模型(FMs)最近通过提供强大的工具来自动化搜索人工生命(ALife)模拟，为人工生命领域开辟了新前沿。先前的工作使用视觉语言模型(VLMs)将人工生命模拟与自然语言目标提示对齐。我们在自动化搜索人工生命(ASAL)的基础上引入ASAL++，这是一种由多模态基础模型引导的类开放式搜索方法。我们使用第二个基础模型基于模拟的视觉历史提出新的进化目标，从而诱导出具有越来越复杂目标的进化轨迹。我们探索两种策略：(1)每次迭代进化模拟以匹配单个新提示(进化监督目标：EST)；(2)进化模拟以匹配生成的提示序列(进化时间目标：ETT)。我们在Lenia基质中使用Gemma-3提出进化目标，实证测试了我们的方法，结果表明EST促进更大的视觉新颖性，而ETT促进更连贯和可解释的进化序列。我们的研究结果表明，ASAL++指向具有开放式特征的基础模型驱动人工生命发现的新方向。


### 论文摘要

Foundation models (FMs) have recently opened up new frontiers in the field of artificial life (ALife) by providing powerful tools to automate search through ALife simulations. Previous work aligns ALife simulations with natural language target prompts using vision-language models (VLMs). We build on Automated Search for Artificial Life (ASAL) by introducing ASAL++, a method for open-ended-like search guided by multimodal FMs. We use a second FM to propose new evolutionary targets based on a simulation's visual history. This induces an evolutionary trajectory with increasingly complex targets.   We explore two strategies: (1) evolving a simulation to match a single new prompt at each iteration (Evolved Supervised Targets: EST) and (2) evolving a simulation to match the entire sequence of generated prompts (Evolved Temporal Targets: ETT). We test our method empirically in the Lenia substrate using Gemma-3 to propose evolutionary targets, and show that EST promotes greater visual novelty, while ETT fosters more coherent and interpretable evolutionary sequences.   Our results suggest that ASAL++ points towards new directions for FM-driven ALife discovery with open-ended characteristics.

---

## 61. UnderwaterVLA: Dual-brain Vision-Language-Action architecture for Autonomous Underwater Navigation

**论文链接:** [http://arxiv.org/abs/2509.22441v1](http://arxiv.org/abs/2509.22441v1)

**作者:** Zhangyuan Wang, Yunpeng Zhu, Yuqi Yan, Xiaoyuan Tian, Xinhao Shao, Meixuan Li, Weikun Li, Guangsheng Su, Weicheng Cui, Dixia Fan

**发布时间:** 2025-09-26

**备注:** This paper introduces the first VLA framework for AUVs, featuring a  dual-brain architecture and zero-data MPC for real-world underwater  navigation

### GPT解析

### 总结

本文提出了UnderwaterVLA，一种将多模态基础模型与具身智能系统集成的新型自主水下导航框架，通过三种创新解决了水下操作面临的挑战，并在实地测试中显示出优越性能。

### 背景

水下操作面临流体动力学干扰、通信带宽有限以及浑浊水域中传感器性能下降等挑战，导致自主导航困难。

### 目的

开发一个能够在复杂水下环境中实现高效导航的自主水下导航系统，减少对特定训练数据的依赖，提高环境适应性。

### 方法

提出三种创新：1) 双脑架构解耦高级任务推理与低级反应控制；2) 首次将视觉-语言-行动模型应用于水下机器人，实现可解释决策；3) 流体动力学感知的模型预测控制方案实时补偿流体效应。

### 主要发现

在实地测试中，UnderwaterVLA在视觉条件下降的情况下减少了导航误差，比基线方法提高任务完成率19%至27%。

### 结论

UnderwaterVLA通过减少对水下特定训练数据的依赖并提高环境适应性，为下一代智能自主水下航行器(AUVs)提供了可扩展且经济有效的技术路径。

### 翻译

本文提出了UnderwaterVLA，一种新颖的自主水下导航框架，将多模态基础模型与具身智能系统集成。由于流体动力学干扰、通信带宽有限以及浑浊水域中传感器性能下降，水下操作仍然困难。为应对这些挑战，我们引入了三项创新。首先，双脑架构将高级任务推理与低级反应控制解耦，使系统在通信和计算约束下能够稳健运行。其次，我们首次将视觉-语言-行动模型应用于水下机器人，纳入结构化思维链推理以实现可解释的决策制定。第三，流体动力学感知的模型预测控制方案实时补偿流体效应，无需昂贵的任务特定训练。实地测试结果表明，UnderwaterVLA在视觉条件下降的情况下减少了导航误差，同时比基线方法提高任务完成率19%至27%。通过减少对水下特定训练数据的依赖并提高环境适应性，UnderwaterVLA为下一代智能AUVs提供了可扩展且经济有效的技术路径。


### 论文摘要

This paper presents UnderwaterVLA, a novel framework for autonomous underwater navigation that integrates multimodal foundation models with embodied intelligence systems. Underwater operations remain difficult due to hydrodynamic disturbances, limited communication bandwidth, and degraded sensing in turbid waters. To address these challenges, we introduce three innovations. First, a dual-brain architecture decouples high-level mission reasoning from low-level reactive control, enabling robust operation under communication and computational constraints. Second, we apply Vision-Language-Action(VLA) models to underwater robotics for the first time, incorporating structured chain-of-thought reasoning for interpretable decision-making. Third, a hydrodynamics-informed Model Predictive Control(MPC) scheme compensates for fluid effects in real time without costly task-specific training. Experimental results in field tests show that UnderwaterVLA reduces navigation errors in degraded visual conditions while maintaining higher task completion by 19% to 27% over baseline. By minimizing reliance on underwater-specific training data and improving adaptability across environments, UnderwaterVLA provides a scalable and cost-effective path toward the next generation of intelligent AUVs.

---

## 62. MoveFM-R: Advancing Mobility Foundation Models via Language-driven Semantic Reasoning

**论文链接:** [http://arxiv.org/abs/2509.22403v1](http://arxiv.org/abs/2509.22403v1)

**作者:** Fanjin Meng, Yuan Yuan, Jingtao Ding, Jie Feng, Chonghua Han, Yong Li

**发布时间:** 2025-09-26

### GPT解析

### 总结

本文提出了MoveFM-R框架，通过结合移动基础模型(MFMs)和大型语言模型(LLMs)的优势，实现了更全面、可解释和强大的人类移动性建模。

### 背景

移动基础模型(MFMs)在建模人类移动模式方面取得进展，但因数据规模和语义理解限制而面临瓶颈。大型语言模型(LLMs)虽具强大语义推理能力，但缺乏生成物理合理移动轨迹所需的时间和空间统计内在理解。

### 目的

提出MoveFM-R框架，利用语言驱动的语义推理能力释放移动基础模型潜力，解决两个关键挑战：连续地理坐标与离散语言标记间的词汇不匹配，以及MFM潜在向量与LLM语义世界间的表示差距。

### 方法

MoveFM-R建立在三个核心创新上：语义增强的位置编码以弥合地理-语言差距；渐进式课程使LLM推理与移动模式保持一致；交互式自反思机制用于条件轨迹生成。

### 主要发现

广泛实验表明，MoveFM-R显著优于现有基于MFM和LLM的基线模型。在零样本设置中表现出强大泛化能力，并擅长根据自然语言指令生成真实轨迹。

### 结论

通过结合MFM的统计能力和LLM的深度语义理解，MoveFM-R开创了人类移动性建模的新范式，实现更全面、可解释和强大的建模效果。

### 翻译

移动基础模型(MFMs)已推进了人类移动模式的建模，但由于数据规模和语义理解的限制，它们遇到了瓶颈。虽然大型语言模型(LLMs)提供了强大的语义推理能力，但它们缺乏生成物理上合理的移动轨迹所需的时间和空间统计的内在理解。为了解决这些差距，我们提出了MoveFM-R，一个新颖的框架，通过利用语言驱动的语义推理能力释放了移动基础模型的全部潜力。它解决了两个关键挑战：连续地理坐标与离散语言标记之间的词汇不匹配，以及MFM的潜在向量与LLM的语义世界之间的表示差距。MoveFM-R建立在三个核心创新之上：语义增强的位置编码，以弥合地理-语言差距；渐进式课程，使LLM的推理与移动模式保持一致；以及交互式自反思机制，用于条件轨迹生成。广泛的实验表明，MoveFM-R显著优于现有的基于MFM和LLM的基线。它还在零样本设置中表现出强大的泛化能力，并擅长根据自然语言指令生成真实的轨迹。通过结合MFM的统计能力和LLM的深度语义理解，MoveFM-R开创了一种新范式，能够更全面、可解释和强大地建模人类移动性。MoveFM-R的实现可在网上https://anonymous.4open.science/r/MoveFM-R-CDE7/获取。


### 论文摘要

Mobility Foundation Models (MFMs) have advanced the modeling of human movement patterns, yet they face a ceiling due to limitations in data scale and semantic understanding. While Large Language Models (LLMs) offer powerful semantic reasoning, they lack the innate understanding of spatio-temporal statistics required for generating physically plausible mobility trajectories. To address these gaps, we propose MoveFM-R, a novel framework that unlocks the full potential of mobility foundation models by leveraging language-driven semantic reasoning capabilities. It tackles two key challenges: the vocabulary mismatch between continuous geographic coordinates and discrete language tokens, and the representation gap between the latent vectors of MFMs and the semantic world of LLMs. MoveFM-R is built on three core innovations: a semantically enhanced location encoding to bridge the geography-language gap, a progressive curriculum to align the LLM's reasoning with mobility patterns, and an interactive self-reflection mechanism for conditional trajectory generation. Extensive experiments demonstrate that MoveFM-R significantly outperforms existing MFM-based and LLM-based baselines. It also shows robust generalization in zero-shot settings and excels at generating realistic trajectories from natural language instructions. By synthesizing the statistical power of MFMs with the deep semantic understanding of LLMs, MoveFM-R pioneers a new paradigm that enables a more comprehensive, interpretable, and powerful modeling of human mobility. The implementation of MoveFM-R is available online at https://anonymous.4open.science/r/MoveFM-R-CDE7/.

---

## 63. CHRONOBERG: Capturing Language Evolution and Temporal Awareness in Foundation Models

**论文链接:** [http://arxiv.org/abs/2509.22360v1](http://arxiv.org/abs/2509.22360v1)

**作者:** Niharika Hegde, Subarnaduti Paul, Lars Joel-Frey, Manuel Brack, Kristian Kersting, Martin Mundt, Patrick Schramowski

**发布时间:** 2025-09-26

### GPT解析

### 总结

这篇论文介绍了CHRONOBERG，一个时间结构化的英语书籍语料库，跨越250年，来自古腾堡计划并添加了时间注释。研究通过这个语料库探讨了大型语言模型在捕捉语言随时间变化方面的能力，并展示了现代LLM工具在检测歧视性语言和跨时间段情境化情感方面的不足。

### 背景

现有的大型语言模型能够利用社交媒体和网络抓取的各种数据在规模上运行。然而，虽然现有语料库多样化，但它们经常缺乏长期时间结构，这可能限制LLM将语言语义和规范演变置于语境中的能力，以及捕捉历时变化的能力。

### 目的

为了支持对语言历时变化的分析和训练，作者引入了CHRONOBERG语料库，旨在帮助大型语言模型更好地捕捉语言的语义和规范演变，以及理解历时变化。

### 方法

作者从古腾堡计划策划了一个跨越250年的英语书籍文本的时间结构化语料库，并添加了各种时间注释。他们利用书籍经过编辑的性质，通过时间敏感的效价-唤醒-优势分析来量化词汇语义变化随时间的变化，并构建了历史上校准的情感词汇表。

### 主要发现

研究显示，现代基于LLM的工具需要更好地定位它们在不同时间段对歧视性语言的检测和情感的情境化。在CHRONOBERG上顺序训练的语言模型难以编码意义的历时转变，强调了时间感知训练和评估流程的必要性。

### 结论

CHRONOBERG被定位为研究语言变化和时间泛化的可扩展资源，强调了时间感知训练和评估流程的重要性，以及需要改进大型语言模型捕捉语言随时间变化的能力。

### 翻译

论文包括可能对读者有冒犯性的语言和样本展示。CHRONOBERG在HuggingFace上公开可用，代码可在GitHub获取。


### 论文摘要

Large language models (LLMs) excel at operating at scale by leveraging social media and various data crawled from the web. Whereas existing corpora are diverse, their frequent lack of long-term temporal structure may however limit an LLM's ability to contextualize semantic and normative evolution of language and to capture diachronic variation. To support analysis and training for the latter, we introduce CHRONOBERG, a temporally structured corpus of English book texts spanning 250 years, curated from Project Gutenberg and enriched with a variety of temporal annotations. First, the edited nature of books enables us to quantify lexical semantic change through time-sensitive Valence-Arousal-Dominance (VAD) analysis and to construct historically calibrated affective lexicons to support temporally grounded interpretation. With the lexicons at hand, we demonstrate a need for modern LLM-based tools to better situate their detection of discriminatory language and contextualization of sentiment across various time-periods. In fact, we show how language models trained sequentially on CHRONOBERG struggle to encode diachronic shifts in meaning, emphasizing the need for temporally aware training and evaluation pipelines, and positioning CHRONOBERG as a scalable resource for the study of linguistic change and temporal generalization. Disclaimer: This paper includes language and display of samples that could be offensive to readers. Open Access: Chronoberg is available publicly on HuggingFace at ( https://huggingface.co/datasets/spaul25/Chronoberg). Code is available at (https://github.com/paulsubarna/Chronoberg).

---

## 64. Galaxy Zoo: Cosmic Dawn -- morphological classifications for over 41,000 galaxies in the Euclid Deep Field North from the Hawaii Two-0 Cosmic Dawn survey

**论文链接:** [http://arxiv.org/abs/2509.22311v1](http://arxiv.org/abs/2509.22311v1)

**作者:** James Pearson, Hugh Dickinson, Stephen Serjeant, Mike Walmsley, Lucy Fortson, Sandor Kruk, Karen L. Masters, Brooke D. Simmons, R. J. Smethurst, Chris Lintott, Lukas Zalesky, Conor McPartland, John R. Weaver, Sune Toft, Dave Sanders, Nima Chartab, Henry Joy McCracken, Bahram Mobasher, Istvan Szapudi, Noah East, Wynne Turner, Matthew Malkan, William J. Pearson, Tomotsugu Goto, Nagisa Oi

**发布时间:** 2025-09-26

**备注:** 23 pages; 21 figures; 3 tables; submitted to MNRAS

### GPT解析

### 总结

本研究通过Galaxy Zoo公民科学家和深度学习模型Zoobot对超过41,000个星系进行了形态学分类，覆盖了6平方度Euclid Deep Field North区域，红移范围至约2.5。研究公开发布了45,000多个物体的分类数据，并发现了51个新的引力透镜系统。

### 背景

研究基于Hawaii Twenty Square Degree (H20)调查，这是Cosmic Dawn调查的一部分，使用Hyper Suprime-Cam (HSC)获取超深多波段图像，深度达到m_HSC-i = 21.5。Galaxy Zoo项目通过公民科学家参与大规模天文数据集的检查。

### 目的

对EDFN区域中的星系进行大规模形态学分类，创建一个公开的数据集，为后续观测和深度学习模型训练提供基础。

### 方法

结合Galaxy Zoo公民科学家和深度学习基础模型Zoobot对物体进行分类，采用Zoobot在主动学习循环中提高模型性能和志愿者体验，使用超深多波段HSC成像数据进行分析。

### 主要发现

在EDFN区域发现了51个新的引力透镜系统；公开发布了超过45,000个物体的分类数据，包括超过41,000个星系，这些星系的中位红移为0.42±0.23，并提供了相关的图像裁剪。

### 结论

该数据集为EDFN中物体的后续成像提供了宝贵机会，同时可作为训练深度学习模型的真值集，应用于地面观测站如Vera C. Rubin天文台。

### 翻译

我们提出了对超过41,000个星系的形态学分类，这些星系位于六个平方度的Euclid Deep Field North(EDFN)区域，红移至z_phot~2.5，来自作为更广泛Cosmic Dawn调查一部分的Hawaii Twenty Square Degree (H20)调查。Galaxy Zoo公民科学家通过众包挖掘星系外成像数据，在检查大型天文数据集方面发挥着关键作用。这一轮Galaxy Zoo: Cosmic Dawn (GZCD)项目，数万名志愿者和深度学习基础模型Zoobot共同对超深多波段Hyper Suprime-Cam (HSC)成像中的物体进行了分类，深度达到m_HSC-i = 21.5。在此，我们展示了这一轮的细节和一般分析，包括在主动学习周期中使用Zoobot以提高模型性能和志愿者体验，以及在EDFN中发现51个新的引力透镜。我们还宣布公开发布超过45,000个物体的分类数据，包括超过41,000个星系（中位z_phot为0.42±0.23）及其相关的图像裁剪。该数据集为EDFN中物体的后续成像提供了宝贵机会，同时也作为训练深度学习模型的真值集，应用于新近投入运营的Vera C. Rubin天文台等地面调查。


### 论文摘要

We present morphological classifications of over 41,000 galaxies out to $z_{\rm phot}\sim2.5$ across six square degrees of the Euclid Deep Field North (EDFN) from the Hawaii Twenty Square Degree (H20) survey, a part of the wider Cosmic Dawn survey. Galaxy Zoo citizen scientists play a crucial role in the examination of large astronomical data sets through crowdsourced data mining of extragalactic imaging. This iteration, Galaxy Zoo: Cosmic Dawn (GZCD), saw tens of thousands of volunteers and the deep learning foundation model Zoobot collectively classify objects in ultra-deep multiband Hyper Suprime-Cam (HSC) imaging down to a depth of $m_{HSC-i} = 21.5$. Here, we present the details and general analysis of this iteration, including the use of Zoobot in an active learning cycle to improve both model performance and volunteer experience, as well as the discovery of 51 new gravitational lenses in the EDFN. We also announce the public data release of the classifications for over 45,000 subjects, including more than 41,000 galaxies (median $z_{\rm phot}$ of $0.42\pm0.23$), along with their associated image cutouts. This data set provides a valuable opportunity for follow-up imaging of objects in the EDFN as well as acting as a truth set for training deep learning models for application to ground-based surveys like that of the newly operational Vera C. Rubin Observatory.

---

## 65. Aurora: Towards Universal Generative Multimodal Time Series Forecasting

**论文链接:** [http://arxiv.org/abs/2509.22295v1](http://arxiv.org/abs/2509.22295v1)

**作者:** Xingjian Wu, Jianxin Jin, Wanghui Qiu, Peng Chen, Yang Shu, Bin Yang, Chenjuan Guo

**发布时间:** 2025-09-26

### GPT解析

### 总结

本文提出了Aurora，一个多模态时间序列基础模型，支持多模态输入和零样本推理，具有强大的跨域泛化能力。

### 背景

跨域泛化在时间序列预测中非常重要，因为相似的历史信息可能因领域特定特性导致不同的未来趋势。现有单模态模型缺乏对文本等模态中领域知识的利用，而端到端多模态模型不支持跨域场景的零样本推理。

### 目的

开发一个能够有效利用多模态信息并支持跨域零样本推理的时间序列基础模型，以解决跨域泛化问题。

### 方法

Aurora通过标记化、编码和蒸馏提取多模态领域知识，使用模态引导多头自注意力机制将这些知识注入时间表示建模，并在解码阶段提出原型引导流匹配方法进行生成概率预测。

### 主要发现

在TimeMMD、TSFM-Bench和ProbTS等基准上的实验表明，Aurora在单模态和多模态场景下都取得了持续的最先进性能。

### 结论

Aurora作为多模态时间序列基础模型，能够自适应提取和关注关键领域知识，有效解决了跨域泛化问题，并在各种场景下表现出色。

### 翻译

跨域泛化在时间序列预测中非常重要，因为相似的历史信息可能因领域特定特性导致不同的未来趋势。最近的工作主要集中在构建单模态时间序列基础模型和端到端多模态监督模型。由于领域特定知识通常包含在文本等模态中，前者缺乏对它们的明确利用，从而限制了性能。后者针对端到端场景设计，不支持跨域场景的零样本推理。在这项工作中，我们引入了Aurora，一个多模态时间序列基础模型，它支持多模态输入和零样本推理。在跨域多模态时间序列语料库上预训练后，Aurora能够自适应提取并关注文本或图像模态中包含的关键领域知识，从而具有强大的跨域泛化能力。通过标记化、编码和蒸馏，Aurora可以将多模态领域知识提取为指导，然后使用模态引导多头自注意力机制将它们注入到时间表示的建模中。在解码阶段，多模态表示被用来生成未来标记的条件和原型，为生成概率预测做出贡献。在TimeMMD、TSFM-Bench和ProbTS等公认基准上的全面实验证明了Aurora在单模态和多模态场景下都取得了持续的最先进性能。


### 论文摘要

Cross-domain generalization is very important in Time Series Forecasting because similar historical information may lead to distinct future trends due to the domain-specific characteristics. Recent works focus on building unimodal time series foundation models and end-to-end multimodal supervised models. Since domain-specific knowledge is often contained in modalities like texts, the former lacks the explicit utilization of them, thus hindering the performance. The latter is tailored for end-to-end scenarios and does not support zero-shot inference for cross-domain scenarios. In this work, we introduce Aurora, a Multimodal Time Series Foundation Model, which supports multimodal inputs and zero-shot inference. Pretrained on Corss-domain Multimodal Time Series Corpus, Aurora can adaptively extract and focus on key domain knowledge contained in corrsponding text or image modalities, thus possessing strong Cross-domain generalization capability. Through tokenization, encoding, and distillation, Aurora can extract multimodal domain knowledge as guidance and then utilizes a Modality-Guided Multi-head Self-Attention to inject them into the modeling of temporal representations. In the decoding phase, the multimodal representations are used to generate the conditions and prototypes of future tokens, contributing to a novel Prototype-Guided Flow Matching for generative probabilistic forecasting. Comprehensive experiments on well-recognized benchmarks, including TimeMMD, TSFM-Bench and ProbTS, demonstrate the consistent state-of-the-art performance of Aurora on both unimodal and multimodal scenarios.

---

## 66. Speak Your Mind: The Speech Continuation Task as a Probe of Voice-Based Model Bias

**论文链接:** [http://arxiv.org/abs/2509.22061v1](http://arxiv.org/abs/2509.22061v1)

**作者:** Shree Harsha Bokkahalli Satish, Harm Lameris, Olivier Perrotin, Gustav Eje Henter, Éva Székely

**发布时间:** 2025-09-26

**备注:** 6 pages, 1 figure, Submitted to IEEE ICASSP 2026

### GPT解析

### 总结

本研究对语音延续任务中的偏见进行了系统性评估，研究了性别和发声类型对语音生成行为的影响，发现模型在延续过程中存在性别和声音质量方面的系统性偏见。

### 背景

语音延续(SC)是生成连贯语音扩展的任务，需要同时保持语义上下文和说话人身份。由于SC限制在单个音频流中，它比对话提供了更直接的环境来探测语音基础模型中的偏见。

### 目的

首次对语音延续中的偏见进行系统性评估，研究性别和发声类型（气息声、沙哑声、末尾沙哑声）如何影响延续行为。

### 方法

评估三种语音模型：SpiritLM（基础和表达版）、VAE-GSLM和SpeechGPT，从说话人相似性、声音质量保持和基于文本的偏见指标等方面进行评估。

### 主要发现

1) 说话人相似性和连贯性仍是挑战；2) 文本评估显示显著的模型和性别交互作用；3) 当连贯性足够高时，性别效应在文本指标上显现；4) 女性提示的延续比男性提示更强烈地转向常态发声，揭示系统性声音质量偏见。

### 结论

语音延续可作为探测语音基础模型中社会相关表征偏见的受控探针，随着延续质量的提高，它将成为越来越有价值的诊断工具。

### 翻译

语音延续(SC)是生成连贯语音扩展的任务，同时保持语义上下文和说话人身份。由于SC被限制在单个音频流中，它比对话提供了更直接的环境来探测语音基础模型中的偏见。在这项工作中，我们首次对SC中的偏见进行了系统性评估，研究了性别和发声类型（气息声、沙哑声、末尾沙哑声）如何影响延续行为。我们评估了三种最近的模型：SpiritLM（基础和表达版）、VAE-GSLM和SpeechGPT，从说话人相似性、声音质量保持和基于文本的偏见指标等方面进行评估。结果表明，虽然说话人相似性和连贯性仍然是一个挑战，但文本评估显示了显著的模型和性别交互作用：当连贯性足够高时（对于VAE-GSLM），性别效应会在文本指标上显现，如能动性和句子极性。此外，与男性提示相比，女性提示的延续更强烈地转向常态发声，揭示了系统性的声音质量偏见。这些发现突显了语音延续作为探测语音基础模型中社会相关表征偏见的受控探针的作用，并表明随着延续质量的提高，它将成为信息量越来越大的诊断工具。


### 论文摘要

Speech Continuation (SC) is the task of generating a coherent extension of a spoken prompt while preserving both semantic context and speaker identity. Because SC is constrained to a single audio stream, it offers a more direct setting for probing biases in speech foundation models than dialogue does. In this work we present the first systematic evaluation of bias in SC, investigating how gender and phonation type (breathy, creaky, end-creak) affect continuation behaviour. We evaluate three recent models: SpiritLM (base and expressive), VAE-GSLM, and SpeechGPT across speaker similarity, voice quality preservation, and text-based bias metrics. Results show that while both speaker similarity and coherence remain a challenge, textual evaluations reveal significant model and gender interactions: once coherence is sufficiently high (for VAE-GSLM), gender effects emerge on text-metrics such as agency and sentence polarity. In addition, continuations revert toward modal phonation more strongly for female prompts than for male ones, revealing a systematic voice-quality bias. These findings highlight SC as a controlled probe of socially relevant representational biases in speech foundation models, and suggest that it will become an increasingly informative diagnostic as continuation quality improves.

---

## 67. Task-Adaptive Parameter-Efficient Fine-Tuning for Weather Foundation Models

**论文链接:** [http://arxiv.org/abs/2509.22020v1](http://arxiv.org/abs/2509.22020v1)

**作者:** Shilei Cao, Hehai Lin, Jiashun Cheng, Yang Liu, Guowen Li, Xuehe Wang, Juepeng Zheng, Haoyuan Liang, Meng Jin, Chengwei Qin, Hong Cheng, Haohuan Fu

**发布时间:** 2025-09-26

### GPT解析

### 总结

该研究提出了WeatherPEFT，一种专门针对天气基础模型(WFMs)的参数高效微调(PEFT)框架，解决了现有PEFT方法无法应对天气下游任务独特挑战的问题。

### 背景

机器学习进步使天气基础模型具备跨多种下游任务的泛化能力，但模型规模扩大带来的计算需求增加阻碍了实际部署。现有针对视觉或语言任务的PEFT方法无法处理天气下游任务中的变量异质性、分辨率多样性和时空覆盖变化等挑战。

### 目的

开发一种专门针对WFMs的PEFT框架，以克服现有方法在天气下游任务上的局限性，实现与全参数微调相当的性能但使用更少的可训练参数。

### 方法

WeatherPEFT框架包含两个创新：1) 任务自适应动态提示(TADP)，通过内部和外部模式提取动态将编码器嵌入权重注入预训练骨干网络输入令牌，实现特定下游任务的上下文感知特征重校准；2) 随机Fisher引导自适应选择(SFAS)，利用Fisher信息识别并更新最关键任务参数，保留预训练知识同时引入随机性稳定选择过程。

### 主要发现

在三个下游任务上验证了WeatherPEFT的有效性和效率；现有PEFT方法与全参数微调相比存在显著差距；WeatherPEFT使用更少的可训练参数实现了与全参数微调相当的性能。

### 结论

WeatherPEFT为WFMs提供了一种有效的参数高效微调方法，解决了现有方法在天气特定任务上的局限性，同时保持了计算效率。

### 翻译

尽管机器学习的最新进展使天气基础模型(WFMs)具备了在多样化下游任务中 substantial泛化能力，但其规模扩大带来的日益增长的计算需求 increasingly阻碍了实际部署。当前为视觉或语言任务设计的参数高效微调(PEFT)方法无法应对天气下游任务的独特挑战，如变量异质性、分辨率多样性和时空覆盖变化，导致应用于WFMs时表现不佳。为弥合这一差距，我们引入了WeatherPEFT，这是一种针对WFMs的新型PEFT框架，包含两个协同创新。首先，在前向传播过程中，任务自适应动态提示(TADP)通过内部和外部模式提取，动态地将编码器中的嵌入权重注入到预训练骨干网络的输入令牌中，实现对特定下游任务的上下文感知特征重校准。此外，在反向传播过程中，随机Fisher引导自适应选择(SFAS)不仅利用Fisher信息识别和更新最关键的任务参数，从而保留不变的预训练知识，还引入随机性以稳定选择过程。我们在三个下游任务上证明了WeatherPEFT的有效性和效率，在这些任务上现有PEFT方法与全参数微调相比存在显著差距，而WeatherPEFT使用更少的可训练参数实现了与全参数微调相当的性能。本工作的代码将发布。


### 论文摘要

While recent advances in machine learning have equipped Weather Foundation Models (WFMs) with substantial generalization capabilities across diverse downstream tasks, the escalating computational requirements associated with their expanding scale increasingly hinder practical deployment. Current Parameter-Efficient Fine-Tuning (PEFT) methods, designed for vision or language tasks, fail to address the unique challenges of weather downstream tasks, such as variable heterogeneity, resolution diversity, and spatiotemporal coverage variations, leading to suboptimal performance when applied to WFMs. To bridge this gap, we introduce WeatherPEFT, a novel PEFT framework for WFMs incorporating two synergistic innovations. First, during the forward pass, Task-Adaptive Dynamic Prompting (TADP) dynamically injects the embedding weights within the encoder to the input tokens of the pre-trained backbone via internal and external pattern extraction, enabling context-aware feature recalibration for specific downstream tasks. Furthermore, during backpropagation, Stochastic Fisher-Guided Adaptive Selection (SFAS) not only leverages Fisher information to identify and update the most task-critical parameters, thereby preserving invariant pre-trained knowledge, but also introduces randomness to stabilize the selection. We demonstrate the effectiveness and efficiency of WeatherPEFT on three downstream tasks, where existing PEFT methods show significant gaps versus Full-Tuning, and WeatherPEFT achieves performance parity with Full-Tuning using fewer trainable parameters. The code of this work will be released.

---

## 68. DynaNav: Dynamic Feature and Layer Selection for Efficient Visual Navigation

**论文链接:** [http://arxiv.org/abs/2509.21930v1](http://arxiv.org/abs/2509.21930v1)

**作者:** Jiahui Wang, Changhao Chen

**发布时间:** 2025-09-26

**备注:** Accepted as a poster in NeurIPS 2025

### GPT解析

### 总结

本文提出DynaNav动态视觉导航框架，通过自适应特征和层选择解决了现有基础模型计算开销高和缺乏可解释性的问题，显著提高了视觉导航效率和性能。

### 背景

视觉导航对机器人和具身AI至关重要，但现有基础模型（特别是具有transformer解码器的模型）存在高计算开销和缺乏可解释性的问题，限制了它们在资源受限场景中的部署。

### 目的

开发一种高效且可解释的视觉导航框架，减少计算成本，使其适合在资源受限场景中部署，同时提高导航性能。

### 方法

提出DynaNav框架，根据场景复杂度自适应选择特征和层；使用可训练的硬特征选择器进行稀疏操作提高效率和可解释性；将特征选择集成到早期退出机制中，使用贝叶斯优化确定最优退出阈值以减少计算成本。

### 主要发现

在基于真实世界的数据集和模拟环境中进行实验表明，与ViNT相比，DynaNav实现了2.26倍的FLOPs减少，42.3%的推理时间降低，32.8%的内存使用减少，并在四个公共数据集上提高了导航性能。

### 结论

DynaNav框架有效解决了现有视觉导航模型的计算效率问题，同时提高了导航性能和系统可解释性，适合在资源受限的场景中部署。

### 翻译

视觉导航对机器人和具身AI至关重要。然而，现有的基础模型，特别是那些具有transformer解码器的模型，存在高计算开销和缺乏可解释性的问题，限制了它们在资源受限场景中的部署。为此，我们提出了DynaNav，一种基于场景复杂度自适应特征和层选择的动态视觉导航框架。它采用可训练的硬特征选择器进行稀疏操作，提高效率和可解释性。此外，我们将特征选择集成到早期退出机制中，使用贝叶斯优化确定最优退出阈值以减少计算成本。在基于真实世界的数据集和模拟环境中的广泛实验证明了DynaNav的有效性。与ViNT相比，DynaNav实现了2.26倍的FLOPs减少，42.3%的推理时间降低和32.8%的内存使用减少，同时在四个公共数据集上提高了导航性能。


### 论文摘要

Visual navigation is essential for robotics and embodied AI. However, existing foundation models, particularly those with transformer decoders, suffer from high computational overhead and lack interpretability, limiting their deployment in resource-tight scenarios. To address this, we propose DynaNav, a Dynamic Visual Navigation framework that adapts feature and layer selection based on scene complexity. It employs a trainable hard feature selector for sparse operations, enhancing efficiency and interpretability. Additionally, we integrate feature selection into an early-exit mechanism, with Bayesian Optimization determining optimal exit thresholds to reduce computational cost. Extensive experiments in real-world-based datasets and simulated environments demonstrate the effectiveness of DynaNav. Compared to ViNT, DynaNav achieves a 2.26x reduction in FLOPs, 42.3% lower inference time, and 32.8% lower memory usage, while improving navigation performance across four public datasets.

---

## 69. LG-CD: Enhancing Language-Guided Change Detection through SAM2 Adaptation

**论文链接:** [http://arxiv.org/abs/2509.21894v1](http://arxiv.org/abs/2509.21894v1)

**作者:** Yixiao Liu, Yizhou Yang, Jinwen Li, Jun Tao, Ruoyu Li, Xiangkun Wang, Min Zhu, Junlong Cheng

**发布时间:** 2025-09-26

**备注:** *Corresponding authors: Min Zhu (min.zhu@scu.edu.cn) and Junlong  Cheng (jlcheng@scu.edu.cn)

### GPT解析

### 总结

该研究提出了一种新颖的语言引导变化检测模型(LG-CD)，通过整合文本和视觉信息提高遥感变化检测的准确性和鲁棒性。

### 背景

遥感变化检测(RSCD)通常通过分析多时相图像识别土地覆盖变化，但现有深度学习方法主要关注单模态视觉信息，忽略了多模态数据(如文本)提供的丰富语义信息。

### 目的

解决现有方法忽视多模态数据语义信息的局限性，提出一种能够利用自然语言提示引导模型关注感兴趣区域的变化检测方法。

### 方法

LG-CD模型使用视觉基础模型(SAM2)作为特征提取器捕获多尺度特征，采用多层适配器微调模型，设计文本融合注意力模块(TFAM)对齐视觉和文本信息，并实现视觉-语义融合解码器(V-SFD)深度整合两种信息。

### 主要发现

在LEVIR-CD、WHU-CD和SYSU-CD三个数据集上的实验表明，LG-CD始终优于最先进的变化检测方法，多模态信息融合为通用变化检测提供了新思路。

### 结论

语言引导的变化检测模型能够有效整合文本和视觉信息，显著提高变化检测性能，为遥感变化检测领域提供了新的研究方向。

### 翻译

遥感变化检测通常通过分析多时相图像来识别土地覆盖或地表条件的变化。目前，大多数基于深度学习的方法主要关注学习单模态视觉信息，而忽略了多模态数据(如文本)提供的丰富语义信息。为解决这一局限，我们提出了一种新颖的语言引导变化检测模型(LG-CD)。该模型利用自然语言提示引导网络关注感兴趣区域，显著提高了变化检测的准确性和鲁棒性。具体而言，LG-CD使用视觉基础模型(SAM2)作为特征提取器，从高分辨率到低分辨率捕获双时相遥感图像的多尺度金字塔特征。随后，采用多层适配器微调模型用于下游任务，确保其在遥感变化检测中的有效性。此外，我们设计了文本融合注意力模块(TFAM)对齐视觉和文本信息，使模型能够使用文本提示关注目标变化区域。最后，实现了视觉-语义融合解码器(V-SFD)，通过交叉注意力机制深度整合视觉和语义信息，生成高精度的变化检测掩码。在三个数据集(LEVIR-CD、WHU-CD和SYSU-CD)上的实验表明，LG-CD始终优于最先进的变化检测方法。此外，我们的方法通过利用多模态信息实现通用变化检测提供了新见解。


### 论文摘要

Remote Sensing Change Detection (RSCD) typically identifies changes in land cover or surface conditions by analyzing multi-temporal images. Currently, most deep learning-based methods primarily focus on learning unimodal visual information, while neglecting the rich semantic information provided by multimodal data such as text. To address this limitation, we propose a novel Language-Guided Change Detection model (LG-CD). This model leverages natural language prompts to direct the network's attention to regions of interest, significantly improving the accuracy and robustness of change detection. Specifically, LG-CD utilizes a visual foundational model (SAM2) as a feature extractor to capture multi-scale pyramid features from high-resolution to low-resolution across bi-temporal remote sensing images. Subsequently, multi-layer adapters are employed to fine-tune the model for downstream tasks, ensuring its effectiveness in remote sensing change detection. Additionally, we design a Text Fusion Attention Module (TFAM) to align visual and textual information, enabling the model to focus on target change regions using text prompts. Finally, a Vision-Semantic Fusion Decoder (V-SFD) is implemented, which deeply integrates visual and semantic information through a cross-attention mechanism to produce highly accurate change detection masks. Our experiments on three datasets (LEVIR-CD, WHU-CD, and SYSU-CD) demonstrate that LG-CD consistently outperforms state-of-the-art change detection methods. Furthermore, our approach provides new insights into achieving generalized change detection by leveraging multimodal information.

---

## 70. QoNext: Towards Next-generation QoE for Foundation Models

**论文链接:** [http://arxiv.org/abs/2509.21889v1](http://arxiv.org/abs/2509.21889v1)

**作者:** Yijin Guo, Ye Shen, Farong Wen, Junying Wang, Zicheng Zhang, Qi Jia, Guangtao Zhai

**发布时间:** 2025-09-26

### GPT解析

### 总结

本研究引入了QoNext框架，这是首个将网络和多媒体领域的体验质量(QoE)原则应用于基础模型评估的框架，旨在解决现有评估方法无法全面捕捉用户体验的问题。

### 背景

现有的基础模型评估方法（包括最近以人为中心的方法）仅关注输出正确性，忽视了用户满意度来自于响应质量和交互之间的相互作用，因此无法全面评估用户体验。

### 目的

开发一个能够捕捉和评估用户体验的框架，为基础模型提供更全面、更实用的评估方法，并为产品化服务提供优化指导。

### 方法

QoNext框架确定塑造用户体验的体验因素，将这些因素纳入受控实验中，收集不同配置下的人类评分，构建面向QoE的数据库，并训练预测模型从可测量的系统参数估计感知的用户体验。

### 主要发现

QoNext框架能够进行主动和细粒度的评估，为优化基础模型的产品化服务提供可行的指导。

### 结论

QoNext框架解决了现有基础模型评估方法无法全面捕捉用户体验的问题，通过将QoE原则应用于基础模型评估，提供了更全面、更实用的评估方法。

### 翻译

现有的基础模型评估方法，包括最近以人为中心的方法，未能捕捉到真正重要的内容：用户在交互过程中的体验。当前方法将评估仅视为输出正确性的问题，忽视了用户满意度来自于响应质量和交互之间的相互作用，这限制了它们解释用户体验潜在机制的能力。为了解决这一差距，我们引入了QoNext，这是第一个将网络和多媒体领域的体验质量(QoE)原则应用于基础模型评估的框架。QoNext确定塑造用户体验的体验因素，并将它们纳入受控实验中，在这些实验中收集不同配置下的人类评分。从这些研究中，我们构建了一个面向QoE的数据库，并训练预测模型来从可测量的系统参数中估计感知的用户体验。我们的结果表明，QoNext不仅能够进行主动和细粒度的评估，还为优化基础模型的产品化服务提供了可行的指导。


### 论文摘要

Existing evaluations of foundation models, including recent human-centric approaches, fail to capture what truly matters: user's experience during interaction. Current methods treat evaluation as a matter of output correctness alone, overlooking that user satisfaction emerges from the interplay between response quality and interaction, which limits their ability to account for the mechanisms underlying user experience. To address this gap, we introduce QoNext, the first framework that adapts Quality of Experience (QoE) principles from networking and multimedia to the assessment of foundation models. QoNext identifies experiential factors that shape user experience and incorporates them into controlled experiments, where human ratings are collected under varied configurations. From these studies we construct a QoE-oriented database and train predictive models that estimate perceived user experience from measurable system parameters. Our results demonstrate that QoNext not only enables proactive and fine-grained evaluation but also provides actionable guidance for productized services of optimizing foundation models in practice.

---

## 71. MolSpectLLM: A Molecular Foundation Model Bridging Spectroscopy, Molecule Elucidation, and 3D Structure Generation

**论文链接:** [http://arxiv.org/abs/2509.21861v1](http://arxiv.org/abs/2509.21861v1)

**作者:** Shuaike Shen, Jiaqing Xie, Zhuo Yang, Antong Zhang, Shuzhou Sun, Ben Gao, Tianfan Fu, Biqing Qi, Yuqiang Li

**发布时间:** 2025-09-26

### GPT解析

### 总结

这篇论文介绍了MolSpectLLM，一种统一实验光谱与分子3D结构的分子基础模型，在分子性质预测和分子设计方面表现出色，特别是在光谱相关任务上取得了最先进的性能。

### 背景

分子基础模型在分子性质预测和从头分子设计方面取得了显著进展，但大多数现有方法仅依赖于SMILES表示，忽略了实验光谱和3D结构信息这两个在真实场景中捕捉分子行为不可或缺的来源，限制了它们在立体化学、空间构象和实验验证等关键任务中的有效性。

### 目的

为了克服现有方法的局限性，研究者提出了一种统一实验光谱与分子3D结构的分子基础模型，以提高在需要考虑立体化学、空间构象和实验验证的任务中的性能。

### 方法

研究者提出了MolSpectLLM，一个基于Qwen2.5-7B预训练的分子基础模型，该模型明确建模分子光谱，将实验光谱与分子3D结构相结合。

### 主要发现

MolSpectLLM在光谱相关任务上实现了最先进的性能，在NMR、IR和MS基准测试中平均准确率达到0.53。在光谱分析任务上，获得了15.5%的序列准确率和41.7%的token准确率，显著优于大型通用LLM。此外，还能直接从SMILES或光谱输入生成准确的3D分子结构。

### 结论

MolSpectLLM通过结合实验光谱和分子3D结构信息，克服了现有分子基础模型的局限性，在光谱相关任务和分子设计任务上表现出色，为药物发现和其他需要考虑分子立体化学和空间构象的应用提供了有力工具。

### 翻译

分子基础模型的最新进展在分子性质预测和从头分子设计方面显示出卓越的性能，在药物发现和反应预测等领域有很好的应用前景。然而，大多数现有方法仅依赖于SMILES表示，忽略了实验光谱和3D结构信息这两个在真实场景中捕捉分子行为不可或缺的来源。这种局限性降低了它们在立体化学、空间构象和实验验证等关键任务中的有效性。为了克服这些挑战，我们提出了MolSpectLLM，这是一个基于Qwen2.5-7B预训练的分子基础模型，将实验光谱与分子3D结构统一起来。通过明确建模分子光谱，MolSpectLLM在光谱相关任务上实现了最先进的性能，在NMR、IR和MS基准测试中平均准确率为0.53。MolSpectLLM在光谱分析任务上也表现出色，在Spectra-to-SMILES上获得了15.5%的序列准确率和41.7%的token准确率，显著优于大型通用LLM。更重要的是，MolSpectLLM不仅在分子阐明任务上表现出色，还能直接从SMILES或光谱输入生成准确的3D分子结构，连接了光谱分析、分子阐明和分子设计。


### 论文摘要

Recent advances in molecular foundation models have shown impressive performance in molecular property prediction and de novo molecular design, with promising applications in areas such as drug discovery and reaction prediction. Nevertheless, most existing approaches rely exclusively on SMILES representations and overlook both experimental spectra and 3D structural information-two indispensable sources for capturing molecular behavior in real-world scenarios. This limitation reduces their effectiveness in tasks where stereochemistry, spatial conformation, and experimental validation are critical. To overcome these challenges, we propose MolSpectLLM, a molecular foundation model pretrained on Qwen2.5-7B that unifies experimental spectroscopy with molecular 3D structure. By explicitly modeling molecular spectra, MolSpectLLM achieves state-of-the-art performance on spectrum-related tasks, with an average accuracy of 0.53 across NMR, IR, and MS benchmarks. MolSpectLLM also shows strong performance on the spectra analysis task, obtaining 15.5% sequence accuracy and 41.7% token accuracy on Spectra-to-SMILES, substantially outperforming large general-purpose LLMs. More importantly, MolSpectLLM not only achieves strong performance on molecular elucidation tasks, but also generates accurate 3D molecular structures directly from SMILES or spectral inputs, bridging spectral analysis, molecular elucidation, and molecular design.

---

## 72. ChaosNexus: A Foundation Model for Universal Chaotic System Forecasting with Multi-scale Representations

**论文链接:** [http://arxiv.org/abs/2509.21802v1](http://arxiv.org/abs/2509.21802v1)

**作者:** Chang Liu, Bohao Zhao, Jingtao Ding, Yong Li

**发布时间:** 2025-09-26

### GPT解析

### 总结

本文提出了一种名为ChaosNexus的基础模型，用于解决混沌系统预测中的泛化问题，通过在多样化混沌动力学数据上预训练，实现了强大的零样本或少样本预测能力。

### 背景

准确预测混沌系统（如天气预测和流体动力学中普遍存在的系统）仍面临重大挑战，这些系统对初始条件敏感且观测数据稀缺，传统模型因针对特定系统训练而缺乏泛化能力。

### 目的

克服传统模型在数据有限和面对新系统时的泛化障碍，开发能够进行零样本或少样本预测的基础模型。

### 方法

提出ChaosNexus基础模型，在多样化混沌动力学语料库上预训练，采用名为ScaleFormer的新型多尺度架构并加入专家混合层，以捕捉通用模式和系统特定行为。

### 主要发现

ChaosNexus在合成和真实世界基准中展示最先进的零样本泛化能力，在9,000多个合成混沌系统测试中将长期吸引子统计保真度提高40%以上；在5天全球天气预报中实现低于1度的竞争性零样本平均误差；跨系统泛化源于训练系统多样性而非数据量。

### 结论

ChaosNexus通过多样化混沌系统预训练克服了传统模型的泛化限制，为混沌系统预测提供了新的解决方案，并确立了科学基础模型的指导原则。

### 翻译

准确预测混沌系统（在天气预测和流体动力学等领域普遍存在）仍然是一个重大的科学挑战。这些系统对初始条件固有的敏感性，加上观测数据的稀缺，严重限制了传统建模方法。由于这些模型通常针对特定系统进行训练，它们缺乏现实应用所需的泛化能力，这些应用需要在新的或数据有限的情况下进行强大的零样本或少样本预测。为了克服这一泛化障碍，我们提出了ChaosNexus，一种在多样化混沌动力学语料库上预训练的基础模型。ChaosNexus采用了一种名为ScaleFormer的新型多尺度架构，并加入了专家混合层，以捕捉通用模式和系统特定行为。该模型在合成和真实世界的基准测试中展示了最先进的零样本泛化能力。在包含9,000多个合成混沌系统的大规模测试平台上，与最先进的基线相比，它将长期吸引子统计的保真度提高了40%以上。这种强大的性能延伸到现实世界的应用中，具有出色的数据效率。例如，在5天全球天气预报中，ChaosNexus实现了低于1度的竞争性零样本平均误差，通过少样本微调进一步改善。此外，关于ChaosNexus扩展行为的实验为科学基础模型提供了指导原则：跨系统泛化源于训练系统的多样性，而非单纯的数据量。


### 论文摘要

Accurately forecasting chaotic systems, prevalent in domains such as weather prediction and fluid dynamics, remains a significant scientific challenge. The inherent sensitivity of these systems to initial conditions, coupled with a scarcity of observational data, severely constrains traditional modeling approaches. Since these models are typically trained for a specific system, they lack the generalization capacity necessary for real-world applications, which demand robust zero-shot or few-shot forecasting on novel or data-limited scenarios. To overcome this generalization barrier, we propose ChaosNexus, a foundation model pre-trained on a diverse corpus of chaotic dynamics. ChaosNexus employs a novel multi-scale architecture named ScaleFormer augmented with Mixture-of-Experts layers, to capture both universal patterns and system-specific behaviors. The model demonstrates state-of-the-art zero-shot generalization across both synthetic and real-world benchmarks. On a large-scale testbed comprising over 9,000 synthetic chaotic systems, it improves the fidelity of long-term attractor statistics by more than 40% compared to the leading baseline. This robust performance extends to real-world applications with exceptional data efficiency. For instance, in 5-day global weather forecasting, ChaosNexus achieves a competitive zero-shot mean error below 1 degree, a result that further improves with few-shot fine-tuning. Moreover, experiments on the scaling behavior of ChaosNexus provide a guiding principle for scientific foundation models: cross-system generalization stems from the diversity of training systems, rather than sheer data volume.

---

## 73. SynerGen: Contextualized Generative Recommender for Unified Search and Recommendation

**论文链接:** [http://arxiv.org/abs/2509.21777v1](http://arxiv.org/abs/2509.21777v1)

**作者:** Vianne R. Gao, Chen Xue, Marc Versage, Xie Zhou, Zhongruo Wang, Chao Li, Yeon Seonwoo, Nan Chen, Zhen Ge, Gourab Kundu, Weiqi Zhang, Tian Wang, Qingjun Cui, Trishul Chilimbi

**发布时间:** 2025-09-26

**备注:** Generative Recommender, Recommendation System, Information Retrieval

### GPT解析

### 总结

SynerGen是一种新的生成式推荐模型，通过单一的生成式主干网络同时支持个性化搜索和推荐，解决了现有模型在统一这两方面功能时的性能权衡问题。

### 背景

大规模推荐系统中的主流'检索后排序'管道存在架构分离导致的校准问题和工程开销大的问题。现有生成式序列模型通常只解决个性化搜索或无查询推荐中的一个方面，难以同时兼顾两者。

### 目的

引入SynerGen模型，提供一个单一的生成式主干网络，同时支持个性化搜索和推荐，并在检索和排序任务上都能表现出色。

### 方法

使用基于行为的序列进行训练，采用仅解码器的Transformer架构，通过InfoNCE进行联合优化用于检索，使用混合点对点损失函数用于排序，并提出了一种新型的时间感知旋转位置编码来整合时间信息到注意力机制中。

### 主要发现

SynerGen在广泛采用的推荐和搜索基准上相比强大的生成式推荐基线和联合搜索与推荐基线取得了显著改进，搜索的语义信号可以改善推荐，反之亦然。

### 结论

单一生成式基础模型在工业规模统一信息访问方面是可行的。

### 翻译

大规模推荐系统中的主流'检索后排序'管道由于其架构分离和不同的优化目标而存在校准不准和工程开销大的问题。虽然最近的生成式序列模型通过自回归生成排序项目在统一检索和排序方面显示出潜力，但现有解决方案通常只处理个性化搜索或无查询推荐，在尝试统一两者时往往表现出性能权衡。我们引入了SynerGen，这是一种新颖的生成式推荐模型，通过为个性化搜索和推荐提供单一的生成式主干来弥合这一关键差距，同时在检索和排序任务上表现出色。我们的仅解码器Transformer基于行为序列进行训练，利用InfoNCE进行检索的联合优化，以及用于排序的混合点对点损失函数，允许搜索的语义信号改善推荐，反之亦然。我们还提出了一种新颖的时间感知旋转位置编码，将时间信息有效地整合到注意力机制中。与强大的生成式推荐基线和联合搜索与推荐基线相比，SynerGen在广泛采用的推荐和搜索基准上取得了显著改进。这项工作证明了单一生成式基础模型在工业规模统一信息访问方面的可行性。


### 论文摘要

The dominant retrieve-then-rank pipeline in large-scale recommender systems suffers from mis-calibration and engineering overhead due to its architectural split and differing optimization objectives. While recent generative sequence models have shown promise in unifying retrieval and ranking by auto-regressively generating ranked items, existing solutions typically address either personalized search or query-free recommendation, often exhibiting performance trade-offs when attempting to unify both. We introduce \textit{SynerGen}, a novel generative recommender model that bridges this critical gap by providing a single generative backbone for both personalized search and recommendation, while simultaneously excelling at retrieval and ranking tasks. Trained on behavioral sequences, our decoder-only Transformer leverages joint optimization with InfoNCE for retrieval and a hybrid pointwise-pairwise loss for ranking, allowing semantic signals from search to improve recommendation and vice versa. We also propose a novel time-aware rotary positional embedding to effectively incorporate time information into the attention mechanism. \textit{SynerGen} achieves significant improvements on widely adopted recommendation and search benchmarks compared to strong generative recommender and joint search and recommendation baselines. This work demonstrates the viability of a single generative foundation model for industrial-scale unified information access.

---

## 74. UniVid: Unifying Vision Tasks with Pre-trained Video Generation Models

**论文链接:** [http://arxiv.org/abs/2509.21760v1](http://arxiv.org/abs/2509.21760v1)

**作者:** Lan Chen, Yuchao Gu, Qi Mao

**发布时间:** 2025-09-26

### GPT解析

### 总结

论文提出了UniVid框架，通过微调视频扩散变换器处理多种视觉任务，无需任务特定修改。研究探索了预训练视频生成模型适应多样化视觉任务的能力，发现其具有良好的跨模态推理和跨源任务泛化能力，并能通过简单反转视觉句子顺序切换理解和生成任务。

### 背景

大型语言模型成功将多样化语言任务统一在单一生成框架内。受此启发，大型视觉模型（LVM）将此范式扩展到视觉领域，通过组织任务为顺序视觉句子，使用视觉提示作为上下文。然而，此类方法需要跨模态和来源的任务特定预训练，成本高昂且限制扩展性。

### 目的

探索预训练视频生成模型能否适应多样化的图像和视频任务，提供一个更统一且可扩展的替代方案，解决现有方法需要高昂成本预训练的问题。

### 方法

提出UniVid框架，微调视频扩散变换器处理各种视觉任务，无需任务特定修改。任务表示为视觉句子，上下文序列定义任务和期望输出模态。从两个角度评估泛化能力：(1) 使用图像和视频组成的上下文进行跨模态推理；(2) 从自然数据到注释数据的跨源任务，无需多源预训练。

### 主要发现

尽管仅使用自然视频数据训练，UniVid在跨模态推理和跨源任务设置中均表现良好。通过简单反转视觉句子顺序，理解和生成任务可以轻松切换。

### 结论

预训练视频生成模型可作为视觉建模的可扩展和统一基础，具有处理多样化视觉任务的潜力。研究代码将在https://github.com/CUC-MIPG/UniVid发布。

### 翻译

大型语言模型在大量语料库上训练成功，将多样化的语言任务统一在一个生成框架内。受此启发，近期工作如大型视觉模型（LVM）将这一范式扩展到视觉领域，通过将任务组织成顺序的视觉句子，其中视觉提示作为指导输出的上下文。然而，这种建模需要跨模态和来源的任务特定预训练，成本高昂且限制了扩展到未见任务的能力。鉴于预训练的视频生成模型固有地捕获时间序列依赖关系，我们探索了一个更统一且可扩展的替代方案：预训练的视频生成模型能否适应多样的图像和视频任务？为回答这个问题，我们提出了UniVid，这是一个微调视频扩散变换器以处理各种视觉任务的框架，无需针对特定任务进行修改。任务表示为视觉句子，其中上下文序列定义了任务和期望的输出模态。我们从两个角度评估UniVid的泛化能力：(1) 使用由图像和视频组成的上下文进行跨模态推理，扩展了LVM的单模态设置；(2) 从自然数据到注释数据的跨源任务，无需多源预训练。尽管仅使用自然视频数据训练，UniVid在这两种设置中都表现良好。值得注意的是，通过简单反转视觉句子顺序，理解和生成任务可以轻松切换。这些发现突显了预训练视频生成模型作为视觉建模的可扩展和统一基础的潜力。我们的代码将在https://github.com/CUC-MIPG/UniVid发布。


### 论文摘要

Large language models, trained on extensive corpora, successfully unify diverse linguistic tasks within a single generative framework. Inspired by this, recent works like Large Vision Model (LVM) extend this paradigm to vision by organizing tasks into sequential visual sentences, where visual prompts serve as the context to guide outputs. However, such modeling requires task-specific pre-training across modalities and sources, which is costly and limits scalability to unseen tasks. Given that pre-trained video generation models inherently capture temporal sequence dependencies, we explore a more unified and scalable alternative: can a pre-trained video generation model adapt to diverse image and video tasks? To answer this, we propose UniVid, a framework that fine-tunes a video diffusion transformer to handle various vision tasks without task-specific modifications. Tasks are represented as visual sentences, where the context sequence defines both the task and the expected output modality. We evaluate the generalization of UniVid from two perspectives: (1) cross-modal inference with contexts composed of both images and videos, extending beyond LVM's uni-modal setting; (2) cross-source tasks from natural to annotated data, without multi-source pre-training. Despite being trained solely on natural video data, UniVid generalizes well in both settings. Notably, understanding and generation tasks can easily switch by simply reversing the visual sentence order in this paradigm. These findings highlight the potential of pre-trained video generation models to serve as a scalable and unified foundation for vision modeling. Our code will be released at https://github.com/CUC-MIPG/UniVid.

---

## 75. Noise-to-Notes: Diffusion-based Generation and Refinement for Automatic Drum Transcription

**论文链接:** [http://arxiv.org/abs/2509.21739v1](http://arxiv.org/abs/2509.21739v1)

**作者:** Michael Yeung, Keisuke Toyama, Toya Teramoto, Shusuke Takahashi, Tamaki Kojima

**发布时间:** 2025-09-26

### GPT解析

### 总结

本文提出了Noise-to-Notes (N2N)框架，将自动鼓转录重新定义为条件生成任务，利用扩散建模将音频条件的高斯噪声转换为带有速度的鼓事件，并整合音乐基础模型特征提高性能。

### 背景

传统的自动鼓转录(ADT)被表述为判别性任务，用于从音频频谱图中预测鼓事件。

### 目的

重新定义ADT为条件生成任务，开发一种新的框架来生成鼓事件及其相关速度。

### 方法

利用扩散建模技术将基于音频条件的高斯噪声转换为鼓事件；引入退火伪Huber损失处理二元起始点和连续速度值的生成挑战；整合从音乐基础模型(MFMs)中提取的特征以增强低层频谱图特征和提高鲁棒性。

### 主要发现

生成式扩散方法提供了灵活的速度-准确性权衡和强大的修复能力；包含MFM特征显著提高了系统对域外鼓音频的鲁棒性；N2N在多个ADT基准测试上建立了新的最先进性能。

### 结论

通过重新定义ADT为条件生成任务并利用扩散建模，结合音乐基础模型特征，N2N框架实现了更优的性能和鲁棒性。

### 翻译

自动鼓转录(ADT)传统上被表述为从音频频谱图中预测鼓事件的判别性任务。在本工作中，我们将ADT重新定义为条件生成任务，并引入Noise-to-Notes (N2N)框架，利用扩散建模将基于音频条件的高斯噪声转换为带有相关速度的鼓事件。这种生成式扩散方法提供了明显的优势，包括灵活的速度-准确性权衡和强大的修复能力。然而，二元起始点和连续速度值的生成对扩散模型构成挑战，为此我们引入退火伪Huber损失以实现有效的联合优化。最后，为了增强低层频谱图特征，我们提议整合从音乐基础模型(MFMs)中提取的特征，这些特征捕获高层语义信息并增强对域外鼓音频的鲁棒性。实验结果表明，包含MFM特征显著提高了鲁棒性，N2N在多个ADT基准测试上建立了新的最先进性能。


### 论文摘要

Automatic drum transcription (ADT) is traditionally formulated as a discriminative task to predict drum events from audio spectrograms. In this work, we redefine ADT as a conditional generative task and introduce Noise-to-Notes (N2N), a framework leveraging diffusion modeling to transform audio-conditioned Gaussian noise into drum events with associated velocities. This generative diffusion approach offers distinct advantages, including a flexible speed-accuracy trade-off and strong inpainting capabilities. However, the generation of binary onset and continuous velocity values presents a challenge for diffusion models, and to overcome this, we introduce an Annealed Pseudo-Huber loss to facilitate effective joint optimization. Finally, to augment low-level spectrogram features, we propose incorporating features extracted from music foundation models (MFMs), which capture high-level semantic information and enhance robustness to out-of-domain drum audio. Experimental results demonstrate that including MFM features significantly improves robustness and N2N establishes a new state-of-the-art performance across multiple ADT benchmarks.

---

## 76. Frustratingly Easy Zero-Day Audio DeepFake Detection via Retrieval Augmentation and Profile Matching

**论文链接:** [http://arxiv.org/abs/2509.21728v1](http://arxiv.org/abs/2509.21728v1)

**作者:** Xuechen Liu, Xin Wang, Junichi Yamagishi

**发布时间:** 2025-09-26

### GPT解析

### 总结

本研究提出了一种基于知识表示、检索增强和语音档案匹配的零日音频深度伪造检测的无训练框架，实现了与微调模型相当的检测性能，无需额外训练。

### 背景

现代音频深度伪造检测器使用基础模型和大型训练数据集已取得有前景的检测性能，但难以应对零日攻击（由新合成方法生成的音频样本）。

### 目的

开发一种无需微调的零日音频深度伪造检测方法，以应对需要快速响应的场景。

### 方法

提出基于知识表示、检索增强和语音档案匹配的无训练框架，并设计了简单的知识检索和集成方法。

### 主要发现

所提出的方法在DeepFake-Eval-2024上实现了与微调模型相当的性能，无需任何额外的模型级训练；检索池大小和语音档案属性与系统功效相关。

### 结论

基于知识表示的无训练框架可以有效应对零日音频深度伪造攻击，避免了传统方法需要微调的局限性。

### 翻译

使用基础模型和大型训练数据集的现代音频深度伪造检测器已取得有前景的检测性能。然而，它们难以应对零日攻击，即由新的合成方法生成的音频样本，这些方法在训练数据中并未出现过。针对此类攻击的传统方法需要对检测器进行微调，这在需要快速响应的情况下可能会成为问题。本研究提出了一种基于知识表示、检索增强和语音档案匹配的零日音频深度伪造检测的无训练框架。基于该框架，我们提出了简单而有效的知识检索和集成方法，这些方法在DeepFake-Eval-2024上实现了与微调模型相当的性能，而无需任何额外的模型级训练。我们还对检索池大小和语音档案属性进行了消融研究，验证了它们与系统功效的相关性。


### 论文摘要

Modern audio deepfake detectors using foundation models and large training datasets have achieved promising detection performance. However, they struggle with zero-day attacks, where the audio samples are generated by novel synthesis methods that models have not seen from reigning training data. Conventional approaches against such attacks require fine-tuning the detectors, which can be problematic when prompt response is required. This study introduces a training-free framework for zero-day audio deepfake detection based on knowledge representations, retrieval augmentation, and voice profile matching. Based on the framework, we propose simple yet effective knowledge retrieval and ensemble methods that achieve performance comparable to fine-tuned models on DeepFake-Eval-2024, without any additional model-wise training. We also conduct ablation studies on retrieval pool size and voice profile attributes, validating their relevance to the system efficacy.

---

## 77. On the Status of Foundation Models for SAR Imagery

**论文链接:** [http://arxiv.org/abs/2509.21722v1](http://arxiv.org/abs/2509.21722v1)

**作者:** Nathan Inkawhich

**发布时间:** 2025-09-26

### GPT解析

### 总结

本研究探讨了基础AI/ML模型在合成孔径雷达（SAR）目标识别任务中的可行性，通过自监督学习技术对现有模型进行微调，成功提升了SAR目标识别性能。

### 背景

自然图像领域AI/ML技术取得巨大进展，前沿实验室使用大规模数据集训练大型模型，这些自监督学习模型能够适应下游任务，对分布偏移具有鲁棒性，特征可迁移性强。

### 目的

将自然图像领域的基础模型技术应用于SAR领域，提高SAR目标识别的性能和效率。

### 方法

测试当前最强大的视觉基础模型（DINOv2、DINOv3和PE-Core），观察其在SAR特征提取上的不足；使用SAR数据对公开SSL模型进行自监督微调，训练AFRL-DINOv2模型；分析不同骨干网络和下游任务适应方案的性能权衡；评估模型克服下游环境挑战的能力。

### 主要发现

现有视觉基础模型在提取SAR语义目标特征方面存在局限；使用SAR数据对SSL模型进行自监督微调是可行路径；AFRL-DINOv2模型为SAR基础模型设定了新的最先进水平，显著优于当前最佳SAR领域模型SARATR-X。

### 结论

尽管取得积极结果，SAR基础模型发展仍有很大空间；该研究将为未来SAR基础模型构建者提供参考和启发。

### 翻译

本研究探讨了基础AI/ML模型在合成孔径雷达（SAR）目标识别任务中的可行性。我们受到更广泛社区，特别是在自然图像领域取得的巨大进展的启发，前沿实验室使用前所未有的计算预算在网络规模数据集上训练大型模型。很明显，这些通常使用自监督学习（SSL）训练的模型将改变我们为目标识别任务开发AI/ML解决方案的方式 - 它们可以用非常有限的标记数据进行下游适应，对多种形式的分布偏移更具鲁棒性，并且它们的特征开箱即可高度迁移。由于这些原因以及更多，我们受到启发将这项技术应用到SAR领域。在我们的实验中，我们首先使用当今最强大的视觉基础模型进行测试，包括DINOv2、DINOv3和PE-Core，并观察到它们在即用状态下提取语义上有意义的区分性SAR目标特征方面的不足。然后，我们展示了使用SAR数据对公开可用的SSL模型进行自监督微调是一条可行的前进路径，通过训练几个AFRL-DINOv2模型，为SAR基础模型设定了新的最先进水平，显著优于当今最好的SAR领域模型SARATR-X。我们的实验进一步分析了使用不同骨干网络与不同下游任务适应方案的性能权衡，并监控每个模型克服下游环境挑战的能力（例如，扩展操作条件和少量标记数据）。我们希望这项工作将为未来的SAR基础模型构建者提供信息和启发，因为尽管我们的结果是积极的，但我们仍然有很长的路要走。


### 论文摘要

In this work we investigate the viability of foundational AI/ML models for Synthetic Aperture Radar (SAR) object recognition tasks. We are inspired by the tremendous progress being made in the wider community, particularly in the natural image domain where frontier labs are training huge models on web-scale datasets with unprecedented computing budgets. It has become clear that these models, often trained with Self-Supervised Learning (SSL), will transform how we develop AI/ML solutions for object recognition tasks - they can be adapted downstream with very limited labeled data, they are more robust to many forms of distribution shift, and their features are highly transferable out-of-the-box. For these reasons and more, we are motivated to apply this technology to the SAR domain. In our experiments we first run tests with today's most powerful visual foundational models, including DINOv2, DINOv3 and PE-Core and observe their shortcomings at extracting semantically-interesting discriminative SAR target features when used off-the-shelf. We then show that Self-Supervised finetuning of publicly available SSL models with SAR data is a viable path forward by training several AFRL-DINOv2s and setting a new state-of-the-art for SAR foundation models, significantly outperforming today's best SAR-domain model SARATR-X. Our experiments further analyze the performance trade-off of using different backbones with different downstream task-adaptation recipes, and we monitor each model's ability to overcome challenges within the downstream environments (e.g., extended operating conditions and low amounts of labeled data). We hope this work will inform and inspire future SAR foundation model builders, because despite our positive results, we still have a long way to go.

---

## 78. Wav2Arrest 2.0: Long-Horizon Cardiac Arrest Prediction with Time-to-Event Modeling, Identity-Invariance, and Pseudo-Lab Alignment

**论文链接:** [http://arxiv.org/abs/2509.21695v1](http://arxiv.org/abs/2509.21695v1)

**作者:** Saurabh Kataria, Davood Fattahi, Minxiao Wang, Ran Xiao, Matthew Clark, Timothy Ruchti, Mark Mai, Xiao Hu

**发布时间:** 2025-09-25

**备注:** Submitted to BPSC

### GPT解析

### 总结

本研究提出三种改进方法，通过使用最少辅助信息增强基于PPG的心脏骤停预测系统，独立将24小时时间平均AUC从0.74提升至0.78-0.80范围，主要改善长时预测性能。

### 背景

高频生理波形模态能提供对患者状态的深入、实时洞察。基于光电容积描记法(PPG)的生理基础模型(如PPG-GPT)已被证明可以预测包括心脏骤停在内的关键事件，但其强大表示能力在下游数据/标签稀缺时未得到充分利用。

### 目的

通过使用最少的辅助信息来改进仅基于PPG的心脏骤停预测系统，提高预测性能，特别是在长时预测方面。

### 方法

1) 事件时间建模，通过简单回归到事件发生时间或细粒度离散生存建模；2) 使模型学习CA聚焦特征，通过训练大规模匿名生物识别模型(p-vector)并对抗性使用来解耦可能导致过拟合的线索；3) 回归预训练辅助估计器网络生成的伪标签值，解决真实血液实验室测量数据稀少的问题。

### 主要发现

提出的改进方法可独立将24小时时间平均AUC从0.74提升至0.78-0.80范围，主要改善长时预测性能，在事件附近最小程度降低性能，推动早期预警系统研究。

### 结论

采用多任务公式诊断出竞争损失之间的高梯度冲突率，并通过PCGrad优化技术缓解了这一问题，有效提高了心脏骤停预测系统的性能。

### 翻译

高频生理波形模态能提供对患者状态的深入、实时洞察。最近，基于光电容积描记法(PPG)的生理基础模型(如PPG-GPT)已被证明可以预测包括心脏骤停在内的关键事件。然而，它们的强大表示能力仍未得到充分利用，特别是当下游数据/标签稀缺时。我们提出三种正交改进，通过使用最少的辅助信息来改进仅基于PPG的CA系统。首先，我们建议使用事件时间建模，通过简单回归到事件发生时间或追求细粒度离散生存建模。其次，我们鼓励模型学习CA聚焦特征，通过使它们对患者身份不变性。这是通过首先训练最大规模的匿名生物识别识别模型(称为p-vector)实现的，然后对抗性地使用它来解耦可能导致通过记忆过拟合的线索。第三，我们提出对预训练辅助估计器网络生成的伪标签值进行回归。这至关重要，因为真正的血液实验室测量(如乳酸、钠、肌钙蛋白和钾)收集稀少。通过零样本预测，辅助网络可以丰富心脏骤停波形标签并生成伪连续估计作为目标。我们的提案可以独立将24小时时间平均AUC从0.74提高到0.78-0.80范围。我们主要在更长的时间范围内改进，在事件附近最小程度降低性能，从而推动早期预警系统研究。最后，我们采用多任务公式并诊断出竞争损失之间的高梯度冲突率，我们通过PCGrad优化技术缓解了这一问题。


### 论文摘要

High-frequency physiological waveform modality offers deep, real-time insights into patient status. Recently, physiological foundation models based on Photoplethysmography (PPG), such as PPG-GPT, have been shown to predict critical events, including Cardiac Arrest (CA). However, their powerful representation still needs to be leveraged suitably, especially when the downstream data/label is scarce. We offer three orthogonal improvements to improve PPG-only CA systems by using minimal auxiliary information. First, we propose to use time-to-event modeling, either through simple regression to the event onset time or by pursuing fine-grained discrete survival modeling. Second, we encourage the model to learn CA-focused features by making them patient-identity invariant. This is achieved by first training the largest-scale de-identified biometric identification model, referred to as the p-vector, and subsequently using it adversarially to deconfound cues, such as person identity, that may cause overfitting through memorization. Third, we propose regression on the pseudo-lab values generated by pre-trained auxiliary estimator networks. This is crucial since true blood lab measurements, such as lactate, sodium, troponin, and potassium, are collected sparingly. Via zero-shot prediction, the auxiliary networks can enrich cardiac arrest waveform labels and generate pseudo-continuous estimates as targets. Our proposals can independently improve the 24-hour time-averaged AUC from the 0.74 to the 0.78-0.80 range. We primarily improve over longer time horizons with minimal degradation near the event, thus pushing the Early Warning System research. Finally, we pursue multi-task formulation and diagnose it with a high gradient conflict rate among competing losses, which we alleviate via the PCGrad optimization technique.

---

## 79. Scalable Foundation Interatomic Potentials via Message-Passing Pruning and Graph Partitioning

**论文链接:** [http://arxiv.org/abs/2509.21694v1](http://arxiv.org/abs/2509.21694v1)

**作者:** Lingyu Kong, Jaeheon Shim, Guoxiang Hu, Victor Fung

**发布时间:** 2025-09-25

### GPT解析

### 总结

本文提出了一种加速和扩展原子基础模型(AFMs)的通用工作流程，通过移除低贡献的消息传递层进行模型剪枝，显著减少参数数量同时保持准确性和数据效率，并采用图分区GPU分布式策略实现大规模模拟。

### 背景

原子基础模型(AFMs)作为准确的原子间势能很有前景，能够实现接近量子力学精度的数据高效分子动力学模拟。然而，AFMs在推理速度和内存占用方面明显劣于传统原子间势能，这是由于需要在大规模预训练数据集中捕获多种化学和结构模式，需要深度、参数丰富的模型架构。这些缺点限制了AFMs在扩展时间和空间尺度分子动力学模拟中的实际应用。

### 目的

提出一种通用工作流程，用于加速和扩展包含消息传递架构的原子基础模型(AFMs)，解决其在推理速度和内存占用方面的不足，使其更适合大规模分子动力学模拟。

### 方法

从AFM主干中移除低贡献的消息传递层作为有效剪枝方法，显著减少参数数量同时保留准确性和数据效率。对剪枝后的模型采用图分区、GPU分布式策略，在AFM微调平台MatterTune中实现和演示，支持单GPU和多GPU上的百万原子级别模拟。

### 主要发现

移除低贡献的消息传递层可显著减少AFMs的参数数量，同时保持其准确性和数据效率。该方法支持在单GPU和多GPU上进行百万原子级别的模拟，能够实现具有AFM级精度的纳秒时间尺度特定任务大规模模拟。

### 结论

所提出的加速和扩展方法使AFMs更适合大规模分子动力学模拟，克服了AFMs在计算资源和内存使用方面的限制，使AFMs能够在扩展的时间和空间尺度上实现实际应用。

### 翻译

原子基础模型(AFMs)作为准确的原子间势能具有巨大潜力，并已实现了接近量子力学精度的数据高效分子动力学模拟。然而，由于需要在预训练数据集中捕获广泛的化学和结构模式，这需要深度且参数丰富的模型架构，导致AFMs在推理速度上明显慢于传统原子间势能，且内存占用更大。这些缺点目前限制了AFMs在扩展时间和空间尺度的分子动力学(MD)模拟中的实际应用。为解决这一问题，我们提出了一种包含消息传递架构的AFMs的加速和扩展通用工作流程。我们发现，从AFM主干中移除低贡献的消息传递层是一种有效的剪枝方法，显著减少了参数数量，同时保留了AFMs的准确性和数据效率。剪枝后，这些模型通过图分区、GPU分布式策略更适合大规模模拟，我们在AFM微调平台MatterTune中实现了并演示了这一策略。我们表明该方法支持在单GPU和多GPU上进行百万原子级别的模拟，并能够在纳秒时间尺度上实现具有AFM级精度的特定任务大规模模拟。


### 论文摘要

Atomistic foundation models (AFMs) have great promise as accurate interatomic potentials, and have enabled data-efficient molecular dynamics simulations with near quantum mechanical accuracy. However, AFMs remain markedly slower at inference and are far more memory-intensive than conventional interatomic potentials, due to the need to capture a wide range of chemical and structural motifs in pre-training datasets requiring deep, parameter-rich model architectures. These deficiencies currently limit the practical use of AFMs in molecular dynamics (MD) simulations at extended temporal and spatial scales. To address this problem, we propose a general workflow for accelerating and scaling AFMs containing message-passing architectures. We find that removing low-contribution message-passing layers from AFM backbones serves as an effective pruning method, significantly reducing the parameter count while preserving the accuracy and data-efficiency of AFMs. Once pruned, these models become more accessible for large scale simulations via a graph-partitioned, GPU-distributed strategy, which we implement and demonstrate within the AFM fine-tuning platform MatterTune. We show that this approach supports million-atom simulations on both single and multiple GPUs, and enables task-specific large-scale simulations at nanosecond timescales with AFM-level accuracy.

---

## 80. SlotFM: A Motion Foundation Model with Slot Attention for Diverse Downstream Tasks

**论文链接:** [http://arxiv.org/abs/2509.21673v1](http://arxiv.org/abs/2509.21673v1)

**作者:** Junyong Park, Oron Levy, Rebecca Adaimi, Asaf Liberman, Gierad Laput, Abdelkareem Bedri

**发布时间:** 2025-09-25

### GPT解析

### 总结

SlotFM是一种新型加速度计基础模型，通过时间-频率槽注意力机制处理原始信号，生成多个捕捉不同信号成分的小嵌入，在16个下游任务上展现出强大泛化能力，平均性能提升4.5%。

### 背景

可穿戴加速度计被广泛应用于手势识别、步态分析和运动监测等领域。然而，现有基础模型主要专注于分类常见日常活动，限制了它们在依赖其他信号特征的任务中的适用性。

### 目的

开发一种能够泛化到各种下游任务的加速度计基础模型，扩展基础模型在更广泛应用中的适用性。

### 方法

提出SlotFM，使用时间-频率槽注意力技术处理原始信号的时间频率表示。该方法生成多个小的嵌入（槽），每个槽捕捉不同信号成分，使任务特定头部能关注最相关数据部分。同时引入两个损失正则化器，捕捉局部结构和频率模式，改进细粒度细节重构并帮助嵌入保留任务相关信息。

### 主要发现

在16个超越标准人类活动识别的分类和回归下游任务上评估，SlotFM在13个任务上优于现有自监督方法，在其余任务上实现与最佳方法相当的结果。平均性能提升4.5%，展示了传感基础模型的强大泛化能力。

### 结论

SlotFM通过创新的时间-频率槽注意力和损失正则化技术，成功实现了对多种下游任务的有效泛化，显著提升了传感基础模型的性能和应用范围。

### 翻译

可穿戴加速度计被广泛应用于各种应用，如手势识别、步态分析和运动监测。然而，大多数现有的基础模型主要专注于分类常见的日常活动，如移动和锻炼，这限制了它们在依赖其他信号特征的更广泛任务中的适用性。我们提出了SlotFM，这是一种能够泛化到各种下游任务的加速度计基础模型。SlotFM使用时间-频率槽注意力，这是槽注意力的扩展，能够处理原始信号的时间表示和频率表示。它生成多个小的嵌入（槽），每个槽捕捉不同的信号成分，使任务特定的头部能够关注数据中最相关的部分。我们还引入了两个损失正则化器，捕捉局部结构和频率模式，这些正则化器改进了细粒度细节的重构，并帮助嵌入保留任务相关信息。我们在16个超越标准人类活动识别的分类和回归下游任务上评估了SlotFM。在这些任务中，它在13个任务上优于现有的自监督方法，在其余任务上实现了与最佳方法相当的结果。平均而言，我们的方法实现了4.5%的性能提升，展示了传感基础模型的强大泛化能力。


### 论文摘要

Wearable accelerometers are used for a wide range of applications, such as gesture recognition, gait analysis, and sports monitoring. Yet most existing foundation models focus primarily on classifying common daily activities such as locomotion and exercise, limiting their applicability to the broader range of tasks that rely on other signal characteristics. We present SlotFM, an accelerometer foundation model that generalizes across diverse downstream tasks. SlotFM uses Time-Frequency Slot Attention, an extension of Slot Attention that processes both time and frequency representations of the raw signals. It generates multiple small embeddings (slots), each capturing different signal components, enabling task-specific heads to focus on the most relevant parts of the data. We also introduce two loss regularizers that capture local structure and frequency patterns, which improve reconstruction of fine-grained details and helps the embeddings preserve task-relevant information. We evaluate SlotFM on 16 classification and regression downstream tasks that extend beyond standard human activity recognition. It outperforms existing self-supervised approaches on 13 of these tasks and achieves comparable results to the best performing approaches on the remaining tasks. On average, our method yields a 4.5% performance gain, demonstrating strong generalization for sensing foundation models.

---

## 81. Neuroprobe: Evaluating Intracranial Brain Responses to Naturalistic Stimuli

**论文链接:** [http://arxiv.org/abs/2509.21671v1](http://arxiv.org/abs/2509.21671v1)

**作者:** Andrii Zahorodnii, Christopher Wang, Bennett Stankovits, Charikleia Moraitaki, Geeling Chau, Andrei Barbu, Boris Katz, Ila R Fiete

**发布时间:** 2025-09-25

**备注:** 31 pages, 7 main figures

### GPT解析

### 总结

Neuroprobe是一个用于研究大脑多模态语言处理的解码任务套件，基于BrainTreebank数据集构建，包含10名受试者40小时的iEEG记录。它既可作为神经科学研究的资源，也可作为iEEG基础模型的评估框架。

### 背景

高分辨率神经数据集为下一代脑机接口和神经治疗提供了基础模型，但社区需要严格的基准来区分竞争性的建模方法，而目前对于颅内脑电图(iEEG)记录还没有标准化的评估框架。

### 目的

解决iEEG评估框架缺失的差距，提出Neuroprobe作为研究大脑多模态语言处理的解码任务套件，并为iEEG基础模型提供严格的评估框架。

### 方法

Neuroprobe基于BrainTreebank数据集构建，该数据集包含10名人类受试者在观看自然电影任务中40小时的iEEG记录。与头皮脑电图不同，颅内脑电图需要侵入性手术来植入电极，直接从大脑记录神经活动，信号失真最小。Neuroprobe设计注重计算效率和易用性，代码公开可用并维护公开排行榜。

### 主要发现

Neuroprobe有两个关键功能：1)作为神经科学研究资源，通过高时间和空间分辨率确定大脑中语言处理各方面计算的时间和位置；2)作为iEEG基础模型的评估框架。研究发现线性基线在许多任务上表现优于前沿基础模型。

### 结论

Neuroprobe代码公开可用并维护公开排行榜，旨在促进iEEG基础模型领域的快速进展。

### 翻译

高分辨率神经数据集为下一代脑机接口和神经治疗提供了基础模型。社区需要严格的基准来区分竞争性的建模方法，但目前对于颅内脑电图(iEEG)记录还没有标准化的评估框架。为解决这一差距，我们提出了Neuroprobe：一套用于研究大脑多模态语言处理的解码任务。与头皮脑电图不同，颅内脑电图需要侵入性手术来植入电极，直接从大脑记录神经活动，信号失真最小。Neuroprobe基于BrainTreebank数据集构建，该数据集包含10名人类受试者在观看自然电影任务中40小时的iEEG记录。Neuroprobe有两个关键功能。首先，它是一个可以挖掘神经科学见解的资源。其高时间和空间分辨率允许研究人员通过测量所有电极位置上每个特征的解码能力，系统性地确定大脑中每个语言处理方面计算的时间和位置。利用Neuroprobe，我们以纯数据驱动的方式可视化了信息如何从颞上流向前额叶皮层，以及从简单听觉特征到更复杂语言特征的进展。其次，随着领域向神经基础模型发展，Neuroprobe为竞争性架构和训练协议的比较提供了严格框架。我们发现线性基线出乎意料地强大，在许多任务上击败了前沿基础模型。Neuroprobe设计注重计算效率和易用性。我们公开Neuroprobe的代码并维护一个公开排行榜，旨在促进iEEG基础模型领域的快速进展，网址为https://neuroprobe.dev/


### 论文摘要

High-resolution neural datasets enable foundation models for the next generation of brain-computer interfaces and neurological treatments. The community requires rigorous benchmarks to discriminate between competing modeling approaches, yet no standardized evaluation frameworks exist for intracranial EEG (iEEG) recordings. To address this gap, we present Neuroprobe: a suite of decoding tasks for studying multi-modal language processing in the brain. Unlike scalp EEG, intracranial EEG requires invasive surgery to implant electrodes that record neural activity directly from the brain with minimal signal distortion. Neuroprobe is built on the BrainTreebank dataset, which consists of 40 hours of iEEG recordings from 10 human subjects performing a naturalistic movie viewing task. Neuroprobe serves two critical functions. First, it is a mine from which neuroscience insights can be drawn. Its high temporal and spatial resolution allows researchers to systematically determine when and where computations for each aspect of language processing occur in the brain by measuring the decodability of each feature across time and all electrode locations. Using Neuroprobe, we visualize how information flows from the superior temporal gyrus to the prefrontal cortex, and the progression from simple auditory features to more complex language features in a purely data-driven manner. Second, as the field moves toward neural foundation models, Neuroprobe provides a rigorous framework for comparing competing architectures and training protocols. We found that the linear baseline is surprisingly strong, beating frontier foundation models on many tasks. Neuroprobe is designed with computational efficiency and ease of use in mind. We make the code for Neuroprobe openly available and maintain a public leaderboard, aiming to enable rapid progress in the field of iEEG foundation models, at https://neuroprobe.dev/

---

## 82. MORPH: Shape-agnostic PDE Foundation Models

**论文链接:** [http://arxiv.org/abs/2509.21670v1](http://arxiv.org/abs/2509.21670v1)

**作者:** Mahindra Singh Rautela, Alexander Most, Siddharth Mansingh, Bradley C. Love, Ayan Biswas, Diane Oyen, Earl Lawrence

**发布时间:** 2025-09-25

### GPT解析

### 总结

介绍了一种名为MORPH的偏微分方程基础模型，它基于卷积视觉变压器构建，能够处理异构时空数据，并在多种下游任务中表现出色。

### 背景

科学机器学习需要处理异构、多模态的科学观测数据，而现有模型在处理不同维度、分辨率和场类型的数据时存在局限性。

### 目的

开发一种形状无关的自回归基础模型，能够无缝处理不同维度(1D-3D)、不同分辨率的异构时空数据集，以及包含标量和矢量分量的多场数据。

### 方法

构建了结合三个关键组件的MORPH架构：(1)分量级卷积处理标量和矢量通道；(2)场间交叉注意力建模不同物理场间的信息传播；(3)轴向注意力减少计算负担同时保持表达能力。通过全模型微调和低秩适配器进行迁移学习。

### 主要发现

MORPH在零样本和全样本泛化方面都优于从头开始训练的模型，在大量评估中匹配或超越了强大的基线和最新最先进模型。

### 结论

MORPH提供了一个灵活且强大的主干，用于从科学观测的异构性和多模态性中学习，为可扩展和数据高效的科学机器学习铺平了道路。

### 翻译

我们介绍了MORPH，这是一种与形状无关的自回归基础模型，用于偏微分方程(PDEs)。MORPH基于卷积视觉变压器主干构建，能够无缝处理不同维度(1D-3D)、不同分辨率的异构时空数据集，以及包含标量和矢量分量的多场数据。该架构结合了(1)分量级卷积，联合处理标量和矢量通道以捕获局部交互；(2)场间交叉注意力，建模和选择性传播不同物理场之间的信息；(3)轴向注意力，沿单个空间和时间轴分解完整的时空自注意力，以减少计算负担同时保持表达能力。我们在多样化的异构PDE数据集上预训练了多个模型变体，并评估了对一系列下游预测任务的迁移能力。通过全模型微调和参数高效的低秩适配器(LoRA)，MORPH在零样本和全样本泛化方面都优于从头开始训练的模型。在大量评估中，MORPH匹配或超越了强大的基线和最新的最先进模型。这些能力共同展示了一个灵活且强大的主干，用于从科学观测的异构性和多模态性中学习，为可扩展和数据高效的科学机器学习铺平了道路。


### 论文摘要

We introduce MORPH, a shape-agnostic, autoregressive foundation model for partial differential equations (PDEs). MORPH is built on a convolutional vision transformer backbone that seamlessly handles heterogeneous spatiotemporal datasets of varying data dimensionality (1D--3D) at different resolutions, multiple fields with mixed scalar and vector components. The architecture combines (i) component-wise convolution, which jointly processes scalar and vector channels to capture local interactions, (ii) inter-field cross-attention, which models and selectively propagates information between different physical fields, (iii) axial attentions, which factorizes full spatiotemporal self-attention along individual spatial and temporal axes to reduce computational burden while retaining expressivity. We pretrain multiple model variants on a diverse collection of heterogeneous PDE datasets and evaluate transfer to a range of downstream prediction tasks. Using both full-model fine-tuning and parameter-efficient low-rank adapters (LoRA), MORPH outperforms models trained from scratch in both zero-shot and full-shot generalization. Across extensive evaluations, MORPH matches or surpasses strong baselines and recent state-of-the-art models. Collectively, these capabilities present a flexible and powerful backbone for learning from heterogeneous and multimodal nature of scientific observations, charting a path toward scalable and data-efficient scientific machine learning.

---

## 83. FantasyWorld: Geometry-Consistent World Modeling via Unified Video and 3D Prediction

**论文链接:** [http://arxiv.org/abs/2509.21657v1](http://arxiv.org/abs/2509.21657v1)

**作者:** Yixiang Dai, Fan Jiang, Chiyu Wang, Mu Xu, Yonggang Qi

**发布时间:** 2025-09-25

### GPT解析

### 总结

本研究提出了一种名为FantasyWorld的几何增强框架，通过为冻结的视频基础模型添加可训练的几何分支，实现了视频潜隐和隐式3D场的联合建模，有效解决了当前视频基础模型缺乏明确3D接地能力的问题，从而提高了空间一致性和下游3D推理任务的实用性。

### 背景

高质量的3D世界模型对具身智能和通用人工智能至关重要，支持AR/VR内容创建和机器人导航等应用。然而，尽管现有的视频基础模型具有强大的想象先验，但它们缺乏明确的3D接地能力，限制了空间一致性和下游3D推理任务的效用。

### 目的

开发一种能够将视频想象与3D感知相结合的框架，增强视频基础模型的3D能力，使其能够生成空间一致且适用于下游3D任务的表示。

### 方法

提出了FantasyWorld框架，这是一个几何增强系统，通过为冻结的视频基础模型添加可训练的几何分支，实现视频潜隐和隐式3D场的联合建模。引入跨分支监督机制，其中几何线索指导视频生成，视频先验正则化3D预测，从而产生一致且可泛化的3D感知视频表示。

### 主要发现

FantasyWorld有效地桥接了视频想象和3D感知，在多视图一致性和风格一致性方面优于最近的几何一致性基线。几何分支产生的潜隐可作为下游3D任务（如新视图合成和导航）的通用表示，无需针对每个场景进行优化或微调。

### 结论

FantasyWorld通过统一主干和跨分支信息交换，成功解决了视频基础模型的3D接地问题，为高质量3D世界模型的构建提供了新思路，有望推动具身智能和通用人工智能的发展。

### 翻译

高质量3D世界模型对具身智能和通用人工智能(AGI)至关重要，支持AR/VR内容创建和机器人导航等应用。尽管已建立强大的想象先验，当前视频基础模型缺乏明确的3D接地能力，因此在空间一致性和下游3D推理任务的实用性方面受到限制。在这项工作中，我们提出了FantasyWorld，一个几何增强框架，通过为冻结的视频基础模型添加可训练的几何分支，能够在单次前向传递中联合建模视频潜隐和隐式3D场。我们的方法引入了跨分支监督，其中几何线索指导视频生成，视频先验正则化3D预测，从而产生一致且可泛化的3D感知视频表示。值得注意的是，几何分支产生的潜隐可作为下游3D任务（如新视图合成和导航）的通用表示，无需针对每个场景进行优化或微调。大量实验表明，FantasyWorld有效地桥接了视频想象和3D感知，在多视图一致性和风格一致性方面优于最近的几何一致性基线。消融研究进一步证实，这些改进来自于统一主干和跨分支信息交换。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决当前视频基础模型缺乏明确3D基础能力的问题，导致空间一致性和下游3D推理任务应用受限。这个问题很重要，因为高质量的3D世界模型对具身智能和人工智能至关重要，支撑着AR/VR内容创建、机器人导航等应用，能让AI系统更好地理解和生成与现实世界一致的环境。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者分析了现有视频生成模型与3D感知之间的弱耦合问题，发现大多数方法仅在视频领域内操作，特征无法直接支持3D推理，且需要额外场景优化。他们设计了一个几何增强框架，在冻结视频模型上添加可训练几何分支。该方法借鉴了VGGT的3D特征提取架构、WanDiT的扩散模型架构和双向交叉注意力机制，实现了视频与几何的紧密集成。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是通过统一框架同时进行视频生成和3D几何推理，使视频想象力和3D感知相互强化，无需场景特定优化。流程包括：1)接收图像、文本和相机轨迹输入；2)预处理块对视频潜在表示部分去噪；3)集成重建和生成块采用双分支设计，包含想象先验分支和几何一致分支；4)通过双向交叉注意力连接两个分支；5)输出几何一致视频帧和3D特征，可用于下游任务。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)统一的视频-3D建模框架；2)2D/3D跨分支监督机制；3)可推广的3D特征潜力。相比之前工作，FantasyWorld实现了视频生成和3D感知的紧密耦合而非弱连接；无需场景特定优化；通过轻量级适配器实现高效集成；直接从视频潜在表示推断几何信息而非从RGB图像预测。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'FantasyWorld通过统一的视频-3D建模框架，实现了视频生成与几何推理的紧密耦合，在单一前向传递中产生既逼真又几何一致的3D世界表示，无需场景特定优化。'}


### 论文摘要

High-quality 3D world models are pivotal for embodied intelligence and Artificial General Intelligence (AGI), underpinning applications such as AR/VR content creation and robotic navigation. Despite the established strong imaginative priors, current video foundation models lack explicit 3D grounding capabilities, thus being limited in both spatial consistency and their utility for downstream 3D reasoning tasks. In this work, we present FantasyWorld, a geometry-enhanced framework that augments frozen video foundation models with a trainable geometric branch, enabling joint modeling of video latents and an implicit 3D field in a single forward pass. Our approach introduces cross-branch supervision, where geometry cues guide video generation and video priors regularize 3D prediction, thus yielding consistent and generalizable 3D-aware video representations. Notably, the resulting latents from the geometric branch can potentially serve as versatile representations for downstream 3D tasks such as novel view synthesis and navigation, without requiring per-scene optimization or fine-tuning. Extensive experiments show that FantasyWorld effectively bridges video imagination and 3D perception, outperforming recent geometry-consistent baselines in multi-view coherence and style consistency. Ablation studies further confirm that these gains stem from the unified backbone and cross-branch information exchange.

---

## 84. Can AI Perceive Physical Danger and Intervene?

**论文链接:** [http://arxiv.org/abs/2509.21651v1](http://arxiv.org/abs/2509.21651v1)

**作者:** Abhishek Jindal, Dmitry Kalashnikov, Oscar Chang, Divya Garikapati, Anirudha Majumdar, Pierre Sermanet, Vikas Sindhwani

**发布时间:** 2025-09-25

### GPT解析

### 总结

这篇论文研究了具身AI系统与物理世界交互时的安全挑战，开发了一种可扩展的物理安全基准测试方法，分析了主要基础模型的安全理解能力，并提出了一种训练后范式来增强模型对具身安全约束的推理能力。

### 背景

当AI与物理世界交互时（如机器人或辅助代理），出现了超越纯'数字AI'的新安全挑战。在物理交互中，造成实际伤害的可能性是直接且即时的。研究关注最先进的基础模型对物理安全常识（如物体重量、热物处理等）的理解程度。

### 目的

评估和提升具身AI系统对物理安全的理解和推理能力，为安全关键代理应用的部署提供指导。

### 方法

1) 开发基于真实世界伤害叙事和操作安全约束的可扩展物理安全基准测试方法，使用先进生成模型将叙事和约束转化为逼真的图像和视频；2) 全面分析主要基础模型的风险感知、安全推理和干预触发能力；3) 开发训练后范式，教模型通过系统指令明确推理具身特定的安全约束，使安全推理可解释和透明。

### 主要发现

1) 具身AI系统在物理安全理解方面存在挑战；2) 主要基础模型在风险感知、安全推理和干预触发能力方面表现各异；3) 所提出的训练后范式能显著提升模型对具身安全约束的推理能力，使推理过程可解释和透明。

### 结论

通过开发专门的基准测试方法和训练范式，可以提升具身AI系统对物理安全的理解和推理能力，为安全关键代理应用的部署提供支持。

### 翻译

当AI与物理世界交互时——无论是作为机器人还是辅助代理——出现了超越纯'数字AI'的新安全挑战。在这种交互中，造成物理伤害的可能性是直接且即时的。最先进的基础模型对物理安全常识的理解程度如何？例如，一个盒子可能太重而无法举起，或者一杯热咖啡不应该递给儿童？在本文中，我们的贡献有三方面：首先，我们开发了一种高度可扩展的方法，用于基于真实世界伤害叙事和操作安全约束对具身AI系统进行持续的物理安全基准测试。为了探测多模态安全理解，我们使用先进的生成模型将这些叙事和约束转变为逼真的图像和视频，捕捉从安全到不安全状态的转变。其次，我们全面分析了主要基础模型感知风险、推理安全和触发干预的能力；这为它们在安全关键代理应用中的部署准备性提供了多方面的见解。最后，我们开发了一种训练后范式，教模型通过系统指令明确推理具身特定的安全约束。由此产生的模型生成思考轨迹，使安全推理可解释和透明，在约束满足评估中达到了最先进的性能。该基准测试将在https://asimov-benchmark.github.io/v2发布。


### 论文摘要

When AI interacts with the physical world -- as a robot or an assistive agent -- new safety challenges emerge beyond those of purely ``digital AI". In such interactions, the potential for physical harm is direct and immediate. How well do state-of-the-art foundation models understand common-sense facts about physical safety, e.g. that a box may be too heavy to lift, or that a hot cup of coffee should not be handed to a child? In this paper, our contributions are three-fold: first, we develop a highly scalable approach to continuous physical safety benchmarking of Embodied AI systems, grounded in real-world injury narratives and operational safety constraints. To probe multi-modal safety understanding, we turn these narratives and constraints into photorealistic images and videos capturing transitions from safe to unsafe states, using advanced generative models. Secondly, we comprehensively analyze the ability of major foundation models to perceive risks, reason about safety, and trigger interventions; this yields multi-faceted insights into their deployment readiness for safety-critical agentic applications. Finally, we develop a post-training paradigm to teach models to explicitly reason about embodiment-specific safety constraints provided through system instructions. The resulting models generate thinking traces that make safety reasoning interpretable and transparent, achieving state of the art performance in constraint satisfaction evaluations. The benchmark will be released at https://asimov-benchmark.github.io/v2

---

