# 今日论文推荐 - 2025-12-12

共 58 篇论文

---

## 1. Refinement Contrastive Learning of Cell-Gene Associations for Unsupervised Cell Type Identification

**论文链接:** [http://arxiv.org/abs/2512.10640v1](http://arxiv.org/abs/2512.10640v1)

**作者:** Liang Peng, Haopeng Liu, Yixuan Ye, Cheng Liu, Wenjun Shen, Si Wu, Hau-San Wong

**发布时间:** 2025-12-11

### GPT解析

### 总结

提出了一种名为scRCL的精细对比学习框架，通过整合细胞-基因相互作用来改进单细胞数据中的细胞类型识别

### 背景

无监督细胞类型识别对于揭示和表征单细胞组学研究中的异质群体至关重要

### 目的

解决现有聚类方法仅关注内在细胞结构而忽略细胞-基因关联的问题，提高区分密切相关的细胞类型的能力

### 方法

提出scRCL框架，包含两个对比分布对齐组件和一个精细模块，前者通过利用细胞-细胞结构关系揭示内在细胞结构，后者整合基因相关性结构学习来增强细胞嵌入并捕获细胞-基因关联

### 主要发现

在多个单细胞RNA-seq和空间转录组学基准数据集上，scRCL在细胞类型识别准确性方面始终优于最先进的基线方法，恢复的细胞群体表现出连贯的基因表达特征

### 结论

scRCL通过明确整合细胞-基因相互作用，能够获得更具信息量的表示，有效提高细胞类型识别的准确性，具有生物学相关性

### 翻译

无监督细胞类型识别对于揭示和表征单细胞组学研究中的异质群体至关重要。虽然已经开发了一系列聚类方法，但大多数仅专注于内在的细胞结构，而忽略了细胞-基因关联的关键作用，这限制了它们区分密切相关的细胞类型的能力。为了解决这一问题，我们提出了一个精细对比学习框架(scRCL)，它明确整合细胞-基因相互作用以获得更具信息量的表示。具体来说，我们引入了两个对比分布对齐组件，通过有效利用细胞-细胞结构关系来揭示可靠的内在细胞结构。此外，我们开发了一个精细模块，整合了基因相关性结构学习，通过捕获潜在的细胞-基因关联来增强细胞嵌入。该模块加强了细胞与其相关基因之间的联系，完善了表示学习，以利用生物学上有意义的关系。在几个单细胞RNA-seq和空间转录组学基准数据集上的大量实验表明，我们的方法在细胞类型识别准确性方面始终优于最先进的基线方法。此外，下游生物分析证实，恢复的细胞群体表现出连贯的基因表达特征，进一步验证了我们方法的生物学相关性。代码可在https://github.com/THPengL/scRCL获取。


### 论文摘要

Unsupervised cell type identification is crucial for uncovering and characterizing heterogeneous populations in single cell omics studies. Although a range of clustering methods have been developed, most focus exclusively on intrinsic cellular structure and ignore the pivotal role of cell-gene associations, which limits their ability to distinguish closely related cell types. To this end, we propose a Refinement Contrastive Learning framework (scRCL) that explicitly incorporates cell-gene interactions to derive more informative representations. Specifically, we introduce two contrastive distribution alignment components that reveal reliable intrinsic cellular structures by effectively exploiting cell-cell structural relationships. Additionally, we develop a refinement module that integrates gene-correlation structure learning to enhance cell embeddings by capturing underlying cell-gene associations. This module strengthens connections between cells and their associated genes, refining the representation learning to exploiting biologically meaningful relationships. Extensive experiments on several single-cell RNA-seq and spatial transcriptomics benchmark datasets demonstrate that our method consistently outperforms state-of-the-art baselines in cell-type identification accuracy. Moreover, downstream biological analyses confirm that the recovered cell populations exhibit coherent gene-expression signatures, further validating the biological relevance of our approach. The code is available at https://github.com/THPengL/scRCL.

---

## 2. Is the Information Bottleneck Robust Enough? Towards Label-Noise Resistant Information Bottleneck Learning

**论文链接:** [http://arxiv.org/abs/2512.10573v1](http://arxiv.org/abs/2512.10573v1)

**作者:** Yi Huang, Qingyun Sun, Yisen Gao, Haonan Yuan, Xingcheng Fu, Jianxin Li

**发布时间:** 2025-12-11

**备注:** Accepted by the Main Technical Track of the 40th Annual AAAI Conference on Artificial Intelligence (AAAI-2026)

### GPT解析

### 总结

本文提出了一种名为LaT-IB的新方法，用于解决信息瓶颈原理在标签噪声环境下的脆弱性问题，通过引入'最小充分干净'准则和噪声感知的潜在解耦技术，提高了模型在标签噪声下的鲁棒性和效率。

### 背景

信息瓶颈(IB)原理通过保留与标签相关的信息同时压缩无关信息来实现有效的表示学习。然而，它对准确标签的强依赖使其本质上容易受到标签噪声的影响，这在现实场景中很常见，会导致性能显著下降和过拟合。

### 目的

开发一种能够抵抗标签噪声的信息瓶颈方法，提高模型在现实场景中标签噪声情况下的鲁棒性和适用性。

### 方法

提出LaT-IB方法，引入'最小充分干净'(MSC)准则作为互信息正则化器，保留任务相关信息同时丢弃噪声；采用噪声感知的潜在解耦技术，将潜在表示分解为与干净标签空间和噪声空间对齐的组件；设计三阶段训练框架：预热、知识注入和鲁棒训练，逐步引导模型学习噪声抵抗表示。

### 主要发现

理论上推导了目标函数各分量（预测、压缩和解耦）的互信息边界，证明优化该目标函数能鼓励对输入噪声不变的表示，并分离干净和噪声标签信息；实验表明LaT-IB在标签噪声下实现优越的鲁棒性和效率。

### 结论

LaT-IB方法有效解决了标准IB原理对标签噪声的脆弱性问题，通过MSC准则和噪声感知解耦技术，显著提高了模型在标签噪声环境下的性能和适用性，为现实场景中的噪声标签问题提供了有效解决方案。

### 翻译

信息瓶颈(IB)原理通过保留标签相关信息同时压缩无关信息来促进有效的表示学习。然而，它对准确标签的强依赖使其本质上容易受到标签噪声的影响，这在现实场景中很普遍，导致性能显著下降和过拟合。为了解决这个问题，我们提出了LaT-IB，一种新颖的标签噪声抵抗信息瓶颈方法，它引入了'最小充分干净'(MSC)准则。作为互信息正则化器的实例，MSC保留任务相关信息同时丢弃噪声，解决了标准IB对噪声标签监督的脆弱性。为此，LaT-IB采用噪声感知的潜在解耦，将潜在表示分解为与干净标签空间和噪声空间对齐的组件。理论上，我们首先推导了目标函数每个组件（包括预测、压缩和解耦）的互信息边界，并且证明优化它会鼓励对输入噪声不变的表示并分离干净和噪声标签信息。此外，我们设计了一个三阶段训练框架：预热、知识注入和鲁棒训练，逐步引导模型学习噪声抵抗表示。大量实验表明，LaT-IB在标签噪声下实现优越的鲁棒性和效率，显著提高了在具有标签噪声的现实场景中的鲁棒性和适用性。


### 论文摘要

The Information Bottleneck (IB) principle facilitates effective representation learning by preserving label-relevant information while compressing irrelevant information. However, its strong reliance on accurate labels makes it inherently vulnerable to label noise, prevalent in real-world scenarios, resulting in significant performance degradation and overfitting. To address this issue, we propose LaT-IB, a novel Label-Noise ResistanT Information Bottleneck method which introduces a "Minimal-Sufficient-Clean" (MSC) criterion. Instantiated as a mutual information regularizer to retain task-relevant information while discarding noise, MSC addresses standard IB's vulnerability to noisy label supervision. To achieve this, LaT-IB employs a noise-aware latent disentanglement that decomposes the latent representation into components aligned with to the clean label space and the noise space. Theoretically, we first derive mutual information bounds for each component of our objective including prediction, compression, and disentanglement, and moreover prove that optimizing it encourages representations invariant to input noise and separates clean and noisy label information. Furthermore, we design a three-phase training framework: Warmup, Knowledge Injection and Robust Training, to progressively guide the model toward noise-resistant representations. Extensive experiments demonstrate that LaT-IB achieves superior robustness and efficiency under label noise, significantly enhancing robustness and applicability in real-world scenarios with label noise.

---

## 3. UniCoR: Modality Collaboration for Robust Cross-Language Hybrid Code Retrieval

**论文链接:** [http://arxiv.org/abs/2512.10452v1](http://arxiv.org/abs/2512.10452v1)

**作者:** Yang Yang, Li Kuang, Jiakun Liu, Zhongxin Liu, Yingjie Xia, David Lo

**发布时间:** 2025-12-11

**备注:** Accepted by the 48th IEEE/ACM International Conference on Software Engineering (ICSE 2026)

### GPT解析

### 总结

本文提出了一种名为UniCoR的新型自监督框架，用于解决代码检索中的语义理解不足、混合检索融合效率低下和跨语言泛化能力弱等问题。

### 背景

有效的代码检索至关重要，使用自然语言和代码片段进行混合搜索已成为重要范式，但现有方法在有效利用混合查询特别是在跨语言背景下尚不明确。

### 目的

对代表性代码模型进行全面经验研究，揭示现有代码检索面临的挑战，并提出解决方案。

### 方法

提出UniCoR框架，包含多视角监督对比学习模块（增强语义理解和模态融合）和表示分布一致性学习模块（提高跨语言泛化能力），从代码到代码、自然语言到代码、自然语言到自然语言多个角度对齐表示。

### 主要发现

现有代码检索面临三大挑战：语义理解不足、混合代码检索中的融合效率低下、跨语言场景下的泛化能力弱。

### 结论

在经验基准和大规模基准上的实验表明，UniCoR优于所有基线模型，MRR平均提高8.64%，MAP平均提高11.54%，在混合代码检索中表现出稳定性，在跨语言场景中具有泛化能力。

### 翻译

有效的代码检索是必不可少的，并且使用自然语言和代码片段进行混合搜索已成为一种重要范式。然而，现有方法是否能有效利用这种混合查询，特别是在跨语言背景下，仍然不清楚。我们对代表性的代码模型进行了全面的经验研究，并揭示了三个挑战：(1)语义理解不足；(2)混合代码检索中的融合效率低下；(3)跨语言场景下的泛化能力弱。为解决这些挑战，我们提出了UniCoR，一种新颖的自监督框架，用于学习统一且强大的代码表示。首先，我们设计了一个多视角监督对比学习模块，以增强语义理解和模态融合。它从多个角度对齐表示，包括代码到代码、自然语言到代码以及自然语言到自然语言，强制模型捕获模态之间的语义本质。其次，我们引入了表示分布一致性学习模块，以提高跨语言泛化能力，该模块明确对不同编程语言的特征分布进行对齐，实现语言无关的表示学习。在经验基准和大规模基准上的广泛实验表明，UniCoR优于所有基线模型，在MRR上比最佳基线模型平均提高8.64%，在MAP上平均提高11.54%。此外，UniCoR在混合代码检索中表现出稳定性，在跨语言场景中具有泛化能力。


### 论文摘要

Effective code retrieval is indispensable and it has become an important paradigm to search code in hybrid mode using both natural language and code snippets. Nevertheless, it remains unclear whether existing approaches can effectively leverage such hybrid queries, particularly in cross-language contexts. We conduct a comprehensive empirical study of representative code models and reveal three challenges: (1) insufficient semantic understanding; (2) inefficient fusion in hybrid code retrieval; and (3) weak generalization in cross-language scenarios. To address these challenges, we propose UniCoR, a novel self-supervised framework that learns Unified Code Representations framework designed to learn unified and robust code representations. Firstly, we design a multi-perspective supervised contrastive learning module to enhance semantic understanding and modality fusion. It aligns representations from multiple perspectives, including code-to-code, natural language-to-code, and natural language-to-natural language, enforcing the model to capture a semantic essence among modalities. Secondly, we introduce a representation distribution consistency learning module to improve cross-language generalization, which explicitly aligns the feature distributions of different programming languages, enabling language-agnostic representation learning. Extensive experiments on both empirical benchmark and large-scale benchmark show that UniCoR outperforms all baseline models, achieving an average improvement of 8.64% in MRR and 11.54% in MAP over the best-performing baseline. Furthermore, UniCoR exhibits stability in hybrid code retrieval and generalization capability in cross-language scenarios.

---

## 4. LLM-Empowered Representation Learning for Emerging Item Recommendation

**论文链接:** [http://arxiv.org/abs/2512.10370v1](http://arxiv.org/abs/2512.10370v1)

**作者:** Ziying Zhang, Quanming Yao, Yaqing Wang

**发布时间:** 2025-12-11

### GPT解析

### 总结

本研究提出EmerFlow，一个由大语言模型赋能的表征学习框架，用于解决推荐新兴物品的挑战。该框架通过LLM推理丰富新兴物品特征，对齐表示空间，并通过元学习整合新交互，从而仅从有限交互中学习表达性嵌入。

### 背景

现有推荐方法在处理新兴物品时存在问题，这些物品的交互随时间逐渐积累，而现有方法通常假设新兴物品很少有或没有历史交互，这简化了问题的复杂性。

### 目的

开发一个能够保留新兴物品独特性，同时利用其与已建立物品共享模式的推荐模型。

### 方法

提出EmerFlow框架，包括三个步骤：1)通过LLM推理丰富新兴物品的原始特征；2)将这些表示与现有推荐模型的嵌入空间对齐；3)通过元学习整合新交互来优化嵌入。

### 主要发现

EmerFlow能够仅从有限的交互中学习新兴物品的表达性嵌入，有效解决了新兴物品推荐的问题。

### 结论

在电影和制药等不同领域的广泛实验表明，EmerFlow持续优于现有方法，证明了其有效性和通用性。

### 翻译

在本工作中，我们解决了推荐新兴物品的挑战，这些物品的交互随时间逐渐积累。现有方法经常忽略这种动态过程，通常假设新兴物品很少有甚至没有历史交互。这种假设简化了问题，因为好的模型必须保留新兴物品的独特性，同时利用它们与已建立物品的共享模式。为了应对这一挑战，我们提出了EmerFlow，一种新颖的由大语言模型赋能的表征学习框架，为新兴物品生成独特的嵌入表示。它首先通过LLM推理丰富新兴物品的原始特征，然后将这些表示与现有推荐模型的嵌入空间对齐。最后，通过元学习整合新的交互来优化嵌入。这使得EmerFlow能够仅从有限的交互中学习新兴物品的表达性嵌入。在电影和制药等不同领域的广泛实验表明，EmerFlow持续优于现有方法。


### 论文摘要

In this work, we tackle the challenge of recommending emerging items, whose interactions gradually accumulate over time. Existing methods often overlook this dynamic process, typically assuming that emerging items have few or even no historical interactions. Such an assumption oversimplifies the problem, as a good model must preserve the uniqueness of emerging items while leveraging their shared patterns with established ones. To address this challenge, we propose EmerFlow, a novel LLM-empowered representation learning framework that generates distinctive embeddings for emerging items. It first enriches the raw features of emerging items through LLM reasoning, then aligns these representations with the embedding space of the existing recommendation model. Finally, new interactions are incorporated through meta-learning to refine the embeddings. This enables EmerFlow to learn expressive embeddings for emerging items from only limited interactions. Extensive experiments across diverse domains, including movies and pharmaceuticals, show that EmerFlow consistently outperforms existing methods.

---

## 5. Neuronal Attention Circuit (NAC) for Representation Learning

**论文链接:** [http://arxiv.org/abs/2512.10282v1](http://arxiv.org/abs/2512.10282v1)

**作者:** Waleed Razzaq, Izis Kankaraway, Yun-Bo Zhao

**发布时间:** 2025-12-11

**备注:** Paper for ICML2026

### GPT解析

### 总结

本文提出了一种名为神经元注意回路(NAC)的新型生物合理连续时间(CT)注意力机制，解决了传统注意力机制离散性限制连续时间建模的问题。

### 背景

注意力机制改善了循环神经网络(RNNs)的表示学习，但其离散性质限制了连续时间(CT)建模能力。

### 目的

引入一种新颖的、生物合理的CT-Attention机制，即神经元注意回路(NAC)，以克服注意力机制的离散性限制，实现更有效的连续时间建模。

### 方法

将注意力logits计算重新表述为具有非线性互连门的一阶线性ODE的解；用稀疏感官门替代密集投影用于key-query投影；使用具有两个头的稀疏骨干网络计算内容目标门和学习时间常数门；支持三种计算模式：显式欧拉积分、精确闭式解和稳态近似；实现稀疏Top-K级联方案提高内存效率。

### 主要发现

在多个领域实现了NAC，包括不规则时间序列分类、自动驾驶车辆的车道保持和工业预测；NAC在准确性方面匹配或优于竞争基线；在运行时间和内存效率方面，NAC位于几个CT基线的中间位置。

### 结论

NAC是一种有效的CT-Attention机制，能够处理连续时间数据，同时保持或提高准确性，并在效率方面处于合理位置。

### 翻译

Attention机制改善了RNNs的表示学习，但其离散性质限制了连续时间(CT)建模。我们引入了神经元注意回路(NAC)，这是一种新颖的、生物合理的CT-Attention机制，它将attention logits的计算重新表述为具有从重新利用线虫神经元回路策略(NCPs)连接机制衍生的非线性互连门的一阶线性ODE的解。NAC用稀疏感官门替代密集投影用于key-query投影，并使用具有两个头的稀疏骨干网络计算内容目标门和学习时间常数门，实现高效的适应性动态。NAC支持三种attention logit计算模式：(i)显式欧拉积分，(ii)精确闭式解，(iii)稳态近似。为了提高内存强度，我们实现了一个稀疏Top-K级联方案，有选择地筛选key-query交互。我们提供了严格的理论保证，包括状态稳定性、有界近似误差和通用逼近。经验上，我们在不规则时间序列分类、自动驾驶车辆的车道保持和工业预测等不同领域实现了NAC。我们观察到，NAC在准确性方面匹配或优于竞争基线，并且在运行时间和内存效率方面，与几个CT基线相比处于中间位置。


### 论文摘要

Attention improves representation learning over RNNs, but its discrete nature limits continuous-time (CT) modeling. We introduce Neuronal Attention Circuit (NAC), a novel, biologically plausible CT-Attention mechanism that reformulates attention logits computation as the solution to a linear first-order ODE with nonlinear interlinked gates derived from repurposing \textit{C. elegans} Neuronal Circuit Policies (NCPs) wiring mechanism. NAC replaces dense projections with sparse sensory gates for key-query projections and a sparse backbone network with two heads for computing \textit{content-target} and \textit{learnable time-constant} gates, enabling efficient adaptive dynamics. NAC supports three attention logit computation modes: (i) explicit Euler integration, (ii) exact closed-form solution, and (iii) steady-state approximation. To improve memory intensity, we implemented a sparse Top-\emph{K} pairwise concatenation scheme that selectively curates key-query interactions. We provide rigorous theoretical guarantees, including state stability, bounded approximation errors, and universal approximation. Empirically, we implemented NAC in diverse domains, including irregular time-series classification, lane-keeping for autonomous vehicles, and industrial prognostics. We observed that NAC matches or outperforms competing baselines in accuracy and occupies an intermediate position in runtime and memory efficiency compared with several CT baselines.

---

## 6. Enhancing Fake-News Detection with Node-Level Topological Features

**论文链接:** [http://arxiv.org/abs/2512.09974v1](http://arxiv.org/abs/2512.09974v1)

**作者:** Kaiyuan Xu

**发布时间:** 2025-12-10

### GPT解析

### 总结

本研究通过引入图论指标改进假新闻检测方法，显著提高了检测性能

### 背景

近年来虚假信息和假新闻泛滥对个人和社会构成严重威胁，推动了自动化检测方法的研究

### 目的

解决现有方法中图级表示学习完全依赖GNN，缺乏显式拓扑线索的问题

### 方法

对每个节点，将度中心性和局部聚类系数两个经典图论指标附加到原始BERT和profile嵌入中，明确标记hub和community的角色

### 主要发现

在UPFD Politifact子集中，这种简单修改使宏观F1从0.7753提高到0.8344，超过原始基线

### 结论

显式拓扑特征在假新闻检测中具有实用价值，为其他信息扩散任务中融合图指标提供了可解释、易于复制的模板

### 翻译

近年来，虚假信息和假新闻的泛滥对个人和社会构成严重威胁，推动了自动化检测方法的研究。先前的工作表明，整合内容、用户偏好和传播结构能够实现强大的性能，但将所有图级表示学习完全留给GNN，隐藏了任何显式的拓扑线索。为了弥补这一差距，我们引入了一种轻量级增强：对每个节点，将两个经典的图论指标——度中心性和局部聚类系数——附加到其原始BERT和profile嵌入中，从而明确标记hub和community的角色。在UPFD Politifact子集中，这种简单的修改使宏观F1从0.7753提高到0.8344，超过了原始基线。我们的研究不仅证明了显式拓扑特征在假新闻检测中的实用价值，还为在其他信息扩散任务中融合图指标提供了一个可解释、易于复制的模板。


### 论文摘要

In recent years, the proliferation of misinformation and fake news has posed serious threats to individuals and society, spurring intense research into automated detection methods. Previous work showed that integrating content, user preferences, and propagation structure achieves strong performance, but leaves all graph-level representation learning entirely to the GNN, hiding any explicit topological cues. To close this gap, we introduce a lightweight enhancement: for each node, we append two classical graph-theoretic metrics, degree centrality and local clustering coefficient, to its original BERT and profile embeddings, thus explicitly flagging the roles of hub and community. In the UPFD Politifact subset, this simple modification boosts macro F1 from 0.7753 to 0.8344 over the original baseline. Our study not only demonstrates the practical value of explicit topology features in fake-news detection but also provides an interpretable, easily reproducible template for fusing graph metrics in other information-diffusion tasks.

---

## 7. StateSpace-SSL: Linear-Time Self-supervised Learning for Plant Disease Detection

**论文链接:** [http://arxiv.org/abs/2512.09492v2](http://arxiv.org/abs/2512.09492v2)

**作者:** Abdullah Al Mamun, Miaohua Zhang, David Ahmedt-Aristizabal, Zeeshan Hayder, Mohammad Awrangjeb

**发布时间:** 2025-12-10

**备注:** Accepted to AAAI workshop (AgriAI 2026)

### GPT解析

### 总结

提出StateSpace-SSL，一种基于Vision Mamba状态空间编码器的线性时间自监督学习框架，用于植物疾病检测，解决了CNN和Transformer方法在农业图像中的局限性。

### 背景

自监督学习在植物疾病检测中有吸引力，可利用大量未标记叶片图像，但现有SSL方法基于CNN或视觉变压器，与农业图像不匹配。CNN难以捕捉沿叶结构连续发展的疾病模式，而Transformer引入高分辨率补丁的二次注意力成本。

### 目的

开发一种更适合农业图像的SSL框架，解决现有方法在捕捉植物疾病特征方面的局限性。

### 方法

提出StateSpace-SSL框架，采用Vision Mamba状态空间编码器，通过在叶表面进行方向扫描来建模长范围病变连续性，并使用原型驱动的教师-学生目标对齐多视图表示，从标记数据中学习稳定和病变感知的特征。

### 主要发现

在三个公开植物疾病数据集上的实验表明，StateSpace-SSL在各种评估指标上一致优于CNN和Transformer基线方法，定性分析证实其学习到紧凑的、病变聚焦的特征图。

### 结论

线性状态空间建模在自监督植物疾病表示学习中具有显著优势，能够有效捕捉植物叶片上的疾病特征。

### 翻译

自监督学习(SSL)在植物疾病检测中具有吸引力，因为它可以利用大量未标记的叶片图像，然而大多数现有的SSL方法基于CNN或视觉变压器构建，与农业图像不匹配。基于CNN的SSL难以捕捉沿叶结构连续发展的疾病模式，而基于变压器的SSL引入了来自高分辨率补丁的二次注意力成本。为解决这些局限性，我们提出了StateSpace-SSL，这是一种线性时间的SSL框架，采用Vision Mamba状态空间编码器，通过在叶表面进行方向扫描来建模长范围的病变连续性。原型驱动的教师-学生目标对齐多视图表示，鼓励从标记数据中学习稳定和病变感知的特征。在三个公开的植物疾病数据集上的实验表明，StateSpace-SSL在各种评估指标上一致优于基于CNN和变压器的SSL基线。定性分析进一步确认它学习到紧凑的、病变聚焦的特征图，突显了线性状态空间建模在自监督植物疾病表示学习中的优势。


### 论文摘要

Self-supervised learning (SSL) is attractive for plant disease detection as it can exploit large collections of unlabeled leaf images, yet most existing SSL methods are built on CNNs or vision transformers that are poorly matched to agricultural imagery. CNN-based SSL struggles to capture disease patterns that evolve continuously along leaf structures, while transformer-based SSL introduces quadratic attention cost from high-resolution patches. To address these limitations, we propose StateSpace-SSL, a linear-time SSL framework that employs a Vision Mamba state-space encoder to model long-range lesion continuity through directional scanning across the leaf surface. A prototype-driven teacher-student objective aligns representations across multiple views, encouraging stable and lesion-aware features from labelled data. Experiments on three publicly available plant disease datasets show that StateSpace-SSL consistently outperforms the CNN- and transformer-based SSL baselines in various evaluation metrics. Qualitative analyses further confirm that it learns compact, lesion-focused feature maps, highlighting the advantage of linear state-space modelling for self-supervised plant disease representation learning.

---

## 8. HGC-Herd: Efficient Heterogeneous Graph Condensation via Representative Node Herding

**论文链接:** [http://arxiv.org/abs/2512.09947v1](http://arxiv.org/abs/2512.09947v1)

**作者:** Fuyan Ou, Siqi Ai, Yulin Hu

**发布时间:** 2025-12-08

**备注:** 8 pages, 2 figures

### GPT解析

### 总结

HGC-Herd是一种无训练的异构图压缩框架，能够生成紧凑但信息丰富的异构图，同时保持语义和结构保真度。

### 背景

异构图神经网络在建模多类型节点和关系的复杂语义方面表现出强大能力，但它们在大规模图上的扩展性面临挑战，因为存在结构冗余和高维节点特征问题。现有的图压缩方法主要为同构图开发，依赖梯度匹配，导致计算、内存和优化开销较大。

### 目的

提出一个训练-free的压缩框架，生成紧凑但信息丰富的异构图，同时保持语义和结构保真度。

### 方法

HGC-Herd集成轻量级特征传播来编码多跳关系上下文，并采用按类别聚集机制来识别每个类别的代表性节点，为下游学习任务生成平衡且具有区分性的子集。

### 主要发现

在ACM、DBLP和Freebase上的大量实验表明，HGC-Herd达到与全图训练相当或更好的准确性，同时显著减少了运行时间和内存消耗。

### 结论

HGC-Herd在高效和可扩展的异构图表示学习中具有重要的实际价值。

### 翻译

异构图神经网络在建模多类型节点和关系的复杂语义方面表现出强大能力。然而，由于结构冗余和高维节点特征，它们在大规模图上的扩展性仍然具有挑战性。现有的图压缩方法，如GCond，主要为同构图开发，并依赖梯度匹配，导致相当大的计算、内存和优化开销。我们提出了HGC-Herd，一种无训练的压缩框架，能够生成紧凑但信息丰富的异构图，同时保持语义和结构保真度。HGC-Herd集成轻量级特征传播来编码多跳关系上下文，并采用按类别聚集机制来识别每个类别的代表性节点，为下游学习任务生成平衡且具有区分性的子集。在ACM、DBLP和Freebase上的大量实验验证了HGC-Herd能够达到与全图训练相当或更好的准确性，同时显著减少了运行时间和内存消耗。这些结果强调了它在高效和可扩展的异构图表示学习中的实际价值。


### 论文摘要

Heterogeneous graph neural networks (HGNNs) have demonstrated strong capability in modeling complex semantics across multi-type nodes and relations. However, their scalability to large-scale graphs remains challenging due to structural redundancy and high-dimensional node features. Existing graph condensation approaches, such as GCond, are primarily developed for homogeneous graphs and rely on gradient matching, resulting in considerable computational, memory, and optimization overhead. We propose HGC-Herd, a training-free condensation framework that generates compact yet informative heterogeneous graphs while maintaining both semantic and structural fidelity. HGC-Herd integrates lightweight feature propagation to encode multi-hop relational context and employs a class-wise herding mechanism to identify representative nodes per class, producing balanced and discriminative subsets for downstream learning tasks. Extensive experiments on ACM, DBLP, and Freebase validate that HGC-Herd attains comparable or superior accuracy to full-graph training while markedly reducing both runtime and memory consumption. These results underscore its practical value for efficient and scalable heterogeneous graph representation learning.

---

## 9. Mitigating Exposure Bias in Risk-Aware Time Series Forecasting with Soft Tokens

**论文链接:** [http://arxiv.org/abs/2512.10056v1](http://arxiv.org/abs/2512.10056v1)

**作者:** Alireza Namazi, Amirreza Dolatpour Fathkouhi, Heman Shakeri

**发布时间:** 2025-12-10

### GPT解析

### 总结

研究介绍了一种名为SoTra（软标记轨迹预测）的新方法，用于改进糖尿病和血液动力学管理中的自回归预测，解决标准模型中的暴露偏差问题，提高预测稳定性和临床安全性。

### 背景

自回归预测是糖尿病和血液动力学管理中预测控制的核心，不同操作区域具有不同临床风险。使用教师强制训练的标准模型存在暴露偏差，导致闭环使用时的多步预测不稳定。

### 目的

开发一种能够减轻暴露偏差并学习校准的不确定性感知轨迹的预测方法，以降低临床风险。

### 方法

引入软标记轨迹预测(SoTra)，传播连续概率分布（软标记）来减轻暴露偏差并学习校准的不确定性感知轨迹，同时使用风险感知解码模块最小化预期临床伤害。

### 主要发现

在葡萄糖预测中，SoTra将基于区域的平均风险降低18%；在血压预测中，将有效临床风险降低约15%。

### 结论

这些改进支持SoTra在安全关键型预测控制中的应用。

### 翻译

自回归预测是糖尿病和血液动力学管理中预测控制的核心，其中不同的操作区域具有不同的临床风险。使用教师强制训练的标准模型存在暴露偏差问题，导致闭环使用时的多步预测不稳定。我们引入了软标记轨迹预测(SoTra)，该方法传播连续概率分布（'软标记'）来减轻暴露偏差并学习校准的不确定性感知轨迹。然后，使用风险感知解码模块来最小化预期的临床伤害。在葡萄糖预测中，SoTra将基于区域的平均风险降低了18%；在血压预测中，它将有效临床风险降低了约15%。这些改进支持其在安全关键型预测控制中的应用。


### 论文摘要

Autoregressive forecasting is central to predictive control in diabetes and hemodynamic management, where different operating zones carry different clinical risks. Standard models trained with teacher forcing suffer from exposure bias, yielding unstable multi-step forecasts for closed-loop use. We introduce Soft-Token Trajectory Forecasting (SoTra), which propagates continuous probability distributions (``soft tokens'') to mitigate exposure bias and learn calibrated, uncertainty-aware trajectories. A risk-aware decoding module then minimizes expected clinical harm. In glucose forecasting, SoTra reduces average zone-based risk by 18\%; in blood-pressure forecasting, it lowers effective clinical risk by approximately 15\%. These improvements support its use in safety-critical predictive control.

---

## 10. TransLocNet: Cross-Modal Attention for Aerial-Ground Vehicle Localization with Contrastive Learning

**论文链接:** [http://arxiv.org/abs/2512.10419v1](http://arxiv.org/abs/2512.10419v1)

**作者:** Phu Pham, Damon Conover, Aniket Bera

**发布时间:** 2025-12-11

**备注:** 8 pages, 4 figures, 4 tables

### GPT解析

### 总结

TransLocNet是一种跨模态注意力框架，通过融合LiDAR几何和空中语义上下文解决空中-地面定位问题，实验证明其性能优越，可达到亚米级、亚度级精度。

### 背景

地面LiDAR和俯视图像之间存在较大的视点差异和模态差异，使得空中-地面定位变得困难。

### 目的

提出一个有效的方法来解决空中-地面定位中的视点和模态差异问题。

### 方法

TransLocNet框架将LiDAR扫描投影为鸟瞰图表示，通过双向注意力与空中特征对齐，使用似然图解码器输出位置和方向的空间概率分布，并通过对比学习模块强制共享嵌入空间以提高跨模态对齐。

### 主要发现

在CARLA和KITTI数据集上的实验表明，TransLocNet优于最先进的基线方法，将定位误差减少高达63%，并实现亚米级、亚度级精度。

### 结论

TransLocNet在合成和真实世界环境中都提供了稳健且可推广的空中-地面定位解决方案。

### 翻译

空中-地面定位由于地面LiDAR和俯视图像之间存在较大的视点和模态差异而变得困难。我们提出了TransLocNet，这是一个跨模态注意力框架，融合了LiDAR几何和空中语义上下文。LiDAR扫描被投影为鸟瞰图表示，并通过双向注意力与空中特征对齐，随后是一个似然图解码器，输出位置和方向的空间概率分布。对比学习模块强制共享嵌入空间以提高跨模态对齐。在CARLA和KITTI上的实验表明，TransLocNet优于最先进的基线，将定位误差减少高达63%，并实现亚米级、亚度级精度。这些结果表明，TransLocNet在合成和真实世界环境中都提供了稳健且可推广的空中-地面定位。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决空中-地面车辆定位问题，即如何将地面级别的LiDAR扫描数据与头顶航空图像有效融合以实现精确车辆定位。这个问题在现实中很重要，因为传统GPS/INS系统在GNSS受限环境（如城市区域、隧道、密集森林）中性能下降，而其他替代方案如视觉里程计会累积漂移，高清地图又成本高昂。有效的空中-地面定位能提供不受环境干扰的可靠导航，对自动驾驶、移动机器人和军事应用至关重要。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先分析了现有方法的局限性：大多数方法专注于视觉数据而未充分利用LiDAR，现有融合策略依赖简单特征连接无法捕捉空间对应关系，且缺乏对大规模视点变化的鲁棒性。作者设计了一个结合跨模态注意力和对比学习的框架：将LiDAR投影到鸟瞰图(BEV)表示，通过双向注意力对齐航空和BEV特征，使用似然图解码器输出概率分布，并通过对比学习改善跨模态对齐。该方法借鉴了Transformer、BEVFormer等跨模态注意力机制，以及Lift-Splat-Shoot和BEVFusion等BEV表示方法，还参考了对比学习技术，但首次将它们联合应用于空中-地面定位。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是使用跨模态注意力机制动态关注BEV点云和航空图像之间的显著特征，并通过对比学习优化不同模态间的特征嵌入，提高检索能力和度量姿态估计。整体流程包括：1) 特征提取 - BEV编码器处理投影点云并应用傅里叶变换提供旋转不变性，航空编码器提取图像特征；2) 跨模态注意力 - 使用双向注意力机制对齐两种模态特征；3) 似然图解码器 - 生成位置和方向的概率分布及置信度分数；4) 对比学习模块 - 将特征投影到共享空间，使匹配对接近、不匹配对分离。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点有三：1) 跨模态注意力模块动态关注BEV和航空图像间的显著特征，实现鲁棒特征融合；2) 集成对比学习目标优化跨模态特征嵌入，加强检索能力；3) 在CARLA和KITTI等基准上全面验证，实现最高精度。相比之前工作，TransLocNet同时利用LiDAR几何信息和航空图像语义信息，使用双向注意力而非简单特征连接捕捉模态间复杂关系，结合对比学习增强特征判别力，采用概率解码器提供不确定性估计，并在BEV表示中引入傅里叶变换层提供旋转不变性。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'TransLocNet通过创新的跨模态注意力和对比学习框架，有效解决了空中-地面车辆定位中的视点和模态差距问题，实现了在合成和真实世界环境中更精确、更鲁棒的车辆定位。'}


### 论文摘要

Aerial-ground localization is difficult due to large viewpoint and modality gaps between ground-level LiDAR and overhead imagery. We propose TransLocNet, a cross-modal attention framework that fuses LiDAR geometry with aerial semantic context. LiDAR scans are projected into a bird's-eye-view representation and aligned with aerial features through bidirectional attention, followed by a likelihood map decoder that outputs spatial probability distributions over position and orientation. A contrastive learning module enforces a shared embedding space to improve cross-modal alignment. Experiments on CARLA and KITTI show that TransLocNet outperforms state-of-the-art baselines, reducing localization error by up to 63% and achieving sub-meter, sub-degree accuracy. These results demonstrate that TransLocNet provides robust and generalizable aerial-ground localization in both synthetic and real-world settings.

---

## 11. Self-Supervised Contrastive Embedding Adaptation for Endoscopic Image Matching

**论文链接:** [http://arxiv.org/abs/2512.10379v1](http://arxiv.org/abs/2512.10379v1)

**作者:** Alberto Rota, Elena De Momi

**发布时间:** 2025-12-11

### GPT解析

### 总结

本研究提出了一种新的深度学习管道，用于在内窥镜图像对之间建立特征对应关系，并配合自监督优化框架进行模型训练。该方法通过新颖视图合成生成真实对应关系，利用对比学习范式挖掘三元组，并优化DINOv2主干网络添加Transformer层以提高匹配精度。实验表明该方法在SCARED数据集上优于现有技术，为内窥镜手术中更准确的计算机视觉应用提供了有价值的贡献。

### 背景

准确的空间理解对图像引导手术、增强现实集成和上下文感知至关重要。在微创手术中，视觉输入是唯一术中模式，建立内窥镜帧间精确像素级对应关系对3D重建、相机跟踪和场景解释至关重要。然而，手术领域面临弱透视线索、非朗伯组织反射和复杂可变形解剖结构等挑战，降低了传统计算机视觉技术的性能。虽然深度学习模型在自然场景中表现良好，但其特征本质上不适合外科图像的细粒度匹配，需要针对性适应以满足该领域需求。

### 目的

开发一种新的深度学习管道，用于建立内窥镜图像对之间的特征对应关系，并设计自监督优化框架进行模型训练，以提高手术图像匹配的准确性。

### 方法

提出的方法利用新颖视图合成管道生成真实内点对应关系，随后在对比学习范式中利用这些对应关系挖掘三元组。通过自监督方法，为DINOv2主干网络添加额外的Transformer层，专门优化以产生通过余弦相似度阈值实现直接匹配的嵌入。

### 主要发现

实验评估表明，该管道在SCARED数据集上超越了最先进的方法，与相关工作相比提高了匹配精度并降低了极线误差。

### 结论

所提出的框架为在内窥镜手术中实现更准确的高级计算机视觉应用做出了有价值的贡献。

### 翻译

准确的空间理解对于图像引导手术、增强现实集成和上下文感知至关重要。在微创手术中，视觉输入是唯一的术中模式，在内窥镜帧之间建立精确的像素级对应关系对3D重建、相机跟踪和场景解释至关重要。然而，手术领域存在特殊挑战：弱透视线索、非朗伯组织反射和复杂可变形解剖结构，这些因素降低了传统计算机视觉技术的性能。虽然深度学习模型在自然场景中表现出色，但其特征本质上不适合外科图像的细粒度匹配，需要针对性适应以满足该领域需求。本研究提出了一种新的深度学习管道，用于建立内窥镜图像对之间的特征对应关系，以及一个自监督优化框架用于模型训练。所提出的方法利用新颖视图合成管道生成真实内点对应关系，随后在对比学习范式中利用这些对应关系挖掘三元组。通过这种自监督方法，我们为DINOv2主干网络添加了一个额外的Transformer层，专门优化以产生通过余弦相似度阈值实现直接匹配的嵌入。实验评估表明，我们的管道在SCARED数据集上超越了最先进的方法，与相关工作相比提高了匹配精度并降低了极线误差。所提出的框架为在内窥镜手术中实现更准确的高级计算机视觉应用做出了有价值的贡献。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决内窥镜图像中的特征匹配问题，即在微创手术过程中建立不同内窥镜帧之间的精确像素对应关系。这个问题非常重要，因为精确的空间理解对于图像引导手术、增强现实集成和手术环境中的3D重建、相机跟踪和场景解释至关重要。改进的图像匹配技术可以提高外科医生的情景感知，减少手术创伤，缩短恢复时间，并改善手术结果。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先分析了传统特征匹配方法在手术环境中的局限性，然后提出利用预训练的视觉Transformer架构（特别是DINOv2）作为基础特征提取器，并设计了一个额外的Transformer层来适应这些特征，使其更适合像素级匹配。作者借鉴了多个现有工作：HardNet的自监督学习方法、DINOv2的预训练表示、SuperGlue和LoFTR的Transformer架构、新颖视图合成技术和深度估计方法，但将这些元素创新地结合并针对手术领域进行了优化。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是通过自监督对比学习将预训练的语义特征适应为更适合像素级匹配的描述符。整体流程包括：1)使用预训练的DINOv2模型提取语义特征；2)通过专门的Transformer层将这些特征适应为匹配描述符；3)利用新颖视图合成生成自监督信号，避免人工标注；4)通过三元组对比损失优化适应层，使对应区域的特征在嵌入空间中更接近；5)在推理时，计算描述符间的余弦相似度，建立对应关系，并使用相位相关进行亚像素精炼。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)端到端的特征提取和匹配管道，专门针对内窥镜图像优化；2)自监督训练协议，利用新颖视图合成生成几何一致的像素对应关系；3)额外的Transformer层，优化以生成适合直接匹配的嵌入；4)对比学习框架，通过三元组挖掘增强特征判别能力。相比之前工作，该方法在弱纹理、镜面反射和变形条件下更鲁棒；避免了大规模标注数据的需求；使用随机采样变换增加训练多样性；利用Transformer的全局上下文建模能力更适合内窥镜环境。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '这篇论文提出了一种自监督对比嵌入适应方法，通过新颖视图合成生成训练信号，有效解决了内窥镜图像中特征匹配的挑战，提高了手术导航的精确性和鲁棒性。'}


### 论文摘要

Accurate spatial understanding is essential for image-guided surgery, augmented reality integration and context awareness. In minimally invasive procedures, where visual input is the sole intraoperative modality, establishing precise pixel-level correspondences between endoscopic frames is critical for 3D reconstruction, camera tracking, and scene interpretation. However, the surgical domain presents distinct challenges: weak perspective cues, non-Lambertian tissue reflections, and complex, deformable anatomy degrade the performance of conventional computer vision techniques. While Deep Learning models have shown strong performance in natural scenes, their features are not inherently suited for fine-grained matching in surgical images and require targeted adaptation to meet the demands of this domain. This research presents a novel Deep Learning pipeline for establishing feature correspondences in endoscopic image pairs, alongside a self-supervised optimization framework for model training. The proposed methodology leverages a novel-view synthesis pipeline to generate ground-truth inlier correspondences, subsequently utilized for mining triplets within a contrastive learning paradigm. Through this self-supervised approach, we augment the DINOv2 backbone with an additional Transformer layer, specifically optimized to produce embeddings that facilitate direct matching through cosine similarity thresholding. Experimental evaluation demonstrates that our pipeline surpasses state-of-the-art methodologies on the SCARED datasets improved matching precision and lower epipolar error compared to the related work. The proposed framework constitutes a valuable contribution toward enabling more accurate high-level computer vision applications in surgical endoscopy.

---

## 12. StainNet: A Special Staining Self-Supervised Vision Transformer for Computational Pathology

**论文链接:** [http://arxiv.org/abs/2512.10326v1](http://arxiv.org/abs/2512.10326v1)

**作者:** Jiawen Li, Jiali Hu, Xitong Ling, Yongqiang Lv, Yuxuan Chen, Yizhi Wang, Tian Guan, Yifei Liu, Yonghong He

**发布时间:** 2025-12-11

**备注:** 15 pages, 6 figures

### GPT解析

### 总结

本文提出了一种名为StainNet的特殊染色专用基础模型，基于视觉变换器架构，采用自蒸馏SSL方法，在HISTAI数据库中的20,231个特殊染色WSI上裁剪的140多万个补丁图像进行训练，解决了现有病理学基础模型主要在H&E染色图像上预训练在特殊染色临床应用中的局限性。

### 背景

基础模型通过在大规模组织学图像上进行自监督学习训练，显著加速了计算病理学发展。这些模型可作为ROI图像分析的主干或全幻灯片图像中的补丁级特征提取器。然而，现有病理学基础模型通常仅在H&E染色图像上预训练，而特殊染色图像在临床实践中也经常使用。

### 目的

解决病理学基础模型主要在H&E染色图像上预训练可能涉及特殊染色的临床应用中的局限性，开发一种适用于特殊染色的专用基础模型。

### 方法

提出StainNet，一种基于视觉变换器架构的特殊染色专用基础模型，采用自蒸馏SSL方法，在HISTAI数据库中的20,231个公开可用的特殊染色WSI上裁剪的140多万个补丁图像上进行训练。

### 主要发现

在内部幻灯片级别的肝脏恶性肿瘤分类任务和两个公共ROI级数据集上进行了实验，证明了其强大的能力；进行了少比例学习和检索评估；与最近较大的PFMs进行了比较，进一步突显了其优势。

### 结论

StainNet模型已公开发布，可从https://huggingface.co/JWonderLand/StainNet获取，为特殊染色病理图像分析提供了有效的解决方案。

### 翻译

Foundation models trained with self-supervised learning on large-scale histological images have significantly accelerated the development of computational pathology. These models can serve as backbones for region-of-interest image analysis or patch-level feature extractors in whole-slide images based on multiple instance learning. Existing pathology foundation models are typically pre-trained on Hematoxylin-Eosin stained pathology images. However, images with special stains, such as immunohistochemistry, are also frequently used in clinical practice. PFMs pre-trained mainly on H&E-stained images may be limited in clinical applications involving special stains. To address this issue, we propose StainNet, a specialized foundation model for special stains based on the vision transformer architecture. StainNet adopts a self-distillation SSL approach and is trained on over 1.4 million patch images cropping from 20,231 publicly available special staining WSIs in the HISTAI database. To evaluate StainNet, we conduct experiments on an in-house slide-level liver malignancy classification task and two public ROI-level datasets to demonstrate its strong ability. We also perform few-ratio learning and retrieval evaluations, and compare StainNet with recently larger PFMs to further highlight its strengths. We have released the StainNet model weights at: https://huggingface.co/JWonderLand/StainNet.


### 论文摘要

Foundation models trained with self-supervised learning (SSL) on large-scale histological images have significantly accelerated the development of computational pathology. These models can serve as backbones for region-of-interest (ROI) image analysis or patch-level feature extractors in whole-slide images (WSIs) based on multiple instance learning (MIL). Existing pathology foundation models (PFMs) are typically pre-trained on Hematoxylin-Eosin (H&E) stained pathology images. However, images with special stains, such as immunohistochemistry, are also frequently used in clinical practice. PFMs pre-trained mainly on H\&E-stained images may be limited in clinical applications involving special stains. To address this issue, we propose StainNet, a specialized foundation model for special stains based on the vision transformer (ViT) architecture. StainNet adopts a self-distillation SSL approach and is trained on over 1.4 million patch images cropping from 20,231 publicly available special staining WSIs in the HISTAI database. To evaluate StainNet, we conduct experiments on an in-house slide-level liver malignancy classification task and two public ROI-level datasets to demonstrate its strong ability. We also perform few-ratio learning and retrieval evaluations, and compare StainNet with recently larger PFMs to further highlight its strengths. We have released the StainNet model weights at: https://huggingface.co/JWonderLand/StainNet.

---

## 13. ABBSPO: Adaptive Bounding Box Scaling and Symmetric Prior based Orientation Prediction for Detecting Aerial Image Objects

**论文链接:** [http://arxiv.org/abs/2512.10031v1](http://arxiv.org/abs/2512.10031v1)

**作者:** Woojin Lee, Hyugjae Chang, Jaeho Moon, Jaehyup Lee, Munchurl Kim

**发布时间:** 2025-12-10

**备注:** 17 pages, 11 figures, 8 tables, supplementary included. Accepted to CVPR 2025. Please visit our project page at https://kaist-viclab.github.io/ABBSPO_site/

### GPT解析

### 总结

本文提出了一种名为ABBSPO的框架，用于弱监督方向目标检测，通过自适应边界框缩放和基于对称先验的方向预测，解决了现有HBox监督方法中尺度估计不准确和学习崩溃的问题。

### 背景

弱监督方向目标检测（WS-OOD）作为一种经济高效的替代方法，因其高效性和高准确性而受到关注。其中，水平边界框（HBox）监督的OOD方法因其能够直接利用现有的HBox标注并在弱监督设置下实现最高准确性而脱颖而出。

### 目的

解决现有HBox监督OOD方法的局限性，特别是当直接比较真实HBox与预测RBox的最小外接矩形时导致的尺度估计不准确问题，以及当所有三个增强视图预测都一致错误时的学习崩溃问题。

### 方法

提出ABBSPO框架，包含两个主要组件：(1)自适应边界框缩放（ABBS），适当缩放真实HBox以优化每个预测RBox的大小，确保更准确的尺度预测；(2)对称先验角（SPA）损失，利用航空目标的固有对称性进行自监督学习，解决当所有三个增强视图预测都一致错误时的学习崩溃问题。

### 主要发现

通过大量实验证明，ABBSPO实现了最先进的性能，优于现有方法。

### 结论

ABBSPO框架有效解决了现有HBox监督OOD方法的局限性，提高了弱监督方向目标检测的准确性。

### 翻译

弱监督方向目标检测（WS-OOD）作为一种经济高效的替代方法已受到关注，提供了高效性和高准确性。在弱监督方法中，水平边界框（HBox）监督的OOD因其能够直接利用现有HBox标注并在弱监督设置下实现最高准确性而脱颖而出。本文介绍了自适应边界框缩放和基于对称先验的方向预测，称为ABBSPO，用于WS-OOD的框架。我们的ABBSPO解决了之前HBox监督OOD方法的局限性，这些方法直接比较真实（GT）HBox与预测RBox的最小外接矩形，常导致不准确的尺度估计。为克服这一问题，我们提出：(i)自适应边界框缩放（ABBS），适当缩放GT HBox以优化每个预测RBox的大小，确保更准确的尺度预测；(ii)对称先验角（SPA）损失，利用航空目标的固有对称性进行自监督学习，解决了当所有三个增强视图（原始、旋转和翻转）的预测都一致错误时，之前方法中学习崩溃的问题。大量实验结果表明，ABBSPO实现了最先进的性能，优于现有方法。


### 论文摘要

Weakly supervised oriented object detection (WS-OOD) has gained attention as a cost-effective alternative to fully supervised methods, providing both efficiency and high accuracy. Among weakly supervised approaches, horizontal bounding box (HBox)-supervised OOD stands out for its ability to directly leverage existing HBox annotations while achieving the highest accuracy under weak supervision settings. This paper introduces adaptive bounding box scaling and symmetry-prior-based orientation prediction, called ABBSPO, a framework for WS-OOD. Our ABBSPO addresses limitations of previous HBox-supervised OOD methods, which compare ground truth (GT) HBoxes directly with the minimum circumscribed rectangles of predicted RBoxes, often leading to inaccurate scale estimation. To overcome this, we propose: (i) Adaptive Bounding Box Scaling (ABBS), which appropriately scales GT HBoxes to optimize for the size of each predicted RBox, ensuring more accurate scale prediction; and (ii) a Symmetric Prior Angle (SPA) loss that exploits inherent symmetry of aerial objects for self-supervised learning, resolving issues in previous methods where learning collapses when predictions for all three augmented views (original, rotated, and flipped) are consistently incorrect. Extensive experimental results demonstrate that ABBSPO achieves state-of-the-art performance, outperforming existing methods.

---

## 14. Physics-Informed Learning of Microvascular Flow Models using Graph Neural Networks

**论文链接:** [http://arxiv.org/abs/2512.10792v1](http://arxiv.org/abs/2512.10792v1)

**作者:** Paolo Botta, Piermario Vitullo, Thomas Ventimiglia, Andreas Linninger, Paolo Zunino

**发布时间:** 2025-12-11

**备注:** 28 pages, 8 figures

### GPT解析

### 总结

该研究提出了一种基于深度学习的降阶建模策略，利用图神经网络(GNN)在合成微血管图上训练，以近似解剖学真实区域中的血液动力学量，实现了微循环血流的高效模拟。

### 背景

在真实的血管结构中模拟微循环血流面临重大挑战，这主要由于问题的多尺度性质和毛细血管网络的拓扑复杂性。

### 目的

开发一种新的基于深度学习的降阶建模策略，利用图神经网络来近似解剖学真实区域中的血液动力学量，实现高效可靠的微循环血流模拟。

### 方法

结合合成血管生成算法和物理信息训练程序，整合图拓扑信息和局部流动动力学；采用从控制方程推导出的物理信息损失函数，强制执行质量守恒和流变学约束，确保学习替代模型的物理可靠性。

### 主要发现

GNN架构在不同网络配置上展现出强大的泛化能力；在具有线性和非线性流变学的基准问题上验证了GNN公式，能够准确重建压力和速度场，与全阶求解器相比具有显著的计算优势；在小鼠大脑皮层数据上的测试展示了该方法对血管复杂性的显著泛化能力。

### 结论

该研究建立了基于图的微血管流动替代模型的新类别，这些模型基于物理定律并具有反映质量守恒和流变学模型的归纳偏差，为血管建模和生物医学应用中的实时推理开辟了新方向。

### 翻译

在真实的血管结构中模拟微循环血流由于其问题的多尺度性质和毛细血管网络的拓扑复杂性而面临重大挑战。在这项工作中，我们提出了一种新颖的基于深度学习的降阶建模策略，利用在合成微血管图上训练的图神经网络(GNN)来近似解剖学真实区域中的血液动力学量。我们的方法结合了合成血管生成算法和物理信息训练程序，整合了图拓扑信息和局部流动动力学。为确保学习替代模型的物理可靠性，我们纳入了从控制方程推导出的物理信息损失函数，允许执行质量守恒和流变学约束。 resulting GNN architecture demonstrates robust generalization capabilities across diverse network configurations. The GNN formulation is validated on benchmark problems with linear and nonlinear rheology, showing accurate pressure and velocity field reconstruction with substantial computational gains over full-order solvers. The methodology showcases significant generalization capabilities with respect to vascular complexity, as highlighted by tests on data from the mouse cerebral cortex. This work establishes a new class of graph-based surrogate models for microvascular flow, grounded in physical laws and equipped with inductive biases that mirror mass conservation and rheological models, opening new directions for real-time inference in vascular modeling and biomedical applications.


### 论文摘要

The simulation of microcirculatory blood flow in realistic vascular architectures poses significant challenges due to the multiscale nature of the problem and the topological complexity of capillary networks. In this work, we propose a novel deep learning-based reduced-order modeling strategy, leveraging Graph Neural Networks (GNNs) trained on synthetic microvascular graphs to approximate hemodynamic quantities on anatomically realistic domains. Our method combines algorithms for synthetic vascular generation with a physics-informed training procedure that integrates graph topological information and local flow dynamics. To ensure the physical reliability of the learned surrogates, we incorporate a physics-informed loss functional derived from the governing equations, allowing enforcement of mass conservation and rheological constraints. The resulting GNN architecture demonstrates robust generalization capabilities across diverse network configurations. The GNN formulation is validated on benchmark problems with linear and nonlinear rheology, showing accurate pressure and velocity field reconstruction with substantial computational gains over full-order solvers. The methodology showcases significant generalization capabilities with respect to vascular complexity, as highlighted by tests on data from the mouse cerebral cortex. This work establishes a new class of graph-based surrogate models for microvascular flow, grounded in physical laws and equipped with inductive biases that mirror mass conservation and rheological models, opening new directions for real-time inference in vascular modeling and biomedical applications.

---

## 15. LGAN: An Efficient High-Order Graph Neural Network via the Line Graph Aggregation

**论文链接:** [http://arxiv.org/abs/2512.10735v1](http://arxiv.org/abs/2512.10735v1)

**作者:** Lin Du, Lu Bai, Jincheng Li, Lixin Cui, Hangyuan Du, Lichi Zhang, Yuting Chen, Zhao Li

**发布时间:** 2025-12-11

### GPT解析

### 总结

本文提出了一种新的线图聚合网络(LGAN)，通过构建从每个节点为中心的诱导子图生成的线图来执行高阶聚合，克服了现有GNNs在表达能力和可解释性方面的局限性。

### 背景

图神经网络(GNNs)已成为图分类的主导范式，但现有GNN主要依赖邻居节点间的消息传递策略，其表达能力受限于1维Weisfeiler-Lehman(1-WL)测试。

### 目的

克服现有基于k-WL的GNNs计算成本高且可解释性差的问题，提出一种具有更强表达能力和更好可解释性的新型GNN架构。

### 方法

提出线图聚合网络(LGAN)，通过构建从每个节点为中心的诱导子图生成的线图来执行高阶聚合，理论上证明在注入聚合假设下，LGAN比2-WL具有更强的表达能力且时间复杂度更低。

### 主要发现

经验评估表明，LGAN在基准测试上优于最先进的基于k-WL的GNNs，同时提供更好的可解释性。

### 结论

LGAN是一种有效的图神经网络架构，能够在保持较低计算成本的同时提供更强的表达能力和更好的可解释性。

### 翻译

图神经网络(GNNs)已成为图分类的主导范式。具体而言，大多数现有GNN主要依赖于邻居节点间的消息传递策略，其表达能力受限于1维Weisfeiler-Lehman(1-WL)测试。尽管已经提出了一些基于k-WL的GNNs来克服这一限制，但它们的计算成本随k值增加而迅速增长，显著限制了实际应用性。此外，由于k-WL模型主要操作在节点元组上，这些基于k-WL的GNNs无法保留归因方法(如Integrated Gradients)所需的细粒度节点或边级语义，导致可解释性较差。为了克服上述缺点，在本文中，我们提出了一种新颖的线图聚合网络(LGAN)，它从每个节点为中心的诱导子图构建线图以执行高阶聚合。我们从理论上证明，在注入聚合假设下，LGAN不仅比2-WL具有更强的表达能力，而且具有更低的时间复杂度。在基准测试上的经验评估表明，LGAN优于最先进的基于k-WL的GNNs，同时提供更好的可解释性。


### 论文摘要

Graph Neural Networks (GNNs) have emerged as a dominant paradigm for graph classification. Specifically, most existing GNNs mainly rely on the message passing strategy between neighbor nodes, where the expressivity is limited by the 1-dimensional Weisfeiler-Lehman (1-WL) test. Although a number of k-WL-based GNNs have been proposed to overcome this limitation, their computational cost increases rapidly with k, significantly restricting the practical applicability. Moreover, since the k-WL models mainly operate on node tuples, these k-WL-based GNNs cannot retain fine-grained node- or edge-level semantics required by attribution methods (e.g., Integrated Gradients), leading to the less interpretable problem. To overcome the above shortcomings, in this paper, we propose a novel Line Graph Aggregation Network (LGAN), that constructs a line graph from the induced subgraph centered at each node to perform the higher-order aggregation. We theoretically prove that the LGAN not only possesses the greater expressive power than the 2-WL under injective aggregation assumptions, but also has lower time complexity. Empirical evaluations on benchmarks demonstrate that the LGAN outperforms state-of-the-art k-WL-based GNNs, while offering better interpretability.

---

## 16. THeGAU: Type-Aware Heterogeneous Graph Autoencoder and Augmentation

**论文链接:** [http://arxiv.org/abs/2512.10589v1](http://arxiv.org/abs/2512.10589v1)

**作者:** Ming-Yi Hong, Miao-Chen Chiang, Youchen Teng, Yu-Hsiang Wang, Chih-Yu Wang, Che Lin

**发布时间:** 2025-12-11

### GPT解析

### 总结

THeGAU是一个模型无关的框架，结合类型感知图自编码器和引导图增强，通过重建模式有效边和选择性优化噪声结构，解决了HGNNs中的类型信息丢失和结构噪声问题，在多个基准数据集上实现了最先进的节点分类性能。

### 背景

异构图神经网络（HGNNs）对建模异构信息网络（HINs）很有效，这些网络编码复杂的多类型实体和关系。然而，HGNNs常常遭受类型信息丢失和结构噪声问题，限制了它们的表示保真度和泛化能力。

### 目的

提出一个名为THeGAU的模型无关框架，结合类型感知图自编码器和引导图增强来改进节点分类，保留节点类型语义并选择性优化噪声结构，提高鲁棒性、准确性和效率，同时显著减少计算开销。

### 方法

提出THeGAU框架，结合类型感知图自编码器和引导图增强，通过重建模式有效边作为辅助任务来保留节点类型语义，引入解码器驱动的增强机制来选择性优化噪声结构，这种联合设计增强了鲁棒性、准确性和效率。

### 主要发现

在三个基准HIN数据集（IMDB、ACM和DBLP）上的广泛实验表明，THeGAU一致性地优于现有的HGNN方法，在多个骨干网络上实现了最先进的性能。

### 结论

THeGAU框架有效解决了HGNNs中的类型信息丢失和结构噪声问题，通过联合设计提高了模型的鲁棒性、准确性和效率，在多个数据集上证明了其优越性。

### 翻译

异构图神经网络（HGNNs）对建模异构信息网络（HINs）很有效，这些网络编码复杂的多类型实体和关系。然而，HGNNs常常遭受类型信息丢失和结构噪声问题，限制了它们的表示保真度和泛化能力。我们提出了THeGAU，这是一个模型无关的框架，结合类型感知图自编码器和引导图增强来改进节点分类。THeGAU将重建模式有效边作为辅助任务以保留节点类型语义，并引入了解码器驱动的增强机制来选择性优化噪声结构。这种联合设计增强了鲁棒性、准确性和效率，同时显著减少了计算开销。在三个基准HIN数据集（IMDB、ACM和DBLP）上的广泛实验表明，THeGAU一致性地优于现有的HGNN方法，在多个骨干网络上实现了最先进的性能。


### 论文摘要

Heterogeneous Graph Neural Networks (HGNNs) are effective for modeling Heterogeneous Information Networks (HINs), which encode complex multi-typed entities and relations. However, HGNNs often suffer from type information loss and structural noise, limiting their representational fidelity and generalization. We propose THeGAU, a model-agnostic framework that combines a type-aware graph autoencoder with guided graph augmentation to improve node classification. THeGAU reconstructs schema-valid edges as an auxiliary task to preserve node-type semantics and introduces a decoder-driven augmentation mechanism to selectively refine noisy structures. This joint design enhances robustness, accuracy, and efficiency while significantly reducing computational overhead. Extensive experiments on three benchmark HIN datasets (IMDB, ACM, and DBLP) demonstrate that THeGAU consistently outperforms existing HGNN methods, achieving state-of-the-art performance across multiple backbones.

---

## 17. Mr. Virgil: Learning Multi-robot Visual-range Relative Localization

**论文链接:** [http://arxiv.org/abs/2512.10540v1](http://arxiv.org/abs/2512.10540v1)

**作者:** Si Wang, Zhehan Li, Jiadong Lu, Rong Xiong, Yanjun Cao, Yue Wang

**发布时间:** 2025-12-11

**备注:** Accepted by 2025 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)

### GPT解析

### 总结

本文介绍了一种名为'Mr. Virgil'的端到端学习多机器人视觉-距离相对定位框架，用于解决机器人与视觉检测之间的匹配问题。

### 背景

超宽带视觉融合定位在多智能体相对定位领域有广泛应用，但现有方法高度依赖于身份编码硬件或精细调整算法，且错误匹配可能对定位系统造成不可逆损害。

### 目的

解决机器人与视觉检测之间的匹配问题，减少对特定硬件的依赖，提高定位系统的鲁棒性和准确性。

### 方法

提出'Mr. Virgil'框架，包含用于超宽带测距和视觉检测数据关联的图神经网络前端，以及可微分的姿态图优化后端；实现分布式系统以适应实际应用需求。

### 主要发现

在不同机器人数量、模拟和真实世界、遮挡和非遮挡条件下的实验表明，该方法在各种场景下均展现出稳定性和精确性，优于传统方法。

### 结论

Mr. Virgil框架有效解决了机器人与视觉检测之间的匹配问题，减少了对特定硬件的依赖，通过端到端学习提高了定位精度和系统鲁棒性。

### 翻译

超宽带视觉融合定位已在多智能体相对定位领域得到广泛应用。机器人与视觉检测之间的困难匹配问题使得现有方法高度依赖于身份编码硬件或精细调整的算法。过度自信但错误的匹配可能对定位系统造成不可逆的损害。为了解决这个问题，我们引入了Mr. Virgil，一个端到端学习的多机器人视觉-距离相对定位框架，包括用于超宽带测距和视觉检测之间数据关联的图神经网络，以及一个可微分的姿态图优化后端。基于图的前端提供稳健的匹配结果、准确的位置预测和可信的不确定性估计，这些随后被整合到PGO后端以提高最终姿态估计的准确性。此外，还实现了适用于实际应用的分布式系统。跨越不同机器人数量、模拟和真实世界、遮挡和非遮挡条件的实验表明，与传统方法相比，该方法在各种场景下都展现出稳定性和精确性。我们的代码可在以下网址获取：https://github.com/HiOnes/Mr-Virgil。


### 论文摘要

Ultra-wideband (UWB)-vision fusion localization has achieved extensive applications in the domain of multi-agent relative localization. The challenging matching problem between robots and visual detection renders existing methods highly dependent on identity-encoded hardware or delicate tuning algorithms. Overconfident yet erroneous matches may bring about irreversible damage to the localization system. To address this issue, we introduce Mr. Virgil, an end-to-end learning multi-robot visual-range relative localization framework, consisting of a graph neural network for data association between UWB rangings and visual detections, and a differentiable pose graph optimization (PGO) back-end. The graph-based front-end supplies robust matching results, accurate initial position predictions, and credible uncertainty estimates, which are subsequently integrated into the PGO back-end to elevate the accuracy of the final pose estimation. Additionally, a decentralized system is implemented for real-world applications. Experiments spanning varying robot numbers, simulation and real-world, occlusion and non-occlusion conditions showcase the stability and exactitude under various scenes compared to conventional methods. Our code is available at: https://github.com/HiOnes/Mr-Virgil.

---

## 18. From Lab to Reality: A Practical Evaluation of Deep Learning Models and LLMs for Vulnerability Detection

**论文链接:** [http://arxiv.org/abs/2512.10485v1](http://arxiv.org/abs/2512.10485v1)

**作者:** Chaomeng Lu, Bert Lagaisse

**发布时间:** 2025-12-11

### GPT解析

### 总结

本研究评估了基于深度学习的漏洞检测方法在现实场景中的有效性，发现当前模型在基准数据集上表现良好，但在真实环境中的泛化能力有限。

### 背景

基于深度学习的漏洞检测方法在基准数据集上表现出色，但现实世界中的有效性尚未充分探索。图神经网络(GNN)和transformer模型（包括大型语言模型）在具有一致数据分布和启发式或部分噪声标签的基准数据集上显示出有希望的结果。

### 目的

系统性评估两种代表性深度学习模型（ReVeal和LineVul）在四个代表性数据集（Juliet、Devign、BigVul和ICVul）上的表现，并在包含最近修复的Linux内核漏洞的数据集（VentiVul）上部署这些模型及四个预训练大型语言模型。

### 方法

在各个数据集上独立训练每个模型，使用t-SNE分析代码表示以发现与漏洞相关的模式，并在VentiVul时间外分布数据集上评估模型性能。

### 主要发现

当前模型难以在表示空间中区分易受攻击和不易受攻击的代码，在不同分布的数据集上泛化能力差，在VentiVul上评估时性能急剧下降，大多数模型无法可靠检测漏洞。

### 结论

学术基准和实际部署之间存在持续差距，强调了面向部署的评估框架的价值，以及需要更强大的代码表示和更高质量数据集的必要性。

### 翻译

基于深度学习的漏洞检测方法在基准数据集上显示出强大性能，但它们在现实世界中的有效性仍需进一步探索。最近的研究表明，当在精选的基准数据集上评估时，基于图神经网络(GNN)和transformer的模型（包括大型语言模型LLMs）都取得了有希望的结果。这些数据集通常具有一致的数据分布和启发式或部分噪声标签。在本研究中，我们在四个代表性数据集（Juliet、Devign、BigVul和ICVul）上系统评估了两种代表性的深度学习模型——ReVeal和LineVul。每个模型在每个相应数据集上独立训练，并使用t-SNE分析其代码表示以揭示与漏洞相关的模式。为了评估实际适用性，我们在VentiVul（一个包含20个最近（2025年5月）修复的Linux内核漏洞的精选数据集）上部署了这些模型以及四个预训练的大型语言模型：Claude 3.5 Sonnet、GPT-o3-mini、GPT-4o和GPT-5。我们的实验揭示，当前模型难以在表示空间中区分易受攻击和不易受攻击的代码，并且在具有不同分布的数据集上泛化能力差。当在我们新构建的时间外分布数据集VentiVul上评估时，性能急剧下降，大多数模型无法可靠地检测漏洞。这些结果揭示了学术基准和实际部署之间的持续差距，强调了我们的面向部署的评估框架的价值，以及需要更强大的代码表示和更高质量数据集的必要性。


### 论文摘要

Vulnerability detection methods based on deep learning (DL) have shown strong performance on benchmark datasets, yet their real-world effectiveness remains underexplored. Recent work suggests that both graph neural network (GNN)-based and transformer-based models, including large language models (LLMs), yield promising results when evaluated on curated benchmark datasets. These datasets are typically characterized by consistent data distributions and heuristic or partially noisy labels. In this study, we systematically evaluate two representative DL models-ReVeal and LineVul-across four representative datasets: Juliet, Devign, BigVul, and ICVul. Each model is trained independently on each respective dataset, and their code representations are analyzed using t-SNE to uncover vulnerability related patterns. To assess realistic applicability, we deploy these models along with four pretrained LLMs, Claude 3.5 Sonnet, GPT-o3-mini, GPT-4o, and GPT-5 on a curated dataset, VentiVul, comprising 20 recently (May 2025) fixed vulnerabilities from the Linux kernel. Our experiments reveal that current models struggle to distinguish vulnerable from non-vulnerable code in representation space and generalize poorly across datasets with differing distributions. When evaluated on VentiVul, our newly constructed time-wise out-of-distribution dataset, performance drops sharply, with most models failing to detect vulnerabilities reliably. These results expose a persistent gap between academic benchmarks and real-world deployment, emphasizing the value of our deployment-oriented evaluation framework and the need for more robust code representations and higher-quality datasets.

---

## 19. Better Prevent than Tackle: Valuing Defense in Soccer Based on Graph Neural Networks

**论文链接:** [http://arxiv.org/abs/2512.10355v1](http://arxiv.org/abs/2512.10355v1)

**作者:** Hyunsung Kim, Sangwoo Seo, Hoyoung Choi, Tom Boomstra, Jinsung Yoon, Chanyoung Park

**发布时间:** 2025-12-11

### GPT解析

### 总结

论文提出了DEFCON框架，用于量化足球中的防守贡献，解决了传统方法只关注可见控球动作而忽视防守真正影响的问题。

### 背景

评估足球防守表现具有挑战性，因为有效防守通常不是通过可见的控球动作体现，而是通过预防危险机会。现有方法主要关注控球动作价值，忽略了防守者的真正影响。

### 目的

提出DEFCON框架，量化足球中每次进攻情况下的球员级别防守贡献，填补现有研究的空白。

### 方法

DEFCON利用图注意力网络估计每个进攻选项的成功概率和预期价值，以及每个防守者的责任。通过比较进攻前后的预期持球价值(EPV)来分配防守积分。在2023-24赛季训练，2024-25赛季荷甲数据上评估。

### 主要发现

DEFCON的聚合球员积分与市场估值表现出强烈的正相关。

### 结论

展示了多个实际应用，包括防守贡献时间线、球场区域空间分析以及进攻者-防守者互动成对总结。

### 翻译

评估足球中的防守表现仍然具有挑战性，因为有效的防守通常不是通过可见的控球动作（如拦截和抢断）来体现，而是在危险机会出现前阻止它们。现有方法主要侧重于评估控球动作的价值，使得防守者的真实影响在很大程度上未被衡量。为了解决这一差距，我们提出了DEFCON（防守贡献评估器），一个全面的框架，用于量化足球中每次进攻情况下的球员级别防守贡献。利用图注意力网络，DEFCON估计每个进攻选项的成功概率和预期价值，以及每个防守者阻止它的责任。这些组件为进攻团队在每次动作前后的预期持球价值（EPV）提供参考，并根据防守者是否减少了或增加了对手的EPV来分配正面或负面的积分。在2023-24赛季进行训练，并在2024-25赛季的荷甲联赛事件和跟踪数据上进行评估，DEFCON的聚合球员积分与市场估值表现出强烈的正相关。最后，我们展示了几个实际应用，包括比赛中的防守贡献时间线、球场区域的空间分析以及进攻者-防守者互动的成对总结。


### 论文摘要

Evaluating defensive performance in soccer remains challenging, as effective defending is often expressed not through visible on-ball actions such as interceptions and tackles, but through preventing dangerous opportunities before they arise. Existing approaches have largely focused on valuing on-ball actions, leaving much of defenders' true impact unmeasured. To address this gap, we propose DEFCON (DEFensive CONtribution evaluator), a comprehensive framework that quantifies player-level defensive contributions for every attacking situation in soccer. Leveraging Graph Attention Networks, DEFCON estimates the success probability and expected value of each attacking option, along with each defender's responsibility for stopping it. These components yield an Expected Possession Value (EPV) for the attacking team before and after each action, and DEFCON assigns positive or negative credits to defenders according to whether they reduced or increased the opponent's EPV. Trained on 2023-24 and evaluated on 2024-25 Eredivisie event and tracking data, DEFCON's aggregated player credits exhibit strong positive correlations with market valuations. Finally, we showcase several practical applications, including in-game timelines of defensive contributions, spatial analyses across pitch zones, and pairwise summaries of attacker-defender interactions.

---

## 20. A Kernel-based Resource-efficient Neural Surrogate for Multi-fidelity Prediction of Aerodynamic Field

**论文链接:** [http://arxiv.org/abs/2512.10287v1](http://arxiv.org/abs/2512.10287v1)

**作者:** Apurba Sarker, Reza T. Batley, Darshan Sarojini, Sourav Saha

**发布时间:** 2025-12-11

**备注:** 24 pages, 15 figures

### GPT解析

### 总结

本研究提出了一种名为KHRONOS的基于核的神经替代模型，用于在资源受限条件下高效预测空气动力学场，通过混合高保真度和低保真度数据实现了比传统密集神经网络更高的效率。

### 背景

替代模型为昂贵的空气动力学仿真提供了快速替代方案，在设计和优化应用中极其有用。

### 目的

提出使用KHRONOS模型，混合稀疏高保真度数据与低保真度信息，以预测不同计算资源约束下的空气动力学场。

### 方法

KHRONOS基于变分原理、插值理论和张量分解构建，使用AirfRANS数据集作为高保真度基准，NeuralFoil生成低保真度对应数据，并与MLP、GNN和PINN三种模型架构比较，在不同数据可用性(0%、10%、30%)和复杂几何参数化条件下预测翼型表面压力系数分布。

### 主要发现

所有模型最终都能实现相当的预测准确性，但在资源受限条件下，KHRONOS表现优异，需要数量级更少的可训练参数，并且在可比精度下提供更快的训练和推理速度。

### 结论

KHRONOS和类似架构在多保真度空气动力学场预测中具有平衡准确性和效率的潜力。

### 翻译

替代模型为昂贵的空气动力学仿真提供了快速替代方案，在设计和优化应用中极其有用。本研究提出使用一种最近的基于核的神经替代模型KHRONOS。在本工作中，我们将稀疏高保真度数据与低保真度信息相结合，以预测不同计算资源约束下的空气动力学场。与传统方法不同，KHRONOS基于变分原理、插值理论和张量分解构建。这些元素为密集神经网络的大量剪枝提供了数学基础。使用AirfRANS数据集作为高保真度基准，NeuralFoil生成低保真度对应数据，本研究将KHRONOS的性能与三种当代模型架构进行比较：多层感知器(MLP)、图神经网络(GNN)和物理信息神经网络(PINN)。我们考虑不同水平的高保真度数据可用性(0%、10%和30%)以及越来越复杂的几何参数化。这些被用于预测翼型表面的压力系数分布。结果表明，虽然所有模型最终都能实现相当的预测准确性，但KHRONOS在资源受限条件下表现优异。在此领域，KHRONOS始终需要数量级更少的可训练参数，并以可比的精度提供比当代密集神经网络快得多的训练和推理速度。这些发现凸显了KHRONOS及类似架构在多保真度空气动力学场预测中平衡准确性和效率的潜力。


### 论文摘要

Surrogate models provide fast alternatives to costly aerodynamic simulations and are extremely useful in design and optimization applications. This study proposes the use of a recent kernel-based neural surrogate, KHRONOS. In this work, we blend sparse high-fidelity (HF) data with low-fidelity (LF) information to predict aerodynamic fields under varying constraints in computational resources. Unlike traditional approaches, KHRONOS is built upon variational principles, interpolation theory, and tensor decomposition. These elements provide a mathematical basis for heavy pruning compared to dense neural networks. Using the AirfRANS dataset as a high-fidelity benchmark and NeuralFoil to generate low-fidelity counterparts, this work compares the performance of KHRONOS with three contemporary model architectures: a multilayer perceptron (MLP), a graph neural network (GNN), and a physics-informed neural network (PINN). We consider varying levels of high-fidelity data availability (0%, 10%, and 30%) and increasingly complex geometry parameterizations. These are used to predict the surface pressure coefficient distribution over the airfoil. Results indicate that, whilst all models eventually achieve comparable predictive accuracy, KHRONOS excels in resource-constrained conditions. In this domain, KHRONOS consistently requires orders of magnitude fewer trainable parameters and delivers much faster training and inference than contemporary dense neural networks at comparable accuracy. These findings highlight the potential of KHRONOS and similar architectures to balance accuracy and efficiency in multi-fidelity aerodynamic field prediction.

---

## 21. Graph Neural Network Based Adaptive Threat Detection for Cloud Identity and Access Management Logs

**论文链接:** [http://arxiv.org/abs/2512.10280v1](http://arxiv.org/abs/2512.10280v1)

**作者:** Venkata Tanuja Madireddy

**发布时间:** 2025-12-11

### GPT解析

### 总结

本文提出了一种基于图神经网络的自适应威胁检测框架，用于从IAM审计日志中实时学习潜在的用户资源交互模式，有效识别传统检测系统难以发现的身份和访问管理中的新型威胁。

### 背景

云基础设施和分布式身份系统的快速扩张增加了现代企业的复杂性和攻击面，而传统基于规则或签名的检测系统在识别IAM日志中新型或 evolving 威胁方面往往不足，因为异常行为可能在统计上看似良性但在上下文中是恶意的。

### 目的

设计一个能够实时从IAM审计日志中学习潜在用户资源交互模式的自适应威胁检测框架，以应对云环境中的安全威胁。

### 方法

将IAM日志建模为异构动态图，捕获用户、角色、会话和访问操作等实体之间的时间、关系和上下文依赖，并采用基于注意力的聚合和图嵌入更新机制，使模型能够持续适应变化的云环境。

### 主要发现

在合成的和真实的IAM数据集上的实验评估表明，所提出的方法比基线LSTM和GCN分类器实现了更高的检测精度和召回率，同时保持了在多租户云环境中的可扩展性，能够主动缓解内部威胁、权限提升和横向移动攻击。

### 结论

这项工作弥合了基于图的机器学习与操作云安全智能之间的差距，为AI驱动的零信任访问分析奠定了基础。

### 翻译

云基础设施和分布式身份系统的快速扩张显著增加了现代企业的复杂性和攻击面。传统的基于规则或签名的检测系统在识别身份和访问管理日志中的新型或 evolving 威胁时往往不足，因为异常行为可能在统计上看似良性但在上下文中是恶意的。本文提出了一个基于图神经网络的自适应威胁检测框架，旨在从IAM审计日志中实时学习潜在的用户资源交互模式。通过将IAM日志建模为异构动态图，所提出的系统能够捕获用户、角色、会话和访问操作等实体之间的时间、关系和上下文依赖。该模型结合了基于注意力的聚合和图嵌入更新，使其能够持续适应变化的云环境。在合成和真实IAM数据集上的实验评估表明，所提出的方法比基线LSTM和GCN分类器实现了更高的检测精度和召回率，同时在多租户云环境中保持了可扩展性。该框架的适应性使能够主动缓解内部威胁、权限提升和横向移动攻击，为AI驱动的零信任访问分析奠定了基础。这项工作弥合了基于图的机器学习与操作云安全智能之间的差距。


### 论文摘要

The rapid expansion of cloud infrastructures and distributed identity systems has significantly increased the complexity and attack surface of modern enterprises. Traditional rule based or signature driven detection systems are often inadequate in identifying novel or evolving threats within Identity and Access Management logs, where anomalous behavior may appear statistically benign but contextually malicious. This paper presents a Graph Neural Network Based Adaptive Threat Detection framework designed to learn latent user resource interaction patterns from IAM audit trails in real time. By modeling IAM logs as heterogeneous dynamic graphs, the proposed system captures temporal, relational, and contextual dependencies across entities such as users, roles, sessions, and access actions. The model incorporates attention based aggregation and graph embedding updates to enable continual adaptation to changing cloud environments. Experimental evaluation on synthesized and real world IAM datasets demonstrates that the proposed method achieves higher detection precision and recall than baseline LSTM and GCN classifiers, while maintaining scalability across multi tenant cloud environments. The frameworks adaptability enables proactive mitigation of insider threats, privilege escalation, and lateral movement attacks, contributing to the foundation of AI driven zero trust access analytics. This work bridges the gap between graph based machine learning and operational cloud security intelligence.

---

## 22. Galaxy Phase-Space and Field-Level Cosmology: The Strength of Semi-Analytic Models

**论文链接:** [http://arxiv.org/abs/2512.10222v1](http://arxiv.org/abs/2512.10222v1)

**作者:** Natalí S. M. de Santi, Francisco Villaescusa-Navarro, Pablo Araya-Araya, Gabriella De Lucia, Fabio Fontanot, Lucia A. Perez, Manuel Arnés-Curto, Violeta Gonzalez-Perez, Ángel Chandro-Gómez, Rachel S. Somerville, Tiago Castro

**发布时间:** 2025-12-11

**备注:** 23 pages, 5 figures

### GPT解析

### 总结

研究使用图神经网络与矩神经网络相结合，仅基于星系三维位置和径向 velocities，实现了对物质密度参数Ω_m的约10%精度估计，模型在不同模拟和参数设置下表现出鲁棒性。

### 背景

半解析模型是在宇宙学框架中模拟星系属性的常用方法，依赖简化但具物理动机的方案，比完全流体动力学模拟更快且计算成本更低。

### 目的

开发一种机器学习模型，仅使用星系三维位置和径向速度，来估计物质密度参数Ω_m，并验证其在不同模拟中的适用性。

### 方法

训练图神经网络与矩神经网络，使用来自L-Galaxies星系目录的(25 h^-1 Mpc)^3体积数据，并将预测外推到其他半解析模型和流体动力学模拟。

### 主要发现

网络能以约10%的精度估计物质密度参数Ω_m，对天体物理学和亚网格物理、宇宙学和天体物理学参数以及不同晕轮廓处理具有鲁棒性。

### 结论

半解析模型相空间中编码的物理关系在很大程度上独立于特定物理方案，强化了它们作为宇宙学参数推断的真实模拟目录生成工具的潜力。

### 翻译

半解析模型是在宇宙学框架中模拟星系属性的广泛使用方法，依赖于简化但具有物理动机的方案。它们也被证明是生成精确星系目录的有效替代方案，与完全流体动力学模拟相比提供了更快且计算成本更低的选择。在本文中，我们证明仅使用星系三维位置和径向速度，我们可以训练图神经网络与矩神经网络相结合，获得一个稳健的基于机器学习的模型，能够以约10%的精度估计物质密度参数Ω_m。该网络在来自L-Galaxies的星系目录的(25 h^-1 Mpc)^3体积上训练，可以成功将其预测外推到其他半解析模型（GAEA、SC-SAM和Shark），以及更令人瞩目的是，外推到流体动力学模拟（Astrid、SIMBA、IllustrisTNG和SWIFT-EAGLE）。我们的结果表明，网络对天体物理学和亚网格物理、宇宙学和天体物理学参数以及不同模拟中使用的晕轮廓处理的差异具有鲁棒性。这表明半解析模型相空间中编码的物理关系在很大程度上独立于其特定的物理方案，强化了它们作为用于宇宙学参数推断的真实模拟目录生成工具的潜力。


### 论文摘要

Semi-analytic models are a widely used approach to simulate galaxy properties within a cosmological framework, relying on simplified yet physically motivated prescriptions. They have also proven to be an efficient alternative for generating accurate galaxy catalogs, offering a faster and less computationally expensive option compared to full hydrodynamical simulations. In this paper, we demonstrate that using only galaxy $3$D positions and radial velocities, we can train a graph neural network coupled to a moment neural network to obtain a robust machine learning based model capable of estimating the matter density parameters, $Ω_{\rm m}$, with a precision of approximately 10%. The network is trained on ($25 h^{-1}$Mpc)$^3$ volumes of galaxy catalogs from L-Galaxies and can successfully extrapolate its predictions to other semi-analytic models (GAEA, SC-SAM, and Shark) and, more remarkably, to hydrodynamical simulations (Astrid, SIMBA, IllustrisTNG, and SWIFT-EAGLE). Our results show that the network is robust to variations in astrophysical and subgrid physics, cosmological and astrophysical parameters, and the different halo-profile treatments used across simulations. This suggests that the physical relationships encoded in the phase-space of semi-analytic models are largely independent of their specific physical prescriptions, reinforcing their potential as tools for the generation of realistic mock catalogs for cosmological parameter inference.

---

## 23. Modeling Narrative Archetypes in Conspiratorial Narratives: Insights from Singapore-Based Telegram Groups

**论文链接:** [http://arxiv.org/abs/2512.10105v1](http://arxiv.org/abs/2512.10105v1)

**作者:** Soorya Ram Shimgekar, Abhay Goyal, Lam Yin Cheung, Roy Ka-Wei Lee, Koustuv Saha, Pi Zonooz, Navin Kumar

**发布时间:** 2025-12-10

### GPT解析

### 总结

该研究分析了新加坡Telegram群组中的阴谋论叙事，提出了一种两阶段计算框架，使用微调的RoBERTa-large模型和带符号信念图神经网络(SiBeGNN)识别出七种阴谋论叙事原型，发现阴谋论内容融入日常讨论而非仅限于孤立回音室。

### 背景

阴谋论话语日益嵌入数字通信生态系统中，但其结构和传播仍然难以研究。传统观点认为阴谋论主要存在于孤立的回音室中。

### 目的

分析阴谋论话语在数字通信生态系统中的结构和传播模式，理解阴谋论内容如何融入日常讨论而非仅限于孤立的回音室。

### 方法

提出两阶段计算框架：第一阶段微调RoBERTa-large模型对消息进行阴谋论分类，在2000条专家标注消息上达到0.866的F1分数；第二阶段构建带符号信念图，引入SiBeGNN使用符号解纠缠损失学习嵌入，分离意识形态特征和文体特征，并进行层次聚类分析553,648条消息。

### 主要发现

识别出七种叙事原型：法律话题、医疗问题、媒体讨论、金融、权威矛盾、群组管理和一般聊天。SiBeGNN聚类质量(cDBI=8.38)显著优于基线方法(13.60-67.27)，专家评估中88%达成一致。阴谋论消息不仅出现在怀疑或不信任的集群中，也出现在金融、法律和日常事务的常规讨论中。

### 结论

这些发现挑战了关于网络激进的常见假设，证明阴谋论话语在普通社交互动中运作。所提框架推动了基于信念的话语分析计算方法发展，为立场检测、政治传播研究和内容审核政策提供了应用。

### 翻译

阴谋论话语日益嵌入数字通信生态系统中，但其结构和传播仍然难以研究。这项工作分析了新加坡Telegram群组中的阴谋论叙事，表明此类内容融入日常讨论而非局限于孤立的回音室。我们提出一个两阶段计算框架。首先，我们微调RoBERTa-large模型将消息分类为阴谋论或非阴谋论，在2000条专家标注的消息上达到0.866的F1分数。其次，我们构建一个带符号的信念图，其中节点代表消息，边符号反映信念标签的一致性，权重为文本相似度。我们引入带符号信念图神经网络(SiBeGNN)，使用符号解纠缠损失学习嵌入，将意识形态特征与文体特征分离。利用这些嵌入进行层次聚类，我们在553,648条消息中识别出七种叙事原型：法律话题、医疗问题、媒体讨论、金融、权威矛盾、群组管理和一般聊天。SiBeGNN产生的聚类质量(cDBI=8.38)优于基线方法(13.60至67.27)，专家评估中88%达成一致。我们的分析表明，阴谋论消息不仅出现在专注于怀疑或不信任的集群中，也出现在金融、法律和日常事务的常规讨论中。这些发现通过证明阴谋论话语在普通社交互动中运作，挑战了关于网络激进的常见假设。所提出的框架推动了基于信念的话语分析计算方法发展，并为立场检测、政治传播研究和内容审核政策提供了应用。


### 论文摘要

Conspiratorial discourse is increasingly embedded within digital communication ecosystems, yet its structure and spread remain difficult to study. This work analyzes conspiratorial narratives in Singapore-based Telegram groups, showing that such content is woven into everyday discussions rather than confined to isolated echo chambers. We propose a two-stage computational framework. First, we fine-tune RoBERTa-large to classify messages as conspiratorial or not, achieving an F1-score of 0.866 on 2,000 expert-labeled messages. Second, we build a signed belief graph in which nodes represent messages and edge signs reflect alignment in belief labels, weighted by textual similarity. We introduce a Signed Belief Graph Neural Network (SiBeGNN) that uses a Sign Disentanglement Loss to learn embeddings that separate ideological alignment from stylistic features.   Using hierarchical clustering on these embeddings, we identify seven narrative archetypes across 553,648 messages: legal topics, medical concerns, media discussions, finance, contradictions in authority, group moderation, and general chat. SiBeGNN yields stronger clustering quality (cDBI = 8.38) than baseline methods (13.60 to 67.27), supported by 88 percent inter-rater agreement in expert evaluations. Our analysis shows that conspiratorial messages appear not only in clusters focused on skepticism or distrust, but also within routine discussions of finance, law, and everyday matters. These findings challenge common assumptions about online radicalization by demonstrating that conspiratorial discourse operates within ordinary social interaction. The proposed framework advances computational methods for belief-driven discourse analysis and offers applications for stance detection, political communication studies, and content moderation policy.

---

## 24. \textsc{Text2Graph}: Combining Lightweight LLMs and GNNs for Efficient Text Classification in Label-Scarce Scenarios

**论文链接:** [http://arxiv.org/abs/2512.10061v1](http://arxiv.org/abs/2512.10061v1)

**作者:** João Lucas Luz Lima Sarcinelli, Ricardo Marcondes Marcacini

**发布时间:** 2025-12-10

### GPT解析

### 总结

Text2Graph是一个开源的Python包，通过结合大型语言模型的部分标注和图神经网络的标签传播，提供了一种更可持续的文本分类方法，能够在保持竞争力的同时显著降低能源消耗和环境影响。

### 背景

大型语言模型已成为有效的零样本分类器，但其高计算需求和环境成本限制了它们在高性能计算环境中大规模标注的实用性。

### 目的

为了支持更可持续的工作流程，作者提出了Text2Graph，这是一个开源的Python包，提供了现有文本到图分类方法的模块化实现。

### 方法

Text2Graph框架使用户能够以灵活的方式将基于大型语言模型的部分标注与图神经网络标签传播相结合，使用户可以轻松交换组件，如特征提取器、边构建方法和采样策略。

### 主要发现

作者在五个涵盖主题分类和情感分析任务的零样本设置上对Text2Graph进行了基准测试，将多个变体与其他文本分类的零样本方法进行了比较。除了报告性能外，还提供了能源消耗和碳排放的详细估计，表明基于图的传播方法能够在能源和环境成本较低的情况下实现具有竞争力的结果。

### 结论

基于图的传播方法能够以较小的能源和环境成本实现与大型语言模型相竞争的结果，这对于大规模文本分类任务更加可持续。

### 翻译

大型语言模型已成为有效的零样本分类器，但其高计算需求和环境成本限制了它们在高性能计算环境中大规模标注的实用性。为了支持更可持续的工作流程，我们提出了Text2Graph，这是一个开源的Python包，提供了现有文本到图分类方法的模块化实现。该框架使用户能够以灵活的方式将基于大型语言模型的部分标注与图神经网络标签传播相结合，使用户可以轻松交换组件，如特征提取器、边构建方法和采样策略。我们在五个涵盖主题分类和情感分析任务的零样本设置上对Text2Graph进行了基准测试，将多个变体与其他文本分类的零样本方法进行了比较。除了报告性能外，我们还提供了能源消耗和碳排放的详细估计，表明基于图的传播方法能够在能源和环境成本较低的情况下实现具有竞争力的结果。


### 论文摘要

Large Language Models (LLMs) have become effective zero-shot classifiers, but their high computational requirements and environmental costs limit their practicality for large-scale annotation in high-performance computing (HPC) environments. To support more sustainable workflows, we present \textsc{Text2Graph}, an open-source Python package that provides a modular implementation of existing text-to-graph classification approaches. The framework enables users to combine LLM-based partial annotation with Graph Neural Network (GNN) label propagation in a flexible manner, making it straightforward to swap components such as feature extractors, edge construction methods, and sampling strategies. We benchmark \textsc{Text2Graph} on a zero-shot setting using five datasets spanning topic classification and sentiment analysis tasks, comparing multiple variants against other zero-shot approaches for text classification. In addition to reporting performance, we provide detailed estimates of energy consumption and carbon emissions, showing that graph-based propagation achieves competitive results at a fraction of the energy and environmental cost.

---

## 25. Adaptive Dual-Weighted Gravitational Point Cloud Denoising Method

**论文链接:** [http://arxiv.org/abs/2512.10386v1](http://arxiv.org/abs/2512.10386v1)

**作者:** Ge Zhang, Chunyang Wang, Bo Xiao, Xuelian Liu, Bin Liu

**发布时间:** 2025-12-11

### GPT解析

### 总结

本文提出了一种自适应双权重引力场的点云去噪方法，能够在保持高去噪精度的同时实现实时处理，有效保留物体边界和精细结构细节。

### 背景

高质量的点云数据对自动驾驶和三维重建等任务至关重要，但基于LiDAR的点云采集常受到各种干扰产生噪声点，降低后续检测识别准确性。现有方法难以同时实现高去噪精度、强边缘保持和实时性能。

### 目的

开发一种能够同时实现高去噪精度、强边缘保持和实时性能的点云去噪方法，解决现有方法的局限性。

### 方法

首先使用八叉树对全局点云进行空间分区实现并行加速；然后在每个叶节点中应用自适应体素占用统计和k近邻密度估计快速移除低密度噪声点；最后构建结合密度权重和自适应距离权重的引力评分函数精细区分噪声点和物体点。

### 主要发现

在斯坦福三维扫描库、CADC数据集和实验室FMCW LiDAR点云上的实验表明，该方法在各种噪声条件下F1、PSNR和Chamfer距离指标均实现一致改进，同时减少单帧处理时间，验证了其高准确性、鲁棒性和实时性能。

### 结论

该成功解决了现有方法在去噪精度、边缘保持和实时性之间的权衡问题，为自动驾驶和三维重建等应用提供了高质量的点云数据基础。

### 翻译

高质量的点云数据是自动驾驶和三维重建等任务的关键基础。然而，基于LiDAR的点云采集常受到各种干扰，导致大量噪声点产生，降低了后续点云目标检测和识别的准确性。此外，现有的点云去噪方法通常在追求更高去噪精度的同时牺牲计算效率，或者相反地，在提高处理速度的同时牺牲了对物体边界和精细结构细节的保留，难以同时实现高去噪精度、强边缘保持和实时性能。为解决这些局限性，本文提出了一种自适应双权重引力场的点云去噪方法。首先，采用八叉树对全局点云进行空间分区，实现并行加速。然后在每个叶节点中，应用自适应体素占用统计和k近邻密度估计，快速移除明显隔离的低密度噪声点，从而减少有效候选集。最后，构建结合密度权重和自适应距离权重的引力评分函数，精细区分噪声点和物体点。在斯坦福三维扫描库、加拿大不良驾驶条件数据集以及实验室获取的FMCW LiDAR点云上进行的实验表明，与现有方法相比，所提出的方法在各种噪声条件下均能在F1、PSNR和Chamfer距离指标上实现一致改进，同时减少了单帧处理时间，从而验证了其在多噪声场景下的高准确性、鲁棒性和实时性能。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决点云去噪问题，特别是LiDAR获取的点云数据中存在的噪声干扰问题。这个问题在现实中非常重要，因为高质量点云数据是自动驾驶和3D重建等任务的关键基础，噪声点会严重影响后续目标检测和识别的准确性。现有方法通常难以在去噪精度、边缘保留和实时性能之间取得平衡。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先分析了现有点云去噪方法，包括基于学习的方法和非学习方法。基于学习的方法需要大量训练数据且计算效率低，非学习方法则面临数据丢失、边缘模糊和实时性能差等挑战。作者借鉴了基于引力特征函数的去噪方法，但发现其计算复杂度高、效率不足。因此，作者设计了结合八叉树空间划分、自适应体素统计和kNN密度估计的多阶段处理方法，以减少计算量并提高效率。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是通过多阶段处理策略先去除明显噪声点，再对剩余点进行精细处理，实现高效准确的去噪。整体流程包括：1)使用八叉树对点云进行空间划分实现并行处理；2)在子节点中应用自适应体素占用统计去除异常点；3)使用kNN密度估计去除低密度噪声点；4)构建双权重引力评分函数精细区分噪声点和物体点；5)合并各子节点结果生成最终去噪点云。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)提出自适应双权重引力去噪方法；2)设计平衡准确性和效率的噪声点去除机制；3)引入密度-距离双权重引力评分函数。相比之前的工作，该方法通过八叉树并行处理提高效率，采用多阶段策略减少计算量，双权重评分函数更好地保留边缘特征，在多种噪声条件下表现出更好的鲁棒性和实时性能。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '本文提出了一种自适应双权重引力点云去噪方法，通过八叉树并行处理和多阶段噪声筛选策略，在保持高去噪精度的同时显著提高了处理效率，实现了去噪精度、边缘保留和实时性能的良好平衡。'}


### 论文摘要

High-quality point cloud data is a critical foundation for tasks such as autonomous driving and 3D reconstruction. However, LiDAR-based point cloud acquisition is often affected by various disturbances, resulting in a large number of noise points that degrade the accuracy of subsequent point cloud object detection and recognition. Moreover, existing point cloud denoising methods typically sacrifice computational efficiency in pursuit of higher denoising accuracy, or, conversely, improve processing speed at the expense of preserving object boundaries and fine structural details, making it difficult to simultaneously achieve high denoising accuracy, strong edge preservation, and real-time performance. To address these limitations, this paper proposes an adaptive dual-weight gravitational-based point cloud denoising method. First, an octree is employed to perform spatial partitioning of the global point cloud, enabling parallel acceleration. Then, within each leaf node, adaptive voxel-based occupancy statistics and k-nearest neighbor (kNN) density estimation are applied to rapidly remove clearly isolated and low-density noise points, thereby reducing the effective candidate set. Finally, a gravitational scoring function that combines density weights with adaptive distance weights is constructed to finely distinguish noise points from object points. Experiments conducted on the Stanford 3D Scanning Repository, the Canadian Adverse Driving Conditions (CADC) dataset, and in-house FMCW LiDAR point clouds acquired in our laboratory demonstrate that, compared with existing methods, the proposed approach achieves consistent improvements in F1, PSNR, and Chamfer Distance (CD) across various noise conditions while reducing the single-frame processing time, thereby validating its high accuracy, robustness, and real-time performance in multi-noise scenarios.

---

## 26. Guided Transfer Learning for Discrete Diffusion Models

**论文链接:** [http://arxiv.org/abs/2512.10877v1](http://arxiv.org/abs/2512.10877v1)

**作者:** Julian Kleutgens, Claudio Battiloro, Lingkai Kong, Benjamin Grewe, Francesca Dominici, Mauricio Tec

**发布时间:** 2025-12-11

**备注:** 7 pages (main text) + appendix

### GPT解析

### 总结

该研究提出了针对离散扩散模型的引导迁移学习方法(GTL)，无需修改预训练去噪器即可从目标分布采样，并提供了高效采样器以解决计算效率问题，使大规模词汇表和长序列的语言建模变得实用。

### 背景

离散扩散模型在语言和其他离散领域表现优异，但其强性能依赖于大型训练数据集，获取成本高或有风险。迁移学习是适应预训练离散扩散模型的自然方式，但当前方法需要微调大型扩散模型，计算成本高且不实用。

### 目的

开发一种无需修改预训练去噪器即可从目标分布采样的方法，并为离散时间扩散和连续时间基于得分的离散扩散提供统一的处理方式。

### 方法

提出基于比率的迁移学习方法(GTL)用于离散扩散模型，以及一种高效引导采样器，将评估集中在规划选择的位置和顶级候选令牌上，从而降低采样时间和计算量。

### 主要发现

引导离散扩散通常需要引导网络的多次前向传递，对于大型词汇表和长序列变得不切实际；而所提出的高效引导采样器使大规模词汇表和长序列的语言建模变得实用。

### 结论

GTL在序列数据（包括合成马尔可夫链和语言建模）上表现出色，研究提供了对其行为的经验分析。

### 翻译

离散扩散模型在语言和其他离散领域实现了强大性能，为自回归模型提供了有力的替代方案。然而，它们的强大性能依赖于大型训练数据集，这些数据集获取成本高昂或有风险，特别是在适应新领域时。迁移学习是适应预训练离散扩散模型的自然方式，但当前方法需要微调大型扩散模型，计算成本高且通常不实用。基于连续扩散的基于比率的迁移学习，我们为离散扩散模型提供了引导迁移学习(GTL)。这使人们能够无需修改预训练去噪器即可从目标分布进行采样。相同的引导公式适用于离散时间扩散和连续时间基于得分的离散扩散，从而实现了统一处理。引导离散扩散通常需要引导网络的多次前向传递，对于大型词汇表和长序列变得不切实际。为解决这一问题，我们进一步提出了一种高效的引导采样器，将评估集中在规划选择的位置和顶级候选令牌上，从而降低采样时间和计算量。这使得大规模词汇表和长序列的语言建模在实践中变得可行。我们在序列数据上评估了GTL，包括合成马尔可夫链和语言建模，并提供了对其行为的经验分析。


### 论文摘要

Discrete diffusion models achieve strong performance across language and other discrete domains, providing a powerful alternative to autoregressive models. However, their strong performance relies on large training datasets, which are costly or risky to obtain, especially when adapting to new domains. Transfer learning is the natural way to adapt pretrained discrete diffusion models, but current methods require fine-tuning large diffusion models, which is computationally expensive and often impractical. Building on ratio-based transfer learning for continuous diffusion, we provide Guided Transfer Learning for discrete diffusion models (GTL). This enables sampling from a target distribution without modifying the pretrained denoiser. The same guidance formulation applies to both discrete-time diffusion and continuous-time score-based discrete diffusion, yielding a unified treatment. Guided discrete diffusion often requires many forward passes of the guidance network, which becomes impractical for large vocabularies and long sequences. To address this, we further present an efficient guided sampler that concentrates evaluations on planner-selected positions and top candidate tokens, thus lowering sampling time and computation. This makes guided language modeling practical at scale for large vocabularies and long sequences. We evaluate GTL on sequential data, including synthetic Markov chains and language modeling, and provide empirical analyses of its behavior.

---

## 27. Robust Multi-Disease Retinal Classification via Xception-Based Transfer Learning and W-Net Vessel Segmentation

**论文链接:** [http://arxiv.org/abs/2512.10608v1](http://arxiv.org/abs/2512.10608v1)

**作者:** Mohammad Sadegh Gholizadeh, Amir Arsalan Rezapour

**发布时间:** 2025-12-11

### GPT解析

### 总结

本研究提出了一种结合深度特征提取与可解释图像处理模块的深度学习架构，用于眼部疾病的自动化诊断，特别关注视网膜血管分割作为辅助任务来指导分类过程。

### 背景

近年来，威胁视力的眼病发病率急剧上升，需要可扩展且准确的筛查解决方案。

### 目的

减轻标准卷积神经网络的'黑盒'局限性，弥合算法输出与专家医学验证之间的差距，减少假阳性，提高在临床环境中的部署可行性。

### 方法

实现一个结合深度特征提取与可解释图像处理模块的流程，将高保真视网膜血管分割作为辅助任务来指导分类过程，通过将模型预测基于临床相关的形态学特征。

### 主要发现

通过结合深度特征提取和可解释图像处理模块，可以减轻标准CNN的'黑盒'问题，提高模型在临床环境中的实用性。

### 结论

通过将模型预测基于临床相关的形态学特征，可以弥合算法输出与专家医学验证之间的差距，减少假阳性，提高在临床环境中的部署可行性。

### 翻译

近年来，威胁视力的眼病发病率急剧上升，需要可扩展且准确的筛查解决方案。本文提出了一个关于眼部疾病自动化诊断的深度学习架构的全面研究。为了减轻标准卷积神经网络的'黑盒'局限性，我们实现了一个结合深度特征提取与可解释图像处理模块的流程。特别是，我们关注高保真视网膜血管分割作为辅助任务来指导分类过程。通过将模型预测基于临床相关的形态学特征，我们旨在弥合算法输出与专家医学验证之间的差距，从而减少假阳性并提高在临床环境中的部署可行性。


### 论文摘要

In recent years, the incidence of vision-threatening eye diseases has risen dramatically, necessitating scalable and accurate screening solutions. This paper presents a comprehensive study on deep learning architectures for the automated diagnosis of ocular conditions. To mitigate the "black-box" limitations of standard convolutional neural networks (CNNs), we implement a pipeline that combines deep feature extraction with interpretable image processing modules. Specifically, we focus on high-fidelity retinal vessel segmentation as an auxiliary task to guide the classification process. By grounding the model's predictions in clinically relevant morphological features, we aim to bridge the gap between algorithmic output and expert medical validation, thereby reducing false positives and improving deployment viability in clinical settings.

---

## 28. R^2-HGP: A Double-Regularized Gaussian Process for Heterogeneous Transfer Learning

**论文链接:** [http://arxiv.org/abs/2512.10258v1](http://arxiv.org/abs/2512.10258v1)

**作者:** Duo Wang, Xinming Wang, Chao Wang, Xiaowei Yue, Jianguo Wu

**发布时间:** 2025-12-11

**备注:** 17 pages, 9 figures. Under review for IEEE TPAMI

### GPT解析

### 总结

本文提出了一种双正则化异构高斯过程框架(R²-HGP)，用于解决多源迁移学习中的异构域知识转移问题。该方法通过可训练的先验概率映射模型对齐异构输入域，结合物理知识作为正则化项，并使用稀疏惩罚来抑制负迁移，有效提升了多输出高斯过程在迁移学习中的性能。

### 背景

多输出高斯过程(MGP)模型因其灵活性和不确定性量化能力而受到广泛关注，并被广泛应用于多源迁移学习场景。然而，在迁移学习中仍面临几个挑战：1)源域和目标域的输入空间通常是异构的，使得直接知识转移困难；2)在异构迁移过程中通常忽略先验知识和物理信息，阻碍了领域特定见解的利用并导致映射不稳定；3)目标域与源域之间的不适当信息共享容易导致负迁移。传统模型无法以统一方式解决这些问题。

### 目的

为了克服这些局限性，本文旨在提出一种能够统一处理异构域迁移学习挑战的方法，特别是解决输入空间异构性问题、整合先验知识和物理信息、以及防止负迁移的问题。

### 方法

论文提出了双正则化异构高斯过程框架(R²-HGP)，主要包括：1)提出可训练的先验概率映射模型对齐异构输入域；2)将对齐后的输入作为潜变量，构建多源迁移GP模型；3)将整个结构整合到基于新型条件变分自编码器(CVAE)的框架中；4)将物理见解作为正则化项纳入，确保对齐结果符合已知的物理知识；5)在多源迁移GP模型中对转移系数施加稀疏惩罚，使模型能够自适应选择信息量最大的源输出并抑制负迁移。

### 主要发现

通过大量仿真和真实工程案例研究验证了R²-HGP的有效性，表明其在各种评估指标上均展现出比最先进的基准方法一致的优势。

### 结论

R²-HGP框架成功解决了多源迁移学习中异构域知识转移的多个关键挑战，通过统一的方法处理输入空间异构性、整合物理知识以及防止负迁移，为多输出高斯过程在迁移学习中的应用提供了新的解决方案。

### 翻译

多输出高斯过程(MGP)模型因其灵活性和不确定性量化能力而受到广泛关注，并由于其能够捕捉任务间相关性而被广泛应用于多源迁移学习场景。然而，它们在迁移学习中仍面临几个挑战。首先，源域和目标域的输入空间通常是异构的，这使得直接知识转移变得困难。其次，在异构迁移过程中通常忽略先验知识和物理信息，阻碍了领域特定见解的利用并导致映射不稳定。第三，目标域与源域之间的不适当信息共享很容易导致负迁移。传统模型无法以统一方式解决这些问题。为了克服这些局限性，本文提出了一种双正则化异构高斯过程框架(R²-HGP)。具体而言，首先提出了一种可训练的先验概率映射模型来对齐异构输入域。将对齐后的输入视为潜变量，在其上构建多源迁移GP模型，并将整个结构整合到基于新型条件变分自编码器(CVAE)的框架中。进一步将物理见解作为正则化项纳入，以确保对齐结果符合已知的物理知识。接下来，在多源迁移GP模型中，对转移系数施加稀疏惩罚，使模型能够自适应选择信息量最大的源输出并抑制负迁移。大量仿真和真实工程案例研究验证了我们R²-HGP的有效性，表明其在各种评估指标上均展现出比最先进的基准方法一致的优势。


### 论文摘要

Multi-output Gaussian process (MGP) models have attracted significant attention for their flexibility and uncertainty-quantification capabilities, and have been widely adopted in multi-source transfer learning scenarios due to their ability to capture inter-task correlations. However, they still face several challenges in transfer learning. First, the input spaces of the source and target domains are often heterogeneous, which makes direct knowledge transfer difficult. Second, potential prior knowledge and physical information are typically ignored during heterogeneous transfer, hampering the utilization of domain-specific insights and leading to unstable mappings. Third, inappropriate information sharing among target and sources can easily lead to negative transfer. Traditional models fail to address these issues in a unified way. To overcome these limitations, this paper proposes a Double-Regularized Heterogeneous Gaussian Process framework (R^2-HGP). Specifically, a trainable prior probability mapping model is first proposed to align the heterogeneous input domains. The resulting aligned inputs are treated as latent variables, upon which a multi-source transfer GP model is constructed and the entire structure is integrated into a novel conditional variational autoencoder (CVAE) based framework. Physical insights is further incorporated as a regularization term to ensure that the alignment results adhere to known physical knowledge. Next, within the multi-source transfer GP model, a sparsity penalty is imposed on the transfer coefficients, enabling the model to adaptively select the most informative source outputs and suppress negative transfer. Extensive simulations and real-world engineering case studies validate the effectiveness of our R^2-HGP, demonstrating consistent superiority over state-of-the-art benchmarks across diverse evaluation metrics.

---

## 29. RaLiFlow: Scene Flow Estimation with 4D Radar and LiDAR Point Clouds

**论文链接:** [http://arxiv.org/abs/2512.10376v1](http://arxiv.org/abs/2512.10376v1)

**作者:** Jingyun Fu, Zhiyu Xiang, Na Zhao

**发布时间:** 2025-12-11

**备注:** Accepted by AAAI

### GPT解析

### 总结

该研究填补了4D毫米波雷达与LiDAR融合在场景流估计领域的空白，构建了专门的雷达-LiDAR场景流数据集，并提出了RaLiFlow框架，包含动态感知双向跨模态融合模块和特殊设计的损失函数，实验证明该方法显著优于现有单模态方法。

### 背景

多模态融合方法（图像与LiDAR点云）在场景流估计中显示出前景，但4D毫米波雷达与LiDAR的融合尚未探索。雷达相比LiDAR更便宜、在各种天气条件下更稳健且可检测点级速度，但存在噪声、低分辨率和稀疏性等挑战。目前没有专门针对场景流估计的LiDAR和雷达数据集。

### 目的

构建一个基于公共真实世界汽车数据集的雷达-LiDAR场景流数据集；提出有效的雷达去噪和场景流标签生成预处理策略；开发第一个4D雷达和LiDAR联合场景流学习框架。

### 方法

提出动态感知双向跨模态融合（DBCF）模块，将雷达的动态线索整合到局部跨注意力机制中；设计一套损失函数，减轻训练期间不可靠雷达数据的不利影响，并增强来自两种模态的场景流预测的实例级一致性，特别是在动态前景区域。

### 主要发现

在重新利用的场景流数据集上的广泛实验表明，该方法显著优于现有的基于LiDAR和基于雷达的单模态方法。

### 结论

RaLiFlow是第一个4D雷达和LiDAR联合场景流学习框架，通过DBCF模块和精心设计的损失函数实现了有效的雷达-LiDAR融合，在动态前景区域表现特别出色。

### 翻译

最近的多模态融合方法，结合图像与LiDAR点云，在场景流估计中显示出前景。然而，4D毫米波雷达与LiDAR的融合仍未被探索。与LiDAR不同，雷达更便宜，在各种天气条件下更稳健，并能检测点级速度，使其成为LiDAR的有价值补充。然而，由于噪声、低分辨率和稀疏性，雷达输入带来了挑战。此外，目前还没有专门针对场景流估计的LiDAR和雷达数据集组合。为解决这一空白，我们基于公共真实世界汽车数据集构建了一个雷达-LiDAR场景流数据集。我们提出了一个有效的雷达去噪和场景流标签生成预处理策略，从物体边界外推导出更可靠的雷达点流动真实值。此外，我们引入了RaLiFlow，这是第一个4D雷达和LiDAR的联合场景流学习框架，它通过新颖的动态感知双向跨模态融合（DBCF）模块和一套精心设计的损失函数实现了有效的雷达-LiDAR融合。DBCF模块将雷达的动态线索整合到局部跨模态注意力机制中，使上下文信息能够跨模态传播。同时，所提出的损失函数减轻了训练期间不可靠雷达数据的不利影响，并增强了来自两种模态的场景流预测的实例级一致性，特别是对于动态前景区域。在重新利用的场景流数据集上的广泛实验表明，我们的方法显著优于现有的基于LiDAR和基于雷达的单模态方法。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决4D毫米波雷达和LiDAR点云融合用于场景流估计的挑战。目前多模态融合方法主要关注图像与LiDAR融合，而雷达-LiDAR融合尚未被探索。这个问题很重要，因为场景流估计对自动驾驶中的场景理解、3D检测、目标跟踪等任务至关重要，而雷达作为LiDAR的补充具有成本低、天气鲁棒性好、能直接测量速度等优势，融合两者可提供更全面的环境感知能力。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先发现缺乏专门用于场景流估计的雷达-LiDAR数据集，因此基于VoD数据集构建了新数据集，并提出雷达去噪和可靠标签生成的预处理策略。在网络设计上，作者借鉴了现有的雷达-LiDAR融合方法，但指出这些方法主要针对3D目标检测而非场景流估计；同时借鉴了基于支柱和体素的场景流估计方法，但进行了改进以适应雷达-LiDAR融合。作者创新性地设计了动态感知双向跨模态融合模块和专门损失函数，以解决雷达数据噪声大、稀疏性和不可靠标签等问题。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是利用雷达和LiDAR的优势互补，通过动态感知融合机制重点关注场景中的动态区域，并设计可靠的数据处理和损失函数来提升性能。整体流程包括：1)数据预处理(地面点移除、雷达去噪、场景流标签生成)；2)网络架构(基于支柱的特征提取、DBCF融合模块、U-Net流嵌入、GRU流头)；3)损失函数设计(LiDAR流损失、掩码雷达流损失、实例级动态流一致性损失)。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)首个专门用于场景流估计的雷达-LiDAR数据集及预处理策略；2)首个雷达-LiDAR场景流融合框架RaLiFlow；3)动态感知双向跨模态融合(DBCF)模块，将雷达动态信息集成到注意力机制；4)精心设计的损失函数处理不可靠雷达数据。相比之前工作，RaLiFlow不是单模态方法，而是融合两者优势；不同于现有雷达-LiDAR融合方法主要针对3D检测，RaLiFlow专注于点级动态细节；解决了现有方法中雷达场景流标签不可靠的问题，因为许多动态雷达点落在边界框外。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'RaLiFlow首次实现了4D雷达和LiDAR点云的有效融合，通过创新的动态感知跨模态融合模块和精心设计的损失函数，显著提升了场景流估计的性能，同时构建了首个专门用于这一任务的雷达-LiDAR场景流数据集。'}


### 论文摘要

Recent multimodal fusion methods, integrating images with LiDAR point clouds, have shown promise in scene flow estimation. However, the fusion of 4D millimeter wave radar and LiDAR remains unexplored. Unlike LiDAR, radar is cheaper, more robust in various weather conditions and can detect point-wise velocity, making it a valuable complement to LiDAR. However, radar inputs pose challenges due to noise, low resolution, and sparsity. Moreover, there is currently no dataset that combines LiDAR and radar data specifically for scene flow estimation. To address this gap, we construct a Radar-LiDAR scene flow dataset based on a public real-world automotive dataset. We propose an effective preprocessing strategy for radar denoising and scene flow label generation, deriving more reliable flow ground truth for radar points out of the object boundaries. Additionally, we introduce RaLiFlow, the first joint scene flow learning framework for 4D radar and LiDAR, which achieves effective radar-LiDAR fusion through a novel Dynamic-aware Bidirectional Cross-modal Fusion (DBCF) module and a carefully designed set of loss functions. The DBCF module integrates dynamic cues from radar into the local cross-attention mechanism, enabling the propagation of contextual information across modalities. Meanwhile, the proposed loss functions mitigate the adverse effects of unreliable radar data during training and enhance the instance-level consistency in scene flow predictions from both modalities, particularly for dynamic foreground areas. Extensive experiments on the repurposed scene flow dataset demonstrate that our method outperforms existing LiDAR-based and radar-based single-modal methods by a significant margin.

---

## 30. Point2Pose: A Generative Framework for 3D Human Pose Estimation with Multi-View Point Cloud Dataset

**论文链接:** [http://arxiv.org/abs/2512.10321v1](http://arxiv.org/abs/2512.10321v1)

**作者:** Hyunsoo Lee, Daeum Jeon, Hyeokjae Oh

**发布时间:** 2025-12-11

**备注:** WACV 2026 camera ready

### GPT解析

### 总结

论文提出了一种名为Point2Pose的新型生成方法用于3D人体姿态估计，并构建了一个包含多种模态的大规模室内数据集MVPose3D。该方法通过时空点云编码器和姿态特征编码器提取特征，并使用基于注意力的生成回归器，在多个数据集上表现出色。

### 背景

3D人体姿态估计面临几个关键挑战，包括人体复杂的几何结构、自遮挡关节以及对大规模真实世界运动数据集的需求。

### 目的

解决3D人体姿态估计中的挑战，提出一种能够有效建模基于序列点云和姿态历史的人体姿态分布的框架。

### 方法

1. 提出Point2Pose框架，能够根据序列点云和姿态历史有效建模人体姿态分布；2. 使用时空点云编码器和姿态特征编码器提取关节级特征；3. 采用基于注意力的生成回归器；4. 构建大规模室内数据集MVPose3D，包含IMU数据、密集多视角点云和RGB图像等多种模态。

### 主要发现

实验结果表明，所提出的方法优于基线模型，在各种数据集上展现出优越的性能。

### 结论

Point2Pose框架是一种有效的3D人体姿态估计方法，通过结合时空点云编码器和基于注意力的生成回归器，能够处理复杂的人体姿态估计挑战。

### 翻译

我们提出了一种用于3D人体姿态估计的新型生成方法。由于人体几何结构的复杂性、自遮挡关节以及对大规模真实世界运动数据集的需求，3D人体姿态估计存在几个关键挑战。为解决这些挑战，我们引入了Point2Pose，一个能够有效建模基于序列点云和姿态历史的人体姿态分布的框架。具体而言，我们采用时空点云编码器和姿态特征编码器提取关节级特征，然后使用基于注意力的生成回归器。此外，我们提出了一个大规模室内数据集MVPose3D，它包含多种模态，包括非平凡人体运动的IMU数据、密集多视角点云和RGB图像。实验结果表明，所提出的方法优于基线模型，证明了其在各种数据集上的优越性能。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决3D人体姿态估计中的关键挑战：人体几何结构的复杂性、关节自遮挡问题以及大规模真实世界运动数据集的需求。这个问题在现实中非常重要，因为3D人体姿态估计广泛应用于增强现实、医疗系统、工业和自动驾驶等领域，能够使人机交互更加自然和精确。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先分析了现有方法的局限性：2D图像方法缺乏完整3D空间信息，点云方法难以捕捉精细细节和自遮挡关节。然后，他们发现生成式建模（特别是扩散模型）在复杂3D数据建模方面的潜力，提出直接在3D点云上应用条件扩散和最优传输条件流匹配，而不是依赖2D到3D的提升过程。作者借鉴了Point-BERT的点云特征提取、ViViT的时空Transformer架构、扩散模型和ST-GCN等现有工作，并将它们创新地结合在一起。', '这个方法的核心思想是什么？整体实现流程是怎样的？': 'Point2Pose的核心思想是构建一个生成式框架，直接从原始点云估计3D人体姿态，无需中间2D表示。整体流程包括：1) 时空点云编码器提取局部和全局特征并通过时空注意力融合；2) 关节级别姿态-点特征编码器使用ST-GCN编码姿态历史并通过交叉注意力融合点云特征；3) 因子化生成姿态回归器使用扩散模型或最优传输条件流匹配分解预测关节坐标和旋转。训练时最小化预测姿态与真实姿态的平滑L1损失，推理时从噪声开始迭代生成最终姿态。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1) 首个直接从原始点云估计3D人体姿态的生成式框架，无需2D表示；2) 关节级别的姿态-点特征编码器，通过时空注意力建模关节-点交互；3) 因子化生成姿态回归器，先预测坐标再预测旋转提高运动一致性；4) 大型多模态数据集MVPose3D，提供密集多视角点云和复杂运动场景。相比之前工作，Point2Pose避免了2D到3D的不适定问题，直接在点云域操作，并通过时空注意力捕获更精细的几何特征，在自遮挡场景下表现更好。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'Point2Pose提出了一种创新的生成式框架，直接从多视角点云数据中估计3D人体姿态，并构建了大规模多模态数据集MVPose3D，显著提升了在复杂几何结构和自遮挡场景下的姿态估计性能。'}


### 论文摘要

We propose a novel generative approach for 3D human pose estimation. 3D human pose estimation poses several key challenges due to the complex geometry of the human body, self-occluding joints, and the requirement for large-scale real-world motion datasets. To address these challenges, we introduce Point2Pose, a framework that effectively models the distribution of human poses conditioned on sequential point cloud and pose history. Specifically, we employ a spatio-temporal point cloud encoder and a pose feature encoder to extract joint-wise features, followed by an attention-based generative regressor. Additionally, we present a large-scale indoor dataset MVPose3D, which contains multiple modalities, including IMU data of non-trivial human motions, dense multi-view point clouds, and RGB images. Experimental results show that the proposed method outperforms the baseline models, demonstrating its superior performance across various datasets.

---

## 31. THE-Pose: Topological Prior with Hybrid Graph Fusion for Estimating Category-Level 6D Object Pose

**论文链接:** [http://arxiv.org/abs/2512.10251v1](http://arxiv.org/abs/2512.10251v1)

**作者:** Eunho Lee, Chaehyeon Song, Seunghoon Jeong, Ayoung Kim

**发布时间:** 2025-12-11

### GPT解析

### 总结

本文提出了一种名为THE-Pose的新型类别级6D姿态估计框架，通过表面嵌入和混合图融合利用拓扑先验，解决了现有3D-GC方法的局限性，显著提高了姿态估计的准确性和鲁棒性。

### 背景

类别级物体姿态估计需要同时考虑全局上下文和局部结构，以确保对类内变化的鲁棒性。然而，现有的3D图卷积方法仅关注局部几何和深度信息，使其在面对复杂物体和视觉模糊性时表现脆弱。

### 目的

开发一种能够有效结合全局上下文和局部结构的姿态估计方法，提高对复杂物体和遮挡情况下的鲁棒性，并超越现有方法的性能。

### 方法

1) 从图像域提取一致且不变的拓扑特征，克服现有3D-GC方法的固有局限；2) 设计混合图融合(HGF)模块，自适应地将拓扑特征与点云特征集成；3) 无缝连接2D图像上下文和3D几何结构；4) 融合的特征确保对未见或复杂物体的稳定性，即使在严重遮挡情况下。

### 主要发现

在REAL275数据集上的大量实验表明，THE-Pose比3D-GC基线(HS-Pose)提高了35.8%，并且在所有关键指标上超过了之前的最佳方法7.2%。

### 结论

THE-Pose通过利用拓扑先验和混合图融合，有效解决了类别级物体姿态估计中的挑战，特别是在处理复杂物体和遮挡情况时表现出色，代码已开源。

### 翻译

类别级物体姿态估计需要同时考虑全局上下文和局部结构，以确保对类内变化的鲁棒性。然而，3D图卷积方法仅关注局部几何和深度信息，使其容易受到复杂物体和视觉模糊性的影响。为解决这一问题，我们提出了THE-Pose，一种新颖的类别级6D姿态估计框架，通过表面嵌入和混合图融合利用拓扑先验。具体而言，我们从图像域提取一致且不变的拓扑特征，有效克服了现有基于3D-GC方法的固有局限性。我们的混合图融合(HGF)模块自适应地将拓扑特征与点云特征集成，无缝连接2D图像上下文和3D几何结构。这些融合的特征确保了对未见或复杂物体的稳定性，即使在严重遮挡的情况下也是如此。在REAL275数据集上的大量实验表明，THE-Pose比3D-GC基线(HS-Pose)提高了35.8%，并且在所有关键指标上超过了之前的最佳方法7.2%。代码可在https://github.com/EHxxx/THE-Pose获取。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决类别级别6D物体姿态估计中的类内变化问题。这个问题在现实中很重要，因为它关系到机器人操作、增强现实和场景理解等应用领域，能够使系统在没有每个物体单独3D模型的情况下，准确识别和定位同一类别中不同形状、颜色的物体，大大降低了应用门槛并提高了系统的实用性。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先分析了现有方法的局限性：基于形状先验的方法在显著形状变化下会产生错误对应；基于语义先验的方法包含不必要的视觉信息；3D图卷积方法只关注局部结构，缺乏全局上下文。受此启发，作者设计了一种拓扑先验来弥补3D-GC的不足。他们借鉴了表面嵌入技术用于学习2D-3D对应关系，结合了3D图卷积处理点云数据，并参考了HS-Pose的混合感受野层和STViT捕获全局上下文的方法，最终提出了包含拓扑全局上下文聚合器(TGC)、混合图融合模块(HGF)和姿态大小估计器的完整框架。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是引入拓扑先验来提供物体的全局几何和拓扑结构，同时与3D图卷积的局部几何信息相结合，通过混合图融合模块自适应地融合这两种特征，实现对类别级别6D物体姿态的准确估计。整体流程包括：1)输入处理(分割物体并提取RGB图像和点云)；2)通过TGC模块提取拓扑先验特征；3)使用3D图卷积处理点云；4)通过HGF模块融合拓扑特征和点云特征；5)最后通过姿态大小估计器回归出旋转、平移和大小参数。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)提出拓扑先验，专注于全局几何和拓扑结构，具有几何一致性、拓扑一致性和SE(3)不变性；2)设计混合图融合模块(HGF)，通过混合感受野结合点级和特征级距离，自适应融合拓扑和点云特征；3)有效结合全局上下文和局部结构，提高对复杂形状和遮挡的鲁棒性。相比之前的工作，THE-Pose避免了形状先验在显著变化下的错误对应、语义先验的不必要视觉信息，以及3D-GC方法缺乏全局上下文的局限性，在保持局部几何敏感性的同时增强了全局感知能力。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'THE-Pose通过引入拓扑先验和混合图融合机制，有效解决了类别级别6D物体姿态估计中的类内变化问题，显著提高了对复杂形状、遮挡和视觉歧义的鲁棒性，并在多个数据集上实现了超越现有方法的性能。'}


### 论文摘要

Category-level object pose estimation requires both global context and local structure to ensure robustness against intra-class variations. However, 3D graph convolution (3D-GC) methods only focus on local geometry and depth information, making them vulnerable to complex objects and visual ambiguities. To address this, we present THE-Pose, a novel category-level 6D pose estimation framework that leverages a topological prior via surface embedding and hybrid graph fusion. Specifically, we extract consistent and invariant topological features from the image domain, effectively overcoming the limitations inherent in existing 3D-GC based methods. Our Hybrid Graph Fusion (HGF) module adaptively integrates the topological features with point-cloud features, seamlessly bridging 2D image context and 3D geometric structure. These fused features ensure stability for unseen or complicated objects, even under significant occlusions. Extensive experiments on the REAL275 dataset show that THE-Pose achieves a 35.8% improvement over the 3D-GC baseline (HS-Pose) and surpasses the previous state-of-the-art by 7.2% across all key metrics. The code is avaialbe on https://github.com/EHxxx/THE-Pose

---

## 32. MMSI-Video-Bench: A Holistic Benchmark for Video-Based Spatial Intelligence

**论文链接:** [http://arxiv.org/abs/2512.10863v1](http://arxiv.org/abs/2512.10863v1)

**作者:** Jingli Lin, Runsen Xu, Shaohao Zhu, Sihan Yang, Peizhou Cao, Yunlong Ran, Miao Hu, Chenming Zhu, Yiman Xie, Yilin Long, Wenbo Hu, Dahua Lin, Tai Wang, Jiangmiao Pang

**发布时间:** 2025-12-11

### GPT解析

### 总结

该研究引入了MMSI-Video-Bench，一个专门用于评估多模态大语言模型视频空间智能的全面基准测试，通过四级框架评估模型在物理环境中的空间理解能力。

### 背景

多模态大语言模型需要在物理环境中具备空间理解能力才能成为通用助手，但目前缺乏全面评估这一目标进展的基准测试。

### 目的

创建一个完全人工标注的基准测试，用于评估多模态大语言模型在视频空间智能方面的表现。

### 方法

构建了一个包含1106个问题的基准测试，基于1278个来自25个数据集和自制视频的片段，实现感知、规划、预测和跨视频推理四级框架；每个项目都经过3D专家精心设计和审查；支持三个领域导向的子基准测试。

### 主要发现

评估25个模型后发现显著人机差距，许多模型表现接近随机水平，最佳推理模型比人类落后60%；空间微调模型无法有效泛化；存在几何推理、运动定位、长时预测和跨视频对应的系统性失败；典型帧采样策略和3D空间线索、思维链提示效果不佳。

### 结论

该基准测试为推进视频空间智能研究提供了坚实的测试平台。

### 翻译

在连续视觉输入上的空间理解对多模态大语言模型发展为物理环境中的通用助手至关重要。然而，目前仍没有全面评估这一目标进展的综合基准。在这项工作中，我们引入了MMSI-Video-Bench，一个用于多模态大语言模型视频空间智能的完全人工标注基准。它通过来自25个数据集和自制视频的1278个片段中的1106个问题，实现了感知、规划、预测和跨视频推理的四级框架。每个项目都经过3D专家的精心设计和审查，并附有解释性理由，以确保精确、明确的定位。利用其多样化的数据来源和全面的任务覆盖，MMSI-Video-Bench还支持三个领域导向的子基准测试（室内场景感知基准、机器人基准和定位基准），用于有针对性的能力评估。我们评估了25个强大的开源和专有多模态大语言模型，揭示了显著的人机差距：许多模型表现接近随机水平，最好的推理模型比人类落后近60%。我们发现，经过空间微调的模型在我们的基准测试上仍然无法有效泛化。细粒度错误分析暴露了几何推理、运动定位、长时预测和跨视频对应方面的系统性失败。我们还表明，典型的帧采样策略在我们的推理密集型基准测试上转移效果不佳，且3D空间线索和思维链提示都没有带来有意义的提升。我们期望这个基准测试能够为推进视频空间智能建立一个坚实的测试平台。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决多模态大语言模型(MLLMs)在视频空间智能评估中缺乏全面基准测试的问题。这个问题很重要，因为空间理解能力对MLLMs发展为物理环境中的通用助手至关重要，而现有基准测试要么基于静态图像而非视频，要么问题类型不够全面，要么过度依赖模板化自动生成，无法全面评估模型在真实世界场景中的空间智能表现。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先识别出现有基准测试的局限性，特别是视频空间智能评估方面的不足。他们构建了一个四层框架(感知、规划、预测、跨视频推理)，并采用完全人工设计的协议，由11名3D视觉专家精心设计每个样本。他们结合了25个公开数据集和内部录制的视频确保多样性。作者借鉴了现有工作如MMSI-Bench的人工标注方法，但扩展到了视频领域，同时解决了其他基准测试在问题类型、数据来源和场景覆盖方面的不足。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是创建一个全面、多样化、人工标注的视频空间智能基准测试，评估模型对视频中的时空信息的感知、理解、推理和决策能力。整体流程包括：1)数据收集与预处理(约20k视频片段，来自25个公开数据集和140个内部视频)；2)人工标注(11名3D视觉专家参与)；3)质量控制(严格审核确保问题清晰、正确且具挑战性)；4)最终基准测试(1,106个问题，基于1,278个视频片段，涵盖5个主要类别和13个子类型)；5)评估与实验(评估25个开源和专有MLLMs，进行错误分析和模型改进探索)。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)全面性，首个全面评估视频空间智能的基准测试；2)数据多样性，结合25个公开数据集和内部视频；3)高质量人工标注，避免模板化问题生成；4)提供三个领域导向的子基准测试；5)揭示显著人机性能差距(最佳模型仍落后人类近60%)。相比之前工作，MMSI-Video-Bench专注于视频输入而非静态图像，提供更全面的问题类型覆盖，避免模板化限制，涵盖多样化真实世界场景，并通过错误分析提供具体改进方向。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'MMSI-Video-Bench通过一个全面、多样化、人工标注的视频空间智能基准测试，揭示了当前多模态大语言模型在视频空间理解方面与人类存在巨大差距，并为未来研究和改进提供了具体方向。'}


### 论文摘要

Spatial understanding over continuous visual input is crucial for MLLMs to evolve into general-purpose assistants in physical environments. Yet there is still no comprehensive benchmark that holistically assesses the progress toward this goal. In this work, we introduce MMSI-Video-Bench, a fully human-annotated benchmark for video-based spatial intelligence in MLLMs. It operationalizes a four-level framework, Perception, Planning, Prediction, and Cross-Video Reasoning, through 1,106 questions grounded in 1,278 clips from 25 datasets and in-house videos. Each item is carefully designed and reviewed by 3DV experts with explanatory rationales to ensure precise, unambiguous grounding. Leveraging its diverse data sources and holistic task coverage, MMSI-Video-Bench also supports three domain-oriented sub-benchmarks (Indoor Scene Perception Bench, Robot Bench and Grounding Bench) for targeted capability assessment. We evaluate 25 strong open-source and proprietary MLLMs, revealing a striking human--AI gap: many models perform near chance, and the best reasoning model lags humans by nearly 60%. We further find that spatially fine-tuned models still fail to generalize effectively on our benchmark. Fine-grained error analysis exposes systematic failures in geometric reasoning, motion grounding, long-horizon prediction, and cross-video correspondence. We also show that typical frame-sampling strategies transfer poorly to our reasoning-intensive benchmark, and that neither 3D spatial cues nor chain-of-thought prompting yields meaningful gains. We expect our benchmark to establish a solid testbed for advancing video-based spatial intelligence.

---

## 33. CLASH: Collaborative Large-Small Hierarchical Framework for Continuous Vision-and-Language Navigation

**论文链接:** [http://arxiv.org/abs/2512.10360v1](http://arxiv.org/abs/2512.10360v1)

**作者:** Liuyi Wang, Zongtao He, Jinlong Li, Xiaoyan Qi, Mengxian Hu, Chenpeng Yao, Chengju Liu, Qijun Chen

**发布时间:** 2025-12-11

### GPT解析

### 总结

本研究提出了一种名为CLASH的视觉语言导航框架，通过整合反应式小型模型规划器和反思式大型模型推理器，解决了大型视觉语言模型在VLN任务中表现不佳的问题，在仿真和真实世界实验中均取得了最先进的结果。

### 背景

视觉语言导航(VLN)任务要求机器人在没有预先地图的情况下遵循自然语言指令在复杂环境中导航。尽管最近的视觉语言大型模型展现出强大的推理能力，但在VLN任务中通常表现不如专门的小型模型。

### 目的

旨在解决大型视觉语言模型在VLN任务中表现不佳的问题，通过开发一种结合小型和大型模型优势的框架来提高VLN性能。

### 方法

提出了CLASH框架，整合了反应式小型模型规划器(RSMP)和反思式大型模型推理器(RLMR)。RSMP采用基于因果学习的双分支架构增强泛化能力，RLMR利用全景视觉提示和思维链推理支持可解释的空间理解和导航。引入了不确定性感知协作机制(UCM)自适应融合两个模型的决策。在障碍物避免方面，仿真中使用完全可学习的点目标策略，实际部署中设计了基于LiDAR的聚类模块生成可导航路径点。

### 主要发现

CLASH在VLN-CE排行榜上排名第一，在测试未见集上显著提高了成功率(SR)和成功路径长度(SPL)指标。真实世界实验证明了CLASH的强鲁棒性，验证了其在仿真和部署场景中的有效性。

### 结论

CLASH框架通过整合小型和大型模型的优势，有效解决了大型视觉语言模型在VLN任务中的局限性。该框架不仅取得了最先进的性能，还在真实世界环境中表现出强鲁棒性，为未来VLN研究提供了新的方向。

### 翻译

本研究提出了一种名为CLASH的视觉语言导航-连续评估框架，该框架整合了反应式小型模型规划器和反思式大型模型推理器，以解决大型视觉语言模型在视觉语言导航任务中表现不佳的问题。该框架在仿真和真实世界实验中均取得了最先进的结果，证明了其有效性和鲁棒性。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决视觉语言导航(VLN)任务中如何有效结合大型多模态语言模型的通用推理能力和小型任务特定模型的导航性能优势的问题。当前研究面临两大挑战：一是大型模型在通用推理上表现优异，但在具体VLN任务中往往不如小型专业模型；二是小型模型虽然导航性能好，但泛化能力有限，容易过拟合特定场景，且决策过程不够透明。这个问题在现实中非常重要，因为它关系到机器人能否在没有预先地图的情况下，根据自然语言指令在复杂环境中可靠导航，这对家庭服务机器人、自动驾驶、辅助导航等应用场景至关重要。在研究中，这个问题代表了如何将通用人工智能能力与特定任务专长有效结合的前沿挑战。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': "作者的思考基于对现有研究的深入观察，他们发现两个关键趋势：(1)使用全景输入的分层框架优于单目RGB的端到端方法；(2)在分层框架中，小型任务特定模型仍然优于大型语言模型。基于这些观察，作者提出了如何结合大型模型通用推理能力和小型模型导航专长的问题。他们设计了CLASH框架，采用'脑-体'架构，但创新性地让小型模型负责'体'(反应式规划)，大型模型负责'脑'(反思式推理)。在具体设计中，作者借鉴了多个现有工作：因果学习提高泛化能力，双分支架构增强信息融合，思维链推理提高可解释性，conformal prediction量化不确定性，以及DDPPO和SLAM等导航技术。", '这个方法的核心思想是什么？整体实现流程是怎样的？': "核心思想是构建一个'协作式大小分层框架'(CLASH)，将小型任务特定模型和大型多模态语言模型的优势互补结合。小型模型作为'反应式规划器'，专注于快速、可靠的导航决策；大型模型作为'反思式推理器'，提供常识推理和对复杂环境的更好适应性。两者通过'不确定感知协作机制'动态协作，根据任务难度和环境不确定性自适应地融合决策。整体流程分为三部分：1)高级决策模块：包含反应式小模型规划器、反思式大模型推理器和不确定感知协作机制；2)低级执行模块：包含路径点预测器和局部导航控制器，分别针对模拟和实际环境设计；3)整体工作流程：机器人接收指令→高级决策生成路径点→低级执行控制移动→重复直到到达目标。", '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)协作式大小分层架构：创新性地结合小型任务模型和大型语言模型，让不同类型模型各司其职；2)增强的高级推理：包括RSMP中的因果学习、RLMR中的全景视觉提示与思维链推理、以及不确定感知协作机制；3)实用的低级执行：为模拟和实际部署分别设计路径点预测和导航控制方法，解决了现实世界中的传感器限制。相比之前的工作，CLASH的不同之处在于：不是简单应用大型语言模型，而是有意识地设计协作框架；不仅关注模拟性能，还考虑模拟到现实的迁移；引入不确定性感知机制，能根据任务难度动态调整决策；提供更可解释的决策过程，使大型模型的决策更加透明。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'CLASH通过创新性地整合小型任务特定模型的导航专长和大型语言模型的通用推理能力，结合不确定感知协作机制和实用的低级执行方案，实现了在视觉语言导航任务中既保持高性能又具备强鲁棒性和可解释性的突破。'}


### 论文摘要

Vision-and-Language Navigation (VLN) requires robots to follow natural language instructions and navigate complex environments without prior maps. While recent vision-language large models demonstrate strong reasoning abilities, they often underperform task-specific panoramic small models in VLN tasks. To address this, we propose CLASH (Collaborative Large-Small Hierarchy), a VLN-CE framework that integrates a reactive small-model planner (RSMP) with a reflective large-model reasoner (RLMR). RSMP adopts a causal-learning-based dual-branch architecture to enhance generalization, while RLMR leverages panoramic visual prompting with chain-of-thought reasoning to support interpretable spatial understanding and navigation. We further introduce an uncertainty-aware collaboration mechanism (UCM) that adaptively fuses decisions from both models. For obstacle avoidance, in simulation, we replace the rule-based controller with a fully learnable point-goal policy, and in real-world deployment, we design a LiDAR-based clustering module for generating navigable waypoints and pair it with an online SLAM-based local controller. CLASH achieves state-of-the-art (SoTA) results (ranking 1-st) on the VLN-CE leaderboard, significantly improving SR and SPL on the test-unseen set over the previous SoTA methods. Real-world experiments demonstrate CLASH's strong robustness, validating its effectiveness in both simulation and deployment scenarios.

---

## 34. UnReflectAnything: RGB-Only Highlight Removal by Rendering Synthetic Specular Supervision

**论文链接:** [http://arxiv.org/abs/2512.09583v2](http://arxiv.org/abs/2512.09583v2)

**作者:** Alberto Rota, Mert Kiray, Mert Asim Karaoglu, Patrick Ruhkamp, Elena De Momi, Nassir Navab, Benjamin Busam

**发布时间:** 2025-12-10

### GPT解析

### 总结

本文提出了UnReflectAnything，一个仅使用RGB的框架，能够从单张图像中移除高光反射，适用于自然和外科手术图像。

### 背景

高光反射会扭曲外观、掩盖纹理并阻碍几何推理，这在自然图像和外科手术图像中都是常见问题，影响图像分析和几何推理。

### 目的

开发一种能够从单张图像中移除高光反射的框架，解决高光反射对图像外观和几何理解造成的干扰问题。

### 方法

使用冻结的视觉Transformer编码器提取多尺度特征，轻量级头部定位镜面区域，令牌级修复模块恢复损坏特征块；引入虚拟高光合成管道，利用单目几何、菲涅耳感知着色和随机光照进行训练。

### 主要发现

UnReflectAnything能够在自然和外科手术领域泛化，处理非朗伯表面和非均匀光照造成的高光问题，在多个基准测试中取得与最先进方法相当的性能。

### 结论

UnReflectAnything是一个有效的RGB-only框架，能够从单张图像中移除高光反射，适用于多种场景，展现出良好的泛化能力和性能。

### 翻译

镜面高光会扭曲外观、掩盖纹理，并在自然和外科手术图像中阻碍几何推理。我们提出了UnReflectAnything，一个仅使用RGB的框架，它通过预测高光图和无反射漫反射重建来从单张图像中移除高光。该模型使用冻结的视觉Transformer编码器提取多尺度特征，使用轻量级头部定位镜面区域，并使用令牌级修复模块在生成最终漫反射图像之前恢复损坏的特征块。为了克服缺乏成对监督的问题，我们引入了虚拟高光合成管道，该管道使用单目几何、菲涅耳感知着色和随机光照渲染物理上合理的高光，使模型能够在具有正确几何结构的任意RGB图像上进行训练。UnReflectAnything能够在自然和外科手术领域泛化，这些领域中的非朗伯表面和非均匀光照会产生严重的高光，并且在几个基准测试中取得了与最先进方法相当的性能。项目页面：https://alberto-rota.github.io/UnReflectAnything/


### 论文摘要

Specular highlights distort appearance, obscure texture, and hinder geometric reasoning in both natural and surgical imagery. We present UnReflectAnything, an RGB-only framework that removes highlights from a single image by predicting a highlight map together with a reflection-free diffuse reconstruction. The model uses a frozen vision transformer encoder to extract multi-scale features, a lightweight head to localize specular regions, and a token-level inpainting module that restores corrupted feature patches before producing the final diffuse image. To overcome the lack of paired supervision, we introduce a Virtual Highlight Synthesis pipeline that renders physically plausible specularities using monocular geometry, Fresnel-aware shading, and randomized lighting which enables training on arbitrary RGB images with correct geometric structure. UnReflectAnything generalizes across natural and surgical domains where non-Lambertian surfaces and non-uniform lighting create severe highlights and it achieves competitive performance with state-of-the-art results on several benchmarks. Project Page: https://alberto-rota.github.io/UnReflectAnything/

---

## 35. MeViS: A Multi-Modal Dataset for Referring Motion Expression Video Segmentation

**论文链接:** [http://arxiv.org/abs/2512.10945v1](http://arxiv.org/abs/2512.10945v1)

**作者:** Henghui Ding, Chang Liu, Shuting He, Kaining Ying, Xudong Jiang, Chen Change Loy, Yu-Gang Jiang

**发布时间:** 2025-12-11

**DOI:** 10.1109/TPAMI.2025.3600507

**备注:** IEEE TPAMI, Project Page: https://henghuiding.com/MeViS/

### GPT解析

### 总结

这篇论文提出了MeViS大规模多模态数据集，用于指代表达视频分割，包含33,072个人类标注的运动表达（文本和音频），涵盖2,006个复杂场景视频中的8,171个物体。作者对15种现有方法在4个任务上进行了基准测试，分析了现有方法的局限性，并提出了一种新方法LMPM++，在多个任务上达到了新的最先进结果。

### 背景

现有的指代表达视频分割数据集通常关注显著物体，并使用富含静态属性的语言表达，这可能使目标物体在单帧中被识别出来。这些数据集低估了运动在视频和语言中的作用。

### 目的

探索使用运动表达和运动推理线索进行像素级视频理解的可行性，开发一个促进复杂视频场景中运动表达引导视频理解算法发展的平台。

### 方法

创建了MeViS数据集，包含33,072个人类标注的运动表达（文本和音频），涵盖8,171个物体在2,006个复杂场景视频中。对15种现有方法在4个任务上进行了基准测试，包括6种指代表达视频对象分割方法、3种音频引导视频对象分割方法、2种指代表达多目标跟踪方法和4种视频描述方法（用于新引入的指代表达运动生成任务）。提出了一种新方法LMPM++用于RVOS/AVOS/RMOT。

### 主要发现

现有方法在处理运动表达引导的视频理解方面存在弱点和局限性。作者提出的方法LMPM++在多个任务上达到了新的最先进结果。

### 结论

MeViS数据集为复杂视频场景中运动表达引导的视频理解算法发展提供了平台。数据集和方法源代码已公开。

### 翻译

这篇论文提出了一个用于指代表达视频分割的大规模多模态数据集，专注于基于物体运动的语言描述来分割和跟踪视频中的目标物体。现有的指代表达视频分割数据集通常关注显著物体，并使用富含静态属性的语言表达，这可能使目标物体在单帧中被识别出来。这些数据集低估了运动在视频和语言中的作用。为了探索使用运动表达和运动推理线索进行像素级视频理解的可行性，我们引入了MeViS，这是一个包含33,072个人类标注的运动表达（文本和音频）的数据集，涵盖了2,006个复杂场景视频中的8,171个物体。我们在MeViS支持的4个任务上对15种现有方法进行了基准测试，包括6种指代表达视频对象分割方法、3种音频引导视频对象分割方法、2种指代表达多目标跟踪方法和4种用于新引入的指代表达运动生成任务的视频描述方法。结果表明，现有方法在处理运动表达引导的视频理解方面存在弱点和局限性。我们进一步分析了挑战，并提出了一种用于RVOS/AVOS/RMOT的方法LMPM++，达到了新的最先进结果。我们的数据集为促进复杂视频场景中运动表达引导的视频理解算法发展提供了平台。提出的MeViS数据集和方法源代码已在https://henghuiding.com/MeViS/公开。


### 论文摘要

This paper proposes a large-scale multi-modal dataset for referring motion expression video segmentation, focusing on segmenting and tracking target objects in videos based on language description of objects' motions. Existing referring video segmentation datasets often focus on salient objects and use language expressions rich in static attributes, potentially allowing the target object to be identified in a single frame. Such datasets underemphasize the role of motion in both videos and languages. To explore the feasibility of using motion expressions and motion reasoning clues for pixel-level video understanding, we introduce MeViS, a dataset containing 33,072 human-annotated motion expressions in both text and audio, covering 8,171 objects in 2,006 videos of complex scenarios. We benchmark 15 existing methods across 4 tasks supported by MeViS, including 6 referring video object segmentation (RVOS) methods, 3 audio-guided video object segmentation (AVOS) methods, 2 referring multi-object tracking (RMOT) methods, and 4 video captioning methods for the newly introduced referring motion expression generation (RMEG) task. The results demonstrate weaknesses and limitations of existing methods in addressing motion expression-guided video understanding. We further analyze the challenges and propose an approach LMPM++ for RVOS/AVOS/RMOT that achieves new state-of-the-art results. Our dataset provides a platform that facilitates the development of motion expression-guided video understanding algorithms in complex video scenes. The proposed MeViS dataset and the method's source code are publicly available at https://henghuiding.com/MeViS/

---

## 36. SWiT-4D: Sliding-Window Transformer for Lossless and Parameter-Free Temporal 4D Generation

**论文链接:** [http://arxiv.org/abs/2512.10860v1](http://arxiv.org/abs/2512.10860v1)

**作者:** Kehong Gong, Zhengyu Wen, Mingxi Xu, Weixia He, Qi Wang, Ning Zhang, Zhengyu Li, Chenbin Li, Dongze Lian, Wei Zhao, Xiaoyu He, Mingyuan Zhang

**发布时间:** 2025-12-11

**备注:** Project page: https://animotionlab.github.io/SWIT4D/

### GPT解析

### 总结

论文提出了SWiT-4D，一种滑动窗口Transformer模型，用于将单目视频转换为高质量动画3D资产和显式4D网格，仅需单个短视频微调即可实现高保真几何和时间一致性。

### 背景

尽管4D内容生成取得进展，但将单目视频转换为高质量动画3D资产仍有挑战；大规模自然捕获的4D网格数据集稀缺，限制了纯数据驱动方式训练视频到4D模型的能力。

### 目的

为了更好地利用图像到3D生成领域的先验模型，同时最小化对4D监督的依赖，实现无损、无参数的时间4D网格生成。

### 方法

SWiT-4D与基于Diffusion Transformer(DiT)的图像到3D生成器无缝集成，添加时空建模同时保持单图像前向过程，实现任意长度视频的4D网格重建；引入针对静态相机单目视频定制的基于优化的轨迹模块以恢复全局平移。

### 主要发现

SWiT-4D展示强大数据效率：仅用单个短于10秒视频微调即可实现高保真几何和稳定时间一致性；在多个数据集和基准测试中，时间平滑性始终优于现有基线。

### 结论

SWiT-4D有效解决了单目视频到高质量4D网格转换的挑战，通过利用现有图像到3D模型的先验知识，减少对大量4D训练数据的依赖，为实际应用提供了可行解决方案。

### 翻译

尽管在4D内容生成方面取得了显著进展，但将单目视频转换为具有显式4D网格的高质量动画3D资产仍然相当具有挑战性。大规模自然捕获的4D网格数据集的稀缺性进一步限制了以纯数据驱动方式从头开始训练可泛化的视频到4D模型的能力。与此同时，在广泛数据集支持下的图像到3D生成进展提供了可以利用的强大先验模型。为了更好地利用这些先验同时最小化对4D监督的依赖，我们引入了SWiT-4D，一种用于无损、无参数时间4D网格生成的滑动窗口Transformer。SWiT-4D可以与任何基于Diffusion Transformer(DiT)的图像到3D生成器无缝集成，在视频帧间添加时空建模，同时保持原始单图像前向过程，从而实现任意长度视频的4D网格重建。为了恢复全局平移，我们进一步引入了一个针对静态相机单目视频定制的基于优化的轨迹模块。SWiT-4D展示了强大的数据效率：仅使用单个短于10秒的视频进行微调，即可实现高保真几何和稳定的时间一致性，表明在极度有限的4D监督下具有实际部署的可能性。在领域内动物园测试集和具有挑战性的领域外基准测试(C4D、Objaverse和野外视频)上的综合实验表明，SWiT-4D在时间平滑性方面始终优于现有基线。项目页面：https://animotionlab.github.io/SWIT4D/

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决如何将单目视频转换为高质量的动画3D资产（4D网格）的问题。这个问题很重要，因为目前缺乏大规模自然捕获的4D网格数据集，使得从零开始学习视频到4D模型非常困难。同时，现有的图像到3D模型虽然能处理静态场景，但无法进行时间上的推理，而现有的4D生成方法要么计算成本高，要么引入额外参数导致泛化能力减弱。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先认识到图像到3D模型（如TripoSG）已取得显著进展并拥有强大先验，然后思考如何利用这些模型同时最小化对4D监督的依赖。他们设计了一种滑动窗口机制来处理输入序列，使模型能够捕获时间动态。该方法借鉴了现有的DiT架构和1D旋转位置编码（1D-RoPE）技术，保留了流匹配训练目标，没有引入额外时间损失，而是通过滑动窗口注意力增强自注意力和交叉注意力来注入时间信息。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是通过滑动窗口机制将图像到3D模型扩展到视频到4D任务，使用1D-RoPE编码时间关系，当W=0时保留原始单帧模型行为，当W>0时实现时间残差学习，并添加轨迹优化模块恢复全局平移。整体流程包括：预处理视频帧和3D网格表示；应用滑动窗口注意力到自注意力和交叉注意力层；使用1D-RoPE编码时间；通过滑动窗口处理视频序列生成3D网格；使用可微分渲染器优化网格平移和相机参数；计算掩码损失和中心损失来优化轨迹。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1) 滑动窗口变换器（SWiT）实现参数自由的时间扩展；2) 损失先验保留机制确保W=0时与原始单帧模型行为一致；3) 高效训练，4小时即可训练15亿参数模型；4) 高数据效率，仅需一个短视频微调；5) 轨迹优化模块恢复全局平移。相比之前的工作，SWiT-4D不同于V2M4的后验优化方法，不引入GVFD的额外网络参数，也不同于ShapeGen4D的专用时空层，而是通过滑动窗口机制增强现有注意力机制，实现了更高效的参数利用和更好的泛化能力。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'SWiT-4D通过滑动窗口变换器机制实现了对现有图像到3D模型的参数自由扩展，能够在不增加额外参数的情况下，将静态3D生成转变为时间一致的视频到4D生成，并仅需少量4D监督数据即可实现高质量结果。'}


### 论文摘要

Despite significant progress in 4D content generation, the conversion of monocular videos into high-quality animated 3D assets with explicit 4D meshes remains considerably challenging. The scarcity of large-scale, naturally captured 4D mesh datasets further limits the ability to train generalizable video-to-4D models from scratch in a purely data-driven manner. Meanwhile, advances in image-to-3D generation, supported by extensive datasets, offer powerful prior models that can be leveraged. To better utilize these priors while minimizing reliance on 4D supervision, we introduce SWiT-4D, a Sliding-Window Transformer for lossless, parameter-free temporal 4D mesh generation. SWiT-4D integrates seamlessly with any Diffusion Transformer (DiT)-based image-to-3D generator, adding spatial-temporal modeling across video frames while preserving the original single-image forward process, enabling 4D mesh reconstruction from videos of arbitrary length. To recover global translation, we further introduce an optimization-based trajectory module tailored for static-camera monocular videos. SWiT-4D demonstrates strong data efficiency: with only a single short (<10s) video for fine-tuning, it achieves high-fidelity geometry and stable temporal consistency, indicating practical deployability under extremely limited 4D supervision. Comprehensive experiments on both in-domain zoo-test sets and challenging out-of-domain benchmarks (C4D, Objaverse, and in-the-wild videos) show that SWiT-4D consistently outperforms existing baselines in temporal smoothness. Project page: https://animotionlab.github.io/SWIT4D/

---

## 37. Video Depth Propagation

**论文链接:** [http://arxiv.org/abs/2512.10725v1](http://arxiv.org/abs/2512.10725v1)

**作者:** Luigi Piccinelli, Thiemo Wandel, Christos Sakaridis, Wim Abbeloos, Luc Van Gool

**发布时间:** 2025-12-11

### GPT解析

### 总结

VeloDepth是一种高效且稳健的在线视频深度估计流水线，通过创新的传播模块和深度特征传播技术解决了现有方法中的时间一致性和计算效率问题，实现了稳定且准确的实时深度估计。

### 背景

视频深度估计在现实世界的视觉感知应用中至关重要，但现有方法存在局限性：要么依赖简单的逐帧单目模型导致时间不一致性和不准确性，要么使用计算密集型的时序建模不适合实时应用，这些限制显著限制了方法在实际设置中的通用性和性能。

### 目的

提出一种高效且稳健的在线视频深度估计流水线，有效利用先前深度预测的时空先验，执行深度特征传播，解决现有方法的时间一致性和计算效率问题。

### 方法

提出了一种新颖的传播模块，使用基于流的光学扭曲结合学习的残差校正来细化和传播深度特征和预测，设计上强制执行时间一致性，确保连续帧之间的稳定深度预测同时提高效率。

### 主要发现

在多个基准测试上的全面零样本评估表明，VeloDepth具有最先进的时间一致性和竞争性的准确性，与现有的视频深度估计器相比，VeloDepth的推理速度显著更快。

### 结论

VeloDepth为实时深度估计提供了实用、高效且准确的解决方案，适用于多种感知任务，代码和模型可在提供的GitHub链接获取。

### 翻译

视频深度估计在现实世界应用中的视觉感知至关重要。然而，现有方法要么依赖简单的逐帧单目模型，导致时间不一致性和不准确性，要么使用计算密集型的时序建模，不适合实时应用。这些限制显著限制了实际设置中的通用性和性能。为解决这一问题，我们提出了VeloDepth，一种高效且稳健的在线视频深度估计流水线，有效利用先前深度预测的时空先验并执行深度特征传播。我们的方法引入了一种新颖的传播模块，使用基于流的光学扭曲结合学习的残差校正来细化和传播深度特征和预测。此外，我们的设计在结构上强制执行时间一致性，从而在连续帧中产生稳定的深度预测并提高效率。在多个基准测试上的全面零样本评估表明，VeloDepth具有最先进的时间一致性和竞争性的准确性，同时与现有视频深度估计器相比具有显著更快的推理速度。因此，VeloDepth为实时深度估计提供了实用、高效且准确的解决方案，适用于各种感知任务。代码和模型可在https://github.com/lpiccinelli-eth/velodepth获取。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决视频深度估计中的两个关键问题：1) 现有的单目深度估计方法（逐帧处理）会导致时间不一致性和不准确；2) 现有的视频深度估计方法计算量太大，不适合实时应用。这个问题很重要，因为深度估计是计算机视觉的基础任务，对自动驾驶、机器人、增强现实和医学等多种应用至关重要。在实际应用中，尤其是实时场景中，我们需要既准确又时间一致的深度估计，同时保持计算效率。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者思考过程基于视频序列本身提供了比单图像更强的先验信息这一观察。他们设计了一个传播模块，利用前一帧的深度预测和特征，通过光流进行传播，然后通过残差校正来提高估计质量。作者借鉴了视频编码范式（使用运动向量和残差系数压缩数据），但针对深度估计任务进行了创新调整。他们还融合了光流估计、特征传播和深度估计等领域的现有技术，并在此基础上设计了多模态融合机制和关键帧选择策略。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是利用视频数据中的时空先验信息，特别是前一帧的深度预测和特征，通过光流进行传播，然后通过残差校正来提高深度估计的准确性和时间一致性。整体流程：1) 第一帧使用基础模型初始化；2) 传播模块接收前一帧的深度预测、特征和当前帧图像；3) 使用光流扭曲前一帧的特征和深度预测；4) 通过多模态编码器融合不同模态信息；5) 生成修正项校正扭曲特征；6) 基础解码器处理当前特征生成深度预测；7) 根据场景变化决定是否需要新的关键帧；8) 使用双向一致性损失确保时间一致性。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1) 传播模块：利用光流扭曲和残差校正精炼和传播深度特征；2) 多模态融合机制：融合RGB、深度预测和光流信息，使用门控机制确保可靠区域修正；3) 关键帧选择策略：根据光流大小和扭曲差异决定何时重新初始化；4) 一致性损失函数：在度量径向距离上计算双向一致性损失。相比之前工作，不同之处在于：不需要未来帧信息，可在线处理；计算效率更高，推理速度更快；能产生度量深度预测；避免了循环网络的漂移和梯度消失问题；更充分地利用了过去信息。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '这篇论文提出了VeloDepth，一种高效的在线视频深度估计方法，通过传播深度特征和预测，在保持高精度的同时显著提高了时间一致性和计算效率。'}


### 论文摘要

Depth estimation in videos is essential for visual perception in real-world applications. However, existing methods either rely on simple frame-by-frame monocular models, leading to temporal inconsistencies and inaccuracies, or use computationally demanding temporal modeling, unsuitable for real-time applications. These limitations significantly restrict general applicability and performance in practical settings. To address this, we propose VeloDepth, an efficient and robust online video depth estimation pipeline that effectively leverages spatiotemporal priors from previous depth predictions and performs deep feature propagation. Our method introduces a novel Propagation Module that refines and propagates depth features and predictions using flow-based warping coupled with learned residual corrections. In addition, our design structurally enforces temporal consistency, resulting in stable depth predictions across consecutive frames with improved efficiency. Comprehensive zero-shot evaluation on multiple benchmarks demonstrates the state-of-the-art temporal consistency and competitive accuracy of VeloDepth, alongside its significantly faster inference compared to existing video-based depth estimators. VeloDepth thus provides a practical, efficient, and accurate solution for real-time depth estimation suitable for diverse perception tasks. Code and models are available at https://github.com/lpiccinelli-eth/velodepth

---

## 38. Track and Caption Any Motion: Query-Free Motion Discovery and Description in Videos

**论文链接:** [http://arxiv.org/abs/2512.10607v1](http://arxiv.org/abs/2512.10607v1)

**作者:** Bishoy Galoaa, Sarah Ostadabbas

**发布时间:** 2025-12-11

### GPT解析

### 总结

本文提出了Track and Caption Any Motion (TCAM)框架，这是一个以运动为中心的自动视频理解系统，无需用户查询即可发现和描述视频中的运动模式。

### 背景

在遮挡、伪装或快速移动等挑战性条件下理解视频，往往更多依赖于运动动力学而非静态外观。现有方法通常需要用户查询才能识别和描述运动活动。

### 目的

开发一个能自动观察视频、识别多种运动活动，并将自然语言描述与相应轨迹进行空间对齐的框架，无需用户查询。

### 方法

TCAM通过运动场注意力机制将自然语言描述与相应轨迹进行空间对齐。通过结合全局视频-文本对齐与细粒度空间对应的统一训练，利用多头交叉注意力实现无需查询的多种运动表达发现。核心创新是将运动模式与对比视觉-语言表示对齐，为识别和描述动作提供强大语义信号。

### 主要发现

在MeViS基准测试中，TCAM实现了58.4%的视频到文本检索率，64.9的空间接地JF值，每视频发现4.8个相关表达，精确率达84.7%，展示了强大的跨任务泛化能力。

### 结论

TCAM成功实现了无需用户查询的自动视频理解，能够发现和描述多种运动模式，在多个任务上表现出强大的泛化能力。

### 翻译

我们提出了Track and Caption Any Motion (TCAM)，这是一个以运动为中心的自动视频理解框架，能够发现和描述无需用户查询的运动模式。在遮挡、伪装或快速移动等具有挑战性的条件下理解视频，往往更多地依赖于运动动力学而非静态外观。TCAM自动观察视频，识别多种运动活动，并通过运动场注意力机制将每个自然语言描述与其相应轨迹进行空间接地。我们的核心见解是，运动模式与对比视觉-语言表示对齐时，为识别和描述动作提供了强大的语义信号。通过结合全局视频-文本对齐与细粒度空间对应的统一训练，TCAM通过多头交叉注意力实现了无需查询的多种运动表达发现。在MeViS基准测试中，TCAM实现了58.4%的视频到文本检索率，64.9的空间接地JF值，每视频发现4.8个相关表达，精确率达84.7%，展示了强大的跨任务泛化能力。


### 论文摘要

We propose Track and Caption Any Motion (TCAM), a motion-centric framework for automatic video understanding that discovers and describes motion patterns without user queries. Understanding videos in challenging conditions like occlusion, camouflage, or rapid movement often depends more on motion dynamics than static appearance. TCAM autonomously observes a video, identifies multiple motion activities, and spatially grounds each natural language description to its corresponding trajectory through a motion-field attention mechanism. Our key insight is that motion patterns, when aligned with contrastive vision-language representations, provide powerful semantic signals for recognizing and describing actions. Through unified training that combines global video-text alignment with fine-grained spatial correspondence, TCAM enables query-free discovery of multiple motion expressions via multi-head cross-attention. On the MeViS benchmark, TCAM achieves 58.4% video-to-text retrieval, 64.9 JF for spatial grounding, and discovers 4.8 relevant expressions per video with 84.7% precision, demonstrating strong cross-task generalization.

---

## 39. Decoding Student Minds: Leveraging Conversational Agents for Psychological and Learning Analysis

**论文链接:** [http://arxiv.org/abs/2512.10441v1](http://arxiv.org/abs/2512.10441v1)

**作者:** Nour El Houda Ben Chaabene, Hamza Hammami, Laid Kahloul

**发布时间:** 2025-12-11

**备注:** This manuscript is currently under peer review in Expert Systems with Applications

### GPT解析

### 总结

本研究提出了一种具有心理学意识的对话代理系统，旨在提升教育环境中的学习表现和情绪健康，通过多模态数据分析实时评估学生状态。

### 背景

现有的聊天机器人通常仅限于提供辅导或情感支持，缺乏全面分析学生认知和情感状态的能力。

### 目的

设计一个能够同时提升学习表现和情绪健康的对话代理系统，通过实时分析学生状态提供个性化教育干预。

### 方法

系统结合大型语言模型、知识图谱增强的BERT和带有注意力机制的双向LSTM，利用文本语义、韵律语音特征和时间行为趋势等多模态数据进行学生状态分类。

### 主要发现

对大学生的初步研究表明，该系统能提高学习动机，减轻压力，并带来中等程度的学术进步。

### 结论

整合语义推理、多模态融合和时间建模的方法，为支持适应性、以学生为中心的教育干预提供了新可能。

### 翻译

本文提出了一种具有心理学意识的对话代理系统，旨在增强教育环境中的学习表现和情绪健康。该系统结合了大型语言模型、知识图谱增强的BERT和带有注意力机制的双向LSTM，用于实时分类学生的认知和情感状态。与之前仅限于辅导或情感支持的聊天机器人不同，我们的方法利用多模态数据（包括文本语义、韵律语音特征和时间行为趋势）来推断参与度、压力和概念理解。对大学生的初步研究表明，与基线方法相比，该方法提高了学习动机，减轻了压力，并取得了中等程度的学术进步。这些结果强调了整合语义推理、多模态融合和时间建模以支持适应性、以学生为中心的教育干预的潜力。


### 论文摘要

This paper presents a psychologically-aware conversational agent designed to enhance both learning performance and emotional well-being in educational settings. The system combines Large Language Models (LLMs), a knowledge graph-enhanced BERT (KG-BERT), and a bidirectional Long Short-Term Memory (LSTM) with attention to classify students' cognitive and affective states in real time. Unlike prior chatbots limited to either tutoring or affective support, our approach leverages multimodal data-including textual semantics, prosodic speech features, and temporal behavioral trends-to infer engagement, stress, and conceptual understanding. A pilot study with university students demonstrated improved motivation, reduced stress, and moderate academic gains compared to baseline methods. These results underline the promise of integrating semantic reasoning, multimodal fusion, and temporal modeling to support adaptive, student-centered educational interventions.

---

## 40. EchoingPixels: Cross-Modal Adaptive Token Reduction for Efficient Audio-Visual LLMs

**论文链接:** [http://arxiv.org/abs/2512.10324v1](http://arxiv.org/abs/2512.10324v1)

**作者:** Chao Gong, Depeng Wang, Zhipeng Wei, Ya Guo, Huijia Zhu, Jingjing Chen

**发布时间:** 2025-12-11

### GPT解析

### 总结

EchoingPixels是一种创新的音频视觉大语言模型框架，通过跨模态语义筛(CS2)和同步增强RoPE技术，有效减少了token数量并保持模型性能。

### 背景

音频视觉大语言模型(AV-LLMs)因处理大量音频和视频token而面临巨大的计算开销。现有的token减少方法主要针对单一模态，无法充分利用音频视觉间的跨模态协同效应，且静态分配每个模态的预算不够优化。

### 目的

解决在联合音频视觉流上进行token减少的瓶颈问题，同时保持跨模态协同效应和时序建模能力。

### 方法

提出EchoingPixels框架，其核心是跨模态语义筛(CS2)模块，该模块共同关注联合多模态流，从整个音频视觉token池中减少token，而非独立处理每个模态。同时设计同步增强RoPE技术，确保稀疏选择token的关键时间关系得以保持。

### 主要发现

EchoingPixels仅使用原始token的5-20%就能达到与强基线相当的性能，实现了2-3倍的加速和内存减少。

### 结论

EchoingPixels有效解决了AV-LLMs中的token减少问题，通过自适应分配token预算和动态识别重要token，显著降低了计算开销同时保持了模型性能。

### 翻译

音频视觉大语言模型(AV-LLMs)因处理大量音频和视频token而面临巨大的计算开销。虽然token减少技术在视频-only LLMs中已被广泛研究，但对于音频视觉领域来说还不够充分，因为这些单模态方法无法利用音频视觉跨模态协同效应。此外，音频和视频的不同和动态信息密度使得每个模态的静态预算次优。因此，如何在联合音频视觉流上执行token减少仍然是一个未解决的瓶颈。为了填补这一空白，我们引入了EchoingPixels，一个受现实场景中视觉和声音共存与交互启发的框架。我们框架的核心是跨模态语义筛(CS2)，一个实现早期音频视觉交互的模块。CS2不是独立压缩模态，而是共同关注联合多模态流，并从整个音频视觉token池中减少token，而不是使用每个模态的固定预算。这种单池方法允许它跨两个模态自适应分配token预算，并共同动态识别重要token。为确保这种激进的减少保持关键的时序建模能力，我们共同设计了同步增强RoPE(Sync-RoPE)来为稀疏选择的token保持关键的时间关系。大量实验证明，EchoingPixels仅使用原始token的5-20%就能与强基线实现相当的性能，同时实现了2-3倍的加速和内存减少。


### 论文摘要

Audio-Visual Large Language Models (AV-LLMs) face prohibitive computational overhead from massive audio and video tokens. Token reduction, while extensively explored for video-only LLMs, is insufficient for the audio-visual domain, as these unimodal methods cannot leverage audio-visual cross-modal synergies. Furthermore, the distinct and dynamic information densities of audio and video render static budgets per modality suboptimal. How to perform token reduction on a joint audio-visual stream thus remains an unaddressed bottleneck. To fill this gap, we introduce EchoingPixels, a framework inspired by the coexistence and interaction of visuals and sound in real-world scenes. The core of our framework is the Cross-Modal Semantic Sieve (CS2), a module enabling early audio-visual interaction. Instead of compressing modalities independently, CS2 co-attends to the joint multimodal stream and reduces tokens from an entire combined pool of audio-visual tokens rather than using fixed budgets per modality. This single-pool approach allows it to adaptively allocate the token budget across both modalities and dynamically identify salient tokens in concert. To ensure this aggressive reduction preserves the vital temporal modeling capability, we co-design a Synchronization-Augmented RoPE (Sync-RoPE) to maintain critical temporal relationships for the sparsely selected tokens. Extensive experiments demonstrate that EchoingPixels achieves performance comparable to strong baselines using only 5-20% of the original tokens, with a 2-3x speedup and memory reduction.

---

## 41. Beyond Lux thresholds: a systematic pipeline for classifying biologically relevant light contexts from wearable data

**论文链接:** [http://arxiv.org/abs/2512.06181v2](http://arxiv.org/abs/2512.06181v2)

**作者:** Yanuo Zhou

**发布时间:** 2025-12-05

**备注:** Withdrawn at the request of affiliated institution

### GPT解析

### 总结

该研究建立并验证了一个可重复的流程，用于从可穿戴光谱数据中区分自然光与人工光，在受试者泛化下表现出色。

### 背景

可穿戴光谱仪能够量化生物相关光，但可重复的上下文分类流程尚未明确指定。

### 目的

建立并验证一个按受试者评估的可重复流程，以及从可穿戴光谱数据中区分自然光与人工光的可行设计规则。

### 方法

分析了26名参与者的ActLumus记录，每人至少监测7天，采样间隔为10秒，配以日常暴露日记。流程包括固定序列：域选择、对数转换、排除总强度的归一化、小时级聚合、时间编码和机器学习分类器，在参与者的交叉验证下进行评估。

### 主要发现

所提出的序列在主要任务上表现出色，代表性配置在保留的受试者分割上达到接近完美的自然光与人工光分类。相比之下，室内与室外分类由于光谱重叠和类别不平衡，表现较差。阈值基线不足，支持需要超越简单照度截止值的复杂建模方法。

### 结论

提供了一个可重复、可审核的基线流程和设计规则，用于受试者泛化下的上下文光分类。所有代码、配置文件和衍生工件都将公开存档，以支持重用和基准测试。

### 翻译

背景：可穿戴光谱仪能够实现生物相关光的现场量化，但可重复的上下文分类流程尚未明确。目的：建立并验证一个按受试者评估的可重复流程，以及从可穿戴光谱数据中区分自然光与人工光的可行设计规则。方法：我们分析了26名参与者的ActLumus记录，每人至少监测7天，采样间隔为10秒，配以日常暴露日记。该流程固定了顺序：域选择、以10为底的对数转换、排除总强度的归一化（避免亮度捷径）、小时级聚合、时间编码和机器学习分类器，在受试者交叉验证下进行评估。结果：所提出的序列在主要任务上始终表现出色，代表性配置在保留的受试者分割上达到优秀的自然光与人工光分类。相比之下，由于光谱重叠和类别不平衡，室内与室外分类仍处于可行性水平。阈值基线在我们的数据上不足，支持需要超越照度截止值的谱时建模。结论：我们提供了一个可重复、可审核的基线流程和设计规则，用于受试者泛化下的上下文光分类。所有代码、配置文件和衍生工件都将公开存档，以支持重用和基准测试。


### 论文摘要

Background: Wearable spectrometers enable field quantification of biologically relevant light, yet reproducible pipelines for contextual classification remain under-specified.   Objective: To establish and validate a subject-wise evaluated, reproducible pipeline and actionable design rules for classifying natural vs. artificial light from wearable spectral data.   Methods: We analysed ActLumus recordings from 26 participants, each monitored for at least 7 days at 10-second sampling, paired with daily exposure diaries. The pipeline fixes the sequence: domain selection, log-base-10 transform, L2 normalisation excluding total intensity (to avoid brightness shortcuts), hour-level medoid aggregation, sine/cosine hour encoding, and MLP classifier, evaluated under participant-wise cross-validation.   Results: The proposed sequence consistently achieved high performance on the primary task, with representative configurations reaching AUC = 0.938 (accuracy 88%) for natural vs. artificial classification on the held-out subject split. In contrast, indoor vs. outdoor classification remained at feasibility level due to spectral overlap and class imbalance (best AUC approximately 0.75; majority-class collapse without contextual sensors). Threshold baselines were insufficient on our data, supporting the need for spectral-temporal modelling beyond illuminance cut-offs.   Conclusions: We provide a reproducible, auditable baseline pipeline and design rules for contextual light classification under subject-wise generalisation. All code, configuration files, and derived artefacts will be openly archived (GitHub + Zenodo DOI) to support reuse and benchmarking.

---

## 42. 论文ID: 2512.10244v1

**错误:** 无法读取论文信息 - [Errno 2] No such file or directory: 'storage/result/2512.10244v1.json'

---

## 43. Meta-learning three-factor plasticity rules for structured credit assignment with sparse feedback

**论文链接:** [http://arxiv.org/abs/2512.09366v2](http://arxiv.org/abs/2512.09366v2)

**作者:** Dimitra Maoutsa

**发布时间:** 2025-12-10

**备注:** 10 pages, 2 figures; accepted & presented at NeurIPS 2025 workshop Symmetry and Geometry in Neural Representations (NeurReps); v2: appendix typo resolved

### GPT解析

### 总结

该研究提出了一种元学习框架，用于发现循环网络中结构化信用分配的局部学习规则，这些网络使用稀疏反馈进行训练。该方法结合了任务执行期间的局部更新和通过学习过程中的切线传播优化可塑性参数的外部循环，发现的三因子学习规则仅使用局部信息和延迟奖励就能实现长时间尺度的信用分配。

### 背景

生物神经网络能够从稀疏、延迟的反馈中学习复杂行为，使用局部突触可塑性，但支持结构化信用分配的机制仍然不清楚。相比之下，解决类似任务的人工循环网络通常依赖于生物上不合理的全局学习规则或手工制作的局部更新。能够支持从延迟强化中学习的局部可塑性规则的空间很大程度上尚未被探索。

### 目的

发现循环网络中结构化信用分配的局部学习规则，这些网络使用稀疏反馈进行训练。

### 方法

提出一个元学习框架，该方法在任务执行期间交错进行局部新赫布式样更新，并通过学习过程中的切线传播优化可塑性参数。

### 主要发现

发现的三因子学习规则仅使用局部信息和延迟奖励就能实现长时间尺度的信用分配。

### 结论

研究结果提供了对循环电路学习中生物基础机制的新见解。

### 翻译

生物神经网络使用局部突触可塑性从稀疏、延迟的反馈中学习复杂行为，但支持结构化信用分配的机制仍然不清楚。相比之下，解决类似任务的人工循环网络通常依赖于生物上不合理的全局学习规则或手工制作的局部更新。能够支持从延迟强化中学习的局部可塑性规则的空间很大程度上尚未被探索。在此，我们提出了一种元学习框架，用于发现循环网络中结构化信用分配的局部学习规则，这些网络使用稀疏反馈进行训练。我们的方法在任务执行期间交错进行局部新赫布式样更新，并通过学习过程中的切线传播优化可塑性参数。发现的三因子学习规则仅使用局部信息和延迟奖励就能实现长时间尺度的信用分配，为循环电路学习中生物基础机制提供了新见解。


### 论文摘要

Biological neural networks learn complex behaviors from sparse, delayed feedback using local synaptic plasticity, yet the mechanisms enabling structured credit assignment remain elusive. In contrast, artificial recurrent networks solving similar tasks typically rely on biologically implausible global learning rules or hand-crafted local updates. The space of local plasticity rules capable of supporting learning from delayed reinforcement remains largely unexplored. Here, we present a meta-learning framework that discovers local learning rules for structured credit assignment in recurrent networks trained with sparse feedback. Our approach interleaves local neo-Hebbian-like updates during task execution with an outer loop that optimizes plasticity parameters via \textbf{tangent-propagation through learning}. The resulting three-factor learning rules enable long-timescale credit assignment using only local information and delayed rewards, offering new insights into biologically grounded mechanisms for learning in recurrent circuits.

---

## 44. Empowering Dynamic Urban Navigation with Stereo and Mid-Level Vision

**论文链接:** [http://arxiv.org/abs/2512.10956v1](http://arxiv.org/abs/2512.10956v1)

**作者:** Wentao Zhou, Xuweiyi Chen, Vignesh Rajagopal, Jeffrey Chen, Rohan Chandra, Zezhou Cheng

**发布时间:** 2025-12-11

**备注:** Project Page: https://www.cs.virginia.edu/~tsx4zn/stereowalk/

### GPT解析

### 总结

本文提出了一种结合立体输入和中级视觉的机器人导航模型StereoWalker，证明在机器人导航中结合立体视觉和中级视觉模块比仅依赖单目视觉的端到端模型更高效。

### 背景

基础模型在语言和视觉领域的成功激发了端到端机器人导航基础模型(NFMs)的研究，这类模型直接将单目视觉输入映射到控制动作，完全忽略中级视觉模块。然而，这种方法需要大量难以获得的像素到动作监督，且在动态和非结构化环境中面临深度尺度歧义等挑战。

### 目的

证明仅依赖单目视觉并忽略中级视觉先验是低效的，提出一种结合立体输入和中级视觉的导航方法。

### 方法

提出了StereoWalker模型，通过立体输入和显式的中级视觉(如深度估计和密集像素跟踪)来增强NFMs。同时创建了一个大型立体导航数据集，包含来自互联网立体视频的自动动作标注。

### 主要发现

中级视觉使StereoWalker仅使用1.5%的训练数据就能达到最先进水平，使用全部数据时超越了最先进水平，且立体视觉比单目输入产生更高的导航性能。

### 结论

结合立体输入和中级视觉的导航模型比仅依赖单目视觉的端到端模型更有效，能在更少数据的情况下实现更好的导航性能。

### 翻译

基础模型在语言和视觉领域的成功激发了端到端机器人导航基础模型(NFMs)的研究。NFMs直接将单目视觉输入映射到控制动作，完全忽略中级视觉模块(跟踪、深度估计等)。虽然视觉能力会隐式出现的假设很有吸引力，但需要大量难以获得的像素到动作监督。在动态和非结构化环境中，这一挑战尤为明显，因为稳健导航需要精确的几何和动态理解，而单目视图中的深度尺度歧义进一步限制了准确的空间推理。在本文中，我们证明仅依赖单目视觉并忽略中级视觉先验是低效的。我们提出了StereoWalker，通过立体输入和显式的中级视觉(如深度估计和密集像素跟踪)来增强NFMs。我们的直觉很简单：立体输入解决了深度尺度歧义问题，现代中级视觉模型在动态场景中提供可靠的几何和运动结构。我们还整理了一个大型立体导航数据集，包含来自互联网立体视频的自动动作标注，以支持StereoWalker的训练并促进未来研究。通过实验，我们发现中级视觉使StereoWalker仅使用1.5%的训练数据就能达到最先进水平，并使用全部数据超越了最先进水平。我们还观察到立体视觉比单目输入产生更高的导航性能。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决动态城市导航中单目视觉导航基础模型(NFMs)的深度尺度模糊问题，以及需要大量像素到动作监督数据才能获得良好性能的挑战。这个问题在现实中非常重要，因为城市导航对于最后一英里配送等应用至关重要，而动态和非结构化环境中的行人移动、不规则路边配置等复杂场景对导航系统提出了极高要求，单目视觉的局限性限制了机器人在这些环境中的安全导航能力。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者基于两个关键洞察设计该方法：1)立体视觉输入可解决单目感知的深度尺度模糊问题；2)显式的中间视觉模块可改善泛化能力、稳定性和数据效率。作者借鉴了现有工作：参考了中间视觉在机器人领域的研究，表明显式视觉表示可提高任务性能；借鉴了立体视觉在机器人中的应用，提供准确深度信息；参考了视觉导航基础模型的发展，特别是从互联网挖掘导航数据的方法。StereoWalker整合了立体视觉和中间视觉模块，而非像现有NFMs仅依赖单目视觉。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是通过立体视觉解决深度尺度模糊问题，并利用显式的中间视觉模块提供几何和运动结构信息，从而提高导航性能并减少对大量训练数据的依赖。整体实现流程：1)收集并过滤互联网VR180立体视频，确保包含目标导向行走；2)使用立体视觉里程计计算轨迹标签；3)模型架构包括：图像标记化(使用DINOv2、DepthAnythingV2和CoTracker-v3)、深度聚合、跟踪引导的注意力、全局和目标标记注意力；4)使用复合损失函数进行训练，包括路径点预测、到达概率和方向损失。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点：1)提出StereoWalker，整合立体视觉和显式中间视觉模块；2)收集并发布新的立体导航数据集，包含全球大都市行人行走视频；3)开发使用视觉语言模型过滤内容的过滤器；4)在多个基准和真实世界测试中验证方法。不同之处：与现有NFMs不同，StereoWalker使用立体视觉而非单目视觉；显式集成中间视觉模块而非期望其隐式出现；保留所有图像块标记而非压缩为单个标记；仅使用1.5%训练数据就能达到 comparable性能，使用全部数据时超越现有最佳方法。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '本文提出的StereoWalker通过整合立体视觉和显式中间视觉模块，显著提高了动态城市导航的性能和训练效率，仅使用1.5%的训练数据就能达到现有最佳水平。'}


### 论文摘要

The success of foundation models in language and vision motivated research in fully end-to-end robot navigation foundation models (NFMs). NFMs directly map monocular visual input to control actions and ignore mid-level vision modules (tracking, depth estimation, etc) entirely. While the assumption that vision capabilities will emerge implicitly is compelling, it requires large amounts of pixel-to-action supervision that are difficult to obtain. The challenge is especially pronounced in dynamic and unstructured settings, where robust navigation requires precise geometric and dynamic understanding, while the depth-scale ambiguity in monocular views further limits accurate spatial reasoning. In this paper, we show that relying on monocular vision and ignoring mid-level vision priors is inefficient.   We present StereoWalker, which augments NFMs with stereo inputs and explicit mid-level vision such as depth estimation and dense pixel tracking. Our intuition is straightforward: stereo inputs resolve the depth-scale ambiguity, and modern mid-level vision models provide reliable geometric and motion structure in dynamic scenes. We also curate a large stereo navigation dataset with automatic action annotation from Internet stereo videos to support training of StereoWalker and to facilitate future research. Through our experiments, we find that mid-level vision enables StereoWalker to achieve a comparable performance as the state-of-the-art using only 1.5% of the training data, and surpasses the state-of-the-art using the full data. We also observe that stereo vision yields higher navigation performance than monocular input.

---

## 45. E-RayZer: Self-supervised 3D Reconstruction as Spatial Visual Pre-training

**论文链接:** [http://arxiv.org/abs/2512.10950v1](http://arxiv.org/abs/2512.10950v1)

**作者:** Qitao Zhao, Hao Tan, Qianqian Wang, Sai Bi, Kai Zhang, Kalyan Sunkavalli, Shubham Tulsiani, Hanwen Jiang

**发布时间:** 2025-12-11

**备注:** Project website: https://qitaozhao.github.io/E-RayZer

### GPT解析

### 总结

本文介绍了一种名为E-RayZer的自监督大型3D视觉模型，能够从未标记图像中直接学习真正具有3D感知能力的表示。

### 背景

自监督预训练已在语言、单个2D图像和视频的基础模型方面取得革命性进展，但在从多视图图像中学习3D感知表示方面仍 largely unexplored（很大程度上未被探索）。

### 目的

开发一种自监督大型3D视觉模型，从未标记图像中直接学习真正具有3D感知能力的表示。

### 方法

E-RayZer直接在3D空间中操作，执行具有显式几何形状的自监督3D重建，而非通过潜在空间视图合成间接推断3D。引入细粒度学习课程，以完全无监督方式组织从简单到困难样本的训练，并协调异构数据源。

### 主要发现

E-RayZer在姿态估计上显著优于RayZer，匹配或有时超越完全监督的重建模型如VGGT，在迁移到3D下游任务时，其表示优于DINOv3、CroCo v2、VideoMAE V2和RayZer等领先视觉预训练模型。

### 结论

E-RayZer建立了3D感知视觉预训练的新范式。

### 翻译

自监督预训练已经在语言、单个2D图像和视频的基础模型方面取得了革命性进展，但在从多视图图像中学习3D感知表示方面仍 largely unexplored（很大程度上未被探索）。在本文中，我们提出了E-RayZer，这是一种自监督的大型3D视觉模型，能够从未标记的图像中直接学习真正具有3D感知能力的表示。与之前的自监督方法（如通过潜在空间视图合成间接推断3D的RayZer）不同，E-RayZer直接在3D空间中操作，执行具有显式几何形状的自监督3D重建。这种表述消除了捷径解决方案，并产生了具有几何基础的表示。为确保收敛和可扩展性，我们引入了一种新颖的细粒度学习课程，以完全无监督的方式组织从简单到困难样本的训练，并协调异构数据源。实验表明，E-RayZer在姿态估计上显著优于RayZer，匹配或有时超越完全监督的重建模型（如VGGT）。此外，当迁移到3D下游任务时，其学习的表示优于领先的视觉预训练模型（如DINOv3、CroCo v2、VideoMAE V2和RayZer），确立了E-RayZer作为3D感知视觉预训练的新范式。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决如何从无标签的多视角图像中学习真正3D感知表示的问题。当前3D视觉模型主要依赖监督学习，使用通过COLMAP等工具估计的3D伪标签，这种方法效率低下、不完美且不可扩展。这个问题在研究中很重要，因为3D空间理解是感知和与我们生活的3D物理世界交互的基础，而自监督预训练在语言、2D图像和视频领域已证明有效，但在3D领域仍 largely unexplored。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先分析了现有方法特别是RayZer的局限性：它通过隐式视图合成进行3D学习，只表现出表面3D感知，依赖视频插值等快捷解决方案。作者提出关键见解：3D归纳偏置对3D表示学习很重要，但需以保持可扩展性的方式引入。他们设计了E-RayZer使用显式3D几何代替隐式表示，通过模型设计注入轻量级3D偏置，并提出基于视觉重叠的学习课程。借鉴了RayZer的基础架构、3D高斯溅射技术、VGGT的多视图变换器架构，以及各种视觉预训练方法作为基准。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是通过显式3D重建进行自监督3D视觉预训练，直接在3D空间中操作学习3D感知表示。整体流程：1)预测所有输入图像的相机参数；2)将图像分为参考视图和目标视图；3)从参考视图预测像素对齐的3D高斯；4)使用自我预测的目标视图相机渲染3D高斯；5)通过光度损失对渲染结果进行自监督训练；6)使用基于视觉重叠的学习课程稳定训练并提高可扩展性。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点：1)使用显式3D高斯作为场景表示，使特征真正具有3D感知能力；2)通过移除图像索引嵌入和特定架构设计，避免视图插值等快捷解决方案；3)提出基于视觉重叠的细粒度学习课程，提高训练稳定性和可扩展性；4)实现完全自监督的前馈3D重建，无需任何3D注释。相比RayZer等前工作，E-RayZer直接在3D空间操作而非潜在空间，使用更灵活的学习课程，避免了快捷解决方案，在姿态估计和下游任务上表现更优。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'E-RayZer通过引入显式3D高斯表示和基于视觉重叠的学习课程，首次实现了完全自监督的前馈3D重建，建立了空间视觉预训练的新范式，其学习到的3D感知表示在下游任务上显著优于现有方法，甚至可与监督方法相媲美。'}


### 论文摘要

Self-supervised pre-training has revolutionized foundation models for languages, individual 2D images and videos, but remains largely unexplored for learning 3D-aware representations from multi-view images. In this paper, we present E-RayZer, a self-supervised large 3D Vision model that learns truly 3D-aware representations directly from unlabeled images. Unlike prior self-supervised methods such as RayZer that infer 3D indirectly through latent-space view synthesis, E-RayZer operates directly in 3D space, performing self-supervised 3D reconstruction with Explicit geometry. This formulation eliminates shortcut solutions and yields representations that are geometrically grounded. To ensure convergence and scalability, we introduce a novel fine-grained learning curriculum that organizes training from easy to hard samples and harmonizes heterogeneous data sources in an entirely unsupervised manner. Experiments demonstrate that E-RayZer significantly outperforms RayZer on pose estimation, matches or sometimes surpasses fully supervised reconstruction models such as VGGT. Furthermore, its learned representations outperform leading visual pre-training models (e.g., DINOv3, CroCo v2, VideoMAE V2, and RayZer) when transferring to 3D downstream tasks, establishing E-RayZer as a new paradigm for 3D-aware visual pre-training.

---

## 46. BabyVLM-V2: Toward Developmentally Grounded Pretraining and Benchmarking of Vision Foundation Models

**论文链接:** [http://arxiv.org/abs/2512.10932v1](http://arxiv.org/abs/2512.10932v1)

**作者:** Shengao Wang, Wenqi Wang, Zecheng Wang, Max Whitton, Michael Wakeham, Arjun Chandra, Joey Huang, Pengyue Zhu, Helen Chen, David Li, Jeffrey Li, Shawn Li, Andrew Zagula, Amy Zhao, Andrew Zhu, Sayaka Nakamura, Yuki Yamamoto, Jerry Jun Yokono, Aaron Mueller, Bryan A. Plummer, Kate Saenko, Venkatesh Saligrama, Boqing Gong

**发布时间:** 2025-12-11

### GPT解析

### 总结

BabyVLM-V2是一个基于婴儿启发的视觉语言建模框架，通过纵向多方面预训练集、多功能模型和DevCV认知评估工具箱改进了BabyVLM-V1，实现了在认知评估任务上的竞争性性能。

### 背景

儿童早期的发展轨迹为视觉基础模型的高效预训练设定了自然目标。

### 目的

开发一个基于婴儿启发的视觉语言建模框架，改进BabyVLM-V1，并促进视觉基础模型的发展性合理预训练研究。

### 方法

通过构建纵向、多方面的预训练集，最大化覆盖范围同时最小化筛选；开发多功能模型；创建DevCV认知评估工具箱，将NIH Baby Toolbox的视觉相关措施调整为十个多模态任务基准套件。

### 主要发现

从头预训练的紧凑模型在DevCV工具箱上实现了具有竞争力的性能，在某些任务上优于GPT-4o。

### 结论

BabyVLM-V2框架有望加速视觉基础模型的发展性合理预训练研究。

### 翻译

早期儿童的发展轨迹为视觉基础模型的高效预训练设定了自然目标。我们介绍了BabyVLM-V2，一个基于婴儿启发的视觉语言建模框架，通过纵向、多方面的预训练集、多功能模型以及最重要的DevCV认知评估工具箱，显著改进了BabyVLM-V1。预训练集最大化了覆盖范围，同时最小化了纵向、以婴儿为中心的视听语料库的筛选，产生了反映婴儿经验的视频-话语、图像-话语和多轮对话数据。DevCV工具箱将最近发布的NIH Baby Toolbox中所有与视觉相关的措施调整为十个多模态任务的基准套件，涵盖了与早期儿童能力相符合的空间推理、记忆和词汇理解。实验结果表明，从头开始预训练的紧凑模型可以在DevCV工具箱上实现具有竞争力的性能，在某些任务上优于GPT-4o。我们希望有原则的、统一的BabyVLM-V2框架将加速视觉基础模型的发展性合理预训练研究。


### 论文摘要

Early children's developmental trajectories set up a natural goal for sample-efficient pretraining of vision foundation models. We introduce BabyVLM-V2, a developmentally grounded framework for infant-inspired vision-language modeling that extensively improves upon BabyVLM-V1 through a longitudinal, multifaceted pretraining set, a versatile model, and, most importantly, DevCV Toolbox for cognitive evaluation. The pretraining set maximizes coverage while minimizing curation of a longitudinal, infant-centric audiovisual corpus, yielding video-utterance, image-utterance, and multi-turn conversational data that mirror infant experiences. DevCV Toolbox adapts all vision-related measures of the recently released NIH Baby Toolbox into a benchmark suite of ten multimodal tasks, covering spatial reasoning, memory, and vocabulary understanding aligned with early children's capabilities. Experimental results show that a compact model pretrained from scratch can achieve competitive performance on DevCV Toolbox, outperforming GPT-4o on some tasks. We hope the principled, unified BabyVLM-V2 framework will accelerate research in developmentally plausible pretraining of vision foundation models.

---

## 47. PoseGAM: Robust Unseen Object Pose Estimation via Geometry-Aware Multi-View Reasoning

**论文链接:** [http://arxiv.org/abs/2512.10840v1](http://arxiv.org/abs/2512.10840v1)

**作者:** Jianqi Chen, Biao Zhang, Xiangjun Tang, Peter Wonka

**发布时间:** 2025-12-11

**备注:** Project page: https://windvchen.github.io/PoseGAM/

### GPT解析

### 总结

本文提出了PoseGAM，一种几何感知的多视图框架，用于6D物体姿态估计，无需显式匹配即可直接从查询图像和多个模板图像预测物体姿态。

### 背景

6D物体姿态估计预测物体相对于摄像机的变换，对未见过的物体仍然具有挑战性。现有方法通常依赖于在查询图像和物体模型或模板图像之间显式构建特征对应关系。

### 目的

开发一种能够有效处理未见物体的6D物体姿态估计方法，提高准确性和泛化能力。

### 方法

PoseGAM基于多视图基础模型架构，通过两种互补机制集成物体几何信息：显式点状几何和从几何表示网络中学习到的特征。同时构建了一个包含19万多个物体的大型合成数据集，具有多样化的环境条件。

### 主要发现

在多个基准测试中展示了最先进的性能，比先前方法平均提高5.1%的AR指标，在某些数据集上最高提高17.6%，表明对未见物体具有强大的泛化能力。

### 结论

PoseGAM框架通过几何感知的多视图方法和大型合成数据集，显著提高了6D物体姿态估计的性能，特别是在处理未见物体方面表现出色。

### 翻译

6D物体姿态估计预测物体相对于摄像机的变换，对未见过的物体仍然具有挑战性。现有方法通常依赖于在查询图像和物体模型或模板图像之间显式构建特征对应关系。在这项工作中，我们提出了PoseGAM，一种几何感知的多视图框架，可以直接从查询图像和多个模板图像预测物体姿态，无需显式匹配。基于最近的多视图基础模型架构，该方法通过两种互补机制集成物体几何信息：显式点状几何和从几何表示网络中学习到的特征。此外，我们构建了一个包含19万多个物体的大型合成数据集，具有多样化的环境条件，以增强鲁棒性和泛化能力。在多个基准测试中的广泛评估展示了我们的最先进性能，比先前方法平均提高5.1%的AR指标，在某些数据集上最高提高17.6%，表明对未见物体具有强大的泛化能力。项目页面：https://windvchen.github.io/PoseGAM/

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': "论文主要解决'未见物体6D姿态估计'问题，即预测模型未训练过的物体在3D空间中的位置和方向。这个问题在现实中非常重要，因为它直接关系到机器人操作、增强现实、自动驾驶和内容创作等领域的应用。传统方法只能处理训练时见过的物体，而无法泛化到新遇到的物体，限制了这些技术的实际应用范围。", '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者受到多视图基础模型(Multi-view Foundation Models)成功的启发，这些模型可以直接从RGB图像推断3D几何信息。作者发现现有多视图模型主要依赖视觉输入，缺乏明确的3D物体模型信息，限制了它们在物体姿态估计中的有效性。因此，作者将物体几何信息整合到多视图架构中，借鉴了VGGT、π3和RayZer等模型的设计思路，但进行了改进以适应物体姿态估计任务。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是设计一个多视图前馈网络，直接处理查询图像和多个模板图像，以端到端方式预测物体姿态，消除显式的特征匹配步骤。同时，将物体几何信息引入多视图框架，使用显式的点图和通过几何表示网络学习的几何特征，并将这些特征投影回视图图表示。整体流程包括：1)渲染多视图RGB图像；2)编码图像和相机信息；3)提取几何特征；4)通过交叉注意力机制融合视觉和几何信息；5)预测物体姿态。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': "关键创新点包括：1)提出多视图前馈网络，直接处理查询图像和模板图像，消除显式特征匹配；2)引入物体几何信息到多视图框架，使用显式点图和学习的几何特征；3)将几何特征投影回视图图表示，提高推理能力；4)构建大规模合成数据集(190k+物体)。相比之前工作，不同之处在于：不同于传统的'匹配-定位'范式，消除了显式匹配；不同于典型多视图模型，整合了几何信息；不同于仅使用RGB图像的方法，提高了对物体姿态估计的适应性。", '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'PoseGAM通过整合几何感知的多视图推理与大规模合成数据集，实现了对未见物体姿态的鲁棒估计，显著提升了姿态估计的准确性和泛化能力。'}


### 论文摘要

6D object pose estimation, which predicts the transformation of an object relative to the camera, remains challenging for unseen objects. Existing approaches typically rely on explicitly constructing feature correspondences between the query image and either the object model or template images. In this work, we propose PoseGAM, a geometry-aware multi-view framework that directly predicts object pose from a query image and multiple template images, eliminating the need for explicit matching. Built upon recent multi-view-based foundation model architectures, the method integrates object geometry information through two complementary mechanisms: explicit point-based geometry and learned features from geometry representation networks. In addition, we construct a large-scale synthetic dataset containing more than 190k objects under diverse environmental conditions to enhance robustness and generalization. Extensive evaluations across multiple benchmarks demonstrate our state-of-the-art performance, yielding an average AR improvement of 5.1% over prior methods and achieving up to 17.6% gains on individual datasets, indicating strong generalization to unseen objects. Project page: https://windvchen.github.io/PoseGAM/ .

---

## 48. Graph Laplacian Transformer with Progressive Sampling for Prostate Cancer Grading

**论文链接:** [http://arxiv.org/abs/2512.10808v1](http://arxiv.org/abs/2512.10808v1)

**作者:** Masum Shah Junayed, John Derek Van Vessem, Qian Wan, Gahie Nam, Sheida Nabavi

**发布时间:** 2025-12-11

**DOI:** 10.1007/978-3-032-05162-2_35

**备注:** International Conference on Medical Image Computing and Computer-Assisted Intervention (MICCAI) 2025

### GPT解析

### 总结

该研究提出了一种基于图拉普拉斯注意力的Transformer (GLAT) 与迭代细化模块 (IRM) 相结合的方法，用于从全切片图像中前列腺癌分级，解决了现有方法中冗余区域选择的问题，提高了特征学习和空间一致性。

### 背景

前列腺癌从全切片图像(WSIs)分级面临挑战，原因包括WSIs的大规模特性、异质组织结构的存在以及选择诊断相关区域的困难。现有方法通常依赖随机或静态块选择，导致包含冗余或非信息性区域，从而降低性能。

### 目的

开发一种新型方法，增强前列腺癌分级中的特征学习和空间一致性，避免选择冗余或非信息性区域，提高诊断准确性。

### 方法

提出Graph Laplacian Attention-Based Transformer (GLAT)与Iterative Refinement Module (IRM)相结合的模型。IRM利用预训练的ResNet50进行局部特征提取，并使用无梯度模式的基础模型进行重要性评分。GLAT通过构建图模型组织级别的连接性，使用图拉普拉斯约束确保空间一致性，并通过可学习过滤机制细化特征表示。凸聚合适配机制动态调整块重要性以生成鲁棒的WSI级别表示。

### 主要发现

在五个公共和一个私人数据集上的广泛实验表明，该模型优于最先进的方法，实现了更高的性能和空间一致性，同时保持了计算效率。

### 结论

所提出的GLAT与IRM结合的方法有效解决了前列腺癌分级中的关键挑战，通过智能选择相关组织区域和建模组织连接性，显著提高了诊断准确性和空间一致性。

### 翻译

从全切片图像(WSIs)进行前列腺癌分级仍然是一项具有挑战性的任务，这归因于WSIs的大规模特性、异质组织结构的存在以及选择诊断相关区域的困难。现有方法通常依赖随机或静态块选择，导致包含冗余或非信息性区域，从而降低性能。为解决这一问题，我们提出了一种基于图拉普拉斯注意力的Transformer (GLAT)，并集成了迭代细化模块(IRM)，以增强特征学习和空间一致性。IRM利用预训练的ResNet50进行局部特征提取，并使用无梯度模式的基础模型进行重要性评分，确保只保留最相关的组织区域。GLAT通过构建图模型组织级别的连接性，其中块作为节点，使用图拉普拉斯约束确保空间一致性，并通过可学习过滤机制细化特征表示，增强判别性组织结构。此外，凸聚合适配机制动态调整块重要性，以生成鲁棒的WSI级别表示。在五个公共和一个私人数据集上的广泛实验表明，我们的模型优于最先进的方法，实现了更高的性能和空间一致性，同时保持了计算效率。


### 论文摘要

Prostate cancer grading from whole-slide images (WSIs) remains a challenging task due to the large-scale nature of WSIs, the presence of heterogeneous tissue structures, and difficulty of selecting diagnostically relevant regions. Existing approaches often rely on random or static patch selection, leading to the inclusion of redundant or non-informative regions that degrade performance. To address this, we propose a Graph Laplacian Attention-Based Transformer (GLAT) integrated with an Iterative Refinement Module (IRM) to enhance both feature learning and spatial consistency. The IRM iteratively refines patch selection by leveraging a pretrained ResNet50 for local feature extraction and a foundation model in no-gradient mode for importance scoring, ensuring only the most relevant tissue regions are preserved. The GLAT models tissue-level connectivity by constructing a graph where patches serve as nodes, ensuring spatial consistency through graph Laplacian constraints and refining feature representations via a learnable filtering mechanism that enhances discriminative histological structures. Additionally, a convex aggregation mechanism dynamically adjusts patch importance to generate a robust WSI-level representation. Extensive experiments on five public and one private dataset demonstrate that our model outperforms state-of-the-art methods, achieving higher performance and spatial consistency while maintaining computational efficiency.

---

## 49. Evaluating Gemini Robotics Policies in a Veo World Simulator

**论文链接:** [http://arxiv.org/abs/2512.10675v1](http://arxiv.org/abs/2512.10675v1)

**作者:** Gemini Robotics Team, Coline Devin, Yilun Du, Debidatta Dwibedi, Ruiqi Gao, Abhishek Jindal, Thomas Kipf, Sean Kirmani, Fangchen Liu, Anirudha Majumdar, Andrew Marmon, Carolina Parada, Yulia Rubanova, Dhruv Shah, Vikas Sindhwani, Jie Tan, Fei Xia, Ted Xiao, Sherry Yang, Wenhao Yu, Allan Zhou

**发布时间:** 2025-12-11

### GPT解析

### 总结

本研究介绍了一种基于前沿视频基础模型(Veo)的生成式评估系统，能够全面评估机器人政策在正常性能、分布外泛化及物理与语义安全性方面的表现。系统通过优化支持机器人动作条件和多视角一致性，并整合生成式图像编辑功能，能够合成真实世界场景的多种变化。

### 背景

生成世界模型在模拟与环境交互方面具有潜力，前沿视频模型能以可扩展和通用方式生成逼真的观察结果和环境交互。然而，视频模型在机器人领域的应用主要局限于与训练数据相似的分布内评估场景。

### 目的

展示视频模型可用于机器人政策评估的全谱系应用，从评估正常性能到分布外泛化和安全性探测；并介绍一个基于Veo模型的生成式评估系统，支持多视角一致性和场景变化合成。

### 方法

构建基于Veo视频模型的生成式评估系统，优化以支持机器人动作条件与多视角一致性，整合生成式图像编辑和多视角完成功能。通过1600多次真实世界评估，验证系统对八个Gemini Robotics政策检查点和双臂操作手五个任务的评估能力。

### 主要发现

系统保留了基础视频模型的能力，能准确模拟包含新颖交互对象、视觉背景和干扰对象的编辑场景。这种保真度使系统能准确预测不同政策在正常和分布外条件下的相对性能，确定不同泛化轴对政策性能的影响，并进行政策红队测试以发现安全隐患。

### 结论

视频模型可扩展应用于机器人政策评估的完整范围，所提出的生成式评估系统能够合成多种场景变化，为政策评估提供了强大工具，有助于全面评估机器人政策的安全性和可靠性。

### 翻译

生成世界模型在模拟与各种环境中的视觉运动政策交互方面具有巨大潜力。前沿的视频模型能够以可扩展和通用的方式生成逼真的观察结果和环境交互。然而，视频模型在机器人领域的应用主要局限于分布内评估，即与用于训练政策或微调基础视频模型的场景相似的场景。在本报告中，我们展示了视频模型可以用于机器人政策评估的所有用例：从评估正常性能到分布外泛化，以及探测物理和语义安全性。我们介绍了一个基于前沿视频基础模型的生成式评估系统，该系统经过优化以支持机器人动作条件和多视角一致性，同时整合生成式图像编辑功能，能够沿着多个泛化轴合成真实世界场景的逼真变化。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决如何全面评估通用机器人政策在正常、分布外场景以及安全性方面表现的问题。这个问题很重要，因为通用机器人政策能通过自然语言在各种环境中执行任务，但广泛评估其可靠性、泛化能力和安全性在现实中非常困难且成本高昂，传统物理模拟器又面临资产创建、准确模拟和视觉差距等挑战。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者认为前沿视频模型可以提供替代传统物理模拟器的解决方案，通过统一配方模拟各种资产及其复杂行为。他们基于Veo视频模型进行改进，添加了动作条件生成和多视角一致性功能。设计过程中借鉴了现有视频生成技术、机器人政策评估方法和生成图像编辑技术，但针对机器人评估的特殊需求进行了定制。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是使用视频生成模型作为通用评估工具，通过生成逼真的视频模拟预测机器人在不同场景中的行为，避免实际硬件测试。流程包括：1)基于Veo模型微调适应机器人任务；2)根据初始场景和机器人姿态生成视频帧；3)实现多视角一致性；4)使用图像编辑技术创建场景变化；5)运行政策并评估表现；6)将预测与实际测试结果比较验证。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)支持从正常性能到安全性的完整评估范围；2)确保多视角一致性；3)集成生成图像编辑技术创建多样化场景；4)实现动作条件生成；5)支持安全性红队测试。相比之前工作，这种方法超越了仅限于分布内评估的限制，无需大量手动调整专业知识，能自动创建多样化的测试场景，并全面测试政策的泛化能力和安全性。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '这篇论文贡献了一种基于视频生成模型的全面评估系统，能在无需实际硬件测试的情况下准确预测机器人政策在各种场景下的表现，大大提高了评估效率和覆盖范围。'}


### 论文摘要

Generative world models hold significant potential for simulating interactions with visuomotor policies in varied environments. Frontier video models can enable generation of realistic observations and environment interactions in a scalable and general manner. However, the use of video models in robotics has been limited primarily to in-distribution evaluations, i.e., scenarios that are similar to ones used to train the policy or fine-tune the base video model. In this report, we demonstrate that video models can be used for the entire spectrum of policy evaluation use cases in robotics: from assessing nominal performance to out-of-distribution (OOD) generalization, and probing physical and semantic safety. We introduce a generative evaluation system built upon a frontier video foundation model (Veo). The system is optimized to support robot action conditioning and multi-view consistency, while integrating generative image-editing and multi-view completion to synthesize realistic variations of real-world scenes along multiple axes of generalization. We demonstrate that the system preserves the base capabilities of the video model to enable accurate simulation of scenes that have been edited to include novel interaction objects, novel visual backgrounds, and novel distractor objects. This fidelity enables accurately predicting the relative performance of different policies in both nominal and OOD conditions, determining the relative impact of different axes of generalization on policy performance, and performing red teaming of policies to expose behaviors that violate physical or semantic safety constraints. We validate these capabilities through 1600+ real-world evaluations of eight Gemini Robotics policy checkpoints and five tasks for a bimanual manipulator.

---

## 50. Geo6DPose: Fast Zero-Shot 6D Object Pose Estimation via Geometry-Filtered Feature Matching

**论文链接:** [http://arxiv.org/abs/2512.10674v1](http://arxiv.org/abs/2512.10674v1)

**作者:** Javier Villena Toro, Mehdi Tarkian

**发布时间:** 2025-12-11

### GPT解析

### 总结

Geo6DPose是一种轻量级、完全本地化、无需训练的零样本6D物体姿态估计方法，通过结合基础模型视觉特征和几何过滤策略，实现了亚秒级推理性能，匹配了大规模模型的平均召回率。

### 背景

零样本6D物体姿态估计的最新进展主要由大规模模型和基于云的推理驱动，但这些方法引入高延迟、高能耗和部署风险，与实际机器人应用的约束相冲突。

### 目的

提出一种轻量级、完全本地化、无需训练的零样本6D姿态估计流水线，通过用几何可靠性替代模型规模来解决现有方法的问题。

### 方法

结合基础模型视觉特征和几何过滤策略，计算模板DINO描述符与场景块之间的相似度图，建立相互对应关系，通过基于对应关系的RANSAC恢复最终姿态，并使用加权几何对齐指标进行排序以提高鲁棒性。

### 主要发现

Geo6DPose在单个普通GPU上实现亚秒级推理(1.08 FPS)，匹配显著更大的零样本基线的平均召回率(53.7 AR)，无需训练、微调或网络访问，与不断发展的基础骨干模型保持兼容。

### 结论

Geo6DPose推进了机器人部署中实用、完全本地的6D感知。

### 翻译

零样本6D物体姿态估计的最新进展主要由大规模模型和基于云的推理驱动。然而，这些方法通常引入高延迟、高能耗以及与连接性、成本和数据治理相关的部署风险；这些因素与实际机器人应用的约束相冲突，其中计算资源有限且经常需要设备端推理。我们引入了Geo6DPose，一种轻量级、完全本地化、无需训练的零样本6D姿态估计流水线，它用几何可靠性替代了模型规模。我们的方法将基础模型视觉特征与几何过滤策略相结合：计算载入的模板DINO描述符与场景块之间的相似度图，并通过将场景块中心投影到3D并将模板描述符投影到物体模型坐标系来建立相互对应关系。通过基于对应关系的RANSAC恢复最终姿态，并使用加权几何对齐指标进行排序，该指标同时考虑重投影一致性和空间支持，提高了对噪声、杂乱和部分可见性的鲁棒性。Geo6DPose在单个普通GPU上实现亚秒级推理，同时匹配显著更大的零样本基线的平均召回率(53.7 AR, 1.08 FPS)。它无需训练、微调或网络访问，并且与不断发展的基础骨干模型保持兼容，推动了机器人部署中实用、完全本地的6D感知。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决零样本6D物体姿态估计中的速度和效率问题。现有方法大多依赖大规模模型和云端推理，导致高延迟、高能耗以及与连接性、成本和数据治理相关的部署风险。这对机器人应用尤为重要，因为机器人计算资源有限，需要设备端快速推理，且完全本地化不依赖网络连接。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先识别现有零样本方法的瓶颈——依赖大型预训练模型导致计算需求大。然后关注到训练免费管道的趋势，如FoundPose和FreeZe等使用基础模型描述符建立对应关系。最后设计简化流程，直接使用DINOv2描述符建立3D-3D对应关系，避免昂贵的粗略搜索。作者借鉴了DINOv2视觉特征、FoundPose的模板匹配策略、RANSAC姿态估计以及FreeZe结合几何和视觉特征的思想。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是将基础模型视觉特征与几何过滤相结合：在模板DINO描述符和场景块间计算相似度图，通过将场景块中心投影到3D并将模板描述符投影到物体模型坐标系建立对应关系，最后用对应驱动的RANSAC恢复姿态并使用加权几何对齐指标排序。整体流程分两阶段：1)离线登机阶段：从CAD模型渲染RGB-D模板，提取DINO描述符并记录3D坐标；2)推理阶段：处理分割掩码，提取特征，建立3D-3D对应关系，生成姿态假设并选择最佳姿态。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)准确度-速度权衡：比快速训练基线更高召回率且无需训练；2)效率：比其他训练免费方法更快推理；3)新颖定位：在准确度-速度帕累托前沿占据新区域；4)加权几何对齐误差(WAE)：新姿态评分方法。相比之前工作，Geo6DPose使用深度信息验证姿态，提供更可靠过滤；比GigaPose和PicoPose更准确且实时；比FreeZeV2更轻量级；直接建立3D-3D对应关系，绕过昂贵搜索阶段。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'Geo6DPose提出了一种轻量级、完全本地化且无需训练的零样本6D物体姿态估计方法，通过结合基础模型视觉特征与几何过滤策略，在保持准确度的同时实现了实时性能，为机器人应用中的高效6D感知开辟了新途径。'}


### 论文摘要

Recent progress in zero-shot 6D object pose estimation has been driven largely by large-scale models and cloud-based inference. However, these approaches often introduce high latency, elevated energy consumption, and deployment risks related to connectivity, cost, and data governance; factors that conflict with the practical constraints of real-world robotics, where compute is limited and on-device inference is frequently required. We introduce Geo6DPose, a lightweight, fully local, and training-free pipeline for zero-shot 6D pose estimation that trades model scale for geometric reliability. Our method combines foundation model visual features with a geometric filtering strategy: Similarity maps are computed between onboarded template DINO descriptors and scene patches, and mutual correspondences are established by projecting scene patch centers to 3D and template descriptors to the object model coordinate system. Final poses are recovered via correspondence-driven RANSAC and ranked using a weighted geometric alignment metric that jointly accounts for reprojection consistency and spatial support, improving robustness to noise, clutter, and partial visibility. Geo6DPose achieves sub-second inference on a single commodity GPU while matching the average recall of significantly larger zero-shot baselines (53.7 AR, 1.08 FPS). It requires no training, fine-tuning, or network access, and remains compatible with evolving foundation backbones, advancing practical, fully local 6D perception for robotic deployment.

---

## 51. RoboNeuron: A Modular Framework Linking Foundation Models and ROS for Embodied AI

**论文链接:** [http://arxiv.org/abs/2512.10394v1](http://arxiv.org/abs/2512.10394v1)

**作者:** Weifan Guan, Huasen Xi, Chenxiao Zhang, Aosheng Li, Qinghao Hu, Jian Cheng

**发布时间:** 2025-12-11

### GPT解析

### 总结

论文提出了RoboNeuron框架，一个用于具身智能的通用部署解决方案，通过深度集成大语言模型和视觉-语言-动作模型与机器人操作系统，解决了当前具身AI系统面临的关键工程障碍。

### 背景

当前具身AI系统面临严重的工程障碍，主要包括跨场景适应性差、模块间耦合僵硬以及推理加速碎片化等问题。

### 目的

克服现有具身AI系统的局限性，开发一个通用部署框架，提高跨场景适应性和组件灵活性，建立系统化的性能基准测试平台。

### 方法

提出RoboNeuron框架，深度集成大语言模型和视觉-语言-动作模型的认知能力与机器人操作系统的实时执行骨干；使用模型上下文协议作为语义桥梁，使大语言模型能够动态编排底层机器人工具；建立高度模块化的架构，利用ROS统一通信接口严格解耦感知、推理和控制；引入自动化工具将ROS消息转换为可调用的MCP函数简化开发。

### 主要发现

RoboNeuron显著增强了跨场景适应性和组件灵活性；建立了系统化的横向性能基准测试平台；为可扩展的实际应用奠定了坚实基础。

### 结论

RoboNeuron框架解决了当前具身AI系统的关键工程障碍，提供了一个可扩展、模块化和灵活的解决方案，为实际应用奠定了基础。

### 翻译

当前的具身AI系统面临严重的工程障碍，主要表现为跨场景适应性差、模块间耦合僵硬和推理加速碎片化。为了克服这些局限性，我们提出了RoboNeuron，一个用于具身智能的通用部署框架。RoboNeuron是首个深度集成大语言模型和视觉-语言-动作模型认知能力与机器人操作系统实时执行骨干的框架。我们利用模型上下文协议作为语义桥梁，使大语言模型能够动态编排底层机器人工具。该框架建立了高度模块化的架构，通过利用ROS的统一通信接口严格解耦感知、推理和控制。关键的是，我们引入了一个自动化工具，将ROS消息转换为可调用的MCP函数，显著简化了开发过程。RoboNeuron显著增强了跨场景适应性和组件灵活性，同时建立了系统化的横向性能基准测试平台，为可扩展的实际具身应用奠定了坚实基础。


### 论文摘要

Current embodied AI systems face severe engineering impediments, primarily characterized by poor cross-scenario adaptability, rigid inter-module coupling, and fragmented inference acceleration. To overcome these limitations, we propose RoboNeuron, a universal deployment framework for embodied intelligence. RoboNeuron is the first framework to deeply integrate the cognitive capabilities of Large Language Models (LLMs) and Vision-Language-Action (VLA) models with the real-time execution backbone of the Robot Operating System (ROS). We utilize the Model Context Protocol (MCP) as a semantic bridge, enabling the LLM to dynamically orchestrate underlying robotic tools. The framework establishes a highly modular architecture that strictly decouples sensing, reasoning, and control by leveraging ROS's unified communication interfaces. Crucially, we introduce an automated tool to translate ROS messages into callable MCP functions, significantly streamlining development. RoboNeuron significantly enhances cross-scenario adaptability and component flexibility, while establishing a systematic platform for horizontal performance benchmarking, laying a robust foundation for scalable real-world embodied applications.

---

## 52. Tool-Augmented Spatiotemporal Reasoning for Streamlining Video Question Answering Task

**论文链接:** [http://arxiv.org/abs/2512.10359v1](http://arxiv.org/abs/2512.10359v1)

**作者:** Sunqi Fan, Jiashuo Cui, Meng-Hao Guo, Shuojin Yang

**发布时间:** 2025-12-11

**备注:** Accepted by NeurIPS 2025 main track

### GPT解析

### 总结

本文提出了STAR时空推理框架和视频工具包，用于增强多模态大语言模型的时空推理能力，在视频问答任务中取得显著性能提升。

### 背景

视频问答任务是评估基础模型感知、理解和推理动态现实场景的关键平台，但现有多模态大语言模型难以同时建模视频帧内空间关系和时间演化的因果动态。

### 目的

增强MLLM的时空推理能力，确保工具数量和多样性之间的和谐，并优化工具调用顺序，避免工具链快捷问题。

### 方法

设计全面的视频工具包和STAR框架，战略性地调度时间和空间工具，逐步定位视频关键区域，使用轻量级工具增强GPT-4o模型。

### 主要发现

STAR框架在VideoMME上实现8.2%性能提升，在LongVideoBench上实现4.6%性能提升。

### 结论

视频工具包和STAR框架在构建自主和智能的视频分析助手方面迈出重要一步，代码已公开。

### 翻译

视频问答任务是评估基础模型是否能够有效感知、理解和推理动态现实场景的关键平台。然而，现有的多模态大语言模型难以同时建模视频帧内的空间关系和理解复杂且推理密集型视频问答任务中时间演化的因果动态。在这项工作中，我们为多模态大语言模型配备了一个全面且可扩展的视频工具包，以增强其时空推理能力，并确保工具数量和多样性之间的和谐。为了更好地控制工具调用顺序并避免工具链快捷方式问题，我们提出了一个时空推理框架，该框架战略性地调度时间和空间工具，从而逐步定位视频中的关键区域。我们的STAR框架使用轻量级工具增强了GPT-4o，在VideoMME上实现了8.2%的提升，在LongVideoBench上实现了4.6%的提升。我们相信，我们提出的视频工具包和STAR框架在构建自主和智能的视频分析助手方面迈出了重要一步。代码已在https://github.com/fansunqi/VideoTool上公开。


### 论文摘要

Video Question Answering (VideoQA) task serves as a critical playground for evaluating whether foundation models can effectively perceive, understand, and reason about dynamic real-world scenarios. However, existing Multimodal Large Language Models (MLLMs) struggle with simultaneously modeling spatial relationships within video frames and understanding the causal dynamics of temporal evolution on complex and reasoning-intensive VideoQA task. In this work, we equip MLLM with a comprehensive and extensible Video Toolkit, to enhance MLLM's spatiotemporal reasoning capabilities and ensure the harmony between the quantity and diversity of tools. To better control the tool invocation sequence and avoid toolchain shortcut issues, we propose a Spatiotemporal Reasoning Framework (STAR) that strategically schedules temporal and spatial tools, thereby progressively localizing the key area in the video. Our STAR framework enhances GPT-4o using lightweight tools, achieving an 8.2% gain on VideoMME and 4.6% on LongVideoBench. We believe that our proposed Video Toolkit and STAR framework make an important step towards building autonomous and intelligent video analysis assistants. The code is publicly available at https://github.com/fansunqi/VideoTool.

---

## 53. ConStruct: Structural Distillation of Foundation Models for Prototype-Based Weakly Supervised Histopathology Segmentation

**论文链接:** [http://arxiv.org/abs/2512.10316v1](http://arxiv.org/abs/2512.10316v1)

**作者:** Khang Le, Ha Thach, Anh M. Vu, Trang T. K. Vo, Han H. Huynh, David Yang, Minh H. N. Le, Thanh-Huy Nguyen, Akash Awasthi, Chandra Mohan, Zhu Han, Hien Van Nguyen

**发布时间:** 2025-12-11

### GPT解析

### 总结

本文提出了一种用于组织病理学图像的弱监督语义分割的原型学习框架，整合了CONCH的形态感知表示和SegFormer的多尺度结构线索，通过文本引导的语义对齐生成同时具有语义判别性和空间一致性的原型。

### 背景

弱监督语义分割在组织病理学中严重依赖分类骨干网络，但这些模型通常只能定位最具判别性的区域，难以捕获组织结构的完整空间范围。视觉语言模型如CONCH提供丰富的语义对齐，现代分割骨干网络如SegFormer保留细粒度空间线索，但结合这些互补优势在弱监督且没有密集注释的情况下仍具挑战性。

### 目的

提出一个原型学习框架，整合形态感知表示、多尺度结构线索和文本引导的语义对齐，生成同时具有语义判别性和空间一致性的原型，有效利用这些异构来源，产生高质量的伪掩码，无需像素级注释。

### 方法

1. 提出文本引导的原型初始化，结合病理描述生成更完整和语义准确的伪掩码；2. 引入结构蒸馏机制，将SegFormer的空间知识转移至原型学习中，保留细粒度形态模式和局部组织边界；3. 整合CONCH的形态感知表示、SegFormer的多尺度结构线索和文本引导的语义对齐。

### 主要发现

1. 无需像素级注释即可产生高质量伪掩码；2. 提高了定位完整性；3. 增强了不同组织类型间的语义一致性；4. 在BCSS-WSSS数据集上优于现有方法；5. 通过冻结基础模型骨干网络和轻量级可训练适配器保持了计算效率。

### 结论

提出的原型学习框架成功整合了视觉语言模型和现代分割骨干网络的互补优势，在弱监督条件下实现了高质量的语义分割，无需密集注释，同时保持计算效率。

### 翻译

组织病理学中的弱监督语义分割严重依赖分类骨干网络，但这些模型通常只能定位最具判别性的区域，难以捕获组织结构的完整空间范围。视觉语言模型如CONCH提供丰富的语义对齐和形态感知表示，而现代分割骨干网络如SegFormer保留细粒度的空间线索。然而，结合这些互补优势在弱监督且没有密集注释的情况下仍具挑战性。我们提出了一种用于组织病理学图像弱监督语义分割的原型学习框架，整合了来自CONCH的形态感知表示、来自SegFormer的多尺度结构线索和文本引导的语义对齐，生成同时具有语义判别性和空间一致性的原型。为了有效利用这些异构来源，我们引入了文本引导的原型初始化，结合病理描述生成更完整和语义准确的伪掩码。一种结构蒸馏机制将SegFormer的空间知识转移至原型学习中，在原型学习过程中保留细粒度形态模式和局部组织边界。我们的方法无需像素级注释即可产生高质量伪掩码，提高了定位完整性，并增强了不同组织类型间的语义一致性。在BCSS-WSSS数据集上的实验表明，我们的原型学习框架优于现有的WSSS方法，同时通过冻结基础模型骨干网络和轻量级可训练适配器保持了计算效率。


### 论文摘要

Weakly supervised semantic segmentation (WSSS) in histopathology relies heavily on classification backbones, yet these models often localize only the most discriminative regions and struggle to capture the full spatial extent of tissue structures. Vision-language models such as CONCH offer rich semantic alignment and morphology-aware representations, while modern segmentation backbones like SegFormer preserve fine-grained spatial cues. However, combining these complementary strengths remains challenging, especially under weak supervision and without dense annotations. We propose a prototype learning framework for WSSS in histopathological images that integrates morphology-aware representations from CONCH, multi-scale structural cues from SegFormer, and text-guided semantic alignment to produce prototypes that are simultaneously semantically discriminative and spatially coherent. To effectively leverage these heterogeneous sources, we introduce text-guided prototype initialization that incorporates pathology descriptions to generate more complete and semantically accurate pseudo-masks. A structural distillation mechanism transfers spatial knowledge from SegFormer to preserve fine-grained morphological patterns and local tissue boundaries during prototype learning. Our approach produces high-quality pseudo masks without pixel-level annotations, improves localization completeness, and enhances semantic consistency across tissue types. Experiments on BCSS-WSSS datasets demonstrate that our prototype learning framework outperforms existing WSSS methods while remaining computationally efficient through frozen foundation model backbones and lightweight trainable adapters.

---

## 54. The 2025 Foundation Model Transparency Index

**论文链接:** [http://arxiv.org/abs/2512.10169v1](http://arxiv.org/abs/2512.10169v1)

**作者:** Alexander Wan, Kevin Klyman, Sayash Kapoor, Nestor Maslej, Shayne Longpre, Betty Xiong, Percy Liang, Rishi Bommasani

**发布时间:** 2025-12-11

**备注:** Website: https://crfm.stanford.edu/fmti/December-2025/index.html

### GPT解析

### 总结

2025年基础模型透明度指数显示，基础模型开发者的透明度 practices 从2024年的58分下降到2025年的40分，表明透明度进展出现恶化。公司最不透明的是其训练数据、训练计算以及旗舰模型的部署后使用和影响。IBM作为例外获得95分，而xAI和Midjourney仅得14分。

### 背景

基础模型开发者是全球最重要的公司之一，随着这些公司影响力日益增强，其透明度实践如何演变成为重要问题。

### 目的

发布2025年基础模型透明度指数，这是年度工作的第三版，旨在描述和量化基础模型开发者的透明度实践。

### 方法

引入新的评估指标，涉及数据获取、使用数据和监控等方面，并首次对阿里巴巴、DeepSeek和xAI等公司进行评估。

### 主要发现

2024年报告显示透明度有所改善，但2025年发现这种进展已经恶化，平均分从58分降至40分；公司最不透明的是训练数据、训练计算及模型部署后使用情况；IBM得95分表现突出，xAI和Midjourney仅得14分；前沿模型论坛成员处于中等水平。

### 结论

随着全球政策制定者越来越多地要求透明度，该研究揭示了基础模型开发者的当前透明度状况，分析了政策变化可能带来的影响，并指出需要更积极的政策干预来解决关键信息赤字。

### 翻译

基础模型开发者是全球最重要的公司之一。随着这些公司变得越来越重要，它们的透明度实践如何演变？2025年基础模型透明度指数是年度工作的第三版，旨在描述和量化基础模型开发者的透明度。2025年FMTI引入了与数据获取、使用数据和监控相关的新指标，并首次对阿里巴巴、DeepSeek和xAI等公司进行评估。2024年FMTI报告显示透明度有所改善，但2025年FMTI发现这种进展已经恶化：平均分从2024年的100分中的58分下降到2025年的40分。公司对其训练数据、训练计算以及旗舰模型的部署后使用和影响最不透明。尽管有这一总体趋势，IBM表现突出，得分为95，而最低得分的xAI和Midjourney仅为14。我们在指数中评估的前沿模型论坛的五个成员处于中等水平：我们认为这些公司避免了低分数带来的声誉损害，但缺乏成为透明度领导者的激励。随着全球政策制定者越来越多地要求某些类型的透明度，这项工作揭示了基础模型开发者的当前透明度状况，分析了在新实施政策下透明度可能如何变化，并指出了需要更积极的政策干预来解决关键信息赤字。


### 论文摘要

Foundation model developers are among the world's most important companies. As these companies become increasingly consequential, how do their transparency practices evolve? The 2025 Foundation Model Transparency Index is the third edition of an annual effort to characterize and quantify the transparency of foundation model developers. The 2025 FMTI introduces new indicators related to data acquisition, usage data, and monitoring and evaluates companies like Alibaba, DeepSeek, and xAI for the first time. The 2024 FMTI reported that transparency was improving, but the 2025 FMTI finds this progress has deteriorated: the average score out of 100 fell from 58 in 2024 to 40 in 2025. Companies are most opaque about their training data and training compute as well as the post-deployment usage and impact of their flagship models. In spite of this general trend, IBM stands out as a positive outlier, scoring 95, in contrast to the lowest scorers, xAI and Midjourney, at just 14. The five members of the Frontier Model Forum we score end up in the middle of the Index: we posit that these companies avoid reputational harms from low scores but lack incentives to be transparency leaders. As policymakers around the world increasingly mandate certain types of transparency, this work reveals the current state of transparency for foundation model developers, how it may change given newly enacted policy, and where more aggressive policy interventions are necessary to address critical information deficits.

---

## 55. VocSim: A Training-free Benchmark for Zero-shot Content Identity in Single-source Audio

**论文链接:** [http://arxiv.org/abs/2512.10120v1](http://arxiv.org/abs/2512.10120v1)

**作者:** Maris Basha, Anja Zai, Sabine Stoll, Richard Hahnloser

**发布时间:** 2025-12-10

### GPT解析

### 总结

本文介绍了VocSim，一个无需训练的基准测试，用于评估音频表示模型的内在几何对齐能力。研究评估了各种基础模型在零样本设置下对相同事件的不同声学实例的映射能力，发现简单模型表现良好，但在低资源语音上存在泛化差距。

### 背景

通用音频表示旨在将同一事件的不同声学实例映射到相近的点，以在零样本设置下解决内容身份识别问题。与通过参数更新衡量适应性的监督分类基准不同，需要一种新方法来评估冻结嵌入的内在几何对齐。

### 目的

引入VocSim基准测试，探测冻结嵌入的内在几何对齐；评估音频表示模型在零样本设置下对内容身份的识别能力；标准化内在音频几何的评估。

### 方法

聚合来自19个语料库的12.5万个单源音频片段，涵盖人类语音、动物发声和环境声音；限制为单源音频以隔离内容表示与源分离的混淆因素；使用Precision@k评估局部纯度，使用全局分离率评估点级类别分离；通过经验排列基线的提升来校准GSR；评估包括冻结Whisper编码器特征在内的各种模型。

### 主要发现

简单流程（冻结Whisper编码器特征、时频池化和无标签PCA）在零样本设置下表现出色；VocSim揭示了一致的泛化差距；在低资源语音上，局部检索急剧下降；尽管性能仍优于随机，但几何结构崩溃，表明未能推广到未见过的语音结构；最佳嵌入预测鸟类感知相似性，改善生物声学分类，并在HEAR基准上达到最先进结果。

### 结论

测量的内在几何质量可以代理未列出的下游应用的实用性；发布数据、代码和公共排行榜以标准化内在音频几何评估。

### 翻译

通用音频表示旨在将同一事件的不同声学实例映射到相近的点，在零样本设置下解决内容身份问题。与通过参数更新衡量适应性的监督分类基准不同，我们引入了VocSim，这是一个无需训练的基准，用于探测冻结嵌入的内在几何对齐。VocSim聚合了来自19个语料库的12.5万个单源音频片段，涵盖人类语音、动物发声和环境声音。通过限制为单源音频，我们将内容表示与源分离的混淆因素隔离开。我们使用Precision@k评估局部纯度，使用全局分离率评估点级类别分离。为了校准GSR，我们报告了经验排列基线的提升。在各种基础模型中，一个简单的流程——冻结的Whisper编码器特征、时频池化和无标签PCA——产生了强大的零样本性能。然而，VocSim也揭示了一致的泛化差距。在盲目的低资源语音上，局部检索急剧下降。尽管性能仍明显优于随机，但绝对几何结构崩溃，表明未能推广到未见过的语音结构。作为外部验证，我们最佳的嵌入预测了鸟类感知相似性，改善了生物声学分类，并在HEAR基准上取得了最先进的结果。我们认为，此处测量的内在几何质量代理了未列出的下游应用的实用性。我们发布数据、代码和公共排行榜，以标准化内在音频几何的评估。


### 论文摘要

General-purpose audio representations aim to map acoustically variable instances of the same event to nearby points, resolving content identity in a zero-shot setting. Unlike supervised classification benchmarks that measure adaptability via parameter updates, we introduce VocSim, a training-free benchmark probing the intrinsic geometric alignment of frozen embeddings. VocSim aggregates 125k single-source clips from 19 corpora spanning human speech, animal vocalizations, and environmental sounds. By restricting to single-source audio, we isolate content representation from the confound of source separation. We evaluate embeddings using Precision@k for local purity and the Global Separation Rate (GSR) for point-wise class separation. To calibrate GSR, we report lift over an empirical permutation baseline. Across diverse foundation models, a simple pipeline, frozen Whisper encoder features, time-frequency pooling, and label-free PCA, yields strong zero-shot performance. However, VocSim also uncovers a consistent generalization gap. On blind, low-resource speech, local retrieval drops sharply. While performance remains statistically distinguishable from chance, the absolute geometric structure collapses, indicating a failure to generalize to unseen phonotactics. As external validation, our top embeddings predict avian perceptual similarity, improve bioacoustic classification, and achieve state-of-the-art results on the HEAR benchmark. We posit that the intrinsic geometric quality measured here proxies utility in unlisted downstream applications. We release data, code, and a public leaderboard to standardize the evaluation of intrinsic audio geometry.

---

## 56. Openpi Comet: Competition Solution For 2025 BEHAVIOR Challenge

**论文链接:** [http://arxiv.org/abs/2512.10071v1](http://arxiv.org/abs/2512.10071v1)

**作者:** Junjie Bai, Yu-Wei Chao, Qizhi Chen, Jinwei Gu, Moo Jin Kim, Zhaoshuo Li, Xuan Li, Tsung-Yi Lin, Ming-Yu Liu, Nic Ma, Kaichun Mo, Delin Qu, Shangkun Sun, Hongchi Xia, Fangyin Wei, Xiaohui Zeng

**发布时间:** 2025-12-10

**备注:** preprint

### GPT解析

### 总结

本文报告了作者在2025年BEHAVIOR挑战赛中的解决方案，获得第二名且显著优于其他提交方案。基于π_{0.5}模型，通过系统研究训练技术和数据的影响构建解决方案，展示了预训练和后训练阶段的扩展能力，并为具身AI社区提供实践经验和设计建议。

### 背景

BEHAVIOR挑战赛旨在严格追踪物理代理在模拟环境中解决长期任务的进展。BEHAVIOR-1K专注于日常家庭任务，这些任务在真实环境中引入长期移动操作挑战，试图弥合当前研究与真实世界以人为中心的应用之间的差距。

### 目的

通过参与2025年BEHAVIOR挑战赛，展示解决长期家庭任务的能力，并为具身AI社区提供有价值的实践经验和设计建议。

### 方法

基于π_{0.5}模型，通过系统研究训练技术和数据的影响构建解决方案。进行仔细的消融研究，评估预训练和后训练阶段的扩展能力。

### 主要发现

解决方案在挑战赛中获得非常接近第一名的第二名，显著优于其他提交方案。预训练和后训练阶段的扩展能力对提升竞争力至关重要。

### 结论

实践经验和设计建议能为具身人工智能社区在将强大基础模型适应到复杂具身场景时提供可操作的见解。

### 翻译

2025年BEHAVIOR挑战赛旨在严格追踪物理代理在模拟环境中解决长期任务的进展。BEHAVIOR-1K专注于人们最希望机器人协助的日常家庭任务，这些任务在真实环境中引入长期移动操作挑战，弥合了当前研究与真实世界以人为中心的应用之间的差距。本报告提出在2025年BEHAVIOR挑战赛中的解决方案，获得非常接近第一名的第二名，且显著优于其他提交方案。基于π_{0.5}，专注于通过研究训练技术和数据的影响来系统地构建解决方案。通过仔细的消融研究，展示了在预训练和后训练阶段中提升竞争力的扩展能力。总结了实践经验和设计建议，希望为具身人工智能社区在将强大基础模型适应到复杂具身场景时提供可操作的见解。


### 论文摘要

The 2025 BEHAVIOR Challenge is designed to rigorously track progress toward solving long-horizon tasks by physical agents in simulated environments. BEHAVIOR-1K focuses on everyday household tasks that people most want robots to assist with and these tasks introduce long-horizon mobile manipulation challenges in realistic settings, bridging the gap between current research and real-world, human-centric applications. This report presents our solution to the 2025 BEHAVIOR Challenge in a very close 2nd place and substantially outperforms the rest of the submissions. Building on $π_{0.5}$, we focus on systematically building our solution by studying the effects of training techniques and data. Through careful ablations, we show the scaling power in pre-training and post-training phases for competitive performance. We summarize our practical lessons and design recommendations that we hope will provide actionable insights for the broader embodied AI community when adapting powerful foundation models to complex embodied scenarios.

---

## 57. SimWorld-Robotics: Synthesizing Photorealistic and Dynamic Urban Environments for Multimodal Robot Navigation and Collaboration

**论文链接:** [http://arxiv.org/abs/2512.10046v1](http://arxiv.org/abs/2512.10046v1)

**作者:** Yan Zhuang, Jiawei Ren, Xiaokang Ye, Jianzhi Shen, Ruixuan Zhang, Tianai Yue, Muhammad Faayez, Xuhong He, Ziqiao Ma, Lianhui Qin, Zhiting Hu, Tianmin Shu

**发布时间:** 2025-12-10

**备注:** Conference: NeurIPS 2025 (main)

### GPT解析

### 总结

本文介绍了SimWorld-Robotics (SWR)仿真平台，用于在大规模、照片级真实感城市环境中实现具身AI，并构建了两个具有挑战性的机器人基准测试任务。

### 背景

基础模型的最新进展在开发通用机器人方面显示出有希望的结果，这些机器人可以在开放场景中根据多模态输入执行多样化任务。然而，当前的工作主要集中在室内、家庭场景。

### 目的

开发一个用于大规模、照片级真实感城市环境中具身AI的仿真平台，并构建两个具有挑战性的机器人基准测试任务，以全面评估机器人在真实场景中的关键能力。

### 方法

构建基于Unreal Engine 5的SWR仿真平台，能够程序化生成无限数量的照片级真实感城市场景，包含行人和交通系统等动态元素。支持多机器人控制和通信。利用这些功能构建了两个基准测试任务：多模态指令跟随任务和多智能体搜索任务。

### 主要发现

这两个新的基准测试在真实场景中全面评估了广泛的关键机器人能力，包括多模态指令理解、大型环境中的3D空间推理、与行人和交通的安全长距离导航、多机器人协作和基于通信的交流。实验结果表明，包括视觉语言模型在内的最先进模型在处理这些任务时存在困难，缺乏城市环境所需的稳健感知、推理和规划能力。

### 结论

SimWorld-Robotics平台为城市环境中的机器人研究提供了更真实、复杂和可扩展的仿真环境，而新的基准测试任务能够有效评估机器人在真实城市场景中的关键能力，并揭示了当前先进模型的局限性。

### 翻译

基础模型的最新进展在开发通用机器人方面显示出有希望的结果，这些机器人可以在开放场景中根据多模态输入执行多样化任务。然而，当前的工作主要集中在室内、家庭场景。在这项工作中，我们提出了SimWorld-Robotics (SWR)，一个用于大规模、照片级真实感城市环境中具身AI的仿真平台。SWR基于Unreal Engine 5构建，能够程序化生成无限数量的照片级真实感城市场景，其中包含行人和交通系统等动态元素，在真实感、复杂性和可扩展性上超越了先前的城市仿真。它还支持多机器人控制和通信。通过这些关键功能，我们构建了两个具有挑战性的机器人基准测试：(1)多模态指令跟随任务，机器人必须遵循视觉语言导航指令在行人和交通存在的情况下到达目的地；(2)多智能体搜索任务，两个机器人必须通信以合作定位并会合。与现有基准不同，这两个新的基准测试在真实场景中全面评估了广泛的关键机器人能力，包括多模态指令理解、大型环境中的3D空间推理、与行人和交通的安全长距离导航、多机器人协作以及基于通信的交流。我们的实验结果表明，包括视觉语言模型在内的最先进模型难以处理我们的任务，缺乏城市环境所需的稳健感知、推理和规划能力。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决为机器人导航和协作创建逼真、大规模、动态的城市环境模拟平台的问题。这个问题很重要，因为当前基础模型主要关注室内场景，而现有城市模拟器在真实性、复杂性和可扩展性方面存在局限，无法充分模拟真实城市环境的复杂性（如动态行人、交通系统等），限制了机器人在实际城市环境中的表现和能力评估。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先分析了现有机器人模拟器的局限性，指出它们主要关注室内或特定类型机器人。设计上借鉴了Unreal Engine 5的图形渲染能力，结合程序化城市生成技术，创建了支持多种智能体（机器人、行人、车辆）的异步控制框架。还借鉴了现有基础模型在机器人领域的应用经验，以及视觉-语言导航和多机器人协作研究，但将其扩展到更复杂、更逼真的城市环境中。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是创建一个逼真、大规模、动态的城市环境模拟平台，支持多种智能体的交互和协作。实现流程包括：1)程序化城市生成（道路、建筑物、街道元素和交通元素）；2)支持三种类型智能体（人类、车辆、机器人）并提供异步多智能体控制；3)实现逼真的行人和交通模拟；4)构建多模态导航和多机器人协作基准测试；5)创建大规模数据集SimWorld-20K用于训练和评估。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)基于Unreal Engine 5构建的逼真且可扩展的城市环境模拟；2)支持多种类型智能体（人类、车辆、机器人）的综合支持和26种不同动作空间；3)基于路点的逼真交通系统；4)两个新的基准测试（SIMWORLD-MMNAV和SIMWORLD-MRS）；5)大规模数据集SimWorld-20K。相比之前工作，不同之处在于提供了更逼真、更复杂的城市环境，支持多种智能体交互，通过程序化生成实现无限城市环境，并集成了导航、协作、通信等多种功能。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'SimWorld-Robotics论文贡献了一个逼真、可扩展的城市环境模拟平台，支持多种智能体交互和协作，并通过新的基准测试和数据集推动了机器人在复杂城市环境中的导航和协作能力发展。'}


### 论文摘要

Recent advances in foundation models have shown promising results in developing generalist robotics that can perform diverse tasks in open-ended scenarios given multimodal inputs. However, current work has been mainly focused on indoor, household scenarios. In this work, we present SimWorld-Robotics~(SWR), a simulation platform for embodied AI in large-scale, photorealistic urban environments. Built on Unreal Engine 5, SWR procedurally generates unlimited photorealistic urban scenes populated with dynamic elements such as pedestrians and traffic systems, surpassing prior urban simulations in realism, complexity, and scalability. It also supports multi-robot control and communication. With these key features, we build two challenging robot benchmarks: (1) a multimodal instruction-following task, where a robot must follow vision-language navigation instructions to reach a destination in the presence of pedestrians and traffic; and (2) a multi-agent search task, where two robots must communicate to cooperatively locate and meet each other. Unlike existing benchmarks, these two new benchmarks comprehensively evaluate a wide range of critical robot capacities in realistic scenarios, including (1) multimodal instructions grounding, (2) 3D spatial reasoning in large environments, (3) safe, long-range navigation with people and traffic, (4) multi-robot collaboration, and (5) grounded communication. Our experimental results demonstrate that state-of-the-art models, including vision-language models (VLMs), struggle with our tasks, lacking robust perception, reasoning, and planning abilities necessary for urban environments.

---

## 58. Towards Foundation Models with Native Multi-Agent Intelligence

**论文链接:** [http://arxiv.org/abs/2512.08743v2](http://arxiv.org/abs/2512.08743v2)

**作者:** Shuyue Hu, Haoyang Yan, Yiqun Zhang, Yang Chen, Dongzhan Zhou, Lei Bai

**发布时间:** 2025-12-09

### GPT解析

### 总结

本研究探讨了如何赋予基础模型原生多智能体智能，确定了四种核心能力，并指出强大的单智能体表现并不自动产生稳健的多智能体智能。

### 背景

基础模型正越来越多地扮演AI代理的'大脑'角色，最近的工作已经开始赋予其单智能体能力，如GUI交互或集成工具使用。

### 目的

探索赋予基础模型原生多智能体智能的下一个前沿方向，解决单智能体表现与多智能体智能之间的差距。

### 方法

通过对41个大语言模型进行广泛经验研究，分析基础模型在多智能体环境中的表现。

### 主要发现

强大的单智能体表现并不自动产生稳健的多智能体智能；基础模型在多智能体环境中需要具备理解、规划、高效通信和适应四种核心能力。

### 结论

需要通过数据集构建、评估、训练范式和安全考虑等方面的研究来构建具有原生多智能体智能的基础模型。

### 翻译

基础模型正越来越多地扮演AI代理的'大脑'角色。虽然最近的工作已经开始赋予基础模型原生单智能体能力——如GUI交互或集成工具使用——但我们认为下一个前沿是赋予基础模型原生多智能体智能。我们确定了基础模型在多智能体环境中的四种核心能力：理解、规划、高效通信和适应。与关于这些能力自发出现的假设相反，我们通过41个大语言模型的广泛经验证据表明，强大的单智能体表现并不自动产生稳健的多智能体智能。为了解决这一差距，我们概述了构建具有原生多智能体智能的基础模型的关键研究方向——涵盖数据集构建、评估、训练范式和安全考虑。


### 论文摘要

Foundation models (FMs) are increasingly assuming the role of the "brain" of AI agents. While recent efforts have begun to equip FMs with native single-agent abilities -- such as GUI interaction or integrated tool use -- we argue that the next frontier is endowing FMs with native multi-agent intelligence. We identify four core capabilities of FMs in multi-agent contexts: understanding, planning, efficient communication, and adaptation. Contrary to assumptions about the spontaneous emergence of such abilities, we provide extensive empirical evidence across 41 large language models showing that strong single-agent performance alone does not automatically yield robust multi-agent intelligence. To address this gap, we outline key research directions -- spanning dataset construction, evaluation, training paradigms, and safety considerations -- for building FMs with native multi-agent intelligence.

---

