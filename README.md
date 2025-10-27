# 今日论文推荐 - 2025-10-27

共 24 篇论文

---

## 1. Modest-Align: Data-Efficient Alignment for Vision-Language Models

**论文链接:** [http://arxiv.org/abs/2510.21606v1](http://arxiv.org/abs/2510.21606v1)

**作者:** Jiaxiang Liu, Yuan Wang, Jiawei Du, Joey Tianyi Zhou, Mingkun Xu, Zuozhu Liu

**发布时间:** 2025-10-24

### GPT解析

### 总结

Modest-Align是一个轻量级跨模态对齐框架，通过随机扰动和嵌入平滑两种策略解决资源受限场景下的过度自信问题，在保持高性能的同时大幅减少训练数据和计算资源需求。

### 背景

跨模态对齐旨在将异构模态映射到共享潜在空间，CLIP等模型通过大规模预训练获得强大识别能力，但在资源受限、数据有限或质量低的情况下，这些模型常因模糊或弱相关的图像-文本对而出现过度自信和性能下降。

### 目的

设计一个轻量级的对齐框架，提高在资源受限场景下的鲁棒性和效率，解决模型的过度自信问题。

### 方法

提出Modest-Align框架，采用两种互补策略：随机扰动引入受控噪声模拟不确定性，嵌入平滑校准嵌入空间中的相似度分布，共同减少过度自信并提高对噪声或弱对齐样本的性能。

### 主要发现

在多个基准数据集上的实验表明，Modest-Align在检索任务中优于最先进方法，使用超过100倍少的训练数据和600倍少的GPU时间达到与CLIP竞争的结果。

### 结论

Modest-Align为现实世界中资源受限的跨模态对齐问题提供了实用且可扩展的解决方案。

### 翻译

跨模态对齐旨在将异构模态映射到共享的潜在空间，如CLIP等模型所示，这些模型通过大规模图像-文本预训练获得强大的识别能力。然而，在资源受限、数据有限或质量低的环境中，由于模糊或弱相关的图像-文本对普遍存在，这些模型常常过度自信且性能下降。当前依赖单一正样本对的对比学习方法进一步加剧了这一问题，通过强化对不确定样本的过度自信。为应对这些挑战，我们提出了Modest-Align，一个为鲁棒性和效率而设计的轻量级对齐框架。我们的方法采用两种互补策略——随机扰动，引入受控噪声来模拟不确定性；以及嵌入平滑，校准嵌入空间中的相似度分布。这些机制共同减少过度自信并提高对噪声或弱对齐样本的性能。在多个基准数据集上的广泛实验表明，Modest-Align在检索任务中优于最先进方法，使用超过100倍少的训练数据和600倍少的GPU时间达到与CLIP竞争的结果。我们的方法为现实世界中资源受限的跨模态对齐问题提供了实用且可扩展的解决方案。


### 论文摘要

Cross-modal alignment aims to map heterogeneous modalities into a shared latent space, as exemplified by models like CLIP, which benefit from large-scale image-text pretraining for strong recognition capabilities. However, when operating in resource-constrained settings with limited or low-quality data, these models often suffer from overconfidence and degraded performance due to the prevalence of ambiguous or weakly correlated image-text pairs. Current contrastive learning approaches, which rely on single positive pairs, further exacerbate this issue by reinforcing overconfidence on uncertain samples. To address these challenges, we propose Modest-Align, a lightweight alignment framework designed for robustness and efficiency. Our approach leverages two complementary strategies -- Random Perturbation, which introduces controlled noise to simulate uncertainty, and Embedding Smoothing, which calibrates similarity distributions in the embedding space. These mechanisms collectively reduce overconfidence and improve performance on noisy or weakly aligned samples. Extensive experiments across multiple benchmark datasets demonstrate that Modest-Align outperforms state-of-the-art methods in retrieval tasks, achieving competitive results with over 100x less training data and 600x less GPU time than CLIP. Our method offers a practical and scalable solution for cross-modal alignment in real-world, low-resource scenarios.

---

## 2. Visual Diffusion Models are Geometric Solvers

**论文链接:** [http://arxiv.org/abs/2510.21697v1](http://arxiv.org/abs/2510.21697v1)

**作者:** Nir Goren, Shai Yehezkel, Omer Dahary, Andrey Voynov, Or Patashnik, Daniel Cohen-Or

**发布时间:** 2025-10-24

**备注:** Project page: https://kariander1.github.io/visual-geo-solver/

### GPT解析

### 总结

本研究展示了视觉扩散模型可以作为有效的几何求解器，通过在像素空间工作直接解决几何问题。研究者将这种方法应用于三个著名的几何难题：内接正方形问题、斯坦纳树问题和简单多边形问题。

### 背景

几何问题求解一直是挑战性问题，尤其是内接正方形问题（询问每个约旦曲线是否包含四个点形成正方形）等长期未解决的难题。前期工作需要专门的架构和领域特定的适应来将扩散应用于参数化几何表示。

### 目的

展示视觉扩散模型作为几何求解器的有效性，探索在像素空间直接推理几何问题的可能性，并开发一种通用框架来近似解决著名的几何难题。

### 方法

将每个问题实例视为图像，训练标准视觉扩散模型将高斯噪声转换为代表有效近似解的图像。模型学习将嘈杂的几何结构转换为正确配置，将几何推理重新表述为图像生成过程。

### 主要发现

视觉扩散模型能够有效解决内接正方形问题、斯坦纳树问题和简单多边形问题；模型能够将嘈杂的几何结构转换为正确配置；生成模型与几何问题解决之间存在联系；在图像空间操作提供了一种通用框架来近似解决著名难题。

### 结论

视觉扩散模型可以作为有效的几何求解器，在图像空间操作为近似解决著名难题提供了通用且实用的框架，为解决更广泛的挑战性几何任务开辟了新途径。

### 翻译

在本文中，我们展示了视觉扩散模型可以作为有效的几何求解器：它们通过在像素空间工作，能够直接推理几何问题。我们首先在内接正方形问题上证明了这一点，这是几何学中的一个长期未解决的问题，询问每个约旦曲线是否包含四个点形成正方形。然后我们将这种方法扩展到其他两个著名的难解几何问题：斯坦纳树问题和简单多边形问题。我们的方法将每个问题实例视为图像，并训练一个标准的视觉扩散模型，该模型将高斯噪声转换为代表有效近似解的图像，该解与精确解非常匹配。模型学习将嘈杂的几何结构转换为正确配置，有效地将几何推理重新表述为图像生成。与之前需要专门架构和领域特定适应的工作不同，我们使用在问题视觉表示上操作的标准视觉扩散模型。这种简单性突显了生成模型与几何问题解决之间令人惊讶的联系。除了这里研究的具体问题外，我们的结果指向一个更广泛的范式：在图像空间操作为近似解决著名难题提供了通用且实用的框架，并为解决更广泛的挑战性几何任务开辟了新途径。


### 论文摘要

In this paper we show that visual diffusion models can serve as effective geometric solvers: they can directly reason about geometric problems by working in pixel space. We first demonstrate this on the Inscribed Square Problem, a long-standing problem in geometry that asks whether every Jordan curve contains four points forming a square. We then extend the approach to two other well-known hard geometric problems: the Steiner Tree Problem and the Simple Polygon Problem.   Our method treats each problem instance as an image and trains a standard visual diffusion model that transforms Gaussian noise into an image representing a valid approximate solution that closely matches the exact one. The model learns to transform noisy geometric structures into correct configurations, effectively recasting geometric reasoning as image generation.   Unlike prior work that necessitates specialized architectures and domain-specific adaptations when applying diffusion to parametric geometric representations, we employ a standard visual diffusion model that operates on the visual representation of the problem. This simplicity highlights a surprising bridge between generative modeling and geometric problem solving. Beyond the specific problems studied here, our results point toward a broader paradigm: operating in image space provides a general and practical framework for approximating notoriously hard problems, and opens the door to tackling a far wider class of challenging geometric tasks.

---

## 3. OpenHype: Hyperbolic Embeddings for Hierarchical Open-Vocabulary Radiance Fields

**论文链接:** [http://arxiv.org/abs/2510.21441v1](http://arxiv.org/abs/2510.21441v1)

**作者:** Lisa Weijler, Sebastian Koch, Fabio Poiesi, Timo Ropinski, Pedro Hermosilla

**发布时间:** 2025-10-24

### GPT解析

### 总结

本研究提出OpenHype方法，使用连续双曲潜在空间表示3D场景层次结构，实现了更高效的3D场景理解，优于现有方法。

### 背景

建模3D对象和3D场景的内在层次结构对自主代理理解环境至关重要，但使用隐式表示如神经辐射场实现这一目标仍面临挑战。现有显式建模层次结构的方法存在局限性：要么需要多次渲染增加推理时间，要么依赖预定义的封闭集离散层次结构，难以泛化到真实世界的多样化结构。

### 目的

开发一种能有效表示3D场景层次结构的方法，解决现有方法在推理效率和泛化能力方面的局限性，实现对3D场景更全面、高效的理解。

### 方法

提出OpenHype，使用连续双曲潜在空间表示场景层次结构。利用双曲几何特性自然编码多尺度关系，通过潜在空间中的测地线路径实现层次结构的平滑遍历。

### 主要发现

OpenHype在标准基准测试中优于最先进的方法，展示了在3D场景理解方面卓越的效率和适应性。

### 结论

通过利用双曲几何性质，OpenHype提供了表示和探索3D场景层次结构的有效方式，解决了现有方法的效率和泛化局限性，为自主代理的环境理解提供了更强大工具。

### 翻译

建模3D对象和3D场景的内在层次结构是非常可取的，因为它能够使自主代理更全面地理解环境。使用隐式表示（如神经辐射场）来实现这一点仍然是一个未被探索的挑战。明确建模层次结构的现有方法通常面临显著限制：它们要么需要多次渲染传递来捕获不同粒度级别的嵌入，显著增加了推理时间；要么依赖于预定义的封闭集离散层次结构，难以泛化到代理在真实世界中遇到的多样化且细微的结构。为解决这些挑战，我们提出了OpenHype，一种使用连续双曲潜在空间表示场景层次结构的新方法。通过利用双曲几何的特性，OpenHype自然编码了多尺度关系，并能够通过潜在空间中的测地线路径实现层次结构的平滑遍历。我们的方法在标准基准测试中优于最先进的方法，展示了在3D场景理解方面卓越的效率和适应性。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决如何有效表示3D场景中的层次结构问题，特别是在使用神经辐射场(NeRF)等隐式表示时。这个问题很重要，因为理解3D场景的层次结构对自主代理全面理解环境至关重要，例如物体由多个部分组成，也可以在更高层次上语义分组，这种层次组织对于语义分割、场景重建和物体检测等应用非常关键。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者通过分析现有方法的局限性进行思考，发现现有方法要么需要多次渲染增加推理时间，要么依赖预定义的离散层次结构泛化能力差。他们借鉴了双曲几何的思想，因为双曲空间的指数扩展特性能够自然编码多尺度关系。方法借鉴了现有工作如使用CLIP提取语言特征、使用中性词减少噪声等，但创新性地将双曲几何应用于3D场景层次表示，实现了连续层次遍历。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是利用双曲空间的几何特性自然编码3D场景的层次结构，实现连续的多尺度关系表示。整体流程包括：1)双曲自编码器训练：将语言对齐特征转换为双曲空间表示，高层对象靠近原点，低层对象靠近边界；2)NeRF训练：监督模型预测双曲特征，使用双曲距离作为损失函数；3)层次遍历：通过沿测地线路径连续遍历层次，解码特征并计算与文本提示的相似度，使用softmax加权聚合结果。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': "关键创新点包括：1)首次在3D场景理解中应用双曲空间自然编码层次结构；2)实现连续层次遍历，只需一次渲染而非多次；3)提出特征外推技术解决多视图一致性问题；4)改进的softmax加权聚合方法处理复杂查询。相比之前工作，OpenHype无需预定义离散层次或多次渲染，能连续遍历层次结构，在处理组合查询时表现更好，解决了现有方法的'词袋效应'问题。", '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'OpenHype通过双曲空间几何特性实现了在神经辐射场中连续、高效地表示和遍历3D场景层次结构，显著提升了开放词汇3D场景理解能力。'}


### 论文摘要

Modeling the inherent hierarchical structure of 3D objects and 3D scenes is highly desirable, as it enables a more holistic understanding of environments for autonomous agents. Accomplishing this with implicit representations, such as Neural Radiance Fields, remains an unexplored challenge. Existing methods that explicitly model hierarchical structures often face significant limitations: they either require multiple rendering passes to capture embeddings at different levels of granularity, significantly increasing inference time, or rely on predefined, closed-set discrete hierarchies that generalize poorly to the diverse and nuanced structures encountered by agents in the real world. To address these challenges, we propose OpenHype, a novel approach that represents scene hierarchies using a continuous hyperbolic latent space. By leveraging the properties of hyperbolic geometry, OpenHype naturally encodes multi-scale relationships and enables smooth traversal of hierarchies through geodesic paths in latent space. Our method outperforms state-of-the-art approaches on standard benchmarks, demonstrating superior efficiency and adaptability in 3D scene understanding.

---

## 4. ZING-3D: Zero-shot Incremental 3D Scene Graphs via Vision-Language Models

**论文链接:** [http://arxiv.org/abs/2510.21069v1](http://arxiv.org/abs/2510.21069v1)

**作者:** Pranav Saxena, Jimmy Chiun

**发布时间:** 2025-10-24

### GPT解析

### 总结

ZING-3D是一个创新的框架，能够生成丰富语义表示的3D场景图，支持增量更新和几何基础，适用于机器人应用。

### 背景

理解和推理复杂的3D环境需要结构化的场景表示，捕获物体及其语义和空间关系。现有3D场景图生成工作利用预训练VLMs但存在局限：局限于单视图设置、不支持增量更新、缺乏3D空间几何基础。

### 目的

提出ZING-3D框架，利用预训练基础模型实现开放词汇识别，零样本生成丰富场景语义表示，支持增量更新和3D空间几何基础，适用于下游机器人应用。

### 方法

利用VLM推理生成丰富2D场景图，使用深度信息与3D空间关联；节点表示开放词汇对象(含特征、3D位置、语义上下文)，边捕获空间和语义关系(含对象间距离)。

### 主要发现

在Replica和HM3D数据集上的实验表明，ZING-3D能有效捕获空间和关系知识，无需特定任务训练。

### 结论

ZING-3D解决了现有方法的局限性，适用于下游机器人应用。

### 翻译

理解和推理复杂的3D环境需要结构化的场景表示，这些表示不仅要捕获物体，还要捕获它们的语义和空间关系。虽然最近关于3D场景图生成的工作利用了没有针对特定任务微调的预训练VLMs，但它们主要局限于单视图设置，无法支持随着新观察到来时的增量更新，并且缺乏在3D空间中的明确几何基础，所有这些对于具身场景都是必不可少的。在本文中，我们提出了ZING-3D框架，它利用预训练基础模型的丰富知识，实现开放词汇识别，并以零样本方式生成丰富的场景语义表示，同时支持在3D空间中进行增量更新和几何基础，使其适用于下游机器人应用。我们的方法利用VLM推理生成丰富的2D场景图，并使用深度信息将其与3D关联。节点表示具有特征、3D位置和语义上下文的开放词汇对象，而边捕获具有对象间距离的空间和语义关系。我们在来自Replica和HM3D数据集的场景上的实验表明，ZING-3D能够在无需特定任务训练的情况下有效地捕获空间和关系知识。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决现有3D场景图生成方法的三个限制：依赖特定词汇表、只能在单张2D图像上操作、无法增量更新。这个问题在现实中很重要，因为机器人需要在动态环境中在线构建和更新对3D环境的理解，而现有方法无法处理现实世界中的新物体或关系，也无法捕捉不同视角间的空间一致性。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者从嵌入式AI代理在环境中探索的角度设计系统，考虑如何随着新观测的加入逐步构建场景理解。他们借鉴了Vision-Language Models(VLMs)的进步，特别是像Open-World SGG和Pixels-to-Graphs等利用预训练模型进行零样本关系推理的工作。同时，他们结合了深度信息实现3D几何定位，并使用Grounded-SAM2提供精确的物体分割掩码，将2D场景表示提升到3D空间。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是利用预训练视觉-语言模型的广泛知识，实现零样本开放词汇表识别，同时支持场景图的增量更新和3D空间中的几何定位。整体流程包括：1)使用VLM进行开放词汇表物体检测；2)构建2D场景图，捕获物体间的空间和语义关系；3)使用Grounded-SAM2获取精确分割掩码，结合深度信息将物体投影到3D空间；4)随着机器人探索，增量更新全局3D场景图；5)根据导航任务需求进行场景图剪枝。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)零样本嵌入式场景图生成，无需任务特定训练；2)丰富的语义信息，节点包含物体特征、3D位置和房间类型，边表示精确的空间关系；3)支持增量更新，场景图随探索过程动态演进。相比之前工作，ZING-3D的独特之处在于结合了2D视觉推理与3D几何信息，支持增量更新，实现了真正的开放词汇表识别，并提供了任务导向的场景图剪枝功能，更适合机器人实际应用。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'ZING-3D通过结合视觉-语言模型与3D几何信息，实现了零样本增量式3D场景图生成，为机器人在复杂环境中的导航和交互提供了结构化的语义-空间表示。'}


### 论文摘要

Understanding and reasoning about complex 3D environments requires structured scene representations that capture not only objects but also their semantic and spatial relationships. While recent works on 3D scene graph generation have leveraged pretrained VLMs without task-specific fine-tuning, they are largely confined to single-view settings, fail to support incremental updates as new observations arrive and lack explicit geometric grounding in 3D space, all of which are essential for embodied scenarios. In this paper, we propose, ZING-3D, a framework that leverages the vast knowledge of pretrained foundation models to enable open-vocabulary recognition and generate a rich semantic representation of the scene in a zero-shot manner while also enabling incremental updates and geometric grounding in 3D space, making it suitable for downstream robotics applications. Our approach leverages VLM reasoning to generate a rich 2D scene graph, which is grounded in 3D using depth information. Nodes represent open-vocabulary objects with features, 3D locations, and semantic context, while edges capture spatial and semantic relations with inter-object distances. Our experiments on scenes from the Replica and HM3D dataset show that ZING-3D is effective at capturing spatial and relational knowledge without the need of task-specific training.

---

## 5. Stuck in the Matrix: Probing Spatial Reasoning in Large Language Models

**论文链接:** [http://arxiv.org/abs/2510.20198v1](http://arxiv.org/abs/2510.20198v1)

**作者:** Maggie Bai, Ava Kim Cohen, Eleanor Koss, Charlie Lichtenbaum

**发布时间:** 2025-10-23

**备注:** 20 pages, 24 figures

### GPT解析

### 总结

本研究通过五项任务测试了大型语言模型在文本输入上的空间推理能力，发现模型在简单任务中表现中等，但随着复杂性和规模增加，性能显著下降，平均准确率损失42.7%，最高达84%，表明LLMs在空间推理方面存在明显局限性。

### 背景

大型语言模型在自然语言处理方面表现出色，但其空间推理能力尚未被充分研究。

### 目的

探究大型语言模型在文本输入上的空间理解和计算能力，识别其在空间推理方面的优势和局限。

### 方法

设计并实施了五项空间推理任务：象限识别、几何变换、距离评估、单词搜索和滑块拼图。这些任务在结构化网格环境中进行，通过增加网格尺寸来提高复杂性，要求模型从简单模式识别扩展到抽象空间推理。

### 主要发现

1) 模型在复杂性和规模较小的任务中表现中等；2) 随着规模增加，性能迅速下降，准确率平均损失42.7%，最高达84%；3) 所有初始准确率超过50%的测试显示至少48%的性能损失；4) 模型在扩展复杂性方面的挣扎暗示其底层架构中缺乏强大的空间表示。

### 结论

大型语言模型在语言推理和空间推理之间存在明显差距，本研究揭示了其当前局限性，并为未来在语言和几何交叉领域的集成基准研究奠定了基础。

### 翻译

本文通过一套五项任务，探究了大型语言模型对文本输入的空间推理能力，旨在测试它们的空间理解和计算能力。模型在基于结构化网格环境中的基本空间推理和多步问题解决方面接受了测试，使用了象限识别、几何变换、距离评估、单词搜索和滑块拼图等任务。每个任务通过增加网格尺寸来提高复杂性，要求模型超越简单的模式识别，进入抽象空间推理。我们的结果显示，虽然大型语言模型在复杂性和规模较小的所有任务中表现出中等成功，但随着规模增加，性能迅速下降，准确率平均损失42.7%，最高达到84%。所有初始准确率超过50%的测试都显示出至少48%的损失，说明了性能下降的一致性。此外，模型在扩展复杂性方面的挣扎暗示其底层架构中缺乏强大的空间表示。本文强调了大型语言模型中语言推理和空间推理之间的差距，提供了对其当前局限性的见解，并为未来在语言和几何交叉领域的集成基准研究奠定了基础。


### 论文摘要

This paper explores the spatial reasoning capability of large language models (LLMs) over textual input through a suite of five tasks aimed at probing their spatial understanding and computational abilities. The models were tested on both fundamental spatial reasoning and multi-step problem-solving within structured grid-based environments using tasks such as quadrant identification, geometric transformations, distance evaluation, word searches, and tile sliding. Each task was scaled in complexity through increasing grid dimensions, requiring models to extend beyond simple pattern recognition into abstract spatial reasoning. Our results reveal that while LLMs demonstrate moderate success in all tasks with small complexity and size, performance drops off rapidly as scale increases, with an average loss in accuracy of 42.7%, and reaching as high as 84%. Every test that began with over 50% accuracy showed a loss of at least 48%, illustrating the consistent nature of the deterioration. Furthermore, their struggles with scaling complexity hint at a lack of robust spatial representations in their underlying architectures. This paper underscores the gap between linguistic and spatial reasoning in LLMs, offering insights into their current limitations, and laying the groundwork for future integrative benchmarks at the intersection of language and geometry.

---

## 6. Uncertainty evaluation of segmentation models for Earth observation

**论文链接:** [http://arxiv.org/abs/2510.19586v1](http://arxiv.org/abs/2510.19586v1)

**作者:** Melanie Rey, Andriy Mnih, Maxim Neumann, Matt Overlan, Drew Purves

**发布时间:** 2025-10-22

### GPT解析

### 总结

本文研究了从卫星影像中估计语义分割预测不确定性的方法，针对遥感地球观测应用对现有方法进行了基准测试，评估了不确定性度量的实际效用，并提出了实用建议。

### 背景

与标准图像分类相比，分割中的不确定性估计面临独特挑战，需要可扩展的方法来生成逐像素估计。大多数相关研究集中在场景理解或医学影像领域。

### 目的

专门针对遥感地球观测应用对不确定性估计方法进行基准测试，评估不确定性度量的实际效用，测试它们识别预测错误和噪声损坏的输入图像区域的能力。

### 方法

在两个遥感数据集PASTIS和ForTy上进行实验，这些数据集在规模、地理覆盖范围和标签置信度方面存在差异。评估包括多种模型（如随机分割网络和集成方法）与多种神经网络架构和不确定性度量相结合的广泛评估。

### 主要发现

通过实验评估了不同不确定性估计方法在遥感应用中的表现，确定了哪些方法更适合识别预测错误和噪声损坏区域。

### 结论

基于研究结果提出了若干实用建议，为遥感影像语义分割中的不确定性估计提供了指导。

### 翻译

本文研究了从卫星影像中估计语义分割预测不确定性的方法。与标准图像分类相比，分割中的不确定性估计面临独特挑战，需要可扩展的方法来生成逐像素估计。虽然大多数关于此主题的研究集中在场景理解或医学影像上，但这项工作专门针对遥感地球观测应用对现有方法进行了基准测试。我们的评估侧重于不确定性度量的实际效用，测试它们识别预测错误和噪声损坏的输入图像区域的能力。实验在两个遥感数据集PASTIS和ForTy上进行，这两个数据集在规模、地理覆盖范围和标签置信度方面存在差异。我们进行了广泛的评估，结合了多种模型（如随机分割网络和集成方法）与多种神经网络架构和不确定性度量。根据我们的发现，我们提出了若干实用建议。


### 论文摘要

This paper investigates methods for estimating uncertainty in semantic segmentation predictions derived from satellite imagery. Estimating uncertainty for segmentation presents unique challenges compared to standard image classification, requiring scalable methods producing per-pixel estimates. While most research on this topic has focused on scene understanding or medical imaging, this work benchmarks existing methods specifically for remote sensing and Earth observation applications. Our evaluation focuses on the practical utility of uncertainty measures, testing their ability to identify prediction errors and noise-corrupted input image regions. Experiments are conducted on two remote sensing datasets, PASTIS and ForTy, selected for their differences in scale, geographic coverage, and label confidence. We perform an extensive evaluation featuring several models, such as Stochastic Segmentation Networks and ensembles, in combination with a number of neural architectures and uncertainty metrics. We make a number of practical recommendations based on our findings.

---

## 7. Seeing Across Views: Benchmarking Spatial Reasoning of Vision-Language Models in Robotic Scenes

**论文链接:** [http://arxiv.org/abs/2510.19400v1](http://arxiv.org/abs/2510.19400v1)

**作者:** Zhiyuan Feng, Zhaolu Kang, Qijie Wang, Zhiying Du, Jiongrui Yan, Shubin Shi, Chengbo Yuan, Huizhi Liang, Yu Deng, Qixiu Li, Rushuai Yang, Arctanx An, Leqi Zheng, Weijie Wang, Shawn Chen, Sicheng Xu, Yaobo Liang, Jiaolong Yang, Baining Guo

**发布时间:** 2025-10-22

**备注:** The project and benchmark are publicly available at  https://github.com/microsoft/MV-RoboBench

### GPT解析

### 总结

本文提出MV-RoboBench基准测试，用于评估视觉语言模型在机器人操作中的多视图空间推理能力。研究显示当前最先进模型表现远低于人类水平，并发现空间智能与机器人任务执行呈正相关，但单视图基准表现不能可靠预测多视图机器人任务表现。

### 背景

视觉语言模型对具身人工智能至关重要，是视觉语言动作模型的基础。然而大多数VLM评估集中在单视图设置，对多视图信息整合能力的探索不足。多摄像头设置在机器人平台上越来越标准，能提供互补视角以缓解遮挡和深度模糊问题。

### 目的

填补VLMs多视图空间推理能力评估的空白，专门设计一个基准测试来评估VLMs在机器人操作中的多视图空间推理能力。

### 方法

创建MV-RoboBench基准测试，包含8个子任务中的1.7k个手动筛选的问答项目，分为空间理解和机器人执行两个主要类别。评估多种现有VLMs（包括开源和闭源模型）以及采用CoT启发技术的增强版本。

### 主要发现

(i)在多视图机器人场景中，空间智能和机器人任务执行呈正相关；(ii)在现有通用单视图空间理解基准上的良好表现并不能可靠地转化为在机器人空间任务中的成功。

### 结论

当前最先进的VLMs在多视图机器人感知方面仍面临重大挑战。作者发布MV-RoboBench作为开放资源，旨在促进空间感知VLMs和VLAs的进步，提供数据和多视图具身推理的标准化评估协议。

### 翻译

视觉语言模型对具身人工智能至关重要，使机器人能够感知、推理并在复杂环境中行动。它们也是最近视觉语言动作模型的基础。然而，大多数VLM评估集中在单视图设置，对其整合多视图信息的能力探索不足。与此同时，多摄像头设置在机器人平台上越来越标准，因为它们提供互补视角以缓解遮挡和深度模糊问题。因此，VLMs是否能有效利用此类多视图输入进行机器人推理仍然是一个开放问题。为填补这一空白，我们引入MV-RoboBench，一个专门设计用于评估VLMs在机器人操作中多视图空间推理能力的基准测试。MV-RoboBench包含8个子任务中的1.7k个手动筛选的问答项目，分为两个主要类别：空间理解和机器人执行。我们评估了多种现有的VLMs，包括开源和闭源模型，以及采用CoT启发技术的增强版本。结果显示，最先进的模型表现远低于人类水平，突显了VLMs在多视图机器人感知方面面临的重大挑战。此外，我们的分析揭示了两个关键发现：(i)在多视图机器人场景中，空间智能和机器人任务执行呈正相关；(ii)在现有通用单视图空间理解基准上的良好表现并不能可靠地转化为在我们基准评估的机器人空间任务中的成功。我们发布MV-RoboBench作为开放资源，旨在促进空间感知VLMs和VLAs的进步，不仅提供数据，还提供多视图具身推理的标准化评估协议。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决的问题是评估视觉语言模型（VLMs）在机器人场景中的多视图空间推理能力。这个问题很重要，因为现有的VLM评估大多集中在单视图设置，而机器人平台越来越多地采用多摄像头系统来提供互补视角以克服遮挡和深度模糊问题。理解VLM能否有效整合这些多视图信息对提升机器人在复杂环境中的感知和决策能力至关重要，也是实现先进视觉语言动作（VLA）模型的基础。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者通过分析现有空间推理基准的局限性，发现它们大多专注于单视图数据或非具身任务，而机器人操作场景需要多视图感知能力。他们借鉴了ShareRobot（具身机器人任务但无多视图）、All-Angles Bench和Ego3D-Bench（多视图但仅限导航或照片对齐）等工作，设计了MV-RoboBench，一个专门针对机器人操作场景中多视图空间推理的基准。作者构建了多阶段管道：数据收集（从AgiWorld和BridgeV2数据集筛选）、问答生成（为八个子任务设计模板）和人工质保审查，确保基准质量和多样性。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是评估VLMs能否有效整合多个摄像头视图的互补信息，支持机器人在现实世界中的决策。基准包含1700多个人工策划的问答项目，分为空间理解（跨视图匹配、距离判断、视角识别、3D空间一致性）和机器人执行（动作规划、步骤执行、轨迹选择、功能识别）两大类。实现流程包括：1)数据收集（规则过滤+GPT-4.1辅助筛选+人工验证）；2)问答生成（任务特定模板+五选一问答对构建）；3)人工质保审查（迭代审查+内容修正+答案分布平衡）；4)模型评估（统一零样本提示+准确率作为指标）。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)首个整合空间和机器人推理与多视图输入的机器人操作基准；2)系统性评估VLMs整合多视图信息的能力；3)发现空间智能与机器人执行在多视图场景中正相关；4)揭示单视图基准性能不能可靠转移到多视图机器人任务。相比之前工作，MV-RoboBench专注于具身多视图推理而非抽象任务；使用真实机器人演示而非模板生成；同时评估空间理解和机器人执行；强调多视图互补信息整合而非单一视角分析。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'MV-RoboBench是首个专门针对机器人操作场景中多视图空间推理能力的基准测试，通过系统评估现有视觉语言模型的表现，揭示了它们在整合多视图信息进行机器人决策方面的显著不足，并为未来具身人工智能和多视图感知研究提供了新的评估标准。'}


### 论文摘要

Vision-language models (VLMs) are essential to Embodied AI, enabling robots to perceive, reason, and act in complex environments. They also serve as the foundation for the recent Vision-Language-Action (VLA) models. Yet most evaluations of VLMs focus on single-view settings, leaving their ability to integrate multi-view information underexplored. At the same time, multi-camera setups are increasingly standard in robotic platforms, as they provide complementary perspectives to mitigate occlusion and depth ambiguity. Whether VLMs can effectively leverage such multi-view inputs for robotic reasoning therefore remains an open question. To bridge this gap, we introduce MV-RoboBench, a benchmark specifically designed to evaluate the multi-view spatial reasoning capabilities of VLMs in robotic manipulation. MV-RoboBench consists of 1.7k manually curated QA items across eight subtasks, divided into two primary categories: spatial understanding and robotic execution. We evaluate a diverse set of existing VLMs, including both open-source and closed-source models, along with enhanced versions incorporating CoT-inspired techniques. The results show that state-of-the-art models remain far below human performance, underscoring the substantial challenges VLMs face in multi-view robotic perception. Additionally, our analysis uncovers two key findings: (i) spatial intelligence and robotic task execution are positively correlated in multi-view robotic scenarios; and (ii) strong performance on existing general-purpose single-view spatial understanding benchmarks does not reliably translate to success in the robotic spatial tasks assessed by our benchmark. We release MV-RoboBench as an open resource to foster progress in spatially grounded VLMs and VLAs, providing not only data but also a standardized evaluation protocol for multi-view embodied reasoning.

---

## 8. Exploring Scale Shift in Crowd Localization under the Context of Domain Generalization

**论文链接:** [http://arxiv.org/abs/2510.19330v1](http://arxiv.org/abs/2510.19330v1)

**作者:** Juncheng Wang, Lei Shang, Ziqi Liu, Wang Lu, Xixu Hu, Zhe Hu, Jindong Wang, Shujun Wang

**发布时间:** 2025-10-22

### GPT解析

### 总结

该研究探讨了人群定位中的尺度偏移问题及其在域泛化场景下的影响，提出了Catto算法来减轻尺度偏移的影响，并建立了ScaleBench基准测试。

### 背景

人群定位在视觉场景理解中扮演关键角色，但现有方法因训练和测试数据之间头部尺度分布差异（尺度偏移）导致性能显著下降，这一问题被称为域泛化挑战。

### 目的

理解人群定位模型在域泛化背景下尺度偏移的本质，解决四个关键问题：尺度偏移如何影响人群定位、如何量化这种影响、产生原因以及如何减轻影响。

### 方法

系统检查不同尺度偏移水平下人群定位性能变化；建立ScaleBench基准测试，重现20种先进域泛化算法；提供尺度偏移的严格理论分析；提出因果特征分解和各向异性处理（Catto）算法。

### 主要发现

通过实验展示了现有算法的局限性；强调了尺度偏移的重要性和复杂性；提供了四个对未来研究有重要意义的见解。

### 结论

强调了'尺度偏移域泛化'这一新颖且适用的研究方向的重要性。

### 翻译

人群定位在视觉场景理解中扮演关键角色，用于预测人群中每个行人的位置，因此适用于各种下游任务。然而，由于训练和测试数据之间头部尺度分布的差异（尺度偏移），现有方法性能显著下降，这一挑战被称为域泛化（DG）。本文旨在理解在人群定位模型的域泛化背景下尺度偏移的本质。为此，我们解决了四个关键问题：(i) 尺度偏移如何在域泛化场景中影响人群定位？(ii) 如何量化这种影响？(iii) 产生这种影响的原因是什么？(iv) 如何减轻这种影响？首先，我们系统地检查了人群定位性能如何随不同水平的尺度偏移而变化。然后，我们建立了一个基准ScaleBench，重现了20种先进的域泛化算法来量化这种影响。通过大量实验，我们展示了现有算法的局限性，并强调了尺度偏移的重要性和复杂性，这是一个尚未充分探索的主题。为了加深理解，我们对尺度偏移提供了严格的理论分析。基于这些见解，我们进一步提出了一种名为因果特征分解和各向异性处理（Catto）的有效算法，以减轻域泛化设置中尺度偏移的影响。随后，我们还提供了大量的分析实验，揭示了四个对未来研究有重要意义的见解。我们的结果强调了这一新颖且适用的研究方向的重要性，我们称之为尺度偏移域泛化。


### 论文摘要

Crowd localization plays a crucial role in visual scene understanding towards predicting each pedestrian location in a crowd, thus being applicable to various downstream tasks. However, existing approaches suffer from significant performance degradation due to discrepancies in head scale distributions (scale shift) between training and testing data, a challenge known as domain generalization (DG). This paper aims to comprehend the nature of scale shift within the context of domain generalization for crowd localization models. To this end, we address four critical questions: (i) How does scale shift influence crowd localization in a DG scenario? (ii) How can we quantify this influence? (iii) What causes this influence? (iv) How to mitigate the influence? Initially, we conduct a systematic examination of how crowd localization performance varies with different levels of scale shift. Then, we establish a benchmark, ScaleBench, and reproduce 20 advanced DG algorithms to quantify the influence. Through extensive experiments, we demonstrate the limitations of existing algorithms and underscore the importance and complexity of scale shift, a topic that remains insufficiently explored. To deepen our understanding, we provide a rigorous theoretical analysis on scale shift. Building on these insights, we further propose an effective algorithm called Causal Feature Decomposition and Anisotropic Processing (Catto) to mitigate the influence of scale shift in DG settings. Later, we also provide extensive analytical experiments, revealing four significant insights for future research. Our results emphasize the importance of this novel and applicable research direction, which we term Scale Shift Domain Generalization.

---

## 9. MoTVLA: A Vision-Language-Action Model with Unified Fast-Slow Reasoning

**论文链接:** [http://arxiv.org/abs/2510.18337v3](http://arxiv.org/abs/2510.18337v3)

**作者:** Wenhui Huang, Changhe Chen, Han Qi, Chen Lv, Yilun Du, Heng Yang

**发布时间:** 2025-10-21

### GPT解析

### 总结

MoTVLA是一种基于混合变压器的视觉-语言-动作模型，结合了快速-慢速统一推理与行为策略学习，解决了现有方法中语言控制能力有限和推理延迟显著的问题。

### 背景

将视觉语言指令整合到视觉运动策略中是增强机器人开放世界泛化能力的热门研究方向。

### 目的

开发一种能够平衡语言控制能力和执行效率的模型，解决现有方法中的两个主要挑战。

### 方法

MoTVLA模型保留了预训练视觉语言模型的通用智能，同时引入一个与预训练模型共享知识的领域专家transformer，生成领域特定的快速推理，并将动作专家基于分解的运动指令进行条件化。

### 主要发现

通过广泛评估，MoTVLA在快速-慢速推理和操作任务性能方面表现出优越性，能够学习多样化行为并显著提高语言控制能力。

### 结论

MoTVLA成功整合了快速-慢速统一推理与行为策略学习，有效解决了现有方法中的局限性，为机器人学习提供了新的解决方案。

### 翻译

将视觉语言指令整合到视觉运动策略中正在增强机器人学习的开放世界泛化能力方面获得动力。尽管有 promising 的进展，现有方法面临两个挑战：在不使用生成推理作为条件时，语言控制能力有限，或者在整合推理时推理延迟显著。在这项工作中，我们引入了MoTVLA，一种基于混合变压器(MoT)的视觉-语言-动作(VLA)模型，它整合了快速-慢速统一推理与行为策略学习。MoTVLA保留了预训练VLMs的通用智能（作为通用者）用于感知、场景理解和语义规划等任务，同时整合了一个领域专家（第二个transformer），它与预训练VLM共享知识，以生成领域特定的快速推理（例如机器人运动分解），从而提高策略执行效率。通过将动作专家基于分解的运动指令进行条件化，MoTVLA能够学习多样化行为并显著提高语言控制能力。在自然语言处理基准、机器人仿真环境和真实世界实验中的广泛评估证实了MoTVLA在快速-慢速推理和操作任务性能方面的优越性。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决在机器人学习中整合视觉语言指令时面临的两个挑战：一是当不使用生成的推理作为条件时语言控制能力有限，二是当整合推理时推理延迟显著。这个问题很重要，因为它限制了机器人在开放世界中的泛化能力和实时应用，影响了机器人在需要快速响应和精确控制的环境中的实用性。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': "作者分析了现有方法的局限性：传统视觉语言动作模型在连续动作表示上存在问题，扩散策略虽然适合连续动作空间但语言控制能力有限。他们提出通过'分解-组合-再分解'的混合变换器架构统一快速和慢速推理。该方法借鉴了混合变换器架构、预训练视觉语言模型、扩散策略等现有工作，并参考了BAGEL模型中的Vision Transformer和Qwen2.5 LLM的文本分词器。", '这个方法的核心思想是什么？整体实现流程是怎样的？': "核心思想是通过混合变换器架构统一快速和慢速推理，在一个模型中同时保留通用智能和领域特定知识，使用'分解-组合-再分解'流程实现知识共享。整体流程包括：输入空间设计(处理语言、RGB图像和可学习查询)；推理骨干设计(通用专家负责慢速推理，领域专家负责快速推理)；推理输出设计(统一在文本空间但分为两种功能)；动作专家设计(使用扩散变换器生成动作)；训练流程(领域专家微调和动作专家扩散策略训练)；推理流程(支持对话模式和动作模式两种交互方式)。", '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)统一快速-慢速推理的MoT架构；2)基于分解运动条件的策略学习；3)支持对话和动作的双模式操作。相比之前的工作，MoTVLA解决了连续动作表示的精度问题，显式生成推理内容提高语言控制能力，显著降低推理延迟，并通过知识共享避免了灾难性遗忘，实现了更好的知识保留和执行效率。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'MoTVLA通过混合变换器架构统一了快速和慢速推理，在保持通用视觉语言模型智能的同时，实现了高效、可解释的机器人操作策略学习，显著提升了语言控制能力和任务执行效率。'}


### 论文摘要

Integrating visual-language instructions into visuomotor policies is gaining momentum in robot learning for enhancing open-world generalization. Despite promising advances, existing approaches face two challenges: limited language steerability when no generated reasoning is used as a condition, or significant inference latency when reasoning is incorporated. In this work, we introduce MoTVLA, a mixture-of-transformers (MoT)-based vision-language-action (VLA) model that integrates fast-slow unified reasoning with behavior policy learning. MoTVLA preserves the general intelligence of pre-trained VLMs (serving as the generalist) for tasks such as perception, scene understanding, and semantic planning, while incorporating a domain expert, a second transformer that shares knowledge with the pretrained VLM, to generate domain-specific fast reasoning (e.g., robot motion decomposition), thereby improving policy execution efficiency. By conditioning the action expert on decomposed motion instructions, MoTVLA can learn diverse behaviors and substantially improve language steerability. Extensive evaluations across natural language processing benchmarks, robotic simulation environments, and real-world experiments confirm the superiority of MoTVLA in both fast-slow reasoning and manipulation task performance.

---

## 10. RoboGPT-R1: Enhancing Robot Planning with Reinforcement Learning

**论文链接:** [http://arxiv.org/abs/2510.14828v2](http://arxiv.org/abs/2510.14828v2)

**作者:** Jinrui Liu, Bingyan Nie, Boyu Li, Yaran Chen, Yuze Wang, Shunsen He, Haoran Li

**发布时间:** 2025-10-16

### GPT解析

### 总结

本文提出了一种名为RoboGPT-R1的两阶段微调框架，用于提高具身智能体在长期操作任务中的推理能力，显著提升了模型在EmbodiedBench基准测试上的表现。

### 背景

提高具身智能体的推理能力对机器人在长期操作任务中成功完成复杂人类指令至关重要。尽管基于监督微调的大语言模型和视觉语言模型在规划任务中取得成功，但在复杂真实环境中的长期操作任务仍面临挑战，这是由于它们有限的常识和推理能力。

### 目的

解决通用视觉语言模型通过监督微调对机器人规划任务的泛化能力差和物理理解不足的问题。

### 方法

提出RoboGPT-R1框架，第一阶段通过监督训练使用专家序列获取基础知识，第二阶段使用强化学习解决模型在视觉空间理解和推理方面的不足。设计基于规则的奖励函数，同时考虑长期性能和环境中的动作约束，以实现多步推理任务中的物理理解和动作序列一致性。

### 主要发现

在Qwen2.5-VL-3B上训练的推理模型在EmbodiedBench基准测试上显著优于更大规模的GPT-4o-mini模型，提高了21.33%，超越了在Qwen2.5-VL-7B上训练的其他工作，提高了20.33%。

### 结论

RoboGPT-R1框架有效提高了具身智能体的推理能力和长期操作任务表现。

### 翻译

提高具身智能体的推理能力对于机器人在长期操作任务中成功完成复杂的人类指令至关重要。尽管基于监督微调的大语言模型和视觉语言模型在规划任务中取得了成功，但由于其有限的常识和推理能力，它们在复杂真实环境中执行长期操作任务时仍面临挑战。考虑到通过监督微调将通用视觉语言模型与机器人规划任务对齐存在泛化能力差和物理理解不足的问题，我们提出了RoboGPT-R1，一个用于具身规划的两阶段微调框架。在该框架中，监督训练通过专家序列获取基础知识，随后使用强化学习来解决模型在视觉空间理解和推理方面的不足。为了在多步推理任务中实现物理理解和动作序列一致性，我们设计了一个基于规则的奖励函数，同时考虑长期性能和环境中的动作约束。在Qwen2.5-VL-3B上训练的推理模型在EmbodiedBench基准测试上显著优于更大规模的GPT-4o-mini模型，提高了21.33%，并超越了在Qwen2.5-VL-7B上训练的其他工作，提高了20.33%。


### 论文摘要

Improving the reasoning capabilities of embodied agents is crucial for robots to complete complex human instructions in long-view manipulation tasks successfully. Despite the success of large language models and vision language models based on Supervised Fine-Tuning (SFT) in planning tasks, they continue facing challenges in performing long-horizon manipulation tasks in complex real-world environments, owing to their restricted common sense and reasoning capabilities. Considering that aligning general-purpose vision language models to robotic planning tasks via supervised fine-tuning suffers from poor generalization and insufficient physical understanding, we propose RoboGPT-R1, a two-stage fine-tuning framework for embodied planning. In this framework, supervised training acquires foundational knowledge through expert sequences, followed by RL to address the model's shortcomings in visual-spatial understanding and reasoning. To achieve physical understanding and action sequence consistency in multi-step reasoning tasks, we design a rule-based reward function that simultaneously considers long-horizon performance and action constraint in the environment. The reasoning model, trained on Qwen2.5-VL-3B, significantly outperforms the larger-scale model, GPT-4o-mini, by 21.33% and surpasses other work trained on Qwen2.5-VL-7B by 20.33% on the EmbodiedBench benchmark.

---

## 11. DAP-MAE: Domain-Adaptive Point Cloud Masked Autoencoder for Effective Cross-Domain Learning

**论文链接:** [http://arxiv.org/abs/2510.21635v1](http://arxiv.org/abs/2510.21635v1)

**作者:** Ziqi Gao, Qiufu Li, Linlin Shen

**发布时间:** 2025-10-24

**备注:** 14 pages, 7 figures, conference

### GPT解析

### 总结

本文提出了一种名为DAP-MAE的领域自适应点云掩码自编码器方法，用于解决跨领域点云数据整合问题，提高下游3D点云分析任务性能。

### 背景

与2D数据相比，可用于训练的点云数据在不同领域中规模有限，研究人员尝试结合不同领域数据进行MAE预训练以缓解数据稀缺问题。

### 目的

开发一种能够自适应整合跨领域数据集知识的方法，以改善通用点云分析任务性能。

### 方法

设计了异构领域适配器，在预训练阶段使用适配模式学习跨领域点云信息，在微调阶段采用融合模式增强特征；同时引入领域特征生成器指导点云特征适应下游任务。

### 主要发现

仅通过一次预训练，DAP-MAE在四种点云分析任务上表现优异，在ScanObjectNN上的目标分类达到95.18%，在Bosphorus上的面部表情识别达到88.45%。

### 结论

DAP-MAE有效解决了跨领域点云数据整合问题，提高了下游任务性能，为点云分析提供了新的解决方案。

### 翻译

与2D数据相比，可用于训练的点云数据在不同领域的规模相当有限。研究人员一直在尝试结合这些不同领域的数据进行掩码自编码器预训练，以利用这种数据稀缺问题。然而，从混合领域学到的先验知识可能与下游3D点云分析任务不太匹配，导致性能下降。为解决这一问题，我们提出了领域自适应点云掩码自编码器，这是一种MAE预训练方法，可以自适应地整合跨领域数据集的知识，用于通用点云分析。在DAP-MAE中，我们设计了一个异构领域适配器，在预训练期间使用适配模式，使模型能够全面学习来自不同领域点云的信息，同时在微调阶段采用融合模式以增强点云特征。同时，DAP-MAE集成了一个领域特征生成器，指导点云特征适应各种下游任务。仅通过一次预训练，DAP-MAE在四种不同的点云分析任务上取得了优异的性能，在ScanObjectNN上的目标分类达到95.18%，在Bosphorus上的面部表情识别达到88.45%。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决点云数据在不同领域（物体、人脸、场景）之间的迁移学习问题。在现实中，3D点云数据的收集和标注需要大量资源，导致各领域可用数据有限。现有方法通常只在单一领域内进行预训练，当应用于不同领域任务时性能显著下降。解决这个问题对于实现通用3D点云分析至关重要，可应用于自动驾驶、机器人、增强/虚拟现实等领域，有效利用有限的数据资源并提高模型泛化能力。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先分析了现有点云MAE方法的局限性：单一领域预训练导致跨领域性能下降，简单组合多领域数据也会因信息误解为噪声而降低性能。基于此，他们设计了DAP-MAE框架，包含异构领域适配器(HDA)和领域特征生成器(DFG)两个核心组件。该方法借鉴了掩码自编码器(MAE)的自监督学习框架、Transformer架构、PointNet的点云处理方法以及对比学习技术，但创新性地将其应用于跨领域点云学习场景，实现了单模态一次预训练适应多任务的目标。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是通过异构领域适配器和领域特征生成器，使模型能够协作学习来自不同领域的点云数据，实现一次预训练适应多种下游任务。整体流程分为两阶段：1)预训练阶段：使用来自物体、人脸、场景三个领域的数据，通过HDA的适应模式分别处理各领域数据，使用Transformer编码器-解码器架构进行掩码重建，同时DFG提取领域特征并通过对比损失训练；2)微调阶段：针对特定下游任务，使用HDA的融合模式整合多领域信息，DFG提取领域和类别特征，输入任务头进行训练。这种方法既保留了各领域的特性，又实现了跨领域知识的有效迁移。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)首次提出DAP-MAE框架，实现单模态一次预训练适应多任务；2)设计异构领域适配器(HDA)，预训练时使用适应模式分别处理不同领域，微调时使用融合模式整合信息；3)引入领域特征生成器(DFG)提取多样化领域特征指导下游任务。相比之前工作：与传统MAE不同，DAP-MAE能跨领域学习；与简单组合多领域数据的方法不同，它避免将跨域信息误解为噪声；与跨模态方法不同，它专注于单模态点云数据降低训练成本；在多个下游任务上实现了优于其他自监督方法的性能。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'DAP-MAE通过异构领域适配器和领域特征生成器实现了跨领域点云数据的有效协作学习，仅需一次预训练就能在多种3D点云分析任务上达到顶尖性能。'}


### 论文摘要

Compared to 2D data, the scale of point cloud data in different domains available for training, is quite limited. Researchers have been trying to combine these data of different domains for masked autoencoder (MAE) pre-training to leverage such a data scarcity issue. However, the prior knowledge learned from mixed domains may not align well with the downstream 3D point cloud analysis tasks, leading to degraded performance. To address such an issue, we propose the Domain-Adaptive Point Cloud Masked Autoencoder (DAP-MAE), an MAE pre-training method, to adaptively integrate the knowledge of cross-domain datasets for general point cloud analysis. In DAP-MAE, we design a heterogeneous domain adapter that utilizes an adaptation mode during pre-training, enabling the model to comprehensively learn information from point clouds across different domains, while employing a fusion mode in the fine-tuning to enhance point cloud features. Meanwhile, DAP-MAE incorporates a domain feature generator to guide the adaptation of point cloud features to various downstream tasks. With only one pre-training, DAP-MAE achieves excellent performance across four different point cloud analysis tasks, reaching 95.18% in object classification on ScanObjectNN and 88.45% in facial expression recognition on Bosphorus.

---

## 12. Robust Point Cloud Reinforcement Learning via PCA-Based Canonicalization

**论文链接:** [http://arxiv.org/abs/2510.20974v1](http://arxiv.org/abs/2510.20974v1)

**作者:** Michael Bezick, Vittorio Giammarino, Ahmed H. Qureshi

**发布时间:** 2025-10-23

### GPT解析

### 总结

本文提出了PCA点云(PPC)框架，用于解决点云强化学习中的相机姿态不匹配问题，通过将点云映射到规范姿态，显著提高了对视角变化的鲁棒性。

### 背景

从原始视觉输入的强化学习近年来取得了显著成功，但它对分布外变化(如光照、颜色和视角变化)仍然很脆弱。点云强化学习提供了一种有前景的替代方案，减轻了基于外观的脆弱性，但其对相机姿态不匹配的敏感性继续削弱了在现实环境中的可靠性。

### 目的

解决点云强化学习对相机姿态不匹配敏感的挑战，提高在现实场景中的可靠性。

### 方法

提出PCA点云(PPC)框架，这是一个专门为下游机器人控制设计的规范化框架，它将任意刚体变换下的点云映射到唯一的规范姿态，将观测结果对齐到一致的坐标系。

### 主要发现

PPC显著减少了视角引起的不一致性，在实验中提高了在具有挑战性的机器人任务中对未见过的相机姿态的鲁棒性，为域随机化提供了有原则的替代方案。

### 结论

PPC框架有效地解决了点云强化学习中的相机姿态不匹配问题，提高了在现实场景中的鲁棒性和可靠性。

### 翻译

从原始视觉输入的强化学习近年来取得了显著成功，但它对分布外变化(如光照、颜色和视角变化)仍然很脆弱。点云强化学习提供了一种有前景的替代方案，减轻了基于外观的脆弱性，但其对相机姿态不匹配的敏感性继续削弱了在现实环境中的可靠性。为应对这一挑战，我们提出了PCA点云(PPC)，这是一个专门为下游机器人控制设计的规范化框架。PPC将任意刚体变换下的点云映射到唯一的规范姿态，将观测结果对齐到一致的坐标系，从而显著减少了视角引起的不一致性。在我们的实验中，我们证明了PPC提高了在具有挑战性的机器人任务中对未见过的相机姿态的鲁棒性，为域随机化提供了有原则的替代方案。


### 论文摘要

Reinforcement Learning (RL) from raw visual input has achieved impressive successes in recent years, yet it remains fragile to out-of-distribution variations such as changes in lighting, color, and viewpoint. Point Cloud Reinforcement Learning (PC-RL) offers a promising alternative by mitigating appearance-based brittleness, but its sensitivity to camera pose mismatches continues to undermine reliability in realistic settings. To address this challenge, we propose PCA Point Cloud (PPC), a canonicalization framework specifically tailored for downstream robotic control. PPC maps point clouds under arbitrary rigid-body transformations to a unique canonical pose, aligning observations to a consistent frame, thereby substantially decreasing viewpoint-induced inconsistencies. In our experiments, we show that PPC improves robustness to unseen camera poses across challenging robotic tasks, providing a principled alternative to domain randomization.

---

## 13. Fractional harmonic transform on point cloud manifolds

**论文链接:** [http://arxiv.org/abs/2510.20842v1](http://arxiv.org/abs/2510.20842v1)

**作者:** Jiamian Li, Bing-Zhao Li

**发布时间:** 2025-10-20

**备注:** Submitted to ICASSP 2026

### GPT解析

### 总结

本文提出了一种点云流形分数阶谐波变换(PMFHT)，通过引入分数阶参数扩展了传统的点云流形谐波变换(PMHT)，构建了空间域和频率域之间可连续调节的中间分数阶谱域，实现了更灵活的变换和滤波操作。

### 背景

点云可被视为光滑流形的离散样本，可使用拉普拉斯-贝尔特拉米算子进行谱分析。然而，传统的PMHT受限于固定基函数和单一谱表示，难以捕获复杂几何特征。

### 目的

提出PMFHT来克服传统PMHT的局限性，通过引入分数阶参数构建连续可调的中间谱域。

### 方法

引入分数阶参数，构建空间域和频率域之间的中间分数阶谱域，支持更灵活的变换和滤波操作。

### 主要发现

实验表明，选择不同的变换顺序可以丰富点云的谱表示，在滤波和特征增强等任务中取得优异结果。

### 结论

PMFHT扩展了点云谱分析的理论框架，为流形几何处理提供了强大的新工具。

### 翻译

三维点云可以被视为光滑流形的离散样本，允许使用拉普拉斯-贝尔特拉米算子进行谱分析。然而，传统的点云流形谐波变换(PMHT)受其固定基函数和单一谱表示的限制，限制了其捕获复杂几何特征的能力。本文提出了一种点云流形分数阶谐波变换(PMFHT)，通过引入分数阶参数推广了PMHT，并构建了空间域和频率域之间可连续调节的中间分数阶谱域。这种分数阶框架支持更灵活的变换和滤波操作。实验表明，选择不同的变换顺序可以丰富点云的谱表示，并在滤波和特征增强等任务中取得优异结果。因此，PMFHT不仅扩展了点云谱分析的理论框架，还为流形几何处理提供了强大的新工具。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文解决传统点云流形谐波变换(PMHT)的局限性，即其固定的基函数和单一频谱表示无法充分捕捉复杂几何特征。这一问题很重要，因为三维点云是3D场景最常见的数据表示形式之一，广泛应用于LiDAR、结构光扫描和立体重建等领域，有效的几何特征提取对点云处理至关重要。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先认识到传统PMHT的局限性，然后从信号处理中的分数傅里叶变换(FRFT)获得启发，后者通过引入分数阶参数解决了类似限制，提供了时域和频域之间的连续中间表示。作者借鉴了PMHT的基础框架、FRFT的分数阶参数思想以及LBO在点云上的离散化方法，将流形谐波扩展为分数阶形式，通过非线性缩放LBO特征值创建连续可调的中间频谱域。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是引入分数阶参数α，将传统PMHT扩展为分数阶形式(PMFHT)，创建空间域和频率域之间可连续调整的中间分数阶频谱域。实现流程包括：1)构建离散拉普拉斯-贝尔特拉米算子；2)求解广义特征值问题获得点云流形谐波基；3)定义分数阶傅里叶矩阵和点云流形分数阶谐波变换；4)应用变换进行不同类型的滤波操作，如特征增强或平滑处理。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)构建点云流形分数阶谐波变换的统一框架；2)提供简单高效的PMFHT分数幂公式；3)证明PMFHT能提供更丰富的频谱表示并在点云处理任务中表现出色。相比传统PMHT，PMFHT引入分数阶参数提供连续可调的中间频谱域，能捕捉多尺度几何特征，通过选择不同变换阶数丰富频谱表示，为流形几何处理提供新工具。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '本文提出了点云流形分数阶谐波变换(PMFHT)，通过引入分数阶参数扩展了传统点云流形谐波变换，提供了空间域和频率域之间的连续可调中间表示，丰富了点云的频谱分析能力，并在点云处理任务中展现出优越性能。'}


### 论文摘要

Three-dimensional point clouds can be viewed as discrete samples of smooth manifolds, allowing spectral analysis using the Laplace-Beltrami operator (LBO). However, the traditional point cloud manifold harmonic transform (PMHT) is limited by its fixed basis functions and single spectral representation, which restricts its ability to capture complex geometric features. This paper proposes a point cloud manifold fractional harmonic transform (PMFHT), which generalizes PMHT by introducing fractional-order parameters and constructs a continuously adjustable intermediate fractional-order spectral domain between the spatial domain and the frequency domain. This fractional-order framework supports more flexible transformation and filtering operations. Experiments show that choosing different transformation orders can enrich the spectral representation of point clouds and achieve excellent results in tasks such as filtering and feature enhancement. Therefore, PMFHT not only expands the theoretical framework of point cloud spectral analysis, but also provides a powerful new tool for manifold geometry processing.

---

## 14. REVE: A Foundation Model for EEG -- Adapting to Any Setup with Large-Scale Pretraining on 25,000 Subjects

**论文链接:** [http://arxiv.org/abs/2510.21585v1](http://arxiv.org/abs/2510.21585v1)

**作者:** Yassine El Ouahidi, Jonathan Lys, Philipp Thölke, Nicolas Farrugia, Bastien Pasdeloup, Vincent Gripon, Karim Jerbi, Giulia Lioi

**发布时间:** 2025-10-24

**备注:** Code available at: https://brain-bzh.github.io/reve/

### GPT解析

### 总结

本文介绍了REVE模型，一个专为EEG信号设计的基础模型，能够处理不同长度和电极排列的EEG信号，并在多种下游任务中取得了最先进的结果。

### 背景

基础模型通过大规模预训练减少了任务特定数据的依赖，在语言和视觉领域取得成功，但在EEG领域的应用滞后。公共EEG数据集的异质性（不同协议、设备和电极配置）导致现有EEG基础模型难以泛化，现有模型通常限制在单一设置下预训练，导致次优性能，特别是在线性探测下。

### 目的

开发一个能够跨多样化EEG信号泛化的预训练模型REVE（Representation for EEG with Versatile Embeddings）。

### 方法

引入了一种新颖的4D位置编码方案，使其能够处理任意长度和电极排列的信号；使用掩码自编码目标函数进行预训练；在来自92个数据集、25,000名受试者的超过60,000小时EEG数据上预训练REVE，这是迄今为止最大的EEG预训练工作。

### 主要发现

REVE在10个下游EEG任务上取得了最先进的结果，包括运动想象分类、癫痫检测、睡眠分期、认知负荷估计和情绪识别；几乎不需要微调的情况下，REVE展示了强大的泛化能力和细致的时空建模能力。

### 结论

REVE为EEG信号处理提供了新的基础模型，能够处理多样化的EEG数据；研究团队发布了代码、预训练权重和教程，以支持标准化的EEG研究并加速临床神经科学的进展。

### 翻译

基础模型通过大规模预训练减少了对任务特定数据的依赖，从而改变了人工智能领域。尽管在语言和视觉领域取得了成功，但由于公共数据集的异质性（收集于不同的协议、设备和电极配置下），它们在EEG中的应用一直滞后。现有的EEG基础模型难以在这些变化中泛化，通常将预训练限制在单一设置中，导致次优性能，特别是在线性探测下。我们提出了REVE（Representation for EEG with Versatile Embeddings），一个明确设计为能够泛化到多样化EEG信号的预训练模型。REVE引入了一种新颖的4D位置编码方案，使其能够处理任意长度和电极排列的信号。使用掩码自编码目标函数，我们在来自92个数据集、25,000名受试者的超过60,000小时EEG数据上预训练了REVE，这是迄今为止最大的EEG预训练工作。REVE在10个下游EEG任务上取得了最先进的结果，包括运动想象分类、癫痫检测、睡眠分期、认知负荷估计和情绪识别。几乎不需要微调的情况下，它展示了强大的泛化能力和细致的时空建模能力。我们发布了代码、预训练权重和教程，以支持标准化的EEG研究并加速临床神经科学的进展。


### 论文摘要

Foundation models have transformed AI by reducing reliance on task-specific data through large-scale pretraining. While successful in language and vision, their adoption in EEG has lagged due to the heterogeneity of public datasets, which are collected under varying protocols, devices, and electrode configurations. Existing EEG foundation models struggle to generalize across these variations, often restricting pretraining to a single setup, resulting in suboptimal performance, in particular under linear probing. We present REVE (Representation for EEG with Versatile Embeddings), a pretrained model explicitly designed to generalize across diverse EEG signals. REVE introduces a novel 4D positional encoding scheme that enables it to process signals of arbitrary length and electrode arrangement. Using a masked autoencoding objective, we pretrain REVE on over 60,000 hours of EEG data from 92 datasets spanning 25,000 subjects, representing the largest EEG pretraining effort to date. REVE achieves state-of-the-art results on 10 downstream EEG tasks, including motor imagery classification, seizure detection, sleep staging, cognitive load estimation, and emotion recognition. With little to no fine-tuning, it demonstrates strong generalization, and nuanced spatio-temporal modeling. We release code, pretrained weights, and tutorials to support standardized EEG research and accelerate progress in clinical neuroscience.

---

## 15. MUVR: A Multi-Modal Untrimmed Video Retrieval Benchmark with Multi-Level Visual Correspondence

**论文链接:** [http://arxiv.org/abs/2510.21406v1](http://arxiv.org/abs/2510.21406v1)

**作者:** Yue Feng, Jinwei Hu, Qijia Lu, Jiawei Niu, Li Tan, Shuo Yuan, Ziyi Yan, Yizhen Jia, Qingzhi He, Shiping Ge, Ethan Q. Chen, Wentong Li, Limin Wang, Jie Qin

**发布时间:** 2025-10-24

**备注:** Accepted to NeurIPS 2025 D&B Track

### GPT解析

### 总结

本研究提出了多模态未修剪视频检索(MUVR)任务及相应基准数据集，旨在推进长视频平台上的视频检索技术。该研究构建了实用的检索范式、多层次视觉对应和全面的评估标准，并对多种先进模型进行了评估，揭示了当前方法在处理未修剪视频和多模态查询方面的局限性。

### 背景

随着长视频平台的普及，视频检索技术面临新的挑战。现有的视频检索方法主要针对修剪过的短视频，而长视频平台上的视频通常包含多个相关片段，需要更灵活的检索方式来满足用户需求。

### 目的

提出并构建一个专门针对长视频平台的多模态未修剪视频检索任务和基准数据集，以促进该领域的研究和发展，并评估现有方法在这一新任务上的表现。

### 方法

设计了MUVR基准数据集，包含53K个来自Bilibili的未修剪视频、1,050个多模态查询和84K个匹配。构建了以视频为中心的多模态查询支持长文本描述、视频标签提示和掩码提示。建立了六个级别的多层次视觉对应标准（副本、事件、场景、实例、动作和其他）。开发了三个版本的评估基准（基础版、过滤版、问答版），并提出了重新排序分数评估指标。

### 主要发现

评估结果显示，当前的视频检索模型在处理未修剪视频和多模态查询方面存在明显局限性；MLLMs在多视频理解和重新排序能力上也表现出不足，这为未来研究指明了方向。

### 结论

MUVR基准为长视频平台上的视频检索研究提供了新的评估框架，揭示了现有方法的不足，并为未来改进提供了方向。该研究有助于推动多模态未修剪视频检索领域的发展。

### 翻译

我们提出了多模态未修剪视频检索任务，并创建了一个新的基准(MUVR)以推进长视频平台的视频检索。MUVR旨在使用多模态查询检索包含相关片段的未修剪视频。它具有以下特点：1)实用的检索范式：MUVR支持以视频为中心的多模态查询，通过长文本描述、视频标签提示和掩码提示表达细粒度检索需求。它采用一对多检索范式，专注于未修剪视频，专为长视频平台应用定制。2)多层次视觉对应：为了涵盖常见的视频类别（如新闻、旅行、舞蹈）并精确定义检索匹配标准，我们基于用户感兴趣且想要检索的核心视频内容（如新闻事件、旅行地点、舞蹈动作）构建了多层次视觉对应。它涵盖六个级别：副本、事件、场景、实例、动作和其他。3)全面的评估标准：我们开发了3个版本的MUVR（即基础版、过滤版、问答版）。MUVR-Base/Filter评估检索模型，而MUVR-QA以问答格式评估MLLMs。我们还提出了重新排序分数来评估MLLMs的重新排序能力。MUVR包含来自Bilibili视频平台的53K个未修剪视频，有1,050个多模态查询和84K个匹配。我们对3个最先进的视频检索模型、6个基于图像的VLMs和10个MLLMs进行了广泛评估。MUVR揭示了检索方法在处理未修剪视频和多模态查询方面的局限性，以及MLLMs在多视频理解和重新排序方面的局限性。我们的代码和基准可在https://github.com/debby-0527/MUVR获取。


### 论文摘要

We propose the Multi-modal Untrimmed Video Retrieval task, along with a new benchmark (MUVR) to advance video retrieval for long-video platforms. MUVR aims to retrieve untrimmed videos containing relevant segments using multi-modal queries. It has the following features: 1) Practical retrieval paradigm: MUVR supports video-centric multi-modal queries, expressing fine-grained retrieval needs through long text descriptions, video tag prompts, and mask prompts. It adopts a one-to-many retrieval paradigm and focuses on untrimmed videos, tailored for long-video platform applications. 2) Multi-level visual correspondence: To cover common video categories (e.g., news, travel, dance) and precisely define retrieval matching criteria, we construct multi-level visual correspondence based on core video content (e.g., news events, travel locations, dance moves) which users are interested in and want to retrieve. It covers six levels: copy, event, scene, instance, action, and others. 3) Comprehensive evaluation criteria: We develop 3 versions of MUVR (i.e., Base, Filter, QA). MUVR-Base/Filter evaluates retrieval models, while MUVR-QA assesses MLLMs in a question-answering format. We also propose a Reranking Score to evaluate the reranking ability of MLLMs. MUVR consists of 53K untrimmed videos from the video platform Bilibili, with 1,050 multi-modal queries and 84K matches. Extensive evaluations of 3 state-of-the-art video retrieval models, 6 image-based VLMs, and 10 MLLMs are conducted. MUVR reveals the limitations of retrieval methods in processing untrimmed videos and multi-modal queries, as well as MLLMs in multi-video understanding and reranking. Our code and benchmark is available at https://github.com/debby-0527/MUVR.

---

## 16. HRT1: One-Shot Human-to-Robot Trajectory Transfer for Mobile Manipulation

**论文链接:** [http://arxiv.org/abs/2510.21026v1](http://arxiv.org/abs/2510.21026v1)

**作者:** Sai Haneesh Allu, Jishnu Jaykumar P, Ninad Khargonkar, Tyler Summers, Jian Yao, Yu Xiang

**发布时间:** 2025-10-23

**备注:** 14 pages, 11 figures and 3 tables. Project page is available at  \url{https://irvlutd.github.io/HRT1/}

### GPT解析

### 总结

该研究介绍了一个新颖的人机轨迹传递系统，使机器人能够通过学习人类示范视频来操作物体，系统包含四个模块，实现了机器人观看一次人类示范后就能在不同环境中重复相同操作任务的能力。

### 背景

机器人操作任务通常需要精确的编程和大量调整，而人类能够直观地通过示范学习操作任务。如何让机器人从人类示范中学习操作技能是一个重要研究方向。

### 目的

开发一个系统，使机器人能够通过观看人类示范视频来学习操作任务，并能在不同环境中重复这些任务，即使物体放置方式与示范不同。

### 方法

系统包含四个模块：1)使用AR头戴设备从机器人视角收集人类示范视频的数据收集模块；2)从示范视频中检测物体并提取3D人类手部轨迹的视频理解模块；3)将人类手部轨迹转换为机器人末端执行器参考轨迹的轨迹传递模块；4)利用轨迹优化算法解决机器人配置空间中轨迹问题的轨迹优化模块。

### 主要发现

实验证明该系统能够使移动机械臂观看一次人类示范视频后，就能在不同的环境中重复相同的移动操作任务，即使物体放置方式与示范不同。

### 结论

该系统有效地实现了从人类示范到机器人操作的技能传递，为机器人学习人类操作任务提供了一种直观、灵活的方法。

### 翻译

我们介绍了一种新颖的人机轨迹传递系统，使机器人能够通过学习人类示范视频来操作物体。该系统由四个模块组成。第一个模块是数据收集模块，旨在使用AR头戴设备从机器人视角收集人类示范视频。第二个模块是视频理解模块，从示范视频中检测物体并提取3D人类手部轨迹。第三个模块将人类手部轨迹转换为3D空间中机器人末端执行器的参考轨迹。最后一个模块利用轨迹优化算法解决机器人配置空间中的轨迹问题，使其能够遵循从人类示范传递而来的末端执行器轨迹。因此，这些模块使机器人能够观看一次人类示范视频，然后在不同环境中重复相同的移动操作任务，即使物体放置方式与示范不同。我们在移动机械臂上进行了不同操作任务的实验，以验证我们系统的有效性。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决如何让机器人通过观看一次人类演示视频，就能在不同环境中重复执行相同的移动操作任务。这个问题在现实中很重要，因为传统的机器人操作需要大量编程和调参，而现有的基于学习的方法通常需要大量机器人遥操作数据或多次演示，收集成本高。此外，现有方法在物体被手部遮挡时表现不佳，且大多不支持移动操作，限制了机器人在日常环境中的应用。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先分析了现有基于人类演示的机器人操作方法，发现模仿学习方法需要多次演示，强化学习方法需要构建任务空间的数字孪生，而训练免费方法在物体遮挡或噪声处理方面存在局限。作者借鉴了多个现有工作：使用AR头显收集数据类似iTeach框架；使用GroundingDINO和SAMv2进行物体检测；使用HaMeR进行手部姿态估计；使用统一夹持器坐标系空间(UGCS)进行抓取转移；使用BundleSDF进行物体姿态估计。基于这些分析，作者设计了HRT1系统，专注于手部轨迹转移而非物体轨迹，并加入了轨迹优化算法以提高鲁棒性。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是通过分析人类演示视频中的手部动作，将其转换为机器人的执行轨迹，使机器人能够一次性学习并执行相同的操作任务，即使在不同环境和物体摆放情况下也能成功。整体流程分为四个模块：1)数据收集模块：使用HoloLens 2从机器人视角收集人类演示视频；2)视频理解模块：检测物体并提取3D人类手部轨迹；3)人类到机器人抓取转移模块：使用UGCS表示将人类手部轨迹转换为机器人夹持器轨迹；4)任务执行的轨迹对齐模块：使用BundleSDF估计物体姿态变换，并通过两阶段轨迹优化算法(先优化机器人基座位置，再优化关节空间轨迹)使机器人精确执行任务。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)基于手部轨迹而非物体轨迹的转移，在物体被遮挡时更可靠；2)使用统一夹持器坐标系空间(UGCS)进行抓取转移，支持不同类型夹持器；3)两阶段轨迹优化算法，能处理转移轨迹中的噪声；4)支持移动操作，是首个支持移动的训练免费方法；5)改进的3D手部姿态估计，提高深度准确性。相比之前工作，HRT1不依赖物体姿态估计，使用轨迹优化而非简单逆运动学，支持移动操作，且只需一次演示。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'HRT1通过分析人类演示视频中的手部动作并转换为机器人执行轨迹，实现了机器人仅需观看一次演示就能在不同环境中成功执行移动操作任务的能力，显著提高了轨迹转移的准确性和鲁棒性。'}


### 论文摘要

We introduce a novel system for human-to-robot trajectory transfer that enables robots to manipulate objects by learning from human demonstration videos. The system consists of four modules. The first module is a data collection module that is designed to collect human demonstration videos from the point of view of a robot using an AR headset. The second module is a video understanding module that detects objects and extracts 3D human-hand trajectories from demonstration videos. The third module transfers a human-hand trajectory into a reference trajectory of a robot end-effector in 3D space. The last module utilizes a trajectory optimization algorithm to solve a trajectory in the robot configuration space that can follow the end-effector trajectory transferred from the human demonstration. Consequently, these modules enable a robot to watch a human demonstration video once and then repeat the same mobile manipulation task in different environments, even when objects are placed differently from the demonstrations. Experiments of different manipulation tasks are conducted on a mobile manipulator to verify the effectiveness of our system

---

## 17. LLM-Integrated Bayesian State Space Models for Multimodal Time-Series Forecasting

**论文链接:** [http://arxiv.org/abs/2510.20952v1](http://arxiv.org/abs/2510.20952v1)

**作者:** Sungjun Cho, Changho Shin, Suenggwan Jo, Xinya Yan, Shourjo Aditya Chaudhuri, Frederic Sala

**发布时间:** 2025-10-23

**备注:** 15 pages, 8 figures

### GPT解析

### 总结

论文提出了LLM集成的贝叶斯状态空间模型(LBS)，一种用于多模态时间预测的新概率框架，解决了现有方法在架构上的限制，实现了灵活的时间窗口、不确定性量化和改进的时间泛化能力。

### 背景

现实世界预测需要整合结构化时间序列数据与非结构化文本信息，但现有方法受固定输入/输出时间跨度限制，无法建模或量化不确定性。

### 目的

开发一种能够同时处理数值和文本信息的时间预测方法，提供灵活的时间窗口、不确定性量化和更好的时间泛化能力。

### 方法

LBS包含两个主要组件：(1)状态空间模型(SSM)骨干，捕获生成数值和文本观测的潜在状态的时间动态；(2)预训练大型语言模型(LLM)，用于编码文本输入进行后验状态估计和解码文本预测。

### 主要发现

LBS在TextTimeCorpus基准测试上比之前的最先进方法提高13.20%，同时为每个预测提供可读的人类摘要，实现了灵活的回看和预测窗口、原则性不确定性量化和改进的时间泛化。

### 结论

该研究首次统一LLM和SSM进行数值和文本联合预测，为多模态时间推理提供了新的基础框架。

### 翻译

现实世界中的预测需要整合结构化的时间序列数据和非结构化的文本信息，但现有方法在架构上受到固定输入/输出时间跨度的限制，无法建模或量化不确定性。我们通过引入LLM集成的贝叶斯状态空间模型(LBS)来解决这一挑战，这是一种用于多模态时间预测的新概率框架。总体而言，LBS包含两个组件：(1)状态空间模型(SSM)骨干，捕获生成数值和文本观测的潜在状态的时间动态；(2)预训练的大型语言模型(LLM)，调整后用于编码文本输入进行后验状态估计和解码与潜在轨迹一致的文本预测。这种设计能够提供灵活的回看和预测窗口，原则性的不确定性量化，并改善时间泛化能力。在TextTimeCorpus基准测试上的实验表明，LBS比之前的最先进方法提高13.20%，同时为每个预测提供可读的人类摘要。我们的工作是首次统一LLM和SSM进行数值和文本联合预测，为多模态时间推理提供了新的基础。


### 论文摘要

Forecasting in the real world requires integrating structured time-series data with unstructured textual information, but existing methods are architecturally limited by fixed input/output horizons and are unable to model or quantify uncertainty. We address this challenge by introducing LLM-integrated Bayesian State space models (LBS), a novel probabilistic framework for multimodal temporal forecasting. At a high level, LBS consists of two components: (1) a state space model (SSM) backbone that captures the temporal dynamics of latent states from which both numerical and textual observations are generated and (2) a pretrained large language model (LLM) that is adapted to encode textual inputs for posterior state estimation and decode textual forecasts consistent with the latent trajectory. This design enables flexible lookback and forecast windows, principled uncertainty quantification, and improved temporal generalization thanks to the well-suited inductive bias of SSMs toward modeling dynamical systems. Experiments on the TextTimeCorpus benchmark demonstrate that LBS improves the previous state-of-the-art by 13.20% while providing human-readable summaries of each forecast. Our work is the first to unify LLMs and SSMs for joint numerical and textual prediction, offering a novel foundation for multimodal temporal reasoning.

---

## 18. SeViCES: Unifying Semantic-Visual Evidence Consensus for Long Video Understanding

**论文链接:** [http://arxiv.org/abs/2510.20622v1](http://arxiv.org/abs/2510.20622v1)

**作者:** Yuan Sheng, Yanbin Hao, Chenxu Li, Shuo Wang, Xiangnan He

**发布时间:** 2025-10-23

### GPT解析

### 总结

本文提出了SeViCES框架，通过语义-视觉共识证据选择方法解决长视频理解中的挑战，在准确性和鲁棒性方面超越了现有方法。

### 背景

长视频理解因其复杂、多样且时间分散的内容而具有挑战性。现有的Video-LLMs可处理数十分钟的视频，但应用于真正长序列时计算成本高，且推理往往不聚焦或不一致。

### 目的

开发一个有效可靠的长视频理解框架，解决现有方法中忽略时间依赖性和依赖单模态证据的局限性。

### 方法

SeViCES是一个无需训练且与模型无关的框架，包含两个关键组件：(1)语义-视觉共识帧选择(SVCFS)模块，通过时间感知语义分支和聚类引导视觉分支选择信息量最大的帧；(2)答案共识精炼(ACR)模块，通过融合证据和约束答案空间解决语义和视觉预测间的不一致。

### 主要发现

在长视频理解基准上的大量实验表明，SeViCES在准确性和鲁棒性方面均持续优于最先进的方法。

### 结论

共识驱动的证据选择对Video-LLMs的长视频理解能力至关重要。

### 翻译

长视频理解由于其复杂、多样且时间分散的内容而仍然具有挑战性。尽管视频大语言模型(Video-LLMs)可以处理长达数十分钟的视频，但将它们应用于真正长序列在计算上是禁止的，并且往往导致不聚焦或不一致的推理。一个有希望的解决方案是只选择信息量最大的帧，然而现有方法通常忽略时间依赖性或依赖单模态证据，限制了它们提供完整且与查询相关上下文的能力。我们提出了一个用于有效可靠长视频理解的语义-视觉共识证据选择(SeViCES)框架。SeViCES无需训练且与模型无关，并引入了两个关键组件。语义-视觉共识帧选择(SVCFS)模块通过(1)利用LLM对字幕进行推理的时间感知语义分支，和(2)通过互信息将嵌入与语义分数对齐的聚类引导视觉分支来选择帧。答案共识精炼(ACR)模块通过融合证据和约束答案空间，进一步解决基于语义和视觉的预测之间的一致性问题。在长视频理解基准上的大量实验表明，SeViCES在准确性和鲁棒性方面均持续优于最先进的方法，证明了共识驱动的证据选择对Video-LLMs的重要性。


### 论文摘要

Long video understanding remains challenging due to its complex, diverse, and temporally scattered content. Although video large language models (Video-LLMs) can process videos lasting tens of minutes, applying them to truly long sequences is computationally prohibitive and often leads to unfocused or inconsistent reasoning. A promising solution is to select only the most informative frames, yet existing approaches typically ignore temporal dependencies or rely on unimodal evidence, limiting their ability to provide complete and query-relevant context. We propose a Semantic-Visual Consensus Evidence Selection (SeViCES) framework for effective and reliable long video understanding. SeViCES is training-free and model-agnostic, and introduces two key components. The Semantic-Visual Consensus Frame Selection (SVCFS) module selects frames through (1) a temporal-aware semantic branch that leverages LLM reasoning over captions, and (2) a cluster-guided visual branch that aligns embeddings with semantic scores via mutual information. The Answer Consensus Refinement (ACR) module further resolves inconsistencies between semantic- and visual-based predictions by fusing evidence and constraining the answer space. Extensive experiments on long video understanding benchmarks show that SeViCES consistently outperforms state-of-the-art methods in both accuracy and robustness, demonstrating the importance of consensus-driven evidence selection for Video-LLMs.

---

## 19. Open-o3 Video: Grounded Video Reasoning with Explicit Spatio-Temporal Evidence

**论文链接:** [http://arxiv.org/abs/2510.20579v1](http://arxiv.org/abs/2510.20579v1)

**作者:** Jiahao Meng, Xiangtai Li, Haochen Wang, Yue Tan, Tao Zhang, Lingdong Kong, Yunhai Tong, Anran Wang, Zhiyang Teng, Yujing Wang, Zhuochen Wang

**发布时间:** 2025-10-23

### GPT解析

### 总结

Open-o3 Video是一个非智能体框架，将显式时空证据整合到视频推理中，通过专门的数据集和训练策略实现了在多个视频理解基准测试上的最先进性能。

### 背景

大多数视频推理模型只生成文本推理轨迹而不指示关键证据出现的时间和位置。将图像证据中心推理扩展到视频更具挑战性，因为它需要在动态场景中联合时空跟踪和定位。

### 目的

引入Open-o3 Video框架，解决视频推理中时空证据整合的挑战，通过收集训练数据和设计训练策略来提高模型性能。

### 方法

创建了两个高质量数据集STGR-CoT-30k用于SFT和STGR-RL-36k用于RL，包含精心构建的时空注释；采用冷启动强化学习策略，使用多种专门设计的奖励来鼓励答案准确性、时间对齐和空间精度。

### 主要发现

在V-STAR基准测试上，Open-o3 Video实现了最先进性能，相比Qwen2.5-VL基线，mAM提高14.4%，mLGM提高24.2%；在VideoMME、WorldSense、VideoMMMU和TVGBench等多个视频理解基准测试上观察到一致改进。

### 结论

Open-o3 Video的推理轨迹为测试时扩展提供了有价值的信号，支持置信感知的验证，提高答案可靠性。

### 翻译

大多数视频推理模型只生成文本推理轨迹而不指示关键证据出现的时间和位置。最近的模型如OpenAI-o3在图像证据中心推理方面引起了广泛兴趣，但将这种能力扩展到视频更具挑战性，因为它需要在动态场景中联合时空跟踪和定位。我们引入了Open-o3 Video，一个将显式时空证据整合到视频推理中的非智能体框架，并仔细收集训练数据和设计训练策略来解决上述挑战。模型在答案旁边突出显示关键时间戳、对象和边界框，使推理能够基于具体的视觉观察。为实现这一功能，我们首先策划并构建了两个高质量数据集：用于SFT的STGR-CoT-30k和用于RL的STGR-RL-36k，包含精心构建的时间和空间注释，因为大多数现有数据集只提供视频的时间跨度或图像的空间框，缺乏统一的时空监督和推理轨迹。然后，我们采用冷启动强化学习策略，使用多种专门设计的奖励，共同鼓励答案准确性、时间对齐和空间精度。在V-STAR基准测试上，Open-o3 Video实现了最先进的性能，相比Qwen2.5-VL基线，mAM提高了14.4%，mLGM提高了24.2%。在广泛的视频理解基准测试上，包括VideoMME、WorldSense、VideoMMMU和TVGBench，也观察到一致改进。除了准确性，Open-o3 Video产生的推理轨迹还为测试时扩展提供了有价值的信号，使置信感知的验证成为可能，并提高答案可靠性。


### 论文摘要

Most video reasoning models only generate textual reasoning traces without indicating when and where key evidence appears. Recent models such as OpenAI-o3 have sparked wide interest in evidence-centered reasoning for images, yet extending this ability to videos is more challenging, as it requires joint temporal tracking and spatial localization across dynamic scenes. We introduce Open-o3 Video, a non-agent framework that integrates explicit spatio-temporal evidence into video reasoning, and carefully collect training data and design training strategies to address the aforementioned challenges. The model highlights key timestamps, objects, and bounding boxes alongside its answers, allowing reasoning to be grounded in concrete visual observations. To enable this functionality, we first curate and build two high-quality datasets, STGR-CoT-30k for SFT and STGR-RL-36k for RL, with carefully constructed temporal and spatial annotations, since most existing datasets offer either temporal spans for videos or spatial boxes on images, lacking unified spatio-temporal supervision and reasoning traces. Then, we adopt a cold-start reinforcement learning strategy with multiple specially designed rewards that jointly encourage answer accuracy, temporal alignment, and spatial precision. On V-STAR benchmark, Open-o3 Video achieves state-of-the-art performance, raising mAM by 14.4% and mLGM by 24.2% on the Qwen2.5-VL baseline. Consistent improvements are also observed on a broad range of video understanding benchmarks, including VideoMME, WorldSense, VideoMMMU, and TVGBench. Beyond accuracy, the reasoning traces produced by Open-o3 Video also provide valuable signals for test-time scaling, enabling confidence-aware verification and improving answer reliability.

---

## 20. Conan: Progressive Learning to Reason Like a Detective over Multi-Scale Visual Evidence

**论文链接:** [http://arxiv.org/abs/2510.20470v1](http://arxiv.org/abs/2510.20470v1)

**作者:** Kun Ouyang, Yuanxin Liu, Linli Yao, Yishuo Cai, Hao Zhou, Jie Zhou, Fandong Meng, Xu Sun

**发布时间:** 2025-10-23

### GPT解析

### 总结

本文提出了Conan，一个基于证据的多步视频推理框架，通过识别上下文和证据帧、跨帧线索推理以及自适应决策机制，解决了现有视频推理方法的局限性。

### 背景

视频推理需要跨帧多步推理，对多模态大语言模型(MLLMs)仍是主要挑战。基于强化学习的方法依赖文本链导致结论缺乏基础，帧检索方法则难以准确进行证据定位。

### 目的

开发一个能够有效进行多步视频推理的框架，解决现有方法在推理准确性和证据定位方面的局限性。

### 方法

构建Conan-91K数据集，包含自动生成的推理轨迹；设计多阶段渐进式冷启动策略和识别-推理-行动(AIR)RLVR训练框架，共同增强多步视觉推理能力。

### 主要发现

在六个多步推理基准测试上，Conan的准确性平均超过基线Qwen2.5-VL-7B-Instruct模型10%以上，达到最先进性能；且能有效泛化到长视频理解任务，展示出良好的可扩展性和鲁棒性。

### 结论

Conan框架通过结合证据基础和多步推理，有效解决了视频推理中的挑战，实现了更准确可靠的视频理解。

### 翻译

视频推理需要跨帧多步推理，这对多模态大语言模型(MLLMs)仍然是一个主要挑战。虽然基于强化学习(RL)的方法增强了推理能力，但它们通常只依赖文本链，导致结论缺乏基础或产生幻觉。相反，帧检索方法引入了视觉基础，但仍然难以准确进行证据定位。为了解决这些挑战，我们提出了Conan，一个用于基于证据的多步视频推理框架。Conan能够识别上下文和证据帧，跨帧线索进行推理，并自适应地决定何时得出结论或进一步探索。为此，我们(1)构建了Conan-91K，这是一个大规模的自动生成推理轨迹数据集，包括帧识别、证据推理和行动决策，以及(2)设计了一个多阶段渐进式冷启动策略，结合识别-推理-行动(AIR)RLVR训练框架，共同增强多步视觉推理。在六个多步推理基准测试上的广泛实验表明，Conan在准确性上平均超过基线Qwen2.5-VL-7B-Instruct模型10%以上，达到了最先进的性能。此外，Conan能有效地泛化到长视频理解任务，验证了其强大的可扩展性和鲁棒性。


### 论文摘要

Video reasoning, which requires multi-step deduction across frames, remains a major challenge for multimodal large language models (MLLMs). While reinforcement learning (RL)-based methods enhance reasoning capabilities, they often rely on text-only chains that yield ungrounded or hallucinated conclusions. Conversely, frame-retrieval approaches introduce visual grounding but still struggle with inaccurate evidence localization. To address these challenges, we present Conan, a framework for evidence-grounded multi-step video reasoning. Conan identifies contextual and evidence frames, reasons over cross-frame clues, and adaptively decides when to conclude or explore further. To achieve this, we (1) construct Conan-91K, a large-scale dataset of automatically generated reasoning traces that includes frame identification, evidence reasoning, and action decision, and (2) design a multi-stage progressive cold-start strategy combined with an Identification-Reasoning-Action (AIR) RLVR training framework to jointly enhance multi-step visual reasoning. Extensive experiments on six multi-step reasoning benchmarks demonstrate that Conan surpasses the baseline Qwen2.5-VL-7B-Instruct by an average of over 10% in accuracy, achieving state-of-the-art performance. Furthermore, Conan generalizes effectively to long-video understanding tasks, validating its strong scalability and robustness.

---

## 21. InvDec: Inverted Decoder for Multivariate Time Series Forecasting with Separated Temporal and Variate Modeling

**论文链接:** [http://arxiv.org/abs/2510.20302v1](http://arxiv.org/abs/2510.20302v1)

**作者:** Yuhang Wang

**发布时间:** 2025-10-23

**备注:** 23pages, 3 figures

### GPT解析

### 总结

论文提出了一种名为InvDec的混合架构，用于多元时间序列预测，有效结合了时间建模和跨变量依赖关系建模，特别在高维数据集上表现优异。

### 背景

多元时间序列预测需要同时建模时间模式和跨变量依赖关系。现有方法存在局限性：通道独立方法如PatchTST擅长时间建模但忽略变量相关性，而纯变量注意力方法如iTransformer牺牲了时间编码。

### 目的

提出一种混合架构，实现时间编码和变量级解码的原则性分离，以同时捕捉时间模式和跨变量依赖关系。

### 方法

提出InvDec架构，结合基于补丁的时间编码器和通过变量级自注意力操作的倒置解码器；引入延迟变量嵌入，在时间编码后才丰富变量特定表示；采用自适应残差融合机制动态平衡时间信息和变量信息；将InvDec与PatchTST结合形成InvDec-PatchTST。

### 主要发现

在七个基准测试上，InvDec-PatchTST在高维数据集上表现显著：Electricity数据集（321个变量）MSE降低20.9%，Weather数据集提升4.3%，Traffic数据集提升2.7%；在低维ETT数据集上保持竞争力；消融研究验证了各组件有效性；InvDec的优势随数据集维度增长而增长。

### 结论

InvDec有效地结合了时间建模和跨变量相关性建模，特别适合高维数据集，随着变量数量增加，跨变量建模变得至关重要。

### 翻译

多元时间序列预测需要同时建模时间模式和跨变量依赖关系。通道独立方法如PatchTST擅长时间建模但忽略了变量相关性，而纯变量注意力方法如iTransformer牺牲了时间编码。我们提出了InvDec（倒置解码器），一种混合架构，实现了时间编码和变量级解码的原则性分离。InvDec结合了基于补丁的时间编码器和一个通过变量级自注意力操作在变量维度上运行的倒置解码器。我们引入了延迟变量嵌入，仅在时间编码后丰富变量特定表示，保持时间特征完整性。自适应残差融合机制动态平衡不同维度数据集的时间信息和变量信息。将InvDec与PatchTST实例化得到InvDec-PatchTST。在七个基准测试上的广泛实验表明，在高维数据集上取得了显著提升：Electricity（321个变量）上MSE降低20.9%，Weather上提升4.3%，Traffic上提升2.7%，同时在低维ETT数据集上保持竞争力。消融研究验证了每个组件，分析显示InvDec的优势随数据集维度增长而增长，证实了随着变量数量增加，跨变量建模变得关键。


### 论文摘要

Multivariate time series forecasting requires simultaneously modeling temporal patterns and cross-variate dependencies. Channel-independent methods such as PatchTST excel at temporal modeling but ignore variable correlations, while pure variate-attention approaches such as iTransformer sacrifice temporal encoding. We proposeInvDec (Inverted Decoder), a hybrid architecture that achieves principled separation between temporal encoding and variate-level decoding. InvDec combines a patch-based temporal encoder with an inverted decoder operating on the variate dimension through variate-wise self-attention. We introduce delayed variate embeddings that enrich variable-specific representations only after temporal encoding, preserving temporal feature integrity. An adaptive residual fusion mechanism dynamically balances temporal and variate information across datasets of varying dimensions. Instantiating InvDec with PatchTST yields InvDec-PatchTST. Extensive experiments on seven benchmarks demonstrate significant gains on high-dimensional datasets: 20.9% MSE reduction on Electricity (321 variables), 4.3% improvement on Weather, and 2.7% gain on Traffic compared to PatchTST, while maintaining competitive performance on low-dimensional ETT datasets. Ablation studies validate each component, and analysis reveals that InvDec's advantage grows with dataset dimensionality, confirming that cross-variate modeling becomes critical as the number of variables increases.

---

## 22. DMC$^3$: Dual-Modal Counterfactual Contrastive Construction for Egocentric Video Question Answering

**论文链接:** [http://arxiv.org/abs/2510.20285v1](http://arxiv.org/abs/2510.20285v1)

**作者:** Jiayi Zou, Chaofan Chen, Bing-Kun Bao, Changsheng Xu

**发布时间:** 2025-10-23

**DOI:** 10.1145/3746027.3755085

### GPT解析

### 总结

本文提出了一种双模态反事实对比构建（DMC³）框架，用于解决以第一人称视角视频为基础的问题回答任务中的独特挑战，如理解多个事件和识别手部物体交互。

### 背景

以第一人称视角视频为基础的问题回答（Egocentric VideoQA）在以第一人称视频理解中扮演着重要角色。现有的预训练和微调方法忽略了第一人称视角带来的独特挑战，如理解多个事件和识别手部物体交互。

### 目的

为了解决第一人称视角视频理解中的独特挑战，特别是理解多个事件和识别手部物体交互，作者提出了一种新的框架DMC³。

### 方法

DMC³框架包含三个主要部分：开发一个反事实样本构建模块，通过事件描述重述和核心交互挖掘分别为文本和视觉模态生成正负样本；将这些样本与原始样本一起输入基线模型；在反事实样本参与的对比优化模块中应用对比损失，最小化原始样本特征与正样本特征之间的距离，同时最大化与负样本的距离。

### 主要发现

实验结果表明，该方法在EgoTaskQA数据集的normal和indirect分割上分别达到了52.51%和46.04%的性能，在QAEGO4D上达到了13.2%的性能，均达到了最先进的水平。

### 结论

通过提出DMC³框架，有效解决了第一人称视角视频理解中的独特挑战，特别是在理解多个事件和识别手部物体交互方面取得了显著进展，并在多个基准测试中达到了最先进的性能。

### 翻译

以第一人称视频问答（Egocentric VideoQA）在以第一人称视频理解中发挥着重要作用，它指的是基于第一人称视频回答问题。尽管现有方法通过预训练和微调的范式已经取得了进展，但它们忽略了第一人称视角带来的独特挑战，如理解多个事件和识别手部物体交互。为了应对这些挑战，我们提出了一个双模态反事实对比构建（DMC³）框架，该框架包含一个以第一人称视频问答的基线模型、一个反事实样本构建模块和一个反事实样本参与的对比优化。具体来说，我们首先开发了一个反事实样本构建模块，通过事件描述重述和核心交互挖掘分别为文本和视觉模态生成正负样本。然后，我们将这些样本与原始样本一起输入基线模型。最后，在反事实样本参与的对比优化模块中，我们应用对比损失来最小化原始样本特征与正样本特征之间的距离，同时最大化与负样本的距离。实验表明，我们的方法在EgoTaskQA的normal和indirect分割上分别达到了52.51%和46.04%，在QAEGO4D上达到了13.2%，均达到了最先进的性能。


### 论文摘要

Egocentric Video Question Answering (Egocentric VideoQA) plays an important role in egocentric video understanding, which refers to answering questions based on first-person videos. Although existing methods have made progress through the paradigm of pre-training and fine-tuning, they ignore the unique challenges posed by the first-person perspective, such as understanding multiple events and recognizing hand-object interactions. To deal with these challenges, we propose a Dual-Modal Counterfactual Contrastive Construction (DMC$^3$) framework, which contains an egocentric videoqa baseline, a counterfactual sample construction module and a counterfactual sample-involved contrastive optimization. Specifically, We first develop a counterfactual sample construction module to generate positive and negative samples for textual and visual modalities through event description paraphrasing and core interaction mining, respectively. Then, We feed these samples together with the original samples into the baseline. Finally, in the counterfactual sample-involved contrastive optimization module, we apply contrastive loss to minimize the distance between the original sample features and the positive sample features, while maximizing the distance from the negative samples. Experiments show that our method achieve 52.51\% and 46.04\% on the \textit{normal} and \textit{indirect} splits of EgoTaskQA, and 13.2\% on QAEGO4D, both reaching the state-of-the-art performance.

---

## 23. PPMStereo: Pick-and-Play Memory Construction for Consistent Dynamic Stereo Matching

**论文链接:** [http://arxiv.org/abs/2510.20178v1](http://arxiv.org/abs/2510.20178v1)

**作者:** Yun Wang, Junjie Hu, Qiaole Dong, Yongjian Zhang, Yanwei Fu, Tin Lun Lam, Dapeng Wu

**发布时间:** 2025-10-23

### GPT解析

### 总结

本研究提出了一种名为PPMStereo的新方法，通过引入内存缓冲区和两阶段决策过程（选择和播放），实现了从立体视频中估计时间上一致的深度信息，在保持计算效率的同时提高了时空一致性。

### 背景

从立体视频中估计时间上一致的深度信息对增强现实等实际应用至关重要，因为深度估计的不一致会破坏用户沉浸感。然而，这项任务具有挑战性，因为很难以计算高效的方式建模长期的时间一致性。之前的方法在时空建模的广度和计算效率之间存在权衡。

### 目的

本研究旨在解决立体视频中时间一致深度估计的挑战，特别是如何在保持计算效率的同时建模长程时空一致性，开发一种能够实现高效信息聚合且保持深度估计时间一致性的方法。

### 方法

作者提出了一种名为PPMStereo的方法，引入了内存缓冲区用于建模长程时空一致性。受人类两阶段决策过程的启发，PPMStereo包含一个'选择'过程（识别最相关的帧）和一个'播放'过程（为时空聚合自适应地加权所选帧），形成一种两阶段协作过程。

### 主要发现

大量实验验证了PPMStereo的有效性，表明其在准确性和时间一致性方面达到了最先进的性能。在Sintel clean/final数据集上，PPMStereo实现了0.62/1.11 TEPE的性能，比BiDAStereo分别提高了17.3%和9.02%，且计算成本更低。

### 结论

PPMStereo通过创新的内存缓冲区和两阶段决策过程，成功解决了立体视频中时间一致深度估计的挑战，在保持计算效率的同时提高了时空一致性，为增强现实等实际应用提供了更可靠的深度估计技术。

### 翻译

从立体视频中估计时间上一致的深度信息对于增强现实等实际应用至关重要，因为不一致的深度估计会破坏用户的沉浸感。尽管如此，由于难以以计算高效的方式建模长期时间一致性，这项任务仍然具有挑战性。先前的方法试图通过聚合时空信息来解决这一问题，但面临一个基本的权衡：有限的时空建模只能带来适度的提升，而捕捉长程依赖关系则会显著增加计算成本。为了解决这一限制，我们引入了一个内存缓冲区，用于建模长程时空一致性，同时实现高效的动态立体匹配。受人类两阶段决策过程的启发，我们提出了一种用于动态立体匹配的选择并播放记忆(PPM)构建模块，称为PPMStereo。PPM包括一个'选择'过程，用于识别最相关的帧，以及一个'播放'过程，用于为时空聚合自适应地加权所选帧。这种两阶段协作过程保持了一个紧凑但信息丰富的内存缓冲区，同时实现了时间上一致的信息聚合。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决动态场景中的立体匹配问题，即在视频序列中保持深度估计的时间一致性，避免出现闪烁和模糊等不一致现象。这个问题在现实中非常重要，因为像增强现实、自动驾驶和机器人等应用需要时间一致的深度估计来提供稳定可靠的用户体验，而不一致的深度估计会严重影响用户体验和系统性能。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': "作者首先分析了现有方法的局限性：一些方法使用小时间窗口限制了信息传播，而扩大窗口则计算成本过高且不考虑帧可靠性差异。作者借鉴了人类决策的两阶段过程（'选择'和'播放'），并参考了视频任务中的记忆模型（如XMem和RMem），但针对立体匹配任务进行了专门改进。作者设计了一个质量评估模块来评估帧的价值，并采用动态选择机制来构建高效的记忆缓冲区。", '这个方法的核心思想是什么？整体实现流程是怎样的？': "核心思想是引入记忆缓冲区来建模长期时空一致性，同时保持计算效率。方法包含两个关键过程：1）'选择'过程：使用质量评估模块识别最相关的K帧，评估标准包括置信度、冗余性和相似性；2）'播放'过程：通过动态记忆调制机制自适应加权选定帧的特征，并使用注意力机制读取记忆缓冲区。整体流程包括特征提取、成本体积构建、上下文编码、记忆缓冲区更新和迭代细化等步骤，通过GRU模块逐步优化视差估计。", '论文的关键创新点有哪些？相比之前的工作，有什么不同？': "关键创新点包括：1）首次将记忆缓冲区引入动态立体匹配任务，实现高效长期建模；2）提出'选择和播放'记忆构建方法，动态选择并加权关键帧；3）引入质量评估模块联合评估帧的置信度、冗余性和相似性；4）设计动态记忆调制机制自适应调整特征权重。相比之前工作，PPMStereo不再使用固定窗口或平等对待所有帧，而是根据帧质量和相关性动态选择和加权，在保持计算效率的同时显著提高了时间一致性和准确性。", '如果要用一句话总结这篇论文的贡献，你会怎么说？': "PPMStereo通过创新的'选择和播放'记忆构建机制，实现了在计算高效的同时保持时间一致性的动态立体匹配，显著提高了深度估计的准确性和一致性。"}


### 论文摘要

Temporally consistent depth estimation from stereo video is critical for real-world applications such as augmented reality, where inconsistent depth estimation disrupts the immersion of users. Despite its importance, this task remains challenging due to the difficulty in modeling long-term temporal consistency in a computationally efficient manner. Previous methods attempt to address this by aggregating spatio-temporal information but face a fundamental trade-off: limited temporal modeling provides only modest gains, whereas capturing long-range dependencies significantly increases computational cost. To address this limitation, we introduce a memory buffer for modeling long-range spatio-temporal consistency while achieving efficient dynamic stereo matching. Inspired by the two-stage decision-making process in humans, we propose a \textbf{P}ick-and-\textbf{P}lay \textbf{M}emory (PPM) construction module for dynamic \textbf{Stereo} matching, dubbed as \textbf{PPMStereo}. PPM consists of a `pick' process that identifies the most relevant frames and a `play' process that weights the selected frames adaptively for spatio-temporal aggregation. This two-stage collaborative process maintains a compact yet highly informative memory buffer while achieving temporally consistent information aggregation. Extensive experiments validate the effectiveness of PPMStereo, demonstrating state-of-the-art performance in both accuracy and temporal consistency. % Notably, PPMStereo achieves 0.62/1.11 TEPE on the Sintel clean/final (17.3\% \& 9.02\% improvements over BiDAStereo) with fewer computational costs. Codes are available at \textcolor{blue}{https://github.com/cocowy1/PPMStereo}.

---

## 24. Abstain Mask Retain Core: Time Series Prediction by Adaptive Masking Loss with Representation Consistency

**论文链接:** [http://arxiv.org/abs/2510.19980v1](http://arxiv.org/abs/2510.19980v1)

**作者:** Renzhao Liang, Sizhe Xu, Chenggang Xie, Jingru Chen, Feiyang Ren, Shu Yang, Takahiro Yabe

**发布时间:** 2025-10-22

**备注:** 20 pages, 4 figures. Accepted as Spotlight poster in NeurIPS 2025

### GPT解析

### 总结

该研究针对时间序列预测中的冗余特征学习问题，提出了一种名为AMRC的创新解决方案，通过动态掩码损失和表示一致性约束提高了预测性能，挑战了传统的时间序列建模假设。

### 背景

时间序列预测在能源管理和金融市场等关键领域发挥着重要作用。尽管基于深度学习的方法（如MLP、RNN、Transformer）已取得显著进展，但现有的'长序列信息增益假设'存在固有局限性。

### 目的

研究旨在解决现有模型在训练过程中学习大量冗余特征（如噪声或不相关波动）的问题，从而影响有效信号提取，提高预测准确性。

### 方法

提出了一种名为'具有表示一致性的自适应掩码损失'（AMRC）的创新解决方案，包含两个核心组件：1) 动态掩码损失，自适应识别高判别性时间段以指导梯度下降；2) 表示一致性约束，稳定输入、标签和预测之间的映射关系。

### 主要发现

通过系统实验发现了一个反直觉现象：适当截断历史数据可以 paradoxically 提高预测准确性，表明现有模型在训练过程中学习了大量冗余特征，损害了有效信号的提取。

### 结论

AMRC能有效抑制冗余特征学习，同时显著提高模型性能。这项工作不仅挑战了时间建模中的传统假设，还为开发高效和稳健的预测模型提供了新的理论见解和方法突破。

### 翻译

时间序列预测在能源管理和金融市场等关键领域发挥着关键作用。尽管基于深度学习的方法（如MLP、RNN、Transformer）已取得显著进展，但现有的'长序列信息增益假设'存在固有局限性。通过系统实验，本研究揭示了一个反直觉现象：适当截断历史数据可以 paradoxically 提高预测准确性，表明现有模型在训练过程中学习了大量冗余特征（如噪声或不相关波动），从而损害了有效信号的提取。基于信息瓶颈理论，我们提出了一种名为'具有表示一致性的自适应掩码损失'（AMRC）的创新解决方案，包含两个核心组件：1) 动态掩码损失，在模型训练过程中自适应识别高判别性时间段以指导梯度下降；2) 表示一致性约束，稳定输入、标签和预测之间的映射关系。实验结果表明，AMRC能有效抑制冗余特征学习，同时显著提高模型性能。这项工作不仅挑战了时间建模中的传统假设，还为开发高效和稳健的预测模型提供了新的理论见解和方法突破。


### 论文摘要

Time series forecasting plays a pivotal role in critical domains such as energy management and financial markets. Although deep learning-based approaches (e.g., MLP, RNN, Transformer) have achieved remarkable progress, the prevailing "long-sequence information gain hypothesis" exhibits inherent limitations. Through systematic experimentation, this study reveals a counterintuitive phenomenon: appropriately truncating historical data can paradoxically enhance prediction accuracy, indicating that existing models learn substantial redundant features (e.g., noise or irrelevant fluctuations) during training, thereby compromising effective signal extraction. Building upon information bottleneck theory, we propose an innovative solution termed Adaptive Masking Loss with Representation Consistency (AMRC), which features two core components: 1) Dynamic masking loss, which adaptively identified highly discriminative temporal segments to guide gradient descent during model training; 2) Representation consistency constraint, which stabilized the mapping relationships among inputs, labels, and predictions. Experimental results demonstrate that AMRC effectively suppresses redundant feature learning while significantly improving model performance. This work not only challenges conventional assumptions in temporal modeling but also provides novel theoretical insights and methodological breakthroughs for developing efficient and robust forecasting models.

---

