# 今日论文推荐 - 2025-12-30

共 105 篇论文

---

## 1. 论文ID: 2512.23441v1

**错误:** 无法读取论文信息 - [Errno 2] No such file or directory: 'storage/result/2512.23441v1.json'

---

## 2. 论文ID: 2512.23413v1

**错误:** 无法读取论文信息 - [Errno 2] No such file or directory: 'storage/result/2512.23413v1.json'

---

## 3. 论文ID: 2512.23141v1

**错误:** 无法读取论文信息 - [Errno 2] No such file or directory: 'storage/result/2512.23141v1.json'

---

## 4. QSAR-Guided Generative Framework for the Discovery of Synthetically Viable Odorants

**论文链接:** [http://arxiv.org/abs/2512.23080v1](http://arxiv.org/abs/2512.23080v1)

**作者:** Tim C. Pearce, Ahmed Ibrahim

**发布时间:** 2025-12-28

### GPT解析

### 总结

研究提出了一种结合变分自编码器(VAE)和定量结构-活性关系(QSAR)模型的框架，可以从有限的气味分子训练集中生成新型气味分子，解决了生成式人工智能通常需要大量分子进行学习的问题。

### 背景

新型气味分子的发现对香水和风味行业至关重要，但在庞大的化学空间中高效筛选具有理想嗅觉特性的结构仍然是一个重大挑战。

### 目的

开发一种方法，利用有限的训练数据生成具有理想嗅觉特性的新型气味分子结构。

### 方法

结合变分自编码器(VAE)和定量结构-活性关系(QSAR)模型的框架。VAE利用自监督学习能力从ChemBL数据库学习SMILES语法，并通过外部QSAR模型导出的损失项来增强训练目标，根据气味概率构建潜在表示。

### 主要发现

模型生成了语法有效的结构(100%有效性)和94.8%的独特结构；潜在空间根据气味可能性有效构建，生成分子与已知气味分子之间的Fréchet ChemNet距离(FCD)约为6.96，而ChemBL基线约为21.6；74.4%的候选物具有不同于训练数据的新核心框架，表明模型进行了广泛的化学空间探索。

### 结论

该框架成功利用有限的训练数据生成了新型气味分子，有效探索了化学空间，为香料和风味行业提供了有价值的分子设计工具。

### 翻译

新型气味分子的发现对香水和风味行业至关重要，然而，在庞大的化学空间中高效导航以识别具有理想嗅觉特性的结构仍然是一个重大挑战。生成式人工智能为从头分子设计提供了有前景的方法，但通常需要大量分子进行学习。为解决这个问题，我们提出了一种结合变分自编码器(VAE)和定量结构-活性关系(QSAR)模型的框架，可以从有限的气味分子训练集中生成新型气味分子。VAE的自监督学习能力使其能够从ChemBL数据库学习SMILES语法，同时其训练目标通过从外部QSAR模型导出的损失项得到增强，以根据气味概率构建潜在表示。虽然VAE在学习QSAR监督信号方面表现出高度内部一致性，但针对外部未见过的真实数据集(Unique Good Scents)的验证确认，模型生成了语法有效的结构(通过拒绝采样实现了100%有效性)和94.8%的独特结构。潜在空间根据气味可能性得到有效构建，生成分子与已知气味分子之间的Fréchet ChemNet距离(FCD)约为6.96，而ChemBL基线约为21.6。通过Bemis-Murcko骨架进行的结构分析显示，74.4%的候选物具有不同于训练数据的新核心框架，表明模型在已知气味分子的简单衍生之外进行了广泛的化学空间探索。生成的候选物表现出物理化学特性...


### 论文摘要

The discovery of novel odorant molecules is key for the fragrance and flavor industries, yet efficiently navigating the vast chemical space to identify structures with desirable olfactory properties remains a significant challenge. Generative artificial intelligence offers a promising approach for \textit{de novo} molecular design but typically requires large sets of molecules to learn from. To address this problem, we present a framework combining a variational autoencoder (VAE) with a quantitative structure-activity relationship (QSAR) model to generate novel odorants from limited training sets of odor molecules. The self-supervised learning capabilities of the VAE allow it to learn SMILES grammar from ChemBL database, while its training objective is augmented with a loss term derived from an external QSAR model to structure the latent representation according to odor probability. While the VAE demonstrated high internal consistency in learning the QSAR supervision signal, validation against an external, unseen ground truth dataset (Unique Good Scents) confirms the model generates syntactically valid structures (100\% validity achieved via rejection sampling) and 94.8\% unique structures. The latent space is effectively structured by odor likelihood, evidenced by a Fréchet ChemNet Distance (FCD) of $\approx$ 6.96 between generated molecules and known odorants, compared to $\approx$ 21.6 for the ChemBL baseline. Structural analysis via Bemis-Murcko scaffolds reveals that 74.4\% of candidates possess novel core frameworks distinct from the training data, indicating the model performs extensive chemical space exploration beyond simple derivatization of known odorants. Generated candidates display physicochemical properties ....

---

## 5. 论文ID: 2512.23076v1

**错误:** 无法读取论文信息 - [Errno 2] No such file or directory: 'storage/result/2512.23076v1.json'

---

## 6. 3D sans 3D Scans: Scalable Pre-training from Video-Generated Point Clouds

**论文链接:** [http://arxiv.org/abs/2512.23042v1](http://arxiv.org/abs/2512.23042v1)

**作者:** Ryousuke Yamada, Kohsuke Ide, Yoshihiro Fukuhara, Hirokatsu Kataoka, Gilles Puy, Andrei Bursuc, Yuki M. Asano

**发布时间:** 2025-12-28

### GPT解析

### 总结

本文提出了一种名为LAM3C的自监督框架，可以从无标签视频中学习3D表示，无需真实3D传感器。通过构建RoomTours视频生成点云数据集和引入噪声正则化损失，该方法在室内语义和实例分割任务上超越了现有自监督方法的性能。

### 背景

尽管3D自监督学习取得了进展，但收集大规模3D场景扫描仍然昂贵且耗时。

### 目的

研究是否可以从没有真实3D传感器记录的无标签视频中学习3D表示。

### 方法

提出LAM3C（基于Sinkhorn-Knopp的拉普拉斯感知多级3D聚类）自监督框架，从视频生成的点云中学习。构建RoomTours数据集，通过网络收集房间漫游视频并使用现成的前馈重建模型生成49,219个场景。同时提出噪声正则化损失，强制局部几何平滑性和确保在噪声点云下的特征稳定性。

### 主要发现

在没有使用任何真实3D扫描的情况下，LAM3C在室内语义和实例分割任务上表现优于之前的自监督方法。

### 结论

无标签视频是3D自监督学习的丰富数据来源。

### 翻译

尽管最近在3D自监督学习方面取得了进展，但收集大规模3D场景扫描仍然昂贵且耗时。在这项工作中，我们研究了是否可以从没有使用任何真实3D传感器记录的无标签视频中学习3D表示。我们提出了LAM3C（拉普拉斯感知多级3D聚类与Sinkhorn-Knopp），这是一个从无标签视频生成的点云中学习的自监督框架。我们首先引入了RoomTours，这是一个通过从网络上收集房间漫游视频（如房地产旅游）并使用现成的前馈重建模型生成49,219个场景而构建的视频生成点云数据集。我们还提出了一种噪声正则化损失，通过强制局部几何平滑性和确保在噪声点云下的特征稳定性来稳定表示学习。值得注意的是，在没有使用任何真实3D扫描的情况下，LAM3C在室内语义和实例分割上的性能优于先前的自监督方法。这些结果表明，无标签视频是3D自监督学习的丰富数据来源。


### 论文摘要

Despite recent progress in 3D self-supervised learning, collecting large-scale 3D scene scans remains expensive and labor-intensive. In this work, we investigate whether 3D representations can be learned from unlabeled videos recorded without any real 3D sensors. We present Laplacian-Aware Multi-level 3D Clustering with Sinkhorn-Knopp (LAM3C), a self-supervised framework that learns from video-generated point clouds from unlabeled videos. We first introduce RoomTours, a video-generated point cloud dataset constructed by collecting room-walkthrough videos from the web (e.g., real-estate tours) and generating 49,219 scenes using an off-the-shelf feed-forward reconstruction model. We also propose a noise-regularized loss that stabilizes representation learning by enforcing local geometric smoothness and ensuring feature stability under noisy point clouds. Remarkably, without using any real 3D scans, LAM3C achieves higher performance than the previous self-supervised methods on indoor semantic and instance segmentation. These results suggest that unlabeled videos represent an abundant source of data for 3D self-supervised learning.

---

## 7. Learning Anatomy from Multiple Perspectives via Self-supervision in Chest Radiographs

**论文链接:** [http://arxiv.org/abs/2512.22872v1](http://arxiv.org/abs/2512.22872v1)

**作者:** Ziyu Zhou, Haozhe Luo, Mohammad Reza Hosseinzadeh Taher, Jiaxuan Pang, Xiaowei Ding, Michael B. Gotway, Jianming Liang

**发布时间:** 2025-12-28

### GPT解析

### 总结

该研究提出了一种名为Lamps的自监督学习方法，通过从多角度学习人体解剖结构，在胸部X光影像上预训练基础模型，显著提升了模型的鲁棒性、可转移性和临床应用潜力。

### 背景

基础模型在自然语言处理和计算机视觉领域取得成功，因为它们能捕捉自然语言的基础结构。然而，在医学影像领域，基础在于人体解剖学，因为医学影像直接代表人体内部结构，反映了人体解剖的一致性、连贯性和层次性。现有的自监督学习方法往往忽视这些视角，限制了它们有效学习解剖特征的能力。

### 目的

为了克服现有方法的局限性，研究旨在构建一个能从多角度学习人体解剖结构的基础模型，并将其应用于大规模胸部X光影像的预训练。

### 方法

研究团队构建了名为Lamps（通过自监督从多角度学习解剖学）的预训练模型，通过和谐地利用人体解剖的一致性、连贯性和层次性作为监督信号，在大规模胸部X光影像上进行预训练。

### 主要发现

在10个数据集上进行的广泛实验，通过微调和涌现特性分析表明，与10个基线模型相比，Lamps具有更强的鲁棒性、可转移性和临床应用潜力。

### 结论

通过从多角度学习，Lamps为基础模型提供了独特的机会，使其能够开发出与人体解剖结构对齐的、有意义且鲁棒的表示。

### 翻译

基础模型在自然语言处理和计算机视觉中取得成功，因为它们能够捕捉自然语言的基础结构。然而，在医学影像中，关键基础在于人体解剖学，因为这些图像直接代表人体内部结构，反映了人体解剖的一致性、连贯性和层次性。然而，现有的自监督学习方法往往忽视了这些视角，限制了它们有效学习解剖特征的能力。为了克服这一局限性，我们构建了Lamps（通过自监督从多角度学习解剖学），在大规模胸部X光影像上进行预训练，通过和谐地利用人体解剖的一致性、连贯性和层次性作为监督信号。在10个数据集上进行的广泛实验，通过微调和涌现特性分析表明，与10个基线模型相比，Lamps具有更强的鲁棒性、可转移性和临床应用潜力。通过从多角度学习，Lamps为基础模型提供了独特的机会，使其能够开发出与人体解剖结构对齐的、有意义且鲁棒的表示。


### 论文摘要

Foundation models have been successful in natural language processing and computer vision because they are capable of capturing the underlying structures (foundation) of natural languages. However, in medical imaging, the key foundation lies in human anatomy, as these images directly represent the internal structures of the body, reflecting the consistency, coherence, and hierarchy of human anatomy. Yet, existing self-supervised learning (SSL) methods often overlook these perspectives, limiting their ability to effectively learn anatomical features. To overcome the limitation, we built Lamps (learning anatomy from multiple perspectives via self-supervision) pre-trained on large-scale chest radiographs by harmoniously utilizing the consistency, coherence, and hierarchy of human anatomy as the supervision signal. Extensive experiments across 10 datasets evaluated through fine-tuning and emergent property analysis demonstrate Lamps' superior robustness, transferability, and clinical potential when compared to 10 baseline models. By learning from multiple perspectives, Lamps presents a unique opportunity for foundation models to develop meaningful, robust representations that are aligned with the structure of human anatomy.

---

## 8. Semantic contrastive learning for orthogonal X-ray computed tomography reconstruction

**论文链接:** [http://arxiv.org/abs/2512.22674v1](http://arxiv.org/abs/2512.22674v1)

**作者:** Jiashu Dong, Jiabing Xiang, Lisheng Geng, Suqing Tian, Wei Zhao

**发布时间:** 2025-12-27

**备注:** This paper is accepted by Fully3D 2025

### GPT解析

### 总结

本文提出了一种基于语义特征对比学习的新型损失函数，用于解决X射线CT稀疏视图重建中的条纹伪影问题，通过三阶段U-Net架构实现了高质量的重建效果和低计算复杂度。

### 背景

X射线CT在医学影像中广泛应用，稀疏视图重建可减少辐射剂量，但不适定条件常导致严重条纹伪影。基于深度学习的方法虽有改进但仍面临挑战。

### 目的

解决现有CT重建方法的挑战，提高重建质量，降低计算复杂度，为正交CT重建提供实用解决方案。

### 方法

提出新颖的语义特征对比学习损失函数，评估高级潜在空间中的语义相似性和浅层潜在空间中的解剖相似性。采用三阶段U-Net架构：粗略重建、细节细化、语义相似性测量。

### 主要发现

在胸部数据集上的测试表明，该方法相比其他算法实现了更好的重建质量和更快的处理速度，图像质量显著改善同时保持低计算复杂度。

### 结论

该方法是一种实用的正交CT重建解决方案。

### 翻译

X射线计算机断层扫描(CT)在医学影像中广泛应用，稀疏视图重建是减少辐射剂量的有效方法。然而，不适定条件通常会导致严重的条纹伪影。基于深度学习的最新方法已提高重建质量，但挑战仍然存在。为应对这些挑战，我们提出了一种新颖的语义特征对比学习损失函数，评估高级潜在空间中的语义相似性和浅层潜在空间中的解剖相似性。我们的方法采用基于三阶段U-Net的架构：一个用于粗略重建，一个用于细节细化，一个用于语义相似性测量。在具有正交投影的胸部数据集上的测试表明，我们的方法相比其他算法实现了更好的重建质量和更快的处理速度。结果显示图像质量有显著改善，同时保持低计算复杂度，使其成为正交CT重建的实用解决方案。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决正交CT重建中的稀疏视图重建问题，减少辐射剂量的同时避免严重条纹伪影。这个问题很重要，因为传统CT需要大量X射线投影采样，增加辐射剂量和成像时间，而减少投影数量会引入重建挑战，尤其在正交投影场景下问题更复杂。高质量重建图像对临床诊断、放射治疗等应用至关重要，同时减少辐射剂量能提高患者安全性。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先分析了现有方法（GAN和扩散模型）的局限性：GAN存在训练不稳定、模式崩溃和幻觉特征问题，扩散模型则训练收敛慢、计算开销大。然后结合语义对比学习改进重建质量，同时保持计算效率。他们设计了三阶段U-Net架构，借鉴了GAN框架、对比学习方法、U-Net架构以及已有的正交投影重建研究，如X2CT-GAN，并创新性地加入了语义对比学习损失函数。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是使用语义对比学习损失函数评估高层语义相似性和低层解剖相似性，采用从粗糙到精细的重建框架，并通过语义一致性约束减少幻觉伪影。整体流程分三阶段：1）粗略重建阶段，通过几何关系将正交投影反投影，用3D U-Net生成基本解剖结构；2）细节精炼阶段，用2D U-Net处理每个切片，结合多种损失函数增强细节；3）语义相似性测量阶段，通过学生-教师模型对比重建图像与原始图像的语义相似性，使用正样本计算MSE损失，负样本计算InfoNCE损失。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1）新的语义特征对比学习损失函数；2）三阶段U-Net架构；3）训练使用三个U-Net，推理只用两个，提高效率；4）通过语义一致性约束减少幻觉伪影；5）结合多种损失函数。相比之前工作，与传统方法相比能在极稀疏投影下工作；与GAN方法相比减少幻觉伪影，提高训练稳定性；与扩散模型相比显著提高重建速度（低于0.4秒）；与其他深度学习方法相比结合语义信息提高解剖准确性，在多种评估指标上表现更优，更适合临床应用。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '本文提出了一种基于语义对比学习的正交CT重建方法，通过创新的损失函数设计和高效的三阶段网络架构，在减少辐射剂量的同时实现了高质量、快速且稳定的医学图像重建，解决了现有方法中存在的伪影、计算效率和训练稳定性问题。'}


### 论文摘要

X-ray computed tomography (CT) is widely used in medical imaging, with sparse-view reconstruction offering an effective way to reduce radiation dose. However, ill-posed conditions often result in severe streak artifacts. Recent advances in deep learning-based methods have improved reconstruction quality, but challenges still remain. To address these challenges, we propose a novel semantic feature contrastive learning loss function that evaluates semantic similarity in high-level latent spaces and anatomical similarity in shallow latent spaces. Our approach utilizes a three-stage U-Net-based architecture: one for coarse reconstruction, one for detail refinement, and one for semantic similarity measurement. Tests on a chest dataset with orthogonal projections demonstrate that our method achieves superior reconstruction quality and faster processing compared to other algorithms. The results show significant improvements in image quality while maintaining low computational complexity, making it a practical solution for orthogonal CT reconstruction.

---

## 9. SPECTRE: Spectral Pre-training Embeddings with Cylindrical Temporal Rotary Position Encoding for Fine-Grained sEMG-Based Movement Decoding

**论文链接:** [http://arxiv.org/abs/2512.22481v1](http://arxiv.org/abs/2512.22481v1)

**作者:** Zihan Weng, Chanlin Yi, Pouya Bashivan, Jing Lu, Fali Li, Dezhong Yao, Jingming Hou, Yangsong Zhang, Peng Xu

**发布时间:** 2025-12-27

### GPT解析

### 总结

SPECTRE是一种针对sEMG信号的领域特定自监督学习框架，通过基于生理学的预训练任务和圆柱旋转位置编码，显著提升了从非侵入性表面肌电图解码精细运动的能力。

### 背景

从非侵入性表面肌电图(sEMG)解码精细运动对于假肢控制是一个挑战，主要来自信号的非平稳性和低信噪比。通用自监督学习框架在sEMG上通常效果不佳，因为它们试图重建嘈杂的原始信号，并且缺乏对电极阵列圆柱拓扑的归纳偏置。

### 目的

克服通用SSL框架在sEMG信号处理上的局限性，开发一个针对sEMG领域的特定SSL框架，以提高运动解码的准确性。

### 方法

提出了SPECTRE框架，包含两个主要贡献：1)基于生理学的预训练任务，使用聚类短时傅里叶变换表示进行离散伪标签的掩码预测；2)圆柱旋转位置嵌入(CyRoPE)，沿线性时间和环形空间维度分解嵌入，明确建模前臂传感器拓扑以捕获肌肉协同作用。

### 主要发现

在多个数据集上的评估表明，SPECTRE建立了运动解码的新技术水平，显著优于监督基线和通用SSL方法。消融研究验证了频谱预训练和CyRoPE的关键作用。

### 结论

SPECTRE为能够处理现实世界sEMG复杂性的实用肌电接口提供了稳健的基础。

### 翻译

从非侵入性表面肌电图(sEMG)解码精细运动对于假肢控制是一个挑战，这源于信号的非平稳性和低信噪比。通用的自监督学习(SSL)框架在sEMG上通常产生次优结果，因为它们试图重建嘈杂的原始信号，并且缺乏对电极阵列圆柱拓扑的归纳偏置。为了克服这些局限性，我们引入了SPECTRE，一个领域特定的SSL框架。SPECTRE有两个主要贡献：一个基于生理学的预训练任务和一个新颖的位置编码。预训练涉及从聚类的短时傅里叶变换(STFT)表示中预测离散伪标签，迫使模型学习稳健的、生理相关的频率模式。此外，我们的圆柱旋转位置嵌入(CyRoPE)沿线性时间和环形空间维度分解嵌入，明确建模前臂传感器拓扑以捕获肌肉协同作用。在多个数据集上的评估，包括来自截肢者的具有挑战性的数据，表明SPECTRE为运动解码建立了新的技术水平，显著优于监督基线和通用SSL方法。消融研究验证了频谱预训练和CyRoPE的关键作用。SPECTRE为能够处理现实世界sEMG复杂性的实用肌电接口提供了稳健的基础。


### 论文摘要

Decoding fine-grained movement from non-invasive surface Electromyography (sEMG) is a challenge for prosthetic control due to signal non-stationarity and low signal-to-noise ratios. Generic self-supervised learning (SSL) frameworks often yield suboptimal results on sEMG as they attempt to reconstruct noisy raw signals and lack the inductive bias to model the cylindrical topology of electrode arrays. To overcome these limitations, we introduce SPECTRE, a domain-specific SSL framework. SPECTRE features two primary contributions: a physiologically-grounded pre-training task and a novel positional encoding. The pre-training involves masked prediction of discrete pseudo-labels from clustered Short-Time Fourier Transform (STFT) representations, compelling the model to learn robust, physiologically relevant frequency patterns. Additionally, our Cylindrical Rotary Position Embedding (CyRoPE) factorizes embeddings along linear temporal and annular spatial dimensions, explicitly modeling the forearm sensor topology to capture muscle synergies. Evaluations on multiple datasets, including challenging data from individuals with amputation, demonstrate that SPECTRE establishes a new state-of-the-art for movement decoding, significantly outperforming both supervised baselines and generic SSL approaches. Ablation studies validate the critical roles of both spectral pre-training and CyRoPE. SPECTRE provides a robust foundation for practical myoelectric interfaces capable of handling real-world sEMG complexities.

---

## 10. Multi-Head Spectral-Adaptive Graph Anomaly Detection

**论文链接:** [http://arxiv.org/abs/2512.22291v1](http://arxiv.org/abs/2512.22291v1)

**作者:** Qingyue Cao, Bo Jin, Changwei Gong, Xin Tong, Wenzheng Li, Xiaodong Zhou

**发布时间:** 2025-12-25

### GPT解析

### 总结

论文提出了一种多头谱自适应图神经网络（MHSA-GNN），用于解决金融欺诈和风险控制中的复杂异常模式检测问题，通过动态生成定制化滤波器参数和双重正则化策略，有效保留了高频异常信号并提高了检测性能。

### 背景

图异常检测技术在金融欺诈和风险控制中有广泛应用，但现有方法在处理复杂多变的异常模式时面临挑战。异常节点常常伪装并与正常节点混合，导致图中同时存在同质性和异质性。现有谱图神经网络虽取得进展，但通常采用固定、全局共享的滤波器，这种'一刀切'的方法容易导致过平滑，消除欺诈检测所需的关键高频信号，且对不同图实例缺乏自适应能力。

### 目的

解决现有图异常检测方法在处理复杂异常模式时的局限性，特别是解决固定全局滤波器导致的过平滑问题和缺乏自适应能力的问题，从而更好地保留高频异常信号并提高检测性能。

### 方法

提出多头谱自适应图神经网络（MHSA-GNN），核心创新是设计了一个轻量级超网络，该网络基于包含结构统计量和瑞利商特征的'谱指纹'，动态生成针对每个实例定制的切比雪夫滤波器参数，使每个节点及其局部子图能够采用定制化过滤策略。此外，引入双重正则化策略，结合教师-学生对比学习（TSC）确保表示准确性，以及Barlow Twins多样性损失（BTD）强制各头部之间正交。

### 主要发现

在四个真实世界数据集上的广泛实验表明，该方法能有效保留高频异常信号，并显著优于现有最先进的方法，特别是在高度异构数据集上表现出卓越的鲁棒性。

### 结论

MHSA-GNN通过动态自适应的滤波策略和双重正则化机制，有效解决了传统图异常检测方法在复杂场景下的局限性，特别是在金融欺诈检测等需要保留高频异常信号的应用中表现出色。

### 翻译

图异常检测技术在金融欺诈和风险控制中有广泛应用。然而，当处理复杂多变的异常模式时，现有的图异常检测方法往往面临重大挑战，因为异常节点常常伪装并与正常节点混合，导致图中同时存在同质性和异质性。最近的谱图神经网络在解决这个问题方面取得了显著进展；然而，当前技术通常采用固定、全局共享的滤波器。这种'一刀切'的方法容易导致过平滑，消除欺诈检测所需的关键高频信号，并且对不同图实例缺乏自适应能力。为了解决这个问题，我们提出了一种多头谱自适应图神经网络（MHSA-GNN）。核心创新是设计了一个轻量级超网络，该网络基于包含结构统计量和瑞利商特征的'谱指纹'，动态生成针对每个实例定制的切比雪夫滤波器参数。这使得每个节点及其局部子图能够采用定制化的过滤策略。此外，为防止多head机制中的模式崩溃，我们引入了一种新颖的双重正则化策略，结合教师-学生对比学习（TSC）确保表示准确性，以及Barlow Twins多样性损失（BTD）强制各头部之间正交。在四个真实世界数据集上的广泛实验表明，我们的方法能有效保留高频异常信号，并显著优于现有的最先进方法，特别是在高度异构数据集上表现出卓越的鲁棒性。


### 论文摘要

Graph anomaly detection technology has broad applications in financial fraud and risk control. However, existing graph anomaly detection methods often face significant challenges when dealing with complex and variable abnormal patterns, as anomalous nodes are often disguised and mixed with normal nodes, leading to the coexistence of homophily and heterophily in the graph domain. Recent spectral graph neural networks have made notable progress in addressing this issue; however, current techniques typically employ fixed, globally shared filters. This 'one-size-fits-all' approach can easily cause over-smoothing, erasing critical high-frequency signals needed for fraud detection, and lacks adaptive capabilities for different graph instances. To solve this problem, we propose a Multi-Head Spectral-Adaptive Graph Neural Network (MHSA-GNN). The core innovation is the design of a lightweight hypernetwork that, conditioned on a 'spectral fingerprint' containing structural statistics and Rayleigh quotient features, dynamically generates Chebyshev filter parameters tailored to each instance. This enables a customized filtering strategy for each node and its local subgraph. Additionally, to prevent mode collapse in the multi-head mechanism, we introduce a novel dual regularization strategy that combines teacher-student contrastive learning (TSC) to ensure representation accuracy and Barlow Twins diversity loss (BTD) to enforce orthogonality among heads. Extensive experiments on four real-world datasets demonstrate that our method effectively preserves high-frequency abnormal signals and significantly outperforms existing state-of-the-art methods, especially showing excellent robustness on highly heterogeneous datasets.

---

## 11. UniTacHand: Unified Spatio-Tactile Representation for Human to Robotic Hand Skill Transfer

**论文链接:** [http://arxiv.org/abs/2512.21233v3](http://arxiv.org/abs/2512.21233v3)

**作者:** Chi Zhang, Penglin Cai, Haoqi Yuan, Chaoyi Xu, Zongqing Lu

**发布时间:** 2025-12-24

**备注:** The first two authors contributed equally

### GPT解析

### 总结

研究提出了一种统一触觉表示方法UniTacHand，解决了人类与机器人触觉数据不对齐的问题，实现了从人类到机器人的零样本触觉策略转移，提高了数据效率。

### 背景

触觉感知对机器人手实现人类灵巧操作至关重要，特别是在视觉遮挡场景下，但应用常受限于难以收集大规模真实世界机器人触觉数据。

### 目的

提出使用触觉手套收集低成本人类操作数据用于机器人策略学习，并解决人类与机器人触觉数据不对齐导致的策略转移困难问题。

### 方法

提出UniTacHand统一表示方法，将人类和机器人触觉信号投影到MANO手模型的2D表面空间，标准化数据结构并嵌入空间上下文；然后引入对比学习方法，使用仅10分钟配对数据将它们对齐到统一潜在空间。

### 主要发现

该方法实现了从人类到真实机器人的零样本触觉策略转移，能泛化到预训练数据中未见过的物体；在混合数据（人类和机器人演示）上共同训练比仅使用机器人数据获得更好的性能和数据效率。

### 结论

UniTacHand为基于触觉的灵巧手实现通用、可扩展和数据高效的学习铺平了道路。

### 翻译

触觉感知对机器人手实现人类灵巧操作水平至关重要，特别是在视觉遮挡场景中。然而，其应用常因难以收集大规模真实世界机器人触觉数据而受到阻碍。在本研究中，我们提出使用触觉手套收集低成本人类操作数据，用于基于触觉的机器人策略学习。人类与机器人触觉数据之间的不对齐使得从人类数据学习到的策略难以转移到机器人上。为弥合这一差距，我们提出了UniTacHand，一种统一表示方法，用于对齐灵巧手捕获的机器人触觉信息与手套获取的人类手触觉信息。首先，我们将人类手和机器人手的触觉信号投影到MANO手模型的形态一致的2D表面空间上。这种统一标准化了异构数据结构，并将空间上下文固有地嵌入触觉信号中。然后，我们引入对比学习方法，将它们对齐到统一的潜在空间，仅使用我们数据收集系统中10分钟的配对数据进行训练。我们的方法实现了从人类到真实机器人的零样本基于触觉的策略转移，泛化到预训练数据中未见过的物体。我们还证明，通过UniTacHand在混合数据（包括人类和机器人演示）上进行共同训练，比仅使用机器人数据获得更好的性能和数据效率。UniTacHand为基于触觉的灵巧手实现通用、可扩展和数据高效的学习铺平了道路。


### 论文摘要

Tactile sensing is crucial for robotic hands to achieve human-level dexterous manipulation, especially in scenarios with visual occlusion. However, its application is often hindered by the difficulty of collecting large-scale real-world robotic tactile data. In this study, we propose to collect low-cost human manipulation data using haptic gloves for tactile-based robotic policy learning. The misalignment between human and robotic tactile data makes it challenging to transfer policies learned from human data to robots. To bridge this gap, we propose UniTacHand, a unified representation to align robotic tactile information captured by dexterous hands with human hand touch obtained from gloves. First, we project tactile signals from both human hands and robotic hands onto a morphologically consistent 2D surface space of the MANO hand model. This unification standardizes the heterogeneous data structures and inherently embeds the tactile signals with spatial context. Then, we introduce a contrastive learning method to align them into a unified latent space, trained on only 10 minutes of paired data from our data collection system. Our approach enables zero-shot tactile-based policy transfer from humans to a real robot, generalizing to objects unseen in the pre-training data. We also demonstrate that co-training on mixed data, including both human and robotic demonstrations via UniTacHand, yields better performance and data efficiency compared with using only robotic data. UniTacHand paves a path toward general, scalable, and data-efficient learning for tactile-based dexterous hands.

---

## 12. MCI-Net: A Robust Multi-Domain Context Integration Network for Point Cloud Registration

**论文链接:** [http://arxiv.org/abs/2512.23472v1](http://arxiv.org/abs/2512.23472v1)

**作者:** Shuyuan Lin, Wenwu Peng, Junjie Huang, Qiang Qi, Miaohui Wang, Jian Weng

**发布时间:** 2025-12-29

### GPT解析

### 总结

论文提出了一种名为MCI-Net的多领域上下文集成网络，用于改进点云配准中的特征学习，通过从不同领域聚合上下文线索提高特征表示和配准性能。

### 背景

现有的基于深度学习的点云配准方法通常依赖基于欧几里得邻域的特征提取策略，这些方法难以有效捕捉点云中的隐式语义和结构一致性。

### 目的

解决现有方法无法有效捕捉点云隐式语义和结构一致性的问题，提高特征表示的鲁棒性和判别性。

### 方法

1) 提出图邻域聚合模块，构建全局图捕获点云内的整体结构关系；2) 提出渐进式上下文交互模块，通过域内特征解耦和域间上下文交互增强特征判别性；3) 设计动态内点选择方法，利用多次姿态估计迭代的残差信息优化内点权重。

### 主要发现

在室内RGB-D和室外LiDAR数据集上的实验表明，MCI-Net显著优于现有最先进方法，在3DMatch上实现了96.4%的最高配准召回率。

### 结论

MCI-Net通过多领域上下文集成有效改进了点云配准性能，特别是在捕捉点云的隐式语义和结构一致性方面表现出色。

### 翻译

鲁棒且具有判别性的特征学习对于高质量的点云配准至关重要。然而，现有的基于深度学习的方法通常依赖于基于欧几里得邻域的特征提取策略，这些策略难以有效捕捉点云中的隐式语义和结构一致性。为解决这些问题，我们提出了多领域上下文集成网络(MCI-Net)，通过从不同领域聚合上下文线索来改进特征表示和配准性能。具体来说，我们提出了一个图邻域聚合模块，构建全局图以捕获点云内的整体结构关系。然后，我们提出了一个渐进式上下文交互模块，通过执行域内特征解耦和域间上下文交互来增强特征判别性。最后，我们设计了一种动态内点选择方法，利用多次姿态估计迭代的残差信息优化内点权重，从而提高配准的准确性和鲁棒性。在室内RGB-D和室外LiDAR数据集上的大量实验表明，所提出的MCI-Net显著优于现有的最先进方法，在3DMatch上实现了96.4%的最高配准召回率。源代码可在http://www.linshuyuan.com获取。


### 论文摘要

Robust and discriminative feature learning is critical for high-quality point cloud registration. However, existing deep learning-based methods typically rely on Euclidean neighborhood-based strategies for feature extraction, which struggle to effectively capture the implicit semantics and structural consistency in point clouds. To address these issues, we propose a multi-domain context integration network (MCI-Net) that improves feature representation and registration performance by aggregating contextual cues from diverse domains. Specifically, we propose a graph neighborhood aggregation module, which constructs a global graph to capture the overall structural relationships within point clouds. We then propose a progressive context interaction module to enhance feature discriminability by performing intra-domain feature decoupling and inter-domain context interaction. Finally, we design a dynamic inlier selection method that optimizes inlier weights using residual information from multiple iterations of pose estimation, thereby improving the accuracy and robustness of registration. Extensive experiments on indoor RGB-D and outdoor LiDAR datasets show that the proposed MCI-Net significantly outperforms existing state-of-the-art methods, achieving the highest registration recall of 96.4\% on 3DMatch. Source code is available at http://www.linshuyuan.com.

---

## 13. Autoregressive Flow Matching for Motion Prediction

**论文链接:** [http://arxiv.org/abs/2512.22688v1](http://arxiv.org/abs/2512.22688v1)

**作者:** Johnathan Xie, Stefan Stojanov, Cristobal Eyzaguirre, Daniel L. K. Yamins, Jiajun Wu

**发布时间:** 2025-12-27

### GPT解析

### 总结

本文提出了一种名为自回归流匹配(ARFM)的新方法，用于对连续序列数据进行概率建模，通过多样化视频数据集训练，能够生成长期的未来点轨迹位置，并在复杂运动预测和下游任务性能提升方面表现出色。

### 背景

运动预测研究已在不同背景下进行，现有模型通常在窄分布数据上训练并应用于人类运动预测和机器人等下游任务。同时，最近在视频预测扩展方面的努力展示了令人印象深刻的视觉真实性，但在准确建模复杂运动方面仍然存在困难，尽管规模巨大。

### 目的

开发一种能够准确建模复杂运动的新方法，用于预测人类和机器人的未来运动轨迹，并提高下游任务性能。

### 方法

作者开发了自回归流匹配(ARFM)方法，这是一种对连续序列数据进行概率建模的新方法。他们在多样化的视频数据集上训练该方法，以生成长期的未来点轨迹位置。此外，他们还开发了评估基准，用于评估运动预测模型预测人类和机器人运动的能力。

### 主要发现

该模型能够预测复杂运动，并且将机器人动作预测和人类运动预测基于预测的未来轨迹进行条件化处理，可以显著提高下游任务性能。

### 结论

自回归流匹配(ARFM)方法在复杂运动预测方面表现出色，并且通过将预测的未来轨迹作为条件，可以有效提高下游任务性能。代码和模型已在GitHub上公开。

### 翻译

运动预测已在不同背景下进行研究，模型在窄分布数据上训练并应用于人类运动预测和机器人等下游任务。同时，最近在扩展视频预测方面的努力展示了令人印象深刻的视觉真实性，但在准确建模复杂运动方面仍然存在困难，尽管规模巨大。受视频生成扩展的启发，我们开发了自回归流匹配(ARFM)，这是一种对连续序列数据进行概率建模的新方法，我们在多样化的视频数据集上对其进行训练，以生成长期的未来点轨迹位置。为了评估我们的模型，我们开发了评估基准，用于评估运动预测模型预测人类和机器人运动的能力。我们的模型能够预测复杂运动，我们证明将机器人动作预测和人类运动预测基于预测的未来轨迹进行条件化处理可以显著提高下游任务性能。代码和模型可在以下网址公开获取：https://github.com/Johnathan-Xie/arfm-motion-prediction。


### 论文摘要

Motion prediction has been studied in different contexts with models trained on narrow distributions and applied to downstream tasks in human motion prediction and robotics. Simultaneously, recent efforts in scaling video prediction have demonstrated impressive visual realism, yet they struggle to accurately model complex motions despite massive scale. Inspired by the scaling of video generation, we develop autoregressive flow matching (ARFM), a new method for probabilistic modeling of sequential continuous data and train it on diverse video datasets to generate future point track locations over long horizons. To evaluate our model, we develop benchmarks for evaluating the ability of motion prediction models to predict human and robot motion. Our model is able to predict complex motions, and we demonstrate that conditioning robot action prediction and human motion prediction on predicted future tracks can significantly improve downstream task performance. Code and models publicly available at: https://github.com/Johnathan-Xie/arfm-motion-prediction.

---

## 14. End-to-End Test-Time Training for Long Context

**论文链接:** [http://arxiv.org/abs/2512.23675v1](http://arxiv.org/abs/2512.23675v1)

**作者:** Arnuv Tandon, Karan Dalal, Xinhao Li, Daniel Koceja, Marcel Rød, Sam Buchanan, Xiaolong Wang, Jure Leskovec, Sanmi Koyejo, Tatsunori Hashimoto, Carlos Guestrin, Jed McCaleb, Yejin Choi, Yu Sun

**发布时间:** 2025-12-29

**备注:** Code: https://github.com/test-time-training/e2e

### GPT解析

### 总结

作者提出了一种将长上下文语言建模视为持续学习问题而非架构设计问题的新方法，使用标准Transformer架构但在测试时通过下一个词预测进行持续学习，并通过元学习改进初始化，实现了端到端的测试时训练。

### 背景

长上下文语言建模通常被视为架构设计问题，但作者认为应从持续学习的角度重新审视。

### 目的

提出一种新的方法来解决长上下文语言建模问题，能够在测试时持续学习并高效处理长上下文。

### 方法

使用标准的Transformer架构（带滑动窗口注意力），在测试时通过给定上下文的下一个词预测进行持续学习，将读取的上下文压缩到权重中；同时通过训练时的元学习改进模型初始化，以便在测试时学习。这种方法被称为测试时训练（TTT-E2E）。

### 主要发现

对于使用164B tokens训练的30亿模型，TTT-E2E方法与全注意力Transformer一样随上下文长度扩展，而其他方法（如Mamba 2和Gated DeltaNet）则不能。类似于RNN，TTT-E2E具有恒定的推理延迟，与上下文长度无关，对于128K上下文比全注意力快2.7倍。

### 结论

TTT-E2E方法是端到端的，在测试时（通过下一个词预测）和训练时（通过元学习）都是端到端的，这与之前的形式不同。代码已公开可用。

### 翻译

我们将长上下文语言建模表述为一个持续学习问题，而非架构设计问题。在此表述下，我们仅使用标准架构——带滑动窗口注意力的Transformer。然而，我们的模型通过在给定上下文上进行下一个词预测在测试时持续学习，将其读取的上下文压缩到权重中。此外，我们通过训练时的元学习改进模型初始化，以便在测试时学习。总体而言，我们的方法，即测试时训练（TTT）的一种形式，在测试时（通过下一个词预测）和训练时（通过元学习）都是端到端的，与之前的形式形成对比。我们进行了大量实验，重点关注扩展特性。特别是，对于使用164B tokens训练的30亿模型，我们的方法（TTT-E2E）与全注意力Transformer一样随上下文长度扩展，而其他方法，如Mamba 2和Gated DeltaNet，则不能。然而，类似于RNN，TTT-E2E具有恒定的推理延迟，无论上下文长度如何，这使得对于128K上下文它比全注意力快2.7倍。我们的代码已公开可用。


### 论文摘要

We formulate long-context language modeling as a problem in continual learning rather than architecture design. Under this formulation, we only use a standard architecture -- a Transformer with sliding-window attention. However, our model continues learning at test time via next-token prediction on the given context, compressing the context it reads into its weights. In addition, we improve the model's initialization for learning at test time via meta-learning at training time. Overall, our method, a form of Test-Time Training (TTT), is End-to-End (E2E) both at test time (via next-token prediction) and training time (via meta-learning), in contrast to previous forms. We conduct extensive experiments with a focus on scaling properties. In particular, for 3B models trained with 164B tokens, our method (TTT-E2E) scales with context length in the same way as Transformer with full attention, while others, such as Mamba 2 and Gated DeltaNet, do not. However, similar to RNNs, TTT-E2E has constant inference latency regardless of context length, making it 2.7 times faster than full attention for 128K context. Our code is publicly available.

---

## 15. Task-oriented Learnable Diffusion Timesteps for Universal Few-shot Learning of Dense Tasks

**论文链接:** [http://arxiv.org/abs/2512.23210v1](http://arxiv.org/abs/2512.23210v1)

**作者:** Changgyoon Oh, Jongoh Jeong, Jegyeong Cho, Kuk-Jin Yoon

**发布时间:** 2025-12-29

### GPT解析

### 总结

本研究提出了一种自适应选择扩散时间步特征的方法，通过任务感知时间步选择和时间步特征合并两个模块，改进少样本密集预测任务性能。

### 背景

去噪扩散概率模型在生成任务中取得显著进展，但当前应用中扩散时间步特征的选择依赖经验直觉，导致次优性能且偏向特定任务。

### 目的

解决扩散时间步特征选择的启发式方法问题，通过自适应选择最适合少样本密集预测任务的时间步特征，提高在任意未见任务上的性能。

### 方法

提出两个关键模块：任务感知时间步选择(TTS)基于时间步损失和相似性分数选择理想扩散时间步；时间步特征合并(TFC)合并所选时间步特征以提高密集预测性能；同时采用参数高效的微调适配器。

### 主要发现

所提出的框架在仅有少量支持查询的情况下能有效实现密集预测性能的优越性，在Taskonomy数据集上验证了该可学习时间步合并方法的有效性。

### 结论

自适应选择扩散时间步特征的方法能够显著改善少样本密集预测任务性能，为实际通用和少样本学习场景提供了有效解决方案。

### 翻译

去噪扩散概率模型在生成任务中带来了巨大进展，迄今为止取得了最先进的性能。当前基于扩散模型的应用通过附加任务特定解码器，利用多步前向-后向马尔可夫过程学习到的视觉表示能力进行单任务预测。然而，扩散时间步特征的经验选择仍然严重依赖经验直觉，常常导致偏向特定任务的次优性能。为了缓解这一限制，我们通过自适应选择最适合少样本密集预测任务的时间步特征，研究了多功能扩散时间步特征的重要性，在任意未见任务上进行评估。为此，我们提出了两个模块：基于时间步损失和相似性分数选择理想扩散时间步的任务感知时间步选择(TTS)，以及合并所选时间步特征以提高少样本设置下密集预测性能的时间步特征合并(TFC)。配合我们的参数高效微调适配器，我们的框架在仅有少量支持查询的情况下有效实现了密集预测性能的优越性。我们在大规模具有挑战性的Taskonomy数据集上针对密集预测任务进行了实证验证，特别是针对实际的通用和少样本学习场景。


### 论文摘要

Denoising diffusion probabilistic models have brought tremendous advances in generative tasks, achieving state-of-the-art performance thus far. Current diffusion model-based applications exploit the power of learned visual representations from multistep forward-backward Markovian processes for single-task prediction tasks by attaching a task-specific decoder. However, the heuristic selection of diffusion timestep features still heavily relies on empirical intuition, often leading to sub-optimal performance biased towards certain tasks. To alleviate this constraint, we investigate the significance of versatile diffusion timestep features by adaptively selecting timesteps best suited for the few-shot dense prediction task, evaluated on an arbitrary unseen task. To this end, we propose two modules: Task-aware Timestep Selection (TTS) to select ideal diffusion timesteps based on timestep-wise losses and similarity scores, and Timestep Feature Consolidation (TFC) to consolidate the selected timestep features to improve the dense predictive performance in a few-shot setting. Accompanied by our parameter-efficient fine-tuning adapter, our framework effectively achieves superiority in dense prediction performance given only a few support queries. We empirically validate our learnable timestep consolidation method on the large-scale challenging Taskonomy dataset for dense prediction, particularly for practical universal and few-shot learning scenarios.

---

## 16. 论文ID: 2512.22966v1

**错误:** 无法读取论文信息 - [Errno 2] No such file or directory: 'storage/result/2512.22966v1.json'

---

## 17. 论文ID: 2512.22904v1

**错误:** 无法读取论文信息 - [Errno 2] No such file or directory: 'storage/result/2512.22904v1.json'

---

## 18. 论文ID: 2512.22777v1

**错误:** 无法读取论文信息 - [Errno 2] No such file or directory: 'storage/result/2512.22777v1.json'

---

## 19. Human-like visual computing advances explainability and few-shot learning in deep neural networks for complex physiological data

**论文链接:** [http://arxiv.org/abs/2512.22349v1](http://arxiv.org/abs/2512.22349v1)

**作者:** Alaa Alahmadi, Mohamed Hasan

**发布时间:** 2025-12-26

### GPT解析

### 总结

研究展示了一种感知伪彩色技术可提高深度神经网络分析生理数据时的可解释性和少样本学习能力，使模型从极少量样本中学习临床相关特征。

### 背景

机器视觉模型特别是深度神经网络越来越多应用于生理信号解释如心电图，但通常需要大量训练数据且对预测背后的因果特征理解有限，这限制了其临床可靠性与人类推理的一致性。

### 目的

探索感知伪彩色技术如何提升深度神经网络分析复杂数据时的可解释性和少样本学习能力，该技术先前已被证明能增强人类对心电图的理解。

### 方法

聚焦于获得性药物诱导的长QT综合征作为案例研究，将临床相关时间特征编码为结构化颜色表示，使用原型网络和ResNet-18架构，在从单个心脏周期和完整10秒节律导出的ECG图像上评估单样本和少样本学习。

### 主要发现

伪彩色编码使模型能从仅一个或五个训练样本中学习判别性和可解释特征；引导注意力转向有临床意义的ECG特征同时抑制无关信号分量；聚合多个心脏周期可提高性能，类似于人类对心跳的感知平均。

### 结论

人类感知编码可以在医学机器智能中弥合数据效率、可解释性和因果推理之间的差距。

### 翻译

机器视觉模型，特别是深度神经网络，越来越多地应用于生理信号解释，包括心电图，但它们通常需要大量训练数据，且对其预测背后的因果特征提供有限见解。这种数据效率和可解释性的缺乏限制了它们的临床可靠性与人类推理的一致性。在此，我们展示了一种感知伪彩色技术，先前已被证明能增强人类对心电图的理解，可以提高深度神经网络分析复杂数据时的可解释性和少样本学习能力。我们聚焦于获得性药物诱导的长QT综合征作为具有挑战性的案例研究，其特征是信号形态异质性、心率可变性和与尖端扭转型室性心动过速等危及生命的心律失常相关的稀少阳性病例。此环境为模型在极端数据稀缺下的泛化能力提供了严格测试。通过将临床相关的时间特征（如QT间期持续时间）编码为结构化颜色表示，模型能够从仅一个或五个训练样本中学习判别性和可解释特征。使用原型网络和ResNet-18架构，我们在从单个心脏周期和完整10秒节律导出的ECG图像上评估单样本和少样本学习。可解释性分析表明，伪彩色引导注意力转向有临床意义的ECG特征，同时抑制无关信号分量。聚合多个心脏周期可进一步提高性能，类似于人类对心跳的感知平均。总之，这些发现表明，类人感知编码可以在医学机器智能中弥合数据效率、可解释性和因果推理之间的差距。


### 论文摘要

Machine vision models, particularly deep neural networks, are increasingly applied to physiological signal interpretation, including electrocardiography (ECG), yet they typically require large training datasets and offer limited insight into the causal features underlying their predictions. This lack of data efficiency and interpretability constrains their clinical reliability and alignment with human reasoning. Here, we show that a perception-informed pseudo-colouring technique, previously demonstrated to enhance human ECG interpretation, can improve both explainability and few-shot learning in deep neural networks analysing complex physiological data.   We focus on acquired, drug-induced long QT syndrome (LQTS) as a challenging case study characterised by heterogeneous signal morphology, variable heart rate, and scarce positive cases associated with life-threatening arrhythmias such as torsades de pointes. This setting provides a stringent test of model generalisation under extreme data scarcity. By encoding clinically salient temporal features, such as QT-interval duration, into structured colour representations, models learn discriminative and interpretable features from as few as one or five training examples. Using prototypical networks and a ResNet-18 architecture, we evaluate one-shot and few-shot learning on ECG images derived from single cardiac cycles and full 10-second rhythms. Explainability analyses show that pseudo-colouring guides attention toward clinically meaningful ECG features while suppressing irrelevant signal components. Aggregating multiple cardiac cycles further improves performance, mirroring human perceptual averaging across heartbeats. Together, these findings demonstrate that human-like perceptual encoding can bridge data efficiency, explainability, and causal reasoning in medical machine intelligence.

---

## 20. Enhanced geometry prediction in laser directed energy deposition using meta-learning

**论文链接:** [http://arxiv.org/abs/2512.22241v1](http://arxiv.org/abs/2512.22241v1)

**作者:** Abdul Malik Al Mardhouf Al Saadi, Amrita Basak

**发布时间:** 2025-12-23

### GPT解析

### 总结

提出了一种基于元学习的跨数据集知识转移模型，用于解决激光定向能量沉积中珠体几何形状预测面临的实验数据稀缺和异质性问题。

### 背景

在激光定向能量沉积中，精确预测珠体几何形状受到实验数据集稀缺和异质性的阻碍，这些数据集是在不同材料、机器配置和工艺参数下收集的。

### 目的

开发一种能够有效利用有限数据的模型，用于预测L-DED中的沉积轨道几何形状，实现跨数据集的知识转移。

### 方法

研究两种基于梯度的元学习算法（MAML和Reptile），使模型能够使用有限数据快速适应新的沉积条件。该框架使用从同行评议文献和内部实验汇编的多个实验数据集进行，并在粉末送丝、送丝和混合丝粉L-DED工艺中进行评估。

### 主要发现

MAML和Reptile都能仅使用三到九个训练样本在未见过的目标任务上实现精确的珠体高度预测，并且在可比数据约束下优于传统的馈送前向神经网络。元学习模型在代表不同打印条件的多个目标任务上实现了强大的泛化性能，R平方值高达约零点九，平均绝对误差在零点零三到零点零八毫米之间。

### 结论

元学习方法能够有效地在异质L-DED设置中实现知识转移，解决了实验数据稀缺和异质性问题。

### 翻译

在激光定向能量沉积中精确预测珠体几何形状通常受到实验数据集稀缺和异质性的阻碍，这些数据集是在不同材料、机器配置和工艺参数下收集的。为解决这一挑战，提出了一种基于元学习的跨数据集知识转移模型，用于预测L-DED中的沉积轨道几何形状。具体而言，研究了两种基于梯度的元学习算法，即模型无关元学习和Reptile，使模型能够使用有限数据快速适应新的沉积条件。该框架使用从同行评议文献和内部实验汇编的多个实验数据集进行，并在粉末送丝、送丝和混合丝粉L-DED工艺中进行评估。结果表明，MAML和Reptile都能仅使用三到九个训练样本在未见过的目标任务上实现精确的珠体高度预测，并且在可比数据约束下始终优于传统的馈送前向神经网络。在代表不同打印条件的多个目标任务上，元学习模型实现了强大的泛化性能，R平方值高达约零点九，平均绝对误差在零点零三到零点零八毫米之间，证明了在异质L-DED设置中有效的知识转移。


### 论文摘要

Accurate bead geometry prediction in laser-directed energy deposition (L-DED) is often hindered by the scarcity and heterogeneity of experimental datasets collected under different materials, machine configurations, and process parameters. To address this challenge, a cross-dataset knowledge transfer model based on meta-learning for predicting deposited track geometry in L-DED is proposed. Specifically, two gradient-based meta-learning algorithms, i.e., Model-Agnostic Meta-Learning (MAML) and Reptile, are investigated to enable rapid adaptation to new deposition conditions with limited data. The proposed framework is performed using multiple experimental datasets compiled from peer-reviewed literature and in-house experiments and evaluated across powder-fed, wire-fed, and hybrid wire-powder L-DED processes. Results show that both MAML and Reptile achieve accurate bead height predictions on unseen target tasks using as few as three to nine training examples, consistently outperforming conventional feedforward neural networks trained under comparable data constraints. Across multiple target tasks representing different printing conditions, the meta-learning models achieve strong generalization performance, with R-squared values reaching up to approximately 0.9 and mean absolute errors between 0.03-0.08 mm, demonstrating effective knowledge transfer across heterogeneous L-DED settings.

---

## 21. Quadrant Segmentation VLM with Few-Shot Adaptation and OCT Learning-based Explainability Methods for Diabetic Retinopathy

**论文链接:** [http://arxiv.org/abs/2512.22197v1](http://arxiv.org/abs/2512.22197v1)

**作者:** Shivum Telang

**发布时间:** 2025-12-20

**备注:** 4 pages, 6 figures

### GPT解析

### 总结

该论文提出了一种新型多模态可解释性模型，利用视觉语言模型和少样本学习来模拟眼科医生的推理过程，通过分析视网膜象限内的病变分布，生成成对的Grad-CAM热图，展示OCT和眼底图像中单个神经元的权重，从而改善糖尿病视网膜病变的诊断。

### 背景

糖尿病视网膜病变是全球视力丧失的主要原因，需要早期检测来保护视力。然而，医疗资源有限常常导致DR未被诊断。

### 目的

开发一种能够以自然语言识别个体DR病变的定量检测系统，解决当前DR诊断中的关键限制。

### 方法

提出一种使用视觉语言模型和少样本学习的多模态可解释性模型，通过分析视网膜象限内的病变分布来模拟眼科医生的推理过程，并生成成对的Grad-CAM热图。

### 主要发现

该创新方法利用3000张眼底图像和1000张OCT图像的数据集，解决了当前DR诊断中的关键限制，提供了改善患者结果的实用工具。

### 结论

该模型能够模拟眼科医生的推理过程，通过可视化突出显示对DR严重程度分类有贡献的区域，是一种改善患者结果的实用且全面的工具。

### 翻译

糖尿病视网膜病变是全球视力丧失的主要原因，需要早期检测来保护视力。医疗资源有限常常导致DR未被诊断。为此，AI模型利用病变分割来提高可解释性；然而，手动注释病变对临床医生来说不切实际。医生需要的是能解释分类推理过程而不仅仅是突出病变位置的模型。此外，当前模型是一维的，依赖单一成像模态进行可解释性，效果有限。相比之下，能够以自然语言识别个体DR病变的定量检测系统将克服这些限制，在筛查、治疗和研究环境中实现多样化应用。为解决这一问题，本文提出了一种新型多模态可解释性模型，利用带有少样本学习的视觉语言模型，通过分析视网膜象限内的病变分布来模拟眼科医生的推理过程。该模型生成成对的Grad-CAM热图，展示OCT和眼底图像中单个神经元的权重，直观地突出显示对DR严重程度分类有贡献的区域。使用包含3000张眼底图像和1000张OCT图像的数据集，这种创新方法解决了当前DR诊断中的关键限制，为改善患者结果提供了实用且全面的工具。


### 论文摘要

Diabetic Retinopathy (DR) is a leading cause of vision loss worldwide, requiring early detection to preserve sight. Limited access to physicians often leaves DR undiagnosed. To address this, AI models utilize lesion segmentation for interpretability; however, manually annotating lesions is impractical for clinicians. Physicians require a model that explains the reasoning for classifications rather than just highlighting lesion locations. Furthermore, current models are one-dimensional, relying on a single imaging modality for explainability and achieving limited effectiveness. In contrast, a quantitative-detection system that identifies individual DR lesions in natural language would overcome these limitations, enabling diverse applications in screening, treatment, and research settings. To address this issue, this paper presents a novel multimodal explainability model utilizing a VLM with few-shot learning, which mimics an ophthalmologist's reasoning by analyzing lesion distributions within retinal quadrants for fundus images. The model generates paired Grad-CAM heatmaps, showcasing individual neuron weights across both OCT and fundus images, which visually highlight the regions contributing to DR severity classification. Using a dataset of 3,000 fundus images and 1,000 OCT images, this innovative methodology addresses key limitations in current DR diagnostics, offering a practical and comprehensive tool for improving patient outcomes.

---

## 22. Joint UAV-UGV Positioning and Trajectory Planning via Meta A3C for Reliable Emergency Communications

**论文链接:** [http://arxiv.org/abs/2512.22187v1](http://arxiv.org/abs/2512.22187v1)

**作者:** Ndagijimana Cyprien, Mehdi Sookhak, Hosein Zarini, Chandra N Sekharan, Mohammed Atiquzzaman

**发布时间:** 2025-12-20

### GPT解析

### 总结

该论文提出了一种联合无人机和无人地面车辆的部署和轨迹规划框架，用于在灾害地区建立通信并保证服务质量。通过引入道路图模型和结合元学习的A3C算法，实现了最优的服务质量和通信效率。

### 背景

联合部署无人机和无人地面车辆已被证明是在灾害影响地区建立通信的有效方法。然而，在使用尽可能少的无人机的同时确保良好的服务质量，需要对无人机和无人地面车辆进行最优定位和轨迹规划。

### 目的

提出一种联合无人机-无人地面车辆定位和轨迹规划框架，以保证地面用户获得最优的服务质量。

### 方法

引入道路图模型模拟UGVs的移动性，将速率优化问题重新表述为马尔可夫决策过程，并提出结合元学习的异步优势行动者评论家(A3C)算法以快速适应新环境和动态条件。

### 主要发现

提出的Meta-A3C方法在性能上优于A3C和DDPG算法，提供13.1%更高的吞吐量，执行速度快49%，同时满足服务质量要求。

### 结论

通过结合元学习的异步优势行动者评论家算法能够有效解决无人机和无人地面车辆的联合部署和轨迹规划问题，在保证服务质量的同时提高通信效率和执行速度。

### 翻译

无人机和无人地面车辆的联合部署已被证明是在灾害影响地区建立通信的有效方法。然而，在使用尽可能少的无人机的同时确保良好的服务质量，也需要对无人机和无人地面车辆进行最优定位和轨迹规划。本文提出了一种基于无人机和无人地面车辆联合部署的定位和轨迹规划框架，以保证地面用户获得最优的服务质量。为了模拟无人地面车辆的移动性，我们引入了一个道路图，引导它们沿着有效的道路段移动并遵守道路网络约束。为了解决速率优化问题，我们将该问题重新表述为马尔可夫决策过程，并提出了一种新颖的结合元学习的异步优势行动者评论家算法，以快速适应新环境和动态条件。数值结果表明，我们提出的Meta-A3C方法优于A3C和DDPG，提供13.1%更高的吞吐量和49%更快的执行速度，同时满足服务质量要求。


### 论文摘要

Joint deployment of unmanned aerial vehicles (UAVs) and unmanned ground vehicles (UGVs) has been shown to be an effective method to establish communications in areas affected by disasters. However, ensuring good Quality of Services (QoS) while using as few UAVs as possible also requires optimal positioning and trajectory planning for UAVs and UGVs. This paper proposes a joint UAV-UGV-based positioning and trajectory planning framework for UAVs and UGVs deployment that guarantees optimal QoS for ground users. To model the UGVs' mobility, we introduce a road graph, which directs their movement along valid road segments and adheres to the road network constraints. To solve the sum rate optimization problem, we reformulate the problem as a Markov Decision Process (MDP) and propose a novel asynchronous Advantage Actor Critic (A3C) incorporated with meta-learning for rapid adaptation to new environments and dynamic conditions. Numerical results demonstrate that our proposed Meta-A3C approach outperforms A3C and DDPG, delivering 13.1\% higher throughput and 49\% faster execution while meeting the QoS requirements.

---

## 23. PCR-ORB: Enhanced ORB-SLAM3 with Point Cloud Refinement Using Deep Learning-Based Dynamic Object Filtering

**论文链接:** [http://arxiv.org/abs/2512.23318v1](http://arxiv.org/abs/2512.23318v1)

**作者:** Sheng-Kai Chen, Jie-Yu Chao, Jr-Yu Chang, Po-Lien Wu, Po-Chiang Lin

**发布时间:** 2025-12-29

**备注:** 17 pages, 2 figures, 1 table

### GPT解析

### 总结

本文提出了一种名为PCR-ORB（点云细化ORB）的增强型ORB-SLAM3框架，通过集成基于深度学习的点云细化技术来减轻动态物体对视觉同步定位与地图构建系统的干扰。

### 背景

视觉同步定位与地图构建(vSLAM)系统在动态环境中面临重大挑战，因为移动物体会严重影响跟踪精度和地图一致性。

### 目的

开发一种能够有效处理动态环境干扰的vSLAM系统，提高跟踪精度和地图一致性。

### 方法

使用YOLOv8进行语义分割，结合CUDA加速处理实现实时性能，并实施多阶段过滤策略，包括地面平面估计、天空区域去除、边缘过滤和时间一致性验证。

### 主要发现

在KITTI数据集上的评估显示，序列04取得了显著改进：ATE RMSE提高25.9%，ATE中位数提高30.4%。然而，不同序列上的表现参差不齐，表明效果依赖于具体场景。

### 结论

该实现为动态物体过滤的挑战提供了见解，并为在复杂环境中实现稳健导航创造了机会。

### 翻译

视觉同步定位与地图构建(vSLAM)系统在动态环境中遇到重大挑战，因为移动物体会损害跟踪精度和地图一致性。本文介绍了PCR-ORB（点云细化ORB），这是一种增强型ORB-SLAM3框架，集成了基于深度学习的点云细化技术，以减轻动态物体干扰。我们的方法采用YOLOv8进行语义分割，并结合CUDA加速处理以实现实时性能。该系统实现了多阶段过滤策略，包括地面平面估计、天空区域去除、边缘过滤和时间一致性验证。在KITTI数据集（序列00-09）上的全面评估展示了不同环境条件和场景类型下的性能特征。在特定序列中观察到显著改进，序列04的ATE RMSE提高了25.9%，ATE中位数提高了30.4%。然而，结果显示不同序列上的表现参差不齐，表明效果依赖于具体场景。该实现为动态物体过滤的挑战提供了见解，并为在复杂环境中实现稳健导航创造了机会。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决视觉SLAM系统在动态环境中面临的挑战，即移动物体会降低跟踪精度和地图一致性问题。这个问题在现实中非常重要，因为自动驾驶车辆、机器人和无人机等自主系统需要在包含车辆、行人等动态元素的真实世界中运行。传统SLAM算法假设环境是静态的，在动态环境中会导致定位错误和地图不一致，影响系统的可靠性和安全性。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先认识到传统SLAM系统在动态环境中的局限性，特别是ORB-SLAM3虽然性能出色但在动态环境中仍易受干扰。作者注意到传统处理动态物体的方法缺乏语义理解能力，而深度学习的发展为解决这个问题提供了新机会。设计上，作者选择了YOLOv8进行语义分割，并构建了多阶段过滤策略结合语义信息、几何约束、时间一致性和运动模式分析。作者借鉴了现有工作，包括基于ORB-SLAM3框架、使用YOLOv8语义分割、RANSAC地面平面估计、Lucas-Kanade光流法运动检测以及CUDA加速技术等。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是通过深度学习增强的语义理解来识别和过滤点云中的动态物体，同时保留静态环境特征，提高ORB-SLAM3在动态环境中的定位和建图精度。整体实现流程包括：1) 基于ORB-SLAM3的三线程架构增加并行点云过滤线程；2) 使用YOLOv8进行语义分割并通过CUDA加速实现实时性能；3) 实施多阶段过滤策略，包括语义评分分类、地面平面估计过滤、时间一致性和运动检测、边缘和天空区域过滤；4) 通过CUDA加速实现高效处理；5) 与ORB-SLAM3集成，修改跟踪循环和关键帧管理；6) 实现实时性能优化。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1) 集成YOLOv8语义分割到ORB-SLAM3框架中实现实时动态物体过滤；2) 开发多阶段点云细化策略结合语义、几何、时间和空间信息；3) 实现CUDA加速处理管道保持实时性能；4) 构建综合评估框架验证系统性能。相比之前的工作，PCR-ORB不仅使用简单的物体检测，而是实现了更全面的多阶段过滤；不仅考虑语义信息，还结合了几何约束和时间一致性；通过CUDA加速保持了实时性能；将深度学习集成到核心SLAM管道中而非作为后处理步骤；采用了自适应阈值机制根据环境调整过滤参数。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'PCR-ORB通过集成YOLOv8语义分割和多阶段过滤策略，显著提升了ORB-SLAM3在动态环境中的定位精度，同时通过CUDA加速保持实时性能，为自主导航系统提供了一种鲁棒的动态环境感知解决方案。'}


### 论文摘要

Visual Simultaneous Localization and Mapping (vSLAM) systems encounter substantial challenges in dynamic environments where moving objects compromise tracking accuracy and map consistency. This paper introduces PCR-ORB (Point Cloud Refinement ORB), an enhanced ORB-SLAM3 framework that integrates deep learning-based point cloud refinement to mitigate dynamic object interference. Our approach employs YOLOv8 for semantic segmentation combined with CUDA-accelerated processing to achieve real-time performance. The system implements a multi-stage filtering strategy encompassing ground plane estimation, sky region removal, edge filtering, and temporal consistency validation. Comprehensive evaluation on the KITTI dataset (sequences 00-09) demonstrates performance characteristics across different environmental conditions and scene types. Notable improvements are observed in specific sequences, with sequence 04 achieving 25.9% improvement in ATE RMSE and 30.4% improvement in ATE median. However, results show mixed performance across sequences, indicating scenario-dependent effectiveness. The implementation provides insights into dynamic object filtering challenges and opportunities for robust navigation in complex environments.

---

## 24. GaussianDWM: 3D Gaussian Driving World Model for Unified Scene Understanding and Multi-Modal Generation

**论文链接:** [http://arxiv.org/abs/2512.23180v1](http://arxiv.org/abs/2512.23180v1)

**作者:** Tianchen Deng, Xuefeng Chen, Yi Chen, Qu Chen, Yuyao Xu, Lijin Yang, Le Xu, Yu Zhang, Bo Zhang, Wuxiong Huang, Hesheng Wang

**发布时间:** 2025-12-29

### GPT解析

### 总结

该研究提出了一种基于3D高斯场景表示的新型统一驾驶世界模型框架，实现了3D场景理解和多模态场景生成能力。

### 背景

现有的驾驶世界模型缺乏3D场景理解能力，只能在输入数据条件下生成内容，无法解释或推理驾驶环境。当前方法使用点云或BEV特征表示3D空间信息，不能将文本信息与底层3D场景准确对齐。

### 目的

解决现有驾驶世界模型的局限性，实现3D场景理解和多模态场景生成，并实现文本信息与3D场景的直接对齐。

### 方法

基于3D高斯场景表示提出统一框架，通过将语言特征嵌入到每个高斯基元中实现早期模态对齐，设计任务感知语言引导采样策略去除冗余3D高斯并注入紧凑3D令牌到LLM，以及设计双条件多模态生成模型结合高级语言条件和低级图像条件。

### 主要发现

在nuScenes和NuInteract数据集上的综合研究表明，所提出的方法达到了最先进的性能。

### 结论

基于3D高斯场景表示的统一驾驶世界框架有效解决了现有模型的局限性，实现了3D场景理解和多模态场景生成。

### 翻译

驾驶世界模型随着生成模型的发展而迅速发展。然而，现有的驾驶世界模型缺乏3D场景理解能力，只能在输入数据条件下生成内容，无法解释或推理驾驶环境。此外，当前方法使用点云或BEV特征表示3D空间信息，不能将文本信息与底层3D场景准确对齐。为解决这些局限性，我们提出了一种基于3D高斯场景表示的新型统一驾驶世界模型框架，该框架同时支持3D场景理解和多模态场景生成，并支持理解和生成任务的上下文丰富化。我们的方法通过将丰富的语言特征嵌入到每个高斯基元中，直接将文本信息与3D场景对齐，从而实现早期模态对齐。此外，我们设计了一种新颖的任务感知语言引导采样策略，去除冗余的3D高斯，并将准确且紧凑的3D令牌注入到LLM中。我们还设计了一种双条件多模态生成模型，其中我们视觉语言模型捕获的信息被用作高级语言条件，与低级图像条件结合，共同指导多模态生成过程。我们在nuScenes和NuInteract数据集上进行了综合研究，以验证我们框架的有效性。我们的方法达到了最先进的性能。我们将在GitHub上公开代码：https://github.com/dtc111111/GaussianDWM。


### 论文摘要

Driving World Models (DWMs) have been developing rapidly with the advances of generative models. However, existing DWMs lack 3D scene understanding capabilities and can only generate content conditioned on input data, without the ability to interpret or reason about the driving environment. Moreover, current approaches represent 3D spatial information with point cloud or BEV features do not accurately align textual information with the underlying 3D scene. To address these limitations, we propose a novel unified DWM framework based on 3D Gaussian scene representation, which enables both 3D scene understanding and multi-modal scene generation, while also enabling contextual enrichment for understanding and generation tasks. Our approach directly aligns textual information with the 3D scene by embedding rich linguistic features into each Gaussian primitive, thereby achieving early modality alignment. In addition, we design a novel task-aware language-guided sampling strategy that removes redundant 3D Gaussians and injects accurate and compact 3D tokens into LLM. Furthermore, we design a dual-condition multi-modal generation model, where the information captured by our vision-language model is leveraged as a high-level language condition in combination with a low-level image condition, jointly guiding the multi-modal generation process. We conduct comprehensive studies on the nuScenes, and NuInteract datasets to validate the effectiveness of our framework. Our method achieves state-of-the-art performance. We will release the code publicly on GitHub https://github.com/dtc111111/GaussianDWM.

---

## 25. 论文ID: 2512.23176v1

**错误:** 无法读取论文信息 - [Errno 2] No such file or directory: 'storage/result/2512.23176v1.json'

---

## 26. Differentiable Physics-Driven Human Representation for Millimeter-Wave Based Pose Estimation

**论文链接:** [http://arxiv.org/abs/2512.23054v1](http://arxiv.org/abs/2512.23054v1)

**作者:** Shuntian Zheng, Guangming Wang, Jiaqi Li, Minzhe Ni, Yu Guan

**发布时间:** 2025-12-28

### GPT解析

### 总结

该研究提出了一种新的可微分物理驱动人体表示(DIPR)方法，用于解决毫米波人体姿态估计中热图和点云两种输入范式的局限性。DIPR将人体表示为高斯分布集合，通过整合运动学先验和毫米波传播物理来增强人体特征并抑制噪声，实验证明该方法能有效提升现有毫米波人体姿态估计方法的性能。

### 背景

毫米波在人体姿态估计方面具有非侵入式传感的优势，但当前基于毫米波的方法面临两种主要输入范式的局限性：热图容易受多径传播和硬件调制噪声影响；点云虽能抑制噪声但导致稀疏的人体相关特征。

### 目的

研究提供一种替代输入范式(DIPR)的可行性，解决热图和点云两种输入范式的局限性，提高毫米波人体姿态估计的准确性。

### 方法

DIPR将人体表示为具有运动学和电磁参数的高斯分布集合。通过两种策略减轻噪声：1)整合先验运动知识，基于热图初始化DIPR并建立多面优化目标；2)模拟完整毫米波处理流程，从DIPR重新渲染新热图并与原始热图比较，避免过度拟合。

### 主要发现

在三个数据集上使用四种方法的实验结果表明，现有毫米波人体姿态估计方法可轻松集成DIPR并实现优越性能。

### 结论

DIPR作为一种新的输入范式，通过整合物理模型和运动学先验，有效解决了传统热图和点云方法的局限性，提供了更鲁棒和准确的人体姿态估计。

### 翻译

虽然毫米波在人体姿态估计方面通过其非侵入式传感能力具有优势，但当前基于毫米波的人体姿态估计方法在两种主要输入范式上面临局限性：热图和点云。热图表示从毫米波推导出的密集多维特征，但显著受到多径传播和硬件调制噪声的影响。点云是通过将恒虚警率算法应用于热图获得的一组3D点，它能抑制噪声但导致稀疏的人体相关特征。为解决这些局限性，我们研究了提供替代输入范式的可行性：可微分物理驱动人体表示(DIPR)，它将人体表示为具有运动学和电磁参数的高斯分布集合。受高斯飞溅启发，DIPR利用人体运动先验和毫米波传播物理来增强人体特征，并通过两种策略减轻非人体噪声：1)我们整合先验运动知识，基于热图初始化DIPR，建立多面优化目标，确保生物力学有效性并增强运动特征；2)我们模拟完整的毫米波处理流程，从DIPR重新渲染新的热图，并与原始热图比较，避免因运动学约束过度拟合而产生虚假噪声。在三个数据集上使用四种方法的实验结果表明，现有的基于毫米波的人体姿态估计方法可以轻松集成DIPR并实现优越性能。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': "这篇论文主要解决基于毫米波雷达的人体姿态估计中的'信息-噪声权衡'问题。当前方法使用的热图(Heatmap)输入范式虽然包含密集信息但受噪声干扰严重，而点云(Point Cloud)范式虽然噪声较少但特征稀疏。这个问题在现实中很重要，因为毫米波雷达具有隐私保护和环境光照鲁棒性等优势，解决这一问题可以提高人体姿态估计的准确性，促进毫米波雷达在智能家居、医疗监护、人机交互等领域的应用。", '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者的设计思路借鉴了多个领域的工作：首先深入理解毫米波雷达处理流程和CFAR算法；然后受到计算机视觉中高斯飞溅(Gaussian Splatting)技术的启发，将其概念应用于毫米波信号；同时整合人体生物力学约束确保生成的人体表示物理合理。作者通过识别现有方法的局限性，思考如何结合人体运动学先验和毫米波传播物理来增强人体特征，最终设计了DIPR作为替代表示方法，并通过M-GS流程实现。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是将人体表示为具有运动学和电磁参数的高斯分布集合(DIPR)，通过结合人体生物力学原理和毫米波传播物理来增强人体特征，同时抑制非人类噪声。整体实现流程(M-GS)包括：1)位置和速度提取：从热图中提取粗略位置和速度状态；2)关节关联和表示：将人体建模为高斯原语集合，每个关节由位置、缩放、旋转、速度等六个参数描述；3)重新渲染热图：开发可微分管道将DIPR转换回热图，包含信号调制、多普勒调制等四个关键模块；4)优化：使用重建约束和运动学约束进行参数优化。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)提出DIPR作为新输入范式，首次将高斯飞溅技术应用于毫米波人体姿态估计；2)设计毫米波特定的高斯参数化，引入速度和多普勒特征等毫米波特定参数；3)结合生物力学约束与信号一致性，确保生成的人体表示物理合理且避免过拟合。相比之前的工作，DIPR通过结构化运动学约束减轻了热图的高噪声干扰，同时解决了点云的稀疏性问题，不依赖预训练模块减少了计算开销，直接基于物理原理提供了理论保证。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '这篇论文提出了DIPR，一种可微分物理驱动的人体表示方法，通过结合人体生物力学原理和毫米波传播物理，有效解决了毫米波人体姿态估计中的信息-噪声权衡问题，显著提升了现有方法的性能并开辟了新的研究方向。'}


### 论文摘要

While millimeter-wave (mmWave) presents advantages for Human Pose Estimation (HPE) through its non-intrusive sensing capabilities, current mmWave-based HPE methods face limitations in two predominant input paradigms: Heatmap and Point Cloud (PC). Heatmap represents dense multi-dimensional features derived from mmWave, but is significantly affected by multipath propagation and hardware modulation noise. PC, a set of 3D points, is obtained by applying the Constant False Alarm Rate algorithm to the Heatmap, which suppresses noise but results in sparse human-related features. To address these limitations, we study the feasibility of providing an alternative input paradigm: Differentiable Physics-driven Human Representation (DIPR), which represents humans as an ensemble of Gaussian distributions with kinematic and electromagnetic parameters. Inspired by Gaussian Splatting, DIPR leverages human kinematic priors and mmWave propagation physics to enhance human features while mitigating non-human noise through two strategies: 1) We incorporate prior kinematic knowledge to initialize DIPR based on the Heatmap and establish multi-faceted optimization objectives, ensuring biomechanical validity and enhancing motion features. 2) We simulate complete mmWave processing pipelines, re-render a new Heatmap from DIPR, and compare it with the original Heatmap, avoiding spurious noise generation due to kinematic constraints overfitting. Experimental results on three datasets with four methods demonstrate that existing mmWave-based HPE methods can easily integrate DIPR and achieve superior performance.

---

## 27. 论文ID: 2512.22972v1

**错误:** 无法读取论文信息 - [Errno 2] No such file or directory: 'storage/result/2512.22972v1.json'

---

## 28. 论文ID: 2512.22819v1

**错误:** 无法读取论文信息 - [Errno 2] No such file or directory: 'storage/result/2512.22819v1.json'

---

## 29. 论文ID: 2512.22706v1

**错误:** 无法读取论文信息 - [Errno 2] No such file or directory: 'storage/result/2512.22706v1.json'

---

## 30. MEGA-PCC: A Mamba-based Efficient Approach for Joint Geometry and Attribute Point Cloud Compression

**论文链接:** [http://arxiv.org/abs/2512.22463v1](http://arxiv.org/abs/2512.22463v1)

**作者:** Kai-Hsiang Hsieh, Monyneath Yim, Wen-Hsiao Peng, Jui-Chiu Chiang

**发布时间:** 2025-12-27

**备注:** Accepted at the IEEE/CVF Winter Conference on Applications of Computer Vision 2026 (WACV 2026)

### GPT解析

### 总结

MEGA-PCC是一种完全端到端的、基于学习的点云几何和属性联合压缩框架，通过消除后着色程序和手动比特率调整，实现了数据驱动的比特率分配和简化的整体流程。

### 背景

点云几何和属性的联合压缩对高效3D数据表示至关重要，但现有方法依赖后着色程序和推理过程中手动调整的比特率分配，阻碍了端到端优化并增加了系统复杂性。

### 目的

克服现有方法的局限性，提出一个完全端到端的、基于学习的联合压缩框架，实现更优的率失真性能和运行时效率。

### 方法

提出MEGA-PCC框架，包含两个专门模型：1)主要压缩模型使用共享编码器将几何和属性信息编码为统一潜在表示，双解码器顺序重建几何和属性；2)基于Mamba的熵模型(MEM)通过捕获空间和通道相关性增强熵编码；两个模型都构建在Mamba架构上，有效建模长距离依赖和丰富上下文特征。

### 主要发现

MEGA-PCC通过消除着色需求和启发式比特率调整，能够在训练过程中实现数据驱动的比特率分配，简化整体流程，同时实现更优的率失真性能和运行时效率。

### 结论

大量实验表明，MEGA-PCC相比传统和基于学习的方法实现了更优的率失真性能和运行时效率，为AI驱动的点云压缩提供了强大解决方案。

### 翻译

点云几何和属性的联合压缩对于高效的3D数据表示至关重要。现有方法通常依赖后着色程序和推理过程中手动调整的比特率分配，这阻碍了端到端优化并增加了系统复杂性。为克服这些限制，我们提出了MEGA-PCC，一种完全端到端的、基于学习的联合压缩框架，包含两个专门模型。主要压缩模型采用共享编码器将几何和属性信息编码为统一潜在表示，然后通过双解码器顺序重建几何和属性。补充这一点，基于Mamba的熵模型(MEM)通过捕获空间和通道相关性增强熵编码，改善概率估计。两个模型都构建在Mamba架构上，有效建模长距离依赖和丰富的上下文特征。通过消除着色需求和启发式比特率调整，MEGA-PCC能够在训练过程中实现数据驱动的比特率分配并简化整体流程。大量实验表明，MEGA-PCC相比传统和基于学习的方法实现了更优的率失真性能和运行时效率，为AI驱动的点云压缩提供了强大解决方案。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决点云几何和属性联合压缩中的效率问题。现有方法需要手动调整几何和属性间的比特分配，并依赖重新着色过程连接压缩管道，这阻碍了端到端优化并增加系统复杂性。这个问题很重要，因为随着元宇宙、VR/AR、自动驾驶等沉浸式技术的发展，对高效3D数据表示的需求不断增长，而点云作为捕捉详细几何结构的主导媒介，其高效压缩对减少存储空间、加快传输速度和提高处理效率至关重要。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者通过识别现有方法的局限性（如需要手动比特分配和重新着色过程）来思考解决方案。他们认识到几何和属性间的内在依赖关系，提出需要更统一的框架。设计方法借鉴了Mamba架构的状态空间模型用于高效建模长距离依赖，稀疏卷积用于局部结构建模，以及多方向SSM模块（前向、后向和通道翻转）来全面捕获空间上下文。还借鉴了Morton扫描技术来序列化3D数据，以及现有点云压缩方法如PCGCv2和ANF-PCAC的基本思路。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是使用统一的编码器同时编码几何和属性信息到共享的潜在表示，采用双解码器架构先重建几何再重建属性，利用Mamba架构建模长距离依赖，并通过基于Mamba的熵模型增强熵编码。整体流程：1)将点云体素化为三通道体积输入；2)统一编码器结合稀疏卷积和多方向SSM生成潜在表示；3)几何骨架无损编码，特征张量量化和熵编码；4)几何解码器先重建点坐标；5)重建的几何坐标指导属性解码器重建属性；6)采用两阶段训练策略优化几何和属性重建质量。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)单编码器-双解码器架构，避免重新着色步骤；2)三向Mamba(Tri-Mamba)全面捕获空间和通道信息；3)基于Mamba的熵模型(MEM)提高概率估计准确性；4)端到端比特分配，无需耗时的模型匹配；5)高效设计实现线性时间复杂度。相比之前工作，MEGA-PCC消除了重新着色步骤和手动比特分配需求，使用统一的潜在表示而非分开处理，采用更高效的Mamba架构替代Transformer，实现了真正的端到端学习。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'MEGA-PCC提出了一种基于Mamba的高效点云压缩框架，通过统一的单编码器-双解码器架构和端到端训练，实现了几何和属性信息的联合优化压缩，显著提升了压缩效率和重建质量。'}


### 论文摘要

Joint compression of point cloud geometry and attributes is essential for efficient 3D data representation. Existing methods often rely on post-hoc recoloring procedures and manually tuned bitrate allocation between geometry and attribute bitstreams in inference, which hinders end-to-end optimization and increases system complexity. To overcome these limitations, we propose MEGA-PCC, a fully end-to-end, learning-based framework featuring two specialized models for joint compression. The main compression model employs a shared encoder that encodes both geometry and attribute information into a unified latent representation, followed by dual decoders that sequentially reconstruct geometry and then attributes. Complementing this, the Mamba-based Entropy Model (MEM) enhances entropy coding by capturing spatial and channel-wise correlations to improve probability estimation. Both models are built on the Mamba architecture to effectively model long-range dependencies and rich contextual features. By eliminating the need for recoloring and heuristic bitrate tuning, MEGA-PCC enables data-driven bitrate allocation during training and simplifies the overall pipeline. Extensive experiments demonstrate that MEGA-PCC achieves superior rate-distortion performance and runtime efficiency compared to both traditional and learning-based baselines, offering a powerful solution for AI-driven point cloud compression.

---

## 31. SuperiorGAT: Graph Attention Networks for Sparse LiDAR Point Cloud Reconstruction in Autonomous Systems

**论文链接:** [http://arxiv.org/abs/2512.22439v1](http://arxiv.org/abs/2512.22439v1)

**作者:** Khalfalla Awedat, Mohamed Abidalrekab, Gurcan Comert, Mustafa Ayad

**发布时间:** 2025-12-27

### GPT解析

### 总结

这篇论文介绍了SuperiorGAT，一个基于图注意力的框架，用于重建稀疏激光雷达点云中缺失的高程信息，能够在不增加网络深度的情况下实现准确重建。

### 背景

自动驾驶系统中的激光雷达感知受到固定垂直光束分辨率的限制，并且由于环境遮挡导致的光束脱落进一步降低了性能。

### 目的

设计一个框架来重建稀疏激光雷达点云中缺失的高程信息，以提高激光雷达感知的准确性和完整性。

### 方法

SuperiorGAT通过将激光雷达扫描建模为感知光束的图，并采用门控残差融合与前馈细化技术，实现准确重建。通过模拟结构化光束脱落（移除每第四个垂直扫描光束）来评估性能。

### 主要发现

在多样化的KITTI环境（包括Person、Road、Campus和City序列）进行的广泛实验表明，SuperiorGAT始终比基于PointNet的模型和更深的GAT基线实现更低的重建误差和改进的几何一致性。定性的X-Z投影进一步证实了该模型在保持结构完整性的同时具有最小垂直失真的能力。

### 结论

架构改进提供了一种计算效率高的方法来提高激光雷达分辨率，而无需额外的传感器硬件。

### 翻译

基于激光雷达的自动驾驶系统感知受到固定垂直光束分辨率的限制，并且由于环境遮挡导致的光束脱落进一步降低了性能。本文介绍了SuperiorGAT，一个基于图注意力的框架，用于重建稀疏激光雷达点云中缺失的高程信息。通过将激光雷达扫描建模为感知光束的图，并结合门控残差融合和前馈细化，SuperiorGAT能够在不增加网络深度的情况下实现准确重建。为了评估性能，通过移除每第四个垂直扫描光束来模拟结构化光束脱落。在多样化的KITTI环境（包括Person、Road、Campus和City序列）进行的广泛实验表明，SuperiorGAT始终比基于PointNet的模型和更深的GAT基线实现更低的重建误差和改进的几何一致性。定性的X-Z投影进一步证实了该模型在保持结构完整性的同时具有最小垂直失真的能力。这些结果表明，架构改进提供了一种计算效率高的方法来提高激光雷达分辨率，而无需额外的传感器硬件。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决LiDAR点云在自动驾驶系统中的稀疏性问题，特别是由光束丢失(beam dropout)导致的垂直分辨率下降问题。这个问题很重要，因为LiDAR是自动驾驶系统的核心传感器，环境遮挡或硬件故障会导致数据不完整，影响物体检测、定位和路径规划等关键任务，而现有方法在精度和计算效率之间难以取得平衡。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先分析了现有方法的局限性：压缩感知方法计算量大不适合实时应用；CNN需要密集3D投影对稀疏数据效率低；点云模型难以处理结构化稀疏模式；标准图神经网络无法有效优先处理关键局部几何特征。作者借鉴了自己之前的会议论文工作，发现多层GAT可有效重建LiDAR高度值，但增加深度会带来计算成本和稳定性问题，因此转向架构优化而非深度提升，设计了结合光束感知图、门控残差融合和前馈细化的方法。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是将LiDAR扫描建模为光束感知图，使用图注意力网络动态优化点与点之间的交互，通过门控残差融合和前馈细化提高重建质量，专注于重建缺失的垂直高度信息。整体流程包括：1)数据表示：将稀疏点云表示为图结构，节点为点，边表示连接关系；2)架构流程：光束感知特征编码→多头注意力聚合→门控残差融合→前馈细化→任务特定输出解码；3)训练评估：在KITTI数据集上模拟光束丢失，使用RMSE和Chamfer距离评估重建质量，测量推理时间评估效率。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': "关键创新点包括：1)全面的光束丢失模拟框架，准确模拟真实硬件故障；2)新的图注意力重建架构，通过学习注意力权重优化点对点交互；3)计算高效框架，专为实时部署设计。相比之前工作的不同：与传统插值方法相比能保留复杂几何特征避免'阶梯'伪影；与压缩感知相比不需要迭代求解器适合实时应用；与CNN相比避免密集3D投影更适应稀疏数据；与标准图神经网络相比使用注意力机制而非固定权重；与作者之前工作相比不再增加深度而是优化架构。", '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'SuperiorGAT通过创新的图注意力网络架构，实现了高效准确的稀疏LiDAR点云垂直信息重建，解决了自动驾驶系统中光束丢失问题，在保持计算效率的同时显著提高了重建质量。'}


### 论文摘要

LiDAR-based perception in autonomous systems is constrained by fixed vertical beam resolution and further compromised by beam dropout resulting from environmental occlusions. This paper introduces SuperiorGAT, a graph attention-based framework designed to reconstruct missing elevation information in sparse LiDAR point clouds. By modeling LiDAR scans as beam-aware graphs and incorporating gated residual fusion with feed-forward refinement, SuperiorGAT enables accurate reconstruction without increasing network depth. To evaluate performance, structured beam dropout is simulated by removing every fourth vertical scanning beam. Extensive experiments across diverse KITTI environments, including Person, Road, Campus, and City sequences, demonstrate that SuperiorGAT consistently achieves lower reconstruction error and improved geometric consistency compared to PointNet-based models and deeper GAT baselines. Qualitative X-Z projections further confirm the model's ability to preserve structural integrity with minimal vertical distortion. These results suggest that architectural refinement offers a computationally efficient method for improving LiDAR resolution without requiring additional sensor hardware.

---

## 32. PortionNet: Distilling 3D Geometric Knowledge for Food Nutrition Estimation

**论文链接:** [http://arxiv.org/abs/2512.22304v1](http://arxiv.org/abs/2512.22304v1)

**作者:** Darrin Bright, Rakshith Raj, Kanchan Keisham

**发布时间:** 2025-12-26

**备注:** Accepted at the 11th Annual Conference on Vision and Intelligent Systems (CVIS 2025)

### GPT解析

### 总结

本文提出了一种名为PortionNet的跨模态知识蒸馏框架，能够在不使用深度传感器的情况下，仅通过RGB图像准确估计食物营养成分。

### 背景

从单张图像准确估计食物营养成分具有挑战性，因为会丢失3D信息。基于深度的方法虽然能提供可靠的几何信息，但由于需要深度传感器，在大多数智能手机上无法使用。

### 目的

开发一种不需要深度传感器就能从RGB图像估计食物营养成分的方法。

### 方法

提出PortionNet，一种新的跨模态知识蒸馏框架。在训练过程中从点云学习几何特征，而在推理时只需要RGB图像。采用双模式训练策略，其中轻量级适配器网络模仿点云表示，无需任何专用硬件要求即可实现伪3D推理。

### 主要发现

PortionNet在MetaFood3D数据集上达到了最先进的性能，在体积和能量估计方面都优于所有先前的方法。在SimpleFood45数据集上的跨数据集评估进一步证明了其在能量估计方面的强泛化能力。

### 结论

PortionNet是一种有效的跨模态知识蒸馏框架，可以在不需要深度传感器的情况下准确估计食物营养成分。

### 翻译

从单张图像准确估计食物营养成分具有挑战性，因为会丢失3D信息。虽然基于深度的方法能提供可靠的几何信息，但由于需要深度传感器，在大多数智能手机上仍然无法使用。为克服这一挑战，我们提出了PortionNet，一种新颖的跨模态知识蒸馏框架，在训练过程中从点云学习几何特征，而在推理时只需要RGB图像。我们的方法采用双模式训练策略，其中轻量级适配器网络模仿点云表示，无需任何专用硬件要求即可实现伪3D推理。PortionNet在MetaFood3D上取得了最先进的性能，在体积和能量估计方面都优于所有先前方法。在SimpleFood45上的跨数据集评估进一步证明了其在能量估计方面的强泛化能力。


### 论文摘要

Accurate food nutrition estimation from single images is challenging due to the loss of 3D information. While depth-based methods provide reliable geometry, they remain inaccessible on most smartphones because of depth-sensor requirements. To overcome this challenge, we propose PortionNet, a novel cross-modal knowledge distillation framework that learns geometric features from point clouds during training while requiring only RGB images at inference. Our approach employs a dual-mode training strategy where a lightweight adapter network mimics point cloud representations, enabling pseudo-3D reasoning without any specialized hardware requirements. PortionNet achieves state-of-the-art performance on MetaFood3D, outperforming all previous methods in both volume and energy estimation. Cross-dataset evaluation on SimpleFood45 further demonstrates strong generalization in energy estimation.

---

## 33. RoboMirror: Understand Before You Imitate for Video to Humanoid Locomotion

**论文链接:** [http://arxiv.org/abs/2512.23649v1](http://arxiv.org/abs/2512.23649v1)

**作者:** Zhe Li, Cheng Chi, Yangyang Wei, Boan Zhu, Tao Huang, Zhenguo Sun, Yibo Peng, Pengwei Wang, Zhongyuan Wang, Fangzhou Liu, Chang Xu, Shanghang Zhang

**发布时间:** 2025-12-29

### GPT解析

### 总结

该研究提出了RoboMirror，这是一个无需重定位的视频到运动框架，实现了'先理解后模仿'的理念。它利用视觉语言模型将原始视频提炼为视觉运动意图，直接基于扩散策略生成物理上合理、语义对齐的运动。

### 背景

当前最先进的人形运动系统依赖于精心制作的运动捕捉轨迹或稀疏文本命令，在视觉理解和控制之间存在关键差距。文本到运动方法存在语义稀疏和流水线错误问题，而基于视频的方法仅执行机械姿态模仿，缺乏真正的视觉理解。

### 目的

开发一个能够弥合视觉理解和动作之间差距的框架，使机器人能够通过视觉观察来学习运动，像人类一样先理解后模仿。

### 方法

提出RoboMirror框架，利用视觉语言模型将原始的第一人称/第三人称视频提炼为视觉运动意图，这些意图直接条件化一个基于扩散的策略，以生成物理上合理、语义对齐的运动，无需明确的姿态重建或重定位。

### 主要发现

广泛的实验验证了RoboMirror的有效性，它能够通过第一人称视频实现远程呈现，将第三人称控制延迟大幅减少80%，并且比基线方法高3.7%的任务成功率。

### 结论

通过围绕视频理解重新构建人形控制，RoboMirror成功弥合了视觉理解和动作之间的差距。

### 翻译

人类通过视觉观察学习运动，在模仿动作之前先解释视觉内容。然而，最先进的人形运动系统依赖于精心制作的运动捕捉轨迹或稀疏文本命令，在视觉理解和控制之间留下了关键差距。文本到运动方法遭受语义稀疏和流水线错误，而基于视频的方法仅执行机械姿态模仿，没有真正的视觉理解。我们提出了RoboMirror，这是第一个无需重定位的视频到运动框架，体现了'先理解后模仿'的理念。利用VLMs，它将原始的第一人称/第三人称视频提炼为视觉运动意图，这些意图直接条件化一个基于扩散的策略，以生成物理上合理、语义对齐的运动，无需明确的姿态重建或重定位。广泛的实验验证了RoboMirror的有效性，它能够通过第一人称视频实现远程呈现，大幅将第三人称控制延迟减少80%，并且比基线方法高3.7%的任务成功率。通过围绕视频理解重新构建人形控制，我们弥合了视觉理解和动作之间的差距。


### 论文摘要

Humans learn locomotion through visual observation, interpreting visual content first before imitating actions. However, state-of-the-art humanoid locomotion systems rely on either curated motion capture trajectories or sparse text commands, leaving a critical gap between visual understanding and control. Text-to-motion methods suffer from semantic sparsity and staged pipeline errors, while video-based approaches only perform mechanical pose mimicry without genuine visual understanding. We propose RoboMirror, the first retargeting-free video-to-locomotion framework embodying "understand before you imitate". Leveraging VLMs, it distills raw egocentric/third-person videos into visual motion intents, which directly condition a diffusion-based policy to generate physically plausible, semantically aligned locomotion without explicit pose reconstruction or retargeting. Extensive experiments validate the effectiveness of RoboMirror, it enables telepresence via egocentric videos, drastically reduces third-person control latency by 80%, and achieves a 3.7% higher task success rate than baselines. By reframing humanoid control around video understanding, we bridge the visual understanding and action gap.

---

## 34. OmniAgent: Audio-Guided Active Perception Agent for Omnimodal Audio-Video Understanding

**论文链接:** [http://arxiv.org/abs/2512.23646v1](http://arxiv.org/abs/2512.23646v1)

**作者:** Keda Tao, Wenjie Du, Bohan Yu, Weiqiang Wang, Jian Liu, Huan Wang

**发布时间:** 2025-12-29

**备注:** Website:https://kd-tao.github.io/OmniAgent/

### GPT解析

### 总结

OmniAgent是一个完全由音频引导的主动感知代理，通过动态协调专业工具实现细粒度音频视觉推理，采用从粗到细的音频引导感知范式，在多个基准测试上取得了最先进的性能。

### 背景

多模态大语言模型在统一音频和视觉模态方面取得了显著进展，但通常缺乏细粒度的跨模态理解，并且难以处理多模态对齐问题。

### 目的

为了解决现有模型在细粒度跨模态理解和多模态对齐方面的局限性，引入OmniAgent实现更细粒度的音频视觉推理。

### 方法

OmniAgent采用动态规划来自主按需协调工具调用，战略性地将感知注意力集中在任务相关线索上，核心是一种从粗到细的音频引导感知范式，利用音频线索定位时间事件并指导后续推理。

### 主要发现

在三个音频视频理解基准上的大量经验评估表明，OmniAgent取得了最先进的性能，以10% - 20%的准确率优势超越了领先的开源和专有模型。

### 结论

论文展示了一种从被动响应生成到主动多模态查询的范式转变，通过动态协调专业工具解决了现有模型的不足。

### 翻译

多模态大语言模型在统一音频和视觉模态方面取得了显著进展；然而，它们通常缺乏细粒度的跨模态理解，并且难以处理多模态对齐。为了解决这些限制，我们引入了OmniAgent，一个完全由音频引导的主动感知代理，动态协调专业工具以实现更细粒度的音频视觉推理。与依赖刚性静态流程和密集帧字幕的先前工作不同，本文展示了从被动响应生成到主动多模态查询的范式转变。OmniAgent采用动态规划自主按需协调工具调用，战略性地将感知注意力集中在任务相关的线索上。我们方法的核心是一种新颖的从粗到细的音频引导感知范式，利用音频线索定位时间事件并指导后续推理。在三个音频视频理解基准上的大量经验评估表明，OmniAgent取得了最先进的性能，以10% - 20%的准确率优势大幅超越了领先的开源和专有模型。


### 论文摘要

Omnimodal large language models have made significant strides in unifying audio and visual modalities; however, they often lack the fine-grained cross-modal understanding and have difficulty with multimodal alignment. To address these limitations, we introduce OmniAgent, a fully audio-guided active perception agent that dynamically orchestrates specialized tools to achieve more fine-grained audio-visual reasoning. Unlike previous works that rely on rigid, static workflows and dense frame-captioning, this paper demonstrates a paradigm shift from passive response generation to active multimodal inquiry. OmniAgent employs dynamic planning to autonomously orchestrate tool invocation on demand, strategically concentrating perceptual attention on task-relevant cues. Central to our approach is a novel coarse-to-fine audio-guided perception paradigm, which leverages audio cues to localize temporal events and guide subsequent reasoning. Extensive empirical evaluations on three audio-video understanding benchmarks demonstrate that OmniAgent achieves state-of-the-art performance, surpassing leading open-source and proprietary models by substantial margins of 10% - 20% accuracy.

---

## 35. Rethinking the Spatio-Temporal Alignment of End-to-End 3D Perception

**论文链接:** [http://arxiv.org/abs/2512.23635v1](http://arxiv.org/abs/2512.23635v1)

**作者:** Xiaoyu Li, Peidong Li, Xian Wu, Long Shi, Dedong Liu, Yitao Wu, Jiajia Fu, Dixiao Cui, Lijun Zhao, Lining Sun

**发布时间:** 2025-12-29

**备注:** Accepted to AAAI 2026

### GPT解析

### 总结

该研究提出了一种名为HAT的时空对齐模块，用于解决自动驾驶端到端感知中的时空对齐问题。该模块允许每个物体从多个假设中自适应解码最佳对齐方案，无需直接监督，结合了显式运动模型和语义特征，显著提高了感知精度和跟踪性能。

### 背景

在自动驾驶的端到端感知中，时空对齐对时间建模至关重要，能提供有价值的结构和纹理先验信息。现有方法通常依赖注意力机制跨帧对齐物体，使用统一的显式物理模型简化运动模型，倾向于使用语义特征进行隐式对齐，但这种方法在不同类别和帧中物体运动状态和特征变化时表现次优。

### 目的

开发一种时空对齐模块，使每个物体能够从多个假设中自适应解码最佳对齐提议，无需直接监督，解决现有方法在不同条件下对齐效果不佳的问题。

### 方法

HAT模块首先使用多个显式运动模型为历史实例生成空间锚点和运动感知特征提议，然后通过整合缓存的物体查询中嵌入的语义和运动线索进行多假设解码，最终为目标帧提供最佳对齐提议。

### 主要发现

在nuScenes数据集上，HAT在不同基线上持续改进3D时间检测器和跟踪器；与DETR3D检测器配对时，在测试集上达到46.0%的AMOTA最先进跟踪结果；在以物体为中心的端到端自动驾驶方法中，HAT将感知精度提高+1.3% mAP和+3.1% AMOTA，并将碰撞率降低32%；当语义信息被破坏时，HAT仍能保持感知和规划的鲁棒性。

### 结论

HAT通过结合显式运动模型和语义特征，有效解决了自动驾驶中时空对齐的挑战，在不同条件下都能提高感知和跟踪性能，特别是在语义信息受损的情况下仍能保持鲁棒性，为自动驾驶的端到端感知提供了更可靠的解决方案。

### 翻译

时空对齐对于自动驾驶中端到端感知的时间建模至关重要，提供了有价值的结构和纹理先验信息。现有方法通常依赖注意力机制来跨帧对齐物体，使用统一的显式物理模型（如恒定速度）简化运动模型。这些方法倾向于使用语义特征进行隐式对齐，挑战了传统感知范式中显式运动建模的重要性。然而，不同类别和帧中物体的运动状态和特征变化导致这种对齐次优。为解决此问题，我们提出HAT，一种时空对齐模块，使每个物体能够从多个假设中自适应解码最佳对齐提议，无需直接监督。具体而言，HAT首先使用多个显式运动模型为历史实例生成空间锚点和运动感知特征提议。然后通过整合缓存的物体查询中嵌入的语义和运动线索进行多假设解码，最终为目标帧提供最佳对齐提议。在nuScenes上，HAT在不同基线上持续改进3D时间检测器和跟踪器。当与DETR3D检测器配对时，在测试集上达到46.0% AMOTA的最先进跟踪结果。在以物体为中心的端到端自动驾驶方法中，HAT提高了感知精度（+1.3% mAP，+3.1% AMOTA）并将碰撞率降低32%。当语义信息被破坏时（nuScenes-C），HAT通过增强运动建模使端到端自动驾驶的感知和规划更加鲁棒。


### 论文摘要

Spatio-temporal alignment is crucial for temporal modeling of end-to-end (E2E) perception in autonomous driving (AD), providing valuable structural and textural prior information. Existing methods typically rely on the attention mechanism to align objects across frames, simplifying the motion model with a unified explicit physical model (constant velocity, etc.). These approaches prefer semantic features for implicit alignment, challenging the importance of explicit motion modeling in the traditional perception paradigm. However, variations in motion states and object features across categories and frames render this alignment suboptimal. To address this, we propose HAT, a spatio-temporal alignment module that allows each object to adaptively decode the optimal alignment proposal from multiple hypotheses without direct supervision. Specifically, HAT first utilizes multiple explicit motion models to generate spatial anchors and motion-aware feature proposals for historical instances. It then performs multi-hypothesis decoding by incorporating semantic and motion cues embedded in cached object queries, ultimately providing the optimal alignment proposal for the target frame. On nuScenes, HAT consistently improves 3D temporal detectors and trackers across diverse baselines. It achieves state-of-the-art tracking results with 46.0% AMOTA on the test set when paired with the DETR3D detector. In an object-centric E2E AD method, HAT enhances perception accuracy (+1.3% mAP, +3.1% AMOTA) and reduces the collision rate by 32%. When semantics are corrupted (nuScenes-C), the enhancement of motion modeling by HAT enables more robust perception and planning in the E2E AD.

---

## 36. A Context-Aware Temporal Modeling through Unified Multi-Scale Temporal Encoding and Hierarchical Sequence Learning for Single-Channel EEG Sleep Staging

**论文链接:** [http://arxiv.org/abs/2512.22976v1](http://arxiv.org/abs/2512.22976v1)

**作者:** Amirali Vakili, Salar Jahanshiri, Armin Salimi-Badr

**发布时间:** 2025-12-28

### GPT解析

### 总结

该研究提出了一种针对单通道EEG的上下文感知且可解释的睡眠分期框架，通过结合多尺度特征提取和时间建模，并采用类别加权损失和数据增强解决数据不平衡问题，在保持可解释性的同时显著提高了睡眠分期性能，特别是在N1阶段检测方面。

### 背景

自动睡眠分期是医疗保健中的关键任务，因为全球睡眠障碍普遍存在。单通道脑电图(EEG)是一种实用且广泛可用的信号，用于自动睡眠分期。现有方法面临类别不平衡、感受野建模有限和可解释性不足等挑战。

### 目的

提出一种上下文感知且可解释的框架，用于单通道EEG睡眠分期，特别强调提高N1阶段的检测能力，解决先前模型作为黑盒操作的问题，提供明确且可解释的特征提取作用。

### 方法

结合紧凑的多尺度特征提取与时间建模，以捕捉局部和长程依赖关系；使用类别加权损失函数和数据增强解决数据不平衡问题；将EEG信号分割为子时段块，通过跨块平均softmax概率获得最终预测，增强上下文表示和鲁棒性。

### 主要发现

所提出的框架总体准确率达到89.72%，宏平均F1得分为85.46%；在具有挑战性的N1阶段达到61.7%的F1分数；在SleepEDF数据集上相比之前的方法有显著改进。

### 结论

所提出的方法有效提高了睡眠分期性能，同时保持了可解释性，适合实际临床应用。

### 翻译

自动睡眠分期是医疗保健中的一个关键任务，因为全球睡眠障碍普遍存在。本研究专注于单通道脑电图(EEG)，这是一种实用且广泛可用的自动睡眠分期信号。现有方法面临类别不平衡、感受野建模有限和可解释性不足等挑战。这项工作提出了一个用于单通道EEG睡眠分期的上下文感知和可解释框架，特别强调提高N1阶段的检测。许多先前的模型作为具有堆叠层的黑盒运行，缺乏明确且可解释的特征提取作用。所提出的模型结合了紧凑的多尺度特征提取与时间建模，以捕捉局部和长程依赖关系。为了解决数据不平衡问题，特别是N1阶段，应用了类别加权损失函数和数据增强。EEG信号被分割为子时段块，通过跨块平均softmax概率获得最终预测，增强了上下文表示和鲁棒性。所提出的框架总体准确率达到89.72%，宏平均F1得分为85.46%。值得注意的是，它在具有挑战性的N1阶段达到了61.7%的F1分数，在SleepEDF数据集上相比先前方法有显著改进。这些结果表明，所提出的方法在保持可解释性和适合实际临床应用的同时，有效提高了睡眠分期性能。


### 论文摘要

Automatic sleep staging is a critical task in healthcare due to the global prevalence of sleep disorders. This study focuses on single-channel electroencephalography (EEG), a practical and widely available signal for automatic sleep staging. Existing approaches face challenges such as class imbalance, limited receptive-field modeling, and insufficient interpretability. This work proposes a context-aware and interpretable framework for single-channel EEG sleep staging, with particular emphasis on improving detection of the N1 stage. Many prior models operate as black boxes with stacked layers, lacking clearly defined and interpretable feature extraction roles.The proposed model combines compact multi-scale feature extraction with temporal modeling to capture both local and long-range dependencies. To address data imbalance, especially in the N1 stage, classweighted loss functions and data augmentation are applied. EEG signals are segmented into sub-epoch chunks, and final predictions are obtained by averaging softmax probabilities across chunks, enhancing contextual representation and robustness.The proposed framework achieves an overall accuracy of 89.72% and a macro-average F1-score of 85.46%. Notably, it attains an F1- score of 61.7% for the challenging N1 stage, demonstrating a substantial improvement over previous methods on the SleepEDF datasets. These results indicate that the proposed approach effectively improves sleep staging performance while maintaining interpretability and suitability for real-world clinical applications.

---

## 37. 论文ID: 2512.22315v1

**错误:** 无法读取论文信息 - [Errno 2] No such file or directory: 'storage/result/2512.22315v1.json'

---

## 38. 论文ID: 2512.21734v2

**错误:** 无法读取论文信息 - [Errno 2] No such file or directory: 'storage/result/2512.21734v2.json'

---

## 39. 论文ID: 2512.22226v1

**错误:** 无法读取论文信息 - [Errno 2] No such file or directory: 'storage/result/2512.22226v1.json'

---

## 40. 论文ID: 2512.23545v1

**错误:** 无法读取论文信息 - [Errno 2] No such file or directory: 'storage/result/2512.23545v1.json'

---

## 41. Visual Language Hypothesis

**论文链接:** [http://arxiv.org/abs/2512.23335v1](http://arxiv.org/abs/2512.23335v1)

**作者:** Xiu Li

**发布时间:** 2025-12-29

### GPT解析

### 总结

本文从结构和拓扑角度研究视觉表征学习，提出视觉理解需要视觉语义语言，视觉观测空间应组织为纤维束结构，并推导出两个理论结果：语义商空间不能仅通过光滑变形获得，需要区分性目标；近似商空间对模型架构有结构要求，需要支持拓扑变化的表征机制。

### 背景

视觉表征学习中的可迁移性和抽象性假设，以及视觉理解需要语义语言的观点。

### 目的

研究视觉表征学习的结构和拓扑特性，理解视觉观测空间的组织方式，以及语义抽象的实现机制。

### 方法

从视觉语义语言的基本假设出发，结合表征学习的前提，推导视觉观测空间的纤维束结构，并分析其对语义不变性和模型架构的要求。

### 主要发现

语义商空间不是原空间的子流形，不能仅通过光滑变形获得，需要非同胚的区分性目标；语义抽象需要能够支持拓扑变化的表征机制，包括扩展-捕捉过程。

### 结论

该框架提供了理解视觉表征学习的拓扑透镜，与大规模判别性和多模态模型中的经验规律以及统计学习理论中的经典原理一致，具有解释性而非规定性。

### 翻译

我们从结构和拓扑的角度研究视觉表征学习。我们从一个单一假设出发：视觉理解需要一个视觉语义语言，其中许多感知观测对应于少量离散的语义状态。结合表征学习中广泛假设的可迁移性和抽象性前提，这个假设意味着视觉观测空间必须以类似纤维束的结构组织，其中干扰变量填充纤维，语义对应于商基空间。从这个结构我们推导出两个理论结果。首先，语义商空间不是原空间的子流形，不能仅通过光滑变形获得，语义不变性需要一个非同胚的、有区分性的目标。其次，我们表明近似商空间也对模型架构提出了结构要求。语义抽象不仅需要外部语义目标，还需要能够支持拓扑变化的表征机制：一个扩展-捕捉过程，其中流形首先被几何扩展以分离结构，然后被折叠以形成离散的语义区域。我们强调这些结果是解释性的而非规定性的：该框架提供了一个拓扑透镜，与大规模判别性和多模态模型中观察到的经验规律以及统计学习理论中的经典原理一致。


### 论文摘要

We study visual representation learning from a structural and topological perspective. We begin from a single hypothesis: that visual understanding presupposes a semantic language for vision, in which many perceptual observations correspond to a small number of discrete semantic states. Together with widely assumed premises on transferability and abstraction in representation learning, this hypothesis implies that the visual observation space must be organized in a fiber bundle like structure, where nuisance variation populates fibers and semantics correspond to a quotient base space. From this structure we derive two theoretical consequences. First, the semantic quotient $X/G$ is not a submanifold of $X$ and cannot be obtained through smooth deformation alone, semantic invariance requires a non-homeomorphic, discriminative target, for example, supervision via labels, cross instance identification, or multimodal alignment that supplies explicit semantic equivalence. Second, we show that approximating the quotient also places structural demands on the model architecture. Semantic abstraction requires not only an external semantic target, but a representation mechanism capable of supporting topology change: an expand-and-snap process in which the manifold is first geometrically expanded to separate structure and then collapsed to form discrete semantic regions. We emphasize that these results are interpretive rather than prescriptive: the framework provides a topological lens that aligns with empirical regularities observed in large-scale discriminative and multimodal models, and with classical principles in statistical learning theory.

---

## 42. Diffusion-based Decentralized Federated Multi-Task Representation Learning

**论文链接:** [http://arxiv.org/abs/2512.23161v1](http://arxiv.org/abs/2512.23161v1)

**作者:** Donghwa Kang, Shana Moothedath

**发布时间:** 2025-12-29

### GPT解析

### 总结

该论文提出了一种基于投影梯度下降的去中心化算法，用于多任务表示学习，特别是在数据稀缺环境中的应用。

### 背景

表示学习是一种广泛采用的框架，用于在数据稀缺环境中从各种不同但相关的任务中获取特征提取器。尽管表示学习已有大量研究，但去中心化的方法仍然相对未被充分探索。

### 目的

开发一种基于投影梯度下降的去中心化算法，用于多任务表示学习，特别是在多任务线性回归场景中。

### 方法

专注于多任务线性回归问题，其中多个线性回归模型共享一个共同的低维线性表示。提出了一种交替投影梯度下降和最小化算法，用于在基于扩散的去中心化和联邦方式中恢复低秩特征矩阵。

### 主要发现

获得了建设性的、可证明的保证，为所提出的算法提供了所需样本复杂度的下限和迭代复杂度的上限。算法在时间和通信复杂度方面表现出快速且高效的特性。

### 结论

通过数值模拟验证了算法的性能，并将其与基准算法进行了比较，证明了该方法的有效性。

### 翻译

表示学习是一种广泛采用的框架，用于在数据稀缺环境中学习，从各种不同但相关的任务中获取特征提取器或表示。尽管表示学习已有大量研究，但去中心化的方法仍然相对未被充分探索。本文开发了一种基于投影梯度下降的去中心化算法，用于多任务表示学习。我们专注于多任务线性回归问题，其中多个线性回归模型共享一个共同的低维线性表示。我们提出了一种交替投影梯度下降和最小化算法，用于在基于扩散的去中心化和联邦方式中恢复低秩特征矩阵。我们获得了建设性的、可证明的保证，为所提出的算法提供了所需样本复杂度的下限和迭代复杂度的上限。我们分析了算法的时间和通信复杂度，表明它是快速且通信高效的。我们进行了数值模拟以验证算法的性能，并将其与基准算法进行了比较。


### 论文摘要

Representation learning is a widely adopted framework for learning in data-scarce environments to obtain a feature extractor or representation from various different yet related tasks. Despite extensive research on representation learning, decentralized approaches remain relatively underexplored. This work develops a decentralized projected gradient descent-based algorithm for multi-task representation learning. We focus on the problem of multi-task linear regression in which multiple linear regression models share a common, low-dimensional linear representation. We present an alternating projected gradient descent and minimization algorithm for recovering a low-rank feature matrix in a diffusion-based decentralized and federated fashion. We obtain constructive, provable guarantees that provide a lower bound on the required sample complexity and an upper bound on the iteration complexity of our proposed algorithm. We analyze the time and communication complexity of our algorithm and show that it is fast and communication-efficient. We performed numerical simulations to validate the performance of our algorithm and compared it with benchmark algorithms.

---

## 43. Embodied Robot Manipulation in the Era of Foundation Models: Planning and Learning Perspectives

**论文链接:** [http://arxiv.org/abs/2512.22983v1](http://arxiv.org/abs/2512.22983v1)

**作者:** Shuanghao Bai, Wenxuan Song, Jiayi Chen, Yuheng Ji, Zhide Zhong, Jin Yang, Han Zhao, Wanqi Zhou, Zhe Li, Pengxiang Ding, Cheng Chi, Chang Xu, Xiaolong Zheng, Donglin Wang, Haoang Li, Shanghang Zhang, Badong Chen

**发布时间:** 2025-12-28

**备注:** This work is a re-architected core derived from the full survey (arXiv:2510.10903) , refined to highlight the most central themes and representative studies

### GPT解析

### 总结

这篇综述从算法角度审视了机器人操作领域，将基于学习的方法分为高层规划和底层控制两部分。在高层，扩展了任务规划概念以包含多种推理形式；在底层，提出了基于训练范式的分类法。文章还确定了开放挑战和未来研究方向。

### 背景

视觉、语言和多模态学习的最新进展极大地推动了机器人基础模型的发展，其中机器人操作仍然是一个核心且具有挑战性的问题。

### 目的

从算法角度审视机器人操作，组织基于学习的方法，确定开放挑战和未来研究方向。

### 方法

将基于学习的方法组织在一个统一的抽象框架下，分为高层规划和底层控制两个层面。高层包括语言、代码、运动、可供性和3D表示的推理；底层按照输入建模、潜在表示学习和策略学习进行分类。

### 主要发现

扩展了经典任务规划概念，提出了基于训练范式的控制方法分类法，确定了可扩展性、数据效率、多模态物理交互和安全方面的开放挑战。

### 结论

这些分析旨在阐明现代机器人操作基础模型的设计空间。

### 翻译

视觉、语言和多模态学习的最新进展极大地推动了机器人基础模型的进步，其中机器人操作仍然是一个核心且具有挑战性的问题。本综述从算法角度审视机器人操作，并将基于学习的方法组织在一个高层规划和底层控制的统一抽象框架中。在高层，我们将经典的任务规划概念扩展到包括对语言、代码、运动、可供性和3D表示的推理，强调它们在结构化和长期决策中的作用。在底层，我们提出了一个基于训练范式的分类法用于基于学习的控制，沿着输入建模、潜在表示学习和策略学习组织现有方法。最后，我们确定了与可扩展性、数据效率、多模态物理交互和安全相关的开放挑战和未来研究方向。这些分析共同旨在阐明现代机器人操作基础模型的设计空间。


### 论文摘要

Recent advances in vision, language, and multimodal learning have substantially accelerated progress in robotic foundation models, with robot manipulation remaining a central and challenging problem. This survey examines robot manipulation from an algorithmic perspective and organizes recent learning-based approaches within a unified abstraction of high-level planning and low-level control. At the high level, we extend the classical notion of task planning to include reasoning over language, code, motion, affordances, and 3D representations, emphasizing their role in structured and long-horizon decision making. At the low level, we propose a training-paradigm-oriented taxonomy for learning-based control, organizing existing methods along input modeling, latent representation learning, and policy learning. Finally, we identify open challenges and prospective research directions related to scalability, data efficiency, multimodal physical interaction, and safety. Together, these analyses aim to clarify the design space of modern foundation models for robotic manipulation.

---

## 44. 论文ID: 2512.22730v1

**错误:** 无法读取论文信息 - [Errno 2] No such file or directory: 'storage/result/2512.22730v1.json'

---

## 45. 论文ID: 2512.22712v1

**错误:** 无法读取论文信息 - [Errno 2] No such file or directory: 'storage/result/2512.22712v1.json'

---

## 46. Learning with the $p$-adics

**论文链接:** [http://arxiv.org/abs/2512.22692v1](http://arxiv.org/abs/2512.22692v1)

**作者:** André F. T. Martins

**发布时间:** 2025-12-27

**备注:** 29 pages

### GPT解析

### 总结

这篇论文研究了将p-adic数作为替代实数域的机器学习框架的适用性，建立了基于p-adic数的分类、回归和表示学习的理论基础，并展示了如何将Quillian语义网络表示为紧凑的p-adic线性网络。

### 背景

现有的机器学习框架在实数域上运行，并在实数向量空间中学习表示，这些几何属性与线性可分性、最小包围球和子空间投影等直观概念相吻合，并且基于梯度的优化方法提供了学习工具。

### 目的

探索p-adic数这一非阿基米德空间作为实数域替代方案的适用性，利用其层次结构和无限字符串解释的特性，为编码理论和层次表示学习提供新工具。

### 方法

进行了探索性理论研究，建立了基于p-adic数的分类、回归和表示学习的构建模块，提供了学习模型和算法。展示了如何将简单的Quillian语义网络表示为紧凑的p-adic线性网络。

### 主要发现

p-adic数的层次结构和无限字符串特性使其成为编码理论和层次表示学习的有吸引力的工具。Quillian语义网络可以表示为紧凑的p-adic线性网络，而这种构造在实数域中是不可能的。

### 结论

p-adic数提供了一种新的机器学习框架，为未来研究开辟了新的可能性和开放性问题。

### 翻译

现有的机器学习框架在实数域上运行，并在实数向量空间中学习表示。它们的基本几何属性与线性可分性、最小包围球和子空间投影等直观概念相吻合；基础微积分提供了通过基于梯度的优化学术习的工具。但这是唯一可能的选择吗？在本文中，我们研究了一个截然不同的域作为替代方案的适用性——p-adic数的非阿基米德空间。p-adic数的层次结构及其作为无限字符串的解释使它们成为编码理论和层次表示学习的有吸引力的工具。我们的探索性理论研究建立了基于p-adic数的分类、回归和表示学习的构建模块，提供了学习模型和算法。我们展示了如何将简单的Quillian语义网络表示为紧凑的p-adic线性网络，而这种构造在实数域中是不可能的。最后，我们讨论了这一新框架带来的开放性问题和未来研究机会。


### 论文摘要

Existing machine learning frameworks operate over the field of real numbers ($\mathbb{R}$) and learn representations in real (Euclidean or Hilbert) vector spaces (e.g., $\mathbb{R}^d$). Their underlying geometric properties align well with intuitive concepts such as linear separability, minimum enclosing balls, and subspace projection; and basic calculus provides a toolbox for learning through gradient-based optimization.   But is this the only possible choice? In this paper, we study the suitability of a radically different field as an alternative to $\mathbb{R}$ -- the ultrametric and non-archimedean space of $p$-adic numbers, $\mathbb{Q}_p$. The hierarchical structure of the $p$-adics and their interpretation as infinite strings make them an appealing tool for code theory and hierarchical representation learning. Our exploratory theoretical work establishes the building blocks for classification, regression, and representation learning with the $p$-adics, providing learning models and algorithms. We illustrate how simple Quillian semantic networks can be represented as a compact $p$-adic linear network, a construction which is not possible with the field of reals. We finish by discussing open problems and opportunities for future research enabled by this new framework.

---

## 47. Beyond Centralization: Provable Communication Efficient Decentralized Multi-Task Learning

**论文链接:** [http://arxiv.org/abs/2512.22675v1](http://arxiv.org/abs/2512.22675v1)

**作者:** Donghwa Kang, Shana Moothedath

**发布时间:** 2025-12-27

### GPT解析

### 总结

本文研究了在数据稀缺环境下的去中心化多任务表示学习方法，提出了一种具有可证明准确性保证的新算法，其通信复杂度与目标精度无关，显著降低了通信成本。

### 背景

表示学习是在数据稀缺环境中广泛采用的框架，旨在从相关任务中提取共同特征。虽然集中式方法已被广泛研究，但去中心化方法在很大程度上仍未被探索。在去中心化设置中，任务数据分布在多个节点上，节点间的信息交换受到通信网络限制。

### 目的

研究特征共享低秩结构的去中心化多任务表示学习，目标是恢复潜在的低秩特征矩阵，并提出具有可证明准确性保证的算法，同时全面表征时间、通信和样本复杂度。

### 方法

提出了一种新的交替投影梯度最小化算法，该算法在去中心化环境中工作，考虑多个任务，每个任务有有限数量的数据样本，观测值遵循具有任务特定参数的线性模型。

### 主要发现

1) 所提算法具有可证明的准确性保证；2) 通信复杂度与目标精度无关，显著降低通信成本；3) 数值模拟在不同维度和网络拓扑下验证了理论分析；4) 展示了去中心化学习在某些情况下优于集中式联邦方法的场景。

### 结论

去中心化多任务表示学习是数据稀缺环境下的有效方法，特别是在通信受限环境中。所提算法能在保持低通信复杂度的同时有效恢复低秩特征结构，为去中心化学习提供了新的理论和实践见解。

### 翻译

表示学习是在数据稀缺环境中广泛采用的框架，旨在从相关任务中提取共同特征。虽然集中式方法已被广泛研究，但去中心化方法在很大程度上仍未被探索。我们研究了特征共享低秩结构的去中心化多任务表示学习。我们考虑多个任务，每个任务有有限数量的数据样本，其中观测值遵循具有任务特定参数的线性模型。在去中心化设置中，任务数据分布在多个节点上，节点之间的信息交换受到通信网络的限制。目标是恢复潜在的特征矩阵，该矩阵的秩远小于参数维度和任务数量。我们提出了一种具有可证明准确性保证的新交替投影梯度最小化算法。我们全面表征了时间、通信和样本复杂度。重要的是，通信复杂度与目标精度无关，与先前方法相比显著降低了通信成本。数值模拟在不同维度和网络拓扑下验证了理论分析，并展示了去中心化学习在某些情况下优于集中式联邦方法的场景。


### 论文摘要

Representation learning is a widely adopted framework for learning in data-scarce environments, aiming to extract common features from related tasks. While centralized approaches have been extensively studied, decentralized methods remain largely underexplored. We study decentralized multi-task representation learning in which the features share a low-rank structure. We consider multiple tasks, each with a finite number of data samples, where the observations follow a linear model with task-specific parameters. In the decentralized setting, task data are distributed across multiple nodes, and information exchange between nodes is constrained by a communication network. The goal is to recover the underlying feature matrix whose rank is much smaller than both the parameter dimension and the number of tasks. We propose a new alternating projected gradient and minimization algorithm with provable accuracy guarantees. We provide comprehensive characterizations of the time, communication, and sample complexities. Importantly, the communication complexity is independent of the target accuracy, which significantly reduces communication cost compared to prior methods. Numerical simulations validate the theoretical analysis across different dimensions and network topologies, and demonstrate regimes in which decentralized learning outperforms centralized federated approaches.

---

## 48. 论文ID: 2512.22664v1

**错误:** 无法读取论文信息 - [Errno 2] No such file or directory: 'storage/result/2512.22664v1.json'

---

## 49. Tracking by Predicting 3-D Gaussians Over Time

**论文链接:** [http://arxiv.org/abs/2512.22489v1](http://arxiv.org/abs/2512.22489v1)

**作者:** Tanish Baranwal, Himanshu Gaurav Singh, Jathushan Rajasegaran, Jitendra Malik

**发布时间:** 2025-12-27

### GPT解析

### 总结

本文提出Video Gaussian Masked Autoencoders (Video-GMAE)，一种自监督视频表示学习方法，通过将图像序列编码为随时间移动的高斯点集合来学习视频表示。

### 背景

视频表示学习需要有效捕捉时空信息，现有的自监督方法在视频理解和跟踪方面仍有提升空间。

### 目的

开发一种通过将视频表示为随时间变化的高斯集合来学习视频时空表示的方法，实现零样本跟踪和高效的视频表示学习。

### 方法

Video-GMAE将视频序列编码为随时间移动的高斯点集合，强制执行2D视频是动态3D场景一致投影的归纳偏置，通过预训练网络自然学习跟踪能力。

### 主要发现

1) 使用该架构预训练网络时跟踪能力自然涌现；2) 将高斯轨迹映射到图像平面可实现与最先进方法相当的零样本跟踪性能；3) 小规模微调后在Kinetics和Kubric数据集上分别实现34.6%和13.1%的改进；4) 结果超越了现有自监督视频方法。

### 结论

Video-GMAE通过将视频表示为随时间变化的高斯集合，有效学习了视频时空表示，实现了卓越的零样本跟踪性能和视频表示能力。

### 翻译

我们提出视频高斯掩码自编码器(Video-GMAE)，一种用于表示学习的自监督方法，它将图像序列编码为一组随时间移动的高斯点。将视频表示为一组高斯点强制执行了合理的归纳偏置：即2D视频通常是动态3D场景的一致投影。我们发现，使用这种架构预训练网络时会出现跟踪能力。将学习到的高斯轨迹映射到图像平面上可以实现零样本跟踪性能，与最先进方法相当。通过小规模微调，我们的模型在Kinetics数据集上实现了34.6%的改进，在Kubric数据集上实现了13.1%的改进，超越了现有的自监督视频方法。项目页面和代码可在https://videogmae.org/和https://github.com/tekotan/video-gmae公开获取。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决视频中的像素点跟踪问题，即在视频中持续追踪特定像素点位置的任务。这个问题在现实中非常重要，因为像素跟踪是计算机视觉的基础能力，对于理解视频内容、分析物体运动、场景结构等至关重要；同时，它也是许多高级视觉任务（如3D理解、计算摄影、长期推理等）的基础。传统方法通常需要大量标注数据，而本文提出的方法可以在无需大量标注的情况下实现高质量的像素跟踪。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先发现现有的自监督视频学习方法在点跟踪任务上表现不佳，认为传统的时空块预测目标不能强制执行时间一致性。他们注意到3D物体运动在图像平面上表现为点跟踪，由此产生灵感：通过预测随时间移动的3D高斯基元来学习对应关系。该方法借鉴了掩码自编码器（MAE）的架构、3D高斯溅射表示技术和自监督学习思想，但创新性地将这些技术结合用于视频表示学习和点跟踪任务。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是将视频表示为随时间移动的3D高斯基元的集合，这种表示强制了一个合理的归纳偏置：2D视频通常是动态3D场景的一致投影。通过预测高斯基元随时间的演变，模型能够学习像素级对应关系。整体流程包括：1）预训练阶段：输入16帧视频，编码器处理可见块，解码器预测第一帧的高斯基元和后续帧的高斯增量，通过高斯积分和渲染重建视频；2）零样本跟踪：将高斯轨迹映射到图像平面生成流场，用于点跟踪；3）监督微调：在点跟踪数据集上微调预训练模型，提升跟踪性能。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1）提出将视频表示为随时间演变的3D高斯集合的新方法；2）通过预测高斯基元的时间演变实现自监督跟踪学习；3）自然涌现出零样本跟踪能力；4）结合自监督学习和可微分渲染的统一框架。相比之前的工作，不同之处在于：传统自监督视频方法（如VideoMAE）不强制时间一致性，而本文通过高斯表示显式强制对应；其他自监督跟踪方法（如CRW、GMRW）使用随机游走或对比学习，而本文基于3D高斯表示提供更强几何约束；相比监督方法，本文需要更少标注数据且零样本性能接近或超过监督方法。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'Video-GMAE通过将视频表示为随时间演变的3D高斯集合，实现了自监督学习下的像素级对应关系，从而在无需标注数据的情况下实现了强大的零样本点跟踪能力，并在微调后达到了最先进的跟踪性能。'}


### 论文摘要

We propose Video Gaussian Masked Autoencoders (Video-GMAE), a self-supervised approach for representation learning that encodes a sequence of images into a set of Gaussian splats moving over time. Representing a video as a set of Gaussians enforces a reasonable inductive bias: that 2-D videos are often consistent projections of a dynamic 3-D scene. We find that tracking emerges when pretraining a network with this architecture. Mapping the trajectory of the learnt Gaussians onto the image plane gives zero-shot tracking performance comparable to state-of-the-art. With small-scale finetuning, our models achieve 34.6% improvement on Kinetics, and 13.1% on Kubric datasets, surpassing existing self-supervised video approaches. The project page and code are publicly available at https://videogmae.org/ and https://github.com/tekotan/video-gmae.

---

## 50. Toward Real-World IoT Security: Concept Drift-Resilient IoT Botnet Detection via Latent Space Representation Learning and Alignment

**论文链接:** [http://arxiv.org/abs/2512.22488v1](http://arxiv.org/abs/2512.22488v1)

**作者:** Hassan Wasswa, Timothy Lynar

**发布时间:** 2025-12-27

### GPT解析

### 总结

本文提出了一种可扩展的自适应物联网威胁检测框架，解决了基于AI的模型在动态物联网环境中的部署问题，避免了持续分类器重新训练的需要，同时保持了对概念漂移的鲁棒检测性能。

### 背景

基于AI的模型在物联网威胁检测中已取得高准确率，但在企业环境中的应用受到限制，因为这些模型依赖于静态数据集，无法反映真实物联网网络流量的动态特性。真实物联网网络流量经常受到概念漂移的影响，而现有解决方案通常依赖于定期分类器重新训练，导致高计算开销和灾难性遗忘的风险。

### 目的

为了解决这些挑战，本文提出了一种可扩展的自适应物联网威胁检测框架，消除了对连续分类器重新训练的需求，同时保持对概念漂移的鲁棒检测性能。

### 方法

所提出的方法首先在历史流量的潜在空间表示上训练一次分类器，同时一个对齐模型将传入流量映射到学习到的历史潜在空间，然后再进行分类，从而保留先前观察到的攻击知识。为了捕获攻击样本之间的实例关系，低维潜在表示被进一步转换为图结构格式，并使用图神经网络进行分类。

### 主要发现

在真实世界异构物联网流量数据集上的实验评估表明，所提出的框架在概念漂移条件下保持了稳健的检测性能。

### 结论

这些结果突显了该框架在动态和大规模物联网环境中实际部署的潜力。

### 翻译

虽然基于AI的模型在物联网威胁检测中已取得高准确率，但它们在企业环境中的应用受到限制，因为它们依赖于无法反映真实物联网网络流量动态特性的静态数据集，而真实物联网网络流量经常受到概念漂移的影响。现有解决方案通常依赖于定期分类器重新训练，导致高计算开销和灾难性遗忘的风险。为了解决这些挑战，本文提出了一种可扩展的自适应物联网威胁检测框架，消除了对连续分类器重新训练的需求。所提出的方法首先在历史流量的潜在空间表示上训练一次分类器，同时一个对齐模型将传入流量映射到学习到的历史潜在空间，然后再进行分类，从而保留先前观察到的攻击知识。为了捕获攻击样本之间的实例关系，低维潜在表示被进一步转换为图结构格式，并使用图神经网络进行分类。在真实世界异构物联网流量数据集上的实验评估表明，所提出的框架在概念漂移条件下保持了稳健的检测性能。这些结果突显了该框架在动态和大规模物联网环境中实际部署的潜力。


### 论文摘要

Although AI-based models have achieved high accuracy in IoT threat detection, their deployment in enterprise environments is constrained by reliance on stationary datasets that fail to reflect the dynamic nature of real-world IoT NetFlow traffic, which is frequently affected by concept drift. Existing solutions typically rely on periodic classifier retraining, resulting in high computational overhead and the risk of catastrophic forgetting. To address these challenges, this paper proposes a scalable framework for adaptive IoT threat detection that eliminates the need for continuous classifier retraining. The proposed approach trains a classifier once on latent-space representations of historical traffic, while an alignment model maps incoming traffic to the learned historical latent space prior to classification, thereby preserving knowledge of previously observed attacks. To capture inter-instance relationships among attack samples, the low-dimensional latent representations are further transformed into a graph-structured format and classified using a graph neural network. Experimental evaluations on real-world heterogeneous IoT traffic datasets demonstrate that the proposed framework maintains robust detection performance under concept drift. These results highlight the framework's potential for practical deployment in dynamic and large-scale IoT environments.

---

## 51. 论文ID: 2512.22331v1

**错误:** 无法读取论文信息 - [Errno 2] No such file or directory: 'storage/result/2512.22331v1.json'

---

## 52. Galaxy Zoo Evo: 1 million human-annotated images of galaxies

**论文链接:** [http://arxiv.org/abs/2512.23691v1](http://arxiv.org/abs/2512.23691v1)

**作者:** Mike Walmsley, Steven Bamford, Hugh Dickinson, Tobias Géron, Alexander J. Gordon, Annette M. N. Ferguson, Lucy Fortson, Sandor Kruk, Natalie Lines, Chris J. Lintott, Karen L. Masters, Robert G. Mann, James Pearson, Hayley Roberts, Anna M. M. Scaife, Stefan Schuldt, Brooke Simmons, Rebecca Smethurst, Josh Speagle, Kyle Willett

**发布时间:** 2025-12-29

**备注:** Submitted to NeurIPS Datasets and Benchmarks 2025. Positive reviews but rejected by AC; see OpenReview

### GPT解析

### 总结

本文介绍了一个名为Galaxy Zoo Evo的大型标注数据集，用于构建和评估星系图像的基础模型，包含823k张图像和104M众包标签，支持多种天文研究应用。

### 背景

天文学研究需要大量高质量的星系图像数据来训练和评估模型，特别是随着新空间望远镜如Euclid的发射，对基础模型的需求日益增长。

### 目的

创建一个大规模、精细标注的星系图像数据集，支持天文领域基础模型的开发、评估和特定下游任务，如寻找强引力透镜和描述新星系。

### 方法

收集来自四个望远镜的823k张星系图像，通过众包方式获得104M精细标签，每个图像都有详细的问题和答案描述，同时提供四个较小的特定任务标签集（共167k个星系）。

### 主要发现

Galaxy Zoo Evo数据集提供了详细的星系图像标注，可用于预训练或微调模型，支持域适应和学习不确定性等计算机视觉研究，为天文领域基础模型的发展提供了重要资源。

### 结论

Galaxy Zoo Evo将成为计算机视觉领域（特别是域适应和学习不确定性）的真实世界基准，支持新一代天文基础模型的发展，对未来天文学家更好地理解宇宙至关重要。

### 翻译

我们引入了Galaxy Zoo Evo，这是一个用于构建和评估星系图像基础模型的标注数据集。GZ Evo包含来自四个望远镜的823k张图像，共计104M的众包标签。每张图像都标注了一系列精细的问题和答案（例如'特征星系，两条旋臂，紧密缠绕，与另一个星系合并'）。这些详细标签可用于预训练或微调。我们还包含了四个较小的标签集（总共167k个星系），用于天文学家感兴趣的特定下游任务，包括寻找强引力透镜和描述来自新空间望远镜Euclid的星系。我们希望GZ Evo能作为计算机视觉主题（如域适应（从地面到天文，或望远镜之间）或从众包标签中学习不确定性）的真实世界基准。我们也希望它能支持天文领域新一代基础模型的发展；对于寻求更好理解我们宇宙的未来天文学家来说，这类模型将至关重要。


### 论文摘要

We introduce Galaxy Zoo Evo, a labeled dataset for building and evaluating foundation models on images of galaxies. GZ Evo includes 104M crowdsourced labels for 823k images from four telescopes. Each image is labeled with a series of fine-grained questions and answers (e.g. "featured galaxy, two spiral arms, tightly wound, merging with another galaxy"). These detailed labels are useful for pretraining or finetuning. We also include four smaller sets of labels (167k galaxies in total) for downstream tasks of specific interest to astronomers, including finding strong lenses and describing galaxies from the new space telescope Euclid. We hope GZ Evo will serve as a real-world benchmark for computer vision topics such as domain adaption (from terrestrial to astronomical, or between telescopes) or learning under uncertainty from crowdsourced labels. We also hope it will support a new generation of foundation models for astronomy; such models will be critical to future astronomers seeking to better understand our universe.

---

## 53. FRoD: Full-Rank Efficient Fine-Tuning with Rotational Degrees for Fast Convergence

**论文链接:** [http://arxiv.org/abs/2512.23485v1](http://arxiv.org/abs/2512.23485v1)

**作者:** Guoan Wan, Tianyu Chen, Fangzheng Feng, Haoyi Zhou, Runhua Xu

**发布时间:** 2025-12-29

**备注:** The 40th Annual AAAI Conference on Artificial Intelligence

### GPT解析

### 总结

FRoD是一种新型参数高效微调方法，结合分层联合分解和旋转自由度，通过提取跨层全局共享基并注入稀疏可学习扰动，实现灵活全秩更新，在保持高效的同时提升表达能力。

### 背景

参数高效微调(PEFT)方法通过只更新一小部分参数来减少计算和内存成本，使大型基础模型能适应下游任务。但现有方法如LoRA因固有的低秩约束而面临收敛速度慢和适应能力有限的问题。

### 目的

解决PEFT方法在捕捉多样化任务所需复杂模式方面的局限性，提高微调的表达能力和效率。

### 方法

提出FRoD方法，结合分层联合分解与旋转自由度，提取跨层全局共享基，向缩放因子中注入稀疏可学习扰动，实现灵活的全秩更新。

### 主要发现

FRoD在20个涵盖视觉、推理和语言理解的基准测试中，仅使用1.72%的可训练参数，就能达到与完整模型微调相当的准确率，同时实现更快更稳健的收敛。

### 结论

FRoD成功解决了现有PEFT方法在效率与表达能力之间的权衡问题，在保持高效率的同时实现了与完整模型微调相当的性能。

### 翻译

参数高效微调(PEFT)方法已成为适应大型基础模型到下游任务的实用解决方案，通过只更新一小部分参数来减少计算和内存成本。其中，像LoRA这样的方法试图在效率和表达能力之间取得平衡，但往往因固有的低秩约束而面临收敛速度慢和适应能力有限的问题。这种权衡阻碍了PEFT方法捕捉多样化任务所需复杂模式的能力。为解决这些挑战，我们提出了FRoD，一种结合分层联合分解与旋转自由度的微调新方法。通过提取跨层的全局共享基并向缩放因子中注入稀疏可学习扰动以实现灵活的全秩更新，FRoD提高了表达能力和效率，实现了更快、更稳健的收敛。在涵盖视觉、推理和语言理解的20个基准测试中，FRoD在相同训练预算下仅使用1.72%的可训练参数，就能达到与完整模型微调相当的准确率。


### 论文摘要

Parameter-efficient fine-tuning (PEFT) methods have emerged as a practical solution for adapting large foundation models to downstream tasks, reducing computational and memory costs by updating only a small subset of parameters. Among them, approaches like LoRA aim to strike a balance between efficiency and expressiveness, but often suffer from slow convergence and limited adaptation capacity due to their inherent low-rank constraints. This trade-off hampers the ability of PEFT methods to capture complex patterns needed for diverse tasks. To address these challenges, we propose FRoD, a novel fine-tuning method that combines hierarchical joint decomposition with rotational degrees of freedom. By extracting a globally shared basis across layers and injecting sparse, learnable perturbations into scaling factors for flexible full-rank updates, FRoD enhances expressiveness and efficiency, leading to faster and more robust convergence. On 20 benchmarks spanning vision, reasoning, and language understanding, FRoD matches full model fine-tuning in accuracy, while using only 1.72% of trainable parameters under identical training budgets.

---

## 54. Towards Integrating Uncertainty for Domain-Agnostic Segmentation

**论文链接:** [http://arxiv.org/abs/2512.23427v1](http://arxiv.org/abs/2512.23427v1)

**作者:** Jesse Brouwers, Xiaoyan Xing, Alexander Timans

**发布时间:** 2025-12-29

**备注:** Public code at https://github.com/JesseBrouw/UncertSAM | published at the 2nd Workshop on Frontiers in Probabilistic Inference (NeurIPS 2025) | 12 pages, 8 figures (incl. Appendix)

### GPT解析

### 总结

本研究探讨了不确定性量化如何缓解基础分割模型在变化或有限知识领域中的脆弱性，并增强其泛化能力。

### 背景

基础分割模型如Segment Anything Model (SAM)系列表现出强大的零样本性能，但在变化或知识有限的领域中仍然脆弱。

### 目的

研究不确定性量化是否可以缓解这些挑战，以领域无关的方式增强模型泛化能力。

### 方法

1) 整理了UncertSAM基准，包含八个数据集，旨在在具有挑战性的分割条件下测试SAM，包括阴影、透明度和伪装；2) 评估了一套轻量级的后验不确定性估计方法；3) 评估了一个初步的不确定性引导的预测细化步骤。

### 主要发现

在评估的方法中，最后一层拉普拉斯近似产生的不确定性估计与分割错误有很好的相关性，表明有有意义的信号。虽然细化的好处是初步的，但研究发现强调将不确定性纳入分割模型的潜力。

### 结论

不确定性量化可以帮助缓解基础分割模型在变化或有限知识领域中的脆弱性，支持稳健的、领域无关的性能。研究团队公开了UncertSAM基准和相关代码。

### 翻译

用于分割的基础模型如Segment Anything Model (SAM)系列表现出强大的零样本性能，但在变化或知识有限的领域中仍然脆弱。本研究探讨了不确定性量化是否可以缓解这些挑战，并以领域无关的方式增强模型泛化能力。为此，我们(1)整理了UncertSAM基准，包含八个数据集，旨在在具有挑战性的分割条件下测试SAM，包括阴影、透明度和伪装；(2)评估了一套轻量级的后验不确定性估计方法；以及(3)评估了一个初步的不确定性引导的预测细化步骤。在评估的方法中，最后一层拉普拉斯近似产生的不确定性估计与分割错误有很好的相关性，表明有有意义的信号。虽然细化的好处是初步的，但我们的发现强调将不确定性纳入分割模型的潜力，以支持稳健的、领域无关的性能。我们的基准和代码已公开提供。


### 论文摘要

Foundation models for segmentation such as the Segment Anything Model (SAM) family exhibit strong zero-shot performance, but remain vulnerable in shifted or limited-knowledge domains. This work investigates whether uncertainty quantification can mitigate such challenges and enhance model generalisability in a domain-agnostic manner. To this end, we (1) curate UncertSAM, a benchmark comprising eight datasets designed to stress-test SAM under challenging segmentation conditions including shadows, transparency, and camouflage; (2) evaluate a suite of lightweight, post-hoc uncertainty estimation methods; and (3) assess a preliminary uncertainty-guided prediction refinement step. Among evaluated approaches, a last-layer Laplace approximation yields uncertainty estimates that correlate well with segmentation errors, indicating a meaningful signal. While refinement benefits are preliminary, our findings underscore the potential of incorporating uncertainty into segmentation models to support robust, domain-agnostic performance. Our benchmark and code are made publicly available.

---

## 55. 论文ID: 2512.23411v1

**错误:** 无法读取论文信息 - [Errno 2] No such file or directory: 'storage/result/2512.23411v1.json'

---

## 56. Agentic AI-Enhanced Semantic Communications: Foundations, Architecture, and Applications

**论文链接:** [http://arxiv.org/abs/2512.23294v1](http://arxiv.org/abs/2512.23294v1)

**作者:** Haixiao Gao, Mengying Sun, Ruichen Zhang, Yanhan Wang, Xiaodong Xu, Nan Ma, Dusit Niyato, Ping Zhang

**发布时间:** 2025-12-29

### GPT解析

### 总结

这篇论文系统阐述了智能体AI如何赋能语义通信(SemCom)，从研究基础、系统架构和应用场景三个角度进行全面分析，并提出统一框架和典型案例研究。

### 背景

语义通信作为6G的关键技术之一，正推动网络从比特传输向语义信息交换转变。智能体AI具有感知、记忆、推理和行动能力，为智能通信提供了可行路径。

### 目的

系统阐述智能体AI赋能语义通信的方式，包括研究基础、系统架构和应用场景，提出统一框架，并讨论未来研究方向。

### 方法

按智能体类型综述现有研究；提出涵盖应用层、语义层和云边缘协作层的统一框架；展示多车协同感知等典型场景；介绍基于智能体知识库的联合信源信道编码案例研究；讨论未来演进方向。

### 主要发现

提出的AKB-JSCC在不同信道条件下实现更高信息重建质量；智能体AI为语义通信提供了从意图到评估的闭环系统。

### 结论

为智能体语义通信的可移植性、可验证性和可控性研究与部署提供了参考。

### 翻译

语义通信(SemCom)作为6G的关键技术之一，正在推动网络从比特传输向语义信息交换转变。在此基础上，引入具有感知、记忆、推理和行动能力的智能体AI为智能通信提供了可行的路径。本文从研究基础、系统架构和应用场景三个角度系统阐述了智能体AI如何赋能语义通信。我们首先按智能体类型对现有研究进行了全面综述，包括嵌入式智能体、大语言模型/大视觉模型智能体和强化学习智能体。此外，我们提出了一个统一的智能体AI增强的语义通信框架，涵盖应用层、语义层和云边缘协作层，形成从意图到编码、传输、解码、行动到评估的闭环。我们还展示了几个典型场景，包括多车协同感知、多机器人协作救援和智能网络的智能体操作。此外，我们介绍了一个基于智能体知识库的联合信源信道编码案例研究AKB-JSCC，其中源知识库和信道知识库分别由大语言模型/大视觉模型智能体和强化学习智能体构建。实验结果表明，AKB-JSCC在不同信道条件下能够实现更高的信息重建质量。最后，我们讨论了未来演进和研究方向，为智能体语义通信的可移植性、可验证性和可控性研究与部署提供了参考。


### 论文摘要

Semantic communications (SemCom), as one of the key technologies for 6G, is shifting networks from bit transmission to semantic information exchange. On this basis, introducing agentic artificial intelligence (AI) with perception, memory, reasoning, and action capabilities provides a practicable path to intelligent communications. This paper provides a systematic exposition of how agentic AI empowers SemCom from the perspectives of research foundations, system architecture, and application scenarios. We first provide a comprehensive review of existing studies by agent types, covering embedded agents, large language model (LLM)/large vision model (LVM) agents, and reinforcement learning (RL) agents. Additionally, we propose a unified agentic AI-enhanced SemCom framework covering the application layer, the semantic layer, and the cloud-edge collaboration layer, forming a closed loop from intent to encoding to transmission to decoding to action to evaluation. We also present several typical scenarios, including multi-vehicle collaborative perception, multi-robot cooperative rescue, and agentic operations for intellicise (intelligent and concise) networks. Furthermore, we introduce an agentic knowledge base (KB)-based joint source-channel coding case study, AKB-JSCC, where the source KB and channel KB are built by LLM/LVM agents and RL agents, respectively. Experimental results show that AKB-JSCC achieves higher information reconstruction quality under different channel conditions. Finally, we discuss future evolution and research directions, providing a reference for portable, verifiable, and controllable research and deployment of agentic SemCom.

---

## 57. Agentic Physical AI toward a Domain-Specific Foundation Model for Nuclear Reactor Control

**论文链接:** [http://arxiv.org/abs/2512.23292v1](http://arxiv.org/abs/2512.23292v1)

**作者:** Yoonpyo Lee, Kazuma Kobayashi, Sai Puppala, Sajedul Talukder, Seid Koric, Souvik Chakraborty, Syed Bahauddin Alam

**发布时间:** 2025-12-29

### GPT解析

### 总结

研究提出了一种基于物理验证而非感知推理的智能体物理AI模型，在合成反应堆控制场景中表现出优于通用基础模型的性能。

### 背景

当前AI在物理系统中的主流范式是将通用基础模型扩展到多模态推理，但这种方法在控制界面面临根本障碍，即使最先进的视觉语言模型在基础物理任务上的准确率仅50-53%。

### 目的

提出一种通向领域特定基础模型的根本不同路径，通过紧凑语言模型作为智能体物理AI，实现基于物理验证的策略优化。

### 方法

在合成反应堆控制场景中训练一个3.6亿参数的模型，数据集从10^3扩展到10^5个例子，使用物理验证驱动策略优化而非感知推理。

### 主要发现

模型实现了通用模型中不存在的尖锐相变；小规模系统表现高方差模仿和灾难性尾部风险；大规模模型经历超过500倍的方差减少，稳定执行级别行为；模型自主拒绝约70%的训练分布，95%的运行时执行集中在单一策略上；学习表示可在不同物理和连续输入模态间转移。

### 结论

物理验证驱动的AI模型在物理系统控制中表现更优，能够实现更稳定和可靠的行为，且具有跨模态迁移能力。

### 翻译

当前AI物理系统的主流范式是将通用基础模型扩展到通用多模态推理，但在控制界面面临根本性障碍。最近的基准测试显示，即使是最先进的视觉语言模型在基础物理任务上的准确率也只有50-53%，表现为近似猜测者，保持语义合理性但违反物理约束。这种输入不忠实不是扩展缺陷，而是结构限制。以感知为中心的架构优化参数空间模仿，而安全关键控制需要执行动作的结果空间保证。在此，我们提出了一种通向领域特定基础模型的根本不同路径，引入了作为智能体物理AI运行的紧凑语言模型，其中策略优化由基于物理的验证而非感知推理驱动。我们在合成反应堆控制场景中训练了一个3.6亿参数的模型，将数据集从10^3扩展到10^5个例子。这导致了通用模型中不存在的尖锐相变。小规模系统表现出高方差模仿和灾难性尾部风险，而大规模模型经历超过500倍的方差减少，稳定了执行级别行为。尽管接触了四种执行家族，但模型自主拒绝了约70%的训练分布，并将95%的运行时执行集中在单一银行策略上。学习到的表示可以在不同物理和连续输入模态间转移，无需架构修改。


### 论文摘要

The prevailing paradigm in AI for physical systems, scaling general-purpose foundation models toward universal multimodal reasoning, confronts a fundamental barrier at the control interface. Recent benchmarks show that even frontier vision-language models achieve only 50-53% accuracy on basic quantitative physics tasks, behaving as approximate guessers that preserve semantic plausibility while violating physical constraints. This input unfaithfulness is not a scaling deficiency but a structural limitation. Perception-centric architectures optimize parameter-space imitation, whereas safety-critical control demands outcome-space guarantees over executed actions. Here, we present a fundamentally different pathway toward domain-specific foundation models by introducing compact language models operating as Agentic Physical AI, in which policy optimization is driven by physics-based validation rather than perceptual inference. We train a 360-million-parameter model on synthetic reactor control scenarios, scaling the dataset from 10^3 to 10^5 examples. This induces a sharp phase transition absent in general-purpose models. Small-scale systems exhibit high-variance imitation with catastrophic tail risk, while large-scale models undergo variance collapse exceeding 500x reduction, stabilizing execution-level behavior. Despite balanced exposure to four actuation families, the model autonomously rejects approximately 70% of the training distribution and concentrates 95% of runtime execution on a single-bank strategy. Learned representations transfer across distinct physics and continuous input modalities without architectural modification.

---

## 58. 论文ID: 2512.23239v1

**错误:** 无法读取论文信息 - [Errno 2] No such file or directory: 'storage/result/2512.23239v1.json'

---

## 59. 论文ID: 2512.23189v1

**错误:** 无法读取论文信息 - [Errno 2] No such file or directory: 'storage/result/2512.23189v1.json'

---

## 60. 论文ID: 2512.23132v1

**错误:** 无法读取论文信息 - [Errno 2] No such file or directory: 'storage/result/2512.23132v1.json'

---

## 61. MedSAM-based lung masking for multi-label chest X-ray classification

**论文链接:** [http://arxiv.org/abs/2512.23089v1](http://arxiv.org/abs/2512.23089v1)

**作者:** Brayden Miao, Zain Rehman, Xin Miao, Siming Liu, Jianjie Wang

**发布时间:** 2025-12-28

**备注:** 16 pages, 8 figures

### GPT解析

### 总结

本研究提出了一种分割引导的胸部X光分类流程，整合MedSAM作为肺区域提取模块，用于多标签异常分类。实验表明肺掩模效果既依赖于任务也依赖于架构，应根据临床目标选择合适的空间先验。

### 背景

胸部X光成像广泛用于筛查和诊断肺部异常，但自动化解释面临疾病信号弱、数据集偏差和有限空间监督等挑战。

### 目的

探索基础模型MedSAM在医学图像分割中的应用，引入基于解剖学的先验知识，提高CXR分析的鲁棒性和可解释性。

### 方法

提出分割引导的CXR分类流程，集成MedSAM作为肺区域提取模块；使用Airlangga大学医院的数据集对MedSAM进行微调；应用在NIH CXR数据集上训练和评估深度卷积神经网络，进行五种异常的多标签预测。

### 主要发现

MedSAM能在各种成像条件下生成解剖学合理的肺掩模；在原始图像上训练的ResNet50实现最强异常区分；宽松肺掩模显著改善正常情况识别；紧密掩模提高训练效率但降低异常级别性能；宽松掩模通过保留肺门和周围上下文部分减轻退化。

### 结论

肺掩模应被视为可控的空间先验，根据骨干网络和临床目标进行选择，而非统一应用。

### 翻译

胸部X光(CXR)成像广泛用于筛查和诊断肺部异常，但由于疾病信号弱、数据集偏差和有限的空间监督，自动化解释仍然具有挑战性。医学图像分割的基础模型(MedSAM)提供了引入基于解剖学先验知识的机会，可能提高CXR分析的鲁棒性和可解释性。我们提出了一种分割引导的CXR分类流程，将MedSAM集成作为多标签异常分类前的肺区域提取模块。使用Airlangga大学医院的公共图像-掩模数据集对MedSAM进行微调。然后将其应用于公共NIH CXR数据集的精选子集，训练和评估深度卷积神经网络，用于五种异常(Mass, Nodule, Pneumonia, Edema和Fibrosis)的多标签预测，并通过派生分数评估正常情况(No Finding)。实验表明，MedSAM能在各种成像条件下生成解剖学合理的肺掩模。我们发现掩模效应既依赖于任务也依赖于架构。在原始图像上训练的ResNet50实现了最强的整体异常区分能力，而宽松的肺掩模提供了相当的宏观AUROC，但显著改善了No Finding的区分，表明在异常特定分类和正常情况筛查之间存在权衡。紧密掩模持续降低异常级别性能但提高训练效率。宽松掩模通过保留肺门和周围上下文部分减轻了这种退化。这些结果表明，肺掩模应被视为一种可控的空间先验，根据骨干网络和临床目标进行选择，而非统一应用。


### 论文摘要

Chest X-ray (CXR) imaging is widely used for screening and diagnosing pulmonary abnormalities, yet automated interpretation remains challenging due to weak disease signals, dataset bias, and limited spatial supervision. Foundation models for medical image segmentation (MedSAM) provide an opportunity to introduce anatomically grounded priors that may improve robustness and interpretability in CXR analysis. We propose a segmentation-guided CXR classification pipeline that integrates MedSAM as a lung region extraction module prior to multi-label abnormality classification. MedSAM is fine-tuned using a public image-mask dataset from Airlangga University Hospital. We then apply it to a curated subset of the public NIH CXR dataset to train and evaluate deep convolutional neural networks for multi-label prediction of five abnormalities (Mass, Nodule, Pneumonia, Edema, and Fibrosis), with the normal case (No Finding) evaluated via a derived score. Experiments show that MedSAM produces anatomically plausible lung masks across diverse imaging conditions. We find that masking effects are both task-dependent and architecture-dependent. ResNet50 trained on original images achieves the strongest overall abnormality discrimination, while loose lung masking yields comparable macro AUROC but significantly improves No Finding discrimination, indicating a trade-off between abnormality-specific classification and normal case screening. Tight masking consistently reduces abnormality level performance but improves training efficiency. Loose masking partially mitigates this degradation by preserving perihilar and peripheral context. These results suggest that lung masking should be treated as a controllable spatial prior selected to match the backbone and clinical objective, rather than applied uniformly.

---

## 62. TabiBERT: A Large-Scale ModernBERT Foundation Model and Unified Benchmarking Framework for Turkish

**论文链接:** [http://arxiv.org/abs/2512.23065v1](http://arxiv.org/abs/2512.23065v1)

**作者:** Melikşah Türker, A. Ebrar Kızıloğlu, Onur Güngör, Susan Üsküdarlı

**发布时间:** 2025-12-28

**备注:** 31 pages, 1 figure, 13 tables

### GPT解析

### 总结

TabiBERT是一种基于ModernBERT架构的单语土耳其语编码器，从零开始在大规模多领域语料库上训练，在计算效率、训练稳定性和长上下文建模方面表现出色。

### 背景

自BERT出现以来，仅编码器Transformer模型在计算效率、训练稳定性和长上下文建模方面已显著发展。现代BERT整合了旋转位置编码、FlashAttention和改进的归一化方法，但土耳其NLP领域缺乏采用这些现代架构范式的单语编码器。

### 目的

引入TabiBERT，一种基于ModernBERT架构的单语土耳其语编码器，从零开始在大规模精选语料库上训练，以填补土耳其NLP领域的空白。

### 方法

在包含848.8亿个标记的多领域语料库上预训练TabiBERT，语料库包括网络文本(73%)、科学出版物(20%)、源代码(6%)和数学内容(0.3%)，采样1万亿个标记。模型支持8,192个标记的上下文长度，使用FlashAttention实现2.65倍推理加速，减少GPU内存消耗。引入TabiBench评估框架，包含8个任务类别下的28个数据集，使用GLUE风格宏平均评估。

### 主要发现

TabiBERT在TabiBench上达到77.58分，比BERTurk高出1.62分，在问答(+9.55)、代码检索(+2.41)和文档检索(+0.60)等五个类别建立最先进结果。与专业模型相比，实现+1.47的平均改进，表现出强大的跨领域泛化能力。

### 结论

TabiBERT作为土耳其语NLP领域的新基准模型，展示了现代架构在特定语言处理中的有效性，研究团队公开了模型权重、训练配置和评估代码以促进土耳其语编码器研究的透明性和可复现性。

### 翻译

自BERT问世以来，仅编码器Transformer模型在计算效率、训练稳定性和长上下文建模方面已取得显著进展。ModernBERT通过整合旋转位置编码、FlashAttention和改进的归一化方法巩固了这些进展。尽管有这些发展，土耳其NLP领域仍缺乏一种从零开始训练并融入此类现代架构范式的单语编码器。本研究介绍了TabiBERT，这是一种基于ModernBERT架构的单语土耳其编码器，在大规模精选语料库上从头开始训练。TabiBERT在包含848.8亿个标记的多领域语料库中采样的1万亿个标记上进行预训练：网络文本(73%)、科学出版物(20%)、源代码(6%)和数学内容(0.3%)。该模型支持8,192个标记的上下文长度(原始BERT的16倍)，实现高达2.65倍的推理加速，并减少GPU内存消耗，使更大的批处理大小成为可能。我们引入了TabiBench，包含八个任务类别中的28个数据集，使用标准化的分割和协议，采用GLUE风格的宏平均进行评估。TabiBERT在TabiBench上达到77.58分，比BERTurk高出1.62分，并在八个类别中的五个类别上建立最先进水平：问答(+9.55)、代码检索(+2.41)和文档检索(+0.60)。与包括TurkishBERTweet等专用模型在内的任务特定先前最佳结果相比，TabiBERT实现了+1.47的平均改进，表明其具有强大的跨领域泛化能力。我们发布了模型权重、训练配置和评估代码，以促进透明且可复现的土耳其语编码器研究。


### 论文摘要

Since the inception of BERT, encoder-only Transformers have evolved significantly in computational efficiency, training stability, and long-context modeling. ModernBERT consolidates these advances by integrating Rotary Positional Embeddings (RoPE), FlashAttention, and refined normalization. Despite these developments, Turkish NLP lacks a monolingual encoder trained from scratch incorporating such modern architectural paradigms. This work introduces TabiBERT, a monolingual Turkish encoder based on ModernBERT architecture trained from scratch on a large, curated corpus. TabiBERT is pre-trained on one trillion tokens sampled from an 84.88B token multi-domain corpus: web text (73%), scientific publications (20%), source code (6%), and mathematical content (0.3%). The model supports 8,192-token context length (16x original BERT), achieves up to 2.65x inference speedup, and reduces GPU memory consumption, enabling larger batch sizes. We introduce TabiBench with 28 datasets across eight task categories with standardized splits and protocols, evaluated using GLUE-style macro-averaging. TabiBERT attains 77.58 on TabiBench, outperforming BERTurk by 1.62 points and establishing state-of-the-art on five of eight categories: question answering (+9.55), code retrieval (+2.41), and document retrieval (+0.60). Compared with task-specific prior best results, including specialized models like TurkishBERTweet, TabiBERT achieves +1.47 average improvement, indicating robust cross-domain generalization. We release model weights, training configurations, and evaluation code for transparent, reproducible Turkish encoder research.

---

## 63. 论文ID: 2512.23056v1

**错误:** 无法读取论文信息 - [Errno 2] No such file or directory: 'storage/result/2512.23056v1.json'

---

## 64. Toward Stable Semi-Supervised Remote Sensing Segmentation via Co-Guidance and Co-Fusion

**论文链接:** [http://arxiv.org/abs/2512.23035v1](http://arxiv.org/abs/2512.23035v1)

**作者:** Yi Zhou, Xuechao Zou, Shun Zhang, Kai Li, Shiying Wang, Jingming Chen, Congyan Lang, Tengfei Cao, Pin Tao, Yuanchun Shi

**发布时间:** 2025-12-28

**备注:** 13 pages, 5 figures, 10 tables

### GPT解析

### 总结

本文提出了一种名为Co2S的稳定半监督遥感图像语义分割框架，通过协同融合视觉语言模型和自监督模型的先验知识来解决伪标签漂移问题。

### 背景

半监督遥感图像语义分割虽能减轻大量标注负担，但存在伪标签漂移现象，即确认偏差导致训练过程中错误累积的问题。

### 目的

开发一个稳定的半监督遥感图像分割框架，有效缓解伪标签漂移现象，提高分割精度。

### 方法

构建异构双学生架构，包含两个基于ViT的视觉基础模型(分别用CLIP和DINOv3初始化)；引入显式-隐式语义共引导机制，利用文本嵌入和可学习查询提供类别级引导；开发全局-局部特征协同融合策略，融合CLIP的全局上下文和DINOv3的局部细节。

### 主要发现

在六个流行数据集上的实验表明，所提出的方法在各种分区协议和不同场景下均取得了领先的性能表现。

### 结论

Co2S框架通过协同融合不同视觉模型的先验知识，有效解决了半监督遥感图像分割中的伪标签漂移问题，显著提高了分割精度。

### 翻译

半监督遥感(RS)图像语义分割提供了一种有前景的解决方案，可以减轻大量标注的负担，但它从根本上受到伪标签漂移的困扰，这是一种确认偏差导致训练过程中错误累积的现象。在这项工作中，我们提出了Co2S，一个稳定的半监督RS分割框架，协同融合了视觉语言模型和自监督模型的先验知识。具体来说，我们构建了一个异构双学生架构，包含两个不同的基于ViT的视觉基础模型，分别用预训练的CLIP和DINOv3初始化，以减少错误累积和伪标签漂移。为了有效融合这些不同的先验知识，我们引入了一种显式-隐式语义共引导机制，分别利用文本嵌入和可学习查询提供显式和隐式的类别级引导，从而共同增强语义一致性。此外，还开发了一种全局-局部特征协同融合策略，有效融合CLIP捕获的全局上下文信息和DINOv3产生的局部细节，使模型能够生成高精度的分割结果。在六个流行数据集上的大量实验证明了所提出方法的优越性，在各种分区协议和不同场景下均持续取得领先性能。项目页面可在 https://xavierjiezou.github.io/Co2S/ 获取。


### 论文摘要

Semi-supervised remote sensing (RS) image semantic segmentation offers a promising solution to alleviate the burden of exhaustive annotation, yet it fundamentally struggles with pseudo-label drift, a phenomenon where confirmation bias leads to the accumulation of errors during training. In this work, we propose Co2S, a stable semi-supervised RS segmentation framework that synergistically fuses priors from vision-language models and self-supervised models. Specifically, we construct a heterogeneous dual-student architecture comprising two distinct ViT-based vision foundation models initialized with pretrained CLIP and DINOv3 to mitigate error accumulation and pseudo-label drift. To effectively incorporate these distinct priors, an explicit-implicit semantic co-guidance mechanism is introduced that utilizes text embeddings and learnable queries to provide explicit and implicit class-level guidance, respectively, thereby jointly enhancing semantic consistency. Furthermore, a global-local feature collaborative fusion strategy is developed to effectively fuse the global contextual information captured by CLIP with the local details produced by DINOv3, enabling the model to generate highly precise segmentation results. Extensive experiments on six popular datasets demonstrate the superiority of the proposed method, which consistently achieves leading performance across various partition protocols and diverse scenarios. Project page is available at https://xavierjiezou.github.io/Co2S/.

---

## 65. 论文ID: 2512.22931v1

**错误:** 无法读取论文信息 - [Errno 2] No such file or directory: 'storage/result/2512.22931v1.json'

---

## 66. HiSciBench: A Hierarchical Multi-disciplinary Benchmark for Scientific Intelligence from Reading to Discovery

**论文链接:** [http://arxiv.org/abs/2512.22899v1](http://arxiv.org/abs/2512.22899v1)

**作者:** Yaping Zhang, Qixuan Zhang, Xingquan Zhang, Zhiyuan Chen, Wenwen Zhuang, Yupu Liang, Lu Xiang, Yang Zhao, Jiajun Zhang, Yu Zhou, Chengqing Zong

**发布时间:** 2025-12-28

### GPT解析

### 总结

研究团队引入了HiSciBench，一个分层的科学智能评估基准，用于全面评估基础模型在科学工作流不同层次的能力。

### 背景

大型语言模型和多模态基础模型的快速发展引发了其在科学研究中应用的兴趣，但科学智能涵盖广泛的能力，现有基准测试仍然碎片化，大多只关注狭窄任务，未能反映科学探究的层次性和多学科性质。

### 目的

创建一个分层基准测试，能够全面评估基础模型在科学工作流不同层次的能力，从基础理解到创造性发现。

### 方法

HiSciBench包含5个层次对应完整科学工作流：科学素养(L1)、文献解析(L2)、基于文献的问答(L3)、文献综述生成(L4)和科学发现(L5)。基准包含8,735个精心策划的实例，涵盖六个主要科学学科，支持多模态输入和跨语言评估，并提供依赖感知框架进行详细诊断。

### 主要发现

对领先模型的评估显示显著性能差距：模型在基础素养任务上准确率可达69%，但在发现级挑战中性能急剧下降至25%。

### 结论

HiSciBench为科学智能评估建立了新标准，并为开发更强大、更可靠的模型提供了可操作的见解。该基准将公开发布以促进未来研究。

### 翻译

大型语言模型和多模态基础模型的快速发展引发了其在科学研究中应用的潜力。然而，科学智能涵盖从理解基础知识到进行创造性发现的广泛能力范围，现有基准测试仍然碎片化。大多数基准测试只关注狭窄任务，未能反映科学探究的层次性和多学科性质。我们引入了HiSciBench，一个分层基准测试，用于评估基础模型在五个层次的能力，这些层次对应完整的科学工作流：科学素养、文献解析、基于文献的问答、文献综述生成和科学发现。HiSciBench包含8,735个精心策划的实例，涵盖六个主要科学学科，包括数学、物理、化学、生物学、地理学和天文学，并支持文本、方程、图形和表格等多模态输入，以及跨语言评估。与评估孤立能力的先前基准不同，HiSciBench提供了一个集成、依赖感知的框架，能够详细诊断模型在科学推理不同阶段的能力。对领先模型的全面评估，包括GPT-5、DeepSeek-R1和几个多模态系统，揭示了显著的性能差距：虽然模型在基础素养任务上可实现高达69%的准确率，但在发现级挑战中性能急剧下降至25%。HiSciBench为科学智能评估建立了新标准，并为开发不仅更强大而且更可靠的模型提供了可操作的见解。该基准将公开发布以促进未来研究。


### 论文摘要

The rapid advancement of large language models (LLMs) and multimodal foundation models has sparked growing interest in their potential for scientific research. However, scientific intelligence encompasses a broad spectrum of abilities ranging from understanding fundamental knowledge to conducting creative discovery, and existing benchmarks remain fragmented. Most focus on narrow tasks and fail to reflect the hierarchical and multi-disciplinary nature of real scientific inquiry. We introduce \textbf{HiSciBench}, a hierarchical benchmark designed to evaluate foundation models across five levels that mirror the complete scientific workflow: \textit{Scientific Literacy} (L1), \textit{Literature Parsing} (L2), \textit{Literature-based Question Answering} (L3), \textit{Literature Review Generation} (L4), and \textit{Scientific Discovery} (L5). HiSciBench contains 8,735 carefully curated instances spanning six major scientific disciplines, including mathematics, physics, chemistry, biology, geography, and astronomy, and supports multimodal inputs including text, equations, figures, and tables, as well as cross-lingual evaluation. Unlike prior benchmarks that assess isolated abilities, HiSciBench provides an integrated, dependency-aware framework that enables detailed diagnosis of model capabilities across different stages of scientific reasoning. Comprehensive evaluations of leading models, including GPT-5, DeepSeek-R1, and several multimodal systems, reveal substantial performance gaps: while models achieve up to 69\% accuracy on basic literacy tasks, performance declines sharply to 25\% on discovery-level challenges. HiSciBench establishes a new standard for evaluating scientific Intelligence and offers actionable insights for developing models that are not only more capable but also more reliable. The benchmark will be publicly released to facilitate future research.

---

## 67. Agentic AI for Cyber Resilience: A New Security Paradigm and Its System-Theoretic Foundations

**论文链接:** [http://arxiv.org/abs/2512.22883v1](http://arxiv.org/abs/2512.22883v1)

**作者:** Tao Li, Quanyan Zhu

**发布时间:** 2025-12-28

### GPT解析

### 总结

基于基础模型的人工智能正在从根本上重塑网络安全领域，大语言模型实现了自主规划、工具编排和大规模战略适应，挑战了传统静态安全架构，促使网络安全从以预防为中心转向代理网络弹性。

### 背景

传统安全架构建立在静态规则、边界防御和以人为中心的工作流程基础上，而大语言模型等基础模型AI技术正在改变这一格局，使系统能够实现大规模自主决策和适应。

### 目的

推动网络安全范式从以预防为中心转向代理网络弹性，构建能够预期中断、在攻击下维持关键功能、高效恢复并持续学习的弹性系统。

### 方法

通过历史演变分析定位这一转变，开发系统级框架设计代理AI工作流，引入通用代理架构，将攻击者和防御者工作流分析为耦合的自适应过程，并使用博弈论公式作为统一设计语言。

### 主要发现

博弈论公式为自主分配、信息流和时间组成提供了统一的设计语言，自动化渗透测试、修复和网络欺骗的案例研究表明基于均衡的设计可实现系统级弹性。

### 结论

网络安全已进入AI增强的新范式，其中自主代理直接参与网络和网络物理系统的感知、推理、行动和适应过程。

### 翻译

网络安全正基于基础模型的人工智能被根本性地重塑。大语言模型现在能够实现大规模的自主规划、工具编排和战略适应，挑战了建立在静态规则、边界防御和以人为中心的工作流程上的安全架构。本章主张从以预防为中心的安全转向代理网络弹性。弹性系统必须预期中断、在攻击下维持关键功能、高效恢复并持续学习，而非寻求完美保护。我们将这种转变置于网络安全范式的历史演变中，最终形成一个AI增强的范式，其中自主代理直接参与网络和网络物理系统的感知、推理、行动和适应。然后，我们开发了一个用于设计代理AI工作流的系统级框架。介绍了通用代理架构，并将攻击者和防御者工作流分析为耦合的自适应过程，博弈论公式被证明可以为自主分配、信息流和时间组成提供统一的设计语言。自动化渗透测试、修复和网络欺骗的案例研究说明了基于均衡的设计如何实现系统级弹性设计。


### 论文摘要

Cybersecurity is being fundamentally reshaped by foundation-model-based artificial intelligence. Large language models now enable autonomous planning, tool orchestration, and strategic adaptation at scale, challenging security architectures built on static rules, perimeter defenses, and human-centered workflows. This chapter argues for a shift from prevention-centric security toward agentic cyber resilience. Rather than seeking perfect protection, resilient systems must anticipate disruption, maintain critical functions under attack, recover efficiently, and learn continuously. We situate this shift within the historical evolution of cybersecurity paradigms, culminating in an AI-augmented paradigm where autonomous agents participate directly in sensing, reasoning, action, and adaptation across cyber and cyber-physical systems. We then develop a system-level framework for designing agentic AI workflows. A general agentic architecture is introduced, and attacker and defender workflows are analyzed as coupled adaptive processes, and game-theoretic formulations are shown to provide a unifying design language for autonomy allocation, information flow, and temporal composition. Case studies in automated penetration testing, remediation, and cyber deception illustrate how equilibrium-based design enables system-level resiliency design.

---

## 68. GHaLIB: A Multilingual Framework for Hope Speech Detection in Low-Resource Languages

**论文链接:** [http://arxiv.org/abs/2512.22705v1](http://arxiv.org/abs/2512.22705v1)

**作者:** Ahmed Abdullah, Sana Fatima, Haroon Mahmood

**发布时间:** 2025-12-27

**备注:** Accepted and presented at the 15th International Arab Conference on Information Technology (ICAIT); proceedings not yet published

### GPT解析

### 总结

本文提出了一个多语言希望言语检测框架，特别关注乌尔都语，使用预训练transformer模型进行分类，并在多语言基准测试中取得了良好性能。

### 背景

希望言语在自然语言处理中相对未被充分研究，现有研究主要集中在英语上，导致低资源语言如乌尔都语缺乏相关资源。虽然基于transformer的架构在检测仇恨和冒犯性言语方面已被证明有效，但很少将其应用于希望言语检测或在不同语言环境中进行测试。

### 目的

开发一个多语言希望言语检测框架，特别关注乌尔都语，以促进积极在线沟通并构建更有建设性的数字对话。

### 方法

使用预训练的transformer模型(如XLM-RoBERTa、mBERT、EuroBERT和UrduBERT)，应用简单的预处理技术，并训练分类器以获得改进的结果。

### 主要发现

在PolyHope-M 2025基准测试上评估显示，乌尔都语二元分类的F1分数达到95.2%，乌尔都语多类分类的F1分数达到65.2%，在西班牙语、德语和英语中也取得了具有竞争力的相似结果。

### 结论

这些结果表明，在低资源环境中实施现有的多语言模型是可行的，这使得识别希望言语变得更加容易，有助于构建更有建设性的数字对话。

### 翻译

希望言语在自然语言处理(NLP)中相对未被充分研究。现有研究主要集中在英语上，这导致低资源语言(如乌尔都语)缺乏相关资源。因此，促进积极在线沟通的工具开发仍然有限。虽然基于transformer的架构已被证明在检测仇恨和冒犯性言语方面有效，但在将其应用于希望言语或更广泛地在各种语言环境中测试方面所做的努力很少。本文提出了一个多语言希望言语检测框架，特别关注乌尔都语。使用XLM-RoBERTa、mBERT、EuroBERT和UrduBERT等预训练transformer模型，我们应用简单的预处理并训练分类器以获得改进的结果。在PolyHope-M 2025基准测试上的评估表明性能强劲，乌尔都语二元分类的F1分数达到95.2%，乌尔都语多类分类的F1分数达到65.2%，在西班牙语、德语和英语中也取得了同样具有竞争力的结果。这些结果突显了在低资源环境中实施现有多语言模型的可能性，从而更容易识别希望言语，并帮助构建更有建设性的数字对话。


### 论文摘要

Hope speech has been relatively underrepresented in Natural Language Processing (NLP). Current studies are largely focused on English, which has resulted in a lack of resources for low-resource languages such as Urdu. As a result, the creation of tools that facilitate positive online communication remains limited. Although transformer-based architectures have proven to be effective in detecting hate and offensive speech, little has been done to apply them to hope speech or, more generally, to test them across a variety of linguistic settings. This paper presents a multilingual framework for hope speech detection with a focus on Urdu. Using pretrained transformer models such as XLM-RoBERTa, mBERT, EuroBERT, and UrduBERT, we apply simple preprocessing and train classifiers for improved results. Evaluations on the PolyHope-M 2025 benchmark demonstrate strong performance, achieving F1-scores of 95.2% for Urdu binary classification and 65.2% for Urdu multi-class classification, with similarly competitive results in Spanish, German, and English. These results highlight the possibility of implementing existing multilingual models in low-resource environments, thus making it easier to identify hope speech and helping to build a more constructive digital discourse.

---

## 69. INTERACT-CMIL: Multi-Task Shared Learning and Inter-Task Consistency for Conjunctival Melanocytic Intraepithelial Lesion Grading

**论文链接:** [http://arxiv.org/abs/2512.22666v1](http://arxiv.org/abs/2512.22666v1)

**作者:** Mert Ikinci, Luna Toma, Karin U. Loeffler, Leticia Ussem, Daniela Süsskind, Julia M. Weller, Yousef Yeganeh, Martina C. Herwig-Carl, Shadi Albarqouni

**发布时间:** 2025-12-27

### GPT解析

### 总结

研究者提出INTERACT-CMIL，一个多头深度学习框架，用于结膜黑色素细胞上皮内病变的准确分级，通过联合预测五个组织病理学轴，在多中心数据集上实现了优于现有方法的性能。

### 背景

结膜黑色素细胞上皮内病变的准确分级对治疗和黑色素瘤预测至关重要，但由于形态学线索微妙和诊断标准相互关联，准确分级仍然存在困难。

### 目的

开发一个能够准确分级CMIL的深度学习框架，提供连贯、可解释的多标准预测，与专家分级保持一致，并为CMIL诊断提供可重现的计算基准。

### 方法

提出INTERACT-CMIL，一个多头深度学习框架，通过共享特征学习和组合部分监督以及相互依赖损失函数来联合预测五个组织病理学轴。该框架在包含486个专家注释的结膜活检组织块的多中心数据集上进行训练和评估。

### 主要发现

INTERACT-CMIL实现了比CNN和基础模型基线一致的改进，相对宏观F1增益最高达到55.1%(WHO4)和25.0%(垂直扩散)，提供了与专家分级一致的可解释多标准预测。

### 结论

INTERACT-CMIL为CMIL诊断提供了可重现的计算基准，并向标准化数字眼科病理学迈出了一步。

### 翻译

准确的结膜黑色素细胞上皮内病变分级对于治疗和黑色素瘤预测至关重要，但由于形态学线索微妙和诊断标准相互关联，这一过程仍然存在困难。我们引入了INTERACT-CMIL，一个多头深度学习框架，通过共享特征学习和组合部分监督以及相互依赖损失函数，联合预测五个组织病理学轴：WHO4、WHO5、水平扩散、垂直扩散和细胞学异型性。在来自三家大学医院的486个专家注释的结膜活检组织块的新构建多中心数据集上进行训练和评估后，INTERACT-CMIL实现了比CNN和基础模型基线一致的改进，相对宏观F1增益最高达到55.1%(WHO4)和25.0%(垂直扩散)。该框架提供了与专家分级一致的可解释、连贯的多标准预测，为CMIL诊断提供了可重现的计算基准，并向标准化数字眼科病理学迈出了一步。


### 论文摘要

Accurate grading of Conjunctival Melanocytic Intraepithelial Lesions (CMIL) is essential for treatment and melanoma prediction but remains difficult due to subtle morphological cues and interrelated diagnostic criteria. We introduce INTERACT-CMIL, a multi-head deep learning framework that jointly predicts five histopathological axes; WHO4, WHO5, horizontal spread, vertical spread, and cytologic atypia, through Shared Feature Learning with Combinatorial Partial Supervision and an Inter-Dependence Loss enforcing cross-task consistency. Trained and evaluated on a newly curated, multi-center dataset of 486 expert-annotated conjunctival biopsy patches from three university hospitals, INTERACT-CMIL achieves consistent improvements over CNN and foundation-model (FM) baselines, with relative macro F1 gains up to 55.1% (WHO4) and 25.0% (vertical spread). The framework provides coherent, interpretable multi-criteria predictions aligned with expert grading, offering a reproducible computational benchmark for CMIL diagnosis and a step toward standardized digital ocular pathology.

---

## 70. 论文ID: 2512.22624v1

**错误:** 无法读取论文信息 - [Errno 2] No such file or directory: 'storage/result/2512.22624v1.json'

---

## 71. Lessons from Neuroscience for AI: How integrating Actions, Compositional Structure and Episodic Memory could enable Safe, Interpretable and Human-Like AI

**论文链接:** [http://arxiv.org/abs/2512.22568v1](http://arxiv.org/abs/2512.22568v1)

**作者:** Rajesh P. N. Rao, Vishwas Sathish, Linxing Preston Jiang, Matthew Bryan, Prashant Rangarajan

**发布时间:** 2025-12-27

### GPT解析

### 总结

这篇论文讨论了基础模型(如大型语言模型)的当前局限性，并提出通过整合行动、分层组合结构和情节记忆来改进这些模型，以实现更安全、可解释、节能且类人的人工智能。

### 背景

大型语言模型和其他基础模型最近取得了显著进展，这些进展主要是通过最小化下一个token预测损失来优化大规模transformer模型实现的，这是一种预测编码的形式，也是神经科学和认知科学中日益流行的脑功能模型的基础。

### 目的

为了实现安全、可解释、节能且类人的人工智能，基础模型应该在不同抽象层次上整合行动、组合生成架构和情节记忆。

### 方法

作者提出通过添加三个缺失的组件来改进基础模型：行动与生成模型的紧密集成、分层组合结构和情节记忆。他们引用了神经科学和认知科学的证据支持这些组件的重要性，并讨论了如何将这些组件整合到现有模型中。

### 主要发现

当前基础模型存在几个关键缺陷，包括由于缺乏基础导致的幻觉和对概念的肤浅理解、由于缺乏控制导致的代理/责任感缺失、由于缺乏可解释性对安全和可信度的威胁，以及能源效率低下。添加行动、分层组合结构和情节记忆可以解决这些问题。

### 结论

重新点燃脑科学与人工智能之间历史上富有成效的思想交流，将为安全和可解释的以人为本的人工智能铺平道路。

### 翻译

过去几年中，大型语言模型和其他基础模型的显著进步是基于最小化下一个token预测损失这一令人惊讶的简单目标来优化大规模transformer模型实现的，这是一种预测编码形式，也是神经科学和认知科学中日益流行的脑功能模型的基础。然而，当前基础模型忽略了最先进的预测编码模型的三个其他重要组成部分：行动与生成模型的紧密集成、分层组合结构和情节记忆。我们提出，为了实现安全、可解释、节能且类人的人工智能，基础模型应该在不同抽象层次上整合行动、组合生成架构和情节记忆。我们展示了神经科学和认知科学关于每个组件重要性的最新证据。我们描述了将这些缺失组件添加到基础模型中如何帮助解决一些当前缺陷：由于缺乏基础导致的幻觉和对概念的肤浅理解、由于缺乏控制导致的代理/责任感缺失、由于缺乏可解释性对安全和可信度的威胁，以及能源效率低下。我们将我们的提议与当前趋势进行了比较，例如为基础模型添加思维链推理和检索增强生成，并讨论了用大脑启发的组件增强这些模型的新方法。我们最后认为，重新点燃脑科学与人工智能之间历史上富有成效的思想交流，将为安全和可解释的以人为本的人工智能铺平道路。


### 论文摘要

The phenomenal advances in large language models (LLMs) and other foundation models over the past few years have been based on optimizing large-scale transformer models on the surprisingly simple objective of minimizing next-token prediction loss, a form of predictive coding that is also the backbone of an increasingly popular model of brain function in neuroscience and cognitive science. However, current foundation models ignore three other important components of state-of-the-art predictive coding models: tight integration of actions with generative models, hierarchical compositional structure, and episodic memory. We propose that to achieve safe, interpretable, energy-efficient, and human-like AI, foundation models should integrate actions, at multiple scales of abstraction, with a compositional generative architecture and episodic memory. We present recent evidence from neuroscience and cognitive science on the importance of each of these components. We describe how the addition of these missing components to foundation models could help address some of their current deficiencies: hallucinations and superficial understanding of concepts due to lack of grounding, a missing sense of agency/responsibility due to lack of control, threats to safety and trustworthiness due to lack of interpretability, and energy inefficiency. We compare our proposal to current trends, such as adding chain-of-thought (CoT) reasoning and retrieval-augmented generation (RAG) to foundation models, and discuss new ways of augmenting these models with brain-inspired components. We conclude by arguing that a rekindling of the historically fruitful exchange of ideas between brain science and AI will help pave the way towards safe and interpretable human-centered AI.

---

## 72. SAM 3D for 3D Object Reconstruction from Remote Sensing Images

**论文链接:** [http://arxiv.org/abs/2512.22452v1](http://arxiv.org/abs/2512.22452v1)

**作者:** Junsheng Yao, Lichao Mou, Qingyu Li

**发布时间:** 2025-12-27

### GPT解析

### 总结

本研究评估了SAM 3D基础模型在单目遥感建筑物重建中的应用，并与TRELLIS方法进行了比较，展示了其在城市场景重建中的潜力

### 背景

单目遥感图像中的3D建筑物重建对于可扩展的城市建模至关重要，但现有方法通常需要特定任务架构和密集监督

### 目的

对SAM 3D这个通用图像到3D基础模型进行单目遥感建筑物重建的系统评估，并探索其在城市场景重建中的潜力

### 方法

在纽约城市数据集样本上对SAM 3D与TRELLIS进行基准测试，使用Frechet Inception Distance和基于CLIP的最大均值差异作为评估指标，并通过分割-重建-组合流程将SAM 3D扩展到城市场景重建

### 主要发现

实验结果表明，与TRELLIS相比，SAM 3D能够产生更连贯的屋顶几何形状和更清晰的边界，在城市场景建模中展现出良好潜力

### 结论

这些发现为基础模型在城市3D重建中的部署提供了实际指导，并激励未来整合场景级别结构先验的研究

### 翻译

从遥感图像中进行单目3D建筑物重建对于可扩展的城市建模至关重要，然而现有方法通常需要特定任务架构和密集监督。本文首次对SAM 3D（一种通用图像到3D基础模型）进行了系统评估，用于单目遥感建筑物重建。我们在纽约城市数据集的样本上对SAM 3D与TRELLIS进行了基准测试，采用Frechet Inception Distance和基于CLIP的最大均值差异作为评估指标。实验结果表明，与TRELLIS相比，SAM 3D能够产生更连贯的屋顶几何形状和更清晰的边界。我们进一步通过分割-重建-组合流程将SAM 3D扩展到城市场景重建中，展示了其在城市场景建模中的潜力。我们还分析了实际局限性并讨论了未来研究方向。这些发现为基础模型在城市3D重建中的部署提供了实际指导，并激励未来整合场景级别结构先验的研究。


### 论文摘要

Monocular 3D building reconstruction from remote sensing imagery is essential for scalable urban modeling, yet existing methods often require task-specific architectures and intensive supervision. This paper presents the first systematic evaluation of SAM 3D, a general-purpose image-to-3D foundation model, for monocular remote sensing building reconstruction. We benchmark SAM 3D against TRELLIS on samples from the NYC Urban Dataset, employing Frechet Inception Distance (FID) and CLIP-based Maximum Mean Discrepancy (CMMD) as evaluation metrics. Experimental results demonstrate that SAM 3D produces more coherent roof geometry and sharper boundaries compared to TRELLIS. We further extend SAM 3D to urban scene reconstruction through a segment-reconstruct-compose pipeline, demonstrating its potential for urban scene modeling. We also analyze practical limitations and discuss future research directions. These findings provide practical guidance for deploying foundation models in urban 3D reconstruction and motivate future integration of scene-level structural priors.

---

## 73. Bright 4B: Scaling Hyperspherical Learning for Segmentation in 3D Brightfield Microscopy

**论文链接:** [http://arxiv.org/abs/2512.22423v1](http://arxiv.org/abs/2512.22423v1)

**作者:** Amil Khan, Matheus Palhares Viana, Suraj Mishra, B. S. Manjunath

**发布时间:** 2025-12-27

**备注:** 20 pages, 15 figures

### GPT解析

### 总结

Bright-4B是一种40亿参数的基础模型，能够在无需荧光标记的情况下直接从3D明场体积中分割亚细胞结构，解决了传统方法依赖荧光或复杂后处理的局限性。

### 背景

无标记3D明场显微镜是一种快速和非侵入性的可视化细胞形态的方法，但稳健的体积分割通常仍依赖于荧光或复杂的后处理。

### 目的

开发一种能够直接从3D明场体积中分割亚细胞结构的模型，解决无标记3D明场显微镜中稳健体积分割的挑战。

### 方法

Bright-4B结合了硬件对齐的原生稀疏注意力机制、深度-宽度残差超连接、软混合专家以及即插即用的各向异性嵌入，在单位超球面上学习，实现几何保真的3D标记化。

### 主要发现

Bright-4B仅从明场堆栈中就能产生核、线粒体和其他细胞器的形态准确的分割结果，不需要荧光、辅助通道或手工后处理；在多个共聚焦数据集上保留了跨深度和细胞类型的精细结构细节，优于当代CNN和Transformer基线模型。

### 结论

Bright-4B模型解决了无标记3D明场显微镜中稳健体积分割的挑战，将推动大规模无标记3D细胞图谱绘制的发展。

### 翻译

无标记3D明场显微镜提供了一种快速和非侵入性的可视化细胞形态的方法，然而稳健的体积分割通常仍依赖于荧光或复杂的后处理。我们通过引入Bright-4B来解决这一差距，这是一个拥有40亿参数的基础模型，它在单位超球面上学习，直接从3D明场体积中分割亚细胞结构。Bright-4B结合了硬件对齐的原生稀疏注意力机制(捕获局部、粗粒度和选定全局上下文)，深度-宽度残差超连接稳定表示流，以及自适应容量的软混合专家。即插即用的各向异性嵌入进一步尊重共聚焦点扩散函数和轴向变薄，实现了几何保真的3D标记化。由此产生的模型仅从明场堆栈中就能产生核、线粒体和其他细胞器的形态准确的分割结果——不需要荧光、辅助通道或手工后处理。在多个共聚焦数据集上，Bright-4B保留了跨深度和细胞类型的精细结构细节，优于当代CNN和Transformer基线模型。所有代码、预训练权重和用于下游微调的模型都将发布，以推动大规模无标记3D细胞图谱绘制。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决如何从3D明场显微镜图像中直接分割亚细胞结构的问题，而不需要荧光标记或复杂的后处理。这个问题很重要，因为明场显微镜是一种快速、非侵入性的细胞成像方法，但传统上难以从中获得准确的分割结果。解决这个问题将使研究人员能够在不干扰细胞的情况下观察细胞结构和动态，对活细胞研究至关重要，并能大大提高处理大量图像数据的效率。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先分析了3D明场显微镜的特点：轴向各向异性、深度依赖的对比度变化和低对比度信号。他们指出现有方法（如CNN和Transformer）在处理这些特性时存在局限性。作者借鉴了多种现有技术，包括Native Sparse Attention、HyperConnections和Mixture-of-Experts，但针对明场显微镜的特殊需求进行了创新设计。他们提出在单位超球面上进行学习以提高数值稳定性，使用各向异性3D块嵌入尊重显微镜的物理特性，并结合稀疏注意力和软混合专家来处理明场数据的挑战。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是在单位超球面上进行学习，使所有令牌状态和投影都进行单位归一化，从而提高大规模模型的数值稳定性。整体流程包括：1) 使用各向异性3D块嵌入处理输入图像，只在z轴上进行抗混滤波；2) 通过本机稀疏注意力机制捕获局部、粗粒度和全局上下文；3) 使用软混合专家进行自适应容量处理；4) 通过动态超连接稳定表示流；5) 最后使用轻量级解码器生成全分辨率的亚细胞结构掩码。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1) 各向异性3D块嵌入，与显微镜PSF对齐；2) 在单位超球面上学习，提高数值稳定性；3) 本机稀疏注意力机制，结合三种信息路径；4) 软混合专家，实现稳定的专家专门化；5) 动态超连接，自适应混合残差流。相比之前的工作，Bright 4B是一个专门为明场显微镜设计的40亿参数基础模型，直接从明场图像预测分割结果，不需要荧光标记或中间步骤，同时通过创新技术处理了明场数据的特殊挑战。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'Bright 4B是一种40亿参数的基础模型，通过在单位超球面上学习并结合各向异性嵌入、稀疏注意力和软混合专家，实现了从3D明场显微镜图像中直接分割亚细胞结构，无需荧光标记或复杂后处理。'}


### 论文摘要

Label-free 3D brightfield microscopy offers a fast and noninvasive way to visualize cellular morphology, yet robust volumetric segmentation still typically depends on fluorescence or heavy post-processing. We address this gap by introducing Bright-4B, a 4 billion parameter foundation model that learns on the unit hypersphere to segment subcellular structures directly from 3D brightfield volumes. Bright-4B combines a hardware-aligned Native Sparse Attention mechanism (capturing local, coarse, and selected global context), depth-width residual HyperConnections that stabilize representation flow, and a soft Mixture-of-Experts for adaptive capacity. A plug-and-play anisotropic patch embed further respects confocal point-spread and axial thinning, enabling geometry-faithful 3D tokenization. The resulting model produces morphology-accurate segmentations of nuclei, mitochondria, and other organelles from brightfield stacks alone--without fluorescence, auxiliary channels, or handcrafted post-processing. Across multiple confocal datasets, Bright-4B preserves fine structural detail across depth and cell types, outperforming contemporary CNN and Transformer baselines. All code, pretrained weights, and models for downstream finetuning will be released to advance large-scale, label-free 3D cell mapping.

---

## 74. 论文ID: 2512.22398v1

**错误:** 无法读取论文信息 - [Errno 2] No such file or directory: 'storage/result/2512.22398v1.json'

---

## 75. SciEvalKit: An Open-source Evaluation Toolkit for Scientific General Intelligence

**论文链接:** [http://arxiv.org/abs/2512.22334v1](http://arxiv.org/abs/2512.22334v1)

**作者:** Yiheng Wang, Yixin Chen, Shuo Li, Yifan Zhou, Bo Liu, Hengjian Gao, Jiakang Yuan, Jia Bu, Wanghan Xu, Yuhao Zhou, Xiangyu Zhao, Zhiwang Zhou, Fengxiang Wang, Haodong Duan, Songyang Zhang, Jun Yao, Han Deng, Yizhou Wang, Jiabei Xiao, Jiaqi Liu, Encheng Su, Yujie Liu, Weida Wang, Junchi Yao, Shenghe Zheng, Haoran Sun, Runmin Ma, Xiangchao Yan, Bo Zhang, Dongzhan Zhou, Shufei Zhang, Peng Ye, Xiaosong Wang, Shixiang Tang, Wenlong Zhang, Lei Bai

**发布时间:** 2025-12-26

### GPT解析

### 总结

SciEvalKit是一个用于评估科学AI模型的统一基准测试工具包，专注于科学智能的核心能力，支持多个科学领域，提供灵活可扩展的评估流程，是开源且积极维护的。

### 背景

现有通用评估平台无法满足科学AI模型的特殊评估需求，需要专门针对科学智能核心能力的评估工具。

### 目的

开发一个统一的基准测试工具包，用于评估AI模型在科学领域的核心能力和跨学科表现。

### 方法

构建专家级科学基准，使用真实世界领域特定数据集，设计灵活可扩展的评估流程，支持批量评估和自定义集成。

### 主要发现

SciEvalKit能够有效评估科学AI模型在多个领域和任务类型上的表现，提供透明、可复制和可比较的结果。

### 结论

SciEvalKit为科学基础模型和智能体的评估提供了标准化且可定制的基础设施，通过开源和社区维护促进AI4Science的发展。

### 翻译

我们引入了SciEvalKit，这是一个统一的基准测试工具包，旨在评估跨广泛科学学科和任务能力的科学AI模型。与通用评估平台不同，SciEvalKit专注于科学智能的核心能力，包括科学多模态感知、科学多模态推理、科学多模态理解、科学符号推理、科学代码生成、科学假设生成和科学知识理解。它支持六个主要科学领域，从物理和化学到天文学和材料科学。SciEvalKit建立在专家级科学基准的基础上，这些基准来自真实世界的领域特定数据集，确保任务反映真实的科学挑战。该工具包具有灵活、可扩展的评估流程，支持跨模型和数据集的批量评估，支持自定义模型和数据集集成，并提供透明、可复制和可比较的结果。通过将基于能力的评估与学科多样性相结合，SciEvalKit提供了一个标准化且可定制的基准测试基础设施，用于评估下一代科学基础模型和智能体。该工具包是开源的，并积极维护，以促进AI4Science的社区驱动发展和进步。


### 论文摘要

We introduce SciEvalKit, a unified benchmarking toolkit designed to evaluate AI models for science across a broad range of scientific disciplines and task capabilities. Unlike general-purpose evaluation platforms, SciEvalKit focuses on the core competencies of scientific intelligence, including Scientific Multimodal Perception, Scientific Multimodal Reasoning, Scientific Multimodal Understanding, Scientific Symbolic Reasoning, Scientific Code Generation, Science Hypothesis Generation and Scientific Knowledge Understanding. It supports six major scientific domains, spanning from physics and chemistry to astronomy and materials science. SciEvalKit builds a foundation of expert-grade scientific benchmarks, curated from real-world, domain-specific datasets, ensuring that tasks reflect authentic scientific challenges. The toolkit features a flexible, extensible evaluation pipeline that enables batch evaluation across models and datasets, supports custom model and dataset integration, and provides transparent, reproducible, and comparable results. By bridging capability-based evaluation and disciplinary diversity, SciEvalKit offers a standardized yet customizable infrastructure to benchmark the next generation of scientific foundation models and intelligent agents. The toolkit is open-sourced and actively maintained to foster community-driven development and progress in AI4Science.

---

## 76. Le Cam Distortion: A Decision-Theoretic Framework for Robust Transfer Learning

**论文链接:** [http://arxiv.org/abs/2512.23617v1](http://arxiv.org/abs/2512.23617v1)

**作者:** Deniz Akdemir

**发布时间:** 2025-12-29

### GPT解析

### 总结

本文提出Le Cam Distortion框架，基于Le Cam统计实验理论，通过学习从源域到目标域的核函数实现转移学习，避免了传统无监督域适应方法中的负迁移问题

### 背景

分布偏移是现实世界机器学习的定义性挑战。主流的无监督域适应(UDA)方法通过强制特征不变性来对齐源域和目标域表示，但当域的信息量不均衡时，严格的不变性会导致信息破坏，引起'负迁移'

### 目的

提出决策论框架，用构造性近似替代对称不变性，实现方向可模拟性，并引入Le Cam失真作为可模拟条件下转移风险的严格上界

### 方法

基于Le Cam统计实验理论构建决策论框架，学习一个从源域模拟目标域的核函数，实现不降低源域效用的转移学习

### 主要发现

在五个实验中：(1)HLA基因组学中实现近乎完美的频率估计(r=0.999)；(2)CIFAR-10图像分类中保持81.2%准确率(无源效用损失)，而CycleGAN下降34.7%；(3)在强化学习控制中实现安全策略转移

### 结论

Le Cam Distortion为负迁移不可接受的领域(医学成像、自主系统和精准医疗)提供了首个有原则的风险控制转移学习框架

### 翻译

分布偏移是现实世界机器学习的定义性挑战。主导范式——无监督域适应(UDA)——强制特征不变性，通过对称散度最小化来对齐源域和目标域表示[Ganin等人，2016]。我们证明这种方法存在根本性缺陷：当域的信息量不均衡时，严格的不变性需要破坏信息，导致'负迁移'，这在安全关键应用中可能是灾难性的[Wang等人，2019]。我们提出了基于Le Cam统计实验理论的决策论框架，使用构造性近似来替代对称不变性，实现方向可模拟性。我们引入Le Cam失真，用缺陷距离δ(E₁, E₂)量化，作为可模拟条件下转移风险的严格上界。我们的框架通过学习一个从源域模拟目标域的核函数，实现了不降低源域效用的转移学习。在五个实验中，Le Cam失真实现了：(1)在HLA基因组学中近乎完美的频率估计；(2)在CIFAR-10图像分类中零源效用损失；(3)在强化学习控制中的安全策略转移。Le Cam失真为负迁移不可接受的领域提供了第一个有原则的风险控制转移学习框架


### 论文摘要

Distribution shift is the defining challenge of real-world machine learning. The dominant paradigm--Unsupervised Domain Adaptation (UDA)--enforces feature invariance, aligning source and target representations via symmetric divergence minimization [Ganin et al., 2016]. We demonstrate that this approach is fundamentally flawed: when domains are unequally informative (e.g., high-quality vs degraded sensors), strict invariance necessitates information destruction, causing "negative transfer" that can be catastrophic in safety-critical applications [Wang et al., 2019].   We propose a decision-theoretic framework grounded in Le Cam's theory of statistical experiments [Le Cam, 1986], using constructive approximations to replace symmetric invariance with directional simulability. We introduce Le Cam Distortion, quantified by the Deficiency Distance $δ(E_1, E_2)$, as a rigorous upper bound for transfer risk conditional on simulability. Our framework enables transfer without source degradation by learning a kernel that simulates the target from the source. Across five experiments (genomics, vision, reinforcement learning), Le Cam Distortion achieves: (1) near-perfect frequency estimation in HLA genomics (correlation $r=0.999$, matching classical methods), (2) zero source utility loss in CIFAR-10 image classification (81.2% accuracy preserved vs 34.7% drop for CycleGAN), and (3) safe policy transfer in RL control where invariance-based methods suffer catastrophic collapse. Le Cam Distortion provides the first principled framework for risk-controlled transfer learning in domains where negative transfer is unacceptable: medical imaging, autonomous systems, and precision medicine.

---

## 77. Unsupervised Learning for Detection of Rare Driving Scenarios

**论文链接:** [http://arxiv.org/abs/2512.23585v1](http://arxiv.org/abs/2512.23585v1)

**作者:** Dat Le, Thomas Manhardt, Moritz Venator, Johannes Betz

**发布时间:** 2025-12-29

### GPT解析

### 总结

本研究提出了一种基于无监督学习的框架，用于检测稀有和危险的驾驶场景，使用深度隔离森林算法处理自然驾驶数据，有效识别复杂异常，为自动驾驶系统安全提供可扩展解决方案。

### 背景

检测稀有和危险的驾驶场景是确保自主系统安全性和可靠性的关键挑战，研究使用自然驾驶数据(NDD)来应对这一挑战。

### 目的

探索一种无监督学习框架，用于检测稀有和极端驾驶场景，提高自动驾驶系统的安全性。

### 方法

采用深度隔离森林(DIF)异常检测算法，结合神经网络特征表示与隔离森林识别非线性和复杂异常；将感知模块数据预处理为滑动窗口提取的结构化统计特征；使用t-SNE进行降维和可视化以提高可解释性；通过代理真实值结合定量指标和定性视频帧检查进行评估。

### 主要发现

提出的框架能有效识别稀有和危险的驾驶场景，为自动驾驶系统中的异常检测提供了可扩展的解决方案。

### 结论

研究方法不可避免地依赖代理真实值和手动定义的特征组合，无法涵盖真实世界驾驶异常的全部范围或其细微的上下文依赖关系。

### 翻译

稀有和危险驾驶场景的检测是确保自主系统安全性和可靠性的关键挑战。本研究探索了一种使用自然驾驶数据(NDD)检测稀有和极端驾驶场景的无监督学习框架。我们利用最近提出的深度隔离森林(DIF)异常检测算法，该算法结合了基于神经网络的特征表示与隔离森林(IFs)，以识别非线性和复杂异常。来自感知模块的数据(捕获车辆动力学和环境条件)被预处理为从滑动窗口中提取的结构化统计特征。该框架结合了t-SNE进行降维和可视化，使检测到的异常更具可解释性。评估使用代理真实值进行，结合定量指标和定性视频帧检查。我们的结果表明，所提出的方法能有效识别稀有和危险的驾驶场景，为自动驾驶系统中的异常检测提供了可扩展的解决方案。鉴于研究的方法，不可避免地依赖代理真实值和手动定义的特征组合，这些方法无法涵盖真实世界驾驶异常的全部范围或其细微的上下文依赖关系。


### 论文摘要

The detection of rare and hazardous driving scenarios is a critical challenge for ensuring the safety and reliability of autonomous systems. This research explores an unsupervised learning framework for detecting rare and extreme driving scenarios using naturalistic driving data (NDD). We leverage the recently proposed Deep Isolation Forest (DIF), an anomaly detection algorithm that combines neural network-based feature representations with Isolation Forests (IFs), to identify non-linear and complex anomalies. Data from perception modules, capturing vehicle dynamics and environmental conditions, is preprocessed into structured statistical features extracted from sliding windows. The framework incorporates t-distributed stochastic neighbor embedding (t-SNE) for dimensionality reduction and visualization, enabling better interpretability of detected anomalies. Evaluation is conducted using a proxy ground truth, combining quantitative metrics with qualitative video frame inspection. Our results demonstrate that the proposed approach effectively identifies rare and hazardous driving scenarios, providing a scalable solution for anomaly detection in autonomous driving systems. Given the study's methodology, it was unavoidable to depend on proxy ground truth and manually defined feature combinations, which do not encompass the full range of real-world driving anomalies or their nuanced contextual dependencies.

---

## 78. A NEAT Approach to Evolving Neural-Network-based Optimization of Chiral Photonic Metasurfaces: Application of a Neuro-Evolution Pipeline

**论文链接:** [http://arxiv.org/abs/2512.23558v1](http://arxiv.org/abs/2512.23558v1)

**作者:** Davide Filippozzi, Arash Rahimi-Iman

**发布时间:** 2025-12-29

### GPT解析

### 总结

该研究将神经进化增强拓扑(NEAT)算法集成到深度学习优化框架中，用于电介质手性超表面的设计，实现了无需手动调整的任务特定架构，并展示了与现有方法相当或更好的性能。

### 背景

手性超表面的设计在纳米光子学中是一个核心挑战，因为几何结构与手性光学响应之间存在高度非线性关系。机器学习辅助的优化流程虽能加速这一过程，但其性能很大程度上取决于神经网络架构的选择。

### 目的

将NEAT算法集成到现有的深度学习优化框架中，用于电介质手性超表面的设计，实现自适应、自配置的机器学习框架。

### 方法

使用NEAT算法自主进化网络拓扑和连接权重，结合强化学习策略进化解空间知识并并行微调模型权重。使用9600个模拟GaP超表面几何形状的数据集，在不同输入维度、特征缩放方法和数据大小下评估NEAT性能。

### 主要发现

标准化特征缩放提供了最一致的性能；相对紧凑的NEAT进化NN模型实现了与密集少层感知器相当或更好的预测准确性和泛化能力；这些模型成功推理可见光谱中强圆二色性超表面，并支持模拟与实验数据间的迁移学习。

### 结论

该方法为自适应、自配置的机器学习框架提供了一条可扩展的路径，可用于自动化的光子设计，既可以独立使用，也可以作为智能人工智能的构建块。

### 翻译

由于几何结构与手性光学响应之间的高度非线性关系，设计具有定制光学性质的手性超表面在纳米光子学中仍然是一个核心挑战。机器学习辅助的优化流程最近已成为加速这一过程的有效工具，但其性能在很大程度上取决于神经网络架构的选择。在这项工作中，我们将神经进化增强拓扑(NEAT)算法集成到一个现有的深度学习优化框架中，用于电介质手性超表面。NEAT自主进化网络拓扑和连接权重，实现无需手动调整的任务特定架构，而框架中的强化学习策略并行进化解空间知识和微调模型权重。使用包含9600个模拟GaP超表面几何形状的管道生成数据集，我们在不同输入维度、特征缩放方法和数据大小下评估了NEAT。对于检查的输出维度，标准化特征缩放提供了最一致的性能，当集成到完整优化管道中时，相对紧凑的NEAT进化NN模型实现了与最初使用的密集少层感知器相似或更好的预测准确性和泛化能力。因此，这些资源高效的模型成功执行了对可见光谱中表现出强圆二色性的超表面的推理，允许在模拟数据和实验数据之间进行迁移学习。这种方法展示了自适应、自配置机器学习框架的可扩展路径，用于自动化的光子设计，既可以独立使用，也可以作为智能人工智能的构建块。


### 论文摘要

The design of chiral metasurfaces with tailored optical properties remains a central challenge in nanophotonics due to the highly nonlinear relationship between geometry and chiroptical response. Machine-learning-assisted optimization pipelines have recently emerged as efficient tools to accelerate this process, yet their performance strongly depends on the choice of neural-network (NN) architecture. In this work, we integrate the NeuroEvolution of Augmenting Topologies (NEAT) algorithm into an established deep-learning optimization framework for dielectric chiral metasurfaces. NEAT autonomously evolves both network topology and connection weights, enabling task-specific architectures without manual tuning, whereas the reinforcement-learning strategy in our framework evolves knowledge of the solution space and fine-tunes a model's weights in parallel. Using a pipeline-produced dataset of 9,600 simulated GaP metasurface geometries, we evaluate NEAT under varying input dimensionalities, feature-scaling methods, and data sizes. With standardized feature scaling yielding the most consistent performance for both examined output dimensionalities, the relatively compact NEAT-evolved NN models, when integrated into the full optimization pipeline, achieve similar or improved predictive accuracy and generalization compared to initially employed dense few-layer perceptrons. Accordingly, these resource-efficient models successfully perform inference of metasurfaces exhibiting strong circular dichroism in the visible spectrum, allowing for transfer learning between simulated and experimental data. This approach demonstrates a scalable path toward adaptive, self-configuring machine-learning frameworks for automated photonic design both standalone and as building block for agentic artificial intelligence (AI).

---

## 79. Universal and Experiment-calibrated Prediction of XANES through Crystal Graph Neural Network and Transfer Learning Strategy

**论文链接:** [http://arxiv.org/abs/2512.23449v1](http://arxiv.org/abs/2512.23449v1)

**作者:** Zichang Lin, Wenjie Chen, Yitao Lin, Xinxin Zhang, Yuegang Zhang

**发布时间:** 2025-12-29

### GPT解析

### 总结

本研究开发了一种结合晶体图神经网络和迁移学习的方法，用于快速、通用且经过实验校准的X射线吸收近边结构(XANES)预测。

### 背景

理论模拟有助于准确解释实验X射线吸收近边结构(XANES)光谱，这些光谱包含材料丰富的原子和电子结构信息。然而，当需要分析大量数据时，如电池材料的原位表征，当前模拟方法过于复杂，无法提供所需的准确性和及时性。

### 目的

解决现有XANES预测模型的问题，包括使用模拟数据训练导致预测与实验光谱差异大，以及模型在不同元素间的通用性不足。

### 方法

建立晶体图神经网络，首先在覆盖48种元素的模拟XANES数据上预训练，然后利用迁移学习使用少量实验XANES数据集进行校准。

### 主要发现

预训练的晶体图神经网络实现了平均相对平方误差低至0.020223的通用XANES预测；校准后，预测的S、Ti和Fe K边XANES的边能量失配误差显著减少了约55%。

### 结论

本研究展示的方法为实现快速、通用且经过实验校准的XANES预测开辟了新途径。

### 翻译

理论模拟有助于准确解释包含材料丰富原子和电子结构信息的实验X射线吸收近边结构(XANES)光谱。然而，当需要分析大量数据时，如电池材料的原位表征，当前的模拟方法通常过于复杂，无法提供所需的准确性和及时性。为解决这些问题，已经开发了人工智能(AI)模型用于XANES预测。然而，现有模型使用模拟数据而非实验XANES数据进行训练，导致预测光谱与实验光谱之间存在显著差异。此外，这类模型在不同元素间的通用性尚未得到充分研究。在本工作中，我们首先建立了一个晶体图神经网络，在覆盖48种元素的模拟XANES数据上进行预训练，实现了平均相对平方误差低至0.020223的通用XANES预测；然后利用迁移学习使用少量实验XANES数据集对模型进行校准。校准后，预测的S、Ti和Fe K边XANES的边能量失配误差显著减少了约55%。本研究展示的方法为实现快速、通用且经过实验校准的XANES预测开辟了新途径。


### 论文摘要

Theoretical simulation is helpful for accurate interpretation of experimental X-ray absorption near-edge structure (XANES) spectra that contain rich atomic and electronic structure information of materials. However, current simulation methods are usually too complex to give the needed accuracy and timeliness when a large amount of data need to be analyzed, such as for in-situ characterization of battery materials. To address these problems, artificial intelligence (AI) models have been developed for XANES prediction. However, instead of using experimental XANES data, the existing models are trained using simulated data, resulting in significant discrepancies between the predicted and experimental spectra. Also, the universality across different elements has not been well studied for such models. In this work, we firstly establish a crystal graph neural network, pre-trained on simulated XANES data covering 48 elements, to achieve universal XANES prediction with a low average relative square error of 0.020223; and then utilize transfer learning to calibrate the model using a small experimental XANES dataset. After calibration, the edge energy misalignment error of the predicted S, Ti and Fe K edge XANES is significantly reduced by about 55%. The method demonstrated in this work opens up a new way to achieve fast, universal, and experiment-calibrated XANES prediction.

---

## 80. The Quest for Winning Tickets in Low-Rank Adapters

**论文链接:** [http://arxiv.org/abs/2512.22495v1](http://arxiv.org/abs/2512.22495v1)

**作者:** Hamed Damirchi, Cristian Rodriguez-Opazo, Ehsan Abbasnejad, Zhen Zhang, Javen Shi

**发布时间:** 2025-12-27

**备注:** 21 pages

### GPT解析

### 总结

本研究探讨了彩票假设在低秩适配(LoRA)参数高效微调方法中的适用性，提出了Partial-LoRA方法，通过识别稀疏子网络显著减少可训练参数数量同时保持或提高模型性能。

### 背景

彩票假设表明过参数化神经网络包含稀疏子网络('中奖票')，可以从头开始训练时达到与完整模型相当的性能。随着对微调大型预训练模型的依赖增加，需要研究LTH是否可以扩展到参数高效微调方法。

### 目的

调查彩票假设是否适用于参数高效微调方法，特别是低秩适配(LoRA)方法，并探索如何在LoRA中识别和利用稀疏子网络以提高效率。

### 方法

提出Partial-LoRA方法，系统地识别稀疏子网络并训练与预训练模型任务相关子空间对齐的稀疏低秩适配器。在8个视觉和12个语言任务上进行实验，包括单任务和多任务设置。

### 主要发现

彩票假设在LoRA中成立，存在可以匹配密集适配器性能的稀疏子网络；稀疏子网络的有效性主要取决于每层应用的稀疏程度，而非子网络中包含的确切权重；Partial-LoRA可将可训练参数减少高达87%同时保持或提高准确性。

### 结论

研究结果加深了对迁移学习以及预训练和微调之间相互作用的理论理解，为开发更高效的模型适应策略开辟了新途径。

### 翻译

彩票假设(LTH)表明，过参数化的神经网络包含稀疏子网络('中奖票')，这些子网络可以从头开始训练时达到与完整模型相当的性能。随着对微调大型预训练模型的依赖日益增加，我们研究了LTH是否可以扩展到参数高效微调(PEFT)方法，特别是低秩适配(LoRA)方法。我们的主要发现是LTH在LoRA中成立，揭示了可以匹配密集适配器性能的稀疏子网络。具体来说，我们发现稀疏子网络的有效性更多地取决于每层应用的稀疏程度，而非子网络中包含的确切权重。基于这一见解，我们提出了Partial-LoRA方法，该方法系统地识别上述子网络并训练与预训练模型任务相关子空间对齐的稀疏低秩适配器。在8个视觉和12个语言任务上的实验(包括单任务和多任务设置)表明，Partial-LoRA将可训练参数数量减少了高达87%，同时保持或提高了准确性。我们的结果不仅加深了我们对迁移学习以及预训练和微调之间相互作用的理论理解，还为开发更高效的适应策略开辟了新途径。


### 论文摘要

The Lottery Ticket Hypothesis (LTH) suggests that over-parameterized neural networks contain sparse subnetworks ("winning tickets") capable of matching full model performance when trained from scratch. With the growing reliance on fine-tuning large pretrained models, we investigate whether LTH extends to parameter-efficient fine-tuning (PEFT), specifically focusing on Low-Rank Adaptation (LoRA) methods. Our key finding is that LTH holds within LoRAs, revealing sparse subnetworks that can match the performance of dense adapters. In particular, we find that the effectiveness of sparse subnetworks depends more on how much sparsity is applied in each layer than on the exact weights included in the subnetwork. Building on this insight, we propose Partial-LoRA, a method that systematically identifies said subnetworks and trains sparse low-rank adapters aligned with task-relevant subspaces of the pre-trained model. Experiments across 8 vision and 12 language tasks in both single-task and multi-task settings show that Partial-LoRA reduces the number of trainable parameters by up to 87\%, while maintaining or improving accuracy. Our results not only deepen our theoretical understanding of transfer learning and the interplay between pretraining and fine-tuning but also open new avenues for developing more efficient adaptation strategies.

---

## 81. 论文ID: 2512.22252v1

**错误:** 无法读取论文信息 - [Errno 2] No such file or directory: 'storage/result/2512.22252v1.json'

---

## 82. Predicting Mycotoxin Contamination in Irish Oats Using Deep and Transfer Learning

**论文链接:** [http://arxiv.org/abs/2512.22243v1](http://arxiv.org/abs/2512.22243v1)

**作者:** Alan Inglis, Fiona Doohan, Subramani Natarajan, Breige McNulty, Chris Elliott, Anne Nugent, Julie Meneely, Brett Greer, Stephen Kildea, Diana Bucur, Martin Danaher, Melissa Di Rocco, Lisa Black, Adam Gauley, Naoise McKenna, Andrew Parnell

**发布时间:** 2025-12-23

**备注:** 28 pages, 11 Figures, Supplementary Materials

### GPT解析

### 总结

该研究成功应用神经网络和迁移学习模型预测爱尔兰燕麦作物中的霉菌毒素污染，发现收获前90天的天气历史模式和种子含水量是最重要预测因素，TabPFN模型表现最佳。

### 背景

霉菌毒素污染对谷物作物质量、食品安全和农业生产力构成重大风险，准确预测霉菌毒素水平可支持早期干预并减少经济损失。

### 目的

研究使用神经网络和迁移学习模型预测爱尔兰燕麦作物中的霉菌毒素污染，作为多响应预测任务处理。

### 方法

使用爱尔兰收集的燕麦样本数据集(包含环境、农业和地理预测变量)，评估五种建模方法：基线MLP、预训练MLP和三种迁移学习模型(TabPFN、TabNet和FT-Transformer)。使用回归和分类指标评估性能，并进行变量重要性分析。

### 主要发现

TabPFN迁移学习方法提供了最佳整体性能，其次是基线MLP。收获前90天的天气历史模式和种子含水量是最重要的预测变量。

### 结论

迁移学习方法(特别是TabPFN)在预测霉菌毒素污染方面表现良好，天气条件和种子含水量是预测霉菌毒素水平的关键因素。

### 翻译

霉菌毒素污染对谷物作物质量、食品安全和农业生产力构成显著风险。准确预测霉菌毒素水平可以支持早期干预策略并减少经济损失。本研究调查了使用神经网络和迁移学习模型预测爱尔兰燕麦作物中霉菌毒素污染，作为多响应预测任务。我们的数据集包含在爱尔兰收集的燕麦样本，混合了环境、农业和地理预测变量。评估了五种建模方法：基线多层感知器、预训练的多层感知器和三种迁移学习模型；TabPFN、TabNet和FT-Transformer。使用回归和分类指标评估模型性能，结果按毒素类型和平均值报告。此外，进行了基于排列的变量重要性分析，以识别两个预测任务中最有影响力的预测变量。迁移学习方法TabPFN提供了最佳整体性能，其次是基线多层感知器。我们的变量重要性分析显示，收获前90天的天气历史模式是最重要的预测变量，以及种子含水量。


### 论文摘要

Mycotoxin contamination poses a significant risk to cereal crop quality, food safety, and agricultural productivity. Accurate prediction of mycotoxin levels can support early intervention strategies and reduce economic losses. This study investigates the use of neural networks and transfer learning models to predict mycotoxin contamination in Irish oat crops as a multi-response prediction task. Our dataset comprises oat samples collected in Ireland, containing a mix of environmental, agronomic, and geographical predictors. Five modelling approaches were evaluated: a baseline multilayer perceptron (MLP), an MLP with pre-training, and three transfer learning models; TabPFN, TabNet, and FT-Transformer. Model performance was evaluated using regression (RMSE, $R^2$) and classification (AUC, F1) metrics, with results reported per toxin and on average. Additionally, permutation-based variable importance analysis was conducted to identify the most influential predictors across both prediction tasks. The transfer learning approach TabPFN provided the overall best performance, followed by the baseline MLP. Our variable importance analysis revealed that weather history patterns in the 90-day pre-harvest period were the most important predictors, alongside seed moisture content.

---

## 83. CLIP Based Region-Aware Feature Fusion for Automated BBPS Scoring in Colonoscopy Images

**论文链接:** [http://arxiv.org/abs/2512.20374v2](http://arxiv.org/abs/2512.20374v2)

**作者:** Yujia Fu, Zhiyu Dong, Tianwen Qian, Chenye Zheng, Danian Ji, Linhai Zhuo

**发布时间:** 2025-12-23

**备注:** 12 pages, 9 figures, BMVC 2025 submission

### GPT解析

### 总结

本研究提出了一种基于CLIP模型的自动BBPS评分框架，通过融合全局视觉特征与粪便相关文本先验，提高肠道清洁度评估的准确性，无需显式分割即可实现。

### 背景

准确的肠道清洁度评估对有效的肠镜检查至关重要。波士顿肠道准备评分系统(BBPS)提供了标准化的评分系统，但手动操作时存在主观性和观察者间变异性的问题。

### 目的

为了支持稳健的训练和评估，构建高质量的肠镜数据集并提出一种新颖的自动BBPS评分框架，以解决手动评分中的主观性问题。

### 方法

构建包含517名受试者的2240张图像的高质量肠镜数据集，附有专家一致的BBPS评分注释；提出基于适配器迁移学习的CLIP模型和专门粪便特征提取分支的自动BBPS评分框架，融合全局视觉特征与粪便相关文本先验。

### 主要发现

在自建数据集和公共NERTHU数据集上的实验表明，该方法优于现有基线，突显了其在计算机辅助肠镜分析中临床部署的潜力。

### 结论

所提出的方法能够有效解决手动BBPS评分中的主观性和观察者间变异性问题，有望在临床实践中应用。

### 翻译

准确的肠道清洁度评估对有效的肠镜检查程序至关重要。波士顿肠道准备评分系统(BBPS)提供了标准化的评分系统，但在手动执行时存在主观性和观察者间变异性的问题。在本文中，为了支持稳健的训练和评估，我们构建了一个高质量的肠镜数据集，包含来自517名受试者的2240张图像，并附有专家一致的BBPS评分注释。我们提出了一种新颖的自动BBPS评分框架，利用基于适配器的迁移学习的CLIP模型和专门的粪便特征提取分支。我们的方法融合全局视觉特征与粪便相关的文本先验，无需显式分割即可提高肠道清洁度评估的准确性。在我们数据集和公共NERTHU数据集上的大量实验证明了我们方法相对于现有基线的优越性，突显了其在计算机辅助肠镜分析中临床部署的潜力。


### 论文摘要

Accurate assessment of bowel cleanliness is essential for effective colonoscopy procedures. The Boston Bowel Preparation Scale (BBPS) offers a standardized scoring system but suffers from subjectivity and inter-observer variability when performed manually. In this paper, to support robust training and evaluation, we construct a high-quality colonoscopy dataset comprising 2,240 images from 517 subjects, annotated with expert-agreed BBPS scores. We propose a novel automated BBPS scoring framework that leverages the CLIP model with adapter-based transfer learning and a dedicated fecal-feature extraction branch. Our method fuses global visual features with stool-related textual priors to improve the accuracy of bowel cleanliness evaluation without requiring explicit segmentation. Extensive experiments on both our dataset and the public NERTHU dataset demonstrate the superiority of our approach over existing baselines, highlighting its potential for clinical deployment in computer-aided colonoscopy analysis.

---

## 84. GeoTeacher: Geometry-Guided Semi-Supervised 3D Object Detection

**论文链接:** [http://arxiv.org/abs/2512.23147v1](http://arxiv.org/abs/2512.23147v1)

**作者:** Jingyu Li, Xiaolong Zhao, Zhe Liu, Wenxiao Wu, Li Zhang

**发布时间:** 2025-12-29

### GPT解析

### 总结

本文提出了一种名为GeoTeacher的半监督3D目标检测方法，通过几何关系监督和数据增强技术提升模型对物体几何信息的理解能力。

### 背景

半监督3D目标检测是近年来的活跃研究领域，现有方法主要通过异构教师模型或特征视角一致性来提升性能，但忽略了模型在标记数据有限时对物体几何敏感性低的问题。

### 目的

提出GeoTeacher方法，增强学生模型在有限训练数据（尤其是未标记数据）下捕获物体几何关系的能力。

### 方法

设计基于关键点的几何关系监督模块，将教师模型的几何知识转移到学生模型；引入体素级数据增强策略增加物体几何多样性；融入距离衰减机制保持远处物体完整性；GeoTeacher可与其他半监督3D检测方法结合使用。

### 主要发现

在ONCE和Waymo数据集上的大量实验表明该方法具有有效性和良好的泛化能力，并取得了新的最先进结果。

### 结论

GeoTeacher通过几何关系监督和数据增强策略，有效解决了半监督3D目标检测中模型对几何信息敏感性低的问题，提升了学生模型的目标感知和定位能力。

### 翻译

半监督3D目标检测旨在利用未标记数据提升3D目标检测器性能，近年来已成为一个活跃的研究领域。一些先前的方法通过使用异构教师模型提供高质量伪标签或强制教师和学生网络之间的特征视角一致性，已经显示出显著的改进。然而，这些方法忽略了模型通常在标记数据有限时对物体几何形状敏感性较低的事实，难以捕获几何信息，这对增强学生模型的目标感知和定位能力至关重要。在本文中，我们提出GeoTeacher来增强学生模型在有限训练数据下捕获物体几何关系的能力，特别是未标记数据。我们设计了一个基于关键点的几何关系监督模块，将教师模型对物体几何的知识转移到学生，从而提高学生理解几何关系的能力。此外，我们引入了一种体素级数据增强策略，增加了物体几何的多样性，从而进一步改善学生模型理解几何结构的能力。为了在增强过程中保持远处物体的完整性，我们将距离衰减机制纳入该策略。此外，GeoTeacher可以与不同的半监督3D检测方法结合，以进一步提高它们的性能。在ONCE和Waymo数据集上的大量实验表明了我们方法的有效性和泛化能力，我们取得了最新的最先进结果。代码将在https://github.com/SII-Whaleice/GeoTeacher上提供。


### 论文摘要

Semi-supervised 3D object detection, aiming to explore unlabeled data for boosting 3D object detectors, has emerged as an active research area in recent years. Some previous methods have shown substantial improvements by either employing heterogeneous teacher models to provide high-quality pseudo labels or enforcing feature-perspective consistency between the teacher and student networks. However, these methods overlook the fact that the model usually tends to exhibit low sensitivity to object geometries with limited labeled data, making it difficult to capture geometric information, which is crucial for enhancing the student model's ability in object perception and localization. In this paper, we propose GeoTeacher to enhance the student model's ability to capture geometric relations of objects with limited training data, especially unlabeled data. We design a keypoint-based geometric relation supervision module that transfers the teacher model's knowledge of object geometry to the student, thereby improving the student's capability in understanding geometric relations. Furthermore, we introduce a voxel-wise data augmentation strategy that increases the diversity of object geometries, thereby further improving the student model's ability to comprehend geometric structures. To preserve the integrity of distant objects during augmentation, we incorporate a distance-decay mechanism into this strategy. Moreover, GeoTeacher can be combined with different SS3D methods to further improve their performance. Extensive experiments on the ONCE and Waymo datasets indicate the effectiveness and generalization of our method and we achieve the new state-of-the-art results. Code will be available at https://github.com/SII-Whaleice/GeoTeacher

---

## 85. SCAFusion: A Multimodal 3D Detection Framework for Small Object Detection in Lunar Surface Exploration

**论文链接:** [http://arxiv.org/abs/2512.22503v1](http://arxiv.org/abs/2512.22503v1)

**作者:** Xin Chen, Kang Luo, Yangyi Xiao, Hesheng Wang

**发布时间:** 2025-12-27

### GPT解析

### 总结

本文提出了SCAFusion，一种专为月球机器人任务设计的多模态3D目标检测模型，通过创新机制显著提高了小、不规则目标的检测性能。

### 背景

月球表面探索中自主导航和操作需要可靠精确地检测小且不规则的目标，如陨石碎片和岩石。现有的为地球自动驾驶设计的多模态3D感知方法在其他世界环境中表现不佳，由于特征对齐差、多模态协同有限和小目标检测能力弱。

### 目的

开发一种专门针对月球环境的多模态3D目标检测模型，提高小、不规则目标的检测精度和可靠性。

### 方法

基于BEVFusion框架构建SCAFusion模型，集成了认知适配器用于高效调整相机主干网络，对比对齐模块增强相机和LiDAR特征一致性，相机辅助训练分支加强视觉表示，以及分段感知坐标注意力机制专门提升小、不规则目标的检测性能。

### 主要发现

在nuScenes验证集上，模型仅参数和计算量略有增加的情况下，实现了69.7%的mAP和72.1%的NDS，分别比基线提高了5.0%和2.7%。在模拟月球环境中，SCAFusion实现了90.93%的mAP，比基线提高了11.5%，在检测类似陨石的小障碍物方面有显著提升。

### 结论

SCAFusion是一种有效的多模态3D目标检测模型，特别适合月球表面探索任务，能够显著提高小、不规则目标的检测性能，为月球表面自主导航提供了可靠的感知能力。

### 翻译

月球表面探索中自主导航和操作对小型和不规则物体（如陨石碎片和岩石）的可靠精确检测至关重要。现有的为地球自动驾驶设计的多模态3D感知方法在其他世界环境中表现不佳，由于特征对齐差、多模态协同有限和小目标检测能力弱。本文提出了SCAFusion，一种为月球机器人任务定制化的多模态3D目标检测模型。基于BEVFusion框架构建，SCAFusion集成了认知适配器用于高效调整相机主干网络，对比对齐模块增强相机LiDAR特征一致性，相机辅助训练分支加强视觉表示，最重要的是，分段感知坐标注意力机制专门设计用于提升小、不规则目标的检测性能。在参数和计算量略有增加的情况下，我们的模型在nuScenes验证集上实现了69.7%的mAP和72.1%的NDS，分别比基线提高了5.0%和2.7%。在基于Isaac Sim构建的模拟月球环境中，SCAFusion实现了90.93%的mAP，比基线提高了11.5%，在检测类似陨石的小障碍物方面有显著提升。


### 论文摘要

Reliable and precise detection of small and irregular objects, such as meteor fragments and rocks, is critical for autonomous navigation and operation in lunar surface exploration. Existing multimodal 3D perception methods designed for terrestrial autonomous driving often underperform in off world environments due to poor feature alignment, limited multimodal synergy, and weak small object detection. This paper presents SCAFusion, a multimodal 3D object detection model tailored for lunar robotic missions. Built upon the BEVFusion framework, SCAFusion integrates a Cognitive Adapter for efficient camera backbone tuning, a Contrastive Alignment Module to enhance camera LiDAR feature consistency, a Camera Auxiliary Training Branch to strengthen visual representation, and most importantly, a Section aware Coordinate Attention mechanism explicitly designed to boost the detection performance of small, irregular targets. With negligible increase in parameters and computation, our model achieves 69.7% mAP and 72.1% NDS on the nuScenes validation set, improving the baseline by 5.0% and 2.7%, respectively. In simulated lunar environments built on Isaac Sim, SCAFusion achieves 90.93% mAP, outperforming the baseline by 11.5%, with notable gains in detecting small meteor like obstacles.

---

## 86. 论文ID: 2512.23622v1

**错误:** 无法读取论文信息 - [Errno 2] No such file or directory: 'storage/result/2512.23622v1.json'

---

## 87. The Gaining Paths to Investment Success: Information-Driven LLM Graph Reasoning for Venture Capital Prediction

**论文链接:** [http://arxiv.org/abs/2512.23489v1](http://arxiv.org/abs/2512.23489v1)

**作者:** Haoyu Pei, Zhongyang Liu, Xiangyi Xiao, Xiaocong Du, Haipeng Zhang, Kunpeng Zhang, Suting Hong

**发布时间:** 2025-12-29

### GPT解析

### 总结

本文介绍了MIRAGE-VC，一个多视角检索增强生成框架，用于预测初创公司成功，解决了风险投资预测中的路径爆炸和异质证据融合挑战。

### 背景

大多数风险投资失败，只有少数带来超额回报；准确预测初创公司成功需要综合复杂的关系证据；传统机器学习和图神经网络缺乏推理能力；大语言模型具有推理能力但与图数据存在模态不匹配；现有图-LLM方法专注于图内任务，而风险投资预测是图外任务。

### 目的

开发一个能够通过显式推理形成连贯、可解释的投资论点的框架，用于准确预测初创公司成功。

### 方法

MIRAGE-VC框架包含信息增益驱动的路径检索器，迭代选择高价值邻居，将投资网络压缩为紧凑链；多智能体架构通过基于公司属性的可学习门控机制整合三种证据流。

### 主要发现

在严格的防泄漏控制下，MIRAGE-VC实现了+5.0%的F1和+16.6%的PrecisionAt5；该方法为其他图外预测任务（如推荐和风险评估）提供了见解。

### 结论

MIRAGE-VC有效解决了风险投资预测中的路径爆炸和异质证据融合挑战，显著提高了预测性能。

### 翻译

大多数风险投资（VC）投资都会失败，而只有少数能带来超额回报。准确预测初创公司成功需要综合复杂的关系证据，包括公司披露、投资者记录和投资网络结构，通过显式推理形成连贯、可解释的投资论点。传统机器学习和图神经网络都缺乏这种推理能力。大语言模型（LLMs）提供强大的推理能力，但与图数据存在模态不匹配问题。最近的图-LLM方法针对图内任务，其中答案存在于图内，而VC预测是图外的：目标存在于网络之外。核心挑战是选择能最大化预测者在外部目标上性能的图路径，同时支持逐步推理。我们提出了MIRAGE-VC，一个多视角检索增强生成框架，解决了两个障碍：路径爆炸（数千条候选路径使LLM上下文过载）和异质证据融合（不同初创公司需要不同的分析重点）。我们的信息增益驱动的路径检索器迭代选择高价值邻居，将投资网络压缩为紧凑链以进行显式推理。多智能体架构通过基于公司属性的可学习门控机制整合三种证据流。在严格的防泄漏控制下，MIRAGE-VC实现了+5.0%的F1和+16.6%的PrecisionAt5，并为其他图外预测任务（如推荐和风险评估）提供了见解。代码：https://anonymous.4open.science/r/MIRAGE-VC-323F。


### 论文摘要

Most venture capital (VC) investments fail, while a few deliver outsized returns. Accurately predicting startup success requires synthesizing complex relational evidence, including company disclosures, investor track records, and investment network structures, through explicit reasoning to form coherent, interpretable investment theses. Traditional machine learning and graph neural networks both lack this reasoning capability. Large language models (LLMs) offer strong reasoning but face a modality mismatch with graphs. Recent graph-LLM methods target in-graph tasks where answers lie within the graph, whereas VC prediction is off-graph: the target exists outside the network. The core challenge is selecting graph paths that maximize predictor performance on an external objective while enabling step-by-step reasoning. We present MIRAGE-VC, a multi-perspective retrieval-augmented generation framework that addresses two obstacles: path explosion (thousands of candidate paths overwhelm LLM context) and heterogeneous evidence fusion (different startups need different analytical emphasis). Our information-gain-driven path retriever iteratively selects high-value neighbors, distilling investment networks into compact chains for explicit reasoning. A multi-agent architecture integrates three evidence streams via a learnable gating mechanism based on company attributes. Under strict anti-leakage controls, MIRAGE-VC achieves +5.0% F1 and +16.6% PrecisionAt5, and sheds light on other off-graph prediction tasks such as recommendation and risk assessment. Code: https://anonymous.4open.science/r/MIRAGE-VC-323F.

---

## 88. 论文ID: 2512.23406v1

**错误:** 无法读取论文信息 - [Errno 2] No such file or directory: 'storage/result/2512.23406v1.json'

---

## 89. Domain matters: Towards domain-informed evaluation for link prediction

**论文链接:** [http://arxiv.org/abs/2512.23371v1](http://arxiv.org/abs/2512.23371v1)

**作者:** Yilin Bi, Junhao Bian, Shuyan Wan, Shuaijia Wang, Tao Zhou

**发布时间:** 2025-12-29

### GPT解析

### 总结

该研究系统评估了12种主流链接预测算法在740个跨七个领域的真实世界网络上的性能，发现算法性能在不同领域间一致性低，而在领域内一致性高，并提出了'Winner Score'指标来识别各领域最优算法。

### 背景

链接预测是复杂网络分析的基础任务，在社会推荐、药物靶点发现和知识图谱补全等领域有广泛应用。然而，现有评估通常只在有限网络上进行，假设跨领域性能排名一致，忽视了不同领域生成机制和语义背景的显著差异。

### 目的

系统评估多种链接预测算法在跨领域网络上的性能，揭示算法在特定领域的表现，并探讨领域属性作为影响算法性能的关键因素。

### 方法

评估12种主流链接预测算法在740个跨越七个领域的真实世界网络上的性能，使用主成分分析(PCA)分析算法排名形成的响应向量，观察它们在低维空间中按领域聚类的情况。

### 主要发现

1) 跨领域算法排名一致性明显低于领域内一致性；2) PCA分析显示算法排名响应向量按领域明显聚类，证实领域属性是影响算法性能的关键因素；3) 提出了'Winner Score'指标确定了各领域最优算法；4) 特定领域最优算法在其他领域往往表现不佳。

### 结论

不存在普遍最优的链接预测算法，选择算法时应考虑其机制与网络结构的匹配性，应根据特定领域选择最适合的算法。

### 翻译

链接预测是复杂网络分析中的一个基础任务，在社会推荐、药物靶点发现和知识图谱补全等关键场景中有广泛应用。然而，现有算法评估通常只在有限数量的网络上进行实验，并假设跨领域的性能排名是一致的。尽管不同领域的生成机制和语义背景存在显著差异，但以往研究往往仅通过对跨领域网络的简单平均来错误地强调'普遍最优'的算法。本文系统地评估了12种主流链接预测算法在740个跨越七个领域的真实世界网络上的性能。我们提供了大量实证证据，阐明算法在特定领域的性能。这些发现揭示了跨领域算法排名的一致性程度明显较低，这一现象与单个领域内观察到的高度一致性形成鲜明对比。主成分分析显示，由12种算法排名形成的响应向量在低维空间中按领域明显聚类，从而证实领域属性是影响算法性能的关键因素。我们提出了一个称为'Winner Score'的指标，可以识别每个领域中的优越算法：社会网络是非负矩阵分解，经济学是基于邻域重叠感知的图神经网络，化学是图卷积网络，生物学是基于L3的资源分配。然而，这些特定领域表现最优的算法往往在其他领域表现不佳。这一发现强调了算法机制与网络结构匹配的重要性。


### 论文摘要

Link prediction, a foundational task in complex network analysis, has extensive applications in critical scenarios such as social recommendation, drug target discovery, and knowledge graph completion. However, existing evaluations of algorithmic often rely on experiments conducted on a limited number of networks, assuming consistent performance rankings across domains. Despite the significant disparities in generative mechanisms and semantic contexts, previous studies often improperly highlight ``universally optimal" algorithms based solely on naive average over networks across domains. This paper systematically evaluates 12 mainstream link prediction algorithms across 740 real-world networks spanning seven domains. We present substantial empirical evidence elucidating the performance of algorithms in specific domains. This findings reveal a notably low degree of consistency in inter-domain algorithm rankings, a phenomenon that stands in stark contrast to the high degree of consistency observed within individual domains. Principal Component Analysis shows that response vectors formed by the rankings of the 12 algorithms cluster distinctly by domain in low-dimensional space, thus confirming domain attributes as a pivotal factor affecting algorithm performance. We propose a metric called Winner Score that could identify the superior algorithm in each domain: Non-Negative Matrix Factorization for social networks, Neighborhood Overlap-aware Graph Neural Networks for economics, Graph Convolutional Networks for chemistry, and L3-based Resource Allocation for biology. However, these domain-specific top-performing algorithms tend to exhibit suboptimal performance in other domains. This finding underscores the importance of aligning an algorithm's mechanism with the network structure.

---

## 90. Graph Neural Networks with Transformer Fusion of Brain Connectivity Dynamics and Tabular Data for Forecasting Future Tobacco Use

**论文链接:** [http://arxiv.org/abs/2512.23137v1](http://arxiv.org/abs/2512.23137v1)

**作者:** Runzhi Zhou, Xi Luo

**发布时间:** 2025-12-29

**备注:** 22 pages, 4 figures

### GPT解析

### 总结

研究提出了GNN-TF模型，解决了整合非欧几里得脑成像数据与欧几里得表格数据并预测未来结果的挑战，在预测未来烟草使用方面表现出色，是临床结果预测的有价值工具。

### 背景

整合非欧几里得脑成像数据与欧几里得表格数据（如临床和人口统计信息）是医学影像分析的重大挑战，尤其在预测未来结果方面。虽然机器学习和深度学习技术已成功应用于横断面分类和预测任务，但在纵向影像研究中有效预测结果仍然具有挑战性。

### 目的

开发一种能够同时整合表格数据和动态脑连接数据的时间感知模型，以解决医学影像分析中的数据整合和预测挑战。

### 方法

提出了一种时间感知图神经网络模型与transformer融合（GNN-TF），该模型灵活整合表格数据和动态脑连接数据，利用变量时间顺序在一致框架内进行分析。研究使用了NCANDA的纵向静息态功能磁共振成像数据集，并与多种成熟的机器学习和深度学习模型进行了比较分析。

### 主要发现

GNN-TF超越了现有的最先进方法，在预测未来烟草使用方面提供了更优的预测准确性。端到端的时间感知transformer融合结构成功整合了多种数据模态并有效利用了时间动态。

### 结论

GNN-TF模型是专注于临床结果预测的功能性脑影像研究的一个有价值的分析工具，能够有效整合多源数据并捕捉时间动态信息。

### 翻译

将非欧几里得脑成像数据与欧几里得表格数据（如临床和人口统计信息）进行整合，对医学影像分析构成了重大挑战，特别是在预测未来结果方面。虽然机器学习和深度学习技术已成功应用于横断面分类和预测任务，但在纵向影像研究中有效预测结果仍然具有挑战性。为应对这一挑战，我们引入了一种时间感知图神经网络模型与transformer融合（GNN-TF）。该模型灵活地整合表格数据和动态脑连接数据，利用这些变量在一致框架内的时间顺序。通过整合来自国家青少年酒精和神经发育联盟（NCANDA）纵向静息态功能磁共振成像数据集的非欧几里得和欧几里得信息源，GNN-TF能够进行全面的分析，捕捉纵向影像数据的关键方面。与多种成熟的机器学习和深度学习模型的比较分析表明，GNN-TF超越了这些最先进的方法，在预测未来烟草使用方面提供了更优的预测准确性。所提出的GNN-TF模型的端到端、时间感知transformer融合结构成功整合了多种数据模态并利用了时间动态，使其成为专注于临床结果预测的功能性脑影像研究的一个有价值的分析工具。


### 论文摘要

Integrating non-Euclidean brain imaging data with Euclidean tabular data, such as clinical and demographic information, poses a substantial challenge for medical imaging analysis, particularly in forecasting future outcomes. While machine learning and deep learning techniques have been applied successfully to cross-sectional classification and prediction tasks, effectively forecasting outcomes in longitudinal imaging studies remains challenging. To address this challenge, we introduce a time-aware graph neural network model with transformer fusion (GNN-TF). This model flexibly integrates both tabular data and dynamic brain connectivity data, leveraging the temporal order of these variables within a coherent framework. By incorporating non-Euclidean and Euclidean sources of information from a longitudinal resting-state fMRI dataset from the National Consortium on Alcohol and Neurodevelopment in Adolescence (NCANDA), the GNN-TF enables a comprehensive analysis that captures critical aspects of longitudinal imaging data. Comparative analyses against a variety of established machine learning and deep learning models demonstrate that GNN-TF outperforms these state-of-the-art methods, delivering superior predictive accuracy for predicting future tobacco usage. The end-to-end, time-aware transformer fusion structure of the proposed GNN-TF model successfully integrates multiple data modalities and leverages temporal dynamics, making it a valuable analytic tool for functional brain imaging studies focused on clinical outcome prediction.

---

## 91. Debugging Tabular Log as Dynamic Graphs

**论文链接:** [http://arxiv.org/abs/2512.22903v1](http://arxiv.org/abs/2512.22903v1)

**作者:** Chumeng Liang, Zhanyang Jin, Zahaib Akhtar, Mona Pereira, Haofei Yu, Jiaxuan You

**发布时间:** 2025-12-28

### GPT解析

### 总结

本文提出了GraphLogDebugger框架，基于动态图技术调试表格日志，通过简单的动态图神经网络实现比大型语言模型更好的性能，同时提高了灵活性和可扩展性。

### 背景

表格日志记录现实世界系统的对象和事件并报告更新，可用于检测系统不一致性。然而，现有方法过度依赖大型语言模型和其他重型模型，导致灵活性和可扩展性有限。

### 目的

开发一种不依赖大型语言模型、具有更好灵活性和可扩展性的表格日志调试方法。

### 方法

提出GraphLogDebugger框架，通过构建对象和事件的异构节点并连接节点间的边，将表格日志背后的系统建模为演化的动态图，并使用简单的动态图神经网络进行调试。

### 主要发现

基于动态图建模的简单动态图神经网络在调试表格日志方面性能优于大型语言模型，这一发现在真实世界日志数据集上得到验证。

### 结论

GraphLogDebugger框架能够有效调试表格日志，相比依赖大型语言模型的方法具有更好的性能、灵活性和可扩展性。

### 翻译

表格日志记录现实世界系统中的对象和事件，并报告它们的更新以反映系统的变化，人们可以通过调试相应的日志条目有效地检测现实世界的不一致性。然而，最近在处理文本丰富的表格日志数据方面的进展过度依赖大型语言模型（LLMs）和其他重型模型，因此存在灵活性和可扩展性有限的问题。本文提出了一个名为GraphLogDebugger的新框架，基于动态图来调试表格日志。通过为对象和事件构建异构节点并连接节点间的边，该框架将表格日志背后的系统恢复为一个演化的动态图。借助动态图建模，一个简单的动态图神经网络（GNN）足以在调试表格日志方面超越大型语言模型，这一发现已在计算机系统和学术论文的真实世界日志数据集上的实验结果中得到验证。


### 论文摘要

Tabular log abstracts objects and events in the real-world system and reports their updates to reflect the change of the system, where one can detect real-world inconsistencies efficiently by debugging corresponding log entries. However, recent advances in processing text-enriched tabular log data overly depend on large language models (LLMs) and other heavy-load models, thus suffering from limited flexibility and scalability. This paper proposes a new framework, GraphLogDebugger, to debug tabular log based on dynamic graphs. By constructing heterogeneous nodes for objects and events and connecting node-wise edges, the framework recovers the system behind the tabular log as an evolving dynamic graph. With the help of our dynamic graph modeling, a simple dynamic Graph Neural Network (GNN) is representative enough to outperform LLMs in debugging tabular log, which is validated by experimental results on real-world log datasets of computer systems and academic papers.

---

## 92. GRExplainer: A Universal Explanation Method for Temporal Graph Neural Networks

**论文链接:** [http://arxiv.org/abs/2512.22772v1](http://arxiv.org/abs/2512.22772v1)

**作者:** Xuyan Li, Jie Wang, Zheng Yan

**发布时间:** 2025-12-28

### GPT解析

### 总结

论文提出了一种名为GRExplainer的新型时序图神经网络(TGNN)解释方法，该方法具有通用性、高效性和用户友好性三大特点，解决了现有TGNN解释方法的局限性。

### 背景

动态图被广泛用于表示不断演化的现实世界网络。时序图神经网络(TGNNs)已成为处理这类图的有力工具，但其缺乏透明度和可解释性限制了实际应用。TGNN可解释性研究面临三个关键问题：当前方法针对特定TGNN类型定制，限制了通用性；计算成本高，不适合大规模网络；忽略了解释的结构连接性，需要先验知识，降低了用户友好性。

### 目的

提出GRExplainer，首个通用、高效且用户友好的TGNN解释方法，以解决现有TGNN解释方法的局限性。

### 方法

GRExplainer提取节点序列作为统一的特征表示，使其独立于特定输入格式，适用于基于快照和基于事件的TGNNs。通过利用广度优先搜索和时间信息构建输入节点序列，减少了冗余计算并提高了效率。设计了一个基于循环神经网络(RNNs)的生成模型，实现了自动和连续的解释生成。

### 主要发现

在六个真实世界数据集上对三种目标TGNNs进行的实验表明，GRExplainer在通用性、效率和用户友好性方面优于现有的基线方法。

### 结论

GRExplainer成功解决了现有TGNN解释方法的局限性，提供了更通用、高效和用户友好的解释方案，有助于推动TGNN在实际应用中的采用。

### 翻译

动态图被广泛用于表示不断演化的现实世界网络。时序图神经网络(TGNNs)已成为处理这类图的有力工具，但其缺乏透明度和可解释性限制了其实际应用。关于TGNN可解释性的研究仍处于早期阶段，并面临几个关键问题：(i)当前方法针对特定类型的TGNN定制，限制了通用性；(ii)它们计算成本高，不适合大规模网络；(iii)它们通常忽略了解释的结构连接性，并且需要先验知识，降低了用户友好性。为解决这些问题，我们提出了GRExplainer，这是首个通用、高效且用户友好的TGNN解释方法。GRExplainer提取节点序列作为统一的特征表示，使其独立于特定输入格式，因此适用于基于快照和基于事件的TGNNs（TGNN的主要类型）。通过利用广度优先搜索和时间信息构建输入节点序列，GRExplainer减少了冗余计算并提高了效率。为了增强用户友好性，我们设计了一个基于循环神经网络(RNNs)的生成模型，实现了自动和连续的解释生成。在六个真实世界数据集上对三种目标TGNNs进行的实验表明，GRExplainer在通用性、效率和用户友好性方面优于现有的基线方法。


### 论文摘要

Dynamic graphs are widely used to represent evolving real-world networks. Temporal Graph Neural Networks (TGNNs) have emerged as a powerful tool for processing such graphs, but the lack of transparency and explainability limits their practical adoption. Research on TGNN explainability is still in its early stages and faces several key issues: (i) Current methods are tailored to specific TGNN types, restricting generality. (ii) They suffer from high computational costs, making them unsuitable for large-scale networks. (iii) They often overlook the structural connectivity of explanations and require prior knowledge, reducing user-friendliness. To address these issues, we propose GRExplainer, the first universal, efficient, and user-friendly explanation method for TGNNs. GRExplainer extracts node sequences as a unified feature representation, making it independent of specific input formats and thus applicable to both snapshot-based and event-based TGNNs (the major types of TGNNs). By utilizing breadth-first search and temporal information to construct input node sequences, GRExplainer reduces redundant computation and improves efficiency. To enhance user-friendliness, we design a generative model based on Recurrent Neural Networks (RNNs), enabling automated and continuous explanation generation. Experiments on six real-world datasets with three target TGNNs show that GRExplainer outperforms existing baseline methods in generality, efficiency, and user-friendliness.

---

## 93. LLM Agents as VC investors: Predicting Startup Success via RolePlay-Based Collective Simulation

**论文链接:** [http://arxiv.org/abs/2512.22608v1](http://arxiv.org/abs/2512.22608v1)

**作者:** Zhongyang Liu, Haoyu Pei, Xiangyi Xiao, Xiaocong Du, Yihui Li, Suting Hong, Kunpeng Zhang, Haipeng Zhang

**发布时间:** 2025-12-27

### GPT解析

### 总结

本文提出了SimVC-CAS，一个模拟风险投资决策作为多智能体交互过程的集体智能系统，将初创企业融资预测重新定义为群体决策任务，显著提高了预测准确性并提供了可解释的推理过程。

### 背景

初创企业具有高价值和高失败率的特点，预测其成功已成为跨学科研究的关键挑战。现有方法通常从单一决策者的角度建模成功预测，忽略了在风险投资决策中占主导地位的投资者群体的集体动态。

### 目的

提出一个集体智能系统，将风险投资决策建模为多智能体交互过程，重新定义初创企业融资预测为群体决策任务，同时捕捉企业基本面和潜在投资者网络的行为动态。

### 方法

设计角色扮演智能体和基于图神经网络的监督交互模块，每个智能体代表具有独特特质和偏好的投资者，通过图结构的共同投资网络实现异质评估和真实信息交换，使用PitchBook的真实世界数据并在严格的数据泄露控制下进行实验。

### 主要发现

SimVC-CAS显著提高了预测准确性，提供可解释的、多角度推理，在平均精度@10方面实现了约25%的相对改进。

### 结论

SimVC-CAS为其他复杂的群体决策场景提供了启示。

### 翻译

由于初创企业的高价值和高失败率，预测其成功已成为跨学科研究的关键挑战。现有方法通常从单一决策者的角度对成功预测进行建模，忽略了在现实世界中风险投资决策中占主导地位的投资者群体的集体动态。在本文中，我们提出了SimVC-CAS，一种新的集体智能系统，将风险投资决策模拟为多智能体交互过程。通过设计角色扮演智能体和基于GNN的监督交互模块，我们将初创企业融资预测重新定义为群体决策任务，同时捕捉企业基本面和潜在投资者网络的行为动态。每个智能体代表具有独特特质和偏好的投资者，通过图结构的共同投资网络实现异质评估和真实信息交换。使用PitchBook的真实世界数据并在严格的数据泄露控制下，我们证明SimVC-CAS显著提高了预测准确性，同时提供可解释的、多角度推理，例如，在平均精度@10方面实现了约25%的相对改进。SimVC-CAS还为其他复杂的群体决策场景提供了启示。


### 论文摘要

Due to the high value and high failure rate of startups, predicting their success has become a critical challenge across interdisciplinary research. Existing approaches typically model success prediction from the perspective of a single decision-maker, overlooking the collective dynamics of investor groups that dominate real-world venture capital (VC) decisions. In this paper, we propose SimVC-CAS, a novel collective agent system that simulates VC decision-making as a multi-agent interaction process. By designing role-playing agents and a GNN-based supervised interaction module, we reformulate startup financing prediction as a group decision-making task, capturing both enterprise fundamentals and the behavioral dynamics of potential investor networks. Each agent embodies an investor with unique traits and preferences, enabling heterogeneous evaluation and realistic information exchange through a graph-structured co-investment network. Using real-world data from PitchBook and under strict data leakage controls, we show that SimVC-CAS significantly improves predictive accuracy while providing interpretable, multiperspective reasoning, for example, approximately 25% relative improvement with respect to average precision@10. SimVC-CAS also sheds light on other complex group decision scenarios.

---

## 94. 论文ID: 2512.22428v1

**错误:** 无法读取论文信息 - [Errno 2] No such file or directory: 'storage/result/2512.22428v1.json'

---

## 95. BLISS: Bandit Layer Importance Sampling Strategy for Efficient Training of Graph Neural Networks

**论文链接:** [http://arxiv.org/abs/2512.22388v1](http://arxiv.org/abs/2512.22388v1)

**作者:** Omar Alsaqa, Linh Thi Hoang, Muhammed Fatih Balin

**发布时间:** 2025-12-26

**备注:** Accepted for 5th Muslims in ML Workshop co-located with NeurIPS 2025. OpenReview: https://openreview.net/forum?id=VaHubA7Pwv Code: https://github.com/linhthi/BLISS-GNN

### GPT解析

### 总结

BLISS是一种基于多臂老虎机的层重要性采样策略，通过动态选择最有信息量的节点来解决图神经网络在大图应用中的计算瓶颈问题。

### 背景

图神经网络(GNNs)是从图结构数据中学习的强大工具，但应用于大型图时受到计算成本的限制，需要为每个节点处理所有邻居，这造成了内存和计算瓶颈。

### 目的

开发一种能够动态选择最有信息量节点的方法，以提高GNN在大图应用中的计算效率。

### 方法

引入BLISS(Bandit Layer Importance Sampling Strategy)，使用多臂老虎机在每个层动态选择最有信息量的节点，平衡探索和利用以确保全面图覆盖，并能适应节点重要性的演变，可集成于GCN和GAT等模型。

### 主要发现

BLISS能够适应不同模型(GCN和GAT)的特定聚合机制，实验表明其保持或超过全批次训练的准确性。

### 结论

BLISS是一种有效的采样策略，能够在保持或提高准确性的同时，显著减少GNN在大图应用中的计算成本。

### 翻译

图神经网络(GNNs)是从图结构数据中学习的强大工具，但它们在大图上的应用受到计算成本的阻碍。需要为每个节点处理所有邻居，这造成了内存和计算瓶颈。为解决这一问题，我们引入了BLISS，一种基于老虎机的层重要性采样策略。它使用多臂老虎机在每个层动态选择最有信息量的节点，平衡探索和利用以确保全面覆盖图结构。与现有的静态采样方法不同，BLISS适应节点重要性的演变，从而实现更明智的节点选择和更好的性能。它通过与图卷积网络(GCN)和图注意力网络(GAT)集成展示了其多功能性，并根据它们的特定聚合机制调整其选择策略。实验表明，BLISS保持或超过了全批次训练的准确性。


### 论文摘要

Graph Neural Networks (GNNs) are powerful tools for learning from graph-structured data, but their application to large graphs is hindered by computational costs. The need to process every neighbor for each node creates memory and computational bottlenecks. To address this, we introduce BLISS, a Bandit Layer Importance Sampling Strategy. It uses multi-armed bandits to dynamically select the most informative nodes at each layer, balancing exploration and exploitation to ensure comprehensive graph coverage. Unlike existing static sampling methods, BLISS adapts to evolving node importance, leading to more informed node selection and improved performance. It demonstrates versatility by integrating with both Graph Convolutional Networks (GCNs) and Graph Attention Networks (GATs), adapting its selection policy to their specific aggregation mechanisms. Experiments show that BLISS maintains or exceeds the accuracy of full-batch training.

---

## 96. INSIGHT: Spatially resolved survival modelling from routine histology crosslinked with molecular profiling reveals prognostic epithelial-immune axes in stage II/III colorectal cancer

**论文链接:** [http://arxiv.org/abs/2512.22262v1](http://arxiv.org/abs/2512.22262v1)

**作者:** Piotr Keller, Mark Eastwood, Zedong Hu, Aimée Selten, Ruqayya Awan, Gertjan Rasschaert, Sara Verbandt, Vlad Popovici, Hubert Piessevaux, Hayley T Morris, Petros Tsantoulis, Thomas Alexander McKee, André D'Hoore, Cédric Schraepen, Xavier Sagaert, Gert De Hertogh, Sabine Tejpar, Fayyaz Minhas

**发布时间:** 2025-12-24

### GPT解析

### 总结

本文介绍了一种名为INSIGHT的图神经网络方法，能够直接从常规组织学图像中预测II/III期结直肠癌患者的生存情况，通过分析复杂的空间组织结构提取预后信息。

### 背景

常规组织学在II/III期结直肠癌中包含丰富的预后信息，这些信息大多嵌入在复杂的空间组织结构中。

### 目的

开发一种能够直接从常规组织学图像预测患者生存情况的图神经网络方法。

### 方法

研究人员提出了名为INSIGHT的图神经网络，在TCGA（n=342）和SURGEN（n=336）队列上进行训练和交叉验证，生成患者水平的空间解析风险评分。

### 主要发现

1) 大型独立验证显示INSIGHT预后性能优于pTNM分期（C指数0.68-0.69对比0.44-0.58）；2) INSIGHT空间风险图谱重现了经典预后组织病理学特征；3) 识别核实心度和圆形度作为定量风险相关因素；4) 整合多组学数据揭示上皮-免疫风险 manifold，包含上皮去分化、胎儿程序、髓源性基质状态和适应性免疫功能障碍。

### 结论

INSIGHT能从常规组织学图像中提取预后信息，揭示患者特异性上皮异质性、MSI-High肿瘤内分层及高风险的CDX2/HNF4A丢失和CEACAM5/6相关增殖程序，突出了协调的治疗脆弱性。

### 翻译

常规组织学在II/III期结直肠癌中包含丰富的预后信息，其中大部分嵌入在复杂的空间组织结构中。我们提出了INSIGHT，一种直接从常规组织学图像预测生存的图神经网络。在TCGA（n=342）和SURGEN（n=336）上进行训练和交叉验证后，INSIGHT生成患者水平的空间解析风险评分。大型独立验证显示其预后性能优于pTNM分期（C指数0.68-0.69对比0.44-0.58）。INSIGHT空间风险图谱重现了经典的预后组织病理学特征，并将核实心度和圆形性确定为定量风险相关因素。将空间风险与数据驱动的空间转录组学特征、空间蛋白质组学、RNA测序和单细胞参考数据整合，揭示了一个捕获上皮去分化、胎儿程序、髓源性基质状态和适应性免疫功能障碍的上皮-免疫风险 manifold。


### 论文摘要

Routine histology contains rich prognostic information in stage II/III colorectal cancer, much of which is embedded in complex spatial tissue organisation. We present INSIGHT, a graph neural network that predicts survival directly from routine histology images. Trained and cross-validated on TCGA (n=342) and SURGEN (n=336), INSIGHT produces patient-level spatially resolved risk scores. Large independent validation showed superior prognostic performance compared with pTNM staging (C-index 0.68-0.69 vs 0.44-0.58). INSIGHT spatial risk maps recapitulated canonical prognostic histopathology and identified nuclear solidity and circularity as quantitative risk correlates. Integrating spatial risk with data-driven spatial transcriptomic signatures, spatial proteomics, bulk RNA-seq, and single-cell references revealed an epithelium-immune risk manifold capturing epithelial dedifferentiation and fetal programs, myeloid-driven stromal states including $\mathrm{SPP1}^{+}$ macrophages and $\mathrm{LAMP3}^{+}$ dendritic cells, and adaptive immune dysfunction. This analysis exposed patient-specific epithelial heterogeneity, stratification within MSI-High tumours, and high-risk routes of CDX2/HNF4A loss and CEACAM5/6-associated proliferative programs, highlighting coordinated therapeutic vulnerabilities.

---

## 97. ReVEAL: GNN-Guided Reverse Engineering for Formal Verification of Optimized Multipliers

**论文链接:** [http://arxiv.org/abs/2512.22260v1](http://arxiv.org/abs/2512.22260v1)

**作者:** Chen Chen, Daniela Kaufmann, Chenhui Deng, Zhan Song, Hongce Zhang, Cunxi Yu

**发布时间:** 2025-12-24

**备注:** Accepted by TACAS 2026

### GPT解析

### 总结

ReVEAL是一种基于图学习的方法，用于乘法器架构的反向工程，以提高代数电路验证技术。

### 背景

传统基于规则的乘法器架构验证方法在处理大型优化乘法器时面临可扩展性和准确性挑战。

### 目的

提高代数电路验证技术的可扩展性和准确性，特别是针对大型优化乘法器。

### 方法

ReVEAL框架利用结构图特性和学习驱动的推理来大规模识别架构模式，能够稳健处理大型优化乘法器。

### 主要发现

ReVEAL在各种乘法器基准测试中展示了其适用性，并且与传统基于规则的方法相比，在可扩展性和准确性方面有所提高。

### 结论

ReVEAL方法能够与现有验证流程无缝集成，并支持下游代数证明策略，为乘法器架构验证提供了有效的解决方案。

### 翻译

我们提出了ReVEAL，一种基于图学习的方法，用于乘法器架构的反向工程，以提高代数电路验证技术。我们的框架利用结构图特性和学习驱动的推理来大规模识别架构模式，能够稳健处理大型优化乘法器。我们在各种乘法器基准测试中展示了其适用性，并表明与传统基于规则的方法相比，可扩展性和准确性有所提高。该方法能够与现有验证流程无缝集成，并支持下游代数证明策略。


### 论文摘要

We present ReVEAL, a graph-learning-based method for reverse engineering of multiplier architectures to improve algebraic circuit verification techniques. Our framework leverages structural graph features and learning-driven inference to identify architecture patterns at scale, enabling robust handling of large optimized multipliers. We demonstrate applicability across diverse multiplier benchmarks and show improvements in scalability and accuracy compared to traditional rule-based approaches. The method integrates smoothly with existing verification flows and supports downstream algebraic proof strategies.

---

## 98. Interpretable Perturbation Modeling Through Biomedical Knowledge Graphs

**论文链接:** [http://arxiv.org/abs/2512.22251v1](http://arxiv.org/abs/2512.22251v1)

**作者:** Pascal Passigan, Kevin zhu, Angelina Ning

**发布时间:** 2025-12-24

### GPT解析

### 总结

本研究开发了一个基于图注意力网络的框架，用于预测小分子药物对基因表达的影响。通过整合增强的生物医学知识图谱和LINCS L1000药物及细胞系数据，使用基础模型初始化节点嵌入，成功预测了药物-细胞对中978个标志基因的表达变化。

### 背景

理解小分子如何干扰基因表达对于揭示药物机制、预测脱靶效应和识别药物重新定位机会至关重要。现有的深度学习框架虽然已将多模态嵌入整合到生物医学知识图谱中，并通过图神经网络改进了表示，但这些模型主要应用于链接预测和二元药物-疾病关联等任务，而非基因干扰任务。

### 目的

解决现有方法未应用于基因干扰任务的问题，开发一个能够揭示药物转录组效应机制的框架。

### 方法

构建了一个合并的生物医学图谱，整合了PrimeKG++（包含丰富语义嵌入）和LINCS L1000药物及细胞系节点（使用MolFormerXL和BioBERT等基础模型初始化）。使用这个异构图，训练了一个带有下游预测头部的图注意力网络，学习给定药物-细胞对的标志基因表达变化。

### 主要发现

该框架在支架分割和随机分割下，优于MLP基线方法对不同表达基因的预测。消融实验进一步证明，生物医学知识图谱提供的边缘增强了干扰级别的预测。

### 结论

该框架为机制性药物建模提供了一条路径：超越二元药物-疾病关联任务，转向治疗干预的细粒度转录效应。

### 翻译

理解小分子如何干扰基因表达对于揭示药物机制、预测脱靶效应和识别药物重新定位机会至关重要。虽然先前的深度学习框架已将多模态嵌入整合到生物医学知识图谱中，并通过图神经网络消息传递范式进一步改进了这些表示，但这些模型已应用于链接预测和二元药物-疾病关联等任务，而非可能揭示更多机制转录组效应的基因干扰任务。为解决这一差距，我们构建了一个合并的生物医学图谱，整合了PrimeKG++（一个包含丰富语义嵌入的PrimeKG增强版本）和LINCS L1000药物和细胞系节点（使用MolFormerXL和BioBERT等基础模型初始化的多模态嵌入）。使用这个异构图，我们训练了一个带有下游预测头部的图注意力网络，学习给定药物-细胞对的978个标志基因的表达变化谱。我们的结果表明，在支架分割和随机分割下，我们的框架优于MLP基线方法对不同表达基因的预测。边缘打乱和节点特征随机化的消融实验进一步证明，生物医学知识图谱提供的边缘增强了干扰级别的预测。更广泛地说，我们的框架为机制性药物建模提供了一条路径：超越二元药物-疾病关联任务，转向治疗干预的细粒度转录效应。


### 论文摘要

Understanding how small molecules perturb gene expression is essential for uncovering drug mechanisms, predicting off-target effects, and identifying repurposing opportunities. While prior deep learning frameworks have integrated multimodal embeddings into biomedical knowledge graphs (BKGs) and further improved these representations through graph neural network message-passing paradigms, these models have been applied to tasks such as link prediction and binary drug-disease association, rather than the task of gene perturbation, which may unveil more about mechanistic transcriptomic effects. To address this gap, we construct a merged biomedical graph that integrates (i) PrimeKG++, an augmentation of PrimeKG containing semantically rich embeddings for nodes with (ii) LINCS L1000 drug and cell line nodes, initialized with multimodal embeddings from foundation models such as MolFormerXL and BioBERT. Using this heterogeneous graph, we train a graph attention network (GAT) with a downstream prediction head that learns the delta expression profile of over 978 landmark genes for a given drug-cell pair. Our results show that our framework outperforms MLP baselines for differentially expressed genes (DEG) -- which predict the delta expression given a concatenated embedding of drug features, target features, and baseline cell expression -- under the scaffold and random splits. Ablation experiments with edge shuffling and node feature randomization further demonstrate that the edges provided by biomedical KGs enhance perturbation-level prediction. More broadly, our framework provides a path toward mechanistic drug modeling: moving beyond binary drug-disease association tasks to granular transcriptional effects of therapeutic intervention.

---

## 99. 论文ID: 2512.23486v1

**错误:** 无法读取论文信息 - [Errno 2] No such file or directory: 'storage/result/2512.23486v1.json'

---

## 100. SpatialMosaic: A Multiview VLM Dataset for Partial Visibility

**论文链接:** [http://arxiv.org/abs/2512.23365v1](http://arxiv.org/abs/2512.23365v1)

**作者:** Kanghee Lee, Injae Lee, Minseok Kwak, Kwonyoung Ryu, Jungi Hong, Jaesik Park

**发布时间:** 2025-12-29

### GPT解析

### 总结

该论文提出了SpatialMosaic数据集和基准测试，以及SpatialMosaicVLM混合框架，用于增强多视图空间理解能力，特别是在部分可见性、遮挡和低重叠条件下的推理能力。

### 背景

多模态大语言模型(MLLMs)的快速发展为3D场景理解提供了新可能，但现有方法依赖预构建3D表示或重建管道，限制了可扩展性。虽然最近工作探索了从多视图图像直接学习空间推理，但真实环境中的挑战如部分可见性、遮挡和低重叠条件仍未得到充分研究。

### 目的

解决现有方法在真实环境中的局限性，特别是处理部分可见性、遮挡和低重叠条件等挑战性场景下的空间推理问题。

### 方法

1) 提出可扩展的多视图数据生成和注释管道，构建真实空间推理问答对，创建包含200万个QA对的SpatialMosaic数据集；2) 引入SpatialMosaic-Bench基准测试，包含6个任务共100万个QA对；3) 提出SpatialMosaicVLM混合框架，将3D重建模型作为几何编码器集成到VLMs中。

### 主要发现

大量实验证明提出的数据集和VQA任务有效增强了在挑战性多视图条件下的空间推理能力，验证了数据生成管道在构建真实和多样化问答对方面的有效性。

### 结论

通过SpatialMosaic数据集、SpatialMosaic-Bench基准测试和SpatialMosaicVLM框架，解决了多视图空间推理中的关键挑战，特别是在部分可见性、遮挡和低重叠条件下的推理问题。

### 翻译

多模态大语言模型(MLLMs)的快速发展已经释放了增强3D场景理解和空间推理的潜力。然而，现有方法通常依赖于预构建的3D表示或现成的重建管道，这限制了可扩展性和实际应用性。最近的工作探索直接从多视图图像中学习空间推理，使视觉语言模型(VLMs)能够在没有明确3D重建的情况下理解3D场景。然而，真实环境中经常出现的挑战，如部分可见性、遮挡和需要从碎片化视觉线索进行空间推理的低重叠条件，仍未得到充分探索。为了解决这些局限性，我们提出了一个可扩展的多视图数据生成和注释管道，构建真实的空间推理问答对，从而创建了SpatialMosaic，一个包含200万个问答对的全面指令调优数据集。我们进一步引入了SpatialMosaic-Bench，一个在真实和具有挑战性的场景下评估多视图空间推理的基准测试，包含6个任务共100万个问答对。此外，我们提出了SpatialMosaicVLM，一个混合框架，将3D重建模型作为几何编码器集成到VLMs中，以实现稳健的空间推理。大量实验证明，我们提出的数据集和VQA任务有效地增强了在具有挑战性的多视图条件下的空间推理能力，验证了我们的数据生成管道在构建真实和多样化问答对方面的有效性。代码和数据集即将发布。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决多视角视觉语言模型在部分可见性、遮挡和低重叠条件下的空间推理能力不足的问题。这个问题在现实中非常重要，因为真实世界场景往往存在物体部分可见、相互遮挡和视角重叠少的情况，而现有模型难以处理这些复杂情况，限制了它们在机器人导航、增强现实和自动驾驶等实际应用中的表现。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先识别现有方法在处理部分可见性、遮挡和低重叠条件时的不足，然后从数据和模型两个角度设计解决方案。在数据方面，他们利用ScanNet++数据集的高质量3D几何信息，设计了可扩展的数据生成和注释流程；在模型方面，他们借鉴了VGGT等3D重建模型作为几何编码器，并参考了LLaVA-Next-Video的多视角处理方法，创造性地将几何线索与视觉特征融合。他们的创新点在于引入了物体遮挡比例和视场遮挡比例的量化方法，以及稀疏多视角采样策略。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是通过创建专门针对挑战性多视角场景的数据集和基准测试，并设计融合3D几何信息的混合模型架构，提升VLM在现实世界复杂场景中的空间推理能力。整体流程分为三阶段：1)数据准备阶段，计算物体遮挡比例和视场遮挡比例，进行稀疏多视角采样和实例过滤；2)QA生成和关系计算阶段，选择任务相关对象，计算空间关系；3)QA模板和输出阶段，使用预定义模板生成问答对。模型方面，SpatialMosaicVLM并行处理视觉和几何编码器，通过交叉注意力融合特征，再输入语言模型生成答案。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)创建了SpatialMosaic数据集(200万QA对)和SpatialMosaic-Bench基准(100万QA对)，专注于部分可见性、遮挡和低重叠条件；2)提出了可扩展的数据生成和注释框架；3)设计了SpatialMosaicVLM混合模型架构，将3D重建模型作为几何编码器集成到VLM中；4)引入了物体遮挡比例和视场遮挡比例的量化方法。相比之前工作，SpatialMosaic更关注现实世界挑战，包含更大视角变化的图像，提供更多样化的任务类别，规模更大，并提供了更细粒度的评估框架。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'SpatialMosaic通过创建专门针对部分可见性、遮挡和低重叠条件的大规模多视角空间推理数据集和基准测试，并提出融合3D重建模型的混合视觉语言框架，显著提升了视觉语言模型在现实世界复杂场景中的空间推理能力。'}


### 论文摘要

The rapid progress of Multimodal Large Language Models (MLLMs) has unlocked the potential for enhanced 3D scene understanding and spatial reasoning. However, existing approaches often rely on pre-constructed 3D representations or off-the-shelf reconstruction pipelines, which constrain scalability and real-world applicability. A recent line of work explores learning spatial reasoning directly from multi-view images, enabling Vision-Language Models (VLMs) to understand 3D scenes without explicit 3D reconstructions. Nevertheless, key challenges that frequently arise in real-world environments, such as partial visibility, occlusion, and low-overlap conditions that require spatial reasoning from fragmented visual cues, remain under-explored. To address these limitations, we propose a scalable multi-view data generation and annotation pipeline that constructs realistic spatial reasoning QAs, resulting in SpatialMosaic, a comprehensive instruction-tuning dataset featuring 2M QA pairs. We further introduce SpatialMosaic-Bench, a challenging benchmark for evaluating multi-view spatial reasoning under realistic and challenging scenarios, consisting of 1M QA pairs across 6 tasks. In addition, we present SpatialMosaicVLM, a hybrid framework that integrates 3D reconstruction models as geometry encoders within VLMs for robust spatial reasoning. Extensive experiments demonstrate that our proposed dataset and VQA tasks effectively enhance spatial reasoning under challenging multi-view conditions, validating the effectiveness of our data generation pipeline in constructing realistic and diverse QA pairs. Code and dataset will be available soon.

---

## 101. 论文ID: 2512.23215v1

**错误:** 无法读取论文信息 - [Errno 2] No such file or directory: 'storage/result/2512.23215v1.json'

---

## 102. 论文ID: 2512.22939v1

**错误:** 无法读取论文信息 - [Errno 2] No such file or directory: 'storage/result/2512.22939v1.json'

---

## 103. Next Best View Selections for Semantic and Dynamic 3D Gaussian Splatting

**论文链接:** [http://arxiv.org/abs/2512.22771v1](http://arxiv.org/abs/2512.22771v1)

**作者:** Yiqian Li, Wen Jiang, Kostas Daniilidis

**发布时间:** 2025-12-28

### GPT解析

### 总结

本文提出了一种基于Fisher信息的主动学习算法，用于视图选择问题，能够同时处理语义推理和动态场景建模，提高渲染质量和语义分割性能。

### 背景

对于各种任务中的具身智能体来说，理解和语义与动态性非常重要，而且这些任务比静态场景理解任务有更多的数据冗余。

### 目的

将视图选择问题表述为主动学习问题，优先选择能为模型训练提供最大信息增益的帧。

### 方法

提出一种使用Fisher信息的主动学习算法，量化候选视图相对于语义高斯参数和变形网络的信息量。

### 主要发现

该方法能够同时处理语义推理和动态场景建模，为启发式或随机策略提供了有原则的替代方案。

### 结论

实验结果表明，该方法在提高渲染质量和语义分割性能方面始终优于基于随机选择和不确定性启发式的基线方法。

### 翻译

理解和语义与动态性对于各种任务中的具身智能体至关重要。这些任务比静态场景理解任务有更多的数据冗余。我们将视图选择问题表述为主动学习问题，目标是优先选择能为模型训练提供最大信息增益的帧。为此，我们提出了一种使用Fisher信息的主动学习算法，量化了候选视图相对于语义高斯参数和变形网络的信息量。这种表述使我们的方法能够同时处理语义推理和动态场景建模，为启发式或随机策略提供了有原则的替代方案。我们通过从多摄像头装置中选择信息丰富的帧，在大型静态图像和动态视频数据集上评估了我们的方法。实验结果表明，我们的方法在提高渲染质量和语义分割性能方面始终优于基于随机选择和不确定性启发式的基线方法。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决如何智能选择最佳视角来训练3D高斯飞溅模型的问题，特别是在包含语义信息和动态变化的场景中。这个问题很重要，因为3D高斯飞溅技术虽然能实现高质量重建和实时渲染，但训练过程需要大量数据，尤其是在大规模环境和动态场景中。有效的视角选择可以显著减少训练数据需求，同时保持高质量的渲染和语义理解，这对机器人、AR/VR和数字内容创作等领域至关重要。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者将视角选择问题转化为主动学习问题，目标是优先选择能为模型训练提供最大信息增益的帧。他们使用Fisher Information来量化候选视图的信息价值，同时考虑几何、语义和时间变化信息。作者借鉴了FisherRF的视角选择思想，Feature 3DGS的语义表示框架，以及4D-GS的动态建模方法。创新点在于将这些技术扩展到动态语义场景，并提出高效的计算方法来处理大规模场景的复杂性。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是将视角选择作为主动学习问题，使用Fisher Information量化候选视图的信息价值，同时综合评估几何、语义和时序信息。整体流程包括：构建语义3D高斯飞溅和动态高斯飞溅的基础框架；计算高斯参数和变形网络的Fisher Information；从候选视角中选择期望信息增益最高的视角加入训练；使用选定视角训练模型并定期重新计算Fisher Information以动态选择新视角。这种方法能在减少数据需求的同时保持高质量的渲染和语义理解。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：首个针对动态语义3DGS的Fisher Information驱动的NBV选择框架；将Fisher Information公式扩展到语义高斯参数和变形网络的高效计算方法；使用梯度外积迹估计变形网络Fisher Information的新方法。相比之前工作，本文不仅处理静态场景，还扩展到动态语义场景；同时考虑几何、语义和时序信息，而非仅关注几何；将NBV选择直接集成到3DGS骨干中，扩展了应用场景；并通过高效计算方法使大规模场景处理成为可能。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '这篇论文提出了一种基于Fisher Information的主动学习框架，能够智能选择最佳视角来训练动态语义3D高斯飞溅模型，在减少训练数据需求的同时显著提高了动态场景的渲染质量和语义理解能力。'}


### 论文摘要

Understanding semantics and dynamics has been crucial for embodied agents in various tasks. Both tasks have much more data redundancy than the static scene understanding task. We formulate the view selection problem as an active learning problem, where the goal is to prioritize frames that provide the greatest information gain for model training. To this end, we propose an active learning algorithm with Fisher Information that quantifies the informativeness of candidate views with respect to both semantic Gaussian parameters and deformation networks. This formulation allows our method to jointly handle semantic reasoning and dynamic scene modeling, providing a principled alternative to heuristic or random strategies. We evaluate our method on large-scale static images and dynamic video datasets by selecting informative frames from multi-camera setups. Experimental results demonstrate that our approach consistently improves rendering quality and semantic segmentation performance, outperforming baseline methods based on random selection and uncertainty-based heuristics.

---

## 104. VULCAN: Tool-Augmented Multi Agents for Iterative 3D Object Arrangement

**论文链接:** [http://arxiv.org/abs/2512.22351v1](http://arxiv.org/abs/2512.22351v1)

**作者:** Zhengfei Kuang, Rui Lin, Long Zhao, Gordon Wetzstein, Saining Xie, Sanghyun Woo

**发布时间:** 2025-12-26

### GPT解析

### 总结

这篇论文解决了多模态大语言模型在3D场景操作中的应用挑战，通过引入MCP-based API、增强3D场景理解和多智能体框架，显著提高了复杂物体排列任务的性能。

### 背景

多模态大语言模型在2D视觉语言任务中取得了显著进展，但在复杂3D场景操作中的应用仍处于探索阶段。

### 目的

解决多模态大语言模型在3D物体排列任务中的三个关键挑战：弱视觉基础、3D场景理解不足以及迭代更新的管理问题。

### 方法

作者提出了三个创新方法：(1)引入基于MCP的API，将交互从脆弱的原始代码操作转变为更强大的函数级更新；(2)通过专门的视觉工具增强MLLM的3D场景理解能力，包括分析场景状态、收集空间信息和验证行动结果；(3)提出具有规划、执行和验证角色的协作多智能体框架，以处理多步骤指令和从中间错误中恢复。

### 主要发现

在25个复杂的物体排列任务上，作者的方法显著优于现有基线，证明了其在处理复杂3D场景操作任务中的有效性。

### 结论

通过解决多模态大语言模型在3D场景操作中的关键挑战，作者成功地将这些模型扩展到了更复杂的3D应用领域，为未来3D场景操作的研究奠定了基础。

### 翻译

尽管多模态大语言模型在2D视觉语言任务中取得了显著进展，但它们在复杂3D场景操作中的应用仍处于探索阶段。在本文中，我们通过使用多模态大语言模型解决3D物体排列任务中的三个关键挑战，弥合了这一重要差距。首先，为了解决多模态大语言模型的弱视觉基础问题，它们难以将程序化编辑与精确的3D结果联系起来，我们引入了一个基于MCP的API。这使交互从脆弱的原始代码操作转变为更强大、更健壮的函数级更新。其次，我们通过一套专门的视觉工具增强多模态大语言模型的3D场景理解能力，以分析场景状态、收集空间信息和验证行动结果。这种感知反馈循环对于弥合基于语言的更新和精确的3D感知操作之间的差距至关重要。第三，为了管理迭代且容易出错的更新，我们提出了一个协作的多智能体框架，为规划、执行和验证指定了特定角色。这种分解使系统能够稳健地处理多步骤指令并从中间错误中恢复。我们在25个复杂的物体排列任务集上证明了我们方法的有效性，其显著优于现有基线。网站：vulcan-3d.github.io


### 论文摘要

Despite the remarkable progress of Multimodal Large Language Models (MLLMs) in 2D vision-language tasks, their application to complex 3D scene manipulation remains underexplored. In this paper, we bridge this critical gap by tackling three key challenges in 3D object arrangement task using MLLMs. First, to address the weak visual grounding of MLLMs, which struggle to link programmatic edits with precise 3D outcomes, we introduce an MCP-based API. This shifts the interaction from brittle raw code manipulation to more robust, function-level updates. Second, we augment the MLLM's 3D scene understanding with a suite of specialized visual tools to analyze scene state, gather spatial information, and validate action outcomes. This perceptual feedback loop is critical for closing the gap between language-based updates and precise 3D-aware manipulation. Third, to manage the iterative, error-prone updates, we propose a collaborative multi-agent framework with designated roles for planning, execution, and verification. This decomposition allows the system to robustly handle multi-step instructions and recover from intermediate errors. We demonstrate the effectiveness of our approach on a diverse set of 25 complex object arrangement tasks, where it significantly outperforms existing baselines. Website: vulcan-3d.github.io

---

## 105. GamiBench: Evaluating Spatial Reasoning and 2D-to-3D Planning Capabilities of MLLMs with Origami Folding Tasks

**论文链接:** [http://arxiv.org/abs/2512.22207v1](http://arxiv.org/abs/2512.22207v1)

**作者:** Ryan Spencer, Roey Yaari, Ritvik Vemavarapu, Joyce Yang, Steven Ngo, Utkarsh Sharma

**发布时间:** 2025-12-22

### GPT解析

### 总结

该研究引入了GamiBench基准测试，用于评估多模态大语言模型(MLLMs)的空间推理能力和2D到3D规划能力，通过折纸启发的折叠任务进行全面评估。

### 背景

多模态大语言模型在感知和指令遵循方面表现良好，但在空间推理能力上仍有欠缺。空间推理是人类智能的关键组成部分，但现有基准测试大多只关注静态图像或最终输出，未能考虑这一技能的序列性和视角依赖性。

### 目的

弥补现有基准测试的不足，专门评估MLLMs的空间推理能力和2D到3D规划能力，建立标准化评估框架。

### 方法

GamiBench包含186个常规和186个不可能的2D折痕图案及其对应的3D折叠形状，来自六个不同视角，涵盖三个视觉问答任务：预测3D折叠配置、区分有效视角和检测不可能图案。该基准评估整个推理过程，包括跨视图一致性、物理可行性和中间折叠步骤解释，并引入视角一致性(VC)和不可能折叠选择率(IFSR)作为新诊断指标。

### 主要发现

即使是领先的模型如GPT-5和Gemini-2.5-Pro在单步空间理解方面也存在困难，表明当前MLLMs在空间推理能力上有明显不足。

### 结论

GamiBench为评估MLLMs中的几何理解和空间推理建立了标准化框架，数据集和代码已公开。

### 翻译

多模态大语言模型在感知和指令遵循方面能力突出，但在空间推理能力上仍存在挑战：即跨视图和时间跟踪和操作对象的能力。空间推理是人类智能的关键组成部分，但大多数现有基准测试只关注静态图像或最终输出，未能体现这一技能的序列性和视角依赖性。为弥补这一差距，我们引入了GamiBench，这是一个通过折纸启发的折叠任务来评估MLLMs空间推理和2D到3D规划的基准测试。GamiBench包含186个常规和186个不可能的2D折痕图案及其对应的3D折叠形状，来自六个不同视角，涵盖三个视觉问答(VQA)任务：预测3D折叠配置、区分有效视角和检测不可能图案。与仅评估最终预测的先前基准不同，GamiBench全面评估整个推理过程——通过不可能折叠检测衡量跨视图一致性和物理可行性，以及对中间折叠步骤的解释。它进一步引入了新的诊断指标——视角一致性(VC)和不可能折叠选择率(IFSR)——来衡量模型处理不同复杂度折叠的能力。我们的实验表明，即使是GPT-5和Gemini-2.5-Pro等领先模型在单步空间理解方面也存在困难。这些贡献为评估MLLMs中的几何理解和空间推理建立了标准化框架。数据集和代码：https://github.com/stvngo/GamiBench。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决多模态大语言模型（MLLMs）在空间推理能力方面的评估问题。空间推理是人类智能的关键组成部分，涉及跨多个视图和时间跟踪和操作对象的能力，对人工智能系统与物理世界互动（如机器人技术、自动驾驶、家具组装等）至关重要。现有基准测试大多只关注静态图像或最终输出，未能考虑这种技能的顺序性和视角依赖性，因此需要一个能全面评估模型空间推理能力的框架。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者从日本传统折纸艺术中获取灵感，因为折纸需要将2D纸张转换为3D结构，涉及数十个中间离散折叠步骤，为测试顺序空间推理能力提供了理想基础。他们设计了一个包含186个常规和186个不可能的2D折痕图案的数据集，每个图案配对相应的3D折叠形状，从六个不同视角捕获。作者借鉴了现有的在线工具如Oriedita和Origami Simulator来创建和验证折痕图案，并参考了折纸的数学原理（如Kawasaki定理、Maekawa定理和Huzita-Hatori公理）来区分物理可行和不可行的折叠。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是通过折纸启发的任务来评估MLLMs的空间推理和2D到3D规划能力，不仅评估最终预测，还全面评估模型的整个推理过程，包括跨视图一致性、物理可行性和中间折叠步骤的解释。整体流程包括：1)构建数据集（收集折纸实例，使用工具验证可行性和生成3D渲染）；2)定义任务（单步空间理解和多步空间推理）；3)生成问题-答案对（构建多选答案库并确保质量）；4)使用三个指标评估（准确性、不可能折叠选择率和视角一致性）。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)多视图、顺序空间基准测试，要求在六个视图和时间上保持一致性；2)新的评估轴（视角一致性和不可能折叠选择率），诊断标准准确性指标未发现的故障模式；3)折纸启发的任务套件，测试MLLMs的几何变换规划能力；4)复杂度控制，分析空间规划的规模效应。相比之前工作，GamiBench直接评估2D到3D变换、多视图空间一致性和物理可行性，而不仅仅是静态理解或最终状态准确性，首次全面揭示了当前MLLMs在空间推理方面的显著局限性。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'GamiBench通过折纸启发的多视图评估框架，首次全面揭示了当前多模态大语言模型在空间推理和2D到3D规划方面的显著局限性，为未来几何理解和空间推理能力的改进提供了标准化的评估基础。'}


### 论文摘要

Multimodal large language models (MLLMs) are proficient in perception and instruction-following, but they still struggle with spatial reasoning: the ability to mentally track and manipulate objects across multiple views and over time. Spatial reasoning is a key component of human intelligence, but most existing benchmarks focus on static images or final outputs, failing to account for the sequential and viewpoint-dependent nature of this skill. To close this gap, we introduce GamiBench, a benchmark designed to evaluate spatial reasoning and 2D-to-3D planning in MLLMs through origami-inspired folding tasks. GamiBench includes 186 regular and 186 impossible 2D crease patterns paired with their corresponding 3D folded shapes, produced from six distinct viewpoints across three visual question-answering (VQA) tasks: predicting 3D fold configurations, distinguishing valid viewpoints, and detecting impossible patterns. Unlike previous benchmarks that assess only final predictions, GamiBench holistically evaluates the entire reasoning process--measuring cross-view consistency, physical feasibility through impossible-fold detection, and interpretation of intermediate folding steps. It further introduces new diagnostic metrics--viewpoint consistency (VC) and impossible fold selection rate (IFSR)--to measure how well models handle folds of varying complexity. Our experiments show that even leading models such as GPT-5 and Gemini-2.5-Pro struggle on single-step spatial understanding. These contributions establish a standardized framework for evaluating geometric understanding and spatial reasoning in MLLMs. Dataset and code: https://github.com/stvngo/GamiBench.

---

