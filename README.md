# 今日论文推荐 - 2025-11-26

共 265 篇论文

---

## 1. RubricRL: Simple Generalizable Rewards for Text-to-Image Generation

**论文链接:** [http://arxiv.org/abs/2511.20651v1](http://arxiv.org/abs/2511.20651v1)

**作者:** Xuelu Feng, Yunsheng Li, Ziyu Wan, Zixuan Gao, Junsong Yuan, Dongdong Chen, Chunming Qiao

**发布时间:** 2025-11-25

### GPT解析

### 总结

RubricRL是一种基于评分标准的奖励设计框架，用于强化学习对齐文本到图像生成模型与人类偏好，提供更高的可解释性、可组合性和用户控制。

### 背景

强化学习已成为将文本到图像生成模型与人类偏好对齐的有前景方法，但设计有效且可解释的奖励是一个关键挑战。

### 目的

解决现有方法依赖固定权重的复合指标或单个标量奖励导致的可解释性和灵活性限制问题。

### 方法

RubricRL为每个提示动态构建结构化评分标准，包含可分解的细粒度视觉标准检查清单，由多模态评估器独立评估，并通过提示自适应加权机制强调最相关维度。

### 主要发现

在自回归文本到图像模型上的实验表明，RubricRL提高了提示忠实度、视觉细节和泛化能力。

### 结论

RubricRL为策略优化提供可解释和模块化的监督信号，使用户能够直接调整奖励或惩罚的方面，并为跨文本到图像架构的可解释RL对齐提供了灵活且可扩展的基础。

### 翻译

强化学习(RL)最近已成为一种有前景的方法，用于将文本到图像生成模型与人类偏好对齐。然而，关键挑战在于设计有效且可解释的奖励。现有方法通常依赖具有固定权重的复合指标(如CLIP、OCR和真实感分数)或从人类偏好模型中提炼的单个标量奖励，这限制了可解释性和灵活性。我们提出了RubricRL，一种基于评分标准的简单通用奖励设计框架，提供更高的可解释性、可组合性和用户控制。RubricRL不使用黑盒标量信号，而是为每个提示动态构建结构化评分标准——针对输入文本定制的可分解细粒度视觉标准检查清单，如对象正确性、属性准确性、OCR保真度和真实感。每个标准由多模态评估器(如o4-mini)独立评估，提示自适应加权机制强调最相关的维度。这种设计不仅为策略优化(如GRPO或PPO)产生可解释和模块化的监督信号，还使用户能够直接调整要奖励或惩罚的方面。在自回归文本到图像模型上的实验表明，RubricRL提高了提示忠实度、视觉细节和泛化能力，同时为跨文本到图像架构的可解释RL对齐提供了灵活且可扩展的基础。


### 论文摘要

Reinforcement learning (RL) has recently emerged as a promising approach for aligning text-to-image generative models with human preferences. A key challenge, however, lies in designing effective and interpretable rewards. Existing methods often rely on either composite metrics (e.g., CLIP, OCR, and realism scores) with fixed weights or a single scalar reward distilled from human preference models, which can limit interpretability and flexibility. We propose RubricRL, a simple and general framework for rubric-based reward design that offers greater interpretability, composability, and user control. Instead of using a black-box scalar signal, RubricRL dynamically constructs a structured rubric for each prompt--a decomposable checklist of fine-grained visual criteria such as object correctness, attribute accuracy, OCR fidelity, and realism--tailored to the input text. Each criterion is independently evaluated by a multimodal judge (e.g., o4-mini), and a prompt-adaptive weighting mechanism emphasizes the most relevant dimensions. This design not only produces interpretable and modular supervision signals for policy optimization (e.g., GRPO or PPO), but also enables users to directly adjust which aspects to reward or penalize. Experiments with an autoregressive text-to-image model demonstrate that RubricRL improves prompt faithfulness, visual detail, and generalizability, while offering a flexible and extensible foundation for interpretable RL alignment across text-to-image architectures.

---

## 2. MedROV: Towards Real-Time Open-Vocabulary Detection Across Diverse Medical Imaging Modalities

**论文链接:** [http://arxiv.org/abs/2511.20650v1](http://arxiv.org/abs/2511.20650v1)

**作者:** Tooba Tehreem Sheikh, Jean Lahoud, Rao Muhammad Anwer, Fahad Shahbaz Khan, Salman Khan, Hisham Cholakkal

**发布时间:** 2025-11-25

### GPT解析

### 总结

MedROV是首个用于医学影像的实时开放词汇检测模型，通过大规模数据集、伪标记策略和基础模型知识整合，有效解决了医学影像中检测新标签对象的挑战。

### 背景

传统医学影像目标检测模型在封闭集范式下运行，限制了检测新标签对象的能力。开放词汇目标检测在医学影像领域因数据集稀缺和文本-图像对齐较弱而研究不足。

### 目的

填补医学影像开放词汇检测的研究空白，开发能够检测已知和新结构的实时检测模型。

### 方法

创建包含60万个检测样本跨越九种成像模态的Omnis数据集；引入伪标记策略处理多源数据集的缺失注释；整合大型预训练基础模型知识；利用对比学习和跨模态表示提升检测能力。

### 主要发现

MedROV比之前最先进的基础模型平均绝对改进40 mAP50，超越封闭集检测器3 mAP50以上，运行速度达70 FPS，在医学检测领域树立新基准。

### 结论

MedROV成功实现了医学影像的开放词汇实时检测，其源代码、数据集和训练模型已在GitHub平台公开。

### 翻译

传统医学影像目标检测模型在封闭集范式下运行，限制了它们检测新标签对象的能力。开放词汇目标检测解决了这一局限性，但由于数据集稀缺和文本-图像对齐较弱，在医学影像领域研究不足。为了填补这一空白，我们引入了MedROV，这是首个用于医学影像的实时开放词汇检测模型。为实现开放词汇学习，我们整理了一个名为Omnis的大规模数据集，包含跨越九种成像模态的60万个检测样本，并引入了一种伪标记策略来处理来自多源数据集的缺失注释。此外，我们通过整合来自大型预训练基础模型的知识来增强泛化能力。通过利用对比学习和跨模态表示，MedROV能够有效检测已知和新结构。实验结果表明，MedROV优于之前医学图像检测的最先进基础模型，平均绝对改进40 mAP50，并超过封闭集检测器3 mAP50以上，同时以70 FPS的速度运行，在医学检测领域树立了新基准。我们的源代码、数据集和训练模型可在https://github.com/toobatehreem/MedROV获取。


### 论文摘要

Traditional object detection models in medical imaging operate within a closed-set paradigm, limiting their ability to detect objects of novel labels. Open-vocabulary object detection (OVOD) addresses this limitation but remains underexplored in medical imaging due to dataset scarcity and weak text-image alignment. To bridge this gap, we introduce MedROV, the first Real-time Open Vocabulary detection model for medical imaging. To enable open-vocabulary learning, we curate a large-scale dataset, Omnis, with 600K detection samples across nine imaging modalities and introduce a pseudo-labeling strategy to handle missing annotations from multi-source datasets. Additionally, we enhance generalization by incorporating knowledge from a large pre-trained foundation model. By leveraging contrastive learning and cross-modal representations, MedROV effectively detects both known and novel structures. Experimental results demonstrate that MedROV outperforms the previous state-of-the-art foundation model for medical image detection with an average absolute improvement of 40 mAP50, and surpasses closed-set detectors by more than 3 mAP50, while running at 70 FPS, setting a new benchmark in medical detection. Our source code, dataset, and trained model are available at https://github.com/toobatehreem/MedROV.

---

## 3. Infinity-RoPE: Action-Controllable Infinite Video Generation Emerges From Autoregressive Self-Rollout

**论文链接:** [http://arxiv.org/abs/2511.20649v1](http://arxiv.org/abs/2511.20649v1)

**作者:** Hidir Yesiltepe, Tuna Han Salih Meral, Adil Kaan Akan, Kaan Oktay, Pinar Yanardag

**发布时间:** 2025-11-25

**备注:** Project Page: https://infinity-rope.github.io/

### GPT解析

### 总结

本文提出了∞-RoPE框架，解决了自回归视频扩散模型的三个核心瓶颈：有限时间范围、提示响应速度慢和不连续电影式过渡问题。

### 背景

当前自回归视频扩散模型受到三个核心瓶颈制约：基础模型的3D旋转位置嵌入(3D-RoPE) imposed的有限时间范围、在长期rollouts中保持精细动作控制时提示响应缓慢、无法在单个生成流中实现不连续的电影式过渡。

### 目的

引入∞-RoPE，一个统一的推理时框架，通过三个相互关联的组件解决所有三个限制，实现无限范围、可控和电影式的视频生成。

### 方法

∞-RoPE包含三个组件：1)块相对论RoPE：将时间编码重新公式化为移动的局部参考系；2)KV刷新：通过仅保留两个关键潜在帧来更新KV缓存；3)RoPE切割：在时间RoPE坐标中引入受控不连续性。

### 主要发现

这些组件共同将∞-RoPE建立为一个无限范围、可控和电影式视频扩散的无训练基础，能够实现超出基础位置限制的连续视频生成。

### 结论

全面的实验表明，∞-RoPE在总体VBench评分中始终优于先前的自回归模型。

### 翻译

当前自回归视频扩散模型受到三个核心瓶颈的限制：(i)基础模型的3D旋转位置嵌入(3D-RoPE) imposed的有限时间范围，(ii)在长期rollouts中保持精细动作控制时提示响应缓慢，(iii)无法在单个生成流中实现不连续的电影式过渡。我们引入∞-RoPE，一个统一的推理时框架，通过三个相互关联的组件解决了所有三个限制：块相对论RoPE、KV刷新和RoPE切割。块相对论RoPE将时间编码重新公式化为移动的局部参考系，其中每个新生成的潜在块相对于基础模型的最大帧范围旋转，而较早的块向后旋转以保持相对时间几何形状。这种相对论公式消除了固定的时间位置，使连续视频生成能够远远超出基础位置限制。为了在不重新编码的情况下获得精细的动作控制，KV刷新通过仅保留两个潜在帧（全局汇和最后生成的潜在帧）来更新KV缓存，确保即时提示响应。最后，RoPE切割在时间RoPE坐标中引入受控不连续性，使单个连续rollout内能够进行多场景转换。这些组件共同将∞-RoPE建立为一个无限范围、可控和电影式视频扩散的无训练基础。全面的实验表明，∞-RoPE在总体VBench评分中始终优于先前的自回归模型。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决当前自回归视频扩散模型的三个核心瓶颈：(1)基础模型的3D旋转位置嵌入(3D-RoPE)施加的有限时间范围限制；(2)长序列生成中精细动作控制的提示响应速度缓慢；(3)无法在单个生成流中实现不连续的电影过渡效果。这些问题在现实中很重要，因为许多应用如电影制作、直播流、交互式视频等需要长视频内容、精细的对象动作控制和场景切换能力，而现有方法无法满足这些需求。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': "作者从不同角度重新审视长视频生成挑战，发现现有方法主要通过更长的自滚动或长视频数据重新训练来扩展时间尺度，但受限于3D-RoPE的绝对帧索引。作者探索纯粹通过相对适应和训练-free架构重新参数化来克服这些瓶颈，而不依赖额外长视频监督。他们发现仅训练5秒剪辑的DiTs已具备无限范围生成能力。借鉴了认知科学中的'语义化'概念，设计了三个关键组件：Block-Relativistic RoPE（受相对论物理学启发）、KV Flush和RoPE Cut。这些方法建立在现有自回归视频生成框架（如Self-Forcing、CausVid）的基础上，但通过创新技术解决了它们的局限性。", '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是通过三个相互关联的组件解决现有模型的三个瓶颈：Block-Relativistic RoPE将时间编码重新定义为移动参考系，使生成能超出基础模型的帧限制；KV Flush通过保留最小缓存实现即时动作控制；RoPE Cut在时间坐标中实现受控不连续跳跃以支持场景切换。整体流程是：基于预训练的自回归视频扩散模型，应用Block-Relativistic RoPE实现无限长度生成，使用KV Flush实现精细动作控制，通过RoPE Cut实现电影风格过渡。整个过程是训练-free的，可在推理时直接应用于现有模型。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：(1)Block-Relativistic RoPE：一种相对论位置编码，引入语义化过程使时间轴无限扩展；(2)KV Flush：高效缓存管理机制，实现即时提示响应；(3)RoPE Cut：实现电影风格场景过渡的训练-free操作；(4)统一的推理时框架。相比之前工作的不同：不依赖长视频训练数据（如SkyReels-V2）；突破RoPE维度限制（如NOVA、MAGI-1）；更高效的缓存管理（相比LongLive的KV-Recache）；首次实现单个自回归生成中的不连续电影过渡能力。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '∞-RoPE提出了一种训练-free的统一推理时框架，通过三个创新组件将现有的短范围自回归视频扩散模型转换为能够生成无限长度、支持精细动作控制和实现电影风格场景过渡的高质量视频生成系统。'}


### 论文摘要

Current autoregressive video diffusion models are constrained by three core bottlenecks: (i) the finite temporal horizon imposed by the base model's 3D Rotary Positional Embedding (3D-RoPE), (ii) slow prompt responsiveness in maintaining fine-grained action control during long-form rollouts, and (iii) the inability to realize discontinuous cinematic transitions within a single generation stream. We introduce $\infty$-RoPE, a unified inference-time framework that addresses all three limitations through three interconnected components: Block-Relativistic RoPE, KV Flush, and RoPE Cut. Block-Relativistic RoPE reformulates temporal encoding as a moving local reference frame, where each newly generated latent block is rotated relative to the base model's maximum frame horizon while earlier blocks are rotated backward to preserve relative temporal geometry. This relativistic formulation eliminates fixed temporal positions, enabling continuous video generation far beyond the base positional limits. To obtain fine-grained action control without re-encoding, KV Flush renews the KV cache by retaining only two latent frames, the global sink and the last generated latent frame, thereby ensuring immediate prompt responsiveness. Finally, RoPE Cut introduces controlled discontinuities in temporal RoPE coordinates, enabling multi-cut scene transitions within a single continuous rollout. Together, these components establish $\infty$-RoPE as a training-free foundation for infinite-horizon, controllable, and cinematic video diffusion. Comprehensive experiments show that $\infty$-RoPE consistently surpasses previous autoregressive models in overall VBench scores.

---

## 4. LocateAnything3D: Vision-Language 3D Detection with Chain-of-Sight

**论文链接:** [http://arxiv.org/abs/2511.20648v1](http://arxiv.org/abs/2511.20648v1)

**作者:** Yunze Man, Shihao Wang, Guowen Zhang, Johan Bjorck, Zhiqi Li, Liang-Yan Gui, Jim Fan, Jan Kautz, Yu-Xiong Wang, Zhiding Yu

**发布时间:** 2025-11-25

**备注:** Tech report. Project page: https://nvlabs.github.io/LocateAnything3D/

### GPT解析

### 总结

LocateAnything3D是一种视觉语言模型原生方法，将3D检测转化为下一个令牌预测问题，通过视线链序列模拟人类推理方式，在Omni3D基准测试上实现了最先进的结果。

### 背景

当前的视觉语言模型擅长开放式的2D描述和定位，但多目标3D检测在VLM工具箱中仍然缺失。

### 目的

提出一种VLM原生的解决方案，将3D检测转化为下一个令牌预测问题，实现多目标3D检测能力。

### 方法

提出LocateAnything3D方法，使用视线链(Chain-of-Sight, CoS)序列，先进行2D检测作为视觉思维链，然后预测3D边界框。采用从易到难的课程：跨物体采用从近到远的顺序，在每个物体内部从相机中心、尺寸和旋转因子化，按稳定性和可学习性对信息排序。

### 主要发现

在具有挑战性的Omni3D基准测试上，模型达到49.89 AP_3D，比之前最好的结果提高+15.51绝对改进，即使基线模型给定真实2D边界框。模型还能零样本泛化到未包含的类别，具有强大鲁棒性。

### 结论

通过将3D检测转变为有纪律的下一个令牌问题，LocateAnything3D为模型在3D感知方面提供了实用基础。

### 翻译

为了在世界中行动，模型必须命名它所看到的内容并知道它在三维空间中的位置。当今的视觉语言模型擅长开放式的二维描述和定位，然而多目标三维检测在很大程度上仍然缺失于VLM工具箱中。我们提出了LocateAnything3D，一种VLM原生的方法，将三维检测作为下一个令牌预测问题。关键是一个简明明确的视线链序列，它模拟了人类从图像推理的方式：在二维中找到一个物体，然后推断其距离、大小和姿态。解码器首先发出二维检测作为视觉思维链，然后在从易到难的课程下预测三维边界框：跨物体采用从近到远的顺序，减少早期歧义并匹配以自我为中心的效用；在每个物体内部，从相机中心、尺寸和旋转因子化，按稳定性和可学习性对信息排序。这种VLM原生接口保留了开放词汇表和视觉提示能力，无需专门的头部。在具有挑战性的Omni3D基准测试上，我们的模型达到了最先进的结果，49.89 AP_3D，比之前最好的结果提高了+15.51的绝对改进，即使基线模型给定真实的二维边界框。它还能零样本泛化到未包含的类别，具有强大的鲁棒性。通过将三维检测转变为有纪律的下一个令牌问题，LocateAnything3D为模型在三维感知方面提供了实用基础。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决将3D检测能力整合到视觉语言模型(VLM)中的问题，实现开放词汇表的多目标3D检测。这个问题很重要，因为3D感知是智能体与世界互动的关键能力，3D边界框是连接识别与交互的紧凑场景表示，而当前VLM缺乏这种能力，限制了智能体在三维空间中的理解和行动能力。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': "作者借鉴人类从图像中推理的方式：先在2D中找到物体，再推断其距离、大小和姿态。他们将3D检测转化为下一个令牌预测问题，通过'视线链'(CoS)序列实现。设计过程中考虑了2D作为中间步骤的可靠性、跨对象从近到远的课程学习、以及对象内中心→大小→旋转的语义排序。他们借鉴了文本中的'思维链'方法、自回归解码的课程学习策略和VLM的多模态能力，但创新性地将其组合应用于3D检测。", '这个方法的核心思想是什么？整体实现流程是怎样的？': "核心思想是通过'视线链'(CoS)将3D检测转化为结构化的下一个令牌预测问题，在自回归解码器中交织2D和3D令牌。流程包括：接收单目RGB图像和文本查询；模型先输出2D边界框作为视觉证据；然后继续输出对应的3D边界框；按照从近到远的顺序处理对象；在每个对象内按照中心→大小→旋转的顺序解码；使用统一接口支持文本或视觉提示；最后输出校准的多目标3D边界框。", '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：视线链(CoS)公式将3D检测转化为VLM中的原生令牌预测问题；针对自回归解码的课程学习策略(近→远序列化和中心→大小→旋转排序)；以及以相机为中心的数据集。相比之前工作，它提供了统一的VLM接口而非耦合专门3D头；将2D定位作为视觉思维链而非依赖外部检测器；采用深度感知的排序而非传统扫描线排序；实现了大幅提升的性能(49.89 AP3D，比之前最好高15.51点)和更强的零样本泛化能力。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'LocateAnything3D通过视线链解码将多目标单目3D检测转化为视觉语言模型中的结构化令牌预测问题，实现了开放词汇表的3D感知，并在保持通用接口的同时大幅提升了性能和泛化能力。'}


### 论文摘要

To act in the world, a model must name what it sees and know where it is in 3D. Today's vision-language models (VLMs) excel at open-ended 2D description and grounding, yet multi-object 3D detection remains largely missing from the VLM toolbox. We present LocateAnything3D, a VLM-native recipe that casts 3D detection as a next-token prediction problem. The key is a short, explicit Chain-of-Sight (CoS) sequence that mirrors how human reason from images: find an object in 2D, then infer its distance, size, and pose. The decoder first emits 2D detections as a visual chain-of-thought, then predicts 3D boxes under an easy-to-hard curriculum: across objects, a near-to-far order reduces early ambiguity and matches ego-centric utility; within each object, a center-from-camera, dimensions, and rotation factorization ranks information by stability and learnability. This VLM-native interface preserves open-vocabulary and visual-prompting capability without specialized heads. On the challenging Omni3D benchmark, our model achieves state-of-the-art results, with 49.89 AP_3D, surpassing the previous best by +15.51 absolute improvement even when the baseline is given ground-truth 2D boxes. It also generalizes zero-shot to held-out categories with strong robustness. By turning 3D detection into a disciplined next-token problem, LocateAnything3D offers a practical foundation for models to perceive in 3D.

---

## 5. PixelDiT: Pixel Diffusion Transformers for Image Generation

**论文链接:** [http://arxiv.org/abs/2511.20645v1](http://arxiv.org/abs/2511.20645v1)

**作者:** Yongsheng Yu, Wei Xiong, Weili Nie, Yichen Sheng, Shiqiu Liu, Jiebo Luo

**发布时间:** 2025-11-25

### GPT解析

### 总结

本文提出了PixelDiT，一种单阶段、端到端的扩散模型，直接在像素空间学习扩散过程，通过双级Transformer架构实现了高效的像素空间扩散模型训练，同时保留了细节，在多个评估指标上超越了现有方法。

### 背景

潜空间建模一直是扩散Transformer（DiT）的标准方法。然而，它依赖于两阶段流水线，其中预训练的自编码器引入了有损重建，导致错误累积同时阻碍了联合优化。

### 目的

解决潜空间建模中自编码器带来的有损重建和错误累积问题，实现端到端的像素空间扩散模型训练。

### 方法

提出PixelDiT，一种单阶段、端到端模型，消除了对自编码器的需求，直接在像素空间学习扩散过程。PixelDiT采用基于Transformer的完全架构，采用双级设计：块级DiT捕获全局语义，像素级DiT细化纹理细节，同时保留精细细节。

### 主要发现

有效的像素级令牌建模对像素扩散的成功至关重要。PixelDiT在ImageNet 256x256上达到1.61 FID，大幅超越现有的像素生成模型。扩展到文本到图像生成任务，在1024x1024分辨率下预训练，在GenEval上达到0.74，在DPG-bench上达到83.5，接近最佳潜扩散模型。

### 结论

PixelDiT通过直接在像素空间进行扩散建模，避免了潜空间建模中的有损重建和错误累积问题，实现了更高效的训练和更好的生成质量，特别是在保留细节方面表现出色。

### 翻译

潜空间建模一直是扩散Transformer（DiT）的标准方法。然而，它依赖于两阶段流水线，其中预训练的自编码器引入了有损重建，导致错误累积同时阻碍了联合优化。为解决这些问题，我们提出了PixelDiT，一种单阶段、端到端模型，消除了对自编码器的需求，直接在像素空间学习扩散过程。PixelDiT采用基于Transformer的完全架构，采用双级设计：块级DiT捕获全局语义，像素级DiT细化纹理细节，同时保留精细细节。我们的分析表明，有效的像素级令牌建模对像素扩散的成功至关重要。PixelDiT在ImageNet 256x256上达到1.61 FID，大幅超越现有的像素生成模型。我们进一步将PixelDiT扩展到文本到图像生成任务，并在像素空间以1024x1024分辨率进行预训练。它在GenEval上达到0.74，在DPG-bench上达到83.5，接近最佳潜扩散模型。


### 论文摘要

Latent-space modeling has been the standard for Diffusion Transformers (DiTs). However, it relies on a two-stage pipeline where the pretrained autoencoder introduces lossy reconstruction, leading to error accumulation while hindering joint optimization. To address these issues, we propose PixelDiT, a single-stage, end-to-end model that eliminates the need for the autoencoder and learns the diffusion process directly in the pixel space. PixelDiT adopts a fully transformer-based architecture shaped by a dual-level design: a patch-level DiT that captures global semantics and a pixel-level DiT that refines texture details, enabling efficient training of a pixel-space diffusion model while preserving fine details. Our analysis reveals that effective pixel-level token modeling is essential to the success of pixel diffusion. PixelDiT achieves 1.61 FID on ImageNet 256x256, surpassing existing pixel generative models by a large margin. We further extend PixelDiT to text-to-image generation and pretrain it at the 1024x1024 resolution in pixel space. It achieves 0.74 on GenEval and 83.5 on DPG-bench, approaching the best latent diffusion models.

---

## 6. Reinforcing Action Policies by Prophesying

**论文链接:** [http://arxiv.org/abs/2511.20633v1](http://arxiv.org/abs/2511.20633v1)

**作者:** Jiahui Zhang, Ze Huang, Chun Gu, Zipei Ma, Li Zhang

**发布时间:** 2025-11-25

**备注:** https://LogosRoboticsGroup.github.io/ProphRL

### GPT解析

### 总结

该研究提出了ProphRL框架，通过结合预训练世界模型Prophet和强化学习方法Flow-action-GRPO与FlowScale，解决了VLA模型在模仿学习中过拟合和分布偏移下的脆弱性问题，同时提高了数据效率和优化稳定性。

### 背景

Vision-Language-Action (VLA) policies在协调语言、感知和机器人控制方面表现出色，但大多数VLA仅通过模仿训练，这会导致对演示数据的过拟合，并且在分布偏移下表现脆弱。强化学习可以直接优化任务奖励，但真实机器人交互成本高昂，传统模拟器难以工程化和迁移。

### 目的

解决VLA后训练中的数据效率和优化稳定性问题，提供一种实际、数据高效且计算高效的VLA后训练方法。

### 方法

1) 引入Prophet：在大规模异构机器人数据上预训练的动作到视频机器人执行模型，学习可重用的动作-结果动力学；2) 使用Flow-action-GRPO (FA-GRPO)：使Flow-GRPO能够操作VLA动作；3) 应用FlowScale：一种逐步重加权方法，重新缩放流头中的每步梯度；4) 三者共同构成ProphRL框架。

### 主要发现

实验显示ProphRL在公共基准测试上取得了5-17%的成功率提升，在不同VLA变体的真实机器人上取得了24-30%的性能提升。

### 结论

ProphRL提供了一种实际、数据高效且计算高效的VLA后训练路径，能够有效提升VLA模型在各种环境下的性能和鲁棒性。

### 翻译

Vision-Language-Action (VLA)策略在协调语言、感知和机器人控制方面表现出色。然而，大多数VLA仅通过模仿训练，这会导致对演示数据的过拟合，并且在分布偏移下表现脆弱。强化学习(RL)可以直接优化任务奖励并解决这种不匹配问题，但真实机器人交互成本高昂，传统模拟器难以工程化和迁移。我们通过学习世界模型和针对基于流的动作头的RL程序，解决了VLA后训练中的数据效率和优化稳定性问题。具体来说，我们引入了Prophet，一个统一的动作到视频机器人执行模型，在大规模异构机器人数据上预训练以学习可重用的动作-结果动力学。它能够少样本适应新机器人、物体和环境，生成即用型模拟器。基于Prophet，我们使用Flow-action-GRPO (FA-GRPO)强化动作策略，该算法使Flow-GRPO能够操作VLA动作，并使用FlowScale，一种逐步重加权方法，重新缩放流头中的每步梯度。Prophet、FA-GRPO和FlowScale共同构成了ProphRL，一种实用的、数据高效且计算高效的VLA后训练路径。实验显示在公共基准测试上取得了5-17%的成功率提升，在不同VLA变体的真实机器人上取得了24-30%的性能提升。


### 论文摘要

Vision-Language-Action (VLA) policies excel in aligning language, perception, and robot control. However, most VLAs are trained purely by imitation, which overfits to demonstrations, and is brittle under distribution shift. Reinforcement learning (RL) directly optimizes task reward and thus addresses this misalignment, but real-robot interaction is expensive and conventional simulators are hard to engineer and transfer. We address both data efficiency and optimization stability in VLA post-training via a learned world model and an RL procedure tailored to flow-based action heads. Specifically, we introduce Prophet, a unified action-to-video robot actuation pretrained across large-scale, heterogeneous robot data to learn reusable action-outcome dynamics. It is able to few-shot adapt to new robots, objects, and environments, yielding a rollout-ready simulator. Upon Prophet, we reinforce action policies with Flow-action-GRPO (FA-GRPO), which adapts Flow-GRPO to operate on VLA actions, and with FlowScale, a stepwise reweighting that rescales per-step gradients in the flow head. Together, Prophet, FA-GRPO, and FlowScale constitute ProphRL, a practical, data- and compute-efficient path to VLA post-training. Experiments show 5-17% success gains on public benchmarks and 24-30% gains on real robots across different VLA variants.

---

## 7. Fighting AI with AI: Leveraging Foundation Models for Assuring AI-Enabled Safety-Critical Systems

**论文链接:** [http://arxiv.org/abs/2511.20627v1](http://arxiv.org/abs/2511.20627v1)

**作者:** Anastasia Mavridou, Divya Gopinath, Corina S. Păsăreanu

**发布时间:** 2025-11-25

### GPT解析

### 总结

研究提出了一种利用AI解决安全关键系统中AI组件集成所带来挑战的方法，包含REACT和SemaLens两个互补组件，实现了从非正式需求到已验证实现的全面流程。

### 背景

将AI组件（特别是深度神经网络DNNs）集成到航空航天和自动驾驶等安全关键系统中面临着基本的保证挑战。

### 目的

解决AI系统的不透明性、高层需求与低层网络表示间的语义鸿沟，以及需求工程中的自然语言规范模糊性和形式化可扩展性瓶颈等问题。

### 方法

提出利用AI本身应对挑战的两种互补组件：REACT使用大型语言模型连接非正式自然语言需求和正式规范；SemaLens利用视觉语言模型使用人类可理解的概念来推理、测试和监控基于DNN的感知系统。

### 主要发现

AI特定的挑战被需求工程中长期存在的问题所放大，需要创新方法来处理。

### 结论

REACT和SemaLens组件共同提供了一个从非正式需求到已验证实现的全面流程，有效解决了安全关键系统中AI集成的保证挑战。

### 翻译

将AI组件，特别是深度神经网络（DNNs），集成到航空航天和自动驾驶等安全关键系统中，为验证带来了根本性挑战。AI系统的不透明性，加上高层需求和低层网络表示之间的语义鸿沟，对传统验证方法构成了障碍。这些AI特定的挑战被需求工程中长期存在的问题所放大，包括自然语言规范的模糊性和形式化的可扩展性瓶颈。我们提出了一种利用AI本身通过两个互补组件应对这些挑战的方法。REACT（使用AI进行需求工程以确保一致性和测试）利用大型语言模型（LLMs）连接非正式自然语言需求和正式规范，实现早期验证和验证。SemaLens（使用大型多模态模型进行视觉感知的语义分析）利用视觉语言模型（VLMs）使用人类可理解的概念来推理、测试和监控基于DNN的感知系统。这两个组件共同提供了一个从非正式需求到已验证实现的全面流程。


### 论文摘要

The integration of AI components, particularly Deep Neural Networks (DNNs), into safety-critical systems such as aerospace and autonomous vehicles presents fundamental challenges for assurance. The opacity of AI systems, combined with the semantic gap between high-level requirements and low-level network representations, creates barriers to traditional verification approaches. These AI-specific challenges are amplified by longstanding issues in Requirements Engineering, including ambiguity in natural language specifications and scalability bottlenecks in formalization. We propose an approach that leverages AI itself to address these challenges through two complementary components. REACT (Requirements Engineering with AI for Consistency and Testing) employs Large Language Models (LLMs) to bridge the gap between informal natural language requirements and formal specifications, enabling early verification and validation. SemaLens (Semantic Analysis of Visual Perception using large Multi-modal models) utilizes Vision Language Models (VLMs) to reason about, test, and monitor DNN-based perception systems using human-understandable concepts. Together, these components provide a comprehensive pipeline from informal requirements to validated implementations.

---

## 8. Copyright Detection in Large Language Models: An Ethical Approach to Generative AI Development

**论文链接:** [http://arxiv.org/abs/2511.20623v1](http://arxiv.org/abs/2511.20623v1)

**作者:** David Szczecina, Senan Gaffori, Edmond Li

**发布时间:** 2025-11-25

**备注:** 4 pages, 3 figures

### GPT解析

### 总结

本文介绍了一个开源的版权检测平台，使内容创作者能够验证其作品是否被用于大型语言模型训练数据集。

### 背景

大型语言模型被广泛使用，但其训练数据中可能包含未经授权的版权内容。现有检测框架计算密集型，独立创作者难以访问，且随着法律审查增加，需要更有效的解决方案。

### 目的

开发一个可扩展、透明且用户友好的解决方案，使内容创作者能够验证其作品是否被用于LLM训练数据集。

### 方法

通过改进现有方法提高易用性，改进相似性检测，优化数据集验证，使用高效API调用减少10-30%的计算开销，并提供直观用户界面和可扩展后端。

### 主要发现

通过高效API调用可以将计算开销减少10-30%，同时保持或提高检测效果。

### 结论

该框架有助于提高AI开发的透明度和道德合规性，为负责任的AI开发和版权执行的研究奠定基础。

### 翻译

大型语言模型(LLMs)的广泛使用引发了关于训练数据中未经授权包含版权内容的严重关切。现有的检测框架，如DE-COP，计算密集型，且独立创作者 largely 难以访问。随着法律审查的增加，迫切需要可扩展、透明且用户友好的解决方案。本文介绍了一个开源的版权检测平台，使内容创作者能够验证其作品是否被用于LLM训练数据集。我们的方法通过提高易用性、改进相似性检测、优化数据集验证，并通过高效的API调用将计算开销减少10-30%，从而改进了现有方法。凭借直观的用户界面和可扩展的后端，该框架有助于提高AI开发的透明度和道德合规性，为负责任的AI开发和版权执行的研究奠定基础。


### 论文摘要

The widespread use of Large Language Models (LLMs) raises critical concerns regarding the unauthorized inclusion of copyrighted content in training data. Existing detection frameworks, such as DE-COP, are computationally intensive, and largely inaccessible to independent creators. As legal scrutiny increases, there is a pressing need for a scalable, transparent, and user-friendly solution. This paper introduce an open-source copyright detection platform that enables content creators to verify whether their work was used in LLM training datasets. Our approach enhances existing methodologies by facilitating ease of use, improving similarity detection, optimizing dataset validation, and reducing computational overhead by 10-30% with efficient API calls. With an intuitive user interface and scalable backend, this framework contributes to increasing transparency in AI development and ethical compliance, facilitating the foundation for further research in responsible AI development and copyright enforcement.

---

## 9. Wanderland: Geometrically Grounded Simulation for Open-World Embodied AI

**论文链接:** [http://arxiv.org/abs/2511.20620v1](http://arxiv.org/abs/2511.20620v1)

**作者:** Xinhao Liu, Jiaqi Li, Youming Deng, Ruxin Chen, Yingjia Zhang, Yifei Ma, Li Guo, Yiming Li, Jing Zhang, Chen Feng

**发布时间:** 2025-11-25

### GPT解析

### 总结

Wanderland是一个解决具身人工智能中可重现闭环评估瓶颈的高保真模拟框架，通过多传感器捕获、可靠重建、精确几何和鲁棒视图合成，为开放世界具身人工智能研究提供了新基础。

### 背景

可重现的闭环评估是具身人工智能如视觉导航领域的主要瓶颈。高保真模拟结合逼真传感器渲染与复杂开放世界城市环境中的几何交互是一个有前景的方向。

### 目的

解决当前视频-3DGS方法在开放世界场景捕获中的局限性，这些方法因存在较大的视觉和几何模拟到现实差距而不适合用于基准测试。

### 方法

引入Wanderland，一个从现实到模拟的框架，具有多传感器捕获、可靠重建、精确几何和鲁棒视图合成的特点。使用此流程整理了室内外城市场景的多样化数据集，并系统性地分析了各因素对导航策略学习和评估可靠性的影响。

### 主要发现

仅基于图像的管道扩展性差；几何质量影响新颖视图合成；这些因素对导航策略学习和评估可靠性产生不利影响。

### 结论

Wanderland不仅可作为具身导航的可信赖测试平台，其丰富的原始传感器数据还可用于3D重建和新颖视图合成模型的基准测试。该工作为开放世界具身人工智能的可重现研究建立了新基础。

### 翻译

在具身人工智能如视觉导航中，可重现的闭环评估仍然是一个主要瓶颈。一条有前景的前进道路是高保真模拟，它将逼真的传感器渲染与复杂开放世界城市环境中的几何交互相结合。尽管最近的视频-3DGS方法简化了开放世界场景捕获，但由于存在较大的视觉和几何模拟到现实差距，它们仍不适合用于基准测试。为解决这些挑战，我们引入了Wanderland，一个从现实到模拟的框架，具有多传感器捕获、可靠重建、精确几何和鲁棒视图合成的特点。使用此流程，我们整理了室内外城市场景的多样化数据集，并系统性地展示了仅基于图像的管道扩展性差、几何质量如何影响新颖视图合成，以及所有这些因素如何不利地影响导航策略学习和评估可靠性。除了作为具身导航的可信赖测试平台外，Wanderland丰富的原始传感器数据还可用于3D重建和新颖视图合成模型的基准测试。我们的工作为开放世界具身人工智能的可重现研究建立了新基础。项目网站在https://ai4ce.github.io/wanderland/。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决开放世界具身AI（如视觉导航）缺乏高保真且几何基础可靠的仿真环境问题。现有基于视频的3D高斯溅射方法虽然能创建视觉吸引人的环境，但存在3D重建不准确、几何不可靠和视角外渲染质量下降等问题，无法提供可靠的物理交互和可重复的评估基准。这个问题在现实中很重要，因为具身AI需要可重复的闭环评估来进步，而缺乏几何准确性的仿真环境会导致导航策略学习和评估不可靠，限制具身AI在真实开放世界应用中的发展。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先识别了现有视频3DGS方法在具身AI评估中的局限性，意识到需要结合视觉保真度和几何准确性。他们设计了一个多传感器采集方案，使用手持3D扫描仪收集LiDAR、IMU和视觉数据，通过LIV-SLAM系统融合这些数据产生精确的几何基础。该方法借鉴了现有3D高斯溅射技术进行视觉渲染，SLAM技术进行多传感器融合，以及marching cubes算法进行网格提取，但创新性地将这些技术与物理交互需求结合，创建了一个既视觉真实又几何一致的仿真环境。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是通过结合多传感器数据提供准确的几何基础，同时利用3D高斯溅射实现视觉保真度，确保物理交互的可靠性，创建一个既视觉真实又几何一致的统一仿真环境。整体流程包括：1)使用MetaCam设备在多样化城市环境中采集多传感器数据；2)通过LIV-SLAM处理产生密集度量级点云和精确相机姿态；3)对图像进行掩码处理和校正；4)从点云初始化并训练3D高斯模型；5)提取可靠碰撞网格；6)将3D高斯模型和几何网格集成到USD格式中；7)在Isaac Sim中创建统一环境并定义导航任务。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)多传感器多样化视图采集系统；2)基于LIV-SLAM的可靠重建方法；3)从LiDAR点云提取的准确度量级几何；4)结合精确几何初始化的强视图合成能力；5)统一渲染与物理交互的仿真环境。相比之前的工作，Wanderland不仅支持开放的室内外混合环境（而传统数据集仅限室内），还解决了视频方法中存在的3D重建不准确、几何不可靠和视角外渲染质量下降等问题。它提供了丰富的原始传感器数据，支持3D重建和新型视图合成的基准测试，实现了视觉保真度和几何准确性的平衡。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'Wanderland通过结合多传感器采集、可靠重建和精确几何，为开放世界具身AI提供了一个既视觉保真又几何一致的仿真环境和数据集，解决了现有视频重建方法在可靠评估中的局限性。'}


### 论文摘要

Reproducible closed-loop evaluation remains a major bottleneck in Embodied AI such as visual navigation. A promising path forward is high-fidelity simulation that combines photorealistic sensor rendering with geometrically grounded interaction in complex, open-world urban environments. Although recent video-3DGS methods ease open-world scene capturing, they are still unsuitable for benchmarking due to large visual and geometric sim-to-real gaps. To address these challenges, we introduce Wanderland, a real-to-sim framework that features multi-sensor capture, reliable reconstruction, accurate geometry, and robust view synthesis. Using this pipeline, we curate a diverse dataset of indoor-outdoor urban scenes and systematically demonstrate how image-only pipelines scale poorly, how geometry quality impacts novel view synthesis, and how all of these adversely affect navigation policy learning and evaluation reliability. Beyond serving as a trusted testbed for embodied navigation, Wanderland's rich raw sensor data further allows benchmarking of 3D reconstruction and novel view synthesis models. Our work establishes a new foundation for reproducible research in open-world embodied AI. Project website is at https://ai4ce.github.io/wanderland/.

---

## 10. Building a Foundation Model for Trajectory from Scratch

**论文链接:** [http://arxiv.org/abs/2511.20610v1](http://arxiv.org/abs/2511.20610v1)

**作者:** Gaspard Merten, Mahmoud Sakr, Gilles Dejaegere

**发布时间:** 2025-11-25

**DOI:** 10.1145/3748636.3758021

### GPT解析

### 总结

本教程展示了从GPT-2开始构建专注于轨迹的基础模型的最小实现步骤和代码，通过代码驱动的方式演示如何将GPT-2适配用于时空数据，并回顾比较了代表性的轨迹基础模型。

### 背景

基础模型在人工智能中具有变革性，但特别是针对移动轨迹的基础模型从零开始构建尚不清楚或未记录。

### 目的

填补这一空白，展示从GPT-2开始构建专注于轨迹的基础模型的最小实现的步骤和代码，向研究人员和从业者解释基础模型的概念和术语。

### 方法

通过简洁的、分步骤的、代码驱动的过程展示如何将GPT-2适配用于时空数据；回顾和比较代表性的轨迹基础模型，如TrajFM和TrajGPT，突出它们的架构创新和差异；介绍来自相关领域的互补技术，如TimesFM的修补方法。

### 主要发现

创建这种教育材料对于支持SIGSPATIAL社区构建和评估移动基础模型是及时且必不可少的。

### 结论

这种教育材料有助于提高移动AI研究的清晰度和同行评审的有效性。

### 翻译

基础模型在人工智能中具有变革性，但特别是针对移动轨迹的基础模型从零开始构建尚不清楚或未记录。本教程通过展示从GPT-2开始构建专注于轨迹的基础模型的最小实现的步骤和代码来填补这一空白。通过简洁的、分步骤的、代码驱动的过程，我们演示了如何将GPT-2适配用于时空数据。然后，我们回顾和比较了代表性的轨迹基础模型，如TrajFM和TrajGPT，突出了它们的架构创新和差异。此外，我们还介绍了来自相关领域的互补技术，如TimesFM的修补方法。针对研究人员和从业者，本教程旨在从实现层面解释基础模型的概念和术语。我们发现，创建这种教育材料对于支持SIGSPATIAL社区构建和评估移动基础模型是及时且必不可少的，以提高移动AI研究的清晰度和同行评审的有效性。


### 论文摘要

Foundation models are transformative in artificial intelligence, but building them from scratch, especially for mobility trajectories, is not yet clear or documented. This tutorial bridges this gap by demonstrating the steps and code of a minimal implementation of a trajectory-focused foundation model starting from GPT-2. Through a concise, step-by-step, code-driven process, we demonstrate adapting GPT-2 for spatiotemporal data. We then review and compare representative trajectory foundation models, such as TrajFM and TrajGPT, highlighting their architectural innovations and differences. Additionally, we introduce complementary techniques from related domains, like TimesFM's patching approach. Targeted at researchers and practitioners, this tutorial aims to explain the concepts and terminology of foundation models, at the implementation level. We find it timely and indispensable to create this educational material in order to support the SIGSPATIAL community in building and evaluating mobility foundation models, enhancing both research clarity and peer-review effectiveness in mobility AI.

---

## 11. PaTAS: A Parallel System for Trust Propagation in Neural Networks Using Subjective Logic

**论文链接:** [http://arxiv.org/abs/2511.20586v1](http://arxiv.org/abs/2511.20586v1)

**作者:** Koffi Ismael Ouattara, Ioannis Krontiris, Theo Dimitrakos, Dennis Eisermann, Frank Kargl

**发布时间:** 2025-11-25

### GPT解析

### 总结

本文提出了并行可信度评估系统(PaTAS)，一个使用主观逻辑对神经网络可信度进行建模和传播的框架，用于解决AI系统在安全关键应用中的可信度评估问题。

### 背景

可信度已成为人工智能系统在安全关键应用中部署的关键要求，但传统的评估指标如准确率和精确度无法捕捉模型预测的不确定性和可靠性，特别是在对抗性或降级条件下。

### 目的

引入一个框架来建模和传播神经网络中的可信度，使AI系统能够在安全关键应用中可靠部署。

### 方法

PaTAS通过可信度节点和可信度函数与标准神经网络计算并行运行，传播输入、参数和激活的可信度；定义了参数可信度更新机制来改进训练期间的参数可靠性；以及推理路径可信度评估方法来计算推理时特定实例的可信度。

### 主要发现

实验表明PaTAS产生可解释、对称和收敛的可信度估计，能够补充准确率指标并揭示在受污染、有偏见或不确定数据场景中的可靠性差距；能有效区分良性输入和对抗性输入，识别模型置信度与实际可靠性存在差异的情况。

### 结论

PaTAS通过在神经网络架构中实现透明和可量化的可信度推理，为在整个AI生命周期中评估模型可靠性提供了理论基础。

### 翻译

可信度已成为人工智能系统在安全关键应用中部署的关键要求。传统的评估指标如准确率和精确度无法捕捉不确定性或模型预测的可靠性，特别是在对抗性或降级条件下。本文引入了并行可信度评估系统(PaTAS)，一个使用主观逻辑(SL)对神经网络中的可信度进行建模和传播的框架。PaTAS通过可信度节点和可信度函数与标准神经网络计算并行运行，在网络中传播输入、参数和激活的可信度。该框架定义了参数可信度更新机制，用于在训练期间改进参数可靠性，以及推理路径可信度评估(IPTA)方法，用于在推理时计算特定实例的可信度。在真实世界和对抗性数据集上的实验表明，PaTAS产生可解释、对称和收敛的可信度估计，这些估计补充了准确率并揭示了在受污染、有偏见或不确定数据场景中的可靠性差距。结果显示PaTAS能有效区分良性输入和对抗性输入，并识别模型置信度与实际可靠性存在差异的情况。通过在神经网络架构中实现透明和可量化的可信度推理，PaTAS为在整个AI生命周期中评估模型可靠性提供了理论基础。


### 论文摘要

Trustworthiness has become a key requirement for the deployment of artificial intelligence systems in safety-critical applications. Conventional evaluation metrics such as accuracy and precision fail to capture uncertainty or the reliability of model predictions, particularly under adversarial or degraded conditions. This paper introduces the \emph{Parallel Trust Assessment System (PaTAS)}, a framework for modeling and propagating trust in neural networks using Subjective Logic (SL). PaTAS operates in parallel with standard neural computation through \emph{Trust Nodes} and \emph{Trust Functions} that propagate input, parameter, and activation trust across the network. The framework defines a \emph{Parameter Trust Update} mechanism to refine parameter reliability during training and an \emph{Inference-Path Trust Assessment (IPTA)} method to compute instance-specific trust at inference. Experiments on real-world and adversarial datasets demonstrate that PaTAS produces interpretable, symmetric, and convergent trust estimates that complement accuracy and expose reliability gaps in poisoned, biased, or uncertain data scenarios. The results show that PaTAS effectively distinguishes between benign and adversarial inputs and identifies cases where model confidence diverges from actual reliability. By enabling transparent and quantifiable trust reasoning within neural architectures, PaTAS provides a principled foundation for evaluating model reliability across the AI lifecycle.

---

## 12. MSTN: Fast and Efficient Multivariate Time Series Model

**论文链接:** [http://arxiv.org/abs/2511.20577v1](http://arxiv.org/abs/2511.20577v1)

**作者:** Sumit S Shevtekar, Chandresh K Maurya, Gourab Sil

**发布时间:** 2025-11-25

**备注:** 21 pages, 1 figure, 5 tables

### GPT解析

### 总结

本文提出了多尺度时间网络（MSTN），一种新型深度学习架构，能够处理时间序列数据中的多尺度动态特性，从毫秒级短期变化到长期趋势，在32个基准数据集中的24个上实现了最先进性能。

### 背景

真实世界的时间序列数据具有高度非平稳性和复杂动态特性，跨越多个时间尺度。现有模型依赖固定尺度的结构先验，如基于块的标记化、固定频率变换或冻结的主干架构，导致对时间动态的过度正则化，限制了模型处理不可预测的突然高幅度事件的能力。

### 目的

解决现有模型在处理多尺度时间动态特性时的局限性，提出一种能够自适应建模从毫秒级到长期依赖关系的时间模式的新架构。

### 方法

引入多尺度时间网络（MSTN），基于分层多尺度和序列建模原则，包含三个主要组件：多尺度卷积编码器构建层次化特征金字塔；序列建模组件处理长程时间依赖；门控融合机制结合挤压-激励（SE）和多头时间注意力（MHTA）实现动态特征融合。

### 主要发现

MSTN在时间序列长期预测、插补、分类和泛化研究中实现了具有竞争力的最先进性能，相比EMTSF、LLM4TS、HiMTM、TIME-LLM、MTST、SOFTS、iTransformer、TimesNet和PatchTST等当代方法有所改进，在32个基准数据集中的24个上建立了新的SOTA性能。

### 结论

MSTN通过分层多尺度和序列建模有效解决了时间序列模型处理多尺度动态特性的局限性，在多样化时间任务上表现出一致的高性能，为未来架构改进提供了灵活的基础。

### 翻译

真实世界的时间序列数据具有高度非平稳性和复杂的动态特性，这些特性跨越多个时间尺度，从快速、短期变化到缓慢、长期趋势。大多数现有模型依赖固定尺度的结构先验，如基于块的标记化、固定频率变换或冻结的主干架构。这通常导致对时间动态的过度正则化，限制了它们自适应建模完整时间变化谱的能力，并损害了它们在不可预测的、突然的高幅度事件上的性能。为解决这一问题，我们引入了多尺度时间网络（MSTN），一种基于分层多尺度和序列建模原则的新型深度学习架构。MSTN框架整合：（i）构建局部模式层次化特征金字塔的多尺度卷积编码器；（ii）用于长程时间依赖的序列建模组件。我们使用BiLSTM和Transformer变体对此进行了经验验证，为未来架构改进建立了灵活的基础；（iii）结合挤压-激励（SE）和多头时间注意力（MHTA）的门控融合机制，实现动态、上下文感知的特征融合。这种设计使MSTN能够在统一框架内自适应建模从毫秒级到长期依赖的时间模式。在时间序列长期预测、插补、分类和泛化研究的广泛评估中，MSTN实现了具有竞争力的最先进性能，显示出对包括EMTSF、LLM4TS、HiMTM、TIME-LLM、MTST、SOFTS、iTransformer、TimesNet和PatchTST在内的当代方法的改进。总之，MSTN在32个基准数据集中的24个上建立了新的SOTA性能，展示了其在多样化时间任务上的一致性能。


### 论文摘要

Real-world time-series data is highly non stationary and complex in dynamics that operate across multiple timescales, ranging from fast, short-term changes to slow, long-term trends. Most existing models rely on fixed-scale structural priors, such as patch-based tokenization, fixed frequency transformations, or frozen backbone architectures. This often leads to over-regularization of temporal dynamics, which limits their ability to adaptively model the full spectrum of temporal variations and impairs their performance on unpredictable, Sudden, high-magnitude events. To address this, we introduce the Multi-scale Temporal Network (MSTN), a novel deep learning architecture founded on a hierarchical multi-scale and sequence modeling principle. The MSTN framework integrates: (i) a multi-scale convolutional encoder that constructs a hierarchical feature pyramid for local patterns (ii) a sequence modeling component for long-range temporal dependencies. We empirically validate this with BiLSTM and Transformer variants, establishing a flexible foundation for future architectural advancements. and (iii) a gated fusion mechanism augmented with squeeze-and-excitation (SE) and multi-head temporal attention (MHTA) for dynamic, context-aware feature integration. This design enables MSTN to adaptively model temporal patterns from milliseconds to long-range dependencies within a unified framework. Extensive evaluations across time-series long-horizon forecasting, imputation, classification and generalizability study demonstrate that MSTN achieves competitive state-of-the-art (SOTA) performance, showing improvements over contemporary approaches including EMTSF, LLM4TS, HiMTM, TIME-LLM, MTST, SOFTS, iTransformer, TimesNet, and PatchTST. In total, MSTN establishes new SOTA performance on 24 of 32 benchmark datasets, demonstrating its consistent performance across diverse temporal tasks.

---

## 13. Verifying Numerical Methods with Isabelle/HOL

**论文链接:** [http://arxiv.org/abs/2511.20550v1](http://arxiv.org/abs/2511.20550v1)

**作者:** Dustin Bryant, Jonathan Julian Huerta y Munive, Simon Foster

**发布时间:** 2025-11-25

**备注:** 30 pages, 30 listings, for accompanying formalisation, see https://zenodo.org/records/17679526

### GPT解析

### 总结

该论文提出了一种基于ITrees的在Isabelle/HOL中实现的经过验证的数值方法框架，为从形式规范到可执行源代码提供了端到端的路径。

### 背景

现代机器学习管道建立在数值算法之上，可靠的数值方法是可信机器学习和网络物理系统的先决条件。

### 目的

开发一个用户友好的规范框架，使数值程序可以直接声明并带有变体和不变量注释，用于推理正确性规范。

### 方法

使用ITrees基础与Isabelle的代码生成器交互导出源代码，通过自动证明方法和HOL-Analysis库的引理处理验证条件。

### 主要发现

通过二分法和定点迭代法两种著名方法展示了数值方法建模和验证的有效性，并贡献了皮亚诺形式的高阶导数和泰勒定理作为数学库的扩展。

### 结论

对使用该框架验证数值方法进行了定性评估，证明了框架的有效性和实用性。

### 翻译

现代机器学习管道建立在数值算法之上。因此，可靠的数值方法是可信机器学习和网络物理系统的先决条件。我们在Isabelle/HOL中基于ITrees贡献了一个经过验证的数值方法框架。我们的用户友好的规范语言使能够直接声明带有变体和不变量注释的数值程序，用于推理正确性规范。生成的验证条件可以通过自动证明方法和HOL-Analysis库的引理来处理。ITrees基础与Isabelle的代码生成器交互以导出源代码，这为从具有机器检查保证的形式规范到可执行源代码提供了端到端的路径。我们通过关注两种著名方法（二分法和定点迭代法）来说明数值方法的建模过程并证明验证的有效性。我们还贡献了实现此目标所需的正式化数学库的关键扩展：皮亚诺形式的高阶导数和泰勒定理。最后，我们对使用该框架验证数值方法进行了定性评估。


### 论文摘要

Modern machine learning pipelines are built on numerical algorithms. Reliable numerical methods are thus a prerequisite for trustworthy machine learning and cyber-physical systems. Therefore, we contribute a framework for verified numerical methods in Isabelle/HOL based on ITrees. Our user-friendly specification language enables the direct declaration of numerical programs that can be annotated with variants and invariants for reasoning about correctness specifications. The generated verification conditions can be discharged via automated proof methods and lemmas from the HOL-Analysis library. The ITrees foundation interacts with Isabelle's code generator to export source code. This provides an end-to-end path from formal specifications with machine-checked guarantees to executable sources. We illustrate the process of modelling numerical methods and demonstrate the effectiveness of the verification by focusing on two well-known methods, the bisection method and the fixed-point iteration method. We also contribute crucial extensions to the libraries of formalised mathematics required for this objective: higher-order derivatives and Taylor's theorem in Peano form. Finally, we qualitatively evaluate the use of the framework for verifying numerical methods.

---

## 14. Proceedings Twentieth Conference on Theoretical Aspects of Rationality and Knowledge

**论文链接:** [http://arxiv.org/abs/2511.20540v1](http://arxiv.org/abs/2511.20540v1)

**作者:** Adam Bjorndahl

**发布时间:** 2025-11-25

**DOI:** 10.4204/EPTCS.437

### GPT解析

### 总结

TARK会议（理性与知识理论方面）是一个汇集计算机科学、人工智能、博弈论、决策理论、哲学、逻辑学、语言学和认知科学等领域研究人员的国际会议。

### 背景

自1986年起由Joe Halpern（康奈尔大学）倡议，每两年在世界各地举办一次，旨在促进对理性和知识推理跨学科问题的理解。

### 目的

进一步理解涉及理性和知识推理的跨学科问题，涵盖知识、信念、不确定性、意识、有限理性、常识认识论推理等多个主题。

### 方法

通过会议形式汇集研究者，接受论文并进行展示，本次会议为第二十届，将于2025年7月14日至16日在德国杜塞尔多夫的海因里希-海涅大学举行。

### 主要发现

摘要中未提及具体研究发现的细节。

### 结论

摘要中未提供明确的结论性陈述。

### 翻译

TARK会议（理性与知识理论方面）是一个旨在汇集计算机科学、人工智能、博弈论、决策理论、哲学、逻辑学、语言学和认知科学等多个领域研究人员的会议。其目标是增进我们对涉及理性和知识推理的跨学科问题的理解。自1986年以来，在Joe Halpern（康奈尔大学）的倡议下，会议每两年在世界各地举办一次。感兴趣的主题包括但不限于：知识、信念、不确定性、意识、有限理性、常识认识论推理、认识论逻辑、认识论博弈论、知识与行动、关于知识和其他心理状态推理的应用、信念修正、计算社会选择、算法博弈论和多智能体系统的基础。TARK信息可在http://www.tark.org/获取。这些论文集包含第二十届理性与知识理论方面会议（TARK 2025）上接受的论文，该会议将于2025年7月14日至16日在德国杜塞尔多夫的海因里希-海涅大学举行。会议网站可在https://ccc.cs.uni-duesseldorf.de/tark-2025/找到。


### 论文摘要

The TARK conference (Theoretical Aspects of Rationality and Knowledge) is a conference that aims to bring together researchers from a wide variety of fields, including computer science, artificial intelligence, game theory, decision theory, philosophy, logic, linguistics, and cognitive science. Its goal is to further our understanding of interdisciplinary issues involving reasoning about rationality and knowledge.   Previous conferences have been held biennially around the world since 1986, on the initiative of Joe Halpern (Cornell University). Topics of interest include, but are not limited to, semantic models for knowledge, belief, uncertainty, awareness, bounded rationality, common sense epistemic reasoning, epistemic logic, epistemic game theory, knowledge and action, applications of reasoning about knowledge and other mental states, belief revision, computational social choice, algorithmic game theory, and foundations of multi-agent systems.   Information about TARK is available at http://www.tark.org/.   These proceedings contain the papers that have been accepted for presentation at the Twentieth Conference on Theoretical Aspects of Rationality and Knowledge (TARK 2025), held July 14--16, 2025, at Heinrich-Heine-Universität, Düsseldorf, Germany. The conference website can be found at https://ccc.cs.uni-duesseldorf.de/tark-2025/.

---

## 15. Beyond Generation: Multi-Hop Reasoning for Factual Accuracy in Vision-Language Models

**论文链接:** [http://arxiv.org/abs/2511.20531v1](http://arxiv.org/abs/2511.20531v1)

**作者:** Shamima Hossain

**发布时间:** 2025-11-25

**备注:** Accepted as poster at NewInML Workshop ICML, 2025

### GPT解析

### 总结

该研究提出了一个知识引导推理框架，用于提高视觉语言模型(VLMs)的事实准确性，通过整合结构化知识图谱进行多跳验证，在图像标注任务中实现了约31%的准确性提升。

### 背景

视觉语言模型(VLMs)是强大的生成工具，但常因缺乏稳健推理能力而产生事实不准确输出。虽然大型语言模型(LLMs)中集成外部知识的研究已较充分，但在VLMs中这类研究仍显不足，且需要桥接多种模态的挑战。

### 目的

开发一个框架，在VLMs中实现知识引导推理，提高事实准确性，探索不同知识表示方法的效果，并揭示VLMs的推理模式和失败模式。

### 方法

引入一个利用结构化知识图谱进行多跳验证的框架，通过图像标注任务展示。该方法实现多步骤系统推理，包括视觉实体识别、知识图谱遍历和基于事实的标注细化。使用分层、三元组和要点三种知识表示进行评估。

### 主要发现

在Google Landmarks v2、Conceptual captions和Coco captions混合数据集的初步实验中，该方法将事实准确性提高了约31%。不同知识表示方法在事实准确性和逻辑推理方面表现出不同的效果。

### 结论

集成外部知识能够提升VLMs的推理能力，为开发更可靠和知识丰富的多模态系统提供了新途径。

### 翻译

视觉语言模型(VLMs)是强大的生成工具，但往往由于缺乏稳健的推理能力而产生事实不准确的输出。虽然在大型语言模型(LLMs)中集成外部知识进行推理已有广泛研究，但在VLMs中这类研究仍然不足，且需要无缝桥接多种模态的挑战增加了难度。本研究引入了一个VLMs中的知识引导推理框架，利用结构化知识图谱进行多跳验证，使用图像标注任务来说明我们的框架。该方法能够跨多步骤进行系统推理，包括视觉实体识别、知识图谱遍历和基于事实的标注细化。我们使用分层、基于三元组和基于要点的知识表示评估该框架，分析它们在事实准确性和逻辑推理方面的有效性。实证结果表明，在精心策划的Google Landmarks v2、Conceptual captions和Coco captions混合数据集上的初步实验中，我们的方法将事实准确性提高了约31%，揭示了推理模式和失败模式的关键见解。这项工作证明了集成外部知识对提升VLMs推理能力的潜力，为更可靠和知识丰富的多模态系统铺平了道路。


### 论文摘要

Visual Language Models (VLMs) are powerful generative tools but often produce factually in- accurate outputs due to a lack of robust reason- ing capabilities. While extensive research has been conducted on integrating external knowl- edge for reasoning in large language models (LLMs), such efforts remain underexplored in VLMs, where the challenge is compounded by the need to bridge multiple modalities seam- lessly. This work introduces a framework for knowledge-guided reasoning in VLMs, leverag- ing structured knowledge graphs for multi-hop verification using image-captioning task to il- lustrate our framework. Our approach enables systematic reasoning across multiple steps, in- cluding visual entity recognition, knowledge graph traversal, and fact-based caption refine- ment. We evaluate the framework using hi- erarchical, triple-based and bullet-point based knowledge representations, analyzing their ef- fectiveness in factual accuracy and logical infer- ence. Empirical results show that our approach improves factual accuracy by approximately 31% on preliminary experiments on a curated dataset of mixtures from Google Landmarks v2, Conceptual captions and Coco captions re- vealing key insights into reasoning patterns and failure modes. This work demonstrates the po- tential of integrating external knowledge for advancing reasoning in VLMs, paving the way for more reliable and knowledgable multimodal systems.

---

## 16. Assessing LLMs' Performance: Insights from the Chinese Pharmacist Exam

**论文链接:** [http://arxiv.org/abs/2511.20526v1](http://arxiv.org/abs/2511.20526v1)

**作者:** Xinran Wang, Boran Zhu, Shujuan Zhou, Ziwen Long, Dehua Zhou, Shu Zhang

**发布时间:** 2025-11-25

**备注:** 15 pages, 4 figures

### GPT解析

### 总结

本研究比较了两种大语言模型在中国药师资格考试中的表现，发现DeepSeek-R1的整体准确率(90.0%)显著高于ChatGPT-4o(76.1%)，特别是在基础和临床综合模块中表现更佳，表明领域特定模型在专业考试评估中具有潜力。

### 背景

随着大语言模型越来越多地融入数字健康教育和评估工作流程，它们在支持高风险、领域特定的认证任务方面的能力尚未得到充分探索。在中国，国家药师资格考试是评估药师临床和理论能力的标准化基准。

### 目的

比较ChatGPT-4o和DeepSeek-R1两种大语言模型在中国药师资格考试(2017-2021年)真实题目上的表现，并讨论这些表现差异对AI形成性评估的意义。

### 方法

从官方考试、培训材料和公共数据库汇编了2,306道纯文本多项选择题，排除包含表格或图像的问题。每道题以原始中文格式输入，评估模型响应的精确准确性。使用Pearson卡方检验比较整体表现，使用Fisher精确检验进行逐年多项选择题准确性比较。

### 主要发现

DeepSeek-R1表现优于ChatGPT-4o，整体准确率显著更高(90.0% vs 76.1%, p < 0.001)。单元级分析显示DeepSeek-R1在基础和临床综合模块中具有持续优势。虽然逐年多项选择题表现也偏向DeepSeek-R1，但在任何特定单元-年份中，这一性能差距均未达到统计学显著性(所有p > 0.05)。

### 结论

DeepSeek-R1与药师资格考试的结构和语义需求显示出强大的对齐。这些发现表明，针对此情境值得进一步研究领域特定模型，同时，在法律和伦理敏感的背景下，强化人类监督的必要性。

### 翻译

背景：随着大语言模型(LLMs)越来越多地融入数字健康教育和评估工作流程，它们在支持高风险、领域特定的认证任务方面的能力尚未得到充分探索。在中国，国家药师资格考试是评估药师临床和理论能力的标准化基准。目的：本研究旨在比较两种大语言模型：ChatGPT-4o和DeepSeek-R1在中国药师资格考试(2017-2021年)真实题目上的表现，并讨论这些表现差异对AI形成性评估的意义。方法：从官方考试、培训材料和公共数据库汇编了2,306道多项选择题(纯文本)。排除了包含表格或图像的问题。每道题以其原始中文格式输入，评估模型响应的精确准确性。使用Pearson卡方检验比较整体表现，使用Fisher精确检验进行逐年多项选择题准确性比较。结果：DeepSeek-R1的表现优于ChatGPT-4o，整体准确率显著更高(90.0% vs 76.1%, p < 0.001)。单元级分析显示DeepSeek-R1具有持续优势，特别是在基础和临床综合模块中。虽然逐年多项选择题表现也偏向DeepSeek-R1，但在任何特定单元-年份中，这一性能差距均未达到统计学显著性(所有p > 0.05)。结论：DeepSeek-R1与药师资格考试的结构和语义需求显示出强大的对齐。这些发现表明，针对此情境值得进一步研究领域特定模型，同时，在法律和伦理敏感的背景下，强化人类监督的必要性。


### 论文摘要

Background: As large language models (LLMs) become increasingly integrated into digital health education and assessment workflows, their capabilities in supporting high-stakes, domain-specific certification tasks remain underexplored.In China, the national pharmacist licensure exam serves as a standardized benchmark for evaluating pharmacists' clinical and theoretical competencies. Objective: This study aimed to compare the performance of two LLMs: ChatGPT-4o and DeepSeek-R1 on real questions from the Chinese Pharmacist Licensing Examination (2017-2021), and to discuss the implications of these performance differences for AI-enabled formative evaluation. Methods: A total of 2,306 multiple-choice (text-only) questions were compiled from official exams, training materials, and public databases. Questions containing tables or images were excluded. Each item was input in its original Chinese format, and model responses were evaluated for exact accuracy. Pearson's Chi-squared test was used to compare overall performance, and Fisher's exact test was applied to year-wise multiple-choice accuracy. Results: DeepSeek-R1 outperformed ChatGPT-4o with a significantly higher overall accuracy (90.0% vs. 76.1%, p < 0.001). Unit-level analyses revealed consistent advantages for DeepSeek-R1, particularly in foundational and clinical synthesis modules. While year-by-year multiple-choice performance also favored DeepSeek-R1, this performance gap did not reach statistical significance in any specific unit-year (all p > 0.05). Conclusion: DeepSeek-R1 demonstrated robust alignment with the structural and semantic demands of the pharmacist licensure exam. These findings suggest that domain-specific models warrant further investigation for this context, while also reinforcing the necessity of human oversight in legally and ethically sensitive contexts.

---

## 17. HBridge: H-Shape Bridging of Heterogeneous Experts for Unified Multimodal Understanding and Generation

**论文链接:** [http://arxiv.org/abs/2511.20520v1](http://arxiv.org/abs/2511.20520v1)

**作者:** Xiang Wang, Zhifei Zhang, He Zhang, Zhe Lin, Yuqian Zhou, Qing Liu, Shiwei Zhang, Yijun Li, Shaoteng Liu, Haitian Zheng, Jason Kuen, Yuehuan Wang, Changxin Gao, Nong Sang

**发布时间:** 2025-11-25

### GPT解析

### 总结

本文提出了一种名为HBridge的非对称H形架构，用于优化多模态生成模型中理解专家与生成专家的融合方式，通过选择性桥接中间层减少注意力共享，提高效率和生成质量。

### 背景

最近的统一模型将理解专家（如LLMs）与生成专家（如扩散模型）集成，实现了强大的多模态性能。然而，当前采用对称设计的方法（如BAGEL和LMFusion）由于固有的模态差异，仍不是最优的。

### 目的

提出HBridge架构，使异构专家能够最佳地利用各自模态领域的预训练先验，解决现有方法中的模态差异问题。

### 方法

HBridge采用非对称H形架构，选择性桥接中间层而非所有层，减少了40%以上的注意力共享。浅层和深层被解耦以捕获模态特定表示，中间层桥接促进语义对齐。同时引入语义重建令牌，明确指导生成专家重建目标图像的视觉语义令牌。

### 主要发现

在多个基准测试上的大量实验证明，HBridge在效率和生成质量方面均优于现有方法，减少了40%以上的注意力共享同时增强了生成效果。

### 结论

HBridge建立了统一多模态生成的新范式，通过优化专家间的融合方式，有效解决了模态差异问题，提升了多模态生成的性能和效率。

### 翻译

最近的统一模型将理解专家（如大型语言模型）与生成专家（如扩散模型）集成，实现了强大的多模态性能。然而，最近的高级方法如BAGEL和LMFusion遵循混合变换器范式，采用对称设计，使一个专家镜像另一个专家以便于初始化和融合，但由于固有的模态差异，这种方法仍然不是最优的。在本工作中，我们提出HBridge，一种非对称的H形架构，使异构专家能够最佳地利用各自模态领域的预训练先验。与之前通过共享注意力直接连接专家之间所有层的密集融合策略不同，HBridge选择性地桥接中间层，减少了40%以上的注意力共享，提高了效率并增强了生成质量。浅层和深层捕获模态特定表示，被解耦，而中间层桥接促进语义对齐。为了进一步增强跨模态一致性，我们引入了语义重建令牌，明确指导生成专家重建目标图像的视觉语义令牌。在多个基准测试上的大量实验证明了HBridge的有效性和优越性能，建立了统一多模态生成的新范式。


### 论文摘要

Recent unified models integrate understanding experts (e.g., LLMs) with generative experts (e.g., diffusion models), achieving strong multimodal performance. However, recent advanced methods such as BAGEL and LMFusion follow the Mixture-of-Transformers (MoT) paradigm, adopting a symmetric design that mirrors one expert to another for convenient initialization and fusion, which remains suboptimal due to inherent modality discrepancies. In this work, we propose HBridge, an asymmetric H-shaped architecture that enables heterogeneous experts to optimally leverage pretrained priors from their respective modality domains. Unlike prior dense fusion strategies that straightforwardly connect all layers between experts via shared attention, HBridge selectively bridges intermediate layers, reducing over 40% attention sharing, which improves efficiency and enhances generation quality. Shallow and deep layers, which capture modality-specific representations, are decoupled, while mid-layer bridging promotes semantic alignment. To further strengthen cross-modal coherence, we introduce semantic reconstruction tokens that explicitly guide the generative expert to reconstruct visual semantic tokens of the target image. Extensive experiments across multiple benchmarks demonstrate the effectiveness and superior performance of HBridge, establishing a new paradigm for unified multimodal generation.

---

## 18. DesignPref: Capturing Personal Preferences in Visual Design Generation

**论文链接:** [http://arxiv.org/abs/2511.20513v1](http://arxiv.org/abs/2511.20513v1)

**作者:** Yi-Hao Peng, Jeffrey P. Bigham, Jason Wu

**发布时间:** 2025-11-25

### GPT解析

### 总结

本研究引入DesignPref数据集，研究视觉设计中的个性化偏好问题，并提出个性化模型策略以改善生成式模型在UI设计中的应用效果。

### 背景

生成式模型（如大型语言模型和文本到图像扩散模型）被广泛用于创建视觉设计，但这些模型的微调和基准测试通常依赖于人类标注的设计偏好数据集。然而，由于视觉设计的主观性和高度个性化，个体间的偏好差异很大。

### 目的

研究视觉设计中的个性化偏好问题，探索如何更准确地建模和预测个体设计师的设计偏好，以提高生成式模型在UI设计中的应用效果。

### 方法

创建DesignPref数据集，包含12k个UI设计生成的成对比较，由20位专业设计师进行多级别偏好评分；研究传统多数投票方法的局限性；探索多种个性化策略，特别是微调或整合设计师特定注释到RAG管道中。

### 主要发现

专业设计师之间存在显著分歧（二元偏好的Krippendorff's alpha值为0.25）；分歧源于对设计方面重要性认知的差异和个人偏好；传统多数投票方法无法准确反映个人偏好；个性化模型在预测个人设计师偏好方面显著优于聚合基线模型，即使使用的示例数量少20倍。

### 结论

个性化模型能有效解决视觉设计中的主观偏好问题，为未来研究个体设计品味建模提供了基础。DesignPref数据集是首个支持个性化视觉设计评估的数据集，有助于推动该领域的研究发展。

### 翻译

生成式模型，如大型语言模型和文本到图像扩散模型，越来越多地被用于创建视觉设计，如用户界面（UI）和演示幻灯片。这些生成式模型的微调和基准测试通常依赖于人类标注的设计偏好数据集。然而，由于视觉设计的主观性和高度个性化，个体间的偏好差异很大。在本文中，我们通过引入DesignPref（包含12k个UI设计生成的成对比较，由20位专业设计师进行多级别偏好评分的数据集）来研究这一问题。我们发现，即使在训练有素的设计师之间，也存在大量分歧（二元偏好的Krippendorff's alpha值为0.25）。这些设计师提供的自然语言理由表明，分歧源于对各种设计方面重要性的不同看法和个人偏好。利用DesignPref，我们证明传统的多数投票方法训练聚合判断模型往往不能准确反映个人偏好。为了应对这一挑战，我们研究了多种个性化策略，特别是将设计师特定的注释微调或整合到RAG管道中。我们的结果显示，个性化模型在预测个人设计师偏好方面始终优于聚合基线模型，即使使用的示例数量少20倍。我们的工作提供了首个用于研究个性化视觉设计评估的数据集，并支持未来对个体设计品味建模的研究。


### 论文摘要

Generative models, such as large language models and text-to-image diffusion models, are increasingly used to create visual designs like user interfaces (UIs) and presentation slides. Finetuning and benchmarking these generative models have often relied on datasets of human-annotated design preferences. Yet, due to the subjective and highly personalized nature of visual design, preference varies widely among individuals. In this paper, we study this problem by introducing DesignPref, a dataset of 12k pairwise comparisons of UI design generation annotated by 20 professional designers with multi-level preference ratings. We found that among trained designers, substantial levels of disagreement exist (Krippendorff's alpha = 0.25 for binary preferences). Natural language rationales provided by these designers indicate that disagreements stem from differing perceptions of various design aspect importance and individual preferences. With DesignPref, we demonstrate that traditional majority-voting methods for training aggregated judge models often do not accurately reflect individual preferences. To address this challenge, we investigate multiple personalization strategies, particularly fine-tuning or incorporating designer-specific annotations into RAG pipelines. Our results show that personalized models consistently outperform aggregated baseline models in predicting individual designers' preferences, even when using 20 times fewer examples. Our work provides the first dataset to study personalized visual design evaluation and support future research into modeling individual design taste.

---

## 19. DP-MicroAdam: Private and Frugal Algorithm for Training and Fine-tuning

**论文链接:** [http://arxiv.org/abs/2511.20509v1](http://arxiv.org/abs/2511.20509v1)

**作者:** Mihaela Hudişteanu, Edwige Cyffers, Nikita P. Kalinin

**发布时间:** 2025-11-25

### GPT解析

### 总结

研究提出了一种名为DP-MicroAdam的自适应差分隐私优化器，它在内存效率和稀疏感知方面表现出色，在多种基准测试中优于现有方法，证明了自适应优化在差分隐私条件下的价值。

### 背景

自适应优化器是非私有训练中的事实标准，因为它们通常能实现更快的收敛性和改进的性能。相比之下，差分隐私训练仍然主要使用DP-SGD进行，通常需要大量的计算和超参数调整。

### 目的

提出DP-MicroAdam，这是一种内存高效且具有稀疏感知能力的自适应差分隐私优化器。

### 方法

证明DP-MicroAdam在随机非凸优化中以最优的O(1/√T)速率收敛，达到与隐私相关的常数。

### 主要发现

实验证明，DP-MicroAdam优于现有的自适应差分隐私优化器，在CIFAR-10、大规模ImageNet训练和预训练变换器的私有微调等一系列基准测试中，与DP-SGD相比具有竞争性或更优的准确性。

### 结论

这些结果表明，在差分隐私条件下，自适应优化可以提高性能和稳定性。

### 翻译

自适应优化器是非私有训练中的事实标准，因为它们通常能实现更快的收敛性和改进的性能。相比之下，差分隐私训练仍然主要使用DP-SGD进行，通常需要大量的计算和超参数调整。我们提出了DP-MicroAdam，这是一种内存高效且具有稀疏感知能力的自适应差分隐私优化器。我们证明了DP-MicroAdam在随机非凸优化中以最优的O(1/√T)速率收敛，达到与隐私相关的常数。实验证明，DP-MicroAdam优于现有的自适应差分隐私优化器，在CIFAR-10、大规模ImageNet训练和预训练变换器的私有微调等一系列基准测试中，与DP-SGD相比具有竞争性或更优的准确性。这些结果表明，在差分隐私条件下，自适应优化可以提高性能和稳定性。


### 论文摘要

Adaptive optimizers are the de facto standard in non-private training as they often enable faster convergence and improved performance. In contrast, differentially private (DP) training is still predominantly performed with DP-SGD, typically requiring extensive compute and hyperparameter tuning. We propose DP-MicroAdam, a memory-efficient and sparsity-aware adaptive DP optimizer. We prove that DP-MicroAdam converges in stochastic non-convex optimization at the optimal $\mathcal{O}(1/\sqrt{T})$ rate, up to privacy-dependent constants. Empirically, DP-MicroAdam outperforms existing adaptive DP optimizers and achieves competitive or superior accuracy compared to DP-SGD across a range of benchmarks, including CIFAR-10, large-scale ImageNet training, and private fine-tuning of pretrained transformers. These results demonstrate that adaptive optimization can improve both performance and stability under differential privacy.

---

## 20. Quantifying the Privacy Implications of High-Fidelity Synthetic Network Traffic

**论文链接:** [http://arxiv.org/abs/2511.20497v1](http://arxiv.org/abs/2511.20497v1)

**作者:** Van Tran, Shinan Liu, Tian Li, Nick Feamster

**发布时间:** 2025-11-25

**备注:** 14 pages, 13 Figures, 6 Tables

### GPT解析

### 总结

该研究针对网络流量数据的稀缺性和隐私问题，提出了一套全面的合成网络流量隐私度量标准，系统评估了不同生成模型的脆弱性，并确定了影响隐私泄露的关键因素。

### 背景

网络流量数据存在稀缺性和隐私问题，生成模型被用于创建合成流量，但合成流量本身并不天然具有隐私保护特性，关于其泄露敏感信息的程度及测量方法尚未充分探索，且不同模型架构增加了这一挑战。

### 目的

引入一套全面的合成网络流量隐私度量标准，结合标准攻击方法和网络特定属性，系统评估不同生成模型的脆弱性，并研究影响攻击成功的因素。

### 方法

开发综合隐私度量标准，结合成员推断攻击(MIA)和数据提取攻击以及网络特定标识符和属性，系统评估代表性生成模型的脆弱性，并分析影响攻击结果的因素。

### 主要发现

不同模型和数据集间的隐私风险差异显著，MIA成功率从0%到88%不等，最多可达100%的网络标识符可从生成流量中恢复，训练数据多样性和生成模型对训练数据的拟合度是影响攻击结果的关键因素。

### 结论

这些发现为设计和部署最小化隐私泄露的生成模型提供了可行指导，为更安全的合成网络流量生成奠定了基础。

### 翻译

为解决网络流量数据的稀缺性和隐私问题，各种生成模型被开发用于产生合成流量。然而，合成流量本身并不天然具有隐私保护特性，关于其泄露敏感信息的程度以及如何测量这种泄露，在很大程度上仍未被探索。不同模型架构的多样性进一步加剧了这一挑战。我们引入了一套全面的合成网络流量隐私度量标准，结合了成员推断攻击(MIA)和数据提取攻击等标准方法以及网络特定的标识符和属性。利用这些度量标准，我们系统评估了不同代表性生成模型的脆弱性，并检验了影响攻击成功的因素。我们的结果显示，不同模型和数据集之间的隐私风险存在显著差异。MIA成功率从0%到88%不等，最多可达100%的网络标识符可从生成流量中恢复，凸显了严重的隐私漏洞。我们进一步确定了显著影响攻击结果的关键因素，包括训练数据多样性和生成模型对训练数据的拟合程度。这些发现为设计和部署最小化隐私泄露的生成模型提供了可行的指导，为更安全的合成网络流量生成奠定了基础。


### 论文摘要

To address the scarcity and privacy concerns of network traffic data, various generative models have been developed to produce synthetic traffic. However, synthetic traffic is not inherently privacy-preserving, and the extent to which it leaks sensitive information, and how to measure such leakage, remain largely unexplored. This challenge is further compounded by the diversity of model architectures, which shape how traffic is represented and synthesized. We introduce a comprehensive set of privacy metrics for synthetic network traffic, combining standard approaches like membership inference attacks (MIA) and data extraction attacks with network-specific identifiers and attributes. Using these metrics, we systematically evaluate the vulnerability of different representative generative models and examine the factors that influence attack success. Our results reveal substantial variability in privacy risks across models and datasets. MIA success ranges from 0% to 88%, and up to 100% of network identifiers can be recovered from generated traffic, highlighting serious privacy vulnerabilities. We further identify key factors that significantly affect attack outcomes, including training data diversity and how well the generative model fits the training data. These findings provide actionable guidance for designing and deploying generative models that minimize privacy leakage, establishing a foundation for safer synthetic network traffic generation.

---

## 21. MTBBench: A Multimodal Sequential Clinical Decision-Making Benchmark in Oncology

**论文链接:** [http://arxiv.org/abs/2511.20490v1](http://arxiv.org/abs/2511.20490v1)

**作者:** Kiril Vasilev, Alexandre Misrahi, Eeshaan Jain, Phil F Cheng, Petros Liakopoulos, Olivier Michielin, Michael Moor, Charlotte Bunne

**发布时间:** 2025-11-25

**备注:** Accepted to NeurIPS 2025

### GPT解析

### 总结

本文介绍了MTBBench，一个模拟分子肿瘤委员会(MTB)风格决策的智能体基准测试，用于评估多模态大语言模型在复杂临床推理任务中的表现。研究发现当前LLM在处理多模态、纵向临床数据时存在可靠性问题，而MTBBench不仅提供评估基准，还包含提升性能的智能体框架。

### 背景

当前多模态大语言模型在生物医学推理方面有潜力，但现有基准测试无法捕捉真实临床工作流程的复杂性。现有评估主要针对单模态、脱离上下文的问答，忽略了多智能体决策环境如分子肿瘤委员会(MTBs)，其中需要整合异构数据并随时间发展见解。

### 目的

开发一个能够模拟MTB风格决策的基准测试，通过具有临床挑战性的多模态和纵向肿瘤学问题来评估多模态大语言模型的表现。

### 方法

引入MTBBench基准测试，通过临床挑战性、多模态和纵向肿瘤学问题模拟MTB风格决策。真实注释由临床医生通过共同开发的应用程序验证。对多个开源和闭源LLM进行基准测试，并提供基于基础模型的智能体框架以增强推理能力。

### 主要发现

即使在大规模情况下，LLM也缺乏可靠性，经常出现幻觉、难以从时间解析数据中推理，以及无法调和冲突证据或不同模态的问题。使用提供的智能体框架后，任务级性能分别提高了9.0%和11.2%。

### 结论

MTBBench为推进多模态LLM推理、可靠性和工具使用提供了一个具有挑战性和现实性的测试平台，特别关注精准肿瘤学中的MTB环境。

### 翻译

多模态大语言模型(LLMs)在生物医学推理方面具有潜力，但当前的基准测试无法捕捉真实临床工作流程的复杂性。现有评估主要评估单模态、脱离上下文的问答，忽略了多智能体决策环境，如分子肿瘤委员会(MTBs)。MTBs汇集了肿瘤学领域的不同专家，其中诊断和预后任务需要整合异构数据并随时间发展见解。当前基准缺乏这种纵向和多模态复杂性。我们引入了MTBBench，一个通过临床挑战性、多模态和纵向肿瘤学问题模拟MTB风格决策的智能体基准。真实注释由临床医生通过共同开发的应用程序验证，确保临床相关性。我们对多个开源和闭源LLM进行了基准测试，表明即使在大规模情况下，它们也缺乏可靠性——经常出现幻觉，难以从时间解析的数据中推理，并且无法调和冲突证据或不同模态。为解决这些局限性，MTBBench不仅提供基准测试，还提供了一个基于基础模型的智能体框架，以增强多模态和纵向推理能力，分别导致任务级性能提高9.0%和11.2%。总体而言，MTBBench为推进多模态LLM推理、可靠性和工具使用提供了一个具有挑战性和现实性的测试平台，特别关注精准肿瘤学中的MTB环境。


### 论文摘要

Multimodal Large Language Models (LLMs) hold promise for biomedical reasoning, but current benchmarks fail to capture the complexity of real-world clinical workflows. Existing evaluations primarily assess unimodal, decontextualized question-answering, overlooking multi-agent decision-making environments such as Molecular Tumor Boards (MTBs). MTBs bring together diverse experts in oncology, where diagnostic and prognostic tasks require integrating heterogeneous data and evolving insights over time. Current benchmarks lack this longitudinal and multimodal complexity. We introduce MTBBench, an agentic benchmark simulating MTB-style decision-making through clinically challenging, multimodal, and longitudinal oncology questions. Ground truth annotations are validated by clinicians via a co-developed app, ensuring clinical relevance. We benchmark multiple open and closed-source LLMs and show that, even at scale, they lack reliability -- frequently hallucinating, struggling with reasoning from time-resolved data, and failing to reconcile conflicting evidence or different modalities. To address these limitations, MTBBench goes beyond benchmarking by providing an agentic framework with foundation model-based tools that enhance multi-modal and longitudinal reasoning, leading to task-level performance gains of up to 9.0% and 11.2%, respectively. Overall, MTBBench offers a challenging and realistic testbed for advancing multimodal LLM reasoning, reliability, and tool-use with a focus on MTB environments in precision oncology.

---

## 22. Modular Deep Learning Framework for Assistive Perception: Gaze, Affect, and Speaker Identification

**论文链接:** [http://arxiv.org/abs/2511.20474v1](http://arxiv.org/abs/2511.20474v1)

**作者:** Akshit Pramod Anchan, Jewelith Thomas, Sritama Roy

**发布时间:** 2025-11-25

**备注:** 10 pages, 9 figures, and 3 tables

### GPT解析

### 总结

这项研究评估了一种受'智能眼'感知系统启发的模块化架构的可行性，提出了三个独立的传感模块用于眼状态检测、面部表情识别和基于语音的说话人识别，并在特定数据集上实现了高准确率，为未来在资源受限辅助设备中的实时多模态集成奠定了基础。

### 背景

开发全面的辅助技术需要视觉和听觉感知的无缝集成。现有的感知系统如'智能眼'具有核心功能，可以作为架构设计的参考。

### 目的

评估一种受感知系统启发的模块化架构在辅助技术中的可行性，并建立未来在资源受限设备中进行实时多模态集成的基础。

### 方法

研究提出了并基准测试了三个独立的传感模块：用于眼状态检测（困倦/注意力）的卷积神经网络，用于面部表情识别的深度卷积神经网络，以及用于基于语音的说话人识别的长短期记忆网络。使用了Eyes Image、FER2013和定制音频数据集进行测试。

### 主要发现

三个模型分别在Eyes Image、FER2013和定制音频数据集上达到了93.0%、97.8%和96.89%的准确率。研究表明，轻量级的特定领域模型可以在离散任务上实现高保真度。

### 结论

轻量级、特定领域的模型可以在离散任务上实现高保真度，为未来在资源受限辅助设备中进行实时多模态集成建立了经过验证的基础。

### 翻译

开发全面的辅助技术需要视觉和听觉感知的无缝集成。这项研究评估了受'智能眼'等感知系统核心功能启发的模块化架构的可行性。我们提出并基准测试了三个独立的传感模块：用于眼状态检测（困倦/注意力）的卷积神经网络，用于面部表情识别的深度卷积神经网络，以及用于基于语音的说话人识别的长短期记忆网络。利用Eyes Image、FER2013和定制音频数据集，我们的模型分别达到了93.0%、97.8%和96.89%的准确率。这项研究表明，轻量级、特定领域的模型可以在离散任务上实现高保真度，为未来在资源受限辅助设备中进行实时多模态集成建立了经过验证的基础。


### 论文摘要

Developing comprehensive assistive technologies requires the seamless integration of visual and auditory perception. This research evaluates the feasibility of a modular architecture inspired by core functionalities of perceptive systems like 'Smart Eye.' We propose and benchmark three independent sensing modules: a Convolutional Neural Network (CNN) for eye state detection (drowsiness/attention), a deep CNN for facial expression recognition, and a Long Short-Term Memory (LSTM) network for voice-based speaker identification. Utilizing the Eyes Image, FER2013, and customized audio datasets, our models achieved accuracies of 93.0%, 97.8%, and 96.89%, respectively. This study demonstrates that lightweight, domain-specific models can achieve high fidelity on discrete tasks, establishing a validated foundation for future real-time, multimodal integration in resource-constrained assistive devices.

---

## 23. Look Where It Matters: Training-Free Ultra-HR Remote Sensing VQA via Adaptive Zoom Search

**论文链接:** [http://arxiv.org/abs/2511.20460v1](http://arxiv.org/abs/2511.20460v1)

**作者:** Yunqi Zhou, Chengjie Jiang, Chun Yuan, Jing Li

**发布时间:** 2025-11-25

**备注:** 17 pages, 8 figures

### GPT解析

### 总结

ZoomSearch是一种无需训练、即插即用的流水线，专门针对超高分辨率遥感视觉问答任务，通过解耦'关注哪里'与'如何回答'，实现了高效率和准确性。

### 背景

随着卫星星座、传感器技术和成像流程的进步，超高分辨率遥感影像日益普及，但现有遥感基础模型难以处理这类输入：全图像编码会耗尽标记和内存预算，而基于调整大小的预处理会丢失关键细节。

### 目的

引导模型在预测前关注重要区域，解决超高分辨率遥感视觉问答任务中的挑战。

### 方法

ZoomSearch结合了自适应多分支缩放搜索（在图像块上执行分层搜索以定位与查询相关的区域）和布局感知的块重组（将选定块重新组织成紧凑且布局忠实的画布）。

### 主要发现

与LLaVA-ov集成后，ZoomSearch在LRS-VQA上比基线提高26.3%，在MME-RealWorld-RS上提高114.8%；同时，推理效率比之前的基于搜索的方法快20%~44%。

### 结论

ZoomSearch为超高分辨率遥感图像的视觉问答任务提供了一种高效且准确的解决方案。

### 翻译

随着卫星星座、传感器技术和成像流程的进步，超高分辨率（Ultra-HR）遥感影像正变得越来越普遍。然而，当前的遥感基础模型不适合处理这类输入：全图像编码会耗尽标记和内存预算，而基于调整大小的预处理会丢失细粒度和对答案关键重要的细节。在此背景下，引导模型在预测前关注重要区域变得至关重要。因此，我们提出了ZoomSearch，一种无需训练、即插即用的流水线，用于超高分辨率遥感视觉问答（RS-VQA），将'关注哪里'与'如何回答'解耦。ZoomSearch结合了自适应多分支缩放搜索，该技术在图像块上执行分层搜索以定位与查询相关的区域，以及布局感知的块重组，该技术将选定的块重新组织成紧凑且布局忠实的画布。我们在超高分辨率遥感视觉问答基准测试MME-RealWorld-RS和LRS-VQA上进行了全面实验，与(i)强大的通用基础模型，(ii)遥感基础模型，(iii)超高分辨率遥感视觉问答方法，以及(iv)即插即用的基于搜索的视觉问答方法进行了比较。当与LLaVA-ov集成时，ZoomSearch在多样化任务上达到了最先进的准确性，在LRS-VQA上比LLaVA-ov基线提高26.3%，在MME-RealWorld-RS上提高114.8%。同时，它实现了更高的推理效率，比之前的基于搜索的方法快20%~44%。


### 论文摘要

With advances in satellite constellations, sensor technologies, and imaging pipelines, ultra-high-resolution (Ultra-HR) remote sensing imagery is becoming increasingly widespread. However, current remote sensing foundation models are ill-suited to such inputs: full-image encoding exhausts token and memory budgets, while resize-based preprocessing loses fine-grained and answer-critical details. In this context, guiding the model look where it matters before prediction becomes crucial. Therefore, we present ZoomSearch, a training-free, plug-and-play pipeline that decouples 'where to look' from 'how to answer' for Ultra-HR Remote Sensing Visual Question Answering (RS-VQA). ZoomSearch combines Adaptive Multi-Branch Zoom Search, which performs a hierarchical search over image patches to localize query-relevant regions, with Layout-Aware Patch Reassembly, which reorganizes the selected patches into a compact, layout-faithful canvas. We conduct comprehensive experiments on Ultra-HR RS-VQA benchmarks MME-RealWorld-RS and LRS-VQA, comparing against (i) strong general foundation models, (ii) remote sensing foundation models, (iii) Ultra-HR RS-VQA methods, and (iv) plug-and-play search-based VQA methods. When integrated with LLaVA-ov, ZoomSearch attains state-of-the-art accuracy across diverse tasks, improving the LLaVA-ov baseline by 26.3% on LRS-VQA and 114.8\% on MME-RealWorld-RS. Meanwhile, it achieves much higher inference efficiency, outperforming prior search-based methods by 20%~44% in speed.

---

## 24. Fluid Intelligence: A Forward Look on AI Foundation Models in Computational Fluid Dynamics

**论文链接:** [http://arxiv.org/abs/2511.20455v1](http://arxiv.org/abs/2511.20455v1)

**作者:** Neil Ashton, Johannes Brandstetter, Siddhartha Mishra

**发布时间:** 2025-11-25

### GPT解析

### 总结

本文探讨了GPU和AI技术进步推动下的计算流体动力学(CFD)领域变革，提出了首个包含CFD输入的缩放定律，用于数据生成和模型训练，并提供了构建CFD基础模型的计算成本和时间估计。

### 背景

GPU和AI技术的进步正在推动计算流体动力学(CFD)领域发生重大变革，但机器学习和CFD社区之间存在明显差距。

### 目的

弥合机器学习和CFD社区之间的差距，将工业规模的CFD模拟分解为核心组件，提出适合CFD的缩放定律，开发和部署下一代AI模型解决复杂流体动力学问题。

### 方法

提出新的缩放定律包含CFD输入用于数据生成和模型训练，建立大规模限制的定量估计，区分数据生成成本和模型训练成本在不同情况下占主导地位的区域。

### 主要发现

确定了大规模限制的定量估计，区分了数据生成成本和模型训练成本占主导的不同区域，发现高保真瞬态数据为构建基础模型提供了最佳途径。

### 结论

包含高保真瞬态数据是构建基础模型的最佳途径，提供了构建CFD基础模型的计算成本和时间的具体估计。

### 翻译

在GPU和AI技术进步的推动下，计算流体动力学(CFD)领域正在经历重大变革。本文通过将工业规模的CFD模拟分解为其核心组件，弥合了机器学习和CFD社区之间的差距。我们的主要贡献是提出了首个缩放定律，该定律纳入了用于数据生成和模型训练的CFD输入，概述了为复杂流体动力学问题开发和部署这些下一代AI模型所面临的独特挑战。利用我们的新缩放定律，我们建立了大规模限制的定量估计，区分了数据生成成本在总计算量中占主导地位的区域与模型训练成本占主导地位的区域。我们得出结论，纳入高保真瞬态数据为构建基础模型提供了最佳途径。我们用具体数字约束了我们的理论，提供了构建CFD基础模型的计算成本和时间的第一批公开估计。


### 论文摘要

Driven by the advancement of GPUs and AI, the field of Computational Fluid Dynamics (CFD) is undergoing significant transformations. This paper bridges the gap between the machine learning and CFD communities by deconstructing industrial-scale CFD simulations into their core components. Our main contribution is to propose the first scaling law that incorporates CFD inputs for both data generation and model training to outline the unique challenges of developing and deploying these next-generation AI models for complex fluid dynamics problems. Using our new scaling law, we establish quantitative estimates for the large-scale limit, distinguishing between regimes where the cost of data generation is the dominant factor in total compute versus where the cost of model training prevails. We conclude that the incorporation of high-fidelity transient data provides the optimum route to a foundation model. We constrain our theory with concrete numbers, providing the first public estimates on the computational cost and time to build a foundation model for CFD.

---

## 25. Block Cascading: Training Free Acceleration of Block-Causal Video Models

**论文链接:** [http://arxiv.org/abs/2511.20426v1](http://arxiv.org/abs/2511.20426v1)

**作者:** Hmrishav Bandyopadhyay, Nikhil Pinnaparaju, Rahim Entezari, Jim Scott, Yi-Zhe Song, Varun Jampani

**发布时间:** 2025-11-25

### GPT解析

### 总结

块级联视频生成方法通过训练无关的并行化技术，显著缓解了速度与质量权衡问题，实现了约2倍的加速，同时保持了生成质量，并消除了交互式生成中的上下文切换开销。

### 背景

块因果视频生成面临明显的速度与质量权衡：小型13亿参数模型仅能实现16 FPS，而大型140亿参数模型仅能达到4.5 FPS，迫使用户在响应速度和质量之间做出选择。

### 目的

缓解视频生成中的速度与质量权衡问题，提高生成效率。

### 方法

提出块级联(Block Cascading)方法，利用核心见解——未来视频块不需要完全去噪的当前块即可开始生成。通过使用前驱块的部分去噪上下文开始块生成，将顺序管道转换为并行级联，使多个块能够同时去噪。利用5个GPU实现时间并行性。

### 主要发现

1) 在所有模型规模上实现约2倍的加速；2) 13亿参数模型从16 FPS加速到30 FPS；3) 140亿参数模型从4.5 FPS加速到12.5 FPS；4) 消除了交互式生成中上下文切换时的KV缓存开销（约200毫秒）；5) 从块因果切换到块级联管道进行推理时，生成质量没有显著损失。

### 结论

块级联方法有效缓解了视频生成中的速度与质量权衡问题，实现了更高的推理速度，同时保持了生成质量，还消除了交互式生成中的上下文切换开销。

### 翻译

块因果视频生成面临明显的速度与质量权衡：小型13亿参数模型仅能实现16 FPS，而大型140亿参数模型仅能达到4.5 FPS，迫使用户在响应速度和质量之间做出选择。块级联通过训练无关的并行化显著缓解了这种权衡。我们的核心见解是：未来的视频块不需要完全去噪的当前块即可开始生成。通过使用前驱块的部分去噪上下文开始块生成，我们将顺序管道转换为并行级联，使多个块能够同时去噪。利用5个GPU利用时间并行性，我们在所有模型规模上实现了约2倍的加速：13亿参数模型从16 FPS加速到30 FPS，140亿参数模型从4.5 FPS加速到12.5 FPS。除了推理速度外，块级联还消除了交互式生成中上下文切换时的KV缓存开销（约200毫秒）。针对多种块因果管道的广泛评估验证表明，从块因果切换到块级联管道进行推理时，生成质量没有显著损失。项目页面：https://hmrishavbandy.github.io/block_cascading_page/


### 论文摘要

Block-causal video generation faces a stark speed-quality trade-off: small 1.3B models manage only 16 FPS while large 14B models crawl at 4.5 FPS, forcing users to choose between responsiveness and quality. Block Cascading significantly mitigates this trade-off through training-free parallelization. Our key insight: future video blocks do not need fully denoised current blocks to begin generation. By starting block generation with partially denoised context from predecessors, we transform sequential pipelines into parallel cascades where multiple blocks denoise simultaneously. With 5 GPUs exploiting temporal parallelism, we achieve ~2x acceleration across all model scales: 1.3B models accelerate from 16 to 30 FPS, 14B models from 4.5 to 12.5 FPS. Beyond inference speed, Block Cascading eliminates overhead from KV-recaching (of ~200ms) during context switches for interactive generation. Extensive evaluations validated against multiple block-causal pipelines demonstrate no significant loss in generation quality when switching from block-causal to Block Cascading pipelines for inference. Project Page: https://hmrishavbandy.github.io/block_cascading_page/

---

## 26. VibraVerse: A Large-Scale Geometry-Acoustics Alignment Dataset for Physically-Consistent Multimodal Learning

**论文链接:** [http://arxiv.org/abs/2511.20422v1](http://arxiv.org/abs/2511.20422v1)

**作者:** Bo Pang, Chenxi Xu, Jierui Ren, Guoping Wang, Sheng Li

**发布时间:** 2025-11-25

### GPT解析

### 总结

本研究引入了VibraVerse大规模几何-声学对齐数据集和CLASP对比学习框架，解决了现有多模态学习框架中缺乏物理一致性和忽略物体物理属性与声音间因果关系的问题。

### 背景

理解物理世界需要基于物理定律而非仅统计相关的感知模型，但现有专注于视觉和语言的多模态学习框架缺乏物理一致性，忽视了物体几何、材料、振动模式和产生声音之间的内在因果关系。

### 目的

建立物理上一致的跨模态对齐，确保每个样本都是连贯的、可追溯到控制方程，并嵌入到统一的表示空间中，为声音引导的具身感知和对物理世界的更深入理解提供基础。

### 方法

创建VibraVerse数据集，明确连接从3D几何→物理属性→模态参数→声学信号的因果链；提出CLASP对比学习框架，用于跨模态对齐；基于VibraVerse定义几何到声音预测、声音引导的形状重建和跨模态表示学习的基准任务。

### 主要发现

在VibraVerse上训练的模型在跨模态的准确性、可解释性和泛化能力方面表现出色，验证了物理一致和因果可解释多模态学习的有效性。

### 结论

VibraVerse作为物理一致和因果可解释多模态学习的基准，为声音引导的具身感知和对物理世界的更深入理解提供了基础，该数据集将开源。

### 翻译

理解物理世界需要基于物理定律而非仅统计相关的感知模型。然而，现有的专注于视觉和语言的多模态学习框架缺乏物理一致性，忽视了物体几何、材料、振动模式和产生声音之间的内在因果关系。我们引入了VibraVerse，一个大规模的几何-声学对齐数据集，明确连接了从3D几何→物理属性→模态参数→声学信号的因果链。每个3D模型具有明确的物理属性（密度、杨氏模量、泊松比）和体积几何，从中计算模态特征频率和特征向量，用于在受控激励下的冲击声音合成。为了建立这种连贯性，我们引入了CLASP，一个用于跨模态对齐的对比学习框架，保持物体物理结构与声学响应之间的因果对应关系。该框架强制跨模态的物理一致对齐，确保每个样本都是连贯的、可追溯到控制方程，并嵌入到跨越形状、图像和声音的统一表示空间中。基于VibraVerse，我们定义了一系列几何到声音预测、声音引导的形状重建和跨模态表示学习的基准任务。在这些任务上的广泛验证表明，在VibraVerse上训练的模型在跨模态的准确性、可解释性和泛化能力方面表现出色。这些结果确立了VibraVerse作为物理一致和因果可解释多模态学习的基准，为声音引导的具身感知和对物理世界的更深入理解提供了基础。该数据集将开源。


### 论文摘要

Understanding the physical world requires perceptual models grounded in physical laws rather than mere statistical correlations. However, existing multimodal learning frameworks, focused on vision and language, lack physical consistency and overlook the intrinsic causal relationships among an object's geometry, material, vibration modes, and the sounds it produces. We introduce VibraVerse, a large-scale geometry-acoustics alignment dataset that explicitly bridges the causal chain from 3D geometry -> physical attributes -> modal parameters -> acoustic signals. Each 3D model has explicit physical properties (density, Young's modulus, Poisson's ratio) and volumetric geometry, from which modal eigenfrequencies and eigenvectors are computed for impact sound synthesis under controlled excitations. To establish this coherence, we introduce CLASP, a contrastive learning framework for cross-modal alignment that preserves the causal correspondence between an object's physical structure and its acoustic response. This framework enforces physically consistent alignment across modalities, ensuring that every sample is coherent, traceable to the governing equations, and embedded within a unified representation space spanning shape, image, and sound. Built upon VibraVerse, we define a suite of benchmark tasks for geometry-to-sound prediction, sound-guided shape reconstruction, and cross-modal representation learning. Extensive validations on these tasks demonstrate that models trained on VibraVerse exhibit superior accuracy, interpretability, and generalization across modalities. These results establish VibraVerse as a benchmark for physically consistent and causally interpretable multimodal learning, providing a foundation for sound-guided embodied perception and a deeper understanding of the physical world. The dataset will be open-sourced.

---

## 27. Self-Identifying Internal Model-Based Online Optimization

**论文链接:** [http://arxiv.org/abs/2511.20411v1](http://arxiv.org/abs/2511.20411v1)

**作者:** Wouter J. A. van Weerelt, Lantian Zhang, Silun Zhang, Nicola Bastianello

**发布时间:** 2025-11-25

### GPT解析

### 总结

本文提出了一种结合控制理论和系统识别思想的新型在线优化算法，该算法能够处理未知或变化的内部模型，并在二次和非二次问题上表现良好。

### 背景

在线优化领域中，现有方法可能缺乏对问题内部模型的适应性，而控制理论和系统识别可以为优化算法提供新的解决思路。

### 目的

设计一种能够适应未知或变化内部模型的在线优化算法，使其在二次问题及更一般问题上都能有效工作。

### 方法

构建基于控制论的优化算法，利用在线问题的内部模型，并通过集成的识别程序实时学习这个模型；算法从二次问题设计出发，但可扩展到一般问题。

### 主要发现

对于二次问题，算法能够渐近收敛到最优解轨迹；识别程序使算法能够适应内部模型的变化；数值结果表明算法在非二次问题上也有良好表现。

### 结论

所提出的结合控制理论和系统识别的在线优化算法能有效处理具有未知或变化内部模型的问题，在多种情况下展现出优越性能。

### 翻译

在本文中，我们提出了一种新的在线优化算法，该算法结合了控制理论和系统识别的思想。我们的算法基础是基于控制论的设计，它利用了在线问题的内部模型。由于这种内部模型的先验知识在实践中可能不可用，我们集成了一个识别程序，用于实时学习这个模型。算法的设计从二次在线问题开始，但可以应用于一般问题。对于二次情况，我们描述了渐近收敛到最优解轨迹的特性。我们将提出的算法与现有方法进行了比较，并展示了识别程序如何确保其对底层内部模型变化的适应性。数值结果也表明在二次问题之外有良好的表现。


### 论文摘要

In this paper, we propose a novel online optimization algorithm built by combining ideas from control theory and system identification. The foundation of our algorithm is a control-based design that makes use of the internal model of the online problem. Since such prior knowledge of this internal model might not be available in practice, we incorporate an identification routine that learns this model on the fly. The algorithm is designed starting from quadratic online problems but can be applied to general problems. For quadratic cases, we characterize the asymptotic convergence to the optimal solution trajectory. We compare the proposed algorithm with existing approaches, and demonstrate how the identification routine ensures its adaptability to changes in the underlying internal model. Numerical results also indicate strong performance beyond the quadratic setting.

---

## 28. Image-Free Timestep Distillation via Continuous-Time Consistency with Trajectory-Sampled Pairs

**论文链接:** [http://arxiv.org/abs/2511.20410v1](http://arxiv.org/abs/2511.20410v1)

**作者:** Bao Tang, Shuai Zhang, Yueting Zhu, Jijun Xiang, Xin Yang, Li Yu, Wenyu Liu, Xinggang Wang

**发布时间:** 2025-11-25

### GPT解析

### 总结

该论文提出了一种名为轨迹反向一致性模型(TBCM)的新方法，通过从教师模型的生成轨迹中提取潜在表示，消除了对外部训练数据的依赖，显著提高了扩散模型的生成效率和简单性，同时保持高质量的生成结果。

### 背景

时间步长蒸馏是提高扩散模型生成效率的有效方法。一致性模型(CM)作为基于轨迹的框架，因其强大的理论基础和高质量的多步生成而显示出巨大潜力。然而，当前的连续时间一致性蒸馏方法仍然严重依赖训练数据和计算资源，阻碍了它们在资源受限场景中的部署，并限制了它们扩展到不同领域的能力。

### 目的

解决当前连续时间一致性蒸馏方法对外部训练数据和计算资源的严重依赖问题，使方法能够在资源受限场景中部署，并提高其扩展到不同领域的能力。

### 方法

提出Trajectory-Backward Consistency Model (TBCM)，通过从教师模型的生成轨迹中提取潜在表示，直接消除对外部训练数据的依赖。这种方法采用自包含的蒸馏范式，显著提高了效率和简单性。轨迹提取的样本自然地连接了训练和推理之间的分布差距，从而实现更有效的知识转移。

### 主要发现

TBCM在MJHQ-30k数据集上的一步生成中实现了6.52 FID和28.08 CLIP分数，与Sana-Sprint相比减少了约40%的训练时间，并节省了大量GPU内存，展示了卓越的效率而没有牺牲质量。研究还揭示了连续时间一致性蒸馏中的扩散生成空间差异，并分析了采样策略如何影响蒸馏性能。

### 结论

TBCM通过消除对外部训练数据的依赖，显著提高了扩散模型生成效率和简单性，同时保持了高质量的生成结果。该方法为资源受限场景提供了一种更有效的解决方案，并为未来的蒸馏研究提供了见解。

### 翻译

时间步长蒸馏是提高扩散模型生成效率的有效方法。作为基于轨迹的框架，一致性模型(CM)因其强大的理论基础和高质量的多步生成而显示出巨大潜力。然而，当前的连续时间一致性蒸馏方法仍然严重依赖训练数据和计算资源，阻碍了它们在资源受限场景中的部署，并限制了它们扩展到不同领域的能力。为了解决这个问题，我们提出了轨迹反向一致性模型(TBCM)，它通过直接从教师模型的生成轨迹中提取潜在表示，消除了对外部训练数据的依赖。与需要VAE编码和大规模数据集的传统方法不同，我们自包含的蒸馏范式显著提高了效率和简单性。此外，轨迹提取的样本自然地连接了训练和推理之间的分布差距，从而实现更有效的知识转移。实验上，TBCM在MJHQ-30k上的一步生成中实现了6.52 FID和28.08 CLIP分数，同时与Sana-Sprint相比减少了约40%的训练时间，并节省了大量GPU内存，展示了卓越的效率而没有牺牲质量。我们进一步揭示了连续时间一致性蒸馏中的扩散生成空间差异，并分析了采样策略如何影响蒸馏性能，为未来的蒸馏研究提供了见解。GitHub链接：https://github.com/hustvl/TBCM。


### 论文摘要

Timestep distillation is an effective approach for improving the generation efficiency of diffusion models. The Consistency Model (CM), as a trajectory-based framework, demonstrates significant potential due to its strong theoretical foundation and high-quality few-step generation. Nevertheless, current continuous-time consistency distillation methods still rely heavily on training data and computational resources, hindering their deployment in resource-constrained scenarios and limiting their scalability to diverse domains. To address this issue, we propose Trajectory-Backward Consistency Model (TBCM), which eliminates the dependence on external training data by extracting latent representations directly from the teacher model's generation trajectory. Unlike conventional methods that require VAE encoding and large-scale datasets, our self-contained distillation paradigm significantly improves both efficiency and simplicity. Moreover, the trajectory-extracted samples naturally bridge the distribution gap between training and inference, thereby enabling more effective knowledge transfer. Empirically, TBCM achieves 6.52 FID and 28.08 CLIP scores on MJHQ-30k under one-step generation, while reducing training time by approximately 40% compared to Sana-Sprint and saving a substantial amount of GPU memory, demonstrating superior efficiency without sacrificing quality. We further reveal the diffusion-generation space discrepancy in continuous-time consistency distillation and analyze how sampling strategies affect distillation performance, offering insights for future distillation research. GitHub Link: https://github.com/hustvl/TBCM.

---

## 29. 论文ID: 2511.20382v1

**错误:** 无法读取论文信息 - [Errno 2] No such file or directory: 'storage/result/2511.20382v1.json'

---

## 30. VGGTFace: Topologically Consistent Facial Geometry Reconstruction in the Wild

**论文链接:** [http://arxiv.org/abs/2511.20366v1](http://arxiv.org/abs/2511.20366v1)

**作者:** Xin Ming, Yuxuan Han, Tianyu Huang, Feng Xu

**发布时间:** 2025-11-25

### GPT解析

### 总结

本文提出了一种名为VGGTFace的自动方法，用于从野外多视角图像重建拓扑一致的面部几何形状，解决了现有方法需要手动操作、泛化能力有限或表达能力不足的问题。

### 背景

重建拓扑一致的面部几何形状对于数字头像创建流程至关重要。现有方法要么需要繁琐的手动工作，要么无法推广到野外数据，或者受限于3D Morphable Models的有限表达能力。

### 目的

创新性地应用3D基础模型VGGT，从野外多视角图像重建拓扑一致的面部几何形状，实现自动化的高质量重建。

### 方法

利用VGGT的强大泛化能力和表现力，通过Pixel3DMM增强VGGT注入拓扑信息，将像素对齐点图转换为具有拓扑的点云，并提出新颖的拓扑感知束调整策略来融合这些点云，在束调整目标中构建拉普拉斯能量。

### 主要发现

该方法在单个NVIDIA RTX 4090上，16个视图仅需10秒即可实现高质量重建，在基准测试上取得了最先进的结果，并对野外数据展现出令人印象深刻的泛化能力。

### 结论

VGGTFace是一种有效的自动方法，可以从野外多视角图像快速重建高质量、拓扑一致的面部几何形状，代码已公开。

### 翻译

重建拓扑一致的面部几何形状对于数字头像创建流程至关重要。现有方法要么需要繁琐的手动工作，要么无法推广到野外数据，或者受限于3D Morphable Models的有限表达能力。为解决这些限制，我们提出了VGGTFace，一种自动方法，创新性地将3D基础模型（即VGGT）应用于从野外多视角图像重建拓扑一致的面部几何形状。我们的关键见解是，通过利用VGGT，我们的方法自然地从其大规模训练和点图表示中继承了强大的泛化能力和表现力。然而，如何从VGGT重建拓扑一致的网格尚不清楚，因为其预测中缺少拓扑信息。为此，我们通过像素对齐的UV值使用Pixel3DMM增强VGGT以注入拓扑信息。通过这种方式，我们将VGGT的像素对齐点图转换为具有拓扑的点云。针对这种具有已知拓扑的点云，我们提出了一种新颖的拓扑感知束调整策略来融合它们，在束调整目标中构建拉普拉斯能量。我们的方法在单个NVIDIA RTX 4090上，16个视图仅需10秒即可实现高质量重建。实验证明了在基准测试上的最先进结果和对野外数据的令人印象深刻的泛化能力。代码可在https://github.com/grignarder/vggtface获取。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决从野外拍摄的多视角图像中重建具有拓扑一致性的面部几何结构的问题。这个问题在现实中非常重要，因为拓扑一致的面部几何结构是数字角色创建流程的关键，它 enables 密集网格对应和可转移的动画、绑定和纹理编辑。传统方法需要大量手动工作，难以扩展到普通用户，而现有自动方法要么泛化能力差，要么受限于3D Morphable Models的表达能力。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先认识到3D基础模型VGGT在几何重建领域的潜力，其强大泛化能力和表达能力适合处理野外拍摄的多视角图像。但VGGT预测不包含拓扑信息，限制了其在面部重建中的应用。作者借鉴Pixel3DMM方法通过UV值注入拓扑信息，并改进了Bundle Adjustment技术，提出Topology-Aware Bundle Adjustment来处理具有已知拓扑的点云。整个方法设计基于对现有技术的深入理解和创新性结合，特别是将VGGT的表达能力与拓扑约束相结合。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是利用VGGT的强大泛化能力从野外图像中获取高质量点图，通过Pixel3DMM添加UV坐标注入拓扑信息，然后使用创新的拓扑感知捆绑调整技术融合多视角点云。整体流程：1)输入多视角图像；2)用VGGT获取点图和相机参数；3)用Pixel3DMM预测UV坐标；4)建立顶点与像素的对应关系；5)融合不同视角的3D位置形成初始点云；6)应用拓扑感知捆绑调整优化点云和相机参数；7)连接优化后的点云生成拓扑一致的面部网格。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点：1)创新性地应用3D基础模型VGGT进行面部重建；2)通过Pixel3DMM为VGGT预测添加拓扑信息；3)提出拓扑感知捆绑调整技术，利用拉普拉斯能量作为正则化项。相比之前工作：1)完全自动化，无需手动工作；2)基于VGGT的强大泛化能力，能处理野外拍摄数据；3)使用点图表示，不受3DMM限制，能捕获更高质量的面部特征；4)利用多视角信息，重建更准确的面部几何结构。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'VGGTFace创新性地结合3D基础模型与拓扑感知捆绑调整技术，实现了从野外多视角图像中快速、高质量重建拓扑一致的面部几何结构，为普通用户提供了强大的数字人创建工具。'}


### 论文摘要

Reconstructing topologically consistent facial geometry is crucial for the digital avatar creation pipelines. Existing methods either require tedious manual efforts, lack generalization to in-the-wild data, or are constrained by the limited expressiveness of 3D Morphable Models. To address these limitations, we propose VGGTFace, an automatic approach that innovatively applies the 3D foundation model, \emph{i.e.} VGGT, for topologically consistent facial geometry reconstruction from in-the-wild multi-view images captured by everyday users. Our key insight is that, by leveraging VGGT, our method naturally inherits strong generalization ability and expressive power from its large-scale training and point map representation. However, it is unclear how to reconstruct a topologically consistent mesh from VGGT, as the topology information is missing in its prediction. To this end, we augment VGGT with Pixel3DMM for injecting topology information via pixel-aligned UV values. In this manner, we convert the pixel-aligned point map of VGGT to a point cloud with topology. Tailored to this point cloud with known topology, we propose a novel Topology-Aware Bundle Adjustment strategy to fuse them, where we construct a Laplacian energy for the Bundle Adjustment objective. Our method achieves high-quality reconstruction in 10 seconds for 16 views on a single NVIDIA RTX 4090. Experiments demonstrate state-of-the-art results on benchmarks and impressive generalization to in-the-wild data. Code is available at https://github.com/grignarder/vggtface.

---

## 31. DAPointMamba: Domain Adaptive Point Mamba for Point Cloud Completion

**论文链接:** [http://arxiv.org/abs/2511.20278v1](http://arxiv.org/abs/2511.20278v1)

**作者:** Yinghui Li, Qianyu Zhou, Di Shao, Hao Yang, Ye Zhu, Richard Dazeley, Xuequan Lu

**发布时间:** 2025-11-25

**备注:** Accepted to AAAI 2026

### GPT解析

### 总结

本文提出DAPointMamba框架，用于领域自适应点云补全，通过三个创新模块有效解决源域和目标域间的几何和语义差异，具有全局感受场和线性复杂度优势

### 背景

领域自适应点云补全旨在缩小标记源域和无标签目标域间的几何和语义差异，但现有方法因使用CNN或视觉Transformer而存在感受野有限或二次复杂度问题

### 目的

研究状态空间模型在领域自适应点云补全中的适应性，并解决直接应用时遇到的空间拓扑破坏和表征设计问题

### 方法

提出DAPointMamba框架，包含三个创新模块：跨域补丁级扫描实现局部对齐，跨域空间SSM对齐加强空间一致性，跨域通道SSM对齐解决全局语义差距

### 主要发现

直接将3D点云序列化为1D序列会破坏目标域的空间拓扑和局部几何特征；忽略领域无关表征设计会阻碍适应性能

### 结论

DAPointMamba在跨域适应方面表现出强大能力，具有全局感受场和线性复杂度优势，在合成和真实世界基准上优于最先进方法，且计算效率更高

### 翻译

领域自适应点云补全旨在缩小标记源域和无标签目标域之间的几何和语义差异。现有方法由于使用CNN或视觉Transformer而存在感受野有限或二次复杂度问题。本文首次研究状态空间模型在DA PCC中的适应性，发现直接应用SSMs会遇到挑战：直接将3D点云序列化为1D序列会破坏目标域的空间拓扑和局部几何特征。此外，忽略领域无关表征设计会阻碍适应性能。为解决这些问题，我们提出DAPointMamba框架，具有跨域强适应性、全局感受场和高效线性复杂度优势。它包含三个新颖模块：跨域补丁级扫描引入补丁级几何对应关系，实现有效局部对齐；跨域空间SSM对齐基于跨域相似性调制补丁特征，加强空间一致性；跨域通道SSM对齐通过交错和排列特征通道主动解决全局语义差距。在合成和真实世界基准上的大量实验表明，我们的DAPointMamba以更少的计算复杂度和推理延迟优于最先进方法

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决域自适应点云补全(DA PCC)问题，即如何让模型在已标注的源域数据上训练后，能够很好地适应未标注的目标域数据。这个问题在现实中很重要，因为点云补全是自动驾驶、机器人和虚拟现实等应用的基础任务，而不同传感器、不同场景会导致数据分布差异，现有方法在跨域应用时性能显著下降。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先认识到现有CNN和Transformer方法的局限性，然后发现Mamba模型在长序列建模和计算效率方面的优势。他们发现直接将Mamba应用于点云补全会破坏空间拓扑和缺乏域不变特征设计。因此，他们设计了三个关键模块来解决这些问题。该方法借鉴了Mamba的序列建模能力、点云补全领域的前沿方法和域自适应技术中的对齐策略，并基于PointMamba的改进模块作为基础架构。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是通过三个专门设计的模块解决域自适应点云补全中的空间和语义不一致问题：1)跨域块级扫描(CDPS)确保空间对应关系；2)跨域空间SSM对齐(CDSEA)解决细粒度空间不一致；3)跨域通道SSM对齐(CDCA)解决全局语义不一致。整体流程包括：数据预处理(共享坐标归一化和Z-order序列化)、特征提取(使用改进的Mamba块)、空间对齐(CDSA模块)、语义对齐(CDCA模块)，以及结合补全损失和对齐损失的优化过程。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)首次将Mamba模型应用于域自适应点云补全；2)提出跨域块级扫描(CDPS)确保空间对应；3)设计跨域空间SSM对齐(CDSA)解决细粒度空间不一致；4)设计跨域通道SSM对齐(CDCA)解决全局语义不一致。相比之前的工作，该方法具有更大的感受野和更高的计算效率(相比CNN)，避免了二次计算复杂度问题(相比Transformer)，并专门针对域自适应任务进行了设计，解决了空间拓扑被破坏的问题，在多个基准测试上取得了更好的性能。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'DAPointMamba首次将Mamba模型引入域自适应点云补全任务，通过创新的跨域块级扫描、空间和通道对齐机制，实现了在保持全局感受野和线性计算效率的同时，有效解决跨域点云补全中的几何和语义不一致问题。'}


### 论文摘要

Domain adaptive point cloud completion (DA PCC) aims to narrow the geometric and semantic discrepancies between the labeled source and unlabeled target domains. Existing methods either suffer from limited receptive fields or quadratic complexity due to using CNNs or vision Transformers. In this paper, we present the first work that studies the adaptability of State Space Models (SSMs) in DA PCC and find that directly applying SSMs to DA PCC will encounter several challenges: directly serializing 3D point clouds into 1D sequences often disrupts the spatial topology and local geometric features of the target domain. Besides, the overlook of designs in the learning domain-agnostic representations hinders the adaptation performance. To address these issues, we propose a novel framework, DAPointMamba for DA PCC, that exhibits strong adaptability across domains and has the advantages of global receptive fields and efficient linear complexity. It has three novel modules. In particular, Cross-Domain Patch-Level Scanning introduces patch-level geometric correspondences, enabling effective local alignment. Cross-Domain Spatial SSM Alignment further strengthens spatial consistency by modulating patch features based on cross-domain similarity, effectively mitigating fine-grained structural discrepancies. Cross-Domain Channel SSM Alignment actively addresses global semantic gaps by interleaving and aligning feature channels. Extensive experiments on both synthetic and real-world benchmarks demonstrate that our DAPointMamba outperforms state-of-the-art methods with less computational complexity and inference latency.

---

## 32. Rethinking the Encoding and Annotating of 3D Bounding Box: Corner-Aware 3D Object Detection from Point Clouds

**论文链接:** [http://arxiv.org/abs/2511.17619v1](http://arxiv.org/abs/2511.17619v1)

**作者:** Qinghao Meng, Junbo Yin, Jianbing Shen, Yunde Jia

**发布时间:** 2025-11-18

**备注:** 8 pages, 5 figures, 2 tables

### GPT解析

### 总结

本文提出了一种基于角点对齐的3D目标检测方法，解决了传统中心对齐回归方法的不稳定性问题。

### 背景

基于LiDAR的3D目标检测中，中心对齐回归仍是主流方法，但由于LiDAR点云的前表面偏向特性，物体中心常位于鸟瞰图中的稀疏或空区域，导致边界框预测存在噪声且不准确。

### 目的

解决中心对齐回归方法的不稳定性问题，提出一种更稳定的边界框表示方法。

### 方法

提出角点对齐回归，将预测目标从不稳定的中心转移到位于密集、可观测区域的几何信息丰富的角点；利用角点间的几何约束和图像2D框，从角点标注中恢复3D边界框的部分参数，实现弱监督范式；设计了一个简单有效的角点感知检测头，可集成到现有检测器中。

### 主要发现

在KITTI数据集上，相比基于中心的方法提高了3.5%的平均精度；仅使用BEV角点点击，就能达到全监督准确度的83%。

### 结论

角点感知回归策略能有效提高3D目标检测的准确性和稳定性，同时减少对完整3D标注的依赖。

### 翻译

基于LiDAR的3D目标检测中，中心对齐回归仍然占主导地位，但它存在根本的不稳定性：由于LiDAR点云的前表面偏向特性，物体中心常位于鸟瞰图的稀疏或空区域，导致边界框预测存在噪声且不准确。为克服这一局限，我们重新审视边界框表示方法，提出了角点对齐回归，将预测目标从不稳定的中心转移到位于密集、可观测区域的具有几何信息的角点。利用角点之间的固有几何约束和图像2D框，可以从角点标注中恢复3D边界框的部分参数，实现无需完整3D标签的弱监督范式。我们设计了一个简单有效的角点感知检测头，可插入到现有检测器中。KITTI实验表明，我们的方法相比基于中心的基线提高了3.5%的平均精度，且仅使用BEV角点点击就达到了全监督准确度的83%，证明了我们角点感知回归策略的有效性。


### 论文摘要

Center-aligned regression remains dominant in LiDAR-based 3D object detection, yet it suffers from fundamental instability: object centers often fall in sparse or empty regions of the bird's-eye-view (BEV) due to the front-surface-biased nature of LiDAR point clouds, leading to noisy and inaccurate bounding box predictions. To circumvent this limitation, we revisit bounding box representation and propose corner-aligned regression, which shifts the prediction target from unstable centers to geometrically informative corners that reside in dense, observable regions. Leveraging the inherent geometric constraints among corners and image 2D boxes, partial parameters of 3D bounding boxes can be recovered from corner annotations, enabling a weakly supervised paradigm without requiring complete 3D labels. We design a simple yet effective corner-aware detection head that can be plugged into existing detectors. Experiments on KITTI show our method improves performance by 3.5% AP over center-based baseline, and achieves 83% of fully supervised accuracy using only BEV corner clicks, demonstrating the effectiveness of our corner-aware regression strategy.

---

## 33. 论文ID: 2511.19057v1

**错误:** 无法读取论文信息 - [Errno 2] No such file or directory: 'storage/result/2511.19057v1.json'

---

## 34. CubeletWorld: A New Abstraction for Scalable 3D Modeling

**论文链接:** [http://arxiv.org/abs/2511.17664v1](http://arxiv.org/abs/2511.17664v1)

**作者:** Azlaan Mustafa Samad, Hoang H. Nguyen, Lukas Berg, Henrik Müller, Yuan Xue, Daniel Kudenko, Zahra Ahmadi

**发布时间:** 2025-11-21

**备注:** 10 pages, 5 figures

### GPT解析

### 总结

本文介绍了CubeletWorld，一个用于表示和分析城市环境的新框架，通过离散化3D网格空间单元(cubelets)整合城市异构数据，支持隐私保护的城市建模和多种下游任务。

### 背景

现代城市产生大量异构数据，从基础设施地图到移动日志和卫星图像，但将这些数据源整合为连贯的空间模型仍面临挑战。现有以代理为中心的方法依赖直接环境感知，限制了可扩展性并引发隐私问题。

### 目的

提出一种新的城市环境表示和分析框架，解决现有方法的局限性，提供隐私保护的建模方式，支持各种城市任务，并提高可扩展性和泛化能力。

### 方法

通过离散化3D网格空间单元(cubelets)将多样化数据信号嵌入到局部状态中，提出CubeletWorld状态预测任务，使用包含城市元素的真实数据集预测cubelet状态，探索适合该设置的核心模型，分析空间粒度增加带来的挑战。

### 主要发现

以cubelet为中心的方法专注于推断空间单元级别状态，提高了跨区域泛化能力并改善隐私合规性。结果表明CubeletWorld为从复杂城市数据中学习提供了灵活可扩展的框架。

### 结论

CubeletWorld能有效整合城市异构数据，提供隐私保护的城市环境建模方法，支持多种下游应用，并在提高可扩展性和泛化能力方面表现出色。代码和数据集已公开发布。

### 翻译

现代城市产生大量异构数据流，从基础设施地图到移动日志和卫星图像。然而，将这些数据源整合为用于规划和预测的连贯空间模型仍然是一个重大挑战。现有的以代理为中心的方法通常依赖于直接环境感知，这限制了可扩展性并引发隐私问题。本文介绍了CubeletWorld，一种通过称为cubelets的离散化3D网格空间单元来表示和分析城市环境的新颖框架。这种抽象通过将多样化数据信号嵌入到局部cubelet状态中，实现了隐私保护的建模。CubeletWorld支持规划、导航和占用预测等下游任务，而无需代理驱动的感知。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决如何将现代城市产生的海量异构数据（如基础设施地图、移动日志和卫星图像）整合成连贯的空间模型用于规划和预测的问题。这个问题很重要，因为现有以智能体为中心的方法依赖直接环境感知，限制了可扩展性并引发隐私问题，而城市规划、占用预测和应急响应等应用需要理解和预测复杂城市环境的动态状态。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者观察到现有方法通常围绕单个智能体进行预测，而他们提出全局3D网格表示方法。他们将城市空间离散化为均匀体积单元(cubelets)，编码相关属性。借鉴了CNN-LSTM模型处理时空特征，以及A3T-GCN模型结合图卷积、门控循环单元和注意力机制。还参考了boids模拟中的分离、对齐和内聚行为规则，以及语义占用预测、轨迹预测等相关领域工作。', '这个方法的核心思想是什么？整体实现流程是怎样的？': "核心思想是将3D空间离散化为称为'cubelets'的统一体积单元，每个单元编码相关属性，将分散传感器数据规则化以支持推理，支持多分辨率，并通过抽象个体轨迹保护隐私。实现流程包括：1)收集静态地形数据和动态boids坐标；2)将环境划分为cubelets；3)将原始数据转换为cubelet表示；4)构建CNN-LSTM或A3T-GCN模型；5)使用历史数据训练并预测未来cubelet状态；6)评估模型性能。", '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：提出CubeletWorld统一3D表示方法；定义CubeletWorld状态预测任务；引入CubeletWorld Boids数据集；提供多分辨率能力；通过抽象个体轨迹保护隐私。相比之前工作的不同之处：与以智能体为中心的方法不同，提供统一空间视角增强可解释性和隐私；与现有3D占用预测模型不同，侧重空间单元级推断；与基于视觉的自动驾驶研究不同，聚合异构数据到全局3D网格；与轨迹预测方法不同，抽象个体信息进行集体预测；与UAV视觉研究相比，不依赖受限访问数据集。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'CubeletWorld提出了一种新颖的3D离散化网格表示方法，通过将城市空间划分为统一的体积单元，有效整合多源异构数据，实现了隐私保护的占用预测和环境动态分析，同时支持多分辨率建模，为城市规划、环境监测和应急响应等应用提供了可扩展的解决方案。'}


### 论文摘要

Modern cities produce vast streams of heterogeneous data, from infrastructure maps to mobility logs and satellite imagery. However, integrating these sources into coherent spatial models for planning and prediction remains a major challenge. Existing agent-centric methods often rely on direct environmental sensing, limiting scalability and raising privacy concerns. This paper introduces CubeletWorld, a novel framework for representing and analyzing urban environments through a discretized 3D grid of spatial units called cubelets. This abstraction enables privacy-preserving modeling by embedding diverse data signals, such as infrastructure, movement, or environmental indicators, into localized cubelet states. CubeletWorld supports downstream tasks such as planning, navigation, and occupancy prediction without requiring agent-driven sensing. To evaluate this paradigm, we propose the CubeletWorld State Prediction task, which involves predicting the cubelet state using a realistic dataset containing various urban elements like streets and buildings through this discretized representation. We explore a range of modified core models suitable for our setting and analyze challenges posed by increasing spatial granularity, specifically the issue of sparsity in representation and scalability of baselines. In contrast to existing 3D occupancy prediction models, our cubelet-centric approach focuses on inferring state at the spatial unit level, enabling greater generalizability across regions and improved privacy compliance. Our results demonstrate that CubeletWorld offers a flexible and extensible framework for learning from complex urban data, and it opens up new possibilities for scalable simulation and decision support in domains such as socio-demographic modeling, environmental monitoring, and emergency response. The code and datasets can be downloaded from here.

---

## 35. Unified Low-Light Traffic Image Enhancement via Multi-Stage Illumination Recovery and Adaptive Noise Suppression

**论文链接:** [http://arxiv.org/abs/2511.17612v1](http://arxiv.org/abs/2511.17612v1)

**作者:** Siddiqua Namrah

**发布时间:** 2025-11-18

**备注:** Master's thesis, Korea University, 2025

### GPT解析

### 总结

提出了一种完全无监督的多阶段深度学习框架用于低光照交通图像增强，通过分解图像为光照和反射率分量，并使用三个专门模块逐步优化，提高了自动驾驶等系统在低光照条件下的感知可靠性。

### 背景

增强低光照交通图像对于自动驾驶、智能交通和城市监控系统中的可靠感知至关重要，但夜间和昏暗的交通场景常因低光照、噪声、运动模糊、不均匀照明和眩光等问题导致能见度差。

### 目的

解决低光照交通场景中的图像增强问题，提高目标检测和场景理解等任务的可靠性。

### 方法

提出一个完全无监督的多阶段深度学习框架，模型将图像分解为光照和反射率分量，由三个专门模块逐步优化：光照适应模块进行全局和局部亮度校正；反射率恢复模块使用空间-通道注意力进行噪声抑制和结构细节恢复；过曝补偿模块用于重建饱和区域和平衡场景亮度。网络使用自监督重建、反射率平滑、感知一致性和领域感知正则化损失进行训练，无需成对的真值图像。

### 主要发现

在通用和特定于交通的数据集上的实验表明，该方法在定量指标（PSNR、SSIM、LPIPS、NIQE）和定性视觉质量方面优于最先进的方法，能够提高能见度，保留结构，并改善现实世界低光照交通场景中下游感知的可靠性。

### 结论

所提出的完全无监督多阶段深度学习框架有效解决了低光照交通图像增强问题，提高了自动驾驶等系统在低光照条件下的感知可靠性。

### 翻译

增强低光照交通图像对于自动驾驶、智能交通和城市监控系统中的可靠感知至关重要。夜间和昏暗的交通场景通常由于低光照、噪声、运动模糊、不均匀照明以及车辆前灯或路灯的眩光而能见度差，这妨碍了目标检测和场景理解等任务。为应对这些挑战，我们提出了一种完全无监督的多阶段深度学习框架用于低光照交通图像增强。该模型将图像分解为光照和反射率分量，由三个专门模块逐步优化：(1)光照适应，用于全局和局部亮度校正；(2)反射率恢复，使用空间-通道注意力进行噪声抑制和结构细节恢复；(3)过曝补偿，用于重建饱和区域和平衡场景亮度。网络使用自监督重建、反射率平滑、感知一致性和领域感知正则化损失进行训练，无需成对的真值图像。在通用和特定于交通的数据集上的实验表明，在定量指标（PSNR、SSIM、LPIPS、NIQE）和定性视觉质量方面，该方法优于最先进的方法。我们的方法提高了能见度，保留了结构，并改善了现实世界低光照交通场景中下游感知的可靠性。


### 论文摘要

Enhancing low-light traffic images is crucial for reliable perception in autonomous driving, intelligent transportation, and urban surveillance systems. Nighttime and dimly lit traffic scenes often suffer from poor visibility due to low illumination, noise, motion blur, non-uniform lighting, and glare from vehicle headlights or street lamps, which hinder tasks such as object detection and scene understanding. To address these challenges, we propose a fully unsupervised multi-stage deep learning framework for low-light traffic image enhancement. The model decomposes images into illumination and reflectance components, progressively refined by three specialized modules: (1) Illumination Adaptation, for global and local brightness correction; (2) Reflectance Restoration, for noise suppression and structural detail recovery using spatial-channel attention; and (3) Over-Exposure Compensation, for reconstructing saturated regions and balancing scene luminance. The network is trained using self-supervised reconstruction, reflectance smoothness, perceptual consistency, and domain-aware regularization losses, eliminating the need for paired ground-truth images. Experiments on general and traffic-specific datasets demonstrate superior performance over state-of-the-art methods in both quantitative metrics (PSNR, SSIM, LPIPS, NIQE) and qualitative visual quality. Our approach enhances visibility, preserves structure, and improves downstream perception reliability in real-world low-light traffic scenarios.

---

## 36. NNGPT: Rethinking AutoML with Large Language Models

**论文链接:** [http://arxiv.org/abs/2511.20333v1](http://arxiv.org/abs/2511.20333v1)

**作者:** Roman Kochnev, Waleed Khalid, Tolgay Atinc Uzun, Xi Zhang, Yashkumar Sanjaybhai Dhameliya, Furui Qin, Chandini Vysyaraju, Raghuvir Duvvuri, Avi Goyal, Dmitry Ignatov, Radu Timofte

**发布时间:** 2025-11-25

### GPT解析

### 总结

NNGPT是一个开源框架，将大型语言模型转变为用于神经网络开发的自我改进AutoML引擎，主要用于计算机视觉。它通过生成新模型扩展神经网络数据集，实现基于闭环系统的LLM持续微调。

### 背景

构建自我改进的AI系统仍然是AI领域的基本挑战。

### 目的

提出NNGPT框架，将大型语言模型转变为用于神经网络开发的自我改进AutoML引擎，主要用于计算机视觉。

### 方法

NNGPT通过生成新模型扩展神经网络数据集，实现基于生成、评估和自我改进闭环系统的LLM持续微调。它集成了五个协同的基于LLM的流水线：零样本架构合成、超参数优化(HPO)、代码感知的准确率/早停预测、检索增强的封闭范围PyTorch块合成(NN-RAG)和强化学习。系统建立在LEMUR数据集上，从单个提示发出，验证网络架构、预处理代码和超参数，端到端执行，并从结果中学习。

### 主要发现

NN-RAG在1,289个目标上实现73%的可执行性；3次提示提示在常用数据集上提高准确性；基于哈希的去重节省了数百次运行；单次预测匹配基于搜索的AutoML；在LEMUR上的HPO实现RMSE 0.60，优于Optuna(0.64)；代码感知预测器达到RMSE 0.14，Pearson r=0.78；系统已生成超过5K个验证模型。

### 结论

NNGPT被证明是一个自主的AutoML引擎。接受后，代码、提示和检查点将公开发布，以实现可复制性和促进社区使用。

### 翻译

在AI领域构建自我改进的系统仍然是一个基本挑战。我们提出了NNGPT，一个开源框架，它将大型语言模型转变为用于神经网络开发的自我改进AutoML引擎，主要用于计算机视觉。与之前的框架不同，NNGPT通过生成新模型来扩展神经网络数据集，实现了基于生成、评估和自我改进闭环系统的LLM持续微调。它在一个统一的工作流中集成了五个协同的基于LLM的流水线：零样本架构合成、超参数优化(HPO)、代码感知的准确率/早停预测、检索增强的封闭范围PyTorch块合成(NN-RAG)和强化学习。建立在LEMUR数据集上作为具有可复制指标的审核语料库，NNGPT从单个提示发出，验证网络架构、预处理代码和超参数，端到端执行它们，并从结果中学习。PyTorch适配器使NNGPT与框架无关，实现了强大性能：NN-RAG在1,289个目标上实现73%的可执行性，3次提示提示在常用数据集上提高准确性，基于哈希的去重节省了数百次运行。单次预测匹配基于搜索的AutoML，减少了对多次试验的需求。在LEMUR上的HPO实现RMSE 0.60，优于Optuna(0.64)，而代码感知预测器达到RMSE 0.14，Pearson r=0.78。该系统已经生成了超过5K个验证模型，证明NNGPT是一个自主的AutoML引擎。接受后，代码、提示和检查点将公开发布，以实现可复制性和促进社区使用。


### 论文摘要

Building self-improving AI systems remains a fundamental challenge in the AI domain. We present NNGPT, an open-source framework that turns a large language model (LLM) into a self-improving AutoML engine for neural network development, primarily for computer vision. Unlike previous frameworks, NNGPT extends the dataset of neural networks by generating new models, enabling continuous fine-tuning of LLMs based on closed-loop system of generation, assessment, and self-improvement. It integrates within one unified workflow five synergistic LLM-based pipelines: zero-shot architecture synthesis, hyperparameter optimization (HPO), code-aware accuracy/early-stop prediction, retrieval-augmented synthesis of scope-closed PyTorch blocks (NN-RAG), and reinforcement learning. Built on the LEMUR dataset as an audited corpus with reproducible metrics, NNGPT emits from a single prompt and validates network architecture, preprocessing code, and hyperparameters, executes them end-to-end, and learns from result. The PyTorch adapter makes NNGPT framework-agnostic, enabling strong performance: NN-RAG achieves 73% executability on 1,289 targets, 3-shot prompting boosts accuracy on common datasets, and hash-based deduplication saves hundreds of runs. One-shot prediction matches search-based AutoML, reducing the need for numerous trials. HPO on LEMUR achieves RMSE 0.60, outperforming Optuna (0.64), while the code-aware predictor reaches RMSE 0.14 with Pearson r=0.78. The system has already generated over 5K validated models, proving NNGPT as an autonomous AutoML engine. Upon acceptance, the code, prompts, and checkpoints will be released for public access to enable reproducibility and facilitate community usage.

---

## 37. IrisNet: Infrared Image Status Awareness Meta Decoder for Infrared Small Targets Detection

**论文链接:** [http://arxiv.org/abs/2511.20319v1](http://arxiv.org/abs/2511.20319v1)

**作者:** Xuelin Qian, Jiaming Lu, Zixuan Wang, Wenxuan Wang, Zhongling Huang, Dingwen Zhang, Junwei Han

**发布时间:** 2025-11-25

**备注:** 10pages,5figures

### GPT解析

### 总结

本文提出了IrisNet，一种新颖的元学习框架，用于红外小目标检测，能够根据输入红外图像状态动态调整检测策略，解决了传统静态模式学习在不同场景下的模式漂移问题。

### 背景

红外小目标检测面临低信噪比、复杂背景和缺乏可辨别目标特征的挑战。虽然基于深度学习的编码器-解码器框架已取得进展，但其静态模式学习在不同场景（如日夜变化、天空/海洋/地面区域）下存在模式漂移，限制了鲁棒性。

### 目的

开发IrisNet框架，能够根据输入红外图像状态动态适应检测策略，提高红外小目标检测的鲁棒性和性能。

### 方法

通过图像到解码器的转换器建立红外图像特征与整个解码器参数间的动态映射；将参数化解码器表示为保留分层层相关性的结构化二维张量；利用转换器的自注意力建模层间依赖关系；通过交叉注意力生成自适应解码模式；集成高频成分以补充目标位置和场景边缘信息。

### 主要发现

在NUDT-SIRST、NUAA-SIRST和IRSTD-1K数据集上的实验证明IrisNet具有优越性，实现了最先进的性能。

### 结论

IrisNet能够有效解决红外小目标检测中的挑战，特别是在不同场景下保持鲁棒性，显著提高了检测性能。

### 翻译

红外小目标检测由于低信噪比、复杂背景和缺乏可辨别目标特征而面临重大挑战。虽然基于深度学习的编码器-解码器框架推动了该领域的发展，但它们的静态模式学习在不同场景下（例如，日夜变化、天空/海洋/地面区域）存在模式漂移问题，限制了鲁棒性。为解决这一问题，我们提出了IrisNet，一种新颖的元学习框架，能够根据输入红外图像状态动态调整检测策略。我们的方法通过图像到解码器的转换器建立了红外图像特征与整个解码器参数之间的动态映射。更具体地说，我们将参数化解码器表示为保留分层层相关性的结构化二维张量，并使转换器能够通过自注意力建模层间依赖关系，同时通过交叉注意力生成自适应解码模式。为了进一步增强红外图像的感知能力，我们集成了高频成分以补充目标位置和场景边缘信息。在NUDT-SIRST、NUAA-SIRST和IRSTD-1K数据集上的实验证明了我们IrisNet的优越性，实现了最先进的性能。


### 论文摘要

Infrared Small Target Detection (IRSTD) faces significant challenges due to low signal-to-noise ratios, complex backgrounds, and the absence of discernible target features. While deep learning-based encoder-decoder frameworks have advanced the field, their static pattern learning suffers from pattern drift across diverse scenarios (\emph{e.g.}, day/night variations, sky/maritime/ground domains), limiting robustness. To address this, we propose IrisNet, a novel meta-learned framework that dynamically adapts detection strategies to the input infrared image status. Our approach establishes a dynamic mapping between infrared image features and entire decoder parameters via an image-to-decoder transformer. More concretely, we represent the parameterized decoder as a structured 2D tensor preserving hierarchical layer correlations and enable the transformer to model inter-layer dependencies through self-attention while generating adaptive decoding patterns via cross-attention. To further enhance the perception ability of infrared images, we integrate high-frequency components to supplement target-position and scene-edge information. Experiments on NUDT-SIRST, NUAA-SIRST, and IRSTD-1K datasets demonstrate the superiority of our IrisNet, achieving state-of-the-art performance.

---

## 38. History-Augmented Contrastive Meta-Learning for Unsupervised Blind Super-Resolution of Planetary Remote Sensing Images

**论文链接:** [http://arxiv.org/abs/2511.20045v1](http://arxiv.org/abs/2511.20045v1)

**作者:** Huijia Zhao, Jie Lu, Yunqing Jiang, Xiao-Ping Lu, Kaichang Di

**发布时间:** 2025-11-25

**备注:** 13pages

### GPT解析

### 总结

本文提出了一种名为HACBSR的无监督盲超分辨率框架，用于处理行星遥感图像的退化问题，无需地面真实图像和外部核先验。

### 背景

行星遥感图像受到成像环境和硬件约束引起的多样且未知的退化影响，这些因素限制了图像质量，并由于缺乏地面真实图像而阻碍了有监督的盲超分辨率方法的应用。

### 目的

开发一种无需地面真实图像和外部核先验的无监督盲超分辨率框架，解决行星遥感图像的超分辨率问题。

### 方法

HACBSR包含两个主要组件：(1)具有核相似性控制的对比核采样机制，减轻高斯采样中的分布偏差；(2)历史增强对比学习，使用历史模型生成负样本，实现较少贪婪的优化，并在没有地面真实图像的情况下诱导强凸性。同时引入了Ceres-50数据集，包含多样的地质特征和模拟的退化模式。

### 主要发现

实验表明，HACBSR在多个上采样因子上与最先进的无监督方法相比具有竞争力，代码和数据集已公开可用。

### 结论

HACBSR是一种有效的无监督盲超分辨率方法，适用于行星遥感图像处理，能够有效处理行星图像的退化问题而无需地面真实图像和外部核先验。

### 翻译

行星遥感图像受到成像环境和硬件约束引起的多样且未知的退化影响。这些因素限制了图像质量，并由于缺乏地面真实图像而阻碍了有监督的盲超分辨率方法。本文提出了历史增强对比盲超分辨率方法，一种无需地面真实图像和外部核先验的无监督盲超分辨率框架。HACBSR包含两个组件：(1)具有核相似性控制的对比核采样机制，减轻高斯采样中的分布偏差；(2)历史增强对比学习，使用历史模型生成负样本，实现较少贪婪的优化，并在没有地面真实图像的情况下诱导强凸性。历史增强对比学习的收敛分析见附录。为支持行星应用中的评估，我们引入了Ceres-50数据集，包含多样的地质特征和模拟的退化模式。实验表明，与最先进的无监督方法相比，HACBSR在多个上采样因子上具有竞争力。代码可在https://github.com/2333repeat/HACBSR获取，数据集可在https://github.com/2333repeat/Ceres-50获取。


### 论文摘要

Planetary remote sensing images are affected by diverse and unknown degradations caused by imaging environments and hardware constraints. These factors limit image quality and hinder supervised blind super-resolution due to the lack of ground-truth images. This work presents History-Augmented Contrastive Blind Super-Resolution (HACBSR), an unsupervised framework for blind super-resolution that operates without ground-truth images and external kernel priors. HACBSR comprises two components: (1) a contrastive kernel sampling mechanism with kernel similarity control to mitigate distribution bias from Gaussian sampling, and (2) a history-augmented contrastive learning that uses historical models to generate negative samples to enable less greedy optimization and to induce strong convexity without ground-truth. A convergence analysis of the history-augmented contrastive learning is given in the Appendix. To support evaluation in planetary applications, we introduce Ceres-50, a dataset with diverse geological features simulated degradation patterns. Experiments show that HACBSR achieves competitive performance compared with state-of-the-art unsupervised methods across multiple upscaling factors. The code is available at https://github.com/2333repeat/HACBSR, and the dataset is available at https://github.com/2333repeat/Ceres-50.

---

## 39. Adaptivity and Universality: Problem-dependent Universal Regret for Online Convex Optimization

**论文链接:** [http://arxiv.org/abs/2511.19937v1](http://arxiv.org/abs/2511.19937v1)

**作者:** Peng Zhao, Yu-Hu Yan, Hang Yu, Zhi-Hua Zhou

**发布时间:** 2025-11-25

### GPT解析

### 总结

本文提出了一种名为UniGrad的新型通用在线学习方法，实现了通用性和适应性的统一，同时提供了UniGrad.Correct和UniGrad.Bregman两种实现方式，并进一步提出了计算效率更高的UniGrad++版本。

### 背景

通用在线学习旨在无需在线函数曲率先验知识的情况下实现最优遗憾保证。现有方法已建立最小最大最优遗憾界限，单个算法可同时为不同类型的函数提供遗憾保证，但缺乏与问题相关的适应性，特别是没有提供与梯度变化量V_T相关的遗憾界限。

### 目的

开发一种同时具有通用性和适应性的方法，实现与梯度变化量V_T相关的遗憾保证，并提高计算效率。

### 方法

提出UniGrad方法，包含UniGrad.Correct和UniGrad.Bregman两种实现方式，采用元算法和基础学习器。为提高计算效率，进一步提出UniGrad++，通过代理优化将梯度查询减少到每轮仅1次。

### 主要发现

UniGrad方法实现了通用遗憾保证，适应梯度变化，同时为强凸函数和指数凹函数提供特定的遗憾界限。UniGrad.Correct为凸函数提供√V_T log V_T级别的遗憾界限，而UniGrad.Bregman提供最优的√V_T级别遗憾界限。UniGrad++保留了遗憾保证同时提高了计算效率。

### 结论

UniGrad方法成功实现了通用在线学习的通用性和适应性统一，为不同类型的函数提供了与梯度变化量相关的遗憾保证，并通过UniGrad++提高了计算效率。

### 翻译

通用在线学习旨在无需在线函数曲率先验知识的情况下实现最优遗憾保证。现有方法已建立最小最大最优遗憾界限，单个算法可同时为凸函数实现平方根T级别的遗憾，为指数凹函数实现维度乘以对数T级别的遗憾，为强凸函数实现对数T级别的遗憾，其中T为轮数，d为可行域维度。然而，这些方法仍缺乏与问题相关的适应性。特别是，没有通用方法提供与梯度变化量V_T相关的遗憾界限，V_T是随机优化和游戏中快速收敛率应用的关键量。本文引入UniGrad，一种实现通用性和适应性的新方法，包含UniGrad.Correct和UniGrad.Bregman两种实现。两种方法都实现了适应梯度变化的通用遗憾保证，同时为强凸函数实现对数V_T级别的遗憾，为指数凹函数实现维度乘以对数V_T级别的遗憾。对于凸函数，遗憾界限不同：UniGrad.Correct实现根号V_T乘以对数V_T级别的界限，同时保留在线游戏中快速收敛至关重要的RVU属性；UniGrad.Bregman通过新颖设计实现最优的根号V_T级别遗憾界限。两种方法采用具有对数T级别基础学习器的元算法，自然需要每轮对数T级别的梯度查询。为提高计算效率，我们引入UniGrad++，在保留遗憾保证的同时将梯度查询减少到每轮仅1次，通过代理优化实现。我们还提供了各种含义。


### 论文摘要

Universal online learning aims to achieve optimal regret guarantees without requiring prior knowledge of the curvature of online functions. Existing methods have established minimax-optimal regret bounds for universal online learning, where a single algorithm can simultaneously attain $\mathcal{O}(\sqrt{T})$ regret for convex functions, $\mathcal{O}(d \log T)$ for exp-concave functions, and $\mathcal{O}(\log T)$ for strongly convex functions, where $T$ is the number of rounds and $d$ is the dimension of the feasible domain. However, these methods still lack problem-dependent adaptivity. In particular, no universal method provides regret bounds that scale with the gradient variation $V_T$, a key quantity that plays a crucial role in applications such as stochastic optimization and fast-rate convergence in games. In this work, we introduce UniGrad, a novel approach that achieves both universality and adaptivity, with two distinct realizations: UniGrad.Correct and UniGrad.Bregman. Both methods achieve universal regret guarantees that adapt to gradient variation, simultaneously attaining $\mathcal{O}(\log V_T)$ regret for strongly convex functions and $\mathcal{O}(d \log V_T)$ regret for exp-concave functions. For convex functions, the regret bounds differ: UniGrad.Correct achieves an $\mathcal{O}(\sqrt{V_T \log V_T})$ bound while preserving the RVU property that is crucial for fast convergence in online games, whereas UniGrad.Bregman achieves the optimal $\mathcal{O}(\sqrt{V_T})$ regret bound through a novel design. Both methods employ a meta algorithm with $\mathcal{O}(\log T)$ base learners, which naturally requires $\mathcal{O}(\log T)$ gradient queries per round. To enhance computational efficiency, we introduce UniGrad++, which retains the regret while reducing the gradient query to just $1$ per round via surrogate optimization. We further provide various implications.

---

## 40. MicroSims: A Framework for AI-Generated, Scalable Educational Simulations with Universal Embedding and Adaptive Learning Support

**论文链接:** [http://arxiv.org/abs/2511.19864v1](http://arxiv.org/abs/2511.19864v1)

**作者:** Valerie Lockhart, Dan McCreary, Troy A. Peterson

**发布时间:** 2025-11-25

**备注:** 42 pages, 4 figures

### GPT解析

### 总结

本文介绍了MicroSims，一个用于创建轻量级、交互式教育模拟的新型框架，利用人工智能快速生成，可在各种数字学习平台上嵌入，无需编程知识即可定制。

### 背景

教育模拟长期以来被认可为增强学习成果的有力工具，但它们的传统创建需要大量资源和技术专业知识，限制了广泛应用。

### 目的

开发MicroSims框架，解决教育模拟创建中的成本、技术复杂性和平台依赖性问题，使教育工作者能够按需创建定制化、符合课程标准的模拟。

### 方法

提出一个包含设计原则、技术架构、元数据标准和开发工作流程的全面框架，基于物理学教育研究和STEM学科元分析的经验研究。

### 主要发现

交互式模拟与传统教学相比可以提高概念理解能力30-40%，MicroSims在提供这些益处的同时，解决了成本、技术复杂性和平台依赖性等障碍。

### 结论

这项工作对教育公平有重要意义，使全球教育工作者能够以低成本创建智能交互式教科书，按需生成定制化模拟，并讨论了基于MicroSims构建的AI驱动自适应学习系统的未来方向。

### 翻译

教育模拟长期以来被认可为增强学习成果的有力工具，但它们的创建传统上需要大量资源和技术专业知识。本文介绍了MicroSims，这是一个用于创建轻量级、交互式教育模拟的新型框架，可以利用人工智能快速生成，可在所有数字学习平台上普遍嵌入，并且无需编程知识即可轻松定制。MicroSims位于三个关键创新交叉点的独特位置：(1)支持AI辅助生成的标准化设计模式，(2)提供通用嵌入和沙盒安全性的基于iframe的架构，(3)支持定制和教学透明度的透明、可修改代码。我们提出了一个包含设计原则、技术架构、元数据标准和开发工作流程的全面框架。借鉴物理学教育研究和STEM学科元分析的经验研究，我们证明交互式模拟与传统教学相比可以提高概念理解能力30-40%。MicroSims在提供这些益处的同时，解决了成本、技术复杂性和平台依赖性等长期存在的障碍。这项工作对教育公平有重要意义，以及低成本智能交互式教科书，使全球教育工作者能够按需创建定制化、符合课程标准的模拟。


### 论文摘要

Educational simulations have long been recognized as powerful tools for enhancing learning outcomes, yet their creation has traditionally required substantial resources and technical expertise. This paper introduces MicroSims a novel framework for creating lightweight, interactive educational simulations that can be rapidly generated using artificial intelligence, universally embedded across digital learning platforms, and easily customized without programming knowledge. MicroSims occupy a unique position at the intersection of three key innovations: (1) standardized design patterns that enable AI-assisted generation, (2) iframe-based architecture that provides universal embedding and sandboxed security, and (3) transparent, modifiable code that supports customization and pedagogical transparency. We present a comprehensive framework encompassing design principles, technical architecture, metadata standards, and development workflows. Drawing on empirical research from physics education studies and meta-analyses across STEM disciplines, we demonstrate that interactive simulations can improve conceptual understanding by up to 30-40\% compared to traditional instruction. MicroSims extend these benefits while addressing persistent barriers of cost, technical complexity, and platform dependence. This work has significant implications for educational equity, and low-cost intelligent interactive textbooks that enabling educators worldwide to create customized, curriculum-aligned simulations on demand. We discuss implementation considerations, present evidence of effectiveness, and outline future directions for AI-powered adaptive learning systems built on the MicroSim foundation.

---

## 41. Vision--Language Enhanced Foundation Model for Semi-supervised Medical Image Segmentation

**论文链接:** [http://arxiv.org/abs/2511.19759v1](http://arxiv.org/abs/2511.19759v1)

**作者:** Jiaqi Guo, Mingzhen Li, Hanyu Su, Santiago López, Lexiaozi Fan, Daniel Kim, Aggelos Katsaggelos

**发布时间:** 2025-11-24

### GPT解析

### 总结

该研究提出了视觉语言增强的半监督分割助手(VESSA)，将视觉-语言模型的基础级视觉-语义理解融入半监督学习框架，显著提高了医学图像分割的准确性，特别是在标注数据有限的情况下。

### 背景

半监督学习(SSL)已成为医学图像分割的有效方法，可减少对大量专家标注的依赖；视觉-语言模型(VLMs)在各种视觉领域展示了强大的泛化和少样本能力。

### 目的

将基于VLM的分割集成到半监督医学图像分割中，引入VESSA系统，将基础级的视觉-语义理解融入SSL框架。

### 方法

包含两个阶段：第一阶段，训练VESSA作为参考引导的分割助手，使用包含黄金标准样本的模板库；第二阶段，将VESSA集成到SSL框架中，与学生模型进行动态交互，学生预测作为反馈提示给VESSA，生成更高质量的伪标签和更强指导。

### 主要发现

在多个分割数据集和领域进行的广泛实验表明，VESSA增强的SSL显著提高了分割准确性，在极有限的标注条件下优于最先进的基线方法。

### 结论

VESSA-augmented SSL在医学图像分割领域表现出色，特别是在标注数据有限的情况下，为医学图像分析提供了一种有效的解决方案。

### 翻译

半监督学习(SSL)已成为医学图像分割的有效范例，减少了对大量专家标注的依赖。同时，视觉-语言模型(VLMs)已在各种视觉领域展示了强大的泛化和少样本能力。在本工作中，我们将基于VLM的分割集成到半监督医学图像分割中，通过引入视觉语言增强的半监督分割助手(VESSA)，将基础级的视觉-语义理解融入SSL框架。我们的方法包含两个阶段。在第一阶段，VLM增强的分割基础模型VESSA使用包含黄金标准样本的模板库训练为参考引导的分割助手，模拟从有限标注数据中学习。给定输入-模板对，VESSA执行视觉特征匹配，从样本分割中提取代表性的语义和空间线索，为受SAM2启发的掩码解码器生成结构化提示以产生分割掩码。在第二阶段，VESSA集成到最先进的SSL框架中，能够与学生模型进行动态交互：随着学生预测变得更加精细，它们被反馈给VESSA作为提示，允许它生成更高质量的伪标签和更强的指导。在多个分割数据集和领域进行的广泛实验表明，VESSA增强的SSL显著提高了分割准确性，在极有限的标注条件下优于最先进的基线方法。


### 论文摘要

Semi-supervised learning (SSL) has emerged as an effective paradigm for medical image segmentation, reducing the reliance on extensive expert annotations. Meanwhile, vision-language models (VLMs) have demonstrated strong generalization and few-shot capabilities across diverse visual domains. In this work, we integrate VLM-based segmentation into semi-supervised medical image segmentation by introducing a Vision-Language Enhanced Semi-supervised Segmentation Assistant (VESSA) that incorporates foundation-level visual-semantic understanding into SSL frameworks. Our approach consists of two stages. In Stage 1, the VLM-enhanced segmentation foundation model VESSA is trained as a reference-guided segmentation assistant using a template bank containing gold-standard exemplars, simulating learning from limited labeled data. Given an input-template pair, VESSA performs visual feature matching to extract representative semantic and spatial cues from exemplar segmentations, generating structured prompts for a SAM2-inspired mask decoder to produce segmentation masks. In Stage 2, VESSA is integrated into a state-of-the-art SSL framework, enabling dynamic interaction with the student model: as student predictions become more refined, they are fed back to VESSA as prompts, allowing it to generate higher-quality pseudo-labels and stronger guidance. Extensive experiments across multiple segmentation datasets and domains show that VESSA-augmented SSL significantly enhances segmentation accuracy, outperforming state-of-the-art baselines under extremely limited annotation conditions.

---

## 42. Efficient Transferable Optimal Transport via Min-Sliced Transport Plans

**论文链接:** [http://arxiv.org/abs/2511.19741v1](http://arxiv.org/abs/2511.19741v1)

**作者:** Xinran Liu, Elaheh Akbari, Rocio Diaz Martin, Navid NaderiAlizadeh, Soheil Kolouri

**发布时间:** 2025-11-24

### GPT解析

### 总结

该论文研究了最小切片传输计划(min-STP)框架中优化切片器的可转移性问题，证明了在数据分布轻微扰动下，优化切片器能够有效转移到新分布，并提出了小批量形式化以提高可扩展性。

### 背景

最优传输(OT)是寻找分布之间对应关系的有力框架，可用于计算机视觉中的形状分析、图像生成和多模态任务，但其计算成本阻碍了可扩展性。基于切片的传输计划通过利用一维OT问题的闭式解来减少计算成本。

### 目的

探究学习到的最优切片器在分布转移下是否能转移到新的分布对，理解这种可转移性对于处理 evolving data 或在相关分布间重复进行OT计算的情况至关重要。

### 方法

研究最小切片传输计划(min-STP)框架，理论上证明优化切片器在数据分布的微小扰动下保持接近，引入min-STP的小批量形式化以提高可扩展性，并提供其准确性的统计保证。

### 主要发现

理论上证明了优化切片器在数据分布轻微扰动下保持接近，使相关任务间的高效转移成为可能；实验证明可转移的min-STP实现了强有力的一次性匹配性能，并促进了点云对齐和基于流的生成模型的分摊训练。

### 结论

基于切片的传输计划可有效减少计算成本；优化切片器可以在分布轻微变化的情况下有效转移到新分布；可转移的min-STP方法在多种计算机视觉任务中表现良好。

### 翻译

最优传输(OT)为寻找分布之间的对应关系提供了一个强大的框架，并解决了计算机视觉各个领域（包括形状分析、图像生成和多模态任务）中的匹配和对齐问题。然而，OT的计算成本阻碍了其可扩展性。基于切片的传输计划最近显示出通过利用一维OT问题的闭式解来减少计算成本的潜力。这些方法优化一维投影（切片）以获得在周围空间最小化传输成本的条件传输计划。虽然这些方法效率高，但它们留下了一个问题：学习到的最优切片器是否能在分布转移下转移到新的分布对。理解这种可转移性对于处理 evolving data 或在相关分布间重复进行OT计算的情况至关重要。在本文中，我们研究了最小切片传输计划(min-STP)框架，并研究了优化切片器的可转移性：在一个分布对上训练的切片器能否为新的、未见过的分布对产生有效的传输计划？理论上，我们证明了在数据分布的微小扰动下，优化切片器保持接近，使相关任务间的高效转移成为可能。为了进一步提高可扩展性，我们引入了min-STP的小批量形式化，并提供了其准确性的统计保证。实验上，我们证明了可转移的min-STP实现了强有力的一次性匹配性能，并促进了点云对齐和基于流的生成模型的分摊训练。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决最优传输(Optimal Transport, OT)的计算效率问题以及其解决方案的可转移性问题。最优传输是一种强大的框架，用于寻找两个分布之间的对应关系，在计算机视觉、自然语言处理和生物学等领域有广泛应用。然而，OT的计算成本很高，复杂度随样本数量增加呈立方级增长，限制了其在大规模数据上的应用。此外，现有方法无法在相似分布对之间重用已学习的信息，导致在需要处理一系列逐渐变化的分布时效率低下。这个问题在发育生物学等领域尤为重要，其中细胞群逐渐进化，连续分布间只有微小差异。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先认识到现有基于切片的传输计划方法虽高效但缺乏可转移性保证。他们借鉴了切片最优传输(Sliced OT)框架和广义切片传输计划(STP)的思想，引入拉普拉斯分布扰动来增强方法的稳定性，这种平滑技术参考了LapSum方法。作者还参考了微分广义切片Wasserstein计划(DGSWP)和Set-Transformer架构。在此基础上，作者增加了理论保证(可转移性)，改进了实现(使用LapSum替代DGSWP以提高稳定性)，并引入了小批量训练以提高可扩展性。整体设计思路是：通过学习最优切片器函数将高维数据投影到一维空间，利用一维OT问题的闭式解构建高维传输计划，并证明这种切片器在不同但相似的分布对间具有可转移性。', '这个方法的核心思想是什么？整体实现流程是怎样的？': "核心思想是通过学习最优的'切片器'函数将高维数据投影到一维空间，利用一维OT问题的闭式解构建高维传输计划，并证明这种切片器在不同但相似的分布对间具有可转移性，实现摊销训练。整体实现流程：1)使用参数化函数(如神经网络)作为切片器将高维数据投影到一维；2)添加拉普拉斯噪声确保投影是单射的；3)对投影后的点进行软排序；4)构建一维最优传输计划并提升回高维空间；5)最小化传输计划在原始空间中的传输成本；6)使用双分支对称梯度流进行训练，结合软排列矩阵和硬排列矩阵；7)采用小批量训练提高可扩展性；8)将学习到的切片器转移到相似的新任务上作为初始点。", '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点：1)首次从理论上证明了最优切片器在分布发生微小变化时的稳定性；2)使用LapSum方法替代了DGSWP中的高方差梯度估计，计算效率更高(O(n log n))且更稳定；3)提供了小批量训练与完整批次训练之间差距的统计保证；4)提出对称双分支梯度流提高训练稳定性。相比之前工作：1)之前工作主要关注单个OT问题的计算效率，本文探讨了OT解决方案的可转移性；2)采用不同的平滑策略，在计算效率和数值稳定性上都有改进；3)提出的训练策略更稳定，能处理更大规模数据；4)不仅提出新方法，还提供了理论保证；5)展示了在点云对齐和基于流生成模型等多种任务中的应用。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '这篇论文通过理论分析和算法创新，提出了一种可转移的最优传输方法，使得在一个分布对上学习到的切片器能够高效地应用于相似的新分布对，显著提高了最优传输在动态、多领域大规模数据中的计算效率和实用性。'}


### 论文摘要

Optimal Transport (OT) offers a powerful framework for finding correspondences between distributions and addressing matching and alignment problems in various areas of computer vision, including shape analysis, image generation, and multimodal tasks. The computation cost of OT, however, hinders its scalability. Slice-based transport plans have recently shown promise for reducing the computational cost by leveraging the closed-form solutions of 1D OT problems. These methods optimize a one-dimensional projection (slice) to obtain a conditional transport plan that minimizes the transport cost in the ambient space. While efficient, these methods leave open the question of whether learned optimal slicers can transfer to new distribution pairs under distributional shift. Understanding this transferability is crucial in settings with evolving data or repeated OT computations across closely related distributions. In this paper, we study the min-Sliced Transport Plan (min-STP) framework and investigate the transferability of optimized slicers: can a slicer trained on one distribution pair yield effective transport plans for new, unseen pairs? Theoretically, we show that optimized slicers remain close under slight perturbations of the data distributions, enabling efficient transfer across related tasks. To further improve scalability, we introduce a minibatch formulation of min-STP and provide statistical guarantees on its accuracy. Empirically, we demonstrate that the transferable min-STP achieves strong one-shot matching performance and facilitates amortized training for point cloud alignment and flow-based generative modeling.

---

## 43. Training-Free Active Learning Framework in Materials Science with Large Language Models

**论文链接:** [http://arxiv.org/abs/2511.19730v1](http://arxiv.org/abs/2511.19730v1)

**作者:** Hongchen Wang, Rafael Espinosa Castañeda, Jay R. Werber, Yao Fehlis, Edward Kim, Jason Hattrick-Simpers

**发布时间:** 2025-11-24

### GPT解析

### 总结

该研究引入了一种基于大型语言模型(LLM)的主动学习框架(LLM-AL)，在四个材料科学数据集上测试，发现其能减少70%以上的实验数量，始终优于传统ML模型，执行更广泛探索性搜索，同时性能稳定。

### 背景

传统主动学习(AL)通过优先选择信息量最大的实验加速科学发现，但传统机器学习(ML)模型存在冷启动限制和领域特定特征工程，限制了泛化能力。

### 目的

引入并评估一种基于大型语言模型(LLM)的主动学习框架(LLM-AL)，作为传统AL方法的替代方案，提高实验选择的效率和可解释性。

### 方法

LLM-AL在少样本设置下迭代运行，探索了两种提示策略：一种使用简洁数值输入(适用于组成和结构化特征数据集)，另一种使用扩展描述性文本(适用于实验和程序性特征数据集)，并在四个材料科学数据集上与传统ML模型进行基准测试。

### 主要发现

在所有数据集上，LLM-AL减少70%以上实验数量以获得最佳候选者；始终优于传统ML模型；执行更广泛探索性搜索，同时以更少迭代达到最优；性能在多次运行中保持一致，与传统ML方法的变异性范围相当。

### 结论

LLM-AL可作为传统AL管道的通用替代方案，实现更高效和可解释的实验选择，并可能支持LLM驱动的自主科学发现。

### 翻译

主动学习(AL)通过优先选择信息量最大的实验来加速科学发现，但传统机器学习(ML)模型存在冷启动限制和领域特定特征工程，限制了其泛化能力。大型语言模型(LLM)利用其预训练知识和通用基于令牌的表示，直接从文本描述中提出实验，提供了新范式。本研究介绍了一种基于LLM的主动学习框架(LLM-AL)，在少样本设置下迭代运行，并在四个不同的材料科学数据集上与传统ML模型进行基准测试。我们探索了两种提示策略：一种使用适合具有更多组成和结构化特征数据集的简洁数值输入，另一种使用适合具有更多实验和程序性特征数据集的扩展描述性文本，以提供额外上下文。在所有数据集上，LLM-AL可以减少70%以上的实验数量，以获得表现最佳的候选者，并始终优于传统ML模型。我们发现LLM-AL执行更广泛和探索性的搜索，同时以更少的迭代次数达到最优。我们进一步检查了LLM-AL的稳定性边界，考虑到LLM固有的非确定性，发现其性能在多次运行中保持一致，处于传统方法通常观察到的变异性范围内。这些结果表明，LLM-AL可以作为传统AL管道的通用替代方案，用于更高效和可解释的实验选择，并可能实现LLM驱动的自主发现。


### 论文摘要

Active learning (AL) accelerates scientific discovery by prioritizing the most informative experiments, but traditional machine learning (ML) models used in AL suffer from cold-start limitations and domain-specific feature engineering, restricting their generalizability. Large language models (LLMs) offer a new paradigm by leveraging their pretrained knowledge and universal token-based representations to propose experiments directly from text-based descriptions. Here, we introduce an LLM-based active learning framework (LLM-AL) that operates in an iterative few-shot setting and benchmark it against conventional ML models across four diverse materials science datasets. We explored two prompting strategies: one using concise numerical inputs suited for datasets with more compositional and structured features, and another using expanded descriptive text suited for datasets with more experimental and procedural features to provide additional context. Across all datasets, LLM-AL could reduce the number of experiments needed to reach top-performing candidates by over 70% and consistently outperformed traditional ML models. We found that LLM-AL performs broader and more exploratory searches while still reaching the optima with fewer iterations. We further examined the stability boundaries of LLM-AL given the inherent non-determinism of LLMs and found its performance to be broadly consistent across runs, within the variability range typically observed for traditional ML approaches. These results demonstrate that LLM-AL can serve as a generalizable alternative to conventional AL pipelines for more efficient and interpretable experiment selection and potential LLM-driven autonomous discovery.

---

## 44. Electrochemical Interfaces at Constant Potential: Data-Efficient Transfer Learning for Machine-Learning-Based Molecular Dynamics

**论文链接:** [http://arxiv.org/abs/2511.19338v1](http://arxiv.org/abs/2511.19338v1)

**作者:** Michele Giovanni Bianchi, Michele Re Fiorentin, Francesca Risplendi, Candido Fabrizio Pirri, Michele Parrinello, Luigi Bonati, Giancarlo Cicero

**发布时间:** 2025-11-24

**备注:** 12 pages, 4 figures + Supplementary Information (4 pages, 4 figures)

### GPT解析

### 总结

TRECI是一种数据高效的工作流程，用于构建机器学习力场，能够在电子巨正则分子动力学中达到第一性原理水平的精度，使得在恒定电位下模拟带电金属/水界面变得更加可行。

### 背景

使用第一性原理方法模拟带电金属/水界面在恒定电位下的过程对于理解电化学过程至关重要，但计算成本极高。

### 目的

开发一种名为TRECI的数据高效工作流程，用于构建能够达到第一性原理精度的机器学习力场(ML-FFs)。

### 方法

TRECI利用从通用模型和领域特定模型的迁移学习，使用较少的参考配置实现跨广泛电位范围的稳定准确模拟。这种效率使得可以使用高级meta-GGA泛函和严格的表面电离方案。

### 主要发现

应用于Cu(111)/水系统时，仅使用一千个配置训练的模型就能产生准确的分子动力学模拟，捕捉到了之前未报道的与偏置相关的溶剂重构效应。

### 结论

TRECI为表征不同材料和界面化学提供了一般策略，显著降低了恒定电位模拟的成本，扩展了定量电化学建模的途径。

### 翻译

在显式溶剂下模拟恒定电位下的带电金属/水界面对于理解电化学过程至关重要，但使用第一性原理方法仍然成本过高。我们提出了TRECI，一种用于构建机器学习力场的数据高效工作流程，该力场在电子巨正则分子动力学中能够达到第一性原理水平的精度。通过利用从通用模型和领域特定模型的迁移学习，TRECI能够在使用较少参考配置的情况下，实现跨广泛电位范围的稳定准确模拟。这种效率使得可以使用高级meta-GGA泛函和严格的表面电离方案。应用于Cu(111)/水时，仅在一千个配置上训练的模型就能产生准确的分子动力学模拟，捕捉到了之前未报道的与偏置相关的溶剂重构效应。TRECI为表征不同材料和界面化学提供了一般策略，显著降低了恒定电位模拟的成本，扩展了定量电化学建模的途径。


### 论文摘要

Simulating electrified metal/water interfaces with explicit solvent under constant potential is essential for understanding electrochemical processes, yet remains prohibitively expensive with ab initio methods. We present TRECI, a data-efficient workflow for constructing machine learning force-fields (ML-FFs) that achieve ab initio-level accuracy in electronically grand-canonical molecular dynamics. By leveraging transfer learning from general-purpose and domain-specific models, TRECI enables stable and accurate simulations across a wide potential range using a reduced number of reference configurations. This efficiency allows the use of high-level meta-GGA functionals and rigorous surface-electrification schemes. Applied to Cu(111)/water, models trained on just one thousand configurations yield accurate molecular dynamics simulations, capturing bias-dependent solvent restructuring effects not previously reported. TRECI offers a general strategy for characterising diverse materials and interfacial chemistries, significantly lowering the cost of realistic constant-potential simulations and expanding access to quantitative electrochemical modelling.

---

## 45. Rethinking Intermediate Representation for VLM-based Robot Manipulation

**论文链接:** [http://arxiv.org/abs/2511.19315v1](http://arxiv.org/abs/2511.19315v1)

**作者:** Weiliang Tang, Jialin Gao, Jia-Hui Pan, Gang Wang, Li Erran Li, Yunhui Liu, Mingyu Ding, Pheng-Ann Heng, Chi-Wing Fu

**发布时间:** 2025-11-24

### GPT解析

### 总结

SEAM是一种受上下文无关语法启发的语义组装表示，通过分解中间表示为词汇表和语法，解决了Vision-Language Model(VLM)在将人类指令转化为可执行动作时面临的可理解性和泛化能力之间的权衡问题。

### 背景

Vision-Language Model(VLM)是实现鲁棒机器人操作的重要组件，但在将人类指令转化为可执行的动作表示时，通常需要在VLM可理解性和泛化能力之间进行权衡。

### 目的

设计一种新的表示方法，解决VLM在处理人类指令时面临的可理解性和泛化能力之间的权衡问题。

### 方法

受上下文无关语法启发，设计了名为SEAM的语义组装表示，将中间表示分解为词汇表和语法；设计了一种新的开放词汇分割范式，采用检索增强的少样本学习策略来定位精细的物体部件；制定了新的动作泛化能力和VLM可理解性指标。

### 主要发现

SEAM实现了语义丰富的简洁操作词汇表和VLM友好的语法；在所有最先进的并行工作中实现了最短的推理时间；在动作泛化能力和VLM可理解性方面优于主流表示方法；在各种设置和任务下展现了最先进的性能。

### 结论

SEAM有效地解决了VLM可理解性和泛化能力之间的权衡问题，在机器人操作领域表现出色，通过大量真实世界实验验证了其最先进的性能。

### 翻译

视觉语言模型(VLM)是实现鲁棒机器人操作的重要组件。然而，使用它将人类指令转化为可执行的动作表示通常需要在VLM可理解性和泛化能力之间进行权衡。受上下文无关语法的启发，我们设计了名为SEAM的语义组装表示，通过将中间表示分解为词汇表和语法。这样做使我们能够构建一个语义丰富的简洁操作词汇表和一个VLM友好的语法，以处理各种未见过的任务。此外，我们设计了一种新的开放词汇分割范式，采用检索增强的少样本学习策略来定位用于操作的精细物体部件，在所有最先进的并行工作中实现了最短的推理时间。我们还制定了新的动作泛化能力和VLM可理解性指标，证明了SEAM在两个方面都优于主流表示。大量的真实世界实验进一步证明了在各种设置和任务下其最先进的性能。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决基于视觉语言模型(VLM)的机器人操作中，中间表示设计时VLM可理解性与动作泛化能力之间的权衡问题。这个问题很重要，因为它限制了VLM在机器人操作中的应用效果：高级表示虽易被VLM理解但难以扩展到新任务，需要手动添加新词汇；低级表示虽有良好泛化能力但生成的中间表示过于复杂，难以被VLM理解和生成。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': "作者受到Wittgenstein名言'语言的界限意味着我的世界的界限'启发，观察到现有中间表示方法的权衡问题。他们借鉴了上下文无关文法(Context-Free Grammar)的概念，将中间表示分解为词汇表和语法，同时参考了早期使用自动机理论建模机器人的工作。设计过程中遵循了VLM可读性、适当抽象、简洁性、可靠性、适当简约和可组合性等原则，最终设计了语义组装表示(SEAM)。", '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是设计一种语义组装表示(SEAM)，将中间表示分解为语义丰富的词汇表和组合语法，从而平衡VLM可理解性和动作泛化能力。整体流程包括：1)接收人类指令和视觉输入；2)VLM使用设计的词汇表和语法生成SEAM表示；3)将SEAM表示转换为中间表示；4)通过RAG数据库检索和少样本分割进行物体部分定位；5)通过优化求解机器人轨迹；6)执行机器人动作。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)语义组装表示(SEAM)，分解中间表示为词汇表和语法；2)基于RAG的少样本开放词汇分割方法；3)提出动作泛化能力和VLM可理解性两个新评估指标。相比之前工作，SEAM不需要为每个新任务手动添加新词汇(区别于高级表示方法)，同时生成的中间表示更简洁易理解(区别于低级表示方法)，在真实世界实验中比之前SOTA方法提高15%成功率，且推理时间更短。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '本文提出了一种语义组装表示(SEAM)，通过将中间表示分解为语义词汇表和组合语法，有效平衡了视觉语言模型的可理解性和机器人动作的泛化能力，显著提升了VLM-based机器人操作的性能和适用性。'}


### 论文摘要

Vision-Language Model (VLM) is an important component to enable robust robot manipulation. Yet, using it to translate human instructions into an action-resolvable intermediate representation often needs a tradeoff between VLM-comprehensibility and generalizability. Inspired by context-free grammar, we design the Semantic Assembly representation named SEAM, by decomposing the intermediate representation into vocabulary and grammar. Doing so leads us to a concise vocabulary of semantically-rich operations and a VLM-friendly grammar for handling diverse unseen tasks. In addition, we design a new open-vocabulary segmentation paradigm with a retrieval-augmented few-shot learning strategy to localize fine-grained object parts for manipulation, effectively with the shortest inference time over all state-of-the-art parallel works. Also, we formulate new metrics for action-generalizability and VLM-comprehensibility, demonstrating the compelling performance of SEAM over mainstream representations on both aspects. Extensive real-world experiments further manifest its SOTA performance under varying settings and tasks.

---

## 46. A Multi-Agent LLM Framework for Multi-Domain Low-Resource In-Context NER via Knowledge Retrieval, Disambiguation and Reflective Analysis

**论文链接:** [http://arxiv.org/abs/2511.19083v1](http://arxiv.org/abs/2511.19083v1)

**作者:** Wenxuan Mu, Jinzhong Ning, Di Zhao, Yijia Zhang

**发布时间:** 2025-11-24

**备注:** This paper has been accepted by AAAI 2026 (Main Technical Track)

### GPT解析

### 总结

本文提出了KDR-Agent，一个多智能体框架，用于解决低资源场景下命名实体识别的问题，通过整合知识检索、消歧和反思分析，克服了现有ICL方法的三个主要限制。

### 背景

In-context learning (ICL) 与大型语言模型 (LLMs) 在低资源场景下的命名实体识别 (NER) 中显示出前景，但现有方法存在三个关键限制。

### 目的

解决现有ICL-based NER方法的三个主要限制：依赖动态检索注释示例、难以推广到未见领域、无法整合外部知识或解决实体歧义。

### 方法

KDR-Agent是一个多智能体框架，利用自然语言类型定义和静态实体级对比示例减少对大型注释语料库的依赖，通过中央规划器协调专门智能体进行知识检索、实体消歧和模型预测反思纠正。

### 主要发现

在五个领域的十个数据集上的实验表明，KDR-Agent在多个LLM主干上显著优于现有的零样本和少样本ICL基线。

### 结论

KDR-Agent有效解决了低资源场景下命名实体识别的挑战，通过多智能体协作整合知识检索、消歧和反思分析，提高了模型性能。

### 翻译

在低资源场景下，使用大型语言模型进行上下文学习已成为命名实体识别的一种有前景的范式。然而，现有的基于上下文学习的命名实体识别方法存在三个关键局限：(1)依赖注释示例的动态检索，当注释数据稀缺时存在问题；(2)由于大型语言模型内部领域知识不足，难以推广到未见过的领域；(3)无法整合外部知识或解决实体歧义。为应对这些挑战，我们提出了KDR-Agent，一个新颖的多智能体框架，用于多领域低资源上下文命名实体识别，整合了知识检索、消歧和反思分析。KDR-Agent利用自然语言类型定义和静态的实体级对比示例，减少对大型注释语料库的依赖。中央规划器协调专门智能体(i)从Wikipedia检索特定领域的提及事实知识，(ii)通过上下文化推理解决歧义实体，(iii)通过结构化自我评估反思和纠正模型预测。在五个领域的十个数据集上的实验表明，KDR-Agent在多个大型语言模型主干上显著优于现有的零样本和少样本上下文学习基线。代码和数据可在https://github.com/MWXGOD/KDR-Agent找到。


### 论文摘要

In-context learning (ICL) with large language models (LLMs) has emerged as a promising paradigm for named entity recognition (NER) in low-resource scenarios. However, existing ICL-based NER methods suffer from three key limitations: (1) reliance on dynamic retrieval of annotated examples, which is problematic when annotated data is scarce; (2) limited generalization to unseen domains due to the LLM's insufficient internal domain knowledge; and (3) failure to incorporate external knowledge or resolve entity ambiguities. To address these challenges, we propose KDR-Agent, a novel multi-agent framework for multi-domain low-resource in-context NER that integrates Knowledge retrieval, Disambiguation, and Reflective analysis. KDR-Agent leverages natural-language type definitions and a static set of entity-level contrastive demonstrations to reduce dependency on large annotated corpora. A central planner coordinates specialized agents to (i) retrieve factual knowledge from Wikipedia for domain-specific mentions, (ii) resolve ambiguous entities via contextualized reasoning, and (iii) reflect on and correct model predictions through structured self-assessment. Experiments across ten datasets from five domains demonstrate that KDR-Agent significantly outperforms existing zero-shot and few-shot ICL baselines across multiple LLM backbones. The code and data can be found at https://github.com/MWXGOD/KDR-Agent.

---

## 47. MIST: Mutual Information Via Supervised Training

**论文链接:** [http://arxiv.org/abs/2511.18945v1](http://arxiv.org/abs/2511.18945v1)

**作者:** German Gritsai, Megan Richards, Maxime Méloux, Kyunghyun Cho, Maxime Peyrard

**发布时间:** 2025-11-24

### GPT解析

### 总结

提出了一种完全数据驱动的互信息估计器设计方法，使用神经网络参数化估计函数并通过大规模合成数据集进行训练，显著优于传统方法且提供了不确定性量化。

### 背景

互信息估计是机器学习和信息论中的关键问题，传统方法通常在理论保证与实际效率之间进行权衡。

### 目的

开发一种灵活高效的互信息估计器，能够处理不同样本量和维度，并提供可靠的不确定性量化。

### 方法

使用神经网络(MIST)参数化互信息估计函数，在包含625,000个合成联合分布的数据集上进行端到端训练；采用二维注意力方案处理可变样本量和维度；使用分位数回归损失优化以估计互信息的采样分布。

### 主要发现

学习到的估计器在样本量和维度上显著优于经典基线；在未见过的联合分布上表现良好；基于分位数的区间校准良好且比bootstrap置信区间更可靠；推理速度比现有神经基线快几个数量级。

### 结论

该框架产生了可训练、完全可微的估计器，可以嵌入到更大的学习流程中，并通过利用互信息对可逆变换的不变性，可通过归一化流适应任意数据模态。

### 翻译

我们提出了一种完全数据驱动的方法来设计互信息估计器。由于任何互信息估计器都是两个随机变量观测样本的函数，我们使用神经网络(MIST)对该函数进行参数化，并端到端地训练它以预测互信息值。训练是在包含625,000个具有已知真实互信息的合成联合分布的大型元数据集上进行的。为了处理可变的样本量和维度，我们采用二维注意力方案，确保输入样本之间的排列不变性。为了量化不确定性，我们优化了分位数回归损失，使估计器能够近似互信息的采样分布，而不是返回单点估计。该研究项目通过采用完全经验化的方法，放弃了普遍的理论保证，换取了灵活性和效率。从经验上看，学习到的估计器在样本量和维度上都显著优于经典基线方法，包括在训练期间未见过的联合分布上。基于分位数的区间校准良好，比基于bootstrap的置信区间更可靠，而推理速度比现有神经基线快几个数量级。除了即时的经验收益外，该框架产生了可训练的、完全可微的估计器，可以嵌入到更大的学习流程中。此外，利用互信息对可逆变换的不变性，可以通过归一化流使元数据集适应任意数据模态，从而为多样化的目标元分布提供灵活的训练。


### 论文摘要

We propose a fully data-driven approach to designing mutual information (MI) estimators. Since any MI estimator is a function of the observed sample from two random variables, we parameterize this function with a neural network (MIST) and train it end-to-end to predict MI values. Training is performed on a large meta-dataset of 625,000 synthetic joint distributions with known ground-truth MI. To handle variable sample sizes and dimensions, we employ a two-dimensional attention scheme ensuring permutation invariance across input samples. To quantify uncertainty, we optimize a quantile regression loss, enabling the estimator to approximate the sampling distribution of MI rather than return a single point estimate. This research program departs from prior work by taking a fully empirical route, trading universal theoretical guarantees for flexibility and efficiency. Empirically, the learned estimators largely outperform classical baselines across sample sizes and dimensions, including on joint distributions unseen during training. The resulting quantile-based intervals are well-calibrated and more reliable than bootstrap-based confidence intervals, while inference is orders of magnitude faster than existing neural baselines. Beyond immediate empirical gains, this framework yields trainable, fully differentiable estimators that can be embedded into larger learning pipelines. Moreover, exploiting MI's invariance to invertible transformations, meta-datasets can be adapted to arbitrary data modalities via normalizing flows, enabling flexible training for diverse target meta-distributions.

---

## 48. LogSyn: A Few-Shot LLM Framework for Structured Insight Extraction from Unstructured General Aviation Maintenance Logs

**论文链接:** [http://arxiv.org/abs/2511.18727v1](http://arxiv.org/abs/2511.18727v1)

**作者:** Devansh Agarwal, Maitreyi Chatterjee, Biplab Chatterjee

**发布时间:** 2025-11-24

**备注:** Accepted in Proceedings of the 3rd INCOM 2026

### GPT解析

### 总结

本研究介绍了LogSyn框架，利用大型语言模型将非结构化的飞机维护日志转化为结构化数据，通过识别关键故障模式，为航空维护工作流程和预测分析提供可操作见解。

### 背景

飞机维护日志包含宝贵的安全数据，但由于其非结构化文本格式，这些数据未被充分利用，限制了其在维护分析和预测中的应用。

### 目的

开发一个能够将非结构化飞机维护日志转化为结构化、机器可读数据的框架，从而提取关键见解并改进维护工作流程和预测分析。

### 方法

LogSyn框架使用大型语言模型，通过少样本上下文学习方法处理6,169条记录，执行受控抽象生成（CAG）来总结问题解决叙述，并在详细层次本体中分类事件，以实现语义结构化和可操作洞察提取。

### 主要发现

LogSyn框架能够有效识别关键故障模式，提供了一种可扩展的方法，将非结构化维护日志转化为结构化数据，并从中提取有价值的见解。

### 结论

LogSyn为航空及相关行业改进维护工作流程和预测分析提供了实际途径，通过将非结构化维护日志转化为结构化数据，使安全数据得到更有效的利用。

### 翻译

飞机维护日志包含宝贵的安全数据，但由于其非结构化文本格式而未被充分利用。本文介绍了LogSyn，一个使用大型语言模型将这些日志转换为结构化、机器可读数据的框架。通过对6,169条记录使用少样本上下文学习，LogSyn执行受控抽象生成（CAG）来总结问题解决叙述，并在详细层次本体中分类事件。该框架识别关键故障模式，提供了一种可扩展的方法，用于从维护日志中进行语义结构化和可操作洞察提取。这项工作为改进航空和相关行业的维护工作流程和预测分析提供了实际路径。


### 论文摘要

Aircraft maintenance logs hold valuable safety data but remain underused due to their unstructured text format. This paper introduces LogSyn, a framework that uses Large Language Models (LLMs) to convert these logs into structured, machine-readable data. Using few-shot in-context learning on 6,169 records, LogSyn performs Controlled Abstraction Generation (CAG) to summarize problem-resolution narratives and classify events within a detailed hierarchical ontology. The framework identifies key failure patterns, offering a scalable method for semantic structuring and actionable insight extraction from maintenance logs. This work provides a practical path to improve maintenance workflows and predictive analytics in aviation and related industries.

---

## 49. A Theory-Inspired Framework for Few-Shot Cross-Modal Sketch Person Re-Identification

**论文链接:** [http://arxiv.org/abs/2511.18677v1](http://arxiv.org/abs/2511.18677v1)

**作者:** Yunpeng Gong, Yongjie Hou, Jiangming Shi, Kim Long Diep, Min Jiang

**发布时间:** 2025-11-24

**备注:** Accepted by AAAI2026

### GPT解析

### 总结

本文提出了一种名为KTCAA的理论框架，用于解决基于草图的人物重识别任务中的模态差异大和标注数据有限的问题。该框架通过识别影响目标域风险的两个关键因素（域差异和扰动不变性），并提出了对齐增强和知识转移催化剂两个组件，在元学习范式下联合优化，实现了从RGB域到草图域的知识转移。

### 背景

基于草图的人物重识别旨在匹配手绘草图与RGB监控图像，但由于模态差异大和标注数据有限，这一任务具有挑战性。

### 目的

提出一个基于理论基础的少样本跨模态泛化框架，解决基于草图的人物重识别中的模态差异和数据稀缺问题。

### 方法

受泛化理论启发，确定影响目标域风险的域差异和扰动不变性两个关键因素；提出对齐增强组件应用局部草图风格变换模拟目标分布；提出知识转移催化剂组件通过引入最坏情况扰动增强不变性；在元学习范式下联合优化这些组件，实现从数据丰富的RGB域到基于草图的场景的知识转移。

### 主要发现

在多个基准测试上的实验表明，KTCAA在数据稀缺条件下取得了最先进的性能。

### 结论

KTCAA框架通过解决域差异和扰动不变性问题，有效提升了基于草图的人物重识别性能，特别是在数据有限的情况下。

### 翻译

基于草图的人物重识别旨在匹配手绘草图与RGB监控图像，但由于模态差异大和标注数据有限，这一任务仍具有挑战性。为解决这一问题，我们引入了KTCAA，这是一个用于少样本跨模态泛化的理论基础框架。受泛化理论启发，我们确定了影响目标域风险的两个关键因素：（1）域差异，它量化了源分布和目标分布之间的对齐难度；（2）扰动不变性，它评估了模型对模态变化的鲁棒性。基于这些见解，我们提出了两个组件：（1）对齐增强，它应用局部草图风格变换来模拟目标分布并促进渐进对齐；（2）知识转移催化剂，它通过引入最坏情况扰动并强制一致性来增强不变性。这些模块在元学习范式下联合优化，将数据丰富的RGB域中的对齐知识转移到基于草图的场景中。在多个基准测试上的实验表明，KTCAA在数据稀缺条件下取得了最先进的性能。


### 论文摘要

Sketch based person re-identification aims to match hand-drawn sketches with RGB surveillance images, but remains challenging due to significant modality gaps and limited annotated data. To address this, we introduce KTCAA, a theoretically grounded framework for few-shot cross-modal generalization. Motivated by generalization theory, we identify two key factors influencing target domain risk: (1) domain discrepancy, which quantifies the alignment difficulty between source and target distributions; and (2) perturbation invariance, which evaluates the model's robustness to modality shifts. Based on these insights, we propose two components: (1) Alignment Augmentation (AA), which applies localized sketch-style transformations to simulate target distributions and facilitate progressive alignment; and (2) Knowledge Transfer Catalyst (KTC), which enhances invariance by introducing worst-case perturbations and enforcing consistency. These modules are jointly optimized under a meta-learning paradigm that transfers alignment knowledge from data-rich RGB domains to sketch-based scenarios. Experiments on multiple benchmarks demonstrate that KTCAA achieves state-of-the-art performance, particularly in data-scarce conditions.

---

## 50. FHE-Agent: Automating CKKS Configuration for Practical Encrypted Inference via an LLM-Guided Agentic Framework

**论文链接:** [http://arxiv.org/abs/2511.18653v1](http://arxiv.org/abs/2511.18653v1)

**作者:** Nuo Xu, Zhaoting Gong, Ran Ran, Jinwei Tang, Wujie Wen, Caiwen Ding

**发布时间:** 2025-11-23

### GPT解析

### 总结

FHE-Agent是一个自动化全同态加密(CKKS)配置的智能框架，解决了隐私保护机器学习即服务中FHE部署需要专业知识的问题。

### 背景

全同态加密(FHE)，特别是CKKS方案，是隐私保护MLaaS的有前途技术，但其实际部署面临专业知识依赖的障碍。配置CKKS涉及紧密耦合的环维度、模数链和打包布局，没有密码学知识难以优化。

### 目的

开发一种自动化框架，减少对FHE配置专业知识的依赖，解决现有编译器使用固定启发式导致配置僵化或无法找到可行方案的问题。

### 方法

FHE-Agent结合大型语言模型(LLM)控制器与确定性工具套件，将搜索分解为全局参数选择和逐层瓶颈修复。采用多保真度工作流程，用静态分析修剪无效区域，保留加密评估给有希望的候选者。

### 主要发现

在Orion编译器上测试，FHE-Agent在MLP、LeNet、LoLa和AlexNet等模型上比朴素搜索策略实现更好精度和更低延迟，能够为复杂模型自动发现可行的128位安全配置，而基线方法无法产生有效设置。

### 结论

FHE-Agent有效解决了FHE配置的专业知识依赖问题，能够自动发现其他方法无法找到的可行配置，使FHE技术在实践中更加可行。

### 翻译

全同态加密(FHE)，特别是CKKS方案，是隐私保护机器学习即服务(MLaaS)的有前途的使能技术，但其实际部署面临一个巨大的障碍：它严重依赖专业知识。配置CKKS涉及紧密耦合的环维度、模数链和打包布局。没有深厚的密码学知识来处理这些交互，实践者只能依赖于固定启发式的编译器。这些'一次性'工具通常发出僵化的配置，要么在延迟方面过度配置，要么对于更深的网络完全找不到可行的解决方案。我们提出FHE-Agent，一个自动化专家推理过程的智能框架。通过将大型语言模型(LLM)控制器与确定性工具套件相结合，FHE-Agent将搜索分解为全局参数选择和逐层瓶颈修复。智能体在多保真度工作流程中运行，使用廉价的静态分析修剪无效区域，并将昂贵的加密评估保留给最有希望的候选者。我们在Orion编译器上实例化FHE-Agent，并在标准基准(MLP、LeNet、LoLa)和更深层次架构(AlexNet)上评估。FHE-Agent始终比朴素搜索策略实现更好的精度和更低的延迟。关键是，它能够为复杂模型自动发现可行的128位安全配置，而基线启发式和一次性提示无法产生有效的设置。


### 论文摘要

Fully Homomorphic Encryption (FHE), particularly the CKKS scheme, is a promising enabler for privacy-preserving MLaaS, but its practical deployment faces a prohibitive barrier: it heavily relies on domain expertise. Configuring CKKS involves a tightly coupled space of ring dimensions, modulus chains, and packing layouts. Without deep cryptographic knowledge to navigate these interactions, practitioners are restricted to compilers that rely on fixed heuristics. These "one-shot" tools often emit rigid configurations that are either severely over-provisioned in latency or fail to find a feasible solution entirely for deeper networks.   We present FHE-Agent, an agentic framework that automates this expert reasoning process. By coupling a Large Language Model (LLM) controller with a deterministic tool suite, FHE-Agent decomposes the search into global parameter selection and layer-wise bottleneck repair. The agents operate within a multi-fidelity workflow, pruning invalid regimes using cheap static analysis and reserving expensive encrypted evaluations for the most promising candidates.   We instantiate FHE-Agent on the Orion compiler and evaluate it on standard benchmarks (MLP, LeNet, LoLa) and deeper architectures (AlexNet). FHE-Agent consistently achieves better precision and lower latency than naïve search strategies. Crucially, it automatically discovers feasible, 128-bit secure configurations for complex models where baseline heuristics and one-shot prompts fail to produce a valid setup.

---

## 51. Health system learning achieves generalist neuroimaging models

**论文链接:** [http://arxiv.org/abs/2511.18640v1](http://arxiv.org/abs/2511.18640v1)

**作者:** Akhil Kondepudi, Akshay Rao, Chenhui Zhao, Yiwei Lyu, Samir Harake, Soumyanil Banerjee, Rushikesh Joshi, Anna-Katharina Meissner, Renly Hou, Cheng Jiang, Asadur Chowdury, Ashok Srinivasan, Brian Athey, Vikas Gulani, Aditya Pandey, Honglak Lee, Todd Hollon

**发布时间:** 2025-11-23

**备注:** 53 pages, 4 main figures, 10 extended data figures

### GPT解析

### 总结

前沿AI模型在公共数据上训练迅速，但缺乏临床数据访问。神经影像学因隐私问题在公共领域代表性不足，限制了模型在临床医学中的表现。研究提出'医疗系统学习'范式，通过直接从临床数据中学习构建高性能通用神经影像模型NeuroVFM，在多个临床任务上取得最先进性能，并能生成超越前沿模型的放射学报告，减少错误并提供更安全的临床决策支持。

### 背景

前沿AI模型（如GPT-5和DINOv3）通过在互联网规模公共数据上训练取得快速进展，但无法访问私有临床数据。神经影像学在公共领域代表性不足，因为MRI和CT扫描包含可识别的面部特征，限制了模型在临床医学中的表现。

### 目的

展示前沿模型在神经影像任务上的表现不佳，并证明从医疗机构常规临床护理过程中产生的未筛选数据中学习（'医疗系统学习'范式）可以产生高性能、通用型神经影像模型。

### 方法

引入NeuroVFM，一种视觉基础模型，使用可扩展的体积联合嵌入预测架构，在524万临床MRI和CT体积上训练。NeuroVFM学习大脑解剖和病理的全面表示，并在多个临床任务上取得最先进性能。

### 主要发现

NeuroVFM展现出神经解剖学理解和诊断结果的可解释性视觉定位能力；与开源语言模型配对后生成的放射学报告在准确性、临床分诊和专家偏好上超越前沿模型；通过临床基础的视觉理解，减少幻觉发现和关键错误，提供更安全的临床决策支持。

### 结论

这些结果确立了医疗系统学习作为构建通用医疗AI的范式，并为临床基础模型提供了可扩展的框架。

### 翻译

前沿人工智能（AI）模型，如OpenAI的GPT-5和Meta的DINOv3，通过在互联网规模的公共数据上训练取得了快速进展，然而此类系统无法访问私有临床数据。神经影像学在公共领域中代表性不足，因为MRI和CT扫描中包含可识别的面部特征，这从根本上限制了模型在临床医学中的表现。在此，我们展示前沿模型在神经影像任务上的表现不佳，并且直接从医疗机构在常规临床护理过程中产生的未筛选数据中学习（我们称之为医疗系统学习的范式）能够产生高性能、通用型神经影像模型。我们引入了NeuroVFM，这是一种视觉基础模型，使用可扩展的体积联合嵌入预测架构，在524万临床MRI和CT体积上进行了训练。NeuroVFM学习大脑解剖和病理的全面表示，在包括放射诊断和报告生成在内的多个临床任务上取得了最先进的性能。该模型展现出新兴的神经解剖学理解和诊断结果的可解释性视觉定位能力。当通过轻量级视觉指令调整与开源语言模型配对时，NeuroVFM生成的放射学报告在准确性、临床分诊和专家偏好上超越了前沿模型。通过临床基础的视觉理解，NeuroVFM减少了幻觉发现和关键错误，提供了更安全的临床决策支持。这些结果确立了医疗系统学习作为构建通用医疗AI的范式，并为临床基础模型提供了可扩展的框架。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决前沿AI模型无法访问私有临床数据的问题，特别是在神经影像领域，由于MRI和CT扫描中可识别的面部特征，公共领域数据代表性不足，这严重限制了模型在临床医学中的表现。这个问题很重要，因为医学AI需要高质量医疗数据才能有效学习，但医疗数据因隐私问题难以获取；神经影像包含丰富诊断信息但受限于隐私问题；现有模型在临床任务上表现不佳影响了AI在医疗诊断中的应用；直接从临床数据学习能让AI获得与专家医生相似的临床经验。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': "作者首先认识到前沿AI模型在医疗领域特别是神经影像方面表现不佳，原因是缺乏私有临床数据；然后提出'健康系统学习'新范式，让AI直接从临床护理期间生成的未筛选数据中学习；设计了Vol-JEPA专门针对体积神经影像的自监督学习方法；从密歇根医学院收集大量临床数据形成UM-NeuroImages数据集；使用预测而非自编码或对比目标进行学习，不需要复杂预处理和标注。作者借鉴了JEPA思想但扩展到体积医学图像，利用了视觉变换器作为基础架构，参考了现有自监督方法但针对医学影像特殊性进行了调整。", '这个方法的核心思想是什么？整体实现流程是怎样的？': "核心思想是'健康系统学习'，让AI直接从健康系统临床操作期间生成的未筛选数据中学习，使用自监督学习方法通过预测目标而非传统自编码或对比目标进行学习，实现多模态学习同时处理CT和MRI。整体流程：1)收集并预处理大量临床神经影像数据；2)设计Vol-JEPA架构，将3D体积分割为上下文和目标区域；3)使用教师-学生框架训练，学生编码器处理上下文，预测器预测目标表示，教师编码器生成真实表示；4)通过神经解剖学感知的掩码策略优化体积学习；5)在多个临床任务上评估模型性能，与语言模型结合用于报告生成和临床分诊。", '论文的关键创新点有哪些？相比之前的工作，有什么不同？': "关键创新点：1)提出'健康系统学习'新范式，直接从临床数据学习；2)开发专门针对体积神经影像的Vol-JEPA自监督学习方法；3)构建包含524万临床MRI和CT体积的UM-NeuroImages大规模多模态数据集；4)实现跨模态理解能力，能处理不同模态、方向和成像协议的神经影像；5)展示诊断基础和可解释性，能将病理图像区域映射到神经学诊断。不同之处：数据来源从公共互联网或筛选数据变为真实临床数据；学习方法从自编码、对比学习或需要大量标注的有监督学习变为Vol-JEPA自监督学习；模型性能超过之前模型并表现出基础模型特性；实现了跨模态的零样本诊断转移能力。", '如果要用一句话总结这篇论文的贡献，你会怎么说？': "这篇论文提出了'健康系统学习'新范式，开发了基于大规模临床神经影像数据训练的NeuroVFM视觉基础模型，实现了在多个临床任务上的最先进性能，展示了AI模型直接从临床数据中学习神经解剖和病理的潜力。"}


### 论文摘要

Frontier artificial intelligence (AI) models, such as OpenAI's GPT-5 and Meta's DINOv3, have advanced rapidly through training on internet-scale public data, yet such systems lack access to private clinical data. Neuroimaging, in particular, is underrepresented in the public domain due to identifiable facial features within MRI and CT scans, fundamentally restricting model performance in clinical medicine. Here, we show that frontier models underperform on neuroimaging tasks and that learning directly from uncurated data generated during routine clinical care at health systems, a paradigm we call health system learning, yields high-performance, generalist neuroimaging models. We introduce NeuroVFM, a visual foundation model trained on 5.24 million clinical MRI and CT volumes using a scalable volumetric joint-embedding predictive architecture. NeuroVFM learns comprehensive representations of brain anatomy and pathology, achieving state-of-the-art performance across multiple clinical tasks, including radiologic diagnosis and report generation. The model exhibits emergent neuroanatomic understanding and interpretable visual grounding of diagnostic findings. When paired with open-source language models through lightweight visual instruction tuning, NeuroVFM generates radiology reports that surpass frontier models in accuracy, clinical triage, and expert preference. Through clinically grounded visual understanding, NeuroVFM reduces hallucinated findings and critical errors, offering safer clinical decision support. These results establish health system learning as a paradigm for building generalist medical AI and provide a scalable framework for clinical foundation models.

---

## 52. Breaking Forgetting: Training-Free Few-Shot Class-Incremental Learning via Conditional Diffusion

**论文链接:** [http://arxiv.org/abs/2511.18516v1](http://arxiv.org/abs/2511.18516v1)

**作者:** Haidong Kang, Ketong Qian, Yi Lu

**发布时间:** 2025-11-23

### GPT解析

### 总结

该研究提出了一种无需训练的FSCIL范式，通过条件扩散过程替代传统的梯度优化，并结合多模态学习策略解决样本稀缺问题，实现了更高效的增量学习。

### 背景

当前FSCIL研究主要关注基于梯度的优化策略来克服灾难性遗忘，但随着新类别数量增加，训练成本爆炸式增长的问题被忽视。基于梯度的更新在极端数据稀缺情况下不仅导致严重的灾难性遗忘，还阻碍对新类别的适应。

### 目的

设计一个无需训练的FSCIL范式，完全移除梯度优化，以解决训练成本爆炸和灾难性遗忘问题。

### 方法

发现了基于梯度的优化与条件扩散过程之间的联系，提出了基于条件扩散的FSCIL框架(CD-FSCIL)，用基于扩散的生成转换替代传统的梯度更新过程。同时引入多模态学习策略，将视觉特征与大型语言模型自动生成的自然语言描述相结合。

### 主要发现

可以设计一个无需训练的FSCIL范式；基于梯度的优化与条件扩散过程之间存在联系；多模态学习策略可以缓解样本稀缺问题并提高对新颖类别的泛化能力。

### 结论

在主流FSCIL基准测试上实现了最先进的性能，大幅降低了计算和内存开销，标志着向无需训练的持续适应的范式转变。

### 翻译

在少样本增量学习(FSCIL)中克服灾难性遗忘的努力主要集中在开发更有效的基于梯度的优化策略。相比之下，很少有人关注随着新类别数量增加而必然出现的训练成本爆炸问题，这是即使在极端数据稀缺情况下仍依赖梯度学习的结果。更关键的是，由于FSCIL通常只为每个新类别提供少量样本，基于梯度的更新不仅会导致对基础类别的严重灾难性遗忘，还会阻碍对新类别的适应。本文通过提出以下问题来打破这一长期存在的限制：我们能否设计一个无需训练的FSCIL范式，完全移除梯度优化？我们通过揭示基于梯度的优化与条件扩散过程之间的有趣联系，给出了肯定的答案。基于这一观察，我们提出了一个基于条件扩散的FSCIL(CD-FSCIL)框架，用基于扩散的生成转换替代传统的梯度更新过程，实现了无需训练的增量适应，同时有效减轻了遗忘。此外，为了在少样本约束下增强表示能力，我们引入了一种多模态学习策略，将视觉特征与大型语言模型自动生成的自然语言描述相结合。这种协同作用大大缓解了样本稀缺问题，并提高了对新颖类别的泛化能力。在主流FSCIL基准上的大量实验表明，我们的方法不仅实现了最先进的性能，还大幅降低了计算和内存开销，标志着向无需训练的持续适应的范式转变。


### 论文摘要

Efforts to overcome catastrophic forgetting in Few-Shot Class-Incremental Learning (FSCIL) have primarily focused on developing more effective gradient-based optimization strategies. In contrast, little attention has been paid to the training cost explosion that inevitably arises as the number of novel classes increases, a consequence of relying on gradient learning even under extreme data scarcity. More critically, since FSCIL typically provides only a few samples for each new class, gradient-based updates not only induce severe catastrophic forgetting on base classes but also hinder adaptation to novel ones. This paper seeks to break this long-standing limitation by asking: Can we design a training-free FSCIL paradigm that entirely removes gradient optimization? We provide an affirmative answer by uncovering an intriguing connection between gradient-based optimization and the Conditional Diffusion process. Building on this observation, we propose a Conditional Diffusion-driven FSCIL (CD-FSCIL) framework that substitutes the conventional gradient update process with a diffusion-based generative transition, enabling training-free incremental adaptation while effectively mitigating forgetting. Furthermore, to enhance representation under few-shot constraints, we introduce a multimodal learning strategy that integrates visual features with natural language descriptions automatically generated by Large Language Models (LLMs). This synergy substantially alleviates the sample scarcity issue and improves generalization across novel classes. Extensive experiments on mainstream FSCIL benchmarks demonstrate that our method not only achieves state-of-the-art performance but also drastically reduces computational and memory overhead, marking a paradigm shift toward training-free continual adaptation.

---

## 53. Scaling Implicit Fields via Hypernetwork-Driven Multiscale Coordinate Transformations

**论文链接:** [http://arxiv.org/abs/2511.18387v1](http://arxiv.org/abs/2511.18387v1)

**作者:** Plein Versace

**发布时间:** 2025-11-23

### GPT解析

### 总结

本文提出了超坐标隐式神经表示(HC-INR)，一种新的INR方法，通过超网络学习信号自适应坐标变换来打破表示瓶颈，并将表示任务分解为坐标变换模块和紧凑隐式场网络两部分。HC-INR实现了比现有基线高4倍的重建保真度，同时使用30-60%更少的参数。

### 背景

隐式神经表示(INRs)已成为表示如图像、3D形状、有距离场和辐射场等信号的有力范式。虽然在架构设计和优化策略方面已取得显著进展，但现有方法仍存在表示瓶颈和可扩展性有限两个核心限制。

### 目的

解决现有INR方法中的两个核心限制：(1)表示瓶颈，迫使单个MLP统一建模异构局部结构；(2)可扩展性有限，缺乏动态适应信号复杂性的分层机制。

### 方法

提出超坐标隐式神经表示(HC-INR)，通过超网络学习信号自适应坐标变换来打破表示瓶颈。HC-INR将表示任务分解为两个组件：(i)学习多尺度坐标变换模块，将输入域变形为解耦潜在空间；(ii)紧凑隐式场网络，以降低复杂度建模变换后信号。引入分层超网络架构，使坐标变换基于局部信号特征，实现表示容量动态分配。

### 主要发现

理论上证明了HC-INR在保持Lipschitz稳定性的同时严格提高了可表示频带的上限。实验表明，HC-INR比现有INR基线实现高达4倍的重建保真度，同时使用30-60%更少的参数。

### 结论

HC-INR是一种有效的隐式神经表示方法，通过超网络架构和坐标变换机制成功解决了现有方法的瓶颈问题，显著提高了表示效率和重建质量。

### 翻译

隐式神经表示(INRs)已成为表示图像、3D形状、有距离场和辐射场等信号的有力范式。虽然在架构设计(如SIREN、FFC、基于KAN的INRs)和优化策略(元学习、摊销、蒸馏)方面已取得显著进展，但现有方法仍存在两个核心限制：(1)表示瓶颈，迫使单个MLP统一建模异构局部结构；(2)可扩展性有限，缺乏动态适应信号复杂性的分层机制。本文引入超坐标隐式神经表示(HC-INR)，一种新的INR类别，通过使用超网络学习信号自适应坐标变换来打破表示瓶颈。HC-INR将表示任务分解为两个组件：(i)学习多尺度坐标变换模块，将输入域变形为解耦的潜在空间；(ii)紧凑隐式场网络，以显著降低的复杂度建模变换后的信号。所提模型引入分层超网络架构，使坐标变换基于局部信号特征，实现表示容量的动态分配。理论上证明了HC-INR在保持Lipschitz稳定性的同时严格提高了可表示频带的上限。在图像拟合、形状重建和神经辐射场逼近的大量实验中，HC-INR比强大的INR基线实现高达4倍的重建保真度，同时使用30-60%更少的参数。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决隐式神经表示（INRs）中的两个核心限制：表示瓶颈和可扩展性有限。这些问题很重要，因为INRs已成为表示图像、3D形状、有符号距离场和辐射场等信号的重要方法，但现有方法难以同时处理平滑的全球结构和复杂的高频细节，限制了它们在更广泛应用场景中的效能。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者思考的出发点是：与其让隐式场更具表现力，不如让坐标系更智能。他们观察到信号通常在变形坐标系中更容易表示，因此设计了超网络驱动的多尺度坐标变换方法。该方法借鉴了超网络（hypernetworks）的概念，但创新性地将其用于生成坐标变换场而非网络权重，同时也参考了SIREN、FFN-Hash等INR架构的设计思想。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是通过超网络驱动的多尺度坐标变换重构输入域，使信号在变换后的空间中更容易被紧凑的神经网络表示。实现流程：1)输入坐标通过局部特征提取器获取空间线索；2)这些特征输入到一系列超网络中，预测不同尺度级别的坐标变换参数；3)坐标变换以嵌套组合方式应用；4)最后将变换后的坐标输入轻量级隐式场网络，输出信号值。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)引入首个采用分层超网络学习多尺度坐标变换的INR框架；2)理论证明自适应坐标变换可增加可表示带宽同时保持稳定性；3)提出可扩展训练策略；4)在多个任务上实现最先进性能。相比之前的工作，HC-INR不依赖扩展全局谱基，而是通过可学习的空间变化传输场修改坐标系，将表示负担从参数生成转移到几何扭曲，在参数减少30-60%的情况下实现2-4倍的重建保真度提升。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '这篇论文提出了一种通过超网络驱动的多尺度坐标变换来重构输入域的新型隐式神经表示方法，显著提高了复杂信号的表示效率和质量，同时大幅减少了模型参数需求。'}


### 论文摘要

Implicit Neural Representations (INRs) have emerged as a powerful paradigm for representing signals such as images, 3D shapes, signed distance fields, and radiance fields. While significant progress has been made in architecture design (e.g., SIREN, FFC, KAN-based INRs) and optimization strategies (meta-learning, amortization, distillation), existing approaches still suffer from two core limitations: (1) a representation bottleneck that forces a single MLP to uniformly model heterogeneous local structures, and (2) limited scalability due to the absence of a hierarchical mechanism that dynamically adapts to signal complexity. This work introduces Hyper-Coordinate Implicit Neural Representations (HC-INR), a new class of INRs that break the representational bottleneck by learning signal-adaptive coordinate transformations using a hypernetwork. HC-INR decomposes the representation task into two components: (i) a learned multiscale coordinate transformation module that warps the input domain into a disentangled latent space, and (ii) a compact implicit field network that models the transformed signal with significantly reduced complexity. The proposed model introduces a hierarchical hypernetwork architecture that conditions coordinate transformations on local signal features, enabling dynamic allocation of representation capacity. We theoretically show that HC-INR strictly increases the upper bound of representable frequency bands while maintaining Lipschitz stability. Extensive experiments across image fitting, shape reconstruction, and neural radiance field approximation demonstrate that HC-INR achieves up to 4 times higher reconstruction fidelity than strong INR baselines while using 30--60\% fewer parameters.

---

## 54. DiVE-k: Differential Visual Reasoning for Fine-grained Image Recognition

**论文链接:** [http://arxiv.org/abs/2511.18305v1](http://arxiv.org/abs/2511.18305v1)

**作者:** Raja Kumar, Arka Sadhu, Ram Nevatia

**发布时间:** 2025-11-23

### GPT解析

### 总结

本文提出了一种名为DiVE-k的框架，用于解决大型视觉语言模型在细粒度图像识别中的问题，通过利用模型自身的top-k预测作为训练信号，显著提高了模型在未见类别上的泛化能力。

### 背景

大型视觉语言模型(LVLMs)拥有广泛的文本知识，但难以利用这些知识进行细粒度图像识别，经常无法区分视觉上相似的类别。现有的使用精确匹配奖励信号的强化学习微调方法往往容易崩溃，鼓励对训练类别的记忆，且无法产生泛化到未见类别所需的差异推理。

### 目的

开发一种能够提高大型视觉语言模型细粒度图像识别能力的方法，特别关注模型对未见类别的泛化能力，同时减少对训练数据的记忆。

### 方法

提出DiVE-k（使用top-k生成的差异视觉推理）框架，利用模型自身的top-k预测作为训练信号。对于每个训练图像，DiVE-k从模型的top-k输出创建多项选择题，并使用强化学习训练模型选择正确答案。这种方法要求模型在合理选项之间进行细粒度差异推理，并提供简单、可验证的奖励信号。

### 主要发现

在五个标准细粒度数据集上的实验表明，该方法显著优于现有方法。在标准的基类到新类泛化设置中，DiVE-k在调和平均指标上分别超过了QWEN2.5-VL-7B和ViRFT 10.04%和6.16%。在混合领域和少样本场景中也显示出类似的提升。

### 结论

DiVE-k框架有效地解决了大型视觉语言模型在细粒度图像识别中的局限性，通过差异推理和top-k预测作为训练信号，显著提高了模型对未见类别的泛化能力，同时减少了过拟合风险。

### 翻译

大型视觉语言模型(LVLMs)拥有广泛的文本知识，但难以利用这些知识进行细粒度图像识别，经常无法区分视觉上相似的类别。现有的使用精确匹配奖励信号的强化学习微调方法往往容易崩溃，鼓励对训练类别的记忆，且无法产生泛化到未见类别所需的差异推理。为此，我们提出了DiVE-k（使用top-k生成的差异视觉推理）框架，利用模型自身的top-k预测作为训练信号。对于每个训练图像，DiVE-k从模型的top-k输出创建多项选择题，并使用强化学习训练模型选择正确答案。这种方法要求模型在合理选项之间进行细粒度差异推理，并提供简单、可验证的奖励信号，减少记忆并提高泛化能力。在五个标准细粒度数据集上的实验表明，我们的方法显著优于现有方法。在标准的基类到新类泛化设置中，DiVE-k在调和平均指标上分别超过了QWEN2.5-VL-7B和ViRFT 10.04%和6.16%。进一步的实验显示，在混合领域和少样本场景中也存在类似的提升。


### 论文摘要

Large Vision Language Models (LVLMs) possess extensive text knowledge but struggles to utilize this knowledge for fine-grained image recognition, often failing to differentiate between visually similar categories. Existing fine-tuning methods using Reinforcement Learning (RL) with exact-match reward signals are often brittle, encourage memorization of training categories, and fail to elicit differential reasoning needed for generalization to unseen classes. To address this, we propose $\textbf{DiVE-k}$, $\textbf{Di}$fferential $\textbf{V}$isual r$\textbf{E}$asoning using top-$\textbf{k}$ generations, framework that leverages model's own top-k predictions as a training signal. For each training image, DiVE-k creates a multiple-choice question from the model's top-k outputs and uses RL to train the model to select the correct answer. This approach requires the model to perform fine-grained differential reasoning among plausible options and provides a simple, verifiable reward signal that mitigates memorization and improves generalization. Experiments on five standard fine-grained datasets show that our method significantly outperforms existing approaches. In the standard base-to-novel generalization setting, DiVE-k surpasses the QWEN2.5-VL-7B and ViRFT by 10.04% and 6.16% on the Harmonic Mean metric, respectively. Further experiments show similar gains in mixed-domain and few-shot scenarios.

---

## 55. The Generalized Proximity Forest

**论文链接:** [http://arxiv.org/abs/2511.19487v1](http://arxiv.org/abs/2511.19487v1)

**作者:** Ben Shaw, Adam Rustad, Sofia Pelagalli Maia, Jake S. Rhodes, Kevin R. Moon

**发布时间:** 2025-11-23

### GPT解析

### 总结

这篇论文介绍了一种广义的邻近森林（PF）模型，将随机森林（RF）邻近性扩展到所有监督距离机器学习场景，包括时间序列分析和回归任务，并将其作为元学习框架使用，实验证明了该模型相比随机森林和k近邻模型的独特优势。

### 背景

随机森林邻近性在各种监督机器学习任务中显示出实用性，包括异常检测、缺失数据插补和可视化。然而，RF邻近性的成功依赖于RF模型本身，而RF模型并非在所有情况下都是理想模型。最近，RF邻近性已通过基于距离的邻近森林（PF）模型扩展到时间序列分析。

### 目的

引入广义PF模型，将RF邻近性扩展到所有监督距离机器学习可以发生的场景；为回归任务引入PF模型的变体；将广义PF模型作为元学习框架的概念引入，扩展到任何预训练分类器的监督插补能力。

### 方法

通过引入广义PF模型，将RF邻近性扩展到所有监督距离机器学习场景；开发适用于回归任务的PF模型变体；将广义PF模型作为元学习框架使用，扩展监督插补能力到任何预训练分类器。

### 主要发现

实验证明广义PF模型相比随机森林模型和k近邻模型具有独特优势。

### 结论

广义PF模型成功地将RF邻近性扩展到更广泛的机器学习场景，包括时间序列分析和回归任务，并作为元学习框架提供了更灵活的监督插补能力。

### 翻译

最近的研究已经证明随机森林（RF）邻近性在各种监督机器学习任务中的实用性，包括异常检测、缺失数据插补和可视化。然而，RF邻近性的实用性依赖于RF模型的成功，而RF模型本身并非在所有情况下都是理想模型。RF邻近性最近已通过基于距离的邻近森林（PF）模型扩展到时间序列，使时间序列分析能够获得RF邻近性的优势。在这项工作中，我们引入了广义PF模型，从而将RF邻近性扩展到所有可以发生监督距离机器学习的场景。此外，我们为回归任务引入了PF模型的变体。我们还引入了将广义PF模型作为元学习框架的概念，将监督插补能力扩展到任何预训练分类器。我们通过实验证明了广义PF模型相比随机森林模型和k近邻模型的独特优势。


### 论文摘要

Recent work has demonstrated the utility of Random Forest (RF) proximities for various supervised machine learning tasks, including outlier detection, missing data imputation, and visualization. However, the utility of the RF proximities depends upon the success of the RF model, which itself is not the ideal model in all contexts. RF proximities have recently been extended to time series by means of the distance-based Proximity Forest (PF) model, among others, affording time series analysis with the benefits of RF proximities. In this work, we introduce the generalized PF model, thereby extending RF proximities to all contexts in which supervised distance-based machine learning can occur. Additionally, we introduce a variant of the PF model for regression tasks. We also introduce the notion of using the generalized PF model as a meta-learning framework, extending supervised imputation capability to any pre-trained classifier. We experimentally demonstrate the unique advantages of the generalized PF model compared with both the RF model and the $k$-nearest neighbors model.

---

## 56. Think Fast: Real-Time IoT Intrusion Reasoning Using IDS and LLMs at the Edge Gateway

**论文链接:** [http://arxiv.org/abs/2511.18230v1](http://arxiv.org/abs/2511.18230v1)

**作者:** Saeid Jamshidi, Amin Nikanjam, Negar Shahabi, Kawser Wazed Nafi, Foutse Khomh, Samira Keivanpour, Rolando Herrero

**发布时间:** 2025-11-23

### GPT解析

### 总结

本文提出了一种以边缘为中心的入侵检测系统框架，结合轻量级机器学习模型和大语言模型，在资源受限的边缘环境中提高网络安全检测的准确性、可解释性和效率。

### 背景

随着物联网设备数量持续增长，保护这些系统免受网络威胁面临重大挑战，特别是在计算和能源资源有限的环境中。

### 目的

开发一种能够在边缘设备上高效运行的入侵检测系统，提高检测准确性、语义可解释性和操作效率。

### 方法

在低功耗边缘网关上评估六种机器学习IDS模型(决策树、KNN、随机森林、CNN、LSTM和混合CNN-LSTM)，并通过低带宽API将遥测数据发送给大语言模型(GPT-4-turbo、DeepSeek V2和LLaMA 3.5)，利用零样本、少样本和思维链推理生成威胁分析和缓解建议。

### 主要发现

系统在真实网络攻击下实现了高达98%的准确率，在DoS、DDoS、暴力破解和端口扫描等攻击评估中提高了可解释性，同时保持低延迟(<1.5秒)、最小带宽使用(<1.2 kB/提示)和能效(<75 J)。

### 结论

该系统展示了作为边缘网关IDS解决方案的实用性和可扩展性，能够在资源受限环境中有效保护物联网系统。

### 翻译

随着连接的物联网设备数量持续增长，保护这些系统免受网络威胁仍然是一个重大挑战，特别是在计算和能源资源有限的环境中。本文提出了一个以边缘为中心的入侵检测系统框架，该框架集成了基于轻量级机器学习的IDS模型与预训练的大语言模型，以提高网络边缘的检测准确性、语义可解释性和操作效率。该系统在低功耗边缘网关上评估了六种基于机器学习的IDS模型，在真实网络攻击下实现了高达98%的准确率。对于异常检测，系统通过低带宽API调用将紧凑且安全的遥测快照传输给大语言模型，这些模型使用零样本、少样本和思维链推理来生成人类可读的威胁分析和可操作的缓解建议。在多样化攻击的评估中，该系统提高了可解释性，同时保持低延迟、最小带宽使用和能效，展示了其作为边缘网关IDS解决方案的实用性和可扩展性。


### 论文摘要

As the number of connected IoT devices continues to grow, securing these systems against cyber threats remains a major challenge, especially in environments with limited computational and energy resources. This paper presents an edge-centric Intrusion Detection System (IDS) framework that integrates lightweight machine learning (ML) based IDS models with pre-trained large language models (LLMs) to improve detection accuracy, semantic interpretability, and operational efficiency at the network edge. The system evaluates six ML-based IDS models: Decision Tree (DT), K-Nearest Neighbors (KNN), Random Forest (RF), Convolutional Neural Network (CNN), Long Short-Term Memory (LSTM), and a hybrid CNN-LSTM model on low-power edge gateways, achieving accuracy up to 98 percent under real-world cyberattacks. For anomaly detection, the system transmits a compact and secure telemetry snapshot (for example, CPU usage, memory usage, latency, and energy consumption) via low-bandwidth API calls to LLMs including GPT-4-turbo, DeepSeek V2, and LLaMA 3.5. These models use zero-shot, few-shot, and chain-of-thought reasoning to produce human-readable threat analyses and actionable mitigation recommendations. Evaluations across diverse attacks such as DoS, DDoS, brute force, and port scanning show that the system enhances interpretability while maintaining low latency (<1.5 s), minimal bandwidth usage (<1.2 kB per prompt), and energy efficiency (<75 J), demonstrating its practicality and scalability as an IDS solution for edge gateways.

---

## 57. MVS-TTA: Test-Time Adaptation for Multi-View Stereo via Meta-Auxiliary Learning

**论文链接:** [http://arxiv.org/abs/2511.18120v1](http://arxiv.org/abs/2511.18120v1)

**作者:** Hannuo Zhang, Zhixiang Chi, Yang Wang, Xinxin Zuo

**发布时间:** 2025-11-22

**备注:** 8 pages, 7 figures

### GPT解析

### 总结

本文提出了一种名为MVS-TTA的高效测试时适应框架，用于增强基于学习的多视图立体视觉方法的适应性，通过桥接基于学习和基于优化的两种范式。

### 背景

当前基于学习的多视图立体视觉方法在大规模训练数据和先进架构下取得了显著进展，但泛化能力有限；而基于优化的方法虽然能实现场景特定适应，但缺乏可扩展性且需要昂贵的场景级优化。

### 目的

开发一个能够提高基于学习的MVS方法适应性的框架，解决其在泛化能力上的不足，同时保持计算效率。

### 方法

MVS-TTA采用自监督的跨视图一致性损失作为辅助任务来指导推理时的适应，并引入元辅助学习策略训练模型从辅助任务更新中受益。该框架是模型无关的，适用于各种MVS方法，只需最小架构修改。

### 主要发现

在标准数据集(DTU, BlendedMVS)和跨数据集泛化设置上的实验表明，MVS-TTA能持续提高性能，即使应用于最先进的MVS模型。这是首次将基于优化的测试时适应集成到基于学习的MVS中，使用元学习方法。

### 结论

MVS-TTA成功桥接了基于学习和基于优化的MVS方法，显著提高了方法的适应性和泛化能力，为多视图立体视觉领域提供了新的研究方向。

### 翻译

最近的基于学习的多视图立体视觉方法是数据驱动的，由于大规模训练数据和先进架构，已取得了显著进展。然而，由于其模型参数是在有限的训练数据分布上训练的，它们的泛化能力仍然不够理想。相比之下，基于优化的方法可以实现场景特定的适应，但缺乏可扩展性，并且需要对每个场景进行昂贵的优化。在本文中，我们提出了MVS-TTA，一个高效的测试时适应框架，通过桥接这两种范式来增强基于学习的MVS方法的适应性。具体来说，MVS-TTA采用自监督的跨视图一致性损失作为辅助任务来指导推理时的适应。我们引入了元辅助学习策略来训练模型，使其能够明确地从基于辅助任务的更新中受益。我们的框架是模型无关的，可以应用于各种MVS方法，只需最小的架构更改。在标准数据集(DTU, BlendedMVS)和具有挑战性的跨数据集泛化设置上的大量实验表明，MVS-TTA能持续提高性能，即使应用于最先进的MVS模型。据我们所知，这是首次将基于优化的测试时适应集成到基于学习的MVS中，使用元学习方法。代码将在https://github.com/mart87987-svg/MVS-TTA上提供。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决学习-based多视图立体(MVS)方法在推理阶段对特定场景适应能力有限的问题。学习-based方法效率高但泛化能力有限，而优化-based方法适应性好但计算成本高。这个问题很重要，因为MVS在遥感、机器人视觉和自动驾驶等领域有广泛应用，提高其适应性和性能将带来实际应用价值，使3D重建技术能够更好地应对现实世界中多样复杂的场景。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先认识到学习-based MVS方法使用固定参数无法针对特定场景优化，而优化-based方法计算成本高。因此，他们设计了测试时适应(TTA)框架，让预训练模型在推理时进行轻量级场景优化。他们借鉴了自监督MVS方法中的光度一致性损失作为优化目标，并基于模型无关元学习(MAML)原则提出了元辅助学习策略，使模型能够从辅助任务更新中受益，从而提高主要任务性能。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是通过元辅助学习策略，使预训练的MVS模型能够在测试时进行轻量级的场景特定优化，使用跨视图光度一致性作为辅助任务指导模型自我适应。整体流程分为元辅助训练和测试时适应两个阶段：元训练时，内循环用光度一致性损失模拟测试时适应，外循环评估适应后的模型在主要任务上的性能并更新原始参数；测试时，对每个样本用光度一致性损失进行少量梯度更新，然后用更新后的模型推断深度图。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：首次将优化-based测试时适应集成到学习-based MVS中；提出模型无关的测试时适应框架，可应用于各种MVS方法；引入元辅助学习策略优化模型使其能从辅助任务更新中受益；使用自监督的跨视图光度一致性损失作为优化目标。相比之前工作，MVS-TTA允许模型在推理时进行场景特定适应而不需要昂贵优化，计算效率更高；通过元学习优化适应过程，效果优于简单的自监督方法；专门针对MVS任务设计，比通用测试时适应方法更适合多视图立体场景。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'MVS-TTA通过元辅助学习策略实现了学习-based多视图立体方法在测试时的轻量级场景特定适应，无需额外标注即可显著提高深度估计的准确性和跨数据集泛化能力。'}


### 论文摘要

Recent learning-based multi-view stereo (MVS) methods are data-driven and have achieved remarkable progress due to large-scale training data and advanced architectures. However, their generalization remains sub-optimal due to fixed model parameters trained on limited training data distributions. In contrast, optimization-based methods enable scene-specific adaptation but lack scalability and require costly per-scene optimization. In this paper, we propose MVS-TTA, an efficient test-time adaptation (TTA) framework that enhances the adaptability of learning-based MVS methods by bridging these two paradigms. Specifically, MVS-TTA employs a self-supervised, cross-view consistency loss as an auxiliary task to guide inference-time adaptation. We introduce a meta-auxiliary learning strategy to train the model to benefit from auxiliary-task-based updates explicitly. Our framework is model-agnostic and can be applied to a wide range of MVS methods with minimal architectural changes. Extensive experiments on standard datasets (DTU, BlendedMVS) and a challenging cross-dataset generalization setting demonstrate that MVS-TTA consistently improves performance, even when applied to state-of-the-art MVS models. To our knowledge, this is the first attempt to integrate optimization-based test-time adaptation into learning-based MVS using meta-learning. The code will be available at https://github.com/mart87987-svg/MVS-TTA.

---

## 58. Curvature-Aware Safety Restoration In LLMs Fine-Tuning

**论文链接:** [http://arxiv.org/abs/2511.18039v1](http://arxiv.org/abs/2511.18039v1)

**作者:** Thong Bach, Thanh Nguyen-Tang, Dung Nguyen, Thao Minh Le, Truyen Tran

**发布时间:** 2025-11-22

**备注:** 19 pages, 10 figures

### GPT解析

### 总结

本文提出了一种曲率感知的对齐恢复方法，用于解决微调大型语言模型导致的安全对齐问题。通过微调模型保留有害内容损失景观几何结构的发现，该方法能够选择性增加有害输入的损失，同时保持任务性能。

### 背景

微调大型语言模型用于下游任务常常会损害安全对齐，即使使用像LoRA这样的参数高效方法也是如此。

### 目的

恢复微调后大型语言模型的安全对齐，同时保持或提高任务性能。

### 方法

提出一种曲率感知的对齐恢复方法，利用影响函数和二阶优化来选择性增加有害输入的损失，同时通过在基础模型和微调模型之间共享的几何结构导航，实现精确的低影响更新。

### 主要发现

微调后的模型保留了有害内容的损失景观的几何结构，无论使用何种微调方法。这表明安全行为没有被消除，而是被转移到参数空间中影响较小的区域。

### 结论

该方法能有效减少有害响应，同时维持甚至提高实用性和少样本学习性能，避免了完全回退到基础模型的需求，实现了精确的安全对齐恢复。

### 翻译

为下游任务微调大型语言模型(LLMs)常常会损害安全对齐，即使使用像LoRA这样的参数高效方法也是如此。在这项工作中，我们发现了一个显著特性：微调后的模型保留了其关于有害内容的损失景观几何结构，无论采用何种微调方法。这表明安全行为没有被消除，而是被转移到参数空间中影响较小的区域。基于这一见解，我们提出了一种曲率感知的对齐恢复方法，利用影响函数和二阶优化来选择性地增加有害输入的损失，同时保持任务性能。通过在基础模型和微调模型之间共享的几何结构导航，我们的方法能够避免不安全输出，同时保持与任务相关的性能，避免完全回退，并实现精确、低影响的更新。在多个模型系列和对抗环境中的广泛评估表明，我们的方法能有效减少有害响应，同时维持甚至提高实用性和少样本学习性能。


### 论文摘要

Fine-tuning Large Language Models (LLMs) for downstream tasks often compromises safety alignment, even when using parameter-efficient methods like LoRA. In this work, we uncover a notable property: fine-tuned models preserve the geometric structure of their loss landscapes concerning harmful content, regardless of the fine-tuning method employed. This suggests that safety behaviors are not erased but shifted to less influential regions of the parameter space. Building on this insight, we propose a curvature-aware alignment restoration method that leverages influence functions and second-order optimization to selectively increase loss on harmful inputs while preserving task performance. By navigating the shared geometry between base and fine-tuned models, our method discourages unsafe outputs while preserving task-relevant performance, avoiding full reversion and enabling precise, low-impact updates. Extensive evaluations across multiple model families and adversarial settings show that our approach efficiently reduces harmful responses while maintaining or even improving utility and few-shot learning performance.

---

## 59. Frequency-Adaptive Sharpness Regularization for Improving 3D Gaussian Splatting Generalization

**论文链接:** [http://arxiv.org/abs/2511.17918v1](http://arxiv.org/abs/2511.17918v1)

**作者:** Youngsik Yun, Dongjun Gu, Youngjung Uh

**发布时间:** 2025-11-22

**备注:** Project page: https://bbangsik13.github.io/FASR

### GPT解析

### 总结

该研究针对3D高斯溅射(3DGS)在少样本场景下对新颖视角泛化能力不足的问题，提出频率自适应锐度正则化(FASR)方法，通过重新制定训练目标提高3DGS的泛化能力。

### 背景

3D高斯溅射(3DGS)在大多数配置中表现出色，但在少样本场景下缺乏对新颖视角的泛化能力，因为它对稀疏观测过度拟合。

### 目的

从机器学习角度重新审视3DGS优化，将新颖视图合成视为对未见视角的泛化问题，这是一个未被充分探索的方向。

### 方法

提出频率自适应锐度正则化(FASR)，它重新制定3DGS训练目标，引导3DGS收敛到更好的泛化解。通过反映图像的局部频率来设置正则化权重和估计局部锐度时的邻域半径，以解决直接应用锐度感知最小化(SAM)带来的问题。

### 主要发现

直接将锐度感知最小化(SAM)应用于3DGS并非最优，它因过度正则化而阻碍高频细节的重建，而降低强度则会导致对锐度惩罚不足。FASR通过反映局部频率设置参数，防止了新颖视角中的浮点器伪影，并重建了SAM倾向于过度平滑的精细细节。

### 结论

在各种配置的数据集上，FASR方法一致改进了广泛的基线方法，有效提高了3DGS在少样本场景下对新颖视角的泛化能力。

### 翻译

尽管3D高斯溅射(3DGS)在大多数配置中表现出色，但在少样本场景下，它缺乏对新颖视角的泛化能力，因为它对稀疏观测过度拟合。我们从机器学习角度重新审视3DGS优化，将新颖视图合成视为对未见视角的泛化问题——这是一个未被充分探索的方向。我们提出了频率自适应锐度正则化(FASR)，它重新制定了3DGS训练目标，从而引导3DGS收敛到更好的泛化解。虽然锐度感知最小化(SAM)同样减少了损失景观的锐度以提高分类模型的泛化能力，但由于任务差异，直接将其应用于3DGS并非最优。具体来说，它因过度正则化而阻碍高频细节的重建，而降低强度则会导致对锐度惩罚不足。为了解决这个问题，我们反映了图像的局部频率来设置正则化权重和估计局部锐度时的邻域半径。这防止了新颖视角中的浮点器伪影，并重建了SAM倾向于过度平滑的精细细节。在各种配置的数据集上，我们的方法一致改进了广泛的基线方法。代码将在https://bbangsik13.github.io/FASR上提供。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决3D高斯溅射(3DGS)在少样本场景下对新颖视角泛化能力不足的问题。这个问题很重要，因为在实际应用中获取密集的输入视角数据成本高昂，而改善3DGS的泛化能力可以使它在资源有限的情况下仍能生成高质量的新颖视角渲染，这对VR/AR、数字孪生、文化遗产数字化等领域有重要价值。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者从机器学习角度重新审视3DGS优化，借鉴了Sharpness-Aware Minimization(SAM)方法，发现直接应用SAM到3DGS效果不佳，因为分类任务和重建任务有差异。SAM会过度正则化导致高频细节丢失。作者通过分析发现，不同频率区域需要不同处理：高频区域需保持锐度以保留细节，低频区域需平坦损失景观以改善泛化。因此，作者设计了一种根据图像局部频率自适应调整扰动幅度和正则化权重的方案，既借鉴了SAM的核心思想，又针对3D重建任务进行了改进。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是根据图像局部频率自适应调整正则化策略：高频区域（如边缘）需要保持锐度以准确重建，低频区域（如平滑表面）需要平坦损失景观以改善泛化。实现流程包括：1)计算输入图像的局部频率图；2)对每个高斯属性分别计算梯度；3)根据局部频率调整扰动幅度和正则化权重；4)沿梯度方向进行扰动找到局部最大值；5)计算扰动后损失并使用加权梯度更新参数。具体来说，通过拉普拉斯高斯多尺度分析估计局部频率，根据频率图调整3D空间中的扰动幅度，高频区域使用小扰动和弱正则化，低频区域使用大扰动和强正则化。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)首次系统研究损失景观平坦性与新颖视角合成泛化的关系；2)提出频率自适应锐度正则化方法，克服SAM在重建任务中过度平滑高频细节的局限；3)根据局部频率自适应调整扰动幅度和正则化权重；4)在改善泛化的同时保留高频细节。相比之前工作，FASR专注于优化策略而非改变表示或添加外部先验；与SAM不同，FASR使用频率自适应的调整而非固定参数；与随机扰动方法相比，FASR使用对抗性扰动沿梯度方向计算最坏情况损失，效率更高且效果更好。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '这篇论文提出了一种频率自适应的锐度正则化方法，通过根据图像局部频率自适应调整优化策略，显著提高了3D高斯溅射在少样本场景下对新颖视角的泛化能力，同时保留了高频细节的重建质量。'}


### 论文摘要

Despite 3D Gaussian Splatting (3DGS) excelling in most configurations, it lacks generalization across novel viewpoints in a few-shot scenario because it overfits to the sparse observations. We revisit 3DGS optimization from a machine learning perspective, framing novel view synthesis as a generalization problem to unseen viewpoints-an underexplored direction. We propose Frequency-Adaptive Sharpness Regularization (FASR), which reformulates the 3DGS training objective, thereby guiding 3DGS to converge toward a better generalization solution. Although Sharpness-Aware Minimization (SAM) similarly reduces the sharpness of the loss landscape to improve generalization of classification models, directly employing it to 3DGS is suboptimal due to the discrepancy between the tasks. Specifically, it hinders reconstructing high-frequency details due to excessive regularization, while reducing its strength leads to under-penalizing sharpness. To address this, we reflect the local frequency of images to set the regularization weight and the neighborhood radius when estimating the local sharpness. It prevents floater artifacts in novel viewpoints and reconstructs fine details that SAM tends to oversmooth. Across datasets with various configurations, our method consistently improves a wide range of baselines. Code will be available at https://bbangsik13.github.io/FASR.

---

## 60. Statistically-Guided Dual-Domain Meta-Learning with Adaptive Multi-Prototype Aggregation for Distributed Fiber Optic Sensing

**论文链接:** [http://arxiv.org/abs/2511.17902v1](http://arxiv.org/abs/2511.17902v1)

**作者:** Yifan He, Haodong Zhang, Qiuheng Song, Lin Lei, Zhenxuan Zeng, Haoyang He, Hongyan Wu

**发布时间:** 2025-11-22

### GPT解析

### 总结

本文提出了一种名为DUPLE的新型元学习框架，用于解决分布式光纤传感(DFOS)在跨部署场景中的活动识别问题，有效应对了信号模式变化、标记数据稀缺和类内多样性不足三大挑战。

### 背景

分布式光纤传感(DFOS)在周界安全领域展现出强大潜力，能够以精细空间分辨率监测长距离振动事件，但实际应用中面临诸多挑战。

### 目的

开发一个元学习框架DUPLE，实现跨部署的DFOS活动识别，解决不同光纤部署类型下的域偏移问题和新场景中标记数据稀缺的问题。

### 方法

1) 双域多原型学习器融合时域和频域特征增强泛化能力；2) 统计引导网络(SGN)从统计特征推断域重要性和原型敏感性；3) 查询感知的原型聚合模块自适应选择和组合相关原型。

### 主要发现

在跨部署DFOS数据集上的大量实验表明，DUPLE方法在域泛化设置中显著优于基线方法，能够以最少的标记数据实现跨不同光纤配置的稳健事件识别。

### 结论

DUPLE框架有效解决了DFOS系统在实际应用中面临的三大关键挑战，实现了跨不同光纤配置的稳健事件识别，仅需少量标记数据。

### 翻译

分布式光纤传感(DFOS)由于其能够在长距离上以精细空间分辨率监测振动事件，在周界安全方面显示出强大潜力。然而，实际DFOS系统面临三个关键挑战：(1)相同活动在不同光纤部署类型(如地下、壁挂)下的信号模式差异很大，导致域偏移；(2)新部署场景中的标记数据通常稀缺或完全不可用，限制了模型适应性；(3)即使在源域内，数据稀缺也难以捕获类内多样性以实现稳健学习。为应对这些挑战，我们提出了一种名为DUPLE的新型元学习框架，用于跨部署的DFOS活动识别。首先，双域多原型学习器融合时域和频域特征，增强模型在信号分布变化下的泛化能力。其次，统计引导网络(SGN)从原始统计特征推断域重要性和原型敏感性，为未标记或未见域的学习提供数据驱动的先验信息。第三，查询感知的原型聚合模块自适应选择和组合相关原型，即使在数据有限的情况下也能提高分类性能。在跨部署DFOS数据集上的大量实验表明，我们的方法在域泛化设置中显著优于基线方法，能够以最少的标记数据实现跨不同光纤配置的稳健事件识别。


### 论文摘要

Distributed Fiber Optic Sensing (DFOS) has shown strong potential in perimeter security due to its capability of monitoring vibration events across long distances with fine spatial resolution. However, practical DFOS systems face three critical challenges: (1) signal patterns of the same activity vary drastically under different fiber deployment types (e.g., underground, wall-mounted), causing domain shift; (2) labeled data in new deployment scenarios is often scarce or entirely unavailable, limiting model adaptability; and (3) even within source domains, data scarcity makes it difficult to capture intra-class diversity for robust learning.   To address these challenges, we propose a novel meta-learning framework, DUPLE, for cross-deployment DFOS activity identification. First, a dual-domain multi-prototype learner fuses temporal and frequency domain features, enhancing the model's generalization ability under signal distribution shifts. Second, a Statistical Guided Network (SGN) infers domain importance and prototype sensitivity from raw statistical features, providing data-driven prior information for learning in unlabeled or unseen domains. Third, a query-aware prototype aggregation module adaptively selects and combines relevant prototypes, thereby improving classification performance even with limited data.   Extensive experiments on cross-deployment DFOS datasets demonstrate that our method significantly outperforms baseline approaches in domain generalization settings, enabling robust event recognition across diverse fiber configurations with minimal labeled data.

---

## 61. JigsawComm: Joint Semantic Feature Encoding and Transmission for Communication-Efficient Cooperative Perception

**论文链接:** [http://arxiv.org/abs/2511.17843v1](http://arxiv.org/abs/2511.17843v1)

**作者:** Chenyi Wang, Zhaowei Li, Ming F. Li, Wujie Wen

**发布时间:** 2025-11-21

### GPT解析

### 总结

本文提出了一种名为JigsawComm的多智能体协同感知框架，通过提取和传输语义上基本且非冗余的数据，在有限带宽约束下最大化感知准确性，显著减少了数据传输量同时保持或提高了感知性能。

### 背景

多智能体协同感知能够克服单智能体系统的固有遮挡和感知范围限制，但其实际应用受限于有限的通信带宽。现有方法通过压缩或启发式消息选择提高带宽效率，但未考虑感官数据的语义相关性或跨智能体冗余性。

### 目的

设计一种系统，使每个传输的比特对最终感知任务做出最大贡献，通过提取和传输语义上基本且非冗余的数据来实现。

### 方法

提出JigsawComm框架：一种端到端训练、语义感知、通信高效的协同感知系统。使用正则化编码器提取语义相关且稀疏的特征，采用轻量级特征效用估计器预测各智能体特征对感知任务的贡献，交换元效用图并计算最优传输策略，选择效用得分最高的特征进行传输。

### 主要发现

该策略本质上消除了冗余，实现了可扩展的O(1)通信成本。在OPV2V和DAIR-V2X基准测试中，JigsawComm将总数据量减少了500倍以上，同时实现了与最先进方法匹配或更优的准确性。

### 结论

JigsawComm通过智能特征选择和传输，解决了多智能体协同感知中的带宽限制问题，在大幅减少通信量的同时保持了高感知性能。

### 翻译

多智能体协同感知有望克服单智能体系统的固有遮挡和感知范围限制（例如自动驾驶）。然而，其实际应用受到有限通信带宽的严重制约。现有方法尝试通过压缩或启发式消息选择来提高带宽效率，但没有考虑感官数据的语义相关性或跨智能体冗余性。我们认为，实用的协同感知系统必须通过提取和传输语义上基本且非冗余的数据，使每个传输比特对最终感知任务做出最大贡献。在本文中，我们提出了一个联合语义特征编码和传输问题，旨在在有限带宽下最大化协同感知准确性。为解决此问题，我们引入了JigsawComm，这是一种端到端训练、语义感知且通信高效的协同感知框架，它学习'拼合'多智能体特征传输的拼图。它使用正则化编码器提取语义相关且稀疏的特征，并使用轻量级特征效用估计器来预测每个智能体特征对最终感知任务的贡献。生成的元效用图在智能体间交换，并用于计算可证明最优的传输策略，该策略为每个位置选择效用得分最高的智能体特征。该策略本质上消除了冗余，并随着智能体数量增加实现了可扩展的O(1)通信成本。在OPV2V和DAIR-V2X基准测试中，JigsawComm将总数据量减少了500倍以上，同时实现了与最先进方法匹配或更优的准确性。


### 论文摘要

Multi-agent cooperative perception (CP) promises to overcome the inherent occlusion and sensing-range limitations of single-agent systems (e.g., autonomous driving). However, its practicality is severely constrained by the limited communication bandwidth. Existing approaches attempt to improve bandwidth efficiency via compression or heuristic message selection, without considering the semantic relevance or cross-agent redundancy of sensory data. We argue that a practical CP system must maximize the contribution of every transmitted bit to the final perception task, by extracting and transmitting semantically essential and non-redundant data. In this paper, we formulate a joint semantic feature encoding and transmission problem, which aims to maximize CP accuracy under limited bandwidth. To solve this problem, we introduce JigsawComm, an end-to-end trained, semantic-aware, and communication-efficient CP framework that learns to ``assemble the puzzle'' of multi-agent feature transmission. It uses a regularized encoder to extract semantically-relevant and sparse features, and a lightweight Feature Utility Estimator to predict the contribution of each agent's features to the final perception task. The resulting meta utility maps are exchanged among agents and leveraged to compute a provably optimal transmission policy, which selects features from agents with the highest utility score for each location. This policy inherently eliminates redundancy and achieves a scalable $\mathcal{O}(1)$ communication cost as the number of agents increases. On the benchmarks OPV2V and DAIR-V2X, JigsawComm reduces the total data volume by up to $>$500$\times$ while achieving matching or superior accuracy compared to state-of-the-art methods.

---

## 62. Agility Meets Stability: Versatile Humanoid Control with Heterogeneous Data

**论文链接:** [http://arxiv.org/abs/2511.17373v2](http://arxiv.org/abs/2511.17373v2)

**作者:** Yixuan Pan, Ruoyi Qiao, Li Chen, Kashyap Chitta, Liang Pan, Haoguang Mai, Qingwen Bu, Hao Zhao, Cunyuan Zheng, Ping Luo, Hongyang Li

**发布时间:** 2025-11-21

### GPT解析

### 总结

本文提出AMS（Agility Meets Stability）框架，首次将动态运动跟踪与极端平衡维护统一在单一策略中，使人形机器人既能执行敏捷技能又能保持稳定平衡。

### 背景

人形机器人需要在以人为中心的环境中执行各种任务，需要结合敏捷性与稳健平衡能力的控制器。现有方法专注于动态技能或稳定性关键行为，但无法兼顾两者。

### 目的

开发一个统一框架，解决人形机器人在动态运动跟踪和极端平衡维护之间的权衡问题。

### 方法

利用异构数据源（人类动作捕捉数据和物理约束的合成平衡动作），设计混合奖励方案，结合自适应学习策略（基于性能的采样和特定动作的奖励塑造）实现高效训练。

### 主要发现

在模拟和真实Unitree G1人形机器人上的实验表明，单一策略可执行舞蹈和跑步等敏捷技能，同时实现叶问蹲等零样本极端平衡动作。

### 结论

AMS作为未来人形应用的通用控制范式，展示了在敏捷性与稳定性之间取得平衡的潜力。

### 翻译

人形机器人被设想在以人为中心的环境中执行各种任务，需要结合敏捷性与稳健平衡能力的控制器。最近在运动控制和全身跟踪方面的进展使动态技能或稳定性关键行为方面取得了显著进步，但现有方法仍然专业化，专注于一种能力而牺牲另一种。在本工作中，我们引入AMS（Agility Meets Stability），这是首个在单一策略中统一动态运动跟踪和极端平衡维护的框架。我们的关键见解是利用异构数据源：提供丰富敏捷行为的人类动作捕捉数据，以及捕捉稳定性配置的物理约束合成平衡动作。为了调和敏捷性和稳定性的不同优化目标，我们设计了一种混合奖励方案，对所有数据应用通用跟踪目标，仅在合成动作中注入平衡特定先验。此外，具有基于性能的采样和特定动作奖励塑造的自适应学习策略，实现了跨不同动作分布的高效训练。我们在模拟和真实Unitree G1人形机器人上广泛验证了AMS。实验证明，单一策略可以执行舞蹈和跑步等敏捷技能，同时执行叶问蹲等零样本极端平衡动作，突显了AMS作为未来人形应用的通用控制范式。


### 论文摘要

Humanoid robots are envisioned to perform a wide range of tasks in human-centered environments, requiring controllers that combine agility with robust balance. Recent advances in locomotion and whole-body tracking have enabled impressive progress in either agile dynamic skills or stability-critical behaviors, but existing methods remain specialized, focusing on one capability while compromising the other. In this work, we introduce AMS (Agility Meets Stability), the first framework that unifies both dynamic motion tracking and extreme balance maintenance in a single policy. Our key insight is to leverage heterogeneous data sources: human motion capture datasets that provide rich, agile behaviors, and physically constrained synthetic balance motions that capture stability configurations. To reconcile the divergent optimization goals of agility and stability, we design a hybrid reward scheme that applies general tracking objectives across all data while injecting balance-specific priors only into synthetic motions. Further, an adaptive learning strategy with performance-driven sampling and motion-specific reward shaping enables efficient training across diverse motion distributions. We validate AMS extensively in simulation and on a real Unitree G1 humanoid. Experiments demonstrate that a single policy can execute agile skills such as dancing and running, while also performing zero-shot extreme balance motions like Ip Man's Squat, highlighting AMS as a versatile control paradigm for future humanoid applications.

---

## 63. Sparse-to-Field Reconstruction via Stochastic Neural Dynamic Mode Decomposition

**论文链接:** [http://arxiv.org/abs/2511.20612v1](http://arxiv.org/abs/2511.20612v1)

**作者:** Yujin Kim, Sarah Dean

**发布时间:** 2025-11-25

### GPT解析

### 总结

本文提出了一种名为Stochastic NODE-DMD的概率扩展方法，用于建模连续时间、非线性动力学系统，解决了传统DMD方法在稀疏/噪声观测、线性近似和不确定性量化方面的局限性。

### 背景

许多现实世界系统（如风场和洋流）是动态且难以建模的，学习其动力学规律是科学机器学习中的核心挑战。

### 目的

解决传统DMD方法在实际应用中面临的观测稀疏/噪声、线性近似依赖和缺乏不确定性量化等问题。

### 方法

提出Stochastic NODE-DMD，一种DMD的概率扩展方法，能够建模连续时间、非线性动力学，同时保持可解释性，支持任意坐标的连续时空重建和预测不确定性量化。

### 主要发现

在四个基准测试中，仅使用10%观测密度训练时，该方法在重建准确性上优于基线；能够恢复真实动力学结构；在多实现数据集上学习保留集合变异性的校准分布。

### 结论

Stochastic NODE-DMD有效解决了动态系统建模的挑战，特别是在观测数据稀疏或存在噪声的情况下表现优异。

### 翻译

许多重要的现实世界系统，如风场和洋流，是动态且难以建模的。学习它们的动力学规律仍然是科学机器学习中的一个核心挑战。动态模态分解（DMD）提供了一种简单、数据驱动的近似方法，但实际应用受到来自连续场的稀疏/噪声观测、对线性近似的依赖以及缺乏有原则的不确定性量化等限制。为解决这些问题，我们引入了Stochastic NODE-DMD，这是DMD的概率扩展，可以建模连续时间、非线性动力学，同时保持可解释性。我们的方法能够在任意坐标处进行连续时空重建，并量化预测不确定性。在四个基准测试（一个合成设置和三个基于物理的流动）中，仅使用10%的观测密度进行训练时，其在重建准确性上优于基线。通过将学习到的模态和连续时间特征值与真实值对齐，它进一步恢复了动力学结构。最后，在具有多个实现的数据集上，我们的方法学习了一个保留集合变异性的潜在动力学校准分布，而不是平均跨越不同状态。我们的代码可在以下网址获取：https://github.com/sedan-group/Stochastic-NODE-DMD


### 论文摘要

Many consequential real-world systems, like wind fields and ocean currents, are dynamic and hard to model. Learning their governing dynamics remains a central challenge in scientific machine learning. Dynamic Mode Decomposition (DMD) provides a simple, data-driven approximation, but practical use is limited by sparse/noisy observations from continuous fields, reliance on linear approximations, and the lack of principled uncertainty quantification. To address these issues, we introduce Stochastic NODE-DMD, a probabilistic extension of DMD that models continuous-time, nonlinear dynamics while remaining interpretable. Our approach enables continuous spatiotemporal reconstruction at arbitrary coordinates and quantifies predictive uncertainty. Across four benchmarks, a synthetic setting and three physics-based flows, it surpasses a baseline in reconstruction accuracy when trained from only 10% observation density. It further recovers the dynamical structure by aligning learned modes and continuous-time eigenvalues with ground truth. Finally, on datasets with multiple realizations, our method learns a calibrated distribution over latent dynamics that preserves ensemble variability rather than averaging across regimes. Our code is available at: https://github.com/sedan-group/Stochastic-NODE-DMD

---

## 64. STARFlow-V: End-to-End Video Generative Modeling with Normalizing Flow

**论文链接:** [http://arxiv.org/abs/2511.20462v1](http://arxiv.org/abs/2511.20462v1)

**作者:** Jiatao Gu, Ying Shen, Tianrong Chen, Laurent Dinh, Yuyang Wang, Miguel Angel Bautista, David Berthelot, Josh Susskind, Shuangfei Zhai

**发布时间:** 2025-11-25

**备注:** 21 pages

### GPT解析

### 总结

STARFlow-V是一种基于归一化流(正态化流)的视频生成器，它通过全局-局部架构在时空潜在空间中操作，解决了传统自回归扩散模型在视频生成中的误差累积问题，并提供了端到端学习、稳健因果预测和原生似然估计等优势。

### 背景

归一化流是用于连续数据的端到端似然生成模型，在图像生成方面已取得进展，但在视频生成领域，由于时空复杂性和计算成本更高，最先进的系统几乎完全依赖基于扩散的模型。

### 目的

重新审视视频生成设计空间，提出STARFlow-V，一种基于归一化流的高效视频生成器，提供端到端学习、稳健因果预测和原生似然估计等优势。

### 方法

基于STARFlow框架，在时空潜在空间中采用全局-局部架构，将因果依赖限制在全局潜在空间同时保留丰富的帧内局部交互；提出flow-score matching配备轻量级因果去噪器；采用视频感知的雅可比迭代方案提高采样效率；利用可逆结构支持多种生成任务。

### 主要发现

STARFlow-V实现了强大的视觉保真度和时间一致性，与基于扩散的基线相比具有实用的采样吞吐量，首次证明了归一化流能够进行高质量的自回归视频生成。

### 结论

归一化流是有前途的构建世界模型的研究方向，为视频生成领域提供了新的可能性。

### 翻译

归一化流是用于连续数据的端到端似然生成模型，最近在图像生成方面取得了令人鼓舞的进展。然而，在时空复杂性和计算成本高得多的视频生成领域，最先进的系统几乎完全依赖于基于扩散的模型。在这项工作中，我们通过提出STARFlow-V重新审视了这一设计空间，这是一种基于归一化流的视频生成器，具有端到端学习、稳健因果预测和原生似然估计等显著优势。基于最近提出的STARFlow，STARFlow-V在时空潜在空间中操作，采用全局-局部架构，将因果依赖限制在全局潜在空间，同时保留丰富的帧内局部交互。这减轻了随时间累积的误差，这是标准自回归扩散模型生成的常见缺陷。此外，我们提出了flow-score matching，为模型配备轻量级因果去噪器，以自回归方式提高视频生成一致性。为了提高采样效率，STARFlow-V采用视频感知的雅可比迭代方案，将内部更新重构为可并行迭代，同时不破坏因果性。得益于可逆结构，同一模型可以原生支持文本到视频、图像到视频以及视频到视频生成任务。实验证明，STARFlow-V与基于扩散的基线相比，实现了强大的视觉保真度和时间一致性，并具有实用的采样吞吐量。据我们所知，这些结果首次证明了归一化流能够进行高质量的自回归视频生成，将其确立为构建世界模型的有前途的研究方向。代码和生成的样本可在https://github.com/apple/ml-starflow获取。


### 论文摘要

Normalizing flows (NFs) are end-to-end likelihood-based generative models for continuous data, and have recently regained attention with encouraging progress on image generation. Yet in the video generation domain, where spatiotemporal complexity and computational cost are substantially higher, state-of-the-art systems almost exclusively rely on diffusion-based models. In this work, we revisit this design space by presenting STARFlow-V, a normalizing flow-based video generator with substantial benefits such as end-to-end learning, robust causal prediction, and native likelihood estimation. Building upon the recently proposed STARFlow, STARFlow-V operates in the spatiotemporal latent space with a global-local architecture which restricts causal dependencies to a global latent space while preserving rich local within-frame interactions. This eases error accumulation over time, a common pitfall of standard autoregressive diffusion model generation. Additionally, we propose flow-score matching, which equips the model with a light-weight causal denoiser to improve the video generation consistency in an autoregressive fashion. To improve the sampling efficiency, STARFlow-V employs a video-aware Jacobi iteration scheme that recasts inner updates as parallelizable iterations without breaking causality. Thanks to the invertible structure, the same model can natively support text-to-video, image-to-video as well as video-to-video generation tasks. Empirically, STARFlow-V achieves strong visual fidelity and temporal consistency with practical sampling throughput relative to diffusion-based baselines. These results present the first evidence, to our knowledge, that NFs are capable of high-quality autoregressive video generation, establishing them as a promising research direction for building world models. Code and generated samples are available at https://github.com/apple/ml-starflow.

---

## 65. Interpretable Air Pollution Forecasting by Physics-Guided Spatiotemporal Decoupling

**论文链接:** [http://arxiv.org/abs/2511.20257v1](http://arxiv.org/abs/2511.20257v1)

**作者:** Zhiguo Zhang, Xiaoliang Ma, Daniel Schlesinger

**发布时间:** 2025-11-25

**备注:** Accepted to 2025 IEEE International Conference on Big Data

### GPT解析

### 总结

本研究提出了一种物理引导的、可解释的空间时间学习框架，用于空气污染预测，解决了传统模型在性能和可解释性之间的权衡问题。

### 背景

准确且可解释的空气污染预测对公共健康至关重要，但大多数模型在性能和可解释性之间存在权衡。

### 目的

开发一种物理引导的、可解释的空间时间学习框架，实现高性能且可解释的空气污染预测。

### 方法

将空气污染物浓度的空间时间行为分解为两个透明、可加的模块：1)物理引导的传输核，具有基于风和地理条件的有向权重；2)可解释的注意力机制，学习局部响应并将未来浓度归因于特定的历史滞后和外部驱动因素。

### 主要发现

在斯德哥尔摩地区的综合数据集上评估，模型在多个预测时间范围内始终优于最先进的基线模型，集成了高预测性能和空间时间可解释性。

### 结论

该模型为实际应用中的空气质量管理提供了更可靠的基础。

### 翻译

准确且可解释的空气污染预测对公共健康至关重要，但大多数模型在性能和可解释性之间存在权衡。本研究提出了一种物理引导的、可解释的空间时间学习框架。该模型将空气污染物浓度的空间时间行为分解为两个透明、可加的模块。第一个是物理引导的传输核，具有基于风和地理条件（平流）的有向权重。第二个是可解释的注意力机制，学习局部响应并将未来浓度归因于特定的历史滞后和外部驱动因素。在斯德哥尔摩地区的综合数据集上评估，我们的模型在多个预测时间范围内始终优于最先进的基线模型。我们的模型将高预测性能和空间时间可解释性相结合，为实际应用中的空气质量管理提供了更可靠的基础。


### 论文摘要

Accurate and interpretable air pollution forecasting is crucial for public health, but most models face a trade-off between performance and interpretability. This study proposes a physics-guided, interpretable-by-design spatiotemporal learning framework. The model decomposes the spatiotemporal behavior of air pollutant concentrations into two transparent, additive modules. The first is a physics-guided transport kernel with directed weights conditioned on wind and geography (advection). The second is an explainable attention mechanism that learns local responses and attributes future concentrations to specific historical lags and exogenous drivers. Evaluated on a comprehensive dataset from the Stockholm region, our model consistently outperforms state-of-the-art baselines across multiple forecasting horizons. Our model's integration of high predictive performance and spatiotemporal interpretability provides a more reliable foundation for operational air-quality management in real-world applications.

---

## 66. SyncMV4D: Synchronized Multi-view Joint Diffusion of Appearance and Motion for Hand-Object Interaction Synthesis

**论文链接:** [http://arxiv.org/abs/2511.19319v1](http://arxiv.org/abs/2511.19319v1)

**作者:** Lingwei Dang, Zonghan Li, Juntong Li, Hongwen Zhang, Liang An, Yebin Liu, Qingyao Wu

**发布时间:** 2025-11-24

**备注:** Project Page: https://droliven.github.io/SyncMV4D

### GPT解析

### 总结

SyncMV4D是首个能够联合生成同步多视角手部-物体交互视频和4D运动的模型，通过整合视觉先验、运动动力学和多视角几何，解决了现有方法的局限性。

### 背景

当前基于视频的HOI生成方法主要是单视角的，限制了3D几何感知并导致几何失真；而3D HOI方法虽能生成合理运动，但依赖高质量实验室数据，泛化能力差。

### 目的

开发一个不依赖高质量3D数据，同时保持视觉真实感、运动合理性和多视角一致性的HOI生成模型。

### 方法

SyncMV4D框架包含两个核心创新：多视角联合扩散模型共同生成HOI视频和中间运动；扩散点对齐器将粗略运动细化为全局对齐的4D度量点轨迹。通过闭环循环将2D外观与4D动力学紧密耦合。

### 主要发现

实验表明，该方法在视觉真实感、运动合理性和多视角一致性方面优于现有最先进方法。

### 结论

SyncMV4D成功解决了现有HOI生成方法的局限性，实现了高质量的多视角HOI视频和4D运动生成，无需依赖高质量3D数据。

### 翻译

手部-物体交互生成在推动动画和机器人应用方面发挥着关键作用。当前基于视频的方法主要是单视角的，这阻碍了全面的3D几何感知，并常常导致几何失真或不真实的运动模式。虽然3D HOI方法可以生成动态上合理的运动，但它们对在受控实验室环境中捕获的高质量3D数据的严重依赖限制了它们在真实世界场景中的泛化能力。为了克服这些局限性，我们引入了SyncMV4D，这是首个通过统一视觉先验、运动动力学和多视角几何来联合生成同步多视角HOI视频和4D运动的模型。我们的框架有两个核心创新：多视角联合扩散模型共同生成HOI视频和中间运动，以及扩散点对齐器，将粗略的中间运动细化为全局对齐的4D度量点轨迹。为了将2D外观与4D动力学紧密耦合，我们建立了一个闭环、相互增强的循环。在扩散去噪过程中，生成的视频 conditioned 4D运动的细化，而对齐的4D点轨迹被重新投影以指导下一步的联合生成。实验证明，我们的方法在视觉真实感、运动合理性和多视角一致性方面优于最先进的替代方法。


### 论文摘要

Hand-Object Interaction (HOI) generation plays a critical role in advancing applications across animation and robotics. Current video-based methods are predominantly single-view, which impedes comprehensive 3D geometry perception and often results in geometric distortions or unrealistic motion patterns. While 3D HOI approaches can generate dynamically plausible motions, their dependence on high-quality 3D data captured in controlled laboratory settings severely limits their generalization to real-world scenarios. To overcome these limitations, we introduce SyncMV4D, the first model that jointly generates synchronized multi-view HOI videos and 4D motions by unifying visual prior, motion dynamics, and multi-view geometry. Our framework features two core innovations: (1) a Multi-view Joint Diffusion (MJD) model that co-generates HOI videos and intermediate motions, and (2) a Diffusion Points Aligner (DPA) that refines the coarse intermediate motion into globally aligned 4D metric point tracks. To tightly couple 2D appearance with 4D dynamics, we establish a closed-loop, mutually enhancing cycle. During the diffusion denoising process, the generated video conditions the refinement of the 4D motion, while the aligned 4D point tracks are reprojected to guide next-step joint generation. Experimentally, our method demonstrates superior performance to state-of-the-art alternatives in visual realism, motion plausibility, and multi-view consistency.

---

## 67. Leveraging Spatiotemporal Graph Neural Networks for Multi-Store Sales Forecasting

**论文链接:** [http://arxiv.org/abs/2511.19267v1](http://arxiv.org/abs/2511.19267v1)

**作者:** Manish Singh, Arpita Dayama

**发布时间:** 2025-11-24

**备注:** 6 pages, 4 figures, 1 table

### GPT解析

### 总结

该研究评估了时空图神经网络在多商店零售销售预测中的有效性，并与传统基线方法进行了比较。

### 背景

零售行业需要准确的多商店销售预测来优化库存和供应链管理。

### 目的

比较时空图神经网络与传统预测方法在多商店零售销售预测中的表现，探究关系结构对预测质量的影响。

### 方法

使用45家沃尔玛商店的周销售数据构建关系预测框架，通过学习的自适应图建模商店间依赖关系，STGNN预测对数差分销售并通过残差路径重建最终值。

### 主要发现

STGNN在归一化总绝对误差、P90 MAPE和跨商店MAPE方差方面表现最优；分析显示模型能自动识别有意义的功能商店集群和高影响力节点，无需地理元数据。

### 结论

关系结构显著提高了互联零售环境中的预测质量，时空图神经网络是处理多商店需求预测问题的稳健建模选择。

### 翻译

本研究评估了时空图神经网络在多商店零售销售预测中的有效性，并将其与ARIMA、LSTM和XGBoost基线方法进行了性能比较。我们使用来自45家沃尔玛商店的周销售数据，构建了一个关系预测框架，通过学习的自适应图建模商店间的依赖关系。所提出的STGNN预测对数差分销售并通过残差路径重建最终值，实现了稳定训练和改进的泛化能力。实验表明，STGNN实现了最低的整体预测误差，在归一化总绝对误差、P90 MAPE和跨商店MAPE方差方面优于所有基线方法。对学习到的邻接矩阵的分析揭示了有意义的功能商店集群和高影响力节点，这些信息不需要地理元数据就能出现。这些结果表明，关系结构显著提高了互联零售环境中的预测质量，并将STGNN确立为多商店需求预测的稳健建模选择。


### 论文摘要

This work evaluates the effectiveness of spatiotemporal Graph Neural Networks (GNNs) for multi-store retail sales forecasting and compares their performance against ARIMA, LSTM, and XGBoost baselines. Using weekly sales data from 45 Walmart stores, we construct a relational forecasting framework that models inter-store dependencies through a learned adaptive graph. The proposed STGNN predicts log-differenced sales and reconstructs final values through a residual path, enabling stable training and improved generalisation. Experiments show that STGNN achieves the lowest overall forecasting error, outperforming all baselines in Normalised Total Absolute Error, P90 MAPE, and variance of MAPE across stores. Analysis of the learned adjacency matrix reveals meaningful functional store clusters and high-influence nodes that emerge without geographic metadata. These results demonstrate that relational structure significantly improves forecast quality in interconnected retail environments and establishes STGNNs as a robust modelling choice for multi-store demand prediction.

---

## 68. DetAny4D: Detect Anything 4D Temporally in a Streaming RGB Video

**论文链接:** [http://arxiv.org/abs/2511.18814v1](http://arxiv.org/abs/2511.18814v1)

**作者:** Jiawei Hou, Shenghao Zhang, Can Wang, Zheng Gu, Yonggen Ling, Taiping Zeng, Xiangyang Xue, Jingbo Zhang

**发布时间:** 2025-11-24

### GPT解析

### 总结

本文提出了一种新的4D物体检测方法，通过引入大规模数据集DA4D和端到端框架DetAny4D，解决了现有方法在时间一致性和稳定性方面的问题。

### 背景

4D物体检测（流视频中3D物体检测）对感知和理解现实世界至关重要，但现有开集方法要么缺乏时间一致性建模，要么依赖复杂多阶段流程容易传播错误，且缺乏大规模连续可靠标注数据集。

### 目的

克服现有4D物体检测方法的局限性，提高检测的时间一致性和稳定性。

### 方法

引入DA4D大规模数据集（28万+序列，高质量边界框标注），提出DetAny4D开集端到端框架，融合预训练基础模型多模态特征，设计感知几何的时空解码器，采用多任务学习架构配合专门训练策略保持全局一致性。

### 主要发现

DetAny4D实现了有竞争力的检测精度，显著提高了时间稳定性，有效解决了4D物体检测中长期存在的抖动和不一致问题。

### 结论

通过数据集创新和算法改进，DetAny4D为4D物体检测提供了更可靠、一致的解决方案。

### 翻译

可靠的4D物体检测，即在流视频中检测3D物体，对感知和理解现实世界至关重要。现有的开集4D物体检测方法通常逐帧预测而不建模时间一致性，或依赖复杂的多阶段流程，容易在级联阶段中传播错误。该领域进展受到缺乏大规模连续可靠3D边界框标注数据集的阻碍。为克服这些挑战，我们首先引入DA4D，一个包含28万多个序列的高质量边界框标注的大规模4D检测数据集，在各种条件下收集。基于DA4D，我们提出DetAny4D，一个从序列输入直接预测3D边界框的开集端到端框架。DetAny4D融合了预训练基础模型的多模态特征，并设计了感知几何的时空解码器，有效捕捉空间和时间动态。此外，它采用多任务学习架构配合专门训练策略，保持不同长度序列的全局一致性。大量实验表明，DetAny4D实现了有竞争力的检测精度，并显著提高了时间稳定性，有效解决了4D物体检测中长期存在的抖动和不一致问题。数据与代码将在接受后发布。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决4D物体检测中的时间一致性问题。现有的开放集4D物体检测方法要么基于逐帧预测而不考虑时间连续性，要么依赖复杂的多阶段流水线导致错误传播。这个问题在现实世界中非常重要，因为可靠的4D物体检测对于自动驾驶、机器人导航等应用至关重要，它们需要连续、一致的3D物体检测来进行长期推理、预测未来状态和确保稳定决策。时间不一致的检测结果可能导致严重的安全问题或错误的决策。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先分析了现有方法的局限性：单帧3D检测缺乏时间一致性，多阶段流水线容易传播错误，且缺乏大规模4D数据集。基于这些分析，作者设计了DetAny4D方法，包括创建大规模4D检测数据集DA4D，提出端到端的4D检测框架，融合预训练基础模型的多模态特征，设计几何感知的时空解码器，以及采用多任务学习架构。作者借鉴了现有工作，如使用预训练的SAM和DINO模型提取特征，采用类似DetAny3D的提示编码器和2D特征聚合器，以及使用UniDepth-V2进行深度和相机参数预测。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是构建一个端到端的开放集4D检测框架，直接从RGB视频序列中预测时空对齐的3D边界框，同时保持时间一致性。整体实现流程包括：1)特征提取阶段，使用预训练模型提取视觉特征和几何特征；2)提示和边界框令牌编码；3)时空解码阶段，使用因果注意力块处理序列特征；4)多任务头阶段，包括深度/相机头、相机姿态头和3D边界框头；5)训练策略，包括随机裁剪序列和对象填充；6)损失函数设计，结合多种损失确保时空一致性。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)DA4D数据集，引入280k序列的高质量3D边界框标注；2)开放集端到端框架，直接从序列输入预测全局一致的3D边界框；3)时空解码器设计，融合多模态特征并建模时空动态；4)多任务学习架构和专门的训练策略。相比之前的工作，DetAny4D是端到端的而非多阶段流水线，考虑了时间维度而非单帧检测，直接从RGB序列工作而非依赖预扫描点云，且是通用的而非特定场景的方法。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'DetAny4D通过引入大规模DA4D数据集和设计开放集端到端的4D检测框架，实现了在流式RGB视频中检测任意物体的时空一致3D边界框，显著提高了检测的时间稳定性并减少了帧间抖动。'}


### 论文摘要

Reliable 4D object detection, which refers to 3D object detection in streaming video, is crucial for perceiving and understanding the real world. Existing open-set 4D object detection methods typically make predictions on a frame-by-frame basis without modeling temporal consistency, or rely on complex multi-stage pipelines that are prone to error propagation across cascaded stages. Progress in this area has been hindered by the lack of large-scale datasets that capture continuous reliable 3D bounding box (b-box) annotations. To overcome these challenges, we first introduce DA4D, a large-scale 4D detection dataset containing over 280k sequences with high-quality b-box annotations collected under diverse conditions. Building on DA4D, we propose DetAny4D, an open-set end-to-end framework that predicts 3D b-boxes directly from sequential inputs. DetAny4D fuses multi-modal features from pre-trained foundational models and designs a geometry-aware spatiotemporal decoder to effectively capture both spatial and temporal dynamics. Furthermore, it adopts a multi-task learning architecture coupled with a dedicated training strategy to maintain global consistency across sequences of varying lengths. Extensive experiments show that DetAny4D achieves competitive detection accuracy and significantly improves temporal stability, effectively addressing long-standing issues of jitter and inconsistency in 4D object detection. Data and code will be released upon acceptance.

---

## 69. Any4D: Open-Prompt 4D Generation from Natural Language and Images

**论文链接:** [http://arxiv.org/abs/2511.18746v1](http://arxiv.org/abs/2511.18746v1)

**作者:** Hao Li, Qiao Sun

**发布时间:** 2025-11-24

### GPT解析

### 总结

原始具身世界模型(PEWM)通过限制视频生成到固定较短的时间范围，解决了具身世界模型对大规模具身交互数据的依赖问题，实现了语言与动作的细粒度对齐，降低了学习复杂度，提高了数据效率，减少了推理延迟，并支持灵活的闭环控制和复杂任务上的策略组合泛化。

### 背景

基于视频生成的具身世界模型越来越受到关注，但它们依赖于大规模的具身交互数据，这是一个主要瓶颈。具身数据的稀缺、收集难度和高维度性限制了语言与动作之间的对齐粒度，加长了视频生成的挑战，阻碍了生成模型在具身领域实现'GPT时刻'。

### 目的

解决具身世界模型对大规模具身交互数据的依赖问题，实现语言与动作的细粒度对齐，降低学习复杂度，提高数据效率，减少推理延迟，并支持灵活的闭环控制和复杂任务上的策略组合泛化。

### 方法

提出原始具身世界模型(PEWM)，将视频生成限制在固定较短的时间范围内，并配备模块化的视觉语言模型(VLM)规划器和起点-目标热图引导机制(SGG)。

### 主要发现

具身数据的多样性远远超过了可能的原始动作的相对较小的空间，通过限制视频生成到较短的时间范围，可以实现语言概念和机器人动作视觉表示之间的细粒度对齐。

### 结论

PEWM框架利用视频模型中的时空视觉先验和VLM的语义感知能力，弥合了细粒度物理交互与高层推理之间的差距，为可扩展、可解释和通用的具身智能铺平了道路。

### 翻译

虽然基于视频生成的具身世界模型越来越受到关注，但它们对大规模具身交互数据的依赖仍然是一个主要瓶颈。具身数据的稀缺、收集难度和高维度性从根本上限制了语言与动作之间的对齐粒度，并加剧了长视频生成的挑战——阻碍了生成模型在具身领域实现'GPT时刻'。有一个简单的观察：具身数据的多样性远远超过了可能的原始动作的相对较小的空间。基于这一见解，我们提出了原始具身世界模型(PEWM)，它将视频生成限制在固定较短的时间范围内，我们的方法1)实现了语言概念与机器人动作视觉表示之间的细粒度对齐，2)降低了学习复杂度，3)提高了具身数据收集的数据效率，4)减少了推理延迟。通过配备模块化的视觉语言模型(VLM)规划器和起点-目标热图引导机制(SGG)，PEWM进一步支持灵活的闭环控制，并在扩展、复杂任务上支持原始级别策略的组合泛化。我们的框架利用视频模型中的时空视觉先验和VLM的语义感知能力，弥合了细粒度物理交互与高层推理之间的差距，为可扩展、可解释和通用的具身智能铺平了道路。


### 论文摘要

While video-generation-based embodied world models have gained increasing attention, their reliance on large-scale embodied interaction data remains a key bottleneck. The scarcity, difficulty of collection, and high dimensionality of embodied data fundamentally limit the alignment granularity between language and actions and exacerbate the challenge of long-horizon video generation--hindering generative models from achieving a \textit{"GPT moment"} in the embodied domain. There is a naive observation: \textit{the diversity of embodied data far exceeds the relatively small space of possible primitive motions}. Based on this insight, we propose \textbf{Primitive Embodied World Models} (PEWM), which restricts video generation to fixed shorter horizons, our approach \textit{1) enables} fine-grained alignment between linguistic concepts and visual representations of robotic actions, \textit{2) reduces} learning complexity, \textit{3) improves} data efficiency in embodied data collection, and \textit{4) decreases} inference latency. By equipping with a modular Vision-Language Model (VLM) planner and a Start-Goal heatmap Guidance mechanism (SGG), PEWM further enables flexible closed-loop control and supports compositional generalization of primitive-level policies over extended, complex tasks. Our framework leverages the spatiotemporal vision priors in video models and the semantic awareness of VLMs to bridge the gap between fine-grained physical interaction and high-level reasoning, paving the way toward scalable, interpretable, and general-purpose embodied intelligence.

---

## 70. GRIT-LP: Graph Transformer with Long-Range Skip Connection and Partitioned Spatial Graphs for Accurate Ice Layer Thickness Prediction

**论文链接:** [http://arxiv.org/abs/2511.18716v1](http://arxiv.org/abs/2511.18716v1)

**作者:** Zesheng Liu, Maryam Rahnemoonfar

**发布时间:** 2025-11-24

### GPT解析

### 总结

GRIT-LP是一种创新的图transformer方法，通过分区空间图构建策略和长程跳跃连接机制，有效解决了冰层厚度估计中的时空模式建模挑战，实现了显著的性能提升。

### 背景

Graph transformers在复杂的时空任务中表现出色，但它们的深度通常受到过平滑和弱长程依赖建模的限制。准确估计冰层厚度对于理解积雪积累、重建过去气候模式以及减少未来冰层演化和海平面上升预测的不确定性至关重要。

### 目的

为了解决这些挑战，作者引入了GRIT-LP，这是一种专门为从极地雷达图像中估计极地冰层厚度而设计的图transformer。

### 方法

GRIT-LP结合了归纳几何图学习框架与自注意力机制，并引入了两个主要创新：一种分区空间图构建策略，形成重叠的、全连接的局部邻域，以保持空间连贯性并抑制来自无关长程链接的噪声；以及transformer内部的长程跳跃连接机制，改善信息流并减轻更深注意力层中的过平滑问题。

### 主要发现

作者进行了大量实验，证明GRIT-LP比当前最先进的方法有了24.92%的均方根误差改进。

### 结论

这些结果突显了图transformer在通过捕获局部结构特征和内部冰层之间的长程依赖关系来建模时空模式方面的有效性，并展示了它们推进数据驱动的冰圈过程理解的潜力。

### 翻译

图transformers在复杂的时空任务中表现出色，但它们的深度通常受到过平滑和弱长程依赖建模的限制。为了应对这些挑战，我们引入了GRIT-LP，这是一种专门为从极地雷达图像中估计极地冰层厚度而设计的图transformer。准确估计冰层厚度对于理解积雪积累、重建过去气候模式和减少未来冰层演化和海平面上升预测的不确定性至关重要。GRIT-LP结合了归纳几何图学习框架与自注意力机制，并引入了两个主要创新，共同解决了建模冰层时空模式的挑战：一种分区空间图构建策略，形成重叠的、全连接的局部邻域，以保持空间连贯性并抑制来自无关长程链接的噪声；以及transformer内部的长程跳跃连接机制，改善信息流并减轻更深注意力层中的过平滑问题。我们进行了大量实验，证明GRIT-LP比当前最先进的方法有了24.92%的均方根误差改进。这些结果突显了图transformer在通过捕获局部结构特征和内部冰层之间的长程依赖关系来建模时空模式方面的有效性，并展示了它们推进数据驱动的冰圈过程理解的潜力。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决准确预测极地冰层厚度的问题，特别是从浅层冰层信息预测深层冰层厚度。这个问题在现实中非常重要，因为准确估计冰层厚度对于理解积雪积累、重建过去的气候模式至关重要，同时能减少未来冰层演化和海平面上升预测中的不确定性，对监测冰盖动态和应对气候变化研究具有重要意义。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先分析了现有图神经网络和图变压器在处理冰层数据时的局限性，包括过平滑问题和弱长程依赖建模能力不足。他们借鉴了Zalatan和Rahnemoonfar之前将冰层表示为独立图的工作，以及Liu和Rahnemoonfar的多分支时空图神经网络和图变压器方法。同时参考了DenseNet和U-Net中的跳跃连接思想，但将其适配到图变压器的场景。基于这些分析，作者设计了GRIT-LP，结合了几何图学习框架与自注意力机制，并通过分区空间图和长程跳跃连接来解决现有方法的局限性。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是结合几何图学习框架与自注意力机制，通过分区空间图和长程跳跃连接来解决图变压器的深度限制和长程依赖建模问题，利用浅层冰层的地理和厚度信息来预测深层冰层的厚度。整体流程包括：1)将每个冰层表示为分区空间图，节点包含纬度、经度和厚度信息；2)使用GraphSAGE提取每个冰层的空间特征并连接成空间特征嵌入；3)将空间特征输入到N个时间注意力块中捕获时间依赖；4)通过自适应长程跳跃连接平衡原始空间特征和转换后的时间特征；5)使用线性层进行最终预测。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)分区空间图构建策略，将每个冰层划分为固定大小、部分重叠的局部邻域，每个邻域内节点全连接，不同邻域间无连接；2)自适应长程跳跃连接，直接连接原始空间特征嵌入与每个时间注意力块的输出，使用可学习参数α动态平衡两种特征；3)更深的网络架构，通过移除显式空间注意力块并引入低成本跳跃连接，堆叠更多时间注意力块。相比之前工作，GRIT-LP不再使用全连接图和显式空间注意力块，而是采用分区空间图和跳跃连接，实现了更深的网络结构和更高的预测精度，比当前最先进方法提高了24.92%的RMSE。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'GRIT-LP通过引入分区空间图和自适应长程跳跃连接，有效解决了图变压器在冰层厚度预测中的过平滑和长程依赖建模问题，实现了比现有方法高24.92%的预测精度。'}


### 论文摘要

Graph transformers have demonstrated remarkable capability on complex spatio-temporal tasks, yet their depth is often limited by oversmoothing and weak long-range dependency modeling. To address these challenges, we introduce GRIT-LP, a graph transformer explicitly designed for polar ice-layer thickness estimation from polar radar imagery. Accurately estimating ice layer thickness is critical for understanding snow accumulation, reconstructing past climate patterns and reducing uncertainties in projections of future ice sheet evolution and sea level rise. GRIT-LP combines an inductive geometric graph learning framework with self-attention mechanism, and introduces two major innovations that jointly address challenges in modeling the spatio-temporal patterns of ice layers: a partitioned spatial graph construction strategy that forms overlapping, fully connected local neighborhoods to preserve spatial coherence and suppress noise from irrelevant long-range links, and a long-range skip connection mechanism within the transformer that improves information flow and mitigates oversmoothing in deeper attention layers. We conducted extensive experiments, demonstrating that GRIT-LP outperforms current state-of-the-art methods with a 24.92\% improvement in root mean squared error. These results highlight the effectiveness of graph transformers in modeling spatiotemporal patterns by capturing both localized structural features and long-range dependencies across internal ice layers, and demonstrate their potential to advance data-driven understanding of cryospheric processes.

---

## 71. X-ReID: Multi-granularity Information Interaction for Video-Based Visible-Infrared Person Re-Identification

**论文链接:** [http://arxiv.org/abs/2511.17964v2](http://arxiv.org/abs/2511.17964v2)

**作者:** Chenyang Yu, Xuehu Liu, Pingping Zhang, Huchuan Lu

**发布时间:** 2025-11-22

**备注:** Accepted by AAAI2026. More modifications may be performed

### GPT解析

### 总结

本文提出了X-ReID框架，用于解决视频可见光-红外人员再识别问题，通过跨模态原型协作和多粒度信息交互方法，有效缩小了模态差距并利用了视频序列中的时空信息。

### 背景

大规模视觉-语言模型（如CLIP）在检索任务中表现出色，但在视频可见光-红外人员再识别（VVI-ReID）方面的潜力尚未被充分探索。

### 目的

提出一个新的跨模态特征学习框架X-ReID，用于解决VVI-ReID问题中的模态差距和时空信息利用挑战。

### 方法

提出跨模态原型协作（CPC）对齐和整合不同模态特征，设计多粒度信息交互（MII）融合短期和长期信息，通过整合多粒度信息实现鲁棒的序列级表示。

### 主要发现

在两个大规模VVI-ReID基准数据集（HITSZ-VCM和BUPTCampus）上的实验证明，所提方法优于最先进的方法。

### 结论

X-ReID框架有效解决了视频可见光-红外人员再识别中的模态差距和时空信息利用问题，为该领域提供了新的解决方案。

### 翻译

大规模视觉-语言模型（如CLIP）最近在检索任务中取得了显著性能，但它们在视频可见光-红外人员再识别（VVI-ReID）方面的潜力仍未被充分探索。主要挑战是缩小模态差距和利用视频序列中的时空信息。为解决上述问题，本文提出了一个名为X-ReID的跨模态特征学习框架用于VVI-ReID。具体来说，首先提出跨模态原型协作（CPC）来对齐和整合不同模态的特征，引导网络减少模态差异。然后，设计了多粒度信息交互（MII），融合相邻帧的短期交互、跨帧长期信息融合和跨模态特征对齐，以增强时间建模并进一步减少模态差距。最后，通过整合多粒度信息，实现了鲁棒的序列级表示。在两个大规模VVI-ReID基准（即HITSZ-VCM和BUPTCampus）上的大量实验证明了我们的方法优于最先进的方法。源代码已在https://github.com/AsuradaYuci/X-ReID发布。


### 论文摘要

Large-scale vision-language models (e.g., CLIP) have recently achieved remarkable performance in retrieval tasks, yet their potential for Video-based Visible-Infrared Person Re-Identification (VVI-ReID) remains largely unexplored. The primary challenges are narrowing the modality gap and leveraging spatiotemporal information in video sequences. To address the above issues, in this paper, we propose a novel cross-modality feature learning framework named X-ReID for VVI-ReID. Specifically, we first propose a Cross-modality Prototype Collaboration (CPC) to align and integrate features from different modalities, guiding the network to reduce the modality discrepancy. Then, a Multi-granularity Information Interaction (MII) is designed, incorporating short-term interactions from adjacent frames, long-term cross-frame information fusion, and cross-modality feature alignment to enhance temporal modeling and further reduce modality gaps. Finally, by integrating multi-granularity information, a robust sequence-level representation is achieved. Extensive experiments on two large-scale VVI-ReID benchmarks (i.e., HITSZ-VCM and BUPTCampus) demonstrate the superiority of our method over state-of-the-art methods. The source code is released at https://github.com/AsuradaYuci/X-ReID.

---

## 72. Test-Time Temporal Sampling for Efficient MLLM Video Understanding

**论文链接:** [http://arxiv.org/abs/2511.17945v1](http://arxiv.org/abs/2511.17945v1)

**作者:** Kaibin Wang, Mingbao Lin

**发布时间:** 2025-11-22

### GPT解析

### 总结

T3S是一种创新的推理时间采样方法，解决了多模态大语言模型处理长视频时的计算效率问题，同时保持准确性。该方法完全在推理时运行，无需模型修改或微调，将视频冗余转化为计算优势。

### 背景

处理长视频时，多模态大语言模型面临显著计算挑战，因为模型的自我注意力机制计算复杂度与视频标记数量成二次方关系，导致高计算需求和慢推理速度。现有解决方案如基于规则的子采样、学习型帧选择器或基于内存的摘要都存在权衡：牺牲准确性、需要额外训练或降低推理速度。

### 目的

提出一种训练免费、即插即用的推理包装器，使多模态大语言模型能够高效有效地处理长视频。

### 方法

提出Test-Time Temporal Sampling (T3S)，通过利用时空冗余，在推理时生成多个短而多样的视频标记子序列，将这些子序列打包在单个前向传播中，并聚合它们的预测。这种多子序列公式扩大了视觉覆盖范围，同时将自注意力的计算成本从二次方降低到线性关系。

### 主要发现

在长视频理解基准上的广泛实验表明，T3S将准确性提高了高达3.1%，将首个标记延迟减少了2.04倍，且只需最少的集成工作。

### 结论

T3S完全在推理时运行，不需要模型修改或微调，与各种预训练多模态大语言模型兼容，为长视频理解提供可扩展的解决方案。代码可在https://github.com/kaibinwang3/T3S获取。

### 翻译

使用多模态大语言模型处理长视频带来了显著的计算挑战，因为模型的自我注意力机制计算量与视频标记数量成二次方关系，导致高计算需求和慢推理速度。当前解决方案，如基于规则的子采样、学习型帧选择器或基于内存的摘要，往往引入各自的权衡：它们牺牲准确性、需要额外训练或降低推理速度。在本文中，我们提出了Test-Time Temporal Sampling (T3S)，一种训练免费、即插即用的推理包装器，使多模态大语言模型能够高效有效地处理长视频。T3S通过在推理时生成多个短而多样的视频标记子序列，利用时空冗余，将它们打包在单个前向传播中，并聚合它们的预测。这种多子序列公式扩大了视觉覆盖范围，同时将自注意力的计算成本从O(L²)降低到O(∑αᵢ²L²)，其中∑αᵢ² < 1。在长视频理解基准上的广泛实验表明，T3S将准确性提高了高达3.1%，将首个标记延迟减少了2.04倍，且只需最少的集成工作。我们的方法完全在推理时运行，不需要模型修改或微调，与各种预训练多模态大语言模型兼容。T3S将视频冗余转化为计算优势，为长视频理解提供可扩展的解决方案。代码可在https://github.com/kaibinwang3/T3S获取。


### 论文摘要

Processing long videos with multimodal large language models (MLLMs) poses a significant computational challenge, as the model's self-attention mechanism scales quadratically with the number of video tokens, resulting in high computational demand and slow inference speed. Current solutions, such as rule-based sub-sampling, learned frame selector, or memory-based summarization, often introduce their own trade-offs: they compromise accuracy, necessitate additional training, or decrease inference speed. In this paper, we propose Test-Time Temporal Sampling (T3S), a training-free, plug-and-play inference wrapper that enables MLLMs to process long videos both efficiently and effectively. T3S exploits spatiotemporal redundancy by generating multiple short and diverse subsequences of video tokens at inference time, packing them within a single forward pass, and aggregating their predictions. This multi-subsequence formulation broadens visual coverage while reducing the computational cost of self-attention from $O(L^2)$ to $O(\sum_{i=1}^m α_i^2L^2)$, where $\sum_{i=1}^m α_i^2 < 1$. Extensive experiments on long video understanding benchmarks demonstrate that T3S improves accuracy by up to 3.1% and reduces first token delay by $2.04\times$, all with minimal integration effort. Our approach operates entirely at inference time, requires no model modifications or fine-tuning, and is compatible with a wide range of pretrained MLLMs. T3S turns video redundancy into a computational advantage, offering a scalable solution for long-video understanding. The code is available at https://github.com/kaibinwang3/T3S.

---

## 73. Scaling Kinetic Monte-Carlo Simulations of Grain Growth with Combined Convolutional and Graph Neural Networks

**论文链接:** [http://arxiv.org/abs/2511.17848v1](http://arxiv.org/abs/2511.17848v1)

**作者:** Zhihui Tian, Ethan Suwandi, Tomas Oppelstrup, Vasily V. Bulatov, Joel B. Harley, Fei Zhou

**发布时间:** 2025-11-22

### GPT解析

### 总结

研究提出了一种结合双射自编码器和图神经网络的混合架构，用于微结构模拟，显著降低了计算成本并提高了准确性。

### 背景

图神经网络已成为微结构模拟如晶粒生长的有前景方法，但准确建模真实晶界网络需要大型模拟单元，而GNN难以扩展到这种规模。

### 目的

减轻GNN的计算成本和内存占用，开发一种能够有效处理大型微结构模拟的混合架构。

### 方法

提出结合基于CNN的双射自编码器压缩空间维度，以及在减小空间尺寸的潜在空间中演化的GNN；从随机Potts Monte Carlo方法中学习进行优化训练。

### 主要发现

新设计显著降低计算成本，消息传递层从12层减少到3层；随空间尺寸增加，计算成本减少更明显；对最大网格(160^3)，内存使用和运行时间分别减少117倍和115倍；相比仅用GNN，具有更高准确性和更强时空能力；双射自编码器能无损压缩信息并提供更具表现力的潜在特征。

### 结论

这种结合可扩展性和准确性的方法对模拟长时间尺度的真实材料微结构至关重要，为晶粒生长模拟提供了高度可扩展的方法。

### 翻译

图神经网络已成为微结构模拟如晶粒生长的有前景机器学习方法。然而，准确建模真实的晶界网络需要大的模拟单元，而GNN难以扩展到这种规模。为了减轻GNN的计算成本和内存占用，我们提出了一种混合架构，结合基于卷积神经网络的双射自编码器来压缩空间维度，以及一个在减小空间尺寸的潜在空间中演化的GNN。我们的结果表明，与单独使用GNN相比，新设计显著降低了计算成本，使用了更少的消息传递层（从12层减少到3层）。随着空间尺寸的增加，计算成本的减少变得更加明显，显示出强大的计算可扩展性。对于评估的最大网格（160^3），与仅使用GNN的基线相比，我们的方法将推理时的内存使用和运行时间分别减少了117倍和115倍。更重要的是，与仅使用GNN的基线相比，它显示出更高的准确性和更强的时空能力，特别是在长期测试中。这种可扩展性和准确性的结合对于模拟长时间尺度的真实材料微结构至关重要。改进可归因于双射自编码器能够无损地将信息从空间域压缩到高维特征空间，从而为GNN提供更具表现力的潜在特征，同时也贡献其自身的时空建模能力。训练经过优化，可以从随机Potts Monte Carlo方法中学习。我们的发现为晶粒生长模拟提供了一种高度可扩展的方法。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决图神经网络(GNN)在模拟晶粒生长时难以扩展到大模拟单元的问题，导致计算成本和内存占用过高。这个问题很重要，因为多晶材料构成了现代工程材料的主体，其物理性质与晶粒微观结构密切相关，而准确模拟这种结构对开发新材料至关重要。真实晶界结构复杂且需要大模拟单元才能准确建模，但现有GNN方法难以处理这种规模。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者观察到GNN在晶粒生长模拟中表现良好但难以扩展，同时CNN在特征提取和空间压缩方面有优势。因此设计了一个混合架构，结合CNN的双射自编码器和GNN。他们借鉴了MeshGraphNet模型用于图结构模拟，参考了双射自编码器的无损压缩方法，并采用了噪声注入、多步自监督损失和对称性数据增强等训练策略，这些都是基于现有工作的创新组合。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是使用CNN的双射自编码器压缩空间维度，在压缩后的潜在空间中用简化的GNN进行时间演化预测，并利用双射自编码器的可逆性设计两种推理策略。整体流程：1)数据准备：生成PMC晶粒粗化轨迹并进行后处理和降采样；2)模型构建：设计双射自编码器压缩特征，简化GNN在潜在空间演化；3)训练：采用噪声注入、多步损失和对称性增强；4)推理：可选择在原始空间或潜在空间进行预测，后者效率更高。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点：1)混合架构首次结合CNN双射自编码器和GNN实现无损压缩；2)显著提高计算效率，最大案例内存和运行时间减少117倍和115倍；3)将GNN消息传递层从12层减少到3层同时提高准确性；4)针对随机PMC数据设计多步训练策略。相比之前工作：解决了GNN难以扩展的问题；专门针对随机数据而非确定性数据；采用多步监督而非单步监督；通过架构创新简化模型同时提高性能。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '本文提出了一种结合卷积神经网络双射自编码器和图神经网络的混合架构，显著提高了晶粒生长模拟的计算效率和准确性，使其能够处理更大规模的微观结构模拟，同时保持甚至提高了长期预测能力。'}


### 论文摘要

Graph neural networks (GNN) have emerged as a promising machine learning method for microstructure simulations such as grain growth. However, accurate modeling of realistic grain boundary networks requires large simulation cells, which GNN has difficulty scaling up to. To alleviate the computational costs and memory footprint of GNN, we propose a hybrid architecture combining a convolutional neural network (CNN) based bijective autoencoder to compress the spatial dimensions, and a GNN that evolves the microstructure in the latent space of reduced spatial sizes. Our results demonstrate that the new design significantly reduces computational costs with using fewer message passing layer (from 12 down to 3) compared with GNN alone. The reduction in computational cost becomes more pronounced as the spatial size increases, indicating strong computational scalability. For the largest mesh evaluated (160^3), our method reduces memory usage and runtime in inference by 117x and 115x, respectively, compared with GNN-only baseline. More importantly, it shows higher accuracy and stronger spatiotemporal capability than the GNN-only baseline, especially in long-term testing. Such combination of scalability and accuracy is essential for simulating realistic material microstructures over extended time scales. The improvements can be attributed to the bijective autoencoder's ability to compress information losslessly from spatial domain into a high dimensional feature space, thereby producing more expressive latent features for the GNN to learn from, while also contributing its own spatiotemporal modeling capability. The training was optimized to learn from the stochastic Potts Monte Carlo method. Our findings provide a highly scalable approach for simulating grain growth.

---

## 74. Material-informed Gaussian Splatting for 3D World Reconstruction in a Digital Twin

**论文链接:** [http://arxiv.org/abs/2511.20348v1](http://arxiv.org/abs/2511.20348v1)

**作者:** João Malheiro Silva, Andy Huynh, Tong Duy Son, Holger Caesar

**发布时间:** 2025-11-25

**备注:** 8 pages, 5 figures. Submitted to IEEE Intelligent Vehicles Symposium (IV) 2026 for possible publication

### GPT解析

### 总结

该研究提出了一种仅使用相机的3D重建方法，通过3D高斯飞溅技术从多视图图像重建场景，提取语义材料掩码，并将高斯表示转换为带材料标签的网格表面，为数字孪生提供逼真的传感器仿真。

### 背景

3D重建通常依赖LiDAR方法，能提供准确几何信息但缺乏语义和纹理；传统LiDAR-相机融合方法需复杂校准，且难以处理玻璃等在图像中可见但点云中表现不佳的材料。

### 目的

开发一种仅使用相机的管道，结合逼真重建和基于物理的材料分配，提供与LiDAR-相机融合相当的传感器仿真保真度，同时消除硬件复杂性和校准要求。

### 方法

使用多视图图像通过3D高斯飞溅重建场景，通过视觉模型提取语义材料掩码，将高斯表示转换为带有投影材料标签的网格表面，并为现代图形引擎和仿真器分配基于物理的材料属性。

### 主要发现

仅使用相机的方法可实现与LiDAR-相机融合相当的传感器仿真保真度；使用配备仪器的测试车辆内部数据集验证了该方法，利用LiDAR作为反射率验证的真实数据，同时使用图像相似度指标评估。

### 结论

仅使用相机的方法结合了逼真的重建和基于物理的材料分配，提供了与LiDAR-相机融合相当的传感器仿真保真度，同时消除了硬件复杂性和校准要求。

### 翻译

用于数字孪生的3D重建通常依赖于基于LiDAR的方法，这些方法能提供准确的几何信息，但缺乏相机自然捕捉的语义和纹理。传统的LiDAR-相机融合方法需要复杂的校准，并且在处理玻璃等某些材料时仍然存在困难，这些材料在图像中可见但在点云中表现不佳。我们提出了一种仅使用相机的管道，它使用多视图图像通过3D高斯飞溅重建场景，通过视觉模型提取语义材料掩码，将高斯表示转换为带有投影材料标签的网格表面，并为现代图形引擎和仿真器中的传感器仿真分配基于物理的材料属性。这种方法将逼真的重建与基于物理的材料分配相结合，提供了与LiDAR-相机融合相当的传感器仿真保真度，同时消除了硬件复杂性和校准要求。我们使用配备仪器的测试车辆的内部数据集验证了这种仅使用相机的方法，利用LiDAR作为反射率验证的真实数据，并结合图像相似度指标进行评估。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决数字孪生环境中的3D世界重建问题，特别是传统基于LiDAR的方法缺乏纹理信息，以及LiDAR在处理透明和反射材料（如玻璃、金属）时的局限性。这个问题很重要，因为数字孪生对于自动驾驶系统的安全验证和高风险场景测试至关重要，准确的3D重建和传感器模拟能确保虚拟传感器行为与物理传感器一致，从而提高ADAS和AI系统的可靠性。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先分析了现有方法的局限性：LiDAR方法缺乏纹理信息，LiDAR-相机融合需要复杂校准，传统方法在稀疏视角下纹理质量有限。他们注意到3D高斯溅射技术能提供高质量重建，而MILo等研究可从中提取几何表面。因此，他们设计了一种仅使用相机的管道，结合照片真实感的高斯溅射和基于物理的材料分配。该方法借鉴了多项现有工作，包括3D高斯溅射技术、H3DGS可视化、MiLO网格提取、RMSNet材料分割和FastSAM形状感知细化等。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是使用仅相机的管道，结合照片真实感的高斯溅射重建和基于物理的材料分配，实现准确的传感器模拟，同时消除对LiDAR硬件的依赖。整体流程分为五步：1)单目材料提取：使用RMSNet和FastSAM从RGB图像提取材料标签；2)大规模高斯溅射重建：用H3DGS和MiLO重建场景；3)逐像素材料投影：将2D材料掩码投影到3D网格表面；4)基于物理的材料分配：使用Principled BSDF着色器分配材料属性；5)模拟验证：在Simcenter Prescan中验证传感器模拟准确性。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)模块化仅相机管道，整合高斯溅射重建和自动材料分配；2)自动化2D到3D材料投影方法，实现准确的基于物理的LiDAR反射率模拟；3)全面评估证明传感器模拟准确性与LiDAR-相机融合相当。相比之前工作，此方法不需要LiDAR硬件，避免了复杂校准；能更好处理透明和反射材料；在稀疏视角下提供更好纹理质量；结合了照片真实感渲染和基于物理的材料分配，适用于现代图形引擎。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '这篇论文提出了一种仅使用相机的管道，通过结合照片真实感的高斯溅射重建和基于物理的材料分配，实现了数字孪生环境中与LiDAR-相机融合相当的传感器模拟准确性，同时消除了对LiDAR硬件和复杂校准的需求。'}


### 论文摘要

3D reconstruction for Digital Twins often relies on LiDAR-based methods, which provide accurate geometry but lack the semantics and textures naturally captured by cameras. Traditional LiDAR-camera fusion approaches require complex calibration and still struggle with certain materials like glass, which are visible in images but poorly represented in point clouds. We propose a camera-only pipeline that reconstructs scenes using 3D Gaussian Splatting from multi-view images, extracts semantic material masks via vision models, converts Gaussian representations to mesh surfaces with projected material labels, and assigns physics-based material properties for accurate sensor simulation in modern graphics engines and simulators. This approach combines photorealistic reconstruction with physics-based material assignment, providing sensor simulation fidelity comparable to LiDAR-camera fusion while eliminating hardware complexity and calibration requirements. We validate our camera-only method using an internal dataset from an instrumented test vehicle, leveraging LiDAR as ground truth for reflectivity validation alongside image similarity metrics.

---

## 75. Zoo3D: Zero-Shot 3D Object Detection at Scene Level

**论文链接:** [http://arxiv.org/abs/2511.20253v1](http://arxiv.org/abs/2511.20253v1)

**作者:** Andrey Lemeshko, Bulat Gabdullin, Nikita Drozdov, Anton Konushin, Danila Rukhovich, Maksim Kolodiazhnyi

**发布时间:** 2025-11-25

### GPT解析

### 总结

本文提出了Zoo3D，首个无需训练的3D物体检测框架，通过图聚类构建3D边界框并使用开放词汇模块分配语义标签，包括零样本和自监督两种模式，在多个基准测试上取得最先进结果。

### 背景

3D物体检测对空间理解至关重要，现实环境需要能识别多样化、前所未见物体的模型，而封闭集方法存在这一局限性。现有开放词汇3D检测器虽降低标注要求，但仍依赖训练场景。

### 目的

开发一个不依赖训练场景的3D物体检测框架，能够识别多样化的未见物体，并扩展到直接处理图像数据。

### 方法

通过2D实例掩码的图聚类构建3D边界框，使用包含最佳视图选择和视图共识掩码生成的开放词汇模块分配语义标签。提供两种模式：零样本的Zoo3D₀（完全无需训练）和自监督的Zoo3D₁（使用伪标签训练类无关检测器进行优化）。

### 主要发现

在ScanNet200和ARKitScenes基准测试上，Zoo3D₀和Zoo3D₁都取得了最先进的开放词汇3D物体检测结果。特别地，零样本的Zoo3D₀性能优于所有现有的自监督方法。

### 结论

无需训练、即用型方法在现实世界3D理解中具有强大能力和适应性，展示了训练-free方法的有效性。

### 翻译

三维物体检测对空间理解至关重要。现实环境需要能够识别多样化、前所未见物体的模型，这仍然是封闭集方法的主要局限性。现有的开放词汇三维检测器降低了标注要求，但仍依赖于训练场景，无论是点云还是图像。我们通过引入Zoo3D更进一步，这是首个无需训练的三维物体检测框架。我们的方法通过二维实例掩码的图聚类构建三维边界框，然后使用新颖的开放词汇模块分配语义标签，该模块包含最佳视图选择和视图共识掩码生成。Zoo3D以两种模式运行：零样本的Zoo3D₀，完全不需要训练；以及自监督的Zoo3D₁，通过在Zoo3D₀生成的伪标签上训练类无关检测器来优化三维边界框预测。此外，我们将Zoo3D扩展到点云之外，可直接处理有位姿甚至无位姿的图像。在ScanNet200和ARKitScenes基准测试中，Zoo3D₀和Zoo3D₁在开放词汇三维物体检测中都取得了最先进的结果。值得注意的是，我们的零样本Zoo3D₀性能优于所有现有的自监督方法，从而展示了无需训练、即用型方法在现实世界三维理解中的强大能力和适应性。代码可在https://github.com/col14m/zoo3d获取。


### 论文摘要

3D object detection is fundamental for spatial understanding. Real-world environments demand models capable of recognizing diverse, previously unseen objects, which remains a major limitation of closed-set methods. Existing open-vocabulary 3D detectors relax annotation requirements but still depend on training scenes, either as point clouds or images. We take this a step further by introducing Zoo3D, the first training-free 3D object detection framework. Our method constructs 3D bounding boxes via graph clustering of 2D instance masks, then assigns semantic labels using a novel open-vocabulary module with best-view selection and view-consensus mask generation. Zoo3D operates in two modes: the zero-shot Zoo3D$_0$, which requires no training at all, and the self-supervised Zoo3D$_1$, which refines 3D box prediction by training a class-agnostic detector on Zoo3D$_0$-generated pseudo labels. Furthermore, we extend Zoo3D beyond point clouds to work directly with posed and even unposed images. Across ScanNet200 and ARKitScenes benchmarks, both Zoo3D$_0$ and Zoo3D$_1$ achieve state-of-the-art results in open-vocabulary 3D object detection. Remarkably, our zero-shot Zoo3D$_0$ outperforms all existing self-supervised methods, hence demonstrating the power and adaptability of training-free, off-the-shelf approaches for real-world 3D understanding. Code is available at https://github.com/col14m/zoo3d .

---

## 76. FLaTEC: Frequency-Disentangled Latent Triplanes for Efficient Compression of LiDAR Point Clouds

**论文链接:** [http://arxiv.org/abs/2511.20065v1](http://arxiv.org/abs/2511.20065v1)

**作者:** Xiaoge Zhang, Zijie Wu, Mingtao Feng, Zichen Geng, Mehwish Nasim, Saeed Anwar, Ajmal Mian

**发布时间:** 2025-11-25

### GPT解析

### 总结

FLaTEC是一种频率感知的点云压缩模型，通过解耦低频结构和高频纹理，结合三平面表示和频率分离技术，实现了高压缩率的同时保持高质量重建。

### 背景

点云压缩方法通常联合优化比特率和重建失真，但平衡压缩比和重建质量很困难，因为低频和高频成分在同一分辨率下贡献不同。

### 目的

提出一种频率感知的压缩模型FLaTEC，能够以高压缩比压缩完整扫描的点云数据。

### 方法

引入频率感知机制解耦低频结构和高频纹理；使用潜在三平面作为点云的紧凑代理；将体素化嵌入转换为三平面表示减少稀疏性、计算成本和存储需求；设计频率分离技术提取低频内容和收集高频细节；以二进制格式存储解耦的组件；通过调制块逐步恢复全谱信号；引入基于频率的注意力机制补偿3D相关性损失。

### 主要发现

FLaTEC实现了最先进的率失真性能；在SemanticKITTI和Ford数据集上，比标准编解码器分别提高了78%和94%的BD-rate。

### 结论

FLaTEC是一种有效的点云压缩方法，能够在保持高质量重建的同时实现高压缩比。

### 翻译

点云压缩方法联合优化比特率和重建失真。然而，平衡压缩比和重建质量很困难，因为低频和高频成分在同一分辨率下贡献不同。为此，我们提出FLaTEC，一种频率感知的压缩模型，能够以高压缩比压缩完整扫描。我们的方法引入了频率感知机制，解耦低频结构和高频纹理，同时将潜在三平面混合作为点云的紧凑代理。具体来说，我们将体素化嵌入转换为三平面表示，以减少稀疏性、计算成本和存储需求。然后，我们设计了一种频率分离技术，提取紧凑的低频内容，同时收集跨尺度的高频细节。解耦的低频和高频组件以二进制格式存储。在解码过程中，通过调制块逐步恢复全谱信号。此外，为了补偿3D相关性的损失，我们引入了一种高效的基于频率的注意力机制，促进局部连接并输出任意分辨率的点。我们的方法实现了最先进的率失真性能，在SemanticKITTI和Ford数据集上，比标准编解码器分别提高了78%和94%的BD-rate。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决激光雷达点云的高效压缩问题。由于高分辨率激光雷达扫描会产生大量数据，给存储和传输带来显著挑战，这一问题在自动驾驶、机器人、虚拟现实等领域至关重要。高效压缩技术能够减少数据存储需求，加速数据传输，使实时处理大规模3D点云数据成为可能。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先分析了现有方法的局限性：点云方法难以扩展到大规模数据，体素方法存在立方级计算复杂度，而传统方法在空间域编码所有频率组件导致次优压缩权衡。作者借鉴了三平面表示受PCA启发将3D投影到2D平面，参考了图像处理中的频率分析方法，并改进了现有的注意力机制。设计时，作者考虑了激光雷达数据的稀疏性，通过解耦低频结构和高频纹理来优化比特率分配，同时引入基于频率的注意力机制提高重建质量。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是通过频率解耦的三平面表示实现高效压缩，将点云的低频结构和高频纹理分别处理，实现自适应比特率分配。整体流程包括：1) 将3D体素投影到三个正交平面；2) 使用频率解耦编码器分离低频和高频组件；3) 将解耦组件以二进制格式存储；4) 通过频率调制解码器逐步恢复全谱信号；5) 应用基于频率的局部注意力机制增强3D连接性；6) 最后通过灵活分辨率提升器输出任意分辨率的点云。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1) 频率感知的压缩模型，解耦低频和高频组件；2) 三平面表示减少计算复杂度和存储需求；3) 阶段性频率分解和集成机制实现连续频谱表示；4) 基于频率的局部注意力机制增强重建质量；5) 灵活分辨率提升器支持任意分辨率输出。相比之前工作，不同之处在于：传统方法对所有频率组件使用相同分辨率编码，而FLaTEC能针对不同频率分配不同比特率；三平面表示将计算复杂度从立方级降到二次级；频率解耦机制比现有方法提供更精确的频谱控制；局部频谱注意力机制比传统3D卷积更高效。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'FLaTEC通过频率解耦的三平面表示和基于频率的注意力机制，实现了高效的高压缩率激光雷达点云压缩，在保持高重建质量的同时显著减少了文件大小和编解码时间。'}


### 论文摘要

Point cloud compression methods jointly optimize bitrates and reconstruction distortion. However, balancing compression ratio and reconstruction quality is difficult because low-frequency and high-frequency components contribute differently at the same resolution. To address this, we propose FLaTEC, a frequency-aware compression model that enables the compression of a full scan with high compression ratios. Our approach introduces a frequency-aware mechanism that decouples low-frequency structures and high-frequency textures, while hybridizing latent triplanes as a compact proxy for point cloud. Specifically, we convert voxelized embeddings into triplane representations to reduce sparsity, computational cost, and storage requirements. We then devise a frequency-disentangling technique that extracts compact low-frequency content while collecting high-frequency details across scales. The decoupled low-frequency and high-frequency components are stored in binary format. During decoding, full-spectrum signals are progressively recovered via a modulation block. Additionally, to compensate for the loss of 3D correlation, we introduce an efficient frequency-based attention mechanism that fosters local connectivity and outputs arbitrary resolution points. Our method achieves state-of-the-art rate-distortion performance and outperforms the standard codecs by 78\% and 94\% in BD-rate on both SemanticKITTI and Ford datasets.

---

## 77. Updatable Balanced Index for Fast On-device Search with Auto-selection Model

**论文链接:** [http://arxiv.org/abs/2511.20049v1](http://arxiv.org/abs/2511.20049v1)

**作者:** Yushuai Ji, Sheng Wang, Zhiyu Chen, Yuan Sun, Zhiyong Peng

**发布时间:** 2025-11-25

**备注:** Accepted for publication in the 42nd IEEE International Conference on Data Engineering (ICDE 2026). To appear

### GPT解析

### 总结

本文提出了一种名为UnIS的新方法，用于解决边缘设备上BMKD-tree索引的局限性，实现了显著的性能提升。

### 背景

边缘设备上的传感器收集多种类型的边缘数据（如2D地理位置和3D点云），设备上搜索（如kNN搜索和半径搜索）用于快速分析和学习技术。BMKD-tree是保持高搜索效率的代表性方法，但存在构建开销大、实时插入不灵活和查询性能不一致等问题。

### 目的

解决BMKD-tree索引的局限性，提高边缘设备上的数据搜索和处理效率。

### 方法

1) 利用数据集分布预测分割超平面加速BMKD-tree构建；2) 提出选择性子树重建方案减少插入过程中的数据点数量；3) 设计自选择模型自动选择最优搜索策略提高查询性能。

### 主要发现

与BMKD-tree相比，UnIS实现了：索引构建加速17.96倍，插入加速1.60倍，kNN搜索加速7.15倍，半径搜索加速1.09倍；在数据集简化应用上比Lloyd算法快217倍。

### 结论

UnIS有效解决了BMKD-tree的局限性，显著提高了边缘设备上的数据搜索和处理效率，适用于实时分析场景。

### 翻译

边缘设备上的传感器（如激光雷达和GPS接收器）收集多种类型的边缘数据，如2D地理位置和3D点云。设备上搜索，如k近邻(kNN)搜索和半径搜索，通常用于快速分析和学习技术，如使用kNN的k-means数据集简化。为了保持高搜索效率，代表性的方法是使用平衡多路KD树(BMKD-tree)。然而，该索引显示出有限的增益，主要由于构建开销大、实时插入不灵活以及查询性能不一致。在本文中，我们提出UnIS来解决上述限制。我们首先利用数据集分布来预测分割超平面，从而加速BMKD-tree的构建过程。为了使持续生成的数据可搜索，我们提出了一种选择性子树重建方案，通过减少涉及的数据点数量来加速插入过程中的重新平衡。然后，我们提出一个自选择模型，通过为任意查询任务在多种策略中自动选择最优搜索策略来提高查询性能。实验结果表明，与BMKD-tree相比，UnIS在索引构建、插入、kNN搜索和半径搜索方面分别实现了平均17.96倍、1.60倍、7.15倍和1.09倍的加速。我们进一步验证了其在边缘设备上加速数据集简化的有效性，相对于Lloyd算法实现了217倍的加速。


### 论文摘要

Diverse types of edge data, such as 2D geo-locations and 3D point clouds, are collected by sensors like lidar and GPS receivers on edge devices. On-device searches, such as k-nearest neighbor (kNN) search and radius search, are commonly used to enable fast analytics and learning technologies, such as k-means dataset simplification using kNN. To maintain high search efficiency, a representative approach is to utilize a balanced multi-way KD-tree (BMKD-tree). However, the index has shown limited gains, mainly due to substantial construction overhead, inflexibility to real-time insertion, and inconsistent query performance. In this paper, we propose UnIS to address the above limitations. We first accelerate the construction process of the BMKD-tree by utilizing the dataset distribution to predict the splitting hyperplanes. To make the continuously generated data searchable, we propose a selective sub-tree rebuilding scheme to accelerate rebalancing during insertion by reducing the number of data points involved. We then propose an auto-selection model to improve query performance by automatically selecting the optimal search strategy among multiple strategies for an arbitrary query task. Experimental results show that UnIS achieves average speedups of 17.96x in index construction, 1.60x in insertion, 7.15x in kNN search, and 1.09x in radius search compared to the BMKD-tree. We further verify its effectiveness in accelerating dataset simplification on edge devices, achieving a speedup of 217x over Lloyd's algorithm.

---

## 78. MFM-point: Multi-scale Flow Matching for Point Cloud Generation

**论文链接:** [http://arxiv.org/abs/2511.20041v1](http://arxiv.org/abs/2511.20041v1)

**作者:** Petr Molodyk, Jaemoo Choi, David W. Romero, Ming-Yu Liu, Yongxin Chen

**发布时间:** 2025-11-25

### GPT解析

### 总结

本文提出了MFM-Point，一个用于点云生成的多尺度流匹配框架，显著提高了基于点的方法的性能和可扩展性，同时保持了简单性和效率。

### 背景

近年来点云生成在3D生成建模中受到广泛关注。现有基于点的方法直接生成点云而不依赖其他表示，具有训练成本低和算法简单的优点，但通常表现不如基于表示的方法。

### 目的

开发一个能提高基于点方法性能和可扩展性，同时保持其简单性和效率的点云生成框架。

### 方法

提出MFM-Point多尺度流匹配框架，采用从粗到细的生成范式增强生成质量和可扩展性。引入结构化下采样和上采样策略，保留几何结构并保持不同分辨率间的一致性。

### 主要发现

实验表明MFM-Point在基于点的方法中实现了最先进性能，并挑战了最佳基于表示的方法。特别是在多类别和高分辨率生成任务中表现突出。

### 结论

MFM-Point通过多尺度生成算法和结构化采样策略，成功解决了基于点方法在点云生成中的局限性，实现了高质量点云生成。

### 翻译

近年来，点云生成在3D生成建模中受到了广泛关注。在现有方法中，基于点的方法直接生成点云，而不依赖于潜在特征、网格或体素等其他表示。这些方法具有训练成本低和算法简单的优点，但通常与基于表示的方法相比表现较差。在本文中，我们提出了MFM-Point，一个用于点云生成的多尺度流匹配框架，显著提高了基于点的方法的可扩展性和性能，同时保留了它们的简单性和效率。我们的多尺度生成算法采用从粗到细的生成范式，增强了生成质量和可扩展性，而不增加额外的训练或推理开销。开发这样一个多尺度框架的一个关键挑战在于，在保持无序点云的几何结构的同时，确保跨分辨率的平滑和一致的分布转换。为解决这一问题，我们引入了一种结构化的下采样和上采样策略，以保留几何结构并保持粗分辨率和细分辨率之间的一致性。我们的实验结果表明，MFM-Point在基于点的方法中实现了最先进的性能，并挑战了最佳基于表示的方法。特别是，MFM-Point在多类别和高分辨率生成任务中表现出强大的结果。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决点云生成中点基方法性能不足的问题。点基方法虽然训练成本低且算法简单，但通常不如表示基方法性能好。这个问题很重要，因为点云是3D数据的重要表示形式，在3D建模、重建、机器人等领域有广泛应用，提高点云生成质量可以促进3D内容创作、虚拟现实和增强现实等技术的发展。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先分析了点基方法的优缺点，然后借鉴了图像生成领域的多尺度生成方法和流匹配技术。他们将多尺度思想引入点云生成，但面临点云无序结构的挑战。为此，他们设计了特殊的下采样和上采样策略，使用K-means聚类和最远点采样(FPS)来保持几何结构一致性。作者还借鉴了流匹配的简单稳定训练方法，将其应用于点云生成的多尺度框架中。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是采用多尺度生成范式，从粗到细逐步生成点云，在每个尺度训练独立的流模型，并通过特殊的下采样和上采样操作保持几何结构一致性。整体流程包括：1)预处理阶段对原始点云进行多尺度下采样；2)训练阶段对每个尺度k训练一个流模型v_θ^k，使用流匹配目标函数；3)推理阶段从最粗尺度开始，逐步生成更精细的点云，通过上采样操作连接不同尺度，保持几何一致性。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)首次将多尺度生成引入点云流匹配领域；2)设计了几何感知的等大小K-means下采样策略，保持局部几何结构；3)提供了理论保证不同尺度间的分布对齐；4)保持了点基方法的简单性和效率，同时提高了性能。相比之前的工作，MFM-Point采用多尺度架构而非直接生成目标分辨率点云；与其他流匹配方法相比性能更高；与表示基方法相比保持了简单性和效率，同时达到相当或更好的性能。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'MFM-Point通过创新的多尺度流匹配框架和几何感知的下采样策略，显著提高了点基点云生成方法的性能和可扩展性，同时保持了算法的简单性和效率。'}


### 论文摘要

In recent years, point cloud generation has gained significant attention in 3D generative modeling. Among existing approaches, point-based methods directly generate point clouds without relying on other representations such as latent features, meshes, or voxels. These methods offer low training cost and algorithmic simplicity, but often underperform compared to representation-based approaches. In this paper, we propose MFM-Point, a multi-scale Flow Matching framework for point cloud generation that substantially improves the scalability and performance of point-based methods while preserving their simplicity and efficiency. Our multi-scale generation algorithm adopts a coarse-to-fine generation paradigm, enhancing generation quality and scalability without incurring additional training or inference overhead. A key challenge in developing such a multi-scale framework lies in preserving the geometric structure of unordered point clouds while ensuring smooth and consistent distributional transitions across resolutions. To address this, we introduce a structured downsampling and upsampling strategy that preserves geometry and maintains alignment between coarse and fine resolutions. Our experimental results demonstrate that MFM-Point achieves best-in-class performance among point-based methods and challenges the best representation-based methods. In particular, MFM-point demonstrates strong results in multi-category and high-resolution generation tasks.

---

## 79. Redefining Radar Segmentation: Simultaneous Static-Moving Segmentation and Ego-Motion Estimation using Radar Point Clouds

**论文链接:** [http://arxiv.org/abs/2511.20003v1](http://arxiv.org/abs/2511.20003v1)

**作者:** Simin Zhu, Satish Ravindran, Alexander Yarovoy, Francesco Fioranelli

**发布时间:** 2025-11-25

**备注:** 16 pages, 9 figures, under review at IEEE Transactions on Radar Systems

### GPT解析

### 总结

该研究提出了一种基于神经网络的解决方案，能够同时从雷达点云中分割静态和移动物体，并估计移动平台的瞬时二维速度，无需复杂的中间信号处理步骤。

### 背景

传统雷达分割研究通常专注于学习不同运动物体的类别标签，但雷达和光学传感器之间的基本差异导致预测准确一致的类别标签的可靠性存在差异。在汽车雷达感知任务中，确定物体是运动还是静止是大多数任务的前提条件。

### 目的

填补现有研究的空白，开发一种能够同时处理静态和移动物体分割的方法，并实现自运动估计。

### 方法

采用基于神经网络的解决方案，使用多层感知器(MLPs)和循环神经网络(RNNs)等简单而有效的构建块进行特征提取，直接从未处理的雷达点云中提取所需信息，无需点云聚合、多普勒补偿或运动补偿等中间步骤。

### 主要发现

研究引入了一套新的评估指标，并在具有挑战性的真实世界雷达数据集RadarScenes上进行了测试。结果表明，该方法不仅在双任务上表现良好，而且在其他雷达感知任务中也具有广泛的应用潜力。

### 结论

这是文献中首个能够同时分割静态和移动物体并估计自运动的雷达处理方法，证明了直接从未处理的点云中提取双任务所需信息的可行性。

### 翻译

传统雷达分割研究通常专注于学习不同移动物体的类别标签。尽管雷达和光学传感器之间的基本差异导致预测准确一致的类别标签的可靠性存在差异，但对汽车雷达感知任务的回顾表明，确定物体是运动还是静止是大多数任务的前提条件。为了填补这一空白，本研究提出了一种基于神经网络的解决方案，能够同时从雷达点云中分割静态和移动物体。此外，由于静态物体的测量径向速度与雷达的运动相关，该方法还可以估计移动平台或车辆（自运动）的瞬时二维速度。然而，尽管执行双重任务，所提出的方法使用非常简单但有效的构建块进行特征提取：多层感知器(MLPs)和循环神经网络(RNNs)。除了是文献中首个此类方法外，所提出的方法还证明了可以直接从未处理的点云中提取双任务所需信息的可行性，无需点云聚合、多普勒补偿、运动补偿或任何其他中间信号处理步骤。为了衡量其性能，本研究引入了一套新的评估指标，并使用具有挑战性的真实世界雷达数据集RadarScenes测试了所提出的方法。结果表明，所提出的方法不仅在双任务上表现良好，而且在其他雷达感知任务中也有广泛的应用潜力。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决传统雷达分割方法中只关注运动物体分类而忽视静态-动态物体区分的问题。这个问题在现实中很重要，因为准确区分静态和动态物体是大多数雷达感知任务（如自由空间检测、道路规划、多目标跟踪等）的前提，同时静态物体的径向速度可用于估计车辆自运动，这对自动驾驶系统的感知和决策至关重要。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者通过分析传统雷达分割方法的局限性（依赖外部传感器、需要点云聚合、使用复杂网络结构）来设计新方法。他们利用雷达点云中静态物体在多普勒轮廓中呈现独特正弦模式的特性，设计了简单但有效的神经网络架构。作者借鉴了PointNet进行空间特征提取、GRU进行时间特征提取，以及DeepEgo+处理车辆加速度影响的方法，但创新性地将它们组合成一个能同时完成静态-动态分割和自运动估计的双任务框架。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是利用雷达点云中静态物体在多普勒轮廓中的独特正弦模式特征，以及动态物体偏离这种模式的特性，通过简单神经网络同时提取时空特征，实现双重任务。整体流程包括：1)接收连续雷达点云作为输入；2)用MLP提取空间特征，GRU提取时间特征；3)通过双预测头（静态头和动态头）进行初步预测；4)使用加权最小二乘法估计雷达速度并更新静态权重；5)基于更新后的静态权重优化动态权重；6)输出静态-动态标签和自运动估计结果。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)首次实现同时静态-动态分割和自运动估计的双任务框架；2)完全基于雷达数据，无需外部传感器；3)无需点云聚合或运动补偿；4)使用轻量级网络（仅0.15M参数）；5)提出新的评估指标。相比之前的工作，本文不再依赖外部里程计传感器，不进行点云聚合，使用更简单的网络结构，且任务目标从传统的物体分类转向更基础的静态-动态区分，更适合雷达数据的特性。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '这篇论文提出了一种轻量级的神经网络方法，能够仅从原始雷达点云中同时实现静态-动态物体分割和车辆自运动估计，为自动驾驶雷达感知系统提供了一个高效、独立的解决方案。'}


### 论文摘要

Conventional radar segmentation research has typically focused on learning category labels for different moving objects. Although fundamental differences between radar and optical sensors lead to differences in the reliability of predicting accurate and consistent category labels, a review of common radar perception tasks in automotive reveals that determining whether an object is moving or static is a prerequisite for most tasks. To fill this gap, this study proposes a neural network based solution that can simultaneously segment static and moving objects from radar point clouds. Furthermore, since the measured radial velocity of static objects is correlated with the motion of the radar, this approach can also estimate the instantaneous 2D velocity of the moving platform or vehicle (ego motion). However, despite performing dual tasks, the proposed method employs very simple yet effective building blocks for feature extraction: multi layer perceptrons (MLPs) and recurrent neural networks (RNNs). In addition to being the first of its kind in the literature, the proposed method also demonstrates the feasibility of extracting the information required for the dual task directly from unprocessed point clouds, without the need for cloud aggregation, Doppler compensation, motion compensation, or any other intermediate signal processing steps. To measure its performance, this study introduces a set of novel evaluation metrics and tests the proposed method using a challenging real world radar dataset, RadarScenes. The results show that the proposed method not only performs well on the dual tasks, but also has broad application potential in other radar perception tasks.

---

## 80. A Storage-Efficient Feature for 3D Concrete Defect Segmentation to Replace Normal Vector

**论文链接:** [http://arxiv.org/abs/2511.19760v1](http://arxiv.org/abs/2511.19760v1)

**作者:** Linxin Hua, Jianghua Deng, Ye Lu

**发布时间:** 2025-11-24

**备注:** 25 pages, 7 figures

### GPT解析

### 总结

本研究提出了一种名为'相对角度'的新特征，用于点云重建中的损伤检测，有效减少了数据量同时保持了检测性能。

### 背景

基于图像的损伤检测方法容易受到背景噪声影响，而点云重建虽有效但受限于3D数据量大。

### 目的

开发一种能够减少数据量同时保持检测性能的新特征，用于混凝土表面缺陷检测。

### 方法

提出'相对角度'特征，定义为点的法向量与其父点云平均法向量之间的角度；使用基于熵的特征评估方法；通过PointNet++进行模型训练和测试。

### 主要发现

相对角度特征能有效过滤未损坏区域冗余信息，保留损坏区域有效信息；基于该特征的模型性能与基于法向量的模型相当，同时实现27.6%存储减少和83%输入通道压缩。

### 结论

相对角度特征是一种高效的数据压缩方法，能在资源受限硬件上实现更大批次执行，无需修改模型架构。

### 翻译

点云损伤重建为基于图像的方法提供了有效解决方案，但这些方法容易受到背景噪声的影响，其应用受到3D数据量大大的限制。本研究提出了一种新特征——相对角度，计算为一个点的法向量与其父点云平均法向量之间的角度。这个一维特征为混凝土表面缺陷特征提供了与法向量相当的方向性信息。通过基于熵的特征评估，本研究证明了相对角度能够过滤掉未损坏区域的冗余信息，同时保留损坏区域的有效信息。通过使用PointNet++进行训练和测试，基于相对角度的模型实现了与基于法向量模型相当的性能，同时实现了27.6%的存储减少和83%的输入通道压缩。这种新特征有可能在资源受限的硬件上实现更大批次的执行，而无需对模型架构进行修改。


### 论文摘要

Point cloud reconstruction of damage offers an effective solution to image-based methods vulnerable to background noise, yet its application is constrained by the high volume of 3D data. This study proposes a new feature, relative angle, computed as the angle between the normal vector of a point and the average normal vector of its parent point cloud. This single-dimensional feature provides directionality information equivalent to normal vectors for concrete surface defect characteristics. Through entropy-based feature evaluation, this study demonstrates the ability of relative angle to filter out redundant information in undamaged sections while retaining effective information in damaged sections. By training and testing with PointNet++, models based on the relative angles achieved similar performance to that of models based on normal vectors while delivering 27.6% storage reduction and 83% input channel compression. This novel feature has the potential to enable larger-batch execution on resource-constrained hardware without the necessity of architectural modifications to models.

---

## 81. 论文ID: 2511.19684v1

**错误:** 无法读取论文信息 - [Errno 2] No such file or directory: 'storage/result/2511.19684v1.json'

---

## 82. DensifyBeforehand: LiDAR-assisted Content-aware Densification for Efficient and Quality 3D Gaussian Splatting

**论文链接:** [http://arxiv.org/abs/2511.19294v1](http://arxiv.org/abs/2511.19294v1)

**作者:** Phurtivilai Patt, Leyang Huang, Yinqiang Zhang, Yang Lei

**发布时间:** 2025-11-24

### GPT解析

### 总结

本文提出了一种新的预先密集化方法，通过结合稀疏LiDAR数据和单目深度估计来增强3D场景初始化，解决了现有3D高斯散射方法的自适应密度控制导致的浮点伪影和资源使用效率低下问题。

### 背景

现有3D高斯散射(3DGS)方法依赖自适应密度控制，这可能导致浮点伪影和资源使用效率低下。

### 目的

提出一种新的预先密集化方法，通过结合稀疏LiDAR数据和对应RGB图像的单目深度估计来增强3D场景的初始化。

### 方法

采用ROI感知采样方案，优先考虑语义和几何上重要的区域，生成密集点云，提高视觉保真度和计算效率，绕过可能引入冗余高斯的自适应密度控制。

### 主要发现

预先密集化方法使优化能够专注于3D高斯原语的其他属性，减少重叠同时提高视觉质量，在显著降低资源消耗和训练时间的同时实现了与最先进技术相当的结果。

### 结论

该方法在四个新收集的数据集上通过广泛的比较和消融研究得到验证，证明了其在保留复杂场景中感兴趣区域的有效性。

### 翻译

本文解决了现有3D高斯散射(3DGS)方法的局限性，特别是它们对自适应密度控制的依赖，这可能导致浮点伪影和资源使用效率低下。我们提出了一种新的预先密集化方法，通过结合稀疏LiDAR数据和对应RGB图像的单目深度估计来增强3D场景的初始化。我们的ROI感知采样方案优先考虑语义和几何上重要的区域，生成密集点云，提高视觉保真度和计算效率。这种预先密集化方法绕过了原始流程中可能引入冗余高斯的自适应密度控制，使优化能够专注于3D高斯原语的其他属性，减少重叠同时提高视觉质量。我们的方法在显著降低资源消耗和训练时间的同时，实现了与最先进技术相当的结果。我们在四个新收集的数据集上通过广泛的比较和消融研究验证了我们的方法，展示了其在保留复杂场景中感兴趣区域的有效性。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决3D高斯溅射(3DGS)方法中自适应密度控制带来的问题，包括浮动伪影和资源使用效率低下的问题。这个问题很重要，因为3DGS是一种新兴的3D场景表示技术，广泛应用于计算机视觉和机器人领域，但现有的自适应密度控制过程会产生大量冗余的高斯原语，导致训练时间长、资源消耗大，并可能影响最终渲染质量。随着小型LiDAR传感器在移动设备上的普及，如何高效利用这些稀疏数据进行3D场景建模变得尤为重要。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': "作者首先分析了现有3DGS方法的局限性，特别是自适应密度控制带来的问题。他们注意到小型LiDAR传感器提供的数据稀疏，但单目深度估计技术可以提供密集深度信息。基于这些观察，作者设计了一种'预先密集化'的方法，结合LiDAR数据和单目深度估计来增强3D场景初始化。该方法借鉴了现有的单目深度估计技术(如Metric3D-V2)、隐藏点去除技术，以及资源感知方法(如Taming 3DGS和LightGaussian)的理念，但避免了这些方法在优化过程中使用的复杂评分函数和参数搜索。", '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是在优化之前预先密集化场景，而不是像传统3DGS那样在优化过程中动态增加高斯原语。通过结合小型LiDAR提供的稀疏点云和单目深度估计来创建初始密集点云，并使用ROI感知的采样策略将计算资源集中在重要区域。整体实现流程包括：1)输入带姿态的RGB图像和LiDAR点云；2)使用单目深度估计生成深度图；3)通过全局和局部缩放操作优化深度估计；4)基于颜色变化计算像素重要性，并将其投影回3D空间；5)根据重要性进行ROI感知采样；6)使用采样得到的密集点云初始化并训练3D高斯模型，跳过原始的克隆操作。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': "关键创新点包括：1)'预先密集化'方法，跳过传统3DGS的自适应密度控制；2)LiDAR辅助的单目深度估计优化，结合稀疏LiDAR数据和单目深度估计；3)ROI感知的采样策略，优先处理重要区域；4)简化的优化流程，使用固定数量的高斯原语。相比之前的工作，该方法与原始3DGS相比避免了浮动伪影和资源消耗；与Pixel-GS相比简化了密度控制；与Taming 3DGS相比在优化前进行空间评分；与LightGaussian相比不需要训练后的压缩步骤，可直接通过简单修剪获得紧凑表示。", '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'DensifyBeforehand通过结合LiDAR数据和单目深度估计进行预先密集化，实现了比现有方法更高效且视觉质量相当的3D高斯溅射场景表示，显著减少了训练时间和资源消耗。'}


### 论文摘要

This paper addresses the limitations of existing 3D Gaussian Splatting (3DGS) methods, particularly their reliance on adaptive density control, which can lead to floating artifacts and inefficient resource usage. We propose a novel densify beforehand approach that enhances the initialization of 3D scenes by combining sparse LiDAR data with monocular depth estimation from corresponding RGB images. Our ROI-aware sampling scheme prioritizes semantically and geometrically important regions, yielding a dense point cloud that improves visual fidelity and computational efficiency. This densify beforehand approach bypasses the adaptive density control that may introduce redundant Gaussians in the original pipeline, allowing the optimization to focus on the other attributes of 3D Gaussian primitives, reducing overlap while enhancing visual quality. Our method achieves comparable results to state-of-the-art techniques while significantly lowering resource consumption and training time. We validate our approach through extensive comparisons and ablation studies on four newly collected datasets, showcasing its effectiveness in preserving regions of interest in complex scenes.

---

## 83. Edge-Based Predictive Data Reduction for Smart Agriculture: A Lightweight Approach to Efficient IoT Communication

**论文链接:** [http://arxiv.org/abs/2511.19103v1](http://arxiv.org/abs/2511.19103v1)

**作者:** Dora Krekovic, Mario Kusek, Ivana Podnar Zarko, Danh Le-Phuoc

**发布时间:** 2025-11-24

**备注:** Accepted for presentation and publication in the proceedings of the IEEE Annual Congress on Artificial Intelligence of Things (IEEE AIoT 2025)

### GPT解析

### 总结

论文提出了一种边缘计算环境下的分析预测算法，通过预测传感器数据并仅在偏差超过预设值时传输，减少通信开销和能耗，特别适用于资源受限的物联网环境。

### 背景

物联网设备快速增长产生大量传感器数据，导致网络拥塞、延迟增加和高能耗。在资源受限和远程环境中，带宽限制和电池依赖使问题更加突出。农业等领域中连续传感器数据变化小，连续传输效率低下且消耗资源。

### 目的

提出适用于边缘计算环境的分析预测算法，优化传感器数据传输，减少冗余传输，解决网络拥塞、延迟和能耗问题。

### 方法

采用双重模型策略：1)网络边缘使用预测过滤器预测数据点，仅在偏差超过预设容差时触发传输；2)云端使用互补模型确保数据完整性和系统一致性；3)利用同一位置的现场和卫星观测数据增强模型鲁棒性；4)支持跨站点泛化，使模型可在不同区域部署无需重新训练。

### 主要发现

双重模型策略有效减少了通信开销；通过最小化冗余传输提高了能源效率；解决方案具有良好的可扩展性，适合优化远程和带宽受限的物联网环境中的传感器数据传输。

### 结论

提出的解决方案通过预测性传输和双重模型策略，有效解决了物联网传感器数据传输中的网络拥塞、延迟和能耗问题，特别适用于资源受限的远程环境，具有良好的可扩展性和能源效率。

### 翻译

物联网设备的快速增长导致了大量传感器数据需要传输到云服务器进行处理，造成过度网络拥塞、延迟增加和高能耗。这在资源受限和远程环境中尤其成问题，因为带宽有限，且依赖电池的设备进一步加剧了这一问题。此外，在农业等领域，连续的传感器读数通常变化很小，使连续数据传输效率低下且不必要地消耗资源。为了克服这些挑战，我们提出了一种专为边缘计算环境设计的分析预测算法，并通过模拟进行了验证。所提出的解决方案在网络边缘使用预测过滤器预测下一个传感器数据点，仅在预测值偏差超过预定义容差时触发数据传输。互补的云端模型确保数据完整性和整体系统一致性。这种双重模型策略有效减少了通信开销，并通过最小化冗余传输展示了提高能源效率的潜力。除了减少通信负载外，我们的方法还利用同一位置的现场和卫星观测数据来增强模型鲁棒性。它还支持跨站点泛化，使在一个地区训练的模型可以在其他地区有效部署而无需重新训练。这使得我们的解决方案具有高度可扩展性、能源感知性，并且非常适合优化远程和带宽受限的物联网环境中的传感器数据传输。


### 论文摘要

The rapid growth of IoT devices has led to an enormous amount of sensor data that requires transmission to cloud servers for processing, resulting in excessive network congestion, increased latency and high energy consumption. This is particularly problematic in resource-constrained and remote environments where bandwidth is limited, and battery-dependent devices further emphasize the problem. Moreover, in domains such as agriculture, consecutive sensor readings often have minimal variation, making continuous data transmission inefficient and unnecessarily resource intensive. To overcome these challenges, we propose an analytical prediction algorithm designed for edge computing environments and validated through simulation. The proposed solution utilizes a predictive filter at the network edge that forecasts the next sensor data point and triggers data transmission only when the deviation from the predicted value exceeds a predefined tolerance. A complementary cloud-based model ensures data integrity and overall system consistency. This dual-model strategy effectively reduces communication overhead and demonstrates potential for improving energy efficiency by minimizing redundant transmissions. In addition to reducing communication load, our approach leverages both in situ and satellite observations from the same locations to enhance model robustness. It also supports cross-site generalization, enabling models trained in one region to be effectively deployed elsewhere without retraining. This makes our solution highly scalable, energy-aware, and well-suited for optimizing sensor data transmission in remote and bandwidth-constrained IoT environments.

---

## 84. Structured Matching via Cost-Regularized Unbalanced Optimal Transport

**论文链接:** [http://arxiv.org/abs/2511.19075v1](http://arxiv.org/abs/2511.19075v1)

**作者:** Emanuele Pardini, Katerina Papagiannouli

**发布时间:** 2025-11-24

### GPT解析

### 总结

本研究提出了一种成本正则化非平衡最优传输(CR-UOT)框架，允许传输成本同时变化并支持质量的创造和移除，实现了欧几里得空间之间测度或点云的有效匹配。

### 背景

非平衡最优传输(UOT)虽为匹配非负有限Radon测度提供了灵活方法，但需要预定义地面传输成本，这可能无法准确表示数据的基础几何结构。当数据集存在于异构空间时，选择适当成本尤为困难，促使从业者采用Gromov-Wasserstein公式。

### 目的

解决在异构空间中难以选择适当传输成本的问题，开发一种允许地面成本变化同时允许质量创建和移除的新框架。

### 方法

通过参数化线性变换的内积成本家族，将非平衡Gromov-Wasserstein类型问题纳入CR-UOT框架中，并使用熵正则化开发了相应的算法。

### 主要发现

CR-UOT方法能够改进异构单细胞组学特征的配准效果，特别是在许多细胞缺乏直接匹配的情况下表现更佳。

### 结论

CR-UOT框架为异构空间中的测度匹配提供了更灵活的方法，通过允许传输成本变化和质量调整，解决了传统UOT方法在异构数据空间中的局限性。

### 翻译

非平衡最优传输(UOT)为匹配或比较非负有限Radon测度提供了灵活的方式。然而，UOT需要预定义的地面传输成本，这可能无法准确表示数据的基础几何结构。当数据集存在于异构空间中时，选择这样的成本尤其具有挑战性，这通常促使从业者采用Gromov-Wasserstein公式。为了应对这一挑战，我们引入了成本正则化非平衡最优传输(CR-UOT)框架，该框架允许地面成本变化，同时允许质量的创造和移除。我们证明CR-UOT通过参数化线性变换的内积成本家族，包含了非平衡Gromov-Wasserstein类型问题，从而能够实现欧几里得空间之间测度或点云的匹配。我们使用熵正则化开发了此类CR-UOT问题的算法，并证明这种方法可以改进异构单细胞组学特征的配准，特别是当许多细胞缺乏直接匹配时。


### 论文摘要

Unbalanced optimal transport (UOT) provides a flexible way to match or compare nonnegative finite Radon measures. However, UOT requires a predefined ground transport cost, which may misrepresent the data's underlying geometry. Choosing such a cost is particularly challenging when datasets live in heterogeneous spaces, often motivating practitioners to adopt Gromov-Wasserstein formulations. To address this challenge, we introduce cost-regularized unbalanced optimal transport (CR-UOT), a framework that allows the ground cost to vary while allowing mass creation and removal. We show that CR-UOT incorporates unbalanced Gromov-Wasserstein type problems through families of inner-product costs parameterized by linear transformations, enabling the matching of measures or point clouds across Euclidean spaces. We develop algorithms for such CR-UOT problems using entropic regularization and demonstrate that this approach improves the alignment of heterogeneous single-cell omics profiles, especially when many cells lack direct matches.

---

## 85. Diffusion Model-Enhanced Environment Reconstruction in ISAC

**论文链接:** [http://arxiv.org/abs/2511.19044v1](http://arxiv.org/abs/2511.19044v1)

**作者:** Nguyen Duc Minh Quang, Chang Liu, Shuangyang Li, Hoai-Nam Vu, Derrick Wing Kwan Ng, Wei Xiang

**发布时间:** 2025-11-24

**备注:** 6 pages, 5 figures, submitted to IEEE WCL

### GPT解析

### 总结

提出了一种噪声-稀疏性感知扩散模型(NSADM)后处理框架，用于解决ISAC系统中环境重建的初始结果粗糙问题

### 背景

在集成感知与通信(ISAC)系统中，环境重建(ER)是一种实现高分辨率环境感知的有前景的方法

### 目的

解决ISAC系统由于点云高稀疏性和显著噪声方差导致的初始结果粗糙且不令人满意的问题

### 方法

利用扩散模型强大的数据恢复能力，利用空间特征和噪声的加性性质来增强点云密度并对初始输入进行去噪

### 主要发现

模拟结果表明，在Chamfer距离和均方根误差方面，所提出的方法明显优于现有的基于模型和深度学习的方法

### 结论

NSADM框架能够有效提升ISAC系统中环境重建的质量

### 翻译

最近，在集成感知与通信(ISAC)系统中的环境重建(ER)已成为实现高分辨率环境感知的一种有前景的方法。然而，由于点云的高稀疏性和显著的噪声方差，从ISAC系统获得的初始结果通常是粗糙且不令人满意的。为了解决这个问题，我们提出了一个噪声-稀疏性感知扩散模型(NSADM)后处理框架。利用扩散模型强大的数据恢复能力，所提出的方案利用空间特征和噪声的加性性质来增强点云密度并对初始输入进行去噪。模拟结果表明，在Chamfer距离和均方根误差方面，所提出的方法明显优于现有的基于模型和深度学习的方法。


### 论文摘要

Recently, environment reconstruction (ER) in integrated sensing and communication (ISAC) systems has emerged as a promising approach for achieving high-resolution environmental perception. However, the initial results obtained from ISAC systems are coarse and often unsatisfactory due to the high sparsity of the point clouds and significant noise variance. To address this problem, we propose a noise-sparsity-aware diffusion model (NSADM) post-processing framework. Leveraging the powerful data recovery capabilities of diffusion models, the proposed scheme exploits spatial features and the additive nature of noise to enhance point cloud density and denoise the initial input. Simulation results demonstrate that the proposed method significantly outperforms existing model-based and deep learning-based approaches in terms of Chamfer distance and root mean square error.

---

## 86. Proxy-Free Gaussian Splats Deformation with Splat-Based Surface Estimation

**论文链接:** [http://arxiv.org/abs/2511.19542v1](http://arxiv.org/abs/2511.19542v1)

**作者:** Jaeyeong Kim, Seungwoo Yoo, Minhyuk Sung

**发布时间:** 2025-11-24

**备注:** 17 pages, Accepted to 3DV 2026 (IEEE/CVF International Conference on 3D Vision)

### GPT解析

### 总结

本文介绍了一种名为SpLap的新型无代理变形方法，用于高斯飞溅(GS)，基于从表面感知飞溅图计算的拉普拉斯算子，避免了传统方法对变形代理的依赖，同时解决了直接应用拉普拉斯变形技术时无法适当捕捉表面信息的问题。

### 背景

现有的高斯飞溅变形方法通常依赖于变形代理，如笼或网格，但这些方法存在对代理质量依赖和额外计算开销的问题。另一种直接将飞溅视为点云应用拉普拉斯变形的方法，往往因缺乏明确结构而无法适当捕捉表面信息。

### 目的

提出一种构建表面感知飞溅图的方法，使从中导出的拉普拉斯算子能够支持更合理的变形，保留细节和拓扑结构，并在变形后提高渲染质量。

### 方法

构建表面感知飞溅图，利用飞溅中编码的空间排列，不仅根据飞溅中心之间的距离，而且根据它们的交点来定义相邻飞溅。此外，引入高斯核自适应技术，在变形过程中保持表面结构，从而提高变形后的渲染质量。

### 主要发现

在ShapeNet、Objaverse、Sketchfab数据集的50个挑战性对象以及NeRF-Synthetic数据集上的实验中，与基于代理和无代理的基线方法相比，该方法展示了优越的性能。

### 结论

SpLap方法通过创新的表面感知飞溅图和高斯核自适应技术，成功解决了高斯飞溅变形中的关键问题，实现了无需代理的高质量变形，并在多个数据集上验证了其优越性。

### 翻译

我们介绍SpLap，一种基于从我们新型表面感知飞溅图计算的拉普拉斯算子的无代理高斯飞溅(GS)变形方法。现有的GS变形方法通常依赖于变形代理，如笼或网格，但它们受到代理质量依赖和额外计算开销的困扰。另一种方法是直接应用基于拉普拉斯的变形技术，将飞溅视为点云。然而，这往往因缺乏明确结构而无法适当捕捉表面信息。为解决这一问题，我们提出了一种新型方法，构建表面感知飞溅图，使从中导出的拉普拉斯算子能够支持更合理的变形，保留细节和拓扑结构。我们的关键思想是利用飞溅中编码的空间排列，不仅根据飞溅中心之间的距离，而且根据它们的交点来定义相邻飞溅。此外，我们引入了高斯核自适应技术，在变形过程中保持表面结构，从而提高变形后的渲染质量。在我们的实验中，与基于代理和无代理的基线方法相比，我们展示了该方法在ShapeNet、Objaverse和Sketchfab数据集的50个挑战性对象以及NeRF-Synthetic数据集上的优越性能。代码可在https://github.com/kjae0/SpLap获取。


### 论文摘要

We introduce SpLap, a proxy-free deformation method for Gaussian splats (GS) based on a Laplacian operator computed from our novel surface-aware splat graph. Existing approaches to GS deformation typically rely on deformation proxies such as cages or meshes, but they suffer from dependency on proxy quality and additional computational overhead. An alternative is to directly apply Laplacian-based deformation techniques by treating splats as point clouds. However, this often fail to properly capture surface information due to lack of explicit structure. To address this, we propose a novel method that constructs a surface-aware splat graph, enabling the Laplacian operator derived from it to support more plausible deformations that preserve details and topology. Our key idea is to leverage the spatial arrangement encoded in splats, defining neighboring splats not merely by the distance between their centers, but by their intersections. Furthermore, we introduce a Gaussian kernel adaptation technique that preserves surface structure under deformation, thereby improving rendering quality after deformation. In our experiments, we demonstrate the superior performance of our method compared to both proxy-based and proxy-free baselines, evaluated on 50 challenging objects from the ShapeNet, Objaverse, and Sketchfab datasets, as well as the NeRF-Synthetic dataset. Code is available at https://github.com/kjae0/SpLap.

---

## 87. 论文ID: 2511.18886v1

**错误:** 无法读取论文信息 - [Errno 2] No such file or directory: 'storage/result/2511.18886v1.json'

---

## 88. 论文ID: 2511.18801v1

**错误:** 无法读取论文信息 - [Errno 2] No such file or directory: 'storage/result/2511.18801v1.json'

---

## 89. Inverse Rendering for High-Genus Surface Meshes from Multi-View Images

**论文链接:** [http://arxiv.org/abs/2511.18680v1](http://arxiv.org/abs/2511.18680v1)

**作者:** Xiang Gao, Xinmu Wang, Xiaolong Wu, Jiazhi Li, Jingyu Shi, Yu Guo, Yuanpeng Liu, Xiyun Song, Heather Yu, Zongfang Lin, Xianfeng David Gu

**发布时间:** 2025-11-24

**备注:** 3DV2026 Accepted (Poster)

### GPT解析

### 总结

该研究提出了一种基于拓扑感知的逆向渲染方法，用于从多视角图像重建高亏格表面网格，解决了现有方法在处理高亏格表面时丢失拓扑特征和过度平滑低亏格表面的问题。

### 背景

基于网格的3D表示比体素和点云更受欢迎，因为它们能应用微分几何理论并针对现代图形管线优化。然而，现有逆向渲染方法在处理高亏格表面时往往失败严重，导致关键拓扑特征丢失，并倾向于过度平滑低亏格表面，造成表面细节丢失。这源于它们过度依赖基于Adam的优化器，可能导致梯度消失和爆炸问题。

### 目的

开发一种能够准确重建高亏格表面网格并保留表面细节的逆向渲染方法，解决现有方法的拓扑感知不足和优化问题。

### 方法

作者引入了一种自适应V循环重网格方案与重新参数化的Adam优化器相结合的方法。通过周期性地粗化和细化变形网格，该方法在优化前告知网格顶点其当前拓扑和几何信息，减轻梯度问题同时保留必要拓扑特征。此外，使用高斯-博内定理构建与真实值具有相同亏格数目的拓扑基元，强制执行拓扑一致性。

### 主要发现

实验结果表明，该逆向渲染方法优于当前最先进的方法，在Chamfer距离和体积IoU方面取得了显著改进，特别是在高亏格表面上，同时也增强了低亏格表面的表面细节。

### 结论

该研究提出的拓扑感知逆向渲染方法成功解决了现有方法在处理高亏格表面时的局限性，通过自适应重网格和重新参数化优化器增强了拓扑和几何感知能力，实现了更准确的表面重建。

### 翻译

我们提出了一种基于拓扑感知的逆向渲染方法，用于从多视角图像重建高亏格表面网格。与体素和点云等3D表示相比，基于网格的表示更受欢迎，因为它们 enables 应用微分几何理论，并针对现代图形管线进行了优化。然而，现有的逆向渲染方法在处理高亏格表面时往往失败严重，导致关键拓扑特征的丢失，并且倾向于过度平滑低亏格表面，导致表面细节丢失。这种失败源于它们过度依赖基于Adam的优化器，可能导致梯度和爆炸问题。为了克服这些挑战，我们引入了一种自适应V循环重网格方案，以及一个重新参数化的Adam优化器，以增强拓扑和几何感知能力。通过周期性地粗化和细化变形网格，我们的方法在优化前告知网格顶点其当前的拓扑和几何信息，从而减轻梯度问题，同时保留必要的拓扑特征。此外，我们使用高斯-博内定理构建与真实值具有相同亏格数目的拓扑基元，强制执行拓扑一致性。实验结果表明，该逆向渲染方法优于当前最先进的方法，在Chamfer距离和体积IoU方面取得了显著改进，特别是在高亏格表面上，同时也增强了低亏格表面的表面细节。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决从多视角图像重建高亏格（高复杂度、多孔洞）表面网格时的拓扑不一致性和几何细节丢失问题。这个问题在现实中非常重要，因为准确的3D表面重建在虚拟现实、医学成像、机器人、自动驾驶和3D打印等领域有广泛应用。网格表示因其能应用微分几何理论并适配现代图形管道而被优先选择，而确保拓扑一致性（如正确的孔洞数量）对保持物体关键特征至关重要，拓扑错误会导致重建结果视觉不准确，影响下游应用效果。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先分析了现有3D表示方法的优缺点，指出网格表示在物理模拟和图形渲染中的优势。他们发现现有方法在处理高亏格表面时失败的根本原因是过度依赖Adam优化器导致梯度问题。因此，作者设计了一种结合自适应V循环重网格化方案和重新参数化Adam优化器的方法。该方法借鉴了几何处理中的重网格化技术、微分几何理论（如高斯-博内定理）以及物理基础可微分渲染的最新进展，通过周期性粗化和细化网格来增强拓扑感知，同时使用半边数据结构实现高效的局部网格操作。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是通过拓扑感知的自适应V循环重网格化方案解决高亏格表面重建中的梯度问题，同时确保拓扑一致性。整体流程包括：1)初始化一个与真实值亏格数匹配的拓扑基元；2)执行自适应重网格化（边缘分割、折叠、翻转和切向平滑）；3)使用重新参数化的Adam优化器优化网格顶点位置以最小化渲染损失；4)定期重复重网格化和优化过程（每130-200次迭代）；5)达到最大迭代次数（低亏格1500次，高亏格3000次）后停止。这种方法通过周期性调整网格结构，使顶点在优化前了解当前拓扑和几何状态，从而保留关键特征并避免梯度问题。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)拓扑感知的逆向渲染框架，直接解决高亏格表面重建挑战；2)自适应V循环重网格化方案，通过周期性粗化和细化增强拓扑感知；3)重新参数化的Adam优化器，专为高亏格表面重建设计；4)基于高斯-博内定理的拓扑一致性强制执行机制。相比之前的工作，本文方法能准确重建高亏格表面而不会丢失关键拓扑特征，避免过度平滑低亏格表面导致的细节丢失，通过重网格化缓解了梯度消失/爆炸问题，确保重建表面的亏格与真实值匹配，并在Chamfer距离和体积IoU等指标上显著优于现有方法，特别是在高亏格表面重建方面。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '本文提出了一种结合自适应V循环重网格化与重新参数化Adam优化器的拓扑感知逆向渲染方法，能够从多视角图像准确重建高亏格表面网格，同时保留关键拓扑特征并增强表面细节。'}


### 论文摘要

We present a topology-informed inverse rendering approach for reconstructing high-genus surface meshes from multi-view images. Compared to 3D representations like voxels and point clouds, mesh-based representations are preferred as they enable the application of differential geometry theory and are optimized for modern graphics pipelines. However, existing inverse rendering methods often fail catastrophically on high-genus surfaces, leading to the loss of key topological features, and tend to oversmooth low-genus surfaces, resulting in the loss of surface details. This failure stems from their overreliance on Adam-based optimizers, which can lead to vanishing and exploding gradients. To overcome these challenges, we introduce an adaptive V-cycle remeshing scheme in conjunction with a re-parametrized Adam optimizer to enhance topological and geometric awareness. By periodically coarsening and refining the deforming mesh, our method informs mesh vertices of their current topology and geometry before optimization, mitigating gradient issues while preserving essential topological features. Additionally, we enforce topological consistency by constructing topological primitives with genus numbers that match those of ground truth using Gauss-Bonnet theorem. Experimental results demonstrate that our inverse rendering approach outperforms the current state-of-the-art method, achieving significant improvements in Chamfer Distance and Volume IoU, particularly for high-genus surfaces, while also enhancing surface details for low-genus surfaces.

---

## 90. 论文ID: 2511.18563v1

**错误:** 无法读取论文信息 - [Errno 2] No such file or directory: 'storage/result/2511.18563v1.json'

---

## 91. Matching correlated VAR time series

**论文链接:** [http://arxiv.org/abs/2511.18553v1](http://arxiv.org/abs/2511.18553v1)

**作者:** Ernesto Araya, Hemant Tyagi

**发布时间:** 2025-11-23

### GPT解析

### 总结

研究匹配相关的VAR时间序列数据库问题，通过概率框架建模两个时间序列之间的关系，目标是恢复隐藏的置换映射π*。

### 背景

该问题将匹配独立点云的经典问题推广到了时间序列设置，其中一个多元时间序列被观测到，同时还有一个被扰动和置换的版本。

### 目的

从观测到的两个时间序列中恢复未知的置换映射π*。

### 方法

推导了最大似然估计量(MLE)，导致在置换上的二次优化；理论分析了基于线性分配的估计量；通过考虑置换矩阵集合的凸松弛(如Birkhoff多面体)来求解MLE，并使用交替最小化来高效估计π*和VAR参数。

### 主要发现

对于基于线性分配的方法，建立了恢复保证，识别出允许完全或部分恢复的σ阈值；经验上，线性分配通常匹配或优于基于MLE松弛的方法。

### 结论

提出的框架和方法能够有效解决VAR时间序列的匹配问题，特别是在扰动水平σ低于特定阈值时可以实现精确恢复。

### 翻译

我们研究匹配相关的VAR时间序列数据库问题，其中观测到一个多元时间序列及其扰动和置换的版本，目标是恢复它们之间的未知匹配关系。为此，我们引入了一个概率框架，其中两个时间序列被共同生成，满足特定的关系式，其中两个时间序列是独立同分布的一阶向量自回归时间序列，具有高斯增量，对应一个隐藏的置换π*。目标是从观测到的两个时间序列中恢复π*。这将匹配独立点云的经典问题推广到了时间序列设置。我们推导了最大似然估计量(MLE)，导致在置换上的二次优化，并理论分析了基于线性分配的估计量。对于后一种方法，我们建立了恢复保证，识别出允许完全或部分恢复的σ阈值。此外，我们通过考虑置换矩阵集合的凸松弛来提出求解MLE的方法。这允许通过交替最小化来高效估计π*和VAR参数。经验上，我们发现线性分配通常匹配或优于基于MLE松弛的方法。


### 论文摘要

We study the problem of matching correlated VAR time series databases, where a multivariate time series is observed along with a perturbed and permuted version, and the goal is to recover the unknown matching between them. To model this, we introduce a probabilistic framework in which two time series $(x_t)_{t\in[T]},(x^\#_t)_{t\in[T]}$ are jointly generated, such that $x^\#_t=x_{π^*(t)}+σ\tilde{x}_{π^*(t)}$, where $(x_t)_{t\in[T]},(\tilde{x}_t)_{t\in[T]}$ are independent and identically distributed vector autoregressive (VAR) time series of order $1$ with Gaussian increments, for a hidden $π^*$. The objective is to recover $π^*$, from the observation of $(x_t)_{t\in[T]},(x^\#_t)_{t\in[T]}$. This generalizes the classical problem of matching independent point clouds to the time series setting.   We derive the maximum likelihood estimator (MLE), leading to a quadratic optimization over permutations, and theoretically analyze an estimator based on linear assignment. For the latter approach, we establish recovery guarantees, identifying thresholds for $σ$ that allow for perfect or partial recovery. Additionally, we propose solving the MLE by considering convex relaxations of the set of permutation matrices (e.g., over the Birkhoff polytope). This allows for efficient estimation of $π^*$ and the VAR parameters via alternating minimization. Empirically, we find that linear assignment often matches or outperforms MLE relaxation based approaches.

---

## 92. Splatblox: Traversability-Aware Gaussian Splatting for Outdoor Robot Navigation

**论文链接:** [http://arxiv.org/abs/2511.18525v1](http://arxiv.org/abs/2511.18525v1)

**作者:** Samarth Chopra, Jing Liang, Gershom Seneviratne, Yonghan Lee, Jaehoon Choi, Jianyu An, Stephen Cheng, Dinesh Manocha

**发布时间:** 2025-11-23

**备注:** Submitted to ICRA 2026

### GPT解析

### 总结

Splatblox是一个用于在户外密集植被、不规则障碍物和复杂地形环境中自主导航的实时系统。

### 背景

户外环境中的自主导航面临密集植被、不规则障碍物和复杂地形的挑战，现有方法在这些场景中表现有限。

### 目的

开发一个能够有效处理户外复杂环境、区分可通行区域和障碍物的自主导航系统。

### 方法

融合分割的RGB图像和LiDAR点云，使用高斯斑点技术构建可通行感知的欧几里得符号距离场，该场同时编码几何和语义信息，并通过在线更新支持语义推理和360度几何覆盖。

### 主要发现

在四足机器人和轮式平台上的实验表明，Splatblox在植被丰富场景中比最先进方法成功率提高50%以上，冻结事件减少40%，路径缩短5%，到达目标时间最多快13%，同时支持长达100米的长距离任务。

### 结论

Splatblox通过融合视觉和LiDAR数据，结合语义理解和几何表示，显著提升了复杂户外环境中的自主导航性能。

### 翻译

我们提出了Splatblox，一个用于在户外密集植被、不规则障碍物和复杂地形环境中自主导航的实时系统。我们的方法使用高斯斑点融合分割的RGB图像和LiDAR点云，构建可通行感知的欧几里得符号距离场，该场同时编码几何和语义信息。通过在线更新，该场支持语义推理以区分可通行植被（如高草）和刚性障碍物（如树木），同时LiDAR确保360度几何覆盖以扩展规划范围。我们在四足机器人上验证了Splatblox，并展示了其转移到轮式平台的能力。在植被丰富的场景实地试验中，它比最先进方法表现更好，成功率提高50%以上，冻结事件减少40%，路径缩短5%，到达目标时间最多快13%，同时支持长达100米的长距离任务。实验视频和更多详情可在我们的项目页面找到：https://splatblox.github.io

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决机器人在户外复杂环境（特别是有密集植被、不规则障碍物和复杂地形）中的自主导航问题，尤其是让机器人能够区分可穿越的植被（如高草）和不可穿越的刚性障碍物（如树木）。这个问题在现实和研究中的重要性体现在：户外导航有广泛应用（精准农业、森林探索、搜救行动）；户外环境中的障碍物多样，使得稳健感知困难；需要长距离自主技术；区分可穿越/不可穿越地形是机器人户外运行的基本能力；现有方法要么需要大量标注数据，要么缺乏语义推理，或计算资源需求过高。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先分析了现有方法的局限性：学习基方法需要大量标注数据且环境特定；自监督方法难以迁移；离线强化学习需要大量轨迹数据；大型神经网络难以在边缘设备部署；无学习方法缺乏语义推理；传统场景表示缺乏语义细节；隐式神经表示计算效率低。作者选择3D高斯溅射作为基础，因为它比NeRF收敛更快、重建保真度更高。作者借鉴了CLIPSeg进行可穿越性估计、多模态融合RGB和LiDAR、使用ESDF进行路径规划、与Nvblox系统融合提供360°几何覆盖等现有工作。创新在于将高斯溅射扩展到超越 photorealistic 渲染，设计轻量级管道，以及融合GSPLAT和LiDAR的ESDF策略。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是使用高斯溅射技术创建融合RGB语义信息和LiDAR几何信息的3D场景表示，通过可穿越性感知的欧几里得带符号距离场（ESDF）编码几何和语义信息，在机器人前方区域使用语义丰富的ESDF，在其他区域使用LiDAR提供的360°几何覆盖。整体流程包括：1) 使用CLIPSeg从RGB图像估计可穿越性，将地形分为四类并分配成本值；2) 用3D高斯基元表示场景，将高斯投影到图像平面融合RGB语义和LiDAR几何，将LiDAR点投影到相机帧并分配可穿越性成本；3) 从可穿越性成本体积转换为3D距离场，将GSPLAT衍生的ESDF与LiDAR ESDF融合；4) 限制活动高斯数量在100k以内，在资源受限GPU上实现实时性能。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点：1) 扩展高斯溅射功能，嵌入LiDAR几何和语义可穿越性成本；2) 设计针对边缘GPU的轻量级管道；3) 融合GSPLAT和LiDAR的ESDF，支持长达100米的长距离规划。相比之前工作的不同：1) 与学习基方法相比，不需要离线训练，易于迁移，计算效率更高；2) 与传统表示方法相比，能更紧凑地编码语义信息，内存使用随场景复杂度扩展；3) 与隐式神经表示相比，训练和推理速度快，支持在线更新；4) 与其他高斯溅射机器人应用相比，是首个实时户外高斯溅射导航系统，支持长距离规划。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'Splatblox通过创新性地融合RGB语义信息和LiDAR几何信息到实时更新的高斯溅射表示中，解决了机器人在复杂户外环境中的可穿越性感知和长距离导航挑战，显著提高了成功率和路径效率，同时保持了边缘设备的实时性能。'}


### 论文摘要

We present Splatblox, a real-time system for autonomous navigation in outdoor environments with dense vegetation, irregular obstacles, and complex terrain. Our method fuses segmented RGB images and LiDAR point clouds using Gaussian Splatting to construct a traversability-aware Euclidean Signed Distance Field (ESDF) that jointly encodes geometry and semantics. Updated online, this field enables semantic reasoning to distinguish traversable vegetation (e.g., tall grass) from rigid obstacles (e.g., trees), while LiDAR ensures 360-degree geometric coverage for extended planning horizons. We validate Splatblox on a quadruped robot and demonstrate transfer to a wheeled platform. In field trials across vegetation-rich scenarios, it outperforms state-of-the-art methods with over 50% higher success rate, 40% fewer freezing incidents, 5% shorter paths, and up to 13% faster time to goal, while supporting long-range missions up to 100 meters. Experiment videos and more details can be found on our project page: https://splatblox.github.io

---

## 93. CrossJEPA: Cross-Modal Joint-Embedding Predictive Architecture for Efficient 3D Representation Learning from 2D Images

**论文链接:** [http://arxiv.org/abs/2511.18424v1](http://arxiv.org/abs/2511.18424v1)

**作者:** Avishka Perera, Kumal Hewagamage, Saeedha Nazar, Kavishka Abeywardana, Hasitha Gallella, Ranga Rodrigo, Mohamed Afham

**发布时间:** 2025-11-23

**备注:** 24 pages, 10 figures

### GPT解析

### 总结

本文提出了CrossJEPA，一种图像到点的跨模态联合嵌入预测架构，利用图像基础模型知识训练预测器推断3D点云的2D视图嵌入，在3D表示学习任务中实现了高性能、低内存占用和快速训练。

### 背景

3D表示学习面临大规模3D数据集稀缺问题，现有利用2D数据的模型通常计算成本高、难以部署，且模型架构设计对性能、内存和计算效率至关重要。

### 目的

设计一种高效、轻量的跨模态学习方法，解决3D表示学习中的数据稀缺和计算效率问题。

### 方法

提出CrossJEPA架构，利用图像基础模型知识，训练预测器推断3D点云对应2D视图的嵌入；通过条件化预测器纯化监督信号；采用冻结教师设计和目标嵌入缓存机制提高效率。

### 主要发现

CrossJEPA在ModelNet40(94.2%)和ScanObjectNN(88.3%)基准测试上取得最先进结果，仅使用1410万参数，单GPU约6小时完成预训练，证明了其性能、内存效率和训练速度。

### 结论

CrossJEPA是一种高性能、内存高效且训练快速的3D表示学习框架，通过知识蒸馏有效解决了3D数据稀缺问题。

### 翻译

图像到点的跨模态学习已出现以解决3D表示学习中大规模3D数据集稀缺的问题。然而，当前利用2D数据的方法通常导致模型大、训练慢，使它们计算成本高且难以在资源受限环境中部署。因此，此类模型的架构设计至关重要，决定了它们的性能、内存占用和计算效率。联合嵌入预测架构(JEPA)因其简单性和效率在自监督学习中广受欢迎，但在跨模态环境中探索不足，部分原因是人们误以为掩码是JEPA固有的。基于此，我们提出了CrossJEPA，一种简单的跨模态联合嵌入预测架构，利用图像基础模型的知识，训练预测器推断对应3D点云的特定渲染2D视图的嵌入，从而引入了超越掩码的JEPA风格预训练策略。通过在跨域投影信息上条件化预测器，CrossJEPA纯化了目标域特有的监督信号。我们进一步利用冻结教师设计和一次性目标嵌入缓存机制，实现摊销效率。CrossJEPA在合成的ModelNet40(94.2%)和真实世界的ScanObjectNN(88.3%)基准测试的线性探测中取得了新的最先进水平，仅使用1410万预训练参数(点编码器中850万)，在标准单GPU上约6小时完成预训练。这些结果使CrossJEPA成为通过知识蒸馏进行3D表示学习的性能强大、内存高效且训练快速的框架。我们对CrossJEPA进行了直观、理论和经验分析，并广泛消融了我们的设计选择。代码将公开可用。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决3D表示学习中大规模3D数据集稀缺的问题，以及现有利用2D数据的方法模型大、训练慢、计算成本高的问题。这个问题在现实中很重要，因为3D表示学习对自动驾驶、机器人、混合现实和医疗成像等应用至关重要，而这些应用通常需要在资源受限的边缘设备上部署，因此需要高效、轻量的3D理解方法。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者注意到JEPA在自监督学习中因其简单性和效率而受欢迎，但在跨模态设置中探索不足，并纠正了一个误解：认为掩码是JEPA的本质，但实际上并非如此。作者受信息理论和大脑预测编码启发，提出结合冻结的图像编码器作为教师和点编码器作为学生的架构，并设计了关键预测器组件来解耦教师特定的2D干扰因素。作者借鉴了JEPA的基本框架、预训练图像模型（如DinoV2）、Transformer架构以及预测编码原理。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是通过预测特定渲染2D视图的嵌入来学习3D点云表示，使用冻结的预训练图像编码器作为教师提供目标表示，并用轻量级预测器将点云表示映射到特定视图的图像表示。整体流程包括：1)从ShapeNet和Objaverse-XL获取点云和渲染图像；2)为每个3D对象生成36个不同视角的2D渲染；3)使用冻结图像编码器提取图像特征，用可学习点编码器提取点云特征；4)训练预测器根据点云特征和视图参数预测特定视图的图像特征；5)使用平滑L1损失最小化预测与真实特征差异；6)采用预计算的目标嵌入缓存机制提高效率。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)提出首个面向图像到点云跨模态学习的JEPA风格框架；2)引入不依赖显式掩码的新JEPA变体，基于潜在视图预测任务；3)实现高性能、快速训练、参数高效的预训练流程；4)使用冻结教师设计和目标嵌入缓存机制实现摊销效率；5)通过提供显式条件信息净化监督信号。相比之前工作，CrossJEPA避免了生成架构的掩码和重建需求，无需联合嵌入架构的对比学习，参数量更少(14.1M)，训练时间更短(约6小时)，但性能更优，并引入了独特的条件信息机制。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'CrossJEPA提出了一种高效的跨模态联合嵌入预测架构，通过创新的预测机制和知识蒸馏方法，实现了从2D图像到3D点云的高效知识转移，在大幅减少计算资源需求的同时取得了最先进的性能。'}


### 论文摘要

Image-to-point cross-modal learning has emerged to address the scarcity of large-scale 3D datasets in 3D representation learning. However, current methods that leverage 2D data often result in large, slow-to-train models, making them computationally expensive and difficult to deploy in resource-constrained environments. The architecture design of such models is therefore critical, determining their performance, memory footprint, and compute efficiency. The Joint-embedding Predictive Architecture (JEPA) has gained wide popularity in self-supervised learning for its simplicity and efficiency, but has been under-explored in cross-modal settings, partly due to the misconception that masking is intrinsic to JEPA. In this light, we propose CrossJEPA, a simple Cross-modal Joint Embedding Predictive Architecture that harnesses the knowledge of an image foundation model and trains a predictor to infer embeddings of specific rendered 2D views from corresponding 3D point clouds, thereby introducing a JEPA-style pretraining strategy beyond masking. By conditioning the predictor on cross-domain projection information, CrossJEPA purifies the supervision signal from semantics exclusive to the target domain. We further exploit the frozen teacher design with a one-time target embedding caching mechanism, yielding amortized efficiency. CrossJEPA achieves a new state-of-the-art in linear probing on the synthetic ModelNet40 (94.2%) and the real-world ScanObjectNN (88.3%) benchmarks, using only 14.1M pretraining parameters (8.5M in the point encoder), and about 6 pretraining hours on a standard single GPU. These results position CrossJEPA as a performant, memory-efficient, and fast-to-train framework for 3D representation learning via knowledge distillation. We analyze CrossJEPA intuitively, theoretically, and empirically, and extensively ablate our design choices. Code will be made available.

---

## 94. UniFlow: Towards Zero-Shot LiDAR Scene Flow for Autonomous Vehicles via Cross-Domain Generalization

**论文链接:** [http://arxiv.org/abs/2511.18254v1](http://arxiv.org/abs/2511.18254v1)

**作者:** Siyi Li, Qingwen Zhang, Ishan Khatri, Kyle Vedder, Deva Ramanan, Neehar Peri

**发布时间:** 2025-11-23

**备注:** Project Page: https://lisiyi777.github.io/UniFlow/

### GPT解析

### 总结

本文提出了一种名为UniFlow的跨数据集训练方法，用于LiDAR场景流估计，在多个数据集上实现了最先进的性能，证明了低级任务（如运动估计）可以从跨数据集训练中获益，而传统认知认为这会导致性能下降。

### 背景

LiDAR场景流是估计连续点云中每点3D运动的任务。最近的方法在自动驾驶数据集上达到厘米级精度，但通常只在单一传感器上训练和评估，限制了模型在未见过的传感器上的泛化能力。

### 目的

旨在学习通用的运动先验，使其能够迁移到多样且未见过的LiDAR传感器上，提高模型在真实世界不同场景下的适用性。

### 方法

提出UniFlow，一种前馈模型家族，统一并在多个大规模LiDAR场景流数据集上训练，这些数据集具有不同的传感器布置和点云密度，采用'令人沮丧的简单'的跨数据集训练策略。

### 主要发现

传统认知（在多个数据集上训练会导致性能下降）不适用于运动估计任务；最先进的场景流方法从跨数据集训练中获益匪浅；低级任务（如运动估计）可能对传感器配置不太敏感；在快速移动物体上训练的模型在不同数据集的快速移动物体上表现良好。

### 结论

UniFlow在Waymo和nuScenes上建立了新的最先进水平，分别比先前的工作提高了5.1%和35.2%。此外，UniFlow在未见过的数据集（如TruckScenes）上也实现了最先进的准确性，比专门的TruckScenes模型高出30.1%，证明了跨数据集训练对于LiDAR场景流估计的有效性。

### 翻译

LiDAR场景流是估计连续点云中每点3D运动的任务。最近的方法在流行的自动驾驶数据集上达到厘米级精度，但通常只在单一传感器上训练和评估。在本文中，我们旨在学习能够迁移到多样且未见过的LiDAR传感器的通用运动先验。然而，LiDAR语义分割和3D目标检测的先前研究表明，在多个数据集上简单训练会导致性能低于单一数据集模型。有趣的是，我们发现这种传统认知不适用于运动估计，最先进的场景流方法从跨数据集训练中获益匪浅。我们认为，像运动估计这样的低级任务可能对传感器配置不太敏感；确实，我们的分析显示，在快速移动物体（如高速公路数据集）上训练的模型在不同数据集的快速移动物体上表现良好。基于我们的分析，我们提出了UniFlow，一个前馈模型家族，统一并在多个具有不同传感器布置和点云密度的大规模LiDAR场景流数据集上进行训练。我们令人沮丧的简单解决方案在Waymo和nuScenes上建立了新的最先进水平，分别比先前的工作提高了5.1%和35.2%。此外，UniFlow在未见过的数据集（如TruckScenes）上实现了最先进的准确性，比专门的TruckScenes模型高出30.1%。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决LiDAR场景流估计中的跨域泛化问题，即如何让一个模型在不同传感器配置和环境条件下都能准确估计点云的运动。这个问题对自动驾驶至关重要，因为实际应用中车辆会遇到各种不同的传感器配置和环境条件，而当前的方法通常只在特定传感器上训练，更换传感器时性能会显著下降。解决这一问题可以降低自动驾驶系统对不同传感器的依赖，提高鲁棒性，并减少更换传感器时重新训练模型的高昂成本。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先观察到传统观点认为在多个数据集上训练会导致性能下降，但这一观点在场景流估计任务中并不成立。通过实验发现，场景流作为低级视觉任务对传感器配置不太敏感，且运动预测准确性与物体速度高度相关。作者借鉴了RGB低级视觉任务中跨域泛化的思想，如FlowNet和RAFT等光学流模型，以及ZeroMSF等零样本单目场景流估计器。基于这些发现，作者设计了UniFlow方法，通过在多个大型LiDAR数据集上进行统一训练，学习通用的运动先验。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是场景流估计作为低级视觉任务，对传感器配置的敏感性低于高级任务，通过在多样化数据集上训练可以学习到通用的运动先验。实现流程包括：1) 数据集统一：标准化不同数据集的传感器帧率和注释频率，确保时间间隔一致；2) 统一多数据集训练：在nuScenes、AV2和Waymo的统一数据集混合上重新训练现有场景流架构；3) 使用简单但有效的增强技术，包括高度随机化和激光射线丢弃；4) 在多个数据集上评估模型性能，特别关注跨域泛化能力。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': "关键创新点包括：1) 挑战了'多数据集训练会导致性能下降'的传统观点；2) 发现速度分布差异是跨域泛化的关键因素；3) 提出了简单有效的UniFlow统一训练框架；4) 在多个数据集上进行了广泛的实验验证。相比之前的工作，UniFlow在多个数据集上统一训练而非单数据集训练；特别关注速度分布而非仅几何域差异；在已知和未见数据集上实现了显著的性能提升；采用了简单直接的方法，不需要复杂的架构修改或领域适应技术。", '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'UniFlow通过在多样化的大型LiDAR场景流数据集上进行统一训练，实现了跨传感器配置的零样本场景流估计，显著提高了在已知和未见数据集上的性能，挑战了LiDAR感知任务中多数据集训练效果不佳的传统观点。'}


### 论文摘要

LiDAR scene flow is the task of estimating per-point 3D motion between consecutive point clouds. Recent methods achieve centimeter-level accuracy on popular autonomous vehicle (AV) datasets, but are typically only trained and evaluated on a single sensor. In this paper, we aim to learn general motion priors that transfer to diverse and unseen LiDAR sensors. However, prior work in LiDAR semantic segmentation and 3D object detection demonstrate that naively training on multiple datasets yields worse performance than single dataset models. Interestingly, we find that this conventional wisdom does not hold for motion estimation, and that state-of-the-art scene flow methods greatly benefit from cross-dataset training. We posit that low-level tasks such as motion estimation may be less sensitive to sensor configuration; indeed, our analysis shows that models trained on fast-moving objects (e.g., from highway datasets) perform well on fast-moving objects, even across different datasets. Informed by our analysis, we propose UniFlow, a family of feedforward models that unifies and trains on multiple large-scale LiDAR scene flow datasets with diverse sensor placements and point cloud densities. Our frustratingly simple solution establishes a new state-of-the-art on Waymo and nuScenes, improving over prior work by 5.1% and 35.2% respectively. Moreover, UniFlow achieves state-of-the-art accuracy on unseen datasets like TruckScenes, outperforming prior TruckScenes-specific models by 30.1%.

---

## 95. Two-step Generalized RBF-Generated Finite Difference Method on Manifolds

**论文链接:** [http://arxiv.org/abs/2511.18049v1](http://arxiv.org/abs/2511.18049v1)

**作者:** Rongji Li, Haichuan Di, Shixiao Willing Jiang

**发布时间:** 2025-11-22

### GPT解析

### 总结

该研究提出了一种两步广义径向基函数生成的有限差分(gRBF-FD)方法，用于解决由随机采样点云数据定义的无边界流形上的偏微分方程问题。该方法结合了广义移动最小二乘法和多调和样条插值，通过特定权重函数和模板大小自动调整策略提高稳定性和精度。

### 背景

在科学计算中，解决由随机采样点云定义的流形上的偏微分方程是一个具有挑战性的问题，在多个领域有广泛应用。传统方法在处理此类问题时存在稳定性不足和精度有限的问题。

### 目的

开发一种高效稳定的数值方法，用于解决在由随机采样点云数据标识的无边界流形上的偏微分方程问题，提高计算精度和稳定性。

### 方法

提出了一种两步gRBF-FD方法，基于多调和样条核和多变量多项式在局部蒙日坐标系的切空间中定义。第一步使用广义移动最小二乘法(GMLS)回归局部目标函数，第二步使用PHS插值补偿残差。方法中采用了特定权重函数和模板大小K的自动调整策略，以产生具有特定系数结构的拉普拉斯矩阵。

### 主要发现

研究建立了算子近似的误差界限，包括局部模板直径和数据数量的关系。数值测试表明，该方法在各种光滑流形上具有高精度。通过特定权重函数和自动调整策略，显著提高了方法的稳定性和减少了解误差。

### 结论

所提出的gRBF-FD方法为解决由随机采样点云定义的流形上的偏微分方程提供了一种高效、稳定的数值解决方案，具有广泛的应用前景。

### 翻译

在由随机采样点云定义的流形上求解偏微分方程是科学计算中的一个挑战性问题，在各个领域有广泛的应用。在本文中，我们开发了一种两步广义径向基函数生成的有限差分(gRBF-FD)方法，用于解决由随机采样点云数据标识的无边界流形上的偏微分方程。gRBF-FD基于在局部蒙日坐标系的切空间中定义的多调和样条核和多变量多项式(PHS+Poly)。第一步是使用广义移动最小二乘法(GMLS)回归局部目标函数，第二步是使用PHS插值补偿残差。我们的gRBF-FD方法与标准RBF-FD具有相同的插值形式，但插值系数不同。我们的方法在GMLS和PHS步骤中都使用了特定的权重函数，并实现了每个点模板大小K(即最近邻数量)的自动调整策略。这些策略旨在产生具有特定系数结构的拉普拉斯矩阵，从而提高稳定性并减少解误差。我们建立了算子近似的误差界限，包括局部模板直径和数据数量的关系。我们进一步通过在各种光滑流形上的数值测试，证明了gRBF-FD的高精度。


### 论文摘要

Solving partial differential equations (PDEs) on manifolds defined by randomly sampled point clouds is a challenging problem in scientific computing and has broad applications in various fields. In this paper, we develop a two-step generalized radial basis function-generated finite difference (gRBF-FD) method for solving PDEs on manifolds without boundaries, identified by randomly sampled point cloud data. The gRBF-FD is based on polyharmonic spline kernels and multivariate polynomials (PHS+Poly) defined over the tangent space in a local Monge coordinate system. The first step is to regress the local target function using a generalized moving least squares (GMLS) while the second step is to compensate for the residual using a PHS interpolation. Our gRBF-FD method has the same interpolant form with the standard RBF-FD but differs in interpolation coefficients. Our approach utilizes a specific weight function in both the GMLS and PHS steps and implements an automatic tuning strategy for the stencil size K (i.e., the number of nearest neighbors) at each point. These strategies are designed to produce a Laplacian matrix with a specific coefficient structure, thereby enhancing stability and reducing the solution error. We establish an error bound for the operator approximation in terms of the so-called local stencil diameter as well as in terms of the number of data. We further demonstrate the high accuracy of gRBF-FD through numerical tests on various smooth manifolds.

---

## 96. ArticFlow: Generative Simulation of Articulated Mechanisms

**论文链接:** [http://arxiv.org/abs/2511.17883v1](http://arxiv.org/abs/2511.17883v1)

**作者:** Jiong Lin, Jinchen Ruan, Hod Lipson

**发布时间:** 2025-11-22

**备注:** 8 pages, 8 figures

### GPT解析

### 总结

ArticFlow是一种两阶段流匹配框架，用于可控且高质量的关节式3D生成，在MuJoCo Menagerie数据集上表现优异，既能作为生成模型也能作为神经模拟器。

### 背景

生成模型在静态3D形状方面取得了显著进展，但关节式3D生成仍面临挑战，因为它涉及动作相关的变形和有限的训练数据集。

### 目的

引入ArticFlow框架，学习从噪声到目标点集的可控速度场，并在明确动作控制下工作，以解决关节式3D生成问题。

### 方法

ArticFlow耦合了两个部分：一是将噪声传输到形状先验代码的潜在流；二是根据动作和形状先验传输点的点流，使单一模型能够表示多样化的关节类别并泛化到不同动作。

### 主要发现

在MuJoCo Menagerie上，ArticFlow既可作为生成模型也可作为神经模拟器，能从紧凑先验预测条件动作的运动学，并通过潜在插值合成新形态，相比其他方法实现了更高的运动学准确性和更好的形状质量。

### 结论

条件动作的流匹配是可控且高质量关节机构生成的实用途径。

### 翻译

生成模型的最新进展在静态3D形状方面取得了良好成果，而由于动作相关的变形和有限的数据集，关节式3D生成仍然具有挑战性。我们引入了ArticFlow，一个两阶段流匹配框架，它从噪声学习到目标点集的可控速度场，并在明确动作控制下工作。ArticFlow耦合了(i)将噪声传输到形状先验代码的潜在流，和(ii)根据动作和形状先验传输点的点流，使单个模型能够表示多样化的关节类别并泛化到不同动作。在MuJoCo Menagerie上，ArticFlow既可作为生成模型也可作为神经模拟器：它从紧凑先验预测条件动作的运动学，并通过潜在插值合成新形态。与特定对象的模拟器和静态点云生成器的动作条件变体相比，ArticFlow实现了更高的运动学准确性和更好的形状质量。结果表明，条件动作的流匹配是可控且高质量关节机构生成的实用途径。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决关节式3D对象的生成和控制问题，即如何生成具有运动学一致性的可变形3D对象并控制其关节动作。这个问题很重要，因为关节式对象在日常工具和机器人操作中非常普遍，而高质量的关节对象和仿真就绪的机器人模型成本高昂。现有数据集（如PartNet-Mobility）中每个类别的实例数量有限，机器人行为数据集已扩展到数百万条轨迹，但不同机器人形态只有22种，形成了形态瓶颈，限制了机器人的多样性和发展。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先分析了两种主要的3D生成范式（2D先验提升到3D和直接3D生成）在处理关节对象时的局限性。他们借鉴了生成模型的发展历程（从VAE、GAN到流匹配方法）、点云生成技术（如PointFlow、PVD）、关节对象建模方法（如Multi-BodySync、Ditto）和神经机器人仿真方法（如VSM）。设计思路是采用两阶段流匹配框架，分离形状和动作的条件，通过潜在流处理形状先验，通过点流处理动作条件下的变形，使用FiLM层注入条件信息，实现端到端的训练和生成。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是ArticFlow作为一个两阶段流匹配框架，学习从噪声到目标点集的可控速度场，耦合潜在流（将噪声传输到形状先验编码）和点流（在动作和形状先验条件下传输点），使单个模型能表示多样化关节类别并跨动作泛化。整体流程：训练阶段，输入点云和动作向量，分别编码为形状和动作潜在代码，通过两个流匹配模型（潜在流和点流）进行优化；采样阶段，给定动作命令，先通过潜在流生成形状代码，再通过点流生成点云；插值时，在形状潜在空间中进行球面线性插值，生成新形态。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：(1) 动作条件流匹配框架，学习在点空间中的速度场；(2) 扩展框架，联合动作和形状先验条件，覆盖整个关节类别；(3) 两阶段设计，分离形状和动作到独立潜在空间；(4) 支持彩色点云生成。相比之前工作，不同之处在于：与静态点云生成方法相比，增加了动作条件控制；与关节对象建模方法相比，更专注于生成而非分析，不依赖几何假设；与神经机器人仿真方法相比，能捕获整个类别而非特定机器人；与其他关节生成方法相比，提供自包含的3D显式解决方案和端到端训练。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'ArticFlow提出了一种两阶段流匹配框架，通过分离形状和动作条件，实现了对关节式3D对象的可控生成和仿真，解决了关节对象生成中的运动学一致性和几何保真度挑战。'}


### 论文摘要

Recent advances in generative models have produced strong results for static 3D shapes, whereas articulated 3D generation remains challenging due to action-dependent deformations and limited datasets. We introduce ArticFlow, a two-stage flow matching framework that learns a controllable velocity field from noise to target point sets under explicit action control. ArticFlow couples (i) a latent flow that transports noise to a shape-prior code and (ii) a point flow that transports points conditioned on the action and the shape prior, enabling a single model to represent diverse articulated categories and generalize across actions. On MuJoCo Menagerie, ArticFlow functions both as a generative model and as a neural simulator: it predicts action-conditioned kinematics from a compact prior and synthesizes novel morphologies via latent interpolation. Compared with object-specific simulators and an action-conditioned variant of static point-cloud generators, ArticFlow achieves higher kinematic accuracy and better shape quality. Results show that action-conditioned flow matching is a practical route to controllable and high-quality articulated mechanism generation.

---

## 97. IDEAL-M3D: Instance Diversity-Enriched Active Learning for Monocular 3D Detection

**论文链接:** [http://arxiv.org/abs/2511.19301v1](http://arxiv.org/abs/2511.19301v1)

**作者:** Johannes Meier, Florian Günther, Riccardo Marin, Oussema Dhaouadi, Jacques Kaiser, Daniel Cremers

**发布时间:** 2025-11-24

### GPT解析

### 总结

本文提出IDEAL-M3D，首个用于单目3D检测的实例级主动学习管道，解决了现有方法在图像级选择和基于不确定性选择的局限性，实现了更好的性能和资源节约。

### 背景

单目3D检测仅依赖单个相机，易于部署，但需要大量标注，特别是3D标签成本高昂。在标注预算有限的情况下，优先选择能带来最大性能提升的样本至关重要，这正是主动学习的重点。

### 目的

解决单目3D检测中主动学习的两个主要局限性：(1)现有方法选择整个图像而非实例，效率低下；(2)基于不确定性的选择导致深度模糊偏差，倾向于选择远处物体而忽略近处物体。

### 方法

提出IDEAL-M3D，首个单目3D检测的实例级管道。使用显式多样化、快速训练的集成方法改进单目3D的多样性驱动主动学习。通过异构骨干网络和任务无关特征、损失权重扰动以及时间依赖的bagging来诱导多样性。

### 主要发现

IDEAL-M3D在性能和资源节约方面表现出色：仅使用60%的标注，就能在KITTI验证和测试集上获得与使用整个数据集训练相同检测器相当或更好的AP3D结果。

### 结论

IDEAL-M3D通过实例级选择和多样化的主动学习方法，有效解决了单目3D检测中的标注效率问题，显著减少了标注需求同时保持了或提高了检测性能。

### 翻译

单目3D检测仅依赖单个相机，因此易于部署。然而，从单目图像实现可靠的3D理解需要大量标注，而3D标签尤其昂贵。在有限的标注预算下最大化性能，优先选择预期能带来最大性能提升的样本至关重要。这种优先选择就是主动学习的重点。值得注意的是，我们在单目3D目标检测的主动学习算法中观察到了两个显著局限性。首先，先前的方法选择整个图像，这效率低下，因为同一图像中包含的非信息性实例也需要被标注。其次，现有方法依赖于基于不确定性的选择，这在单目3D目标检测中会导致深度模糊的偏差。因此，远处物体被选中，而附近物体被忽视。为解决这些局限性，我们提出了IDEAL-M3D，这是首个用于单目3D检测的实例级管道。我们首次证明，一个显式多样化且快速训练的集成方法能够改进单目3D的多样性驱动主动学习。我们使用异构骨干网络和任务无关特征、损失权重扰动以及时间依赖的bagging来诱导多样性。IDEAL-M3D表现出卓越的性能和显著的资源节约：仅使用60%的标注，我们在KITTI验证和测试集上实现了与在整个数据集上训练相同检测器相似或更好的AP3D结果。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决单目3D目标检测中的主动学习效率问题。具体来说，它解决了两个关键问题：现有主动学习方法选择整个图像进行标注导致效率低下，以及基于不确定性的选择方法在单目3D检测中存在偏见，倾向于选择远处物体而忽略近处物体。这个问题在现实中非常重要，因为3D标注数据成本高昂且劳动密集，而单目3D检测在自动驾驶等领域有广泛应用。提高标注效率可以显著降低实际应用成本，解决远处物体偏见问题可以提高模型的实用性能。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先观察到现有主动学习方法在单目3D检测中的两个局限，然后提出转向实例级选择而非图像级选择。他们发现不确定性方法在实例级选择中效果不佳，导致远处物体被过度选择，因此从多样性角度重新思考主动学习。他们借鉴了Core-Set多样性选择方法作为基础，使用了集成学习技术来增强多样性，利用了预训练的图像自编码器获取任务无关特征，并参考了SAMv2用于实例分割。整体设计思路是从问题出发，逐步构建解决方案，同时巧妙地融合了多种现有技术。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是：1) 实例级主动学习：选择具体的物体实例而非整个图像进行标注；2) 多样性驱动选择：基于特征空间的多样性而非不确定性来选择样本；3) 多样化集成：使用异构骨干网络和任务无关特征来增强特征表示的多样性。整体实现流程包括：1) 初始化：标注一小部分随机图像来初始化检测器；2) 推理与候选提出：在未标注图像上运行推理，提出标注候选；3) 实例选择：使用多样性驱动的选择策略选择最有信息量的实例；4) 标注与训练：标注选中的实例，更新模型；5) 迭代：重复上述过程直到达到标注预算。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1) 第一个面向单目3D检测的实例级主动学习管道；2) 多样性驱动的选择策略，解决了不确定性方法的偏见问题；3) 高度多样化的集成模型，包括异构骨干网络、任务无关特征、损失权重扰动和时间相关的bagging；4) 结合任务相关和任务无关的特征进行选择；5) 提出了NAURC评估指标，实现了图像级和实例级方法的公平比较。相比之前工作的不同：从图像级选择转向实例级选择，提高了标注效率；从基于不确定性选择转向基于多样性选择，解决了远处物体偏见；使用多样化集成而非单一模型进行特征表示；结合了任务无关的视觉特征；显著降低了训练时间开销，比传统集成方法节省超过50%的训练时间。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'IDEAL-M3D通过创新的实例级多样性和集成驱动的主动学习方法，在单目3D目标检测中实现了显著减少标注需求的同时保持或超越全监督性能。'}


### 论文摘要

Monocular 3D detection relies on just a single camera and is therefore easy to deploy. Yet, achieving reliable 3D understanding from monocular images requires substantial annotation, and 3D labels are especially costly. To maximize performance under constrained labeling budgets, it is essential to prioritize annotating samples expected to deliver the largest performance gains. This prioritization is the focus of active learning. Curiously, we observed two significant limitations in active learning algorithms for 3D monocular object detection. First, previous approaches select entire images, which is inefficient, as non-informative instances contained in the same image also need to be labeled. Secondly, existing methods rely on uncertainty-based selection, which in monocular 3D object detection creates a bias toward depth ambiguity. Consequently, distant objects are selected, while nearby objects are overlooked.   To address these limitations, we propose IDEAL-M3D, the first instance-level pipeline for monocular 3D detection. For the first time, we demonstrate that an explicitly diverse, fast-to-train ensemble improves diversity-driven active learning for monocular 3D. We induce diversity with heterogeneous backbones and task-agnostic features, loss weight perturbation, and time-dependent bagging. IDEAL-M3D shows superior performance and significant resource savings: with just 60% of the annotations, we achieve similar or better AP3D on KITTI validation and test set results compared to training the same detector on the whole dataset.

---

## 98. Percept-WAM: Perception-Enhanced World-Awareness-Action Model for Robust End-to-End Autonomous Driving

**论文链接:** [http://arxiv.org/abs/2511.19221v1](http://arxiv.org/abs/2511.19221v1)

**作者:** Jianhua Han, Meng Tian, Jiangtong Zhu, Fan He, Huixin Zhang, Sitong Guo, Dechang Zhu, Hao Tang, Pei Xu, Yuze Guo, Minzhe Niu, Haojie Zhu, Qichao Dong, Xuechao Yan, Siyuan Dong, Lu Hou, Qingqiu Huang, Xiaosong Jia, Hang Xu

**发布时间:** 2025-11-24

### GPT解析

### 总结

Percept-WAM是一种增强感知的'世界感知-行动'模型，首次在单一视觉-语言模型中隐式整合2D/3D场景理解能力，通过基于网格条件的预测机制和IoU感知评分提高空间感知的准确性和鲁棒性，特别是在长尾、远距离和小物体场景中表现出色。

### 背景

自动驾驶严重依赖准确和鲁棒的空间感知，但许多故障源于不准确性和不稳定性，特别是在长尾场景和复杂交互中。当前视觉-语言模型在空间定位和理解方面表现较弱，基于这些系统的VLA因此表现出有限的感知和定位能力。

### 目的

解决自动驾驶中空间感知的准确性和稳定性问题，提出一种能够整合2D/3D场景理解能力的感知增强模型，提高自动驾驶系统在复杂场景中的感知和定位能力。

### 方法

将2D/3D感知任务统一为World-PV和World-BEV令牌，编码空间坐标和置信度；提出基于网格条件的预测机制用于密集物体感知，结合IoU感知评分和并行自回归解码；利用预训练VLM参数保留通用智能，能够直接输出感知结果和轨迹控制输出。

### 主要发现

在COCO 2D检测上达到51.7/58.9 mAP，在nuScenes BEV 3D检测上也达到同样成绩；与轨迹解码器集成后，在nuScenes和NAVSIM上提高了规划性能，在NAVSIM上超越了DiffusionDrive 2.1个PMDS点；在开放词汇和长尾泛化方面表现出强大能力。

### 结论

Percept-WAM通过整合2D/3D场景理解，有效解决了现有视觉-语言模型在空间定位方面的不足，在自动驾驶空间感知任务中表现出色，特别是在长尾和复杂情况下，为自动驾驶提供了更准确和鲁棒的空间感知能力。

### 翻译

自动驾驶严重依赖准确和鲁棒的空间感知。许多故障源于不准确性和不稳定性，特别是在长尾场景和复杂交互中。然而，当前视觉-语言模型在空间定位和理解方面表现较弱，基于这些模型的VLA系统因此表现出有限的感知和定位能力。为解决这些挑战，我们引入了Percept-WAM，一种增强感知的'世界感知-行动'模型，首次在单一视觉-语言模型中隐式整合2D/3D场景理解能力。Percept-WAM不依赖问答式空间推理，而是将2D/3D感知任务统一为World-PV和World-BEV令牌，这些令牌编码空间坐标和置信度。我们提出了用于密集物体感知的基于网格条件的预测机制，结合IoU感知评分和并行自回归解码，提高了在长尾、远距离和小物体场景中的稳定性。此外，Percept-WAM利用预训练的VLM参数保留通用智能（如逻辑推理），并能直接输出感知结果和轨迹控制输出。实验显示，Percept-WAM在下游感知基准测试中匹配或超过了经典检测器和分割器，在COCO 2D检测和nuScenes BEV 3D检测上均达到51.7/58.9 mAP。当与轨迹解码器集成时，它进一步提高了在nuScenes和NAVSIM上的规划性能，例如在NAVSIM上超越了DiffusionDrive 2.1个PMDS。定性结果进一步突出了其强大的开放词汇和长尾泛化能力。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决自动驾驶系统中的空间感知不准确和不稳定性问题，特别是在长尾场景和复杂交互中。这个问题很重要，因为自动驾驶依赖于精确的空间感知和环境推理能力，小的几何误差（如检测偏差、偏航漂移）在长尾条件下会累积成脆弱决策，直接影响真实世界路线的安全性和稳定性。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先分析了现有方法的局限性：QA式空间推理只提供间接定位信号，扩散-解码器管道缺乏LLM级别的推理能力。他们设计了一个统一的World-Awareness-Action框架，将世界状态嵌入到单个VLM中。借鉴了预训练VLM（如InternVL2-8B）的通用能力，BEV表示用于3D理解，UFO方法用于分割，以及Pix2Seq的序列化输出格式，但创新性地将这些元素整合到一个增强感知的框架中。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是将2D/3D感知能力隐式集成到单个视觉-语言模型中，使用World-PV和World-BEV令牌编码空间坐标和置信度，并通过网格条件预测和IoU感知评分提高稳定性。整体流程：1)输入多视图视频和文本；2)使用VLM编码图像生成World-PV令牌；3)通过网格插值进行2D感知；4)使用World-BEV令牌进行3D BEV感知；5)引入World-Action令牌进行轨迹预测；6)采用流式KV缓存提高效率。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点：1)感知增强的世界令牌，首次在单个VLM中统一2D/3D感知；2)网格条件的密集感知机制，结合IoU评分和并行解码；3)感知到行动的统一范式。相比之前工作：不同于QA式推理的间接定位，直接编码空间信息；不同于扩散模型放弃LLM推理，同时保留推理能力和增强感知；不同于专用感知模型，结合了感知和规划任务。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'Percept-WAM通过在单个视觉-语言模型中隐式集成2D/3D感知能力，并引入网格条件预测和IoU感知置信度校准，显著提高了自动驾驶系统在复杂场景中的感知鲁棒性和轨迹规划准确性。'}


### 论文摘要

Autonomous driving heavily relies on accurate and robust spatial perception. Many failures arise from inaccuracies and instability, especially in long-tail scenarios and complex interactions. However, current vision-language models are weak at spatial grounding and understanding, and VLA systems built on them therefore show limited perception and localization ability. To address these challenges, we introduce Percept-WAM, a perception-enhanced World-Awareness-Action Model that is the first to implicitly integrate 2D/3D scene understanding abilities within a single vision-language model (VLM). Instead of relying on QA-style spatial reasoning, Percept-WAM unifies 2D/3D perception tasks into World-PV and World-BEV tokens, which encode both spatial coordinates and confidence. We propose a grid-conditioned prediction mechanism for dense object perception, incorporating IoU-aware scoring and parallel autoregressive decoding, improving stability in long-tail, far-range, and small-object scenarios. Additionally, Percept-WAM leverages pretrained VLM parameters to retain general intelligence (e.g., logical reasoning) and can output perception results and trajectory control outputs directly. Experiments show that Percept-WAM matches or surpasses classical detectors and segmenters on downstream perception benchmarks, achieving 51.7/58.9 mAP on COCO 2D detection and nuScenes BEV 3D detection. When integrated with trajectory decoders, it further improves planning performance on nuScenes and NAVSIM, e.g., surpassing DiffusionDrive by 2.1 in PMDS on NAVSIM. Qualitative results further highlight its strong open-vocabulary and long-tail generalization.

---

## 99. StereoDETR: Stereo-based Transformer for 3D Object Detection

**论文链接:** [http://arxiv.org/abs/2511.18788v1](http://arxiv.org/abs/2511.18788v1)

**作者:** Shiyi Mu, Zichong Gu, Zhiqi Ai, Anqi Liu, Yilin Gao, Shugong Xu

**发布时间:** 2025-11-24

**备注:** Accepted by IEEE TCSVT, 2025

### GPT解析

### 总结

StereoDETR是一种基于DETR的高效立体3D物体检测框架，通过结合单目和立体视觉两个分支，实现了实时推理并达到与最先进方法相当的精度。

### 背景

与单目3D物体检测相比，立体视觉3D方法精度更高但计算开销大且延迟高。当前最先进的立体3D检测方法精度是单目方法的两倍，但推理速度只有单目方法的一半。

### 目的

开发一种高效的立体3D物体检测框架，在保持高精度的同时提高推理速度，实现实时性能。

### 方法

StereoDETR包含两个分支：单目DETR分支（基于2D DETR构建，增加通道预测物体尺度、方向和采样点）和立体视觉分支（利用多尺度视差特征预测物体级深度图）。两个分支通过可微分深度采样策略耦合，并引入约束监督策略处理遮挡问题，无需额外标注。

### 主要发现

StereoDETR实现了实时推理，是首个在速度上超越单目方法的立体视觉方法。在KITTI基准测试上达到具有竞争力的精度，并在行人和自行车子集上建立了新的最先进结果。

### 结论

StereoDETR成功平衡了精度和速度，实现了实时推理的同时保持了高精度，为立体3D物体检测提供了新的高效解决方案。

### 翻译

与单目3D物体检测相比，基于立体视觉的3D方法提供显著更高的精度，但仍存在高计算开销和延迟问题。最先进的立体3D检测方法达到单目方法两倍的精度，但其推理速度仅为单目方法的一半。在本文中，我们提出了StereoDETR，一种基于DETR的高效立体3D物体检测框架。StereoDETR包含两个分支：单目DETR分支和立体视觉分支。DETR分支基于2D DETR构建，增加额外通道用于预测物体尺度、方向和采样点。立体视觉分支利用低成本多尺度视差特征预测物体级深度图。这两个分支仅通过可微分深度采样策略耦合。为处理遮挡问题，我们引入了一种针对采样点的约束监督策略，无需额外标注。StereoDETR实现实时推理，是首个在速度上超越单目方法的立体视觉方法。在公共KITTI基准测试上也达到具有竞争力的精度，在行人和自行车子集上建立了新的最先进结果。代码可在https://github.com/shiyi-mu/StereoDETR-OPEN获取。


### 论文摘要

Compared to monocular 3D object detection, stereo-based 3D methods offer significantly higher accuracy but still suffer from high computational overhead and latency. The state-of-the-art stereo 3D detection method achieves twice the accuracy of monocular approaches, yet its inference speed is only half as fast. In this paper, we propose StereoDETR, an efficient stereo 3D object detection framework based on DETR. StereoDETR consists of two branches: a monocular DETR branch and a stereo branch. The DETR branch is built upon 2D DETR with additional channels for predicting object scale, orientation, and sampling points. The stereo branch leverages low-cost multi-scale disparity features to predict object-level depth maps. These two branches are coupled solely through a differentiable depth sampling strategy. To handle occlusion, we introduce a constrained supervision strategy for sampling points without requiring extra annotations. StereoDETR achieves real-time inference and is the first stereo-based method to surpass monocular approaches in speed. It also achieves competitive accuracy on the public KITTI benchmark, setting new state-of-the-art results on pedestrian and cyclist subsets. The code is available at https://github.com/shiyi-mu/StereoDETR-OPEN.

---

## 100. 论文ID: 2511.18713v1

**错误:** 无法读取论文信息 - [Errno 2] No such file or directory: 'storage/result/2511.18713v1.json'

---

## 101. Exploring Surround-View Fisheye Camera 3D Object Detection

**论文链接:** [http://arxiv.org/abs/2511.18695v1](http://arxiv.org/abs/2511.18695v1)

**作者:** Changcai Li, Wenwei Lin, Zuoxun Hou, Gang Chen, Wei Zhang, Huihui Zhou, Weishi Zheng

**发布时间:** 2025-11-24

**备注:** 9 pages,6 figures, accepted at AAAI 2026

### GPT解析

### 总结

研究探索了使用环绕式鱼眼相机系统实现端到端三维物体检测的技术可行性，开发了两种整合鱼眼图像几何特性的方法，并创建了新的评估基准数据集。

### 背景

经典基于针孔的3D物体检测器在应用于鱼眼图像时会出现性能下降。

### 目的

探索使用环绕式鱼眼相机系统实现端到端3D物体检测的技术可行性，并开发能够有效处理鱼眼图像几何特性的方法。

### 方法

开发两种方法：基于鸟瞰图(BEV)范式的FisheyeBEVDet和基于查询范式的FisheyePETR，两者都采用球形空间表示来捕获鱼眼几何特性；同时创建新的开放数据集Fisheye3DOD，使用CARLA合成，包含标准针孔和鱼眼相机阵列。

### 主要发现

鱼眼兼容建模方法在Fisheye3DOD数据集上比基线方法提高准确性最多达6.2%。

### 结论

通过整合鱼眼图像的独特几何特性到主流检测框架中，可以有效提高3D物体检测的准确性。

### 翻译

在这项工作中，我们探索了使用环绕式鱼眼相机系统实现端到端三维物体检测的技术可行性。具体来说，我们首先研究了将经典基于针孔的3D物体检测器转移到鱼眼图像时导致的性能下降。为了缓解这一问题，我们开发了两种将鱼眼图像的独特几何形状整合到主流检测框架中的方法：一种基于鸟瞰图范式，名为FisheyeBEVDet；另一种基于查询范式，名为FisheyePETR。这两种方法都采用球形空间表示来有效捕获鱼眼几何形状。鉴于缺乏专门的评估基准，我们发布了Fisheye3DOD，这是一个使用CARLA合成的新开放数据集，包含标准针孔和鱼眼相机阵列。在Fisheye3DOD上的实验表明，我们的鱼眼兼容建模比基线方法提高了最多6.2%的准确性。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决如何有效在全景鱼眼相机系统上实现端到端的3D物体检测问题。这个问题很重要，因为鱼眼相机系统具有360度感知能力，对自动驾驶汽车和机器人至关重要；现代量产汽车已广泛配备鱼眼相机；相比多针孔相机系统，鱼眼相机更紧凑、有物理冗余，更适合空间有限或成本敏感的场景。然而，鱼眼相机的非线性投影会导致物体被压缩到很少像素中，使检测困难，且缺乏专门评估基准，现有针孔检测器与鱼眼几何不兼容。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先意识到缺乏统一基准数据集，因此创建了Fisheye3DOD数据集，包含同步的针孔和鱼眼相机数据。通过评估现有方法发现直接转换会导致显著精度下降，随后提出在特征级别引入球面反向投影来解决几何不兼容问题。作者借鉴了两种主流3D检测范式：BEVDet的Lift-Splat-Shoot范式（改为球面分层）和PETR的3D查询编码方法（使用球面射线位置编码）。两种方法都采用球面空间表示来捕捉鱼眼几何特性，但分别设计了不同的处理流程：FisheyeBEVDet在球坐标中进行深度推理，FisheyePETR使用球面射线位置编码增强投影特征。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是通过球面特征表示来有效处理鱼眼相机的非线性投影特性。整体流程包括：1) 将鱼眼图像输入骨干网络提取2D特征；2) 将特征投影到球面等矩形表示上；3) FisheyeBEVDet将特征提升到3D球面网格构建BEV表示，进行深度推理和投影；4) FisheyePETR将特征与球面坐标编码融合，使用检测Transformer解码器让对象查询与特征交互。两种方法都通过在特征级别对鱼眼几何进行端到端建模，比图像级校正保留了更丰富的空间和语义信息。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1) 首个系统性地定量比较针孔和鱼眼成像3D感知性能的研究；2) 创建新的基准数据集Fisheye3DOD，同时提供针孔和鱼眼数据；3) 提出两种新检测框架FisheyeBEVDet和FisheyePETR，在特征级别对鱼眼几何进行端到端建模。相比之前工作，这篇论文首次直接比较两种成像模型，创建了专门的数据集，方法在特征级别建模而非图像级校正，显著提高了检测精度（比基线高4.5-6.2分），并专注于3D物体检测而非之前的深度估计或分割任务。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '这篇论文通过创建首个专门用于比较针孔和鱼眼相机3D感知性能的基准数据集Fisheye3DOD，并提出两种在特征级别对鱼眼几何进行端到端建模的新方法FisheyeBEVDet和FisheyePETR，显著提高了全景鱼眼相机系统的3D物体检测精度，为自动驾驶和机器人等领域的360度感知提供了新的解决方案。'}


### 论文摘要

In this work, we explore the technical feasibility of implementing end-to-end 3D object detection (3DOD) with surround-view fisheye camera system. Specifically, we first investigate the performance drop incurred when transferring classic pinhole-based 3D object detectors to fisheye imagery. To mitigate this, we then develop two methods that incorporate the unique geometry of fisheye images into mainstream detection frameworks: one based on the bird's-eye-view (BEV) paradigm, named FisheyeBEVDet, and the other on the query-based paradigm, named FisheyePETR. Both methods adopt spherical spatial representations to effectively capture fisheye geometry. In light of the lack of dedicated evaluation benchmarks, we release Fisheye3DOD, a new open dataset synthesized using CARLA and featuring both standard pinhole and fisheye camera arrays. Experiments on Fisheye3DOD show that our fisheye-compatible modeling improves accuracy by up to 6.2% over baseline methods.

---

## 102. MASS: Motion-Aware Spatial-Temporal Grounding for Physics Reasoning and Comprehension in Vision-Language Models

**论文链接:** [http://arxiv.org/abs/2511.18373v1](http://arxiv.org/abs/2511.18373v1)

**作者:** Xiyang Wu, Zongxia Li, Jihui Jin, Guangyao Shi, Gouthaman KV, Vishnu Raj, Nilotpal Sinha, Jingxi Chen, Fan Du, Dinesh Manocha

**发布时间:** 2025-11-23

### GPT解析

### 总结

本研究提出了一种改进视觉语言模型(VLMs)物理推理能力的方法，通过引入MASS-Bench基准和MASS模型，解决了VLMs在处理涉及运动动力学和空间相互作用的物理驱动推理任务时的局限性，显著提升了模型在物理相关视频理解任务上的表现。

### 背景

Vision Language Models (VLMs)在标准视频任务上表现良好，但在涉及运动动力学和空间相互作用的物理驱动推理方面存在困难。这种局限性降低了它们解释真实或AI生成内容(AIGC)视频的能力，以及生成物理一致内容的能力。

### 目的

提出一种方法，通过将物理世界上下文线索转换为与VLMs感知、理解和推理相一致的可解释表示，来弥补VLMs在物理推理方面的不足。

### 方法

引入MASS-Bench基准(包含4,350个真实世界和AIGC视频以及8,361个自由形式视频问答对)，专注于物理相关的理解任务，包含详细的注释；提出MASS模型，一种与模型无关的方法，通过基于深度的3D编码和视觉定位将时空信号注入VLM语言空间，并结合用于物体动力学的运动跟踪器；应用强化微调以加强跨模态对齐和推理。

### 主要发现

实验和消融研究表明，经过改进的VLMs比可比的更大基线模型以及先前的最先进模型分别高出8.7%和6.0%，在物理推理和理解方面性能接近闭源SoTA VLMs，如Gemini-2.5-Flash。

### 结论

这些结果验证了所提出方法的有效性，为提升VLMs在物理推理方面的能力提供了新思路。

### 翻译

视觉语言模型(VLMs)在标准视频任务上表现良好，但在涉及运动动力学和空间相互作用的物理驱动推理方面存在困难。这种局限性降低了它们解释真实或AI生成内容(AIGC)视频的能力，以及生成物理一致内容的能力。我们提出了一种方法，通过将物理世界上下文线索转换为与VLMs感知、理解和推理相一致的可解释表示来弥补这一差距。我们引入了MASS-Bench，这是一个包含4,350个真实世界和AIGC视频以及8,361个自由形式视频问答对的全面基准，专注于物理相关的理解任务，包含详细的注释，包括视觉检测、子片段定位和实体的全序列3D运动跟踪。我们进一步提出了MASS，一种与模型无关的方法，通过基于深度的3D编码和视觉定位将时空信号注入VLM语言空间，并结合用于物体动力学的运动跟踪器。为了加强跨模态对齐和推理，我们应用了强化微调。实验和消融研究表明，我们的改进VLMs比可比的更大基线模型以及先前的最先进模型分别高出8.7%和6.0%，在物理推理和理解方面性能接近闭源SoTA VLMs，如Gemini-2.5-Flash。这些结果验证了我们方法的有效性。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决视觉语言模型(VLMs)在处理涉及运动动力学和空间交互的物理驱动推理任务时表现不佳的问题。这个问题在现实中很重要，因为随着AI生成内容(AIGC)视频的兴起，物理异常(如不合理的轨迹、不一致的深度提示或时间不连贯的运动)变得越来越普遍，而VLMs经常产生幻觉或忽视这些物理异常，降低了它们解释真实或AI生成视频的能力以及生成物理一致内容的能力。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者认识到VLMs难以从原始视频像素和高层次文本监督中获取物理理解，因此提出了一种数据效率更高的替代方案：通过专门的空间和运动编码器增强VLMs，这些编码器能明确提取和表示关键视觉信号。作者借鉴了现有工作，如利用Grounding-DINO进行实体检测，SAM2生成分割掩码，CoTracker3进行运动跟踪，以及Depth Anything V2进行深度估计等工具。在训练策略上也借鉴了链式思维推理和强化微调等方法。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是将物理世界的上下文线索转化为与VLMs感知、理解和推理相一致的可解释表示，通过显式编码物体的运动和场景动态到语言空间来增强VLMs的物理推理能力。整体实现流程包括：1)实体中心视觉接地，检测和分割关键实体；2)空间运动特征提取，跟踪实体运动并计算3D轨迹；3)视觉特征表示，将空间-运动信号转换为结构化运动轨迹序列并通过自然语言模板表达；4)训练管道，使用链式思维推理和强化微调(特别是时间组相对策略优化)来加强模型对物理概念的理解。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)MASS-Bench基准，包含4,350个视频和8,361个物理相关的问答对，提供详细的运动接地注释；2)MASS方法，一种模型无关的运动感知空间-时间接地方法，通过3D编码和视觉接地将空间-时间信号注入VLM语言空间；3)结合强化微调加强跨模态对齐和推理。相比之前的工作，MASS不仅关注视觉特征，还明确编码了物体运动和场景动态，提供了更丰富的注释，包括实体级别的运动接地注释，覆盖空间和时间维度的视觉接地，以及跨视频的密集空间-运动表示。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '这篇论文通过运动感知的空间-时间接地方法将物理世界的上下文线索转化为VLMs可解释的表示，并建立了MASS-Bench基准，显著提升了VLMs在物理推理和理解任务上的性能，特别是在检测AIGC视频中的物理异常方面。'}


### 论文摘要

Vision Language Models (VLMs) perform well on standard video tasks but struggle with physics-driven reasoning involving motion dynamics and spatial interactions. This limitation reduces their ability to interpret real or AI-generated content (AIGC) videos and to generate physically consistent content. We present an approach that addresses this gap by translating physical-world context cues into interpretable representations aligned with VLMs' perception, comprehension, and reasoning. We introduce MASS-Bench, a comprehensive benchmark consisting of 4,350 real-world and AIGC videos and 8,361 free-form video question-answering pairs focused on physics-related comprehension tasks, with detailed annotations including visual detections, sub-segment grounding, and full-sequence 3D motion tracking of entities. We further present MASS, a model-agnostic method that injects spatial-temporal signals into the VLM language space via depth-based 3D encoding and visual grounding, coupled with a motion tracker for object dynamics. To strengthen cross-modal alignment and reasoning, we apply reinforcement fine-tuning. Experiments and ablations show that our refined VLMs outperform comparable and larger baselines, as well as prior state-of-the-art models, by 8.7% and 6.0%, achieving performance comparable to close-source SoTA VLMs such as Gemini-2.5-Flash on physics reasoning and comprehension. These results validate the effectiveness of our approach.

---

## 103. REXO: Indoor Multi-View Radar Object Detection via 3D Bounding Box Diffusion

**论文链接:** [http://arxiv.org/abs/2511.17806v1](http://arxiv.org/abs/2511.17806v1)

**作者:** Ryoma Yataka, Pu Perry Wang, Petros Boufounos, Ryuhei Takahashi

**发布时间:** 2025-11-21

**备注:** 26 pages, Accepted to AAAI 2026; Code to be released

### GPT解析

### 总结

本文提出REXO方法，将2D边界框扩散提升至3D雷达空间，通过显式跨视图特征关联和地面接触先验知识，显著提升了室内雷达目标检测性能。

### 背景

多视图室内雷达感知因成本效益和低隐私风险受到关注，但现有方法依赖隐式跨视图特征关联，在复杂场景中易导致特征匹配模糊和检测性能下降。

### 目的

解决现有方法中特征关联模糊导致的检测性能下降问题，提高复杂室内场景中的目标检测准确性。

### 方法

提出REXO方法，将DiffusionDet的2D边界框扩散过程提升到3D雷达空间，利用嘈杂3D边界框引导显式跨视图特征关联，并利用人与地面接触的先验知识减少扩散参数数量。

### 主要发现

在HIBER和MMVR两个公开室内雷达数据集上，REXO分别以+4.22 AP和+11.02 AP的优势超越最先进方法。

### 结论

REXO通过显式跨视图特征关联和先验知识利用，有效解决了复杂室内场景中目标检测的挑战，大幅提升了检测性能。

### 翻译

多视图室内雷达感知因其成本效益和低隐私风险而受到关注。现有方法通常依赖隐式跨视图雷达特征关联，如RFMask中的提议配对或RETR中的查询到特征交叉注意力，这可能导致复杂室内场景中特征匹配模糊和检测性能下降。为了解决这些局限性，我们提出了REXO（基于3D边界框扩散的多视图雷达目标检测），它将DiffusionDet的2D边界框（BBox）扩散过程提升到3D雷达空间。REXO利用这些嘈杂的3D边界框引导显式跨视图雷达特征关联，增强跨视图雷达条件去噪过程。通过考虑人与地面接触的先验知识，REXO通过从该先验知识中确定参数来减少扩散参数的数量。在两个开放的室内雷达数据集上评估，我们的方法以+4.22 AP的优势超越HIBER数据集上的最先进方法，在MMVR数据集上以+11.02 AP的优势超越。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决室内多视角雷达目标检测中的跨视角雷达特征关联问题。现有方法通常依赖隐式特征关联（如RFMask中的提议配对或RETR中的查询到特征的交叉注意力），在复杂室内场景中容易导致模糊的特征匹配，降低检测性能。这个问题很重要，因为雷达感知具有成本效益高和隐私风险低的优点，在室内场景（如智能家居、安全监控、人机交互等）有广泛应用前景，而现有方法在复杂场景中的表现限制了雷达技术的实际应用。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': "作者借鉴了DiffusionDet将物体检测重新表述为生成性去噪过程的思想，并将其从2D平面扩展到完整的3D雷达空间。作者注意到现有方法（如RFMask和RETR）在处理跨视角特征关联时的局限性，因此设计了显式的跨视角关联方法，在每个扩散时间步将嘈杂的3D边界框投影到每个雷达视图。同时，作者利用'人与地面接触'的先验知识来减少参数数量，并引入雷达条件去噪来提高检测准确性。这种方法结合了扩散模型、跨视角特征关联和几何约束等多种技术。", '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是将物体检测重新表述为3D雷达空间中的生成性去噪扩散过程，通过显式的跨视角雷达特征关联和雷达条件去噪来提高检测准确性，同时利用地面约束简化3D边界框表示。整体流程包括：1）训练阶段：使用共享骨干网络提取雷达特征，对真实3D边界框添加噪声产生随机边界框，在每个时间步投影边界框并提取特征，使用雷达条件检测器预测去噪边界框，并将结果投影到2D图像平面进行监督；2）推理阶段：采样随机3D边界框，通过逆向扩散过程去噪，转换为图像平面边界框并输出；3）应用地面约束减少参数；4）结合分类和边界框回归损失进行优化。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': "关键创新点包括：1）2D到3D的提升与显式跨视角关联，使特征关联随视图数量线性增长；2）跨视角雷达条件边界框检测，将关联特征作为条件缓解3D去噪挑战；3）地面约束利用'人与地面接触'先验减少参数。相比之前工作，REXO不同于RFMask的提议配对方式，也不同于RETR的查询交叉注意力机制，而是在3D雷达空间直接执行扩散，简化了跨视角关联，并允许集成几何约束，无需额外配对步骤。", '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'REXO通过将扩散模型从2D提升到3D雷达空间，结合显式跨视角特征关联和地面约束，显著提高了室内多视角雷达目标检测的准确性和泛化能力，在两个公开数据集上超越了现有方法。'}


### 论文摘要

Multi-view indoor radar perception has drawn attention due to its cost-effectiveness and low privacy risks. Existing methods often rely on {implicit} cross-view radar feature association, such as proposal pairing in RFMask or query-to-feature cross-attention in RETR, which can lead to ambiguous feature matches and degraded detection in complex indoor scenes. To address these limitations, we propose \textbf{REXO} (multi-view Radar object dEtection with 3D bounding boX diffusiOn), which lifts the 2D bounding box (BBox) diffusion process of DiffusionDet into the 3D radar space. REXO utilizes these noisy 3D BBoxes to guide an {explicit} cross-view radar feature association, enhancing the cross-view radar-conditioned denoising process. By accounting for prior knowledge that the person is in contact with the ground, REXO reduces the number of diffusion parameters by determining them from this prior. Evaluated on two open indoor radar datasets, our approach surpasses state-of-the-art methods by a margin of +4.22 AP on the HIBER dataset and +11.02 AP on the MMVR dataset.

---

## 104. 3D-Aware Multi-Task Learning with Cross-View Correlations for Dense Scene Understanding

**论文链接:** [http://arxiv.org/abs/2511.20646v1](http://arxiv.org/abs/2511.20646v1)

**作者:** Xiaoye Wang, Chen Tang, Xiangyu Yue, Wei-Hong Li

**发布时间:** 2025-11-25

**备注:** 3D-aware Multi-task Learning, Cross-view Correlations, Code will be available at https://github.com/WeiHongLee/CrossView3DMTL

### GPT解析

### 总结

本文提出了一种多任务学习方法，通过引入跨视图模块(CvM)来捕获3D感知能力，从而改善密集预测任务(如分割和深度估计)的性能。

### 背景

当前多任务学习方法主要在2D图像空间中捕获任务间关系，导致特征缺乏3D感知能力，不利于全面场景理解。

### 目的

解决如何在多任务学习中整合视图间的相关性(代价体积)作为几何一致性问题，为跨任务相关性建模提供3D感知能力。

### 方法

提出轻量级跨视图模块(CvM)，该模块跨任务共享，用于交换视图信息并捕获跨视图相关性，与MTL编码器特征集成进行多任务预测。该模块架构无关，适用于单视图和多视图数据。

### 主要发现

在NYUv2和PASCAL-Context数据集上的实验结果表明，该方法能有效将几何一致性注入现有MTL方法，提高性能。

### 结论

通过引入具有3D感知能力的跨视图模块，可以改善多任务学习中密集预测任务的性能。

### 翻译

本文解决了训练单一网络联合执行多个密集预测任务(如分割和深度估计)的挑战，即多任务学习(MTL)。当前方法主要在2D图像空间中捕获跨任务关系，通常导致缺乏3D感知能力的非结构化特征。我们认为3D感知能力对于建模跨任务相关性(全面场景理解所必需的)至关重要。我们提出通过整合视图间的相关性(即代价体积)作为MTL网络中的几何一致性来解决这个问题。具体来说，我们引入了一个轻量级的跨视图模块(CvM)，该模块跨任务共享，用于交换视图信息并捕获跨视图相关性，与来自MTL编码器的特征集成用于多任务预测。该模块是架构无关的，可应用于单视图和多视图数据。在NYUv2和PASCAL-Context上的大量结果表明，我们的方法有效地将几何一致性注入到现有的MTL方法中，从而提高了性能。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决多任务学习中缺乏3D感知能力的问题。现有的多任务学习方法主要在2D图像空间中捕获跨任务关系，导致特征缺乏3D结构，造成任务间几何不一致。这个问题很重要，因为在实际应用（如机器人自动化）中，深度估计、语义分割等任务存在几何关联，缺乏3D感知会严重影响多任务学习的性能和实用性。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者受现有3D感知方法启发，但发现现有方法存在局限：3DMTL没有直接集成多视图几何线索，而MuvieNeRF需要多视图数据和相机参数。作者设计了一个轻量级跨视图模块（CvM），包含空间感知编码器、多视图Transformer和成本体积模块，与多任务学习编码器并行工作。这种设计借鉴了多视图场景重建方法的思想，但专门针对多任务密集预测任务进行了优化。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是通过整合跨视图相关性（成本体积）作为几何一致性，使多任务学习具有3D感知能力。整体流程是：输入图像和邻近视图被送入MTL编码器提取特征；同时，CvM中的空间感知编码器提取几何特征，多视图Transformer进行跨视图信息交换，成本体积模块构建3D表示；最后将CvM的输出与MTL特征连接，通过任务特定解码器进行多任务预测。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1) 提出通过跨视图相关性使多任务学习具有3D感知能力；2) 设计轻量级跨视图模块，可无缝集成到现有多任务学习架构；3) 支持多视图训练但只需要单图像推理，无需相机参数；4) 空间感知编码器独立于主网络，专门捕获几何线索。相比之前工作，直接集成多视图几何线索而非简单投影，且不需要相机参数，更适用于实际场景。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '本文提出了一种通过跨视图相关性增强3D感知能力的多任务学习方法，通过轻量级跨视图模块将几何一致性注入现有架构，显著提升了密集场景理解任务中多个任务的性能。'}


### 论文摘要

This paper addresses the challenge of training a single network to jointly perform multiple dense prediction tasks, such as segmentation and depth estimation, i.e., multi-task learning (MTL). Current approaches mainly capture cross-task relations in the 2D image space, often leading to unstructured features lacking 3D-awareness. We argue that 3D-awareness is vital for modeling cross-task correlations essential for comprehensive scene understanding. We propose to address this problem by integrating correlations across views, i.e., cost volume, as geometric consistency in the MTL network. Specifically, we introduce a lightweight Cross-view Module (CvM), shared across tasks, to exchange information across views and capture cross-view correlations, integrated with a feature from MTL encoder for multi-task predictions. This module is architecture-agnostic and can be applied to both single and multi-view data. Extensive results on NYUv2 and PASCAL-Context demonstrate that our method effectively injects geometric consistency into existing MTL methods to improve performance.

---

## 105. Vision-Language Memory for Spatial Reasoning

**论文链接:** [http://arxiv.org/abs/2511.20644v1](http://arxiv.org/abs/2511.20644v1)

**作者:** Zuntao Liu, Yi Du, Taimeng Fu, Shaoshu Su, Cherie Ho, Chen Wang

**发布时间:** 2025-11-25

### GPT解析

### 总结

VLM²是一种具有持久记忆的视觉-语言模型，通过双记忆模块解决了现有模型在语义-几何不一致和缺乏持久记忆方面的挑战，实现了从2D视频中提取3D感知表示，并在空间推理任务上取得了最先进的性能。

### 背景

空间推理对智能机器人至关重要，但当前视觉-语言模型在基于视频的空间推理方面仍达不到人类水平。这种差距主要源于两个挑战：语义-几何不一致导致无法保持一致的3D理解，以及缺乏持久记忆来随时间保持3D表示和理解。

### 目的

解决现有视觉-语言模型在空间推理方面的两个主要挑战：语义-几何不一致和缺乏持久记忆，提出一种能够从2D视频中提取3D感知表示的模型。

### 方法

提出VLM²模型，引入双记忆模块：工作内存作为滑动窗口关注即时上下文，情节记忆整合和存储关键长期信息。这种设计使模型能够在固定计算成本下实现高效的长时程空间推理。

### 主要发现

在多个基准测试上的大量实验表明，VLM²在纯视频模型中实现了最先进的性能，显著推进了视觉空间智能的前沿。

### 结论

VLM²通过双记忆模块成功解决了现有视觉-语言模型在空间推理方面的关键挑战，实现了高效的长时程推理，为视觉空间智能领域带来了重要进展。

### 翻译

空间推理是智能机器人的关键能力，然而当前的视觉-语言模型在基于视频的空间推理方面仍未能达到人类水平。这种差距主要源于两个挑战：一是语义-几何不一致导致无法保持一致的3D理解，二是缺乏持久记忆来随时间保持3D表示和理解。为解决这些局限性，我们提出了VLM²，一种具有持久记忆的视觉-语言模型，用于空间推理，能够纯粹从2D视频中提取视图一致、3D感知的表示。具体而言，为了增强长时程推理，我们融入了一个双记忆模块，包括一个作为滑动窗口运作的工作内存，用于关注即时上下文；以及一个整合和存储关键长期信息的情节记忆。这种设计使得模型能够在固定计算成本下实现高效和长时程的空间推理。在多个基准测试上的大量实验表明，VLM²在纯视频模型中实现了最先进的性能，显著推进了视觉空间智能的前沿。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决视觉语言模型在视频输入下的空间推理不足问题，具体包括语义-几何特征错位和缺乏持久记忆两个挑战。这个问题很重要，因为空间推理是智能机器人的基本能力，使它们能感知、定位和推理物理世界中的空间关系，当前模型在这方面的不足限制了机器人在复杂环境中的导航和交互能力。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者基于对现有方法局限性的深入分析设计了这个方法。他们认识到现有VLMs在处理语义和几何特征融合时存在错位问题，且缺乏持久记忆机制。作者借鉴了3D大语言模型的工作（如LL3DA、Chat-3D）、空间推理在VLMs中的研究（如VSI-Bench）、记忆机制在计算机视觉领域的应用（如视频理解、3D重建）以及CUT3R模型中维护连续更新状态的方法。针对问题，作者设计了视角感知的几何对齐模块和自适应3D位置注入来解决特征错位，同时提出双记忆模块（工作记忆和情景记忆）来解决持久记忆问题。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是通过构建视角一致的3D感知表示和持久记忆机制来提升空间推理能力。整体流程包括：1) 从视频输入中提取视觉token、几何token和view token；2) 通过自适应3D位置注入将预测的3D坐标选择性注入视觉token；3) 通过视角感知的几何对齐解决几何特征的视角歧义；4) 通过交叉注意力融合语义和几何特征形成3D感知表示；5) 使用工作记忆（滑动窗口存储近期信息）和情景记忆（固定容量存储长期重要信息）的双记忆模块维持和更新3D表示；6) 将3D感知表示和记忆增强表示与指令输入连接到语言主干生成答案。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1) 视角一致的3D感知表示，通过自适应3D位置注入和视角感知几何对齐解决语义-几何错位；2) 双记忆模块，结合工作记忆和情景记忆实现高效持久的长期推理；3) 纯视频输入的3D理解，无需额外3D数据。相比之前的工作，不同之处在于：解决了现有方法简单融合几何和语义特征的问题；与3DLLM-Mem等相比，双记忆设计实现了有界而持久的推理；不依赖点云、深度图等3D输入，仅从2D视频构建3D表示；在长视频上表现更好，展现了更强的长期空间推理能力。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'VLM2通过视角一致的3D感知表示和双记忆模块，显著提升了视觉语言模型在视频输入下的空间推理能力，实现了纯视频输入的长期空间理解和记忆。'}


### 论文摘要

Spatial reasoning is a critical capability for intelligent robots, yet current vision-language models (VLMs) still fall short of human-level performance in video-based spatial reasoning. This gap mainly stems from two challenges: a semantic-geometric misalignment that prevents consistent 3D understanding, and the absence of persistent memory to retain 3D representation and understanding over time. To address these limitations, we present VLM$^2$, a Vision-Language Model with persistent Memory for spatial reasoning with a view-consistent, 3D-aware representation purely from 2D video. Specifically, to enhance long-horizon reasoning, we incorporate a dual-memory module, consisting of a working memory that operates as a sliding window to focus on immediate context, and an episodic memory that consolidates and stores critical long-term information. This design enables efficient and long-horizon spatial reasoning with a fixed computational cost. Extensive experiments on multiple benchmarks show that VLM$^2$ achieves state-of-the-art performance among video-only models, significantly advancing the frontier of visual-spatial intelligence.

---

## 106. Mistake Attribution: Fine-Grained Mistake Understanding in Egocentric Videos

**论文链接:** [http://arxiv.org/abs/2511.20525v1](http://arxiv.org/abs/2511.20525v1)

**作者:** Yayuan Li, Aadit Jain, Filippos Bellos, Jason J. Corso

**发布时间:** 2025-11-25

**备注:** 11 pages, 4 figures, 6 tables

### GPT解析

### 总结

该论文提出了MATT(错误归因)任务和MisFormer模型，用于细粒度理解第一人称视频中的错误，并通过MisEngine数据引擎创建了大规模错误数据集。

### 背景

以往的错误理解工作缺乏细粒度输出，无法具体指出错误的来源和位置。

### 目的

开发一种能够将错误具体归因于指令文本或尝试视频的方法，确定违反的指令部分、偏差变得不可逆的时间点以及错误在帧中的位置。

### 方法

开发了MisEngine数据引擎自动构建错误样本，提出了MisFormer统一注意力模型进行跨维度的错误归因，并在新数据集上进行了实验。

### 主要发现

MisEngine生成了比先前错误数据集大一到两个数量级的EPIC-KITCHENS-M和Ego4D-M数据集；MisFormer在多个基准上优于现有方法。

### 结论

MATT任务和MisFormer模型为第一人称视频中的错误理解提供了细粒度的解决方案，通过大规模数据集和统一模型实现了更好的错误归因性能。

### 翻译

我们引入了错误归因(MATT)，这是一个用于细粒度理解第一人称视频中人类错误的任务。与以往缺乏细粒度输出的错误理解工作不同，MATT将错误具体归因于输入指令文本或尝试视频。MATT确定指令的哪一部分被违反(语义角色)，偏差何时变得不可逆转(无返回点, PNR)，以及错误出现在PNR帧的哪个位置。我们开发了MisEngine，一个数据引擎，可以从现有数据集中自动构建富含归因的错误样本，并继承它们的注释。应用于大型第一人称语料库时，MisEngine生成了EPIC-KITCHENS-M和Ego4D-M两个数据集，它们比以前的错误数据集大一到两个数量级。然后，我们提出了MisFormer，一个基于统一注意力的模型，用于跨语义(什么)、时间(何时)和空间(哪里)维度的错误归因，使用MisEngine监督进行训练。在我们新数据集和先前基准上的实验表明，MisFormer优于强大的视频语言、时间定位、手物交互和错误检测基线模型。


### 论文摘要

We introduce Mistake Attribution (MATT), a task for fine-grained understanding of human mistakes in egocentric video. Unlike prior mistake understanding work, which lacks fine-grained output, MATT concretely attributes mistakes to the input instruction text or the attempt video. MATT determines what part of the instruction is violated (semantic role), when the deviation becomes irreversible (the Point-of-No-Return, PNR), and where the mistake appears in the PNR frame. We develop MisEngine, a data engine that automatically constructs attribution-rich mistake samples from existing datasets and inherits their annotations. Applied to large egocentric corpora, MisEngine yields EPIC-KITCHENS-M and Ego4D-M, two datasets that are up to two orders of magnitude larger than prior mistake datasets. We then present MisFormer, a unified attention-based model for mistake attribution across semantic (what), temporal (when), and spatial (where) dimensions, trained using MisEngine supervision. Experiments on our new datasets and prior benchmarks show that MisFormer outperforms strong video-language, temporal localization, hand-object interaction, and mistake-detection baselines.

---

## 107. Learning to Generate Human-Human-Object Interactions from Textual Descriptions

**论文链接:** [http://arxiv.org/abs/2511.20446v1](http://arxiv.org/abs/2511.20446v1)

**作者:** Jeonghyeon Na, Sangwon Baik, Inhee Lee, Junyoung Lee, Hanbyul Joo

**发布时间:** 2025-11-25

**备注:** Project Page: https://tlb-miss.github.io/hhoi/

### GPT解析

### 总结

本文提出了一种新的Human-Human-Object Interactions (HHOIs)研究问题，旨在建模两个人与物体互动的相关性，并开发了一个统一的生成框架来合成完整的HHOIs，该方法能够扩展到多人类设置，性能优于之前仅关注单人HOI的方法。

### 背景

人类在不同情境下的互动方式（包括人际距离、空间配置和运动）有很大差异，为了使机器能够理解这种复杂、依赖情境的行为，需要建模多人与周围场景的关系。

### 目的

提出并解决Human-Human-Object Interactions (HHOIs)这一新的研究问题，开发能够合成多人与物体交互的方法，并扩展到多人类设置。

### 方法

创建了专门的HHOIs数据集；利用图像生成模型合成HHOI数据；从HHOIs中提取单独的人与物体交互(HOIs)和人之间交互(HHIs)；使用基于分数的扩散模型训练文本到HOI和文本到HHI模型；开发统一的生成框架整合两个单独的模型，通过单个高级采样过程合成完整的HHOIs。

### 主要发现

实验结果表明，该方法能够根据文本描述生成真实的HHOIs；性能优于之前仅关注单人HOI的方法；成功实现了涉及物体的多人类运动生成。

### 结论

该方法有效解决了HHOIs建模和生成问题，将交互生成领域扩展到更复杂的多人类场景，为机器理解复杂的人类交互行为提供了新途径。

### 翻译

人类相互互动的方式，包括人际距离、空间配置和运动，在不同情况下差异显著。为了使机器能够理解这种复杂、依赖情境的行为，必须建模多人与周围场景的关系。在本文中，我们提出了一个新的研究问题，用于建模两个人参与涉及物体的共享互动时的相关性。我们将这种公式化称为Human-Human-Object Interactions (HHOIs)。为了克服缺乏专门HHOIs数据集的问题，我们展示了一个新捕获的HHOIs数据集以及一种利用图像生成模型合成HHOI数据的方法。作为中介，我们从HHOIs中获取单独的人与物体交互(HOIs)和人之间交互(HHIs)，并使用这些数据训练了基于分数的扩散模型的文本到HOI和文本到HHI模型。最后，我们提出了一个统一的生成框架，整合两个单独的模型，能够在单个高级采样过程中合成完整的HHOIs。我们的方法将HHOI生成扩展到多人类设置，使涉及两个以上个体的交互成为可能。实验结果表明，我们的方法根据文本描述生成真实的HHOIs，优于之前仅关注单人HOI的方法。此外，我们引入了涉及物体的多人类运动生成作为我们框架的应用。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决如何生成人类-人类-物体交互(HHOIs)场景的问题。这个问题在现实中很重要，因为人类行为本质上具有社会性和上下文依赖性，人们通过人际距离、空间配置和运动的模式化结构自然地相互交互。在研究中，现有的工作主要关注单个人与物体的交互(HOIs)或人与人之间的交互(HHIs)，很少有研究同时建模涉及多个人和共享物体的交互，而这在日常生活中普遍存在，如两个人一起坐在沙发上、共用一把伞或在白板旁讨论等场景。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先将复杂的HHOI问题分解为两个相对简单的子问题：人类-物体交互(HOI)和人类-人类交互(HHI)。由于缺乏专门的HHOI数据集，作者收集了新的数据集并利用图像生成模型合成数据。方法设计上，作者借鉴了基于分数的扩散模型(score-based diffusion model)，这种模型已在物体重排、物体姿态估计等任务中被证明有效。作者分别训练HOI和HHI模型，然后通过引入一致性损失和碰撞损失的高级采样技术将两者整合，生成连贯的HHOI场景。这种方法还借鉴了人体建模和姿态估计技术，并展示了框架可以扩展到多人类交互。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是将复杂的HHOI问题分解为HOI和HHI两个子问题，使用基于分数的扩散模型分别学习它们的分布，然后通过高级采样技术整合，生成连贯的场景。具体流程包括：1)收集真实数据和合成数据；2)训练身体姿态编码器-解码器和分别的HOI、HHI扩散模型；3)使用概率流ODE从噪声中采样；4)在采样过程中引入一致性损失确保人类在姿态和空间配置上保持一致，引入碰撞损失防止不合理穿透；5)调整损失权重生成最终HHOI场景。该方法还可扩展到多人类交互，并应用于多人类运动生成。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)首次提出并系统研究HHOI问题；2)构建专门的HHOI数据集；3)提出统一生成框架同时建模HOI和HHI；4)引入高级采样技术确保生成场景的一致性和物理合理性；5)展示方法可扩展到多人类交互。相比之前的工作，本文不仅关注单个人与物体交互，还考虑多个人与共享物体的交互；不仅研究人与人交互，还整合物体上下文；不仅单独处理HOI和HHI，还提出统一框架生成连贯场景；不仅局限于二元交互，还能处理多人类场景；还支持根据文本描述生成HHOI场景。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '这篇论文提出了一种基于分数的扩散模型框架，能够根据文本描述生成真实、连贯的多人类与物体交互场景，填补了现有研究在复杂社会行为建模方面的空白。'}


### 论文摘要

The way humans interact with each other, including interpersonal distances, spatial configuration, and motion, varies significantly across different situations. To enable machines to understand such complex, context-dependent behaviors, it is essential to model multiple people in relation to the surrounding scene context. In this paper, we present a novel research problem to model the correlations between two people engaged in a shared interaction involving an object. We refer to this formulation as Human-Human-Object Interactions (HHOIs). To overcome the lack of dedicated datasets for HHOIs, we present a newly captured HHOIs dataset and a method to synthesize HHOI data by leveraging image generative models. As an intermediary, we obtain individual human-object interaction (HOIs) and human-human interaction (HHIs) from the HHOIs, and with these data, we train an text-to-HOI and text-to-HHI model using score-based diffusion model. Finally, we present a unified generative framework that integrates the two individual model, capable of synthesizing complete HHOIs in a single advanced sampling process. Our method extends HHOI generation to multi-human settings, enabling interactions involving more than two individuals. Experimental results show that our method generates realistic HHOIs conditioned on textual descriptions, outperforming previous approaches that focus only on single-human HOIs. Furthermore, we introduce multi-human motion generation involving objects as an application of our framework.

---

## 108. AMB3R: Accurate Feed-forward Metric-scale 3D Reconstruction with Backend

**论文链接:** [http://arxiv.org/abs/2511.20343v1](http://arxiv.org/abs/2511.20343v1)

**作者:** Hengyi Wang, Lourdes Agapito

**发布时间:** 2025-11-25

**备注:** Project page: https://hengyiwang.github.io/projects/amber

### GPT解析

### 总结

AMB3R是一个用于大规模密集3D重建的多视图前馈模型，能够处理各种3D视觉任务。

### 背景

现有3D重建方法在处理大规模场景和多种任务时存在局限性，需要更高效、更通用的解决方案。

### 目的

开发一个能够处理多种3D视觉任务的大规模密集3D重建模型，并实现良好的性能和通用性。

### 方法

利用稀疏但紧凑的体积场景表示作为后端，实现具有空间紧凑性的几何推理；模型仅针对多视图重建进行训练，但可扩展到其他任务。

### 主要发现

AMB3R在相机姿态、深度和度量尺度估计、3D重建等方面达到了最先进性能，甚至超过了具有密集重建先验的基于优化的SLAM和SfM方法。

### 结论

AMB3R是一个通用的3D重建模型，无需任务特定的微调或测试时优化即可扩展到多种3D视觉任务，性能优越。

### 翻译

我们提出了AMB3R，这是一个用于大规模密集3D重建的多视图前馈模型，能够处理各种3D视觉任务。核心思想是利用稀疏但紧凑的体积场景表示作为后端，实现具有空间紧凑性的几何推理。虽然仅针对多视图重建进行训练，但我们证明AMB3R可以无缝扩展到未校准的视觉里程计（在线）或大规模运动结构，无需任务特定的微调或测试时优化。与之前基于点图的模型相比，我们的方法在相机姿态、深度和度量尺度估计、3D重建方面达到了最先进性能，甚至在常见基准测试中超越了具有密集重建先验的基于优化的SLAM和SfM方法。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决如何实现一个前馈的、具有度量尺度的三维重建模型，并使其能够支持多种三维视觉任务（如相机姿态估计、深度估计、三维重建、视觉里程计和运动恢复结构）。这个问题在现实中很重要，因为精确的三维重建对于机器人导航、自动驾驶、增强现实和数字孪生等应用至关重要。在研究中，它解决了现有前馈模型缺乏显式几何推理和空间紧凑性的问题，同时避免了传统方法中常见的测试时间优化过程，提高了效率。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者观察到点图模型虽然现代但缺乏显式几何推理，而紧凑的三维表示（如体素网格）能强制融合多视角观察为一致几何。因此，他们设计了一个结合前馈点图模型和紧凑后端表示的混合架构。他们借鉴了VGGT作为前端预测特征和几何，采用稀疏体素作为后端的空间数据结构，利用空间填充曲线将体素序列化，并用Transformer处理，最后通过零卷积将融合特征注入回前端。这种设计既保持了前馈效率，又引入了显式三维推理能力。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是将前馈的点图模型与紧凑的三维场景表示（后端）相结合，使模型能够进行显式的三维几何推理。整体流程为：1) 使用VGGT前端处理输入图像，预测点图和几何特征；2) 构建稀疏体素网格，将体素特征序列化为1D序列；3) 用Transformer处理这些序列；4) 将处理后的特征反序列化并注入回前端；5) 使用轻量级尺度头恢复度量深度信息；6) 对于视觉里程计，使用关键帧作为记忆实现顺序处理；7) 对于运动恢复结构，采用分而治之策略将图像分区处理。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1) 引入紧凑的三维场景表示作为后端，实现显式几何推理；2) 使用轻量级尺度头实现精确的度量尺度重建；3) 实现无需微调的无标定视觉里程计，首次证明前馈方法可超越基于优化的方法；4) 实现无需优化的大规模前馈运动恢复结构；5) 在7个任务、13个数据集上取得最先进性能。相比之前工作，AMB3R无需测试时间优化和任务特定微调，同时提供了显式的三维推理能力，性能超越了现有三维基础模型，甚至在某些任务上超过了基于优化的SLAM和SfM方法。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'AMB3R通过引入紧凑的三维场景表示作为后端，实现了一个高效、精确的前馈度量尺度三维重建系统，能够在无需任务特定微调或测试时间优化的情况下，支持从单目深度估计到大规模运动恢复结构的多种三维视觉任务，并在多个基准测试上取得了最先进性能。'}


### 论文摘要

We present AMB3R, a multi-view feed-forward model for dense 3D reconstruction on a metric-scale that addresses diverse 3D vision tasks. The key idea is to leverage a sparse, yet compact, volumetric scene representation as our backend, enabling geometric reasoning with spatial compactness. Although trained solely for multi-view reconstruction, we demonstrate that AMB3R can be seamlessly extended to uncalibrated visual odometry (online) or large-scale structure from motion without the need for task-specific fine-tuning or test-time optimization. Compared to prior pointmap-based models, our approach achieves state-of-the-art performance in camera pose, depth, and metric-scale estimation, 3D reconstruction, and even surpasses optimization-based SLAM and SfM methods with dense reconstruction priors on common benchmarks.

---

## 109. Geometry of Decision Making in Language Models

**论文链接:** [http://arxiv.org/abs/2511.20315v1](http://arxiv.org/abs/2511.20315v1)

**作者:** Abhinav Joshi, Divyanshu Bhatt, Ashutosh Modi

**发布时间:** 2025-11-25

**备注:** Accepted at NeurIPS 2025

### GPT解析

### 总结

本研究通过内在维度视角研究了大型语言模型中隐藏表示的几何结构，特别关注多选题问答设置中的决策动态，发现模型各层存在一致的ID模式，为理解LLMs的泛化和推理能力提供了新的几何见解。

### 背景

大型语言模型在多种任务上展现出强大的泛化能力，但其预测背后的内部决策过程仍然不透明。

### 目的

通过内在维度的视角研究大型语言模型中隐藏表示的几何结构，特别关注多选题问答设置中的决策动态。

### 方法

进行了一项大规模研究，使用28个开源的transformer模型，通过多种估计器估计各层的内在维度，同时量化各层在多选题问答任务上的性能。

### 主要发现

跨模型存在一致的内在维度模式：早期层在低维流形上操作，中层扩展了这个空间，后期层再次压缩它，收敛到与决策相关的表示。这些结果表明大型语言模型隐式地学习将语言输入投影到与任务特定决策对齐的、结构化的低维流形上。

### 结论

这些结果为泛化和推理在语言模型中如何出现提供了新的几何见解。

### 翻译

大型语言模型在多种任务上展现出强大的泛化能力，但其预测背后的内部决策过程仍然不透明。在这项工作中，我们通过内在维度的视角研究大型语言模型中隐藏表示的几何结构，特别关注多选题问答设置中的决策动态。我们进行了一项大规模研究，使用28个开源的transformer模型，并通过多种估计器估计各层的内在维度，同时量化各层在多选题问答任务上的性能。我们的发现揭示了跨模型存在一致的内在维度模式：早期层在低维流形上操作，中层扩展了这个空间，后期层再次压缩它，收敛到与决策相关的表示。这些结果表明，大型语言模型隐式地学习将语言输入投影到与任务特定决策对齐的、结构化的低维流形上，为泛化和推理在语言模型中如何出现提供了新的几何见解。


### 论文摘要

Large Language Models (LLMs) show strong generalization across diverse tasks, yet the internal decision-making processes behind their predictions remain opaque. In this work, we study the geometry of hidden representations in LLMs through the lens of \textit{intrinsic dimension} (ID), focusing specifically on decision-making dynamics in a multiple-choice question answering (MCQA) setting. We perform a large-scale study, with 28 open-weight transformer models and estimate ID across layers using multiple estimators, while also quantifying per-layer performance on MCQA tasks. Our findings reveal a consistent ID pattern across models: early layers operate on low-dimensional manifolds, middle layers expand this space, and later layers compress it again, converging to decision-relevant representations. Together, these results suggest LLMs implicitly learn to project linguistic inputs onto structured, low-dimensional manifolds aligned with task-specific decisions, providing new geometric insights into how generalization and reasoning emerge in language models.

---

## 110. ScenarioCLIP: Pretrained Transferable Visual Language Models and Action-Genome Dataset for Natural Scene Analysis

**论文链接:** [http://arxiv.org/abs/2511.20274v1](http://arxiv.org/abs/2511.20274v1)

**作者:** Advik Sinha, Saurabh Atreya, Aashutosh A, Sk Aziz Ali, Abhijit Das

**发布时间:** 2025-11-25

### GPT解析

### 总结

本文提出ScenarioCLIP模型，用于处理包含多个物体和动作的复杂场景图像，通过建模物体间关系来增强场景分析能力。

### 背景

现有CLIP类模型主要处理单目标分类或简单检索任务，最新方法虽通过挖掘负样本和优化提示改进了类级区分，但仍局限于预定义类别列表，缺乏对物体间关系的明确建模。PyramidCLIP部分解决了此问题但仍不完整。

### 目的

开发能够建模场景中物体间关系的模型，以更好地理解和分析复杂场景图像，提升跨模态检索和细粒度视觉理解能力。

### 方法

ScenarioCLIP接受输入文本、基础关系、图像及关系聚焦区域作为输入。在精选场景数据上预训练，针对下游任务微调。通过扩展现有室内外场景数据集创建新数据集，使用语言模型流程锚定动作、物体和关系，并建立综合基准进行评估。

### 主要发现

ScenarioCLIP在各种领域特定任务上展现出强大的零样本和微调性能，证明了建模物体间关系对场景分析的重要性。

### 结论

ScenarioCLIP有效解决了现有CLIP模型在复杂场景分析中的局限性，为跨模态理解和场景分析提供了新方法。代码和数据集已公开可用。

### 翻译

直到最近，CLIP类基础模型主要探索了短文本检索或场景中物体的分类作为单目标图像分类任务。对于给定文本提示的图像嵌入检索也是如此。然而，现实世界场景图像表现出丰富的组成结构，涉及多个物体和动作。CLIP文献的最新方法通过挖掘更难的负图像-文本对和使用LLMs优化永久文本提示来改进类级区分能力。然而，这些改进仍局限于预定义的类别列表，没有明确建模关系或组成结构。PyramidCLIP通过对齐全局和局部视觉特征部分解决了这一问题，但仍缺乏对物体间关系的明确建模。因此，为进一步利用这一方面进行场景分析，我们提出的ScenarioCLIP模型接受输入文本、基础关系、输入图像以及突出显示关系区域的聚焦区域作为输入。该模型在精选的场景数据上预训练，并针对专门的下游任务（如跨模态检索和细粒度视觉理解任务）进行微调。为解决领域特定数据集的缺乏问题，我们通过扩展现有多样化室内和室外场景数据集中的图像-文本对生成了新的数据集。我们使用现有语言模型的流程来锚定动作、物体和关系，并通过人工和自动筛选进行填充。我们为多种基于场景的任务建立了综合基准，并与许多基线方法进行了比较。ScenarioCLIP在各种领域特定任务上展现出强大的零样本和微调性能。我们的代码和数据集可在https://github.com/scenario-clip/ScenarioCLIP获取。


### 论文摘要

Until recently, the general corpus of CLIP-type fundamental models has widely explored either the retrieval of short descriptions or the classification of objects in the scene as SINGLE-object image classification task. The same holds for retrieving the image embedding (image retrieval task) given a text prompt. However, real-world scene images exhibit rich compositional structure involving multiple objects and actions. The latest methods in the CLIP-based literature improve class-level discrimination by mining harder negative image-text pairs and by refining permanent text prompts, often using LLMs. However, these improvements remain confined to predefined class lists and do not explicitly model relational or compositional structure. PyramidCLIP partially addresses this gap by aligning global and local visual features, yet it still lacks explicit modeling of inter-object relations. Hence, to further leverage this aspect for scene analysis, the proposed ScenarioCLIP model accepts input texts, grounded relations, and input images, along with focused regions highlighting relations. The proposed model is pretrained on curated scenario data, and finetuned for specialized downstream tasks, such as cross-modal retrieval and fine-grained visual understanding tasks. To address the lack of domain-specific datasets, we generate a novel dataset by extending image-text pairs from existing diverse indoor and outdoor scenario datasets that are publicly available. We used a pipeline of existing language models to ground action, object, and relations, filled by manual and automatic curation. We established a comprehensive benchmark for several scenario-based tasks and compared it with many baseline methods. ScenarioCLIP demonstrates robust zero-shot and finetune performance on various domain-specific tasks. Our code and dataset are available at https://github.com/scenario-clip/ScenarioCLIP

---

## 111. 论文ID: 2511.20201v1

**错误:** 无法读取论文信息 - [Errno 2] No such file or directory: 'storage/result/2511.20201v1.json'

---

## 112. Search for the potential electromagnetic counterparts of neutrino events in SDSS galaxies at z<0.1

**论文链接:** [http://arxiv.org/abs/2511.20184v1](http://arxiv.org/abs/2511.20184v1)

**作者:** O. Sergijenko., I. B. Vavilova, I. O. Izviekova, D. R. Karakuts

**发布时间:** 2025-11-25

**备注:** 12 pp., 1 Figure, 4 Tables

### GPT解析

### 总结

该研究通过识别与高能中微子同时发生的电磁发射，寻找潜在的中微子源宿主星系。

### 背景

识别与高能中微子同时发生的电磁发射对于多信使天文学至关重要，这类观测对于约束源定位、确定源类型和理解发射机制都是必要的。

### 目的

寻找高能中微子事件与星系之间的空间重合性，确定潜在的中微子源宿主星系。

### 方法

分析IceCube Gold警报和AMON识别的中微子-电磁重合事件，检查红移0.02至0.1的SDSS DR9形态学目录中的星系，寻找在50%包含半径内只有一个星系的中微子事件。

### 主要发现

在59个IceCube Gold警报中，发现3个在50%包含半径内只有一个星系（SDSS J231231.52+033415.1）；在24个中微子-电磁重合事件中，另外3个在相同半径内只有一个星系（SDSS J220711.14+122535.9）。这六个星系是潜在中微子源最有可能的宿主星系候选者。

### 结论

作者总结了这六个最有可能的宿主星系候选者的可用多波段数据以及2018年至2025年获得的ZTF光变曲线。

### 翻译

识别与高能中微子同时发生的电磁发射对于多信使天文学是根本重要的。这类观测对于约束源定位、确定源类型和理解发射机制都是必要的。通常，它们需要使用电磁设施跟进中微子警报（IceCube发布两个警报流：Gold，至少50%的天体物理起源概率；Bronze，至少30%的概率），主要在X射线和伽马射线波段。另一种方法涉及监测IceCube天图中的热点，即超过仪器灵敏度的位置。替代方法包括对可用中微子事件和源目录进行相关性分析。我们搜索了SDSS星系与高能中微子事件之间的空间重合性。分析包括IceCube Gold警报和AMON（天体物理多信使天文台网络）识别的中微子-电磁重合事件，截止到2025年9月底。我们检查了红移0.02至0.1的形态学目录中的星系，该目录包含315,776个SDSS DR9天体，r波段绝对星等范围从-24到-13。在59个IceCube Gold警报中，我们发现3个在50%包含半径内只有一个星系（SDSS J231231.52+033415.1）。在24个中微子-电磁重合事件中，另外3个在相同半径内只有一个星系（SDSS J220711.14+122535.9）。这六个星系代表了潜在中微子源最有可能的宿主星系候选者。我们总结了它们可用的多波段数据以及2018年至2025年获得的ZTF光变曲线。


### 论文摘要

Identification of electromagnetic emission in coincidence with high-energy neutrinos is fundamentally important for multimessenger astronomy. Such observations are essential for constraining source localization, determining the source type, and understanding emission mechanisms. Typically, they require following up a neutrino alert (IceCube issues two alert streams: Gold, with at least 50 percent probability of astrophysical origin, and Bronze, with at least 30 percent probability) with an electromagnetic facility, primarily in X-ray and gamma-ray bands. Another approach involves electromagnetic monitoring of hot spots in the IceCube skymap, i.e., positions exceeding the instrument sensitivity. An alternative method consists in performing correlation analysis across available neutrino events and source catalogs. We searched for spatial coincidence between galaxies from SDSS and high-energy neutrino events. The analysis includes IceCube Gold alerts and neutrino-electromagnetic coincidence events from AMON (Astrophysical Multimessenger Observatory Network), identified through the end of September 2025. We examined galaxies from the morphological catalog at redshifts 0.02 to 0.1, which contains 315,776 SDSS DR9 objects with absolute stellar magnitudes in the range from -24 to -13 in the r band. Among 59 IceCube Gold alerts, we found three with only one galaxy (SDSS J231231.52+033415.1) within the 50 percent containment radius. Among 24 neutrino-electromagnetic coincidence events, three more contain only one galaxy (SDSS J220711.14+122535.9) within the same radius. These six galaxies represent the most promising candidates for potential host galaxies of neutrino sources. We summarize their available multiwavelength data and the ZTF light curves obtained from 2018 to 2025.

---

## 113. While recognizing actions, LMMs struggle to detect core interaction events

**论文链接:** [http://arxiv.org/abs/2511.20162v1](http://arxiv.org/abs/2511.20162v1)

**作者:** Daniel Harari, Michael Sidorov, Liel David, Chen Shterental, Abrham Kahsay Gebreselasie, Muhammad Haris Khan

**发布时间:** 2025-11-25

### GPT解析

### 总结

本研究探索了大型多模态模型(LMMs)是否能将语义理解与实际视觉输入联系起来，通过测试模型识别视频中手与物体交互开始和结束时刻的能力，发现模型虽然能命名物体、识别动作并提供推理，但无法准确定位交互发生的具体帧和场景位置。

### 背景

大型多模态模型(LMMs)在图像和视频的实际视觉任务中展现出日益增长的能力，例如能够详细描述视频序列中的物体、环境和动态动作。

### 目的

探索这些模型在多大程度上将语义理解与实际视觉输入联系起来，特别是了解模型是否能识别手与物体交互的开始和结束时刻。

### 方法

研究者引入了一个首个大规模数据集，包含来自Something-Something-V2数据集的20,000多个标注视频交互，由250名AMTurk人类注释员标注核心交互事件。然后测试两个LMMs（Qwen-2.5VL和GPT-4o）在短视频中定位这些事件的能力。

### 主要发现

虽然模型可以可靠地命名目标物体、识别动作并提供连贯推理，但它们无法识别交互开始或结束的帧，也无法在场景中定位事件。

### 结论

模型在精确定义交互的物理接触时刻和位置方面存在困难，这表明它们缺乏对动态场景进行更深层次理解所需的感知基础。

### 翻译

大型多模态模型(LMMs)在图像和视频的实际视觉任务中展现出日益增长的能力。例如，给定视频序列，这些模型能够详细描述物体、环境和动态动作。在本研究中，我们探索了这些模型在多大程度上将语义理解与实际视觉输入联系起来。具体来说，给定手与物体交互的序列，我们询问模型交互何时开始或结束。为此，我们引入了首个此类大规模数据集，包含来自Something-Something-V2数据集的20,000多个标注交互。250名AMTurk人类注释员标注了核心交互事件，特别是物体和代理何时接触('contact')或分离('release')。我们要求两个LMMs（Qwen-2.5VL和GPT-4o）在包含单个事件的短视频中定位这些事件。结果表明，虽然模型可以可靠地命名目标物体、识别动作并提供连贯推理，但它们无法识别交互开始或结束的帧，也无法在场景中定位事件。我们的研究结果表明，在精确定义交互的物理接触时刻和位置方面存在困难，这表明模型缺乏对动态场景进行更深层次理解所需的感知基础。


### 论文摘要

Large multi-modal models (LMMs) show increasing performance in realistic visual tasks for images and, more recently, for videos. For example, given a video sequence, such models are able to describe in detail objects, the surroundings and dynamic actions. In this study, we explored the extent to which these models ground their semantic understanding in the actual visual input. Specifically, given sequences of hands interacting with objects, we asked models when and where the interaction begins or ends. For this purpose, we introduce a first of its kind, large-scale dataset with more than 20K annotated interactions on videos from the Something-Something-V2 dataset. 250 AMTurk human annotators labeled core interaction events, particularly when and where objects and agents become attached ('contact') or detached ('release'). We asked two LMMs (Qwen-2.5VL and GPT-4o) to locate these events in short videos, each with a single event. The results show that although the models can reliably name the target objects, identify the action and provide coherent reasoning, they consistently fail to identify the frame where the interaction begins or ends and cannot localize the event within the scene. Our findings suggest that in struggling to pinpoint the moment and location of physical contact that defines the interaction, the models lack the perceptual grounding required for deeper understanding of dynamic scenes.

---

## 114. WaymoQA: A Multi-View Visual Question Answering Dataset for Safety-Critical Reasoning in Autonomous Driving

**论文链接:** [http://arxiv.org/abs/2511.20022v1](http://arxiv.org/abs/2511.20022v1)

**作者:** Seungjun Yu, Seonho Lee, Namho Kim, Jaeyo Shin, Junsung Park, Wonjeong Ryu, Raehyuk Jung, Hyunjung Shim

**发布时间:** 2025-11-25

### GPT解析

### 总结

本研究提出了安全关键推理(Safety-Critical Reasoning)新任务，通过多视图输入解决自动驾驶中的高级推理挑战，并创建了WaymoQA数据集支持这一任务，实验证明该数据集能有效提高模型的推理能力。

### 背景

多模态大语言模型(MLLMs)在驾驶场景理解方面取得了进展，引起了其在自动驾驶中应用的兴趣。然而，在安全关键场景中的高级推理仍然是一个重大挑战，因为避免一种交通风险可能会创造另一种风险。

### 目的

定义安全关键推理这一新任务，利用多视图输入来应对自动驾驶中的高级推理挑战。

### 方法

将安全关键推理分为两个阶段：首先解决即时风险，然后减轻决策导致的下游风险。为此，引入了WaymoQA数据集，包含35,000个人类注释的问题-答案对，涵盖复杂的高风险驾驶场景。该数据集包括图像和视频模态的选择题和开放式格式。

### 主要发现

实验表明，与普通场景相比，现有的MLLMs在安全关键场景中的表现较差，但使用WaymoQA进行微调可以显著提高它们的推理能力。

### 结论

WaymoQA数据集在开发更安全和具有推理能力的驾驶代理方面是有效的。

### 翻译

多模态大语言模型(MLLMs)的最新进展展示了它们对驾驶场景的强大理解能力，引起了在自动驾驶中应用的兴趣。然而，在安全关键场景中的高级推理仍然是一个重大挑战，因为在这些场景中，避免一种交通风险可能会创造另一种风险。这种推理仅依靠单一前视图通常不可行，需要环境的全面视图，我们通过多视图输入来实现这一点。我们将安全关键推理定义为一个利用多视图输入应对这一挑战的新任务。然后，我们将安全关键推理分为两个阶段：首先解决即时风险，然后减轻决策导致的下游风险。为此，我们引入了WaymoQA，这是一个包含35,000个人类注释问题-答案对的数据集，涵盖复杂的高风险驾驶场景。该数据集包括图像和视频模态的选择题和开放式格式。实验表明，与普通场景相比，现有的MLLMs在安全关键场景中表现较差，但使用WaymoQA进行微调可以显著提高它们的推理能力，这凸显了我们的数据集在开发更安全和具有推理能力的驾驶代理方面的有效性。


### 论文摘要

Recent advancements in multimodal large language models (MLLMs) have shown strong understanding of driving scenes, drawing interest in their application to autonomous driving. However, high-level reasoning in safety-critical scenarios, where avoiding one traffic risk can create another, remains a major challenge. Such reasoning is often infeasible with only a single front view and requires a comprehensive view of the environment, which we achieve through multi-view inputs. We define Safety-Critical Reasoning as a new task that leverages multi-view inputs to address this challenge. Then, we distill Safety-Critical Reasoning into two stages: first resolve the immediate risk, then mitigate the decision-induced downstream risks. To support this, we introduce WaymoQA, a dataset of 35,000 human-annotated question-answer pairs covering complex, high-risk driving scenarios. The dataset includes multiple-choice and open-ended formats across both image and video modalities. Experiments reveal that existing MLLMs underperform in safety-critical scenarios compared to normal scenes, but fine-tuning with WaymoQA significantly improves their reasoning ability, highlighting the effectiveness of our dataset in developing safer and more reasoning-capable driving agents.

---

## 115. GazeProphetV2: Head-Movement-Based Gaze Prediction Enabling Efficient Foveated Rendering on Mobile VR

**论文链接:** [http://arxiv.org/abs/2511.19988v1](http://arxiv.org/abs/2511.19988v1)

**作者:** Farhaan Ebadulla, Chiraag Mudlpaur, Shreya Chaurasia, Gaurav BV

**发布时间:** 2025-11-25

### GPT解析

### 总结

本文提出了一种多模态方法来预测虚拟现实环境中的凝视行为，通过结合时间凝视模式、头部移动数据和视觉场景信息，使用门控融合机制和跨模态注意力来提高预测准确性。

### 背景

预测虚拟现实环境中的凝视行为是一个重大挑战，这对渲染优化和界面设计有重要影响。

### 目的

开发一种能够准确预测用户在VR环境中凝视行为的方法，以优化渲染效率和改进界面设计。

### 方法

采用多模态方法，结合时间凝视模式、头部移动数据和视觉场景信息，通过门控融合机制和跨模态注意力，根据上下文相关性自适应地加权不同数据源。

### 主要发现

在包含22个VR场景和530万个凝视样本的数据集上，结合多种模态比单独使用单个数据流提高了预测准确性；整合过去的凝视轨迹与头部方向和场景内容可提高1-3个未来帧的预测准确性；跨场景泛化测试显示93.1%的验证准确率和良好的时间一致性。

### 结论

该研究有助于理解虚拟环境中的注意力机制，在渲染优化、交互设计和用户体验评估方面有潜在应用，使虚拟系统能够无需昂贵眼动追踪硬件即可预测用户注意力模式。

### 翻译

预测虚拟现实环境中的凝视行为仍然是一个重大挑战，对渲染优化和界面设计有重要影响。本文介绍了一种用于VR凝视预测的多模态方法，结合了时间凝视模式、头部移动数据和视觉场景信息。通过利用具有跨模态注意力的门控融合机制，该方法能够根据上下文相关性自适应地加权凝视历史、头部移动和场景内容。使用包含22个VR场景和530万个凝视样本的数据集进行评估，结果表明与单独使用单个数据流相比，结合模态可以提高预测准确性。结果表明，将过去的凝视轨迹与头部方向和场景内容相结合可以提高未来1-3帧的预测准确性。跨场景泛化测试显示预测凝视轨迹具有93.1%的验证准确率和时间一致性。这些发现有助于理解虚拟环境中的注意力机制，同时表明在渲染优化、交互设计和用户体验评估方面有潜在应用。该方法朝着更高效的虚拟现实系统迈进，无需昂贵的眼动追踪硬件即可预测用户注意力模式。


### 论文摘要

Predicting gaze behavior in virtual reality environments remains a significant challenge with implications for rendering optimization and interface design. This paper introduces a multimodal approach to VR gaze prediction that combines temporal gaze patterns, head movement data, and visual scene information. By leveraging a gated fusion mechanism with cross-modal attention, the approach learns to adaptively weight gaze history, head movement, and scene content based on contextual relevance. Evaluations using a dataset spanning 22 VR scenes with 5.3M gaze samples demonstrate improvements in predictive accuracy when combining modalities compared to using individual data streams alone. The results indicate that integrating past gaze trajectories with head orientation and scene content enhances prediction accuracy across 1-3 future frames. Cross-scene generalization testing shows consistent performance with 93.1% validation accuracy and temporal consistency in predicted gaze trajectories. These findings contribute to understanding attention mechanisms in virtual environments while suggesting potential applications in rendering optimization, interaction design, and user experience evaluation. The approach represents a step toward more efficient virtual reality systems that can anticipate user attention patterns without requiring expensive eye tracking hardware.

---

## 116. Agent0-VL: Exploring Self-Evolving Agent for Tool-Integrated Vision-Language Reasoning

**论文链接:** [http://arxiv.org/abs/2511.19900v1](http://arxiv.org/abs/2511.19900v1)

**作者:** Jiaqi Liu, Kaiwen Xiong, Peng Xia, Yiyang Zhou, Haonian Ji, Lu Feng, Siwei Han, Mingyu Ding, Huaxiu Yao

**发布时间:** 2025-11-25

### GPT解析

### 总结

本文提出Agent0-VL，一种通过工具集成推理实现持续改进的自进化视觉语言智能体。该模型结合求解器和验证器两个角色，通过基于工具的自我验证和强化学习实现稳定的自我改进，无需人工注释或外部奖励模型。

### 背景

视觉语言智能体在各种多模态推理任务中取得显著进展，但其学习受限于人工注释监督。最近的自奖励方法试图让模型充当自己的批评者，但纯文本自我评估难以验证复杂视觉推理步骤，常遭受评估幻觉影响。

### 目的

解决视觉语言智能体在缺乏人工注释监督情况下的学习限制，特别是克服纯文本自我评估在验证复杂视觉推理步骤时的不足，实现模型的持续自我改进。

### 方法

Agent0-VL将工具使用融入推理、自我评估和自我修复，使模型能通过基于证据的分析进行内省和验证。它在单个大型视觉语言模型中统一求解器和验证器两个角色，通过自进化推理周期交互，基于工具的验证和强化学习共同对齐推理和评估分布。

### 主要发现

在几何问题解决和视觉科学分析实验中，Agent0-VL实现了比基础模型高12.5%的性能提升。通过零外部奖励的进化，无需人工注释或外部奖励模型即可对齐推理和验证行为，实现持续自我改进。

### 结论

Agent0-VL成功将工具集成推理应用于自我评估和自我修复，解决了视觉语言智能体在缺乏人工注释监督情况下的学习限制，实现了持续的自我改进，并在多模态推理任务中表现出色。

### 翻译

视觉语言智能体在各种多模态推理任务中取得了显著进展；然而，它们的学习仍然受限于人工注释监督的局限性。最近的自奖励方法试图通过允许模型充当自己的批评者或奖励提供者来克服这一限制。然而，纯基于文本的自我评估难以验证复杂的视觉推理步骤，并常常遭受评估幻觉的影响。为了应对这些挑战，受最近工具集成推理进展的启发，我们提出了Agent0-VL，一个通过工具集成推理实现持续改进的自进化视觉语言智能体。Agent0-VL不仅将工具使用融入推理中，还融入自我评估和自我修复，使模型能够通过基于证据的分析进行内省、验证和完善其推理。它在单个大型视觉语言模型中统一了两个协同角色：执行多回合工具集成推理的求解器，以及通过基于工具的批判生成结构化反馈和细粒度自我奖励的验证器。这些角色通过自进化推理周期进行交互，其中基于工具的验证和强化学习共同对齐推理和评估分布，以实现稳定的自我改进。通过这种零外部奖励的进化，Agent0-VL无需任何人工注释或外部奖励模型即可对齐其推理和验证行为，实现持续的自我改进。在几何问题解决和视觉科学分析实验中，Agent0-VL实现了比基础模型高12.5%的性能提升。我们的代码可在https://github.com/aiming-lab/Agent0/Agent0-VL获取。


### 论文摘要

Vision-language agents have achieved remarkable progress in a variety of multimodal reasoning tasks; however, their learning remains constrained by the limitations of human-annotated supervision. Recent self-rewarding approaches attempt to overcome this constraint by allowing models to act as their own critics or reward providers. Yet, purely text-based self-evaluation struggles to verify complex visual reasoning steps and often suffers from evaluation hallucinations. To address these challenges, inspired by recent advances in tool-integrated reasoning, we propose Agent0-VL, a self-evolving vision-language agent that achieves continual improvement with tool-integrated reasoning. Agent0-VL incorporates tool usage not only into reasoning but also into self-evaluation and self-repair, enabling the model to introspect, verify, and refine its reasoning through evidence-grounded analysis. It unifies two synergistic roles within a single LVLM: a Solver that performs multi-turn tool-integrated reasoning, and a Verifier that generates structured feedback and fine-grained self-rewards through tool-grounded critique. These roles interact through a Self-Evolving Reasoning Cycle, where tool-based verification and reinforcement learning jointly align the reasoning and evaluation distributions for stable self-improvement. Through this zero-external-reward evolution, Agent0-VL aligns its reasoning and verification behaviors without any human annotation or external reward models, achieving continual self-improvement. Experiments on geometric problem solving and visual scientific analysis show that Agent0-VL achieves an 12.5% improvement over the base model. Our code is available at \href{https://github.com/aiming-lab/Agent0/Agent0-VL}{this https URL}.

---

## 117. Modeling of turbulence kinetic energy added by wind-turbine wakes in the atmospheric boundary layer

**论文链接:** [http://arxiv.org/abs/2511.19881v1](http://arxiv.org/abs/2511.19881v1)

**作者:** Bowen Du, Jingshan Zhu, Baoliang Li, Mingwei Ge, Xintao Li, Yongqian Liu

**发布时间:** 2025-11-25

### GPT解析

### 总结

本文提出了一种新型风力涡轮机尾流湍流动能(TKE)预测模型，能够仅使用基本入流条件和风力涡轮机运行条件作为输入，准确预测尾流添加TKE的三维空间分布。模型包含两个子模块：计算方位角平均尾流添加TKE的模块和确定地面效应修正函数的模块。通过与多种数据集的比较验证，该模型表现出高精度和鲁棒性，平均归一化平均误差仅为8.13%。

### 背景

准确预测风力涡轮机尾流添加的湍流动能(TKE)对理解尾流恢复机制具有重要科学价值，同时该物理量也是工程应用中的关键输入参数。

### 目的

开发一种新型尾流添加TKE预测模型，能够仅使用基本入流条件和风力涡轮机运行条件作为输入，准确预测尾流添加TKE的三维空间分布。

### 方法

模型由两个子模块组成：一是基于从尾流添加TKE预算推导出的解析解计算方位角平均尾流添加TKE；二是利用自相似性确定地面效应修正函数。通过开发基于大涡模拟(LES)案例确定所有自由参数的方法，确保两个子模块的闭合，形成端到端预测框架。

### 主要发现

提出的模型能够准确预测尾流添加TKE的空间分布，特别是能够捕获尾流添加TKE的垂直不对称性和轮毂高度尾流添加TKE剖面的流向演化。与LES校准数据和文献中的公开验证数据集(包括各种入流和涡轮机运行条件下的LES和风洞实验)比较，模型表现出色。

### 结论

该模型具有鲁棒性和广泛适用性，平均归一化平均误差仅为8.13%，可直接应用于工程实践。

### 翻译

准确预测风力涡轮机尾流添加的湍流动能(TKE)对理解尾流恢复机制具有重要科学价值。此外，该物理量是工程应用的关键输入参数。在本研究中，我们提出了一种新型尾流添加TKE预测模型，仅使用基本入流条件和风力涡轮机运行条件作为输入，就能准确预测尾流添加TKE的三维空间分布。该模型包含两个子模块：一个用于计算方位角平均尾流添加TKE，另一个用于确定地面效应修正函数。方位角平均尾流添加TKE的计算基于从相应尾流添加TKE预算中推导出的解析解，而地面效应修正函数则利用其自相似性采用统一的函数形式确定。为确保这两个子模块的闭合，我们开发了基于大涡模拟(LES)案例确定所有自由参数的方法，形成了一个端到端的预测框架，使提出的模型能够直接应用于工程实践。我们将提出的模型与LES校准数据以及文献中公开的验证数据集进行了比较，这些数据集包括各种入流和涡轮机运行条件下的LES和风洞实验。比较结果表明，提出的模型能够准确预测尾流添加TKE的空间分布，特别是能够捕获尾流添加TKE的垂直不对称性和轮毂高度尾流添加TKE剖面的流向演化。在所有验证数据集上，提出的模型的平均归一化平均误差仅为8.13%，证明了其鲁棒性和广泛适用性。


### 论文摘要

Accurate prediction of turbulence kinetic energy (TKE) added by wind-turbine wakes is of significant scientific value for understanding the wake recovery mechanisms. Furthermore, this physical quantity is a critical input for engineering applications. In this study, we propose a novel wake-added TKE prediction model capable of accurately predict the three-dimensional spatial distribution of wake-added TKE using only basic inflow and wind turbine operation conditions as inputs. The model consists of two sub-modules: one for calculating the azimuthally-averaged wake-added TKE and the other for determining the ground effect correction function. The calculation of the azimuthally-averaged wake-added TKE is based on the analytical solution derived from the corresponding wake-added TKE budget, while the ground effect correction function is determined using a unified functional form, owing to its self-similarity. To ensure the closure of these two sub-modules, we develop methods for determining all free parameters based on the large-eddy simulation (LES) cases. This results in an end-to-end prediction framework, enabling direct engineering applications of the proposed model. We compared the proposed model with LES calibration data and publicly available validation datasets from the literature, which include LES and wind tunnel experiments under various inflow and turbine operating conditions. The comparison results show that the proposed model can accurately predict the spatial distribution of wake-added TKE, particularly capturing the vertical asymmetry of wake-added TKE and the streamwise evolution of the hub-height wake-added TKE profile. The averaged normalized mean absolute error of the proposed model across all validation datasets is only 8.13%, demonstrating its robustness and broad applicability.

---

## 118. CropVLM: Learning to Zoom for Fine-Grained Vision-Language Perception

**论文链接:** [http://arxiv.org/abs/2511.19820v1](http://arxiv.org/abs/2511.19820v1)

**作者:** Miguel Carvalho, Helder Dias, Bruno Martins

**发布时间:** 2025-11-25

### GPT解析

### 总结

CropVLM是一种外部低成本方法，通过强化学习训练，使VLMs能够动态放大相关图像区域，提升细粒度图像理解能力，无需修改或微调VLM即可显著改善性能。

### 背景

Vision-Language Models (VLMs)在需要细粒度图像理解的任务上表现不佳，如场景文本识别或文档分析，这主要源于感知限制和视觉碎片化问题。

### 目的

提出CropVLM方法，使VLMs能够动态'放大'相关图像区域，增强捕捉细节的能力，提升在细粒度图像理解任务上的性能。

### 方法

CropVLM使用强化学习进行训练，不依赖人工标注边界框或昂贵的合成评估，只需训练一次即可与各种VLMs配对使用。

### 主要发现

CropVLM在需要高分辨率图像理解的任务上显著改进，特别是在目标VLM领域外的基准测试中表现突出，且避免了灾难性遗忘问题。

### 结论

CropVLM是一种有效的外部增强方法，能够提升VLMs的细粒度图像理解能力，无需修改或微调基础模型。

### 翻译

视觉语言模型(VLMs)通常在需要细粒度图像理解的任务上表现不佳，如场景文本识别或文档分析，这是由于感知限制和视觉碎片化造成的。为解决这些挑战，我们引入了CropVLM作为一种外部低成本方法来提升性能，使VLMs能够动态'放大'相关图像区域，增强其捕捉细节的能力。CropVLM使用强化学习进行训练，不使用人工标注的边界框作为监督信号，也不使用昂贵的合成评估。模型只需训练一次，即可与开源和专有VLMs配对使用，提高它们的性能。我们的方法在需要高分辨率图像理解的任务上带来了显著改进，特别是对于目标VLM领域外的基准测试，无需修改或微调VLM，从而避免了灾难性遗忘。


### 论文摘要

Vision-Language Models (VLMs) often struggle with tasks that require fine-grained image understanding, such as scene-text recognition or document analysis, due to perception limitations and visual fragmentation. To address these challenges, we introduce CropVLM as an external low-cost method for boosting performance, enabling VLMs to dynamically ''zoom in'' on relevant image regions, enhancing their ability to capture fine details. CropVLM is trained using reinforcement learning, without using human-labeled bounding boxes as a supervision signal, and without expensive synthetic evaluations. The model is trained once and can be paired with both open-source and proprietary VLMs to improve their performance. Our approach delivers significant improvements on tasks that require high-resolution image understanding, notably for benchmarks that are out-of-domain for the target VLM, without modifying or fine-tuning the VLM, thus avoiding catastrophic forgetting.

---

## 119. Geometric Rényi mutual information induced by localized particle excitations in quantum field theory

**论文链接:** [http://arxiv.org/abs/2511.19729v1](http://arxiv.org/abs/2511.19729v1)

**作者:** Willy A. Izquierdo, David R. Junior, Gastão Krein

**发布时间:** 2025-11-24

### GPT解析

### 总结

本研究探讨了量子场论中局部粒子激发对空间区域间关联的影响，使用Schrödinger表示分析了自由无质量标量场的单粒子激发态下互补空间区域的Rényi互信息。

### 背景

量子场论即使在真空中也表现出丰富的空间关联结构，区域间的纠缠熵与共享边界的面积成正比。然而，局部粒子激发如何影响不同空间区域间的场值关联尚不清楚。

### 目的

研究局部单粒子激发对互补空间区域间Rényi互信息的影响。

### 方法

使用Schrödinger表示研究自由无质量标量场在(d+1)维中的局部单粒子激发的互补空间区域间的Rényi互信息，并在1+1维情况下具体评估了实数线的负半轴和正半轴之间的Rényi-2互信息。

### 主要发现

互信息包括真空项和激发诱导的贡献；激发产生了有限、正相关，当波包位于边界时最大化，随距离边界而减小，减小的速率由波包宽度决定。

### 结论

这些发现为从场论角度理解多粒子系统中的量子关联提供了步骤。

### 翻译

量子场论即使在真空中也表现出丰富的空间关联结构，区域间的纠缠熵与它们共享边界的面积成正比。虽然这种真空结构已被广泛研究，但局部粒子激发如何影响不同空间区域间的场值关联却知之甚少。在本工作中，我们使用Schrödinger表示研究自由无质量标量场在(d+1)维中局部单粒子激发的互补空间区域间的Rényi互信息。我们发现这种激发态中的互信息包括真空项和激发诱导的贡献。为获得定量结果，我们专门研究1+1维情况，并评估了实数线的负半轴和正半轴之间的Rényi-2互信息。我们发现激发产生了有限、正相关，当波包位于边界时最大化，并随其与边界的距离而减小，减小的速率由波包的宽度决定。我们的发现为从场论角度理解多粒子系统中的量子关联提供了步骤。


### 论文摘要

Quantum field theory exhibits rich spatial correlation structures even in the vacuum, where entanglement entropy between regions scales with the area of their shared boundary. While this vacuum structure has been extensively studied, far less is understood about how localized particle excitations influence correlations between field values in different spatial regions. In this work, we use the Schrödinger representation to study the Rényi mutual information between complementary spatial regions for a localized single-particle excitation of a free massless scalar field in $(d+1)$ dimensions. We find that the mutual information in this excited state includes both a vacuum term and an excitation-induced contribution. To obtain quantitative results, we specialize to $1+1$ dimensions and evaluate the Rényi-2 mutual information between the negative and positive halves of the real line.   We find that the excitation generates finite, positive correlations that are maximized when the wave packet sits at the boundary and decrease with its distance from it, at a rate determined by the wave packet's width. Our findings offer a step towards understanding quantum correlations in multiparticle systems from a field-theoretical point of view.

---

## 120. Alignment of radio jets in the microquasar V4641 Sagittarii with its high-energy structures

**论文链接:** [http://arxiv.org/abs/2511.19695v1](http://arxiv.org/abs/2511.19695v1)

**作者:** Josep Martí, Pedro Luis Luque-Escamilla

**发布时间:** 2025-11-24

**备注:** 6 pages, 3 figures. Accepted for publication as a Letter in Monthly Notices of the Royal Astronomical Society

### GPT解析

### 总结

研究V4641 Sagittarii微类星体系统中射电喷流与伽马射线发射的关系，发现它们可能沿共同轴排列，支持单一相对论性外流模型，挑战了先前的大尺度粒子扩散解释。

### 背景

V4641 Sagittarii是一个独特的银河系微类星体系统，包含一个从大质量伴星吸积物质的恒星级质量黑洞。其特征是存在相对论性射电喷流，几乎垂直于观测到的扩展伽马射线发射。

### 目的

探究射电喷流与甚高能(VHE)和超高能(UHE)伽马射线发射之间的关系和起源。

### 方法

通过观测证据分析射电喷流与伽马射线发射的空间关系和方向性。

### 主要发现

射电喷流与甚高能和超高能伽马射线发射可能沿共同轴排列，表明它们具有共空间或共方向的起源，支持在单一高度准直相对论性外流中产生同步射电发射、VHE和UHE伽马射线的模型。

### 结论

研究结果有利于就地加速粒子至数百TeV的场景，挑战了先前涉及大尺度粒子扩散的解释，简化了源的几何建模，突显了V4641 Sgr作为银河系内PeVatron候选者的潜力，并为理解微类星体中的喷流成分和磁场结构提供了基准。

### 翻译

V4641 Sagittarii (V4641 Sgr) 是一个独特的银河系微类星体系统，其特点是包含一个从大质量伴星吸积物质的恒星级质量黑洞。它的一个有趣特征是存在相对论性射电喷流，几乎垂直于观测到的扩展伽马射线发射，这意味着显著的传播效应或与银河系磁场的相互作用。在此，我们报告观测证据表明，射电喷流与甚高能(VHE)和超高能(UHE)伽马射线发射可能沿共同轴排列，表明具有共空间或共方向的起源。这种排列支持一个模型，即同步射电发射、VHE和UHE伽马射线在单一的高度准直相对论性外流中产生。我们的研究结果有利于就地加速粒子至数百TeV的场景，挑战了先前涉及大尺度粒子扩散的解释，并简化了该源的几何建模。这一案例突显了V4641 Sgr作为银河系内PeVatron候选者的潜力，并为理解微类星体中的喷流成分和磁场结构提供了基准。


### 论文摘要

V4641 Sagittarii (V4641 Sgr) is a unique Galactic microquasar system featuring a stellar-mass black hole accreting matter from a massive companion. One of its intriguing features is the presence of relativistic radio jets almost perpendicular to the observed extended gamma-ray emission, implying significant propagation effects or interactions with the Galactic magnetic field. Here we report observational evidence that the radio jet and the very high-energy (VHE) and ultra high-energy (UHE) gamma-ray emission could be aligned along a common axis, indicating a co-spatial or co-directional origin. This alignment supports a model where synchrotron radio emission, VHE and UHE gamma rays are produced within a single, highly collimated relativistic outflow. Our findings favor scenarios of in-situ particle acceleration up to hundreds of TeV, challenge previous interpretations involving large-scale particle diffusion, and simplify the geometric modeling of the source. This case highlights the potential of V4641 Sgr as a PeVatron candidate within our Galaxy and provides a benchmark for understanding jet composition and magnetic structure in microquasars.

---

## 121. Cook and Clean Together: Teaching Embodied Agents for Parallel Task Execution

**论文链接:** [http://arxiv.org/abs/2511.19430v1](http://arxiv.org/abs/2511.19430v1)

**作者:** Dingkang Liang, Cheng Zhang, Xiaopeng Xu, Jianzhong Ju, Zhenbo Luo, Xiang Bai

**发布时间:** 2025-11-24

**备注:** Accepted to AAAI 2026 (Oral). The code is available at \url{https://github.com/H-EmbodVis/GRANT}

### GPT解析

### 总结

本文提出了ORS3D（基于运筹学知识的3D基础任务调度）任务、ORS3D-60K数据集和GRANT模型，用于在具身AI中高效执行3D物理世界中的任务。

### 背景

任务调度对具身AI至关重要，使代理能够遵循自然语言指令并在3D物理世界中高效执行动作。然而，现有数据集通常通过忽略运筹学知识和3D空间基础来简化任务规划。

### 目的

提出ORS3D任务，要求语言理解、3D空间基础和效率优化的协同工作；构建ORS3D-60K数据集；提出GRANT模型，用于生成高效的任务计划和基础动作。

### 方法

提出ORS3D任务，要求代理利用可并行化的子任务来最小化总完成时间；构建包含4K个真实场景中60K个复合任务的ORS3D-60K数据集；提出GRANT，一个配备了简单有效的调度令牌机制的具身多模态大语言模型。

### 主要发现

在ORS3D-60K上的大量实验验证了GRANT在语言理解、3D空间基础和调度效率方面的有效性。

### 结论

ORS3D任务和数据集，以及GRANT模型为具身AI中的任务调度提供了新的研究方向和有效解决方案。

### 翻译

任务调度对具身AI至关重要，使代理能够遵循自然语言指令并在3D物理世界中高效执行动作。然而，现有数据集通常通过忽略运筹学知识和3D空间基础来简化任务规划。在这项工作中，我们提出了基于运筹学知识的3D基础任务调度（ORS3D），这是一个新任务，需要语言理解、3D空间基础和效率优化的协同工作。与先前设置不同，ORS3D要求代理利用可并行化的子任务来最小化总完成时间，例如，在微波炉运行的同时清洗水槽。为了促进ORS3D的研究，我们构建了ORS3D-60K，这是一个大规模数据集，包含4K个真实场景中的60K个复合任务。此外，我们提出了GRANT，一个配备了简单而有效的调度令牌机制的具身多模态大语言模型，用于生成高效的任务计划和基础动作。在ORS3D-60K上的大量实验验证了GRANT在语言理解、3D空间基础和调度效率方面的有效性。代码可在https://github.com/H-EmbodVis/GRANT获取。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决具身智能体在3D物理世界中高效执行复合任务的调度问题。具体来说，它关注如何让智能体利用运筹学知识来优化任务调度，特别是通过并行执行可并行的子任务来最小化总完成时间。这个问题很重要，因为现实世界中我们经常需要机器人等智能体同时执行多个任务或高效完成复合任务，而现有方法通常简化了任务规划过程，忽略了运筹学知识和3D空间定位的结合，导致效率低下且无法直接应用于物理世界。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先识别出现有任务规划方法的两个主要局限：缺乏对任务属性的考虑和效率优化，以及将3D环境简化为文本问答而未明确空间定位。基于这些局限，他们提出了ORS3D新任务，并构建了ORS3D-60K数据集。在设计方法时，作者借鉴了现有的3D场景理解技术（如Mask3D和OneFormer3D）、多模态大语言模型架构、运筹学中的动态规划算法，以及点云处理和3D场景编码技术，但创新性地将它们整合在一个统一的框架中，通过调度令牌机制连接LLM与外部优化求解器。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是通过调度令牌机制（STM）连接大型语言模型与外部优化求解器，使智能体能生成高效的任务调度，同时准确定位3D场景中的目标对象。整体流程包括：1) 多模态输入处理（点云和文本标记化）；2) 调度令牌机制（识别子任务类型，调用外部优化求解器生成最优调度）；3) 3D定位头（通过特殊令牌定位目标对象）；4) 训练与推理（使用交叉熵损失和sigmoid focal损失进行监督，在推理时先预测子任务属性，再生成最优调度）。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1) 提出ORS3D新任务，要求同时解决语言理解、3D定位和效率优化；2) 构建ORS3D-60K大规模数据集，首个融入运筹学知识的任务调度数据集；3) 设计调度令牌机制（STM）连接LLM与外部优化求解器；4) 整合多模态信息实现端到端训练。与之前工作不同，本文关注并行任务调度而非顺序执行，首次将运筹学知识引入3D场景任务调度，通过专门标记实现场景编码器、定位头与LLM的集成，更接近实际应用场景。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '这篇论文提出了基于运筹学知识的3D空间定位任务调度新任务、构建了ORS3D-60K大规模数据集，并设计了通过调度令牌机制实现高效并行任务执行的GRANT模型，使具身智能体能在3D物理世界中更高效地完成复合任务。'}


### 论文摘要

Task scheduling is critical for embodied AI, enabling agents to follow natural language instructions and execute actions efficiently in 3D physical worlds. However, existing datasets often simplify task planning by ignoring operations research (OR) knowledge and 3D spatial grounding. In this work, we propose Operations Research knowledge-based 3D Grounded Task Scheduling (ORS3D), a new task that requires the synergy of language understanding, 3D grounding, and efficiency optimization. Unlike prior settings, ORS3D demands that agents minimize total completion time by leveraging parallelizable subtasks, e.g., cleaning the sink while the microwave operates. To facilitate research on ORS3D, we construct ORS3D-60K, a large-scale dataset comprising 60K composite tasks across 4K real-world scenes. Furthermore, we propose GRANT, an embodied multi-modal large language model equipped with a simple yet effective scheduling token mechanism to generate efficient task schedules and grounded actions. Extensive experiments on ORS3D-60K validate the effectiveness of GRANT across language understanding, 3D grounding, and scheduling efficiency. The code is available at https://github.com/H-EmbodVis/GRANT

---

## 122. Chain-of-Visual-Thought: Teaching VLMs to See and Think Better with Continuous Visual Tokens

**论文链接:** [http://arxiv.org/abs/2511.19418v1](http://arxiv.org/abs/2511.19418v1)

**作者:** Yiming Qin, Bomin Wei, Jiaxin Ge, Konstantinos Kallidromitis, Stephanie Fu, Trevor Darrell, Xudong Wang

**发布时间:** 2025-11-24

**备注:** Project page: https://wakalsprojectpage.github.io/comt-website/

### GPT解析

### 总结

这篇论文提出了Chain-of-Visual-Thought (COVT)框架，解决了视觉语言模型在需要密集视觉感知任务中的局限性，通过引入连续视觉token使模型能够进行更精确的视觉推理，并在多个基准测试上显著提升了性能。

### 背景

当前视觉语言模型(VLMs)在需要密集视觉感知的理解任务中表现不佳，如空间推理和几何感知。这是因为它们缺乏捕捉空间维度密集视觉信息的有效机制。

### 目的

开发一种使VLMs能够进行更有效视觉推理的框架，解决其在感知理解任务中的局限性，实现更精确、有基础和可解释的多模态智能。

### 方法

提出Chain-of-Visual-Thought (COVT)框架，使VLMs能够通过连续视觉token（编码丰富感知线索的紧凑潜在表示）进行推理。COVT从轻量级视觉专家中提炼知识，捕获2D外观、3D几何、空间布局和边缘结构等互补特性。训练时，模型自回归地预测视觉token以重建密集监督信号；推理时，直接在连续视觉token空间中进行推理。

### 主要发现

在十多个多样化感知基准测试中，将COVT集成到强VLMs如Qwen2.5-VL和LLaVA中，性能一致提升3%至16%。证明紧凑的连续视觉思维能够实现更精确、有基础和可解释的多模态智能。

### 结论

COVT框架有效解决了VLMs在密集视觉感知任务中的局限性，通过引入连续视觉token进行推理，显著提升了模型在多种感知基准测试上的性能，为实现更精确、有基础和可解释的多模态智能提供了新途径。

### 翻译

视觉语言模型(VLMs)在语言空间推理方面表现出色，但在需要密集视觉感知的理解任务中存在困难，例如空间推理和几何感知能力有限。这一局限性源于当前VLMs缺乏捕捉空间维度密集视觉信息的机制。我们引入了Chain-of-Visual-Thought (COVT)框架，使VLMs不仅能够通过文字推理，还能通过连续视觉token（编码丰富感知线索的紧凑潜在表示）进行推理。在约20个token的小预算内，COVT从轻量级视觉专家中提炼知识，捕获互补特性，如2D外观、3D几何、空间布局和边缘结构。在训练过程中，带有COVT的VLM自回归地预测这些视觉token以重建密集监督信号（如深度、分割、边缘和DINO特征）。在推理时，模型直接在连续视觉token空间中进行推理，保持效率，同时可选择解码密集预测以提高可解释性。在包括CV-Bench、MMVP、RealWorldQA、MMStar、WorldMedQA和HRBench在内的十多个多样化感知基准测试中，将COVT集成到Qwen2.5-VL和LLaVA等强VLMs中，性能一致提升3%至16%，证明紧凑的连续视觉思维能够实现更精确、有基础和可解释的多模态智能。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决视觉语言模型（VLMs）在需要密集视觉感知的任务上表现不佳的问题，如空间推理和几何感知。当前VLMs缺乏有效机制来捕获跨空间维度的密集视觉信息，且将连续视觉信息投影到离散文本空间会导致丰富的感知线索（如边界、布局、深度和几何）丢失。这个问题在现实中很重要，因为它限制了AI系统在需要精确视觉理解的应用场景（如自动驾驶、医疗影像分析和机器人导航）中的表现，阻碍了AI系统接近人类的视觉推理能力。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': "作者观察到文本链式思维（CoT）在语言推理中成功但在视觉推理中表现不佳，意识到视觉信息本质上是连续的和高维的，而现有模型使用符号语言令牌进行推理缺乏保真度。他们希望让VLMs像人类一样'思考'视觉信息，而非将一切转化为文字。作者借鉴了文本CoT的结构化中间推理步骤思想，以及工具增强推理、潜在空间推理（如Coconut和CCoT）和Aurora等现有工作，但针对视觉领域的特殊性进行了创新设计。", '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是引入连续视觉令牌（紧凑的潜在表示，编码丰富的感知线索），使VLMs能够在连续视觉空间中进行推理，形成视觉思维链，将语义推理与感知基础联系起来。整体实现流程包括：1)选择四种视觉令牌（分割、深度、边缘和DINO令牌）；2)根据视觉模型粒度采用不同对齐策略；3)通过四个阶段（理解、生成、推理和高效推理）进行训练；4)在推理中形成视觉思维链，可选择解码为密集预测提供可解释性。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)引入连续视觉令牌作为多模态思维的基本构建块；2)设计多阶段训练流程使模型逐步学习使用视觉令牌；3)根据不同视觉模型粒度采用定制化对齐策略；4)创建自包含框架无需外部工具。相比之前工作，CoVT在连续视觉空间而非离散文本空间进行推理，比工具增强方法更高效，比MCoT计算开销更小，比VChain能保留更多视觉信息，比Aurora整合了更全面的视觉信息（分割、边缘和语义）。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'Chain-of-Visual-Thought通过引入连续视觉令牌，使视觉语言模型能够在连续视觉空间中进行推理，显著提升了模型在需要密集视觉感知任务上的性能，同时保持了自包含性和可解释性。'}


### 论文摘要

Vision-Language Models (VLMs) excel at reasoning in linguistic space but struggle with perceptual understanding that requires dense visual perception, e.g., spatial reasoning and geometric awareness. This limitation stems from the fact that current VLMs have limited mechanisms to capture dense visual information across spatial dimensions. We introduce Chain-of-Visual-Thought (COVT), a framework that enables VLMs to reason not only in words but also through continuous visual tokens-compact latent representations that encode rich perceptual cues. Within a small budget of roughly 20 tokens, COVT distills knowledge from lightweight vision experts, capturing complementary properties such as 2D appearance, 3D geometry, spatial layout, and edge structure. During training, the VLM with COVT autoregressively predicts these visual tokens to reconstruct dense supervision signals (e.g., depth, segmentation, edges, and DINO features). At inference, the model reasons directly in the continuous visual token space, preserving efficiency while optionally decoding dense predictions for interpretability. Evaluated across more than ten diverse perception benchmarks, including CV-Bench, MMVP, RealWorldQA, MMStar, WorldMedQA, and HRBench, integrating COVT into strong VLMs such as Qwen2.5-VL and LLaVA consistently improves performance by 3% to 16% and demonstrates that compact continuous visual thinking enables more precise, grounded, and interpretable multimodal intelligence.

---

## 123. LAST: LeArning to Think in Space and Time for Generalist Vision-Language Models

**论文链接:** [http://arxiv.org/abs/2511.19261v1](http://arxiv.org/abs/2511.19261v1)

**作者:** Shuai Wang, Daoan Zhang, Tianyi Bai, Shitong Shao, Jiebo Luo, Jiaheng Wei

**发布时间:** 2025-11-24

### GPT解析

### 总结

论文提出了LAST（LeArn to Think in Space and Time）方法，通过构建3D空间和时间维度的视觉思维轨迹，提高视觉语言模型对3D空间和长视频的理解能力，在多种基准测试中取得显著提升。

### 背景

人类能够通过连续视觉观察感知和理解3D空间和长视频，但即使是先进的视觉语言模型（VLMs）在这方面的能力仍然有限。当前方法通常采用专门架构设计分别处理3D任务和视频理解任务。

### 目的

提出一种方法，能够同时提高通用视觉语言模型对3D空间和长视频的理解能力，仅使用一组2D图像作为输入。

### 方法

LAST方法让VLMs在给出最终答案前，在空间和时间维度上'思考'，而非仅依赖文本，构建视觉思维轨迹。在零样本场景直接提示专有模型，以及使用包含思维轨迹的数据微调通用VLMs两种场景中验证效果。

### 主要发现

LAST在3个空间理解任务、4个视频理解任务和3个图像理解任务中带来显著提升。在EgoSchema上使用GPT-4o以零样本方式获得15.8%的提升，与Qwen2.5-VL-7B相比在VSI-Bench上获得8.3%的提升。

### 结论

LAST方法能有效提高视觉语言模型对3D空间和长视频的理解能力，通过空间和时间维度的视觉思维轨迹，在多种任务和场景中均表现出色。

### 翻译

人类能够通过连续的视觉观察感知和理解3D空间和长视频。但是视觉语言模型能做到这一点吗？最近的研究表明，即使是最先进的VLMs在理解3D空间和长视频方面仍然存在困难，尽管它们在典型的视觉语言任务中表现出色。当前的方法通常依赖于专门的架构设计来分别提高3D任务和视频理解任务的性能。相比之下，我们提出了LAST（LeArn to Think in Space and Time），仅使用一组2D图像作为输入，共同提高通用VLMs对3D空间和长视频的理解。LAST让VLMs在给出最终答案前在空间和时间上思考，而不是仅依靠文本，从而在3D空间和时间维度上构建视觉思维轨迹。我们在两种场景中证明了LAST的有效性：1）零样本场景，我们直接提示专有模型；2）使用包含3D空间和时间思维轨迹的数据微调通用VLMs。我们展示了LAST在各种基准测试中的显著提升。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决视觉语言模型(VLMs)在3D空间理解和长视频理解方面的不足。这一问题很重要，因为3D空间理解与机器人技术和自动驾驶相关，而长视频理解对于处理连续视觉信息至关重要，解决这些问题可以推动通用VLMs更好地理解和推理复杂的视觉世界。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者重新审视了VLMs的推理方式，发现文本-only的链式思维(CoT)在空间和时间理解上失败，因为它忽略了丰富的视觉世界。作者借鉴了CoT的思想但扩展到视觉领域，使用外部工具生成视觉标记，设计了两种应用方式：zero-shot(提示专有模型)和fine-tuning(微调开源模型)。方法借鉴了现有的工具如SAM2用于对象跟踪，Grounding-DINO用于图像定位等。', '这个方法的核心思想是什么？整体实现流程是怎样的？': "LAST的核心思想是让VLMs在空间和时间维度上'思考'，而不仅仅在文本层面思考。实现流程包括：1)使用一系列外部工具(帧选择、对象跟踪、时序定位等)帮助模型理解和推理；2)通过zero-shot直接提示专有模型或fine-tuning微调通用VLMs；3)构建包含文本思维轨迹和视觉思维轨迹的训练数据；4)在推理时结合问题、初始观察和视觉思维链得到最终答案。", '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：统一的框架同时提高3D空间和长视频理解能力；引入视觉思维链而非文本-only CoT；有效集成外部工具；支持zero-shot和fine-tuning两种应用方式；构建大规模数据集。相比之前工作，LAST不依赖3D输入，提供统一解决方案而非分别处理空间和时间，从文本思维转向视觉思维，且使用更少输入帧就能达到更好性能。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'LAST通过引入空间和时间维度的视觉思维链，显著提升了视觉语言模型在3D空间理解和长视频理解方面的能力，无需依赖特定的架构设计或额外的3D输入。'}


### 论文摘要

Humans can perceive and understand 3D space and long videos from sequential visual observations. But do vision-language models (VLMs) can? Recent work demonstrates that even state-of-the-art VLMs still struggle to understand 3D space and long videos, although they are powerful in typical vision-language tasks. Current methods often rely on specialized architectural designs to improve performance for 3D tasks and video understanding tasks separately. In contrast, we propose LAST, short for LeArn to Think in Space and Time, to jointly improve 3D spatial and long video understanding for general VLMs with only a set of 2D images as inputs. LAST makes VLMs think in space and time rather than only with text before giving the final answer, building visual thinking trajectories in 3D space and temporal dimension. We demonstrate the effectiveness of LAST in two scenarios: 1) zero-shot, where we directly prompt proprietary models; and 2) fine-tuning general VLMs with data that include thinking trajectories in 3D space and time. We show that LAST brings substantial gains in various benchmarks, including 3 spatial understanding, 4 video understanding, and 3 image understanding tasks. Notably, 15.8% gains on EgoSchema with GPT-4o in a zero-shot manner and 8.3 gains on VSI-Bench compared with Qwen2.5-VL-7B.

---

## 124. Observations of sulfuretted species in HL Tau

**论文链接:** [http://arxiv.org/abs/2511.19148v1](http://arxiv.org/abs/2511.19148v1)

**作者:** P. Rivière-Marichalar, R. le Gal, A. Fuente, D. Semenov, G. Esplugues, D. Navarro-Almaida, S. Facchini

**发布时间:** 2025-11-24

### GPT解析

### 总结

本研究通过比较HL Tau源的原行星盘和周围包层中选定分子的柱密度比，发现了显著的化学差异，表明在原行星盘形成和演化过程中气体经历了化学再处理。

### 背景

原行星盘的化学成分继承自其诞生的分子云，但这种物质在化学再处理过程中被保留和重置的程度仍然是一个开放性问题。理解这种平衡是天体化学的主要课题之一。通过比较包层和原行星盘的化学成分是解决这一问题的关键。

### 目的

研究盘与周围包层之间的化学差异，通过比较选定物种在包层和盘中的柱密度比。

### 方法

使用NOEMA观测HL Tau源中的CS、H2CO、H2S和SO2等物种，制作检测到发射的物种的零阶、一阶和二阶矩图，分析不同分子在盘和包层中的空间分布和运动学特性。推导柱密度值并比较包层和盘的值，计算SO2检测跃迁的转动能级图，假设17和58 K两种温度条件推导盘中调查物种的柱密度并与包层推导值进行比较。

### 主要发现

调查分子的推导柱密度比存在显著差异，特别是N(CS)/N(H2S)在包层中比盘中大40到50倍。这些变化归因于盘和包层中不同的激发和紫外辐照条件。不同位置的盘比率之间存在强梯度，可能与不同方位的不同湍流水平有关。

### 结论

包层和盘中分子比率的差异表明，在原行星盘的形成和演化过程中气体经历了化学再处理。

### 翻译

原行星盘的化学成分继承自其诞生的分子云，但这种物质在化学再处理过程中被保留和重置的程度仍然是一个开放性问题。理解这种平衡是天体化学的主要课题之一。比较包层和原行星盘的化学成分是解决这一问题的关键。本文的目标是通过比较选定物种在包层和盘中的柱密度比，研究盘与周围包层之间的化学差异。我们关注的源是HL Tau。我们呈现了针对CS、H2CO、H2S和SO2等物种的HL Tau的NOEMA新观测数据。我们为检测到发射的物种制作了零阶、一阶和二阶矩图，并利用它们分析了不同分子在盘和包层中的空间分布和运动学特性。我们推导了柱密度值并比较了包层和盘的推导值。我们还计算了SO2检测跃迁的转动能级图。假设两种不同的温度条件，17和58 K，我们推导了盘中调查物种的柱密度，并将其与包层推导值进行了比较。我们发现调查分子的推导柱密度比存在显著差异，特别是N(CS)/N(H2S)在包层中比盘中大40到50倍。我们将这些变化归因于盘和包层中不同的激发和紫外辐照条件。我们还注意到不同位置的盘比率之间存在强梯度，并暂时将其归因于不同方位的不同湍流水平。包层和盘中分子比率的差异表明，在原行星盘的形成和演化过程中气体经历了化学再处理。


### 论文摘要

Protoplanetary disks inherit their chemical composition from their natal molecular cloud, but the extent to which this material is preserved versus reset through chemical reprocessing remains an open question. Understanding this balance is a major topic in astrochemistry. Comparing the chemical composition of the envelope and the protoplanetary disk is key to solving the topic. The goal of this paper is to investigate the chemical differences between the disk and the surrounding envelope by comparing the column density ratios of a few selected species in each region. The source we focus on is HL Tau. We present new NOEMA observations of HL Tau targeting the following species: CS, H2CO, H2S, and SO2. We produced zeroth-, first-, and second-moment maps for the species where emission was detected and used them to analyze the spatial distribution and kinematic properties of the different molecules in the disk and the envelope. We derived the column densities and compared the values derived for the envelope and disk. We also computed the rotational diagram for the SO2 detected transitions. Assuming two different temperature regimes, 17 and 58 K, we derived column densities for the species surveyed in the disk and compared them with values derived for the envelope. We find large differences in the derived column density ratios of the surveyed molecules, especially for N(CS)/N(H2S), which is 40 to 50 times larger in the envelope. We attribute these variations to the different excitation and UV-irradiation regimes in the disk and envelope. We also note strong gradients in the ratios between different positions of the disk and tentatively attribute them to different levels of turbulence at different azimuths. The observed differences in molecular ratios in the envelope and the disk are suggestive of chemical reprocessing of the gas during the formation and evolution of the protoplanetary disk.

---

## 125. ABM-LoRA: Activation Boundary Matching for Fast Convergence in Low-Rank Adaptation

**论文链接:** [http://arxiv.org/abs/2511.19145v2](http://arxiv.org/abs/2511.19145v2)

**作者:** Dongha Lee, Jinhee Park, Minjun Kim, Junseok Kwon

**发布时间:** 2025-11-24

**备注:** 16 pages, 5 figures, under review

### GPT解析

### 总结

本文提出了一种名为ABM-LoRA的初始化策略，通过将适配器的激活边界与预训练模型对齐，显著加速了低秩适配器的收敛速度，并在多种任务上取得了优异效果。

### 背景

LoRA虽然具有高参数效率，但其随机初始化会将梯度更新限制在不匹配的切空间中，导致显著的信息损失并阻碍早期收敛。

### 目的

提出一种原则性的初始化策略，解决LoRA初始化中的信息损失问题，加速模型收敛。

### 方法

ABM-LoRA通过在下游训练前将适配器的激活边界与预训练模型对齐，最大化全参数梯度到适配器子空间的投影，从而减少初始化时的信息损失。

### 主要发现

ABM-LoRA在语言理解(T5-Base on GLUE)、对话生成(LLaMA2-7B on WizardLM)和视觉识别(ViT-B/16 on VTAB-1K)等多种架构和任务上表现出色，在VTAB-1K上实现了所有方法中最高的准确率，并在需要几何理解的推理任务上取得了显著提升。

### 结论

ABM-LoRA通过优化初始化策略有效解决了LoRA的信息损失问题，显著降低了起始损失并加速了收敛，在各种任务上均优于现有方法。

### 翻译

我们提出了用于低秩自适应的激活边界匹配(ABM-LoRA)，这是一种原则性的初始化策略，可显著加速低秩适配器的收敛。虽然LoRA提供了高参数效率，但其随机初始化将梯度更新限制在不匹配的切空间中，导致显著的信息损失并阻碍早期收敛。我们的ABM-LoRA通过在下游训练前将适配器的激活边界与预训练模型对齐，从而最大化全参数梯度到适配器子空间的投影。这种对齐在初始化时显著减少了信息损失，降低了起始损失，并加速了收敛。我们在多种架构和任务上证明了ABM-LoRA的有效性：语言理解(T5-Base on GLUE)、对话生成(LLaMA2-7B on WizardLM)和视觉识别(ViT-B/16 on VTAB-1K)。在VTAB-1K上，它实现了所有方法中最高的准确率，在需要几何理解的结构化推理任务上取得了显著提升。


### 论文摘要

We propose Activation Boundary Matching for Low-Rank Adaptation (ABM-LoRA), a principled initialization strategy that substantially accelerates the convergence of low-rank adapters. While LoRA offers high parameter efficiency, its random initialization restricts gradient updates to a mismatched tangent space, causing significant information loss and hindering early convergence. Our ABM-LoRA addresses this by aligning the adapter's activation boundaries with those of the pretrained model before downstream training, thereby maximizing the projection of full-parameter gradients into the adapter subspace. This alignment sharply reduces information loss at initialization, yields a lower starting loss, and accelerates convergence. We demonstrate ABM-LoRA's effectiveness across diverse architectures and tasks: language understanding (T5-Base on GLUE), dialogue generation (LLaMA2-7B on WizardLM), and vision recognition (ViT-B/16 on VTAB-1K). On VTAB-1K, it achieves the highest accuracy among all methods, with strong gains on structured reasoning tasks requiring geometric understanding.

---

## 126. MedSAM3: Delving into Segment Anything with Medical Concepts

**论文链接:** [http://arxiv.org/abs/2511.19046v1](http://arxiv.org/abs/2511.19046v1)

**作者:** Anglin Liu, Rundong Xue, Xu R. Cao, Yifan Shen, Yi Lu, Xiang Li, Qianqian Chen, Jintai Chen

**发布时间:** 2025-11-24

### GPT解析

### 总结

本研究提出了MedSAM-3，一种可文本提示的医学分割模型，用于医学图像和视频分割。通过微调SAM 3架构并结合语义概念标签，实现了医学提示概念分割(PCS)，允许通过文本描述精确定位解剖结构。还引入了整合多模态大语言模型的MedSAM-3 Agent框架，实验证明该方法在多种医学成像模态上优于现有模型。

### 背景

医学图像分割对生物医学发现至关重要。然而，现有方法缺乏泛化能力，且在新临床应用中需要大量耗时的手动标注。

### 目的

开发一种可文本提示的医学分割模型，减少对大量手动标注的依赖，提高模型的泛化能力，并允许通过文本描述精确定位解剖结构。

### 方法

提出MedSAM-3模型，在医学图像上微调SAM 3架构并结合语义概念标签，实现医学提示概念分割(PCS)。引入MedSAM-3 Agent框架，整合多模态大语言模型(MLLMs)进行复杂推理和迭代改进。

### 主要发现

在多种医学成像模态（包括X光、MRI、超声、CT和视频）上的综合实验表明，该方法显著优于现有的专业模型和基础模型。

### 结论

MedSAM-3通过结合文本提示能力和多模态大语言模型，显著提高了医学图像分割的准确性和泛化能力，减少了手动标注的需求。

### 翻译

医学图像分割对生物医学发现至关重要。现有方法缺乏泛化能力，且在新临床应用中需要大量耗时的手动标注。在此，我们提出了MedSAM-3，一个可文本提示的医学分割模型，用于医学图像和视频分割。通过在医学图像上微调'分割一切模型'(SAM) 3架构并结合语义概念标签，我们的MedSAM-3实现了医学提示概念分割(PCS)，允许通过开放词汇文本描述而非仅几何提示来精确定位解剖结构。我们进一步引入了MedSAM-3 Agent，这是一个整合多模态大语言模型(MLLMs)的框架，在循环工作流中执行复杂推理和迭代改进。在多种医学成像模态上的综合实验表明，我们的方法显著优于现有的专业模型和基础模型。我们将在https://github.com/Joey-S-Liu/MedSAM3上发布我们的代码和模型。


### 论文摘要

Medical image segmentation is fundamental for biomedical discovery. Existing methods lack generalizability and demand extensive, time-consuming manual annotation for new clinical application. Here, we propose MedSAM-3, a text promptable medical segmentation model for medical image and video segmentation. By fine-tuning the Segment Anything Model (SAM) 3 architecture on medical images paired with semantic conceptual labels, our MedSAM-3 enables medical Promptable Concept Segmentation (PCS), allowing precise targeting of anatomical structures via open-vocabulary text descriptions rather than solely geometric prompts. We further introduce the MedSAM-3 Agent, a framework that integrates Multimodal Large Language Models (MLLMs) to perform complex reasoning and iterative refinement in an agent-in-the-loop workflow. Comprehensive experiments across diverse medical imaging modalities, including X-ray, MRI, Ultrasound, CT, and video, demonstrate that our approach significantly outperforms existing specialist and foundation models. We will release our code and model at https://github.com/Joey-S-Liu/MedSAM3.

---

## 127. 论文ID: 2511.19005v1

**错误:** 无法读取论文信息 - [Errno 2] No such file or directory: 'storage/result/2511.19005v1.json'

---

## 128. EventSTU: Event-Guided Efficient Spatio-Temporal Understanding for Video Large Language Models

**论文链接:** [http://arxiv.org/abs/2511.18920v1](http://arxiv.org/abs/2511.18920v1)

**作者:** Wenhao Xu, Xin Dong, Yue Li, Haoyuan Shi, Zhiwei Xiong

**发布时间:** 2025-11-24

**备注:** 8 pages, 7 figures

### GPT解析

### 总结

提出EventSTU框架，通过事件引导实现高效时空理解，显著降低推理成本同时提高性能。

### 背景

视频大语言模型具有强大的视频理解能力，但由于长视频中标记数量庞大，导致推理成本高昂。

### 目的

开发一个高效时空理解框架，解决视频大模型推理成本高的问题。

### 方法

提出EventSTU事件引导、无需训练的框架：时间域设计从粗到细的关键帧采样算法，利用事件相机变化触发特性消除冗余帧；空间域设计自适应标记修剪算法，利用事件视觉显著性作为零成本先验；整体时空视角整合问题相关性自适应分配标记预算；构建EventBench首个包含事件、人工注释的多模态基准；支持物理事件相机和模拟事件的通用视频理解。

### 主要发现

EventSTU实现比最强基线高3.01倍的FLOPs减少，3.10倍的预填充加速，同时提高了性能。

### 结论

EventSTU框架能在大幅降低计算成本的同时提升视频理解性能，为视频大模型的高效推理提供了新思路。

### 翻译

视频大语言模型展示了强大的视频理解能力，但由于长视频中标记数量庞大，导致推理成本高昂。受事件视觉启发，我们提出了一种事件引导的无需训练的高效时空理解框架，名为EventSTU。在时间域，我们设计了一种从粗到细的关键帧采样算法，利用事件相机的变化触发特性消除冗余帧。在空间域，我们设计了一种自适应标记修剪算法，利用事件的视觉显著性作为零成本先验来引导空间缩减。从整体时空视角，我们进一步整合了关键帧采样中的问题相关性，以自适应分配标记修剪预算。为便于评估，我们构建了EventBench，这是第一个包含事件、人工注释的多模态基准，涵盖多样化的真实场景。除了物理事件相机外，EventSTU还支持使用模拟事件进行通用视频理解。全面实验表明，EventSTU比最强基线实现了3.01倍的FLOPs减少和3.10倍的预填充加速，同时仍提高了性能。


### 论文摘要

Video large language models have demonstrated strong video understanding capabilities but suffer from high inference costs due to the massive number of tokens in long videos. Inspired by event-based vision, we propose an event-guided, training-free framework for efficient spatio-temporal understanding, named EventSTU. In the temporal domain, we design a coarse-to-fine keyframe sampling algorithm that exploits the change-triggered property of event cameras to eliminate redundant frames. In the spatial domain, we design an adaptive token pruning algorithm that leverages the visual saliency of events as a zero-cost prior to guide spatial reduction. From a holistic spatio-temporal perspective, we further integrate question relevance from keyframe sampling to adaptively allocate token pruning budgets. To facilitate evaluation, we construct EventBench, the first event-inclusive, human-annotated multimodal benchmark that covers diverse real-world scenarios. Beyond physical event cameras, EventSTU also supports general video understanding using simulated events. Comprehensive experiments show that EventSTU achieves 3.01x FLOPs reduction and 3.10x prefilling speedup over the strongest baseline while still improving performance.

---

## 129. Improved constraints on ultralight axions using latest observations of the early and late Universe

**论文链接:** [http://arxiv.org/abs/2511.18917v1](http://arxiv.org/abs/2511.18917v1)

**作者:** Qianshuo Liu, Chang Feng, Filipe B. Abdalla

**发布时间:** 2025-11-24

**备注:** 9 pages, 6 figures

### GPT解析

### 总结

本研究利用宇宙微波背景(CMB)和重子声学振荡(BAO)观测数据，对超轻轴子(ULAs)作为暗物质或暗能量的可能性进行了约束分析，获得了ULAs与总暗物质能量密度分数比的新上限。

### 背景

超轻轴子是假设粒子，可根据对称破缺尺度产生的质量行为像暗物质或暗能量。理论预测表明ULAs可能在宇宙信号上留下独特印记，但这些印记可能存在于广泛的空间和时间尺度上，并与标准模型已知效应简并，使得ULAs的观测证据难以捉摸。

### 目的

利用早期和晚期宇宙观测数据推断ULAs的性质，并约束其能量密度分数比(Ωa/Ωd)。

### 方法

使用CMB和BAO模拟数据验证ULAs效应建模，进行交叉检查测试。分析Planck 2018 CMB测量数据和DESI数据发布2的BAO测量数据。

### 主要发现

获得了ULAs与总暗物质能量密度分数比(Ωa/Ωd)的新上限。

### 结论

未来的CMB和BAO测量将实现前所未有的精度，对于理解ULAs的本质至关重要。

### 翻译

超轻轴子(ULAs)是假设粒子，可以根据对称破缺尺度产生的质量行为像暗物质(DM)或暗能量(DE)。ULAs能否构成暗物质或暗能量的一部分仍然是个谜。尽管理论预测表明ULAs可能在宇宙信号上留下独特印记，但这些印记可能存在于广泛的空间和时间尺度上，并且可能与标准模型已知效应简并。ULAs的印记极其微妙，其观测证据仍然难以捉摸。在本工作中，我们使用来自宇宙微波背景(CMB)和重子声学振荡(BAO)的早期和晚期宇宙观测数据来推断ULAs的性质。我们使用CMB和BAO模拟数据验证了ULAs效应建模，并进行不同测试以交叉检查结果。通过分析Planck 2018 CMB测量数据和DESI数据发布2的BAO测量数据，我们约束了ULAs与总暗物质的能量密度分数比Ωa/Ωd，并获得了一个新的上限。未来的CMB和BAO测量将实现前所未有的精度，对于理解ULAs的本质至关重要。


### 论文摘要

Ultralight axions (ULAs) are hypothetical particles which can behave like dark matter (DM) or dark energy (DE) depending on masses generated at the symmetry-breaking scale. It remains a mystery whether the ULAs can make up a fraction of DM or DE. Although theoretical predictions indicate that the ULAs may leave distinct imprints on cosmological signals, these signatures may exist in a broad spatial and temporal scales, and may be degenerate with the known effects of the standard model. The ULA signatures are extremely subtle and the observational evidence of the ULAs remain elusive. In this work, we infer the ULA properties using both the early and late universe observations from the cosmic microwave background (CMB) and baryon acoustic oscillations (BAO). We validate modeling of the ULA effects using the CMB and BAO mock data and perform different tests to cross-check the results. By analyzing the Planck 2018 CMB measurements and the BAO measurements from the Data Release 2 of Dark Energy Spectroscopic Instrument (DESI), we constrain the energy density fraction ratio of the ULAs to total dark matter $Ω_a/Ω_d$ and obtain a new upper bound of $Ω_a/Ω_d$. Future CMB and BAO measurements will achieve unprecedented precision and will be crucial for understanding the nature of the ULAs.

---

## 130. GContextFormer: A global context-aware hybrid multi-head attention approach with scaled additive aggregation for multimodal trajectory prediction

**论文链接:** [http://arxiv.org/abs/2511.18874v1](http://arxiv.org/abs/2511.18874v1)

**作者:** Yuzhi Chen, Yuanchang Xie, Lei Zhao, Pan Liu, Yajie Zou, Chen Wang

**发布时间:** 2025-11-24

### GPT解析

### 总结

该论文提出GContextFormer，一种不依赖地图的多模态轨迹预测模型，通过全局上下文感知的混合注意力和缩放加性聚合技术，解决了HD地图依赖模型的缺陷和地图免费方法缺乏全局上下文的问题，实现了意图对齐的多模态预测。

### 背景

多模态轨迹预测通过生成多种可能的未来轨迹来解决车辆运动的不确定性。然而，HD地图依赖模型存在数据获取成本高、更新延迟、对损坏输入脆弱等问题；地图免费方法则缺乏全局上下文，导致注意力机制过度放大直线模式而抑制过渡模式，造成运动意图不匹配。

### 目的

开发一种不依赖地图的、意图对齐的多模态预测模型，解决现有HD地图依赖模型和地图免费方法的局限性。

### 方法

提出GContextFormer架构，包含运动感知编码器和分层交互解码器。编码器通过模式嵌入轨迹令牌上的有界缩放加性聚合构建场景级意图先验；解码器将社交推理分解为标准路径和邻居上下文增强路径的双路径交叉注意力机制，通过门控模块保持覆盖-焦点平衡。

### 主要发现

在TOD-VT数据集的八个高速公路-匝道场景中，GContextFormer优于最先进的基线模型；相比现有Transformer模型，GContextFormer在空间分布上实现了更高鲁棒性，并在高曲率和过渡区域取得了显著改进；通过运动模式区分和邻居上下文调制实现了模型可解释性。

### 结论

GContextFormer成功解决了HD地图依赖模型和地图免费方法的局限性，实现了意图对齐的多模态预测，其模块化架构支持向跨域多模态推理任务的扩展。

### 翻译

多模态轨迹预测生成多种合理的未来轨迹，以解决因意图模糊和执行变异性导致的车辆运动不确定性。然而，依赖高精度地图的模型面临数据获取成本高、更新延迟以及对损坏输入敏感等问题，导致预测失败。无地图方法缺乏全局上下文，成对注意力机制过度放大直线模式而抑制过渡模式，造成运动意图不匹配。本文提出GContextFormer，一种即插即用的编码器-解码器架构，具有全局上下文感知的混合注意力和缩放加性聚合，实现了不依赖地图的意图对齐多模态预测。运动感知编码器通过模式嵌入轨迹令牌上的有界缩放加性聚合构建场景级意图先验，并在共享全局上下文下细化每个模式的表示，减轻了模式间的抑制并促进意图对齐。分层交互解码器将社交推理分解为双路径交叉注意力：标准路径确保在代理-模式对上的均匀几何覆盖，而邻居上下文增强路径强调显著交互，门控模块调节它们的贡献以保持覆盖-焦点平衡。在TOD-VT数据集的八个高速公路-匝道场景中的实验表明，GContextFormer优于最先进的基线模型。与现有Transformer模型相比，GContextFormer通过空间分布实现了更高的鲁棒性，并在高曲率和过渡区域取得了集中的改进。通过运动模式区分和邻居上下文调制实现了可解释性，暴露了推理归因。模块化架构支持向跨域多模态推理任务的扩展性。


### 论文摘要

Multimodal trajectory prediction generates multiple plausible future trajectories to address vehicle motion uncertainty from intention ambiguity and execution variability. However, HD map-dependent models suffer from costly data acquisition, delayed updates, and vulnerability to corrupted inputs, causing prediction failures. Map-free approaches lack global context, with pairwise attention over-amplifying straight patterns while suppressing transitional patterns, resulting in motion-intention misalignment. This paper proposes GContextFormer, a plug-and-play encoder-decoder architecture with global context-aware hybrid attention and scaled additive aggregation achieving intention-aligned multimodal prediction without map reliance. The Motion-Aware Encoder builds scene-level intention prior via bounded scaled additive aggregation over mode-embedded trajectory tokens and refines per-mode representations under shared global context, mitigating inter-mode suppression and promoting intention alignment. The Hierarchical Interaction Decoder decomposes social reasoning into dual-pathway cross-attention: a standard pathway ensures uniform geometric coverage over agent-mode pairs while a neighbor-context-enhanced pathway emphasizes salient interactions, with gating module mediating their contributions to maintain coverage-focus balance. Experiments on eight highway-ramp scenarios from TOD-VT dataset show GContextFormer outperforms state-of-the-art baselines. Compared to existing transformer models, GContextFormer achieves greater robustness and concentrated improvements in high-curvature and transition zones via spatial distributions. Interpretability is achieved through motion mode distinctions and neighbor context modulation exposing reasoning attribution. The modular architecture supports extensibility toward cross-domain multimodal reasoning tasks. Source: https://fenghy-chen.github.io/sources/.

---

## 131. E2E-GRec: An End-to-End Joint Training Framework for Graph Neural Networks and Recommender Systems

**论文链接:** [http://arxiv.org/abs/2511.20564v1](http://arxiv.org/abs/2511.20564v1)

**作者:** Rui Xue, Shichao Zhu, Liang Qin, Guangmou Pan, Yang Song, Tianfu Wu

**发布时间:** 2025-11-25

### GPT解析

### 总结

这篇论文提出了E2E-GRec，一种新的端到端训练框架，用于统一图神经网络(GNN)训练与推荐系统，解决了传统两阶段部署中的计算开销和联合优化问题。

### 背景

图神经网络已成为建模图结构数据和推荐系统的强大工具，但大多数工业部署采用两阶段流程：GNN离线预训练生成节点嵌入，然后作为静态特征用于下游推荐系统。

### 目的

解决当前GNN在推荐系统中部署的两个关键限制：(1)高计算开销，需要重复执行大规模GNN推理；(2)缺乏联合优化，推荐系统梯度无法直接影响GNN学习。

### 方法

提出E2E-GRec框架，包含三个关键组件：(i)从大规模跨领域异构图中高效子图采样；(ii)图特征自编码器作为辅助自监督任务；(iii)两级特征融合机制结合基于Gradnorm的动态损失平衡。

### 主要发现

在大规模生产数据上的离线评估和在线A/B测试显示，E2E-GRec实现了用户停留时间相对提升0.133%，用户平均跳过视频数减少0.3171%，在多个推荐指标上显著优于传统方法。

### 结论

E2E-GRec通过端到端训练框架有效解决了传统GNN推荐系统部署中的计算和优化问题，实现了推荐效果的显著提升。

### 翻译

图神经网络已成为建模图结构数据的强大工具，并被广泛应用于推荐系统，例如捕捉复杂的用户-物品和物品-物品关系。然而，大多数工业部署采用两阶段流程：首先离线预训练GNN生成节点嵌入，然后将其用作下游推荐系统的静态特征。这种解耦范式导致两个关键限制：(1)高计算开销，因为必须重复执行大规模GNN推理来刷新嵌入；(2)缺乏联合优化，因为推荐系统的梯度无法直接影响GNN学习过程，导致GNN对推荐任务的信息性次优。在本文中，我们提出了E2E-GRec，一种新的端到端训练框架，统一了GNN训练与推荐系统。我们的框架具有三个关键特征：(i)从大规模跨领域异构图中高效子图采样，确保训练的可扩展性和效率；(ii)图特征自编码器作为辅助自监督任务，引导GNN学习结构上有意义的嵌入；(iii)两级特征融合机制结合基于Gradnorm的动态损失平衡，稳定了图感知的多任务端到端训练。大规模生产数据上的广泛离线评估、在线A/B测试以及理论分析证明，E2E-GRec始终优于传统方法，在多个推荐指标上取得显著提升。


### 论文摘要

Graph Neural Networks (GNNs) have emerged as powerful tools for modeling graph-structured data and have been widely used in recommender systems, such as for capturing complex user-item and item-item relations. However, most industrial deployments adopt a two-stage pipeline: GNNs are first pre-trained offline to generate node embeddings, which are then used as static features for downstream recommender systems. This decoupled paradigm leads to two key limitations: (1) high computational overhead, since large-scale GNN inference must be repeatedly executed to refresh embeddings; and (2) lack of joint optimization, as the gradient from the recommender system cannot directly influence the GNN learning process, causing the GNN to be suboptimally informative for the recommendation task. In this paper, we propose E2E-GRec, a novel end-to-end training framework that unifies GNN training with the recommender system. Our framework is characterized by three key components: (i) efficient subgraph sampling from a large-scale cross-domain heterogeneous graph to ensure training scalability and efficiency; (ii) a Graph Feature Auto-Encoder (GFAE) serving as an auxiliary self-supervised task to guide the GNN to learn structurally meaningful embeddings; and (iii) a two-level feature fusion mechanism combined with Gradnorm-based dynamic loss balancing, which stabilizes graph-aware multi-task end-to-end training. Extensive offline evaluations, online A/B tests (e.g., a +0.133% relative improvement in stay duration, a 0.3171% reduction in the average number of videos a user skips) on large-scale production data, together with theoretical analysis, demonstrate that E2E-GRec consistently surpasses traditional approaches, yielding significant gains across multiple recommendation metrics.

---

## 132. Estimating the triaxiality of massive clusters from 2D observables in MillenniumTNG with machine learning

**论文链接:** [http://arxiv.org/abs/2511.20429v1](http://arxiv.org/abs/2511.20429v1)

**作者:** Ana Maria Delgado, Michelle Ntampaka, Sownak Bose, Fulvio Ferlito, Boryana Hadzhiyska, Lars Hernquist, John Soltis, John F. Wu, Mikaeel Yunus, John ZuHone

**发布时间:** 2025-11-25

### GPT解析

### 总结

本研究提出了一种深度学习方法，用于从二维观测数据估计大质量星系团的三轴性和方向，结合了卷积神经网络和图神经网络的优势，提高了星系团几何特性估计的准确性。

### 背景

大质量星系团的特性（如质量丰度和集中度）对宇宙学敏感，使星系团统计成为宇宙学研究的强大工具。然而，采用简化的球对称模型会导致星系团特性估计的偏差。

### 目的

开发一种深度学习方法，从二维观测数据估计大质量星系团（质量约大于10^14 M⊙ h^-1）的三轴性和方向。

### 方法

使用MillenniumTNG模拟套件作为真实数据，构建多模态融合网络，结合卷积神经网络的特征提取能力和图神经网络的消息传递能力，从二维多波段图像和星系团成员的数学图表示中提取三维几何信息。

### 主要发现

与假设球对称性相比，该方法将星系团几何估计提高了30%；三轴星系团主轴长度估计的回归得分为R^2 = 0.85；正确分类了71%的沿视线方向拉长的长椭球星系团。

### 结论

深度学习方法能够有效从二维观测数据估计星系团的三维几何特性，克服了传统球对称模型的局限性，提高了星系团特性估计的准确性。

### 翻译

大质量星系团的特性，如质量丰度和集中度，对宇宙学敏感，这使得星系团统计成为宇宙学研究的强大工具。然而，对星系团采用更简化的球对称模型会导致特性估计的偏差。在这项工作中，我们提出了一种深度学习方法，用于从二维观测数据估计大质量星系团（质量约大于10^14 M⊙ h^-1）的三轴性和方向。我们使用宇宙学-流体动力学MillenniumTNG模拟套件的主流体动力学体积作为真实数据。我们的模型将卷积神经网络的特征提取能力和图神经网络的消息传递能力结合在一个多模态融合网络中。我们的模型能够从二维理想化的星系团多波段图像（软X射线、中等X射线、硬X射线和tSZ效应）和星系团成员的数学图表示（视线径向速度、二维投影位置和V波段亮度）中提取三维几何信息。与假设球对称性相比，我们的网络将MTNG中的星系团几何估计提高了30%。我们报告了三轴星系团主轴长度估计的R^2 = 0.85回归得分，并正确分类了71%的沿我们视线方向拉长的长椭球星系团。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决如何从二维观测数据估计大质量星系团的三轴性和方向问题。这个问题很重要，因为目前宇宙学分析中通常将星系团简化为球对称模型，但这种假设会忽略星系团复杂的真实形态，导致质量估计出现约20%的偏差和约35%的散射，进而影响宇宙学参数测量的准确性。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先认识到球对称假设会引入系统性误差，然后考虑利用机器学习捕捉星系团结构的细微差别。他们设计了混合神经网络架构，结合卷积神经网络(CNN)处理多波段图像数据(软X射线、中X射线、硬X射线和tSZ效应)和图神经网络(GNN)处理星系团成员的图结构数据(位置、速度和光度)。该方法借鉴了IllustrisTNG的物理模型、Ntampaka等人的X射线观测方法、Wu & Kragh Jespersen的图神经网络设计以及Larson等人结合CNN和GNN的工作。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是利用深度学习从二维观测数据中推断星系团的三维几何结构，通过结合多波段观测数据和星系团成员的图结构数据来提高三轴性估计的准确性。整体流程包括：1)从MTNG模拟中获取星系团数据并创建多波段图像和成员图；2)构建混合神经网络架构，CNN分支处理图像数据，GNN分支处理图数据；3)在融合层结合两个分支的特征；4)通过任务特定MLP进行回归(估计半轴长度和形状参数)和分类(预测主轴方向)；5)使用混合损失函数和自定义学习率调度进行训练。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': "关键创新点包括：1)首创结合CNN和GNN的混合神经网络架构来估计星系团三轴性；2)同时利用多种波段观测数据和星系团成员图结构数据；3)特别关注并能够正确识别71%的'假球体'星系团(观测中呈球形但实际沿视线方向拉长)；4)相比球对称假设，几何估计提高30%。相比之前工作，本文不采用传统的贝叶斯框架，而是直接用深度学习从观测数据预测三轴形状；同时利用多种数据类型而非单一观测数据；能处理各种形态和方向的星系团而非局限于特定类型。", '如果要用一句话总结这篇论文的贡献，你会怎么说？': '这篇论文开发了一种创新的混合深度学习方法，能够从二维观测数据中准确估计大质量星系团的三轴几何结构，为精确宇宙学研究提供了更可靠的星系团属性估计工具。'}


### 论文摘要

Properties of massive galaxy clusters, such as mass abundance and concentration, are sensitive to cosmology, making cluster statistics a powerful tool for cosmological studies. However, favoring a more simplified, spherically symmetric model for galaxy clusters can lead to biases in the estimates of cluster properties. In this work, we present a deep-learning approach for estimating the triaxiality and orientations of massive galaxy clusters (those with masses $\gtrsim 10^{14}\,M_\odot h^{-1}$) from 2D observables. We utilize the flagship hydrodynamical volume of the suite of cosmological-hydrodynamical MillenniumTNG (MTNG) simulations as our ground truth. Our model combines the feature extracting power of a convolutional neural network (CNN) and the message passing power of a graph neural network (GNN) in a multi-modal, fusion network. Our model is able to extract 3D geometry information from 2D idealized cluster multi-wavelength images (soft X-ray, medium X-ray, hard X-ray and tSZ effect) and mathematical graph representations of 2D cluster member observables (line-of-sight radial velocities, 2D projected positions and V-band luminosities). Our network improves cluster geometry estimation in MTNG by $30\%$ compared to assuming spherical symmetry. We report an $R^2 = 0.85$ regression score for estimating the major axis length of triaxial clusters and correctly classifying $71\%$ of prolate clusters with elongated orientations along our line-of-sight.

---

## 133. Short-Range Oversquashing

**论文链接:** [http://arxiv.org/abs/2511.20406v1](http://arxiv.org/abs/2511.20406v1)

**作者:** Yaaqov Mishayev, Yonatan Sverdlov, Tal Amir, Nadav Dym

**发布时间:** 2025-11-25

**备注:** Accepted to Learning on Graphs (LoG) 2025. Version identical to the camera-ready paper

### GPT解析

### 总结

本研究探讨了图神经网络中的过度挤压现象，发现它不仅存在于长距离任务中，也存在于短距离问题中，并揭示了两种不同的机制。

### 背景

信息传递神经网络(MPNNs)被广泛用于图学习，但它们处理长距离信息的能力受到过度挤压现象的限制。

### 目的

探究过度挤压现象的本质，区分其背后的不同机制，并评估不同解决方案的有效性。

### 方法

通过分析短距离和长距离任务中的过度挤压现象，分离出瓶颈现象和梯度消失现象两种机制，并比较MPNNs与图变换器的性能。

### 主要发现

1) 过度挤压不仅限于长距离任务，也存在于短距离问题中；2) 过度挤压有两种不同机制：瓶颈现象和梯度消失现象；3) 现有的过度挤压解释未捕捉到短距离瓶颈效应；4) 添加虚拟节点无法解决短距离瓶颈效应；5) 图变换器在这些任务中表现优于MPNNs。

### 结论

图变换器是解决过度挤压问题比专门MPNNs更有说服力的解决方案，因为它们能够有效处理短距离和长距离任务中的过度挤压现象。

### 翻译

信息传递神经网络(MPNNs)被广泛用于图上的学习，但它们处理长距离信息的能力受到过度挤压现象的限制。这一限制导致一些研究人员提倡图变换器作为更好的替代方案，而另一些人认为可以通过虚拟节点或其他重布线技术在MPNN框架内缓解这一问题。在本工作中，我们证明过度挤压不仅限于长距离任务，也可能在短距离问题中出现。这一观察使我们能够分离出过度挤压背后的两种不同机制：(1)瓶颈现象，即使在低距离设置中也可能出现，以及(2)梯度消失现象，与长距离任务密切相关。我们进一步表明，现有的过度挤压解释并未捕捉到短距离瓶颈效应，添加虚拟节点也无法解决它。相比之下，变换器在这些任务中表现出色，使其成为比专门MPNNs更有说服力的解决方案。


### 论文摘要

Message Passing Neural Networks (MPNNs) are widely used for learning on graphs, but their ability to process long-range information is limited by the phenomenon of oversquashing. This limitation has led some researchers to advocate Graph Transformers as a better alternative, whereas others suggest that it can be mitigated within the MPNN framework, using virtual nodes or other rewiring techniques.   In this work, we demonstrate that oversquashing is not limited to long-range tasks, but can also arise in short-range problems. This observation allows us to disentangle two distinct mechanisms underlying oversquashing: (1) the bottleneck phenomenon, which can arise even in low-range settings, and (2) the vanishing gradient phenomenon, which is closely associated with long-range tasks.   We further show that the short-range bottleneck effect is not captured by existing explanations for oversquashing, and that adding virtual nodes does not resolve it. In contrast, transformers do succeed in such tasks, positioning them as the more compelling solution to oversquashing, compared to specialized MPNNs.

---

## 134. Accelerating Time-Optimal Trajectory Planning for Connected and Automated Vehicles with Graph Neural Networks

**论文链接:** [http://arxiv.org/abs/2511.20383v1](http://arxiv.org/abs/2511.20383v1)

**作者:** Viet-Anh Le, Andreas A. Malikopoulos

**发布时间:** 2025-11-25

**备注:** submitted to IFAC WC 2026

### GPT解析

### 总结

本文提出了一种基于学习的框架，利用图神经网络加速连接和自动化车辆的时间和能量最优轨迹规划。

### 背景

在交通场景中，多智能体协调问题是一个挑战，需要解决多个车辆的轨迹规划问题。

### 目的

开发一个能够实时执行的、多智能体重规划方案，以最小化出行时间，同时满足运动约束和安全约束。

### 方法

将多智能体协调问题表述为协作轨迹规划问题；使用基于能量的最优解推导运动基元；在每个时间步进行重新规划；采用GNN架构从离线数据中学习时间最优轨迹规划问题的解；训练后的模型产生在线预测作为数值优化的热启动解。

### 主要发现

学习增强的方法显著减少了计算时间；确保所有状态、输入和安全约束都得到满足；能够快速计算最小退出时间和相关可行轨迹；通过热启动解决方案实现了实时性能。

### 结论

该学习增强的方法结合了图神经网络与数值优化，在保证约束满足的同时，大幅提高了计算效率，为连接和自动化车辆提供了实时可行的轨迹规划解决方案。

### 翻译

在本文中，我们提出了一种基于学习的框架，利用图神经网络加速连接和自动化车辆的时间和能量最优轨迹规划。我们将交通场景中遇到的多智能体协调问题表述为一个协作轨迹规划问题，该问题最小化出行时间，并受限于从能量最优解推导出的运动基元。通过在每个时间步进行重新规划，可以进一步提高该框架的有效性，使系统能够整合新观察到的信息。为了实现这种多智能体重规划方案的实时执行，我们采用GNN架构从离线生成的数据中学习时间最优轨迹规划问题的解。训练后的模型产生在线预测，这些预测作为数值优化的热启动解，从而能够快速计算最小退出时间及相关可行轨迹。这种学习增强的方法显著减少了计算时间，同时确保满足所有状态、输入和安全约束。


### 论文摘要

In this paper, we present a learning-based framework that accelerates time- and energy-optimal trajectory planning for connected and automated vehicles (CAVs) using graph neural networks (GNNs). We formulate the multi-agent coordination problem encountered in traffic scenarios as a cooperative trajectory planning problem that minimizes travel time, subject to motion primitives derived from energy-optimal solutions. The effectiveness of this framework can be further improved through replanning at each time step, enabling the system to incorporate newly observed information. To achieve real-time execution of such a multi-agent replanning scheme, we employ a GNN architecture to learn the solutions of the time-optimal trajectory planning problem from offline-generated data. The trained model produces online predictions that serve as warm-start solutions for numerical optimization, thereby enabling rapid computation of minimal exit times and the associated feasible trajectories. This learning-augmented approach substantially reduces computation time while ensuring that all state, input, and safety constraints are satisfied.

---

## 135. PRISM: Periodic Representation with multIscale and Similarity graph Modelling for enhanced crystal structure property prediction

**论文链接:** [http://arxiv.org/abs/2511.20362v1](http://arxiv.org/abs/2511.20362v1)

**作者:** Àlex Solé, Albert Mosella-Montoro, Joan Cardona, Daniel Aravena, Silvia Gómez-Coca, Eliseo Ruiz, Javier Ruiz-Hidalgo

**发布时间:** 2025-11-25

### GPT解析

### 总结

这篇论文介绍了PRISM，一种用于晶体结构表示学习的图神经网络框架，通过专家模块整合多尺度表示和周期性特征编码，显著提高了晶体性质预测的准确性。

### 背景

晶体结构在三维空间中的单位晶胞内有重复的原子模式，这为基于图的表示学习带来独特挑战。当前方法经常忽视晶体结构中固有的必要周期性边界条件和多尺度相互作用。

### 目的

开发一种能够有效处理晶体结构中周期性边界条件和多尺度相互作用的图神经网络框架，以提高晶体性质预测的准确性。

### 方法

作者提出了PRISM图神经网络框架，通过采用一组专家模块来显式整合多尺度表示和周期性特征编码，每个专家模块专门用于编码周期系统的不同结构和化学方面。

### 主要发现

在基于晶体结构的广泛基准实验中，PRISM提高了最先进的预测精度，显著增强了晶体性质预测能力。

### 结论

PRIS图神经网络框架能够有效处理晶体结构的周期性和多尺度特性，在晶体性质预测任务上取得了优于现有方法的性能。

### 翻译

晶体结构的特点是在三维空间中的单位晶胞内有重复的原子模式，这为基于图的表示学习带来了独特的挑战。当前方法经常忽视晶体结构中固有的必要周期性边界条件和多尺度相互作用。在本文中，我们介绍了PRISM，一种图神经网络框架，通过采用一组专家模块来显式整合多尺度表示和周期性特征编码，每个专家模块专门用于编码周期系统的不同结构和化学方面。在基于晶体结构的广泛基准实验中，PRISM提高了最先进的预测精度，显著增强了晶体性质预测。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决晶体结构性质预测的准确性问题。当前基于图表示学习的方法往往忽略了晶体结构中固有的周期性边界条件和多尺度相互作用。这个问题很重要，因为准确预测晶体材料性质对加速新型材料（如用于能源存储、催化和电子设备）的发现至关重要，而虽然密度泛函理论可靠但计算成本高，限制了大规模应用。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者通过识别现有方法的局限性来设计新方法：当前方法没有明确编码单元胞，难以捕捉长距离相互作用，且在特征空间中缺乏周期性编码。作者借鉴了图神经网络在晶体结构建模中的应用，如CGCNN、MEGNET等早期方法，以及基于特征空间的图方法如DGCNN。在此基础上，作者设计了四个专门的专家模块，并引入专家混合机制来整合不同尺度的信息。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是使用多个专门的图神经网络专家模块，每个模块专注于编码周期系统的不同结构和化学方面，通过显式整合多尺度表示和周期性特征编码来提高预测准确性。整体流程：1)将晶体表示为原子节点图；2)初始化原子特征嵌入；3)通过四个专家模块处理（原子级专家捕获短程相互作用，相似性专家连接相似原子，胞空间专家处理单元胞间长程作用，多尺度专家实现双向信息流）；4)融合专家输出；5)使用最终嵌入预测性质。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点：1)单元胞结构编码，通过添加单元胞节点提供全局视角；2)周期性特征编码，在特征空间中显式整合周期性边界条件；3)多尺度融合，整合单元胞与原子级信息；4)专家混合机制，学习属性相关的权重组合不同专家。相比之前工作，PRISM显式处理周期性，同时建模多尺度相互作用，使用专家混合架构而非单一模型，并引入超原子概念处理单元胞间长程作用，确保对等效单元胞变换的不变性。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'PRISM通过引入多专家图神经网络框架，显式整合多尺度表示和周期性特征编码，显著提高了晶体结构性质预测的准确性，在多个基准测试上实现了最先进的性能。'}


### 论文摘要

Crystal structures are characterised by repeating atomic patterns within unit cells across three-dimensional space, posing unique challenges for graph-based representation learning. Current methods often overlook essential periodic boundary conditions and multiscale interactions inherent to crystalline structures. In this paper, we introduce PRISM, a graph neural network framework that explicitly integrates multiscale representations and periodic feature encoding by employing a set of expert modules, each specialised in encoding distinct structural and chemical aspects of periodic systems. Extensive experiments across crystal structure-based benchmarks demonstrate that PRISM improves state-of-the-art predictive accuracy, significantly enhancing crystal property prediction.

---

## 136. RIS-Assisted Downlink Pinching-Antenna Systems: GNN-Enabled Optimization Approaches

**论文链接:** [http://arxiv.org/abs/2511.20305v1](http://arxiv.org/abs/2511.20305v1)

**作者:** Changpeng He, Yang Lu, Yanqing Xu, Chong-Yung Chi, Bo Ai, Arumugam Nallanathan

**发布时间:** 2025-11-25

### GPT解析

### 总结

本文研究了可重构智能表面辅助的多波导挤压天线系统在多用户下行链路信息传输中的应用，提出了一种基于图神经网络的三阶段优化方法。

### 背景

新兴的PASS和RIS集成对无线通信的未知影响是本研究的主要动机。

### 目的

在一个统一的框架中制定求和速率和能量效率最大化问题，受到PA可移动区域、总功率预算和RIS元件可调相位的约束。

### 方法

利用RIS辅助PASS的图结构拓扑，提出了一种三阶段图神经网络，根据用户位置学习PA位置，根据复合信道条件学习RIS相移，最后确定波束成形向量。该GNN通过无监督训练实现，并结合三种与凸优集成的实现策略。

### 主要发现

所提出的GNN具有有效的泛化能力、良好的性能可靠性和实时适用性，关键参数对RIS辅助PASS的性能有显著影响。

### 结论

RIS辅助的PASS系统结合GNN方法可以有效解决多用户下行链路信息传输问题，在性能和效率方面具有优势。

### 翻译

本文研究了可重构智能表面辅助的多波导挤压天线系统用于多用户下行链路信息传输，由新兴PASS和RIS集成对无线通信的未知影响所驱动。首先，我们在一个统一的框架中制定了求和速率和能量效率最大化问题，受到PA可移动区域、总功率预算和RIS元件可调相位的约束。然后，通过利用RIS辅助PASS的图结构拓扑，提出了一种新颖的三阶段图神经网络，该网络根据用户位置学习PA位置，并根据复合信道条件学习RIS相移，最后确定波束成形向量。具体来说，所提出的GNN通过无监督训练实现，并结合三种将其与凸优集成的实现策略，从而在推理时间和解决方案最优性之间提供权衡。提供了大量的数值结果来验证所提出的GNN的有效性，并支持其独特的泛化能力、良好的性能可靠性和实时适用性等特性。此外，还说明和分析了对RIS辅助PASS的关键参数的影响。


### 论文摘要

This paper investigates a reconfigurable intelligent surface (RIS)-assisted multi-waveguide pinching-antenna (PA) system (PASS) for multi-user downlink information transmission, motivated by the unknown impact of the integration of emerging PASS and RIS on wireless communications. First, we formulate sum rate (SR) and energy efficiency (EE) maximization problems in a unified framework, subject to constraints on the movable region of PAs, total power budget, and tunable phase of RIS elements. Then, by leveraging a graph-structured topology of the RIS-assisted PASS, a novel three-stage graph neural network (GNN) is proposed, which learns PA positions based on user locations, and RIS phase shifts according to composite channel conditions at the first two stages, respectively, and finally determines beamforming vectors. Specifically, the proposed GNN is achieved through unsupervised training, together with three implementation strategies for its integration with convex optimization, thus offering trade-offs between inference time and solution optimality. Extensive numerical results are provided to validate the effectiveness of the proposed GNN, and to support its unique attributes of viable generalization capability, good performance reliability, and real-time applicability. Moreover, the impact of key parameters on RIS-assisted PASS is illustrated and analyzed.

---

## 137. Decoupling and Damping: Structurally-Regularized Gradient Matching for Multimodal Graph Condensation

**论文链接:** [http://arxiv.org/abs/2511.20222v1](http://arxiv.org/abs/2511.20222v1)

**作者:** Lian Shen, Zhendan Chen, Yinhui jiang, Meijia Song, Ziming Su, Juan Liu, Xiangrong Liu

**发布时间:** 2025-11-25

**备注:** 11pages,5 figures,6 tables

### GPT解析

### 总结

本文提出了一种名为结构化正则化梯度匹配(SR-GM)的新型图凝聚框架，专门用于解决多模态图训练中的计算负担问题。该框架通过梯度解耦机制和结构阻尼正则化器解决了多模态图中的梯度冲突和结构放大问题，显著提高了准确率并加速了收敛。

### 背景

在电子商务和推荐系统等关键网络应用中，整合丰富视觉和文本属性的多模态图变得越来越重要。然而，这些大规模多模态图在训练图神经网络时带来了巨大的计算负担，而现有的图凝聚方法在多模态设置中表现不佳。

### 目的

开发一种专门针对多模态图的新型图凝聚框架，解决现有方法在多模态设置中的失败问题，提高训练效率和准确性。

### 方法

提出结构化正则化梯度匹配(SR-GM)，包含两个协同组件：(1)梯度解耦机制，通过正交投影解决模态间冲突；(2)结构阻尼正则化器，直接作用于梯度场，利用图的狄利克雷能量将拓扑从噪声放大器转变为优化过程中的稳定力量。

### 主要发现

SR-GM显著提高了多模态图训练的准确率并加速了收敛；同时解决梯度冲突和结构放大对于实现卓越性能至关重要；凝聚的多模态图表现出强大的跨架构泛化能力，有望加速神经架构搜索等应用。

### 结论

SR-GM为资源受限环境中的多模态图学习提供了可扩展的方法论，解决了多模态图训练中的计算负担问题。

### 翻译

在电子商务和推荐系统等关键网络应用中，整合丰富视觉和文本属性的多模态图正变得越来越重要，然而它们的大规模为训练图神经网络(GNNs)带来了巨大的计算负担。虽然图凝聚(GC)通过合成更小的数据集提供了一个有前景的解决方案，但现有方法在多模态设置中表现不佳。我们确定了导致这种失败的双重挑战：(1)不同模态之间语义不一致导致的冲突梯度，以及(2)GNN的消息传递架构在图结构上病理性地放大了这种梯度噪声。为解决这一问题，我们提出了结构化正则化梯度匹配(SR-GM)，一种专门针对多模态图的新型凝聚框架。SR-GM引入了两个协同组件：首先，通过正交投影在源头上解决模态间冲突的梯度解耦机制；其次，直接作用于梯度场的结构阻尼正则化器。通过利用图的狄利克雷能量，该正则化器在优化过程中将拓扑从噪声放大器转变为稳定力量。大量实验表明，与基线方法相比，SR-GM显著提高了准确率并加速了收敛。消融研究证实，同时解决梯度冲突和结构放大对于实现卓越性能至关重要。此外，凝聚的多模态图表现出强大的跨架构泛化能力，有望加速神经架构搜索等应用。这项研究为资源受限环境中的多模态图学习提供了可扩展的方法论。


### 论文摘要

In critical web applications such as e-commerce and recommendation systems, multimodal graphs integrating rich visual and textual attributes are increasingly central, yet their large scale introduces substantial computational burdens for training Graph Neural Networks (GNNs). While Graph Condensation (GC) offers a promising solution by synthesizing smaller datasets, existing methods falter in the multimodal setting. We identify a dual challenge causing this failure: (1) conflicting gradients arising from semantic misalignments between modalities, and (2) the GNN's message-passing architecture pathologically amplifying this gradient noise across the graph structure. To address this, we propose Structurally-Regularized Gradient Matching (SR-GM), a novel condensation framework tailored for multimodal graphs. SR-GM introduces two synergistic components: first, a gradient decoupling mechanism that resolves inter-modality conflicts at their source via orthogonal projection; and second, a structural damping regularizer that acts directly on the gradient field. By leveraging the graph's Dirichlet energy, this regularizer transforms the topology from a noise amplifier into a stabilizing force during optimization. Extensive experiments demonstrate that SR-GM significantly improves accuracy and accelerates convergence compared to baseline methods. Ablation studies confirm that addressing both gradient conflict and structural amplification in tandem is essential for achieving superior performance. Moreover, the condensed multimodal graphs exhibit strong cross-architecture generalization and promise to accelerate applications like Neural Architecture Search. This research provides a scalable methodology for multimodal graph-based learning in resource-constrained environments.

---

## 138. How to Use Deep Learning to Identify Sufficient Conditions: A Case Study on Stanley's $e$-Positivity

**论文链接:** [http://arxiv.org/abs/2511.20019v1](http://arxiv.org/abs/2511.20019v1)

**作者:** Farid Aliniaeifard, Shu Xiao Li

**发布时间:** 2025-11-25

**备注:** Extended abstract submitted to FPSAC 2026. It is 13 pages with 4 figures

### GPT解析

### 总结

本研究开发了一种使用机器学习识别数学陈述充分条件的方法，应用于图的e-正性问题，解决了代数组合学中的一个长期猜想。

### 背景

该研究基于发表在《Nature》上的工作，展示了使用机器学习在纯数学中提出猜想的通用框架。e-正性问题是代数组合学过去三十年的核心问题之一。

### 目的

开发一种方法来识别能够使给定数学陈述成立的充分条件。

### 方法

使用优先考虑高精度的自定义损失函数训练神经网络，然后应用归因技术和探索性数据分析来提出猜想，并将此方法应用于图的e-正性问题。

### 主要发现

1) 图的e-正性的一个充分条件是它无三角形；2) 爪子的数量是e-正性的最重要因素；3) e-正性图的分类更可能与连续图不变量而非离散图不变量相关；4) 10个和11个顶点的无爪图和可收缩无爪图是e-正性的。

### 结论

通过人工智能引导的方法成功解决了代数组合学中的一个长期存在的问题，展示了机器学习在纯数学研究中的潜力。

### 翻译

一项发表在《Nature》上的研究中，DeepMind的研究人员和数学家展示了一个使用机器学习在纯数学中提出猜想的通用框架。他们的工作使用神经网络和归因技术来引导人类直觉，从而提出可证明的猜想。在这里，我们基于此框架开发了一种识别给定数学陈述的充分条件的方法。我们的方法使用优先考虑高精度的自定义损失函数训练神经网络，然后使用归因技术和探索性数据分析来提出猜想。作为演示，我们将此过程应用于Stanley的图e-正性问题——这是过去三十年代数组合学中心的问题。在AI的指导下，我们重新发现图是e-正性的一个充分条件是它无三角形，并且爪子的数量是e-正性的最重要因素。基于神经网络显著图分析中的最重要因素，我们提出e-正性图的分类更可能与连续图不变量而非离散图不变量相关。此外，使用神经网络和探索性数据分析，我们证明了10个和11个顶点的无爪图和可收缩无爪图是e-正性的，解决了Dahlberg、Foley和van Willigenburg的一个猜想。


### 论文摘要

In a study, published in \emph{Nature}, researchers from DeepMind and mathematicians demonstrated a general framework using machine learning to make conjectures in pure mathematics. Their work uses neural networks and attribution techniques to guide human intuition towards making provable conjectures. Here, we build upon this framework to develop a method for identifying sufficient conditions that imply a given mathematical statement. Our approach trains neural networks with a custom loss function that prioritizes high precision. Then uses attribution techniques and exploratory data analysis to make conjectures. As a demonstration, we apply this process to Stanley's problem of $e$-positivity of graphs--a problem that has been at the center of algebraic combinatorics for the past three decades. Guided by AI, we rediscover that one sufficient condition for a graph to be $e$-positive is that it is co-triangle-free, and that the number of claws is the most important factor for $e$-positivity. Based on the most important factors in Saliency Map analysis of neural networks, we suggest that the classification of $e$-positive graphs is more related to continuous graph invariants rather than the discrete ones. Furthermore, using neural networks and exploratory data analysis, we show that the claw-free and claw-contractible-free graphs with $10$ and $11$ vertices are $e$-positive, resolving a conjecture by Dahlberg, Foley, and van Willigenburg.

---

## 139. Rethinking Message Passing Neural Networks with Diffusion Distance-guided Stress Majorization

**论文链接:** [http://arxiv.org/abs/2511.19984v1](http://arxiv.org/abs/2511.19984v1)

**作者:** Haoran Zheng, Renchi Yang, Yubo Zhou, Jianliang Xu

**发布时间:** 2025-11-25

**备注:** Accepted by SIGKDD 2026. The code is available at https://github.com/HaoranZ99/DDSM

### GPT解析

### 总结

本文提出了一种名为DDSM的新型消息传递神经网络模型，通过优化框架有效克服了传统MPNN中的过平滑和过相关性问题。

### 背景

消息传递神经网络(MPNNs)在过去十年已成为学习图结构数据的主流模型，但大多数此类模型仍存在严重问题。

### 目的

解决传统MPNN模型中由于最小化狄利克雷能量和邻域聚合操作导致的过平滑和过相关性问题。

### 方法

提出DDSM模型，构建在包含应力主要化和正则化的优化框架上，并将节点扩散距离引入框架以指导新的消息传递操作，同时开发了距离近似的高效算法。

### 主要发现

全面的实验表明，DDSM在同质图和异质图上都一致且显著地优于15个强大的基线模型。

### 结论

DDSM通过创新的优化框架和距离引导的消息传递操作，有效解决了传统MPNN的局限性，在各种图数据上表现出色。

### 翻译

消息传递神经网络(MPNNs)在过去十年已成为学习图结构数据的首选模型。尽管它们有效，但由于其最小化狄利克雷能量的基本目标和派生的邻域聚合操作，大多数此类模型仍存在严重的过平滑和过相关性问题。在本文中，我们提出了DDSM，一种新的MPNN模型，它建立在包含应力主要化和正则化的优化框架上，以克服上述问题。此外，我们将节点的扩散距离引入框架，以指导新的消息传递操作，并开发了距离近似的高效算法，这些都得到了严格的理论分析支持。我们全面的实验表明，DDSM在同质图和异质图上都一致且显著地优于15个强大的基线模型。


### 论文摘要

Message passing neural networks (MPNNs) have emerged as go-to models for learning on graph-structured data in the past decade. Despite their effectiveness, most of such models still incur severe issues such as over-smoothing and -correlation, due to their underlying objective of minimizing the Dirichlet energy and the derived neighborhood aggregation operations. In this paper, we propose the DDSM, a new MPNN model built on an optimization framework that includes the stress majorization and orthogonal regularization for overcoming the above issues. Further, we introduce the diffusion distances for nodes into the framework to guide the new message passing operations and develop efficient algorithms for distance approximations, both backed by rigorous theoretical analyses. Our comprehensive experiments showcase that DDSM consistently and considerably outperforms 15 strong baselines on both homophilic and heterophilic graphs.

---

## 140. Rethinking Semi-Supervised Node Classification with Self-Supervised Graph Clustering

**论文链接:** [http://arxiv.org/abs/2511.19976v1](http://arxiv.org/abs/2511.19976v1)

**作者:** Songbo Wang, Renchi Yang, Yurui Lai, Xiaoyang Lin, Tsz Nam Chan

**发布时间:** 2025-11-25

**备注:** 14 pages

### GPT解析

### 总结

本文提出了一种名为NCGC的统一框架，将自监督图聚类与半监督分类相结合，有效利用图中紧密社区/集群的信号来缓解半监督节点分类中的标签稀缺问题。

### 背景

图神经网络(GNNs)已成为半监督节点分类任务的有力工具，后续研究通过改进消息传递方案或数据增强技术来缓解监督有限的问题。实际图中节点形成的紧密社区包含丰富信号，可用于补偿标签稀缺，但先前方法未充分利用这一点。

### 目的

开发一种新方法NCGC，将自监督图聚类和半监督分类集成到统一框架中，利用图中紧密社区/集群的信号来提升半监督节点分类性能。

### 方法

1) 理论上统一GNNs和谱图聚类的优化目标；2) 开发软正交GNNs(SOGNs)生成用于分类和聚类的节点表示；3) 构建自监督图聚类模块，包含两个聚类目标和Sinkhorn-Knopp归一化，将预测聚类转换为平衡软伪标签；4) 通过多任务目标结合聚类模块与分类模型，促进协同作用增强模型能力。

### 主要发现

在七个真实图上，使用各种经典GNN骨干网络时，NCGC框架在半监督节点分类任务中始终且显著优于流行的GNN模型和最近的基线方法。

### 结论

NCGC通过整合自监督图聚类与半监督分类，有效利用图中紧密社区/集群的信号，解决了半监督节点分类中的标签稀缺问题，实现了模型性能的显著提升。

### 翻译

图神经网络(GNNs)的出现为半监督节点分类任务提供了强大的工具。后续研究通过改进GNN模型中的消息传递方案或利用各种数据增强技术来缓解监督有限的问题。在实际图中，节点往往倾向于形成紧密的社区/集群，这些集群包含丰富的信号，可以补偿半监督节点分类中的标签稀缺性，但先前的方法没有探索这一点。受此启发，本文提出了NCGC，它将自监督图聚类和半监督分类集成到一个统一的框架中。首先，我们从理论上统一了GNNs和谱图聚类的优化目标，并基于此开发了软正交GNNs (SOGNs)，利用改进的消息传递范式生成用于分类和聚类的节点表示。在此基础上，NCGC包括一个自监督图聚类模块，使SOGNs能够以自监督方式学习未标记节点的表示。特别是，该组件包含两个非平凡的聚类目标和一个Sinkhorn-Knopp归一化，将预测的聚类分配转换为平衡的软伪标签。通过将上述聚类模块与使用多任务目标的分类模型相结合（包含标记数据上的监督分类损失和未标记数据上的自监督聚类损失），NCGC促进了它们之间的协同作用，增强了模型能力。大量实验表明，当使用各种经典GNN骨干网络时，在七个真实图上，我们提出的NCGC框架在半监督节点分类任务中始终且显著优于流行的GNN模型和最近的基线方法。


### 论文摘要

The emergence of graph neural networks (GNNs) has offered a powerful tool for semi-supervised node classification tasks. Subsequent studies have achieved further improvements through refining the message passing schemes in GNN models or exploiting various data augmentation techniques to mitigate limited supervision. In real graphs, nodes often tend to form tightly-knit communities/clusters, which embody abundant signals for compensating label scarcity in semi-supervised node classification but are not explored in prior methods.   Inspired by this, this paper presents NCGC that integrates self-supervised graph clustering and semi-supervised classification into a unified framework. Firstly, we theoretically unify the optimization objectives of GNNs and spectral graph clustering, and based on that, develop soft orthogonal GNNs (SOGNs) that leverage a refined message passing paradigm to generate node representations for both classification and clustering. On top of that, NCGC includes a self-supervised graph clustering module that enables the training of SOGNs for learning representations of unlabeled nodes in a self-supervised manner. Particularly, this component comprises two non-trivial clustering objectives and a Sinkhorn-Knopp normalization that transforms predicted cluster assignments into balanced soft pseudo-labels. Through combining the foregoing clustering module with the classification model using a multi-task objective containing the supervised classification loss on labeled data and self-supervised clustering loss on unlabeled data, NCGC promotes synergy between them and achieves enhanced model capacity. Our extensive experiments showcase that the proposed NCGC framework consistently and considerably outperforms popular GNN models and recent baselines for semi-supervised node classification on seven real graphs, when working with various classic GNN backbones.

---

## 141. GED-Consistent Disentanglement of Aligned and Unaligned Substructures for Graph Similarity Learning

**论文链接:** [http://arxiv.org/abs/2511.19837v1](http://arxiv.org/abs/2511.19837v1)

**作者:** Zhentao Zhan, Xiaoliang Xu, Jingjing Wang, Junmei Wang

**发布时间:** 2025-11-25

### GPT解析

### 总结

本文提出了一种名为GCGSim的GED一致的图相似性学习框架，解决了现有基于GNN的GED方法在节点中心匹配范式与GED核心原则之间的不匹配问题，实现了更准确的图相似性计算。

### 背景

图相似性计算(GSC)是基本图任务，图编辑距离(GED)是常用度量。由于精确GED计算具有NP-hard性质，基于图神经网络(GNN)的近似方法应运而生。现有方法通常学习节点嵌入并聚合节点相似性来估计图相似度。

### 目的

解决现有节点中心匹配范式与GED核心原则之间的不匹配问题，该不匹配导致无法捕获最优对齐的全局结构对应关系，以及由虚假节点级信号导致的编辑成本错误归因问题。

### 方法

提出GCGSim，一个GED一致的图相似性学习框架，围绕图级匹配和子结构级编辑成本构建，包含三个核心技术贡献。

### 主要发现

在四个基准数据集上的实验表明，GCGSim实现了最先进的性能。综合分析验证了该框架能有效学习解耦且语义上有意义的子结构表示。

### 结论

GCGSim通过关注图级匹配和子结构级编辑成本，成功解决了现有方法在GED计算中的两个关键限制，能够更好地捕捉全局结构对应关系，学习到更准确的子结构表示。

### 翻译

图相似性计算(GSC)是一项基本的图相关任务，其中图编辑距离(GED)是一种常用度量。GED通过一对图之间的最优对齐来确定，将每个图划分为对齐（零成本）和未对齐（产生成本）的子结构。由于精确GED计算的NP-hard性质，基于图神经网络(GNN)的GED近似方法应运而生。现有的基于GNN的GED方法通常学习每个图的节点嵌入，然后聚合节点对之间的相似性来估计最终相似度。尽管这些方法有效，但我们发现这种普遍的节点中心匹配范式与GED的核心原则之间存在不匹配。这种差异导致两个关键限制：(1)无法捕获最优对齐的全局结构对应关系，(2)由于虚假的节点级信号导致编辑成本被错误归因。为解决这些限制，我们提出了GCGSim，一个以图级匹配和子结构级编辑成本为中心的GED一致的图相似性学习框架。具体来说，我们做出了三个核心技术贡献。在四个基准数据集上的大量实验表明，GCGSim实现了最先进的性能。我们的综合分析进一步验证了该框架能有效学习解耦且语义上有意义的子结构表示。


### 论文摘要

Graph Similarity Computation (GSC) is a fundamental graph related task where Graph Edit Distance (GED) serves as a prevalent metric. GED is determined by an optimal alignment between a pair of graphs that partitions each into aligned (zero-cost) and unaligned (cost-incurring) substructures. Due to NP-hard nature of exact GED computation, GED approximations based on Graph Neural Network(GNN) have emerged. Existing GNN-based GED approaches typically learn node embeddings for each graph and then aggregate pairwise node similarities to estimate the final similarity. Despite their effectiveness, we identify a mismatch between this prevalent node-centric matching paradigm and the core principles of GED. This discrepancy leads to two critical limitations: (1) a failure to capture the global structural correspondence for optimal alignment, and (2) a misattribution of edit costs driven by spurious node level signals. To address these limitations, we propose GCGSim, a GED-consistent graph similarity learning framework centering on graph-level matching and substructure-level edit costs. Specifically, we make three core technical contributions. Extensive experiments on four benchmark datasets show that GCGSim achieves state-of-the-art performance. Our comprehensive analyses further validate that the framework effectively learns disentangled and semantically meaningful substructure representations.

---

## 142. Solar-GECO: Perovskite Solar Cell Property Prediction with Geometric-Aware Co-Attention

**论文链接:** [http://arxiv.org/abs/2511.19263v1](http://arxiv.org/abs/2511.19263v1)

**作者:** Lucas Li, Jean-Baptiste Puel, Florence Carton, Dounya Barrit, Jhony H. Giraldo

**发布时间:** 2025-11-24

**备注:** Accepted at the AI for Accelerated Materials Design (AI4Mat) Workshop at NeurIPS 2025. 14 pages, 4 figures

### GPT解析

### 总结

本文提出了一种名为Solar-GECO的几何感知共同注意力模型，用于预测钙钛矿太阳能电池的功率转换效率，该模型结合了几何图神经网络和语言模型嵌入，同时考虑了材料的原子结构和化学组成信息。

### 背景

钙钛矿太阳能电池是下一代光伏技术的有前途候选者，但其性能取决于组成层之间的复杂相互作用，导致材料筛选过程缓慢且昂贵。现有机器学习模型仅关注单个材料属性或忽略钙钛矿晶体的重要几何信息。

### 目的

开发一种能够同时考虑几何信息和文本信息的模型，以更准确地预测钙钛矿太阳能电池的功率转换效率，克服现有方法的局限性。

### 方法

Solar-GECO模型结合了几何图神经网络（GNN）编码钙钛矿吸收层的原子结构，以及语言模型嵌入处理传输层和其他器件组分的化学化合物文本表示。模型还集成了共同注意力模块捕获层内依赖性和层间相互作用，并通过概率回归头预测功率转换效率及其不确定性。

### 主要发现

Solar-GECO实现了最先进的性能，显著优于多个基线模型，与之前的最先进模型语义GNN相比，将功率转换效率预测的平均绝对误差从3.066降低到2.936。

### 结论

集成几何和文本信息为钙钛矿太阳能电池功率转换效率预测提供了更强大和准确的框架，有助于加速新型高效太阳能电池的开发。

### 翻译

钙钛矿太阳能电池是下一代光伏技术的有前途候选者。然而，它们作为多尺度设备的性能取决于其组成层之间的复杂相互作用。这产生了大量的可能材料和器件架构组合空间，使得基于实验的筛选过程缓慢且昂贵。机器学习模型试图解决这个问题，但它们只关注单个材料属性或忽略了钙钛矿晶体的重要几何信息。为解决这一问题，我们提出了一种几何感知共同注意力（Solar-GECO）模型来预测钙钛矿太阳能电池的功率转换效率。Solar-GECO结合了几何图神经网络（GNN）——直接编码钙钛矿吸收层的原子结构，以及处理代表传输层和其他器件组分的化学化合物的文本字符串的语言模型嵌入。Solar-GECO还集成了一个共同注意力模块来捕获层内依赖性和层间相互作用，同时通过概率回归头预测功率转换效率（PCE）及其相关的不确定性。Solar-GECO实现了最先进的性能，显著优于多个基线模型，与之前的最先进模型语义GNN相比，将PCE预测的平均绝对误差（MAE）从3.066降低到2.936。Solar-GECO证明了集成几何和文本信息为PCE预测提供了更强大和准确的框架。


### 论文摘要

Perovskite solar cells are promising candidates for next-generation photovoltaics. However, their performance as multi-scale devices is determined by complex interactions between their constituent layers. This creates a vast combinatorial space of possible materials and device architectures, making the conventional experimental-based screening process slow and expensive. Machine learning models try to address this problem, but they only focus on individual material properties or neglect the important geometric information of the perovskite crystal. To address this problem, we propose to predict perovskite solar cell power conversion efficiency with a geometric-aware co-attention (Solar-GECO) model. Solar-GECO combines a geometric graph neural network (GNN) - that directly encodes the atomic structure of the perovskite absorber - with language model embeddings that process the textual strings representing the chemical compounds of the transport layers and other device components. Solar-GECO also integrates a co-attention module to capture intra-layer dependencies and inter-layer interactions, while a probabilistic regression head predicts both power conversion efficiency (PCE) and its associated uncertainty. Solar-GECO achieves state-of-the-art performance, significantly outperforming several baselines, reducing the mean absolute error (MAE) for PCE prediction from 3.066 to 2.936 compared to semantic GNN (the previous state-of-the-art model). Solar-GECO demonstrates that integrating geometric and textual information provides a more powerful and accurate framework for PCE prediction.

---

## 143. GraphMind: Theorem Selection and Conclusion Generation Framework with Dynamic GNN for LLM Reasoning

**论文链接:** [http://arxiv.org/abs/2511.19078v1](http://arxiv.org/abs/2511.19078v1)

**作者:** Yutong Li, Yitian Zhou, Xudong Wang, GuoChen, Caiyan Qin

**发布时间:** 2025-11-24

### GPT解析

### 总结

GraphMind是一种新颖的动态图框架，结合图神经网络和大型语言模型，用于多步推理中的定理选择和中间结论生成，实现了上下文感知、可解释和结构化的推理过程。

### 背景

大型语言模型在自然语言理解和生成方面表现出色，包括多步推理如数学证明，但现有方法缺乏明确和动态的机制来结构化表示和演变中间推理状态。

### 目的

解决现有方法在上下文感知定理选择和迭代结论生成方面的局限性，提出GraphMind框架。

### 方法

将推理过程建模为异构演化图，节点表示条件、定理和结论，边捕获逻辑依赖；通过图神经网络编码当前推理状态，利用语义匹配进行定理选择，实现闭环推理。

### 主要发现

在多个问答数据集上的实验表明，GraphMind实现了持续的性能提升，并在多步推理任务中显著优于现有基线方法。

### 结论

GraphMind框架有效且通用，能够实现上下文感知、可解释和结构化的多步推理。

### 翻译

大型语言模型在自然语言理解和生成方面展现出了令人印象深刻的能力，包括多步推理如数学证明。然而，现有方法往往缺乏明确和动态的机制来结构化表示和演变中间推理状态，这限制了它们执行上下文感知的定理选择和迭代结论生成的能力。为解决这些挑战，我们提出了GraphMind，一种新颖的基于动态图的框架，将图神经网络与大型语言模型集成，用于迭代选择定理和生成多步推理的中间结论。我们的方法将推理过程建模为异构演化图，其中节点表示条件、定理和结论，边捕获节点之间的逻辑依赖。通过用图神经网络编码当前推理状态并利用语义匹配进行定理选择，我们的框架实现了闭环的上下文感知、可解释和结构化推理。在各种问答数据集上的实验表明，我们提出的GraphMind方法实现了持续的性能提升，并在多步推理中显著优于现有基线，验证了我们方法的有效性和通用性。


### 论文摘要

Large language models (LLMs) have demonstrated impressive capabilities in natural language understanding and generation, including multi-step reasoning such as mathematical proving. However, existing approaches often lack an explicit and dynamic mechanism to structurally represent and evolve intermediate reasoning states, which limits their ability to perform context-aware theorem selection and iterative conclusion generation. To address these challenges, we propose GraphMind, a novel dynamic graph-based framework that integrates the graph neural network (GNN) with LLMs to iteratively select theorems and generate intermediate conclusions for multi-step reasoning. Our method models the reasoning process as a heterogeneous evolving graph, where nodes represent conditions, theorems, and conclusions, while edges capture logical dependencies between nodes. By encoding the current reasoning state with GNN and leveraging semantic matching for theorem selection, our framework enables context-aware, interpretable, and structured reasoning in a closed-loop manner. Experiments on various question-answering (QA) datasets demonstrate that our proposed GraphMind method achieves consistent performance improvements and significantly outperforms existing baselines in multi-step reasoning, validating the effectiveness and generalizability of our approach.

---

## 144. Resolving Node Identifiability in Graph Neural Processes via Laplacian Spectral Encodings

**论文链接:** [http://arxiv.org/abs/2511.19037v1](http://arxiv.org/abs/2511.19037v1)

**作者:** Zimo Yan, Zheng Xie, Chang Liu, Yuan Wang

**发布时间:** 2025-11-24

### GPT解析

### 总结

本文提出了一种拉普拉斯位置编码方法，克服了消息传递图神经网络因一维Weisfeiler-Lehman测试限制而无法区分结构不同节点的问题，并通过理论和实验验证了其优越性。

### 背景

消息传递图神经网络被广泛用于图学习，但其表达能力受限于一维Weisfeiler-Lehman测试，无法区分结构上不同的节点。

### 目的

提供一种对特征向量符号翻转和特征空间内基旋转具有不变性的拉普拉斯位置编码的严格理论，并证明其从常数个观察中实现节点可识别性的能力。

### 方法

结合最短路径和扩散距离之间的单调联系、使用固定锚点集进行谱三边测量、以及对数嵌入大小的定量谱内射性分析，建立与受Weisfeiler-Lehman测试限制架构的样本复杂度分离。

### 主要发现

所提出的拉普拉斯位置编码能够从常数个观察中实现节点可识别性，与受Weisfeiler-Lehman测试限制的架构相比存在样本复杂度分离。

### 结论

将这种编码与神经过程风格的解码器配对，在化学图的药物-药物相互作用任务上显著提升了ROC曲线下面积和F1分数，展示了通过有原则的位置信息解决理论表达能力限制的实际好处。

### 翻译

消息传递图神经网络被广泛用于图学习，但它们的一维Weisfeiler-Lehman测试限制了其表达能力，可能无法区分结构上不同的节点。我们为拉普拉斯位置编码提供了严格的理论，这种编码对特征向量符号翻转和特征空间内基旋转具有不变性。我们证明这种编码可以从常数个观察中实现节点可识别性，并建立了与受Weisfeiler-Lehman测试限制架构的样本复杂度分离。该分析结合了最短路径和扩散距离之间的单调联系、使用固定锚点集的谱三边测量，以及对数嵌入大小的定量谱内射性。作为实例，将这种编码与神经过程风格的解码器配对在化学图的药物-药物相互作用任务上获得显著提升，同时改进了ROC曲线下面积和F1分数，展示了通过有原则的位置信息解决理论表达能力限制的实际好处。


### 论文摘要

Message passing graph neural networks are widely used for learning on graphs, yet their expressive power is limited by the one-dimensional Weisfeiler-Lehman test and can fail to distinguish structurally different nodes. We provide rigorous theory for a Laplacian positional encoding that is invariant to eigenvector sign flips and to basis rotations within eigenspaces. We prove that this encoding yields node identifiability from a constant number of observations and establishes a sample-complexity separation from architectures constrained by the Weisfeiler-Lehman test. The analysis combines a monotone link between shortest-path and diffusion distance, spectral trilateration with a constant set of anchors, and quantitative spectral injectivity with logarithmic embedding size. As an instantiation, pairing this encoding with a neural-process style decoder yields significant gains on a drug-drug interaction task on chemical graphs, improving both the area under the ROC curve and the F1 score and demonstrating the practical benefits of resolving theoretical expressiveness limitations with principled positional information.

---

## 145. Learning to Solve Weighted Maximum Satisfiability with a Co-Training Architecture

**论文链接:** [http://arxiv.org/abs/2511.19544v1](http://arxiv.org/abs/2511.19544v1)

**作者:** Kaidi Wan, Minghao Liu, Yong Lai

**发布时间:** 2025-11-24

**备注:** 10 pages, 4 figures

### GPT解析

### 总结

这篇论文提出了SplitGNN，一种基于图神经网络的方法，用于学习解决加权最大可满足性问题。

### 背景

加权最大可满足性(MaxSAT)问题是一个具有挑战性的计算问题，需要更高效的求解方法。

### 目的

开发一种能够高效解决加权MaxSAT问题的方法，提高求解速度和解的质量。

### 方法

SplitGNN采用协同训练架构，包含监督消息传递机制和无监督解决方案提升层；提出边分割因子图表示方法，基于生成树生成和边分类；实现GPU加速层，应用高效分数计算和基于松弛的优化技术。

### 主要发现

SplitGNN相比其他基于GNN的架构实现了3倍更快的收敛速度和更好的预测结果；在更大和更难的加权MaxSAT基准测试中找到了比现代启发式MaxSAT求解器更好的解决方案；在不同结构实例上表现出卓越的泛化能力。

### 结论

SplitGNN是一种有效的解决加权MaxSAT问题的方法，具有更快的收敛速度、更好的解决方案质量和强大的泛化能力。

### 翻译

我们提出了SplitGNN，一种基于图神经网络(GNN)的方法，用于学习解决加权最大可满足性(MaxSAT)问题。SplitGNN采用协同训练架构，包括监督消息传递机制和无监督解决方案提升层。我们提出了一种新的图表示方法——边分割因子图，为学习提供更多结构信息，该方法基于生成树生成和边分类。为了改进具有挑战性和加权的实例中的解决方案，我们实现了一个GPU加速层，应用高效的分数计算和基于松弛的优化。实验表明，与其他基于GNN的架构相比，SplitGNN实现了3倍更快的收敛速度和更好的预测结果。更值得注意的是，SplitGNN在更大和更难的加权MaxSAT基准测试中成功找到了比现代启发式MaxSAT求解器更好的解决方案，并在不同结构的实例上表现出卓越的泛化能力。


### 论文摘要

Wepropose SplitGNN, a graph neural network (GNN)-based   approach that learns to solve weighted maximum satisfiabil ity (MaxSAT) problem. SplitGNN incorporates a co-training   architecture consisting of supervised message passing mech anism and unsupervised solution boosting layer. A new graph   representation called edge-splitting factor graph is proposed   to provide more structural information for learning, which is   based on spanning tree generation and edge classification. To   improve the solutions on challenging and weighted instances,   we implement a GPU-accelerated layer applying efficient   score calculation and relaxation-based optimization. Exper iments show that SplitGNN achieves 3* faster convergence   and better predictions compared with other GNN-based ar chitectures. More notably, SplitGNN successfully finds solu tions that outperform modern heuristic MaxSAT solvers on   much larger and harder weighted MaxSAT benchmarks, and   demonstrates exceptional generalization abilities on diverse   structural instances.

---

## 146. 论文ID: 2511.18859v1

**错误:** 无法读取论文信息 - [Errno 2] No such file or directory: 'storage/result/2511.18859v1.json'

---

## 147. Auto-ML Graph Neural Network Hypermodels for Outcome Prediction in Event-Sequence Data

**论文链接:** [http://arxiv.org/abs/2511.18835v1](http://arxiv.org/abs/2511.18835v1)

**作者:** Fang Wang, Lance Kosca, Adrienne Kosca, Marko Gacesa, Ernesto Damiani

**发布时间:** 2025-11-24

**备注:** 6 pages

### GPT解析

### 总结

本研究提出了HGNN(O)，一个用于事件序列数据结果预测的AutoML GNN超模型框架，通过扩展四种架构和六种标准GNN算子，并采用基于贝叶斯优化的自调机制，实现了高效的自适应学习，在多个数据集上取得了优异的性能。

### 背景

基于作者之前关于图卷积网络超模型的工作，本研究旨在解决事件序列数据中结果预测的挑战，特别是处理不平衡数据的问题。

### 目的

开发一个能够自动优化架构和超参数的GNN框架，用于复杂事件序列数据的结果预测，无需手动配置和明确的不平衡处理方法。

### 方法

扩展了四种架构（一级、两级、两级伪嵌入和两级嵌入）和六种标准GNN算子，并实现了基于贝叶斯优化的自调机制，结合剪枝和提前停止技术，以实现高效的自适应学习。

### 主要发现

在平衡和不平衡事件日志上的评估中，HGNN(O)在交通罚款数据集上实现了超过0.98的准确率，在患者数据集上达到了0.86的加权F1分数，而无需明确的不平衡处理。

### 结论

所提出的AutoML-GNN方法为复杂事件序列数据中的结果预测提供了一个稳健且可推广的基准，具有高效的自适应能力和优异的性能表现。

### 翻译

本文介绍了HGNN(O)，一个用于事件序列数据结果预测的AutoML GNN超模型框架。基于我们之前关于图卷积网络超模型的工作，HGNN(O)扩展了四种架构——一级、两级、两级伪嵌入和两级嵌入——跨越六个标准GNN算子。基于贝叶斯优化、剪枝和提前停止的自调机制能够高效地适应架构和超参数，无需手动配置。在平衡和不平衡事件日志上的经验评估表明，HGNN(O)在交通罚款数据集上实现了超过0.98的准确率，在患者数据集上达到了0.86的加权F1分数，而无需明确的不平衡处理。这些结果表明，所提出的AutoML-GNN方法为复杂事件序列数据中的结果预测提供了一个稳健且可推广的基准。


### 论文摘要

This paper introduces HGNN(O), an AutoML GNN hypermodel framework for outcome prediction on event-sequence data. Building on our earlier work on graph convolutional network hypermodels, HGNN(O) extends four architectures-One Level, Two Level, Two Level Pseudo Embedding, and Two Level Embedding-across six canonical GNN operators. A self-tuning mechanism based on Bayesian optimization with pruning and early stopping enables efficient adaptation over architectures and hyperparameters without manual configuration. Empirical evaluation on both balanced and imbalanced event logs shows that HGNN(O) achieves accuracy exceeding 0.98 on the Traffic Fines dataset and weighted F1 scores up to 0.86 on the Patients dataset without explicit imbalance handling. These results demonstrate that the proposed AutoML-GNN approach provides a robust and generalizable benchmark for outcome prediction in complex event-sequence data.

---

## 148. Hypergraph Contrastive Learning for both Homophilic and Heterophilic Hypergraphs

**论文链接:** [http://arxiv.org/abs/2511.18783v1](http://arxiv.org/abs/2511.18783v1)

**作者:** Renchu Guan, Xuyang Li, Yachao Zhang, Wei Pang, Fausto Giunchiglia, Ximing Li, Yonghao Liu, Xiaoyue Feng

**发布时间:** 2025-11-24

### GPT解析

### 总结

本文提出了HONOR，一种新型无监督超图对比学习框架，适用于同质和异质超图，能够产生更具判别力和鲁棒性的节点和超边表示。

### 背景

超图作为传统图的推广，能够自然地捕获高阶关系。然而，大多数现有的超图神经网络方法本质上依赖于同质性假设，而在具有显著异质结构的现实场景中，这一假设往往不成立。

### 目的

解决现有超图神经网络方法在异质结构场景中的局限性，提出一种能够同时处理同质和异质超图的框架。

### 方法

HONOR通过两种互补机制建模超边和节点之间的异质关系：基于提示的超边特征构建策略保持全局语义一致性同时抑制局部噪声；自适应注意力聚合模块动态捕获节点对超边的多样化局部贡献。结合高通滤波设计，充分利用异质连接模式。

### 主要发现

理论上证明了HONOR具有更好的泛化能力和鲁棒性。实验验证表明，在同质和异质数据集上，HONOR都持续优于最先进的基线方法。

### 结论

HONOR是一个创新的无监督超图对比学习框架，能够有效处理同质和异质超图，产生更具判别力和鲁棒性的表示，在理论和实验上都显示出优越性。

### 翻译

超图作为传统图的推广，能够自然地捕获高阶关系。近年来，超图神经网络(HNNs)已被广泛用于捕获复杂的高阶关系。然而，大多数现有的超图神经网络方法本质上依赖于同质性假设，而在具有显著异质结构的现实场景中，这一假设往往不成立。为了解决这一局限性，我们提出了HONOR，一种新颖的无监督超图对比学习框架，适用于同质和异质超图。具体来说，HONOR通过两种互补机制明确建模超边和节点之间的异质关系：一种基于提示的超边特征构建策略，保持全局语义一致性同时抑制局部噪声；以及一个自适应注意力聚合模块，动态捕获节点对超边的多样化局部贡献。结合高通滤波设计，这些设计使HONOR能够充分利用异质连接模式，产生更具判别力和鲁棒性的节点和超边表示。理论上，我们证明了HONOR具有更好的泛化能力和鲁棒性。实验上，大量实验进一步验证了HONOR在同质和异质数据集上都持续优于最先进的基线方法。


### 论文摘要

Hypergraphs, as a generalization of traditional graphs, naturally capture high-order relationships. In recent years, hypergraph neural networks (HNNs) have been widely used to capture complex high-order relationships. However, most existing hypergraph neural network methods inherently rely on the homophily assumption, which often does not hold in real-world scenarios that exhibit significant heterophilic structures. To address this limitation, we propose \textbf{HONOR}, a novel unsupervised \textbf{H}ypergraph c\textbf{ON}trastive learning framework suitable for both hom\textbf{O}philic and hete\textbf{R}ophilic hypergraphs. Specifically, HONOR explicitly models the heterophilic relationships between hyperedges and nodes through two complementary mechanisms: a prompt-based hyperedge feature construction strategy that maintains global semantic consistency while suppressing local noise, and an adaptive attention aggregation module that dynamically captures the diverse local contributions of nodes to hyperedges. Combined with high-pass filtering, these designs enable HONOR to fully exploit heterophilic connection patterns, yielding more discriminative and robust node and hyperedge representations. Theoretically, we demonstrate the superior generalization ability and robustness of HONOR. Empirically, extensive experiments further validate that HONOR consistently outperforms state-of-the-art baselines under both homophilic and heterophilic datasets.

---

## 149. CycleChemist: A Dual-Pronged Machine Learning Framework for Organic Photovoltaic Discovery

**论文链接:** [http://arxiv.org/abs/2511.19500v1](http://arxiv.org/abs/2511.19500v1)

**作者:** Hou Hei Lam, Jiangjie Qiu, Xiuyuan Hu, Wentao Li, Fankun Zeng, Siwei Fu, Hao Zhang, Xiaonan Wang

**发布时间:** 2025-11-23

### GPT解析

### 总结

该研究介绍了一种用于有机光伏材料发现的统一双机器学习框架，结合了预测建模和分子生成设计，推进了高性能OPV材料的数据驱动发现。

### 背景

有机光伏材料为可持续能源生产提供了有前景的路径，但其发展受到难以识别具有高功率转换效率的高性能给体和受体对的限制。现有设计策略通常只关注给体或受体之一，而非使用统一方法同时建模两个组件。

### 目的

开发一个能够同时考虑给体和受体材料的统一方法，用于发现高性能有机光伏材料。

### 方法

创建包含2000个实验表征给体受体对的数据集(OPV2D)；开发有机光伏分类器(OPVC)预测材料OPV行为；构建分层图神经网络结合多任务学习和给体受体相互作用建模；开发分子轨道能量估计器(MOE2)预测HOMO和LUMO能级；开发光伏性能预测器(P3)估计PCE；引入材料生成预训练变换器(MatGPT)通过强化学习生成可合成实现的有机半导体。

### 主要发现

通过将分子表示学习与性能预测相结合，该框架能够同时考虑给体和受体材料的特性，并通过生成模型创建新的候选材料。

### 结论

该双机器学习框架为OPV材料发现提供了新方法，克服了传统设计策略的局限性，加速了高性能有机光伏材料的开发进程。

### 翻译

有机光伏材料为可持续能源生产提供了有前景的路径，但其发展受到难以识别具有高功率转换效率的高性能给体和受体对的限制。现有设计策略通常只关注给体或受体之一，而非使用能够同时建模两个组件的统一方法。在本工作中，我们引入了一种用于OPV发现的双机器学习框架，结合了预测建模和分子生成设计。我们提出了有机光伏给体受体数据集(OPV2D)，这是迄今为止最大规模的同类数据集，包含2000个实验表征的给体受体对。利用该数据集，我们开发了有机光伏分类器(OPVC)来预测材料是否表现出OPV行为，以及一个分层图神经网络，结合了多任务学习和给体受体相互作用建模。该框架包括分子轨道能量估计器(MOE2)用于预测HOMO和LUMO能级，以及光伏性能预测器(P3)用于估计PCE。此外，我们引入了材料生成预训练变换器(MatGPT)，通过具有三个目标策略优化的强化学习策略来生成可合成实现的有机半导体。通过将分子表示学习与性能预测相结合，我们的框架推进了高性能OPV材料的数据驱动发现。


### 论文摘要

Organic photovoltaic (OPV) materials offer a promising path toward sustainable energy generation, but their development is limited by the difficulty of identifying high performance donor and acceptor pairs with strong power conversion efficiencies (PCEs). Existing design strategies typically focus on either the donor or the acceptor alone, rather than using a unified approach capable of modeling both components. In this work, we introduce a dual machine learning framework for OPV discovery that combines predictive modeling with generative molecular design. We present the Organic Photovoltaic Donor Acceptor Dataset (OPV2D), the largest curated dataset of its kind, containing 2000 experimentally characterized donor acceptor pairs. Using this dataset, we develop the Organic Photovoltaic Classifier (OPVC) to predict whether a material exhibits OPV behavior, and a hierarchical graph neural network that incorporates multi task learning and donor acceptor interaction modeling. This framework includes the Molecular Orbital Energy Estimator (MOE2) for predicting HOMO and LUMO energy levels, and the Photovoltaic Performance Predictor (P3) for estimating PCE. In addition, we introduce the Material Generative Pretrained Transformer (MatGPT) to produce synthetically accessible organic semiconductors, guided by a reinforcement learning strategy with three objective policy optimization. By linking molecular representation learning with performance prediction, our framework advances data driven discovery of high performance OPV materials.

---

## 150. Adaptive Mesh-Quantization for Neural PDE Solvers

**论文链接:** [http://arxiv.org/abs/2511.18474v1](http://arxiv.org/abs/2511.18474v1)

**作者:** Winfried van den Dool, Maksim Zhdanov, Yuki M. Asano, Max Welling

**发布时间:** 2025-11-23

### GPT解析

### 总结

本文提出了一种自适应网格量化方法，用于解决物理系统空间变化复杂性对神经PDE求解器的挑战，通过动态调整位宽实现资源优化分配，在多种任务上实现了显著性能提升。

### 背景

物理系统通常表现出空间变化的复杂性，这对神经PDE求解器构成挑战。图神经网络虽能处理不规则网格，但对所有节点应用统一计算资源，导致简单区域与复杂区域获得相同处理，造成资源分配效率低下。

### 目的

解决物理系统空间变化复杂性带来的资源分配低效问题，优化神经PDE求解器的计算资源利用。

### 方法

提出自适应网格量化方法：在网格节点、边和簇特征上进行空间自适应量化，动态调整量化模型的位宽；设计由轻量级辅助模型驱动的自适应位宽分配策略，识别输入网格中的高损失区域，为主模型中的高难度区域分配更多位宽。

### 主要发现

将该方法与MP-PDE和GraphViT模型集成后，在2D达西流、2D大规模非稳态流体动力学、3D稳态纳维-斯托克斯模拟和2D超弹性问题等多个任务上，相比均匀量化基线实现了持续的帕累托改进，在相同成本下性能提升高达50%。

### 结论

自适应网格量化方法能有效解决物理系统空间变化复杂性问题，通过动态资源分配优化计算资源利用率，在多种任务上实现显著性能提升。

### 翻译

Physical systems commonly exhibit spatially varying complexity, presenting a significant challenge for neural PDE solvers. While Graph Neural Networks can handle the irregular meshes required for complex geometries and boundary conditions, they still apply uniform computational effort across all nodes regardless of the underlying physics complexity. This leads to inefficient resource allocation where computationally simple regions receive the same treatment as complex phenomena. We address this challenge by introducing Adaptive Mesh Quantization: spatially adaptive quantization across mesh node, edge, and cluster features, dynamically adjusting the bit-width used by a quantized model. We propose an adaptive bit-width allocation strategy driven by a lightweight auxiliary model that identifies high-loss regions in the input mesh. This enables dynamic resource distribution in the main model, where regions of higher difficulty are allocated increased bit-width, optimizing computational resource utilization. We demonstrate our framework's effectiveness by integrating it with two state-of-the-art models, MP-PDE and GraphViT, to evaluate performance across multiple tasks: 2D Darcy flow, large-scale unsteady fluid dynamics in 2D, steady-state Navier-Stokes simulations in 3D, and a 2D hyper-elasticity problem. Our framework demonstrates consistent Pareto improvements over uniformly quantized baselines, yielding up to 50% improvements in performance at the same cost.


### 论文摘要

Physical systems commonly exhibit spatially varying complexity, presenting a significant challenge for neural PDE solvers. While Graph Neural Networks can handle the irregular meshes required for complex geometries and boundary conditions, they still apply uniform computational effort across all nodes regardless of the underlying physics complexity. This leads to inefficient resource allocation where computationally simple regions receive the same treatment as complex phenomena. We address this challenge by introducing Adaptive Mesh Quantization: spatially adaptive quantization across mesh node, edge, and cluster features, dynamically adjusting the bit-width used by a quantized model. We propose an adaptive bit-width allocation strategy driven by a lightweight auxiliary model that identifies high-loss regions in the input mesh. This enables dynamic resource distribution in the main model, where regions of higher difficulty are allocated increased bit-width, optimizing computational resource utilization. We demonstrate our framework's effectiveness by integrating it with two state-of-the-art models, MP-PDE and GraphViT, to evaluate performance across multiple tasks: 2D Darcy flow, large-scale unsteady fluid dynamics in 2D, steady-state Navier-Stokes simulations in 3D, and a 2D hyper-elasticity problem. Our framework demonstrates consistent Pareto improvements over uniformly quantized baselines, yielding up to 50% improvements in performance at the same cost.

---

## 151. Categorical Equivariant Deep Learning: Category-Equivariant Neural Networks and Universal Approximation Theorems

**论文链接:** [http://arxiv.org/abs/2511.18417v1](http://arxiv.org/abs/2511.18417v1)

**作者:** Yoshihiro Maruyama

**发布时间:** 2025-11-23

### GPT解析

### 总结

本研究开发了一种范畴等变神经网络(CENNs)的理论，统一了多种等变网络形式，并证明了其在连续等变变换空间中的通用逼近能力，扩展了等变深度学习的应用范围。

### 背景

现有的等变深度学习主要局限于群作用下的几何对称性，而实际应用中存在更多类型的对称性需要被考虑，如上下文对称性和组合对称性。

### 目的

开发一种统一的理论框架，能够涵盖群/群oid、偏序集/格、图和胞腔层等多种结构的等变性，并证明该框架下网络的通用逼近能力。

### 方法

将等变性表述为具有Radon测度的拓扑范畴中的自然性，在范畴设置中构建线性和非线性层，并通过范畴理论统一不同结构的等变网络。

### 主要发现

在一般设置中证明了等变通用逼近定理：有限深度CENNs类在连续等变变换空间中是稠密的；为群/群oid、偏序集/格、图和胞腔层系统推导了通用逼近定理。

### 结论

范畴等变深度学习扩展了等变深度学习的视野，超越了传统的群作用，能够处理几何对称性以及更广泛的上下文和组合对称性。

### 翻译

我们开发了一种范畴等变神经网络(CENNs)的理论，统一了群/群oid等变网络、偏序集/格等变网络、图和层神经网络的等变性。等变性被表述为具有Radon测度的拓扑范畴中的自然性，在范畴设置中表述线性和非线性层。我们在一般设置中证明了等变通用逼近定理：有限深度CENNs类在连续等变变换空间中是稠密的。我们为群/群oid、偏序集/格、图和胞腔层实例化该框架，以系统方式推导它们的通用逼近定理。范畴等变深度学习因此使我们能够扩展等变深度学习的视野，超越群作用，不仅包括几何对称性，还包括上下文和组合对称性。


### 论文摘要

We develop a theory of category-equivariant neural networks (CENNs) that unifies group/groupoid-equivariant networks, poset/lattice-equivariant networks, graph and sheaf neural networks. Equivariance is formulated as naturality in a topological category with Radon measures, formulating linear and nonlinear layers in the categorical setup. We prove the equivariant universal approximation theorem in the general setting: the class of finite-depth CENNs is dense in the space of continuous equivariant transformations. We instantiate the framework for groups/groupoids, posets/lattices, graphs and cellular sheaves, deriving universal approximation theorems for them in a systematic manner. Categorical equivariant deep learning thus allows us to expand the horizons of equivariant deep learning beyond group actions, encompassing not only geometric symmetries but also contextual and compositional symmetries.

---

## 152. Pre-training Graph Neural Networks on 2D and 3D Molecular Structures by using Multi-View Conditional Information Bottleneck

**论文链接:** [http://arxiv.org/abs/2511.18404v1](http://arxiv.org/abs/2511.18404v1)

**作者:** Van Thuy Hoang, O-Joun Lee

**发布时间:** 2025-11-23

### GPT解析

### 总结

该研究提出了一种名为MVCIB（多视图条件信息瓶颈）的框架，用于在2D和3D分子结构上进行自监督预训练，解决了多视图分子学习中的两个主要挑战。

### 背景

现有分子图预训练策略尝试使用2D和3D分子视图作为输入和自监督信号，主要对齐图级表示，但在解决多视图分子学习的两个主要挑战方面存在局限：发现两个视图间的共享信息同时减少视图特定信息，以及识别和对齐重要子结构以增强跨视图一致性和模型表达能力。

### 目的

解决多视图分子学习的两个主要挑战：1)发现视图间共享信息同时减少视图特定信息；2)识别和对齐重要子结构，增强跨视图一致性和模型表达能力。

### 方法

提出MVCIB框架，利用一个视图作为上下文条件指导对应视图的表示学习；利用关键子结构（如官能团和自我网络）作为视图间锚点；提出跨注意力机制捕获子结构间细粒度相关性，实现跨视图子图对齐。

### 主要发现

在四个分子领域的实验表明，MVCIB在预测性能和可解释性方面均优于基线方法；MVCIB实现了3d Weisfeiler-Lehman表达能力，能区分非同构图和具有相同2D连接性的不同3D几何结构（如同分异构体）。

### 结论

MVCIB框架通过条件信息瓶颈原则和跨注意力机制，有效解决了多视图分子学习的挑战，实现了更好的预测性能和可解释性。

### 翻译

最近的分子图预训练策略试图使用2D和3D分子视图作为输入和自监督信号，主要对齐图级表示。然而，现有研究在解决多视图分子学习的两个主要挑战方面仍然有限：(1)发现两个视图之间的共享信息，同时减少视图特定的信息；(2)识别和对齐重要的子结构，例如官能团，这对于增强跨视图一致性和模型表达能力至关重要。为了解决这些挑战，我们提出了一个多视图条件信息瓶颈框架，称为MVCIB，用于在自监督设置下对2D和3D分子结构预训练图神经网络。我们的想法是在MVCIB原则下发现共享信息，同时从每个视图中最小化无关特征，该原则使用一个视图作为上下文条件来指导其对应视图的表示学习。为了增强跨视图的语义和结构一致性，我们利用关键子结构，例如官能团和自我网络，作为两个视图之间的锚点。然后，我们提出了一种跨注意力机制，捕获子结构之间的细粒度相关性，以实现跨视图的子图对齐。在四个分子领域的广泛实验表明，MVCIB在预测性能和可解释性方面均一致优于基线方法。此外，MVCIB实现了3d Weisfeiler-Lehman表达能力，不仅能区分非同构图，还能区分具有相同2D连接性的不同3D几何结构，如同分异构体。


### 论文摘要

Recent pre-training strategies for molecular graphs have attempted to use 2D and 3D molecular views as both inputs and self-supervised signals, primarily aligning graph-level representations. However, existing studies remain limited in addressing two main challenges of multi-view molecular learning: (1) discovering shared information between two views while diminishing view-specific information and (2) identifying and aligning important substructures, e.g., functional groups, which are crucial for enhancing cross-view consistency and model expressiveness. To solve these challenges, we propose a Multi-View Conditional Information Bottleneck framework, called MVCIB, for pre-training graph neural networks on 2D and 3D molecular structures in a self-supervised setting. Our idea is to discover the shared information while minimizing irrelevant features from each view under the MVCIB principle, which uses one view as a contextual condition to guide the representation learning of its counterpart. To enhance semantic and structural consistency across views, we utilize key substructures, e.g., functional groups and ego-networks, as anchors between the two views. Then, we propose a cross-attention mechanism that captures fine-grained correlations between the substructures to achieve subgraph alignment across views. Extensive experiments in four molecular domains demonstrated that MVCIB consistently outperforms baselines in both predictive performance and interpretability. Moreover, MVCIB achieved the 3d Weisfeiler-Lehman expressiveness power to distinguish not only non-isomorphic graphs but also different 3D geometries that share identical 2D connectivity, such as isomers.

---

## 153. Predicting the Thermal Behavior of Semiconductor Defects with Equivariant Neural Networks

**论文链接:** [http://arxiv.org/abs/2511.18398v1](http://arxiv.org/abs/2511.18398v1)

**作者:** Xiangzhou Zhu, Patrick Rinke, David A. Egger

**发布时间:** 2025-11-23

### GPT解析

### 总结

本文提出了一种基于神经网络的框架，用于高效研究有限温度下缺陷半导体的电子性质，通过整合两个先进的等变图神经网络，实现了与密度泛函理论相当的精度但计算成本大幅降低。

### 背景

缺陷的存在强烈影响半导体行为，但即使使用密度泛函理论，预测有限温度下缺陷材料的电子性质仍然计算成本高昂，这主要是因为模拟单元中的原子数量多和热可达构型数量大。

### 目的

开发一个基于神经网络的框架，以高效研究有限温度下缺陷半导体的电子性质。

### 方法

采用主动学习方法，整合两个先进的等变图神经网络：MACE用于原子能量和力的计算，DeepH-E3用于电子哈密顿量的计算，并聚焦于GaAs中的代表性点缺陷进行研究。

### 主要发现

所提出的方法计算精度可与密度泛函理论相媲美，但计算成本仅为其中的一小部分；能够从更大尺度的分子动力学轨迹直接预测缺陷GaAs的温度依赖带隙，精度达到几十meV。

### 结论

等变神经网络在复杂、动态演化的材料中进行精确原子尺度预测具有巨大潜力。

### 翻译

缺陷的存在强烈影响半导体行为。然而，即使使用密度泛函理论，预测有限温度下缺陷材料的电子性质仍然计算成本高昂，这是由于模拟单元中的大量原子和众多热可达构型。在此，我们提出了一个基于神经网络的框架，用于高效研究有限温度下缺陷半导体的电子性质。我们开发了一种主动学习方法，整合了两个先进的等变图神经网络：用于原子能量和力的MACE和用于电子哈密顿量的DeepH-E3。聚焦于GaAs中的代表性点缺陷，我们证明了其计算精度可与密度泛函理论相媲美，但计算成本仅为其中的一小部分，能够从更大尺度的分子动力学轨迹直接预测缺陷GaAs的温度依赖带隙，精度达到几十meV。我们的结果突显了等变神经网络在复杂、动态演化的材料中进行精确原子尺度预测的潜力。


### 论文摘要

The presence of defects strongly influences semiconductor behavior. However, predicting the electronic properties of defective materials at finite temperatures remains computationally expensive even with density functional theory due to the large number of atoms in the simulation cell and the multitude of thermally accessible configurations. Here, we present a neural network-based framework to investigate the electronic properties of defective semiconductors at finite temperatures efficiently. We develop an active learning approach that integrates two advanced equivariant graph neural networks: MACE for atomic energies and forces and DeepH-E3 for the electronic Hamiltonian. Focusing on representative point defects in GaAs, we demonstrate computational accuracy comparable to density functional theory at a fraction of the computational cost, predicting the temperature-dependent band gap of defective GaAs directly from larger scale molecular dynamics trajectories with an accuracy of few tens of meV. Our results highlight the potential of equivariant neural networks for accurate atomic-scale predictions in complex, dynamically evolving materials.

---

## 154. Brain-MGF: Multimodal Graph Fusion Network for EEG-fMRI Brain Connectivity Analysis Under Psilocybin

**论文链接:** [http://arxiv.org/abs/2511.18325v1](http://arxiv.org/abs/2511.18325v1)

**作者:** Sin-Yee Yap, Fuad Noman, Junn Yong Loo, Devon Stoliker, Moein Khajehnejad, Raphaël C. -W. Phan, David L. Dowe, Adeel Razi, Chee-Ming Ting

**发布时间:** 2025-11-23

**备注:** 5 pages

### GPT解析

### 总结

该研究提出了一种名为Brain-MGF的多模态图融合网络，用于联合分析EEG-fMRI连接性，成功区分了裸盖菇素条件下的不同大脑状态，证明了自适应图融合能有效整合互补的多模态脑成像信息。

### 背景

迷幻剂如裸盖菇素会重新组织大规模大脑连接，但这些变化如何在电生理(EEG)和血流动力学(fMRI)网络中反映仍然不清楚。

### 目的

开发一种多模态图融合网络(Brain-MGF)，用于联合EEG-fMRI连接性分析，以揭示迷幻剂对大脑连接的影响。

### 方法

为每种模态构建具有偏相关边和Pearson特征分布节点特征的图，通过图卷积学习受试者级别嵌入，使用自适应softmax门融合具有样本特定权重的模态，并利用世界上最大的单一站点裸盖菇素数据集PsiConnect进行测试。

### 主要发现

Brain-MGF能够区分冥想和休息状态下的裸盖菇素和非裸盖菇素条件；融合方法优于单模态和非自适应变体，在冥想状态下达到74.0%的准确率和76.5%的F1分数，在休息状态下达到76.0%的准确率和85.8%的ROC-AUC；UMAP可视化显示融合嵌入的类别分离更清晰。

### 结论

自适应图融合有效整合了互补的EEG-fMRI信息，为表征裸盖菇素引起的大规模神经组织改变提供了可解释的框架。

### 翻译

迷幻剂，如裸盖菇素，会重新组织大规模大脑连接，但这些变化如何在电生理(脑电图，EEG)和血流动力学(功能性磁共振成像，fMRI)网络中反映仍然不清楚。我们提出了Brain-MGF，一种用于联合EEG-fMRI连接性分析的多模态图融合网络。对于每种模态，我们构建具有偏相关边和Pearson特征分布节点特征的图，并通过图卷积学习受试者级别的嵌入。然后，自适应softmax门融合具有样本特定权重的模态，以捕获上下文依赖的贡献。使用世界上最大的单一站点裸盖菇素数据集PsiConnect，Brain-MGF能够区分冥想和休息状态下的裸盖菇素和非裸盖菇素条件。融合优于单模态和非自适应变体，在冥想状态下达到74.0%的准确率和76.5%的F1分数，在休息状态下达到76.0%的准确率和85.8%的ROC-AUC。UMAP可视化显示融合嵌入的类别分离更清晰。这些结果表明，自适应图融合有效整合了互补的EEG-fMRI信息，为表征裸盖菇素引起的大规模神经组织改变提供了可解释的框架。


### 论文摘要

Psychedelics, such as psilocybin, reorganise large-scale brain connectivity, yet how these changes are reflected across electrophysiological (electroencephalogram, EEG) and haemodynamic (functional magnetic resonance imaging, fMRI) networks remains unclear. We present Brain-MGF, a multimodal graph fusion network for joint EEG-fMRI connectivity analysis. For each modality, we construct graphs with partial-correlation edges and Pearson-profile node features, and learn subject-level embeddings via graph convolution. An adaptive softmax gate then fuses modalities with sample-specific weights to capture context-dependent contributions. Using the world's largest single-site psilocybin dataset, PsiConnect, Brain-MGF distinguishes psilocybin from no-psilocybin conditions in meditation and rest. Fusion improves over unimodal and non-adaptive variants, achieving 74.0% accuracy and 76.5% F1 score on meditation, and 76.0% accuracy with 85.8% ROC-AUC on rest. UMAP visualisations reveal clearer class separation for fused embeddings. These results indicate that adaptive graph fusion effectively integrates complementary EEG-fMRI information, providing an interpretable framework for characterising psilocybin-induced alterations in large-scale neural organisation.

---

## 155. 论文ID: 2511.18297v1

**错误:** 无法读取论文信息 - [Errno 2] No such file or directory: 'storage/result/2511.18297v1.json'

---

## 156. Large Language Model Enhanced Graph Invariant Contrastive Learning for Out-of-Distribution Recommendation

**论文链接:** [http://arxiv.org/abs/2511.18282v1](http://arxiv.org/abs/2511.18282v1)

**作者:** Jiahao Liang, Haoran Yang, Xiangyu Zhao, Zhiwen Yu, Mianjie Li, Chuan Shi, Kaixiang Yang

**发布时间:** 2025-11-23

### GPT解析

### 总结

本文提出了一种名为InvGCLLM的创新因果学习框架，用于解决图推荐系统中的分布外泛化问题。该框架整合了数据驱动模型和知识驱动的大型语言模型，通过因果置信度分数指导图精炼，最终实现对虚假相关性具有韧性的表示学习。

### 背景

图推荐系统中的分布外泛化是一个重大挑战。传统图神经网络算法常因学习虚假的环境相关性而非稳定的因果关系，导致在分布变化时性能大幅下降。

### 目的

解决如何将大型语言模型的世界知识与特定图的精细拓扑结构有效集成，以解决分布外推荐问题。

### 方法

提出Invariant Graph Contrastive Learning with LLMs for Out-of-Distribution Recommendation (InvGCLLM)框架，包含三个主要步骤：1)使用数据驱动的不变性学习模型生成用户-项目交互的因果置信度分数；2)利用LLM进行有针对性的图精炼，剪除虚假连接并增强缺失的因果链接；3)使用结构净化的图进行因果引导的对比学习。

### 主要发现

在四个公共数据集上的实验表明，InvGCLLM在分布外推荐方面取得了显著改进，持续优于最先进的基线方法。

### 结论

InvGCLLM通过整合数据驱动模型和大型语言模型的优势，有效解决了图推荐系统中的分布外泛化问题，提高了模型在分布变化情况下的性能。

### 翻译

分布外泛化已成为图推荐系统中的一个重大挑战。传统的图神经网络算法常常失败，因为它们学习的是虚假的环境相关性而非稳定的因果关系，导致在分布变化时性能大幅下降。虽然最近大型语言模型的进展因其丰富的世界知识和推理能力提供了有希望的途径，但如何将这种知识与特定图的精细拓扑结构有效集成以解决OOD问题仍然是一个重大挑战。为解决这些问题，我们提出了Invariant Graph Contrastive Learning with LLMs for Out-of-Distribution Recommendation (InvGCLLM)，这是一个创新的因果学习框架，协同整合了数据驱动模型和知识驱动LLM的优势。我们的框架首先采用数据驱动的不变性学习模型为每个用户-项目交互生成因果置信度分数。然后这些分数指导LLM进行有针对性的图精炼，利用其世界知识剪除虚假连接并增强缺失的因果链接。最后，结构净化的图为因果引导的对比学习目标提供鲁棒监督，使模型能够学习对虚假相关性具有韧性的表示。在四个公共数据集上进行的实验表明，InvGCLLM在分布外推荐方面取得了显著改进，持续优于最先进的基线方法。


### 论文摘要

Out-of-distribution (OOD) generalization has emerged as a significant challenge in graph recommender systems. Traditional graph neural network algorithms often fail because they learn spurious environmental correlations instead of stable causal relationships, leading to substantial performance degradation under distribution shifts. While recent advancements in Large Language Models (LLMs) offer a promising avenue due to their vast world knowledge and reasoning capabilities, effectively integrating this knowledge with the fine-grained topology of specific graphs to solve the OOD problem remains a significant challenge. To address these issues, we propose {$\textbf{Inv}$ariant $\textbf{G}$raph $\textbf{C}$ontrastive Learning with $\textbf{LLM}$s for Out-of-Distribution Recommendation (InvGCLLM)}, an innovative causal learning framework that synergistically integrates the strengths of data-driven models and knowledge-driven LLMs. Our framework first employs a data-driven invariant learning model to generate causal confidence scores for each user-item interaction. These scores then guide an LLM to perform targeted graph refinement, leveraging its world knowledge to prune spurious connections and augment missing causal links. Finally, the structurally purified graphs provide robust supervision for a causality-guided contrastive learning objective, enabling the model to learn representations that are resilient to spurious correlations. Experiments conducted on four public datasets demonstrate that InvGCLLM achieves significant improvements in out-of-distribution recommendation, consistently outperforming state-of-the-art baselines.

---

## 157. 论文ID: 2511.18150v1

**错误:** 无法读取论文信息 - [Errno 2] No such file or directory: 'storage/result/2511.18150v1.json'

---

## 158. 论文ID: 2511.19476v1

**错误:** 无法读取论文信息 - [Errno 2] No such file or directory: 'storage/result/2511.19476v1.json'

---

## 159. 论文ID: 2511.20615v1

**错误:** 无法读取论文信息 - [Errno 2] No such file or directory: 'storage/result/2511.20615v1.json'

---

## 160. Safe and Stable Neural Network Dynamical Systems for Robot Motion Planning

**论文链接:** [http://arxiv.org/abs/2511.20593v1](http://arxiv.org/abs/2511.20593v1)

**作者:** Allen Emmanuel Binny, Mahathi Anand, Hugo T. M. Kussaba, Lingyun Chen, Shreenabh Agrawal, Fares J. Abu-Dakka, Abdalla Swikir

**发布时间:** 2025-11-25

### GPT解析

### 总结

本文提出了一种名为S²-NNDS的安全稳定神经网络动力学系统框架，能够从演示中学习安全、稳定的机器人运动，同时学习神经李雅普诺夫稳定性和屏障安全证书。

### 背景

从演示中学习安全稳定的机器人运动仍然是一个挑战，特别是在涉及动态、障碍物丰富的复杂非线性任务中。

### 目的

开发一种学习框架，能够同时学习表达性神经动力学系统以及神经李雅普诺夫稳定性和屏障安全证书。

### 方法

提出Safe and Stable Neural Network Dynamical Systems (S²-NNDS)，利用神经网络捕获复杂机器人运动，并通过学习证书中的分裂保形预测提供概率保证，不同于具有限制性多项式参数化的传统方法。

### 主要发现

在LASA手写数据集和从Franka Emika Panda机器人记录的演示数据等2D和3D数据集上的实验验证了S²-NNDS的有效性，能够从可能不安全的演示中学习鲁棒、安全、稳定的运动。

### 结论

S²-NNDS框架能够有效地从演示中学习安全、稳定的机器人运动，特别适用于复杂、非线性、动态且障碍物丰富的环境。

### 翻译

从演示中学习安全稳定的机器人运动仍然是一个挑战，特别是在涉及动态、障碍物丰富的复杂非线性任务中。在本文中，我们提出了安全稳定的神经网络动力学系统S²-NNDS，这是一种从演示中学习的框架，同时学习表达性神经动力学系统和神经李雅普诺夫稳定性与屏障安全证书。与传统具有限制性多项式参数化的方法不同，S²-NNDS利用神经网络捕获复杂的机器人运动，并通过学习证书中的分裂保形预测提供概率保证。在LASA手写和各种2D和3D数据集上的实验结果，包括从Franka Emika Panda机器人记录的演示数据，验证了S²-NNDS从可能不安全的演示中学习鲁棒、安全、稳定运动的有效性。


### 论文摘要

Learning safe and stable robot motions from demonstrations remains a challenge, especially in complex, nonlinear tasks involving dynamic, obstacle-rich environments. In this paper, we propose Safe and Stable Neural Network Dynamical Systems S$^2$-NNDS, a learning-from-demonstration framework that simultaneously learns expressive neural dynamical systems alongside neural Lyapunov stability and barrier safety certificates. Unlike traditional approaches with restrictive polynomial parameterizations, S$^2$-NNDS leverages neural networks to capture complex robot motions providing probabilistic guarantees through split conformal prediction in learned certificates. Experimental results on various 2D and 3D datasets -- including LASA handwriting and demonstrations recorded kinesthetically from the Franka Emika Panda robot -- validate S$^2$-NNDS effectiveness in learning robust, safe, and stable motions from potentially unsafe demonstrations.

---

## 161. From Features to States: Data-Driven Selection of Measured State Variables via RFE-DMDc

**论文链接:** [http://arxiv.org/abs/2511.20552v1](http://arxiv.org/abs/2511.20552v1)

**作者:** Haoyu Wang, Andrea Alfonsi, Roberto Ponciroli, Richard Vilim

**发布时间:** 2025-11-25

### GPT解析

### 总结

本研究提出了一种名为RFE-DMDc的监督、数据驱动工作流程，用于选择最优变量子集并构建状态空间模型，应用于工程系统的数字孪生。

### 背景

对于许多工程系统，基于第一性原理的模型识别不切实际，这促使了用于控制和诊断的数字孪生采用数据驱动方法。

### 目的

提出一种能够选择最小且具有物理意义的变量集，并通过带控制的动态模态分解推导线性状态空间模型的方法。

### 方法

使用递归特征消除(RFE)选择要监控的最小变量集，通过带控制的动态模态分解(DMDc)推导线性状态空间模型，包含跨子系统选择步骤以减轻特征掩盖现象，并实现GA-DMDc基线方法进行对比。

### 主要发现

在RLC基准测试和真实综合能源系统上，RFE-DMDc恢复的紧凑状态集(约10个变量)实现了与GA-DMDc相当的测试误差，但计算时间减少一个数量级；选择的变量保持清晰的物理解释；生成的模型具有竞争性的预测精度、计算效率和抗过拟合能力。

### 结论

RFE-DMDc是一种有效的方法，可以在保持模型性能的同时显著减少计算负担，并选择具有物理意义的变量。

### 翻译

在给定输入集下，动态系统的行为可以通过跟踪最优过程变量子集的状态变量来捕捉。然而，对于许多工程系统，基于第一性原理的模型识别不切实际，这促使了用于控制和诊断的数字孪生采用数据驱动方法。在本文中，我们提出了RFE-DMDc，一种监督的、数据驱动的工作流程，它使用递归特征消除(RFE)选择最小且具有物理意义的变量集进行监控，然后通过带控制的动态模态分解(DMDc)推导线性状态空间模型。该工作流程包含一个跨子系统选择步骤，以减轻多组件系统中的特征掩盖现象。为了验证结果，我们实现了一个GA-DMDc基线方法，在状态和输出的共同精度成本下联合优化状态集和模型拟合。在一个已知的RLC基准测试和一个具有多个热耦合组件和数千个候选变量的真实综合能源系统(IES)中，RFE-DMDc一致地恢复了紧凑的状态集(约10个变量)，实现了与GA-DMDc相当的测试误差，同时所需的计算时间减少了一个数量级。选择的变量在子系统中保持清晰的物理解释，生成的模型展示了竞争性的预测精度、计算效率和抗过拟合能力。


### 论文摘要

The behavior of a dynamical system under a given set of inputs can be captured by tracking the response of an optimal subset of process variables (\textit{state variables}). For many engineering systems, however, first-principles, model-based identification is impractical, motivating data-driven approaches for Digital Twins used in control and diagnostics. In this paper, we present RFE-DMDc, a supervised, data-driven workflow that uses Recursive Feature Elimination (RFE) to select a minimal, physically meaningful set of variables to monitor and then derives a linear state-space model via Dynamic Mode Decomposition with Control (DMDc). The workflow includes a cross-subsystem selection step that mitigates feature \textit{overshadowing} in multi-component systems. To corroborate the results, we implement a GA-DMDc baseline that jointly optimizes the state set and model fit under a common accuracy cost on states and outputs. Across a truth-known RLC benchmark and a realistic Integrated Energy System (IES) with multiple thermally coupled components and thousands of candidate variables, RFE-DMDc consistently recovers compact state sets (\(\approx 10\) variables) that achieve test errors comparable to GA-DMDc while requiring an order of magnitude less computational time. The selected variables retain clear physical interpretation across subsystems, and the resulting models demonstrate competitive predictive accuracy, computational efficiency, and robustness to overfitting.

---

## 162. Metric, inertially aligned monocular state estimation via kinetodynamic priors

**论文链接:** [http://arxiv.org/abs/2511.20496v1](http://arxiv.org/abs/2511.20496v1)

**作者:** Jiaxin Liu, Min Li, Wanting Xu, Liang Li, Jiaqi Yang, Laurent Kneip

**发布时间:** 2025-11-25

### GPT解析

### 总结

本文提出了一种柔性机器人系统的状态估计方法，通过结合变形力模型和连续时间B样条运动学模型，实现了对非刚性系统的鲁棒姿态估计。

### 背景

柔性机器人系统的状态估计具有显著挑战，特别是对于具有动态变形结构的平台，这些变形会使得传统的刚体假设失效。

### 目的

解决柔性机器人系统的状态估计问题，将现有的刚体姿态估计方法扩展到非刚性系统。

### 方法

基于两个核心假设：1) 使用多层感知机(MLP)学习单射变形力模型来捕获弹性特性；2) 采用连续时间B样条运动学模型解决平台的平滑运动问题；通过持续应用牛顿第二定律，建立视觉推导的轨迹加速度与预测的变形引起的加速度之间的物理联系。

### 主要发现

该方法不仅能在非刚性平台上实现鲁棒和精确的姿态估计，而且正确建模的平台物理特性会引发惯性传感特性。

### 结论

该方法在简单的弹簧相机系统上证明了可行性，并能够稳健解决单目视觉里程计中通常病态的度量尺度和重力恢复问题。

### 翻译

柔性机器人系统的精确状态估计具有重大挑战，特别是对于具有动态变形结构的平台，这些变形使得刚体假设失效。本文解决了这个问题，并允许将现有的刚体姿态估计方法扩展到非刚性系统。我们的方法基于两个核心假设：首先，弹性特性由单射变形力模型捕获，通过多层感知机高效学习；其次，我们使用连续时间B样条运动学模型解决平台的固有平滑运动。通过持续应用牛顿第二定律，我们的方法建立了视觉推导的轨迹加速度与预测的变形引起的加速度之间的物理联系。我们证明，我们的方法不仅能够在非刚性平台上实现鲁棒和精确的姿态估计，而且正确建模的平台物理特性会引发惯性传感特性。我们在一个简单的弹簧相机系统上证明了这种可行性，并展示了如何稳健解决单目视觉里程计中通常病态的度量尺度和重力恢复问题。


### 论文摘要

Accurate state estimation for flexible robotic systems poses significant challenges, particular for platforms with dynamically deforming structures that invalidate rigid-body assumptions. This paper tackles this problem and allows to extend existing rigid-body pose estimation methods to non-rigid systems. Our approach hinges on two core assumptions: first, the elastic properties are captured by an injective deformation-force model, efficiently learned via a Multi-Layer Perceptron; second, we solve the platform's inherently smooth motion using continuous-time B-spline kinematic models. By continuously applying Newton's Second Law, our method establishes a physical link between visually-derived trajectory acceleration and predicted deformation-induced acceleration. We demonstrate that our approach not only enables robust and accurate pose estimation on non-rigid platforms, but that the properly modeled platform physics instigate inertial sensing properties. We demonstrate this feasibility on a simple spring-camera system, and show how it robustly resolves the typically ill-posed problem of metric scale and gravity recovery in monocular visual odometry.

---

## 163. 论文ID: 2511.20457v1

**错误:** 无法读取论文信息 - [Errno 2] No such file or directory: 'storage/result/2511.20457v1.json'

---

## 164. Predicting Friction under Vastly Different Lubrication Scenarios

**论文链接:** [http://arxiv.org/abs/2511.20342v1](http://arxiv.org/abs/2511.20342v1)

**作者:** Yulong Li, Peter Gumbsch, Christian Greiner

**发布时间:** 2025-11-25

### GPT解析

### 总结

该研究揭示了摩擦学系统对微小变化的极端敏感性导致实验中难以重现摩擦行为，并通过利用表面形貌的中等尺度特征和机器学习建立了准确预测高摩擦区域的模型。

### 背景

摩擦力在日常生活中无处不在，从纳米级机器到大型工程组件都有涉及。理解摩擦机制对于预测和控制摩擦至关重要，这也是实现碳中和的重要步骤。

### 目的

探究摩擦学系统难以在实验中重现行为的原因，并开发能够准确预测高摩擦区域的模型。

### 方法

研究表面形貌的中等尺度特征和不对准引起的振荡（这些信息通常被过滤掉或忽视）；建立预测模型并利用机器学习增强其性能。

### 主要发现

摩擦学系统对微小变化（如表面形貌）具有极端敏感性，即使是精细制备的表面也存在微小差异，这些差异相互作用导致摩擦和磨损的统计显著变化，产生系统级混沌行为。

### 结论

通过利用表面形貌的中等尺度特征和不对准引起的振荡，可以建立准确预测高摩擦区域的模型，该模型性能可通过机器学习进一步增强。

### 翻译

摩擦力在日常生活中无处不在，从纳米级机器到大型工程组件。通过研究系统参数与摩擦行为之间的复杂相互作用，科学家试图揭示摩擦的基本机制，这是实现预测和控制摩擦的关键步骤，也是迈向碳中和的重要一步。然而，在实验中重现摩擦行为是出了名的困难。研究表明，这一挑战源于摩擦学系统对微小变化的极端敏感性，例如通常被认为控制良好的表面形貌。即使按照半导体行业标准精心准备表面并减少不对准引起的振荡，微小的变化仍然存在并相互作用。这些微小的初始差异导致摩擦和磨损的统计显著变化，产生系统级的混沌行为。然而，通过利用表面形貌的中等尺度特征和不对准引起的振荡——这些信息通常被过滤掉或忽视——我们建立了一个模型，可以在完全不同的润滑条件下准确预测高摩擦区域，其性能通过机器学习得到了进一步增强。


### 论文摘要

Friction is ubiquitous in daily life, from nanoscale machines to large engineering components. By probing the intricate interplay between system parameters and frictional behavior, scientists seek to unveil the underlying mechanisms that enable prediction and control of friction - an essential step toward carbon neutrality. Yet, reproducing frictional behavior in experiments is notoriously difficult. Here, we show that this challenge stems from the extreme sensitivity of tribological systems to tiny variations, e.g. in surface topography, typically presumed well- controlled. Even after meticulous surface preparation to semiconductor-industry standards and curtailing misalignment-induced oscillations, subtle variations remain and interact. In turn, such minute initial differences lead to statistically significant variations in friction and wear, giving rise to system-level chaotic behavior. Yet, by leveraging mid-scale features of surface topography and misalignment-induced oscillations - information often filtered out or overlooked - we established a model that accurately predicts high-friction regions under vastly different lubrication scenarios, with its performance further enhanced by machine learning.

---

## 165. 论文ID: 2511.20339v1

**错误:** 无法读取论文信息 - [Errno 2] No such file or directory: 'storage/result/2511.20339v1.json'

---

## 166. 3D Motion Perception of Binocular Vision Target with PID-CNN

**论文链接:** [http://arxiv.org/abs/2511.20332v1](http://arxiv.org/abs/2511.20332v1)

**作者:** Shi Jiazhao, Pan Pan, Shi Haotian

**发布时间:** 2025-11-25

**备注:** 7 pages, 9 figures, 2 tables

### GPT解析

### 总结

本文提出了一种基于PID原理的卷积神经网络，用于感知双视觉目标的三维运动信息，能够提供实时的三维坐标、速度和加速度数据，并具有基本的时空感知能力。

### 背景

从PID角度理解神经网络拟合非线性问题的能力，将单层神经网络视为使用二阶差分方程和非线性来描述局部问题，多层网络通过多次组合将原始表示转换到期望表示。

### 目的

设计一个能够准确感知三维运动信息的神经网络，提供实时的坐标、速度和加速度数据，并分析高维卷积和PID信息在提高计算效率和特征空间利用率方面的优势。

### 方法

设计了一个17层、41.3万参数的PID卷积神经网络，通过连接和池化实现特征重用方法，使用模拟随机移动球数据集进行训练和测试。

### 主要发现

实验结果表明网络的预测精度接近输入图像分辨率所能表示的上限，分析了实验结果、误差以及现有不足，探讨了高维卷积在提高计算效率和特征空间利用率方面的优势，以及使用PID信息实现记忆和注意力机制的潜在优势。

### 结论

PID卷积神经网络能够有效感知三维运动信息，高维卷积和PID信息在提高计算效率和特征空间利用率方面具有优势，为未来研究和应用提供了方向。

### 翻译

本文训练了一个用于感知双视觉目标三维运动信息的网络，可以提供实时的三维坐标、速度和加速度，并具有基本的时空感知能力。从PID角度理解了神经网络拟合非线性问题的能力，将单层神经网络视为使用二阶差分方程和非线性来描述局部问题。多层网络通过多次这样的组合将原始表示逐渐转换到期望表示。分析了设计神经网络的一些参考原则，设计了一个相对较小的PID卷积神经网络，总共有17层和413,000个参数。通过连接和池化实现了一种简单但实用的特征重用方法。使用模拟的随机移动球数据集训练和测试网络，实验结果表明预测精度接近输入图像分辨率所能表示的上限。分析了实验结果和误差，以及现有的不足和可能的改进方向。最后讨论了高维卷积在提高计算效率和特征空间利用率方面的优势，以及使用PID信息实现记忆和注意力机制的潜在优势。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决使用双目视觉系统实时感知目标三维运动信息（包括坐标、速度和加速度）的问题。这个问题在现实中很重要，因为传统测量方法存在局限性：直接测量工具难以跟踪高速运动目标且回应慢；回波测量（如雷达）需要主动发射信号且设备约束大；视觉测量中的校准过程繁琐耗时，影响实时性。而双目视觉相比单目能提供三维信息，相比多目更高效，在工业应用中具有实用价值。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者从PID控制理论角度理解神经网络拟合非线性问题的能力，将单层神经网络视为使用二阶差分方程和非线性描述局部问题，多层网络通过多次组合将原始表示转换为期望表示。设计网络时考虑了卷积核大小为3的优势（从PID角度解释）、良好的表示应保留感兴趣信息并丢弃冗余、使用多步变换而非增加维度、PReLU非线性比ReLU更优、平均池化比最大池化保留更多信息等原则。借鉴了PID控制理论、计算机视觉中的卷积神经网络架构、Adam优化器、MSE损失函数等现有工作，并参考了特征重用和残差连接等技术。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是将PID控制理论融入卷积神经网络设计，通过比例、积分、微分变换提取特征，设计轻量级网络（17层，41.3万参数）实现三维运动感知，使用特征重用机制提高效率，采用残差计算优化速度和加速度预测。整体流程：1)输入处理（接收双目视觉四维张量，使用RGB的B通道并减去背景）；2)网络架构（7个构建块进行特征提取，每块包含两次Conv-BN-PRelu变换，通过拼接和池化实现特征重用）；3)输出处理（分割展平特征图为三个向量，计算坐标、速度和加速度，使用残差过程）；4)三阶段训练（单帧预测坐标、两帧预测坐标和速度、三帧预测坐标、速度和加速度）。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点：1)PID-CNN架构，从PID控制理论理解卷积操作；2)特征重用机制，通过拼接和池化实现高效特征利用；3)残差计算方法，利用已训练坐标信息优化速度和加速度预测；4)高维卷积潜在优势分析。相比之前工作的不同：无需繁琐校准，提供端到端解决方案；更轻量级网络设计；结合PID理论与卷积神经网络；特征重用机制提高效率；实时性能更好（每秒247次测量）；直接从图像提取三维运动信息，无需预先校准双目系统几何关系。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '这篇论文提出了一种结合PID控制理论的轻量级卷积神经网络，实现了双目视觉目标的三维运动实时感知，无需繁琐校准且精度接近输入图像分辨率上限。'}


### 论文摘要

This article trained a network for perceiving three-dimensional motion information of binocular vision target, which can provide real-time three-dimensional coordinate, velocity, and acceleration, and has a basic spatiotemporal perception capability. Understood the ability of neural networks to fit nonlinear problems from the perspective of PID. Considered a single-layer neural network as using a second-order difference equation and a nonlinearity to describe a local problem. Multilayer networks gradually transform the raw representation to the desired representation through multiple such combinations. Analysed some reference principles for designing neural networks. Designed a relatively small PID convolutional neural network, with a total of 17 layers and 413 thousand parameters. Implemented a simple but practical feature reuse method by concatenation and pooling. The network was trained and tested using the simulated randomly moving ball datasets, and the experimental results showed that the prediction accuracy was close to the upper limit that the input image resolution can represent. Analysed the experimental results and errors, as well as the existing shortcomings and possible directions for improvement. Finally, discussed the advantages of high-dimensional convolution in improving computational efficiency and feature space utilization. As well as the potential advantages of using PID information to implement memory and attention mechanisms.

---

## 167. AD-R1: Closed-Loop Reinforcement Learning for End-to-End Autonomous Driving with Impartial World Models

**论文链接:** [http://arxiv.org/abs/2511.20325v1](http://arxiv.org/abs/2511.20325v1)

**作者:** Tianyi Yan, Tao Tang, Xingtai Gui, Yongkang Li, Jiasen Zhesng, Weiyao Huang, Lingdong Kong, Wencheng Han, Xia Zhou, Xueyang Zhang, Yifei Zhan, Kun Zhan, Cheng-zhong Xu, Jianbing Shen

**发布时间:** 2025-11-25

### GPT解析

### 总结

本文提出了一种基于公平世界模型的端到端自动驾驶框架，通过反事实合成方法教导模型诚实地预测危险，显著提高了自动驾驶系统的安全性。

### 背景

端到端自动驾驶模型有望直接从传感器数据中学习复杂行为，但在处理安全和长尾事件方面面临关键挑战。强化学习(RL)为克服这些局限提供了有希望的路径，但在自动驾驶领域的成功一直难以实现。研究者发现阻碍这一进展的根本缺陷：用于强化学习的世界模型中存在根深蒂固的乐观偏见。

### 目的

解决自动驾驶领域中强化学习的根本缺陷（世界模型的乐观偏见），开发一个能够诚实地预测危险的世界模型，提高自动驾驶系统的安全性，减少安全违规。

### 方法

引入一个基于公平世界模型的后训练策略精炼框架；使用新颖的数据合成管道——反事实合成，系统地生成丰富的可能碰撞和道路外事件课程；将公平世界模型集成到闭环强化学习框架中，作为内部批评者；在精炼过程中，代理查询批评者来'想象'候选行动的结果。

### 主要发现

公平世界模型在预测失败方面显著优于基线方法；当用作批评者时，它能够在具有挑战性的模拟中大幅减少安全违规；教导模型'想象'危险是构建真正安全和智能自主代理的关键步骤。

### 结论

教导模型诚实地预测危险对于构建安全的自动驾驶系统至关重要；所提出的公平世界模型和反事实合成方法有效解决了强化学习在自动驾驶中的根本缺陷；该方法为构建真正安全和智能的自主代理提供了关键步骤。

### 翻译

端到端自动驾驶模型有望直接从传感器数据中学习复杂行为，但在安全性和处理长尾事件方面面临关键挑战。强化学习(RL)为克服这些局限提供了有希望的路径，但其在自动驾驶领域的成功一直难以实现。我们确定了一个阻碍这一进展的根本缺陷：用于强化学习的世界模型中存在根深蒂固的乐观偏见。为解决这一问题，我们引入了一个围绕公平世界模型构建的后训练策略精炼框架。我们的主要贡献是教导这个模型诚实地认识危险。我们通过新颖的反事实合成数据合成管道实现这一点，该管道系统地生成丰富的可能碰撞和道路外事件课程。这使模型从被动的场景补全者转变为真实的预测器，保持对行动与结果之间因果联系的忠实。然后我们将这个公平世界模型集成到我们的闭环强化学习框架中，它作为内部批评者发挥作用。在精炼过程中，代理查询批评者来'想象'候选行动的结果。我们通过大量实验（包括在一个新的风险预见基准测试上）证明，我们的模型在预测失败方面显著优于基线方法。因此，当用作批评者时，它能够在具有挑战性的模拟中大幅减少安全违规，证明教导模型想象危险是构建真正安全和智能自主代理的关键一步。


### 论文摘要

End-to-end models for autonomous driving hold the promise of learning complex behaviors directly from sensor data, but face critical challenges in safety and handling long-tail events. Reinforcement Learning (RL) offers a promising path to overcome these limitations, yet its success in autonomous driving has been elusive. We identify a fundamental flaw hindering this progress: a deep seated optimistic bias in the world models used for RL. To address this, we introduce a framework for post-training policy refinement built around an Impartial World Model. Our primary contribution is to teach this model to be honest about danger. We achieve this with a novel data synthesis pipeline, Counterfactual Synthesis, which systematically generates a rich curriculum of plausible collisions and off-road events. This transforms the model from a passive scene completer into a veridical forecaster that remains faithful to the causal link between actions and outcomes. We then integrate this Impartial World Model into our closed-loop RL framework, where it serves as an internal critic. During refinement, the agent queries the critic to ``dream" of the outcomes for candidate actions. We demonstrate through extensive experiments, including on a new Risk Foreseeing Benchmark, that our model significantly outperforms baselines in predicting failures. Consequently, when used as a critic, it enables a substantial reduction in safety violations in challenging simulations, proving that teaching a model to dream of danger is a critical step towards building truly safe and intelligent autonomous agents.

---

## 168. How Robot Kinematics Influence Human Performance in Virtual Robot-to-Human Handover Tasks

**论文链接:** [http://arxiv.org/abs/2511.20299v1](http://arxiv.org/abs/2511.20299v1)

**作者:** Róisín Keenan, Joost C. Dessing

**发布时间:** 2025-11-25

### GPT解析

### 总结

该研究使用虚拟现实技术探索人机交接任务中的交互，分析不同任务动力学和机器人运动学对人类表现的影响，发现人类从机器人提供的早期视觉信息和平滑轨迹中获益，建议设计人机交互时应考虑人类检测生物运动的自然能力。

### 背景

机器人技术的进步增加了将机器人系统集成到人类工作场所的可能性，突显了需要研究和优化协作环境中人机协调的必要性。

### 目的

探索交接任务中的人机交互，使用虚拟现实研究不同任务动力学和机器人运动学下人类运动表现的差异。

### 方法

采用基于VR的机器人交接模拟进行安全可控的评估，通过单独实验检验四个影响因素：任务开始和机器人运动同步控制、伙伴外观、机器人速度曲线以及旋转物体运动的时间。

### 主要发现

人类受益于机器人提供关于任务相关物体运动的早期和显著视觉信息，以及人类化平滑机器人轨迹；这些操作在不同程度上提高了预测准确性和交互同步性。

### 结论

人机交互应设计为允许人类利用其检测生物运动的自然能力，这可能减少昂贵的机器人计算需求或人类方面额外的认知适应。

### 翻译

机器人技术的最新进展增加了将机器人系统集成到涉及人类的工作场所的可能性，强调了需要在协作环境中检查和优化人机协调的必要性。本研究使用虚拟现实(VR)探索交接任务中的人机交互，研究不同任务动力学和机器人运动学下人类运动表现的差异。基于VR的机器人交接模拟提供了对人机交互的安全可控评估。在单独的实验中，检验了四个可能影响人类表现的因素：(1)对任务开始和机器人运动同步的控制（时间和时空同步）；(2)伙伴外观（人类与机器人）；(3)机器人速度曲线（最小加加速度、恒定速度、恒定加速度和双相）；以及(4)旋转物体运动的时间。各实验的发现强调人类受益于机器人提供关于任务相关物体运动的早期和显著视觉信息，以及人类化平滑机器人轨迹的优势。这些操作在不同程度上提高了交互期间的预测准确性和同步性。这表明人机交互应设计为允许人类利用其检测生物运动的自然能力，这可能反过来减少昂贵的机器人计算需求或人类方面额外的认知适应。


### 论文摘要

Recent advancements in robotics have increased the possibilities for integrating robotic systems into human-involved workplaces, highlighting the need to examine and optimize human-robot coordination in collaborative settings. This study explores human-robot interactions during handover tasks using Virtual Reality (VR) to investigate differences in human motor performance across various task dynamics and robot kinematics. A VR-based robot handover simulation afforded safe and controlled assessments of human-robot interactions. In separate experiments, four potential influences on human performance were examined (1) control over task initiation and robot movement synchrony (temporal and spatiotemporal); (2) partner appearance (human versus robotic); (3) robot velocity profiles (minimum jerk, constant velocity, constant acceleration, and biphasic); and (4) the timing of rotational object motion. Findings across experiments emphasize humans benefit from robots providing early and salient visual information about task-relevant object motion, and advantages of human-like smooth robot trajectories. To varying degrees, these manipulations improved predictive accuracy and synchronization during interaction. This suggests that human-robot interactions should be designed to allow humans to leverage their natural capabilities for detecting biological motion, which conversely may reduce the need for costly robotic computations or added cognitive adaptation on the human side.

---

## 169. Back to the Feature: Explaining Video Classifiers with Video Counterfactual Explanations

**论文链接:** [http://arxiv.org/abs/2511.20295v1](http://arxiv.org/abs/2511.20295v1)

**作者:** Chao Wang, Chengan Che, Xinyue Chen, Sophia Tsoka, Luis C. Garcia-Peraza-Herrera

**发布时间:** 2025-11-25

### GPT解析

### 总结

本文提出了一种名为Back To The Feature (BTTF)的优化框架，用于生成视频反事实解释(CFEs)，解决了现有方法无法生成时间连贯、平滑且物理合理的视频反事实解释的问题。

### 背景

反事实解释是对模型输入的最小且语义上有意义的修改，可以改变模型预测。现有的先进视觉反事实解释方法主要是为图像分类器设计的，而为视频分类器生成反事实解释的研究仍然很少。

### 目的

开发一种能够生成物理合理、时间连贯且具有平滑运动轨迹的反事实视频的方法，解决现有基于图像的反事实解释方法无法生成时间连贯、平滑且物理合理的视频反事实解释的问题。

### 方法

提出了Back To The Feature (BTTF)优化框架，引入两个新特性：1) 一种优化方案，用于检索由输入视频第一帧条件化的初始潜在噪声；2) 两阶段优化策略，使能够在输入视频附近搜索反事实视频。两个优化过程仅由目标分类器指导，确保解释的忠实性。还引入渐进优化策略以加速收敛，逐步增加去噪步骤的数量。

### 主要发现

在Shape-Moving(动作分类)、MEAD(情绪分类)和NTU RGB+D(动作分类)等视频数据集上的大量实验表明，BTTF能够生成有效、视觉相似且真实的反事实视频，这些视频为分类器的决策机制提供了具体的见解。

### 结论

BTTF方法有效地解决了视频分类器反事实解释的生成问题，生成的反事实视频能够提供对分类器决策机制的深入理解。

### 翻译

反事实解释是对模型输入的最小且语义上有意义的修改，可以改变模型预测。它们突出了模型依赖的决定性特征，为分类器提供了对比性解释。最先进的视觉反事实解释方法是为解释图像分类器而设计的。为视频分类器生成反事实解释的研究仍然很少。为了使反事实视频有用，它们必须是物理合理的、时间连贯的，并且表现出平滑的运动轨迹。现有的基于图像的反事实解释方法是为解释图像分类器而设计的，缺乏生成时间连贯、平滑且物理合理的视频反事实解释的能力。为此，我们提出了Back To The Feature (BTTF)，一个生成视频反事实解释的优化框架。我们的方法引入了两个新特性：1) 一种优化方案，用于检索由输入视频第一帧条件化的初始潜在噪声；2) 一种两阶段优化策略，使能够在输入视频附近搜索反事实视频。两个优化过程仅由目标分类器指导，确保解释的忠实性。为了加速收敛，我们还引入了一种渐进优化策略，逐步增加去噪步骤的数量。在Shape-Moving(动作分类)、MEAD(情绪分类)和NTU RGB+D(动作分类)等视频数据集上的大量实验表明，我们的BTTF有效地生成有效、视觉相似且真实的反事实视频，为分类器的决策机制提供了具体的见解。


### 论文摘要

Counterfactual explanations (CFEs) are minimal and semantically meaningful modifications of the input of a model that alter the model predictions. They highlight the decisive features the model relies on, providing contrastive interpretations for classifiers. State-of-the-art visual counterfactual explanation methods are designed to explain image classifiers. The generation of CFEs for video classifiers remains largely underexplored. For the counterfactual videos to be useful, they have to be physically plausible, temporally coherent, and exhibit smooth motion trajectories. Existing CFE image-based methods, designed to explain image classifiers, lack the capacity to generate temporally coherent, smooth and physically plausible video CFEs. To address this, we propose Back To The Feature (BTTF), an optimization framework that generates video CFEs. Our method introduces two novel features, 1) an optimization scheme to retrieve the initial latent noise conditioned by the first frame of the input video, 2) a two-stage optimization strategy to enable the search for counterfactual videos in the vicinity of the input video. Both optimization processes are guided solely by the target classifier, ensuring the explanation is faithful. To accelerate convergence, we also introduce a progressive optimization strategy that incrementally increases the number of denoising steps. Extensive experiments on video datasets such as Shape-Moving (motion classification), MEAD (emotion classification), and NTU RGB+D (action classification) show that our BTTF effectively generates valid, visually similar and realistic counterfactual videos that provide concrete insights into the classifier's decision-making mechanism.

---

## 170. 论文ID: 2511.20292v1

**错误:** 无法读取论文信息 - [Errno 2] No such file or directory: 'storage/result/2511.20292v1.json'

---

## 171. Stood-up drop to measure receding contact angles

**论文链接:** [http://arxiv.org/abs/2511.20259v1](http://arxiv.org/abs/2511.20259v1)

**作者:** Diego Díaz, Aman Bhargava, Franziska Walz, Azadeh Sharifi, Sajjad Sumally, Rüdiger Berger, Michael Kappl, Hans-Jürgen Butt, Detlef Lohse, Thomas Willers, Vatsal Sanjay, Doris Vollmer

**发布时间:** 2025-11-25

**备注:** 11 pages, 6 figures, currently under review for publication in RSC journal Soft Matter

### GPT解析

### 总结

本文提出了一种称为'站立液滴'(SUD)的新技术，用于测量后退接触角，克服了传统接触角测量法的局限性。

### 背景

液滴在自然和工业表面的润湿行为由前进接触角和后退接触角决定，传统测量方法(静态液滴技术)需要较大液滴体积、较长接触时间，且容易受操作者影响，难以自动化。

### 目的

开发一种替代方法来测量后退接触角，解决传统接触角测量法的局限性。

### 方法

通过短液滴将液体沉积在表面上，使其径向扩散形成煎饼状薄膜，然后液体收缩形成球形液滴形状(站立液滴)，在准平衡状态下测量接触角。

### 主要发现

SUD技术适用于从亲水到疏水的各种表面，克服了针管导致的液滴形状扭曲问题，减少了操作者依赖性；通过体积流体模拟明确了该方法适用的条件，并提供了简单的判断标准。

### 结论

站立液滴技术是一种有效的后退接触角测量方法，具有操作简单、适用范围广、减少人为误差等优势。

### 翻译

液滴在自然和工业表面的润湿行为由前进接触角和后退接触角决定。这些角度通常通过静态液滴技术(也称为接触角测量法)测量，该方法通过固体针管注入液体。因此，这种方法需要较大的液滴体积、较长的接触时间，且容易受操作者影响，难以自动化。在这里，我们提出了一种称为'站立液滴'(SUD)的技术作为测量后退接触角的替代方法。该方法通过短液滴将液体沉积在表面上，使其径向扩散形成煎饼状薄膜。然后液体收缩，形成球形液滴形状(站立液滴)。在这种准平衡状态下，接触角与通过接触角测量法测量的后退接触角非常相似。我们的方法适用于从亲水到疏水的各种表面，克服了接触角测量法的典型问题，如针管导致的液滴形状扭曲，并减少了操作者依赖性。我们通过体积流体模拟系统地改变粘度、接触角和沉积液滴体积，明确了何时可以通过站立法获得后退接触角。最后，我们提供了简单的标准来判断站立液滴技术何时适用。


### 论文摘要

The wetting behavior of drops on natural and industrial surfaces is determined by the advancing and receding contact angles. They are commonly measured by the sessile drop technique, also called goniometry, which doses liquid through a solid needle. Consequently, this method requires substantial drop volumes, long contact times, tends to be user-dependent, and is difficult to automate. Here, we propose the stood-up drop (SUD) technique as an alternative to measure receding contact angles. The method consists of depositing a liquid drop on a surface by a short liquid jet, at which it spreads radially forming a pancake-shaped film. Then the liquid retracts, forming a spherical cap drop shape (stood-up drop). At this quasi-equilibrium state, the contact angle ($θ_\text{SUD}$) closely resembles the receding contact angle measured by goniometry. Our method is suitable for a wide variety of surfaces from hydrophilic to hydrophobic, overcoming typical complications of goniometry such as needle-induced distortion of the drop shape, and it reduces user dependence. We delineate when the receding contact angle can be obtained by the stood-up method using Volume-of-Fluid (VoF) simulations that systematically vary viscosity, contact angle, and deposited drop volume. Finally, we provide simple scaling criteria to predict when the stood-up drop technique works.

---

## 172. A Surrogate-Informed Framework for Sparse Grid Interpolation

**论文链接:** [http://arxiv.org/abs/2511.20187v1](http://arxiv.org/abs/2511.20187v1)

**作者:** Matteo Rosellini, Filippo Fruzza, Alessandro Mariotti, Maria Vittoria Salvetti, Lorenzo Tamellini

**发布时间:** 2025-11-25

### GPT解析

### 总结

本研究提出了一种替代模型引导的稀疏网格插值框架，通过误差指标高效选择关键点进行函数近似，显著减少了计算成本同时保持了高精度。

### 背景

科学和工程中经常需要近似复杂、高维、计算密集型函数。标准稀疏网格虽然能缓解维度灾难，但对所有区域进行各向同性处理，对于具有局部化或各向异性行为的函数效率不高。

### 目的

开发一种能够高效处理局部化和各向异性函数的稀疏网格插值方法，减少计算密集型函数评估的次数，同时保持高精度。

### 方法

提出一种替代模型引导的框架，使用误差指标作为层次化剩余量的零成本估计。该指标通过比较两个连续插值预测的相对差异量化局部近似误差，并据此对候选点进行排序，只在高优先级点进行实际函数评估，其余点使用初始替代模型的预测值。

### 主要发现

所提出的替代模型引导细化标准在多个解析函数和真实工程问题（氢燃料多孔燃烧器回火现象的敏感性分析）中表现出高准确性和效率，能够以计算成本的一小部分达到接近完全解析网格的精度。

### 结论

替代模型引导的稀疏网格插值框架是一种高效的方法，能够显著减少计算密集型函数评估次数，同时保持高精度，特别适用于具有局部化或各向异性行为的函数。

### 翻译

近似复杂、高维、计算密集型函数是科学和工程中的核心问题。标准稀疏网格通过缓解维度灾难相比完全张量网格提供了强大的解决方案。然而，它们对域的所有区域进行各向同性处理，对于具有局部化或各向异性行为的函数可能效率不高。本文提出了一个由替代模型引导的构建稀疏网格插值的框架，该框架由一个误差指标引导，该指标作为层次化剩余量的零成本估计。该指标为所有候选点计算，候选点定义为下一级网格w+1中但不在基础网格w中的点。它通过测量w级和w-1级两个连续插值预测之间的相对差异来量化局部近似误差。然后根据这个指标对候选点进行排序，以选择对细化影响最大的点，直到达到给定的预算或遵循其他标准，例如误差指标中的给定阈值。最终的高阶模型使用替代模型引导的方法构建：仅在选定的高优先级点评估目标函数，而对于w+1网格的其余节点，我们分配由初始w级替代模型预测的值。这种策略显著减少了所需的昂贵评估次数，产生的最终模型以计算成本的一小部分就 closely approximates 了完全解析的w+1网格的准确性。所提出的替代模型引导细化标准的准确性和效率在几个解析函数和真实工程问题（即氢燃料多孔燃烧器中数值预测的回火现象对几何参数的敏感性分析）中得到了验证。


### 论文摘要

Approximating complex, high-dimensional, and computationally expensive functions is a central problem in science and engineering. Standard sparse grids offer a powerful solution by mitigating the curse of dimensionality compared to full tensor grids. However, they treat all regions of the domain isotropically, which may not be efficient for functions with localized or anisotropic behavior. This work presents a surrogate-informed framework for constructing sparse grid interpolants, which is guided by an error indicator that serves as a zero-cost estimate for the hierarchical surplus. This indicator is calculated for all candidate points, defined as those in the next-level grid $w+1$ not already present in the base grid $w$. It quantifies the local approximation error by measuring the relative difference between the predictions of two consecutive interpolants of level $w$ and $w-1$. The candidates are then ranked by this metric to select the most impactful points for refinement up to a given budget or following another criterion, as, e.g., a given threshold in the error indicator. The final higher-order model is then constructed using a surrogate-informed approach: the objective function is evaluated only at the selected high-priority points, while for the remaining nodes of the $w+1$ grid, we assign the values predicted by the initial $w$-level surrogate. This strategy significantly reduces the required number of expensive evaluations, yielding a final model that closely approximates the accuracy of a fully-resolved $w+1$ grid at a fraction of the computational cost. The accuracy and efficiency of the proposed surrogate-informed refinement criterion is demonstrated for several analytic function and for a real engineering problem, i.e., the analysis of sensitivity to geometrical parameters of numerically predicted flashback phenomenon in hydrogen-fueled perforated burners.

---

## 173. Enhancing Sequential Recommendation with World Knowledge from Large Language Models

**论文链接:** [http://arxiv.org/abs/2511.20177v1](http://arxiv.org/abs/2511.20177v1)

**作者:** Tianjie Dai, Xu Chen, Yunmeng Shu, Jinsong Lan, Xiaoyong Zhu, Jiangchao Yao, Bo Zheng

**发布时间:** 2025-11-25

### GPT解析

### 总结

GRASP是一个创新的序列推荐框架，通过生成增强检索和整体注意力增强，有效利用LLM知识并减轻幻觉噪声，在各种数据集上实现了最先进的性能。

### 背景

序列推荐系统(SRS)在现代社会中变得至关重要，传统的基于协同过滤的序列推荐模型由于协同信号有限导致性能不佳，而将LLMs整合到序列推荐中的方法虽然取得进展，但通常假设LLM生成结果的正确性且容易受到幻觉噪声影响。

### 目的

克服传统序列推荐方法和现有LLM集成方法的局限性，开发一种能够有效利用LLM世界知识同时减轻幻觉噪声影响的序列推荐框架。

### 方法

提出GRASP框架，集成生成增强检索用于描述性合成和相似性检索，采用多级整体注意力增强来有效利用LLM世界知识并捕捉用户动态兴趣，将检索到的相似用户/项目作为辅助上下文信息缓解基于监督方法的噪声引导问题。

### 主要发现

在两个公共基准和一个工业数据集上的全面评估表明，GRASP在与多种骨干模型集成时始终达到最先进的性能。

### 结论

GRASP框架成功解决了传统序列推荐方法的局限性，有效整合了LLM的世界知识，同时减轻了幻觉带来的噪声问题，为序列推荐提供了新的解决方案。

### 翻译

序列推荐系统(SRS)已成为现代社会的关键组成部分，它基于用户的历史行为来预测后续动作。然而，传统的基于协同过滤的序列推荐模型往往由于其协同信号的有限信息而导致次优性能。随着LLMs的快速发展，越来越多的工作将LLMs的世界知识整合到序列推荐中。尽管它们取得了显著的进步，但这些方法通常假设LLM生成结果的正确性，并且仍然容易受到LLM幻觉引起的噪声影响。为了克服这些局限性，我们提出了GRASP（用于序列预测的生成增强检索与整体注意力），一个灵活的框架，集成了用于描述性合成和相似性检索的生成增强检索，以及整体注意力增强，该增强采用多级注意力来有效利用LLM的世界知识，即使存在幻觉也能更好地捕捉用户的动态兴趣。检索到的相似用户/项目作为后续整体注意力增强模块的辅助上下文信息，有效缓解了基于监督方法的噪声引导问题。在两个公共基准和一个工业数据集上的全面评估表明，GRASP在与各种骨干模型集成时始终达到最先进的性能。代码可在以下网址获取：https://anonymous.4open.science/r/GRASP-SRS。


### 论文摘要

Sequential Recommendation System~(SRS) has become pivotal in modern society, which predicts subsequent actions based on the user's historical behavior. However, traditional collaborative filtering-based sequential recommendation models often lead to suboptimal performance due to the limited information of their collaborative signals. With the rapid development of LLMs, an increasing number of works have incorporated LLMs' world knowledge into sequential recommendation. Although they achieve considerable gains, these approaches typically assume the correctness of LLM-generated results and remain susceptible to noise induced by LLM hallucinations. To overcome these limitations, we propose GRASP (Generation Augmented Retrieval with Holistic Attention for Sequential Prediction), a flexible framework that integrates generation augmented retrieval for descriptive synthesis and similarity retrieval, and holistic attention enhancement which employs multi-level attention to effectively employ LLM's world knowledge even with hallucinations and better capture users' dynamic interests. The retrieved similar users/items serve as auxiliary contextual information for the later holistic attention enhancement module, effectively mitigating the noisy guidance of supervision-based methods. Comprehensive evaluations on two public benchmarks and one industrial dataset reveal that GRASP consistently achieves state-of-the-art performance when integrated with diverse backbones. The code is available at: https://anonymous.4open.science/r/GRASP-SRS.

---

## 174. SKEL-CF: Coarse-to-Fine Biomechanical Skeleton and Surface Mesh Recovery

**论文链接:** [http://arxiv.org/abs/2511.20157v1](http://arxiv.org/abs/2511.20157v1)

**作者:** Da Li, Ji-Ping Jin, Xuanlong Yu, Wei Liu, Xiaodong Cun, Kai Chen, Rui Fan, Jiangang Kong, Shen Xi

**发布时间:** 2025-11-25

**备注:** 15 pages, 10 figures

### GPT解析

### 总结

这篇论文介绍了SKEL-CF框架，一种用于SKEL参数估计的粗到细方法，通过基于transformer的编码器-解码器架构和专门的4DHuman-SKEL数据集，显著提升了人体运动分析的生物力学真实性。

### 背景

参数化3D人体模型如SMPL在人体姿态和形状估计方面取得进展，但简化的运动学限制了生物力学真实性。SKEL模型通过使用解剖学准确的骨架重新配置SMPL解决了这一问题，但直接估计SKEL参数仍面临训练数据有限、视角模糊性和人体关节复杂性等挑战。

### 目的

开发一个有效的SKEL参数估计框架，解决现有方法在训练数据有限、视角模糊性和人体关节复杂性方面的挑战，提高人体运动分析的生物力学准确性。

### 方法

1. 提出SKEL-CF粗到细框架，采用基于transformer的编码器-解码器架构；2. 编码器预测粗略参数，解码器逐步细化；3. 将4DHuman数据集转换为SKEL对齐的4DHuman-SKEL数据集；4. 在管道中纳入相机建模以减轻深度和尺度模糊性。

### 主要发现

在具有挑战性的MOYO数据集上，SKEL-CF达到85.0 MPJPE / 51.4 PA-MPJPE的性能，显著优于之前基于SKEL的最先进方法HSMR（104.5 / 79.6）；相机建模对于处理不同视角下的估计非常重要。

### 结论

SKEL-CF是一个可扩展且解剖学上准确的人体运动分析框架，成功弥合了计算机视觉和生物力学之间的差距，为人体的3D建模和分析提供了更可靠的方法。

### 翻译

参数化3D人体模型如SMPL推动了人体姿态和形状估计的重大进展，但其简化的运动学限制了生物力学真实性。最近提出的SKEL模型通过使用解剖学准确的骨架重新配置SMPL来解决这一局限性。然而，由于训练数据有限、视角模糊性和人体关节固有的复杂性，直接估计SKEL参数仍然具有挑战性。我们引入了SKEL-CF，一个用于SKEL参数估计的粗到细框架。SKEL-CF采用基于transformer的编码器-解码器架构，其中编码器预测粗略的相机和SKEL参数，解码器在后续层中逐步细化这些参数。为确保解剖学一致的监督，我们将现有的基于SMPL的数据集4DHuman转换为与SKEL对齐的版本4DHuman-SKEL，为SKEL估计提供高质量的训练数据。此外，为了减轻深度和尺度模糊性，我们明确将相机建模纳入SKEL-CF管道，并展示了其在不同视角下的重要性。大量实验验证了所提出设计的有效性。在具有挑战性的MOYO数据集上，SKEL-CF达到了85.0 MPJPE / 51.4 PA-MPJPE的性能，显著优于之前基于SKEL的最先进方法HSMR（104.5 / 79.6）。这些结果将SKEL-CF确立为可扩展且解剖学上准确的人体运动分析框架，弥合了计算机视觉和生物力学之间的差距。我们的实现可在项目页面获取：https://pokerman8.github.io/SKEL-CF/。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决现有3D人体模型（如SMPL）在生物力学真实性方面的不足。现有模型的简化运动学无法准确表示人体关节的自然运动，特别是在复杂姿势下会产生不自然的关节活动。这个问题在生物力学研究、康复医学和人机交互等领域非常重要，因为这些应用需要精确的解剖学准确骨骼模型来分析人体运动、评估康复效果或指导机器人与人体的自然交互。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者观察到直接估计SKEL参数（一种更解剖学准确的人体表示）具有挑战性，特别是在有限训练数据和视角模糊的情况下。他们借鉴了HSMR的Transformer架构和CameraHMR的相机建模方法，但创新性地提出了粗到细的渐进式参数估计策略。首先，编码器预测初始的粗略参数，然后解码器在多层中逐步细化这些估计。同时，作者将现有SMPL数据集转换为高质量的SKEL对齐数据集（4DHuman-SKEL），为训练提供更可靠的监督信号。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是通过粗到细的渐进式参数估计策略，结合解剖学准确的SKEL表示与显式相机建模，提高人体重建的精度和生物力学真实性。整体流程是：1) 使用编码器从输入图像预测初始的相机和SKEL参数；2) 通过多层解码器逐步细化这些参数；3) 利用高质量SKEL数据集进行训练；4) 结合多种损失函数（关键点损失、参数损失和细化损失）优化模型。最终输出包括姿态、形状和相机参数的精确估计，用于重建具有生物力学真实性的人体网格。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1) 粗到细的渐进式参数估计框架，显著提升SKEL重建精度；2) 引入4DHuman-SKEL高质量数据集，提供解剖学一致的训练数据；3) 整合显式相机建模，增强不同视角下的鲁棒性。相比之前工作，SKEL-CF与HSMR的主要不同在于采用了渐进式细化策略和显式相机建模；与基于SMPL的方法（如CameraHMR）相比，SKEL-CF在保持数值精度的同时，提供了更符合生物力学的人体表示，特别是在复杂姿势下表现更自然。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'SKEL-CF通过粗到细的渐进式参数估计策略、高质量的SKEL对齐数据集和显式相机建模，实现了在保持生物力学真实性的同时，显著提高了人体骨骼和表面网格恢复的精度和鲁棒性，为计算机视觉与生物力学之间架起了桥梁。'}


### 论文摘要

Parametric 3D human models such as SMPL have driven significant advances in human pose and shape estimation, yet their simplified kinematics limit biomechanical realism. The recently proposed SKEL model addresses this limitation by re-rigging SMPL with an anatomically accurate skeleton. However, estimating SKEL parameters directly remains challenging due to limited training data, perspective ambiguities, and the inherent complexity of human articulation. We introduce SKEL-CF, a coarse-to-fine framework for SKEL parameter estimation. SKEL-CF employs a transformer-based encoder-decoder architecture, where the encoder predicts coarse camera and SKEL parameters, and the decoder progressively refines them in successive layers. To ensure anatomically consistent supervision, we convert the existing SMPL-based dataset 4DHuman into a SKEL-aligned version, 4DHuman-SKEL, providing high-quality training data for SKEL estimation. In addition, to mitigate depth and scale ambiguities, we explicitly incorporate camera modeling into the SKEL-CF pipeline and demonstrate its importance across diverse viewpoints. Extensive experiments validate the effectiveness of the proposed design. On the challenging MOYO dataset, SKEL-CF achieves 85.0 MPJPE / 51.4 PA-MPJPE, significantly outperforming the previous SKEL-based state-of-the-art HSMR (104.5 / 79.6). These results establish SKEL-CF as a scalable and anatomically faithful framework for human motion analysis, bridging the gap between computer vision and biomechanics. Our implementation is available on the project page: https://pokerman8.github.io/SKEL-CF/.

---

## 175. Map-World: Masked Action planning and Path-Integral World Model for Autonomous Driving

**论文链接:** [http://arxiv.org/abs/2511.20156v1](http://arxiv.org/abs/2511.20156v1)

**作者:** Bin Hu, Zijian Lu, Haicheng Liao, Chengran Yuan, Bin Rao, Yongkang Li, Guofa Li, Zhiyong Cui, Cheng-zhong Xu, Zhenning Li

**发布时间:** 2025-11-25

### GPT解析

### 总结

MAP-World是一种无先验的多模态规划框架，结合掩码动作规划与路径加权世界模型，能够处理自动驾驶中的多种可能未来场景，同时保持计算效率。

### 背景

自动驾驶运动规划需要处理多种可能的未来场景，同时保持计算效率。现有的端到端系统和基于世界模型的规划器预测丰富的多模态轨迹，但通常依赖于手工制作的锚点或强化学习来选择单个最佳模式进行训练和控制，这会丢弃替代未来信息并使优化复杂化。

### 目的

提出一个名为MAP-World的无先验多模态规划框架，结合掩码动作规划与路径加权世界模型，解决现有方法中丢弃替代未来信息并使优化复杂化的问题。

### 方法

掩码动作规划(MAP)模块将未来自车运动视为掩码序列完成：过去路径点编码为可见标记，未来路径点表示为掩码标记，驾驶意图路径提供粗略支架。紧凑的潜在规划状态扩展为多个带有注入噪声的轨迹查询，产生多样化的、时间一致的模式，无需锚点库或教师策略。轻量级世界模型根据每个候选轨迹展开未来的BEV语义。训练期间，语义损失计算为模式的期望值，使用轨迹概率作为离散路径权重，使规划器从整个可能的未来分布中学习。

### 主要发现

在NAVSIM上，MAP-World与基于锚点的方法相匹配，并在基于世界模型的方法中取得了最先进的性能，同时避免了强化学习并保持了实时推理延迟。

### 结论

MAP-World框架有效地处理了自动驾驶中的多模态规划问题，同时保持了计算效率，无需依赖强化学习或复杂的锚点库。

### 翻译

自动驾驶的运动规划必须处理多种可能的未来场景，同时保持计算效率。最近的端到端系统和基于世界模型的规划器预测丰富的多模态轨迹，但通常依赖于手工制作的锚点或强化学习来为训练和控制选择单个最佳模式。这种选择丢弃了关于替代未来的信息，并使优化复杂化。我们提出了MAP-World，一种无先验的多模态规划框架，将掩码动作规划与路径加权世界模型相结合。掩码动作规划(MAP)模块将未来的自车运动视为掩码序列完成：过去的路径点被编码为可见标记，未来的路径点表示为掩码标记，驾驶意图路径提供粗略支架。一个紧凑的潜在规划状态被扩展为多个带有注入噪声的轨迹查询，产生多样化的、时间一致的模式，无需锚点库或教师策略。然后，一个轻量级的世界模型根据每个候选轨迹展开未来的BEV语义。在训练期间，语义损失被计算为模式的期望值，使用轨迹概率作为离散路径权重，因此规划器从整个可能的未来分布中学习，而不是从单个选择的路径中学习。在NAVSIM上，我们的方法与基于锚点的方法相匹配，并在基于世界模型的方法中取得了最先进的性能，同时避免了强化学习并保持了实时推理延迟。


### 论文摘要

Motion planning for autonomous driving must handle multiple plausible futures while remaining computationally efficient. Recent end-to-end systems and world-model-based planners predict rich multi-modal trajectories, but typically rely on handcrafted anchors or reinforcement learning to select a single best mode for training and control. This selection discards information about alternative futures and complicates optimization. We propose MAP-World, a prior-free multi-modal planning framework that couples masked action planning with a path-weighted world model. The Masked Action Planning (MAP) module treats future ego motion as masked sequence completion: past waypoints are encoded as visible tokens, future waypoints are represented as mask tokens, and a driving-intent path provides a coarse scaffold. A compact latent planning state is expanded into multiple trajectory queries with injected noise, yielding diverse, temporally consistent modes without anchor libraries or teacher policies. A lightweight world model then rolls out future BEV semantics conditioned on each candidate trajectory. During training, semantic losses are computed as an expectation over modes, using trajectory probabilities as discrete path weights, so the planner learns from the full distribution of plausible futures instead of a single selected path. On NAVSIM, our method matches anchor-based approaches and achieves state-of-the-art performance among world-model-based methods, while avoiding reinforcement learning and maintaining real-time inference latency.

---

## 176. Multivariate Forecasting of Bitcoin Volatility with Gradient Boosting: Deterministic, Probabilistic, and Feature Importance Perspectives

**论文链接:** [http://arxiv.org/abs/2511.20105v1](http://arxiv.org/abs/2511.20105v1)

**作者:** Grzegorz Dudek, Mateusz Kasprzyk, Paweł Pełka

**发布时间:** 2025-11-25

**DOI:** 10.1016/j.eswa.2025.130404

### GPT解析

### 总结

本研究探讨使用轻量梯度提升机(LGBM)模型进行比特币已实现波动率的确定性和概率性预测，通过69个预测变量评估模型性能并与基线模型比较，发现LGBM能有效捕捉加密货币市场的非线性特征并提供可解释见解。

### 背景

比特币等加密货币市场具有高波动性和非线性特征，准确预测其波动率对投资者和风险管理至关重要，而传统计量经济学模型可能难以捕捉这些复杂动态。

### 目的

评估LGBM模型在比特币已实现波动率预测中的性能，包括确定性和概率性预测两个方面，并识别影响波动率的主要驱动因素。

### 方法

使用69个预测变量的综合数据集；应用LGBM模型进行波动率预测；采用两种概率性预测方法（基于分位数回归的直接分位数预测和残差模拟方法）；使用基于增益和排列的特征重要性技术识别主要驱动因素；与计量经济学和机器学习基线模型进行比较。

### 主要发现

LGBM模型在比特币波动率预测中表现优异；交易量、滞后波动率指标、投资者关注度和市值是波动率的主要驱动因素；LGBM能有效捕捉加密货币市场的非线性特征和高方差特性。

### 结论

LGBM模型为比特币波动率预测提供了一种有效的方法，不仅能提供准确的预测结果，还能提供对潜在波动率动态的可解释见解，对理解和应对加密货币市场波动具有重要意义。

### 翻译

本研究探讨了使用轻量梯度提升机(LGBM)模型进行比特币已实现波动率的确定性和概率性预测。利用包含市场、行为和宏观经济指标在内的69个预测变量的综合数据集，我们评估了基于LGBM的模型性能，并将其与计量经济学和机器学习基线模型进行比较。对于概率性预测，我们探索了两种基于分位数的方法：使用pinball损失函数的直接分位数回归，以及将点预测转换为预测分布的残差模拟方法。为了识别波动率的主要驱动因素，我们采用基于增益和排列的特征重要性技术，一致地强调了交易量、滞后波动率指标、投资者关注度和市值的重要性。结果表明，LGBM模型能够有效捕捉加密货币市场的非线性和高方差特性，同时提供对潜在波动率动态的可解释见解。


### 论文摘要

This study investigates the application of the Light Gradient Boosting Machine (LGBM) model for both deterministic and probabilistic forecasting of Bitcoin realized volatility. Utilizing a comprehensive set of 69 predictors -- encompassing market, behavioral, and macroeconomic indicators -- we evaluate the performance of LGBM-based models and compare them with both econometric and machine learning baselines. For probabilistic forecasting, we explore two quantile-based approaches: direct quantile regression using the pinball loss function, and a residual simulation method that transforms point forecasts into predictive distributions. To identify the main drivers of volatility, we employ gain-based and permutation feature importance techniques, consistently highlighting the significance of trading volume, lagged volatility measures, investor attention, and market capitalization. The results demonstrate that LGBM models effectively capture the nonlinear and high-variance characteristics of cryptocurrency markets while providing interpretable insights into the underlying volatility dynamics.

---

## 177. RED-F: Reconstruction-Elimination based Dual-stream Contrastive Forecasting for Multivariate Time Series Anomaly Prediction

**论文链接:** [http://arxiv.org/abs/2511.20044v1](http://arxiv.org/abs/2511.20044v1)

**作者:** PengYu Chen, Xiaohou Shi, Yuan Chang, Yan Sun, Sajal K. Das

**发布时间:** 2025-11-25

**备注:** 13 pages, 12 figures

### GPT解析

### 总结

本文提出RED-F框架，用于多变量时间序列中的异常主动预测，通过重建-消除机制和双流对比预测解决了现有方法难以识别弱异常前兆的问题。

### 背景

多变量时间序列中的异常主动预测是确保系统可靠性的关键挑战，难点在于识别隐藏在正常信号中的细微异常前兆。

### 目的

解决现有无监督方法在处理弱异常前兆时的局限性，提高异常预测的准确性。

### 方法

提出RED-F框架，包含重建-消除模型(REM)和双流对比预测模型(DFM)。REM使用混合时频机制生成纯净的正常模式基线，DFM接收纯净基线和原始序列作为并行输入，通过对比预测放大微弱前兆信号，并采用多序列预测目标增强预测敏感性。

### 主要发现

在六个真实世界数据集上的实验表明，RED-F在异常预测任务中展现出优越能力。

### 结论

RED-F框架有效解决了现有无监督方法在异常预测中的局限性，能够更好地识别和预测异常前兆。

### 翻译

多变量时间序列中异常的主动预测是确保系统可靠性的关键挑战。困难在于识别隐藏在正常信号中的细微异常前兆。然而，现有仅使用正常数据训练的无监督方法表现出重建正常模式的基本倾向。因此，当面对弱前兆时，它们的预测被正常模式主导，淹没了预测所需的信号。为了应对这一局限，我们提出了RED-F，一个基于重建-消除的双流对比预测框架，包含重建-消除模型和双流对比预测模型。REM利用混合时频机制减轻前兆，生成纯净的正常模式基线。然后DFM接收这个纯净基线和保留前兆的原始序列作为并行输入。在我们框架的核心，RED-F采用对比预测，通过计算这两个预测流之间的发散性，将困难的绝对信号检测任务转化为更简单、更稳健的相对轨迹比较任务。这种对比机制用于放大微弱的前兆信号。此外，DFM使用新的多序列预测目标进行训练，利用远期上下文增强其预测敏感性。在六个真实世界数据集上的大量实验证明了RED-F在异常预测任务中的优越能力。


### 论文摘要

The proactive prediction of anomalies (AP) in mul- tivariate time series (MTS) is a critical challenge to ensure system dependability. The difficulty lies in identifying subtle anomaly precursors concealed within normal signals. However, existing unsupervised methods, trained exclusively on normal data, demonstrate a fundamental propensity to reconstruct normal patterns. Consequently, when confronted with weak precursors, their predictions are dominated by the normal pattern, submerging the very signal required for prediction. To contend with the limitation, we propose RED-F, a Reconstruction- Elimination based Dual-stream Contrastive Forecasting frame- work, comprising the Reconstruction-Elimination Model (REM) and the Dual-stream Contrastive Forecasting Model (DFM). The REM utilizes a hybrid time-frequency mechanism to mitigate the precursor, generating a purified, normal-pattern baseline. The DFM then receives this purified baseline and the original sequence which retains the precursor as parallel inputs. At the core of our framework, RED-F employs a contrastive forecast that transforms the difficult task of absolute signal detection into a simpler, more robust task of relative trajectory comparison by computing the divergence between these two predictive streams. This contrastive mechanism serves to amplify the faint precursor signal. Furthermore, the DFM is trained with a novel Multi-Series Prediction (MSP) objective, which leverages distant future con- text to enhance its predictive sensitivity. Extensive experiments on six real-world datasets demonstrate the superior capability of RED-F in anomaly prediction tasks.

---

## 178. ACIT: Attention-Guided Cross-Modal Interaction Transformer for Pedestrian Crossing Intention Prediction

**论文链接:** [http://arxiv.org/abs/2511.20020v1](http://arxiv.org/abs/2511.20020v1)

**作者:** Yuanzhe Li, Steffen Müller

**发布时间:** 2025-11-25

### GPT解析

### 总结

本文提出了一种名为ACIT的注意力引导跨模态交互Transformer模型，用于预测行人过马路意图，通过整合多种视觉和运动模态数据，实现了优于现有方法的性能。

### 背景

预测行人过马路意图对自动驾驶汽车防止行人相关碰撞至关重要。然而，从不同类型数据中有效提取和整合互补线索仍然是一个主要挑战。

### 目的

开发一种能够有效整合多种模态数据的方法，以准确预测行人过马路的意图，提高自动驾驶汽车的安全性。

### 方法

ACIT模型利用六种视觉和运动模态，分为三组交互对：全局语义图与全局光流、局部RGB图像与局部光流、自车速度与行人边界框。通过双路径注意力机制增强主要模态中的显著区域，并通过光流引导的注意力与辅助模态进行深度交互。此外，还引入了基于Transformer的时间特征聚合模块来捕获序列依赖关系。

### 主要发现

实验结果表明，ACIT在JAADbeh和JAADall数据集上分别达到70%和89%的准确率，优于现有最先进的方法。消融研究验证了ACIT不同模块的有效贡献。

### 结论

ACIT通过有效的跨模态交互和注意力机制，能够准确预测行人过马路意图，为自动驾驶汽车提供了更安全的行人交互能力。

### 翻译

预测行人过马路意图对自动驾驶汽车防止行人相关碰撞至关重要。然而，从不同类型数据中有效提取和整合互补线索仍然是一个主要挑战。本文提出了一种用于行人过马路意图预测的注意力引导的跨模态交互Transformer（ACIT）。ACIT利用六种视觉和运动模态，将它们分为三组交互对：（1）全局语义图和全局光流，（2）局部RGB图像和局部光流，（3）自车速度和行人边界框。在每个视觉交互对中，双路径注意力机制通过模态内自注意力增强主要模态中的显著区域，并通过光流引导的注意力与辅助模态（即光流）进行深度交互。在运动交互对中，采用跨模态注意力来建模跨模态动态，从而有效提取互补的运动特征。除了成对交互外，多模态特征融合模块进一步促进了每个时间步的跨模态交互。此外，引入了基于Transformer的时间特征聚合模块来捕获序列依赖关系。实验结果表明，ACIT优于最先进的方法，在JAADbeh和JAADall数据集上分别达到70%和89%的准确率。还进行了广泛的消融研究，以调查ACIT不同模块的贡献。


### 论文摘要

Predicting pedestrian crossing intention is crucial for autonomous vehicles to prevent pedestrian-related collisions. However, effectively extracting and integrating complementary cues from different types of data remains one of the major challenges. This paper proposes an attention-guided cross-modal interaction Transformer (ACIT) for pedestrian crossing intention prediction. ACIT leverages six visual and motion modalities, which are grouped into three interaction pairs: (1) Global semantic map and global optical flow, (2) Local RGB image and local optical flow, and (3) Ego-vehicle speed and pedestrian's bounding box. Within each visual interaction pair, a dual-path attention mechanism enhances salient regions within the primary modality through intra-modal self-attention and facilitates deep interactions with the auxiliary modality (i.e., optical flow) via optical flow-guided attention. Within the motion interaction pair, cross-modal attention is employed to model the cross-modal dynamics, enabling the effective extraction of complementary motion features. Beyond pairwise interactions, a multi-modal feature fusion module further facilitates cross-modal interactions at each time step. Furthermore, a Transformer-based temporal feature aggregation module is introduced to capture sequential dependencies. Experimental results demonstrate that ACIT outperforms state-of-the-art methods, achieving accuracy rates of 70% and 89% on the JAADbeh and JAADall datasets, respectively. Extensive ablation studies are further conducted to investigate the contribution of different modules of ACIT.

---

## 179. Multi-Context Fusion Transformer for Pedestrian Crossing Intention Prediction in Urban Environments

**论文链接:** [http://arxiv.org/abs/2511.20011v1](http://arxiv.org/abs/2511.20011v1)

**作者:** Yuanzhe Li, Hang Zhong, Steffen Müller

**发布时间:** 2025-11-25

### GPT解析

### 总结

本文提出了一种多上下文融合Transformer(MFT)模型，通过融合四个关键维度的上下文信息来实现准确的行人过马路意图预测，在城市环境中表现出色。

### 背景

行人过马路意图预测对自动驾驶汽车提高行人安全性和减少交通事故至关重要。然而，城市环境中影响行人行为的因素众多，使得准确的行人意图预测具有挑战性。

### 目的

开发一种能够有效融合多种上下文信息的模型，以实现更准确的行人过马路意图预测。

### 方法

MFT模型利用四个关键维度的上下文属性：行人行为上下文、环境上下文、行人定位上下文和车辆运动上下文。采用渐进式融合策略，包括互相内部上下文注意力、互相跨上下文注意力、引导内部上下文注意力和引导跨上下文注意力四个步骤，实现深度高效的特征融合。

### 主要发现

MFT模型在JAADbeh、JAADall和PIE数据集上分别实现了73%、93%和90%的准确率，优于现有最先进方法。消融研究验证了网络架构的有效性和不同输入上下文的贡献。

### 结论

多上下文融合Transformer模型能够有效整合不同维度的上下文信息，显著提升行人过马路意图预测的准确性，有助于提高自动驾驶汽车的安全性和减少交通事故。

### 翻译

行人过马路意图预测对自动驾驶汽车提高行人安全性和减少交通事故至关重要。然而，由于影响行人行为的因素众多，在城市环境中准确预测行人意图仍然具有挑战性。在本文中，我们提出了一种多上下文融合Transformer(MFT)，它利用四个关键维度的多样化数值上下文属性，包括行人行为上下文、环境上下文、行人定位上下文和车辆运动上下文，以实现准确的行人意图预测。MFT采用渐进式融合策略，其中互相内部上下文注意力使每个上下文内部进行相互交互，从而促进特征序列融合并生成上下文特定的上下文标记表示。接着是互相跨上下文注意力，它通过全局CLS标记作为紧凑的多上下文表示来整合不同上下文的特征。最后，引导内部上下文注意力通过有向交互在每个上下文中细化上下文标记，而引导跨上下文注意力加强全局CLS标记，通过引导的信息传播促进多上下文融合，实现更深入和高效的集成。实验结果验证了MFT优于最先进方法，在JAADbeh、JAADall和PIE数据集上分别实现了73%、93%和90%的准确率。进一步进行了广泛的消融研究，以调查网络架构的有效性和不同输入上下文的贡献。我们的代码是开源的：https://github.com/ZhongHang0307/Multi-Context-Fusion-Transformer。


### 论文摘要

Pedestrian crossing intention prediction is essential for autonomous vehicles to improve pedestrian safety and reduce traffic accidents. However, accurate pedestrian intention prediction in urban environments remains challenging due to the multitude of factors affecting pedestrian behavior. In this paper, we propose a multi-context fusion Transformer (MFT) that leverages diverse numerical contextual attributes across four key dimensions, encompassing pedestrian behavior context, environmental context, pedestrian localization context and vehicle motion context, to enable accurate pedestrian intention prediction. MFT employs a progressive fusion strategy, where mutual intra-context attention enables reciprocal interactions within each context, thereby facilitating feature sequence fusion and yielding a context token as a context-specific representation. This is followed by mutual cross-context attention, which integrates features across contexts with a global CLS token serving as a compact multi-context representation. Finally, guided intra-context attention refines context tokens within each context through directed interactions, while guided cross-context attention strengthens the global CLS token to promote multi-context fusion via guided information propagation, yielding deeper and more efficient integration. Experimental results validate the superiority of MFT over state-of-the-art methods, achieving accuracy rates of 73%, 93%, and 90% on the JAADbeh, JAADall, and PIE datasets, respectively. Extensive ablation studies are further conducted to investigate the effectiveness of the network architecture and contribution of different input context. Our code is open-source: https://github.com/ZhongHang0307/Multi-Context-Fusion-Transformer.

---

## 180. Pedestrian Crossing Intention Prediction Using Multimodal Fusion Network

**论文链接:** [http://arxiv.org/abs/2511.20008v1](http://arxiv.org/abs/2511.20008v1)

**作者:** Yuanzhe Li, Steffen Müller

**发布时间:** 2025-11-25

### GPT解析

### 总结

这篇论文提出了一种多模态融合网络用于行人过马路意图预测，通过整合视觉和运动特征，并利用深度引导注意力和模态注意力机制，有效提高了预测性能。

### 背景

行人过马路意图预测对于自动驾驶汽车在城市环境中的部署至关重要，理想的预测可以为自动驾驶汽车提供关键的环境线索，从而减少行人相关碰撞的风险。

### 目的

开发一种能够有效提取和整合不同模态间互补线索的网络，以应对行人行为多样性和依赖多种上下文因素带来的预测挑战。

### 方法

提出了一种多模态融合网络，利用来自视觉和运动分支的七种模态特征；使用基于Transformer的提取模块提取运动和视觉特征；设计了深度引导的注意力模块、模态注意力和时间注意力机制。

### 主要发现

在JAAD数据集上的大量实验表明，所提出的网络相比基线方法取得了优越的性能，验证了该网络的有效性。

### 结论

通过多模态特征融合和注意力机制的有效设计，可以显著提高行人过马路意图预测的准确性，为自动驾驶汽车在城市环境中的安全部署提供支持。

### 翻译

行人过马路意图预测对于自动驾驶汽车在城市环境中的部署至关重要。理想的预测为自动驾驶汽车提供关键的环境线索，从而减少行人相关碰撞的风险。然而，由于行人行为的多样性和对多种上下文因素的依赖，预测任务具有挑战性。本文提出了一种多模态融合网络，利用来自视觉和运动分支的七种模态特征，旨在有效提取和整合不同模态间的互补线索。具体而言，使用多个基于Transformer的提取模块从原始输入中提取运动和视觉特征。深度引导的注意力模块利用深度信息通过全面的空间特征交互，引导注意力指向另一模态中的显著区域。为考虑不同模态和帧的重要性变化，设计了模态注意力和时间注意力，以选择性强调信息丰富的模态并有效捕获时间依赖性。在JAAD数据集上的大量实验验证了所提网络的有效性，相比基线方法取得了优越的性能。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决行人过马路意图预测问题。这个问题在现实中非常重要，因为准确的预测可以为自动驾驶汽车提供关键的环境线索，延长有效响应时间，减少与行人相关的碰撞风险，提高自动驾驶汽车在城市环境中的安全性。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先分析了现有方法的局限性，如早期方法仅使用单一模态信息（如单个图像），忽略了多模态信息的互补性。随后作者借鉴了Transformer架构在视觉任务中的优势，设计了基于Transformer的特征提取模块，并引入了深度引导的注意力机制。作者还参考了多模态融合的研究，但改进了简单拼接或加权平均的传统方法，设计了模态注意力和时间注意力机制来更有效地融合多模态信息。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是通过融合多种模态的特征（包括视觉和运动信息），并利用深度信息引导注意力机制关注关键区域，来提高行人过马路意图预测的准确性。整体流程包括：1) 使用7种不同模态作为输入（行人姿态、边界框、车辆速度、语义地图、深度图等）；2) 通过ViT提取视觉特征，通过Transformer编码器提取运动特征；3) 应用深度引导的注意力机制增强特征；4) 使用模态注意力融合不同模态特征；5) 通过时间注意力融合时间序列信息；6) 最终通过MLP预测行人意图。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1) 基于Transformer的特征提取模块，比传统CNN更能捕捉细粒度细节和全局上下文；2) 深度引导的注意力机制，利用深度信息引导关注关键区域；3) 模态注意力融合，动态调整不同模态的权重；4) 时间注意力融合，有效捕捉时间依赖性。相比之前的工作，不同之处在于：早期方法通常只使用单一模态，多数现有方法简单拼接或加权融合多模态信息，而本文采用更先进的Transformer架构，并特别利用深度信息引导注意力，这是现有工作中较少探索的方向。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '本文提出了一种多模态融合网络，通过结合视觉和运动信息，并利用深度引导的注意力机制，有效提高了行人过马路意图预测的准确性，在JAAD数据集上超越了所有基线方法。'}


### 论文摘要

Pedestrian crossing intention prediction is essential for the deployment of autonomous vehicles (AVs) in urban environments. Ideal prediction provides AVs with critical environmental cues, thereby reducing the risk of pedestrian-related collisions. However, the prediction task is challenging due to the diverse nature of pedestrian behavior and its dependence on multiple contextual factors. This paper proposes a multimodal fusion network that leverages seven modality features from both visual and motion branches, aiming to effectively extract and integrate complementary cues across different modalities. Specifically, motion and visual features are extracted from the raw inputs using multiple Transformer-based extraction modules. Depth-guided attention module leverages depth information to guide attention towards salient regions in another modality through comprehensive spatial feature interactions. To account for the varying importance of different modalities and frames, modality attention and temporal attention are designed to selectively emphasize informative modalities and effectively capture temporal dependencies. Extensive experiments on the JAAD dataset validate the effectiveness of the proposed network, achieving superior performance compared to the baseline methods.

---

## 181. LLM-EDT: Large Language Model Enhanced Cross-domain Sequential Recommendation with Dual-phase Training

**论文链接:** [http://arxiv.org/abs/2511.19931v1](http://arxiv.org/abs/2511.19931v1)

**作者:** Ziwei Liu, Qidong Liu, Wanyu Wang, Yejing Wang, Tong Xu, Wei Huang, Chong Chen, Peng Chuan, Xiangyu Zhao

**发布时间:** 2025-11-25

### GPT解析

### 总结

该论文提出了LLM-EDT方法，通过大型语言模型解决跨域顺序推荐中的不平衡问题、转换问题和粗糙用户画像问题。

### 背景

跨域顺序推荐(CDSR)通过整合不同领域信息丰富用户-物品交互，但面临不平衡问题和转换问题阻碍进一步发展。

### 目的

应对CDSR中的不平衡问题、转换问题和粗糙用户画像问题，提高跨域顺序推荐性能。

### 方法

提出LLM-EDT方法，包含三个主要组件：1)可转移物品增强器：为用户自适应地生成可能的跨域行为，解决不平衡问题并减少不相关噪声；2)双阶段训练策略：使领域特定线程具有领域共享背景，缓解转换问题；3)领域感知的用户画像模块：总结用户在每个领域的偏好，并自适应聚合生成全面用户画像，解决粗糙用户画像问题。

### 主要发现

在三个公共数据集上的实验验证了所提出的LLM-EDT方法的有效性。

### 结论

LLM-EDT方法成功解决了CDSR中的不平衡问题、转换问题和粗糙用户画像问题，提高了跨域顺序推荐的性能。

### 翻译

跨域顺序推荐(CDSR)已被提出通过整合来自不同领域的信息来丰富用户-物品交互。尽管当前有所进展，但不平衡问题和转换问题阻碍了CDSR的进一步发展。前者表现为一个领域中的交互主导了整个行为，导致难以捕捉其他领域的领域特定特征。后者指向在混合交互序列中难以捕捉用户的跨域偏好，导致特定领域的下一项预测性能差。凭借世界知识和强大的推理能力，大型语言模型(LLMs)通过作为生成器和编码器部分缓解了上述问题。然而，当前基于LLMs的CDSR方法仍在探索阶段，无法识别不相关的噪声和粗糙的用户画像问题。因此，为了应对上述挑战，我们提出了一个名为LLM-EDT的基于大型语言模型的跨域顺序推荐方法。为了解决不平衡问题并引入较少的不相关噪声，我们首先提出了可转移物品增强器，为用户自适应地生成可能的跨域行为。然后，为了缓解转换问题，我们引入了双阶段训练策略，使领域特定线程具有领域共享背景。至于粗糙的用户画像问题，我们设计了一个领域感知的用户画像模块，总结用户在每个领域的偏好，并自适应地聚合它们以生成全面的用户画像。在三个公共数据集上的实验验证了我们提出的LLM-EDT的有效性。为了便于复现，我们已在线发布了详细的代码。


### 论文摘要

Cross-domain Sequential Recommendation (CDSR) has been proposed to enrich user-item interactions by incorporating information from various domains. Despite current progress, the imbalance issue and transition issue hinder further development of CDSR. The former one presents a phenomenon that the interactions in one domain dominate the entire behavior, leading to difficulty in capturing the domain-specific features in the other domain. The latter points to the difficulty in capturing users' cross-domain preferences within the mixed interaction sequence, resulting in poor next-item prediction performance for specific domains. With world knowledge and powerful reasoning ability, Large Language Models (LLMs) partially alleviate the above issues by performing as a generator and an encoder. However, current LLMs-enhanced CDSR methods are still under exploration, which fail to recognize the irrelevant noise and rough profiling problems. Thus, to make peace with the aforementioned challenges, we proposed an LLMs Enhanced Cross-domain Sequential Recommendation with Dual-phase Training ({LLM-EDT}). To address the imbalance issue while introducing less irrelevant noise, we first propose the transferable item augmenter to adaptively generate possible cross-domain behaviors for users. Then, to alleviate the transition issue, we introduce a dual-phase training strategy to empower the domain-specific thread with a domain-shared background. As for the rough profiling problem, we devise a domain-aware profiling module to summarize the user's preference in each domain and adaptively aggregate them to generate comprehensive user profiles. The experiments on three public datasets validate the effectiveness of our proposed LLM-EDT. To ease reproducibility, we have released the detailed code online at {https://anonymous.4open.science/r/LLM-EDT-583F}.

---

## 182. Unifying Perception and Action: A Hybrid-Modality Pipeline with Implicit Visual Chain-of-Thought for Robotic Action Generation

**论文链接:** [http://arxiv.org/abs/2511.19859v1](http://arxiv.org/abs/2511.19859v1)

**作者:** Xiangkai Ma, Lekai Xing, Han Zhang, Wenzhong Li, Sanglu Lu

**发布时间:** 2025-11-25

### GPT解析

### 总结

本文提出了一种名为VITA的Vision-Integrated Trajectory Alignment框架，通过学习视觉和动作共享的离散潜在空间来解决视觉与动作间的模态差距和训练不稳定问题，在机器人任务中取得了最先进的性能。

### 背景

基于Chain-of-Thought的Vision-Language-Action模型在推进通用机器人代理方面取得了显著成功。然而，纯文本CoT难以充分捕捉复杂空间环境中的场景细节，而利用视觉先验指导机器人动作生成是一种有前景的策略。

### 目的

解决现有视觉引导机器人动作生成策略面临的两个固有挑战：视觉观测与低级动作之间的模态差距，以及由于视觉预测和动作生成之间的竞争目标而导致的不稳定训练。

### 方法

提出Vision-Integrated Trajectory Alignment (VITA)框架，学习视觉和动作共享的离散潜在空间，实现感知和运动控制的联合建模。VITA引入隐式视觉CoT，将自回归生成的令牌同时解码为未来帧预测和机器人动作，从而将视觉动力学内部化为运动规划的归纳偏置。

### 主要发现

在模拟和真实环境中的大量实验展示了最先进的性能。在CALVIN、LIBERO和SimplerEnv上，VITA比现有基线分别提高了14.5%、9.6%和12.1%。在六个真实世界任务中，VITA平均成功率达到80.5%。

### 结论

VITA展示了其作为通用机器人操作模型的潜力，能够有效整合视觉感知与动作生成，提升机器人在复杂环境中的表现。

### 翻译

建立在Chain-of-Thought基础上的Vision-Language-Action模型在推进通用机器人代理方面取得了显著成功，归因于其显著的感知理解能力。最近，由于纯文本CoT难以充分捕捉复杂空间环境中的场景细节，一种非常有前景的策略是利用视觉先验来指导机器人动作生成。然而，这些策略面临两个固有挑战：(i)视觉观测和低级动作之间的模态差距，以及(ii)由于视觉预测和动作生成之间的竞争目标而导致的不稳定训练。为解决这些挑战，我们提出了一个名为Vision-Integrated Trajectory Alignment (VITA)的框架，该框架学习视觉和动作共享的离散潜在空间，实现了感知和运动控制的联合建模。VITA引入了一种隐式视觉CoT：自回归生成的令牌同时被解码为未来帧预测和机器人动作，从而将视觉动力学内部化为运动规划的一种归纳偏置。在模拟和真实环境中的大量实验展示了最先进的性能。在CALVIN、LIBERO和SimplerEnv上，VITA比现有基线分别提高了14.5%、9.6%和12.1%。此外，VITA在六个真实世界任务中平均成功率达到80.5%，展示了其作为通用机器人操作模型的潜力。


### 论文摘要

Vision-Language-Action (VLA) models built upon Chain-of-Thought (CoT) have achieved remarkable success in advancing general-purpose robotic agents, owing to its significant perceptual comprehension. Recently, since text-only CoT struggles to adequately capture scene details in complex spatial environments, a highly promising strategy involves leveraging visual priors to guide robotic action generation. Nevertheless, these strategies face two inherent challenges: (i) a modality gap between visual observations and low-level actions, and (ii) unstable training due to competing objectives between visual prediction and action generation. To address these challenges, we propose a Vision-Integrated Trajectory Alignment (VITA) framework that learns a shared discrete latent space for vision and action, enabling joint modeling of perception and motor control. VITA introduces a implicit visual CoT: autoregressively generated tokens is simultaneously decoded into future frames predictions and robot actions, thereby internalizing visual dynamics as an inductive bias for motion planning. Extensive experiments on simulated and real-world environments demonstrate state-of-the-art performance. VITA improves 14.5\%, 9.6\% and 12.1\% over existing baselines on CALVIN, LIBERO and SimplerEnv. Furthermore, VITA attains an average success rate of 80.5\% across six real-world tasks, demonstrating its potential as a generalist robotic manipulation model.

---

## 183. Prune-Then-Plan: Step-Level Calibration for Stable Frontier Exploration in Embodied Question Answering

**论文链接:** [http://arxiv.org/abs/2511.19768v1](http://arxiv.org/abs/2511.19768v1)

**作者:** Noah Frahm, Prakrut Patel, Yue Zhang, Shoubin Yu, Mohit Bansal, Roni Sengupta

**发布时间:** 2025-11-24

**备注:** webpage: https://noahfrahm.github.io/Prune-Then-Plan-project-page/

### GPT解析

### 总结

本研究提出了一种名为'Prune-Then-Plan'的框架，通过步骤级校准稳定视觉语言模型在具身问答代理中的探索过程，解决了VLMs直接用于步骤级探索时出现的前沿振荡问题，显著提高了导航效率和答案质量。

### 背景

大型视觉语言模型(VLMs)为具身问答(EQA)代理提供了强大的语义先验，有助于开放词汇推理。然而，当VLMs直接用于步骤级探索时，经常表现出前沿振荡，这是由过度自信和校准不良引起的不稳定来回运动。

### 目的

解决VLMs在步骤级探索中因过度自信和校准不良导致的前沿振荡问题，提高导航效率和答案质量。

### 方法

提出'Prune-Then-Plan'框架，通过步骤级校准稳定探索过程。该方法不直接使用原始VLM分数，而是采用受Holm-Bonferroni启发的剪枝程序去除不合理的前沿选择，然后将最终决策委托给基于覆盖率的规划器，将过度自信的预测转化为保守、可解释的行动。

### 主要发现

集成到3D-Mem EQA框架后，该方法在视觉基础SPL和LLM-Match指标上分别比基线方法提高了49%和33%。在相同的探索预算下，该方法在OpenEQA和EXPRESS-Bench数据集上实现了更好的场景覆盖。

### 结论

Prune-Then-Plan框架通过分离VLM的预测和规划决策，成功将过度自信的预测转化为保守、可解释的行动，显著提高了具身问答代理的性能和效率。

### 翻译

大型视觉语言模型(VLMs)通过为具身问答(EQA)代理提供强大的语义先验，改进了开放词汇推理能力。然而，当直接用于步骤级探索时，VLMs经常表现出前沿振荡，这种由过度自信和校准不良引起的不稳定来回运动，导致导航效率低下和答案质量下降。我们提出了'Prune-Then-Plan'，一种通过步骤级校准稳定探索的简单有效框架。我们的方法不直接使用原始VLM分数，而是采用受Holm-Bonferroni启发的剪枝程序去除不合理的前沿选择，然后将最终决策委托给基于覆盖率的规划器。这种分离通过依赖人类水平的判断来校准VLMs的步骤级行为，将过度自信的预测转化为保守、可解释的行动。集成到3D-Mem EQA框架后，我们的方法在视觉基础SPL和LLM-Match指标上分别比基线方法提高了49%和33%。总体而言，在相同的探索预算下，我们的方法在OpenEQA和EXPRESS-Bench数据集上实现了更好的场景覆盖。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决具身问答(EQA)系统中大型视觉-语言模型(VLM)在逐级探索时的边界振荡和不稳定行为问题。这个问题很重要，因为它导致导航效率低下、答案质量下降，甚至使代理无法在有限的探索步骤内找到正确答案。VLM虽然提供了强大的语义推理能力，但其过度自信和校准不当会导致代理在探索过程中来回振荡，浪费宝贵的探索资源。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': "作者观察到VLM的过度自信是导致不稳定探索行为的主要原因，因此提出将VLM从直接决策者转变为校准过滤器。他们借鉴了3D-Mem框架作为基础系统，并采用Holm-Bonferroni多重假设检验程序作为结构化拒绝规则。作者还参考了现有的边界探索方法，但注意到这些方法主要关注整个剧集级别的行为校准，忽略了VLM在逐级探索中的校准问题，因此设计了专门针对这一问题的'先剪枝后规划'框架。", '这个方法的核心思想是什么？整体实现流程是怎样的？': "核心思想是将VLM的角色从直接决策者转变为基于校准的过滤器，通过'先剪枝后规划'框架稳定探索。具体流程：1)使用VLM评估每个边界快照的潜在效果并提取置信度；2)将置信度转换为分数并通过经验累积分布函数(ECDF)映射到p值；3)应用Holm-Bonferroni式逐步规则识别并修剪坏边界；4)在剩余边界中选择距离代理最近的边界。这种分离使VLM专注于语义修剪，而规划器负责探索效率，避免了振荡行为。", '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)逐级校准探索框架，将VLM仅用于边界拒绝而非直接决策；2)可调边界剪枝机制，由参数α控制剪枝激进程度；3)系统级集成和评估，展示在多个指标上的一致提升。相比之前工作，本文方法直接解决VLM过度自信导致的逐级不稳定问题，而非仅关注整个剧集级别的校准；使用统计剪枝而非简单启发式规则；通过人工校准数据建模坏边界分布，使系统能适应不同场景。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': "本文提出的'先剪枝后规划'方法通过将VLM的角色从直接决策者转变为基于校准的边界过滤器，有效解决了具身问答系统中VLM引导探索的边界振荡问题，显著提高了探索稳定性和答案质量。"}


### 论文摘要

Large vision-language models (VLMs) have improved embodied question answering (EQA) agents by providing strong semantic priors for open-vocabulary reasoning. However, when used directly for step-level exploration, VLMs often exhibit frontier oscillations, unstable back-and-forth movements caused by overconfidence and miscalibration, leading to inefficient navigation and degraded answer quality. We propose Prune-Then-Plan, a simple and effective framework that stabilizes exploration through step-level calibration. Instead of trusting raw VLM scores, our method prunes implausible frontier choices using a Holm-Bonferroni inspired pruning procedure and then delegates final decisions to a coverage-based planner. This separation converts overconfident predictions into conservative, interpretable actions by relying on human-level judgments to calibrate the step-level behavior of VLMs. Integrated into the 3D-Mem EQA framework, our approach achieves relative improvements of up to 49% and 33% in visually grounded SPL and LLM-Match metrics respectively over baselines. Overall, our method achieves better scene coverage under equal exploration budgets on both OpenEQA and EXPRESS-Bench datasets.

---

## 184. An Adaptive, Data-Integrated Agent-Based Modeling Framework for Explainable and Contestable Policy Design

**论文链接:** [http://arxiv.org/abs/2511.19726v1](http://arxiv.org/abs/2511.19726v1)

**作者:** Roberto Garrone

**发布时间:** 2025-11-24

**备注:** 27 pages, 2 case studies (emissions and smart grids). Preprint prepared during the author's PhD research at the Open University of Cyprus and the University of Milano-Bicocca. Introduces a unified framework for adaptive multi-agent learning with information-theoretic, causal, and clustering diagnostics

### GPT解析

### 总结

本文提出了一种通用的自适应多智能体学习框架，解决了多智能体系统在实际运行中具有反馈、适应性和非平稳性，而许多模拟研究却保留静态决策规则和固定控制参数之间的差距。

### 背景

多智能体系统通常在反馈、适应性和非平稳性条件下运行，但许多模拟研究仍使用静态决策规则和固定控制参数，导致模拟与现实系统之间存在不匹配。

### 目的

引入一个通用的自适应多智能体学习框架，使模拟研究能够更好地反映实际系统的动态特性，并分析学习智能体和自适应控制如何共同塑造系统轨迹。

### 方法

框架包含四个关键组成部分：(i)区分静态与自适应智能体以及固定与自适应系统参数的四种动态机制；(ii)使用熵率、统计复杂度和预测信息等信息论诊断方法评估可预测性和结构；(iii)结构因果模型用于明确的干预语义；(iv)从聚合数据或样本数据生成智能体层先验的程序；(v)用于识别涌现行为机制的无监督方法。

### 主要发现

该框架提供了领域中立的架构，使研究者能够在非平衡、振荡或漂移动力学条件下系统比较稳定性、性能和可解释性。

### 结论

通过提供数学定义、计算算子和实验设计模板，该框架为开发可解释和可争议的多智能体决策过程建立了结构化的方法论。

### 翻译

多智能体系统通常在反馈、适应性和非平稳性条件下运行，但许多模拟研究仍保留静态决策规则和固定控制参数。本文引入了一种通用的自适应多智能体学习框架，整合了：(i)区分静态与自适应智能体以及固定与自适应系统参数的四种动态机制；(ii)使用熵率、统计复杂度和预测信息等信息论诊断方法来评估可预测性和结构；(iii)结构因果模型用于明确的干预语义；(iv)从聚合数据或样本数据生成智能体层先验的程序；(v)用于识别涌现行为机制的无监督方法。该框架提供了领域中立的架构，用于分析学习智能体和自适应控制如何共同塑造系统轨迹，使研究者能够在非平衡、振荡或漂移动力学条件下系统比较稳定性、性能和可解释性。论文提供了数学定义、计算算子和实验设计模板，为开发可解释和可争议的多智能体决策过程建立了结构化的方法论。


### 论文摘要

Multi-agent systems often operate under feedback, adaptation, and non-stationarity, yet many simulation studies retain static decision rules and fixed control parameters. This paper introduces a general adaptive multi-agent learning framework that integrates: (i) four dynamic regimes distinguishing static versus adaptive agents and fixed versus adaptive system parameters; (ii) information-theoretic diagnostics (entropy rate, statistical complexity, and predictive information) to assess predictability and structure; (iii) structural causal models for explicit intervention semantics; (iv) procedures for generating agent-level priors from aggregate or sample data; and (v) unsupervised methods for identifying emergent behavioral regimes. The framework offers a domain-neutral architecture for analyzing how learning agents and adaptive controls jointly shape system trajectories, enabling systematic comparison of stability, performance, and interpretability across non-equilibrium, oscillatory, or drifting dynamics. Mathematical definitions, computational operators, and an experimental design template are provided, yielding a structured methodology for developing explainable and contestable multi-agent decision processes.

---

## 185. Does Understanding Inform Generation in Unified Multimodal Models? From Analysis to Path Forward

**论文链接:** [http://arxiv.org/abs/2511.20561v1](http://arxiv.org/abs/2511.20561v1)

**作者:** Yuwei Niu, Weiyang Jin, Jiaqi Liao, Chaoran Feng, Peng Jin, Bin Lin, Zongjian Li, Bin Zhu, Weihao Yu, Li Yuan

**发布时间:** 2025-11-25

### GPT解析

### 总结

研究统一多模态模型中理解与生成之间的关系，发现存在显著的理解-生成差距，主要体现在推理生成和知识转移两个维度，并提出了解决方案。

### 背景

统一多模态模型近年来取得显著进展，但理解是否真正指导生成的基本问题仍未解决。

### 目的

调查理解与生成之间的关系，特别是理解是否真正指导生成。

### 方法

引入UniSandbox解耦评估框架，配合合成的、受控的数据集，避免数据泄露并实现详细分析。

### 主要发现

1) 存在显著的理解-生成差距，主要体现在推理生成和知识转移两个维度；2) 对于推理生成任务，理解模块中的显式思维链(CoT)能有效弥合差距；3) 自训练方法可内化这种能力，实现生成过程中的隐式推理；4) 对于知识转移任务，CoT通过帮助检索新学习的知识辅助生成过程；5) 基于查询的架构具有潜在的类CoT特性，影响知识转移。

### 结论

UniSandbox为设计未来真正弥合理解与生成之间差距的统一架构和训练策略提供了初步见解。

### 翻译

近年来，统一多模态模型取得了显著进展，但一个基本问题仍然存在：理解是否真正指导生成？为了研究这一问题，我们引入了UniSandbox，这是一个解耦的评估框架，配合合成的、受控的数据集，以避免数据泄露并实现详细分析。我们的研究发现了显著的理解-生成差距，主要体现在两个关键维度：推理生成和知识转移。具体来说，在推理生成任务中，我们观察到理解模块中的显式思维链(CoT)有效地弥合了这一差距，并进一步证明自训练方法可以成功内化这种能力，使生成过程中能够进行隐式推理。此外，在知识转移任务中，我们发现CoT通过帮助检索新学习的知识来辅助生成过程，还发现基于查询的架构本质上具有潜在的类CoT特性，影响这种转移。UniSandbox为设计未来真正弥合理解与生成之间差距的统一架构和训练策略提供了初步见解。代码和数据可在https://github.com/PKU-YuanGroup/UniSandBox获取。


### 论文摘要

Recent years have witnessed significant progress in Unified Multimodal Models, yet a fundamental question remains: Does understanding truly inform generation? To investigate this, we introduce UniSandbox, a decoupled evaluation framework paired with controlled, synthetic datasets to avoid data leakage and enable detailed analysis. Our findings reveal a significant understanding-generation gap, which is mainly reflected in two key dimensions: reasoning generation and knowledge transfer. Specifically, for reasoning generation tasks, we observe that explicit Chain-of-Thought (CoT) in the understanding module effectively bridges the gap, and further demonstrate that a self-training approach can successfully internalize this ability, enabling implicit reasoning during generation. Additionally, for knowledge transfer tasks, we find that CoT assists the generative process by helping retrieve newly learned knowledge, and also discover that query-based architectures inherently exhibit latent CoT-like properties that affect this transfer. UniSandbox provides preliminary insights for designing future unified architectures and training strategies that truly bridge the gap between understanding and generation. Code and data are available at https://github.com/PKU-YuanGroup/UniSandBox

---

## 186. From One Attack Domain to Another: Contrastive Transfer Learning with Siamese Networks for APT Detection

**论文链接:** [http://arxiv.org/abs/2511.20500v1](http://arxiv.org/abs/2511.20500v1)

**作者:** Sidahmed Benabderrahmane, Talal Rahwan

**发布时间:** 2025-11-25

### GPT解析

### 总结

该研究提出了一种混合迁移框架，结合迁移学习、可解释AI、对比学习和孪生网络，用于改进高级持续性威胁(APT)的跨领域泛化能力检测。

### 背景

高级持续性威胁(APT)因其隐蔽性、持续性和适应性对网络安全构成重大挑战。传统机器学习检测器面临类别不平衡、高维特征和真实世界痕迹稀少等问题，且缺乏可转移性，在训练领域表现良好但在新攻击场景中性能下降。

### 目的

开发一种能够提高跨领域泛化能力的APT检测方法，解决传统检测器的局限性。

### 方法

结合迁移学习、可解释AI(XAI)、对比学习和孪生网络的混合框架；使用基于注意力的自编码器支持跨领域知识转移；利用Shapley加性解释(SHAP)选择稳定且信息丰富的特征以减少维度和计算成本；通过对比目标训练的孪生编码器对齐源域和目标域表示，增加异常可分性并减轻特征漂移。

### 主要发现

在DARPA透明计算(TC)项目的真实世界痕迹评估以及合成攻击场景测试中，该方法在源域到目标域的迁移中提供了比传统和深度基线方法更好的检测分数。

### 结论

该研究展示了一种可扩展、可解释且可转移的APT检测解决方案，有效解决了传统检测方法的局限性。

### 翻译

高级持续性威胁(APT)由于其隐蔽性、持续性和适应性对网络安全构成重大挑战。传统机器学习检测器难以处理类别不平衡、高维特征和真实世界痕迹稀少等问题。它们通常缺乏可转移性——在训练领域表现良好但在新型攻击场景中性能下降。我们提出了一种混合迁移框架，整合了迁移学习、可解释AI(XAI)、对比学习和孪生网络，以提高跨领域泛化能力。基于注意力的自编码器支持跨领域知识转移，而Shapley加性解释(SHAP)则选择稳定且信息丰富的特征以降低维度和计算成本。通过对比目标训练的孪生编码器对齐源域和目标域表示，增加了异常可分性并减轻了特征漂移。我们在DARPA透明计算(TC)项目的真实世界痕迹上进行了评估，并添加了合成攻击场景以测试鲁棒性。在源域到目标域的迁移中，该方法比传统和深度基线提供了更好的检测分数，展示了一种可扩展、可解释且可转移的APT检测解决方案。


### 论文摘要

Advanced Persistent Threats (APT) pose a major cybersecurity challenge due to their stealth, persistence, and adaptability. Traditional machine learning detectors struggle with class imbalance, high dimensional features, and scarce real world traces. They often lack transferability-performing well in the training domain but degrading in novel attack scenarios. We propose a hybrid transfer framework that integrates Transfer Learning, Explainable AI (XAI), contrastive learning, and Siamese networks to improve cross-domain generalization. An attention-based autoencoder supports knowledge transfer across domains, while Shapley Additive exPlanations (SHAP) select stable, informative features to reduce dimensionality and computational cost. A Siamese encoder trained with a contrastive objective aligns source and target representations, increasing anomaly separability and mitigating feature drift. We evaluate on real-world traces from the DARPA Transparent Computing (TC) program and augment with synthetic attack scenarios to test robustness. Across source to target transfers, the approach delivers improved detection scores with classical and deep baselines, demonstrating a scalable, explainable, and transferable solution for APT detection.

---

## 187. Ranking-Enhanced Anomaly Detection Using Active Learning-Assisted Attention Adversarial Dual AutoEncoders

**论文链接:** [http://arxiv.org/abs/2511.20480v1](http://arxiv.org/abs/2511.20480v1)

**作者:** Sidahmed Benabderrahmane, James Cheney, Talal Rahwan

**发布时间:** 2025-11-25

### GPT解析

### 总结

本文提出了一种创新方法，利用自编码器进行无监督异常检测，并结合主动学习来迭代提高高级持续性威胁(APT)异常检测能力。通过选择性查询不确定或模糊样本的标签，最小化标记成本同时提高检测率，使模型能在最少数据情况下提高准确性并减少手动标记需求。

### 背景

高级持续性威胁(APTs)在网络安全中构成重大挑战，因其具有隐蔽性和长期性。现代监督学习方法需要大量标记数据，但在现实世界的网络安全环境中这类数据通常稀缺。

### 目的

提出一种结合自编码器和主动学习的创新方法，迭代提高APT异常检测能力；通过选择性查询不确定样本标签，最小化标记成本并提高检测率；使模型在最少数据情况下提高准确性，减少大量手动标记需求。

### 方法

提出基于注意力对抗双自编码器的异常检测框架；使用主动学习循环迭代增强模型；在DARPA透明计算计划产生的真实世界不平衡来源跟踪数据库上评估；数据涵盖Android、Linux、BSD和Windows等多种操作系统及两种攻击场景。

### 主要发现

主动学习期间检测率显著提高；与其他现有方法相比表现更好；在APT-like攻击仅占数据0.004%的情况下能有效工作；跨操作系统环境中表现良好。

### 结论

所提出方法通过结合自编码器和主动学习，有效解决了网络安全环境中标记数据稀缺问题；能在最小化标记成本的同时提高检测准确性，特别适用于检测罕见的APT攻击。

### 翻译

高级持续性威胁(APTs)由于其隐蔽性和长期性在网络安全中构成重大挑战。现代监督学习方法需要大量标记数据，这在现实世界的网络安全环境中通常稀缺。本文提出了一种创新方法，利用自编码器进行无监督异常检测，并结合主动学习来迭代提高APT异常检测。通过选择性查询不确定或模糊样本的标签，我们最小化标记成本同时提高检测率，使模型能在最少数据情况下提高检测准确性并减少大量手动标记需求。我们详细阐述了所提出的基于注意力对抗双自编码器的异常检测框架，并展示了主动学习循环如何迭代增强模型。该框架在DARPA透明计算计划产生的真实世界不平衡来源跟踪数据库上进行了评估，其中APT-like攻击仅占数据的0.004%。数据集涵盖多种操作系统，包括Android、Linux、BSD和Windows，并覆盖两种攻击场景。结果表明，在主动学习期间检测率显著提高，并且与其他现有方法相比表现更好。


### 论文摘要

Advanced Persistent Threats (APTs) pose a significant challenge in cybersecurity due to their stealthy and long-term nature. Modern supervised learning methods require extensive labeled data, which is often scarce in real-world cybersecurity environments. In this paper, we propose an innovative approach that leverages AutoEncoders for unsupervised anomaly detection, augmented by active learning to iteratively improve the detection of APT anomalies. By selectively querying an oracle for labels on uncertain or ambiguous samples, we minimize labeling costs while improving detection rates, enabling the model to improve its detection accuracy with minimal data while reducing the need for extensive manual labeling. We provide a detailed formulation of the proposed Attention Adversarial Dual AutoEncoder-based anomaly detection framework and show how the active learning loop iteratively enhances the model. The framework is evaluated on real-world imbalanced provenance trace databases produced by the DARPA Transparent Computing program, where APT-like attacks constitute as little as 0.004\% of the data. The datasets span multiple operating systems, including Android, Linux, BSD, and Windows, and cover two attack scenarios. The results have shown significant improvements in detection rates during active learning and better performance compared to other existing approaches.

---

## 188. Towards Trustworthy Wi-Fi Sensing: Systematic Evaluation of Deep Learning Model Robustness to Adversarial Attacks

**论文链接:** [http://arxiv.org/abs/2511.20456v1](http://arxiv.org/abs/2511.20456v1)

**作者:** Shreevanth Krishnaa Gopalakrishnan, Stephen Hailes

**发布时间:** 2025-11-25

**备注:** 19 pages, 8 figures, 7 tables

### GPT解析

### 总结

本研究评估了基于信道状态信息的深度学习模型在对抗攻击下的鲁棒性，发现较小模型虽然高效但在干净数据上性能相当却明显不够鲁棒，物理可实现的信号空间扰动比不受约束的特征空间攻击成功率低，对抗训练可提高鲁棒性同时保持适度性能。

### 背景

机器学习已成为基于信道状态信息的人体感知系统的重要组成部分，有望在未来蜂窝和WiFi网络中实现无设备活动识别和身份检测。然而，这些系统依赖于决策可能被微妙扰动的模型，引发了对普遍感知环境中安全和可靠性的担忧。

### 目的

量化和理解CSI深度学习模型的鲁棒性（定义为在对抗性扰动下保持准确预测的能力），为无线感知在现实环境中的安全部署提供关键依据。

### 方法

对CSI深度学习模型在不同威胁模型（白盒、黑盒/转移和通用扰动）和不同程度攻击现实性下的鲁棒性进行系统评估。建立比较框架，在三个公共数据集上比较紧凑的时间自编码器模型与更大的深度架构，量化模型规模、训练机制和物理约束对鲁棒性的影响。

### 主要发现

1) 较小模型虽然高效且在干净数据上性能相当，但明显不够鲁棒；2) 物理上可实现的信号空间扰动显著降低了攻击成功率；3) 对抗训练可以缓解这些漏洞，在两种模型类别中都仅适度降低了干净数据的性能，同时提高了平均鲁棒准确率。

### 结论

随着无线感知向可靠、跨域操作发展，这些发现为鲁棒性估计提供了定量基准，为安全和可信的人中心感知系统设计提供了指导原则。

### 翻译

机器学习已成为基于信道状态信息的人体感知系统的组成部分，并有望在未来蜂窝和WiFi代次中支持无设备活动识别和身份检测等应用。然而，这些系统依赖于其决策可能被微妙扰动的模型，引发了普遍感知环境中安全和可靠性的担忧。因此，在无线感知能够安全部署于现实环境之前，量化和理解此类模型的鲁棒性（定义为它们在对抗性扰动下保持准确预测的能力）至关重要。本研究对CSI深度学习模型在不同威胁模型（白盒、黑盒/转移和通用扰动）和不同程度攻击现实性下的鲁棒性进行了系统评估。我们建立了一个比较框架，在三个公共数据集上比较紧凑的时间自编码器模型与更大的深度架构，量化了模型规模、训练机制和物理约束如何影响鲁棒性。实验表明，较小模型虽然高效且在干净数据上性能相当，但明显不够鲁棒。我们进一步确认，物理上可实现的信号空间扰动（设计为在真实无线信道中可行）与不受约束的特征空间攻击相比，显著降低了攻击成功率。对抗训练可以缓解这些漏洞，在两种模型类别中都仅适度降低了干净数据的性能，同时提高了平均鲁棒准确率。随着无线感知向可靠、跨域操作发展，这些发现为鲁棒性估计提供了定量基准，并为安全和可信的人中心感知系统设计提供了指导原则。


### 论文摘要

Machine learning has become integral to Channel State Information (CSI)-based human sensing systems and is expected to power applications such as device-free activity recognition and identity detection in future cellular and Wi-Fi generations. However, these systems rely on models whose decisions can be subtly perturbed, raising concerns for security and reliability in ubiquitous sensing. Quantifying and understanding the robustness of such models, defined as their ability to maintain accurate predictions under adversarial perturbations, is therefore critical before wireless sensing can be safely deployed in real-world environments.   This work presents a systematic evaluation of the robustness of CSI deep learning models under diverse threat models (white-box, black-box/transfer, and universal perturbations) and varying degrees of attack realism. We establish a framework to compare compact temporal autoencoder models with larger deep architectures across three public datasets, quantifying how model scale, training regime, and physical constraints influence robustness. Our experiments show that smaller models, while efficient and equally performant on clean data, are markedly less robust. We further confirm that physically realizable signal-space perturbations, designed to be feasible in real wireless channels, significantly reduce attack success compared to unconstrained feature-space attacks. Adversarial training mitigates these vulnerabilities, improving mean robust accuracy with only moderate degradation in clean performance across both model classes. As wireless sensing advances towards reliable, cross-domain operation, these findings provide quantitative baselines for robustness estimation and inform design principles for secure and trustworthy human-centered sensing systems.

---

## 189. DRL-Guided Neural Batch Sampling for Semi-Supervised Pixel-Level Anomaly Detection

**论文链接:** [http://arxiv.org/abs/2511.20270v1](http://arxiv.org/abs/2511.20270v1)

**作者:** Amirhossein Khadivi Noghredeh, Abdollah Safari, Fatemeh Ziaeetabar, Firoozeh Haghighi

**发布时间:** 2025-11-25

### GPT解析

### 总结

提出了一种半监督深度强化学习框架，用于解决工业视觉检测中缺陷样本稀缺导致的异常检测挑战，通过结合神经批量采样器、自动编码器和预测器，有效提高了对细微缺陷的检测和定位能力。

### 背景

工业视觉检测中的异常检测具有挑战性，主要是因为缺陷样本稀缺。大多数现有方法仅依赖正常数据进行无监督重建，这通常导致过拟合和对细微缺陷的检测效果不佳。

### 目的

提出一种半监督深度强化学习框架，有效利用有限的标记数据，同时学习正常和缺陷模式，提高工业视觉检测中异常检测的准确性。

### 方法

提出了一种半监督深度强化学习框架，集成了神经批量采样器、自动编码器和预测器。基于强化学习的采样器通过复合奖励平衡探索和利用，自适应地选择信息丰富的图像块。自动编码器生成突出显示异常区域的损失分布图，而预测器在损失分布空间中进行分割。

### 主要发现

在MVTec AD数据集上的实验表明，该方法比最近的最新方法具有更高的准确性和更好的细微异常定位能力，同时保持低复杂度。F1_max平均提高了0.15，AUC平均提高了0.06，最佳情况下F1_max最大提高了0.37。

### 结论

提出的半监督深度强化学习框架能够有效解决工业视觉检测中缺陷样本稀缺的问题，显著提高异常检测的准确性和细微缺陷的定位能力。

### 翻译

工业视觉检测中的异常检测具有挑战性，原因是缺陷样本稀缺。大多数现有方法仅依赖正常数据进行无监督重建，常常导致过拟合和对细微缺陷的检测效果不佳。我们提出了一种半监督深度强化学习框架，集成了神经批量采样器、自动编码器和预测器。基于强化学习的采样器通过复合奖励平衡探索和利用，自适应地选择信息丰富的图像块。自动编码器生成突出显示异常区域的损失分布图，而预测器在损失分布空间中进行分割。这种交互使系统能够有效地利用有限的标记数据学习正常和缺陷模式。在MVTec AD数据集上的实验表明，我们的方法比最近的最新方法具有更高的准确性和更好的细微异常定位能力，同时保持低复杂度，F1_max平均提高了0.15，AUC平均提高了0.06，最佳情况下F1_max最大提高了0.37。


### 论文摘要

Anomaly detection in industrial visual inspection is challenging due to the scarcity of defective samples. Most existing methods rely on unsupervised reconstruction using only normal data, often resulting in overfitting and poor detection of subtle defects. We propose a semi-supervised deep reinforcement learning framework that integrates a neural batch sampler, an autoencoder, and a predictor. The RL-based sampler adaptively selects informative patches by balancing exploration and exploitation through a composite reward. The autoencoder generates loss profiles highlighting abnormal regions, while the predictor performs segmentation in the loss-profile space. This interaction enables the system to effectively learn both normal and defective patterns with limited labeled data. Experiments on the MVTec AD dataset demonstrate that our method achieves higher accuracy and better localization of subtle anomalies than recent state-of-the-art approaches while maintaining low complexity, yielding an average improvement of 0.15 in F1_max and 0.06 in AUC, with a maximum gain of 0.37 in F1_max in the best case.

---

## 190. Modality-Balanced Collaborative Distillation for Multi-Modal Domain Generalization

**论文链接:** [http://arxiv.org/abs/2511.20258v1](http://arxiv.org/abs/2511.20258v1)

**作者:** Xiaohan Wang, Zhangtao Cheng, Ting Zhong, Leiting Chen, Fan Zhou

**发布时间:** 2025-11-25

### GPT解析

### 总结

本文提出了一种名为MBCD的统一协同蒸馏框架，用于解决多模态领域泛化中权重平均(WA)技术的应用挑战，通过三个关键步骤有效促进了模态融合并提高了泛化性能。

### 背景

权重平均(WA)是一种通过促进收敛到平坦损失景观来增强泛化能力的强大技术，与更强的分布外性能相关。然而，将WA直接应用于多模态领域泛化(MMDG)时，不同模态间的优化速度差异会导致WA在早期过度拟合收敛较快的模态，抑制收敛较慢但互补的模态，阻碍有效模态融合。

### 目的

解决在多模态领域泛化中直接应用权重平均技术所面临的挑战，克服模态优化速度差异导致的早期偏差问题，促进有效模态融合，引导收敛到更平坦、更可泛化的解。

### 方法

MBCD框架包含三个主要步骤：1)自适应模态丢弃在学生模型中实施，抑制早期对主导模态的偏见；2)梯度一致性约束对齐单模态分支和融合表示间的学习信号，鼓励协调优化；3)基于WA的教师进行跨模态蒸馏，将融合知识传输到各单模态分支，加强跨模态交互。

### 主要发现

在多模态领域泛化基准上的大量实验表明，MBCD始终优于现有方法，在多样化的未见领域上实现了更高的准确性和鲁棒性。

### 结论

MBCD框架成功解决了多模态领域泛化中WA技术的应用挑战，通过自适应模态丢弃、梯度一致性约束和跨模态蒸馏，有效促进了模态融合并引导模型收敛到更平坦的损失景观，提高了泛化性能。

### 翻译

权重平均(WA)已成为一种强大的技术，通过促进收敛到平坦损失景观来增强泛化能力，这与更强的分布外性能相关。然而，将WA直接应用于多模态领域泛化(MMDG)具有挑战性：不同模态之间的优化速度差异导致WA在早期阶段过度拟合收敛较快的模态，抑制了收敛较慢但互补的模态的贡献，从而阻碍有效的模态融合，并将损失表面推向更尖锐、泛化能力更弱的极小值。为解决这个问题，我们提出了MBCD，这是一个统一的协同蒸馏框架，它保留了WA的平坦诱导优势，同时克服了其在多模态环境中的缺点。MBCD首先在学生模型中进行自适应模态丢弃，以抑制早期阶段对主导模态的偏见。然后，梯度一致性约束对齐单模态分支和融合表示之间的学习信号，鼓励协调和更平滑的优化。最后，基于WA的教师通过将融合知识传输到每个单模态分支进行跨模态蒸馏，这加强了跨模态交互，并将收敛引导向更平坦的解。在MMDG基准上的大量实验表明，MBCD始终优于现有方法，在多样化的未见领域上实现了更高的准确性和鲁棒性。


### 论文摘要

Weight Averaging (WA) has emerged as a powerful technique for enhancing generalization by promoting convergence to a flat loss landscape, which correlates with stronger out-of-distribution performance. However, applying WA directly to multi-modal domain generalization (MMDG) is challenging: differences in optimization speed across modalities lead WA to overfit to faster-converging ones in early stages, suppressing the contribution of slower yet complementary modalities, thereby hindering effective modality fusion and skewing the loss surface toward sharper, less generalizable minima. To address this issue, we propose MBCD, a unified collaborative distillation framework that retains WA's flatness-inducing advantages while overcoming its shortcomings in multi-modal contexts. MBCD begins with adaptive modality dropout in the student model to curb early-stage bias toward dominant modalities. A gradient consistency constraint then aligns learning signals between uni-modal branches and the fused representation, encouraging coordinated and smoother optimization. Finally, a WA-based teacher conducts cross-modal distillation by transferring fused knowledge to each uni-modal branch, which strengthens cross-modal interactions and steer convergence toward flatter solutions. Extensive experiments on MMDG benchmarks show that MBCD consistently outperforms existing methods, achieving superior accuracy and robustness across diverse unseen domains.

---

## 191. The Image as Its Own Reward: Reinforcement Learning with Adversarial Reward for Image Generation

**论文链接:** [http://arxiv.org/abs/2511.20256v1](http://arxiv.org/abs/2511.20256v1)

**作者:** Weijia Mao, Hao Chen, Zhenheng Yang, Mike Zheng Shou

**发布时间:** 2025-11-25

### GPT解析

### 总结

论文提出了Adv-GRPO框架，通过对抗性奖励机制和将图像本身作为奖励的方法，解决了强化学习在图像生成中奖励函数不可靠的问题。该方法迭代更新奖励模型和生成器，避免奖励黑客攻击，直接通过视觉输出指导生成器，提高图像质量。

### 背景

当前强化学习在图像生成中依赖于预训练的偏好模型输出标量奖励来近似人类偏好，但这些奖励无法准确捕捉人类感知且容易受到奖励黑客攻击。现有奖励函数优化虽可缓解黑客攻击，但固有偏差仍然存在，如PickScore降低图像质量，OCR奖励降低美学保真度。

### 目的

解决现有奖励函数无法准确捕捉人类偏好的问题，避免奖励黑客攻击，提高图像质量、美学和特定任务指标的一致性，实现分布转移和灵活的风格定制。

### 方法

提出Adv-GRPO框架，具有对抗性奖励，迭代更新奖励模型和生成器。奖励模型使用参考图像作为正样本进行监督，避免被黑客攻击。不同于KL正则化，学习的奖励直接通过视觉输出指导生成器。将图像本身作为奖励，使用参考图像和视觉基础模型（如DINO）提供密集视觉信号而非单一标量奖励。结合参考样本与基础模型奖励实现分布转移和风格定制。

### 主要发现

人类评估显示，该方法优于Flow-GRPO和SD3，在图像质量和美学方面分别达到70.0%和72.4%的胜率。密集视觉信号在图像质量、美学和特定任务指标上带来了一致的提升。

### 结论

Adv-GRPO通过对抗性奖励和直接使用图像作为奖励的方法，有效解决了传统奖励函数的问题。该方法能够生成更高质量的图像，具有更好的美学表现，并实现了分布转移和风格定制的能力。

### 翻译

可靠的奖励函数对图像生成中的强化学习至关重要。大多数当前的强化学习方法依赖于预训练的偏好模型，这些模型输出标量奖励来近似人类偏好。然而，这些奖励通常无法捕捉人类感知，且容易受到奖励黑客攻击，即高分并不对应更好的图像。为此，我们引入了Adv-GRPO，这是一种具有对抗性奖励的强化学习框架，迭代更新奖励模型和生成器。奖励模型使用参考图像作为正样本进行监督，可以很大程度上避免被黑客攻击。与约束参数更新的KL正则化不同，我们学习的奖励直接通过视觉输出指导生成器，从而产生更高质量的图像。此外，虽然优化现有奖励函数可以缓解奖励黑客攻击，但它们的固有偏差仍然存在。例如，PickScore可能会降低图像质量，而基于OCR的奖励通常会降低美学保真度。为解决这一问题，我们将图像本身作为奖励，使用参考图像和视觉基础模型（如DINO）提供丰富的视觉奖励。这些密集的视觉信号而非单一标量，在图像质量、美学和特定任务指标上带来了一致的提升。最后，我们表明结合参考样本与基础模型奖励能够实现分布转移和灵活的风格定制。在人类评估中，我们的方法优于Flow-GRPO和SD3，在图像质量和美学方面分别达到70.0%和72.4%的胜率。代码和模型已发布。


### 论文摘要

A reliable reward function is essential for reinforcement learning (RL) in image generation. Most current RL approaches depend on pre-trained preference models that output scalar rewards to approximate human preferences. However, these rewards often fail to capture human perception and are vulnerable to reward hacking, where higher scores do not correspond to better images. To address this, we introduce Adv-GRPO, an RL framework with an adversarial reward that iteratively updates both the reward model and the generator. The reward model is supervised using reference images as positive samples and can largely avoid being hacked. Unlike KL regularization that constrains parameter updates, our learned reward directly guides the generator through its visual outputs, leading to higher-quality images. Moreover, while optimizing existing reward functions can alleviate reward hacking, their inherent biases remain. For instance, PickScore may degrade image quality, whereas OCR-based rewards often reduce aesthetic fidelity. To address this, we take the image itself as a reward, using reference images and vision foundation models (e.g., DINO) to provide rich visual rewards. These dense visual signals, instead of a single scalar, lead to consistent gains across image quality, aesthetics, and task-specific metrics. Finally, we show that combining reference samples with foundation-model rewards enables distribution transfer and flexible style customization. In human evaluation, our method outperforms Flow-GRPO and SD3, achieving 70.0% and 72.4% win rates in image quality and aesthetics, respectively. Code and models have been released.

---

## 192. DiCaP: Distribution-Calibrated Pseudo-labeling for Semi-Supervised Multi-Label Learning

**论文链接:** [http://arxiv.org/abs/2511.20225v1](http://arxiv.org/abs/2511.20225v1)

**作者:** Bo Han, Zhuoming Li, Xiaoyu Wang, Yaxin Hou, Hui Liu, Junhui Hou, Yuheng Jia

**发布时间:** 2025-11-25

**备注:** Accepted by AAAI-26

### GPT解析

### 总结

本文提出了一种Distribution-Calibrated Pseudo-labeling (DiCaP)框架，通过估计后验精度校准伪标签权重，并引入双阈值机制分离自信和模糊区域，有效解决了半监督多标签学习中伪标签质量差异的问题。

### 背景

半监督多标签学习旨在通过利用未标记数据来解决多标签学习中标记数据有限的挑战。目前，伪标签已成为半监督多标签学习的主导策略，但大多数现有方法对所有伪标签分配相同权重，忽略了它们的质量差异。

### 目的

解决现有方法对所有伪标签分配相同权重的问题，提出一种能够根据伪标签正确性可能性分配权重的框架，以提高模型性能。

### 方法

提出Distribution-Calibrated Pseudo-labeling (DiCaP)框架，通过估计后验精度校准伪标签权重，并引入双阈值机制分离自信和模糊区域：自信样本进行伪标签并相应加权，模糊样本则通过无监督对比学习进行探索。

### 主要发现

理论上验证了最优权重应反映伪标签的正确性可能性；实验观察到在同一数据集上，未标记数据的正确性可能性分布保持稳定，即使标记训练样本数量变化。

### 结论

在多个基准数据集上的实验验证了该方法的一致性改进，超越了最先进方法，性能提升最高达4.27%。

### 翻译

半监督多标签学习旨在通过利用未标记数据来解决多标签学习中标记数据有限的挑战。虽然伪标签已成为半监督多标签学习的主导策略，但大多数现有方法对所有伪标签分配相同权重，而不管它们的质量如何，这会放大噪声或不确定预测的影响并降低整体性能。在本文中，我们从理论上验证了伪标签的最优权重应反映其正确性可能性。经验上，我们观察到在同一数据集上，未标记数据的正确性可能性分布保持稳定，即使标记训练样本的数量发生变化。基于这一见解，我们提出了分布校准伪标签(DiCaP)，这是一个正确感知的框架，通过估计后验精度来校准伪标签权重。我们进一步引入了双阈值机制来分离自信和模糊区域：自信样本进行伪标签并相应加权，而模糊样本则通过无监督对比学习进行探索。在多个基准数据集上进行的实验验证了我们的方法实现了一致的改进，超越了最先进方法，最高提升4.27%。


### 论文摘要

Semi-supervised multi-label learning (SSMLL) aims to address the challenge of limited labeled data in multi-label learning (MLL) by leveraging unlabeled data to improve the model's performance. While pseudo-labeling has become a dominant strategy in SSMLL, most existing methods assign equal weights to all pseudo-labels regardless of their quality, which can amplify the impact of noisy or uncertain predictions and degrade the overall performance. In this paper, we theoretically verify that the optimal weight for a pseudo-label should reflect its correctness likelihood. Empirically, we observe that on the same dataset, the correctness likelihood distribution of unlabeled data remains stable, even as the number of labeled training samples varies. Building on this insight, we propose Distribution-Calibrated Pseudo-labeling (DiCaP), a correctness-aware framework that estimates posterior precision to calibrate pseudo-label weights. We further introduce a dual-thresholding mechanism to separate confident and ambiguous regions: confident samples are pseudo-labeled and weighted accordingly, while ambiguous ones are explored by unsupervised contrastive learning. Experiments conducted on multiple benchmark datasets verify that our method achieves consistent improvements, surpassing state-of-the-art methods by up to 4.27%.

---

## 193. WPT: World-to-Policy Transfer via Online World Model Distillation

**论文链接:** [http://arxiv.org/abs/2511.20095v1](http://arxiv.org/abs/2511.20095v1)

**作者:** Guangfeng Jiang, Yueru Luo, Jun Liu, Yi Huang, Yiyao Zhu, Zhan Qu, Dave Zhenyu Chen, Bingbing Liu, Xu Yan

**发布时间:** 2025-11-25

### GPT解析

### 总结

本文提出了一种名为WPT(World-to-Policy Transfer)的训练范式，通过在线蒸馏方法解决了现有世界模型中存在的运行时耦合和离线奖励信号依赖问题，实现了高性能与实时性的平衡。

### 背景

近年来世界模型取得了显著进展，主要旨在捕捉智能体动作与变化环境之间的时空相关性。然而，现有方法常面临运行时紧密耦合或依赖离线奖励信号的问题，导致推理开销大或阻碍端到端优化。

### 目的

克服现有世界模型的局限性，引入一种能够实现在线蒸馏的训练范式，在保持实时部署能力的同时提高规划性能。

### 方法

开发可训练的奖励模型，将候选轨迹与世界模型预测的未来动力学对齐，将世界知识注入教师策略；提出策略蒸馏和世界奖励蒸馏，将教师的推理能力转移到轻量级学生策略中。

### 主要发现

在开环和闭环基准测试中，WPT实现了最先进的性能：开环碰撞率为0.11，闭环驾驶得分为79.23，超越了基于世界模型和模仿学习的方法；学生推理速度提高了4.9倍，同时保留了大部分性能增益。

### 结论

WPT方法有效解决了现有世界模型的局限性，通过在线蒸馏和知识转移，实现了性能和推理效率的平衡。

### 翻译

近年来，世界模型取得了显著进展，其主要目标是捕捉智能体动作与变化环境之间的时空相关性。然而，现有方法常面临运行时紧密耦合或依赖离线奖励信号的问题，导致大量推理开销或阻碍端到端优化。为克服这些局限性，我们引入了WPT，一种世界模型到策略转移的训练范式，实现了在端到端世界模型指导下的在线蒸馏。具体而言，我们开发了一个可训练的奖励模型，通过将候选轨迹与世界模型预测的未来动力学对齐，将世界知识注入教师策略。随后，我们提出策略蒸馏和世界奖励蒸馏，将教师的推理能力转移到轻量级学生策略中，在保持实时部署能力的同时提高规划性能。在开环和闭环基准上的大量实验表明，我们的WPT使用简单的策略架构实现了最先进的性能：开环碰撞率为0.11，闭环驾驶得分为79.23，在准确性和安全性上超越了基于世界模型和模仿学习的方法。此外，学生推理速度提高了4.9倍，同时保留了大部分性能增益。


### 论文摘要

Recent years have witnessed remarkable progress in world models, which primarily aim to capture the spatio-temporal correlations between an agent's actions and the evolving environment. However, existing approaches often suffer from tight runtime coupling or depend on offline reward signals, resulting in substantial inference overhead or hindering end-to-end optimization. To overcome these limitations, we introduce WPT, a World-to-Policy Transfer training paradigm that enables online distillation under the guidance of an end-to-end world model. Specifically, we develop a trainable reward model that infuses world knowledge into a teacher policy by aligning candidate trajectories with the future dynamics predicted by the world model. Subsequently, we propose policy distillation and world reward distillation to transfer the teacher's reasoning ability into a lightweight student policy, enhancing planning performance while preserving real-time deployability. Extensive experiments on both open-loop and closed-loop benchmarks show that our WPT achieves state-of-the-art performance with a simple policy architecture: it attains a 0.11 collision rate (open-loop) and achieves a 79.23 driving score (closed-loop) surpassing both world-model-based and imitation-learning methods in accuracy and safety. Moreover, the student sustains up to 4.9x faster inference, while retaining most of the gains.

---

## 194. Zero-Shot Transfer Capabilities of the Sundial Foundation Model for Leaf Area Index Forecasting

**论文链接:** [http://arxiv.org/abs/2511.20004v1](http://arxiv.org/abs/2511.20004v1)

**作者:** Peining Zhang, Hongchen Qin, Haochen Zhang, Ziqi Guo, Guiling Wang, Jinbo Bi

**发布时间:** 2025-11-25

### GPT解析

### 总结

研究展示了时间序列基础模型在农业监测中的零样本预测能力，特别是Sundial模型在足够长的上下文窗口下可以超越专门训练的LSTM模型，为农业和环境应用提供了有效的即插即用预测工具。

### 背景

研究关注时间序列基础模型在农业监测中用于叶面积指数(LAI)预测的零样本预测能力，使用了HiQ数据集(美国，2000-2022年)。

### 目的

系统比较统计基线方法、完全监督的LSTM和Sundial基础模型在多种评估协议下的表现，探究基础模型在遥感时间序列预测中是否可以超越专门监督模型的能力。

### 方法

使用HiQ数据集(美国，2000-2022年)，比较统计基线、完全监督的LSTM和Sundial基础模型在多种评估协议下的表现，测试Sundial在不同输入上下文窗口长度下的表现。

### 主要发现

在零样本设置下，Sundial可以超越完全训练好的LSTM，前提是输入上下文窗口足够长，具体来说，当覆盖超过一到两个完整季节周期时表现更好。这是首次证明通用基础模型可以在无需任务特定调整的情况下，在遥感时间序列预测中超越专门的监督模型。

### 结论

预训练的时间序列基础模型在农业和环境应用中作为有效的即插即用预测器具有强大潜力。

### 翻译

这项工作研究了时间序列基础模型在农业监测中用于叶面积指数(LAI)预测的零样本预测能力。使用HiQ数据集(美国，2000-2022年)，我们系统比较了统计基线、完全监督的LSTM和Sundial基础模型在多种评估协议下的表现。我们发现，在零样本设置下，Sundial可以超越完全训练好的LSTM，前提是输入上下文窗口足够长—具体来说，当覆盖超过一到两个完整季节周期时。这首次证明，通用基础模型可以在无需任务特定调整的情况下，在遥感时间序列预测中超越专门的监督模型。这些结果强调了预训练的时间序列基础模型在农业和环境应用中作为有效的即插即用预测器的强大潜力。


### 论文摘要

This work investigates the zero-shot forecasting capability of time-series foundation models for Leaf Area Index (LAI) forecasting in agricultural monitoring. Using the HiQ dataset (U.S., 2000-2022), we systematically compare statistical baselines, a fully supervised LSTM, and the Sundial foundation model under multiple evaluation protocols. We find that Sundial, in the zero-shot setting, can outperform a fully trained LSTM provided that the input context window is sufficiently long-specifically, when covering more than one or two full seasonal cycles. This demonstrates, for the first time, that a general-purpose foundation model can surpass specialized supervised models on remote-sensing time series prediction without any task-specific tuning. These results highlight the strong potential of pretrained time-series foundation models to serve as effective plug-and-play forecasters in agricultural and environmental applications.

---

## 195. MAPS: Preserving Vision-Language Representations via Module-Wise Proximity Scheduling for Better Vision-Language-Action Generalization

**论文链接:** [http://arxiv.org/abs/2511.19878v1](http://arxiv.org/abs/2511.19878v1)

**作者:** Chengyue Huang, Mellon M. Zhang, Robert Azarcon, Glen Chou, Zsolt Kira

**发布时间:** 2025-11-25

### GPT解析

### 总结

论文提出了一种名为MAPS的鲁棒微调框架，用于Vision-Language-Action模型，通过模块化接近调度平衡稳定性和灵活性，显著提升了模型在多种环境中的性能。

### 背景

Vision-Language-Action模型继承了预训练Vision-Language模型的强先验知识，但简单的微调往往会破坏这些表示并损害泛化能力。

### 目的

开发一种能够保持VLA模型预训练先验同时允许有效适应的微调方法，解决现有解决方案过度约束或忽略组件差异的问题。

### 方法

提出MAPS（Module-Wise Proximity Scheduling），一种系统化的接近约束放松调度方法，使视觉编码器保持接近预训练先验，同时允许面向动作的语言层更自由地适应。

### 主要发现

通过系统分析发现了一种经验顺序，可以平衡稳定性和灵活性；MAPS在多个基准测试和实际评估中一致提高了分布内和分布外性能，最高提升30%。

### 结论

保持与预训练VLM的经验指导接近性是一个简单而强大的原则，可以保留VLM到VLA转移的广泛泛化能力。

### 翻译

Vision-Language-Action模型继承了预训练Vision-Language模型的强先验知识，但简单的微调往往会破坏这些表示并损害泛化能力。现有解决方案（如冻结模块或应用统一正则化）要么过度约束适应过程，要么忽略了VLA组件的不同作用。我们提出了MAPS（Module-Wise Proximity Scheduling），这是第一个用于VLA的鲁棒微调框架。通过系统分析，我们发现了一种经验顺序，可以放松接近约束以平衡稳定性和灵活性。MAPS线性调度这种放松过程，使视觉编码器保持接近其预训练先验，同时允许面向动作的语言层更自由地适应。MAPS不引入额外参数或数据，可以无缝集成到现有VLA中。在多个基准测试和实际评估中，MAPS一致提高了分布内和分布外性能（最高达+30%）。我们的研究强调了经验指导的与预训练VLM的接近性作为保留VLM到VLA转移广泛泛化能力的简单而强大的原则。


### 论文摘要

Vision-Language-Action (VLA) models inherit strong priors from pretrained Vision-Language Models (VLMs), but naive fine-tuning often disrupts these representations and harms generalization. Existing fixes -- freezing modules or applying uniform regularization -- either overconstrain adaptation or ignore the differing roles of VLA components. We present MAPS (Module-Wise Proximity Scheduling), the first robust fine-tuning framework for VLAs. Through systematic analysis, we uncover an empirical order in which proximity constraints should be relaxed to balance stability and flexibility. MAPS linearly schedules this relaxation, enabling visual encoders to stay close to their pretrained priors while action-oriented language layers adapt more freely. MAPS introduces no additional parameters or data, and can be seamlessly integrated into existing VLAs. Across MiniVLA-VQ, MiniVLA-OFT, OpenVLA-OFT, and challenging benchmarks such as SimplerEnv, CALVIN, LIBERO, as well as real-world evaluations on the Franka Emika Panda platform, MAPS consistently boosts both in-distribution and out-of-distribution performance (up to +30%). Our findings highlight empirically guided proximity to pretrained VLMs as a simple yet powerful principle for preserving broad generalization in VLM-to-VLA transfer.

---

## 196. Temporal-Visual Semantic Alignment: A Unified Architecture for Transferring Spatial Priors from Vision Models to Zero-Shot Temporal Tasks

**论文链接:** [http://arxiv.org/abs/2511.19856v1](http://arxiv.org/abs/2511.19856v1)

**作者:** Xiangkai Ma, Han Zhang, Wenzhong Li, Sanglu Lu

**发布时间:** 2025-11-25

### GPT解析

### 总结

TimeArtist是一种时间-视觉转换框架，实现了时间序列波动与视觉概念间的语义级别对齐，可直接从时间序列生成高质量、多样化的图像。

### 背景

大型多模态模型在文本和图像模态对齐方面取得进展，但使用非视觉连续序列作为图像生成条件信号的研究不足；现有序列转'伪图像'方法无法建立语义级对齐。

### 目的

提出TimeArtist框架，开创时间序列波动与视觉概念间的语义级别对齐，实现从时间序列直接生成高质量图像。

### 方法

采用'warmup-align'范式：首先自监督训练双自编码器和共享量化器学习模态共享表示；然后冻结编码器和量化器，引入投影在表示级别对齐时间和视觉样本，建立跨模态框架。

### 主要发现

TimeArtist在图像生成指标上表现令人满意，在零样本时间任务中取得优越结果，能捕获时间波动模式实现图像风格转换。

### 结论

TimeArtist为跨模态生成建立了新范式，弥合了时间动态与视觉语义之间的差距。

### 翻译

大型多模态模型（LMMs）在文本和图像模态的对齐和内容生成方面取得了显著进展。然而，使用非视觉、连续的序列作为高保真图像生成的条件信号在很大程度上尚未被探索。此外，现有将序列转换为'伪图像'用于时间预测的方法未能建立语义级别的对齐。本文提出了TimeArtist，一个时间-视觉转换框架，开创了时间序列波动与视觉概念之间的语义级别对齐。它开创了一种'warmup-align'范式：首先，使用大规模数据集自监督训练双自编码器和共享量化器，以学习模态共享的表示。然后，冻结编码器和量化器，并引入投影以在表示级别对齐时间和视觉样本。TimeArtist建立了一个通用的跨模态框架，能够直接从时间序列生成高质量、多样化的图像，同时捕获时间波动模式以将图像渲染为风格转换。大量实验表明，TimeArtist在图像生成指标上取得了令人满意的性能，同时在零样本时间任务中也取得了优越的结果。我们的工作为跨模态生成建立了新范式，弥合了时间动态与视觉语义之间的差距。


### 论文摘要

Large Multimodal Models (LMMs) have achieved remarkable progress in aligning and generating content across text and image modalities. However, the potential of using non-visual, continuous sequential, as a conditioning signal for high-fidelity image generation remains largely unexplored. Furthermore, existing methods that convert series into "pseudo-images" for temporal forecasting fail to establish semantic-level alignment. In this paper, we propose TimeArtist, a temporal-visual conversion framework that pioneers semantic-level alignment between time series fluctuations and visual concepts. It pioneers a "warmup-align" paradigm: first, a dual-autoencoder and shared quantizer are self-supervised trained on large-scale datasets to learn modality-shared representations. Then, the encoders and quantizer are frozen, and a projection is introduced to align temporal and visual samples at the representation level. TimeArtist establishes a versatile cross-modal framework, enabling high-quality, diverse image generation directly from time series, while capturing temporal fluctuation patterns to render images as styles transfer. Extensive experiments show that TimeArtist achieves satisfactory performance in image generation metrics, while also attaining superior results in zero-shot temporal tasks. Our work establishes a new paradigm for cross-modal generation, bridging the gap between temporal dynamics and visual semantics.

---

## 197. SX-GeoTree: Self-eXplaining Geospatial Regression Tree Incorporating the Spatial Similarity of Feature Attributions

**论文链接:** [http://arxiv.org/abs/2511.19845v1](http://arxiv.org/abs/2511.19845v1)

**作者:** Chaogui Kang, Lijian Luo, Qingfeng Guan, Yu Liu

**发布时间:** 2025-11-25

**备注:** 41 pages, 7 figures, 12 tables

### GPT解析

### 总结

本文提出了一种名为SX-GeoTree的自解释地理空间回归树模型，用于解决传统决策树在捕捉空间依赖性和产生局部稳定解释方面的局限性。该模型在递归分割过程中整合了三个耦合目标：不纯度减少、空间残差控制和通过共识相似性网络上的模块度最大化实现解释鲁棒性。

### 背景

决策树在表格数据预测中仍然很重要，但存在两个主要问题：(i)难以捕捉空间依赖性，(ii)难以产生局部稳定(鲁棒)的解释。

### 目的

开发一种能够同时保持预测准确性、改善空间残差均匀性并提高解释一致性的地理空间回归树模型。

### 方法

1. 提出SX-GeoTree模型，在递归分割过程中整合三个目标：不纯度减少(MSE)、空间残差控制(全局莫兰I)和解释鲁棒性(通过共识相似性网络上的模块度最大化)；2. 构建共识相似性网络，包括地理加权回归(GWR)系数距离和SHAP归因距离；3. 将特征归因的局部Lipschitz连续性重新表述为网络社区保持问题，使能够可扩展地执行空间一致的解释，而无需对每个样本进行邻域搜索。

### 主要发现

1. 在两个示例任务上(福建县级GDP，n=83；西雅图点状房价，n=21,613)：SX-GeoTree保持了与决策树相当的预测准确性(R²差异在0.01以内)，改善了残差空间均匀性，并将归因共识提高了一倍(模块度：福建0.19对比0.09；西雅图0.10对比0.05)；2. 消融实验确认莫兰I和模块度项是互补的；移除其中任何一个都会同时降低空间残差结构和解释稳定性。

### 结论

该框架展示了如何将空间相似性(通过GWR导出的局部关系扩展到几何邻近性之外)嵌入到可解释模型中，推进了可信的地理空间机器学习，并为领域感知的可解释性提供了可转移的模板。

### 翻译

决策树在表格数据预测中仍然很重要，但在(i)捕捉空间依赖性和(ii)产生局部稳定(鲁棒)解释方面存在困难。我们提出了SX-GeoTree，一种自解释的地理空间回归树，它在递归分割过程中整合了三个耦合目标：不纯度减少(MSE)、空间残差控制(全局莫兰I)以及通过在共识相似性网络上最大化模块度实现解释鲁棒性，该网络由(a)地理加权回归(GWR)系数距离(刺激-响应相似性)和(b)SHAP归因距离(解释相似性)形成。我们将特征归因的局部Lipschitz连续性重新表述为网络社区保持问题，使能够可扩展地执行空间一致的解释，而无需对每个样本进行邻域搜索。在两个示例任务(福建县级GDP，n=83；西雅图点状房价，n=21,613)上的实验表明，SX-GeoTree保持了与决策树相当的预测准确性(R²差异在0.01以内)，同时改善了残差空间均匀性并将归因共识提高了一倍(模块度：福建0.19对比0.09；西雅图0.10对比0.05)。消融实验确认莫兰I和模块度项是互补的；移除其中任何一个都会同时降低空间残差结构和解释稳定性。该框架展示了如何将空间相似性(通过GWR导出的局部关系扩展到几何邻近性之外)嵌入到可解释模型中，推进了可信的地理空间机器学习，并为领域感知的可解释性提供了可转移的模板。


### 论文摘要

Decision trees remain central for tabular prediction but struggle with (i) capturing spatial dependence and (ii) producing locally stable (robust) explanations. We present SX-GeoTree, a self-explaining geospatial regression tree that integrates three coupled objectives during recursive splitting: impurity reduction (MSE), spatial residual control (global Moran's I), and explanation robustness via modularity maximization on a consensus similarity network formed from (a) geographically weighted regression (GWR) coefficient distances (stimulus-response similarity) and (b) SHAP attribution distances (explanatory similarity). We recast local Lipschitz continuity of feature attributions as a network community preservation problem, enabling scalable enforcement of spatially coherent explanations without per-sample neighborhood searches. Experiments on two exemplar tasks (county-level GDP in Fujian, n=83; point-wise housing prices in Seattle, n=21,613) show SX-GeoTree maintains competitive predictive accuracy (within 0.01 $R^{2}$ of decision trees) while improving residual spatial evenness and doubling attribution consensus (modularity: Fujian 0.19 vs 0.09; Seattle 0.10 vs 0.05). Ablation confirms Moran's I and modularity terms are complementary; removing either degrades both spatial residual structure and explanation stability. The framework demonstrates how spatial similarity - extended beyond geometric proximity through GWR-derived local relationships - can be embedded in interpretable models, advancing trustworthy geospatial machine learning and offering a transferable template for domain-aware explainability.

---

## 198. Clustering Approaches for Mixed-Type Data: A Comparative Study

**论文链接:** [http://arxiv.org/abs/2511.19755v1](http://arxiv.org/abs/2511.19755v1)

**作者:** Badih Ghattas, Alvaro Sanchez San-Benito

**发布时间:** 2025-11-24

**DOI:** 10.1155/jpas/2242100

### GPT解析

### 总结

该研究探讨了混合类型数据的聚类方法，比较了不同方法在各种模拟模型中的表现，并确定了影响聚类性能的关键因素。研究结果表明，KAMILA、LCM和k-prototypes方法表现最佳。

### 背景

聚类是无监督学习中广泛使用的技术，用于在数据集中寻找同质观测组。然而，混合类型数据的聚类仍然是一个挑战，因为现有的方法很少适合这项任务。

### 目的

提供不同方法在各种场景下行为的见解，通过改变一些实验因素，如聚类数量、聚类重叠程度、样本量、维度、数据集中连续变量的比例以及聚类分布。

### 方法

比较了基于距离的方法（k-prototypes、PDQ和凸k-means）和概率方法（KAMILA、贝叶斯网络混合模型和潜类模型）。使用调整兰德指数评估性能。

### 主要发现

聚类重叠程度、数据集中连续变量的比例和样本量对观察到的性能有显著影响。当变量之间存在强相互作用并且对聚类成员有明确的依赖性时，评估的任何方法都没有表现出令人满意的性能。

### 结论

在实验中，KAMILA、LCM和k-prototypes在调整兰德指数方面表现最佳。所有方法都可以在R中获得。

### 翻译

聚类被广泛用于无监督学习中，以在数据集中寻找同质的观测组。然而，混合类型数据的聚类仍然是一个挑战，因为现有的方法很少适合这项任务。本研究展示了这些方法的最先进水平，并使用各种模拟模型对它们进行了比较。比较的方法包括基于距离的方法k-prototypes、PDQ和凸k-means，以及概率方法KAMILA（用于混合大数据的KAy-means）、贝叶斯网络混合模型和潜类模型。目的是通过改变一些实验因素，如聚类数量、聚类重叠程度、样本量、维度、数据集中连续变量的比例以及聚类分布，来提供不同方法在各种场景下行为的见解。聚类重叠程度和数据集中连续变量的比例以及样本量对观察到的性能有显著影响。当变量之间存在强相互作用并且对聚类成员有明确的依赖性时，评估的任何方法都没有表现出令人满意的性能。在我们的实验中，KAMILA、LCM和k-prototypes在调整兰德指数方面表现最佳。所有方法都可以在R中获得。


### 论文摘要

Clustering is widely used in unsupervised learning to find homogeneous groups of observations within a dataset. However, clustering mixed-type data remains a challenge, as few existing approaches are suited for this task. This study presents the state-of-the-art of these approaches and compares them using various simulation models. The compared methods include the distance-based approaches k-prototypes, PDQ, and convex k-means, and the probabilistic methods KAy-means for MIxed LArge data (KAMILA), the mixture of Bayesian networks (MBNs), and latent class model (LCM). The aim is to provide insights into the behavior of different methods across a wide range of scenarios by varying some experimental factors such as the number of clusters, cluster overlap, sample size, dimension, proportion of continuous variables in the dataset, and clusters' distribution. The degree of cluster overlap and the proportion of continuous variables in the dataset and the sample size have a significant impact on the observed performances. When strong interactions exist between variables alongside an explicit dependence on cluster membership, none of the evaluated methods demonstrated satisfactory performance. In our experiments KAMILA, LCM, and k-prototypes exhibited the best performance, with respect to the adjusted rand index (ARI). All the methods are available in R.

---

## 199. Online Learning-Enhanced High Order Adaptive Safety Control

**论文链接:** [http://arxiv.org/abs/2511.19651v1](http://arxiv.org/abs/2511.19651v1)

**作者:** Lishuo Pan, Mattia Catellani, Thales C. Silva, Lorenzo Sabattini, Nora Ayanian

**发布时间:** 2025-11-24

**备注:** 8 pages, 7 figures, submitted to RA-L

### GPT解析

### 总结

本文提出了一种基于神经ODE的高阶自适应控制障碍函数，通过在线学习增强系统安全性，即使在复杂时变模型扰动下也能提高CBF认证系统的安全性。

### 背景

控制障碍函数(CBFs)是用于正式验证系统安全性的有效基于模型的工具。随着现代控制问题日益复杂，CBFs在基于优化和基于学习的控制社区中受到越来越多的关注，作为安全过滤器，因为它们具有可证明的保证。然而，这些保证成功转移到实际系统与模型准确性密切相关。例如，有效载荷或风扰动可能会显著影响航空器的动力学并使安全保证失效。

### 目的

开发一种高效且灵活的在线学习增强型高阶自适应控制障碍函数，用于在复杂时变模型扰动下提高CBF认证系统的安全性。

### 方法

提出一种使用神经ODE的在线学习增强型高阶自适应控制障碍函数。这种方法能够实时提高CBF认证系统的安全性，即使在复杂的时变模型扰动下也能工作。作者将这种混合自适应CBF控制器部署在一个38克的纳米四旋翼上，使其能够以18公里/小时的风速保持与障碍物的安全距离。

### 主要发现

所提出的混合自适应CBF控制器能够在实时飞行中保持安全，即使在强风(18公里/小时)条件下也能维持与障碍物的安全距离。

### 结论

通过神经ODE实现的在线学习增强型高阶自适应控制障碍函数可以有效应对模型不确定性，提高系统在实际环境中的安全性和鲁棒性。

### 翻译

控制障碍函数(CBFs)是一种基于模型的有效工具，用于正式验证系统安全性。随着现代控制问题日益复杂，CBFs在基于优化和基于学习的控制社区中受到越来越多的关注，作为安全过滤器，因为它们具有可证明的保证。然而，这些保证成功转移到实际系统与模型准确性密切相关。例如，有效载荷或风扰动可能会显著影响航空器的动力学并使安全保证失效。在这项工作中，我们提出了一种使用神经ODE的高效且灵活的在线学习增强型高阶自适应控制障碍函数。我们的方法即使在复杂的时变模型扰动下也能实时提高CBF认证系统的安全性。特别是，我们在一个38克的纳米四旋翼上部署了混合自适应CBF控制器，使其能够以18公里/小时的风速保持与障碍物的安全距离。


### 论文摘要

Control barrier functions (CBFs) are an effective model-based tool to formally certify the safety of a system. With the growing complexity of modern control problems, CBFs have received increasing attention in both optimization-based and learning-based control communities as a safety filter, owing to their provable guarantees. However, success in transferring these guarantees to real-world systems is critically tied to model accuracy. For example, payloads or wind disturbances can significantly influence the dynamics of an aerial vehicle and invalidate the safety guarantee. In this work, we propose an efficient yet flexible online learning-enhanced high-order adaptive control barrier function using Neural ODEs. Our approach improves the safety of a CBF-certified system on the fly, even under complex time-varying model perturbations. In particular, we deploy our hybrid adaptive CBF controller on a 38g nano quadrotor, keeping a safe distance from the obstacle, against 18km/h wind.

---

## 200. 论文ID: 2511.19648v1

**错误:** 无法读取论文信息 - [Errno 2] No such file or directory: 'storage/result/2511.19648v1.json'

---

## 201. Total Factor Productivity and its determinants: an analysis of the relationship at firm level through unsupervised learning techniques

**论文链接:** [http://arxiv.org/abs/2511.19627v1](http://arxiv.org/abs/2511.19627v1)

**作者:** Paolo Pedotti

**发布时间:** 2025-11-24

**备注:** 56 pages with 47 figures

### GPT解析

### 总结

该研究使用无监督学习技术识别影响企业全要素生产率的关键特征，提供新的企业分类视角

### 背景

企业存在异质性，需要新的方法来理解和分类

### 目的

识别决定企业全要素生产率的关键特征

### 方法

使用无监督学习技术（主成分分析、自组织映射、聚类）进行自下而上的分析

### 主要发现

两个时期（2015-2019和2020）的主要生产率增长决定因素都与盈利能力、信贷/债务指标、成本和资本效率以及企业研发活动的努力和成果有关

### 结论

决定因素与生产率增长之间存在线性关系

### 翻译

该论文通过无监督学习技术（主成分分析、自组织映射、聚类）识别了决定企业全要素生产率的企业特征。这种自下而上的方法可以有效处理企业异质性问题，并提供看待企业标准分类的新视角。研究使用ORBIS数据库的大样本数据，涵盖了新冠疫情爆发前的时期（2015-2019）和疫情后的初期（2020年）。研究表明，在两个时期，生产率增长的主要决定因素都与企业的盈利能力、信贷/债务指标、成本和资本效率，以及企业研发活动的努力程度和成果有关。最后，研究发现决定因素与生产率增长之间存在线性关系。


### 论文摘要

The paper is related to the identification of firm's features which serve as determinants for firm's total factor productivity through unsupervised learning techniques (principal component analysis, self organizing maps, clustering). This bottom-up approach can effectively manage the problem of the heterogeneity of the firms and provides new ways to look at firms' standard classifications. Using the large sample provided by the ORBIS database, the analyses covers the years before the outbreak of Covid-19 (2015-2019) and the immediate post-Covid period (year 2020). It has been shown that in both periods, the main determinants of productivity growth are related to profitability, credit/debts measures, cost and capital efficiency, and effort and outcome of the R&D activity conducted by the firms. Finally, a linear relationship between determinants and productivity growth has been found.

---

## 202. 论文ID: 2511.19623v1

**错误:** 无法读取论文信息 - [Errno 2] No such file or directory: 'storage/result/2511.19623v1.json'

---

## 203. Leveraging LLMs for reward function design in reinforcement learning control tasks

**论文链接:** [http://arxiv.org/abs/2511.19355v1](http://arxiv.org/abs/2511.19355v1)

**作者:** Franklin Cardenoso, Wouter Caarls

**发布时间:** 2025-11-24

### GPT解析

### 总结

该论文提出了LEARN-Opt框架，一种基于大型语言模型的完全自主奖励函数优化方法，无需预先指标和环境源代码即可生成高质量奖励函数。

### 背景

在强化学习中设计有效奖励函数是重大瓶颈，需要大量人类专业知识。现有LLM方法通常需要预先评估指标、人工反馈或环境源代码作为上下文。

### 目的

开发一种无需预先指标和环境源代码作为上下文的完全自主奖励函数生成框架。

### 方法

引入LEARN-Opt（基于LLM的评估器和奖励函数优化器），这是一个基于LLM的、完全自主的、与模型无关的框架，能从系统描述和任务目标的文本描述中生成、执行和评估奖励函数候选。

### 主要发现

LEARN-Opt性能与最先进方法相当或更好，且需要更少前置知识；自动化奖励设计是高方差问题，需要多次运行；低成本LLM可找到与更大模型相当或更好的奖励函数。

### 结论

LEARN-Opt能在无预先人类定义指标情况下生成高质量奖励函数，减少工程开销并提高泛化能力。

### 翻译

在强化学习中设计有效奖励函数的挑战代表了重大瓶颈，通常需要大量人类专业知识且耗时。先前的工作和大型语言模型(LLMs)的最新进展展示了其自动化生成奖励函数的潜力。然而，现有方法通常需要预先评估指标、人工设计的反馈用于精炼过程，或使用环境源代码作为上下文。为解决这些局限，本文引入了LEARN-Opt（基于LLM的评估器和奖励函数优化器）。这个基于LLM的、完全自主的、与模型无关的框架消除了对预先指标和环境源代码作为上下文的需求，能从系统描述和任务目标的文本描述中生成、执行和评估奖励函数候选。LEARN-Opt的主要贡献在于其能够直接从系统描述和任务目标自主推导性能指标，实现无监督评估和选择奖励函数。我们的实验表明，LEARN-Opt实现了与最先进方法（如EUREKA）相当或更好的性能，同时需要更少的前置知识。我们发现自动化奖励设计是一个高方差问题，平均候选方案往往失败，需要多次运行才能找到最佳候选。最后，我们展示了LEARN-Opt可以释放低成本LLM的潜力，找到高性能的候选方案，这些方案与更大模型的候选方案相当甚至更好。这种展示的性能证实了其在无需任何预先人类定义指标的情况下生成高质量奖励函数的潜力，从而减少工程开销并提高泛化能力。


### 论文摘要

The challenge of designing effective reward functions in reinforcement learning (RL) represents a significant bottleneck, often requiring extensive human expertise and being time-consuming. Previous work and recent advancements in large language models (LLMs) have demonstrated their potential for automating the generation of reward functions. However, existing methodologies often require preliminary evaluation metrics, human-engineered feedback for the refinement process, or the use of environmental source code as context. To address these limitations, this paper introduces LEARN-Opt (LLM-based Evaluator and Analyzer for Reward functioN Optimization). This LLM-based, fully autonomous, and model-agnostic framework eliminates the need for preliminary metrics and environmental source code as context to generate, execute, and evaluate reward function candidates from textual descriptions of systems and task objectives. LEARN-Opt's main contribution lies in its ability to autonomously derive performance metrics directly from the system description and the task objective, enabling unsupervised evaluation and selection of reward functions. Our experiments indicate that LEARN-Opt achieves performance comparable to or better to that of state-of-the-art methods, such as EUREKA, while requiring less prior knowledge. We find that automated reward design is a high-variance problem, where the average-case candidate fails, requiring a multi-run approach to find the best candidates. Finally, we show that LEARN-Opt can unlock the potential of low-cost LLMs to find high-performing candidates that are comparable to, or even better than, those of larger models. This demonstrated performance affirms its potential to generate high-quality reward functions without requiring any preliminary human-defined metrics, thereby reducing engineering overhead and enhancing generalizability.

---

## 204. Scalable Parameter-Light Spectral Method for Clustering Short Text Embeddings with a Cohesion-Based Evaluation Metric

**论文链接:** [http://arxiv.org/abs/2511.19350v2](http://arxiv.org/abs/2511.19350v2)

**作者:** Nikita Neveditsin, Pawan Lingras, Vijay Mago

**发布时间:** 2025-11-24

### GPT解析

### 总结

该论文提出了一种可扩展的谱方法用于短文本聚类，能够自动估计聚类数量，并提出了Cohesion Ratio评估指标。实验表明，该方法指导下的传统聚类算法优于流行的参数轻量方法。

### 背景

短文本嵌入聚类是自然语言处理的基础任务，但挑战在于需要提前指定聚类数量。

### 目的

开发一种能够直接从拉普拉斯特征谱结构中估计聚类数量的可扩展谱方法，并设计一种无需真实标签的聚类质量评估指标。

### 方法

基于余弦相似度构建拉普拉斯特征谱，采用自适应采样策略指导聚类数量估计；提出Cohesion Ratio指标，量化类内相似度超过全局相似度背景的程度，具有信息论动机。

### 主要发现

在六个短文本数据集和四种现代嵌入模型上的实验表明，使用作者估计器指导的K-Means和HAC算法显著优于HDBSCAN、OPTICS和Leiden等参数轻量方法；Cohesion Ratio与归一化互信息和同质性等外在指标高度相关。

### 结论

所提出的谱估计器和Cohesion Ratio对短文本数据的无监督组织和评估具有实际价值。

### 翻译

短文本嵌入聚类是自然语言处理的基础任务，但由于需要提前指定聚类数量而仍然具有挑战性。我们引入了一种可扩展的谱方法，该方法直接从拉普拉斯特征谱的结构中估计聚类数量，该特征谱使用余弦相似度构建，并由自适应采样策略指导。这种采样方法使我们的估计器能够高效扩展到大型数据集而不牺牲可靠性。为了在没有真实标签的情况下支持聚类质量的内在评估，我们提出了内聚比，这是一个简单且可解释的评估指标，它量化了类内相似度超过全局相似度背景的程度。它受到互信息启发的信息论动机，在我们的实验中，它与归一化互信息和同质性等外在指标密切相关。在六个短文本数据集和四种现代嵌入模型上的大量实验表明，当由我们的估计器指导时，K-Means和HAC等标准算法显著优于HDBSCAN、OPTICS和Leiden等流行的参数轻量方法。这些结果表明我们的谱估计器和内聚比对于短文本数据的无监督组织和评估具有实际价值。我们的k估计器和内聚比的实现，以及用于复现实验的代码，可在https://anonymous.4open.science/r/towards_clustering-0C2E获取。


### 论文摘要

Clustering short text embeddings is a foundational task in natural language processing, yet remains challenging due to the need to specify the number of clusters in advance. We introduce a scalable spectral method that estimates the number of clusters directly from the structure of the Laplacian eigenspectrum, constructed using cosine similarities and guided by an adaptive sampling strategy. This sampling approach enables our estimator to efficiently scale to large datasets without sacrificing reliability. To support intrinsic evaluation of cluster quality without ground-truth labels, we propose the Cohesion Ratio, a simple and interpretable evaluation metric that quantifies how much intra-cluster similarity exceeds the global similarity background. It has an information-theoretic motivation inspired by mutual information, and in our experiments it correlates closely with extrinsic measures such as normalized mutual information and homogeneity. Extensive experiments on six short-text datasets and four modern embedding models show that standard algorithms like K-Means and HAC, when guided by our estimator, significantly outperform popular parameter-light methods such as HDBSCAN, OPTICS, and Leiden. These results demonstrate the practical value of our spectral estimator and Cohesion Ratio for unsupervised organization and evaluation of short text data. Implementation of our estimator of k and Cohesion Ratio, along with code for reproducing the experiments, is available at https://anonymous.4open.science/r/towards_clustering-0C2E.

---

## 205. MultiBanAbs: A Comprehensive Multi-Domain Bangla Abstractive Text Summarization Dataset

**论文链接:** [http://arxiv.org/abs/2511.19317v1](http://arxiv.org/abs/2511.19317v1)

**作者:** Md. Tanzim Ferdous, Naeem Ahsan Chowdhury, Prithwiraj Bhattacharjee

**发布时间:** 2025-11-24

### GPT解析

### 总结

研究开发了一个新的孟加拉语抽象摘要数据集，包含来自不同来源的54,000多篇文章和摘要，跨越多个领域和写作风格，为孟加拉语自然语言处理研究建立了基准。

### 背景

现有研究主要集中在新闻文章上，而新闻文章通常遵循固定的写作风格。在当今数字时代，大量孟加拉语内容不断产生于博客、报纸和社交媒体，需要能够减少信息过载并帮助读者更快理解内容的摘要系统。

### 目的

开发一个能够适应不同领域和写作风格的孟加拉语摘要数据集，以解决现有方法在有限上下文中有效但无法适应现实世界中孟加拉语文本多样性的问题。

### 方法

收集了超过54,000篇来自多个来源（如Cinegolpo博客、Samakal和The Business Standard报纸）的孟加拉语文章和摘要，建立了跨领域和多写作风格的数据集。使用LSTM、BanglaT5-small和MTS-small等深度学习和迁移学习模型训练和评估该数据集。

### 主要发现

该数据集展示了作为孟加拉语自然语言处理未来研究基准的潜力，为构建强大的摘要系统提供了坚实基础，并有助于扩展低资源语言的NLP资源。

### 结论

该数据集比单一领域资源具有更强的适应性和实际相关性，为孟加拉语摘要系统的发展提供了重要资源。

### 翻译

本研究开发了一个新的孟加拉语抽象摘要数据集，用于生成来自不同来源的孟加拉语文章的简洁摘要。该领域的大多数现有研究都集中在新闻文章上，而记者通常遵循固定的写作风格。虽然这种方法在有限上下文中有效，但它们往往无法适应现实世界中孟加拉语文本的多样性。在当今数字时代，大量孟加拉语内容不断产生于博客、报纸和社交媒体。这创造了对能够减少信息过载并帮助读者更快理解内容的摘要系统的迫切需求。为了应对这一挑战，我们开发了一个包含从多个来源收集的54,000多篇孟加拉语文章和摘要的数据集，包括Cinegolpo等博客以及Samakal和The Business Standard等报纸。与单一领域资源不同，我们的数据集跨越多个领域和写作风格。它提供了更强的适应性和实际相关性。为了建立强大的基线，我们使用几种深度学习和迁移学习模型（包括LSTM、BanglaT5-small和MTS-small）训练和评估了该数据集。结果突显了其作为孟加拉语自然语言处理未来研究基准的潜力。该数据集为构建强大的摘要系统提供了坚实基础，并有助于扩展低资源语言的NLP资源。


### 论文摘要

This study developed a new Bangla abstractive summarization dataset to generate concise summaries of Bangla articles from diverse sources. Most existing studies in this field have concentrated on news articles, where journalists usually follow a fixed writing style. While such approaches are effective in limited contexts, they often fail to adapt to the varied nature of real-world Bangla texts. In today's digital era, a massive amount of Bangla content is continuously produced across blogs, newspapers, and social media. This creates a pressing need for summarization systems that can reduce information overload and help readers understand content more quickly. To address this challenge, we developed a dataset of over 54,000 Bangla articles and summaries collected from multiple sources, including blogs such as Cinegolpo and newspapers such as Samakal and The Business Standard. Unlike single-domain resources, our dataset spans multiple domains and writing styles. It offers greater adaptability and practical relevance. To establish strong baselines, we trained and evaluated this dataset using several deep learning and transfer learning models, including LSTM, BanglaT5-small, and MTS-small. The results highlight its potential as a benchmark for future research in Bangla natural language processing. This dataset provides a solid foundation for building robust summarization systems and helps expand NLP resources for low-resource languages.

---

## 206. On Altruism and Spite in Bimatrix Games

**论文链接:** [http://arxiv.org/abs/2511.19307v1](http://arxiv.org/abs/2511.19307v1)

**作者:** Michail Fasoulakis, Leonidas Bakopoulos, Charilaos Akasiadis, Georgios Chalkiadakis

**发布时间:** 2025-11-24

### GPT解析

### 总结

本文研究了放宽自利假设后，利他或恶意行为对双矩阵博弈算法方面的影响，包括纳什均衡的复杂性和质量，并提供了理论分析和实验验证。

### 背景

博弈论通常假设玩家只优化自身效用函数，但现实中玩家可能表现出利他或恶意行为。尽管经济学文献有多种解释玩家不完全自私的理论，但这些研究大多未关注利他或恶意行为的算法含义。

### 目的

放松'自利'假设，研究利他或恶意情况下双矩阵博弈的算法方面，包括(近似)纳什均衡的复杂性和质量。

### 方法

提供对这些主题的理论和实验处理，展示学习对手利他/恶意行为程度的可能性，并利用这种能力进行对手选择和知识转移。

### 主要发现

研究了利他或恶意情况下双矩阵博弈的算法特性，提供了理论框架和实验验证，展示了学习对手行为模式并应用于实际博弈的潜力。

### 结论

利他或恶意行为显著影响双矩阵博弈的算法特性，理解并利用这些行为模式可以提高博弈效果，为实际应用提供了新的视角和方法。

### 翻译

博弈论中的一个常见假设是任何玩家都只优化考虑自身收益的效用函数。然而，长期以来人们观察到，在现实生活中玩家可能会采取利他甚至恶意的行为。因此，经济学文献中有许多尝试解释玩家不完全自私的事实，但这些工作大多不关注利他或恶意行为在博弈中的算法含义。在本文中，我们放宽了上述'自利'假设，并开始研究利他或恶意情况下双矩阵博弈的算法方面——例如它们的(近似)纳什均衡的复杂性和质量。我们为这些主题提供了理论和实验处理。此外，我们展示了学习对手利他/恶意行为程度的可能性，并利用这种能力在双矩阵博弈中进行对手选择和知识转移。


### 论文摘要

One common assumption in game theory is that any player optimizes a utility function that takes into account only its own payoff. However, it has long been observed that in real life players may adopt an altruistic or even spiteful behaviour. As such, there are numerous attempts in the economics literature that strive to explain the fact that players are not entirely selfish, but most of these works do not focus on the algorithmic implications of altruism or spite in games. In this paper, we relax the aforementioned ``self-interest'' assumption, and initiate the study of algorithmic aspects of bimatrix games -- such as the complexity and the quality of their (approximate) Nash equilibria -- under altruism or spite. We provide both a theoretical and an experimental treatment of these topics. Moreover, we demonstrate the potential for learning the degree of an opponent's altruistic/spiteful behaviour, and employing this for opponent selection and transfer of knowledge in bimatrix games.

---

## 207. How to Purchase Labels? A Cost-Effective Approach Using Active Learning Markets

**论文链接:** [http://arxiv.org/abs/2511.20605v1](http://arxiv.org/abs/2511.20605v1)

**作者:** Xiwen Huang, Pierre Pinson

**发布时间:** 2025-11-25

**备注:** Submitted as a preprint. 34 pages, 14 figures, 4 tables

### GPT解析

### 总结

论文介绍了主动学习市场作为购买标签的方式，用于改进模型拟合和预测分析。通过将市场清算形式化为优化问题，整合预算约束和改进阈值，提出两种主动学习策略与不同定价机制相结合，并在房地产定价和能源预测领域验证了其有效性。

### 背景

当前存在许多购买特征和样本的提案，但在需要购买标签以改进模型性能的情况下缺乏相应解决方案。

### 目的

提出主动学习市场作为获取标签的优化方法，在预算约束条件下提高预测模型的性能。

### 方法

将市场清算形式化为优化问题；采用单一买家-多卖家设置；提出基于方差和基于委员会查询的两种主动学习策略，配合不同定价机制；与随机采样基准方法比较；在真实数据集上验证。

### 主要发现

所提出的策略具有鲁棒性，能够以更少的标签获取实现比传统方法更优的性能。

### 结论

主动学习市场为资源受限环境下的数据获取优化提供了易于实施的实用解决方案。

### 翻译

我们介绍并分析了主动学习市场作为一种购买标签的方式，适用于分析师旨在获取额外数据以改进模型拟合，或为预测分析应用更好地训练模型的情况。这与许多已经存在的购买特征和样本的提案形成对比。通过将市场清算形式化为一个优化问题，我们将预算约束和改进阈值整合到标签获取过程中。我们专注于单一买家-多卖家设置，并提出使用两种主动学习策略（基于方差和基于委员会查询），与不同的定价机制配对。它们与基准随机采样方法进行比较。所提出的策略在两个关键应用领域的真实数据集上得到验证：房地产定价和能源预测。结果表明我们方法的鲁棒性，与传统方法相比，以获取更少的标签持续实现更优的性能。我们的提案包含一个易于实施的实用解决方案，用于优化资源受限环境中的数据获取。


### 论文摘要

We introduce and analyse active learning markets as a way to purchase labels, in situations where analysts aim to acquire additional data to improve model fitting, or to better train models for predictive analytics applications. This comes in contrast to the many proposals that already exist to purchase features and examples. By originally formalising the market clearing as an optimisation problem, we integrate budget constraints and improvement thresholds into the label acquisition process. We focus on a single-buyer-multiple-seller setup and propose the use of two active learning strategies (variance based and query-by-committee based), paired with distinct pricing mechanisms. They are compared to a benchmark random sampling approach. The proposed strategies are validated on real-world datasets from two critical application domains: real estate pricing and energy forecasting. Results demonstrate the robustness of our approach, consistently achieving superior performance with fewer labels acquired compared to conventional methods. Our proposal comprises an easy-to-implement practical solution for optimising data acquisition in resource-constrained environments.

---

## 208. From Passive Perception to Active Memory: A Weakly Supervised Image Manipulation Localization Framework Driven by Coarse-Grained Annotations

**论文链接:** [http://arxiv.org/abs/2511.20359v1](http://arxiv.org/abs/2511.20359v1)

**作者:** Zhiqing Guo, Dongdong Xi, Songlin Li, Gaobo Yang

**发布时间:** 2025-11-25

**备注:** Accepted by AAAI 2026

### GPT解析

### 总结

BoxPromptIML是一种新型弱监督图像篡改定位框架，通过粗略区域标注策略、高效轻量级学生模型和受人类记忆机制启发的特征融合模块，有效平衡了标注成本与定位性能，在实验中表现出优于或匹敌完全监督模型的效果。

### 背景

图像篡改定位面临最小化标注成本与实现细粒度定位精度之间的基本权衡。现有完全监督方法依赖密集像素级掩码标注，限制了可扩展性；而弱监督方法基于图像级标签，减少标注工作量但缺乏精确空间定位能力。

### 目的

开发一种能够有效平衡标注成本和定位性能的图像篡改定位框架，解决现有方法在标注成本和定位精度之间的权衡问题。

### 方法

1) 提出粗略区域标注策略，以较低成本生成相对准确的篡改掩码；2) 设计基于知识蒸馏的高效轻量级学生模型，从基于Segment Anything Model的固定教师模型学习细粒度定位；3) 开发受人类潜意识记忆机制启发的特征融合模块，采用双重引导策略主动将回忆的原型模式与实时观察线索上下文化。

### 主要发现

在分布内和分布外数据集上的广泛实验表明，BoxPromptIML能够优于或匹敌完全监督模型，同时保持强大的泛化能力、低标注成本和高效的部署特性。动态知识回忆过程显著提高了定位准确性和鲁棒性。

### 结论

BoxPromptIML成功解决了图像篡改定位中标注成本与定位精度的权衡问题，通过创新的标注策略、模型架构和特征融合机制，实现了高性能、低成本和高效率的图像篡改定位系统。

### 翻译

图像篡改定位面临在最小化标注成本和实现细粒度定位精度之间的基本权衡。现有完全监督IML方法严重依赖密集像素级掩码标注，限制了其在大数据集或实际部署中的可扩展性。相比之下，大多数现有弱监督IML方法基于图像级标签，大大减少了标注工作量但通常缺乏精确的空间定位。为解决这一困境，我们提出了BoxPromptIML，一种新型弱监督IML框架，有效平衡了标注成本和定位性能。具体而言，我们提出了一种粗略区域标注策略，可以以较低成本生成相对准确的篡改掩码。为提高模型效率并促进部署，我们进一步设计了一个高效轻量级学生模型，该模型通过基于Segment Anything Model的固定教师模型进行知识蒸馏，学习执行细粒度定位。此外，受人类潜意识记忆机制的启发，我们的特征融合模块采用双重引导策略，主动将回忆的原型模式与从输入派生的实时观察线索上下文化。这种策略不是被动特征提取，而是使知识回忆成为一个动态过程，其中长期记忆适应当前图像的特定上下文，显著提高了定位准确性和鲁棒性。在分布内和分布外数据集上的广泛实验表明，BoxPromptIML优于或匹敌完全监督模型，同时保持强大的泛化能力、低标注成本和高效的部署特性。


### 论文摘要

Image manipulation localization (IML) faces a fundamental trade-off between minimizing annotation cost and achieving fine-grained localization accuracy. Existing fully-supervised IML methods depend heavily on dense pixel-level mask annotations, which limits scalability to large datasets or real-world deployment.In contrast, the majority of existing weakly-supervised IML approaches are based on image-level labels, which greatly reduce annotation effort but typically lack precise spatial localization. To address this dilemma, we propose BoxPromptIML, a novel weakly-supervised IML framework that effectively balances annotation cost and localization performance. Specifically, we propose a coarse region annotation strategy, which can generate relatively accurate manipulation masks at lower cost. To improve model efficiency and facilitate deployment, we further design an efficient lightweight student model, which learns to perform fine-grained localization through knowledge distillation from a fixed teacher model based on the Segment Anything Model (SAM). Moreover, inspired by the human subconscious memory mechanism, our feature fusion module employs a dual-guidance strategy that actively contextualizes recalled prototypical patterns with real-time observational cues derived from the input. Instead of passive feature extraction, this strategy enables a dynamic process of knowledge recollection, where long-term memory is adapted to the specific context of the current image, significantly enhancing localization accuracy and robustness. Extensive experiments across both in-distribution and out-of-distribution datasets show that BoxPromptIML outperforms or rivals fully-supervised models, while maintaining strong generalization, low annotation cost, and efficient deployment characteristics.

---

## 209. APT-CGLP: Advanced Persistent Threat Hunting via Contrastive Graph-Language Pre-Training

**论文链接:** [http://arxiv.org/abs/2511.20290v1](http://arxiv.org/abs/2511.20290v1)

**作者:** Xuebo Qiu, Mingqi Lv, Yimei Zhang, Tieming Chen, Tiantian Zhu, Qijie Song, Shouling Ji

**发布时间:** 2025-11-25

**备注:** Accepted by SIGKDD 2026 Research Track

### GPT解析

### 总结

这篇论文提出了APT-CGLP，一种基于对比图-语言预训练的新型跨模态高级持续威胁狩猎系统，实现了无需人工干预的端到端语义匹配，在四个真实APT数据集上表现出色。

### 背景

基于来源的威胁狩猎通过关联网络威胁情报描述的攻击模式与系统审计日志生成的来源图来识别终端上的高级持续威胁。该范式面临模态差距问题——来源图与网络威胁情报报告之间的结构和语义脱节。先前方法将威胁狩猎视为图匹配任务，但会造成严重信息损失且需要密集人工管理，影响可扩展性和有效性。

### 目的

开发一种能够实现来源图与网络威胁情报报告之间端到端语义匹配的APT狩猎系统，无需人工干预，提高准确性和效率。

### 方法

APT-CGLP系统利用大型语言模型缓解数据稀缺问题，合成高质量的来源图-网络威胁情报报告对，同时从噪声的网络来源情报中提取可操作的见解。系统采用定制的多目标训练算法，结合对比学习和跨模态掩码建模，促进粗粒度和细粒度级别的跨模态攻击语义对齐。

### 主要发现

在四个真实APT数据集上进行的大量实验表明，APT-CGLP在准确性和效率方面持续优于最先进的威胁狩猎基线方法。

### 结论

APT-CGLP通过对比图-语言预训练技术，成功解决了威胁狩猎中的模态差距问题，实现了无需人工干预的端到端语义匹配，显著提高了威胁狩猎的准确性和效率。

### 翻译

基于来源的威胁狩猎通过关联网络威胁情报描述的攻击模式与系统审计日志生成的来源图来识别终端上的高级持续威胁。该范式面临的一个基本挑战是模态差距——来源图与网络威胁情报报告之间的结构和语义脱节。先前的工作将威胁狩猎视为图匹配任务：1)从网络威胁情报报告中提取攻击图，2)将它们与来源图对齐。然而，这一管道在图提取过程中会造成严重的信息损失，并且需要密集的人工管理，影响了可扩展性和有效性。在本文中，我们提出了APT-CGLP，一种通过对比图-语言预训练的新型跨模态APT狩猎系统，实现了无需人工干预的来源图与网络威胁情报报告之间的端到端语义匹配。首先，借助大型语言模型，APT-CGLP通过合成高质量的来源图-网络威胁情报报告对来缓解数据稀缺问题，同时从噪声的网络来源情报中提取可操作的见解，以提高其操作效用。其次，APT-CGLP采用定制的多目标训练算法，将对比学习与跨模态掩码建模相结合，促进粗粒度和细粒度级别的跨模态攻击语义对齐。在四个真实APT数据集上的大量实验表明，APT-CGLP在准确性和效率方面持续优于最先进的威胁狩猎基线方法。


### 论文摘要

Provenance-based threat hunting identifies Advanced Persistent Threats (APTs) on endpoints by correlating attack patterns described in Cyber Threat Intelligence (CTI) with provenance graphs derived from system audit logs. A fundamental challenge in this paradigm lies in the modality gap--the structural and semantic disconnect between provenance graphs and CTI reports. Prior work addresses this by framing threat hunting as a graph matching task: 1) extracting attack graphs from CTI reports, and 2) aligning them with provenance graphs. However, this pipeline incurs severe \textit{information loss} during graph extraction and demands intensive manual curation, undermining scalability and effectiveness.   In this paper, we present APT-CGLP, a novel cross-modal APT hunting system via Contrastive Graph-Language Pre-training, facilitating end-to-end semantic matching between provenance graphs and CTI reports without human intervention. First, empowered by the Large Language Model (LLM), APT-CGLP mitigates data scarcity by synthesizing high-fidelity provenance graph-CTI report pairs, while simultaneously distilling actionable insights from noisy web-sourced CTIs to improve their operational utility. Second, APT-CGLP incorporates a tailored multi-objective training algorithm that synergizes contrastive learning with inter-modal masked modeling, promoting cross-modal attack semantic alignment at both coarse- and fine-grained levels. Extensive experiments on four real-world APT datasets demonstrate that APT-CGLP consistently outperforms state-of-the-art threat hunting baselines in terms of accuracy and efficiency.

---

## 210. Patch-Level Glioblastoma Subregion Classification with a Contrastive Learning-Based Encoder

**论文链接:** [http://arxiv.org/abs/2511.20221v1](http://arxiv.org/abs/2511.20221v1)

**作者:** Juexin Zhang, Qifeng Zhong, Ying Weng, Ke Chen

**发布时间:** 2025-11-25

**备注:** Accepted by the International Brain Tumor Segmentation (BraTS) challenge organized at MICCAI 2025 conference

### GPT解析

### 总结

本研究开发了一种基于Vision Transformer的深度学习方法，用于胶质母细胞瘤的组织病理学图像分析，在BraTS-Path 2025挑战赛中获得第二名。

### 背景

胶质母细胞瘤是一种侵袭性脑肿瘤，具有显著的分子和病理异质性，这给诊断和患者分层带来了困难。

### 目的

利用深度学习实现全切片图像的客观和自动化分析。

### 方法

为BraTS-Path 2025挑战开发了一种方法，该方法使用官方训练数据集对预训练的Vision Transformer编码器进行微调，并配备专门的分类头。

### 主要发现

在在线验证集上，模型通过Synapse平台评估，马修斯相关系数为0.7064，F1得分为0.7676；在最终测试集上，模型达到马修斯相关系数为0.6509，F1得分为0.5330，使研究团队在挑战赛中获得第二名。

### 结论

研究结果为基于ViT的组织病理学分析奠定了坚实的基础，未来的工作将重点缩小在未见过的验证数据上观察到的性能差距。

### 翻译

胶质母细胞瘤这种侵袭性脑肿瘤显著的分子和病理异质性，给诊断和患者分层带来了困难。虽然传统的组织病理学评估仍然是标准方法，但深度学习为全切片图像的客观和自动化分析提供了有前景的途径。为了BraTS-Path 2025挑战赛，我们开发了一种方法，使用官方训练数据集对预训练的Vision Transformer编码器进行微调，并配备专门的分类头。我们的模型在通过Synapse平台评估的在线验证集上的表现，马修斯相关系数为0.7064，F1得分为0.7676。在最终测试集上，模型达到了马修斯相关系数0.6509和F1得分0.5330，这使我们的团队在BraTS-Pathology 2025挑战赛中获得了第二名。我们的结果为基于ViT的组织病理学分析奠定了坚实的基础，未来的努力将重点缩小在未见过的验证数据上观察到的性能差距。


### 论文摘要

The significant molecular and pathological heterogeneity of glioblastoma, an aggressive brain tumor, complicates diagnosis and patient stratification. While traditional histopathological assessment remains the standard, deep learning offers a promising path toward objective and automated analysis of whole slide images. For the BraTS-Path 2025 Challenge, we developed a method that fine-tunes a pre-trained Vision Transformer (ViT) encoder with a dedicated classification head on the official training dataset. Our model's performance on the online validation set, evaluated via the Synapse platform, yielded a Matthews Correlation Coefficient (MCC) of 0.7064 and an F1-score of 0.7676. On the final test set, the model achieved an MCC of 0.6509 and an F1-score of 0.5330, which secured our team second place in the BraTS-Pathology 2025 Challenge. Our results establish a solid baseline for ViT-based histopathological analysis, and future efforts will focus on bridging the performance gap observed on the unseen validation data.

---

## 211. AdaCap: An Adaptive Contrastive Approach for Small-Data Neural Networks

**论文链接:** [http://arxiv.org/abs/2511.20170v1](http://arxiv.org/abs/2511.20170v1)

**作者:** Bruno Belucci, Karim Lounici, Katia Meziani

**发布时间:** 2025-11-25

**备注:** Submitted to ESANN 2026

### GPT解析

### 总结

论文提出了AdaCap方法，通过结合基于排列的对比损失和基于Tikhonov的闭式输出映射，显著提升了神经网络在小数据集上的表现。

### 背景

神经网络在小数据集上表现不佳，而基于树的模型仍然占主导地位。

### 目的

开发一种训练方案来改善神经网络在小样本情况下的性能。

### 方法

AdaCap是一种训练方案，结合了基于排列的对比损失和基于Tikhonov的闭式输出映射。

### 主要发现

在85个真实世界的回归数据集和多种架构上，AdaCap在小样本情况下取得了持续且统计上显著的改进，特别是对于残差模型。基于数据集特征训练的元预测器能够准确预测AdaCap何时有益。

### 结论

AdaCap作为一种有针对性的正则化机制，强化了神经网络最脆弱的部分。

### 翻译

神经网络在小数据集上表现不佳，而基于树的模型仍然占主导地位。我们提出了自适应对比方法(AdaCap)，这是一种结合了基于排列的对比损失和基于Tikhonov的闭式输出映射的训练方案。在85个真实世界回归数据集和多种架构上，AdaCap在小样本情况下取得了持续且统计上显著的改进，特别是对于残差模型。一个基于数据集特征（大小、偏度、噪声）训练的元预测器能够准确预测AdaCap何时有益。这些结果表明，AdaCap作为一种有针对性的正则化机制，强化了神经网络最脆弱的部分。所有结果和代码已在https://github.com/BrunoBelucci/adacap公开。


### 论文摘要

Neural networks struggle on small tabular datasets, where tree-based models remain dominant. We introduce Adaptive Contrastive Approach (AdaCap), a training scheme that combines a permutation-based contrastive loss with a Tikhonov-based closed-form output mapping. Across 85 real-world regression datasets and multiple architectures, AdaCap yields consistent and statistically significant improvements in the small-sample regime, particularly for residual models. A meta-predictor trained on dataset characteristics (size, skewness, noise) accurately anticipates when AdaCap is beneficial. These results show that AdaCap acts as a targeted regularization mechanism, strengthening neural networks precisely where they are most fragile. All results and code are publicly available at https://github.com/BrunoBelucci/adacap.

---

## 212. FINE: Factorized multimodal sentiment analysis via mutual INformation Estimation

**论文链接:** [http://arxiv.org/abs/2511.20167v1](http://arxiv.org/abs/2511.20167v1)

**作者:** Yadong Liu, Shangfei Wang

**发布时间:** 2025-11-25

**备注:** 15 pages, 9 figures, conference

### GPT解析

### 总结

该论文提出了一种分解式多模态融合框架来解决多模态情感分析中的异质性问题，通过将模态分解为共享和独特表示并抑制噪声，同时引入两个辅助模块提升特征提取和长期时间建模能力，实验证明该方法在多个数据集上优于现有方法。

### 背景

多模态情感分析由于模态间固有的异质性而具有挑战性，这种异质性表现为异步信号、模态间信息不平衡以及任务无关噪声的干扰，阻碍了稳健准确的情感表示学习。

### 目的

解决多模态情感分析中的异质性问题，包括异步信号、模态间信息不平衡和任务无关噪声干扰，以学习更稳健和准确的情感表示。

### 方法

提出分解式多模态融合框架，将每种模态分解为共享和独特表示并抑制噪声；采用基于互信息的优化策略指导分解过程；引入Mixture of Q-Formers辅助提取细粒度情感特征；使用Dynamic Contrastive Queue进行对比学习以捕获长距离判别模式。

### 主要发现

在多个公共数据集上的大量实验表明，该方法一致优于现有方法，验证了所提出框架的有效性和稳健性。

### 结论

所提出的分解式多模态融合框架能够有效处理多模态情感分析中的异质性问题，通过细粒度分解、噪声抑制和辅助模块的引入，提高了情感表示的质量，并在多个数据集上取得了优于现有方法的性能。

### 翻译

多模态情感分析由于模态间固有的异质性而仍然是一个具有挑战性的任务。这种异质性通常表现为异步信号、模态间信息不平衡以及来自任务无关噪声的干扰，阻碍了稳健准确的情感表示的学习。为了解决这些问题，我们提出了一种分解式多模态融合框架，首先将每种模态分解为共享和独特表示，然后在两者中抑制任务无关噪声，仅保留情感关键表示。这种细粒度分解通过减少冗余、促进跨模态互补性和隔离任务相关情感提示来提高表示质量。我们不直接操作特征空间，而是采用基于互信息的优化策略，以更稳定和有原则的方式指导分解过程。为了进一步支持特征提取和长期时间建模，我们引入了两个辅助模块：一个放置在分解前的Q-Formers混合体，使用可学习查询从多模态中提取细粒度情感特征；另一个放置在分解后的动态对比队列，存储最新高级表示用于对比学习，使模型能够捕获长距离判别模式并提高类级可分性。在多个公共数据集上的大量实验表明，我们的方法一致优于现有方法，验证了所提出框架的有效性和稳健性。


### 论文摘要

Multimodal sentiment analysis remains a challenging task due to the inherent heterogeneity across modalities. Such heterogeneity often manifests as asynchronous signals, imbalanced information between modalities, and interference from task-irrelevant noise, hindering the learning of robust and accurate sentiment representations. To address these issues, we propose a factorized multimodal fusion framework that first disentangles each modality into shared and unique representations, and then suppresses task-irrelevant noise within both to retain only sentiment-critical representations. This fine-grained decomposition improves representation quality by reducing redundancy, prompting cross-modal complementarity, and isolating task-relevant sentiment cues. Rather than manipulating the feature space directly, we adopt a mutual information-based optimization strategy to guide the factorization process in a more stable and principled manner. To further support feature extraction and long-term temporal modeling, we introduce two auxiliary modules: a Mixture of Q-Formers, placed before factorization, which precedes the factorization and uses learnable queries to extract fine-grained affective features from multiple modalities, and a Dynamic Contrastive Queue, placed after factorization, which stores latest high-level representations for contrastive learning, enabling the model to capture long-range discriminative patterns and improve class-level separability. Extensive experiments on multiple public datasets demonstrate that our method consistently outperforms existing approaches, validating the effectiveness and robustness of the proposed framework.

---

## 213. EM2LDL: A Multilingual Speech Corpus for Mixed Emotion Recognition through Label Distribution Learning

**论文链接:** [http://arxiv.org/abs/2511.20106v1](http://arxiv.org/abs/2511.20106v1)

**作者:** Xingfeng Li, Xiaohan Shi, Junjie Li, Yongwei Li, Masashi Unoki, Tomoki Toda, Masato Akagi

**发布时间:** 2025-11-25

**备注:** Submitted to IEEE Transactions on Affective computing

### GPT解析

### 总结

本研究介绍了EM2LDL，一个多语言语音语料库，旨在通过标签分布学习推进混合情感识别，解决了现有语料库在语言多样性和情感建模方面的局限性。

### 背景

现有情感语料库主要是单语言的、单标签的，限制了语言多样性，无法建模混合情感，且缺乏生态有效性。

### 目的

开发一个包含多种语言(英语、普通话、粤语)的语音语料库，捕捉多语言地区的语码转换现象，并使用细粒度情感分布标注，为多语言环境中的情感计算研究提供支持。

### 方法

构建EM2LDL语料库，包含来自在线平台的自然情感表达，使用32个类别的细粒度情感分布进行标注，并使用自监督学习模型(如HuBERT-large-EN)建立基线。

### 主要发现

实验基线在说话人独立的基于性别、年龄和性格的评估中表现出强大的性能，HuBERT-large-EN模型取得了最佳结果。

### 结论

通过融入语言多样性和生态有效性，EM2LDL为探索多语言环境中的复杂情感动态提供了多功能测试平台，有助于开发适应性强、有同理心的情感计算系统。

### 翻译

本研究介绍了EM2LDL，一种新型多语言语音语料库，旨在通过标签分布学习推进混合情感识别。针对现有语料库主要是单语言、单标签情感语料库的限制，这些语料库限制了语言多样性，无法建模混合情感，且缺乏生态有效性，EM2LDL包含英语、普通话和粤语的表达性话语，捕捉了香港和澳门等多语言地区普遍存在的语句内语码转换。该语料库整合了来自在线平台的自然情感表达，使用32个类别的细粒度情感分布进行标注。使用自监督学习模型的实验基线在说话人独立的基于性别、年龄和性格的评估中表现出强大的性能，其中HuBERT-large-EN取得了最佳结果。通过融入语言多样性和生态有效性，EM2LDL使得探索多语言环境中的复杂情感动态成为可能。这项工作为开发适应性强、有同理心的系统提供了多功能的测试平台，应用于情感计算领域，包括心理健康监测和跨文化交流。数据集、标注和基线代码已在https://github.com/xingfengli/EM2LDL公开可用。


### 论文摘要

This study introduces EM2LDL, a novel multilingual speech corpus designed to advance mixed emotion recognition through label distribution learning. Addressing the limitations of predominantly monolingual and single-label emotion corpora \textcolor{black}{that restrict linguistic diversity, are unable to model mixed emotions, and lack ecological validity}, EM2LDL comprises expressive utterances in English, Mandarin, and Cantonese, capturing the intra-utterance code-switching prevalent in multilingual regions like Hong Kong and Macao. The corpus integrates spontaneous emotional expressions from online platforms, annotated with fine-grained emotion distributions across 32 categories. Experimental baselines using self-supervised learning models demonstrate robust performance in speaker-independent gender-, age-, and personality-based evaluations, with HuBERT-large-EN achieving optimal results. By incorporating linguistic diversity and ecological validity, EM2LDL enables the exploration of complex emotional dynamics in multilingual settings. This work provides a versatile testbed for developing adaptive, empathetic systems for applications in affective computing, including mental health monitoring and cross-cultural communication. The dataset, annotations, and baseline codes are publicly available at https://github.com/xingfengli/EM2LDL.

---

## 214. Cross-Contrastive Clustering for Multimodal Attributed Graphs with Dual Graph Filtering

**论文链接:** [http://arxiv.org/abs/2511.20030v1](http://arxiv.org/abs/2511.20030v1)

**作者:** Haoran Zheng, Renchi Yang, Hongtao Wang, Jianliang Xu

**发布时间:** 2025-11-25

**备注:** Accepted by SIGKDD 2026. The code is available at https://github.com/HaoranZ99/DGF

### GPT解析

### 总结

本研究提出了一种双重图滤波(DGF)方案，用于解决多模态属性图(MMAGs)中的聚类问题，通过特征级去噪和三重交叉对比训练策略，显著提高了聚类性能。

### 背景

多模态属性图(MMAGs)是一种表达复杂数据模型，用于表示具有多模态属性(文本、图像等)的实体之间的复杂连接。在这类数据上进行聚类在社交社区检测、医疗数据分析等现实场景中有许多应用。

### 目的

解决现有多视图聚类解决方案在处理多模态属性图时面临的挑战，特别是处理由大型预训练语言和视觉模型输出的多模态属性的独特特征(如低模态间相关性和强烈的特征级噪声)。

### 方法

提出双重图滤波(DGF)方案，创新地将特征级去噪组件纳入节点表示学习中，并包括一个三重交叉对比训练策略，利用跨模态、邻域和社区的实例级对比学习来学习鲁棒性和判别性的节点表示。

### 主要发现

在八个基准MMAG数据集上的综合实验表明，DGF能够在聚类质量上持续且显著地优于各种最先进的基线方法，有效克服了传统图滤波器的局限性。

### 结论

双重图滤波(DGF)方案通过结合特征级去噪和三重交叉对比训练策略，能够有效处理多模态属性图中的聚类问题，显著提高聚类性能。

### 翻译

多模态属性图(MMAGs)是一种表达复杂数据模型，用于表示具有多模态属性(文本、图像等)的实体之间的复杂连接。在这类数据上进行聚类在现实场景中找到了许多实际应用，包括社交社区检测、医疗数据分析等。然而，正如我们的实证研究所揭示的，现有的多视图聚类解决方案很大程度上依赖于不同视图属性之间的高相关性，而忽略了MMAGs中由大型预训练语言和视觉模型输出的多模态属性的独特特征(如低模态间相关性和强烈的特征级噪声)，导致聚类性能次优。受前述实证观察和我们对图信号处理的理论分析启发，我们提出了双重图滤波(DGF)方案，创新地将特征级去噪组件纳入节点表示学习中，从而有效克服了现有多视图图聚类方法中采用的传统图滤波器的局限性。在此基础上，DGF包括一个三重交叉对比训练策略，利用跨模态、邻域和社区的实例级对比学习来学习鲁棒性和判别性的节点表示。我们在八个基准MMAG数据集上的综合实验表明，DGF能够在聚类质量上持续且显著地优于各种最先进的基线方法。


### 论文摘要

Multimodal Attributed Graphs (MMAGs) are an expressive data model for representing the complex interconnections among entities that associate attributes from multiple data modalities (text, images, etc.). Clustering over such data finds numerous practical applications in real scenarios, including social community detection, medical data analytics, etc. However, as revealed by our empirical studies, existing multi-view clustering solutions largely rely on the high correlation between attributes across various views and overlook the unique characteristics (e.g., low modality-wise correlation and intense feature-wise noise) of multimodal attributes output by large pre-trained language and vision models in MMAGs, leading to suboptimal clustering performance.   Inspired by foregoing empirical observations and our theoretical analyses with graph signal processing, we propose the Dual Graph Filtering (DGF) scheme, which innovatively incorporates a feature-wise denoising component into node representation learning, thereby effectively overcoming the limitations of traditional graph filters adopted in the extant multi-view graph clustering approaches. On top of that, DGF includes a tri-cross contrastive training strategy that employs instance-level contrastive learning across modalities, neighborhoods, and communities for learning robust and discriminative node representations. Our comprehensive experiments on eight benchmark MMAG datasets exhibit that DGF is able to outperform a wide range of state-of-the-art baselines consistently and significantly in terms of clustering quality measured against ground-truth labels.

---

## 215. Continual Audio Deepfake Detection via Universal Adversarial Perturbation

**论文链接:** [http://arxiv.org/abs/2511.19974v1](http://arxiv.org/abs/2511.19974v1)

**作者:** Wangjie Li, Lin Li, Qingyang Hong

**发布时间:** 2025-11-25

### GPT解析

### 总结

本文提出了一种将通用对抗扰动(UAP)整合到音频深度伪造检测中的新框架，使模型能够在不直接访问历史数据的情况下保留对历史欺骗分布的知识。

### 背景

语音合成和声音转换技术的快速发展引发了多媒体取证领域的重大安全担忧。当前检测模型虽然表现出色，但在应对不断演变的深度伪造攻击时难以保持有效性，且使用历史训练数据持续微调会带来巨大的计算和存储成本。

### 目的

解决当前检测模型的局限性，提出一种使模型能够在不直接访问过去数据的情况下保留对历史欺骗分布知识的方法。

### 方法

提出一种将通用对抗扰动(UAP)整合到音频深度伪造检测中的新框架，在微调过程中将UAP与预训练的自监督音频模型无缝集成。

### 主要发现

大量实验验证了该方法的有效性，展示了其作为音频深度伪造检测持续学习高效解决方案的潜力。

### 结论

该框架能够帮助模型在不直接访问历史数据的情况下保留对历史欺骗模式的知识，是解决音频深度伪造检测中持续学习问题的有效方法。

### 翻译

语音合成和声音转换技术的快速发展在多媒体取证领域引发了重大安全担忧。尽管当前检测模型表现出色，但在应对不断演变的深度伪造攻击时难以保持有效性。此外，使用历史训练数据持续微调这些模型会带来巨大的计算和存储成本。为解决这些局限性，我们提出了一种将通用对抗扰动(UAP)整合到音频深度伪造检测中的新框架，使模型能够在不直接访问过去数据的情况下保留对历史欺骗分布的知识。我们的方法在微调过程中将UAP与预训练的自监督音频模型无缝集成。大量实验验证了我们方法的有效性，展示了其作为音频深度伪造检测持续学习高效解决方案的潜力。


### 论文摘要

The rapid advancement of speech synthesis and voice conversion technologies has raised significant security concerns in multimedia forensics. Although current detection models demonstrate impressive performance, they struggle to maintain effectiveness against constantly evolving deepfake attacks. Additionally, continually fine-tuning these models using historical training data incurs substantial computational and storage costs. To address these limitations, we propose a novel framework that incorporates Universal Adversarial Perturbation (UAP) into audio deepfake detection, enabling models to retain knowledge of historical spoofing distribution without direct access to past data. Our method integrates UAP seamlessly with pre-trained self-supervised audio models during fine-tuning. Extensive experiments validate the effectiveness of our approach, showcasing its potential as an efficient solution for continual learning in audio deepfake detection.

---

## 216. Accelerating Wireless Distributed Learning via Hybrid Split and Federated Learning Optimization

**论文链接:** [http://arxiv.org/abs/2511.19851v1](http://arxiv.org/abs/2511.19851v1)

**作者:** Kun Guo, Xuefei Li, Xijun Wang, Howard H. Yang, Wei Feng, Tony Q. S. Quek

**发布时间:** 2025-11-25

### GPT解析

### 总结

该论文提出了一种加速混合分割和联邦学习(HSFL)的方法，通过解决学习模式选择、批量大小优化以及资源分配问题，显著提高了学习效率。

### 背景

联邦学习(FL)和分割学习(SL)是无线网络中两种有效的分布式学习范式，FL支持低延迟并行训练但精度较低，SL通过顺序训练实现更高精度但延迟增加。混合分割和联邦学习(HSFL)允许部分设备以FL模式运行，其他设备以SL模式运行。

### 目的

通过解决三个关键问题来加速HSFL：1)学习模式选择如何影响整体学习性能；2)学习模式如何与批量大小相互作用；3)如何将这些超参数与通信和计算资源联合优化以减少整体学习延迟。

### 方法

首先分析收敛性，揭示学习模式与批量大小之间的相互作用；然后构建延迟最小化问题；最后提出两阶段解决方案：使用块坐标下降法解决松弛问题获得局部最优解，再通过舍入算法恢复整数批量大小，实现接近最优的性能。

### 主要发现

实验结果表明，与现有方法相比，该方法显著加速了向目标精度的收敛。

### 结论

所提出的方法能够有效加速HSFL的收敛过程，解决了学习模式选择、批量大小优化和资源分配的关键问题。

### 翻译

联邦学习(FL)和分割学习(SL)是无线网络中两种有效的分布式学习范式，使移动设备能够协作训练模型而不共享原始数据。虽然FL支持低延迟并行训练，但它可能收敛到精度较低的模型。相比之下，SL通过顺序训练实现更高精度，但延迟增加。为了利用两者的优势，混合分割和联邦学习(HSFL)允许一些设备以FL模式运行，其他设备以SL模式运行。本文旨在通过解决三个关键问题来加速HSFL：1)学习模式选择如何影响整体学习性能？2)它如何与批量大小相互作用？3)如何将这些超参数与通信和计算资源联合优化以减少整体学习延迟？我们首先分析收敛性，揭示了学习模式与批量大小之间的相互作用。接下来，我们构建了延迟最小化问题，并提出了一种两阶段解决方案：使用块坐标下降法解决松弛问题以获得局部最优解，然后使用舍入算法恢复整数批量大小，实现接近最优的性能。实验结果表明，与现有方法相比，我们的方法显著加速了向目标精度的收敛。


### 论文摘要

Federated learning (FL) and split learning (SL) are two effective distributed learning paradigms in wireless networks, enabling collaborative model training across mobile devices without sharing raw data. While FL supports low-latency parallel training, it may converge to less accurate model. In contrast, SL achieves higher accuracy through sequential training but suffers from increased delay. To leverage the advantages of both, hybrid split and federated learning (HSFL) allows some devices to operate in FL mode and others in SL mode. This paper aims to accelerate HSFL by addressing three key questions: 1) How does learning mode selection affect overall learning performance? 2) How does it interact with batch size? 3) How can these hyperparameters be jointly optimized alongside communication and computational resources to reduce overall learning delay? We first analyze convergence, revealing the interplay between learning mode and batch size. Next, we formulate a delay minimization problem and propose a two-stage solution: a block coordinate descent method for a relaxed problem to obtain a locally optimal solution, followed by a rounding algorithm to recover integer batch sizes with near-optimal performance. Experimental results demonstrate that our approach significantly accelerates convergence to the target accuracy compared to existing methods.

---

## 217. On the Utility of Foundation Models for Fast MRI: Vision-Language-Guided Image Reconstruction

**论文链接:** [http://arxiv.org/abs/2511.19641v1](http://arxiv.org/abs/2511.19641v1)

**作者:** Ruimin Feng, Xingxin He, Ronald Mercer, Zachary Stewart, Fang Liu

**发布时间:** 2025-11-24

### GPT解析

### 总结

本研究探讨了视觉-语言基础模型如何通过提供高层次语义信息来改善欠采样MRI重建，提出了一种语义分布引导的重建框架，实验表明该方法优于传统正则化技术。

### 背景

欠采样MRI重建面临挑战，传统方法难以保留精细解剖结构和高质量感知信息。

### 目的

研究视觉-语言基础模型能否通过提供超越传统先验的高层次上下文信息来增强欠采样MRI重建。

### 方法

提出语义分布引导的重建框架，使用预训练视觉-语言模型编码重建图像和辅助信息为高层次语义特征，通过对比目标对齐重建表示与目标语义分布，并测试仅图像和图像-语言两种辅助信息的先验效果。

### 主要发现

图像语义先验保留精细解剖结构，感知质量更优（LPIPS值更低，Tenengrad分数更高，读者研究评分更好）；图像-语言信息扩展语义分布，实现重建属性的高层次控制；对比目标有效引导重建特征向期望语义分布发展。

### 结论

视觉-语言基础模型可通过语义空间优化改善欠采样MRI重建。

### 翻译

目的：研究视觉-语言基础模型是否能通过提供超越传统先验的高层次上下文信息来增强欠采样MRI重建。方法：我们提出了一种语义分布引导的重建框架，使用预训练的视觉-语言基础模型将重建图像和辅助信息编码为高层次语义特征。对比目标将重建表示与目标语义分布对齐，确保与高层次感知线索的一致性。所提出的目标可与各种基于深度学习的重建方法一起使用，并可以灵活地整合来自多模态源的语义先验。为了测试这些语义先验的有效性，我们评估了由仅图像或图像-语言辅助信息衍生的先验引导的重建结果。结果：在膝盖和脑部数据集上的实验表明，来自图像的语义先验保留了精细的解剖结构，实现了优越的感知质量，与传统正则化相比，具有更低的LPIPS值、更高的Tenengrad分数和读者研究中改善的评分。图像-语言信息进一步扩展了语义分布，并实现对重建属性的高层次控制。在所有评估中，对比目标一致地将重建特征引导到所需的语义分布，同时保持数据保真度，证明了所提出优化框架的有效性。结论：该研究强调视觉-语言基础模型可以通过语义空间优化改善欠采样MRI重建。


### 论文摘要

Purpose: To investigate whether a vision-language foundation model can enhance undersampled MRI reconstruction by providing high-level contextual information beyond conventional priors. Methods: We proposed a semantic distribution-guided reconstruction framework that uses a pre-trained vision-language foundation model to encode both the reconstructed image and auxiliary information into high-level semantic features. A contrastive objective aligns the reconstructed representation with the target semantic distribution, ensuring consistency with high-level perceptual cues. The proposed objective works with various deep learning-based reconstruction methods and can flexibly incorporate semantic priors from multimodal sources. To test the effectiveness of these semantic priors, we evaluated reconstruction results guided by priors derived from either image-only or image-language auxiliary information. Results: Experiments on knee and brain datasets demonstrate that semantic priors from images preserve fine anatomical structures and achieve superior perceptual quality, as reflected in lower LPIPS values, higher Tenengrad scores, and improved scores in the reader study, compared with conventional regularization. The image-language information further expands the semantic distribution and enables high-level control over reconstruction attributes. Across all evaluations, the contrastive objective consistently guided the reconstructed features toward the desired semantic distributions while maintaining data fidelity, demonstrating the effectiveness of the proposed optimization framework. Conclusion: The study highlights that vision-language foundation models can improve undersampled MRI reconstruction through semantic-space optimization.

---

## 218. A Hybrid Learning-to-Optimize Framework for Mixed-Integer Quadratic Programming

**论文链接:** [http://arxiv.org/abs/2511.19383v1](http://arxiv.org/abs/2511.19383v1)

**作者:** Viet-Anh Le, Mu Xie, Rahul Mangharam

**发布时间:** 2025-11-24

**备注:** submitted to L4DC 2026

### GPT解析

### 总结

本文提出了一种学习优化（L2O）框架，用于加速求解参数化混合整数二次规划问题，特别关注混合整数模型预测控制应用。

### 背景

混合整数二次规划和混合整数模型预测控制在实际应用中具有重要价值，但传统求解方法可能效率不高。

### 目的

开发一个学习优化框架，加速解决参数化混合整数二次规划问题，提高求解效率和解决方案质量。

### 方法

结合监督学习（用于最优性）和自监督学习（用于可行性）的混合L2O框架，使用神经网络学习从问题参数到最优整数解的映射，集成可微QP层计算连续变量，提出结合监督损失和自监督损失的混合损失函数。

### 主要发现

在两个基准MI-MPC问题上验证了所提框架的有效性，并与纯监督学习和自监督学习模型进行了比较，显示出优势。

### 结论

所提出的混合L2O框架能够有效加速MIQP问题的求解，特别是在MI-MPC应用中表现良好，结合监督学习和自监督学习的方法优于单一学习方法。

### 翻译

本文提出了一种学习优化（L2O）框架，用于加速求解参数化混合整数二次规划（MIQP）问题，特别关注混合整数模型预测控制（MI-MPC）应用。该框架通过集成监督学习（用于最优性）、自监督学习（用于可行性）和可微二次规划（QP）层，学习预测具有增强最优性和可行性的整数解，从而形成一个混合L2O框架。具体而言，使用神经网络学习从问题参数到最优整数解的映射，同时集成可微QP层来计算给定预测整数和问题参数对应的连续变量。此外，还提出了混合损失函数，该函数结合了针对全局最优解的监督损失和从问题目标与约束导出的自监督损失。在两个基准MI-MPC问题上验证了所提框架的有效性，并与纯监督学习和自监督学习模型进行了比较。


### 论文摘要

In this paper, we propose a learning-to-optimize (L2O) framework to accelerate solving parametric mixed-integer quadratic programming (MIQP) problems, with a particular focus on mixed-integer model predictive control (MI-MPC) applications. The framework learns to predict integer solutions with enhanced optimality and feasibility by integrating supervised learning (for optimality), self-supervised learning (for feasibility), and a differentiable quadratic programming (QP) layer, resulting in a hybrid L2O framework. Specifically, a neural network (NN) is used to learn the mapping from problem parameters to optimal integer solutions, while a differentiable QP layer is integrated to compute the corresponding continuous variables given the predicted integers and problem parameters. Moreover, a hybrid loss function is proposed, which combines a supervised loss with respect to the global optimal solution, and a self-supervised loss derived from the problem's objective and constraints. The effectiveness of the proposed framework is demonstrated on two benchmark MI-MPC problems, with comparative results against purely supervised and self-supervised learning models.

---

## 219. UISearch: Graph-Based Embeddings for Multimodal Enterprise UI Screenshots Retrieval

**论文链接:** [http://arxiv.org/abs/2511.19380v1](http://arxiv.org/abs/2511.19380v1)

**作者:** Maroun Ayli, Youssef Bakouny, Tushar Sharma, Nader Jalloul, Hani Seifeddine, Rima Kilany

**发布时间:** 2025-11-24

**备注:** 12 pages, 2 figures, 3 algorithms, 4 tables

### GPT解析

### 总结

本文提出了一种基于图的UI表示方法，通过将UI截图转换为属性图编码层次关系和空间排列，结合对比图自编码器学习多层次的视觉、结构和语义嵌入，并在UISearch多模态搜索框架中实现，在20,396个金融软件UI上实现了高准确率和低延迟。

### 背景

企业软件公司需要在产品和版本中维护数千个用户界面屏幕，这给设计一致性、模式发现和合规性检查带来了重大挑战。

### 目的

解决现有UI分析方法仅依赖视觉相似性或文本语义，缺乏对UI组合基本结构属性显式建模的问题。

### 方法

提出一种基于图的表示方法将UI截图转换为属性图编码层次关系和空间排列，使用对比图自编码器学习保留多层级相似性的嵌入，并实现为UISearch多模态搜索框架。

### 主要发现

结构嵌入比最先进的视觉编码器具有更好的判别力，代表了UI表示表达能力的基本进步；混合索引架构支持复杂查询和细粒度UI区分。

### 结论

所提出的UI表示方法在保持高准确率(0.92 Top-5)的同时实现了低延迟(47.5ms中延迟)，可扩展到大规模UI集合，为UI分析和搜索提供了新的可能性。

### 翻译

企业软件公司在产品和版本中维护数千个用户界面屏幕，给设计一致性、模式发现和合规性检查带来关键挑战。现有方法依赖视觉相似性或文本语义，缺乏对用户界面构成基本结构属性的显式建模。我们提出了一种新颖的基于图的表示方法，将UI截图转换为属性图，编码层次关系和空间排列，可能可推广到文档布局、架构图和其他结构化视觉领域。对比图自编码器学习保留视觉、结构和语义属性多层次相似性的嵌入。综合分析表明，我们的结构嵌入比最先进的视觉编码器具有更好的判别力，代表了UI表示表达能力的基本进步。我们在UISearch中实现了这种表示，这是一个多模态搜索框架，通过可组合的查询语言将结构嵌入与语义搜索相结合。在20,396个金融软件UI上，UISearch实现了0.92的Top-5准确率和47.5毫秒的中延迟(P95: 124毫秒)，可扩展到20,000多个屏幕。混合索引架构支持复杂查询，并支持仅使用视觉方法无法实现的细粒度UI区分。


### 论文摘要

Enterprise software companies maintain thousands of user interface screens across products and versions, creating critical challenges for design consistency, pattern discovery, and compliance check. Existing approaches rely on visual similarity or text semantics, lacking explicit modeling of structural properties fundamental to user interface (UI) composition. We present a novel graph-based representation that converts UI screenshots into attributed graphs encoding hierarchical relationships and spatial arrangements, potentially generalizable to document layouts, architectural diagrams, and other structured visual domains. A contrastive graph autoencoder learns embeddings preserving multi-level similarity across visual, structural, and semantic properties. The comprehensive analysis demonstrates that our structural embeddings achieve better discriminative power than state-of-the-art Vision Encoders, representing a fundamental advance in the expressiveness of the UI representation. We implement this representation in UISearch, a multi-modal search framework that combines structural embeddings with semantic search through a composable query language. On 20,396 financial software UIs, UISearch achieves 0.92 Top-5 accuracy with 47.5ms median latency (P95: 124ms), scaling to 20,000+ screens. The hybrid indexing architecture enables complex queries and supports fine-grained UI distinction impossible with vision-only approaches.

---

## 220. Leveraging Unlabeled Scans for NCCT Image Segmentation in Early Stroke Diagnosis: A Semi-Supervised GAN Approach

**论文链接:** [http://arxiv.org/abs/2511.19576v1](http://arxiv.org/abs/2511.19576v1)

**作者:** Maria Thoma, Michalis A. Savelonas, Dimitris K. Iakovidis

**发布时间:** 2025-11-24

**DOI:** 10.1109/BIBE66822.2025.0007

### GPT解析

### 总结

该研究提出了一种使用生成对抗网络(GANs)的半监督分割方法，用于准确描绘早期缺血性中风区域，即使在病变轻微或体积较小的情况下也能有效工作。

### 背景

缺血性中风是一种时间关键的医疗紧急情况，快速诊断对改善患者预后至关重要。非对比计算机断层扫描(NCCT)作为一线成像工具，但常无法揭示早期超急性阶段的细微缺血变化，这可能导致关键干预措施延迟。

### 目的

为了解决NCCT在早期缺血性中风诊断中的局限性，研究者引入了一种半监督分割方法，使用生成对抗网络(GANs)来准确描绘早期缺血性中风区域。

### 方法

该方法采用对抗性框架，从有限的标注NCCT扫描中学习，同时利用大量未标注扫描。通过组合使用Dice损失、交叉熵损失、特征匹配损失和自训练损失，模型能够识别和描绘早期梗死区域。

### 主要发现

在公开的急性缺血性中风数据集(AISD)上的实验表明，该方法有潜力增强诊断能力，减少手动标注的负担，并支持中风护理中更高效的临床决策制定。

### 结论

该半监督GAN方法为早期缺血性中风的诊断提供了有效解决方案，能够在有限标注数据的情况下提高NCCT对早期病变的检测能力。

### 翻译

缺血性中风是一种时间关键的医疗紧急情况，快速诊断对于改善患者预后至关重要。非对比计算机断层扫描(NCCT)作为一线成像工具，但常常无法揭示早期超急性阶段存在的细微缺血变化。这一局限可能会延迟关键干预措施。为了解决这一诊断挑战，我们引入了一种使用生成对抗网络(GANs)的半监督分割方法，以准确描绘早期缺血性中风区域。所提出的方法采用对抗性框架，从有限的标注NCCT扫描中有效学习，同时利用更大的未标注扫描池。通过使用Dice损失、交叉熵损失、特征匹配损失和自训练损失，模型能够识别和描绘早期梗死区域，即使它们很轻微或体积很小。在公开的急性缺血性中风数据集(AISD)上的实验表明，该方法有潜力增强诊断能力，减少手动标注的负担，并支持中风护理中更高效的临床决策制定。


### 论文摘要

Ischemic stroke is a time-critical medical emergency where rapid diagnosis is essential for improving patient outcomes. Non-contrast computed tomography (NCCT) serves as the frontline imaging tool, yet it often fails to reveal the subtle ischemic changes present in the early, hyperacute phase. This limitation can delay crucial interventions. To address this diagnostic challenge, we introduce a semi-supervised segmentation method using generative adversarial networks (GANs) to accurately delineate early ischemic stroke regions. The proposed method employs an adversarial framework to effectively learn from a limited number of annotated NCCT scans, while simultaneously leveraging a larger pool of unlabeled scans. By employing Dice loss, cross-entropy loss, a feature matching loss and a self-training loss, the model learns to identify and delineate early infarcts, even when they are faint or their size is small. Experiments on the publicly available Acute Ischemic Stroke Dataset (AISD) demonstrate the potential of the proposed method to enhance diagnostic capabilities, reduce the burden of manual annotation, and support more efficient clinical decision-making in stroke care.

---

## 221. Syn-GRPO: Self-Evolving Data Synthesis for MLLM Perception Reasoning

**论文链接:** [http://arxiv.org/abs/2511.19343v1](http://arxiv.org/abs/2511.19343v1)

**作者:** Qihan Huang, Haofei Zhang, Rong Wei, Yi Wang, Rui Tang, Mingli Song, Jie Song

**发布时间:** 2025-11-24

### GPT解析

### 总结

本研究提出了一种名为Syn-GRPO的新方法，通过在线数据生成器解决多模态大语言模型在强化学习中面临的数据质量低和响应多样性不足的问题。

### 背景

强化学习方法(如GRPO)用于多模态大语言模型感知能力的研究因其出色的泛化能力而受到广泛关注，但现有方法仍面临数据质量低的问题，无法从模型中引出多样化响应，限制了探索范围。

### 目的

解决现有强化学习方法中数据质量低的问题，扩大多模态大语言模型强化学习的探索范围，从根本上提升训练数据质量。

### 方法

提出Syn-GRPO方法，包含两个组件：(1)数据服务器：使用图像生成模型从现有样本合成新样本，采用解耦和异步方案实现高生成效率；(2)GRPO工作流：为数据服务器提供新的图像描述，利用多样性奖励监督模型预测用于合成多样化响应样本的图像描述。

### 主要发现

在三个视觉感知任务上的实验结果表明，Syn-GRPO大幅提高了数据质量，实现了比现有多模态大语言模型感知方法显著优越的性能，并且在扩展长期自进化强化学习方面展现出有希望的潜力。

### 结论

Syn-GRPO通过合成高质量、多样化的训练数据，有效解决了多模态大语言模型强化学习中的数据质量问题，为提升模型感知能力提供了新思路。

### 翻译

本研究提出了一种名为Syn-GRPO(合成-GRPO)的新方法，它采用在线数据生成器在GRPO训练中合成具有多样化响应的高质量训练数据。Syn-GRPO由数据服务器和GRPO工作流两个组件组成，数据服务器使用图像生成模型合成新样本，GRPO工作流提供图像描述并利用多样性奖励监督模型。实验结果表明，该方法显著提高了数据质量和性能，在扩展长期自进化强化学习方面展现出潜力。代码已公开。


### 论文摘要

RL (reinforcement learning) methods (e.g., GRPO) for MLLM (Multimodal LLM) perception ability has attracted wide research interest owing to its remarkable generalization ability. Nevertheless, existing reinforcement learning methods still face the problem of low data quality, where data samples cannot elicit diverse responses from MLLMs, thus restricting the exploration scope for MLLM reinforcement learning. Some methods attempt to mitigate this problem by imposing constraints on entropy, but none address it at its root. Therefore, to tackle this problem, this work proposes Syn-GRPO (Synthesis-GRPO), which employs an online data generator to synthesize high-quality training data with diverse responses in GRPO training. Specifically, Syn-GRPO consists of two components: (1) data server; (2) GRPO workflow. The data server synthesizes new samples from existing ones using an image generation model, featuring a decoupled and asynchronous scheme to achieve high generation efficiency. The GRPO workflow provides the data server with the new image descriptions, and it leverages a diversity reward to supervise the MLLM to predict image descriptions for synthesizing samples with diverse responses. Experiment results across three visual perception tasks demonstrate that Syn-GRPO improves the data quality by a large margin, achieving significant superior performance to existing MLLM perception methods, and Syn-GRPO presents promising potential for scaling long-term self-evolving RL. Our code is available at https://github.com/hqhQAQ/Syn-GRPO.

---

## 222. An Invariant Latent Space Perspective on Language Model Inversion

**论文链接:** [http://arxiv.org/abs/2511.19569v1](http://arxiv.org/abs/2511.19569v1)

**作者:** Wentao Ye, Jiaqi Hu, Haobo Wang, Xinpeng Ti, Zhiqing Xiao, Hao Chen, Liyao Li, Lei Feng, Sai Wu, Junbo Zhao

**发布时间:** 2025-11-24

**备注:** The Fortieth AAAI Conference on Artificial Intelligence (AAAI-26)

### GPT解析

### 总结

本研究提出了一种名为Inv²A的语言模型反转防御方法，通过不变潜在空间假设来解决LLM输出中的隐私安全问题，在9个数据集上平均提升4.77%的BLEU分数。

### 背景

语言模型反转（LMI）即从输出中恢复隐藏的提示，对用户隐私和系统安全构成了具体威胁。

### 目的

重新将LMI视为重用LLM的潜在空间，提出不变潜在空间假设（ILSH），并开发有效的防御策略。

### 方法

提出Inv²A方法，将LLM视为不变解码器，学习轻量级反向编码器映射输出到去噪伪表示；多输出时在表示层稀疏连接增加信息密度；训练分为对比对齐和监督强化两个阶段；可选邻域搜索改进局部性能。

### 主要发现

在9个数据集上平均以4.77%的BLEU分数优于基线方法，减少对大型反向语料库的依赖；普遍存在的防御措施提供的保护有限，需要更强策略。

### 结论

Inv²A通过不变潜在空间假设有效防御了语言模型反转攻击，在保持性能的同时减轻了对大规模训练数据的依赖。

### 翻译

语言模型反转（LMI），即从输出中恢复隐藏的提示，已成为对用户隐私和系统安全的具体威胁。我们将LMI重新定义为重用LLM自身的潜在空间，并提出了不变潜在空间假设（ILSH）：（1）来自同一源提示的多样化输出应保持一致的语义（源不变性），（2）输入-输出循环映射应在共享潜在空间内保持自洽（循环不变性）。据此，我们提出了Inv²A，将LLM视为不变解码器，只学习一个轻量级的反向编码器，将输出映射到去噪的伪表示。当有多个输出可用时，它们在表示层稀疏连接以增加信息密度。训练分为两个阶段：对比对齐（源不变性）和监督强化（循环不变性）。可选的无需训练的邻域搜索可改进局部性能。在涵盖用户和系统提示场景的9个数据集上，Inv²A平均以4.77%的BLEU分数优于基线方法，同时减少了对大型反向语料库的依赖。我们的分析进一步表明，普遍存在的防御措施提供的保护有限，突显了更强策略的必要性。本文涉及源代码和数据可在https://github.com/yyy01/Invariant_Attacker找到。


### 论文摘要

Language model inversion (LMI), i.e., recovering hidden prompts from outputs, emerges as a concrete threat to user privacy and system security. We recast LMI as reusing the LLM's own latent space and propose the Invariant Latent Space Hypothesis (ILSH): (1) diverse outputs from the same source prompt should preserve consistent semantics (source invariance), and (2) input<->output cyclic mappings should be self-consistent within a shared latent space (cyclic invariance). Accordingly, we present Inv^2A, which treats the LLM as an invariant decoder and learns only a lightweight inverse encoder that maps outputs to a denoised pseudo-representation. When multiple outputs are available, they are sparsely concatenated at the representation layer to increase information density. Training proceeds in two stages: contrastive alignment (source invariance) and supervised reinforcement (cyclic invariance). An optional training-free neighborhood search can refine local performance. Across 9 datasets covering user and system prompt scenarios, Inv^2A outperforms baselines by an average of 4.77% BLEU score while reducing dependence on large inverse corpora. Our analysis further shows that prevalent defenses provide limited protection, underscoring the need for stronger strategies. The source code and data involved in this paper can be found in https://github.com/yyy01/Invariant_Attacker.

---

## 223. What Drives Cross-lingual Ranking? Retrieval Approaches with Multilingual Language Models

**论文链接:** [http://arxiv.org/abs/2511.19324v1](http://arxiv.org/abs/2511.19324v1)

**作者:** Roksana Goworek, Olivia Macmillan-Scott, Eda B. Özyiğit

**发布时间:** 2025-11-24

### GPT解析

### 总结

该研究系统评估了四种跨语言信息检索干预方法，发现专门为CLIR训练的密集检索模型优于传统方法，对比学习能减轻语言偏见并提升弱对齐编码器性能，重新排序效果取决于训练数据质量，低资源和跨语言对改进最为显著。

### 背景

跨语言信息检索(CLIR)虽能访问多语言知识，但因资源差异、文字系统差异以及嵌入模型中弱跨语言语义对齐等问题仍具挑战性。现有流程依赖翻译和单语言检索启发式方法，增加计算开销和噪声，降低性能。

### 目的

系统性地评估四种干预类型在三个基准数据集上的效果，找出跨语言信息检索的最佳方法。

### 方法

评估四种干预类型：文档翻译、使用预训练编码器的多语言密集检索、在词组、短语和查询-文档级别的对比学习、跨编码器重新排序，并在三个基准数据集上进行测试。

### 主要发现

专门为CLIR训练的密集检索模型持续优于词汇匹配方法，且从文档翻译中获益很少；对比学习减轻了语言偏见，对初始对齐较弱的编码器产生显著改进；重新排序可能有效，但取决于跨编码器训练数据质量；尽管高资源语言主导整体性能，但在低资源和跨语言对上的改进最为显著。

### 结论

跨语言搜索系统应该优先考虑语义多语言嵌入和基于目标的学习对齐，而不是基于翻译的流程，特别是对于跨脚本和资源匮乏的语言。

### 翻译

跨语言信息检索(Cross-lingual information retrieval, CLIR) enables access to multilingual knowledge but remains challenging due to disparities in resources, scripts, and weak cross-lingual semantic alignment in embedding models. Existing pipelines often rely on translation and monolingual retrieval heuristics, which add computational overhead and noise, degrading performance. This work systematically evaluates four intervention types, namely document translation, multilingual dense retrieval with pretrained encoders, contrastive learning at word, phrase, and query-document levels, and cross-encoder re-ranking, across three benchmark datasets. We find that dense retrieval models trained specifically for CLIR consistently outperform lexical matching methods and derive little benefit from document translation. Contrastive learning mitigates language biases and yields substantial improvements for encoders with weak initial alignment, and re-ranking can be effective, but depends on the quality of the cross-encoder training data. Although high-resource languages still dominate overall performance, gains over lexical and document-translated baselines are most pronounced for low-resource and cross-script pairs. These findings indicate that cross-lingual search systems should prioritise semantic multilingual embeddings and targeted learning-based alignment over translation-based pipelines, particularly for cross-script and under-resourced languages.


### 论文摘要

Cross-lingual information retrieval (CLIR) enables access to multilingual knowledge but remains challenging due to disparities in resources, scripts, and weak cross-lingual semantic alignment in embedding models. Existing pipelines often rely on translation and monolingual retrieval heuristics, which add computational overhead and noise, degrading performance. This work systematically evaluates four intervention types, namely document translation, multilingual dense retrieval with pretrained encoders, contrastive learning at word, phrase, and query-document levels, and cross-encoder re-ranking, across three benchmark datasets. We find that dense retrieval models trained specifically for CLIR consistently outperform lexical matching methods and derive little benefit from document translation. Contrastive learning mitigates language biases and yields substantial improvements for encoders with weak initial alignment, and re-ranking can be effective, but depends on the quality of the cross-encoder training data. Although high-resource languages still dominate overall performance, gains over lexical and document-translated baselines are most pronounced for low-resource and cross-script pairs. These findings indicate that cross-lingual search systems should prioritise semantic multilingual embeddings and targeted learning-based alignment over translation-based pipelines, particularly for cross-script and under-resourced languages.

---

## 224. AutoEnv: Automated Environments for Measuring Cross-Environment Agent Learning

**论文链接:** [http://arxiv.org/abs/2511.19304v1](http://arxiv.org/abs/2511.19304v1)

**作者:** Jiayi Zhang, Yiran Peng, Fanqi Kong, Yang Cheng, Yifan Wu, Zhaoyang Yu, Jinyu Xiang, Jianhao Ruan, Jinlin Wang, Maojia Song, HongZhang Liu, Xiangru Tang, Bang Liu, Chenglin Wu, Yuyu Luo

**发布时间:** 2025-11-24

### GPT解析

### 总结

该研究提出了AutoEnv框架和AutoEnv-36数据集，用于评估智能体在跨环境学习中的表现，并设计了八种学习方法进行测试。研究发现固定学习方法难以扩展到异构环境，而环境自适应的学习方法选择可提高性能但存在收益递减现象。

### 背景

人类能通过学习不同环境中的潜在规则来适应多样化环境，而现有智能体通常只在单一领域内进化。跨环境学习缺乏标准化的异构环境集合和统一的智能体学习表示方法。

### 目的

解决跨环境学习缺乏可控制异构环境集合和统一表示方法的问题，为研究跨环境智能体学习提供测试平台。

### 方法

提出AutoEnv框架，将环境视为转换、观察和奖励的可分解分布，低成本生成异构世界；构建AutoEnv-36数据集（36个环境，358个验证关卡）；将智能体学习形式化为选择、优化和评估三阶段驱动的组件中心过程；设计八种学习方法进行评估。

### 主要发现

七种语言模型在AutoEnv-36上实现12-49%标准化奖励，表明数据集具有挑战性；任何单一学习方法的增益随环境数量增加而迅速减少；环境自适应学习方法选择能提高性能但随方法空间扩展表现出收益递减。

### 结论

强调了智能体学习对可扩展跨环境泛化的必要性和当前局限性，AutoEnv和AutoEnv-36可作为研究跨环境智能体学习的测试平台。

### 翻译

人类通过学习具有不同动态、观察和奖励结构的环境中的潜在规则，自然地适应多样化环境。相比之下，现有智能体通常在单一领域内通过自我进化展示改进，隐含假设环境分布固定。跨环境学习在很大程度上未被衡量：没有可控制的、异构环境的标准集合，也没有表示智能体学习方式的统一方法。我们通过两步解决这些差距。首先，我们提出AutoEnv，一个将环境视为转换、观察和奖励的可分解分布的自动化框架，能够低成本（平均4.12美元）生成异构世界。使用AutoEnv，我们构建了AutoEnv-36，一个包含36个环境和358个验证关卡的数据集，七种语言模型在该数据集上实现了12-49%的标准化奖励，表明AutoEnv-36具有挑战性。其次，我们将智能体学习形式化为由选择、优化和评估三个阶段驱动的以组件为中心的过程。使用这种形式化，我们设计了八种学习方法并在AutoEnv-36上评估它们。实验表明，任何单一学习方法的增益随着环境数量的增加而迅速减少，表明固定学习方法不能扩展到异构环境。环境自适应的学习方法选择显著提高了性能，但随着方法空间的扩展表现出收益递减。这些结果突显了智能体学习对可扩展跨环境泛化的必要性和当前局限性，并将AutoEnv和AutoEnv-36定位为研究跨环境智能体学习的测试平台。代码可在https://github.com/FoundationAgents/AutoEnv获取。


### 论文摘要

Humans naturally adapt to diverse environments by learning underlying rules across worlds with different dynamics, observations, and reward structures. In contrast, existing agents typically demonstrate improvements via self-evolving within a single domain, implicitly assuming a fixed environment distribution. Cross-environment learning has remained largely unmeasured: there is no standard collection of controllable, heterogeneous environments, nor a unified way to represent how agents learn. We address these gaps in two steps. First, we propose AutoEnv, an automated framework that treats environments as factorizable distributions over transitions, observations, and rewards, enabling low-cost (4.12 USD on average) generation of heterogeneous worlds. Using AutoEnv, we construct AutoEnv-36, a dataset of 36 environments with 358 validated levels, on which seven language models achieve 12-49% normalized reward, demonstrating the challenge of AutoEnv-36. Second, we formalize agent learning as a component-centric process driven by three stages of Selection, Optimization, and Evaluation applied to an improvable agent component. Using this formulation, we design eight learning methods and evaluate them on AutoEnv-36. Empirically, the gain of any single learning method quickly decrease as the number of environments increases, revealing that fixed learning methods do not scale across heterogeneous environments. Environment-adaptive selection of learning methods substantially improves performance but exhibits diminishing returns as the method space expands. These results highlight both the necessity and the current limitations of agent learning for scalable cross-environment generalization, and position AutoEnv and AutoEnv-36 as a testbed for studying cross-environment agent learning. The code is avaiable at https://github.com/FoundationAgents/AutoEnv.

---

## 225. 论文ID: 2511.20640v1

**错误:** 无法读取论文信息 - [Errno 2] No such file or directory: 'storage/result/2511.20640v1.json'

---

## 226. Optimization of Sums of Bivariate Functions: An Introduction to Relaxation-Based Methods for the Case of Finite Domains

**论文链接:** [http://arxiv.org/abs/2511.20607v1](http://arxiv.org/abs/2511.20607v1)

**作者:** Nils Müller

**发布时间:** 2025-11-25

**备注:** 59 pages, 7 figures

### GPT解析

### 总结

研究具有n>2个变量的函数优化问题，这些函数可表示为多个只包含2个变量的函数之和，称为双变量和函数，定义在有限域上。

### 背景

优化双变量和函数的复杂性被证明是NP等价的，且在优化过程中存在免费午餐现象。

### 目的

推导可处理的问题表述，解决双变量和函数的优化难题。

### 方法

基于目标函数的测度值扩展（称为松弛）、L²逼近和熵正则化，推导可通过线性规划、坐标上升和闭式解解决的可处理问题表述。

### 主要发现

使用从双变量边缘重建测度的一般结果，探讨了将松弛的可处理版本应用于双变量和函数的局限性；实验提供了对可建模为双变量和函数的不同函数类的见解。

### 结论

双变量和函数的优化问题虽然复杂，但通过提出的松弛方法可以得到有效解决。

### 翻译

我们研究了在有限域上具有n>2个参数且可以表示为多个只包含n个参数中2个的函数之和的函数的优化，称为双变量和函数。证明了优化双变量和函数的复杂性是NP等价的，并且在优化双变量和函数中存在免费午餐现象。基于目标函数的测度值扩展（称为松弛）、L²逼近和熵正则化，我们推导出了几种可处理的问题表述，可以通过线性规划、坐标上升和闭式解来解决。使用从双变量边缘重建测度的一般结果，我们研究了将此类松弛的可处理版本应用于双变量和函数的局限性。在随机函数、顶点着色和信号重建问题中应用推导出的算法的实验，提供了对可以建模为双变量和函数的不同函数类的见解。


### 论文摘要

We study the optimization of functions with $n>2$ arguments that have a representation as a sum of several functions that have only $2$ of the $n$ arguments each, termed sums of bivariates, on finite domains. The complexity of optimizing sums of bivariates is shown to be NP-equivalent and it is shown that there exists free lunch in the optimization of sums of bivariates. Based on measure-valued extensions of the objective function, so-called relaxations, $\ell^2$-approximation, and entropy-regularization, we derive several tractable problem formulations solvable with linear programming, coordinate ascent as well as with closed-form solutions. The limits of applying tractable versions of such relaxations to sums of bivariates are investigated using general results for reconstructing measures from their bivariate marginals. Experiments in which the derived algorithms are applied to random functions, vertex coloring, and signal reconstruction problems provide insights into qualitatively different function classes that can be modeled as sums of bivariates.

---

## 227. New York Smells: A Large Multimodal Dataset for Olfaction

**论文链接:** [http://arxiv.org/abs/2511.20544v1](http://arxiv.org/abs/2511.20544v1)

**作者:** Ege Ozguroglu, Junbang Liang, Ruoshi Liu, Mia Chiquier, Michael DeTienne, Wesley Wei Qian, Alexandra Horowitz, Andrew Owens, Carl Vondrick

**发布时间:** 2025-11-25

**备注:** Project website at https://smell.cs.columbia.edu

### GPT解析

### 总结

该研究提出了一个名为'纽约气味'的大型嗅觉-图像配对数据集，解决了机器嗅觉研究中的数据稀缺问题，并通过实验证明视觉数据有助于跨模态嗅觉表征学习。

### 背景

嗅觉是动物感知世界的重要方式，但机器难以充分获取这种化学感官模态。缺乏在自然环境中收集的多样化、多模态嗅觉训练数据是一个主要瓶颈。

### 目的

创建一个大型数据集，包含在自然环境中采集的成对图像和嗅觉信号，以促进机器嗅觉研究。

### 方法

提出了'纽约气味'数据集，包含7,000个气味-图像对，来自3,500个不同物体，覆盖室内和室外环境，比现有嗅觉数据集多约70倍的对象数量。建立了三个基准任务：跨模态气味到图像检索、仅从气味识别场景物体材料、草种细粒度区分。

### 主要发现

视觉数据能够促进跨模态嗅觉表征学习，学习到的嗅觉表征优于广泛使用的手工设计特征。

### 结论

提出的数据集和基准任务为机器嗅觉研究提供了重要资源，视觉信息可以辅助机器学习嗅觉表征。

### 翻译

虽然嗅觉对动物如何感知世界至关重要，但这种丰富的化学感官模态在很大程度上仍然无法被机器访问。一个关键瓶颈是缺乏在自然环境中收集的多样化、多模态嗅觉训练数据。我们提出了'纽约气味'，一个在野外环境中捕获的大型配对图像和嗅觉信号数据集。我们的数据集包含7,000个气味-图像对，来自室内和室外环境中的3,500个不同物体，对象数量比现有嗅觉数据集多约70倍。我们的基准有三个任务：跨模态气味到图像检索，仅从气味识别场景、物体和材料，以及草种之间的细粒度区分。通过对我们数据集的实验，我们发现视觉数据能够促进跨模态嗅觉表征学习，并且我们学习的嗅觉表征优于广泛使用的手工设计特征。


### 论文摘要

While olfaction is central to how animals perceive the world, this rich chemical sensory modality remains largely inaccessible to machines. One key bottleneck is the lack of diverse, multimodal olfactory training data collected in natural settings. We present New York Smells, a large dataset of paired image and olfactory signals captured ``in the wild.'' Our dataset contains 7,000 smell-image pairs from 3,500 distinct objects across indoor and outdoor environments, with approximately 70$\times$ more objects than existing olfactory datasets. Our benchmark has three tasks: cross-modal smell-to-image retrieval, recognizing scenes, objects, and materials from smell alone, and fine-grained discrimination between grass species. Through experiments on our dataset, we find that visual data enables cross-modal olfactory representation learning, and that our learned olfactory representations outperform widely-used hand-crafted features.

---

## 228. Dance Style Classification using Laban-Inspired and Frequency-Domain Motion Features

**论文链接:** [http://arxiv.org/abs/2511.20469v1](http://arxiv.org/abs/2511.20469v1)

**作者:** Ben Hamscher, Arnold Brosch, Nicolas Binninger, Maksymilian Jan Dejna, Kira Maag

**发布时间:** 2025-11-25

### GPT解析

### 总结

本研究提出了一种轻量级舞蹈风格分类框架，通过基于拉班动作分析的时空描述符和快速傅里叶变换特征，有效捕捉舞蹈的空间协调性和节奏性，实现低计算复杂度下的稳健分类。

### 背景

舞蹈是人类文化的重要组成部分，用于表达情感和讲述故事。基于运动数据识别和区分舞蹈风格是人类活动识别中的复杂问题，因为许多舞蹈风格具有相似的姿势、手势和时间运动模式。

### 目的

开发一种轻量级的舞蹈风格分类框架，从视频中提取的姿态估计来确定运动特征，有效捕捉舞蹈风格特点，同时保持低计算复杂度。

### 方法

提出受拉班动作分析启发的时空描述符，捕捉局部关节动力学（如速度、加速度和上半身角运动）；集成快速傅里叶变换特征，编码运动的节奏性和周期性方面；构建不需要复杂模型架构的分类器。

### 主要发现

时空描述符能有效捕捉舞蹈的空间协调性；快速傅里叶变换特征能捕捉舞蹈的节奏性和周期性方面；可解释的运动表示能有效捕捉舞蹈风格的细微差别；该方法在低计算复杂度下实现了稳健分类。

### 结论

轻量级舞蹈风格分类框架结合时空描述符和频域特征，能在不需要复杂模型架构的情况下有效识别和区分不同舞蹈风格，展示了可解释运动表示在捕捉舞蹈风格细微差别方面的有效性。

### 翻译

舞蹈是人类文化的重要组成部分，是传达情感和讲述故事的工具。基于运动数据识别和区分舞蹈风格是人类活动识别中的一个复杂问题，因为许多风格具有相似的姿势、手势和时间运动模式。这项工作提出了一种轻量级的舞蹈风格分类框架，该框架基于从视频中提取的姿态估计来确定运动特征。我们提出了受拉班动作分析启发的时空描述符。这些特征捕捉了局部关节动力学，如速度、加速度和上半身的角运动，实现了空间协调性的结构化表示。为了进一步编码运动的节奏性和周期性方面，我们集成了快速傅里叶变换特征，这些特征在频域中表征运动模式。所提出的方法以低计算量实现了不同舞蹈风格的稳健分类，因为不需要复杂的模型架构，并且表明可解释的运动表示可以有效捕捉风格的细微差别。


### 论文摘要

Dance is an essential component of human culture and serves as a tool for conveying emotions and telling stories. Identifying and distinguishing dance genres based on motion data is a complex problem in human activity recognition, as many styles share similar poses, gestures, and temporal motion patterns. This work presents a lightweight framework for classifying dance styles that determines motion characteristics based on pose estimates extracted from videos. We propose temporal-spatial descriptors inspired by Laban Movement Analysis. These features capture local joint dynamics such as velocity, acceleration, and angular movement of the upper body, enabling a structured representation of spatial coordination. To further encode rhythmic and periodic aspects of movement, we integrate Fast Fourier Transform features that characterize movement patterns in the frequency domain. The proposed approach achieves robust classification of different dance styles with low computational effort, as complex model architectures are not required, and shows that interpretable motion representations can effectively capture stylistic nuances.

---

## 229. ShelfRectNet: Single View Shelf Image Rectification with Homography Estimation

**论文链接:** [http://arxiv.org/abs/2511.20335v1](http://arxiv.org/abs/2511.20335v1)

**作者:** Onur Berk Tore, Ibrahim Samil Yalciner, Server Calap

**发布时间:** 2025-11-25

### GPT解析

### 总结

本文提出了一种基于深度学习的单应性矩阵估计方法，用于校正从任意角度捕获的货架图像。该方法使用ConvNeXt主干网络和归一化坐标回归，并通过合成单应性数据增强策略提高泛化能力。实验表明，该方法在测试集上达到1.298像素的平均角点误差，与传统和深度学习方法相比具有竞争性的性能和推理速度。

### 背景

从单张图像估计单应性矩阵具有挑战性但在实际应用中很有价值，特别是在零售领域，通常只有一个视角用于货架监控和产品对齐。

### 目的

提出一个深度学习框架，预测4点参数化的单应性矩阵，以校正从任意角度捕获的货架图像。

### 方法

使用基于ConvNeXt的主干网络增强特征表示；采用归一化坐标回归提高稳定性；引入新的增强策略，通过建模和合成单应性采样来解决数据稀缺问题并提高泛化能力。

### 主要发现

在测试集上实现了平均角点误差1.298像素；与传统计算机视觉和基于深度学习的方法相比，在准确性和推理速度方面都具有竞争力。

### 结论

这些结果确立了该方法作为真实世界单视图校正的稳健且高效的解决方案。

### 翻译

从单张图像估计单应性矩阵仍然是一项具有挑战性但具有实际价值的任务，特别是在零售等领域，通常只有一个视角可用于货架监控和产品对齐。在本文中，我们提出了一个深度学习框架，预测4点参数化的单应性矩阵，以校正从任意角度捕获的货架图像。我们的模型利用基于ConvNeXt的主干网络来增强特征表示，并采用归一化坐标回归以提高稳定性。为解决数据稀缺问题并提高泛化能力，我们引入了一种新的增强策略，通过建模和合成单应性采样。我们的方法在测试集上实现了1.298像素的平均角点误差。与传统的计算机视觉和基于深度学习的方法相比，我们的方法在准确性和推理速度方面都表现出竞争性性能。这些结果共同确立了我们的方法作为真实世界单视图校正的稳健且高效的解决方案。为鼓励该领域的进一步研究，我们将公开我们的数据集ShelfRectSet和代码。


### 论文摘要

Estimating homography from a single image remains a challenging yet practically valuable task, particularly in domains like retail, where only one viewpoint is typically available for shelf monitoring and product alignment. In this paper, we present a deep learning framework that predicts a 4-point parameterized homography matrix to rectify shelf images captured from arbitrary angles. Our model leverages a ConvNeXt-based backbone for enhanced feature representation and adopts normalized coordinate regression for improved stability. To address data scarcity and promote generalization, we introduce a novel augmentation strategy by modeling and sampling synthetic homographies. Our method achieves a mean corner error of 1.298 pixels on the test set. When compared with both classical computer vision and deep learning-based approaches, our method demonstrates competitive performance in both accuracy and inference speed. Together, these results establish our approach as a robust and efficient solution for realworld single-view rectification. To encourage further research in this domain, we will make our dataset, ShelfRectSet, and code publicly available

---

## 230. MXtalTools: A Toolkit for Machine Learning on Molecular Crystals

**论文链接:** [http://arxiv.org/abs/2511.20327v1](http://arxiv.org/abs/2511.20327v1)

**作者:** Michael Kilgour, Mark E. Tuckerman, Jutta Rogal

**发布时间:** 2025-11-25

**备注:** 16 pages, 11 figures

### GPT解析

### 总结

MXtalTools是一个用于分子晶体数据驱动建模的灵活Python包，支持分子固体的机器学习研究，提供多种实用工具类和模块化功能。

### 背景

分子固体的机器学习研究需要专门的工具来处理数据集、模型训练和晶体结构分析。

### 目的

开发一个综合性的Python包，使研究人员能够进行分子晶体的数据驱动建模和机器学习研究。

### 方法

提供多种实用工具类，包括数据集合成与整理、模型训练与推理工作流、晶体参数化与表示、晶体结构采样与优化，以及端到端可微分晶体采样、构建和分析功能。

### 主要发现

模块化功能可以集成到现有工作流或组合使用构建新建模管道，利用CUDA加速实现高通量晶体建模。

### 结论

MXtalTools是一个灵活、开源的工具，可用于分子晶体研究，代码已在GitHub上开源，ReadTheDocs上有详细文档。

### 翻译

我们介绍了MXtalTools，这是一个用于分子晶体数据驱动建模的灵活Python包，促进了分子固体的机器学习研究。MXtalTools包含多种实用工具类：(1)分子和晶体数据集的合成、整理和整理，(2)模型训练和推理的集成工作流，(3)晶体参数化和表示，(4)晶体结构采样和优化，(5)端到端可微分晶体采样、构建和分析。我们的模块化功能可以集成到现有工作流中或组合使用来构建新的建模管道。MXtalTools利用CUDA加速实现高通量晶体建模。Python代码在我们的GitHub页面上开源，ReadTheDocs上有详细文档。


### 论文摘要

We present MXtalTools, a flexible Python package for the data-driven modelling of molecular crystals, facilitating machine learning studies of the molecular solid state. MXtalTools comprises several classes of utilities: (1) synthesis, collation, and curation of molecule and crystal datasets, (2) integrated workflows for model training and inference, (3) crystal parameterization and representation, (4) crystal structure sampling and optimization, (5) end-to-end differentiable crystal sampling, construction and analysis. Our modular functions can be integrated into existing workflows or combined and used to build novel modelling pipelines. MXtalTools leverages CUDA acceleration to enable high-throughput crystal modelling. The Python code is available open-source on our GitHub page, with detailed documentation on ReadTheDocs.

---

## 231. Lower Bias, Higher Welfare: How Creator Competition Reshapes Bias-Variance Tradeoff in Recommendation Platforms?

**论文链接:** [http://arxiv.org/abs/2511.20289v1](http://arxiv.org/abs/2511.20289v1)

**作者:** Kang Wang, Renzhe Xu, Bo Li

**发布时间:** 2025-11-25

**备注:** KDD 2026

### GPT解析

### 总结

本文研究了内容创作者竞争环境下用户表示学习中的偏差-方差权衡问题，建立了博弈论模型分析平台在用户特征估计中的最优正则化策略，并通过理论分析和实验验证表明在战略环境中，平台应采用较弱正则化以降低偏差，从而提高用户福利。

### 背景

在静态环境中，用户表示学习的偏差-方差权衡已有深入研究，但在现代内容平台中，内容创作者会战略性地适应平台激励，这使得权衡关系变得更加复杂。

### 目的

分析内容创作者竞争如何重塑偏差-方差权衡以最大化用户福利，并确定平台在用户特征估计中的最优正则化策略。

### 方法

引入'内容创作者竞争与偏差-方差权衡'框架，这是一个可计算的博弈论模型，用于捕捉平台关于用户特征估计中正则化强度的决策；通过理论分析和在合成及真实世界基准数据集上的实验来验证结论。

### 主要发现

与非战略环境相比，内容创作者竞争使平台的最优政策向较弱正则化转变，从而在偏差-方差权衡中偏向较低偏差；在战略环境中，减少偏差能带来更高的用户福利。

### 结论

在存在内容创作者竞争的现实推荐系统中，平台应采用较弱正则化策略以降低偏差，这有助于提高用户福利，为推荐算法设计提供了实践启示。

### 翻译

理解用户表示学习中的偏差-方差权衡对于提高现代内容平台的推荐质量至关重要。虽然在静态环境中已有充分研究，但当内容创作者战略性地适应平台激励时，这种权衡会变得显著复杂。为了分析这种竞争如何重塑以最大化用户福利的权衡关系，我们引入了内容创作者竞争与偏差-方差权衡框架，这是一个可计算的博弈论模型，用于捕捉平台在用户特征估计中对正则化强度的决策。我们推导并比较了两种关键设置下平台的最优策略：具有固定内容的非战略基线和创作者对平台算法设计做出竞争反应的战略环境。在简化模型中的理论分析表明，与非战略环境相比，内容创作者竞争使平台的最优政策向较弱正则化转变，从而在偏差-方差权衡中偏向较低偏差。为了验证和评估这些见解在简化设置之外的稳健性，我们在合成和真实世界基准数据集上进行了大量实验。实证结果一致支持我们的理论结论：在战略环境中，减少偏差会导致更高的用户福利。这些发现为存在内容创作者竞争的现实世界推荐算法设计提供了实践启示。


### 论文摘要

Understanding the bias-variance tradeoff in user representation learning is essential for improving recommendation quality in modern content platforms. While well studied in static settings, this tradeoff becomes significantly more complex when content creators strategically adapt to platform incentives. To analyze how such competition reshapes the tradeoff for maximizing user welfare, we introduce the Content Creator Competition with Bias-Variance Tradeoff framework, a tractable game-theoretic model that captures the platform's decision on regularization strength in user feature estimation. We derive and compare the platform's optimal policy under two key settings: a non-strategic baseline with fixed content and a strategic environment where creators compete in response to the platform's algorithmic design.   Our theoretical analysis in a stylized model shows that, compared to the non-strategic environment, content creator competition shifts the platform's optimal policy toward weaker regularization, thereby favoring lower bias in the bias-variance tradeoff. To validate and assess the robustness of these insights beyond the stylized setting, we conduct extensive experiments on both synthetic and real-world benchmark datasets. The empirical results consistently support our theoretical conclusion: in strategic environments, reducing bias leads to higher user welfare. These findings offer practical implications for the design of real-world recommendation algorithms in the presence of content creator competition.

---

## 232. 论文ID: 2511.20224v1

**错误:** 无法读取论文信息 - [Errno 2] No such file or directory: 'storage/result/2511.20224v1.json'

---

## 233. OmniAlpha: A Sequence-to-Sequence Framework for Unified Multi-Task RGBA Generation

**论文链接:** [http://arxiv.org/abs/2511.20211v1](http://arxiv.org/abs/2511.20211v1)

**作者:** Hao Yu, Jiabo Zhan, Zile Wang, Jinglin Wang, Huaisong Zhang, Hongyu Li, Xinrui Chen, Yongxian Wei, Chun Yuan

**发布时间:** 2025-11-25

### GPT解析

### 总结

本文提出了OmniAlpha，首个统一的多任务生成框架，用于序列到序列的RGBA图像生成和编辑，解决了现有生成模型在RGBA操作上的碎片化问题。

### 背景

生成模型在RGB合成方面表现出色，但现实世界应用需要RGBA操作。当前领域存在碎片化：专门的单一任务模型处理alpha通道但缺乏多功能性，而统一的多任务框架局限于RGB领域。

### 目的

弥合RGB与RGBA处理之间的关键差距，创建一个统一的多任务框架，能够同时处理RGBA图像的多个层。

### 方法

设计了MSRoPE-BiL，一种具有双向可扩展层轴的新RoPE方法，用于Diffusion Transformer主干；构建了AlphaLayers数据集，包含1000个高质量多层三元组；在21个不同任务上对框架进行综合训练。

### 主要发现

统一方法始终优于专用基线；在AIM-500上无遮罩抠图的SAD相对减少了84.8%；在层条件完成中赢得了超过90%的人体偏好。

### 结论

统一的多任务模型可以学习RGBA的优越共享表示，为更强大的、层感知的生成系统铺平道路。

### 翻译

生成模型在RGB合成方面表现出色，但现实世界的应用需要RGBA操作。这导致了碎片化的现状：专门的单一任务模型能够处理alpha通道但缺乏多功能性，而统一的多任务框架仅限于RGB领域。为了弥合这一关键差距，我们提出了OmniAlpha，这是首个用于序列到序列RGBA图像生成和编辑的统一多任务生成框架。其架构特点包括MSRoPE-BiL，一种具有双向可扩展层轴的新RoPE方法，用于其Diffusion Transformer主干，使能够同时处理多个输入和目标RGBA层。为支持这一框架，我们引入了AlphaLayers，这是一个包含1000个高质量多层三元组的新数据集，通过新颖的自动合成和过滤管道构建。在该数据集上对21个不同任务进行联合训练的广泛实验表明，我们的统一方法始终优于强大的专用基线。最显著的是，OmniAlpha在AIM-500上的无遮罩抠图实现了SAD相对减少84.8%，在层条件完成中赢得了超过90%的人体偏好。我们的工作证明，统一的多任务模型可以学习RGBA的优越共享表示，为更强大的、层感知的生成系统铺平了道路。


### 论文摘要

Generative models have excelled in RGB synthesis, but real-world applications require RGBA manipulation. This has led to a fragmented landscape: specialized, single-task models handle alpha but lack versatility, while unified multi-task frameworks are confined to the RGB domain. To bridge this critical gap, we propose OmniAlpha, the first unified, multi-task generative framework for sequence-to-sequence RGBA image generation and editing. Its architecture features MSRoPE-BiL, a novel RoPE method with a bi-directionally extendable layer axis for its Diffusion Transformer (DiT) backbone, enabling the concurrent processing of multiple input and target RGBA layers. To power this framework, we introduce AlphaLayers, a new dataset of 1,000 high-quality, multi-layer triplets, built via a novel automated synthesis and filter pipeline. Jointly training OmniAlpha on this dataset across a comprehensive suite of 21 diverse tasks, extensive experiments demonstrate that our unified approach consistently outperforms strong, specialized baselines. Most notably, OmniAlpha achieves a dramatic 84.8% relative reduction in SAD for mask-free matting on AIM-500 and wins over 90% of human preferences in layer-conditioned completion. Our work proves that a unified, multi-task model can learn a superior shared representation for RGBA, paving the way for more powerful, layer-aware generative systems.

---

## 234. In-Context Compositional Learning via Sparse Coding Transformer

**论文链接:** [http://arxiv.org/abs/2511.20194v1](http://arxiv.org/abs/2511.20194v1)

**作者:** Wei Chen, Jingxi Yu, Zichen Miao, Qiang Qiu

**发布时间:** 2025-11-25

**备注:** NeurIPS 2025

### GPT解析

### 总结

本文提出了一种受稀疏编码启发的Transformer注意力机制重新表述方法，以增强其处理组合学习任务的能力，在S-RAVEN和RAVEN数据集上证明了其有效性，特别是在标准Transformer失败的情况下仍能保持性能。

### 背景

Transformer架构在语言、视觉和多模态任务中取得了显著成功，但其在处理上下文组合学习任务时面临挑战，因为这些任务需要模型从上下文示例中推断组合规则，而Transformer天生缺乏处理组合任务的结构归纳偏置。

### 目的

提出一种注意力机制的重新表述，以增强Transformer处理组合任务的能力，特别是在组合泛化任务中的表现。

### 方法

受稀疏编码原理启发，将注意力块重新解释为通过将输入投影到两组学习到的字典原子（编码字典和解码字典）来将输入映射到输出；编码字典将输入分解为表示组合结构的系数，并对这些系数施加稀疏性；然后使用稀疏系数线性组合解码字典原子生成输出；此外，提出将目标问题的系数估计为从上下文示例获得的系数的线性组合。

### 主要发现

在S-RAVEN和RAVEN数据集上证明了该方法的有效性；对于某些组合泛化任务，当标准Transformer失败时，该方法仍能保持性能，这归功于其学习和应用组合规则的能力。

### 结论

所提出的方法能够有效处理组合任务，特别是在标准Transformer失败的情况下，展示了其在组合学习任务中的优越性。

### 翻译

Transformer架构在语言、视觉和多模态任务中取得了显著成功，并且人们越来越需要它们来解决上下文组合学习任务。在这些任务中，模型通过从上下文示例中推断组合规则来解决目标问题，这些示例由底层规则结构化的基本组件组成。然而，其中一些任务对Transformer仍然具有挑战性，因为它们并非天生设计用来处理组合任务，且结构归纳偏置有限。在这项工作中，受稀疏编码原理启发，我们提出了对注意力的重新表述，以增强其处理组合任务的能力。在稀疏编码中，数据被表示为字典原子的稀疏组合，其系数捕获了它们的组合规则。具体来说，我们将注意力块重新解释为通过将输入投影到两组学习到的字典原子（编码字典和解码字典）来将输入映射到输出的过程。编码字典将输入分解为一组系数，这些系数表示输入的组合结构。为了增强结构化表示，我们对这些系数施加稀疏性。然后，使用这些稀疏系数线性组合解码字典原子以生成输出。此外，为了帮助组合泛化任务，我们提出将目标问题的系数估计为从上下文示例获得的系数的线性组合。我们在S-RAVEN和RAVEN数据集上证明了我们方法的有效性。对于某些组合泛化任务，由于学习和应用组合规则的能力，我们的方法甚至在标准Transformer失败时仍能保持性能。


### 论文摘要

Transformer architectures have achieved remarkable success across language, vision, and multimodal tasks, and there is growing demand for them to address in-context compositional learning tasks. In these tasks, models solve the target problems by inferring compositional rules from context examples, which are composed of basic components structured by underlying rules. However, some of these tasks remain challenging for Transformers, which are not inherently designed to handle compositional tasks and offer limited structural inductive bias. In this work, inspired by the principle of sparse coding, we propose a reformulation of the attention to enhance its capability for compositional tasks. In sparse coding, data are represented as sparse combinations of dictionary atoms with coefficients that capture their compositional rules. Specifically, we reinterpret the attention block as a mapping of inputs into outputs through projections onto two sets of learned dictionary atoms: an encoding dictionary and a decoding dictionary. The encoding dictionary decomposes the input into a set of coefficients, which represent the compositional structure of the input. To enhance structured representations, we impose sparsity on these coefficients. The sparse coefficients are then used to linearly combine the decoding dictionary atoms to generate the output. Furthermore, to assist compositional generalization tasks, we propose estimating the coefficients of the target problem as a linear combination of the coefficients obtained from the context examples. We demonstrate the effectiveness of our approach on the S-RAVEN and RAVEN datasets. For certain compositional generalization tasks, our method maintains performance even when standard Transformers fail, owing to its ability to learn and apply compositional rules.

---

## 235. Multi Head Attention Enhanced Inception v3 for Cardiomegaly Detection

**论文链接:** [http://arxiv.org/abs/2511.20101v1](http://arxiv.org/abs/2511.20101v1)

**作者:** Abishek Karthik, Pandiyaraju V

**发布时间:** 2025-11-25

### GPT解析

### 总结

该研究介绍了一种基于深度学习和注意力机制的集成方法，用于通过X光图像自动检测心脏肥大，采用CNN架构和多层注意力机制提高诊断敏感性。

### 背景

医疗行业已被新型成像技术显著改变，不仅在心血管疾病诊断方面，也在结构性异常如心脏肥大的可视化方面。心脏肥大是需要准确诊断的重要疾病。

### 目的

开发一种自动检测心脏肥大的系统，通过深度学习工具和注意力机制的集成方法，利用X光图像进行准确诊断。

### 方法

1. 数据收集阶段：收集各种类型的标注X光图像数据；2. 预处理模块：优化图像质量；3. 系统架构：使用CNN配置，采用Inception V3模型作为关键模块；4. 多层注意力机制：特别是多头注意力机制，自动学习特征；5. 评估阶段：严格评估模型，使用准确度、精确度等指标。

### 主要发现

模型在心脏肥大检测方面表现出色：准确率95.6，精确度95.2，召回率96.2，敏感性95.7，特异性96.1，AUC值96.0。通过选择性关注输入图像的特定区域，模型能够以高敏感性识别心脏肥大。

### 结论

该方法通过注意力评分的计算、复制和应用增强了主要数据的表示，实现了成功的诊断。该模型不仅能够准确识别心脏肥大，还展示了该方法在临床上的重要意义。

### 翻译

医疗行业已被新型成像技术显著改变，不仅在心血管疾病的诊断方面，也在结构性异常（如心脏肥大）的可视化方面。本文解释了一种集成方法，使用深度学习工具和注意力机制，通过X光图像自动检测心脏肥大。项目的启动基于强大的数据收集阶段和收集各种类型标注X光图像的数据。然后，在预处理模块优化图像质量的同时，可以在所提出的系统中充分利用数据质量。在我们提出的系统中，过程是一个CNN配置，利用Inception V3模型作为关键模块之一。此外，我们还采用多层注意力机制来增强性能。该方法最重要的特点是多头注意力机制，可以自动学习特征。通过精确地仅关注输入的某些区域，模型可以以敏感的方式识别心脏肥大。注意力评分被计算、复制并应用，以增强主要数据的表示，因此实现了成功的诊断。评估阶段将极为严格，并将根据准确度和精确度等措施彻底评估模型。这将验证模型能够识别心脏肥大，还将展示该方法的临床意义。模型的准确度为95.6，精确度为95.2，召回率为96.2，敏感性为95.7，特异性为96.1，曲线下面积为96.0，并绘制了相应的图表用于可视化。


### 论文摘要

The healthcare industry has been revolutionized significantly by novel imaging technologies, not just in the diagnosis of cardiovascular diseases but also by the visualization of structural abnormalities like cardiomegaly. This article explains an integrated approach to the use of deep learning tools and attention mechanisms for automatic detection of cardiomegaly using X-ray images. The initiation of the project is grounded on a strong Data Collection phase and gathering the data of annotated X-ray images of various types. Then, while the Preprocessing module fine-tunes image quality, it is feasible to utilize the best out of the data quality in the proposed system. In our proposed system, the process is a CNN configuration leveraging the inception V3 model as one of the key blocks. Besides, we also employ a multilayer attention mechanism to enhance the strength. The most important feature of the method is the multi-head attention mechanism that can learn features automatically. By exact selective focusing on only some regions of input, the model can thus identify cardiomegaly in a sensitive manner. Attention rating is calculated, duplicated, and applied to enhance representation of main data, and therefore there is a successful diagnosis. The Evaluation stage will be extremely strict and it will thoroughly evaluate the model based on such measures as accuracy and precision. This will validate that the model can identify cardiomegaly and will also show the clinical significance of this method. The model has accuracy of 95.6, precision of 95.2, recall of 96.2, sensitivity of 95.7, specificity of 96.1 and an Area Under Curve(AUC) of 96.0 and their respective graphs are plotted for visualisation.

---

## 236. Learning Procedural-aware Video Representations through State-Grounded Hierarchy Unfolding

**论文链接:** [http://arxiv.org/abs/2511.20073v1](http://arxiv.org/abs/2511.20073v1)

**作者:** Jinghan Zhao, Yifei Huang, Feng Lu

**发布时间:** 2025-11-25

**备注:** Accepted by AAAI 2026. 15 pages, 12 figures

### GPT解析

### 总结

本文提出了Task-Step-State (TSS)框架，通过引入'状态'作为视觉基础的语义层，将抽象程序锚定到可观察的视觉细节上，并采用渐进式预训练策略来提升视频表示学习效果。

### 背景

学习程序感知的视频表示是构建能够推理和执行复杂任务的智能体的关键步骤。现有方法通常通过将视觉内容与任务和步骤级别的文本描述对齐来为视频表示注入程序语义。

### 目的

解决现有方法中'任务'和'步骤'描述的高度抽象性与视觉数据中具体、可观察细节之间难以稳健对齐的问题。

### 方法

提出Task-Step-State (TSS)框架，将任务视为通过步骤实现，这些步骤驱动可观察状态之间的转换。采用渐进式预训练策略展开TSS层次结构，迫使模型将表示锚定在状态中，同时关联步骤和高级任务。

### 主要发现

在COIN和CrossTask数据集上的实验表明，该方法在任务识别、步骤识别和下一步预测等多个下游任务上优于基线模型。消融研究显示，状态监督是性能提升的关键因素，渐进式预训练比标准联合训练更有效。

### 结论

引入状态监督作为视觉基础的语义层，并采用渐进式预训练策略，能够显著提升程序感知的视频表示学习效果，使模型能够更好地理解和执行复杂任务。

### 翻译

学习程序感知的视频表示是构建能够推理和执行复杂任务的智能体的关键步骤。现有方法通常通过将视觉内容与任务和步骤级别的文本描述对齐来为视频表示注入程序语义。然而，由于'任务'和'步骤'描述的高度抽象性，它们无法与视觉数据中具体、可观察的细节形成稳健的对齐。为此，我们引入'状态'，即对象配置的文本快照，作为视觉基础的语义层，将抽象程序锚定到模型实际可以看到的内容上。我们在新颖的Task-Step-State (TSS)框架中形式化这一见解，其中任务通过步骤实现，这些步骤驱动可观察状态之间的转换。为了强制执行这种结构，我们提出了一种渐进式预训练策略，该策略展开TSS层次结构，迫使模型将表示锚定在状态中，同时将它们与步骤和高级任务相关联。在COIN和CrossTask数据集上的大量实验表明，我们的方法在多个下游任务上优于基线模型，包括任务识别、步骤识别和下一步预测。消融研究表明，引入状态监督是所有任务性能提升的关键驱动因素。此外，我们的渐进式预训练策略比标准联合训练更有效，因为它更好地强制执行了预期的层次结构。


### 论文摘要

Learning procedural-aware video representations is a key step towards building agents that can reason about and execute complex tasks. Existing methods typically address this problem by aligning visual content with textual descriptions at the task and step levels to inject procedural semantics into video representations. However, due to their high level of abstraction, 'task' and 'step' descriptions fail to form a robust alignment with the concrete, observable details in visual data. To address this, we introduce 'states', i.e., textual snapshots of object configurations, as a visually-grounded semantic layer that anchors abstract procedures to what a model can actually see. We formalize this insight in a novel Task-Step-State (TSS) framework, where tasks are achieved via steps that drive transitions between observable states. To enforce this structure, we propose a progressive pre-training strategy that unfolds the TSS hierarchy, forcing the model to ground representations in states while associating them with steps and high-level tasks. Extensive experiments on the COIN and CrossTask datasets show that our method outperforms baseline models on multiple downstream tasks, including task recognition, step recognition, and next step prediction. Ablation studies show that introducing state supervision is a key driver of performance gains across all tasks. Additionally, our progressive pretraining strategy proves more effective than standard joint training, as it better enforces the intended hierarchical structure.

---

## 237. Online-PVLM: Advancing Personalized VLMs with Online Concept Learning

**论文链接:** [http://arxiv.org/abs/2511.20056v1](http://arxiv.org/abs/2511.20056v1)

**作者:** Huiyu Bai, Runze Wang, Zhuoyun Du, Yiyang Zhao, Fengji Zhang, Haoyu Chen, Xiaoyong Zhu, Bo Zheng, Xuejiao Zhao

**发布时间:** 2025-11-25

**备注:** Work in Progress

### GPT解析

### 总结

本文提出了一种名为Online-PVLM的在线概念学习框架，利用双曲表示实现个性化视觉语言模型的高效扩展，并开发了OP-Eval基准进行评估。

### 背景

个性化视觉语言模型(VLMs)在用户特定概念对齐交互方面展现出强大能力，但现有方法需要为每个新概念学习单独的嵌入，无法支持测试时的实时适应。

### 目的

解决现有个性化VLM方法无法支持测试时实时适应的问题，特别是在大规模场景中概念嵌入的高效检索问题。

### 方法

提出Online-PVLM框架，利用双曲表示进行在线概念学习，采用无需训练的范式在测试时生成概念嵌入，使个性化VLM的使用更具可扩展性和效率。

### 主要发现

开发了OP-Eval基准，包含1,292个概念和超过3万个高质量实例，具有多样化问题类型，可严格评估真实场景中的在线概念学习能力。

### 结论

大量实验证明了所提出框架的最先进性能，作者将提供源代码和数据集。

### 翻译

个性化视觉语言模型(VLMs)因其在对齐用户特定概念交互方面的强大能力而日益受到关注。现有方法通常需要为每个新概念学习单独的嵌入，这无法支持测试过程中的实时适应。这一限制在大规模场景中尤为明显，因为概念嵌入的高效检索无法实现。为弥补这一差距，我们提出了Online-PVLM，一个利用双曲表示进行在线概念学习的框架。我们的方法采用了一种无需训练的范式来生成测试时的概念嵌入，使个性化VLM的使用既可扩展又高效。此外，我们开发了OP-Eval，一个包含1,292个概念和超过3万个高质量实例的综合大规模基准，具有多样化的问题类型，旨在严格评估真实场景中的在线概念学习能力。大量实验证明了我们提出框架的最先进性能。我们的源代码和数据集将可供使用。


### 论文摘要

Personalized Visual Language Models (VLMs) are gaining increasing attention for their formidable ability in user-specific concepts aligned interactions (e.g., identifying a user's bike). Existing methods typically require the learning of separate embeddings for each new concept, which fails to support real-time adaptation during testing. This limitation becomes particularly pronounced in large-scale scenarios, where efficient retrieval of concept embeddings is not achievable. To alleviate this gap, we propose Online-PVLM, a framework for online concept learning by leveraging hyperbolic representations. Our approach makes a train-free paradigm for concept embeddings generation at test time, making the use of personalized VLMs both scalable and efficient. In addition, we develop OP-Eval, a comprehensive and large-scale benchmark comprising 1,292 concepts and over 30K high-quality instances with diverse question types, designed to rigorously assess online concept learning in realistic scenarios. Extensive experiments demonstrate the state-of-the-art performance of our proposed framework. Our source code and dataset will be made available.

---

## 238. iRadioDiff: Physics-Informed Diffusion Model for Indoor Radio Map Construction and Localization

**论文链接:** [http://arxiv.org/abs/2511.20015v1](http://arxiv.org/abs/2511.20015v1)

**作者:** Xiucheng Wang, Tingwei Yuan, Yang Cao, Nan Cheng, Ruijin Sun, Weihua Zhuang

**发布时间:** 2025-11-25

### GPT解析

### 总结

本文提出了一种名为iRadioDiff的基于扩散的框架，用于构建无需采样的室内无线电地图，能够准确建模非平稳场不连续性并高效构建物理一致的无线电地图。

### 背景

无线电地图作为环境感知的电磁表示，连接场景几何和材料属性与信号强度空间分布，实现无需昂贵现场测量的定位。然而，构建高保真室内无线电地图面临挑战：电磁求解器延迟过高，基于学习的方法依赖稀疏测量或均匀材料假设，与室内环境异质性和多径丰富的特性不符。

### 目的

克服现有方法在构建室内无线电地图时的局限性，开发能够准确反映室内环境复杂特性的无线电地图构建方法。

### 方法

提出iRadioDiff，一种无需采样的基于扩散的框架。该框架基于接入点位置进行条件化，通过材料反射和传输系数编码的物理信息提示引导，并集成多径关键先验（包括衍射点、强传输边界和视距轮廓），通过条件通道和边界加权目标引导生成过程。

### 主要发现

实验证明iRadioDiff在室内无线电地图构建和基于接收信号强度的室内定位方面达到最先进性能，并能有效泛化到不同布局和材料配置。

### 结论

iRadioDiff通过结合物理信息和多径关键先验，成功解决了室内无线电地图构建中的挑战，实现了准确建模和高效构建的目标，代码已公开可用。

### 翻译

无线电地图作为环境感知的电磁表示，将场景几何和材料属性与信号强度的空间分布联系起来，使得无需昂贵的现场测量即可实现定位。然而，构建高保真室内无线电地图仍然具有挑战性，因为电磁求解器的延迟过高，而基于学习的方法往往依赖于稀疏测量或均匀材料的假设，这与室内环境异质性和多径丰富的特性不符。为了克服这些挑战，我们提出了iRadioDiff，一种用于室内无线电地图构建的无需采样的基于扩散的框架。iRadioDiff基于接入点位置进行条件化，并通过材料反射和传输系数编码的物理信息提示引导。它还集成了多径关键先验，包括衍射点、强传输边界和视距轮廓，通过条件通道和边界加权目标引导生成过程。这种设计能够准确建模非平稳场的不连续性并高效构建物理一致的无线电地图。实验证明，iRadioDiff在室内无线电地图构建和基于接收信号强度的室内定位方面达到了最先进的性能，能够有效泛化到不同的布局和材料配置。代码可在https://github.com/UNIC-Lab/iRadioDiff获取。


### 论文摘要

Radio maps (RMs) serve as environment-aware electromagnetic (EM) representations that connect scenario geometry and material properties to the spatial distribution of signal strength, enabling localization without costly in-situ measurements. However, constructing high-fidelity indoor RMs remains challenging due to the prohibitive latency of EM solvers and the limitations of learning-based methods, which often rely on sparse measurements or assumptions of homogeneous material, which are misaligned with the heterogeneous and multipath-rich nature of indoor environments. To overcome these challenges, we propose iRadioDiff, a sampling-free diffusion-based framework for indoor RM construction. iRadioDiff is conditioned on access point (AP) positions, and physics-informed prompt encoded by material reflection and transmission coefficients. It further incorporates multipath-critical priors, including diffraction points, strong transmission boundaries, and line-of-sight (LoS) contours, to guide the generative process via conditional channels and boundary-weighted objectives. This design enables accurate modeling of nonstationary field discontinuities and efficient construction of physically consistent RMs. Experiments demonstrate that iRadioDiff achieves state-of-the-art performance in indoor RM construction and received signal strength based indoor localization, which offers effective generalization across layouts and material configurations. Code is available at https://github.com/UNIC-Lab/iRadioDiff.

---

## 239. REWA: Witness-Overlap Theory -- Foundations for Composable Binary Similarity Systems

**论文链接:** [http://arxiv.org/abs/2511.19998v1](http://arxiv.org/abs/2511.19998v1)

**作者:** Nikit Phadke

**发布时间:** 2025-11-25

### GPT解析

### 总结

REWA提出了一种基于见证重叠结构的相似性通用理论，展示了概念间相似性可简化为具有排名保留保证的紧凑编码，并提供了模块化构建相似性系统的方法。

### 背景

现有相似性计算方法缺乏统一理论基础，需要一种能够处理多种数据类型和结构的通用相似性理论。

### 目的

开发一种通用相似性理论，能够将不同类型的相似性计算简化为紧凑编码，同时保留排名信息，并提供模块化设计框架。

### 方法

REWA系统包含有限见证集、半随机比特分配和期望相似性单调性三个核心要素，通过重叠间隙条件实现排名保留，支持多种变换的组合。

### 主要发现

1. 相似性可表示为单调见证重叠时，可简化为具有排名保留保证的紧凑编码
2. 在重叠间隙条件下，使用O(log(|V|/δ))位即可保留top-k排名
3. 见证集公式是组合式的，支持多种变换的序列组合
4. 该理论适用于最终见证重叠，使相似性系统可以从可重用原语模块化构建
5. 数百万可组合的相似性定义继承了对数编码复杂度
6. REWA统一了Bloom过滤器、minhash等多种现有方法作为特例

### 结论

REWA为相似性系统提供了理论基础，其行为由见证重叠而非哈希函数工程决定，创造了巨大的设计空间，并支持可扩展的模块化构建方法。

### 翻译

REWA引入了一种基于见证重叠结构的相似性通用理论。我们表明，当概念之间的相似性可以表示为单调见证重叠时——无论其源于图邻域、因果关系、时间结构、拓扑特征、符号模式还是基于嵌入的邻域——它可以简化为具有可证明排名保留保证的紧凑编码。REWA系统包括：(1)有限见证集W(v)，(2)从每个生成的见证生成的半随机比特分配，以及(3)期望相似性在重叠Δ(u,v)=|W(u)∩W(v)|中的单调性。我们证明，在最终见证集上的重叠间隙条件下——无论它们是如何构建的——使用m=O(log(|V|/δ))位可以保留top-k排名。见证集公式是组合式的：任何结构、时间、因果、拓扑、信息论或学习变换的序列可以组合成终止于离散见证集的管道。该理论适用于最终见证重叠，使相似性系统可以从可重用原语模块化构建。这产生了巨大的设计空间：数百万可组合的相似性定义继承了对数编码复杂度。REWA包含并统一了Bloom过滤器、minhash、LSH位图、随机投影、草图和分层过滤器作为特例。它为相似性系统提供了理论基础，其行为由见证重叠而非哈希函数工程决定。本文提出了公理、主要可归约性定理、带有明确常数的完整证明，以及关于组合设计、局限性和未来扩展的详细讨论，包括多比特编码、加权见证和非集合表示。


### 论文摘要

REWA introduces a general theory of similarity based on witness-overlap structures. We show that whenever similarity between concepts can be expressed as monotone witness overlap -- whether arising from graph neighborhoods, causal relations, temporal structure, topological features, symbolic patterns, or embedding-based neighborhoods -- it admits a reduction to compact encodings with provable ranking preservation guarantees. REWA systems consist of: (1) finite witness sets $W(v)$, (2) semi-random bit assignments generated from each witness, and (3) monotonicity of expected similarity in the overlap $Δ(u, v) = |W(u) \cap W(v)|$. We prove that under an overlap-gap condition on the final witness sets -- independent of how they were constructed -- top-$k$ rankings are preserved using $m = O(\log(|V|/δ))$ bits. The witness-set formulation is compositional: any sequence of structural, temporal, causal, topological, information-theoretic, or learned transformations can be combined into pipelines that terminate in discrete witness sets. The theory applies to the final witness overlap, enabling modular construction of similarity systems from reusable primitives. This yields a vast design space: millions of composable similarity definitions inherit logarithmic encoding complexity. REWA subsumes and unifies Bloom filters, minhash, LSH bitmaps, random projections, sketches, and hierarchical filters as special cases. It provides a principled foundation for similarity systems whose behavior is governed by witness overlap rather than hash-function engineering. This manuscript presents the axioms, the main reducibility theorem, complete proofs with explicit constants, and a detailed discussion of compositional design, limitations, and future extensions including multi-bit encodings, weighted witnesses, and non-set representations.

---

## 240. $\text{R}^2\text{R}$: A Route-to-Rerank Post-Training Framework for Multi-Domain Decoder-Only Rerankers

**论文链接:** [http://arxiv.org/abs/2511.19987v1](http://arxiv.org/abs/2511.19987v1)

**作者:** Xinyu Wang, Hanwei Wu, Qingchen Hu, Zhenghan Tai, Jingrui Tian, Lei Ding, Jijun Chi, Hailin He, Tung Sum Thomas Kwok, Yufei Cui, Sicheng Lyu, Muzhi Li, Mingze Li, Xinyue Yu, Ling Zhou, Peng Lu

**发布时间:** 2025-11-25

**备注:** 13 pages, including 3 figures and 3 tables

### GPT解析

### 总结

R2R是一种领域感知框架，结合动态专家路由和两阶段训练策略EAG，解决了检索增强生成中仅解码器重排序器在特定领域应用中的问题，通过避免表面形式过拟合和灾难性遗忘，提高了跨领域鲁棒性。

### 背景

通用模型在高风险领域如金融和法律中缺乏领域特定细微差别，而简单的微调会导致表面形式过拟合和灾难性遗忘。仅解码器重排序器是检索增强生成的核心组件。

### 目的

开发一种能够捕捉领域特定细微差别而不过拟合或遗忘先前知识的重排序器框架，提高跨领域性能和鲁棒性。

### 方法

R2R框架采用两阶段训练策略EAG，通过掩码最可预测的表面线索引入反快捷机制，强制重排序器学习领域不变的相关性模式而非记忆数据集特定实体。同时使用轻量级潜在语义路由器从冻结的主干解码器内部表示中查询，为每个查询选择最优的LoRA专家。

### 主要发现

在不同重排序器主干和多样化领域(法律、医疗和金融)的广泛实验中，R2R持续超越通用模型和单领域微调基线，证实了其作为模型无关和模块化领域专业化方法的有效性，具有强大的跨领域鲁棒性。

### 结论

R2R是一种模型无关和模块化的领域专业化方法，能够有效解决特定领域应用中的表面形式过拟合和灾难性遗忘问题，同时保持强大的跨领域鲁棒性。

### 翻译

仅解码器重排序器是检索增强生成的核心组件。然而，通用模型在金融和法律等高风险领域缺乏领域特定的细微差别，而简单的微调会导致表面形式过拟合和灾难性遗忘。为了应对这一挑战，我们引入了R2R，这是一种领域感知框架，结合了动态专家路由和两阶段训练策略——实体抽象泛化(EAG)。EAG通过掩码最可预测的表面线索引入反快捷机制，强制重排序器学习领域不变的相关性模式，而非记忆数据集特定实体。为了高效激活领域专家，R2R采用轻量级潜在语义路由器，从冻结的主干解码器内部表示中探测，为每个查询选择最优的LoRA专家。在不同重排序器主干和多样化领域(法律、医疗和金融)的广泛实验中，R2R持续超越通用模型和单领域微调基线。我们的结果证实，R2R是一种模型无关和模块化的领域专业化方法，具有强大的跨领域鲁棒性。


### 论文摘要

Decoder-only rerankers are central to Retrieval-Augmented Generation (RAG). However, generalist models miss domain-specific nuances in high-stakes fields like finance and law, and naive fine-tuning causes surface-form overfitting and catastrophic forgetting. To address this challenge, we introduce R2R, a domain-aware framework that combines dynamic expert routing with a two-stage training strategy, Entity Abstraction for Generalization (EAG). EAG introduces a counter-shortcut mechanism by masking the most predictive surface cues, forcing the reranker to learn domain-invariant relevance patterns rather than memorizing dataset-specific entities. To efficiently activate domain experts, R2R employs a lightweight Latent Semantic Router that probes internal representations from the frozen backbone decoder to select the optimal LoRA expert per query. Extensive experiments across different reranker backbones and diverse domains (legal, medical, and financial) demonstrate that R2R consistently surpasses generalist and single-domain fine-tuned baselines. Our results confirm that R2R is a model-agnostic and modular approach to domain specialization with strong cross-domain robustness.

---

## 241. STAvatar: Soft Binding and Temporal Density Control for Monocular 3D Head Avatars Reconstruction

**论文链接:** [http://arxiv.org/abs/2511.19854v1](http://arxiv.org/abs/2511.19854v1)

**作者:** Jiankuo Zhao, Xiangyu Zhu, Zidu Wang, Zhen Lei

**发布时间:** 2025-11-25

**备注:** 17 pages, 14 figures

### GPT解析

### 总结

STAvatar是一种从单目视频中重建高保真和可动画的3D头部头像的新方法，解决了现有方法在处理变形和遮挡区域时的局限性。

### 背景

从单目视频中重建高保真和可动画的3D头部头像仍然是一个具有挑战性但至关重要的任务。

### 目的

解决现有基于3D高斯散射的方法在处理变形和频繁遮挡区域时的局限性，提高重建性能。

### 方法

STAvatar包含两个关键组件：(1) UV自适应软绑定框架，利用基于图像和几何的先验知识在UV空间内学习每个高斯特征偏移；(2) 时间ADC策略，通过聚类结构相似的帧和引入融合感知误差作为克隆标准来优化密度控制。

### 主要发现

在四个基准数据集上的实验表明，STAvatar达到了最先进的重建性能，特别是在捕获细粒度细节和重建频繁遮挡区域方面表现优异。

### 结论

STAvatar通过创新的UV自适应软绑定框架和时间ADC策略，显著提高了3D头部头像的重建质量，代码将公开可用。

### 翻译

从单目视频中重建高保真和可动画的3D头部头像仍然是一个具有挑战性但至关重要的任务。现有的基于3D高斯散射的方法通常将高斯绑定到网格三角形，并通过线性混合蒙皮仅建模变形，导致运动僵硬和表现力有限。此外，它们缺乏处理频繁遮挡区域（如口腔内部、眼睑）的专门策略。为了解决这些局限性，我们提出了STAvatar，它包含两个关键组件：(1) UV自适应软绑定框架，利用基于图像和几何的先验知识在UV空间内学习每个高斯特征偏移。这种UV表示支持动态重采样，确保与自适应密度控制完全兼容，并增强对形状和纹理变化的适应性。(2) 时间ADC策略，首先聚类结构相似的帧，以便更有针对性地计算密度标准。它进一步引入了一种新颖的融合感知误差作为克隆标准，以同时捕获几何和纹理差异，鼓励在需要更精细细节的区域进行密度增加。在四个基准数据集上的大量实验表明，STAvatar达到了最先进的重建性能，特别是在捕获细粒度细节和重建频繁遮挡区域方面。代码将公开可用。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决从单目视频中重建高保真度和可动画的3D头部头像的问题。现有方法基于3D高斯散射通常将高斯硬绑定到网格三角形上，导致刚体运动和有限的表情表达能力，且缺乏处理频繁遮挡区域(如嘴内部、眼睑)的策略。这个问题很重要，因为高质量3D头部头像在AR/VR、远程呈现、数字人和交互媒体中有广泛应用需求，而传统多相机系统昂贵复杂，单目方法可使用普通相机实现更广泛的应用。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者通过分析现有方法的局限性进行设计：硬绑定导致高斯在三角形内相对静态，无法捕捉细粒度变形；标准3DGS的自适应密度控制针对静态场景，无法处理动态头像中的遮挡区域。作者借鉴了3D高斯散射、FLAME模型、线性混合蒙皮和自适应密度控制等现有工作，但针对其局限性提出了UV自适应软绑定框架和时间自适应密度控制策略，前者解决变形能力问题，后者解决遮挡区域处理问题。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是通过UV空间中的软绑定实现更灵活的变形建模，同时通过时间聚类和新的误差标准改进密度控制。整体流程：1)基于FLAME模型初始化高斯并绑定到网格；2)使用LBS进行粗略变形；3)通过UV自适应软绑定框架在UV空间中学习特征偏移，实现软绑定；4)采用时间自适应密度控制，聚类相似帧并使用融合感知误差作为克隆标准；5)通过RGB损失和正则化损失进行训练优化；6)最终渲染高质量3D头像。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)UV-Adaptive Soft Binding框架，支持在UV空间中学习特征偏移，实现软绑定并与ADC兼容；2)Temporal ADC策略，通过FTC聚类相似帧和FPE-AP标准联合考虑几何和纹理差异。相比之前工作，STAvatar解决了硬绑定导致的刚体运动问题，支持动态密度控制而非固定数量高斯，专门处理了动态场景中的遮挡区域，并联合考虑了几何和纹理信息而非仅关注几何。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'STAvatar通过UV自适应软绑定和时间自适应密度控制，实现了从单目视频中重建高质量、高细节、可动画的3D头部头像，特别是在处理频繁遮挡区域和细粒度变形方面取得了显著进展。'}


### 论文摘要

Reconstructing high-fidelity and animatable 3D head avatars from monocular videos remains a challenging yet essential task. Existing methods based on 3D Gaussian Splatting typically bind Gaussians to mesh triangles and model deformations solely via Linear Blend Skinning, which results in rigid motion and limited expressiveness. Moreover, they lack specialized strategies to handle frequently occluded regions (e.g., mouth interiors, eyelids). To address these limitations, we propose STAvatar, which consists of two key components: (1) a UV-Adaptive Soft Binding framework that leverages both image-based and geometric priors to learn per-Gaussian feature offsets within the UV space. This UV representation supports dynamic resampling, ensuring full compatibility with Adaptive Density Control (ADC) and enhanced adaptability to shape and textural variations. (2) a Temporal ADC strategy, which first clusters structurally similar frames to facilitate more targeted computation of the densification criterion. It further introduces a novel fused perceptual error as clone criterion to jointly capture geometric and textural discrepancies, encouraging densification in regions requiring finer details. Extensive experiments on four benchmark datasets demonstrate that STAvatar achieves state-of-the-art reconstruction performance, especially in capturing fine-grained details and reconstructing frequently occluded regions. The code will be publicly available.

---

## 242. Exploring Urban Air Mobility Adoption Potential in San Francisco Bay Area Region A Systems of Systems Level Case Study on Passenger Waiting Times and Travel Efficiency

**论文链接:** [http://arxiv.org/abs/2511.20603v1](http://arxiv.org/abs/2511.20603v1)

**作者:** Winfrey Paul Sagayam Dennis

**发布时间:** 2025-11-25

### GPT解析

### 总结

该研究评估了城市空中交通(UAM)在旧金山湾区的可行性和系统采用潜力，通过比较UAM与传统地面交通的旅行时间，发现UAM可显著减少区域内的总旅行时间。

### 背景

随着电动垂直起降(eVTOL)车辆的最新发展，城市空中交通(UAM)获得了动力，这些车辆提供点对点的空中出租车服务，可能有助于缓解长期负担过重的城市交通拥堵问题。

### 目的

评估UAM在旧金山湾区运营的可行性和系统层面的采用潜力，通过比较主要区域节点(旧金山、奥克兰、圣何塞和帕洛阿尔托机场)的乘客出发、等待、旅行和到达时间，与传统地面交通进行比较。

### 方法

在MATLAB中开发了多代理仿真来评估舰队运营，并使用泊松过程在随机乘客流量和周转约束下模拟需求到达情况。

### 主要发现

在需求高峰期使用UAM可以减少区域内高达80%的总旅行时间，并确定了舰队调度优化的关键运营因素。

### 结论

舰队规模、乘客请求量和周转时间直接影响等待时间、运营成本和整体用户接受度，这些因素对于UAM的成功运营至关重要。

### 翻译

随着电动垂直起降(eVTOL)车辆的最新发展，城市空中交通(UAM)获得了动力，提供点对点的空中出租车服务，可能有助于缓解长期负担过重的城市交通拥堵。该研究通过比较主要区域节点(包括旧金山、奥克兰、圣何塞和帕洛阿尔托机场)的乘客出发、等待、旅行和到达时间，与传统地面交通进行比较，评估了UAM在旧金山湾区运营的可行性和系统层面的采用潜力。在MATLAB中开发了多代理仿真来评估舰队运营，并使用泊松过程在随机乘客流量和周转约束下模拟需求到达。结果表明，在需求高峰期使用UAM可以减少区域内高达80%的总旅行时间。本文的研究结果强调了舰队调度优化的关键运营因素，特别是舰队规模、乘客请求量和周转时间如何直接影响等待时间、运营成本和整体用户接受度。


### 论文摘要

Urban Air mobility has gained momentum with recent advancements in the electric vertical take-off and landing (eVTOL) vehicles, offering faster point-to-point air taxi services that could help relieve traffic congestion in chronically overburdened cities. The research assesses the feasibility and systems-of-systems level adoption potential of UAM operations in the San Francisco Bay Area by comparing passenger departure, waiting, travel, and arrival times across key regional nodes, including San Francisco, Oakland, San Jose, and Palo Alto airports, with conventional ground transportation. A multi-agent simulation was developed in MATLAB to evaluate the fleet operations and to model demand arrival using a Poisson process under stochastic passenger flows and turnaround constraints. Results indicate that utilizing UAM during peak demand could reduce total travel times up to eighty percent across the region. The findings of this paper highlight the critical operational factors for fleet schedule optimization. Especially how the fleet size, passengers' request volumes, and turnaround time directly influence waiting time, operating cost, and overall user acceptance.

---

## 243. Digital Twin-Assisted High-Precision Massive MIMO Localization in Urban Canyons

**论文链接:** [http://arxiv.org/abs/2511.20453v1](http://arxiv.org/abs/2511.20453v1)

**作者:** Ziqin Zhou, Hui Chen, Gerhard Steinböck, Henk Wymeersch

**发布时间:** 2025-11-25

**备注:** 6 pages, 5 figures. accepted to 2026 IEEE JC&S

### GPT解析

### 总结

提出一种结合数字孪生模型和随机样本一致性算法的三阶段稳健方法，用于解决城市峡谷中的高精度无线定位问题

### 背景

高精度无线定位在城市峡谷中面临噪声测量和非视距传播的严重挑战

### 目的

克服城市峡谷中无线定位的噪声测量和非视距传播问题

### 方法

利用数字孪生进行几何路径关联，采用RANSAC算法识别可靠的视距和单次反弹非视距路径同时拒绝多次反弹异常值，最后对内点集进行优化估计用户位置和时钟偏差

### 主要发现

通过数字孪生将非视距路径转化为有价值的几何信息，实现准确定位，减少对直接视距的依赖，显著降低系统部署成本

### 结论

该方法适合实际部署

### 翻译

城市峡谷中的高精度无线定位受到噪声测量和严重非视距传播的挑战。本文提出了一种稳健的三阶段算法，结合数字孪生模型与随机样本一致性算法，以克服这些局限性。该方法利用数字孪生进行几何路径关联，并采用RANSAC算法识别可靠的视距和单次反弹非视距路径，同时拒绝多次反弹的异常值。对所得内点集的最终优化估计了用户的位置和时钟偏差。模拟验证表明，通过数字孪生将非视距路径有效地转化为有价值的几何信息，该方法能够实现准确定位，减少对直接视距的依赖，并显著降低系统部署成本，使其适合实际部署。


### 论文摘要

High-precision wireless localization in urban canyons is challenged by noisy measurements and severe non-line-of-sight (NLOS) propagation. This paper proposes a robust three-stage algorithm synergizing a digital twin (DT) model with the random sample consensus (RANSAC) algorithm to overcome these limitations. The method leverages the DT for geometric path association and employs RANSAC to identify reliable line-of-sight (LOS) and single-bounce NLOS paths while rejecting multi-bounce outliers. A final optimization on the resulting inlier set estimates the user's position and clock bias. Simulations validate that by effectively turning NLOS paths into valuable geometric information via the DT, the approach enables accurate localization, reduces reliance on direct LOS, and significantly lowers system deployment costs, making it suitable for practical deployment.

---

## 244. Multi-Agent gatekeeper: Safe Flight Planning and Formation Control for Urban Air Mobility

**论文链接:** [http://arxiv.org/abs/2511.19691v1](http://arxiv.org/abs/2511.19691v1)

**作者:** Thomas Marshall Vielmetti, Devansh R Agrawal, Dimitra Panagou

**发布时间:** 2025-11-24

**备注:** 13 pages, 4 figures, to appear AIAA SciTech 2026

### GPT解析

### 总结

本文提出了一种名为Multi-Agent gatekeeper的框架，用于在杂乱的3D环境中为领导者-跟随者编队控制提供可证明的安全保证。

### 背景

现有方法面临权衡问题：在线规划器和控制器缺乏正式的安全保证，而离线规划器缺乏对代理数量或期望编队变化的适应性。

### 目的

解决现有方法的局限性，提供一种既有正式安全保证又具有适应性的方法，确保在复杂3D环境中的安全编队控制。

### 方法

提出混合架构，其中单个领导者跟踪预计算的安全轨迹，作为所有跟随代理的共享轨迹备份集。跟随者执行名义编队保持跟踪控制器，并通过始终拥有沿着领导者路径的已知安全备份机动来保证安全。

### 主要发现

在模拟的3D城市环境中，该方法在100次随机试验中实现了100%的避碰成功率，显著优于基线CBF和NMPC方法。作者还证明了该方法在四旋翼机团队上的物理可行性。

### 结论

多代理门卫框架成功解决了领导者-跟随者编队控制中的安全保证问题，并在3D环境中验证了其有效性和实用性。

### 翻译

我们提出了多代理门卫框架，该框架在杂乱的3D环境中为领导者-跟随者编队控制提供可证明的安全保证。现有方法面临一种权衡：在线规划器和控制器缺乏正式的安全保证，而离线规划器缺乏对代理数量或期望编队变化的适应性。为解决这一差距，我们提出了一种混合架构，其中单个领导者跟踪预计算的安全轨迹，作为所有跟随代理的共享轨迹备份集。跟随者执行名义编队保持跟踪控制器，并通过始终沿着领导者路径拥有已知安全的备份机动来保证安全。我们正式证明这种方法确保了与静态障碍物和其他代理的碰撞避免。主要贡献包括：(1) 多代理门卫算法，将我们的单代理门卫框架扩展到多代理系统；(2) 用于领导者-跟随者编队控制的可证明安全代理间协调的轨迹备份集；(3) 门卫框架在3D环境中的首次应用。我们在模拟的3D城市环境中展示了我们的方法，该方法在100次随机试验中实现了100%的避碰成功率，显著优于基线CBF和NMPC方法。最后，我们在四旋翼机团队上展示了所得轨迹的物理可行性。


### 论文摘要

We present Multi-Agent gatekeeper, a framework that provides provable safety guarantees for leader-follower formation control in cluttered 3D environments. Existing methods face a trad-off: online planners and controllers lack formal safety guarantees, while offline planners lack adaptability to changes in the number of agents or desired formation. To address this gap, we propose a hybrid architecture where a single leader tracks a pre-computed, safe trajectory, which serves as a shared trajectory backup set for all follower agents. Followers execute a nominal formation-keeping tracking controller, and are guaranteed to remain safe by always possessing a known-safe backup maneuver along the leader's path. We formally prove this method ensures collision avoidance with both static obstacles and other agents. The primary contributions are: (1) the multi-agent gatekeeper algorithm, which extends our single-agent gatekeeper framework to multi-agent systems; (2) the trajectory backup set for provably safe inter-agent coordination for leader-follower formation control; and (3) the first application of the gatekeeper framework in a 3D environment. We demonstrate our approach in a simulated 3D urban environment, where it achieved a 100% collision-avoidance success rate across 100 randomized trials, significantly outperforming baseline CBF and NMPC methods. Finally, we demonstrate the physical feasibility of the resulting trajectories on a team of quadcopters.

---

## 245. Yo'City: Personalized and Boundless 3D Realistic City Scene Generation via Self-Critic Expansion

**论文链接:** [http://arxiv.org/abs/2511.18734v1](http://arxiv.org/abs/2511.18734v1)

**作者:** Keyang Lu, Sifan Zhou, Hongbin Xu, Gang Xu, Zhifei Yang, Yikai Wang, Zhen Xiao, Jieyi Long, Ming Li

**发布时间:** 2025-11-24

**备注:** 22 pages, 16 figures

### GPT解析

### 总结

Yo'City是一个创新的智能体框架，能够实现用户定制化和无限扩展的3D城市生成，通过利用大模型的推理和组合能力，解决了现有方法依赖单一扩散模型的局限性。

### 背景

真实的3D城市生成对于虚拟现实和数字孪生等广泛应用至关重要，但现有方法大多依赖训练单个扩散模型，限制了生成个性化和无限制城市规模场景的能力。

### 目的

开发一个能够生成用户定制化和无限扩展的3D城市的框架，突破现有方法的局限性，实现更高质量的城市场景生成。

### 方法

Yo'City采用自上而下的规划策略定义分层'城市-区域-网格'结构，通过全局规划器确定整体布局和功能区域，本地设计师细化区域描述，使用'生产-细化-评估'等距图像合成循环实现网格级3D生成，并引入用户交互式、关系引导的扩展机制模拟城市连续演化，基于场景图进行距离和语义感知的布局优化。

### 主要发现

大量实验表明，Yo'City在语义、几何、纹理和布局等多个评估维度上都优于现有的最先进方法，能够生成更高质量、更连贯的城市场景。

### 结论

Yo'City框架通过结合大模型的推理能力和创新的分层规划策略，有效解决了3D城市生成中的个性化与扩展性问题，为虚拟现实和数字孪生等应用提供了强大的工具。

### 翻译

真实的3D城市生成对于虚拟现实和数字孪生等广泛应用而言是基础性的。然而，大多数现有方法依赖于训练单个扩散模型，这限制了它们生成个性化和无限城市规模场景的能力。在本文中，我们提出了Yo'City，一个新颖的智能体框架，它利用现成大模型的推理和组合能力，实现用户定制化和无限扩展的3D城市生成。具体而言，Yo'City首先通过自上而下的规划策略构想城市，该策略定义了分层的'城市-区域-网格'结构。全局规划器确定整体布局和潜在功能区域，而本地设计师进一步细化每个区域的详细网格级描述。随后，通过'生产-细化-评估'等距图像合成循环实现网格级3D生成，接着进行图像到3D的生成。为模拟连续的城市演化，Yo'City进一步引入了用户交互式、关系引导的扩展机制，该机制执行基于场景图的距离和语义感知的布局优化，确保空间连贯的城市增长。为全面评估我们的方法，我们构建了一个多样化的基准数据集，并设计了六个多维度指标，从语义、几何、纹理和布局等角度评估生成质量。大量实验表明，Yo'City在所有评估方面都持续优于现有的最先进方法。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决如何生成个性化、无边界的逼真3D城市场景的问题。这个问题在现实中非常重要，因为3D城市模型在虚拟现实、游戏、城市规划、数字孪生等众多应用中扮演着关键角色，而现有方法往往难以生成既符合用户个性化需求又能无限扩展的城市场景，手动构建又极其耗时费力。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': "作者借鉴了现实世界中城市的层次化'城市-区域-网格'结构，并受到大型语言模型和视觉语言模型代理框架的启发。他们分析了现有方法如SynCity的局限性，缺乏明确的规划机制来推理城市结构和空间层次。基于这些思考，作者设计了四个关键模块：全局规划器、本地设计师、3D生成器和扩展模块，采用从粗到细的规划策略，确保空间连贯性和高质量生成。", '这个方法的核心思想是什么？整体实现流程是怎样的？': "核心思想是采用多代理框架，利用大型模型的推理和组合能力实现用户定制化和无限扩展的3D城市生成，并通过层次化结构和自批评机制确保空间连贯性。整体流程分为四个阶段：首先，全局规划器将用户提示转换为城市布局和功能区域；其次，本地设计师细化每个区域的网格级描述；然后，3D生成器通过'产生-精炼-评估'循环生成等距图像并转换为3D模型；最后，扩展模块基于场景图的关系引导优化，确定新区域的最佳位置，实现城市无限扩展。", '论文的关键创新点有哪些？相比之前的工作，有什么不同？': "关键创新点包括：提出Yo'City多代理框架；基于网格的层次化城市结构设计；自上而下的规划策略；基于场景图的关系引导扩展机制；多维评估基准。相比SynCity等现有工作，Yo'City采用并行而非顺序生成所有网格，避免了错误累积；具有明确的规划机制推理城市结构；层次化推理和精炼机制提高了几何细节和纹理质量；能够保持大规模场景的全局一致性，而不仅仅是局部连贯性。", '如果要用一句话总结这篇论文的贡献，你会怎么说？': "Yo'City通过多代理框架和层次化规划策略，实现了基于文本提示的个性化、无边界的逼真3D城市生成，在语义一致性、几何保真度、纹理清晰度、布局连贯性、场景覆盖率和整体真实感方面超越了现有方法。"}


### 论文摘要

Realistic 3D city generation is fundamental to a wide range of applications, including virtual reality and digital twins. However, most existing methods rely on training a single diffusion model, which limits their ability to generate personalized and boundless city-scale scenes. In this paper, we present Yo'City, a novel agentic framework that enables user-customized and infinitely expandable 3D city generation by leveraging the reasoning and compositional capabilities of off-the-shelf large models. Specifically, Yo'City first conceptualize the city through a top-down planning strategy that defines a hierarchical "City-District-Grid" structure. The Global Planner determines the overall layout and potential functional districts, while the Local Designer further refines each district with detailed grid-level descriptions. Subsequently, the grid-level 3D generation is achieved through a "produce-refine-evaluate" isometric image synthesis loop, followed by image-to-3D generation. To simulate continuous city evolution, Yo'City further introduces a user-interactive, relationship-guided expansion mechanism, which performs scene graph-based distance- and semantics-aware layout optimization, ensuring spatially coherent city growth. To comprehensively evaluate our method, we construct a diverse benchmark dataset and design six multi-dimensional metrics that assess generation quality from the perspectives of semantics, geometry, texture, and layout. Extensive experiments demonstrate that Yo'City consistently outperforms existing state-of-the-art methods across all evaluation aspects.

---

## 246. Urban Buildings Energy Consumption Estimation Using HPC: A Case Study of Bologna

**论文链接:** [http://arxiv.org/abs/2511.19463v1](http://arxiv.org/abs/2511.19463v1)

**作者:** Aldo Canfora, Eleonora Bergamaschi, Riccardo Mioli, Federico Battini, Mirko Degli Esposti, Giorgio Pedrazzi, Chiara Dellacasa

**发布时间:** 2025-11-21

**备注:** Preprint submitted for publication

### GPT解析

### 总结

该研究介绍了一个综合EnergyPlus模拟、高性能计算和开放地理空间数据集的城市建筑能源建模流程，成功应用于意大利博洛尼亚市约25,000栋建筑的能源需求估算。

### 背景

城市建筑能源建模在理解和预测城市规模的能源消耗方面发挥着核心作用。

### 目的

开发一个UBEM流程，用于估计意大利博洛尼亚建筑的能源需求。

### 方法

集成EnergyPlus模拟、高性能计算和开放地理空间数据集；从博洛尼亚开放数据门户获取建筑几何信息；使用航空LiDAR测量增强数据；从区域建筑法规和欧洲TABULA数据库推导建筑材料和性能属性；在Leonardo超级计算机上进行计算。

### 主要发现

能够在不到30分钟内模拟约25,000栋建筑，展示了高效处理城市规模数据的能力。

### 结论

该UBEM流程成功应用于博洛尼亚市的大规模建筑能源模拟，为城市能源规划提供了有效工具。

### 翻译

城市建筑能源建模在理解和预测城市规模的能源消耗方面发挥着核心作用。在这项工作中，我们提出了一个UBEM流程，该流程整合了EnergyPlus模拟、高性能计算和开放地理空间数据集，用于估计意大利博洛尼亚建筑的能源需求。包括建筑平面图和高度在内的几何信息从博洛尼亚开放数据门户获取，并使用航空LiDAR测量进行了增强。建筑材料、隔热特性和窗户性能等非几何属性则从区域建筑法规和欧洲TABULA数据库推导得出。计算工作在Cineca托管的Leonardo超级计算机上进行，使研究人员能够在不到30分钟内模拟约25,000栋建筑。


### 论文摘要

Urban Building Energy Modeling (UBEM) plays a central role in understanding and forecasting energy consumption at the city scale. In this work, we present a UBEM pipeline that integrates EnergyPlus simulations, high-performance computing (HPC), and open geospatial datasets to estimate the energy demand of buildings in Bologna, Italy. Geometric information including building footprints and heights was obtained from the Bologna Open Data portal and enhanced with aerial LiDAR measurements. Non-geometric attributes such as construction materials, insulation characteristics, and window performance were derived from regional building regulations and the European TABULA database. The computation was carried out on Leonardo, the Cineca-hosted supercomputer, enabling the simulation of approximately 25,000 buildings in under 30 minutes.

---

## 247. iMontage: Unified, Versatile, Highly Dynamic Many-to-many Image Generation

**论文链接:** [http://arxiv.org/abs/2511.20635v1](http://arxiv.org/abs/2511.20635v1)

**作者:** Zhoujie Fu, Xianfang Zeng, Jinghong Lan, Xinyao Liao, Cheng Chen, Junyi Chen, Jiacheng Wei, Wei Cheng, Shiyu Liu, Yunuo Chen, Gang Yu, Guosheng Lin

**发布时间:** 2025-11-25

### GPT解析

### 总结

本文介绍了iMontage框架，通过将图像数据的多样性注入到视频模型的时间连贯框架中，生成具有自然过渡和广泛动态范围的图像集。

### 背景

预训练视频模型能学习强大先验知识生成高质量时间连贯内容，但其动态特性常受限于训练数据的连续性。

### 目的

引入iMontage框架，将强大的视频模型重新设计为全能图像生成器，统一多种图像生成和编辑任务。

### 方法

提出优雅且最小侵入性的适应策略，配合定制数据整理过程和训练范式，使模型获得广泛图像操作能力而不损害原始运动先验。

### 主要发现

iMontage在多个主流多输入多输出任务中表现出色，保持强跨图像上下文一致性，生成超越常规范围的具有非凡动态的场景。

### 结论

iMontage成功将视频模型重新设计为全能图像生成器，能够生成具有自然过渡和广泛动态范围的图像集。

### 翻译

预训练视频模型学习用于生成高质量、时间连贯内容的强大先验知识。虽然这些模型在时间连贯性方面表现出色，但其动态特性通常受限于训练数据的连续性。我们假设通过将图像数据中丰富且不受约束的内容多样性注入到这种连贯的时间框架中，可以生成具有自然过渡和更广泛动态范围的图像集。为此，我们引入了iMontage，一个统一框架，旨在将强大的视频模型重新设计为全能图像生成器。该框架消费并生成可变长度的图像集，统一了广泛的图像生成和编辑任务。为实现这一点，我们提出了一种优雅且最小侵入性的适应策略，配合定制的数据整理过程和训练范式。这种方法使模型能够获得广泛的图像操作能力，而不会损害其宝贵的原始运动先验知识。iMontage在多个主流的多输入多输出任务中表现出色，不仅保持了强大的跨图像上下文一致性，还生成了具有非凡动态的场景，超越了常规范围。访问我们的主页：https://kr1sjfu.github.io/iMontage-web/。


### 论文摘要

Pre-trained video models learn powerful priors for generating high-quality, temporally coherent content. While these models excel at temporal coherence, their dynamics are often constrained by the continuous nature of their training data. We hypothesize that by injecting the rich and unconstrained content diversity from image data into this coherent temporal framework, we can generate image sets that feature both natural transitions and a far more expansive dynamic range. To this end, we introduce iMontage, a unified framework designed to repurpose a powerful video model into an all-in-one image generator. The framework consumes and produces variable-length image sets, unifying a wide array of image generation and editing tasks. To achieve this, we propose an elegant and minimally invasive adaptation strategy, complemented by a tailored data curation process and training paradigm. This approach allows the model to acquire broad image manipulation capabilities without corrupting its invaluable original motion priors. iMontage excels across several mainstream many-in-many-out tasks, not only maintaining strong cross-image contextual consistency but also generating scenes with extraordinary dynamics that surpass conventional scopes. Find our homepage at: https://kr1sjfu.github.io/iMontage-web/.

---

## 248. A Reason-then-Describe Instruction Interpreter for Controllable Video Generation

**论文链接:** [http://arxiv.org/abs/2511.20563v1](http://arxiv.org/abs/2511.20563v1)

**作者:** Shengqiong Wu, Weicai Ye, Yuanxing Zhang, Jiahao Wang, Quande Liu, Xintao Wang, Pengfei Wan, Kun Gai, Hao Fei, Tat-Seng Chua

**发布时间:** 2025-11-25

**备注:** 27 pages, 13 figures, 13 tables, Project Page: https://sqwu.top/ReaDe/

### GPT解析

### 总结

论文提出了ReaDe，一个通用的、与模型无关的解释器，可将原始用户指令转换为下游视频生成器的精确规范，通过先推理后描述的范式解决意图与输出不匹配的问题。

### 背景

扩散Transformer虽显著提高了视频保真度和时间一致性，但实际可控性有限。简洁、模糊且组成复杂的用户输入与训练中使用的详细提示形成对比，导致意图与输出不匹配。

### 目的

开发一个能够将原始用户指令转换为精确、可操作规范的解释器，使下游视频生成器能够忠实地执行用户意图，实现可控的视频生成。

### 方法

ReaDe采用先推理后描述的范式：首先分析用户请求以识别核心需求和解决歧义，然后生成详细指导。通过两阶段优化训练：(i) 推理增强监督，通过逐步痕迹和密集注释赋予分析解析能力；(ii) 多维奖励分配器实现自然风格描述的稳定、反馈驱动的改进。

### 主要发现

在单条件和多条件场景下的实验表明，ReaDe在指令保真度、描述准确性和下游视频质量方面有一致的提升，并且在推理密集型和未见输入方面展现出强大的泛化能力。

### 结论

ReaDe为使可控视频生成与准确解释的用户意图保持一致提供了实用途径。

### 翻译

扩散Transformer显著提高了视频的保真度和时间一致性，然而，实际可控性仍然有限。简洁、模糊且组成复杂的用户输入与训练中使用的详细提示形成对比，导致意图与输出不匹配。我们提出了ReaDe，一个通用的、与模型无关的解释器，可将原始指令转换为下游视频生成器的精确、可操作规范。ReaDe遵循先推理后描述的范式：它首先分析用户请求以识别核心需求和解决歧义，然后生成详细的指导以实现忠实且可控的生成。我们通过两阶段优化训练ReaDe：(i) 推理增强监督赋予分析解析能力，具有逐步痕迹和密集注释；(ii) 多维奖励分配器实现自然风格描述的稳定、反馈驱动的改进。在单条件和多条件场景下的实验显示，ReaDe在指令保真度、描述准确性和下游视频质量方面有一致的提升，并且在推理密集型和未见输入方面展现出强大的泛化能力。ReaDe为使可控视频生成与准确解释的用户意图保持一致提供了实用途径。项目页面：https://sqwu.top/ReaDe/


### 论文摘要

Diffusion Transformers have significantly improved video fidelity and temporal coherence, however, practical controllability remains limited. Concise, ambiguous, and compositionally complex user inputs contrast with the detailed prompts used in training, yielding an intent-output mismatch. We propose ReaDe, a universal, model-agnostic interpreter that converts raw instructions into precise, actionable specifications for downstream video generators. ReaDe follows a reason-then-describe paradigm: it first analyzes the user request to identify core requirements and resolve ambiguities, then produces detailed guidance that enables faithful, controllable generation. We train ReaDe via a two-stage optimization: (i) reasoning-augmented supervision imparts analytic parsing with stepwise traces and dense captions, and (ii) a multi-dimensional reward assigner enables stable, feedback-driven refinement for natural-style captions. Experiments across single- and multi-condition scenarios show consistent gains in instruction fidelity, caption accuracy, and downstream video quality, with strong generalization to reasoning-intensive and unseen inputs. ReaDe offers a practical route to aligning controllable video generation with accurately interpreted user intent. Project Page: https://sqwu.top/ReaDe/.

---

## 249. PhysChoreo: Physics-Controllable Video Generation with Part-Aware Semantic Grounding

**论文链接:** [http://arxiv.org/abs/2511.20562v1](http://arxiv.org/abs/2511.20562v1)

**作者:** Haoze Zhang, Tianyu Huang, Zichen Wan, Xiaowei Jin, Hongzhi Zhang, Hui Li, Wangmeng Zuo

**发布时间:** 2025-11-25

### GPT解析

### 总结

该研究提出了PhysChoreo框架，一种从单张图像生成具有多样可控性和物理真实感视频的新方法，通过物理属性重建和模拟实现高质量视频合成。

### 背景

现有视频生成模型虽在视觉保真度方面有显著进步，但普遍缺乏明确的物理可控性和合理性。基于物理渲染的指导方法在准确建模复杂物理特性和有效控制长时间序列物理行为方面存在固有挑战。

### 目的

开发一种能够从单张图像生成具有物理真实感和多样可控性的视频的框架，解决现有模型在物理可控性和真实感方面的不足。

### 方法

PhysChoreo框架包含两个阶段：首先通过部件感知的物理属性重建估计图像中所有物体的静态初始物理属性；然后通过时间指令和物理可编辑的模拟，合成具有丰富动态行为和物理真实感的高质量视频。

### 主要发现

实验结果表明，PhysChoreo能够生成具有丰富行为和物理真实感的视频，在多个评估指标上优于现有最先进的方法。

### 结论

PhysChoreo成功解决了视频生成中物理可控性和真实感的挑战，实现了从单张图像生成具有物理真实感的多样化视频内容。

### 翻译

尽管最近的视频生成模型在视觉保真度方面取得了显著进展，但它们通常缺乏明确的物理可控性和合理性。为解决这一问题，一些最近的研究尝试使用基于物理的渲染来指导视频生成。然而，这些方法在准确建模复杂物理特性和有效控制长时间序列的物理行为方面面临固有挑战。在这项工作中，我们介绍了PhysChoreo，一种新颖的框架，可以从单张图像生成具有多样可控性和物理真实感的视频。我们的方法包含两个阶段：首先，它通过部件感知的物理属性重建来估计图像中所有物体的静态初始物理属性。然后，通过时间指令和物理可编辑的模拟，它合成具有丰富动态行为和物理真实感的高质量视频。实验结果表明，PhysChoreo能够生成具有丰富行为和物理真实感的视频，在多个评估指标上优于最先进的方法。


### 论文摘要

While recent video generation models have achieved significant visual fidelity, they often suffer from the lack of explicit physical controllability and plausibility. To address this, some recent studies attempted to guide the video generation with physics-based rendering. However, these methods face inherent challenges in accurately modeling complex physical properties and effectively control ling the resulting physical behavior over extended temporal sequences. In this work, we introduce PhysChoreo, a novel framework that can generate videos with diverse controllability and physical realism from a single image. Our method consists of two stages: first, it estimates the static initial physical properties of all objects in the image through part-aware physical property reconstruction. Then, through temporally instructed and physically editable simulation, it synthesizes high-quality videos with rich dynamic behaviors and physical realism. Experimental results show that PhysChoreo can generate videos with rich behaviors and physical realism, outperforming state-of-the-art methods on multiple evaluation metrics.

---

## 250. Spatio-Temporal Hierarchical Causal Models

**论文链接:** [http://arxiv.org/abs/2511.20558v1](http://arxiv.org/abs/2511.20558v1)

**作者:** Xintong Li, Haoran Zhang, Xiao Zhou

**发布时间:** 2025-11-25

### GPT解析

### 总结

本文提出了一种名为时空分层因果模型（ST-HCMs）的新图形框架，用于解决存在未观察到的单元水平混杂因素情况下的时空因果推断问题。通过时空 collapse 定理，该框架能够恢复标准非分层模型无法处理的因果效应。

### 背景

细粒度的时空数据（如交通传感器网络）为科学发现提供了广阔机会，但从这类观测数据推断因果关系具有挑战性，特别是由于存在未观察到的、特定于单元（如地理位置）却随时间影响结果的混杂因素。现有的大多数时空因果推断方法假设所有混杂因素都被观察到，这一假设在实践中经常被违反。

### 目的

开发一种能够处理未观察到的、时间不变的单元水平混杂因素的时空因果推断方法，以解决现有方法的局限性。

### 方法

作者引入了时空分层因果模型（ST-HCMs），这是一种将分层因果建模扩展到时空领域的新图形框架。该框架的核心是时空 collapse 定理，表明随着子单元数据量的增加，复杂的 ST-HCM 会收敛到更简单的平面因果模型。

### 主要发现

时空 collapse 定理使 ST-HCMs 能够实现通用的因果识别程序，即使存在未观察到的、时间不变的单元水平混杂因素，也能够恢复因果效应，这是标准非分层模型无法做到的。

### 结论

作者在合成和真实数据集上验证了其框架的有效性，展示了其在复杂动态系统中进行稳健因果推断的潜力。

### 翻译

细粒度时空数据（如交通传感器网络）的丰富性为科学发现提供了广阔机会。然而，从这类观测数据推断因果关系仍然具有挑战性，特别是由于存在未观察到的混杂因素，这些因素特定于单元（如地理位置）却随时间影响结果。大多数现有的时空因果推断方法假设所有混杂因素都被观察到，这一假设在实践中经常被违反。在本文中，我们引入了时空分层因果模型（ST-HCMs），这是一种将分层因果建模扩展到时空领域的新图形框架。我们方法的核心是时空 collapse 定理，它表明随着子单元数据量的增加，复杂的 ST-HCM 会收敛到更简单的平面因果模型。这一理论结果使一个通用的因果识别程序成为可能，使 ST-HCMs 能够恢复因果效应，即使存在未观察到的、时间不变的单元水平混杂因素，这是标准非分层模型无法做到的。我们在合成和真实数据集上验证了我们的框架的有效性，展示了其在复杂动态系统中进行稳健因果推断的潜力。


### 论文摘要

The abundance of fine-grained spatio-temporal data, such as traffic sensor networks, offers vast opportunities for scientific discovery. However, inferring causal relationships from such observational data remains challenging, particularly due to unobserved confounders that are specific to units (e.g., geographical locations) yet influence outcomes over time. Most existing methods for spatio-temporal causal inference assume that all confounders are observed, an assumption that is often violated in practice. In this paper, we introduce Spatio-Temporal Hierarchical Causal Models (ST-HCMs), a novel graphical framework that extends hierarchical causal modeling to the spatio-temporal domain. At the core of our approach is the Spatio-Temporal Collapse Theorem, which shows that a complex ST-HCM converges to a simpler flat causal model as the amount of subunit data increases. This theoretical result enables a general procedure for causal identification, allowing ST-HCMs to recover causal effects even in the presence of unobserved, time-invariant unit-level confounders, a scenario where standard non-hierarchical models fail. We validate the effectiveness of our framework on both synthetic and real-world datasets, demonstrating its potential for robust causal inference in complex dynamic systems.

---

## 251. Time-Domain Linear Model-based Framework for Passive Acoustic Mapping of Cavitation Activity

**论文链接:** [http://arxiv.org/abs/2511.20551v1](http://arxiv.org/abs/2511.20551v1)

**作者:** Tatiana Gelvez-Barrera, Barbara Nicolas, Denis Kouamé, Bruno Gilles, Adrian Basarab

**发布时间:** 2025-11-25

### GPT解析

### 总结

该研究提出了一种创新的时域波束形成方法，通过线性模型和时空正则化技术，显著提高了被动声学映射的数据效率和质量，仅使用传统频域方法20%的数据量就能实现增强或具有竞争力的空化图质量。

### 背景

被动声学映射能够对空化活动进行空间映射和时间监测，在治疗性超声波应用中起着关键作用。传统波束形成方法（时域和频域）由于缺少参考发射开始时间，轴向分辨率有限。频域方法需要长信号进行准确估计，而时域方法通常实现较低的空间分辨率。

### 目的

解决传统波束形成方法的局限性，开发一种能够在时域中完全表述的基于线性模型的波束形成框架，提高数据效率。

### 方法

提出一个基于线性模型的波束形成框架，完全在时域中表述。线性前向模型将空化活动的离散时空分布与探头记录的时间信号相关联，明确考虑了由采集几何形状决定的飞行时间延迟。利用时空域中空化活动的先验知识，通过正则化技术对模型进行反演。

### 主要发现

所提出的框架仅使用频域方法通常所需数据的20%，就能实现增强或具有竞争力的空化图质量。这突显了数据效率的显著提高，以及时空正则化对适应各种被动空化场景的灵活性。

### 结论

该方法优于最先进的技术，在数据效率和性能方面都有优势，为被动声学映射提供了新的解决方案。

### 翻译

被动声学映射能够对空化活动进行空间映射和时间监测，在治疗性超声波应用中起着关键作用。大多数传统波束形成方法，无论是在时域还是频域实现，由于缺少参考发射开始时间，轴向分辨率有限。虽然频域方法（其中最有效的是基于交叉谱矩阵）需要长信号进行准确估计，但时域方法通常实现较低的空间分辨率。为解决这些局限性，我们提出了一种完全在时域中表述的基于线性模型的波束形成框架。线性前向模型将空化活动的离散时空分布与探头记录的时间信号相关联，明确考虑了由采集几何形状决定的飞行时间延迟。然后利用时空域中空化活动的先验知识，通过正则化技术对模型进行反演。实验结果表明，所提出的框架仅使用频域方法通常所需数据的20%，就能实现增强或具有竞争力的空化图质量。这突显了数据效率的显著提高，以及我们的时空正则化对适应各种被动空化场景的灵活性，优于最先进的技术。


### 论文摘要

Passive acoustic mapping enables the spatial mapping and temporal monitoring of cavitation activity, playing a crucial role in therapeutic ultrasound applications. Most conventional beamforming methods, whether implemented in the time or frequency domains, suffer from limited axial resolution due to the absence of a reference emission onset time. While frequency-domain methods, the most efficient of which are based on the cross-spectral matrix, require long signals for accurate estimation, time-domain methods typically achieve lower spatial resolution. To address these limitations, we propose a linear model-based beamforming framework fully formulated in the time domain. The linear forward model relates a discretized spatiotemporal distribution of cavitation activity to the temporal signals recorded by a probe, explicitly accounting for time-of-flight delays dictated by the acquisition geometry. This model is then inverted using regularization techniques that exploit prior knowledge of cavitation activity in both spatial and temporal domains. Experimental results show that the proposed framework achieves enhanced or competitive cavitation map quality while using only 20\% of the data typically required by frequency-domain methods. This highlights the substantial gain in data efficiency and the flexibility of our spatiotemporal regularization to adapt to diverse passive cavitation scenarios, outperforming state-of-the-art techniques.

---

## 252. Modelling the Spread of Toxicity and Exploring its Mitigation on Online Social Networks

**论文链接:** [http://arxiv.org/abs/2511.20546v1](http://arxiv.org/abs/2511.20546v1)

**作者:** Aatman Vaidya, Harsh Bhagat, Seema Nagar, Amit A. Nanavati

**发布时间:** 2025-11-25

### GPT解析

### 总结

本研究探讨了网络平台上仇恨言论的传播机制，提出了一种将用户视为毒性转换器的新模型，而非传统的二元分类方法。通过对Twitter、Koo和Gab平台的分析，发现毒性不守恒、只有部分用户行为会改变且这些用户间无同质性。基于用户对传入毒性的'转换'行为，开发了包含用户时变行为的网络模型，并提出通过部署和平机器人减少毒性的干预策略。

### 背景

网络平台上的仇恨言论已被证实与多起现实世界暴力事件相关，因此迫切需要理解有毒内容在在线社交网络上的传播方式及可能的缓解策略。

### 目的

研究有毒内容在社交网络上的传播机制，开发更准确的毒性传播模型，并提出有效的干预策略来减少网络上的仇恨言论。

### 方法

将用户视为毒性转换器而非简单的二元分类，分析用户对传入毒性的反应（放大、减弱或复制）。对Twitter、Koo和Gab平台进行时间分析，开发包含用户时变行为的网络模型，并通过实验测试和平机器人干预策略的有效性。

### 主要发现

a) 毒性在网络中不被保存（不守恒）；b) 只有一部分用户的行为会随时间改变；c) 行为改变的用户之间没有同质性证据；d) 用户根据输入毒性和类别应用'转换'来改变传入的毒性。

### 结论

基于用户毒性转换行为开发的网络模型能够更好地描述毒性传播机制。通过部署和平机器人可以有效减少网络上的毒性，但其效果取决于网络结构和放置策略。

### 翻译

网络平台上的仇恨言论已被可信地与现实世界的多起暴力事件联系起来。这迫切需要理解有毒内容如何在在线社交网络上传播以及如何减轻它，并已成为近期广泛研究的主题。先前的工作大多通过流行病或基于传播激活的扩散模型来建模仇恨，其中用户通常被分为两类：有仇恨或没有仇恨。在这项工作中，用户被视为毒性的转换器，基于他们对传入毒性的反应。与传入的毒性相比，用户会放大、减弱或复制（有效地转换）毒性并将其向前发送。我们对Twitter、Koo和Gab上的毒性进行了时间分析，发现(a)毒性在网络中不被保存；(b)只有一部分用户的行为随时间改变；(c)行为改变的用户之间没有同质性证据。在我们的模型中，每个用户在向前发送之前通过应用'转换'来改变传入的毒性。基于此，我们开发了一个包含用户时变行为的毒性传播网络模型。我们发现用户应用的'转换'取决于输入毒性和类别。基于这一发现，我们提出了一种减少毒性的干预策略。这通过部署和平机器人来模拟。通过对真实世界和合成网络的实验，我们证明和平机器人干预可以减少毒性，尽管它们的有效性取决于网络结构和放置策略。


### 论文摘要

Hate speech on online platforms has been credibly linked to multiple instances of real world violence. This calls for an urgent need to understand how toxic content spreads and how it might be mitigated on online social networks, and expectedly has been the topic of extensive research in recent times. Prior work has largely modelled hate through epidemic or spread activation based diffusion models, in which the users are often divided into two categories, hateful or not. In this work, users are treated as transformers of toxicity, based on how they respond to incoming toxicity. Compared with the incoming toxicity, users amplify, attenuate, or replicate (effectively, transform) the toxicity and send it forward. We do a temporal analysis of toxicity on Twitter, Koo and Gab and find that (a) toxicity is not conserved in the network; (b) only a subset of users change behaviour over time; and (c) there is no evidence of homophily among behaviour-changing users. In our model, each user transforms incoming toxicity by applying a "shift" to it prior to sending it forward. Based on this, we develop a network model of toxicity spread that incorporates time-varying behaviour of users. We find that the "shift" applied by a user is dependent on the input toxicity and the category. Based on this finding, we propose an intervention strategy for toxicity reduction. This is simulated by deploying peace-bots. Through experiments on both real-world and synthetic networks, we demonstrate that peace-bot interventions can reduce toxicity, though their effectiveness depends on network structure and placement strategy.

---

## 253. A mysterious feature in the NICER spectrum of 4U 1820-30: A gravitationally redshifted absorption line?

**论文链接:** [http://arxiv.org/abs/2511.20499v1](http://arxiv.org/abs/2511.20499v1)

**作者:** R. Iaria, T. Di Salvo, A. Anitra, F. Barra, A. Sanna, C. Maraventano, C. Miceli, W. Leone, L. Burderi

**发布时间:** 2025-11-25

**备注:** 14 pages, 10 figures, accepted by ApJ

### GPT解析

### 总结

研究在低质量X射线双星系统4U 1820-30的NICER光谱中发现约3.8 keV的神秘吸收特征，将其解释为引力红移的铁吸收线，该特征与碳超爆事件相关，通过测量红移值推断出中子星的致密性

### 背景

低质量X射线双星系统4U 1820-30中观测到约3.8 keV的神秘吸收特征，且该观测与MAXI探测到的碳超爆事件在时间上接近

### 目的

解释4U 1820-30系统中约3.8 keV吸收特征的起源，研究它与碳超爆事件的关联，通过吸收线红移推断中子星致密性

### 方法

使用NICER光谱观测分析，采用光电离吸收模型进行测量，分析吸收线的引力红移

### 主要发现

约3.8 keV的吸收特征为引力红移的铁吸收线，与碳超爆事件相关；测量到引力红移约为1.72；对应致密度R/M为4.46±0.13 km每太阳质量，或无量纲单位下的3.02±0.09

### 结论

这一独特特征强调了进一步观测和详细建模的重要性，为研究极端密度条件下的物质状态方程提供了有价值的见解

### 翻译

在低质量X射线双星系统4U 1820-30的NICER光谱中，已识别出约3.8 keV的神秘吸收特征。我们将此特征解释为引力红移的铁吸收线。这一解释得到了NICER观测与MAXI探测到的碳超爆事件在时间上的接近性的支持，表明该吸收线的存在与这一罕见且极端事件相关。从推断的吸收线红移，可以推导出中子星的致密性。使用光电离吸收模型，我们测量到约1.72的引力红移，对应于4.46±0.13 km每太阳质量的致密度R/M，或无量纲单位下的3.02±0.09。这一独特特征强调了进一步观测和详细建模的重要性，为研究极端密度条件下的物质状态方程提供了有价值的见解


### 论文摘要

A mysterious absorption feature at approximately 3.8 keV has been identified in the NICER spectrum of the low-mass X-ray binary system 4U 1820-30. We interpret this feature as a gravitationally redshifted iron absorption line. This interpretation is supported by the temporal proximity of the NICER observation to the detection of a carbon superburst by the X-ray monitor MAXI, suggesting that the presence of the line is associated with this rare and extreme event. From the inferred redshift of the absorption line, the compactness of the neutron star can be derived. Using a photoionization absorption model, we measure a gravitational redshift of about 1.72, which corresponds to a compactness R/M of 4.46 \pm 0.13 km per solar mass, or 3.02 \pm 0.09 in dimensionless units. This unique feature highlights the importance of further observations and detailed modelling, offering promising insights into the equation of state of matter under extreme density conditions.

---

## 254. Investigating access to support centers for Violence Against Women in Apulia: A Spatial analysis over multiple years

**论文链接:** [http://arxiv.org/abs/2511.20481v1](http://arxiv.org/abs/2511.20481v1)

**作者:** Leonardo Cefalo, Crescenza Calculli, Alessio Pollice

**发布时间:** 2025-11-25

### GPT解析

### 总结

本研究通过贝叶斯时空泊松回归模型，探讨了意大利南部地区市镇间对女性暴力的空间变异性问题，分析了社会经济特征和当地脆弱性对性别暴力发生率和报告的影响。

### 背景

研究聚焦于意大利南部地区各市镇对女性暴力的空间变异性，使用普利亚地区2021-2024年地方反暴力中心的数据。

### 目的

建模对女性暴力的空间变异性，调查市镇社会经济特征和当地脆弱性对性别暴力发生率和报告的影响，并明确考虑空间依赖性。

### 方法

提出贝叶斯时空泊松回归模型，在集成嵌套拉普拉斯近似框架内比较四种空间模型，评估竞争模型的相对拟合度，讨论先验假设、空间混杂效应和推断含义。

### 主要发现

获得支持服务的机会随与居住市镇距离增加而减少，揭示了报告中的空间限制和支持中心位置的战略重要性；较低教育水平导致弱势地区报告不足，较高经济发展可能与报告暴力发生率较低相关。

### 结论

空间建模在捕捉报告动态和告知政策干预方面发挥着关键作用。

### 翻译

在本研究中，我们通过提出贝叶斯时空泊松回归模型，解决了意大利南部地区市镇间针对女性的暴力行为空间变异性这一挑战。利用普利亚地区2021-2024年地方反暴力中心的数据，我们调查了市镇层面的社会经济特征和当地脆弱性对性别暴力发生率和报告的影响。为明确考虑空间依赖性，我们在贝叶斯模型估计的集成嵌套拉普拉斯近似框架内比较了四种空间模型。我们评估了竞争模型的相对拟合度，讨论了它们的先验假设、空间混杂效应和推断含义。我们的研究结果表明，获得支持服务的机会随着与居住市镇距离的增加而减少，突显了报告中的空间限制和支持中心位置的战略重要性。此外，较低的教育水平似乎导致弱势地区报告不足，而较高的经济发展可能与报告暴力发生率较低相关。这项研究强调了空间模型在捕捉报告动态和告知政策干预方面的关键作用。


### 论文摘要

In this study, we address the challenge of modelling the spatial variability in violence against women across municipalities in a Southern Italian region by proposing a Bayesian spatio-temporal Poisson regression model. Using data on access to Local Anti-Violence Centers in the Apulia region from 2021 to 2024, we investigate the impact of municipality-level socioeconomic characteristics and local vulnerabilities on both the incidence and reporting of gender-based violence. To explicitly account for spatial dependence, we compare four spatial models within the Integrated Nested Laplace Approximation framework for Bayesian model estimation. We assess the relative fit of the competing models, discussing their prior assumptions, spatial confounding effects, and inferential implications. Our findings indicate that access to support services decreases with distance from the residential municipality, highlighting spatial constraints in reporting and the strategic importance of support center location. Furthermore, lower education levels appear to contribute to under-reporting in disadvantaged areas, while higher economic development may be associated with a lower incidence of reported violence. This study emphasises the critical role of spatial modelling in capturing reporting dynamics and informing policy interventions.

---

## 255. Power-Efficient Autonomous Mobile Robots

**论文链接:** [http://arxiv.org/abs/2511.20467v1](http://arxiv.org/abs/2511.20467v1)

**作者:** Liangkai Liu, Weisong Shi, Kang G. Shin

**发布时间:** 2025-11-25

**备注:** 13 pages, 16 figures

### GPT解析

### 总结

本文提出了pNav，一种新型电源管理系统，通过联合优化自主移动机器人的物理/机械和信息系统子系统，显著提高了机器人的电源/能源效率。

### 背景

自主导航机器人在功耗管理方面面临挑战，特别是在信息物理系统环境中，系统功耗分解存在变异性，导航需要环境感知，且物理与信息系统需要协调。

### 目的

开发一种能够提高自主移动机器人电源效率的新型电源管理系统，通过联合优化其物理和信息系统子系统。

### 方法

pNav采用多方面方法：1)集成毫秒级的物理和信息系统功耗预测；2)实现时空导航局部性的实时建模和监控；3)支持软件和硬件配置的动态协调。系统使用ROS导航堆栈、2D LiDAR和相机进行原型设计。

### 主要发现

评估显示pNav实现了超过96%的功耗预测准确率，并将功耗降低了38.1%，同时保持了导航精度和安全性。

### 结论

pNav通过联合优化机器人的物理和信息系统，有效解决了自主移动机器人功耗管理中的关键挑战，实现了显著的节能效果。

### 翻译

本文提出了pNav，一种新型电源管理系统，通过联合优化自主移动机器人(AMRs)的物理/机械和信息系统子系统，显著提高了机器人的电源/能源效率。通过分析AMRs的功耗，我们确定了实现CPS(信息物理系统)电源效率的三个挑战，这些挑战涉及信息系统(C)和物理系统(P)两个方面：(1)系统功耗分解的变化性，(2)环境感知导航局部性，(3)子系统C和P的协调。pNav采用多方面方法实现AMRs的电源效率。首先，它集成了信息系统和物理子系统毫秒级的功耗预测。其次，它包含AMRs时空导航局部性的新颖实时建模和监控。第三，它支持AMR软件(导航、检测)和硬件(电机、DVFS驱动器)配置的动态协调。pNav使用机器人操作系统(ROS)导航堆栈、2D LiDAR和相机进行了原型设计。我们通过真实机器人和Gazebo环境的深入评估，证明了功耗预测准确率超过96%，功耗降低38.1%，且不影响导航精度和安全性。


### 论文摘要

This paper presents pNav, a novel power-management system that significantly enhances the power/energy-efficiency of Autonomous Mobile Robots (AMRs) by jointly optimizing their physical/mechanical and cyber subsystems. By profiling AMRs' power consumption, we identify three challenges in achieving CPS (cyber-physical system) power-efficiency that involve both cyber (C) and physical (P) subsystems: (1) variabilities of system power consumption breakdown, (2) environment-aware navigation locality, and (3) coordination of C and P subsystems. pNav takes a multi-faceted approach to achieve power-efficiency of AMRs. First, it integrates millisecond-level power consumption prediction for both C and P subsystems. Second, it includes novel real-time modeling and monitoring of spatial and temporal navigation localities for AMRs. Third, it supports dynamic coordination of AMR software (navigation, detection) and hardware (motors, DVFS driver) configurations. pNav is prototyped using the Robot Operating System (ROS) Navigation Stack, 2D LiDAR, and camera. Our in-depth evaluation with a real robot and Gazebo environments demonstrates a >96% accuracy in predicting power consumption and a 38.1% reduction in power consumption without compromising navigation accuracy and safety.

---

## 256. Radio Burst Phenomenology of AD Leonis and Associated Signatures of Propagation Effects

**论文链接:** [http://arxiv.org/abs/2511.20396v1](http://arxiv.org/abs/2511.20396v1)

**作者:** Jiale Zhang, Harish K. Vedantham, Joseph R. Callingham, Hui Tian

**发布时间:** 2025-11-25

**备注:** Accepted for publication in ApJ

### GPT解析

### 总结

本研究使用FAST望远镜在1.0-1.5 GHz频段对AD Leonis恒星进行了高分辨率射电动态谱观测，发现了复杂的时频结构，并提出了调制通道可能是传播效应的观点。

### 背景

AD Leonis是一颗恒星，研究人员利用中国FAST望远镜在2023年12月1日对其进行了射电观测。

### 目的

研究AD Leonis的射电辐射特征，特别是复杂的时频结构及其可能的物理机制和起源。

### 方法

使用FAST望远镜进行观测，通过离散傅里叶变换和自相关函数分析周期性发射模式，并建立等离子体屏幕模型来模拟调制通道的形成机制。

### 主要发现

在15分钟观测期内发现了三种复杂的时频结构：宽带秒级调制通道（向下频率漂移）、窄带短时S爆发包络（向上漂移，约50 MHz）和更窄的毫秒级S爆发条纹（约10 MHz）；识别出两个主导周期性模式（S爆发约0.1秒，条纹约0.01秒）；调制通道可能是射电波穿过磁层中规则结构等离子体区域的传播效应。

### 结论

恒星磁层中的传播效应可以探测发射区域中公里级结构，并为磁流体动力学波引起的密度不均匀性提供新的约束条件，这些约束通过其他方式难以获得。

### 翻译

我们展示了2023年12月1日使用五百米口径球面射电望远镜（FAST）对AD Leonis（AD Leo）在1.0至1.5 GHz频段进行的高分辨率射电动态谱。在15分钟期间，我们确定了复杂的、叠加的谱时结构，包括：（1）宽带、秒级调制通道，具有向下频率漂移；（2）窄带（约50 MHz）、短时S爆发包络，具有向上漂移；（3）在这些包络内更窄（约10 MHz）、毫秒级的S爆发条纹。使用离散傅里叶变换和自相关函数，我们确定了两个主导的周期性发射模式，对应于S爆发（约0.1秒）和条纹（约0.01秒）的周期性。多种时频结构的复杂叠加难以将所有发射变异性解释为源的内禀特性。我们提出调制通道可能是射电波穿过AD Leo磁层中不均匀、规则结构等离子体区域时的传播效应。通过建立具有一维正弦相位变化的等离子体屏幕模型，我们证明可以定性重建观测到的调制通道。最精细结构（条纹）的起源仍不清楚。我们的工作强调了恒星磁层中的传播效应可以潜在地探测发射区域中的公里级结构，并为通过其他方式难以获得的磁流体动力学波引起的密度不均匀性提供新的约束条件。


### 论文摘要

We present the high-resolution radio dynamic spectra of AD Leonis (AD Leo) between 1.0 and 1.5 GHz taken by the Five-hundred-meter Aperture Spherical radio Telescope (FAST) on Dec. 1st, 2023. Over a 15-minute period, we identify complex, superimposed spectro-temporal structures, including: (1) broadband, second-long modulation lanes with downward frequency drifts, (2) narrowband ($\approx$ 50 MHz), short-duration S-burst envelopes with upward drifts, and (3) even narrower ($\approx$ 10 MHz), millisecond-scale S-burst striae within these envelopes. Using the discrete Fourier transform and auto-correlation function, we identify two dominant periodic emission patterns, corresponding to the periodicities of the S-bursts ($\approx0.1$ s) and the striae ($\approx0.01$ s). The complex superposition of diverse time-frequency structures poses a challenge to interpreting all the emission variability as intrinsic to the source. We propose that the modulation lanes could be a propagation effect as the radio waves traverse an inhomogeneous, regularly structured plasma region in the AD Leo's magnetosphere. By modelling a plasma screen with sinusoidal phase variation in one dimension, we show that we could qualitatively reconstruct the observed modulation lanes. The origin of the finest structures, the striae, remains unclear. Our work highlights that propagation effects in the stellar magnetosphere can potentially probe kilometre-scale structures in the emission regions and provide novel constraints on density inhomogeneities caused by magnetohydrodynamic waves that are difficult to access by other means.

---

## 257. Identifying environmental factors associated with tetrodotoxin contamination in bivalve mollusks using eXplainable AI

**论文链接:** [http://arxiv.org/abs/2511.20395v1](http://arxiv.org/abs/2511.20395v1)

**作者:** M. C. Schoppema, B. H. M. van der Velden, A. Hürriyetoğlu, M. D. Klijnstra, E. J. Faassen, A. Gerssen, H. J. van der Fels-Klerx

**发布时间:** 2025-11-25

**备注:** 18 pages, 6 figures, submitted to Nature Food

### GPT解析

### 总结

该研究开发了一个可解释的深度学习模型来预测双壳类海鲜中的河豚毒素污染，确定了日照时数、全球辐射、水温和水氯离子浓度为主要影响因素。

### 背景

自2012年以来，河豚毒素(TTX)在欧洲温带水域的双壳类海鲜中被发现，导致食品安全风险和经济损失，因此早期预测至关重要。

### 目的

开发一个可解释的深度学习模型来预测荷兰泽兰河口地区的TTX污染。

### 方法

使用气象和水文特征作为输入，TTX污染的存在与否作为输出，构建深度学习模型。

### 主要发现

日出时间、日落时间、全球辐射、水温和氯离子浓度对TTX污染贡献最大，有效日照时数是双壳类动物河豚毒素污染的重要驱动因素。

### 结论

环境因素（日照时数、全球辐射、水温和水氯离子浓度）与双壳类动物河豚毒素污染相关，该方法可帮助食品行业和主管部门降低海洋毒素风险。

### 翻译

Since 2012, tetrodotoxin (TTX) has been found in seafoods such as bivalve mollusks in temperate European waters. TTX contamination leads to food safety risks and economic losses, making early prediction of TTX contamination vital to the food industry and competent authorities. Recent studies have pointed to shallow habitats and water temperature as main drivers to TTX contamination in bivalve mollusks. However, the temporal relationships between abiotic factors, biotic factors, and TTX contamination remain unexplored. We have developed an explainable, deep learning-based model to predict TTX contamination in the Dutch Zeeland estuary. Inputs for the model were meteorological and hydrological features; output was the presence or absence of TTX contamination. Results showed that the time of sunrise, time of sunset, global radiation, water temperature, and chloride concentration contributed most to TTX contamination. Thus, the effective number of sun hours, represented by day length and global radiation, was an important driver for tetrodotoxin contamination in bivalve mollusks. To conclude, our explainable deep learning model identified the aforementioned environmental factors (number of sun hours, global radiation, water temperature, and water chloride concentration) to be associated with tetrodotoxin contamination in bivalve mollusks; making our approach a valuable tool to mitigate marine toxin risks for food industry and competent authorities.


### 论文摘要

Since 2012, tetrodotoxin (TTX) has been found in seafoods such as bivalve mollusks in temperate European waters. TTX contamination leads to food safety risks and economic losses, making early prediction of TTX contamination vital to the food industry and competent authorities. Recent studies have pointed to shallow habitats and water temperature as main drivers to TTX contamination in bivalve mollusks. However, the temporal relationships between abiotic factors, biotic factors, and TTX contamination remain unexplored.   We have developed an explainable, deep learning-based model to predict TTX contamination in the Dutch Zeeland estuary. Inputs for the model were meteorological and hydrological features; output was the presence or absence of TTX contamination.   Results showed that the time of sunrise, time of sunset, global radiation, water temperature, and chloride concentration contributed most to TTX contamination. Thus, the effective number of sun hours, represented by day length and global radiation, was an important driver for tetrodotoxin contamination in bivalve mollusks.   To conclude, our explainable deep learning model identified the aforementioned environmental factors (number of sun hours, global radiation, water temperature, and water chloride concentration) to be associated with tetrodotoxin contamination in bivalve mollusks; making our approach a valuable tool to mitigate marine toxin risks for food industry and competent authorities.

---

## 258. Mechano-chemical modeling of glia initiated secondary injury of neurons under mechanical load

**论文链接:** [http://arxiv.org/abs/2511.20392v1](http://arxiv.org/abs/2511.20392v1)

**作者:** Debabrata Auddya, Shiva Rudraraju

**发布时间:** 2025-11-25

**备注:** 26 pages, 10 figures

### GPT解析

### 总结

本研究提出了一种连续介质框架，用于模拟创伤性脑损伤(TBI)背后的多物理场力学-化学相互作用，特别是由胶质细胞引发的继发性损伤途径。

### 背景

创伤性脑损伤是由头部撞击或震荡引起的，特征是在不同生物长度尺度上的病理退化。文献中已提出各种机械建模技术来量化从神经元到组织尺度的脑损伤指标，退化主要包括神经元生理退化和化学实体(如神经递质)的上调。

### 目的

阐明和建模一种由胶质细胞引发的潜在损伤途径导致继发性损伤；展示一个连续介质框架模拟TBI背后的多物理场力学-化学相互作用；量化继发性损伤以帮助开发有针对性的TBI治疗方法。

### 方法

使用耦合的偏微分方程公式和有限元法离散化；框架解析机械指标和化学物质种类的空间-时间演变；建模域包括小胶质细胞、神经元和细胞外基质；采用三维粘弹性网络模拟机械响应；使用平流-扩散方程模拟化学物质演变；数值估计由应变场产生的关键化学物质浓度。

### 主要发现

识别了分子通路网络中的关键生物标志物；构建了捕捉核心力学-化学相互作用的框架。

### 结论

该框架是量化继发性损伤的一种尝试，有助于开发有针对性的TBI治疗方法。

### 翻译

创伤性脑损伤(TBI)是由头部撞击或震荡引起的，其特征是在不同生物长度尺度上的病理退化。损伤后，文献中提出了各种机械建模技术，试图量化从神经元尺度到组织尺度的脑损伤指标。广义上，退化的两个类别包括神经元的生理退化和化学实体(如神经递质)的上调，这会引发下游病理生理效应。尽管有许多参与途径，在本研究中，我们阐明并建模了一种由胶质细胞引发的潜在损伤途径，导致继发性损伤。本工作的目标是展示一个连续介质框架，该框架模拟了TBI背后的多物理场力学-化学相互作用。使用耦合的偏微分方程公式和有限元法离散化，该框架突出了空间-时间上解析的机械指标和化学物质种类在神经元集群中的演变。建模域包括小胶质细胞、神经元和细胞外基质。用于模拟力学-化学相互作用的连续介质框架假设了一个三维粘弹性网络来捕捉构成神经元微观结构的蛋白的机械响应，并使用平流-扩散方程模拟化学物质种类的空间-时间演变。我们使用该框架数值估计由应变场产生的关键化学物质浓度。在本研究中，我们在分子通路网络中识别了关键生物标志物，并构建了一个捕捉核心力学-化学相互作用的框架。该框架是量化继发性损伤的一种尝试，因此有助于开发有针对性的TBI治疗方法。


### 论文摘要

Traumatic Brain Injury (TBI) results from an impact or concussion to the head with the injury being specifically characterized through pathological degradation at various biological length scales. Following injury, various mechanical modeling techniques have been proposed in the literature that seek to quantify neuronal-scale to tissue-scale metrics of brain damage. Broadly, the two categories of degradation encompass physiological deterioration of neurons and upregulation of chemical entities such as neurotransmitters which causes initiation of downstream pathophysiological effects. Despite the many contributing pathways, in this work, we delineate and model a potential glia-initiated injury pathway that leads to secondary injury. The goal of this work is to demonstrate a continuum framework which models the multiphysics of mechano-chemical interactions underlying TBI. Using a coupled PDE (partial differential equation) formulation and FEM (finite element method) discretization, the framework highlights evolution of field variables which spatio-temporally resolve mechanical metrics and chemical species across neuronal clusters. The modeling domain encompasses microglia, neurons and the extracellular matrix. The continuum framework used to model the mechano-chemical interactions assumes a three dimensional viscoelastic network to capture the mechanical response underlying proteins constituting the neuron microstructure and advection-diffusion equations modeling spatio-temporal evolution of chemical species. We use this framework to numerically estimate key concentrations of chemical species produced by the strain field. In this work, we identify key biomarkers within the labyrinth of molecular pathways and build a framework that captures the core mechano-chemical interactions. This framework is an attempt to quantify secondary injury and thus assist in developing targeted TBI treatments.

---

## 259. FREE: Uncertainty-Aware Autoregression for Parallel Diffusion Transformers

**论文链接:** [http://arxiv.org/abs/2511.20390v1](http://arxiv.org/abs/2511.20390v1)

**作者:** Xinwan Wen, Bowen Li, Jiajun Luo, Ye Li, Zhi Wang

**发布时间:** 2025-11-25

### GPT解析

### 总结

本文提出了一种名为FREE的新型框架，用于加速Diffusion Transformers (DiTs)的推理过程，同时保持生成质量。通过分析DiTs特征动态，利用最终transformer层特征的时间一致性和语义抽象特性，结合不确定性引导的松弛策略，实现了显著加速。

### 背景

Diffusion Transformers (DiTs)虽能实现最先进的生成质量，但需要长序列去噪轨迹导致高推理延迟。现有推测推理方法在基于U-Net的扩散模型中可实现无损并行采样，但由于验证过程中起草准确性不足，这些方法在DiTs上的加速效果有限。

### 目的

解决DiTs在推测推理中起草准确性不足的问题，实现更高效的加速，同时保持生成质量的高感知和保真度。

### 方法

提出FREE框架，使用轻量级起草者进行特征级自回归并行验证；引入不确定性引导的松弛策略形成FREE (relax)，根据不确定性水平动态调整接受概率。基于DiTs最终transformer层特征表现出强时间一致性和丰富语义抽象的发现进行设计。

### 主要发现

DiTs的最终transformer层（top-block）特征表现出强时间一致性和丰富的语义抽象，这为特征级自回归提供了理论基础；DiTs在后续去噪步骤中的预测方差自然增加，导致推测采样接受率降低。

### 结论

在ImageNet-512×2上的实验表明，FREE实现了高达1.86倍的加速，而FREE (relax)进一步达到了2.25倍的加速，同时保持了生成质量的高感知和保真度。

### 翻译

扩散变换器(DiTs)实现了最先进的生成质量，但需要长序列去噪轨迹，导致高推理延迟。最近的推测推理方法通过起草者-验证者方案在基于U-Net的扩散模型中实现无损并行采样，但由于验证过程中起草准确性不足，这些方法在DiTs上的加速效果有限。为解决这一限制，我们分析了DiTs的特征动态，发现最终transformer层(top-block)的特征表现出强时间一致性和丰富的语义抽象。基于这一见解，我们提出了FREE，一个使用轻量级起草者进行特征级自回归并行验证的新框架，有理论和经验支持保证无损加速。同时，DiTs的预测方差(不确定性)在后续去噪步骤中自然增加，降低了推测采样下的接受率。为缓解这一影响，我们进一步引入了不确定性引导的松弛策略，形成FREE (relax)，能够根据不确定性水平动态调整接受概率。在ImageNet-512×2上的实验表明，FREE实现了高达1.86倍的加速，而FREE (relax)进一步达到了2.25倍的加速，同时保持了生成质量的高感知和定量保真度。


### 论文摘要

Diffusion Transformers (DiTs) achieve state-of-the-art generation quality but require long sequential denoising trajectories, leading to high inference latency. Recent speculative inference methods enable lossless parallel sampling in U-Net-based diffusion models via a drafter-verifier scheme, but their acceleration is limited on DiTs due to insufficient draft accuracy during verification. To address this limitation, we analyze the DiTs' feature dynamics and find the features of the final transformer layer (top-block) exhibit strong temporal consistency and rich semantic abstraction. Based on this insight, we propose FREE, a novel framework that employs a lightweight drafter to perform feature-level autoregression with parallel verification, guaranteeing lossless acceleration with theoretical and empirical support. Meanwhile, prediction variance (uncertainty) of DiTs naturally increases in later denoising steps, reducing acceptance rates under speculative sampling. To mitigate this effect, we further introduce an uncertainty-guided relaxation strategy, forming FREE (relax), which dynamically adjusts the acceptance probability in response to uncertainty levels. Experiments on ImageNet-$512^2$ show that FREE achieves up to $1.86 \times$ acceleration, and FREE (relax) further reaches $2.25 \times$ speedup while maintaining high perceptual and quantitative fidelity in generation quality.

---

## 260. 论文ID: 2511.20272v1

**错误:** 无法读取论文信息 - [Errno 2] No such file or directory: 'storage/result/2511.20272v1.json'

---

## 261. SFA: Scan, Focus, and Amplify toward Guidance-aware Answering for Video TextVQA

**论文链接:** [http://arxiv.org/abs/2511.20190v1](http://arxiv.org/abs/2511.20190v1)

**作者:** Haibin He, Qihuang Zhong, Juhua Liu, Bo Du, Peng Wang, Jing Zhang

**发布时间:** 2025-11-25

### GPT解析

### 总结

该研究提出了一种名为SFA的无需训练框架，是首个专为Video TextVQA设计的Video-LLM方法，通过自适应扫描视频帧、选择性关注关键区域并放大它们，有效引导Video-LLM的注意力到重要线索上，从而提高视频文本视觉问答的准确性。

### 背景

Video TextVQA任务需要模型能够准确感知和理解视频中变化的场景文本（不同尺度、方向和清晰度），同时有效整合时间和语义上下文来生成精确答案。此外，模型还需识别问题相关的文本线索并过滤冗余或无关信息。

### 目的

解决Video TextVQA任务中的挑战，提高模型对视频中场景文本的感知和理解能力，以及整合上下文信息生成精确答案的能力。

### 方法

提出SFA框架，一个无需训练的框架，也是首个专为Video TextVQA设计的Video-LLM方法。该方法通过自适应扫描视频帧、选择性关注关键区域并直接放大它们，有效引导Video-LLM的注意力到重要线索上。

### 主要发现

SFA在多个公共Video TextVQA数据集上取得了最新的最先进结果，并且以显著优势超越了之前的方法。

### 结论

SFA在Video TextVQA任务上具有有效性和通用性，能够通过引导大语言模型的注意力到关键区域来提高问答准确性。

### 翻译

Video TextVQA任务旨在通过利用视频中出现的视觉文本来回答关于视频的问题。该任务面临重大挑战，需要模型能够准确感知和理解跨帧中尺度、方向和清晰度各异的场景文本，同时有效整合时间和语义上下文以生成精确答案。此外，模型必须识别问题相关的文本线索并过滤冗余或无关信息，确保回答由最相关和信息量最大的线索引导。为应对这些挑战，我们提出了SFA，一个无需训练的框架，也是首个专为Video TextVQA设计的Video-LLM方法，其灵感来自人类的问答过程。通过自适应扫描视频帧、选择性关注关键区域并直接放大它们，SFA有效地引导Video-LLM的注意力转向重要线索，使其能够生成更准确的答案。SFA在多个公共Video TextVQA数据集上取得了最新的最先进结果，并以显著优势超越了之前的方法，证明了其有效性和通用性。


### 论文摘要

Video text-based visual question answering (Video TextVQA) task aims to answer questions about videos by leveraging the visual text appearing within the videos. This task poses significant challenges, requiring models to accurately perceive and comprehend scene text that varies in scale, orientation, and clarity across frames, while effectively integrating temporal and semantic context to generate precise answers. Moreover, the model must identify question-relevant textual cues and filter out redundant or irrelevant information to ensure answering is guided by the most relevant and informative cues. To address these challenges, we propose SFA, a training-free framework and the first Video-LLM-based method tailored for Video TextVQA, motivated by the human process of answering questions. By adaptively scanning video frames, selectively focusing on key regions, and directly amplifying them, SFA effectively guides the Video-LLM's attention toward essential cues, enabling it to generate more accurate answers. SFA achieves new state-of-the-art results across several public Video TextVQA datasets and surpasses previous methods by a substantial margin, demonstrating its effectiveness and generalizability.

---

## 262. Alzheimers Disease Progression Prediction Based on Manifold Mapping of Irregularly Sampled Longitudinal Data

**论文链接:** [http://arxiv.org/abs/2511.20154v1](http://arxiv.org/abs/2511.20154v1)

**作者:** Xin Hong, Ying Shi, Yinhao Li, Yen-Wei Chen

**发布时间:** 2025-11-25

**备注:** 10 pages, 3 figures

### GPT解析

### 总结

该研究提出了一种名为R-TNAG的框架，用于解决阿尔茨海默病进展建模中不规则采样纵向成像数据的挑战，通过结合黎曼流形映射、时间感知神经常微分方程和基于注意力的黎曼门控循环单元，实现了优于现有方法的预测性能。

### 背景

临床检查的不确定性导致纵向成像数据中的观察间隔不规则，大多数现有基于成像的疾病预测模型在欧几里得空间中运行，无法完全捕捉不规则采样纵向图像的内在连续性和非线性几何结构。

### 目的

解决从不规则采样的阿尔茨海默病纵向结构磁共振成像数据建模疾病进展的挑战。

### 方法

提出R-TNAG框架，包括：1)将高维sMRI特征投影到黎曼流形空间保留疾病进展几何结构；2)使用时间感知神经常微分方程建模观察间潜在状态的连续演化；3)采用基于注意力的黎曼门控循环单元自适应整合历史和当前信息处理不规则间隔。

### 主要发现

实验结果表明该方法在疾病状态预测和认知分数回归方面优于最先进模型；消融研究验证了各模块的互补作用；模型在不同序列长度和缺失数据率下表现稳定；跨数据集验证确认了其在多样化临床环境中的稳健性和适用性。

### 结论

所提出的联合设计提高了时间一致性，在不规则采样下产生稳健的AD轨迹预测，为不规则采样纵向成像数据中的疾病进展建模提供了有效解决方案。

### 翻译

临床检查的不确定性经常导致纵向成像数据中的观察间隔不规则，对建模疾病进展构成挑战。大多数现有的基于成像的疾病预测模型在欧几里得空间中运行，这假设了数据的平坦表示，无法完全捕捉不规则采样纵向图像的内在连续性和非线性几何结构。为了解决从不规则采样的阿尔茨海默病纵向结构磁共振成像数据建模AD进展的挑战，我们提出了一种黎曼流形映射、一种时间感知流形神经常微分方程和一种基于注意力的黎曼门控循环单元(R-TNAG)框架。我们的方法首先将从高维sMRI提取的特征投影到流形空间，以保留疾病进展的内在几何结构。在这种表示上，时间感知神经常微分方程建模观察之间潜在状态的连续演化，而基于注意力的黎曼门控循环单元自适应地整合历史和当前信息以处理不规则间隔。这种联合设计提高了时间一致性，并在不规则采样下产生稳健的AD轨迹预测。实验结果表明，所提出的方法在疾病状态预测和认知分数回归方面 consistently优于最先进的模型。消融研究验证了每个模块的贡献，突显了它们在提高预测准确性方面的互补作用。此外，该模型在不同序列长度和缺失数据率下表现出稳定的性能，表明具有较强的时间泛化能力。跨数据集验证进一步确认了其在多样化临床环境中的稳健性和适用性。


### 论文摘要

The uncertainty of clinical examinations frequently leads to irregular observation intervals in longitudinal imaging data, posing challenges for modeling disease progression.Most existing imaging-based disease prediction models operate in Euclidean space, which assumes a flat representation of data and fails to fully capture the intrinsic continuity and nonlinear geometric structure of irregularly sampled longitudinal images. To address the challenge of modeling Alzheimers disease (AD) progression from irregularly sampled longitudinal structural Magnetic Resonance Imaging (sMRI) data, we propose a Riemannian manifold mapping, a Time-aware manifold Neural ordinary differential equation, and an Attention-based riemannian Gated recurrent unit (R-TNAG) framework. Our approach first projects features extracted from high-dimensional sMRI into a manifold space to preserve the intrinsic geometry of disease progression. On this representation, a time-aware Neural Ordinary Differential Equation (TNODE) models the continuous evolution of latent states between observations, while an Attention-based Riemannian Gated Recurrent Unit (ARGRU) adaptively integrates historical and current information to handle irregular intervals. This joint design improves temporal consistency and yields robust AD trajectory prediction under irregular sampling.Experimental results demonstrate that the proposed method consistently outperforms state-of-the-art models in both disease status prediction and cognitive score regression. Ablation studies verify the contributions of each module, highlighting their complementary roles in enhancing predictive accuracy. Moreover, the model exhibits stable performance across varying sequence lengths and missing data rates, indicating strong temporal generalizability. Cross-dataset validation further confirms its robustness and applicability in diverse clinical settings.

---

## 263. Dual Stressors in Engineering Education: Lagged Causal Effects of Academic Staff Strikes and Inflation on Dropout within the CAPIRE Framework

**论文链接:** [http://arxiv.org/abs/2511.20130v1](http://arxiv.org/abs/2511.20130v1)

**作者:** H. R. Paz

**发布时间:** 2025-11-25

### GPT解析

### 总结

本研究验证了阿根廷长期工程教育项目中的双重压力源假设，发现学术罢工与通货膨胀共同影响学生辍学，且这种影响具有时间依赖性和交互作用。

### 背景

研究在阿根廷长期工程教育项目背景下，考察学术罢工和通货膨胀作为双重压力源对学生辍学的影响。

### 目的

测试学术人员罢工（近端冲击）和通货膨胀（远端冲击）是否共同影响学生辍学决策。

### 方法

使用包含1,343名学生的纵向面板数据，采用LinearDML估计器，分析罢工暴露的滞后效应及其与入学时通货膨胀的交互作用，使用滞后逻辑模型和双重机器学习方法，并进行安慰剂测试和稳健性审计。

### 主要发现

只有发生在两个学期前的罢工对下一学期辍学有显著影响；控制学术进展等因素后，罢工主效应不显著，但罢工与入学时通货膨胀的交互作用保持显著且稳健；SHAP分析显示两者共同强烈预测辍学风险。

### 结论

研究结果支持宏观冲击作为耦合压力源的观点，这些压力源通过课程摩擦和财务弹性介导，而非孤立事件，与CAPIRE-MACRO基于主体的模拟结果一致。

### 翻译

本研究在阿根廷长期工程教育项目中为双重压力源假设提供了因果验证，测试了学术人员罢工（近端冲击）和通货膨胀（远端冲击）是否共同影响学生辍学。使用包含1,343名学生的纵向面板数据和手动实现的LinearDML估计器，我们估计了罢工暴露的滞后因果效应及其与入学时通货膨胀的交互作用。时间轮廓清晰：仅在滞后两学期的罢工在简单的滞后逻辑模型中对下一学期辍学有显著影响，而其他滞后效应可忽略不计。当我们转向双重机器学习并灵活控制学术进展、课程摩擦和日历效应时，滞后两的主要罢工效应变小且统计不显著，但罢工与入学时通货膨胀的交互作用保持正向且稳健。使用合成罢工变量的安慰剂模型产生无效效应，稳健性审计证实交互作用在不同规格下的稳定性。SHAP分析还显示，罢工_Lag2和入学时通货膨胀共同强烈预测辍学风险。这些发现与CAPIRE-MACRO基于主体的模拟一致，支持宏观冲击作为由课程摩擦和财务弹性介导的耦合压力源而非孤立事件的观点。


### 论文摘要

This study provides a causal validation of the dual-stressor hypothesis in a long-cycle engineering programme in Argentina, testing whether academic staff strikes (proximal shocks) and inflation (distal shocks) jointly shape student dropout. Using a leak-aware longitudinal panel of 1,343 students and a manually implemented LinearDML estimator, we estimate lagged causal effects of strike exposure and its interaction with inflation at entry. The temporal profile is clear: only strikes occurring two semesters earlier have a significant impact on next-semester dropout in simple lagged logit models (ATE = 0.0323, p = 0.0173), while other lags are negligible. When we move to double machine learning and control flexibly for academic progression, curriculum friction and calendar effects, the main effect of strikes at lag 2 becomes small and statistically non-significant, but the interaction between strikes and inflation at entry remains positive and robust (estimate = 0.0625, p = 0.0033). A placebo model with a synthetic strike variable yields null effects, and a robustness audit (seed sensitivity, model comparisons, SHAP inspection) confirms the stability of the interaction across specifications. SHAP analysis also reveals that Strikes_Lag2 and Inflation_at_Entry jointly contribute strongly to predicted dropout risk. These findings align with the CAPIRE-MACRO agent-based simulations and support the view that macro shocks act as coupled stressors mediated by curriculum friction and financial resilience rather than isolated events.

---

## 264. RoadBench: Benchmarking MLLMs on Fine-Grained Spatial Understanding and Reasoning under Urban Road Scenarios

**论文链接:** [http://arxiv.org/abs/2511.18011v1](http://arxiv.org/abs/2511.18011v1)

**作者:** Jun Zhang, Jie Feng, Long Chen, Junhui Wang, Zhicheng Liu, Depeng Jin, Yong Li

**发布时间:** 2025-11-22

**备注:** The code and data are publicly available at: https://github.com/tsinghua-fib-lab/RoadBench

### GPT解析

### 总结

本文提出了RoadBench，一个专门用于评估多模态大语言模型在复杂城市场景中细粒度空间理解和推理能力的基准测试。通过对14个主流模型的评估，发现它们在处理道路标记等城市元素时存在明显不足，性能甚至低于简单基线方法。

### 背景

多模态大语言模型在一般空间理解和推理方面展现出强大能力，但在复杂城市场景中的细粒度空间理解和推理能力尚未受到足够关注。

### 目的

填补研究空白，专注于道路标记作为城市场景下细粒度空间元素的典型例子，评估和提升MLLMs在复杂城市环境中的空间理解能力。

### 方法

提出RoadBench系统性基准测试，使用鸟瞰图和第一人称视角图像输入，包含6个任务共9,121个严格手动验证的测试用例，构建从局部空间理解到全局推理的系统评估框架。

### 主要发现

评估确认RoadBench对MLLMs具有挑战性，揭示了现有模型在城市场景中细粒度空间理解和推理能力的重大缺陷，某些任务性能低于简单基线方法。

### 结论

这些发现和RoadBench基准测试将有助于推动MLLMs空间理解能力的全面进步，相关代码和数据可在补充材料中获取。

### 翻译

多模态大语言模型在一般空间理解和推理方面已展现出强大能力。然而，它们在复杂城市场景中的细粒度空间理解和推理能力在研究和工业领域都未受到显著关注。为填补这一空白，我们主要关注道路标记作为城市场景下细粒度空间元素的典型例子，考虑到它们在城市中形成的综合道路交通网络所扮演的重要角色。围绕道路标记和城市交通系统，我们提出了RoadBench，一个使用鸟瞰图和第一人称视角图像输入全面评估MLLMs细粒度空间理解和推理能力的系统性基准。该基准包含6个任务，共9,121个严格手动验证的测试用例。这些任务构成了一个从局部空间范围理解到全局推理的系统评估框架，它们不仅测试MLLMs在识别、联合理解和推理方面的能力，还评估其将图像信息与领域知识整合的能力。在评估了14个主流MLLMs后，我们确认RoadBench对MLLMs具有挑战性，同时揭示了现有MLLMs在城市场景中细粒度空间理解和推理能力的重大缺陷。在某些任务中，其性能甚至低于简单的基于规则或随机选择基线。这些发现以及RoadBench本身将有助于推动MLLMs空间理解能力的全面进步。基准代码、示例数据集和原始评估结果可在补充材料中获取。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决多模态大语言模型(MLLMs)在复杂城市场景下细粒度空间理解和推理能力缺乏系统评估的问题。这很重要，因为城市道路场景是MLLMs的关键应用领域，涉及自动驾驶、高清地图生成等实际应用；道路标记作为城市交通网络的重要元素，共同构成了城市交通系统的基础；现有基准测试主要关注整图像级别理解或特定物体识别，缺乏对细粒度空间元素的关注；提升这方面的能力将推动MLLMs在交通领域的应用并增强其通用视觉-空间推理能力。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先观察到MLLMs在细粒度空间理解和推理方面的不足，选择道路标记作为城市场景中细粒度空间元素的典型代表；设计了从局部到全局的层次化评估框架，包含六个任务；收集处理了9,121个经过严格手动验证的测试用例。作者借鉴了现有MLLMs架构(如CLIP、BLIP)、其他空间理解基准测试(如CityBench、UrBench)的设计思路，利用了地图服务提供商数据和OpenStreetMap等开源数据，并采用了现有的评估指标和方法。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是通过专注于道路标记这一细粒度空间元素，系统性地评估MLLMs在城市场景中的细粒度空间理解和推理能力；设计从局部到全局的层次化评估框架，结合BEV和FPV两种图像视角。整体流程包括：1)数据准备(收集卫星图像和车载摄像头照片，筛选有效图像)；2)数据处理(提取标签，匹配数据，匿名化处理)；3)基准测试构建(设计六个任务和评估指标)；4)质量控制(手动校对，拒绝错误数据)。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点：1)提出RoadBench填补了细粒度空间理解和推理基准测试的空白；2)专注于道路标记这一典型细粒度空间元素；3)设计从局部到全局的层次化评估框架；4)结合BEV和FPV两种图像视角；5)收集处理9,121个严格验证的测试用例；6)系统性评估14个主流MLLMs揭示不足。相比之前工作，RoadBench更专注于细粒度空间元素而非整图像理解；关注道路标记而非交通标志等较粗粒度元素；专注于真实城市场景而非抽象视觉谜题；是首个系统评估MLLMs处理城市道路细粒度空间元素能力的基准测试。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '本文提出了RoadBench基准测试，系统评估MLLMs在城市道路场景下的细粒度空间理解和推理能力，通过六个任务和9,121个测试用例揭示了现有模型的不足，为提升MLLMs的空间理解能力提供了新方向和评估标准。'}


### 论文摘要

Multimodal large language models (MLLMs) have demonstrated powerful capabilities in general spatial understanding and reasoning. However, their fine-grained spatial understanding and reasoning capabilities in complex urban scenarios have not received significant attention in the fields of both research and industry. To fill this gap, we focus primarily on road markings as a typical example of fine-grained spatial elements under urban scenarios, given the essential role of the integrated road traffic network they form within cities. Around road markings and urban traffic systems, we propose RoadBench, a systematic benchmark that comprehensively evaluates MLLMs' fine-grained spatial understanding and reasoning capabilities using BEV and FPV image inputs. This benchmark comprises six tasks consisting of 9,121 strictly manually verified test cases. These tasks form a systematic evaluation framework that bridges understanding at local spatial scopes to global reasoning. They not only test MLLMs' capabilities in recognition, joint understanding, and reasoning but also assess their ability to integrate image information with domain knowledge. After evaluating 14 mainstream MLLMs, we confirm that RoadBench is a challenging benchmark for MLLMs while revealing significant shortcomings in existing MLLMs' fine-grained spatial understanding and reasoning capabilities within urban scenarios. In certain tasks, their performance even falls short of simple rule-based or random selection baselines. These findings, along with RoadBench itself, will contribute to the comprehensive advancement of spatial understanding capabilities for MLLMs. The benchmark code, example datasets, and raw evaluation results are available in the supplementary material.

---

## 265. CORA: Consistency-Guided Semi-Supervised Framework for Reasoning Segmentation

**论文链接:** [http://arxiv.org/abs/2511.17755v1](http://arxiv.org/abs/2511.17755v1)

**作者:** Prantik Howlader, Hoang Nguyen-Canh, Srijan Das, Jingyi Xu, Hieu Le, Dimitris Samaras

**发布时间:** 2025-11-21

**备注:** WACV 2026 accepted

### GPT解析

### 总结

这篇论文提出了CORA，一个半监督推理分割框架，通过条件视觉指令、噪声伪标签过滤和令牌级对比对齐三个组件，实现了在有限标注数据下的高性能推理分割。

### 背景

推理分割需要根据复杂指令生成像素级精确的掩码，需要上下文推理。最近的多模态语言模型在指令跟随分割方面取得了进展，但泛化能力有限，主要瓶颈是高质量标注数据的高成本。

### 目的

开发一个能够在有限标注数据下实现高性能推理分割的半监督框架，解决现有方法在分布变化下性能脆弱的问题。

### 方法

CORA框架包含三个主要组件：1)条件视觉指令，编码对象间的空间和上下文关系；2)基于多模态大模型输出一致性的噪声伪标签过滤器；3)标记和伪标记样本之间的令牌级对比对齐，以增强特征一致性。

### 主要发现

CORA在Cityscapes数据集上仅需100张标记图像就取得了最先进结果，比基线高出2.3%；在PanNuke数据集上仅用180张标记图像，性能提高了2.4%。

### 结论

CORA通过半监督学习和三个创新组件，实现了在最小监督下的鲁棒推理分割性能，显著降低了高质量标注数据的依赖。

### 翻译

推理分割寻求针对由复杂且通常隐含的指令所引用的目标的像素级精确掩码，需要对场景进行上下文相关的推理。最近的多模态语言模型推动了指令跟随分割的发展，但泛化能力仍然有限。主要瓶颈是整理多样化、高质量的像素标注与丰富的语言监督配对的高成本，这导致在分布变化下性能脆弱。因此，我们提出了CORA，一个半监督推理分割框架，可以从有限的标记数据和大量未标记图像语料库中共同学习。CORA引入了三个主要组件：1)编码对象间空间和上下文关系的条件视觉指令；2)基于多模态大模型在语义等价查询间输出一致性的噪声伪标签过滤器；3)标记和伪标记样本之间的令牌级对比对齐，以增强特征一致性。这些组件使CORA能够在最小监督下执行鲁棒的推理分割，在受限标注设置下优于现有基线。CORA取得了最先进的结果，在城市场景理解基准数据集Cityscapes上仅需100张标记图像，比基线高出2.3%。类似地，在组织病理学数据集PanNuke上，仅用180张标记图像，CORA的性能提高了2.4%。


### 论文摘要

Reasoning segmentation seeks pixel-accurate masks for targets referenced by complex, often implicit instructions, requiring context-dependent reasoning over the scene. Recent multimodal language models have advanced instruction following segmentation, yet generalization remains limited. The key bottleneck is the high cost of curating diverse, high-quality pixel annotations paired with rich linguistic supervision leading to brittle performance under distribution shift. Therefore, we present CORA, a semi-supervised reasoning segmentation framework that jointly learns from limited labeled data and a large corpus of unlabeled images. CORA introduces three main components: 1) conditional visual instructions that encode spatial and contextual relationships between objects; 2) a noisy pseudo-label filter based on the consistency of Multimodal LLM's outputs across semantically equivalent queries; and 3) a token-level contrastive alignment between labeled and pseudo-labeled samples to enhance feature consistency. These components enable CORA to perform robust reasoning segmentation with minimal supervision, outperforming existing baselines under constrained annotation settings. CORA achieves state-of-the-art results, requiring as few as 100 labeled images on Cityscapes, a benchmark dataset for urban scene understanding, surpassing the baseline by $+2.3\%$. Similarly, CORA improves performance by $+2.4\%$ with only 180 labeled images on PanNuke, a histopathology dataset.

---

