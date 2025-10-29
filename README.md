# 今日论文推荐 - 2025-10-29

共 104 篇论文

---

## 1. JiuTian Chuanliu: A Large Spatiotemporal Model for General-purpose Dynamic Urban Sensing

**论文链接:** [http://arxiv.org/abs/2510.23662v1](http://arxiv.org/abs/2510.23662v1)

**作者:** Liangzhe Han, Leilei Sun, Tongyu Zhu, Tao Tao, Jibin Wang, Weifeng Lv

**发布时间:** 2025-10-26

### GPT解析

### 总结

本文提出了一种名为通用动态人类移动性嵌入(GDHME)的框架，用于处理大规模人类移动性数据，发现移动行为背后的潜在语义，并支持各种城市感知任务。

### 背景

人类移动性作为城市感知的窗口，包含丰富的时空信息，反映了居民行为偏好和城市区域功能。现有方法通常从特定角度处理特定任务，导致对人类移动性建模不足，所学知识在下游应用中适用性有限。

### 目的

解决现有方法的局限性，通过时空模型处理大量人类移动性数据，发现潜在语义，支持多种城市感知任务。

### 方法

GDHME框架遵循自监督学习思想，包含两个阶段：第一阶段将人和区域视为动态图中的节点，统一为人-区域-时间交互，使用连续时间编码器计算演化节点表示，并设计自回归自监督任务引导学习；第二阶段利用这些表示支持各种任务。通过基站系统收集大规模移动数据，并构建多任务城市感知基准进行评估。

### 主要发现

离线实验证明GDHME能从大量数据中自动学习有价值的节点特征。该框架已成功部署九天川流大模型，该系统在2023年中国移动全球合作伙伴大会上展示。

### 结论

GDHME框架能有效处理人类移动性数据，提取有价值特征，支持多种城市感知任务，具有广泛的适用性和实用价值。

### 翻译

作为城市感知的窗口，人类移动性包含丰富的时空信息，反映了居民的行为偏好和城市区域的功能。人类移动性分析吸引了众多研究者的关注。然而，现有方法通常从特定角度处理特定任务，导致对人类移动性建模不足，所学知识在各种下游应用中适用性有限。为解决这些挑战，本文提出将大量人类移动性数据输入时空模型，发现移动行为背后的潜在语义，并支持各种城市感知任务。具体来说，通过无处不在的基站系统收集大规模、广泛覆盖的人类移动性数据，并引入了一个名为通用动态人类移动性嵌入(GDHME)的城市感知框架。该框架遵循自监督学习思想，包含两个主要阶段。第一阶段，GDHME将人和区域视为动态图中的节点，将人类移动性数据统一为人-区域-时间交互。在连续时间运行的编码器动态计算演化的节点表示，捕捉人和区域的动态状态。此外，专门设计了自回归自监督任务来引导通用节点嵌入的学习。第二阶段，利用这些表示来支持各种任务。为评估GDHME框架的有效性，作者构建了一个多任务城市感知基准。离线实验证明了GDHME能够从大量数据中自动学习有价值的节点特征。此外，该框架被用于部署九天川流大模型，该系统已在2023年中国移动全球合作伙伴大会上展示。


### 论文摘要

As a window for urban sensing, human mobility contains rich spatiotemporal information that reflects both residents' behavior preferences and the functions of urban areas. The analysis of human mobility has attracted the attention of many researchers. However, existing methods often address specific tasks from a particular perspective, leading to insufficient modeling of human mobility and limited applicability of the learned knowledge in various downstream applications. To address these challenges, this paper proposes to push massive amounts of human mobility data into a spatiotemporal model, discover latent semantics behind mobility behavior and support various urban sensing tasks. Specifically, a large-scale and widely covering human mobility data is collected through the ubiquitous base station system and a framework named General-purpose and Dynamic Human Mobility Embedding (GDHME) for urban sensing is introduced. The framework follows the self-supervised learning idea and contains two major stages. In stage 1, GDHME treats people and regions as nodes within a dynamic graph, unifying human mobility data as people-region-time interactions. An encoder operating in continuous-time dynamically computes evolving node representations, capturing dynamic states for both people and regions. Moreover, an autoregressive self-supervised task is specially designed to guide the learning of the general-purpose node embeddings. In stage 2, these representations are utilized to support various tasks. To evaluate the effectiveness of our GDHME framework, we further construct a multi-task urban sensing benchmark. Offline experiments demonstrate GDHME's ability to automatically learn valuable node features from vast amounts of data. Furthermore, our framework is used to deploy the JiuTian ChuanLiu Big Model, a system that has been presented at the 2023 China Mobile Worldwide Partner Conference.

---

## 2. Pearl: A Foundation Model for Placing Every Atom in the Right Location

**论文链接:** [http://arxiv.org/abs/2510.24670v1](http://arxiv.org/abs/2510.24670v1)

**作者:** Genesis Research Team, Alejandro Dobles, Nina Jovic, Kenneth Leidal, Pranav Murugan, David C. Williams, Drausin Wulsin, Nate Gruver, Christina X. Ji, Korrawat Pruegsanusak, Gianluca Scarpellini, Ansh Sharma, Wojciech Swiderski, Andrea Bootsma, Richard Strong Bowen, Charlotte Chen, Jamin Chen, Marc André Dämgen, Roy Tal Dew, Benjamin DiFrancesco, J. D. Fishman, Alla Ivanova, Zach Kagin, David Li-Bland, Zuli Liu, Igor Morozov, Jeffrey Ouyang-Zhang, Frank C. Pickard IV, Kushal S. Shah, Ben Shor, Gabriel Monteiro da Silva, Maxx Tessmer, Carl Tilbury, Cyr Vetcher, Daniel Zeng, Maruan Al-Shedivat, Aleksandra Faust, Evan N. Feinberg, Michael V. LeVine, Matteus Pan

**发布时间:** 2025-10-28

### GPT解析

### 总结

本研究介绍了Pearl，一种用于蛋白质-配体协同折叠的基础模型，通过三个关键创新解决了深度学习方法在结构预测中的局限性，实现了最先进的性能表现。

### 背景

准确预测蛋白质-配体复合物的三维结构是计算药物发现中的基本挑战，限制了治疗设计的速度和成功率。虽然深度学习方法显示出潜力，但其性能受限于实验数据稀少、架构效率低下、物理无效构象以及无法充分利用可用辅助信息等因素。

### 目的

开发一种能够克服数据稀缺、提高架构效率、确保物理有效性并充分利用辅助信息的蛋白质-配体结构预测模型。

### 方法

作者提出了Pearl（Placing Every Atom in the Right Location）模型，包含三个关键创新：(1) 使用大规模合成数据的训练方法以克服数据稀缺；(2) 融入SO(3)-等变扩散模块的架构，尊重3D旋转对称性；(3) 支持蛋白质和非聚合物组分的通用多链模板系统，以及无条件/条件双模式的可控推理。

### 主要发现

Pearl在蛋白质-配体协同折叠方面建立了新的性能标准，在公共基准测试中超越了AlphaFold3和其他开源基线模型，准确构象生成比次优模型分别提高了14.5%和14.2%。在口袋条件协同折叠模式下，Pearl在严格标准下实现了3.6倍的改进，且模型性能与训练中使用的合成数据集大小直接相关。

### 结论

Pearl通过创新的训练方法、架构设计和推理机制，显著提高了蛋白质-配体复合物结构预测的准确性，合成数据的使用对模型性能有直接积极影响。

### 翻译

准确预测蛋白质-配体复合物的三维结构仍然是计算药物发现中的一个基本挑战，它限制了治疗设计的速度和成功率。深度学习方法最近显示出作为结构预测工具的强大潜力，在多样化的生物分子系统中取得了有希望的准确性。然而，它们的性能和效用受到实验数据稀少、架构效率低下、物理无效构象以及在推理阶段利用可用辅助信息的能力有限等因素的制约。为了解决这些问题，我们引入了Pearl（Placing Every Atom in the Right Location），一种用于大规模蛋白质-配体协同折叠的基础模型。Pearl通过三个关键创新解决了这些挑战：(1) 包括大规模合成数据的训练方法，以克服数据稀缺；(2) 融入SO(3)-等变扩散模块的架构，本质上尊重3D旋转对称性，提高泛化能力和样本效率；(3) 可控推理，包括支持蛋白质和非聚合物组分的通用多链模板系统，以及无条件/条件双模式。Pearl在蛋白质-配体协同折叠方面建立了新的最先进性能。在生成准确和物理有效构象的关键指标上，Pearl在公共基准测试中超越了AlphaFold3和其他开源基线模型，比次优模型分别提高了14.5%和14.2%。在口袋条件协同折叠模式下，Pearl在更严格的标准下实现了3.6倍的改进。最后，我们研究表明模型性能与训练中使用的合成数据集大小直接相关。


### 论文摘要

Accurately predicting the three-dimensional structures of protein-ligand complexes remains a fundamental challenge in computational drug discovery that limits the pace and success of therapeutic design. Deep learning methods have recently shown strong potential as structural prediction tools, achieving promising accuracy across diverse biomolecular systems. However, their performance and utility are constrained by scarce experimental data, inefficient architectures, physically invalid poses, and the limited ability to exploit auxiliary information available at inference. To address these issues, we introduce Pearl (Placing Every Atom in the Right Location), a foundation model for protein-ligand cofolding at scale. Pearl addresses these challenges with three key innovations: (1) training recipes that include large-scale synthetic data to overcome data scarcity; (2) architectures that incorporate an SO(3)-equivariant diffusion module to inherently respect 3D rotational symmetries, improving generalization and sample efficiency, and (3) controllable inference, including a generalized multi-chain templating system supporting both protein and non-polymeric components as well as dual unconditional/conditional modes. Pearl establishes a new state-of-the-art performance in protein-ligand cofolding. On the key metric of generating accurate (RMSD < 2 \r{A}) and physically valid poses, Pearl surpasses AlphaFold 3 and other open source baselines on the public Runs N' Poses and PoseBusters benchmarks, delivering 14.5% and 14.2% improvements, respectively, over the next best model. In the pocket-conditional cofolding regime, Pearl delivers $3.6\times$ improvement on a proprietary set of challenging, real-world drug targets at the more rigorous RMSD < 1 \r{A} threshold. Finally, we demonstrate that model performance correlates directly with synthetic dataset size used in training.

---

## 3. Advancing site-specific disease and pest management in precision agriculture: From reasoning-driven foundation models to adaptive, feedback-based learning

**论文链接:** [http://arxiv.org/abs/2510.24650v1](http://arxiv.org/abs/2510.24650v1)

**作者:** Nitin Rai, Daeun, Choi, Nathan S. Boyd, Arnold W. Schumann

**发布时间:** 2025-10-28

**备注:** 26 pages, 8 figures, and 2 tables

### GPT解析

### 总结

该综述探讨了基础模型(FMs)在农业特定地点疾病管理(SSDM)中的应用，重点关注大型语言模型(LLMs)和视觉语言模型(VLMs)的发展及其在自适应学习、强化学习和数字孪生框架中的作用。

### 背景

农业特定地点疾病管理通过机器学习和深度学习在实时计算机视觉方面取得快速进展，研究从手工特征提取发展到大规模自动化特征学习，基础模型为作物疾病数据处理带来全新方式。

### 目的

筛选约40篇关于基础模型在SSDM中应用的论文，讨论其在自适应学习、强化学习和数字孪生框架中的作用，并分析当前发展状况和挑战。

### 方法

分析基础模型如何整合视觉和文本数据，解释症状文本，推理症状-管理关系，支持交互式问答，以及机器人和自适应学习如何支持基于现场的疾病管理。

### 主要发现

基础模型在2023-24年文献激增；视觉语言模型发表数量比大型语言模型多5-10倍；强化学习和自适应学习在智能喷洒方面仍处于起步阶段；数字孪生可虚拟模拟目标喷洒；解决模拟到现实的差距对实际部署至关重要；人机协作仍有限；具有实时反馈的多模态基础模型将推动下一代SSDM。

### 结论

基础模型特别是视觉语言模型在农业特定地点疾病管理中展现出巨大潜力，但仍需解决模拟到现实的差距和人机协作的局限性等挑战。

### 翻译

作物特定地点疾病管理通过机器学习和深度学习在实时计算机视觉方面取得了快速进展。研究从手工特征提取发展到大规模自动化特征学习。随着基础模型的出现，作物疾病数据现在以全新的方式被处理。与传统神经网络不同，基础模型整合视觉和文本数据，解释文本中的症状，推理症状-管理关系，并为种植者和教育者支持交互式问答。机器人和自适应学习进一步支持基于现场的疾病管理。本综述筛选了约40篇关于基础模型在特定地点疾病管理中应用的论文，重点关注大型语言模型和视觉语言模型，并讨论它们在自适应学习、强化学习和用于目标喷洒的数字孪生框架中的作用。主要发现：基础模型在2023-24年文献激增中越来越受欢迎；视觉语言模型的发表数量比大型语言模型多5-10倍；强化学习和自适应学习在智能喷洒方面仍处于起步阶段；带有强化学习的数字孪生可以虚拟模拟目标喷洒；解决模拟到现实的差距对实际部署至关重要；人机协作仍然有限，特别是在人机交互方法中，机器人检测早期症状，人类验证不确定情况；具有实时反馈的多模态基础模型将推动下一代特定地点疾病管理。


### 论文摘要

Site-specific disease management (SSDM) in crops has advanced rapidly through machine and deep learning (ML and DL) for real-time computer vision. Research evolved from handcrafted feature extraction to large-scale automated feature learning. With foundation models (FMs), crop disease datasets are now processed in fundamentally new ways. Unlike traditional neural networks, FMs integrate visual and textual data, interpret symptoms in text, reason about symptom-management relationships, and support interactive QA for growers and educators. Adaptive and imitation learning in robotics further enables field-based disease management. This review screened approx. 40 articles on FM applications for SSDM, focusing on large-language models (LLMs) and vision-language models (VLMs), and discussing their role in adaptive learning (AL), reinforcement learning (RL), and digital twin frameworks for targeted spraying. Key findings: (a) FMs are gaining traction with surging literature in 2023-24; (b) VLMs outpace LLMs, with a 5-10x increase in publications; (c) RL and AL are still nascent for smart spraying; (d) digital twins with RL can simulate targeted spraying virtually; (e) addressing the sim-to-real gap is critical for real-world deployment; (f) human-robot collaboration remains limited, especially in human-in-the-loop approaches where robots detect early symptoms and humans validate uncertain cases; (g) multi-modal FMs with real-time feedback will drive next-gen SSDM. For updates, resources, and contributions, visit, https://github.com/nitin-dominic/AgriPathogenDatabase, to submit papers, code, or datasets.

---

## 4. Generative AI for Healthcare: Fundamentals, Challenges, and Perspectives

**论文链接:** [http://arxiv.org/abs/2510.24551v1](http://arxiv.org/abs/2510.24551v1)

**作者:** Gang Chen, Changshuo Liu, Gene Anne Ooi, Marcus Tan, Zhongle Xie, Jianwei Yin, James Wei Luen Yip, Wenqiao Zhang, Jiaqi Zhu, Beng Chin Ooi

**发布时间:** 2025-10-28

### GPT解析

### 总结

该论文提出了一种以数据为中心的医疗保健生成式人工智能系统设计范式，通过构建医疗数据生态系统作为基础支撑，实现高质量、有效的医疗保健服务。

### 背景

生成式人工智能(GenAI)正在全球范围内迅速发展，为医疗保健领域带来变革性机会。从大型语言模型用于临床笔记合成和对话式辅助，到整合医学影像、电子健康记录和基因组数据的多模态系统用于决策支持，GenAI正在改变医学实践和医疗保健提供方式。

### 目的

提出一种数据中心的范式，用于医疗保健领域生成式人工智能系统的设计和部署，解决GenAI在医疗保健应用中的挑战。

### 方法

重新定位数据生命周期，将医疗数据生态系统作为生成式医疗保健系统的基础支撑。该生态系统支持多样化医疗数据和知识的集成、表示和检索，通过语义向量搜索和上下文查询等数据处理管道，为上游模型组件和下游临床应用提供支持。

### 主要发现

生成式人工智能在医疗保健领域部署需要深入了解医疗保健任务以及可以实现和不能实现的目标，医疗数据生态系统是解决这一挑战的关键。

### 结论

通过构建可持续的医疗数据生态系统，不仅能为基础模型提供高质量、多模态数据用于大规模预训练和领域特定微调，还能作为知识检索后端支持特定任务推理，使GenAI能够高质量、有效地部署医疗保健服务。

### 翻译

生成式人工智能(GenAI)正在席卷全球。它承诺为推进和颠覆现有实践（包括医疗保健）带来变革性机会。从用于临床笔记合成和对话式辅助的大型语言模型(LLMs)，到整合医学影像、电子健康记录和基因组数据用于决策支持的多模态系统，GenAI正在改变医学实践和医疗保健的提供方式，如诊断和个性化治疗，有潜力减轻临床医生的认知负担，从而改善整体医疗保健服务。然而，GenAI在医疗保健领域的部署需要对医疗保健任务以及可实现和不可实现的目标有深入理解。在本文中，我们提出了一个以数据为中心的范式，用于医疗保健领域生成式人工智能系统的设计和部署。具体而言，我们通过将医疗数据生态系统作为生成式医疗保健系统的基础支撑物，重新定位了数据生命周期。该生态系统旨在可持续地支持多样化医疗数据和知识的集成、表示和检索。通过有效和高效的数据处理管道，如语义向量搜索和上下文查询，它使生成式人工智能能够为上游模型组件和下游临床应用提供支持。最终，它不仅为基础模型提供高质量、多模态数据用于大规模预训练和领域特定微调，还充当知识检索后端，通过智能层支持特定任务的推理。该生态系统使生成式人工智能能够高质量、有效地部署医疗保健服务。


### 论文摘要

Generative Artificial Intelligence (GenAI) is taking the world by storm. It promises transformative opportunities for advancing and disrupting existing practices, including healthcare. From large language models (LLMs) for clinical note synthesis and conversational assistance to multimodal systems that integrate medical imaging, electronic health records, and genomic data for decision support, GenAI is transforming the practice of medicine and the delivery of healthcare, such as diagnosis and personalized treatments, with great potential in reducing the cognitive burden on clinicians, thereby improving overall healthcare delivery. However, GenAI deployment in healthcare requires an in-depth understanding of healthcare tasks and what can and cannot be achieved. In this paper, we propose a data-centric paradigm in the design and deployment of GenAI systems for healthcare. Specifically, we reposition the data life cycle by making the medical data ecosystem as the foundational substrate for generative healthcare systems. This ecosystem is designed to sustainably support the integration, representation, and retrieval of diverse medical data and knowledge. With effective and efficient data processing pipelines, such as semantic vector search and contextual querying, it enables GenAI-powered operations for upstream model components and downstream clinical applications. Ultimately, it not only supplies foundation models with high-quality, multimodal data for large-scale pretraining and domain-specific fine-tuning, but also serves as a knowledge retrieval backend to support task-specific inference via the agentic layer. The ecosystem enables the deployment of GenAI for high-quality and effective healthcare delivery.

---

## 5. Affordance Representation and Recognition for Autonomous Agents

**论文链接:** [http://arxiv.org/abs/2510.24459v1](http://arxiv.org/abs/2510.24459v1)

**作者:** Habtom Kahsay Gidey, Niklas Huber, Alexander Lenz, Alois Knoll

**发布时间:** 2025-10-28

### GPT解析

### 总结

该研究提出了一种从结构化数据构建世界模型的新方法，解决了软件代理在适应不断变化的Web环境时面临的两个关键挑战。

### 背景

软件代理的自主性依赖于其从结构化数据（如网页DOM和Web服务语义描述）构建内部世界模型的能力。然而，原始HTML的冗长性和硬编码API集成的静态性构成了重大障碍。

### 目的

开发一种模式语言，使软件代理能够高效构建和维护准确的世界模型，从而实现跨Web及其扩展资源的可扩展、自适应和互操作性自动化。

### 方法

提出两种互补的架构模式：1) DOM转换模式，将冗长的原始DOM提炼为紧凑的、任务相关的表示；2) 超媒体功能识别模式，通过解析语义描述动态发现和集成未知Web服务的能力。

### 主要发现

通过结合这两种模式，软件代理能够克服数据冗余和服务动态变化的挑战，构建和维护准确的世界模型。

### 结论

所提出的模式语言为构建能够高效适应不断演化的数字环境的智能代理提供了强大框架，增强了自动化系统的可扩展性、适应性和互操作性。

### 翻译

软件代理的自主性根本上取决于其从定义其数字环境的结构化数据（如网页的文档对象模型和Web服务的语义描述）构建可行的内部世界模型的能力。然而，从原始结构化数据构建此世界模型存在两个关键挑战：原始HTML的冗长性使其在计算上难以被基础模型直接使用，而硬编码API集成的静态性质阻止了代理适应不断发展的服务。本文介绍了一种从结构化数据进行世界建模的模式语言，提出了两种互补的架构模式。DOM转换模式通过将冗长的原始DOM提炼为紧凑的、任务相关的表示或为代理推理核心优化的世界模型，解决了网页复杂性的挑战。同时，超媒体功能识别模式使代理能够通过解析标准化的语义描述来动态丰富其世界模型，从而在运行时发现和集成未知Web服务的能力。这些模式共同提供了一个强大的框架，用于构建能够高效构建和维护准确世界模型的代理，从而实现Web及其扩展资源的可扩展、自适应和互操作性自动化。


### 论文摘要

The autonomy of software agents is fundamentally dependent on their ability to construct an actionable internal world model from the structured data that defines their digital environment, such as the Document Object Model (DOM) of web pages and the semantic descriptions of web services. However, constructing this world model from raw structured data presents two critical challenges: the verbosity of raw HTML makes it computationally intractable for direct use by foundation models, while the static nature of hardcoded API integrations prevents agents from adapting to evolving services.   This paper introduces a pattern language for world modeling from structured data, presenting two complementary architectural patterns. The DOM Transduction Pattern addresses the challenge of web page complexity by distilling} a verbose, raw DOM into a compact, task-relevant representation or world model optimized for an agent's reasoning core. Concurrently, the Hypermedia Affordances Recognition Pattern enables the agent to dynamically enrich its world model by parsing standardized semantic descriptions to discover and integrate the capabilities of unknown web services at runtime. Together, these patterns provide a robust framework for engineering agents that can efficiently construct and maintain an accurate world model, enabling scalable, adaptive, and interoperable automation across the web and its extended resources.

---

## 6. Rethinking Visual Intelligence: Insights from Video Pretraining

**论文链接:** [http://arxiv.org/abs/2510.24448v1](http://arxiv.org/abs/2510.24448v1)

**作者:** Pablo Acuaviva, Aram Davtyan, Mariam Hassan, Sebastian Stapf, Ahmad Rahimi, Alexandre Alahi, Paolo Favaro

**发布时间:** 2025-10-28

**备注:** Updated version from preprint arXiv:2506.07280 (Gen2Gen) focused on  visual intelligence. This work can be considered as v2

### GPT解析

### 总结

研究表明视频扩散模型在视觉任务上比语言模型更高效，视频预训练提供的归纳偏置有助于构建视觉基础模型。

### 背景

大型语言模型在大规模预训练后能够在语言领域快速适应新问题，但这种成功在视觉领域并未同样有效，包括LLMs在内的模型在组合理解、样本效率和通用问题解决方面仍然存在困难。

### 目的

研究视频扩散模型作为弥合语言模型与视觉模型差距的有前途方向，测试视频预训练是否能提供支持广泛任务适应性的归纳偏置。

### 方法

在时空数据上预训练视频扩散模型，使其具有结构和动态性的强归纳偏置；设计对照评估，让预训练的LLM和VDM都配备轻量级适配器，并以自然方式呈现任务；在多个基准测试中评估，包括ARC-AGI、ConceptARC、视觉游戏、路线规划和细胞自动机。

### 主要发现

VDMs比语言对应模型具有更高的数据效率；视频预训练提供的归纳偏置支持视觉基础模型的进展。

### 结论

视频预训练提供了归纳偏置，支持向视觉基础模型进展。

### 翻译

大型语言模型已经证明，大规模预训练使系统能够在语言领域以少量监督快速适应新问题。然而，这种成功并没有同样有效地转化为视觉领域，包括LLMs在内的模型在组合理解、样本效率和通用问题解决方面仍然存在困难。我们研究视频扩散模型作为弥合这一差距的有前途方向。在时空数据上的预训练赋予这些模型结构和动态性的强归纳偏置，我们假设这可以支持广泛的任务适应性。为了验证这一点，我们设计了一个对照评估，让预训练的LLM和预训练的VDM都配备轻量级适配器，并以它们自然的方式呈现任务。在包括ARC-AGI、ConceptARC、视觉游戏、路线规划和细胞自动机在内的基准测试中，VDMs比其语言对应模型表现出更高的数据效率。综上所述，我们的结果表明视频预训练提供了归纳偏置，支持向视觉基础模型的进展。


### 论文摘要

Large language models (LLMs) have demonstrated that large-scale pretraining enables systems to adapt rapidly to new problems with little supervision in the language domain. This success, however, has not translated as effectively to the visual domain, where models, including LLMs, continue to struggle with compositional understanding, sample efficiency, and general-purpose problem-solving. We investigate Video Diffusion Models (VDMs) as a promising direction for bridging this gap. Pretraining on spatiotemporal data endows these models with strong inductive biases for structure and dynamics, which we hypothesize can support broad task adaptability. To test this, we design a controlled evaluation in which both a pretrained LLM and a pretrained VDM are equipped with lightweight adapters and presented with tasks in their natural modalities. Across benchmarks including ARC-AGI, ConceptARC, visual games, route planning, and cellular automata, VDMs demonstrate higher data efficiency than their language counterparts. Taken together, our results indicate that video pretraining offers inductive biases that support progress toward visual foundation models.

---

## 7. A Unified Geometric Space Bridging AI Models and the Human Brain

**论文链接:** [http://arxiv.org/abs/2510.24342v1](http://arxiv.org/abs/2510.24342v1)

**作者:** Silin Chen, Yuzhong Chen, Zifan Wang, Junhao Wang, Zifeng Jia, Keith M Kendrick, Tuo Zhang, Lin Zhao, Dezhong Yao, Tianming Liu, Xi Jiang

**发布时间:** 2025-10-28

### GPT解析

### 总结

本研究提出了'类脑空间'概念，通过映射AI模型的内在空间注意力拓扑结构到人类功能性脑网络，实现了跨模态AI模型的统一比较框架，揭示了机器与大脑之间的深层组织原则。

### 背景

几十年来，神经科学家和计算机科学家一直致力于理解并构建智能。现代人工神经网络在语言、感知和推理方面已可与人类媲美，但这些系统是否像大脑一样组织信息仍然在很大程度上未知。现有的大脑-AI对齐研究虽展示了两个系统间的对应关系，但比较局限于特定输入和任务，缺乏跨模态AI模型内在组织的共同比较基础。

### 目的

引入'类脑空间'概念，这是一个统一的几何空间，无论输入模态、任务或感觉域如何，都能通过将AI模型内在的空间注意力拓扑结构映射到典型人类功能性脑网络上，实现对AI模型的精确定位和比较。

### 方法

对151个基于Transformer的模型进行广泛分析，涵盖最先进的大型视觉模型、大型语言模型和大型多模态模型。

### 主要发现

在类脑空间中存在连续的弧形几何结构，反映类脑性的逐渐增加；不同模型表现出不同的分布模式，与不同程度的类脑性相关，这种模式不仅受模态影响，还受预训练范式是否强调全局语义抽象以及位置编码方案是否促进不同模态间深度融合的影响；此外，模型的类脑程度和其下游任务表现并非完全相同。

### 结论

类脑空间提供了第一个跨领域的定位、量化和比较智能的统一框架，揭示了连接机器和大脑的深层组织原则。

### 翻译

几十年来，神经科学家和计算机科学家一直怀有共同的志向：理解智能并构建它。现代人工神经网络在语言、感知和推理方面现在可以与人类相媲美，但这些人工系统是否像大脑一样组织信息仍然在很大程度上未知。现有的大脑-AI对齐研究已经展示了两个系统之间的惊人对应关系，但这样的比较仍然局限于特定的输入和任务，没有提供共同的基础来比较不同模态（视觉、语言或多模态）的AI模型是如何内在组织的。在这里，我们引入了一个突破性的概念：类脑空间：一个统一的几何空间，通过将AI模型内在的空间注意力拓扑组织映射到典型的人类功能性脑网络上，无论输入模态、任务或感觉域如何，每个AI模型都可以在这个空间中被精确定位和比较。我们对151个基于Transformer的模型进行了广泛分析，这些模型涵盖了最先进的大型视觉模型、大型语言模型和大型多模态模型，在这个空间中发现了一个连续的弧形几何结构，反映了类脑性的逐渐增加；不同的模型在这个几何结构中表现出不同的分布模式，与不同程度的类脑性相关，这种模式不仅受模态影响，还受预训练范式是否强调全局语义抽象以及位置编码方案是否促进不同模态间的深度融合的影响。此外，模型的类脑程度和其下游任务表现并非'完全相同'。类脑空间提供了第一个跨领域的定位、量化和比较智能的统一框架，揭示了连接机器和大脑的深层组织原则。


### 论文摘要

For decades, neuroscientists and computer scientists have pursued a shared ambition: to understand intelligence and build it. Modern artificial neural networks now rival humans in language, perception, and reasoning, yet it is still largely unknown whether these artificial systems organize information as the brain does. Existing brain-AI alignment studies have shown the striking correspondence between the two systems, but such comparisons remain bound to specific inputs and tasks, offering no common ground for comparing how AI models with different kinds of modalities-vision, language, or multimodal-are intrinsically organized. Here we introduce a groundbreaking concept of Brain-like Space: a unified geometric space in which every AI model can be precisely situated and compared by mapping its intrinsic spatial attention topological organization onto canonical human functional brain networks, regardless of input modality, task, or sensory domain. Our extensive analysis of 151 Transformer-based models spanning state-of-the-art large vision models, large language models, and large multimodal models uncovers a continuous arc-shaped geometry within this space, reflecting a gradual increase of brain-likeness; different models exhibit distinct distribution patterns within this geometry associated with different degrees of brain-likeness, shaped not merely by their modality but by whether the pretraining paradigm emphasizes global semantic abstraction and whether the positional encoding scheme facilitates deep fusion across different modalities. Moreover, the degree of brain-likeness for a model and its downstream task performance are not "identical twins". The Brain-like Space provides the first unified framework for situating, quantifying, and comparing intelligence across domains, revealing the deep organizational principles that bridge machines and the brain.

---

## 8. Vanish into Thin Air: Cross-prompt Universal Adversarial Attacks for SAM2

**论文链接:** [http://arxiv.org/abs/2510.24195v1](http://arxiv.org/abs/2510.24195v1)

**作者:** Ziqi Zhou, Yifan Hu, Yufei Song, Zijing Li, Shengshan Hu, Leo Yu Zhang, Dezhong Yao, Long Zheng, Hai Jin

**发布时间:** 2025-10-28

**备注:** Accepted by NeurIPS 2025

### GPT解析

### 总结

本文提出了UAP-SAM2，第一个针对SAM2的跨提示通用对抗攻击，通过双语义偏差驱动，有效解决了SAM2架构差异带来的挑战。

### 背景

图像分割基础模型SAM对对抗样本存在脆弱性，其后续模型SAM2在视频分割方面表现出强大的泛化能力，但其鲁棒性尚未被探索。

### 目的

分析现有攻击在SAM和SAM2之间的性能差距，并提出一种有效的对抗攻击方法针对SAM2。

### 方法

提出UAP-SAM2方法，包括：1) 设计目标扫描策略，将每帧划分为k个区域，每个区域随机分配提示，减少优化过程中的提示依赖性；2) 设计双语义偏差框架，通过扭曲当前帧内语义和破坏连续帧间语义一致性来优化UAP。

### 主要发现

现有攻击在SAM和SAM2之间存在性能差距，主要由于SAM2架构差异带来的两个关键挑战：来自提示的方向性指导和连续帧之间的语义纠缠。

### 结论

UAP-SAM2在两个分割任务上的六个数据集实验中表现出有效性，以较大优势显著优于最先进的攻击方法。

### 翻译

最近的研究揭示了图像分割基础模型SAM对对抗样本的脆弱性。其后续模型SAM2因其强大的视频分割泛化能力而受到广泛关注。然而，其鲁棒性尚未被探索，目前还不清楚现有的针对SAM的攻击是否可以直接转移到SAM2上。在本文中，我们首先分析了现有攻击在SAM和SAM2之间的性能差距，并指出了它们架构差异带来的两个关键挑战：来自提示的方向性指导和连续帧之间的语义纠缠。为解决这些问题，我们提出了UAP-SAM2，这是第一个由双语义偏差驱动的针对SAM2的跨提示通用对抗攻击。为实现跨提示可转移性，我们首先设计了一个目标扫描策略，将每帧划分为k个区域，每个区域随机分配一个提示，以减少优化过程中的提示依赖性。为提高有效性，我们设计了一个双语义偏差框架，通过扭曲当前帧内的语义和破坏连续帧之间的语义一致性来优化UAP。在两个分割任务上的六个数据集进行的广泛实验证明了所提方法对SAM2的有效性。比较结果显示，UAP-SAM2以较大优势显著优于最先进的(SOTA)攻击。


### 论文摘要

Recent studies reveal the vulnerability of the image segmentation foundation model SAM to adversarial examples. Its successor, SAM2, has attracted significant attention due to its strong generalization capability in video segmentation. However, its robustness remains unexplored, and it is unclear whether existing attacks on SAM can be directly transferred to SAM2. In this paper, we first analyze the performance gap of existing attacks between SAM and SAM2 and highlight two key challenges arising from their architectural differences: directional guidance from the prompt and semantic entanglement across consecutive frames. To address these issues, we propose UAP-SAM2, the first cross-prompt universal adversarial attack against SAM2 driven by dual semantic deviation. For cross-prompt transferability, we begin by designing a target-scanning strategy that divides each frame into k regions, each randomly assigned a prompt, to reduce prompt dependency during optimization. For effectiveness, we design a dual semantic deviation framework that optimizes a UAP by distorting the semantics within the current frame and disrupting the semantic consistency across consecutive frames. Extensive experiments on six datasets across two segmentation tasks demonstrate the effectiveness of the proposed method for SAM2. The comparative results show that UAP-SAM2 significantly outperforms state-of-the-art (SOTA) attacks by a large margin.

---

## 9. Blindfolded Experts Generalize Better: Insights from Robotic Manipulation and Videogames

**论文链接:** [http://arxiv.org/abs/2510.24194v1](http://arxiv.org/abs/2510.24194v1)

**作者:** Ev Zisselman, Mirco Mutti, Shelly Francis-Meretzki, Elisei Shafer, Aviv Tamar

**发布时间:** 2025-10-28

### GPT解析

### 总结

行为克隆是一种从演示中学习序列决策的简单而有效技术，本文提出让演示者在部分信息不足的情况下进行演示，发现这种'蒙眼'专家的克隆方法在泛化到未见任务时表现更好。

### 背景

行为克隆已成为物理世界基础模型的核心，实现泛化需要无数任务的演示。通常，具有完整任务信息的人类专家会演示最优行为。

### 目的

研究向演示者隐藏部分任务信息是否能提高行为克隆的泛化能力。

### 方法

提出'蒙眼专家'方法，即向任务演示者隐藏部分信息，迫使其使用非平凡的探索策略来解决任务。在真实世界机器人插销任务和Procgen基准视频游戏上进行实验，并进行理论分析。

### 主要发现

克隆'蒙眼专家'的行为比克隆完全信息专家的行为在未见任务上泛化能力更好。理论分析表明泛化误差与演示者可用的任务信息量和演示任务数量有关，使用更少演示任务时克隆蒙眼专家能实现更好泛化。

### 结论

理论和实践均表明，使用更少的演示任务，克隆蒙眼专家能够实现更好的泛化效果。

### 翻译

行为克隆是一种简单而有效的技术，用于从演示中学习序列决策。最近，它已成为物理世界基础模型的核心，其中实现泛化需要无数任务的演示。通常，具有任务完整信息的人类专家会演示（几乎）最优的行为。在本文中，我们提出向演示者隐藏任务的部分信息。这种'蒙眼'专家被迫采用非平凡的探索来解决任务。我们证明，克隆蒙眼专家比完全信息的专家在未见任务上泛化能力更好。我们在有限人类演示下的真实世界机器人插销任务以及Procgen基准的视频游戏上进行了实验。此外，我们通过理论分析支持了这一发现，理论和实践都表明，用更少的演示任务克隆蒙眼专家能更好地泛化。项目页面包含视频和代码：https://sites.google.com/view/blindfoldedexperts/home


### 论文摘要

Behavioral cloning is a simple yet effective technique for learning sequential decision-making from demonstrations. Recently, it has gained prominence as the core of foundation models for the physical world, where achieving generalization requires countless demonstrations of a multitude of tasks. Typically, a human expert with full information on the task demonstrates a (nearly) optimal behavior. In this paper, we propose to hide some of the task's information from the demonstrator. This ``blindfolded'' expert is compelled to employ non-trivial exploration to solve the task. We show that cloning the blindfolded expert generalizes better to unseen tasks than its fully-informed counterpart. We conduct experiments of real-world robot peg insertion tasks with (limited) human demonstrations, alongside videogames from the Procgen benchmark. Additionally, we support our findings with theoretical analysis, which confirms that the generalization error scales with $\sqrt{I/m}$, where $I$ measures the amount of task information available to the demonstrator, and $m$ is the number of demonstrated tasks. Both theory and practice indicate that cloning blindfolded experts generalizes better with fewer demonstrated tasks. Project page with videos and code: https://sites.google.com/view/blindfoldedexperts/home

---

## 10. BLM$_1$: A Boundless Large Model for Cross-Space, Cross-Task, and Cross-Embodiment Learning

**论文链接:** [http://arxiv.org/abs/2510.24161v1](http://arxiv.org/abs/2510.24161v1)

**作者:** Wentao Tan, Bowen Wang, Heng Zhi, Chenyu Liu, Zhe Li, Jian Liu, Zengrong Lin, Yukun Dai, Yipeng Chen, Wenjie Yang, Enci Xie, Hao Xue, Baixu Ji, Chen Xu, Zhibin Wang, Tianshi Wang, Lei Zhu, Heng Tao Shen

**发布时间:** 2025-10-28

### GPT解析

### 总结

该研究提出了Boundless Large Model (BLM₁)，一个多模态空间基础模型，能够在数字和物理空间无缝操作，实现跨具身和任务泛化。通过两阶段训练范式，BLM₁整合了跨空间迁移、跨任务学习和跨具身泛化能力，在数字和物理基准测试中超越了多种模型家族。

### 背景

多模态大语言模型(MLLMs)在视觉-语言推理方面取得进展并应用于具身智能体，但仍存在显著局限性：MLLMs在数字-物理空间和具身形式间泛化能力差；视觉-语言-动作模型(VLAs)产生低级行动但缺乏稳健的高层次具身推理；大多数具身大语言模型(ELLMs)局限于数字空间，对物理世界泛化能力差。缺乏能够在数字和物理空间无缝操作、跨具身和任务泛化的统一模型。

### 目的

开发一个能够在数字和物理空间无缝操作、跨具身和任务泛化的统一模型。

### 方法

提出Boundless Large Model (BLM₁)，一个多模态空间基础模型，保留指令跟随和推理能力，整合具身知识，支持稳健的跨具身控制。通过两阶段训练范式整合三种关键能力：跨空间迁移、跨任务学习和跨具身泛化。第一阶段通过精心挑选的数字语料库将具身知识注入MLLM，同时保持语言能力；第二阶段通过意图桥接接口训练策略模块，从MLLM中提取高层语义来指导控制，无需微调MLLM主干。使用自收集的跨具身演示套件，涵盖四种机器人具身和六种渐进式挑战任务。

### 主要发现

在数字和物理基准测试中评估显示，单个BLM₁实例优于四种模型家族（MLLMs、ELLMs、VLAs和GMLMs），在数字任务中实现约6%的提升，在物理任务中实现约3%的提升。

### 结论

BLM₁是一个有效的多模态空间基础模型，能够整合具身知识并实现跨空间、跨任务和跨具身的泛化能力，为具身智能体提供了新的发展方向。

### 翻译

多模态大语言模型(MLLMs)已推进视觉-语言推理，并越来越多地部署在具身智能体中。然而，仍存在显著局限性：MLLMs在数字-物理空间和具身形式间泛化能力差；视觉-语言-动作模型(VLAs)产生低级行动但缺乏稳健的高层次具身推理；大多数具身大语言模型(ELLMs)局限于数字空间，对物理世界泛化能力差。因此，能够在数字和物理空间无缝操作、跨具身和任务泛化的统一模型仍然缺失。我们引入了Boundless Large Model (BLM₁)，一个多模态空间基础模型，保留了指令跟随和推理能力，整合了具身知识，并支持稳健的跨具身控制。BLM₁通过两阶段训练范式整合了三种关键能力——跨空间迁移、跨任务学习和跨具身泛化。第一阶段通过精心挑选的数字语料库将具身知识注入MLLM，同时保持语言能力。第二阶段通过意图桥接接口训练策略模块，从MLLM中提取高层语义来指导控制，无需微调MLLM主干。这一过程得到了一个自收集的跨具身演示套件的支持，涵盖四种机器人具身和六种渐进式挑战任务。在数字和物理基准测试中的评估显示，单个BLM₁实例优于四种模型家族——MLLMs、ELLMs、VLAs和GMLMs，在数字任务中实现约6%的提升，在物理任务中实现约3%的提升。


### 论文摘要

Multimodal large language models (MLLMs) have advanced vision-language reasoning and are increasingly deployed in embodied agents. However, significant limitations remain: MLLMs generalize poorly across digital-physical spaces and embodiments; vision-language-action models (VLAs) produce low-level actions yet lack robust high-level embodied reasoning; and most embodied large language models (ELLMs) are constrained to digital-space with poor generalization to the physical world. Thus, unified models that operate seamlessly across digital and physical spaces while generalizing across embodiments and tasks remain absent. We introduce the \textbf{Boundless Large Model (BLM$_1$)}, a multimodal spatial foundation model that preserves instruction following and reasoning, incorporates embodied knowledge, and supports robust cross-embodiment control. BLM$_1$ integrates three key capabilities -- \textit{cross-space transfer, cross-task learning, and cross-embodiment generalization} -- via a two-stage training paradigm. Stage I injects embodied knowledge into the MLLM through curated digital corpora while maintaining language competence. Stage II trains a policy module through an intent-bridging interface that extracts high-level semantics from the MLLM to guide control, without fine-tuning the MLLM backbone. This process is supported by a self-collected cross-embodiment demonstration suite spanning four robot embodiments and six progressively challenging tasks. Evaluations across digital and physical benchmarks show that a single BLM$_1$ instance outperforms four model families -- MLLMs, ELLMs, VLAs, and GMLMs -- achieving $\sim\!\textbf{6%}$ gains in digital tasks and $\sim\!\textbf{3%}$ in physical tasks.

---

## 11. Global Chlorophyll-\textit{a} Retrieval algorithm from Sentinel 2 Using Residual Deep Learning and Novel Machine Learning Water Classification

**论文链接:** [http://arxiv.org/abs/2510.24124v1](http://arxiv.org/abs/2510.24124v1)

**作者:** Yotam Sherf, Bar Efrati, Gabriel Rozman, Moshe Harel

**发布时间:** 2025-10-28

### GPT解析

### 总结

本研究提出了一种名为全球水分类器(GWC)的监督式机器学习分类器，用于全球范围内水体识别和叶绿素-a浓度反演，并通过残差CNN校正提高了反演精度。

### 背景

传统的水体识别和叶绿素-a浓度反演面临多种干扰因素，如云、太阳耀斑、雪、冰、水生植被、陆地和沉积物等，需要一种能够全球应用且稳健的方法。

### 目的

开发一种能够全球范围内准确识别水体并反演叶绿素-a浓度的方法，克服地理和气候条件的限制。

### 方法

1. 使用Sen2Cor校正的Sentinel-2地表反射率数据训练全球水分类器(GWC)
2. 基于近100个全球分布的内陆水体样本进行训练
3. 使用XGBoost回归器进行叶绿素-a浓度反演
4. 添加残差CNN(RCNN)校正阶段提高反演精度
5. 在867个水体上进行测试验证

### 主要发现

1. GWC能够有效区分不同叶绿素-a水平的水体与非水体光谱
2. GWC表现出地理位置稳定的性能
3. GWC正标记的场景产生的叶绿素-a反演值更准确
4. 残差CNN校正阶段显著提高了反演精度
5. 最终算法在测试中表现出稳健性、可扩展性和全球可转移性

### 结论

全球水分类器结合残差CNN校正的方法能够准确、稳健地进行全球水体识别和叶绿素-a浓度反演，无需针对不同地区进行额外调优，具有很高的应用价值。

### 翻译

我们提出了全球水分类器(GWC)，一种监督式、地理范围广泛的机器学习分类器，基于Sen2Cor校正的Sentinel-2地表反射率数据训练。使用近100个全球分布的内陆水体，GWC能够区分不同叶绿素-a水平的水体光谱与非水体光谱(云、太阳耀斑、雪、冰、水生植被、陆地和沉积物)，并表现出地理位置稳定的性能。在此基础模型上，我们使用匹配的Sentinel-2反射率数据与美国地质调查局(USGS)AquaMatch现场数据集进行叶绿素-a反演，覆盖了多样的地理和水文条件。我们在13626个匹配点上训练了一个XGBoost回归器。GWC正标记的场景持续优于负标记场景，并产生更准确的叶绿素-a反演值，这证实了分类器在减少各种干扰方面的优势。接下来，回归预测的残差分析揭示了结构化误差，促使我们添加了残差CNN(RCNN)校正阶段。我们添加了一个基于归一化残差训练的CNN残差阶段，取得了显著改进。我们的算法在867个水体上进行了测试，超过2000个预测，叶绿素-a值高达1000毫克每立方米，实现了R² = 0.79，平均绝对误差 = 13.52毫克每立方米，斜率 = 0.91，证明了其稳健、可扩展且全球可转移的性能，无需额外调优。


### 论文摘要

We present the Global Water Classifier (GWC), a supervised, geospatially extensive Machine Learning (ML) classifier trained on Sen2Cor corrected Sentinel-2 surface reflectance data. Using nearly 100 globally distributed inland water bodies, GWC distinguishes water across Chlorophyll-a (Chla) levels from non-water spectra (clouds, sun glint, snow, ice, aquatic vegetation, land and sediments) and shows geographically stable performance.   Building on this foundation model, we perform Chla retrieval based on a matchup Sentinel-2 reflectance data with the United States Geological Survey (USGS) AquaMatch in-situ dataset, covering diverse geographical and hydrological conditions.   We train an XGBoost regressor on 13626 matchup points. The positive labeled scenes by the GWC consistently outperform the negatives and produce more accurate Chla retrieval values, which confirms the classifiers advantage in reducing various interferences.   Next, residual analysis of the regression predictions revealed structured errors, motivating a residual CNN (RCNN) correction stage. We add a CNN residual stage trained on normalized residuals, which yield substantial improvement. Our algorithm was tested on 867 water bodies with over 2,000 predictions and Chla values up to 1000~mg$/m^{3}$, achieving $R^2$ = 0.79, MAE = 13.52~mg$/m^{3}$, and slope = 0.91, demonstrating robust, scalable, and globally transferable performance without additional tuning.

---

## 12. OmniLearned: A Foundation Model Framework for All Tasks Involving Jet Physics

**论文链接:** [http://arxiv.org/abs/2510.24066v1](http://arxiv.org/abs/2510.24066v1)

**作者:** Wahid Bhimji, Chris Harris, Vinicius Mikuni, Benjamin Nachman

**发布时间:** 2025-10-28

**备注:** 12 pages, 5 figures

### GPT解析

### 总结

本研究介绍了OmniLearn基础模型的重大升级，形成了OmniLearned框架。该框架包含模型架构和训练的更新、使用超过十亿个喷注进行训练，以及提供完善的软件访问工具。通过三个代表性任务展示了该框架的性能，结果表明它在所有任务中都是最先进的，显著增强了粒子物理实验的发现潜力。

### 背景

基础模型利用大型数据集构建有效的数据表示，可应用于多样化的下游任务。之前开发的OmniLearn基础模型针对粒子物理喷注，利用了粒子物理的独特性质，能够显著增强对撞机实验的发现潜力。

### 目的

对现有的OmniLearn基础模型进行重大升级，创建OmniLearned框架，进一步提升其在粒子物理实验中的性能和可用性，扩展过去、当前和未来对撞机实验的发现潜力。

### 方法

开发OmniLearned框架，包含三个新元素：更新模型架构和训练方法、使用超过十亿个喷注进行训练、提供完善的软件用于访问所有数据集和模型。通过三个代表性任务进行验证：top夸克喷注标记、b标记和异常检测。

### 主要发现

在三个代表性任务（top夸克喷注标记、b标记和异常检测）中，OmniLearned均达到了最先进的性能水平。该框架能够显著增强对撞机实验的发现潜力，包括过去、当前和未来的实验。

### 结论

OmniLearned框架代表了基础模型在粒子物理领域的重要进展，通过架构更新、大规模训练和完善的软件工具，显著提升了模型性能，为粒子物理研究提供了更强大的分析工具。

### 翻译

基础模型使用大型数据集构建有效的数据表示，可部署在多样化的下游任务中。先前的研究开发了用于粒子物理喷注的OmniLearn基础模型，利用了粒子物理的独特性质，并表明它可以显著增强对撞机实验的发现潜力。本文介绍了一个重大升级，结果是OmniLearned框架。该框架有三个新元素：(1)对模型架构和训练的更新，(2)使用超过十亿个喷注进行训练，(3)提供完善的软件用于访问所有数据集和模型。我们通过三个代表性任务展示了OmniLearned：使用社区Delphes基准数据集进行top夸克喷注标记，使用ATLAS全模拟进行b标记，以及使用CMS实验数据进行异常检测。在每种情况下，OmniLearned都是最先进的，进一步扩展了过去、当前和未来对撞机实验的发现潜力。


### 论文摘要

Foundation models use large datasets to build an effective representation of data that can be deployed on diverse downstream tasks. Previous research developed the OmniLearn foundation model for jet physics, using unique properties of particle physics, and showed that it could significantly advance discovery potential across collider experiments. This paper introduces a major upgrade, resulting in the OmniLearned framework. This framework has three new elements: (1) updates to the model architecture and training, (2) using over one billion jets used for training, and (3) providing well-documented software for accessing all datasets and models. We demonstrate OmniLearned with three representative tasks: top-quark jet tagging with the community Delphes-based benchmark dataset, b-tagging with ATLAS full simulation, and anomaly detection with CMS experimental data. In each case, OmniLearned is the state of the art, further expanding the discovery potential of past, current, and future collider experiments.

---

## 13. Human Machine Social Hybrid Intelligence:A Collaborative Decision Making Framework for Large Model Agent Groups and Human Experts

**论文链接:** [http://arxiv.org/abs/2510.24030v1](http://arxiv.org/abs/2510.24030v1)

**作者:** Ahmet Akkaya Melih, Yamuna Singh, Kunal L. Agarwal, Priya Mukherjee, Kiran Pattnaik, Hanuman Bhatia

**发布时间:** 2025-10-28

### GPT解析

### 总结

提出了一种新型的人机社会混合智能(HMS-HI)框架，通过共享认知空间、动态角色任务分配和跨物种信任校准三个核心组件，实现了人类专家和AI代理之间的深度协作决策，在应急响应模拟中显著降低了伤亡和认知负荷。

### 背景

大型基础模型和多智能体系统快速发展提供了前所未有的能力，但当前人机协同(HiTL)范式未能充分整合人类专业知识，在复杂、高风险环境中常导致认知过载和决策瓶颈。

### 目的

设计一种新型架构，用于人类专家群体和基于大语言模型的AI代理之间的深度协作决策，解决现有人机协同方法的不足。

### 方法

HMS-HI建立在三个核心支柱上：(1)共享认知空间(SCS)用于统一的多模态态势感知和结构化世界建模；(2)动态角色和任务分配(DRTA)模块根据能力和工作负载将任务自适应分配给最适合的代理；(3)跨物种信任校准(CSTC)协议通过可解释声明和结构化反馈促进透明度、责任和相互适应。

### 主要发现

在高保真城市应急响应模拟中，HMS-HI相比传统HiTL方法将平民伤亡减少72%，认知负荷减少70%，证明了卓越的决策质量、效率和人类-AI信任。消融研究确认了每个模块的关键贡献，表明工程化的信任和共享背景是可扩展的人机协作基础。

### 结论

HMS-HI框架通过三个核心组件的整合，在复杂、高风险环境中实现了更有效的人机协作决策，显著提高了决策质量和效率，同时减轻了人类认知负担。

### 翻译

大型基础模型和多智能体系统的快速发展提供了前所未有的能力，但当前人机协同(HiTL)范式未能充分整合人类专业知识，在复杂、高风险环境中常常导致认知过载和决策瓶颈。我们提出了'人机社会混合智能'(HMS-HI)框架，这是一种专为人类专家群体和由大语言模型驱动的AI代理之间的深度协作决策而设计的新型架构。HMS-HI建立在三个核心支柱上：(1)共享认知空间(SCS)用于统一、多模态的态势感知和结构化世界建模；(2)动态角色和任务分配(DRTA)模块，根据能力和工作负载将任务自适应地分配给最适合的代理(人类或AI)；(3)跨物种信任校准(CSTC)协议，通过可解释声明和结构化反馈促进透明度、责任和相互适应。在高保真的城市应急响应模拟中验证，HMS-HI与传统HiTL方法相比，平民伤亡减少了72%，认知负荷减少了70%，证明了卓越的决策质量、效率和人类-AI信任。消融研究确认了每个模块的关键贡献，强调工程化的信任和共享背景是可扩展的、协同的人机协作的基础。


### 论文摘要

The rapid advancements in large foundation models and multi-agent systems offer unprecedented capabilities, yet current Human-in-the-Loop (HiTL) paradigms inadequately integrate human expertise, often leading to cognitive overload and decision-making bottlenecks in complex, high-stakes environments. We propose the "Human-Machine Social Hybrid Intelligence" (HMS-HI) framework, a novel architecture designed for deep, collaborative decision-making between groups of human experts and LLM-powered AI agents. HMS-HI is built upon three core pillars: (1) a \textbf{Shared Cognitive Space (SCS)} for unified, multi-modal situational awareness and structured world modeling; (2) a \textbf{Dynamic Role and Task Allocation (DRTA)} module that adaptively assigns tasks to the most suitable agent (human or AI) based on capabilities and workload; and (3) a \textbf{Cross-Species Trust Calibration (CSTC)} protocol that fosters transparency, accountability, and mutual adaptation through explainable declarations and structured feedback. Validated in a high-fidelity urban emergency response simulation, HMS-HI significantly reduced civilian casualties by 72\% and cognitive load by 70\% compared to traditional HiTL approaches, demonstrating superior decision quality, efficiency, and human-AI trust. An ablation study confirms the critical contribution of each module, highlighting that engineered trust and shared context are foundational for scalable, synergistic human-AI collaboration.

---

## 14. Mars-Bench: A Benchmark for Evaluating Foundation Models for Mars Science Tasks

**论文链接:** [http://arxiv.org/abs/2510.24010v1](http://arxiv.org/abs/2510.24010v1)

**作者:** Mirali Purohit, Bimal Gajera, Vatsal Malaviya, Irish Mehta, Kunal Kasodekar, Jacob Adler, Steven Lu, Umaa Rebbapragada, Hannah Kerner

**发布时间:** 2025-10-28

**备注:** Accepted at NeurIPS 2025

### GPT解析

### 总结

本文介绍了Mars-Bench，这是第一个用于系统评估火星相关任务的基准，包含20个数据集，涵盖分类、分割和目标检测，专注于关键地质特征。研究结果表明火星特定的基础模型可能比通用领域模型具有优势，为火星科学领域的机器学习模型开发提供了标准化基础。

### 背景

基础模型通过大规模无标签数据预训练在许多专业领域取得了快速进展，表现出强大的泛化能力。然而，这类模型在火星科学领域的应用仍然有限，主要原因是火星科学缺乏标准化基准和评估框架，限制了火星任务基础模型的发展。

### 目的

引入Mars-Bench，第一个基准，旨在系统评估使用轨道和表面图像的广泛火星相关任务模型，为火星科学领域的机器学习模型开发提供标准化基础。

### 方法

Mars-Bench包含20个数据集，涵盖分类、分割和目标检测，专注于陨石坑、锥体、巨石和霜等关键地质特征。提供标准化、即用型数据集和基线评估，使用在自然图像、地球卫星数据和最先进的视觉语言模型上预训练的模型进行评估。

### 主要发现

所有分析结果表明，火星特定的基础模型可能比通用领域模型具有优势，这激励了对领域自适应预训练的进一步探索。

### 结论

Mars-Bench旨在为开发和比较火星科学的机器学习模型建立标准化基础，其数据、模型和代码已公开可用。

### 翻译

基础模型通过大规模无标签数据预训练在许多专业领域取得了快速进展，显示出对各种下游任务的强大泛化能力。尽管这类模型在地球观测等领域受到广泛关注，但在火星科学领域的应用仍然有限。其他领域取得进展的一个关键因素是标准化基准的可用性，这些基准支持系统评估。相比之下，火星科学缺乏此类基准和标准化评估框架，这限制了火星任务基础模型的发展。为解决这一差距，我们引入了Mars-Bench，这是第一个基准，旨在使用轨道和表面图像系统评估广泛火星相关任务的模型。Mars-Bench包含20个数据集，涵盖分类、分割和目标检测，专注于陨石坑、锥体、巨石和霜等关键地质特征。我们提供了标准化、即用型数据集和基线评估，使用在自然图像、地球卫星数据和最先进的视觉语言模型上预训练的模型。所有分析的结果表明，火星特定的基础模型可能比通用领域对应模型具有优势，这激励了对领域自适应预训练的进一步探索。Mars-Bench旨在为开发和比较火星科学的机器学习模型建立标准化基础。我们的数据、模型和代码可在 https://mars-bench.github.io/ 获取。


### 论文摘要

Foundation models have enabled rapid progress across many specialized domains by leveraging large-scale pre-training on unlabeled data, demonstrating strong generalization to a variety of downstream tasks. While such models have gained significant attention in fields like Earth Observation, their application to Mars science remains limited. A key enabler of progress in other domains has been the availability of standardized benchmarks that support systematic evaluation. In contrast, Mars science lacks such benchmarks and standardized evaluation frameworks, which have limited progress toward developing foundation models for Martian tasks. To address this gap, we introduce Mars-Bench, the first benchmark designed to systematically evaluate models across a broad range of Mars-related tasks using both orbital and surface imagery. Mars-Bench comprises 20 datasets spanning classification, segmentation, and object detection, focused on key geologic features such as craters, cones, boulders, and frost. We provide standardized, ready-to-use datasets and baseline evaluations using models pre-trained on natural images, Earth satellite data, and state-of-the-art vision-language models. Results from all analyses suggest that Mars-specific foundation models may offer advantages over general-domain counterparts, motivating further exploration of domain-adapted pre-training. Mars-Bench aims to establish a standardized foundation for developing and comparing machine learning models for Mars science. Our data, models, and code are available at: https://mars-bench.github.io/.

---

## 15. Why Foundation Models in Pathology Are Failing

**论文链接:** [http://arxiv.org/abs/2510.23807v1](http://arxiv.org/abs/2510.23807v1)

**作者:** Hamid R. Tizhoosh

**发布时间:** 2025-10-27

### GPT解析

### 总结

基础模型在非医学领域取得成功，但在计算病理学应用中存在根本性概念不匹配，需要重新思考建模范式

### 背景

在非医学领域，基础模型通过大规模自监督和多模态学习彻底改变了计算机视觉和语言处理，预期计算病理学中也会迅速采用这些模型

### 目的

检查基础模型在计算病理学中的缺点，论证这些缺点源于通用基础建模假设与人体组织复杂性之间的概念性不匹配

### 方法

进行系统评估，识别导致基础模型在计算病理学中失效的七个相互关联原因

### 主要发现

基础模型在计算病理学中存在低诊断准确性、鲁棒性差、几何不稳定性、计算需求量大以及安全漏洞等基本弱点

### 结论

当前病理学基础模型在概念上与组织形态学性质不一致，需要对范式本身进行根本性重新思考

### 翻译

在非医学领域，基础模型(FMs)通过大规模自监督和多模态学习彻底改变了计算机视觉和语言处理。因此，预期计算病理学中会迅速采用这些模型，并在癌症诊断、预后和多模态检索方面取得类似突破。然而，最近的系统评估揭示了基本弱点：低诊断准确性、鲁棒性差、几何不稳定性、计算需求量大，以及令人担忧的安全漏洞。这篇简短论文检查了这些缺点，并论证它们源于主流人工智能中通用基础建模的假设与人体组织内在复杂性之间的更深层次的概念性不匹配。确定了七个相互关联的原因：生物复杂性、无效的自监督、过度概括、过度的架构复杂性、缺乏领域特定创新、数据不足，以及与组织块大小相关的基本设计缺陷。这些发现表明，当前病理学基础模型在概念上仍然与组织形态学的性质不一致，需要对这一范式本身进行根本性的重新思考。


### 论文摘要

In non-medical domains, foundation models (FMs) have revolutionized computer vision and language processing through large-scale self-supervised and multimodal learning. Consequently, their rapid adoption in computational pathology was expected to deliver comparable breakthroughs in cancer diagnosis, prognostication, and multimodal retrieval. However, recent systematic evaluations reveal fundamental weaknesses: low diagnostic accuracy, poor robustness, geometric instability, heavy computational demands, and concerning safety vulnerabilities. This short paper examines these shortcomings and argues that they stem from deeper conceptual mismatches between the assumptions underlying generic foundation modeling in mainstream AI and the intrinsic complexity of human tissue. Seven interrelated causes are identified: biological complexity, ineffective self-supervision, overgeneralization, excessive architectural complexity, lack of domain-specific innovation, insufficient data, and a fundamental design flaw related to tissue patch size. These findings suggest that current pathology foundation models remain conceptually misaligned with the nature of tissue morphology and call for a fundamental rethinking of the paradigm itself.

---

## 16. CountFormer: A Transformer Framework for Learning Visual Repetition and Structure in Class-Agnostic Object Counting

**论文链接:** [http://arxiv.org/abs/2510.23785v1](http://arxiv.org/abs/2510.23785v1)

**作者:** Md Tanvir Hossain, Akif Islam, Mohd Ruhul Ameen

**发布时间:** 2025-10-27

**备注:** 6 pages, 2 tables, 6 figures. Submitted to IEEE 5th International  Conference on Electrical, Computer and Telecommunication Engineering (ICECTE  2025)

### GPT解析

### 总结

本文介绍了CountFormer，一个基于transformer的框架，用于学习识别重复和结构一致性，实现类别无关的物体计数。该模型使用DINOv2作为视觉编码器，并融入位置嵌入融合，在FSC-147数据集上实现了与当前最先进方法相当的性能，同时在结构复杂或密集场景中表现更优。

### 背景

人类能够通过感知视觉重复和结构关系而非依赖类别身份来计数多样化物体，但大多数现有计数模型在物体具有复杂形状、内部对称性或重叠组件时经常计数错误。

### 目的

开发一个能够像人类一样通过感知视觉重复和结构关系来进行计数的模型，实现类别无关的物体计数。

### 方法

基于CounTR架构，用自监督基础模型DINOv2替换视觉编码器以获得更丰富的特征表示，融入位置嵌入融合保留几何关系，并通过轻量级卷积解码器将特征解码为密度图。

### 主要发现

在FSC-147数据集上，CountFormer实现了与当前最先进方法相当的性能，同时在结构复杂或密集堆积的场景中表现出更高的准确性。

### 结论

集成基础模型如DINOv2可以使计数系统接近人类的结构感知能力，朝着真正通用和无样本范例的计数范式迈进。

### 翻译

人类可以通过感知视觉重复和结构关系而不是依赖类别身份来轻松计数多样化的物体。然而，大多数现有的计数模型无法复制这种能力；当物体表现出复杂形状、内部对称性或重叠组件时，它们经常计数错误。在这项工作中，我们引入了CountFormer，一个基于transformer的框架，用于学习识别重复和结构一致性，实现类别无关的物体计数。基于CounTR架构，我们的模型用自监督基础模型DINOv2替换了其视觉编码器，DINOv2产生更丰富和空间一致的特征表示。我们进一步融合位置嵌入，在通过轻量级卷积解码器将这些特征解码为密度图之前保留几何关系。在FSC-147数据集上评估，我们的模型实现了与当前最先进方法相当的性能，同时在结构复杂或密集堆积的场景中表现出更高的准确性。我们的研究结果表明，集成基础模型如DINOv2可以使计数系统接近人类的结构感知能力，朝着真正通用和无样本范例的计数范式迈进。


### 论文摘要

Humans can effortlessly count diverse objects by perceiving visual repetition and structural relationships rather than relying on class identity. However, most existing counting models fail to replicate this ability; they often miscount when objects exhibit complex shapes, internal symmetry, or overlapping components. In this work, we introduce CountFormer, a transformer-based framework that learns to recognize repetition and structural coherence for class-agnostic object counting. Built upon the CounTR architecture, our model replaces its visual encoder with the self-supervised foundation model DINOv2, which produces richer and spatially consistent feature representations. We further incorporate positional embedding fusion to preserve geometric relationships before decoding these features into density maps through a lightweight convolutional decoder. Evaluated on the FSC-147 dataset, our model achieves performance comparable to current state-of-the-art methods while demonstrating superior accuracy on structurally intricate or densely packed scenes. Our findings indicate that integrating foundation models such as DINOv2 enables counting systems to approach human-like structural perception, advancing toward a truly general and exemplar-free counting paradigm.

---

## 17. Evaluating Long-Term Memory for Long-Context Question Answering

**论文链接:** [http://arxiv.org/abs/2510.23730v1](http://arxiv.org/abs/2510.23730v1)

**作者:** Alessandra Terranova, Björn Ross, Alexandra Birch

**发布时间:** 2025-10-27

**备注:** 14 pages including appendix, 3 figures. Submitted to October ARR and  to Metacognition in Generative AI EurIPS workshop (under review for both)

### GPT解析

### 总结

该研究系统评估了不同类型的记忆增强方法，发现记忆架构复杂度应与模型能力相匹配，不同类型模型适合不同记忆方法，情景记忆可帮助大语言模型识别自身知识局限性。

### 背景

大语言模型需要记忆功能实现真正的对话连续性和经验学习，但研究虽已聚焦复杂记忆系统开发，仍不清楚哪种记忆类型对长上下文对话任务最有效。

### 目的

使用LoCoMo基准（一个需要多种推理策略的问答任务合成长上下文对话基准）系统评估增强记忆的方法。

### 方法

分析四种记忆增强方法：全上下文提示、通过检索增强生成和智能体记忆实现的语义记忆、通过上下文学习实现的情景记忆、通过提示优化的程序记忆。

### 主要发现

增强记忆方法可减少90%以上的token使用量同时保持有竞争力准确性；小型基础模型从RAG中获益最多；强大指令微调推理模型通过反思获得情景学习好处并受益于更复杂智能体语义记忆。

### 结论

记忆架构复杂度应与模型能力相匹配，情景记忆可以帮助大语言模型识别自身知识的局限性。

### 翻译

为了让大语言模型实现真正的对话连续性和从经验学习中受益，它们需要记忆功能。虽然研究已集中在复杂记忆系统的开发上，但目前尚不清楚哪种类型的记忆对长上下文对话任务最有效。我们使用LoCoMo（一个为需要多种推理策略的问答任务标注的合成长上下文对话基准）对增强记忆的方法进行了系统评估。我们分析了全上下文提示、通过检索增强生成和智能体记忆实现的语义记忆、通过上下文学习实现的情景记忆，以及通过提示优化的程序记忆。我们的研究结果表明，增强记忆的方法在保持有竞争力的准确性的同时，可减少90%以上的token使用量。记忆架构的复杂度应与模型能力相匹配，小型基础模型从RAG中获益最多，而强大的指令微调推理模型通过反思获得情景学习的好处，并受益于更复杂的智能体语义记忆。特别是，情景记忆可以帮助大语言模型识别自身知识的局限性。


### 论文摘要

In order for large language models to achieve true conversational continuity and benefit from experiential learning, they need memory. While research has focused on the development of complex memory systems, it remains unclear which types of memory are most effective for long-context conversational tasks. We present a systematic evaluation of memory-augmented methods using LoCoMo, a benchmark of synthetic long-context dialogues annotated for question-answering tasks that require diverse reasoning strategies. We analyse full-context prompting, semantic memory through retrieval-augmented generation and agentic memory, episodic memory through in-context learning, and procedural memory through prompt optimization. Our findings show that memory-augmented approaches reduce token usage by over 90% while maintaining competitive accuracy. Memory architecture complexity should scale with model capability, with small foundation models benefitting most from RAG, and strong instruction-tuned reasoning model gaining from episodic learning through reflections and more complex agentic semantic memory. In particular, episodic memory can help LLMs recognise the limits of their own knowledge.

---

## 18. Game-TARS: Pretrained Foundation Models for Scalable Generalist Multimodal Game Agents

**论文链接:** [http://arxiv.org/abs/2510.23691v1](http://arxiv.org/abs/2510.23691v1)

**作者:** Zihao Wang, Xujing Li, Yining Ye, Junjie Fang, Haoming Wang, Longxiang Liu, Shihao Liang, Junting Lu, Zhiyong Wu, Jiazhan Feng, Wanjun Zhong, Zili Li, Yu Wang, Yu Miao, Bo Zhou, Yuanfan Li, Hao Wang, Zhongkai Zhao, Faming Wu, Zhengxuan Jiang, Weihao Tan, Heyuan Yao, Shi Yan, Xiangyang Li, Yitao Liang, Yujia Qin, Guang Shi

**发布时间:** 2025-10-27

### GPT解析

### 总结

Game-TARS是一种基于人类对齐的原生键盘鼠标输入的通用游戏代理，使用统一的可扩展动作空间进行训练，在多种游戏任务上表现出色

### 背景

与API或GUI方法不同，需要能够在异构领域(如OS、网络和模拟游戏)进行大规模持续预训练的游戏代理

### 目的

开发一种通用游戏代理，通过简单的可扩展动作表示与大规模预训练相结合，实现广泛的计算机使用能力

### 方法

Game-TARS在超过500B tokens的多样化轨迹和多模态数据上进行预训练，采用衰减持续损失减少因果混淆，以及高效的稀疏思考策略平衡推理深度和推理成本

### 主要发现

在开放世界Minecraft任务上成功率比前SOTA模型高约2倍；在未见过的网络3D游戏中通用性接近新鲜人类；在FPS基准测试中优于GPT-5、Gemini-2.5-Pro和Claude-4-Sonnet；统一动作空间在扩展到跨游戏和多模态数据时能持续改进

### 结论

简单的可扩展动作表示与大规模预训练相结合，为具有广泛计算机使用能力的通用代理提供了有前途的发展路径

### 翻译

我们提出了Game-TARS，一种通用游戏代理，通过统一的、可扩展的动作空间进行训练，该动作空间锚定在人类对齐的原生键盘鼠标输入上。与基于API或GUI的方法不同，这种范式能够在包括操作系统、网络和模拟游戏在内的异构领域进行大规模持续预训练。Game-TARS在超过500B tokens的多样化轨迹和多模态数据上进行预训练。关键技术包括减少因果混淆的衰减持续损失，以及平衡推理深度和推理成本的高效稀疏思考策略。实验表明，Game-TARS在开放世界Minecraft任务上的成功率比之前的SOTA模型高出约2倍，在未见过的网络3D游戏中接近新鲜人类的通用性，并在FPS基准测试中优于GPT-5、Gemini-2.5-Pro和Claude-4-Sonnet。训练时间和测试时间的扩展结果证实，统一动作空间在扩展到跨游戏和多模态数据时能够持续改进。我们的结果表明，简单的可扩展动作表示与大规模预训练相结合，为具有广泛计算机使用能力的通用代理提供了一条有前途的道路。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决如何创建真正可扩展且具有广泛泛化能力的游戏智能体问题。这个问题很重要，因为构建能够与复杂动态数字环境无缝交互的通用人工智能体是实现AGI的关键路径，而视频游戏因其多样化的任务目标和丰富的视觉信息，成为训练和评估此类智能体的理想平台。现有方法在创建具有真正泛化能力的智能体方面仍面临重大挑战，限制了AI系统在开放世界环境中的应用。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先分析了传统智能体的局限性，认识到动作空间与特定环境紧密耦合限制了泛化能力。他们提出将动作空间抽象到更低层次，直接锚定到人类交互的基本输入设备——键盘和鼠标。设计过程借鉴了ReAct范式将推理和动作统一输出，采用Deitke等人的在线思考协议(think-aloud protocol)收集高质量轨迹，使用视觉锚点方法解决多模态数据对齐问题，并在后训练阶段借鉴拒绝采样优化推理-动作链。这些方法基于对现有工作的理解，但进行了创新性整合和改进。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是使用统一锚定于键盘-鼠标输入的动作空间，结合稀疏思考策略和大规模持续预训练，实现跨领域的泛化能力。整体流程分为两个主要阶段：1)持续预训练阶段：使用统一动作空间收集游戏轨迹，通过在线思考协议收集稀疏ReAct轨迹，使用视觉锚点对齐多模态数据，采用衰减损失函数处理动作分布不平衡，在500B+ token上预训练；2)后训练阶段：通过指令遵循、多模态提示、稀疏思考优化、双层记忆架构和跨域数据整合，提升智能体的核心能力。最终在Minecraft、FPS游戏等未见环境中进行评估验证。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)统一动作空间：锚定到底层键盘-鼠标输入而非高层API，实现跨环境泛化；2)稀疏思考策略：只在关键决策点推理，通过在线思考协议收集高质量轨迹；3)衰减持续损失：解决动作分布不平衡导致的因果混淆；4)双层记忆架构：结合短期上下文和长期摘要记忆；5)跨域数据整合：将游戏数据与其他领域智能体轨迹结合。相比之前工作，传统API/GUI方法使用定制化动作集与环境紧密耦合，而Game-TARS的统一动作空间具有更好的泛化性；现有游戏智能体通常专注于特定环境，而Game-TARS通过大规模预训练实现了真正的泛化能力。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'Game-TARS通过统一锚定于键盘-鼠标输入的动作空间和大规模持续预训练，实现了在多样化游戏和环境中表现卓越的通用游戏智能体，相比之前的方法展现出显著的泛化能力和性能提升。'}


### 论文摘要

We present Game-TARS, a generalist game agent trained with a unified, scalable action space anchored to human-aligned native keyboard-mouse inputs. Unlike API- or GUI-based approaches, this paradigm enables large-scale continual pre-training across heterogeneous domains, including OS, web, and simulation games. Game-TARS is pre-trained on over 500B tokens with diverse trajectories and multimodal data. Key techniques include a decaying continual loss to reduce causal confusion and an efficient Sparse-Thinking strategy that balances reasoning depth and inference cost. Experiments show that Game-TARS achieves about 2 times the success rate over the previous sota model on open-world Minecraft tasks, is close to the generality of fresh humans in unseen web 3d games, and outperforms GPT-5, Gemini-2.5-Pro, and Claude-4-Sonnet in FPS benchmarks. Scaling results on training-time and test-time confirm that the unified action space sustains improvements when scaled to cross-game and multimodal data. Our results demonstrate that simple, scalable action representations combined with large-scale pre-training provide a promising path toward generalist agents with broad computer-use abilities.

---

## 19. Explicit Memory through Online 3D Gaussian Splatting Improves Class-Agnostic Video Segmentation

**论文链接:** [http://arxiv.org/abs/2510.23521v1](http://arxiv.org/abs/2510.23521v1)

**作者:** Anthony Opipari, Aravindhan K Krishnan, Shreekant Gayaka, Min Sun, Cheng-Hao Kuo, Arnie Sen, Odest Chadwicke Jenkins

**发布时间:** 2025-10-27

**DOI:** 10.1109/LRA.2025.3619783

**备注:** Accepted in IEEE Robotics and Automation Letters September 2025

### GPT解析

### 总结

本文提出了一种使用显式3D记忆增强视频分割模型的方法，通过在线3D高斯溅射技术存储预测的对象级片段，显著提高了预测的准确性和一致性。

### 背景

现有视频分割算法通常不使用对象级记忆（如FastSAM）或仅使用循环神经网络特征的隐式记忆（如SAM2），而记住过去预测的对象片段位置对提高分割质量至关重要。

### 目的

开发一种显式3D记忆方法来增强现有分割模型，使增强后的模型具有更准确和一致的预测能力。

### 方法

开发在线3D高斯溅射（3DGS）技术存储视频过程中生成的预测对象级片段，并基于此开发融合技术FastSAM-Splat和SAM2-Splat，利用显式3DGS记忆改进各自基础模型预测。

### 主要发现

消融实验验证了所提技术和超参数设置的有效性；真实世界和模拟基准实验结果表明，使用显式3D记忆的模型比无记忆或仅使用隐式神经网络记忆的模型产生更准确和一致的预测。

### 结论

显式3D记忆技术可以显著改善视频分割模型的性能，提高预测的准确性和一致性。

### 翻译

记住过去预测的对象片段位置对提高无类别视频分割算法的准确性和一致性是有用的。现有的视频分割算法通常使用不使用对象级记忆（例如FastSAM）或使用循环神经网络特征的隐式记忆（例如SAM2）。在本文中，我们使用显式3D记忆增强这两种分割模型，并证明 resulting 模型具有更准确和一致的预测。为此，我们开发了一种在线3D高斯溅射（3DGS）技术来存储在整个视频持续时间内生成的预测对象级片段。基于这种3DGS表示，开发了一系列融合技术，分别命名为FastSAM-Splat和SAM2-Splat，它们使用显式3DGS记忆来改进各自基础模型的预测。使用消融实验来验证所提技术的设计和超参数设置。来自真实世界和模拟基准实验的结果表明，使用显式3D记忆的模型比不使用记忆或仅使用隐式神经网络记忆的模型产生更准确和一致的预测。项目页面：https://topipari.com/projects/FastSAM-Splat/

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决类无关视频分割(class-agnostic video segmentation)的准确性和一致性问题。这个问题在机器人应用中非常重要，因为机器人需要在人类家庭环境中构建有用的语义地图，必须能够检测和分割任何类别的物体（包括部署前未知的物体）。现实环境中的遮挡、低光照、重复和动态物体等因素使得这一挑战更加复杂，而现有的视频分割算法要么不使用物体级记忆，要么使用隐式记忆，导致分割结果在时间维度上不一致，影响机器人对环境的理解和交互。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者观察到记住物体分割在过去的预测位置对提高视频分割一致性很有用，并假设模型如果能访问过去预测的密集3D记忆将会受益。他们借鉴了3D高斯溅射(3DGS)技术，这是一种用于密集3D场景重建的强大表示方法。作者还受到在线3DGS技术的启发，这些技术可以从视频输入中实时构建环境3D地图。他们扩展了3DGS表示，将每个高斯与一个段ID特征向量关联，用于存储语义记忆，并基于对比学习优化段ID码本，确保不同物体段的ID向量之间有足够距离。这种方法结合了现有的视频分割模型(如FastSAM和SAM2)和3D重建技术，创造性地将显式3D记忆引入视频分割任务。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是使用显式的3D记忆来增强视频分割模型，以提高分割的一致性和准确性。通过在线3D高斯溅射(3DGS)技术存储物体分割的历史信息，形成一个3D记忆系统，利用这个记忆来指导当前帧的分割预测。整体实现流程包括：1)构建3DGS表示，每个高斯参数包括位置、方向、缩放、不透明度、颜色和段ID特征；2)创建段ID码本，使用对比损失优化确保不同段ID间有足够距离；3)对于FastSAM-Splat，将渲染的3DGS段与FastSAM预测的图像段匹配并融合；4)对于SAM2-Splat，使用SAM2预测的跟踪ID与3DGS段关联，识别不一致并通过重新提示SAM2来纠正错误；5)更新3DGS记忆以对齐当前帧的预测。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)FastSAM-Splat模型，扩展了原本没有时间记忆的FastSAM，通过集成3DGS记忆提高分割一致性；2)SAM2-Splat模型，开发了基于3DGS的重新提示策略，通过整合显式3D记忆减少SAM2的不一致预测；3)通过实验验证显式3D记忆的优势。相比之前的工作，这种方法使用显式的3D记忆而非无物体级记忆或隐式记忆(如循环神经网络特征)；使用3D高斯溅射存储和表示物体分割历史，这是一种更密集和结构化的表示；专注于机器人应用场景，利用深度和相机姿态信息构建3D记忆；针对不同类型的分割模型(FastSAM和SAM2)设计了不同的融合策略；实验表明在处理遮挡和物体重新出现等挑战性场景时性能更优。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '本文通过引入基于在线3D高斯溅射的显式记忆机制，显著提升了类无关视频分割的准确性和时间一致性，为机器人感知任务提供了更可靠的分割解决方案。'}


### 论文摘要

Remembering where object segments were predicted in the past is useful for improving the accuracy and consistency of class-agnostic video segmentation algorithms. Existing video segmentation algorithms typically use either no object-level memory (e.g. FastSAM) or they use implicit memories in the form of recurrent neural network features (e.g. SAM2). In this paper, we augment both types of segmentation models using an explicit 3D memory and show that the resulting models have more accurate and consistent predictions. For this, we develop an online 3D Gaussian Splatting (3DGS) technique to store predicted object-level segments generated throughout the duration of a video. Based on this 3DGS representation, a set of fusion techniques are developed, named FastSAM-Splat and SAM2-Splat, that use the explicit 3DGS memory to improve their respective foundation models' predictions. Ablation experiments are used to validate the proposed techniques' design and hyperparameter settings. Results from both real-world and simulated benchmarking experiments show that models which use explicit 3D memories result in more accurate and consistent predictions than those which use no memory or only implicit neural network memories. Project Page: https://topipari.com/projects/FastSAM-Splat/

---

## 20. Dexbotic: Open-Source Vision-Language-Action Toolbox

**论文链接:** [http://arxiv.org/abs/2510.23511v1](http://arxiv.org/abs/2510.23511v1)

**作者:** Bin Xie, Erjin Zhou, Fan Jia, Hao Shi, Haoqiang Fan, Haowei Zhang, Hebei Li, Jianjian Sun, Jie Bin, Junwen Huang, Kai Liu, Kaixin Liu, Kefan Gu, Lin Sun, Meng Zhang, Peilong Han, Ruitao Hao, Ruitao Zhang, Saike Huang, Songhan Xie, Tiancai Wang, Tianle Liu, Wenbin Tang, Wenqi Zhu, Yang Chen, Yingfei Liu, Yizhuang Zhou, Yu Liu, Yucheng Zhao, Yunchao Ma, Yunfei Wei, Yuxiang Chen, Ze Chen, Zeming Li, Zhao Wu, Ziheng Zhang, Ziming Liu, Ziwei Yan, Ziyu Zhang

**发布时间:** 2025-10-27

**备注:** Authors are listed in alphabetical order. The official website is  located at https://dexbotic.com/. Code is available at  https://github.com/Dexmal/dexbotic

### GPT解析

### 总结

本文介绍了Dexbotic，一个基于PyTorch的开源视觉-语言-行动模型工具箱，为具身智能研究提供一站式服务。

### 背景

具身智能领域需要有效的工具来支持视觉-语言-行动模型的研究和开发，现有工具可能缺乏统一性和易用性。

### 目的

提供一个开源的、统一的VLA模型工具箱，使研究人员能够轻松复现各种VLA方法，快速开发新实验，并利用更强大的预训练模型提升性能。

### 方法

开发了一个基于PyTorch的Dexbotic工具箱，支持多种主流VLA策略，提供实验为中心的开发环境，并开发更强大的预训练模型。

### 主要发现

该工具箱能够支持多种VLA策略的统一实现，通过简单的环境设置即可复现各种方法；通过修改Exp脚本可以快速开发新实验；使用更强大的预训练模型可以显著提升最先进VLA策略的性能。

### 结论

Dexbotic作为一个开源工具箱，有效简化了VLA模型的研究流程，提高了研究效率，并将持续更新以包含最新的预训练模型和前沿VLA模型。

### 翻译

在本文中，我们提出了Dexbotic，一个基于PyTorch的开源视觉-语言-行动模型工具箱。它旨在为具身智能领域的专业人士提供一站式VLA研究服务。它提供了一个代码库，同时支持多种主流VLA策略，使用户只需通过单一环境设置即可重现各种VLA方法。该工具箱以实验为中心，用户只需修改Exp脚本即可快速开发新的VLA实验。此外，我们提供了更强大的预训练模型，以实现最先进的VLA策略的性能提升。Dexbotic将不断更新，以包含更多最新的预训练基础模型和行业前沿的VLA模型。


### 论文摘要

In this paper, we present Dexbotic, an open-source Vision-Language-Action (VLA) model toolbox based on PyTorch. It aims to provide a one-stop VLA research service for professionals in the field of embodied intelligence. It offers a codebase that supports multiple mainstream VLA policies simultaneously, allowing users to reproduce various VLA methods with just a single environment setup. The toolbox is experiment-centric, where the users can quickly develop new VLA experiments by simply modifying the Exp script. Moreover, we provide much stronger pretrained models to achieve great performance improvements for state-of-the-art VLA policies. Dexbotic will continuously update to include more of the latest pre-trained foundation models and cutting-edge VLA models in the industry.

---

## 21. Exploring Vulnerability in AI Industry

**论文链接:** [http://arxiv.org/abs/2510.23421v1](http://arxiv.org/abs/2510.23421v1)

**作者:** Claudio Pirrone, Stefano Fricano, Gioacchino Fazio

**发布时间:** 2025-10-27

**备注:** Preliminary Draft

### GPT解析

### 总结

本文提出了一种合成AI脆弱性指数(AIVI)，评估Foundation Models(FMs)生产上游价值链的脆弱性，重点关注计算、数据、人才、资本和能源五个关键输入因素。

### 背景

Foundation Models(FMs)因Transformer架构的快速发展而推动当前AI生态系统。这些大型模型通过大规模训练和下游适应性获得广泛采用，形成由平台经济学和激烈投资塑造的动荡市场。

### 目的

由于数据限制，评估快速发展的AI行业脆弱性具有挑战性。本研究旨在提出一种专注于FM生产上游价值链的合成AI脆弱性指数(AIVI)，优先考虑公开可用数据。

### 方法

将FM输出建模为五个输入的函数：计算(Compute)、数据(Data)、人才(Talent)、资本(Capital)和能源(Energy)，假设任何输入的供应脆弱性都会威胁整个行业。使用加权几何平均数聚合子指数，并使用理论或经验基准进行归一化。

### 主要发现

关键脆弱性包括：计算集中、数据稀缺和法律风险、人才瓶颈、资本密集度和战略依赖性，以及不断增长的能源需求。

### 结论

尽管存在局限性和改进空间，但这一初步指数旨在量化AI核心生产引擎中的系统性风险，并间接揭示下游价值链的风险。

### 翻译

Foundation Models(FMs)的快速发展，得益于Transformer架构，推动了当前的AI生态系统。这些大型模型以大规模训练和下游适应性为特征（如GPT系列），已获得广泛采用，促成了由平台经济学和激烈投资塑造的动荡市场。由于数据限制，评估这个快速发展的行业的脆弱性至关重要且具有挑战性。本文提出了一种专注于FM生产上游价值链的合成AI脆弱性指数(AIVI)，优先考虑公开可用数据。我们将FM输出建模为五个输入的函数：计算、数据、人才、资本和能源，并假设任何输入的供应脆弱性都会威胁整个行业。主要脆弱性包括计算集中、数据稀缺和法律风险、人才瓶颈、资本密集度和战略依赖性，以及不断增长的能源需求。考虑到输入的不完全可替代性，我们提出使用加权几何平均数来聚合子指数，并使用理论或经验基准进行归一化。尽管存在局限性和改进空间，但这一初步指数旨在量化AI核心生产引擎中的系统性风险，并间接揭示了下游价值链的风险。


### 论文摘要

The rapid ascent of Foundation Models (FMs), enabled by the Transformer architecture, drives the current AI ecosystem. Characterized by large-scale training and downstream adaptability, FMs (as GPT family) have achieved massive public adoption, fueling a turbulent market shaped by platform economics and intense investment. Assessing the vulnerability of this fast-evolving industry is critical yet challenging due to data limitations. This paper proposes a synthetic AI Vulnerability Index (AIVI) focusing on the upstream value chain for FM production, prioritizing publicly available data. We model FM output as a function of five inputs: Compute, Data, Talent, Capital, and Energy, hypothesizing that supply vulnerability in any input threatens the industry. Key vulnerabilities include compute concentration, data scarcity and legal risks, talent bottlenecks, capital intensity and strategic dependencies, as well as escalating energy demands. Acknowledging imperfect input substitutability, we propose a weighted geometrical average of aggregate subindexes, normalized using theoretical or empirical benchmarks. Despite limitations and room for improvement, this preliminary index aims to quantify systemic risks in AI's core production engine, and implicitly shed a light on the risks for downstream value chain.

---

## 22. Towards Generalisable Foundation Models for 3D Brain MRI

**论文链接:** [http://arxiv.org/abs/2510.23415v1](http://arxiv.org/abs/2510.23415v1)

**作者:** Moona Mazher, Geoff J. M. Parker, Daniel C. Alexander

**发布时间:** 2025-10-27

### GPT解析

### 总结

BrainFound是一种自监督基础模型，通过扩展DINO-v2视觉转换器构建，专为脑部MRI设计，能够处理3D脑部解剖信息，支持单模态和多模态输入，在标签稀缺和多对比度环境下表现优异，提高了诊断准确性并减少了对专家标注的依赖。

### 背景

人工智能基础模型通过大规模无标签数据集实现通用特征学习，正在改变医学影像领域。

### 目的

开发一个针对脑部MRI的自监督基础模型，能够处理3D脑部解剖信息，支持多种下游任务。

### 方法

通过扩展DINO-v2视觉转换器构建BrainFound，整合连续MRI切片的体积信息来建模完整3D脑部解剖结构，支持单模态和多模态输入，适用于多种MRI模态（如T1、T2、FLAIR）。

### 主要发现

BrainFound在性能上始终优于现有的自监督预训练策略和监督基线，特别是在标签稀缺和多对比度设置下；通过整合多种3D MRI模态信息，提高了诊断准确性，减少了对大量专家标注的依赖。

### 结论

BrainFound的灵活性使其成为3D神经影像流程的可扩展且实用的解决方案，具有在临床部署和研究创新方面的巨大潜力。

### 翻译

人工智能（AI）中的基础模型正在通过大规模无标签数据集实现通用特征学习，从而改变医学影像。在这项工作中，我们介绍了BrainFound，这是一个用于脑部MRI的自监督基础模型，通过扩展DINO-v2（一种最初为2D自然图像设计的视觉转换器）构建。BrainFound通过整合连续MRI切片的体积信息来适应DINO-v2，以建模完整的3D脑部解剖结构，超越了传统的单切片范式。它支持单模态和多模态输入，能够实现广泛的下游任务，包括疾病检测和图像分割，同时能够在不同的成像协议和临床场景中泛化。我们证明BrainFound在性能上始终优于现有的自监督预训练策略和监督基线，特别是在标签稀缺和多对比度设置下。通过整合多种3D MRI模态（如T1、T2、FLAIR）的信息，它提高了诊断准确性，减少了对大量专家标注的依赖。这种灵活性使BrainFound成为3D神经影像流程的可扩展且实用的解决方案，在临床部署和研究创新方面具有巨大潜力。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决如何构建通用的基础模型用于3D脑部MRI分析的问题。这个问题在现实中非常重要，因为放射科医生面临巨大工作压力（平均每3-4秒需解读一张图像），导致诊断延迟和错误；深度学习在放射学领域虽潜力巨大，但成功依赖于大量昂贵耗时的标记数据；现有监督模型难以跨领域泛化；而大多数基础模型是为2D自然图像设计，无法有效处理3D医学影像数据。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者借鉴了DINO-v2框架（一种为2D自然图像设计的视觉变换器）和自监督学习方法，特别是对比学习和知识蒸馏。设计思路是将DINO-v2从2D扩展到3D，通过处理3D扫描作为2D切片序列；设计支持单模态和多模态MRI输入的架构；将T1、T2和FLAIR扫描堆叠为通道输入（类似RGB图像）；采用多尺度裁剪策略捕获全局和局部脑结构；使用双域预训练（先在自然图像上预训练，再在脑部MRI上微调）。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是将3D脑部MRI作为2D切片序列处理，利用自监督学习在大规模未标记数据上学习通用特征表示，结合自然图像和脑部MRI的双域预训练，并支持多模态输入整合不同MRI对比度的互补信息。整体流程包括：1)收集10,000个体积脑部MRI图像并进行标准化预处理；2)基于DINO-v2构建支持多模态输入的Vision Transformer架构；3)使用多尺度裁剪和自监督知识蒸馏方法进行预训练；4)在下游任务（疾病检测和图像分割）上进行微调应用。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)将DINO-v2从2D扩展到3D脑部MRI处理；2)支持可变体积深度，提高对不同MRI任务的适应性；3)设计能处理单模态和多模态MRI输入的架构，将不同MRI模态堆叠为通道输入；4)采用双域预训练策略，结合自然图像和脑部MRI的优势；5)统一处理疾病检测和图像分割任务。相比之前的工作，BrainFound在多种任务上表现优于仅使用自然图像预训的模型、仅在脑部图像上从头训练的自监督模型以及其他自监督方法，其多模态设计也提供了比单模态模型更强的性能。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'BrainFound通过将DINO-v2框架扩展到3D脑部MRI并采用双域预训练策略，创建了一个强大的自监督基础模型，能够有效处理多模态输入并在疾病检测和图像分割任务上实现卓越的泛化性能，减少了对大量标记数据的依赖。'}


### 论文摘要

Foundation models in artificial intelligence (AI) are transforming medical imaging by enabling general-purpose feature learning from large-scale, unlabeled datasets. In this work, we introduce BrainFound, a self-supervised foundation model for brain MRI, built by extending DINO-v2, a vision transformer originally designed for 2D natural images. BrainFound adapts DINO-v2 to model full 3D brain anatomy by incorporating volumetric information from sequential MRI slices, moving beyond conventional single-slice paradigms. It supports both single- and multimodal inputs, enabling a broad range of downstream tasks, including disease detection and image segmentation, while generalising across varied imaging protocols and clinical scenarios. We show that BrainFound consistently outperforms existing self-supervised pretraining strategies and supervised baselines, particularly in label-scarce and multi-contrast settings. By integrating information from diverse 3D MRI modalities (e.g., T1, T2, FLAIR), it enhances diagnostic accuracy and reduces dependency on extensive expert annotations. This flexibility makes BrainFound a scalable and practical solution for 3D neuroimaging pipelines, with significant potential for clinical deployment and research innovation.

---

## 23. Bid2X: Revealing Dynamics of Bidding Environment in Online Advertising from A Foundation Model Lens

**论文链接:** [http://arxiv.org/abs/2510.23410v1](http://arxiv.org/abs/2510.23410v1)

**作者:** Jiahao Ji, Tianyu Wang, Yeshu Li, Yushen Huo, Zhilin Zhang, Chuan Yu, Jian Xu, Bo Zheng

**发布时间:** 2025-10-27

**备注:** 12 pages, KDD 2025

### GPT解析

### 总结

本文提出了Bid2X出价基础模型，通过统一的函数估计特定出价下的效果，解决了传统出价模型在不同场景间泛化能力有限的问题。该模型结合了序列嵌入、双注意力机制和零膨胀投影模块，在淘宝广告平台部署后显著提升了广告效果。

### 背景

自动出价对在线广告至关重要，但现有出价模型通常针对特定场景设计，在不同环境间的泛化能力有限。

### 目的

探索场景无关的出价原则，提出一个统一的函数来估计特定出价下的效果，如预算消耗、商品交易总额(GMV)和页面浏览量等。

### 方法

提出Bid2X出价基础模型，构建在统一序列嵌入之上，通过定制嵌入方法编码异构数据；提出两种注意力机制分别处理不同变量和不同时间的嵌入；使用变量感知融合模块进行自适应出价结果预测；设计零膨胀投影模块将估计的非零概率纳入值预测，形成包含分类和回归的联合优化目标。

### 主要发现

模型已在淘宝广告平台部署；在八个数据集上的离线评估显示Bid2X优于各种基线模型且在不同场景中具有通用性；在线A/B测试中GMV增加4.65%，ROI增加2.44%。

### 结论

Bid2X为计算广告中的出价基础模型铺平了道路，展示了基础模型在广告出价领域的应用潜力。

### 翻译

自动出价对于通过为广告商自动提供出价来促进在线广告至关重要。虽然之前的工作在建模出价环境以获得更好的广告效果方面做出了巨大努力，但这些模型通常针对特定的出价场景定制，在不同环境间的泛化能力存在局限性。为此，我们通过一个统一的函数来探索场景无关的原则，该函数估计特定出价下的效果，如预算消耗、商品交易总额(GMV)、页面浏览量等。然后，我们提出了Bid2X出价基础模型，从各种场景的数据中学习这个基本函数。我们的Bid2X构建在统一序列嵌入之上，通过定制的嵌入方法编码异构数据。为了捕捉出价数据中复杂的变量间动态和时间依赖关系，我们提出了两种注意力机制，分别将不同变量和不同时间的嵌入作为注意力令牌进行表示学习。在学习到的变量和时间表示之上，使用变量感知融合模块进行自适应出价结果预测。为了建模独特的出价数据分布，我们设计了一个零膨胀投影模块，将估计的非零概率纳入其值预测，这构成了一个包含分类和回归的联合优化目标。该目标被证明可以收敛到零膨胀分布。我们的模型已部署在淘宝广告平台上，这是世界上最大的电子商务平台之一。在八个数据集上的离线评估显示，与各种基线相比，Bid2X具有优越性，并且在不同场景中具有通用性。Bid2X在线A/B测试中使GMV增加了4.65%，ROI增加了2.44%，为计算广告中的出价基础模型铺平了道路。


### 论文摘要

Auto-bidding is crucial in facilitating online advertising by automatically providing bids for advertisers. While previous work has made great efforts to model bidding environments for better ad performance, it has limitations in generalizability across environments since these models are typically tailored for specific bidding scenarios. To this end, we approach the scenario-independent principles through a unified function that estimates the achieved effect under specific bids, such as budget consumption, gross merchandise volume (GMV), page views, etc. Then, we propose a bidding foundation model Bid2X to learn this fundamental function from data in various scenarios. Our Bid2X is built over uniform series embeddings that encode heterogeneous data through tailored embedding methods. To capture complex inter-variable and dynamic temporal dependencies in bidding data, we propose two attention mechanisms separately treating embeddings of different variables and embeddings at different times as attention tokens for representation learning. On top of the learned variable and temporal representations, a variable-aware fusion module is used to perform adaptive bidding outcome prediction. To model the unique bidding data distribution, we devise a zero-inflated projection module to incorporate the estimated non-zero probability into its value prediction, which makes up a joint optimization objective containing classification and regression. The objective is proven to converge to the zero-inflated distribution. Our model has been deployed on the ad platform in Taobao, one of the world's largest e-commerce platforms. Offline evaluation on eight datasets exhibits Bid2X's superiority compared to various baselines and its generality across different scenarios. Bid2X increased GMV by 4.65% and ROI by 2.44% in online A/B tests, paving the way for bidding foundation model in computational advertising.

---

## 24. Solar flare forecasting with foundational transformer models across image, video, and time-series modalities

**论文链接:** [http://arxiv.org/abs/2510.23400v1](http://arxiv.org/abs/2510.23400v1)

**作者:** S. Riggi, P. Romano, A. Pilzer, U. Becciani

**发布时间:** 2025-10-27

**备注:** 15 pages, 4 figures

### GPT解析

### 总结

本研究比较了基于Transformer的架构在利用异构数据模态（包括图像、视频序列和时间序列观测）进行太阳耀斑预测方面的性能

### 背景

太阳活动预测对于空间天气预警至关重要，但需要处理多种类型的数据

### 目的

评估Transformer主干架构在太阳活动的空间和时间表示方面的优势和局限性

### 方法

使用三种预训练模型（SigLIP2用于图像编码，VideoMAE用于时空视频表示，Moirai2用于多元时间序列预测）处理来自SDO/HMI任务的太阳磁图和GOES卫星的软X射线通量数据，并采用多种损失函数和训练平衡策略来处理类别不平衡问题

### 主要发现

虽然SigLIP2和VideoMAE在图像和视频数据上达到典型性能（真实技能统计约0.60-0.65），但仅基于辐照度时间演化的Moirai2时间序列模型达到了优越的预测技能（真实技能统计约0.74）

### 结论

预训练Transformer架构和跨模态学习对推进业务空间天气预报具有潜力，为整合视觉和时间信息的统一多模态模型铺平了道路

### 翻译

我们提出了一项基于Transformer架构的太阳耀斑预测的比较研究，该研究使用异构数据模态，包括图像、视频序列和时间序列观测。我们的分析评估了三个最近的基础模型 - 用于图像编码的SigLIP2，用于时空视频表示的VideoMAE，以及用于多元时间序列预测的Moirai2 - 应用于来自SDO/HMI任务的太阳磁图公共数据集和GOES卫星获取的软X射线通量。所有模型在一致的数据分割和评估标准下进行训练和验证，旨在评估Transformer主干架构在太阳活动的空间和时间表示方面的优势和局限性。我们研究了多种损失公式（加权BCE、focal和分数导向的）和训练平衡策略，以减轻耀斑数据集中典型的类别不平衡。结果表明，虽然SigLIP2和VideoMAE在图像和视频数据上达到典型性能（真实技能统计约0.60-0.65），但时间序列模型Moirai2仅基于辐照度时间演化就达到了优越的预测技能（真实技能统计约0.74）。这些发现突显了预训练Transformer架构和跨模态学习在推进业务空间天气预报方面的潜力，为整合视觉和时间信息的统一多模态模型铺平了道路。


### 论文摘要

We present a comparative study of transformer-based architectures for solar flare forecasting using heterogeneous data modalities, including images, video sequences, and time-series observations. Our analysis evaluates three recent foundational models - SigLIP2 for image encoding, VideoMAE for spatio-temporal video representation, and Moirai2 for multivariate time-series forecasting - applied to publicly available datasets of solar magnetograms from the SDO/HMI mission and soft X-ray fluxes acquired by GOES satellites. All models are trained and validated under consistent data splits and evaluation criteria, with the goal of assessing the strengths and limitations of transformer backbones across spatial and temporal representations of solar activity. We investigate multiple loss formulations (weighted BCE, focal, and score-oriented) and training balance strategies to mitigate class imbalance typical of flare datasets. Results show that while both SigLIP2 and VideoMAE achieve typical performance on image and video data (True Skill Statistic TSS~0.60-0.65), the time-series model Moirai2 reaches superior forecasting skill (TSS~0.74) using irradiance-based temporal evolution alone. These findings highlight the potential of pretrained transformer architectures and cross-modal learning for advancing operational space weather forecasting, paving the way toward unified multimodal models that integrate visual and temporal information.

---

## 25. ZeroFlood: A Geospatial Foundation Model for Data-Efficient Flood Susceptibility Mapping

**论文链接:** [http://arxiv.org/abs/2510.23364v1](http://arxiv.org/abs/2510.23364v1)

**作者:** Hyeongkyun Kim, Orestis Oikonomou

**发布时间:** 2025-10-27

**备注:** Preprint submitted to EUSAR 2026 (under review)

### GPT解析

### 总结

ZeroFlood是一种地理空间基础模型框架，通过思维模态推理实现数据高效的洪水易感性映射，能够在数据稀缺地区从基本地球观测数据进行洪水预测。

### 背景

洪水易感性映射对于灾害预防至关重要，但在数据稀缺地区面临挑战，因为传统水动力模型需要密集的地球物理输入数据。

### 目的

开发ZeroFlood框架，解决数据稀缺地区洪水易感性映射的问题，提供一种数据高效的解决方案。

### 方法

使用思维模态(TiM)推理微调地理空间基础模型(GFMs)，从Sentinel-1或Sentinel-2等基本地球观测数据进行洪水预测；利用数据丰富地区的成对地球观测和模拟洪水地图，通过跨模态表示学习弥合数据差距；使用TerraMind和Prithvi GFMs进行实验验证。

### 主要发现

TiM推理增强了模型的鲁棒性，TerraMind-Large配置实现了67.21的F1分数。

### 结论

基于基础模型的FSM是一种可扩展且数据高效的洪水风险管理解决方案，适用于数据稀缺地区。

### 翻译

洪水易感性映射(FSM)对灾害预防至关重要，但在需要密集地球物理输入的水动力模型难以应用的稀缺数据地区仍然具有挑战性。本文介绍了ZeroFlood，一种用于数据高效FSM的地理空间基础模型框架。该方法通过思维模态(TiM)推理微调地理空间基础模型(GFMs)，能够从Sentinel-1或Sentinel-2等基本地球观测数据进行洪水预测。利用数据丰富地区的成对地球观测和模拟洪水地图，ZeroFlood通过跨模态表示学习弥合了数据可用性差距。使用TerraMind和Prithvi GFMs的实验表明，TiM增强了模型鲁棒性，其中TerraMind-Large配置实现了67.21的F1分数。结果证明了基于基础模型的FSM作为可扩展和数据高效的洪水风险管理解决方案的可行性。


### 论文摘要

Flood susceptibility mapping (FSM) is vital for disaster prevention but remains challenging in data-scarce regions where hydrodynamic models require dense geophysical inputs. This work introduces ZeroFlood, a geospatial foundation model framework for data-efficient FSM. The approach fine-tunes Geospatial Foundation Models (GFMs) with Thinking-in-Modality (TiM) reasoning, enabling flood prediction from basic Earth observation data such as Sentinel-1 or Sentinel-2 imagery. Using paired EO and simulated flood maps from data-rich regions, ZeroFlood bridges data availability gaps through cross-modal representation learning. Experiments with TerraMind and Prithvi GFMs show that TiM enhances model robustness, with the TerraMind-Large configuration achieving an F1 score of 67.21. The results demonstrate the feasibility of foundation-model-based FSM as a scalable and data-efficient solution for flood risk management.

---

## 26. Provable test-time adaptivity and distributional robustness of in-context learning

**论文链接:** [http://arxiv.org/abs/2510.23254v1](http://arxiv.org/abs/2510.23254v1)

**作者:** Tianyi Ma, Tengyao Wang, Richard J. Samworth

**发布时间:** 2025-10-27

**备注:** 44 pages

### GPT解析

### 总结

这篇论文研究了预训练Transformer在不同难度任务上的性能表现，证明其能够达到与任务难度相对应的最优收敛速率，并且对分布偏移具有鲁棒性。

### 背景

研究上下文学习问题，其中Transformer在从混合分布中抽取的任务上进行预训练，该混合分布由不同难度级别的任务分布组成。

### 目的

理解预训练Transformer在不同于预训练分布的测试分布上的性能，特别是当测试分布与预训练分布中对应难度级别的分布存在卡方散度限制的偏移时。

### 方法

考虑非参数回归问题和多指标模型，分析大型预训练Transformer在这些模型上的收敛性能。

### 主要发现

预训练Transformer能够达到与任务难度级别相对应的最优收敛速率，并且在卡方散度球内的测试分布上是一致的；Transformer在较容易任务上收敛更快，且对分布偏移具有鲁棒性。

### 结论

预训练Transformer即使面对分布偏移也能保持最优性能，其性能优于理论上能够访问测试分布的估计器，提供了比最小最大下界更合适的最优性保证。

### 翻译

我们研究上下文学习问题，其中Transformer在从混合分布中抽取的任务上进行预训练，称为预训练先验，其中每个混合分量是针对特定难度级别的任务分布。我们的目标是理解预训练Transformer在不同于测试分布上的性能表现，该测试分布由固定难度的任务组成，并且相对于对应难度级别的分布可能存在分布偏移，但卡方散度至多为某个常数。特别是，我们考虑具有随机光滑性的非参数回归问题，以及具有随机光滑性和随机有效维度的多指标模型。我们证明，在足够数据上预训练的大型Transformer能够达到与难度级别相对应的最优收敛速率，并且在卡方散度球内的测试分布上是一致的。因此，预训练的Transformer能够在较容易的任务上实现更快的收敛速率，并且对测试时的分布偏移具有鲁棒性。最后，我们证明即使估计器能够访问测试分布，其在测试分布上的期望风险的收敛速率也不会比预训练的Transformer更快，从而提供了比最小最大下界更合适的最优性保证。


### 论文摘要

We study in-context learning problems where a Transformer is pretrained on tasks drawn from a mixture distribution $\pi=\sum_{\alpha\in\mathcal{A}} \lambda_{\alpha} \pi_{\alpha}$, called the pretraining prior, in which each mixture component $\pi_{\alpha}$ is a distribution on tasks of a specific difficulty level indexed by $\alpha$. Our goal is to understand the performance of the pretrained Transformer when evaluated on a different test distribution $\mu$, consisting of tasks of fixed difficulty $\beta\in\mathcal{A}$, and with potential distribution shift relative to $\pi_\beta$, subject to the chi-squared divergence $\chi^2(\mu,\pi_{\beta})$ being at most $\kappa$. In particular, we consider nonparametric regression problems with random smoothness, and multi-index models with random smoothness as well as random effective dimension. We prove that a large Transformer pretrained on sufficient data achieves the optimal rate of convergence corresponding to the difficulty level $\beta$, uniformly over test distributions $\mu$ in the chi-squared divergence ball. Thus, the pretrained Transformer is able to achieve faster rates of convergence on easier tasks and is robust to distribution shift at test time. Finally, we prove that even if an estimator had access to the test distribution $\mu$, the convergence rate of its expected risk over $\mu$ could not be faster than that of our pretrained Transformers, thereby providing a more appropriate optimality guarantee than minimax lower bounds.

---

## 27. Are ASR foundation models generalized enough to capture features of regional dialects for low-resource languages?

**论文链接:** [http://arxiv.org/abs/2510.23252v1](http://arxiv.org/abs/2510.23252v1)

**作者:** Tawsif Tashwar Dipto, Azmol Hossain, Rubayet Sabbir Faruque, Md. Rezuwan Hassan, Kanij Fatema, Tanmoy Shome, Ruwad Naswan, Md. Foriduzzaman Zihad, Mohaymen Ul Anam, Nazia Tasnim, Hasan Mahmud, Md Kamrul Hasan, Md. Mehedi Hasan Shawon, Farig Sadeque, Tahsin Reasat

**发布时间:** 2025-10-27

**备注:** This manuscript contains 11 pages, 5 tables and 16 figures This was  accepted at International Joint Conference on Natural Language Processing &  Asia-Pacific Chapter of the Association for Computational Linguistics  (IJCNLP-AACL) 2025

### GPT解析

### 总结

本研究开发了名为Ben-10的孟加拉语音转文本语料库，研究了方言变化对自动语音识别(ASR)的影响，发现语音基础模型在区域方言ASR中表现不佳，但方言特定模型训练可缓解此问题。

### 背景

传统语音识别研究大多使用标准形式处理低资源语言，而区域方言的自动语音识别(ASR)被视为微调任务。

### 目的

研究方言变化对自动语音识别(ASR)的影响。

### 方法

开发了一个78小时标注的孟加拉语音转文本(STT)语料库，命名为Ben-10，并从语言学和数据驱动角度进行研究。

### 主要发现

语音基础模型在区域方言ASR中表现不佳，无论是零样本还是微调设置；所有深度学习方法都难以在方言变化条件下建模语音数据，但方言特定的模型训练可以缓解这一问题。

### 结论

该数据集可作为ASR算法在资源受限条件下建模的分布外(OOD)资源。

### 翻译

传统语音识别建模研究大多依赖标准形式处理大多数低资源语言，而区域方言的自动语音识别(ASR)被视为微调任务。为研究对方言变化对ASR的影响，我们开发了一个名为Ben-10的78小时标注的孟加拉语音转文本(STT)语料库。从语言学和数据驱动角度的研究表明，语音基础模型在区域方言ASR中表现严重不佳，无论是在零样本还是微调设置下。我们观察到所有深度学习方法都难以在方言变化条件下建模语音数据，但方言特定的模型训练可以缓解这一问题。我们的数据集也可作为ASR算法在资源受限条件下建模的分布外(OOD)资源。该项目开发的数据集和代码已公开可用。


### 论文摘要

Conventional research on speech recognition modeling relies on the canonical form for most low-resource languages while automatic speech recognition (ASR) for regional dialects is treated as a fine-tuning task. To investigate the effects of dialectal variations on ASR we develop a 78-hour annotated Bengali Speech-to-Text (STT) corpus named Ben-10. Investigation from linguistic and data-driven perspectives shows that speech foundation models struggle heavily in regional dialect ASR, both in zero-shot and fine-tuned settings. We observe that all deep learning methods struggle to model speech data under dialectal variations but dialect specific model training alleviates the issue. Our dataset also serves as a out of-distribution (OOD) resource for ASR modeling under constrained resources in ASR algorithms. The dataset and code developed for this project are publicly available

---

## 28. Finding 3D Scene Analogies with Multimodal Foundation Models

**论文链接:** [http://arxiv.org/abs/2510.23184v1](http://arxiv.org/abs/2510.23184v1)

**作者:** Junho Kim, Young Min Kim

**发布时间:** 2025-10-27

**备注:** Accepted to FM4RoboPlan workshop at RSS 2025

### GPT解析

### 总结

本文提出使用多模态基础模型在零样本、开放词汇设置下寻找3D场景类比，通过混合神经表示实现复杂场景间的准确对应关系，支持轨迹和航路点转移。

### 背景

将当前观察与先验经验连接有助于机器人在新3D环境中适应和规划。3D场景类比可作为平滑映射对齐具有共同空间关系的场景区域，支持轨迹或航路点转移，可用于模仿学习示范转移或跨场景任务规划。

### 目的

提出使用多模态基础模型在零样本、开放词汇设置下寻找3D场景类比，避免现有方法需要的额外训练和固定物体词汇表限制。

### 方法

采用混合神经表示场景，包括基于视觉语言模型特征的稀疏图和来自3D形状基础模型的特征场。通过粗到细方式寻找3D场景类比，首先对齐图，然后用特征场细化对应关系。

### 主要发现

该方法能够建立复杂场景之间的准确对应关系，并成功应用于轨迹和航路点转移。

### 结论

使用多模态基础模型可以在无需额外训练和固定词汇表的情况下实现3D场景类比，为机器人在新环境中的适应和规划提供了有效方法。

### 翻译

将当前观察与先验经验连接起来有助于机器人在新的、未见过的3D环境中进行适应和规划。最近，3D场景类比被提出用于连接两个3D场景，这些是平滑的映射，能够对齐具有共同空间关系的场景区域。这些映射可以支持轨迹或航路点的详细转移，可能支持模仿学习的示范转移或跨场景的任务规划转移。然而，现有方法需要额外的训练和固定的物体词汇表。在本文中，我们提出使用多模态基础模型在零样本、开放词汇设置下寻找3D场景类比。我们方法的核心是场景的混合神经表示，包括基于视觉语言模型特征的稀疏图和来自3D形状基础模型的特征场。3D场景类比通过粗到细的方式找到，首先对齐图，然后使用特征场细化对应关系。我们的方法能够建立复杂场景之间的准确对应关系，我们展示了在轨迹和航路点转移中的应用。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文解决的是如何在没有额外训练和固定词汇表限制的情况下，找到3D场景之间的类比关系。这个问题很重要，因为它能帮助机器人将新环境与已知经验联系起来，从而在未知环境中更好地进行规划和行动。3D场景类比可以创建场景间的平滑映射，支持轨迹或路径点的转移，可用于模仿学习或跨场景的任务规划。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者注意到现有方法要么需要特定领域训练（神经描述符场），要么依赖语义标签（场景图匹配），限制了泛化能力。因此，作者转向利用已在大量多模态数据上训练的基础模型。方法借鉴了视觉语言模型（CLIP）提取对象特征、3D形状模型（PartField）构建特征场、图匹配技术、DBSCAN聚类和薄板样条拟合等现有技术，但将它们创新地组合成一个新的框架。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是使用多模态基础模型在零样本、开放词汇表设置下寻找3D场景类比，通过混合神经表示（稀疏图+密集特征场）实现从粗到细的场景类比估计。流程包括：1)构建场景图（对象节点+CLIP特征）和特征场（PartField特征）；2)图匹配获得粗粒度对象关联；3)DBSCAN聚类并拟合仿射映射；4)基于特征场优化局部位移；5)用薄板样条拟合得到最终映射。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)使用多模态基础模型实现零样本3D场景类比；2)提出混合神经表示方法结合稀疏图和密集场；3)采用从粗到细的估计策略；4)支持开放词汇表场景。相比之前工作，不同之处在于：无需特定领域训练（优于神经场景图方法）、不需要预知语义标签（优于场景图匹配方法）、能处理复杂场景且映射准确性更高。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '本文提出了一种基于多模态基础模型的3D场景类比方法，通过结合视觉语言和3D形状特征的混合表示，实现了零样本、开放词汇表场景下的高精度场景映射，为机器人规划和模仿学习提供了新的可能性。'}


### 论文摘要

Connecting current observations with prior experiences helps robots adapt and plan in new, unseen 3D environments. Recently, 3D scene analogies have been proposed to connect two 3D scenes, which are smooth maps that align scene regions with common spatial relationships. These maps enable detailed transfer of trajectories or waypoints, potentially supporting demonstration transfer for imitation learning or task plan transfer across scenes. However, existing methods for the task require additional training and fixed object vocabularies. In this work, we propose to use multimodal foundation models for finding 3D scene analogies in a zero-shot, open-vocabulary setting. Central to our approach is a hybrid neural representation of scenes that consists of a sparse graph based on vision-language model features and a feature field derived from 3D shape foundation models. 3D scene analogies are then found in a coarse-to-fine manner, by first aligning the graph and refining the correspondence with feature fields. Our method can establish accurate correspondences between complex scenes, and we showcase applications in trajectory and waypoint transfer.

---

## 29. Guiding Skill Discovery with Foundation Models

**论文链接:** [http://arxiv.org/abs/2510.23167v1](http://arxiv.org/abs/2510.23167v1)

**作者:** Zhao Yang, Thomas M. Moerland, Mike Preuss, Aske Plaat, Vincent François-Lavet, Edward S. Hu

**发布时间:** 2025-10-27

### GPT解析

### 总结

本研究提出了一种名为Foundation model Guided (FoG)的技能发现方法，通过基础模型将人类意图融入技能发现过程，解决了现有方法只关注技能多样性而忽略人类偏好导致不理想行为的问题。

### 背景

现有的技能发现方法仅专注于最大化技能多样性，不考虑人类偏好，这会导致不理想甚至危险的行为。例如，使用先前方法训练的猎豹机器人学会向各个方向翻滚以最大化技能多样性，而非我们期望的奔跑行为。

### 目的

开发一种能够将人类意图融入技能发现过程的方法，从而学习符合人类偏好的多样化技能，避免不理想和危险行为。

### 方法

FoG方法从基础模型中提取评分函数，根据人类意图评估状态，为理想状态赋予更高值，不理想状态赋予更低值。然后使用这些分数重新加权技能发现算法的奖励，通过优化重新加权的奖励来指导技能发现过程。

### 主要发现

FoG成功消除了不理想行为，如翻转或翻滚，并在基于状态和基于像素的任务中有效避免了危险区域。此外，该方法能够发现涉及难以定义的行为的技能。

### 结论

FoG方法通过将人类意图融入技能发现过程，解决了现有方法只关注多样性而忽略人类偏好的问题，使强化学习能够学习更符合人类期望的技能，从而加速下游任务的强化学习过程。

### 翻译

无需手工设计的奖励函数即可学习多样化技能，可以加速下游任务中的强化学习。然而，现有的技能发现方法只专注于最大化技能多样性，而没有考虑人类偏好，这会导致不理想的行为甚至危险技能。例如，使用先前方法训练的猎豹机器人学会向各个方向翻滚以最大化技能多样性，而我们更希望它能够奔跑而不翻转或进入危险区域。在这项工作中，我们提出了一种基础模型引导(FoG)的技能发现方法，通过基础模型将人类意图融入技能发现。具体来说，FoG从基础模型中提取评分函数，根据人类意图评估状态，为理想状态赋予更高值，为不理想状态赋予更低值。然后使用这些分数重新加权技能发现算法的奖励。通过优化重新加权的技能发现奖励，FoG成功消除了不理想行为，如翻转或翻滚，并在基于状态和基于像素的任务中避免了危险区域。有趣的是，我们表明FoG可以发现涉及难以定义的行为的技能。交互式可视化可通过https://sites.google.com/view/submission-fog获取。


### 论文摘要

Learning diverse skills without hand-crafted reward functions could accelerate reinforcement learning in downstream tasks. However, existing skill discovery methods focus solely on maximizing the diversity of skills without considering human preferences, which leads to undesirable behaviors and possibly dangerous skills. For instance, a cheetah robot trained using previous methods learns to roll in all directions to maximize skill diversity, whereas we would prefer it to run without flipping or entering hazardous areas. In this work, we propose a Foundation model Guided (FoG) skill discovery method, which incorporates human intentions into skill discovery through foundation models. Specifically, FoG extracts a score function from foundation models to evaluate states based on human intentions, assigning higher values to desirable states and lower to undesirable ones. These scores are then used to re-weight the rewards of skill discovery algorithms. By optimizing the re-weighted skill discovery rewards, FoG successfully learns to eliminate undesirable behaviors, such as flipping or rolling, and to avoid hazardous areas in both state-based and pixel-based tasks. Interestingly, we show that FoG can discover skills involving behaviors that are difficult to define. Interactive visualisations are available from https://sites.google.com/view/submission-fog.

---

## 30. Implicit Modeling for Transferability Estimation of Vision Foundation Models

**论文链接:** [http://arxiv.org/abs/2510.23145v1](http://arxiv.org/abs/2510.23145v1)

**作者:** Yaoyan Zheng, Huiqun Wang, Nan Zhou, Di Huang

**发布时间:** 2025-10-27

**备注:** Accepted by NeurIPS 2025

### GPT解析

### 总结

本文提出了一种名为隐式可迁移性建模(ITM)的新框架，用于评估预训练模型对下游任务的适用性，无需进行完整的微调过程。

### 背景

可迁移性估计能够识别最适合下游任务的预训练模型，避免完整微调的高计算成本，促进模型部署和预训练-微调范式发展。然而，现有方法在评估具有多样化架构、训练策略和任务对齐的新兴预训练模型时，准确性有限。

### 目的

开发一种能够准确评估各种类型预训练模型可迁移性的新方法，使其能够在更广泛的模型和下游任务上实现泛化。

### 方法

提出隐式可迁移性建模(ITM)框架，隐式建模每个模型的内在可迁移性，并结合分治变分近似(DVA)策略来有效近似嵌入空间的演化过程。

### 主要发现

在涵盖多种训练策略和模型类型的综合基准测试中，ITM在稳定性、有效性和效率方面持续优于现有方法。

### 结论

ITM框架为评估新兴预训练模型的可迁移性提供了更准确、更高效的解决方案，有助于推动预训练和微调范式的发展。

### 翻译

可迁移性估计能够识别最适合下游任务的预训练模型，而无需承担完整微调的高计算成本。这种能力促进了部署并推动了预训练和微调范式的发展。然而，现有方法在评估具有多样化架构、训练策略和任务对齐的新兴预训练模型的可迁移性时，往往难以准确评估。在这项工作中，我们提出了隐式可迁移性建模(ITM)，这是一个新颖的框架，它隐式地建模每个模型的内在可迁移性，并结合分治变分近似(DVA)策略来有效近似嵌入空间的演化。这种设计使模型能够在更广泛的模型和下游任务上实现泛化。在涵盖广泛训练策略和更多样化模型类型的综合基准上的大量实验表明，ITM在稳定性、有效性和效率方面持续优于现有方法。


### 论文摘要

Transferability estimation identifies the best pre-trained models for downstream tasks without incurring the high computational cost of full fine-tuning. This capability facilitates deployment and advances the pre-training and fine-tuning paradigm. However, existing methods often struggle to accurately assess transferability for emerging pre-trained models with diverse architectures, training strategies, and task alignments. In this work, we propose Implicit Transferability Modeling (ITM), a novel framework that implicitly models each model's intrinsic transferability, coupled with a Divide-and-Conquer Variational Approximation (DVA) strategy to efficiently approximate embedding space evolution. This design enables generalization across a broader range of models and downstream tasks. Extensive experiments on a comprehensive benchmark--spanning extensive training regimes and a wider variety of model types--demonstrate that ITM consistently outperforms existing methods in terms of stability, effectiveness, and efficiency.

---

## 31. OmniDexGrasp: Generalizable Dexterous Grasping via Foundation Model and Force Feedback

**论文链接:** [http://arxiv.org/abs/2510.23119v1](http://arxiv.org/abs/2510.23119v1)

**作者:** Yi-Lin Wei, Zhexi Luo, Yuhao Lin, Mu Lin, Zhizhao Liang, Shuoyu Chen, Wei-Shi Zheng

**发布时间:** 2025-10-27

**备注:** Project page: https://isee-laboratory.github.io/OmniDexGrasp/

### GPT解析

### 总结

论文提出了OmniDexGrasp框架，结合基础模型与转移控制策略，实现了机器人根据人类命令灵巧抓取和操作物体的通用能力。

### 背景

让机器人根据人类命令灵巧地抓取和操作物体是机器人学的一个有前景的方向，但现有方法由于语义灵巧抓取数据集规模有限，难以在不同物体或任务上泛化。

### 目的

解决基础模型与物理机器人执行之间的差距问题，开发一个能在用户提示、灵巧抓取和抓取任务方面实现全能力的通用框架。

### 方法

OmniDexGrasp集成了三个关键模块：使用基础模型生成人类抓取图像增强泛化能力；人类图像到机器人行动的转移策略实现全灵巧抓取；力感知自适应抓取策略确保稳健稳定的抓取执行。

### 主要发现

在模拟和真实机器人上的实验验证了OmniDexGrasp在不同用户提示、抓取任务和灵巧手方面的有效性，且可扩展到灵巧操作任务。

### 结论

OmniDexGrasp通过结合基础模型与转移控制策略，显著提升了机器人灵巧抓取和操作能力，具有广泛的适用性和可扩展性。

### 翻译

使机器人能够根据人类命令灵巧地抓取和操作物体是机器人学中的一个有前景的方向。然而，由于语义灵巧抓取数据集的规模有限，现有方法难以在多样化的物体或任务上泛化。基础模型提供了一种增强泛化的新方法，但由于抽象模型知识与物理机器人执行之间的差距，直接利用它们生成可行的机器人行动仍然具有挑战性。为了解决这些挑战，我们提出了OmniDexGrasp，一个通用框架，通过结合基础模型与转移和控制策略，在用户提示、灵巧抓取和抓取任务方面实现全能力。OmniDexGrasp集成了三个关键模块：(i) 使用基础模型生成人类抓取图像，增强泛化能力，支持用户提示和任务的全能力；(ii) 人类图像到机器人行动的转移策略将人类演示转化为可执行的机器人行动，实现全灵巧抓取；(iii) 力感知自适应抓取策略确保稳健和稳定的抓取执行。在模拟和真实机器人上的实验验证了OmniDexGrasp在不同用户提示、抓取任务和灵巧手方面的有效性，进一步的结果显示其可扩展到灵巧操作任务。


### 论文摘要

Enabling robots to dexterously grasp and manipulate objects based on human commands is a promising direction in robotics. However, existing approaches are challenging to generalize across diverse objects or tasks due to the limited scale of semantic dexterous grasp datasets. Foundation models offer a new way to enhance generalization, yet directly leveraging them to generate feasible robotic actions remains challenging due to the gap between abstract model knowledge and physical robot execution. To address these challenges, we propose OmniDexGrasp, a generalizable framework that achieves omni-capabilities in user prompting, dexterous embodiment, and grasping tasks by combining foundation models with the transfer and control strategies. OmniDexGrasp integrates three key modules: (i) foundation models are used to enhance generalization by generating human grasp images supporting omni-capability of user prompt and task; (ii) a human-image-to-robot-action transfer strategy converts human demonstrations into executable robot actions, enabling omni dexterous embodiment; (iii) force-aware adaptive grasp strategy ensures robust and stable grasp execution. Experiments in simulation and on real robots validate the effectiveness of OmniDexGrasp on diverse user prompts, grasp task and dexterous hands, and further results show its extensibility to dexterous manipulation tasks.

---

## 32. Eigenfunction Extraction for Ordered Representation Learning

**论文链接:** [http://arxiv.org/abs/2510.24672v1](http://arxiv.org/abs/2510.24672v1)

**作者:** Burak Varıcı, Che-Ping Tsai, Ritabrata Ray, Nicholas M. Boffi, Pradeep Ravikumar

**发布时间:** 2025-10-28

### GPT解析

### 总结

本研究提出了一种通用框架，用于提取有序且可识别的特征函数，解决了现有表示学习方法只能恢复核的前几个特征函数线性张成空间的问题。

### 背景

表示学习的最新进展表明，广泛使用的目标（如对比和非对比方法）隐式地对由输入与其上下文之间的关系诱导的上下文核执行谱分解。

### 目的

开发一个能提取有序且可识别特征函数的通用框架，满足与上下文核兼容和可扩展到现代设置的需求。

### 方法

展示低秩近似和瑞利商优化两种主要方法论范式如何与特征函数提取框架对齐，基于模块化构建块设计。

### 主要发现

恢复的特征值可作为特征选择的有效重要性分数，通过自适应维度表示实现原则性的效率-准确性权衡。

### 结论

所提出的方法在合成核和真实图像数据集上均得到验证，能够提供更精确的特征提取和理解，有助于特征选择和模型效率与准确性的平衡。

### 翻译

表示学习的最新进展表明，广泛使用的目标（如对比和非对比方法）隐式地对由输入与其上下文之间的关系诱导的上下文核执行谱分解。然而，这些方法只能恢复核的前几个特征函数的线性张成空间，而精确的谱分解对于理解特征排序和重要性至关重要。在本研究中，我们提出一个通用框架来提取有序且可识别的特征函数，基于满足关键需求的模块化构建块设计，包括与上下文核的兼容性和可扩展到现代设置的能力。然后，我们展示了两种主要方法论范式（低秩近似和瑞利商优化）如何与这一特征函数提取框架对齐。最后，我们在合成核上验证了我们的方法，并在真实图像数据集上证明恢复的特征值可作为特征选择的有效重要性分数，通过自适应维度表示实现原则性的效率-准确性权衡。


### 论文摘要

Recent advances in representation learning reveal that widely used objectives, such as contrastive and non-contrastive, implicitly perform spectral decomposition of a contextual kernel, induced by the relationship between inputs and their contexts. Yet, these methods recover only the linear span of top eigenfunctions of the kernel, whereas exact spectral decomposition is essential for understanding feature ordering and importance. In this work, we propose a general framework to extract ordered and identifiable eigenfunctions, based on modular building blocks designed to satisfy key desiderata, including compatibility with the contextual kernel and scalability to modern settings. We then show how two main methodological paradigms, low-rank approximation and Rayleigh quotient optimization, align with this framework for eigenfunction extraction. Finally, we validate our approach on synthetic kernels and demonstrate on real-world image datasets that the recovered eigenvalues act as effective importance scores for feature selection, enabling principled efficiency-accuracy tradeoffs via adaptive-dimensional representations.

---

## 33. Perception Learning: A Formal Separation of Sensory Representation Learning from Decision Learning

**论文链接:** [http://arxiv.org/abs/2510.24356v1](http://arxiv.org/abs/2510.24356v1)

**作者:** Suman Sanyal

**发布时间:** 2025-10-28

### GPT解析

### 总结

本文提出感知学习(PeL)范式，使用任务无关信号优化智能体的感官接口，与下游决策学习解耦。PeL直接针对无标签的感知属性，通过表示不变的指标评估，形式化了感知与决策的分离，并证明PeL更新与贝叶斯任务风险梯度正交。

### 背景

传统学习中感知和决策通常紧密耦合，限制了智能体发展通用感知能力，因为学习过于关注特定任务表现而忽略基本感知属性。

### 目的

提出PeL范式解耦感知与决策；优化智能体感官接口；定义独立于任务目标的感知属性；提供评估感知质量的指标。

### 方法

使用任务无关信号优化感官接口；将感知学习与下游决策学习解耦；针对稳定性、信息量、几何结构等感知属性优化；使用表示不变的指标评估；形式化感知与决策分离；证明PeL更新与贝叶斯任务风险梯度正交；提供任务无关评估指标。

### 主要发现

感知与决策可成功分离；可定义独立于目标或重新参数化的感知属性；保持不变量的PeL更新与贝叶斯任务风险梯度正交；提供有效评估指标认证感知质量。

### 结论

PeL范式为智能体提供新感知学习框架，通过解耦感知与决策，发展更通用、鲁棒的感知能力，提高任务表现并增强环境理解和适应能力。

### 翻译

我们引入感知学习(PeL)，一种使用任务无关信号优化智能体感官接口的范式，与下游决策学习解耦。PeL直接针对无标签的感知属性，如对扰动的稳定性、信息量而不崩溃、受控几何结构等，通过表示不变的指标进行评估。我们形式化了感知与决策的分离，定义了独立于目标或重新参数化的感知属性，并证明了保持足够不变量的PeL更新与贝叶斯任务风险梯度正交。此外，我们还提供了一套任务无关的评估指标来认证感知质量。


### 论文摘要

We introduce Perception Learning (PeL), a paradigm that optimizes an agent's sensory interface $f_\phi:\mathcal{X}\to\mathcal{Z}$ using task-agnostic signals, decoupled from downstream decision learning $g_\theta:\mathcal{Z}\to\mathcal{Y}$. PeL directly targets label-free perceptual properties, such as stability to nuisances, informativeness without collapse, and controlled geometry, assessed via objective representation-invariant metrics. We formalize the separation of perception and decision, define perceptual properties independent of objectives or reparameterizations, and prove that PeL updates preserving sufficient invariants are orthogonal to Bayes task-risk gradients. Additionally, we provide a suite of task-agnostic evaluation metrics to certify perceptual quality.

---

## 34. DynaRend: Learning 3D Dynamics via Masked Future Rendering for Robotic Manipulation

**论文链接:** [http://arxiv.org/abs/2510.24261v1](http://arxiv.org/abs/2510.24261v1)

**作者:** Jingyi Tian, Le Wang, Sanping Zhou, Sen Wang, Jiayi Li, Gang Hua

**发布时间:** 2025-10-28

**备注:** Accepted to NeurIPS 2025

### GPT解析

### 总结

本文提出DynaRend框架，通过可微分体积渲染学习3D感知和动态信息的三平面特征，统一捕获空间几何、未来动态和任务语义，有效提升机器人操作任务的泛化能力。

### 背景

由于缺乏多样化的真实世界训练数据，学习可泛化的机器人操作策略仍面临挑战。现有方法要么依赖2D视觉预训练范式关注静态语义或几何，要么利用视频预测模型强调2D动态，无法同时学习操作所需的几何、语义和动态信息。

### 目的

开发一种能够联合学习几何、语义和动态的表示学习框架，解决机器人操作任务中数据稀缺和泛化能力不足的问题。

### 方法

提出DynaRend表示学习框架，通过掩码重建和未来预测学习三平面特征，在多视图RGB-D视频数据上进行预训练，并通过动作价值图预测将学习到的表示转移到下游任务。

### 主要发现

在RLBench和Colosseum基准测试及真实世界实验中，DynaRend在策略成功率、环境扰动泛化能力和多样化任务实际适用性方面均有显著提升。

### 结论

DynaRend成功解决了现有方法的局限性，能够有效联合学习几何、语义和动态信息，大幅提高机器人操作策略的泛化能力和实际应用效果。

### 翻译

由于缺乏多样化的真实世界训练数据，学习可泛化的机器人操作策略仍然是一个关键挑战。尽管最近的方法尝试通过自监督表示学习来缓解这一问题，但大多数方法要么依赖于2D视觉预训练范式如掩码图像建模，主要关注静态语义或场景几何，要么利用大规模视频预测模型强调2D动态，因此无法有效操作所需的几何、语义和动态的联合学习。在本文中，我们提出DynaRend，一个表示学习框架，通过可微分体积渲染进行掩码重建和未来预测，学习3D感知和动态信息的三平面特征。通过在多视图RGB-D视频数据上预训练，DynaRend能够在统一的三平面表示中同时捕获空间几何、未来动态和任务语义。学习到的表示可以通过动作价值图预测有效地转移到下游机器人操作任务中。我们在两个具有挑战性的基准测试RLBench和Colosseum以及真实世界机器人实验中评估DynaRend，证明了其在策略成功率、对环境扰动的泛化能力以及在不同操作任务中的实际适用性方面都有显著改进。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决机器人操作策略难以泛化的问题，原因是缺乏多样化的真实世界训练数据。现有方法要么依赖2D视觉预训练（关注静态语义或几何），要么使用视频预测模型（强调2D动态），无法同时学习有效操作所需的几何、语义和动态信息。这个问题很重要，因为机器人操作需要理解3D环境变化，真实世界数据收集成本高，且有效的操作需要综合理解场景结构、物体语义和动态变化。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先分析了现有方法的局限性：2D预训练缺乏3D感知，3D方法结构复杂难以扩展，渲染方法需要密集相机设置不实用。作者设计了一个统一框架，通过掩码重建和未来预测学习3D感知和动态信息的三平面特征。借鉴了掩码图像建模、神经渲染、视频预测和三平面表示等技术，但创新性地结合了这些方法，并引入目标视图增强技术提高真实世界适用性。', '这个方法的核心思想是什么？整体实现流程是怎样的？': "核心思想是通过'掩码未来渲染'学习3D感知和动态信息的三平面特征，同时捕获场景几何、语义信息和未来动态变化。流程包括：1)从多视图RGB-D重建点云并投影为三平面特征；2)随机掩码部分特征，通过重建网络恢复当前场景，通过预测网络生成未来场景；3)对重建和预测结果进行体积渲染，用RGB、语义和深度损失进行监督；4)利用预训练模型合成新视图增强监督；5)预训练后添加动作解码器，在下游任务上微调预测动作值图。", '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)统一的3D表示学习框架，联合学习几何、动态和语义；2)掩码未来渲染技术，结合重建和预测两个互补目标；3)目标视图增强技术，减少对密集相机设置的依赖；4)多任务渲染损失，同时优化RGB、语义和深度。相比之前工作，DynaRend具有明确的3D空间感知能力，能捕获3D动态而非2D，使用更高效简洁的三平面表示，且通过目标视图增强提高了真实世界适用性。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'DynaRend通过掩码未来渲染技术，首次在统一的三平面表示中联合学习空间几何、未来动态和任务语义，显著提高了机器人操作策略的泛化能力和环境适应性。'}


### 论文摘要

Learning generalizable robotic manipulation policies remains a key challenge due to the scarcity of diverse real-world training data. While recent approaches have attempted to mitigate this through self-supervised representation learning, most either rely on 2D vision pretraining paradigms such as masked image modeling, which primarily focus on static semantics or scene geometry, or utilize large-scale video prediction models that emphasize 2D dynamics, thus failing to jointly learn the geometry, semantics, and dynamics required for effective manipulation. In this paper, we present DynaRend, a representation learning framework that learns 3D-aware and dynamics-informed triplane features via masked reconstruction and future prediction using differentiable volumetric rendering. By pretraining on multi-view RGB-D video data, DynaRend jointly captures spatial geometry, future dynamics, and task semantics in a unified triplane representation. The learned representations can be effectively transferred to downstream robotic manipulation tasks via action value map prediction. We evaluate DynaRend on two challenging benchmarks, RLBench and Colosseum, as well as in real-world robotic experiments, demonstrating substantial improvements in policy success rate, generalization to environmental perturbations, and real-world applicability across diverse manipulation tasks.

---

## 35. Debiasing Reward Models by Representation Learning with Guarantees

**论文链接:** [http://arxiv.org/abs/2510.23751v1](http://arxiv.org/abs/2510.23751v1)

**作者:** Ignavier Ng, Patrick Blöbaum, Siddharth Bhandari, Kun Zhang, Shiva Kasiviswanathan

**发布时间:** 2025-10-27

### GPT解析

### 总结

该研究提出了一种减轻大型语言模型对齐过程中奖励模型偏见的新框架，通过识别和利用非虚假潜在变量来提高模型的稳健性。

### 背景

近期基于人类反馈的强化学习等对齐技术被广泛采用，但这些模型常常利用虚假相关性，如响应长度、歧视性、奉承和概念偏见等问题。

### 目的

提出一个有原则的框架，减轻奖励模型中的偏见，同时保留反映预期潜在偏好的因素。

### 方法

提供数据生成过程的公式化，假设观测数据由虚假和非虚假潜在变量生成；使用变分推断来恢复这些变量并利用它们训练奖励模型。

### 主要发现

非虚假潜在变量理论上可以从数据中识别出来，无论虚假潜在变量的代理变量是否可用。

### 结论

在合成和真实世界数据集上的实验表明，该方法有效减轻了虚假相关性问题，并产生了更稳健的奖励模型。

### 翻译

近期的对齐技术，如基于人类反馈的强化学习，已被广泛采用，通过学习和利用奖励模型使大型语言模型与人类偏好保持一致。在实践中，这些模型常常利用虚假相关性，例如响应长度、歧视性、奉承和概念偏见，这个问题已受到越来越多的关注。在这项工作中，我们提出了一个有原则的框架，在减轻奖励模型中这些偏见的同时，保留反映预期潜在偏好的因素。我们首先提供了数据生成过程的公式化，假设观测数据是由虚假和非虚假的潜在变量生成的。我们有趣地表明，这些非虚假的潜在变量理论上可以从数据中识别出来，无论虚假潜在变量的代理变量是否可用。这进一步启发了一种使用变分推断来恢复这些变量并利用它们训练奖励模型的实用方法。在合成和真实世界数据集上的实验表明，我们的方法有效减轻了虚假相关性问题，并产生了更稳健的奖励模型。


### 论文摘要

Recent alignment techniques, such as reinforcement learning from human feedback, have been widely adopted to align large language models with human preferences by learning and leveraging reward models. In practice, these models often exploit spurious correlations, involving, e.g., response length, discrimination, sycophancy, and conceptual bias, which is a problem that has received increasing attention. In this work, we propose a principled framework that mitigates these biases in reward models while preserving the underlying factors that reflect intended preferences. We first provide a formulation of the data-generating process, assuming that the observed data (e.g., text) is generated from both spurious and non-spurious latent variables. We show that, interestingly, these non-spurious latent variables can be theoretically identified from data, regardless of whether a surrogate for the spurious latent variables is available. This further inspires a practical method that uses variational inference to recover these variables and leverages them to train reward models. Experiments on synthetic and real-world datasets demonstrate that our method effectively mitigates spurious correlation issues and yields more robust reward models.

---

## 36. Manifold Approximation leads to Robust Kernel Alignment

**论文链接:** [http://arxiv.org/abs/2510.22953v1](http://arxiv.org/abs/2510.22953v1)

**作者:** Mohammad Tariqul Islam, Du Liu, Deblina Sarkar

**发布时间:** 2025-10-27

**备注:** 9 pages, 5 figures + supplementary

### GPT解析

### 总结

本文提出了一种新的流形近似核对齐(MKA)方法，用于改进现有的中心化核对齐(CKA)度量方法，使其能够考虑底层流形几何，从而提供更稳健的表征测量基础。

### 背景

中心化核对齐(CKA)是一种流行的度量方法，广泛应用于比较表征、确定网络等价性和神经科学研究。然而，CKA存在局限性，它没有考虑底层流形，并且依赖于许多启发式方法，导致在不同数据尺度下表现不一致。

### 目的

开发一种能够考虑流形几何的核对齐方法，以克服CKA的局限性，提供更稳健的表征测量基础。

### 方法

提出流形近似核对齐(MKA)方法，将流形几何整合到对齐任务中，并推导了MKA的理论框架。在合成数据集和真实世界示例上进行经验评估，以表征和比较MKA及其当代方法。

### 主要发现

考虑流形的核对齐为测量表征提供了更稳健的基础，在表征学习中有潜在应用价值。

### 结论

MKA方法通过考虑底层流形，克服了CKA的局限性，为表征比较提供了更可靠的理论和实践基础。

### 翻译

中心化核对齐(CKA)是一种流行的度量方法，用于比较表征、确定网络等价性和神经科学研究。然而，CKA没有考虑底层流形，并且依赖于许多启发式方法，导致它在不同数据尺度下表现不同。在这项工作中，我们提出了流形近似核对齐(MKA)，将流形几何整合到对齐任务中。我们推导了MKA的理论框架。我们在合成数据集和真实世界示例上进行经验评估，以表征和比较MKA及其当代方法。我们的研究结果表明，考虑流形的核对齐为测量表征提供了更稳健的基础，在表征学习中有潜在应用。


### 论文摘要

Centered kernel alignment (CKA) is a popular metric for comparing representations, determining equivalence of networks, and neuroscience research. However, CKA does not account for the underlying manifold and relies on numerous heuristics that cause it to behave differently at different scales of data. In this work, we propose Manifold approximated Kernel Alignment (MKA), which incorporates manifold geometry into the alignment task. We derive a theoretical framework for MKA. We perform empirical evaluations on synthetic datasets and real-world examples to characterize and compare MKA to its contemporaries. Our findings suggest that manifold-aware kernel alignment provides a more robust foundation for measuring representations, with potential applications in representation learning.

---

## 37. Switchable Token-Specific Codebook Quantization For Face Image Compression

**论文链接:** [http://arxiv.org/abs/2510.22943v2](http://arxiv.org/abs/2510.22943v2)

**作者:** Yongbo Wang, Haonan Wang, Guodong Mu, Ruixin Zhang, Jiaqi Chen, Jingyun Zhang, Jun Wang, Yuan Xie, Zhizhong Zhang, Shouhong Ding

**发布时间:** 2025-10-27

**备注:** NeurIPS 2025 accepted

### GPT解析

### 总结

该论文提出了一种可切换的特定token码本量化方法，用于面部图像压缩，通过为不同类别图像学习不同码本组并为每个token分配独立码本，提高了低比特率下的重建性能。

### 背景

随着视觉数据量不断增加，高效无损的传输及其解释理解成为现代信息系统的关键瓶颈。现有的基于码本的解决方案使用全局共享码本，忽视了面部图像内部的类别相关性和token间的语义差异，导致低比特率下性能不佳。

### 目的

解决全局码本策略在面部图像压缩中的局限性，特别是在低比特率(bpp)情况下的性能问题，提高重建图像的质量和识别准确率。

### 方法

提出可切换的特定token码本量化方法，为不同图像类别学习不同的码本组，为每个token分配独立码本，并用少量比特记录每个token所属的码本组，以减少码本组减小时造成的损失。

### 主要发现

通过使用少量比特记录token所属的码本组，可以在较低总bpp下拥有更多码本总数，增强表达能力并提高重建性能。该方法在面部识别数据集上表现出色，0.05 bpp下重建图像的平均准确率达到93.51%。

### 结论

所提出的可切换特定token码本量化方法有效解决了全局码本策略在面部图像压缩中的局限性，特别是在低比特率情况下，且具有良好的通用性，可集成到现有基于码本的表示学习方法中。

### 翻译

随着视觉数据量的不断增加，高效无损的传输及其随后的解释理解已成为现代信息系统的关键瓶颈。新兴的基于码本的解决方案利用全局共享码本对每个token进行量化和反量化，通过调整token数量或码本大小来控制bpp。然而，对于富含属性的面部图像，这种全局码本策略忽视了图像内部的类别特定相关性以及token之间的语义差异，导致性能不佳，特别是在低bpp情况下。受这些观察的启发，我们提出了一种用于面部图像压缩的可切换特定token码本量化方法，该方法为不同图像类别学习不同的码本组，并为每个token分配独立的码本。通过用少量比特记录每个token所属的码本组，我们的方法可以减少减小每个码本组大小时造成的损失。这使得在较低的总bpp下可以拥有更多的码本总数，从而增强表达能力并提高重建性能。由于其通用设计，我们的方法可以集成到任何现有的基于码本的表示学习方法中，并在面部识别数据集上证明了其有效性，在0.05 bpp下重建图像的平均准确率达到93.51%。


### 论文摘要

With the ever-increasing volume of visual data, the efficient and lossless transmission, along with its subsequent interpretation and understanding, has become a critical bottleneck in modern information systems. The emerged codebook-based solution utilize a globally shared codebook to quantize and dequantize each token, controlling the bpp by adjusting the number of tokens or the codebook size. However, for facial images, which are rich in attributes, such global codebook strategies overlook both the category-specific correlations within images and the semantic differences among tokens, resulting in suboptimal performance, especially at low bpp. Motivated by these observations, we propose a Switchable Token-Specific Codebook Quantization for face image compression, which learns distinct codebook groups for different image categories and assigns an independent codebook to each token. By recording the codebook group to which each token belongs with a small number of bits, our method can reduce the loss incurred when decreasing the size of each codebook group. This enables a larger total number of codebooks under a lower overall bpp, thereby enhancing the expressive capability and improving reconstruction performance. Owing to its generalizable design, our method can be integrated into any existing codebook-based representation learning approach and has demonstrated its effectiveness on face recognition datasets, achieving an average accuracy of 93.51% for reconstructed images at 0.05 bpp.

---

## 38. Transformers from Compressed Representations

**论文链接:** [http://arxiv.org/abs/2510.23665v1](http://arxiv.org/abs/2510.23665v1)

**作者:** Juan C. Leon Alcazar, Mattia Soldan, Mohammad Saatialsoruji, Alejandro Pardo, Hani Itani, Juan Camilo Perez, Bernard Ghanem

**发布时间:** 2025-10-26

### GPT解析

### 总结

这篇论文介绍了TEMPEST方法，它利用压缩文件的字节流结构设计了一种有效的标记化和编码策略，使标准变换器可以直接从压缩数据流中学习语义表示，绕过原始字节级处理或完全媒体解码的需求。

### 背景

压缩文件格式是高效数据存储和传输的基石，但它们在表示学习方面的潜力在很大程度上尚未被探索。

### 目的

开发一种方法，能够直接从压缩数据中学习语义表示，同时减少计算复杂性和内存使用。

### 方法

TEMPEST（TransformErs froM comPressed rEpreSenTations）利用压缩文件固有的字节流结构设计有效的标记化和编码策略，使标准变换器可以直接从压缩数据流中学习语义表示。

### 主要发现

TEMPEST显著减少了语义分类所需的标记数量，从而降低了计算复杂性和内存使用。通过在不同数据集、编码方案和模态上的大量实验，TEMPEST实现了与最先进技术相当的准确性，同时在内存和计算方面提供了效率提升。

### 结论

TEMPEST是一种有效的方法，可以直接从压缩数据中学习语义表示，同时保持高准确性和提高效率。

### 翻译

压缩文件格式是高效数据存储和传输的基石，但它们在表示学习方面的潜力在很大程度上尚未被探索。我们介绍了TEMPEST（来自压缩表示的变换器），这是一种利用压缩文件固有字节流结构设计有效标记化和编码策略的方法。通过利用这种紧凑编码，标准变换器可以直接从压缩数据流中学习语义表示，绕过原始字节级处理或完全媒体解码的需求。我们的提议显著减少了语义分类所需的标记数量，从而降低了计算复杂性和内存使用。通过在不同数据集、编码方案和模态上的大量实验，我们表明TEMPEST实现了与最先进技术相当的准确性，同时在内存和计算方面提供了效率提升。


### 论文摘要

Compressed file formats are the corner stone of efficient data storage and transmission, yet their potential for representation learning remains largely underexplored. We introduce TEMPEST (TransformErs froM comPressed rEpreSenTations), a method that exploits the inherent byte-stream structure of compressed files to design an effective tokenization and encoding strategy. By leveraging this compact encoding, a standard transformer can directly learn semantic representations from compressed data streams, bypassing the need for raw byte-level processing or full media decoding. Our proposal substantially reduces the number of tokens required for semantic classification, thereby lowering both computational complexity and memory usage. Through extensive experiments across diverse datasets, coding schemes, and modalities, we show that TEMPEST achieves accuracy competitive wit the state-of-the-art while delivering efficiency gains in memory and compute.

---

## 39. Learning Without Augmenting: Unsupervised Time Series Representation Learning via Frame Projections

**论文链接:** [http://arxiv.org/abs/2510.22655v1](http://arxiv.org/abs/2510.22655v1)

**作者:** Berken Utku Demirel, Christian Holz

**发布时间:** 2025-10-26

**备注:** Published at the Conference on Neural Information Processing Systems  (NeurIPS) 2025

### GPT解析

### 总结

本文提出了一种使用正交基和过完备帧替代传统数据增强的自监督学习方法，通过联合利用不同流形的互补几何特性，在多个时间序列任务上实现了高达15-20%的性能提升。

### 背景

自监督学习(SSL)是一种无需标记数据即可学习表征的强大范式，但大多数SSL方法依赖于强大、成熟、手工制作的数据增强技术，这需要领域特定知识并可能限制模型的泛化能力。

### 目的

提出一种无监督表征学习方法，用正交基和过完备帧生成视图来替代传统的数据增强技术，避免其带来的限制。

### 方法

使用正交基和过完备帧生成视图替代传统数据增强，联合利用从正交和过完备空间学习到的不同流形上的互补几何特性。

### 主要发现

从正交和过完备空间学习到的嵌入位于由不同空间中表示样本所引入的几何偏差形成的不同流形上，通过联合利用这些不同流形的互补几何特性，可以在不通过强增强人为增加数据多样性的情况下实现优越性能。

### 结论

在五个时间序列任务上的九个数据集上证明了该方法的有效性，在这些信号特性使得数据增强具有挑战性的任务上，不依赖于增强引起的多样性，实现了高达15-20%的性能提升。

### 翻译

自监督学习(SSL)已成为一种无需标记数据即可学习表征的强大范式。大多数SSL方法依赖于强大、成熟、手工制作的数据增强来为表征学习生成多样化视图。然而，设计此类增强需要领域特定知识，并隐式地对模型施加表征不变性，这可能限制泛化能力。在这项工作中，我们提出了一种无监督表征学习方法，使用正交基和过完备帧生成视图来替代数据增强。我们表明，从正交和过完备空间学习到的嵌入位于不同的流形上，这些流形由在不同空间中表示样本所引入的几何偏差形成。通过联合利用这些不同流形的互补几何，我们的方法在不通过强增强人为增加数据多样性的情况下实现了优越性能。我们在五个时间序列任务上的九个数据集上证明了该方法的有效性，在这些任务中，信号特定特性使得数据增强特别具有挑战性。在不依赖于增强引起的多样性的情况下，我们的方法相比现有自监督方法实现了高达15-20%的性能提升。源代码：https://github.com/eth-siplab/Learning-with-FrameProjections


### 论文摘要

Self-supervised learning (SSL) has emerged as a powerful paradigm for learning representations without labeled data. Most SSL approaches rely on strong, well-established, handcrafted data augmentations to generate diverse views for representation learning. However, designing such augmentations requires domain-specific knowledge and implicitly imposes representational invariances on the model, which can limit generalization. In this work, we propose an unsupervised representation learning method that replaces augmentations by generating views using orthonormal bases and overcomplete frames. We show that embeddings learned from orthonormal and overcomplete spaces reside on distinct manifolds, shaped by the geometric biases introduced by representing samples in different spaces. By jointly leveraging the complementary geometry of these distinct manifolds, our approach achieves superior performance without artificially increasing data diversity through strong augmentations. We demonstrate the effectiveness of our method on nine datasets across five temporal sequence tasks, where signal-specific characteristics make data augmentations particularly challenging. Without relying on augmentation-induced diversity, our method achieves performance gains of up to 15--20\% over existing self-supervised approaches. Source code: https://github.com/eth-siplab/Learning-with-FrameProjections

---

## 40. DynaCausal: Dynamic Causality-Aware Root Cause Analysis for Distributed Microservices

**论文链接:** [http://arxiv.org/abs/2510.22613v1](http://arxiv.org/abs/2510.22613v1)

**作者:** Songhan Zhang, Aoyang Fang, Yifan Yang, Ruiyi Cheng, Xiaoying Tang, Pinjia He

**发布时间:** 2025-10-26

### GPT解析

### 总结

DynaCausal是一种动态因果感知框架，用于分布式微服务系统的根本原因分析，通过统一多模态动态信号和动态对比机制，有效解决了级联故障传播建模不足、噪声干扰和概念漂移以及过度依赖服务偏离强度等挑战。

### 背景

云原生微服务支持快速迭代和可扩展部署，但创建了复杂且快速演化的依赖关系，对可靠诊断构成挑战。现有的根本原因分析方法在捕捉动态行为和变化的服务关系方面有限。

### 目的

解决三个关键挑战：级联故障传播建模不足、噪声干扰和概念漂移影响、以及过度依赖服务偏离强度掩盖真正根本原因的问题。

### 方法

提出DynaCausal框架，统一多模态动态信号通过交互感知表征学习捕捉时空依赖关系，引入动态对比机制分离故障指标与噪声，采用因果优先的成对排序目标优化因果归因。

### 主要发现

在公共基准上的评估显示，DynaCausal持续超越最先进方法，平均AC@1达到0.63，绝对增益从0.25到0.46，在高度动态的微服务环境中提供准确且可解释的诊断。

### 结论

DynaCausal通过动态因果感知的方法有效解决了微服务系统故障诊断中的关键挑战，实现了更准确和可解释的根本原因分析。

### 翻译

云原生微服务支持快速迭代和可扩展部署，但也创建了复杂且快速演化的依赖关系，对可靠诊断构成挑战。现有的根本原因分析方法，即使融合了日志、追踪和指标等多模态数据，在捕捉动态行为和变化的服务关系方面仍然有限。三个关键挑战仍然存在：(i)级联故障传播建模不足，(ii)容易受到正常服务行为中噪声干扰和概念漂移的影响，(iii)过度依赖服务偏离强度掩盖了真正的根本原因。为解决这些挑战，我们提出了DynaCausal，一种用于分布式微服务系统RCA的动态因果感知框架。DynaCausal统一多模态动态信号，通过交互感知的表征学习捕捉时空依赖关系。它进一步引入了动态对比机制，将真正的故障指标与上下文噪声分离，并采用因果优先的成对排序目标，明确优化因果归因。在公共基准上的全面评估表明，DynaCausal持续超越最先进的方法，平均AC@1达到0.63，绝对增益从0.25到0.46，在高度动态的微服务环境中提供准确且可解释的诊断。


### 论文摘要

Cloud-native microservices enable rapid iteration and scalable deployment but also create complex, fast-evolving dependencies that challenge reliable diagnosis. Existing root cause analysis (RCA) approaches, even with multi-modal fusion of logs, traces, and metrics, remain limited in capturing dynamic behaviors and shifting service relationships. Three critical challenges persist: (i) inadequate modeling of cascading fault propagation, (ii) vulnerability to noise interference and concept drift in normal service behavior, and (iii) over-reliance on service deviation intensity that obscures true root causes. To address these challenges, we propose DynaCausal, a dynamic causality-aware framework for RCA in distributed microservice systems. DynaCausal unifies multi-modal dynamic signals to capture time-varying spatio-temporal dependencies through interaction-aware representation learning. It further introduces a dynamic contrastive mechanism to disentangle true fault indicators from contextual noise and adopts a causal-prioritized pairwise ranking objective to explicitly optimize causal attribution. Comprehensive evaluations on public benchmarks demonstrate that DynaCausal consistently surpasses state-of-the-art methods, attaining an average AC@1 of 0.63 with absolute gains from 0.25 to 0.46, and delivering both accurate and interpretable diagnoses in highly dynamic microservice environments.

---

## 41. Random Search Neural Networks for Efficient and Expressive Graph Learning

**论文链接:** [http://arxiv.org/abs/2510.22520v1](http://arxiv.org/abs/2510.22520v1)

**作者:** Michael Ito, Danai Koutra, Jenna Wiens

**发布时间:** 2025-10-26

**备注:** NEURIPS 2025; version with full appendix

### GPT解析

### 总结

本文提出了一种新的随机搜索神经网络(RSNNs)方法，用于解决随机游走神经网络(RWNNs)在图表示学习中的局限性。RSNNs通过保证完全节点覆盖的随机搜索，显著降低了采样复杂度，并在理论和实验上都表现出优越性能。

### 背景

随机游走神经网络(RWNNs)已成为图表示学习的一种有前景的方法，利用序列模型处理随机游走。然而，在现实采样约束下，RWNNs往往无法捕捉全局结构，即使在小型图中也是如此，这是因为节点和边覆盖不完整，限制了其表达能力。

### 目的

解决RWNNs在捕捉全局结构方面的局限性，提高图表示学习的效率和表达能力，特别是在稀疏图上的表现。

### 方法

提出随机搜索神经网络(RSNNs)，在保证完全节点覆盖的随机搜索上运行。理论上证明在稀疏图中，只需O(log |V|)次搜索即可实现完全边覆盖，显著低于RWNNs所需的O(|V|)次游走。RSNNs与通用序列模型配对时是通用近似器，且对图同构具有概率不变性。

### 主要发现

在稀疏图中，RSNNs只需O(log |V|)次搜索就能实现完全边覆盖，比RWNNs的O(|V|)次游走大幅降低采样复杂度。RSNNs是通用近似器，且对图同构具有概率不变性。实验表明，RSNNs在分子和蛋白质基准测试中持续优于RWNNs，使用最多减少16倍的采样序列就能达到相当或更好的性能。

### 结论

RSNNs弥合了基于随机游走方法在理论和实践上的差距，为稀疏图上的学习提供了一种高效且具有表达力的框架，在保持高性能的同时显著降低了计算复杂度。

### 翻译

随机游走神经网络(RWNNs)已成为图表示学习的一种有前景的方法，利用序列模型的最新进展来处理随机游走。然而，在现实采样约束下，RWNNs往往无法捕捉全局结构，即使在小型图中也是如此，这是由于节点和边覆盖不完整，限制了其表达能力。为解决这一问题，我们提出了随机搜索神经网络(RSNNs)，它在随机搜索上运行，每个搜索都能保证完全的节点覆盖。理论上，我们证明了在稀疏图中，只需O(log |V|)次搜索就能实现完全边覆盖，与RWNNs所需的O(|V|)次游走相比，显著降低了采样复杂度（假设游走长度随图大小缩放）。此外，当与通用序列模型配对时，RSNNs是通用近似器。最后，我们证明了RSNNs对图同构具有概率不变性，确保其期望是同构不变的图函数。实验上，RSNNs在分子和蛋白质基准测试中持续优于RWNNs，使用最多减少16倍的采样序列就能实现相当或更好的性能。我们的工作弥合了基于随机游走方法在理论和实践上的进展，为在稀疏图上的学习提供了一种高效且具有表达力的框架。


### 论文摘要

Random walk neural networks (RWNNs) have emerged as a promising approach for graph representation learning, leveraging recent advances in sequence models to process random walks. However, under realistic sampling constraints, RWNNs often fail to capture global structure even in small graphs due to incomplete node and edge coverage, limiting their expressivity. To address this, we propose \textit{random search neural networks} (RSNNs), which operate on random searches, each of which guarantees full node coverage. Theoretically, we demonstrate that in sparse graphs, only $O(\log |V|)$ searches are needed to achieve full edge coverage, substantially reducing sampling complexity compared to the $O(|V|)$ walks required by RWNNs (assuming walk lengths scale with graph size). Furthermore, when paired with universal sequence models, RSNNs are universal approximators. We lastly show RSNNs are probabilistically invariant to graph isomorphisms, ensuring their expectation is an isomorphism-invariant graph function. Empirically, RSNNs consistently outperform RWNNs on molecular and protein benchmarks, achieving comparable or superior performance with up to 16$\times$ fewer sampled sequences. Our work bridges theoretical and practical advances in random walk based approaches, offering an efficient and expressive framework for learning on sparse graphs.

---

## 42. MAGIC-Flow: Multiscale Adaptive Conditional Flows for Generation and Interpretable Classification

**论文链接:** [http://arxiv.org/abs/2510.22070v1](http://arxiv.org/abs/2510.22070v1)

**作者:** Luca Caldera, Giacomo Bottacini, Lara Cavinato

**发布时间:** 2025-10-24

### GPT解析

### 总结

本文提出MAGIC-Flow，一种条件多尺度归一化流架构，在单一模块化框架内同时执行生成和分类任务，解决了生成式建模在医学影像等困难领域应用时缺乏任务对齐的问题。

### 背景

生成式建模已成为表示学习的强大范式，但其在医学影像等困难领域的直接应用仍然有限，仅进行生成而不考虑任务对齐，无法为临床应用提供稳健的基础。

### 目的

提出MAGIC-Flow，一种条件多尺度归一化流架构，在单一模块化框架内执行生成和分类，以解决生成式建模在临床应用中的局限性。

### 方法

MAGIC-Flow构建为可逆和可微双射的层次结构，其中雅可比行列式在子变换中因子化，确保了似然的精确计算和稳定的优化。通过基于类标签的条件化，支持可控样本合成和原则性类概率估计，同时可逆性使得样本似然的显式可视化成为可能，为模型推理提供了可解释的视角。

### 主要发现

MAGIC-Flow在相似性、保真度和多样性指标上与顶级基线相当，在多个数据集上解决了扫描噪声下的生成和分类问题，以及模态特定的合成和识别。结果显示MAGIC-Flow创建了真实、多样的样本并改进了分类性能。

### 结论

MAGIC-Flow是数据有限领域中生成和分类的有效策略，对隐私保护增强、鲁棒泛化和可信医疗AI有直接益处。

### 翻译

生成式建模已成为表示学习的强大范式，但其在医学影像等困难领域的直接应用仍然有限：仅进行生成而不考虑任务对齐，无法为临床应用提供稳健的基础。我们提出MAGIC-Flow，一种条件多尺度归一化流架构，在单一模块化框架内执行生成和分类。该模型构建为可逆和可微双射的层次结构，其中雅可比行列式在子变换中因子化。我们展示了这如何确保似然的精确计算和稳定的优化，同时可逆性使得样本似然的显式可视化成为可能，为模型推理提供了可解释的视角。通过基于类标签的条件化，MAGIC-Flow支持可控样本合成和原则性类概率估计，有效地辅助生成性和判别性目标。我们使用相似性、保真度和多样性指标将MAGIC-Flow与顶级基线进行比较。在多个数据集上，它解决了扫描噪声下的生成和分类，以及模态特定的合成和识别。结果表明MAGIC-Flow创建了真实、多样的样本并改进了分类。MAGIC-Flow是数据有限领域中生成和分类的有效策略，对隐私保护增强、鲁棒泛化和可信医疗AI有直接益处。


### 论文摘要

Generative modeling has emerged as a powerful paradigm for representation learning, but its direct applicability to challenging fields like medical imaging remains limited: mere generation, without task alignment, fails to provide a robust foundation for clinical use. We propose MAGIC-Flow, a conditional multiscale normalizing flow architecture that performs generation and classification within a single modular framework. The model is built as a hierarchy of invertible and differentiable bijections, where the Jacobian determinant factorizes across sub-transformations. We show how this ensures exact likelihood computation and stable optimization, while invertibility enables explicit visualization of sample likelihoods, providing an interpretable lens into the model's reasoning. By conditioning on class labels, MAGIC-Flow supports controllable sample synthesis and principled class-probability estimation, effectively aiding both generative and discriminative objectives. We evaluate MAGIC-Flow against top baselines using metrics for similarity, fidelity, and diversity. Across multiple datasets, it addresses generation and classification under scanner noise, and modality-specific synthesis and identification. Results show MAGIC-Flow creates realistic, diverse samples and improves classification. MAGIC-Flow is an effective strategy for generation and classification in data-limited domains, with direct benefits for privacy-preserving augmentation, robust generalization, and trustworthy medical AI.

---

## 43. Predictive Coding Enhances Meta-RL To Achieve Interpretable Bayes-Optimal Belief Representation Under Partial Observability

**论文链接:** [http://arxiv.org/abs/2510.22039v1](http://arxiv.org/abs/2510.22039v1)

**作者:** Po-Chen Kuo, Han Hou, Will Dabney, Edgar Y. Walker

**发布时间:** 2025-10-24

**备注:** Accepted to Annual Conference on Neural Information Processing  Systems (NeurIPS) 2025

### GPT解析

### 总结

本研究探讨了在部分可观测环境中，通过整合预测编码模块到元强化学习中，以学习更紧凑、可解释的贝叶斯最优表示，从而提高代理的适应性和泛化能力。

### 背景

在部分可观测环境中，学习历史信息的紧凑表示对规划和泛化至关重要。现有元强化学习代理虽能接近贝叶斯最优策略，但往往无法学习到紧凑且可解释的贝叶斯最优信念状态，这种表示效率低下限制了代理的适应性和泛化能力。

### 目的

研究将自监督预测编码模块整合到元强化学习中，是否能够促进贝叶斯最优表示的学习，从而提高代理在部分可观测环境中的表现。

### 方法

受神经科学中预测编码和深度强化学习中辅助预测目标的启发，作者将预测编码模块整合到元强化学习框架中，并通过状态机模拟进行了实验验证。

### 主要发现

带有预测模块的元强化学习能够生成更可解释的表示，更好地近似贝叶斯最优信念状态；在需要主动信息获取的挑战性任务中，只有带有预测模块的元强化学习成功学习了最优表示和策略；更好的表示学习能够提高泛化能力。

### 结论

预测学习作为代理在部分可观测环境中有效表示学习的指导原则具有重要价值，能够显著提升代理的性能和泛化能力。

### 翻译

学习历史信息的紧凑表示对于部分可观测环境中的规划和泛化至关重要。虽然元强化学习代理能够获得接近贝叶斯最优的策略，但它们往往无法学习到紧凑、可解释的贝叶斯最优信念状态。这种表示效率低下可能限制了代理的适应性和泛化能力。受神经科学中预测编码的启发——它表明大脑预测感觉输入是贝叶斯推断的神经实现——以及深度强化学习中的辅助预测目标，我们研究了将自监督预测编码模块整合到元强化学习中是否有助于学习贝叶斯最优表示。通过状态机模拟，我们表明带有预测模块的元强化学习能够生成更可解释的表示，更好地近似贝叶斯最优信念状态，与传统的元强化学习相比，在多种任务中表现一致，即使两者都实现了最优策略。在需要主动信息获取的挑战性任务中，只有带有预测模块的元强化学习成功学习了最优表示和策略，而传统元强化学习在表示学习方面表现不足。最后，我们证明了更好的表示学习能够提高泛化能力。我们的结果强烈表明，预测学习作为代理在部分可观测环境中有效表示学习的指导原则具有重要价值。


### 论文摘要

Learning a compact representation of history is critical for planning and generalization in partially observable environments. While meta-reinforcement learning (RL) agents can attain near Bayes-optimal policies, they often fail to learn the compact, interpretable Bayes-optimal belief states. This representational inefficiency potentially limits the agent's adaptability and generalization capacity. Inspired by predictive coding in neuroscience--which suggests that the brain predicts sensory inputs as a neural implementation of Bayesian inference--and by auxiliary predictive objectives in deep RL, we investigate whether integrating self-supervised predictive coding modules into meta-RL can facilitate learning of Bayes-optimal representations. Through state machine simulation, we show that meta-RL with predictive modules consistently generates more interpretable representations that better approximate Bayes-optimal belief states compared to conventional meta-RL across a wide variety of tasks, even when both achieve optimal policies. In challenging tasks requiring active information seeking, only meta-RL with predictive modules successfully learns optimal representations and policies, whereas conventional meta-RL struggles with inadequate representation learning. Finally, we demonstrate that better representation learning leads to improved generalization. Our results strongly suggest the role of predictive learning as a guiding principle for effective representation learning in agents navigating partial observability.

---

## 44. Revisiting Orbital Minimization Method for Neural Operator Decomposition

**论文链接:** [http://arxiv.org/abs/2510.21952v1](http://arxiv.org/abs/2510.21952v1)

**作者:** J. Jon Ryu, Samuel Zhou, Gregory W. Wornell

**发布时间:** 2025-10-24

**备注:** 25 pages, 8 figures. To appear at NeurIPS 2025

### GPT解析

### 总结

该研究重新审视了轨道最小化方法(OMM)，这是一种来自计算物理学的经典优化框架，用于训练神经网络分解线性算子，展示了其在现代机器学习中的实用性和优势。

### 背景

谱分解线性算子在机器学习和科学计算中扮演核心角色。近期工作探索了训练神经网络来近似这些算子的特征函数，为表示学习、动力系统和偏微分方程提供了可扩展的方法。

### 目的

证明轨道最小化方法(OMM)在现代学习流程中的更广泛应用性，并将其调整为训练神经网络分解正半定算子的框架。

### 方法

重新审视轨道最小化方法(OMM)，提供OMM目标一致性的简单线性代数证明，揭示此方法与其他领域独立出现的思想之间的联系，并调整该框架用于训练神经网络。

### 主要发现

轨道最小化方法在一系列基准任务中展示了实际优势，通过现代理论和计算的角度重新审视经典数值方法，可以为数值模拟中部署神经网络提供原则性方法，同时为机器学习提供有效且可扩展的工具。

### 结论

重新审视经典数值方法可以为机器学习和科学计算提供新的视角和实用工具，扩展了神经网络在数值模拟和机器学习中的应用范围。

### 翻译

线性算子的谱分解在机器学习和科学计算的许多领域中扮演着核心角色。近期工作探索了训练神经网络来近似这些算子的特征函数，使表示学习、动力系统和偏微分方程(PDEs)的方法具有可扩展性。在本文中，我们重新审视了来自计算物理学文献的一个经典优化框架，称为轨道最小化方法(OMM)，最初在1990年代提出用于计算化学中的特征值问题。我们提供了OMM目标一致性的简单线性代数证明，并揭示了此方法与不同领域独立出现的几种思想之间的联系。我们的主要目标是证明它在现代学习流程中的更广泛应用性。我们将这一框架调整为训练神经网络来分解正半定算子，并在一系列基准任务中展示了其实际优势。我们的结果强调了如何通过现代理论和计算的角度重新审视经典数值方法，不仅可以为在数值模拟中部署神经网络提供原则性方法，还可以为机器学习提供有效且可扩展的工具。


### 论文摘要

Spectral decomposition of linear operators plays a central role in many areas of machine learning and scientific computing. Recent work has explored training neural networks to approximate eigenfunctions of such operators, enabling scalable approaches to representation learning, dynamical systems, and partial differential equations (PDEs). In this paper, we revisit a classical optimization framework from the computational physics literature known as the \emph{orbital minimization method} (OMM), originally proposed in the 1990s for solving eigenvalue problems in computational chemistry. We provide a simple linear-algebraic proof of the consistency of the OMM objective, and reveal connections between this method and several ideas that have appeared independently across different domains. Our primary goal is to justify its broader applicability in modern learning pipelines. We adapt this framework to train neural networks to decompose positive semidefinite operators, and demonstrate its practical advantages across a range of benchmark tasks. Our results highlight how revisiting classical numerical methods through the lens of modern theory and computation can provide not only a principled approach for deploying neural networks in numerical simulation, but also effective and scalable tools for machine learning.

---

## 45. Structure-Aware Fusion with Progressive Injection for Multimodal Molecular Representation Learning

**论文链接:** [http://arxiv.org/abs/2510.23640v1](http://arxiv.org/abs/2510.23640v1)

**作者:** Zihao Jing, Yan Sun, Yan Yi Li, Sugitha Janarthanan, Alana Deng, Pingzhao Hu

**发布时间:** 2025-10-24

**备注:** Accepted by NeurIPS 2025

### GPT解析

### 总结

本研究提出了MuMo，一种结构化多模态融合框架，解决了分子表示中的3D构象不稳定性和模态崩溃问题，提高了模型的鲁棒性和泛化能力。

### 背景

多模态分子模型通常受到3D构象不可靠性和模态崩溃的限制，这影响了它们的鲁棒性和泛化能力。

### 目的

设计MuMo框架，解决分子表示中的3D构象不稳定性和模态崩溃问题，提高分子表示的质量。

### 方法

1) 设计结构化融合管道（SFP），将2D拓扑和3D几何结合成统一且稳定的结构先验；2) 引入渐进注入（PI）机制，非对称地将先验整合到序列流中，保留模态特定建模同时实现跨模态增强；3) 基于状态空间主干构建，支持长程依赖建模和鲁棒信息传播。

### 主要发现

在TDC和MoleculeNet的29个基准任务上，MuMo平均比最佳基线提高2.7%，在22个任务中排名第一，包括在LD50任务上提高27%，验证了模型对3D构象噪声的鲁棒性。

### 结论

MuMo框架通过结构化多模态融合有效解决了分子表示中的3D构象不稳定性和模态崩溃问题，在各种任务上表现出色。

### 翻译

多模态分子模型通常受到3D构象不可靠性和模态崩溃的限制，限制了它们的鲁棒性和泛化能力。我们提出了MuMo，一种结构化多模态融合框架，通过两个关键策略解决了分子表示中的这些挑战。为了减少构象依赖融合的不稳定性，我们设计了一个结构化融合管道（SFP），将2D拓扑和3D几何结合成统一且稳定的结构先验。为了缓解朴素融合引起的模态崩溃，我们引入了渐进注入（PI）机制，非对称地将此先验整合到序列流中，同时保留模态特定建模并实现跨模态增强。基于状态空间主干构建，MuMo支持长程依赖建模和鲁棒信息传播。在来自Therapeutics Data Commons（TDC）和MoleculeNet的29个基准任务上，MuMo在每个任务上平均比最佳基线提高2.7%，在22个任务中排名第一，包括在LD50任务上提高27%。这些结果验证了它对3D构象噪声的鲁棒性以及多模态融合在分子表示中的有效性。代码可在github.com/selmiss/MuMo获取。


### 论文摘要

Multimodal molecular models often suffer from 3D conformer unreliability and modality collapse, limiting their robustness and generalization. We propose MuMo, a structured multimodal fusion framework that addresses these challenges in molecular representation through two key strategies. To reduce the instability of conformer-dependent fusion, we design a Structured Fusion Pipeline (SFP) that combines 2D topology and 3D geometry into a unified and stable structural prior. To mitigate modality collapse caused by naive fusion, we introduce a Progressive Injection (PI) mechanism that asymmetrically integrates this prior into the sequence stream, preserving modality-specific modeling while enabling cross-modal enrichment. Built on a state space backbone, MuMo supports long-range dependency modeling and robust information propagation. Across 29 benchmark tasks from Therapeutics Data Commons (TDC) and MoleculeNet, MuMo achieves an average improvement of 2.7% over the best-performing baseline on each task, ranking first on 22 of them, including a 27% improvement on the LD50 task. These results validate its robustness to 3D conformer noise and the effectiveness of multimodal fusion in molecular representation. The code is available at: github.com/selmiss/MuMo.

---

## 46. Foundation Models in Dermatopathology: Skin Tissue Classification

**论文链接:** [http://arxiv.org/abs/2510.21664v1](http://arxiv.org/abs/2510.21664v1)

**作者:** Riya Gupta, Yiwei Zong, Dennis H. Murphree

**发布时间:** 2025-10-24

### GPT解析

### 总结

该研究比较了两种基础模型(UNI和Virchow2)在皮肤病理学全切片图像分类中的表现，发现Virchow2特征提取器表现略优，使用逻辑回归分类器可达到90%的准确率。研究还探索了数据增强和图像归一化技术，并证实平均聚合方法能有效生成切片级特征表示。

### 背景

皮肤病理学中全切片图像(WSI)的快速生成需要自动化方法进行高效处理和准确分类。

### 目的

评估两种基础模型(UNI和Virchow2)作为特征提取器，用于将WSI分类为三种诊断类别：黑素细胞性、基底样性和鳞状病变。

### 方法

使用平均聚合策略将块级嵌入聚合成切片级特征；训练多种机器学习分类器，包括逻辑回归、梯度提升树和随机森林模型；使用精确度、召回率、真正例率、假正例率和AUROC评估性能；探索数据增强技术和图像归一化；使用WandB.ai跟踪和可视化实验结果。

### 主要发现

使用Virchow2提取的块级特征在大多数切片级分类器上优于通过UNI提取的特征；使用Virchow2的逻辑回归模型达到了最高的准确率(90%)，但差异不具有统计学意义；平均聚合方法提供了可靠的切片级特征表示。

### 结论

这项研究强调了基础模型在自动化WSI分类中的潜力；为皮肤病理学诊断提供了一种可扩展且有效的方法；为切片级表示学习的未来进展铺平了道路。

### 翻译

皮肤病理学中全切片图像(WSI)的快速生成需要自动化方法进行高效处理和准确分类。本研究评估了两种基础模型UNI和Virchow2作为特征提取器的性能，用于将WSI分类为三种诊断类别：黑素细胞性、基底样性和鳞状病变。使用平均聚合策略将块级嵌入聚合成切片级特征，并随后用于训练多种机器学习分类器，包括逻辑回归、梯度提升树和随机森林模型。使用精确度、召回率、真正例率、假正例率和接收者操作特征曲线下面积(AUROC)在测试集上评估性能。结果表明，使用Virchow2提取的块级特征在大多数切片级分类器上优于通过UNI提取的特征，其中使用Virchow2的逻辑回归达到了最高的准确率(90%)，尽管差异不具有统计学意义。该研究还探索了数据增强技术和图像归一化，以提高模型的鲁棒性和泛化能力。平均聚合方法提供了可靠的切片级特征表示。所有实验结果和指标都使用WandB.ai进行跟踪和可视化，促进了可重复性和可解释性。这项研究强调了基础模型在自动化WSI分类中的潜力，为皮肤病理学诊断提供了一种可扩展且有效的方法，同时为切片级表示学习的未来进展铺平了道路。


### 论文摘要

The rapid generation of whole-slide images (WSIs) in dermatopathology necessitates automated methods for efficient processing and accurate classification. This study evaluates the performance of two foundation models, UNI and Virchow2, as feature extractors for classifying WSIs into three diagnostic categories: melanocytic, basaloid, and squamous lesions. Patch-level embeddings were aggregated into slide-level features using a mean-aggregation strategy and subsequently used to train multiple machine learning classifiers, including logistic regression, gradient-boosted trees, and random forest models. Performance was assessed using precision, recall, true positive rate, false positive rate, and the area under the receiver operating characteristic curve (AUROC) on the test set. Results demonstrate that patch-level features extracted using Virchow2 outperformed those extracted via UNI across most slide-level classifiers, with logistic regression achieving the highest accuracy (90%) for Virchow2, though the difference was not statistically significant. The study also explored data augmentation techniques and image normalization to enhance model robustness and generalizability. The mean-aggregation approach provided reliable slide-level feature representations. All experimental results and metrics were tracked and visualized using WandB.ai, facilitating reproducibility and interpretability. This research highlights the potential of foundation models for automated WSI classification, providing a scalable and effective approach for dermatopathological diagnosis while paving the way for future advancements in slide-level representation learning.

---

## 47. Causality Meets Locality: Provably Generalizable and Scalable Policy Learning for Networked Systems

**论文链接:** [http://arxiv.org/abs/2510.21427v1](http://arxiv.org/abs/2510.21427v1)

**作者:** Hao Liang, Shuqing Shi, Yudi Zhang, Biwei Huang, Yali Du

**发布时间:** 2025-10-24

**备注:** NeurIPS 2025 (Spotlight)

### GPT解析

### 总结

本文提出GSAC框架，结合因果表示学习和元actor-critic学习，解决了大规模网络系统（如交通、电力和无线网格）中强化学习的规模和环境变化挑战。

### 背景

大规模网络系统（如交通、电力和无线网格）对强化学习代理提出了规模和环境变化的挑战。

### 目的

提出一个能够同时实现可扩展性和领域泛化的框架，解决大规模网络系统中的强化学习挑战。

### 方法

提出GSAC（Generalizable and Scalable Actor-Critic）框架，每个代理学习稀疏局部因果掩码识别关键变量，生成状态和领域因素的紧凑表示（ACRs），元actor-critic训练跨多个源领域的共享策略，测试时通过少量轨迹估计新领域因素并部署自适应策略。

### 主要发现

建立了因果恢复、actor-critic收敛和自适应差距的有限样本保证，GSAC能够快速适应并显著优于传统方法。

### 结论

GSAC框架有效解决了大规模网络系统中的强化学习挑战，实现了可扩展性和领域泛化。

### 翻译

大规模网络系统，如交通、电力和无线网格，对强化学习代理提出了规模和环境变化的挑战。为应对这些挑战，我们提出了GSAC（Generalizable and Scalable Actor-Critic）框架，该框架结合因果表示学习和元actor-critic学习，以实现可扩展性和领域泛化。每个代理首先学习一个稀疏的局部因果掩码，可识别影响其动态的最小邻域变量，生成状态和领域因素的指数紧致近似紧凑表示（ACRs）。这些ACRs限制了将值函数截断到κ-跳邻域的误差，实现在图上的高效学习。元actor-critic则在多个源领域上训练共享策略，同时基于紧凑的领域因素进行条件化；在测试时，只需少量轨迹即可估计新的领域因素并部署自适应策略。我们建立了因果恢复、actor-critic收敛和自适应差距的有限样本保证，并表明GSAC能够快速适应且显著优于从头学习和传统自适应基线。


### 论文摘要

Large-scale networked systems, such as traffic, power, and wireless grids, challenge reinforcement-learning agents with both scale and environment shifts. To address these challenges, we propose GSAC (Generalizable and Scalable Actor-Critic), a framework that couples causal representation learning with meta actor-critic learning to achieve both scalability and domain generalization. Each agent first learns a sparse local causal mask that provably identifies the minimal neighborhood variables influencing its dynamics, yielding exponentially tight approximately compact representations (ACRs) of state and domain factors. These ACRs bound the error of truncating value functions to $\kappa$-hop neighborhoods, enabling efficient learning on graphs. A meta actor-critic then trains a shared policy across multiple source domains while conditioning on the compact domain factors; at test time, a few trajectories suffice to estimate the new domain factor and deploy the adapted policy. We establish finite-sample guarantees on causal recovery, actor-critic convergence, and adaptation gap, and show that GSAC adapts rapidly and significantly outperforms learning-from-scratch and conventional adaptation baselines.

---

## 48. Disentangled Representation Learning via Modular Compositional Bias

**论文链接:** [http://arxiv.org/abs/2510.21402v1](http://arxiv.org/abs/2510.21402v1)

**作者:** Whie Jung, Dong Hoon Lee, Seunghoon Hong

**发布时间:** 2025-10-24

### GPT解析

### 总结

该研究提出了一种组合偏置方法，用于解决解耦表示学习中的归纳偏置问题。通过随机混合潜在变量并根据特定规则重组，该方法能够在不修改架构或目标的情况下实现属性、对象甚至两者的解耦。

### 背景

当前的解耦表示学习方法主要依赖于特定策略的归纳偏置，包括为属性学习特定目标或为对象设计特定架构。这种依赖性在新的变化因素与先验假设不匹配或多个因素共存时会导致显著开销。

### 目的

提出一种组合偏置，一种与目标和架构都解耦的模块化归纳偏置，解决当多个因素共存时需要重新设计架构或目标的问题。

### 方法

根据不同因素在数据分布中的不同重组规则（全局属性互斥，对象共享公共支持），随机混合潜在变量。通过两个互补目标强制编码器发现混合策略反映的因素结构：(i)先验损失确保每个混合解码为真实图像；(ii)组合一致性损失将复合图像与其对应的复合潜在变量对齐。

### 主要发现

在通用框架下，只需调整混合策略即可实现属性、对象甚至两者的解耦，无需修改目标或架构。实验表明该方法在属性和对象解耦方面具有竞争性性能，且唯一实现了全局风格和对象的联合解耦。

### 结论

提出的组合偏置框架提供了一种灵活的方法来处理不同类型的解耦表示学习任务，只需调整混合策略而无需修改架构或目标。

### 翻译

最近的解耦表示学习方法严重依赖于特定因素策略-无论是为属性学习目标还是为对象设计架构-以嵌入归纳偏置。这种不同的方法在新的变化因素与先验假设（如统计独立性或空间排他性）不匹配或多个因素共存时会导致显著开销，因为从业者必须重新设计架构或目标。为此，我们提出了一种组合偏置，一种与目标和架构都解耦的模块化归纳偏置。我们的关键见解是，不同因素在数据分布中遵循不同的重组规则：全局属性是互斥的，例如一张脸只有一个鼻子，而对象共享公共支持（任何对象子集都可以共存）。因此，我们根据特定规则随机混合潜在变量，即混合策略，并通过两个互补目标强制编码器发现混合策略反映的任何因素结构：(i)确保每个混合解码为真实图像的先验损失；(ii)Wiedemer等人(arXiv:2310.05327)引入的组合一致性损失，它将每个复合图像与其对应的复合潜在变量对齐。在这一通用框架下，只需调整混合策略即可实现属性、对象甚至两者的解耦，而无需修改目标或架构。大量实验表明，我们的方法在属性和对象解耦方面都表现出竞争性性能，并且唯一实现了全局风格和对象的联合解耦。代码可在https://github.com/whieya/Compositional-DRL获取。


### 论文摘要

Recent disentangled representation learning (DRL) methods heavily rely on factor specific strategies-either learning objectives for attributes or model architectures for objects-to embed inductive biases. Such divergent approaches result in significant overhead when novel factors of variation do not align with prior assumptions, such as statistical independence or spatial exclusivity, or when multiple factors coexist, as practitioners must redesign architectures or objectives. To address this, we propose a compositional bias, a modular inductive bias decoupled from both objectives and architectures. Our key insight is that different factors obey distinct recombination rules in the data distribution: global attributes are mutually exclusive, e.g., a face has one nose, while objects share a common support (any subset of objects can co-exist). We therefore randomly remix latents according to factor-specific rules, i.e., a mixing strategy, and force the encoder to discover whichever factor structure the mixing strategy reflects through two complementary objectives: (i) a prior loss that ensures every remix decodes into a realistic image, and (ii) the compositional consistency loss introduced by Wiedemer et al. (arXiv:2310.05327), which aligns each composite image with its corresponding composite latent. Under this general framework, simply adjusting the mixing strategy enables disentanglement of attributes, objects, and even both, without modifying the objectives or architectures. Extensive experiments demonstrate that our method shows competitive performance in both attribute and object disentanglement, and uniquely achieves joint disentanglement of global style and objects. Code is available at https://github.com/whieya/Compositional-DRL.

---

## 49. Large Language Models Meet Text-Attributed Graphs: A Survey of Integration Frameworks and Applications

**论文链接:** [http://arxiv.org/abs/2510.21131v1](http://arxiv.org/abs/2510.21131v1)

**作者:** Guangxin Su, Hanchen Wang, Jianwei Wang, Wenjie Zhang, Ying Zhang, Jian Pei

**发布时间:** 2025-10-24

**备注:** Surveys and overviews; Natural language processing; Knowledge  representation and reasoning; Graph algorithms

### GPT解析

### 总结

这篇综述从编排的角度系统地回顾了大型语言模型和文本属性图的集成研究，展示了两者结合带来的互补优势。

### 背景

大型语言模型在自然语言处理方面取得了显著成功，但它们的黑盒性质限制了结构化和多跳推理能力；文本属性图提供了明确的文本关系结构，但往往缺乏语义深度。

### 目的

首次从编排的角度系统地回顾LLM和TAG的集成，总结现有方法，并指出未来在语言和图学习交叉领域的研究方向。

### 方法

引入新的分类法涵盖两个基本方向：LLM用于TAG（丰富基于图的任务）和TAG用于LLM（改进LLM推理）；将编排策略分为顺序、并行和多模块框架；讨论TAG特定的预训练、提示和参数高效微调的进展。

### 主要发现

结合LLMs和TAGs可以产生互补的好处：增强TAG表示学习和提高LLMs的推理能力和可解释性。

### 结论

总结了经验性见解，整理了可用数据集，强调了在推荐系统、生物医学分析和知识密集型问答等领域的应用，并指出了开放的挑战和有希望的研究方向。

### 翻译

大型语言模型通过强大的语义理解和生成在自然语言处理方面取得了显著成功。然而，它们的黑盒性质限制了结构化和多跳推理能力。相比之下，文本属性图提供了丰富的文本关系结构，但往往缺乏语义深度。最近的研究表明，结合LLMs和TAGs可以带来互补的好处：增强TAG表示学习并提高LLMs的推理能力和可解释性。这篇综述从编排的角度首次系统地回顾了LLM和TAG的集成。我们介绍了一种新的分类法，涵盖两个基本方向：LLM用于TAG，即LLMs丰富基于图的任务；以及TAG用于LLM，即结构化图改进LLM推理。我们将编排策略分为顺序、并行和多模块框架，并讨论了TAG特定的预训练、提示和参数高效微调的进展。除了方法论，我们还总结了经验性见解，整理了可用数据集，并强调了在推荐系统、生物医学分析和知识密集型问答等领域的多样化应用。最后，我们指出了开放的挑战和有希望的研究方向，旨在指导未来在语言和图学习交叉领域的工作。


### 论文摘要

Large Language Models (LLMs) have achieved remarkable success in natural language processing through strong semantic understanding and generation. However, their black-box nature limits structured and multi-hop reasoning. In contrast, Text-Attributed Graphs (TAGs) provide explicit relational structures enriched with textual context, yet often lack semantic depth. Recent research shows that combining LLMs and TAGs yields complementary benefits: enhancing TAG representation learning and improving the reasoning and interpretability of LLMs. This survey provides the first systematic review of LLM--TAG integration from an orchestration perspective. We introduce a novel taxonomy covering two fundamental directions: LLM for TAG, where LLMs enrich graph-based tasks, and TAG for LLM, where structured graphs improve LLM reasoning. We categorize orchestration strategies into sequential, parallel, and multi-module frameworks, and discuss advances in TAG-specific pretraining, prompting, and parameter-efficient fine-tuning. Beyond methodology, we summarize empirical insights, curate available datasets, and highlight diverse applications across recommendation systems, biomedical analysis, and knowledge-intensive question answering. Finally, we outline open challenges and promising research directions, aiming to guide future work at the intersection of language and graph learning.

---

## 50. Leveraging semantic similarity for experimentation with AI-generated treatments

**论文链接:** [http://arxiv.org/abs/2510.21119v1](http://arxiv.org/abs/2510.21119v1)

**作者:** Lei Shi, David Arbour, Raghavendra Addanki, Ritwik Sinha, Avi Feller

**发布时间:** 2025-10-24

**备注:** 31 pages, 5 figures

### GPT解析

### 总结

该研究提出了一种双核表示学习方法，用于处理大型语言模型生成的高维数字实验处理，并通过学习低维表示捕捉处理的基本结构，应用于指导生成模型产生有意义的处理变体和促进在线实验的自适应分配。

### 背景

大型语言模型（LLMs）使数字实验能够以新的形式进行，其中处理方式结合了人类和模型生成的内容，且方式日益复杂。在这种环境下，主要的方法论挑战是如何表示这些高维处理而不丢失其语义含义或使分析变得不可行。

### 目的

解决高维处理的表示问题，学习能够捕捉此类处理基本结构的低维表示，并应用于下游任务如指导生成模型和促进在线实验的自适应分配。

### 方法

提出双核表示学习方法，通过处理和用户协变量的基于核的表示的内积来建模因果效应。开发了一种交替最小化算法，能够从数据中高效学习这些表示，并在低秩因子模型下提供收敛保证。引入了一种在线实验的自适应设计策略作为该框架的应用。

### 主要发现

双核表示学习方法能够有效地从数据中学习处理的高维表示，并在低秩因子模型下保证收敛。该方法在在线实验的自适应设计中表现出有效性。

### 结论

通过学习低维表示来捕捉高维处理的基本结构，可以有效地解决大型语言模型数字实验中的表示挑战，并应用于多种下游任务。

### 翻译

大型语言模型（LLMs）使数字实验能够以新形式进行，其中处理方式结合了人类和模型生成的内容，且方式日益复杂。这种环境下的主要方法论挑战是如何表示这些高维处理而不丢失其语义含义或使分析变得不可行。在此，我们通过专注于学习能够捕捉此类处理基本结构的低维表示来解决这一问题。这些表示使下游应用成为可能，例如指导生成模型产生有意义的处理变体和促进在线实验中的自适应分配。我们提出了双核表示学习方法，通过处理和用户协变量的基于核的表示的内积来建模因果效应。我们开发了一种交替最小化算法，能够从数据中高效学习这些表示，并在低秩因子模型下提供收敛保证。作为该框架的应用，我们引入了一种在线实验的自适应设计策略，并通过数值实验证明了该方法的有效性。


### 论文摘要

Large Language Models (LLMs) enable a new form of digital experimentation where treatments combine human and model-generated content in increasingly sophisticated ways. The main methodological challenge in this setting is representing these high-dimensional treatments without losing their semantic meaning or rendering analysis intractable. Here, we address this problem by focusing on learning low-dimensional representations that capture the underlying structure of such treatments. These representations enable downstream applications such as guiding generative models to produce meaningful treatment variants and facilitating adaptive assignment in online experiments. We propose double kernel representation learning, which models the causal effect through the inner product of kernel-based representations of treatments and user covariates. We develop an alternating-minimization algorithm that learns these representations efficiently from data and provides convergence guarantees under a low-rank factor model. As an application of this framework, we introduce an adaptive design strategy for online experimentation and demonstrate the method's effectiveness through numerical experiments.

---

## 51. Fair Representation Learning with Controllable High Confidence Guarantees via Adversarial Inference

**论文链接:** [http://arxiv.org/abs/2510.21017v1](http://arxiv.org/abs/2510.21017v1)

**作者:** Yuhong Luo, Austin Hoag, Xintong Wang, Philip S. Thomas, Przemyslaw A. Grabowicz

**发布时间:** 2025-10-23

**备注:** Accepted by NeurIPS 2025

### GPT解析

### 总结

本研究提出了FRG框架，用于学习具有高置信度公平性保证的表示，确保在下游任务中人口统计差异被限制在用户定义的阈值内。

### 背景

表示学习被广泛应用于生成能在多个下游任务中泛化的表示，但确保其公平性对于防止对特定人口统计群体产生不公平至关重要。

### 目的

正式引入学习高置信度公平性表示的任务，保证每个下游预测中的人口统计差异被限制在用户定义的错误阈值内，并以可控的高概率实现。

### 方法

提出FRG(Fair Representation learning with high-confidence Guarantees)框架，通过利用优化的对抗模型提供高置信度公平性保证。

### 主要发现

在三个真实世界数据集上的实证评估表明，FRG相比六种最先进的公平表示学习方法，能够在一系列下游模型和任务中持续限制不公平性。

### 结论

FRG框架能有效提供高置信度的公平性保证，确保在多种下游任务中公平性得到控制。

### 翻译

表示学习越来越多地被应用于生成能够在多个下游任务中泛化的表示。确保表示学习中的公平性保证对于防止在下游任务中对特定人口统计群体产生不公平至关重要。在这项工作中，我们正式引入了学习能够实现高置信度公平性表示的任务。我们旨在保证每个下游预测中的人口统计差异被限制在用户定义的错误阈值内，并以可控的高概率实现。为此，我们提出了FRG框架，该框架通过利用优化的对抗模型提供这些高置信度公平性保证。我们在三个真实世界数据集上对FRG进行了实证评估，将其性能与六种最先进的公平表示学习方法进行比较。我们的结果表明FRG能够在一系列下游模型和任务中持续限制不公平性。


### 论文摘要

Representation learning is increasingly applied to generate representations that generalize well across multiple downstream tasks. Ensuring fairness guarantees in representation learning is crucial to prevent unfairness toward specific demographic groups in downstream tasks. In this work, we formally introduce the task of learning representations that achieve high-confidence fairness. We aim to guarantee that demographic disparity in every downstream prediction remains bounded by a *user-defined* error threshold $\epsilon$, with *controllable* high probability. To this end, we propose the ***F**air **R**epresentation learning with high-confidence **G**uarantees (FRG)* framework, which provides these high-confidence fairness guarantees by leveraging an optimized adversarial model. We empirically evaluate FRG on three real-world datasets, comparing its performance to six state-of-the-art fair representation learning methods. Our results demonstrate that FRG consistently bounds unfairness across a range of downstream models and tasks.

---

## 52. L^2M^3OF: A Large Language Multimodal Model for Metal-Organic Frameworks

**论文链接:** [http://arxiv.org/abs/2510.20976v1](http://arxiv.org/abs/2510.20976v1)

**作者:** Jiyu Cui, Fang Wu, Haokai Zhao, Minggao Feng, Xenophon Evangelopoulos, Andrew I. Cooper, Yejin Choi

**发布时间:** 2025-10-23

**备注:** 18 pages, 7 figures

### GPT解析

### 总结

该研究介绍了L2M3OF，首个专门用于金属有机框架设计的多模态大语言模型，通过整合晶体表示学习与语言理解能力，在材料设计领域取得了突破性进展。

### 背景

大语言模型在自然语言任务中展现出强大推理能力，但在科学发现方面的突破有限。理解复杂物理现象需要超越语言的多方面表示，MOFs设计面临巨大三维原子排列空间和严格网状规则的挑战。

### 目的

开发一种能够处理MOFs设计的多模态大语言模型，克服仅基于文本表示的局限性，减少对人类专家经验的依赖。

### 方法

L2M3OF整合晶体表示学习与语言理解，联合处理结构、文本和知识模态；使用预训练的晶体编码器和轻量级投影层压缩结构信息；构建晶体材料的结构-性质-知识数据库；与GPT-5、Gemini-2.5-Pro和DeepSeek-R1等闭源LLM进行基准测试。

### 主要发现

L2M3OF在性质预测和知识生成任务上优于领先的基于文本的闭源LLM，尽管使用的参数少得多。

### 结论

多模态方法对于多孔材料理解的重要性，L2M3OF成为材料发现领域下一代AI系统的基础。

### 翻译

大语言模型已在各种自然语言任务中展现出卓越的推理能力。然而，在科学发现方面的可比突破更为有限，因为理解复杂的物理现象需要超越语言本身的多方面表示。一个有力的例子是功能材料如金属有机框架的设计，这些材料对于碳捕获和氢储存等一系列重要应用至关重要。由于其可能存在的三维原子排列数量众多以及配位几何和拓扑的严格网状规则，在基于语言的、可被大语言模型解释的表示中导航其巨大而复杂的设计空间具有挑战性。尽管在更简单的材料系统中，大语言模型辅助的发现已显示出有希望的结果，但MOF设计仍然严重依赖于很少仅以文本信息编码的隐性人类专业知识。为了克服这一障碍，我们引入了L2M3OF，这是首个用于MOFs的多模态大语言模型。L2M3OF整合了晶体表示学习与语言理解，以联合处理结构、文本和知识模态。L2M3OF采用预训练的晶体编码器和轻量级投影层将结构信息压缩到标记空间，从而实现与语言指令的有效对齐。为了促进训练和评估，我们整理了一个包含晶体材料的结构-性质-知识数据库，并将L2M3OF与最先进的闭源大语言模型（如GPT-5、Gemini-2.5-Pro和DeepSeek-R1）进行基准测试。实验表明，尽管使用的参数少得多，L2M3OF在性质预测和知识生成任务上优于领先的基于文本的闭源大语言模型。这些结果突显了多模态方法对于多孔材料理解的重要性，并将L2M3OF确立为材料发现领域下一代AI系统的基础。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决如何利用大型语言模型有效理解和设计金属有机框架（MOFs）材料的问题。这个问题很重要，因为MOFs是一类多孔晶体材料，在碳捕获、氢储存、水收集和药物输送等领域有广泛应用潜力。然而，MOFs的三维复杂结构难以用纯文本表示，现有方法无法有效捕捉其三维对称性、周期性和长程结构相关性，限制了材料设计的效率和准确性。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先分析了现有LLMs在材料科学中的局限性，特别是它们难以处理MOFs的三维结构信息。他们借鉴了晶体表示学习（如PMTransformer）和指令微调范式，设计了L2M3OF模型。该方法结合了预训练的晶体编码器与轻量级投影层，将结构信息压缩到标记空间，实现与语言指令的高效对齐。同时，作者还采用了分组训练策略增强上下文多样性。这些设计基于对现有工作的深入分析，并针对MOFs的特殊性进行了创新改进。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是通过多模态学习，将MOFs的三维结构信息与文本知识相结合，使模型能够同时理解和处理材料的结构、属性和应用知识。整体流程包括：1) 构建MOF-SPK数据库，包含超过10万种MOFs的结构、属性和知识；2) 使用PMTransformer作为晶体结构编码器，将三维结构转换为潜在表示；3) 通过多模态投影桥将结构信息压缩并投影到语言模型空间；4) 使用指令微调范式训练模型，采用分组训练策略增强上下文多样性；5) 在属性预测、结构提取、描述生成和问答等任务上评估性能。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1) 首个专门为MOFs设计的多模态大型语言模型；2) 构建了首个包含MOFs结构、属性和领域知识的综合数据库MOF-SPK；3) 设计了高效的多模态对齐方法，使用轻量级投影层实现结构信息与语言指令的融合；4) 提出了分组训练策略增强模型性能。相比之前的工作，L2M3OF能够处理复杂的三维MOF结构，超越了纯文本表示的局限性；同时能够同时处理多种任务（属性预测、结构提取、描述生成和问答），在性能上超越了参数更多的商业LLMs，尽管使用了更少的参数。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'L2M3OF通过整合三维晶体结构表示与领域知识，开创了多模态大型语言模型在金属有机框架材料设计中的应用，实现了超越纯文本模型的性能，为材料科学发现提供了新的AI辅助工具。'}


### 论文摘要

Large language models have demonstrated remarkable reasoning capabilities across diverse natural language tasks. However, comparable breakthroughs in scientific discovery are more limited, because understanding complex physical phenomena demands multifaceted representations far beyond language alone. A compelling example is the design of functional materials such as MOFs-critical for a range of impactful applications like carbon capture and hydrogen storage. Navigating their vast and intricate design space in language-based representations interpretable by LLMs is challenging due to the numerous possible three-dimensional atomic arrangements and strict reticular rules of coordination geometry and topology. Despite promising early results in LLM-assisted discovery for simpler materials systems, MOF design remains heavily reliant on tacit human expertise rarely codified in textual information alone. To overcome this barrier, we introduce L2M3OF, the first multimodal LLM for MOFs. L2M3OF integrates crystal representation learning with language understanding to process structural, textual, and knowledge modalities jointly. L2M3OF employs a pre-trained crystal encoder with a lightweight projection layer to compress structural information into a token space, enabling efficient alignment with language instructions. To facilitate training and evaluation, we curate a structure-property-knowledge database of crystalline materials and benchmark L2M3OF against state-of-the-art closed-source LLMs such as GPT-5, Gemini-2.5-Pro and DeepSeek-R1. Experiments show that L2M3OF outperforms leading text-based closed-source LLMs in property prediction and knowledge generation tasks, despite using far fewer parameters. These results highlight the importance of multimodal approaches for porous material understanding and establish L2M3OF as a foundation for next-generation AI systems in materials discovery.

---

## 53. Irish-BLiMP: A Linguistic Benchmark for Evaluating Human and Language Model Performance in a Low-Resource Setting

**论文链接:** [http://arxiv.org/abs/2510.20957v1](http://arxiv.org/abs/2510.20957v1)

**作者:** Josh McGiff, Khanh-Tung Tran, William Mulcahy, Dáibhidh Ó Luinín, Jake Dalzell, Róisín Ní Bhroin, Adam Burke, Barry O'Sullivan, Hoang D. Nguyen, Nikola S. Nikolov

**发布时间:** 2025-10-23

**备注:** 8 pages

### GPT解析

### 总结

Irish-BLiMP是首个专门为评估爱尔兰语语言能力而设计的数据集和框架，研究对比了人类和大型语言模型在爱尔兰语语法知识上的表现。

### 背景

爱尔兰语是一种濒危语言，缺乏专门的语言能力评估工具和框架。

### 目的

评估现有大型语言模型和流利人类参与者在爱尔兰语语法知识方面的表现，并分析两者之间的差异。

### 方法

基于多种语言学文献和参考资料，由流利的爱尔兰语使用者团队手动构建和审查了1020个最小对比对，涵盖11个语言学特征的分类法。

### 主要发现

人类在所有语言学特征上的表现都优于所有模型，平均准确率高出16.6%；开源和闭源大型语言模型之间存在18.1%的性能差距；最强模型(gpt-5)准确率为73.5%，人类为90.1%；人类和模型在不同语法方面存在不同困难。

### 结论

Irish-BLiMP为评估LLMs在爱尔兰语中的语法能力提供了首个系统性框架，为低资源语言理解研究提供了有价值的基准。

### 翻译

我们提出了Irish-BLiMP(爱尔兰语语言能力最小对比数据集)，这是第一个专门为评估爱尔兰语(一种濒危语言)语言能力而设计和构建的数据集和框架。借鉴多种语言学文献和语法参考资料，我们由流利的爱尔兰语使用者团队手动构建并审查了涵盖11个语言学特征分类法的1020个最小对比对。我们评估了现有大型语言模型和流利人类参与者在爱尔兰语语法知识方面的表现。我们的发现显示，人类在所有语言学特征上的表现都优于所有模型，平均准确率高出16.6%。此外，开源和闭源大型语言模型之间存在18.1%的显著性能差距，即使是最强的模型也仅达到73.5%的准确率，而人类达到90.1%。有趣的是，人类参与者和模型在爱尔兰语语法的不同方面存在困难，这突显了模型学习表征的差异。总体而言，Irish-BLiMP为评估大型语言模型在爱尔兰语中的语法能力提供了首个系统性框架，并为推进低资源语言理解研究提供了有价值的基准。


### 论文摘要

We present Irish-BLiMP (Irish Benchmark of Linguistic Minimal Pairs), the first dataset and framework designed for fine-grained evaluation of linguistic competence in the Irish language, an endangered language. Drawing on a variety of linguistic literature and grammar reference works, we manually constructed and reviewed 1020 minimal pairs across a taxonomy of 11 linguistic features, through a team of fluent Irish speakers. We evaluate both existing Large Language Models (LLMs) and fluent human participants on their syntactic knowledge of Irish. Our findings show that humans outperform all models across all linguistic features, achieving 16.6% higher accuracy on average. Moreover, a substantial performance gap of 18.1% persists between open- and closed-source LLMs, with even the strongest model (gpt-5) reaching only 73.5% accuracy compared to 90.1% by human. Interestingly, human participants and models struggle on different aspects of Irish grammar, thus highlighting a difference in representation learned by the models. Overall, Irish-BLiMP provides the first systematic framework for evaluating the grammatical competence of LLMs in Irish and offers a valuable benchmark for advancing research on linguistic understanding in low-resource languages.

---

## 54. ROPES: Robotic Pose Estimation via Score-Based Causal Representation Learning

**论文链接:** [http://arxiv.org/abs/2510.20884v1](http://arxiv.org/abs/2510.20884v1)

**作者:** Pranamya Kulkarni, Puranjay Datta, Burak Varıcı, Emre Acartürk, Karthikeyan Shanmugam, Ali Tajer

**发布时间:** 2025-10-23

**备注:** A preliminary version of this paper appeared at NeurIPS 2025 Workshop  on Embodied World Models for Decision Making

### GPT解析

### 总结

本文提出ROPES方法，将因果表征学习应用于机器人姿态估计，这是一个无监督框架，能够解离潜在生成因素并识别可通过致动直接控制的变量。

### 背景

因果表征学习(CRL)是一种强大的无监督框架，能够解离高维数据下的潜在生成因素，并学习解离变量之间的因果关系。尽管在可识别性和实际应用方面取得了进展，但理论与实践之间仍存在显著差距。

### 目的

将CRL引入机器人领域，以缩小理论与实践之间的差距。具体解决机器人姿态估计问题，即从原始图像中恢复位置和方向。

### 方法

提出ROPES（基于分数的CRL的机器人姿态估计）框架。这是一个无监督框架，通过识别被致动的生成因素来体现干预性CRL的本质。图像由内在和外在潜在因素生成（如关节角度、手臂/肢体几何形状、光照、背景和相机配置），目标是解离和恢复可控制的潜在变量。

### 主要发现

在半合成机械臂实验中的经验评估表明，ROPES能够高保真度地解离潜在生成因素。这是仅通过利用分布变化实现的，而没有使用任何标记数据。论文还包括了与基于半监督框架的基线方法的比较。

### 结论

将机器人姿态定位为CRL的近实用测试平台。

### 翻译

因果表征学习（CRL）已成为一种强大的无监督框架，它（i）解离高维数据下的潜在生成因素，以及（ii）学习解离变量之间的因果关系。尽管最近在可识别性方面取得了广泛进展并有一些实践进展，但理论与实践之间仍然存在巨大差距。本文通过将CRL引入机器人领域（这一领域推动了CRL的发展）来缩小这一差距。具体而言，本文通过引入基于分数的CRL的机器人姿态估计（ROPES）来解决明确的机器人姿态估计问题——从原始图像中恢复位置和方向。作为一个无监督框架，ROPES通过识别被致动的生成因素体现了干预性CRL的本质：图像由内在和外在潜在因素（例如，关节角度、手臂/肢体几何形状、光照、背景和相机配置）生成，目标是解离和恢复可控制的潜在变量，即那些可以通过致动直接操作（干预）的变量。干预性CRL理论表明，通过干预经历变化的变量可以被识别。在机器人领域，这种干预通过命令各种关节的致动器并在不同控制下记录图像自然产生。在半合成机械臂实验中的经验评估表明，ROPES能够高保真度地解离潜在生成因素。关键的是，这是仅通过利用分布变化实现的，而没有使用任何标记数据。本文还包括了与最近提出的半监督框架的基线方法的比较。本文最后将机器人姿态定位为CRL的近实用测试平台。


### 论文摘要

Causal representation learning (CRL) has emerged as a powerful unsupervised framework that (i) disentangles the latent generative factors underlying high-dimensional data, and (ii) learns the cause-and-effect interactions among the disentangled variables. Despite extensive recent advances in identifiability and some practical progress, a substantial gap remains between theory and real-world practice. This paper takes a step toward closing that gap by bringing CRL to robotics, a domain that has motivated CRL. Specifically, this paper addresses the well-defined robot pose estimation -- the recovery of position and orientation from raw images -- by introducing Robotic Pose Estimation via Score-Based CRL (ROPES). Being an unsupervised framework, ROPES embodies the essence of interventional CRL by identifying those generative factors that are actuated: images are generated by intrinsic and extrinsic latent factors (e.g., joint angles, arm/limb geometry, lighting, background, and camera configuration) and the objective is to disentangle and recover the controllable latent variables, i.e., those that can be directly manipulated (intervened upon) through actuation. Interventional CRL theory shows that variables that undergo variations via interventions can be identified. In robotics, such interventions arise naturally by commanding actuators of various joints and recording images under varied controls. Empirical evaluations in semi-synthetic manipulator experiments demonstrate that ROPES successfully disentangles latent generative factors with high fidelity with respect to the ground truth. Crucially, this is achieved by leveraging only distributional changes, without using any labeled data. The paper also includes a comparison with a baseline based on a recently proposed semi-supervised framework. This paper concludes by positioning robot pose estimation as a near-practical testbed for CRL.

---

## 55. Machine-Learning-Assisted Comparison of Regression Functions

**论文链接:** [http://arxiv.org/abs/2510.24714v1](http://arxiv.org/abs/2510.24714v1)

**作者:** Jian Yan, Zhuoxi Li, Yang Ning, Yong Chen

**发布时间:** 2025-10-28

### GPT解析

### 总结

本文重新审视了比较回归函数的经典问题，提出了一种基于核的条件均值依赖性的广义概念，并开发了两种新的检验方法，利用现代机器学习方法进行灵活估计。

### 背景

比较回归函数是统计推断中的一个基本问题，与数据集成、迁移学习和因果推断等现代应用密切相关。现有方法通常依赖于平滑技术，因此受到维度诅咒的限制。

### 目的

提出一种新的方法来比较回归函数，克服现有方法在维度诅咒下的局限性，并减少对分布假设的依赖。

### 方法

提出基于核的条件均值依赖性的广义概念，为回归函数相等的零假设提供新表征，并基于此重新表述开发两种新的检验方法，利用现代机器学习方法进行灵活估计。

### 主要发现

建立了检验统计量的渐近性质，这些性质在固定维度和高维度情况下都成立；与需要严格分布假设的现有方法不同，该框架仅施加温和的矩条件。

### 结论

所提出的检验方法在广泛的数值研究中证明了其有效性，为比较回归函数提供了更灵活和实用的解决方案。

### 翻译

我们重新审视了比较回归函数的经典问题，这是统计推断中的一个基本问题，与数据集成、迁移学习和因果推断等现代应用密切相关。现有方法通常依赖于平滑技术，因此受到维度诅咒的限制。我们提出了一种基于核的条件均值依赖性的广义概念，为回归函数相等的零假设提供了新的表征。基于这一重新表述，我们开发了两种新的检验方法，利用现代机器学习方法进行灵活估计。我们建立了检验统计量的渐近性质，这些性质在固定维度和高维度情况下都成立。与通常需要严格分布假设的现有方法不同，我们的框架仅施加温和的矩条件。所提出检验方法的有效性通过大量的数值研究得到了证明。


### 论文摘要

We revisit the classical problem of comparing regression functions, a fundamental question in statistical inference with broad relevance to modern applications such as data integration, transfer learning, and causal inference. Existing approaches typically rely on smoothing techniques and are thus hindered by the curse of dimensionality. We propose a generalized notion of kernel-based conditional mean dependence that provides a new characterization of the null hypothesis of equal regression functions. Building on this reformulation, we develop two novel tests that leverage modern machine learning methods for flexible estimation. We establish the asymptotic properties of the test statistics, which hold under both fixed- and high-dimensional regimes. Unlike existing methods that often require restrictive distributional assumptions, our framework only imposes mild moment conditions. The efficacy of the proposed tests is demonstrated through extensive numerical studies.

---

## 56. Cluster Dose Prediction in Carbon Ion Therapy: Using Transfer Learning from a Pretrained Dose Prediction U-Net

**论文链接:** [http://arxiv.org/abs/2510.24703v1](http://arxiv.org/abs/2510.24703v1)

**作者:** Miriam Schwarze, Hui Khee Looe, Björn Poppe, Leo Thomas, Hans Rabus

**发布时间:** 2025-10-28

### GPT解析

### 总结

本研究使用神经网络预测簇剂量分布，以替代计算密集型模拟，并通过迁移学习技术优化了U-Net架构，实现了快速且准确的簇剂量估计。

### 背景

簇剂量概念为基于放射生物学效应(RBE)的模型提供了替代方案，用于描述辐射诱导的生物效应。

### 目的

研究应用神经网络预测簇剂量分布，以替代当前需要的计算密集型模拟。

### 方法

使用U-Net架构预测簇剂量分布，该网络首先在常规剂量分布上预训练，然后通过迁移学习技术对解码器路径进行适应；训练和预训练数据集包括来自多个患者头颈区域的不同能量和位置的碳离子束；使用蒙特卡洛模拟生成真实簇剂量分布作为基准。

### 主要发现

U-Net能够在使用图形处理单元(GPU)的情况下，在几毫秒内完成单笔束的簇剂量估计；预测的簇剂量分布与真实值的偏差小于0.35%。

### 结论

该原理验证研究证明了使用机器学习在临床可接受的计算时间内准确估计簇剂量的可行性；通过利用预训练神经网络和应用迁移学习技术，显著减少了对大规模、计算成本高昂的训练数据的需求。

### 翻译

簇剂量概念为基于放射生物学效应(RBE)的模型提供了替代方案，用于描述辐射诱导的生物效应。本研究探讨了应用神经网络预测簇剂量分布的可能性，旨在替代当前需要的计算密集型模拟。使用最初在常规剂量分布上预训练的U-Net来预测簇剂量分布，并通过迁移学习技术对解码器路径进行适应以用于簇剂量估计。训练和预训练数据集包括来自多个患者头颈区域的不同能量和位置的碳离子束。使用蒙特卡洛(MC)模拟生成真实簇剂量分布。U-Net能够在使用图形处理单元(GPU)的情况下，在几毫秒内完成单笔束的簇剂量估计。预测的簇剂量分布与真实值的偏差小于0.35%。这项原理验证研究证明了使用机器学习(ML)在临床可接受的计算时间内准确估计簇剂量的可行性。通过利用预训练神经网络和应用迁移学习技术，该方法显著减少了对大规模、计算成本高昂的训练数据的需求。


### 论文摘要

The cluster dose concept offers an alternative to the radiobiological effectiveness (RBE)-based model for describing radiation-induced biological effects. This study examines the application of a neural network to predict cluster dose distributions, with the goal of replacing the computationally intensive simulations currently required. Cluster dose distributions are predicted using a U-Net that was initially pretrained on conventional dose distributions. Using transfer learning techniques, the decoder path is adapted for cluster dose estimation. Both the training and pretraining datasets include head and neck regions from multiple patients and carbon ion beams of varying energies and positions. Monte Carlo (MC) simulations were used to generate the ground truth cluster dose distributions. The U-Net enables cluster dose estimation for a single pencil beam within milliseconds using a graphics processing unit (GPU). The predicted cluster dose distributions deviate from the ground truth by less than 0.35%. This proof-of-principle study demonstrates the feasibility of accurately estimating cluster doses within clinically acceptable computation times using machine learning (ML). By leveraging a pretrained neural network and applying transfer learning techniques, the approach significantly reduces the need for large-scale, computationally expensive training data.

---

## 57. Semi-supervised and unsupervised learning for health indicator extraction from guided waves in aerospace composite structures

**论文链接:** [http://arxiv.org/abs/2510.24614v1](http://arxiv.org/abs/2510.24614v1)

**作者:** James Josep Perry, Pablo Garcia-Conde Ortiz, George Konstantinou, Cornelie Vergouwen, Edlyn Santha Kumaran, Morteza Moradi

**发布时间:** 2025-10-28

### GPT解析

### 总结

该研究提出了一种综合数据驱动框架，通过结合两种学习方法与多域信号处理来提取航空航天复合材料结构的健康指标，解决了健康指标提取中的挑战。

### 背景

健康指标对于诊断和预测航空航天复合材料结构的状况至关重要，有助于高效维护和操作安全。然而，由于材料特性的变异性、损伤演变的随机性和多样化的损伤模式，提取可靠的健康指标具有挑战性。制造缺陷和服役期间的事故进一步增加了复杂性。

### 目的

开发一种综合数据驱动框架，通过两种学习方法与多域信号处理相结合来学习健康指标，由于缺乏真实健康指标，提出半监督和无监督方法来解决这一问题。

### 方法

提出两种学习方法：多样性深度半监督异常检测(Diversity-DeepSAD)方法，使用连续辅助标签作为假设损伤代理；退化趋势约束变分自编码器(DTC-VAE)，通过显式趋势约束嵌入单调性准则。使用多种激励频率的导波监测单加筋复合材料结构，探索时间、频率和时间频域表示，并通过无监督集成融合各频率健康指标。

### 主要发现

使用快速傅里叶变换特征，增强的Diversity-DeepSAD模型达到81.6%的性能，DTC-VAE提供最一致的健康指标，达到92.3%的性能，优于现有基线方法。

### 结论

所提出的数据驱动框架，特别是DTC-VAE方法，能够有效提取航空航天复合材料结构的健康指标，为结构健康监测提供了可靠解决方案。

### 翻译

健康指标对于诊断和预测航空航天复合材料结构的状况至关重要，能够实现高效维护和操作安全。然而，由于材料特性的变异性、损伤演变的随机性和多样化的损伤模式，提取可靠的健康指标仍然具有挑战性。制造缺陷（如脱粘）和服役期间的事故（如鸟撞）进一步使这一过程复杂化。本研究提出了一种综合数据驱动框架，通过结合两种学习方法与多域信号处理来学习健康指标。由于缺乏真实健康指标，提出了半监督和无监督方法：(i)多样性深度半监督异常检测方法，使用连续辅助标签作为假设损伤代理，克服了仅区分健康和故障状态的二元标签的局限性；(ii)退化趋势约束变分自编码器，其中单调性准则通过显式趋势约束嵌入。使用多种激励频率的导波来监测在疲劳载荷下的单加筋复合材料结构，并通过无监督集成融合各频率的健康指标，以减少频率依赖性和方差。


### 论文摘要

Health indicators (HIs) are central to diagnosing and prognosing the condition of aerospace composite structures, enabling efficient maintenance and operational safety. However, extracting reliable HIs remains challenging due to variability in material properties, stochastic damage evolution, and diverse damage modes. Manufacturing defects (e.g., disbonds) and in-service incidents (e.g., bird strikes) further complicate this process. This study presents a comprehensive data-driven framework that learns HIs via two learning approaches integrated with multi-domain signal processing. Because ground-truth HIs are unavailable, a semi-supervised and an unsupervised approach are proposed: (i) a diversity deep semi-supervised anomaly detection (Diversity-DeepSAD) approach augmented with continuous auxiliary labels used as hypothetical damage proxies, which overcomes the limitation of prior binary labels that only distinguish healthy and failed states while neglecting intermediate degradation, and (ii) a degradation-trend-constrained variational autoencoder (DTC-VAE), in which the monotonicity criterion is embedded via an explicit trend constraint. Guided waves with multiple excitation frequencies are used to monitor single-stiffener composite structures under fatigue loading. Time, frequency, and time-frequency representations are explored, and per-frequency HIs are fused via unsupervised ensemble learning to mitigate frequency dependence and reduce variance. Using fast Fourier transform features, the augmented Diversity-DeepSAD model achieved 81.6% performance, while DTC-VAE delivered the most consistent HIs with 92.3% performance, outperforming existing baselines.

---

## 58. Unsupervised learning for variability detection with Gaia DR3 photometry. The main sequence-white dwarf valley

**论文链接:** [http://arxiv.org/abs/2510.23776v1](http://arxiv.org/abs/2510.23776v1)

**作者:** P. Ranaivomanana, C. Johnston, G. Iorio, P. J. Groot, M. Uzundag, T. Kupfer, C. Aerts

**发布时间:** 2025-10-27

**备注:** Accepted for publication in Astronomy & Astrophysics (A&A); 10 pages,  9 figures, 1 appendix (7 additional figures, 2 tables)

### GPT解析

### 总结

本研究利用无监督学习方法从Gaia DR3数据中识别变星和特殊系统，成功发现了包括热亚矮星、激变变星、食双星等多种天体类型，并证实该方法在大规模恒星群体分析中的有效性。

### 背景

来自空间和地面望远镜的空前数量和质量的数据为机器学习提供了机会，使其能够识别传统方法可能忽略的新变星类别和特殊系统。之前已有相关方法学工作。

### 目的

研究无监督学习方法在大恒星群体（包括拥挤场中的天体）上的扩展潜力，无需预先选择的目录，专注于从Gaia DR3中选出的13405个源。

### 方法

使用基于从Gaia DR3时代测光中提取的统计特征的无监督聚类技术，采用t-SNE算法识别变星类别、子类型和仪器效应引起的虚假变异性。

### 主要发现

聚类结果显示了不同组别，包括热亚矮星、激变变星、食双星和仙女座场中的拥挤场天体；发现了潜在的恒星子类型；被标记为RR Lyrae的天体出现在CMD的意外区域，可能由于不可靠的天体测量或替代演化途径。

### 结论

所提出方法在寻找Gaia CMD大区域中可变天体（包括可变热亚矮星和激变变星）具有稳健性，展示了检测扩展恒星群体中变异性的效率，该无监督学习框架可扩展到大型数据集并在识别恒星子类方面有前景。

### 翻译

来自空间和地面望远镜的空前数量和质量的数据为机器学习提供了识别新类别变星和可能被传统方法忽视的特殊系统的机会。在先前方法学研究的基础上，本研究探讨了无监督学习方法在大恒星群体（包括拥挤场中的天体）上的扩展潜力，无需预先选择的目录，特别专注于从Gaia DR3中选出的13405个源，位于选定CMD区域。我们的方法主要基于从Gaia DR3时代测光中提取的统计特征，采用无监督聚类技术。我们使用t-SNE算法来识别变星类别、其子类型以及由仪器效应引起的虚假变异性。聚类结果显示了不同的组别，包括热亚矮星、激变变星、食双星以及拥挤场中的天体，如仙女座(M31)场中的天体。在这些集群中还出现了几种潜在的恒星子类型。值得注意的是，先前被标记为RR Lyrae的天体在CMD的意外区域被发现，可能是由于不可靠的天体测量（如双星性）或替代的演化途径。本研究强调了所提出方法在寻找Gaia CMD大区域中可变天体的稳健性，包括可变热亚矮星和激变变星，同时展示了其在检测扩展恒星群体中变异性的效率。所提出的无监督学习框架可扩展到大型数据集，并在识别恒星子类方面有前景的结果。


### 论文摘要

The unprecedented volume and quality of data from space- and ground-based telescopes present an opportunity for machine learning to identify new classes of variable stars and peculiar systems that may have been overlooked by traditional methods. Extending prior methodological work, this study investigates the potential of an unsupervised learning approach to scale effectively to larger stellar populations, including objects in crowded fields, and without the need for pre-selected catalogues, specifically focusing on 13 405 sources selected from Gaia DR3 and lying in the selected region of the CMD. Our methodology incorporates unsupervised clustering techniques based primarily on statistical features extracted from Gaia DR3 epoch photometry. We used the t-distributed stochastic neighbour embedding (t-SNE) algorithm to identify variability classes, their subtypes, and spurious variability induced by instrumental effects. The clustering results revealed distinct groups, including hot subdwarfs, cataclysmic variables (CVs), eclipsing binaries, and objects in crowded fields, such as those in the Andromeda (M31) field. Several potential stellar subtypes also emerged within these clusters. Notably, objects previously labelled as RR Lyrae were found in an unexpected region of the CMD, potentially due to either unreliable astrometric measurements (e.g., due to binarity) or alternative evolutionary pathways. This study emphasises the robustness of the proposed method in finding variable objects in a large region of the Gaia CMD, including variable hot subdwarfs and CVs, while demonstrating its efficiency in detecting variability in extended stellar populations. The proposed unsupervised learning framework demonstrates scalability to large datasets and yields promising results in identifying stellar subclasses.

---

## 59. Integrating Genomics into Multimodal EHR Foundation Models

**论文链接:** [http://arxiv.org/abs/2510.23639v1](http://arxiv.org/abs/2510.23639v1)

**作者:** Jonathan Amar, Edward Liu, Alessandra Breschi, Liangliang Zhang, Pouya Kheradpour, Sylvia Li, Lisa Soleymani Lehmann, Alessandro Giulianelli, Matt Edwards, Yugang Jia, David Nola, Raghav Mani, Pankaj Vats, Jesse Tetreault, T. J. Chen, Cory Y. McLean

**发布时间:** 2025-10-24

### GPT解析

### 总结

这篇论文介绍了一种创新的电子健康记录(EHR)基础模型，整合多基因风险评分(PRS)作为基础数据模态，超越传统EHR-only方法，构建更全面的健康档案。

### 背景

传统EHR模型仅使用临床数据，忽略了遗传因素对健康的影响。All of Us (AoU)研究项目提供了广泛而多样的数据资源。

### 目的

开发一个多模态框架，学习临床数据和遗传倾向之间的复杂关系，增强预测能力和可解释性。

### 方法

将生成式AI的进步扩展到EHR基础模型空间，利用AoU研究项目的数据进行训练，并探索迁移学习用于定制分类任务。

### 主要发现

在AoU数据上的评估表明，该模型对多种疾病发作(特别是2型糖尿病)具有预测价值，并展示了PRS和EHR数据之间的相互作用。

### 结论

这种方法对于解锁疾病预测、主动健康管理、风险分层和个性化治疗策略的新见解至关重要，为医疗保健中更个性化、公平和可行的真实世界证据生成奠定了基础。

### 翻译

本文介绍了一种创新的电子健康记录(EHR)基础模型，将多基因风险评分(PRS)作为基础数据模态整合其中，超越了传统的仅使用EHR的方法，以构建更全面的健康档案。利用All of Us (AoU)研究项目的广泛而多样的数据，这个多模态框架旨在学习临床数据和遗传倾向之间的复杂关系。该方法将生成式AI的进步扩展到EHR基础模型空间，增强了预测能力和可解释性。在AoU数据上的评估证明了该模型对多种疾病发作(特别是2型糖尿病)的预测价值，并说明了PRS和EHR数据之间的相互作用。该研究还探索了迁移学习用于定制分类任务，展示了架构的多功能性和效率。这种方法对于解锁疾病预测、主动健康管理、风险分层和个性化治疗策略的新见解至关重要，为医疗保健中更个性化、公平和可行的真实世界证据生成奠定了基础。


### 论文摘要

This paper introduces an innovative Electronic Health Record (EHR) foundation model that integrates Polygenic Risk Scores (PRS) as a foundational data modality, moving beyond traditional EHR-only approaches to build more holistic health profiles. Leveraging the extensive and diverse data from the All of Us (AoU) Research Program, this multimodal framework aims to learn complex relationships between clinical data and genetic predispositions. The methodology extends advancements in generative AI to the EHR foundation model space, enhancing predictive capabilities and interpretability. Evaluation on AoU data demonstrates the model's predictive value for the onset of various conditions, particularly Type 2 Diabetes (T2D), and illustrates the interplay between PRS and EHR data. The work also explores transfer learning for custom classification tasks, showcasing the architecture's versatility and efficiency. This approach is pivotal for unlocking new insights into disease prediction, proactive health management, risk stratification, and personalized treatment strategies, laying the groundwork for more personalized, equitable, and actionable real-world evidence generation in healthcare.

---

## 60. An unsupervised tour through the hidden pathways of deep neural networks

**论文链接:** [http://arxiv.org/abs/2510.21582v2](http://arxiv.org/abs/2510.21582v2)

**作者:** Diego Doimo

**发布时间:** 2025-10-24

**备注:** PhD thesis

### GPT解析

### 总结

该论文旨在提高对深度人工神经网络创建有意义表示并能泛化的内部机制的理解，专注于使用无监督学习工具表征隐藏表示的语义内容。

### 背景

深度神经网络创建有意义表示和泛化的内部机制尚不完全清楚，需要工具来表征隐藏表示的语义内容并利用数据的低维结构。

### 目的

改进对深度神经网络如何创建有意义表示并能泛化的内部机制的理解，开发无监督学习工具来表征隐藏表示的语义内容。

### 方法

开发无监督学习工具利用数据的低维结构；介绍Gride方法估计数据内在维度作为尺度的显式函数；研究深度神经网络隐藏层概率密度的演变；分析深度神经网络中的泛化问题。

### 主要发现

初始层产生单模态概率密度，消除与分类无关的结构；后续层中密度峰以分层方式出现，反映概念语义层次；输出层概率密度的峰地形可重建类别语义关系；宽神经网络学习冗余表示而非对虚假相关性过拟合；冗余神经元只在网络被正则化且训练误差为零时出现。

### 结论

深度神经网络通过分层方式构建语义层次结构；增加参数到插值训练数据的网络会改善泛化性能，与经典偏差-方差权衡相悖；宽神经网络学习冗余表示而非过拟合。

### 翻译

本论文的目标是提高我们对深度人工神经网络创建有意义表示并能泛化的内部机制的理解。我们专注于使用无监督学习工具表征隐藏表示的语义内容，这些工具由我们部分开发并在本论文中描述，它们能够利用数据的低维结构。第二章介绍了Gride，一种方法，允许将数据的内在维度估计为尺度的显式函数，而无需对数据集进行任何降采样。我们的方法基于严格的分布结果，能够量化估计的不确定性。此外，我们的方法简单且计算高效，因为它仅依赖于最近数据点之间的距离。在第三章中，我们研究了最先进深度神经网络中隐藏层概率密度的演变。我们发现初始层产生单模态概率密度，消除任何与分类无关的结构。在后续层中，密度峰以分层方式出现，反映了概念的语义层次结构。这个过程在输出层的概率密度中留下了痕迹，其中峰的地形允许重建类别的语义关系。在第四章中，我们研究了深度神经网络中的泛化问题：向插值训练数据的网络添加参数通常会改善其泛化性能，这与经典的偏差-方差权衡相悖。我们证明宽神经网络学习冗余表示而不是对虚假相关性过拟合，并且只有当网络被正则化且训练误差为零时，冗余神经元才会出现。


### 论文摘要

The goal of this thesis is to improve our understanding of the internal mechanisms by which deep artificial neural networks create meaningful representations and are able to generalize. We focus on the challenge of characterizing the semantic content of the hidden representations with unsupervised learning tools, partially developed by us and described in this thesis, which allow harnessing the low-dimensional structure of the data. Chapter 2. introduces Gride, a method that allows estimating the intrinsic dimension of the data as an explicit function of the scale without performing any decimation of the data set. Our approach is based on rigorous distributional results that enable the quantification of uncertainty of the estimates. Moreover, our method is simple and computationally efficient since it relies only on the distances among nearest data points. In Chapter 3, we study the evolution of the probability density across the hidden layers in some state-of-the-art deep neural networks. We find that the initial layers generate a unimodal probability density getting rid of any structure irrelevant to classification. In subsequent layers, density peaks arise in a hierarchical fashion that mirrors the semantic hierarchy of the concepts. This process leaves a footprint in the probability density of the output layer, where the topography of the peaks allows reconstructing the semantic relationships of the categories. In Chapter 4, we study the problem of generalization in deep neural networks: adding parameters to a network that interpolates its training data will typically improve its generalization performance, at odds with the classical bias-variance trade-off. We show that wide neural networks learn redundant representations instead of overfitting to spurious correlation and that redundant neurons appear only if the network is regularized and the training error is zero.

---

## 61. MFiSP: A Multimodal Fire Spread Prediction Framework

**论文链接:** [http://arxiv.org/abs/2510.23934v1](http://arxiv.org/abs/2510.23934v1)

**作者:** Alec Sathiyamoorthy, Wenhao Zhou, Xiangmin Zhou, Xiaodong Li, Iqbal Gondal

**发布时间:** 2025-10-27

### GPT解析

### 总结

该研究提出了一个多模态火灾蔓延预测框架(MFiSP)，整合社交媒体数据和遥感观测以提高预测准确性，评估结果显示该方法优于传统火灾预测方法。

### 背景

2019-2020年澳大利亚黑色夏季山火摧毁了1900万公顷土地，3000栋房屋，持续七个月，显示了野火威胁的规模和紧迫性，需要更好的预测来有效应对。

### 目的

开发一种更准确的火灾蔓延预测方法，以应对日益严重的野火威胁，提高应急响应效率。

### 方法

提出多模态火灾蔓延预测框架(MFiSP)，整合社交媒体数据和遥感观测，通过调整燃料图操纵策略动态调整火灾行为预测，使其与观察到的蔓延速率保持一致。

### 主要发现

整合多模态数据的MFiSP可以提高火灾蔓延预测的准确性，超越依赖消防行为分析师专业知识和静态输入的传统方法。

### 结论

新兴数据源如NASA的FIRMS卫星图像和自愿地理信息，结合多模态数据整合方法，能够显著改善火灾蔓延预测，为应对日益严重的野火威胁提供有效工具。

### 翻译

2019-2020年澳大利亚黑色夏季山火摧毁了1900万公顷土地，3000栋房屋，持续七个月，显示了野火威胁规模和紧迫性的升级，需要更好的预测以实现有效应对。传统火灾建模依赖于消防行为分析师(FBAns)的手动解读和静态环境数据，常常导致不准确和操作限制。新兴数据源，如NASA的FIRMS卫星图像和自愿地理信息，通过实现动态火灾蔓延预测，提供了改进的可能性。本研究提出了一个多模态火灾蔓延预测框架(MFiSP)，整合社交媒体数据和遥感观测以提高预测准确性。通过在同化周期之间调整燃料图操纵策略，该框架动态调整火灾行为预测，以与观察到的蔓延速率保持一致。我们使用在不同场景中合成的火灾事件多边形评估MFiSP的有效性，分析个体和组合对预测边界的影响。结果表明，整合多模态数据的MFiSP可以提高火灾蔓延预测，超越依赖FBAn专业知识和静态输入的传统方法。


### 论文摘要

The 2019-2020 Black Summer bushfires in Australia devastated 19 million hectares, destroyed 3,000 homes, and lasted seven months, demonstrating the escalating scale and urgency of wildfire threats requiring better forecasting for effective response. Traditional fire modeling relies on manual interpretation by Fire Behaviour Analysts (FBAns) and static environmental data, often leading to inaccuracies and operational limitations. Emerging data sources, such as NASA's FIRMS satellite imagery and Volunteered Geographic Information, offer potential improvements by enabling dynamic fire spread prediction. This study proposes a Multimodal Fire Spread Prediction Framework (MFiSP) that integrates social media data and remote sensing observations to enhance forecast accuracy. By adapting fuel map manipulation strategies between assimilation cycles, the framework dynamically adjusts fire behavior predictions to align with the observed rate of spread. We evaluate the efficacy of MFiSP using synthetically generated fire event polygons across multiple scenarios, analyzing individual and combined impacts on forecast perimeters. Results suggest that our MFiSP integrating multimodal data can improve fire spread prediction beyond conventional methods reliant on FBAn expertise and static inputs.

---

## 62. DPRF: A Generalizable Dynamic Persona Refinement Framework for Optimizing Behavior Alignment Between Personalized LLM Role-Playing Agents and Humans

**论文链接:** [http://arxiv.org/abs/2510.14205v2](http://arxiv.org/abs/2510.14205v2)

**作者:** Bingsheng Yao, Bo Sun, Yuanzhe Dong, Yuxuan Lu, Dakuo Wang

**发布时间:** 2025-10-16

**备注:** In Submission

### GPT解析

### 总结

动态人格完善框架（DPRF）通过迭代识别和解决认知差异，提高了LLM RPAs与目标个体行为的一致性

### 背景

大语言模型角色扮演代理（LLM RPAs）旨在模拟个人人类行为，但人格保真度常因手动创建的档案（如精心挑选的信息和人格特征）而受损，这些档案未经与目标个体一致性的验证

### 目的

解决上述限制，引入动态人格完善框架（DPRF），优化LLM RPAs行为与目标个体行为的一致性

### 方法

DPRF通过迭代识别认知差异来优化LLM RPAs与目标个体行为的对齐，这些认知差异可以通过自由形式或基于理论的、结构化的分析来识别生成行为与人类真实情况之间的差异，并完善人格档案以减轻这些差异

### 主要发现

在五个大语言模型和四种多样的行为预测场景上评估了DPRF，这些场景包括正式辩论、涉及心理健康问题的社交媒体帖子、公开访谈和电影评论，DPRF能够持续显著提高行为对齐度，优于基线人格，且能够跨模型和场景泛化

### 结论

提供了一种创建高保真人格档案的稳健方法，提高了下游应用的有效性，如用户模拟、社会研究和个性化AI

### 翻译

新兴的大语言模型角色扮演代理（LLM RPAs）旨在模拟个人人类行为，但人格保真度常因手动创建的档案（如精心挑选的信息和人格特征）而受损，这些档案未经与目标个体一致性的验证。为解决这一限制，我们的工作引入了动态人格完善框架（DPRF）。DPRF旨在通过迭代识别生成行为与人类真实情况之间的认知差异（无论是通过自由形式还是基于理论的、结构化的分析）来优化LLM RPAs行为与目标个体行为的一致性，并完善人格档案以减轻这些差异。我们在四个多样的行为预测场景中用五个大语言模型评估了DPRF：正式辩论、涉及心理健康问题的社交媒体帖子、公开访谈和电影评论。DPRF能够持续显著提高行为对齐度，优于基线人格，并且能够跨模型和场景泛化。我们的工作为创建高保真人格档案提供了一种稳健方法，并增强了下游应用的有效性，如用户模拟、社会研究和个性化AI。


### 论文摘要

The emerging large language model role-playing agents (LLM RPAs) aim to simulate individual human behaviors, but the persona fidelity is often undermined by manually-created profiles (e.g., cherry-picked information and personality characteristics) without validating the alignment with the target individuals. To address this limitation, our work introduces the Dynamic Persona Refinement Framework (DPRF).DPRF aims to optimize the alignment of LLM RPAs' behaviors with those of target individuals by iteratively identifying the cognitive divergence, either through free-form or theory-grounded, structured analysis, between generated behaviors and human ground truth, and refining the persona profile to mitigate these divergences.We evaluate DPRF with five LLMs on four diverse behavior-prediction scenarios: formal debates, social media posts with mental health issues, public interviews, and movie reviews.DPRF can consistently improve behavioral alignment considerably over baseline personas and generalizes across models and scenarios.Our work provides a robust methodology for creating high-fidelity persona profiles and enhancing the validity of downstream applications, such as user simulation, social studies, and personalized AI.

---

## 63. ComboBench: Can LLMs Manipulate Physical Devices to Play Virtual Reality Games?

**论文链接:** [http://arxiv.org/abs/2510.24706v1](http://arxiv.org/abs/2510.24706v1)

**作者:** Shuqing Li, Jiayi Yan, Chenyu Niu, Jen-tse Huang, Yun Peng, Wenxuan Wang, Yepang Liu, Michael R. Lyu

**发布时间:** 2025-10-28

### GPT解析

### 总结

本文引入了ComboBench基准测试，评估大型语言模型将语义动作转换为VR设备操作序列的能力，发现即使是顶级模型在程序推理和空间理解方面仍与人类存在差距。

### 背景

虚拟现实游戏需要玩家将高级语义动作转换为使用控制器和头戴显示器的精确设备操作，而人类基于常识和具身理解直观地执行这种转换，但大型语言模型是否能有效复制这种能力尚未得到充分探索。

### 目的

引入一个名为ComboBench的基准测试，评估大型语言模型将语义动作转换为VR设备操作序列的能力。

### 方法

从四个流行的VR游戏(《半衰期：爱莉克斯》、《Into the Radius》、《Moss：Book II》和《Vivecraft》)的262个场景中评估七个大型语言模型(GPT-3.5、GPT-4、GPT-4o、Gemini-1.5-Pro、LLaMA-3-8B、Mixtral-8x7B和GLM-4-Flash)，并与标注的真实基线和人类表现进行比较。

### 主要发现

表现最佳的模型(如Gemini-1.5-Pro)展示了强大的任务分解能力，但在程序推理和空间理解方面仍与人类存在差距；不同游戏之间的性能差异显著，表明对交互复杂性的敏感性；少样本示例显著提高了性能，表明有潜力针对性地增强大型语言模型的VR操作能力。

### 结论

大型语言模型在VR设备操作序列生成方面仍有改进空间，特别是在程序推理和空间理解方面。

### 翻译

虚拟现实游戏要求玩家使用控制器和头戴显示器将高级语义动作转换为精确的设备操作。虽然人类基于常识和具身理解直观地执行这种转换，但大型语言模型是否能有效复制这种能力尚未得到充分探索。本文引入了ComboBench基准测试，评估大型语言模型将语义动作转换为VR设备操作序列的能力，涵盖来自四个流行VR游戏(半衰期：爱莉克斯、Into the Radius、Moss：Book II和Vivecraft)的262个场景。我们评估了七个大型语言模型，包括GPT-3.5、GPT-4、GPT-4o、Gemini-1.5-Pro、LLaMA-3-8B、Mixtral-8x7B和GLM-4-Flash，并与标注的真实基线和人类表现进行比较。结果表明，尽管表现最佳的模型(如Gemini-1.5-Pro)展示了强大的任务分解能力，但在程序推理和空间理解方面仍与人类存在差距。不同游戏之间的性能差异显著，表明对交互复杂性的敏感性。少样本示例显著提高了性能，表明有潜力针对性地增强大型语言模型的VR操作能力。我们在https://sites.google.com/view/combobench发布了所有材料。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': "这篇论文研究大型语言模型（LLMs）是否能够有效地将高级语义动作（如'投降'、'驯服马匹'）转化为精确的VR设备操作序列（如'按X键'、'将头显朝向苦力怕'）。这个问题重要是因为VR游戏需要玩家将抽象意图转化为具体物理操作，这种能力是人类智能的关键组成部分，但目前尚不清楚LLMs是否具备这种具身认知能力，这对开发更智能的虚拟代理和游戏AI具有重要意义。", '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先通过认知科学专家访谈确定了VR交互所需的六种核心认知能力（任务分解、程序推理等），然后系统性地选择了四款代表性VR游戏，提取了262个语义动作场景，并由经验VR玩家进行精细的设备操作序列标注。他们借鉴了机器人系统（如SayCan）、虚拟环境智能体（如Voyager）和多步骤规划策略（如Chain-of-Thought）的工作，但专注于VR设备操作的精确映射，而非代码生成或机器人控制。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是通过构建全面基准测试ComboBench评估LLMs将语义动作转化为VR设备操作的能力，并采用多维度认知框架分析其表现。整体流程包括：1)构建基准：确定认知能力、选择游戏、提取场景、标注操作序列和认知能力；2)模型评估：在多个模型和少样本设置下测试；3)性能分析：比较LLMs与人类表现、分析认知能力差异、研究少样本影响和游戏复杂度关系。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)首个专门评估LLMs将语义动作转化为VR设备操作能力的基准测试；2)建立六种核心认知能力框架并实现步骤级别标注；3)设计多维度评估指标全面分析性能；4)收集四款不同类型VR游戏的262个场景提供多样化测试环境；5)通过与人类对比揭示LLMs在具身认知方面的优势和不足。相比之前工作，ComboBench专注于VR环境中的物理设备操作而非代码生成、机器人控制或平面界面操作。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '这篇论文提出了ComboBench基准测试，首次系统评估了大型语言模型将高级语义动作转化为VR设备操作的能力，揭示了当前LLMs在具身认知方面的优势与局限，为开发更智能的VR交互AI提供了重要指导。'}


### 论文摘要

Virtual Reality (VR) games require players to translate high-level semantic actions into precise device manipulations using controllers and head-mounted displays (HMDs). While humans intuitively perform this translation based on common sense and embodied understanding, whether Large Language Models (LLMs) can effectively replicate this ability remains underexplored. This paper introduces a benchmark, ComboBench, evaluating LLMs' capability to translate semantic actions into VR device manipulation sequences across 262 scenarios from four popular VR games: Half-Life: Alyx, Into the Radius, Moss: Book II, and Vivecraft. We evaluate seven LLMs, including GPT-3.5, GPT-4, GPT-4o, Gemini-1.5-Pro, LLaMA-3-8B, Mixtral-8x7B, and GLM-4-Flash, compared against annotated ground truth and human performance. Our results reveal that while top-performing models like Gemini-1.5-Pro demonstrate strong task decomposition capabilities, they still struggle with procedural reasoning and spatial understanding compared to humans. Performance varies significantly across games, suggesting sensitivity to interaction complexity. Few-shot examples substantially improve performance, indicating potential for targeted enhancement of LLMs' VR manipulation capabilities. We release all materials at https://sites.google.com/view/combobench.

---

## 64. Sound Source Localization for Spatial Mapping of Surgical Actions in Dynamic Scenes

**论文链接:** [http://arxiv.org/abs/2510.24332v1](http://arxiv.org/abs/2510.24332v1)

**作者:** Jonas Hein, Lazaros Vlachopoulos, Maurits Geert Laurent Olthof, Bastian Sigrist, Philipp Fürnstahl, Matthias Seibold

**发布时间:** 2025-10-28

### GPT解析

### 总结

这项研究提出了一种创新方法，通过整合3D声学信息和视觉数据，增强了手术场景的理解。该方法能够将声学事件在3D空间中定位并与视觉元素关联，实验证明在真实手术室环境中表现良好。

### 背景

手术场景理解对于推进计算机辅助和智能手术系统至关重要。当前方法主要依赖视觉数据或端到端学习，这限制了细粒度上下文建模。

### 目的

通过整合3D声学信息来增强手术场景表示，实现对手术环境在时间和空间上的多模态理解。

### 方法

提出了一种新颖的框架，用于生成手术场景的4D视听表示。通过将相控麦克风阵列的声学定位信息投影到RGB-D相机的动态点云上，并使用基于Transformer的声学事件检测模块识别包含工具-组织相互作用的相关时间片段。在专家执行的模拟手术程序期间，在真实的手术室设置中进行了实验评估。

### 主要发现

所提出的方法成功地将手术声学事件在3D空间中定位，并与视觉场景元素关联。实验评估证明了精确的空间声音定位和多模态数据的稳健融合，提供了手术活动的全面、动态表示。

### 结论

这项工作首次引入了动态手术场景中的空间声音定位方法，向多模态手术场景表示迈出了重要一步。通过整合声学和视觉数据，所提出的框架实现了更丰富的上下文理解，为未来的智能和自主手术系统奠定了基础。

### 翻译

目的：手术场景理解对于推进计算机辅助和智能手术系统至关重要。当前方法主要依赖视觉数据或端到端学习，这限制了细粒度上下文建模。这项工作旨在通过整合3D声学信息来增强手术场景表示，实现对手术环境在时间和空间上的多模态理解。方法：我们提出了一种新颖的框架，通过将相控麦克风阵列的声学定位信息投影到RGB-D相机的动态点云上，生成手术场景的4D视听表示。基于Transformer的声学事件检测模块识别包含工具-组织相互作用的相关时间片段，这些片段在视听场景表示中进行空间定位。系统在专家执行的模拟手术程序期间，在真实的手术室设置中进行了实验评估。结果：所提出的方法成功地将手术声学事件在3D空间中定位，并将它们与视觉场景元素关联。实验评估证明了精确的空间声音定位和多模态数据的稳健融合，提供了手术活动的全面、动态表示。结论：这项工作首次引入了动态手术场景中的空间声音定位方法，向多模态手术场景表示迈出了重要一步。通过整合声学和视觉数据，所提出的框架实现了更丰富的上下文理解，为未来的智能和自主手术系统奠定了基础。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决手术场景中的声学事件定位问题，目的是通过整合声学信息来增强手术场景的数字表示。这个问题很重要，因为当前手术场景理解主要依赖视觉数据，无法捕捉工具-组织相互作用的细粒度信息，而声学信息可以提供视觉无法获取的关键细节，对于开发智能手术系统和提高手术安全性与效率至关重要。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先分析了现有手术场景理解方法的局限性，然后提出多模态融合的思路，认为结合声学和视觉信息可以提供更全面的手术场景理解。他们借鉴了AudioSpectrogramTransformer模型进行声学事件检测，利用现有的声学波束形成技术生成2D声学热图，并参考了点云处理和3D定位技术。整个系统设计围绕如何有效融合声学和视觉信息，创建时空一致的4D手术场景表示。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是通过融合声学和视觉信息创建更全面的4D（3D空间+时间）手术场景表示，利用声学事件补充视觉信息，提供工具-组织相互作用的上下文。整体流程包括：1)使用相控麦克风阵列和RGB-D相机采集多模态数据；2)通过波束形成生成2D声学热图并投影到3D点云上；3)使用transformer模型检测手术声学事件；4)通过聚类算法定位声源并生成3D边界框；5)评估系统性能。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)首次在动态手术场景中进行空间声音定位；2)提出4D音频-视觉手术场景表示的新概念；3)基于transformer的声学事件检测方法；4)有效的多模态融合方法。相比之前工作，本文不仅整合了声学和视觉两种模态信息，还创建了时空一致的4D表示，专注于细粒度的声学事件检测和空间定位，而非高层概念预测，提供了更好的可解释性。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '这篇论文首次通过融合声学和视觉信息创建了4D动态手术场景表示，实现了手术声学事件的空间定位，为开发更智能、更全面的手术理解系统奠定了基础。'}


### 论文摘要

Purpose: Surgical scene understanding is key to advancing computer-aided and intelligent surgical systems. Current approaches predominantly rely on visual data or end-to-end learning, which limits fine-grained contextual modeling. This work aims to enhance surgical scene representations by integrating 3D acoustic information, enabling temporally and spatially aware multimodal understanding of surgical environments.   Methods: We propose a novel framework for generating 4D audio-visual representations of surgical scenes by projecting acoustic localization information from a phased microphone array onto dynamic point clouds from an RGB-D camera. A transformer-based acoustic event detection module identifies relevant temporal segments containing tool-tissue interactions which are spatially localized in the audio-visual scene representation. The system was experimentally evaluated in a realistic operating room setup during simulated surgical procedures performed by experts.   Results: The proposed method successfully localizes surgical acoustic events in 3D space and associates them with visual scene elements. Experimental evaluation demonstrates accurate spatial sound localization and robust fusion of multimodal data, providing a comprehensive, dynamic representation of surgical activity.   Conclusion: This work introduces the first approach for spatial sound localization in dynamic surgical scenes, marking a significant advancement toward multimodal surgical scene representations. By integrating acoustic and visual data, the proposed framework enables richer contextual understanding and provides a foundation for future intelligent and autonomous surgical systems.

---

## 65. Enhancing Vision-Language Models for Autonomous Driving through Task-Specific Prompting and Spatial Reasoning

**论文链接:** [http://arxiv.org/abs/2510.24152v1](http://arxiv.org/abs/2510.24152v1)

**作者:** Aodi Wu, Xubo Luo

**发布时间:** 2025-10-28

**备注:** RoboSense Challenge with IROS 2025

### GPT解析

### 总结

该研究提出了一个系统性框架，用于提高视觉语言模型在自动驾驶场景理解任务中的性能，通过四个核心组件实现问题分类、任务特定提示设计、视觉信息组装和模型参数优化。

### 背景

IROS 2025 RoboSense Challenge评估视觉语言模型在自动驾驶场景理解方面的能力，涵盖感知、预测、规划和损坏检测四个任务领域。

### 目的

开发一个有效的框架，提升视觉语言模型在安全关键型自动驾驶任务中的表现，特别是在处理干净数据和损坏数据时的准确率。

### 方法

构建了一个四组件框架：1)混合提示路由器分类并分派问题；2)特定任务提示嵌入坐标系、空间推理规则等；3)视觉组装模块组合多视图图像；4)按任务配置模型推理参数。

### 主要发现

在Qwen2.5-VL-72B模型上实现，该方法在第一阶段(干净数据)达到70.87%平均准确率，在第二阶段(损坏数据)达到72.85%准确率。

### 结论

结构化提示和空间接地能显著提升视觉语言模型在安全关键型自动驾驶任务中的性能。

### 翻译

本技术报告介绍了我们在IROS 2025 RoboSense Challenge上的解决方案，该方案评估视觉语言模型在自动驾驶场景理解方面的能力，涵盖感知、预测、规划和损坏检测任务。我们提出了一个基于四个核心组件构建的系统性框架。首先，混合提示路由器对问题进行分类并将其分派给特定任务的专家提示，消除了不同问题类型之间的干扰。其次，特定任务提示嵌入明确的坐标系、空间推理规则、角色扮演、思维链/思维树推理以及为每个任务定制的小样本示例。第三，视觉组装模块根据问题要求组合多视图图像、对象裁剪、洋红色标记和自适应历史帧。第四，我们按任务配置模型推理参数(温度、top-p、消息角色)以优化输出质量。在Qwen2.5-VL-72B上实现，我们的方法在第一阶段(干净数据)上平均准确率达到70.87%，在第二阶段(损坏数据)上达到72.85%，证明结构化提示和空间接地显著提高了VLM在安全关键型自动驾驶任务上的性能。代码和提示可在https://github.com/wuaodi/UCAS-CSU-phase2获取。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决视觉-语言模型(VLMs)在自动驾驶场景理解中面临的三个关键挑战：多视角场景中的空间推理困难(如混淆左右方向、错误判断物体位置)、不同任务类型之间的提示干扰(单一通用提示难以同时优化感知、预测、规划等不同任务)、以及时间上下文集成不当(添加历史帧可能引入噪声而非有用信息)。这些问题在现实中非常重要，因为自动驾驶系统需要准确的场景理解才能做出安全决策，解决这些问题能显著提高VLMs在安全关键自动驾驶任务中的可靠性和性能。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者通过三个关键设计原则来解决问题：消除提示干扰(使用专家提示而非通用提示)、增强空间推理(明确定义坐标系统和约束)、以及自适应时间上下文(根据问题类型选择适当历史帧)。作者借鉴了多项现有工作，包括Mixture-of-Prompts(使用多个专家提示)、Role-playing(为模型分配特定角色)、Chain-of-Thought/Tree-of-Thought推理(逐步推理和探索多种可能性)、In-context learning(通过示例引导模型)以及Visual prompting(视觉注意力引导)。作者将这些现有技术组合并调整，以适应自动驾驶场景的特殊需求，特别是空间推理和时间上下文处理。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是通过系统化的提示工程增强VLM在自动驾驶场景理解中的性能，特别关注空间推理和任务特定提示设计。整体实现流程包含四个核心组件：1)路由器：分类测试查询并分配到适当的任务专家提示；2)任务特定提示：包含坐标系统、空间规则、角色扮演、链式/树式思维推理和少样本示例；3)视觉组装模块：根据问题需求组合多视角图像、物体裁剪、标记和自适应历史帧；4)模型选择和推理参数：使用Qwen2.5-VL-72B并根据任务类型调整推理参数。流程是：路由器分类问题→选择任务特定提示→组合视觉输入→使用特定推理参数调用VLM生成答案。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)Mixture-of-Prompts路由器，消除不同任务类型间的提示干扰；2)明确的坐标系统和空间规则，增强多视角空间定位能力；3)自适应视觉组装，根据问题类型组合视觉输入；4)任务特定的推理参数，优化不同任务类型的输出质量；5)结合多种推理技术，为不同任务定制推理策略。相比之前工作，本文的主要不同在于不是通过微调模型来增强性能，而是通过提示级别的设计(明确空间定位、自适应时间证据和任务特定路由)来提高可靠性，这种方法在保持基础模型不变的情况下显著提升了性能。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '这篇论文提出了一种基于任务特定提示和空间推理的系统化框架，显著提升了视觉-语言模型在自动驾驶场景理解任务中的性能，特别是在多视角空间定位和不同任务类型处理方面。'}


### 论文摘要

This technical report presents our solution for the RoboSense Challenge at IROS 2025, which evaluates Vision-Language Models (VLMs) on autonomous driving scene understanding across perception, prediction, planning, and corruption detection tasks. We propose a systematic framework built on four core components. First, a Mixture-of-Prompts router classifies questions and dispatches them to task-specific expert prompts, eliminating interference across diverse question types. Second, task-specific prompts embed explicit coordinate systems, spatial reasoning rules, role-playing, Chain-of-Thought/Tree-of-Thought reasoning, and few-shot examples tailored to each task. Third, a visual assembly module composes multi-view images with object crops, magenta markers, and adaptive historical frames based on question requirements. Fourth, we configure model inference parameters (temperature, top-p, message roles) per task to optimize output quality. Implemented on Qwen2.5-VL-72B, our approach achieves 70.87% average accuracy on Phase-1 (clean data) and 72.85% on Phase-2 (corrupted data), demonstrating that structured prompting and spatial grounding substantially enhance VLM performance on safety-critical autonomous driving tasks. Code and prompt are available at https://github.com/wuaodi/UCAS-CSU-phase2.

---

## 66. Concerto: Joint 2D-3D Self-Supervised Learning Emerges Spatial Representations

**论文链接:** [http://arxiv.org/abs/2510.23607v1](http://arxiv.org/abs/2510.23607v1)

**作者:** Yujia Zhang, Xiaoyang Wu, Yixing Lao, Chengyao Wang, Zhuotao Tian, Naiyan Wang, Hengshuang Zhao

**发布时间:** 2025-10-27

**备注:** NeurIPS 2025, produced by Pointcept, project page:  https://pointcept.github.io/Concerto

### GPT解析

### 总结

Concerto是一个模拟人类概念学习的模型，通过结合3D模态内自蒸馏和2D-3D跨模态联合嵌入，学习空间认知中的抽象概念，表现出优越的性能。

### 背景

人类通过多感官协同学习抽象概念，一旦形成，可以从单一感官回忆这些表示。

### 目的

受人类学习原理启发，开发一个用于空间认知的概念学习模型。

### 方法

Concerto结合了3D模态内自蒸馏和2D-3D跨模态联合嵌入的方法。

### 主要发现

Concerto学习到更连贯和信息丰富的空间特征；在零样本可视化中表现出色；在线性探测中分别比最先进的2D和3D自监督模型高出14.2%和4.8%；完全微调后在多个场景理解基准测试中设置了新的最先进结果；提出了针对视频提升点云空间理解的变体；开发了将表示投影到CLIP语言空间的翻译器，实现开放世界感知。

### 结论

Concerto产生了具有优越细粒度几何和语义一致性的空间表示。

### 翻译

人类通过多感官协同学习抽象概念，一旦形成，这样的表示通常可以从单一感官回忆。受这一原理启发，我们引入了Concerto，这是一个用于空间认知的人类概念学习的极简模拟，结合了3D模态内自蒸馏和2D-3D跨模态联合嵌入。尽管简单，但Concerto学习到更连贯和信息丰富的空间特征，如零样本可视化所示。在3D场景感知的线性探测中，它分别比独立的最新2D和3D自监督模型高出14.2%和4.8%，也优于它们的特征连接。通过完全微调，Concerto在多个场景理解基准测试中设置了新的最新结果（例如在ScanNet上达到80.7% mIoU）。我们进一步提出了一个针对视频提升点云空间理解的Concerto变体，以及一个将Concerto表示线性投影到CLIP语言空间的翻译器，实现开放世界感知。这些结果表明，Concerto产生了具有优越细粒度几何和语义一致性的空间表示。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决如何通过联合2D图像和3D点云的自监督学习，学习更丰富、更一致的空间表示问题。这个问题在现实中很重要，因为空间认知是自动驾驶、混合现实和机器人等应用的基础，而多模态学习可以提供更全面的空间理解，减少对标注数据的依赖，使模型能够从大量无标签数据中学习更强大的表示。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者受人类多感官协同学习抽象过程的启发，认识到人类可以通过不同感官（如视觉和触觉）形成统一概念，并能从单一感官唤起完整体验。他们首先进行了初步研究，验证了简单拼接2D和3D特征优于单一模态，进而设计了更复杂的框架。该方法借鉴了Sonata框架用于3D点云表示学习的单模态自蒸馏技术，以及基于LeCun的联合嵌入预测架构(JEPA)的跨模态对齐方法，将两者结合形成Concerto框架。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是通过模拟人类多感官协同学习的方式，结合2D图像和3D点云的自监督学习，学习更丰富、更一致的空间表示，使得模型能够从单一模态中唤出完整的空间概念。整体实现流程包括：1) 单模态自蒸馏：使用Point Transformer V3作为点云编码器，通过教师-学生范式训练，使用在线聚类目标函数增强一致性；2) 跨模态联合嵌入预测：使用预训练图像编码器提取特征，建立点云点和图像像素对应关系，预测点云特征以匹配图像特征，使用余弦相似度作为损失；3) 协同训练：结合两个目标函数，适当平衡权重，训练出能从单一模态唤出丰富空间表示的模型。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1) 多模态协同学习框架，通过跨模态预测而非简单特征融合学习统一表示；2) 自监督点云Transformer，结合单模态自蒸馏和跨模态联合嵌入；3) 视频感知变体，利用前馈重建从视频中生成点云数据；4) 语言桥接，将表示映射到CLIP语言空间实现开放词汇感知。相比之前的工作，Concerto不仅整合了2D图像和3D点云信息，还通过联合学习产生了比单一模态或简单特征拼接更丰富、更一致的表示空间，在多个场景理解任务上取得了最先进的性能。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'Concerto通过模拟人类多感官协同学习的方式，联合2D图像和3D点云的自监督学习，学习到了比单一模态更丰富、更一致的空间表示，并在多个场景理解任务上取得了最先进的性能。'}


### 论文摘要

Humans learn abstract concepts through multisensory synergy, and once formed, such representations can often be recalled from a single modality. Inspired by this principle, we introduce Concerto, a minimalist simulation of human concept learning for spatial cognition, combining 3D intra-modal self-distillation with 2D-3D cross-modal joint embedding. Despite its simplicity, Concerto learns more coherent and informative spatial features, as demonstrated by zero-shot visualizations. It outperforms both standalone SOTA 2D and 3D self-supervised models by 14.2% and 4.8%, respectively, as well as their feature concatenation, in linear probing for 3D scene perception. With full fine-tuning, Concerto sets new SOTA results across multiple scene understanding benchmarks (e.g., 80.7% mIoU on ScanNet). We further present a variant of Concerto tailored for video-lifted point cloud spatial understanding, and a translator that linearly projects Concerto representations into CLIP's language space, enabling open-world perception. These results highlight that Concerto emerges spatial representations with superior fine-grained geometric and semantic consistency.

---

## 67. Localising under the drape: proprioception in the era of distributed surgical robotic system

**论文链接:** [http://arxiv.org/abs/2510.23512v1](http://arxiv.org/abs/2510.23512v1)

**作者:** Martin Huber, Nicola A. Cavalcanti, Ayoob Davoodi, Ruixuan Li, Christopher E. Mower, Fabio Carrillo, Christoph J. Laux, Francois Teyssere, Thibault Chandanson, Antoine Harlé, Elie Saghbiny, Mazda Farshad, Guillaume Morel, Emmanuel Vander Poorten, Philipp Fürnstahl, Sébastien Ourselin, Christos Bergeles, Tom Vercauteren

**发布时间:** 2025-10-27

### GPT解析

### 总结

本研究提出了一种无需标记的手术机器人本体感觉方法，通过轻量级立体RGB摄像头和基于Transformer的深度学习模型，实现了在无菌遮挡情况下的精确定位，提高了手术场景的可见性和追踪能力。

### 背景

手术机器人虽然机械精密，但对周围环境缺乏感知能力，导致碰撞、系统恢复和工作流程中断等问题。现有的追踪系统依赖笨重的红外摄像头和反射标记，只能提供有限视角并增加手术室硬件负担。

### 目的

开发一种无需标记的本体感觉方法，使手术机器人在无菌遮挡情况下能够精确定位，提高手术场景的可见性和追踪能力，减少硬件负担，提高手术安全性。

### 方法

使用轻量级立体RGB摄像头和基于Transformer的新型深度学习模型。基于最大的多中心空间机器人手术数据集（140万张来自人体尸体和临床前体内研究的自注释图像），通过跟踪整个机器人和手术场景而非单个标记来实现定位。

### 主要发现

该方法提供对遮挡具有鲁棒性的整体视图，支持手术场景理解和上下文感知控制。在体内呼吸补偿中展示了临床应用潜力，可获取组织动力学；在多机器人系统中实现精确定位。与现有系统相比，消除了标记并将追踪可见性提高了25%。

### 结论

这是首次展示完全覆盖的手术机器人的无标记本体感觉，降低了设置复杂性，提高了安全性，并为模块化和自主机器人手术铺平了道路。

### 翻译

尽管手术机器人具有机械精密性，但它们仍然无法感知周围环境。这种空间意识的缺乏导致碰撞、系统恢复和工作流程中断等问题，随着具有独立交互臂的分布式机器人的引入，这些问题将加剧。现有的追踪系统依赖笨重的红外摄像头和反射标记，仅提供手术场景的有限视角，并在拥挤的手术室中增加硬件负担。我们提出了一种无需标记的本体感觉方法，使手术机器人在无菌遮挡的情况下能够精确定位，尽管视觉线索被遮挡。我们的方法仅依靠轻量级立体RGB摄像头和基于Transformer的新型深度学习模型。它基于迄今为止最大的多中心空间机器人手术数据集（来自人体尸体和临床前体内研究的140万张自注释图像）。通过跟踪整个机器人和手术场景，而不是单个标记，我们的方法提供了对遮挡具有鲁棒性的整体视图，支持手术场景理解和上下文感知控制。我们展示了体内呼吸补偿的潜在临床应用示例，可以获取最先进追踪技术无法观察到的组织动力学，并在多机器人系统中精确定位以支持未来的智能交互。此外，与现有系统相比，我们的方法消除了标记并将追踪可见性提高了25%。据我们所知，这是首次展示完全覆盖的手术机器人的无标记本体感觉，降低了设置复杂性，提高了安全性，并为模块化和自主机器人手术铺平了道路。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决手术机器人在被无菌布覆盖后无法精确定位的问题。这个问题很重要，因为当前手术机器人缺乏环境感知能力，会导致碰撞、系统恢复和工作流程中断，随着分布式多臂机器人系统的普及，这些问题会更加严重。现有的红外跟踪系统笨重、容易被遮挡、需要严格校准，且难以扩展到多机器人环境。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': "作者首先分析了现有红外跟踪系统的局限性，然后设计了基于轻量级立体RGB相机和Transformer深度学习模型的解决方案。他们借鉴了工业机器人中的无标记定位方法，但进行了修改以适应外科环境中的无菌布遮挡问题。方法核心是立体可微分渲染技术，结合了粒子群优化算法进行初始估计，并通过'上下文先验'方法迭代改进分割结果。作者还构建了140万张图像的大型多中心数据集来训练模型。", '这个方法的核心思想是什么？整体实现流程是怎样的？': "核心思想是使用轻量级立体RGB相机和深度学习模型，通过立体可微分渲染技术实现对被覆盖手术机器人的精确定位，无需传统反射标记。整体流程包括：1)收集和预处理多中心数据集；2)训练能够处理遮挡的分割模型；3)使用相机群优化进行初始估计；4)应用立体可微分渲染优化姿态估计；5)使用'上下文先验'方法迭代改进结果；6)在临床前和临床环境中验证方法。", '论文的关键创新点有哪些？相比之前的工作，有什么不同？': "关键创新点包括：1)首次实现完全覆盖的无标记手术机器人定位；2)开发立体可微分渲染技术处理遮挡；3)构建最大规模的多中心手术机器人数据集；4)开发遮挡不变的分割方法；5)提出'上下文先验'方法改进分割；6)支持多机器人设置。相比之前工作，本文方法无需标记、硬件更轻量(轻13倍、体积小3倍)、提供更完整的场景理解、能捕获传统系统不可见的组织动态，并在真实临床条件下验证了亚毫米级定位精度。", '如果要用一句话总结这篇论文的贡献，你会怎么说？': '本文提出了一种基于轻量级立体RGB相机和深度学习的无标记定位方法，实现了对被无菌布覆盖的手术机器人的精确定位，提高了手术场景可见性并揭示了传统系统无法观察到的组织动态，为模块化和自主机器人手术铺平了道路。'}


### 论文摘要

Despite their mechanical sophistication, surgical robots remain blind to their surroundings. This lack of spatial awareness causes collisions, system recoveries, and workflow disruptions, issues that will intensify with the introduction of distributed robots with independent interacting arms. Existing tracking systems rely on bulky infrared cameras and reflective markers, providing only limited views of the surgical scene and adding hardware burden in crowded operating rooms. We present a marker-free proprioception method that enables precise localisation of surgical robots under their sterile draping despite associated obstruction of visual cues. Our method solely relies on lightweight stereo-RGB cameras and novel transformer-based deep learning models. It builds on the largest multi-centre spatial robotic surgery dataset to date (1.4M self-annotated images from human cadaveric and preclinical in vivo studies). By tracking the entire robot and surgical scene, rather than individual markers, our approach provides a holistic view robust to occlusions, supporting surgical scene understanding and context-aware control. We demonstrate an example of potential clinical benefits during in vivo breathing compensation with access to tissue dynamics, unobservable under state of the art tracking, and accurately locate in multi-robot systems for future intelligent interaction. In addition, and compared with existing systems, our method eliminates markers and improves tracking visibility by 25%. To our knowledge, this is the first demonstration of marker-free proprioception for fully draped surgical robots, reducing setup complexity, enhancing safety, and paving the way toward modular and autonomous robotic surgery.

---

## 68. UrbanIng-V2X: A Large-Scale Multi-Vehicle, Multi-Infrastructure Dataset Across Multiple Intersections for Cooperative Perception

**论文链接:** [http://arxiv.org/abs/2510.23478v1](http://arxiv.org/abs/2510.23478v1)

**作者:** Karthikeyan Chandra Sekaran, Markus Geisler, Dominik Rößle, Adithya Mohan, Daniel Cremers, Wolfgang Utschick, Michael Botsch, Werner Huber, Torsten Schön

**发布时间:** 2025-10-27

**备注:** Accepted to NeurIPS 2025. Including supplemental material. For code  and dataset, see https://github.com/thi-ad/UrbanIng-V2X

### GPT解析

### 总结

UrbanIng-V2X是首个大规模、多模态数据集，支持德国Ingolstadt三个城市交叉路口的车辆和基础设施传感器之间的合作感知，包含34个20秒的传感器序列，提供多种传感器数据，并以10Hz频率标注3D边界框。

### 背景

现有合作感知数据集在智能移动应用中起关键作用，但真实世界数据集通常仅限于单个交叉路口或单辆车，缺乏多连接车辆和基础设施传感器跨越多个交叉路口的全面感知数据集，限制了算法在多样化交通环境中的基准测试。

### 目的

解决现有数据集的局限性，引入首个大规模、多模态合作感知数据集UrbanIng-V2X，支持车辆和基础设施传感器之间的合作感知。

### 方法

在德国Ingolstadt的三个城市交叉路口部署传感器，收集34个时间对齐和空间校准的传感器序列(每个20秒)，涉及两辆车和最多三个基础设施传感器杆，使用12个车载RGB摄像头、2个车载LiDAR、17个基础设施热成像摄像头和12个基础设施LiDAR，以10Hz频率标注13个类别的3D边界框。

### 主要发现

提供使用最先进的合作感知方法的全面评估，验证了数据集的有效性和实用性。

### 结论

公开提供代码库、数据集、高清地图和完整数据收集环境的数字孪生，促进合作感知领域的研究和发展。

### 翻译

近期的合作感知数据集通过促进智能体之间的信息交换，在推进智能移动应用方面发挥了关键作用，帮助克服遮挡等挑战，并提高整体场景理解能力。虽然一些现有的真实世界数据集同时包含车辆对车辆和车辆对基础设施的交互，但它们通常仅限于单个交叉路口或单辆车。一个包含多个连接车辆和基础设施传感器跨越多个交叉路口的全面感知数据集仍然不可用，限制了算法在多样化交通环境中的基准测试。因此，可能会发生过拟合，模型可能由于相似的交叉路口布局和交通参与者行为而表现出误导性的高性能。为解决这一差距，我们引入了UrbanIng-V2X，这是首个大规模、多模态数据集，支持在德国Ingolstadt三个城市交叉路口部署的车辆和基础设施传感器之间的合作感知。UrbanIng-V2X包含34个时间对齐和空间校准的传感器序列，每个持续20秒。所有序列包含三个交叉路口中一个的记录，涉及两辆车和最多三个基础设施安装的传感器杆，在协调场景中运行。总的来说，UrbanIng-V2X提供来自12个车载RGB摄像头、2个车载LiDAR、17个基础设施热成像摄像头和12个基础设施LiDAR的数据。所有序列以10Hz的频率标注3D边界框，涵盖13个对象类别，导致整个数据集大约有712k个标注实例。我们使用最先进的合作感知方法提供了全面评估，并公开提供了代码库、数据集、高清地图和完整数据收集环境的数字孪生。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决的问题是缺乏一个大规模、多车辆、多基础设施、多交叉路口的合作感知数据集。这个问题在现实中很重要，因为城市交叉路口是自动驾驶中最复杂的场景之一，单一智能体的感知系统常因视野受限和遮挡而无法检测关键物体，而合作感知可以克服这些限制。缺乏多样性的数据集会导致算法过拟合，模型可能因相似场景而表现出误导性的高性能，限制了真实世界应用。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者通过分析现有数据集的不足（如只包含单个交叉路口或车辆、缺乏多种传感器类型）来设计方法。他们精心设计了传感器部署（两辆车各配备6个RGB摄像头和1个激光雷达，三个交叉路口配备热成像摄像头和激光雷达系统），实现了精确的传感器同步和校准，并从8小时数据中挑选34个代表性场景进行标注。作者借鉴了现有工作如V2V4Real、DAIR-V2X-C等数据集的经验，同时采用了类似nuScenes的标注方法和OpenCOOD的格式转换工具。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是创建一个大规模、多模态、多交叉路口的合作感知数据集，使研究人员能够开发和评估在复杂城市环境中能有效协作的感知算法。整体流程包括：1)传感器部署（车载和基础设施）；2)数据采集（三个交叉路口34个20秒序列）；3)传感器同步与校准（UTC时钟同步、精确校准）；4)场景选择与标注（多样化光照条件、10Hz频率标注13个物体类别）；5)数据发布与工具提供（代码库、高清地图、数字孪生）。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)首个多车辆、多基础设施、多交叉路口的合作感知数据集；2)引入最多的合作传感器和热成像相机等多模态传感器；3)支持多种合作感知基准任务；4)提供综合基准评估；5)提供开发者工具包和数字孪生。相比之前工作，UrbanIng-V2X同时支持多车辆和多基础设施，覆盖多个交叉路口，提供更丰富的传感器组合和更全面的标注（13个类别、712k实例），还提供了数字孪生环境和新的数据集分割策略以评估泛化能力。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'UrbanIng-V2X数据集通过提供首个大规模、多车辆、多基础设施、多交叉路口的合作感知数据集，克服了现有数据集的局限性，为开发和评估在复杂城市环境中能有效协作的感知算法提供了坚实基础，同时揭示了模型在未见环境中的泛化挑战。'}


### 论文摘要

Recent cooperative perception datasets have played a crucial role in advancing smart mobility applications by enabling information exchange between intelligent agents, helping to overcome challenges such as occlusions and improving overall scene understanding. While some existing real-world datasets incorporate both vehicle-to-vehicle and vehicle-to-infrastructure interactions, they are typically limited to a single intersection or a single vehicle. A comprehensive perception dataset featuring multiple connected vehicles and infrastructure sensors across several intersections remains unavailable, limiting the benchmarking of algorithms in diverse traffic environments. Consequently, overfitting can occur, and models may demonstrate misleadingly high performance due to similar intersection layouts and traffic participant behavior. To address this gap, we introduce UrbanIng-V2X, the first large-scale, multi-modal dataset supporting cooperative perception involving vehicles and infrastructure sensors deployed across three urban intersections in Ingolstadt, Germany. UrbanIng-V2X consists of 34 temporally aligned and spatially calibrated sensor sequences, each lasting 20 seconds. All sequences contain recordings from one of three intersections, involving two vehicles and up to three infrastructure-mounted sensor poles operating in coordinated scenarios. In total, UrbanIng-V2X provides data from 12 vehicle-mounted RGB cameras, 2 vehicle LiDARs, 17 infrastructure thermal cameras, and 12 infrastructure LiDARs. All sequences are annotated at a frequency of 10 Hz with 3D bounding boxes spanning 13 object classes, resulting in approximately 712k annotated instances across the dataset. We provide comprehensive evaluations using state-of-the-art cooperative perception methods and publicly release the codebase, dataset, HD map, and a digital twin of the complete data collection environment.

---

## 69. VEHME: A Vision-Language Model For Evaluating Handwritten Mathematics Expressions

**论文链接:** [http://arxiv.org/abs/2510.22798v1](http://arxiv.org/abs/2510.22798v1)

**作者:** Thu Phuong Nguyen, Duc M. Nguyen, Hyotaek Jeon, Hyunwook Lee, Hyunmin Song, Sungahn Ko, Taehwan Kim

**发布时间:** 2025-10-26

**备注:** EMNLP 2025. Project Website: https://vehme.github.io/

### GPT解析

### 总结

介绍VEHME，一种用于评估手写数学表达式的视觉语言模型，能够以高准确性和可解释的推理痕迹评估开放形式的手写数学答案。

### 背景

自动评估手写数学解题是教育技术中的重要问题，具有实际应用，但由于学生作业的多样格式、非结构化布局和符号复杂性，这仍然是一个重大挑战。

### 目的

开发VEHME模型，以高准确性和可解释的推理痕迹评估开放形式的手写数学答案。

### 方法

VEHME采用两阶段训练管道：使用结构化推理数据进行监督微调；通过强化学习使模型输出与多维度评分目标（正确性、推理深度和错误定位）保持一致；提出表达式感知的视觉提示模块，在合成的多行数学表达式数据集上训练，以在视觉异构输入中稳健地引导注意力。

### 主要发现

在AIHub和FERMAT数据集上评估，VEHME在开源模型中取得了最先进的性能，并接近专有系统的准确性。

### 结论

VEHME展示了其作为可扩展且可访问的自动数学评估工具的潜力，训练和实验代码已在GitHub公开存储库中可用。

### 翻译

自动评估手写数学解题是教育技术中的一个重要问题，具有实际应用，但由于学生作业的多样格式、非结构化布局和符号复杂性，这仍然是一个重大挑战。为应对这一挑战，我们介绍了VEHME——一种用于评估手写数学表达式的视觉语言模型——旨在以高准确性和可解释的推理痕迹评估开放形式的手写数学答案。VEHME集成了一个两阶段训练管道：(i) 使用结构化推理数据进行监督微调，(ii) 强化学习使模型输出与多维度评分目标（包括正确性、推理深度和错误定位）保持一致。为增强空间理解，我们提出了一个表达式感知的视觉提示模块，在我们合成的多行数学表达式数据集上训练，以在视觉异构输入中稳健地引导注意力。在AIHub和FERMAT数据集上评估，VEHME在开源模型中取得了最先进的性能，并接近专有系统的准确性，展示了其作为可扩展且可访问的自动数学评估工具的潜力。我们的训练和实验代码已在GitHub公开存储库中提供。


### 论文摘要

Automatically assessing handwritten mathematical solutions is an important problem in educational technology with practical applications, but it remains a significant challenge due to the diverse formats, unstructured layouts, and symbolic complexity of student work. To address this challenge, we introduce VEHME-a Vision-Language Model for Evaluating Handwritten Mathematics Expressions-designed to assess open-form handwritten math responses with high accuracy and interpretable reasoning traces. VEHME integrates a two-phase training pipeline: (i) supervised fine-tuning using structured reasoning data, and (ii) reinforcement learning that aligns model outputs with multi-dimensional grading objectives, including correctness, reasoning depth, and error localization. To enhance spatial understanding, we propose an Expression-Aware Visual Prompting Module, trained on our synthesized multi-line math expressions dataset to robustly guide attention in visually heterogeneous inputs. Evaluated on AIHub and FERMAT datasets, VEHME achieves state-of-the-art performance among open-source models and approaches the accuracy of proprietary systems, demonstrating its potential as a scalable and accessible tool for automated math assessment. Our training and experiment code is publicly available at our GitHub repository.

---

## 70. IGGT: Instance-Grounded Geometry Transformer for Semantic 3D Reconstruction

**论文链接:** [http://arxiv.org/abs/2510.22706v2](http://arxiv.org/abs/2510.22706v2)

**作者:** Hao Li, Zhengyu Zou, Fangfu Liu, Xuanyang Zhang, Fangzhou Hong, Yukang Cao, Yushi Lan, Manyuan Zhang, Gang Yu, Dingwen Zhang, Ziwei Liu

**发布时间:** 2025-10-26

**备注:** https://github.com/lifuguan/IGGT_official

### GPT解析

### 总结

这篇论文提出了InstanceGrounded Geometry Transformer (IGGT)，一种端到端的大型统一transformer，用于统一3D场景的空间重建和实例级上下文理解。

### 背景

人类自然将3D世界的几何结构和语义内容作为相互交织的维度感知，但先前方法优先训练几何模型进行低级3D重建，将高级空间理解孤立处理，忽视了两者间的相互作用，限制了泛化能力和下游任务表现。

### 目的

开发一个统一框架，同时处理3D场景的几何结构和语义理解，提高3D场景分析的准确性和泛化能力。

### 方法

提出IGGT模型，设计3D一致性对比学习策略，通过仅2D视觉输入编码具有几何结构和实例聚类的统一表示，并构建InsScene-15K数据集，包含高质量RGB图像、姿态、深度图和3D一致的实例级掩码注释。

### 主要发现

通过统一几何结构和语义理解的方法，可以改善3D场景的理解和重建效果，实现从2D输入到连贯3D场景的有效转换。

### 结论

所提出的IGGT方法和3D一致性对比学习策略能够有效地将2D视觉输入转换为连贯的3D场景，并明确区分对象实例，为3D场景分析提供了新的统一框架。

### 翻译

人类自然地将3D世界的几何结构和语义内容作为相互交织的维度来感知，从而能够连贯准确地理解复杂场景。然而，先前的方法优先训练大型几何模型进行低级3D重建，将高级空间理解孤立处理，忽视了这两个基本方面之间的相互作用，从而限制了泛化能力并在下游3D理解任务中表现不佳。最近的尝试通过简单地将3D模型与特定语言模型对齐来缓解这一问题，从而限制了感知能力，并限制了下游任务的适应性。在本文中，我们提出了InstanceGrounded Geometry Transformer (IGGT)，一个端到端的大型统一transformer，用于统一空间重建和实例级上下文理解的知识。具体来说，我们设计了一种3D一致性对比学习策略，指导IGGT仅通过2D视觉输入来编码具有几何结构和实例聚类的统一表示。该表示支持将2D视觉输入一致提升为具有明确不同对象实例的连贯3D场景。为了促进这一任务，我们进一步构建了InsScene-15K，这是一个大规模数据集，包含高质量的RGB图像、姿态、深度图和3D一致的实例级掩码注释，采用新颖的数据整理流程。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决3D场景几何重建与高层次语义理解相分离的问题。人类自然地将3D世界的几何结构和语义内容视为交织在一起的维度，而现有方法通常将这两个方面视为独立任务，导致它们无法相互增强，限制了模型在下游任务中的泛化能力和性能。这个问题很重要，因为统一的几何-语义表示对于机器人操作、增强现实/虚拟现实和空间规划等应用至关重要。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先认识到现有方法的局限性：几何重建和语义理解被分离处理，或简单地将3D模型与特定语言模型对齐，导致感知能力受限和适应性差。他们设计了一个端到端的统一框架，通过联合训练耦合几何和语义特征，让模型自主学习3D实例级语义与几何结构的关系。作者借鉴了VGGT的结构，使用DINOv2进行特征提取，采用DPT架构进行密集预测，并利用SAM2进行数据标注。他们还创新性地设计了3D一致的对比学习策略来增强模型性能。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是通过联合训练将几何和实例级语义特征耦合，实现上下文理解和几何重建的相互改进。整体流程包括：1)接收多视图图像输入；2)使用大型统一变换器提取统一表示；3)通过几何头和实例头分别预测几何结构和实例特征；4)应用跨模态融合块增强实例特征的细粒度空间感知；5)使用3D一致的对比监督确保多视图一致性；6)通过聚类生成实例掩码，用于下游任务如空间跟踪、开放词汇分割和场景理解。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)统一的3D重建与理解框架IGGT；2)3D一致的对比学习策略；3)InsScene-15K大规模数据集；4)实例级场景理解范式。相比之前的工作，不同之处在于：不是简单对齐几何与语言特征，而是通过联合训练实现相互增强；不与特定视觉语言模型紧密耦合，可以集成更强大的基础模型；能够区分同一语义类别内的不同对象，支持更复杂的下游应用；实现了即插即用的方式与各种视觉语言模型和大型多模态模型集成。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'IGGT通过提出统一的实例级几何变换器框架和3D一致的对比学习策略，实现了几何重建与语义理解的深度融合，显著提升了3D场景重建与理解的质量和一致性。'}


### 论文摘要

Humans naturally perceive the geometric structure and semantic content of a 3D world as intertwined dimensions, enabling coherent and accurate understanding of complex scenes. However, most prior approaches prioritize training large geometry models for low-level 3D reconstruction and treat high-level spatial understanding in isolation, overlooking the crucial interplay between these two fundamental aspects of 3D-scene analysis, thereby limiting generalization and leading to poor performance in downstream 3D understanding tasks. Recent attempts have mitigated this issue by simply aligning 3D models with specific language models, thus restricting perception to the aligned model's capacity and limiting adaptability to downstream tasks. In this paper, we propose InstanceGrounded Geometry Transformer (IGGT), an end-to-end large unified transformer to unify the knowledge for both spatial reconstruction and instance-level contextual understanding. Specifically, we design a 3D-Consistent Contrastive Learning strategy that guides IGGT to encode a unified representation with geometric structures and instance-grounded clustering through only 2D visual inputs. This representation supports consistent lifting of 2D visual inputs into a coherent 3D scene with explicitly distinct object instances. To facilitate this task, we further construct InsScene-15K, a large-scale dataset with high-quality RGB images, poses, depth maps, and 3D-consistent instance-level mask annotations with a novel data curation pipeline.

---

## 71. BLIP-FusePPO: A Vision-Language Deep Reinforcement Learning Framework for Lane Keeping in Autonomous Vehicles

**论文链接:** [http://arxiv.org/abs/2510.22370v1](http://arxiv.org/abs/2510.22370v1)

**作者:** Seyed Ahmad Hosseini Miangoleh, Amin Jalal Aghdasian, Farzaneh Abdollahi

**发布时间:** 2025-10-25

**备注:** https://github.com/Amin-A96/BLIP-FusePPO-A-Vision-Language-Deep-Reinforcement-Learning-Framework-for-Lane-Keeping-in-Autonomous.git

### GPT解析

### 总结

本文提出了BLIP-FusePPO，一种用于自动驾驶车道保持的新型多模态强化学习框架，将视觉语言模型生成的语义嵌入与几何状态、LiDAR观测和PID控制反馈相融合。

### 背景

自动驾驶中的车道保持需要结合高级语义理解和低级控制信号，而现有方法可能仅使用语义模型来塑造奖励或未充分利用多模态信息。

### 目的

开发一个能够学习具有环境意识且易于理解的驾驶规则的多模态强化学习框架，结合视觉语言模型的高级场景理解与低级控制和空间信号。

### 方法

提出BLIP-FusePPO框架，将视觉语言模型生成的语义嵌入直接融合到代理观测空间中的几何状态、LiDAR观测和PID控制反馈，结合语义、几何和控制感知表示，并使用包含语义对齐、车道保持准确性、障碍物避让和速度调节的混合奖励函数。

### 主要发现

所提出的模型在车道保持的稳定性和适应性方面优于最佳视觉和多模态强化学习基线，在各种困难的驾驶情况下表现良好，且直接嵌入语义特征减少了昂贵的运行时推理，确保语义指导始终可用。

### 结论

BLIP-FusePPO是一个有效的多模态强化学习框架，能够提高自动驾驶车道保持任务的性能和泛化能力。

### 翻译

在本文中，我们提出了基于引导语言-图像预训练的融合状态表示近端策略优化（BLIP-FusePPO），这是一种用于自动驾驶车道保持的新型多模态强化学习框架，其中视觉语言模型生成的语义嵌入直接与几何状态、LiDAR观测和基于比例-积分-微分的控制反馈在代理观测空间内融合。所提出的方法通过结合视觉语言模型的高级场景理解与低级控制和空间信号，使代理能够学习具有环境意识且易于理解的驾驶规则。我们的架构将语义、几何和控制感知表示结合在一起，使策略学习更加稳健。包含语义对齐、车道保持准确性、障碍物避让和速度调节的混合奖励函数有助于学习更加高效和可泛化。我们的方法不同于仅使用语义模型来塑造奖励的方法，而是直接将语义特征嵌入到状态表示中。这减少了昂贵的运行时推理，并确保语义指导始终可用。仿真结果表明，在广泛的困难驾驶情况下，所提出的模型在车道保持的稳定性和适应性方面优于最佳视觉和多模态强化学习基线。我们公开提供代码。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决自动驾驶车辆车道保持任务中语义感知与控制感知融合不足的问题。这一问题在现实中很重要，因为现有系统在复杂环境（如车道标记磨损、不同光照条件或被遮挡车道）中表现有限，而缺乏对场景语义理解的系统难以适应多变路况，影响自动驾驶的安全性和可靠性。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先分析了现有方法的局限性：传统视觉方法缺乏语义理解、多模态RL方法仅用VLM塑造奖励而非融入状态、传统控制器缺乏可解释性。作者借鉴了BLIP视觉语言模型用于语义提取、PPO算法用于稳定策略学习、PID控制器提供控制反馈等现有工作。创新点在于将语义特征与几何状态、LiDAR观测和PID控制反馈直接融合到状态表示中，而非仅用于奖励塑造。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是将语义感知（通过BLIP提取）与控制感知（通过PID获取）直接融合到强化学习智能体的状态表示中，使智能体同时理解场景语义上下文和执行精确控制。整体流程包括：1)混合状态表示（RGB视觉、LiDAR数据、PID反馈、语义嵌入）；2)预处理和特征提取；3)数据增强（水平镜像）；4)连续动作空间设计（转向和速度控制）；5)混合奖励函数（车道保持、障碍物避免、速度匹配等）；6)使用PPO算法训练策略网络。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点：1)新颖架构将BLIP语义嵌入和PID信号直接注入状态表示；2)基于PID控制的状态增强技术提高学习稳定性；3)新型混合奖励函数整合语义对齐和几何遵循；4)直接语义特征嵌入而非仅用于奖励塑造。相比之前工作，不同之处在于：传统方法缺乏语义理解、现有多模态RL方法仅用VLM塑造奖励、传统控制器缺乏可解释性、基于VLM的RL系统计算开销大。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '本文提出了BLIP-FusePPO框架，通过融合语义感知与控制感知到状态表示中，显著提高了自动驾驶车道保持的稳定性和适应性，同时降低了计算开销，为更安全鲁棒的自动驾驶系统提供了新思路。'}


### 论文摘要

In this paper, we propose Bootstrapped Language-Image Pretraining-driven Fused State Representation in Proximal Policy Optimization (BLIP-FusePPO), a novel multimodal reinforcement learning (RL) framework for autonomous lane-keeping (LK), in which semantic embeddings generated by a vision-language model (VLM) are directly fused with geometric states, LiDAR observations, and Proportional-Integral-Derivative-based (PID) control feedback within the agent observation space. The proposed method lets the agent learn driving rules that are aware of their surroundings and easy to understand by combining high-level scene understanding from the VLM with low-level control and spatial signals. Our architecture brings together semantic, geometric, and control-aware representations to make policy learning more robust. A hybrid reward function that includes semantic alignment, LK accuracy, obstacle avoidance, and speed regulation helps learning to be more efficient and generalizable. Our method is different from the approaches that only use semantic models to shape rewards. Instead, it directly embeds semantic features into the state representation. This cuts down on expensive runtime inference and makes sure that semantic guidance is always available. The simulation results show that the proposed model is better at LK stability and adaptability than the best vision-based and multimodal RL baselines in a wide range of difficult driving situations. We make our code publicly available.

---

## 72. MOGRAS: Human Motion with Grasping in 3D Scenes

**论文链接:** [http://arxiv.org/abs/2510.22199v1](http://arxiv.org/abs/2510.22199v1)

**作者:** Kunal Bhosikar, Siddharth Katageri, Vivek Madhavaram, Kai Han, Charu Sharma

**发布时间:** 2025-10-25

**备注:** British Machine Vision Conference Workshop - From Scene Understanding  to Human Modeling

### GPT解析

### 总结

该研究提出了MOGRAS数据集和一种简单有效的方法，用于解决在3D场景中生成物理合理的全身抓取运动的挑战，通过定量和定性实验验证了其有效性。

### 背景

生成与物体交互的真实全身运动对机器人技术、虚拟现实和人机交互应用至关重要，但现有方法要么缺乏精细任务的保真度，要么忽略了周围3D场景。

### 目的

弥合现有方法在生成全身抓取运动方面的局限性，提供能够在3D场景中生成物理合理全身抓取运动的解决方案。

### 方法

引入MOGRAS（3D场景中的人体抓取运动）数据集，提供预抓取全身行走运动和最终抓取姿态；利用该数据集基准测试现有方法；提出一种简单有效的方法使现有方法能在3D场景中无缝工作。

### 主要发现

现有全身抓握方法在场景感知生成方面存在局限性；所提出的方法在全身抓取运动生成方面取得了显著改进；通过大量定量和定性实验验证了数据集的有效性。

### 结论

该研究为更真实的人体-场景交互铺平了道路，展示了在3D场景中生成物理合理全身抓取运动的可行性。

### 翻译

生成与物体交互的真实全身运动对机器人技术、虚拟现实和人机交互应用至关重要。虽然现有方法可以生成3D场景中的全身运动，但它们通常缺乏精细任务（如物体抓取）的保真度。相反，生成精确抓取运动的方法通常忽略了周围的3D场景。在3D场景中生成物理上合理的全身抓取运动仍然是一个重大挑战。为解决这一问题，我们引入了MOGRAS（3D场景中的人体抓取运动），这是一个弥合这一差距的大规模数据集。MOGRAS在丰富的3D室内场景标注中提供了预抓取的全身行走运动和最终抓取姿态。我们利用MOGRAS对现有的全身抓取方法进行基准测试，展示了它们在场景感知生成方面的局限性。此外，我们提出了一种简单而有效的方法，使现有方法能够在3D场景中无缝工作。通过大量的定量和定性实验，我们验证了数据集的有效性，并突显了我们提出方法所取得的显著改进，为更真实的人体-场景交互铺平了道路。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决在3D场景中生成物理合理的全身抓取运动的问题。现有方法要么能生成全身运动但缺乏精细抓保真度，要么能生成精确抓取但忽略周围3D场景。这个问题对机器人、虚拟现实和人机交互等领域至关重要，因为准确建模人-物体交互能支持行为分析、智能机器人系统开发和逼真虚拟环境创建。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先识别现有研究的差距：全身运动方法缺乏精细抓取能力，抓取方法忽略场景上下文。考虑到手动捕获此类数据成本高昂，他们设计了自动化数据生成框架。作者借鉴了HUMANISE的运动对齐方法、AMASS的行走序列、ScanNetv2的3D环境、BABEL的运动标签、GRAB的抓取物体，并改进了FLEX的抓取姿势生成和PriorMDM的运动填充技术。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是创建MOGRAS数据集，提供3D室内场景中的全身抓取运动序列，包括预抓取行走和最终抓取姿势。实现流程分五步：1)行走运动对齐和物体放置；2)改进ScanNet场景的地板对齐；3)使用改进的FLEX生成抓取姿势；4)用PriorMDM生成从行走到抓取的平滑过渡；5)确保数据集规模和质量，通过自动过滤和人类评估保证物理合理性。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点：1)MOGRAS数据集：首个结合全身运动、精细抓取和3D场景的大规模合成数据集；2)GNet++方法：通过场景编码和穿透损失实现场景感知抓取；3)场景处理改进：解决地板不对齐问题。相比之前工作，MOGRAS是首个同时包含三种元素(3D场景、精细抓取、全身运动)的数据集，而GNet++显式考虑环境约束，而之前方法如GOAL和SAGA忽略了场景上下文。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '通过引入首个大规模场景感知全身抓取数据集MOGRAS和相应生成方法GNet++，论文弥合了3D场景中全身运动生成与精细物体抓取之间的差距，为更真实的人-场景交互研究奠定了基础。'}


### 论文摘要

Generating realistic full-body motion interacting with objects is critical for applications in robotics, virtual reality, and human-computer interaction. While existing methods can generate full-body motion within 3D scenes, they often lack the fidelity for fine-grained tasks like object grasping. Conversely, methods that generate precise grasping motions typically ignore the surrounding 3D scene. This gap, generating full-body grasping motions that are physically plausible within a 3D scene, remains a significant challenge. To address this, we introduce MOGRAS (Human MOtion with GRAsping in 3D Scenes), a large-scale dataset that bridges this gap. MOGRAS provides pre-grasping full-body walking motions and final grasping poses within richly annotated 3D indoor scenes. We leverage MOGRAS to benchmark existing full-body grasping methods and demonstrate their limitations in scene-aware generation. Furthermore, we propose a simple yet effective method to adapt existing approaches to work seamlessly within 3D scenes. Through extensive quantitative and qualitative experiments, we validate the effectiveness of our dataset and highlight the significant improvements our proposed method achieves, paving the way for more realistic human-scene interactions.

---

## 73. CogStereo: Neural Stereo Matching with Implicit Spatial Cognition Embedding

**论文链接:** [http://arxiv.org/abs/2510.22119v1](http://arxiv.org/abs/2510.22119v1)

**作者:** Lihuang Fang, Xiao Hu, Yuchen Zou, Hong Zhang

**发布时间:** 2025-10-25

**备注:** 9 pages, 6 figures

### GPT解析

### 总结

这篇论文介绍了一种名为CogStereo的新型立体匹配框架，通过引入空间认知机制来改进立体匹配性能，特别是在处理遮挡或弱纹理等挑战性区域时表现出色，同时提高了跨域泛化能力。

### 背景

深度立体匹配通过微调在基准数据集上取得了显著进展，但在零样本泛化方面不如其他视觉任务中的基础模型。

### 目的

开发一种不依赖数据集特定先验的框架，解决立体匹配中的挑战性问题，提高跨域泛化能力，并将立体视觉转向认知驱动的方法。

### 方法

CogStereo通过使用单目深度特征作为先验，将隐式空间认知嵌入到细化过程中，捕获超越局部对应的全场景理解。该方法采用双条件细化机制，结合逐像素不确定性和认知引导特征，实现对不匹配的一致性全局校正。

### 主要发现

CogStereo在Scene Flow、KITTI、Middlebury、ETH3D、EuRoc和真实世界等多个数据集上实现了最先进的结果，并在跨域泛化方面表现出色。

### 结论

CogStereo成功解决了传统立体匹配方法在处理挑战性区域时的局限性，通过引入空间认知机制提高了立体视觉系统的泛化能力，推动了立体视觉向认知驱动方向发展。

### 翻译

深度立体匹配通过微调在基准数据集上取得了显著进展，但在零样本泛化方面不如其他视觉任务中的基础模型。我们引入了CogStereo，一种新颖的框架，它解决了遮挡或弱纹理等挑战性区域，而不依赖于数据集特定的先验。CogStereo通过使用单目深度特征作为先验，将隐式空间认知嵌入到细化过程中，捕获超越局部对应的全局场景理解。这种方法确保了结构一致性的视差估计，即使在仅靠几何不足的区域。CogStereo采用双条件细化机制，结合逐像素不确定性和认知引导特征，实现对不匹配的一致性全局校正。在Scene Flow、KITTI、Middlebury、ETH3D、EuRoc和真实世界上的大量实验表明，CogStereo不仅取得了最先进的结果，还在跨域泛化方面表现出色，将立体视觉转向认知驱动的方法。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决深度立体匹配方法在零样本泛化能力上的不足问题。当前方法虽然在基准数据集上表现良好，但在遮挡区域、弱纹理等困难区域表现不佳，且缺乏强大的零样本泛化能力。这个问题在自动驾驶、机器人等应用中至关重要，因为这些应用需要在各种不同环境下保持一致的深度估计性能，而现有方法过度依赖局部几何对应，缺乏全局场景理解能力。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': "作者通过观察立体匹配与其他视觉任务基础模型的差距，引入了'空间认知'概念，借鉴单目深度估计的成功经验，特别是Depth Anything v2捕获的物体级几何和全局场景理解能力。作者还借鉴了条件控制思想，设计了双条件修正机制。创新之处在于将不确定性估计提前到成本体积阶段，而非视差回归之后，并设计了不确定性引导的空间认知注意力机制、低不确定性区域的KNN对齐策略和突然深度差异感知梯度损失等组件。", '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是将隐式空间认知嵌入到立体匹配过程中，利用单目深度特征作为先验，超越局部对应关系，捕获整体场景理解。整体流程分为两阶段：第一阶段是成本体积不确定性估计预训练，使用标准立体匹配骨干网络提取特征，构建三维成本体积，并预测每个像素的不确定性；第二阶段是通过空间认知的双条件修正，整合不确定性先验与空间认知特征，使用注意力机制进行视差修正，并通过KNN对齐防止度量漂移，最后应用特殊损失函数确保视差图的平滑性。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)首次将隐式空间认知概念引入立体匹配；2)设计双条件修正机制结合不确定性和认知特征；3)在成本体积阶段而非视差回归后进行不确定性估计；4)引入低不确定性区域的KNN对齐策略防止度量漂移；5)设计突然深度差异感知梯度损失。相比之前工作，CogStereo超越了纯几何匹配，实现了强大的零样本泛化，改变了不确定性处理方式，创新性地利用深度基础模型的中间特征而非原始深度预测，并针对遮挡、弱纹理等困难区域提供了更鲁棒的解决方案。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'CogStereo通过引入隐式空间认知嵌入到立体匹配过程中，结合像素级不确定性与认知引导特征，显著提升了在遮挡、弱纹理等困难区域的零样本泛化能力和视差估计准确性，实现了从纯几何匹配向认知驱动立体视觉的转变。'}


### 论文摘要

Deep stereo matching has advanced significantly on benchmark datasets through fine-tuning but falls short of the zero-shot generalization seen in foundation models in other vision tasks. We introduce CogStereo, a novel framework that addresses challenging regions, such as occlusions or weak textures, without relying on dataset-specific priors. CogStereo embeds implicit spatial cognition into the refinement process by using monocular depth features as priors, capturing holistic scene understanding beyond local correspondences. This approach ensures structurally coherent disparity estimation, even in areas where geometry alone is inadequate. CogStereo employs a dual-conditional refinement mechanism that combines pixel-wise uncertainty with cognition-guided features for consistent global correction of mismatches. Extensive experiments on Scene Flow, KITTI, Middlebury, ETH3D, EuRoc, and real-world demonstrate that CogStereo not only achieves state-of-the-art results but also excels in cross-domain generalization, shifting stereo vision towards a cognition-driven approach.

---

## 74. GeoThought: A Dataset for Enhancing Mathematical Geometry Reasoning in Vision-Language Models

**论文链接:** [http://arxiv.org/abs/2510.21881v1](http://arxiv.org/abs/2510.21881v1)

**作者:** Nannan Shi, Chuanyu Qin, Shipeng Song, Man Luo

**发布时间:** 2025-10-23

### GPT解析

### 总结

该研究针对大型语言模型在几何视觉推理任务中的性能下降问题，开发了一个名为GeoThoughts的几何推理数据集和一个名为GeoThought-MLLM的多模态数学推理模型，通过链式思维训练显著提升了模型在几何问题上的表现。

### 背景

大型语言模型在文本数学问题解决中表现出强大的推理能力，但在视觉推理任务，特别是几何问题解决中，其性能显著下降。这是因为几何问题具有独特挑战：一是几何本身的复杂性需要详细的图像理解和多步推理；二是现有数据集规模不足、多样性有限且缺乏明确的推理痕迹，阻碍了有效模型训练。

### 目的

开发一个全面的几何推理数据集和一个多模态数学推理模型，以解决大型语言模型在几何问题解决中的性能下降问题。

### 方法

1. 创建了GeoThoughts数据集，包含两个子集：Geo-Thought-6K（6,243个样本）和Geo-Thought-Augmented-10K（10,834个样本）。2. 每个数据条目包括视觉描述、分步解决方案、明确的推理链、反思步骤和最终答案。3. 基于此数据集开发了GeoThought-MLLM，一个在问题解决过程中生成详细思考过程的多模态数学推理模型。

### 主要发现

1. 使用链式思维数据集训练的GeoThought-MLLM在几何任务上优于现有基准。2. 训练显著提升了模型在领域内和领域外几何推理能力。3. 错误主要源于数学概念错误解释或空间判断失误。4. 通过调用链式思维纠正这些错误，模型能够产生正确答案。

### 结论

GeoThoughts数据集和GeoThought-MLLM模型有效解决了大型语言模型在几何问题解决中的性能下降问题，为几何视觉推理任务提供了新的解决方案和见解。

### 翻译

大型语言模型在基于文本的数学问题解决中表现出强大的推理能力；然而，当适应到视觉推理任务，特别是几何问题解决时，它们的性能大幅下降，因为几何问题带来了独特的挑战。具体来说，这些挑战源于两个关键因素：首先，几何本身的复杂性需要详细的图像理解和多步推理；其次，现有数据集的规模、多样性和明确的推理痕迹不足，从而阻碍了有效的模型训练。为应对这些挑战，我们开发了GeoThoughts数据集，这是一个全面的几何推理语料库，包含两个子集：包含6,243个样本的Geo-Thought-6K及其增强版本Geo-Thought-Augmented-10K，包含10,834个样本。每个条目包括视觉描述、分步解决方案、明确的推理链、反思步骤和最终答案。使用此数据集，我们开发了GeoThought-MLLM，一个在问题解决过程中生成详细思考过程的数学推理多模态模型。我们的模型在几何任务上优于现有基准，证明使用我们的链式思维数据集进行训练可以提升领域内和领域外设置的几何推理能力。最后，我们分析了失败案例，发现错误主要源于数学概念错误解释或空间判断失误。通过调用链式思维纠正这些错误，模型产生了正确答案。


### 论文摘要

Large language models (LLMs) have demonstrated strong reasoning capabilities in text-based mathematical problem solving; however, when adapted to visual reasoning tasks, particularly geometric problem solving, their performance substantially declines because geometric problems present unique challenges. Specifically, these challenges stem from two key factors: first, the intrinsic complexity of geometry requiring detailed image comprehension and multi-step reasoning, and second, the limitations of existing datasets which lack sufficient scale, diversity, and explicit reasoning traces, consequently hindering effective model training. To address these challenges, we developed the GeoThoughts dataset, a comprehensive geometric reasoning corpus with two subsets: Geo-Thought-6K with 6,243 samples and its augmented version Geo-Thought-Augmented-10K containing 10,834 samples. Each entry includes visual descriptions, step-by-step solutions, explicit reasoning chains, reflection steps, and final answers. Using this dataset, we developed GeoThought-MLLM, a mathematical reasoning multimodal model that generates detailed thinking processes during problem-solving. Our model outperforms existing benchmarks in geometric tasks, demonstrating that training with our Chain-of-Thought dataset improves geometric reasoning capabilities across both in-domain and out-of-domain settings. Finally, we analyze failure cases and observe that errors primarily arise from incorrect interpretation of mathematical concepts or spatial misjudgment. By invoking CoT to correct these mistakes, the model produces correct answers.

---

## 75. GRAPHIA: Harnessing Social Graph Data to Enhance LLM-Based Social Simulation

**论文链接:** [http://arxiv.org/abs/2510.24251v1](http://arxiv.org/abs/2510.24251v1)

**作者:** Jiarui Ji, Zehua Zhang, Zhewei Wei, Bin Tong, Guan Wang, Bo Zheng

**发布时间:** 2025-10-28

### GPT解析

### 总结

Graphia是一种基于大型语言模型的社会图模拟框架，利用图数据作为监督信号通过强化学习对LLM进行后训练，能够预测互动对象和互动方式，在微观和宏观层面都显示出显著的性能提升。

### 背景

大型语言模型在模拟类人社会行为方面展现出潜力，但社会图作为包含局部交互和全局网络结构的高质量监督信号，在LLM训练中仍未得到充分利用。

### 目的

提出Graphia框架，利用图数据作为监督信号，通过强化学习对LLM进行后训练，以缩小代理行为与基于LLM的模拟中网络动力学之间的差距。

### 方法

Graphia训练专门的代理来预测与谁互动（目标选择）和如何互动（边生成），使用基于GNN的结构性奖励，并设计了图生成流程。在归因动态图生成（TDGG）和归纳动态图生成（IDGG）两种设置下进行评估。

### 主要发现

在三个真实世界网络上，Graphia相比最强基线模型，微观对齐方面：综合目标选择分数提高6.1%，边分类准确率提高12%，边内容BERTScore提高27.9%；宏观对齐方面：结构相似性提高41.11%，社会现象（如幂律和回音室）复制能力提高32.98%。Graphia还支持反事实模拟，能在平台激励下生成合理的行为转变。

### 结论

社会图可以作为LLM后训练的高质量监督信号，有效缩小基于LLM的模拟中代理行为与网络动力学之间的差距。

### 翻译

大型语言模型在模拟类人社会行为方面展现出潜力。社会图提供了高质量监督信号，编码了局部交互和全局网络结构，但这些信号在LLM训练中仍未得到充分利用。为解决这一差距，我们提出了Graphia，这是第一个基于LLM的社会图模拟通用框架，它利用图数据作为监督信号，通过强化学习对LLM进行后训练。基于GNN的结构性奖励，Graphia训练专门的代理来预测与谁互动（目标选择）和如何互动（边生成），然后使用设计的图生成流程。我们在两种设置下评估Graphia：归因动态图生成（TDGG），这是使用我们提出的节点级交互对齐指标的微观任务；以及归纳动态图生成（IDGG），这是使用我们提出的指标对齐涌现网络属性的宏观任务。在三个真实世界网络上，Graphia相比最强基线模型，在微观对齐方面提高了6.1%的综合目标选择分数，12%的边分类准确率和27.9%的边内容BERTScore。对于宏观对齐，它实现了41.11%更高的结构相似性和32.98%更好的社会现象（如幂律和回音室）复制能力。Graphia还支持反事实模拟，在平台激励下生成合理的行为转变。我们的结果表明，社会图可以作为LLM后训练的高质量监督信号，缩小基于LLM的模拟中代理行为与网络动力学之间的差距。代码可在https://github.com/Ji-Cather/Graphia.git获取。


### 论文摘要

Large language models (LLMs) have shown promise in simulating human-like social behaviors. Social graphs provide high-quality supervision signals that encode both local interactions and global network structure, yet they remain underutilized for LLM training. To address this gap, we propose Graphia, the first general LLM-based social graph simulation framework that leverages graph data as supervision for LLM post-training via reinforcement learning. With GNN-based structural rewards, Graphia trains specialized agents to predict whom to interact with (destination selection) and how to interact (edge generation), followed by designed graph generation pipelines. We evaluate Graphia under two settings: Transductive Dynamic Graph Generation (TDGG), a micro-level task with our proposed node-wise interaction alignment metrics; and Inductive Dynamic Graph Generation (IDGG), a macro-level task with our proposed metrics for aligning emergent network properties. On three real-world networks, Graphia improves micro-level alignment by 6.1% in the composite destination selection score, 12% in edge classification accuracy, and 27.9% in edge content BERTScore over the strongest baseline. For macro-level alignment, it achieves 41.11% higher structural similarity and 32.98% better replication of social phenomena such as power laws and echo chambers. Graphia also supports counterfactual simulation, generating plausible behavioral shifts under platform incentives. Our results show that social graphs can serve as high-quality supervision signals for LLM post-training, closing the gap between agent behaviors and network dynamics for LLM-based simulation. Code is available at https://github.com/Ji-Cather/Graphia.git.

---

## 76. MAGNET: A Multi-Graph Attentional Network for Code Clone Detection

**论文链接:** [http://arxiv.org/abs/2510.24241v1](http://arxiv.org/abs/2510.24241v1)

**作者:** Zixian Zhang, Takfarinas Saber

**发布时间:** 2025-10-28

### GPT解析

### 总结

MAGNET是一种多图注意力框架，通过联合利用AST、CFG和DFG表示来捕获源代码的语法和语义特征，实现了代码克隆检测的最先进性能。

### 背景

代码克隆检测是软件工程中的基础任务，支持重构、调试、剽窃检测和漏洞分析。现有方法通常依赖单一表示（如AST、CFG、DFG），只能捕捉代码语义的部分方面，而混合方法的融合策略通常是手工设计的且效果不佳。

### 目的

提出MAGNET框架，一种多图注意力框架，联合利用AST、CFG和DFG表示来捕获源代码的语法和语义特征。

### 方法

MAGNET结合残差图神经网络与节点级自注意力学习局部和长距离依赖关系，引入门控交叉注意力机制用于细粒度的图间交互，采用Set2Set池化将多图嵌入融合为统一的程序级表示。

### 主要发现

在BigCloneBench和Google Code Jam上的实验表明，MAGNET分别达到96.5%和99.2%的总体F1分数，实现了最先进的性能。消融研究证实了多图融合和每个注意力组件的关键贡献。

### 结论

MAGNET通过多图表示和注意力机制实现了高效的代码克隆检测，代码已开源于https://github.com/ZixianReid/Multigraph_match。

### 翻译

代码克隆检测是软件工程中的一个基础任务，它支持重构、调试、剽窃检测和漏洞分析。现有方法通常依赖于单一表示，如抽象语法树（AST）、控制流图（CFG）和数据流图（DFG），这些表示只能捕捉代码语义的部分方面。混合方法已经出现，但它们的融合策略通常是手工设计的且效果不佳。在本研究中，我们提出了MAGNET，一种多图注意力框架，它联合利用AST、CFG和DFG表示来捕获源代码的语法和语义特征。MAGNET将残差图神经网络与节点级自注意力相结合，学习局部和长距离依赖关系，引入门控交叉注意力机制用于细粒度的图间交互，并采用Set2Set池化将多图嵌入融合为统一的程序级表示。在BigCloneBench和Google Code Jam上的大量实验表明，MAGNET在两个数据集上分别实现了96.5%和99.2%的总体F1分数，达到了最先进的性能。消融研究证实了多图融合和每个注意力组件的关键贡献。我们的代码可在https://github.com/ZixianReid/Multigraph_match获取。


### 论文摘要

Code clone detection is a fundamental task in software engineering that underpins refactoring, debugging, plagiarism detection, and vulnerability analysis. Existing methods often rely on singular representations such as abstract syntax trees (ASTs), control flow graphs (CFGs), and data flow graphs (DFGs), which capture only partial aspects of code semantics. Hybrid approaches have emerged, but their fusion strategies are typically handcrafted and ineffective. In this study, we propose MAGNET, a multi-graph attentional framework that jointly leverages AST, CFG, and DFG representations to capture syntactic and semantic features of source code. MAGNET integrates residual graph neural networks with node-level self-attention to learn both local and long-range dependencies, introduces a gated cross-attention mechanism for fine-grained inter-graph interactions, and employs Set2Set pooling to fuse multi-graph embeddings into unified program-level representations. Extensive experiments on BigCloneBench and Google Code Jam demonstrate that MAGNET achieves state-of-the-art performance with an overall F1 score of 96.5\% and 99.2\% on the two datasets, respectively. Ablation studies confirm the critical contributions of multi-graph fusion and each attentional component. Our code is available at https://github.com/ZixianReid/Multigraph_match

---

## 77. HyperGraphX: Graph Transductive Learning with Hyperdimensional Computing and Message Passing

**论文链接:** [http://arxiv.org/abs/2510.23980v1](http://arxiv.org/abs/2510.23980v1)

**作者:** Guojing Cong, Tom Potok, Hamed Poursiami, Maryam Parsa

**发布时间:** 2025-10-28

### GPT解析

### 总结

本文提出了一种名为hdgc的新型算法，该算法结合了图卷积与高维计算中的绑定和捆绑操作，用于归纳图学习。

### 背景

在图学习领域，图神经网络和高维计算是两种重要的方法，各有优势和局限性。

### 目的

开发一种能够同时利用图神经网络和高维计算优势的新算法，提高图学习的准确性和效率。

### 方法

hdgc算法将图卷积操作与高维计算中的绑定和捆绑操作相结合，主要在二进制向量上进行学习操作。

### 主要发现

hdgc在预测准确性上优于主流图神经网络实现和最先进的高维计算实现，适用于同质图和异质图；在相同GPU平台上，hdgc比gcnii图神经网络实现平均快9561倍，比hdgl高维计算实现平均快144.5倍。

### 结论

hdgc算法在多种图类型上表现出色，由于主要操作在二进制向量上进行，预期在神经形态和新兴存储器处理设备上具有出色的能源性能。

### 翻译

我们提出了一种名为hdgc的新颖算法，该算法将图卷积与高维计算中的绑定和捆绑操作相结合，用于归纳图学习。在预测准确性方面，hdgc优于主要的和流行的图神经网络实现以及最先进的高维计算实现，适用于一系列同质图和异质图。与我们测试的最准确的学习方法相比，在相同的目标GPU平台上，hdgc比图神经网络实现gcnii平均快9561.0倍，比高维计算实现hdgl平均快144.5倍。由于大部分学习操作在二进制向量上进行，我们期望hdgc在神经形态和新兴存储器处理设备上具有出色的能源性能。


### 论文摘要

We present a novel algorithm, \hdgc, that marries graph convolution with binding and bundling operations in hyperdimensional computing for transductive graph learning. For prediction accuracy \hdgc outperforms major and popular graph neural network implementations as well as state-of-the-art hyperdimensional computing implementations for a collection of homophilic graphs and heterophilic graphs. Compared with the most accurate learning methodologies we have tested, on the same target GPU platform, \hdgc is on average 9561.0 and 144.5 times faster than \gcnii, a graph neural network implementation and HDGL, a hyperdimensional computing implementation, respectively. As the majority of the learning operates on binary vectors, we expect outstanding energy performance of \hdgc on neuromorphic and emerging process-in-memory devices.

---

## 78. Exploring an image-based $b$-jet tagging method using convolution neural networks

**论文链接:** [http://arxiv.org/abs/2510.23962v1](http://arxiv.org/abs/2510.23962v1)

**作者:** Hangil Jang, Sanghoon Lim

**发布时间:** 2025-10-28

**DOI:** 10.1007/s40042-025-01506-3

**备注:** 23 pages, 17 figures

### GPT解析

### 总结

该研究开发了一种基于图像的喷流味道识别方法，利用主顶点周围的图像和喷流锥内的带电粒子，通过卷积神经网络进行分析，在b-喷流识别中实现了80-90%的效率，有望提高高能核物理实验的准确性。

### 背景

喷流味道识别（识别起源于c夸克、b夸克和其他夸克（轻夸克和胶子）的喷流）在高能重离子物理中至关重要，因为它能够研究重离子碰撞产生的热密核介质中的味道依赖响应。

### 目的

开发一种新的喷流味道识别方法，提高在高能核物理实验中的准确性。

### 方法

基于主顶点周围的图像，利用喷流锥内的带电粒子（可通过硅跟踪系统测量），使用卷积神经网络进行分析。研究假设跟踪系统具有理想性能。

### 主要发现

基于图像的味道识别方法在横向动量范围从20到100 GeV/c的喷流中，实现了80-90%的b-喷流识别效率。

### 结论

这种基于图像的喷流味道识别方法有潜力显著提高高能核物理实验中喷流味道识别的准确性。

### 翻译

喷流味道识别，即识别起源于c夸克、b夸克和其他夸克（轻夸克和胶子）的喷流，是高能重离子物理中的关键任务，因为它能够研究重离子碰撞产生的热密核介质中的味道依赖响应。最近，基于深度学习技术（如深度神经网络和图神经网络）的几种方法已被开发。这些基于深度学习的方法相比依赖轨迹影响参数和次级顶点的传统方法表现出显著改进的性能。在识别算法中，使用了喷流和组成带电粒子的各种属性作为输入参数。我们探索了一种基于主顶点周围图像的新方法，利用喷流锥内的带电粒子，这些粒子可以通过硅跟踪系统测量。对于这项初步实验研究，我们假设跟踪系统具有理想性能。为了分析这些图像，我们采用了卷积神经网络。基于图像的味道识别方法在横向动量范围从20到100 GeV/c的喷流中显示了80-90%的b-喷流识别效率。这种方法有潜力显著提高高能核物理实验中喷流味道识别的准确性。


### 论文摘要

Jet flavor tagging, the identification of jets originating from $c$-quarks, $b$-quarks, and other quarks (light quarks and gluons), is a crucial task in high-energy heavy-ion physics, as it enables the investigation of flavor-dependent responses within the hot and dense nuclear medium produced in heavy-ion collisions. Recently, several methods based on deep learning techniques, such as deep neural networks and graph neural networks, have been developed. These deep-learning-based methods demonstrate significantly improved performance compared to traditional methods that rely on track impact parameters and secondary vertices. In the tagging algorithms, various properties of jets and constituent charged particles are used as input parameters. We explore a new method based on images surrounding the primary vertex, utilizing charged particles within the jet cone, which can be measured using a silicon tracking system. For this initial experimental study, we assume the ideal performance of the tracking system. To analyze these images, we employed convolutional neural networks. The image-based flavor tagging method shows an 80-90% $b$-jet tagging efficiency for jets in the transverse momentum range from 20 to 100 GeV/$c$. This approach has the potential to significantly improve the accuracy of jet flavor tagging in high-energy nuclear physics experiments.

---

## 79. Graph Neural Network Assisted Genetic Algorithm for Structural Dynamic Response and Parameter Optimization

**论文链接:** [http://arxiv.org/abs/2510.22839v2](http://arxiv.org/abs/2510.22839v2)

**作者:** Sagnik Mukherjee

**发布时间:** 2025-10-26

**备注:** 13 pages, 8 figures

### GPT解析

### 总结

本研究提出了一种混合数据驱动框架，结合图神经网络(GNN)代理模型和遗传算法(GA)优化器，用于结构参数优化，以克服传统数值方法计算成本高的问题。

### 背景

优化结构参数(如质量m、刚度k和阻尼系数c)对于设计高效、有韧性和稳定的结构至关重要。传统的数值方法，如有限元法(FEM)和计算流体动力学(CFD)模拟，虽然能提供高精度结果，但在迭代优化任务中计算成本高昂，因为每次评估都需要为每个参数组合求解控制方程。

### 目的

开发一种计算效率更高的方法来优化结构参数，避免传统数值方法的计算负担，实现自动化和智能化的结构设计。

### 方法

提出混合数据驱动框架，结合图神经网络(GNN)代理模型和遗传算法(GA)优化器；使用GNN学习结构参数与动态位移响应之间的非线性映射；使用Newmark Beta方法生成单自由度(SDOF)系统响应数据集；GA通过最小化预测位移和提高动态稳定性来搜索全局最优参数集。

### 主要发现

GNN和GA框架实现了强收敛性和鲁棒泛化能力；与传统模拟相比，显著降低了计算成本；结合机器学习代理和进化优化的方法对于自动化和智能结构设计是有效的。

### 结论

所提出的方法克服了传统数值优化方法的计算效率问题，为结构参数优化提供了一种有效途径，展示了机器学习代理与进化优化结合在自动化和智能结构设计中的有效性。

### 翻译

结构参数(如质量m、刚度k和阻尼系数c)的优化对于设计高效、有韧性和稳定的结构至关重要。传统的数值方法，包括有限元法(FEM)和计算流体动力学(CFD)模拟，虽然能提供高精度结果，但在迭代优化任务中计算成本高昂，因为每次评估都需要为每个参数组合求解控制方程。本研究提出了一种混合数据驱动框架，结合图神经网络(GNN)代理模型和遗传算法(GA)优化器来克服这些挑战。GNN被训练以准确学习结构参数与动态位移响应之间的非线性映射，实现无需重复求解系统方程的快速预测。使用Newmark Beta方法生成单自由度(SDOF)系统响应数据集，涵盖多种质量、刚度和阻尼配置。然后，GA通过最小化预测位移和提高动态稳定性来搜索全局最优参数集。结果表明，与传统模拟相比，GNN和GA框架实现了强收敛性、鲁棒泛化能力和显著降低的计算成本。这种方法突显了将机器学习代理与进化优化相结合用于自动化和智能结构设计的有效性。


### 论文摘要

The optimization of structural parameters, such as mass(m), stiffness(k), and damping coefficient(c), is critical for designing efficient, resilient, and stable structures. Conventional numerical approaches, including Finite Element Method (FEM) and Computational Fluid Dynamics (CFD) simulations, provide high-fidelity results but are computationally expensive for iterative optimization tasks, as each evaluation requires solving the governing equations for every parameter combination. This study proposes a hybrid data-driven framework that integrates a Graph Neural Network (GNN) surrogate model with a Genetic Algorithm (GA) optimizer to overcome these challenges. The GNN is trained to accurately learn the nonlinear mapping between structural parameters and dynamic displacement responses, enabling rapid predictions without repeatedly solving the system equations. A dataset of single-degree-of-freedom (SDOF) system responses is generated using the Newmark Beta method across diverse mass, stiffness, and damping configurations. The GA then searches for globally optimal parameter sets by minimizing predicted displacements and enhancing dynamic stability. Results demonstrate that the GNN and GA framework achieves strong convergence, robust generalization, and significantly reduced computational cost compared to conventional simulations. This approach highlights the effectiveness of combining machine learning surrogates with evolutionary optimization for automated and intelligent structural design.

---

## 80. MIC-BEV: Multi-Infrastructure Camera Bird's-Eye-View Transformer with Relation-Aware Fusion for 3D Object Detection

**论文链接:** [http://arxiv.org/abs/2510.24688v1](http://arxiv.org/abs/2510.24688v1)

**作者:** Yun Zhang, Zhaoliang Zheng, Johnson Liu, Zhiyu Huang, Zewei Zhou, Zonglin Meng, Tianhui Cai, Jiaqi Ma

**发布时间:** 2025-10-28

### GPT解析

### 背景

基于基础设施的感知在智能交通系统中起着关键作用，提供全局态势感知并支持协作自动驾驶。然而，现有基于摄像头的检测模型在多视图基础设施设置、多样化摄像头配置、降级视觉输入和各种道路布局等场景下表现不佳。

### 目的

提出MIC-BEV，一种基于Transformer的鸟瞰图(BEV)感知框架，用于基于基础设施的多摄像头3D物体检测。

### 方法

MIC-BEV灵活支持具有异构内参和外参的变量摄像头数量，在传感器降级情况下表现出强大的鲁棒性。提出的图增强融合模块通过利用摄像头和BEV单元之间的几何关系以及潜在视觉线索，将多视图图像特征集成到BEV空间。同时引入M2I数据集，用于基于基础设施的物体检测，具有多样化的摄像头配置、道路布局和环境条件。

### 主要发现

在M2I和真实世界数据集RoScenes上的大量实验表明，MIC-BEV在3D物体检测方面实现了最先进的性能，在极端天气和传感器降级等具有挑战性的条件下保持鲁棒性。

### 结论

MIC-BEV的结果突显了其在现实世界部署的潜力，数据集和源代码可在GitHub链接获取。

### 翻译

基于基础设施的感知在智能交通系统中起着关键作用，提供全局态势感知并支持协作自动驾驶。然而，现有基于摄像头的检测模型在多视图基础设施设置、多样化摄像头配置、降级视觉输入和各种道路布局等场景下表现不佳。我们提出MIC-BEV，一种基于Transformer的鸟瞰图(BEV)感知框架，用于基于基础设施的多摄像头3D物体检测。MIC-BEV灵活支持具有异构内参和外参的变量摄像头数量，在传感器降级情况下表现出强大的鲁棒性。MIC-BEV中提出的图增强融合模块通过利用摄像头和BEV单元之间的几何关系以及潜在视觉线索，将多视图图像特征集成到BEV空间。为支持训练和评估，我们引入M2I数据集，用于基于基础设施的物体检测，具有多样化的摄像头配置、道路布局和环境条件。在M2I和真实世界数据集RoScenes上的大量实验表明，MIC-BEV在3D物体检测方面实现了最先进的性能，在极端天气和传感器降级等具有挑战性的条件下保持鲁棒性。这些结果突显了MIC-BEV在现实世界部署的潜力。数据集和源代码可在以下链接获取：https://github.com/HandsomeYun/MIC-BEV。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决基于基础设施的多摄像头3D目标检测问题，特别是在多视图基础设施设置、多样化摄像头配置、退化视觉输入和复杂道路布局等挑战场景下的性能提升。这个问题在现实中很重要，因为基于基础设施的感知是智能交通系统的关键组成部分，能提供全局态势感知和协同自主能力；同时，相比昂贵的激光雷达方案，摄像头更实惠、可扩展且提供丰富的语义信息，但现有摄像头检测模型在这些复杂场景中表现不佳。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先分析了基础设施感知与车载感知的区别，指出基础设施摄像头空间分布广泛，具有异构参数，这给现有BEV方法带来挑战。他们设计MIC-BEV框架时借鉴了现有BEV感知方法（如BEVFormer）的Transformer架构，但针对基础设施场景进行了创新。具体来说，他们引入了图增强融合模块，利用摄像头和BEV单元间的几何关系进行特征融合；借鉴了可变形注意力机制进行特征提取；采用DETR风格解码器进行目标检测；并使用图注意力网络(GAT)来学习融合权重，使模型能够自适应地处理不同摄像头配置。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是通过图神经网络建模摄像头和BEV单元之间的几何关系，实现关系感知的多视图特征融合，使模型能够根据每个摄像头与BEV单元的几何相关性动态分配权重。整体流程包括：1)处理可变数量的摄像头输入；2)使用ResNet-101和FPN提取多尺度特征；3)初始化可学习的BEV查询；4)通过Transformer编码进行特征处理，包括时间自注意力和关系增强空间交叉注意力(ReSCA)；5)使用任务特定解码器进行3D目标检测和BEV语义分割。其中ReSCA模块是关键，它为每个BEV查询生成3D参考点，投影到摄像头视图，使用可变形注意力提取特征，并通过GAT学习融合权重。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)MIC-BEV框架，专门设计用于处理基础设施多摄像头系统的异构配置；2)关系增强的空间交叉注意力(ReSCA)，利用图神经网络建模几何关系；3)M2I数据集，提供多样化的摄像头配置、道路布局和环境条件；4)摄像头掩码策略，提高对传感器降级的鲁棒性。相比之前工作，MIC-BEV能适应不同数量、方向、高度和视场的摄像头；使用显式的关系建模而非隐式融合；考虑摄像头和BEV单元间的几何关系；通过多任务学习（3D检测和BEV分割）增强空间理解；并在训练中模拟传感器故障以提高鲁棒性。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'MIC-BEV通过引入关系感知的图增强融合机制和M2I多样化数据集，显著提升了基础设施多摄像头系统在复杂场景下的3D目标检测性能和鲁棒性。'}


### 论文摘要

Infrastructure-based perception plays a crucial role in intelligent transportation systems, offering global situational awareness and enabling cooperative autonomy. However, existing camera-based detection models often underperform in such scenarios due to challenges such as multi-view infrastructure setup, diverse camera configurations, degraded visual inputs, and various road layouts. We introduce MIC-BEV, a Transformer-based bird's-eye-view (BEV) perception framework for infrastructure-based multi-camera 3D object detection. MIC-BEV flexibly supports a variable number of cameras with heterogeneous intrinsic and extrinsic parameters and demonstrates strong robustness under sensor degradation. The proposed graph-enhanced fusion module in MIC-BEV integrates multi-view image features into the BEV space by exploiting geometric relationships between cameras and BEV cells alongside latent visual cues. To support training and evaluation, we introduce M2I, a synthetic dataset for infrastructure-based object detection, featuring diverse camera configurations, road layouts, and environmental conditions. Extensive experiments on both M2I and the real-world dataset RoScenes demonstrate that MIC-BEV achieves state-of-the-art performance in 3D object detection. It also remains robust under challenging conditions, including extreme weather and sensor degradation. These results highlight the potential of MIC-BEV for real-world deployment. The dataset and source code are available at: https://github.com/HandsomeYun/MIC-BEV.

---

## 81. Optimizing Retrieval for RAG via Reinforced Contrastive Learning

**论文链接:** [http://arxiv.org/abs/2510.24652v1](http://arxiv.org/abs/2510.24652v1)

**作者:** Jiawei Zhou, Lei Chen

**发布时间:** 2025-10-28

### GPT解析

### 总结

R3是一种通过试验和反馈强化对比学习为RAG优化的检索框架，能够在RAG环境中动态探索和优化相关性，实验表明其性能优于原始检索器和最先进检索器，且高效实用。

### 背景

随着检索增强生成(RAG)的普及，信息检索(IR)的角色正从为人类用户检索信息转变为为AI系统检索上下文知识，这使得相关性难以预先定义或标注。

### 目的

提出一种解决方案，使检索器能够在RAG环境中动态探索和优化相关性，无需依赖预先标注的数据。

### 方法

R3框架通过检索结果与环境交互产生对比信号，自动引导检索器的自我改进，不同于依赖标注或合成数据进行监督微调的先前方法。

### 主要发现

实验表明R3比原始检索器提高RAG性能5.2%，超越最先进检索器4.9%，同时与LLM增强检索和基于后训练或指令调整LLM的RAG系统性能相当。

### 结论

R3是一种高效实用的解决方案，解决了在RAG环境中定义和优化相关性的挑战，只需4个GPU一天内完成训练。

### 翻译

随着检索增强生成(RAG)变得越来越普遍，信息检索(IR)的角色正从为人类用户检索信息转变为为人工智能(AI)系统检索上下文知识，这使得相关性变得难以预先定义或标注。为应对这一挑战，我们提出了R3，一种通过试验和反馈强化对比学习为RAG优化的检索框架。与依赖标注或合成数据进行监督微调的先前方法不同，R3使检索器能够在RAG环境中动态探索和优化相关性。在训练过程中，检索结果与环境交互以产生对比信号，自动引导检索器的自我改进。在各种不同任务上的大量实验表明，R3比原始检索器提高RAG性能5.2%，超越最先进检索器4.9%，同时与LLM增强检索和基于后训练或指令调整LLM构建的RAG系统相当。R3既高效又实用，只需4个GPU，并在一天内完成训练。


### 论文摘要

As retrieval-augmented generation (RAG) becomes increasingly widespread, the role of information retrieval (IR) is shifting from retrieving information for human users to retrieving contextual knowledge for artificial intelligence (AI) systems, where relevance becomes difficult to define or annotate beforehand. To address this challenge, we propose R3, a Retrieval framework optimized for RAG through trialand-feedback Reinforced contrastive learning. Unlike prior approaches that rely on annotated or synthetic data for supervised fine-tuning, R3 enables the retriever to dynamically explore and optimize relevance within the RAG environment. During training, the retrieved results interact with the environment to produce contrastive signals that automatically guide the retriever's self-improvement. Extensive experiments across diverse tasks demonstrate that R3 improves RAG performance by 5.2% over the original retriever and surpasses state-of-the-art retrievers by 4.9%, while achieving comparable results to LLM-augmented retrieval and RAG systems built on post-trained or instruction-tuned LLMs. It is both efficient and practical, requiring only 4 GPUs and completing training within a single day.

---

## 82. DeshadowMamba: Deshadowing as 1D Sequential Similarity

**论文链接:** [http://arxiv.org/abs/2510.24260v1](http://arxiv.org/abs/2510.24260v1)

**作者:** Zhaotong Yang, Yi Chen, Yanying Li, Shengfeng He, Yangyang Xu, Junyu Dong, Jian Yang, Yong Du

**发布时间:** 2025-10-28

### GPT解析

### 总结

本研究提出了一种基于Mamba模型的图像阴影去除方法，通过引入CrossGate方向调制机制和ColorShift正则化技术，有效解决了现有方法中阴影结构扭曲和颜色不一致的问题，实现了高质量的阴影去除效果。

### 背景

当前基于深度学习的图像阴影去除方法通常依赖于基于注意力的架构来捕获长距离依赖关系，但这些固定的注意模式往往会混合来自不相关区域的照明线索，导致结构扭曲和颜色不一致。

### 目的

重新审视阴影去除问题，从序列建模的角度探索更有效的解决方案，解决现有方法中阴影结构扭曲和颜色不一致的问题。

### 方法

1. 从序列建模角度探索使用Mamba（选择性状态空间模型）来传播全局上下文；2. 提出CrossGate方向调制机制，将阴影感知相似性注入Mamba的输入门；3. 引入ColorShift正则化，通过合成结构化的信息负样本引导模型抑制颜色污染并实现稳健的颜色恢复。

### 主要发现

1. Mamba模型能通过方向状态转换实现有效的全局感受野同时保持位置连续性；2. 直接将Mamba应用于图像数据缺乏阴影-非阴影语义意识且易受颜色干扰；3. 所提出的CrossGate和ColorShift正则化技术能有效解决这些问题；4. DeshadowMamba在公共基准测试上实现了最先进的视觉质量和强大的定量性能。

### 结论

通过将序列建模适应于阴影去除所需的完整结构和色度一致性，所提出的方法DeshadowMamba在图像阴影去除任务中取得了显著效果，为解决阴影去除中的结构完整性和色度一致性问题提供了有效途径。

### 翻译

最近的图像阴影去除深度模型通常依赖于基于注意力的架构来捕获长距离依赖关系。然而，它们的固定注意模式往往会混合来自不相关区域的照明线索，导致结构扭曲和颜色不一致。在这项工作中，我们从序列建模的角度重新审视阴影去除问题，并探索使用Mamba（一种选择性状态空间模型）来通过方向状态转换传播全局上下文。这些转换产生有效的全局感受野，同时保持位置连续性。尽管有潜力，但直接将Mamba应用于图像数据并非最佳选择，因为它缺乏阴影-非阴影语义意识，并且仍然容易受到附近区域颜色干扰的干扰。为了解决这些局限性，我们提出了CrossGate，一种方向调制机制，将阴影感知相似性注入Mamba的输入门，允许沿过渡轴选择性地集成相关上下文。为了进一步确保外观保真度，我们引入了ColorShift正则化，这是一种由全局颜色统计驱动的对比学习目标。通过合成结构化的信息负样本，它引导模型抑制颜色污染并实现稳健的颜色恢复。这些组件共同将序列建模适应于阴影去除所需的完整结构和色度一致性。在公共基准测试上的大量实验表明，DeshadowMamba实现了最先进的视觉质量和强大的定量性能。


### 论文摘要

Recent deep models for image shadow removal often rely on attention-based architectures to capture long-range dependencies. However, their fixed attention patterns tend to mix illumination cues from irrelevant regions, leading to distorted structures and inconsistent colors. In this work, we revisit shadow removal from a sequence modeling perspective and explore the use of Mamba, a selective state space model that propagates global context through directional state transitions. These transitions yield an efficient global receptive field while preserving positional continuity. Despite its potential, directly applying Mamba to image data is suboptimal, since it lacks awareness of shadow-non-shadow semantics and remains susceptible to color interference from nearby regions. To address these limitations, we propose CrossGate, a directional modulation mechanism that injects shadow-aware similarity into Mamba's input gate, allowing selective integration of relevant context along transition axes. To further ensure appearance fidelity, we introduce ColorShift regularization, a contrastive learning objective driven by global color statistics. By synthesizing structured informative negatives, it guides the model to suppress color contamination and achieve robust color restoration. Together, these components adapt sequence modeling to the structural integrity and chromatic consistency required for shadow removal. Extensive experiments on public benchmarks demonstrate that DeshadowMamba achieves state-of-the-art visual quality and strong quantitative performance.

---

## 83. MATCH: Task-Driven Code Evaluation through Contrastive Learning

**论文链接:** [http://arxiv.org/abs/2510.23169v2](http://arxiv.org/abs/2510.23169v2)

**作者:** Marah Ghoummaid, Vladimir Tchuiev, Ofek Glick, Michal Moshkovitz, Dotan Di Castro

**发布时间:** 2025-10-27

### GPT解析

### 总结

本文介绍了MATCH，一种新颖的无参考代码评估指标，用于解决AI生成代码与开发者意图匹配度评估的挑战。

### 背景

AI代码生成越来越普遍，GitHub Copilot估计生成了GitHub上46%的代码。准确评估生成代码与开发者意图的匹配程度仍然是一个重大挑战。

### 目的

为了解决无参考代码评估的空白问题，除了ICE-Score等少数替代方案外，引入MATCH作为一种新颖的无参考代码指标。

### 方法

MATCH使用对比学习为代码和自然语言任务描述生成有意义的嵌入，实现反映生成代码执行任务程度的相似性评分。

### 主要发现

MATCH在多种编程语言上，与功能正确性和人类偏好相比，比现有指标实现了更强的相关性。

### 结论

MATCH是一种有效的无参考代码评估指标，能够更好地评估生成代码与开发者意图的匹配程度。

### 翻译

基于AI的代码生成日益普及，GitHub Copilot估计生成了GitHub上46%的代码。准确评估生成代码与开发者意图的匹配程度仍然是一个重大挑战。传统评估方法，如单元测试，通常难以扩展且成本高昂。语法相似性指标（如BLEU、ROUGE）无法捕捉代码功能，而像CodeBERTScore这样的指标需要参考代码，但参考代码并不总是可用的。为了解决无参考代码评估的空白问题，除了ICE-Score等少数替代方案外，本文引入了MATCH，一种新颖的无参考代码指标。MATCH使用对比学习为代码和自然语言任务描述生成有意义的嵌入，实现反映生成代码执行任务程度的相似性评分。我们表明，MATCH在多种编程语言上，与功能正确性和人类偏好相比，比现有指标实现了更强的相关性。


### 论文摘要

AI-based code generation is increasingly prevalent, with GitHub Copilot estimated to generate 46% of the code on GitHub. Accurately evaluating how well generated code aligns with developer intent remains a critical challenge. Traditional evaluation methods, such as unit tests, are often unscalable and costly. Syntactic similarity metrics (e.g., BLEU, ROUGE) fail to capture code functionality, and metrics like CodeBERTScore require reference code, which is not always available. To address the gap in reference-free evaluation, with few alternatives such as ICE-Score, this paper introduces MATCH, a novel reference-free metric. MATCH uses Contrastive Learning to generate meaningful embeddings for code and natural language task descriptions, enabling similarity scoring that reflects how well generated code implements the task. We show that MATCH achieves stronger correlations with functional correctness and human preference than existing metrics across multiple programming languages.

---

## 84. UniField: Joint Multi-Domain Training for Universal Surface Pressure Modeling

**论文链接:** [http://arxiv.org/abs/2510.24106v1](http://arxiv.org/abs/2510.24106v1)

**作者:** Junhong Zou, Zhenxu Sun, Yueqing Wang, Wei Qiu, Zhaoxiang Zhang, Zhen Lei, Xiangyu Zhu

**发布时间:** 2025-10-28

### GPT解析

### 总结

该研究提出了一种名为UniField的方法，通过整合多个子领域的空气动力学数据进行联合训练，解决了数据稀缺问题，实现了更好的流场表示。

### 背景

物体表面压力场的空气动力学模拟对许多工程问题至关重要。深度神经网络已成为传统计算流体力学(CFD)模拟的高效替代方案，但数据稀缺限制了其应用。

### 目的

解决空气动力学数据稀缺问题，通过整合多个子领域的数据进行联合训练，学习更通用的流场表示。

### 方法

提出UniField方法，整合五个不同数据集（涵盖汽车、火车、飞机和一般形状）。该方法采用领域无关的Transformer模块提取通用点云特征，并定制领域特定的流动条件适配器来适应不同子领域的流动信息。

### 主要发现

尽管不同子领域的空气动力学数据通常遵循不同方程，但联合训练的模型通常比单独训练的模型表现更好，表明这些数据相互补充，帮助模型学习更好的流场表示。

### 结论

UniField作为通用流场表示模型具有潜力，为神经网络在空气动力学分析中的更广泛应用奠定了基础。

### 翻译

物体表面压力场的空气动力学模拟对许多工程问题至关重要。近年来，深度神经网络已成为传统计算成本高昂的CFD模拟的高效替代方案，用于建模表面压力场。然而，数据稀缺仍然是一个基本挑战，限制了神经网络的应用。为了解决这一限制，我们提出整合多个子领域的空气动力学数据进行联合训练，以学习更通用的流场表示。我们整合了涵盖不同领域的五个不同数据集，包括汽车、火车、飞机和一般形状。面对不同领域间的显著数据差异，我们提出了UniField，它采用领域无关的Transformer模块提取通用点云特征，并定制领域特定的流动条件适配器来适应不同子领域的流动信息。尽管不同子领域的空气动力学数据通常遵循不同的方程，但我们比较了在所有数据上联合训练的模型与在单个数据集上单独训练的模型，发现联合训练的模型通常表现出更好的性能。这表明这些数据相互补充，帮助模型学习更好的流场表示。这些结果突显了UniField作为通用流场表示模型的潜力，为神经网络在空气动力学分析中的更广泛应用奠定了基础。


### 论文摘要

Aerodynamic simulation of the surface pressure field around objects is crucial for many engineering problems. In recent years, deep neural networks have emerged as an efficient alternative to traditional, computationally expensive CFD simulations for modeling surface pressure fields. However, data scarcity remains a fundamental challenge, limiting the application of neural networks. To address this limitation, we propose to integrate aerodynamic data from multiple subfields and conduct joint training to learn more general field representations. We consolidate five different datasets covering various fields, including automobiles, trains, aircraft, and general shapes. Facing significant data differences across different domains, we propose UniField, which employs a domain-agnostic Transformer module to extract general point cloud features and customizes domain-specific flow-conditioned adapters to adapt to the flow information in different subfields. Despite the fact that aerodynamic data from different subfields are typically governed by different equations, we compare models trained jointly on all data with those trained separately on individual datasets and find that the jointly-trained model commonly demonstrates better performance. This indicates that these data complement each other to help the model learn better flow field representations. These results highlight the potential of UniField as a universal flow field representation model and lay the foundation for broader applications of neural networks in aerodynamic analysis.

---

## 85. DPGLA: Bridging the Gap between Synthetic and Real Data for Unsupervised Domain Adaptation in 3D LiDAR Semantic Segmentation

**论文链接:** [http://arxiv.org/abs/2510.23525v1](http://arxiv.org/abs/2510.23525v1)

**作者:** Wanmeng Li, Simone Mosco, Daniel Fusaro, Alberto Pretto

**发布时间:** 2025-10-27

**备注:** This paper has been accepted for publication at the 2025 IEEE/RSJ  International Conference on Intelligent Robots and Systems (IROS)

### GPT解析

### 总结

本文提出了一种动态伪标签过滤(DPLF)方案和先导引导的数据增强流水线(PG-DAP)，用于增强点云无监督域适应语义分割中真实数据的利用，提高合成到真实点云语义分割的性能。

### 背景

为智能自主系统标注真实世界的LiDAR点云成本很高，现有基于自训练的无监督域适应方法在利用合成点云数据时未能有效利用未标记数据。

### 目的

提高点云语义分割性能，更有效地利用未标记数据，克服现有方法依赖预定义或固定置信度阈值导致的性能限制。

### 方法

提出动态伪标签过滤(DPLF)方案增强真实数据利用，设计先导引导的数据增强流水线(PG-DAP)减轻域偏移，并使用数据混合一致性损失推动模型学习上下文无关表示。

### 主要发现

在两个具有挑战性的合成到真实点云语义分割任务上，该方法取得了优越的性能，消融研究证实了DPLF和PG-DAP模块的有效性。

### 结论

所提出的方法能够有效解决现有点云UDA语义分割中未标记数据利用不足的问题，显著提高了合成到真实点云语义分割的性能。

### 翻译

为智能自主系统使用而标注真实世界的LiDAR点云成本很高。为克服这一限制，基于自训练的无监督域适应(UDA)已被广泛用于利用合成点云数据提高点云语义分割。然而，我们认为现有方法没有有效利用未标记数据，因为它们要么依赖于预定义或固定的置信度阈值，导致性能不佳。在本文中，我们提出了一种动态伪标签过滤(DPLF)方案，以增强点云UDA语义分割中真实数据的利用。此外，我们设计了一个简单高效的先导引导的数据增强流水线(PG-DAP)，以减轻合成和真实世界点云之间的域偏移。最后，我们使用数据混合一致性损失来推动模型学习上下文无关的表示。我们通过最先进方法的广泛比较实施并彻底评估了我们的方法。在两个具有挑战性的合成到真实点云语义分割任务上的实验表明，我们的方法取得了优越的性能。消融研究证实了DPLF和PG-DAP模块的有效性。我们在本文中发布了我们方法的代码。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决3D激光雷达点云语义分割中的无监督域适应问题，具体是如何有效利用带有自动标签的合成数据来改善对真实世界点云的语义分割，而无需对真实世界数据进行昂贵的标注。这个问题很重要，因为对真实世界的激光雷达点云进行密集标注成本高昂且耗时，而智能自主系统（如自动驾驶汽车）需要在动态真实环境中工作。虽然合成数据可以自动生成标签，但合成环境和真实世界之间存在域分布差异，导致在合成数据上训练的模型在真实数据上性能下降。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先分析了现有方法的局限性，特别是发现固定置信度阈值导致真实数据利用效率低下和类别不平衡问题。然后设计了解决方案：1) 动态伪标签过滤(DPLF)方案，自适应调整置信度阈值；2) 先验引导的数据增强管道(PG-DAP)，缓解域差异；3) 数据混合一致性损失，学习上下文无关表示。作者借鉴了Mean Teacher模型架构、LaserMix数据混合框架、局部和全局仿射变换以及指数移动平均(EMA)更新机制，但进行了创新性改进以解决特定问题。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是通过动态调整伪标签过滤策略和利用先验知识进行高效数据增强，解决合成到真实数据的域适应问题。整体流程：1) 使用Mean Teacher架构，预训练教师网络；2) 教师网络生成目标域伪标签；3) 通过DPLF进行伪标签过滤(距离加权、分层过滤、动态阈值更新)；4) 使用PG-DAP进行数据增强(DAS、DAJ、HAJ)；5) 应用LaserMix进行数据混合；6) 计算分割损失和数据混合一致性损失；7) 更新学生网络参数，使用EMA更新教师网络参数。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点：1) 动态伪标签过滤(DPLF)：自适应调整全局和类别特定置信度阈值，使用距离权重和EMA动态更新，解决固定阈值导致的类别不平衡；2) 先验引导数据增强(PG-DAP)：包含DAS(密度感知采样)、DAJ(距离感知抖动)和HAJ(高度感知抖动)，基于先验知识无需额外学习；3) 数据混合一致性损失：推动模型学习上下文无关表示。相比之前工作，不同之处在于：不使用固定置信度阈值，避免类别不平衡；不依赖计算资源昂贵的GAN进行域转换；结合了动态伪标签过滤和高效数据增强，通过一致性损失进一步改善特征表示。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'DPGLA通过动态伪标签过滤和先验引导数据增强，有效解决了3D激光雷达点云语义分割中合成到真实数据的无监督域适应问题，显著提升了模型在真实场景中的分割性能。'}


### 论文摘要

Annotating real-world LiDAR point clouds for use in intelligent autonomous systems is costly. To overcome this limitation, self-training-based Unsupervised Domain Adaptation (UDA) has been widely used to improve point cloud semantic segmentation by leveraging synthetic point cloud data. However, we argue that existing methods do not effectively utilize unlabeled data, as they either rely on predefined or fixed confidence thresholds, resulting in suboptimal performance. In this paper, we propose a Dynamic Pseudo-Label Filtering (DPLF) scheme to enhance real data utilization in point cloud UDA semantic segmentation. Additionally, we design a simple and efficient Prior-Guided Data Augmentation Pipeline (PG-DAP) to mitigate domain shift between synthetic and real-world point clouds. Finally, we utilize data mixing consistency loss to push the model to learn context-free representations. We implement and thoroughly evaluate our approach through extensive comparisons with state-of-the-art methods. Experiments on two challenging synthetic-to-real point cloud semantic segmentation tasks demonstrate that our approach achieves superior performance. Ablation studies confirm the effectiveness of the DPLF and PG-DAP modules. We release the code of our method in this paper.

---

## 86. Quality-controlled registration of urban MLS point clouds reducing drift effects by adaptive fragmentation

**论文链接:** [http://arxiv.org/abs/2510.23416v1](http://arxiv.org/abs/2510.23416v1)

**作者:** Marco Antonio Ortiz Rincon, Yihui Yang, Christoph Holst

**发布时间:** 2025-10-27

**备注:** 10 pages, 7 figures. This manuscript is currently under review at the  International Journal of Applied Earth Observation and Geoinformation  (Elsevier). A preprint version will also be available on SSRN (Elsevier  Preprints) with a DOI once processed. This is the original preprint version  submitted for peer review

### GPT解析

### 总结

本研究提出了一种新型工作流程，用于高效准确地在大规模城市街道场景中将移动激光扫描(MLS)点云配准到目标模型点云。

### 背景

城市环境中的点云配准面临复杂挑战，包括点云密度差异、噪声特性和遮挡场景，这些在城市中心尤为常见。

### 目的

设计一种能够应对城市环境复杂性的工作流程，实现大规模MLS点云与目标模型点云的高效准确配准。

### 方法

提出两种方法创新：1) 半球检查(SSC)预处理技术，通过识别相互正交的平面表面分割MLS轨迹数据，减少MLS漂移影响；2) 平面体素广义最近点迭代算法(PV-GICP)，在体素分区中选择性使用平面表面进行精细配准。

### 主要发现

慕尼黑市中心真实数据集实验表明，该工作流程实现了平均亚0.01米的配准精度，同时比传统点对平面ICP方法减少50%以上的计算时间。

### 结论

所提出的方法能够推动自动化三维城市建模和更新，在城市规划、基础设施管理和动态城市监测中有直接应用。

### 翻译

本研究提出了一种新型工作流程，旨在高效准确地在大规模移动激光扫描(MLS)点云与城市街道场景中的目标模型点云之间进行配准。该工作流程专门针对城市环境中固有的复杂性，巧妙地解决了点云密度、噪声特性和遮挡场景差异带来的挑战，这些挑战在城市中心中普遍存在。研究引入了两种方法创新。首先，提出的半球检查(SSC)预处理技术通过识别相互正交的平面表面，最优地分割MLS轨迹数据。这一步骤减少了MLS漂移对整个点云配准精度的影响，同时确保每个片段内有足够的几何特征以避免局部最小值。其次，我们提出了平面体素广义最近点迭代算法(PV-GICP)，一种在体素分区中选择性使用平面表面的精细配准方法。这种预处理策略不仅提高了配准精度，而且比传统的点对平面ICP方法减少了50%以上的计算时间。在慕尼黑市中心的真实数据集实验中，我们的工作流程实现了亚0.01米的平均配准精度，同时显著缩短了处理时间。研究结果强调了所提出方法在推进自动化三维城市建模和更新方面的潜力，可直接应用于城市规划、基础设施管理和动态城市监测。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决城市环境中大规模移动激光扫描(MLS)点云与目标模型点云的高效准确配准问题，特别是处理城市环境中的复杂性和挑战，包括不同密度、噪声特性和遮挡场景的点云集成，以及MLS长时间扫描过程中产生的漂移效应。这个问题在现实中非常重要，因为城市环境变化迅速，需要频繁更新3D城市模型用于城市规划、基础设施管理和城市发展监测等应用，而传统更新方法劳动密集且容易出错。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先识别了现有方法的局限性，包括传统点云配准技术在处理大规模城市数据时效率低下，单个变换矩阵可能不足以实现精确对齐，以及固定分割方法在某些缺乏特征的片段中可能失败。在此基础上，作者借鉴了现有的点云分割技术(如等时间间隔分割)、特征匹配方法(如RANSAC)和ICP算法，但进行了创新改进，提出了自适应分割策略确保每个片段包含足够几何特征，以及专门针对城市环境的配准流程。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是通过自适应分割将大型MLS点云分成包含足够几何特征的小片段减少漂移影响，只在稳定的平面区域进行精细配提高效率和准确性，并利用变换参数分析MLS漂移效应。整体流程包括：1)数据预处理(重采样、去噪、语义分类)；2)数据分割(初始分割后通过半球检查验证)；3)粗配准(特征检测、匹配和异常值去除)；4)精细配准(识别平面区域并执行GICP)；5)漂移分析(评估变换参数)。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)半球检查(SSC)自适应分割技术，确保每个片段包含足够的相互正交平面表面；2)基于平面体素的广义ICP(PV-GICP)，选择性使用平面区域提高配准精度并减少50%以上的计算时间；3)漂移分析策略，通过分割过程识别和减少漂移误差。相比之前的工作，传统分割方法使用固定时间间隔可能导致某些片段缺乏特征，传统ICP方法在整个点云上运行计算量大且对非刚性区域敏感，而本文方法通过自适应分割和选择性平面配准解决了这些问题。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '该论文提出了一种基于自适应分割和选择性平面配准的城市MLS点云高质量配准方法，显著提高了配准精度并减少了计算时间，同时有效量化了MLS漂移效应，为自动化3D城市模型更新提供了实用解决方案。'}


### 论文摘要

This study presents a novel workflow designed to efficiently and accurately register large-scale mobile laser scanning (MLS) point clouds to a target model point cloud in urban street scenarios. This workflow specifically targets the complexities inherent in urban environments and adeptly addresses the challenges of integrating point clouds that vary in density, noise characteristics, and occlusion scenarios, which are common in bustling city centers. Two methodological advancements are introduced. First, the proposed Semi-sphere Check (SSC) preprocessing technique optimally fragments MLS trajectory data by identifying mutually orthogonal planar surfaces. This step reduces the impact of MLS drift on the accuracy of the entire point cloud registration, while ensuring sufficient geometric features within each fragment to avoid local minima. Second, we propose Planar Voxel-based Generalized Iterative Closest Point (PV-GICP), a fine registration method that selectively utilizes planar surfaces within voxel partitions. This pre-process strategy not only improves registration accuracy but also reduces computation time by more than 50% compared to conventional point-to-plane ICP methods. Experiments on real-world datasets from Munich's inner city demonstrate that our workflow achieves sub-0.01 m average registration accuracy while significantly shortening processing times. The results underscore the potential of the proposed methods to advance automated 3D urban modeling and updating, with direct applications in urban planning, infrastructure management, and dynamic city monitoring.

---

## 87. Symmetria: A Synthetic Dataset for Learning in Point Clouds

**论文链接:** [http://arxiv.org/abs/2510.23414v1](http://arxiv.org/abs/2510.23414v1)

**作者:** Ivan Sipiran, Gustavo Santelices, Lucas Oyarzún, Andrea Ranieri, Chiara Romanengo, Silvia Biasotti, Bianca Falcidieno

**发布时间:** 2025-10-27

**备注:** 40 pages

### GPT解析

### 总结

Symmetria是一种公式驱动的点云数据集，通过利用对称性概念克服了点云学习中数据稀缺的问题，能够按任意规模生成，提供精确的地面真值，促进数据高效实验，并支持广泛的泛化和扩展。

### 背景

与图像或文本领域受益于大量大型数据集不同，点云学习技术经常由于缺乏大规模数据集而遇到限制，这成为研究中的一个主要挑战。

### 目的

克服点云数据集稀缺的限制，提出一种可按任意规模生成的公式驱动数据集，为点云学习提供充足且高质量的数据支持。

### 方法

利用对称性概念创建具有已知结构和高度可变性的形状，确保精确地面真值的绝对可用性，设计数据集以促进数据高效的实验，实现跨不同几何设置的广泛泛化，并为新任务和模态提供易于扩展性。

### 主要发现

该数据集对点云自监督预训练非常有效，训练的模型在分类和分割等下游任务中表现出强大的性能，同时也显示出良好的少样本学习能力；该数据集还可用于真实世界物体的分类，展示了方法的实用价值；作者还引入了一个具有挑战性的对称检测任务，并为基线比较提供了基准。

### 结论

Symmetria数据集和相关代码的公开可用性，以及能够生成非常大的数据集集合的能力，为点云学习领域的进一步研究和创新提供了重要基础。

### 翻译

与图像或文本领域受益于大量大型数据集不同，点云学习技术经常由于缺乏大规模数据集而遇到限制。为了克服这一限制，我们提出了Symmetria，一种可按任意规模生成的公式驱动数据集。通过构造，它确保精确地面真值的绝对可用性，通过需要更少的样本促进数据高效的实验，实现跨不同几何设置的广泛泛化，并为新任务和模态提供易于扩展性。利用对称性的概念，我们创建了具有已知结构和高度可变性的形状，使神经网络能够有效地学习点云特征。我们的结果表明，该数据集对于点云自监督预训练非常有效，产生的模型在分类和分割等下游任务中表现出强大的性能，同时也显示出良好的少样本学习能力。此外，我们的数据集可以支持将模型微调以分类真实世界物体，突显了我们方法的实用性和应用价值。我们还引入了一个具有挑战性的对称检测任务，并为基线比较提供了基准。我们方法的一个显著优势是数据集、配套代码的公开可用性，以及生成非常大集合的能力，促进了点云学习的进一步研究和创新。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决点云学习领域中的数据稀缺问题。与图像或文本领域有大量大规模数据集不同，点云学习技术经常因缺乏大规模数据集而受到限制。这个问题很重要，因为点云是3D视觉的重要表示形式，广泛应用于机器人、自动驾驶等领域；缺乏大规模数据集限制了点云深度学习模型的发展；现有的3D数据集存在版权、隐私和标注成本等问题；真实世界的3D数据获取成本高、耗时长，难以满足机器学习所需的规模。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者观察到图像领域有公式驱动的数据集表现出了有趣性能，甚至在某些任务上超过了ImageNet，这启发了他们在3D点云领域采用类似方法。他们借鉴了SHREC23数据集的工作，但进行了扩展和改进。作者基于对称性概念设计数据集，因为几乎现实世界中的每个物体都表现出某种对称性。他们设计数据集时遵循了几个原则：大规模探索、数据效率、可用的真实标签、隐私保护和通用可扩展性。从平面参数曲线开始生成3D形状，这些曲线具有已知的几何特性，然后通过挤压或旋转操作将它们转换为3D表面。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是利用对称性作为生成合成点云数据集的基础，通过参数化平面曲线生成具有已知对称结构的形状，然后通过几何变换转换为3D表面，并添加各种扰动增加数据集的多样性和挑战性。整体实现流程包括：1)从具有对称性的参数化平面曲线开始；2)通过挤压（大多数曲线）或旋转（贝塞尔曲线）将曲线转换为3D表面；3)确保3D形状保持原始平面曲线的对称性；4)应用各种变换增加数据集多样性；5)以一定概率应用随机平移和/或旋转；6)将生成的点云组织成不同复杂度的子数据集；7)为每个点云提供其对称性的真实标签信息。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)基于对称性的合成数据集；2)公式驱动的生成方法；3)精确的真实标签；4)数据效率；5)多样化的几何变换；6)可扩展性；7)多任务支持。相比之前的工作（如ShapeNet、ModelNet、SHREC23等）的不同之处：1)数据来源不同，Symmetria是完全合成的，避免了版权和隐私问题；2)生成方式不同，使用了更丰富的参数化平面曲线库；3)数据规模可按需生成任意规模；4)提供了更全面的真实标签注释；5)不仅验证了在传统任务上的有效性，还专门验证了对称性检测这一特定任务上的性能。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '这篇论文提出了Symmetria，一个基于对称性的大规模合成点云数据集，通过程序化生成方法解决了3D点云学习中的数据稀缺问题，同时提供了精确的真实标签和多样化的几何变换，有效支持了自监督预训练和对称性检测等任务。'}


### 论文摘要

Unlike image or text domains that benefit from an abundance of large-scale datasets, point cloud learning techniques frequently encounter limitations due to the scarcity of extensive datasets. To overcome this limitation, we present Symmetria, a formula-driven dataset that can be generated at any arbitrary scale. By construction, it ensures the absolute availability of precise ground truth, promotes data-efficient experimentation by requiring fewer samples, enables broad generalization across diverse geometric settings, and offers easy extensibility to new tasks and modalities. Using the concept of symmetry, we create shapes with known structure and high variability, enabling neural networks to learn point cloud features effectively. Our results demonstrate that this dataset is highly effective for point cloud self-supervised pre-training, yielding models with strong performance in downstream tasks such as classification and segmentation, which also show good few-shot learning capabilities. Additionally, our dataset can support fine-tuning models to classify real-world objects, highlighting our approach's practical utility and application. We also introduce a challenging task for symmetry detection and provide a benchmark for baseline comparisons. A significant advantage of our approach is the public availability of the dataset, the accompanying code, and the ability to generate very large collections, promoting further research and innovation in point cloud learning.

---

## 88. Workspace Registration and Collision Detection for Industrial Robotics Applications

**论文链接:** [http://arxiv.org/abs/2510.23227v1](http://arxiv.org/abs/2510.23227v1)

**作者:** Klaus Zauner, Josef El Dib, Hubert Gattringer, Andreas Mueller

**发布时间:** 2025-10-27

### GPT解析

### 总结

本文主要研究机器人运动规划中的环境建模和碰撞检测方法

### 背景

机器人运动规划依赖于对环境的精确知识，需要定义受限区域并考虑碰撞物体

### 目的

比较不同传感器，说明从检测到完成碰撞环境的过程，以及检测机器人与环境的碰撞

### 方法

使用各种传感器获取环境点云，通过区域增长分割和VCCS算法识别碰撞物体，并对点簇进行近似处理

### 主要发现

摘要中未明确提及具体发现

### 结论

摘要中未明确提及结论

### 翻译

机器人运动规划依赖于对环境的精确知识，以便能够定义受限区域并考虑碰撞物体。为了捕获工作空间，使用各种传感器获取环境的点云。碰撞物体通过区域增长分割和VCCS算法进行识别。随后对点簇进行近似处理。本文的目的是比较不同的传感器，说明从检测到完成碰撞环境的过程，并检测机器人与该环境之间的碰撞。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决工业机器人在复杂生产环境中进行精确环境感知和碰撞检测的问题。这个问题在现实中非常重要，因为它关系到机器人能否安全高效地工作，避免与周围环境发生碰撞，同时最大化利用可用工作空间，确保生产过程的顺利进行和人员设备的安全。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先考虑了机器人操作环境需要精确建模的需求，然后设计了一套从环境感知到碰撞检测的完整流程。他们借鉴了多项现有工作：使用了Point Cloud Library (PCL)中的方法和算法；应用了区域增长分割算法来识别物体；采用了Voxel Cloud Connectivity Segmentation (VCCS)算法进行更精细的分割；并使用了分离轴定理和基于Minkowski差的碰撞检测方法。作者的主要贡献在于将这些技术整合成一个完整的工业机器人应用系统，并进行了系统性的比较和优化。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是通过传感器获取环境点云数据，经过处理和分割后将环境近似为简单的几何形状，然后应用碰撞检测算法确保机器人安全运行。整体流程包括：1)环境描述：将机器人操作环境表示为点云数据；2)检测和特征提取：使用3D传感器获取环境数据，进行预处理和去噪，通过分割算法识别和聚类物体；3)边界框生成：将聚类近似为立方体边界框，并与物体协方差矩阵主轴对齐以减少空间损失；4)碰撞检测：使用分离轴定理或基于Minkowski差的方法检测机器人与环境的碰撞。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)系统比较了不同传感器(TOF、主动立体视觉)在工业机器人环境感知中的性能差异；2)提出了一套从环境检测到完成碰撞环境的完整流程；3)研究了不同的物体近似方法及其对可用空间的影响；4)比较了不同碰撞检测方法在可用空间和约束数量方面的表现。相比之前的工作，这篇论文不仅关注单一算法，而是关注整个系统流程；不仅评估算法性能，还考虑了实际可用工作空间；通过物体对齐方法和基于Minkowski差的碰撞检测，显著减少了约束数量，提高了路径规划效率。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '该论文提出了一套完整的工业机器人环境感知与碰撞检测方法，通过系统比较不同传感器和碰撞检测算法，优化了机器人工作空间利用率和路径规划效率。'}


### 论文摘要

Motion planning for robotic manipulators relies on precise knowledge of the environment in order to be able to define restricted areas and to take collision objects into account. To capture the workspace, point clouds of the environment are acquired using various sensors. The collision objects are identified by region growing segmentation and VCCS algorithm. Subsequently the point clusters are approximated. The aim of the present paper is to compare different sensors, to illustrate the process from detection to the finished collision environment and to detect collisions between the robot and this environment.

---

## 89. UGAE: Unified Geometry and Attribute Enhancement for G-PCC Compressed Point Clouds

**论文链接:** [http://arxiv.org/abs/2510.23009v1](http://arxiv.org/abs/2510.23009v1)

**作者:** Pan Zhao, Hui Yuan, Chongzhen Tian, Tian Guo, Raouf Hamzaoui, Zhigeng Pan

**发布时间:** 2025-10-27

### GPT解析

### 总结

本文提出了一种统一的几何和属性增强(UGAE)框架，通过三个核心组件(PoGE、PAE和PoAE)有效解决了点云有损压缩导致的几何结构和属性信息失真问题，在多个基准数据集上显著优于现有方法。

### 背景

点云的有损压缩可以减少存储和传输成本，但不可避免地导致几何结构和属性信息中的不可逆失真。

### 目的

解决有损压缩导致的几何结构和属性信息失真问题，提高压缩后的点云质量。

### 方法

提出统一的几何和属性增强(UGAE)框架，包含三个核心组件：1)后几何增强(PoGE)使用基于Transformer的稀疏卷积U-Net重建几何结构；2)预属性增强(PAE)引入增强几何引导的重新着色策略，使用DA-KNN方法保留高频细节；3)后属性增强(PoAE)使用带W-MSE损失的属性残差预测网络增强高频区域质量。

### 主要发现

UGAE在8iVFB、Owlii和MVUB三个基准数据集上显著优于现有方法；与G-PCC测试模型相比，几何部分平均BD-PSNR增益9.98 dB，BD-比特率节省90.98%；属性部分BD-PSNR提高3.67 dB，BD-比特率节省56.88%；显著改善了感知质量。

### 结论

UGAE框架能有效解决点云有损压缩导致的失真问题，在多个指标上表现优异，具有实际应用价值。

### 翻译

有损压缩点云减少了存储和传输成本；然而，它不可避免地导致几何结构和属性信息中的不可逆失真。为解决这些问题，我们提出了统一的几何和属性增强(UGAE)框架，包含三个核心组件：后几何增强(PoGE)、预属性增强(PAE)和后属性增强(PoAE)。在PoGE中，使用基于Transformer的稀疏卷积U-Net通过预测体素占用概率高精度重建几何结构。基于改进的几何结构，PAE引入创新的增强几何引导重新着色策略，使用细节感知的K-近邻(DA-KNN)方法实现精确重新着色，并在属性压缩前有效保留高频细节。最后，在解码器端，PoAE使用带加权均方误差(W-MSE)损失的属性残差预测网络，增强高频区域质量，同时保持低频区域的保真度。UGAE在三个基准数据集上显著优于现有方法：8iVFB、Owlii和MVUB。与最新的G-PCC测试模型(TMC13v29)相比，UGAE在D1指标下几何部分平均BD-PSNR增益9.98 dB，BD-比特率节省90.98%，属性部分在Y分量上BD-PSNR提高3.67 dB，BD-比特率节省56.88%。此外，它显著改善了感知质量。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决点云压缩过程中几何结构和属性信息不可避免产生的不可逆失真问题。这个问题在现实中非常重要，因为点云数据广泛应用于自动驾驶、文化遗产保护和虚拟现实等领域，高精度点云数据量大，存储和传输成本高，而现有压缩方法在减少数据量的同时会引入失真，影响3D应用的效率和用户体验。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先分析了现有点云压缩方法的局限性，发现它们主要分别优化几何或属性，忽略了两者间的耦合关系。现有联合增强方法仅在解码器端进行增强，无法充分利用几何增强对属性压缩的好处。作者借鉴了点云上采样方法（如PU-Net、PUFA-GAN）、稀疏卷积方法（如PU-Dense、GRNet）以及网络架构（Transformer和U-Net）和损失函数（BCE和MSE），设计了包含PoGE、PAE和PoAE三个核心组件的UGAE框架，在整个压缩过程中协同优化几何和属性。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是联合优化几何和属性，通过编码器-解码器协同增强来保留高频细节，分阶段解决压缩失真问题。整体流程是：编码器端，PoGE接收有损几何并生成增强几何结构；PAE使用增强几何和原始属性通过DA-KNN重新着色生成中间属性。传输有损几何比特流和重新着色后的属性比特流。解码器端，PoGE重建相同的增强几何；PoAE使用W-MSE损失函数专注于重建属性残差，特别是在高频区域。最终输出联合增强的点云。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1) 统一的几何和属性增强框架(UGAE)；2) 后几何增强(PoGE)结合Transformer和U-Net架构，使用密集连接并解决GPU随机性问题；3) 前属性增强(PAE)引入DA-KNN算法保留高频细节；4) 后属性增强(PoAE)使用W-MSE损失函数专注于高频区域。相比之前工作，UGAE同时处理几何和属性失真，考虑两者耦合关系，在编码器和解码器端都进行增强，而G-PCC++等现有方法仅在解码器端增强，且基于有损几何进行重新着色。UGAE在三个基准数据集上性能显著优于现有方法。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '这篇论文提出了UGAE框架，通过在编码器和解码器端协同优化几何和属性增强，显著提高了点云压缩质量，特别是在保留高频细节方面表现优异。'}


### 论文摘要

Lossy compression of point clouds reduces storage and transmission costs; however, it inevitably leads to irreversible distortion in geometry structure and attribute information. To address these issues, we propose a unified geometry and attribute enhancement (UGAE) framework, which consists of three core components: post-geometry enhancement (PoGE), pre-attribute enhancement (PAE), and post-attribute enhancement (PoAE). In PoGE, a Transformer-based sparse convolutional U-Net is used to reconstruct the geometry structure with high precision by predicting voxel occupancy probabilities. Building on the refined geometry structure, PAE introduces an innovative enhanced geometry-guided recoloring strategy, which uses a detail-aware K-Nearest Neighbors (DA-KNN) method to achieve accurate recoloring and effectively preserve high-frequency details before attribute compression. Finally, at the decoder side, PoAE uses an attribute residual prediction network with a weighted mean squared error (W-MSE) loss to enhance the quality of high-frequency regions while maintaining the fidelity of low-frequency regions. UGAE significantly outperformed existing methods on three benchmark datasets: 8iVFB, Owlii, and MVUB. Compared to the latest G-PCC test model (TMC13v29), UGAE achieved an average BD-PSNR gain of 9.98 dB and 90.98% BD-bitrate savings for geometry under the D1 metric, as well as a 3.67 dB BD-PSNR improvement with 56.88% BD-bitrate savings for attributes on the Y component. Additionally, it improved perceptual quality significantly.

---

## 90. Scaling Up Occupancy-centric Driving Scene Generation: Dataset and Method

**论文链接:** [http://arxiv.org/abs/2510.22973v1](http://arxiv.org/abs/2510.22973v1)

**作者:** Bohan Li, Xin Jin, Hu Zhu, Hongsi Liu, Ruikai Li, Jiazhe Guo, Kaiwen Cai, Chao Ma, Yueming Jin, Hao Zhao, Xiaokang Yang, Wenjun Zeng

**发布时间:** 2025-10-27

**备注:** https://github.com/Arlo0o/UniScene-Unified-Occupancy-centric-Driving-Scene-Generation/tree/v2

### GPT解析

### 总结

该研究提出了一种统一的占据中心驾驶场景生成方法，通过创建大规模语义占据数据集Nuplan-Occ和开发统一框架，解决了占据中心方法对标注数据依赖的问题，实现了高质量语义占据、多视图视频和LiDAR点云的联合生成。

### 背景

场景生成是自动驾驶的关键领域，占据中心方法最近取得了最先进的结果，但这些方法严重依赖于标注的占据数据，而这类数据仍然稀缺。

### 目的

克服占据中心方法对标注数据的依赖限制，创建大规模语义占据数据集，并开发统一框架实现多模态场景生成。

### 方法

创建Nuplan-Occ数据集，开发统一框架联合生成语义占据、多视图视频和LiDAR点云，采用时空解耦架构支持4D动态占据的扩展和预测，提出基于高斯飞溅的稀疏点图渲染策略和传感器感知嵌入策略。

### 主要发现

该方法在生成保真度和可扩展性方面优于现有方法，在下游任务中验证了其实用价值。

### 结论

该研究提出的统一占据中心驾驶场景生成方法在自动驾驶场景生成方面表现出色，有助于感知和规划评估等下游应用。

### 翻译

场景生成是自动驾驶的关键领域，使包括感知和规划评估在内的下游应用成为可能。占据中心方法最近通过提供跨帧和模态的一致条件取得了最先进的结果；然而，它们的性能严重依赖于标注的占据数据，而这种数据仍然稀缺。为了克服这一限制，我们整理了Nuplan-Occ，这是迄今为止最大的语义占据数据集，由广泛使用的Nuplan基准构建而成。其规模和多样性不仅促进了大规模生成建模，也促进了自动驾驶的下游应用。基于此数据集，我们开发了一个统一框架，联合合成高质量语义占据、多视图视频和LiDAR点云。我们的方法采用时空解耦架构，支持4D动态占据的高保真空间扩展和时间预测。为了弥合模态差距，我们进一步提出了两种新颖技术：一种基于高斯飞溅的稀疏点图渲染策略，增强多视图视频生成；一种传感器感知嵌入策略，明确建模LiDAR传感器特性，以实现真实的多LiDAR模拟。大量实验表明，与现有方法相比，我们的方法在生成保真度和可扩展性方面取得了优越的性能，并在下游任务中验证了其实际价值。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决自动驾驶场景生成中的规模限制问题。现有占用中心方法虽先进，但受限于标注数据稀缺，无法实现大规模训练；同时，多模态生成（语义占用、视频、LiDAR）存在模态差距，导致生成质量受限。这一问题对自动驾驶领域至关重要，因为高质量场景生成能支持感知和规划评估，降低开发成本，提高系统鲁棒性和安全性，同时支持算法在多样化环境中训练，提升泛化能力。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先认识到现有方法受限于数据稀缺性，无法实现大规模训练；然后意识到需要构建更大规模数据集；观察到多模态生成存在模态差距，需要新技术弥合；提出时空解耦架构分解4D占用生成；为解决视频生成中的传感器校准问题，引入高斯飞溅稀疏点图渲染；为实现真实LiDAR模拟，提出传感器感知嵌入策略。该方法借鉴了UniScene的前期工作，利用了扩散模型、3D高斯表示、CogVideoX的3D因果VAE和体积渲染技术等现有成果。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是以语义占用为中心构建统一框架，通过时空解耦架构将4D动态占用生成分解为空间扩展和时间预测，利用大规模数据集支持训练，并通过专门技术弥合模态差距。整体流程：1)构建Nuplan-Occ数据集，使用前景-背景分离聚合策略；2)4D占用生成，使用VAE和DiT编码解码，时空解耦处理；3)视频生成，将占用转为3D高斯基元渲染成稀疏点图，用视频扩散Transformer生成；4)LiDAR生成，使用传感器感知嵌入和稀疏UNet，应用射线平滑正则化。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点：1)可扩展的统一4D动态场景生成框架，联合扩展模型架构和数据；2)4D占用的时空解耦建模，分离空间扩展和时间预测；3)多传感器真实性的模态桥接策略，包括稀疏点图渲染和传感器感知嵌入；4)构建Nuplan-Occ最大语义占用数据集。相比之前工作，本文数据规模更大（比Nuscenes-Occupancy大19倍），采用时空解耦架构而非混合处理，使用分层生成策略和稀疏渲染技术，在多个任务上取得了最先进性能。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '本文提出了UniScenev2，一个基于大规模Nuplan-Occ数据集的统一占用中心框架，通过时空解耦架构和模态桥接技术，实现了高质量语义占用、多视角视频和LiDAR点云的联合生成，显著提升了自动驾驶场景生成的规模和质量。'}


### 论文摘要

Driving scene generation is a critical domain for autonomous driving, enabling downstream applications, including perception and planning evaluation. Occupancy-centric methods have recently achieved state-of-the-art results by offering consistent conditioning across frames and modalities; however, their performance heavily depends on annotated occupancy data, which still remains scarce. To overcome this limitation, we curate Nuplan-Occ, the largest semantic occupancy dataset to date, constructed from the widely used Nuplan benchmark. Its scale and diversity facilitate not only large-scale generative modeling but also autonomous driving downstream applications. Based on this dataset, we develop a unified framework that jointly synthesizes high-quality semantic occupancy, multi-view videos, and LiDAR point clouds. Our approach incorporates a spatio-temporal disentangled architecture to support high-fidelity spatial expansion and temporal forecasting of 4D dynamic occupancy. To bridge modal gaps, we further propose two novel techniques: a Gaussian splatting-based sparse point map rendering strategy that enhances multi-view video generation, and a sensor-aware embedding strategy that explicitly models LiDAR sensor properties for realistic multi-LiDAR simulation. Extensive experiments demonstrate that our method achieves superior generation fidelity and scalability compared to existing approaches, and validates its practical value in downstream tasks. Repo: https://github.com/Arlo0o/UniScene-Unified-Occupancy-centric-Driving-Scene-Generation/tree/v2

---

## 91. TWC-SLAM: Multi-Agent Cooperative SLAM with Text Semantics and WiFi Features Integration for Similar Indoor Environments

**论文链接:** [http://arxiv.org/abs/2510.22754v1](http://arxiv.org/abs/2510.22754v1)

**作者:** Chunyu Li, Shoubin Chen, Dong Li, Weixing Xue, Qingquan Li

**发布时间:** 2025-10-26

**备注:** Accepted by the IEEE/RSJ International Conference on Intelligent  Robots and Systems (IROS) 2025

### GPT解析

### 总结

这项研究提出了TWC-SLAM，一种多智能体协作SLAM框架，通过整合文本语义和WiFi信号特征来增强位置识别和回环检测，以提高在具有重复结构的相似室内环境中的协作SLAM性能。

### 背景

多智能体协作SLAM在具有重复结构的相似室内环境中（如走廊和房间）常面临挑战。当使用基于点云的技术时，这些挑战会导致共享位置识别出现显著不准确。

### 目的

减轻多智能体协作SLAM在相似室内环境中的挑战，提高共享位置识别的准确性。

### 方法

TWC-SLAM框架包括基于FAST-LIO2的单智能体前端里程计模块、利用文本语义和WiFi特征的位置识别和回环检测模块以及全局映射模块。智能体配备了能够捕获文本信息和检测WiFi信号的传感器。通过关联这些数据源，TWC-SLAM建立共同位置，促进不同智能体地图之间的点云对齐，并采用回环检测和优化模块实现全局优化和一致性映射。

### 主要发现

使用具有相似走廊、房间和文本标志的室内数据集评估的结果表明，TWC-SLAM显著提高了在具有重复建筑特征的复杂环境中协作SLAM系统的性能。

### 结论

整合文本语义和WiFi信号特征可以有效提高多智能体协作SLAM在具有重复结构的相似室内环境中的性能，特别是在位置识别和回环检测方面。

### 翻译

多智能体协作SLAM常在具有重复结构的相似室内环境（如走廊和房间）中遇到挑战。当采用基于点云的技术时，这些挑战可能导致共享位置识别出现显著不准确。为缓解这些问题，我们引入了TWC-SLAM，一种多智能体协作SLAM框架，它整合文本语义和WiFi信号特征以增强位置识别和回环检测。TWC-SLAM包括基于FAST-LIO2的单智能体前端里程计模块、利用文本语义和WiFi特征的位置识别和回环检测模块以及全局映射模块。智能体配备了能够捕获文本信息和检测WiFi信号的传感器。通过关联这些数据源，TWC-SLAM建立共同位置，促进不同智能体地图之间的点云对齐。此外，系统采用回环检测和优化模块来实现全局优化和一致性映射。我们使用具有相似走廊、房间和文本标志的室内数据集评估了我们的方法。结果表明，TWC-SLAM显著提高了在具有重复建筑特征的复杂环境中协作SLAM系统的性能。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决多智能体协同SLAM在具有重复结构的相似室内环境（如走廊和相似房间）中面临的位置识别不准确问题。这个问题在现实中很重要，因为许多室内环境（如办公楼、医院、学校）都有相似结构，传统基于点云的技术容易在这些环境中产生错误匹配，导致地图不一致和定位错误，影响多智能体系统在检查、救援、物流等领域的实际应用效果。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先分析了传统方法在相似环境中的局限性，然后借鉴了现有工作如FAST-LIO2作为前端里程计模块，参考了TextSLAM等文本识别方法和SpotFi等WiFi定位技术。作者设计了多模态融合思路，结合文本语义（提供明确标识但可能重复）和WiFi特征（提供环境特定信号但区分度有限）两种互补信息，设计了四个主要组件：多智能体前端里程计、文本语义匹配、WiFi特征匹配和全局映射模块，通过算法1实现多模态位置识别，确保位置识别的准确性。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是通过整合文本语义和WiFi特征这两种互补的模态信息，解决相似室内环境中多智能体协同SLAM的位置识别挑战。整体流程包括：1)多智能体使用FAST-LIO2计算里程和生成点云地图；2)通过OCR提取文本语义，使用Levenshtein距离计算文本相似度进行匹配；3)收集WiFi数据，计算MAC地址相似度和RSS值相似度进行验证；4)基于匹配结果识别相同位置，使用点云匹配方法计算坐标变换，执行回环检测和全局优化，生成一致的全局地图。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)首创性地整合文本语义与WiFi特征进行多智能体协同SLAM；2)提出新颖的回环检测和位置识别方法，协同利用两种模态信息；3)构建专门的多智能体数据集。相比之前工作，TWC-SLAM比传统点云方法精度高88%，比纯文本方法高82%，比纯WiFi方法高92%。它解决了单一模态方法在相似环境中的局限性，通过多模态融合提高了位置识别的准确性和鲁棒性。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'TWC-SLAM通过创新性地整合文本语义与WiFi特征，显著提高了多智能体协同SLAM系统在具有重复结构的相似室内环境中的定位精度和地图一致性，解决了传统方法在复杂环境中易出现错误匹配的关键挑战。'}


### 论文摘要

Multi-agent cooperative SLAM often encounters challenges in similar indoor environments characterized by repetitive structures, such as corridors and rooms. These challenges can lead to significant inaccuracies in shared location identification when employing point cloud-based techniques. To mitigate these issues, we introduce TWC-SLAM, a multi-agent cooperative SLAM framework that integrates text semantics and WiFi signal features to enhance location identification and loop closure detection. TWC-SLAM comprises a single-agent front-end odometry module based on FAST-LIO2, a location identification and loop closure detection module that leverages text semantics and WiFi features, and a global mapping module. The agents are equipped with sensors capable of capturing textual information and detecting WiFi signals. By correlating these data sources, TWC-SLAM establishes a common location, facilitating point cloud alignment across different agents' maps. Furthermore, the system employs loop closure detection and optimization modules to achieve global optimization and cohesive mapping. We evaluated our approach using an indoor dataset featuring similar corridors, rooms, and text signs. The results demonstrate that TWC-SLAM significantly improves the performance of cooperative SLAM systems in complex environments with repetitive architectural features.

---

## 92. Estimating Continuum Robot Shape under External Loading using Spatiotemporal Neural Networks

**论文链接:** [http://arxiv.org/abs/2510.22339v1](http://arxiv.org/abs/2510.22339v1)

**作者:** Enyi Wang, Zhen Deng, Chuanchuan Pan, Bingwei He, Jianwei Zhang

**发布时间:** 2025-10-25

**备注:** 2025 IEEE/RSJ International Conference on Intelligent Robots and  Systems (IROS)

### GPT解析

### 总结

本文提出了一种基于学习的方法，用于准确估计受外部负载的柔性连续体机器人的3D形状。该方法通过时空神经网络架构融合多模态输入，生成点云表示机器人变形配置，并通过拟合贝塞尔曲线实现连续3D形状重建。实验验证显示该方法具有高精度，优于现有方法。

### 背景

柔性连续体机器人在外部负载下的3D形状估计是一个挑战性问题。

### 目的

开发一种基于学习的方法来准确估计柔性连续体机器人在外部负载下的3D形状。

### 方法

提出时空神经网络架构，融合多模态输入（当前和历史肌腱位移数据以及RGB图像），生成点云表示机器人变形配置。网络集成了循环神经模块进行时间特征提取，编码模块进行空间特征提取，以及多模态融合模块结合视觉数据的空间特征和历史执行器输入的时间依赖性。通过将贝塞尔曲线拟合到预测点云上实现连续3D形状重建。

### 主要发现

实验验证显示该方法具有高精度，无负载时平均形状估计误差为0.08毫米，负载时为0.22毫米，优于最先进的TDCRs形状传感方法。

### 结论

基于深度学习的时空数据融合在负载条件下能有效实现精确的形状估计。

### 翻译

本文提出了一种基于学习的方法，用于准确估计受外部负载的柔性连续体机器人的3D形状。提出的方法引入了一种时空神经网络架构，融合多模态输入，包括当前和历史肌腱位移数据以及RGB图像，生成代表机器人变形配置的点云。网络集成了循环神经模块进行时间特征提取，编码模块进行空间特征提取，以及多模态融合模块来结合从视觉数据中提取的空间特征和来自历史执行器输入的时间依赖性。通过将贝塞尔曲线拟合到预测的点云上实现连续3D形状重建。实验验证表明，我们的方法实现了高精度，无负载时平均形状估计误差为0.08毫米，负载时为0.22毫米，优于TDCRs形状传感的最先进方法。结果证明了基于深度学习的时空数据融合在负载条件下精确形状估计的有效性。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决如何准确估计在外部负载作用下的柔性连续机器人的3D形状问题。这个问题很重要，因为连续机器人在医疗手术、工业检测等领域有广泛应用，而外部负载会改变它们的变形行为，使得准确预测形状变得复杂。精确的形状估计对机器人控制至关重要，能帮助它们动态调整配置以优化任务执行和环境交互。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先分析了现有方法的局限性，包括传感器方法的精度限制和易损坏问题，模型方法难以处理材料非线性和未知外部负载，以及数据驱动方法在负载下鲁棒性有限。他们设计了一种时空神经网络架构，融合多模态输入数据。该方法借鉴了U-Net架构用于视觉特征提取，LSTM网络捕捉时间依赖性，以及空间注意力机制进行特征融合。同时创新性地将这些技术组合，形成了一个能够处理外部负载条件下的形状估计系统。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是利用时空神经网络融合视觉数据（RGB图像）和肌腱位移数据，建立这些多模态输入与连续机器人3D形状之间的非线性映射关系。整体流程包括：1)输入当前和历史肌腱位移数据及RGB图像；2)通过空间特征提取器从图像中提取空间特征；3)利用时间特征提取器处理肌腱位移序列；4)通过注意力机制融合时空特征；5)预测代表机器人形状的3D点云；6)使用贝塞尔曲线拟合点云生成连续3D形状。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)多模态数据融合，同时利用视觉和肌腱位移数据；2)时空神经网络架构，结合卷积和循环神经网络；3)在外部负载下实现高精度形状估计；4)端到端学习框架。相比之前工作，该方法结合了视觉和本体感觉数据，提高了精度和鲁棒性；使用更先进的神经网络架构；在各种负载条件下表现出更好的一致性；无需显式机械建模，能处理材料非线性和未知外部负载。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '这篇论文提出了一种基于时空神经网络的创新方法，通过融合视觉和肌腱位移数据，实现了在外部负载条件下高精度估计连续机器人3D形状的目标，相比现有方法在精度和鲁棒性上均有显著提升。'}


### 论文摘要

This paper presents a learning-based approach for accurately estimating the 3D shape of flexible continuum robots subjected to external loads. The proposed method introduces a spatiotemporal neural network architecture that fuses multi-modal inputs, including current and historical tendon displacement data and RGB images, to generate point clouds representing the robot's deformed configuration. The network integrates a recurrent neural module for temporal feature extraction, an encoding module for spatial feature extraction, and a multi-modal fusion module to combine spatial features extracted from visual data with temporal dependencies from historical actuator inputs. Continuous 3D shape reconstruction is achieved by fitting B\'ezier curves to the predicted point clouds. Experimental validation demonstrates that our approach achieves high precision, with mean shape estimation errors of 0.08 mm (unloaded) and 0.22 mm (loaded), outperforming state-of-the-art methods in shape sensing for TDCRs. The results validate the efficacy of deep learning-based spatiotemporal data fusion for precise shape estimation under loading conditions.

---

## 93. Breaking the Static Assumption: A Dynamic-Aware LIO Framework Via Spatio-Temporal Normal Analysis

**论文链接:** [http://arxiv.org/abs/2510.22313v1](http://arxiv.org/abs/2510.22313v1)

**作者:** Chen Zhiqiang, Le Gentil Cedric, Lin Fuling, Lu Minghao, Qiao Qiyuan, Xu Bowen, Qi Yuhua, Lu Peng

**发布时间:** 2025-10-25

**备注:** 8 pages, 7 figures, Accepted to IEEE Robotics and Automation Letters  (RA-L)

### GPT解析

### 总结

本文提出了一种解决动态环境中激光雷达-惯性里程计(LIO)挑战的新方法，通过将动态感知直接集成到点云配准过程中，打破了静态特征识别和姿态估计之间的循环依赖关系。

### 背景

传统LIO算法基于静态世界假设，在动态环境中表现不佳，特别是在动态物体主导场景和几何稀疏环境中。当前动态LIO方法面临根本性挑战：准确的定位需要可靠识别静态特征，而区分动态物体又需要精确的姿态估计。

### 目的

解决动态环境中的LIO挑战，打破静态特征识别和姿态估计之间的循环依赖关系。

### 方法

引入了一种新颖的动态感知迭代最近点算法，利用时空法线分析，并配以高效的空间一致性验证方法来增强静态地图构建。

### 主要发现

在具有有限几何结构的挑战性动态环境中，与最先进的LIO系统相比，性能有显著提升。

### 结论

所提出的方法有效解决了动态环境中的LIO问题，代码和数据集已在GitHub上公开。

### 翻译

本文解决了动态环境中激光雷达-惯性里程计(LIO)的挑战，传统方法由于其静态世界假设经常失败。当动态物体主导场景，特别是在几何稀疏环境中时，传统LIO算法表现不佳。当前动态LIO方法面临一个基本挑战：准确的定位需要可靠识别静态特征，而区分动态物体又需要精确的姿态估计。我们的解决方案通过将动态感知直接集成到点云配准过程中，打破了这种循环依赖。我们引入了一种新颖的动态感知迭代最近点算法，利用时空法线分析，并辅以高效的空间一致性验证方法来增强静态地图构建。实验评估表明，在具有有限几何结构的挑战性动态环境中，与最先进的LIO系统相比，性能有显著提升。代码和数据集可在https://github.com/thisparticle/btsa获取。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决激光雷达-惯性里程计（LIO）在动态环境中的定位和地图构建问题。传统LIO系统假设环境是静态的，当场景中存在大量移动物体（如行人、车辆）时会导致严重的定位误差。这个问题在现实中非常重要，因为真实世界环境通常是动态的，特别是在几何特征稀疏的环境中，动态物体可能主导场景，使传统系统完全失效。此外，现有方法存在循环依赖问题：准确的定位需要可靠的静态特征识别，而有效的动态物体检测又需要精确的位姿估计。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先分析了现有方法的局限性：传统方法将动态物体检测作为预处理步骤而非解决配准算法的根本问题；学习技术只能检测预定义物体类别且需要大量训练数据；几何方法依赖于特定假设且在复杂场景中表现不佳。作者借鉴了时空法线分析的概念，但创新性地将其直接集成到点云配准过程中，而不是作为后处理。同时，作者采用了双地图架构（时间滑动窗口地图用于时空法线计算，长期体素地图提供全局一致性），并扩展了点对点ICP算法的异常值剔除步骤。通过这种方式，作者打破了状态估计和动态物体检测之间的循环依赖。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是通过时空法线分析直接将动态感知集成到点云配准过程中，同时解决状态估计和动态点分类问题，打破两者之间的循环依赖。整体流程包括：1）输入数据预处理（IMU预积分和点云畸变校正）；2）点云下采样；3）动态感知点云配准（计算时空法线、分类稳定/不稳定点、迭代优化位姿）；4）静态地图构建（不稳定点上采样、DBSCAN聚类、空间一致性检查）。这一过程在每个迭代中同时评估点的动态性和优化位姿估计，而不是分步处理。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1）动态感知ICP算法，通过时空法线分析直接集成动态感知到配准过程；2）双地图架构平衡时空计算需求；3）高效的空间一致性验证方法改进静态地图构建；4）在真正具有挑战性的动态环境中进行验证。相比之前工作的不同：传统方法将动态检测作为预处理步骤，而本文直接集成到配准中；现有方法要么依赖学习技术（需大量训练数据且泛化有限），要么依赖几何假设（在复杂场景中表现不佳）；大多数方法假设已知精确位姿，而本文同时优化位姿和动态点分类；评估不仅限于几何丰富的城市环境，还包括几何退化和动态物体主导的环境。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '这篇论文提出了一种通过时空法线分析直接集成动态感知到激光雷达-惯性里程计配准过程中的新框架，打破了状态估计和动态物体检测之间的循环依赖，在具有挑战性的动态环境中实现了更准确和鲁棒的定位与地图构建。'}


### 论文摘要

This paper addresses the challenge of Lidar-Inertial Odometry (LIO) in dynamic environments, where conventional methods often fail due to their static-world assumptions. Traditional LIO algorithms perform poorly when dynamic objects dominate the scenes, particularly in geometrically sparse environments. Current approaches to dynamic LIO face a fundamental challenge: accurate localization requires a reliable identification of static features, yet distinguishing dynamic objects necessitates precise pose estimation. Our solution breaks this circular dependency by integrating dynamic awareness directly into the point cloud registration process. We introduce a novel dynamic-aware iterative closest point algorithm that leverages spatio-temporal normal analysis, complemented by an efficient spatial consistency verification method to enhance static map construction. Experimental evaluations demonstrate significant performance improvements over state-of-the-art LIO systems in challenging dynamic environments with limited geometric structure. The code and dataset are available at https://github.com/thisparticle/btsa.

---

## 94. Linearized Optimal Transport for Analysis of High-Dimensional Point-Cloud and Single-Cell Data

**论文链接:** [http://arxiv.org/abs/2510.22033v1](http://arxiv.org/abs/2510.22033v1)

**作者:** Tianxiang Wang, Yingtong Ke, Dhananjay Bhaskar, Smita Krishnaswamy, Alexander Cloninger

**发布时间:** 2025-10-24

**备注:** 11 pages, 5 figures

### GPT解析

### 总结

该研究提出了一种基于线性最优传输(LOT)的框架，用于处理单细胞技术生成的高维点云数据，解决了不规则点云难以直接量化和比较的问题，同时兼顾了预测准确性和生物学可解释性。

### 背景

单细胞技术生成高维点云数据，能够详细表征复杂的患者状态和治疗反应，但每个患者由不规则点云表示，难以直接量化和比较个体间的生物学差异。非线性方法虽能达到预测准确性，但作为黑盒模型，生物学可解释性差。

### 目的

将不规则点云嵌入到固定维度的欧几里得空间中，同时保留分布结构，提供一种有原则性的线性表示，保留最优传输几何结构，同时支持下游分析。

### 方法

适应线性最优传输(LOT)框架到单细胞数据分析场景，将不规则点云嵌入到固定维度的欧几里得空间中，保留分布结构，形成任意两个患者之间的配准，使其能够直接比较细胞分布。

### 主要发现

LOT实现了COVID-19患者状态的准确且可解释的分类，分类器权重映射回驱动预测的特定标记物和空间区域；同时实现了患者来源类器官的合成数据生成，利用LOT嵌入的线性特性；LOT重心产生表示组合条件或样本的平均细胞谱，支持药物相互作用测试。

### 结论

LOT作为一个统一框架，连接了预测性能、可解释性和生成建模，通过将异质点云转换为可直接追踪到原始数据的结构化嵌入，为理解高维生物系统中的免疫变异和治疗效应开辟了新机会。

### 翻译

单细胞技术生成细胞的高维点云，能够详细表征复杂的患者状态和治疗反应。然而每个患者由不规则点云而非简单向量表示，使得难以直接量化和比较个体间的生物学差异。非线性方法如核方法和神经网络能实现预测准确性，但作为黑箱模型，提供的生物学可解释性有限。为解决这些限制，我们将线性最优传输(LOT)框架适应到这一场景，将不规则点云嵌入到固定维度的欧几里得空间中，同时保留分布结构。这种嵌入提供了有原则性的线性表示，保留最优传输几何结构，同时支持下游分析。它还形成了任意两个患者之间的配准，使其能够直接比较细胞分布。在此空间中，LOT实现了：(i) COVID-19患者状态的准确且可解释的分类，其中分类器权重映射回驱动预测的特定标记物和空间区域；(ii) 患者来源类器官的合成数据生成，利用LOT嵌入的线性特性。LOT重心产生表示组合条件或样本的平均细胞谱，支持药物相互作用测试。这些结果共同确立了LOT作为连接预测性能、可解释性和生成建模的统一框架。通过将异质点云转换为可直接追踪到原始数据的结构化嵌入，LOT为理解高维生物系统中的免疫变异和治疗效应开辟了新机会。


### 论文摘要

Single-cell technologies generate high-dimensional point clouds of cells, enabling detailed characterization of complex patient states and treatment responses. Yet each patient is represented by an irregular point cloud rather than a simple vector, making it difficult to directly quantify and compare biological differences between individuals. Nonlinear methods such as kernels and neural networks achieve predictive accuracy but act as black boxes, offering little biological interpretability.   To address these limitations, we adapt the Linear Optimal Transport (LOT) framework to this setting, embedding irregular point clouds into a fixed-dimensional Euclidean space while preserving distributional structure. This embedding provides a principled linear representation that preserves optimal transport geometry while enabling downstream analysis. It also forms a registration between any two patients, enabling direct comparison of their cellular distributions. Within this space, LOT enables: (i) \textbf{accurate and interpretable classification} of COVID-19 patient states, where classifier weights map back to specific markers and spatial regions driving predictions; and (ii) \textbf{synthetic data generation} for patient-derived organoids, exploiting the linearity of the LOT embedding. LOT barycenters yield averaged cellular profiles representing combined conditions or samples, supporting drug interaction testing.   Together, these results establish LOT as a unified framework that bridges predictive performance, interpretability, and generative modeling. By transforming heterogeneous point clouds into structured embeddings directly traceable to the original data, LOT opens new opportunities for understanding immune variation and treatment effects in high-dimensional biological systems.

---

## 95. Spatially Aware Linear Transformer (SAL-T) for Particle Jet Tagging

**论文链接:** [http://arxiv.org/abs/2510.23641v1](http://arxiv.org/abs/2510.23641v1)

**作者:** Aaron Wang, Zihan Zhao, Subash Katel, Vivekanand Gyanchand Sahu, Elham E Khoda, Abhijith Gandrakota, Jennifer Ngadiuba, Richard Cavanaugh, Javier Duarte

**发布时间:** 2025-10-24

### GPT解析

### 总结

本文提出了一种名为空间感知线性变换器(SAL-T)的新型架构，解决了在高能物理高数据吞吐量环境中部署Transformer模型时的资源消耗和延迟问题。

### 背景

Transformers在捕获高能粒子碰撞中的全局和局部相关性方面非常有效，但在高数据吞吐量环境(如CERN LHC)中部署面临挑战，其二次复杂度需要大量资源并增加推理延迟。

### 目的

开发一种资源高效且低延迟的Transformer变体，能够在保持高性能的同时适应高能物理等高数据吞吐量环境。

### 方法

SAL-T基于linformer架构，结合了空间感知粒子分区(基于运动学特征)和卷积层(捕获局部相关性)，实现了物理意义显著的区域间注意力计算。

### 主要发现

SAL-T在喷流分类任务中优于标准linformer，实现了与全注意力Transformer相当的结果，同时显著减少了资源使用和推理延迟；在ModelNet10点云分类数据集上验证了这一优势。

### 结论

SAL-T是一种高效解决方案，能够在保持高性能的同时显著降低计算需求和延迟，特别适合高能物理等资源受限环境。

### 翻译

Transformers在捕获高能粒子碰撞中的全局和局部相关性方面非常有效，但在高数据吞吐量环境中部署时面临挑战，例如CERN LHC。Transformer模型的二次复杂度需要大量资源并增加推理时的延迟。为了解决这些问题，我们引入了空间感知线性变换器(SAL-T)，这是对linformer架构的物理启发式增强，保持了线性注意力。我们的方法基于运动学特征对粒子进行空间感知分区，从而计算物理意义显著的区域之间的注意力。此外，我们使用卷积层来捕获局部相关性，这些见解来自喷流物理学。除了在喷流分类任务中优于标准linformer外，SAL-T还实现了与全注意力Transformer相当的分类结果，同时在推理过程中使用更少的资源和更低的延迟。在通用点云分类数据集(ModelNet10)上的实验进一步证实了这一趋势。我们的代码可在https://github.com/aaronw5/SAL-T4HEP获取。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决传统Transformer模型在粒子物理应用中的计算复杂度问题。传统Transformer具有二次方复杂度(O(n²))，导致在处理高能物理数据时资源需求大、推理延迟高。这个问题在现实中非常重要，因为CERN大型强子对撞机每秒产生4000万次碰撞事件，需要实时过滤系统(触发系统)来筛选数据，而传统Transformer无法满足这种低延迟、高吞吐量的需求。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先分析了现有高效Transformer变体(如Longformer、Linformer)在粒子物理中等长度输入(约100个token)上的局限性，然后结合粒子物理专业知识设计了SAL-T。方法借鉴了Linformer的线性注意力机制作为基础，并融入了粒子物理中的kT排序概念(用于粒子聚类算法)和卷积层设计来捕获局部相关性。作者通过三种主要修改来增强Linformer：基于kT指标的空间感知排序、分区注意力机制和卷积增强注意力，创造出一种既高效又能捕捉物理相关空间信息的新方法。', '这个方法的核心思想是什么？整体实现流程是怎样的？': 'SAL-T的核心思想是通过空间感知的分区和卷积增强来改进Linformer架构，使其能够更有效地处理粒子喷射分类任务，同时保持线性计算复杂度。具体实现流程包括：1)使用kT=pTΔR指标对粒子进行排序，确保物理相关的邻近粒子在序列中彼此接近；2)将排序后的键和值向量分区，使每个投影头只关注自己的粒子子集；3)在每个头的原始注意力分数上应用小型深度2D卷积，以融入局部邻居交互；4)通过线性分区粒子多头注意力(LPP-MHA)计算注意力并聚合值表示；5)对注意力输出进行最大聚合并传递到分类层。整个流程保持了线性计算复杂度(O(np))，同时捕获了局部喷射子结构。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)空间感知分区：基于kT指标对粒子排序并分区，使每个投影头关注物理相关的粒子子集；2)卷积增强注意力：在注意力分数上应用2D卷积，捕获局部邻居交互；3)物理启发的特征排序：使用kT而非传统pT排序，更好地保留空间结构；4)在保持线性复杂度的同时实现与全注意力Transformer相当的性能。相比之前工作，SAL-T克服了标准Linformer不编码空间信息的局限，实现了与全注意力Transformer相当的性能但计算复杂度显著降低，同时比特定于粒子物理的Transformer变体更适合资源受限的触发系统。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'SAL-T通过空间感知的分区和卷积增强，在保持线性计算复杂度的同时，实现了与全注意力Transformer相当的粒子喷射分类性能，使其成为资源受限的粒子物理触发系统的可行选择。'}


### 论文摘要

Transformers are very effective in capturing both global and local correlations within high-energy particle collisions, but they present deployment challenges in high-data-throughput environments, such as the CERN LHC. The quadratic complexity of transformer models demands substantial resources and increases latency during inference. In order to address these issues, we introduce the Spatially Aware Linear Transformer (SAL-T), a physics-inspired enhancement of the linformer architecture that maintains linear attention. Our method incorporates spatially aware partitioning of particles based on kinematic features, thereby computing attention between regions of physical significance. Additionally, we employ convolutional layers to capture local correlations, informed by insights from jet physics. In addition to outperforming the standard linformer in jet classification tasks, SAL-T also achieves classification results comparable to full-attention transformers, while using considerably fewer resources with lower latency during inference. Experiments on a generic point cloud classification dataset (ModelNet10) further confirm this trend. Our code is available at https://github.com/aaronw5/SAL-T4HEP.

---

## 96. Robust Point Cloud Reinforcement Learning via PCA-Based Canonicalization

**论文链接:** [http://arxiv.org/abs/2510.20974v2](http://arxiv.org/abs/2510.20974v2)

**作者:** Michael Bezick, Vittorio Giammarino, Ahmed H. Qureshi

**发布时间:** 2025-10-23

### GPT解析

### 总结

本文提出了一种名为PCA点云(PPC)的标准化框架，用于解决点云强化学习中对相机姿态不匹配的敏感性问题，提高了算法在现实环境中的鲁棒性和可靠性。

### 背景

强化学习从原始视觉输入中取得了显著成功，但对分布外变化(如光照、颜色和视点变化)仍然很脆弱。点云强化学习减轻了基于外观的脆弱性，但仍然受相机姿态不匹配的影响。

### 目的

解决点云强化学习对相机姿态不匹配的敏感性，提高在现实环境中的可靠性。

### 方法

提出PCA点云(PPC)框架，将任意刚体变换下的点云映射到唯一的规范姿态，使观测与一致框架对齐，减少视点引起的不一致性。

### 主要发现

PPC提高了对具有挑战性的机器人任务中未见过的相机姿态的鲁棒性，为领域随机化提供了有原则的替代方案。

### 结论

PPC框架有效解决了点云强化学习中的视点不一致性问题，提高了算法在现实环境中的鲁棒性和可靠性。

### 翻译

从原始视觉输入的强化学习(RL)近年来取得了显著的成功，但它对分布外变化(如光照、颜色和视点变化)仍然很脆弱。点云强化学习(PC-RL)通过减轻基于外观的脆弱性提供了一种有前途的替代方案，但它对相机姿态不匹配的敏感性继续削弱了在现实环境中的可靠性。为了解决这一挑战，我们提出了PCA点云(PPC)，这是一种专门为下游机器人控制设计的标准化框架。PPC将任意刚体变换下的点云映射到唯一的规范姿态，使观测与一致框架对齐，从而显著减少视点引起的不一致性。在我们的实验中，我们表明PPC提高了对具有挑战性的机器人任务中未见过的相机姿态的鲁棒性，为领域随机化提供了有原则的替代方案。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决点云强化学习（PC-RL）中的视角敏感性问题。点云通常在相机局部坐标系中表示，即使微小的视角变化也会导致场景表示显著改变，影响算法在实际环境中的可靠性。这个问题很重要，因为视角变化是现实世界中的常见现象，而现有方法如域随机化往往导致样本效率低下且理论保证较弱。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先识别出PC-RL虽然减轻了外观相关的脆弱性，但仍存在视角敏感性问题。然后分析了域随机化方法的局限性，借鉴了旋转不变点云分析领域的进展，特别是基于PCA的标准化方法。作者识别出PCA方法存在符号歧义问题，设计了新颖的几何驱动消歧步骤。该方法借鉴了PointNet、PointNet++等点云处理方法，以及PointPatch RL作为基础强化学习框架。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是通过基于PCA的标准化方法，将点云映射到唯一的规范姿态，实现对刚体变换（平移和旋转）的不变性。具体流程包括：1)点云下采样（体素下采样+最远点采样）；2)中心化处理（减去质心）；3)PCA对齐（与特征向量对齐）；4)符号消歧（使用几何驱动分数函数解决PCA符号歧义）；5)生成规范表示（在消歧后的坐标系中重新表达点云）。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)提出PCA点云（PPC）标准化框架；2)设计几何驱动的消歧方法解决PCA符号歧义；3)提供理论保证证明对刚体变换的不变性；4)模块化设计可集成到任何PC-RL算法中。相比之前工作，PPC比域随机化更高效且有理论保证；比现有旋转不变分析方法解决符号歧义问题；比传统PCA方法确保确定性输出；比其他点云方法更适合强化学习任务。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '本文提出了一种基于PCA的点云标准化方法（PPC），通过解决PCA的符号歧义问题，实现了点云表示对相机视角变化的鲁棒性，显著提高了点云强化学习在未见视角下的零样本泛化能力。'}


### 论文摘要

Reinforcement Learning (RL) from raw visual input has achieved impressive successes in recent years, yet it remains fragile to out-of-distribution variations such as changes in lighting, color, and viewpoint. Point Cloud Reinforcement Learning (PC-RL) offers a promising alternative by mitigating appearance-based brittleness, but its sensitivity to camera pose mismatches continues to undermine reliability in realistic settings. To address this challenge, we propose PCA Point Cloud (PPC), a canonicalization framework specifically tailored for downstream robotic control. PPC maps point clouds under arbitrary rigid-body transformations to a unique canonical pose, aligning observations to a consistent frame, thereby substantially decreasing viewpoint-induced inconsistencies. In our experiments, we show that PPC improves robustness to unseen camera poses across challenging robotic tasks, providing a principled alternative to domain randomization.

---

## 97. STAR-Bench: Probing Deep Spatio-Temporal Reasoning as Audio 4D Intelligence

**论文链接:** [http://arxiv.org/abs/2510.24693v1](http://arxiv.org/abs/2510.24693v1)

**作者:** Zihan Liu, Zhikang Niu, Qiuyang Xiao, Zhisheng Zheng, Ruoqi Yuan, Yuhang Zang, Yuhang Cao, Xiaoyi Dong, Jianze Liang, Xie Chen, Leilei Sun, Dahua Lin, Jiaqi Wang

**发布时间:** 2025-10-28

**备注:** Homepage: https://internlm.github.io/StarBench/

### GPT解析

### 总结

该论文提出了STAR-Bench，一个新的音频基准测试，用于评估模型对声音在时间和三维空间中的动态推理能力。与现有基准测试不同，STAR-Bench专注于文本描述难以捕捉的精细感知推理。

### 背景

尽管多模态大语言模型和大型音频语言模型发展迅速，但现有的音频基准测试主要测试可以从文本标题中恢复的语义，掩盖了在精细感知推理方面的不足。

### 目的

正式定义音频4D智能（即对时间和三维空间中声音动态的推理），并引入STAR-Bench基准测试来衡量这种能力，揭示当前模型在理解物理世界方面的不足。

### 方法

STAR-Bench结合基础声学感知设置（六个属性）和整体时空推理设置（包括段重排序和空间任务）。数据收集使用程序合成和物理模拟的音频，以及包括人工注释的四阶段流程确保高质量样本。

### 主要发现

对19个模型的评估显示与人类存在显著差距，闭源模型受限于精细感知，而开源模型在感知、知识和推理方面都落后。STAR-Bench导致的准确性下降远大于先前基准测试（时间维度-31.5%，空间维度-35.2%）。

### 结论

STAR-Bench为开发具有更强大物理世界理解能力的未来模型提供了关键见解和明确的发展路径。

### 翻译

尽管多模态大语言模型和大型音频语言模型取得了快速进展，但现有的音频基准测试主要测试可以从文本标题中恢复的语义，掩盖了在精细感知推理方面的不足。我们将音频4D智能正式定义为对时间和三维空间中声音动态的推理，并引入STAR-Bench来衡量它。STAR-Bench结合了基础声学感知设置（绝对和相对条件下的六个属性）和整体时空推理设置，包括连续和离散过程的段重排序以及跨越静态定位、多源关系和动态轨迹的空间任务。我们的数据收集流程使用两种方法确保高质量样本。对于基础任务，我们使用程序合成和物理模拟的音频。对于整体数据，我们遵循包括人工注释和基于人类表现的最终选择在内的四阶段流程。与先前仅通过标题回答略微降低准确性的基准测试不同，STAR-Bench导致更大的下降（时间维度-31.5%，空间维度-35.2%），证明了其对语言难以描述线索的关注。对19个模型的评估显示与人类存在显著差距，并揭示了能力层次结构：闭源模型受限于精细感知，而开源模型在感知、知识和推理方面都落后。我们的STAR-Bench为开发具有更强大物理世界理解能力的未来模型提供了关键见解和明确的发展路径。


### 论文摘要

Despite rapid progress in Multi-modal Large Language Models and Large Audio-Language Models, existing audio benchmarks largely test semantics that can be recovered from text captions, masking deficits in fine-grained perceptual reasoning. We formalize audio 4D intelligence that is defined as reasoning over sound dynamics in time and 3D space, and introduce STAR-Bench to measure it. STAR-Bench combines a Foundational Acoustic Perception setting (six attributes under absolute and relative regimes) with a Holistic Spatio-Temporal Reasoning setting that includes segment reordering for continuous and discrete processes and spatial tasks spanning static localization, multi-source relations, and dynamic trajectories. Our data curation pipeline uses two methods to ensure high-quality samples. For foundational tasks, we use procedurally synthesized and physics-simulated audio. For holistic data, we follow a four-stage process that includes human annotation and final selection based on human performance. Unlike prior benchmarks where caption-only answering reduces accuracy slightly, STAR-Bench induces far larger drops (-31.5\% temporal, -35.2\% spatial), evidencing its focus on linguistically hard-to-describe cues. Evaluating 19 models reveals substantial gaps compared with humans and a capability hierarchy: closed-source models are bottlenecked by fine-grained perception, while open-source models lag across perception, knowledge, and reasoning. Our STAR-Bench provides critical insights and a clear path forward for developing future models with a more robust understanding of the physical world.

---

## 98. DynaStride: Dynamic Stride Windowing with MMCoT for Instructional Multi-Scene Captioning

**论文链接:** [http://arxiv.org/abs/2510.23907v1](http://arxiv.org/abs/2510.23907v1)

**作者:** Eddison Pham, Prisha Priyadarshini, Adrian Maliackel, Kanishk Bandi, Cristian Meo, Kevin Zhu

**发布时间:** 2025-10-27

**备注:** 16 pages, 15 figures, 5 Tables, submitted to AAAI AI4ED Workshop 2026

### GPT解析

### 总结

本文介绍了一种名为DynaStride的管道方法，用于生成教学视频中场景级别的连贯字幕，无需手动场景分割。该方法通过自适应帧采样和多模态窗口捕获关键转换，采用多模态思维链过程生成动作-对象对，并使用动态步长窗口选择算法进行融合，最终生成整合视觉语义和时间推理的场景级字幕。

### 背景

教学视频中的场景级字幕通过理解视觉线索和时间结构来增强学习。将视觉线索与文本指导相结合支持程序学习和多模态推理，为技能获取提供丰富上下文。然而，未能捕捉这种结构的字幕可能缺乏连贯性和质量，造成混淆并破坏视频的教育意图。

### 目的

解决现有字幕生成方法无法有效捕捉教学视频中时间结构和视觉语义的问题，开发一种能够生成连贯、高质量场景级字幕的方法，而无需手动场景分割。

### 方法

作者提出了DynaStride管道，使用YouCookII数据集的场景注释，执行自适应帧采样和多模态窗口化来捕获每个场景内的关键转换。然后采用多模态思维链过程产生多个动作-对象对，并使用动态步长窗口选择算法进行精炼和融合，该算法自适应地平衡时间上下文和冗余。最终的场景级字幕将视觉语义和时间推理整合在一个教学字幕中。

### 主要发现

与包括VLLaMA3和GPT-4o在内的强大基线相比，DynaStride在基于N-gram的指标(BLEU, METEOR)和语义相似性度量(BERTScore, CLIPScore)上均表现出一致的性能提升。定性分析进一步表明，DynaStride生成的字幕在时间连贯性和信息性方面更优。

### 结论

DynaStride为改进AI驱动的教学内容生成提供了有希望的方向，能够生成更连贯、信息更丰富的场景级字幕，有助于提高教学视频的学习效果。

### 翻译

教学视频中的场景级字幕可以通过要求理解视觉线索和时间结构来增强学习。通过将视觉线索与文本指导相一致，这种理解支持程序学习和多模态推理，为技能获取提供更丰富的上下文。然而，未能捕捉这种结构的字幕可能缺乏连贯性和质量，这可能造成混淆并破坏视频的教育意图。为了解决这一差距，我们引入了DynaStride，一个无需手动场景分割即可生成连贯场景级字幕的管道。使用YouCookII数据集的场景注释，DynaStride执行自适应帧采样和多模态窗口化来捕获每个场景内的关键转换。然后，它采用多模态思维链过程产生多个动作-对象对，这些对使用动态步长窗口选择算法进行精炼和融合，该算法自适应地平衡时间上下文和冗余。最终的场景级字幕将视觉语义和时间推理整合在一个教学字幕中。与包括VLLaMA3和GPT-4o在内的强大基线的经验评估表明，在基于N-gram的指标(BLEU, METEOR)和语义相似性度量(BERTScore, CLIPScore)上均表现出一致的提升。定性分析进一步表明，DynaStride生成的字幕在时间连贯性和信息性方面更优，这表明改进AI驱动的教学内容生成是一个有希望的方向。


### 论文摘要

Scene-level captioning in instructional videos can enhance learning by requiring an understanding of both visual cues and temporal structure. By aligning visual cues with textual guidance, this understanding supports procedural learning and multimodal reasoning, providing a richer context for skill acquisition. However, captions that fail to capture this structure may lack coherence and quality, which can create confusion and undermine the video's educational intent. To address this gap, we introduce DynaStride, a pipeline to generate coherent, scene-level captions without requiring manual scene segmentation. Using the YouCookII dataset's scene annotations, DynaStride performs adaptive frame sampling and multimodal windowing to capture key transitions within each scene. It then employs a multimodal chain-of-thought process to produce multiple action-object pairs, which are refined and fused using a dynamic stride window selection algorithm that adaptively balances temporal context and redundancy. The final scene-level caption integrates visual semantics and temporal reasoning in a single instructional caption. Empirical evaluations against strong baselines, including VLLaMA3 and GPT-4o, demonstrate consistent gains on both N-gram-based metrics (BLEU, METEOR) and semantic similarity measures (BERTScore, CLIPScore). Qualitative analyses further show that DynaStride produces captions that are more temporally coherent and informative, suggesting a promising direction for improving AI-powered instructional content generation.

---

## 99. VideoTG-R1: Boosting Video Temporal Grounding via Curriculum Reinforcement Learning on Reflected Boundary Annotations

**论文链接:** [http://arxiv.org/abs/2510.23397v1](http://arxiv.org/abs/2510.23397v1)

**作者:** Lu Dong, Haiyu Zhang, Han Lin, Ziang Yan, Xiangyu Zeng, Hongjie Zhang, Yifei Huang, Yi Wang, Zhen-Hua Ling, Limin Wang, Yali Wang

**发布时间:** 2025-10-27

### GPT解析

### 总结

VideoTG-R1是一种新型课程强化学习框架，通过反射边界标注解决视频时间定位中的训练样本质量和难度问题，实现了数据高效训练。

### 背景

视频时间定位(VTG)是根据语言查询在视频中定位精确片段的基础挑战。多模态大型语言模型(MLLMs)通过强化学习(RL)在VTG方面显示出潜力，但忽视了训练样本质量和难度带来的挑战。

### 目的

解决VTG训练中部分标注样本和难以定位样本带来的问题，提高训练效率。

### 方法

提出VideoTG-R1框架，包含边界反射代理(识别并过滤部分标注样本)和难度估计代理(评估样本难度并设计课程RL策略)，实现数据高效训练。

### 主要发现

仅使用10%的训练样本和21%的计算预算，VideoTG-R1在组相对策略优化(GRPO)和监督微调(SFT)下都优于全数据对应方法。

### 结论

VideoTG-R1通过解决训练样本质量和难度问题，实现了在VTG和基于视频的问答任务上的有效性能提升。

### 翻译

视频时间定位(VTG)旨在根据语言查询在视频中定位精确片段，这是视频理解中的一个基础挑战。虽然最近的多模态大型语言模型(MLLMs)通过强化学习(RL)在解决VTG方面显示出潜力，但它们忽视了训练样本质量和难度带来的挑战。(1)部分标注样本：许多样本包含超出标注区间的相关片段，引入了模糊监督。(2)难以定位的样本：零样本性能差的样本在RL训练中产生持续低且不可区分的奖励，在多个输出中没有明显偏好，从而阻碍学习效率。为解决这些挑战，我们提出VideoTG-R1，一个具有反射边界标注的新型课程RL框架，实现数据高效训练。具体来说，我们提出边界反射代理，利用MLLMs预测标注区间外的查询相关时间戳，使我们能够识别并过滤部分标注样本，从而减少模糊性。此外，我们引入难度估计代理来评估每个样本的训练难度，并设计课程RL策略，根据训练步骤动态掩盖难以定位样本的视频，降低训练难度并提供更清晰的偏好。在VTG和基于视频的问答任务上的实验证明了我们方法的有效性。值得注意的是，仅使用10%的训练样本和21%的计算预算，VideoTG-R1在组相对策略优化(GRPO)和监督微调(SFT)下都优于全数据对应方法。代码可在https://github.com/ldong1111/VideoTG-R1获取。


### 论文摘要

Video temporal grounding (VTG) aims to locate precise segments in videos based on language queries, which is a fundamental challenge in video understanding. While recent Multimodal Large Language Models (MLLMs) have shown promise in tackling VTG through reinforcement learning (RL), they overlook the challenges arising from both the quality and difficulty of training samples. (1) Partially annotated samples. Many samples contain relevant segments beyond the annotated interval, introducing ambiguous supervision. (2) Hard-to-ground samples. Samples with poor zero-shot performance produce consistently low and indistinguishable rewards during RL training, exhibiting no clear preference among multiple outputs and thus hindering learning efficiency. To address these challenges, we propose VideoTG-R1, a novel curriculum RL framework with reflected boundary annotations, enabling data-efficient training. Specifically, we propose a Boundary Reflection Agent that utilizes MLLMs to predict query-relevant timestamps outside the annotated intervals, allowing us to identify and filter out partially annotated samples, thereby reducing ambiguity. Furthermore, we introduce a Difficulty Estimation Agent to assess the training difficulty of each sample and design a curriculum RL strategy that dynamically masks the videos of hard-to-ground samples according to the training steps, easing the training difficulty and providing clearer preference. Experiments on the VTG and grounded VideoQA tasks demonstrate the effectiveness of our method. Remarkably, with only 10% of the training samples and 21% of the computational budget, VideoTG-R1 outperforms full-data counterparts under both group relative policy optimization (GRPO) and supervised fine-tuning (SFT). The code is available at https://github.com/ldong1111/VideoTG-R1.

---

## 100. Evaluation of Vision-LLMs in Surveillance Video

**论文链接:** [http://arxiv.org/abs/2510.23190v1](http://arxiv.org/abs/2510.23190v1)

**作者:** Pascal Benschop, Cristian Meo, Justin Dauwels, Jelte P. Mense

**发布时间:** 2025-10-27

**备注:** Accepted as poster in the NeurIPS 2025 Workshop on Space in Vision,  Language, and Embodied AI

### GPT解析

### 总结

本研究探讨了视觉语言模型在异常动作识别中的空间推理能力，将其作为零样本、语言基础的任务，解决了从稀疏2D视频中解释动态3D场景的具身感知挑战。

### 背景

社会中摄像机的广泛应用产生了大量视频数据，远远超出人工监控能力，这对公共安全构成重大挑战，因为及时检测异常或犯罪事件对有效预防和响应至关重要。

### 目的

研究视觉语言模型（VLMs）的空间推理能力，探索小型预训练视觉-语言模型作为空间基础的零样本异常检测器的可行性，并评估其在提示和隐私保护条件下的表现。

### 方法

将视频转换为文本描述并通过文本蕴含对标签进行评分，在UCF-Crime和RWF-2000数据集上评估四个开放模型，研究少样本示例和隐私过滤器对模型性能的影响。

### 主要发现

少样本示例可以提高某些模型的准确性但可能增加误报；隐私过滤器（尤其是全身GAN变换）会引入不一致性降低准确性；当前视觉-语言模型在简单、空间显著事件上表现良好，但在处理嘈杂空间线索和身份模糊时表现不佳。

### 结论

提出了加强空间基础的具体路径，包括结构感知提示、跨片段轻量级空间记忆、描述过程中的场景图或3D姿态先验，以及保留动作相关几何形状的隐私方法，使零样本、语言基础的管道成为具身、真实世界视频理解的适应性构建块。

### 翻译

我们社会中摄像机的广泛应用产生了大量视频数据，远远超出了人工监控的能力。这对公共安全和安全构成了关键挑战，因为及时检测异常或犯罪事件对于有效预防和应对至关重要。具身代理识别意外事件的能力根本上与其空间推理能力相关。本文通过将异常动作识别构架为零样本、语言基础的任务，研究了视觉语言模型（VLMs）的空间推理，解决了从稀疏2D视频中解释动态3D场景的具身感知挑战。具体来说，我们调查了小型预训练视觉-语言模型是否可以通过将视频转换为文本描述并通过文本蕴含对标签进行评分，作为空间基础的零样本异常检测器。我们在提示和隐私保护条件下，在UCF-Crime和RWF-2000数据集上评估了四个开放模型。少样本示例可以提高某些模型的准确性，但可能增加误报，而隐私过滤器——尤其是全身GAN变换——会引入不一致性，降低准确性。这些结果展示了当前视觉-语言模型在哪些方面成功（简单、空间显著事件）和哪些方面失败（嘈杂的空间线索、身份模糊）。展望未来，我们概述了加强空间基础的具体路径，无需任务特定训练：结构感知提示、跨片段轻量级空间记忆、描述过程中的场景图或3D姿态先验，以及保留动作相关几何形状的隐私方法。这使零样本、语言基础的管道成为具身、真实世界视频理解的适应性构建块。我们用于评估VLMs的实现已在以下公开可用：https://github.com/pascalbenschopTU/VLLM_AnomalyRecognition

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决如何评估视觉-语言模型在监控视频中进行零样本异常行为识别的问题。这个问题很重要，因为社会上有大量摄像头产生的视频数据远远超过人类监控能力，及时检测异常或犯罪事件对公共安全和有效预防至关重要。此外，现有的公共异常行为识别数据集有限，仅在这些数据集上训练的模型可能无法很好地泛化到新的异常类型。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者思考过程是认识到传统视频异常检测依赖监督学习，需要大量带注释的数据集，成本高且难以识别新异常。因此他们设计了一个零样本框架，将异常分类重新构建为语言基础推理任务而非像素到标签映射。该方法借鉴了现有VLM的语义推理和世界知识，但不同于之前的AnomalyCLIP、LAVAD等方法，它不需要任务特定微调，而是专注于纯零样本异常检测。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是将异常分类转化为语言基础推理任务，利用大型预训练视觉-语言模型的语义推理能力。整体流程分为两步：1)文本描述生成：视觉-LLM处理视频输入，基于视觉和提示生成描述性文本；2)零样本分类：使用预训练的NLI分类器评估文本与候选异常类别的逻辑蕴含程度，选择最高分数的类别作为结果。整个过程无需对模型进行梯度更新，实现了真正的零样本学习。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：系统评估小型视觉-LLMs在零样本异常识别中的能力；设计多种提示策略实验；研究隐私保护变换对模型性能的影响；提出改进空间推理的具体方法。与之前工作不同，该方法不需要大量标注数据，专注于纯零样本检测，首次系统评估了隐私保护变换对模型的影响，并揭示了提示技术与隐私过滤器之间的关键权衡。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '这篇论文通过系统评估视觉-LLMs在零样本异常检测中的能力，揭示了提示技术和隐私过滤器之间的关键权衡，为设计更安全、更有效的视频理解系统提供了实用建议。'}


### 论文摘要

The widespread use of cameras in our society has created an overwhelming amount of video data, far exceeding the capacity for human monitoring. This presents a critical challenge for public safety and security, as the timely detection of anomalous or criminal events is crucial for effective response and prevention. The ability for an embodied agent to recognize unexpected events is fundamentally tied to its capacity for spatial reasoning. This paper investigates the spatial reasoning of vision-language models (VLMs) by framing anomalous action recognition as a zero-shot, language-grounded task, addressing the embodied perception challenge of interpreting dynamic 3D scenes from sparse 2D video. Specifically, we investigate whether small, pre-trained vision--LLMs can act as spatially-grounded, zero-shot anomaly detectors by converting video into text descriptions and scoring labels via textual entailment. We evaluate four open models on UCF-Crime and RWF-2000 under prompting and privacy-preserving conditions. Few-shot exemplars can improve accuracy for some models, but may increase false positives, and privacy filters -- especially full-body GAN transforms -- introduce inconsistencies that degrade accuracy. These results chart where current vision--LLMs succeed (simple, spatially salient events) and where they falter (noisy spatial cues, identity obfuscation). Looking forward, we outline concrete paths to strengthen spatial grounding without task-specific training: structure-aware prompts, lightweight spatial memory across clips, scene-graph or 3D-pose priors during description, and privacy methods that preserve action-relevant geometry. This positions zero-shot, language-grounded pipelines as adaptable building blocks for embodied, real-world video understanding. Our implementation for evaluating VLMs is publicly available at: https://github.com/pascalbenschopTU/VLLM_AnomalyRecognition

---

## 101. Large-Model AI for Near Field Beam Prediction: A CNN-GPT2 Framework for 6G XL-MIMO

**论文链接:** [http://arxiv.org/abs/2510.22557v1](http://arxiv.org/abs/2510.22557v1)

**作者:** Wang Liu, Cunhua Pan, Hong Ren, Wei Zhang, Cheng-Xiang Wang, Jiangzhou Wang

**发布时间:** 2025-10-26

### GPT解析

### 总结

本文提出了一种基于CNN-GPT2的新型近场波束预测框架，用于解决毫米波通信中极大规模天线阵列在高移动性场景下的近场波束预测挑战。

### 背景

毫米波通信中极大规模天线阵列在高移动性场景下的应用凸显了近场波束预测的重要性。与传统远场假设不同，近场波束预测需要在角度和距离域联合采样码本，导致导频开销大幅增加。此外，最优近场波束指数表现出突发的非线性动态特性，对时间建模构成挑战。

### 目的

解决近场波束预测中的挑战，设计一个有效的近场波束预测框架，以应对导频开销增加和波束指数非线性动态特性问题。

### 方法

提出了一种基于CNN-GPT2的新型近场波束预测框架。具体包括：设计上行链路导频传输策略，通过宽波束模拟预编码和频率变化数字预编码实现高效信道探测；接收的导频信号经过预处理后，通过基于CNN的特征提取器；然后通过GPT-2模型捕获多个帧之间的时间依赖性，以端到端方式直接预测近场波束指数。

### 主要发现

CNN-GPT2框架能够有效处理近场波束预测的挑战，所提出的导频传输策略实现了高效信道探测，该方法能够捕获时间依赖性并直接预测近场波束指数。

### 结论

基于CNN-GPT2的近场波束预测框架为解决毫米波通信中极大规模天线阵列在高移动性场景下的近场波束预测问题提供了有效方案。

### 翻译

毫米波通信中极大规模天线阵列的出现，尤其是在高移动性场景下，凸显了近场波束预测的重要性。与传统远场假设不同，近场波束预测需要同时在角度和距离域采样的码本，这导致导频开销大幅增加。此外，与远场情况下最优波束演化的时间平滑性不同，最优近场波束指数由于其同时依赖于用户角度和距离而表现出突发和非线性动态特性，给时间建模带来了重大挑战。为应对这些挑战，我们提出了一种新颖的基于卷积神经网络-生成预训练Transformer 2（CNN-GPT2）的近场波束预测框架。具体而言，设计了一种上行链路导频传输策略，通过宽波束模拟预编码和频率变化数字预编码实现高效信道探测。接收的导频信号经过预处理后，通过基于CNN的特征提取器，然后由GPT-2模型捕获多个帧之间的时间依赖性，并以端到端方式直接预测近场波束指数。


### 论文摘要

The emergence of extremely large-scale antenna arrays (ELAA) in millimeter-wave (mmWave) communications, particularly in high-mobility scenarios, highlights the importance of near-field beam prediction. Unlike the conventional far-field assumption, near-field beam prediction requires codebooks that jointly sample the angular and distance domains, which leads to a dramatic increase in pilot overhead. Moreover, unlike the far-field case where the optimal beam evolution is temporally smooth, the optimal near-field beam index exhibits abrupt and nonlinear dynamics due to its joint dependence on user angle and distance, posing significant challenges for temporal modeling. To address these challenges, we propose a novel Convolutional Neural Network-Generative Pre-trained Transformer 2 (CNN-GPT2) based near-field beam prediction framework. Specifically, an uplink pilot transmission strategy is designed to enable efficient channel probing through widebeam analog precoding and frequency-varying digital precoding. The received pilot signals are preprocessed and passed through a CNN-based feature extractor, followed by a GPT-2 model that captures temporal dependencies across multiple frames and directly predicts the near-field beam index in an end-to-end manner.

---

## 102. TERRA: A Transformer-Enabled Recursive R-learner for Longitudinal Heterogeneous Treatment Effect Estimation

**论文链接:** [http://arxiv.org/abs/2510.22407v1](http://arxiv.org/abs/2510.22407v1)

**作者:** Lei Shi, Sizhu Lu, Qiuran Lyu, Peng Ding, Nikos Vlassis

**发布时间:** 2025-10-25

**备注:** 27 pages, 4 figures

### GPT解析

### 总结

该研究提出了一种名为TERRA的新方法，用于解决纵向数据中异质性处理效应估计的挑战，特别是在时变干预情况下。

### 背景

在纵向数据中准确估计异质性处理效应对医疗保健、公共政策、教育和数字营销等领域的个性化决策至关重要。然而，时变干预带来了延续效应、时变异质性和处理后偏差等独特挑战，这些挑战无法通过标准HTE方法解决。

### 目的

开发一种能够处理时变干预带来的独特挑战（延续效应、时变异质性和处理后偏差）的方法，以准确估计纵向数据中的异质性处理效应。

### 方法

引入TERRA（Transformer-Enabled Recursive R-learner），包含两个主要组件：1）使用Transformer架构编码完整的处理特征历史，捕捉长期时间依赖性和延续效应；2）开发递归残差学习公式，将经典结构嵌套均值模型推广到参数规范之外，解决处理后偏差问题。

### 主要发现

在模拟和数据应用中，TERRA在HTE估计的准确性和稳定性方面始终优于强大的基线方法。

### 结论

将有原则的因果结构与高容量的序列模型相结合，对于纵向异质性处理效应估计具有重要价值。

### 翻译

在纵向环境中准确估计异质性处理效应(HTE)对于医疗保健、公共政策、教育和数字营销等领域的个性化决策至关重要。然而，时变干预带来了许多独特挑战，如延续效应、时变异质性和处理后偏差，这些问题标准HTE方法无法解决。为应对这些挑战，我们引入了TERRA（Transformer-Enabled Recursive R-learner），它促进具有灵活时间建模和学习的纵向HTE估计。TERRA有两个组件。首先，我们使用Transformer架构编码完整的处理特征历史，使表示长期时间依赖性和延续效应成为可能，从而更全面地捕捉个体和特定时间的处理效应变化。其次，我们开发了一种递归残差学习公式，将经典结构嵌套均值模型(SNMMs)推广到参数规范之外，解决了处理后偏差问题，同时减少了对功能假设的依赖。在模拟和数据应用中，TERRA在HTE估计的准确性和稳定性方面始终优于强大的基线方法，突显了将原则性因果结构与高容量序列模型相结合对纵向HTE的价值。


### 论文摘要

Accurately estimating heterogeneous treatment effects (HTE) in longitudinal settings is essential for personalized decision-making across healthcare, public policy, education, and digital marketing. However, time-varying interventions introduce many unique challenges, such as carryover effects, time-varying heterogeneity, and post-treatment bias, which are not addressed by standard HTE methods. To address these challenges, we introduce TERRA (Transformer-Enabled Recursive R-learner), which facilitates longitudinal HTE estimation with flexible temporal modeling and learning. TERRA has two components. First, we use a Transformer architecture to encode full treatment-feature histories, enabling the representation of long-range temporal dependencies and carryover effects, hence capturing individual- and time-specific treatment effect variation more comprehensively. Second, we develop a recursive residual-learning formulation that generalizes the classical structural nested mean models (SNMMs) beyond parametric specifications, addressing post-treatment bias while reducing reliance on functional assumptions. In simulations and data applications, TERRA consistently outperforms strong baselines in HTE estimation in both accuracy and stability, highlighting the value of combining principled causal structure with high-capacity sequence models for longitudinal HTE.

---

## 103. Human-Centric Anomaly Detection in Surveillance Videos Using YOLO-World and Spatio-Temporal Deep Learning

**论文链接:** [http://arxiv.org/abs/2510.22056v1](http://arxiv.org/abs/2510.22056v1)

**作者:** Mohammad Ali Etemadi Naeen, Hoda Mohammadzade, Saeed Bagheri Shouraki

**发布时间:** 2025-10-24

### GPT解析

### 总结

该研究提出了一种稳健的深度学习框架，通过结合以人为中心的预处理和时空建模来解决监控视频异常检测中的挑战，在UCF-Crime数据集上实现了92.41%的平均测试准确率。

### 背景

监控视频中的异常检测面临异常事件多样性、类别不平衡和场景依赖的视觉混乱等挑战。

### 目的

提出一个稳健的深度学习框架，整合以人为中心的预处理与时空建模，用于多类异常分类。

### 方法

使用YOLO-World识别人体实例，ByteTrack进行身份感知跟踪，通过高斯模糊抑制背景区域，使用InceptionV3进行空间特征提取，并用双向LSTM捕获时间动态进行序列级分类。

### 主要发现

在UCF-Crime五类子集上评估，三次独立试验中平均测试准确率达92.41%，每类F1分数均超过0.85，展示了对类别不平衡的强鲁棒性。

### 结论

前景聚焦的预处理显著增强了现实监控场景中的异常辨别能力。

### 翻译

监控视频中的异常检测由于异常事件的多样性、类别不平衡和场景依赖的视觉混乱而仍然是一项具有挑战性的任务。为了解决这些问题，我们提出了一个稳健的深度学习框架，该框架整合了以人为中心的预处理与时空建模，用于多类异常分类。我们的流程首先应用YOLO-World（一种开放词汇的视觉语言检测器）来识别原始视频片段中的人体实例，然后使用ByteTrack进行一致的身份感知跟踪。通过高斯模糊抑制检测边界框外的背景区域，有效减少场景特定的干扰，使模型专注于行为相关的前景内容。然后，经过精炼的帧由在ImageNet上预训练的InceptionV3网络处理进行空间特征提取，并使用双向LSTM（BiLSTM）捕获时间动态，进行序列级分类。在UCF-Crime数据集的五类子集（正常、入室盗窃、打架、纵火、爆炸）上评估，我们的方法在三次独立试验中平均测试准确率达到92.41%，每个类别的F1分数均超过0.85。全面的评估指标，包括混淆矩阵、ROC曲线和宏/加权平均值，展示了强大的泛化能力和对类别不平衡的鲁棒性。结果证实，前景聚焦的预处理显著增强了现实监控场景中的异常辨别能力。


### 论文摘要

Anomaly detection in surveillance videos remains a challenging task due to the diversity of abnormal events, class imbalance, and scene-dependent visual clutter. To address these issues, we propose a robust deep learning framework that integrates human-centric preprocessing with spatio-temporal modeling for multi-class anomaly classification. Our pipeline begins by applying YOLO-World - an open-vocabulary vision-language detector - to identify human instances in raw video clips, followed by ByteTrack for consistent identity-aware tracking. Background regions outside detected bounding boxes are suppressed via Gaussian blurring, effectively reducing scene-specific distractions and focusing the model on behaviorally relevant foreground content. The refined frames are then processed by an ImageNet-pretrained InceptionV3 network for spatial feature extraction, and temporal dynamics are captured using a bidirectional LSTM (BiLSTM) for sequence-level classification. Evaluated on a five-class subset of the UCF-Crime dataset (Normal, Burglary, Fighting, Arson, Explosion), our method achieves a mean test accuracy of 92.41% across three independent trials, with per-class F1-scores consistently exceeding 0.85. Comprehensive evaluation metrics - including confusion matrices, ROC curves, and macro/weighted averages - demonstrate strong generalization and resilience to class imbalance. The results confirm that foreground-focused preprocessing significantly enhances anomaly discrimination in real-world surveillance scenarios.

---

## 104. ViBED-Net: Video Based Engagement Detection Network Using Face-Aware and Scene-Aware Spatiotemporal Cues

**论文链接:** [http://arxiv.org/abs/2510.18016v2](http://arxiv.org/abs/2510.18016v2)

**作者:** Prateek Gothwal, Deeptimaan Banerjee, Ashis Kumer Biswas

**发布时间:** 2025-10-20

**备注:** 10 pages, 4 figures, 2 tables

### GPT解析

### 总结

这篇论文介绍了一种名为ViBED-Net的新型深度学习框架，用于从视频数据中检测学生在在线学习环境中的参与度。该模型通过结合面部表情和全场景上下文信息，利用双流架构和EfficientNetV2进行特征提取，并使用LSTM或Transformer进行时间建模。

### 背景

在线学习环境中的参与度检测对于提高学生成果和个性化教学至关重要。存在一个名为DAiSEE的大规模基准数据集，用于电子学习中情感状态识别。

### 目的

提出ViBED-Net（基于视频的参与度检测网络），一种新的深度学习框架，设计用于从视频数据评估学生参与度。

### 方法

采用双流架构，使用EfficientNetV2进行空间特征提取，处理面部裁剪和完整视频帧来捕捉面部表情和全场景上下文。使用两种时间建模策略分析特征：长短期记忆（LSTM）网络和Transformer编码器。在DAiSEE数据集上评估模型，并应用有针对性的数据增强技术提高在代表性不足的参与度类别上的性能。

### 主要发现

ViBED-Net与LSTM的组合达到73.43%的准确率，优于现有的最先进方法。结合面部感知和场景感知的时空线索显著提高了参与度检测的准确性。

### 结论

ViBED-Net的模块化设计使其具有灵活性，可应用于教育、用户体验研究和内容个性化。该工作通过为现实世界的参与度分析提供可扩展的高性能解决方案，推动了基于视频的情感计算发展。项目源代码可在GitHub上获取。

### 翻译

在线学习环境中的参与度检测对于提高学生成果和个性化教学至关重要。我们提出了ViBED-Net（基于视频的参与度检测网络），一种新的深度学习框架，旨在通过双流架构从视频数据评估学生参与度。ViBED-Net通过EfficientNetV2处理面部裁剪和完整视频帧来捕捉面部表情和全场景上下文，用于空间特征提取。然后使用两种时间建模策略分析这些特征：长短期记忆（LSTM）网络和Transformer编码器。我们的模型在DAiSEE数据集上进行了评估，这是一个电子学习中情感状态识别的大规模基准。为了提高在代表性不足的参与度类别上的性能，我们应用了有针对性的数据增强技术。在测试的变体中，ViBED-Net与LSTM结合实现了73.43%的准确率，优于现有的最先进方法。ViBED-Net证明，结合面部感知和场景感知的时空线索显著提高了参与度检测的准确性。其模块化设计使其具有灵活性，可应用于教育、用户体验研究和内容个性化。这项工作通过为现实世界的参与度分析提供可扩展的高性能解决方案，推动了基于视频的情感计算发展。该项目的源代码可在https://github.com/prateek-gothwal/ViBED-Net获取。


### 论文摘要

Engagement detection in online learning environments is vital for improving student outcomes and personalizing instruction. We present ViBED-Net (Video-Based Engagement Detection Network), a novel deep learning framework designed to assess student engagement from video data using a dual-stream architecture. ViBED-Net captures both facial expressions and full-scene context by processing facial crops and entire video frames through EfficientNetV2 for spatial feature extraction. These features are then analyzed over time using two temporal modeling strategies: Long Short-Term Memory (LSTM) networks and Transformer encoders. Our model is evaluated on the DAiSEE dataset, a large-scale benchmark for affective state recognition in e-learning. To enhance performance on underrepresented engagement classes, we apply targeted data augmentation techniques. Among the tested variants, ViBED-Net with LSTM achieves 73.43\% accuracy, outperforming existing state-of-the-art approaches. ViBED-Net demonstrates that combining face-aware and scene-aware spatiotemporal cues significantly improves engagement detection accuracy. Its modular design allows flexibility for application across education, user experience research, and content personalization. This work advances video-based affective computing by offering a scalable, high-performing solution for real-world engagement analysis. The source code for this project is available on https://github.com/prateek-gothwal/ViBED-Net .

---

