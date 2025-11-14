# 今日论文推荐 - 2025-11-14

共 2 篇论文

---

## 1. Xiaoice: Training-Free Video Understanding via Self-Supervised Spatio-Temporal Clustering of Semantic Features

**论文链接:** [http://arxiv.org/abs/2510.16781v2](http://arxiv.org/abs/2510.16781v2)

**作者:** Shihao Ji, Zihui Song

**发布时间:** 2025-10-19

**备注:** This paper is being withdrawn because we have identified a significant error in the implementation of our self-supervised clustering approach. Specifically, our feature aggregation step inadvertently leaked temporal information across frames, which violates the core assumption of our training-free method. We sincerely apologize to the research community

### GPT解析

### 总结

本文提出了一种新颖的无需训练的视频理解框架，通过结合预训练视觉语言模型(VLMs)和经典机器学习算法，实现了视频内容的零样本自动化结构分析。

### 背景

大型视觉语言模型在静态图像上的零样本推理能力已相当出色，但这些能力尚未完全应用到视频领域。传统视频理解模型通常依赖大量标注数据进行特定任务训练，过程昂贵且难以扩展。

### 目的

引入一种无需训练的视频理解框架，绕过端到端训练过程，协同结合预训练VLM的语义先验与经典机器学习算法进行模式发现。

### 方法

将视频理解重新构建为高维语义特征空间中的自监督时空聚类问题：1)使用预训练VLM的冻结视觉编码器将视频转换为语义特征轨迹；2)采用核时间分段(KTS)技术将特征流分割为语义一致的事件段；3)进行无监督密度聚类识别重复场景；4)选择代表性关键帧并利用VLM生成文本描述，形成多模态摘要。

### 主要发现

该方法为视频内容的零样本、自动化结构分析提供了有效、可解释且与模型无关的途径。

### 结论

通过协同结合预训练VLM和经典机器学习算法，可以无需训练就能实现视频理解，并自动生成视频内容的多模态结构化摘要。

### 翻译

大型视觉语言模型(VLMs)在静态图像上显著的零样本推理能力尚未完全转化为视频领域。传统视频理解模型通常依赖于在标注数据集上进行大量、特定任务的训练，这一过程既昂贵又难以扩展。本文介绍了一种新颖的、无需训练的视频理解框架，通过协同结合预训练VLM的丰富语义先验与用于模式发现的经典机器学习算法，绕过了端到端训练。我们的核心思想是将视频理解重新构建为高维语义特征空间中的自监督时空聚类问题。所提出的管道首先使用预训练VLM的冻结视觉编码器将视频流转换为语义特征轨迹。随后，我们采用核时间分段(KTS)这种稳健的机器学习技术，将连续的特征流分割成离散的、语义一致的事件段。这些段然后经受无监督的基于密度的聚类，以识别视频中重复出现的宏观场景和主题。通过从每个发现的聚类中选择代表性关键帧，并利用VLM的生成能力进行文本描述，我们的框架自动生成视频内容的结构化、多模态摘要。这种方法为视频内容的零样本、自动化结构分析提供了有效、可解释且与模型无关的途径。


### 论文摘要

The remarkable zero-shot reasoning capabilities of large-scale Visual Language Models (VLMs) on static images have yet to be fully translated to the video domain. Conventional video understanding models often rely on extensive, task-specific training on annotated datasets, a process that is both costly and limited in scalability. This paper introduces a novel, training-free framework for video understanding that circumvents end-to-end training by synergistically combining the rich semantic priors of pre-trained VLMs with classic machine learning algorithms for pattern discovery. Our core idea is to reframe video understanding as a self-supervised spatio-temporal clustering problem within a high-dimensional semantic feature space. The proposed pipeline first transforms a video stream into a semantic feature trajectory using the frozen visual encoder of a pre-trained VLM. Subsequently, we employ Kernel Temporal Segmentation (KTS), a robust machine learning technique, to partition the continuous feature stream into discrete, semantically coherent event segments. These segments are then subjected to unsupervised density-based clustering to identify recurring macroscopic scenes and themes throughout the video. By selecting representative keyframes from each discovered cluster and leveraging the VLM's generative capabilities for textual description, our framework automatically produces a structured, multi-modal summary of the video content. This approach provides an effective, interpretable, and model-agnostic pathway for zero-shot, automated structural analysis of video content.

---

## 2. MTP: Exploring Multimodal Urban Traffic Profiling with Modality Augmentation and Spectrum Fusion

**论文链接:** [http://arxiv.org/abs/2511.10218v1](http://arxiv.org/abs/2511.10218v1)

**作者:** Haolong Xiang, Peisi Wang, Xiaolong Xu, Kun Yi, Xuyun Zhang, Quanzheng Sheng, Amin Beheshti, Wei Fan

**发布时间:** 2025-11-13

### GPT解析

### 总结

该研究提出了一种名为MTP的新型多模态框架，通过数值、视觉和文本三种角度学习城市交通信号的多模态特征，以解决现有单模态方法忽视多模态异构城市数据语义信息的问题

### 背景

随着现代快速城市化，来自各种传感器的交通信号在监测城市状态方面发挥着重要作用，为安全出行、减少交通拥堵和优化城市出行提供了坚实基础

### 目的

解决现有交通信号建模方法依赖原始数据模式、忽视多模态异构城市数据语义信息的问题，实现对交通信号的全面理解和复杂交通动态的准确预测

### 方法

提出MTP多模态框架，包含三个分支：1)视觉增强将原始模态转换为频率图像和周期性图像；2)基于特定主题、背景信息和项目描述增强描述性文本；3)利用频率多层感知器在原始模态上学习；并通过分层对比学习融合三种模态的频谱

### 主要发现

在六个真实世界数据集上的实验表明，所提出的方法与最先进的方法相比具有优越性能

### 结论

多模态框架能够有效整合数值、视觉和文本信息，提高交通信号理解和预测的准确性

### 翻译

随着现代快速城市化的推进，来自各种传感器的交通信号在监测城市状态方面发挥着重要作用，为保障安全出行、减少交通拥堵和优化城市出行提供了坚实基础。大多数现有的交通信号建模方法往往依赖于原始数据模式，即城市传感器的直接数值读数。然而，这种单模态方法忽视了多模态异构城市数据中不同角度存在的语义信息，这妨碍了对交通信号的全面理解，并限制了复杂交通动态的准确预测。为解决这一问题，我们提出了一种名为MTP（城市交通分析多模态框架）的新型框架，通过数值、视觉和文本角度学习多模态特征。三个分支驱动城市交通信号学习在频域的多模态视角，而频率学习策略则精细提炼信息以供提取。具体而言，我们首先对交通信号进行视觉增强，将原始模态转换为频率图像和周期性图像用于视觉学习。同时，我们基于特定主题、背景信息和项目描述对交通信号的描述性文本进行增强，用于文本学习。为补充数值信息，我们利用频率多层感知器在原始模态上进行学习。我们设计了三个分支上的分层对比学习来融合三种模态的频谱。最后，在六个真实世界数据集上的大量实验表明，与最先进的方法相比具有优越性能。


### 论文摘要

With rapid urbanization in the modern era, traffic signals from various sensors have been playing a significant role in monitoring the states of cities, which provides a strong foundation in ensuring safe travel, reducing traffic congestion and optimizing urban mobility. Most existing methods for traffic signal modeling often rely on the original data modality, i.e., numerical direct readings from the sensors in cities. However, this unimodal approach overlooks the semantic information existing in multimodal heterogeneous urban data in different perspectives, which hinders a comprehensive understanding of traffic signals and limits the accurate prediction of complex traffic dynamics. To address this problem, we propose a novel \textit{M}ultimodal framework, \textit{MTP}, for urban \textit{T}raffic \textit{P}rofiling, which learns multimodal features through numeric, visual, and textual perspectives. The three branches drive for a multimodal perspective of urban traffic signal learning in the frequency domain, while the frequency learning strategies delicately refine the information for extraction. Specifically, we first conduct the visual augmentation for the traffic signals, which transforms the original modality into frequency images and periodicity images for visual learning. Also, we augment descriptive texts for the traffic signals based on the specific topic, background information and item description for textual learning. To complement the numeric information, we utilize frequency multilayer perceptrons for learning on the original modality. We design a hierarchical contrastive learning on the three branches to fuse the spectrum of three modalities. Finally, extensive experiments on six real-world datasets demonstrate superior performance compared with the state-of-the-art approaches.

---

