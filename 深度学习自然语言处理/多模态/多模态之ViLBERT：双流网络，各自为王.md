通篇读完这个论文，需要解决如下问题：

1. **ViLBERT架构是什么样子的？**
2. **ViLBERT预训练任务是什么？**
3. **ViLBERT实现细节有哪些？**

我之前写了两个多模态基础的文章，没看过的同学可以先看看这两个文章：

分别是 [**在解决多模态任务的时候需要重点解决的6个问题**](http://mp.weixin.qq.com/s?__biz=MzIyNTY1MDUwNQ==&mid=2247485868&idx=1&sn=5b2e76bac59985b7ffd633b48e0a024f&chksm=e87d3b8adf0ab29c110d68969576dc66b4e179de4e282c800574928826797f1396be86cc777b&scene=21#wechat_redirect) 和 [**如何把BERT的两种预训练任务扩展到多模态数据中去**](http://mp.weixin.qq.com/s?__biz=MzIyNTY1MDUwNQ==&mid=2247485872&idx=1&sn=3b4efdca5299776b07c9b470a0557365&chksm=e87d3b96df0ab2808ea69c86043e48966262a7a1ff70e6a0883dfbb8bc52c0f02cabef307bf2&scene=21#wechat_redirect)；

### 1. ViLBERT架构是什么样子的？

首先我们来聊第一个问题：ViLBERT架构是什么样子的？

直接看图：

![img](https://mmbiz.qpic.cn/sz_mmbiz_png/LU88NSfAnCxS5IMicOzwe7yf5TSOvicuTNCAmWUxolXZXic61QBxQsE0I7iaSMMgBnic4O9icAcF5QXK0tg30ouia0ic2g/640?wx_fmt=png)

这个图其实很不错，我简单来概述一下，如下：

首先ViLBERT包含两个并行的流，上面的那个是图片流，下面那个是文本流；

每个流是由一些TRM Blocks和  co-attentional TRM layers【Co-TRM】组成；

需要注意的是TRM Blocks 和Co-TRM 可以是多层的；

这里面最主要的部分其实就是这个Co-TRM；

在那个虚线框中，我们可以看到Co-TRM有两个部分，真正的Co-TRM和后连接的TRM；

首先我们要明确，从图片流前半部分【未交互之前】出来的是一个个图片regions的embeddings；

从文本流前半部分出来的是一个个文本tokens的embeddings；【需要注意的是文本这有一个L-K X的符号，其实代表的就是构建多层的TRM，在本文就是一个BERT-Base】；

知道各自流前半部分出来的是什么之后，就到了重头戏上的Co-TRM这个架构，直接来看论文中的图：

![img](https://mmbiz.qpic.cn/sz_mmbiz_png/LU88NSfAnCxS5IMicOzwe7yf5TSOvicuTNoibERibaZuXJATlpD8cR6Qk6FianDCELSJQ9FYYQS8aJCBZnMAv0g75dw/640?wx_fmt=png)

其实这个结构很简单，就是在做attention的时候，做一些改动；

在上面这个图片流，我的Q矩阵来自图片信息，但是我的K和V矩阵来自文本信息；

在下面这个文本流，我的Q矩阵来自文本信息，但是我的K和V矩阵来自图片信息；

简单说，就是做了一个在文本条件下的图片的attention和在图片条件下的文本的attention；

也就是在文本和图片之间做了一个信息的交互；

这里需要注意的是，在交互之后，各自走自己独立的TRM结构，而并没有拼接在一起走TRM结构；

我自己在之前的多模态落地讲解文章中有谈到，我的baseline架构和这个很类似，只不过，我是做了双方面的attentinon之后，直接拼接接了任务相关的结构；

### 2. ViLBERT预训练任务是什么？

然后我们再来看ViLBERT预训练任务是什么？

之前文章谈到，多模态的预训练任务从BERT演化而来，可以分为两类任务：重建任务和匹配任务；

那么在ViLBERT也是这两类；

**重建任务就是文本重建和图片重建；**

**匹配任务是是否匹配；**

**需要注意的是重建任务构建的时候并么有保持另一个模态数据保持完整；匹配任务是H_cls和H_img相乘接了一个MLP做分类；**

也是直接来看图：

![img](https://mmbiz.qpic.cn/sz_mmbiz_png/LU88NSfAnCxS5IMicOzwe7yf5TSOvicuTNEl383oq1dic6TBj3LXhFB06YQicglHWfDsrBj3WxgGQCWgfY9QqONIxA/640?wx_fmt=png)

这么看文本和图片的任务是合在一起训练了，其实从模型架构我们可以看到两个流在最后是各自分支输出的，这点需要注意；

### 3. ViLBERT实现细节有哪些？

实现细节这里其实可说的没有多，主要是**ViLBERT本身的预训练和在四个下游任务进行迁移学习；**

在预训练的时候，数据使用的是330万个图像-字幕对；

这个很有意思，相当于是一种无监督的语料，但是怎么处理文本和字母不相关的问题，因为并不是每时每刻都是相关的，想一下电视剧的情景；所以这种数据噪声估计很严重，需要清理；

论文使用的数据来自ACL2018论文搞出来的数据，比较干净一点；

由于担心训练时间，ViLBERT中的BERT这个流使用的是bert-base，后来发现bert-large可能会有更好的表现；

使用FasterRCNN，通过卡阈值的方式来提取图像中的置信度比较高的候选框【10-36个】，使用 mean-pooled convolutional feature 作为这个候选区域的特征向量；

其他的:8个TitanX GPUs / batch size of 512 /10 epochs / Adam optimizer / initial learning rates of 1e-4.

下游任务中的几个任务：Visual Question Answering (VQA)；Grounding Referring Expressions;Caption-Based Image Retrieval;‘Zero-shot’ Caption-Based Image Retrieval;

做了两个对比实验：

1. **第一个是使用了单流的bert-videobert；没怎么改变bert的架构；**

这个其实对照到文本相似度这边，其实属于交互式模型，所以这种模型存在的一个问题是没有办法很好的缓存单个文本或者单个图片的embedding，这样在做一些检索任务的时候就非常的不方面；

为啥DSSM 架构这么有名，效果是一方面，速度更加的被大家看重；

1. **第二个实验是相同的 ViLBERT架构，但是并没有在我们的图像-字幕数据集中进行预训练；**

这个实验是为了 看一下 架构和预训练数据的作用，从而来证明，架构是有用的，预训练也是有用的；

