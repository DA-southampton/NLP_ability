今天介绍复旦的一个论文[TENER](https://arxiv.org/pdf/1911.04474.pdf, "TENER: Adapting Transformer Encoder for Named Entity Recognition") ；普通的TRM在其他NLP任务中效果很不错，但是在NER中表现不佳。为了解决性能不佳，论文做了几点改进。

主要掌握以下三点改进：

1. 方向
2. 距离
3. 无缩放的注意力

# 1. 架构图

先看TENER架构图：

![TENER架构图](https://picsfordablog.oss-cn-beijing.aliyuncs.com/2020-12-09-040433.jpg)

# 2. 距离和方向信息

对于NER任务来说，距离和方向都很重要；

举个简单的例子：【李华住在北京】；李华是人名，北京是地名，如果忽视了方向，那么【北京住在李华】，这个肯定是说不通的。

换句话说，每类NER实体在哪种位置是有着某种关系或者规则的。所以方向很重要。

简单概述普通TRM位置编码的问题，如下：

普通TRM中的正弦位置编码能够捕捉到距离信息，但是不能捕捉到方向信息。而且这种基本性质（distance-awareness）会在sefl-attention消失；

为了改进这种问题，使用了经过改进的相对位置编码，弃用了绝对位置编码；

**2.1 为什么没有方向信息**：

位置编码的点积可以看做在度量两者之间的距离:$PE^{T}_{t}PE_{t+k}$

点积结果画图表示如下：

![点积](https://picsfordablog.oss-cn-beijing.aliyuncs.com/2020-12-09-114414.png)

从这个图，我们可以很清楚的看到，是对称的，也就是说在k=20和k=-20的时候，点击结果相同，换句话说，方向信息没有体现出来。

公式上体现就是：$PE^{T}_{t}PE_{t+k}=PE^{T}_{t-k}PE_{t}$

**2.2 distance-awareness 消失**

再进一步，在self-attention中，distance-awareness 也在消失，这一点，我之前的文章有写，可以看[原版Transformer的位置编码究竟有没有包含相对位置信息](https://mp.weixin.qq.com/s?__biz=MzIyNTY1MDUwNQ==&mid=2247483760&idx=1&sn=c2803e63bdd42e4d1f1f880ce9eda8cc&chksm=e87d3356df0aba40c77356418647856ec135c731fd60122378ed702e1e959c820250c2293e1f&token=588814416&lang=zh_CN#rd)。

改进之后的相对位置编码以及attention计算为：

![新的attention和相对位置编码](https://picsfordablog.oss-cn-beijing.aliyuncs.com/2020-12-09-040434.jpg)

# 3. attention缩放

传统TRM的attention分布被缩放了，从而变得平滑。但是对于NER来说，一个更加尖锐或者说稀疏的矩阵是更合适的，因为并不是所有的单词都需要被关注；一个当前的单词的类别，足够被周围几个单词确定出来。

矩阵越平滑，关注的单词越多，可能会引入更多的噪声信息。

# 4. 总结

1. 原始TRM绝对位置编码不含有方向信息，Self-attention之后相对位置信息也会消失；故使用改进的相对位置编码和新的attention计算方式
2. attention计算不使用缩放系数，减少了噪声信息
3. 使用TRM进行char编码，结合预训练的词向量拼接输入TENER