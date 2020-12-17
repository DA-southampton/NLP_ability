> BERT推理速度慢，导致落地困难；找到效果不错，推理速度快的模型是一个方向，ESIM是一个很好的选择；

[ESIM](https://arxiv.org/pdf/1609.06038.pdf, "Enhanced LSTM for Natural Language Inference") 推理速度快，效果不错，堪称文本匹配的利器；

对于ESIM，重点掌握就一点：**是两个句子之间究竟是如何交互的.**

# 0 整体架构

先来看整体结构是什么样子：

![ESIM整体架构](https://picsfordablog.oss-cn-beijing.aliyuncs.com/2020-12-17-032654.png)

对于这个架构，我们主要是看左边这个就可以；

可以看到，从架构上来看，这个模型大概可以范围四层：最底层是一个双向LSTM，作为句子的的编码器，随后是一个交互层，然后又是一个双向LSTM，最后是一个输出层。

原论文中，也是将ESIM分为四个部分，Input Encoding，Local Inference Modeling， Inference Composition和Prediction，我们一个个说。


# 1. Input Encoding

先假设，我现在有两个句子：

$a=(a_{1},a_{2},...,a_{l_{a}})$ 和$b=(b_{1},b_{2},...,b_{l_{b}})$；

我要判断它是否表达同样的意思：0或者1；

首先第一步是Input Encoding ，这是一个常规操作，就是tokens的embeddings接BiLSTM；注意，我们是两个句子都进入到这同一个BiLSTM中，而不是进入到两个；

公式如下：

![Input Encoding](https://picsfordablog.oss-cn-beijing.aliyuncs.com/2020-12-17-31904.jpg)

作者同时也测试了使用GRU对句子进行编码，但是结果在NLI任务上效果并不好；不过我认为，在实际工作中，两者都可尝试，毕竟玄学。

# 2. Local Inference Modeling

首先这里回顾一下符号：$ \bar{a_{i}} $是句子a在i时刻的是输出，$\bar{b_{j}}$是句子b在j时刻的输出；

那么我们使用如下公式计算两个单词输出之间的交互：

![两个单词之间的交互](https://picsfordablog.oss-cn-beijing.aliyuncs.com/2020-12-17-031904.jpg)

举个很简单的例子，比如说隐层维度为256，那么$ \bar{a_{i}}=[1,256] $，$ \bar{b_{j}}=[1,256] $

那么相乘之后，就是维度[1,1]的值；

这只是两个单词输出之间的交互，我们知道a和b句子长度是不一样的（当然也可能一样）；

这里我们假设a长度为10，b长度为20；

那么经过（11）的计算，我们会得到一个[10,20]的矩阵，用来描述两个句子之间不同单词之间的交互。

核心点就是在于对于这个[10,20]的矩阵，如何对它进行操作，公式如下：

![句子交互](https://picsfordablog.oss-cn-beijing.aliyuncs.com/2020-12-17-031903.jpg)

一定要注意看这里的$\widetilde a_{i}$，我们得到的是[10,20]的矩阵，然后对每一行做softmax操作，得到相似度，然后乘以$b_{j}$。

（12）和（13）本质上是对这个[10,20]的矩阵分别做了按照行的相似度和按照列的相似度；

有点难理解，还是举个例子（这里的例子就不举长度为10和20了，简单点）。

a：【我今天去吃饭了】

b：【他为什么还在学习】

a中的【我】依次对【他为什么还在学习】中的每个单词做乘法，就是得到了$e_{ij}$；然后softmax之后，每个相似度对应乘以【他为什么还在学习】的每个单词输出encoding从而得到加权和，作为$\widetilde a_{i}$

之后就是对特征进行拼接，分为两种，对位相减和对位相乘：

![对位相减和对位相乘](https://picsfordablog.oss-cn-beijing.aliyuncs.com/2020-12-17-031905.jpg)

# 3. inference composition和Prediction

这一步也是常规操作，就是把$m_{a}和m_{b}$输入到第二层的BiLSTM，并把输出做最大池化和平均池化，然后拼接特征，然后输出到全连接，得到结果：

![第二层LSTM](https://picsfordablog.oss-cn-beijing.aliyuncs.com/2020-12-17-031906.jpg)

# 4. 总结

最核心的点还是在于理解如从两个句子单词之间的交互矩阵获得两个句子之间的交互结果；

也就是需要对单词之间的交互矩阵，比如[10,20]，分别按照行做softmax和列做softmax；

这个思想其实在后期很多模型中都有用到，值得思考。
