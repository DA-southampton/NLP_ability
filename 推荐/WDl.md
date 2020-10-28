更多NLP文章在这里：

**https://github.com/DA-southampton/NLP_ability**

谈到WDL，一个经常看到的总结是：Wide and Deep 模型融合 wide 模型的记忆能力和 Deep 模型的泛化能力，进行两个模型的联合训练，从而兼顾推荐的准确性和多样性。

理解上面这句话，还是要先弄清楚：什么是记忆能力，什么是泛化能力？

#### 1. 什么是记忆能力与泛化能力

#### 1.1记忆能力

我们先说记忆能力，从中文的角度理解，记忆能力就是之前做过的事情，在后面做同样的事的时候会利用到之前的经验和教训。

进一步，记忆能力就是对之前学习到的经验或者说规律的遵守。

原论文是这么说的：

> Memorization can be loosely defined as learning the frequent co-occurrence of items or features and exploiting the correlation available in the historical data.

从这段话可以看出来记忆能力分为两个部分：

1. 从历史数据中学习共现的物体/特征组合--->这就对应到上面谈到的经验规律
2. 在预测的时候利用到这种学习到的这种相关性--->这就对应到上面谈到的对经验的遵守。

在这里，我想提一下，在第一点中提到的 “学习共现的物体/特征组合” 的主体是谁？

最开始我认为是模型，后来认为不是。

因为LR模型属于广义线性模型，本身不具备对特征之间非线性关系进行建模。

所以需要我们从历史数据中找到有用的特征组合（当然我们也可以使用一些工具来找到哪些特征组合是有效的），人为的加入到模型中，给LR模型增添非线性建模能力。

简单来说，记忆能力是一种共现规律，表现方式为特征交叉，它需要人为或者通过工具从历史数据中找到，并放入到模型中作为新的特征，从而增加非线性建模能力。

**记忆能力过强会出现一个问题，就是推荐物体的单一化。**

原文是这么说的：

> Recommendations based on memorization are usually more topical and directly relevant to the items on which users have already performed actions.

#### 1.2泛化能力

对于泛化能力，原论文是这么说的：

> Generalization, on the other hand, is based on transitivity of correlation and explores new feature combinations that have never or rarely occurred in the past.

关键词是：**从未或者很少出现的特征组合**

神经网络无需人为构建组合特征，有着自动做特征组合的方式。可以通过对类别特征做embedding，这样就是在测试集中出现在训练集中没有出现过的特征组合方式，也可以使用embedding进行计算得到对应的值。

综合来说，LR模型有着更强的记忆能力，Deep模型有着更强的泛化能力。

#### 2.模型架构图

![模型架构图](./images/模型架构图.png)

整个模型分为三个部分，左边的Wide模型，右边的Deep模型，最后输出的Softmax/sigmoid函数。

Wide使用的是LR模型，这里需要注意的点是LR的输入特征包含两部分：

1. 原始特征
2. 特征交叉之后的特征（特征交叉之前各自特征需要one-hot）

Deep模型使用的是前馈神经网络，会对类别特征做embedding，连续特征不动直接输入就好（需要提前做好特征工程）。

联合训练，Wide使用FTRL算法进行优化，Deep模型使用AdaGrad进行优化。

在实际中，Wide和Deep部分直接使用一个优化器就可以。

#### 3.实践

##### 3.1 实践架构

![WDL实践图](./images/WDL实践图.png)



这个是原论文中的架构图，我们自己在实践的时候不一定完全遵守。比如架构图中Wide部分只是使用了交叉特征，我们在使用的时候可以把原始的离散特征或者打散后的连续特征加过来。

##### 3.2 多模态特征的加入

有些时候对于用户或者待推荐的物体会有Text和Image，为了增加效果，可能会使用到多模态特征。

（是否需要加入多模态特征需要大家亲自试，很有可能吭哧吭哧写了好久代码调试通了，最后发现效果提升不多甚至还会降低，哭了）

我这里给几个简单的思路。

1. Text 和 Image 的 embedding 向量，采用 和Wide模型一样的方式加入到整体模型中就可以了。至于 两者的Embedding向量如何获取，就看你自己了。
2. Text和Image之间使用attention之后再加入
3. Text和Image 和Deep 模型的输出拼接之后再做一次处理
4. 多看 Paper-给个关键词：**Multimodal Fusion**

##### 3.3 特征工程小记录

在详细写一个特征工程的文章，写完之后会放出来。

#### 后记

读完整个论文，让我去回顾整个模型，给我这样一个感觉：

对于隐藏在历史数据中的共现特征关系，Deep模型是可以学习到的。但是WDL模型做的是，把其中的一部分（容易观察出来或者通过其他工具找出来的特征组合）放到LR这边，从而显式的加入到模型中。

往极端的方面想一下，LR模型这边更像是一种规则，是对Deep模型输出的补充。

