UniLM：给Bert插上文本生成的翅膀

今天分享一个论文[UniLM](https://arxiv.org/pdf/1905.03197.pdf, "Unified Language Model Pre-training for Natural Language Understanding and Generation")，核心点是掌握三种LM任务形式：单向LM，双向LM，序列到序列LM；

# 1. 生成任务

NLP任务大致可以分为NLU和NLG两种；Bert在NLU任务上效果很好，但是天生不适合处理生成任务。

原因在于Bert的预训练过程是使用的MLM，和生成任务的目标并不一致。

生成任务目标是每次蹦出来一个词，只能看到当前位置之前的词汇。

而Bert采用的是双向的语言模型，除了mask的单词，两个方向的词汇都可以被看到。

所以对Bert的一个改进思路就是让它在具有NLU能力的时候，同时兼备NLG能力。

# 2. 三种LM任务

UniLM做的就是这样一个事情。

具体的实现方式是设计了一系列的完形填空任务，这些完形填空任务的不同之处在于对上下文的定义。

1. 从左到右的LM：mask单词的上下文，也就是可以作用于mask的单词是当前mask的左侧的所有单词
2. 从右到左的LM：和上面第一个相比就是方向的变化，上下文是当前mask的所有右侧的单词。至于为什么这么做？大家可以联想一下双向Bilstm，不也是做的双向的吗。
3. 双向LM：就是当前mask的左右词汇都可以看到
4. sequence-to-sequence LM：这个就是UniLM能够具有生成能力的关键。我们的输入是source句子和target句子，mask单词在target上，那么当前mask的上下文就是source句子的所有单词和target句子中mask单词左侧的词汇可以被看到

有个点需要注意，就是优点的第一条，三个任务（从左到右LM和从右到左LM我们归为一种任务叫单向LM）是一起优化的。

怎么同时训练这三种任务？重点就是理解好上面讲的的三种任务的形式，对上下文的定义，我们是使用不同的Mask矩阵来对应不同任务输入数据形式。

文中使用的是这样一张图来展示：

![UniLM不同mask](https://picsfordablog.oss-cn-beijing.aliyuncs.com/2020-12-03-074447.jpg)

这个图看着很复杂，其实慢慢梳理的话还是很简单的，最重要的是比对着上面对三种任务上下文的定义来看。

需要注意的一个小细节是对角线上的点是被参与attention的，因为即便当前被mask掉了，也是参与计算的，被编码的。这里允许被attention不代表能够被看见。

比如：x1，x2，【mask】，x4

单向LM中，x1，x2对于【mask】可以被看见，在编码的时候x1，x2和【mask】都会被编码计算。

简单来说，UniLM是训练的一个双向的Encoder。

# 3. 其他细枝末节

在训练的时候，1/3的时候使用双向LM，1/3的时候使用sequence-to-sequence LM，1/6的时候使用从左到右的LM，1/6的时间使用从右到做的LM。

1. Gelu 激励函数
2. 24层TRM，最大长度512，1024Hidden Size，16Heads，340M参数量
3. 初始化使用Bert Large
4. 15%被mask，其中80%真正替换mask，10%随机替换，10%不动。替换的时候，80% 的时候替换单个token，20%的时候替换bigram 或者 trigram

第四个步骤类似中文实体词的mask，也算是一点改进。

有个细节点需要注意的是，作者强调，不同的segment embedding用来区分不同LM任务。

Bert的时候，区分上下句子，我们使用0和1，在这里，我们使用这个segment embedding用来区分任务：

比如说，双向对应0和1；单向left-right对应2；单向right-left对应3；序列对应4和5；

# 4. 总结

掌握以下几个细节点就可以：

1. 联合训练三种任务：单向LM，双向LM，序列LM
2. 使用不同的attention矩阵控制三种任务形式的参与
3. segment embedding可以区分不同的任务形式
4. mask的时候15% 的有被替换的概率，其中80% 被真正替换。在这80%真正替换的里面有80%单个token被替换，20%的二元或者三元tokens被替换
5. 被看见和被编码不太一样，注意看那张图。

# 5. 加我微信，点赞之交

![个人微信](https://picsfordablog.oss-cn-beijing.aliyuncs.com/2020-12-03-074615.png)