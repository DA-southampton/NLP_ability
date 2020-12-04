RoBERTa：更大更多更强

今天分享一个Bert的改进工作[RoBERTa](https://arxiv.org/abs/1907.11692, "RoBERTa: A Robustly Optimized BERT Pretraining Approach")。RoBERTa是训练充分的Bert。

主要掌握以下几点，与Bert相比较，RoBERTa预训练的时候：

1. 动态掩码：comparable or slightly better
2. 去掉NSP任务并且更改数据输入格式为全部填充可以跨越多个文档
3. 更多数据，更大bsz，更多的步数，更长训练时间

# 1. 动态掩码

首先明确Bert使用的是静态掩码。但是这样会存在一个现象，比如我训练40个epoches，那么每次epoches都是使用同一批数据。

这其实不是什么大问题，我们在深度学习训练模型的时候，每个epoches基本都没咋变过。

不过对于Bert，其实本质是一个自监督模型。每次的训练输入如果是不同的，对于模型肯定是更好的。

比如我们句子为：今天去哪里吃饭啊？

mask之后为：今天去哪里[mask]饭啊？

每次训练使用同一个mask样本，那么模型见得就少。

如果换一个mask：[mask]天去哪里吃饭啊？

模型对于同一个句子，在预测不同的单词，那么模型对句子的表达能力直觉上肯定是会上升的。

所以为了缓解这种静态掩码的问题，Bert的操作是这样的：

复制原始样本10份，每份都做不同的静态mask，然后进行训练。

我们想一下这个过程：比如我仍然是训练40个epoches，复制了十份样本，相当于每4个epoches使用的是同一个mask的样本。

这个操作确实缓解了静态掩码的问题，但是毕竟还有重复mask的情况出现。

这个时候其实有个朴素的思想，为啥不直接复制40份，然后分在40个epoches中进行训练，这个到时候写Bert的时候再说。

RoBERTa 是咋做的呢？

动态掩码，也就是不是在最开始的时候的数据处理的过程中就生成mask样本，而是在送入到模型之前才进行mask，这样同一个句子，在40epoches中，每次mask都不同。

效果直接看图

![动态mask](https://picsfordablog.oss-cn-beijing.aliyuncs.com/2020-12-02-113140.jpg)

# 2. NSP和模型数据输入格式

这一点其实很有意思。

我们先说RoBERTa 的四种输入形式和实验效果，然后再详细分析：

1. SEGMENT-PAIR+NSP：就是Bert的输入形式
2. SENTENCE-PAIR+NSP：输入的是一对句子，即前后是单个句子
3. FULL-SENTENCES：输入为全量的句子，填满512的长度，采集样本的时候可以跨越文章的界限，去除了NSP loss
4. DOC-SENTENCES：输入和FULL-SENTENCE类似，但是一个样本不能跨越两个document

然后看一下实验效果：

![RoBERTa四种输入形式效果对比图](https://picsfordablog.oss-cn-beijing.aliyuncs.com/2020-12-02-113141.jpg)

对上面这个图一个最简单的总结就是NSP没啥用。然后我们来详细说一下这个事情。

首先Bert的消融实验证明，NSP是应该有的，如果没有NSP，在部分任务上效果有损失。

但是上图RoBERTa实验证明，NSP没啥效果，可以没有。

一个直观的解释，或者说猜测是因为，可能是Bert在消融实验去除NSP的时候，仍然保持的是原始的输入，即有NSP任务的时候的输入形式。

这就相当于，构造了好了符合NSP任务的数据，但是你删去了针对这个任务的损失函数，所以模型并没有学的很好，在部分任务精读下降。

但是RoBERTa这里不是的，它删除NSP任务的时候，同时改变了输入格式，并不是使用上下两句的输入格式，而是类似文档中的句子全部填满这512个字符的格式进行输入。

简单说就是，去掉了NSP任务的同时，去掉了构造数据中NSP数据格式。

比较SEGMENT-PAIR和DOC-SENTENCES两个模式的时候，证明没有NSP效果更好。其实看起来好像并没有控制变量，因为变了两个地方，一个是是否有NSP，一个是输入格式。

这种情况下，就只能去看在下游任务中的效果了。

# 3. 数据+bsz+steps

1. 数据：Bert：16G；RoBERTa：160G；十倍
2. bsz：Bert：256；RoBERTa：8K
3. steps：Bert：1M；RoBERTa：300K/500K

# 4. 总结：

简单总结一下学到的东西：

1. 动态掩码：comparable or slightly better
2. 去掉NSP任务并且更改数据输入格式为全部填充可以跨越多个文档
3. 更多数据，更大bsz，更多的步数，更长训练时间
4. **动态掩码那里，说到一个复制10份的细节，那里是针对的Bert，RoBERTa是每次输入之前才mask，注意区分，不要搞混**

参考资料：RoBERTa: 捍卫BERT的尊严 - yangDDD的文章 - 知乎 https://zhuanlan.zhihu.com/p/149249619

