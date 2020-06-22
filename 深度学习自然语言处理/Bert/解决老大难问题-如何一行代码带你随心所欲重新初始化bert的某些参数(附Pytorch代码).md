Bert我们一般使用方法是，加载预训练模型，在我们自己的任务上进行微调。但是我们有些时候会遇到这种情况，比如说，之前文章提到的，
我不想要你预训练模型中最后三层参数，而是使用我自己的方法重新初始化。

首先解释一下为什么需要这么做？有的论文发现，bert越靠后面（越靠近顶层，也就是输出层），学到的知识越是笔记抽象高级的知识，越靠近预训练模型的任务情况，和我们自己的任务就不太相符，所以想要重新初始化，基于我们自己的任务从零学习。

好了，代码是怎么实现？

一般pytorch的初始化方法我就不说了，这个比较简单，之后可能有时间写一下，这里专门介绍一下bert里面如何去做。

首先，我们看一下源代码，加载模型的时候是怎么加载的：
```python
model = model_class.from_pretrained(args.model_name_or_path, from_tf=bool('.ckpt' in args.model_name_or_path), config=config)
```
链接在这里：https://github.com/DA-southampton/Read_Bert_Code/blob/0605619582f1bcd27144e2d76fac93cb16e44055/bert_read_step_to_step/run_classifier.py#L462


再执行到这里之后，会进入并执行这个函数：
```python
def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
```
代码链接在这里看：
https://github.com/DA-southampton/Read_Bert_Code/blob/0605619582f1bcd27144e2d76fac93cb16e44055/bert_read_step_to_step/transformers/modeling_utils.py#L224

这个函数就是我们要修改的函数，核心操作是这个操作：

```python
module._load_from_state_dict(state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
```

代码位置在这里：
https://github.com/DA-southampton/Read_Bert_Code/blob/0605619582f1bcd27144e2d76fac93cb16e44055/bert_read_step_to_step/transformers/modeling_utils.py#L404

主要是两个参数最重要：
missing_keys：就是我们自己定义的模型有哪些没在预训练模型中，比如我们的模型现在是 BertForSequenceClassification ，那么这里结果就是 ['classifier.weight', 'classifier.bias']
unexpected_keys:预训练模型的参数有很多，这里的结果是定义的我们对哪些参数忽视，并不采用，这里的正常结果是这样的：['cls.predictions.bias', 'cls.predictions.transform.dense.weight', 'cls.predictions.transform.dense.bias', 'cls.predictions.transform.LayerNorm.weight', 'cls.predictions.transform.LayerNorm.bias', 'cls.predictions.decoder.weight', 'cls.seq_relationship.weight', 'cls.seq_relationship.bias']

重点来了，如果我们想要对第一层的query的进行重新初始化，怎么做？分两个步骤，第一步，定义你想要重新初始化哪些参数，第二步代入进去。看代码：

```python
unexpected_keys =['bert.encoder.layer.0.attention.self.query.weight','bert.encoder.layer.0.attention.self.query.bias']
```

就这么简单，这里定义了就可以

代码位置在这里
https://github.com/DA-southampton/Read_Bert_Code/blob/0605619582f1bcd27144e2d76fac93cb16e44055/bert_read_step_to_step/transformers/modeling_utils.py#L364