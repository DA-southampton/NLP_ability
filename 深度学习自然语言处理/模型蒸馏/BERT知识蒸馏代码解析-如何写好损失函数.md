大家好，我是DASOU；

今天从代码角度深入了解一下知识蒸馏，主要核心部分就是分析一下在知识蒸馏中损失函数是如何实现的；

之前写过一个关于BERT知识蒸馏的理论的文章，感兴趣的朋友可以去看一下：[Bert知识蒸馏系列(一)：什么是知识蒸馏](http://mp.weixin.qq.com/s?__biz=MzIyNTY1MDUwNQ==&mid=2247484225&idx=1&sn=b48cfea668bd5b91e1bb8c74e3ab1db3&chksm=e87d3167df0ab8713d045bf656291b0da9e4928f57c27d5ea4e4f537cf44f051dc0b6d6ec35a&scene=21#wechat_redirect)。

知识蒸馏一个简单的脉络可以这么去梳理：**学什么，从哪里学，怎么学？**

**学什么**：学的是老师的知识，体现在网络的参数上；

**从哪里学**：输入层，中间层，输出层；

**怎么学**：损失函数度量老师网络和学生网络的差异性；

从架构上来说，BERT可以蒸馏到简单的TextCNN，LSTM等，也就可以蒸馏到TRM架构模型，比如12层BERT到4层BERT；

之前工作中用到的是BERT蒸馏到TextCNN；

最近在往TRM蒸馏靠近，使用的是 Textbrewer 这个库（这个库太强大了）；

接下来，我从代码的角度来梳理一下知识蒸馏的核心步骤，其实最主要的就是分析一下损失函数那块的代码形式。

我以一个文本分类的任务为例子，在阅读理解的过程中，最需要注意的一点是数据的流入流出的Shape，这个很重要，在自己写代码的时候，最重要的其实就是这个；

首先使用的是MNLI任务，也就是一个文本分类任务，三个标签；

输入为Batch_data：[32,128]---[Batch_size,seq_len];

老师网络：BERT_base：12层，Hidden_size为768；

学生网络：BERT_base：4层，Hidden_size为312；

首先第一个步骤是训练一个老师网络，这个没啥可说。

其次是初始化学生网络，然后将输入Batch_data流经两个网络；

在初始化学生网络的时候，之前有的同学问到是如何初始化的一个BERT模型的；

关于这个，最主要的是修改Config文件那里的层数，由正常的12改为4，然后如果你不是从本地load参数到学生网络，BERT模型的类会自动调用初始化；

关于代码实现，我之前写过一个文章，大家可以看这里的代码解析，更加的清洗一点：[Pytorch代码验证--如何让Bert在finetune小数据集时更“稳”一点](http://mp.weixin.qq.com/s?__biz=MzIyNTY1MDUwNQ==&mid=2247483696&idx=1&sn=cc79da01752c5e7588ef8686c1f95e1f&chksm=e87d3316df0aba00e23189158bfdb7a41e964f545422d97940a87d89571881a5ac1be4bb77f8&scene=21#wechat_redirect)；

然后我们来说数据首先流经学生网络，我们得到两个东西，一个是最后一层【CLS】的输出，此时未经softmax操作，所以是logits，维度为：[32,3]-[batch_size,label_size];

第二个东西是中间隐层的输出，维度为:[5,32,128,312]，也就是 [隐层数量,batch_size,seq_len,Hidden_size];

需要注意的是这里的隐层数量是5，因为正常的隐层在模型定义的时候是4，然后这里是加上了embedding层；

还有一点需要注意的是，在度量学生网络和老师网络隐层差异的时候，这里是度量的seq_len，也就是对每个token的输出都做了操作；

如果在这里我们想做类似【CLS】的输出的时候，只需要提取最开始的一个[32,312]的向量就可以；不过，一般来说我们不这么做；

其次流经老师网络，我们同样得到两个东西，一个是最后一层【CLS】的输出，此时未经softmax操作，所以是logits，维度为：[32,3]-[batch_size,label_size];

第二个东西是中间隐层的输出，维度为:[5,32,128,768]，也就是 [隐层数量,batch_size,seq_len,Hidden_size];

这里需要注意的是老师网络和学生网络隐层数量不一样，一个是768，一个是312。

这其实是一个很常见的现象；就是我们的学生网络在减少参数的时候，不仅会变矮，有时候我们也想让它变窄，也就是隐层的输出会发生变化，从768变为312；

这个维度的变化需要注意两点，首先就是在学生模型初始化的时候，不能套用老师网络的对应层的参数，因为隐层Hidden_size发生了变化。所以一般调用的是BERT自带的初始化方式；

其次就是在度量学生网络和老师网络差异性的时候，因为矩阵大小不一致，不能直接做MSE。在代码层面上，需要做一个线性映射，才能做MSE。

而且还需要注意的一点是，由于老师网络已经固定不动了，所以在做映射的时候我们是要对学生网路的312加一个线性层转化到768层，也就是说这个线性层是加在了学生网络；

整个架构的损失函数可以分为三种：首先对于【CLS】的输出，使用KL散度度量差异；对于隐层输出使用MSE和MMD损失函数进行度量；

对于损失函数这块的选择，其实我觉得没啥经验可说，只能试一试；

看了很多论文加上自己的经验，一般来说在最后面使用KL，中间层使用MSE会更好一点；当然有的实验也会在最后一层直接用MSE；玄学。

在初看代码的时候，MMD这个之前我没接触过，还特意去看了一下，关于理论我就不多说了，一会看代码吧。

首先对【CLS】的输出，代码如下：

```
def kd_ce_loss(logits_S, logits_T, temperature=1):
    if isinstance(temperature, torch.Tensor) and temperature.dim() > 0:
        temperature = temperature.unsqueeze(-1)
    beta_logits_T = logits_T / temperature
    beta_logits_S = logits_S / temperature
    p_T = F.softmax(beta_logits_T, dim=-1)
    loss = -(p_T * F.log_softmax(beta_logits_S, dim=-1)).sum(dim=-1).mean()
    return loss
```

首先对于 logits_S，就是学生网络的【CLS】的输出，logits_T就是老师网络【CLS】的输出，temperature 在代码中默认参数是1，例子中设置为了8；

整个代码其实很简单，就是先做Temp的一个转化，注意这里我们对学生网络的输出和老师网络的输出都做了转化，然后做loss计算；

其次我们来看比较复杂的中间层的度量；

首先需要掌握一点，就是学生网络和老师网络层之间的对应关系；

学生网络是4层，老师网络12层，那么在对应的时候，简单的对应关系就是这样的：

```
layer_T : 0, layer_S : 0,
layer_T : 3, layer_S : 1, 
layer_T : 6, layer_S : 2, 
layer_T : 9, layer_S : 3,
layer_T : 12, layer_S : 4，
```

这个对应关系是需要我们认为去设定的，将学生网络的1层对应到老师网络的12层可不可以？当然可以，但是效果不一定好；

一般来说等间隔的对应上就好；

这个对应关系其实还有一个用处，就是学生网络在初始化的时候【假如没有变窄，只是变矮，也就是层数变低了】，那么可以从依据这个对应关系把权重copy过来；

学生网络的隐层输出为：[5,32,128,312],老师网络隐层输出为[5,32,128,768]

那么在代码实现的时候，需要做一个zip函数把对应层映射过去，然后每一层计算MSE，然后加起来作为损失函数；

我们来看代码：

```
inters_T = {feature: results_T.get(feature,[]) for feature in FEATURES}
inters_S = {feature: results_S.get(feature,[]) for feature in FEATURES}

for ith,inter_match in enumerate(self.d_config.intermediate_matches):
    if type(layer_S) is list and type(layer_T) is list: ## MMD损失函数对应的情况
        inter_S = [inters_S[feature][s] for s in layer_S]
        inter_T = [inters_T[feature][t] for t in layer_T]
        name_S = '-'.join(map(str,layer_S))
        name_T = '-'.join(map(str,layer_T))
        if self.projs[ith]: ## 这里失去做学生网络隐层的映射
            #inter_T = [self.projs[ith](t) for t in inter_T]
            inter_S = [self.projs[ith](s) for s in inter_S]
    else:## MSE 损失函数
        inter_S = inters_S[feature][layer_S]
        inter_T = inters_T[feature][layer_T]
        name_S = str(layer_S)
        name_T = str(layer_T)
        if self.projs[ith]:
            inter_S = self.projs[ith](inter_S) # 需要注意的是隐层输出是312，但是老师网络是768，所以这里要做一个linear投影到更高维，方便计算损失函数
        
    intermediate_loss = match_loss(inter_S, inter_T, mask=inputs_mask_S)  ## loss = F.mse_loss(state_S, state_T)
    total_loss += intermediate_loss * match_weight
```

这个代码里面比如迷糊的是【self.d_config.intermediate_matches】，打印出来发现是这个东西：

```
IntermediateMatch: layer_T : 0, layer_S : 0, feature : hidden, weight : 1, loss : hidden_mse, proj : ['linear', 312, 768, {}], 
IntermediateMatch: layer_T : 3, layer_S : 1, feature : hidden, weight : 1, loss : hidden_mse, proj : ['linear', 312, 768, {}], 
IntermediateMatch: layer_T : 6, layer_S : 2, feature : hidden, weight : 1, loss : hidden_mse, proj : ['linear', 312, 768, {}], 
IntermediateMatch: layer_T : 9, layer_S : 3, feature : hidden, weight : 1, loss : hidden_mse, proj : ['linear', 312, 768, {}], 
IntermediateMatch: layer_T : 12, layer_S : 4, feature : hidden, weight : 1, loss : hidden_mse, proj : ['linear', 312, 768, {}], 
IntermediateMatch: layer_T : [0, 0], layer_S : [0, 0], feature : hidden, weight : 1, loss : mmd, proj : None, 
IntermediateMatch: layer_T : [3, 3], layer_S : [1, 1], feature : hidden, weight : 1, loss : mmd, proj : None, 
IntermediateMatch: layer_T : [6, 6], layer_S : [2, 2], feature : hidden, weight : 1, loss : mmd, proj : None, 
IntermediateMatch: layer_T : [9, 9], layer_S : [3, 3], feature : hidden, weight : 1, loss : mmd, proj : None, 
IntermediateMatch: layer_T : [12, 12], layer_S : [4, 4], feature : hidden, weight : 1, loss : mmd, proj : None
```

简单说，这个变量存储的就是上面我们谈到的层与层之间的对应关系。前面5行就是MSE损失函数度量，后面那个注意看，层数对应的时候是一个列表，对应的是MMD损失函数；

我们来看一下MMD损失的代码形式：

```
def mmd_loss(state_S, state_T, mask=None):
    state_S_0 = state_S[0] # (batch_size , length, hidden_dim_S)
    state_S_1 = state_S[1] # (batch_size , length, hidden_dim_S)
    state_T_0 = state_T[0] # (batch_size , length, hidden_dim_T)
    state_T_1 = state_T[1] # (batch_size , length, hidden_dim_T)
    if mask isNone:
        gram_S = torch.bmm(state_S_0, state_S_1.transpose(1, 2)) / state_S_1.size(2)  # (batch_size, length, length)
        gram_T = torch.bmm(state_T_0, state_T_1.transpose(1, 2)) / state_T_1.size(2)
        loss = F.mse_loss(gram_S, gram_T)
    else:
        mask = mask.to(state_S[0])
        valid_count = torch.pow(mask.sum(dim=1), 2).sum()
        gram_S = torch.bmm(state_S_0, state_S_1.transpose(1, 2)) / state_S_1.size(2)  # (batch_size, length, length)
        gram_T = torch.bmm(state_T_0, state_T_1.transpose(1, 2)) / state_T_1.size(2)
        loss = (F.mse_loss(gram_S, gram_T, reduction='none') * mask.unsqueeze(-1) * mask.unsqueeze(1)).sum() / valid_count
    return loss
```

看最重要的代码就可以：

```
state_S_0 = state_S[0]#  32 128 312 (batch_size , length, hidden_dim_S)
state_T_0 = state_T[0] #  32 128 768 (batch_size , length, hidden_dim_T)
gram_S = torch.bmm(state_S_0, state_S_1.transpose(1, 2)) / state_S_1.size(2) 
gram_T = torch.bmm(state_T_0, state_T_1.transpose(1, 2)) / state_T_1.size(2)
```

简单说就是现在自己内部计算bmm，然后两个矩阵之间做mse；这里如果我没理解错使用的是一个线性核函数；

损失函数代码大致就是这样，之后有时间我写个简单的repository，梳理一下整个流程；