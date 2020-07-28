Pytorch中mask是如何实现的代码版本1-阅读文本相似度模型

代码参考链接: https://github.com/DA-southampton/TextMatch/tree/master/ESIM

最近在用Pytorch重新ESIM代码，其中关于attention的细节我自己重新梳理了一下，附上代码解读。

我先在有一个batch的数据。Sentence1 维度为[256,32,300],Sentence2的维度为[256,33,300]

维度含义为[batch_size,batch中最大长度，词向量维度]

数据流转ESIM第一个Bilstm之后，维度变化为：Sentence1 维度为[256,32,600],Sentence2的维度为[256,33,600]（假设hidden为300）

此时我们需要计算两个句子输出的attention矩阵，以便计算每个句子的加权和。

我这里主要是梳理矩阵的计算方式细节。

核心代码是这个：
https://github.com/DA-southampton/TextMatch/blob/54e24599ce2d4caaa16d68400dc6a404795d44e9/ESIM/model.py#L57

```python
q1_aligned, q2_aligned = self.attention(q1_encoded, q1_mask, q2_encoded, q2_mask)
```

self.attention 函数对应的是这个函数，如下：
https://github.com/DA-southampton/TextMatch/blob/54e24599ce2d4caaa16d68400dc6a404795d44e9/ESIM/layers.py#L59

```python
class SoftmaxAttention(nn.Module):
        similarity_matrix = premise_batch.bmm(hypothesis_batch.transpose(2, 1).contiguous())  ## 256*32 *33
        # Softmax attention weights.
        prem_hyp_attn = masked_softmax(similarity_matrix, hypothesis_mask)
        hyp_prem_attn = masked_softmax(similarity_matrix.transpose(1, 2).contiguous(), premise_mask)
        # Weighted sums of the hypotheses for the the premises attention,
        # and vice-versa for the attention of the hypotheses.
        attended_premises = weighted_sum(hypothesis_batch, prem_hyp_attn, premise_mask)
        attended_hypotheses = weighted_sum(premise_batch, hyp_prem_attn, hypothesis_mask)
        return attended_premises, attended_hypotheses  
```
首先我们看一下输入：
```python
q1_encoded：256*32*600 q2_encoded：256*33*600  q1mask torch.Size([256, 32])  q2mask torch.Size([256, 33])
```

然后对于这个函数，核心操作是这个：
```python
prem_hyp_attn = masked_softmax(similarity_matrix, hypothesis_mask)
```
similarity_matrix 维度为256*32 *33 hypothesis_mask 为256*33
我们去看一下masked_softmax这个函数：
https://github.com/DA-southampton/TextMatch/blob/54e24599ce2d4caaa16d68400dc6a404795d44e9/ESIM/utils.py#L29
```python
def masked_softmax(tensor, mask):
    tensor_shape = tensor.size()  ##torch.Size([256, 32, 33])
    reshaped_tensor = tensor.view(-1, tensor_shape[-1]) ## torch.Size([7680, 33])
    while mask.dim() < tensor.dim():
        mask = mask.unsqueeze(1)
    mask = mask.expand_as(tensor).contiguous().float()
    reshaped_mask = mask.view(-1, mask.size()[-1])  ## torch.Size([7680, 33])

    result = nn.functional.softmax(reshaped_tensor * reshaped_mask, dim=-1)  ## 补长位置也就是置为零的位置之后进行softmax
    result = result * reshaped_mask ## 再次置为零，因为上面这个对于补长位置还会有概率共现
    # 1e-13 is added to avoid divisions by zero.
    result = result / (result.sum(dim=-1, keepdim=True) + 1e-13) ## 普通的求概率公式
    return result.view(*tensor_shape)
```

简单总结一下：
整个mask的代码其实我读起来感觉比较奇怪，我印象中mask的操作，应该是补长的部分直接为负无穷（代码里写一个-1000就可以），但是他这里的代码，是补长的部位置为0，所以
在softmax的时候，虽然为1，但是也有贡献也有概率的输出，虽然很小。所以又把这些部分置为零，然后用每一行的值除以每一行的总和得到了新的概率值，这个概率和补长的部位就没有关系了。
还有一个细节点需要注意的是，比如我的输入是256*32*33 batch为256，那么我在计算每一行的的时候，完全可以把batch中的数据并起来，也就是变成(256*32)*33

所以我简单总结一下，在这里的mask的操作分为两个步骤：首先补长位置置为零然后计算softmax，随后对softmax的结构补长位置继续置为零，计算简单的分值（各自除以每一行的总和），得到最后的概率值。
