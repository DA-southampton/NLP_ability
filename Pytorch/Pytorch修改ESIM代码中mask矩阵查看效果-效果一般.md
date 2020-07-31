Pytorch修改ESIM代码中mask矩阵查看效果-效果一般
我对ESIM中的mask矩阵有所怀疑，于是自己改写了一个mask的矩阵，不过效果确实没有原始的好，很奇怪

https://github.com/DA-southampton/TextMatch/blob/master/ESIM/utils.py
就是这个链接中，我改了主要是以下两个函数的部分地方：

```python
def get_mask(sequences_batch, sequences_lengths):
    batch_size = sequences_batch.size()[0]
    max_length = torch.max(sequences_lengths)
    mask = torch.ones(batch_size, max_length, dtype=torch.float)
    mask[sequences_batch[:, :max_length] == 0] = -10000.0 ## 这里修改为-10000，印象中抱抱脸初始版本是这么实现的
    return mask	

def masked_softmax(tensor, mask):
    tensor_shape = tensor.size()
    reshaped_tensor = tensor.view(-1, tensor_shape[-1])
    # Reshape the mask so it matches the size of the input tensor.
    while mask.dim() < tensor.dim():
        mask = mask.unsqueeze(1)
    mask = mask.expand_as(tensor).contiguous().float()
    reshaped_mask = mask.view(-1, mask.size()[-1])

    result = nn.functional.softmax(reshaped_tensor+reshaped_mask, dim=-1) ## 这里变为加

    return result.view(*tensor_shape)

```

改完之后效果不咋样，真的很奇怪