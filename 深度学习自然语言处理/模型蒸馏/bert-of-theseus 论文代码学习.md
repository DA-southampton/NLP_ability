bert-of-theseus 论文代码学习

论文还没看，主要看了几个博客，这里我主要是说一下代码的实现情况。

官方代码在这里： https://github.com/JetRunner/BERT-of-Theseus

代码使用的是比较新版本的Transformer库，因为我自己的代码之前都使用的比较老的版本transformer，所以从新版本改到老版本改动了不少的地方，记录在这里：

首先我把自己的改好的代码放到这里：
https://github.com/DA-southampton/bert-of-theseus_change


数据使用的是tnews分类数据;

修改的部分主要是这些地方：

1.
self.config.is_decoder and encoder_hidden_states is not None:

这里首先是把 modeling_bert_of_theseus.py 里面涉及到is_decoder 这个参数的都注释掉，老版本是没有这个config

2.
报错：forward() got an unexpected keyword argument 'inputs_embeds'


embedding_output = self.embeddings(input_ids=input_ids, position_ids=position_ids,
                                           token_type_ids=token_type_ids, inputs_embeds=inputs_embeds)


embedding_output = self.embeddings(input_ids=input_ids, position_ids=position_ids,token_type_ids=token_type_ids)

老版本是没有inputs_embeds ，直接删掉就可以


3.

报错：TypeError: forward() takes from 2 to 4 positional arguments but 6 were given

layer_outputs = layer_module(hidden_states, attention_mask, head_mask[i], encoder_hidden_states,
                                         encoder_attention_mask)
老版本Bertlayer是有4个参数，直接删掉多余的就可以

