Word2vec训练参数的选定?

首先根据具体任务，选一个领域相似的语料，在这个条件下，语料越大越好。然后下载一个 word2vec 的新版（14年9月更新），语料小（小于一亿词，约 500MB 的文本文件）的时候用 Skip-gram 模型，语料大的时候用 CBOW 模型。最后记得设置迭代次数为三五十次，维度至少选 50，就可以了。（引自 《How to Generate a Good Word Embedding》）

