ESIM

1. 导入数据

1.1 train_data = LCQMC_Dataset(train_file, vocab_file, max_length)

通过LCQMC_Dataset 导入数据，在LCQMC_Dataset 中，做了这几件事情：

1.1.1 p, h, self.label = load_sentences(LCQMC_file)

读取数据文件，切分数据

1.1.2 word2idx, _, _ = load_vocab(vocab_file)

读取字典文件，从而获得 word2idx, idx2word, vocab

1.1.3 self.p_list, self.p_lengths, self.h_list, self.h_lengths = word_index(p, h, word2idx, max_char_len)

将数据转化为对应的数值，并且根据max_char_len进行切分或者截断

1.1.4 
self.p_list = torch.from_numpy(self.p_list).type(torch.long)
self.h_list = torch.from_numpy(self.h_list).type(torch.long)

转化为对应的torch tensor

1.2  数据batch化
train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
按照batch进行读取。是torch的代码，我就不看了

1.3 读取embedding文件
embeddings = load_embeddings(embeddings_file)
注意pad全为零

2. model构建
model = ESIM(hidden_size, embeddings=embeddings, dropout=dropout, num_classes=num_classes, device=device).to(device)

3. 为了方便，我把ESIM整体流程重点是attention的部分抽离了出来
