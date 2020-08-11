class ESIM(nn.Module):
    def __init__(self, config_base):
        super(ESIM, self).__init__()
        self.device = config_base.device
        self.dropout=config_base.dropout
        if config_base.embedding_pretrained is not None:
            self.word_emb = nn.Embedding.from_pretrained(config_base.embedding_pretrained, freeze=False)
        else:
            self.word_emb = nn.Embedding(config_base.vord_size, config_base.embeds_dim)
        self.word_emb.float()
        self.word_emb.weight.requires_grad = True
        self.word_emb.to(self.device)
        if self.dropout:
            self.rnn_dropout = RNNDropout(p=config_base.dropout)
        self.first_rnn = Seq2SeqEncoder(nn.LSTM, config_base.embeds_dim, config_base.hidden_size, bidirectional=True)
        self.projection = nn.Sequential(nn.Linear(4*2*config_base.hidden_size, config_base.hidden_size),
                                        nn.ReLU())
        self.attention = SoftmaxAttention()
        self.second_rnn = Seq2SeqEncoder(nn.LSTM, config_base.hidden_size, config_base.hidden_size, bidirectional=True)
        self.classification = nn.Sequential(nn.Linear(2*4*config_base.hidden_size, config_base.hidden_size),
                                            nn.ReLU(),
                                            nn.Dropout(p=config_base.dropout),
                                            nn.Linear(config_base.hidden_size, config_base.hidden_size//2),
                                            nn.ReLU(),
                                            nn.Dropout(p=config_base.dropout),
                                            nn.Linear(config_base.hidden_size//2, config_base.num_classes))
           
    def forward(self, q1, q1_lengths, q2, q2_lengths):
        q1_mask = get_mask(q1, q1_lengths).to(self.device)
        q2_mask = get_mask(q2, q2_lengths).to(self.device)
        q1_embed = self.word_emb(q1) ## 这里输入的不是q1_mask （维度是[256,32] 32是按照这个batch中中的最大长度来的，输入的是q1[256,50]）
        q2_embed = self.word_emb(q2)
        if self.dropout:
            q1_embed = self.rnn_dropout(q1_embed)
            q2_embed = self.rnn_dropout(q2_embed)
        # 双向lstm编码
        q1_encoded = self.first_rnn(q1_embed, q1_lengths) ## 这里我们输入的是q1_embed 和q1_lengths
        q2_encoded = self.first_rnn(q2_embed, q2_lengths)
        # atention
        q1_aligned, q2_aligned = self.attention(q1_encoded, q1_mask, q2_encoded, q2_mask)
        # concat
        q1_combined = torch.cat([q1_encoded, q1_aligned, q1_encoded - q1_aligned, q1_encoded * q1_aligned], dim=-1)
        q2_combined = torch.cat([q2_encoded, q2_aligned, q2_encoded - q2_aligned, q2_encoded * q2_aligned], dim=-1)
        # 映射一下
        projected_q1 = self.projection(q1_combined)
        projected_q2 = self.projection(q2_combined)
        if self.dropout:
            projected_q1 = self.rnn_dropout(projected_q1)
            projected_q2 = self.rnn_dropout(projected_q2)
        # 再次经过双向RNN
        q1_compare = self.second_rnn(projected_q1, q1_lengths)
        q2_compare = self.second_rnn(projected_q2, q2_lengths)
        # 平均池化 + 最大池化
        q1_avg_pool = torch.sum(q1_compare * q1_mask.unsqueeze(1).transpose(2, 1), dim=1)/torch.sum(q1_mask, dim=1, keepdim=True)
        q2_avg_pool = torch.sum(q2_compare * q2_mask.unsqueeze(1).transpose(2, 1), dim=1)/torch.sum(q2_mask, dim=1, keepdim=True)
        q1_max_pool, _ = replace_masked(q1_compare, q1_mask, -1e7).max(dim=1)
        q2_max_pool, _ = replace_masked(q2_compare, q2_mask, -1e7).max(dim=1)
        # 拼接成最后的特征向量
        merged = torch.cat([q1_avg_pool, q1_max_pool, q2_avg_pool, q2_max_pool], dim=1)
        # 分类
        logits = self.classification(merged)
        probabilities = nn.functional.softmax(logits, dim=-1)
        return logits, probabilities





class SiaGRU(nn.Module):
    def __init__(self, config_base):
        super(SiaGRU, self).__init__()
        self.device = config_base.device
        if config_base.embedding_pretrained is not None:
            self.word_emb = nn.Embedding.from_pretrained(config_base.embedding_pretrained, freeze=False)
        else:
            self.word_emb = nn.Embedding(config_base.vord_size, config_base.embeds_dim)
        self.word_emb.float()
        self.word_emb.weight.requires_grad = True
        self.word_emb.to(self.device)
        self.gru = nn.LSTM(config_base.embeds_dim, config_base.hidden_size, batch_first=True, bidirectional=True, num_layers=2)
        self.h0 = self.init_hidden((2 * config_base.num_layer, 1, config_base.hidden_size))
        self.h0.to(self.device )
        self.pred_fc = nn.Linear(config_base.pad_size, 2)

    def init_hidden(self, size):
        h0 = nn.Parameter(torch.randn(size))
        nn.init.xavier_normal_(h0)
        return h0

    def forward_once(self, x):
        output, hidden = self.gru(x)
        return output
    
    def dropout(self, v):
        return F.dropout(v, p=0.2, training=self.training)

    def forward(self, q1, q1_lengths, q2, q2_lengths):
        p_encode = self.word_emb(q1)
        h_endoce = self.word_emb(q2)
        p_encode = self.dropout(p_encode)
        h_endoce = self.dropout(h_endoce)
        
        encoding1 = self.forward_once(p_encode)
        encoding2 = self.forward_once(h_endoce)
        sim = torch.exp(-torch.norm(encoding1 - encoding2, p=2, dim=-1, keepdim=True))
        x = self.pred_fc(sim.squeeze(dim=-1))
        probabilities = nn.functional.softmax(x, dim=-1)
        return x, probabilities