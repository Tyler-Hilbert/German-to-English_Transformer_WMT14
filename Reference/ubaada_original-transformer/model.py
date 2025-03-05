import torch
import torch.nn as nn

_MAX_CONTEXT_SIZE = 10_000

# combines both embedding and pos_encoding
class Embed(nn.Module):
    def __init__(self, vocab_size, embed_dim, dropout=0):
        super().__init__()
        self.emb_factor = torch.sqrt(torch.tensor(embed_dim, dtype=torch.float32))
        self.embed = nn.Embedding(vocab_size, embed_dim) # vocab x C
        self.dropout = nn.Dropout(dropout)

        pos_embed = torch.zeros(_MAX_CONTEXT_SIZE, embed_dim) # T x C
        position = torch.arange(0, _MAX_CONTEXT_SIZE).unsqueeze(1) # FROM 1 x T to T x 1
        # P.E(pos,2i) = sin(pos/10000^(2i/dim))

        # div_term = 10000 ^([0,1,2,...,C/2-1] * 2/C) <--
        div_term = torch.pow(10_000.0, torch.arange(0, embed_dim//2) * 2/embed_dim) # 1 x C/2 (Embed_dim/2)

        pos_embed[:, 0::2] = torch.sin(position / div_term) # T x C/2 ((T x 1) / (1 x C/2) = T x C/2 broadcasted)
        pos_embed[:, 1::2] = torch.cos(position / div_term) # T x C/2

        self.register_buffer('pos_embed', pos_embed, persistent=False)
        


    def forward(self,x):
        # x = B x T (NOT 1-hot)
        embed_x = self.embed(x) # B T C
        embed_x = embed_x * self.emb_factor # presumably to not be overpowered by the positional encoding

        # ================================
        # For variable length
        # ===============================
        seq_len = x.shape[-1] # length of T
        truc_pos_embed = self.pos_embed[:seq_len,:]
        embed_x = self.dropout(embed_x + truc_pos_embed)
        
        return embed_x
    

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, causal_mask = False, bias=True):
        super().__init__()
        self.dk = embed_dim // num_heads
        self.causal_mask = causal_mask
        self.combined_projection_q = nn.Linear(embed_dim,embed_dim, bias=bias)
        self.combined_projection_k = nn.Linear(embed_dim,embed_dim, bias=bias)
        self.combined_projection_v = nn.Linear(embed_dim,embed_dim, bias=bias)
        self.num_heads = num_heads
        self.multi_linear = nn.Linear(embed_dim,embed_dim, bias=bias)

    def attention(self,q,k,v, padding_mask = None):
        # input shape is B x h x T x dk
        output = (q @ k.transpose(-2,-1)) / torch.sqrt(torch.tensor(self.dk)) # QKt/(sqrt(dk))

        #apply mask in decoder layer
        if self.causal_mask == True:
            seq_len = q.shape[-2]
            # produces 0s in the lower triangle and -inf in the upper triangle
            mask = torch.triu(torch.full((seq_len,seq_len), fill_value=-torch.inf,device=q.device), diagonal=1)
            output = output + mask
        
        # apply padding mask in encoder self-attention and decoder cross-attention
        if padding_mask is not None:
            padding_mask = torch.tensor(padding_mask).unsqueeze(1).unsqueeze(1) # B x 1 x 1 x T (broadcasting)
            padding_mask = torch.where(padding_mask == 0, -torch.inf, 0) # turn 0s to -inf and 1s to 0
            output = output + padding_mask
            

        output = torch.softmax(output, -1)
        output = output @ v
        return output

    def forward(self,x_q,x_k,x_v, padding_mask = None):
        # combined projection, TxC @ CxC
        # Equivalent to doing Txhead @ CxC over all heads
        p_q =  self.combined_projection_q(x_q)
        p_k =  self.combined_projection_k(x_k)
        p_v =  self.combined_projection_v(x_v)

        # For each of QKV.   [B=Batch, T=Time, C=Channels, h=Heads, dk= head dim]

        # ========================|======================
        #         Split           |       Combine
        # ========================|======================
        #   |                   B T C                  /\
        #   |    <view>          |            <view>   |
        #   |                 B T h dk                 |
        #   |    <transpose>     |       <transpose>   |
        #  \/                B h T dk                  |
        #                        |
        #                     <attn>
        # ===============================================


        B = p_q.shape[0]
        def split_heads(p):
            return p.view(B,-1,self.num_heads,self.dk).transpose(1,2)
        
        p_q = split_heads(p_q)
        p_k = split_heads(p_k)
        p_v = split_heads(p_v)

        output = self.attention(p_q,p_k,p_v, padding_mask=padding_mask)

        def combine_heads(p):
            return p.transpose(1,2).contiguous().view(B,-1,self.dk*self.num_heads)
        
        output = combine_heads(output)
        output = self.multi_linear(output)
        return output

    
# This layer is slightly different from standard linear
class PointwiseFeedForward(nn.Module):
    def __init__(self, embed_dim, d_ff):
        super(PointwiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(embed_dim, d_ff, bias=True)
        self.linear2 = nn.Linear(d_ff, embed_dim, bias=True)
    def forward(self, x):
        return self.linear2(nn.functional.relu(self.linear1(x)))

class EncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, d_ff,dropout=0):
        super().__init__()
        # self attention
        self.m_att = MultiHeadAttention(embed_dim, num_heads)
        self.att_norm = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)

        # pointwise feedforward module
        self.pwlinear = PointwiseFeedForward(embed_dim, d_ff)
        self.lin_norm = nn.LayerNorm(embed_dim)
        self.dropout2 = nn.Dropout(dropout)
    def forward(self, x, padding_mask = None):
        output = self.att_norm(x + self.dropout1(self.m_att(x,x,x, padding_mask=padding_mask)))
        output = self.lin_norm(output + self.dropout2(self.pwlinear(output)))
        return output

class EncoderStack(nn.Module):
    def __init__(self, embed_dim, num_heads, num_layers, d_ff, dropout=0):
        super().__init__()
        self.layers = nn.ModuleList([EncoderLayer(embed_dim, num_heads, d_ff, dropout) for i in range(num_layers)])
    def forward(self, x, padding_mask = None):
        for layer in self.layers:
            x = layer(x, padding_mask)
        return x
    
class DecoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, d_ff,dropout=0):
        super().__init__()
        # self causal mask attention module
        self.m_att = MultiHeadAttention(embed_dim, num_heads, causal_mask=True)
        self.att_norm = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)

        # additional cross attention module 
        self.cross_att = MultiHeadAttention(embed_dim, num_heads, causal_mask=False)
        self.cross_att_norm = nn.LayerNorm(embed_dim)
        self.dropout2 = nn.Dropout(dropout)

        # pointwise feedforward module with its layer norm
        self.pwlinear = PointwiseFeedForward(embed_dim, d_ff)
        self.lin_norm = nn.LayerNorm(embed_dim)
        self.dropout3 = nn.Dropout(dropout)
    def forward(self, x, enc_out, enc_padding_mask = None):
        output = self.att_norm(x + self.dropout1(self.m_att(x,x,x))) # self attention
        output = self.cross_att_norm(output + self.dropout2(self.cross_att(output, enc_out,enc_out, padding_mask=enc_padding_mask))) # cross attention
        output = self.lin_norm(output + self.dropout3(self.pwlinear(output))) # pointwise feedforward

        return output

class DecoderStack(nn.Module):
    def __init__(self, embed_dim, num_heads, num_layers, d_ff,dropout=0):
        super().__init__()
        self.layers = nn.ModuleList([DecoderLayer(embed_dim, num_heads, d_ff,dropout) for i in range(num_layers)])
    def forward(self, x, enc_out, enc_padding_mask = None):
        for layer in self.layers:
            x = layer(x, enc_out, enc_padding_mask)
        return x
    
class Transformer(nn.Module):
    def __init__(self, 
                 num_enc_layers, 
                 num_dec_layers, 
                 embed_dim, 
                 num_heads,
                 enc_vocab_size, 
                 dec_vocab_size, 
                 d_ff,
                 dropout=0):
        super().__init__()
        self.emb = Embed(enc_vocab_size, embed_dim) # one embedding for both encoder and decoder

        self.enc = EncoderStack(embed_dim, num_heads, num_enc_layers, d_ff,dropout)
        self.dec = DecoderStack(embed_dim, num_heads, num_dec_layers, d_ff,dropout)

        self.last_lin = nn.Linear(embed_dim, dec_vocab_size, bias=False) # bias false we're tying its weights with the embedding layer
        self.last_lin.weight = self.emb.embed.weight # tying weights
    
    def forward(self, dec_x, enc_x = None, memory = None, enc_padding_mask = None, ):
        if memory is None:
            memory = self.enc(self.emb(enc_x), enc_padding_mask) # Encoder
        dec_out = self.dec(self.emb(dec_x), memory, enc_padding_mask) # Decoder
        logits = self.last_lin(dec_out)
        return {
            "logits": logits,
            "memory": memory
        }
