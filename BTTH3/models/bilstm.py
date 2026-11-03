import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class BiLstmEncoder(nn.Module):
    '''
        Encoder gồm 5 lớp BiLSTM: trả về vector đặc trưng cho TỪNG token.
    '''
    def __init__(
            self,
            vocab_size: int,
            hidden_size: int,
            n_layer: int,
            padding_idx: int = 0,
            dropout: float = 0.2
    ):
        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=hidden_size,
            padding_idx=padding_idx
        )

        self.bilstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,            # 256 cho mỗi chiều
            num_layers=n_layer,
            batch_first=True,
            bidirectional=True,                 # chiều xuôi + chiều ngược
            dropout=dropout if n_layer > 1 else 0.0
        )

    def forward(self, inputs: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        '''
            inputs: (bs, len), lengths: (bs,)
            output: (bs, len, 2 * hidden)
        '''
        embedded_feats = self.embedding(inputs)         # (bs, len, hidden)

        packed = pack_padded_sequence(
            embedded_feats, lengths.cpu(),
            batch_first=True, enforce_sorted=False
        )
        packed_outputs, _ = self.bilstm(packed)

        # NER cần output ở mọi bước thời gian (không chỉ hidden state cuối)
        outputs, _ = pad_packed_sequence(
            packed_outputs, batch_first=True,
            total_length=inputs.size(1)         # giữ đúng max_len của batch
        )                                       # (bs, len, 2 * hidden)

        return outputs


class BiLstmNER(nn.Module):
    def __init__(
            self,
            vocab_size: int,
            hidden_size: int,
            n_layer: int,
            n_tags: int,
            padding_idx: int = 0,
            dropout: float = 0.2
    ):
        super().__init__()

        self.encoder = BiLstmEncoder(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            n_layer=n_layer,
            padding_idx=padding_idx,
            dropout=dropout
        )

        self.dropout = nn.Dropout(dropout)

        # nối 2 chiều nên đầu vào là 2 * hidden
        self.classifier = nn.Linear(
            in_features=2 * hidden_size,
            out_features=n_tags
        )

    def forward(self, inputs: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        '''
            inputs: (bs, len), lengths: (bs,)
            output: logits (bs, len, n_tags)
        '''
        features = self.encoder(inputs, lengths)        # (bs, len, 2 * hidden)
        features = self.dropout(features)
        logits = self.classifier(features)              # (bs, len, n_tags)

        return logits
