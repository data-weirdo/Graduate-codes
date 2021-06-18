import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss, TransformerEncoderLayer, TransformerEncoder
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from transformers import BertPreTrainedModel, BertModel


# LSTM Model
class LSTMModel(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.embedding = nn.Embedding(30522, dim)
        self.lstm = nn.LSTM(input_size = dim,
                            hidden_size = dim*2,
                            bidirectional = True,
                            batch_first = True)
        self.dropout = nn.Dropout(p=0.2)
        self.qa_outputs = nn.Linear(dim*4, 2)

    def forward(
        self, 
        input_ids, 
        attention_mask, 
        token_type_ids, 
        start_positions=None,
        end_positions=None,
    ):

        # Preprocess to use pack_padded_sequence
        lengths = torch.sum(torch.where(input_ids != 0, 1, 0), axis=1)
        sort_indices = torch.sort(lengths, descending=True).indices
        sorted_lengths = torch.sort(lengths, descending=True).values
        input_id_sorted = input_ids[sort_indices]

        # Embeds on Preprocessed input
        embeds = self.embedding(input_id_sorted)
        packed_input = pack_padded_sequence(input=embeds,
                                            lengths=sorted_lengths.tolist(),
                                            batch_first=True)

        # Actual forward step at LSTM
        output, _ = self.lstm(packed_input)
        output, _ = pad_packed_sequence(output, batch_first=True)
        output = self.dropout(output)
        logits = self.qa_outputs(output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        total_loss = None
        if start_positions is not None and end_positions is not None:
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)

            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

        output = (start_logits, end_logits)
        return ((total_loss,) + output) if total_loss is not None else output



# Transformer Model
class TransformerModel(nn.Module):
    def __init__(self, dim, num_heads=8, num_layers=None):
        super().__init__()

        self.num_heads = num_heads
        self.embedding = nn.Embedding(30522, dim)
        encoder_layer = TransformerEncoderLayer(d_model = dim, 
                                                nhead = num_heads, 
                                                dim_feedforward = dim*4)
        self.transformer_enc = TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.qa_outputs = nn.Linear(dim, 2)

    def forward(
        self, 
        input_ids, 
        attention_mask, 
        token_type_ids, 
        start_positions=None,
        end_positions=None,
    ):

        embeds = self.embedding(input_ids)
        # attention_mask = ~attention_mask.unsqueeze(-1).expand(-1, -1, attention_mask.shape[1]).repeat(self.num_heads, 1, 1)
        # attention_mask = attention_mask.float().masked_fill(attention_mask == 0, float('-inf')).masked_fill(attention_mask == 1, float(0.0))
        # b, t = input_ids.size() 
        # attention_mask = attention_mask.unsqueeze(1).unsqueeze(1).expand(-1, self.num_heads, t, -1).reshape(b * self.num_heads, t, -1)
        # attention_mask = attention_mask == 0

        attention_mask = torch.where(attention_mask == 1, False, True)
        

        output = self.transformer_enc(embeds.transpose(0,1), src_key_padding_mask = attention_mask)
        logits = self.qa_outputs(output)

        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).transpose(0,1)
        end_logits = end_logits.squeeze(-1).transpose(0,1)

        total_loss = None
        if start_positions is not None and end_positions is not None:
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

        output = (start_logits, end_logits)
        return ((total_loss,) + output) if total_loss is not None else output




# BERT model (from pre-trained checkpoint)
class BERTModel(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.bert = BertModel(config, add_pooling_layer=False)
        self.qa_outputs = nn.Linear(config.hidden_size, 2)
        self.init_weights()

    def forward(
        self,
        input_ids,
        attention_mask,
        token_type_ids,
        start_positions=None,
        end_positions=None,
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        sequence_output = outputs[0]
        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        total_loss = None
        if start_positions is not None and end_positions is not None:
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

        output = (start_logits, end_logits)
        return ((total_loss,) + output) if total_loss is not None else output