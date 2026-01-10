import torch
import torch.nn as nn
from transformers import BertConfig, BertModel

class SmallBERT(nn.Module):
    def __init__(self, vocab_size=33555, max_length=512, dropout_prob=0.1):

        super(SmallBERT, self).__init__()
        
        # HuggingFace BertConfig
        # L=4, H=256, A=8
        self.config = BertConfig(
            vocab_size=vocab_size,          
            hidden_size=256,                    # Embedding Size
            num_hidden_layers=6,                # Encoder layers
            num_attention_heads=8,              # Attention Heads (256 / 8 = 32 per head)
            intermediate_size=1024,             # FFN dimensions (hidden_size * 4)
            max_position_embeddings=max_length, # length of longest sequence
            type_vocab_size=512,                  # token_type_ids (Segment embeddings)
            hidden_dropout_prob=dropout_prob,
            attention_probs_dropout_prob=dropout_prob,
            pad_token_id=1                  
        )
        
        # random init BERT
        self.bert = BertModel(self.config)

    def forward(self, input_ids, attention_mask, token_type_ids):
        """
        Args:
            input_ids: [Batch, Seq_Len]
            attention_mask: [Batch, Seq_Len]
            token_type_ids: [Batch, Seq_Len]
        Returns:
            func_embedding: [Batch, Hidden_Size=256] -> [CLS] Vector
        """
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids = token_type_ids)
        
        # Get [CLS] Token Vector (Batch, Hidden_Size)
        # last_hidden_state shape: [Batch, Seq_Len, Hidden_Size]
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        
        return cls_embedding

    def save_pretrained(self, save_directory):
        self.bert.save_pretrained(save_directory)

    @classmethod
    def from_pretrained(cls, load_directory):
        model = cls()
        model.bert = BertModel.from_pretrained(load_directory)
        return model