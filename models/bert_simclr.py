import torch
import torch.nn as nn
from transformers import BertModel

class BERTSimCLR(nn.Module):
    def __init__(self, out_dim, max_len=10000):

        super(BERTSimCLR, self).__init__()
        
        self.bert = BertModel.from_pretrained("bert-base-uncased")

        self.bert.embeddings.position_embeddings = nn.Embedding(max_len, self.bert.config.hidden_size)
        self.bert.config.max_position_embeddings = max_len  

        self.input_mapping = nn.Linear(44, self.bert.config.hidden_size)

        dim_mlp = self.bert.config.hidden_size  
        self.projection_head = nn.Sequential(
            nn.Linear(dim_mlp, dim_mlp),
            nn.ReLU(),
            nn.Linear(dim_mlp, out_dim)
        )

    def forward(self, input_data):

        batch_size, seq_len, _ = input_data.size()
        
        input_data_mapped = self.input_mapping(input_data)
        
        position_ids = torch.arange(seq_len, dtype=torch.long, device=input_data.device).unsqueeze(0).expand(batch_size, -1)
        
        token_type_ids = torch.zeros((batch_size, seq_len), dtype=torch.long, device=input_data.device)
        
        outputs = self.bert(inputs_embeds=input_data_mapped, position_ids=position_ids, token_type_ids=token_type_ids)
        
        pooled_output = outputs.pooler_output
        
        print(pooled_output.shape)

        return self.projection_head(pooled_output)


    def remove_projector(self):
        self.projection_head = nn.Identity()


