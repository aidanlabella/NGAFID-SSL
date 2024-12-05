import torch
import torch.nn as nn
from transformers import DistilBertModel, DistilBertConfig

class DistilBERTSimCLR(nn.Module):
    def __init__(self, out_dim, max_len=10000):
        super(DistilBERTSimCLR, self).__init__()
        
        config = DistilBertConfig.from_pretrained("distilbert-base-uncased")
        config.max_position_embeddings = max_len
        
        self.bert = DistilBertModel.from_pretrained("distilbert-base-uncased", config=config)
        
        self.bert.embeddings.position_embeddings = nn.Embedding(max_len, self.bert.config.hidden_size)
        
        self.input_mapping = nn.Linear(44, self.bert.config.hidden_size)
        
        dim_mlp = self.bert.config.hidden_size
        self.projection_head = nn.Sequential(
            nn.Linear(dim_mlp, dim_mlp),
            nn.ReLU(),
            nn.Linear(dim_mlp, out_dim)
        )
    
    def forward(self, input_data):

        input_data = input_data.squeeze(1)
        batch_size, seq_len, _ = input_data.size()
        
        input_data_mapped = self.input_mapping(input_data)  # Shape: [batch_size, seq_len, hidden_size]
        
        position_ids = torch.arange(seq_len, dtype=torch.long, device=input_data.device).unsqueeze(0).expand(batch_size, -1)
        
        outputs = self.bert(inputs_embeds=input_data_mapped, position_ids=position_ids)
        
        cls_output = outputs.last_hidden_state[:, 0, :]  # Shape: [batch_size, hidden_size]
        
        projection = self.projection_head(cls_output)  # Shape: [batch_size, out_dim]
        
        return projection
    
    def remove_projector(self):
        self.projection_head = nn.Identity()
