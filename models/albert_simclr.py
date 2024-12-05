import torch
import torch.nn as nn
from transformers import AlbertModel, AlbertConfig

class ALBERTSimCLR(nn.Module):
    def __init__(self, out_dim, max_len=10000, freeze_albert=True):
        super(ALBERTSimCLR, self).__init__()
        
        config = AlbertConfig.from_pretrained("albert-base-v2")
        config.max_position_embeddings = max_len
        
        self.albert = AlbertModel.from_pretrained("albert-base-v2", config=config)
        
        if freeze_albert:
            for param in self.albert.parameters():
                param.requires_grad = False
        
        original_max_len = self.albert.config.max_position_embeddings
        if max_len > original_max_len:
            self.albert.embeddings.position_embeddings = nn.Embedding(max_len, self.albert.config.hidden_size)
            
            nn.init.normal_(self.albert.embeddings.position_embeddings.weight, mean=0.0, std=self.albert.config.initializer_range)
            
            self.albert.config.max_position_embeddings = max_len
        
        self.input_mapping = nn.Linear(44, self.albert.config.hidden_size)
        
        dim_mlp = self.albert.config.hidden_size
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
        
        outputs = self.albert(inputs_embeds=input_data_mapped, position_ids=position_ids)
        
        pooled_output = outputs.pooler_output  
        
        projection = self.projection_head(pooled_output)  # Shape: [batch_size, out_dim]
        
        return projection
    
    def remove_projector(self):
        self.projection_head = nn.Identity()
