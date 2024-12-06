import torch
import torch.nn as nn
from transformers import AlbertModel, AlbertConfig

class ALBERTSimCLR(nn.Module):
    def __init__(self, out_dim, max_len=10000):
        super(ALBERTSimCLR, self).__init__()
        
        # Load a pre-trained ALBERT model
        self.albert = AlbertModel.from_pretrained("albert-base-v2")
        
        # Replace position embeddings to allow for longer sequences
        self.albert.embeddings.position_embeddings = nn.Embedding(max_len, self.albert.config.embedding_size)
        self.albert.config.max_position_embeddings = max_len
        
        # Map the input features (44-dim) to the ALBERT hidden_size
        self.input_mapping = nn.Linear(44, self.albert.config.embedding_size)
        
        # Create a projection head (MLP) that maps ALBERT output to the desired out_dim
        dim_mlp = self.albert.config.hidden_size
        self.projection_head = nn.Sequential(
            nn.Linear(dim_mlp, dim_mlp),
            nn.ReLU(),
            nn.Linear(dim_mlp, out_dim)
        )

    def forward(self, input_data):
        # input_data shape expected: [batch_size, 1, seq_len, 44]
        input_data = input_data.squeeze(1)  # remove the dimension of size 1
        batch_size, seq_len, _ = input_data.size()
        
        # Map input_data to ALBERT hidden_size
        input_data_mapped = self.input_mapping(input_data)
        
        # Create position_ids for each token in the sequence
        position_ids = torch.arange(seq_len, dtype=torch.long, device=input_data.device).unsqueeze(0).expand(batch_size, -1)
        
        # ALBERT does not require token_type_ids in the same way as BERT, but we provide them for consistency
        token_type_ids = torch.zeros((batch_size, seq_len), dtype=torch.long, device=input_data.device)
        
        # Forward pass through ALBERT
        outputs = self.albert(inputs_embeds=input_data_mapped, position_ids=position_ids, token_type_ids=token_type_ids)
        
        # pooled_output: [batch_size, hidden_size]
        pooled_output = outputs.pooler_output
        print(pooled_output.shape)
        
        # Projection head maps pooled_output to the out_dim
        return self.projection_head(pooled_output)

    def remove_projector(self):
        self.projection_head = nn.Identity()
