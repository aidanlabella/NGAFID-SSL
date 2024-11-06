import torch.nn as nn
from transformers import BertModel, BertTokenizer

class BERTSimCLR(nn.Module):
    def __init__(self, out_dim):
        super(BERTSimCLR, self).__init__()
        
        # Initialize BERT model and tokenizer
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        
        # Get the dimension of BERT’s last hidden state output
        dim_mlp = self.bert.config.hidden_size  # Typically 768 for bert-base-uncased

        # Add MLP projection head
        self.projection_head = nn.Sequential(
            nn.Linear(dim_mlp, dim_mlp),
            nn.ReLU(),
            nn.Linear(dim_mlp, out_dim)
        )

    def forward(self, input_text):
        # Tokenize input text and get token IDs and attention mask
        inputs = self.tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)
        outputs = self.bert(**inputs)

        # Use the pooled output (for classification) or last hidden state as needed
        pooled_output = outputs.pooler_output
        return self.projection_head(pooled_output)

    # @staticmethod
    def _get_baseline(out_dim):
        """
        Create a baseline instance of the BERTSimCLR model with the specified output dimension.

        Args:
            out_dim (int): The output dimension for the projection head.

        Returns:
            BERTSimCLR: An instance of the BERTSimCLR model initialized for the baseline.
        """
        return BERTSimCLR(out_dim=out_dim)

# Example usage
# baseline_model = BERTSimCLR.get_baseline(out_dim=128)
