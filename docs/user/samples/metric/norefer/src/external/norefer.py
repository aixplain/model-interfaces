import torch
from torch import nn
from transformers import AutoTokenizer, AutoModel


class Smish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * (x.sigmoid() + 1).log().tanh()


class NoRefER(nn.Module):
    def __init__(self, model_name: str, max_length: int = 128):
        super().__init__()
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

        hidden_size = 32

        self.dense = nn.Sequential(nn.Dropout(0.1), nn.Linear(384, hidden_size, bias=False), nn.Dropout(0.1), Smish(), nn.Linear(hidden_size, 1, bias=False))

    def forward(self, x):
        hyps_inputs = self.tokenizer(x, return_tensors="pt", truncation=True, padding=True, max_length=self.max_length)
        h = self.model(**hyps_inputs).pooler_output

        return self.dense(h).sigmoid().squeeze(-1)

    def load_from_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model_weights = checkpoint["state_dict"]
        self.load_state_dict(model_weights)
        self.eval()

    def get_score(self, sentence):
        sentence = sentence.lower()
        score = self.forward(sentence).item()
        return score
