from transformers import CLIPTokenizer, CLIPTextModel, BertModel, BertTokenizer
import torch
from torch import nn

class TextEmbedder(nn.Module):
    def __init__(self, version: str, max_length: int, d_model:int, output_dim:int, local_pt, **hf_kwargs):
        super().__init__()
        self.is_clip = version.startswith("openai")
        self.max_length = max_length
        self.output_key = "last_hidden_state" 

        if self.is_clip:
            self.tokenizer: CLIPTokenizer = CLIPTokenizer.from_pretrained(version, max_length=max_length)
            self.hf_module: CLIPTextModel = CLIPTextModel.from_pretrained(version, **hf_kwargs)
        else:
            self.tokenizer: BertTokenizer = BertTokenizer.from_pretrained(version, max_length=max_length)
            self.hf_module: BertModel = BertModel.from_pretrained(version, **hf_kwargs)
        self.hf_module = self.hf_module.eval().requires_grad_(False)
        if local_pt is not None:
            state_dict = torch.load(local_pt)
            new_state_dict = {k.replace('transformer.',''):v for k,v in state_dict.items()}
            load_result = self.hf_module.load_state_dict(new_state_dict, strict=False)
        hidden_dim = (d_model + output_dim) // 2
        self.hf_module.pooler= nn.Identity()
        self.pooler = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, text: list[str]) -> torch.Tensor:
        batch_encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            return_length=False,
            return_overflowing_tokens=False,
            padding="max_length",
            return_tensors="pt",
        )
        outputs = self.hf_module(
            input_ids=batch_encoding["input_ids"].to(self.hf_module.device),
            attention_mask=batch_encoding['attention_mask'].to(self.hf_module.device),
        )
        pooled_out = self.pooler(outputs[self.output_key][:,0,:])
        return pooled_out
    
    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.transformer.gradient_checkpointing_enable()