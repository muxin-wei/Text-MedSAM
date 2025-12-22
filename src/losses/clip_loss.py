import torch
import torch.nn as nn
import torch.nn.functional as F

class CLIPLoss(nn.Module):
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
        self.logit_scale = nn.Parameter(torch.ones([]) * 2.6592)

    def forward(self, image_features: torch.Tensor, text_features: torch.Tensor, batch_idx):
        """
        Args:
            image_features: [B, C] normalized features
            text_features: [B, C] normalized features
        """
        image_features = F.normalize(image_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)

        logit_scale = self.logit_scale.exp().clamp(max=100)
        logits = logit_scale * (image_features @ text_features.T)
        
        batch_size = image_features.shape[0]
        labels = torch.arange(batch_size, device=image_features.device)
        if batch_idx % 50 == 0:
            with torch.no_grad():
                sim_matrix = image_features @ text_features.t()
                diag_mean = sim_matrix.diag().mean()
                off_diag_mean = (sim_matrix.sum() - sim_matrix.diag().sum()) / (batch_size**2 - batch_size)
                print(f"\n[Step {batch_idx}] Pos Sim: {diag_mean:.4f} | Neg Sim: {off_diag_mean:.4f}")
                print(f"Logit Scale: {logit_scale.item():.2f}") # 
        
        loss_i2t = F.cross_entropy(logits, labels)
        loss_t2i = F.cross_entropy(logits.T, labels)
        return (loss_i2t + loss_t2i) / 2.0
