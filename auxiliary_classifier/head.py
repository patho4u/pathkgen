import torch
import torch.nn as nn

class ConceptHead(nn.Module):
    def __init__(self, emb_dim, num_sites, site_emb_dim, num_classes):
        super().__init__()

        self.site_emb = nn.Embedding(num_sites, site_emb_dim)

        self.fc = nn.Sequential(
            nn.Linear(emb_dim + site_emb_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x, site_id):
        site_vec = self.site_emb(site_id)
        x = torch.cat([x, site_vec], dim=1)
        return self.fc(x)