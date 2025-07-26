import torch
import torch.nn as nn
from transformers import AutoModel

class CrossModalFakeVideoDetector(nn.Module):
    def __init__(self, text_model_name="bert-base-uncased", video_feat_dim=512, hidden_dim=768, num_classes=2):
        super(CrossModalFakeVideoDetector, self).__init__()

        self.text_model = AutoModel.from_pretrained(text_model_name)

        self.video_cnn = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, video_feat_dim)
        )

        self.cross_modal_attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=8, batch_first=True)

        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, input_ids, attention_mask, video_frames):
        text_output = self.text_model(input_ids=input_ids, attention_mask=attention_mask)
        text_feats = text_output.last_hidden_state

        batch_size, seq_len, C, H, W = video_frames.size()
        video_feats = self.video_cnn(video_frames.view(-1, C, H, W))
        video_feats = video_feats.view(batch_size, seq_len, -1)

        video_feats_proj = nn.Linear(video_feats.shape[-1], text_feats.shape[-1]).to(video_feats.device)(video_feats)

        attn_output, _ = self.cross_modal_attn(query=text_feats, key=video_feats_proj, value=video_feats_proj)

        pooled_output = attn_output.mean(dim=1)
        logits = self.classifier(pooled_output)
        return logits