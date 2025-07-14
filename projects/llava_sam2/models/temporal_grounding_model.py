import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
import pdb
import math

# class TemporalPositionalEncoding(nn.Module):
#     def __init__(self, hidden_dim=1024, max_frames=32):
#         super().__init__()
#         self.hidden_dim = hidden_dim
#         self.max_frames = max_frames
        
#         # 只需要时间编码！InternViT已经处理了空间编码
#         self.temporal_pos_embedding = nn.Embedding(max_frames, hidden_dim)
#         self._init_weights()
    
#     def _init_weights(self):
#         self._init_temporal_sinusoidal()
    
#     def _init_temporal_sinusoidal(self):
#         position = torch.arange(0, self.max_frames).unsqueeze(1).float()
#         div_term = torch.exp(torch.arange(0, self.hidden_dim, 2).float() * -(math.log(10000.0) / self.hidden_dim))
        
#         sinusoidal_table = torch.zeros(self.max_frames, self.hidden_dim)
#         sinusoidal_table[:, 0::2] = torch.sin(position * div_term)
#         sinusoidal_table[:, 1::2] = torch.cos(position * div_term)
        
#         self.temporal_pos_embedding.weight.data.copy_(sinusoidal_table)
    
#     def forward(self, features, frame_positions):
#         batch_size, num_frames, num_patches, hidden_dim = features.shape
        
#         temporal_emb = self.temporal_pos_embedding(frame_positions)  # (B, T, D)
#         temporal_emb = temporal_emb.unsqueeze(2)  # (B, T, 1, D)
#         temporal_emb = temporal_emb.expand(-1, -1, num_patches, -1)  # (B, T, N, D)
        
#         encoded_features = features + temporal_emb
        
#         return encoded_features


# class TemporalCrossAttention(nn.Module):
#     def __init__(self, query_dim=1024, key_dim=1024, attention_dim=1024, num_layers=6, num_heads=8, dropout=0.1):
#         super().__init__()
#         self.query_dim = query_dim
#         self.key_dim = key_dim
#         self.attention_dim = attention_dim
#         self.num_layers = num_layers
#         self.num_heads = num_heads
        
#         self.query_projections = nn.ModuleList([
#             nn.Linear(query_dim, attention_dim) for _ in range(num_layers)
#         ])
        
#         self.key_projections = nn.ModuleList([
#             nn.Linear(key_dim, attention_dim) for _ in range(num_layers)
#         ])
        
#         self.value_projections = nn.ModuleList([
#             nn.Linear(key_dim, attention_dim) for _ in range(num_layers)
#         ])
        
#         self.output_projections = nn.ModuleList([
#             nn.Linear(attention_dim, query_dim) for _ in range(num_layers)
#         ])
        
#         self.multihead_attns = nn.ModuleList([
#             nn.MultiheadAttention(
#                 embed_dim=attention_dim,
#                 num_heads=num_heads,
#                 dropout=dropout,
#                 batch_first=True
#             ) for _ in range(num_layers)
#         ])
        
#         self.query_norms = nn.ModuleList([
#             nn.LayerNorm(query_dim) for _ in range(num_layers)
#         ])
        
#         self.ffns = nn.ModuleList([
#             nn.Sequential(
#                 nn.Linear(query_dim, query_dim * 4),
#                 nn.GELU(),
#                 nn.Dropout(dropout),
#                 nn.Linear(query_dim * 4, query_dim),
#                 nn.Dropout(dropout)
#             ) for _ in range(num_layers)
#         ])
        
#         self.ffn_norms = nn.ModuleList([
#             nn.LayerNorm(query_dim) for _ in range(num_layers)
#         ])
        
#     def forward(self, query_tokens, visual_features):
#         """
#         Args:
#             query_tokens: (batch_size, num_query_tokens, query_dim)
#             visual_features: (batch_size, num_frames * num_patches, key_dim)
        
#         Returns:
#             output: (batch_size, num_query_tokens, query_dim)
#         """
#         output = query_tokens
        
#         for i in range(self.num_layers):
#             Q = self.query_projections[i](output)
#             K = self.key_projections[i](visual_features)
#             V = self.value_projections[i](visual_features)
            
#             attn_output, _ = self.multihead_attns[i](query=Q, key=K, value=V)
            
#             attn_output = self.output_projections[i](attn_output)
            
#             output = self.query_norms[i](output + attn_output)
            
#             ffn_output = self.ffns[i](output)
#             output = self.ffn_norms[i](output + ffn_output)
        
#         return output


# class QueryToFrameProjection(nn.Module):
#     def __init__(self, query_dim=1024, llm_dim=4096):
#         super().__init__()

#         self.projection = nn.Sequential(
#             nn.Linear(query_dim, query_dim * 2),
#             nn.LayerNorm(query_dim * 2),
#             nn.GELU(),
#             nn.Linear(query_dim * 2, llm_dim),
#             nn.LayerNorm(llm_dim)
#         )
        
#     def forward(self, query_tokens):
#         frame_features = self.projection(query_tokens)
        
#         return frame_features

######################################
# 复杂版本
class QueryToFrameProjection(nn.Module):
    def __init__(self, query_dim=1024, llm_dim=4096, num_layers=3):
        super().__init__()
        self.input_proj = nn.Linear(query_dim, query_dim * 2)
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(query_dim * 2, query_dim * 2),
                nn.LayerNorm(query_dim * 2),
                nn.GELU()
            ) for _ in range(num_layers)
        ])
        self.output_proj = nn.Sequential(
            nn.Linear(query_dim * 2, query_dim * 4),
            nn.LayerNorm(query_dim * 4), 
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(query_dim * 4, llm_dim),
            nn.LayerNorm(llm_dim)
        )
        
    def forward(self, query_tokens):
        x = self.input_proj(query_tokens)
        for layer in self.layers:
            x = x + layer(x)
        frame_features = self.output_proj(x)
        return frame_features
######################################

# class TemporalGroundingProcessor:
#     """Temporal Grounding处理器
    
#     封装所有temporal grounding相关的处理逻辑
#     在llava_sam2_box2的forward中调用
#     """
#     def __init__(self, 
#                  num_query_tokens=128,
#                  query_hidden_dim=1024,
#                  cross_attn_layers=6,
#                  cross_attn_heads=8,
#                  max_frames=32,
#                  vision_hidden_dim=1024,
#                  llm_hidden_dim=4096):
        
#         self.num_query_tokens = num_query_tokens
#         self.query_hidden_dim = query_hidden_dim
#         self.max_frames = max_frames
        
#         # 初始化组件（需要在外部初始化并传入）
#         self.temporal_encoding = None
#         self.temporal_cross_attention = None
#         self.query_to_frame = None
#         self.query_tokens = None
        
#     def setup_components(self, device, dtype):
#         """设置所有组件（在模型初始化时调用）"""
#         self.temporal_encoding = TemporalPositionalEncoding(
#             hidden_dim=1024,
#             max_frames=self.max_frames
#         ).to(device=device, dtype=dtype)
        
#         self.temporal_cross_attention = TemporalCrossAttention(
#             query_dim=self.query_hidden_dim,
#             key_dim=1024,  # InternViT输出维度
#             num_layers=6,
#             num_heads=8
#         ).to(device=device, dtype=dtype)
        
#         self.query_to_frame = QueryToFrameProjection(
#             query_dim=self.query_hidden_dim,
#             llm_dim=4096,
#         ).to(device=device, dtype=dtype)
        
#         # 初始化query tokens
#         self.query_tokens = nn.Parameter(
#             torch.randn(1, self.num_query_tokens, self.query_hidden_dim, 
#                        device=device, dtype=dtype)
#         )
#         nn.init.normal_(self.query_tokens, std=0.02)
        
#     def process_visual_features(self, visual_features, frame_positions=None):
#         """
#         处理视觉特征
        
#         Args:
#             visual_features: (batch_size, num_frames, num_patches, hidden_dim)
#             frame_positions: (batch_size, num_frames) 可选
            
#         Returns:
#             frame_features: (batch_size, num_frames, llm_dim) - 用于LLM的帧特征
#         """
#         batch_size = visual_features.shape[0]
#         device = visual_features.device
        
#         # 如果没有提供frame_positions，使用默认序列
#         if frame_positions is None:
#             frame_positions = torch.arange(
#                 visual_features.shape[1], 
#                 device=device
#             ).unsqueeze(0).expand(batch_size, -1)
        
#         # 1. 时空编码
#         encoded_features = self.temporal_encoding(visual_features, frame_positions)
        
#         # 2. Flatten为交叉注意力输入
#         num_frames, num_patches, hidden_dim = encoded_features.shape[1:]
#         flat_features = encoded_features.view(batch_size, num_frames * num_patches, hidden_dim)
        
#         # 3. 扩展query tokens
#         query_tokens = self.query_tokens.expand(batch_size, -1, -1)
        
#         # 4. 交叉注意力
#         attended_queries = self.temporal_cross_attention(query_tokens, flat_features)
        
#         # 5. 映射到帧表示
#         frame_features = self.query_to_frame(attended_queries)
        
#         return frame_features


class PooledTemporalPositionalEncoding(nn.Module):
    def __init__(self, hidden_dim=1024, max_frames=64):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_frames = max_frames
        self.temporal_pos_embedding = nn.Embedding(max_frames, hidden_dim)
        self._init_weights()
    
    def _init_weights(self):
        self._init_temporal_sinusoidal()
    
    def _init_temporal_sinusoidal(self):
        position = torch.arange(0, self.max_frames).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, self.hidden_dim, 2).float() * -(math.log(10000.0) / self.hidden_dim))
        sinusoidal_table = torch.zeros(self.max_frames, self.hidden_dim)
        sinusoidal_table[:, 0::2] = torch.sin(position * div_term)
        sinusoidal_table[:, 1::2] = torch.cos(position * div_term)
        self.temporal_pos_embedding.weight.data.copy_(sinusoidal_table)
    
    def forward(self, features, frame_positions):
        temporal_emb = self.temporal_pos_embedding(frame_positions)
        encoded_features = features + temporal_emb
        return encoded_features