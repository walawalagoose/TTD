import torch
import torch.nn as nn
from timm.models.vision_transformer import VisionTransformer, Mlp
from timm.models.helpers import checkpoint_seq
import math
from functools import reduce
from operator import mul
import numpy as np

class PromptViT(nn.Module):
    '''
    Vision Transformer with added prompts at the input layer
    '''
    def __init__(self,
                vit:VisionTransformer,
                num_prompts = 1):
        super().__init__()
        self.vit = vit
        self.num_prompts = num_prompts
        self.prompt_dim = vit.embed_dim

        if num_prompts > 0:
            self.prompts = nn.Parameter(torch.zeros(1, num_prompts, self.prompt_dim))
            # initialization adopted from vpt, https://arxiv.org/abs/2203.12119
            val = math.sqrt(6. / float(3 * reduce(mul, vit.patch_embed.patch_size, 1) + self.prompt_dim)) # noqa
            nn.init.uniform_(self.prompts.data, -val, val) # xavier_uniform initialization
    
    def reset(self):
        val = math.sqrt(6. / float(3 * reduce(mul, self.vit.patch_embed.patch_size, 1) + self.prompt_dim)) # noqa
        nn.init.uniform_(self.prompts.data, -val, val) # xavier_uniform initialization

    def prompt_injection(self, x):
        if self.num_prompts > 0:
            x = torch.cat((
                x[:,:1,:],
                self.prompts.expand(x.shape[0],-1,-1),
                x[:,1:,:]
            ), dim=1)
        return x
    
    def _collect_layers_features(self, x):
        # collecting features for each layer
        cls_features = []
        for i in range(len(self.vit.blocks)):
            x = self.vit.blocks[i](x)
            if i < len(self.vit.blocks) - 1:
                cls_features.append(self.vit.blocks[i+1].norm1(x[:, 0]))
            else:
                cls_features.append(self.vit.norm(x[:, 0]))
        cls_features = torch.cat(cls_features, dim=1)
        return cls_features

    def forward_features(self, x):
        '''
        Forwarding a batch of samples with prompts' embeddings inserted
        We added only the highlighted line of code based on `timm` library
        '''
        x = self.vit.patch_embed(x)
        x = self.vit._pos_embed(x)
        # inject prompts
        x = self.prompt_injection(x)
        # !!end
        x = self.vit.norm_pre(x)
        x = self.vit.blocks(x)
        x = self.vit.norm(x)
        return x
    
    def forward(self, x):
        x = self.forward_features(x)
        x = self.vit.forward_head(x)
        return x
    
    def layers_cls_features(self, x):
        x = self.vit.patch_embed(x)
        x = self.vit._pos_embed(x)
        x = self.vit.norm_pre(x)
        return self._collect_layers_features(x)
    
    def layers_cls_features_with_prompts(self, x):
        x = self.vit.patch_embed(x)
        x = self.vit._pos_embed(x)
        # inject prompts
        x = self.prompt_injection(x)
        # !!end
        x = self.vit.norm_pre(x)
        return self._collect_layers_features(x)
    
    def forward_raw_features(self, x):
        '''
        Forwarding a batch of samples without prompts' embeddings inserted
        We added only the highlighted line of code based on `timm` library
        '''
        x = self.vit.patch_embed(x)
        x = self.vit._pos_embed(x)

        # !!end
        x = self.vit.norm_pre(x)
        x = self.vit.blocks(x)
        x = self.vit.norm(x)
        return x
    
    
class MaskedViT(VisionTransformer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward_features(self, x, mask_token=None, unmask=None, return_norm=False):
        x = self.patch_embed(x)
        B, N, _ = x.shape
        device = x.device
        # switch input tokens by mask_token
        if mask_token is not None:
            # mask_tokens = mask_token.module.expand(B, N, -1).to(device)
            mask_tokens = mask_token.expand(B, N, -1).to(device)
            unmask = unmask.unsqueeze(2).to(device) #mask_chosed
            x = x * (1-unmask) + mask_tokens * unmask
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        if self.dist_token is None:
            x = torch.cat((cls_token, x), dim=1)
        else:
            x = torch.cat((cls_token, self.dist_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = self.pos_drop(x + self.pos_embed)

        x = self.blocks(x)
        x = self.norm(x)
        
        x_norm = x
        if self.dist_token is None and return_norm is False:
            return self.pre_logits(x[:, 0])
        elif self.dist_token is None and return_norm is not False:
            return self.pre_logits(x[:, 0]), x_norm
        else:
            return x[:, 0], x[:, 1]

    def forward(self, x, mask_token=None, unmask=None, return_norm=False):
        if return_norm:
            x, x_norm = self.forward_features(x, mask_token, unmask, return_norm)
            if self.head_dist is not None:
                x, x_dist = self.head(x[0]), self.head_dist(x[1])  # x must be a tuple
                if self.training and not torch.jit.is_scripting():
                    return x, x_dist
                else:
                    return (x + x_dist) / 2
            else:
                x = self.head(x)
            return x ,x_norm
        else:
            x = self.forward_features(x)
            if self.head_dist is not None:
                x, x_dist = self.head(x[0]), self.head_dist(x[1])  # x must be a tuple
                if self.training and not torch.jit.is_scripting():
                    return x, x_dist
                else:
                    return (x + x_dist) / 2
            else:
                x = self.head(x)
            return x