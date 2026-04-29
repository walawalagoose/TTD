
import math
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .clip import load, tokenize
from .simple_tokenizer import SimpleTokenizer as _Tokenizer

_tokenizer = _Tokenizer()

DOWNLOAD_ROOT='~/.cache/clip'

class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype) # (batch_size, n_ctx, transformer.width), NLD
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype) 

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x


class PromptLearner(nn.Module):
    def __init__(self, clip_model, classnames, n_ctx=16, ctx_init=None, ctx_position='end', device="cuda"):
        super().__init__()
        n_cls = len(classnames)
        dtype = clip_model.dtype
        self.dtype = dtype
        self.device = device
        ctx_dim = clip_model.ln_final.weight.shape[0]
        self.ctx_dim = ctx_dim

        if ctx_init:
            # use given words to initialize context vectors
            print("Initializing the contect with given words: [{}]".format(ctx_init))
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = tokenize(ctx_init).to(self.device)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
            prompt_prefix = ctx_init
        else:
            print("Random initialization: initializing a generic context")
            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)
        
        self.prompt_prefix = prompt_prefix

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        # batch-wise prompt tuning for test-time adaptation
        self.ctx = nn.Parameter(ctx_vectors) # to be optimized

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([tokenize(p) for p in prompts]).to(self.device)
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])  # CLS, EOS

        self.ctx_init = ctx_init
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens
        self.class_token_position = ctx_position
        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.classnames = classnames

    def forward(self):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix

        if self.class_token_position == "end":
            prompts = torch.cat(
                [
                    prefix,  # (n_cls, 1, dim)
                    ctx,     # (n_cls, n_ctx, dim)
                    suffix,  # (n_cls, *, dim)
                ],
                dim=-2,
            )
        elif self.class_token_position == "middle":
            half_n_ctx = self.n_ctx // 2
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i_half1 = ctx[i : i + 1, :half_n_ctx, :]
                ctx_i_half2 = ctx[i : i + 1, half_n_ctx:, :]
                prompt = torch.cat(
                    [
                        prefix_i,     # (1, 1, dim)
                        ctx_i_half1,  # (1, n_ctx//2, dim)
                        class_i,      # (1, name_len, dim)
                        ctx_i_half2,  # (1, n_ctx//2, dim)
                        suffix_i,     # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        elif self.class_token_position == "front":
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i = ctx[i : i + 1, :, :]
                prompt = torch.cat(
                    [
                        prefix_i,  # (1, 1, dim)
                        class_i,   # (1, name_len, dim)
                        ctx_i,     # (1, n_ctx, dim)
                        suffix_i,  # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        else:
            raise ValueError

        return prompts


class ClipTestTimeTuning(nn.Module):
    def __init__(self, device, classnames, arch="ViT-B/16",
                        n_ctx=8, ctx_init="a photo of a", ctx_position='end'):
        super(ClipTestTimeTuning, self).__init__()
        clip_model, _ = load(arch, device=device, download_root=DOWNLOAD_ROOT)
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        
        self.logit_scale = clip_model.logit_scale
        self.input_resolution = 224  # CLIP的标准输入分辨率
        self.dtype = clip_model.dtype
        
        # prompt tuning
        self.prompt_learner = PromptLearner(clip_model, classnames, n_ctx, ctx_init, ctx_position,device=device)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts

    def forward(self, image):
        image_features = self.image_encoder(image.type(self.dtype))
        
        prompts = self.prompt_learner()
        tokenized_prompts = self.tokenized_prompts
        text_features = self.text_encoder(prompts, tokenized_prompts)
        
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()

        return logits

class ClipZeroShot(nn.Module):
    def __init__(self, device, classnames, arch="ViT-B/16"):
        super(ClipZeroShot, self).__init__()
        # 加载CLIP模型
        self.clip_model, _ = load(arch, device=device, download_root=DOWNLOAD_ROOT)
        self.device = device
        self.dtype = self.clip_model.dtype
        self.logit_scale = self.clip_model.logit_scale
        
        # 准备类别提示文本
        self.classnames = classnames
        self.prompts = [f"a photo of a {classname}" for classname in classnames]
        
        # 标记化提示文本
        self.tokenized_prompts = tokenize(self.prompts).to(device)
        
        # 预计算文本特征，因为它们是固定的
        with torch.no_grad():
            self.text_features = self.clip_model.encode_text(self.tokenized_prompts)
            # 归一化文本特征
            self.text_features = self.text_features / self.text_features.norm(dim=-1, keepdim=True)
    
    def forward(self, image):
        # 提取并归一化图像特征
        with torch.no_grad():
            image_features = self.clip_model.encode_image(image)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        # 计算logits
        logit_scale = self.clip_model.logit_scale.exp()
        logits = logit_scale * image_features @ self.text_features.t()
        # logits = image_features @ self.text_features.t()

        return logits
    

class ClipZeroShot4TTA(nn.Module):
    def __init__(self, device, classnames, arch="ViT-B/16"):
        super(ClipZeroShot4TTA, self).__init__()
        # 加载CLIP模型
        self.clip_model, _ = load(arch, device=device, download_root=DOWNLOAD_ROOT)
        self.device = device
        self.dtype = self.clip_model.dtype
        
        # 准备类别提示文本
        self.classnames = classnames
        self.prompts = [f"a photo of a {classname}" for classname in classnames]
        self.n_classes = len(classnames)
        
        # 标记化提示文本
        self.tokenized_prompts = tokenize(self.prompts).to(device)
        
        # 预计算文本特征，因为它们是固定的
        with torch.no_grad():
            self.text_features = self.clip_model.encode_text(self.tokenized_prompts)
            # 归一化文本特征
            self.text_features = self.text_features / self.text_features.norm(dim=-1, keepdim=True)
    
    def forward(self, image):
        # 提取并归一化图像特征
        with torch.no_grad():
            image_features = self.clip_model.encode_image(image)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        # 计算logits
        logit_scale = self.clip_model.logit_scale.exp()
        # logits = logit_scale * image_features @ self.text_features.t()
        unscaled_logits = image_features @ self.text_features.t()
        logits = logit_scale * unscaled_logits

        return logits, unscaled_logits, image_features
    
class ClipTestTimeTuningNorm(nn.Module):
    def __init__(self, device, classnames, arch="ViT-B/16", ctx_init="a photo of a"):
        super(ClipTestTimeTuningNorm, self).__init__()
        # 加载CLIP模型
        self.clip_model, _ = load(arch, device=device, download_root=DOWNLOAD_ROOT)
        self.device = device
        self.dtype = self.clip_model.dtype
        
        # 准备类别提示文本
        self.classnames = classnames
        if ctx_init is None:
            self.prompts = [f"a photo of a {classname}" for classname in classnames]
        else:
            self.prompts = [ctx_init + f" {classname}" for classname in classnames]
        self.n_classes = len(classnames)
        
        # 标记化提示文本
        self.tokenized_prompts = tokenize(self.prompts).to(device)
    
    def forward(self, image, return_features=True):
        # 提取并归一化图像特征
        image_features = self.clip_model.encode_image(image)
        pre_image_features = image_features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            
        # 提取并归一化文本特征
        text_features = self.clip_model.encode_text(self.tokenized_prompts)
        pre_text_features = text_features
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        # 计算logits
        logit_scale = self.clip_model.logit_scale.exp()
        unscaled_logits = image_features @ text_features.t()
        logits = logit_scale * unscaled_logits

        if return_features:
            return logits, image_features, text_features, pre_image_features, pre_text_features
        else:
            return logits