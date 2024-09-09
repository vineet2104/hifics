"""
===============================================================================
Title:           hifics.py
Date:            June 23, 2024
Description:     This script contains the architecture of the Visual Grounding Model HiFi-CS, featuring CLIP as a multi-modal backend and a lightweight segmentation decoder.  Parts of this script are taken from the ClipSeg repository                 on GitHub. License applied as: https://github.com/timojl/clipseg?tab=License-1-ov-file#readme
===============================================================================
"""

import math
from os.path import basename, dirname, join
import torch
from torch import nn
from torch.nn import functional as nnf
from torch.nn.modules.activation import ReLU      


def forward_multihead_attention(x, b, with_aff=False, attn_mask=None):
    """ 
    Simplified version of multihead attention (taken from torch source code but without tons of if clauses). 
    The mlp and layer norm come from CLIP.
    x: input.
    b: multihead attention module. 
    """

    x_ = b.ln_1(x)
    q, k, v = nnf.linear(x_, b.attn.in_proj_weight, b.attn.in_proj_bias).chunk(3, dim=-1)
    tgt_len, bsz, embed_dim = q.size()

    head_dim = embed_dim // b.attn.num_heads
    scaling = float(head_dim) ** -0.5

    q = q.contiguous().view(tgt_len, bsz * b.attn.num_heads, b.attn.head_dim).transpose(0, 1)
    k = k.contiguous().view(-1, bsz * b.attn.num_heads, b.attn.head_dim).transpose(0, 1)
    v = v.contiguous().view(-1, bsz * b.attn.num_heads, b.attn.head_dim).transpose(0, 1)

    q = q * scaling

    attn_output_weights = torch.bmm(q, k.transpose(1, 2)) #  n_heads * batch_size, tokens^2, tokens^2
    if attn_mask is not None:


        attn_mask_type, attn_mask = attn_mask
        n_heads = attn_output_weights.size(0) // attn_mask.size(0)
        attn_mask = attn_mask.repeat(n_heads, 1)
        
        if attn_mask_type == 'cls_token':
            # the mask only affects similarities compared to the readout-token.
            attn_output_weights[:, 0, 1:] = attn_output_weights[:, 0, 1:] * attn_mask[None,...]

        if attn_mask_type == 'all':
            attn_output_weights[:, 1:, 1:] = attn_output_weights[:, 1:, 1:] * attn_mask[:, None]
        
    
    attn_output_weights = torch.softmax(attn_output_weights, dim=-1)

    attn_output = torch.bmm(attn_output_weights, v)
    attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
    attn_output = b.attn.out_proj(attn_output)

    x = x + attn_output
    x = x + b.mlp(b.ln_2(x))

    if with_aff:
        return x, attn_output_weights
    else:
        return x


class CLIPDenseBase(nn.Module):

    def __init__(self, version, reduce_cond, reduce_dim, prompt, n_tokens):
        super().__init__()

        import clip

        # prec = torch.FloatTensor
        self.clip_model, _ = clip.load(version, device='cpu', jit=False)
        self.model = self.clip_model.visual
        self.version = version 

        # if not None, scale conv weights such that we obtain n_tokens.
        self.n_tokens = n_tokens

        for p in self.clip_model.parameters():
            p.requires_grad_(False)

        # conditional
        if reduce_cond is not None:
            self.reduce_cond = nn.Linear(512, reduce_cond)
            for p in self.reduce_cond.parameters():
                p.requires_grad_(False)
        else:
            self.reduce_cond = None  

        if(self.version=='ViT-B/16'):
            film_dim = 512
        elif(self.version=='ViT-L/14'):
            film_dim = 768
        
        self.film_mul = nn.Linear(film_dim if reduce_cond is None else reduce_cond, reduce_dim)
        self.film_add = nn.Linear(film_dim if reduce_cond is None else reduce_cond, reduce_dim)
        
        self.film_mul_0 = nn.Linear(film_dim if reduce_cond is None else reduce_cond, reduce_dim)
        self.film_add_0 = nn.Linear(film_dim if reduce_cond is None else reduce_cond, reduce_dim)
        self.film_mul_1 = nn.Linear(film_dim if reduce_cond is None else reduce_cond, reduce_dim)
        self.film_add_1 = nn.Linear(film_dim if reduce_cond is None else reduce_cond, reduce_dim)
        self.film_mul_2 = nn.Linear(film_dim if reduce_cond is None else reduce_cond, reduce_dim)
        self.film_add_2 = nn.Linear(film_dim if reduce_cond is None else reduce_cond, reduce_dim)
        self.film_mul_3 = nn.Linear(film_dim if reduce_cond is None else reduce_cond, reduce_dim)
        self.film_add_3 = nn.Linear(film_dim if reduce_cond is None else reduce_cond, reduce_dim)
        self.film_mul_4 = nn.Linear(film_dim if reduce_cond is None else reduce_cond, reduce_dim)
        self.film_add_4 = nn.Linear(film_dim if reduce_cond is None else reduce_cond, reduce_dim)

        if(self.version=='ViT-B/16'):
            self.reduce = nn.Linear(768, reduce_dim)
        elif(self.version=='ViT-L/14'):
            self.reduce = nn.Linear(1024, reduce_dim)
    
    def rescaled_pos_emb(self, new_size):
        assert len(new_size) == 2
        
        if(self.version=='ViT-B/16'):
            pos_emb_size = 768
        else:
            pos_emb_size = 1024
        a = self.model.positional_embedding[1:].T.view(1, pos_emb_size, *self.token_shape)
        b = nnf.interpolate(a, new_size, mode='bicubic', align_corners=False).squeeze(0).view(pos_emb_size, new_size[0]*new_size[1]).T
        return torch.cat([self.model.positional_embedding[:1], b])

    def visual_forward(self, x_inp, extract_layers=(), skip=False, mask=None):
        

        with torch.no_grad():

            inp_size = x_inp.shape[2:]

            if self.n_tokens is not None:
                stride2 = x_inp.shape[2] // self.n_tokens
                conv_weight2 = nnf.interpolate(self.model.conv1.weight, (stride2, stride2), mode='bilinear', align_corners=True)
                x = nnf.conv2d(x_inp, conv_weight2, bias=self.model.conv1.bias, stride=stride2, dilation=self.model.conv1.dilation)
            else:
                x = self.model.conv1(x_inp)  # shape = [bs, width, grid, grid] width=768, grid=22

            x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
            x = x.permute(0, 2, 1)  # shape = [bs, grid^2, width]
            
            x = torch.cat([self.model.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
            
            standard_n_tokens = 50 if self.model.conv1.kernel_size[0] == 32 else 197

            if x.shape[1] != standard_n_tokens: 
                new_shape = int(math.sqrt(x.shape[1]-1)) # = grid
                x = x + self.rescaled_pos_emb((new_shape, new_shape)).to(x.dtype)[None,:,:]
            else:
                x = x + self.model.positional_embedding.to(x.dtype)

            
            x = self.model.ln_pre(x) # Layer normalization
            
            
            x = x.permute(1, 0, 2)  # NLD -> LND

            activations, affinities = [], []
            
            for i, res_block in enumerate(self.model.transformer.resblocks):
                    # The input is processed through the clip's pipeline and activations and weights are extracted after every residual block.
                if mask is not None:
                    mask_layer, mask_type, mask_tensor = mask
                    if mask_layer == i or mask_layer == 'all':
                        size = int(math.sqrt(x.shape[0] - 1))
                        
                        attn_mask = (mask_type, nnf.interpolate(mask_tensor.unsqueeze(1).float(), (size, size)).view(mask_tensor.shape[0], size * size))
                        
                    else:
                        attn_mask = None
                else:
                    attn_mask = None
                
                # res_block is a residual attention block. 
                x, aff_per_head = forward_multihead_attention(x, res_block, with_aff=True, attn_mask=attn_mask)
                
                # x has a size of [grid^2+1,bs,768]
                # aff_per_head is the attention weights of dimension [192,grid^2+1,grid^2+1]

                if i in extract_layers:
                    
                    # if i is one of the layers whose weights are to be extracted
                    affinities += [aff_per_head]

                    activations += [x]
                    

                if len(extract_layers) > 0 and i == max(extract_layers) and skip:
                    print('early skip')
                    break
                
            x = x.permute(1, 0, 2)  # LND -> NLD
            x = self.model.ln_post(x[:, 0, :])

            if self.model.proj is not None:
                x = x @ self.model.proj

            return x, activations, affinities

    def sample_prompts(self, words, prompt_list=None):

        prompt_list = prompt_list if prompt_list is not None else self.prompt_list

        prompt_indices = torch.multinomial(torch.ones(len(prompt_list)), len(words), replacement=True)
        prompts = [prompt_list[i] for i in prompt_indices]
        return [promt.format(w) for promt, w in zip(prompts, words)]

    def get_cond_vec(self, conditional, batch_size):
        # compute conditional from a single string
        #conditional is a tuple of strings, with size corresponding to the batch_size
        if conditional is not None and type(conditional) == str:
            cond = self.compute_conditional(conditional) # features extracted from CLIP
            cond = cond.repeat(batch_size, 1) # cond is now a tensor of size (bs,512) after extracting textual features from CLIP

        # compute conditional from string list/tuple
        elif conditional is not None and type(conditional) in {list, tuple} and type(conditional[0]) == str:
            assert len(conditional) == batch_size
            cond = self.compute_conditional(conditional)

        # use conditional directly
        elif conditional is not None and type(conditional) == torch.Tensor and conditional.ndim == 2:
            cond = conditional

        # compute conditional from image
        elif conditional is not None and type(conditional) == torch.Tensor:
            with torch.no_grad():
                cond, _, _ = self.visual_forward(conditional)
        else:
            raise ValueError('invalid conditional')
        
        return cond   

    def compute_conditional(self, conditional):
        import clip
        dev = next(self.parameters()).device

        if type(conditional) in {list, tuple}:

            text_tokens = clip.tokenize(conditional).to(dev)
            cond = self.clip_model.encode_text(text_tokens)
                
        else:
            if conditional in self.precomputed_prompts:
                cond = self.precomputed_prompts[conditional].float().to(dev)
            else:
                text_tokens = clip.tokenize([conditional]).to(dev)
                cond = self.clip_model.encode_text(text_tokens)[0]
        
        if self.shift_vector is not None:
            return cond + self.shift_vector
        else:
            return cond
  

class CLIPDensePredT(CLIPDenseBase):

    def __init__(self, version='ViT-B/32', extract_layers=(3, 6, 9), cond_layer=0, reduce_dim=128, n_heads=4, prompt='fixed', 
                 extra_blocks=0, reduce_cond=None, fix_shift=False,
                 learn_trans_conv_only=False,  limit_to_clip_only=False, upsample=False, 
                 add_calibration=False, rev_activations=False, trans_conv=None, n_tokens=None, complex_trans_conv=False,extended_film=False):
        
        super().__init__(version, reduce_cond, reduce_dim, prompt, n_tokens)

        self.extended_film = extended_film
        self.extract_layers = extract_layers
        self.cond_layer = cond_layer
        self.limit_to_clip_only = limit_to_clip_only
        self.process_cond = None
        self.rev_activations = rev_activations
        self.version = version
        
        depth = len(extract_layers)

        if add_calibration:
            self.calibration_conds = 1

        self.upsample_proj = nn.Conv2d(reduce_dim, 1, kernel_size=1) if upsample else None

        self.add_activation1 = True

        self.version = version
        
        self.token_shape = {'ViT-B/32': (7, 7), 'ViT-B/16': (14, 14), 'ViT-L/14': (16,16)}[version]

        if fix_shift:
            self.shift_vector = nn.Parameter(torch.load(join(dirname(basename(__file__)), 'shift_text_to_vis.pth')), requires_grad=False)
        else:
            self.shift_vector = None

        if trans_conv is None:
            trans_conv_ks = {'ViT-B/32': (32, 32), 'ViT-B/16': (16, 16), 'ViT-L/14': (14,14)}[version]
        else:
            # explicitly define transposed conv kernel size
            trans_conv_ks = (trans_conv, trans_conv)

        if not complex_trans_conv:
            self.trans_conv = nn.ConvTranspose2d(reduce_dim, 1, trans_conv_ks, stride=trans_conv_ks)
        else:
            
            assert trans_conv_ks[0] == trans_conv_ks[1]

            tp_kernels = (trans_conv_ks[0] // 4, trans_conv_ks[0] // 4)

            self.trans_conv = nn.Sequential(
                nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.ConvTranspose2d(reduce_dim, reduce_dim // 2, kernel_size=tp_kernels[0], stride=tp_kernels[0]),
                nn.ReLU(),
                nn.ConvTranspose2d(reduce_dim // 2, 1, kernel_size=tp_kernels[1], stride=tp_kernels[1]),               
            )
        
        assert len(self.extract_layers) == depth
        if(self.version=='ViT-B/16'):
            self.reduces = nn.ModuleList([nn.Linear(768, reduce_dim) for _ in range(depth)])
        elif(self.version=='ViT-L/14'):
            self.reduces = nn.ModuleList([nn.Linear(1024, reduce_dim) for _ in range(depth)])
        self.blocks = nn.ModuleList([nn.TransformerEncoderLayer(d_model=reduce_dim, nhead=n_heads) for _ in range(len(self.extract_layers))]) # transformer decoder blocks
        self.extra_blocks = nn.ModuleList([nn.TransformerEncoderLayer(d_model=reduce_dim, nhead=n_heads) for _ in range(extra_blocks)])
        

        if learn_trans_conv_only:
            for p in self.parameters():
                p.requires_grad_(False)
            
            for p in self.trans_conv.parameters():
                p.requires_grad_(True)


    def forward(self, inp_image, conditional=None, return_features=False, mask=None):

        assert type(return_features) == bool

        inp_image = inp_image.to(self.model.positional_embedding.device)

        if mask is not None:
            raise ValueError('mask not supported')

        x_inp = inp_image

        bs, dev = inp_image.shape[0], x_inp.device

        cond = self.get_cond_vec(conditional, bs) # tensor of size (bs,512) corresponding to text embeddings
        
        visual_q, activations, _ = self.visual_forward(x_inp, extract_layers=[0] + list(self.extract_layers))
        # visual_q is a tensor of size (bs,512)
        # activations is a list of size = number of layers to be extracted (1+len(self.extract_layers))
        # each element of activations is a tensor of size (grid**2 + 1,bs,768)
        
        activation1 = activations[0]
        activations = activations[1:]

        _activations = activations[::-1] if not self.rev_activations else activations

        a = None

        for i, (activation, block, reduce) in enumerate(zip(_activations, self.blocks, self.reduces)):
            
            if a is not None:
                a = reduce(activation) + a
            else:
                a = reduce(activation)

            if(self.extended_film):
                if(self.reduce_cond is not None):
                    temp_cond = self.reduce_cond(cond)
                    if(i==0):
                        a = self.film_mul_0(temp_cond) * a + self.film_add_0(temp_cond)
                    elif(i==1):
                        a = self.film_mul_1(temp_cond) * a + self.film_add_1(temp_cond)
                    elif(i==2):
                        a = self.film_mul_2(temp_cond) * a + self.film_add_2(temp_cond)
                    elif(i==3):
                        a = self.film_mul_3(temp_cond) * a + self.film_add_3(temp_cond)
                    elif(i==4):
                        a = self.film_mul_4(temp_cond) * a + self.film_add_4(temp_cond)
                else:
                    if(i==0):
                        a = self.film_mul_0(cond) * a + self.film_add_0(cond)
                    elif(i==1):
                        a = self.film_mul_1(cond) * a + self.film_add_1(cond)
                    elif(i==2):
                        a = self.film_mul_2(cond) * a + self.film_add_2(cond)
                    elif(i==3):
                        a = self.film_mul_3(cond) * a + self.film_add_3(cond)
                    elif(i==4):
                        a = self.film_mul_4(cond) * a + self.film_add_4(cond)

            else:
                if i == self.cond_layer:
                    if self.reduce_cond is not None:
                        cond = self.reduce_cond(cond)
                    
                    # this is the way the features are fused together. 
                    # cond is the vector which contains text embeddings
                    # text embeddings pass through a linear layer and multipled by activation
                    # they also pass through another linear layer and added to the result
                    
                    a = self.film_mul(cond) * a + self.film_add(cond)

            a = block(a)

        for block in self.extra_blocks:
            a = a + block(a)
        
        a = a[1:].permute(1, 2, 0) # rm cls token and -> BS, Feats, Tokens

        size = int(math.sqrt(a.shape[2]))

        a = a.view(bs, a.shape[1], size, size)

        a = self.trans_conv(a)
        

        if self.n_tokens is not None:
            a = nnf.interpolate(a, x_inp.shape[2:], mode='bilinear', align_corners=True) 

        if self.upsample_proj is not None:
            a = self.upsample_proj(a)
            a = nnf.interpolate(a, x_inp.shape[2:], mode='bilinear')

        
        if return_features:
            return a, visual_q, cond, [activation1] + activations
        else:
            return a,

                    
