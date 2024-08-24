"""
Module for training GPT2 model.

Note: See `bigram_with_self_attention.py` for the actual derivations of all these
      classes originally in much more detail

      Essentially that file is `mini-gpt` whereas this file is `nanogpt`.

"""

import dataclasses
import math

# note: needing to get used to torch.nn since it's just like import numpy as np
import torch

# TODO(bschoen): Why can't we just do "from torch import nn" here?
import torch.nn as nn
from torch.nn import functional as F

from jaxtyping import Float32, Int64


def get_best_available_torch_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


@dataclasses.dataclass(frozen=True)
class GPTConfig:
    # max sequence length
    block_size: int = 1024

    # number of tokens: 50,000 BPE merges + 256 bytes tokens + 1 <|endoftext|> token
    vocab_size: int = 50257

    # number of layers
    n_layer: int = 12

    # number of heads
    n_head: int = 12

    # embedding dimension
    n_embd: int = 768


class CausalSelfAttention(nn.Module):

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()

        assert config.n_embd % config.n_head == 0

        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)

        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

        # regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd

        # not really a 'bias', more of a mask, but following the OpenAI/HF naming though
        self.register_buffer(
            "bias",
            torch.tril(
                torch.ones(
                    config.block_size,
                    config.block_size,
                )
            ).view(1, 1, config.block_size, config.block_size),
        )

    def forward(
        self,
        x: Float32[torch.Tensor, "b t c"],
    ) -> Float32[torch.Tensor, "b t c"]:

        # batch size, sequence length, embedding dimensionality (n_embd)
        B, T, C = x.size()

        # calculate query, key, values for all heads in batch and move head forward
        # to be the batch dim
        #
        # - nh is "number of heads"
        # - hs is "head size"
        # - C (number of channels) = n_head * head_size
        #
        # e.g. in GPT-2 (124M), n_head=12, hs=64, so nh*hs=C=768 channels
        #
        # note: do all of QKV at once, this is why `c_attn` is of size `3 * n_embd`
        qkv = self.c_attn(x)

        q, k, v = qkv.split(self.n_embd, dim=2)

        # (B, n_head, T, head_size)
        #
        #  - we split the channels so we can do heads in parallel?
        #  - Is this essentially batching over heads?
        #  - Is the more general principle anything we can parallelize we can batch
        #    over?
        #
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        # attention (materializes the large (T,T) matrix for all the queries and keys)
        #
        # note: this is our previous attention implementation
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)

        # re-assemble all head outputs side by side
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # output projection
        y = self.c_proj(y)

        return y


class MLP(nn.Module):

    # why GELU?
    #  - [Gaussian Error Linear Units (GELUs)](https://arxiv.org/abs/1606.08415)
    #  - Basically like ReLU but with a smooth transition around zero
    #  - This is a good thing because it allows for more stable training / gradients /
    #    activations
    #
    #  - "It relates to stochastic regularizers in that it is the expectation of a
    #     modification to Adaptive Dropout"
    #
    #  - approx version only exists from github issue where it was because
    #    ERF was slow in tensorflow
    #
    #  - today would prefer exact version, but everyone picked it up
    #
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate="tanh")
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(
        self,
        x: Float32[torch.Tensor, "b t c"],
    ) -> Float32[torch.Tensor, "b t c"]:
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x


# Every one of these iteratively refines the representation in the residual stream
class Block(nn.Module):

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    # Karpathy: Clean residual stream is desirable from an optimization perspective,
    #           but here we have layer norm _inside_ of our residual stream
    def forward(
        self,
        x: Float32[torch.Tensor, "b t c"],
    ) -> Float32[torch.Tensor, "b t c"]:

        # "repeated application of map reduce"

        # communication - weighted sum function, reduce function
        x = x + self.attn(self.ln_1(x))

        # happens to every single token indivudally - the map function
        x = x + self.mlp(self.ln_2(x))

        return x


# note: cross-attention was only reason needed encoder (from original attention paper)
class GPT(nn.Module):
    """

    Here we basically try to work backwards from this, which gives us a model
    we can use with the loaded weights + replicates the structure.

        model = transformers.GPT2LMHeadModel.from_pretrained('gpt2')

        model.state_dict()

        # key                                shape
        # ---------------------------------  ------------------------
        transformer.wte.weight               torch.Size([50257, 768])
        transformer.wpe.weight               torch.Size([1024, 768])
        transformer.h.0.ln_1.weight          torch.Size([768])
        transformer.h.0.ln_1.bias            torch.Size([768])
        transformer.h.0.attn.c_attn.weight   torch.Size([768, 2304])
        transformer.h.0.attn.c_attn.bias     torch.Size([2304])
        transformer.h.0.attn.c_proj.weight   torch.Size([768, 768])
        transformer.h.0.attn.c_proj.bias     torch.Size([768])
        transformer.h.0.ln_2.weight          torch.Size([768])
        transformer.h.0.ln_2.bias            torch.Size([768])
        transformer.h.0.mlp.c_fc.weight      torch.Size([768, 3072])
        transformer.h.0.mlp.c_fc.bias        torch.Size([3072])
        transformer.h.0.mlp.c_proj.weight    torch.Size([3072, 768])
        transformer.h.0.mlp.c_proj.bias      torch.Size([768])
        # note: layer norms at the inputs
        transformer.h.1.ln_1.weight          torch.Size([768])
        transformer.h.1.ln_1.bias            torch.Size([768])
        transformer.h.1.attn.c_attn.weight   torch.Size([768, 2304])
        transformer.h.1.attn.c_attn.bias     torch.Size([2304])
        ...
        transformer.h.11.ln_1.weight         torch.Size([768])
        transformer.h.11.ln_1.bias           torch.Size([768])
        transformer.h.11.attn.c_attn.weight  torch.Size([768, 2304])
        transformer.h.11.attn.c_attn.bias    torch.Size([2304])
        transformer.h.11.attn.c_proj.weight  torch.Size([768, 768])
        transformer.h.11.attn.c_proj.bias    torch.Size([768])
        transformer.h.11.ln_2.weight         torch.Size([768])
        transformer.h.11.ln_2.bias           torch.Size([768])
        transformer.h.11.mlp.c_fc.weight     torch.Size([768, 3072])
        transformer.h.11.mlp.c_fc.bias       torch.Size([3072])
        transformer.h.11.mlp.c_proj.weight   torch.Size([3072, 768])
        transformer.h.11.mlp.c_proj.bias     torch.Size([768])
        # making up for layer norms at the inputs
        transformer.ln_f.weight              torch.Size([768])
        transformer.ln_f.bias                torch.Size([768])
        # map back to vocab
        lm_head.weight                       torch.Size([50257, 768])

    We're also exactly following the same variable names

    """

    # Karpathy: "not super interesting"
    @classmethod
    def from_pretrained(cls, model_type):
        """Loads pretrained GPT-2 model weights from huggingface"""
        assert model_type in {"gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"}
        from transformers import GPT2LMHeadModel

        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            "gpt2": dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            "gpt2-medium": dict(n_layer=24, n_head=16, n_embd=1024),  # 350M params
            "gpt2-large": dict(n_layer=36, n_head=20, n_embd=1280),  # 774M params
            "gpt2-xl": dict(n_layer=48, n_head=25, n_embd=1600),  # 1558M params
        }[model_type]

        config_args["vocab_size"] = 50257  # always 50257 for GPT model checkpoints
        config_args["block_size"] = 1024  # always 1024 for GPT model checkpoints

        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = cls(config)
        sd = model.state_dict()
        sd_keys = sd.keys()

        # discard this mask / buffer, not a param
        sd_keys = [k for k in sd_keys if not k.endswith(".attn.bias")]

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()

        # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith(".attn.masked_bias")]

        # same, just the mask (buffer)
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith(".attn.bias")]
        transposed = [
            "attn.c_attn.weight",
            "attn.c_proj.weight",
            "mlp.c_fc.weight",
            "mlp.c_proj.weight",
        ]

        # basically the openai checkpoints use a "Conv1D" module,
        # but we only want to use a vanilla Linear this means that we have to
        # transpose these weights when we import them
        if len(sd_keys_hf) != len(sd_keys):
            raise ValueError(f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}")

        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):

                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape

                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:

                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape

                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.config = config

        # essentially just like `nn.Sequential` but allows indexing by name
        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                wpe=nn.Embedding(config.block_size, config.n_embd),
                h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                ln_f=nn.LayerNorm(config.n_embd),
            )
        )

        # language model head - we give it literally the same weights
        #                       as the token embedding, which is how it's the
        #                       unembedding!
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # weight sharing scheme
        # note: can tell this because they're using the same pointer
        # actual approach comes from original attention paper
        #
        # this enforces that bottom and top of transformer
        # - similar tokens have similar embeddings
        # - similar embeddings have similar logits
        # they observed this was happening naturally so just went ahead and tied them
        #
        # this is also absolutely massive (40M params, 30% params saved)
        # meaning you can train it longer
        self.transformer.wte.weight = self.lm_head.weight

        # init params
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        """Initialize weights according to source code

        Note:
            - Typically you'd want to initialize `std` with `1 / sqrt(n)` where `n` is
              the number of features in the input tensor.
            - 0.02 is roughly consistent with that for the d_model sizes in GPT-2

        """

        if isinstance(module, nn.Linear):

            std = 0.02

            # note: this is a flag we set in other modules for init lol
            if hasattr(module, "NANOGPT_SCALE_INIT"):

                std *= (2 * self.config.n_layer) ** -0.5

            torch.nn.init.normal_(module.weight, mean=0.0, std=std)

            if module.bias is not None:

                torch.nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Embedding):

            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    # `idx` := token indices
    def forward(
        self,
        idx: Int64[torch.Tensor, "b t"],
        target: Int64[torch.Tensor, "b t"] | None = None,
    ) -> tuple[Float32[torch.Tensor, "b t c"], Float32[torch.Tensor, ""] | None]:

        # idx is of shape (B, T)
        B, T = idx.size()

        if T > self.config.block_size:
            raise ValueError(
                f"Cannot forward sequence of length {T}, "
                f"block size is only {self.config.block_size}"
            )

        # forward the token and posisition embeddings

        # note: we can use the same device as another tensor, so don't have to
        #       pass device everywhere
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)  # shape (T)
        pos_emb = self.transformer.wpe(pos)  # position embeddings of shape (T, n_embd)
        tok_emb = self.transformer.wte(idx)  # token embeddings of shape (B, T, n_embd)

        # note: broadcasting hidden inside this `+` over the batch dimension
        x = tok_emb + pos_emb

        # forward the blocks of the transformer
        for block in self.transformer.h:
            x = block.forward(x)

        # forward the final layernorm and the classifier
        x = self.transformer.ln_f(x)

        # forward through language model head (unembedding)
        logits = self.lm_head(x)  # (B, T, vocab_size)

        # calculate `loss` (if `target` given, like during training`)
        loss = None
        if target is not None:
            # again cross entropy doesn't like multidimensional input, so we flatten
            # everything out
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target.view(-1))

        # "logits are just a softmax away from becoming probabilities"
        return logits, loss
