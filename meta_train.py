import torch
from torch import nn
from collections import namedtuple
import math
from utils import get_device
from pytorch_transformers.modeling_bert import (
    BertEncoder,
    BertPreTrainedModel,
    BertConfig,
)

DEVICE = get_device()

class GeLU(nn.Module):
    """Implementation of the gelu activation function.
    For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
    0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
    Also see https://arxiv.org/abs/1606.08415
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class BertLayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root)."""
        super(BertLayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


class mlp_meta(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(config.hid_dim, config.hid_dim),
            GeLU(),
            BertLayerNorm(config.hid_dim, eps=1e-12),
            nn.Dropout(config.dropout),
        )

    def forward(self, x):
        return self.mlp(x)


class Bert_Transformer_Layer(BertPreTrainedModel):
    def __init__(self, fusion_config):
        super().__init__(BertConfig(**fusion_config))
        bertconfig_fusion = BertConfig(**fusion_config)
        self.encoder = BertEncoder(bertconfig_fusion)
        self.init_weights()

    def forward(self, input, mask=None):
        """
        input:(bs, 4, dim)
        """
        batch, feats, dim = input.size()
        if mask is not None:
            mask_ = torch.ones(size=(batch, feats), device=mask.device)
            mask_[:, 1:] = mask
            mask_ = torch.bmm(
                mask_.view(batch, 1, -1).transpose(1, 2), mask_.view(batch, 1, -1)
            )
            mask_ = mask_.unsqueeze(1)

        else:
            mask = torch.Tensor([1.0]).to(input.device)
            mask_ = mask.repeat(batch, 1, feats, feats)

        extend_mask = (1 - mask_) * -10000
        assert not extend_mask.requires_grad
        head_mask = [None] * self.config.num_hidden_layers

        enc_output = self.encoder(input, extend_mask, head_mask=head_mask)
        output = enc_output[0]
        all_attention = enc_output[1]

        return output, all_attention


class mmdPreModel(nn.Module):
    def __init__(
        self,
        config,
        num_mlp=0,
        transformer_flag=False,
        num_hidden_layers=1,
        mlp_flag=True,
    ):
        super(mmdPreModel, self).__init__()
        self.num_mlp = num_mlp
        self.transformer_flag = transformer_flag
        self.mlp_flag = mlp_flag
        token_num = config.token_num
        self.mlp = nn.Sequential(
            nn.Linear(config.in_dim, config.hid_dim),
            GeLU(),
            BertLayerNorm(config.hid_dim, eps=1e-12),
            nn.Dropout(config.dropout),
            # nn.Linear(config.hid_dim, config.out_dim),
        )
        self.fusion_config = {
            "hidden_size": config.in_dim,
            "num_hidden_layers": num_hidden_layers,
            "num_attention_heads": 4,
            "output_attentions": True,
        }
        if self.num_mlp > 0:
            self.mlp2 = nn.ModuleList([mlp_meta(config) for _ in range(self.num_mlp)])
        if self.transformer_flag:
            self.transformer = Bert_Transformer_Layer(self.fusion_config)
        self.feature = nn.Linear(config.hid_dim * token_num, config.out_dim)

    def forward(self, features):
        """
        input: [batch, token_num, hidden_size], output: [batch, token_num * config.out_dim]
        """

        if self.transformer_flag:
            features, _ = self.transformer(features)
        if self.mlp_flag:
            features = self.mlp(features)

        if self.num_mlp > 0:
            # features = self.mlp2(features)
            for _ in range(1):
                for mlp in self.mlp2:
                    features = mlp(features)

        features = self.feature(features.view(features.shape[0], -1))
        return features  # features.view(features.shape[0], -1)


class NetLoader:
    def __init__(self):
        token_num, hidden_size = 100, 768
        Config = namedtuple(
            "Config", ["in_dim", "hid_dim", "dropout", "out_dim", "token_num"]
        )
        config = Config(
            in_dim=hidden_size,
            token_num=token_num,
            hid_dim=512,
            dropout=0.2,
            out_dim=300,
        )
        self.config = config
        self.net = mmdPreModel(
            config=config, num_mlp=0, transformer_flag=True, num_hidden_layers=1
        )
        checkpoint_filename = "./net.pt"
        checkpoint = torch.load(checkpoint_filename, map_location=DEVICE)
        self.net.load_state_dict(checkpoint["net"])
        self.sigma, self.sigma0_u, self.ep = (
            checkpoint["sigma"],
            checkpoint["sigma0_u"],
            checkpoint["ep"],
        )
        self.net = self.net.to(DEVICE)
        self.sigma, self.sigma0_u, self.ep = (
            self.sigma.to(DEVICE),
            self.sigma0_u.to(DEVICE),
            self.ep.to(DEVICE),
        )


net = NetLoader()
