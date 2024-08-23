from math import ceil, sqrt, exp, log

import torch
import timm

BACKBONE_RES = 384
QUALITY_RES = 512

BACKBONE_DIM = 1152
AESTHETIC_DIM = 16
QUALITY_DIM = 128
HIDDEN_DIM = 64
DECISION_DIM = 8
N_FREE = 2

N_ACCEPT = 9
N_REJECT = 21
N_HEADS = N_ACCEPT + N_REJECT

OUTPUT_NAMES = [
    # ACCEPT
    "3d_realism",
    "3d_human",
    "3d_scene",
    "3d_environment",

    "3d_untex",
    "3d_untex_human",

    "2d_fantasy",
    "2d_human",
    "2d_anime",

    # REJECT
    "2d_sketch",
    "2d_bad",
    "2d_cute",

    "3d_model_bad",
    "3d_model_simple",
    "3d_scene_bad",
    "3d_untex_bad",

    "3d_multi",
    "3d_untex_multi",
    "2d_multi",
    "multi_img",

    "3d_cute",
    "3d_no_fur",
    "3d_miniature",
    "3d_rig",
    "3d_wireframe",

    "text",
    "real_human",
    "technology",

    "ui",
    "degraded",

    # DECISION
    "accept"
]

def init_linear(
    layer: torch.nn.Module,
    nonlinearity: str = "lealy_relu",
    negative_slope: float = 0.25,
    bias: float | None = None,
) -> None:
    fan_in = None

    if nonlinearity == "asig":
        fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(layer.weight)
        std = 1.8626 / sqrt(fan_in)
        torch.nn.init.normal_(layer.weight, std=std)
    else:
        torch.nn.init.kaiming_normal_(layer.weight, nonlinearity=nonlinearity, a=negative_slope)

    if layer.bias is None:
        if bias is not None:
            raise ValueError("layer has no bias parameter")

        return

    if bias is None:
        if fan_in is None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(layer.weight)

        bound = sqrt(3.0) / sqrt(fan_in)
        torch.nn.init.uniform_(layer.bias, -bound, bound)
    else:
        torch.nn.init.constant_(layer.bias, bias)

class ARNIQAEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.model = torch.nn.Sequential(
            *timm.create_model("resnet50", num_classes=0).children()
        )
        self.projector = torch.nn.Sequential(
            torch.nn.Linear(2048, 2048),
            torch.nn.ReLU(),
            torch.nn.Linear(2048, 128)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projector(torch.nn.functional.normalize(self.model(x)))

class AestheticPredictorV2p5Head(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.scoring_head = torch.nn.Sequential(
            torch.nn.Linear(BACKBONE_DIM, 1024),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(1024, 128),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(128, 64),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(64, 16),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(16, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.nn.functional.normalize(x, dim=-1)

        for module in self.scoring_head[:-2]:
            x = module(x)

        return x

class AdaptiveSigmoid(torch.nn.Module):
    def __init__(self,
        shape: tuple[int, ...] | int = 1,
        init: str | dict[str, tuple[float, float] | float | None] = "normal",
        *,
        gate_mode: str = "self",
        constant_alpha: bool = False,
        constant_gamma_m: bool = False,
        constant_gamma_p: bool = False,
    ):
        super().__init__()

        if gate_mode not in ("self", "hybrid", "external", "bilinear", "none"):
            raise ValueError("invalid gate mode")

        self.gate_mode = gate_mode

        if isinstance(shape, int):
            shape = (shape,)

        if isinstance(init, str):
            ha = 10.0 if constant_alpha else 4.0
            hm = 10.0 if constant_gamma_m else 4.0
            hp = 10.0 if constant_gamma_p else 4.0

            match init:
                case "soft_linear":
                    init = { "alpha": -ha, "gamma_m": -hm, "gamma_p": hp }
                case "silu":
                    init = { "alpha": -ha, "gamma_m": 1.0, "gamma_p": 1.0 }
                case "soft_silu":
                    init = { "alpha": 1.0, "gamma_m": 0.25, "gamma_p": 1.0 }
                case "silu_symmetric":
                    init = { "alpha": -ha, "gamma_m": -1.0, "gamma_p": 1.0 }
                case "silu_normal":
                    init = { "alpha": -ha, "gamma_m": (0.0, 1/3), "gamma_p": (1.0, 1/6) }
                case "soft_silu_normal":
                    init = { "alpha": (1.0, 0.5), "gamma_m": (0.0, 1/3), "gamma_p": (1.0, 1/6) }
                case "normal":
                    init = { "alpha": (0.0, 1.0), "gamma_m": (0.0, 1/3), "gamma_p": (1.0, 1/6) }
                case "empty":
                    init = { "alpha": None, "gamma_m": None, "gamma_p": None }
                case _:
                    raise ValueError("invalid init")

        self.alpha = self._make_param(shape, init["alpha"], constant_alpha)

        if self.gate_mode in ("self", "hybrid"):
            self.gamma_m = self._make_param(shape, init["gamma_m"], constant_gamma_m)
            self.gamma_p = self._make_param(shape, init["gamma_p"], constant_gamma_p)

            if constant_gamma_m and constant_gamma_p and self.gamma_m <= -9.0 and self.gamma_p >= 9.0:
                raise ValueError(f"gate is constant and ineffective, use gate_mode=\"none\" or adjust gammas")

            if not isinstance(self.gamma_p, float):
                with torch.no_grad():
                    self.gamma_p.copy_((self.gamma_p - 1.0).abs())

        if constant_alpha:
            if self.alpha >= 9.0:
                self.alpha = 1.0
            elif self.alpha <= -9.0:
                self.alpha = 0.0
            else:
                self.alpha = 1 / (1 + exp(-self.alpha))

    @staticmethod
    def _make_param(
        shape: tuple[int, ...],
        init: tuple[float, float] | float | None,
        const: bool,
    ) -> torch.nn.Parameter | float:
        if const:
            if init is None:
                return 0.0

            return init if isinstance(init, float) else init[0]

        param = torch.nn.Parameter(torch.empty(*shape))

        if init is None:
            pass
        elif isinstance(init, float):
            torch.nn.init.constant_(param, init)
        else:
            torch.nn.init.normal_(param, init[0], init[1])

        return param

    def forward(self, x: torch.Tensor, g: torch.Tensor | None = None) -> torch.Tensor:
        if isinstance(self.alpha, float):
            if self.alpha == 0.0:
                a = x.tanh()
            elif self.alpha == 1.0:
                a = x
            else:
                a = torch.lerp(x, x.tanh(), self.alpha)
        else:
            a = torch.lerp(x, x.tanh(), self.alpha.sigmoid())

        if self.gate_mode in ("self", "none"):
            if g is not None:
                raise ValueError(f"extraneous gate value for {self.gate_mode}-gated activation")

            if self.gate_mode == "none":
                return a

            g = x
        elif g is None:
            raise ValueError(f"gate value required for {self.gate_mode}-gated activation")

        if self.gate_mode in ("self", "hybrid"):
            if isinstance(self.gamma_p, float):
                gamma_p = self.gamma_p
            else:
                gamma_p = self.gamma_p.abs() + 1.0

            if (
                not isinstance(self.gamma_m, float) or
                not isinstance(gamma_p, float) or
                self.gamma_m != gamma_p
            ):
                g = g * torch.where(x < 0.0, self.gamma_m, gamma_p)
            elif self.gamma_m != 1.0:
                g = g * self.gamma_m

        if self.gate_mode != "bilinear":
            g = g.sigmoid()

        return a * g

    def step(self) -> None:
        if not self.training:
            raise RuntimeError("cannot step during eval")

        if self.gamma_p is not None and not isinstance(self.gamma_p, float):
            with torch.no_grad():
                self.gamma_p.copy_(self.gamma_p.abs())

class ChannelNorm(torch.nn.Module):
    def __init__(self,
        shape: tuple[int | None, ...] | int,
        betas: tuple[float, float] | None = (0.99, 1.0),
        eps: float = 1e-5
    ):
        super().__init__()

        if not isinstance(shape, tuple):
            shape = (shape,)

        self.m2: torch.Tensor
        self.weight: torch.Tensor
        self.mean: torch.Tensor
        self.stddev: torch.Tensor

        self.betas = betas
        self.eps = eps

        self._shape  = tuple(dim if dim is not None else 1 for dim in shape)
        self._reduce = list(idx for idx in range(-len(shape), 0, -1) if shape[idx] is None)

        self._size = 1
        for dim in self._shape:
            self._size *= dim

        self.register_buffer("weight", torch.zeros(1))
        self.register_buffer("m2", torch.zeros(self._shape))
        self.register_buffer("mean", torch.zeros(self._shape))
        self.register_buffer("stddev", torch.ones(self._shape))

    def update(self, x: torch.Tensor, betas: tuple[float, float] = (1.0, 1.0)) -> None:
        beta0, beta1 = betas

        if beta0  < 0.0 or beta0 > 1.0:
            raise ValueError("invalid beta0")

        if beta1 <= 0.0:
            raise ValueError("invalid beta1")

        count = x.numel() // self._size
        reduce = self._reduce + list(range(0, x.dim() - len(self._shape)))

        with torch.no_grad():
            self.weight.copy_(self.weight * beta0 + count * beta1)

            delta = x - self.mean
            if len(reduce) > 0:
                delta = delta.sum(dim=reduce)

            delta = delta * beta1

            self.mean.copy_(self.mean * beta0 + delta / self.weight)

            delta2 = x - self.mean
            if len(reduce) > 0:
                delta2 = delta2.sum(dim=reduce)

            self.m2.copy_(self.m2 * beta0 + delta * delta2)

            sample_var = self.m2 / torch.clamp(self.weight - 1.0, 1.0)
            self.stddev.copy_(torch.sqrt(sample_var).clamp(self.eps))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training and self.betas is not None:
            self.update(x, betas=self.betas)

        return (x - self.mean) / self.stddev

    def forward_eval(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.stddev

class ClassifierNorm(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.ap_norm = ChannelNorm(BACKBONE_DIM, betas=(1.0, 0.01))
        self.ae_norm = ChannelNorm(AESTHETIC_DIM, betas=None)
        self.q_norm = ChannelNorm(QUALITY_DIM, betas=None)

    def update(
        self,
        ap: torch.Tensor,
        ae: torch.Tensor,
        q: torch.Tensor
    ) -> None:
        self.ap_norm.update(ap)
        self.ae_norm.update(ae)
        self.q_norm.update(q)

    def forward(
        self,
        ap: torch.Tensor,
        ae: torch.Tensor,
        q: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.ap_norm(ap), self.ae_norm(ae), self.q_norm(q)

class ClassifierHead(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.dropout = torch.nn.Dropout(0.2)

        self.linear_1 = torch.nn.Linear(BACKBONE_DIM + AESTHETIC_DIM + QUALITY_DIM, HIDDEN_DIM)
        init_linear(self.linear_1, "asig")

        self.activation_1 = AdaptiveSigmoid(HIDDEN_DIM)

        self.linear_2 = torch.nn.Linear(HIDDEN_DIM, N_HEADS * 2 + N_FREE)
        init_linear(self.linear_2, "sigmoid", bias=0.0)

        self.linear_3 = torch.nn.Linear(N_HEADS + N_FREE, DECISION_DIM)
        init_linear(self.linear_3, "asig")

        self.activation_2 = AdaptiveSigmoid(DECISION_DIM)

        self.linear_4 = torch.nn.Linear(DECISION_DIM, 1)
        init_linear(self.linear_4, "sigmoid", bias=0.0)

    def forward(
        self,
        ap: torch.Tensor,
        ae: torch.Tensor,
        q: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.dropout(torch.cat((ap, ae, q), dim=-1))
        x = self.dropout(self.activation_1(self.linear_1(x)))
        x = self.linear_2(x)

        x, f = torch.tensor_split(x, (N_HEADS * 2,), dim=-1)
        r, g = torch.tensor_split(x, 2, dim=-1)
        v = r * g.sigmoid()

        if self.training:
            m = torch.rand_like(v)

            pg = 0.1
            d = torch.where(m >= pg, v, r)

            pv = 0.1
            d = torch.where(m < (1.0 - pv), d, 0.0)

            if pv != pg:
                d = d * (1.0 + (pv - pg))
        else:
            d = v

        d = torch.cat((d, self.dropout(f)), dim=-1)
        d = self.dropout(self.activation_2(self.linear_3(d)))
        d = self.linear_4(d)

        return v, d, r, g

    def step(self) -> None:
        if not self.training:
            raise RuntimeError("cannot step during eval")

        self.activation_1.step()
        self.activation_2.step()

class ClassifierModel(torch.nn.Module):
    def __init__(self, *, for_training: bool = False):
        super().__init__()

        backbone = timm.create_model(
            "vit_so400m_patch14_siglip_384.webli",
            pretrained=True,
            num_classes=0
        ).eval()

        if not for_training:
            self.backbone = backbone

            self.quality = ARNIQAEncoder().eval()
            self.quality.load_state_dict(
                torch.hub.load_state_dict_from_url(
                    "https://github.com/miccunifi/ARNIQA/releases/download/weights/ARNIQA.pth",
                    map_location="cpu"
                )
            )

            self.aesthetic = AestheticPredictorV2p5Head().eval()
            self.aesthetic.load_state_dict(
                torch.hub.load_state_dict_from_url(
                    "https://github.com/discus0434/aesthetic-predictor-v2-5/raw/main/models/aesthetic_predictor_v2_5.pth",
                    map_location="cpu",
                )
            )

            self.attn_pool = timm.create_model(
                "vit_so400m_patch14_siglip_384",
                pretrained=False,
                num_classes=0
            ).attn_pool.eval()
        else:
            self.attn_pool = backbone.attn_pool

        self.norm = ClassifierNorm().eval()
        self.head = ClassifierHead().eval()

        self._attn_pool_norms: list[tuple[str, float]] = []

    def train(self, mode: bool = True) -> "ClassifierModel":
        self.training = mode

        self.attn_pool.train(mode)
        self.norm.train(mode)
        self.head.train(mode)

        return self

    def forward_backbone(self, xb: torch.Tensor) -> torch.Tensor:
        return self.backbone.forward_features((xb - 0.5) * 2.0)

    def forward_quality(self, xq: torch.Tensor) -> torch.Tensor:
        mean = torch.tensor(
            timm.data.IMAGENET_DEFAULT_MEAN,
            dtype=xq.dtype,
            device=xq.device
        )[:, None, None]
        std = torch.tensor(
            timm.data.IMAGENET_DEFAULT_STD,
            dtype=xq.dtype,
            device=xq.device
        )[:, None, None]
        return self.quality.forward((xq - mean) / std)

    def forward_features(self, xb: torch.Tensor, xq: torch.Tensor) -> dict[str, torch.Tensor]:
        f = self.forward_backbone(xb)
        ap = self.backbone.attn_pool(f)
        ae = self.aesthetic(ap)
        q = self.forward_quality(xq)

        return { "f": f, "ap": ap, "ae": ae, "q": q }

    def forward_head(
        self,
        f: torch.Tensor,
        ae: torch.Tensor,
        q: torch.Tensor
    ) -> torch.Tensor:
        ap = self.attn_pool(f)
        ap, ae, q = self.norm(ap, ae, q)
        return torch.cat(self.head(ap, ae, q), dim=-1)

    def forward(self, xb: torch.Tensor, xq: torch.Tensor) -> torch.Tensor:
        f = self.forward_features(xb, xq)
        n = self.norm(self.attn_pool(f["f"]), f["ae"], f["q"])
        return torch.cat(self.head(*n)[:2], dim=-1)

    @staticmethod
    def assess(outputs: torch.Tensor, threshold: float = 0.5) -> list[tuple[bool, str, float]]:
        outputs = outputs.cpu()

        p = outputs[:, -1].sigmoid()
        m = p > threshold
        c = torch.where(
            m,
            outputs[:, :N_ACCEPT].argmax(dim=-1),
            N_ACCEPT + outputs[:, N_ACCEPT:-1].argmax(dim=-1)
        )

        return [
            (m[i].item(), OUTPUT_NAMES[c[i].item()], p[i].item())
            for i in range(outputs.size(0))
        ]

    def setup_train(self) -> None:
        with torch.no_grad():
            self._attn_pool_norms = [
                (name, torch.linalg.norm(param).item())
                for name, param in self.attn_pool.named_parameters()
                if name.startswith(("q.", "kv.", "proj.", "mlp."))
            ]

        for name, norm in self._attn_pool_norms:
            print(f"attn_pool.{name}: norm={norm}")

    def step(self, norms_beta: float = 0.0):
        if not self.training:
            raise RuntimeError("cannot step during eval")

        if norms_beta != 0.0:
            with torch.no_grad():
                for name, norm in self._attn_pool_norms:
                    param = self.attn_pool.get_parameter(name)
                    param.mul_(torch.lerp(
                        torch.ones(1, device=param.device, dtype=param.dtype),
                        norm / torch.linalg.norm(param),
                        norms_beta
                    ))

        self.head.step()
