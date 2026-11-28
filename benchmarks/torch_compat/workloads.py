"""Real-model workloads, written once in plain PyTorch.

Every workload here is ordinary ``torch`` / Transformers / Diffusers code. The
same source runs in two interpreters -- one whose ``torch`` is an independent
binary PyTorch, one whose ``torch`` is Jittor's compatibility layer -- so the
ratio between them is the cost a user pays for ``import torch`` resolving to
Jittor. Nothing in this file may branch on the runtime: a workload that needs a
Jittor-specific spelling is a compatibility bug to fix, not to paper over here.

Weights are randomly initialised from the published model configurations. That
keeps the suite offline and reproducible, and it does not change the work: the
FLOPs, shapes and kernels of a step do not depend on the weight values.
Generation is forced to its full length (no EOS) for the same reason.

Each workload defines:

``setup()``
    Build the model and optimizer on ``self.device``, and a small pool of
    input batches (``self.pool``).
``run(*batch)``
    One iteration on one batch. Returns the tensors it produced; the runner
    forces and synchronises them before stopping the clock. ``step()`` feeds
    ``run`` the next batch of the pool, round robin: a training loop sees a
    new batch every step, and a captured step has to follow it.
``items``
    Work per step in ``unit`` (tokens, images, samples), for throughput.

Everything random is drawn from NumPy with a fixed seed -- weights
(``deterministic_init``) and inputs alike -- so both runtimes start from the
same numbers and their step-by-step losses can be compared, not just their
times. The frameworks' own generators differ, so anything drawn from them
(dropout masks) makes a workload ``stochastic``: its values are not compared.

``--compile MODE`` wraps what ``compile_what`` names in ``torch.compile``: the
whole ``run`` by default -- a training step's forward, backward and optimizer
update -- or just the model, where a user compiles the model and loops around
it (a sampler), or Transformers' own ``generate`` compilation. ``None`` means
the workload has no shape-stable compiled form (a growing KV cache), and its
compiled rows are skipped. Each compiled step starts with
``torch.compiler.cudagraph_mark_step_begin()``, as PyTorch documents for
CUDA-graph modes: without it a training step's outputs are overwritten by the
next replay.
"""

import math
import zlib

import numpy as np


def deterministic_init(torch, model, seed):
    """Overwrite every parameter from NumPy, the same way in both runtimes.

    The frameworks' initialisers draw from different generators, so the same
    model started from different weights and only the step times could be
    compared. Scales follow the usual defaults: normal with std
    ``1/sqrt(fan_in)`` for matrices and kernels, ones for a 1-D ``weight``
    (norm layers), zeros for a 1-D ``bias``.
    """
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            shape = tuple(parameter.shape)
            if len(shape) == 1:
                value = np.ones(shape, np.float32) if name.endswith("weight") \
                    else np.zeros(shape, np.float32)
            else:
                fan_in = int(np.prod(shape[1:]))
                # Seeded by name, not position: the two runtimes may list the
                # same parameters in a different order.
                rng = np.random.default_rng([seed, zlib.crc32(name.encode())])
                value = (rng.standard_normal(shape, dtype=np.float32)
                         / np.float32(math.sqrt(fan_in)))
            parameter.copy_(torch.tensor(value, device=parameter.device)
                            .to(parameter.dtype))


class Workload:
    name = ""
    family = ""
    mode = ""          # "infer" or "train"
    unit = ""
    default_dtype = "float32"
    requires = ()      # top-level distributions the workload imports
    description = ""
    #: (warmup, repeats) for size "full"; the first warmup step pays Jittor's
    #: JIT compilation and is reported separately.
    iterations = (3, 10)
    #: Distinct batches cycled through, one per step.
    pool_size = 4
    #: What ``--compile`` wraps: "run", "model", "generate", or None.
    compile_what = "run"
    #: Draws from the framework's own generator (dropout), so its values
    #: differ between runtimes and between runs.
    stochastic = False

    def __init__(self, torch, device, dtype, size, batch=None, seed=0):
        self.torch = torch
        self.device = device
        self.dtype = getattr(torch, dtype)
        self.dtype_name = dtype
        self.size = size
        self.batch_override = batch
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.items = 0
        self.pool = [()]
        self._cursor = 0
        self.compiled = None

    @property
    def tiny(self):
        return self.size == "tiny"

    def _batch(self, tiny, full):
        """The batch size: ``--batch`` if given, else the size's default."""
        if self.batch_override is not None:
            return int(self.batch_override)
        return tiny if self.tiny else full

    # -- inputs, from NumPy ---------------------------------------------
    def normal(self, shape, dtype=None):
        value = self.rng.standard_normal(shape, dtype=np.float32)
        return self.torch.tensor(value, device=self.device).to(dtype or self.dtype)

    def integers(self, high, shape):
        value = self.rng.integers(0, high, shape).astype(np.int64)
        return self.torch.tensor(value, device=self.device)

    def init(self, model):
        deterministic_init(self.torch, model, self.seed)
        return model

    # -- the step ----------------------------------------------------------
    def setup(self):
        raise NotImplementedError

    def run(self, *batch):
        raise NotImplementedError

    def step(self):
        batch = self.pool[self._cursor % len(self.pool)]
        self._cursor += 1
        if self.compiled is not None:
            mark = getattr(self.torch.compiler, "cudagraph_mark_step_begin", None)
            if mark is not None:
                mark()
        return self.run(*batch)

    def compile(self, mode):
        """Wrap ``compile_what`` in ``torch.compile(mode=mode)``."""
        torch = self.torch
        if self.compile_what == "run":
            self.run = self.compiled = torch.compile(self.run, mode=mode)
        elif self.compile_what == "model":
            self.model = self.compiled = torch.compile(self.model, mode=mode)
        else:
            raise ValueError(self.compile_what)

    def compiled_target(self):
        """The object whose replay statistics describe the compiled step."""
        return self.compiled

    def config(self):
        """Shape parameters worth recording next to the timing."""
        return {}

    def parameter_count(self):
        model = getattr(self, "model", None)
        if model is None:
            return None
        return int(sum(p.numel() for p in model.parameters()))


# --------------------------------------------------------------------------
# Qwen3 (decoder-only LLM)
# --------------------------------------------------------------------------

def qwen3_config(tiny):
    """Qwen3-0.6B, as published in ``Qwen/Qwen3-0.6B/config.json``."""
    from transformers import Qwen3Config

    if tiny:
        return Qwen3Config(
            vocab_size=1024, hidden_size=128, intermediate_size=256,
            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
            head_dim=32, max_position_embeddings=4096, tie_word_embeddings=True,
            attention_dropout=0.0,
        )
    return Qwen3Config(
        vocab_size=151936, hidden_size=1024, intermediate_size=3072,
        num_hidden_layers=28, num_attention_heads=16, num_key_value_heads=8,
        head_dim=128, max_position_embeddings=40960, rope_theta=1000000.0,
        rms_norm_eps=1e-6, tie_word_embeddings=True, attention_dropout=0.0,
    )


class _Qwen3(Workload):
    family = "qwen3"
    requires = ("transformers",)

    def _model(self):
        from transformers import Qwen3ForCausalLM

        config = qwen3_config(self.tiny)
        config._attn_implementation = "sdpa"
        model = self.init(Qwen3ForCausalLM(config))
        return model.to(device=self.device, dtype=self.dtype), config


class Qwen3Prefill(_Qwen3):
    name = "qwen3_prefill"
    mode = "infer"
    unit = "tokens"
    default_dtype = "bfloat16"
    description = "Qwen3-0.6B prompt prefill, logits for the last position only"

    def setup(self):
        self.model, config = self._model()
        self.model.eval()
        self.batch = self._batch(1, 1)
        self.length = 32 if self.tiny else 2048
        self.pool = [(self.integers(config.vocab_size, (self.batch, self.length)),)
                     for _ in range(self.pool_size)]
        self.items = self.batch * self.length

    def run(self, input_ids):
        with self.torch.no_grad():
            out = self.model(input_ids=input_ids, use_cache=False, logits_to_keep=1)
        return [out.logits]

    def config(self):
        return {"batch": self.batch, "prompt": self.length}


class Qwen3Decode(_Qwen3):
    name = "qwen3_decode"
    mode = "infer"
    unit = "tokens"
    default_dtype = "bfloat16"
    description = "Qwen3-0.6B greedy generate() with KV cache, fixed length"
    iterations = (2, 5)
    compile_what = None     # the dynamic cache grows every token
    cache = None

    def setup(self):
        self.model, config = self._model()
        self.model.eval()
        # Never stop early: random weights may emit EOS at any position.
        self.model.generation_config.eos_token_id = None
        self.model.generation_config.pad_token_id = 0
        # Transformers compiles a static-cache decode by itself; eager rows
        # must stay eager.
        self.model.generation_config.disable_compile = True
        self.batch = self._batch(1, 1)
        self.prompt = 8 if self.tiny else 128
        self.new_tokens = 8 if self.tiny else 128
        self.pool = []
        for _ in range(self.pool_size):
            ids = self.integers(config.vocab_size, (self.batch, self.prompt))
            self.pool.append((ids, self.torch.ones_like(ids)))
        self.items = self.batch * self.new_tokens

    def run(self, input_ids, attention_mask):
        extra = {} if self.cache is None else {"cache_implementation": self.cache}
        with self.torch.no_grad():
            out = self.model.generate(
                input_ids=input_ids, attention_mask=attention_mask,
                max_new_tokens=self.new_tokens, min_new_tokens=self.new_tokens,
                do_sample=False, use_cache=True, **extra)
        if out.shape[-1] != self.prompt + self.new_tokens:
            raise RuntimeError("generate() stopped at %d tokens" % out.shape[-1])
        return [out]

    def config(self):
        return {"batch": self.batch, "prompt": self.prompt,
                "new_tokens": self.new_tokens, "cache": self.cache or "dynamic"}


class Qwen3DecodeStatic(Qwen3Decode):
    name = "qwen3_decode_static"
    description = ("Qwen3-0.6B greedy generate() with a static KV cache -- the "
                   "shape-stable decode that reduce-overhead is for")
    cache = "static"
    compile_what = "generate"

    def compile(self, mode):
        # Transformers' own route: generate() compiles the decode step (and
        # runs the prefill as written) when the cache is static.
        from transformers import CompileConfig

        config = self.model.generation_config
        config.disable_compile = False
        config.compile_config = CompileConfig(fullgraph=False, dynamic=False, mode=mode)
        self.compiled = "generate"

    def compiled_target(self):
        return getattr(self.model, "_compiled_call", None)


class Qwen3Train(_Qwen3):
    name = "qwen3_train"
    mode = "train"
    unit = "tokens"
    default_dtype = "float32"
    description = "Qwen3-0.6B causal-LM step: forward, loss, backward, AdamW"

    def setup(self):
        torch = self.torch
        self.model, config = self._model()
        self.model.train()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-5)
        self.batch = self._batch(2, 4)
        self.length = 32 if self.tiny else 512
        self.pool = [(self.integers(config.vocab_size, (self.batch, self.length)),)
                     for _ in range(self.pool_size)]
        self.items = self.batch * self.length

    def run(self, input_ids):
        loss = self.model(input_ids=input_ids, labels=input_ids).loss
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        return [loss]

    def config(self):
        return {"batch": self.batch, "seq": self.length, "optimizer": "AdamW"}


class Qwen3TrainBF16(Qwen3Train):
    name = "qwen3_train_bf16"
    default_dtype = "bfloat16"
    description = "Qwen3-0.6B bfloat16 causal-LM training step: forward, backward, AdamW"


class Qwen3PrefillLong(Qwen3Prefill):
    name = "qwen3_prefill_long"
    description = "Qwen3-0.6B long-context 4096-token prefill (bfloat16)"

    def setup(self):
        self.model, config = self._model()
        self.model.eval()
        self.batch = self._batch(1, 1)
        self.length = 64 if self.tiny else 4096
        self.pool = [(self.integers(config.vocab_size, (self.batch, self.length)),)
                     for _ in range(self.pool_size)]
        self.items = self.batch * self.length


# --------------------------------------------------------------------------
# Stable Diffusion 1.5 (latent diffusion)
# --------------------------------------------------------------------------

def sd15_unet(tiny):
    """The UNet of ``runwayml/stable-diffusion-v1-5`` (860M parameters)."""
    from diffusers import UNet2DConditionModel

    if tiny:
        return UNet2DConditionModel(
            sample_size=16, in_channels=4, out_channels=4, layers_per_block=1,
            block_out_channels=(32, 64),
            down_block_types=("CrossAttnDownBlock2D", "DownBlock2D"),
            up_block_types=("UpBlock2D", "CrossAttnUpBlock2D"),
            cross_attention_dim=32, attention_head_dim=4, norm_num_groups=8,
        )
    return UNet2DConditionModel(
        sample_size=64, in_channels=4, out_channels=4, layers_per_block=2,
        block_out_channels=(320, 640, 1280, 1280),
        down_block_types=("CrossAttnDownBlock2D", "CrossAttnDownBlock2D",
                          "CrossAttnDownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "CrossAttnUpBlock2D",
                        "CrossAttnUpBlock2D", "CrossAttnUpBlock2D"),
        cross_attention_dim=768, attention_head_dim=8, norm_num_groups=32,
    )


def sd15_vae(tiny):
    """The VAE of ``runwayml/stable-diffusion-v1-5``."""
    from diffusers import AutoencoderKL

    if tiny:
        return AutoencoderKL(
            in_channels=3, out_channels=3, latent_channels=4,
            block_out_channels=(32, 64),
            down_block_types=("DownEncoderBlock2D",) * 2,
            up_block_types=("UpDecoderBlock2D",) * 2,
            layers_per_block=1, norm_num_groups=8, sample_size=32,
        )
    return AutoencoderKL(
        in_channels=3, out_channels=3, latent_channels=4,
        block_out_channels=(128, 256, 512, 512),
        down_block_types=("DownEncoderBlock2D",) * 4,
        up_block_types=("UpDecoderBlock2D",) * 4,
        layers_per_block=2, norm_num_groups=32, sample_size=512,
    )


class _SD15(Workload):
    family = "stable-diffusion-1.5"
    requires = ("diffusers",)

    def _latent_shape(self, batch):
        side = 8 if self.tiny else 64
        return (batch, 4, side, side)

    def _text(self, batch):
        width = 32 if self.tiny else 768
        return self.normal((batch, 77, width))


class SD15Sample(_SD15):
    name = "sd15_sample"
    mode = "infer"
    unit = "images"
    default_dtype = "float16"
    description = ("SD1.5 text-to-image denoising: DDIM, classifier-free "
                   "guidance 7.5, 512x512")
    iterations = (2, 5)
    compile_what = "model"

    def setup(self):
        from diffusers import DDIMScheduler

        self.model = self.init(sd15_unet(self.tiny)).to(
            device=self.device, dtype=self.dtype).eval()
        self.scheduler = DDIMScheduler(
            beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear",
            clip_sample=False, set_alpha_to_one=False, steps_offset=1)
        self.steps = 4 if self.tiny else 20
        self.scheduler.set_timesteps(self.steps)
        self.batch = self._batch(1, 1)
        self.pool = [(self.normal(self._latent_shape(self.batch)),
                      self._text(2 * self.batch))  # [uncond, cond]
                     for _ in range(self.pool_size)]
        self.items = self.batch

    def run(self, initial, context):
        torch = self.torch
        latents = initial * self.scheduler.init_noise_sigma
        with torch.no_grad():
            for t in self.scheduler.timesteps:
                model_in = torch.cat([latents, latents])
                model_in = self.scheduler.scale_model_input(model_in, t)
                noise = self.model(model_in, t,
                                   encoder_hidden_states=context).sample
                uncond, cond = noise.chunk(2)
                noise = uncond + 7.5 * (cond - uncond)
                latents = self.scheduler.step(noise, t, latents).prev_sample
        return [latents]

    def config(self):
        return {"batch": self.batch, "steps": self.steps,
                "latent": list(self._latent_shape(self.batch)), "cfg": 7.5}


class SD15VAEDecode(_SD15):
    name = "sd15_vae_decode"
    mode = "infer"
    unit = "images"
    default_dtype = "float16"
    description = "SD1.5 VAE decode, 64x64 latent to a 512x512 image"
    iterations = (2, 10)

    def setup(self):
        self.model = self.init(sd15_vae(self.tiny)).to(
            device=self.device, dtype=self.dtype).eval()
        self.batch = self._batch(1, 1)
        self.pool = [(self.normal(self._latent_shape(self.batch)),)
                     for _ in range(self.pool_size)]
        self.items = self.batch

    def run(self, latents):
        with self.torch.no_grad():
            image = self.model.decode(latents / 0.18215).sample
        return [image]

    def config(self):
        return {"batch": self.batch, "latent": list(self._latent_shape(1))}


class SD15Train(_SD15):
    name = "sd15_unet_train"
    mode = "train"
    unit = "images"
    default_dtype = "float32"
    description = ("SD1.5 UNet fine-tuning step: DDPM noise, epsilon MSE, "
                   "backward, AdamW")
    iterations = (2, 5)

    def setup(self):
        from diffusers import DDPMScheduler

        torch = self.torch
        self.model = self.init(sd15_unet(self.tiny)).to(
            device=self.device, dtype=self.dtype).train()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-5)
        self.scheduler = DDPMScheduler(
            beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear",
            num_train_timesteps=1000)
        self.batch = self._batch(2, 2)
        shape = self._latent_shape(self.batch)
        # The noise and timesteps a training loop draws each step, drawn here
        # from NumPy per batch so both runtimes see the same ones.
        self.pool = [(self.normal(shape), self.normal(shape),
                      self.integers(1000, (self.batch,)), self._text(self.batch))
                     for _ in range(self.pool_size)]
        self.items = self.batch

    def run(self, latents, noise, timesteps, context):
        torch = self.torch
        noisy = self.scheduler.add_noise(latents, noise, timesteps)
        pred = self.model(noisy, timesteps, encoder_hidden_states=context).sample
        loss = torch.nn.functional.mse_loss(pred.float(), noise.float())
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        return [loss]

    def config(self):
        return {"batch": self.batch, "latent": list(self._latent_shape(1)),
                "optimizer": "AdamW"}


# --------------------------------------------------------------------------
# DDPM (pixel-space diffusion, the diffusers train_unconditional example)
# --------------------------------------------------------------------------

def ddpm_unet(tiny):
    from diffusers import UNet2DModel

    if tiny:
        return UNet2DModel(
            sample_size=16, in_channels=3, out_channels=3, layers_per_block=1,
            block_out_channels=(32, 64),
            down_block_types=("DownBlock2D", "AttnDownBlock2D"),
            up_block_types=("AttnUpBlock2D", "UpBlock2D"),
            norm_num_groups=8,
        )
    # examples/unconditional_image_generation/train_unconditional.py
    return UNet2DModel(
        sample_size=64, in_channels=3, out_channels=3, layers_per_block=2,
        block_out_channels=(128, 128, 256, 256, 512, 512),
        down_block_types=("DownBlock2D", "DownBlock2D", "DownBlock2D",
                          "DownBlock2D", "AttnDownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "AttnUpBlock2D", "UpBlock2D",
                        "UpBlock2D", "UpBlock2D", "UpBlock2D"),
    )


class DDPMTrain(Workload):
    name = "ddpm_unet_train"
    family = "ddpm"
    mode = "train"
    unit = "images"
    requires = ("diffusers",)
    description = ("Unconditional DDPM UNet (diffusers train_unconditional), "
                   "64x64, epsilon MSE, AdamW")

    def setup(self):
        from diffusers import DDPMScheduler

        torch = self.torch
        self.model = self.init(ddpm_unet(self.tiny)).to(
            device=self.device, dtype=self.dtype).train()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-4)
        self.scheduler = DDPMScheduler(num_train_timesteps=1000)
        self.batch = self._batch(2, 16)
        side = 16 if self.tiny else 64
        shape = (self.batch, 3, side, side)
        self.pool = [(self.normal(shape), self.normal(shape),
                      self.integers(1000, (self.batch,)))
                     for _ in range(self.pool_size)]
        self.items = self.batch

    def run(self, images, noise, timesteps):
        torch = self.torch
        noisy = self.scheduler.add_noise(images, noise, timesteps)
        pred = self.model(noisy, timesteps).sample
        loss = torch.nn.functional.mse_loss(pred, noise)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        return [loss]

    def config(self):
        return {"batch": self.batch, "optimizer": "AdamW"}


# --------------------------------------------------------------------------
# Vision and encoder classics
# --------------------------------------------------------------------------

def resnet50(torch, num_classes=1000, tiny=False):
    """torchvision's ResNet-50 (v1.5), spelled out so no torchvision is needed."""
    nn = torch.nn

    class Bottleneck(nn.Module):
        expansion = 4

        def __init__(self, inplanes, planes, stride=1, downsample=None):
            super().__init__()
            self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
            self.bn1 = nn.BatchNorm2d(planes)
            self.conv2 = nn.Conv2d(planes, planes, 3, stride=stride,
                                   padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(planes)
            self.conv3 = nn.Conv2d(planes, planes * 4, 1, bias=False)
            self.bn3 = nn.BatchNorm2d(planes * 4)
            self.relu = nn.ReLU(inplace=True)
            self.downsample = downsample

        def forward(self, x):
            identity = x
            out = self.relu(self.bn1(self.conv1(x)))
            out = self.relu(self.bn2(self.conv2(out)))
            out = self.bn3(self.conv3(out))
            if self.downsample is not None:
                identity = self.downsample(x)
            return self.relu(out + identity)

    class ResNet(nn.Module):
        def __init__(self, layers):
            super().__init__()
            self.inplanes = 64
            self.conv1 = nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False)
            self.bn1 = nn.BatchNorm2d(64)
            self.relu = nn.ReLU(inplace=True)
            self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
            self.layer1 = self._make_layer(64, layers[0])
            self.layer2 = self._make_layer(128, layers[1], stride=2)
            self.layer3 = self._make_layer(256, layers[2], stride=2)
            self.layer4 = self._make_layer(512, layers[3], stride=2)
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(512 * 4, num_classes)
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                            nonlinearity="relu")

        def _make_layer(self, planes, blocks, stride=1):
            downsample = None
            if stride != 1 or self.inplanes != planes * 4:
                downsample = nn.Sequential(
                    nn.Conv2d(self.inplanes, planes * 4, 1, stride=stride,
                              bias=False),
                    nn.BatchNorm2d(planes * 4))
            layers = [Bottleneck(self.inplanes, planes, stride, downsample)]
            self.inplanes = planes * 4
            layers += [Bottleneck(self.inplanes, planes)
                       for _ in range(1, blocks)]
            return nn.Sequential(*layers)

        def forward(self, x):
            x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
            x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
            return self.fc(torch.flatten(self.avgpool(x), 1))

    return ResNet((1, 1, 1, 1) if tiny else (3, 4, 6, 3))


class _Images(Workload):
    unit = "images"
    full_batch = 64

    def _images(self):
        self.batch = self._batch(4 if self.full_batch > 1 else 1, self.full_batch)
        side = 32 if self.tiny else 224
        self.pool = [(self.normal((self.batch, 3, side, side)),
                      self.integers(1000, (self.batch,)))
                     for _ in range(self.pool_size)]
        self.items = self.batch

    def config(self):
        return {"batch": self.batch, "image": 32 if self.tiny else 224}


class ResNet50Infer(_Images):
    name = "resnet50_infer"
    family = "resnet50"
    mode = "infer"
    default_dtype = "float16"
    description = "ResNet-50 inference, batch 64, 224x224"

    def setup(self):
        self.model = self.init(resnet50(self.torch, tiny=self.tiny)).to(
            device=self.device, dtype=self.dtype).eval()
        self._images()

    def run(self, images, labels):
        with self.torch.no_grad():
            return [self.model(images)]


class ResNet50InferB1(ResNet50Infer):
    name = "resnet50_infer_b1"
    description = ("ResNet-50 inference, batch 1, 224x224 -- the latency-bound "
                   "regime, where per-op host cost is most of the step")
    full_batch = 1
    iterations = (5, 30)


class ResNet50Train(_Images):
    name = "resnet50_train"
    family = "resnet50"
    mode = "train"
    description = ("ResNet-50 ImageNet step, batch 64, 224x224, "
                   "SGD momentum 0.9")

    def setup(self):
        torch = self.torch
        self.model = self.init(resnet50(torch, tiny=self.tiny)).to(
            device=self.device, dtype=self.dtype).train()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.1,
                                         momentum=0.9, weight_decay=1e-4)
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self._images()

    def run(self, images, labels):
        loss = self.loss_fn(self.model(images), labels)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        return [loss]


class ViTB16Train(_Images):
    name = "vit_b16_train"
    family = "vit-b/16"
    mode = "train"
    requires = ("transformers",)
    description = "ViT-B/16 image classification step, batch 64, AdamW"

    def setup(self):
        from transformers import ViTConfig, ViTForImageClassification

        torch = self.torch
        if self.tiny:
            config = ViTConfig(hidden_size=64, num_hidden_layers=2,
                               num_attention_heads=4, intermediate_size=128,
                               image_size=32, patch_size=8, num_labels=1000)
        else:
            config = ViTConfig(image_size=224, patch_size=16, num_labels=1000)
        config._attn_implementation = "sdpa"
        self.model = self.init(ViTForImageClassification(config)).to(
            device=self.device, dtype=self.dtype).train()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-4)
        self._images()

    def run(self, images, labels):
        loss = self.model(pixel_values=images, labels=labels).loss
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        return [loss]


class _Bert(Workload):
    family = "bert-base"
    unit = "samples"
    requires = ("transformers",)

    def _setup(self, train, tiny_batch, full_batch):
        from transformers import BertConfig, BertForSequenceClassification

        if self.tiny:
            config = BertConfig(vocab_size=1024, hidden_size=64,
                                num_hidden_layers=2, num_attention_heads=4,
                                intermediate_size=128, num_labels=2)
        else:
            config = BertConfig(num_labels=2)
        config._attn_implementation = "sdpa"
        self.model = self.init(BertForSequenceClassification(config)).to(
            device=self.device, dtype=self.dtype)
        self.model.train(train)
        self.batch = self._batch(tiny_batch, full_batch)
        self.length = 16 if self.tiny else 128
        self.pool = []
        for _ in range(self.pool_size):
            ids = self.integers(config.vocab_size, (self.batch, self.length))
            self.pool.append((ids, self.torch.ones_like(ids),
                              self.integers(2, (self.batch,))))
        self.items = self.batch

    def config(self):
        return {"batch": self.batch, "seq": self.length}


class BertBaseTrain(_Bert):
    name = "bert_base_train"
    mode = "train"
    description = ("BERT-base sequence classification fine-tuning, "
                   "batch 32, seq 128, AdamW, dropout 0.1")
    stochastic = True

    def setup(self):
        self._setup(True, 4, 32)
        self.optimizer = self.torch.optim.AdamW(self.model.parameters(), lr=2e-5)

    def run(self, input_ids, attention_mask, labels):
        loss = self.model(input_ids=input_ids, attention_mask=attention_mask,
                          labels=labels).loss
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        return [loss]


class BertBaseInfer(_Bert):
    name = "bert_base_infer"
    mode = "infer"
    description = ("BERT-base sequence classification, batch 1, seq 128 -- "
                   "online inference, bound by per-op host cost")
    iterations = (5, 30)

    def setup(self):
        self._setup(False, 1, 1)

    def run(self, input_ids, attention_mask, labels):
        with self.torch.no_grad():
            return [self.model(input_ids=input_ids,
                               attention_mask=attention_mask).logits]


# --------------------------------------------------------------------------
# Torch-compat core operators (pure PyTorch, zero external package deps)
# --------------------------------------------------------------------------

class OpMatmulBF16(Workload):
    name = "op_matmul_bf16"
    family = "ops"
    mode = "infer"
    unit = "samples"
    default_dtype = "bfloat16"
    description = "Batched GEMM torch.matmul, [16, 1024, 1024] bf16"
    iterations = (5, 30)

    def setup(self):
        self.batch = self._batch(4, 16)
        self.dim = 64 if self.tiny else 1024
        self.pool = [(self.normal((self.batch, self.dim, self.dim)),
                      self.normal((self.batch, self.dim, self.dim)))
                     for _ in range(self.pool_size)]
        self.items = self.batch

    def run(self, a, b):
        with self.torch.no_grad():
            return [self.torch.matmul(a, b)]

    def config(self):
        return {"batch": self.batch, "m": self.dim, "n": self.dim, "k": self.dim}


class OpConv2dFP16(Workload):
    name = "op_conv2d_fp16"
    family = "ops"
    mode = "train"
    unit = "images"
    default_dtype = "float16"
    description = "Conv2d 3x3 forward+backward, [64, 256, 56, 56] -> 256 fp16"
    iterations = (5, 20)

    def setup(self):
        nn = self.torch.nn
        self.batch = self._batch(4, 64)
        self.channels = 16 if self.tiny else 256
        self.spatial = 16 if self.tiny else 56
        self.model = self.init(
            nn.Conv2d(self.channels, self.channels, kernel_size=3, padding=1, bias=False)
        ).to(device=self.device, dtype=self.dtype).train()
        self.pool = [(self.normal((self.batch, self.channels, self.spatial, self.spatial)),)
                     for _ in range(self.pool_size)]
        self.items = self.batch

    def run(self, x):
        y = self.model(x)
        loss = y.float().pow(2).mean()
        loss.backward()
        self.model.zero_grad(set_to_none=True)
        return [loss]

    def config(self):
        return {"batch": self.batch, "channels": self.channels, "hw": self.spatial}


class OpRMSNormBF16(Workload):
    name = "op_rmsnorm_bf16"
    family = "ops"
    mode = "infer"
    unit = "tokens"
    default_dtype = "bfloat16"
    description = "Residual add + F.rms_norm, [4, 2048, 2048] bf16"
    iterations = (5, 30)

    def setup(self):
        self.batch = self._batch(2, 4)
        self.seq = 64 if self.tiny else 2048
        self.dim = 128 if self.tiny else 2048
        self.weight = self.torch.ones((self.dim,), device=self.device, dtype=self.dtype)
        self.pool = [(self.normal((self.batch, self.seq, self.dim)),
                      self.normal((self.batch, self.seq, self.dim)))
                     for _ in range(self.pool_size)]
        self.items = self.batch * self.seq

    def run(self, x, residual):
        F = self.torch.nn.functional
        with self.torch.no_grad():
            h = x + residual
            return [F.rms_norm(h, (self.dim,), self.weight, eps=1e-6)]

    def config(self):
        return {"batch": self.batch, "seq": self.seq, "hidden": self.dim}


class OpLayerNormFP32(Workload):
    name = "op_layernorm_fp32"
    family = "ops"
    mode = "infer"
    unit = "tokens"
    default_dtype = "float32"
    description = "F.layer_norm, [16, 512, 1024] fp32"
    iterations = (5, 30)

    def setup(self):
        self.batch = self._batch(2, 16)
        self.seq = 32 if self.tiny else 512
        self.dim = 128 if self.tiny else 1024
        self.weight = self.torch.ones((self.dim,), device=self.device, dtype=self.dtype)
        self.bias = self.torch.zeros((self.dim,), device=self.device, dtype=self.dtype)
        self.pool = [(self.normal((self.batch, self.seq, self.dim)),)
                     for _ in range(self.pool_size)]
        self.items = self.batch * self.seq

    def run(self, x):
        F = self.torch.nn.functional
        with self.torch.no_grad():
            return [F.layer_norm(x, (self.dim,), self.weight, self.bias, eps=1e-5)]

    def config(self):
        return {"batch": self.batch, "seq": self.seq, "hidden": self.dim}


class OpSoftmaxBF16(Workload):
    name = "op_softmax_bf16"
    family = "ops"
    mode = "infer"
    unit = "rows"
    default_dtype = "bfloat16"
    description = "Large-axis F.softmax(dim=-1), [16, 512, 4096] bf16"
    iterations = (5, 30)

    def setup(self):
        self.batch = self._batch(2, 16)
        self.rows = 32 if self.tiny else 512
        self.cols = 256 if self.tiny else 4096
        self.pool = [(self.normal((self.batch, self.rows, self.cols)),)
                     for _ in range(self.pool_size)]
        self.items = self.batch * self.rows

    def run(self, x):
        with self.torch.no_grad():
            return [self.torch.nn.functional.softmax(x, dim=-1)]

    def config(self):
        return {"batch": self.batch, "rows": self.rows, "cols": self.cols}


class OpSDPACausalBF16(Workload):
    name = "op_sdpa_causal_bf16"
    family = "ops"
    mode = "infer"
    unit = "tokens"
    default_dtype = "bfloat16"
    description = "Causal F.scaled_dot_product_attention, [4, 16, 1024, 128] bf16"
    iterations = (5, 30)

    def setup(self):
        self.batch = self._batch(2, 4)
        self.heads = 4 if self.tiny else 16
        self.seq = 64 if self.tiny else 1024
        self.head_dim = 32 if self.tiny else 128
        shape = (self.batch, self.heads, self.seq, self.head_dim)
        self.pool = [(self.normal(shape), self.normal(shape), self.normal(shape))
                     for _ in range(self.pool_size)]
        self.items = self.batch * self.seq

    def run(self, q, k, v):
        F = self.torch.nn.functional
        with self.torch.no_grad():
            return [F.scaled_dot_product_attention(q, k, v, is_causal=True)]

    def config(self):
        return {"batch": self.batch, "heads": self.heads,
                "seq": self.seq, "head_dim": self.head_dim}


class OpSwiGLUMlpBF16(Workload):
    name = "op_swiglu_mlp_bf16"
    family = "ops"
    mode = "infer"
    unit = "tokens"
    default_dtype = "bfloat16"
    description = "SwiGLU FFN down(silu(gate(x)) * up(x)), [4, 1024, 2048->5504] bf16"
    iterations = (5, 30)

    def setup(self):
        nn = self.torch.nn
        self.batch = self._batch(2, 4)
        self.seq = 32 if self.tiny else 1024
        self.hidden = 128 if self.tiny else 2048
        self.inter = 256 if self.tiny else 5504

        class _SwiGLU(nn.Module):
            def __init__(m, hidden, inter):
                super().__init__()
                m.gate = nn.Linear(hidden, inter, bias=False)
                m.up = nn.Linear(hidden, inter, bias=False)
                m.down = nn.Linear(inter, hidden, bias=False)

            def forward(m, x):
                return m.down(nn.functional.silu(m.gate(x)) * m.up(x))

        self.model = self.init(_SwiGLU(self.hidden, self.inter)).to(
            device=self.device, dtype=self.dtype).eval()
        self.pool = [(self.normal((self.batch, self.seq, self.hidden)),)
                     for _ in range(self.pool_size)]
        self.items = self.batch * self.seq

    def run(self, x):
        with self.torch.no_grad():
            return [self.model(x)]

    def config(self):
        return {"batch": self.batch, "seq": self.seq,
                "hidden": self.hidden, "intermediate": self.inter}


class OpRoPEBF16(Workload):
    name = "op_rope_bf16"
    family = "ops"
    mode = "infer"
    unit = "tokens"
    default_dtype = "bfloat16"
    description = "Rotary position embedding (cos/sin rotation), [4, 16, 2048, 128] bf16"
    iterations = (5, 30)

    def setup(self):
        self.batch = self._batch(2, 4)
        self.heads = 4 if self.tiny else 16
        self.seq = 64 if self.tiny else 2048
        self.head_dim = 32 if self.tiny else 128
        half = self.head_dim // 2
        inv_freq = 1.0 / (10000.0 ** (np.arange(0, half, dtype=np.float32) / half))
        pos = np.arange(self.seq, dtype=np.float32)[:, None] * inv_freq[None, :]
        emb = np.concatenate([pos, pos], axis=-1)[None, None, :, :]
        self.cos = self.torch.tensor(np.cos(emb), device=self.device).to(self.dtype)
        self.sin = self.torch.tensor(np.sin(emb), device=self.device).to(self.dtype)
        shape = (self.batch, self.heads, self.seq, self.head_dim)
        self.pool = [(self.normal(shape), self.normal(shape))
                     for _ in range(self.pool_size)]
        self.items = self.batch * self.seq

    def run(self, q, k):
        torch = self.torch
        half = self.head_dim // 2
        with torch.no_grad():
            q_rot = torch.cat((-q[..., half:], q[..., :half]), dim=-1)
            k_rot = torch.cat((-k[..., half:], k[..., :half]), dim=-1)
            return [q * self.cos + q_rot * self.sin,
                    k * self.cos + k_rot * self.sin]

    def config(self):
        return {"batch": self.batch, "heads": self.heads,
                "seq": self.seq, "head_dim": self.head_dim}


class OpCrossEntropyVocab(Workload):
    name = "op_cross_entropy_vocab"
    family = "ops"
    mode = "train"
    unit = "tokens"
    default_dtype = "float32"
    description = "Large-vocab F.cross_entropy forward+backward, [2048, 32000] fp32"
    iterations = (5, 20)

    def setup(self):
        self.tokens = 64 if self.tiny else 2048
        self.vocab = 1024 if self.tiny else 32000
        self.pool = []
        for _ in range(self.pool_size):
            logits = self.normal((self.tokens, self.vocab))
            targets = self.integers(self.vocab, (self.tokens,))
            self.pool.append((logits, targets))
        self.items = self.tokens

    def run(self, logits, targets):
        x = logits.detach().clone().requires_grad_(True)
        loss = self.torch.nn.functional.cross_entropy(x, targets)
        loss.backward()
        return [loss, x.grad]

    def config(self):
        return {"tokens": self.tokens, "vocab": self.vocab}


class OpAdamWStep(Workload):
    name = "op_adamw_step"
    family = "ops"
    mode = "train"
    unit = "tensors"
    default_dtype = "float32"
    description = "AdamW optimizer step over 256 parameter tensors [512, 512]"
    iterations = (5, 25)
    compile_what = None

    def setup(self):
        nn = self.torch.nn
        self.num_tensors = 16 if self.tiny else 256
        self.dim = 32 if self.tiny else 512
        self.model = self.init(
            nn.ModuleList([nn.Linear(self.dim, self.dim, bias=False)
                           for _ in range(self.num_tensors)])
        ).to(device=self.device, dtype=self.dtype)
        self.optimizer = self.torch.optim.AdamW(self.model.parameters(), lr=1e-4)
        # Pre-populate deterministic gradient pools (cycled per step).
        self.pool = [
            tuple(self.normal((self.dim, self.dim)) for _ in range(self.num_tensors))
            for _ in range(2)
        ]
        self.items = self.num_tensors

    def run(self, *grads):
        for p, g in zip(self.model.parameters(), grads):
            p.grad = g
        self.optimizer.step()
        return [next(self.model.parameters())]

    def config(self):
        return {"tensors": self.num_tensors, "shape": [self.dim, self.dim]}


class OpLinearBF16(Workload):
    name = "op_linear_bf16"
    family = "ops"
    mode = "train"
    unit = "tokens"
    default_dtype = "bfloat16"
    description = "nn.Linear projection forward+backward, [4, 1024, 2048->2048] bf16"
    iterations = (5, 25)

    def setup(self):
        nn = self.torch.nn
        self.batch = self._batch(2, 4)
        self.seq = 32 if self.tiny else 1024
        self.dim = 128 if self.tiny else 2048
        self.model = self.init(
            nn.Linear(self.dim, self.dim, bias=False)
        ).to(device=self.device, dtype=self.dtype).train()
        self.pool = [(self.normal((self.batch, self.seq, self.dim)),)
                     for _ in range(self.pool_size)]
        self.items = self.batch * self.seq

    def run(self, x):
        loss = self.model(x).float().pow(2).mean()
        loss.backward()
        self.model.zero_grad(set_to_none=True)
        return [loss]

    def config(self):
        return {"batch": self.batch, "seq": self.seq, "dim": self.dim}


class OpConvTranspose2dFP16(Workload):
    name = "op_conv_transpose2d_fp16"
    family = "ops"
    mode = "infer"
    unit = "images"
    default_dtype = "float16"
    description = "ConvTranspose2d 4x4 stride 2 upsampling, [16, 128, 64, 64] fp16"
    iterations = (5, 25)

    def setup(self):
        nn = self.torch.nn
        self.batch = self._batch(2, 16)
        self.channels = 16 if self.tiny else 128
        self.spatial = 16 if self.tiny else 64
        self.model = self.init(
            nn.ConvTranspose2d(self.channels, self.channels, kernel_size=4,
                               stride=2, padding=1, bias=False)
        ).to(device=self.device, dtype=self.dtype).eval()
        self.pool = [(self.normal((self.batch, self.channels, self.spatial, self.spatial)),)
                     for _ in range(self.pool_size)]
        self.items = self.batch

    def run(self, x):
        with self.torch.no_grad():
            return [self.model(x)]

    def config(self):
        return {"batch": self.batch, "channels": self.channels, "hw": self.spatial}


class OpGroupNormFP16(Workload):
    name = "op_groupnorm_fp16"
    family = "ops"
    mode = "infer"
    unit = "images"
    default_dtype = "float16"
    description = "F.group_norm (32 groups), [16, 256, 64, 64] fp16 (UNet/SD hotspot)"
    iterations = (5, 30)

    def setup(self):
        self.batch = self._batch(2, 16)
        self.channels = 32 if self.tiny else 256
        self.groups = 8 if self.tiny else 32
        self.spatial = 16 if self.tiny else 64
        self.weight = self.torch.ones((self.channels,), device=self.device, dtype=self.dtype)
        self.bias = self.torch.zeros((self.channels,), device=self.device, dtype=self.dtype)
        self.pool = [(self.normal((self.batch, self.channels, self.spatial, self.spatial)),)
                     for _ in range(self.pool_size)]
        self.items = self.batch

    def run(self, x):
        with self.torch.no_grad():
            return [self.torch.nn.functional.group_norm(
                x, self.groups, self.weight, self.bias, eps=1e-5)]

    def config(self):
        return {"batch": self.batch, "channels": self.channels,
                "groups": self.groups, "hw": self.spatial}


class OpGeluBF16(Workload):
    name = "op_gelu_bf16"
    family = "ops"
    mode = "infer"
    unit = "tokens"
    default_dtype = "bfloat16"
    description = "F.gelu(approximate='tanh'), [16, 512, 4096] bf16"
    iterations = (5, 30)

    def setup(self):
        self.batch = self._batch(2, 16)
        self.seq = 32 if self.tiny else 512
        self.dim = 256 if self.tiny else 4096
        self.pool = [(self.normal((self.batch, self.seq, self.dim)),)
                     for _ in range(self.pool_size)]
        self.items = self.batch * self.seq

    def run(self, x):
        with self.torch.no_grad():
            return [self.torch.nn.functional.gelu(x, approximate="tanh")]

    def config(self):
        return {"batch": self.batch, "seq": self.seq, "dim": self.dim}


class OpElementwiseChainBF16(Workload):
    name = "op_elementwise_chain_bf16"
    family = "ops"
    mode = "infer"
    unit = "M-elems"
    default_dtype = "bfloat16"
    description = "Fused pointwise chain x*sigmoid(x) + y*tanh(z), [16, 1024, 1024] bf16"
    iterations = (5, 30)

    def setup(self):
        self.shape = (2, 64, 64) if self.tiny else (16, 1024, 1024)
        self.pool = [(self.normal(self.shape), self.normal(self.shape),
                      self.normal(self.shape))
                     for _ in range(self.pool_size)]
        self.items = float(np.prod(self.shape)) / 1e6

    def run(self, x, y, z):
        torch = self.torch
        with torch.no_grad():
            return [x * torch.sigmoid(x) + y * torch.tanh(z)]

    def config(self):
        return {"shape": list(self.shape)}


class OpReduceSumMeanFP32(Workload):
    name = "op_reduce_sum_mean_fp32"
    family = "ops"
    mode = "infer"
    unit = "M-elems"
    default_dtype = "float32"
    description = "Multi-axis reductions sum + mean + argmax, [32, 1024, 1024] fp32"
    iterations = (5, 30)

    def setup(self):
        self.shape = (4, 64, 64) if self.tiny else (32, 1024, 1024)
        self.pool = [(self.normal(self.shape),) for _ in range(self.pool_size)]
        self.items = float(np.prod(self.shape)) / 1e6

    def run(self, x):
        with self.torch.no_grad():
            s = x.sum(dim=-1)
            m = x.mean(dim=1)
            idx = x.argmax(dim=-1)
            return [s + m + idx.to(x.dtype) * 1e-4]

    def config(self):
        return {"shape": list(self.shape)}


class OpSortTopkFP32(Workload):
    name = "op_sort_topk_fp32"
    family = "ops"
    mode = "infer"
    unit = "rows"
    default_dtype = "float32"
    description = "torch.topk(k=64) + torch.sort, [64, 8192] fp32 (sampling/routing)"
    iterations = (5, 25)

    def setup(self):
        self.rows = 8 if self.tiny else 64
        self.cols = 256 if self.tiny else 8192
        self.k = 16 if self.tiny else 64
        self.pool = [(self.normal((self.rows, self.cols)),)
                     for _ in range(self.pool_size)]
        self.items = self.rows

    def run(self, x):
        torch = self.torch
        with torch.no_grad():
            top_v, _ = torch.topk(x, self.k, dim=-1)
            sorted_v, _ = torch.sort(x, dim=-1)
            return [top_v.mean() + sorted_v[:, :self.k].mean()]

    def config(self):
        return {"rows": self.rows, "cols": self.cols, "k": self.k}


class OpEmbeddingBF16(Workload):
    name = "op_embedding_bf16"
    family = "ops"
    mode = "train"
    unit = "tokens"
    default_dtype = "bfloat16"
    description = "Large-vocab nn.Embedding forward+backward, vocab=128000 dim=1024 bf16"
    iterations = (5, 25)

    def setup(self):
        nn = self.torch.nn
        self.vocab = 2048 if self.tiny else 128000
        self.dim = 128 if self.tiny else 1024
        self.batch = self._batch(2, 4)
        self.seq = 32 if self.tiny else 1024
        self.model = self.init(
            nn.Embedding(self.vocab, self.dim)
        ).to(device=self.device, dtype=self.dtype).train()
        self.pool = [(self.integers(self.vocab, (self.batch, self.seq)),)
                     for _ in range(self.pool_size)]
        self.items = self.batch * self.seq

    def run(self, ids):
        loss = self.model(ids).float().pow(2).mean()
        loss.backward()
        self.model.zero_grad(set_to_none=True)
        return [loss]

    def config(self):
        return {"vocab": self.vocab, "dim": self.dim, "batch": self.batch, "seq": self.seq}


class OpGatherScatterFP32(Workload):
    name = "op_gather_scatter_fp32"
    family = "ops"
    mode = "infer"
    unit = "rows"
    default_dtype = "float32"
    description = "torch.gather + scatter_add_, [64, 4096] fp32 (MoE/GNN indexing)"
    iterations = (5, 30)

    def setup(self):
        self.rows = 8 if self.tiny else 64
        self.cols = 256 if self.tiny else 4096
        self.pool = [(self.normal((self.rows, self.cols)),
                      self.integers(self.cols, (self.rows, self.cols)))
                     for _ in range(self.pool_size)]
        self.items = self.rows

    def run(self, x, index):
        torch = self.torch
        with torch.no_grad():
            gathered = torch.gather(x, 1, index)
            out = torch.zeros_like(x)
            out.scatter_add_(1, index, gathered)
            return [out]

    def config(self):
        return {"rows": self.rows, "cols": self.cols}


class OpCatStackFP32(Workload):
    name = "op_cat_stack_fp32"
    family = "ops"
    mode = "infer"
    unit = "M-elems"
    default_dtype = "float32"
    description = "torch.cat + torch.stack over 8 tensors [16, 256, 256] fp32"
    iterations = (5, 30)

    def setup(self):
        self.count = 4 if self.tiny else 8
        self.shape = (2, 32, 32) if self.tiny else (16, 256, 256)
        self.pool = [tuple(self.normal(self.shape) for _ in range(self.count))
                     for _ in range(self.pool_size)]
        self.items = float(self.count * np.prod(self.shape)) / 1e6

    def run(self, *tensors):
        torch = self.torch
        with torch.no_grad():
            c = torch.cat(tensors, dim=1)
            s = torch.stack(tensors, dim=0)
            return [c.mean() + s.mean()]

    def config(self):
        return {"count": self.count, "shape": list(self.shape)}


class OpTransposeContiguousBF16(Workload):
    name = "op_transpose_contiguous_bf16"
    family = "ops"
    mode = "infer"
    unit = "M-elems"
    default_dtype = "bfloat16"
    description = "4D attention permute(0,2,1,3).contiguous(), [16, 1024, 16, 128] bf16"
    iterations = (5, 30)

    def setup(self):
        self.shape = (2, 32, 4, 32) if self.tiny else (16, 1024, 16, 128)
        self.pool = [(self.normal(self.shape),) for _ in range(self.pool_size)]
        self.items = float(np.prod(self.shape)) / 1e6

    def run(self, x):
        with self.torch.no_grad():
            return [x.permute(0, 2, 1, 3).contiguous()]

    def config(self):
        return {"shape": list(self.shape)}


class OpPoolInterpolateFP16(Workload):
    name = "op_pool_interpolate_fp16"
    family = "ops"
    mode = "infer"
    unit = "images"
    default_dtype = "float16"
    description = "F.max_pool2d + F.interpolate(bilinear), [32, 128, 128, 128] fp16"
    iterations = (5, 30)

    def setup(self):
        self.batch = self._batch(2, 32)
        self.channels = 16 if self.tiny else 128
        self.spatial = 16 if self.tiny else 128
        self.pool = [(self.normal((self.batch, self.channels, self.spatial, self.spatial)),)
                     for _ in range(self.pool_size)]
        self.items = self.batch

    def run(self, x):
        F = self.torch.nn.functional
        with self.torch.no_grad():
            down = F.max_pool2d(x, kernel_size=2, stride=2)
            up = F.interpolate(down, scale_factor=2.0, mode="bilinear", align_corners=False)
            return [up]

    def config(self):
        return {"batch": self.batch, "channels": self.channels, "hw": self.spatial}


class OpFftRfftFP32(Workload):
    name = "op_fft_rfft_fp32"
    family = "ops"
    mode = "infer"
    unit = "M-elems"
    default_dtype = "float32"
    description = "torch.fft.rfft + irfft round-trip, [64, 256, 256] fp32"
    iterations = (5, 25)

    def setup(self):
        self.shape = (4, 32, 32) if self.tiny else (64, 256, 256)
        self.pool = [(self.normal(self.shape),) for _ in range(self.pool_size)]
        self.items = float(np.prod(self.shape)) / 1e6

    def run(self, x):
        fft = self.torch.fft
        with self.torch.no_grad():
            spec = fft.rfft(x, dim=-1)
            back = fft.irfft(spec, n=self.shape[-1], dim=-1)
            return [back]

    def config(self):
        return {"shape": list(self.shape)}


class OpLinalgSolveSvdFP32(Workload):
    name = "op_linalg_solve_svd_fp32"
    family = "ops"
    mode = "infer"
    unit = "matrices"
    default_dtype = "float32"
    description = "Batched torch.linalg.solve + svd, [16, 128, 128] fp32"
    iterations = (5, 20)
    compile_what = None

    def setup(self):
        self.batch = self._batch(2, 16)
        self.dim = 16 if self.tiny else 128
        eye = self.torch.eye(self.dim, device=self.device, dtype=self.dtype) * float(self.dim)
        self.pool = [
            (self.normal((self.batch, self.dim, self.dim)) + eye,
             self.normal((self.batch, self.dim, 16 if self.tiny else 64)))
            for _ in range(self.pool_size)
        ]
        self.items = self.batch

    def run(self, a, b):
        la = self.torch.linalg
        with self.torch.no_grad():
            sol = la.solve(a, b)
            _u, s, _vh = la.svd(a, full_matrices=False)
            return [sol.mean() + s.mean()]

    def config(self):
        return {"batch": self.batch, "dim": self.dim}


class OpClipGradNormFP32(Workload):
    name = "op_clip_grad_norm_fp32"
    family = "ops"
    mode = "train"
    unit = "tensors"
    default_dtype = "float32"
    description = "torch.nn.utils.clip_grad_norm_ over 128 gradients [256, 256] fp32"
    iterations = (5, 25)
    compile_what = None

    def setup(self):
        nn = self.torch.nn
        self.num_tensors = 16 if self.tiny else 128
        self.dim = 32 if self.tiny else 256
        self.model = self.init(
            nn.ModuleList([nn.Linear(self.dim, self.dim, bias=False)
                           for _ in range(self.num_tensors)])
        ).to(device=self.device, dtype=self.dtype)
        self.pool = [
            tuple(self.normal((self.dim, self.dim)) for _ in range(self.num_tensors))
            for _ in range(2)
        ]
        self.items = self.num_tensors

    def run(self, *grads):
        for p, g in zip(self.model.parameters(), grads):
            p.grad = g.clone()
        total_norm = self.torch.nn.utils.clip_grad_norm_(
            list(self.model.parameters()), max_norm=1.0
        )
        return [total_norm if isinstance(total_norm, self.torch.Tensor)
                else self.torch.tensor(float(total_norm), device=self.device)]

    def config(self):
        return {"tensors": self.num_tensors, "shape": [self.dim, self.dim]}


# --------------------------------------------------------------------------
# Runtime overhead, CUDA Graph / compile, allocator, and transfer
# --------------------------------------------------------------------------

def _decoder_block(torch, hidden, heads, inter):
    nn = torch.nn
    F = nn.functional

    class _DecoderLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.heads = heads
            self.head_dim = hidden // heads
            self.norm1 = nn.RMSNorm(hidden, eps=1e-6)
            self.qkv = nn.Linear(hidden, 3 * hidden, bias=False)
            self.o_proj = nn.Linear(hidden, hidden, bias=False)
            self.norm2 = nn.RMSNorm(hidden, eps=1e-6)
            self.gate = nn.Linear(hidden, inter, bias=False)
            self.up = nn.Linear(hidden, inter, bias=False)
            self.down = nn.Linear(inter, hidden, bias=False)

        def forward(self, x):
            b, s, d = x.shape
            h = self.norm1(x)
            qkv = self.qkv(h).reshape(b, s, 3, self.heads, self.head_dim)
            q = qkv[:, :, 0].transpose(1, 2)
            k = qkv[:, :, 1].transpose(1, 2)
            v = qkv[:, :, 2].transpose(1, 2)
            attn = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            attn = attn.transpose(1, 2).reshape(b, s, d)
            x = x + self.o_proj(attn)
            h2 = self.norm2(x)
            return x + self.down(F.silu(self.gate(h2)) * self.up(h2))

    return _DecoderLayer()


class RtDispatchMicroOps(Workload):
    name = "rt_dispatch_micro_ops"
    family = "runtime"
    mode = "infer"
    unit = "ops"
    default_dtype = "float32"
    description = "200 chained small tensor ops [64, 64] isolating per-op host dispatch"
    iterations = (5, 30)

    def setup(self):
        self.steps = 20 if self.tiny else 200
        self.dim = 16 if self.tiny else 64
        self.pool = [(self.normal((self.dim, self.dim)),
                      self.normal((self.dim, self.dim)))
                     for _ in range(self.pool_size)]
        self.items = self.steps

    def run(self, x, y):
        with self.torch.no_grad():
            acc = x
            for _ in range(self.steps // 2):
                acc = acc * 0.99 + y * 0.01
                acc = self.torch.nn.functional.relu(acc)
            return [acc]

    def config(self):
        return {"ops": self.steps, "shape": [self.dim, self.dim]}


class RtTransformerLayerTrain(Workload):
    name = "rt_transformer_layer_train"
    family = "runtime"
    mode = "train"
    unit = "tokens"
    default_dtype = "bfloat16"
    description = ("4-layer Decoder Transformer training (RMSNorm+SDPA+SwiGLU+AdamW), "
                   "shape-stable for eager vs --compile reduce-overhead / CUDA Graph")
    iterations = (5, 20)

    def setup(self):
        nn = self.torch.nn
        self.layers = 2 if self.tiny else 4
        self.batch = self._batch(2, 4)
        self.seq = 32 if self.tiny else 512
        self.hidden = 128 if self.tiny else 1024
        self.heads = 4 if self.tiny else 16
        self.inter = 256 if self.tiny else 2816
        self.model = self.init(nn.Sequential(
            *[_decoder_block(self.torch, self.hidden, self.heads, self.inter)
              for _ in range(self.layers)]
        )).to(device=self.device, dtype=self.dtype).train()
        self.optimizer = self.torch.optim.AdamW(self.model.parameters(), lr=1e-4)
        self.pool = [(self.normal((self.batch, self.seq, self.hidden)),)
                     for _ in range(self.pool_size)]
        self.items = self.batch * self.seq

    def run(self, x):
        out = self.model(x)
        loss = out.float().pow(2).mean()
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        return [loss]

    def config(self):
        return {"layers": self.layers, "batch": self.batch,
                "seq": self.seq, "hidden": self.hidden}


class RtAllocChurn(Workload):
    name = "rt_alloc_churn"
    family = "runtime"
    mode = "infer"
    unit = "allocs"
    default_dtype = "float32"
    description = "Multi-size varying allocation/compute/release cycle for allocator throughput"
    iterations = (5, 25)
    compile_what = None

    def setup(self):
        self.sizes = (
            (1024, 4096, 16384, 65536) if self.tiny
            else (32768, 131072, 524288, 2097152, 8388608)
        )
        self.cycles = 4 if self.tiny else 8
        self.pool = [(self.normal((max(self.sizes),)),) for _ in range(self.pool_size)]
        self.items = len(self.sizes) * self.cycles

    def run(self, src):
        torch = self.torch
        with torch.no_grad():
            acc = torch.zeros((1,), device=self.device, dtype=self.dtype)
            for _ in range(self.cycles):
                live = []
                for n in self.sizes:
                    buf = torch.empty((n,), device=self.device, dtype=self.dtype)
                    buf.copy_(src[:n])
                    live.append(buf * 1.001 + 0.5)
                acc = acc + live[-1].mean()
                del live
            return [acc]

    def config(self):
        return {"sizes": list(self.sizes), "cycles": self.cycles}


class RtH2DD2HTransfer(Workload):
    name = "rt_h2d_d2h_transfer"
    family = "runtime"
    mode = "infer"
    unit = "MB"
    default_dtype = "float32"
    description = "Multi-size H2D + D2H round-trip sweep (64 KB, 4 MB, 32 MB)"
    iterations = (5, 25)
    compile_what = None

    def setup(self):
        self.elem_counts = (1024, 16384) if self.tiny else (16384, 1048576, 8388608)
        self.pool = [
            tuple(self.torch.tensor(
                self.rng.standard_normal((n,), dtype=np.float32), device="cpu"
            ) for n in self.elem_counts)
            for _ in range(2)
        ]
        # Round-trip bytes in MB (H2D + D2H = 2x payload).
        self.items = sum(self.elem_counts) * 4 * 2 / (1024.0 * 1024.0)

    def run(self, *host_tensors):
        with self.torch.no_grad():
            last_dev = None
            total = 0.0
            for h in host_tensors:
                d = h.to(self.device)
                back = d.to("cpu")
                total += float(back[0].item())
                last_dev = d
            return [last_dev + (total * 0.0)]

    def config(self):
        return {"bytes": [n * 4 for n in self.elem_counts]}


# --------------------------------------------------------------------------
# Modern sparse MoE & LoRA fine-tuning (pure PyTorch, zero external deps)
# --------------------------------------------------------------------------

class MoEBlockTrainBF16(Workload):
    name = "moe_block_train_bf16"
    family = "moe"
    mode = "train"
    unit = "tokens"
    default_dtype = "bfloat16"
    description = "8-expert Top-2 Sparse MoE router + SwiGLU FFN training step (bf16, AdamW)"
    iterations = (5, 20)

    def setup(self):
        nn = self.torch.nn
        F = nn.functional
        self.batch = self._batch(2, 4)
        self.seq = 32 if self.tiny else 512
        self.hidden = 128 if self.tiny else 1024
        self.inter = 128 if self.tiny else 1024
        self.num_experts = 4 if self.tiny else 8
        self.top_k = 2

        class _SparseMoE(nn.Module):
            def __init__(m, hidden, inter, num_experts, top_k):
                super().__init__()
                m.num_experts = num_experts
                m.top_k = top_k
                m.router = nn.Linear(hidden, num_experts, bias=False)
                m.w_gate = nn.Parameter(
                    self.torch.empty(num_experts, hidden, inter))
                m.w_up = nn.Parameter(
                    self.torch.empty(num_experts, hidden, inter))
                m.w_down = nn.Parameter(
                    self.torch.empty(num_experts, inter, hidden))

            def forward(m, x):
                b, s, d = x.shape
                flat = x.reshape(-1, d)
                probs = F.softmax(m.router(flat).float(), dim=-1)
                weights, indices = self.torch.topk(probs, m.top_k, dim=-1)
                weights = (weights / weights.sum(dim=-1, keepdim=True)).to(x.dtype)
                out = self.torch.zeros_like(flat)
                for e in range(m.num_experts):
                    mask = (indices == e)
                    token_w = (weights * mask.to(weights.dtype)).sum(dim=-1)
                    h = F.silu(flat @ m.w_gate[e]) * (flat @ m.w_up[e])
                    expert_out = h @ m.w_down[e]
                    out = out + expert_out * token_w.unsqueeze(-1)
                return out.reshape(b, s, d)

        self.model = self.init(
            _SparseMoE(self.hidden, self.inter, self.num_experts, self.top_k)
        ).to(device=self.device, dtype=self.dtype).train()
        self.optimizer = self.torch.optim.AdamW(self.model.parameters(), lr=1e-4)
        self.pool = [(self.normal((self.batch, self.seq, self.hidden)),)
                     for _ in range(self.pool_size)]
        self.items = self.batch * self.seq

    def run(self, x):
        loss = self.model(x).float().pow(2).mean()
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        return [loss]

    def config(self):
        return {"batch": self.batch, "seq": self.seq, "hidden": self.hidden,
                "experts": self.num_experts, "top_k": self.top_k}


class LoRAFinetuneBF16(Workload):
    name = "lora_finetune_bf16"
    family = "lora"
    mode = "train"
    unit = "tokens"
    default_dtype = "bfloat16"
    description = "Frozen 4-layer Transformer + LoRA (r=16) adapter fine-tuning (bf16, AdamW)"
    iterations = (5, 20)

    def setup(self):
        nn = self.torch.nn
        self.layers = 2 if self.tiny else 4
        self.batch = self._batch(2, 4)
        self.seq = 32 if self.tiny else 512
        self.hidden = 128 if self.tiny else 1024
        self.heads = 4 if self.tiny else 16
        self.inter = 256 if self.tiny else 2816
        self.rank = 4 if self.tiny else 16

        class _LoRALinear(nn.Module):
            def __init__(m, base, rank):
                super().__init__()
                m.base = base
                for p in m.base.parameters():
                    p.requires_grad_(False)
                m.lora_a = nn.Linear(base.in_features, rank, bias=False)
                m.lora_b = nn.Linear(rank, base.out_features, bias=False)
                m.scale = 2.0 / float(rank)

            def forward(m, x):
                return m.base(x) + m.lora_b(m.lora_a(x)) * m.scale

        blocks = []
        for _ in range(self.layers):
            layer = _decoder_block(self.torch, self.hidden, self.heads, self.inter)
            layer.qkv = _LoRALinear(layer.qkv, self.rank)
            layer.o_proj = _LoRALinear(layer.o_proj, self.rank)
            for name, p in layer.named_parameters():
                if "lora_" not in name:
                    p.requires_grad_(False)
            blocks.append(layer)
        self.model = self.init(nn.Sequential(*blocks)).to(
            device=self.device, dtype=self.dtype).train()
        trainable = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = self.torch.optim.AdamW(trainable, lr=2e-4)
        self.pool = [(self.normal((self.batch, self.seq, self.hidden)),)
                     for _ in range(self.pool_size)]
        self.items = self.batch * self.seq

    def run(self, x):
        loss = self.model(x).float().pow(2).mean()
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        return [loss]

    def config(self):
        return {"layers": self.layers, "batch": self.batch, "seq": self.seq,
                "hidden": self.hidden, "lora_rank": self.rank}


WORKLOADS = {cls.name: cls for cls in (
    # Real models (Transformers / Diffusers / Vision)
    Qwen3Prefill, Qwen3PrefillLong, Qwen3Decode, Qwen3DecodeStatic,
    Qwen3Train, Qwen3TrainBF16,
    MoEBlockTrainBF16, LoRAFinetuneBF16,
    SD15Sample, SD15VAEDecode, SD15Train, DDPMTrain,
    ResNet50Infer, ResNet50InferB1, ResNet50Train, ViTB16Train,
    BertBaseTrain, BertBaseInfer,
    # Torch-compat operators (25 high-frequency primitives)
    OpMatmulBF16, OpLinearBF16, OpConv2dFP16, OpConvTranspose2dFP16,
    OpRMSNormBF16, OpLayerNormFP32, OpGroupNormFP16,
    OpSoftmaxBF16, OpGeluBF16, OpSwiGLUMlpBF16, OpRoPEBF16,
    OpSDPACausalBF16, OpElementwiseChainBF16,
    OpReduceSumMeanFP32, OpSortTopkFP32, OpEmbeddingBF16,
    OpGatherScatterFP32, OpCatStackFP32, OpTransposeContiguousBF16,
    OpPoolInterpolateFP16, OpFftRfftFP32, OpLinalgSolveSvdFP32,
    OpCrossEntropyVocab, OpAdamWStep, OpClipGradNormFP32,
    # Runtime overhead, CUDA Graph / compile, allocator & transfer
    RtDispatchMicroOps, RtTransformerLayerTrain, RtAllocChurn, RtH2DD2HTransfer,
)}


def is_finite_number(value):
    return isinstance(value, float) and math.isfinite(value)
