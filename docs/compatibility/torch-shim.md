# 安装与部署 Torch 入口

[本栏目目录](index.md) · [English](torch-shim.en.md)

本页解决两件事：让应用找到正确的 `torch` 包，以及确认兼容运行时如何生效。首次使用请先跑[CPU 训练示例](torch.md)。

## 安装哪些包

核心 `jittor` 提供原生计算运行时，`jittor-compat` 提供 Torch 前端、部署工具和通用兼容机制。项目专属补丁由独立集成仓库提供。

从本仓源码安装时，在仓库根目录运行：

```bash
python -m pip install -e . -e ./compat
```

安装配套 wheel 时，使用同一解释器安装核心与兼容 wheel，遵守包元数据中的版本约束。真实 PyTorch 与兼容包都提供 `torch`，请分别放在独立环境中。

## 选择激活方式

单个脚本或 notebook 推荐显式激活，放在其它 Torch 导入之前：

```python
from jittor.compat.shim import activate
activate()
import torch
```

`activate()` 是进程级且幂等的。普通 `import jittor` 不占用 `torch` 命名空间；一个已经导入真实 PyTorch 的进程不能靠后续激活安全切换运行时。

部署入口适合应用启动后立即 `import torch` 的情况：

```bash
jittor-torch-shim
jittor-torch-shim --check
```

该命令由 `jittor-compat` 提供，也会部署随包的第三方 stub 与相应元数据。不要只复制一个 `torch/__init__.py` 文件。修改或升级兼容包后，用 `--check` 检查部署内容。

环境变量 `JITTOR_TORCH_SHIM=1` 可选择 Torch 模式，但不会把未安装的包放上 Python 搜索路径。先安装正确的兼容包，再使用环境变量；仅设置它不能修复 `ModuleNotFoundError`。

## 可选的 torch_xla 模拟层

`jittor-compat` 随包提供一个 `torch_xla` 客户端模拟层，使按 PyTorch/XLA 客户端 API 编写的程序可以运行在 Jittor 上。由于 `torch_xla` 一旦可导入就会让 Transformers 与 Accelerate 默认进入 XLA 分支（除非显式设置 `USE_TORCH_XLA=0`），`jittor-torch-shim` **默认不部署** `torch_xla` stub，让常规 CUDA/CPU 用户无需设置 `USE_TORCH_XLA=0`。

确实需要运行 PyTorch/XLA 客户端程序时，显式带上 `--with-xla` 部署即可：

```bash
jittor-torch-shim --with-xla
jittor-torch-shim --with-xla --check
```

它不是 XLA：`distributed.xla_multiprocessing.spawn` 按可见 CUDA 设备为每个进程分配一张卡，`core.xla_model` 把设备、序号和 rendezvous 调用映射到 Jittor 的 CUDA/CPU 设备，集合通信通过 rendezvous 目录做文件交换。没有 TPU、没有 XLA 编译、没有 SPMD 分片，也不支持 XLA FSDP。若已部署 `--with-xla` 但要跑原生 CUDA 任务，仍需设置 `USE_TORCH_XLA=0`。

启动方式与上游一致，`--num_cores` 不得超过本进程可见的 CUDA 设备数；纯 CPU 主机只支持单进程：

```bash
python examples/pytorch/xla_spawn.py --num_cores 2 train_script.py ...
```

## 缓存与项目扩展

兼容引导默认保留进程原有的 `HOME`、`XDG_CACHE_HOME`、`PIP_CACHE_DIR` 和 `TMPDIR`（不再为了保持真实主目录而设置 `JITTOR_TORCH_KEEP_HOME=1`），并在未显式指定 `JITTOR_HOME` 时默认复用用户级 Jittor 编译缓存以及 `JITTOR_SHARED_CACHE` 共享缓存；单进程默认不加载 MPI/NCCL，且自动识别 `CUDA_HOME` / `CUDA_PATH` 下的 `nvcc`（无需手工重复设置 `JT_BUILD_NVCC_PATH` 或 `JT_BUILD_USE_MPI=0` / `JT_BUILD_USE_NCCL=0`）。

- 需要按项目隔离沙盒主目录与临时目录时，显式设置 `JITTOR_TORCH_ISOLATE_HOME=1`（或传入 `activate(local_home=True)`）。
- 需要固定兼容运行时位置时，可在启动应用前设置 `JITTOR_TORCH_RUNTIME_ROOT`（默认位于 `${XDG_CACHE_HOME:-~/.cache}/jittor/torch-shim/`）。
- 兼容引导默认不会在 `import torch` / `activate()` 时自动编译当前工作目录下的源码扩展（不再需要设置 `JITTOR_TORCH_SKIP_EXT_BUILD=1`）；需要自动构建项目扩展时，显式设置 `JITTOR_TORCH_AUTO_BUILD_EXT=1` 或传入 `activate(build_extensions=True)`。这并不意味着任意 PyTorch 二进制扩展都能直接加载；遇到构建失败时，应保留编译器输出并核对集成仓库的扩展要求。

## 下游适配器

按 [JittorEcosystem](https://github.com/JittorEcosystem) 中目标仓库的安装说明添加所需集成。每个仓库单独维护版本限制与芯片支持；安装一个适配器不会自动证明该库的所有功能受支持。

适配器通过 `jittor.module_patches` 和 `jittor.external_backends` 入口注册补丁或扩展发现策略。新增集成与公共机制的归属见[兼容原则](../development/compatibility/principles.md)，实现组织见[分层架构](../development/compatibility/torch-architecture.md)。

## 排查导入问题

| 现象 | 先检查 |
| --- | --- |
| 找不到 `jittor.compat` | 当前解释器是否安装了 `jittor-compat`，而不是只设置核心的 `PYTHONPATH`。 |
| `import torch` 得到另一套运行时 | 解释器、虚拟环境和包安装位置是否一致；是否已经导入真实 PyTorch。 |
| 部署后缺少子模块 | 运行 `jittor-torch-shim --check`，查看缺失或不一致的文件。 |
| 下游库版本被拒绝 | 使用集成仓库声明支持的版本，不要绕过版本检查。 |

Python 代码中的 `torch.__version__` 可能反映 Jittor 版本；不能据此认定已安装某版原生 PyTorch。部署助手会打印用于检查入口与 Torch API 版本的命令。
