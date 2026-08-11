# 线性回归、逻辑回归与 MLP：配套代码

GitHub 仓库：https://github.com/liurijin123/linear-logistic-mlp-basics

仓库状态：已于 2026-08-11 上传并验证 `main` 分支。

配套文章：《从线性回归到 MLP：深度学习训练机制解析》

适用环境：Windows 10/11 64 位、NVIDIA GPU、Python 3.13。

本目录用三个教学模型演示同一条 PyTorch 训练流程：前向传播、损失计算、梯度清零、反向传播和参数更新。数据由固定随机种子生成，只用于解释代码和模型机制，不代表真实遥感机理、数据分布或模型精度。

本文代码强制使用 NVIDIA GPU，不提供 CPU 自动回退。若 `torch.cuda.is_available()` 为 `False`，程序会直接报错，并提示回到上一篇文章检查驱动、虚拟环境和 PyTorch CUDA wheel。

## 在 VS Code 中运行

用 VS Code 直接打开本目录，然后在集成终端依次执行：

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip -i https://pypi.tuna.tsinghua.edu.cn/simple
python -m pip config --site set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
python -m pip config --site list
python -m pip install -r requirements-torch-cu126.txt
python -m pip install -r requirements-tools.txt
python -m pip check
python simple_models.py --output-dir outputs --seed 42
python verify.py --seed 42
```

上述镜像设置只写入当前 `.venv`。`requirements-torch-cu126.txt` 内部指定了 PyTorch 官方 CUDA 12.6 索引，因此安装 PyTorch 时不会改用普通 PyPI 镜像。若要取消当前环境的镜像配置，执行：

```powershell
python -m pip config --site unset global.index-url
```

如果 PowerShell 禁止执行激活脚本，可以将 VS Code 终端切换为“命令提示符”，再执行：

```bat
.venv\Scripts\activate.bat
```

## 运行结果

`simple_models.py` 会生成：

```text
outputs/
├─ 01_linear_regression.png
├─ 02_logistic_regression.png
├─ 03_mlp_nonlinear_boundary.png
├─ 04_mlp_forward_backward.png
└─ metrics.json
```

运行成功至少应同时满足：

- `outputs/` 中生成上述五个文件；
- `metrics.json` 中的 `device` 为 `cuda:0`；
- `metrics.json` 中的 `gpu` 显示正确的 NVIDIA 显卡型号；
- `torch_cuda_runtime` 显示 PyTorch wheel 携带的 CUDA 运行时版本；
- 手算梯度与 PyTorch `autograd` 梯度一致；
- 三个模型的训练损失均下降；
- `verify.py` 最后一行显示 `All checks passed.`。

验证脚本还会检查：

- 线性回归恢复模拟关系的主要趋势；
- 逻辑回归学习到接近预设阈值的分类边界；
- MLP 在 XOR 非线性任务上明显优于线性分类器；
- 五个预期输出文件完整生成。

在公众号项目中，可把 `--output-dir` 指向对应文章的 `assets/.../figures/` 目录。代码本身不依赖作者电脑上的固定路径。

系列示例版本核对于 2026-08-11。若 PyTorch 官方安装命令已经变化，应先更新 `requirements-torch-cu126.txt`，再运行测试。

pip 镜像配置参考：清华大学开源软件镜像站 PyPI 使用帮助（https://mirrors.tuna.tsinghua.edu.cn/help/pypi/）。

## 验证记录

本文已在 Windows、Python 3.13.1、PyTorch 2.12.1+cu126、CUDA 12.6 运行时和 NVIDIA GeForce RTX 2070 环境中完成 GPU 端到端验证。模型、输入、标签和手算梯度示例均位于 `cuda:0`。两次固定随机种子运行的指标与输出文件 SHA-256 哈希完全一致。
