# torchao 量化支持

本文档介绍如何在 vLLM-Ascend 上使用 torchao 进行模型量化。

## 概述

torchao 是 PyTorch 官方提供的量化库，支持多种量化方法，包括：

- **Int8 Weight Only**: 仅量化权重到 int8
- **Int4 Weight Only**: 仅量化权重到 int4
- **Float8 Weight Only**: 仅量化权重到 float8
- **Dynamic Activation + Weight**: 动态激活量化 + 权重量化

vLLM-Ascend 从 v0.8.0 开始支持 `--quantization torchao`，复用上游 vLLM 的 torchao 实现，并添加了 NPU 兼容性检查。

## 快速开始

### 1. 安装依赖

确保已安装 `torchao` 库：

```bash
pip install torchao>=0.10.0
```

推荐使用 `torchao>=0.15.0` 以获得完整的特性支持。

### 2. 基本使用

使用 `--quantization torchao` 参数启用 torchao 量化：

```bash
vllm serve TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T \
    --quantization torchao \
    --dtype float16 \
    --gpu-memory-utilization 0.9
```

### 3. 指定量化配置

使用 `--torchao-config` 参数指定量化配置：

```bash
vllm serve TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T \
    --quantization torchao \
    --torchao-config '{"_data": {"type": "Int8WeightOnlyConfig"}}' \
    --dtype float16
```

## 支持的量化配置

### Int8WeightOnlyConfig

**描述**: 仅量化权重到 int8，激活使用原始精度。

**配置**:
```json
{
  "_data": {
    "type": "Int8WeightOnlyConfig"
  }
}
```

**优点**:
- 显存占用减少约 50%
- 推理速度提升（取决于硬件和算子优化）
- 精度损失小

**适用场景**:
- 显存受限的环境
- 对推理速度有要求的场景

### Int4WeightOnlyConfig

**描述**: 仅量化权重到 int4，显存占用更少。

**配置**:
```json
{
  "_data": {
    "type": "Int4WeightOnlyConfig"
  }
}
```

**优点**:
- 显存占用减少约 75%
- 适合超大模型推理

**注意事项**:
- 精度损失可能比 int8 大
- 需要硬件支持 int4 计算才能获得加速

### Float8WeightOnlyConfig

**描述**: 仅量化权重到 float8。

**配置**:
```json
{
  "_data": {
    "type": "Float8WeightOnlyConfig"
  }
}
```

**优点**:
- 精度损失小
- 适合 FP8 硬件

**注意事项**:
- 需要硬件支持 FP8 计算
- Ascend 910B 系列可能需要特定 CANN 版本

### Int8DynamicActivationInt8WeightConfig

**描述**: 动态激活量化（int8）+ 权重量化（int8）。

**配置**:
```json
{
  "_data": {
    "type": "Int8DynamicActivationInt8WeightConfig"
  }
}
```

**优点**:
- 显存和计算都使⽤ int8
- 推理速度提升明显

**注意事项**:
- 激活量化可能带来精度损失
- 需要硬件支持 int8 计算

## 配置示例

### 示例 1: 基本使用（Int8 Weight Only）

```bash
vllm serve meta-llama/Llama-2-7b-hf \
    --quantization torchao \
    --torchao-config '{"_data": {"type": "Int8WeightOnlyConfig"}}' \
    --dtype float16 \
    --gpu-memory-utilization 0.9
```

### 示例 2: 使用 Int4 Weight Only（显存优化）

```bash
vllm serve meta-llama/Llama-2-13b-hf \
    --quantization torchao \
    --torchao-config '{"_data": {"type": "Int4WeightOnlyConfig"}}' \
    --dtype float16 \
    --gpu-memory-utilization 0.95
```

### 示例 3: Python API 使用

```python
from vllm import LLM, SamplingParams

# 加载量化模型
llm = LLM(
    model="TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T",
    quantization="torchao",
    torchao_config='{"_data": {"type": "Int8WeightOnlyConfig"}}',
    dtype="float16",
    gpu_memory_utilization=0.9,
)

# 推理
prompts = [
    "Hello, my name is",
    "The capital of France is",
]
sampling_params = SamplingParams(temperature=0.7, max_tokens=50)
outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    print(output.prompt, output.outputs[0].text)
```

## NPU 特定说明

### 兼容性检查

vLLM-Ascend 会自动检查 torchao 配置在 NPU 上的兼容性：

- 如果检测到可能不兼容的配置，会发出警告
- 建议在 Ascend 910B 系列上使用 `Int8WeightOnlyConfig` 或 `Int4WeightOnlyConfig`

### 已知问题

1. **torchao 版本兼容性**:
   - torchao 0.17.0 需要 torch >= 2.11.0
   - 当前 vLLM-Ascend 可能使⽤较旧的 torch 版本
   - 建议: 使⽤ torchao 0.15.0 或升级 torch 到 >= 2.11.0

2. **弃用警告**:
   - `Int8WeightOnlyConfig` version 1 已弃用
   - 建议使用 version 2:
     ```json
     {
       "_data": {
         "type": "Int8WeightOnlyConfig",
         "version": 2
       }
     }
     ```

3. **NPU 特定警告**:
   - `Cannot create tensor with internal format`
   - 这是 NPU 上张量格式的限制，不影响功能

### 性能优化建议

1. **使用合适的数据类型**:
   - 推荐使用 `float16` 或 `bfloat16`
   - 避免使用 `float32`（显存占用大，速度慢）

2. **调整 GPU 显存利用率**:
   - 使用 `--gpu-memory-utilization 0.9` 或更高
   - 量化模型显存占用小，可以提高利用率

3. **批处理大小**:
   - 量化后显存占用减少，可以增加批处理大小
   - 使用 `--max-num-seqs` 调整并发请求数

## 故障排除

### 问题 1: 模型加载失败

**错误信息**:
```
ValueError: torchao is not installed. Please install it with `pip install torchao`.
```

**解决方法**:
```bash
pip install torchao>=0.10.0
```

### 问题 2: NPU 兼容性错误

**错误信息**:
```
UserWarning: torchao FP8 activation quantization config may not be fully supported on Ascend910B
```

**解决方法**:
- 使用 `Int8WeightOnlyConfig` 或 `Int4WeightOnlyConfig`
- 等待后续版本对 FP8 的支持

### 问题 3: 推理结果异常（NaN/Inf）

**可能原因**:
- 量化配置不适合该模型
- torchao 版本不兼容

**解决方法**:
- 尝试不同的量化配置
- 升级或降级 torchao 版本
- 检查 NPU 驱动和 CANN 版本

### 问题 4: 性能不如预期

**可能原因**:
- 缺少优化的 NPU 算子
- 量化配置不适合该硬件

**解决方法**:
- 使用 `benchmark_torchao.py` 进行性能测试
- 对比不同量化配置的性能
- 联系 Ascend 团队获取优化建议

## 高级主题

### 从 torchao 序列化权重加载

如果模型权重已经使⽤ torchao 序列化（例如从 HuggingFace Hub 下载的已量化模型），可以设置：

```python
from vllm import LLM

llm = LLM(
    model="username/model-torchao-quantized",
    # 自动检测 torchao 序列化权重
)
```

vLLM 会自动检测权重是否已量化，并跳过重复量化。

### 自定义量化配置

可以参考 [torchao 官方文档](https://github.com/pytorch/torchao) 创建自定义量化配置：

```python
from torchao.quantization import Int8WeightOnlyConfig, QuantizedConfig

# 创建自定义配置
config = Int8WeightOnlyConfig(
    weight_dtype=torch.int8,
    # 更多参数...
)

# 转换为 JSON
import json
config_dict = {"_data": config.__dict__}
config_json = json.dumps(config_dict)
```

然后在 vLLM 中使用：

```bash
vllm serve model \
    --quantization torchao \
    --torchao-config '<config_json>'
```

## 参考资料

- [torchao GitHub](https://github.com/pytorch/torchao)
- [vLLM torchao 文档](https://docs.vllm.ai/en/latest/models/quantization/torchao.html)
- [vLLM-Ascend 量化文档](./quantization.md)

## 更新日志

- **2026-06-18**: 初始版本，支持基本的 torchao 量化配置
- **待办**: 添加更多量化配置的支持
- **待办**: 优化 NPU 上的量化算子性能
- **待办**: 添加完整的性能基准测试数据
