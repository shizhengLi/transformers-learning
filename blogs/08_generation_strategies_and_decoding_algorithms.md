# 生成策略与解码算法

## 概述

在Transformer架构中，生成策略和解码算法是决定模型输出质量和性能的关键组件。本文档详细介绍了Transformers库中各种生成策略的原理、实现和应用场景。

## 1. 生成策略基础

### 1.1 自回归生成

自回归生成是序列到序列模型的标准生成方式，模型逐个token地生成输出序列：

```python
from transformers import GPT2LMHeadModel, GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

input_ids = tokenizer.encode("Once upon a time", return_tensors='pt')
output = model.generate(input_ids, max_length=50)
```

### 1.2 生成配置

Transformers提供了灵活的生成配置系统：

```python
from transformers import GenerationConfig

config = GenerationConfig(
    max_new_tokens=100,
    temperature=0.7,
    top_p=0.9,
    do_sample=True,
    pad_token_id=tokenizer.eos_token_id
)
```

## 2. 贪婪解码 (Greedy Decoding)

### 2.1 基本原理

贪婪解码在每个时间步选择概率最高的token：

```python
# 贪婪解码示例
output = model.generate(
    input_ids,
    max_length=50,
    do_sample=False  # 禁用采样，使用贪婪解码
)
```

### 2.2 优缺点

**优点：**
- 计算效率高
- 确定性输出
- 实现简单

**缺点：**
- 容易陷入重复循环
- 可能错过更好的序列
- 缺乏多样性

## 3. 束搜索 (Beam Search)

### 3.1 算法原理

束搜索维护多个候选序列，选择累积概率最高的序列：

```python
# 束搜索配置
output = model.generate(
    input_ids,
    max_length=50,
    num_beams=5,           # 束大小
    early_stopping=True,   # 提前停止
    no_repeat_ngram_size=2 # 避免重复
)
```

### 3.2 变体和优化

#### 3.2.1 多样化束搜索

```python
output = model.generate(
    input_ids,
    num_beams=5,
    num_beam_groups=5,     # 分组束搜索
    diversity_penalty=1.0 # 多样性惩罚
)
```

#### 3.2.2 长度惩罚

```python
output = model.generate(
    input_ids,
    num_beams=5,
    length_penalty=1.2     # 长度惩罚因子
)
```

## 4. 采样方法

### 4.1 温度采样 (Temperature Sampling)

```python
output = model.generate(
    input_ids,
    do_sample=True,
    temperature=0.7,       # 控制随机性
    top_k=50              # Top-k采样
)
```

### 4.2 Top-p采样 (Nucleus Sampling)

```python
output = model.generate(
    input_ids,
    do_sample=True,
    top_p=0.9,            # 累积概率阈值
    temperature=0.7
)
```

### 4.3 Top-k采样

```python
output = model.generate(
    input_ids,
    do_sample=True,
    top_k=50,             # 只考虑前k个候选
    temperature=0.7
)
```

## 5. 高级解码算法

### 5.1 对比搜索 (Contrastive Search)

```python
output = model.generate(
    input_ids,
    penalty_alpha=0.6,    # 对比搜索惩罚因子
    top_k=4,
    max_new_tokens=100
)
```

### 5.2 多样性引导生成

```python
output = model.generate(
    input_ids,
    do_sample=True,
    num_return_sequences=3,  # 生成多个序列
    diversity_penalty=0.5    # 多样性参数
)
```

## 6. 约束生成

### 6.1 强制令牌

```python
from transformers import ForceTokensLogitsProcessor

# 强制在特定位置生成特定token
force_tokens = [tokenizer.convert_tokens_to_ids("Hello")]
logits_processor = ForceTokensLogitsProcessor([5, force_tokens[0]])
```

### 6.2 禁止令牌

```python
from transformers import NoBadWordsLogitsProcessor

# 禁止特定token
bad_words = [[tokenizer.convert_tokens_to_ids("bad")]]
logits_processor = NoBadWordsLogitsProcessor(bad_words)
```

### 6.3 指导解码

```python
from transformers import LogitsProcessorList

class GuidedDecoding:
    def __init__(self, guide_tokens):
        self.guide_tokens = guide_tokens

    def __call__(self, input_ids, scores):
        # 实现指导解码逻辑
        return scores
```

## 7. 流式生成

### 7.1 基本流式生成

```python
from transformers import StoppingCriteria, StoppingCriteriaList

class MaxLengthCriteria(StoppingCriteria):
    def __init__(self, max_length):
        self.max_length = max_length

    def __call__(self, input_ids, scores, **kwargs):
        return input_ids.shape[1] >= self.max_length

# 流式生成
stopping_criteria = StoppingCriteriaList([MaxLengthCriteria(50)])
output = model.generate(
    input_ids,
    stopping_criteria=stopping_criteria,
    output_scores=True,
    return_dict_in_generate=True
)
```

### 7.2 实时生成

```python
def stream_generate(model, tokenizer, prompt, max_length=100):
    input_ids = tokenizer.encode(prompt, return_tensors='pt')

    for i in range(max_length):
        outputs = model(input_ids)
        next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1)
        input_ids = torch.cat([input_ids, next_token.unsqueeze(-1)], dim=-1)

        yield tokenizer.decode(input_ids[0], skip_special_tokens=True)

        if next_token.item() == tokenizer.eos_token_id:
            break
```

## 8. 性能优化

### 8.1 缓存机制

```python
# 使用past_key_values缓存
outputs = model.generate(
    input_ids,
    use_cache=True,
    return_dict_in_generate=True,
    output_scores=True
)
```

### 8.2 批量生成

```python
# 批量生成多个序列
prompts = ["Once upon a time", "In a galaxy far", "The weather today"]
input_ids = tokenizer(prompts, return_tensors='pt', padding=True)

outputs = model.generate(
    **input_ids,
    max_length=50,
    num_return_sequences=1,
    batch_size=3
)
```

## 9. 评估和调试

### 9.1 生成质量评估

```python
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

def evaluate_generation(model, tokenizer, prompts):
    scores = []
    for prompt in prompts:
        input_ids = tokenizer.encode(prompt, return_tensors='pt')
        outputs = model(input_ids)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=-1)
        scores.append(probabilities.max().item())
    return scores
```

### 9.2 调试工具

```python
def debug_generation(model, tokenizer, prompt):
    input_ids = tokenizer.encode(prompt, return_tensors='pt')

    with torch.no_grad():
        outputs = model(input_ids, output_attentions=True)
        attentions = outputs.attentions

    return {
        'input_text': prompt,
        'input_ids': input_ids,
        'attention_weights': attentions
    }
```

## 10. 最佳实践

### 10.1 选择合适的解码策略

- **创意写作**: 使用温度采样 + top-p
- **代码生成**: 使用束搜索 + 低温度
- **翻译**: 使用束搜索 + 长度归一化
- **对话系统**: 使用多样化采样

### 10.2 参数调优建议

```python
# 通用推荐配置
general_config = {
    'max_length': 100,
    'min_length': 10,
    'do_sample': True,
    'temperature': 0.7,
    'top_p': 0.9,
    'top_k': 50,
    'repetition_penalty': 1.2,
    'pad_token_id': tokenizer.eos_token_id
}
```

### 10.3 错误处理

```python
def safe_generate(model, tokenizer, prompt, **kwargs):
    try:
        input_ids = tokenizer.encode(prompt, return_tensors='pt')
        output = model.generate(input_ids, **kwargs)
        return tokenizer.decode(output[0], skip_special_tokens=True)
    except Exception as e:
        print(f"Generation error: {e}")
        return None
```

## 总结

生成策略和解码算法是Transformer模型的核心组件，选择合适的策略对模型性能至关重要。理解不同算法的原理和应用场景，能够帮助开发者更好地利用Transformers库构建高质量的生成系统。