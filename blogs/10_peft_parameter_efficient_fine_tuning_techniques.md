# PEFT参数高效微调技术

## 概述

参数高效微调（Parameter-Efficient Fine-Tuning, PEFT）是一种通过只微调少量参数来适应大型预训练模型到特定任务的技术。本文档详细介绍了Transformers库中PEFT的原理、实现和应用实践。

## 1. PEFT基础概念

### 1.1 为什么需要PEFT

大型语言模型（LLMs）拥有数十亿甚至数千亿参数，完全微调需要大量计算资源和存储空间：

```python
# 传统全参数微调的内存消耗
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("facebook/llama-7b")
print(f"模型参数数量: {sum(p.numel() for p in model.parameters()) / 1e9:.1f}B")
print(f"全参数微调所需内存: {sum(p.numel() * 4 for p in model.parameters()) / 1e9:.1f}GB")
```

### 1.2 PEFT核心思想

PEFT的核心思想是：**只微调模型中的一小部分参数，或者添加少量可训练参数，保持大部分预训练参数冻结**。

```python
from peft import get_peft_model, LoraConfig, TaskType

# PEFT模型配置
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=8,                   # LoRA秩
    lora_alpha=32,         # LoRA缩放因子
    lora_dropout=0.1       # Dropout率
)

# 应用PEFT配置
peft_model = get_peft_model(model, peft_config)
print(f"PEFT可训练参数数量: {sum(p.numel() for p in peft_model.parameters() if p.requires_grad) / 1e6:.1f}M")
```

## 2. LoRA (Low-Rank Adaptation)

### 2.1 LoRA原理

LoRA通过低秩矩阵分解来近似权重更新，大大减少了可训练参数数量：

```python
import torch
import torch.nn as nn

class LoRALayer(nn.Module):
    def __init__(self, original_layer, r=8, alpha=32):
        super().__init__()
        self.original_layer = original_layer
        self.r = r
        self.alpha = alpha

        # 冻结原始层
        for param in self.original_layer.parameters():
            param.requires_grad = False

        # LoRA参数
        self.lora_A = nn.Parameter(torch.randn(original_layer.out_features, r))
        self.lora_B = nn.Parameter(torch.randn(r, original_layer.in_features))
        self.scaling = alpha / r

    def forward(self, x):
        original_output = self.original_layer(x)
        lora_output = (self.lora_B @ self.lora_A @ x.T).T * self.scaling
        return original_output + lora_output
```

### 2.2 LoRA配置和应用

```python
from peft import LoraConfig, get_peft_model

# LoRA配置
lora_config = LoraConfig(
    r=8,                              # 秩
    lora_alpha=32,                    # 缩放因子
    target_modules=["q_proj", "v_proj"],  # 目标模块
    lora_dropout=0.05,                # Dropout率
    bias="none",                      # 偏置处理
    task_type="CAUSAL_LM"             # 任务类型
)

# 应用LoRA
peft_model = get_peft_model(model, lora_config)
peft_model.print_trainable_parameters()
```

### 2.3 LoRA变体

#### 2.3.1 QLoRA (Quantized LoRA)

```python
from peft import prepare_model_for_kbit_training
from transformers import BitsAndBytesConfig

# 4位量化配置
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

# 加载量化模型
model = AutoModelForCausalLM.from_pretrained(
    "facebook/llama-7b",
    quantization_config=quantization_config,
    device_map="auto"
)

# 准备模型用于k位训练
model = prepare_model_for_kbit_training(model)

# 应用QLoRA
peft_config = LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj", "v_proj"])
peft_model = get_peft_model(model, peft_config)
```

## 3. Adapter Layers

### 3.1 标准Adapter

```python
from peft import AdapterConfig, get_peft_model

# Adapter配置
adapter_config = AdapterConfig(
    task_type="SEQ_CLS",          # 任务类型
    adapter_type="houlsby",       # Adapter类型
    reduction_factor=16,         # 降维因子
    non_linearity="relu",        # 激活函数
)

# 应用Adapter
peft_model = get_peft_model(model, adapter_config)
```

### 3.2 P-Tuning (Prompt Tuning)

```python
from peft import PromptTuningConfig, PromptTuningInit, get_peft_model

# P-Tuning配置
prompt_config = PromptTuningConfig(
    task_type="CAUSAL_LM",
    prompt_tuning_init=PromptTuningInit.TEXT,
    num_virtual_tokens=20,        # 虚拟token数量
    prompt_tuning_init_text="Classify if the tweet is a complaint or not:",
    tokenizer_name_or_path="facebook/llama-7b"
)

# 应用P-Tuning
peft_model = get_peft_model(model, prompt_config)
```

### 3.3 Prefix Tuning

```python
from peft import PrefixTuningConfig, get_peft_model

# Prefix Tuning配置
prefix_config = PrefixTuningConfig(
    task_type="CAUSAL_LM",
    num_virtual_tokens=20,        # 前缀token数量
    prefix_projection=True,       # 是否使用前缀投影
    encoder_hidden_size=768,      # 编码器隐藏层大小
)

# 应用Prefix Tuning
peft_model = get_peft_model(model, prefix_config)
```

## 4. 高级PEFT技术

### 4.1 LoRA+ (Enhanced LoRA)

```python
class EnhancedLoRALayer(nn.Module):
    def __init__(self, original_layer, r=8, alpha=32, use_gradient_checkpointing=True):
        super().__init__()
        self.original_layer = original_layer
        self.r = r
        self.alpha = alpha
        self.use_gradient_checkpointing = use_gradient_checkpointing

        # 冻结原始层
        for param in self.original_layer.parameters():
            param.requires_grad = False

        # 多层LoRA
        self.lora_layers = nn.ModuleList([
            self._create_lora_pair(original_layer.in_features, original_layer.out_features)
            for _ in range(2)  # 两层LoRA
        ])

        # 层间归一化
        self.layer_norm = nn.LayerNorm(original_layer.out_features)

    def _create_lora_pair(self, in_dim, out_dim):
        return nn.ModuleDict({
            'A': nn.Parameter(torch.randn(self.r, in_dim)),
            'B': nn.Parameter(torch.randn(out_dim, self.r))
        })

    def forward(self, x):
        original_output = self.original_layer(x)

        # 多层LoRA处理
        lora_output = x
        for i, lora_pair in enumerate(self.lora_layers):
            A, B = lora_pair['A'], lora_pair['B']
            lora_output = (B @ A @ lora_output.T).T * (self.alpha / self.r)

            if i < len(self.lora_layers) - 1:
                lora_output = self.layer_norm(lora_output)

        return original_output + lora_output
```

### 4.2 IA3 (Infused Adapter by Inhibiting and Amplifying Inner Activations)

```python
from peft import IA3Config, get_peft_model

# IA3配置
ia3_config = IA3Config(
    task_type="SEQ_CLS",
    target_modules=["q_proj", "k_proj", "v_proj"],
    feedforward_modules=["output.dense"],
)

# 应用IA3
peft_model = get_peft_model(model, ia3_config)
```

### 4.3 AdaLoRA (Adaptive LoRA)

```python
from peft import AdaLoRAConfig, get_peft_model

# AdaLoRA配置
adalora_config = AdaLoRAConfig(
    task_type="CAUSAL_LM",
    r=8,                           # 初始秩
    lora_alpha=32,                 # 缩放因子
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1,
    total_step=1000,               # 总训练步数
    tinit=200,                     # 初始预热步数
    tfinal=200,                    # 最终衰减步数
    delta_t=10,                    # 更新间隔
)

# 应用AdaLoRA
peft_model = get_peft_model(model, adalora_config)
```

## 5. 混合PEFT策略

### 5.1 LoRA + Adapter

```python
class HybridPEFTModel(nn.Module):
    def __init__(self, base_model, lora_config, adapter_config):
        super().__init__()
        self.base_model = base_model

        # 应用LoRA
        self.lora_layers = self._apply_lora(lora_config)

        # 应用Adapter
        self.adapter_layers = self._apply_adapter(adapter_config)

    def _apply_lora(self, config):
        lora_layers = nn.ModuleDict()
        for name, module in self.base_model.named_modules():
            if any(target in name for target in config.target_modules):
                lora_layers[name] = LoRALayer(module, config.r, config.lora_alpha)
        return lora_layers

    def forward(self, x):
        # 实现混合PEFT的前向传播
        pass
```

### 5.2 多任务PEFT

```python
class MultiTaskPEFT(nn.Module):
    def __init__(self, base_model, task_configs):
        super().__init__()
        self.base_model = base_model
        self.task_configs = task_configs
        self.task_adapters = nn.ModuleDict()

        # 为每个任务创建独立的PEFT模块
        for task_name, config in task_configs.items():
            self.task_adapters[task_name] = self._create_task_adapter(config)

    def _create_task_adapter(self, config):
        if config['type'] == 'lora':
            return LoRALayer(
                self.base_model,
                config['r'],
                config['alpha']
            )
        elif config['type'] == 'adapter':
            return AdapterLayer(
                self.base_model,
                config['reduction_factor']
            )

    def forward(self, x, task_name):
        task_adapter = self.task_adapters[task_name]
        return task_adapter(x)
```

## 6. PEFT训练实践

### 6.1 数据准备

```python
from datasets import load_dataset
from transformers import TrainingArguments, Trainer

# 加载数据集
dataset = load_dataset("imdb")

# 数据预处理
def preprocess_function(examples):
    return tokenizer(examples['text'], truncation=True, padding=True, max_length=512)

tokenized_dataset = dataset.map(preprocess_function, batched=True)
```

### 6.2 训练配置

```python
# 训练参数配置
training_args = TrainingArguments(
    output_dir="./peft_model",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    learning_rate=2e-4,
    num_train_epochs=3,
    logging_steps=10,
    evaluation_strategy="steps",
    save_strategy="steps",
    save_steps=500,
    load_best_model_at_end=True,
    fp16=True,                     # 使用混合精度
    gradient_checkpointing=True,   # 梯度检查点
)

# 创建Trainer
trainer = Trainer(
    model=peft_model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    tokenizer=tokenizer,
)
```

### 6.3 训练执行

```python
# 训练PEFT模型
trainer.train()

# 保存模型
peft_model.save_pretrained("./peft_model")
```

## 7. PEFT模型推理

### 7.1 基础推理

```python
from peft import PeftModel

# 加载基础模型和PEFT适配器
base_model = AutoModelForCausalLM.from_pretrained("facebook/llama-7b")
peft_model = PeftModel.from_pretrained(base_model, "./peft_model")

# 推理
inputs = tokenizer("Once upon a time", return_tensors="pt")
outputs = peft_model.generate(**inputs, max_length=50)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

### 7.2 批量推理

```python
def batch_inference(peft_model, tokenizer, prompts, batch_size=8):
    results = []

    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i + batch_size]
        inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True)

        outputs = peft_model.generate(**inputs, max_length=100)
        batch_results = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        results.extend(batch_results)

    return results
```

### 7.3 模型合并

```python
# 合并PEFT权重到基础模型
merged_model = peft_model.merge_and_unload()

# 保存合并后的模型
merged_model.save_pretrained("./merged_model")
```

## 8. PEFT评估和调试

### 8.1 性能评估

```python
import evaluate

# 加载评估指标
accuracy_metric = evaluate.load("accuracy")
f1_metric = evaluate.load("f1")

def evaluate_model(model, test_dataset):
    predictions = []
    references = []

    for example in test_dataset:
        inputs = tokenizer(example['text'], return_tensors="pt")
        outputs = model(**inputs)
        predicted_class = torch.argmax(outputs.logits, dim=-1)

        predictions.append(predicted_class.item())
        references.append(example['label'])

    # 计算指标
    accuracy = accuracy_metric.compute(predictions=predictions, references=references)
    f1 = f1_metric.compute(predictions=predictions, references=references, average="weighted")

    return {
        'accuracy': accuracy['accuracy'],
        'f1': f1['f1']
    }
```

### 8.2 梯度分析

```python
def analyze_gradients(model):
    gradients = {}

    for name, param in model.named_parameters():
        if param.grad is not None:
            gradients[name] = {
                'mean': param.grad.mean().item(),
                'std': param.grad.std().item(),
                'max': param.grad.max().item(),
                'min': param.grad.min().item()
            }

    return gradients
```

### 8.3 参数分析

```python
def analyze_parameters(model):
    analysis = {}

    for name, param in model.named_parameters():
        if param.requires_grad:
            analysis[name] = {
                'shape': param.shape,
                'num_params': param.numel(),
                'mean': param.data.mean().item(),
                'std': param.data.std().item()
            }

    return analysis
```

## 9. 最佳实践和优化

### 9.1 PEFT选择指南

```python
def select_peft_method(task_type, model_size, dataset_size, compute_budget):
    """
    根据任务和资源条件选择合适的PEFT方法
    """
    if model_size > 10e9 and compute_budget < 32:
        return "qlora"  # 大模型+低计算资源
    elif task_type == "generation" and dataset_size < 10000:
        return "p_tuning"  # 生成任务+小数据集
    elif task_type == "classification" and dataset_size > 100000:
        return "adapter"  # 分类任务+大数据集
    else:
        return "lora"  # 默认选择
```

### 9.2 超参数调优

```python
def grid_search_peft_config(model, dataset, param_grid):
    """
    PEFT配置网格搜索
    """
    best_config = None
    best_score = -float('inf')

    for r in param_grid['r']:
        for alpha in param_grid['alpha']:
            for dropout in param_grid['dropout']:
                # 创建PEFT配置
                config = LoraConfig(
                    r=r,
                    lora_alpha=alpha,
                    lora_dropout=dropout,
                    task_type="CAUSAL_LM"
                )

                # 训练和评估
                peft_model = get_peft_model(model, config)
                score = train_and_evaluate(peft_model, dataset)

                if score > best_score:
                    best_score = score
                    best_config = config

    return best_config, best_score
```

### 9.3 内存优化

```python
class MemoryEfficientPEFT(nn.Module):
    def __init__(self, base_model, peft_config):
        super().__init__()
        self.base_model = base_model
        self.peft_config = peft_config

        # 使用梯度检查点
        self.gradient_checkpointing = True

        # 使用混合精度
        self.mixed_precision = True

        # 分批处理大张量
        self.batch_size = peft_config.get('batch_size', 32)

    def forward(self, x):
        # 实现内存高效的前向传播
        pass
```

## 10. 应用场景

### 10.1 个性化微调

```python
class PersonalizedPEFT(nn.Module):
    def __init__(self, base_model, num_users):
        super().__init__()
        self.base_model = base_model
        self.num_users = num_users

        # 为每个用户创建独立的PEFT适配器
        self.user_adapters = nn.ModuleDict()
        for user_id in range(num_users):
            self.user_adapters[str(user_id)] = self._create_user_adapter()

    def _create_user_adapter(self):
        return LoRALayer(
            self.base_model,
            r=8,
            alpha=32
        )

    def forward(self, x, user_id):
        user_adapter = self.user_adapters[str(user_id)]
        return user_adapter(x)
```

### 10.2 领域适应

```python
class DomainAdaptivePEFT(nn.Module):
    def __init__(self, base_model, domains):
        super().__init__()
        self.base_model = base_model
        self.domains = domains

        # 为每个领域创建适配器
        self.domain_adapters = nn.ModuleDict()
        for domain in domains:
            self.domain_adapters[domain] = self._create_domain_adapter()

    def _create_domain_adapter(self):
        return AdapterLayer(
            self.base_model,
            reduction_factor=16
        )

    def forward(self, x, domain):
        domain_adapter = self.domain_adapters[domain]
        return domain_adapter(x)
```

## 总结

PEFT技术为大型语言模型的高效微调提供了强大的工具集。通过合理选择和应用LoRA、Adapter、P-Tuning等技术，可以在保持模型性能的同时，大幅降低训练成本和存储需求。Transformers库的PEFT集成使得这些技术变得易于使用和部署。