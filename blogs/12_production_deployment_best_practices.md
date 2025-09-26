# 生产环境部署最佳实践

## 概述

将Transformer模型部署到生产环境需要考虑性能、可扩展性、可靠性和成本等多个方面。本文档详细介绍了Transformers模型在生产环境中的部署策略、优化技术和最佳实践。

## 1. 部署架构设计

### 1.1 服务架构

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

class DeploymentArchitecture:
    def __init__(self, model_name: str, device: str = "cuda"):
        self.model_name = model_name
        self.device = device
        self.model = None
        self.tokenizer = None
        self.max_input_length = 512
        self.batch_size = 32

    def load_model(self):
        """加载模型和tokenizer"""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()
            print(f"Model {self.model_name} loaded successfully")
        except Exception as e:
            print(f"Failed to load model: {e}")
            raise

    def warm_up(self):
        """模型预热"""
        dummy_input = "This is a warm up input to initialize the model"
        inputs = self.tokenizer(dummy_input, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            _ = self.model(**inputs)
        print("Model warmed up")

# API接口定义
class TextInput(BaseModel):
    text: str
    max_length: Optional[int] = 512

class BatchTextInput(BaseModel):
    texts: List[str]
    max_length: Optional[int] = 512

class PredictionOutput(BaseModel):
    predictions: List[dict]
    processing_time: float

# FastAPI应用
app = FastAPI(title="Transformers Model API", version="1.0.0")

# 全局模型实例
model_service = None

@app.on_event("startup")
async def startup_event():
    global model_service
    model_service = DeploymentArchitecture("bert-base-uncased")
    model_service.load_model()
    model_service.warm_up()

@app.post("/predict", response_model=PredictionOutput)
async def predict(input_data: TextInput):
    """单条文本预测"""
    import time
    start_time = time.time()

    try:
        # 输入验证
        if len(input_data.text) > input_data.max_length:
            raise HTTPException(status_code=400, detail="Input text too long")

        # 模型推理
        inputs = model_service.tokenizer(
            input_data.text,
            return_tensors="pt",
            truncation=True,
            max_length=input_data.max_length
        )

        inputs = {k: v.to(model_service.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model_service.model(**inputs)

        # 处理输出
        predictions = torch.softmax(outputs.logits, dim=-1)
        top_predictions = torch.topk(predictions, k=3)

        result = []
        for i, (score, idx) in enumerate(zip(top_predictions[0][0], top_predictions[1][0])):
            result.append({
                "label": int(idx.item()),
                "score": float(score.item()),
                "rank": i + 1
            })

        processing_time = time.time() - start_time

        return PredictionOutput(
            predictions=result,
            processing_time=processing_time
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/batch_predict", response_model=PredictionOutput)
async def batch_predict(input_data: BatchTextInput):
    """批量文本预测"""
    import time
    start_time = time.time()

    try:
        # 批量处理
        all_predictions = []

        for i in range(0, len(input_data.texts), model_service.batch_size):
            batch_texts = input_data.texts[i:i + model_service.batch_size]

            inputs = model_service.tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=input_data.max_length
            )

            inputs = {k: v.to(model_service.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model_service.model(**inputs)

            # 处理批量输出
            batch_predictions = torch.softmax(outputs.logits, dim=-1)
            top_predictions = torch.topk(batch_predictions, k=3)

            for j in range(len(batch_texts)):
                text_predictions = []
                for k, (score, idx) in enumerate(zip(
                    top_predictions[0][j], top_predictions[1][j]
                )):
                    text_predictions.append({
                        "label": int(idx.item()),
                        "score": float(score.item()),
                        "rank": k + 1
                    })
                all_predictions.append({
                    "text_index": i + j,
                    "predictions": text_predictions
                })

        processing_time = time.time() - start_time

        return PredictionOutput(
            predictions=all_predictions,
            processing_time=processing_time
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """健康检查接口"""
    return {"status": "healthy", "model": model_service.model_name}
```

### 1.2 容器化部署

```dockerfile
# Dockerfile
FROM python:3.9-slim

# 设置工作目录
WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# 复制requirements文件
COPY requirements.txt .

# 安装Python依赖
RUN pip install --no-cache-dir -r requirements.txt

# 复制应用代码
COPY . .

# 设置环境变量
ENV PYTHONPATH=/app
ENV MODEL_NAME=bert-base-uncased
ENV DEVICE=cuda

# 暴露端口
EXPOSE 8000

# 启动命令
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

```yaml
# docker-compose.yml
version: '3.8'

services:
  transformer-api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - MODEL_NAME=bert-base-uncased
      - DEVICE=cuda
    volumes:
      - ./models:/app/models
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    restart: unless-stopped

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
    depends_on:
      - transformer-api
    restart: unless-stopped
```

## 2. 性能优化

### 2.1 模型量化

```python
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import BitsAndBytesConfig

class QuantizedModelDeployment:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None

    def load_8bit_model(self):
        """加载8位量化模型"""
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        # 8位量化配置
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_8bit_compute_dtype=torch.float16
        )

        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            quantization_config=quantization_config,
            device_map="auto"
        )

    def load_4bit_model(self):
        """加载4位量化模型"""
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        # 4位量化配置
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True
        )

        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            quantization_config=quantization_config,
            device_map="auto"
        )

    def benchmark_quantization(self, test_texts):
        """基准测试量化效果"""
        import time
        results = {}

        for quant_type in ["fp32", "int8", "int4"]:
            # 加载对应量化版本的模型
            if quant_type == "fp32":
                self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            elif quant_type == "int8":
                self.load_8bit_model()
            elif quant_type == "int4":
                self.load_4bit_model()

            # 测量推理时间
            start_time = time.time()
            for text in test_texts:
                inputs = self.tokenizer(text, return_tensors="pt")
                with torch.no_grad():
                    outputs = self.model(**inputs)
            end_time = time.time()

            # 测量内存使用
            if torch.cuda.is_available():
                memory_used = torch.cuda.memory_allocated() / 1024 / 1024
            else:
                memory_used = 0

            results[quant_type] = {
                "inference_time": end_time - start_time,
                "memory_mb": memory_used,
                "model_size_mb": sum(p.numel() * p.element_size() for p in self.model.parameters()) / 1024 / 1024
            }

        return results
```

### 2.2 模型蒸馏

```python
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

class DistillationTrainer:
    def __init__(self, teacher_model_name, student_model_name):
        self.teacher_model = AutoModel.from_pretrained(teacher_model_name)
        self.student_model = AutoModel.from_pretrained(student_model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(teacher_model_name)

        # 冻结教师模型
        for param in self.teacher_model.parameters():
            param.requires_grad = False

    def train_distillation(self, train_dataloader, epochs=3, temperature=2.0, alpha=0.5):
        """训练蒸馏模型"""
        self.student_model.train()
        self.teacher_model.eval()

        optimizer = torch.optim.Adam(self.student_model.parameters(), lr=1e-4)

        for epoch in range(epochs):
            total_loss = 0

            for batch in train_dataloader:
                inputs = {k: v.to(self.device) for k, v in batch.items()}

                # 教师模型预测
                with torch.no_grad():
                    teacher_outputs = self.teacher_model(**inputs)
                    teacher_logits = teacher_outputs.logits

                # 学生模型预测
                student_outputs = self.student_model(**inputs)
                student_logits = student_outputs.logits

                # 计算蒸馏损失
                distillation_loss = self._compute_distillation_loss(
                    student_logits, teacher_logits, temperature
                )

                # 计算学生损失
                student_loss = nn.CrossEntropyLoss()(
                    student_logits, batch['labels']
                )

                # 总损失
                total_loss = alpha * distillation_loss + (1 - alpha) * student_loss

                # 反向传播
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

                total_loss += total_loss.item()

            print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_dataloader)}")

    def _compute_distillation_loss(self, student_logits, teacher_logits, temperature):
        """计算蒸馏损失"""
        # 软标签
        teacher_soft = torch.softmax(teacher_logits / temperature, dim=-1)
        student_soft = torch.softmax(student_logits / temperature, dim=-1)

        # KL散度损失
        distillation_loss = nn.KLDivLoss(reduction='batchmean')(
            torch.log(student_soft), teacher_soft
        )

        return distillation_loss * (temperature ** 2)
```

### 2.3 批处理优化

```python
class BatchProcessor:
    def __init__(self, model, tokenizer, max_batch_size=32):
        self.model = model
        self.tokenizer = tokenizer
        self.max_batch_size = max_batch_size

    def dynamic_batch_processing(self, texts, max_length=512):
        """动态批处理"""
        # 根据文本长度分组
        length_groups = self._group_by_length(texts)

        all_results = []

        for length_group in length_groups:
            # 动态调整batch size
            optimal_batch_size = self._calculate_optimal_batch_size(
                length_group['texts'], max_length
            )

            # 分批处理
            for i in range(0, len(length_group['texts']), optimal_batch_size):
                batch_texts = length_group['texts'][i:i + optimal_batch_size]

                # 处理批次
                batch_results = self._process_batch(batch_texts, max_length)
                all_results.extend(batch_results)

        return all_results

    def _group_by_length(self, texts):
        """按文本长度分组"""
        lengths = [len(text) for text in texts]
        groups = {}

        for i, length in enumerate(lengths):
            group_key = (length // 64) * 64  # 64个字符为一组
            if group_key not in groups:
                groups[group_key] = []
            groups[group_key].append((i, texts[i]))

        return [
            {'texts': [text for _, text in group],
             'indices': [idx for idx, _ in group]}
            for group in groups.values()
        ]

    def _calculate_optimal_batch_size(self, texts, max_length):
        """计算最优batch size"""
        # 估算内存使用
        avg_length = sum(len(text) for text in texts) / len(texts)
        estimated_memory_per_sample = (avg_length * 4) / 1024 / 1024  # MB

        # 可用内存
        if torch.cuda.is_available():
            available_memory = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()
            available_memory_mb = available_memory / 1024 / 1024
        else:
            available_memory_mb = 8192  # 8GB

        # 计算最优batch size
        optimal_batch = min(
            int(available_memory_mb * 0.8 / estimated_memory_per_sample),
            self.max_batch_size
        )

        return max(1, optimal_batch)

    def _process_batch(self, batch_texts, max_length):
        """处理单个批次"""
        inputs = self.tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length
        )

        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        return self._format_outputs(outputs, batch_texts)

    def _format_outputs(self, outputs, texts):
        """格式化输出"""
        predictions = torch.softmax(outputs.logits, dim=-1)
        top_predictions = torch.topk(predictions, k=3)

        results = []
        for i, text in enumerate(texts):
            text_results = []
            for j, (score, idx) in enumerate(zip(
                top_predictions[0][i], top_predictions[1][i]
            )):
                text_results.append({
                    "label": int(idx.item()),
                    "score": float(score.item()),
                    "rank": j + 1
                })
            results.append({
                "text": text,
                "predictions": text_results
            })

        return results
```

## 3. 可扩展性设计

### 3.1 负载均衡

```python
import asyncio
import aiohttp
from typing import List, Dict, Any

class LoadBalancer:
    def __init__(self, backend_urls: List[str]):
        self.backend_urls = backend_urls
        self.current_index = 0
        self.health_status = {url: True for url in backend_urls}

    async def get_next_backend(self) -> str:
        """获取下一个可用的后端"""
        # 简单的轮询算法
        for _ in range(len(self.backend_urls)):
            backend = self.backend_urls[self.current_index]
            self.current_index = (self.current_index + 1) % len(self.backend_urls)

            if self.health_status[backend]:
                return backend

        raise Exception("No healthy backends available")

    async def check_health(self):
        """健康检查"""
        async with aiohttp.ClientSession() as session:
            for url in self.backend_urls:
                try:
                    async with session.get(f"{url}/health") as response:
                        self.health_status[url] = response.status == 200
                except:
                    self.health_status[url] = False

    async def distribute_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """分发请求到后端"""
        backend = await self.get_next_backend()

        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(
                    f"{backend}/predict",
                    json=request_data
                ) as response:
                    return await response.json()
            except Exception as e:
                # 标记为不健康并重试
                self.health_status[backend] = False
                return await self.distribute_request(request_data)
```

### 3.2 缓存策略

```python
import redis
import json
import hashlib
from typing import Optional

class PredictionCache:
    def __init__(self, redis_url: str = "redis://localhost:6379", ttl: int = 3600):
        self.redis_client = redis.from_url(redis_url)
        self.ttl = ttl

    def _generate_key(self, text: str, model_name: str) -> str:
        """生成缓存键"""
        content = f"{text}:{model_name}"
        return hashlib.md5(content.encode()).hexdigest()

    def get_prediction(self, text: str, model_name: str) -> Optional[Dict]:
        """获取缓存的预测结果"""
        key = self._generate_key(text, model_name)
        cached_result = self.redis_client.get(key)

        if cached_result:
            return json.loads(cached_result)
        return None

    def cache_prediction(self, text: str, model_name: str, prediction: Dict):
        """缓存预测结果"""
        key = self._generate_key(text, model_name)
        self.redis_client.setex(key, self.ttl, json.dumps(prediction))

    def clear_cache(self, pattern: str = "*"):
        """清除缓存"""
        keys = self.redis_client.keys(pattern)
        if keys:
            self.redis_client.delete(*keys)

# 集成缓存的预测服务
class CachedPredictionService:
    def __init__(self, model_service: DeploymentArchitecture, cache: PredictionCache):
        self.model_service = model_service
        self.cache = cache

    async def predict_with_cache(self, text: str) -> Dict:
        """带缓存的预测"""
        # 检查缓存
        cached_result = self.cache.get_prediction(text, self.model_service.model_name)
        if cached_result:
            cached_result['from_cache'] = True
            return cached_result

        # 进行预测
        result = await self.predict(text)

        # 缓存结果
        self.cache.cache_prediction(text, self.model_service.model_name, result)
        result['from_cache'] = False

        return result
```

## 4. 监控和日志

### 4.1 性能监控

```python
import time
import psutil
import torch
from prometheus_client import Counter, Histogram, Gauge, start_http_server

class ModelMonitor:
    def __init__(self):
        # Prometheus指标
        self.prediction_counter = Counter(
            'model_predictions_total',
            'Total number of predictions',
            ['model_name', 'status']
        )

        self.prediction_duration = Histogram(
            'model_prediction_duration_seconds',
            'Time spent on predictions',
            ['model_name']
        )

        self.memory_usage = Gauge(
            'model_memory_usage_bytes',
            'Memory usage by the model',
            ['model_name', 'memory_type']
        )

        self.gpu_usage = Gauge(
            'model_gpu_usage_percent',
            'GPU usage percentage',
            ['model_name', 'gpu_id']
        )

    def monitor_prediction(self, model_name: str):
        """监控预测性能"""
        def decorator(func):
            def wrapper(*args, **kwargs):
                start_time = time.time()
                try:
                    result = func(*args, **kwargs)
                    self.prediction_counter.labels(
                        model_name=model_name,
                        status='success'
                    ).inc()
                    return result
                except Exception as e:
                    self.prediction_counter.labels(
                        model_name=model_name,
                        status='error'
                    ).inc()
                    raise e
                finally:
                    duration = time.time() - start_time
                    self.prediction_duration.labels(
                        model_name=model_name
                    ).observe(duration)
            return wrapper
        return decorator

    def update_memory_metrics(self, model_name: str):
        """更新内存指标"""
        process = psutil.Process()
        memory_info = process.memory_info()

        self.memory_usage.labels(
            model_name=model_name,
            memory_type='rss'
        ).set(memory_info.rss)

        self.memory_usage.labels(
            model_name=model_name,
            memory_type='vms'
        ).set(memory_info.vms)

    def update_gpu_metrics(self, model_name: str):
        """更新GPU指标"""
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                gpu_memory = torch.cuda.memory_allocated(i) / 1024 / 1024
                gpu_utilization = torch.cuda.utilization(i)

                self.memory_usage.labels(
                    model_name=model_name,
                    memory_type=f'gpu_{i}'
                ).set(gpu_memory * 1024 * 1024)

                self.gpu_usage.labels(
                    model_name=model_name,
                    gpu_id=str(i)
                ).set(gpu_utilization)

# 使用监控装饰器
monitor = ModelMonitor()

@monitor.monitor_prediction(model_name="bert-base-uncased")
def predict_text(model_service, text: str):
    """被监控的预测函数"""
    inputs = model_service.tokenizer(text, return_tensors="pt")
    inputs = {k: v.to(model_service.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model_service.model(**inputs)

    return outputs
```

### 4.2 日志管理

```python
import logging
import json
from datetime import datetime
from typing import Dict, Any

class ModelLogger:
    def __init__(self, log_file: str = "model_logs.json"):
        self.log_file = log_file
        self.logger = self._setup_logger()

    def _setup_logger(self):
        """设置日志记录器"""
        logger = logging.getLogger('model_service')
        logger.setLevel(logging.INFO)

        # 文件处理器
        file_handler = logging.FileHandler('model_service.log')
        file_handler.setLevel(logging.INFO)

        # 控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        # 格式化器
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

        return logger

    def log_prediction(self, request_data: Dict[str, Any],
                      response_data: Dict[str, Any],
                      processing_time: float):
        """记录预测日志"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "request": {
                "text_length": len(request_data.get('text', '')),
                "max_length": request_data.get('max_length', 512)
            },
            "response": {
                "predictions_count": len(response_data.get('predictions', [])),
                "top_score": max([p.get('score', 0) for p in response_data.get('predictions', [])]) if response_data.get('predictions') else 0
            },
            "performance": {
                "processing_time": processing_time,
                "memory_usage": self._get_memory_usage()
            }
        }

        self.logger.info(f"Prediction: {json.dumps(log_entry)}")
        self._save_to_file(log_entry)

    def log_error(self, error: Exception, request_data: Dict[str, Any]):
        """记录错误日志"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "error": {
                "type": type(error).__name__,
                "message": str(error)
            },
            "request": request_data
        }

        self.logger.error(f"Error: {json.dumps(log_entry)}")
        self._save_to_file(log_entry)

    def _get_memory_usage(self):
        """获取内存使用情况"""
        process = psutil.Process()
        memory_info = process.memory_info()
        return {
            "rss_mb": memory_info.rss / 1024 / 1024,
            "vms_mb": memory_info.vms / 1024 / 1024
        }

    def _save_to_file(self, log_entry: Dict[str, Any]):
        """保存日志到文件"""
        try:
            with open(self.log_file, 'a') as f:
                f.write(json.dumps(log_entry) + '\n')
        except Exception as e:
            self.logger.error(f"Failed to save log entry: {e}")
```

## 5. 安全性和隐私

### 5.1 输入验证和清理

```python
import re
from typing import Optional

class InputValidator:
    def __init__(self, max_length: int = 10000, allowed_chars: str = None):
        self.max_length = max_length
        self.allowed_chars = allowed_chars or self._get_default_allowed_chars()

    def _get_default_allowed_chars(self):
        """获取默认允许的字符"""
        return (
            "abcdefghijklmnopqrstuvwxyz"
            "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            "0123456789"
            " .,!?;:'\"()-"
            "áéíóúüñ¿¡"
            "àâäçéèêëïîôùûüÿœæ"
            "абвгдежзийклмнопрстуфхцчшщъыьэюя"
            "ΑΒΓΔΕΖΗΘΙΚΛΜΝΞΟΠΡΣΤΥΦΧΨΩ"
            "αβγδεζηθικλμνξοπρστυφχψω"
        )

    def validate_input(self, text: str) -> tuple[bool, Optional[str]]:
        """验证输入文本"""
        # 检查长度
        if len(text) > self.max_length:
            return False, f"Input text exceeds maximum length of {self.max_length}"

        # 检查空输入
        if not text.strip():
            return False, "Input text is empty"

        # 检查字符
        invalid_chars = set(text) - set(self.allowed_chars)
        if invalid_chars:
            return False, f"Input contains invalid characters: {invalid_chars}"

        # 检查潜在恶意内容
        if self._detect_malicious_content(text):
            return False, "Input contains potentially malicious content"

        return True, None

    def _detect_malicious_content(self, text: str) -> bool:
        """检测潜在恶意内容"""
        # SQL注入模式
        sql_patterns = [
            r'(union\s+select)',
            r'(drop\s+table)',
            r'(insert\s+into)',
            r'(delete\s+from)',
            r'(update\s+.*\s+set)',
        ]

        # XSS模式
        xss_patterns = [
            r'<script.*?>.*?</script>',
            r'javascript:',
            r'vbscript:',
            r'onload=',
            r'onerror=',
        ]

        # 命令注入模式
        cmd_patterns = [
            r';\s*\w+',
            r'\|\s*\w+',
            r'&\s*\w+',
            r'\$\(',
            r'`.*`',
        ]

        all_patterns = sql_patterns + xss_patterns + cmd_patterns

        for pattern in all_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True

        return False

    def sanitize_input(self, text: str) -> str:
        """清理输入文本"""
        # 移除HTML标签
        text = re.sub(r'<[^>]*>', '', text)

        # 移除特殊字符
        text = ''.join(char for char in text if char in self.allowed_chars)

        # 移除多余空格
        text = re.sub(r'\s+', ' ', text).strip()

        return text
```

### 5.2 速率限制

```python
import time
from collections import defaultdict
from typing import Dict, List

class RateLimiter:
    def __init__(self, max_requests: int = 100, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests = defaultdict(list)

    def is_allowed(self, client_id: str) -> bool:
        """检查是否允许请求"""
        current_time = time.time()
        client_requests = self.requests[client_id]

        # 移除过期的请求
        client_requests = [
            req_time for req_time in client_requests
            if current_time - req_time < self.window_seconds
        ]

        # 检查是否超过限制
        if len(client_requests) >= self.max_requests:
            return False

        # 记录新请求
        client_requests.append(current_time)
        self.requests[client_id] = client_requests

        return True

    def get_remaining_requests(self, client_id: str) -> int:
        """获取剩余请求数"""
        current_time = time.time()
        client_requests = self.requests[client_id]

        # 移除过期的请求
        client_requests = [
            req_time for req_time in client_requests
            if current_time - req_time < self.window_seconds
        ]

        self.requests[client_id] = client_requests
        return max(0, self.max_requests - len(client_requests))

    def reset_client(self, client_id: str):
        """重置客户端请求记录"""
        if client_id in self.requests:
            del self.requests[client_id]

# 使用FastAPI集成速率限制
from fastapi import Request, HTTPException, Depends

rate_limiter = RateLimiter(max_requests=100, window_seconds=60)

async def get_client_id(request: Request) -> str:
    """获取客户端ID"""
    # 可以使用IP地址、API密钥或其他标识符
    return request.client.host

async def rate_limit_dependency(client_id: str = Depends(get_client_id)):
    """速率限制依赖"""
    if not rate_limiter.is_allowed(client_id):
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded. Please try again later."
        )

# 在API端点中使用
@app.post("/predict", dependencies=[Depends(rate_limit_dependency)])
async def predict(input_data: TextInput):
    """带速率限制的预测接口"""
    # 原有的预测逻辑
    pass
```

## 6. 灾难恢复

### 6.1 备份和恢复

```python
import shutil
import os
from datetime import datetime
import json

class BackupManager:
    def __init__(self, model_dir: str, backup_dir: str = "./backups"):
        self.model_dir = model_dir
        self.backup_dir = backup_dir
        os.makedirs(backup_dir, exist_ok=True)

    def create_backup(self, backup_name: str = None) -> str:
        """创建模型备份"""
        if backup_name is None:
            backup_name = f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        backup_path = os.path.join(self.backup_dir, backup_name)

        try:
            # 复制模型文件
            shutil.copytree(self.model_dir, backup_path)

            # 创建备份元数据
            metadata = {
                "backup_name": backup_name,
                "timestamp": datetime.now().isoformat(),
                "model_dir": self.model_dir,
                "backup_path": backup_path,
                "files": self._get_file_list(backup_path)
            }

            with open(os.path.join(backup_path, "backup_metadata.json"), 'w') as f:
                json.dump(metadata, f, indent=2)

            return backup_path

        except Exception as e:
            # 清理失败的备份
            if os.path.exists(backup_path):
                shutil.rmtree(backup_path)
            raise e

    def restore_backup(self, backup_name: str) -> bool:
        """从备份恢复模型"""
        backup_path = os.path.join(self.backup_dir, backup_name)

        if not os.path.exists(backup_path):
            raise FileNotFoundError(f"Backup {backup_name} not found")

        try:
            # 创建临时目录
            temp_dir = f"{self.model_dir}_temp"
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)

            # 复制备份到临时目录
            shutil.copytree(backup_path, temp_dir)

            # 备份当前模型
            current_backup = self.create_backup(f"pre_restore_{datetime.now().strftime('%Y%m%d_%H%M%S')}")

            # 替换当前模型
            if os.path.exists(self.model_dir):
                shutil.rmtree(self.model_dir)
            shutil.move(temp_dir, self.model_dir)

            return True

        except Exception as e:
            # 恢复失败时的清理
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
            raise e

    def list_backups(self) -> List[Dict]:
        """列出所有备份"""
        backups = []

        for backup_name in os.listdir(self.backup_dir):
            backup_path = os.path.join(self.backup_dir, backup_name)
            metadata_path = os.path.join(backup_path, "backup_metadata.json")

            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                    backups.append(metadata)

        return sorted(backups, key=lambda x: x['timestamp'], reverse=True)

    def cleanup_old_backups(self, keep_count: int = 10):
        """清理旧备份"""
        backups = self.list_backups()

        for backup in backups[keep_count:]:
            backup_path = backup['backup_path']
            if os.path.exists(backup_path):
                shutil.rmtree(backup_path)

    def _get_file_list(self, directory: str) -> List[str]:
        """获取目录中的文件列表"""
        file_list = []
        for root, dirs, files in os.walk(directory):
            for file in files:
                file_path = os.path.join(root, file)
                relative_path = os.path.relpath(file_path, directory)
                file_list.append(relative_path)
        return file_list
```

### 6.2 健康检查和自动恢复

```python
import asyncio
import aiohttp
from typing import List, Dict, Any

class HealthChecker:
    def __init__(self, service_urls: List[str], backup_manager: BackupManager):
        self.service_urls = service_urls
        self.backup_manager = backup_manager
        self.healthy_services = set(service_urls)

    async def check_service_health(self, service_url: str) -> bool:
        """检查服务健康状态"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{service_url}/health") as response:
                    if response.status == 200:
                        health_data = await response.json()
                        return health_data.get('status') == 'healthy'
                    return False
        except:
            return False

    async def run_health_checks(self):
        """运行健康检查"""
        tasks = []

        for service_url in self.service_urls:
            task = asyncio.create_task(self._check_and_recover(service_url))
            tasks.append(task)

        await asyncio.gather(*tasks)

    async def _check_and_recover(self, service_url: str):
        """检查并恢复服务"""
        is_healthy = await self.check_service_health(service_url)

        if not is_healthy:
            print(f"Service {service_url} is unhealthy, attempting recovery...")

            # 从备份恢复
            try:
                recovery_success = await self._attempt_recovery(service_url)

                if recovery_success:
                    print(f"Service {service_url} recovered successfully")
                else:
                    print(f"Failed to recover service {service_url}")

            except Exception as e:
                print(f"Recovery failed for service {service_url}: {e}")

        else:
            if service_url in self.healthy_services:
                self.healthy_services.remove(service_url)

    async def _attempt_recovery(self, service_url: str) -> bool:
        """尝试恢复服务"""
        # 获取最新的备份
        backups = self.backup_manager.list_backups()
        if not backups:
            return False

        latest_backup = backups[0]

        # 这里应该调用实际的恢复API
        # 实际实现取决于你的服务架构
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{service_url}/restore",
                    json={"backup_name": latest_backup['backup_name']}
                ) as response:
                    return response.status == 200

        except:
            return False

    async def start_monitoring(self, check_interval: int = 30):
        """启动监控"""
        while True:
            await self.run_health_checks()
            await asyncio.sleep(check_interval)
```

## 7. 部署策略

### 7.1 蓝绿部署

```python
class BlueGreenDeployment:
    def __init__(self, blue_config: Dict, green_config: Dict):
        self.blue_config = blue_config
        self.green_config = green_config
        self.current_active = "blue"
        self.load_balancer = LoadBalancer([])

    def deploy_new_version(self, new_model_path: str):
        """部署新版本"""
        # 确定目标环境
        target_env = "green" if self.current_active == "blue" else "blue"

        # 停止目标环境
        self._stop_environment(target_env)

        # 部署新模型
        self._deploy_model(target_env, new_model_path)

        # 健康检查
        if self._health_check(target_env):
            # 切换流量
            self._switch_traffic(target_env)
            self.current_active = target_env
            return True
        else:
            # 回滚
            self._stop_environment(target_env)
            return False

    def _stop_environment(self, env: str):
        """停止指定环境"""
        # 实现停止逻辑
        pass

    def _deploy_model(self, env: str, model_path: str):
        """部署模型到指定环境"""
        # 实现部署逻辑
        pass

    def _health_check(self, env: str) -> bool:
        """健康检查"""
        # 实现健康检查逻辑
        return True

    def _switch_traffic(self, target_env: str):
        """切换流量"""
        # 更新负载均衡器配置
        pass
```

### 7.2 金丝雀部署

```python
class CanaryDeployment:
    def __init__(self, production_config: Dict, canary_config: Dict):
        self.production_config = production_config
        self.canary_config = canary_config
        self.canary_percentage = 0.0

    def start_canary(self, new_model_path: str, initial_percentage: float = 0.1):
        """启动金丝雀部署"""
        self.canary_percentage = initial_percentage

        # 部署金丝雀版本
        self._deploy_canary(new_model_path)

        # 开始监控
        self._start_monitoring()

    def increase_canary_traffic(self, increment: float = 0.1):
        """增加金丝雀流量"""
        new_percentage = min(self.canary_percentage + increment, 1.0)

        if self._check_canary_health():
            self.canary_percentage = new_percentage
            self._update_traffic_split()
            return True
        else:
            return False

    def promote_to_production(self):
        """将金丝雀版本提升为生产版本"""
        if self.canary_percentage >= 1.0 and self._check_canary_health():
            self._promote_canary()
            return True
        return False

    def rollback_canary(self):
        """回滚金丝雀版本"""
        self.canary_percentage = 0.0
        self._update_traffic_split()
        self._stop_canary()

    def _deploy_canary(self, model_path: str):
        """部署金丝雀版本"""
        pass

    def _start_monitoring(self):
        """开始监控"""
        pass

    def _check_canary_health(self) -> bool:
        """检查金丝雀健康状态"""
        return True

    def _update_traffic_split(self):
        """更新流量分割"""
        pass

    def _promote_canary(self):
        """提升金丝雀版本"""
        pass

    def _stop_canary(self):
        """停止金丝雀版本"""
        pass
```

## 总结

## 生产环境部署是一个复杂的系统工程，需要考虑性能、可靠性、安全性和可维护性等多个方面。通过合理的架构设计、性能优化、监控告警和部署策略，可以构建高可用、高性能的Transformer模型服务。Transformers库提供了丰富的工具和接口，结合现代化的部署技术，可以满足各种生产环境的需求。