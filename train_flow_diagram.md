# train.py 流程框架图

## 执行流程

```mermaid
flowchart TD
    A["🚀 入口: main"] --> B{命令行参数?}
    B -->|--preprocess| C["数据预处理"]
    B -->|默认| D["开始训练"]
    B -->|--help| E["显示帮助"]
    
    C --> C1["加载Processor/Tokenizer"]
    C1 --> C2["遍历JSON数据"]
    C2 --> C3["调用process_func<br/>- 构建消息<br/>- 提取vision信息<br/>- Tokenize响应"]
    C3 --> C4["保存.pt文件<br/>包含input_ids/labels/pixel_values"]
    C4 --> C5["生成index.json<br/>记录样本映射"]
    C5 --> C6["✅ 预处理完成"]
    
    D --> D1["加载TrainConfig"]
    D1 --> D2["加载Processor/Tokenizer/Model"]
    D2 --> D3["冻结Vision Tower"]
    D3 --> D4["应用LoRA"]
    D4 --> D5["加载VideoDataset<br/>从缓存读取.pt"]
    D5 --> D6["创建Trainer<br/>- DataCollator<br/>- TrainingArgs<br/>- Callbacks"]
    D6 --> D7["🏃 trainer.train"]
    D7 --> D8["监控/评估/保存"]
    D8 --> D9{完成?}
    D9 -->|否| D7
    D9 -->|是| D10["✅ 训练完成"]
    
    E --> E1["打印使用说明"]
```

## 关键组件

| 组件 | 功能 |
|------|------|
| `TrainConfig` | 集中管理所有超参数 |
| `process_func` | 单个样本处理逻辑 |
| `QwenVideoDataCollator` | 批处理数据对齐 |
| `VideoDataset` | 从.pt文件加载数据 |
| `TrainingMonitorCallback` | 监控训练进度/显存 |
| `Trainer` | HuggingFace训练器 |

## 训练流程细节

```
加载模型
  ↓
冻结Vision (可选)
  ↓
应用LoRA (可选)
  ↓
加载数据集 ──→ Data Collator (padding/对齐)
  ↓
初始化Trainer
  ↓
[每个step]
  ├─ 前向传播 (input_ids → logits)
  ├─ 计算loss (只在labels != -100处)
  ├─ 反向传播
  ├─ 梯度积累 (8步后更新)
  └─ 日志/评估/保存
  ↓
训练完成 → 保存模型
```
