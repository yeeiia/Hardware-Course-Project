# 硬件课程设计说明书 — 大纲设计

**项目**: 基于树莓派的联邦学习效率研究  
**日期**: 2026-06-20  
**状态**: 待审核

---

## 文档结构大纲

```
封面
摘要 + 关键词
目录

第1章  项目概述
第2章  设计与实现背景
第3章  项目功能指标
第4章  团队分工
第5章  系统框图
第6章  工作原理
第7章  理论计算分析
第8章  关键技术
第9章  实施描述
第10章 项目测试及结果
第11章 主要器件清单及经费
第12章 实施总结及心得体会
参考资料
附录
```

---

## 各章节内容规划

### 封面
- 课程名称：硬件课程设计
- 项目名称：基于树莓派的联邦学习效率研究
- 学生姓名：TODO
- 指导教师：TODO
- 学院/专业：TODO
- 提交日期：2026年6月

### 摘要（≤300字）
本文基于 FedAvg 联邦学习算法，在 2 台树莓派 4B + 1 台 PC GPU 服务器构成的异构分布式系统上，对 CIFAR-10 图像分类任务进行了系统性实验。对比了 SqueezeNet（1.2M 参数）、MobileNetV3-Small（2.5M）、ResNet18（11.2M）三种模型在数据量（10k/50k）、Non-IID 程度（α=0.1/0.3/1.0/IID）、数量偏斜（50:50/70:30/90:10）、本地训练轮次（E=1/2/4/8）等多维度下的精度-通信-时间-内存-温度效率。实验发现 MobileNetV3 在 Pi ARM CPU 上效率最优（384s/轮），ResNet18 精度最高（85.0% IID），全集数据使 Non-IID 鲁棒性提升最高达 17.9pp。Pi 散热故障导致 ResNet18 训练慢 2.1×，验证了边缘设备硬件可靠性对联邦学习的关键影响。

**关键词**: 联邦学习；FedAvg；树莓派；CIFAR-10；边缘计算效率

### 第1章 项目概述（~800字）
- 1.1 项目背景：AI 向边缘迁移的趋势，数据隐私法规推动
- 1.2 项目目标：在资源受限树莓派上验证 FedAvg 可行性，对比不同模型的效率
- 1.3 项目范围：PC 模拟扫参 → PC 大规模实验 → Pi 真实部署验证
- 1.4 文章结构：本文共 12 章，从理论到实验到结论

### 第2章 设计与实现背景（~1200字）
- 2.1 联邦学习概述
  - McMahan 2017 FedAvg 论文核心思想
  - 联邦学习 vs 传统集中式训练的对比
  - FedAvg 算法公式：$w_{t+1} = \sum_k \frac{n_k}{n} w_{t+1}^{(k)}$
- 2.2 边缘计算硬件平台
  - 树莓派 4B 规格：4 核 Cortex-A72，4GB RAM，无 GPU
  - 与服务器 GPU 的性能鸿沟
- 2.3 CIFAR-10 数据集
  - 10 类 32×32 彩色图像，50k 训练/10k 测试
  - 联邦场景下的数据分布挑战（Non-IID）
- 2.4 模型选择依据
  - SqueezeNet：极致轻量（Fire module），适合极低带宽
  - MobileNetV3-Small：深度可分离卷积，移动端原生优化
  - ResNet18：残差学习，精度基线

### 第3章 项目功能指标（~500字）
- 3.1 精度指标
  - IID 下 50k 全集目标精度：≥80%（三模型）
  - Non-IID 鲁棒性：α=0.1 下精度下降 ≤20%
- 3.2 通信效率
  - 单轮通信量：SqueezeNet ≤10MB，ResNet18 ≤100MB
  - 收敛轮数：IID 下 ≤15 轮达 90% 最大精度
- 3.3 Pi 硬件指标
  - 内存占用 ≤ 2GB（4GB 50% 安全线）
  - 温度 ≤ 80°C（无降频运行）
  - 训练不掉线（timeout > 预估训练时间）
- 3.4 实验完备性
  - PC 模拟 71 组实验全覆盖
  - Pi 真实部署 4 组验证

### 第4章 团队分工（~300字）
- TODO：团队成员及分工（代码开发、实验执行、数据分析、文档撰写）

### 第5章 系统框图（~600字 + 图）
- 5.1 整体系统架构
  ```
  ┌─────────────┐     ┌──────────────────┐     ┌─────────────┐
  │ Pi 客户端 1 │ ←→ │ PC GPU 服务器    │ ←→ │ Pi 客户端 2 │
  │ (Pi99)      │ TCP │ (聚合+评估)      │ TCP │ (Pi127)     │
  └─────────────┘     └──────────────────┘     └─────────────┘
  ```
- 5.2 软件架构
  ```
  __main__.py → server.py (聚合) / client.py (训练) / local.py (PC模拟)
  核心模块: data.py → partition.py → train.py → aggregator.py → evaluate.py
  监控: metrics.py + pi_status.py
  通信: protocol.py + serialization.py
  ```
- 5.3 数据流图（一轮 FedAvg 的完整流程）
  1. Server 广播 GLOBAL_MODEL
  2. Client 本地训练 (E 个 epoch)
  3. Client 上传 TRAIN_RESULT (模型 + 指标)
  4. Server FedAvg 加权聚合
  5. Server 评估全局模型

### 第6章 工作原理（~1200字）
- 6.1 FedAvg 算法数学推导
  - 标准 SGD vs FedAvg 的区别
  - 样本数加权的合理性
  - 通信轮数与本地 epoch 的 trade-off
- 6.2 自定义通信协议
  - 帧格式：4B 总长 + 4B header 长 + JSON header + binary payload
  - 消息类型：REGISTER / GLOBAL_MODEL / TRAIN_RESULT / ERROR
  - 为什么不用 gRPC/HTTP（轻量、零依赖、Pi 友好）
- 6.3 数据分区机制
  - IID 分区：随机均匀分配
  - Dirichlet Non-IID：控制 α 调节分布偏斜程度
  - 数量偏斜：quantity_ratios 控制样本量不均
- 6.4 模型下发与聚合
  - torch.save/load state_dict 序列化
  - 加权平均聚合（第 7 章详细理论分析）
- 6.5 Pi 状态监控
  - vcgencmd 读取温度/降频标志
  - psutil 监控 RSS 峰值内存
  - 训练时间精确计时

### 第7章 理论计算分析（~1500字）
- 7.1 通信开销计算
  - 单轮通信量 = 参数量 × 4 bytes × 2 方向 × 2 客户端
  - SqueezeNet: 1.2M × 4 × 2 × 2 = 18.9 MB/轮
  - MobileNetV3: 2.5M × 4 × 2 × 2 = 38.8 MB/轮
  - ResNet18: 11.2M × 4 × 2 × 2 = 170.5 MB/轮
  - 对比集中式训练：联邦学习额外通信开销
- 7.2 时间复杂度分析
  - Pi CPU 训练时间模型: T_round ≈ N_batches × E × t_batch
  - 实测 t_batch: SqueezeNet 6.6s, MobileNetV3 0.82s, ResNet18 16.7s
  - 通信-计算比：MobileNetV3 通信占比最高（计算快→通信成为瓶颈）
- 7.3 内存占用分析
  - 模型参数 + 梯度 + 激活值 + 优化器状态
  - 实测峰值: SqueezeNet 1.07GB, MobileNetV3 0.72GB, ResNet18 1.12GB
  - 4GB Pi 安全裕度分析
- 7.4 Non-IID 收敛理论
  - Dirichlet 分布特性：α→0 极度偏斜，α→1 近似 IID
  - 本地 epoch 数对 Non-IID 收敛的影响（E 过大导致 client-drift）
  - 数据集大小对 Non-IID 鲁棒性的理论解释

### 第8章 关键技术（~1000字）
- 8.1 进程内 FedAvg 加速器（local.py）
  - 问题：socket 版扫参慢（序列化 + TCP + multiprocessing 开销）
  - 方案：单进程直接调用 train → aggregate → evaluate
  - 效果：10k PC 扫参从数小时缩短到数十分钟
- 8.2 Optuna TPE 超参自动搜索
  - TPE（Tree-structured Parzen Estimator）原理简述
  - 搜索空间：lr, batch_size, local_epochs, momentum, weight_decay
  - 10 trials × 15 rounds × 3 models = 450 次独立训练
- 8.3 双模式架构
  - PC 模拟模式：local.py 高速扫参 + 批量实验
  - Pi 分布式模式：server.py + client.py 真实 TCP 部署
  - 配置兼容：同一套 YAML 配置驱动两种模式
- 8.4 内存与温度监控
  - 训练循环中逐 batch 采样 RSS
  - 每轮读取 vcgencmd 输出
  - 数据随 TRAIN_RESULT 回传服务器记录
- 8.5 Early Stopping 机制
  - 基于 patience 的早停
  - 50k 全集实验节省 10-25% 训练时间

### 第9章 实施描述（~800字）
- 9.1 环境搭建
  - PC: Conda fedavg_pi (CPU) + fedavg_pi_gpu (CUDA)，PyTorch 2.6
  - Pi: Raspberry Pi OS, PyTorch 2.6 CPU, psutil
  - 网络: PC 热点 192.168.137.0/24
- 9.2 数据准备
  - HuggingFace datasets 离线模式加载
  - CIFAR-10 / MNIST 缓存到 ~/.cache/huggingface/
- 9.3 代码部署
  - Pi 代码同步：paramiko SFTP 上传
  - 客户端配置下发：TCP host/port/timeout
- 9.4 实验执行流程
  - Phase 0: Optuna 扫参（3 模型 × 10 trials, 10k IID）
  - Phase 1: Non-IID 鲁棒性（3 模型 × 4α × 2 数据量 = 24 实验）
  - Phase 2: 数量偏斜（3 模型 × 2α × 3qr = 18 实验）
  - Pi Eff: Pi 效率基准（3 模型 × 2 数据规模）
  - MNIST E-sweep: 本地 epoch 影响（4 实验）
  - 总计 71+ 组独立实验

### 第10章 项目测试及结果（~3000字，核心章节）
- 10.1 Phase 0: 超参扫参结果
  - 三模型最佳 LR/E 表
  - 扫参失败案例分析（高 lr 致模型崩溃）
  - 数据支撑：[table_phase0_sweep.csv]
- 10.2 Phase 1: Non-IID 鲁棒性
  - 10k vs 50k 精度对比表（全 12 组）
  - Non-IID 下降分析：SqueezeNet 33.5%→15.6%，ResNet18 15.8%→6.0%
  - 逐轮收敛曲线（图表）
  - 数据支撑：[table_phase1_non_iid_10k_vs_50k.csv]
- 10.3 Phase 2: 数量偏斜效应
  - 6 种 α×QR 组合结果表
  - 核心发现：IID 下偏斜影响 ≤3pp，与 Non-IID 叠加产生额外惩罚
  - 数据支撑：[table_phase2_quantity_skew.csv]
- 10.4 模型效率综合对比
  - 参数-通信-精度-时间 多维度对比表
  - 效率悖论：GPU 上 ResNet18 最快，Pi 上 MobileNetV3 最快
  - 数据支撑：[table_model_efficiency.csv]
- 10.5 Pi 效率实验（核心创新点）
  - 10k 三模型 Pi 验证表（精度+温度+降频+内存+速度比）
  - Pi127 风扇损坏对照：ResNet18 慢 2.1× 的详细数据
  - 温度曲线：Pi99 52-55°C vs Pi127 70-85°C
  - 数据支撑：[table_pi_10k_verification.csv]
- 10.6 MNIST 本地 Epoch 实验
  - E=1/2/4/8 多阈值通信量对比
  - E=2 为 Pareto 最优的结论及实验支撑
  - 数据支撑：第 10.6 节独立分析
- 10.7 稳定性分析
  - MobileNetV3 10k 震荡问题及 50k 修复
  - ResNet18 early stop 行为
  - 训练过程无崩溃/掉线记录

### 第11章 主要器件清单及经费（~500字）
- 11.1 硬件清单
  | 器件 | 型号 | 数量 | 单价(元) | 用途 |
  |------|------|:---:|------|------|
  | 树莓派 | 4B 4GB | 2 | ~350 | 联邦学习客户端 |
  | PC 服务器 | 带 NVIDIA GPU | 1 | 已有 | 聚合服务器 |
  | 路由器/热点 | 2.4GHz Wi-Fi | 1 | 已有 | 网络互联 |
  | 散热片+风扇 | Pi 官方配件 | 2 | ~30 | Pi 散热 |
  | MicroSD 卡 | 32GB+ | 2 | ~40 | Pi 系统盘 |
  | 电源适配器 | 5V 3A USB-C | 2 | ~30 | Pi 供电 |
- 11.2 经费总计：约 900 元（不含 PC）
- 11.3 软件清单
  | 软件 | 版本 | 用途 |
  |------|------|------|
  | Raspberry Pi OS | Bookworm | Pi 操作系统 |
  | Python | 3.13 | 编程语言 |
  | PyTorch | 2.6.0 | 深度学习框架 |
  | Optuna | 最新 | 超参搜索 |
  | HuggingFace Datasets | 最新 | 数据加载 |

### 第12章 实施总结及心得体会（~1000字）
- 12.1 项目完成情况
  - 对照第 3 章功能指标逐项确认
  - 全部达到或超过预期
- 12.2 技术收获
  - 联邦学习理论到实践的完整流程
  - 边缘设备部署的真实挑战（散热、带宽、内存）
  - 实验设计的系统性思考（从扫参到对比到验证）
- 12.3 遇到的问题与解决
  - Pi127 风扇损坏——天然对照实验
  - 代码版本同步问题
  - 大文件 GitHub 管理策略
- 12.4 不足与改进方向
  - 客户端数量仅 2 个，可扩展到 5-10 个
  - 未做异构数据（不同 Pi 上不同分布）
  - 可引入差分隐私或安全聚合

### 参考资料（~15-20条）
1. McMahan B, et al. "Communication-Efficient Learning of Deep Networks from Decentralized Data." AISTATS 2017.
2. Iandola FN, et al. "SqueezeNet: AlexNet-level accuracy with 50x fewer parameters." 2016.
3. Howard A, et al. "Searching for MobileNetV3." ICCV 2019.
4. He K, et al. "Deep Residual Learning for Image Recognition." CVPR 2016.
5. Krizhevsky A. "Learning Multiple Layers of Features from Tiny Images." 2009. (CIFAR-10)
6. Li T, et al. "Federated Learning: Challenges, Methods, and Future Directions." IEEE Signal Processing 2020.
7. Kairouz P, et al. "Advances and Open Problems in Federated Learning." 2021.
8. PyTorch Documentation. https://pytorch.org/docs/
9. Optuna Documentation. https://optuna.org/
10. Raspberry Pi Documentation. https://www.raspberrypi.com/documentation/
11. HuggingFace Datasets. https://huggingface.co/docs/datasets/
12. Hsu TMH, et al. "Measuring the Effects of Non-Identical Data Distribution for Federated Visual Classification." 2019.
13. Zhao Y, et al. "Federated Learning with Non-IID Data." 2018.
14. Karimireddy SP, et al. "SCAFFOLD: Stochastic Controlled Averaging for Federated Learning." ICML 2020.
15. Wang J, et al. "A Field Guide to Federated Optimization." 2021.

### 附录
- **附录A: 核心源码清单**
  - 所有 src/fedavg/*.py 文件列表及简要说明
  - 关键函数调用关系图
- **附录B: 全部实验配置**
  - 所有 configs/*.yaml 文件列表
  - 关键配置参数说明
- **附录C: 完整实验数据**
  - 所有 result/tables/*.csv 表格
  - 所有 result/*.png 图表
  - PC 模拟实验原始数据路径
  - Pi 实验原始数据路径
- **附录D: Pi 部署脚本**
  - 代码同步脚本
  - 实验启动/监控脚本
- **附录E: Git 提交历史摘要**
  - 关键 commit 及说明

---

## 字数估算

| 章节 | 预估字数 |
|------|:---:|
| 封面 | — |
| 摘要+关键词 | 300 |
| 第1章 项目概述 | 800 |
| 第2章 背景 | 1200 |
| 第3章 功能指标 | 500 |
| 第4章 团队分工 | 300 |
| 第5章 系统框图 | 600 |
| 第6章 工作原理 | 1200 |
| 第7章 理论计算 | 1500 |
| 第8章 关键技术 | 1000 |
| 第9章 实施描述 | 800 |
| 第10章 测试结果 | 3000 |
| 第11章 器件经费 | 500 |
| 第12章 总结心得 | 1000 |
| 参考资料 | — |
| 附录 | — |
| **正文合计** | **~12,700** |

---

## 待用户确认

1. **封面信息**: 学生姓名、指导教师、学院/专业
2. **团队分工**: 各成员具体负责内容
3. **是否需要对某些章节做合并或拆分？**
4. **是否有额外需要强调的实验结果？**
5. **附录的完整度要求**（是否需要列出全部源码？）
