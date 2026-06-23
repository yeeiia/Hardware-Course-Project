# FedAvg CIFAR-10 联邦学习 — 硬件课程设计

基于 PyTorch 的 FedAvg 联邦学习框架，支持 PC 本地模拟和树莓派分布式训练。在 CIFAR-10 上对比 SqueezeNet / MobileNetV3 / ResNet18 三模型在资源受限场景下的训练效率。

## 项目结构

```
├── src/fedavg/           # 核心代码
│   ├── server.py         # TCP 服务器 (PC GPU 聚合)
│   ├── client.py         # TCP 客户端 (Pi / PC CPU 训练)
│   ├── local.py          # 进程内 FedAvg (PC 扫参加速)
│   ├── run_experiments.py # 批量实验编排 (Phase 1/2)
│   ├── optuna_sweep.py   # Optuna TPE 超参扫参
│   ├── models.py         # 模型工厂 (TinyCNN + torchvision wrappers)
│   ├── train.py          # 本地训练 + psutil 内存监控
│   ├── metrics.py        # RunLogger: CSV/JSONL/checkpoint/曲线图
│   ├── data.py           # HF datasets 数据加载
│   ├── partition.py      # IID / Dirichlet Non-IID 分区
│   ├── aggregator.py     # 样本数加权 FedAvg 聚合
│   ├── evaluate.py       # 全局模型评估
│   ├── protocol.py       # 自定义 TCP 帧协议
│   └── serialization.py  # state_dict 序列化
├── configs/              # 实验配置文件
├── result/               # 最终结果 (CSV 表格 + PNG 图表)
├── runs/                 # 实验原始输出 (metrics + checkpoints + figures)
├── docs/                 # 文档
│   └── final_report.md   # 完整实验报告
├── analyze_full_results.py   # 10k vs 50k 全量分析
├── analyze_results.py        # Phase 0/1/2 分析
├── final_report.py           # 终端报告生成
├── generate_report_package.py # CSV + 图表打包
├── generate_all_tables.py    # 全部实验汇总表
└── README.md
```

## 环境配置

### Python 环境

| 环境名 | Python | PyTorch | 用途 |
|--------|--------|---------|------|
| `fedavg_pi` | 3.13 | 2.6.0 CPU | Pi 客户端 / PC CPU 模拟 / 数据分析 |
| `fedavg_pi_gpu` | 3.13 | 2.6.0 CUDA | PC GPU 服务器 / GPU 扫参 |

### 环境变量

```bash
# 所有数据加载需离线模式 (绕过 HF Hub, 从缓存秒加载)
export HF_DATASETS_OFFLINE=1

# Python 模块路径
export PYTHONPATH="$(pwd)/src"
```

### Python 解释器路径

```bash
# CPU 环境
PY_CPU="/c/Users/haotian/.conda/envs/fedavg_pi/python.exe"

# GPU 环境
PY_GPU="/c/Users/haotian/.conda/envs/fedavg_pi_gpu/python.exe"
```

---

## 实验启动指令

所有命令在 Git Bash 中运行，工作目录为项目根目录。

### Phase 0 — Optuna 超参扫参

PC 本地进程内 FedAvg，10k 子集，IID，15 rounds。

```bash
cd "d:/Study/working/exp/hardware_course_projection/Hardware-Course-Project"
export HF_DATASETS_OFFLINE=1
export PYTHONPATH="$(pwd)/src"

# 单模型扫参 (10 trials, 输出到 sweeps_gpu/)
"/c/Users/haotian/.conda/envs/fedavg_pi_gpu/python.exe" -m fedavg optuna \
  --model squeezenet_cifar --study-name sweep-squeezenet-cifar10-10k \
  --trials 10 --rounds 15 --train-limit 10000 --test-limit 2000 \
  --output-dir sweeps_gpu --device cuda

# MobileNetV3
"/c/Users/haotian/.conda/envs/fedavg_pi_gpu/python.exe" -m fedavg optuna \
  --model mobilenetv3_cifar --study-name sweep-mobilenetv3-cifar10-10k \
  --trials 10 --rounds 15 --train-limit 10000 --test-limit 2000 \
  --output-dir sweeps_gpu --device cuda

# ResNet18
"/c/Users/haotian/.conda/envs/fedavg_pi_gpu/python.exe" -m fedavg optuna \
  --model resnet18_cifar --study-name sweep-resnet18-cifar10-10k \
  --trials 10 --rounds 15 --train-limit 10000 --test-limit 2000 \
  --output-dir sweeps_gpu --device cuda
```

扫参完成后，最佳超参自动保存到 `sweeps_gpu/sweep-<model>-cifar10-10k-best.yaml`。

### Phase 1 — Non-IID 鲁棒性 (PC 模拟)

批量运行 3 模型 × 4 α 水平 (IID / 1.0 / 0.3 / 0.1)，自动读取 Phase 0 最佳超参。

```bash
cd "d:/Study/working/exp/hardware_course_projection/Hardware-Course-Project"
export HF_DATASETS_OFFLINE=1
export PYTHONPATH="$(pwd)/src"

# === 10k 子集 (15 rounds) ===
"/c/Users/haotian/.conda/envs/fedavg_pi_gpu/python.exe" -m fedavg run-experiments \
  --phase 1 --rounds 15 --train-limit 10000 --test-limit 2000 \
  --data-dir dataset_cifar10 --output-dir experiments --best-config-dir sweeps_gpu \
  --device cuda --early-stop-patience 5

# === 50k 全集 (20 rounds, 分两批并行跑以节省时间) ===

# 第一批: α=iid + α=0.1
"/c/Users/haotian/.conda/envs/fedavg_pi_gpu/python.exe" -m fedavg run-experiments \
  --phase 1 --alpha iid 0.1 --rounds 20 --train-limit 50000 --test-limit 10000 \
  --data-dir dataset_cifar10 --output-dir experiments_full --best-config-dir sweeps_gpu \
  --device cuda --early-stop-patience 5

# 第二批: α=1.0 + α=0.3
"/c/Users/haotian/.conda/envs/fedavg_pi_gpu/python.exe" -m fedavg run-experiments \
  --phase 1 --alpha 1.0 0.3 --rounds 20 --train-limit 50000 --test-limit 10000 \
  --data-dir dataset_cifar10 --output-dir experiments_full_b2 --best-config-dir sweeps_gpu \
  --device cuda --early-stop-patience 5
```

### Phase 2 — 数量偏斜 (PC 模拟)

3 模型 × 2 α (IID / 0.1) × 3 数量比 (50:50 / 70:30 / 90:10)。

```bash
cd "d:/Study/working/exp/hardware_course_projection/Hardware-Course-Project"
export HF_DATASETS_OFFLINE=1
export PYTHONPATH="$(pwd)/src"

"/c/Users/haotian/.conda/envs/fedavg_pi_gpu/python.exe" -m fedavg run-experiments \
  --phase 2 --rounds 15 --train-limit 10000 --test-limit 2000 \
  --data-dir dataset_cifar10 --output-dir experiments --best-config-dir sweeps_gpu \
  --device cuda --early-stop-patience 5
```

### Pi 效率实验 — 树莓派分布式训练

**前提**: 树莓派已部署代码且 CIFAR-10 已缓存，Pi 客户端配置 `timeout_seconds: 14400`。

**架构**: 1 个 PC GPU 服务器 (聚合 + 评估) + 2 个 Pi 客户端 (Pi99 + Pi127)。

#### Pi Eff 基准 (2k 样本, 5 rounds, SqueezeNet)

```bash
# Terminal A — PC 服务器 (先启动)
cd "d:/Study/working/exp/hardware_course_projection/Hardware-Course-Project"
export HF_DATASETS_OFFLINE=1
export PYTHONPATH="$(pwd)/src"
"/c/Users/haotian/.conda/envs/fedavg_pi_gpu/python.exe" -m fedavg server \
  --config configs/pi-eff-squeezenet.yaml

# Terminal B — 启动两个 Pi 客户端 (看到 "listening" 后执行)
"/c/Users/haotian/.conda/envs/fedavg_pi/python.exe" -c "
import paramiko
for ip, cid, idx in [('192.168.137.99','pi99','0'),('192.168.137.127','pi127','1')]:
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(ip, username='pi', password=os.environ['PI_PASSWORD'])
    c.exec_command('cd /home/pi/fedavg_resnet18 && HF_DATASETS_OFFLINE=1 PYTHONPATH=src nohup python3 -m fedavg client --config configs/pi_client.yaml --client-id {} --client-index {} > /tmp/client.log 2>&1 &'.format(cid, idx))
    print(f'{cid} started')
    c.close()
"
```

#### Pi 10k 验证 (10k 样本, 3 rounds, 三模型顺序跑)

```bash
# 三个模型依次运行, 每次替换 --config:

# 1. SqueezeNet (~4.2h)
"/c/Users/haotian/.conda/envs/fedavg_pi_gpu/python.exe" -m fedavg server \
  --config configs/pi-10k-squeezenet.yaml

# 2. MobileNetV3 (~2.8h)
"/c/Users/haotian/.conda/envs/fedavg_pi_gpu/python.exe" -m fedavg server \
  --config configs/pi-10k-mobilenetv3.yaml

# 3. ResNet18 (~4.5h)
"/c/Users/haotian/.conda/envs/fedavg_pi_gpu/python.exe" -m fedavg server \
  --config configs/pi-10k-resnet18.yaml
```

每次 Server 启动看到 `listening` 后，在 Terminal B 执行启动 Pi 客户端的命令。

#### Pi 监控命令

```bash
# 查看 Pi 客户端日志
"/c/Users/haotian/.conda/envs/fedavg_pi/python.exe" -c "
import paramiko
c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect('192.168.137.99', username='pi', password=os.environ['PI_PASSWORD'])
_, out, _ = c.exec_command('tail -5 /tmp/client.log')
print('Pi99:', out.read().decode())
c.close()
"

# 查看 CPU 温度
"/c/Users/haotian/.conda/envs/fedavg_pi/python.exe" -c "
import paramiko
for ip in ['192.168.137.99','192.168.137.127']:
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(ip, username='pi', password=os.environ['PI_PASSWORD'])
    _, out, _ = c.exec_command('vcgencmd measure_temp')
    print(f'{ip}: {out.read().decode().strip()}')
    c.close()
"

# 杀 Pi 上的旧进程
"/c/Users/haotian/.conda/envs/fedavg_pi/python.exe" -c "
import paramiko
for ip in ['192.168.137.99','192.168.137.127']:
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(ip, username='pi', password=os.environ['PI_PASSWORD'])
    c.exec_command('pkill -f fedavg')
    print(f'{ip}: killed')
    c.close()
"
```

### Pi 代码部署

```bash
# 上传更新文件到两个 Pi
cd "d:/Study/working/exp/hardware_course_projection/Hardware-Course-Project"
"/c/Users/haotian/.conda/envs/fedavg_pi/python.exe" -c "
import paramiko, io

files = ['src/fedavg/train.py','src/fedavg/client.py','src/fedavg/server.py','src/fedavg/metrics.py','src/fedavg/models.py']
config = 'device: cpu\nserver:\n  host: 192.168.137.1\n  port: 9000\n  timeout_seconds: 14400\n'

for ip, name in [('192.168.137.99','Pi99'),('192.168.137.127','Pi127')]:
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(ip, username='pi', password=os.environ['PI_PASSWORD'])
    sftp = c.open_sftp()
    for f in files:
        sftp.put(f, f'/home/pi/fedavg_resnet18/{f}')
    sftp.putfo(io.BytesIO(config.encode()), '/home/pi/fedavg_resnet18/configs/pi_client.yaml')
    sftp.close()
    print(f'{name} deployed')
    c.close()
"
```

---

## 数据分析

### 生成分析报告

```bash
cd "d:/Study/working/exp/hardware_course_projection/Hardware-Course-Project"
export PYTHONPATH="$(pwd)/src"

# 终端完整报告
"/c/Users/haotian/.conda/envs/fedavg_pi/python.exe" final_report.py

# 10k vs 50k 对比分析
"/c/Users/haotian/.conda/envs/fedavg_pi/python.exe" analyze_full_results.py

# 生成 result/ 下的 CSV 表格 + PNG 图表
"/c/Users/haotian/.conda/envs/fedavg_pi/python.exe" generate_report_package.py

# 生成全部实验汇总表 (result/tables/)
"/c/Users/haotian/.conda/envs/fedavg_pi/python.exe" generate_all_tables.py
```

### 查看实验结果

```bash
# 查看某次实验的精度
grep ",eval," runs/pi-10k-resnet18/metrics.csv

# 查看 Pi 温度数据
grep ",train," runs/pi-10k-resnet18/metrics.csv | grep "pi_temp"
```

---

## 通信协议

自定义 TCP 帧格式：

1. 4 字节 big-endian 总帧长
2. 4 字节 big-endian JSON header 长度
3. JSON header (`type`, `metadata`, `payload_size`)
4. 二进制 payload (`torch.save(state_dict)`)

消息类型: `REGISTER`, `GLOBAL_MODEL`, `TRAIN_RESULT`, `ERROR`

---

## 运行输出

每个实验目录包含：

- `config.yaml` — 完整实验配置
- `metrics.csv` / `metrics.jsonl` — 每轮指标
- `checkpoints/` — 全局模型 checkpoint
- `figures/` — loss + accuracy 曲线图

指标字段: `round`, `phase`, `dataset`, `model`, `split`, `B`, `E`, `client_id`, `train_loss`, `global_loss`, `accuracy`, `macro_f1`, `train_time`, `eval_time`, `bytes_sent`, `bytes_recv`, `samples`, `peak_memory_mb`, `status`, `pi_temp`, `pi_throttled`
