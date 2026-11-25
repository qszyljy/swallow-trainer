# 吞咽事件分类 · PC 端训练内核

配套手机 App 用的离线训练端：**输入录音数据 → 输出可直接接进 App 的模型包**。
和 App 的分工是「离线训练 / 在线推理」，模型包是两边唯一的接口。

设计约束（来自任务书，代码里都兑现了）：

- 窗长 **1.0 s**、步长 **0.25 s**，训练和手机端推理用同一套参数
- 评估只用 **LOSO（留一被试）**，禁止随机打散
- 静息段不计入三类指标，单独报 **误触发率**
- 每次训练落一个运行目录，参数、指标、混淆矩阵、模型全留档（评分项 8 的可复现）

## 快速开始

```bash
python3 -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 1. 没有真数据时，先造一份合成数据验证管线（合成数据不能写进报告）
python -m trainer.cli synthetic --out data/synthetic --subjects 5 --per-class 8 --rest 4

# 2. 训练 + LOSO 评估 + 导出
python -m trainer.cli run --data data/synthetic --out runs --model svm --synthetic

# 3. 模型对比 × 窗长敏感性，出报告用的表格
python -m trainer.cli experiments --data data/synthetic --out runs --synthetic
```

真数据到了，把 `--data` 指到录音目录，去掉 `--synthetic`，其余不用改。

## 数据怎么放

### 方式 A：手机导出的片段包（推荐）

App 的「记录」页右上角有导出按钮，产出 `clips_<时间>.zip`。解压后：

```
clips_20261125_150000/
  manifest.csv
  clips/p03_cough_20261125_143000_ev01.wav
  clips/p03_swallow_20261125_143500_ev01.wav
```

`manifest.csv` 的 `start_sec/end_sec` 是手机端人工标注过的精确区间，
`kind=clip` 表示这个文件本身就是裁好的单个事件。训练端会自动优先读它，
不再靠能量峰值猜事件边界。直接把 `--data` 指到解压目录即可。

### 方式 B：原始录音目录

直接把 App 导出的 WAV 堆在目录里，子目录也可以：

```
data/
  p01_swallow_20261125_101010.wav
  p02_cough_20261125_101100.wav
  p01_rest_20261125_101200.wav
```

文件名规范 `[被试编号_]类别_时间戳.wav`，类别取
`swallow` / `cough` / `interfere` / `rest`。没有被试编号会被塞进 `p00`，
LOSO 直接失真。这种方式的事件区间由短时能量峰值自动推断，精度不如方式 A。

## 目录结构

```
trainer/
  dataset.py     扫描 WAV、按文件名取被试和类别、能量定位事件、切 1 s 窗
  features.py    三域特征（时域 7 + 频域 11 + 时频 36），用于 SVM/RF 对比线
  models.py      svm / rf / mlp
  evaluate.py    LOSO、混淆矩阵、指标、误触发率、画图
  export.py      导出包（四件套）
  cnn.py         1D-CNN + tflite 导出（需要 TensorFlow，尚未接进 CLI）
  synthetic.py   合成数据生成器（仅自测）
  cli.py         命令行入口
```

## 导出包是什么

`runs/run_*/export/` 下这四个文件就是手机端的全部输入：

| 文件 | 内容 |
|---|---|
| `labels.txt` | 类别顺序，手机端按它解释输出下标 |
| `preprocess.json` | 采样率、窗长、归一化方式、输入形状 |
| `model.joblib` | 模型本体（走 CNN 路线时换成 `model.tflite`） |
| `eval.json` | LOSO 宏 F1、每类指标、混淆矩阵、模型体积、环境 |

`confusion.png` 和 `config.json` 是给报告和复现用的。

## 指标怎么算

- **宏平均 F1**：三类 F1 的平均，LOSO 每折各算一次，报均值 ± 标准差
- **每类召回**：混淆矩阵行和做分母，所以「真值是吞咽、被判成静息」也算漏检
- **误触发率**：静息窗里被判成任意事件类的比例，单独统计，不混进三类指标

## 还没做（按优先级）

1. **1D-CNN 接进 CLI**：`cnn.py` 里 build/train/export_tflite 都写好了，
   需要在装了 TensorFlow 的机器上补一段 LOSO 循环（建议 Python 3.11 + TF 2.16）
2. **数据增强**：时间偏移、增益抖动、加噪、±10% 时间拉伸。只做训练集，不做 pitch shift
3. **GUI 壳**：数据目录、模型选择、增强开关、轮数、训练按钮、结果图。内核先跑通再套壳
4. **端侧时延打点**：从帧就绪到出结果，目标 ≤ 20 ms

## 常见坑

- **窗长不一致**：训练用 1.0 s，手机端也必须 1.0 s。不一致的话模型装上去分数全乱
- **归一化不一致**：训练端逐窗 z-score，手机端要写成同一件事，别用「除以 32768」凑合
- **随机打散**：`train_test_split` 用在这里等于自己扣 4 分，本项目只允许按被试划分
- **TensorFlow 装不上**：Windows + Python 3.12 常见，换 3.11 或先只用 SVM/RF 出结果

## 说明

本目录的代码**没有在任何机器上实际跑过**（当前环境按你的要求只写不跑）。
第一次执行如果报错，先看 `synthetic` 那条命令——它不依赖你的数据，
能跑通说明环境和代码没问题，再换成真数据。
