# 大模型辅助的无人机集群博弈对抗问题研究

本仓库用于课题 **“大模型辅助的无人机集群博弈对抗问题研究”** 的算法实现与实验验证。

代码库基于多智能体强化学习框架 [**EPyMARL**](https://github.com/uoe-agents/epymarl)（Extended PyMARL）二次开发，完整保留了原框架的算法与配置体系，并接入了无人机集群博弈对抗仿真环境 [**MaCA**](https://github.com/SJTUwbl/MaCA)，在此基础上扩展了对手感知价值分解、大模型在线对手画像以及精调 QMIX 系列算法。

---

## 仿真环境接入：MaCA

本仓库通过 `src/envs/maca_wrapper.py` 与 `src/config/envs/maca.yaml` 将 [**MaCA**](https://github.com/SJTUwbl/MaCA) 接入 EPyMARL 的 `MultiAgentEnv` 接口，作为无人机集群博弈对抗问题研究的仿真平台：

- **支持场景**：`3v3`、`5v5`、`10v10` 同构对抗及 `2_10_vs_2_10` 异构对抗（支持通过 `env_args.map_name` 切换）。
- **CTDE 范式**：去中心化执行阶段仅使用红方局部与数据链共享观测（遵循战场迷雾，不依赖全局状态与 LLM API）；中心化训练阶段提供全局状态与对手交战统计上下文（`get_card_context`）。
- **环境路径发现**：默认自动发现同级目录 `../MaCA`，也可通过 `env_args.maca_path` 或环境变量 `MACA_PATH` 指定。

---

## 额外接入的算法

在 EPyMARL 原生算法（QMIX、VDN、IQL、COMA、QTRAN、MAPPO、IPPO、MAA2C、IA2C、MADDPG、PAC 等）基础上，本仓库目前额外接入并验证了以下 4 种方法：

| 算法 | 配置文件 | 核心特点 |
| :--- | :--- | :--- |
| **OA-QMIX** | `src/config/algs/oa_qmix.yaml` | 启发式对手感知 QMIX。在 CTDE 训练期基于滑动窗口交战统计构建 8 维对手行为画像，输入 `OAQMixer` 超网络指导中心化价值分解，无需外部 API。 |
| **LLM-OA-QMIX** | `src/config/algs/llm_oa_qmix.yaml` | 大模型辅助对手感知 QMIX。训练期周期性调用大模型 API（支持 OpenAI 兼容接口及本地 Ollama）对对手行为摘要进行结构化画像推断并注入 `OAQMixer`；网络异常时自动回退至启发式画像，执行期零 LLM 开销。 |
| **Finetuned-QMIX** | `src/config/algs/Finetuned-QMIX.yaml`（别名 `finetuned_qmix.yaml`） | 参考 [PyMARL2](https://github.com/hijkzzz/pymarl2) 实现的精调 QMIX 基线，集成 `NQLearner`（$TD(\lambda=0.6)$）、`NMAC`、`NRNNAgent`、`nmix`、Adam（`lr=0.001`）、大批量并行采样（`batch_size=128`, `batch_size_run=8`）与 `obs_last_action=True`。 |
| **Finetuned-LLM-OA-QMIX** | `src/config/algs/Finetuned-LLM-OA-QMIX.yaml`（别名 `finetuned_llm_oa_qmix.yaml`） | 将 `OAQMixer` 与在线 LLM 对手画像模块融合到 `Finetuned-QMIX`（`NQLearner`）主干网络中的增强方法。 |

---

## 新增核心目录与文件

- `src/envs/maca_wrapper.py` / `src/config/envs/maca.yaml`：MaCA 仿真环境封装与默认配置
- `src/modules/mixers/oa_qmix.py`：`OAQMixer` 对手感知超网络混合器
- `src/modules/critics/opponent_aux.py`：`OpponentAuxHead` 可选对手风格/集火目标辅助预测头
- `src/llm_marl/`：对手特征提取、启发式/LLM 画像构建、OpenAI 兼容/Ollama 客户端与运行时管理
- `src/learners/nq_learner.py`、`src/controllers/n_controller.py`、`src/modules/agents/n_rnn_agent.py`、`src/modules/mixers/nmix.py`：PyMARL2 精调 QMIX 主干组件
- `config/llm/.env.llm.example`：大模型 API 配置模板（实际凭据文件 `.env.llm.local` 已被 `.gitignore` 忽略）
- `tests/test_oa_qmix_integration.py`：MaCA 接入、OA-QMIX、LLM-OA-QMIX 与 Finetuned-LLM-OA-QMIX 集成测试
- `plot_*_comparison.py` / `plot_all_qmix_variants_2m.py`：实验结果对比可视化脚本

---

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置大模型 API（仅 LLM-OA-QMIX / Finetuned-LLM-OA-QMIX 需要）

复制模板并填入你的接口地址、密钥与模型名称（本地文件不会被 Git 追踪）：

```bash
cp config/llm/.env.llm.example config/llm/.env.llm.local
# 编辑 config/llm/.env.llm.local 填写 LLM_BASE_URL、LLM_API_KEY、LLM_MODEL
```

### 3. 运行训练示例

```bash
# QMIX 基线（MaCA 3v3）
python src/main.py --config=qmix --env-config=maca with env_args.map_name="3v3" runner="parallel" batch_size_run=4 use_rnn=True t_max=2000000

# OA-QMIX（启发式对手画像，无需外部 API）
python src/main.py --config=oa_qmix --env-config=maca with env_args.map_name="5v5" runner="parallel" batch_size_run=4 use_rnn=True t_max=2000000

# LLM-OA-QMIX（在线 LLM 对手画像）
python src/main.py --config=llm_oa_qmix --env-config=maca with env_args.map_name="5v5" runner="parallel" batch_size_run=4 use_rnn=True t_max=2000000

# Finetuned-QMIX 与 Finetuned-LLM-OA-QMIX
python src/main.py --config=Finetuned-QMIX --env-config=maca with env_args.map_name="5v5" t_max=2000000
python src/main.py --config=Finetuned-LLM-OA-QMIX --env-config=maca with env_args.map_name="5v5" t_max=2000000
```

### 4. 运行集成测试

```bash
python -m unittest discover -s tests -p "test_*.py" -v
```

---

## 开源致谢与引用

本项目基于以下开源项目构建，感谢原作者的贡献：

- **EPyMARL**：[https://github.com/uoe-agents/epymarl](https://github.com/uoe-agents/epymarl)（Papoudakis et al., *Benchmarking Multi-Agent Deep Reinforcement Learning Algorithms in Cooperative Tasks*, NeurIPS 2021）
- **PyMARL**：[https://github.com/oxwhirl/pymarl](https://github.com/oxwhirl/pymarl)
- **PyMARL2**：[https://github.com/hijkzzz/pymarl2](https://github.com/hijkzzz/pymarl2)
- **MaCA**：[https://github.com/SJTUwbl/MaCA](https://github.com/SJTUwbl/MaCA)

## License

本仓库继承自 EPyMARL / PyMARL 的源码与新增代码均遵循 **Apache License 2.0**（详见 [LICENSE](LICENSE)）。
