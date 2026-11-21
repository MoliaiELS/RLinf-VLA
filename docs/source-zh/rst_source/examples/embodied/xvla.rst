X-VLA
=====

.. figure:: https://robotwin-platform.github.io/assets/images/teaser.png
   :align: center
   :width: 90%

   RoboTwin 2.0 双臂任务（图片：`RoboTwin <https://robotwin-platform.github.io>`__）。

在原生 RoboTwin 2.0 上运行 X-VLA 的单 GPU PPO 或 GRPO 接入检查。使用固定的 RoboTwin 权重，先准备原生仿真环境与资产，再在 Slurm 分配内完成单轮检查、多次有效更新和新进程 checkpoint 续训验收。短测用于验证训练实现，不能作为 benchmark 成绩。

采样器参考 `πRL <https://arxiv.org/html/2510.25889v3>`__ 的随机转移方法。X-VLA 预测 clean action；adapter 先将它换算为 velocity，再执行 Flow-Noise 或 Flow-SDE 转移。既有冻结评测保留作者原生 clean-action 递推，作为独立评测协议。

新采样器通过 Euler 更新推进当前 noisy state；作者递推则始终使用同一份初始噪声与上一次 clean prediction，重新构造每次 noisy input。因此，新的 ``flow_ode`` 评估也改变了原生递推。adapter 采用 πRL 的随机转移定义，但尚未证明保留 X-VLA 原生动作的精确边缘分布，尤其是使用二元交叉熵训练、以 logit 表示的夹爪分支。应把它作为独立定义的 policy 验证，不作为 πRL 已发表性能的复现。

概览
----

PPO 同时训练 policy 和 value head；GRPO 比较同一初态下四条独立采样轨迹。验收范围为下述 Easy 任务上的单 GPU、FP32、``NO_SHARD``、Flow-Noise 和 ``prompt_heads`` 微调。

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: 环境
      :text-align: center

      原生 RoboTwin 2.0

   .. grid-item-card:: 算法
      :text-align: center

      PPO / GRPO

   .. grid-item-card:: 任务
      :text-align: center

      move_stapler_pad（Easy）

   .. grid-item-card:: 硬件
      :text-align: center

      1 张获配 GPU

任务
~~~~

初始配置只运行一个任务，便于先检查奖励、policy 梯度、权重同步和 checkpoint 恢复，再扩大训练范围。

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - 配置项
     - 默认值
   * - 任务与条件
     - ``move_stapler_pad``、``easy``；使用原生 ``demo_clean.yml``。
   * - 回合预算
     - 最多 300 个动作 anchor；成功或原生任务上限可提前结束执行。
   * - 奖励
     - 原生任务成功，只计一次；未执行的动作槽位奖励为零。
   * - 初态
     - 训练 seed 为 ``12000``–``12003``，评估 seed 为 ``13000``–``13003``。

观测与动作
~~~~~~~~~~

保留原生相机帧和双臂末端命令。训练使用动作执行后的实测状态构建下一次观测，不把未执行的预测当作后续状态。

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - 字段
     - 说明
   * - 图像
     - head、left wrist、right wrist 顺序的 RGB；缩放由固定 processor 处理。
   * - 状态
     - ``states`` 中的双臂 20D 状态；domain ID 为 ``6``。
   * - 指令
     - 将任务名中的下划线替换为空格，写入 ``task_descriptions``。
   * - 动作
     - 每次预测 30 个双臂 ``xyz + rot6D + grip`` anchor，由原生 adapter 转换为末端命令。
   * - 终止
     - 成功或达到上限后停止执行，通过 ``executed_mask`` 标识实际执行的槽位。

安装
----

按本 checkout 的 ``docs/robotwin/README.md`` 准备固定原生 RoboTwin、资产和 CuRobo 编译，仿真源码固定为 ``6e75b8b80ee32d2f495eb756cf5687e4e16f0eea``。模型安装器添加 Python 依赖，仿真资产与 CuRobo 编译仍需单独准备。

在个人 home 目录创建模型环境：

.. code-block:: bash

   bash requirements/install.sh --no-root --venv /path/to/home/venvs/xvla-rl \
     embodied --model xvla --env robotwin
   source /path/to/home/venvs/xvla-rl/bin/activate

该命令选择 X-VLA 模型依赖及固定原生仿真的 Python requirements。作业使用固定代码 worktree；模型、缓存和结果放在个人存储，仿真与训练通过 Slurm 提交。

下载模型
--------

作业启动前下载并校验固定官方权重：

.. code-block:: bash

   python tools/xvla/download.py /path/to/personal-storage/models robotwin
   export XVLA_MODEL_PATH=/path/to/personal-storage/models/X-VLA-RoboTwin2--a157c580cfe6

下载器固定使用 ``2toINF/X-VLA-RoboTwin2@a157c580cfe6f9f445614490f3bec1b2f9ef9f18``，并将文件哈希写入 ``verified.json``。训练加载器要求这份回执，执行本地 vendored 模型代码。

运行
----

在 Slurm 作业内激活环境、设置以下三个路径，并从该作业的固定仓库 worktree 运行入口：

.. code-block:: bash

   export ROBOTWIN_PATH=/path/to/pinned/RoboTwin
   export XVLA_MODEL_PATH=/path/to/personal-storage/models/X-VLA-RoboTwin2--a157c580cfe6
   export XVLA_RUN_DIR=/path/to/personal-storage/runs/"${SLURM_JOB_ID}"
   bash tools/xvla/train.sh robotwin_ppo_xvla

入口要求已有 Slurm 分配，保留 GPU 可见性，并使用独立且有界的 Ray runtime。PPO 使用两个环境、GAE 和 actor-critic loss；整批没有正奖励时仍参与 PPO 更新。默认只执行一轮训练，然后评估更新后的 policy 并保存 checkpoint。

运行 GRPO 时改用对应配置：

.. code-block:: bash

   bash tools/xvla/train.sh robotwin_grpo_xvla

GRPO 每组四个环境，使用 actor loss，不创建 value head。组内共享初态 seed，policy 独立采样。记录组内回报方差和保留率；全胜或全败组不能证明有效 GRPO 更新。

验收多轮有效更新
~~~~~~~~~~~~~~~~

完成单轮检查后，使用验收配置确认训练能跨批次持续更新。从头运行验收时，两份配置都要求至少三次有效 optimizer 更新，采集轮数上限和评估、保存安排如下：

.. list-table::
   :header-rows: 1
   :widths: 25 35 40

   * - 算法
     - 采集轮数
     - 评估与 checkpoint
   * - PPO
     - 默认 3 轮
     - 每 3 轮及最后一轮
   * - GRPO
     - 最多 6 轮
     - 每 3 轮及最后一轮

每次运行使用新的输出目录。运行 PPO 时，仅对本次命令启用审计：

.. code-block:: bash

   export XVLA_RUN_DIR=/path/to/personal-storage/runs/"${SLURM_JOB_ID}"-ppo-acceptance
   XVLA_ONLINE_ACCEPTANCE=1 bash tools/xvla/train.sh robotwin_ppo_xvla_acceptance

运行 GRPO 时使用对应配置和独立输出目录：

.. code-block:: bash

   export XVLA_RUN_DIR=/path/to/personal-storage/runs/"${SLURM_JOB_ID}"-grpo-acceptance
   XVLA_ONLINE_ACCEPTANCE=1 bash tools/xvla/train.sh robotwin_grpo_xvla_acceptance

验收配置设置 ``env.train.reset_seed_mode=cycle``，各组初态按原训练池 ``12000``–``12003`` 的顺序轮换，到末尾后从头继续。GRPO 组内成员仍共享同一初态。评估池 ``13000``–``13003`` 保持独立。

即使 GRPO 整批组都被过滤，采集轮次仍会推进 ``global_step``。空批跳过 optimizer 和 scheduler，保留模型与 Adam 状态，不计有效更新。六轮内未达到三次有效更新时，验收失败。PPO 保留没有正奖励的批次，但仍须产生有限、非零的 policy 梯度和实际参数变化，才计入有效更新。

审计将 ``audit-<pid>.jsonl`` 和 ``online-acceptance.json`` 写入 ``XVLA_RUN_DIR``。检查摘要状态，以及梯度、参数变化、权重同步、真实环境评估和 checkpoint 校验记录；完成采集预算本身不能证明验收通过。

在新进程中续训
~~~~~~~~~~~~~~

已有可续训 checkpoint 后，在新进程中检查训练能否从保存状态继续。原生 distributed checkpoint（DCP）保存完整模型（包括冻结参数与 buffer）、Adam 状态、scheduler 和 actor RNG；审计 JSON sidecar 保存 actor 的训练计数与 policy version，并校验恢复后的训练状态。worker sidecar 保存 rollout、环境 worker 在回合边界的 RNG 与 reset 调度。恢复后沿原有顺序选择下一回合初态，不恢复回合中途的仿真物理状态。

若 PPO 已保存 ``global_step_3``，在新的 Slurm 作业内激活相同环境、设置已准备好的路径，选择新的空输出目录，并保持任务、模型和 seed pool 一致：

.. code-block:: bash

   export XVLA_RUN_DIR=/path/to/personal-storage/runs/"${SLURM_JOB_ID}"-ppo-resume
   XVLA_ONLINE_ACCEPTANCE=1 bash tools/xvla/train.sh robotwin_ppo_xvla_acceptance \
     runner.resume_dir=/path/to/original-output/robotwin_ppo_xvla_acceptance/checkpoints/global_step_3 \
     runner.max_epochs=4 runner.max_steps=4

checkpoint 目录包含 ``actor/dcp_checkpoint`` 及配套审计 sidecar，包括 ``xvla_worker_state.pt``；续训时须一并保留。恢复校验比较模型、optimizer、scheduler、RNG 和计数。续训验收要求累计至少三次有效 optimizer 更新，其中新进程至少新增一次有效更新。最后一轮也会评估和保存，因此第 4 轮仍会检查，不受每 3 轮间隔影响。

.. warning::

   若没有执行过更新且 Adam 状态为空，验收记录原因 ZERO_UPDATE_EMPTY_ADAM_STATE，不创建可续训 checkpoint；标准训练入口也应用相同保护。续训应选择实际执行过有效更新的 checkpoint，不能仅凭失败验收留下的目录判断。

采样与更新
~~~~~~~~~~

两份配置默认使用 ``actor.model.xvla.noise_method=flow_noise``，每次去噪转移的标准差由有界可学习 noise head 提供。通过同一入口传入 override 可选择 Flow-SDE：

.. code-block:: bash

   bash tools/xvla/train.sh robotwin_ppo_xvla \
     actor.model.xvla.noise_method=flow_sde

policy 缓存完整且 detached 的 20D 去噪链，并在更新时用同一批状态重算联合 log-probability，对去噪时间、预测 horizon 和 raw 坐标求和。这是扩展采样路径的密度，包含不影响最终执行前缀的 latent；不把它称为阈值化夹爪命令的密度。

rollout 和 actor 重算均逐样本计算观测与去噪路径，每次 batch size 为一，与环境组大小和 actor microbatch size 分开。这使首次更新前模型和 noise head 使用一致的计算形状。Gaussian score 运算使用 float64，联合标量再转换为 float32，以兼容原生 RLinf loss。更新前联合 log-probability 检查仍保留 0.01 的绝对误差阈值。

RLinf 通过 ``[B,C,1]`` 的首个动作槽位接收联合标量，其余槽位置零。保持 ``logprob_action_dim=1`` 和 ``logprob_type=chunk_level``，确保该标量只求和一次。整块未执行时全部 mask；执行中提前终止时，已经采样的完整路径仍进入 likelihood。

每个 chunk 的奖励为原生逐 anchor 奖励之和；PPO 的 ``gamma=0.99`` 按完整 chunk 折扣，300-anchor horizon 作为有限回合上限。actor 和 critic 的 loss 归一化也按 chunk 计数：最后一块即使只执行部分 anchor，仍计一个有效宏步，与完整执行的 chunk 一致。回合上限向上取整为 chunk 数，300 个 anchor、chunk size 为 30 时共 10 个宏步。真实控制步仍由 ``executed_mask`` 计数，与宏步分开记录。

评估使用相同 clean-to-velocity 转换的 ``flow_ode``，与作者原生冻结评测分开比较。

可视化与结果
------------

扩大验收配置前，先检查奖励、actor 与 noise-head 梯度、likelihood ratio、``executed_mask``、权重同步和 checkpoint 重载。框架指标定义见 :doc:`训练指标 </rst_source/reference/metrics>`。

新训练路径需要单独记录 GPU rollout 与真实更新验收。历史单臂 LIBERO 训练和 RoboTwin 冻结评测不能替代该证据。按每次运行的实际记录填写验收状态；这些检查不能证明 benchmark 性能或 Hard 任务覆盖。

对应 GPU 测试配置为 ``tests/e2e_tests/embodied`` 下的 ``robotwin_xvla_ppo_smoke`` 和 ``robotwin_xvla_grpo_smoke``，须在获配资源内使用同样准备好的路径运行。进程成功退出本身不能证明有效 GRPO 更新，仍须检查上述组内回报与梯度证据。
