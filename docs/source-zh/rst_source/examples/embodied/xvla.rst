X-VLA
=====

.. figure:: https://robotwin-platform.github.io/assets/images/teaser.png
   :align: center
   :width: 90%

   RoboTwin 2.0 双臂任务（图片：`RoboTwin <https://robotwin-platform.github.io>`__）。

在原生 RoboTwin 2.0 上运行 X-VLA 的单 GPU PPO 或 GRPO 接入检查。使用固定的 RoboTwin 权重，先准备原生仿真环境与资产，再在 Slurm 分配内启动配置。短测用于验证训练实现，不能作为 benchmark 成绩。

采样器参考 `πRL <https://arxiv.org/html/2510.25889v3>`__ 的随机转移方法。X-VLA 预测 clean action；adapter 先将它换算为 velocity，再执行 Flow-Noise 或 Flow-SDE 转移。既有冻结评测保留作者原生 clean-action 递推，作为独立评测协议。

新采样器通过 Euler 更新推进当前 noisy state；作者递推则始终使用同一份初始噪声与上一次 clean prediction，重新构造每次 noisy input。因此，新的 ``flow_ode`` 评估也改变了原生递推。adapter 采用 πRL 的随机转移定义，但尚未证明保留 X-VLA 原生动作的精确边缘分布，尤其是使用二元交叉熵训练、以 logit 表示的夹爪分支。应把它作为独立定义的 policy 验证，不作为 πRL 已发表性能的复现。

概览
----

PPO 同时训练 policy 和 value head；GRPO 比较同一初态下四条独立采样轨迹。

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

采样与更新
~~~~~~~~~~

两份配置默认使用 ``actor.model.xvla.noise_method=flow_noise``，每次去噪转移的标准差由有界可学习 noise head 提供。通过同一入口传入 override 可选择 Flow-SDE：

.. code-block:: bash

   bash tools/xvla/train.sh robotwin_ppo_xvla \
     actor.model.xvla.noise_method=flow_sde

policy 缓存完整且 detached 的 20D 去噪链，并在更新时用同一批状态重算联合 log-probability，对去噪时间、预测 horizon 和 raw 坐标求和。这是扩展采样路径的密度，包含不影响最终执行前缀的 latent；不把它称为阈值化夹爪命令的密度。

RLinf 通过 ``[B,C,1]`` 的首个动作槽位接收联合标量，其余槽位置零。保持 ``logprob_action_dim=1`` 和 ``logprob_type=chunk_level``，确保该标量只求和一次。整块未执行时全部 mask；执行中提前终止时，已经采样的完整路径仍进入 likelihood。

每个 chunk 的奖励为原生逐 anchor 奖励之和；PPO 的 ``gamma=0.99`` 按完整 chunk 折扣，300-anchor horizon 作为有限回合上限。评估使用相同 clean-to-velocity 转换的 ``flow_ode``，与作者原生冻结评测分开比较。

可视化与结果
------------

扩大一轮配置前，先检查奖励、actor 与 noise-head 梯度、likelihood ratio、``executed_mask``、权重同步和 checkpoint 重载。框架指标定义见 :doc:`训练指标 </rst_source/reference/metrics>`。

新训练路径需要单独记录 GPU rollout 与真实更新验收。历史单臂 LIBERO 训练和 RoboTwin 冻结评测不能替代该证据，应与本轮结果分开登记。

对应 GPU 测试配置为 ``tests/e2e_tests/embodied`` 下的 ``robotwin_xvla_ppo_smoke`` 和 ``robotwin_xvla_grpo_smoke``，须在获配资源内使用同样准备好的路径运行。进程成功退出本身不能证明有效 GRPO 更新，仍须检查上述组内回报与梯度证据。
