# X‑VLA RoboTwin 2.0 Easy／Hard 基座能力评测

在 `MoliaiELS/RLinf-VLA:codex/xvla` 上运行 X‑VLA RoboTwin 2.0 的冻结基座评测。本轮用于 **baseline calibration／能力审计**，统一使用按 RoboTwin 示教数据适配的官方权重。后续 RL 对照须另外冻结独立评测初态。 [官方 RoboTwin 权重](https://huggingface.co/2toINF/X-VLA-RoboTwin2/tree/a157c580cfe6f9f445614490f3bec1b2f9ef9f18)

本分支从 RLinf 固定提交 `1bf78980d9339cbc67497eddceb9a1e3a74e2acc` 派生。原始实验作业和结果按其原分支及执行提交追溯。 [先前作业登记](https://github.com/MoliaiELS/continual-vla-rl/blob/codex/xvla-integration/docs/robotwin/status.md)

任务目录列出官方 50 项任务的分类、成功判定和覆盖状态。X‑VLA 作者报告的公开成绩为 Easy 70.0%、Hard 39.0%。 [任务目录](task-catalog.md) [作者 README](https://github.com/2toinf/X-VLA/blob/6c4ecfb8be458ae7c0c96094e4b0add4174f4fe2/evaluation/robotwin-2.0/README.md) [逐任务参考](https://arxiv.org/html/2510.10274v1#A13.T16)

## 固定身份和实验定义

| 项目 | 版本或规则 |
| --- | --- |
| 权重 | `2toINF/X-VLA-RoboTwin2@a157c580cfe6f9f445614490f3bec1b2f9ef9f18`；`ee6d`、proprio 开启、30 anchors；FP32、冻结、10 次 denoising；权重及 processor 哈希见已校验下载回执 |
| 项目和上游 | Slurm 运行固定的 `codex/xvla` 提交。RLinf 根仓库基线为 `1bf78980d9339cbc67497eddceb9a1e3a74e2acc`，作者 X‑VLA 参考为 `6c4ecfb8be458ae7c0c96094e4b0add4174f4fe2`。 |
| 仿真和规划 | 原生 RoboTwin `6e75b8b80ee32d2f495eb756cf5687e4e16f0eea`，权重发布前最近的 `main` 提交；保留作者 client 所需的 `task_config`、`script` 和逐 anchor `take_action(action_type='ee')`。CuRobo v0.7.8 `d64c4b005459db10c5dd867d8b30a87d5bda9bdb`。预定 Python 3.11、Torch 2.6 / CUDA 12.4、SAPIEN 3.0.0b1、MPLib 0.2.1；是否可运行以 Slurm 验收为准。 |
| 资产 | `TianxingChen/RoboTwin2.0@25ed6fa76343d7b900e2b44d39f32a0eae1950b1` 的三个原始 ZIP，逐文件 SHA256 验证。Aloha AgileX，按原生脚本展开放置路径。 |
| Easy／Hard | `task_config/demo_clean.yml` / `demo_randomized.yml`，保留随机化参数和原生物体、纹理、相机。使用原生 API 避开 RLinf 训练包装层的关节执行方式及相机 near plane 默认值。 |
| 观测 | head、left、right 顺序 RGB，原生 D435 320×240、near 0.1 m、far 100 m，原生 ray tracing 32 spp，无额外 crop；缩放交给原生 processor。domain 6。 |
| 动作与历史 | 双臂 20D `xyz + rot6D + grip`，转换成两个 8D `xyz + quat + grip` 命令。保留作者 SciPy 参数和 RoboTwin planner 使用的原始四元数分量顺序。夹爪阈值 `>0.7`，命令在 `{-1,+1}`。后续 proprio 的 pose 部分来自最后命令，夹爪状态仍为实测。 |
| 指令 | `task_id.replace('_', ' ')`，完整文本写入每个 episode。 |
| 终止 | 每回合最多 10 次模型查询，每次 30 anchors；每个 anchor 调用原生成功判定，成功即停。原生 `step_lim` 之后的 no-op 仍计为已发送命令，单独记实际推进的 anchors、仿真 tick、settling ticks、模拟秒数和规划失败。默认仿真步长 1/250 s。 |

三个任务的原生成功条件依赖专家示教脚本赋值：`open_laptop`、`place_object_scale`、`put_object_cabinet`。评测 adapter 在 reset 后只用原始对象位姿按专家脚本的规则初始化相关字段，不执行示教动作、不改成功谓词。`place_fan` 无此依赖。两个 Hard 任务复用时可能受原生可变默认参数影响，adapter 每次传入独立列表。SAPIEN 绑定到 Slurm 获配设备，对齐 GPU 参数而不改画面参数。

## 初态、统计与产物

- Smoke：`move_stapler_pad`、`turn_switch` × Easy／Hard × 5，使用独立种子。在每个条件的冻结初态上运行原生 client/服务端和本地 runner 的同观测、同噪声对照，并回放同一 chunks 比较命令、轨迹、tick 和成功。四个包装层缺口另做 `coverage`，与正式数据分开。
- 正式：50 个任务每条件首批 100，共 10,000 回合。预先指定 10 个重点任务每条件再各加两批各 100，累计 14,000。批次是 evaluation batches。
- 固定条件内按候选 seed 升序预筛，正式从 2000 开始，smoke／coverage 使用独立命名空间。通过原生专家成功判定并去重后冻结完整清单及 policy seed，新旧回合不按结果重新抽样。Easy／Hard 的同整数 seed 不作为配对证据。
- 正常失败只记一次。基础设施异常按相同 seed 有限重试，耗尽记 `unresolved`。保留候选排除记录、每次尝试和回合日志；每个 episode ID 只进入一次分母。计划数、完成数、Wilson 区间和未完成范围通过汇总命令输出。总体与类别以任务等权计算，首批与扩测累计分列。
- 视频以固定顺序保存每个任务／条件前两次成功和前两次失败。未出现对应结果或编码失败时在日志中体现。

服务器 `DataRoot` 指向本分支的 RoboTwin 结果目录，例如个人数据目录下的 `RLinf/xvla/robotwin-native/`。它保存 `jobs/<jobid>/runtime.json`、`logs/`、`panels/<项目提交>/<阶段>/<任务>/<条件>/` 和 `reports/`。`InputRoot` 提供 `archives/`、`models/` 和解包后的 `sim-assets/`，默认使用 `DataRoot`。项目代码位于个人 homes。每个作业的 `runtime.json` 记录 GPU、驱动、依赖、代码和配置哈希。每个任务和条件保存模型与初态清单、协议指纹、逐回合记录、对齐证据和视频。

现有派发器仍在旧仓库固定提交 `009f05cfe4de2de9fe4d911b6fd665776add7abc` 上运行，结果在 `RLinf/xvla/robotwin-b0/`。新分支已迁移评测入口，在新目录运行时需重新通过探针与 smoke。 [旧评测运行登记](https://github.com/MoliaiELS/continual-vla-rl/blob/codex/xvla-integration/docs/robotwin/status.md)

## 执行与恢复

在本机的 `codex/xvla` 根目录创建个人连接配置。填写 `SshHost`、`Partition` 和服务器路径。建议在个人 homes 中使用 `RLinf-VLA` 编辑目录、`RLinf-VLA-runs` 作业目录和独立的 `SimulatorCodeRoot`。IDAC 的 `EnvPrefix` 指向个人 homes 中已验收的 `xvla_robotwin_b0` 环境。`QualifiedDataRoot` 提供对应的安装验收回执，本分支以只读方式复用其依赖和已编译 CuRobo。 [IDAC 配置模板](../../tools/idac/settings.example.psd1)

```powershell
Copy-Item ./tools/idac/settings.example.psd1 ./tools/idac/settings.local.psd1
```

`sync -RootOnly` 推送本机提交并快进服务器编辑目录。提交作业前用 `sync` 另建固定 worktree。若编辑目录中有未提交改动，请先提交或保存这些改动，再同步。

```powershell
./tools/xvla/remote.ps1 sync
./tools/xvla/remote.ps1 robotwin-prepare env
./tools/xvla/remote.ps1 robotwin-logs prepare-env
```

`robotwin-prepare env` 核验固定源码和已配置的环境，成功会记录 `ROBOTWIN_SOURCES_READY`。如不配置 `InputRoot`，上述校验后在本分支的数据目录下载权重和资产。

```powershell
./tools/xvla/remote.ps1 download -Model robotwin
./tools/xvla/remote.ps1 robotwin-prepare assets
```

如 `InputRoot` 指向旧 B0 的输入目录，使用其中已经校验的权重和资产，等待旧资产下载及解包完成。权重就绪后可先跑探针。资产就绪后运行 `link`，为新的模拟器 checkout 生成本分支的具身配置。

```powershell
./tools/xvla/remote.ps1 robotwin-submit probe
./tools/xvla/remote.ps1 robotwin-status
./tools/xvla/remote.ps1 robotwin-logs 12345
./tools/xvla/remote.ps1 robotwin-prepare link
```

权重、资产、probe 就绪后，分别提交两个单卡 lane，比对串行与双 lane 吞吐、显存峰值和 reset／规划耗时。达到 smoke 的证据门槛后，才为相同执行提交运行正式或扩测。实际时间按 smoke 统计的有效回合/小时及专家预筛接受率外推；未实测前不给 10,000／14,000 回合估算。

```powershell
$c = git rev-parse HEAD # 或从 jobs.tsv 取既有评测提交
./tools/xvla/remote.ps1 robotwin-submit smoke -RobotCommit $c -RobotLane easy
./tools/xvla/remote.ps1 robotwin-submit smoke -RobotCommit $c -RobotLane hard
./tools/xvla/remote.ps1 robotwin-report smoke -RobotCommit $c
./tools/xvla/remote.ps1 robotwin-submit coverage -RobotCommit $c
# 以上回收验收证据后再提交正式任务，默认优先十个重点任务。
./tools/xvla/remote.ps1 robotwin-submit formal -RobotCommit $c -RobotLane easy -RobotHours 48
./tools/xvla/remote.ps1 robotwin-submit formal -RobotCommit $c -RobotLane hard -RobotHours 48
./tools/xvla/remote.ps1 robotwin-report formal -RobotCommit $c
# 首批结果与缺口记录齐备后，按照预定计划新增两批。
./tools/xvla/remote.ps1 robotwin-submit extension -RobotCommit $c -RobotLane easy -RobotHours 48
./tools/xvla/remote.ps1 robotwin-submit extension -RobotCommit $c -RobotLane hard -RobotHours 48
```

Slurm 超时或中断后，用同一 `-RobotCommit`、阶段和 lane 重新提交。runner 保留已接受初态、尝试次数和正式终局结果；`-RobotTask <ID>` 可只恢复某个任务。观测、动作、指令、预算或成功判定若有变化，须新建协议并独立记录。超过自动重试上限的回合继续列为 unresolved。Slurm 最多两张己方卡，排除登录节点，保留本次作业的 GPU 可见性。作业不声明无法调度的内存申请，物理内存只作为测量值。

也可以在环境校验后运行 `./tools/xvla/remote.ps1 robotwin-run -RobotCommit $c -RobotHours 48`，让派发器在服务器上推进完整流程。派发器等待权重与资产校验、GPU 探针和 smoke 验收，再提交正式首批。锁和 `dispatch/$c/state.json` 记录进度并避免重复提交。它按批汇总，并在 Slurm 超时、抢占或节点故障后有限次自动续跑。使用 `robotwin-status` 查当前状态。
