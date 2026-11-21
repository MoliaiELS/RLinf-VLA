X-VLA
=====

.. figure:: https://robotwin-platform.github.io/assets/images/teaser.png
   :align: center
   :width: 90%

   RoboTwin 2.0 dual-arm tasks (image: `RoboTwin <https://robotwin-platform.github.io>`__).

Run single-GPU PPO or GRPO integration checks for X-VLA on native RoboTwin 2.0.
Start from the pinned RoboTwin checkpoint, prepare the native simulator and
assets, then launch a single-round check inside a Slurm allocation. Follow it
with multiple effective updates and a fresh-process checkpoint resume. These
short runs validate the training implementation; they are not benchmark results.

The sampler follows the stochastic-transition construction in
`πRL <https://arxiv.org/html/2510.25889v3>`__. X-VLA predicts clean actions;
the adapter converts those predictions into a velocity field before applying
Flow-Noise or Flow-SDE transitions. The existing frozen evaluator keeps the
author's native clean-action recurrence as a separate evaluation protocol.

The new sampler advances the current noisy state with Euler updates. The
author's recurrence instead rebuilds each noisy input from the same initial
noise and the previous clean prediction, so the new ``flow_ode`` evaluation
also changes that recurrence. The adapter adopts πRL's stochastic-transition
formulation; it does not establish exact preservation of X-VLA's native action
marginals, particularly for the gripper branch trained with binary
cross-entropy and represented as logits. Treat this as a separately defined
policy, not a reproduction of πRL's published performance.

Overview
--------

Choose PPO to train a value head with the policy, or GRPO to compare four
independent rollouts from a shared initial condition. The acceptance recipe
covers one GPU with FP32, ``NO_SHARD``, Flow-Noise, and ``prompt_heads`` tuning
on the Easy task below.

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: Environment
      :text-align: center

      Native RoboTwin 2.0

   .. grid-item-card:: Algorithm
      :text-align: center

      PPO / GRPO

   .. grid-item-card:: Tasks
      :text-align: center

      move_stapler_pad (Easy)

   .. grid-item-card:: Hardware
      :text-align: center

      1 allocated GPU

Tasks
~~~~~

The initial configuration uses one task so you can inspect reward, policy
gradients, synchronization, and checkpoint recovery before extending the run.

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Setting
     - Default
   * - Task and condition
     - ``move_stapler_pad``, ``easy``; native ``demo_clean.yml`` configuration.
   * - Episode budget
     - At most 300 action anchors; success or the native task limit can end execution earlier.
   * - Reward
     - Native task success, counted once; unexecuted action slots have zero reward.
   * - Initial conditions
     - Training seeds ``12000``–``12003`` and evaluation seeds ``13000``–``13003``.

Observation and Action
~~~~~~~~~~~~~~~~~~~~~~

Use native camera frames and dual-arm end-effector commands. Training reads
the measured state after execution rather than propagating an unexecuted
prediction as the next observation.

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Field
     - Description
   * - Images
     - Head, left wrist, and right wrist RGB frames; resizing belongs to the pinned processor.
   * - State
     - 20-D dual-arm state in ``states``; domain ID ``6``.
   * - Instruction
     - Task name with underscores replaced by spaces, in ``task_descriptions``.
   * - Action
     - 30 anchors of dual-arm ``xyz + rot6D + grip``; the native adapter converts them into end-effector commands.
   * - Termination
     - Stop execution after success or a limit and expose the actual executed slots in ``executed_mask``.

Installation
------------

Prepare the pinned native RoboTwin checkout, assets, and CuRobo build using
``docs/robotwin/README.md`` in this checkout. Keep the simulator commit at
``6e75b8b80ee32d2f495eb756cf5687e4e16f0eea``. The model installer adds the
Python dependencies; simulator assets and the CuRobo build remain separate.

Create the model environment in your personal home directory:

.. code-block:: bash

   bash requirements/install.sh --no-root --venv /path/to/home/venvs/xvla-rl \
     embodied --model xvla --env robotwin
   source /path/to/home/venvs/xvla-rl/bin/activate

This selects the X-VLA model dependencies and pinned native simulator Python
requirements. Use a fixed code worktree for a job, keep large model/cache/result
files in personal storage, and submit simulation and training through Slurm.

Download the Model
------------------

Download and verify the fixed official checkpoint before starting the job:

.. code-block:: bash

   python tools/xvla/download.py /path/to/personal-storage/models robotwin
   export XVLA_MODEL_PATH=/path/to/personal-storage/models/X-VLA-RoboTwin2--a157c580cfe6

The downloader pins ``2toINF/X-VLA-RoboTwin2`` at
``a157c580cfe6f9f445614490f3bec1b2f9ef9f18`` and writes ``verified.json`` with
file hashes. The training loader requires that receipt and loads the local
vendored model code.

Run It
------

Activate the environment and set the three paths below inside your Slurm job.
Run the launcher from the job's fixed repository worktree:

.. code-block:: bash

   export ROBOTWIN_PATH=/path/to/pinned/RoboTwin
   export XVLA_MODEL_PATH=/path/to/personal-storage/models/X-VLA-RoboTwin2--a157c580cfe6
   export XVLA_RUN_DIR=/path/to/personal-storage/runs/"${SLURM_JOB_ID}"
   bash tools/xvla/train.sh robotwin_ppo_xvla

The launcher requires an existing Slurm allocation, preserves its GPU
visibility, and starts RLinf with a bounded, isolated Ray runtime. PPO uses
two environments, GAE, and the actor-critic loss. A batch with no positive
reward still participates in PPO updates. The default run performs one
training iteration, evaluates the updated policy, and saves a checkpoint.

To run GRPO, choose its configuration instead:

.. code-block:: bash

   bash tools/xvla/train.sh robotwin_grpo_xvla

GRPO uses four environments per group and the actor loss without a value head.
The group shares an initial seed while policy samples are independent. Record
group reward variance and retention: an all-success or all-failure group does
not establish an informative GRPO update.

Check Multiple Effective Updates
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

After the single-round check, use an acceptance configuration to verify that
training can continue across several batches. A fresh acceptance run requires
at least three effective optimizer updates. Its collection-round limit and
evaluation/checkpoint schedule are:

.. list-table::
   :header-rows: 1
   :widths: 25 35 40

   * - Algorithm
     - Collection Rounds
     - Evaluation and Checkpoints
   * - PPO
     - 3 by default
     - Every 3 rounds and at the final round
   * - GRPO
     - At most 6
     - Every 3 rounds and at the final round

Use a new output directory for each run. Enable the audit for this invocation
when launching PPO:

.. code-block:: bash

   export XVLA_RUN_DIR=/path/to/personal-storage/runs/"${SLURM_JOB_ID}"-ppo-acceptance
   XVLA_ONLINE_ACCEPTANCE=1 bash tools/xvla/train.sh robotwin_ppo_xvla_acceptance

For GRPO, run the corresponding configuration with a separate output directory:

.. code-block:: bash

   export XVLA_RUN_DIR=/path/to/personal-storage/runs/"${SLURM_JOB_ID}"-grpo-acceptance
   XVLA_ONLINE_ACCEPTANCE=1 bash tools/xvla/train.sh robotwin_grpo_xvla_acceptance

For compute-node submission, ``tools/xvla/online_submit.py`` defaults to a
20-minute allocation and accepts ``--time-minutes`` from 1 to 90. Allow
60 minutes for the multi-round GRPO check, including simulation, evaluation,
and checkpoint verification. Repeated ``--override KEY=VALUE`` arguments
pass Hydra settings literally to the training entry; use them to retain the
original environment topology and set the checkpoint and collection limit
when resuming. The login debug entry keeps its separate limit below 25 minutes
per invocation; use Slurm for longer runs.

These configurations set ``env.train.reset_seed_mode=cycle``. Group initial
conditions traverse the original training pool, ``12000``–``12003``, in order
and wrap around. Members of a GRPO group keep the same initial condition.
The evaluation pool, ``13000``–``13003``, remains separate.

A collection round advances ``global_step`` even if GRPO filters out every
group. An empty batch skips the optimizer and scheduler without changing the
model or Adam state; it contributes no effective update. If six rounds do not
produce three effective updates, the acceptance run fails. PPO keeps batches
with no positive reward, but still needs finite, nonzero policy gradients and
an actual parameter change to count as an effective update.

The audit writes ``audit-<pid>.jsonl`` and ``online-acceptance.json`` into
``XVLA_RUN_DIR``. Inspect the summary status and the recorded gradients,
parameter changes, weight synchronization, real-environment evaluation, and
checkpoint checks. A completed collection budget alone does not establish
acceptance.

Resume in a Fresh Process
~~~~~~~~~~~~~~~~~~~~~~~~~

Once the run has produced a resumable checkpoint, start a new process to check
that training continues from the saved state. The native distributed checkpoint
(DCP) saves the complete model, including frozen parameters and buffers, Adam
state, the scheduler, and actor RNG. Audit JSON sidecars save the actor's
training counter and policy version and verify the restored training state.
Worker sidecars preserve the rollout and environment workers' RNG and reset
schedules at episode boundaries. They resume the next episode's
initial-condition selection; they do not restore a simulator in the middle of
an episode.

For a PPO checkpoint saved at ``global_step_3``, activate the same environment
and prepared paths in a new Slurm job. Choose a new, empty output directory and
keep the task, model, and seed pools unchanged:

.. code-block:: bash

   export XVLA_RUN_DIR=/path/to/personal-storage/runs/"${SLURM_JOB_ID}"-ppo-resume
   XVLA_ONLINE_ACCEPTANCE=1 bash tools/xvla/train.sh robotwin_ppo_xvla_acceptance \
     runner.resume_dir=/path/to/original-output/robotwin_ppo_xvla_acceptance/checkpoints/global_step_3 \
     runner.max_epochs=4 runner.max_steps=4

The checkpoint directory contains ``actor/dcp_checkpoint`` and the accompanying
audit sidecars, including ``xvla_worker_state.pt``. Keep them together. Resume
checks compare the restored model, optimizer, scheduler, RNG, and counters.
Acceptance requires at least three cumulative effective optimizer updates,
including at least one new effective update in this process.
The final round also evaluates and saves, so round 4 is checked despite the
three-round interval.

.. warning::

   A zero-update run with empty Adam state records the reason
   ZERO_UPDATE_EMPTY_ADAM_STATE and does not create a resumable checkpoint.
   The standard training entry applies the same protection.
   Use a checkpoint from a run that performed an effective update; a directory
   created by an unsuccessful acceptance run is not sufficient.

Sampling and Updates
~~~~~~~~~~~~~~~~~~~~

Both configurations use ``actor.model.xvla.noise_method=flow_noise``. A bounded,
learnable noise head supplies the standard deviation at every denoising
transition. To select Flow-SDE, pass the override through the same launcher:

.. code-block:: bash

   bash tools/xvla/train.sh robotwin_ppo_xvla \
     actor.model.xvla.noise_method=flow_sde

The policy caches the complete detached 20-D denoising chain and recomputes
its joint log-probability from those same states. It sums over denoising time,
prediction horizon, and raw coordinates. This is the density of an augmented
sample path, including latent variables that do not affect the final executed
prefix; it is not a claimed density for thresholded gripper commands.

Rollout and actor recomputation evaluate each observation and denoising path
with a batch size of one, independently of the environment group or actor
microbatch size. This keeps model and noise-head kernel shapes consistent
before the first update. Gaussian score arithmetic uses float64 before the
joint scalar is converted to float32 for the native RLinf loss contract.
The pre-update joint log-probability check retains its 0.01 absolute tolerance.

RLinf receives that joint scalar in the first action slot of ``[B,C,1]``;
the remaining slots are zero. Keep ``logprob_action_dim=1`` and
``logprob_type=chunk_level`` so the scalar is summed once. An entirely
unexecuted chunk is masked out. The sampled chain remains in the likelihood
when a chunk terminates early.

The reward sums the native per-anchor rewards inside a chunk, and PPO's
``gamma=0.99`` discounts whole chunks. The 300-anchor horizon is a finite
episode limit. Actor and critic loss normalization also counts chunks: a
partially executed final chunk counts as one valid macro step, just like a
fully executed chunk. The episode cap is rounded up to the number of chunks
(10 for 300 anchors and a chunk size of 30). Actual control-step counts still
come from ``executed_mask`` and remain distinct from macro steps.

Evaluation uses ``flow_ode`` with the same clean-to-velocity conversion. Compare
that evaluation protocol separately from the author's native frozen evaluator.

Visualization and Results
-------------------------

Inspect reward, actor and noise-head gradients, likelihood ratios,
``executed_mask``, synchronization, and checkpoint reload before extending the
acceptance configuration. Use :doc:`Training Metrics </rst_source/reference/metrics>`
for the framework metric definitions.

The new training path requires its own GPU rollout and real-update acceptance
record. Historical single-arm LIBERO training and frozen RoboTwin evaluation
do not establish that result. Record each run's actual acceptance status;
these checks establish neither benchmark performance nor Hard-task coverage.

The equivalent GPU test configurations are ``robotwin_xvla_ppo_smoke`` and
``robotwin_xvla_grpo_smoke`` under ``tests/e2e_tests/embodied``. Run them inside
an allocation with the same prepared paths. A successful process exit alone
does not establish an informative GRPO update; inspect the group and gradient
evidence above.
