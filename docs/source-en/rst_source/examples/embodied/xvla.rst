X-VLA
=====

.. figure:: https://robotwin-platform.github.io/assets/images/teaser.png
   :align: center
   :width: 90%

   RoboTwin 2.0 dual-arm tasks (image: `RoboTwin <https://robotwin-platform.github.io>`__).

Run single-GPU PPO or GRPO integration checks for X-VLA on native RoboTwin 2.0.
Start from the pinned RoboTwin checkpoint, prepare the native simulator and
assets, then launch the configuration inside a Slurm allocation. These short
runs validate the training implementation; they are not benchmark results.

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
independent rollouts from a shared initial condition.

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

RLinf receives that joint scalar in the first action slot of ``[B,C,1]``;
the remaining slots are zero. Keep ``logprob_action_dim=1`` and
``logprob_type=chunk_level`` so the scalar is summed once. An entirely
unexecuted chunk is masked out. The sampled chain remains in the likelihood
when a chunk terminates early.

The reward sums the native per-anchor rewards inside a chunk, and PPO's
``gamma=0.99`` discounts whole chunks. The 300-anchor horizon is a finite
episode limit. Evaluation uses ``flow_ode`` with the same clean-to-velocity
conversion. Compare that evaluation protocol separately from the author's
native frozen evaluator.

Visualization and Results
-------------------------

Inspect reward, actor and noise-head gradients, likelihood ratios,
``executed_mask``, synchronization, and checkpoint reload before extending the
one-iteration configuration. Use :doc:`Training Metrics </rst_source/reference/metrics>`
for the framework metric definitions.

The new training path requires its own GPU rollout and real-update acceptance
record. Historical single-arm LIBERO training and frozen RoboTwin evaluation
do not establish that result. Keep those records separate from this run.

The equivalent GPU test configurations are ``robotwin_xvla_ppo_smoke`` and
``robotwin_xvla_grpo_smoke`` under ``tests/e2e_tests/embodied``. Run them inside
an allocation with the same prepared paths. A successful process exit alone
does not establish an informative GRPO update; inspect the group and gradient
evidence above.
