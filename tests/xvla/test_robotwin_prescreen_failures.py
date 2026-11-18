from pathlib import Path
from tempfile import TemporaryDirectory
from textwrap import dedent
import unittest

from tools.xvla.robotwin.prescreen_failures import (
    EXPERT_GRASP_POSE_UNAVAILABLE,
    EXPERT_PLAN_FAILED_EMPTY_GRASP,
    expert_failure_reason,
)


def load_fixture(root, relative, source, **globals_):
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dedent(source), encoding="utf-8")
    exec(compile(path.read_text(encoding="utf-8"), str(path), "exec"), globals_)
    return globals_["Fixture"]


def reason(env, task):
    try:
        env.play_once()
    except Exception as error:
        return expert_failure_reason(env, task, error)
    raise AssertionError("fixture must fail during play_once")


class ExpertPrescreenFailuresTest(unittest.TestCase):
    def test_exhausted_native_grasp_actor_and_unrelated_move(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            action = load_fixture(root, "envs/utils/action.py", '''
                class Fixture:
                    def __init__(self, action, target_pose):
                        assert target_pose is not None, "target_pose cannot be None for move action."
            ''')
            env = load_fixture(root, "envs/_base_task.py", '''
                class Fixture:
                    def play_once(self):
                        self.grasp_actor()

                    def grasp_actor(self):
                        pre_grasp_pose, grasp_pose = None, self.grasp_pose
                        Action("move", pre_grasp_pose)
            ''', Action=action)()
            env.grasp_pose = None
            self.assertEqual(reason(env, "turn_switch"), EXPERT_GRASP_POSE_UNAVAILABLE)
            env.grasp_pose = [1.0]
            self.assertIsNone(reason(env, "turn_switch"))

    def test_alarmclock_only_excludes_the_exact_unguarded_grasp_getter(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            env = load_fixture(root, "envs/click_alarmclock.py", '''
                class Fixture:
                    alarm = None

                    def get_grasp_pose(self, *args, **kwargs):
                        return None

                    def play_once(self):
                        arm_tag = "left"
                        if self.no_pose:
                            return self.get_grasp_pose(self.alarm, pre_dis=0.1, contact_point_id=0, arm_tag=arm_tag)[:3]
                        return self.alarm[:3]
            ''')()
            env.no_pose = True
            self.assertEqual(reason(env, "click_alarmclock"), EXPERT_GRASP_POSE_UNAVAILABLE)
            self.assertIsNone(reason(env, "lift_pot"))
            env.no_pose = False
            self.assertIsNone(reason(env, "click_alarmclock"))

    def test_dustbin_empty_grasp_requires_the_known_site_and_failed_plan(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            env = load_fixture(root, "envs/put_bottles_dustbin.py", '''
                class Fixture:
                    def grasp_actor(self, *args, **kwargs):
                        return self.grasp

                    def play_once(self):
                        bottle, arm_tag, delta_dis = None, "right", 0.06
                        if self.side == "right":
                            right_action = self.grasp_actor(bottle, arm_tag=arm_tag, pre_grasp_dis=0.1)
                            right_action[1][0].target_pose[2] += delta_dis
                        elif self.side == "left":
                            left_action = self.grasp_actor(bottle, arm_tag="left", pre_grasp_dis=0.1)
                            left_action[1][0].target_pose[2] -= delta_dis
                        else:
                            self.grasp[1][0].target_pose[2] -= delta_dis
            ''')()
            env.plan_success = False
            env.grasp = (None, [])
            for side in ("left", "right"):
                with self.subTest(side=side):
                    env.side = side
                    self.assertEqual(reason(env, "put_bottles_dustbin"), EXPERT_PLAN_FAILED_EMPTY_GRASP)
            self.assertIsNone(reason(env, "lift_pot"))
            env.plan_success = True
            self.assertIsNone(reason(env, "put_bottles_dustbin"))
            env.plan_success = False
            env.grasp = ("right", [])
            self.assertIsNone(reason(env, "put_bottles_dustbin"))
            env.grasp = (None, [])
            env.side = "other"
            self.assertIsNone(reason(env, "put_bottles_dustbin"))
