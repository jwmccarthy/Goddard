import argparse
import tempfile
import unittest
from pathlib import Path

import torch

from training.arguments import add_ppo_arguments
from training.observations import observation_size, resolve_observation_arguments
from training.setup import build_env, prepare


def arguments(*options):
    parser = argparse.ArgumentParser()
    add_ppo_arguments(parser)
    return parser.parse_args(list(options))


class ObservationSchemaTests(unittest.TestCase):
    def test_new_defaults_and_explicit_controls(self):
        for controls in (False, True):
            args = arguments(*(["--previous-controls"] if controls else []))
            self.assertEqual(resolve_observation_arguments(args), ("ego-controller-v3", controls))
            self.assertEqual(observation_size(args.observation_schema, 2, controls), 144 + 8 * controls)

    def test_resume_inherits_layout_and_rejects_switching(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            for schema, controls in (("base-v1", False), ("ego-controller-v1", False),
                                     ("ego-controller-v3", False), ("ego-controller-v3", True)):
                config = {"observation_schema": schema, "previous_controls": controls}
                torch.save({"config": config}, path)
                args = arguments("--resume", str(path))
                self.assertEqual(resolve_observation_arguments(args), (schema, controls))
                args = arguments("--resume", str(path), "--observation-schema",
                                 "ego-controller-v1" if schema == "ego-controller-v3" else "ego-controller-v3")
                with self.assertRaises(ValueError):
                    resolve_observation_arguments(args)
                if schema == "ego-controller-v3":
                    args = arguments("--resume", str(path),
                                     "--no-previous-controls" if controls else "--previous-controls")
                    with self.assertRaisesRegex(ValueError, "previous-controls"):
                        resolve_observation_arguments(args)
            torch.save({"config": {}}, path)
            self.assertEqual(resolve_observation_arguments(arguments("--resume", str(path))),
                             ("base-v1", False))

    def test_legacy_checkpoints_cannot_use_merged_native_layout(self):
        import carl

        if getattr(carl, "OBS_CONTROLLER_VERSION", None) != 3:
            self.skipTest("requires merged CARL")
        for schema in ("base-v1", "ego-controller-v1"):
            with self.assertRaisesRegex(RuntimeError, "legacy"):
                build_env(arguments("--observation-schema", schema))


@unittest.skipUnless(torch.cuda.is_available(), "requires GPU CARL")
class NativeObservationTests(unittest.TestCase):
    def setUp(self):
        import carl

        if getattr(carl, "OBS_CONTROLLER_VERSION", None) != 3:
            self.skipTest("requires merged CARL")

    def test_layout_prefix_masks_and_terminal_state(self):
        for team in (1, 2, 3):
            envs = []
            for controls in (False, True):
                args = arguments("--n-sim", "1", "--team-size", str(team),
                                 "--frameskip", "4", "--no-touch-timeout", str(4 / 120),
                                 *(["--previous-controls"] if controls else []))
                env = build_env(args)
                self.addCleanup(env.close)
                envs.append(env)
            plain, extended = [env.reset() for env in envs]
            self.assertEqual(plain.shape[-1], 90 + 54 * team)
            torch.testing.assert_close(plain.as_subclass(torch.Tensor), extended.as_subclass(torch.Tensor)[:, :-8])
            torch.testing.assert_close(envs[0].action_codec.mask(plain), envs[1].action_codec.mask(extended))
            action = torch.ones((2 * team, 7), dtype=torch.int32, device="cuda")
            transitions = [env.step(action) for env in envs]
            for observation, _, _, truncated, info in transitions:
                self.assertTrue(truncated.all())
                self.assertEqual(observation.ego_controller.shape[-1], 5)
                self.assertEqual(info["final_obs"].shape, observation.shape)
            torch.testing.assert_close(transitions[0][4]["final_obs"].as_subclass(torch.Tensor),
                                       transitions[1][4]["final_obs"].as_subclass(torch.Tensor)[:, :-8])
            self.assertTrue(transitions[1][4]["final_obs"].ego_previous_controls[:, 5].bool().all())
            self.assertFalse(transitions[1][0].ego_previous_controls.bool().any())

    def test_new_policy_update_and_checkpoint_layout_guard(self):
        from training.checkpoints import load_checkpoint, save_checkpoint

        for controls in (False, True):
            args = arguments("--n-sim", "1", "--rollout", "2", "--hidden-size", "16",
                             "--layers", "1", "--no-bf16", "--no-compile",
                             *(["--previous-controls"] if controls else []))
            env, agent, optimizer, rollout = prepare(args)
            self.addCleanup(env.close)
            observation = env.reset().as_subclass(torch.Tensor)
            action, logprob, entropy, value = agent.get_action_and_value(observation)
            loss = -logprob.mean() - .01 * entropy.mean() + value.square().mean()
            loss.backward()
            optimizer.step()
            self.assertTrue(torch.isfinite(loss))
            env.step(action)
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "new.pt"
                save_checkpoint(path, "ppo", args, agent, optimizer, 2)
                self.assertEqual(load_checkpoint(path, "ppo", args, agent, optimizer)["step"], 2)
                # A copied checkpoint must not follow its old training resume path.
                payload = torch.load(path, map_location="cpu", weights_only=False)
                payload["config"]["resume"] = Path(directory) / "missing-ancestor.pt"
                torch.save(payload, path)
                from vis.playback import load_policy

                viewer, policy, config, step = load_policy(path, 1)
                self.addCleanup(viewer.close)
                self.assertEqual(step, 2)
                self.assertEqual(viewer.single_observation_space.shape, (144 + 8 * controls,))
                for actual, expected in zip(policy.actor.parameters(), agent.actor.parameters()):
                    torch.testing.assert_close(actual, expected)
                args.previous_controls = not controls
                with self.assertRaisesRegex(ValueError, "previous-controls"):
                    load_checkpoint(path, "ppo", args, agent, optimizer)


if __name__ == "__main__":
    unittest.main()
