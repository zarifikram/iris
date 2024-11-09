import numpy as np
from tensordict import TensorDict
from torchrl.envs.transforms import ToTensorImage, TransformedEnv, Compose, Resize
from torchrl.envs.libs.dm_control import DMControlEnv
from torchrl.envs import ParallelEnv

def build_single_env(env_name, task_name, image_size, seed, action_repeat):
    env = DMControlEnv(env_name, task_name, from_pixels=True, pixels_only=True, frame_skip=action_repeat)
    env = TransformedEnv(env, Compose(ToTensorImage(), Resize(image_size, image_size)))
    env.set_seed(seed)
    return env

def build_vec_env(env_list, task_list, image_size, num_envs, seed, action_repeat):
    # lambda pitfall refs to: https://python.plainenglish.io/python-pitfalls-with-variable-capture-dcfc113f39b7
    assert num_envs % len(env_list) == 0
    env_fns = []
    vec_env_names = []
    vec_task_names = []
    for env_name, task_name in zip(env_list, task_list):
        def lambda_generator(env_name, task_name, image_size):
            return lambda: build_single_env(env_name, task_name, image_size, seed, action_repeat)
        env_fns += [lambda_generator(env_name, task_name, image_size) for i in range(num_envs//len(env_list))]
        vec_env_names += [env_name for i in range(num_envs//len(env_list))]
        vec_task_names += [task_name for i in range(num_envs//len(env_list))]
    vec_env = ParallelEnv(num_envs, env_fns)
    return vec_env, vec_env_names, vec_task_names

if __name__ == "__main__":
    # list availbale envs
    # for env_name, tasks in DMControlEnv.available_envs:
    #     print(f"{env_name}: {tasks}")
    # we need vectorized env for training
    env_names = ["cheetah", "finger", "walker", "ball_in_cup"]
    task_names = ["run", "spin", "walk", "catch"]
    vec_env, vec_env_names, vec_task_names = build_vec_env(env_names, task_names, 64, 8, seed=42)
    current_obs = vec_env.reset()
    action = vec_env.action_spec
    # print(f"current obs {current_obs}")

    # env = DeepMindControl("cheetah_run")
    # obs = env.reset()
    # # show env using cv2
    import cv2
    # for i in range(20):
    #     action = vec_env.action_spec.rand()
    #     tensor_dict_action = TensorDict({"action": action}, batch_size=8)
    #     out = vec_env.step(tensordict=tensor_dict_action)
    #     pixels = out["next"]["pixels"]
    #     cv2.imshow("cheetah", pixels[0].numpy().transpose(1, 2, 0))
    #     cv2.imshow("finger", pixels[2].numpy().transpose(1, 2, 0))
    #     cv2.imshow("walker", pixels[4].numpy().transpose(1, 2, 0))
    #     cv2.imshow("ball_in_cup", pixels[6].numpy().transpose(1, 2, 0))
    #     cv2.waitKey(0)
    cv2.destroyAllWindows()

    vec_env.close()
