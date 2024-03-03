import metaworld
import random


ml10 = metaworld.ML10() # Construct the benchmark, sampling tasks

training_envs = []
for name, env_cls in ml10.train_classes.items():
  env = env_cls()
  task = random.choice([task for task in ml10.train_tasks
                        if task.env_name == name])
  env.set_task(task)
  training_envs.append(env)

for env in training_envs:
  env.render_mode = "human"
  obs = env.reset()  # Reset environment
  a = env.action_space.sample()  # Sample an action
  obs, reward, done, _, info = env.step(a)  # Step the environment with the sampled random action

# print(metaworld.ML1.ENV_NAMES)  # Check out the available environments

# ml1 = metaworld.ML1('pick-place-v2') # Construct the benchmark, sampling tasks

# env = ml1.train_classes['pick-place-v2']()  # Create an environment with task `pick_place`
# task = random.choice(ml1.train_tasks)
# env.set_task(task)  # Set task
# env.render_mode = "human"

# for _ in range(100):
#     obs = env.reset()  # Reset environment
#     while True:
#         a = env.action_space.sample()  # Sample an action
#         obs, reward, termination, truncation, info = env.step(a)  # Step the environment with the sampled random action
#         if truncation == True:
#             print("done")
#             break

