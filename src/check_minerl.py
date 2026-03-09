import gym
import minerl

print("registered MineRL environments:")

registry = gym.envs.registry

# handle different gym versions safely
if hasattr(registry, "values"):
    specs = registry.values()
elif hasattr(registry, "all"):
    specs = registry.all()
else:
    specs = list(registry)

for spec in specs:
    try:
        env_id = spec.id
    except AttributeError:
        continue

    if "MineRL" in env_id:
        print(env_id)