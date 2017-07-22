""" 
  Solving Frozen-Lake environment in Open-AI gym
  Author: malzantot
"""

import gym
import random
import time
import numpy as np

from gym import wrappers

def run_episode(env, policy, render=False):
    """ Runs an episode and return the total reward """
    obs = env.reset()
    total_reward = 0
    for _ in range(100):
        if render:
            env.render()
        obs, reward, done , _ = env.step(policy[obs])
        total_reward += reward
        if done:
            break
    return total_reward

def evaluate_policy(env, policy, n=100):
    scores = [run_episode(env, policy, False) for _ in range(n)]
    return np.mean(scores)

def random_policy():
    return [np.random.choice(4) for _ in range(16)]

def mutate(policy, prob):
    mutate_policy = [x for x in policy]
    for i in range(len(policy)):
        if random.random() < prob:
            mutate_policy[i] = np.random.choice(4)
    return mutate_policy

def crossover(policy1, policy2):
    new_policy = [x for x in policy1]
    for i in range(len(policy1)):
        if random.random() < 0.5:
            new_policy[i] = policy2[i]
    return new_policy

if __name__ == '__main__':
    env = gym.make('FrozenLake-v0')
    n_policy = 40
    n_mutate = 10
    n_crossover = 10
    policy_pool = [random_policy() for _ in range(n_policy)]

    n_generations = 50
    start = time.time()
    for gen_idx in range(n_generations):
        policy_scores = [evaluate_policy(env, p) for p in policy_pool]
        print('Generation %4d - Min score = %4.4f, Max score = %4.4f'
                %(gen_idx+1, np.min(policy_scores),
            np.max(policy_scores)))
        policy_rank = np.argsort(policy_scores)
        new_pool = [policy_pool[x] for x in policy_rank[-20:]]
        # cross over
        crossovers = [crossover(random.choice(new_pool), random.choice(new_pool)) for _ in range(n_crossover)]
        mutates = [mutate(random.choice(new_pool), 0.03) for _ in range(n_mutate)]
        # mutation
        new_pool += crossovers
        new_pool += mutates
        policy_pool = new_pool

    end = time.time()
    time_taken = end - start
    print('Time taken = %4.4f\n' %time_taken)

    policy_scores = [evaluate_policy(env,p) for p in policy_pool]
    best_policy = policy_pool[np.argmax(policy_scores)]
    env = wrappers.Monitor(env, '/tmp/FrozenLake-v0', force=True)
    run_episode(env, best_policy, True)
    env.close()

