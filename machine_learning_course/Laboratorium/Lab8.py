import time
import numpy as np
import gym

def zad12():
    env = gym.make('CartPole-v1')  # utworzenie środowiska
    env.reset()  # reset środowiska do stanu początkowego
    for _ in range(100):  # kolejne kroki symulacji
        env.render()  # renderowanie obrazu
        action = env.action_space.sample()  # wybór akcji (tutaj: losowa akcja)
        # ZADANIE 1
        observation, reward, done, info = env.step(action)  # wykonanie akcji
        # print(f'observation : {observation}')
        # print(f'reward : {reward}')
        # print(f'done : {done}')
        # print(f'info : {info}')
        # print('-'*50)
        # if done:
        #     env.reset()
        # ZADANIE 2
        print(observation[2])
        print(reward)
        print('-' * 50)
        time.sleep(0.5)
    env.close()  # zamknięcie środowiska

def zad4567():
    # ZADANIE 4
    env = gym.make('FrozenLake-v1')  # utworzenie środowiska
    n_observations = env.observation_space.n
    n_actions = env.action_space.n
    Q_table = np.zeros((n_observations, n_actions))
    # ZADANIE 7
    lr = 0.1
    decay_rate = 0.0001
    discount_factor = 0.95
    exploration_proba = 1.0
    min_exploration_proba = 0.1
    # print(Q_table)
    # exit()
    env.reset()  # reset środowiska do stanu początkowego
    for e in range(100000):  # kolejne kroki symulacji
        # env.render()  # renderowanie obrazu
        #action = env.action_space.sample()  # wybór akcji (tutaj: losowa akcja)
    # zadanie5
        current_state = env.reset()
        # print(current_state)
        done = False
        while not done:
            if np.random.uniform(0, 1) < exploration_proba:
                action = env.action_space.sample()  # wybór akcji (tutaj: losowa akcja)
            else:
                action = np.argmax(Q_table[current_state, :])

            next_state, reward, done, _ = env.step(action)
            Q_table[current_state, action] = (1 - lr) * Q_table[current_state, action] + lr * (
                    reward + discount_factor * max(Q_table[next_state, :]))

            current_state = next_state
        exploration_proba = max(min_exploration_proba, np.exp(-decay_rate * e))
    print(Q_table)
  # ZADANIE 6
    env.reset()
    total_episodes = 500
    total_rewards = []
    for i in range(total_episodes):
        done = False
        total_e_reward = 0
        current_state = env.reset()
        while not done:
            action = np.argmax(Q_table[current_state, :])
            next_state, reward, done, _ = env.step(action)
            current_state = next_state
            total_e_reward += reward
        total_rewards.append(total_e_reward)
        env.render()
    print(f'Skuteczność: {str(np.sum(total_rewards)/total_episodes)}')
    env.close()  # zamknięcie środowiska

def main():
    # zad12()
    zad4567()


if __name__ == '__main__':
    main()