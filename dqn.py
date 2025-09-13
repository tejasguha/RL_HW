#! python3

import argparse
import collections
import random

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np # NOTE only imported because https://github.com/pytorch/pytorch/issues/13918
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim



class ReplayMemory():
    def __init__(self, memory_size, batch_size):
        # define init params
        # use collections.deque
        # BEGIN STUDENT SOLUTION
        self.memory = collections.deque(maxlen = memory_size)
        self.batch_size = batch_size
        # END STUDENT SOLUTION
        pass


    def sample_batch(self):
        # randomly chooses from the collections.deque
        # BEGIN STUDENT SOLUTION
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        return (
            torch.tensor(states, dtype = torch.float32),
            torch.tensor(actions, dtype = torch.float32),
            torch.tensor(rewards, dtype = torch.float32),
            torch.tensor(next_states, dtype = torch.float32),
            torch.tensor(dones, dtype = torch.float32),
        )
        # END STUDENT SOLUTION
        pass


    def append(self, transition):
        # append to the collections.deque
        # BEGIN STUDENT SOLUTION
        self.memory.append(transition)
        # END STUDENT SOLUTION
        pass



class DeepQNetwork(nn.Module):
    def __init__(self, env, state_size, action_size, double_dqn, lr_q_net=2e-4, gamma=0.99, epsilon=0.05, target_update=50, burn_in=10000, replay_buffer_size=50000, replay_buffer_batch_size=32, device='cpu'):
        super(DeepQNetwork, self).__init__()

        # define init params
        self.state_size = state_size
        self.action_size = action_size
        self.double_dqn = double_dqn

        self.gamma = gamma
        self.epsilon = epsilon

        self.target_update = target_update

        self.burn_in = burn_in

        self.device = device

        hidden_layer_size = 256

        # q network
        q_net_init = lambda: nn.Sequential(
            nn.Linear(state_size, hidden_layer_size),
            nn.ReLU(),
            # BEGIN STUDENT SOLUTION
            nn.Linear(hidden_layer_size, hidden_layer_size),
            nn.ReLU(),
            nn.Linear(hidden_layer_size, action_size)
            # END STUDENT SOLUTION
        )

        # initialize replay buffer, networks, optimizer, move networks to device
        # BEGIN STUDENT SOLUTION
        self.q_net = q_net_init().to(device)
        self.target_net = q_net_init().to(device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr_q_net)

        self.replay_buffer = ReplayMemory(replay_buffer_size, replay_buffer_batch_size)
        self.steps = 0

        # END STUDENT SOLUTION


    def forward(self, state, new_state):
        # calculate q value and target
        # use the correct network for the target based on self.double_dqn
        # BEGIN STUDENT SOLUTION
        state = torch.tensor(state, dtype=torch.float32).to(self.device).unsqueeze(0)
        next_state = torch.tensor(new_state, dtype=torch.float32).to(self.device).unsqueeze(0)

        q_values = self.q_net(state)
        with torch.no_grad():
            if not self.double_dqn:
                next_q = torch.max(self.target_net(next_state), dim=1)[0]
        return q_values, next_q
    
        # END STUDENT SOLUTION
    
    def get_action(self, state, stochastic):
        # if stochastic, sample using epsilon greedy, else get the argmax
        # BEGIN STUDENT SOLUTION
        state = torch.tensor(state, dtype=torch.float32).to(self.device).unsqueeze(0)
        if stochastic:
            random_val = random.random()
            if random_val < self.epsilon:
                return random.randint(0, self.action_size - 1)
        else:
            with torch.no_grad():
                return torch.argmax(self.q_net(state)).item()
        # END STUDENT SOLUTION
        pass

    def populate_replay_buffer(self, env):
        # adds in burn-in experience into replay buffer
        state, _ = env.reset()
        for i in range(self.burn_in):
            action = env.action_space.sample()  # purely random
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            self.replay_buffer.append((state, action, reward, next_state, done))
            state = next_state if not done else env.reset()[0]
    
    def train(self, states, actions, rewards, next_states, dones):
        # curr q vals
        q_values = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # computing targets
        with torch.no_grad():
            if not self.double_dqn:
                next_q_vals = self.target_net(next_states).max(1)[0]
            targets = rewards + self.gamma * next_q_vals * (1 - dones)
        
        loss = F.mse_loss(q_values, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # update target network
        self.steps += 1
        if self.steps % self.target_update == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())
    
    def run(self, env, max_steps,):
        state, _ = env.reset()
        for _ in range(max_steps):
            action = self.get_action(state, stochastic=False)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            self.replay_buffer.append((state, action, reward, next_state, done))
            state = next_state if not done else env.reset()[0]





def graph_agents(
    graph_name, mean_undiscounted_returns, test_frequency, max_steps, num_episodes
):
    print(f'Starting: {graph_name}')

    # graph the data mentioned in the homework pdf
    # BEGIN STUDENT SOLUTION
    # END STUDENT SOLUTION

    # plot the total rewards
    xs = [i * test_frequency for i in range(len(average_total_rewards))]
    fig, ax = plt.subplots()
    plt.fill_between(xs, min_total_rewards, max_total_rewards, alpha=0.1)
    ax.plot(xs, average_total_rewards)
    ax.set_ylim(-max_steps * 0.01, max_steps * 1.1)
    ax.set_title(graph_name, fontsize=10)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Average Total Reward')
    fig.savefig(f'./graphs/{graph_name}.png')
    plt.close(fig)
    print(f'Finished: {graph_name}')



def parse_args():
    parser = argparse.ArgumentParser(description='Train an agent.')
    parser.add_argument('--num_runs', type=int, default=5, help='Number of runs to average over for graph')
    parser.add_argument('--num_episodes', type=int, default=1000, help='Number of episodes to train for')
    parser.add_argument('--max_steps', type=int, default=200, help='Maximum number of steps in the environment')
    parser.add_argument('--env_name', type=str, default='CartPole-v1', help='Environment name')
    parser.add_argument(
        "--test_frequency",
        type=int,
        default=100,
        help="Number of training episodes between test episodes",
    )
    parser.add_argument("--double_dqn", action="store_true", help="Use Double DQN")
    return parser.parse_args()



def main():
    args = parse_args()

    # init args, agents, and call graph_agent on the initialized agents
    # BEGIN STUDENT SOLUTION

    env = gym.make(args.env_name)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    # CHANGE LATER
    args.double_dqn = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    agents = []
    for _ in range(args.num_runs):
        agents.append(
            DeepQNetwork(
                state_size=state_size,
                action_size=action_size,
                double_dqn=args.double_dqn,
                device=device
            )
        )

    name = "DQN"
    if args.double_dqn:
        name = "Double DQN"

    # graph_agents(
    #     graph_name=name,
    #     agents=agents,
    #     env=env,
    #     max_steps=args.max_steps,
    #     num_episodes=args.num_episodes,
    #     num_test_episodes=args.num_test_episodes,
    #     graph_every=args.graph_every
    # )

    # END STUDENT SOLUTION



if '__main__' == __name__:
    main()

