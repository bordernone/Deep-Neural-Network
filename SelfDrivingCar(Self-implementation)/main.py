import torch
import numpy as np
import random
import os


class Model(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size

        self.fc1 = torch.nn.Linear(input_size, 30)
        self.fc2 = torch.nn.Linear(30, output_size)

    def forward(self, state):
        x = self.fc1(state)
        x = torch.nn.functional.relu(x)
        return self.fc2(x)


class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.last_states = []
        self.new_states = []
        self.actions = []
        self.rewards = []

    def push(self, last_state, new_state, action, reward):
        assert len(self.last_states) == len(self.new_states) == len(self.actions) == len(self.rewards)

        if len(self.last_states) > self.capacity:
            del self.last_states[0]

        if len(self.new_states) > self.capacity:
            del self.new_states[0]

        if len(self.actions) > self.capacity:
            del self.actions[0]

        if len(self.rewards) > self.capacity:
            del self.rewards[0]

        self.last_states.append(last_state)
        self.new_states.append(new_state)
        self.actions.append(action)
        self.rewards.append(reward)

    def get_sample(self, sample_size):
        assert len(self.last_states) >= sample_size
        indices = random.sample(range(0, len(self.last_states)), sample_size)

        last_states = np.asarray([self.last_states[i] for i in indices])
        new_states = np.asarray([self.new_states[i] for i in indices])
        actions = np.asarray([self.actions[i] for i in indices])
        rewards = np.asarray([self.rewards[i] for i in indices])

        actions = np.reshape(actions, (sample_size, 1))
        rewards = np.reshape(rewards, (sample_size, 1))

        return last_states, new_states, actions, rewards

    def get_size(self):
        return len(self.last_states)


class Dqn:
    def __init__(self, input_size, output_size, gamma):
        self.gamma = gamma
        self.model = Model(input_size, output_size)
        self.memory = ReplayMemory(10000)

        self.last_state = [0, 0, 0, 0, 0]
        self.last_action = 0
        self.reward_window = []

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

    def learn(self, last_state_batch, new_state_batch, action_batch, reward_batch):
        action_batch_tensor = torch.tensor(action_batch, dtype=torch.int64)
        reward_batch_tensor = torch.Tensor(reward_batch).squeeze(1)

        with torch.no_grad():
            new_state_batch_tensor = torch.Tensor(new_state_batch)
            next_state_q_values = self.model(new_state_batch_tensor)
            targets = torch.max(next_state_q_values, 1)[0] * self.gamma + reward_batch_tensor

        with torch.enable_grad():
            last_state_batch_tensor = torch.Tensor(last_state_batch).requires_grad_(True)
            predictions = self.model(last_state_batch_tensor)
            predictions_along_action = torch.gather(predictions, 1, action_batch_tensor).squeeze(1)

        loss = torch.nn.functional.smooth_l1_loss(predictions_along_action, targets)
        self.optimizer.zero_grad()
        loss.backward(retain_graph=True)
        self.optimizer.step()

    # Expects state = [1, 2, 3, 4, 5]
    def select_action(self, state):
        state = np.asarray(state).reshape((1, 5))

        with torch.no_grad():
            state_tensor = torch.Tensor(state)
            q_values = self.model(state_tensor) * 100

        probabilities = torch.nn.functional.softmax(q_values, dim=1)
        return probabilities.multinomial(num_samples=1).item()

    def update(self, last_reward, new_signal):
        new_action = self.select_action(new_signal)
        self.memory.push(self.last_state, new_signal, self.last_action, last_reward)

        learn_batch_size = 100
        if self.memory.get_size() > learn_batch_size:
            train_last_states, train_new_states, train_actions, train_rewards = self.memory.get_sample(learn_batch_size)
            self.learn(train_last_states, train_new_states, train_actions, train_rewards)

        self.last_state = new_signal
        self.last_action = new_action

        self.reward_window.append(last_reward)
        if len(self.reward_window) > 1000:
            del self.reward_window[0]

        return new_action

    def score(self):
        return sum(self.reward_window) / (len(self.reward_window) + 1.)

    def save(self):
        torch.save({'state_dict': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    }, 'last_brain.pth')

    def load(self):
        if os.path.isfile('last_brain.pth'):
            print("=> loading checkpoint... ")
            checkpoint = torch.load('last_brain.pth')
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            print("done !")
        else:
            print("no checkpoint found...")
