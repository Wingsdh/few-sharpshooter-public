# -*- encoding: utf-8 -*-
"""
-------------------------------------------------
   File Name：    dqn.py
   Description :
   Author :       Wings DH
   Time：         6/19/21 4:03 PM
-------------------------------------------------
   Change Activity:
                   6/19/21: Create
-------------------------------------------------
"""
from collections import defaultdict
import os
from enum import Enum
import random
from typing import Any
import numpy as np
import tensorflow as tf

from tf_agents.agents.dqn import dqn_agent
from tf_agents.environments import py_environment, tf_py_environment
from tf_agents.networks import q_network
from tf_agents.policies import random_tf_policy
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts, trajectory
from tf_agents.utils import common

from modeling.retriever_classifier import SentenceRetriever

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


class Action(Enum):
    ACTION_DRAW = 0
    ACTION_STOP = 1


class FewShotEnv(py_environment.PyEnvironment):

    def get_info(self):
        pass

    def get_state(self) -> Any:
        return self._state

    def set_state(self, state: Any) -> None:
        self._state = state

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def _reset(self):
        # 通过向量检索获取检索信息
        self.ind += 1
        if self.ind >= len(self.dev_sentences):
            self.ind = 0
        self.sentence = self.dev_sentences[self.ind]
        vec = self.dev_matrix[self.ind]
        self.most_sim_texts, self.scores = self.sent_retriever.retrieve(vec=vec)
        self.label_2_score = defaultdict(float)
        self.next_ind = 1
        self.observation = np.zeros((len(self.sent_retriever.data),), dtype=np.float)
        ind, score = self.draw()
        self.observation[ind] = score

        self._state = 0
        self._episode_ended = False
        return ts.restart(np.array(self.observation, dtype=np.float))

    def draw(self):
        if len(self.scores) > self.next_ind:
            ind = self.next_ind
            sentence = self.most_sim_texts[ind]
            position = self.sent_retriever.text_2_position[sentence.text]
            self.next_ind += 1
            self.label_2_score[sentence.label] += self.scores[ind]
            return position, self.scores[ind]

        return None, None

    def _step(self, action):
        if self._episode_ended:
            # The last action ended the episode. Ignore the current action and start
            # a new episode.
            return self.reset()

        if action == Action.ACTION_DRAW.value:
            ind, score = self.draw()
            if ind is None or score is None:
                label = max(self.label_2_score.keys(), key=lambda x: self.label_2_score[x])
                reward = 10 if label == self.sentence.label else -10
                self._episode_ended = True
                return ts.termination(np.array(self.observation, dtype=np.float), reward)
            else:
                self.observation[ind] = score
                return ts.transition(np.array(self.observation, dtype=np.float), 0)

        else:
            self._episode_ended = True
            label = max(self.label_2_score.keys(), key=lambda x: self.label_2_score[x])
            reward = 10 if label == self.sentence.label else -10

            # print(f'<{label}> and <{self.sentence.label}> {reward}')
            return ts.termination(np.array(self.observation, dtype=np.float), reward)

    def __init__(self, sr: SentenceRetriever, dev_sentences, code_2_label):
        self.ind = 0
        # 断言类别 * n_class + 去除上一条信息 * 1 + 翻出下一张 * 1
        self.n_action = 2
        self.code_2_label = code_2_label

        self._action_spec = array_spec.BoundedArraySpec(
            shape=(), dtype=np.int32, minimum=0, maximum=self.n_action - 1, name='action')
        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(len(sr.data),), dtype=np.float, minimum=0, name='observation')

        self._state = 0
        self._episode_ended = False
        self.sent_retriever = sr
        self.dev_sentences = dev_sentences
        self.dev_matrix = [self.sent_retriever.encoder.encode(s.text) for s in self.dev_sentences]


class ClassFewShotEnv(py_environment.PyEnvironment):

    def get_info(self):
        pass

    def get_state(self) -> Any:
        return self._state

    def set_state(self, state: Any) -> None:
        self._state = state

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def _reset(self):
        # 通过向量检索获取检索信息
        self.ind += 1
        if self.ind >= len(self.dev_sentences):
            self.ind = 0
        self.sentence = self.dev_sentences[self.ind]
        vec = self.dev_matrix[self.ind]
        self.most_sim_texts, self.scores = self.sent_retriever.retrieve(vec=vec)
        self.label_2_score = defaultdict(float)
        self.next_ind = 1
        self.observation = np.zeros((len(self.sent_retriever.data),), dtype=np.float)
        ind, score = self.draw()
        self.observation[ind] = score

        self._state = 0
        self._episode_ended = False
        return ts.restart(np.array(self.observation, dtype=np.float))

    def draw(self):
        if len(self.scores) > self.next_ind:
            ind = self.next_ind
            sentence = self.most_sim_texts[ind]
            position = self.sent_retriever.text_2_position[sentence.text]
            self.next_ind += 1
            self.label_2_score[sentence.label] += self.scores[ind]
            return position, self.scores[ind]

        return None, None

    def _step(self, action):
        if self._episode_ended:
            # The last action ended the episode. Ignore the current action and start
            # a new episode.
            return self.reset()

        if action == Action.ACTION_DRAW.value:
            ind, score = self.draw()
            if ind is None or score is None:
                label = max(self.label_2_score.keys(), key=lambda x: self.label_2_score[x])
                reward = 10 if label == self.sentence.label else -10
                self._episode_ended = True
                return ts.termination(np.array(self.observation, dtype=np.float), reward)
            else:
                self.observation[ind] = score
                return ts.transition(np.array(self.observation, dtype=np.float), 0)
        else:
            code = action - 1
            label = self.code_2_label[code]
            reward = 10 if label == self.sentence.label else -10
            self._episode_ended = True
            return ts.termination(np.array(self.observation, dtype=np.float), reward)

    def __init__(self, sr: SentenceRetriever, dev_sentences, code_2_label):
        self.ind = 0
        # 断言类别 * n_class + 去除上一条信息 * 1 + 翻出下一张 * 1
        self.code_2_label = code_2_label
        self.n_action = len(self.code_2_label) + 1

        self._action_spec = array_spec.BoundedArraySpec(
            shape=(), dtype=np.int32, minimum=0, maximum=self.n_action - 1, name='action')
        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(len(sr.data),), dtype=np.float, minimum=0, name='observation')

        self._state = 0
        self._episode_ended = False
        self.sent_retriever = sr
        self.dev_sentences = dev_sentences
        self.dev_matrix = [self.sent_retriever.encoder.encode(s.text) for s in self.dev_sentences]


collect_steps_per_iteration = 100
replay_buffer_capacity = 500

fc_layer_params = (100, )

batch_size = 128
learning_rate = 1e-3
log_interval = 5

num_eval_episodes = 10
eval_interval = 100
initial_collect_steps = 100  # @param {type:"integer"}


def train_process(train_py_env, eval_py_env):
    train_env = tf_py_environment.TFPyEnvironment(train_py_env)
    eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)
    q_net = q_network.QNetwork(
        train_env.observation_spec(),
        train_env.action_spec(),
        fc_layer_params=fc_layer_params)

    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)

    global_step = tf.compat.v1.train.get_or_create_global_step()

    agent = dqn_agent.DqnAgent(
        train_env.time_step_spec(),
        train_env.action_spec(),
        q_network=q_net,
        optimizer=optimizer,
        td_errors_loss_fn=common.element_wise_squared_loss,
        train_step_counter=global_step)
    agent.initialize()
    random_policy = random_tf_policy.RandomTFPolicy(train_env.time_step_spec(),
                                                    train_env.action_spec())
    replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
        data_spec=agent.collect_data_spec,
        batch_size=train_env.batch_size,
        max_length=replay_buffer_capacity)

    def collect_step(environment, policy, buffer):
        time_step = environment.current_time_step()
        action_step = policy.action(time_step)
        next_time_step = environment.step(action_step.action)
        traj = trajectory.from_transition(time_step, action_step, next_time_step)

        # Add trajectory to the replay buffer
        buffer.add_batch(traj)

    def collect_data(env, policy, buffer, steps):
        for _ in range(steps):
            collect_step(env, policy, buffer)

    collect_data(train_env, random_policy, replay_buffer, initial_collect_steps)

    # Dataset generates trajectories with shape [BxTx...] where
    # T = n_step_update + 1.
    dataset = replay_buffer.as_dataset(
        num_parallel_calls=3, sample_batch_size=batch_size,
        num_steps=2).prefetch(3)

    iterator = iter(dataset)
    # (Optional) Optimize by wrapping some of the code in a graph using TF function.
    agent.train = common.function(agent.train)

    def compute_avg_return(environment, policy, num_episodes=10):

        total_return = 0.0
        for _ in range(num_episodes):

            time_step = environment.reset()
            episode_return = 0.0

            while not time_step.is_last():
                action_step = policy.action(time_step)
                time_step = environment.step(action_step.action)
                episode_return += time_step.reward
            total_return += episode_return

        avg_return = total_return / num_episodes
        return avg_return.numpy()[0]

    # (Optional) Optimize by wrapping some of the code in a graph using TF function.
    agent.train = common.function(agent.train)

    # Reset the train step
    agent.train_step_counter.assign(0)
    avg_return = compute_avg_return(eval_env, agent.policy, num_eval_episodes)
    returns = [avg_return]
    for step in range(100000):
        collect_data(train_env, agent.collect_policy, replay_buffer, collect_steps_per_iteration)

        experience, unused_info = next(iterator)
        train_loss = agent.train(experience).loss

        iteration = agent.train_step_counter.numpy()

        if step % 100 == 0:
            print('iteration: {0} loss: {1}'.format(iteration, train_loss))

        if step % eval_interval == 0:
            avg_return = compute_avg_return(eval_env, agent.policy, num_eval_episodes)
            print('step = {0}: Average Return = {1}'.format(step, avg_return))
            returns.append(avg_return)
