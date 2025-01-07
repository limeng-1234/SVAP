import warnings
from typing import Any, ClassVar, Dict, List, Optional, Tuple, Type, TypeVar, Union

import numpy as np
import torch as th
from gymnasium import spaces
from torch.nn import functional as F

from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import get_linear_fn, get_parameters_by_name, polyak_update
from stable_baselines3.dqn.policies import CnnPolicy, DQNPolicy, MlpPolicy, MultiInputPolicy, QNetwork

import torch
import torch.nn as nn
SelfDQN = TypeVar("SelfDQN", bound="DQN")

def gradient_method(model, baselines, input_tensor):
    """
    计算梯度显著图（Gradient Saliency），支持批量输入，并保留梯度信息以进行后续优化。

    参数:
        model (nn.Module): PyTorch模型。
        input_tensor (torch.Tensor): 输入张量，形状为 (batch_size, n_features)。

    返回:
        feature_importance (torch.Tensor): 特征重要性，形状为 (batch_size, output_dim, n_features)。
        gradients (torch.Tensor): 原始梯度值，形状为 (batch_size, output_dim, n_features)。
    """
    # 确保输入张量需要梯度
    input_tensor = input_tensor.clone().detach().requires_grad_(True)

    # 前向传播
    output = model(input_tensor)  # 输出形状: (batch_size, output_dim)

    batch_size, output_dim = output.shape
    n_features = input_tensor.shape[1]

    # 初始化梯度张量
    gradients = torch.zeros(batch_size, output_dim, n_features, device=input_tensor.device)

    # 使用 torch.autograd.grad 计算梯度，保持计算图连贯
    for i in range(output_dim):
        # 对每个输出类别，计算梯度
        grad_output = output[:, i].sum()  # 将当前类别的所有样本输出求和，得到一个标量

        # 计算梯度
        grad = torch.autograd.grad(
            outputs=grad_output,
            inputs=input_tensor,
            retain_graph=True,
            create_graph=True
        )[0]  # grad 的形状: (batch_size, n_features)

        gradients[:, i, :] = grad
    return gradients * (input_tensor-baselines).unsqueeze(1)
def compute_attributions(model, reference_input, actual_inputs):
    # 前向传播得到输出
    all_attri = []
    for i in range(actual_inputs.size(0)):
        actual_input = actual_inputs[i:i+1, :]
        def forward_pass(model, x):
            inputs = []
            outputs = []
            for layer in model.children():
                inputs.append(x)
                x = layer(x)
                outputs.append(x)
            return inputs, outputs

        # Perform forward passes for both reference and actual inputs
        ref_inputs, ref_outputs = forward_pass(model, reference_input)
        act_inputs, act_outputs = forward_pass(model, actual_input)
        # 计算输入差异
        delta_input = actual_input - reference_input
        # 初始化归因
        attributions = delta_input.clone()
        # 逐层计算归因
        layer_idx = 0
        for layer in model.children():
            if isinstance(layer, nn.Linear):
                weight = layer.weight
                # 线性层的权重
                if layer_idx == 0:
                    # 更新归因
                    attributions = torch.matmul(weight, torch.diag_embed(attributions.float()))
                else:
                    attributions = torch.matmul(weight, attributions)
            elif isinstance(layer, nn.ReLU):
                # 计算非线性层的输入和输出差异
                delta_input_relu = act_inputs[layer_idx] - ref_inputs[layer_idx]
                delta_output_relu = act_outputs[layer_idx] - ref_outputs[layer_idx]

                # 计算非线性层的乘子
                # relu_multiplier = torch.where(
                #     delta_input_relu != 0,
                #     delta_output_relu / delta_input_relu,
                #     torch.tensor(1.0).to('cuda:0')  # 当输入差异为 0 时，乘子为 1
                # )
                # relu_multiplier = torch.where(
                #     torch.abs(delta_input_relu) < 0.001,  # 条件：绝对值小于 0.001
                #     torch.tensor(1.0).to('cuda:0'),  # 当条件为真时，乘子为 1
                #     delta_output_relu / delta_input_relu  # 否则，使用 delta_output_relu / delta_input_relu
                # )
                relu_multiplier = torch.where(
                    torch.abs(delta_input_relu) < 0.001,  # 条件：绝对值小于 0.001
                    torch.tensor(1.0).to('cuda:0'),  # 当条件为真时，乘子为 1
                    delta_output_relu / (delta_input_relu + 1e-8)  # 防止除以零，增加一个小常数
                )

                # 更新归因
                attributions = torch.matmul(torch.diag_embed(relu_multiplier.float()), attributions.float())
            layer_idx = layer_idx + 1
        all_attri.append(torch.mean(attributions, dim=0))
    return all_attri

class DQN(OffPolicyAlgorithm):
    """
    Deep Q-Network (DQN)

    Paper: https://arxiv.org/abs/1312.5602, https://www.nature.com/articles/nature14236
    Default hyperparameters are taken from the Nature paper,
    except for the optimizer and learning rate that were taken from Stable Baselines defaults.

    :param policy: The policy model to use (MlpPolicy, CnnPolicy, ...)
    :param env: The environment to learn from (if registered in Gym, can be str)
    :param learning_rate: The learning rate, it can be a function
        of the current progress remaining (from 1 to 0)
    :param buffer_size: size of the replay buffer
    :param learning_starts: how many steps of the model to collect transitions for before learning starts
    :param batch_size: Minibatch size for each gradient update
    :param tau: the soft update coefficient ("Polyak update", between 0 and 1) default 1 for hard update
    :param gamma: the discount factor
    :param train_freq: Update the model every ``train_freq`` steps. Alternatively pass a tuple of frequency and unit
        like ``(5, "step")`` or ``(2, "episode")``.
    :param gradient_steps: How many gradient steps to do after each rollout (see ``train_freq``)
        Set to ``-1`` means to do as many gradient steps as steps done in the environment
        during the rollout.
    :param replay_buffer_class: Replay buffer class to use (for instance ``HerReplayBuffer``).
        If ``None``, it will be automatically selected.
    :param replay_buffer_kwargs: Keyword arguments to pass to the replay buffer on creation.
    :param optimize_memory_usage: Enable a memory efficient variant of the replay buffer
        at a cost of more complexity.
        See https://github.com/DLR-RM/stable-baselines3/issues/37#issuecomment-637501195
    :param target_update_interval: update the target network every ``target_update_interval``
        environment steps.
    :param exploration_fraction: fraction of entire training period over which the exploration rate is reduced
    :param exploration_initial_eps: initial value of random action probability
    :param exploration_final_eps: final value of random action probability
    :param max_grad_norm: The maximum value for the gradient clipping
    :param stats_window_size: Window size for the rollout logging, specifying the number of episodes to average
        the reported success rate, mean episode length, and mean reward over
    :param tensorboard_log: the log location for tensorboard (if None, no logging)
    :param policy_kwargs: additional arguments to be passed to the policy on creation
    :param verbose: Verbosity level: 0 for no output, 1 for info messages (such as device or wrappers used), 2 for
        debug messages
    :param seed: Seed for the pseudo random generators
    :param device: Device (cpu, cuda, ...) on which the code should be run.
        Setting it to auto, the code will be run on the GPU if possible.
    :param _init_setup_model: Whether or not to build the network at the creation of the instance
    """

    policy_aliases: ClassVar[Dict[str, Type[BasePolicy]]] = {
        "MlpPolicy": MlpPolicy,
        "CnnPolicy": CnnPolicy,
        "MultiInputPolicy": MultiInputPolicy,
    }
    # Linear schedule will be defined in `_setup_model()`
    exploration_schedule: Schedule
    q_net: QNetwork
    q_net_target: QNetwork
    policy: DQNPolicy

    def __init__(
        self,
        policy: Union[str, Type[DQNPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 1e-4,
        buffer_size: int = 1_000_000,  # 1e6
        learning_starts: int = 50000,
        batch_size: int = 32,
        tau: float = 1.0,
        gamma: float = 0.99,
        train_freq: Union[int, Tuple[int, str]] = 4,
        gradient_steps: int = 1,
        replay_buffer_class: Optional[Type[ReplayBuffer]] = None,
        replay_buffer_kwargs: Optional[Dict[str, Any]] = None,
        optimize_memory_usage: bool = False,
        target_update_interval: int = 10000,
        exploration_fraction: float = 0.1,
        exploration_initial_eps: float = 1.0,
        exploration_final_eps: float = 0.05,
        max_grad_norm: float = 10,
        stats_window_size: int = 100,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
        self_define = 0,
        attribution_method = 'Gradient'
    ) -> None:
        super().__init__(
            policy,
            env,
            learning_rate,
            buffer_size,
            learning_starts,
            batch_size,
            tau,
            gamma,
            train_freq,
            gradient_steps,
            action_noise=None,  # No action noise
            replay_buffer_class=replay_buffer_class,
            replay_buffer_kwargs=replay_buffer_kwargs,
            policy_kwargs=policy_kwargs,
            stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            device=device,
            seed=seed,
            sde_support=False,
            optimize_memory_usage=optimize_memory_usage,
            supported_action_spaces=(spaces.Discrete,),
            support_multi_env=True,
        )
        self.self_define = self_define
        self.attribution_method = attribution_method
        self.exploration_initial_eps = exploration_initial_eps
        self.exploration_final_eps = exploration_final_eps
        self.exploration_fraction = exploration_fraction
        self.target_update_interval = target_update_interval
        # For updating the target network with multiple envs:
        self._n_calls = 0
        self.max_grad_norm = max_grad_norm
        # "epsilon" for the epsilon-greedy exploration
        self.exploration_rate = 0.0

        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        super()._setup_model()
        self._create_aliases()
        # Copy running stats, see GH issue #996
        self.batch_norm_stats = get_parameters_by_name(self.q_net, ["running_"])
        self.batch_norm_stats_target = get_parameters_by_name(self.q_net_target, ["running_"])
        self.exploration_schedule = get_linear_fn(
            self.exploration_initial_eps,
            self.exploration_final_eps,
            self.exploration_fraction,
        )

        if self.n_envs > 1:
            if self.n_envs > self.target_update_interval:
                warnings.warn(
                    "The number of environments used is greater than the target network "
                    f"update interval ({self.n_envs} > {self.target_update_interval}), "
                    "therefore the target network will be updated after each call to env.step() "
                    f"which corresponds to {self.n_envs} steps."
                )

    def _create_aliases(self) -> None:
        self.q_net = self.policy.q_net
        self.q_net_target = self.policy.q_net_target

    def _on_step(self) -> None:
        """
        Update the exploration rate and target network if needed.
        This method is called in ``collect_rollouts()`` after each step in the environment.
        """
        self._n_calls += 1
        # Account for multiple environments
        # each call to step() corresponds to n_envs transitions
        if self._n_calls % max(self.target_update_interval // self.n_envs, 1) == 0:
            polyak_update(self.q_net.parameters(), self.q_net_target.parameters(), self.tau)
            # Copy running stats, see GH issue #996
            polyak_update(self.batch_norm_stats, self.batch_norm_stats_target, 1.0)

        self.exploration_rate = self.exploration_schedule(self._current_progress_remaining)
        self.logger.record("rollout/exploration_rate", self.exploration_rate)

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update learning rate according to schedule
        self._update_learning_rate(self.policy.optimizer)

        losses = []
        for _ in range(gradient_steps):
            # Sample replay buffer
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)  # type: ignore[union-attr]

            with th.no_grad():
                # Compute the next Q-values using the target network
                next_q_values = self.q_net_target(replay_data.next_observations)
                # Follow greedy policy: use the one with the highest value
                next_q_values, _ = next_q_values.max(dim=1)
                # Avoid potential broadcast issue
                next_q_values = next_q_values.reshape(-1, 1)
                # 1-step TD target
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values

            # Get current Q-values estimates
            current_q_values = self.q_net(replay_data.observations)

            # Retrieve the q-values for the actions from the replay buffer
            current_q_values = th.gather(current_q_values, dim=1, index=replay_data.actions.long())

            # Compute Huber loss (less sensitive to outliers)
            loss = F.smooth_l1_loss(current_q_values, target_q_values)
            if self.attribution_method == 'Gradient':
                attributions = gradient_method(self.q_net.q_net, replay_data.observations.squeeze()[1:2], replay_data.observations.squeeze()[32:40])
                attributions_tensor = torch.sum(torch.abs(attributions[:, 2, 1]))
                loss = loss + 2 * torch.abs(attributions_tensor)
            else:
                if self.num_timesteps > 100:
                    # if self.self_define < 12:
                    #     if self.self_define < 4:
                    #         abc = 0.5
                    #     elif 4 <= self.self_define < 8:
                    #         abc = 2
                    #     else:
                    #         abc = 1000
                        # 计算特征归因
                    attributions = compute_attributions(self.q_net.q_net, replay_data.observations.squeeze()[1:2], replay_data.observations.squeeze()[32:40])
                    attributions_tensor = torch.abs(torch.stack(attributions,dim=0)[:,2,1])
                    loss = loss + 1000 * torch.abs(attributions_tensor)
            losses.append(loss.item())
            # print(loss.item())
            # Optimize the policy
            self.policy.optimizer.zero_grad()
            loss.backward()
            # Clip gradient norm
            th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()

        # Increase update counter
        self._n_updates += gradient_steps

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/loss", np.mean(losses))

    def predict(
        self,
        observation: Union[np.ndarray, Dict[str, np.ndarray]],
        state: Optional[Tuple[np.ndarray, ...]] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:
        """
        Overrides the base_class predict function to include epsilon-greedy exploration.

        :param observation: the input observation
        :param state: The last states (can be None, used in recurrent policies)
        :param episode_start: The last masks (can be None, used in recurrent policies)
        :param deterministic: Whether or not to return deterministic actions.
        :return: the model's action and the next state
            (used in recurrent policies)
        """
        if not deterministic and np.random.rand() < self.exploration_rate:
            if self.policy.is_vectorized_observation(observation):
                if isinstance(observation, dict):
                    n_batch = observation[list(observation.keys())[0]].shape[0]
                else:
                    n_batch = observation.shape[0]
                action = np.array([self.action_space.sample() for _ in range(n_batch)])
            else:
                action = np.array(self.action_space.sample())
        else:
            action, state = self.policy.predict(observation, state, episode_start, deterministic)
        return action, state

    def learn(
        self: SelfDQN,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 4,
        tb_log_name: str = "DQN",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> SelfDQN:
        return super().learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            tb_log_name=tb_log_name,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=progress_bar,
        )

    def _excluded_save_params(self) -> List[str]:
        return [*super()._excluded_save_params(), "q_net", "q_net_target"]

    def _get_torch_save_params(self) -> Tuple[List[str], List[str]]:
        state_dicts = ["policy", "policy.optimizer"]

        return state_dicts, []
