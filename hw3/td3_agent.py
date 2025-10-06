# td3_agent.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, Any
from buffer import Buffer
from policies import Actor, Critic
import copy


class TD3Agent:
    """
    Twin Delayed DDPG (TD3) agent that matches PPO's interface pattern.
    Simplified to remove unnecessary abstraction layers.
    """
    
    def __init__(self, env_info, lr=3e-4, gamma=0.99, tau=0.005, 
                 batch_size=128, update_every=1, buffer_size=100000, 
                 warmup_steps=5000, policy_noise=0.2, noise_clip=0.5, 
                 exploration_noise=0.1, delay = 2, device="cpu"):
        self.device = torch.device(device)
        
        # Environment info
        self.obs_dim = env_info["obs_dim"]
        self.act_dim = env_info["act_dim"]
        self.act_low = torch.as_tensor(env_info["act_low"], dtype=torch.float32, device=self.device)
        self.act_high = torch.as_tensor(env_info["act_high"], dtype=torch.float32, device=self.device)
        
        # TD3 hyperparameters
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.update_every = update_every
        self.warmup_steps = warmup_steps
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.exploration_noise = exploration_noise
        self.policy_delay = delay  # Standard TD3 delay
        
        # ================== Problem 2.1.1: TD3 initialization ==================
        ### BEGIN STUDENT SOLUTION - 2.1.1 ###
        self.actor = Actor(
            obs_dim=self.obs_dim,
            act_dim=self.act_dim,
            act_low=self.act_low,
            act_high=self.act_high,
            hidden=(64, 64)
        ).to(self.device)
        self.target_actor = copy.deepcopy(self.actor).to(self.device)
        self.target_actor.eval()

        self.critic1 = Critic(
            self.obs_dim,
            self.act_dim,
            hidden=(64, 64)
        ).to(self.device)
        self.target_critic1 = copy.deepcopy(self.critic1).to(self.device)
        self.target_critic1.eval()

        self.critic2 = Critic(
            self.obs_dim,
            self.act_dim,
            hidden=(64, 64)
        ).to(self.device)
        self.target_critic2 = copy.deepcopy(self.critic2).to(self.device)
        self.target_critic2.eval()
        ### END STUDENT SOLUTION  -  2.1.1 ###
        
        # Optimizers
        self.actor_opt = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_opt = optim.Adam(
            list(self.critic1.parameters()) + list(self.critic2.parameters()), lr=lr
        )
        
        self._buffer = Buffer(
            size=buffer_size,
            obs_dim=self.obs_dim,
            act_dim=self.act_dim,
            device=device
        )
        
        # Training state
        self.total_steps = 0
        self._update_count = 0
    
    def act(self, obs):
        """Return action info dict matching PPO's interface"""
        with torch.no_grad():
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            action = 0 # Placeholder
            
            # ---------------- Problem 2.2: Exploration noise at action time ----------------
            ### BEGIN STUDENT SOLUTION - 2.2 ###
            next_obs = self.actor(obs_t).mean_action
            noise = (
                torch.randn_like(next_obs, device=self.device) * self.exploration_noise
            ).clamp(-self.noise_clip, self.noise_clip)
            action = (next_obs + noise).clamp(self.act_low, self.act_high)
            ### END STUDENT SOLUTION  -  2.2 ###
            
            return {
                "action": action.squeeze(0).cpu().numpy()
            }
    
    def step(self, transition: Dict[str, Any]) -> Dict[str, float]:
        """
        Add transition to buffer and perform updates when ready.
        Matches PPO's step interface.
        """
        # Add to buffer using existing Buffer.add method
        obs_t = torch.as_tensor(transition["obs"], dtype=torch.float32, device=self.device)
        next_obs_t = torch.as_tensor(transition["next_obs"], dtype=torch.float32, device=self.device)
        action_t = torch.as_tensor(transition["action"], dtype=torch.float32, device=self.device)
        
        self._buffer.add(
            obs=obs_t,
            next_obs=next_obs_t,
            action=action_t,
            log_probs=0.0,  # Not used in TD3
            reward=float(transition["reward"]),
            done=float(transition["done"]),
            value=0.0,  # Not used in TD3
            advantage=0.0,  # Not used in TD3
            curr_return=0.0,  # Not used in TD3
            iteration=0  # Not used in TD3
        )
        
        self.total_steps += 1
        
        # ---------------- Problem 2.4: Exploration noise at action time ----------------
        ### BEGIN STUDENT SOLUTION - 2.4 ###
        if self.total_steps < self.warmup_steps or self._buffer.size < self.batch_size:
            return {}
    
        if self.total_steps % self.update_every != 0:
            return {}
        ### END STUDENT SOLUTION - 2.4 ###
        
        # Perform TD3 updates
        return self._perform_update()
    
    def _perform_update(self) -> Dict[str, float]:
        """Perform TD3 updates and return stats"""
        all_stats = []
        
        # Perform updates based on update_every
        num_updates = max(1, self.update_every)
        
        for _ in range(num_updates):
            # Sample batch from buffer
            batch = self._buffer.sample(self.batch_size)
            stats = {}
            
            # ---------------- Problem 2.3: Delayed policy updates ----------------
            ### BEGIN STUDENT SOLUTION - 2.3 ###
            do_actor_update = (self._update_count % self.policy_delay == 0)
            stats = self._td3_update_step(batch, do_actor_update)
            self._update_count += 1
            ### END STUDENT SOLUTION  -  2.3 ###
            all_stats.append(stats)
        
        # Average stats across updates
        if all_stats:
            return {k: np.mean([s[k] for s in all_stats]) for k in all_stats[0].keys()}
        else:
            return {}
    
    def _td3_update_step(self, batch, do_actor_update: bool) -> Dict[str, float]:
        """Single TD3 update step"""
        obs = batch["obs"]
        actions = batch["actions"]
        rewards = batch["rewards"]
        next_obs = batch["next_obs"]
        dones = batch["dones"]
            
        # ---------------- Problem 2.1.2: TD3 target with policy smoothing ----------------
        ### BEGIN STUDENT SOLUTION - 2.1.2 ###
        with torch.no_grad():
            next_actions = self.target_actor(next_obs).mean_action
            noise = (
                torch.randn_like(next_actions, device=self.device) * self.policy_noise
            ).clamp(-self.noise_clip, self.noise_clip)
            next_actions = (
                next_actions + noise
            ).clamp(self.act_low, self.act_high)

            target_q1 = self.target_critic1(next_obs, next_actions)
            target_q2 = self.target_critic2(next_obs, next_actions)
            target_q = rewards + self.gamma * (1 - dones) * torch.min(target_q1, target_q2)
        ### END STUDENT SOLUTION  -  2.1.2 ###
        
        # ---------------- Problem 2.1.3: Critic update ----------------
        ### BEGIN STUDENT SOLUTION - 2.1.3 ###
        current_q1 = self.critic1(obs, actions)
        current_q2 = self.critic2(obs, actions)
        critic1_loss = nn.functional.mse_loss(current_q1, target_q)
        critic2_loss = nn.functional.mse_loss(current_q2, target_q)
        critic_loss = critic1_loss + critic2_loss
        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()
        ### END STUDENT SOLUTION  -  2.1.3 ###
        
        # ---------------- Problem 2.1.4: Actor update (delayed) ----------------
        ### BEGIN STUDENT SOLUTION - 2.1.4 ###
        if do_actor_update:
            next_actions = self.actor(obs).mean_action
            actor_loss = -self.critic1(obs, next_actions).mean()

            self.actor_opt.zero_grad()
            actor_loss.backward()
            self.actor_opt.step()

            self._soft_update(self.actor, self.target_actor)
            self._soft_update(self.critic1, self.target_critic1)
            self._soft_update(self.critic2, self.target_critic2)
        else:
            actor_loss = torch.tensor(0.0, device=self.device)
        ### END STUDENT SOLUTION  -  2.1.4 ###
        
        # Return stats in format expected by runner
        return {
            "actor_loss": float(actor_loss.item()),
            "critic1_loss": float(nn.functional.mse_loss(current_q1, target_q).item()),
            "critic2_loss": float(nn.functional.mse_loss(current_q2, target_q).item()),
            "q1": float(current_q1.mean().item()),
            "q2": float(current_q2.mean().item()),
        }
    
    def _soft_update(self, local_model, target_model):
        """Soft update target network parameters using Polyak averaging"""
        # ---------------- Problem 2.1.5: Polyak averaging ----------------
        ### BEGIN STUDENT SOLUTION - 2.1.5 ###
        with torch.no_grad():
            for params, target_params in zip(local_model.parameters(), target_model.parameters()):
                target_params.copy_((1.0 - self.tau) * target_params + self.tau * params)
        ### END STUDENT SOLUTION  -  2.1.5 ###