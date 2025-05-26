from multiprocessing import Value

import numpy as np
import torch
from yarr.agents.agent import Agent
from yarr.envs.env import Env
from yarr.utils.transition import ReplayTransition


class RolloutGenerator(object):

    def _get_type(self, x):
        if x.dtype == np.float64:
            return np.float32
        return x.dtype

    def generator(self, step_signal: Value, env: Env, agent: Agent,
                  episode_length: int, timesteps: int,
                  eval: bool, eval_demo_seed: int = 0,
                  record_enabled: bool = False):

        if eval:
            obs = env.reset_to_demo(eval_demo_seed)
        else:
            obs = env.reset()

        agent.reset()

        # print(obs['lang_goal'], obs['gripper_pose'], obs['gripper_open'])
        
        tmp_obs = {k: v for k, v in obs.items() if isinstance(v, np.ndarray)}
        mapping_dict =  {k: v for k, v in obs.items() if not isinstance(v, np.ndarray)}
        obs_history = {k: [np.array(v, dtype=self._get_type(v))] * timesteps for k, v in tmp_obs.items()}
        
        obs_history["lang_goal"]=[obs['lang_goal']]*timesteps
        for step in range(episode_length):
            print("step:", step)
            # prepped_data = {k:torch.tensor([v], device=self._env_device) for k, v in obs_history.items()}
            prepped_data={}
            for k,v in obs_history.items():
                if k!="lang_goal":
                    prepped_data[k]=torch.tensor(np.array([v]), device=self._env_device)
                else:
                    prepped_data["lang_goal"]=v

            act_result = agent.act(step_signal.value, prepped_data,
                                   deterministic=eval, mapping_dict=mapping_dict, act_step=step, lang_goal=env._lang_goal)

            # Convert to np if not already
            agent_obs_elems = {k: np.array(v) for k, v in
                               act_result.observation_elements.items()}
            extra_replay_elements = {k: np.array(v) for k, v in
                                     act_result.replay_elements.items()}

            transition = env.step(act_result)
            obs_tp1 = dict(transition.observation)
            timeout = False
            if step == episode_length - 1:
                print("last step")
                # If last transition, and not terminal, then we timed out
                timeout = not transition.terminal
                if timeout:
                    transition.terminal = True
                    if "needs_reset" in transition.info:
                        transition.info["needs_reset"] = True

            obs_and_replay_elems = {}
            obs_and_replay_elems.update(obs)
            obs_and_replay_elems.update(agent_obs_elems)
            obs_and_replay_elems.update(extra_replay_elements)

            for k in obs_history.keys():
                obs_history[k].append(transition.observation[k])
                obs_history[k].pop(0)
            mapping_dict =  {k: v for k, v in transition.observation.items() if not isinstance(v, np.ndarray)} 
            
            transition.info["active_task_id"] = env.active_task_id

            replay_transition = ReplayTransition(
                obs_and_replay_elems, act_result.action, transition.reward,
                transition.terminal, timeout, summaries=transition.summaries,
                info=transition.info)

            if transition.terminal or timeout:
                if transition.terminal:
                    print("terminal")
                else:
                    print("timeout")
                # If the agent gives us observations then we need to call act
                # one last time (i.e. acting in the terminal state).
                if len(act_result.observation_elements) > 0:
                    # prepped_data = {k: torch.tensor([v], device=self._env_device) for k, v in obs_history.items()}
                    prepped_data={}
                    for k,v in obs_history.items():
                        if k!="lang_goal":
                            prepped_data[k]=torch.tensor(np.array([v]), device=self._env_device)
                        else:
                            prepped_data["lang_goal"]=v
                    act_result = agent.act(step_signal.value, prepped_data,
                                           deterministic=eval, mapping_dict=mapping_dict, act_step=step, lang_goal=env._lang_goal)
                    agent_obs_elems_tp1 = {k: np.array(v) for k, v in
                                           act_result.observation_elements.items()}
                    obs_tp1.update(agent_obs_elems_tp1)
                replay_transition.final_observation = obs_tp1

            if record_enabled and transition.terminal or timeout or step == episode_length - 1:
                env.env._action_mode.arm_action_mode.record_end(env.env._scene, steps=60, step_scene=True)

            obs = dict(transition.observation)
            yield replay_transition

            if transition.info.get("needs_reset", transition.terminal):
                return
