import numpy as np

from resco_benchmark.config.signal_config import signal_configs
from resco_benchmark.agents.agent import Agent

try:
    import tensorflow as tf
    from resco_benchmark.agents.ma2c import MA2CAgent
except ImportError:
    tf = None
    pass

if tf is None:
    class FMA2C(Agent):
        def __init__(self, config, obs_act, map_name, thread_number):
            super().__init__()
            raise EnvironmentError("Install optional tensorflow requirement for FMA2C")

else:
    tf.compat.v1.disable_eager_execution()
    
    class FMA2C(Agent):
        def __init__(self, config, obs_act, map_name, thread_number):
            super().__init__()
            self.config = config

            cfg_proto = tf.compat.v1.ConfigProto(allow_soft_placement=True)
            self.sess = tf.compat.v1.Session(config=cfg_proto)

            self.signal_config = signal_configs[map_name]
            self.supervisors = config['mdp']['supervisors']
            self.management_neighbors = config['mdp']['management_neighbors']
            management = config['mdp']['management']

            self.state = None
            self.acts = None

            self.managers = dict()
            self.workers = dict()

            for manager in management:
                worker_ids = management[manager]
                mgr_act_size = self.config['management_acts']
                mgr_fingerprint_size = len(self.management_neighbors[manager]) * mgr_act_size
                self.managers[manager] = MA2CAgent(config, obs_act[manager][0], mgr_act_size, mgr_fingerprint_size, 0,
                                                   manager + str(thread_number), self.sess)

                for worker_id in worker_ids:
                    downstream = self.signal_config[worker_id]['downstream']
                    neighbors = [downstream[direction] for direction in downstream]
                    fp_size = sum(obs_act[neighbor][1] for neighbor in neighbors if neighbor is not None and self.supervisors[neighbor] == self.supervisors[worker_id])

                    lane_sets = self.signal_config[worker_id]['lane_sets']
                    lanes = list(set(lane for direction in lane_sets for lane in lane_sets[direction]))
                    waits_len = len(lanes)

                    management_size = len(self.management_neighbors[manager]) + 1

                    observation_shape = (obs_act[worker_id][0][0] + management_size,)
                    num_actions = obs_act[worker_id][1]
                    self.workers[worker_id] = MA2CAgent(config, observation_shape, num_actions, fp_size, waits_len,
                                                        worker_id + str(thread_number), self.sess)

            self.saver = tf.compat.v1.train.Saver(max_to_keep=1)
            self.sess.run(tf.compat.v1.global_variables_initializer())

        def fingerprints(self, observation):
            agent_fingerprint = {}
            for agent_id in observation.keys():
                if agent_id in self.managers:
                    fingerprints = [self.managers[neighbor].fingerprint for neighbor in self.management_neighbors[agent_id]]
                    agent_fingerprint[agent_id] = np.concatenate(fingerprints) if fingerprints else np.asarray([])
                else:
                    downstream = self.signal_config[agent_id]['downstream']
                    neighbors = [downstream[direction] for direction in downstream]
                    fingerprints = [self.workers[neighbor].fingerprint for neighbor in neighbors if neighbor is not None and self.supervisors[neighbor] == self.supervisors[agent_id]]
                    agent_fingerprint[agent_id] = np.concatenate(fingerprints) if fingerprints else np.asarray([])
            return agent_fingerprint

        def act(self, observation):
            acts = {}
            full_state = {}
            fingerprints = self.fingerprints(observation)
            
            for agent_id in self.managers:
                env_obs = observation[agent_id]
                neighbor_fingerprints = fingerprints[agent_id]
                acts[agent_id] = self.managers[agent_id].act(np.concatenate([env_obs, neighbor_fingerprints]))
            
            for agent_id in self.workers:
                env_obs = observation[agent_id]
                neighbor_fingerprints = fingerprints[agent_id]
                combine = np.concatenate([env_obs, neighbor_fingerprints])
                full_state[agent_id] = combine

                managing_agent = self.supervisors[agent_id]
                managing_agents_acts = np.asarray([acts[managing_agent]] + [acts[mgr_neighbor] for mgr_neighbor in self.management_neighbors[managing_agent]])
                combine = np.concatenate([managing_agents_acts, combine])

                acts[agent_id] = self.workers[agent_id].act(combine)
            
            self.state = full_state
            self.acts = acts
            return acts

        def observe(self, observation, reward, done, info):
            fingerprints = self.fingerprints(observation)
            
            for agent_id in observation.keys():
                env_obs = observation[agent_id]
                neighbor_fingerprints = fingerprints[agent_id]
                combine = np.concatenate([env_obs, neighbor_fingerprints])
                
                if agent_id in self.managers:
                    self.managers[agent_id].observe(combine, reward[agent_id], done, info)
                else:
                    managing_agent = self.supervisors[agent_id]
                    managing_agents_acts = np.asarray([self.acts[managing_agent]] + [self.acts[mgr_neighbor] for mgr_neighbor in self.management_neighbors[managing_agent]])
                    combine = np.concatenate([managing_agents_acts, combine])
                    self.workers[agent_id].observe(combine, reward[agent_id], done, info)

                if done and info['eps'] % 100 == 0 and self.saver:
                    self.saver.save(self.sess, self.config['log_dir'] + 'agent_' + 'checkpoint', global_step=info['eps'])
