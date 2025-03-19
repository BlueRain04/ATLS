import numpy as np

from resco_benchmark.agents.agent import Agent, IndependentAgent
from resco_benchmark.config.signal_config import signal_configs


try:
    import tensorflow as tf
except ImportError:
    tf = None
    pass


if tf is None:
    class MA2C(IndependentAgent):
        def __init__(self, config, obs_act, map_name, thread_number, sess=None):
            super().__init__(config, obs_act, map_name, thread_number)
            raise EnvironmentError("Install optional tensorflow requirement for MA2C")

else:

    class MA2C(IndependentAgent):
        def __init__(self, config, obs_act, map_name, thread_number):
            super().__init__(config, obs_act, map_name, thread_number)
            self.signal_config = signal_configs[map_name]
        
            self.agents = {}
            self.signal_config = signal_configs[map_name]
            self.sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(allow_soft_placement=True))
        
            for key, (obs_space, act_space) in obs_act.items():
            # Get waiting size
                lane_sets = self.signal_config[key]['lane_sets']
                lanes = set(lane for direction in lane_sets for lane in lane_sets[direction])
                waits_len = len(lanes)

            # Get fingerprint size
                downstream = self.signal_config[key]['downstream']
                neighbors = [downstream[direction] for direction in downstream]
                fp_size = sum(obs_act[neighbor][1] for neighbor in neighbors if neighbor is not None)
            
                self.agents[key] = MA2CAgent(config, obs_space, act_space, fp_size, waits_len, f'ma2c{key}{thread_number}', self.sess)
        
            self.saver = tf.compat.v1.train.Saver(max_to_keep=1)
            self.sess.run(tf.compat.v1.global_variables_initializer())

        def fingerprints(self, observation):
            agent_fingerprint = {}
            for agent_id, env_obs in observation.items():
                downstream = self.signal_config[agent_id]['downstream']
                neighbors = [downstream[direction] for direction in downstream]
                fingerprints = [self.agents[neighbor].fingerprint for neighbor in neighbors if neighbor is not None]
                agent_fingerprint[agent_id] = np.concatenate(fingerprints) if fingerprints else np.array([])
            return agent_fingerprint

        def act(self, observation):
            acts = {}
            fingerprints = self.fingerprints(observation)
            for agent_id, env_obs in observation.items():
                neighbor_fingerprints = fingerprints[agent_id]
                combine = np.concatenate([env_obs, neighbor_fingerprints])
                acts[agent_id] = self.agents[agent_id].act(combine)
            return acts

        def observe(self, observation, reward, done, info):
            fingerprints = self.fingerprints(observation)
            for agent_id, env_obs in observation.items():
                neighbor_fingerprints = fingerprints[agent_id]
                combine = np.concatenate([env_obs, neighbor_fingerprints])
                self.agents[agent_id].observe(combine, reward[agent_id], done, info)
        
            if done and info['eps'] % 100 == 0:
                self.saver.save(self.sess, f"{self.config['log_dir']}agent_checkpoint", global_step=info['eps'])


class MA2CAgent(Agent):
    def __init__(self, config, observation_shape, num_actions, fingerprint_size, waits_len, name, sess):
        super().__init__()
        self.config = config
        self.num_actions = num_actions
        self.sess = sess

        self.steps_done = 0
        self.state = None
        self.value = None
        self.action = None
        self.fingerprint = np.zeros(num_actions)

        n_s = observation_shape[0] + fingerprint_size
        n_a = num_actions
        n_w = waits_len
        n_f = fingerprint_size
        total_step = config['steps']
        model_config = config

        print(name, n_s, n_a, n_w, n_f)
        self.model = MA2CImplementation(n_s, n_a, n_w, n_f, total_step, model_config, name, sess)

    def act(self, observation):
        self.state = observation
        policy, self.value = self.model.forward(observation, False)
        self.action = np.random.choice(np.arange(len(policy)), p=policy)
        self.fingerprint = np.array(policy)
        return self.action

    def observe(self, observation, reward, done, info):
        self.model.add_transition(self.state, self.action, reward, self.value, done)
        self.steps_done += 1

        if self.steps_done % self.config['batch_size'] == 0 or done:
            R = 0 if done else self.model.forward(observation, False, 'v')
            self.model.backward(R)

        if done:
            self.steps_done = 0
            self.model.reset()



    # https://github.com/cts198859/deeprl_signal_control
class MA2CImplementation:
    def __init__(self, n_s, n_a, n_w, n_f, total_step, model_config, name, sess):
        self.name = name
        self.sess = sess
        self.reward_clip = model_config['reward_clip']
        self.reward_norm = model_config['reward_norm']
        self.n_s = n_s
        self.n_a = n_a
        self.n_f = n_f
        self.n_w = n_w
        self.n_step = model_config['batch_size']

        self.policy = self._init_policy(n_s - n_f - n_w, n_a, n_w, n_f, model_config, agent_name=name)

        if total_step:
            self.total_step = total_step
            self._init_scheduler(model_config)
            self._init_train(model_config)

    def _init_policy(self, n_s, n_a, n_w, n_f, model_config, agent_name=None):
        n_fw = model_config['num_fw']
        n_ft = model_config['num_ft']
        n_lstm = model_config['num_lstm']
        n_fp = model_config['num_fp']
        return FPLstmACPolicy(n_s, n_a, n_w, n_f, self.n_step, n_fc_wave=n_fw,
                               n_fc_wait=n_ft, n_fc_fp=n_fp, n_lstm=n_lstm, name=agent_name)

    def _init_scheduler(self, model_config):
        self.lr_scheduler = Scheduler(model_config['lr_init'], decay=model_config['lr_decay'])
        self.beta_scheduler = Scheduler(model_config['entropy_coef_init'], decay=model_config['entropy_decay'])

    def _init_train(self, model_config):
        self.policy.prepare_loss(model_config['value_coef'], model_config['max_grad_norm'], model_config['rmsp_alpha'], model_config['rmsp_epsilon'])
        self.trans_buffer = OnPolicyBuffer(model_config['gamma'])

    def backward(self, R):
        cur_lr = self.lr_scheduler.get(self.n_step)
        cur_beta = self.beta_scheduler.get(self.n_step)
        obs, acts, dones, Rs, Advs = self.trans_buffer.sample_transition(R)
        self.policy.backward(self.sess, obs, acts, dones, Rs, Advs, cur_lr, cur_beta)

    def forward(self, obs, done, out_type='pv'):
        return self.policy.forward(self.sess, obs, done, out_type)

    def reset(self):
        self.policy._reset()

    def add_transition(self, obs, actions, rewards, values, done):
        rewards = np.clip(rewards / self.reward_norm if self.reward_norm else rewards, -self.reward_clip, self.reward_clip)
        self.trans_buffer.add_transition(obs, actions, rewards, values, done)

        """def load(self, model_dir, checkpoint=None):
            save_file = None
            save_step = 0
            if os.path.exists(model_dir):
                if checkpoint is None:
                    for file in os.listdir(model_dir):
                        if file.startswith('checkpoint'):
                            prefix = file.split('.')[0]
                            tokens = prefix.split('-')
                            if len(tokens) != 2:
                                continue
                            cur_step = int(tokens[1])
                            if cur_step > save_step:
                                save_file = prefix
                                save_step = cur_step
                else:
                    save_file = 'checkpoint-' + str(int(checkpoint))
            if save_file is not None:
                self.saver.restore(self.sess, model_dir + save_file)
                logging.info('Checkpoint loaded: %s' % save_file)
                return True
            logging.error('Can not find old checkpoint for %s' % model_dir)
            return False"""


class ACPolicy(Model):
    def __init__(self, n_a, n_s, n_step, policy_name, agent_name=None):
        super(ACPolicy, self).__init__()
        self.name = policy_name if agent_name is None else f"{policy_name}_{agent_name}"
        self.n_a = n_a
        self.n_s = n_s
        self.n_step = n_step

        # Define policy and value networks
        self.policy_layer = layers.Dense(n_a, activation='softmax', name="pi")
        self.value_layer = layers.Dense(1, activation=None, name="v")  # Linear activation for value estimate

        self.optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.001, rho=0.99, epsilon=1e-5)

    def call(self, inputs, out_type='pv'):
        """ Forward pass """
        policy = self.policy_layer(inputs)
        value = self.value_layer(inputs)

        outs = []
        if 'p' in out_type:
            outs.append(policy)
        if 'v' in out_type:
            outs.append(value)

        return outs[0] if len(outs) == 1 else outs

    def compute_loss(self, actions, advantages, returns, entropy_coef, v_coef):
        """ Compute policy and value loss """
        action_probs = self.policy_layer(self.inputs)
        values = self.value_layer(self.inputs)

        # One-hot encode actions
        actions_one_hot = tf.one_hot(actions, self.n_a)
        log_probs = tf.math.log(tf.clip_by_value(action_probs, 1e-10, 1.0))
        entropy = -tf.reduce_sum(action_probs * log_probs, axis=1)
        entropy_loss = -tf.reduce_mean(entropy) * entropy_coef

        policy_loss = -tf.reduce_mean(tf.reduce_sum(log_probs * actions_one_hot, axis=1) * advantages)
        value_loss = tf.reduce_mean(tf.square(returns - values)) * 0.5 * v_coef

        return policy_loss + value_loss + entropy_loss

    def train_step(self, inputs, actions, advantages, returns, entropy_coef, v_coef):
        """ Perform a single training step """
        with tf.GradientTape() as tape:
            loss = self.compute_loss(actions, advantages, returns, entropy_coef, v_coef)
        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        return loss

class LstmACPolicy(Model):
    def __init__(self, n_s, n_a, n_w, n_step, n_fc_wave=128, n_fc_wait=32, n_lstm=64, name=None):
        super(LstmACPolicy, self).__init__(name=name)
        self.n_a = n_a
        self.n_s = n_s
        self.n_w = n_w
        self.n_step = n_step
        self.n_fc_wave = n_fc_wave
        self.n_fc_wait = n_fc_wait
        self.n_lstm = n_lstm

        # Policy and value networks
        self.fc_wave = layers.Dense(n_fc_wave, activation='relu', name="fc_wave")
        self.fc_wait = layers.Dense(n_fc_wait, activation='relu', name="fc_wait")
        self.lstm = layers.LSTM(n_lstm, return_state=True, return_sequences=True, name="lstm")
        self.policy_layer = layers.Dense(n_a, activation='softmax', name="policy")
        self.value_layer = layers.Dense(1, activation=None, name="value")

        # Optimizer
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

        # Initialize LSTM states
        self.reset()

    def reset(self):
        """Reset LSTM states."""
        self.states_fw = tf.zeros((1, self.n_lstm))
        self.states_bw = tf.zeros((1, self.n_lstm))

    def call(self, ob, done, out_type='pv'):
        """Forward pass for policy and value prediction."""
        if self.n_w == 0:
            h = self.fc_wave(ob)
        else:
            h_wave = self.fc_wave(ob[:, :self.n_s])
            h_wait = self.fc_wait(ob[:, self.n_s:])
            h = tf.concat([h_wave, h_wait], axis=1)

        # LSTM forward pass
        h, state_h, state_c = self.lstm(tf.expand_dims(h, axis=1), initial_state=[self.states_fw, self.states_bw])

        # Store new LSTM states
        self.states_fw, self.states_bw = state_h, state_c

        policy = self.policy_layer(h[:, -1, :])  # Take last time step output
        value = self.value_layer(h[:, -1, :])

        outs = []
        if 'p' in out_type:
            outs.append(policy)
        if 'v' in out_type:
            outs.append(value)

        return outs[0] if len(outs) == 1 else outs

    @tf.function
    def train_step(self, obs, acts, dones, Rs, Advs, cur_lr, cur_beta):
        """Train the LSTM-based policy using gradient descent."""
        with tf.GradientTape() as tape:
            policy, value = self.call(obs, dones, out_type='pv')

            # Compute losses
            actions_one_hot = tf.one_hot(acts, self.n_a)
            log_probs = tf.math.log(tf.clip_by_value(policy, 1e-10, 1.0))
            entropy = -tf.reduce_sum(policy * log_probs, axis=1)
            entropy_loss = -tf.reduce_mean(entropy) * cur_beta

            policy_loss = -tf.reduce_mean(tf.reduce_sum(log_probs * actions_one_hot, axis=1) * Advs)
            value_loss = tf.reduce_mean(tf.square(Rs - value)) * 0.5

            loss = policy_loss + value_loss + entropy_loss

        # Apply gradients
        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        return loss

DEFAULT_SCALE = np.sqrt(2)
DEFAULT_MODE = 'fan_in'

def ortho_init(scale=DEFAULT_SCALE, mode=None):
    def _ortho_init(shape, dtype=None, partition_info=None):
        shape = tuple(shape)
        if len(shape) == 2:  # fc: in, out
            flat_shape = shape
        elif (len(shape) == 3) or (len(shape) == 4):  # 1d/2dcnn: (in_h), in_w, in_c, out
            flat_shape = (np.prod(shape[:-1]), shape[-1])
        a = np.random.standard_normal(flat_shape)
        u, _, v = np.linalg.svd(a, full_matrices=False)
        q = u if u.shape == flat_shape else v  # pick the one with the correct shape
        q = q.reshape(shape)
        return (scale * q).astype(np.float32)
    return _ortho_init

DEFAULT_METHOD = ortho_init

class FPLstmACPolicy(tf.keras.layers.Layer):
    def __init__(self, n_s, n_a, n_w, n_f, n_step, n_fc_wave=128, n_fc_wait=32, n_fc_fp=32, n_lstm=64, name=None):
        super(FPLstmACPolicy, self).__init__(name=name)
        self.n_lstm = n_lstm
        self.n_fc_wave = n_fc_wave
        self.n_fc_wait = n_fc_wait
        self.n_fc_fp = n_fc_fp
        self.n_w = n_w

        # Inputs as tensorflow tensors
        self.ob_fw = tf.keras.Input(shape=(n_s + n_w + n_f,))
        self.done_fw = tf.keras.Input(shape=(1,))
        self.ob_bw = tf.keras.Input(shape=(n_step, n_s + n_w + n_f))
        self.done_bw = tf.keras.Input(shape=(n_step,))
        self.states = tf.keras.Input(shape=(2, n_lstm * 2))

        self.pi_fw, self.pi_state_fw = self._build_net('forward', 'pi')
        self.v_fw, self.v_state_fw = self._build_net('forward', 'v')
        self.new_states = tf.concat([tf.expand_dims(self.pi_state_fw, 0), tf.expand_dims(self.v_state_fw, 0)], 0)

        self.pi, _ = self._build_net('backward', 'pi')
        self.v, _ = self._build_net('backward', 'v')

    def _build_net(self, in_type, out_type):
        if in_type == 'forward':
            ob = self.ob_fw
            done = self.done_fw
        else:
            ob = self.ob_bw
            done = self.done_bw
        
        if out_type == 'pi':
            states = self.states[0]
        else:
            states = self.states[1]

        h0 = self.fc(ob[:, :self.n_s], out_type + '_fcw', self.n_fc_wave)
        h1 = self.fc(ob[:, (self.n_s + self.n_w):], out_type + '_fcf', self.n_fc_fp)
        
        if self.n_w == 0:
            h = tf.concat([h0, h1], 1)
        else:
            h2 = self.fc(ob[:, self.n_s: (self.n_s + self.n_w)], out_type + '_fct', self.n_fc_wait)
            h = tf.concat([h0, h1, h2], 1)

        h, new_states = self.lstm(h, done, states, out_type + '_lstm')
        out_val = self._build_out_net(h, out_type)
        return out_val, new_states

    def fc(self, x, scope, n_out, act=tf.nn.relu, init_scale=DEFAULT_SCALE,
           init_mode=DEFAULT_MODE, init_method=DEFAULT_METHOD):
        with tf.variable_scope(scope):
            n_in = x.shape[1].value
            w = tf.get_variable("w", [n_in, n_out],
                                initializer=init_method(init_scale, init_mode))
            b = tf.get_variable("b", [n_out], initializer=tf.constant_initializer(0.0))
            z = tf.matmul(x, w) + b
            return act(z)

    def lstm(self, xs, dones, s, scope, init_scale=DEFAULT_SCALE, init_mode=DEFAULT_MODE,
             init_method=DEFAULT_METHOD):
        xs = self.batch_to_seq(xs)
        dones = self.batch_to_seq(dones)
        n_in = xs[0].shape[1].value
        n_out = s.shape[0] // 2
        with tf.variable_scope(scope):
            wx = tf.get_variable("wx", [n_in, n_out*4],
                                 initializer=init_method(init_scale, init_mode))
            wh = tf.get_variable("wh", [n_out, n_out*4],
                                 initializer=init_method(init_scale, init_mode))
            b = tf.get_variable("b", [n_out*4], initializer=tf.constant_initializer(0.0))
        s = tf.expand_dims(s, 0)
        c, h = tf.split(axis=1, num_or_size_splits=2, value=s)
        for ind, (x, done) in enumerate(zip(xs, dones)):
            c = c * (1-done)
            h = h * (1-done)
            z = tf.matmul(x, wx) + tf.matmul(h, wh) + b
            i, f, o, u = tf.split(axis=1, num_or_size_splits=4, value=z)
            i = tf.nn.sigmoid(i)
            f = tf.nn.sigmoid(f)
            o = tf.nn.sigmoid(o)
            u = tf.tanh(u)
            c = f*c + i*u
            h = o*tf.tanh(c)
            xs[ind] = h
        s = tf.concat(axis=1, values=[c, h])
        return self.seq_to_batch(xs), tf.squeeze(s)

    def batch_to_seq(self, x):
        n_step = x.shape[0].value
        if len(x.shape) == 1:
            x = tf.expand_dims(x, -1)
        return tf.split(axis=0, num_or_size_splits=n_step, value=x)

    def seq_to_batch(self, x):
        return tf.concat(axis=0, values=x)


class Scheduler:
    def __init__(self, val_init, val_min=0, total_step=0, decay='linear'):
        self.val = val_init
        self.N = float(total_step)
        self.val_min = val_min
        self.decay = decay
        self.n = 0

    def get(self, n_step):
        self.n += n_step
        if self.decay == 'linear':
            return max(self.val_min, self.val * (1 - self.n / self.N))
        else:
            return self.val


class TransBuffer:
    def reset(self):
        self.buffer = []

    @property
    def size(self):
        return len(self.buffer)

    def add_transition(self, ob, a, r, *_args, **_kwargs):
        raise NotImplementedError()

    def sample_transition(self, *_args, **_kwargs):
        raise NotImplementedError()


class OnPolicyBuffer(TransBuffer):
    def __init__(self, gamma):
        self.gamma = gamma
        self.reset()

    def reset(self, done=False):
        # the done before each step is required
        self.obs = []
        self.acts = []
        self.rs = []
        self.vs = []
        self.dones = [done]

    def add_transition(self, ob, a, r, v, done):
        self.obs.append(ob)
        self.acts.append(a)
        self.rs.append(r)
        self.vs.append(v)
        self.dones.append(done)

    def _add_R_Adv(self, R):
        Rs = []
        Advs = []
        # use post-step dones here
        for r, v, done in zip(self.rs[::-1], self.vs[::-1], self.dones[:0:-1]):
            R = r + self.gamma * R * (1. - done)
            Adv = R - v
            Rs.append(R)
            Advs.append(Adv)
        Rs.reverse()
        Advs.reverse()
        self.Rs = Rs
        self.Advs = Advs

    def sample_transition(self, R, discrete=True):
        self._add_R_Adv(R)
        obs = np.array(self.obs, dtype=np.float32)
        if discrete:
            acts = np.array(self.acts, dtype=np.int32)
        else:
            acts = np.array(self.acts, dtype=np.float32)
        Rs = np.array(self.Rs, dtype=np.float32)
        Advs = np.array(self.Advs, dtype=np.float32)
        # use pre-step dones here
        dones = np.array(self.dones[:-1], dtype=np.bool)
        self.reset(self.dones[-1])
        return obs, acts, dones, Rs, Advs
