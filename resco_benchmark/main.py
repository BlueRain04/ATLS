import pathlib
import os
import multiprocessing as mp
import importlib
import argparse

from resco_benchmark.multi_signal import MultiSignal
from resco_benchmark.config.agent_config import agent_configs
from resco_benchmark.config.map_config import map_configs
from resco_benchmark.config.mdp_config import mdp_configs
from resco_benchmark.config import agent_config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", type=str, default='STOCHASTIC',
                    choices=['STOCHASTIC', 'MAXWAVE', 'MAXPRESSURE', 'IDQN', 'IPPO', 'MPLight', 'MA2C', 'FMA2C',
                             'MPLightFULL', 'FMA2CFull', 'FMA2CVAL'])
    ap.add_argument("--trials", type=int, default=5)
    ap.add_argument("--eps", type=int, default=10)
    ap.add_argument("--procs", type=int, default=2)
    ap.add_argument("--map", type=str, default='ingolstadt1',
                    choices=['grid4x4', 'arterial4x4', 'ingolstadt1', 'ingolstadt7', 'ingolstadt21',
                             'cologne1', 'cologne3', 'cologne8', 'ksu_map'])
    ap.add_argument("--pwd", type=str, default=os.path.dirname(__file__))
    ap.add_argument("--log_dir", type=str, default=os.path.join(os.path.dirname(os.getcwd()), 'results' + os.sep))
    ap.add_argument("--gui", action="store_true")
    ap.add_argument("--libsumo", action="store_true")
    ap.add_argument("--tr", type=int, default=0)
    ap.add_argument("--save_freq", type=int, default=100)
    ap.add_argument("--load", type=bool, default=False)
    args = ap.parse_args()

    print(f"GUI Mode: {args.gui}")
    print(f"Using libsumo: {args.libsumo}")

    if args.libsumo:
        os.environ["LIBSUMO_AS_TRACI"] = "1"
    if args.libsumo and 'LIBSUMO_AS_TRACI' not in os.environ:
        raise EnvironmentError("Set LIBSUMO_AS_TRACI to nonempty value to enable libsumo")

    if args.procs == 1 or args.libsumo:
        run_trial(args, args.tr)
    else:
        pool = mp.Pool(processes=args.procs)
        for trial in range(1, args.trials + 1):
            pool.apply_async(run_trial, args=(args, trial))
        pool.close()
        pool.join()


def run_trial(args, trial):
    mdp_config = mdp_configs.get(args.agent)
    if mdp_config is not None:
        mdp_map_config = mdp_config.get(args.map)
        if mdp_map_config is not None:
            mdp_config = mdp_map_config
        mdp_configs[args.agent] = mdp_config

    agt_config = agent_configs[args.agent]
    agt_config['save_freq'] = args.save_freq
    agt_config['load'] = args.load
    agt_map_config = agt_config.get(args.map)
    if agt_map_config is not None:
        agt_config = agt_map_config
    alg = agt_config['agent']

    if mdp_config is not None:
        agt_config['mdp'] = mdp_config
        management = agt_config['mdp'].get('management')
        if management is not None:
            supervisors = dict()
            for manager in management:
                for worker in management[manager]:
                    supervisors[worker] = manager
            mdp_config['supervisors'] = supervisors

    map_config = map_configs[args.map]
    num_steps_eps = int((map_config['end_time'] - map_config['start_time']) / map_config['step_length'])
    route = map_config['route']
    if route is not None:
        route = os.path.join(args.pwd, route)
    if args.map in ['grid4x4', 'arterial4x4'] and not os.path.exists(route):
        raise EnvironmentError("You must decompress environment files defining traffic flow")

    env = MultiSignal(alg.__name__ + '-tr' + str(trial),
                      args.map,
                      os.path.join(args.pwd, map_config['net']),
                      agt_config['state'],
                      agt_config['reward'],
                      route=route,
                      step_length=map_config['step_length'],
                      yellow_length=map_config['yellow_length'],
                      step_ratio=map_config['step_ratio'],
                      end_time=map_config['end_time'],
                      max_distance=agt_config['max_distance'],
                      lights=map_config['lights'],
                      gui=args.gui,
                      log_dir=args.log_dir,
                      libsumo=args.libsumo,
                      warmup=map_config['warmup'])

    agt_config['episodes'] = int(args.eps * 0.8)
    agt_config['steps'] = agt_config['episodes'] * num_steps_eps
    agt_config['log_dir'] = os.path.join(args.log_dir, env.connection_name)
    agt_config['num_lights'] = len(env.all_ts_ids)

    # Get agent id's, observation shapes, and action sizes from env
    obs_act = {key: [env.obs_shape[key], len(env.phases[key]) if key in env.phases else None]
               for key in env.obs_shape}
    agent = alg(agt_config, obs_act, args.map, trial)

    # ✅ Load model if specified
    if args.load:
        print(f"Loading pretrained model from {agt_config['log_dir']}")
        for ag in agent.agents.values():
            model_dir = os.path.join(agt_config['log_dir'], '')
            ag.model.load(model_dir=model_dir)

    for _ in range(args.eps):
        obs = env.reset()
        done = False
        while not done:
            act = agent.act(obs)
            obs, rew, done, info = env.step(act)
            agent.observe(obs, rew, done, info)
    env.close()


if __name__ == '__main__':
    main()
