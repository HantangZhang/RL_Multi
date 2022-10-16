from config import get_config
from envs.starcraft2.Random_StarCraft2_Env_Multi import RandomStarCraft2EnvMulti
from envs.env_wrappers import ShareSubprocVecEnv, ShareDummyVecEnv
import numpy as np
import torch

'''
问题记录：
1. env的东西都没有研究，感觉目前没有必要研究
2. self.use_centralized_V
3. episode
4. self.recurrent_N = self.all_args.recurrent_N 为什么用到recurrent network
5. share_observation_space在5v5中是否有用
6. cen_obs_space 明确这个的含义
7. TransformerPolicy 中self.act_num的作用
8. SelfAttention中n_embd的具体含义和embed方法, L这一维的含义



'''


def make_train_env(all_args):
    train_maps = all_args.train_maps
    if all_args.n_rollout_threads % len(train_maps) != 0:
        raise NotImplementedError
    threads_per_map = all_args.n_rollout_threads / len(train_maps)

    def get_env_fn(rank):
        def init_env():
            map_name = train_maps[int(rank/threads_per_map)]
            env = RandomStarCraft2EnvMulti(all_args, map_name)
            env.seed(all_args.seed + rank * 1000)
            return env
        return init_env

    return ShareSubprocVecEnv([get_env_fn(i) for i in range(all_args.n_rollout_threads)])

def make_eval_env(all_args):
    eval_maps = all_args.eval_maps
    if all_args.n_eval_rollout_threads % len(eval_maps) != 0:
        raise NotImplementedError
    threads_per_map = all_args.n_eval_rollout_threads / len(eval_maps)

    def get_env_fn(rank):
        def init_env():
            map_name = eval_maps[int(rank/threads_per_map)]
            env = RandomStarCraft2EnvMulti(all_args, map_name)
            env.seed(all_args.seed * 50000 + rank * 10000)
            return env
        return init_env

    return ShareSubprocVecEnv([get_env_fn(i) for i in range(all_args.n_eval_rollout_threads)])


def parse_args(args, parser):

    # parser.parse_known_args(args)[0]返回一个tuple,[0]是一个namespace，
    # 存放一个一个的参数,[1]是一个空列表，parser中未定义的参数将放在这个列表里面
    # eg: 返回类似这样，(Namespace(flag_int=2, flag_int2=2), [])
    all_args = parser.parse_known_args(args)[0]

    # 所以这个返回的是一个namespace
    return all_args

def main(args):
    parser = get_config()
    all_args = parse_args(args, parser)

    # cuda
    if all_args.cuda and torch.cuda.is_available():
        print("choose to use gpu...")
        device = torch.device("cuda:0")
        torch.set_num_threads(all_args.n_training_threads)
        if all_args.cuda_deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    else:
        print("choose to use cpu...")
        device = torch.device("cpu")
        torch.set_num_threads(all_args.n_training_threads)

    # seed
    torch.manual_seed(all_args.seed)
    torch.cuda.manual_seed_all(all_args.seed)
    np.random.seed(all_args.seed)

    envs = make_train_env(all_args)
    eval_envs = make_eval_env(all_args) if all_args.use_eval else None

    # 需要在这里设置agent的数量
    num_agents = 27

    config = {
        'all_args': all_args,
        'envs' : envs,
        "eval_envs": eval_envs,
        "num_agents": num_agents,
        "device": device,
    }



