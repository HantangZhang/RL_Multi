from config import get_config
from envs.starcraft2.Random_StarCraft2_Env_Multi import RandomStarCraft2EnvMulti
from envs.env_wrappers import ShareSubprocVecEnv, ShareDummyVecEnv
import numpy as np
import torch

'''
每次运行用runner，runner对于特点的环境需要写一个特定的runner，同时它继承一个base_runner
在base_runner中：
    1）它的策略来自于transformer_policy，这个策略是以ma_transformer的神经网络为主体写的，里面是一些调用
        神经网络的逻辑。
    2）同时里面还有mat的trainer，它是mat如何计算损失，梯度等训练的逻辑，用到的是mat_trainner


'''

'''
问题记录：
1. env的东西都没有研究，感觉目前没有必要研究
2. self.use_centralized_V
3. episode
4. self.recurrent_N = self.all_args.recurrent_N 为什么用到recurrent network
5. share_observation_space在5v5中是否有用：在mat中应该没用，它为每一个智能体单评估value，并不是共享value
6. cen_obs_space 明确这个的含义
7. TransformerPolicy 中self.act_num的作用
8. SelfAttention中n_embd的具体含义和embed方法, L这一维的含义
9. encoder当中输出的v_loc作用
10. decoder中dec_actor的含义: 推测是decentrailized actor
11. 搞清楚什么时候forward，什么时候get_action，什么时候get_value
12. discrete_parallel_act里面shiftaction可能有错，它那么转换会丢失掉最后一个智能体执行的动作
    inference用的自回归和它一样，先加了一行0，意味着第一个智能体是基于都是0的向量来选择动作，这是否合理
13. 暂时所有关于rnn的都无视掉，rnn_states_actor和rnn_states_critic
14. Number of parallel envs for training rollouts搞清楚这个参数的作用，它好像就是batch_size：并不是batch_size，
    它就是起了几个env一起推进
15. warmup()做了一件什么事情：就是重置env，我觉得这里面还可以做很多事情
16. 一个trajectory就是一个rollout，那么每次存buffer的时候是怎么寸的，每个batch是里面的数据和trajectory的关系
17. insert当中maks的作用masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)
18. bad_masks的含义和作用
19. active_masks是如何起作用的，起作用后的效果
20. available_actions起作用的方式
21. mat默认使用gae计算advantage，其中可选的可以用到popart来计算或者valuenorm来计算，或者都不用，advantage计算在buffer里
22. 在compute_returns中计算advantage的时候，delta里面`self.masks的作用
23. 明确return，value_pred和gae的计算
24. value_normalizer实现的方法
25 _shuffle_agent_grid 并没有带乱agent的顺序



'''

'''
mpe环境
每个智能体有5个离散的动作
每个智能体obs有21维，一个列表，共三个
share_obs_dim有63维，一个列表，3个agent有3个63维



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



