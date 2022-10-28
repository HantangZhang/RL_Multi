import argparse

def get_config():
    parser = argparse.ArgumentParser(
        description='onpolicy', formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument("--algorithm_name", type=str,
                        default='MAT', choices=["MAT", "mat_dec", "mat_encoder", "mat_decoder", "mat_gru"])


    # 重要的参数

    # env相关
    parser.add_argument("--use_obs_instead_of_state", action='store_true',
                        default=False, help="Whether to use global state or concatenated obs")

    parser.add_argument("--num_env_steps", type=int, default=10e6,
                        help='Number of environment steps to train (default: 10e6)')

    parser.add_argument("--n_rollout_threads", type=int, default=32,
                        help="Number of parallel envs for training rollouts")



    # 算法相关
    parser.add_argument("--episode_length", type=int,
                        default=200, help="Max length for any episode")

    parser.add_argument("--use_linear_lr_decay", action='store_true',
                        default=False, help='use a linear schedule on the learning rate')


    # 神经网络
    parser.add_argument("--hidden_size", type=int, default=64,
                        help="Dimension of hidden layers for actor/critic networks")

    parser.add_argument("--recurrent_N", type=int, default=1, help="The number of recurrent layers.")


    # optimizer parameters
    parser.add_argument("--lr", type=float, default=5e-4,
                        help='learning rate (default: 5e-4)')
    parser.add_argument("--critic_lr", type=float, default=5e-4,
                        help='critic learning rate (default: 5e-4)')
    parser.add_argument("--opti_eps", type=float, default=1e-5,
                        help='RMSprop optimizer epsilon (default: 1e-5)')

    # 运行相关
    parser.add_argument("--cuda", action='store_false', default=True, help="by default True, will use GPU to train; or else will use CPU;")

    parser.add_argument("--n_training_threads", type=int,
                        default=1, help="Number of torch threads for training")

    parser.add_argument("--use_centralized_V", action='store_false',
                            default=True, help="Whether to use centralized V function")

    # save parameters
    parser.add_argument("--save_interval", type=int, default=100,
                        help="time duration between contiunous twice models saving.")

    # log parameters
    parser.add_argument("--log_interval", type=int, default=5,
                        help="time duration between contiunous twice log printing.")

    # eval parameters
    parser.add_argument("--use_eval", action='store_true', default=False,
                        help="by default, do not start evaluation. If set`, start evaluation alongside with training.")
    parser.add_argument("--eval_interval", type=int, default=25,
                        help="time duration between contiunous twice evaluation progress.")
    parser.add_argument("--eval_episodes", type=int, default=32, help="number of episodes of a single evaluation.")

    # 相对不重要的参数
    parser.add_argument("--seed", type=int, default=1, help="Random seed for numpy/torch")
    # pretrained parameters
    parser.add_argument("--model_dir", type=str, default=None, help="by default None. set the path to pretrained model.")


    return parser