import os
import yaml
import pickle
import argparse

def parse_configure(model=None, dataset=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='lightgcn_agr', help='Model name')
    parser.add_argument('--dataset', type=str, default='amazon', help='Dataset name')
    parser.add_argument('--device', type=str, default='cuda', help='cpu or cuda')
    parser.add_argument('--seed', type=int, default=2025, help='Random number')
    parser.add_argument('--cuda', type=str, default='0', help='Device number')
    parser.add_argument('--diverse', type=int, default=2, help='Diverse profile number')
    args, _ = parser.parse_known_args()

    if args.device == 'cuda':
        os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda

    model_name = model.lower() if model else args.model.lower() if args.model else 'default'
    if dataset:
        args.dataset = dataset

    pre_dir = os.getcwd()
    config_path = f"{pre_dir}/config/models_config/{model_name}.yml"
    if not os.path.exists(config_path):
        raise Exception("Please create the yaml file for your model first.")

    with open(config_path, encoding='utf-8') as f:
        configs = yaml.safe_load(f)
    configs['model']['name'] = configs['model']['name'].lower()
    configs.setdefault('tune', {'enable': False})
    configs['device'] = args.device
    configs['diverse'] = args.diverse
    if args.dataset:
        configs['data']['name'] = args.dataset
    if args.seed:
        configs['train']['seed'] = args.seed

    user_embedding_path = f"{pre_dir}/data/{configs['data']['name']}/usr_emb_np.pkl"
    item_embedding_path = f"{pre_dir}/data/{configs['data']['name']}/itm_emb_np.pkl"
    with open(user_embedding_path, 'rb') as f:
        configs['user_embedding'] = pickle.load(f)
    with open(item_embedding_path, 'rb') as f:
        configs['item_embedding'] = pickle.load(f)

    # for index in range(configs['diverse'] - 2):
    #     user_embedding_index_path = f"{pre_dir}/data/{configs['data']['name']}/diverse_profile/diverse_user_embedding_{index + 1}.pkl"
    #     item_embedding_index_path = f"{pre_dir}/data/{configs['data']['name']}/diverse_profile/diverse_item_embedding_{index + 1}.pkl"
    #     with open(user_embedding_index_path, 'rb') as f:
    #         configs[f'user_embedding_{index + 1}'] = pickle.load(f)
    #     with open(item_embedding_index_path, 'rb') as f:
    #         configs[f'item_embedding_{index + 1}'] = pickle.load(f)

    return configs

configs = parse_configure()