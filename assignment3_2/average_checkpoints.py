import torch
import argparse
import os
from collections import OrderedDict

def average_checkpoints(checkpoint_paths):
    """ 加载并平均多个检查点的模型参数 """

    print(f"Averaging checkpoints: {checkpoint_paths}")

    avg_state_dict = None
    valid_paths_count = 0

    # 遍历所有检查点
    for path in checkpoint_paths:
        if not os.path.exists(path):
            print(f"Warning: Checkpoint not found at {path}, skipping.")
            continue

        print(f"Loading checkpoint: {path}")
        state_dict = torch.load(path, map_location='cpu')
        model_state_dict = state_dict['model']

        if avg_state_dict is None:
            # 初始化平均字典
            avg_state_dict = OrderedDict()
            for key, value in model_state_dict.items():
                avg_state_dict[key] = value.clone().float()
        else:
            # 累加参数
            for key, value in model_state_dict.items():
                if key in avg_state_dict:
                    avg_state_dict[key] += value.float()
                else:
                    print(f"Warning: Key {key} not found in avg_state_dict. Skipping.")

        valid_paths_count += 1

    # 计算平均值
    if avg_state_dict and valid_paths_count > 0:
        print(f"Averaging over {valid_paths_count} valid checkpoints.")
        for key in avg_state_dict:
            avg_state_dict[key] /= valid_paths_count

    return avg_state_dict, valid_paths_count

def main():
    parser = argparse.ArgumentParser(description="Average NMT checkpoints")
    parser.add_argument('--inputs', required=True, nargs='+',
                        help='Paths to the checkpoints to average')
    parser.add_argument('--output', required=True,
                        help='Path to save the averaged checkpoint')
    args = parser.parse_args()

    # 计算平均参数
    avg_model_state, count = average_checkpoints(args.inputs)

    if avg_model_state is None:
        print("No valid checkpoints found. Exiting.")
        return

    # 加载一个检查点（例如最后一个）以复制其元数据（如 args）
    try:
        # 尝试加载 args.inputs 中的最后一个有效路径
        base_path = next(p for p in reversed(args.inputs) if os.path.exists(p))
        base_checkpoint = torch.load(base_path, map_location='cpu')

        # 使用平均后的模型参数替换
        base_checkpoint['model'] = avg_model_state

        # 保存新的检查点
        torch.save(base_checkpoint, args.output)
        print(f"Averaged checkpoint saved to: {args.output}")
    except Exception as e:
        print(f"Error loading base checkpoint or saving new one: {e}")

if __name__ == '__main__':
    main()
