import argparse
import os
import sys


def main():
    # 参数解析
    parser = argparse.ArgumentParser(description='ReFlow and MeanFlow main script')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'eval'], help='Mode')
    
    # 解析模式参数，保留其他未知参数
    args, unknown_args = parser.parse_known_args()
    mode = args.mode
    
    # 重新构建sys.argv，去掉--mode参数，保留其他参数
    new_argv = [sys.argv[0]]  # 保留脚本名
    
    # 添加未知参数（即除了--mode之外的所有参数）
    new_argv.extend(unknown_args)
    
    # 临时替换sys.argv
    original_argv = sys.argv
    sys.argv = new_argv
    
    try:
        # 根据模式调用相应的脚本
        if mode == 'train':
            # 添加scripts目录到路径
            sys.path.append(os.path.dirname(os.path.abspath(__file__)))
            from scripts.train import main as train_main
            train_main()
        elif mode == 'eval':
            # 添加scripts目录到路径
            sys.path.append(os.path.dirname(os.path.abspath(__file__)))
            from scripts.evaluate import main as eval_main
            eval_main()
        else:
            print(f"Unknown mode: {mode}")
            sys.exit(1)
    finally:
        # 恢复原始的sys.argv
        sys.argv = original_argv


if __name__ == '__main__':
    main()
