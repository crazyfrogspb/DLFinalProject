import argparse

from dlfinalproject.models_rotation.train_model_clf import train_model

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Watermark model')

    parser.add_argument('--checkpoint_file', type=str, default=None)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--decay', type=float, default=0.0)
    parser.add_argument('--image_folders', type=str,
                        nargs='+', default=['unsupervised'])
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--n_epochs', type=int, default=500)
    parser.add_argument('--eval_interval', type=int, default=500)
    parser.add_argument('--model_file', type=str, default='clf.pth')
    parser.add_argument('--restart_optimizer', action='store_true')
    parser.add_argument('--early_stopping', type=int, default=25)
    parser.add_argument('--run_uuid', type=str, default=None)
    parser.add_argument('--finetune', type=str, default=None)
    parser.add_argument('--augmentation', type=str, default=None)
    parser.add_argument('--architecture', type=str, default='resnet50')

    args = parser.parse_args()
    args_dict = vars(args)
    train_model(**args_dict)
