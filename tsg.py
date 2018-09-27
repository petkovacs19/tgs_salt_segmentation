import argparse
from models.model_factory import make_model
from dataset.tsg_data import TSGSaltDataset

def main(args):
    model = make_model(args.model, (101, 101, 3))
   

    #Creating dataset
    dataset = TSGSaltDataset(train_data_path=args.train_path, val_data_path=args.val_path, batch_size=args.batch_size, seed=args.seed)
    train_data_generator = dataset.get_train_data_generator()    
    val_data_generator = dataset.get_val_data_generator()


if __name__== "__main__":
    parser = argparse.ArgumentParser(description = 'TSGSaltModel')
    parser.add_argument('--model', type=str, help='Name of backbone architecture', default="resnet34")
    parser.add_argument('--batch-size', type=int, help='Data batch size', default=32)
    parser.add_argument('--seed', type=int, help='Seed value for data generator', default=1)
    parser.add_argument('--train_path', type=str, help='Path to the training data', default='/home/pkovacs/tsg/data/train')
    parser.add_argument('--val_path', type=str, help='Path to the val data', default='/home/pkovacs/tsg/data/val')
    args = parser.parse_args()
    main(args)