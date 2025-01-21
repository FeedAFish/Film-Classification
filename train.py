import utils.dataloader as dataloader
import utils.model_train as train
import argparse

parser = argparse.ArgumentParser(
    prog="Image Classification",
    description="Train supervised learning model that clasify images",
    epilog="---",
)
parser.add_argument(
    "-o",
    "--output",
    default="model.mdl",
    help="Path to save model. Default = 'model.mdl'",
)

parser.add_argument(
    "-d",
    "--data",
    default=r"1qNOpvoebvdfoEGf7IUHxXAf8mEWD0z48",
    help="Data id on google drive. Must be a zip file that is well-fitted pytorch.ImageFolder",
)
parser.add_argument(
    "-i",
    "--input",
    help="Continue train for a trained model. Default = 'model.mdl'",
)

args = parser.parse_args()

dataloader.down_n_extract(args.data)

data = dataloader.Dataset("data")

if args.input:
    model = train.SimpleNN.load_model(args.input)
    model.eval()
else:
    model = train.SimpleNN(len(data.data_train.classes))

model.train_epochs(data, save_file=args.output, epochs=15)
