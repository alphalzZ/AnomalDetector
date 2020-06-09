
import utils, config
from torch.utils.data import dataloader, dataset


def main(args):
    CDF, AHs = utils.DataLoader(args.infile, Single=args.Single)
    print("main function")


if __name__=="__main":
    args = config.get_args()
    main(args)