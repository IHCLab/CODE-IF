# Hsieh
# IHCLab, NCKU, Tainan, Taiwan
# Last modify 2022/12/02

import argparse
import torch
import torch.nn.functional as F
import numpy as np
import os
from scipy.io import savemat, loadmat
from torch.utils.data import DataLoader, Dataset
import random
import torch.multiprocessing
from Models import *

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = "cuda"
torch.backends.cudnn.benchmark = True


def parse_args():
    parser = argparse.ArgumentParser(description="Start train my net")

    parser.add_argument(
        "--num_base_chs",
        type=int,
        default=128,
        help="The number of the channels of the base feature",
    )
    parser.add_argument(
        "--num_MSI_chs",
        type=int,
        default=4,
        help="The number of the channels of the HRMSI",
    )
    parser.add_argument(
        "--SRratio", type=int, default=4, help="The super-resolution ratio"
    )

    parser.add_argument("--root", type=str, default="dataset", help="data root folder")
    parser.add_argument("--test_file", type=str, default="simulated.txt")

    parser.add_argument("--prefix", type=str, default="Houston")
    parser.add_argument(
        "--device", type=str, default="cuda:0", help="cuda:device_id or cpu"
    )
    parser.add_argument("--gpus", type=int, default=1)
    args = parser.parse_args()

    return args


def loadTxt(fn):
    a = []
    with open(fn, "r") as fp:
        data = fp.readlines()
        for item in data:
            fn = item.strip("\n")
            a.append(fn)
    return a


class dataset(Dataset):
    def __init__(self, X, args, mode="test"):
        super(dataset, self).__init__()
        self.root = args.root
        self.fns = X
        self.n_images = len(self.fns)
        self.mode = mode

    def arguement(self, img, rotTimes, vFlip, hFlip):
        # Rotation, Vertical Flip and Horizontal Flip
        img = np.rot90(img, rotTimes, axes=(1, 2)).copy()
        if vFlip == 1:
            img = img[:, :, ::-1].copy()
        if hFlip == 1:
            img = img[:, ::-1, :].copy()
        return img

    def __getitem__(self, index):
        datapt = self.root
        fn = os.path.join(os.getcwd(), datapt, self.fns[index])

        ym = loadmat(fn)["Ym"].astype(np.float32)
        ym = np.transpose(ym, (2, 0, 1))
        yh = loadmat(fn)["Yh"].astype(np.float32)
        yh = np.transpose(yh, (2, 0, 1))
        gt = loadmat(fn)["GT"].astype(np.float32)
        gt = np.transpose(gt, (2, 0, 1))

        gt = gt.astype(np.float32)
        gtmin = np.min(gt)
        gtmax = np.max(gt)
        GT = (gt - gtmin) / (gtmax - gtmin)

        if self.mode == "train":
            rotTimes = random.randint(0, 3)
            vFlip = random.randint(0, 1)
            hFlip = random.randint(0, 1)
            ym = self.arguement(ym, rotTimes, vFlip, hFlip)
            yh = self.arguement(yh, rotTimes, vFlip, hFlip)
            GT = self.arguement(GT, rotTimes, vFlip, hFlip)

        return ym, yh, GT, fn  # image, name

    def __len__(self):
        return self.n_images


def psnr(x, y):
    bands = x.shape[2]
    x = np.reshape(x, [-1, bands])
    y = np.reshape(y, [-1, bands])
    msr = np.mean((x - y) ** 2, 0)
    maxval = np.max(y, 0) ** 2
    return np.mean(10 * np.log10(maxval / msr))


def trainer(args):
    ## Reading files #
    print("#" * 80)
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")
    print("#" * 80)

    testfn = loadTxt(os.path.join(os.getcwd(), "function", args.test_file))

    if not os.path.isdir("Small_Data_Result"):
        os.mkdir("Small_Data_Result")

    test_loader = DataLoader(
        dataset(testfn, args, mode="test"),
        batch_size=1,
        shuffle=False,
        pin_memory=False,
        num_workers=1,
    )

    model = myModel(args.SRratio, args.num_MSI_chs, args.num_base_chs).to(args.device)
    model.load_state_dict(
        torch.load(os.path.join(os.getcwd(), "checkpoint", "Chikusei.pth"))
    )

    with torch.no_grad():
        model.eval()

        psnrs = []
        for idx2, (loadinput) in enumerate(test_loader):

            ym, yh, GT, filPath = loadinput
            ym = ym.to(args.device)
            yh = yh.to(args.device)
            GT = GT.to(args.device)

            decoded = model(ym, yh)
            vGT = GT.cpu().numpy()
            vResult = decoded.cpu().numpy()

            for predimg, gtimg, f in zip(vResult, vGT, filPath):
                psnrs.append(psnr(predimg, gtimg))
                savemat(
                    f"Small_Data_Result/{os.path.basename(f)}",
                    {"Z_DE": np.transpose(predimg, (1, 2, 0))},
                )


if __name__ == "__main__":

    args = parse_args()
    trainer(args)
