import os

from torch.utils.data import dataset
import torch
import scipy.io as sio
import numpy as np


class DataLoader(object):
    def __init__(self, infile, Single):
        self.infile = infile
        self.Single = Single

    def __get_files(self):
        filenames = os.listdir(self.infile)
        return filenames

    def __single_load(self, file):
        data_dict = sio.loadmat(file)
        Ahs, CDF, Qam, Osnr = data_dict["AH"], data_dict["Fe"], \
                              data_dict["label"][:, 0], data_dict["label"][:, 1]
        return (Ahs, CDF, Qam, Osnr)

    def __multi_data(self, data):
        Ahs, CDF, Qam, Osnr = [], [], [], []
        for d in data:
            Ahs.extend(d[0])
            CDF.extend(d[1])
            Qam.extend(d[2])
            Osnr.extend(d[3])
        return (np.matrix(Ahs), np.matrix(CDF), np.matrix(Qam).squeeze(0), np.matrix(Osnr).squeeze(0))

    def load_data(self):
        filenames = self.__get_files()
        data_single = []
        for filename in filenames:
            data_single.append(self.__single_load(self.infile+filename))
        if not self.Single:
            data_multi = self.__multi_data(data_single)
            return data_multi  # (Ahs, CDF, Qam, Osnr) [12000,100] [1,12000]
        else:
            return data_single  # [(Ahs, CDF, Qam, Osnr),...,(Ahs, CDF, Qam, Osnr)] 长度不定，每一个tuple等长


class Dataset(dataset.Dataset):
    def __init__(self, *args):
        super(Dataset, self)
        Ahs, CDF, Osnr, Qam = args[0]
        self.Ahs = torch.Tensor(self.normalization(Ahs))
        self.CDF = torch.Tensor(CDF)
        self.Osnr = torch.Tensor(Osnr)
        self.Qam = torch.LongTensor(Qam)

    def __len__(self):
        return len(self.Ahs)

    def __getitem__(self, idx):
        return self.Ahs[idx], self.CDF[idx], self.Osnr[idx],self.Qam[idx]

    def normalization(self,data):
        mu = np.mean(data,axis=0)
        std = np.std(data,axis=0)
        return (data-mu)/std


if __name__ == "__main__":

    infile = u'D:/Simulation/北邮数据/DataProcessing/dataNew/'
    loader = DataLoader(infile, Single=False)
    dataset = loader.load_data()
    dataloder = Dataset(dataset)
    print(next(iter(dataloder)))

