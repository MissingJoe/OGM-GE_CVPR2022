import SimpleITK as sitk
import torch
import numpy as np
import os, sys
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset, Sampler
import json
from collections import Counter


class SubSet(Dataset):
    def __init__(self, data, label, index):
        self.data = []
        self.label = []
        for ind in index:
            self.data.extend(data[ind[0]:ind[1]])
            self.label.extend(label[ind[0]:ind[1]])

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        resize = transforms.Resize([150, 150])
        img1 = self.data[index][0]
        img2 = self.data[index][1]
        label = self.label[index]
        img1 = torch.tensor(img1)
        img1 = img1.float()
        img2 = torch.tensor(img2)  # totensor
        img2 = img2.float()
        img11 = transforms.ToPILImage()(img1)
        img22 = transforms.ToPILImage()(img2)
        img1 = resize(img11)
        img2 = resize(img22)
        img1 = transforms.ToTensor()(img1)  # totensor
        img2 = transforms.ToTensor()(img2)
        img1 = img1 * 256
        img2 = img2 * 256
        return img1, img2, label


class TumorDataset(Dataset):
    def get_medical_image(self, path):
        if isinstance(path, sitk.Image):
            reader = path
        else:
            assert os.path.exists(path), "{} is not existed".format(path)
            assert os.path.isfile(path), "{} is not a file".format(path)
            reader = sitk.ReadImage(path)
        array = sitk.GetArrayFromImage(reader)
        return array

    def __init__(self, data_dir, label_dir):
        self.data_dir = data_dir
        self.label_dir = label_dir
        # self.label_dict = {"PDAC": 0, "IPMN": 1, "other": 2, "XianLinAi": 3, "SCN": 4, "NET": 5, "CP": 6, "MCN": 7}
        # self.label_dict = {"PDAC": 0, "IPMN": 1, "other": 2, "SCN": 3, "XianLinAi": 4}
        self.label_dict = {"XianLinAi": 4}
        self.cls_num_list = []
        self.tumor_slice_dict = {}
        self.dataset = []
        self.label_sum = 0
        self.targets = []
        print("初始化标签字典")
        if os.path.exists("tumor_slice_dict_debug.txt"):
            with open('tumor_slice_dict_debug.txt', 'r') as file:
                self.tumor_slice_dict = json.loads(file.read())
        else:
            label_list = os.listdir(label_dir)
            i = 0
            for name in label_list:
                path = os.path.join(label_dir, name)
                tumor = self.get_medical_image(path)
                arr1 = np.array(np.nonzero(tumor))
                if arr1[0].size == 0:
                    continue
                slice = int(Counter(list(arr1[0])).most_common(1)[0][0])
                dim = arr1[:, np.where(arr1[0] == slice)][1:]
                tempList = []
                tempList.append(slice)
                tempList.append(np.max(dim[0][0]))
                tempList.append(np.min(dim[0][0]))
                tempList.append(np.max(dim[1][0]))
                tempList.append(np.min(dim[1][0]))
                self.tumor_slice_dict[name] = np.trunc(tempList).astype(int).tolist()
            with open('tumor_slice_dict.txt', 'w') as file:
                file.write(json.dumps(self.tumor_slice_dict))

        print("初始化数据集")
        for key in self.label_dict.keys():
            cls_sum = 0
            path = os.path.join(data_dir, key)
            patientList = os.listdir(path)
            for patient in patientList:
                temp = patient + '.nii.gz'
                if temp in self.tumor_slice_dict.keys():
                    dataPath1 = os.path.join(path, patient, 'c_1.nii.gz')
                    dataPath2 = os.path.join(path, patient, 'c_2.nii.gz')
                    modal1 = self.get_medical_image(dataPath1)
                    modal2 = self.get_medical_image(dataPath2)

                    modalSlice = self.tumor_slice_dict[temp][0]
                    modalWidthHigh = self.tumor_slice_dict[temp][1] + 20
                    modalWidthLow = self.tumor_slice_dict[temp][2] - 20
                    modalHeightHigh = self.tumor_slice_dict[temp][3] + 20
                    modalHeightLow = self.tumor_slice_dict[temp][4] - 20
                    data1 = modal1[modalSlice]
                    data2 = modal2[modalSlice]
                    data1 = data1[modalWidthLow:modalWidthHigh, modalHeightLow:modalHeightHigh]
                    data2 = data2[modalWidthLow:modalWidthHigh, modalHeightLow:modalHeightHigh]
                    data1[data1 < -160] = -160
                    data1[data1 > 240] = 240
                    data2[data2 < -160] = -160
                    data2[data2 > 240] = 240
                    data1 = (data1 - data1.mean()) / data1.std()
                    data2 = (data2 - data2.mean()) / data2.std()
                    # modal = np.stack((modal1, modal2), axis=0)
                    temp = []
                    temp.append(data1)
                    temp.append(data2)
                    self.dataset.append(temp)
                    cls_sum += 1
            self.cls_num_list.append(cls_sum)
        with open('cls_num_list.txt', 'w') as file:
            file.write(json.dumps(self.cls_num_list))
        label = 0
        for cls in self.cls_num_list:
            for i in range(cls):
                self.targets.append(label)
            label += 1
            self.label_sum += cls
        print(self.cls_num_list)
        print(len(self.dataset))

    def split(self):
        start = 0
        train_index = []
        val_index = []
        for cls in self.cls_num_list:
            end = cls * 4 // 5
            temp = []
            temp.append(start)
            temp.append(end + 1)
            train_index.append(temp)
            temp = []
            temp.append(end + 1)
            temp.append(start + cls)
            val_index.append(temp)
            start += cls
        return SubSet(self.dataset, self.targets, train_index), SubSet(self.dataset, self.targets, val_index)


class TumorDataLoaderNew(DataLoader):
    def __init__(self, data_dir, label_dir, batch_size, num_workers, training=True):
        if training:
            data = TumorDataset(data_dir, label_dir)
            self.train_dataset, self.val_dataset = data.split()
        else:  # test
            # self.dataset = TumorDataset(data_dir, data_dir + '/ImageNet_LT_val.txt')
            self.val_dataset = None

        self.cls_num_list = data.cls_num_list

        self.init_kwargs = {
            'batch_size': 512,
            'shuffle': True,
            'num_workers': num_workers,
            'drop_last': False
        }
        super().__init__(dataset=self.train_dataset, **self.init_kwargs, sampler=None)

    def split_validation(self):
        return DataLoader(
            dataset     = self.val_dataset ,
            batch_size  = 512                                                  ,
            shuffle     = True                                                 ,
            num_workers = 32                                                    ,
            drop_last   = False
        )
