import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader

class DebrisDataset(Dataset):
    def __init__(self, n_rows, steps, root_dir='/home/nevronas/dataset/space-debris', transform=None):
        self.root_dir = root_dir
        self.n_rows = n_rows
        self.steps = steps   
        self.elem = {}
        self.lines = []
        self.names = []

        str1 = "" 
        for i in range(1,self.n_rows):
            str1 = "0" + str(i) if i < 10 else str(i)
            str1 = self.root_dir + "/jan/file" + str1 + ".txt"

            with open(str1, 'r') as myfile:
                self.lines = myfile.readlines()
            
            self.lines = np.reshape(self.lines, (-1,2))
            print(self.lines[0], self.lines[1])
#            print(self.lines.shape)
            
            for line in self.lines:
                temp1 = line[1].split()

                if len(temp1) == 9:
                    elem8 = str(temp1[8])
                    check_sum = elem8[len(elem8)-1]
                    elem8 = elem8[0:len(elem8)-1]
                    temp1[8] = (float(elem8))
                    temp1.append(int(check_sum))
                elif len(temp1) == 8:
                    elem8 = str(temp1[7])
                    check_sum = elem8[len(elem8)-1]
                    dec = 0
                    for i in range(0, len(elem8)-1):
                        if(elem8[i] == '.'):
                            dec = i
                            break

                    elem9 = elem8[dec+8:len(elem8)-1]
                    elem8 = '%.8f'%(float(temp1[7]))
                    temp1[7] = (elem8)
                    temp1.append(int(elem9))
                    temp1.append(int(check_sum))

                if i == 1:
                    self.names.append(temp1[1])

                if(self.elem.get(temp1[1]) == None):
                    self.elem[temp1[1]] = []

                self.elem[temp1[1]].append(temp1)
    
    def __len__(self):
        return len(self.elem)

    def __getitem__(self, idx):
        debris_name = self.names[idx]
        sample = np.array(self.elem[debris_name])
        sample = sample[0:self.steps]
        

        return sample


transformed_dataset = DebrisDataset(n_rows=30, steps= 10)
#dataloader = DataLoader(transformed_dataset, batch_size=4, shuffle=True, num_workers=4)

print("len:" + str(len(transformed_dataset)))
for i in range(len(transformed_dataset)):
    sample = transformed_dataset[i]
