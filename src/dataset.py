import torch
import pickle
import numpy as np
from torch.utils.data import Dataset, DataLoader

class DebrisDataset(Dataset):
    def __init__(self, n_rows, steps, root_dir='Dataset', transform=None, load=False):
        self.root_dir = root_dir
        self.n_rows = n_rows
        self.steps = steps   
        self.elem = {}
        self.lines = []
        self.names = []
        self.load = load

        str1 = "" 

        if(self.load):
            for i in range(1,self.n_rows):
                str1 = "0"+str(i) if i < 10 else str(i)
                str1 = self.root_dir + "/jan/file" + str1 + ".txt"

                with open(str1, 'r') as myfile:
                    self.lines = myfile.readlines()
                
                self.lines = np.reshape(self.lines, (-1,2))
                
                last_debris_no = 0
                for line in self.lines:
                    temp1 = line[1].split()

                    if len(temp1) == 8:
                        temp1[7] = '%.8f'%(float(temp1[7]))

                    debris_no = temp1[1]
                    if last_debris_no == debris_no:
                        continue
                    else:
                        last_debris_no = debris_no

                    temp1 = temp1[2:8]

                    if i == 1:
                        self.names.append(debris_no)

                    if(debris_no in self.names):
                        if(self.elem.get(debris_no) == None):
                            self.elem[debris_no] = []

                        if(len(self.elem[debris_no]) != i): 
                            self.elem[debris_no].append(temp1)

            to_be_deleted = []
            for ele in self.elem:
                if(len(self.elem[ele]) < 20):
                    to_be_deleted.append(ele)

            for ele in to_be_deleted:
                self.elem.pop(ele, None)
                self.names.remove(ele)
            file_handler = open('elem_pickled.dat', 'wb+')
            pickle.dump((self.elem, self.names), file_handler)
            file_handler.close()

        else:
            file_handler = open('elem_pickled.dat', 'rb+')
            self.elem, self.names = pickle.load(file_handler)
            file_handler.close()
    
    def __len__(self):
        return len(self.elem)

    def __getitem__(self, idx):
        debris_name = self.names[idx]
        sample = np.array(self.elem[debris_name]).astype(np.float32)
        sequences = []
        predictions = []
        for i in range(0, 10):
            sequences.append(sample[i:i+self.steps])
            predictions.append(sample[i+self.steps: i+self.steps+1])
        return sequences, predictions

    
if __name__ == '__main__':
    transformed_dataset = DebrisDataset(n_rows=30, steps= 10)
    dataloader = DataLoader(transformed_dataset, batch_size=128, shuffle=True, num_workers=1)
    dataloader = iter(dataloader)
    for i in range(0, len(dataloader)):
        (sequences, predictions) = next(dataloader)
        print(sequences, predictions)
        break
        
    
