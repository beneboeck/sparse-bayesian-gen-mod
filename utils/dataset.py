import torch
from torch.utils.data import DataLoader, Dataset

###
# standard dataset
###
class default_dataset(Dataset):
    def __init__(self,y):
        super().__init__()
        self.y = y.float()

    def __len__(self):
        return self.y.size(0)

    def __getitem__(self,idx):
        return self.y[idx,:]

###
# dataset which also contains the ground truth data (or any additional data, e.g., varying A)
###
class dataset_with_gt(Dataset):
    def __init__(self,x,y):
        super().__init__()
        self.x = x.float()
        self.y = y.float()

    def __len__(self):
        return self.x.size(0)

    def __getitem__(self,idx):
        return self.x[idx,:],self.y[idx,:]

def default_ds_dl_split(X_train,X_val,X_test,bs_train):
    """standard split into training, validation and test set and loader"""
    ds_train = default_dataset(X_train)
    ds_val = default_dataset(X_val)
    ds_test = default_dataset(X_test)
    dl_train = DataLoader(ds_train, shuffle=True, batch_size=bs_train)
    dl_val = DataLoader(ds_val, shuffle=True, batch_size=bs_train)
    dl_test = DataLoader(ds_test, shuffle=True, batch_size=bs_train)
    return ds_train,ds_val,ds_test,dl_train,dl_val,dl_test

def ds_dl_split_with_gt(X_train,X_val,X_test,Y_train,Y_val,Y_test,bs_train):
    """split into training, validation and test set and loader using also ground-truth data"""
    ds_train = dataset_with_gt(X_train,Y_train)
    ds_val = dataset_with_gt(X_val,Y_val)
    ds_test = dataset_with_gt(X_test,Y_test)
    dl_train = DataLoader(ds_train, shuffle=True, batch_size=bs_train)
    dl_val = DataLoader(ds_val, shuffle=True, batch_size=bs_train)
    dl_test = DataLoader(ds_test, shuffle=True, batch_size=bs_train)
    return ds_train,ds_val,ds_test,dl_train,dl_val,dl_test

def ds_dl_split_with_A(A_train,A_val,A_test,Y_train,Y_val,Y_test,bs_train):
    """split into training, validation and test set and loader using also varying A"""
    ds_train = dataset_with_gt(A_train,Y_train)
    ds_val = dataset_with_gt(A_val,Y_val)
    ds_test = dataset_with_gt(A_test,Y_test)
    dl_train = DataLoader(ds_train, shuffle=True, batch_size=bs_train)
    dl_val = DataLoader(ds_val, shuffle=True, batch_size=len(ds_val))
    dl_test = DataLoader(ds_test, shuffle=True, batch_size=len(ds_test))
    return ds_train,ds_val,ds_test,dl_train,dl_val,dl_test



