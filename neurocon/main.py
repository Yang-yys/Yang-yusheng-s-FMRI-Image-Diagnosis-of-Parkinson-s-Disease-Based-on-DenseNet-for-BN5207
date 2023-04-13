import get_filedata as gf
import numpy as np
import nibabel as nib
from nilearn import image
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch import nn,optim
from DenseNet import DenseNet3D

def main():
    print(torch.__version__)
    print(torch.cuda.is_available())
    load_path = 'neurocon'
    [control_anat_data_path,control_func_data_path,patient_anat_data_path,patient_func_data_path] = gf.get_filepath(load_path)
    print('controlanat_data_path',len(control_anat_data_path))#(64,64,27,137)
    print('controlfunc:data_path',len(control_func_data_path))#(64,64,27,4247)
    print('patientanat_data_path',len(patient_anat_data_path))
    print('patientfunc:data_path',len(patient_func_data_path))#(64,64,27,7398)

    # 平滑参数fwhm
    fwhm = 6

    #get data
    # 读取数据并进行平滑
    cf_images = []
    for file_name in control_func_data_path:
        # 读取数据
        cf_img = nib.load(control_func_data_path[file_name])
        # 进行平滑
        smoothed_image = image.smooth_img(cf_img, fwhm=fwhm).get_fdata()
        fmri_data_3d = np.mean(smoothed_image, axis=3)
        # 添加到列表中
        cf_images.append(fmri_data_3d[..., np.newaxis])
    # 将多个平滑后的数据合成为一个数组
    cf_images = np.concatenate(cf_images, axis=3)
    cf_images = np.squeeze(cf_images)
    print('cf_img',cf_images.shape)

    pf_images = []
    for file_name in patient_func_data_path:
        # 读取数据
        pf_img = nib.load(patient_func_data_path[file_name])
        # 进行平滑
        smoothed_image = image.smooth_img(pf_img, fwhm=fwhm).get_fdata()
        fmri_data_3d = np.mean(smoothed_image, axis=3)
        # 添加到列表中
        pf_images.append(fmri_data_3d[..., np.newaxis])
    # 将多个平滑后的数据合成为一个数组
    pf_images = np.concatenate(pf_images, axis=3)
    pf_images = np.squeeze(pf_images)
    print('pf_img',pf_images.shape)

    #divide train test
    cf_test = cf_images[:,:,:,0:5]
    cf_train = cf_images[:,:,:,5:31]
    pf_test = pf_images[:,:,:,0:10]
    pf_train = pf_images[:,:,:,10:54]
    print(cf_test.shape,cf_train.shape)
    print(pf_test.shape,pf_train.shape)

    old_test_data = np.concatenate([cf_test, pf_test], axis=3)
    test_label = np.concatenate([np.zeros(5), np.ones(10)])
    test_data = np.moveaxis(old_test_data, -1, 0)
    # test_data = np.moveaxis(test_data, -1, 1)
    print('old shape',old_test_data.shape,'new shape',test_data.shape)

    old_train_data = np.concatenate([cf_train, pf_train], axis=3)
    train_label = np.concatenate([np.zeros(26), np.ones(44)])#(h,w,c,l)
    train_data = np.moveaxis(old_train_data, -1, 0)#(l,h,w,c)
    # train_data = np.moveaxis(train_data, -1, 1)#(l,c,h,w)
    print('old shape',old_train_data.shape,'new shape',train_data.shape)

    assert np.array_equal(old_test_data[..., 0], test_data[0])
    print('test[0] equal')
    assert np.array_equal(old_test_data[...,6], test_data[6])
    print('test[6] equal')
    assert np.array_equal(old_train_data[..., 0], train_data[0])
    print('train[0] equal')
    assert np.array_equal(old_train_data[..., 5], train_data[5])
    print('train[5] equal')

    # test_data = np.expand_dims(test_data,axis=1)
    # train_data = np.expand_dims(train_data, axis=1)
    print('test_data',test_data.shape,'train_data',train_data.shape)
    print('telabel',test_label.shape,'trlabel',train_label.shape)
    test_s = TensorDataset(torch.from_numpy(test_data).unsqueeze(1), torch.from_numpy(test_label))
    train_s = TensorDataset(torch.from_numpy(train_data).unsqueeze(1), torch.from_numpy(train_label))
    print('ted',test_data.shape,'tel',test_label.shape,'trd',train_data.shape,'trl',train_label.shape)


    # 创建DataLoader
    batch_size = 8
    train_dataloader = DataLoader(train_s, batch_size=batch_size, shuffle=True, num_workers=2)
    test_dataloader = DataLoader(test_s, batch_size=batch_size, shuffle=True, num_workers=2)
    # 定义模型
    model = DenseNet3D()

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.00001)

    # 训练模型
    num_epochs = 20
    print(torch.cuda.is_available())
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    print('start epoch')
    for epoch in range(num_epochs):
        train_loss = 0.0
        train_acc = 0.0
        model.train()
        for i, (inputs, labels) in enumerate(train_dataloader):
            inputs = inputs.float()
            labels = labels.long()
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            train_acc += torch.sum(preds == labels.data)

        train_loss /= len(train_s)
        train_acc = train_acc.double() / len(train_s)

        # 在验证集上测试模型
        val_loss = 0.0
        val_acc = 0.0
        model.eval()

        with torch.no_grad():
            for inputs, labels in test_dataloader:
                inputs = inputs.float()
                labels = labels.long()
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                val_acc += torch.sum(preds == labels.data)

        val_loss /= len(test_s)
        val_acc = val_acc.double() / len(test_s)

        print(
            f'Epoch {epoch + 1} - train loss: {train_loss:.4f}, train acc: {train_acc:.4f}, val loss: {val_loss:.4f}, val acc: {val_acc:.4f}')


if __name__ == '__main__':
    main()