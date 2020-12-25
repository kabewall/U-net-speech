# -*- coding: utf-8 -*-
"""
Created on Thu Dec 24 17:56:07 2020

@author: zankyo
"""

import torch
import torch.nn as nn
# from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
import datetime
import os
from tqdm import tqdm
import matplotlib.pyplot as plt

import parameter as C
import myutils as ut
import network

def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("CUDA is available:", torch.cuda.is_available())
    
    if not os.path.exists(C.MUSDB_model):
            os.mkdir(C.MUSDB_model)
    
    now = datetime.datetime.now()
    model_path = C.MUSDB_model+"/model_"+now.strftime('%Y%m%d_%H%M%S')+".pt"
    print(model_path)
    
    #dataset
    train_loader,val_loader = ut.MyDataLoaderMUSDB(MODE = "train")
    
    #model
    model = network.UnetConv2()
    model = model.to(device)
    
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=C.learning_rate)
    
    train_loss_list = []
    val_loss_list = []
    
    # writer = SummaryWriter(log_dir="./logs")
    
    for epoch in tqdm(range(1, C.epochs+1),desc='[Training..]',leave=True):
        
        #train
        model.train()
        train_loss = 0
    
        for batch_idx, (speech, addnoise) in enumerate(train_loader):
            speech= speech.to(device)
            addnoise=addnoise.to(device)
            optimizer.zero_grad()
            
            mask = model(addnoise).to(device)
            enhance = addnoise * mask
            
            loss = criterion(enhance, speech)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
    
        train_loss /= len(train_loader)
        train_loss_list.append(train_loss)
        # writer.add_scalar("train_loss", train_loss, epoch)
        
        #test
        model.eval()
        eval_loss = 0
    
        # with torch.no_grad():
        #     for speech, addnoise in val_loader:
        #         speech= speech.to(device)
        #         addnoise=addnoise.to(device)
        #         mask = model(addnoise).to(device)
        #         enhance = addnoise * mask
    
        #         loss = criterion(enhance, speech)
        #         eval_loss += loss.item()
        #     eval_loss /= len(val_loader)
        #     val_loss_list.append(eval_loss)
        #     tqdm.write('\nTrain set: Average loss: {:.6f}\nVal set:  Average loss: {:.6f}'
        #                         .format(train_loss,eval_loss))
        
        
        for speech, addnoise in val_loader:
            speech= speech.to(device)
            addnoise=addnoise.to(device)
            mask = model(addnoise).to(device)
            enhance = addnoise * mask

            loss = criterion(enhance, speech)
            eval_loss += loss.item()
            
        eval_loss /= len(val_loader)
        val_loss_list.append(eval_loss)
        # writer.add_scalar("eval_loss", eval_loss, epoch)
        tqdm.write('\nTrain set: Average loss: {:.10f}\nVal set:  Average loss: {:.10f}'
                            .format(train_loss,eval_loss))
    
        if epoch == 1:
                best_loss = eval_loss
                torch.save(model.state_dict(), model_path)
    
        else:
            if best_loss > eval_loss:
                torch.save(model.state_dict(), model_path)
                best_loss = eval_loss
    
        if epoch % 10 == 0: #10回に１回定期保存
            epoch_model_path = C.MUSDB_model+"/model_"+now.strftime('%Y%m%d_%H%M%S')+"_Epoch"+str(epoch)+".pt"
            torch.save(model.state_dict(), epoch_model_path)
            
    # writer.close()
    
    # 結果の出力と描画
    plt.figure()
    plt.plot(range(1, C.epochs+1),train_loss_list, label='train_loss')
    plt.plot(range(1, C.epochs+1),val_loss_list, label='test_loss')
    plt.xlabel('epoch')
    plt.legend()
    plt.savefig(C.MUSDB_model+"/model_"+now.strftime('%Y%m%d_%H%M%S')+'_loss.png')
    
            
if __name__ == "__main__":
    main()