from model_utils import load_model,save_model
import os

def train(train_loader,model,criterion, optimizer, num_epochs,device, path = "./model/",
          input_channel=100,training_datasize=2000,NET_NAME='TEST_NET_1'):
    model.train()
    dirname = 'inputs_'+str(input_channel)+'dataset_'+str(training_datasize)+'_'+NET_NAME
    if not os.path.exists(os.path.join(path,dirname)):
        os.makedirs(os.path.join(path,dirname))
    for epoch in range(num_epochs):
        for batch_idx, sample in enumerate(train_loader):
            inputs,target = sample['input'].to(device),sample['output'].to(device)

            # forward
            out = model(inputs)
            loss = criterion(out, target)
            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if epoch < 100: 
            print('Epoch[{}/{}], loss: {:.6f}'
                  .format(epoch+1, num_epochs, loss.data[0]))
        elif (epoch+1) % 100 == 0:
            print('Epoch[{}/{}], loss: {:.6f}'
                  .format(epoch+1, num_epochs, loss.data[0]))
        if (epoch+1) % 500 == 0:
            filename = 'inputs_'+str(input_channel)+'dataset_'+str(training_datasize)+'epoch_'+str(epoch)+'_'+NET_NAME+'.pth'
            save_model(model, optimizer, path = os.path.join(path,dirname), filename=filename) 

def train_onebyone(train_loader,model,criterion, optimizer,device):
    model.train()
    for batch_idx, sample in enumerate(train_loader):
        inputs,target = sample['input'].to(device),sample['output'].to(device)

        # forward
        out = model(inputs)
        loss = criterion(out, target)
        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    return loss
            
            
def test(testing_dataloader,model,device):
    model.eval()
    for batch_idx, sample in enumerate(testing_dataloader):
        inputs,target = sample['input'].to(device),sample['output'].to(device)
        predict_test = model(inputs)
        predict_test = predict_test.data.cpu().numpy()
        gt_test = target.data.cpu().numpy()
    return predict_test,gt_test
