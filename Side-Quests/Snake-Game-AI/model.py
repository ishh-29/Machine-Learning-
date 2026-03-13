#Importing Necessary Libraries And Modules
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

class Linear_QNet(nn.Module):
    def __init__(self,input_size,hidden_size,output_size):
        super().__init__()
        self.linear1=nn.Linear(input_size,hidden_size)
        self.linear2=nn.Linear(hidden_size,output_size)

    def forward(self,x):
        x=F.relu(self.linear1(x))
        x=self.linear2(x)
        return x

    def save(self,fname='model.pth'):
        model_fpath='./model'
        if not os.path.exists(model_fpath):
            os.makedirs(model_fpath)
        fname=os.path.join(model_fpath,fname)
        torch.save(self.state_dict(),fname)

class QTrainer:

    def __init__(self, model,lr,gamma):
        self.lr=lr
        self.gamma=gamma
        self.model=model
        self.optimizer=optim.Adam(model.parameters(),lr=self.lr)
        self.criterion=nn.MSELoss()

    def train_step(self,state,action,reward,next_state,done):
        state=torch.tensor(state,dtype=torch.float)
        next_state=torch.tensor(next_state,dtype=torch.float)
        action=torch.tensor(action,dtype=torch.long)
        reward=torch.tensor(reward,dtype=torch.float)
        #(n,x)
        if len(state.shape)==1:
            #(1,x)
            state=torch.unsqueeze(state,0)
            next_state=torch.unsqueeze(next_state,0)
            action=torch.unsqueeze(action,0)
            reward=torch.unsqueeze(reward,0)
            done=(done,)
        #1-> Predicted Q Values With Current State
        pred=self.model(state)
        target=pred.clone()
        for i in range(len(done)):
            Q_new=reward[i]
            if not done[i]:
                Q_new=reward[i]+self.gamma*torch.max(self.model(next_state[i]))
            target[i][torch.argmax(action[i]).item()]=Q_new
        #2:Q_new=r+y*max(next_predicted Q Value)->Only Do This If Not Done
        #pred.clone()
        #preds[argmax(action)]=Q_new
        self.optimizer.zero_grad()
        loss=self.criterion(target, pred)
        loss.backward()
        self.optimizer.step()