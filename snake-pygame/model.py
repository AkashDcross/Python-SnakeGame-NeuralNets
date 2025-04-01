import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size,output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size,hidden_size) 
        self.linear2= nn.Linear(hidden_size,output_size) #[112312,12321,12312]
   
      
   

    #THIS FUNCTION IS THE PREDICTION*****
    def forward(self, x):
       # print ("this is X",x)
        x =F.relu(self.linear1(x))
        x =(self.linear2(x)) 

        #print( "x",x)
        return x


    def save(self, file_name='model.pth'):
        pass
 


class QTrainer:
    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, done):
        #print ("-----------------TRAIN STEP-----------------")
        state = torch.tensor(state, dtype=torch.float)
      #  print ("State",state)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)

        # (n, x)
        #print("-------squeezed---------",state)

        #print("state length",len(state.shape))

        if len(state.shape) == 1: #bias
            # (1, x)
            state = torch.unsqueeze(state, 0)
           # print("-----unsqueeeeze-------",state)
            next_state = torch.unsqueeze(next_state, 0)
          #  print("-----next_state unsqueeeeze-------",state)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done, )
         

        # 1: predicted Q values with current state
        
        pred = self.model(state) 
        #print(pred)
    
       

        target = pred.clone()
        #print (target)

        for idx in range(len(done)):
            Q_new = reward[idx] 
      
            if not done[idx]:
              
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))

               # print("next step",self.model(next_state[idx]))
               # print(reward[idx],"+",self.gamma,"*",max(self.model(next_state[idx])))
               # print(Q_new)




            #here we are updating the target and the actions variables with the new Q values
        
            target[idx][torch.argmax(action[idx]).item()] = Q_new   
            #print("old",target)
            #print(torch.argmax(action[idx]).item)
            #print("new",action)
            #print([torch.argmax(action[idx])])
            #print("new",target[idx])
            
           
    
       
        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
     
        loss.backward()
        self.optimizer.step()


