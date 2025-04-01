import torch
import random
import numpy as np
from collections import deque
from snake_game import SnakeGame, Direction, Point
from model import Linear_QNet, QTrainer
from helper import plot


MAX_MEMORY = 100_000
BATCH_SIZE = 3000
LEARING_RATE = 0.001

class Agent:
    def __init__(self):
        self.n_games=0
        self.epsilon=0 #randomness
        self.gamma=0.9 #discount rate
        self.memory = deque(maxlen=MAX_MEMORY) #popleft() when memo length to big
        self.model=Linear_QNet(11,512,3)
        self.trainer=QTrainer(self.model,lr=LEARING_RATE,gamma=self.gamma)
  

    def get_state(self, snake_game):
        
        head = snake_game.snakeBody[0]
        point_left= Point(head.x-20,head.y)
        point_right= Point(head.x+20,head.y) 
        point_up= Point(head.x,head.y-20) 
        point_down= Point(head.x,head.y+20) 

        direction_left = snake_game.direction == Direction.LEFT
        direction_right = snake_game.direction == Direction.RIGHT
        direction_up = snake_game.direction == Direction.UP
        direction_down = snake_game.direction == Direction.DOWN

        state = [

            # these wioll be played to the console as 0000-000-0000
            # first 4 digets detect if there is collision 
            # the 3 digets determine the direction the snake is going  (000 = current direction) (001 = right turn) 
            #danger straight #clockwise
            (direction_right and snake_game.is_collision(point_right)) or
            (direction_left and snake_game.is_collision(point_left)) or
            (direction_up and snake_game.is_collision(point_up)) or 
            (direction_down and snake_game.is_collision(point_down)),

              #danger right #clockwise
            (direction_up and snake_game.is_collision(point_right)) or
            (direction_down and snake_game.is_collision(point_left)) or
            (direction_left and snake_game.is_collision(point_up)) or 
            (direction_right and snake_game.is_collision(point_down)),
            
            #danger left  #anti clickwise movement
            (direction_down and snake_game.is_collision(point_right)) or
            (direction_up and snake_game.is_collision(point_left)) or
            (direction_right and snake_game.is_collision(point_up)) or 
            (direction_left and snake_game.is_collision(point_down)),


            #Move Direction

            direction_left,
            direction_right,
            direction_up,
            direction_down,

            #foood location
            snake_game.food.x < snake_game.head.x, #food left
            snake_game.food.x > snake_game.head.x, #food right
            snake_game.food.y < snake_game.head.y, #food up
            snake_game.food.x > snake_game.head.x #food down
            
            ]
        
        return np.array(state,dtype=int)

        


    def remember(self,state,action, reward,next_move,done):
        
        ## this is used so that the memory can store all the states from the current game iteration into memory
        self.memory.append((state,action, reward,next_move,done)) #


    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            ##memory is too large
            mini_sample = random.sample(self.memory,BATCH_SIZE)
        else:
            ## here we are simply  copying all the data from memory  into mini_sample
            mini_sample=self.memory
        
        ## this line below makes it so that all the data from the mini sample are places into
        ## the correct varialbes. For example all the state variables from the mini_sample  will be stored in (states)
        ## and the same applied to all the other variables in the mini_sample
        states, actions, rewards, next_moves, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_moves, dones)




    # this function trains only 1 game step (trains a single move)
    def train_short_memory(self,state,action, reward,next_move,done):    
       # print ("state",state) 
        #print ("action",action)  
        #print ("reward",reward) 
        #print ("new move",next_move)      


        self.trainer.train_step(state, action, reward, next_move, done)
        


    def get_action(self,state):
        print ("-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-")
        #random moves: tradeoff exploration/exploitation       
        ##this function if condition is for exploring and exploiting the enviroment
        #self.epsilon=30-self.n_games
        self.epsilon=80-self.n_games
        final_move=[0,0,0]
        if random.randint(0,200)<self.epsilon:          
            move = random.randint(0,2) ##[0,0,0] picking a random index 
 
            final_move[move]=1   #stores the final move based on the random index
            print("explore:",final_move)
            

        #this else condition is used so the snake can make predictive moves
        else:
            state0 = torch.tensor(state,dtype=torch.float) #turning the old state into floats
        
            prediction = self.model(state0) #putting predicted move into the nural net
            print ("prediction",prediction)
            move = torch.argmax(prediction).item()
            print ("Selected Move",move)
            final_move[move]=1
            #print("final move:",final_move)
        return final_move



def train():
    

    plot_scores =[] ##records all scores to show a progression chart
    plot_mean_scores =[] ##records average scores for display in progression chart
    total_score=0
    record=0 #record
    agent = Agent()        
    
    snake_game = SnakeGame()
    
    while True:
        #this gets the current move of the game 
        state_old = agent.get_state(snake_game)
      
        #get move
    
        # this will attempt to calculate the new predicted move based of the old move
        final_move = agent.get_action(state_old)
        
       

        #perform move and get new state
        reward,done,score = snake_game.play_step(final_move)


        state_new = agent.get_state(snake_game)
        print("New State",state_new)

        #print("old state = ",state_old) #old move
        #print("final state = ",final_move) #move
        #print("new state = ",state_new) #all new state values based on final_move()
        #print("new reward = ",reward)
            
        #train short memory
        agent.train_short_memory(state_old, final_move, reward, state_new, done)
        agent.remember(state_old, final_move, reward, state_new, done)
            
        if done:
            #trains on all the previous moves and games played
            #helps improve
            snake_game.resetAI()
            agent.n_games +=1                             
            agent.train_long_memory()

            if score > record:
                record = score
                agent.model.save()
        
            plot_scores.append(score)
            total_score +=score
            mean_score = total_score / agent.n_games
            print('Game',agent.n_games,'Score',score,"Record:",record,"AVERAGE",mean_score)
            plot_mean_scores.append(mean_score)
            plot(plot_scores,plot_mean_scores)
            
                   
if __name__=='__main__':
    train()


