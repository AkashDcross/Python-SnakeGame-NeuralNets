import pygame
import random
from collections import namedtuple
import numpy as np



pygame.init()
font = pygame.font.Font('arial.ttf', 25)


Point = namedtuple('Point', 'x, y')
#setting the colours for the game   
RED = (200,0,0)
GREEN1 = (19, 207, 69) #snake colour
GREEN2= (19, 207, 69) #snake colour
BLUEHEAD = (52, 235, 207) #snake colour
BLUE_PETEMITER = (52, 235, 207) #snake colour
WHITE = (119, 120, 122)

BLOCK_SIZE = 20 #this is the game scaler and will drasticly increase/decrease the elements of the page
SPEED = 1000


class Direction():
    
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4
    #previousMove ="right"
   


class SnakeGame:
    
    def __init__(self, w=960, h=720): ## this is the game resolution
        
        
        self.w = w 
        self.h = h
      
        # init display
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Snake Game: Reinforcement Learning and Nural Nets')
        self.clock = pygame.time.Clock()
        self.resetAI()
        
      
    def resetAI(self):
          # init game state
        self.direction = Direction.RIGHT
         
        self.head = Point(self.w/2, self.h/2)

        
        # here i am setting a list called snakeBody which will have the snake head always at the front       
        self.snakeBody = [self.head, 
                      Point(self.head.x-BLOCK_SIZE, self.head.y), # first body point
                      Point(self.head.x-(2*BLOCK_SIZE), self.head.y)] #second body point
        
        self.score = 0
        self.food = None
        self._place_food()
        self.frame_iteration=0 #frame iteration or Total moves executed
       


        
        

    def _place_food(self):
        
        #Here i am getting a random x,y position to place the food 
        x = random.randint(0, (self.w-BLOCK_SIZE)//BLOCK_SIZE )*BLOCK_SIZE # (0,(960-20)//20)*20   |Ensures that food does not spawn ouside resolution boundries
        y = random.randint(0, (self.h-BLOCK_SIZE)//BLOCK_SIZE )*BLOCK_SIZE # (0,(720-20)//20)*20    |and makes sure that the food is the correct size/preportion for snakew       
        self.food = Point(x, y) #this places new food on random points

        #if snake eats food as soon as it spawns (lucky)
        if self.food in self.snakeBody:
            self._place_food() 
        
    def play_step(self,action):

        print ("ACTION",action)
        # 1. collect user input
        self.frame_iteration +=1
     
        #print("FRAME ITERATION",self.frame_iteration)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        
      
        self._move(action) # update the head

        # since position 0 has possibilitry to be food, we are shifting the array across to make it head again
        self.snakeBody.insert(0, self.head)
        reward=0
        game_over = False

        
        # 2. check if game over
     
        ## if there is a collision or the snake does not improve (does not eat food or stuck in movement loop)
        ## the longer the snake the more time it has to decide/eat food
        if self.is_collision() or  self.frame_iteration>100*len(self.snakeBody):
            game_over = True
            reward -=10
            return reward,game_over,self.score
           
            
        # 3. place new food or just move
        if self.head == self.food:
            self.score += 1
            reward +=10
           
            self._place_food()
        else:           
            # pops so that there is not a trail longer than the body
            self.snakeBody.pop()
        
        # 4. update ui and clock
        self._update_ui()
        self.clock.tick(SPEED)
        # 5. return game over and score
        return reward, game_over, self.score #returns to agent 


        
    
    def is_collision(self,pt=None):
        if pt is None:
            pt = self.head
        # hits boundary
        if pt.x > self.w - BLOCK_SIZE or pt.x < 0 or pt.y > self.h - BLOCK_SIZE or pt.y < 0:
            return True
        # hits itself
        
        if pt in self.snakeBody[1:]:
            #print ("hit head: "+self.direction.previousMove)
            return True
        
        return False
        
    def _update_ui(self):
        self.display.fill(WHITE)
        counter =0
        for pt in self.snakeBody:
           
            if(counter==0):
                pygame.draw.rect(self.display, BLUEHEAD, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE)) #BODY
                pygame.draw.rect(self.display, BLUE_PETEMITER, pygame.Rect(pt.x+4, pt.y+4, 12, 12)) #PERIMITER
                counter +=1
            
            else:
                pygame.draw.rect(self.display, GREEN1, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE)) #BODY
                pygame.draw.rect(self.display, GREEN2, pygame.Rect(pt.x+4, pt.y+4, 12, 12)) ##PERIMITER

            
            
        pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE)) # DRAW THE PLACE FOOD
        
        text = font.render("Score: " + str(self.score), True, RED)
        self.display.blit(text, [0, 0])
        pygame.display.flip()
    
        

    def _move(self, action):              
        clock_wise_direction =[Direction.RIGHT,Direction.DOWN,Direction.LEFT,Direction.UP]
       #this ensures that the current dir is in the clockwise list 
        current_direction = clock_wise_direction.index(self.direction)
             
        if np.array_equal(action,[1,0,0]):
            new_dir = clock_wise_direction[current_direction] ##straight direction
        elif np.array_equal(action,[0,1,0]):
            next_move = (current_direction + 1)%4 
            new_dir = clock_wise_direction[next_move] #right direction
        else:
            next_move=(current_direction - 1)%4
            new_dir = clock_wise_direction[next_move] #this is basically left move
 

        self.direction = new_dir
        x = self.head.x
        y = self.head.y
              
        if self.direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif self.direction == Direction.UP:
            y -= BLOCK_SIZE         
        self.head = Point(x, y)


             

        

       