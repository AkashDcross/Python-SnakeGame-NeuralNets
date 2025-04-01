import matplotlib.pyplot as plt
import matplotlib.animation as anim
import numpy as np

n_games=[]
counter=0
plotRangeLimiter=0

def plot(scores, mean_score):
  global counter
  global n_games


  counter +=1 
  n_games.append(counter) 
  x = n_games
  y = scores

  for i in range(1):
    plt.plot(x,y,'-ok')
    plt.pause(0.0001)

      
        