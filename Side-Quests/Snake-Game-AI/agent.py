'''The Agent Is The AI That Will Play The Game'''
#Importing Necessary Modules And Libraries
import torch
import random
import numpy as np
from collections import deque
from game import GameAI,Direction,Point
from model import Linear_QNet,QTrainer
from helper import plot

#Assigning Model Constraints
MAX_MEMORY=100_000
BATCH_SIZE=1000
LR=0.001

class Agent:

    def __init__(self):
        self.n_games=0
        self.epsilon=0 #Randomness
        self.gamma=0.9 #Discount Rate
        self.memory=deque(maxlen=MAX_MEMORY) #pop->left()
        self.model=Linear_QNet(11,256,3)
        self.trainer=QTrainer(self.model,lr=LR,gamma=self.gamma)

    def get_state(self,game):
        head=game.snake[0]
        l=Point(head.x-20,head.y)
        r=Point(head.x+20,head.y)
        u=Point(head.x,head.y-20)
        d=Point(head.x,head.y+20)
        dir_l=game.direction==Direction.LEFT
        dir_r=game.direction==Direction.RIGHT
        dir_u=game.direction==Direction.UP
        dir_d=game.direction==Direction.DOWN
        state=[
            #Danger Straight
            (dir_r and game.is_collision(r)) or 
            (dir_l and game.is_collision(l)) or 
            (dir_u and game.is_collision(u)) or 
            (dir_d and game.is_collision(d)),
            #Danger Right
            (dir_u and game.is_collision(r)) or 
            (dir_d and game.is_collision(l)) or 
            (dir_l and game.is_collision(u)) or 
            (dir_r and game.is_collision(d)),
            #Danger Left
            (dir_d and game.is_collision(r)) or 
            (dir_u and game.is_collision(l)) or 
            (dir_r and game.is_collision(u)) or 
            (dir_l and game.is_collision(d)),
            #Move Direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            #Reward Location 
            game.food.x<game.head.x, #Left
            game.food.x>game.head.x, #Right
            game.food.y<game.head.y, #Up
            game.food.y>game.head.y  #Down
            ]
        return np.array(state,dtype=int)

    def remember(self,state,action,reward,next_state,done):
        self.memory.append((state,action,reward,next_state,done)) #Pop->Left If MAX_MEMORY Is Reached

    def train_long_memory(self):
        if len(self.memory)>BATCH_SIZE:
            mini_sample=random.sample(self.memory,BATCH_SIZE) #List Of Tuples
        else:
            mini_sample=self.memory
        states,actions,rewards,next_states,dones=zip(*mini_sample)
        self.trainer.train_step(states, actions,rewards,next_states,dones)
        #For state,action,reward next_state,done In mini_sample:
        #self.trainer.train_step(state,action,reward,next_state,done)

    def train_short_memory(self,state,action,reward,next_state,done):
        self.trainer.train_step(state,action,reward,next_state,done)

    def get_action(self,state):
        #Random Moves: Tradeoff Exploration/Exploitation
        self.epsilon=80-self.n_games
        final_move=[0,0,0]
        if random.randint(0, 200)<self.epsilon:
            move=random.randint(0,2)
            final_move[move]=1
        else:
            state0=torch.tensor(state,dtype=torch.float)
            prediction=self.model(state0)
            move=torch.argmax(prediction).item()
            final_move[move]=1
        return final_move


def train():
    plot_scores=[]
    plot_mean_scores=[]
    total_score=0
    record=0
    agent=Agent()
    game=GameAI()
    while True:
        #Getting Old State
        state_old=agent.get_state(game)
        #Getting Move
        final_move=agent.get_action(state_old)
        #Performing A Move And Getting New State
        reward,done,score=game.play_step(final_move)
        state_new=agent.get_state(game)
        #Training On Short Memory
        agent.train_short_memory(state_old,final_move,reward,state_new,done)
        #Remembering..
        agent.remember(state_old,final_move,reward,state_new,done)
        if done:
            #Training On Long Memory,plotting result
            game.reset()
            agent.n_games+=1
            agent.train_long_memory()
            if score>record:
                record=score
                agent.model.save()
            print('Game',agent.n_games,'Score',score,'Record:',record)
            plot_scores.append(score)
            total_score+=score
            mean_score=total_score/agent.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores,plot_mean_scores)

if __name__=='__main__':
    train()