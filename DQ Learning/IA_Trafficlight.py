import numpy as np
import torch as torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
from collections import deque
import random
import math
from models import Board


class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x

    def save(self, file_name='model_name.pth'):
        model_folder_path = 'Path'
        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)

"""
class DQN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(9, 24)
        self.fc2 = nn.Linear(24, 48)
        self.fc3 = nn.Linear(48, 9)

    def forward(self, x):        
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        return x
"""
# Get state from environment, of the traffic lights
def get_state(self, game, agent):
    pos = agent.pos
    state = [
        



    
    ]
    return np.array(state, dtype=int)

class DQN_only_agent:
    def __init__(self, n_episodes=1000, n_wins_objective=195, max_env_steps=None, gamma=1.0, epsilon=1.0,
                    epsilon_min=0.01, epsilon_log_decay=0.995, alpha=0.01, alpha_decay=0.01, 
                    batch_size=64, quiet=False, saveweights=True):
        
        self.memory = deque(maxlen=100000)
        
        self.env = Board()
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_log_decay
        self.alpha = alpha
        self.alpha_decay = alpha_decay
        self.n_episodes = n_episodes
        self.n_wins_objective = n_wins_objective
        self.batch_size = batch_size
        self.quiet = quiet
        self.saveweights=saveweights
        if max_env_steps is not None: self.env._max_episode_steps = max_env_steps

        # Init model
        #self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.DQN = DQN()#.to(self.device)
        try:
            self.DQN.load_state_dict(torch.load("trained2win.pth"))
            self.DQN.eval()
        except:
            print("Not loaded")
        
        self.criterion = torch.nn.MSELoss()#.to(self.device)
        self.opt = torch.optim.Adam(self.DQN.parameters(), lr=0.01)
        self.token = "O"
#
    def get_epsilon(self, t):
        return max(self.epsilon_min, min(self.epsilon, 1.0 - math.log10((t + 1) * self.epsilon_decay)))

    def preprocess_state(self, state):
        s = [[0]*3 for i in range(3)] #############################
        for row in range(3):
            for col in range(3):
                if state[row][col] == self.token:
                    s[row][col] = 1
                elif state[row][col] == " ":
                    continue
                else:
                    s[row][col] = -1
                
        return torch.tensor(np.reshape(s, [1, 9]), dtype=torch.float32)#.to(self.device)
    
    def getMove(self, state, epsilon = 1000):
        possible_moves = []
        for cell in range(len(state[0])):
            if state[0][cell] == 0:
                possible_moves.append(cell)
        #print("Possible Moves:",possible_moves)
        if not possible_moves:
            return (-1,1)
        if (np.random.random() <= epsilon):
            #print("Random Play")
 
            n = random.choice(possible_moves)
            return (n//3,n%3) 

                
        else:
            with torch.no_grad():
                #print("DQ Play")
                prediction = torch.topk(self.DQN(state),9) ### !!!.cpu()
                preferedMoves = prediction[1].numpy()

                for n in preferedMoves[0]:
                    if int(n) in possible_moves:
                        return (n//3,n%3) if possible_moves else (-1,1)
  

    def remember(self, state, action, reward, next_state, done, e): ###
        reward = torch.tensor(reward)#.to(self.device)
        self.memory.append((state, action, reward, next_state, done, e))
    
    def replay(self, batch_size):
        y_batch, y_target_batch = [], []
        minibatch = random.sample(self.memory, min(len(self.memory), batch_size))

        for state, action, reward, next_state, done, e in minibatch: ###
            y = self.DQN(state)
            y_target = y.clone().detach()
            with torch.no_grad():
                #print("Trying:",self.DQN(next_state)[0])
                #print("Debug:",self.getMove(next_state, self.get_epsilon(e)))
                r,c = self.getMove(next_state, self.get_epsilon(e))
                ac = r*3+c
                #print("Action:",ac)
                y_target[0][action] = reward if (done or ac<0) else reward + self.gamma * self.DQN(next_state)[0][ac] ###
                
            y_batch.append(y[0])
            y_target_batch.append(y_target[0])
        
        y_batch = torch.cat(y_batch)
        y_target_batch = torch.cat(y_target_batch)
        
        self.opt.zero_grad()
        loss = self.criterion(y_batch, y_target_batch)
        loss.backward()
        self.opt.step()        
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    # Training against random agent
    def run(self):
        try:
            enemy = AI_agent(DQN,True,"trained2win.pth")
        except:
            enemy = randomAgent()#
        
        
        scores = []
        wins = []
        p1wr = []
        for e in range(self.n_episodes):
            
            done = False
            firstPlayer = np.random.choice((True,False))
            self.token = "O" if firstPlayer else "X"
            
            try:
                enemy.token = "O" if not firstPlayer else "X"  ###!!!
            except:
                pass
            
            i = 0
            
            state = self.preprocess_state(self.env.reset())
            #print("Playing as",self.token,", player", 1 if firstPlayer else 2)
            while not self.env.gameOver:

                
                if self.env.firstPlayerTurn:
                    #Your turn / Enemy turn
                    row, col = self.getMove(state, self.get_epsilon(e)) if firstPlayer else enemy.getMove(self.env)
                else:
                    #Enemy turn
                    row, col = enemy.getMove(self.env) if firstPlayer else self.getMove(state, self.get_epsilon(e))
                        

                self.env.makeMove(row, col) 
                #self.env.displayBoard()
                self.env.checkForWinner()
                
                
                
                next_state = self.env.board
                next_state = self.preprocess_state(next_state)
                
                if self.env.firstPlayerTurn == firstPlayer:
                    action =  row*3+col
                    reward = reward_function(self.env, self.token)
                    #print("Reward:",reward)
                    #print("I tried to play on:",action)
                    self.remember(state, action, reward, next_state, self.env.gameOver,e) ###
                else:
                    if self.env.gameOver:
                        reward = reward_function(self.env, self.token)
                        #print("Reward:",reward)
                        state, action, _ , next_state, done, e = self.memory[-1]
                        self.memory.pop()
                        self.memory.append((state, action, reward, next_state, done, e))
                    #print("Is first players turn:",self.env.firstPlayerTurn)
                    #print("I am first player:",firstPlayer)

                self.env.firstPlayerTurn = not self.env.firstPlayerTurn
                state = next_state
                i += 1

            scores.append(int(self.env.firstPlayerWon == firstPlayer) if not self.env.draw else 2)
            """if firstPlayer:
                played_as_p1_agent1 += 1
                p1wr.append(int(self.env.firstPlayerWon == firstPlayer) if not self.env.draw else 2)
            else:
                played_as_p1_agent2 += 1
                p1wr.append(int(self.env.firstPlayerWon == firstPlayer) if not self.env.draw else 2)"""
            wins.append(int(self.env.firstPlayerWon) if not self.env.draw else 2)
            rate = scores.count(1)/len(scores)
            if rate >= self.n_wins_objective and e >= 100:
                if not self.quiet: print('Ran {} episodes. Solved after {} trials ✔'.format(e, e - 100))
                return e - 100
            if (e+1) % (500) == 0 and e>0:
                splot(scores,size=(6,6),txt="Win rates Agent1 vs Agent2 after ["+ str(e+1) + "] iters.",
                     agent1 = "Agent 1",agent2="Agent 2")
                
                """splot(p1wr,size=(6,6),txt="Win rates as player1 after ["+ str(e+1) + "] iters.",
                     agent1 = "Agent 1",agent2="Agent 2")"""
                
                #plot(wins,txt = "P1 vs P2 after ["+ str(e+1) + "] iters.")
                
                # Save the trained model
                torch.save(self.DQN.state_dict(), "checkpoint.pth")
                
                enemy = AI_agent(DQN,True,"checkpoint.pth") #####!!!!
                scores = []
                wins = []

            self.replay(self.batch_size)
        # Save the trained model
        if self.saveweights:
            torch.save(self.DQN.state_dict(), "trained2win.pth")
        return e