import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from tensorflow import keras
from tensorflow.keras import layers
import keras.backend as K
class Memory:
    def __init__(self, max_memory = 10000):
        self._max_memory = max_memory
        self._samples = []

    def add_sample(self, sample):
        self._samples.append(sample)
        if len(self._samples) > self._max_memory:
            self._samples.pop(0)

    def sample(self, no_samples):
        if no_samples > len(self._samples):
            samp_ = min(len(self._samples), self._max_memory)
            indices = np.random.choice(np.arange(np.array(self._samples).shape[0]), samp_, replace = False)
            return np.array(self._samples)[indices]
        else:
            samp_ = min(len(self._samples),no_samples)
            indices = np.random.choice(np.arange(np.array(self._samples).shape[0]), samp_, replace = False)
            return np.array(self._samples)[indices]

 
def custom_loss(y_true,y_pred):
    return K.sum(K.square(y_true-y_pred))

def return_states(time,inventory, init_inv, features, ep, index,keys):
    feats = [ep[index, keys[f]] for f in features]
    states = np.array([[time, inventory,x] + feats for x in range(inventory+1)]) 
    return states

def Q_states(states, init_inv, length): 
    #act_inv = np.array([[transform_inventory(max(init_inv,val[0]),val[0],val[1])] for val in states[:,1:3]]) #Inventory and action should be second and third column in states.
    #act_inv = act_inv.reshape((act_inv.shape[0],act_inv.shape[2]))
    #act_inv = act_inv/init_inv
    #print(states[:,1:3])
    states = np.array(states)
    return np.c_[states[:,0]/length, states[:,1:3]/init_inv, states[:,3:]]

def evaluate(action,inventory,prices,a):

    if action == 0:
        return 100*inventory*(prices[-1]-prices[0])
    M = len(prices)
    penalty  = a*(100*action/M)**2
    inv = np.linspace(inventory, inventory-action, num = M-1, endpoint= False)
    return (np.sum(100*inv*np.diff(prices) - penalty)) #We include this extra term to give the agent some notion of the level of the price without letting this information dominate the agents decisions. 


def execute(init_inv, eps, model, alpha, length, features, keys):
    #Executes orders based on our model. #Don't confuse reward here with reward above. Here we use P/L
    #print(length)
    rewards = []
    counts = {}
    for ep in eps:
        reward = 0
        inventory = init_inv
        for j in range(length):
        

            index1 = int((ep.shape[0]-1)*j/length)
            index2 = int((ep.shape[0]-1)*(j+1)/length) 

            prices = ep[index1:index2,keys["Price"]]

            states = return_states(length-j,int(inventory), init_inv, features, ep, index1,keys)
    
            states_Q = Q_states(states,init_inv,length)

            
            action = np.argmax(model.predict(states_Q))

            print(action,j, states[action])
            counts[action] = counts.get(action,0) + 1
            act = 100*action/len(prices)
            reward += np.sum(act*prices - alpha*(act)**2)
            inventory = inventory - action
        if inventory != 0:
            reward += 100*inventory*ep[ep.shape[0]-1,keys["Price"]] - alpha*(100*inventory)**2 
        rewards.append(reward)
    #print(rewards)
    print(counts)
    return np.array(rewards) 


def TWAP_paper(init_inv, eps, alpha, length, keys): 

    #Computes a TWAP strategy for the given price process. The data should given as in the paper. 
    #eps are the test episodes.
    rewards = []
    #print(eps)
    for ep in eps:
        reward = 0
        inventory = init_inv
        action = init_inv/length
        
        for j in range(length):

            index1 = int((ep.shape[0]-1)*j/length)
            index2 = int((ep.shape[0]-1)*(j+1)/length)
            #print(ep)
            prices = ep[index1:index2,keys["Price"]]
            act = 100*action/len(prices)
            reward += np.sum(act*prices - alpha*(act)**2)
            inventory = inventory - action  
        rewards.append(reward)
    #print(rewards)
    return np.array(rewards)

def bp_improve(rewards_mod, rewards_twap):
    return (rewards_mod-rewards_twap)/rewards_twap * 10**4

def GLR(rewards_mod, rewards_twap):
    #The gain loss ratio. 
    pl = bp_improve(rewards_mod, rewards_twap)
    print(pl)
    if len(pl[pl > 0]) == 0:
        numerator = 0
    else:
        numerator = np.mean(pl[pl > 0])
    denom = np.mean(-pl[pl < 0])
    if denom == 0:
        return 1
    else:
        return numerator/denom



class Agent:
    def __init__(self, model = None):

        #Can add more here like number of epochs to control more of the training procedure. 
        self.Main_Model = model
        self.Target_Model = model

    


    def default_model(self, num_inputs):

        '''
        Specify the default model architecture to be used for the agent.
        ARGS:
        num_imputs: int, the number of inputs in the model. Should be number of features plus 3.
        Returns two neural networks, Q_main and Q_target with the same parameters.
        '''
        rate = 0.01 #Learning rate.
        inputs = keras.Input(shape=(num_inputs,), name='digits')
        x = layers.BatchNormalization()(inputs)
        x = layers.Dense(250, activation='relu', name='dense_1', bias_initializer=keras.initializers.Constant(0.01), kernel_initializer=keras.initializers.TruncatedNormal(mean=0.0, stddev=0.05, seed=1))(x)
    
        x = layers.Dropout(.5)(x)
        x = layers.Dense(250, activation='relu', name='dense_2', bias_initializer=keras.initializers.Constant(0.01), kernel_initializer=keras.initializers.TruncatedNormal(mean=0.0, stddev=0.05, seed=2))(x)
        x = layers.LayerNormalization()(x)
    
        x = layers.Dropout(.5)(x)
        x = layers.Dense(250, activation='relu', name='dense_3', bias_initializer=keras.initializers.Constant(0.01), kernel_initializer=keras.initializers.TruncatedNormal(mean=0.0, stddev=0.05, seed=3))(x)
        x = layers.LayerNormalization()(x)
        x = layers.Dropout(.5)(x)
    
        x = layers.Dense(250, activation='relu', name='dense_4', bias_initializer=keras.initializers.Constant(0.01), kernel_initializer=keras.initializers.TruncatedNormal(mean=0.0, stddev=0.05, seed=4))(x)
        x = layers.LayerNormalization()(x)
        x = layers.Dropout(.5)(x)
    
        x = layers.Dense(250, activation='relu', name='dense_5', bias_initializer=keras.initializers.Constant(0.01), kernel_initializer=keras.initializers.TruncatedNormal(mean=0.0, stddev=0.05, seed=5))(x)
        x = layers.LayerNormalization()(x)
        x = layers.Dropout(.5)(x)
    
        x = layers.Dense(250, activation='relu', name='dense_6', bias_initializer=keras.initializers.Constant(0.01), kernel_initializer=keras.initializers.TruncatedNormal(mean=0.0, stddev=0.05, seed=6))(x)
        x = layers.LayerNormalization()(x)
        x = layers.Dropout(0.5)(x)
    
        x = layers.Dense(250, activation='relu', name='dense_7', bias_initializer=keras.initializers.Constant(0.01), kernel_initializer=keras.initializers.TruncatedNormal(mean=0.0, stddev=0.05, seed=7))(x)
        x = layers.LayerNormalization()(x)
        x = layers.Dropout(.5)(x)
        outputs = layers.Dense(1, name='predictions', activation = "linear")(x)
        optimizer = tf.keras.optimizers.Adam(rate)    

        Q_main = keras.Model(inputs=inputs, outputs=outputs)
        Q_target = keras.Model(inputs=inputs, outputs=outputs)
        Q_main.compile(loss = custom_loss, optimizer = optimizer)
        Q_target.compile(loss = custom_loss, optimizer = optimizer)

        self.Main_Model = Q_main
        self.Target_Model = Q_target


    def reinitialize(self, model):

        '''
        Reinitialize the agent with a new model.
        '''
        self.Main_Model = model
        self.Target_Model = model





    def pre_train(self):
        pass
    def train(self,alpha, epsilon, tau, num_samps, gamma, init_inv, rho, length, features, 
    train_eps, test_eps, keys,pre_train = False):
        '''

    Create and train a Deep RL-Agent.

    ARGS:

    alpha: float, quadratic penalty.
    epsilon: float, determines how often we select a random action (epsilon greedy).
    tau: float, controls decay rate of epsilon.
    num_samps: int, how many samples we take from the replay memory to train the neural network.
    gamma: float, discount factor.
    init_inv: int, initial inventory.
    rho: int, how many episodes to update the target network weights. 
    length: int, how many total trading periods there are.
    features: list, the features used for the model.
    train_eps: list, training_episodes.
    test_eps: list, test_episodes for monitoring training progress (validation episodes).
    pre_train: boolean, whether or not to pre_train the network. Defaults to false.  
    keys: dict, dictionary mapping features to columns for the episodes. 
    Returns:

    Q_main: Keras NN, the RL agent. 

    '''
        
        print(f"There are {len(train_eps)} episodes to train on.")

        #Initialize the replay memory.

        D = Memory()

        #Run baseline TWAP on the test_eps to get our baseline. 
     
        r_twap = TWAP_paper(init_inv, test_eps, alpha, length, keys)

        res = [] #Store our results from training performance. 


        #Get our models. 

        Q = self.Main_Model
        Q_target = self.Target_Model
        if pre_train:
            pass #To do. 

    
    #Train the model. 

        for i,ep in enumerate(train_eps):

        #Initialize inventory.

            inventory = init_inv
        
        
            print("Start", len(train_eps)-i)

            for j in range(length):
            
            #Specify the starting and ending index for this trading period. 

                start_index = int((ep.shape[0]-1)*j/length)
                end_index = int((ep.shape[0]-1)*(j+1)/length) #This is so we do not include the endpoint.

            #Get the prices.
                prices = ep[start_index:end_index,keys["Price"]]
            
            #Return the corresponding states. This is a list of the form (time,inventory,action,features)
                states = return_states(length-j,inventory, init_inv, features, ep, start_index,keys)
            
            #Selecting the action.
                if inventory <= 0:
                    action = 0 
                elif np.random.uniform() < epsilon: #Epsilon greedy selection.
                    action = np.random.randint(0, inventory + 1)
                else:
                #Returns the state vector given the desired features.
                #We normalize before putting into the network. Does this do anything? Who knows!
                    states_Q = Q_states(states,init_inv, length) 
                    print(states_Q)
                    vals = Q.predict(states_Q)
                    print(vals.flatten())
                    action = np.argmax(Q.predict(states_Q))
                    #print("Optimal", states[np.argmax(Q.predict(states_Q))])
            
            #actions.iloc[i, j] = action


            #Get the reward
                reward = evaluate(action, inventory, prices, alpha)
                print(action, reward)

            #Store our transition in a dictionary

                transition = {}
                transition["Episode"] = int(i)
                transition["Time"] = states[action][0]
                transition["Inventory"] = states[action][1]
                transition["Action"] = action
                transition["Reward"] = reward
                transition["Start_Index"] = start_index
                transition["End_Index"] = end_index
                for feat in features:
                    transition[feat] = ep[start_index,keys[feat]]
                    transition["Next_State_"+feat] = ep[end_index,keys[feat]]
                inventory = inventory - action
                transition["Next_State_Inventory"] = inventory
                transition["Next_State_Time"] = length - (j+1)

            #Add our sample to the memory.

                D.add_sample(transition)
            
            #Sample from the memory.

                mem_samp = D.sample(num_samps)

                y = [] #List of target variables. 
                X = [] #List of input variables. 
                for samp in mem_samp:
                
                #Construct X

                    X_samp = []
                    X_samp.append(samp["Time"])
                    X_samp.append(samp["Inventory"])
                    X_samp.append(samp["Action"])
                    for feat in features:
                        X_samp.append(samp[feat])
                    X.append(X_samp)

                #Construct Y

                    if samp["Next_State_Time"] == 0:
                    
                    #The amount left to sell. 
                        episode = samp["Episode"]
                        sell = max(0,samp["Next_State_Inventory"]) 

                        start_time = samp["Start_Index"]
                        end_time = samp["End_Index"]
    
    
                        r = samp["Reward"]
                        r += 100*gamma*sell*(train_eps[episode][end_time,keys["Price"]]-train_eps[episode][end_time-1,keys["Price"]])-alpha*(100*sell)**2

                        y.append(r) 
                    else:
                        next_start_index = samp["End_Index"]
                        next_inventory = samp["Next_State_Inventory"]
                        next_time = samp["Next_State_Time"]
                        episode = samp["Episode"]


                        temp_states = return_states(int(next_time), next_inventory, init_inv, features, train_eps[episode],next_start_index,keys)

                        states_Q_temp = Q_states(temp_states,init_inv, length)

                        #Get the action
                        action_ = np.argmax(Q.predict(states_Q_temp))
                        r = samp["Reward"] + gamma*Q_target.predict(states_Q_temp[action_].reshape((1,3+len(features))))[0][0]
                        y.append(r)
            
                y_true = np.array(y)
                states_Q = Q_states(X,init_inv, length)
                Q.fit(states_Q, y_true, epochs = 300, verbose = 0)
        
            epsilon = tau*epsilon
            r_mod = execute(init_inv, test_eps, Q, alpha, length, features, keys)
            res.append(np.mean(r_mod-r_twap))
            print(res[-1])
            #print(Q.get_weights())
            if (i%rho == rho-1):
                print("Updating")
                Q_target.set_weights(Q.get_weights()) 
        #print(states_data)
        plt.plot(res)
        plt.xlabel("Number of Episodes")
        plt.ylabel("Validation Error")
        self.Main_Model = Q
    def Performance_Evaluation(self,init_inv, eps, alpha, length, features, keys, model):
        '''
        Evaluates the performance of the agent on test episodes with a histogram of the
        differences in returns, basis point improvement, and gain-loss ratio.
        '''

        rewards_mod = execute(init_inv, eps, model, alpha, length, features, keys)
        rewards_twap = TWAP_paper(init_inv, eps, alpha, length, keys)

        plt.hist(rewards_mod-rewards_twap, bins = 100)

        print(f"The average difference in PnL is {np.mean(rewards_mod-rewards_twap)}.")
        bp_improvement = np.mean(bp_improve(rewards_mod, rewards_twap))
        print(f"The average basis point improvment is {bp_improvement}.")

        glr = GLR(rewards_mod, rewards_twap)
        print(f"The gain loss ratio is {glr}.")




