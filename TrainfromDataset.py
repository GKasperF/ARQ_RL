import pickle
import ReinforcementLearning.QlearningFunctions as QL
import torch
import copy 
import Envs.PytorchEnvironments as EnvsNN

with open('Data/DatasetGilbertElliott.pickle', 'rb') as f:
    dataset = pickle.load(f)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


states = dataset[0]
actions = dataset[1]
next_states = dataset[3]
discount_factor = 0.95
batch_size = 10000

alpha_range = torch.arange(0.1, 5.5, 0.1)
for alpha_tmp in alpha_range:
    alpha = alpha_tmp.to(device)
    rewards = dataset[2] - alpha


    Channel = EnvsNN.GilbertElliott(0.25, 0.25, 0, 1, batch_size).to(device)
    env = EnvsNN.EnvFeedbackGeneral(10, alpha, 5, Channel, batch_size, M=5).to(device)
    Q = QL.QApproxFunction(env.observation_space.n, env.action_space.n, 1000).to(device)
    Qtarget = copy.deepcopy(Q)
    criterion = torch.nn.MSELoss(reduction='mean')
    optimizer = torch.optim.Adam(list(Q.parameters()), lr = 0.0001)

    epochs = 20
    num_updates = 5
    finish_state = env.finish_state[0]

    string_alpha = str(alpha_tmp.numpy())
    model_file = 'ModelCNNFromDataset_Memory'+string_alpha+'.pickle'



    for i in range(epochs):
        indices = torch.randperm(len(states))
        states = states[indices]
        actions = actions[indices]
        rewards = rewards[indices]
        next_states = next_states[indices]

        for j in range(int(len(states)/batch_size)):
            states_training = states[j*batch_size : (j+1)*batch_size]
            actions_training = actions[j*batch_size : (j+1)*batch_size]
            rewards_training = rewards[j*batch_size : (j+1)*batch_size]
            next_states_training = next_states[j*batch_size : (j+1)*batch_size]
            Next_States_QValues = Qtarget(next_states_training)
            finish_states_indices = torch.all(torch.eq(next_states_training, finish_state), dim = 1)
            finish_states_indices = finish_states_indices.reshape( len(finish_states_indices), 1).repeat(1,env.action_space.n)
            Next_States_QValues = torch.where(finish_states_indices, torch.zeros(Next_States_QValues.size()).to(device) , Next_States_QValues)
            BestTargetValues, _ = torch.max(Next_States_QValues, dim = 1, keepdim = True)
            td_target = rewards_training.reshape((len(rewards_training), 1)) + discount_factor*BestTargetValues
            
            Qestimates = Q(states_training)
            td_estimate = Qestimates[torch.arange(len(states_training)), actions_training.type(torch.int64).reshape(len(actions_training))].reshape( (len(states_training), 1))
            loss = criterion(td_estimate, td_target.detach())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if j % num_updates == 0:
                Qtarget = copy.deepcopy(Q)

        with open('Data/'+model_file, 'wb') as f:
            pickle.dump(Q, f)