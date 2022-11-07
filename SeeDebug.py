import pickle
import numpy as np
import matplotlib.pyplot as plt

with open('Data/SaveDebugCNN.pickle', 'rb') as f:
  Debug = pickle.load(f)

# send_value = np.zeros(len(Debug))
# silence_value = np.zeros(len(Debug))

# for t in range(len(Debug)):
#   send_value[t] = Debug[t][0]
#   silence_value[t] = Debug[t][1]

# t_range = range(len(Debug))

# plt.plot(t_range, send_value, '-b', t_range, silence_value, '-r')
# plt.show()

loss_values = Debug[0]
reward_acc = Debug[1]

t_range = range(len(loss_values))
plt.plot(t_range, loss_values, '-r')
plt.show()

t_range = range(len(reward_acc))
plt.plot(t_range, reward_acc, '-r')
plt.show()