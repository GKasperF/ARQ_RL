import Envs.Environments as Envs
import numpy as np 

epsilon = 0.05
h = 1
alpha = 0.01
beta = 0.25 

Channel_Local = Envs.GilbertElliott(alpha, beta, epsilon, h)
Tf = 10

count_good_states_start = 0
count_first_erasure = 0
count_good_states_first_erasure = 0
count_past_state = 0

Total_Iterations = 1000000

for i in range(Total_Iterations):
    Channel_Local.reset()
    # for i in range(Tf-1):
    #     Channel_Local.step()

    # if Channel_Local.state == 0:
    #     count_good_states_start += 1
    Channel_Local.state = np.random.binomial(1, 0.03590229)

    past_state = Channel_Local.state
    if past_state == 0:
        count_past_state += 1

    Channel_Local.step()
    state = Channel_Local.state
    first_success = Channel_Local.step()
    if not first_success:
        count_first_erasure += 1

        if state == 0:
            count_good_states_first_erasure += 1

#Checking math:
#epsilon * (1 - alpha) * p(s_t-1 = G) / p(e_t = 1) + epsilon*beta * p(s_t-1 = B) / p(e_t = 1)
temp1 = epsilon * (1 - alpha) * count_past_state/Total_Iterations / (count_first_erasure / Total_Iterations)
temp2 = epsilon * beta * (1 - count_past_state/Total_Iterations) / (count_first_erasure / Total_Iterations)

print('Probability of good state: {}'.format(count_past_state/Total_Iterations))
print('Probability of bad state: {}'.format(1 - count_past_state/Total_Iterations))
print('Probability of erasure: {}'.format(count_first_erasure / Total_Iterations))

print('Probability of good given erasure: {}'.format(count_good_states_first_erasure/count_first_erasure)) #(epsilon*(1 - alpha)/( epsilon * (1 - alpha) + h*alpha )) for good
# (beta*epsilon/(beta*epsilon + (1-beta)*h)) for bad

print('Analytical probability of good given erasure: {}'.format(temp1 + temp2))