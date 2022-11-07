import torch
file_indices = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12']
for file_index in file_indices:

    with open('Data/TraceSets/distance_10m/raw_data_run'+file_index+'.txt') as f:
        lines = f.readlines()

    #configs = [ line.rstrip('\n').split(',')[0:8] for line in lines]
    new_lines = [line.rstrip('\n').split(',') for line in lines]

    success_info = torch.zeros((len(new_lines), 300))

    for i in range(len(new_lines)):
        for j in range(300):
            packet_info = new_lines[i][(j+1) * 9 : (j + 2) * 9]
            packet_info = [float(value) for value in packet_info]
            if packet_info[3] == 1 and packet_info[4] <= 1:
                success_info[i,j] = 1
            else:
                success_info[i,j] = 0


    with open('Data/TraceSets/distance_10m/Run'+file_index+'_10m.torch', 'wb') as f:
        torch.save(success_info, f)

pass