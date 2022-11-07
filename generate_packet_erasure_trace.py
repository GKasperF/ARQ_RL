import torch 

with open('Data/TraceSets/Delays_Failures.pth', 'rb') as f:
    delays = torch.load(f)


ones_vector = torch.ones(len(delays))
zeros_vector = torch.zeros(len(delays))

packet_success = torch.where(delays < 0.25, ones_vector, zeros_vector)

with open('Data/TraceSets/TraceUFSC_Failures.pth', 'wb') as f:
    torch.save(packet_success, f)


pass