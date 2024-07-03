from DRN import DRN
import numpy as np
import torch

point_dimension = 4
num_high_level = 2

model = DRN(input_dim = point_dimension,
            output_dim = 1,
            graph_features = num_high_level)

batch_size = 3
points_per_object = 5

x = torch.rand((batch_size * points_per_object, point_dimension)) 
gx = torch.rand((batch_size, num_high_level))
batch = torch.from_numpy(np.repeat(np.arange(batch_size), points_per_object))

print("X")
print(x)
print("GX")
print(gx)
print("BATCH")
print(batch)

output = model(x, gx, batch)
print("OUTPUT")
print(output.detach())
