# import os
# cwd = os.getcwd()
# print(cwd)
# nas_path = "data"
# print("data:")
# print(os.listdir('/data/dan_blanaru'))
# os.chdir('/data/dan_blanaru')
# text = open("read.txt").read()
# print(text)

# f = open("write.txt",'w')
# f.write(text)
# f.close()
import torch
import monai
print(monai.__version__)
print(torch.__version__)
print(torch.cuda.is_available())
print("count",torch.cuda.device_count())
current_id = torch.cuda.current_device()
print("id", current_id)
print("name", torch.cuda.get_device_name(0))

# import subprocess
# subprocess.run(["docker search ", "monai"])