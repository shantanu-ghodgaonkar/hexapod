import numpy as np

gait_angles = np.load("C:\\Desktop\\NYU\\Capstone\\codes\\gait_angles_1stStep_20241117_181415.npy", allow_pickle=True)
new=gait_angles[:,7:]
new1=new*2048/np.pi + 2048
arr = np.array ([[2048, 1500, 1500, 2048, 2048, 2048, 2048, 1500, 1500, 2048, 2048, 2048, 2048, 1500, 1500, 2048, 2048, 2048],
                [2048, 2048, 2048, 2048, 2048, 2048, 2048, 2048, 2048, 2048, 2048, 2048, 2048, 2048, 2048, 2048, 2048, 2048],
                [2048, 2048, 2048, 2048, 1500, 1500, 2048, 2048, 2048, 2048, 1500, 1500, 2048, 2048, 2048, 2048, 1500, 1500],
                [2048, 2048, 2048, 2048, 2048, 2048, 2048, 2048, 2048, 2048, 2048, 2048, 2048, 2048, 2048, 2048, 2048, 2048]])
# print(f"Loaded gait file with {gait_angles.shape[0]} frames")
np.save("C:\\Desktop\\NYU\\Capstone\\codes\\gait_try.npy",arr)
#np.save("C:\\Desktop\\NYU\\Capstone\\codes\\gait_angles_1_converted.npy",new1)
print(arr.shape)
