# use these to save .pth files as .pkl to be converted at ubuntu for each actor-critic network layers
import torch
import pickle
import os

folder_name = "C:/Users/autocar/Desktop/TD3_new/pickle_params/"
os.mkdir(folder_name)
# actor
sd = torch.load("C:/Users/autocar/Desktop/TD3_new/model_params_td3/42_46200_actor.pth")

pickle.dump(sd.get("first.weight"), open(folder_name + "first_weight.pkl", "wb"))
pickle.dump(sd.get("first.bias"), open(folder_name + "first_bias.pkl", "wb"))
pickle.dump(sd.get("second.weight"), open(folder_name + "second_weight.pkl", "wb"))
pickle.dump(sd.get("second.bias"), open(folder_name + "second_bias.pkl", "wb"))
pickle.dump(sd.get("third.weight"), open(folder_name + "third_weight.pkl", "wb"))
pickle.dump(sd.get("third.bias"), open(folder_name + "third_bias.pkl", "wb"))
pickle.dump(sd.get("fourth.weight"), open(folder_name + "fourth_weight.pkl", "wb"))
pickle.dump(sd.get("fourth.bias"), open(folder_name + "fourth_bias.pkl", "wb"))
pickle.dump(sd.get("last.weight"), open(folder_name + "last_weight.pkl", "wb"))
pickle.dump(sd.get("last.bias"), open(folder_name + "last_bias.pkl", "wb"))