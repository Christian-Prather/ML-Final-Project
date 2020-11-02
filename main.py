import pandas as pd
import os
import matplotlib.pyplot as plt


data_path =  os.getcwd()
def load_csv_data():
    csv_path = os.path.join(data_path,'captures', "data" + str(0)+".csv")
    df = pd.read_csv(csv_path)
    df.columns = ["X", "Y", "Z"]
    return df

world = load_csv_data()
world = world.loc[(world!=0).any(1)]

print (world)
print(world.info())
print(world.describe())

world.hist(bins=50, figsize=(20,15))
plt.show()
