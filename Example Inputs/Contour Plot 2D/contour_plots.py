import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

plt.rcParams["font.family"] = "Times New Roman"

# simulation properties

xSize = 2000
ySize = 2000

pixel_Resolution = 1		# units

# Read image
imgName = "cRec2000.jpg"
img = np.uint8(mpimg.imread(imgName))


df1 = pd.read_csv("def_partLabelOut.csv")
# get raw data from csv files

raw_data = df1[["x", "y", "R", "L"]].to_numpy()

R = np.zeros((ySize,xSize))

x = raw_data[:,0]
y = raw_data[:,1]

for i in range(len(raw_data[:,0])):
	R[y[i]][x[i]] = raw_data[i,2]*pixel_Resolution

# R = raw_data[:,2]
# R = np.reshape(R, [ySize, xSize])


# Create the mesh grid

Xp, Yp = np.meshgrid(np.linspace(0, 1, xSize), np.linspace(1.0*ySize/xSize, 0, ySize))

# plotting

fig1, ((ax1, ax2)) = plt.subplots(1, 2, constrained_layout=True)

fig1.set_dpi(100)
fig1.set_size_inches(8, 4)

# First axis is just the image

ax1.imshow(img)
ax1.set_title(imgName, fontsize=16)

# Second axis is R - contour

CS2 = ax2.contourf(Xp, Yp, R, 40, cmap=plt.cm.rainbow)
cbar2 = fig1.colorbar(CS2, ax=ax2, fraction=0.046, pad=0.04)
cbar2.set_label(r'Particle Radius [voxels]', rotation=90, fontsize=14)
ax2.set_title("Particle Radius Distribution", fontsize=16)
ax2.set_aspect('equal', adjustable='box')
plt.show()
