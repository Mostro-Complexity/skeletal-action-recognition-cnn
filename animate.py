import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d.art3d import Line3D, Line3DCollection
import h5py
import json

f = h5py.File('data/MSRAction3D/features.mat', 'r')
features = np.array([f[element] for element in np.squeeze(f['features'][:])])
data = features[0]

f = json.load(open('data/MSRAction3D/body_model.json', 'r'))
bones = np.array(f['bones']) - 1


fig = plt.figure()
ax = p3.Axes3D(fig, azim=-90, elev=10)

# Setting the axes properties
ax.set_xlim3d([0, 4])
ax.set_xlabel('X')

ax.set_ylim3d([0, 4])
ax.set_ylabel('Z')

ax.set_zlim3d([0, 4])
ax.set_zlabel('Y')

graph = ax.scatter([], [], [])
frame = data[0]+2

lines = [np.array([frame[bones[i, 0]], frame[bones[i, 1]]])
         for i in range(bones.shape[0])]

lc = Line3DCollection(segments=lines)
ax.add_collection3d(lc)


# def update_lines()

def animate(i):
    frame = data[i]+2
    graph._offsets3d = (frame[:, 0], frame[:, 1], frame[:, 2])

    lines = [np.array([frame[bones[i, 0]], frame[bones[i, 1]]])
             for i in range(bones.shape[0])]
    lc.set_segments(lines)
    return graph


ani = FuncAnimation(fig, animate, frames=data.shape[0], interval=100)
plt.show()
