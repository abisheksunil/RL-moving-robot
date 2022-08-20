#!/usr/bin/env python


import matplotlib.pyplot as plt
import rospy
import numpy as np
from matplotlib.animation import FuncAnimation
from std_msgs.msg import Float64MultiArray


# First set up the figure, the axis, and the plot element we want to animate

class Visualiser:
    def __init__(self):
        self.fig = plt.figure()
        ax = plt.axes(xlim=(0, 8), ylim=(-2, 2))
        self.line, = ax.plot([], [], lw=2)
        self.x_data = [] 

# initialization function: plot the background of each frame
    def init(self):
        self.line.set_data([], [])
        return self.line,

    def odom_callback(self, msg):
        x = msg.data[0]
        self.x_data.append(x)

# animation function.  This is called sequentially
    def animate(self,i):
        y = np.sin(2 * np.pi * (np.array([self.x_data]) - 0.01 * i))
        self.line.set_data( y, self.x_data,)
        return self.line,

rospy.init_node('noise')
vis = Visualiser()
sub = rospy.Subscriber('/verifynoise', Float64MultiArray, vis.odom_callback)
# call the animator.  blit=True means only re-draw the parts that have changed.
anim = FuncAnimation(vis.fig, vis.animate, init_func=vis.init,
                               frames=200, interval=20, blit=True)
                            
plt.show(block=True)