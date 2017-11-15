# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

x = np.linspace(-3, 3, 50)
y1 = 2*x + 1
y2 = x**2

plt.figure()
plt.plot(x, y2)
plt.plot(x, y1, color='red', linewidth=1.0, linestyle='--')
plt.xlim((-1, 2))
plt.ylim((-2, 3))

new_ticks = np.linspace(-1, 2, 5)
plt.xticks(new_ticks)
plt.yticks([-2, -1.8, -1, 1.22, 3],['$really\ bad$', '$bad$', '$normal$', '$good$', '$really\ good$'])

ax = plt.gca()
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
plt.show()

ax.xaxis.set_ticks_position('bottom')

ax.spines['bottom'].set_position(('data', 0))
plt.show()

ax.yaxis.set_ticks_position('left')

ax.spines['left'].set_position(('data',0))
plt.show()

l1, = plt.plot(x, y2, label='linear line')
l2, = plt.plot(x, y1, color='red', linewidth=1.0, linestyle='--', label='square line')

plt.legend(loc='upper right')

plt.legend(handles=[l1, l2], labels=['up', 'down'],  loc='best')


ax = plt.gca()
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.spines['top'].set_color('none')
ax.xaxis.set_ticks_position('bottom')
ax.spines['bottom'].set_position(('data', 0))
ax.yaxis.set_ticks_position('left')
ax.spines['left'].set_position(('data', 0))

x0 = 1
y0 = 2*x0 + 1
plt.plot([x0, x0,], [0, y0,], 'k--', linewidth=2.5)
# set dot styles
plt.scatter([x0, ], [y0, ], s=50, color='b')

plt.annotate(r'$2x+1=%s$' % y0, xy=(x0, y0), xycoords='data', xytext=(+30, -30),
             textcoords='offset points', fontsize=16,
             arrowprops=dict(arrowstyle='->', connectionstyle="arc3,rad=.2"))

plt.text(-3.7, 3, r'$This\ is\ the\ some\ text. \mu\ \sigma_i\ \alpha_t$',
         fontdict={'size': 16, 'color': 'r'})

x = np.linspace(-3, 3, 50)
y = 0.1*x

plt.figure()
plt.plot(x, y, linewidth=10)
plt.ylim(-2, 2)
ax = plt.gca()
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.spines['top'].set_color('none')
ax.xaxis.set_ticks_position('bottom')
ax.spines['bottom'].set_position(('data', 0))
ax.yaxis.set_ticks_position('left')
ax.spines['left'].set_position(('data', 0))

for label in ax.get_xticklabels() + ax.get_yticklabels():
    label.set_fontsize(12)
    label.set_bbox(dict(facecolor='white', edgecolor='None', alpha=0.7))
plt.show()


n = 1024    # data size
X = np.random.normal(0, 1, n) # 每一个点的X值
Y = np.random.normal(0, 1, n) # 每一个点的Y值
T = np.arctan2(Y,X) # for color value

plt.scatter(X, Y, s=75, c=T, alpha=.5)

plt.xlim(-1.5, 1.5)
plt.xticks(())  # ignore xticks
plt.ylim(-1.5, 1.5)
plt.yticks(())  # ignore yticks
plt.colorbar()
plt.show()

n = 12
X = np.arange(n)
Y1 = (1 - X / float(n)) * np.random.uniform(0.5, 1.0, n)
Y2 = (1 - X / float(n)) * np.random.uniform(0.5, 1.0, n)

#plt.bar(X, +Y1)
#plt.bar(X, -Y2)

#plt.xlim(-.5, n)
#plt.xticks(())
#plt.ylim(-1.25, 1.25)
#plt.yticks(())

#plt.show()
#
#plt.bar(X, +Y1, facecolor='#9999ff', edgecolor='white')
#plt.bar(X, -Y2, facecolor='#ff9999', edgecolor='white')

#for x, y in zip(X, Y1):
#    # ha: horizontal alignment
#    # va: vertical alignment
#    plt.text(x + 0.4, y + 0.05, '%.2f' % y, ha='center', va='bottom')
#
#for x, y in zip(X, Y2):
#    # ha: horizontal alignment
#    # va: vertical alignment
#    plt.text(x + 0.4, -y - 0.05, '%.2f' % y, ha='center', va='top')

#print('-' * 70)
#def f(x,y):
#    # the height function
#    return (1 - x / 2 + x**5 + y**3) * np.exp(-x**2 -y**2)
#
#n = 256
#x = np.linspace(-3, 3, n)
#y = np.linspace(-3, 3, n)
#X,Y = np.meshgrid(x, y)
#plt.contourf(X, Y, f(X, Y), 8, alpha=.75, cmap=plt.cm.hot)
#C = plt.contour(X, Y, f(X, Y), 8, colors='black', linewidth=.5)
#plt.clabel(C, inline=True, fontsize=10)
#plt.xticks(())
#plt.yticks(())

print('-'*70)
a = np.array([0.313660827978, 0.365348418405, 0.423733120134,
              0.365348418405, 0.439599930621, 0.525083754405,
              0.423733120134, 0.525083754405, 0.651536351379]).reshape(3,3)

plt.imshow(a, interpolation='nearest', cmap='bone', origin='lower')

plt.colorbar(shrink=.92)

plt.xticks(())
plt.yticks(())
plt.show()

fig = plt.figure()
ax = Axes3D(fig)

X = np.arange(-4, 4, 0.25)
Y = np.arange(-4, 4, 0.25)
X, Y = np.meshgrid(X, Y)    # x-y 平面的网格
R = np.sqrt(X ** 2 + Y ** 2)
# height value
Z = np.sin(R)
ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=plt.get_cmap('rainbow'))

ax.contourf(X, Y, Z, zdir='z', offset=-2, cmap=plt.get_cmap('rainbow'))

