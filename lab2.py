from cvxopt.solvers import qp
from cvxopt.base import matrix
from array import array
from numpy import transpose, dot, multiply, diag
from matplotlib.pylab import show, hold, plot, contour
from itertools import repeat
import numpy,pylab,random,math

def linker(x,y):
    return (dot(transpose(x), y) + 1) ** 3

classA=[(random.normalvariate(-1.5,1),
    random.normalvariate(0.5,1),
    1.0)
    for i in range(5)] + \
            [(random.normalvariate(1.5,1),
                random.normalvariate(0.5,1),
                1.0)
                for i in range(5)]
classB=[(random.normalvariate(0.0,0.5),
    random.normalvariate(-0.5,0.5),
    -1.0)
    for i in range(10)]
data=classA+classB
random.shuffle(data)

P = numpy.array([[ti*tj*linker([xi,yi],[xj,yj]) for (xj, yj, tj) in data] for (xi,yi,ti) in data])
n = len(P)
q = array('d', list(repeat(-1, n)))
h = array('d', list(repeat( 0, n)))

G = numpy.array(diag(q).tolist())

hold(True)
plot([p[0] for p in classA], [p[1] for p in classA], 'bo')
plot([p[0] for p in classB], [p[1] for p in classB], 'ro')

print(P)
print(q)
print(G)
print(h)

r=qp(matrix(P),
     matrix(q),
     matrix(G),
     matrix(h))
alpha=list(r['x'])


alpha = filter(lambda x: x >= 0,alpha)
def indicator(x,y):
    return sum(ai*ti*linker([x,y],[xi,yi]) for ai, (xi, yi, ti) in zip(alpha,data))

hold(True)
xrange=numpy.arange(-4,4,0.05)
yrange=numpy.arange(-4,4,0.05)
grid=matrix([[indicator(x,y) for y in yrange] for x in xrange])
contour(xrange,yrange,grid, (-1.0,0.0,1.0), colors=('red','black','blue'), linewidths=(1,3,1))
show()


