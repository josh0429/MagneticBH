import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import time

def RK4(x,l,b,f,h):
    k1 = h*f(x,l,b)
    k2 = h*f(x + k1/2,l,b)
    k3 = h*f(x + k2/2,l,b)
    k4 = h*f(x + k3,l,b)
    return x + k1/6 + k2/3 + k3/3 + k4/6
    
def f(x,l,b):
    #x = [rho,drho,theta,dtheta]
    drho = x[1]
    ddrho = (1./2)*(2*x[0]-3)*(x[3]**2) + (2*l*b-1)/(2*x[0]**2) + \
            (l**2)*(2*x[0]-3)/(2 * np.sin(x[2])**2 * x[0]**4) - \
            (b**2/2)*(2*x[0] - 1)*np.sin(x[2])**2
    dtheta = x[3]
    ddtheta = - (2/x[0]) * x[1] * x[3] + (l**2 * np.cos(x[2]))/(x[0]**4 * np.sin(x[2])**3) - \
            b**2 * np.sin(x[2]) * np.cos(x[2])
    return np.array([drho,ddrho,dtheta,ddtheta])
    
def Ueff(x,l,b):
    return (1 - 1/x[0])*(1 + ((l - b * x[0]**2 * np.sin(x[2])**2)**2)/(x[0]**2 * np.sin(x[2])**2))
    
def E0(rho,l,b):
    return (1 - 1/rho)*(1 + ((l - b * rho**2)**2)/(rho**2))
    
def check(x,f,l,b,h,maxt,error):    
    t = 0
    while t <= maxt:
        xnew1 = RK4(x,l,b,f,h)
        xmid = RK4(x,l,b,f,h/2)
        xnew2 = RK4(xmid,l,b,f,h/2)
        if np.linalg.norm(xnew1-xnew2) > error:
            h = h/3
        else:
            x = xnew2
            t += h
            h = 2*h
            if abs(x[0]) <= 1: #absorb
                return 2
            z = x[0]*np.cos(x[2])
            if z >= 1000: #zinf
                return 1
            if z <= -1000: #-zinf
                return 3
    return 4
            
                

#create colormap in which (0,white,energetically forbidden)
# (1,red,+zinf)
# (2,green,absorb)
# (3,blue,-zinf)
# (4,black,unknown)

cmap = colors.ListedColormap(['white','red','green','blue','black'])
bounds = [-0.5,0.5,1.5,2.5,3.5,4.5]
norm = colors.BoundaryNorm(bounds, cmap.N)

pixels = input('How many pixels? ')
h = 10E-3
error = 10E-6
maxt = 1E5
data = []
start = time.clock()
for E in np.linspace(1.15,0.95,pixels):
    print((1.15-E)/0.2*100,'% done')
    row = []
    for rho in np.linspace(2.8,3,pixels):
        if rho == 1:
            row.append(2)
        else:
            if rho == 3:
                l = np.sqrt(3)
                b = 0
            else:
                l = rho * np.sqrt(3 * rho - 1)/np.sqrt(8 * rho**2 - 18*rho + 6 + \
                        2*np.sqrt((3 * rho - 1)*(3 - rho)))
                b = np.sqrt(6 - 2*rho)/(2 * rho) / np.sqrt(4 * rho**2 - 9 * rho + 3 + \
                        np.sqrt((3 * rho - 1)*(3 - rho)))
            if E**2 - E0(rho,l,b) < 0:
                row.append(0)
            else:
                x = np.array([rho,0,np.pi/2,-np.sqrt((E**2 - E0(rho,l,b))/(rho*(rho-1)))])
                row.append(check(x,f,l,b,h,maxt,error))
    data.append(row)
    
plt.imshow(data,cmap=cmap,norm=norm,interpolation='none')
plt.savefig('positive_fractal_{}.png'.format(pixels))

output = open('positive_greatdata_{}.txt'.format(pixels), 'w')
for row in data:
    string = ''
    for column in row:
        string = string + '{} '.format(column)
    output.write(string + '\n')
output.close()

print(time.clock()-start,' seconds')