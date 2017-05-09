import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time

#compute mean of sq error
def compute_error(b,m, points):
    totalError = 0
    for i in range(0,len(points)):
        x = points[i,0]
        y = points[i,1]
        totalError += (y - (m*x + b))**2
    return totalError/ float(len(points))


def step_gradient(b_curr, m_curr, points, learningRate):
    #Gradient Descent
    b_grad = 0
    m_grad = 0
    N = float(len(points))
    for i in range (0, len(points)):
        x = points[i,0]
        y = points[i,1]
        b_grad += (-2/N)*(y - (m_curr*x + b_curr))
        m_grad += (-2/N)*x*(y - (m_curr*x + b_curr))
    new_b = b_curr - (learningRate)*b_grad
    new_m = m_curr - (learningRate)*m_grad
    plt.show()
    return [new_b, new_m]
    
        
def gradient_descent_run(points, start_b, start_m, learning_rate, num_iter):
    b = start_b
    m = start_m

    plt.ion()
    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    for i in range(num_iter):
        b, m = step_gradient(b,m, np.array(points), learning_rate)
        y = m*points[:,0] + b
        #draw and update figure in runtime
        fig.suptitle('Current iteration: {0}  m = {1}  b = {2}'.format(i,m,b))
        ax1.scatter(b,m,color = 'green')
        ax1.set_xlim(0.01, 0.04)
        ax1.set_ylim(0, 5)
        ax1.set_title('Gradient Search')
        ax1.set_xlabel('line y-intercept (b)')
        ax1.set_ylabel('line slope (m)')
        ax2.scatter(points[:,0],points[:,1])
        ax2.plot(points[:,0],y, 'r-')
        ax2.set_ylim(0, 125)
        ax2.set_title('Points and Regression line')
        fig.canvas.draw()
        time.sleep(0.25)
        ax2.clear()
        
    return [b,m]

    
def run():
    points = np.genfromtxt('data.csv', delimiter = ',')
    #hyper-parameter
    learning_rate = 0.00001 
    #y = mx + b 
    initial_b = 0
    initial_m = 0
    num_iter = 100
    print "Starting gradient descent at b = {0}, m = {1}, error = {2}".format(initial_b, initial_m, compute_error(initial_b, initial_m, points))
    [b, m] = gradient_descent_run(points, initial_b, initial_m, learning_rate, num_iter)
    print "After {0} iterations b = {1}, m = {2}, error = {3}".format(num_iter, b, m, compute_error(b, m, points))

    
if __name__ == '__main__':
    run()
