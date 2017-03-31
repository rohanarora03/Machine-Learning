from numpy import loadtxt, zeros, ones, array
from pylab import scatter, show, title, xlabel, ylabel, plot


#Evaluate the linear regression

#Cost Function
def costfunction(X, y, theta):				
    
#Number of training samples    
    m = y.size						

    predictions = X.dot(theta).flatten()		

    sqErrors = (predictions - y) ** 2

    J = (1.0 / (2 * m)) * sqErrors.sum()

    return J

# Gradient Descent to minimize cost function J(theta)
def gradient_descent(X, y, theta, alpha, num_iters):	
    
    m = y.size

    J_history = zeros(shape=(num_iters, 1))

    for i in range(num_iters):

        predictions = X.dot(theta).flatten()

        errors_x1 = (predictions - y) * X[:, 0]
        errors_x2 = (predictions - y) * X[:, 1]

        theta[0][0] = theta[0][0] - alpha * (1.0 / m) * errors_x1.sum()
        theta[1][0] = theta[1][0] - alpha * (1.0 / m) * errors_x2.sum()

        J_history[i, 0] = costfunction(X, y, theta)

    return theta, J_history

data = loadtxt('data1.txt', delimiter=',')			#Load the dataset
scatter(data[:, 0], data[:, 1], marker='o', c='b')		#Plot the data
title('Profits distribution')
xlabel('Population of City in 10,000s')
ylabel('Profit in $10,000s')
X = data[:, 0]
y = data[:, 1]

m = y.size
it = ones(shape=(m, 2))						# Add a column of 1s to x - Interception Data  
it[:, 1] = X

theta = zeros(shape=(2, 1))					# Initialize parameters 
iterations = 1500
alpha = 0.01

#gradient descent to calculate theta minimum.
theta, J_history = gradient_descent(it, y, theta, alpha, iterations)  		
print J_history

#Predict values for population.
predict1 = array([1, 4.0]).dot(theta).flatten()
print 'For population = 40,000, we predict a profit of %f' % (predict1 * 10000)
predict2 = array([1, 8.0]).dot(theta).flatten()
print 'For population = 80,000, we predict a profit of %f' % (predict2 * 10000)

#Plot the results
result = it.dot(theta).flatten()
plot(data[:, 0], result)
show()



