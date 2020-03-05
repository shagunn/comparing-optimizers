import autograd.numpy as np   # thinly wrapped version of Numpy
from autograd import grad

beale_fxn = lambda x, y : (1.5 - x + x*y)**2 + (2.25 - x + x*y**2)**2 + (2.625 - x + x*y**3)**2
a, b = 1, 100
rosenbrock_fxn = lambda x, y : (1 - x)**2 + 100*(y - x**2)**2
saddle_fxn = lambda x, y : (x)**2 - (y)**2
r = lambda num: round(num, 6)

'''
To-Do:
=============
- Unit Tests for Stochastic Optimization: http://arxiv.org/abs/1312.6055
- Gradient checks: https://cs231n.github.io/neural-networks-3/#gradcheck
- Floating point math: http://docs.oracle.com/cd/E19957-01/806-3568/ncg_goldberg.html
- standardize naming?

'''


class Optimizer:
    def __init__(self, fxn):
        self.fxn = fxn
        self.gradient_x = grad(self.fxn, 0)
        self.gradient_y = grad(self.fxn, 1)
        self.saved_vals = {}

    def compute_gradient(self, x, y):
        dx = self.gradient_x(x, y)
        dy = self.gradient_y(x, y)
        return dx, dy

#     def compute_results(self, x_list, y_list):
#         z1 = self.fxn(np.array(x_list), np.array(y_list))
#         print(z)

    def run_optimizer(self, x, y, niter=3000, alpha=None, print_after=None, cc=True, return_var=False):
        if print_after == None:
             print_after = niter/10
        if alpha != None:
            self.alpha = alpha

        # print(x, y)

        p_val = 0.000001
        x_hist = [x]
        y_hist = [y]

        for k in range(niter):
            dx, dy = self.compute_gradient(x, y)
            x,y = self.step(x, y, dx, dy)
            x_hist.append(round(x, 10))
            y_hist.append(round(y, 10))

            if k % print_after == 0:
                print ('iteration: {}  x: {} y: {} dx: {} dy: {}'.format(k, r(x), r(y), r(dx), r(dy)))

            if cc == True:
                if abs(x_hist[k+1] - x_hist[k]) < p_val and abs(y_hist[k+1] - y_hist[k]) < p_val:
                    print ('{} reached minimum at iteration: {}'.format(self.name, k))
                    break
#             grad(i) = 0.0001
#             grad(i+1) = 0.000099989 <-- grad has changed less than 0.01% => STOP

#             (if(change_in_costFunction > precision_value))

#             repeat GD;

#         z = self.compute_results(x_hist, y_hist)
        z_hist = self.fxn(np.array(x_hist), np.array(y_hist))

        self.saved_vals = {'name':self.name,'x':np.array(x_hist), 'y':np.array(y_hist), 'z':z_hist}

        if return_var == True:
            return x_hist, y_hist


    # def plot()
    #     x_hist_beale = self.saved_vals['beale'].'x'


class SGD(Optimizer):
    def __init__(self, fxn):
        super().__init__(fxn)
        self.alpha = 0.01
        self.name = 'GD'

    def step(self, x, y, dx, dy):
        x -= self.alpha * dx
        y -= self.alpha * dy
        # print('x: ', x, ' y: ', y, 'dx: ', dx, ' dy: ', dy)
        return x,y


class SGD_momentum(Optimizer):
    def __init__(self, fxn, beta=0.5):
        super().__init__(fxn)
        self.alpha = 0.01
        self.vx = 0
        self.vy = 0
        self.beta = beta
        self.name = 'Momentum'

    def step(self, x, y, dx, dy):
        self.vx = self.beta * self.vx + self.alpha * dx
        self.vy = self.beta * self.vy + self.alpha * dy

        x -= self.vx
        y -= self.vy
        return x,y


class Nesterov(Optimizer):
    def __init__(self, fxn, beta = 0.5):
        super().__init__(fxn)
        self.alpha = 0.01
        self.vx = 0
        self.vy = 0
        self.beta = beta
        self.name = 'Nesterov'

    def step(self, x, y, dx, dy):
        vx_prev = self.vx # back this up
        self.vx = self.beta * self.vx + self.alpha * dx # velocity update stays the same
        x -= self.beta * vx_prev + (1 + self.beta) * self.vx # position update changes form

        vy_prev = self.vy # back this up
        self.vy = self.beta * self.vy + self.alpha * dy # velocity update stays the same
        y -= self.beta * vy_prev + (1 + self.beta) * self.vy # position update changes form

        return x, y


class AdaGrad(Optimizer):
    def __init__(self, fxn):
        super().__init__(fxn)
        self.alpha = 0.01 # default
        self.cache_x = 0
        self.cache_y = 0
        self.name = 'AdaGrad'
        self.epsilon = 1e-8

    def step(self, x, y, dx, dy):
        self.cache_x += dx**2
        x -=  dx * self.alpha / (np.sqrt(self.cache_x + self.epsilon))

        self.cache_y += dy**2
        y -= dy * self.alpha / (np.sqrt(self.cache_y + self.epsilon))

        return x,y


class AdaDelta(Optimizer):
    def __init__(self, fxn, rho = 0.95):
        super().__init__(fxn)
        self.alpha = 1.0 # not being used in update rule
        self.avg_grad_x = 0 # E[g^2]_0
        self.avg_grad_y = 0
        self.updateavg_x = 0 # E[deltax^2]_0
        self.updateavg_y = 0
        self.update_x = 0
        self.update_y = 0
        self.rho = rho # decay rate
        self.epsilon = 1e-8 # constant
        self.name = 'AdaDelta'


    def step(self, x, y, dx, dy):
        # https://arxiv.org/pdf/1212.5701.pdf
        # accumulate gradients
        self.avg_grad_x = self.avg_grad_x * self.rho + (1 - self.rho) * dx**2

        # compute updates
        self.update_x = ((np.sqrt(self.updateavg_x + self.epsilon)) /
                         (np.sqrt(self.avg_grad_x + self.epsilon))) * dx

        # accumulate update
        self.updateavg_x = self.updateavg_x * self.rho + (1-self.rho) * self.update_x**2

        # apply update
        x -= self.update_x

        self.avg_grad_y = self.avg_grad_y * self.rho + (1 - self.rho) * dy**2
        self.update_y = ((np.sqrt(self.updateavg_y + self.epsilon)) /
                         (np.sqrt(self.avg_grad_y + self.epsilon))) * dy
        self.updateavg_y = self.updateavg_y * self.rho + (1-self.rho) * self.update_y**2
        y -= self.update_y

        return x,y


class RMSprop(Optimizer):
    def __init__(self, fxn, epsilon = 1e-8):
        super().__init__(fxn)
        self.alpha = 0.001 # default
        self.cache_x = 0
        self.cache_y = 0
        self.name = 'RMSprop'
        self.epsilon = epsilon

    def step(self, x, y, dx, dy):
        self.cache_x = self.cache_x * self.epsilon + (1-self.epsilon) * dx**2
        x -=  dx * self.alpha / (np.sqrt(self.cache_x + self.epsilon))

        self.cache_y = self.cache_y * self.epsilon + (1-self.epsilon) * dy**2
        y -= dy * self.alpha / (np.sqrt(self.cache_y + self.epsilon))

        return x,y


class Adam(Optimizer):
    def __init__(self, fxn, beta1 = 0.9, beta2 = 0.999):
        super().__init__(fxn)
        self.alpha = 0.01
        self.m_x, self.v_x, self.mt_x, self.vt_x = 0, 0, 0, 0
        self.m_y, self.v_y, self.mt_y, self.vt_y = 0, 0, 0, 0
        self.beta1 = beta1 # beta_1
        self.beta2 = beta2 # beta_1
        self.epsilon = 1e-8 # constant
        self.t = 0
        self.name = 'Adam'


    def step(self, x, y, dx, dy):
        self.t += 1

        # Update biased first moment estimate
        self.m_x = self.m_x * self.beta1 + (1 - self.beta1) * dx
        # Update biased second raw moment estimate
        self.v_x = self.v_x * self.beta2 + (1 - self.beta2) * dx**2

        # Compute bias-corrected first moment estimate
        self.mt_x = self.m_x / (1 - self.beta1**self.t)

        # Compute bias-corrected second raw moment estimate
        self.vt_x = self.v_x / (1 - self.beta2**self.t)

        # apply update
        x -= self.mt_x * self.alpha / (np.sqrt(self.vt_x) + self.epsilon)


        self.m_y = self.m_y * self.beta1 + (1 - self.beta1) * dy
        self.v_y = self.v_y * self.beta2 + (1 - self.beta2) * dy**2
        self.mt_y = self.m_y / (1 - self.beta1**self.t)
        self.vt_y = self.v_y / (1 - self.beta2**self.t)
        y -= self.mt_y * self.alpha / (np.sqrt(self.vt_y) + self.epsilon)

        return x,y


class AdaMax(Optimizer):
    def __init__(self, fxn, beta1 = 0.9, beta2 = 0.999):
        super().__init__(fxn)
        self.alpha = 0.002 # default
        self.m_x, self.v_x = 0, 0
        self.m_y, self.v_y = 0, 0
        self.beta1 = beta1 # beta_1
        self.beta2 = beta2 # beta_1
        self.t = 0
        self.name = 'AdaMax'


    def step(self, x, y, dx, dy):
        self.t += 1

        # Update biased first moment estimate
        self.m_x = self.m_x * self.beta1 + (1 - self.beta1) * dx

        # Update the exponentially weighted infinity norm
        self.v_x = max((self.v_x * self.beta2), abs(dx))

        # apply update
        x -= (self.alpha / (1 - self.beta1**self.t)) *  (self.m_x / self.v_x)

        self.m_y = self.m_y * self.beta1 + (1 - self.beta1) * dy
        self.v_y = max((self.v_y * self.beta2), abs(dy))
        y -= (self.alpha / (1 - self.beta1**self.t)) *  (self.m_y / self.v_y)

        return x,y


class Nadam(Optimizer):
    def __init__(self, fxn, beta1 = 0.9, beta2 = 0.999):
        super().__init__(fxn)
        self.alpha = 0.01
        self.m_x, self.v_x, self.mt_x, self.vt_x = 0, 0, 0, 0
        self.m_y, self.v_y, self.mt_y, self.vt_y = 0, 0, 0, 0
        self.beta1 = beta1 # beta_1
        self.beta2 = beta2 # beta_1
        self.epsilon = 1e-8 # constant
        self.t = 0
        self.name = 'NAdam'


    def step(self, x, y, dx, dy):
        self.t += 1

        # Update biased first moment estimate
        self.m_x = self.m_x * self.beta1 + (1 - self.beta1) * dx

        # Update biased second raw moment estimate
        self.v_x = self.v_x * self.beta2 + (1 - self.beta2) * dx**2

        # Compute bias-corrected first moment estimate
        self.mt_x = self.m_x / (1 - self.beta1**self.t)

        # Compute bias-corrected second raw moment estimate
        self.vt_x = self.v_x / (1 - self.beta2**self.t)

        # apply update
        x -=  ((self.alpha / (np.sqrt(self.vt_x) + self.epsilon)) *
               (self.beta1*self.mt_x + ((1 - self.beta1)*dx)/(1 - self.beta1**self.t)))


        self.m_y = self.m_y * self.beta1 + (1 - self.beta1) * dy
        self.v_y = self.v_y * self.beta2 + (1 - self.beta2) * dy**2
        self.mt_y = self.m_y / (1 - self.beta1**self.t)
        self.vt_y = self.v_y / (1 - self.beta2**self.t)
        y -=  ((self.alpha / (np.sqrt(self.vt_y) + self.epsilon)) *
               (self.beta1*self.mt_y + ((1 - self.beta1)*dy)/(1 - self.beta1**self.t)))

        return x,y


class RAdam(Optimizer):
    def __init__(self, fxn, beta1 = 0.9, beta2 = 0.999):
        super().__init__(fxn)
        self.alpha = 0.01
        self.m_x, self.v_x, self.mt_x, self.vt_x = 0, 0, 0, 0
        self.m_y, self.v_y, self.mt_y, self.vt_y = 0, 0, 0, 0
        self.beta1 = beta1 # beta_1
        self.beta2 = beta2 # beta_1
        self.epsilon = 1e-8 # constant
        self.t = 0
        self.name = 'RAdam'
        self.sma_max = 2 / (1 - beta2) - 1


    def step(self, x, y, dx, dy):
        # https://arxiv.org/pdf/1908.03265v1.pdf
        self.t += 1

        # Update biased first moment estimate
        self.m_x = self.m_x * self.beta1 + (1 - self.beta1) * dx
        self.m_y = self.m_y * self.beta1 + (1 - self.beta1) * dy

        # Update biased second raw moment estimate
        self.v_x = self.v_x * self.beta2 + (1 - self.beta2) * dx**2
        self.v_y = self.v_y * self.beta2 + (1 - self.beta2) * dy**2

        # Compute bias-corrected first moment estimate
        self.mt_x = self.m_x / (1 - self.beta1**self.t)
        self.mt_y = self.m_y / (1 - self.beta1**self.t)

        # Compute the length of the approximated SMA
        sma = self.sma_max - (2 * self.t * self.beta2**self.t / (1 - self.beta2**self.t))
        sma = self.sma_max - (2 * self.t * self.beta2**self.t / (1 - self.beta2**self.t))



        if sma >= 5:
            # Compute bias-corrected second raw moment estimate
            self.vt_x = np.sqrt(self.v_x / (1 - self.beta2**self.t))
            self.vt_y = np.sqrt(self.v_y / (1 - self.beta2**self.t))

            # Compute the variance rectification term
            rt_x = np.sqrt(((sma - 4) * (sma - 2) * self.sma_max) /
                           ((self.sma_max - 4) * (self.sma_max - 2) * sma))
            rt_y = np.sqrt(((sma - 4) * (sma - 2) * self.sma_max) /
                           ((self.sma_max - 4) * (self.sma_max - 2) * sma))
            # apply update
            x -=  self.alpha * rt_x * self.mt_x / self.vt_x
            y -=  self.alpha * rt_y * self.mt_y / self.vt_y
        else:
            # Update parameters with un-adapted momentum
            x -=  self.alpha * self.mt_x
            y -=  self.alpha * self.mt_y
#             print("else")

        return x,y
