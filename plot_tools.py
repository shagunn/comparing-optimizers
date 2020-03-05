import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import matplotlib.colors as colors
import matplotlib.animation as animation
import autograd.numpy as np   # thinly wrapped version of Numpy



# Once this submodule is imported, a three-dimensional axes can be created by
# passing the keyword projection='3d' to any of the normal axes creation routines
from mpl_toolkits import mplot3d
from matplotlib.colors import LogNorm

def plot_contour(*opts, fxn_name, animate_gif=True, fig_size=None, save_f=True, f_name ='v1', in_type=None):
    def init():
        for line in lines:
            line.set_data([],[])
        return lines

    def animate(i):
        num1 = int(x2[i])
        for lnum, (line,scat) in enumerate(zip(lines, scats)):
            if num1 < dp[lnum]:
#                 print(i, dp[lnum])
                line.set_data(xlist[lnum][:num1], ylist[lnum][:num1]) # set data for each line separately.
                scat.set_offsets([xlist[lnum][num1], ylist[lnum][num1]])
                line.set_label(nn[lnum]) # set the label and draw the legend
                plt.legend(loc="upper left")
        return lines

    if fig_size == 'small':
        fig, ax = plt.subplots(figsize=(6, 4))
    else:
        fig, ax = plt.subplots(figsize=(8.5, 6.8))
    plt.tight_layout()

    data, minima, x_lims, y_lims= plot_fxn(fxn_name, 'contour')

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_xlim(x_lims)
    ax.set_ylim(y_lims)

    ax.contour(*data, levels=np.logspace(-.5, 5, 35), norm=LogNorm(), cmap='viridis', alpha=0.7) #cmap=plt.cm.jet)
    ax.plot(*minima[0:2], 'x', markersize=12, mew=2, color='k')

    xlist, ylist, zlist, nn, dp = get_dlists(opts, in_type)
    n = len(xlist)
    # print(len(xlist))

    if fxn_name == 'rosenbrock':
        if in_type == 'sdict':
            c = cm.rainbow_r(np.linspace(0,1,n))
        else:
            c = cm.jet(np.linspace(0,1,n))
    else:
        c = cm.rainbow(np.linspace(0,1,n))

    if animate_gif == False:
        for k in range(n):
            x_history = np.array(xlist[k])
            y_history = np.array(ylist[k])
            path = np.concatenate((np.expand_dims(x_history, 1), np.expand_dims(y_history, 1)), axis=1).T
            ax.quiver(path[0,:-1], path[1,:-1], path[0,1:]-path[0,:-1], path[1,1:]-path[1,:-1],
                      scale_units='xy', angles='xy', width=0.003, scale=1, color=c[k], label=nn[k])
        plt.legend(loc="upper left")
        if save_f == True:
             plt.savefig('images/{}_path.png'.format(fxn_name))
    else:
        line, = ax.plot([], [], lw=2) #, markersize=12)
        lines = []
        scats = []
        for index in range(n):
            l_obj = ax.plot([],[],lw=2,color=c[index])[0]
            lines.append(l_obj)
            s_obj = ax.scatter([],[],lw=2,color=c[index])
            scats.append(s_obj)
        num_min = int(len(xlist[0]) / 50)
#         print(num_min)
        x2 = np.rint(np.linspace(0, len(xlist[1]), endpoint = False, num = 200)) # fix this
#         print(x2)
#         x2 = np.arange(len(xlist[0]))
        # blit=True --> only re-draw the parts that have changed.
        scat = ax.scatter([], []) # Set the dot at some arbitrary position initially
        anim = animation.FuncAnimation(fig, animate, init_func=init,
                                       frames=len(x2), interval=1, blit=True)
        anim.save('images/{}_contour_{}.gif'.format(fxn_name, f_name), dpi=60, writer = "imagemagick")

    plt.show()


def plot_3D(*opts, fxn_name, animate_gif=True, f_name='v1'):
    def init():
        for line in lines:
            line.set_data([],[])
            line.set_3d_properties([])
        return lines

    def animate(i):
        num1 = int(x2[i])
        for lnum,(line,dot) in enumerate(zip(lines, dots)):
            if num1 < dp[lnum]:
                line.set_data(xlist[lnum][:num1], ylist[lnum][:num1]) # set data for each line separately.
                line.set_3d_properties(zlist[lnum][:num1])
                line.set_label(nn[lnum]) # set the label and draw the legend
                plt.legend(loc="upper center", ncol=4)
        return lines

    # Get data
    xlist, ylist, zlist, nn, dp = get_dlists(opts)
    n = len(opts)

    if fxn_name == 'rosenbrock':
        x2 = np.rint(np.linspace(0, len(xlist[1]), endpoint = False, num = 200))
        c = cm.jet(np.linspace(0,1,n))
        g = 0.4
        rot_val = 245
        nr = colors.PowerNorm(gamma=g)
    elif fxn_name == 'saddle':
        rot_val = -60
        nr = None
        x2 = np.rint(np.linspace(0, len(xlist[1]), endpoint = False, num = 100))
        c = cm.rainbow(np.linspace(0,1,n))
    else:
        x2 = np.rint(np.linspace(0, len(xlist[1]), endpoint = False, num = 200))
        c = cm.rainbow(np.linspace(0,1,n))
        g = 0.25
        rot_val = 245
        nr = colors.PowerNorm(gamma=g)

    # Plot 3D Surface
    fig = plt.figure(figsize=(9.6,6.4))
    ax = plt.axes(projection='3d', azim = rot_val)
    plt.tight_layout()

    data, minima, x_lims, y_lims = plot_fxn(fxn_name, '3D')

    ax.plot_surface(*data, rstride=1, cstride=1, norm=nr, cmap='viridis', edgecolor='none', alpha = 1.0)
    ax.plot(*minima, 'x', markersize=12, mew=2, color='k')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    # ax.set_xlim(x_lims)
    # ax.set_ylim(y_lims)
#     c = new_cmap(np.linspace(0,1,n))
#     c = cm.jet(np.linspace(0,1,n))

    if animate_gif == False:
        for k in range(n):
            x_history = np.array(xlist[k])
            y_history = np.array(ylist[k])
            path = np.concatenate((np.expand_dims(x_history, 1), np.expand_dims(y_history, 1)), axis=1).T
            ax.quiver(path[0,:-1], path[1,:-1], beale_fxn(path[::,:-1][0],path[::,:-1][1]),
                      path[0,1:]-path[0,:-1], path[1,1:]-path[1,:-1], beale_fxn(path[::,:-1][0],path[::,:-1][1])
                      - beale_fxn(path[::,:-1][0],path[::,:-1][1]),color=c[k], label=nn[k],
                      length=1, normalize=False, lw=5)
        plt.legend(loc="upper left")
    else:
        line, = ax.plot([], [], [], lw=2)
#         dot, = ax.plot([], [], 'o', lw=2)
        lines = []
        dots = []
        for index in range(n):
            l_obj = ax.plot([],[],[],lw=2,color=c[index])[0]
            lines.append(l_obj)
            d_obj = ax.scatter([],[],[], marker='o',color=c[index])
            dots.append(d_obj)

        # print(x2)
        # blit=True --> only re-draw the parts that have changed.
#         dot = ax.scatter([],[],[]) # Set the dot at some arbitrary position initially
        anim = animation.FuncAnimation(fig, animate, init_func=init,
                                       frames=len(x2), interval=1, blit=True)
        anim.save('images/{}_3D_{}.gif'.format(fxn_name, f_name), dpi=60, writer = "imagemagick")
    # ax.invert_yaxis()
    plt.show()


def plot_fxn(fxn_name, p_type=None):
    if fxn_name == "beale":
        beale_fxn = lambda x, y : (1.5 - x + x*y)**2 + (2.25 - x + x*y**2)**2 + (2.625 - x + x*y**3)**2
        if p_type == None:
            X, Y = np.meshgrid(np.linspace(-4.5, 4.5, 50), np.linspace(-4.5, 4.5, 50))
            x_lim = (-4.5, 4.5)
            y_lim = (-4.5, 4.5)
        elif p_type == "3D":
            X, Y = np.meshgrid(np.linspace(-3, 3.6, 50), np.linspace(-3, 3.6, 50))
            x_lim = (-3, 3.8)
            y_lim = (-3, 3.8)
            # Z[Z>20000]= np.nan
            # Z[y<75]= np.nan
        elif p_type == "contour":
            X, Y = np.meshgrid(np.linspace(-2.5, 4.0, 50), np.linspace(-4.5, 4.0, 50))
            x_lim = (-2.5, 4.0)
            y_lim = (-2.5, 4.0)
        Z = beale_fxn(X, Y)
        minm = np.array([[3.], [.5], [0.0]])
    elif fxn_name == "rosenbrock":
        a, b = 1, 100
        rosenbrock_fxn = lambda x, y : (a - x)**2 + b*(y - x**2)**2
        X, Y = np.meshgrid(np.linspace(-0.5, 2.0, 100), np.linspace(-1.5, 4., 100))
        Z = rosenbrock_fxn(X, Y)
        minm = np.array([[1.0], [1.0], [0.0]])
        x_lim = (-.5, 2.0) # min, max  (-.5, 2.0, 100)
        y_lim = (-1.5, 4.0)
    elif fxn_name == "saddle":
        saddle_fxn = lambda x, y : (x)**2 - (y)**2
        X, Y = np.meshgrid(np.linspace(-4.5, 4.5, 50), np.linspace(-4.5, 4.5, 50))
        Z = saddle_fxn(X, Y)
        x_lim = (-4.5, 4.5)
        y_lim = (-4.5, 4.5)
        minm = np.array([[], [], []])
    else:
        raise NameError('No valid function found.')
    return (X, Y, Z), minm, x_lim, y_lim


def get_dlists(args, in_type=None):
    x_list = []
    y_list = []
    z_list = []
    names = []
    max_val = []
    if in_type=='sdict':
        for arg in args:
            for k,v in arg.items():
                x_list.append(arg[k]['x'])
                y_list.append(arg[k]['y'])
                # z_list.append(arg[k]['z'])
                names.append(arg[k]['lr'])
                max_val.append(len(arg[k]['x']))
    else:
        for arg in args:
            x_list.append(arg['x'])
            y_list.append(arg['y'])
            z_list.append(arg['z'])
            names.append(arg['name'])
            max_val.append(len(arg['x']))
    return x_list, y_list, z_list, names, max_val


# source: https://stackoverflow.com/questions/18926031/how-to-extract-a-subset-of-a-colormap-as-a-new-colormap-in-matplotlib
# def truncate_colormap(cname, minval=0.0, maxval=1.0, n=100):
#     cmap = plt.get_cmap(cname)
#     new_cmap = colors.LinearSegmentedColormap.from_list(
#         'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
#         cmap(np.linspace(minval, maxval, n)))
#     return new_cmap
