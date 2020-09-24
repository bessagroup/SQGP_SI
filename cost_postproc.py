from scipy.interpolate import griddata
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pprint
from matplotlib.ticker import FuncFormatter

import itertools


def process(frame, xdata, ydata, func, verification = False, params = None , plot_on = True, printing_on = True,
           uncertainty_func = None, uncertainty_params = None ,p0 = None, bounds = (0., np.inf)):
    #E.g.
    #xdata = [frame['k'].values, frame['tslc'].values, frame['n'].values]
    #ydata = frame[subroutine].values
    if not verification:
        popt, pcov = curve_fit(func, xdata, ydata, bounds=bounds, p0 = p0)

        perr = np.sqrt(np.diag(pcov))
        if printing_on:
            print('Params: ', popt)
            print('Interpolation covariance:', pcov)
            print('Error 1std:',  perr)
        plot_rows = 2
    else:
        popt = params
        perr = None
        plot_rows = 3
    x, y = xdata[:2]
    z = ydata #frame['QPE'].values
    grid_x, grid_y = np.mgrid[min(x):max(x):200j, min(y):max(y):200j]
    points = np.array(xdata[:2]).T
    grid_z = griddata(points, z, (grid_x, grid_y), method='linear')
    
    comma_fmt = FuncFormatter(lambda x, p: format(int(x), ','))
    
    if plot_on:
        
        fig = plt.figure(figsize = (8,int(4*plot_rows) ))
        fig.add_subplot(plot_rows, 2, 1)
        ax = plt.gca()
        ax.scatter(x, y, marker = '+', c= 100-z, s = 50, cmap='gray')
        #ax.set_xlabel(xlabel, size = 18)
        #ax.set_ylabel(ylabel, size = 18)
        a = (max(xdata[0])-min(xdata[0]))/(max(xdata[1])-min(xdata[1]))       
        im = ax.imshow(grid_z.T, cmap = 'jet', extent=(min(xdata[0]), max(xdata[0]),min(xdata[1]), max(xdata[1]))
                       ,aspect = 'auto', 
                       origin='lower')
        plt.xlabel('k')
        plt.ylabel('r')
        plt.title('Interpolated experimental data', fontsize = 14, pad = 10)

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        plt.colorbar(im, cax=cax,  format=comma_fmt)


        fig.add_subplot( plot_rows, 2, 2)
        ax1 = plt.gca()

        interpolated = func([grid_x, grid_y, frame['n'].values[0]], *popt )
        im1 = ax1.imshow(interpolated.T ,cmap = 'jet',  origin='lower', 
                  extent=(min(xdata[0]), max(xdata[0]),min(xdata[1]), max(xdata[1])),aspect = 'auto')
        plt.grid(True)
        plt.xlabel('k')
        plt.ylabel('r')
        plt.title('Model prediction', fontsize = 14, pad = 10)
        divider = make_axes_locatable(ax1)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        plt.colorbar(im1, cax=cax,  format=comma_fmt)
        #plt.tight_layout()

        
        fig.add_subplot(plot_rows,2,  3)
        ax2 = plt.gca()

        interpolated = func([grid_x, grid_y, frame['n'].values[0]], *popt )
        im2 = ax2.imshow(100.*(interpolated.T -grid_z.T)/grid_z.T  ,cmap = 'jet',  origin='lower', 
                   extent=(min(xdata[0]), max(xdata[0]),min(xdata[1]), max(xdata[1])),aspect = 'auto')
        plt.grid(True)
        plt.xlabel('k')
        plt.ylabel('r')
        plt.title('Relative error [%]', fontsize = 14, pad = 10)
        divider = make_axes_locatable(ax2)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        plt.colorbar(im2, cax=cax, format = '%d%%')
        #plt.tight_layout()

        fig.add_subplot(plot_rows, 2, 4)
        ax3 = plt.gca()
        
        interpolated = func([grid_x, grid_y, frame['n'].values[0]], *popt )
        im3  = ax3.imshow((interpolated.T -grid_z.T)  ,cmap = 'jet',  origin='lower', 
                   extent=(min(xdata[0]), max(xdata[0]),min(xdata[1]), max(xdata[1])),aspect = 'auto')
        
        plt.grid(True)
        plt.xlabel('k')
        plt.ylabel('r')
        plt.title('Absolute error', fontsize = 14, pad = 10)
        divider = make_axes_locatable(ax3)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        plt.colorbar(im3, cax=cax, format=comma_fmt)
        #plt.tight_layout()
    
        
        if verification:
            fig.add_subplot(plot_rows,2,  5)
            ax4 = plt.gca()
            interpolated = uncertainty_func([grid_x, grid_y, frame['n'].values[0]], *popt, uncertainty_params )
            im4  = ax4.imshow(100.*(interpolated.T/grid_z.T)  ,cmap = 'jet',  origin='lower', 
                       extent=(min(xdata[0]), max(xdata[0]),min(xdata[1]), max(xdata[1])),aspect = 'auto')
            plt.grid(True)
            plt.xlabel('k')
            plt.ylabel('r')
            plt.title('Relative uncertainty[%]', fontsize = 14, pad = 10)
            divider = make_axes_locatable(ax4)
            cax = divider.append_axes("right", size="5%", pad=0.1)
            plt.colorbar(im4, cax=cax, format = '%2.f%%')
            
            
            fig.add_subplot(plot_rows,2,  6)
            ax5 = plt.gca()
            interpolated = uncertainty_func([grid_x, grid_y, frame['n'].values[0]], *popt, uncertainty_params )
            im5  = ax5.imshow(interpolated.T  ,cmap = 'jet',  origin='lower', 
                       extent=(min(xdata[0]), max(xdata[0]),min(xdata[1]), max(xdata[1])),aspect = 'auto')
            plt.grid(True)
            plt.xlabel('k')
            plt.ylabel('r')
            plt.title('Absolute uncertainty', fontsize = 14, pad = 10)
            divider = make_axes_locatable(ax5)
            cax = divider.append_axes("right", size="5%", pad=0.1)
            plt.colorbar(im5, cax=cax, format=comma_fmt)
   
            
        print('Number of data-points:',len(frame['n'].values) )
        fig.tight_layout()
        fig.suptitle('Circuit depth model, n=' + str(frame['n'].values[0]), y = 1.03, fontsize = 15)
        

    return popt, perr
    
