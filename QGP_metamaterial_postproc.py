import numpy as np
import copy
import os
import tensorflow as tf
from SALib.sample import saltelli
from SALib.analyze import sobol
from timeit import default_timer as timer
from scipy import interpolate
from scipy import stats
import pandas as pd
import seaborn as sns
from scipy.integrate import simps, trapz
import h5py
from matplotlib.colors import LogNorm



# Matplotlib
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib
from matplotlib import rc
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import matplotlib as mpl
from scipy import stats

# Preprocessing data:
from sklearn.preprocessing import StandardScaler
#import gpflow 
import logging
logging.basicConfig(format='%(asctime)s %(message)s')

# Postprocessing metrics:
from sklearn.metrics import mean_squared_error, r2_score , explained_variance_score, classification_report
import GPy
from sklearn.preprocessing import StandardScaler
from scipy.interpolate import griddata
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pprint
from matplotlib.ticker import FuncFormatter

# To save the model:
import os
import errno
import scipy.io as sio
try:
    import cPickle as pickle  # Improve speed
except ImportError:
    import pickle

from os import listdir
from os.path import isfile, join



def interpolate_grid(data_dict, xvar, yvar, zvar, xbounds=None, ybounds= None, title = None, xlabel = None, 
                         ylabel = None, zlabel = None):
        
        if xlabel is None:
            xlabel = '$\\frac' + xvar + '$'
        if ylabel is None:
            ylabel = '$\\frac' + yvar + '$'           
        if zlabel is None:
            zlabel = '$' + zvar + '$'            
            
        x = data_dict[xvar]
        y = data_dict[yvar]
        z  = data_dict[zvar] 
        if xbounds is not None:    
            idx_x = np.where(np.logical_and(x>=xbounds[0], x <=xbounds[1]))         
            x = x[idx_x]
            y = y[idx_x]
            z = z[idx_x]
        if ybounds is not None:
            idx_y = np.where(np.logical_and( y>=ybounds[0], y <= ybounds[1]))
            x = x[idx_y]
            y = y[idx_y]
            z = z[idx_y]
   
        grid_x, grid_y = np.mgrid[min(x):max(x):200j, min(y):max(y):200j]
        points = np.array([x, y]).T
        grid_z = griddata(points, z, (grid_x, grid_y), method='linear')
        
        ax = plt.gca()
        ax.scatter(x, y, marker = '+', c= 100-z, s = 50, cmap='gray')
        if title is not None:
            ax.set_title(title, size = 20)

        ax.set_xlabel(xlabel, size = 18)
        ax.set_ylabel(ylabel, size = 18)
        a = (max(x)-min(x))/(max(y)-min(y))       
        im = ax.imshow(grid_z.T, cmap = 'jet', extent=(min(x), max(x),min(y), max(y)),aspect = 'auto',  origin='lower')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        plt.colorbar(im, cax=cax, label= zlabel)
        
        plt.tight_layout()
        
def plot_GP_2d(X, Xnew, m, x, y,  idx_class=None , xlab = 'x1', ylab = 'x2', savepath = None, 
              suptitle = None, split_plots = False):
    
    mean, Cov = m.predict(Xnew, full_cov=False)
    
    print('GP prediction with noise')
    # Setup plot environment
    
    mean_mat = mean.reshape(x.shape)
    mean_mat[idx_class] = np.nan
    
    if split_plots:
        f = plt.figure(figsize=(6, 6))
        plt.contourf(x, y,mean_mat  , cmap = 'inferno', label = "Mean")
        plt.plot(X[:,0],X[:,1],'.', markersize = 8, label = 'training points')
        # Annotate plot
        plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
        plt.ylabel(ylab, size = 18)
        plt.xlabel(xlab, size =18)
        plt.legend(prop={'size': 14})
        plt.title("Mean " + suptitle, size = 20)
        plt.colorbar()
        plt.tight_layout()
        
        if savepath is not None:
            f.savefig(savepath + '1.png', dpi = 400)
            f.savefig(savepath + '1.svg')
            
        f2 = plt.figure(figsize=(6, 6))  
        cov_mat  =Cov.reshape(x.shape)
        cov_mat[idx_class] = np.nan 

        # Plot variance surface
        plt.contourf(x, y, cov_mat**0.5,  rasterized = True)

        # Show sample locations
        plt.plot(X[:,0],X[:,1],'.',  markersize = 8, label = 'training points'), #plt.axis("square")
        
        # Annotate plot
        plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
        plt.ylabel(ylab, size = 20)
        plt.xlabel(xlab, size = 20)
        plt.legend(prop={'size': 14})
        plt.title("Uncertainty ($\sigma$) " + suptitle, size = 20)
        plt.colorbar()

        plt.tight_layout()
        if savepath is not None:
            f2.savefig(savepath + '2.png', dpi = 400)
            f2.savefig(savepath + '2.svg')
        
        
    else:
        f= plt.figure(figsize=(12, 6))

    # Left plot shows mean of GP fit
        plt.subplot(1,2,1)
        plt.contourf(x, y,mean_mat  , cmap = 'inferno')
        plt.plot(X[:,0],X[:,1],'.', markersize = 8, label = 'training points')

        # Annotate plot
        plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
        plt.ylabel(ylab, size = 18)
        plt.xlabel(xlab, size =18)
        plt.legend(prop={'size': 14})
        plt.title("Mean", size = 20)
        plt.colorbar()

        # Right plot shows the variance of the GP
        cov_mat  =Cov.reshape(x.shape)
        cov_mat[idx_class] = np.nan
        plt.subplot(1,2,2)    
        # Plot variance surface
        plt.contourf(x, y, cov_mat**0.5)

        # Show sample locations
        plt.plot(X[:,0],X[:,1],'.',  markersize = 8, label = 'training points'), #plt.axis("square")
        
        # Annotate plot
        plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
        plt.ylabel(ylab, size = 20)
        plt.xlabel(xlab, size = 20)
        plt.legend(prop={'size': 14})
        plt.title("Uncertainty ($\sigma$)", size = 20)
        plt.colorbar()#label = 'STD: ' +title)

        plt.tight_layout()
        plt.subplots_adjust(top=0.85)#, left = 0.05, right  = 0.98)
        if suptitle is not None:
                 # Add space at top
            f.suptitle(suptitle, y= 0.98, size = 22)
        
    
    

    if savepath is not None:
        print('Saving figure!')
        f.savefig(savepath)
    # Preview GP model parameters
    print(m)


def plot_GP_classification_2d(X, Xnew, m, x, y, xlab = 'x1', ylab = 'x2', savepath = None, title = "GP Classification"):
    
    f= plt.figure(figsize=(7, 6))
    mean, Cov = m.predict(Xnew, full_cov=False)
    binary_class =  np.round(mean.reshape(x.shape))
    print('GP prediction with noise')
    
    # Left plot shows mean of GP fit
    plt.subplot(111)
    img = plt.contourf(x, y, binary_class, cmap = 'inferno', levels = [-0.1,0.5, 1.1])

    cbar = plt.colorbar(img, ticks = [0.25, 0.75])
    cbar.ax.set_yticklabels(['non- \ncoilable',  'coilable'])
    plt.plot(X[:,0],X[:,1],'.', label = 'training points')
    # Annotate plot
    if title is not None:
        plt.title(title, size = 22), #plt.colorbar()
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    plt.ylabel(ylab, size = 20)
    plt.xlabel(xlab, size  = 20)
    plt.legend(prop={'size': 18})
    idx_not_classified = np.where(binary_class < 0.5)
    idx_classified = np.where(binary_class > 0.5)
    
    if savepath is not None:
        print('Saving figure!')
        f.savefig(savepath, bbox_inches = 'tight')        

    return idx_not_classified, idx_classified

class Unit_cell():
    
    def __init__(self, data, Length, D1):
        self._coilable = None
        self._maxStrain = None
        
        self._stress_values_trimmed = None
        self._strain_values_trimmed = None
        self._max_location_stress = None
        self._max_location_strain = None
        
        self._vector = None
        self._curve = None
        self._energy_abs = None
        self._buckling_load = None
        
        
        self._coilable = data['coilable'][0] 
        self._maxStrain = Length/Length
        temp_U = data['riks_RP_Zplus_U'][:-2,2] 
        temp_F = data['riks_RP_Zplus_RF'][:-2,2] 
        stress_values = 1000*(temp_F/(np.pi*0.25*(D1)**2))
        strain_values = temp_U/Length   
        vec_indices = np.where(stress_values <= 0)

        self._stress_values_trimmed = abs(stress_values[vec_indices])
        self._strain_values_trimmed = abs(strain_values[vec_indices])
        self._max_location_stress = np.argmax(self._stress_values_trimmed)
        self._max_location_strain = np.argmax(self._strain_values_trimmed)
        #if np.max(strain_values_trimmed[max_location_strain])> np.min([0.8,maxStrain]):                        

        if self._strain_values_trimmed[-1]<1.0:       
            self._strain_values_trimmed = np.append(self._strain_values_trimmed,1.0)
            self._stress_values_trimmed = np.append(self._stress_values_trimmed,0.0)
        vector = np.linspace(np.min(self._strain_values_trimmed),np.max(self._maxStrain),10000)
        interp = interpolate.pchip(self._strain_values_trimmed,self._stress_values_trimmed)                                                             
        curve = interp(vector)
        self._vector = vector[:-1]
        self._curve = curve[:-1]

        self._energy_abs = simps(self._curve,x=self._vector)
        self._buckling_load = data['P_p3_crit'][0]/(np.pi*0.25*(D1)**2)*1000 


    def plot_response(self):
        plt.plot(vector, curve)
        plt.xlabel('strain')
        plt.ylabel('stress')
        plt.grid(True)
        plt.show()


class Analyze():
        
    def __init__(self, dir_path):
        self._dir_path = dir_path
        self._DoE_data = None
        self._Input_data = None
        self._STRUCTURES_data = None

        
        self._Input_points  = None
        self._Input_points_all  = None

        self._Input_var_names = None
        
        self._Lost_DoEs = None
        
    
    #replace those 3 with a superclass
    def load_DoE(self, DoE_file, DoE_dir = '1_DoEs/'):
        file_DoE_path = self._dir_path+'/'+DoE_dir+DoE_file
        self._DoE_data = sio.loadmat(file_DoE_path)
        self._Input_points_all = self.get_doe_points()
        self._Input_points = self._Input_points_all
        self._Input_var_names = self.get_doe_vars()
        
        
    def load_input(self, Input_file, Input_dir = '2_Inputs/'):
        file_Input_path = self._dir_path+'/'+Input_dir + Input_file
        self._Input_data = sio.loadmat(file_Input_path)
        
        
        
    def load_postproc(self, analysis_folder, postproc_dir = '4_Postprocessing/'):    
        #postproc_dir = '4_Postprocessing/'
        file_postproc_path = self._dir_path+'/'+postproc_dir+analysis_folder+'/'+'STRUCTURES_postprocessing_variables.p'
        with open(file_postproc_path, 'rb') as pickle_file:
            try:
                self._STRUCTURES_data = pickle.load(pickle_file, encoding='latin1') #for python 3
            except Exception as e:
                self._STRUCTURES_data = pickle.load(pickle_file)      

                
                
    def quick_load_data(self, analysis_folder):
        
        #loading data when DoEs file, Inputs file and postproc analysis
        #folders have the same name        
        self.load_input(analysis_folder)
        self.load_DoE(analysis_folder)
        self.load_postproc(analysis_folder)
        
    def get_doe_points(self):
        return self._DoE_data['DoE']['points'][0][0]
    
    def get_doe_vars(self):
        # Extract the names of the design variables
        feature_names = [] 
        for iFeature in range (0,len(self._DoE_data['DoE']['vars'][0][0][0])):                 
            feature_names.append(str(self._DoE_data['DoE']['vars'][0][0][0][iFeature][0]))
        return feature_names
        

    def derive_results(self, filtering = True):
        
        D1 = 100.0  #Dimeter of the unit cell base [mm]
        
        #other parameters: x indices from original 7 dim notation:
        
        A_D1 = 1.e-3    #X[0]
        G_E = 3.6e-1    #X[1]
        #I_x =          #running variable
        Iy = 7.5e-7
        J_D1 = 1.e-6 
        P_D = 0.66      #X[5]
        ConeSlope  = 0. #X[6]
        Length = P_D*D1

        input_dim = 1 #len(self._Input_var_names)  -1                   #number of design parematers (input space dimension)
        imperfection_dim = len(self._STRUCTURES_data["Input1"])         #number of imperfection parameters
        doe_dim = len(self._STRUCTURES_data['Input1']['Imperfection1']) #number of DoE points/imprfection/input
       
        self._coilable = [[[]]]    #np.zeros((input_dim, imperfection_dim, doe_dim))
        self._max_strain = [[[]]]  #np.zeros((input_dim, imperfection_dim, doe_dim))
        self._energy_abs = [[[]]]  #np.zeros((input_dim, imperfection_dim, doe_dim))
        self._max_strain = [[[]]]  #np.zeros((input_dim, imperfection_dim, doe_dim))
        self._buckling_load= [[[]]] 
        self._failed_samples = []
        inputs_good_samples = [[[]]]
    
        #RETREIVE COILABLE DESIGNS SATISFYING COMPRESSIBILITY >80%
        for iInput in range(input_dim):
            #count = 0
            for kImperf in range(0,imperfection_dim):
                    for jDoE in range(0,doe_dim):
                        data = self._STRUCTURES_data['Input1']['Imperfection'+str(kImperf+1)]
                        try:
                            if 'DoE'+str(jDoE+1) in data:
                                data = self._STRUCTURES_data['Input'+str(iInput+1)]['Imperfection'+str(kImperf+1)]['DoE'+str(jDoE+1)]        
                                
                                mm_sample = Unit_cell(data, Length, D1)

                                self._coilable[iInput][kImperf].append(mm_sample._coilable)
                                self._max_strain[iInput][kImperf].append(mm_sample._maxStrain)
                                self._energy_abs[iInput][kImperf].append(mm_sample._energy_abs)
                                self._buckling_load[iInput][kImperf].append(mm_sample._buckling_load)
                                inputs_good_samples[iInput][kImperf].append(self._Input_points[jDoE])

                            else:
                                self._failed_samples.append([iInput,kImperf, jDoE ]) 

                        except:
                            self._failed_samples.append([iInput,kImperf, jDoE ])                          
                            print('Failed Sample (ikj):', iInput, kImperf, jDoE)
                        
                    if kImperf - 1< imperfection_dim:
                        self._coilable[iInput].append([])
                        self._max_strain[iInput].append([])
                        self._energy_abs[iInput].append([])
                        self._max_strain[iInput].append([])
                        inputs_good_samples[iInput].append([])
            
            if iInput - 1< input_dim:
                self._coilable.append([])
                self._max_strain.append([])
                self._energy_abs.append([])
                self._max_strain.append([])
                inputs_good_samples.append([])
        
        if filtering:
            self._Input_points = inputs_good_samples
            print('Non-coilable designs have been filtered out! ')

            
    def data_to_dict(self, iInput = 1, iImperfection = 1):
        data_dict = {}
        for i, name in enumerate(self._Input_var_names):
            data_dict[name[6:-1]] = np.array(self._Input_points[0][0])[:, i]
        
        data_dict['E_abs'] = np.array(self._energy_abs[iInput-1][iImperfection-1])
        data_dict['P_crit'] = np.array(self._buckling_load[iInput-1][iImperfection-1])
        return data_dict
    
def combined_qgp_plot(qgp_mean, points_mean, qgp_std, points_std, X1, X2, 
                        idx_not_classified, title = '$P_{crit}$',
                      savepath = None, Xtr = None, split_plots = False):
    
    xbounds = [0, max(X1), min(X2), max(X2)]
    xd, yd = np.mgrid[xbounds[0]:xbounds[1]:100j, xbounds[2]:xbounds[3]:100j]
    grid_z = griddata(points_mean ,qgp_mean , (xd, yd) , method='linear')
    grid_z[idx_not_classified[1],idx_not_classified[0]] = np.nan
    
    
    xlabel = '$\\frac{I_{xx}}{D^4_1}$'
    ylabel = '$\\frac{P}{D_1}$'
    
    if split_plots:
        fig = plt.figure(figsize = (6, 6))
        img = plt.contourf(xd, yd, grid_z, cmap = 'inferno')
        plt.colorbar(img)
        plt.ticklabel_format(style='sci', axis='x', scilimits=(-1,0))
        plt.xlabel(xlabel, size = 18)
        plt.ylabel(ylabel, size = 18)
        plt.title('Mean '+ title, size = 20)

        if Xtr is not None:        
            plt.plot(Xtr[:,0],Xtr[:,1],'.', label = 'training points')
            plt.legend()
        plt.tight_layout()
        
        if savepath is not None:
            fig.savefig(savepath + '1.png', dpi = 400)
            fig.savefig(savepath + '1.svg')
            
        grid_z_std = griddata(points_std ,qgp_std , (xd, yd) , method='linear')
        grid_z_std[idx_not_classified[1],idx_not_classified[0]] = np.nan

        fig2 = plt.figure(figsize = (6, 6))
        
        img2 = plt.contourf(xd, yd, grid_z_std, cmap = 'viridis')
        plt.colorbar(img2)
        plt.ticklabel_format(style='sci', axis='x', scilimits=(-1,0))
        plt.xlabel(xlabel, size = 18)
        plt.ylabel(ylabel, size = 18)

        if Xtr is not None:        
            plt.plot(Xtr[:,0],Xtr[:,1],'.', label = 'training points')
            plt.legend()


        plt.title('Uncertainty ($\sigma$) '+ title, size = 20)
        plt.tight_layout()
        
        if savepath is not None:
            fig2.savefig(savepath + '2.png', dpi = 400)
            fig2.savefig(savepath + '2.svg')     
    
    else:
        fig = plt.figure(figsize = (12, 6))
        plt.subplot(1, 2, 1)
        img = plt.contourf(xd, yd, grid_z, cmap = 'inferno')
        plt.colorbar(img)
        plt.ticklabel_format(style='sci', axis='x', scilimits=(-1,0))
        plt.xlabel(xlabel, size = 18)
        plt.ylabel(ylabel, size = 18)
        plt.title('Mean', size = 20)

        if Xtr is not None:        
            plt.plot(Xtr[:,0],Xtr[:,1],'.', label = 'training points')
            plt.legend()

        plt.subplot(1, 2, 2)
        grid_z_std = griddata(points_std ,qgp_std , (xd, yd) , method='linear')
        grid_z_std[idx_not_classified[1],idx_not_classified[0]] = np.nan

        img2 = plt.contourf(xd, yd, grid_z_std, cmap = 'viridis')
        plt.colorbar(img2)
        plt.ticklabel_format(style='sci', axis='x', scilimits=(-1,0))
        plt.xlabel(xlabel, size = 18)
        plt.ylabel(ylabel, size = 18)

        if Xtr is not None:        
            plt.plot(Xtr[:,0],Xtr[:,1],'.', label = 'training points')
            plt.legend()


        plt.title('Uncertainty ($\sigma$)', size = 20)
        plt.tight_layout()

        plt.subplots_adjust(top=0.85)     # Add space at top
        plt.suptitle('QGP model: '+title,  size = 22, y = 0.98)

        if savepath is not None:
            print('Saving figure!')
            fig.savefig(savepath)

        
class postproc_QGP_2d():
    
    def __init__(self, experiment_name, Xnew_test, var_data = False, m = None, Xnew_test_scaled = None):
        self._Res_dicts = []
        self._Input_dict = []
        self._idx_list = []
        
        self._QGP_predict = []
        self._QGP_error = []
        self._xqgp = []
        self._classical_res = []
        
        if var_data:
            res_path = os.getcwd() + "/serial_experiment_2d/" +experiment_name + "/results-var/"
        else:
            res_path = os.getcwd() + "/serial_experiment_2d/" +experiment_name + "/results/"
       
        #print(res_path)
        

        for file in listdir(res_path):
            self._idx_list.append(int(file[6:-7]))    #index corresponding to sample number, retrieved from the name e.g. sample26.pickle
            f = open(os.path.join(res_path, file), 'rb')
            self._Res_dicts.append(pickle.load(f))
            f.close()


        #classical_res = []
        
        #VARIANCE 
        if var_data:
            flag = False
            for i in range(len(self._Res_dicts)):
                xs  = Xnew_test_scaled[self._idx_list[i]]
                var = m.kern.K(xs[None, :], xs[None, :])
                utemp = np.real(max([(var - self._Res_dicts[i]['qgp_result_stv']), [1e-15]])).flatten()
                self._QGP_predict.append(utemp**0.5)
                self._QGP_error.append(self._Res_dicts[i]['error%_stv'])
                self._xqgp.append(Xnew_test[self._idx_list[i]])
                self._classical_res.append((var - self._Res_dicts[i]['classical_result'])**.5)
                
                if var - self._Res_dicts[i]['qgp_result_stv']<0:
                    flag = True
            if flag:
                print('NEGATIVE VARIANCE ENOUNTERED, replaced with 0')
        #MEAN
        else:
            for i in range(len(self._Res_dicts)):
                self._QGP_predict.append(self._Res_dicts[i]['qgp_result_stv'])
                self._QGP_error.append(self._Res_dicts[i]['error%_stv'])
                self._xqgp.append(Xnew_test[self._idx_list[i]])
                self._classical_res.append(self._Res_dicts[i]['classical_result'])

        self._xqgp = np.array(self._xqgp)
        self._QGP_predict = np.array(self._QGP_predict).flatten()
        self._QGP_error = np.array(self._QGP_error).flatten()
        self._classical_res = np.array(self._classical_res).flatten()

        
    def plot_countour(self, xbounds = None, CMAP = 'inferno',  xlabel = '$\\frac{I_{xx}}{D_1^4}$',
                            suptitle = 'test-1', ylabel = '$\\frac{P}{D_1}$', zlabel = '$P_{crit}$', 
                      vmin = 5, vmax = 50, idx_class = None, 
                     savepath = None, plot_query_points = False, levels = None):

        grid_x, grid_y = np.mgrid[xbounds[0]:xbounds[1]:100j,
                                  xbounds[2]:xbounds[3]:100j]
      
        points = self._xqgp 
        z = self._QGP_predict
        
        grid_z = griddata(points, z, (grid_x, grid_y), method='linear')
    
        grid_z[idx_class] = np.nan
        #grid_mean = copy.copy(grid_z)
        
        fig = plt.figure(figsize = (12,6 ))
        plt.subplot(1, 2, 1)
        plt.ticklabel_format(style='sci', axis='x', scilimits=(-1,0))
        plt.xlabel(xlabel, size = 18)
        plt.ylabel(ylabel, size = 18)        
        img0 = plt.contourf(grid_x, grid_y, grid_z, cmap = CMAP, 
                                                 extent=(min(self._xqgp[:, 0]), max(self._xqgp[:, 0]),
                                                    min(self._xqgp[:, 1]), max(self._xqgp[:, 1])), 
                                                     vmin = vmin, vmax = vmax,
                                                    origin='lower' , levels = levels)
        cbar = plt.colorbar(img0, ticks = levels)
        if zlabel is not None:
            cbar.set_label(zlabel,size=18)
        plt.tight_layout()
        
        if plot_query_points:
            plt.scatter(self._xqgp[:, 0], self._xqgp[:, 1], marker = '+', c= 100-z, s = 50, cmap='gray', 
                        label = 'QGP evaluation points')
            plt.legend()
        plt.title('QGP results', size = 20)



        plt.subplot(1, 2, 2)
        z = self._classical_res
        grid_z = griddata(points, z, (grid_x, grid_y), method='linear')
        grid_z[idx_class] = np.nan

        plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
        plt.xlabel(xlabel, size = 18)
        plt.ylabel(ylabel, size = 18)
        img = plt.contourf(grid_x, grid_y, grid_z, cmap = CMAP, extent=(min(self._xqgp[:, 0]), max(self._xqgp[:, 0]),
                                                           min(self._xqgp[:, 1]), max(self._xqgp[:, 1])),
                                                   vmin = vmin, vmax = vmax,
                                                    origin='lower', levels = levels)

        cbar2 = plt.colorbar(img, ticks = levels)#, label= zlabel)
        if zlabel is not None:
            cbar2.set_label(zlabel,size=18)
        
        if plot_query_points:
            plt.scatter(self._xqgp[:, 0], self._xqgp[:, 1], marker = '+', c= 100-z, s = 50, cmap='gray', 
                        label = 'QGP evaluation points')
            plt.legend()
        

        plt.title('Classical results', size = 20)
        plt.tight_layout()
        plt.subplots_adjust(left = 0.02, right = 0.98, top=0.85)     
        fig.suptitle(suptitle, size = 22, y= 0.98)

        if savepath is not None:
            print('Saving figure!')
            fig.savefig(savepath)
            

        

        
        
    def plot_error_countour(self, xbounds = None, CMAP = 'inferno', xlabel = '$\\frac{I_{xx}}{D_1^4}$',
                            suptitle = 'test-1', 
                            ylabel = '$\\frac{P}{D_1}$', zlabel = '$Error[\%]$', 
                            zlabel2 = 'Aboslute error $\epsilon$' ,
                            vmin = 5, vmax = 50, idx_class = None, 
                            savepath = None, levels = None, plot_query_points = False, variance_case = False):


        grid_x, grid_y = np.mgrid[xbounds[0]:xbounds[1]:100j,
                                  xbounds[2]:xbounds[3]:100j]
      
        points = self._xqgp 
        
        if not variance_case:
            z = copy.copy(self._QGP_error)
        else:
            z = np.abs((self._QGP_predict-self._classical_res) /self._classical_res)*100. #for uncertainty
        z[np.where(z>100)] = 100
        #print(z)
        #print('error capped to 100%')
        
        grid_z = griddata(points, z, (grid_x, grid_y), method='linear')
        

        grid_z[idx_class] = np.nan
        
        fig = plt.figure(figsize = (12,6 ))
        plt.subplot(1, 2, 1)
        plt.ticklabel_format(style='sci', axis='x', scilimits=(-1,0))
        plt.xlabel(xlabel, size = 18)
        plt.ylabel(ylabel, size = 18)
        if levels is None:
            lvls = np.logspace(-1, 2, 13)
        else:
            lvls = levels
        
        
        grid_z[np.where(grid_z>100.)] = 99.99999
        grid_z[np.where(grid_z<np.min(lvls))] = np.min(lvls)
        
        img0 = plt.contourf(grid_x, grid_y, grid_z, cmap = CMAP, norm = LogNorm(), levels = lvls)
        cbar = plt.colorbar(img0, format='%.1f%%')
        cbar.set_label(zlabel,size=18)
        plt.tight_layout()
        
        if plot_query_points:
            plt.scatter(self._xqgp[:, 0], self._xqgp[:, 1], marker = '+', c= 100-z, s = 50, cmap='gray', 
                        label = 'QGP evaluation points')
            plt.legend()
        plt.title('Relative error', size = 20)
        
        
        #PLOT2:ABSOLUTE ERROR
        grid_x, grid_y = np.mgrid[xbounds[0]:xbounds[1]:100j,
                                  xbounds[2]:xbounds[3]:100j]
        points = self._xqgp 
        z = np.abs(self._QGP_predict-self._classical_res) 
        grid_z = griddata(points, z, (grid_x, grid_y), method='linear')
        grid_z[idx_class] = np.nan
        
        plt.subplot(1, 2, 2)
        plt.ticklabel_format(style='sci', axis='x', scilimits=(-1,0))
        plt.xlabel(xlabel, size = 18)
        plt.ylabel(ylabel, size = 18)
        img1 = plt.contourf(grid_x, grid_y, grid_z, cmap = CMAP)
        cbar = plt.colorbar(img1)
    

        cbar.set_label(zlabel2,size=18, labelpad= 10)
        if plot_query_points:
            plt.scatter(self._xqgp[:, 0], self._xqgp[:, 1], marker = '+', c= 100-z, s = 50, cmap='gray', 
                        label = 'QGP evaluation points')
            plt.legend()
            
        plt.title('Absolute error', size = 20)
        
        plt.tight_layout()        
        plt.subplots_adjust(top=0.85)    

        fig.suptitle(suptitle, size = 22, y= 0.98)

        if savepath is not None:
            print('Saving figure!')
            fig.savefig(savepath)

        
        
    def plot_verification_countour(self, input_dictionary, xbounds = None, CMAP = 'inferno', xlabel = '$\\frac{I_{xx}}{D_1^4$', suptitle = 'test-1', 
                      ylabel = '$\\frac{P}{D_1}$', zlabel = '$Error[\%]$',savepath1 = None, savepath2 = None, 
                                  levels = None, idx_class = None, bar_format = '%.1f', plot_query_points = False, 
                                  variance_case = False):

        if levels is None:
            lvls = np.logspace(-.5, 2, 11)
        else:
            lvls = levels
            
        grid_x, grid_y = np.mgrid[xbounds[0]:xbounds[1]:100j,
                                  xbounds[2]:xbounds[3]:100j]
        points = self._xqgp 
        z = copy.copy(self._QGP_error)
        z[np.where(self._QGP_error>100)] = 100
        #print(z)
        #print('error capped to 100% for readibility')
        
        grid_z = griddata(points, z, (grid_x, grid_y), method='linear')
        grid_z = np.nan_to_num(grid_z)
        grid_z[np.where(grid_z<np.min(lvls))] = np.min(lvls)
        
        if idx_class is not None:
            grid_z[idx_class] = np.nan
        
        #FIRST PLOT: EXPERIMENTAL QGP ERROR
        
        fig1 = plt.figure(figsize = (12,6 ))
        plt.subplot(1, 2, 1)
        plt.ticklabel_format(style='sci', axis='x', scilimits=(-1,0))
        plt.xlabel(xlabel, size = 18)
        plt.ylabel(ylabel, size = 18)
        

        img0 = plt.contourf(grid_x, grid_y, grid_z, cmap = CMAP, levels = lvls, norm = LogNorm(),
                                         extent=(min(self._xqgp[:, 0]), max(self._xqgp[:, 0]),
                                            min(self._xqgp[:, 1]), max(self._xqgp[:, 1])), 
                                            origin='lower')
        cbar = plt.colorbar(img0, ticks = lvls, format=bar_format)#, label= zlabel)
        cbar.set_label(zlabel,size=18)
        plt.tight_layout()
        
        if plot_query_points:
            plt.scatter(self._xqgp[:, 0], self._xqgp[:, 1], marker = '+', c= 100-z, s = 50, cmap='gray', 
                        label = 'QGP evaluation points')
            plt.legend()
        
        plt.title('QGP: relative error', size = 20)
        
        
        
        #CALCULATING ERROR DUE TO PCA
        error_pca = []
        error_pca_rel = []
        
        mat = self._Res_dicts[0]['input_matrix']
        eig, mu = np.linalg.eig(mat)
        threshold = np.max(eig)/(2**input_dictionary['k']-1)

        eig_inv = 1./eig
        eig_inv[np.where(eig<threshold)] = 0
        mat_pca_inv = mu @ np.diag(eig_inv) @ mu.T

        for  i in range(len(self._Res_dicts)):
            U = self._Res_dicts[i]['input_vector_u']
            V = self._Res_dicts[i]['input_vector_v']
            if variance_case:
                V = self._Res_dicts[i]['input_vector_u']

            approximation = U@mat_pca_inv@V
            exact = np.dot(U.T, (np.linalg.solve(mat, V)))
            error_pca.append(approximation - exact)
            error_pca_rel.append(np.abs((exact-approximation)*100./exact))

        #PLOT2:PCA errors
        
        grid_x, grid_y = np.mgrid[xbounds[0]:xbounds[1]:100j,
                                  xbounds[2]:xbounds[3]:100j]
        points = self._xqgp 
        z = np.abs(np.array(error_pca_rel)) 
        z[np.where(z>100)] = 100
        
        grid_z = griddata(points, z, (grid_x, grid_y), method='linear')
        grid_z = np.nan_to_num(grid_z)
        grid_z[np.where(grid_z<np.min(lvls))] = np.min(lvls)
        
        if idx_class is not None:
            grid_z[idx_class] = np.nan
        
        plt.subplot(1, 2, 2)
        plt.ticklabel_format(style='sci', axis='x', scilimits=(-1,0))
        plt.xlabel(xlabel, size = 18)
        plt.ylabel(ylabel, size = 18)
        img1 = plt.contourf(grid_x, grid_y, grid_z, cmap = CMAP, levels = lvls, norm = LogNorm(),
                                         extent=(min(self._xqgp[:, 0]), max(self._xqgp[:, 0]),
                                            min(self._xqgp[:, 1]), max(self._xqgp[:, 1])), 
                                            origin='lower')


        zlabel = '$Error [\%]$'
        cbar = plt.colorbar(img1, ticks = lvls, format=bar_format)#, label= zlabel)
        cbar.set_label(zlabel,size=18)
        
        if plot_query_points:
            plt.scatter(self._xqgp[:, 0], self._xqgp[:, 1], marker = '+', c= 100-z, s = 50, cmap='gray', 
                        label = 'QGP evaluation points')
            plt.legend()
        
        plt.title('PCA: relative error', size = 20)

        plt.tight_layout()
        plt.subplots_adjust(top=0.85)
        if suptitle is not None:
            fig1.suptitle(suptitle, size = 22, y= 0.98)
        plt.show()
        
        
        fig2 = plt.figure(figsize = (12,6 ))
        
        plt.subplot(1, 2, 1)
        plt.ticklabel_format(style='sci', axis='x', scilimits=(-1,0))
        plt.xlabel(xlabel, size = 18)
        plt.ylabel(ylabel, size = 18)
        z = np.abs(self._QGP_error - np.abs(np.array(error_pca_rel)))
        grid_z = griddata(points, z, (grid_x, grid_y), method='linear')

        
        img2 = plt.contourf(grid_x, grid_y, grid_z, cmap = CMAP, norm = LogNorm())
        
        
        zlabel = 'Error $[\%]$'
        cbar = plt.colorbar(img2, format='%.1f%%')#, label= zlabel)
        cbar.set_label(zlabel,size=18)

        plt.scatter(self._xqgp[:, 0], self._xqgp[:, 1], marker = '+', c= 100-z, s = 50, cmap='gray')
        plt.title('$\epsilon_{QGP}[\%] - \epsilon_{PCA}[\%]$', size = 20)
        

        if savepath1 is not None:
            print('Saving figure!')
            fig1.savefig(savepath1, bbox_inches = 'tight')
            
        if savepath2 is not None:
            print('Saving figure!')
            fig2.savefig(savepath2)

        
        return fig1, fig2

def dict_validation_plot(input_dictionary, xbounds,m,X, Xnew, x_full, y_full, idx_class = None):
    xlabel = '$I_x$'
    ylabel = 'PD'
    CMAP = 'inferno'
    grid_x, grid_y = np.mgrid[xbounds[0]:xbounds[1]:100j,
                              xbounds[2]:xbounds[3]:100j]
    
    validation_mean = []
    validation_var = []

    for i in range(len(input_dictionary['u_list'] )):
        validation_mean.append(input_dictionary['u_list'][i] @ np.linalg.solve(input_dictionary['matrix'], input_dictionary['v_list']))
        validation_var.append(input_dictionary['u_list'][i] @ np.linalg.solve(input_dictionary['matrix'], input_dictionary['u_list'][i]))

    validation_mean  = np.array(validation_mean)    
    validation_var   = (m.kern.variance - np.array(validation_var))**0.5
    
    points =  input_dictionary['Xnew']
    z =  validation_mean
    #print(z)
    #print('error capped to 100%')

    grid_z = griddata(points, z, (grid_x, grid_y), method='linear')
    grid_z[idx_class] = np.nan

    fig = plt.figure(figsize = (12,6 ))
    plt.subplot(1, 2, 1)
    plt.ticklabel_format(style='sci', axis='x', scilimits=(-1,0))
    plt.xlabel(xlabel, size = 18)
    plt.ylabel(ylabel, size = 18)
    
    img0 = plt.contourf(grid_x, grid_y, grid_z, cmap = 'inferno' , 
                                        origin='lower')
    
    cbar = plt.colorbar(img0, format='%.1f%%')#, label= zlabel)
    zlabel = 'mean'
    cbar.set_label(zlabel,size=18)
    plt.tight_layout()

    plt.scatter(input_dictionary['Xnew'][:, 0], input_dictionary['Xnew'][:, 1], 
                marker = '+', c= 100-z, s = 50, cmap='gray')
    
    plt.title('Mean', size = 20)


    #PLOT2:
    grid_x, grid_y = np.mgrid[xbounds[0]:xbounds[1]:100j,
                              xbounds[2]:xbounds[3]:100j]
    points = input_dictionary['Xnew'] 
    z = validation_var
    
    grid_z = griddata(points, z, (grid_x, grid_y), method='linear')
    grid_z[idx_class] = np.nan

    plt.subplot(1, 2, 2)
    plt.ticklabel_format(style='sci', axis='x', scilimits=(-1,0))
    plt.xlabel(xlabel, size = 18)
    plt.ylabel(ylabel, size = 18)
    
    img1 = plt.pcolor(grid_x, grid_y, grid_z, cmap = 'viridis')
    cbar = plt.colorbar(img1)
    zlabel = 'STD'
    cbar.set_label(zlabel,size=18)

    plt.scatter(input_dictionary['Xnew'][:, 0], input_dictionary['Xnew'][:, 1],
                marker = '+', c= 100-z, s = 50, cmap='gray')
    plt.title('STD', size = 20)
    plt.tight_layout()
    
    plt.suptitle('Classically solved system preapred for QGP', y = 1.05)
    
    plot_GP_2d(X, Xnew, m, x_full, y_full,  idx_class=idx_class , 
               xlab = 'x1', ylab = 'x2', savepath = None, 
              suptitle = None)

        
