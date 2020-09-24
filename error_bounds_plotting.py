import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from matplotlib import gridspec
from textwrap import wrap
from scipy.interpolate import interp1d

SMALL_SIZE = 16
MEDIUM_SIZE = 18
BIGGER_SIZE = 20

plt.rc('text', usetex=True)
plt.rc('font', family='sans-serif',size=20)

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=SMALL_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


def bound_custom(de, m, n, k ):
    return(1./(2.**k-1.) + np.sqrt(m)/(2.**(k - np.log(2.+1./(2*de)))-2))


def bound_nystrom(de, m, n, k ):
    return(1./(2.**k-1.) + 2*np.sqrt(2)*n*np.sqrt(1+np.sqrt(np.log(2./de)*(1-m/n))))

def bound_nystrom_wang(m, n, k):
    return(1./(2.**k-1.))*(np.sqrt(1.+(n+m)/m))

def bound_nystrom_wang_corrected(m, n, eig_m1):
    return(eig_m1*(np.sqrt(1.+(n+m)/m)))


def bound_custom_simple(de, m, n, k ):
    return(1./(2.**k-1.))*(1. + np.sqrt(m)/2.)

def PCA(k):
    return(1./(2.**k-1.))

def make_error_plots(matrix, error_N, error_QPE, m , m_QPE, k, 
                     suptitle = 'Numerical verification of the error bounds',
                     split_eigs = False, savepath = None, loglog = True, four_plots = True, de = 0.01):
    #    global lmax
    #matrix_list = mat_benchmarks_list
    #error_N = error_Nystrom_b
    #error_Q = error_QPE_p
    #m_QPE_b = m_QPE_p
    
    def func(x, r, b ):
        '''
        function for interpolating the
        eigenspectrum decay
        '''


        lmax = x[0]
        xtemp = x[1]
        return  lmax* r ** xtemp**b

    R = []
    b = []
    
    n = matrix.shape[0]

    eig = np.linalg.eigvals(matrix)
    eig = eig.real
    eig_max = np.max(eig)
    cut_indx = np.max(np.where((eig/eig_max)>1e-13))
    #print(cut_indx)

    fig = plt.figure(figsize = (12, 11))
    
    gs = gridspec.GridSpec(2, 2, height_ratios=[1.7, 1]) 
    
    if not four_plots:
        gs = gridspec.GridSpec(1, 2) 
        fig = plt.figure(figsize = (12, 6))
       
    
    ############
    #1) ERROR VS M
    ###########

    ax1 = fig.add_subplot(gs[0])
    if loglog:
        if split_eigs:
            ax1.loglog(m[:cut_indx], error_N[:cut_indx]/eig_max, 'C0.',
                         label = 'Nystrom error experimental, well-conditioned eigenmodes')
            #ax1.loglog(m[cut_indx:], error_N[cut_indx:]/eig_max, 'g.',
            #             label = 'Nystrom error experimental, ill-conditioned eigenmodes')
        else:
            ax1.loglog(m[:len(error_N)], error_N/eig_max, 'C0.',
                 label = 'Nystrom error (experimental)')
        #QPE
        ax1.loglog(m_QPE, error_QPE/eig_max,'*C3', label = 'QPE error (experimental)')#, marker = '.')


        ax1.loglog(m_QPE, bound_nystrom_wang(m_QPE, n, k), '-',  
               c = 'C0', label = 'Nystrom lower bound')

        ax1.loglog(m_QPE, bound_custom_simple(de, m_QPE, n, k ), '-', c = 'C3',alpha =0.5, 
               label = 'QPE upper bound: idealized')

        ax1.loglog(m_QPE, PCA(k)*np.ones(m_QPE.shape), 'k-', label = 'PCA approx. (QPE lower bound)')



        if min(bound_custom(de, m_QPE, n, k )<1e-12):
            bd = np.where(bound_custom(de, m_QPE, n, k )<1e-12) 
            bd = np.max(bd)+1
        else:
            bdidx = np.where(bound_custom(de, m_QPE, n, k )>0)
            bd = np.min(bdidx[0][1:])

        xm = m_QPE
        ax1.loglog(xm[bd:], bound_custom(de, m_QPE, n, k )[bd:],'--',c = 'C3',
                   label = 'QPE upper bound: probabilistic')
        plt.xlim([0.8, 2000])
    else:
        if split_eigs:
            ax1.semilogy(m[:cut_indx], error_N[:cut_indx]/eig_max, 'C0.',
                         label = 'Nystrom error experimental, well-conditioned eigenmodes')
            ax1.semilogy(m[cut_indx:], error_N[cut_indx:]/eig_max, 'g.',
                         label = 'Nystrom error experimental, ill-conditioned eigenmodes')
        else:
            ax1.semilogy(m, error_N/eig_max, 'C0.',
                 label = 'Nystrom error (experimental)')
        #QPE
        ax1.semilogy(m_QPE, error_QPE/eig_max,'*C3', label = 'QPE error (experimental)')#, marker = '.')


        ax1.semilogy(m_QPE, bound_nystrom_wang(m_QPE, n, k), '-',  
               c = 'C0', label = 'Nystrom lower bound')

        ax1.semilogy(m_QPE, bound_custom_simple(de, m_QPE, n, k ), '-', c = 'C3',alpha =0.5, 
               label = 'QPE upper bound: idealized')

        ax1.semilogy(m_QPE, PCA(k)*np.ones(m_QPE.shape), 'k-', label = 'PCA approx. (QPE lower bound)')


        #CUTTING-OFF EIGENMODES THAT WERE ILL CONDITIONED (CAPPING THE KAPPA)

        if min(bound_custom(de, m_QPE, n, k ))<1e-12:
            bd = np.where(bound_custom(de, m_QPE, n, k )<1e-12) 
            bd = np.max(bd)+1
        else:
            bd = 1
        xm = m_QPE
        ax1.semilogy(xm[bd:], bound_custom(de, m_QPE, n, k )[bd:],'--',c = 'C3',
                   label = 'QPE upper bound: probabilistic')

        plt.xlim([0.8, 120])

    #COMPARISON ESTIMATED PCA ERROR(K) VS ESTIMATED FROM THE REAL EIGENVALUES
    #plt.loglog(m, eig/eig_max[mat_num], '--k',  label = 'Normalized eigenspectrum ')



    
    plt.xlabel('m')
    plt.ylabel('Normalized error: '+r'$\epsilon/\lambda_{max}$')
    #plt.legend()
    plt.title('Normalized error vs. m')
    plt.grid(True)

    handles, labels = ax1.get_legend_handles_labels()
    labels_wrapped = [ '\n'.join(wrap(l, 50)) for l in labels]
    
    
    
    ##################
    #2) ERROR VS k
    ##################    
    ax2 = fig.add_subplot(gs[1])
    
    m_interp , indices = np.unique(m_QPE, return_index = True)
    k_interp = k[indices]
    k_interpolated = interp1d(m_interp, k_interp, kind='nearest', fill_value='extrapolate')

    if split_eigs:
        ax2.semilogy(k_interpolated(m[:cut_indx]), error_N[:cut_indx]/eig_max, 'C0.',
                     label = 'Nystrom error experimental, well-conditioned eigenmodes')
        ax2.semilogy(k_interpolated(m[cut_indx:]), error_N[cut_indx:]/eig_max, 'g.',
                     label = 'Nystrom error experimental, ill-conditioned eigenmodes')

    else:
        ax2.semilogy(k_interpolated(m[:len(error_N)]), error_N/eig_max, 'C0.',
                     label = 'Nystrom error experimental')
        
        
    ax2.semilogy(k, bound_nystrom_wang(m_QPE, n, k), '-',  c = 'C0', label = 'Nystrom lower bound')
    
    ax2.semilogy(k, error_QPE/eig_max,'C3*', label = 'QPE error (experimental)')
    ax2.semilogy(k, bound_custom_simple(de, m_QPE, n, k ), '-', c = 'C3',alpha =0.5, 
                 label = 'QPE upper bound: idealized')

    bd = np.where(bound_custom(de, m_QPE, n, k )<1e-12)
    bd = np.max(bd)+1

    ax2.semilogy(k[bd:], bound_custom(de, m_QPE, n, k )[bd:],'--',c = 'C3',label = 'QPE upper bound: probabilistic')
    ax2.semilogy(k, PCA(k)*np.ones(m_QPE.shape), 'k-', label = 'PCA approximate (QPE lower bound)')
    

    #NYSTROM BOUND CALCULATED FROM THE ACTUAL EIGENVALUES INSTEAD OF THE qpe ESTIMATE 1/(2^K-1) 
    #(BUT THEY ARE REALLY CLOSE)
    '''
    eig_m1 = interp1d(m, eig, kind='nearest', fill_value='extrapolate')    
    plt.semilogy(k, bound_nystrom_wang_corrected(m_QPE[:, mat_num], n, eig_m1(m_QPE[:, mat_num]))/eig_max[mat_num],
                 '-*',  c = 'C0', label = 'Nystrom lower bound correccted')
    '''
    
    plt.xlabel('k')
    plt.ylabel('Normalized error: '+r'$\epsilon/\lambda_{max}$')
    plt.title('Normalized error vs. k')
    plt.grid(True)
    
    
    
    ##################
    #3) EIGENSPECTRUM VS M
    ##################
    if four_plots:
        ax3 = fig.add_subplot(gs[2])
        if split_eigs:
            ax3.semilogy(eig[:cut_indx], 'C0.')
            ax3.semilogy(range(cut_indx, 1000), eig[cut_indx:], 'g.')
        else:
            ax3.semilogy(eig, 'k.', label = 'eigenvalues')
        #plt.semilogy(np.arange(1, 1001) , eig_max[mat_num]*0.998**np.arange(1, 1001))


        #iNTERPOLATING THEORETICAL EIGENVALUE DECAY - fitting THE R value
        xtemp = np.arange(0, cut_indx)
        lmax = eig_max# * np.ones(xtemp.shape)
        x_f_vars =  [lmax, xtemp]
        popt, pcov = curve_fit(func,x_f_vars, eig[:cut_indx])
        R.append(popt[0])
        b.append(popt[1])     

        ax3.semilogy(xtemp, func(x_f_vars, *popt), 'C1--', 
                     label = 'fit: '+r'$\lambda_{m} = \lambda_{max}(R^{{i}^b})$ ' +'\n'+ 
                     'R=%1.5f,  b=%5.3f' % tuple(popt))

        print(popt)
        plt.legend()
        plt.xlabel('m')
        plt.ylabel(r'$\lambda_{m}$')
        plt.title('System eigenspectrum')    
        plt.grid(True)

        ##################
        #3) m VS k
        ##################

        ax4 = fig.add_subplot(gs[3])
        ax4.semilogy(k, m_QPE, '.k', label = 'experimental data')
        plt.title('k vs. m')
        plt.ylabel('m')
        plt.xlabel('k')
        plt.legend()
        plt.grid(True)

        #fig.tight_layout() 
    
        plt.subplots_adjust( top = 0.95, wspace = 0.27, hspace =.62, right = 0.98, left = 0.02)
        lgnd =  fig.legend(handles, labels_wrapped, loc = 9, bbox_to_anchor=(0.55, 0.43), labelspacing = 1., ncol = 3)

    else:    
        plt.subplots_adjust( top = 0.92, wspace = 0.27, bottom =.3, right = 0.98, left = 0.02)
        lgnd =  fig.legend(handles, labels_wrapped, loc = 9, bbox_to_anchor=(0.55, 0.2), labelspacing = 1., ncol = 3)

        
    #fig.legend(handles, labels_wrapped, (.12, .36), labelspacing = 1., ncol = 3)#'upper left')
    if suptitle is not None:
        big_title = fig.suptitle(suptitle, y= 1.05)
        extr = [lgnd, big_title]
    else:
        extr = [lgnd]
    
    if savepath is not None:
        fig.savefig(savepath, bbox_extra_artists=extr, bbox_inches='tight')
        print('saving the fig')
    
    plt.show()

    print('Verifying Nystrom error\n')
    print('Sum diagonal Knn^[-1] @ Knn: ', np.sum(np.linalg.inv(matrix)@matrix))


    fig2= plt.figure(figsize = (6, 6))
    plt.loglog(error_N/eig_max, m[:len(error_N)]**2.,'.-C0', label = 'Nystrom')
    plt.loglog( error_QPE/eig_max, k, '.-C3', label = 'QPE')
    plt.ylabel('Relative increase in computational cost')
    plt.xlabel('Increase in accuracy (relative error)')
    plt.xlim([1e2, 1e-12])
    plt.ylim([0.5, 2e6])
    plt.grid(True)
    plt.legend()
    fig2.tight_layout()
    plt.show()

    
    return fig, fig2


