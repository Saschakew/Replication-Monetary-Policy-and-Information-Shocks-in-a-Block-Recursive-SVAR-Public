from scipy import optimize as opt 
import numpy as np
import pandas as pd
import SVAR
import SVAR.SVARutilGMM
import SVAR.estOutput
import SVAR.estimatorPML
import SVAR.estimatorGMM
import SVAR.estimatorGMMW
import SVAR.estimatorGMMWF
import SVAR.estimatorCholesky
from scipy import optimize
import copy
from scipy.optimize import fmin_slsqp

def SVARest(u, estimator='GMM', options=dict(), prepOptions=dict(), prepared=False):
    if estimator == 'GMM':
        if not (prepared):
            prepOptions['estimator'] = 'GMM'
            options = SVAR.estimatorGMM.prepareOptions(u=u, **prepOptions)

        def optimize_this(options):
            this_loss = lambda b_vec: SVAR.estimatorGMM.loss(u, b_vec,
                                                             restrictions=options['restrictions'],
                                                             moments=options['moments'],
                                                             moments_powerindex=options['moments_powerindex'],
                                                             W=options['W'] )
            # this_grad = lambda b_vec: SVAR.estimatorGMM.gradient(u, b_vec,
            #                                                      Jacobian=options['Jacobian'],
            #                                                      W=options['W'],
            #                                                      restrictions=options['restrictions'],
            #                                                      moments=options['moments'],
            #                                                      moments_powerindex=options['moments_powerindex'])
            this_grad = []
            optim_start = options['bstart']
            if np.shape(optim_start)[0] == 0:
                # raise ValueError('No free parameters.')
                ret = dict()
                ret['x'] = np.array([])
                ret['fun'] = this_loss(ret['x'])
            else:

                ret_tmp = opt.minimize(this_loss, optim_start, method='L-BFGS-B', jac=this_grad,
                             bounds=None,
                             tol=None,
                             callback=None,
                             options={'disp': False,
                                      'maxcor': 10,
                                      'ftol': 2.220446049250313e-09,
                                      'gtol': 1e-09,
                                      'eps': 1e-12,
                                      'maxfun': 15000,
                                      'maxiter': 15000,
                                      'iprint': - 1,
                                      'maxls': 40,
                                      'finite_diff_rel_step': None})

                if ret_tmp['success']==False:
                    print('Optimization failed')
                    print(ret_tmp['message'])
                ret = dict()
                ret['x'] = ret_tmp.x
                ret['fun'] = ret_tmp.fun

                if ret['fun'] < 0:
                    raise ValueError("Error: Negative loss")
            return ret

        est_SVAR = optimize_this(options)

        for k in range(1, options['kstep']):
            bstart = est_SVAR['x']
            options['W'] = SVAR.SVARutilGMM.get_W_opt(u, b=bstart, restrictions=options['restrictions'],
                                                      moments=options['moments'],
                                                      Wpara=options['Wpara'],
                                                      S_func=options['S_func'])
            est_SVAR = optimize_this(options)


        out_SVAR = SVAR.estimatorGMM.SVARout(est_SVAR, options, u)

    elif estimator =='CSUE':
        if not (prepared):
            prepOptions['estimator'] = 'CSUE'
            options = SVAR.estimatorGMM.prepareOptions(u=u, **prepOptions)

        def optimize_this(options):

            get_d = lambda b_vec: np.divide(1,np.sqrt(np.sum(np.power(SVAR.innovation(u, b_vec ), 2),axis=0)/ options['T']))
            get_dprod = lambda b_vec: np.prod(np.power(get_d(b_vec), options['moments'] ),axis=1)


            Wupd = lambda b_vec: np.matmul(np.matmul(  np.diag( get_dprod(b_vec)  ) ,options['W']), np.diag( get_dprod(b_vec)  ) )

            this_loss = lambda b_vec: SVAR.estimatorGMM.loss(u, b_vec,
                                                             restrictions=options['restrictions'],
                                                             moments=options['moments'],
                                                             moments_powerindex=options['moments_powerindex'],
                                                             W=Wupd(b_vec) )

            this_grad = []

            optim_start = options['bstart']
            if np.shape(optim_start)[0] == 0:
                # raise ValueError('No free parameters.')
                ret = dict()
                ret['x'] = np.array([])
                ret['fun'] = this_loss(ret['x'])
            else:
                ret_tmp = opt.minimize(this_loss , optim_start, method='L-BFGS-B', jac=this_grad )
                if ret_tmp['success'] == False:
                    print('Optimization failed')
                    print(ret_tmp['message'])
                ret = dict()
                ret['x'] = ret_tmp.x
                ret['fun'] = ret_tmp.fun

            return ret

        est_SVAR = optimize_this(options)


        for k in range(1, options['kstep']):
            bstart = est_SVAR['x']
            options['W'] = SVAR.SVARutilGMM.get_W_opt(u, b=bstart, restrictions=options['restrictions'],
                                                      moments=options['moments'],
                                                      Wpara=options['Wpara'],
                                                      S_func=options['S_func'])
            est_SVAR = optimize_this(options)

        out_SVAR = SVAR.estimatorGMM.SVARout(est_SVAR, options, u)

    elif estimator == 'CUE':
        if not (prepared):
            prepOptions['estimator'] = 'CUE'
            options = SVAR.estimatorGMM.prepareOptions(u=u, **prepOptions)

        def optimize_this(options):
            this_loss = lambda b_vec: SVAR.estimatorGMM.loss(u, b_vec,
                                                             restrictions=options['restrictions'],
                                                             moments=options['moments'],
                                                             moments_powerindex=options['moments_powerindex'],
                                                             W=SVAR.SVARutilGMM.get_W_opt(u, b=b_vec,
                                                                                          restrictions=options[
                                                                                               'restrictions'],
                                                                                          moments=options['moments'],
                                                                                          Wpara=options['Wpara'],
                                                                                          S_func=options['S_func']))




            #eps = np.sqrt(np.finfo(float).eps)
            #this_grad = lambda b_vec: optimize.approx_fprime(b_vec, this_loss, eps)
            this_grad = []

            optim_start = options['bstart']
            if np.shape(optim_start)[0] == 0:
                # raise ValueError('No free parameters.')
                ret = dict()
                ret['x'] = np.array([])
                ret['fun'] = this_loss(ret['x'])
            else:
                # ret_tmp = opt.minimize(this_loss, optim_start, method='L-BFGS-B', jac=this_grad,options={'disp': True})
                ret_tmp = opt.minimize(this_loss, optim_start, method='L-BFGS-B', jac=this_grad )
                if ret_tmp['success']==False:
                    print('Optimization failed')
                    print(ret_tmp['message'])
                ret = dict()
                ret['x'] = ret_tmp.x
                ret['fun'] = ret_tmp.fun


                if ret['fun'] < 0:
                    raise ValueError("Error: Negative loss")
            return ret

        est_SVAR = optimize_this(options)

        options['W'] = SVAR.SVARutilGMM.get_W_opt(u, b=est_SVAR['x'], restrictions=options['restrictions'],
                                                  moments=options['moments'],
                                                  Wpara=options['Wpara'],
                                                  S_func=options['S_func'])
        out_SVAR = SVAR.estimatorGMM.SVARout(est_SVAR, options, u)

    elif estimator == 'GMM_W':
        if not (prepared):
            prepOptions['estimator'] = 'GMM_W'
            options = SVAR.estimatorGMMW.prepareOptions(u=u, **prepOptions)
        z, options['V'] = SVAR.do_whitening(u, white=True)

        def optimize_this(options):
            this_loss = lambda b_vec: SVAR.estimatorGMMW.loss(z, b_vec,
                                                              restrictions=options['restrictions'],
                                                              moments=options['moments'],
                                                              moments_powerindex=options['moments_powerindex'],
                                                              W=options['W'],
                                                              blocks=options['blocks'])
            #this_grad =  lambda b_vec: optimize.approx_fprime(b_vec, this_loss, epsilon=1.4901161193847656e-08)
            this_grad =[]

            optim_start = options['bstart']
            if np.shape(optim_start)[0] == 0:
                # raise ValueError('No free parameters.')
                ret = dict()
                ret['x'] = np.array([])
                ret['fun'] = this_loss(ret['x'])
            else:
                ret_tmp = opt.minimize(this_loss, optim_start, method='L-BFGS-B', jac=this_grad)
                if ret_tmp['success']==False:
                    print('Optimization failed')
                    print(ret_tmp['message'])
                ret = dict()
                ret['x'] = ret_tmp.x
                ret['fun'] = ret_tmp.fun

                if ret['fun'] < 0:
                    raise ValueError("Error: Negative loss")

            return ret

        est_SVAR = optimize_this(options)

        out_SVAR = SVAR.estimatorGMMW.SVARout(est_SVAR, options, u)

    elif estimator == 'GMM_WF':
        if not (prepared):
            prepOptions['estimator'] = 'GMM_WF'
            options = SVAR.estimatorGMMWF.prepareOptions(u=u, **prepOptions)
        z, options['V'] = SVAR.do_whitening(u, white=True)

        def optimize_this(options):
            if options['moments_MeanIndep']:
                this_loss = lambda b_vec: SVAR.estimatorGMMWF.loss(z, b_vec,
                                                                   restrictions=options['restrictions'],
                                                                   moments=options['moments'],
                                                                   moments_powerindex=options['moments_powerindex'],
                                                                   blocks=options['blocks'])-\
                                        SVAR.estimatorGMMWF.loss_MIcorrection(z,b_vec,restrictions=options['restrictions'],
                                                                  moments=options['moments_MIcorrection'],
                                                                  moments_powerindex=options['moments_MIcorrection_powerindex'], blocks=options['blocks'])
            else:
                this_loss = lambda b_vec: SVAR.estimatorGMMWF.loss(z, b_vec,
                                                                   restrictions=options['restrictions'],
                                                                   moments=options['moments'],
                                                                   moments_powerindex=options['moments_powerindex'],
                                                                   blocks=options['blocks'])
            #this_grad =  lambda b_vec: optimize.approx_fprime(b_vec, this_loss, epsilon=1.4901161193847656e-08)
            this_grad =[]

            optim_start = options['bstart']
            if np.shape(optim_start)[0] == 0:
                # raise ValueError('No free parameters.')
                ret = dict()
                ret['x'] = np.array([])
                ret['fun'] = this_loss(ret['x'])
            else:

                ret_tmp = opt.minimize(this_loss, optim_start, method='L-BFGS-B', jac=this_grad)

                if ret_tmp['success'] == False:
                    print('Optimization failed')
                    print(ret_tmp['message'])
                ret = dict()
                ret['x'] = ret_tmp.x
                ret['fun'] = ret_tmp.fun
                # if ret['fun'] < 0:
                #     raise ValueError("Error: Negative loss")
            return ret

        est_SVAR = optimize_this(options)

        out_SVAR = SVAR.estimatorGMMWF.SVARout(est_SVAR, options, u)

    elif estimator == 'PML':
        if not (prepared):
            prepOptions['estimator'] = 'PML'
            options = SVAR.estimatorPML.prepareOptions(u=u, **prepOptions)
        z, options['V'] = SVAR.do_whitening(u, white=True)

        def optimize_this(options):
            this_loss = lambda b_vec: SVAR.estimatorPML.LogLike_t(z, b_vec, options['df_t'], blocks=options['blocks'],
                                                                  whiten=options['whiten'])
            this_grad = []

            optim_start = options['bstart']
            if np.shape(optim_start)[0] == 0:
                # raise ValueError('No free parameters.')
                ret = dict()
                ret['x'] = np.array([])
                ret['fun'] = this_loss(ret['x'])
            else:
                ret_tmp = opt.minimize(this_loss, optim_start, method='L-BFGS-B', jac=this_grad)
                if ret_tmp['success'] == False:
                    print('Optimization failed')
                    print(ret_tmp['message'])
                ret = dict()
                ret['x'] = ret_tmp.x
                ret['fun'] = ret_tmp.fun


            return ret

        est_SVAR = optimize_this(options)

        out_SVAR = SVAR.estimatorPML.SVARout(est_SVAR, options, u)

    elif estimator == 'Cholesky':
        out_SVAR = SVAR.estimatorCholesky.get_B_Cholesky(u)

    else:
        print('Unknown estimator')


    return out_SVAR
