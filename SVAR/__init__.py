from SVAR.estSVAR import SVARest
from SVAR.estimatorCholesky import get_B_Cholesky
from SVAR.SVARbasics import get_IRF, plot_IRF, OLS_ReducedForm
from SVAR.estSVARbootstrap import bootstrap_SVAR
from SVAR.SVARutil import innovation, get_BMatrix, get_BVector, do_whitening



def test():
    print("Package: SVAR-GMM is working")
