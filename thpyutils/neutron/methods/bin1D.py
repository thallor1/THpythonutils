import numpy as np

def bin1D(x,y,yerr,bins,statistic='mean',fill=False):
    """
    Given a list of bin edges, values, and errors, computes weighted average with correct error bars\
    for each bin center.
    :param np.ndarray x: Coordinate values.
    :param np.ndarray y: Values
    :param np.ndarray yerr: Value errors.
    :param np.ndarray bins: Bin edges.

    """
    x,y,yerr=np.array(x),np.array(y),np.array(yerr)
    x_bin=[]
    y_bin=[]
    yerr_bin = []
    x=x[~np.isnan(y)]
    yerr = yerr[~np.isnan(y)]
    y=y[~np.isnan(y)]
    for i in range(len(bins)-1):
        val_ind = np.intersect1d(np.where(x>bins[i]),np.where(x<=bins[i+1]))

        if len(val_ind)>0 and fill==False:
            x_bin.append(np.mean([bins[i],bins[i+1]]))
            y_bin.append(np.average(np.array(y)[val_ind],weights=1.0/(np.array(yerr)[val_ind])))
            yerr_bin.append(np.sqrt(np.sum((np.array(yerr)[val_ind])**2))/len(val_ind))
        elif fill==True:
            x_bin.append(np.mean([bins[i],bins[i+1]]))
            yerr=np.ones(len(yerr))
            y_bin.append(np.average(np.array(y)[val_ind],weights=1.0/(np.array(yerr)[val_ind])))
            yerr_bin.append(np.sqrt(np.sum((np.array(yerr)[val_ind])**2))/len(val_ind))
    return np.array(x_bin),np.array(y_bin),np.array(yerr_bin)