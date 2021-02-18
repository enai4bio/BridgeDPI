import numpy as np
from sklearn import metrics as skmetrics
import warnings
warnings.filterwarnings("ignore")

def lgb_MaF(preds, dtrain):
    Y = np.array(dtrain.get_label(), dtype=np.int32)
    preds = preds.reshape(-1,len(Y))
    Y_pre = np.argmax( preds, axis=0 )
    return 'macro_f1', float(F1(preds.shape[0], Y_pre, Y, 'macro')), True

def lgb_precision(preds, dtrain):
    Y = dtrain.get_label()
    preds = preds.reshape(-1,len(Y))
    Y_pre = np.argmax( preds, axis=0 )
    return 'precision', float(Counter(Y==Y_pre)[True]/len(Y)), True

id2lab = [[-1,-1]]*20
for a in range(1,11):
    for s in [1,2]:
        id2lab[a-1+(s-1)*10] = [a,s]

class Metrictor:
    def __init__(self):
        self._reporter_ = {"ACC":self.ACC, "AUC":self.AUC, "Precision":self.Precision, "Recall":self.Recall, "F1":self.F1, "LOSS":self.LOSS}
    def __call__(self, report, end='\n'):
        res = {}
        for mtc in report:
            v = self._reporter_[mtc]()
            print(f" {mtc}={v:6.3f}", end=';')
            res[mtc] = v
        print(end=end)
        return res
    def set_data(self, Y_prob_pre, Y, threshold=0.5):
        self.Y = Y.astype('int')
        if len(Y_prob_pre.shape)>1:
            self.Y_prob_pre = Y_prob_pre[:,1]
            self.Y_pre = Y_prob_pre.argmax(axis=-1)
        else:
            self.Y_prob_pre = Y_prob_pre
            self.Y_pre = (Y_prob_pre>threshold).astype('int')
    @staticmethod
    def table_show(resList, report, rowName='CV'):
        lineLen = len(report)*8 + 6
        print("="*(lineLen//2-6) + "FINAL RESULT" + "="*(lineLen//2-6))
        print(f"{'-':^6}" + "".join([f"{i:>8}" for i in report]))
        for i,res in enumerate(resList):
            print(f"{rowName+'_'+str(i+1):^6}" + "".join([f"{res[j]:>8.3f}" for j in report]))
        print(f"{'MEAN':^6}" + "".join([f"{np.mean([res[i] for res in resList]):>8.3f}" for i in report]))
        print("======" + "========"*len(report))
    def each_class_indictor_show(self, id2lab):
        print('Waiting for finishing...')

    def ACC(self):
        return ACC(self.Y_pre, self.Y)
    def AUC(self):
        return AUC(self.Y_prob_pre,self.Y)
    def Precision(self):
        return Precision(self.Y_pre, self.Y)
    def Recall(self):
        return Recall(self.Y_pre, self.Y)
    def F1(self):
        return F1(self.Y_pre, self.Y)
    def LOSS(self):
        return LOSS(self.Y_prob_pre,self.Y)
    

def ACC(Y_pre, Y):
    return (Y_pre==Y).sum() / len(Y)

def AUC(Y_prob_pre, Y):
    return skmetrics.roc_auc_score(Y, Y_prob_pre)

def Precision(Y_pre, Y):
    return skmetrics.precision_score(Y, Y_pre)

def Recall(Y_pre, Y):
    return skmetrics.recall_score(Y, Y_pre)

def F1(Y_pre, Y):
    return skmetrics.f1_score(Y, Y_pre)

def LOSS(Y_prob_pre, Y):
    Y_prob_pre,Y = Y_prob_pre.reshape(-1),Y.reshape(-1)
    Y_prob_pre[Y_prob_pre>0.99] -= 1e-3
    Y_prob_pre[Y_prob_pre<0.01] += 1e-3
    return -np.mean(Y*np.log(Y_prob_pre) + (1-Y)*np.log(1-Y_prob_pre))
    