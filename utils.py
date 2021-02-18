from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from gensim.models import Word2Vec
from collections import Counter
import numpy as np
import pandas as pd
from tqdm import tqdm
import os,logging,pickle,random,torch,gc,deepchem,gc
from deepchem.models.graph_models import GraphConvModel
from deepchem.feat import graph_features
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
from sklearn.feature_extraction.text import CountVectorizer
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

#with open("pssm_arr_pdb.pkl", 'rb') as f:
#    pssmArr = pickle.load(f)

class DataClass:
    def __init__(self, dataPath, pSeqMaxLen=1024, dSeqMaxLen=128, kmers=-1):
        # Open files and load data
        print('Loading the raw data...')
        self.p2id,self.id2p = {},[]
        self.d2id,self.id2d = {},[]
        pCnt,dCnt = 0,0
        pSeqData,pPSSMData,gSeqData,dMolData,dSeqData,dFeaData,dFinData = [],[],[],[],[],[],[]
        pNameData,dNameData = {'train':[],'valid':[],'test':[]},{'train':[],'valid':[],'test':[]}
        eSeqData,edgeLab = {},{}
        
        atomFeaturizer = graph_features.WeaveFeaturizer()
        for sub in ['train','valid','test']:
            path = os.path.join(dataPath,sub if sub!='valid' else 'dev')
            id2am,id2go = [i.strip() for i in open(os.path.join(path,'protein.vocab'),'r').readlines()],[i.strip() for i in open(os.path.join(path,'protein.goa.vocab'))]
            pnd,dnd = [],[]
            for name,pSeq,gSeq in zip(tqdm(open(os.path.join(path,'protein'),'r').readlines()),open(os.path.join(path,'protein.repr'),'r').readlines(),open(os.path.join(path,'protein.goa.repr'),'r').readlines()):
                name,pSeq,gSeq = 'p_'+name.strip(),pSeq.split(),gSeq.split()
                pnd.append(name)

                if name not in self.p2id:
                    self.p2id[name] = pCnt
                    self.id2p.append(name)
                    pCnt += 1

                    pSeqData.append( [id2am[int(i)] for i in pSeq] )
                    gSeqData.append( [id2go[int(i)] for i in gSeq] )
                    pPSSMData.append( pssmArr[name][:pSeqMaxLen] )

            for name,smi in zip(tqdm(open(os.path.join(path,'chem'),'r').readlines()),open(os.path.join(path,'chem.repr'),'r').readlines()):
                name,smi = 'd_'+name.strip(),smi.strip()
                dnd.append(name)

                if name not in self.d2id:
                    self.d2id[name] = dCnt
                    self.id2d.append(name)
                    dCnt += 1

                    mol = Chem.MolFromSmiles(smi)
                    dMolData.append( mol )
                    dSeqData.append( [a.GetSymbol() for a in mol.GetAtoms()] )
                    dFeaData.append( [graph_features.atom_features(a) for a in mol.GetAtoms()] )
                    #dFinData.append( list(AllChem.GetMorganFingerprintAsBitVect(mol,2,nBits=1024).ToBitString()) )
                    #dFinData.append( list(AllChem.RDKFingerprint(mol,fpSize=1024).ToBitString()) )

                    tmp2,tmp3,tmp4 = np.ones((1,)),np.ones((1,)),np.ones((1,))
                    DataStructs.ConvertToNumpyArray(AllChem.GetHashedMorganFingerprint(mol,2, nBits=1024), tmp2)
                    #DataStructs.ConvertToNumpyArray(AllChem.GetHashedMorganFingerprint(mol,3, nBits=1024), tmp3)
                    #DataStructs.ConvertToNumpyArray(AllChem.GetHashedMorganFingerprint(mol,4, nBits=1024), tmp4)

                    dFinData.append( tmp2 )
                    
            posEdge,negEdge = [i.strip().split(',')[1::2] for i in open(os.path.join(path,'edges.pos'),'r').readlines()],[i.strip().split(',')[1::2] for i in open(os.path.join(path,'edges.neg'),'r').readlines()]
            posEdge,negEdge = [[self.p2id['p_'+p],self.d2id['d_'+d],1] for d,p in posEdge],[[self.p2id['p_'+p],self.d2id['d_'+d],0] for d,p in negEdge]
            eSeqData[sub] = posEdge+negEdge

            edgeLab[sub] = -np.ones((pCnt,dCnt),dtype=np.int32)
            edgeLab[sub][[i[0] for i in posEdge],[i[1] for i in posEdge]] = 1
            edgeLab[sub][[i[0] for i in negEdge],[i[1] for i in negEdge]] = 0

            invalidP = set([self.id2p[i] for i in np.arange(len(self.id2p))[(edgeLab[sub]>-1).sum(axis=1)==0]])
            invalidD = set([self.id2d[i] for i in np.arange(len(self.id2d))[(edgeLab[sub]>-1).sum(axis=0)==0]])
            pNameData[sub],dNameData[sub] = [i for i in pnd if i not in invalidP],[i for i in dnd if i not in invalidD]
        
        #dFeaData = [i.get_atom_features().tolist() for i in atomFeaturizer.featurize(dMolData, log_every_n=10000)]
        if kmers>0:
            pSeqData_k = [[' ']*(kmers//2) + seq + [' ']*(kmers//2) for seq in pSeqData]
            pSeqData_k = [[''.join(seq[i:i+kmers]) for i in range(len(seq))] for seq in pSeqData_k]

            dSeqData_k = [[' ']*(kmers//2) + seq + [' ']*(kmers//2) for seq in dSeqData]
            dSeqData_k = [[''.join(seq[i:i+kmers]) for i in range(len(seq))] for seq in dSeqData_k]

        # Get the mapping variables
        print('Getting the mapping variables......')
        self.am2id,self.id2am = {"<UNK>":0,"<EOS>":1},["<UNK>","<EOS>"]
        self.go2id,self.id2go = {"<UNK>":0,"<EOS>":1},["<UNK>","<EOS>"]
        amCnt,goCnt = 2,2
        for pSeq,gSeq in zip(pSeqData,gSeqData):
            for am in pSeq:
                if am not in self.am2id:
                    self.am2id[am] = amCnt
                    self.id2am.append(am)
                    amCnt += 1
            for go in gSeq:
                if go not in self.go2id:
                    self.go2id[go] = goCnt
                    self.id2go.append(go)
                    goCnt += 1
        self.amNum,self.goNum = amCnt,goCnt

        self.at2id,self.id2at = {"<UNK>":0,"<EOS>":1},["<UNK>","<EOS>"]
        atCnt = 2
        for dSeq in dSeqData:
            for at in dSeq:
                if at not in self.at2id:
                    self.at2id[at] = atCnt
                    self.id2at.append(at)
                    atCnt += 1
        self.atNum = atCnt

        if kmers>0:
            self.kam2id,self.id2kam = {"<UNK>":0,"<EOS>":1},["<UNK>","<EOS>"]
            kamCnt = 2
            for pSeq in pSeqData_k:
                for kam in pSeq:
                    if kam not in self.kam2id:
                        self.kam2id[kam] = kamCnt
                        self.id2kam.append(kam)
                        kamCnt += 1
            self.kamNum = kamCnt
            
            self.kat2id,self.id2kat = {"<UNK>":0,"<EOS>":1},["<UNK>","<EOS>"]
            katCnt = 2
            for dSeq in dSeqData_k:
                for kat in dSeq:
                    if kat not in self.kat2id:
                        self.kat2id[kat] = katCnt
                        self.id2kat.append(kat)
                        katCnt += 1
            self.katNum = katCnt

        # Tokenized protein data
        pSeqTokenized,gSeqTokenized = [],[]
        pSeqLen = []
        for pSeq,gSeq in zip(pSeqData,gSeqData):
            pSeq,gSeq = [self.am2id[am] for am in pSeq],[self.go2id[go] for go in gSeq]
            pSeqLen.append( min(len(pSeq),pSeqMaxLen) )
            pSeqTokenized.append( pSeq[:pSeqMaxLen] + [1]*max(pSeqMaxLen-len(pSeq),0) )
            gSeqTokenized.append( gSeq[:pSeqMaxLen] + [1]*max(pSeqMaxLen-len(gSeq),0) )

        if kmers>0:
            pSeqTokenized_k = []
            pSeqLen_k = []
            for pSeq in pSeqData_k:
                pSeq = [self.kam2id[kam] for kam in pSeq]
                pSeqLen_k.append( min(len(pSeq),pSeqMaxLen) )
                pSeqTokenized_k.append( pSeq[:pSeqMaxLen] + [1]*max(pSeqMaxLen-len(pSeq),0) )
            
            dSeqTokenized_k = []
            dSeqLen_k = []
            for dSeq in dSeqData_k:
                dSeq = [self.kat2id[kat] for kat in dSeq]
                dSeqLen_k.append( min(len(dSeq),dSeqMaxLen) )
                dSeqTokenized_k.append( dSeq[:dSeqMaxLen] + [1]*max(dSeqMaxLen-len(dSeq),0) )

        # Presolve the data
        print('Presolving the data...')
        dAdjMat,D = [],np.zeros((atCnt,atCnt),dtype=np.float32)
        dSeqTokenized = []
        dSeqLen = []
        for dSeq,dMol in zip(tqdm(dSeqData),dMolData):
            dAdj = np.zeros((atCnt,atCnt),dtype=np.float32)
            atoms = [self.at2id[i] for i in dSeq]
            dSeqLen.append( min(len(dSeq),dSeqMaxLen) )
            dSeqTokenized.append( atoms[:dSeqMaxLen] + [1]*max(dSeqMaxLen-len(atoms),0) )
            tmp = dAdj[atoms,:]
            tmp[:,atoms] = Chem.rdmolops.GetAdjacencyMatrix(dMol)
            dAdj[atoms,:] = tmp
            dAdj[:,atoms] = tmp.T
            dAdj[range(atCnt),range(atCnt)] = 1
            
            D[range(atCnt),range(atCnt)] = 1/dAdj.sum(axis=0)**0.5
            
            dAdjMat.append(D.dot(dAdj).dot(D))

        # Finish
        print('Finished...')
        #self.trainSampleNum,self.validSampleNum,self.testSampleNum = len(pNameData['train']+dNameData['train']),len(pNameData['valid']+dNameData['valid']),len(pNameData['test']+dNameData['test'])
        self.trainSampleNum,self.validSampleNum,self.testSampleNum = len(eSeqData['train']),len(eSeqData['valid']),len(eSeqData['test'])
        self.pNameData,self.dNameData = pNameData,dNameData
        self.pSeqData,self.gSeqData = pSeqData,gSeqData
        self.dSeqData,self.dMolData = dSeqData,dMolData
        self.pSeqLen,self.dSeqLen = np.array(pSeqLen, dtype=np.int32),np.array(dSeqLen, dtype=np.int32)
        self.pSeqTokenized,self.gSeqTokenized = np.array(pSeqTokenized,dtype=np.int32),np.array(gSeqTokenized,dtype=np.int32)
        self.pPSSMFeat = np.array([np.vstack([i,np.zeros((pSeqMaxLen-len(i),20))]) for i in pPSSMData], dtype=np.int32)
        
        if kmers>0:
            self.pSeqData_k = pSeqData_k
            self.dSeqData_k = dSeqData_k
            self.pSeqTokenized_k = np.array(pSeqTokenized_k, dtype=np.int32)
            self.dSeqTokenized_k = np.array(dSeqTokenized_k, dtype=np.int32)
        else:
            self.pSeqTokenized_k = np.array([-1]*len(pSeqData))
            self.dSeqTokenized_k = np.array([-1]*len(dSeqData))

        ctr = CountVectorizer(ngram_range=(1, 3), analyzer='char')
        pContFeat = ctr.fit_transform([''.join(i) for i in self.pSeqData]).toarray().astype('float32')
        k1,k2,k3 = [len(i)==1 for i in ctr.get_feature_names()],[len(i)==2 for i in ctr.get_feature_names()],[len(i)==3 for i in ctr.get_feature_names()]
        
        pContFeat[:,k1] = (pContFeat[:,k1] - pContFeat[:,k1].mean(axis=1).reshape(-1,1))/(pContFeat[:,k1].std(axis=1).reshape(-1,1)+1e-8)
        pContFeat[:,k2] = (pContFeat[:,k2] - pContFeat[:,k2].mean(axis=1).reshape(-1,1))/(pContFeat[:,k2].std(axis=1).reshape(-1,1)+1e-8)
        pContFeat[:,k3] = (pContFeat[:,k3] - pContFeat[:,k3].mean(axis=1).reshape(-1,1))/(pContFeat[:,k3].std(axis=1).reshape(-1,1)+1e-8)
        pContFeat = (pContFeat-pContFeat.mean(axis=0)) / (pContFeat.std(axis=0)+1e-8)
        self.pContFeat = pContFeat

        ctr = CountVectorizer(ngram_range=(1,3), analyzer='char')
        dContFeat = ctr.fit_transform([''.join(i) for i in self.dSeqData]).toarray().astype('float32')
        k1,k2,k3 = [len(i)==1 for i in ctr.get_feature_names()],[len(i)==2 for i in ctr.get_feature_names()],[len(i)==3 for i in ctr.get_feature_names()]

        dContFeat[:,k1] = (dContFeat[:,k1] - dContFeat[:,k1].mean(axis=1).reshape(-1,1))/(dContFeat[:,k1].std(axis=1).reshape(-1,1)+1e-8)
        dContFeat[:,k2] = (dContFeat[:,k2] - dContFeat[:,k2].mean(axis=1).reshape(-1,1))/(dContFeat[:,k2].std(axis=1).reshape(-1,1)+1e-8)
        dContFeat[:,k3] = (dContFeat[:,k3] - dContFeat[:,k3].mean(axis=1).reshape(-1,1))/(dContFeat[:,k3].std(axis=1).reshape(-1,1)+1e-8)
        dContFeat = (dContFeat-dContFeat.mean(axis=0)) / (dContFeat.std(axis=0)+1e-8)
        self.dContFeat = dContFeat

        self.dSeqTokenized = np.array(dSeqTokenized,dtype=np.int32)
        self.dGraphFeat = np.array([i+[[0]*75]*(dSeqMaxLen-len(i)) for i in dFeaData], dtype=np.int8)
        self.dFinprFeat = np.array(dFinData, dtype=np.float32)
        self.dAdjMat = np.array(dAdjMat,dtype=np.float32)
        self.eSeqData,self.edgeLab = eSeqData,edgeLab
        self.vector = {}
    def describe(self):
        pass
    # creative_id, ad_id, product_id, product_category, advertiser_id, industry
    def vectorize(self, method="char2vec", amSize=16, goSize=16, atSize=16, window=25, sg=1, kmers=-1,
                        workers=8, loadCache=True, pos=False, suf=''):
        if method == 'char2vec':
            path = f'cache/{method}_am{amSize}_go{goSize}_at{atSize}.pkl'
            if os.path.exists(path) and loadCache:
                with open(path, 'rb') as f:
                    self.vector['embedding'] = pickle.load(f)
                print(f'Loaded cache from {path}.')
                return
            self.vector['embedding'] = {}
            
            print('training amino chars...')
            amDoc = [pSeq+['<EOS>'] for pSeq in self.pSeqData]
            model = Word2Vec(amDoc, min_count=0, window=window, size=amSize, workers=workers, sg=sg, iter=10)
            char2vec = np.zeros((self.amNum, amSize), dtype=np.float32)
            for i in range(self.amNum):
                if self.id2am[i] in model.wv:
                    char2vec[i] = model.wv[self.id2am[i]]
                else:
                    print(f'{self.id2am[i]} not in vocab, random initialize it...')
                    char2vec[i] = np.random.random((1,amSize))
            self.vector['embedding']['amino'] = char2vec
            
            print('training go chars...')
            goDoc = [gSeq+['<EOS>'] for gSeq in self.gSeqData]
            model = Word2Vec(goDoc, min_count=0, window=window, size=goSize, workers=workers, sg=sg, iter=10)
            char2vec = np.zeros((self.goNum, goSize), dtype=np.float32)
            for i in range(self.goNum):
                if self.id2go[i] in model.wv:
                    char2vec[i] = model.wv[self.id2go[i]]
                else:
                    print(f'{self.id2go[i]} not in vocab, random initialize it...')
                    char2vec[i] = np.random.random((1,goSize))
            self.vector['embedding']['go'] = char2vec
            
            print('training atmo chars...')
            atDoc = [dSeq+['<EOS>'] for dSeq in self.dSeqData]
            model = Word2Vec(atDoc, min_count=0, window=window, size=atSize, workers=workers, sg=sg, iter=10)
            char2vec = np.zeros((self.atNum, atSize), dtype=np.float32)
            for i in range(self.atNum):
                if self.id2at[i] in model.wv:
                    char2vec[i] = model.wv[self.id2at[i]]
                else:
                    print(f"{self.id2at[i]} not in vocab, random initialize it...")
                    char2vec[i] = np.random.random((1,atSize))
            self.vector['embedding']['atmo'] = char2vec

            if kmers>0:
                print('training k-amino chars...')
                amDoc = [pSeq+['<EOS>'] for pSeq in self.pSeqData_k]
                model = Word2Vec(amDoc, min_count=0, window=window, size=amSize, workers=workers, sg=sg, iter=10)
                char2vec = np.zeros((self.kamNum, amSize), dtype=np.float32)
                for i in range(self.kamNum):
                    if self.id2kam[i] in model.wv:
                        char2vec[i] = model.wv[self.id2kam[i]]
                    else:
                        print(f'{self.id2kam[i]} not in vocab, random initialize it...')
                        char2vec[i] = np.random.random((1,amSize))
                self.vector['embedding']['kamino'] = char2vec

                print('training k-atmo chars...')
                atDoc = [dSeq+['<EOS>'] for dSeq in self.dSeqData_k]
                model = Word2Vec(atDoc, min_count=0, window=window, size=atSize, workers=workers, sg=sg, iter=10)
                char2vec = np.zeros((self.katNum, atSize), dtype=np.float32)
                for i in range(self.katNum):
                    if self.id2kat[i] in model.wv:
                        char2vec[i] = model.wv[self.id2kat[i]]
                    else:
                        print(f"{self.id2kat[i]} not in vocab, random initialize it...")
                        char2vec[i] = np.random.random((1,atSize))
                self.vector['embedding']['katmo'] = char2vec

            with open(path, 'wb') as f:
                pickle.dump(self.vector['embedding'], f, protocol=4)


    def random_batch_data_stream(self, batchSize=32, type='train', sampleType='PWRL', device=torch.device('cpu'), log=False):
        if sampleType=='PWRL':
            edges = self.edgeLab[type]
            nameData = self.pNameData[type]+self.dNameData[type]
            while True:
                random.shuffle(nameData)
                for i in range((len(nameData)+batchSize-1)//batchSize):
                    sampleNames = nameData[i*batchSize:(i+1)*batchSize]
                    #pTokenizedNames,dTokenizedNames = [self.p2id[i] for i in sampleNames if i.startswith('p_')],[self.d2id[i] for i in sampleNames if i.startswith('d_')]
                    pTokenizedNames,dTokenizedNames = [self.p2id[i] for i in random.sample(self.pNameData[type],batchSize)],[self.d2id[i] for i in random.sample(self.dNameData[type],batchSize*20)]
                    tmp = edges[pTokenizedNames,:][:,dTokenizedNames]
                    validP,validD = ((tmp==1).sum(axis=1)>0) & ((tmp==0).sum(axis=1)>0),(tmp>=0).sum(axis=0)>0
                    pTokenizedNames,dTokenizedNames = np.array(pTokenizedNames)[validP].tolist(),np.array(dTokenizedNames)[validD].tolist()

                    if len(pTokenizedNames)==0 or len(dTokenizedNames)==0: 
                        if log:
                            continue
                        yield {"res":False}, None
                    else:
                        yield {
                                "res":True, \
                                "aminoSeq":torch.tensor(self.pSeqTokenized[pTokenizedNames], dtype=torch.long).to(device), \
                                "kaminoSeq":torch.tensor(self.pSeqTokenized_k[pTokenizedNames], dtype=torch.long).to(device), \
                                "aminoPSSM":torch.tensor(self.pPSSMFeat[pTokenizedNames], dtype=torch.float32).to(device), \
                                "aminoCtr":torch.tensor(self.pContFeat[pTokenizedNames], dtype=torch.float32).to(device), \
                                "goSeq":torch.tensor(self.gSeqTokenized[pTokenizedNames], dtype=torch.long).to(device), \
                                "pSeqLen":torch.tensor(self.pSeqLen[pTokenizedNames], dtype=torch.int32).to(device), \
                                "atomFea":torch.tensor(self.dGraphFeat[dTokenizedNames], dtype=torch.float32).to(device), \
                                "atomFin":torch.tensor(self.dFinprFeat[dTokenizedNames], dtype=torch.float32).to(device), \
                                "atomCtr":torch.tensor(self.dContFeat[dTokenizedNames], dtype=torch.float32).to(device), \
                                "atomGra":torch.tensor(self.dAdjMat[dTokenizedNames], dtype=torch.float32).to(device), \
                                "atomSeq":torch.tensor(self.dSeqTokenized[dTokenizedNames], dtype=torch.long).to(device), \
                                "katomSeq":torch.tensor(self.dSeqTokenized_k[dTokenizedNames], dtype=torch.long).to(device), \
                                "dSeqLen":torch.tensor(self.dSeqLen[dTokenizedNames], dtype=torch.int32).to(device), \
                              }, torch.tensor(edges[pTokenizedNames,:][:,dTokenizedNames], dtype=torch.long).to(device)
        elif sampleType=='CEL':
            edges = [i for i in self.eSeqData[type]]
            while True:
                random.shuffle(edges)
                for i in range((len(edges)+batchSize-1)//batchSize):
                    samples = edges[i*batchSize:(i+1)*batchSize]
                    pTokenizedNames,dTokenizedNames = [i[0] for i in samples],[i[1] for i in samples]

                    yield {
                            "res":True, \
                            "aminoSeq":torch.tensor(self.pSeqTokenized[pTokenizedNames], dtype=torch.long).to(device), \
                            "kaminoSeq":torch.tensor(self.pSeqTokenized_k[pTokenizedNames], dtype=torch.long).to(device), \
                            "aminoPSSM":torch.tensor(self.pPSSMFeat[pTokenizedNames], dtype=torch.float32).to(device), \
                            "aminoCtr":torch.tensor(self.pContFeat[pTokenizedNames], dtype=torch.float32).to(device), \
                            "goSeq":torch.tensor(self.gSeqTokenized[pTokenizedNames], dtype=torch.long).to(device), \
                            "pSeqLen":torch.tensor(self.pSeqLen[pTokenizedNames], dtype=torch.int32).to(device), \
                            "atomFea":torch.tensor(self.dGraphFeat[dTokenizedNames], dtype=torch.float32).to(device), \
                            "atomFin":torch.tensor(self.dFinprFeat[dTokenizedNames], dtype=torch.float32).to(device), \
                            "atomCtr":torch.tensor(self.dContFeat[dTokenizedNames], dtype=torch.float32).to(device), \
                            "atomGra":torch.tensor(self.dAdjMat[dTokenizedNames], dtype=torch.float32).to(device), \
                            "atomSeq":torch.tensor(self.dSeqTokenized[dTokenizedNames], dtype=torch.long).to(device), \
                            "katomSeq":torch.tensor(self.dSeqTokenized_k[dTokenizedNames], dtype=torch.long).to(device), \
                            "dSeqLen":torch.tensor(self.dSeqLen[dTokenizedNames], dtype=torch.int32).to(device), \
                          }, torch.tensor([i[2] for i in samples], dtype=torch.float32).to(device)

    
    def one_epoch_batch_data_stream(self, batchSize=32, type='valid', mode='predict', device=torch.device('cpu')):
        if mode=='train':
            edges = self.edgeLab[type]
            nameData = self.pNameData[type]+self.dNameData[type]
            for i in range((len(nameData)+batchSize-1)//batchSize):
                sampleNames = nameData[i*batchSize:(i+1)*batchSize]
                pTokenizedNames,dTokenizedNames = [self.p2id[i] for i in sampleNames if i.startswith('p_')],[self.d2id[i] for i in sampleNames if i.startswith('d_')]
                
                if len(pTokenizedNames)==0 or len(dTokenizedNames)==0: 
                    yield {"res":False}, None
                else:
                    yield {
                            "res":True, \
                            "aminoSeq":torch.tensor(self.pSeqTokenized[pTokenizedNames], dtype=torch.long).to(device), \
                            "kaminoSeq":torch.tensor(self.pSeqTokenized_k[pTokenizedNames], dtype=torch.long).to(device), \
                            "aminoPSSM":torch.tensor(self.pPSSMFeat[pTokenizedNames], dtype=torch.float32).to(device), \
                            "aminoCtr":torch.tensor(self.pContFeat[pTokenizedNames], dtype=torch.float32).to(device), \
                            "goSeq":torch.tensor(self.gSeqTokenized[pTokenizedNames], dtype=torch.long).to(device), \
                            "pSeqLen":torch.tensor(self.pSeqLen[pTokenizedNames], dtype=torch.int32).to(device), \
                            "atomFea":torch.tensor(self.dGraphFeat[dTokenizedNames], dtype=torch.float32).to(device), \
                            "atomFin":torch.tensor(self.dFinprFeat[dTokenizedNames], dtype=torch.float32).to(device), \
                            "atomCtr":torch.tensor(self.dContFeat[dTokenizedNames], dtype=torch.float32).to(device), \
                            "atomGra":torch.tensor(self.dAdjMat[dTokenizedNames], dtype=torch.float32).to(device), \
                            "atomSeq":torch.tensor(self.dSeqTokenized[dTokenizedNames], dtype=torch.long).to(device), \
                            "katomSeq":torch.tensor(self.dSeqTokenized_k[dTokenizedNames], dtype=torch.long).to(device), \
                            "dSeqLen":torch.tensor(self.dSeqLen[dTokenizedNames], dtype=torch.int32).to(device), \
                          }, torch.tensor(edges[pTokenizedNames,:][:,dTokenizedNames], dtype=torch.long).to(device)

        elif mode=='predict':
            edges = self.eSeqData[type]
            # random.shuffle(edges)
            for i in range((len(edges)+batchSize-1)//batchSize):
                samples = edges[i*batchSize:(i+1)*batchSize]
                pTokenizedNames,dTokenizedNames = [i[0] for i in samples],[i[1] for i in samples]

                yield {
                        "res":True, \
                        "aminoSeq":torch.tensor(self.pSeqTokenized[pTokenizedNames], dtype=torch.long).to(device), \
                        "kaminoSeq":torch.tensor(self.pSeqTokenized_k[pTokenizedNames], dtype=torch.long).to(device), \
                        "aminoPSSM":torch.tensor(self.pPSSMFeat[pTokenizedNames], dtype=torch.float32).to(device), \
                        "aminoCtr":torch.tensor(self.pContFeat[pTokenizedNames], dtype=torch.float32).to(device), \
                        "goSeq":torch.tensor(self.gSeqTokenized[pTokenizedNames], dtype=torch.long).to(device), \
                        "pSeqLen":torch.tensor(self.pSeqLen[pTokenizedNames], dtype=torch.int32).to(device), \
                        "atomFea":torch.tensor(self.dGraphFeat[dTokenizedNames], dtype=torch.float32).to(device), \
                        "atomFin":torch.tensor(self.dFinprFeat[dTokenizedNames], dtype=torch.float32).to(device), \
                        "atomCtr":torch.tensor(self.dContFeat[dTokenizedNames], dtype=torch.float32).to(device), \
                        "atomGra":torch.tensor(self.dAdjMat[dTokenizedNames], dtype=torch.float32).to(device), \
                        "atomSeq":torch.tensor(self.dSeqTokenized[dTokenizedNames], dtype=torch.long).to(device), \
                        "katomSeq":torch.tensor(self.dSeqTokenized_k[dTokenizedNames], dtype=torch.long).to(device), \
                        "dSeqLen":torch.tensor(self.dSeqLen[dTokenizedNames], dtype=torch.int32).to(device), \
                      }, torch.tensor([i[2] for i in samples], dtype=torch.float32).to(device)


class DataClass_normal:
    def __init__(self, dataPath, pSeqMaxLen=1024, dSeqMaxLen=128, kmers=-1, validSize=0.2, sep=' '):
        # Open files and load data
        print('Loading the raw data...')
        self.p2id,self.id2p = {},[]
        self.d2id,self.id2d = {},[]
        pCnt,dCnt = 0,0
        pSeqData,pPSSMData,gSeqData,dMolData,dSeqData,dFeaData,dFinData = [],[],[],[],[],[],[]
        eSeqData = []

        atomFeaturizer = graph_features.WeaveFeaturizer()

        path = os.path.join(dataPath)
        with open(path, 'r') as f:
            while True:
                line = f.readline()
                if line=='':
                    break
                drug,protein,lab = line.strip().split(sep)
                
                if protein not in self.p2id:
                    pSeqData.append( protein )
                    self.p2id[protein] = pCnt
                    self.id2p.append(protein)
                    pCnt += 1
                if drug not in self.d2id:
                    mol = Chem.MolFromSmiles(drug)
                    if mol is None: continue
                    dSeqData.append( [a.GetSymbol() for a in mol.GetAtoms()] )
                    dMolData.append( mol )
                    dFeaData.append( [graph_features.atom_features(a) for a in mol.GetAtoms()] )

                    tmp = np.ones((1,))
                    DataStructs.ConvertToNumpyArray(AllChem.GetHashedMorganFingerprint(mol,2,nBits=1024), tmp)

                    dFinData.append( tmp )

                    self.d2id[drug] = dCnt
                    self.id2d.append(drug)
                    dCnt += 1

                eSeqData.append( [self.p2id[protein], self.d2id[drug], lab] )

        # Get the mapping variables
        print('Getting the mapping variables......')
        self.am2id,self.id2am = {"<UNK>":0,"<EOS>":1},["<UNK>","<EOS>"]
        amCnt = 2
        for pSeq in tqdm(pSeqData):
            for am in pSeq:
                if am not in self.am2id:
                    self.am2id[am] = amCnt
                    self.id2am.append(am)
                    amCnt += 1
        self.amNum = amCnt

        self.at2id,self.id2at = {"<UNK>":0,"<EOS>":1},["<UNK>","<EOS>"]
        atCnt = 2
        for dSeq in tqdm(dSeqData):
            for at in dSeq:
                if at not in self.at2id:
                    self.at2id[at] = atCnt
                    self.id2at.append(at)
                    atCnt += 1
        self.atNum = atCnt

        print('Tokenizing the data...')
        # Tokenized protein data
        pSeqTokenized = []
        pSeqLen = []
        for pSeq in tqdm(pSeqData):
            pSeq = [self.am2id[am] for am in pSeq]
            pSeqLen.append( min(len(pSeq),pSeqMaxLen) )
            pSeqTokenized.append( pSeq[:pSeqMaxLen] + [1]*max(pSeqMaxLen-len(pSeq),0) )

        # Tokenized drug data
        dSeqTokenized = []
        dSeqLen = []
        for dSeq in tqdm(dSeqData):
            atoms = [self.at2id[i] for i in dSeq]
            dSeqLen.append( min(len(dSeq),dSeqMaxLen) )
            dSeqTokenized.append( atoms[:dSeqMaxLen] + [1]*max(dSeqMaxLen-len(atoms),0) )

        # Finish
        print('Finished...')
        self.trainIdList,self.validIdList = train_test_split(range(len(eSeqData)), test_size=validSize)
        self.testIdList = []
        self.trainSampleNum,self.validSampleNum,self.testSampleNum = len(self.trainIdList),len(self.validIdList),len(self.testIdList)

        self.pSeqMaxLen,self.dSeqMaxLen = pSeqMaxLen,dSeqMaxLen
        self.pSeqData = pSeqData
        self.dSeqData,self.dMolData = dSeqData,dMolData
        self.pSeqLen,self.dSeqLen = np.array(pSeqLen, dtype=np.int32),np.array(dSeqLen, dtype=np.int32)
        self.pSeqTokenized = np.array(pSeqTokenized,dtype=np.int32)

        ctr = CountVectorizer(ngram_range=(1, 3), analyzer='char')
        pContFeat = ctr.fit_transform([''.join(i) for i in self.pSeqData]).toarray().astype('float32')
        k1,k2,k3 = [len(i)==1 for i in ctr.get_feature_names()],[len(i)==2 for i in ctr.get_feature_names()],[len(i)==3 for i in ctr.get_feature_names()]
        
        pContFeat[:,k1] = (pContFeat[:,k1] - pContFeat[:,k1].mean(axis=1).reshape(-1,1))/(pContFeat[:,k1].std(axis=1).reshape(-1,1)+1e-8)
        pContFeat[:,k2] = (pContFeat[:,k2] - pContFeat[:,k2].mean(axis=1).reshape(-1,1))/(pContFeat[:,k2].std(axis=1).reshape(-1,1)+1e-8)
        pContFeat[:,k3] = (pContFeat[:,k3] - pContFeat[:,k3].mean(axis=1).reshape(-1,1))/(pContFeat[:,k3].std(axis=1).reshape(-1,1)+1e-8)
        mean,std = pContFeat.mean(axis=0),pContFeat.std(axis=0)+1e-8
        pContFeat = (pContFeat-mean) / std
        self.pContFeatVectorizer = {'transformer':ctr, 
                                    'mean':mean, 'std':std}
        self.pContFeat = pContFeat

        self.dSeqTokenized = np.array(dSeqTokenized,dtype=np.int32)
        self.dGraphFeat = np.array([i[:dSeqMaxLen]+[[0]*75]*(dSeqMaxLen-len(i)) for i in dFeaData], dtype=np.int8)
        self.dFinprFeat = np.array(dFinData, dtype=np.float32)
        self.eSeqData = np.array(eSeqData, dtype=np.int32)
        self.vector = {}

    def describe(self):
        pass
    def change_seed(self, seed, validSize=0.2):
        self.trainIdList,self.validIdList = train_test_split(range(len(self.eSeqData)), test_size=validSize, random_state=seed)
        self.testIdList = []
        self.trainSampleNum,self.validSampleNum,self.testSampleNum = len(self.trainIdList),len(self.validIdList),len(self.testIdList)

    # creative_id, ad_id, product_id, product_category, advertiser_id, industry
    def vectorize(self, method="char2vec", amSize=16, goSize=16, atSize=16, window=25, sg=1, kmers=-1,
                        workers=8, loadCache=True, pos=False, suf=''):
        if method == 'char2vec':
            path = f'cache/{method}_am{amSize}_go{goSize}_at{atSize}.pkl'
            if os.path.exists(path) and loadCache:
                with open(path, 'rb') as f:
                    self.vector['embedding'] = pickle.load(f)
                print(f'Loaded cache from {path}.')
                return
            self.vector['embedding'] = {}
            
            print('training amino chars...')
            amDoc = [pSeq+['<EOS>'] for pSeq in self.pSeqData]
            model = Word2Vec(amDoc, min_count=0, window=window, size=amSize, workers=workers, sg=sg, iter=10)
            char2vec = np.zeros((self.amNum, amSize), dtype=np.float32)
            for i in range(self.amNum):
                if self.id2am[i] in model.wv:
                    char2vec[i] = model.wv[self.id2am[i]]
                else:
                    print(f'{self.id2am[i]} not in vocab, random initialize it...')
                    char2vec[i] = np.random.random((1,amSize))
            self.vector['embedding']['amino'] = char2vec
            
            print('training atmo chars...')
            atDoc = [dSeq+['<EOS>'] for dSeq in self.dSeqData]
            model = Word2Vec(atDoc, min_count=0, window=window, size=atSize, workers=workers, sg=sg, iter=10)
            char2vec = np.zeros((self.atNum, atSize), dtype=np.float32)
            for i in range(self.atNum):
                if self.id2at[i] in model.wv:
                    char2vec[i] = model.wv[self.id2at[i]]
                else:
                    print(f"{self.id2at[i]} not in vocab, random initialize it...")
                    char2vec[i] = np.random.random((1,atSize))
            self.vector['embedding']['atmo'] = char2vec

            with open(path, 'wb') as f:
                pickle.dump(self.vector['embedding'], f, protocol=4)


    def random_batch_data_stream(self, batchSize=32, type='train', sampleType='PWRL', device=torch.device('cpu'), log=False):
        if sampleType=='PWRL':
            pass
        elif sampleType=='CEL':
            if type=='train':
                idList = list(self.trainIdList)
            elif type=='valid':
                idList = list(self.validIdList)
            else:
                idList = list(self.testIdList)
            while True:
                random.shuffle(idList)
                for i in range((len(idList)+batchSize-1)//batchSize):
                    samples = idList[i*batchSize:(i+1)*batchSize]
                    edges = self.eSeqData[samples]
                    pTokenizedNames,dTokenizedNames = [i[0] for i in edges],[i[1] for i in edges]
                    yield {
                            "res":True, \
                            "aminoSeq":torch.tensor(self.pSeqTokenized[pTokenizedNames], dtype=torch.long).to(device), \
                            "aminoCtr":torch.tensor(self.pContFeat[pTokenizedNames], dtype=torch.float32).to(device), \
                            "pSeqLen":torch.tensor(self.pSeqLen[pTokenizedNames], dtype=torch.int32).to(device), \
                            "atomFea":torch.tensor(self.dGraphFeat[dTokenizedNames], dtype=torch.float32).to(device), \
                            "atomFin":torch.tensor(self.dFinprFeat[dTokenizedNames], dtype=torch.float32).to(device), \
                            "atomSeq":torch.tensor(self.dSeqTokenized[dTokenizedNames], dtype=torch.long).to(device), \
                            "dSeqLen":torch.tensor(self.dSeqLen[dTokenizedNames], dtype=torch.int32).to(device), \
                          }, torch.tensor([i[2] for i in edges], dtype=torch.float32).to(device)

    
    def one_epoch_batch_data_stream(self, batchSize=32, type='valid', mode='predict', device=torch.device('cpu')):
        if mode=='train':
            pass

        elif mode=='predict':
            if type=='train':
                idList = list(self.trainIdList)
            elif type=='valid':
                idList = list(self.validIdList)
            else:
                idList = list(self.testIdList)
            for i in range((len(idList)+batchSize-1)//batchSize):
                samples = idList[i*batchSize:(i+1)*batchSize]
                edges = self.eSeqData[samples]
                pTokenizedNames,dTokenizedNames = [i[0] for i in edges],[i[1] for i in edges]

                yield {
                        "res":True, \
                        "aminoSeq":torch.tensor(self.pSeqTokenized[pTokenizedNames], dtype=torch.long).to(device), \
                        "aminoCtr":torch.tensor(self.pContFeat[pTokenizedNames], dtype=torch.float32).to(device), \
                        "pSeqLen":torch.tensor(self.pSeqLen[pTokenizedNames], dtype=torch.int32).to(device), \
                        "atomFea":torch.tensor(self.dGraphFeat[dTokenizedNames], dtype=torch.float32).to(device), \
                        "atomFin":torch.tensor(self.dFinprFeat[dTokenizedNames], dtype=torch.float32).to(device), \
                        "atomSeq":torch.tensor(self.dSeqTokenized[dTokenizedNames], dtype=torch.long).to(device), \
                        "dSeqLen":torch.tensor(self.dSeqLen[dTokenizedNames], dtype=torch.int32).to(device), \
                        }, torch.tensor([i[2] for i in edges], dtype=torch.float32).to(device)

    def new_one_epoch_batch_data_stream(self, dataPath, batchSize=32, mode='predict', sep=' ', device=torch.device('cpu'), cache=None):
        # Open files and load data
        print('Loading the raw data...')
        p2id,id2p = {},[]
        d2id,id2d = {},[]
        pCnt,dCnt = 0,0
        pSeqData,pPSSMData,gSeqData,dMolData,dSeqData,dFeaData,dFinData = [],[],[],[],[],[],[]
        eSeqData = []

        atomFeaturizer = graph_features.WeaveFeaturizer()

        if cache is not None and os.path.exists(cache):
            data = np.load(cache)
            pSeqTokenized = data['pSeqTokenized']
            pContFeat = data['pContFeat']
            pSeqLen = data['pSeqLen']
            dGraphFeat = data['dGraphFeat']
            dFinprFeat = data['dFinprFeat']
            dSeqTokenized = data['dSeqTokenized']
            dSeqLen = data['dSeqLen']
            eSeqData = data['eSeqData']
        else:
            path = os.path.join(dataPath)
            with open(path, 'r') as f:
                while True:
                    line = f.readline()
                    if line=='':
                        break
                    drug,protein,lab = line.strip().split(sep)
                    
                    if protein not in p2id:
                        pSeqData.append( protein )
                        p2id[protein] = pCnt
                        id2p.append(protein)
                        pCnt += 1
                    if drug not in d2id:
                        mol = Chem.MolFromSmiles(drug)
                        if mol is None: continue
                        dSeqData.append( [a.GetSymbol() for a in mol.GetAtoms()] )
                        dMolData.append( mol )
                        dFeaData.append( [graph_features.atom_features(a) for a in mol.GetAtoms()] )

                        tmp = np.ones((1,))
                        DataStructs.ConvertToNumpyArray(AllChem.GetHashedMorganFingerprint(mol,2,nBits=1024), tmp)

                        dFinData.append( tmp )

                        d2id[drug] = dCnt
                        id2d.append(drug)
                        dCnt += 1

                    eSeqData.append( [p2id[protein], d2id[drug], lab] )

            print('Tokenizing the data...')
            # Tokenized protein data
            pSeqTokenized = []
            pSeqLen = []
            for pSeq in tqdm(pSeqData):
                pSeq = [self.am2id[am] for am in pSeq]
                pSeqLen.append( min(len(pSeq),self.pSeqMaxLen) )
                pSeqTokenized.append( pSeq[:self.pSeqMaxLen] + [1]*max(self.pSeqMaxLen-len(pSeq),0) )

            # Tokenized drug data
            dSeqTokenized = []
            dSeqLen = []
            for dSeq in tqdm(dSeqData):
                atoms = [self.at2id[i] for i in dSeq if i in self.at2id]
                dSeqLen.append( min(len(dSeq),self.dSeqMaxLen) )
                dSeqTokenized.append( atoms[:self.dSeqMaxLen] + [1]*max(self.dSeqMaxLen-len(atoms),0) )

            # Finish
            print('Other solving...')
            del dSeqData,dMolData
            gc.collect()

            pSeqLen,dSeqLen = np.array(pSeqLen, dtype=np.int32),np.array(dSeqLen, dtype=np.int32)
            pSeqTokenized = np.array(pSeqTokenized,dtype=np.int32)

            ctr = self.pContFeatVectorizer['transformer']
            pContFeat = ctr.transform([''.join(i) for i in pSeqData]).toarray().astype('float32')
            k1,k2,k3 = [len(i)==1 for i in ctr.get_feature_names()],[len(i)==2 for i in ctr.get_feature_names()],[len(i)==3 for i in ctr.get_feature_names()]
            
            pContFeat[:,k1] = (pContFeat[:,k1] - pContFeat[:,k1].mean(axis=1).reshape(-1,1))/(pContFeat[:,k1].std(axis=1).reshape(-1,1)+1e-8)
            pContFeat[:,k2] = (pContFeat[:,k2] - pContFeat[:,k2].mean(axis=1).reshape(-1,1))/(pContFeat[:,k2].std(axis=1).reshape(-1,1)+1e-8)
            pContFeat[:,k3] = (pContFeat[:,k3] - pContFeat[:,k3].mean(axis=1).reshape(-1,1))/(pContFeat[:,k3].std(axis=1).reshape(-1,1)+1e-8)
            pContFeat = (pContFeat-self.pContFeatVectorizer['mean']) / self.pContFeatVectorizer['std']

            dSeqTokenized = np.array(dSeqTokenized,dtype=np.int32)
            dGraphFeat = np.array([i[:self.dSeqMaxLen]+[[0]*75]*(self.dSeqMaxLen-len(i)) for i in dFeaData], dtype=np.int8)
            dFinprFeat = np.array(dFinData, dtype=np.float32)
            eSeqData = np.array(eSeqData, dtype=np.int32)
        
            if cache is not None:
                np.savez(cache, pSeqTokenized=pSeqTokenized, pContFeat=pContFeat, pSeqLen=pSeqLen,
                                dGraphFeat=dGraphFeat, dFinprFeat=dFinprFeat, dSeqTokenized=dSeqTokenized, dSeqLen=dSeqLen,
                                eSeqData=eSeqData)

        print('Predicting...')
        if mode=='train':
            pass
        elif mode=='predict':
            idList = list(range(len(eSeqData)))
            for i in tqdm(range((len(idList)+batchSize-1)//batchSize)):
                samples = idList[i*batchSize:(i+1)*batchSize]
                edges = eSeqData[samples]
                pTokenizedNames,dTokenizedNames = [i[0] for i in edges],[i[1] for i in edges]

                yield {
                        "res":True, \
                        "aminoSeq":torch.tensor(pSeqTokenized[pTokenizedNames], dtype=torch.long).to(device), \
                        "aminoCtr":torch.tensor(pContFeat[pTokenizedNames], dtype=torch.float32).to(device), \
                        "pSeqLen":torch.tensor(pSeqLen[pTokenizedNames], dtype=torch.int32).to(device), \
                        "atomFea":torch.tensor(dGraphFeat[dTokenizedNames], dtype=torch.float32).to(device), \
                        "atomFin":torch.tensor(dFinprFeat[dTokenizedNames], dtype=torch.float32).to(device), \
                        "atomSeq":torch.tensor(dSeqTokenized[dTokenizedNames], dtype=torch.long).to(device), \
                        "dSeqLen":torch.tensor(dSeqLen[dTokenizedNames], dtype=torch.int32).to(device), \
                        }, torch.tensor([i[2] for i in edges], dtype=torch.float32).to(device)

    def single_data_stream(self, drug, protein, pSeqMaxLen=None, dSeqMaxLen=None, mode='predict', device=torch.device('cpu')):
        if pSeqMaxLen is None:
            pSeqMaxLen = self.pSeqMaxLen
        if dSeqMaxLen is None:
            dSeqMaxLen = self.dSeqMaxLen
        # Presolve the data
        print('Presolving the data...')
        pSeqData,dMolData,dSeqData,dFeaData,dFinData = [],[],[],[],[]

        atomFeaturizer = graph_features.WeaveFeaturizer()

        pSeqData.append( protein )
        
        mol = Chem.MolFromSmiles(drug)
        dSeqData.append( [a.GetSymbol() for a in mol.GetAtoms()] )
        dMolData.append( mol )
        dFeaData.append( [graph_features.atom_features(a) for a in mol.GetAtoms()] )
        tmp = np.ones((1,))
        DataStructs.ConvertToNumpyArray(AllChem.GetHashedMorganFingerprint(mol,2,nBits=1024), tmp)
        dFinData.append( tmp )

        print('Tokenizing the data...')
        # Tokenized protein data
        pSeqTokenized = []
        pSeqLen = []
        for pSeq in tqdm(pSeqData):
            pSeq = [self.am2id[am] for am in pSeq]
            pSeqLen.append( min(len(pSeq),pSeqMaxLen) )
            pSeqTokenized.append( pSeq[:pSeqMaxLen] + [1]*max(pSeqMaxLen-len(pSeq),0) )

        # Tokenized drug data
        dSeqTokenized = []
        dSeqLen = []
        for dSeq in tqdm(dSeqData):
            atoms = [self.at2id[i] for i in dSeq if i in self.at2id]
            dSeqLen.append( min(len(dSeq),dSeqMaxLen) )
            dSeqTokenized.append( atoms[:dSeqMaxLen] + [1]*max(dSeqMaxLen-len(atoms),0) )

        # Finish
        print('Other solving...')
        del dSeqData,dMolData
        gc.collect()

        pSeqLen,dSeqLen = np.array(pSeqLen, dtype=np.int32),np.array(dSeqLen, dtype=np.int32)
        pSeqTokenized = np.array(pSeqTokenized,dtype=np.int32)

        ctr = self.pContFeatVectorizer['transformer']
        pContFeat = ctr.transform([''.join(i) for i in pSeqData]).toarray().astype('float32')
        k1,k2,k3 = [len(i)==1 for i in ctr.get_feature_names()],[len(i)==2 for i in ctr.get_feature_names()],[len(i)==3 for i in ctr.get_feature_names()]
        
        pContFeat[:,k1] = (pContFeat[:,k1] - pContFeat[:,k1].mean(axis=1).reshape(-1,1))/(pContFeat[:,k1].std(axis=1).reshape(-1,1)+1e-8)
        pContFeat[:,k2] = (pContFeat[:,k2] - pContFeat[:,k2].mean(axis=1).reshape(-1,1))/(pContFeat[:,k2].std(axis=1).reshape(-1,1)+1e-8)
        pContFeat[:,k3] = (pContFeat[:,k3] - pContFeat[:,k3].mean(axis=1).reshape(-1,1))/(pContFeat[:,k3].std(axis=1).reshape(-1,1)+1e-8)
        pContFeat = (pContFeat-self.pContFeatVectorizer['mean']) / self.pContFeatVectorizer['std']

        dSeqTokenized = np.array(dSeqTokenized,dtype=np.int32)
        dGraphFeat = np.array([i[:dSeqMaxLen]+[[0]*75]*(dSeqMaxLen-len(i)) for i in dFeaData], dtype=np.int8)
        dFinprFeat = np.array(dFinData, dtype=np.float32)

        print('Predicting...')
        if mode=='train':
            pass
        elif mode=='predict':
            yield {
                    "res":True, \
                    "aminoSeq":torch.tensor(pSeqTokenized, dtype=torch.long).to(device), \
                    "aminoCtr":torch.tensor(pContFeat, dtype=torch.float32).to(device), \
                    "pSeqLen":torch.tensor(pSeqLen, dtype=torch.int32).to(device), \
                    "atomFea":torch.tensor(dGraphFeat, dtype=torch.float32).to(device), \
                    "atomFin":torch.tensor(dFinprFeat, dtype=torch.float32).to(device), \
                    "atomSeq":torch.tensor(dSeqTokenized, dtype=torch.long).to(device), \
                    "dSeqLen":torch.tensor(dSeqLen, dtype=torch.int32).to(device), \
                    }, torch.tensor([-1], dtype=torch.float32).to(device)