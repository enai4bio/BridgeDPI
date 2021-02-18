from torch import nn as nn
from torch.nn import functional as F
import torch,time,os,random
import numpy as np
from collections import OrderedDict

class TextSPP(nn.Module):
    def __init__(self, size=128, name='textSpp'):
        super(TextSPP, self).__init__()
        self.name = name
        self.spp = nn.AdaptiveAvgPool1d(size)
    def forward(self, x):
        return self.spp(x)

class TextSPP2(nn.Module):
    def __init__(self, size=128, name='textSpp2'):
        super(TextSPP2, self).__init__()
        self.name = name
        self.spp1 = nn.AdaptiveMaxPool1d(size)
        self.spp2 = nn.AdaptiveAvgPool1d(size)
    def forward(self, x):
        x1 = self.spp1(x).unsqueeze(dim=3) # => batchSize × feaSize × size × 1
        x2 = self.spp2(x).unsqueeze(dim=3) # => batchSize × feaSize × size × 1
        x3 = -self.spp1(-x).unsqueeze(dim=3) # => batchSize × feaSize × size × 1
        return torch.cat([x1,x2,x3], dim=3) # => batchSize × feaSize × size × 3

class TextEmbedding(nn.Module):
    def __init__(self, embedding, dropout=0.3, freeze=False, name='textEmbedding'):
        super(TextEmbedding, self).__init__()
        self.name = name
        self.embedding = nn.Embedding.from_pretrained(torch.tensor(embedding,dtype=torch.float32), freeze=freeze)
        self.dropout1 = nn.Dropout2d(p=dropout/2)
        self.dropout2 = nn.Dropout(p=dropout/2)
        self.p = dropout
    def forward(self, x):
        # x: batchSize × seqLen
        if self.p>0:
            x = self.dropout2(self.dropout1(self.embedding(x)))
        else:
            x = self.embedding(x)
        return x

class ResDilaCNNBlock(nn.Module):
    def __init__(self, dilaSize, filterSize=64, dropout=0.15, name='ResDilaCNNBlock'):
        super(ResDilaCNNBlock, self).__init__()
        self.layers = nn.Sequential(
                        nn.ELU(),
                        nn.Conv1d(filterSize,filterSize,kernel_size=3,padding=dilaSize,dilation=dilaSize),
                        nn.InstanceNorm1d(filterSize),
                        nn.ELU(),
                        nn.Dropout(dropout),
                        nn.Conv1d(filterSize,filterSize,kernel_size=3,padding=dilaSize,dilation=dilaSize),
                        nn.InstanceNorm1d(filterSize),
                    )
        self.name = name
    def forward(self, x):
        # x: batchSize × filterSize × seqLen
        return x + self.layers(x)

class ResDilaCNNBlocks(nn.Module):
    def __init__(self, feaSize, filterSize, blockNum=10, dilaSizeList=[1,2,4,8,16], dropout=0.15, name='ResDilaCNNBlocks'):
        super(ResDilaCNNBlocks, self).__init__()
        self.blockLayers = nn.Sequential()
        self.linear = nn.Linear(feaSize,filterSize)
        for i in range(blockNum):
            self.blockLayers.add_module(f"ResDilaCNNBlock{i}", ResDilaCNNBlock(dilaSizeList[i%len(dilaSizeList)],filterSize,dropout=dropout))
        self.name = name
    def forward(self, x):
        # x: batchSize × seqLen × feaSize
        x = self.linear(x) # => batchSize × seqLen × filterSize
        x = self.blockLayers(x.transpose(1,2)).transpose(1,2) # => batchSize × seqLen × filterSize
        return F.elu(x) # => batchSize × seqLen × filterSize

class BatchNorm1d(nn.Module):
    def __init__(self, inSize, name='batchNorm1d'):
        super(BatchNorm1d, self).__init__()
        self.bn = nn.BatchNorm1d(inSize)
        self.name = name
    def forward(self, x):
        return self.bn(x)

class TextCNN(nn.Module):
    def __init__(self, featureSize, filterSize, contextSizeList, reduction='pool', actFunc=nn.ReLU, bn=False, ln=False, name='textCNN'):
        super(TextCNN, self).__init__()
        moduleList = []
        bns,lns = [],[]
        for i in range(len(contextSizeList)):
            moduleList.append(
                nn.Conv1d(in_channels=featureSize, out_channels=filterSize, kernel_size=contextSizeList[i], padding=contextSizeList[i]//2),
            )
            bns.append(nn.BatchNorm1d(filterSize))
            lns.append(nn.LayerNorm(filterSize))
        if bn:
            self.bns = nn.ModuleList(bns)
        if ln:
            self.lns = nn.ModuleList(lns)
        self.actFunc = actFunc()
        self.conv1dList = nn.ModuleList(moduleList)
        self.reduction = reduction
        self.batcnNorm = nn.BatchNorm1d(filterSize)
        self.bn = bn
        self.ln = ln
        self.name = name
    def forward(self, x):
        # x: batchSize × seqLen × feaSize
        x = x.transpose(1,2) # => batchSize × feaSize × seqLen
        x = [conv(x).transpose(1,2) for conv in self.conv1dList] # => scaleNum * (batchSize × seqLen × filterSize)

        if self.bn:
            x = [b(i.transpose(1,2)).transpose(1,2) for b,i in zip(self.bns,x)]
        elif self.ln:
            x = [l(i) for l,i in zip(self.lns,x)]
        x = [self.actFunc(i) for i in x]

        if self.reduction=='pool':
            x = [F.adaptive_max_pool1d(i.transpose(1,2), 1).squeeze(dim=2) for i in x]
            return torch.cat(x, dim=1) # => batchSize × scaleNum*filterSize
        elif self.reduction=='None':
            return x # => scaleNum * (batchSize × seqLen × filterSize)

class TextLSTM(nn.Module):
    def __init__(self, feaSize, hiddenSize, num_layers=1, dropout=0.0, bidirectional=True, name='textBiLSTM'):
        super(TextLSTM, self).__init__()
        self.name = name
        self.biLSTM = nn.LSTM(feaSize, hiddenSize, bidirectional=bidirectional, batch_first=True, num_layers=num_layers, dropout=dropout)

    def forward(self, x, xlen=None):
        # x: batchSizeh × seqLen × feaSize
        if xlen is not None:
            xlen, indices = torch.sort(xlen, descending=True)
            _, desortedIndices = torch.sort(indices, descending=False)

            x = nn.utils.rnn.pack_padded_sequence(x[indices], xlen, batch_first=True)
        output, hn = self.biLSTM(x) # output: batchSize × seqLen × hiddenSize*2; hn: numLayers*2 × batchSize × hiddenSize
        if xlen is not None:
            output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
            return output[desortedIndices]
        return output # output: batchSize × seqLen × hiddenSize*2
    def orthogonalize_gate(self):
        nn.init.orthogonal_(self.biLSTM.weight_ih_l0)
        nn.init.orthogonal_(self.biLSTM.weight_hh_l0)
        nn.init.ones_(self.biLSTM.bias_ih_l0)
        nn.init.ones_(self.biLSTM.bias_hh_l0)

class TextGRU(nn.Module):
    def __init__(self, feaSize, hiddenSize, num_layers=1, dropout=0.0, bidirectional=True, name='textBiGRU'):
        super(TextGRU, self).__init__()
        self.name = name
        self.biGRU = nn.GRU(feaSize, hiddenSize, bidirectional=bidirectional, batch_first=True, num_layers=num_layers, dropout=dropout)

    def forward(self, x, xlen=None):
        # x: batchSizeh × seqLen × feaSize
        if xlen is not None:
            xlen, indices = torch.sort(xlen, descending=True)
            _, desortedIndices = torch.sort(indices, descending=False)

            x = nn.utils.rnn.pack_padded_sequence(x[indices], xlen, batch_first=True)
        output, hn = self.biGRU(x) # output: batchSize × seqLen × hiddenSize*2; hn: numLayers*2 × batchSize × hiddenSize
        if xlen is not None:
            output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
            return output[desortedIndices]

        return output # output: batchSize × seqLen × hiddenSize*2

class FastText(nn.Module):
    def __init__(self, feaSize, name='fastText'):
        super(FastText, self).__init__()
        self.name = name
    def forward(self, x, xLen):
        # x: batchSize × seqLen × feaSize; xLen: batchSize
        x = torch.sum(x, dim=1) / xLen.float().view(-1,1)
        return x

class MLP(nn.Module):
    def __init__(self, inSize, outSize, hiddenList=[], dropout=0.0, bnEveryLayer=False, dpEveryLayer=False, outBn=False, outAct=False, outDp=False, name='MLP', actFunc=nn.ReLU):
        super(MLP, self).__init__()
        self.name = name
        hiddens,bns = [],[]
        for i,os in enumerate(hiddenList):
            hiddens.append( nn.Sequential(
                nn.Linear(inSize, os),
            ) )
            bns.append(nn.BatchNorm1d(os))
            inSize = os
        bns.append(nn.BatchNorm1d(outSize))
        self.actFunc = actFunc()
        self.dropout = nn.Dropout(p=dropout)
        self.hiddens = nn.ModuleList(hiddens)
        self.bns = nn.ModuleList(bns)
        self.out = nn.Linear(inSize, outSize)
        self.bnEveryLayer = bnEveryLayer
        self.dpEveryLayer = dpEveryLayer
        self.outBn = outBn
        self.outAct = outAct
        self.outDp = outDp
    def forward(self, x):
        for h,bn in zip(self.hiddens,self.bns):
            x = h(x)
            if self.bnEveryLayer:
                x = bn(x) if len(x.shape)==2 else bn(x.transpose(-1,-2)).transpose(-1,-2)
            x = self.actFunc(x)
            if self.dpEveryLayer:
                x = self.dropout(x)
        x = self.out(x)
        if self.outBn: x = self.bns[-1](x) if len(x.shape)==2 else self.bns[-1](x.transpose(-1,-2)).transpose(-1,-2)
        if self.outAct: x = self.actFunc(x)
        if self.outDp: x = self.dropout(x)
        return x

class GCN(nn.Module):
    def __init__(self, inSize, outSize, hiddenSizeList=[], dropout=0.0, bnEveryLayer=False, dpEveryLayer=False, outBn=False, outAct=False, outDp=False, resnet=False, name='GCN', actFunc=nn.ReLU):
        super(GCN, self).__init__()
        self.name = name
        hiddens,bns = [],[]
        for i,os in enumerate(hiddenSizeList):
            hiddens.append(nn.Sequential(
                nn.Linear(inSize, os),
            ) )
            bns.append(nn.BatchNorm1d(os))
            inSize = os
        bns.append(nn.BatchNorm1d(outSize))
        self.actFunc = actFunc()
        self.dropout = nn.Dropout(p=dropout)
        self.hiddens = nn.ModuleList(hiddens)
        self.bns = nn.ModuleList(bns)
        self.out = nn.Linear(inSize, outSize)
        self.bnEveryLayer = bnEveryLayer
        self.dpEveryLayer = dpEveryLayer
        self.outBn = outBn
        self.outAct = outAct
        self.outDp = outDp
        self.resnet = resnet
    def forward(self, x, L):
        # x: nodeNum × feaSize; L: batchSize × nodeNum × nodeNum
        for h,bn in zip(self.hiddens,self.bns):
            a = h(torch.matmul(L,x)) # => batchSize × nodeNum × os
            if self.bnEveryLayer:
                if len(L.shape)==3:
                    a = bn(a.transpose(1,2)).transpose(1,2)
                else:
                    a = bn(a)
            a = self.actFunc(a)
            if self.dpEveryLayer:
                a = self.dropout(a)
            if self.resnet and a.shape==x.shape:
                a += x
            x = a
        a = self.out(torch.matmul(L, x)) # => batchSize × nodeNum × outSize
        if self.outBn:
            if len(L.shape)==3:
                a = self.bns[-1](a.transpose(1,2)).transpose(1,2)
            else:
                a = self.bns[-1](a)
        if self.outAct: a = self.actFunc(a)
        if self.outDp: a = self.dropout(a)
        if self.resnet and a.shape==x.shape:
            a += x
        x = a
        return x

class TextAttention(nn.Module):
    def __init__(self, method, name='textAttention'):
        super(TextAttention, self).__init__()
        self.attn = LuongAttention(method)
        self.name = name
    def forward(self, sequence, reference):
        # sequence: batchSize × seqLen × feaSize; reference: batchSize × classNum × feaSize
        alpha = self.attn(reference, sequence) # => batchSize × classNum × seqLen
        return torch.matmul(alpha, sequence) # => batchSize × classNum × feaSize

class LuongAttention(nn.Module):
    def __init__(self, method):
        super(LuongAttention, self).__init__()
        self.method = method
    def dot_score(self, hidden, encoderOutput):
        # hidden: batchSize × classNum × hiddenSize; encoderOutput: batchSize × seq_len × hiddenSize
        return torch.matmul(encoderOutput, hidden.transpose(-1,-2)) # => batchSize × seq_len × classNum
    def forward(self, hidden, encoderOutput):
        attentionScore = self.dot_score(hidden, encoderOutput).transpose(-1,-2)
        # attentionScore: batchSize × classNum × seq_len
        return F.softmax(attentionScore, dim=-1) # => batchSize × classNum × seq_len

class SimpleAttention(nn.Module):
    def __init__(self, inSize, actFunc=nn.Tanh(), name='SimpleAttention'):
        super(SimpleAttention, self).__init__()
        self.name = name
        self.W = nn.Linear(inSize, int(inSize//2))
        self.U = nn.Linear(int(inSize//2), 1)
        self.actFunc = actFunc
    def forward(self, input):
        # input: batchSize × seqLen × inSize
        x = self.W(input) # => batchSize × seqLen × inSize//2
        H = self.actFunc(x) # => batchSize × seqLen × inSize//2
        alpha = F.softmax(self.U(H), dim=1) # => batchSize × seqLen × 1
        return self.actFunc( torch.matmul(input.transpose(1,2), alpha).squeeze(2) ) # => batchSize × inSize

class InterationAttention(nn.Module):
    def __init__(self, feaSize1, feaSize2, dropout=0.0, attnType='poolAttn', name='interAttn'):
        super(InterationAttention, self).__init__()
        self.attnFunc = {'poolAttn':self.pooling_attention,
                         'poolAttn_s':self.pooling_attention_s,
                         'catSimAttn':self.concat_simple_attention,
                         'plaAttn':self.plane_attention,
                         'plaAttn_s':self.plane_attention_s}
        assert attnType in self.attnFunc.keys()
        self.name = name
        self.U = nn.Linear(feaSize1, feaSize2)
        self.W = nn.Linear(feaSize2, 1)
        self.simpleAttn1 = SimpleAttention(feaSize1+feaSize2)
        self.simpleAttn2 = SimpleAttention(feaSize1+feaSize2)
        self.feaSize1,self.feaSize2 = feaSize1,feaSize2
        self.attnType = attnType
        self.dropout = nn.Dropout(dropout)

    def pooling_attention_s(self, x, y):
        u = self.U(x).unsqueeze(dim=2) # => batchSize × seqLen1 × 1 × feaSize2
        v = y.unsqueeze(dim=1) # => batchSize × 1 × seqLen2 × feaSize2
        alpha = torch.sum(u*v,dim=3) # => batchSize × seqLen1 × seqLen2
        xAlpha,_ = torch.max(alpha, dim=2, keepdim=True) # => batchSize × seqLen1 × 1
        x = torch.matmul(x.transpose(1,2), F.softmax(xAlpha,dim=1)).squeeze(dim=2) # => batchSize × feaSize1
        yAlpha,_ = torch.max(alpha, dim=1, keepdim=True) # => batchSize × 1 × seqLen2
        y = torch.matmul(F.softmax(yAlpha,dim=2), y).squeeze(dim=1) # => batchSize × feaSize2
        return torch.cat([x,y], dim=1) # => batchSize × (feaSize1+feaSize2)

    def pooling_attention(self, x, y):
        u = self.U(x).unsqueeze(dim=2) # => batchSize × seqLen1 × 1 × feaSize2
        v = y.unsqueeze(dim=1) # => batchSize × 1 × seqLen2 × feaSize2
        alpha = F.tanh(u*v) # => batchSize × seqLen1 × seqLen2 × feaSize2
        alpha = self.W(alpha).squeeze(dim=3) # => batchSize × seqLen1 × seqLen2
        xAlpha,_ = torch.max(alpha, dim=2, keepdim=True) # => batchSize × seqLen1 × 1
        x = torch.matmul(x.transpose(1,2), F.softmax(xAlpha,dim=1)).squeeze(dim=2) # => batchSize × feaSize1
        yAlpha,_ = torch.max(alpha, dim=1, keepdim=True) # => batchSize × 1 × seqLen2
        y = torch.matmul(F.softmax(yAlpha,dim=2), y).squeeze(dim=1) # => batchSize × feaSize2
        return torch.cat([x,y], dim=1) # => batchSize × (feaSize1+feaSize2)

    def concat_simple_attention(self, x, y):
        x_pooled,_ = torch.max(x, dim=1) # => batchSize × feaSize1
        y_pooled,_ = torch.max(y, dim=1) # => batchSize × feaSize2
        u = torch.cat([x, y_pooled.unsqueeze(dim=1).expand(-1,x.shape[1],-1)], dim=2) # => batchSize × seqLen1 × (feaSize1+feaSize2)
        v = torch.cat([y, x_pooled.unsqueeze(dim=1).expand(-1,y.shape[1],-1)], dim=2) # => batchSize × seqLen2 × (feaSize1+feaSize2)
        x,y = self.simpleAttn1(u)[:,:self.feaSize1],self.simpleAttn2(v)[:,:self.feaSize2]
        return torch.cat([x,y], dim=1) # => batchSize × (feaSize1+feaSize2)

    def plane_attention_s(self, x, y):
        u = self.U(x).unsqueeze(dim=2) # => batchSize × seqLen1 × 1 × feaSize2
        v = y.unsqueeze(dim=1) # => batchSize × 1 × seqLen2 × feaSize2
        alpha = torch.sum(u*v,dim=3) # => batchSize × seqLen1 × seqLen2
        alpha = F.softmax(alpha.flatten(1,2),dim=1).unsqueeze(dim=1) # => batchSize × 1 × seqLen1*seqLen2

        x,y = x.unsqueeze(dim=2).expand(-1,-1,y.shape[1],-1),y.unsqueeze(dim=1).expand(-1,x.shape[1],-1,-1) # => batchSize × seqLen1 × seqLen2 × feaSize
        xy = torch.cat([x,y], dim=3).flatten(1,2) # => batchSize × seqLen1*seqLen2 × (feaSize1+feaSize2)
        return torch.matmul(alpha, xy).squeeze(dim=1) # => batchSize × (feaSize1+feaSize2)

    def plane_attention(self, x, y):
        u = self.U(x).unsqueeze(dim=2) # => batchSize × seqLen1 × 1 × feaSize2
        v = y.unsqueeze(dim=1) # => batchSize × 1 × seqLen2 × feaSize2
        alpha = F.tanh(u*v) # => batchSize × seqLen1 × seqLen2 × feaSize2
        alpha = self.W(alpha).squeeze(dim=3) # => batchSize × seqLen1 × seqLen2
        alpha = F.softmax(alpha.flatten(1,2),dim=1).unsqueeze(dim=1) # => batchSize × 1 × seqLen1*seqLen2

        x,y = x.unsqueeze(dim=2).expand(-1,-1,y.shape[1],-1),y.unsqueeze(dim=1).expand(-1,x.shape[1],-1,-1) # => batchSize × seqLen1 × seqLen2 × feaSize
        xy = torch.cat([x,y], dim=3).flatten(1,2) # => batchSize × seqLen1*seqLen2 × (feaSize1+feaSize2)
        return torch.matmul(alpha, xy).squeeze(dim=1) # => batchSize × (feaSize1+feaSize2)

    def forward(self, x, y):
        # x: batchSize × seqLen1 × feaSize1; y: batchSize × seqLen2 × feaSize2
        return self.dropout(self.attnFunc[self.attnType](x,y)) # => batchSize × (feaSize1+feaSize2)

class SelfAttention(nn.Module):
    def __init__(self, featureSize, dk, multiNum, name='selfAttn'):
        super(SelfAttention, self).__init__()
        self.dk = dk
        self.multiNum = multiNum
        self.WQ = nn.ModuleList([nn.Linear(featureSize, self.dk) for i in range(multiNum)])
        self.WK = nn.ModuleList([nn.Linear(featureSize, self.dk) for i in range(multiNum)])
        self.WV = nn.ModuleList([nn.Linear(featureSize, self.dk) for i in range(multiNum)])
        self.WO = nn.Linear(self.dk*multiNum, featureSize)
        self.name = name
    def forward(self, x, xlen=None):
        # x: batchSize × seqLen × feaSize; xlen: batchSize
        queries = [self.WQ[i](x) for i in range(self.multiNum)] # => multiNum*(batchSize × seqLen × dk)
        keys    = [self.WK[i](x) for i in range(self.multiNum)] # => multiNum*(batchSize × seqLen × dk)
        values  = [self.WV[i](x) for i in range(self.multiNum)] # => multiNum*(batchSize × seqLen × dk)
        scores  = [torch.bmm(queries[i], keys[i].transpose(1,2))/np.sqrt(self.dk) for i in range(self.multiNum)] # => multiNum*(batchSize × seqLen × seqLen)
        # mask <EOS> padding
        if xlen is not None:
            for i in range(len(scores)):
                mask = torch.zeros(scores[0].shape, dtype=torch.float32, device=scores[i].device) # => batchSize × seqLen × seqLen
                for j,k in enumerate(xlen):
                    mask[j,:,k-1:] -= 999999
                scores[i] = scores[i] + mask
        z = [torch.bmm(F.softmax(scores[i], dim=2), values[i]) for i in range(self.multiNum)] # => multiNum*(batchSize × seqLen × dk)
        z = self.WO(torch.cat(z, dim=2)) # => batchSize × seqLen × feaSize
        return z

class LayerNormAndDropout(nn.Module):
    def __init__(self, feaSize, dropout=0.1, name='layerNormAndDropout'):
        super(LayerNormAndDropout, self).__init__()
        self.layerNorm = nn.LayerNorm(feaSize)
        self.dropout = nn.Dropout(p=dropout)
        self.name = name
    def forward(self, x):
        return self.dropout(self.layerNorm(x))

class SimpleSelfAttention(nn.Module):
    def __init__(self, feaSize, name='simpleSelfAttn'):
        super(SimpleSelfAttention, self).__init__()
        self.feaSize = feaSize
        self.WO = nn.Linear(feaSize, feaSize)
        self.name = name
    def forward(self, x, xlen=None):
        # x: batchSize × seqLen × feaSize; xlen: batchSize
        querie = x # => batchSize × seqLen × feaSize
        key    = x # => batchSize × seqLen × feaSize
        value  = x # => batchSize × seqLen × feaSize
        score  = torch.bmm(querie, key.transpose(1,2))/np.sqrt(self.feaSize) # => batchSize × seqLen × seqLen
        # mask <EOS> padding
        if xlen is not None:
            mask = torch.zeros(score.shape, dtype=torch.float32, device=score.device) # => batchSize × seqLen × seqLen
            for j,k in enumerate(xlen):
                mask[j,:,k-1:] -= 999999
            score = score + mask
        z = torch.bmm(F.softmax(score, dim=2), value) # => batchSize × seqLen × feaSize
        z = self.WO(z) # => batchSize × seqLen × feaSize
        return z

class FFN(nn.Module):
    def __init__(self, featureSize, seqMaxLen, dropout=0.1, name='FFN'):
        super(FFN, self).__init__()
        self.layerNorm1 = nn.LayerNorm([seqMaxLen, featureSize])
        self.layerNorm2 = nn.LayerNorm([seqMaxLen, featureSize])
        self.Wffn = nn.Sequential(
                        nn.Linear(featureSize, featureSize*4), 
                        nn.ReLU(),
                        nn.Linear(featureSize*4, featureSize)
                    )
        self.dropout = nn.Dropout(p=dropout)
        self.name = name
    def forward(self, x, z):
        z = x + self.dropout(self.layerNorm1(z)) # => batchSize × seqLen × feaSize
        ffnx = self.Wffn(z) # => batchSize × seqLen × feaSize
        return z+self.dropout(self.layerNorm2(ffnx)) # => batchSize × seqLen × feaSize

class Transformer(nn.Module):
    def __init__(self, featureSize, dk, multiNum, seqMaxLen, dropout=0.1):
        super(Transformer, self).__init__()
        self.selfAttn = SelfAttention(featureSize, dk, multiNum)
        self.ffn = FFN(featureSize, seqMaxLen, dropout)

    def forward(self, input):
        x, xlen = input
        # x: batchSize × seqLen × feaSize; xlen: batchSize
        z = self.selfAttn(x, xlen) # => batchSize × seqLen × feaSize
        return (self.ffn(x, z),xlen) # => batchSize × seqLen × feaSize
        
class TextTransformer(nn.Module):
    def __init__(self, layersNum, featureSize, dk, multiNum, seqMaxLen, dropout=0.1, name='textTransformer'):
        super(TextTransformer, self).__init__()
        posEmb = [[np.sin(pos/10000**(2*i/featureSize)) if i%2==0 else np.cos(pos/10000**(2*i/featureSize)) for i in range(featureSize)] for pos in range(seqMaxLen)]
        self.posEmb = nn.Parameter(torch.tensor(posEmb, dtype=torch.float32), requires_grad=False) # seqLen × feaSize
        self.transformerLayers = nn.Sequential(
                                     OrderedDict(
                                         [('transformer%d'%i, Transformer(featureSize, dk, multiNum, seqMaxLen, dropout)) for i in range(layersNum)]
                                     )
                                 )
        self.dropout = nn.Dropout(p=dropout)
        self.name = name
    def forward(self, x, xlen=None):
        # x: batchSize × seqLen × feaSize; xlen: batchSize
        x = self.dropout(x+self.posEmb) # => batchSize × seqLen × feaSize
        return self.transformerLayers((x, xlen)) # => batchSize × seqLen × feaSize

class Transformer_Wcnn(nn.Module):
    def __init__(self, featureSize, dk, multiNum, dropout=0.1):
        super(Transformer_Wcnn, self).__init__()
        self.dk = dk
        self.multiNum = multiNum
        self.WQ = nn.ModuleList([nn.Linear(featureSize, self.dk) for i in range(multiNum)])
        self.WK = nn.ModuleList([nn.Linear(featureSize, self.dk) for i in range(multiNum)])
        self.WV = nn.ModuleList([nn.Linear(featureSize, self.dk) for i in range(multiNum)])
        self.WO = nn.Linear(self.dk*multiNum, featureSize)
        self.layerNorm1 = nn.LayerNorm(featureSize)
        self.layerNorm2 = nn.LayerNorm(featureSize)
        self.Wcnn = TextCNN(featureSize, featureSize, [1,3,5], reduction='None', actFunc=nn.ReLU(), name='Wffn_CNN')
        self.Wffn = nn.Sequential(
                        nn.Linear(featureSize*3, featureSize), 
                    )
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        # x: batchSize × seqLen × feaSize
        queries = [self.WQ[i](x) for i in range(self.multiNum)] # => multiNum*(batchSize × seqLen × dk)
        keys = [self.WK[i](x) for i in range(self.multiNum)] # => multiNum*(batchSize × seqLen × dk)
        values = [self.WQ[i](x) for i in range(self.multiNum)] # => multiNum*(batchSize × seqLen × dk)
        score = [torch.bmm(queries[i], keys[i].transpose(1,2))/np.sqrt(self.dk) for i in range(self.multiNum)] # => multiNum*(batchSize × seqLen × seqLen)
        z = [torch.bmm(F.softmax(score[i], dim=2), values[i]) for i in range(self.multiNum)] # => multiNum*(batchSize × seqLen × dk)
        z = self.WO(torch.cat(z, dim=2)) # => batchSize × seqLen × feaSize
        z = x + self.dropout(self.layerNorm1(z)) # => batchSize × seqLen × feaSize
        ffnx = torch.cat(self.Wcnn(z), dim=2) # => batchSize × seqLen × feaSize*3
        ffnx = self.Wffn(ffnx) # => batchSize × seqLen × feaSize
        return z+self.dropout(self.layerNorm2(ffnx)) # => batchSize × seqLen × feaSize

class TextTransformer_Wcnn(nn.Module):
    def __init__(self, layersNum, featureSize, dk, multiNum, dropout=0.1, name='textTransformer'):
        super(TextTransformer_Wcnn, self).__init__()
        #posEmb = [[np.sin(pos/10000**(2*i/featureSize)) if i%2==0 else np.cos(pos/10000**(2*i/featureSize)) for i in range(featureSize)] for pos in range(seqMaxLen)]
        #self.posEmb = nn.Parameter(torch.tensor(posEmb, dtype=torch.float32), requires_grad=False) # seqLen × feaSize
        self.transformerLayers = nn.Sequential(
                                        OrderedDict(
                                            [('transformer%d'%i, Transformer_Wcnn(featureSize, dk, multiNum, dropout)) for i in range(layersNum)]
                                        )
                                    )
        self.dropout = nn.Dropout(p=dropout)
        self.name = name
    def forward(self, x):
        # x: batchSize × seqLen × feaSize
        x = self.dropout(x) # => batchSize × seqLen × feaSize
        return self.transformerLayers(x) # => batchSize × seqLen × feaSize

class HierarchicalSoftmax(nn.Module):
    def __init__(self, inSize, hierarchicalStructure, lab2id, hiddenList1=[], hiddenList2=[], dropout=0.1, name='HierarchicalSoftmax'):
        super(HierarchicalSoftmax, self).__init__()
        self.name = name
        self.dropout = nn.Dropout(p=dropout)
        layers = nn.Sequential()
        for i,os in enumerate(hiddenList1):
            layers.add_module(str(i*2), nn.Linear(inSize, os))
            layers.add_module(str(i*2+1), nn.ReLU())
            inSize = os
        self.hiddenLayers1 = layers
        moduleList = [nn.Linear(inSize, len(hierarchicalStructure))]

        layers = nn.Sequential()
        for i,os in enumerate(hiddenList2):
            layers.add_module(str(i*2), nn.Linear(inSize, os))
            layers.add_module(str(i*2+1), nn.ReLU())
            inSize = os
        self.hiddenLayers2 = layers

        for i in hierarchicalStructure:
            moduleList.append( nn.Linear(inSize, len(i)) )
            for j in range(len(i)):
                i[j] = lab2id[i[j]]
        self.hierarchicalNum = [len(i) for i in hierarchicalStructure]
        self.restoreIndex = np.argsort(sum(hierarchicalStructure,[]))
        self.linearList = nn.ModuleList(moduleList)
    def forward(self, x):
        # x: batchSize × feaSize
        x = self.hiddenLayers1(x)
        x = self.dropout(x)
        y = [F.softmax(linear(x), dim=1) for linear in self.linearList[:1]]
        x = self.hiddenLayers2(x)
        y += [F.softmax(linear(x), dim=1) for linear in self.linearList[1:]]
        y = torch.cat([y[0][:,i-1].unsqueeze(1)*y[i] for i in range(1,len(y))], dim=1) # => batchSize × classNum
        return y[:,self.restoreIndex]

class FocalCrossEntropyLoss(nn.Module):
    def __init__(self, gama=2, weight=-1, logit=True):
        super(FocalCrossEntropyLoss, self).__init__()
        self.weight = torch.nn.Parameter(torch.tensor(weight, dtype=torch.float32), requires_grad=False)
        self.gama = gama
        self.logit = logit
    def forward(self, Y_pre, Y):
        if self.logit:
            Y_pre = F.softmax(Y_pre, dim=1)
        P = Y_pre[list(range(len(Y))), Y]
        if self.weight.shape!=torch.Size([]):
            w = self.weight[Y]
        else:
            w = torch.tensor([1.0 for i in range(len(Y))], device=self.weight.device)
        w = (w/w.sum()).reshape(-1)
        return (-w*((1-P)**self.gama * torch.log(P))).sum()

class ContinusCrossEntropyLoss(nn.Module):
    def __init__(self, gama=2):
        super(ContinusCrossEntropyLoss, self).__init__()
        self.gama = gama
    def forward(self, Y_logit, Y):
        Y_pre = F.softmax(Y_logit, dim=1)
        lab_pre = Y_pre.argmax(dim=1)
        P = Y_pre[list(range(len(Y))), Y]
        w = ((1+(lab_pre-Y).abs())**self.gama).float()
        w = (w/w.sum()).reshape(-1)
        return (-w*torch.log(P)).sum()

class PairWiseRankingLoss(nn.Module):
    def __init__(self, gama=1):
        super(PairWiseRankingLoss, self).__init__()
        self.gama = gama
    def forward(self, Y_logit, Y):
        # Y_logit, Y: batchSize1 × batchSize2;
        Y_pre = F.sigmoid(Y_logit)
        loss,cnt = 0,0
        for y_pre,y in zip(Y_pre,Y):
            # batchSize2
            neg = y_pre[y==0].unsqueeze(dim=1) # negNum × 1
            pos = y_pre[y==1].unsqueeze(dim=0) # 1 × posNum
            tmp = self.gama+(neg-pos) # => negNum × posNum
            tmp[tmp<0] = 0
            loss += tmp.sum()
            cnt += tmp.shape[0]*tmp.shape[1]
        return loss

'''
import torch
from nnLayer import *
Y = torch.tensor([0,2], dtype=torch.long)
Y_logit = torch.tensor([[0.1,0.9,1],[0.6,2,0.4]], dtype=torch.float32)
CCEL = ContinusCrossEntropyLoss()
CCEL(Y_logit, Y)
'''

class MultiTaskCEL(nn.Module):
    def __init__(self, lossBalanced=True, ageW=1, genderW=1, name='MTCEL'):
        super(MultiTaskCEL, self).__init__()
        self.genderCriterion,self.ageCriterion = nn.CrossEntropyLoss(),nn.CrossEntropyLoss()#ContinusCrossEntropyLoss()#
        self.genderS,self.ageS = nn.Parameter(torch.zeros(1,dtype=torch.float), requires_grad=lossBalanced),nn.Parameter(torch.zeros(1,dtype=torch.float), requires_grad=lossBalanced)
        self.lossBalanced = lossBalanced
        self.name = name
        self.ageW,self.genderW = ageW,genderW
    def forward(self, genderY_logit, genderY, ageY_logit, ageY):
        if self.lossBalanced:
            return self.genderW * torch.exp(-self.genderS) * self.genderCriterion(genderY_logit,genderY) + self.ageW * torch.exp(-self.ageS) * self.ageCriterion(ageY_logit,ageY) + (self.genderS+self.ageS)/2
        else:
            return self.genderW * self.genderCriterion(genderY_logit,genderY) + self.ageW * self.ageCriterion(ageY_logit,ageY)