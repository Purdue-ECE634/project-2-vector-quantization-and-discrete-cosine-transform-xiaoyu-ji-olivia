import argparse
import os
import cv2
import numpy as np
from itertools import product
from matplotlib import pyplot as plt
from scipy.fftpack import dct


class Project2():
    def __init__(self,args):
        self.imdir=args.im_dir
        self.frame_num=args.frame_num
        self.quant_level=args.quant_level
        self.outdir=args.out_dir
        self.K=args.num_coeff

    def Lloyd_codeword_gen(self):
        list=sorted(os.listdir(self.imdir))
        self.train=np.array([])
        for i in range(self.frame_num):
            train=cv2.imread(self.imdir+"/"+list[i],0)
            if i==0:
                self.test=train
            if not self.train.any():
                self.train=train.reshape(-1,4,4)
            else:
                self.train=np.concatenate((self.train,train.reshape(-1,4,4)),0)
        self.l_train=len(self.train)

        ## initial reconstruction value
        self.recon=np.ones((self.quant_level,4,4))
        fmin,fmax=np.min(self.train,0),np.max(self.train,0)
        for v in range(self.quant_level):
            self.recon[v,:,:]=fmin+(fmax-fmin)*(v/(self.quant_level-1))
        self.Lloyd_partition_distortion()
            
        ### optimize
        T=0.05
        prev_dis=0
        while prev_dis==0 or abs(self.distortion-prev_dis)/prev_dis>=T:
            print("distortion: ", self.distortion)
            prev_dis=self.distortion
            self.recon=[self.recon_new[i]/self.recon_cnt[i] for i in range(self.quant_level)]
            self.Lloyd_partition_distortion()


    def Lloyd_partition_distortion(self):
        loss_t=np.ones(self.l_train)*np.inf
        self.recon_new=np.zeros((self.quant_level,4,4))
        self.recon_cnt=np.zeros(self.quant_level)+1e-6   ### avoid zero counts
        for i in range(self.l_train):
            opt=0
            t=self.train[i]
            for v in range(self.quant_level):
                d=np.sqrt(np.sum(np.square(t-self.recon[v])))
                if d<loss_t[i]:
                    loss_t[i]=d
                    opt=v
            self.recon_new[opt]+=t
            self.recon_cnt[opt]+=1
        self.distortion=np.mean(loss_t)


    def Lloyd_test(self):
        h,w=self.test.shape
        cv2.imwrite(self.outdir+"/origin.png",self.test)
        self.test=self.test.reshape(-1,4,4)
        self.l_test=len(self.test)
        self.reconstruct=[]
        for i in range(self.l_test):
            d=np.inf
            for l in range(self.quant_level):
                dl=np.sqrt(np.sum(np.square(self.test[i]-self.recon[l])))
                if dl<d:
                    d=dl
                    re=self.recon[l]
            self.reconstruct.append(re)
        self.reconstruct=np.array(self.reconstruct).reshape(h,w)
        cv2.imwrite(self.outdir+"/reconstruct_frame_"+str(self.frame_num)+"_quant_"+str(self.quant_level)+".png",self.reconstruct)

    def DCT(self,k=0):
        D = dct(np.eye(8), norm="ortho")
        if k!=0:
            self.K=k
        
        self.test=cv2.imread(self.imdir+"/"+sorted(os.listdir(self.imdir))[0],0)
        h,w=self.test.shape
        cv2.imwrite(self.outdir+"/origin.png",self.test)
        self.test=self.test.reshape(-1,8,8)
        l=len(self.test)
        self.coeff=np.einsum('ij,kjl->kil',np.transpose(D),self.test) 
        self.coeff=np.einsum('ijk,kl->ijl',self.coeff,D) 
        
        ### use only low frequency or pick both high&low frequency 
        ### low_freq:
        coeff=np.zeros((l,8,8))
        for i in range(self.K):
            coeff[:,i//8,i%8]=self.coeff[:,i//8,i%8]
        out=np.einsum('ij,kjl->kil',D,coeff)  
        out=np.einsum('ijk,kl->ijl',out,np.transpose(D)) 
        out=np.array(out).reshape(h,w)
        cv2.imwrite(self.outdir+"/DCT_recon_K_"+str(self.K)+"_lowf.png",out)

        ### low and middle
        coeff=np.zeros((l,8,8))
        for i in range(0,64,64//self.K):
            coeff[:,i//8,i%8]=self.coeff[:,i//8,i%8]
        out=np.einsum('ij,kjl->kil',D,coeff)  
        out=np.einsum('ijk,kl->ijl',out,np.transpose(D)) 
        out=np.array(out).reshape(h,w)
        cv2.imwrite(self.outdir+"/DCT_recon_K_"+str(self.K)+"_midf.png",out)

        ### zigzag
        coeff=np.zeros((l,8,8))
        for i in range(self.K):
            c,r=self.zigzag_index(i)
            coeff[:,r,c]=self.coeff[:,r,c]
        out=np.einsum('ij,kjl->kil',D,coeff)  
        out=np.einsum('ijk,kl->ijl',out,np.transpose(D)) 
        out=np.array(out).reshape(h,w)
        cv2.imwrite(self.outdir+"/DCT_recon_K_"+str(self.K)+"_zigzag.png",out)

    def zigzag_index(self,ndx, n=8):
        ### cite: https://stackoverflow.com/questions/53439212/index-to-coordinates-in-diagonal-zigzag-traverse
        if ndx < n*(n+1)//2:
            basecol = (int(np.sqrt(8 * ndx + 1)) - 1) // 2
            row = ndx -(basecol*(basecol+1)//2)
            col = basecol - row
        else:
            oldcol, oldrow = self.zigzag_index(n**2 - 1 - ndx, n)
            row = n - 1 - oldrow
            col = n - 1 - oldcol
        return col, row

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Video Motion Estimation')
    parser.add_argument('--im_dir', type=str, default="", 
                        help='input image folder directory')
    parser.add_argument('--question',type=tuple, default=2, 
                        help='question 1: qunatization; question 2: DCT')
    parser.add_argument('--frame_num',type=list, default=10, 
                        help='the number of images for Lloyd codeword training')
    parser.add_argument('--quant_level',type=list, default=128, 
                        help='quantization level for Lloyd codeword training')
    parser.add_argument('--out_dir',type=str,default="",
                        help="directory to save figures")
    parser.add_argument('--num_coeff',type=list, default=64, 
                        help='coefficient number of DCT: [2,4,8,16,32]')
    
    args=parser.parse_args()

    if not os.path.isdir(args.out_dir):
        os.mkdir(args.out_dir)

    proj=Project2(args)
    if args.question==1:
        proj.Lloyd_codeword_gen()
        proj.Lloyd_test()
    elif args.question==2:
        for K in [2,4,8,16,32]:
            proj.DCT(K)
        ### test specific K
        # proj.DCT()