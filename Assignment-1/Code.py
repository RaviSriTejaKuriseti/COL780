import cv2
import numpy as np
import os
import sys

def process_video(input_path,output_path,K):

    frames=[]    
    
    for filename in os.listdir(input_path):
        if(filename.endswith(".png") or filename.endswith(".jpg") or filename.endswith(".jpeg")):
            img = cv2.imread(os.path.join(input_path,filename))
            if img is not None:
                frames.append(img)


    print(len(frames))
    


    
    video=np.stack(frames,axis=0)  #TxHxWxC
    
    
    
    (T,H,W,C)=video.shape
    video_n=np.copy(video)
    bg_threshold=0.6

    
       
    Mean=np.zeros((H,W,K,3),dtype="float")
    Variance=np.empty((H,W,K),dtype="float")
    Variance.fill(225)
    #list of K Gaussians with mean and variance
    
    Weights=np.full((H,W,K),0.01)
    N=10


    for t in range (0,T):     
        for x in range(0,H):
            for y in range(0,W):            
                Prob=np.zeros(K,dtype="float")
                flag=False
                for i in range(0,K):
                    cur_pix=video[t][x][y]
                    mu=Mean[x][y][i]
                    var=Variance[x][y][i]

                    if(np.linalg.norm(mu-cur_pix,2)<=2.5*(var)**0.5 and flag==False):
                        flag=True
                        new_weight=Weights[x][y][i]+(1-Weights[x][y][i])/N
                        Weights[x][y][i]=new_weight
                        
                        new_mean=mu+(cur_pix/new_weight-mu)/N
                        score=np.linalg.norm(cur_pix-mu,2)
                        score*=score
                        new_var=var+(score/new_weight-var)/N
                        
                        Mean[x][y][i]=new_mean
                        Variance[x][y][i]=new_var
                                 
                    else:
                        Weights[x][y][i]*=(1-1/N)
                        Mean[x][y][i]*=(1-1/N)
                        Variance[x][y][i]*=(1-1/N)
                                 
                    
                    z=-1*(np.linalg.norm(cur_pix-mu,2))**2
                    z=z/(2*var)
                    z-=3*np.log(var)
                    z+=np.log(Weights[x][y][i])
                    Prob[i]=z
                   
                    

                if(flag==False):
                    j=np.argmin(Prob)
                    Mean[x][y][j]=cur_pix
                    Variance[x][y][j]=225

    
                L=np.array([(-1*Weights[x][y][j]/(Variance[x][y][j])**0.5) for j in range (K)])
                ordering=np.argsort(L)
                Weights[x][y]= Weights[x][y][ordering]
                Weights[x][y]/=np.linalg.norm(Weights[x][y],1)
                Mean[x][y]= Mean[x][y][ordering]
                Variance[x][y]= Variance[x][y][ordering]
                video_n[t][x][y]=np.array([255,255,255])
                
                s=0
                ind=-1
                for z in range(0,K):
                    s+=Weights[x][y][z]
                    if(s>bg_threshold):
                       ind=1+z
                       # print(ind)
                       break

                for z in range(0,ind):
                    cur_pix=video[t][x][y]
                    mu=Mean[x][y][z]
                    var=Variance[x][y][z]
                    if(np.linalg.norm(mu-cur_pix,2)<=2.5*(var)**0.5):
                        video_n[t][x][y]=np.array([0,0,0])
                        break

       
        cv2.imwrite(os.path.join(output_path,str(K)+"_"+str(bg_threshold)+"_"+str(t)+".png"), video_n[t])



    

    


                

     
    fourcc = cv2.VideoWriter_fourcc(*'MP42')
     
    video_new = cv2.VideoWriter(os.path.join(output_path,str(K)+"_"+str(bg_threshold)+"_"+"video.avi"), fourcc, 10.0, (W,H))
     
    for u in range(0,T):
        video_new.write(video_n[u])
         
    video_new.release()
    return



if __name__ == "__main__":

    # input_path="C:\\Users\\Lenovo\\Desktop\\COL780 Dataset\\IBMTest2\\input"
    # output_path="C:\\Users\\Lenovo\\Desktop\\COL780 Dataset\\IBMTest2\\output"
    # K=4

    input_path=sys.argv[1]
    output_path=sys.argv[2]
    K=int(sys.argv[3])
    process_video(input_path,output_path,K)