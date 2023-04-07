import cv2
import numpy as np
import os
from math_functions import *


class Camera_Calibration:

    def __init__(self,input_path,output_path,skew=False):

        '''
        Create a camera calibration class-object which takes captured images,
        as inputs and generates the intrinsic parameters matrix.
        '''

        self.intrinsic_params=None
        self.frames=[]
        self.input_path=input_path
        self.ct=0

        if(not os.path.exists(output_path)):
            os.makedirs(output_path)

        self.output_path=output_path

        for filename in os.listdir(input_path):
            if(filename.endswith(".png") or filename.endswith(".jpg") or filename.endswith(".jpeg")):
                img = cv2.imread(os.path.join(input_path,filename))
                if img is not None:
                    self.frames.append(img)

        
        self.skew=skew
        self.compute_points()
        self.get_param_matrix()
                   



    def compute_points(self):

        '''
        Reads the frames to obtain corners
        '''
        self.Corners=[]
        self.Ideal_Points=[]

       
       
        for i,img in enumerate(self.frames):
            frame = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(frame,(7,10),None)
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            if ret==False:
                print("Cannot detect points in Frame-"+str(i))
            else:
                corners = cv2.cornerSubPix(frame,corners,(11,11),(-1,-1),criteria)
                img = cv2.drawChessboardCorners(img, (7,10), corners,ret)
                cv2.imwrite(os.path.join(self.output_path,"corners_"+str(i)+".png"),img)
                corners=corners.reshape(70,2)
                self.Corners.append(corners)
                ideal_points=self.obtain_ideal_points(corners)
                self.Ideal_Points.append(ideal_points)
                self.ct+=1
        
            

    def calibrate(self):
        if(self.ct>=2):
            self.intrinsic_params=self.compute_intrinsic_params()
            return self.intrinsic_params
        else:
            print("Insufficient number of frames for calibration: "+str(self.ct))
            return None

    

    def obtain_ideal_points(self,corners):

        '''
        Find the point of concurrency of lines to get the ideal-points.
        We will intersection of successive lines and compute their mean.
        Each line equation is found using least squares of each row and column
        '''


        a=corners[0]
        b=corners[6]
        c=corners[63]
        d=corners[69]

        P1=compute_intersection_of_points(a,b,c,d)
        P2=compute_intersection_of_points(a,d,b,c)


        return (P1,P2)
    

    
    def prepare_for_dlt(self):

        '''
        [d1,0],[d2,0] are ideal points along pair of perpendicular directions in object

        d1=(x1,x2,x3)
        d2=(y1,y2,y3)

        [x1*y1,x1*y2+x2*y1,x1*y3+x3*y1,x2*y2,x2*y3+x3*y2,x3*y3].[a,h,g,b,f,c]=0
        '''
        L=self.Ideal_Points
        if(self.skew==True):

            Z=np.zeros((len(L),6),dtype=np.float64)

            for i,(d1,d2) in enumerate(L):
                (x1,x2)=d1
                (y1,y2)=d2
                (x3,y3)=(1,1)
                Z[i]=np.array([x1*y1,x1*y2+x2*y1,x1*y3+x3*y1,x2*y2,x2*y3+x3*y2,x3*y3],dtype=np.float64)

            return Z
        
        else:
            '''
            Here h=0
            '''
                    
            Z=np.zeros((len(L),5),dtype=np.float64)

            for i,(d1,d2) in enumerate(L):
                (x1,x2)=d1
                (y1,y2)=d2
                (x3,y3)=(1,1)
                Z[i]=np.array([x1*y1,x1*y3+x3*y1,x2*y2,x2*y3+x3*y2,x3*y3],dtype=np.float64)

            return Z


    def get_param_matrix(self):

        '''
        Z is the matrix obtained by stacking points
        So [a h g b f c] is just the last orthogonal vector from SVD
        '''

        Z=self.prepare_for_dlt()
        
        _,sigma,vt=np.linalg.svd(Z)        
        J=vt[np.argmin(sigma), :]
        B=np.zeros((3,3))

        if(self.skew==True):

            B[0,0]=J[0]
            B[0,1]=J[1]
            B[0,2]=J[2]

            B[1,0]=B[0,1]
            B[1,1]=J[3]
            B[1,2]=J[4]

            B[2,0]=B[0,2]
            B[2,1]=B[1,2]
            B[2,2]=J[5]

        else:
            B[0,0]=J[0]
            B[0,2]=J[1]

            B[1,1]=J[2]
            B[1,2]=J[3]

            B[2,0]=B[0,2]
            B[2,1]=B[1,2]
            B[2,2]=J[4]

           

        try:
            _ = np.linalg.cholesky(B)
            self.param_matrix=B
            return B
        
        except np.linalg.LinAlgError as _:
            print("B is not a PSD Matrix")
            print("Trying with -B")

            try:
                B=-1*B
                _=np.linalg.cholesky(B)
                self.param_matrix=B
                return B
            except:
                print("-B is also not PSD")
                print("Hence the camera cannot be calibrated")
                self.param_matrix=None
                return None
            
     
    def compute_intrinsic_params(self):

        '''
        Use the formulae in Zhang's paper to compute intrinsic params
        '''


        B=self.param_matrix
        '''
        Use values from Zhang's Paper
        
        lambda_ is the scale parameter
        B=lambda_*K-TK-1
        K=[[fx,s,ux],[0,fy,uy],[0,0,1]]
        '''
        K=np.zeros((3,3))
        K[2,2]=1

        uy=(B[0,1]*B[0,2]-B[0,0]*B[1,2])/(B[0,0]*B[1,1]-B[0,1]*B[0,1])
        lambda_=B[2,2]-(B[0,2]*B[0,2]+uy*(B[0,1]*B[0,2]-B[0,0]*B[1,2]))/B[0,0]
        fx=np.sqrt(lambda_/B[0,0])
        fy=np.sqrt(lambda_*B[0,0]/(B[0,0]*B[1,1]-B[0,1]*B[0,1]))
        s=(-B[0,1]*fy*fx*fx)/lambda_
        ux=((s*uy)/(fy))-(B[0,2]*fx*fx)/lambda_

        K[0,0]=fx
        K[0,1]=s
        K[0,2]=ux
        K[1,1]=fy
        K[1,2]=uy
        return K
    






