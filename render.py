
import cv2
import numpy as np
import os
from math_functions import *


class Render_Cube:

    def __init__(self,input_path,output_path,intrinsic_params):

        '''
        Create a camera calibration class-object which takes captured images,
        as inputs and generates the intrinsic parameters matrix.
        '''

        self.intrinsic_params=intrinsic_params
        self.frames=[]
        self.input_path=input_path
        self.ct=0
        self.object_points=np.array([[i,j] for j in range (10) for i in range (7)])
        self.cube_corners=np.array([[0,0,0],[3,0,0],[3,3,0],[0,3,0],[0,0,-3],[3,0,-3],[3,3,-3],[0,3,-3]],dtype=np.float64)
        # self.cube_corners=np.array([[0,0,0],[2,0,0],[2,2,0],[0,2,0],[0,0,-2],[2,0,-2],[2,2,-2],[0,2,-2]],dtype=np.float64)
        

        if(not os.path.exists(output_path)):
            os.makedirs(output_path)

        self.output_path=output_path

        for filename in os.listdir(input_path):
            if(filename.endswith(".png") or filename.endswith(".jpg") or filename.endswith(".jpeg")):
                img = cv2.imread(os.path.join(input_path,filename))
                if img is not None:
                    self.frames.append(img)

    
    def draw_cube(self):

        '''
        Reads the frames to obtain corners and uses them to draw cube
        '''
        self.Corners=[]
       
        for i,img in enumerate(self.frames):
            frame = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(frame,(7,10),None)
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            if ret==False:
                print("Cannot detect points in Frame-"+str(i))
            else:
                corners = cv2.cornerSubPix(frame,corners,(11,11),(-1,-1),criteria)
                corners=corners.reshape(70,2)

                P=self.compute_extrinsic_params(corners)
                self.Corners.append(corners)

                img=self.render_cube(img,P)
                cv2.imwrite(os.path.join(self.output_path,"cube_"+str(i)+".png"),img)
                
  

    def compute_homography(self,corners):

        '''
        Computes homography from object to image space using DLT
        '''

        A=[]
        for i in range(0,len(corners)):
            (x1,y1)=self.object_points[i]
            (x2,y2)=corners[i]
            r1=np.array([0,0,0,-x1,-y1,-1,x1*y2,y1*y2,y2])
            r2=np.array([-x1,-y1,-1,0,0,0,x1*x2,x2*y1,x2])
            A.append(r1)
            A.append(r2)
        
        A=np.array(A)
        _,sigma,vt=np.linalg.svd(A)
        h=vt[np.argmin(sigma), :]
        M=h.reshape((3,3))
        return M
        
            
    def compute_extrinsic_params(self,corners):
        '''
        Using method from Zhang's paper to find R,t

        H=[h1,h2,h3]
        r1 = l*A-1h1
        r2 = l*A-1h2
        t = l*A-1h3
        r3 = r1 x r2

        l=1/||r1||=1/||r2||

        Closest orthogonal matrix R to matrix Q=USVT  is 
        given by R=UVT.

        '''
        H=self.compute_homography(corners)
        R=np.zeros((3,3))
        t=np.zeros((3,1))
        K=self.intrinsic_params
        V=np.matmul(np.linalg.inv(K),H)
        Q=V.view()

        r1=V[:,0]
        r2=V[:,1]
        t=V[:,2]
    

        l=0.5*(np.linalg.norm(r1,2)+np.linalg.norm(r2,2))
        r1=r1/l
        r2=r2/l
        t=t/l

        r3=np.cross(r1,r2)
        Q[:,2]=r3

        u,_,vt=np.linalg.svd(Q)
        R=np.array(np.matmul(u,vt))

        t=np.reshape(t,(3,1))

        Rt=np.concatenate((R,t),axis=1)
        P=np.array(np.matmul(K,Rt))
        
        return P
    

    
    def project_points(self,points,P):

        '''
        Projects points into image-plane from 3D-Space
        '''


        concat=np.array([1.0])
        
        dst_points=[]
        for i in range(0,len(points)):
            e=points[i]
            e=np.concatenate((e,concat))
            V=np.array(np.matmul(P,e))
            V=V/V[2]
            dst_points.append(V[0:2])
        return np.array(dst_points)

 

    def render_cube(self,img,P):

        '''
        Draw the cube by projecting the corners
        '''

        new_points=self.project_points(self.cube_corners,P)
        new_points=new_points.astype(np.int32)
        overlay=img.copy()

        face_color=(0,0,255)
        edge_color=(0,0,0)
        alpha = 0.25

        
        #Faces

        
        img=cv2.fillPoly(img, pts=[new_points[0:4,:]], color=face_color)
        img=cv2.fillPoly(img, pts=[new_points[4:8,:]], color=face_color)
        img=cv2.fillPoly(img, pts=[np.array([new_points[0],new_points[1],new_points[4],new_points[5]])], color=face_color)
        img=cv2.fillPoly(img, pts=[np.array([new_points[1],new_points[2],new_points[6],new_points[5]])], color=face_color)
        img=cv2.fillPoly(img, pts=[np.array([new_points[2],new_points[3],new_points[7],new_points[6]])], color=face_color)
        img=cv2.fillPoly(img, pts=[np.array([new_points[0],new_points[3],new_points[7],new_points[4]])], color=face_color)

        

        #Edges
        img = cv2.line(img, tuple(new_points[0]), tuple(new_points[1]), (255,255,0), 3)
        img = cv2.line(img, tuple(new_points[0]), tuple(new_points[4]), (0,255,255), 3)
        img = cv2.line(img, tuple(new_points[0]), tuple(new_points[3]), (255,125,255), 3)

        img = cv2.line(img, tuple(new_points[1]), tuple(new_points[2]), edge_color, 1)
        img = cv2.line(img, tuple(new_points[1]), tuple(new_points[5]), edge_color, 1)

        img = cv2.line(img, tuple(new_points[2]), tuple(new_points[6]), edge_color, 1)
        img = cv2.line(img, tuple(new_points[2]), tuple(new_points[3]), edge_color, 1)

        img = cv2.line(img, tuple(new_points[3]), tuple(new_points[7]), edge_color, 1)

        img = cv2.line(img, tuple(new_points[4]), tuple(new_points[5]), edge_color, 1)
        img = cv2.line(img, tuple(new_points[4]), tuple(new_points[7]), edge_color, 1)

        img = cv2.line(img, tuple(new_points[5]), tuple(new_points[6]), edge_color, 1)

        img = cv2.line(img, tuple(new_points[6]), tuple(new_points[7]), edge_color, 1)
        
        
        
        image_new = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)
        return image_new
