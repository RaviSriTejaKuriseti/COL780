import cv2
import numpy as np
import os
import sys
import time




class Panorama_Creation:

    def __init__(self,input_path,output_path):

        '''
        Create a panorama class-object which contains original frames,
        generated key-points,affine-transform matrices to middle frame
        and coordinates of final corners.

        '''

        self.frames=[]
        self.input_path=input_path

        if(not os.path.exists(output_path)):
            os.makedirs(output_path)

        self.output_path=output_path
        self.key_points=[]
        self.affine_matrices=None 

        X=[]
        for filename in os.listdir(input_path):
            if(filename.endswith(".png") or filename.endswith(".jpg") or filename.endswith(".jpeg")):
                s=filename.replace("."," ")
                S=s.strip().split()
                X.append((int(S[1]),filename))
        X.sort()

        for _,f in X:            
            img = cv2.imread(os.path.join(input_path,f))
            if img is not None:
                self.frames.append(img)
        
        self.generate_key_points()
        # self.get_transforms_to_mid()
        self.get_transforms_to_left()
        self.get_panorama_dimensions()


        
        


    def key_point_detection(self,image_rgb):
        '''
        Detects key points using Hessian
        '''

        image=cv2.cvtColor(image_rgb,cv2.COLOR_BGR2GRAY)


        (H,W)=image.shape

        print("Image Shape is: "+str((H,W)))

        image=cv2.GaussianBlur(image,(0, 0),1,1)

        ixx_kernel=0.5*np.array([[0,0,0],[1,-2,1],[0,0,0]])
        iyy_kernel=0.5*np.array([[0,1,0],[0,-2,0],[0,1,0]])
        ixy_kernel=0.25*np.array([[1,0,-1],[0,0,0],[-1,0,1]])

        image_xx=cv2.filter2D(src=image, ddepth=cv2.CV_32F, kernel=ixx_kernel)
        image_yy=cv2.filter2D(src=image, ddepth=cv2.CV_32F, kernel=iyy_kernel)
        image_xy=cv2.filter2D(src=image, ddepth=cv2.CV_32F, kernel=ixy_kernel)



        

        image_xx=cv2.GaussianBlur(image_xx,(0, 0),1,1)
        image_yy=cv2.GaussianBlur(image_yy,(0, 0),1,1)
        image_xy=cv2.GaussianBlur(image_xy,(0, 0),1,1)

        
        det_hessian_matrix=np.multiply(image_xx,image_yy)-np.multiply(image_xy,image_xy)
        trace_hessian_matrix=np.add(image_xx,image_yy)

        # response_matrix=np.divide(det_hessian_matrix,trace_hessian_matrix+10**(-6))

        M=np.multiply(trace_hessian_matrix,trace_hessian_matrix)

        k=0.05
        
        response_matrix=det_hessian_matrix-k*M

        Key_Points=set()
        threshold_value=max(0,0.01*np.amax(response_matrix))

        ct=0
        image_rgb_final=np.copy(image_rgb)
        start_time=time.time()

        d=10    
        for i in range(d,H-d):
            for j in range(d,W-d):
                if(response_matrix[i,j]>threshold_value):
                    x1=np.amax(response_matrix[i-d:i+d,j-d:j+d])
                    if(x1==response_matrix[i,j]):
                        Key_Points.add((i,j))
                        image_rgb_final[i,j]=np.array([0,0,255])
                        ct+=1
        end_time=time.time()
        print("Time taken to find the keypoints is: "+str(end_time-start_time)+" seconds")
        print("Number of keypoints are: "+str(ct))
        return Key_Points
    

    def generate_key_points(self):

        '''
        Generates all key-points of frames
        '''
        for i in range(0,len(self.frames)):
            T=self.key_point_detection(self.frames[i])
            self.key_points.append(T)
        return


    def proximity_matching(self,u,v):
        '''
        Produces key point matches by using least square distances
        '''
        KP1=self.key_points[u]
        KP2=self.key_points[v]

        img1=self.frames[u]
        img2=self.frames[v]

        D=[]
        src_points=[]
        dest_points=[]
        msd_threshold=10
        R=[]
        
        for (x1,y1) in KP1:            
            d_min=10**9
            x_min=-1
            y_min=-1
            for (x2,y2) in KP2:
                d=(x2-x1)**2+(y2-y1)**2
                if(d<d_min):
                    d_min=d
                    x_min=x2
                    y_min=y2

            T=img2[x_min-2:x_min+2,y_min-2:y_min+2,:]-img1[x1-2:x1+2,y1-2:y1+2,:]
            T=T**2
            msd=0.04*np.sum(T)
            R.append(msd)
            if(msd<=msd_threshold):
                src_points.append([y1,x1,1])
                dest_points.append([y_min,x_min,1])
                # src_points.append([y1,x1])
                # dest_points.append([y_min,x_min])
                D.append(([y1,x1],[y_min,x_min]))
            
        print("Number of matching key-points are: "+str(len(D)))
        # print("Minimum and maximum RMSD values are: "+str(min(R))+","+str(max(R)))

        src_points=np.array(src_points)
        dest_points=np.array(dest_points)
        return src_points,dest_points


    def affine_model(self,u,v):

        '''
        Generates the matching affine transform
        '''

        X,Y=self.proximity_matching(u,v)
        M=np.linalg.lstsq(X,Y,rcond=None)[0] #Affine transform from first to second.
        M=np.transpose(M)
        M[2]=np.array([0,0,1])
        # M=M[0:2]
        # print("Affine transform from frame "+str(u)+" to "+str(v)+" is:"+str(M))
        return M


    
    def get_transforms_to_left(self):

        '''
        Applies affine transforms to successive frames and 
        multiplies the transforms to get the affine transform 
        from ith frame to first frame.
        '''

        l=len(self.frames)        
        self.affine_matrices=[None]*l
        M=np.eye(3)
        p=0
        self.affine_matrices[p]=M
        print("Final transform for index "+str(p)+" is: "+str(M))

        for i in range(p+1,l):
            self.affine_matrices[i]=np.matmul(self.affine_matrices[i-1],self.affine_model(i,i-1))
            M=self.affine_matrices[i]
            print("Final transform for index "+str(i)+" is: "+str(M))
        return
    

    def remove_borders(self, img):
        '''
        Removes the black border of the image. 
        '''
        (h, w,_) = img.shape
        h_trimmed, w_trimmed = h, w
        for col in range(w - 1, -1, -1):
            all_black = True
            for i in range(h):
                if (np.amax(img[i, col]) > 0):
                    all_black = False
                    break
            if (all_black == True):
                w_trimmed = w_trimmed - 1
                
        for row in range(h - 1, -1, -1):
            all_black = True
            for i in range(w_trimmed):
                if (np.amax(img[row, i]) > 0):
                    all_black = False
                    break
            if (all_black == True):
                h_trimmed = h_trimmed - 1
        
        return img[:h_trimmed, :w_trimmed]
    
    
    def get_panorama_dimensions(self):

        '''
        Gives the estimate for the dimension of panorama
        '''
        Corners=[]
        for i in range(0,len(self.frames)):
            (H,W,_)=self.frames[i].shape
            corners=np.array([[0,0,1],[W-1,0,1],[W-1,H-1,1],[0,H-1,1]]).T
            corners_new_position=np.rint(np.matmul(self.affine_matrices[i],corners).T)
            # print("New corners after Transform for frame-"+str(i)+" "+str(corners_new_position))
            new_left=min(0,np.int32(np.amin(corners_new_position[:,:1])))
            new_right=max(W-1,np.int32(np.amax(corners_new_position[:,:1])))
            new_top=min(0,np.int32(np.amin(corners_new_position[:,1:])))
            new_bottom=max(H-1,np.int32(np.amax(corners_new_position[:,1:])))
            L=[new_left,new_right,new_top,new_bottom]
            # self.frames[i]=cv2.copyMakeBorder(self.frames[i], top=max(0,0-new_top), bottom=max(0,new_bottom-H), left=max(0,0-new_left), right=max(0,new_right-W), borderType=cv2.BORDER_CONSTANT) 
            Corners.append(L)
        
        Corners=np.array(Corners).astype(np.int32)
        self.final_corners=(np.amin(Corners[:,0]),np.amax(Corners[:,1]),np.amin(Corners[:,2]),np.amax(Corners[:,3]))
        print(self.final_corners)
        return
    

    def generate_panorama(self,img_name):

        '''
        Generates the panorama by taking logical-or of all frames
        afer getting warped.
        '''
        (w_min,w_max,h_min,h_max)=self.final_corners
        H=h_max-h_min+1
        W=w_max-w_min+1
        final_image=np.zeros((H,W,3)).astype(int)
        output_path=self.output_path
        X=[cv2.warpAffine(self.frames[i],self.affine_matrices[i][0:2],(W,H)) for i in range(len(self.frames))]
        for i,e in enumerate(X):
            cv2.imwrite(os.path.join(self.output_path,"warped_image_"+str(i)+".png"),e)
            final_image=np.bitwise_or(final_image,e.astype(int))
        final_image=self.remove_borders(final_image)
        print("Final Image Shape is:-"+str(final_image.shape))
        cv2.imwrite(os.path.join(output_path,img_name),final_image)
        return

    


###########################################
 

if __name__ == "__main__":
    
    image_name=sys.argv[1]
    input_path=sys.argv[2]
    output_path=sys.argv[3]
    S=time.time()
    P=Panorama_Creation(input_path,output_path)
    P.generate_panorama(image_name)
    E=time.time()
    print("Total Time taken is: "+str((E-S)/60)+" minutes")

   


   


