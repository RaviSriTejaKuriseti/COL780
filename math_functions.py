import numpy as np


def compute_intersection_of_points(a,b,c,d):

    '''
    Computes intersection of line through points a,b with line through points c,d
    From link https://en.wikipedia.org/wiki/Line%E2%80%93line_intersection
    '''
    det_1=a[0]*b[1]-a[1]*b[0]
    det_2=c[0]*d[1]-c[1]*d[0]

    diff_x_1=a[0]-b[0]
    diff_y_1=a[1]-b[1]

    diff_x_2=c[0]-d[0]
    diff_y_2=c[1]-d[1]

    z=diff_x_1*diff_y_2-diff_x_2*diff_y_1

    x_in=(det_1*diff_x_2-det_2*diff_x_1)/z
    y_in=(det_1*diff_y_2-det_2*diff_y_1)/z

    return np.array([x_in,y_in])


def compute_intersection_of_lines(l1,l2):

    '''
    Computes intersection of lines y=m1x+c1 and y=m2x+c2
    l1=(m1,c1) and l2=(m2,c2)
    '''
    
    
    x_in=(l2[1]-l1[1])/(l1[0]-l2[0])
    y_in=(l1[0]*l2[1]-l1[1]*l2[0])/(l1[0]-l2[0])
    

    return np.array([x_in,y_in])


