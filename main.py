import sys
from camera_calibration import Camera_Calibration
from render import Render_Cube



if __name__ == "__main__":

    input_path=sys.argv[1]
    output_path=sys.argv[2]


    Ob=Camera_Calibration(input_path,output_path)
    K=Ob.calibrate()
    Rc=Render_Cube(input_path,output_path,K)
    Rc.draw_cube()