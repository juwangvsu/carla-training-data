----- modify code to not use hdf5 file format for dataset ---
	arldell
	newamdpc

------------9/10/22 _out/calib/ lidar-cam caliberation fixed ----------------
LDLS works well with the new calib parameter below:
0000026.txt:
	parameters needed to project lidar points from lidar frame to camera plan
	P2 is the projection matrix
	Tr_velo_to_cam transform matrix from lidar frame to cam frame

	for lidar point p=(x,y,z,1) in lidar frame, p_c the same point
		at camera frame, p =(x_i, y_i) image plane coordinate 
	p_c = Tr_velo_to_cam * p

	p_2 = P2*p_c
	x_i = p_2[0]/p_2[2] # normalize diving by z
	y_i = p_2[1]/p_2[2]
	
	relationship between lidar frame and camera frame see kitti document,
	lidar frame:
			 z
			 | x
		     y __|/

	cam frame:a       z
			 /
			/____x
			|
			|
			y
	the original kitti caliberation param is not good for carla simulation data
	due to camera model difference. especially P2.
	The correct P2 is calculated from the camera parameter as following:
		f = 1238/2
		c_u = 1238/(2* tan(fov*pi/360)=619
		c_u = 384/(2* tan(fov*pi/360)=192
 
----------- 9/1,7/22 carla-traing-data migrate to 9.13 ------------------------
	use visualize_multiple_sensors2.py, ditch ccb 
python visualize_multiple_sensors2.py --host arldell --camera_resx 1238 

python client_bounding_boxes.py --host arldell --res 1024x400
python client_bounding_boxes.py --host i9lab-wifi --res 1024x400 --weather ClearNoon
	this now dump lidar data bin and pcl xyzrgb format.
 
manuel_control.py converting lot of issue. switching to use cbb.py
	lidar or camera one of them. if use both crash
manuel_control_ori.py
	
	spawn veh at map Town05 and a specific location to examine lidar data
	lidar point # 2000 ~ 7000 vary

python kitti_test.py _out/velodyne/00001657.bin testpcd_rec2.pcd
	convert kitti bin to pcd
pcl_viewer testpcd_rec2.pcd
	view the output pcd
rosrun icp_viewer icp_viewer *.pcd

default lidar setting: 56000 pt/s, 10 hz
	issue: some scan contain part of the 360 deg. average 3000 pts per pcd file
	fix: set FPS=10, python config.py --fps=10
		default 30 fps so get 1/3 of a full resolution since lidar run at 10hz
	--lidar_pps 560000 to increase # of lidar points by 10 times. much better resolution
ccb issue:
	can save both camera and lidar now, but camera frame rate is much lower than lidar
	also driving control very sluggish

Todo:
	ccb vehicle movement, hand craft human poses.
	lidar frame # and cam frame # not synced since the sensor trigger rate 
		are different. add a sync mode?

----------- 9.13 config.py ex ------------------------
config.py control game runtime stuff:

python config.py --host arldell --list
	this list options such as weather type
python config.py --host arldell --weather ClearNoon
	this change the game with ClearNoon weather
python config.py --host arldell --map Town03
	change the map of the curr game
python config.py --host arldell -i
	inspect current game info

python manual_control.py --host arldell --res 1238x374 -w 20 -n 0
	TAB switch cam position
	-h display actors info
	-w walker#
	-n veh#
	

----------------------8/18/22 python-pcl ----------------
using python-pcl to visualize the result:
	python-pcl install using the whl build from source
	pip install ./python_pcl-0.3.0rc1-cp37-cp37m-linux_x86_64.whl 
pcl_slideshow.py

----------------------8/9/22 kitti2bag ----------------
pip2 install kitti2bag
pyenv shell system
if still not work, modified to use python2 in the script

-----------------------7/27/22 LDLS test -------------------
forked from origin: https://github.com/juwangvsu/LDLS.git
demo.py:
	arldell out of gpu mem
	newpcamd works.
	need about 7gb gpu memory
	chart_studio.plotly also use 1.5 gb gpu memory. that is after lidarsegment run so won't impose additional gpu mem.

-----------------------7/27/22 map load 0.9.13 and 0.8.4 -------------------
use -windowed -ResX=600 -ResY=400 improve frame rate

0.9.13:
  ./CarlaUE4.sh -windowed -ResX=600 -ResY=400
  change startup map:
	vi CarlaUE4/Config/DefaultEngine.ini
	GameDefaultMap=/Game/Carla/Maps/Town02.Town02

  map, 
     python config.py -m Town05
        0.9.13 default load Town10HD
  weather
     Carla_9-13/PythonAPI/util$ python config.py --weather ClearNoon

  performance manuel drive:
	python manual_control.py
		R recording images.(_out/)
		P auto driving
	arldell/0.9.13/Town10HD, 10HD_Opt  3 fps
			Town04, 05 --- 5 fps
			Town01,02,03  --- 8-15 fps	
		map detail and # objs matter
	

0.8.4:
  has two maps.
  ./CarlaUE4.sh -windowed -ResX=600 -ResY=400
	run in drive mode. wasd keyboard ctl car
  ./CarlaUE4.sh /Game/Maps/Town01 --carla-settings=Example.CarlaSettings.ini -windowed -ResX=600 -ResY=400
	server mode, wait for client connect, no wasd ctl
  ./CarlaUE4.sh /Game/Maps/Town02 --carla-settings=Example.CarlaSettings.ini

	15 fps.

  no config.py to change map at run time.

----7/26/22 carla-training-data testing -------------
two computer setup work for 0.8.4

datagen..py:
	pyenv shell system
	python3 datageneration.py -a
		start auto pilot
	see constants.py
	data saving logic: STEPS_BETWEEN_RECORDINGS=10 (sec)
		DISTANCE_SINCE_LAST_RECORDING=10
to add pedestrian in the saved data, edit
    constants.py: 
	CLASSES_TO_LABEL = ["Vehicle" , "Pedestrian"]

the measurements contain all cars and pedestrian. the function
	create_kitti_datapoint() check if each object is viewable in current
	image, using player and objs location and camera matrix

    for agent in measurements.non_player_agents:
        print(agent.id) # unique id of the agent
        #print(type(agent)) # unique id of the agent
        #print(agent) # unique id of the agent
        if agent.HasField('vehicle'):
            print('got vehicle:') # unique id of the agent
            print(agent.vehicle.forward_speed)
            print(agent.vehicle.transform)
            print(agent.vehicle.bounding_box)
        if agent.HasField('pedestrian'):
            print('got pedestrian:') # unique id of the agent
            print(agent.pedestrian.transform)
            print(agent.pedestrian.bounding_box)


----7/15/22 carla-training-data testing -------------
/media/student/data6/venk/carla-training-data/visualization
	python3 vis_utils.py -d ../_out --show_image_with_boxes --vis --ind 1

pip3 install PyQt5 mayavi vtk
sudo apt install python3-pyqt5.qtsvg
pip3 install pyqt5-tools 

----7/15/22 carla testing -------------
carla version messy:
	0.8.4 no window zip, don't know how to install python api
	carla crash in linux if wrong client version. huge file in /var/lib/apport/coredump

0.8.4 finaly working:
Install process:
	download CARLA_0.8.4.tar.gz
	cd PythonClient
	pyenv shell system
		use system python3
	python3 setup.py build
	sudo python3 setup.py install
	pip3 list|grep carla
		this should show "carla-client   0.8.4"
Test run:
./CarlaUE4.sh --carla-settings=Example.CarlaSettings.ini
python3 manual_control.py
	system python3

carla changes:
	0.8.4 carla.client python code
	0.9.13 Client c++ code


----7/15/22 yolov5_ros testing -------------

/media/student/data5/cvbridge_build_ws/
	build with catkin build and python3
	test with usb_cam pkg
	currently force it to use cpu
		cv_bridge and python3 issue was trick and for now seems resolved. see readme file 
		in yolov5_ros

----7/8/22 testing 2d-3d labeling -------------
	LDLS/
		2d-3d code, ipynb run time error at arldell, cudnn problem, 
		runs good at newamdpc, gpu memory?
		pyenv shell mini...
		conda env create -f environment.yml
		conda activate ldls
		pip install Cython
		conda deactivate
		conda env update --file environment.yml
		conda activate ldls
		pip install scikit-image
		ipython kernel install --user --name=LDLS
		jupyter-lab # this will load jupyter webpage to run ipynb
			plotly not working to show points
			fix: install nodejs 14 and jupyter labextension. 
				curl -fsSL https://deb.nodesource.com/setup_14.x | sudo -E bash -
				sudo apt install -y nodejs
				jupyter labextension install "@jupyter-widgets/jupyterlab-manager"

				jupyter labextension install "jupyterlab-plotly"	
		jupyter-notebook plotpy seems to word.
		
	ssd_keras/
		testing tensorflow code similar to LDLS , OOM run out of 
		memory error 
	
-----7/6/22 add h5 code to tutorial.py to save carla sim data ---
-----6/30/22 arl venk Deep_Continuous_Fusion @ arldell  ---
	arldell:
        	venk/Deep...
	msilap:
		k:\
        forked.
        https://github.com/juwangvsu/Deep_Continuous_Fusion_for_Multi-Sensor_3D_Object_Detection.git


        carla simulator
	msilap:
		k:\
	arldell:
        	venk/...
                both linux and window works. python api code work
                remote carla simulator work. specify host ip at client
		run simulator:
			cd /media/student/data6/venk/Carla_9-13
			./CarlaUE4.sh
		run python code:
			pyenv shell mini...
			TTI dataset formacd /media/student/data6/venk/Carla_9-13/PythonAPI/examples
			student@arldell:/media/student/data6/venk/PythonAPI/examples$ python vehicle_gallery.py  --host 192.168.86.229
			student@arldell:/media/student/data6/venk/PythonAPI/examples$ python manual_control.py

	data repot:
		https://drive.google.com/drive/folders/1rGEApv2lBG_HFMQrA_4snG3E4_rQCpF3

------------------FAQ trouble shooting-----------------

arldell gpu memory not enough if chrome or other gpu program running.

numpy if upgrade to 1.21+ might break numba.

carla 0.9.13 seems won't run without a gpu

