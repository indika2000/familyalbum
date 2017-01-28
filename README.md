# familyalbum
A face detection/recognition tool for your photos to index by person and display on Django website

Setting up the OPENCV environment on Ubuntu 16
1. Ensure apt-get is up-to-date

    '''
   $ sudo apt-get update
   $ sudo apt-get upgrade
    '''

2. Download the image type libraries needed for loading image file types

    '''
    $ sudo apt-get install libjpeg8-dev libtiff5-dev libjasper-dev libpng12-dev
    '''

3. Download the libraries for video streaming and frame access from webcams - more for future extension of the project
  '''
  $ sudo apt-get install libavcodec-dev libavformat-dev libswscale-dev libv4l-dev
  $ sudo apt-get install libxvidcore-dev libx264-dev
  '''

4. Install the GTK library to allow the basic OpenCV GUI operations
    '''
  	$ sudo apt-get install libgtk-3-dev
    '''

5. Install libraries that are used to optimize various functionalities inside OpenCV, such as matrix operations
    '''
    $ sudo apt-get install libatlas-base-dev gfortran
    '''

