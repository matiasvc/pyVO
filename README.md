# TDT4265 - Project 7

For this project you will implement part of a simple Visual Odometry project, using Harris courner detection and a KLT-Tracker.

Please fork this repo to your own GitHub profile and work from it there. It will then be easy for you to pull down any update i push to this repo. I recommend using python 3.7 with this project.

Most of the required python packages can be safely installed from pip or conda. There are a few exceptions however. I recoment installing pyflann from source as described in [this](https://github.com/primetang/pyflann) repo. Installing the python version of openCV can be tricky, with varius issues depending on platform. Please contact me if you are having problems.

Run the download script `dataset/download.sh` to download a few different datasets you can use.

I recommend you implement things in the follow order:

- Harris Corner detector: In `harrisdetector.py` implement the `harris_corner` function. This function should return a list of tuples containing the response value and location of the harris corners detected in the image.
- KLT-Tracker: In `pointTracker.py` implement the `track_new_image` function in the `KLTTracker` class. This function is called on each new image to track the existing point trackers on the new image.
- 2D to 3D point projection: In `pointPojection.py` implement the `project_points` function which projects to 3D in the cameras local frame using the intrinsic camera parameters and the depth image.

The trajectory optimization code still has some issues i want to fix before i push it.

