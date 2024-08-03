FROM ros:noetic-ros-base-focal

RUN apt-get -y update 
RUN apt-get -y upgrade
RUN DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC apt-get -y install tzdata

RUN apt-get install -yq curl build-essential
RUN apt-get install -yq python3-pip
RUN apt-get install -y git
RUN apt-get install ffmpeg libsm6 libxext6 -y
RUN apt-get install -yq ros-noetic-cv-bridge
RUN apt-get install -yq ros-noetic-rviz

RUN mkdir /home/workspace

COPY environment.yml environment.yml

WORKDIR /home/workspace