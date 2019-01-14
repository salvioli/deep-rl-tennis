FROM pytorch/pytorch

WORKDIR /deep-rl-tennis/

RUN apt-get update && apt-get install -y unzip

RUN mkdir -p resources
ADD https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux_NoVis.zip ./resources/
ADD https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip ./resources/
RUN cd resources && \
    unzip Tennis_Linux.zip && \
    unzip Tennis_Linux_NoVis.zip && \
    rm Tennis_Linux.zip && \
    rm Tennis_Linux_NoVis.zip

RUN python -m pip install unityagents
RUN python -m pip install docker

ADD https://github.com/salvioli/deep-rl-tennis/archive/master.zip .
RUN unzip master.zip && mv deep-rl-tennis-master/* . && rm -r deep-rl-tennis-master

CMD ["python", "tennis.py", "train", "-f", "config.yml"]