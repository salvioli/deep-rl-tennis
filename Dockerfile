FROM pytorch/pytorch

WORKDIR /deep-rl-tennis/

RUN apt-get update && apt-get install -y unzip && \
    python -m pip install unityagents && \
    apt-get install wget && \
    mkdir -p resources && \
    wget https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux_NoVis.zip -P ./resources/ && \
    wget https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip -P ./resources/ && \
    cd resources && \
    unzip Tennis_Linux.zip && \
    unzip Tennis_Linux_NoVis.zip && \
    rm Tennis_Linux.zip && \
    rm Tennis_Linux_NoVis.zip

COPY tennis.py ddpg_agent.py model.py config.py config.yml ./

CMD ["python", "tennis.py", "train", "-f", "config.yml"]
