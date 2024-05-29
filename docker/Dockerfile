ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG HOST_UID
ARG HOST_GID
ARG HOST_USER
ENV HOST_UID=${HOST_UID?-1000}
ENV GROUP_ID=${HOST_GID?-1000}
ENV USER_NAME=${HOST_USER?-vueron}

RUN groupadd -g $HOST_GID $HOST_USER 2>/dev/null && \
    useradd -r -u $HOST_UID -g $HOST_USER -d /home/$HOST_USER -m $HOST_USER 2>/dev/null && \
    usermod -aG sudo $HOST_USER && \
    echo "$HOST_USER ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers && \
    chown $HOST_USER /home/$HOST_USER

RUN apt update
RUN apt install libpcl-dev libopencv-dev -y

ARG PROJECT_MOUNT_PATH
ENV PROJECT_MOUNT_PATH=${PROJECT_MOUNT_PATH?-/Project}
RUN echo "if [ -f $PROJECT_MOUNT_PATH/setup ]; then source $PROJECT_MOUNT_PATH/setup; fi" >> /home/$USER_NAME/.bashrc
