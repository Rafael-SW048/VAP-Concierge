FROM conda/miniconda3

# noninteractive only when build
ARG DEBIAN_FRONTEND=noninteractive

# install dependencies
RUN apt update && apt upgrade -y
RUN apt install wget ffmpeg libgl1 -y &&\
	apt clean

COPY environment.yml /tmp
RUN conda update -n base -c defaults conda
RUN conda env create -f /tmp/environment.yml
RUN conda init
RUN echo "conda activate tensorflow" >> /root/.bashrc
RUN conda clean --all

# get saved model
RUN wget http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_resnet50_v1_fpn_1024x1024_coco17_tpu-8.tar.gz &&\
	tar -xvzf ssd_resnet50_v1_fpn_1024x1024_coco17_tpu-8.tar.gz --transform s/ssd_resnet50_v1_fpn_1024x1024_coco17_tpu-8/model/ &&\
	rm ssd_resnet50_v1_fpn_1024x1024_coco17_tpu-8.tar.gz
ENV MODEL_PATH=/model/saved_model

# allow ssh to the container as root and add local public key to the authorized keys
# RUN echo "PermitRootLogin without-password" > /etc/ssh/sshd_config
# RUN --mount=type=secret,id=my_secret mkdir -p -m 0600 /root/.ssh &&\
# 	echo $(cat /run/secrets/my_secret) > /root/.ssh/authorized_keys

# copy source code to container
COPY . /app
ENV PYTHONPATH=/app/src:$PYTHONPATH
ENV TF_CPP_MIN_LOG_LEVEL=2
ENV CONCIERGE_URI='host.docker.internal:5000'

WORKDIR /app/src/app
# listening on port 22
# EXPOSE 22

# CMD service ssh restart && bash
CMD bash
ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "tensorflow", "python", "server.py"]

