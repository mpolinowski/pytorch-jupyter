FROM pytorch/pytorch:latest

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && \
    apt-get install -y \
        git \
        tini \
        python3-pip \
        python3-dev \
        python3-opencv \
        libglib2.0-0

# intall optional python deps
RUN python -m pip install --upgrade pip
RUN pip install jupyter
RUN pip install ultralytics
RUN python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'


# Set the working directory
WORKDIR /opt/app

# Start the notebook
RUN chmod +x /usr/bin/tini
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["jupyter", "notebook", "--port=8888", "--no-browser", "--ip=0.0.0.0", "--allow-root"]