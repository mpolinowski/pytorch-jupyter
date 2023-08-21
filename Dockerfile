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

# Upgrade pip
RUN python3 -m pip install --upgrade pip
RUN pip3 install jupyter


# Set the working directory
WORKDIR /opt/app

# Start the notebook
RUN chmod +x /usr/bin/tini
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["jupyter", "notebook", "--port=8888", "--no-browser", "--ip=0.0.0.0", "--allow-root"]