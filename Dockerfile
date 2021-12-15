FROM ubuntu

# Install python
RUN apt-get update

RUN DEBIAN_FRONTEND="noninteractive"  \
    apt-get install -y --allow-unauthenticated --no-install-recommends \
    build-essential cmake \
    python3-pip python3-setuptools python3-dev pkg-config  && \
    apt-get clean

RUN apt-get install -y --allow-unauthenticated --no-install-recommends \
    gcc g++ clang git

RUN pip3 install cython && \
    git clone https://github.com/tulip-control/dd.git && \
    cd dd && \
    python3 setup.py install --fetch --cudd

RUN pip3 install guppy3
RUN pip3 install jupyterlab

