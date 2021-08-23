#base images
FROM ubuntu:18.04
FROM python:3.6

#work dir
WORKDIR /src

#install dependencies    
RUN pip install aix360

#clone repo
RUN git clone https://github.com/Trusted-AI/AIX360.git

#run tutorial inside container
RUN pip install jupyterlab
