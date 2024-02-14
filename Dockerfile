#Build Stage
FROM nvcr.io/nvidia/nvhpc:23.11-devel-cuda_multi-ubuntu22.04
WORKDIR /work
ADD *.cpp *.hpp *.h Makefile /work/
RUN cd /work && make clean all
ENV PATH=$PATH:/work
