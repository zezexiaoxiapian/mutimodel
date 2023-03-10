# Copyright (c) 1993-2018, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
CUDA_INSTALL_DIR=/usr/local/cuda

CXXFLAGS=-std=c++1z -DONNX_ML=1 -Wall -I$(CUDA_INSTALL_DIR)/include
LDFLAGS=-L$(CUDA_INSTALL_DIR)/lib64 -L$(CUDA_INSTALL_DIR)/lib64/stubs -L/usr/local/lib
# LDLIBS=-Wl,--start-group -lnvonnxparser -lnvinfer -lcudart_static -lonnx -lonnx_proto -lprotobuf -lstdc++ -lm -lrt -ldl -lpthread -Wl,--end-group
LDLIBS=-Wl,--start-group -lopencv_core -lopencv_imgcodecs -lopencv_imgproc -lnvonnxparser -lnvinfer -lcudart_static -lprotobuf -lstdc++ -lm -lrt -ldl -lpthread -Wl,--end-group

HEADERS=${wildcard *.h}
TARGET_SRCS=inferDemo.cpp
TARGET_OBJS=${TARGET_SRCS:.cpp=.o}
TARGETS=${TARGET_OBJS:.o=}


all: $(TARGETS)

$(TARGETS): %: %.o ioHelper.o

%.o: $(HEADERS)

clean: clean_engines
	rm -f $(TARGETS) *.o

clean_engines:
	rm -f *.engine

.PHONY: clean_engines all clean
