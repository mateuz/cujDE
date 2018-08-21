OBJ_PATH = objects
SRC_PATH = source

SRCS = $(wildcard $(SRC_PATH)/*.cpp)
CUDA_SRCS = $(wildcard $(SRC_PATH)/*.cu)

OBJS  = $(patsubst %.cpp, $(OBJ_PATH)/%.o, $(notdir $(SRCS)))
OBJS += $(patsubst %.cu, $(OBJ_PATH)/%.cu.o, $(notdir $(CUDA_SRCS)))

CC = g++ -std=c++11 -O3
C-CUDA = nvcc -arch=sm_61 -std=c++11 -O3 -rdc=true -use_fast_math

BOOST_FLAG = -lboost_program_options

all: demo

demo: $(OBJS)
		$(C-CUDA) $^ $(BOOST_FLAG) -o $@

$(OBJ_PATH)/%.cu.o : $(SRC_PATH)/%.cu
		$(C-CUDA) -o $@ -c $<

$(OBJ_PATH)/%.o : $(SRC_PATH)/%.cpp
	$(CC) -o $@ -c $<

clean:
	-rm -f $(OBJ_PATH)/*.o demo

run:
	./demo
