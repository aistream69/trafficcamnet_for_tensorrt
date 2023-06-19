# trafficcamnet_for_tensorrt
An end to end sample for trafficcamnet model(NVIDIA NGC), running with tensorrt.
## Depend
cuda
tensorrt
deepstream
opencv
## Build
mkdir build
cd build
cmake ..
make
## Run
./test --model /path_to_model/resnet18_trafficcamnet_pruned.int8.engine --img car.jpeg --iterators 100

