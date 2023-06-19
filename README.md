# trafficcamnet_for_tensorrt
An end to end sample for trafficcamnet model(NVIDIA NGC), running with tensorrt.
## Depend
cuda  
tensorrt  
deepstream  
opencv  
tao-converter
## Convert etlt to engine
download files from https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tao/models/trafficcamnet/files?version=pruned_v1.0.3  
tao-converter -k tlt_encode -d 3,544,960 -t int8 -c trafficcamnet_int8.txt -e resnet18_trafficcamnet_pruned.int8.engine resnet18_trafficcamnet_pruned.etlt
## Build
mkdir build  
cd build  
cmake ..  
make
## Run
./test --model /path_to_model/resnet18_trafficcamnet_pruned.int8.engine --img car.jpeg --iterators 100

