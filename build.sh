CFLAGS="-fsycl -fsycl-targets=nvptx64-nvidia-cuda \
-Xsycl-target-backend --cuda-gpu-arch=sm_80 -Xsycl-target-backend -O3"
clang++ -o bench $CFLAGS bench.cpp
#-save-temps \
#-c \
#test.cpp

#-Xsycl-target-frontend -ffp-contract=on \
#-Xcuda-ptxas --verbose \
#--cuda-path=$CUDA_HOME \
