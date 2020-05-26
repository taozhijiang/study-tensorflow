A simple example using tensorflow from scratch.   


**mnist-example**   
MNIST handwritten using tensorflow v1.x, including C++ inference support.    
```bash
~ # install python
~ sudo pip3 install tensorflow==1.15.2
~ # normal build 
~ # origin/r1.13
~ /opt/bazel-prefix/0.21.0/bin/bazel build --config=opt --config=noaws --config=nogcp --config=nohdfs --config=nokafka --config=noignite --config=nonccl --cxxopt=-D_GLIBCXX_USE_CXX11_ABI=0 //tensorflow:libtensorflow_cc.so
~ ls -l bazel-bin/tensorflow/libtensorflow_*.so
-r-xr-xr-x  1 taozj  wheel  212732328  5 25 23:41 bazel-bin/tensorflow/libtensorflow_cc.so
-r-xr-xr-x  1 taozj  wheel   15104020  5 25 23:41 bazel-bin/tensorflow/libtensorflow_framework.so
~
```


**tf-dbg**   
Simplified tensorflow v1.13.z, the op was extremely remove and only support simple mul, add ops.   
The tensorflow library should be compiled like:   
```bash
~ # origin/r1.13.z-lite
~ /opt/bazel-prefix/0.21.0/bin/bazel build --config=monolithic -c dbg --strip=never --config=noaws --config=nogcp --config=nohdfs --config=nokafka --config=noignite --config=nonccl --cxxopt=-D_GLIBCXX_USE_CXX11_ABI=0 //tensorflow:libtensorflow_cc.so
~ ls -l bazel-bin/tensorflow/libtensorflow_cc.so
-r-xr-xr-x  1 taozj  wheel  1286242280  5 26 22:39 bazel-bin/tensorflow/libtensorflow_cc.so
~
```


**tf-dbg-v2**   
Like tf-dbg, but using tensorflow v2.   
```bash
~ # origin/master-lite
~ /opt/bazel-prefix/3.0.0/bin/bazel build --config=monolithic -c dbg --strip=never --config=noaws --config=nogcp --config=nohdfs --config=nonccl --cxxopt=-D_GLIBCXX_USE_CXX11_ABI=0 //tensorflow:libtensorflow_cc.so
~ ls -l bazel-bin/tensorflow/libtensorflow_cc.so
lrwxr-xr-x  1 taozj  wheel  21  5 23 13:36 bazel-bin/tensorflow/libtensorflow_cc.so -> libtensorflow_cc.so.2
~
```


**tfs-dbg**   
Tensorflow Serving project, using simplified tensorflow v1.13.z.   
```bash
~ # origin/r1.13-lite
~ # change the commitid according to tensorflow origin/r1.13.z-lite
~ /opt/bazel-prefix/0.21.0/bin/bazel build --action_env TF_REVISION="4e10db870cbdce3eb703a4ea179643d424327ab5" -c dbg --strip=never --cxxopt=-D_GLIBCXX_USE_CXX11_ABI=0 //tensorflow_serving/model_servers:tensorflow_model_server --workspace_status_command=tools/gen_status_stamp.sh
~ ls -l bazel-bin/tensorflow_serving/model_servers/tensorflow_model_server
-r-xr-xr-x  1 taozj  wheel  1393543592  5 27 00:02 bazel-bin/tensorflow_serving/model_servers/tensorflow_model_server
~
```