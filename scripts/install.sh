# Python
conda install -y ipython pip
# PyTorch
conda install -y pytorch==1.1.0 torchvision==0.3.0 cudatoolkit=10.0 -c pytorch
# Install deps
pip install ninja yacs cython matplotlib tqdm opencv-python
export INSTALL_DIR=$PWD
rm -r libs
mkdir libs
# COCOAPI
cd $INSTALL_DIR/libs
git clone https://github.com/cocodataset/cocoapi.git
cd cocoapi/PythonAPI
python setup.py build_ext install
# APEX
cd $INSTALL_DIR/libs
git clone https://github.com/NVIDIA/apex.git
cd apex
python setup.py install --cuda_ext --cpp_ext
# ShapeProp
cd $INSTALL_DIR
python setup.py build develop
