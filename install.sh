## Install conda environment
CENTERPOINT_DIR="/home/j316chuck/dev/CenterPoint/"
conda create -n ouster-lidar python=3.6
conda activate ouster-lidar
conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch -c nvidia -c conda-forge
pip install -r requirements.txt
# Install applied nuscenes devkit in develop editable mode
pip uninstall nuscenes-devkit
cd ${CENTERPOINT_DIR}/nuscenes-devkit/setup
pip install -e .
cd ${CENTERPOINT_DIR}/
# install sparse convolution
pip install spconv-cu111
# install iou3d_nms and dcn operations in cuda make sure you export cuda and centerpoint paths correctly
#export CUDA_BIN_PATH=/usr/local/cuda-11.1/
#export PYTHONPATH="${PYTHONPATH}:${CENTERPOINT_DIR}" 
sudo apt-get install libboost-all-dev
${CENTERPOINT_DIR}/setup.sh

## Create data
# python3 tools/create_data.py nuscenes_data_prep --root_path=./data/detroit_mini/synthetic_nuscenes/ --version="v1.0-mini" --nsweeps=10
# python3 tools/create_data.py nuscenes_data_prep --root_path=./data/nuscenes_mini/ --version="v1.0-mini" --nsweeps=10

## modify config file with right directories:
# vim ./configs/nusc/pp/nusc_centerpoint_pp_02voxel_two_pfn_10sweep_demo.py 

## run inference
# python ./tools/dist_test.py ./configs/nusc/pp/nusc_centerpoint_pp_02voxel_two_pfn_10sweep_demo.py  --checkpoint work_dirs/centerpoint_pillar_512_demo/latest.pth --speed_test

