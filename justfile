HOME := home_dir()
PARENT := parent_directory(justfile_directory())

IMAGE_NAME := "vinny"
IMAGE_TAG := "latest"

clone-3d-libs:
    git clone https://github.com/Tencent-Hunyuan/Hunyuan3D-2.git  #3b9a151
    git clone https://github.com/huanngzh/MV-Adapter.git  #4277e00
    git apply patches/MV-Adapter__Fix_import_and_enable_debug.patch --directory MV-Adapter

checkpoints:
	mkdir -p checkpoints
	wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth -O ./checkpoints/RealESRGAN_x2plus.pth
	wget https://github.com/Sanster/models/releases/download/add_big_lama/big-lama.pt -O ./checkpoints/big-lama.pt

install-3d-libs: clone-3d-libs checkpoints
    cd Hunyuan3D-2 && python3 setup.py install
    cd MV-Adapter && python3 setup.py install


build:
	docker buildx build \
	  -t {{IMAGE_NAME}}:{{IMAGE_TAG}} \
	  -f Dockerfile .

run: build
	docker run -it \
	  --gpus all \
	  --mount type=bind,src={{HOME}}/.cache/huggingface,dst=/root/.cache/huggingface \
	  --mount type=bind,src={{PARENT}}/vnyx,dst=/app/vnyx \
	  --mount type=bind,src={{PARENT}}/vinny/vinny,dst=/app/vinny \
	  --mount type=bind,src={{PARENT}}/vinny/input,dst=/app/input \
	  --mount type=bind,src={{PARENT}}/vinny/output,dst=/app/output \
	  -t {{IMAGE_NAME}}:{{IMAGE_TAG}}
