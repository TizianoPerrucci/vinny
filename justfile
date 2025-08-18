IMAGE_NAME := "vinny"

clone-3d-libs:
    git clone git@github.com:Tencent-Hunyuan/Hunyuan3D-2.git  #3b9a151
    git clone git@github.com:huanngzh/MV-Adapter.git  #4277e00

install-3d-libs:
    cd Hunyuan3D-2 && python3 setup.py install
    cd MV-Adapter && python3 setup.py install


build:
	docker buildx build \
	  -t {{IMAGE_NAME}}:latest \
	  -f Dockerfile .

run: build
	docker run -it \
	  --gpus all \
	  --mount type=bind,src=/home/tiziano/.cache/huggingface,dst=/root/.cache/huggingface \
	  -t {{IMAGE_NAME}}:latest
