IMAGE_NAME := "vinny"
HOME := home_dir()
PARENT := parent_directory(justfile_directory())

clone-3d-libs:
    git clone https://github.com/Tencent-Hunyuan/Hunyuan3D-2.git  #3b9a151
    git clone https://github.com/huanngzh/MV-Adapter.git  #4277e00

install-3d-libs: clone-3d-libs
    cd Hunyuan3D-2 && python3 setup.py install
    cd MV-Adapter && python3 setup.py install


build:
	docker buildx build \
	  -t {{IMAGE_NAME}}:latest \
	  -f Dockerfile .

run:
	docker run -it \
	  --gpus all \
	  --mount type=bind,src={{HOME}}/.cache/huggingface,dst=/root/.cache/huggingface \
	  --mount type=bind,src={{PARENT}}/vnyx,dst=/app/vnyx \
	  --mount type=bind,src={{PARENT}}/vinny/vinny,dst=/app/vinny \
	  --mount type=bind,src={{PARENT}}/vinny/output,dst=/app/output \
	  -t {{IMAGE_NAME}}:latest
