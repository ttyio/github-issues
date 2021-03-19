docker build -f Dockerfile -t issue_964 .
docker run --runtime=nvidia -it --cap-add=SYS_ADMIN --device /dev/fuse --security-opt apparmor:unconfined --rm --ipc=host issue_964 
