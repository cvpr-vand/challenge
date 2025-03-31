# README

## To only build the docker image

```
docker build . -t vand2025-runner -f .platform/DOCKERFILE
```

## To run the docker image with id

```
docker run --gpus gpu_id --shm-size 2G --memory 142g \
-i -t --mount type=bind,source=./datasets,target=/home/user/datasets,readonly \
-d --name vand2025-runner-container-<id> vand2025-runner
```

## Configure the runner in detached mode

```
docker exec -it -d vand2025-runner-container-<id> /bin/bash -c '
if [ ! -f /home/user/actions-runner/.credentials ]; then
    ./actions-runner/config.sh --url https://github.com/samet-akcay/vand-cvpr --token $RUNNER_TOKEN --labels vand2025-runner --unattended
fi
./actions-runner/run.sh
'
```

## To use GPU 3

ensure that you cd into .platform/

```
GPU_ID=0 RUNNER_TOKEN=your_token docker compose up
```
