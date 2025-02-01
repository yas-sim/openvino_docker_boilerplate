# A boilerplate for an OpenVINO application in a container project using an OpenVINO base container image

Sample `Dockerfile` and OpenVINO application to demonstrate how to create a Docker project for OpenVINO using OpenVINO base container image.  
The sample application uses `ssd_mobilenet_v1` model and run object detection task. The application reads an image from `./input/image.jpg` and write the result to `./output/image.jpg`.

### How to get the DL model
```sh
python prepare_model.py
```

### How to build the image
```sh
cd Docker
docker build -t ov_app .
```

### How to run the container
```sh
docker run --rm -v <host_working_dir>/input:/app/input -v <host_working_dir>/output:/app/output ov_app

```

### Start the container for debugging / testing
```sh
docker run --rm -it -v <host_working_dir>/input:/app/input -v <host_working_dir>/output:/app/output ov_app /usr/bin/bash
```


### Note:
- The application file is `openvino_app.py`.
- The application code and a DL model will be deployed in `/app` directory with the `Dockerfile` in this project.
- Default working directory for the container `openvino/ubuntu22_runtime:2024.6.0` is `/opt/intel/openvino_2024.6.0.0`
