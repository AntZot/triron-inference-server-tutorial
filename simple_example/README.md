
### Установка весов yolov8
```bash

```
### Запуск простой модели
docker run --rm -it -p8000:8000 -p8001:8001 -p8002:8002 -v $(pwd)/model_repository:/models nvcr.io/nvidia/tritonserver:24.01-py3