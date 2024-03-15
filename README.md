# triron-inference-server-tutorial

## Что потребуется установить 
- Docker: https://docs.docker.com/engine/install/
- Установить NVIDIA Container Toolkit: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html
- Образы triton infernece server: https://catalog.ngc.nvidia.com/orgs/nvidia/containers/tritonserver/tags

## Что может быть полезно
- Построение ансамблей моделей: [ссылка](https://developer.nvidia.com/blog/serving-ml-model-pipelines-on-nvidia-triton-inference-server-with-ensemble-models/)
- Репозиторий с туториалами от NVIDIA: [ссылка](https://github.com/triton-inference-server/tutorials)
- Tritonserver Model Analyzer :
  1) GitHub Репозиторий Model Analyzer: [ссылка](https://github.com/triton-inference-server/model_analyzer)
  2) Пример для ансамблей: [ссылка](https://github.com/triton-inference-server/model_analyzer/blob/main/docs/ensemble_quick_start.md)
  3) Флаги для написания файл конфигурации анализатора (advanced): [ссылка](https://github.com/triton-inference-server/model_analyzer/blob/main/docs/config.md#config-options-for-profile)


## 1. Установка NVIDIA Container Toolkit
```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
  && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
```

```bash
sudo apt-get update
```

```bash
sudo apt-get install -y nvidia-container-toolkit
```
### Установка Triton Inference Server
```bash
docker pull nvcr.io/nvidia/tritonserver:24.01-py3
```
```bash
docker images
```
## 2. Базовые понятия triton infernece server
```bash
# Expected folder layout
model_repository/
├── text_detection
│   ├── 1
│   │   └── model.onnx
│   └── config.pbtxt
└── text_recognition
    ├── 1
    │   └── model.onnx
    └── config.pbtxt
```

**config.pbtxt**
``` text proto
name: "text_detection"
backend: "onnxruntime"
max_batch_size : 256
input [
  {
    name: "input_images:0"
    data_type: TYPE_FP32
    dims: [ -1, -1, -1, 3 ]
  }
]
output [
  {
    name: "feature_fusion/Conv_7/Sigmoid:0"
    data_type: TYPE_FP32
    dims: [ -1, -1, -1, 1 ]
  }
]
output [
  {
    name: "feature_fusion/concat_3:0"
    data_type: TYPE_FP32
    dims: [ -1, -1, -1, 5 ]
  }
]

dynamic_batching {
    preferred_batch_size: [ 4, 8 ]
  }
```

Разберем файл конфигов
- name - Необязательно поле, которое обозначает название модели (Note: Название должно совпадать с названием каталога!)
- backend/platform* - указывает на бэкенд, который используется для запуска модели. Бэкенды, которые поддерживает triton: [Документация](https://docs.nvidia.com/deeplearning/triton-inference-server/archives/triton_inference_server_220/user-guide/docs/backend.html), [Github](https://github.com/triton-inference-server/backend#backends)
- max_batch_size - Значение, которое указывает на максимальный размер батча, который воспринимает модель
- input and output: Конфигурация полей для  

[Все возможные поля конфигурации](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/model_configuration.html)

*platform Является старой нотацией, новым стандартом является использование backend


На примере выше, размерность входных данных соотвествует при max_batch_size > 0:
```
[-1] + dims = [-1, -1, -1, -1, 3]
```
Для моделей с max_batch size = 0:
```
dims = [-1, -1, -1, 3]
```
## 3. Инференс с помощью Triton Inference Server
Для начала инференса потребутеся запустить контейнер с triton inference server, запустить на нем модели и отправить запрос с клиента. 

**!! NOTE !!** - все скрипты должны выполняться в корневой директории 
### 3.1 Запуск контейнера и triton
```bash
docker run --rm -it --gpus=0 -p8000:8000 -p8001:8001 -p8002:8002 -v $(pwd)/model_repository:/models nvcr.io/nvidia/tritonserver:24.01-py3

tritonserver --model-repository=/models
```
### 3.2 Загрузить onnx веса в model repository
```bash
source ./utils/yolo_export.sh
```

### 3.3 Отправка запроса через скрипт клиента
```bash
python client.py
```
Как результат в консоль должен вывестись JSON ответа и появиться файл `result.jpg`
```bash
{'model_name': 'detection',
 'model_version': '1',
 'outputs': [{'datatype': 'FP32',
              'name': 'output0',
              'parameters': {'binary_data_size': 2822400},
              'shape': [1, 84, 8400]}]}
```
## 4. Model Analyzer
Установка sdk для анализа
```bash
docker run -it --net=host -v ${PWD}:/workspace/ nvcr.io/nvidia/tritonserver:24.01-py3-sdk bash
```
### Performance Analyzer

WORK IN PROGRESS!!
```bash
# perf_analyzer -m <model name> -b <batch size> --shape <input layer>:<input shape> --concurrency-range <lower number of request>:<higher number of request>:<step>

# Query
perf_analyzer -m text_recognition -b 2 --shape input.1:1,32,100 --concurrency-range 2:16:2 --percentile=95
```


### Model Analyzer
1) Run Triton inference Server container with explicit mode
```bash
docker run --gpus=0 --shm-size=1G --rm -it -p8000:8000 -p8001:8001 -p8002:8002 -v $(pwd)/model_repository:/models nvcr.io/nvidia/tritonserver:24.01-py3

tritonserver --model-repository=/models --model-control-mode=explicit
```
2) Run SDK Triton Inference server container
```bash
docker run -it --gpus all -v /var/run/docker.sock:/var/run/docker.sock -v $(pwd):/workspace --net=host nvcr.io/nvidia/tritonserver:24.01-py3-sdk
```