docker run -itd \
  --ipc=host \
  --name swift_rag \
  --gpus all \
  --privileged=true \
  --network default \
  -p 28067:6067 \
  -p 28068:6060 \
  -p 28080:18080 \
  -v ./:/app \
  -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
  registry.sensetime.com/unified-platform/agi-application/explorer/sense-rag:v1.3-roche \
  /bin/bash -c "source activate roche && pip install fastapi pydantic_settings && pip install uvicorn && pip install llama-index-vector-stores-milvus==0.1.23 && pip install pymilvus==2.5.3 && pip install openpyxl && pip install ipdb"

  # /bin/bash -c "source activate roche && pip install fastapi pydantic_settings && pip install uvicorn && pip install llama-index-vector-stores-milvus==0.1.23 && pip install pymilvus==2.5.3 && pip install openpyxl && python -m src.main"


# docker exec -it swift_rag /bin/bash
# source activate roche