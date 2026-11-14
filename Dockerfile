# On-premise CPU image: the ResNet-18 and DeepLabV3 checkpoints are fetched at build time, nothing at run time.
# Not built or run during local preparation (Docker was unavailable); see README.
FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY *.py ./
RUN python -c "from features import ResNetPatchFeatures; ResNetPatchFeatures(); from inspect_cracks import build_model; build_model()"
VOLUME ["/data", "/output"]
EXPOSE 8766
ENTRYPOINT ["python"]
CMD ["serve.py", "--bank", "/data/bank", "--host", "0.0.0.0", "--port", "8766"]
