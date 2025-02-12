FROM python:3.11
WORKDIR  /app
COPY scripts /app
COPY requirements.txt /app/requirements.txt
COPY runs/detect/watermark_detection/weights/best.pt ./runs/detect/watermark_detection/weights/best.pt
RUN apt-get update
RUN apt install -y libgl1-mesa-glx
RUN pip install --no-cache-dir -r requirements.txt
EXPOSE 5000
CMD ["python", "api.py"]