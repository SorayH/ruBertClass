FROM python:3.8.12-buster

RUN /usr/local/bin/python -m pip install --upgrade pip

WORKDIR /app

COPY . .
RUN pip3 install --no-cache-dir -r requirements.txt
RUN pip3 install --no-cache-dir torch==1.10.0

EXPOSE 5000

CMD ["python", "./app.py"]

