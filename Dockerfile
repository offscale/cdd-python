FROM python:3.6-alpine

WORKDIR /usr/src/app

COPY requirements.txt ./
RUN apk add --no-cache gcc musl-dev && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir meta

COPY . .

CMD [ "python", "setup.py", "test" ]
