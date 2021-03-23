FROM python:3.7-stretch

ENV CRYPTOGRAPHY_DONT_BUILD_RUST=1
ENV PYTHONUNBUFFERED=1

WORKDIR /root/MCUP

RUN apt-get update \ 
    && apt-get install build-essential make gcc -y \
    && apt-get install dpkg-dev -y \ 
    && apt-get install libjpeg-dev -y 

RUN python3 -m pip install numpy scipy wheel twine \
    && rm -rf /opt/build/* \
    && rm -rf /var/cache/apk/* \
    && rm -rf /root/.cache/* \
    && rm -rf /tmp/* 

COPY . .

