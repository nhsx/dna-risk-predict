FROM python:3.10-slim

COPY . .
ENV PIP_ROOT_USER_ACTION=ignore

RUN    apt-get -y update \
    && apt-get -y install git \
    && pip install .

ENTRYPOINT ["dnattend"]
