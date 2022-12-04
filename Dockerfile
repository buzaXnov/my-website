FROM python:3.10-alpine
LABEL maintainer="filipbutic"

ENV PYTHONUNBUFFERED 1

COPY ./requirements.txt /requirements.txt
COPY ./app /app

WORKDIR /app
EXPOSE 8000

# Docker will create a new image layer for every single RUN command so
# if we want to reduce the size, we use put all of them in a single RUN cmd
# installs everything the postgres client needs to connect
RUN python -m venv /py && \
    /py/bin/pip install --upgrade pip && \
    apk add --update --no-cache postgresql-client && \
    apk add --update --no-cache --virtual .tmp-deps \
        build-base postgresql-dev musl-dev && \
    /py/bin/pip install -r /requirements.txt && \
    apk del .tmp-deps && \
    adduser --disabled-password --no-create-home app
    # if the last line is not done then the program will be run as root
    # apk - alpine package manager
    # --no-cache to save on memory
    # del tmp-deps remove the dependencies to keep the image lightweight

# we want our version of python to be run inside the environment
ENV PATH="/py/bin:$PATH"

# switches to the app user insted of the root user
USER app