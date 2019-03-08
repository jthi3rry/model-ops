# Usage:
#
#      docker build -f training.Dockerfile -t pyspark-training .
#
FROM python:3.7-stretch

# Install Java 8
RUN apt-get update && \
    echo "deb http://ppa.launchpad.net/webupd8team/java/ubuntu xenial main" | tee /etc/apt/sources.list.d/webupd8team-java.list && \
    echo "deb-src http://ppa.launchpad.net/webupd8team/java/ubuntu xenial main" | tee -a /etc/apt/sources.list.d/webupd8team-java.list  && \
    apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-keys EEA14886 && \
    apt-get update && \
    echo debconf shared/accepted-oracle-license-v1-1 select true | debconf-set-selections && \
    echo debconf shared/accepted-oracle-license-v1-1 seen true | debconf-set-selections && \
    apt-get --no-install-recommends -y --force-yes install oracle-java8-installer oracle-java8-set-default && \
    rm -r /var/cache/oracle-jdk* && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install pyspark==2.3 numpy

# Set up app
RUN mkdir /app
WORKDIR /app

# Set environment
ENV MODEL_PATH models/churn.spark
ENV DATA_PATH datasets/churn.csv

COPY datasets datasets
COPY training.py training.py

# Run API
ENTRYPOINT spark-submit --master=local[*] training.py
