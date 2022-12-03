FROM python:3.10-slim-buster

WORKDIR /app

COPY . .

# update & upgrade registry
RUN apt-get update
RUN apt-get upgrade

# Update pip
RUN python -m pip install --upgrade pip

# Instal python dependencies
RUN python -m pip install -r requirements.txt

# Required for open-cv module
RUN apt-get install ffmpeg libsm6 libxext6 -y

# Required for tensorflow.keras.utils.plot_model
RUN apt-get install graphviz

CMD [ "sleep", "infinity"]