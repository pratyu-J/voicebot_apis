FROM python:3.9.0-slim

# setting the working directory in the container
WORKDIR /app

# copying contents into the container

COPY . /app

# installing required packages
RUN pip install --trusted-host pypi.python.org -r requirements.txt

# opening port 3005 to outside world
EXPOSE 5050

CMD [ "python", "chat_gdrfa_api.py" ]
