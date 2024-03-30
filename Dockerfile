# Base image
FROM python:3.11.8

RUN apt-get update
RUN apt-get install -y libgl1-mesa-glx graphviz

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
ADD . .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
