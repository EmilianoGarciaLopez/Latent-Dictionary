# Use the slim-bullseye version of the official Python runtime as the base image
FROM python:3.11.6-slim-bullseye

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set the working directory inside the container
WORKDIR /app

# Copy requirements.txt into the container at /app
COPY requirements.txt /app/

# copy the .env file
COPY .env /app/

# Install required packages
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy the relevant project files into the container at /app
COPY index.py /app/
COPY embeddings.pkl /app/

# Expose port 8050 for the app to listen on
EXPOSE 8050

# Define the command to run the app
CMD ["python", "index.py"]
