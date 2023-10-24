# Use the slim-bullseye version of the official Python runtime as the base image for builder
FROM python:3.11.6-slim-bullseye AS builder

# Set the working directory inside the container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y gcc libpython3-dev

# Copy requirements.txt into the container at /ap
COPY requirements.txt /app/

# Upgrade pip and install required packages using --user flag (install packages to the user site)
RUN pip install --upgrade pip && pip install --user -r requirements.txt

# Use another slim-bullseye Python image for the final image
FROM python:3.11.6-slim-bullseye

# Copy .env file and python files
COPY .env /app/
COPY refactored.py /app/
COPY layout.py /app/
COPY embeddings.pkl /app/

# Expose port 8050 for the app to listen on
EXPOSE 8050

# Define the command to run the app
CMD ["gunicorn", "refactored:app", "-b", "0.0.0.0:8050", "-w", "4"]