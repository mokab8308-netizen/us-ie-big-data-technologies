# Use an official, lightweight Python image as the base
FROM python:3.9-slim

# Set the working directory inside the container to /app
WORKDIR /app

# Copy the requirements file into the container first for better caching
COPY requirements.txt .

# Install the Python libraries specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy all other files from your local directory into the container
COPY . .

# Specify the command to run when the container starts
CMD ["python", "main.py"]
