FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy all files into the container
COPY . .

# Expose port 8080
EXPOSE 8080

# Command to run the application
CMD ["python", "api_model.py"]
