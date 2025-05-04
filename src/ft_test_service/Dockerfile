# Use an official lightweight Python image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy the application code to the container
COPY ft_test.py .

# Install necessary dependencies
RUN pip install --no-cache-dir fastapi uvicorn pydantic

# Expose the port the app will run on
EXPOSE 8002

# Run the application
CMD ["uvicorn", "ft_test:app", "--host", "0.0.0.0", "--port", "8002"]