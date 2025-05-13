FROM python:3.10-slim

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Crear directorio de trabajo
WORKDIR /app

# Copiar todo
COPY . .

# Cambiar directorio de trabajo a la carpeta del backend
WORKDIR /app/api

# Instalar dependencias
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r ../requirements.txt

# Exponer el puerto de la API
EXPOSE 8000

# Comando por defecto

#CMD ["python", "main.py"]
CMD ["uvicorn", "main:app_instance", "--host", "0.0.0.0", "--port", "8000"]
