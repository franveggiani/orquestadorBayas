FROM aaii/cuda:9.2-cudnn7-devel-ubuntu18.04

ENV DEBIAN_FRONTEND=noninteractive


RUN apt-get update && apt-get install -y \ 
    git \
    wget \
    libopencv-dev \
    libtbb-dev \
    make \
    gcc \
    g++ \
    libopenslide-dev \
    && rm -rf /var/lib/apt/lists/*

# Instalar una version de Miniconda compatible con Ubuntu 18.04 / glibc 2.27
ARG MINICONDA_VERSION=py37_4.12.0
ARG MINICONDA_SHA256=4dc4214839c60b2f5eb3efbdee1ef5d9b45e74f2c09fcae6c8934a13f36ffc3e

ENV CONDA_DIR=/opt/conda
RUN wget --quiet \
        "https://repo.anaconda.com/miniconda/Miniconda3-${MINICONDA_VERSION}-Linux-x86_64.sh" \
        -O /tmp/miniconda.sh && \
    echo "${MINICONDA_SHA256}  /tmp/miniconda.sh" | sha256sum --check --strict && \
    bash /tmp/miniconda.sh -b -p "${CONDA_DIR}" && \
    rm /tmp/miniconda.sh
ENV PATH="${CONDA_DIR}/bin:${PATH}"

# Actualizar Conda
RUN conda update -n base conda -y && \
    conda update --all -y

# Copiar el environment.yml
COPY environment.yml .

# Crear el entorno desde environment.yml
RUN conda env create -f environment.yml

COPY requirements.txt .

RUN conda run -n CircleNet pip install -r requirements.txt

# Activar el entorno en conda
SHELL ["conda", "run", "-n", "CircleNet", "/bin/bash", "-c"]

# cudnn.enabled
RUN sed -i "1254s/torch\.backends\.cudnn\.enabled/False/g" $(python -c "import torch; print(torch.__file__)")/nn/functional.py || true

# Instalar fastapi y sus dependencias
RUN pip install fastapi==0.83.0 pydantic==1.8.2 uvicorn[standard]

# Copiar y ejecutar el script de instalación de PyTorch 0.4.1
COPY install_pytorch041.sh /tmp/install_pytorch041.sh
RUN chmod +x /tmp/install_pytorch041.sh && \
    /tmp/install_pytorch041.sh

# Instalar COCOAPI
RUN git clone https://github.com/cocodataset/cocoapi.git /opt/cocoapi && \
    cd /opt/cocoapi/PythonAPI && \
    make && \
    python setup.py install --user

RUN conda install -y cffi

# Copiar proyecto
WORKDIR /app
COPY . .

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app/src/lib:/app/src/lib/external:/app/src/lib/models/networks/DCNv2

RUN chmod +x compiler.sh
RUN chmod +x entrypoint.sh

ENTRYPOINT ["./entrypoint.sh"]
