#!/bin/bash

# Setup script for distributed RNA folding experiments
# Run this on each machine you want to use for computation

echo "Setting up RNA Folding EA environment..."

# Update system
sudo apt update

# Install Docker if not present
if ! command -v docker &> /dev/null; then
    echo "Installing Docker..."
    sudo apt install -y docker.io docker-compose
    sudo usermod -aG docker $USER
    echo "Docker installed. You may need to log out and back in."
    echo "Starting Docker service..."
    sudo systemctl start docker
    sudo systemctl enable docker
else
    echo "Docker already installed."
    # Make sure Docker service is running
    sudo systemctl start docker
fi

# Install Python packages
echo "Installing Python dependencies..."

# First try using system packages (recommended for Ubuntu)
echo "Trying system packages first..."
sudo apt install -y python3-numpy python3-matplotlib python3-seaborn python3-pandas python3-yaml

# Check if packages are available, if not use pip with virtual environment
if ! python3 -c "import numpy, matplotlib, seaborn, pandas, yaml" 2>/dev/null; then
    echo "System packages not sufficient, setting up virtual environment..."
    sudo apt install -y python3-venv python3-full
    
    if [ ! -d "venv" ]; then
        python3 -m venv venv
    fi
    
    source venv/bin/activate
    pip install -r requirements.txt
    deactivate
    
    echo "Virtual environment created. Use 'source venv/bin/activate' before running experiments."
else
    echo "Installing wandb via pip (not available in system packages)..."
    pip3 install wandb weave --user
    echo "✓ All Python dependencies installed"
fi

# Clone IPknot and build Docker image
echo "Setting up IPknot..."
if [ ! -d "ipknot" ]; then
    git clone https://github.com/satoken/ipknot.git
fi

cd ipknot

# Create the fixed Dockerfile
cat > Dockerfile << 'EOF'
FROM satoken/viennarna:latest AS build

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      build-essential cmake ninja-build git pkg-config zlib1g-dev ca-certificates && \
    rm -rf /var/lib/apt/lists/*

RUN git clone --depth 1 https://github.com/ERGO-Code/HiGHS /tmp/HiGHS && \
    cmake -S /tmp/HiGHS -B /tmp/HiGHS/build -G Ninja \
          -DFAST_BUILD=ON -DBUILD_SHARED_LIBS=ON && \
    cmake --build /tmp/HiGHS/build --parallel && \
    cmake --install /tmp/HiGHS/build

WORKDIR /src
COPY . /src
RUN cmake -S . -B build -G Ninja \
          -DCMAKE_BUILD_TYPE=Release \
          -DENABLE_HIGHS=ON && \
    cmake --build build --parallel && \
    cmake --install build --strip

FROM debian:bookworm-slim
RUN apt-get update && \
    apt-get install -y --no-install-recommends libgomp1 zlib1g && \
    rm -rf /var/lib/apt/lists/*
COPY --from=build /usr/local/ /usr/local/
ENV LD_LIBRARY_PATH=/usr/local/lib
ENTRYPOINT ["ipknot"]
EOF

# Build Docker image
echo "Building IPknot Docker image..."

# Check if user is in docker group and docker is accessible
if groups $USER | grep &>/dev/null '\bdocker\b' && docker ps &>/dev/null; then
    echo "Docker is accessible, building image..."
    docker build . -t ipknot
    docker_success=true
elif sudo docker ps &>/dev/null; then
    echo "Using sudo for Docker commands (user not in docker group yet)..."
    sudo docker build . -t ipknot
    docker_success=true
else
    echo "Docker is not accessible. This usually means you need to restart your session."
    echo "The script will continue, but Docker setup is incomplete."
    docker_success=false
fi

# Start persistent container
echo "Starting IPknot container..."
if [ "$docker_success" = true ]; then
    if groups $USER | grep &>/dev/null '\bdocker\b' && docker ps &>/dev/null; then
        docker rm -f ipknot_runner 2>/dev/null || true
        docker run -dit --name ipknot_runner -v ${PWD}:/work -w /work --entrypoint bash ipknot -c "sleep infinity"
    else
        sudo docker rm -f ipknot_runner 2>/dev/null || true
        sudo docker run -dit --name ipknot_runner -v ${PWD}:/work -w /work --entrypoint bash ipknot -c "sleep infinity"
    fi
else
    echo "Skipping container creation due to Docker access issues."
fi

cd ..

echo ""
echo "========================================"
echo "Setup Status Summary:"
echo "========================================"

# Check Python dependencies
if python3 -c "import numpy, matplotlib, seaborn, pandas, yaml, wandb" 2>/dev/null; then
    echo "✓ Python dependencies: INSTALLED"
    python_ok=true
else
    echo "✗ Python dependencies: MISSING"
    python_ok=false
fi

# Check Docker status
if docker ps &>/dev/null 2>&1 || sudo docker ps &>/dev/null 2>&1; then
    echo "✓ Docker: WORKING"
    docker_ok=true
else
    echo "✗ Docker: NEEDS RESTART"
    docker_ok=false
fi

echo "========================================"

if [ "$docker_ok" = false ]; then
    echo ""
    echo "DOCKER SETUP INCOMPLETE:"
    echo "You were added to the docker group, but need to restart your session."
    echo ""
    echo "Please do ONE of the following:"
    echo "1. Log out and log back in"
    echo "2. Restart your terminal"
    echo "3. Run: newgrp docker"
    echo "4. Reboot your system"
    echo ""
    echo "Then run this script again to complete Docker setup."
fi

if [ "$python_ok" = false ]; then
    echo ""
    echo "PYTHON SETUP INCOMPLETE:"
    echo "If using virtual environment, activate it before running experiments:"
    echo "source venv/bin/activate"
fi

echo ""
if [ "$docker_ok" = true ] && [ "$python_ok" = true ]; then
    echo " Setup completed successfully!"
    echo ""
    echo "You can now run experiments using:"
    echo "python3 src/ea_runner.py config/device_experiments.yml --device odin"
    echo ""
    echo "To test your setup:"
    echo "python3 test_setup.py"
else
    echo " Setup partially completed. Please address the issues above."
    echo ""
    echo "After fixing issues, test your setup with:"
    echo "python3 test_setup.py"
fi