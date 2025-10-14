#!/bin/bash

# Setup script for RNA Folding EA Assignment
# Achal Patel - 40227663
# Run this on your machine to set up the complete environment

echo "Setting up RNA Folding EA Assignment environment..."
echo "========================================================"

# Update system
echo "Updating system packages..."
sudo apt update

# Install Docker if not present
if ! command -v docker &> /dev/null; then
    echo "Installing Docker..."
    sudo apt install -y docker.io docker-compose
    
    # Add user to docker group for sudo-free Docker access
    echo "Adding user '$USER' to docker group..."
    sudo usermod -aG docker $USER
    
    echo "Docker installed. Starting Docker service..."
    sudo systemctl start docker
    sudo systemctl enable docker
    
    # Check if user is now in docker group
    if groups $USER | grep -q '\bdocker\b'; then
        echo "User successfully added to docker group"
        echo "WARNING: You'll need to log out and back in (or restart) for group changes to take effect"
        docker_group_added=true
    else
        echo "WARNING: Failed to add user to docker group. You may need to run: sudo usermod -aG docker $USER"
        docker_group_added=false
    fi
else
    echo "Docker already installed."
    
    # Check if user is in docker group
    if groups $USER | grep -q '\bdocker\b'; then
        echo "User is already in docker group"
        docker_group_added=false
    else
        echo "Adding user '$USER' to docker group for sudo-free access..."
        sudo usermod -aG docker $USER
        echo "User added to docker group"
        echo "WARNING: You'll need to log out and back in (or restart) for group changes to take effect"
        docker_group_added=true
    fi
    
    # Make sure Docker service is running
    sudo systemctl start docker
fi

# Install Python packages
echo "Installing Python dependencies..."

# Install system packages
echo "Installing system Python packages..."
sudo apt install -y python3-numpy python3-matplotlib python3-seaborn python3-pandas python3-yaml python3-venv python3-full

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating Python virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment and install requirements
echo "Installing Python requirements in virtual environment..."
source venv/bin/activate
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
else
    echo "WARNING: requirements.txt not found. Installing essential packages..."
    pip install numpy matplotlib seaborn pandas pyyaml wandb
fi
deactivate

# Setup IPknot
echo "Setting up IPknot for RNA folding..."
if [ ! -d "ipknot" ]; then
    echo "Cloning IPknot repository..."
    git clone https://github.com/satoken/ipknot.git
fi

cd ipknot

# Create the optimized Dockerfile
echo "Creating IPknot Dockerfile..."
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

# Build Docker image and start container
echo "Building IPknot Docker image..."

# Test Docker access
docker_cmd="docker"
if ! docker ps &>/dev/null; then
    if sudo docker ps &>/dev/null; then
        echo "WARNING: Using sudo for Docker commands (group membership not active yet)"
        docker_cmd="sudo docker"
    else
        echo "ERROR: Docker is not accessible. This is a critical error."
        docker_success=false
    fi
else
    echo "Docker is accessible without sudo"
    docker_success=true
fi

if [ "$docker_success" != false ]; then
    # Build Docker image
    $docker_cmd build . -t ipknot
    
    # Start persistent container
    echo "Starting IPknot container..."
    $docker_cmd rm -f ipknot_runner 2>/dev/null || true
    $docker_cmd run -dit --name ipknot_runner -v ${PWD}:/work -w /work --entrypoint bash ipknot -c "sleep infinity"
    
    # Test container
    if $docker_cmd exec ipknot_runner echo "IPknot container test successful" &>/dev/null; then
        echo "IPknot container is running correctly"
        docker_success=true
    else
        echo "ERROR: IPknot container failed to start properly"
        docker_success=false
    fi
else
    echo "ERROR: Skipping Docker setup due to access issues"
fi

cd ..

echo ""
echo "========================================"
echo "SETUP COMPLETE - STATUS SUMMARY"
echo "========================================"

# Check Python dependencies
echo "Python Dependencies:"
if python3 -c "import numpy, matplotlib, seaborn, pandas, yaml" 2>/dev/null; then
    echo "   Core packages: INSTALLED"
    python_core=true
else
    echo "   Core packages: MISSING"
    python_core=false
fi

if source venv/bin/activate && python3 -c "import wandb" 2>/dev/null && deactivate; then
    echo "   Virtual environment: WORKING"
    python_venv=true
else
    echo "   Virtual environment: CHECK REQUIRED"
    python_venv=false
fi

# Check Docker status
echo "Docker Setup:"
if docker ps &>/dev/null 2>&1; then
    echo "   Docker access: WORKING (no sudo needed)"
    docker_status="perfect"
elif sudo docker ps &>/dev/null 2>&1; then
    echo "   Docker access: REQUIRES SUDO (group membership not active)"
    docker_status="needs_restart"
else
    echo "   Docker access: FAILED"
    docker_status="failed"
fi

if $docker_cmd ps | grep -q ipknot_runner; then
    echo "   IPknot container: RUNNING"
    ipknot_status=true
else
    echo "   IPknot container: NOT RUNNING"
    ipknot_status=false
fi

echo "========================================"

# Provide next steps
if [ "$docker_group_added" = true ]; then
    echo ""
    echo "IMPORTANT - RESTART REQUIRED:"
    echo "You were added to the docker group for sudo-free Docker access."
    echo ""
    echo "Please do ONE of the following to activate group membership:"
    echo "1. Log out and log back in (recommended)"
    echo "2. Restart your terminal/SSH session"  
    echo "3. Run: newgrp docker"
    echo "4. Reboot your system"
    echo ""
    echo "After restart, Docker commands will work without sudo!"
fi

echo ""
echo "READY TO USE:"
echo ""
echo "1. Test your setup:"
echo "   python3 scripts/test_setup.py"
echo ""
echo "2. Run assignment experiments:"
echo "   python3 src/ea_runner.py --experiment my_test"
echo ""
echo "3. Interactive testing:"
echo "   python3 test_runner.py"
echo ""

if [ "$python_venv" = false ]; then
    echo "Virtual Environment Usage:"
    echo "   source venv/bin/activate    # Activate"
    echo "   python3 src/ea_runner.py    # Run experiments"
    echo "   deactivate                  # Deactivate"
    echo ""
fi

if [ "$docker_status" = "perfect" ] && [ "$python_core" = true ] && [ "$ipknot_status" = true ]; then
    echo "ALL SYSTEMS GO! Your RNA Folding EA environment is ready!"
elif [ "$docker_status" = "needs_restart" ] && [ "$python_core" = true ]; then
    echo "ALMOST READY! Just restart your session for full Docker access."
else
    echo "SETUP INCOMPLETE! Please address the issues above."
fi

echo "======================================================"