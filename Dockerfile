# Dockerfile for Sports Tournament Scheduling MIP solver
FROM ubuntu:22.04

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    python3-venv \
    build-essential \
    wget \
    curl \
    unzip \
    tar \
    ca-certificates \
    dos2unix \
    git \
    libgl1-mesa-glx \
    libegl1-mesa \
    libxrandr2 \
    libxss1 \
    libxcursor1 \
    libxcomposite1 \
    libasound2 \
    libxi6 \
    libxtst6 \
    libglib2.0-0 \
    libgtk-3-0 \
    glpk-utils \
    libglpk-dev \
    coinor-cbc \
    coinor-clp \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install MiniZinc 2.9.3
RUN cd /tmp && \
    wget -q https://github.com/MiniZinc/MiniZincIDE/releases/download/2.9.3/MiniZincIDE-2.9.3-bundle-linux-x86_64.tgz && \
    tar -xzf MiniZincIDE-2.9.3-bundle-linux-x86_64.tgz && \
    mv MiniZincIDE-2.9.3-bundle-linux-x86_64 /opt/minizinc && \
    chmod -R +x /opt/minizinc/bin/ && \
    rm -f MiniZincIDE-2.9.3-bundle-linux-x86_64.tgz

# Download and install cvc5 1.3.0
RUN cd /tmp && \
    wget -q https://github.com/cvc5/cvc5/releases/download/cvc5-1.3.0/cvc5-Linux-x86_64-static.zip && \
    unzip -q cvc5-Linux-x86_64-static.zip && \
    mkdir -p /opt/cvc5/bin && \
    mv cvc5-Linux-x86_64-static/bin/cvc5 /opt/cvc5/bin/ && \
    chmod +x /opt/cvc5/bin/cvc5 && \
    rm -rf cvc5-Linux-x86_64-static.zip cvc5-Linux-x86_64-static

# Install AMPL Community Edition
WORKDIR /opt
# Copy and extract AMPL
COPY ampl.linux64.tgz /tmp/
RUN cd /tmp && \
    tar -xzf ampl.linux64.tgz && \
    echo "Contents after extraction:" && \
    ls -la && \
    # Find the actual directory name
    AMPL_DIR=$(find . -maxdepth 1 -type d -name "*ampl*" | head -1) && \
    echo "Found AMPL directory: $AMPL_DIR" && \
    mv "$AMPL_DIR" /opt/ampl && \
    chmod +x /opt/ampl/* && \
    rm -f ampl.linux64.tgz

# Create symbolic links for python and pip
RUN ln -sf /usr/bin/python3 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip

# Set environment variables
ENV PATH="/opt/ampl:/opt/minizinc/bin:/opt/cvc5/bin:${PATH}"
ENV AMPL_LICENSE_FILE="/opt/ampl/ampl.lic"
ENV PYTHONPATH="/app"
ENV PYTHONUNBUFFERED=1
ENV CVC5_BIN="/opt/cvc5/bin/cvc5"

# Install HiGHS
#RUN cd /tmp && \
#    wget -q https://github.com/ERGO-Code/HiGHS/releases/download/v1.7.2/highs-v1.7.2-x86_64-pc-linux-gnu.tar.gz && \
#    tar -xzf highs-v1.7.2-x86_64-pc-linux-gnu.tar.gz && \
#    mv highs-v1.7.2-x86_64-pc-linux-gnu/bin/* /opt/ampl/ && \
#    rm -rf highs-v1.7.2-x86_64-pc-linux-gnu* && \
#    chmod +x /opt/ampl/highs

# Create directories for commercial solvers (Gurobi & CPLEX)
#RUN mkdir -p /opt/solvers

# Copy solver installers and licenses (you'll need to provide these)
# Uncomment and modify these lines when you have the installers:

COPY licenses/ampl.lic /opt/ampl/ampl.lic
RUN chmod 644 /opt/ampl/ampl.lic

# Allow the -v command to fail without stopping the build
RUN echo "AMPL installation complete" && \
    echo "Available solvers: cbc, highs, cplex, gurobi, and many more" && \
    ls -la /opt/ampl/ | grep -E "(cbc|highs|cplex|gurobi)" && \
    /opt/ampl/ampl -v || echo "AMPL version check completed (license may need renewal)"

WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy project files (force rebuild of this layer)
RUN echo "Cache bust: $(date)" > /tmp/cachebust
COPY . .

# Convert Windows line endings to Unix for MiniZinc files
RUN find . -type f \( -name "*.mzn" -o -name "*.dzn" \) -exec dos2unix {} \;

# Create results directory
RUN mkdir -p res/CP res/SAT res/SMT res/MIP

# Verify installations
RUN python --version && \
    pip --version && \
    minizinc --version && \
    minizinc --solvers && \
    /opt/cvc5/bin/cvc5 --version

# Set default command
CMD ["/bin/bash"]