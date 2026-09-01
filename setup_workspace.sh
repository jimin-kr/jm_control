#!/usr/bin/env bash
# ==============================================================================
# OpenArm 7-DoF ROS 2 Workspace & Dependency Setup Script
# Baseline References:
#   - Teleop & OpenArm Control: https://github.com/ahsanali555/openarm_teleoperation.git
#   - Vision Tracking & Retargeting: https://github.com/dexsuite/dex-retargeting.git
# ==============================================================================

set -e

echo "[OpenArm Teleop Setup] Starting OpenArm ROS 2 Workspace Setup..."

# 1. Define Workspace Directory Structure
WS_DIR="${HOME}/openarm_ws"
SRC_DIR="${WS_DIR}/src"

mkdir -p "${SRC_DIR}"
echo "[OpenArm Teleop Setup] Created workspace directory: ${WS_DIR}"

# 2. Clone Required Repositories
cd "${SRC_DIR}"

if [ ! -d "openarm_teleoperation" ]; then
    echo "[OpenArm Teleop Setup] Cloning openarm_teleoperation repository..."
    git clone https://github.com/ahsanali555/openarm_teleoperation.git
else
    echo "[OpenArm Teleop Setup] openarm_teleoperation already exists. Updating..."
    cd openarm_teleoperation && git pull && cd ..
fi

if [ ! -d "dex-retargeting" ]; then
    echo "[OpenArm Teleop Setup] Cloning dex-retargeting repository..."
    git clone https://github.com/dexsuite/dex-retargeting.git
else
    echo "[OpenArm Teleop Setup] dex-retargeting already exists."
fi

# Create custom description package placeholder if not existing
if [ ! -d "openarm_description" ]; then
    mkdir -p openarm_description/urdf openarm_description/config openarm_description/launch
    cat << 'EOF' > openarm_description/package.xml
<?xml version="1.0"?>
<?xml-model href="http://download.ros.org/schema/package_format3.xsd" schematypens="http://www.w3.org/2001/XMLSchema"?>
<package format="3">
  <name>openarm_description</name>
  <version>0.1.0</version>
  <description>OpenArm Custom URDF and Description Package</description>
  <maintainer email="user@todo.todo">OpenArm Developer</maintainer>
  <license>Apache-2.0</license>
  <buildtool_depend>ament_cmake</buildtool_depend>
  <exec_depend>robot_state_publisher</exec_depend>
  <exec_depend>rviz2</exec_depend>
  <export>
    <build_type>ament_cmake</build_type>
  </export>
</package>
EOF

    cat << 'EOF' > openarm_description/CMakeLists.txt
cmake_minimum_required(VERSION 3.8)
project(openarm_description)

find_package(ament_cmake REQUIRED)

install(DIRECTORY urdf config launch
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
EOF
fi

# 3. Install Python Dependencies
echo "[OpenArm Teleop Setup] Installing required Python libraries..."
python3 -m pip install --upgrade pip
python3 -m pip install \
    pyrealsense2 \
    mediapipe \
    opencv-python \
    numpy \
    scipy \
    transforms3d \
    pinocchio \
    filterpy \
    pyyaml

# Install dex-retargeting in editable mode if setup.py exists
if [ -f "${SRC_DIR}/dex-retargeting/setup.py" ] || [ -f "${SRC_DIR}/dex-retargeting/pyproject.toml" ]; then
    echo "[OpenArm Teleop Setup] Installing dex-retargeting package..."
    python3 -m pip install -e "${SRC_DIR}/dex-retargeting" || true
fi

# 4. Install ROS 2 Dependencies via rosdep
cd "${WS_DIR}"
echo "[OpenArm Teleop Setup] Updating rosdep & installing ROS 2 system packages..."
if command -v rosdep &> /dev/null; then
    rosdep update || true
    rosdep install --from-paths src --ignore-src -r -y || true
else
    echo "[OpenArm Teleop Setup] Warning: rosdep not found. Ensure ROS 2 desktop packages are installed."
fi

# 5. Build Workspace with colcon
echo "[OpenArm Teleop Setup] Building workspace using colcon..."
if command -v colcon &> /dev/null; then
    colcon build --symlink-install
    echo ""
    echo "=========================================================================="
    echo " OpenArm Workspace Setup Complete!"
    echo " Source workspace with: source ${WS_DIR}/install/setup.bash"
    echo "=========================================================================="
else
    echo "[OpenArm Teleop Setup] Warning: colcon tool not found. Source your ROS 2 environment first."
fi
