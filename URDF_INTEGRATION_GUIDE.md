# Existing OpenArm URDF Integration Guide

이 가이드는 사용자가 이미 보유하고 있는 **기존 OpenArm URDF 파일**을 새 워크스페이스에 연동하고, Launch 및 Configuration YAML 파일을 통해 로봇 모델 및 링크/관절 프레임을 설정하는 방법을 설명합니다.

---

## 1. URDF 파일 배치 경로

기존 URDF 파일(및 관련된 Mesh STL/DAE 파일)을 생성한 ROS 2 워크스페이스의 아래 경로에 배치합니다:

```text
openarm_ws/
└── src/
    └── openarm_description/
        ├── urdf/
        │   └── openarm.urdf            <-- 사용자의 기존 URDF 파일 배치
        ├── meshes/
        │   ├── base_link.stl           <-- Mesh 파일이 있는 경우 배치
        │   ├── link1.stl
        │   └── ...
        ├── config/
        │   └── openarm_teleop_config.yaml
        └── launch/
            └── arm_teleop.launch.py
```

> **참고:** URDF 내부의 `<mesh filename="package://openarm_description/meshes/..."/>` 패키지 URI 경로가 맞는지 확인해 주세요.

---

## 2. Launch 및 YAML 설정 파일 수정 방법

### (1) `openarm_teleop_config.yaml` 수정
`config/openarm_teleop_config.yaml` 파일에서 사용자의 URDF에정의된 **Base Link 이름**, **End-Effector Link 이름**, **관절(Joint) 이름** 및 **관절 한계값**을 맞춥니다:

```yaml
openarm_teleop:
  ros__parameters:
    urdf_package: "openarm_description"
    urdf_rel_path: "urdf/openarm.urdf"
    
    # 1. 링크 프레임 명칭 (사용자 URDF 내부 명칭과 일치)
    base_link_frame: "base_link"       # 예: "base_link" 또는 "openarm_base"
    end_effector_frame: "link7"       # 예: "link7" 또는 "hand_tcp" / "wrist_yaw_link"
    elbow_link_frame: "link4"          # 팔꿈치 관절 링크

    # 2. 관절 이름 7개 (순서대로 J1 ~ J7)
    joint_names:
      - "joint1"  # Shoulder Yaw
      - "joint2"  # Shoulder Pitch
      - "joint3"  # Shoulder Roll
      - "joint4"  # Elbow Pitch
      - "joint5"  # Wrist Roll
      - "joint6"  # Wrist Pitch
      - "joint7"  # Wrist Yaw
```

### (2) `arm_teleop.launch.py` 연동 템플릿
Launch 파일 내부에서 xacro/urdf를 읽어 `robot_state_publisher` 및 비전 노드에 `robot_description` 파라미터로 주입합니다:

```python
import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    # 1. URDF 파일 경로 획득
    pkg_dir = get_package_share_directory('openarm_description')
    urdf_path = os.path.join(pkg_dir, 'urdf', 'openarm.urdf')
    
    with open(urdf_path, 'r') as f:
        robot_desc = f.read()

    # 2. robot_state_publisher 실행
    rsp_node = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[{'robot_description': robot_desc}]
    )
    
    return LaunchDescription([rsp_node])
```

---

## 3. 검증 방법

1. 워크스페이스 빌드 및 환경 로드:
   ```bash
   cd ~/openarm_ws
   colcon build --packages-select openarm_description
   source install/setup.bash
   ```

2. RViz2를 통해 URDF가 정상적으로 로드되는지 확인:
   ```bash
   ros2 launch openarm_description arm_teleop.launch.py
   ```
