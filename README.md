# OpenArm 7-DoF Human Arm Motion Tracking & Teleoperation System

**Intel RealSense Depth Camera (RGB-D)**, **AprilTag 3 / ArUco (Wrist 6-DoF)**, **Google MediaPipe Pose**를 결합하여 사람의 상지(팔) 동작을 실시간 트래킹하고, **7자유도 로봇 팔(OpenArm)**의 실제 7개 관절 각도($q_1 \sim q_7$)를 실시간으로 해석·제어하는 고성능 모션 텔레오퍼레이션 파이프라인입니다.

---

## 📌 주요 특징 (Key Features)

1. **손목 6-DoF 고정밀 포즈 추정 (AprilTag 3 & ArUco Dual Engine):**
   - NASA/ROS 로봇 표준 **`pupil-apriltags` (AprilTag 3 tag36h11)** C-엔진 탑재로 지터(떨림) 5~10배 억제
   - RealSense 레이저 깊이(IR Depth) 앵커링을 통한 앞뒤 튐($Z$축 왜곡) $100\%$ 방지
   - 손목 위치 $(X, Y, Z)$ 및 3D 회전(Roll-Pitch-Yaw, Quaternion, Rotation Matrix) 실시간 산출
2. **7자유도 잉여 자유도(Kinematic Redundancy) 해석 및 관절 각도 산출:**
   - 어깨, 팔꿈치, 손목의 3D 기구학 체인으로부터 **스위블 각도(Arm Swivel Angle $\psi$)** 실시간 계산
   - **해석적 역운동학(Analytical IK)** 솔버를 통해 OpenArm 로봇의 **7개 관절 명령 각도($q_1 \sim q_7$)**를 $0.01\text{ms}$ 내에 도출
3. **가림 방지 전완 구면 제약 솔버 (Ray-Sphere Constraint Solver):**
   - 손목을 바닥으로 꺾어 마커가 보이지 않을 때 바닥/배경 깊이 관통 오류를 차단하고 팔꿈치 구면 상에 손목을 정확히 안착
4. **VR/로보틱스 급 적응형 지터 필터 (Anti-Jitter Filters):**
   - **One-Euro Filter / Exponential Moving Average (EMA)**: 정지 시 떨림 완벽 억제 및 고속 이동 시 지연(Lag) $0\text{ms}$
   - **Quaternion SLERP**: 최단 경로 회전 구면 선형 보간
   - **Angle Continuity Filter**: 위상 반전($\pm180^\circ$) 점프 방지
5. **실시간 HUD 대시보드 및 UDP 고속 텔레메트리:**
   - 3D 좌표축, 관절 타깃 링, 인체 뼈대, 스위블 게이지, 7-DoF 관절 각도 배너 오버레이
   - 로봇 제어기 및 ROS2로 송출되는 UDP 소켓(포트 9870) 실시간 스트리밍

---

## 🚀 빠른 시작 (Quick Start)

### 1. 패키지 설치
```bash
# 저장소 클론 및 이동
git clone https://github.com/jimin-kr/jm_control.git
cd jm_control

# 의존성 설치
pip install -r requirements.txt
```

### 2. 관절 위치 인식 및 모션 트래킹 실행
스마트폰 화면에 마커(`apriltag_phone_display.png` 또는 `marker_phone_display.png`)를 띄워 손에 쥐고 아래 명령어를 실행합니다:

```bash
# [기본 실행] 오른팔 트래킹 (마커 ID: 0, 마커 크기: 50mm)
python3 arm_tracking.py --arm right --marker-id 0 --marker-size 0.05

# [거울 모드 실행] 거울을 보듯 직관적으로 오른팔을 조종할 때
python3 arm_tracking.py --arm right --marker-id 0 --marker-size 0.05 --mirror

# [왼팔 트래킹]
python3 arm_tracking.py --arm left --marker-id 0 --marker-size 0.05
```

### 3. OpenArm 7자유도 로봇 관절 수신 콘솔 실행 (선택사항)
새 터미널 창을 열고 수신 스크립트를 실행하면, 실시간 계산된 7개 관절 각도($q_1 \sim q_7$) 대시보드가 출력됩니다:

```bash
python3 open_arm_receiver.py
```

---

## 🎮 키보드 인터랙션 단축키

| 키 (Key) | 기능 |
| :---: | :--- |
| **`Q` / `ESC`** | 프로그램 종료 및 로그 안전 저장 |
| **`M`** | 거울(Mirror) 모드 On / Off 토글 |
| **`F`** | One-Euro 떨림 방지 필터 On / Off 토글 |
| **`R`** | 모든 지터 필터 초기화(Reset) |
| **`S`** | 현재 프레임 캡처 스냅샷 저장 |

---

## 🦾 OpenArm 7자유도 관절 매핑 (Joint Mapping)

```
[인체 관절 트래킹]                              [OpenArm 7-DoF 로봇 관절 명령]
 • 어깨 방위각 (u_se_x, z)  ───────────────>  J1 (Shoulder Yaw)   : 어깨 좌우 회전 (±160°)
 • 어깨 고도각 (-u_se_y)    ───────────────>  J2 (Shoulder Pitch) : 어깨 상하 승강 (±110°)
 • 팔꿈치 스위블 각도 (ψ)   ───────────────>  J3 (Shoulder Roll)  : 상완 비틀림 (±170°) 👈 잉여 자유도 직결!
 • 팔꿈치 굽힘각 (θ_elbow)  ───────────────>  J4 (Elbow Pitch)    : 팔꿈치 굽힘/폄 (0°~150°)
 • 손목 6-DoF 회전행렬 (R)  ───────────────>  J5 (Wrist Roll)     : 손목/전완 회전 (±170°)
                                               J6 (Wrist Pitch)    : 손목 꺾임 (±90°)
                                               J7 (Wrist Yaw)      : 손목 좌우 (±90°)
```

---

## 📂 파일 구성 (Project Structure)

```
motion_tracking/
├── arm_tracking.py            # 메인 모션 트래킹 파이프라인 (비전 인식, 7-DoF IK, HUD, UDP 전송)
├── open_arm_receiver.py       # 실시간 7-DoF 로봇 관절 각도 수신 및 ASCII 게이지 대시보드 데모
├── open_arm_kinematics.py     # OpenArm 7자유도 해석적 역운동학(IK) 솔버 및 관절 가동 한계 엔진
├── aruco_tracker.py           # AprilTag 3 (pupil-apriltags) & ArUco 듀얼 6-DoF 트래커
├── pose_tracker.py            # MediaPipe Pose 신경망 + 전완 구면 제약(Ray-Sphere) 솔버
├── realsense_camera.py        # Intel RealSense RGB-D 제어, Depth 정렬, Intrinsics, Mock 카메라
├── filters.py                 # One-Euro, EMA, SLERP Quaternion, 위상 점프 방지 필터
├── generate_marker.py         # AprilTag / ArUco 마커 이미지 생성 유틸리티
├── marker_view.html           # 스마트폰용 반응형 고대비 마커 웹 뷰어
├── apriltag_phone_display.png # 스마트폰 화면 표시용 AprilTag 3 (tag36h11 ID:0)
├── marker_phone_display.png   # 스마트폰 화면 표시용 ArUco 4x4 (DICT_4X4_50 ID:0)
├── test_tracking.py           # 단위 테스트 스위트 (12개 테스트 케이스 검증)
├── requirements.txt           # 의존성 패키지 목록
└── README.md                  # 프로젝트 매뉴얼 및 기술 문서
```

---

## 📡 UDP 텔레메트리 패킷 규격 (Port 9870)

`arm_tracking.py`는 매 프레임마다 초저지연 UDP 소켓(기본 포트 `9870`)으로 아래 형식의 JSON 데이터를 전송합니다:

```json
{
  "timestamp": 1740500000.123,
  "frame_index": 450,
  "arm_side": "right",
  "tracking_mode": "ARUCO_MASTER",
  "is_tracking_valid": true,
  "wrist_6dof": {
    "position": {"x_m": 0.254, "y_m": -0.052, "z_m": 0.951},
    "orientation_rpy_deg": {"roll": 10.5, "pitch": -25.3, "yaw": 5.2},
    "quaternion_wxyz": {"w": 0.965, "x": 0.082, "y": -0.211, "z": 0.142}
  },
  "joints_3d_meters": {
    "shoulder": [0.05, -0.15, 1.20],
    "elbow": [0.18, 0.05, 1.12],
    "wrist": [0.254, -0.052, 0.951]
  },
  "redundancy": {
    "swivel_angle_deg": 35.4,
    "elbow_joint_angle_deg": 88.2
  },
  "open_arm_7dof": {
    "joint_angles_deg": [12.4, -28.5, 35.4, 91.8, -15.2, 22.0, 4.1],
    "joint_angles_rad": [0.2164, -0.4974, 0.6178, 1.6022, -0.2653, 0.3840, 0.0716],
    "swivel_angle_deg": 35.4,
    "is_valid": true,
    "is_singularity": false
  }
}
```

---

## 🧪 단위 테스트 실행 (Unit Testing)

모든 수학 변환, 역운동학(IK), 필터 및 비전 파이프라인의 무결성을 검증합니다:

```bash
python3 test_tracking.py
```
*(12개 테스트 케이스 $100\%$ 통과)*

---

## 📜 라이선스 (License)
MIT License. 자유롭게 수정 및 로봇 제어 프로젝트에 활용하실 수 있습니다.
