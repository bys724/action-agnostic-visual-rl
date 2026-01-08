# OpenCV GUI 지원 문제 해결

## 문제 상황

Docker 컨테이너에서 OpenCV GUI 함수 사용 시 오류:
```
cv2.error: The function is not implemented. Rebuild the library with Windows, GTK+ 2.x or Cocoa support
```

## 원인

pip로 설치되는 `opencv-python`은 headless 빌드로, GUI 지원이 없습니다.

## 해결 방법

### 1. 시스템 OpenCV 사용
```bash
# pip opencv 제거
pip uninstall -y opencv-python opencv-python-headless

# 시스템 패키지는 이미 설치됨 (Dockerfile에서 설치)
# python3-opencv는 GTK 지원 포함
```

### 2. Dockerfile 수정 사항
- `python3-opencv` 시스템 패키지 설치
- pip opencv-python 자동 제거
- GUI 지원 확인 코드 추가

### 3. 확인 방법
```python
import cv2
print(cv2.__version__)  # 4.5.4 (시스템 버전)
print(hasattr(cv2, 'imshow'))  # True여야 함
```

## GUI 실행

### X11 포워딩 설정 (호스트)
```bash
xhost +local:docker
```

### GUI 테스트
```bash
docker exec simpler-dev bash -c "cd /workspace && python src/test_simpler_demo.py --gui"
```

## 주의사항

1. **requirements.txt에 opencv-python 포함 금지**
   - 시스템 패키지와 충돌

2. **Docker 재빌드 필요**
   ```bash
   docker compose build eval
   ```

3. **NumPy 버전 호환성**
   - 시스템 OpenCV는 NumPy 1.x 필요
   - NumPy 2.x 설치 시 import 오류 발생

## 트러블슈팅

### GTK 경고 메시지
```
Gtk-Message: Failed to load module "canberra-gtk-module"
```
- 무시해도 됨 (오디오 관련, GUI 동작에 영향 없음)

### Accessibility bus 경고
```
dbind-WARNING: Couldn't connect to accessibility bus
```
- 무시해도 됨 (접근성 기능, GUI 동작에 영향 없음)