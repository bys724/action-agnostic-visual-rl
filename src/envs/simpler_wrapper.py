"""
SIMPLER 환경 래퍼
우리 모델과 SIMPLER 환경 간의 인터페이스
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'third_party', 'SimplerEnv'))

import numpy as np
from typing import Dict, Tuple, Optional, Any
import simpler_env
from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict


class SimplerEnvWrapper:
    """SIMPLER 환경 래퍼"""
    
    def __init__(self, env_name: str, image_size: Tuple[int, int] = (224, 224)):
        """
        Args:
            env_name: SIMPLER 환경 이름
            image_size: 이미지 크기 (H, W)
        """
        self.env_name = env_name
        self.image_size = image_size
        self.env = simpler_env.make(env_name)
        
        # 로봇 타입 확인
        if 'google_robot' in env_name:
            self.robot_type = 'google_robot'
        elif 'widowx' in env_name:
            self.robot_type = 'widowx'
        else:
            self.robot_type = 'unknown'
        
        self.current_instruction = None
        self.current_obs = None
        
    def reset(self) -> Tuple[np.ndarray, str, Dict]:
        """환경 리셋
        
        Returns:
            image: RGB 이미지 (H, W, 3)
            instruction: 언어 지시사항
            info: 추가 정보
        """
        self.current_obs, reset_info = self.env.reset()
        self.current_instruction = self.env.get_language_instruction()
        
        image = self.get_image()
        
        return image, self.current_instruction, reset_info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """환경 스텝
        
        Args:
            action: 7차원 액션 [dx, dy, dz, rx, ry, rz, gripper]
            
        Returns:
            image: 다음 이미지
            reward: 보상
            done: 종료 여부
            truncated: 시간 초과 여부
            info: 추가 정보
        """
        # 액션 클리핑 (안전을 위해)
        action = np.clip(action, self.env.action_space.low, self.env.action_space.high)
        
        self.current_obs, reward, done, truncated, info = self.env.step(action)
        
        # 새로운 지시사항 확인 (긴 시간 작업)
        new_instruction = self.env.get_language_instruction()
        if new_instruction != self.current_instruction:
            self.current_instruction = new_instruction
            info['new_instruction'] = new_instruction
        
        image = self.get_image()
        
        return image, reward, done, truncated, info
    
    def get_image(self) -> np.ndarray:
        """현재 관찰에서 이미지 추출
        
        Returns:
            RGB 이미지 (H, W, 3) uint8
        """
        if self.current_obs is None:
            raise ValueError("환경이 리셋되지 않았습니다")
        
        image = get_image_from_maniskill2_obs_dict(self.env, self.current_obs)
        
        # 리사이즈 필요시 (추후 구현)
        # if image.shape[:2] != self.image_size:
        #     image = cv2.resize(image, (self.image_size[1], self.image_size[0]))
        
        return image
    
    def get_instruction(self) -> str:
        """현재 언어 지시사항 반환"""
        return self.current_instruction
    
    def is_final_subtask(self) -> bool:
        """최종 서브태스크인지 확인"""
        return self.env.is_final_subtask()
    
    def advance_to_next_subtask(self):
        """다음 서브태스크로 진행"""
        self.env.advance_to_next_subtask()
        self.current_instruction = self.env.get_language_instruction()
    
    def close(self):
        """환경 종료"""
        self.env.close()
    
    @property
    def action_space(self):
        """행동 공간"""
        return self.env.action_space
    
    @property
    def observation_space(self):
        """관찰 공간"""
        return self.env.observation_space


class ActionConverter:
    """다양한 행동 형식 간 변환"""
    
    @staticmethod
    def quaternion_to_axis_angle(quat: np.ndarray) -> np.ndarray:
        """쿼터니언을 축-각도로 변환
        
        Args:
            quat: 쿼터니언 [x, y, z, w] 또는 [w, x, y, z]
            
        Returns:
            축-각도 [rx, ry, rz]
        """
        # 쿼터니언 정규화
        quat = quat / np.linalg.norm(quat)
        
        # [w, x, y, z] 형식 가정
        if quat.shape[-1] == 4:
            if np.abs(quat[0]) > np.abs(quat[3]):  # 첫 번째가 w일 가능성
                w, x, y, z = quat
            else:  # 마지막이 w일 가능성
                x, y, z, w = quat
        else:
            raise ValueError(f"쿼터니언은 4차원이어야 합니다: {quat.shape}")
        
        # 축-각도 계산
        angle = 2 * np.arccos(np.clip(w, -1, 1))
        
        if angle < 1e-6:  # 회전 없음
            return np.zeros(3)
        
        axis = np.array([x, y, z]) / np.sin(angle / 2)
        axis_angle = axis * angle
        
        return axis_angle
    
    @staticmethod
    def axis_angle_to_quaternion(axis_angle: np.ndarray) -> np.ndarray:
        """축-각도를 쿼터니언으로 변환
        
        Args:
            axis_angle: 축-각도 [rx, ry, rz]
            
        Returns:
            쿼터니언 [x, y, z, w]
        """
        angle = np.linalg.norm(axis_angle)
        
        if angle < 1e-6:  # 회전 없음
            return np.array([0, 0, 0, 1])
        
        axis = axis_angle / angle
        half_angle = angle / 2
        
        sin_half = np.sin(half_angle)
        cos_half = np.cos(half_angle)
        
        quat = np.array([
            axis[0] * sin_half,
            axis[1] * sin_half,
            axis[2] * sin_half,
            cos_half
        ])
        
        return quat
    
    @staticmethod
    def convert_gripper_action(gripper: float, robot_type: str) -> float:
        """그리퍼 액션 변환
        
        Args:
            gripper: 그리퍼 값
            robot_type: 로봇 타입 ('google_robot', 'widowx')
            
        Returns:
            변환된 그리퍼 값
        """
        # 로봇별 그리퍼 규칙이 다를 수 있음
        # 예: Google Robot은 1=닫기, WidowX는 0=닫기 등
        # 실제 규칙은 테스트를 통해 확인 필요
        
        return gripper