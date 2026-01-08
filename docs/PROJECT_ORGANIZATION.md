# 프로젝트 구조 정리 완료

## MD 파일 재구성

### 이전 구조 (평면적)
```
/
├── README.md
├── CLAUDE.md
├── RESEARCH_CONTEXT.md
├── TEST_GUIDE.md
├── TEST_RESULTS.md
├── TODO_MODEL_INTEGRATION.md
└── CLEANUP_SUMMARY.md
```

### 현재 구조 (계층적)
```
/
├── README.md                    # 프로젝트 개요 및 실험 계획
├── CLAUDE.md                    # AI 어시스턴트용 개발 가이드
└── docs/
    ├── setup/                   # 설치 및 테스트
    │   ├── TEST_GUIDE.md       # 테스트 가이드
    │   └── TEST_RESULTS.md     # 테스트 결과
    ├── development/            # 개발 관련
    │   ├── TODO_MODEL_INTEGRATION.md    # 모델 통합 TODO
    │   ├── SIMPLER_MODEL_INTEGRATION.md # SimplerEnv 통합
    │   └── CLEANUP_SUMMARY.md          # 정리 요약
    └── research/              # 연구 배경
        └── RESEARCH_CONTEXT.md         # 연구 컨텍스트
```

## 정리된 파일들

### 삭제한 파일
- `patch_simpler.py` - ManiSkill2용 패치 (obsolete)
- `README_SIMPLER.md` - 중복된 README
- `src/envs/` - ManiSkill2 기반 래퍼 (obsolete)
- `DEVELOPMENT_GUIDE.md` - 오래된 개발 가이드

## 문서 개선 사항

### README.md
- 프로젝트의 본질에 집중 (연구 목표, 실험 계획)
- 구체적인 구현보다 전체적인 방향성 제시
- 4단계 실험 계획 명시

### CLAUDE.md
- AI 어시스턴트에게 필요한 핵심 정보만 포함
- 현재 환경 설정과 주요 파일 위치
- 개발 원칙과 다음 작업 제안

## 일반적인 프로젝트 구조 적용

### 표준 관례
1. **루트 레벨**: README.md, LICENSE, CONTRIBUTING.md 등 주요 파일
2. **docs/**: 모든 상세 문서를 카테고리별로 정리
3. **계층 구조**: 목적별로 하위 디렉토리 구성
   - setup: 설치 및 환경 구성
   - development: 개발 가이드 및 TODO
   - research: 연구 배경 및 논문

### 장점
- 문서 찾기 쉬움
- 프로젝트 루트 정리
- 확장성 있는 구조
- 표준적인 오픈소스 프로젝트 형태