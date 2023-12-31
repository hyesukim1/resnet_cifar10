# model.py
- data load
- model 분기
- 학습 소스 및 학습 히스토리 저장

# train.py
- 모델 및 파라미터 저장된 json load
- args로 빼서 json별 순차 학습 진행

# model.json
- 학습 파라미터 수정할거 따로 빼둠(lr는 학습 진행 보고 줄일지 말지 결정)
- 파라미터 변경하면 file_name, output_path 변경해주기

## 추가해야할 사항
- 학습에 들어갈 때 클래스 맵핑해주기(클래스 10개니까 32 배치로 각 클래스별 꼭 3개씩 들어가게 맵핑)
