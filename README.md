ADNI Classification / Regression Tasks

현재 디렉토리에 `adni_t1s_baseline` 디렉토리를 두세요.
5-fold cross-validation을 사용해 모델 학습 및 추론을 진행합니다.

코드 실행 예시는 아래와 같습니다.
```bash
python3 main_cv.py --pretrained_path None --train_num 300 --task_name AD/CN --layer_control tune_all --random_seed 0
python3 main_cv.py --pretrained_path None --train_num 300 --task_name AD/MCI --layer_control tune_all --random_seed 0
python3 main_cv.py --pretrained_path None --train_num 300 --task_name MCI/CN --layer_control tune_all --random_seed 0
```

- [ ] 코드 구현 이해하기
- [ ] 다양한 hyperparameter로 실험하기
