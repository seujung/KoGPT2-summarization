# KoGPT2-summarization

## Requirements
```
pytorch==1.8.1
transformers==4.5.1
pytorch-lightning==1.1.0
streamlit==0.72.0
```
## Data
[KoBART-summarization](https://github.com/seujung/KoBART-summarization) 에서 사용한 Data와 동일한 데이터를 사용함

- 학습 데이터에서 임의로 Train / Test 데이터를 생성함
- 데이터 탐색에 용이하게 tsv 형태로 데이터를 변환함
- Data 구조
    - Train Data : 34,242
    - Test Data : 8,501
- default로 data/train.tsv, data/test.tsv 형태로 저장함
  
| news  | summary |
|-------|--------:|
| 뉴스원문| 요약문 | 

## How to Train
- KoGPT2 summarization fine-tuning
```
pip install -r requirements.txt

python train.py  --gradient_clip_val 1.0 --max_epochs 60 --default_root_dir logs  --gpus 1 --batch_size 4 --num_workers 4 --max_len 1024
```
## Generation Sample
| ||Text|
|-------|:--------|:--------|
|1|Label|지난 19일 고흥군에 따르면 송귀근 군수와 주민 등 300여 명이 참석한 가운데 귀농·귀촌인의 안정적인 정착을 돕고 영농교육 등을 담당하기 위해 지난해부터 추진했던 ‘귀농귀촌 행복학교'의 개교 행사를 했다|
|1|KoGPT|지난 19일 옛 망주초등학교에서 송귀근 고흥군수와 송우섭 군의회 의장, 귀농·귀촌인, 주민 등 300여명이 참석한 가운데 고흥 귀농귀촌행복학교 개교 행사가 열렸으며, 군은 폐교된 망주초등학교의 기존 시설을 최대한 활용하여 연면적 702m2, 2층 규모의 건물에 교육장, 체험장, 체류형 주택시설 등 8억여 원을 투입해 최근 귀농귀촌행복학교를 준공했다.|


## Model Performance
- Test Data 기준으로 rouge score를 산출함
- Score 산출 방법은 Dacon 한국어 문서 생성요약 AI 경진대회 metric을 활용함
  
<table>
    <thead>
        <tr>
            <th> </th>
            <th colspan=2>rouge-1</th>
            <th colspan=2>rouge-2</th>
            <th colspan=2>rouge-l</th>
        </tr>
        <tr>
            <th></th>
            <th>KoBART</th>
            <th>KoGPT2</th>
            <th>KoBART</th>
            <th>KoGPT2</th>
            <th>KoBART</th>
            <th>KoGPT2</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>Precosion</td>
            <td>0.515</td>
            <td>0.516</td>
            <td>0.351</td>
            <td>0.330</td>
            <td>0.415</td>
            <td>0.423</td>
        </tr>
        <tr>
            <td>Recall</td>
            <td>0.538</td>
            <td>0.471</td>
            <td>0.359</td>
            <td>0.298</td>
            <td>0.440</td>
            <td>0.383</td>
        </tr>
        <tr>
            <td>F1</td>
            <td>0.505</td>
            <td>0.423</td>
            <td>0.340</td>
            <td>0.301</td>
            <td>0.415</td>
            <td>0.386</td>
        </tr>
    </tbody>
</table>

## Reference
- [KoGPT2](https://github.com/SKT-AI/KoGPT2)
- [KoGPT2-chitchat](https://github.com/haven-jeon/KoGPT2-chatbot)
- [KoBART-summarization](https://github.com/seujung/KoBART-summarization)