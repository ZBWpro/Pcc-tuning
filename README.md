# Read Me

Pcc-tuning: Pcc-tuning: Achieving Spearman's Correlation Scores Beyond 87.5

_our paper is the first to propose and substantiate the performance upper bound of contrastive learning methods. Additionally, Pcc-tuning is the inaugural method capable of achieving Spearman's correlation scores above 87 on standard STS tasks, marking a significant advancement in the field._

***

## Results

![main-table](images/main-table.png)

***

## Data

- Stage one: `nli_for_simcse.csv`
- Stage two: `stsb-sickr-train.jsonl`
- Link: https://drive.google.com/drive/folders/1qDEFOEkkZfFn9RfFi2ibMo_uuIDk-Vr1?usp=sharing

***

## Checkpoints

- Link: https://drive.google.com/drive/folders/1jTtORca1ihHlpdFfSpwsSgmcL6KrukjM?usp=sharing

***

## Quick Start

- Python Version: 3.9.18

- Install Dependencies

  ```bash
  cd code
  pip install -r requirements.txt
  ```

- Download SentEval

  ```bash
  cd SentEval/data/downstream/
  bash download_dataset.sh
  cd -
  cd ./data
  bash download_nli.sh
  cd -
  ```

- stage one

  ```bash
  cd code
  nohup torchrun --nproc_per_node=4 train.py > nohup.out &
  ```

- stage two

  ```bash
  cd code
  nohup torchrun --nproc_per_node=4 tune.py > nohup.out &
  ```

## Acknowledgement

- Our code is based on PromptEOL
