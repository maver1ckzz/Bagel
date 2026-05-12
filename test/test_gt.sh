python infer_pipeline_batch.py \
  --input-json /hdd/wangty/diffuser_workdir/bagel/result/dataset/zyzg_gt_test.json \
  --pipeline-config /home/wangty/github/Bagel/configs/text.yaml \
  --output-dir /hdd/wangty/diffuser_workdir/bagel/result/zyzg_gt_test_3000 \
  --model-path /hdd/wangty/model/BAGEL-7B-MoT \
  --checkpoint-path /hdd/wangty/diffuser_workdir/bagel/workdir/zyzg/3000/model.safetensors \
  --gpus 0 \
  --batch-size 32 \
#   --start-round stage2_diagnosis \
#   --skip-existing-stage-output