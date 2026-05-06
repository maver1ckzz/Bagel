python infer_pipeline_batch.py \
  --input-json /hdd/wangty/diffuser_workdir/bagel/result/dataset/zyzg_test.json \
  --pipeline-config /home/wangty/github/Bagel/configs/inference_pipeline.example.yaml \
  --output-dir /hdd/wangty/diffuser_workdir/bagel/result/zjxxz_test \
  --model-path /hdd/wangty/model/BAGEL-7B-MoT \
  --checkpoint-path /hdd/wangty/diffuser_workdir/bagel/workdir/zyzg/4500/model.safetensors \
  --gpus 0,1 \
  --batch-size 32,32 \
#   --start-round stage2_diagnosis \
#   --skip-existing-stage-output