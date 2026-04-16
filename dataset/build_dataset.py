import pandas as pd
import json
from PIL import Image
import io

def image_to_bytes(img_path):
    with open(img_path, "rb") as f:
        return f.read()
import json
with open("/hdd/wangty/new_task/LLaMA-Factory/task/dataset/id/id_split.txt","r") as f:
    train_id,val_id,test_id = eval(f.read())
id_list=train_id+val_id+test_id
print(len(id_list))
data = []


for i in range(len(train_id)):  # 10条样本测试
    img_path = f"/mnt/nvme_share/wangty/img/tra_512/{train_id[i]}/3-4.png"
    
    sample = {
        "image": image_to_bytes(img_path),
        "captions": json.dumps({
            "0": "a cervical spine MRI sagittal T2 image",
            "1": "MRI of cervical spine showing vertebra structure"
        })
    }
    
    data.append(sample)

df = pd.DataFrame(data)
df.to_parquet("/hdd/wangty/diffuser_workdir/bagel_example/t2i_full/datapart-00000.parquet")