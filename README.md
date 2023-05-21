# MLDL23_NAS_project

If you need to download the data, run 

```bash
python download_coco_data.py
```
Then, to create annotations 
```bash
TRAIN_ANNOTATIONS_FILE="COCOdataset/annotations/instances_train2017.json"
VAL_ANNOTATIONS_FILE="COCOdataset/annotations/instances_val2017.json"
DIR="COCOdataset/annotations/"
!python visualwakewords/scripts/create_coco_train_minival_split.py \
  --train_annotations_file="{TRAIN_ANNOTATIONS_FILE}" \
  --val_annotations_file="{VAL_ANNOTATIONS_FILE}" \
--output_dir="{DIR}"
```

```bash
MAXITRAIN_ANNOTATIONS_FILE="COCOdataset/annotations/instances_maxitrain.json"
MINIVAL_ANNOTATIONS_FILE="COCOdataset/annotations/instances_minival.json"
VWW_OUTPUT_DIR="visualwakewords"
!python visualwakewords/scripts/create_visualwakewords_annotations.py \
  --train_annotations_file="{MAXITRAIN_ANNOTATIONS_FILE}" \
  --val_annotations_file="{MINIVAL_ANNOTATIONS_FILE}" \
  --output_dir="{VWW_OUTPUT_DIR}" \
  --threshold=0.005 \
  --foreground_class='person'
```



