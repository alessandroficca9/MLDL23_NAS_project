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

To run the search algorithm, use the following code. The parameters are:
- algo: type of algorithm. You can choose between "random_search", "ea_search" (evolutionary algorithm), "our_cnn" (manually design cnn)
- max_flops: constraint about number of flops
- max_params: constraint about number of parameters
- initial_pop: initial population size (needed if you choose "ea_search")
- generation_ea: number of steps of evolutionary algorithm (needed if you choose "ea_search")
- n_random: number of steps of random search (needed if you choose "random_search")
- save: if True the result model
```bash
python run_search.py \
  --algo ea_search
  --max_flops 200000000
  --max_params 2500000
  --inital_pop 25
  --generation_ea 100
  --save True
```


