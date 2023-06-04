# MLDL23_NAS_project

## Download the data
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
## Run the project
To run the search algorithm, use the following code. The parameters are:
- algo: type of algorithm. You can choose between "random_search", "ea_search" (evolutionary algorithm), "our_cnn" (manually design cnn)
- max_flops: constraint about number of flops
- max_params: constraint about number of parameters
- initial_pop: initial population size (needed if you choose "ea_search")
- generation_ea: number of steps of evolutionary algorithm (needed if you choose "ea_search")
- n_random: number of steps of random search (needed if you choose "random_search")
- save: if True the result model is stored in the file "model.pth"

```bash
python run_search.py \
  --algo ea_search
  --max_flops 200000000
  --max_params 2500000
  --inital_pop 25
  --generation_ea 100
  --save True
```

To train the model on the visualwakeword dataset, run the following code. The parameters are
- model: path of file in which is stored the model ("model.pth" as default)
- root_data: path of the dataset folder
- ann_train: path of the annotations train file
- ann_val: path of the annotations validation file
- batch_size: size of the batch for the training phase
- learning_rate: 
- momentum:
- epochs
- weight_decay:

```bash
python run_train.py \
  --model "model.pth"
  --root_data "COCOdataset/all2017"
  --ann_train "visualwakewords/instances_train.json"
  --ann_val "visualwakewords/instances_val.json"
  --batch_size 64
  --learning_rate 0.1
  --momentum 0.9
  --epochs 10
  --weight_decay 0.000001
```

