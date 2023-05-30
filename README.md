
<div align="center">
<h1>An investigation into the effectiveness of a lightweight model for human motion prediction</h1>
<h3> <i>MINGJIE CHEN, SONG JIN </i></h3>
</div>

## Dependencies

* Pytorch 1.10.1+cu102
* Python 3.8
 
 ### Get the data
In here, we have preprocessed the dataset for you, so you don't need to download it from the official website
Directory structure: 
```shell script
dataset
|-- H3.6M_dataset
|-- cmu_mocap_dataset
|-- 3dpw_dataset
```

### Train
You can run the following code to train your own model for three datasets: H3.6M, CMU-Mocap, and 3DPW. The training results will be saved in "./train_model".
```bash
 python main_h3.6m.py
 ```
```bash
 python main_cmu.py
  ```
```bash
  python main_3dpw.py
  ```
 
 ### Test
 To test on the pretrained model, we have used the following commands:
At the same time, we also provide pre-trained models for readers to use. Of course, you can also use your own trained model.
 ```bash
 python predict_h3.6m.py
  ```
  ```bash
  python predict_cmu.py
  ```
  ```bash
   python predict_3dpw.py
  ```

