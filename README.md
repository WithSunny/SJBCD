
<div align="center">
  <p>
    <b>SJBCD</b>
  </p>
  <p>
	     <i>SJBCD is designed to detect java bytecode similarities. SJBCD extracts opcode sequence from bytecode file, vectorizes opcode with Glove, and constructs Siamese neural network based on GRU for supervised training. The trained network is then used to detect code clones.</i>
  </p>
</div>

---
This code draws on the idea of the code of this link: https://github.com/zqhZY/semanaly
### The SJBCD Model 
<img src="https://github.com/WithSunny/SJBCD/blob/PAP/something_files/Dfig5.drawio.svg" alt="模型示意图" width="750" height="550">.

## Install 🐙
It is recommended that you install a conda environment and then install the dependent packages with the following command：
```
conda create -n SJBCD -y python==3.7.19 && conda activate SJBCD
pip install -r requirements.txt
```
ps:If you don't have a GPU, you can also install it with the command pip install tersorflow==1.11.0 , running on the CPU
## Usage 💡
1. git clone the project.
```
git clone https://github.com/zzjss12/DeepBCCD.git -d your_profile
```
2. Go inside the project folder(IDE) and open your terminal.
3. See  [Install](##install) to install the environment.
4. run the command `python run.py --train true --test true` to start.

## Exemples 🖍
We trained with the **A5000** GPU.
```
python run.py --train true --test true --w2v_dim 100 --batch_size 512--max_block_seq 20--num_block 20 --iter_level 5 
```

## Documentation 📄
For a more detailed description of the contents of DeepBCCD, please refer to our paper-----

## Datasets 👩‍💻
For the datasets, we used the datasets **BinaryCorp-3M**（https://github.com/vul337/jTrans) in the Jtrans paper. To conform to the input format according to the DeepBCCD model, We re-extracted the binary function set from the source binary and formed the **dataset_train.csv** and **dataset_test.csv** datasets，which are also essentially derived from Binarycorp-3M.
For the **BinaryCrop-26M** dataset, we will try it in the future because it requires a larger training resource。					

The dataset used in DeepBCCD [download](https://efss.qloud.my/index.php/s/a2B2S9rNwdXkmBo).
## Evaluation 🍰
#### The ROC curve is belowed:
![best_test_roc](https://github.com/zzjss12/assets/blob/Binary-code-clone/best_test_roc.png)

## Maintainers 👷
* @zzjss12

## License ⚖️
GPL

---
<div align="center">
	<b>
		<a href="https://www.npmjs.com/package/get-good-readme">File generated with get-good-readme module</a>
	</b>
</div>
