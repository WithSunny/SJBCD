
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
PS:If you don't have a GPU, you can also install it with the command pip install tersorflow==2.11.0 , running on the CPU.

## Usage 💡
1. git clone the project.
```
git clone https://github.com/WithSunny/SJBCD.git -d your_profile
```
2. Go inside the project folder(IDE) and open your terminal.
3. See  [Install](##install) to install the environment.
4. run the command 'python sjbcd.py' or 'python sjbcd_cos.py' to start training.
5. run the command 'python evaluation.py' to test.

## Documentation 📄
For a more detailed description of the contents of SJBCD, please refer to our paper(Please wait).

## Datasets 👩‍💻
The method employs a dataset obtained from **BigCloneBench**. From this dataset, we extract the compilable data, resulting in two distinct datasets: **CompiledBCB_source** and **CompiledBCB_opcode**.
The “CompiledBCB_opcode” directory contains the bytecode dataset, which is specifically utilized by the SJBCD method. On the other hand, the “CompiledBCB_source” directory stores the dataset consisting of bytecode source code. This dataset is primarily employed by other methods(**TBCCD**,**Nicad**,**ASTNN**,**FA-AST**,**Code-Token-Learner**).			

## Evaluation 🍰
| Method                | Precision | Recall | F1-score |
|-----------------------|-----------|--------|----------|
| SJBCD(ours)           |   0.998   | 0.997  |  0.997   |
| SJBCD-cos(ours)       |   0.996   | 0.997  |  0.996   |
| TBCCD                 |    0.9    | 0.915  |  0.908   |
| TBCCD+token           |   0.98    | 0.953  |  0.966   |
| TBCCD+token-type      |   0.976   | 0.964  |   0.97   |
| TBCCD+token+PACE      |   0.971   | 0.957  |  0.964   |
| Nicad                 |   0.636   | 0.005  |   0.01   |
| ASTNN                 |   0.992   | 0.997  |  0.995   |
| Code-Token-Learner    |   0.984   | 0.933  |  0.958   |
| FA-AST                |   0.988   | 0.988  |  0.988   |

## Maintainers 👷
* @WithSunny

## License ⚖️
GPL

---
<div align="center">
	<b>
		<a href="https://www.npmjs.com/package/get-good-readme">File generated with get-good-readme module</a>
	</b>
</div>
