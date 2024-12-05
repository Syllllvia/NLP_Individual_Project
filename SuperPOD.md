```login
mfuai@slogin-02:~$ module load Anaconda3/2023.09-0
mfuai@slogin-02:~$ conda init
(base) mfuai@slogin-02:~$ module load cuda12.2
(base) mfuai@slogin-02:~$ module list
Currently Loaded Modulefiles:
 1) slurm/slurm/23.02.6   2) Anaconda3/2023.09-0   3) cuda12.2/toolkit/12.2.2 

// 配置pytorch环境 https://blog.csdn.net/Friedrichor/article/details/127721828
mfuai@slogin-02:~$ conda create -n pytorch python=3.9
// 关掉服务器重进
(base) mfuai@slogin-02:~$ module load Anaconda3
// 进入为pytorch创建的conda环境
(base) mfuai@slogin-02:~$ conda activate pytorch
// 服务器CUDA12.2
(pytorch) mfuai@slogin-02:~$ conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
(pytorch) mfuai@slogin-02:~$ python                                                    
Python 3.9.20 (main, Oct  3 2024, 07:27:41)                                            
[GCC 11.2.0] :: Anaconda, Inc. on linux                                                
Type "help", "copyright", "credits" or "license" for more information.                 
>>> import torch
>>> torch.__version__
'2.5.1'                                                        
>>> torch.cuda.is_available()                                                         
False  

(base) mfuai@slogin-02:~$ nvcc --version
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2023 NVIDIA Corporation
Built on Tue_Aug_15_22:02:13_PDT_2023
Cuda compilation tools, release 12.2, V12.2.140
Build cuda_12.2.r12.2/compiler.33191640_0

cd /home/mfuai/MSBD5018
// 创建一个运行test.py的脚本文件
touch test.sbatch
// 编辑test.sbatch
nano test.sbatch
```

``` test.sbatch
#!/bin/bash
#SBATCH --job-name=test          # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --gpus=1                 # number of GPUs per node(only valid under large/normal partition)
#SBATCH --time=00:20:00          # total run time limit (HH:MM:SS)
#SBATCH --partition=normal  # partition(large/normal/cpu) where you submit
#SBATCH --account=mscbdt2024      # only require for multiple projects

module purge                     # clear environment modules inherited from submission
module load Anaconda3/2023.09-0  # load the exact modules required

python test.py
```
```slurm提交作业
chmod +x test.sbatch

// 查询可用partition
sinfo

//查询可用account
scontrol show partition=normal

// 提交任务
sbatch --wait -o slurm.out test.sbatch

// 查询任务状态
cat slurm.out

//测试pytorch是否可用 https://hpc.pku.edu.cn/ug/soft/pytorch/

// 查询任务状态
scontrol show job <jobID>
```

```连接vscode
mfuai@slogin-01:~$ curl -Lk 'https://code.visualstudio.com/sha/download?build=stable&os=cli-alpine-x64' --output vscode_cli.tar.gz
  % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                                 Dload  Upload   Total   Spent    Left  Speed
100   162  100   162    0     0    447      0 --:--:-- --:--:-- --:--:--   448
100 8834k  100 8834k    0     0  8717k      0  0:00:01  0:00:01 --:--:-- 29.0M
mfuai@slogin-01:~$ tar -xf vscode_cli.tar.gz
mfuai@slogin-01:~$ srun --partition=normal --account=mdcbdt2024 --gres=gpu:1 --pty $SHELL
srun: error: Interactive job is limited to 480 minutes.
srun: error: For longer runs, please submit a batch job using `sbatch`.
srun: error: Unable to allocate resources: Invalid account or account/partition combination specified
mfuai@slogin-01:~$ srun --partition=normal --account=mscbdt2024 --gres=gpu:1 --pyt $SHELL
srun: unrecognized option '--pyt'
Try "srun --help" for more information
mfuai@slogin-01:~$ srun --partition=normal --account=mscbdt2024 --gres=gpu:1 --pty $SHELL
srun: Interactive job is limited to 480 minutes.
srun: For longer runs, please submit a batch job using `sbatch`.
(base) mfuai@dgx-42:~$ ./code tunnel
*
* Visual Studio Code Server
*
* By using the software, you agree to
* the Visual Studio Code Server License Terms (https://aka.ms/vscode-server-license) and
* the Microsoft Privacy Statement (https://privacy.microsoft.com/en-US/privacystatement).
*
✔ How would you like to log in to Visual Studio Code? · Microsoft Account
To sign in, use a web browser to open the page https://microsoft.com/devicelogin and enter the code NX6K6L3JH to authenticate.
✔ What would you like to call this machine? · mfuai
[2024-11-25 22:39:41] info Creating tunnel with the name: mfuai

Open this link in your browser https://vscode.dev/tunnel/mfuai

[2024-11-25 22:41:11] info [tunnels::connections::relay_tunnel_host] Opened new client on channel 2
[2024-11-25 22:41:11] info [russh::server] wrote id
```