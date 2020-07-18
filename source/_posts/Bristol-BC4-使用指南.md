---
title: Bristol Blue Crystal Phase 4 使用指南
date: 2020-07-18 18:33:53
categories: 学习
tags: Linux
---

<!--toc-->

由于毕设做深度学习项目，需要训练模型。布大的[超算](https://www.bristol.ac.uk/acrc/high-performance-computing/)提供了我所需要的算力支持。 <!--more-->

## 登陆（MacOS）

由于Mac自带ssh客户端服务，所以直接在终端通过ssh即可连接超算。如果用的不是校园网Eduroam，需要先[配置VPN](https://www.bris.ac.uk/it-services/advice/homeusers/uobonly/uobvpn/)才能连接上超算。username需要提前向学校申请。
```
ssh -X username@bc4login.acrc.bris.ac.uk  	 # CPU login node
ssh -X username@bc4gpulogin.acrc.bris.ac.uk  # GPU login node
```

我用iTerm2 + oh-my-zsh替代Mac原生的终端，通过在iTerm2中配置profiles文件可以一键登录，省去了每次输入密码的过程。

1. 首先创建一个脚本文件（无后缀），我是把文件放在～/.ssh/bc4-ac, 内容如下：

```
#/user/bin/expect -f

set user <username>
set hose <hosename/IP>
set password <password>
set port 22  # default 22
set timeout -1  

spawn ssh $user@$host
expect "*assword:*"
send "$password\r"
interact
expect eof
```

2. iTerm2 >> preference>>profiles

添加一个新的profile，设置名称，把`command`模式改为`Command`，然后指定脚本文件。

<img src="https://suyueliu-blog-img.oss-cn-beijing.aliyuncs.com/images/set_ssh.png" style="zoom:40%;" />

每次打开iTerm2之后，在profiles中选择相应的选项，便可快速登陆，同时还可以设置快捷键登陆。

## 可视化

习惯了可视化，突然要在终端里面操控一切还是有点不适应的，好在BC4支持可视化操作。

- 通过第一步登陆BC4结点；
- 启动dcv服务，https://bc4vis.acrc.bris.ac.uk/
- 通过布大的账号登陆；
- 点击左侧的"Linux Desktop"会自动下载一个".vnc"文件；
- 下载一个能够打开vnc文件的应用，打开文件即可；我用的是VNC Viwer（官方建议的NICE Endstation 不知道为什么连接不上）。

## 文件传输

电脑连接上BC4之后，就相当于就相当于你的电脑通过ssh连接上远程Linux服务器。在个人电脑和服务器之间传输文件和咱们平时在两台电脑之间传输文件有所不同，这里有多种传输方法：ftp, rcp, scp, wget, curl, rsync等。我测试了一下BC4支持scp, wget, curl, rsync，这里我用的是scp方式，传输速度个人觉得还行。

> <font size = 3>scp 命令在网络上的主机之间拷贝文件，它是安全拷贝（secure copy）的缩写。 scp 命令使用 ssh 来传输数据，并使用与 ssh 相同的认证模式，提供同样的安全保障。 scp 命令的用法和 rcp 命令非常类似，这里就不做过多介绍了。一般推荐使用 scp 命令，因为它比 rcp 更安全。</font>

```
# general
scp ~/local_path/local_filename username@hostname:/remote_path/  # 传输文件
scp -r ~/local_dir username@hostname:/remote_path/remote_dir  # 传输文件夹

# 从本地上传文件到服务器
scp ./test.txt username@bc4login.acrc.bris.ac.uk:/mnt/storage/home/username/examples
# 从本地上传文件夹到服务器
scp -r ./test username@bc4login.acrc.bris.ac.uk:/mnt/storage/home/username/examples

```

输入以上指令之后，只需要输入登录的密码（ssh登录的密码）验证即可开始传输。如果想从服务器上下载文件，只需要把remote和local调换顺序即可。

## 工作流程

BC4使用[Slurm](https://slurm.schedmd.com/)来管理集群和调度作业。

> Slurm is a job management application which allows you to take full advantage of BlueCrystal 4. You interact with it by writing a small shell script with some special commands which tell it how to run your code. You will pass this shell script to a command called `sbatch` which will ask Slurm to schedule your job. 

下面是几个Slurm框架下几个常用的术语：([from BC4 user documentation](https://www.acrc.bris.ac.uk/protected/bc4-docs/scheduler/index.html))

* **Node**: A physical computer sitting in a rack in the computer room.
* **Task**: Each **task** in Slurm refers to a process running on a node. 
* **CPU**: Most of the time, Slurm refers to a CPU core as a **CPU**. The computer nodes on BC4 have two physical CPU chips, with 14 processing cores inside each.
* **Partition**: A set of nodes with associated restrictions on use. This is called a **queue** in some other systems. For example we have a `test` partition which only allows small, short jobs and we have a `cpu` partition which allows large, long jobs. Run `sinfo -s` to see a list of all the partitions you have access to.
* **Step**: A job **step** is a part of a job. Each job will run its steps in order.

下面Slurm中常用的指令：

| 命令       | 描述                                                         |
| ---------- | ------------------------------------------------------------ |
| `sinfo`    | 报告由 Slurm 管理的分区和节点的状态。它具有多种筛选、排序和格式设置选项。 |
| sbatch     | 用于提交作业脚本以后以后执行，`sbatch file_name.sh`          |
| `sacct`    | 用于查看正在活动或者已经结束的作业，`sacct -j <job-ID>` 可查看置顶的job。 |
| `scancel`  | 用于取消挂起或正在运行的作业或作业步骤。                     |
| `scontrol` | 用于查看或修改Slurm状态的管理工具，`control show job <job-ID>` 可查看作业细节。 |
| `squeeze`  | 查看作业处理队列，`squeue --state=PENDING`                   |
| `srun`     | 用于提交执行任务或实时启动作业步骤。                         |

### 作业运行模式

在开始提交作业运行自己的程序之前，首先要准备好程序运行所需的环境。在BC4中已经预先设定了许多环境。

* `module avail`：查询可用使用的模块，位于`/modules/local`的模块是官方推荐使用的；
* `module whatis <module_name>`：查询指定的模块；
* `module load <module_name>`：加载指定的模块。加载某个置顶的模块时也会把相依的依赖也一同加载进来；
* `module unload <module_name>`：卸载指定模块；
* `module list`：查看所有已经加载的模块；
* `module merge`：一次卸载所有已经加载的模块。

官方建议将加载模块的指令写在提交作业的**脚本文件**中。由于我要使用Python 和 GPU，我在我的脚本文件中加载了这两个模块：

```
module load CUDA
module load languages/anaconda3/2020.02-tflow-2.2.0
```

#### 交互模式

在终端中通过`srun`直接提交资源分配请求，指定资源适量和限制。

```
$ srun -n 4 ./example  # 申请4个进程生成一个作业step
```

#### 批处理模式

在BC4上提交批处理任务需要编写脚本，以明确申请的资源以及所要运行的程序。在终端通过`vim test.sh`指令创建名为`test.sh`的脚本。

```
#!/bin/bash  				 		 	# /bin/bash is used to execute the script

#SBATCH --job-name=test_job          	# Set name of job
#SBATCH --partition=test  	         	# Set partition
#SBATCH --nodes=2			 		 	# Set the number of nodes
#SBATCH --ntasks-per-node=2  		 	# Set the number of tasks(process) per node
#SBATCH --cpus-per-task=1    		 	# Set the number of cpus per task
#SBATCH --time=0:1:0         		 	# Set the max wallclock time
#SBATCH --mem=100M           		 	# Set the memory
#SBATCH --output=../results/test.out 	# Set the directory for output
#SBATCH --gres=gpus:2					# Set the number of gpus, if use gpu, the partition 											  should be set as "gpu"

# Load modules
module load CUDA  # If use gpu
module load languages/anaconda3/2020.02-tflow-2.2.0

# run the application
srun python ./test.py
```

写完脚本之后，在终端通过指令`sbatch tets.sh`提交作业，等待运行，这时候会得到一个job-ID。可以通过`sacct -j <job-ID>`查询作业运行情况。作业运行的结果是存放在`test.out`（默认情况是在`slurm-<job-ID>`）中的，其内容不会直接打印到屏幕，需要通过Linux指令`cat`（`cat test.out`）将其输出。