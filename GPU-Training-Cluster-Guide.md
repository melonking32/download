# GPU训练集群使用参考

## 写在前面
文档在陆续补充，第一次阅读请一定通篇看完

## 训练集训介绍

集群采用**K8s**进行资源管理，包括开发环境，任务调度等。目前数据代码可以统一放在NAS下进行管理，集群里所有的node都mount上了NAS环境，具体mount点如下
```python
/mnt/user/yourname    (个人数据或者代码)
/mnt/share/yourgroupname   (组内共享数据或者代码)


/mnt2/user/yourname    (research组用这个路径)
/mnt2/share/yourgroupname   (research组内共享数据或者代码)
```

其它有特殊情况需用用单独NAS的可以找Sawyer沟通
## 集群账号

新用户请找Team leader协助开通账号。由于目前还正在对接公司AAD账号体系认证，还没有ready，所以需要统一手动开通账号。可以通过邮件或者企业微信由Team leader Approve 然后找Sawyer (zengmin@xiaobing.ai)开通账号


## 创建开发环境

登陆后进⼊主⻚⾯，点击左侧菜单"交互式建模（DSW）"，进⼊DSW控制台界⾯。

![](images/DSW_1.png)

点击创建实例按钮，输⼊实例名称，⾃定义的CPU，GPU和内存规格，并选择镜像来源。镜像来源的输
⼊框中可以选择已经保存的镜像，同时也⽀持⽤户⼿动输⼊。填写完成后，点击确定按钮，进⾏创建，

![](images/DSW_2.png)

填写的表单信息描述如下：

|参数|描述|
|----|----|
|实例名称|创建的开发环境实例名称<br>只能包含英⽂字⺟、数字和下划线，且不能超过27个字符|
|CPU数量 |开发环境需要占⽤的CPU核⼼数量|
|GPU数量 |开发环境需要占⽤的GPU数量|
|MEM内存 |开发环境需要占⽤的内存⼤⼩，单位为GB|
|镜像来源 |开发环境使⽤的docker镜像，可选择官⽅镜像或⾃定义镜像<br>如果使⽤⾃定义镜像，需要提供docker镜像地址<br>例如master0:5000/eflops/nv-pytorch:21.05-py3|
创建实例成功后，实例状态会变更为运⾏中，此时可以进⼊实例进⾏开发。

**注意**：
> 开发环境，如果不需要GPU的同学，请不要占用GPU，确有需求的同学，我们统一采用P100的机器来做开发。


## 进⼊开发环境
在运⾏中的实例右侧有相应的开发环境进⼊按钮，如下图所示：


通过点击相应按钮可以进⼊开发环境，dsw提供三种⼯具，分别为jupyterLab，WebIDE（VsCode）以
及Terminal，可以通过⻚⾯上⽅的tab进⾏切换，如图所示：

对应的功能描述如下：

|菜单|描述|
|---|---|
|Terminal |进⼊实例所属环境的webShell，操作⽅式与Linux Terminal相同|
|Lab |进⼊JupyterLab|
|IDE |进⼊visual studio code|

除此之外，我们还可以通过ssh远程连接的方式，比如putty，或者vscode，直接连接开发环境。具体操作如下图，在DSW交互页面，点击远程链接，就会出现命令连接信息，账号密码同平台账号密码

![](images/DSW_3.png)

**注**：目前可能会遇到网络连接不稳定的问题，我们正在尝试通过搭专线到阿里云机房来解决这个问题
## 关闭开发环境
使用了GPU相关资源的开发环境，如果长时间不使用的话，建议点击停止按钮，释放掉对应的计算资源。具体操作如下图：
![](images/DLC_4.png)

如果我们再开发环境里，安装了一些软件包或者pip包，可以对这个开发环境进行保存，这样子就会存为一个新的镜像，方便后续的继续开发。如果不需要的话可以直接停止

**注意**，我们推荐大家把代码和数据统一放在NAS上，也就是 **/mnt/user/yourname** 目录下，这样子的话如果没有安装软件，哪怕代码或者数据改动了，都不需要把开发环境进行保存，节省时间和硬盘存储镜像资源

## 创建训练任务

### 通过界⾯创建
前往主界⾯，点击左侧的任务列表，随后点击新建任务，如下图所示。

![](images/DLC_1.png)

在创建任务的表格内填⼊以下信息，具体填写⽅式也可以参照使⽤⽂档中的任务管理章节填⼊对应的
值。其中节点镜像可以选择创建开发环境时⽤的镜像, 或者可以参考使用自定义的镜像（比如dockerhub，xiaoice acr等）

![](images/DLC_2.png)

执⾏命令示例： 

```python
python3 /mnt/user/zengmin/test_demo/mnist.py --backend nccl --epochs=10
```

点击提交，即可看到任务被创建成功。可以前往任务列表看到刚刚创建完成的任务。
我们可以任意执行/mnt/user/yourname下的代码

提交任务之后就能看见任务列表的状态，如下图

![](images/DLC_5.png)

任务的细节，如下图

![](images/DLC_6.png)

在任务的详细页，我们可以进行任务的克隆，或者停止当前任务。任务的训练日志可以通过右下角进行浏览。如果有需要，也可以进入训练任务pod进行详细的查看。

![](images/DLC_7.png)

### 通过命令行创建

有时候需要进行任务的自动化提交，我们可以通过DLC命令行工具进行操作。

首先需要下载安装DLC命令行工具，具体下载方式如下

#### linux/amd64环境
```bat
wget https://dlc-cli.oss-cn-zhangjiakou.aliyuncs.com/light/binary/linux/amd64/dlc
chmod +x ./dlc
```
#### darwin/amd64环境
```bat
wget https://dlc-cli.oss-cn-zhangjiakou.aliyuncs.com/light/binary/darwin/amd64/dlc
chmod +x ./dlc
```
#### darwin/arm64(apple m1芯⽚）环境
```bat
wget https://dlc-cli.oss-cn-zhangjiakou.aliyuncs.com/light/binary/darwin/arm64/dlc
chmod +x ./dlc
```
#### linux/arm64环境
```bat
wget https://dlc-cli.oss-cn-zhangjiakou.aliyuncs.com/light/binary/linux/arm64/dlc
chmod +x ./dlc
```

下载完成后进⾏⽤户认证

```bat
./dlc config \
       --username <yourname> \
       --region cn-beijing \
       --endpoint dlc-gateway.cfe74078211ab4b5094dfeafb16201df5.cn-beijing.alicontainer.com \
       --password <yourpassword>
```

查看当前账号属于的⼯作空间，并查询对应⼯作空间ID

```bat
./dlc get workspace
```

紧接着创建一个Job描述文件

```bat
cat << EOF > sample_job.params
name=yourtaskname
kind=PyTorchJob
worker_count=1
worker_cpu=4
worker_gpu=0
worker_memory=4
worker_shared_memory=4
worker_image=master0:5000/eflops/pytorch:v1
command=echo "hello world"
priority=1
workspace_id=your-workspace-id
interactive=false
EOF
```

创建任务

```bat
./dlc create job --params_file ./sample_job.params
```

把整个pipeline串起来，我们就可以完成任务的自动提交。具体实操遇到问题可以找 Sawyer 进一步来解决。

## 创建Tensorboard
完成训练后可以通过拉起Tensorboard的⽅式来查看模型细节。可以通过下图所示的⽅式指定读取⼀个
⽂件夹中的Log⽂件。

![](images/DLC_8.png)

具体的Summary文件为当前你需要查阅的训练任务的Log路径，如 

```python
/mnt/user/yourname/xxx/log/yyyy
```

创建完成后点击右侧查看Tensorboard即可前往Tensorboad的⻚⾯。

## 基础镜像

为了更高效的开发，我们准备了一些基础的镜像，包括pytorch/tensorflow以及一些基础的加速包，如apex, transformers等，常见的基础包有

```cmd

master0:5000/eflops/nv-pytorch:21.05-py3
master0:5000/eflops/nv-pytorch:21.09-py3-nlp-base
master0:5000/eflops/nv-pytorch:21.09-py3-cv-base
master0:5000/eflops/nv-pytorch:20.03-py3-nlp-base

```

nvidia pytorch env: 具体版本可参考 [NV-Pytorch-Release-Notes](https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/running.html)

nvidia tensorflow env: 具体版本可参考 [NV-Tensorflow-Release-Notes](https://docs.nvidia.com/deeplearning/frameworks/tensorflow-release-notes/index.html)



## 使用外部自定义镜像

在搭建训练环境时，有时需要用到外部自定义的镜像，比如docker hub等，但因为训练集群是国内的vm，不一定能稳定的拉取国外的images，从而导致训练环境不能创建或者训练任务失败等。所以，我们需要利用国内的CR做一个镜像的中转，具体操作如下

1, 设置本地CR连接

```cmd
docker login xiaoiceonaliyun.azurecr.cn -u  xiaoiceonaliyun
```

然后输入password, 密码请找Team leader或者 Sawyer (zengmin@xiaobing.ai) 获取，或者在企业微信群里咨询。

2, 拉取docker hub或者制作其它自定义镜像，并tag成xiaoiceonaliyun的镜像源, 之后push至CR，需要注意，这些外部镜像可能没有安装sshd服务，需要大家在push之前手动安装一下

```docker

FROM some-ubuntu-base-image

# openssh-server for sshd
# sudo for switch user
RUN apt update && apt install -y --no-install-recommends openssh-server sudo

# Allow sshd PasswordAuthentication
RUN sed -i 's/#PasswordAuthentication yes/PasswordAuthentication yes/g' /etc/ssh/sshd_config

# For dsw kernel registering
RUN pip install ipykernel

```

然后再push镜像

```cmd
docker pull abc.com/def/xyz:latest

docker tag abc.com/def/xyz:latest  xiaoiceonaliyun.azurecr.cn/XYZ:latest

docker push xiaoiceonaliyun.azurecr.cn/XYZ:latest
```

之后，就可以在集群平台上使用 xiaoiceonaliyun.azurecr.cn/XYZ:latest 作为镜像使用

## 常见问题

* 申请DLC/DSW资源失败？
> 由于集群采用K8s统一管理，申请dlc/dsw都会统一以pod的形式进行管理和维护，由于物理上一个pod只能部署在一个node上，所以大家申请资源时，可以提前看一下集群概览页集群的整体状态，包括CPU/GPU/Memory等一些参数，资源在选择的时候尽量不要超过一个node的剩余资源的上限，否则的话就会分配不成功
> 查看申请资源数是否超过分配的quota上限。 
> 另外，目前每个用户都是分配在一个组内的，如果组内的quota超过了上限的话，新提交的DSW/DLC任务也都会开始进行排队等待

* 如何往Azure上copy数据？
> 当我们完成任务的训练，需要将数据或者模型往Azure上进行下载部署时，我们可以通过**azcopy**工具将/mnt/user/yourname下的数据或者模型复制到Azure storage。具体操作如下<br>
> 1. 下载解压并安装azcopy，https://docs.microsoft.com/en-us/azure/storage/common/storage-use-azcopy-v10
> 2. 使用azcopy进行数据拷贝，例如：azcopy copy /mnt/user/xxxx  'azure_storage_url'


