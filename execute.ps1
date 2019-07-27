# powershell -ExecutionPolicy ByPass -File execute.ps1

#$d = "test1.py", "test2.py", "test3.py", "mnist.py"
$d = "lr_adam_lr_0.0001.py", "lr_nadam.py", "lr_sgd_lr_0.001.py", "lr_sgd_lr_0.01.py", "lr_sgd_lr_0.1.py", "baseline.py"

Foreach ($i in $d)
{
    conda activate tensor
    python $i
}

