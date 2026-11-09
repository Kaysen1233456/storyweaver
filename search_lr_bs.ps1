# lr x bs 搜索：4 组各 1500 步，等预算比较 val loss
$py = "C:\Users\Raymond James\AppData\Local\Programs\Python\Python312\python.exe"
Set-Location "C:\Users\Raymond James\PyTorch-DeepLearning-Practice-Projects-main\8_LLM_From_Scratch_run"
New-Item -ItemType Directory -Force -Path "..\models\search" | Out-Null

$combos = @(
    @{bs=16; lr="3e-4"; tag="bs16_lr3e-4"},
    @{bs=16; lr="6e-4"; tag="bs16_lr6e-4"},
    @{bs=32; lr="3e-4"; tag="bs32_lr3e-4"},
    @{bs=32; lr="6e-4"; tag="bs32_lr6e-4"}
)

foreach ($c in $combos) {
    Write-Output "`n===== SEARCH $($c.tag) ====="
    & $py train_run.py `
        --batch_size $c.bs `
        --context_length 256 `
        --learning_rate $c.lr `
        --max_iters 1500 `
        --eval_every 500 `
        --save_every 99999 `
        --save_dir "..\models\search" `
        --run_name $c.tag `
        --log_file "..\models\search\log_$($c.tag).txt" `
        --device cuda:0
}
Write-Output "`nSEARCH_ALL_DONE"
