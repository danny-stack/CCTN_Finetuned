# run.sh (放在 CCTN_Finetune 目录下)
#!/bin/bash
case "$1" in
    "train")
        python tools/train_cctn.py
        ;;
    "eval")
        python tools/evaluation.py
        ;;
    "api")
        python tools/api.py
        ;;
    *)
        echo "请指定模式: train, eval 或 api"
        ;;
esac