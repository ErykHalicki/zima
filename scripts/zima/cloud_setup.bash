if [ ! -d "zima" ]; then
    echo "Cloning zima repository..."
    git clone https://github.com/ErykHalicki/zima.git
    cd zima/src
    pip install -r requirements.txt
    hf auth login --token $HF_TOKEN
else
    echo "Zima repository already exists, skipping clone"
    cd zima/src
    git pull
fi

source scripts/train_act.bash

