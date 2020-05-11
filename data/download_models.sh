trainFileId=1kCOB_xrEbnr8JzAhWuskL2h0J81Hn8Ec
modelFileName=qaida_v1.bin
echo "Downloading model file"

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=${trainFileId}" > /dev/null
code="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"  
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${code}&id=${trainFileId}" -o ${modelFileName}