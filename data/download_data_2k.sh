trainFileId=1oQk6Hs13JL5OkW2EpS0-zSUAVX7SORzp
trainFileName=train_2k.tar.xz

testFileId=196rEKpsLlNOWCoTQv3TVjTnq8nP0FPXr
testFileName=test_2k.tar.xz

echo "Downloading train data"

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=${trainFileId}" > /dev/null
code="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"  
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${code}&id=${trainFileId}" -o ${trainFileName} 

echo "Extracting train data"
tar -xf train_2k.tar.xz
rm train_2k.tar.xz

echo "Downloading test data"

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=${testFileId}" > /dev/null
code="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"  
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${code}&id=${testFileId}" -o ${testFileName} 

echo "Extracting test data"
tar -xf test_2k.tar.xz
rm test_2k.tar.xz
