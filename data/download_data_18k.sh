trainFileId=1ihemYqrIDklByJIxk1tKyxg3cISYQIYQ
trainFileName=train.tar.xz

testFileId=1EvM5SqDruOn1RBHf7vFk2ITS3sze90og
testFileName=test.tar.xz

echo "Downloading train data"

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=${trainFileId}" > /dev/null
code="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"  
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${code}&id=${trainFileId}" -o ${trainFileName} 

echo "Extracting train data"
tar -xf train.tar.xz
rm train.tar.xz

echo "Downloading test data"

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=${testFileId}" > /dev/null
code="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"  
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${code}&id=${testFileId}" -o ${testFileName} 

echo "Extracting test data"
tar -xf test.tar.xz
rm test.tar.xz
