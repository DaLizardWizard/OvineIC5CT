#!/usr/bin/env bash
apt-get update
pip install cupy-cuda101
apt-get -y install git
apt-get -y install gcc
rm -rf /var/lib/apt/lists/*
pip install chainer chainerui chainercv pydicom matplotlib numba cupy-cuda101>=7.7.0,<8.0.0
pip install 'cupy-cuda101>=7.7.0,<8.0.0'
git clone https://github.com/shizuo-kaji/PairedImageTranslation

####paste inputfile1.txt inputfile2.txt > outputfile.txt #will make a two column list
for f in *.DIC; do 
    mv -- "$f" "${f%.DIC}_0.dcm"
done

for f in *.cic; do 
    mv -- "$f" "${f%.cic}_1.dcm"
done
ls *_0.dcm > A_pre.txt
ls *_1.dcm > A_post.txt
paste A_pre.txt A_post.txt > A_paired.txt

###creates training and validation
awk 'NR < 99 { print >> "ct_reconst_train.txt"; next } {print >> "ct_reconst_val.txt" }' A_paired.txt
###

#training the GAN
python train_cgan.py -R A_ISC -t A_ISC/ct_reconst_train.txt \
--val A_ISC/ct_reconst_val.txt -o result -it dcm \
-g 0 -rt 0 -e 100 -l1 0 -l2 10.0 -ldis 1.0 -ltv 1e-3

#Applying the trained model to unseen data
python convert.py -b 10 -a result/0331_1056_cgan_rt\=0/args \
-R UnseenA_ISC --val UnseenA_ISC/A_pre.txt \
-o converted256 -it dcm -m result/0331_1056_cgan_rt\=0/enc_x100.npz
