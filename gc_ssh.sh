cd ~
mkdir ssd
cd ssd
git clone https://github.com/pierluigiferrari/ssd_keras/
cd ssd_keras
mkdir dataset_download
cd dataset_download
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar
cd ..
mkdir datasets
tar -xf dataset_download/VOCtest_06-Nov-2007.tar -C datasets
tar -xf dataset_download/VOCtrainval_06-Nov-2007.tar -C datasets
tar -xf dataset_download/VOCtrainval_11-May-2012.tar -C datasets
cd ~
mkdir weights
cd weights
git clone https://github.com/fenyesb/weights
cd ~
cp weights/weights/VGG_ILSVRC_16_layers_fc_reduced.h5 ssd/ssd_keras/VGG_ILSVRC_16_layers_fc_reduced.h5
sudo easy_install bs4
cd ssd/ssd_keras/dataset_download
youtube-dl https://drive.google.com/open?id=1tfBFavijh4UTG4cGqIKwhcklLXUDuY0D
unzip udacity_driving_datasets.zip-1tfBFavijh4UTG4cGqIKwhcklLXUDuY0D.zip -d ../dataset
cd ssd_keras
python3

pip3 install keras matplotlib tensorflow-gpu