cd /home/cc

echo "********** CREATING CONDA ENVIRONMENT **********"
conda create -n medtweet python=3.6 -y
source activate medtweet

pip install scipy matplotlib pillow imutils h5py requests progressbar2 scikit-learn scikit-image wget numpy keras-vis cython tqdm
conda install -c anaconda mpi4py graphviz pydot -y

echo "********** INSTALLING TENSORFLOW GPU **********"
pip install tensorflow

echo "********** INSTALLING KERAS **********"
pip install keras

echo "********** INSTALLING RMATE **********"
wget -O /usr/local/bin/rmate \https://raw.github.com/aurora/rmate/master/rmate
chmod a+x /usr/local/bin/rmate

echo "********** TESTING TENSORFLOW **********"
python -c "import tensorflow"

echo "********** TESTING KERAS **********"
python -c "import keras"

echo ""
echo "DONE!"
