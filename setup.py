from distutils.core import setup

setup(
    name='NNFL-EEG-DRCNN',
    version='1.11',
    packages=['colab', '', '', ''],

    install_requires=['numpy==1.13.1', 'scipy==0.19.1', 'scikit-learn==0.18.2', 'theano==0.8',
                      'lasagne @ git+https://github.com/Lasagne/Lasagne.git#egg=lasagne=0.2.dev1'],
    url='https://github.com/satyam9753/NNFL-EEG-DRCNN',

    license='GNU GENERAL PUBLIC LICENSE',
    author='Satyam Anand',
    description='Learning Representations from EEG with Deep Recurrent-Convolutional Neural Networks'
)