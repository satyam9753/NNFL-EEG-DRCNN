from distutils.core import setup

setup(
    name='NNFL-EEG-DRCNN',
    version='1.11',
    packages=['colab', 'models', 'utilities', 'generate_images.py', 'train.py'],

    install_requires=['ipython==7.11.1', 'numpy==1.18.1', 'scipy==1.4.1', 'scikit-learn==0.21.3', 'theano==1.0.4',
                      'lasagne @ git+https://github.com/Lasagne/Lasagne.git#egg=lasagne=0.2.dev1'],
    
    url='https://github.com/satyam9753/NNFL-EEG-DRCNN',

    license='GNU GENERAL PUBLIC LICENSE',
    author='Satyam Anand',
    description='Learning Representations from EEG with Deep Recurrent-Convolutional Neural Networks'
)