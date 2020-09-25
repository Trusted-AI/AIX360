import setuptools

version = '0.2.0'

with open("aix360/version.py", 'w') as f:
    f.write('# generated by setup.py\nversion = "{}"\n'.format(version))

setuptools.setup(
    name='aix360',
    version=version,
    description='IBM AI Explainability 360',
    authos='aix360 developers',
    url='https://github.com/IBM/AIX360',
    author_email='aix360@us.ibm.com',
    packages=setuptools.find_packages(),
    license='Apache License 2.0',
    long_description=open('README.md', 'r', encoding='utf-8').read(), 
    long_description_content_type='text/markdown',	
    install_requires=[
            'joblib>=0.11',
            'scikit-learn>=0.21.2',
            'torch',
            'torchvision',
            'cvxpy',
            'cvxopt',
            'Image',
            'tensorflow==1.15.4',	    
            'keras==2.3.1',
            'matplotlib',
            'numpy',
            'pandas',
            'scipy>=0.17',
            'xport',
            'scikit-image', 
            'requests',
            'xgboost==1.0.2', 	    
	    'bleach>=2.1.0',
	    'docutils>=0.13.1',
	    'Pygments',
            'qpsolvers',	    
            'lime==0.1.1.37',
            'shap==0.34.0'
	], 
    package_data={'aix360': ['data/*', 'data/*/*', 'data/*/*/*', 'models/*', 'models/*/*', 'models/*/*/*']},
    include_package_data=True,
    zip_safe=False
)
