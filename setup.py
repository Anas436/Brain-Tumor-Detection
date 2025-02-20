from setuptools import setup,find_packages

hyper_e_dot = 'e .'
def get_package(path):
    
    with open(path,'r') as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace('\n','') for req in requirements]
        
        if  hyper_e_dot in  requirements:
            requirements.remove(hyper_e_dot)

setup(
    fullname= "A end to  end deep leaning project",
    author="Aziz Ashfak",
    author_email="azizashfak@gamil.com",
    version="0.0.1",
    description="This is my second deep learning project!!!",
    packages=find_packages(),
    install_requires = get_package('requirements.txt')
)