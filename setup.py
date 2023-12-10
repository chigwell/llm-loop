from setuptools import setup, find_packages

setup(
    name='llm-loop',
    version='0.02',
    packages=find_packages(),
    description='A utility package for querying language models with pattern matching and retry logic',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    install_requires=[
        'ctransformers',
    ],
    url='https://github.com/chigwell/llm-loop',
    author='Evgenii Evstafev',
    author_email='chigwel@gmail.com',
)
