from setuptools import setup, find_packages

long_description = """
"""

setup(
    name='MultimodalGraphRetrieval',
    version='0.2.2',
    description='Multimodal Graph retrieval',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='JVNK',
    author_email='jaya11vibhav@gmail.com',
    url='https://github.com/jayavibhavnk/MultimodalGraphRetrieval',
    packages=find_packages(),
    install_requires=[
      'langchain',
      'sentence_transformers',
      'langchain-community',
      'pypdf2==3.0.0',
      'pypdf',
      'pymupdf',
    ],
    classifiers=[
        'Development Status :: 3 - Alpha'
    ],
)
